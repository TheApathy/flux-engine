// SPDX-License-Identifier: AGPL-3.0-only

//! Single-token decode body for [`super::super::Qwen3AttentionLayer`],
//! split out of the trait impl for file-size budget. The trait impl
//! delegates 1:1 to [`Qwen3AttentionLayer::decode_inner`].

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use super::{diag_norm, diag_norm_f32, gemma4_diag_enabled};
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen4K5DeviceAttentionPlan;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_qwen4_batched_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        _state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        _h_intermediate: DevicePtr,
        _conv_intermediate: DevicePtr,
        _h_intermediate_stride: usize,
        _conv_intermediate_stride: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (attn_hyper, mlp_hyper) = match (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper) {
            (Some(attn), Some(mlp)) => (attn, mlp),
            _ => anyhow::bail!("Qwen4 batched verify requested without hyperconnections"),
        };
        let metadata = ctx
            .attn_metadata
            .ok_or_else(|| anyhow::anyhow!("Qwen4 batched attention requires metadata"))?;
        // Every exact verify width (2..=9 rows) journals its QSA writes; the
        // commit restores the rejected suffix (crate::layers::qwen4_qsa::
        // verify_journal_rows mirrors this on the commit side).
        let qsa_journal = metadata.qwen4_qsa_required
            && crate::layers::qwen4_qsa::verify_journal_rows(num_tokens);
        if qsa_journal {
            self.qwen4_qsa
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Qwen4 K5 QSA transaction lacks an indexer"))?
                .prepare_verify_k5_transaction(kv_cache, ctx.gpu)?;
        }
        let device_attention_plan =
            Qwen4K5DeviceAttentionPlan::from_host(ctx.config, num_tokens, seq_len, metadata)?;
        if let Some(plan) = device_attention_plan {
            self.prepare_qwen4_k5_device_attention(plan, kv_cache, metadata, ctx)?;
        }
        let h = ctx.config.hidden_size;
        let row_bytes = ctx.config.residual_width() * 2;
        let core_bytes = h * 2;
        let eps = ctx.config.rms_norm_eps as f32;
        let mb = metadata.max_blocks_per_seq as usize;
        let hybrid_k5 = crate::layers::qwen4_hybrid_rows(num_tokens);
        let exact_hyper_k5 = hybrid_k5 && qwen4_k5_batch_hyper_disabled();

        // Batch the four-stream projection weights. KV writes and attention
        // remain token ordered to preserve exact causal semantics.
        // `attention_forward_oproj` reuses norm_output, so preserve every
        // mixed row before the first token's attention core clobbers it.
        let attn_inputs = ctx.buffers.gate_logits();
        anyhow::ensure!(
            num_tokens * core_bytes <= ctx.buffers.sizes().gate_logits,
            "Qwen4 batched attention input staging exceeds gate-logit workspace"
        );
        let pending_prev_inject = crate::layers::qwen4_hyper::take_pending_verify_hc_inject();
        if exact_hyper_k5 {
            if let Some(pending) = pending_prev_inject {
                attn_hyper.inject_saved_batched(
                    pending.hyper,
                    pending.core,
                    pending.residual,
                    pending.rows,
                    ctx.gpu,
                    pending.stream,
                )?;
            }
            for row in 0..num_tokens {
                let (mixed, _) = attn_hyper.prepare_decode(
                    hidden.offset(row * row_bytes),
                    residual.offset(row * row_bytes),
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    mixed,
                    attn_inputs.offset(row * core_bytes),
                    core_bytes,
                    stream,
                )?;
            }
        } else {
            match pending_prev_inject {
                Some(pending) if pending.matches(hidden, residual, num_tokens, stream) => {
                    attn_hyper.prepare_batched_into_after_inject(
                        hidden,
                        pending.core,
                        residual,
                        num_tokens,
                        ctx.buffers,
                        Some(attn_inputs),
                        ctx.gpu,
                        eps,
                        stream,
                    )?;
                }
                other => {
                    if let Some(pending) = other {
                        attn_hyper.inject_saved_batched(
                            pending.hyper,
                            pending.core,
                            pending.residual,
                            pending.rows,
                            ctx.gpu,
                            pending.stream,
                        )?;
                    }
                    attn_hyper.prepare_batched_into(
                        hidden,
                        residual,
                        num_tokens,
                        ctx.buffers,
                        Some(attn_inputs),
                        ctx.gpu,
                        eps,
                        stream,
                    )?;
                }
            }
        }
        spark_runtime::cublaslt::stage_timer::mark("v_hc_in", stream);
        // The projection helper retains the legacy exact K5/K9 path and may
        // independently admit exact K16 for gamma-15 verification. Inside a
        // scoped batched-verify TC dense frame (`rows >= 4`, i.e.
        // `DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS`), route QKV via
        // zero-VRAM coalesced NVFP4 (`qwen4_k5_project_qkv_tc_dense`) before
        // falling back to exact scalar GEMV.
        let exact_qkv = if device_attention_plan.is_none()
            && crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active()
        {
            match self.qwen4_k5_project_qkv_tc_dense(attn_inputs, num_tokens, ctx, stream)? {
                Some(qkv) => Some(qkv),
                None => self.qwen4_k5_project_qkv_exact(attn_inputs, num_tokens, ctx, stream)?,
            }
        } else {
            self.qwen4_k5_project_qkv_exact(attn_inputs, num_tokens, ctx, stream)?
        };
        spark_runtime::cublaslt::stage_timer::mark("v_qkv", stream);
        // ATLAS_QWEN4_VERIFY_OPROJ_ROWS=1: stage each row's gated attention
        // output and project all rows in one exact multi-row launch (the
        // exact tiers and the K1 w4a16_gemv/_sw share lane ownership, operand
        // order and tree per row), then inject the rows from their saved
        // scales. Attention reads only the staged inputs and the KV cache, so
        // deferring the per-row inject does not change any later row.
        let attn_width = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads)
            * self.head_dim_override.unwrap_or(ctx.config.head_dim);
        let attn_row_bytes = attn_width * 2;
        let oproj_rows = device_attention_plan.is_none()
            && exact_qkv.is_some()
            && self.verify_oproj_rows_admits(num_tokens, attn_row_bytes, core_bytes, ctx);
        let raw_rows = ctx.buffers.ssm_qkvz();
        let qsa_rows_precomputed = if let Some(qsa) = self.qwen4_qsa.as_ref() {
            qsa.update_and_select_verify_rows(
                attn_inputs,
                num_tokens,
                seq_len,
                qsa_journal,
                ctx.graph_capture,
                device_attention_plan,
                kv_cache,
                metadata,
                h as u32,
                eps,
                ctx.config.rope_theta as f32,
                ctx.config.rotary_dim() as u32,
                ctx.gpu,
                stream,
            )?
        } else {
            false
        };
        let route_telemetry =
            qsa_journal && crate::layers::qwen4_qsa::verify_route_telemetry_enabled();
        let attention_result =
            (|| -> Result<Option<crate::layers::qwen4_qsa::VerifyAttentionRoute>> {
                if qsa_rows_precomputed
                    && let Some(route) = self.try_qsa_sparse_attention_rows(
                        hidden,
                        residual,
                        raw_rows,
                        num_tokens,
                        seq_len,
                        exact_qkv,
                        device_attention_plan,
                        oproj_rows,
                        attn_row_bytes,
                        row_bytes,
                        kv_cache,
                        metadata,
                        ctx,
                        stream,
                    )?
                {
                    return Ok(route_telemetry.then_some(route));
                }
                if (crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active()
                    || qwen4_c1_dense_attn_enabled())
                    && let Some(route) = self.try_qwen4_batched_dense_attention_rows(
                        raw_rows,
                        num_tokens,
                        seq_len,
                        exact_qkv,
                        device_attention_plan,
                        oproj_rows,
                        attn_row_bytes,
                        qsa_rows_precomputed,
                        kv_cache,
                        metadata,
                        ctx,
                        stream,
                    )?
                {
                    return Ok(route_telemetry.then_some(route));
                }

                let mut row_routes = crate::layers::qwen4_qsa::VerifyRowRouteCounts::default();
                for row in 0..num_tokens {
                    let hidden_row = hidden.offset(row * row_bytes);
                    let residual_row = residual.offset(row * row_bytes);
                    let token_metadata = crate::layer::AttnMetadataDev {
                        positions: metadata.positions.offset(row * 4),
                        positions_h: metadata.positions_h.offset(row * 4),
                        positions_w: metadata.positions_w.offset(row * 4),
                        slot: metadata.slot.offset(row * 8),
                        seq_len: metadata.seq_len.offset(row * 4),
                        block_table: metadata.block_table.offset(row * mb * 4),
                        num_seqs: 1,
                        ..metadata
                    };
                    let token_ctx = ForwardContext {
                        attn_metadata: Some(token_metadata),
                        ..*ctx
                    };
                    // QSA owns only four raw rows per physical page, while an exact
                    // speculative frame owns five candidate rows. Save the physical
                    // raw/compressed destinations before attention mutates them; the
                    // model commit restores the rejected suffix in reverse order.
                    // This launch is deliberately inside the per-row captured body,
                    // including row zero's journal reset, so graph replay cannot reuse
                    // stale transaction state.
                    if qsa_journal && !qsa_rows_precomputed {
                        self.qwen4_qsa
                            .as_ref()
                            .ok_or_else(|| {
                                anyhow::anyhow!("Qwen4 K5 QSA transaction lacks an indexer")
                            })?
                            .snapshot_verify_k5_row(
                                kv_cache,
                                token_metadata.slot,
                                row,
                                ctx.gpu,
                                stream,
                            )?;
                    }
                    let attn_input = attn_inputs.offset(row * core_bytes);
                    if oproj_rows && let Some((qkv, qkv_row_bytes)) = exact_qkv {
                        let raw_row_dst = raw_rows.offset(row * attn_row_bytes);
                        let (_, row_route) = self.attention_forward_preprojected_deferred_into(
                            attn_input,
                            qkv.offset(row * qkv_row_bytes),
                            seq_len + row,
                            Some(raw_row_dst),
                            qsa_journal,
                            block_table,
                            disk_block_ids,
                            disk_last_offloaded_per_layer,
                            kv_cache,
                            &token_ctx,
                            stream,
                        )?;
                        if route_telemetry {
                            row_routes.record(row_route);
                        }
                        continue;
                    }
                    let (attn_out, row_route) = if let Some(plan) = device_attention_plan {
                        let (qkv, qkv_row_bytes) = exact_qkv.ok_or_else(|| {
                            anyhow::anyhow!(
                                "ATLAS_QWEN4_K5_DEVICE_ATTN_GRAPH lost exact K5 QKV admission"
                            )
                        })?;
                        self.attention_forward_preprojected_device(
                            attn_input,
                            qkv.offset(row * qkv_row_bytes),
                            plan.row(row)?,
                            block_table,
                            disk_block_ids,
                            disk_last_offloaded_per_layer,
                            kv_cache,
                            &token_ctx,
                            stream,
                        )?
                    } else {
                        let preprojected_qkv =
                            exact_qkv.map(|(qkv, qkv_row_bytes)| qkv.offset(row * qkv_row_bytes));
                        self.attention_forward_verify_host_row_with_route(
                            attn_input,
                            preprojected_qkv,
                            seq_len + row,
                            qsa_journal,
                            block_table,
                            disk_block_ids,
                            disk_last_offloaded_per_layer,
                            kv_cache,
                            &token_ctx,
                            stream,
                        )?
                    };
                    if route_telemetry {
                        row_routes.record(row_route);
                    }
                    attn_hyper.inject_decode(
                        hidden_row,
                        attn_out,
                        attn_hyper.saved_inject(residual_row),
                        ctx.gpu,
                        stream,
                    )?;
                }
                Ok(route_telemetry.then_some(
                    crate::layers::qwen4_qsa::VerifyAttentionRoute::RowSerial(row_routes),
                ))
            })();
        // Clear the host row cursor on both success and every recoverable
        // failure from the sparse or row-by-row attention paths.
        if qsa_rows_precomputed && let Some(qsa) = self.qwen4_qsa.as_ref() {
            qsa.clear_precomputed_verify_rows();
        }
        if let Some(attention_route) = attention_result? {
            crate::layers::qwen4_qsa::record_verify_route(
                attention_route,
                num_tokens,
                seq_len,
                self.attn_layer_idx,
            );
        }
        spark_runtime::cublaslt::stage_timer::mark("v_attn", stream);
        let mut deferred_o_out = None;
        if oproj_rows {
            VERIFY_OPROJ_ROWS_LAUNCHED.call_once(|| {
                tracing::info!(
                    "ENGAGED ATLAS_QWEN4_VERIFY_OPROJ_ROWS: multi-row O projection launched (rows={num_tokens})"
                );
            });
            let o_out = ctx.buffers.norm_output();
            let tc_dense_fp8 = crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active()
                && crate::layers::qwen4_hyper::batched_verify_tc_component_active(
                    crate::layers::qwen4_hyper::BatchedVerifyTcComponent::Fp8Dense,
                    num_tokens,
                    ctx.graph_capture,
                );
            if !(tc_dense_fp8
                && self.try_qwen4_oproj_direct_nvfp4(
                    raw_rows, o_out, num_tokens, h, attn_width, ctx, stream,
                )?)
            {
                ops::w4a16_gemv_batch_logits_exact_with(
                    ctx.gpu,
                    self.w4a16_exact_o_proj_kernels,
                    raw_rows,
                    &self.attn.o_proj,
                    o_out,
                    num_tokens as u32,
                    ctx.config.hidden_size as u32,
                    attn_width as u32,
                    stream,
                    ops::w4a16_exact_rt2_for_rows(
                        ops::w4a16_gemv_rt2_enabled(),
                        ops::w4a16_exact_rt2_min_rows(),
                        num_tokens as u32,
                    ),
                )?;
            }
            if ctx.config.tp_world_size > 1
                && let Some(comm) = ctx.comm
            {
                comm.all_reduce_async(o_out.0, num_tokens * core_bytes, stream)?;
            }
            if exact_hyper_k5 {
                attn_hyper
                    .inject_saved_batched(hidden, o_out, residual, num_tokens, ctx.gpu, stream)?;
            } else {
                deferred_o_out = Some(o_out);
            }
        }

        spark_runtime::cublaslt::stage_timer::mark("v_mixer", stream);
        let ffn_inputs = if exact_hyper_k5 {
            let staging = ctx
                .buffers
                .qkv_output()
                .offset(ctx.config.residual_width() * 2);
            anyhow::ensure!(
                ctx.config.residual_width() * 2 + num_tokens * core_bytes
                    <= ctx.buffers.sizes().qkv_output,
                "Qwen4 K=5 exact MLP staging exceeds QKV arena"
            );
            for row in 0..num_tokens {
                let (mixed, _) = mlp_hyper.prepare_decode(
                    hidden.offset(row * row_bytes),
                    residual.offset(row * row_bytes),
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    mixed,
                    staging.offset(row * core_bytes),
                    core_bytes,
                    stream,
                )?;
            }
            staging
        } else if let Some(o_out) = deferred_o_out {
            mlp_hyper.prepare_batched_after_inject(
                hidden,
                o_out,
                residual,
                num_tokens,
                ctx.buffers,
                ctx.gpu,
                eps,
                stream,
            )?
        } else {
            mlp_hyper.prepare_batched(
                hidden,
                residual,
                num_tokens,
                ctx.buffers,
                ctx.gpu,
                eps,
                stream,
            )?
        };
        spark_runtime::cublaslt::stage_timer::mark("v_hc_in2", stream);
        match num_tokens {
            // Every admitted hybrid width (2..=9 under ATLAS_QWEN4_HYBRID_ALL_K).
            n if hybrid_k5 => self
                .ffn
                .forward_qwen4_exact_rows(ffn_inputs, n, ctx, stream)?,
            2 => self.ffn.forward_k2(ffn_inputs, ctx, stream)?,
            3 => self.ffn.forward_k3(ffn_inputs, ctx, stream)?,
            n => self.ffn.forward_prefill(ffn_inputs, n, ctx, stream)?,
        }
        spark_runtime::cublaslt::stage_timer::mark("v_moe", stream);
        let ffn_output = ctx.buffers.moe_output();
        if hybrid_k5
            && !exact_hyper_k5
            && crate::layers::qwen4_hyper::verify_hc_defer_mlp_tail_allowed()
        {
            crate::layers::qwen4_hyper::defer_pending_verify_hc_inject(
                crate::layers::qwen4_hyper::PendingVerifyHcInject {
                    hyper: hidden,
                    core: ffn_output,
                    residual,
                    rows: num_tokens,
                    stream,
                },
            )?;
        } else if hybrid_k5 {
            mlp_hyper
                .inject_saved_batched(hidden, ffn_output, residual, num_tokens, ctx.gpu, stream)?;
        } else {
            for row in 0..num_tokens {
                let hidden_row = hidden.offset(row * row_bytes);
                let residual_row = residual.offset(row * row_bytes);
                mlp_hyper.inject_decode(
                    hidden_row,
                    ffn_output.offset(row * core_bytes),
                    mlp_hyper.saved_inject(residual_row),
                    ctx.gpu,
                    stream,
                )?;
            }
        }
        spark_runtime::cublaslt::stage_timer::mark("v_inject", stream);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_qwen4_batched_mtp_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        emit_last_row: bool,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (attn_hyper, mlp_hyper) = match (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper) {
            (Some(attn), Some(mlp)) => (attn, mlp),
            _ => anyhow::bail!("Qwen4 batched MTP requested without hyperconnections"),
        };
        let metadata = ctx
            .attn_metadata
            .ok_or_else(|| anyhow::anyhow!("Qwen4 batched MTP attention requires metadata"))?;
        if metadata.qwen4_qsa_required
            || self.mla.is_some()
            || self.high_speed_swap_engaged(kv_cache)
            || num_tokens == 0
        {
            return self.decode_qwen4_batched_inner(
                hidden,
                residual,
                num_tokens,
                state,
                kv_cache,
                seq_len,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                DevicePtr::NULL,
                DevicePtr::NULL,
                0,
                0,
                ctx,
                stream,
            );
        }
        let h = ctx.config.hidden_size;
        let row_bytes = ctx.config.residual_width() * 2;
        let core_bytes = h * 2;
        let eps = ctx.config.rms_norm_eps as f32;
        let mb = metadata.max_blocks_per_seq as usize;
        let hybrid_k5 = crate::layers::qwen4_hybrid_rows(num_tokens);
        let exact_hyper_k5 = hybrid_k5 && qwen4_k5_batch_hyper_disabled();
        let attn_inputs = ctx.buffers.gate_logits();
        anyhow::ensure!(
            num_tokens * core_bytes <= ctx.buffers.sizes().gate_logits,
            "Qwen4 batched MTP attention input staging exceeds gate-logit workspace"
        );
        if exact_hyper_k5 {
            for row in 0..num_tokens {
                let (mixed, _) = attn_hyper.prepare_decode(
                    hidden.offset(row * row_bytes),
                    residual.offset(row * row_bytes),
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    mixed,
                    attn_inputs.offset(row * core_bytes),
                    core_bytes,
                    stream,
                )?;
            }
        } else {
            attn_hyper.prepare_batched_into(
                hidden,
                residual,
                num_tokens,
                ctx.buffers,
                Some(attn_inputs),
                ctx.gpu,
                eps,
                stream,
            )?;
        }
        let exact_qkv = self.qwen4_k5_project_mtp_qkv_exact(
            attn_inputs,
            num_tokens,
            emit_last_row,
            ctx,
            stream,
        )?;
        let kv_only_rows = if emit_last_row {
            num_tokens - 1
        } else {
            num_tokens
        };
        for row in 0..kv_only_rows {
            let token_metadata = crate::layer::AttnMetadataDev {
                positions: metadata.positions.offset(row * 4),
                positions_h: metadata.positions_h.offset(row * 4),
                positions_w: metadata.positions_w.offset(row * 4),
                slot: metadata.slot.offset(row * 8),
                seq_len: metadata.seq_len.offset(row * 4),
                block_table: metadata.block_table.offset(row * mb * 4),
                num_seqs: 1,
                ..metadata
            };
            let token_ctx = ForwardContext {
                attn_metadata: Some(token_metadata),
                ..*ctx
            };
            self.attention_forward_kv_only(
                attn_inputs.offset(row * core_bytes),
                exact_qkv.map(|(qkv, qkv_row_bytes)| qkv.offset(row * qkv_row_bytes)),
                seq_len + row,
                block_table,
                kv_cache,
                &token_ctx,
                stream,
            )?;
        }
        if emit_last_row {
            let last = num_tokens - 1;
            let hidden_last = hidden.offset(last * row_bytes);
            let residual_last = residual.offset(last * row_bytes);
            let token_metadata = crate::layer::AttnMetadataDev {
                positions: metadata.positions.offset(last * 4),
                positions_h: metadata.positions_h.offset(last * 4),
                positions_w: metadata.positions_w.offset(last * 4),
                slot: metadata.slot.offset(last * 8),
                seq_len: metadata.seq_len.offset(last * 4),
                block_table: metadata.block_table.offset(last * mb * 4),
                num_seqs: 1,
                ..metadata
            };
            let token_ctx = ForwardContext {
                attn_metadata: Some(token_metadata),
                ..*ctx
            };
            let attn_input = attn_inputs.offset(last * core_bytes);
            let attn_out = if let Some((qkv, qkv_row_bytes)) = exact_qkv {
                self.attention_forward_preprojected(
                    attn_input,
                    qkv.offset(last * qkv_row_bytes),
                    seq_len + last,
                    false,
                    block_table,
                    disk_block_ids,
                    disk_last_offloaded_per_layer,
                    kv_cache,
                    &token_ctx,
                    stream,
                )?
            } else {
                self.attention_forward(
                    attn_input,
                    seq_len + last,
                    false,
                    block_table,
                    disk_block_ids,
                    disk_last_offloaded_per_layer,
                    kv_cache,
                    &token_ctx,
                    stream,
                )?
            };
            let (mixed_mlp, inject_mlp) = mlp_hyper.prepare_decode_after_inject_nosave(
                hidden_last,
                attn_out,
                attn_hyper.saved_inject(residual_last),
                residual_last,
                ctx.buffers,
                ctx.gpu,
                eps,
                stream,
            )?;
            let moe_out = self.ffn.forward(mixed_mlp, ctx, stream)?;
            mlp_hyper.inject_decode(
                hidden_last,
                moe_out,
                inject_mlp.expect("Qwen4 decoder mixer has injection weights"),
                ctx.gpu,
                stream,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_qwen4_cross_seq_mtp_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        seq_lens: &[usize],
        block_tables: &mut [&mut Vec<u32>],
        disk_block_ids: &mut [&mut Vec<u32>],
        disk_last_offloaded_per_layer: &mut [&mut Vec<u32>],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let num_seqs = seq_lens.len();
        anyhow::ensure!(
            qwen4_cross_seq_mtp_rows_valid(num_seqs)
                && block_tables.len() == num_seqs
                && disk_block_ids.len() == num_seqs
                && disk_last_offloaded_per_layer.len() == num_seqs,
            "Qwen4 cross-sequence MTP requires 1..=64 matched sequences, got {num_seqs}"
        );
        let (attn_hyper, mlp_hyper) = match (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper) {
            (Some(attn), Some(mlp)) => (attn, mlp),
            _ => anyhow::bail!("Qwen4 cross-sequence MTP requested without hyperconnections"),
        };
        let metadata = ctx.attn_metadata.ok_or_else(|| {
            anyhow::anyhow!("Qwen4 cross-sequence MTP attention requires metadata")
        })?;
        anyhow::ensure!(
            !metadata.qwen4_qsa_required
                && self.mla.is_none()
                && !self.high_speed_swap_engaged(kv_cache),
            "Qwen4 cross-sequence MTP requires standard BF16 or NVFP4 KV without QSA or HSS"
        );
        if num_seqs == 1 {
            let single_ctx = ForwardContext {
                attn_metadata: Some(crate::layer::AttnMetadataDev {
                    num_seqs: 1,
                    ..metadata
                }),
                ..*ctx
            };
            let mut empty_state = crate::layer::EmptyLayerState;
            return self.decode_inner(
                hidden,
                residual,
                &mut empty_state,
                kv_cache,
                seq_lens[0],
                false,
                block_tables[0],
                disk_block_ids[0],
                disk_last_offloaded_per_layer[0],
                &single_ctx,
                stream,
            );
        }
        let h = ctx.config.hidden_size;
        let row_bytes = ctx.config.residual_width() * 2;
        let core_bytes = h * 2;
        let eps = ctx.config.rms_norm_eps as f32;
        let mb = metadata.max_blocks_per_seq as usize;
        let attn_inputs = ctx.buffers.gate_logits();
        anyhow::ensure!(
            num_seqs * core_bytes <= ctx.buffers.sizes().gate_logits,
            "Qwen4 cross-sequence MTP attention input staging exceeds gate-logit workspace"
        );
        let tc_dense_scope =
            crate::layers::qwen4_hyper::batched_verify_tc_dense_active(num_seqs, ctx.graph_capture);
        let _tc_dense_guard =
            crate::layers::qwen4_hyper::BatchedVerifyTcDenseGuard::enter(tc_dense_scope);
        let tc_dense_fp8 = crate::layers::qwen4_hyper::batched_verify_tc_component_active(
            crate::layers::qwen4_hyper::BatchedVerifyTcComponent::Fp8Dense,
            num_seqs,
            ctx.graph_capture,
        );
        attn_hyper.prepare_batched_into(
            hidden,
            residual,
            num_seqs,
            ctx.buffers,
            Some(attn_inputs),
            ctx.gpu,
            eps,
            stream,
        )?;
        let exact_qkv =
            match self.qwen4_k5_project_qkv_tc_dense(attn_inputs, num_seqs, ctx, stream)? {
                Some(qkv) => Some(qkv),
                None => self.qwen4_k5_project_qkv_exact(attn_inputs, num_seqs, ctx, stream)?,
            };
        let attn_width = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads)
            * self.head_dim_override.unwrap_or(ctx.config.head_dim);
        let attn_row_bytes = attn_width * 2;
        let oproj_rows = exact_qkv.is_some()
            && self.mla.is_none()
            && self.o_dense_bf16.is_none()
            && self.o_weight.as_ref().and_then(|w| w.as_fp8()).is_none()
            && !self.attn.o_proj.is_null()
            && matches!(
                self.w4a16_exact_o_proj_kernels
                    .route_for_rows(num_seqs as u32),
                Some(ops::ExactLmHeadRoute::Exact(_))
            )
            && num_seqs * attn_row_bytes <= ctx.buffers.sizes().ssm_qkvz
            && num_seqs * core_bytes <= ctx.buffers.sizes().norm_output;
        let raw_rows = ctx.buffers.ssm_qkvz();
        let can_batch_rope = if !self.qwen4_yarn_inv_freq.is_null() {
            self.rope_strided_yarn_scaled_k.0 != 0
        } else {
            !self.rope_proportional && !self.mrope_interleaved && self.rope_strided_b3_k.0 != 0
        };
        let uniform_positions = metadata.positions_h == metadata.positions
            && metadata.positions_w == metadata.positions;
        let can_batch_kv = matches!(
            self.kv_dtype,
            spark_runtime::kv_cache::KvCacheDtype::Bf16
                | spark_runtime::kv_cache::KvCacheDtype::Nvfp4
        ) && kv_cache.dtype_for_layer(self.attn_layer_idx) == self.kv_dtype
            && (self.kv_dtype == spark_runtime::kv_cache::KvCacheDtype::Bf16 || num_seqs <= 32);
        if oproj_rows
            && can_batch_rope
            && uniform_positions
            && can_batch_kv
            && let Some((qkv, qkv_row_bytes)) = exact_qkv
            && qkv_row_bytes.is_multiple_of(2)
        {
            let nq = self
                .num_q_heads_override
                .unwrap_or(ctx.config.num_attention_heads) as u32;
            let nkv = self
                .num_kv_heads_override
                .unwrap_or(ctx.config.num_key_value_heads) as u32;
            let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
            let bs = kv_cache.block_size() as u32;
            let q_dim = nq * hd;
            let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
            let q_proj_bytes = q_proj_dim as usize * 2;
            let kv_bytes = (nkv * hd) as usize * 2;
            let qkv_row_stride_elems = (qkv_row_bytes / 2) as u32;
            let k_offset_bf16 = (q_proj_bytes / 2) as u32;
            let rotary_dim = self
                .rotary_dim_override
                .unwrap_or(ctx.config.rotary_dim() as u32);
            let rope_theta = self
                .rope_theta_override
                .unwrap_or(ctx.config.rope_theta as f32);

            if !self.qwen4_yarn_inv_freq.is_null() {
                let block_x = (rotary_dim / 2).next_power_of_two().clamp(32, 128);
                spark_runtime::kernel_args::KernelLaunch::new(
                    ctx.gpu,
                    self.rope_strided_yarn_scaled_k,
                )
                .grid([nq + nkv, num_seqs as u32, 1])
                .block([block_x, 1, 1])
                .arg_ptr(qkv)
                .arg_ptr(metadata.positions)
                .arg_u32(num_seqs as u32)
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
            } else {
                ops::rope_strided_b3(
                    ctx.gpu,
                    self.rope_strided_b3_k,
                    qkv,
                    metadata.positions,
                    num_seqs as u32,
                    qkv_row_stride_elems,
                    k_offset_bf16,
                    nq,
                    nkv,
                    hd,
                    rotary_dim,
                    rope_theta,
                    stream,
                )?;
            }

            let k_base = qkv.offset(q_proj_bytes);
            let v_base = k_base.offset(kv_bytes);
            self.write_kv_cache(
                ctx.gpu,
                k_base,
                v_base,
                kv_cache,
                metadata.slot,
                num_seqs as u32,
                nkv,
                hd,
                bs,
                qkv_row_stride_elems,
                qkv_row_stride_elems,
                stream,
                ctx.graph_capture,
            )?;

            let inv_sqrt_d = self.effective_attn_scale(hd);
            let ragged_ws = ctx.buffers.ssm_deinterleaved();
            let ragged_ws_cap = ctx.buffers.sizes().ssm_deinterleaved;
            let ran_ragged = attn_row_bytes == (q_dim as usize) * 2
                && !ragged_ws.is_null()
                && ragged_ws != qkv
                && ragged_ws != raw_rows
                && self.try_run_gqa12_ragged_paged_decode(
                    ctx.gpu,
                    qkv,
                    kv_cache,
                    raw_rows,
                    ragged_ws,
                    ragged_ws_cap,
                    metadata.block_table,
                    metadata.seq_len,
                    metadata.max_blocks_per_seq,
                    num_seqs as u32,
                    nq,
                    nkv,
                    hd,
                    bs,
                    inv_sqrt_d,
                    qkv_row_stride_elems,
                    false,
                    stream,
                )?;
            let fused_dense_gate = if ran_ragged {
                false
            } else {
                let max_seq_len_host = if ctx.graph_capture {
                    0
                } else {
                    seq_lens.iter().copied().max().unwrap_or(0) as u32 + 1
                };
                self.run_paged_decode_with_gate(
                    ctx.gpu,
                    qkv,
                    kv_cache,
                    raw_rows,
                    metadata.block_table,
                    metadata.seq_len,
                    metadata.max_blocks_per_seq,
                    num_seqs as u32,
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
                    max_seq_len_host,
                    self.gated
                        .then(|| (qkv.offset(q_dim as usize * 2), qkv_row_stride_elems)),
                    stream,
                )?
            };
            if self.gated && !fused_dense_gate {
                if self.sigmoid_gate_mul_batched_k.0 != 0 && attn_row_bytes == (q_dim as usize) * 2
                {
                    ops::sigmoid_gate_mul_batched(
                        ctx.gpu,
                        self.sigmoid_gate_mul_batched_k,
                        raw_rows,
                        qkv.offset(q_dim as usize * 2),
                        raw_rows,
                        q_dim,
                        qkv_row_stride_elems,
                        num_seqs as u32,
                        stream,
                    )?;
                } else {
                    for row in 0..num_seqs {
                        let row_attn = raw_rows.offset(row * attn_row_bytes);
                        let gate_ptr = qkv.offset(row * qkv_row_bytes + q_dim as usize * 2);
                        ops::sigmoid_gate_mul(
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
        } else {
            for row in 0..num_seqs {
                let hidden_row = hidden.offset(row * row_bytes);
                let residual_row = residual.offset(row * row_bytes);
                let token_metadata = crate::layer::AttnMetadataDev {
                    positions: metadata.positions.offset(row * 4),
                    positions_h: metadata.positions_h.offset(row * 4),
                    positions_w: metadata.positions_w.offset(row * 4),
                    slot: metadata.slot.offset(row * 8),
                    seq_len: metadata.seq_len.offset(row * 4),
                    block_table: metadata.block_table.offset(row * mb * 4),
                    num_seqs: 1,
                    ..metadata
                };
                let token_ctx = ForwardContext {
                    attn_metadata: Some(token_metadata),
                    ..*ctx
                };
                let attn_input = attn_inputs.offset(row * core_bytes);
                if oproj_rows && let Some((qkv, qkv_row_bytes)) = exact_qkv {
                    let raw_row_dst = raw_rows.offset(row * attn_row_bytes);
                    let _ = self.attention_forward_preprojected_deferred_into(
                        attn_input,
                        qkv.offset(row * qkv_row_bytes),
                        seq_lens[row],
                        Some(raw_row_dst),
                        false,
                        block_tables[row],
                        disk_block_ids[row],
                        disk_last_offloaded_per_layer[row],
                        kv_cache,
                        &token_ctx,
                        stream,
                    )?;
                    continue;
                }
                let preprojected_qkv =
                    exact_qkv.map(|(qkv, qkv_row_bytes)| qkv.offset(row * qkv_row_bytes));
                let (attn_out, _) = self.attention_forward_verify_host_row_with_route(
                    attn_input,
                    preprojected_qkv,
                    seq_lens[row],
                    false,
                    block_tables[row],
                    disk_block_ids[row],
                    disk_last_offloaded_per_layer[row],
                    kv_cache,
                    &token_ctx,
                    stream,
                )?;
                attn_hyper.inject_decode(
                    hidden_row,
                    attn_out,
                    attn_hyper.saved_inject(residual_row),
                    ctx.gpu,
                    stream,
                )?;
            }
        }
        if oproj_rows {
            let o_out = ctx.buffers.norm_output();
            if !(tc_dense_fp8
                && self.try_qwen4_oproj_direct_nvfp4(
                    raw_rows, o_out, num_seqs, h, attn_width, ctx, stream,
                )?)
            {
                ops::w4a16_gemv_batch_logits_exact_with(
                    ctx.gpu,
                    self.w4a16_exact_o_proj_kernels,
                    raw_rows,
                    &self.attn.o_proj,
                    o_out,
                    num_seqs as u32,
                    h as u32,
                    attn_width as u32,
                    stream,
                    ops::w4a16_exact_rt2_for_rows(
                        ops::w4a16_gemv_rt2_enabled(),
                        ops::w4a16_exact_rt2_min_rows(),
                        num_seqs as u32,
                    ),
                )?;
            }
            if ctx.config.tp_world_size > 1
                && let Some(comm) = ctx.comm
            {
                comm.all_reduce_async(o_out.0, num_seqs * h * 2, stream)?;
            }
            attn_hyper.inject_saved_batched(hidden, o_out, residual, num_seqs, ctx.gpu, stream)?;
        }
        let ffn_inputs = mlp_hyper.prepare_batched(
            hidden,
            residual,
            num_seqs,
            ctx.buffers,
            ctx.gpu,
            eps,
            stream,
        )?;
        // Single-pass MoE across all `num_seqs` (2..=64) on the unified `_t_lanes`
        // route: avoids splitting `b=16` into `9 + 7` worklist tiles (and the
        // redundant preflight/graph-capture gate in `forward_qwen4_native_target_rows`),
        // letting `forward_qwen4_exact` run one TC MoE or one wide worklist pass per step.
        if let crate::layers::FfnComponent::Moe(moe) = &self.ffn
            && moe.t_lanes_rows_route()
            && (2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&num_seqs)
        {
            moe.forward_qwen4_exact(ffn_inputs, num_seqs, ctx, stream)?;
        } else if num_seqs <= crate::layers::QWEN4_HYBRID_MAX_ROWS
            && crate::layers::qwen4_hybrid_rows(num_seqs)
        {
            self.ffn
                .forward_qwen4_exact_rows(ffn_inputs, num_seqs, ctx, stream)?;
        } else {
            self.ffn
                .forward_qwen4_native_target_rows(ffn_inputs, num_seqs, ctx, stream)?;
        }
        let ffn_output = ctx.buffers.moe_output();
        mlp_hyper.inject_saved_batched(hidden, ffn_output, residual, num_seqs, ctx.gpu, stream)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_kv_only_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let meta_qsa = ctx.attn_metadata.is_some_and(|m| m.qwen4_qsa_required);
        if let Some(attn_hyper) = &self.qwen4_attn_hyper
            && self.qwen4_mlp_hyper.is_some()
            && self.mla.is_none()
            && !meta_qsa
            && !self.high_speed_swap_engaged(kv_cache)
        {
            let eps = ctx.config.rms_norm_eps as f32;
            let (mixed, _) = attn_hyper.prepare_decode_nosave(
                hidden,
                residual,
                ctx.buffers,
                ctx.gpu,
                eps,
                stream,
            )?;
            return self.attention_forward_kv_only(
                mixed,
                None,
                seq_len,
                block_table,
                kv_cache,
                ctx,
                stream,
            );
        }
        self.decode_inner(
            hidden,
            residual,
            state,
            kv_cache,
            seq_len,
            false,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    pub(super) fn decode_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        _state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        transactional_qsa: bool,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        if let (Some(attn_hyper), Some(mlp_hyper)) = (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper)
        {
            let (mixed, inject) = attn_hyper.prepare_decode_nosave(
                hidden,
                residual,
                ctx.buffers,
                ctx.gpu,
                eps,
                stream,
            )?;
            spark_runtime::cublaslt::stage_timer::mark("hc_in", stream);
            let attn_out = self.attention_forward(
                mixed,
                seq_len,
                transactional_qsa,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                kv_cache,
                ctx,
                stream,
            )?;
            let (mixed, inject) = mlp_hyper.prepare_decode_after_inject_nosave(
                hidden,
                attn_out,
                inject.expect("Qwen4 decoder mixer has injection weights"),
                residual,
                ctx.buffers,
                ctx.gpu,
                eps,
                stream,
            )?;
            spark_runtime::cublaslt::stage_timer::mark("hc_in2", stream);
            let moe_out = self.ffn.forward(mixed, ctx, stream)?;
            mlp_hyper.inject_decode(
                hidden,
                moe_out,
                inject.expect("Qwen4 decoder mixer has injection weights"),
                ctx.gpu,
                stream,
            )?;
            spark_runtime::cublaslt::stage_timer::mark("inject2", stream);
            return Ok(());
        }
        // Disable diagnostics during CUDA graph capture — diag_norm does d2h
        // copy + sync which invalidates stream capture (status 901).
        let gemma4_diag =
            ctx.config.model_type == "gemma4" && gemma4_diag_enabled() && !ctx.graph_capture;
        // When Gemma-4 FP32 residual is active, `hidden` is a FP32 buffer;
        // use the FP32 diag reader so we don't alias 4-byte floats as two
        // 2-byte BF16s (producing bogus NaNs in the print).
        let hidden_is_fp32 = ctx.config.use_fp32_residual();
        let diag_hidden =
            |gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize, stream: u64, label: &str| {
                if hidden_is_fp32 {
                    diag_norm_f32(gpu, ptr, n, stream, label);
                } else {
                    diag_norm(gpu, ptr, n, stream, label);
                }
            };

        let normed = ctx.buffers.norm_output();
        if gemma4_diag {
            diag_hidden(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("L{:02} hidden_in", self.attn_layer_idx),
            );
        }
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            1,
            h as u32,
            eps,
            stream,
        )?;
        if gemma4_diag {
            diag_norm(
                ctx.gpu,
                normed,
                h,
                stream,
                &format!("L{:02} normed", self.attn_layer_idx),
            );
        }

        let attn_out = self.attention_forward(
            normed,
            seq_len,
            transactional_qsa,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_cache,
            ctx,
            stream,
        )?;
        // Note: TP all-reduce on `attn_out` is performed inside `attention_forward_oproj`
        // (and inside `attention_forward_mla` where applicable) so both Qwen4 HC
        // and standard attention blocks reduce exactly once.
        if gemma4_diag {
            diag_norm(
                ctx.gpu,
                attn_out,
                h,
                stream,
                &format!("L{:02} attn_out", self.attn_layer_idx),
            );
        }

        // Gemma-4: post-attention norm (applied unscaled to attn output before
        // residual add; `layer_scalar` is applied to `hidden` at layer end).
        if let Some(ref post_norm) = self.post_attn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                attn_out,
                post_norm,
                attn_out,
                1,
                h as u32,
                eps,
                stream,
            )?;
            if gemma4_diag {
                diag_norm(
                    ctx.gpu,
                    attn_out,
                    h,
                    stream,
                    &format!("L{:02} post_attn_normed", self.attn_layer_idx),
                );
            }
        }

        // Standalone attention (Nemotron-H): no post-attn FFN
        if self.ffn.is_none() {
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                attn_out,
                h as u32,
                stream,
            )?;
            return Ok(());
        }

        // Profile: time attention vs FFN/MoE separately without diverging from
        // the canonical FFN path (including Gemma-4 26B dual-FFN).
        let profile_t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            attn_out,
            &self.post_attn_norm,
            normed2,
            residual,
            1,
            h as u32,
            eps,
            stream,
        )?;

        // Gemma-4 26B MoE dual FFN: run MoE FIRST (before dense FFN result is used)
        // to avoid buffer conflicts (MoE fused kernel uses attn_output internally).
        //
        // HF reference: combined = norm(norm1(mlp_out) + norm2(moe_out))
        //               hidden = residual + combined
        if let (Some(moe_ffn), Some(_pre_norm), Some(post_norm), Some(dense_norm)) = (
            &self.moe_ffn,
            &self.pre_moe_norm,
            &self.post_moe_out_norm,
            &self.post_dense_ffn_norm,
        ) {
            // 1. Run MoE on raw residual (before dense FFN output is touched).
            //    MoE writes result to moe_output buffer.
            let moe_out = moe_ffn.forward(hidden, ctx, stream)?;
            // post-MoE norm (in-place on moe_output)
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                moe_out,
                post_norm,
                moe_out,
                1,
                h as u32,
                eps,
                stream,
            )?;
            // Save normed MoE output — dense FFN will overwrite moe_output.
            // Use logits buffer (vocab_size * 2 bytes >> h * 2) — gate_logits is too small
            let moe_saved = ctx.buffers.logits();
            ctx.gpu.copy_d2d_async(moe_out, moe_saved, h * 2, stream)?;

            // 2. Dense FFN (writes to moe_output, overwriting MoE result)
            let dense_out = self.ffn.forward(normed2, ctx, stream)?;
            // post-dense norm (layernorm_1)
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                dense_out,
                dense_norm,
                dense_out,
                1,
                h as u32,
                eps,
                stream,
            )?;

            // 3. Combine: dense_normed + moe_normed → dense_out (in-place)
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                dense_out,
                moe_saved,
                h as u32,
                stream,
            )?;

            // 4. post_feedforward_layernorm on combined
            if let Some(ref combined_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    dense_out,
                    combined_norm,
                    dense_out,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }

            if let Some(t0) = profile_t0 {
                ctx.gpu.synchronize(stream)?;
                let moe_us = t0.elapsed().as_micros();
                tracing::info!("  Attn-MoE: {:.1}ms", moe_us as f64 / 1000.0);
            }

            // 5. Residual add: hidden += combined
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                dense_out,
                h as u32,
                stream,
            )?;
        } else {
            // Non-MoE (31B dense)
            if gemma4_diag {
                diag_norm(
                    ctx.gpu,
                    normed2,
                    h,
                    stream,
                    &format!("L{:02} normed2", self.attn_layer_idx),
                );
            }
            let dense_out = self.ffn.forward(normed2, ctx, stream)?;
            if gemma4_diag {
                diag_norm(
                    ctx.gpu,
                    dense_out,
                    h,
                    stream,
                    &format!("L{:02} dense_out", self.attn_layer_idx),
                );
            }
            if let Some(ref post_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    dense_out,
                    post_norm,
                    dense_out,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
                if gemma4_diag {
                    diag_norm(
                        ctx.gpu,
                        dense_out,
                        h,
                        stream,
                        &format!("L{:02} post_ffn_normed", self.attn_layer_idx),
                    );
                }
            }
            if let Some(t0) = profile_t0 {
                ctx.gpu.synchronize(stream)?;
                let moe_us = t0.elapsed().as_micros();
                tracing::info!("  Attn-MoE: {:.1}ms", moe_us as f64 / 1000.0);
            }
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                dense_out,
                h as u32,
                stream,
            )?;
        }

        if gemma4_diag {
            diag_hidden(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("L{:02} post_residual", self.attn_layer_idx),
            );
        }

        // Gemma-4: hidden *= layer_scalar at end of layer
        if let Some(scalar) = self.layer_scalar {
            self.apply_layer_scalar(
                ctx.gpu,
                hidden,
                h,
                scalar,
                stream,
                ctx.config.use_fp32_residual(),
            )?;
            if gemma4_diag {
                diag_hidden(
                    ctx.gpu,
                    hidden,
                    h,
                    stream,
                    &format!(
                        "L{:02} post_layer_scalar(scalar={:.4})",
                        self.attn_layer_idx, scalar
                    ),
                );
            }
        }

        Ok(())
    }
}

impl Qwen3AttentionLayer {
    /// Batched dense paged attention for `C=1` Qwen4 verify frames when every
    /// row lies strictly below the QSA 2048 threshold (`seq_len + num_tokens <= 2048`
    /// or `!qsa_required`). Replaces the per-row serial loop (RoPE + KV-write +
    /// paged decode + gate per row) with a single batched RoPE, single batched
    /// KV-write, single `try_run_gqa12_ragged_paged_decode` (or fused
    /// `run_paged_decode_with_gate`), and single batched sigmoid gate.
    #[allow(clippy::too_many_arguments)]
    fn try_qwen4_batched_dense_attention_rows(
        &self,
        raw_rows: DevicePtr,
        num_tokens: usize,
        seq_len: usize,
        exact_qkv: Option<(DevicePtr, usize)>,
        device_attention_plan: Option<Qwen4K5DeviceAttentionPlan>,
        oproj_rows: bool,
        attn_row_bytes: usize,
        qsa_rows_precomputed: bool,
        kv_cache: &mut PagedKvCache,
        metadata: crate::layer::AttnMetadataDev,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<crate::layers::qwen4_qsa::VerifyAttentionRoute>> {
        let Some((qkv, qkv_row_bytes)) = exact_qkv else {
            return Ok(None);
        };
        if !qkv_row_bytes.is_multiple_of(2) {
            return Ok(None);
        }
        let uniform_positions = metadata.positions_h == metadata.positions
            && metadata.positions_w == metadata.positions;
        let mrope_batch_k = if self.qwen4_yarn_inv_freq.is_null() && self.mrope_interleaved {
            Some(super::multi_seq::attn::cached_multiseq_attn_batch_kernels(
                ctx.gpu,
            ))
        } else {
            None
        };
        let rope_route = select_qwen4_batched_dense_rope_route(
            !self.qwen4_yarn_inv_freq.is_null(),
            self.rope_strided_yarn_scaled_k.0 != 0,
            self.mrope_interleaved,
            self.yarn.is_some(),
            self.rope_proportional,
            uniform_positions,
            mrope_batch_k.is_some_and(|k| k.mrope_multi_seq.0 != 0),
            mrope_batch_k.is_some_and(|k| k.mrope_yarn_multi_seq.0 != 0),
            self.rope_strided_b3_k.0 != 0,
        );
        let can_batch_rope = rope_route.is_some();
        let layer_kv_dtype = kv_cache.dtype_for_layer(self.attn_layer_idx);
        let can_batch_kv = matches!(
            self.kv_dtype,
            spark_runtime::kv_cache::KvCacheDtype::Bf16
                | spark_runtime::kv_cache::KvCacheDtype::Nvfp4
        ) && layer_kv_dtype == self.kv_dtype
            && (self.kv_dtype == spark_runtime::kv_cache::KvCacheDtype::Bf16 || num_tokens <= 32);
        let qsa_required = ctx.config.is_qwen4_exp() && metadata.qwen4_qsa_required;
        if !qwen4_batched_dense_attn_eligible(
            num_tokens,
            seq_len,
            qsa_required,
            qsa_rows_precomputed,
            device_attention_plan.is_some(),
            oproj_rows,
            self.mla.is_some(),
            self.high_speed_swap_engaged(kv_cache),
            can_batch_kv,
            can_batch_rope,
        ) || num_tokens * attn_row_bytes > ctx.buffers.sizes().ssm_qkvz
        {
            return Ok(None);
        }
        let Some(rope_route) = rope_route else {
            return Ok(None);
        };

        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let q_dim = nq * hd;
        if attn_row_bytes != (q_dim as usize) * 2 {
            return Ok(None);
        }
        let bs = kv_cache.block_size() as u32;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let q_proj_bytes = q_proj_dim as usize * 2;
        let kv_bytes = (nkv * hd) as usize * 2;
        let qkv_row_stride_elems = (qkv_row_bytes / 2) as u32;
        let k_offset_bf16 = (q_proj_bytes / 2) as u32;
        let rotary_dim = self
            .rotary_dim_override
            .unwrap_or(ctx.config.rotary_dim() as u32);
        let rope_theta = self
            .rope_theta_override
            .unwrap_or(ctx.config.rope_theta as f32);

        static LOGGED_BATCHED_DENSE: std::sync::Once = std::sync::Once::new();
        LOGGED_BATCHED_DENSE.call_once(|| {
            tracing::info!(
                rows = num_tokens,
                rope_route = rope_route.as_str(),
                kv_dtype = ?self.kv_dtype,
                "QWEN4_BATCHED_DENSE_ATTN_ENGAGED"
            );
        });

        match rope_route {
            Qwen4BatchedDenseRopeRoute::YarnScaled => {
                let block_x = (rotary_dim / 2).next_power_of_two().clamp(32, 128);
                spark_runtime::kernel_args::KernelLaunch::new(
                    ctx.gpu,
                    self.rope_strided_yarn_scaled_k,
                )
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
            }
            Qwen4BatchedDenseRopeRoute::MropeYarnMultiSeq => {
                let batch_k = mrope_batch_k.expect("mrope kernels loaded for MropeYarnMultiSeq");
                let yarn = self
                    .yarn
                    .expect("yarn params present for MropeYarnMultiSeq");
                ops::rope_mrope_interleaved_yarn_multi_seq(
                    ctx.gpu,
                    batch_k.mrope_yarn_multi_seq,
                    qkv,
                    metadata.positions,
                    metadata.positions_h,
                    metadata.positions_w,
                    num_tokens as u32,
                    qkv_row_stride_elems,
                    k_offset_bf16,
                    nq,
                    nkv,
                    hd,
                    rotary_dim,
                    rope_theta,
                    yarn.factor,
                    yarn.correction_low,
                    yarn.correction_high,
                    yarn.attention_factor,
                    stream,
                )?;
            }
            Qwen4BatchedDenseRopeRoute::MropeMultiSeq => {
                let batch_k = mrope_batch_k.expect("mrope kernels loaded for MropeMultiSeq");
                ops::rope_mrope_interleaved_multi_seq(
                    ctx.gpu,
                    batch_k.mrope_multi_seq,
                    qkv,
                    metadata.positions,
                    metadata.positions_h,
                    metadata.positions_w,
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
            }
            Qwen4BatchedDenseRopeRoute::StridedB3 => {
                ops::rope_strided_b3(
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
        let ragged_ws = ctx.buffers.ssm_deinterleaved();
        let ragged_ws_cap = ctx.buffers.sizes().ssm_deinterleaved;
        let ran_ragged = !ragged_ws.is_null()
            && ragged_ws != qkv
            && ragged_ws != raw_rows
            && self.try_run_gqa12_ragged_paged_decode(
                ctx.gpu,
                qkv,
                kv_cache,
                raw_rows,
                ragged_ws,
                ragged_ws_cap,
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
                false,
                stream,
            )?;
        let fused_dense_gate = if ran_ragged {
            false
        } else {
            let max_seq_len_host = if ctx.graph_capture {
                0
            } else {
                (seq_len + num_tokens) as u32
            };
            self.run_paged_decode_with_gate(
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
                max_seq_len_host,
                self.gated
                    .then(|| (qkv.offset(q_dim as usize * 2), qkv_row_stride_elems)),
                stream,
            )?
        };
        spark_runtime::cublaslt::stage_timer::mark("core", stream);

        if self.gated && !fused_dense_gate {
            if self.sigmoid_gate_mul_batched_k.0 != 0 {
                ops::sigmoid_gate_mul_batched(
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
                    ops::sigmoid_gate_mul(
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

    /// `ATLAS_QWEN4_VERIFY_OPROJ_ROWS=1` (or an active batched-verify TC dense
    /// scope at `rows >= 4`, i.e. `DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS`) and
    /// an ordinary NVFP4 O projection whose exact multi-row tier exists for
    /// `rows`, with room to stage the rows (gated attention rows in the SSM QKVZ
    /// arena, projected rows in the norm arena, as the exact O16 prefill does).
    pub(super) fn verify_oproj_rows_admits(
        &self,
        rows: usize,
        attn_row_bytes: usize,
        core_bytes: usize,
        ctx: &ForwardContext,
    ) -> bool {
        let tc_dense_oproj = crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active()
            && crate::layers::qwen4_hyper::batched_verify_tc_component_active(
                crate::layers::qwen4_hyper::BatchedVerifyTcComponent::Fp8Dense,
                rows,
                ctx.graph_capture,
            );
        (qwen4_verify_oproj_rows_selected() || tc_dense_oproj)
            && self.mla.is_none()
            && self.o_dense_bf16.is_none()
            && self.o_weight.as_ref().and_then(|w| w.as_fp8()).is_none()
            && !self.attn.o_proj.is_null()
            && matches!(
                self.w4a16_exact_o_proj_kernels.route_for_rows(rows as u32),
                Some(ops::ExactLmHeadRoute::Exact(_))
            )
            && rows * attn_row_bytes <= ctx.buffers.sizes().ssm_qkvz
            && rows * core_bytes <= ctx.buffers.sizes().norm_output
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
const fn qwen4_batched_dense_attn_eligible(
    num_tokens: usize,
    seq_len: usize,
    qsa_required: bool,
    qsa_rows_precomputed: bool,
    device_attention_plan_active: bool,
    oproj_rows: bool,
    mla_active: bool,
    hss_active: bool,
    kv_batch_supported: bool,
    rope_batch_supported: bool,
) -> bool {
    if num_tokens < 2
        || !oproj_rows
        || device_attention_plan_active
        || mla_active
        || hss_active
        || !kv_batch_supported
        || !rope_batch_supported
    {
        return false;
    }
    if qsa_required {
        if !qsa_rows_precomputed {
            return false;
        }
        match seq_len.checked_add(num_tokens) {
            Some(end) if end <= crate::layers::qwen4_qsa::VERIFY_ROWS_DENSE_ENDPOINT => {}
            _ => return false,
        }
    }
    true
}

#[inline]
#[must_use]
const fn qwen4_cross_seq_mtp_rows_valid(num_seqs: usize) -> bool {
    num_seqs >= 1 && num_seqs <= 64
}

/// Opt-in gate (`ATLAS_QWEN4_C1_DENSE_ATTN=1`, default off) enabling the
/// batched dense attention path (`try_run_gqa12_ragged_paged_decode`) on
/// single-sequence `C=1` verify frames. Keeping default off preserves strict
/// 0-ULP scalar `paged_attention_decode` parity at `C=1`.
#[inline]
#[must_use]
fn qwen4_c1_dense_attn_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_C1_DENSE_ATTN")
            .ok()
            .as_deref()
            .map(str::trim)
            == Some("1")
    })
}

/// `ATLAS_QWEN4_K5_BATCH_HYPER=0`: select row-exact hyperconnection prep
/// (batched hyperconnection prep is default ON when unset or `!= "0"`).
fn qwen4_k5_batch_hyper_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("ATLAS_QWEN4_K5_BATCH_HYPER").ok().as_deref() == Some("0"))
}

/// Receipt for the first multi-row O projection (the env read alone proves nothing).
static VERIFY_OPROJ_ROWS_LAUNCHED: std::sync::Once = std::sync::Once::new();

/// `ATLAS_QWEN4_VERIFY_OPROJ_ROWS=1`: the K-row verify's attention O
/// projection runs once for all rows on the exact multi-row tier
/// (verify-cost oproj_rows_bench) instead of once per row.
fn qwen4_verify_oproj_rows_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_VERIFY_OPROJ_ROWS")
            .ok()
            .as_deref()
            == Some("1")
            || crate::layers::qwen4_verify_fast_host_selected();
        if on {
            tracing::info!(
                "ATLAS_QWEN4_VERIFY_OPROJ_ROWS=1 requested: exact multi-row verify O projection"
            );
        }
        on
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Qwen4BatchedDenseRopeRoute {
    YarnScaled,
    MropeMultiSeq,
    MropeYarnMultiSeq,
    StridedB3,
}

impl Qwen4BatchedDenseRopeRoute {
    const fn as_str(self) -> &'static str {
        match self {
            Self::YarnScaled => "yarn_scaled",
            Self::MropeMultiSeq => "mrope_multi_seq",
            Self::MropeYarnMultiSeq => "mrope_yarn_multi_seq",
            Self::StridedB3 => "strided_b3",
        }
    }
}

#[inline]
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "pure const decision table for batched dense verify RoPE kernel selection"
)]
const fn select_qwen4_batched_dense_rope_route(
    has_qwen4_yarn_inv_freq: bool,
    has_yarn_scaled_k: bool,
    mrope_interleaved: bool,
    has_yarn: bool,
    rope_proportional: bool,
    uniform_positions: bool,
    has_mrope_multi_seq_k: bool,
    has_mrope_yarn_multi_seq_k: bool,
    has_strided_b3_k: bool,
) -> Option<Qwen4BatchedDenseRopeRoute> {
    if has_qwen4_yarn_inv_freq {
        return if has_yarn_scaled_k {
            Some(Qwen4BatchedDenseRopeRoute::YarnScaled)
        } else {
            None
        };
    }
    if mrope_interleaved {
        return if has_yarn {
            if has_mrope_yarn_multi_seq_k {
                Some(Qwen4BatchedDenseRopeRoute::MropeYarnMultiSeq)
            } else {
                None
            }
        } else if has_mrope_multi_seq_k {
            Some(Qwen4BatchedDenseRopeRoute::MropeMultiSeq)
        } else {
            None
        };
    }
    if !has_yarn && !rope_proportional && uniform_positions && has_strided_b3_k {
        Some(Qwen4BatchedDenseRopeRoute::StridedB3)
    } else {
        None
    }
}

#[cfg(test)]
mod decode_inner_tests {
    use super::{
        Qwen4BatchedDenseRopeRoute, qwen4_batched_dense_attn_eligible,
        qwen4_cross_seq_mtp_rows_valid, select_qwen4_batched_dense_rope_route,
    };

    #[test]
    fn batched_dense_rope_route_supports_interleaved_mrope_and_fails_closed() {
        // Precomputed Qwen4 YaRN inv_freq takes precedence when its kernel is present.
        assert_eq!(
            select_qwen4_batched_dense_rope_route(
                true, true, true, true, false, true, true, true, true
            ),
            Some(Qwen4BatchedDenseRopeRoute::YarnScaled)
        );
        assert_eq!(
            select_qwen4_batched_dense_rope_route(
                true, false, true, true, false, true, true, true, true
            ),
            None
        );

        // Native 256K Qwen3.8-Flash-Next (interleaved mRoPE, no YaRN): selects MropeMultiSeq
        // even for non-uniform multimodal positions because the kernel consumes pos_t/pos_h/pos_w.
        assert_eq!(
            select_qwen4_batched_dense_rope_route(
                false, false, true, false, false, false, true, false, false
            ),
            Some(Qwen4BatchedDenseRopeRoute::MropeMultiSeq)
        );
        assert_eq!(
            select_qwen4_batched_dense_rope_route(
                false, false, true, false, false, true, false, true, true
            ),
            None
        );

        // Interleaved mRoPE with static YaRN selects MropeYarnMultiSeq.
        assert_eq!(
            select_qwen4_batched_dense_rope_route(
                false, false, true, true, false, false, true, true, false
            ),
            Some(Qwen4BatchedDenseRopeRoute::MropeYarnMultiSeq)
        );
        assert_eq!(
            select_qwen4_batched_dense_rope_route(
                false, false, true, true, false, true, true, false, true
            ),
            None
        );

        // Non-mRoPE scalar RoPE selects StridedB3 only for non-YaRN, non-proportional, uniform pos.
        assert_eq!(
            select_qwen4_batched_dense_rope_route(
                false, false, false, false, false, true, false, false, true
            ),
            Some(Qwen4BatchedDenseRopeRoute::StridedB3)
        );
        assert_eq!(
            select_qwen4_batched_dense_rope_route(
                false, false, false, false, true, true, false, false, true
            ),
            None
        );
        assert_eq!(
            select_qwen4_batched_dense_rope_route(
                false, false, false, false, false, false, false, false, true
            ),
            None
        );
    }

    #[test]
    fn batched_dense_attn_admits_pure_dense_prefix_and_refuses_sparse_or_straddling_frames() {
        // Pure dense (non-QSA) admits any num_tokens >= 2 regardless of seq_len.
        for num_tokens in [2usize, 5, 7, 8, 16] {
            assert!(qwen4_batched_dense_attn_eligible(
                num_tokens, 0, false, false, false, true, false, false, true, true
            ));
            assert!(qwen4_batched_dense_attn_eligible(
                num_tokens, 8_192, false, false, false, true, false, false, true, true
            ));
        }
        // Single-row or disabled oproj_rows never batches.
        assert!(!qwen4_batched_dense_attn_eligible(
            1, 0, false, false, false, true, false, false, true, true
        ));
        assert!(!qwen4_batched_dense_attn_eligible(
            7, 0, false, false, false, false, false, false, true, true
        ));

        // QSA-required admits pure-dense prefix (seq_len + num_tokens <= 2048) only when
        // QSA Stage A already precomputed raw_ring/compressed_keys for all rows.
        for num_tokens in [2usize, 5, 7, 8, 9] {
            assert!(qwen4_batched_dense_attn_eligible(
                num_tokens, 0, true, true, false, true, false, false, true, true
            ));
            assert!(qwen4_batched_dense_attn_eligible(
                num_tokens,
                2_048 - num_tokens,
                true,
                true,
                false,
                true,
                false,
                false,
                true,
                true
            ));
            // Missing QSA precomputation fails closed to row-serial so QSA side cache is updated.
            assert!(!qwen4_batched_dense_attn_eligible(
                num_tokens, 0, true, false, false, true, false, false, true, true
            ));
            // Straddling or post-2048 frames fail closed to sparse/row-serial routes.
            assert!(!qwen4_batched_dense_attn_eligible(
                num_tokens,
                2_048 - num_tokens + 1,
                true,
                true,
                false,
                true,
                false,
                false,
                true,
                true
            ));
            assert!(!qwen4_batched_dense_attn_eligible(
                num_tokens, 2_048, true, true, false, true, false, false, true, true
            ));
        }

        // Device attention plan, MLA, HSS, or unsupported KV/RoPE fail closed.
        assert!(!qwen4_batched_dense_attn_eligible(
            7, 0, false, false, true, true, false, false, true, true
        ));
        assert!(!qwen4_batched_dense_attn_eligible(
            7, 0, false, false, false, true, true, false, true, true
        ));
        assert!(!qwen4_batched_dense_attn_eligible(
            7, 0, false, false, false, true, false, true, true, true
        ));
        assert!(!qwen4_batched_dense_attn_eligible(
            7, 0, false, false, false, true, false, false, false, true
        ));
        assert!(!qwen4_batched_dense_attn_eligible(
            7, 0, false, false, false, true, false, false, true, false
        ));
    }

    #[test]
    fn cross_seq_mtp_accepts_one_through_sixty_four_sequences() {
        assert!(!qwen4_cross_seq_mtp_rows_valid(0));
        for num_seqs in [1usize, 2, 4, 8, 16, 32, 64] {
            assert!(
                qwen4_cross_seq_mtp_rows_valid(num_seqs),
                "num_seqs={num_seqs}"
            );
        }
        assert!(!qwen4_cross_seq_mtp_rows_valid(65));
        assert!(!qwen4_cross_seq_mtp_rows_valid(usize::MAX));
    }
}
