// SPDX-License-Identifier: AGPL-3.0-only

//! Batched exact continuation prefill for suppressed Qwen4 attention slices
//! (such as multi-turn Marconi prefix-cache continuations).
//!
//! Uploads the slice's per-token `seq_len` scalars in a single transfer into
//! the idle `ssm_gates` arena instead of synchronizing per token, and steps
//! `2..=8` rows through the exact `qwen4_hybrid_rows` projections and `_t_lanes`
//! MoE kernels while keeping causal KV writes and committed QSA side-cache
//! updates strictly row-ordered (`qsa_journal = false`).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, HostToDeviceCopy};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_qwen4_suppressed(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        base_meta: AttnMetadataDev,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (attn_hyper, mlp_hyper) = match (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper) {
            (Some(attn), Some(mlp)) => (attn, mlp),
            _ => anyhow::bail!("qwen4_exp suppressed prefill missing hyperconnections"),
        };
        let h = ctx.config.hidden_size;
        let core_bytes = h * 2;
        let row_bytes = ctx.config.residual_width() * 2;
        let eps = ctx.config.rms_norm_eps as f32;
        let attn_width = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads)
            * self.head_dim_override.unwrap_or(ctx.config.head_dim);
        let attn_row_bytes = attn_width * 2;

        let seq_lens_u32: Vec<u32> = (1..=num_tokens)
            .map(|i| u32::try_from(seq_len_start + i))
            .collect::<std::result::Result<Vec<u32>, _>>()?;
        let can_stage_all_seq_lens = num_tokens.saturating_mul(4) <= ctx.buffers.sizes().ssm_gates;
        let seq_lens_base = ctx.buffers.ssm_gates();
        if can_stage_all_seq_lens && num_tokens > 0 {
            // SAFETY: `seq_lens_u32` is a live `Vec<u32>` of `num_tokens` elements,
            // so viewing `num_tokens * 4` bytes is in-bounds and properly initialized,
            // and `copy_h2d_group_on_stream` completes before `seq_lens_u32` drops.
            let bytes = unsafe {
                std::slice::from_raw_parts(seq_lens_u32.as_ptr().cast::<u8>(), num_tokens * 4)
            };
            ctx.gpu
                .copy_h2d_group_on_stream(&[HostToDeviceCopy::new(bytes, seq_lens_base)], stream)?;
        }

        let max_tile = if crate::layers::qwen4_prefill_moe::family_suppressed()
            && ctx.ddtree_parent_ids_dev.is_none()
        {
            if crate::layers::qwen4_hybrid_rows(8) {
                8
            } else if crate::layers::qwen4_hybrid_rows(5) {
                5
            } else {
                1
            }
        } else {
            1
        };
        let staging_rows = (ctx.buffers.sizes().gate_logits / core_bytes).max(1);

        let mut t = 0usize;
        while t < num_tokens {
            let width = (num_tokens - t).min(max_tile).min(staging_rows);
            if width == 1 || !crate::layers::qwen4_hybrid_rows(width) {
                let seq_len_ptr = if can_stage_all_seq_lens {
                    seq_lens_base.offset(t * 4)
                } else {
                    let seq_len_bytes = seq_lens_u32[t].to_ne_bytes();
                    ctx.gpu.copy_h2d_group_on_stream(
                        &[HostToDeviceCopy::new(&seq_len_bytes, base_meta.seq_len)],
                        stream,
                    )?;
                    base_meta.seq_len
                };
                let token_meta = AttnMetadataDev {
                    positions: base_meta.positions.offset(t * 4),
                    positions_h: base_meta.positions_h.offset(t * 4),
                    positions_w: base_meta.positions_w.offset(t * 4),
                    slot: base_meta.slot.offset(t * 8),
                    seq_len: seq_len_ptr,
                    ..base_meta
                };
                let token_ctx = ForwardContext {
                    attn_metadata: Some(token_meta),
                    ..*ctx
                };
                self.decode_inner(
                    hidden.offset(t * row_bytes),
                    residual.offset(t * row_bytes),
                    state,
                    kv_cache,
                    seq_len_start + t,
                    false,
                    block_table,
                    disk_block_ids,
                    disk_last_offloaded_per_layer,
                    &token_ctx,
                    stream,
                )?;
                t += 1;
                continue;
            }

            let hidden_t = hidden.offset(t * row_bytes);
            let residual_t = residual.offset(t * row_bytes);
            let attn_inputs = ctx.buffers.gate_logits();
            attn_hyper.prepare_batched_into(
                hidden_t,
                residual_t,
                width,
                ctx.buffers,
                Some(attn_inputs),
                ctx.gpu,
                eps,
                stream,
            )?;
            let exact_qkv = self.qwen4_k5_project_qkv_exact(attn_inputs, width, ctx, stream)?;
            let oproj_rows = exact_qkv.is_some()
                && self.verify_oproj_rows_admits(width, attn_row_bytes, core_bytes, ctx);
            let raw_rows = ctx.buffers.ssm_qkvz();

            for r in 0..width {
                let idx = t + r;
                let seq_len_ptr = if can_stage_all_seq_lens {
                    seq_lens_base.offset(idx * 4)
                } else {
                    let seq_len_bytes = seq_lens_u32[idx].to_ne_bytes();
                    ctx.gpu.copy_h2d_group_on_stream(
                        &[HostToDeviceCopy::new(&seq_len_bytes, base_meta.seq_len)],
                        stream,
                    )?;
                    base_meta.seq_len
                };
                let token_meta = AttnMetadataDev {
                    positions: base_meta.positions.offset(idx * 4),
                    positions_h: base_meta.positions_h.offset(idx * 4),
                    positions_w: base_meta.positions_w.offset(idx * 4),
                    slot: base_meta.slot.offset(idx * 8),
                    seq_len: seq_len_ptr,
                    num_seqs: 1,
                    ..base_meta
                };
                let token_ctx = ForwardContext {
                    attn_metadata: Some(token_meta),
                    ..*ctx
                };
                let attn_input = attn_inputs.offset(r * core_bytes);
                if oproj_rows && let Some((qkv, qkv_row_bytes)) = exact_qkv {
                    self.attention_forward_preprojected_deferred_into(
                        attn_input,
                        qkv.offset(r * qkv_row_bytes),
                        seq_len_start + idx,
                        Some(raw_rows.offset(r * attn_row_bytes)),
                        false,
                        block_table,
                        disk_block_ids,
                        disk_last_offloaded_per_layer,
                        kv_cache,
                        &token_ctx,
                        stream,
                    )?;
                } else {
                    let preprojected_qkv =
                        exact_qkv.map(|(qkv, qkv_row_bytes)| qkv.offset(r * qkv_row_bytes));
                    let (attn_out, _) = self.attention_forward_verify_host_row_with_route(
                        attn_input,
                        preprojected_qkv,
                        seq_len_start + idx,
                        false,
                        block_table,
                        disk_block_ids,
                        disk_last_offloaded_per_layer,
                        kv_cache,
                        &token_ctx,
                        stream,
                    )?;
                    attn_hyper.inject_decode(
                        hidden_t.offset(r * row_bytes),
                        attn_out,
                        attn_hyper.saved_inject(residual_t.offset(r * row_bytes)),
                        ctx.gpu,
                        stream,
                    )?;
                }
            }

            let ffn_inputs = if oproj_rows {
                let o_out = ctx.buffers.norm_output();
                ops::w4a16_gemv_batch_logits_exact_with(
                    ctx.gpu,
                    self.w4a16_exact_o_proj_kernels,
                    raw_rows,
                    &self.attn.o_proj,
                    o_out,
                    width as u32,
                    h as u32,
                    attn_width as u32,
                    stream,
                    ops::w4a16_exact_rt2_for_rows(
                        ops::w4a16_gemv_rt2_enabled(),
                        ops::w4a16_exact_rt2_min_rows(),
                        width as u32,
                    ),
                )?;
                if ctx.config.tp_world_size > 1
                    && let Some(comm) = ctx.comm
                {
                    comm.all_reduce_async(o_out.0, width * h * 2, stream)?;
                }
                mlp_hyper.prepare_batched_after_inject(
                    hidden_t,
                    o_out,
                    residual_t,
                    width,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?
            } else {
                let staging = ctx.buffers.qkv_output().offset(row_bytes);
                mlp_hyper.prepare_batched_into(
                    hidden_t,
                    residual_t,
                    width,
                    ctx.buffers,
                    Some(staging),
                    ctx.gpu,
                    eps,
                    stream,
                )?
            };
            self.ffn.forward_prefill(ffn_inputs, width, ctx, stream)?;
            mlp_hyper.inject_saved_batched(
                hidden_t,
                ctx.buffers.moe_output(),
                residual_t,
                width,
                ctx.gpu,
                stream,
            )?;
            t += width;
        }
        Ok(())
    }
}
