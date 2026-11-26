// SPDX-License-Identifier: AGPL-3.0-only

//! Native Qwen4 cross-sequence attention lane for the opt-in C2-C256
//! (`2..=QWEN4_NATIVE_MAX_ROWS`) model route.
//!
//! This component performs the attention hyperconnection, exact multi-row QKV,
//! independent KV writes, one batched paged-attention launch, O projection,
//! and the attention injection. `decode_a2.rs` selects it only as part of the
//! complete target-only native model loop. Long-context requests remain on the
//! serialized Qwen4 path unless `ATLAS_QWEN4_NATIVE_BATCH_QSA=1` routes their
//! QSA rows through `decode/qwen4_native_qsa.rs`; selector-enabled frames wider
//! than `QWEN4_NATIVE_MAX_ROWS` (256) fail closed.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::attn::attn_qkv_mega_enabled;
use super::ctx::MultiSeqCtx;
use super::qwen4_native_policy::{Admission, RowLayout, validate};
use crate::layer::{EmptyLayerState, ForwardContext, LayerState};
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::layers::qwen3_attention::decode::qwen4_native_qsa::NativeQsaFrame;
use crate::layers::qwen4_qsa::native_batch::{NativeAttnRoute, dense_policy_seq_lens};

fn validate_native_paged_decode_selectors(
    bf16_kgamma_enabled: bool,
) -> std::result::Result<(), &'static str> {
    if bf16_kgamma_enabled {
        return Err(
            "native Qwen4 independent rows cannot use the shared-history BF16 K-gamma kernel; \
             unset ATLAS_UNSAFE_UNVERIFIED_BF16_KGAMMA_EXACT",
        );
    }
    Ok(())
}

impl Qwen3AttentionLayer {
    /// Whole-layer, read-only admission used by the model before it allocates
    /// KV blocks or submits any GPU work.
    pub(in crate::layers::qwen3_attention) fn preflight_qwen4_native_target_inner(
        &self,
        num_seqs: usize,
        states: &[&(dyn LayerState + 'static)],
        ctx: &ForwardContext,
    ) -> Result<()> {
        anyhow::ensure!(
            states.len() == num_seqs,
            "native Qwen4 attention state extent {} != active rows {num_seqs}",
            states.len()
        );
        for (row, state) in states.iter().enumerate() {
            anyhow::ensure!(
                state.as_any().downcast_ref::<EmptyLayerState>().is_some(),
                "native Qwen4 attention row {row} carries non-attention state"
            );
        }
        self.preflight_qwen4_native_target_core(num_seqs, ctx)
    }

    /// Layer capability checks shared by model-wide admission and the
    /// execution hooks. This function is read-only and launches no work.
    pub(in crate::layers::qwen3_attention) fn preflight_qwen4_native_target_core(
        &self,
        num_seqs: usize,
        ctx: &ForwardContext,
    ) -> Result<()> {
        anyhow::ensure!(
            (2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&num_seqs),
            "native Qwen4 attention admits only C2-C{}",
            crate::layers::QWEN4_NATIVE_MAX_ROWS
        );
        anyhow::ensure!(
            ctx.config.is_qwen4_exp()
                && ctx.config.hc_count == 4
                && ctx.config.hidden_size == 2_560
                && ctx.config.residual_width() == 10_240
                && !ctx.config.use_fp32_residual(),
            "native Qwen4 attention requires the canonical BF16 Flash-Next geometry"
        );
        anyhow::ensure!(
            matches!(self.kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Nvfp4)
                && !ctx.graph_capture
                && ctx.config.tp_world_size <= 4
                && (ctx.config.ep_world_size <= 1
                    || ctx.config.ep_world_size == ctx.config.tp_world_size)
                && ctx.tree_aware_attn.is_none()
                && ctx.ddtree_parent_ids_dev.is_none()
                && ctx.self_spec_sparse_draft.is_none()
                && ctx.ffn_defer.is_none(),
            "native Qwen4 attention requires eager BF16/NVFP4 TP<=4 (+ matching EP) target execution"
        );
        anyhow::ensure!(
            self.mrope_interleaved,
            "native Qwen4 attention requires canonical interleaved MRoPE"
        );
        anyhow::ensure!(
            !attn_qkv_mega_enabled(),
            "ATLAS_ATTN_QKV_MEGA is incompatible with native Qwen4 interleaved MRoPE"
        );
        validate_native_paged_decode_selectors(crate::layers::flash_attn_kgamma_bf16_enabled())
            .map_err(anyhow::Error::msg)?;
        let rope_kernel = if self.yarn.is_some() {
            self.rope_mrope_interleaved_yarn_k
        } else {
            self.rope_mrope_interleaved_k
        };
        anyhow::ensure!(
            rope_kernel.0 != 0
                && self.reshape_cache_k.0 != 0
                && self.paged_decode_k.0 != 0
                && self.rms_norm_k.0 != 0
                && self.sigmoid_gate_mul_k.0 != 0,
            "native Qwen4 attention kernel set is incomplete"
        );
        self.validate_yarn_multi_seq_controls()?;
        self.preflight_qwen4_native_tail_selectors(num_seqs)?;
        self.validate_qwen4_native_exact_qkv(num_seqs)?;

        let (attn_hyper, mlp_hyper) = match (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper) {
            (Some(attn), Some(mlp)) => (attn, mlp),
            _ => anyhow::bail!("native Qwen4 attention requires both hyperconnections"),
        };
        anyhow::ensure!(
            attn_hyper.residual_width() == ctx.config.residual_width()
                && mlp_hyper.residual_width() == ctx.config.residual_width(),
            "native Qwen4 attention hyperconnection geometry mismatch"
        );
        attn_hyper.preflight_qwen4_native_batched(num_seqs, true)?;
        mlp_hyper.preflight_qwen4_native_batched(num_seqs, true)?;
        let ordinary_nvfp4_o = self
            .o_weight
            .as_ref()
            .and_then(|weight| weight.as_fp8())
            .is_none()
            && self.o_dense_bf16.is_none()
            && !self.attn.o_proj.is_null();
        anyhow::ensure!(
            ordinary_nvfp4_o,
            "native Qwen4 attention requires the exact ordinary-NVFP4 O projection"
        );
        let o_route = self
            .w4a16_exact_o_proj_kernels
            .route_for_rows(num_seqs as u32)
            .ok_or_else(|| {
                anyhow::anyhow!("native Qwen4 attention has no exact/K1 O-projection tier")
            })?;
        let serial_o_proj = crate::layers::attn_out_serial_enabled();
        if serial_o_proj || matches!(o_route, crate::layers::ops::ExactLmHeadRoute::SerialK1(_)) {
            anyhow::ensure!(
                crate::layers::ops::use_gemv_sw(self.gemv_sw, self.w4a16_gemv_sw_k)
                    || self.w4a16_gemv_k.0 != 0,
                "native Qwen4 attention is missing the K1 O-projection fallback kernel"
            );
        }
        self.ffn.preflight_qwen4_native_exact_rows(num_seqs, ctx)?;

        let core_bytes = num_seqs
            .checked_mul(ctx.config.hidden_size)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention core extent overflow"))?;
        let hyper_bytes = num_seqs
            .checked_mul(ctx.config.residual_width())
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention hyper extent overflow"))?;
        let nq = u32::try_from(
            self.num_q_heads_override
                .unwrap_or(ctx.config.num_attention_heads),
        )
        .map_err(|_| anyhow::anyhow!("native Qwen4 attention Q heads exceed u32"))?;
        let nkv = u32::try_from(
            self.num_kv_heads_override
                .unwrap_or(ctx.config.num_key_value_heads),
        )
        .map_err(|_| anyhow::anyhow!("native Qwen4 attention KV heads exceed u32"))?;
        let hd = u32::try_from(self.head_dim_override.unwrap_or(ctx.config.head_dim))
            .map_err(|_| anyhow::anyhow!("native Qwen4 attention head dim exceeds u32"))?;
        let q_dim_u32 = nq
            .checked_mul(hd)
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention Q extent overflow"))?;
        let q_proj_dim_u32 = q_dim_u32
            .checked_mul(if self.gated { 2 } else { 1 })
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention Q projection overflow"))?;
        let kv_dim_u32 = nkv
            .checked_mul(hd)
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention KV extent overflow"))?;
        let q_dim = q_dim_u32 as usize;
        let q_proj_dim = q_proj_dim_u32 as usize;
        let kv_dim = kv_dim_u32 as usize;
        let per_seq_qkv = q_proj_dim
            .checked_add(
                kv_dim
                    .checked_mul(2)
                    .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention dual-KV overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention QKV row overflow"))?;
        self.preflight_bf16_gqa4_kernel(ctx.gpu, nq, nkv, hd, num_seqs as u32)?;
        let qkv_bytes = num_seqs
            .checked_mul(per_seq_qkv)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention QKV extent overflow"))?;
        let q_scratch_bytes = num_seqs
            .checked_mul(q_proj_dim)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention Q scratch extent overflow"))?;
        let query_bytes = num_seqs
            .checked_mul(q_dim)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention query extent overflow"))?;
        let dual_kv_bytes = num_seqs
            .checked_mul(kv_dim)
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention KV scratch extent overflow"))?;
        let hyper_scratch = num_seqs
            .checked_mul(ctx.config.hc_lowrank)
            .and_then(|elements| elements.checked_mul(2))
            .and_then(|down_bytes| {
                num_seqs
                    .checked_mul(ctx.config.hc_count)
                    .and_then(|elements| elements.checked_mul(2))
                    .and_then(|inject_bytes| down_bytes.checked_add(inject_bytes))
            })
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention HC scratch overflow"))?;
        let arena = ctx.buffers.sizes();
        anyhow::ensure!(
            hyper_bytes <= arena.hidden_states
                && hyper_bytes <= arena.residual
                && hyper_bytes <= arena.qkv_output
                && core_bytes <= arena.norm_output
                && core_bytes <= arena.moe_output
                && qkv_bytes <= arena.qkv_output
                && q_scratch_bytes <= arena.ssm_qkvz
                && query_bytes <= arena.attn_output
                && dual_kv_bytes <= arena.attn_output
                && hyper_scratch <= arena.ssm_ba,
            "native Qwen4 attention rows exceed the serving arenas"
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn decode_qwen4_native_target_attention_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_qwen4_native_target_attention_routed(
            hidden,
            residual,
            num_seqs,
            kv_cache,
            seq_lens,
            block_tables,
            None,
            ctx,
            stream,
        )
    }

    /// `qsa_routes` is `Some` only for an `ATLAS_QWEN4_NATIVE_BATCH_QSA=1`
    /// frame with at least one QSA row; `None` is the unchanged dense lane.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn decode_qwen4_native_target_attention_routed(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        qsa_routes: Option<&[NativeAttnRoute]>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.preflight_qwen4_native_target_core(num_seqs, ctx)?;
        let metadata = ctx
            .attn_metadata
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention requires batch metadata"))?;
        // QSA rows are admitted per row by `native_attn_route`; the dense-lane
        // policy below then sees only the rows that keep no side index.
        let qsa_policy_seq_lens;
        let (policy_seq_lens, policy_qsa_required) = match qsa_routes {
            None => (seq_lens, metadata.qwen4_qsa_required),
            Some(routes) => {
                anyhow::ensure!(
                    routes.len() == num_seqs
                        && metadata.qwen4_qsa_required
                            == routes.iter().any(|route| route.maintains_index()),
                    "native Qwen4 QSA routes disagree with the frame metadata"
                );
                qsa_policy_seq_lens =
                    dense_policy_seq_lens(routes, seq_lens).map_err(anyhow::Error::msg)?;
                self.preflight_qwen4_native_qsa_rows(routes, seq_lens, kv_cache)?;
                (qsa_policy_seq_lens.as_slice(), false)
            }
        };
        validate(Admission {
            rows: num_seqs,
            metadata_rows: metadata.num_seqs as usize,
            seq_lens: policy_seq_lens,
            block_table_rows: block_tables.len(),
            qsa_required: policy_qsa_required,
            is_qwen4: ctx.config.is_qwen4_exp(),
            bf16_kv: matches!(self.kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Nvfp4),
            bf16_residual: !ctx.config.use_fp32_residual(),
            tp_world_size: ctx.config.tp_world_size,
            ep_world_size: ctx.config.ep_world_size,
            comm_present: ctx.comm.is_some(),
            graph_capture: ctx.graph_capture,
            tree_aware: ctx.tree_aware_attn.is_some(),
            ddtree_parent_ids: ctx.ddtree_parent_ids_dev.is_some(),
            self_spec_sparse: ctx.self_spec_sparse_draft.is_some(),
            ffn_defer: ctx.ffn_defer.is_some(),
            mrope_interleaved: self.mrope_interleaved,
            qkv_mega: attn_qkv_mega_enabled(),
        })
        .map_err(anyhow::Error::msg)?;
        let row_layout = RowLayout::new(
            ctx.config.hidden_size,
            ctx.config.hc_count,
            metadata.max_blocks_per_seq as usize,
        )
        .map_err(anyhow::Error::msg)?;
        let row_extents = row_layout.extents(num_seqs).map_err(anyhow::Error::msg)?;
        let arena = ctx.buffers.sizes();
        anyhow::ensure!(
            !metadata.positions.is_null()
                && !metadata.slot.is_null()
                && !metadata.seq_len.is_null()
                && !metadata.block_table.is_null(),
            "native Qwen4 attention metadata pointers must be non-null"
        );
        anyhow::ensure!(
            !self.mrope_interleaved
                || (!metadata.positions_h.is_null() && !metadata.positions_w.is_null()),
            "native Qwen4 MRoPE metadata pointers must be non-null"
        );
        anyhow::ensure!(
            row_extents.hyper <= arena.hidden_states && row_extents.hyper <= arena.residual,
            "native Qwen4 attention hyper rows exceed the hidden/residual arenas"
        );
        anyhow::ensure!(
            row_extents.hyper <= arena.qkv_output,
            "native Qwen4 attention hyper logits exceed the QKV arena"
        );
        anyhow::ensure!(
            row_extents.core <= arena.norm_output,
            "native Qwen4 attention core rows exceed the norm arena"
        );
        let hyper_scratch = num_seqs
            .checked_mul(ctx.config.hc_lowrank)
            .and_then(|elements| elements.checked_mul(2))
            .and_then(|down_bytes| {
                num_seqs
                    .checked_mul(ctx.config.hc_count)
                    .and_then(|elements| elements.checked_mul(2))
                    .and_then(|inject_bytes| down_bytes.checked_add(inject_bytes))
            })
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention HC scratch overflow"))?;
        anyhow::ensure!(
            hyper_scratch <= arena.ssm_ba,
            "native Qwen4 attention HC scratch exceeds the SSM BA arena"
        );
        anyhow::ensure!(
            block_tables.iter().all(|table| !table.is_empty()),
            "native Qwen4 attention requires a non-empty KV block table for every row"
        );
        anyhow::ensure!(
            block_tables
                .iter()
                .all(|table| table.len() <= metadata.max_blocks_per_seq as usize),
            "native Qwen4 attention host block table exceeds its device row stride"
        );
        self.validate_yarn_multi_seq_controls()?;
        self.validate_qwen4_native_exact_qkv(num_seqs)?;

        let attn_hyper = self.qwen4_attn_hyper.as_ref().ok_or_else(|| {
            anyhow::anyhow!("native Qwen4 attention is missing its hyperconnection")
        })?;
        anyhow::ensure!(
            attn_hyper.residual_width() == ctx.config.residual_width(),
            "native Qwen4 attention hyperconnection geometry does not match model residual width"
        );
        let ordinary_nvfp4_o = self
            .o_weight
            .as_ref()
            .and_then(|weight| weight.as_fp8())
            .is_none()
            && self.o_dense_bf16.is_none()
            && !self.attn.o_proj.is_null();
        anyhow::ensure!(
            ordinary_nvfp4_o,
            "native Qwen4 attention requires the exact ordinary-NVFP4 O projection"
        );

        let block_size_host = kv_cache.block_size();
        anyhow::ensure!(
            block_size_host > 0,
            "native Qwen4 attention KV block size must be non-zero"
        );
        anyhow::ensure!(
            seq_lens
                .iter()
                .zip(block_tables)
                .all(|(&seq_len, table)| seq_len / block_size_host < table.len()),
            "native Qwen4 attention block table does not cover the next KV slot"
        );
        let block_size = u32::try_from(block_size_host)
            .map_err(|_| anyhow::anyhow!("native Qwen4 attention block size exceeds u32"))?;
        let max_seq_len_host = seq_lens
            .iter()
            .copied()
            .max()
            .and_then(|len| len.checked_add(1))
            .and_then(|len| u32::try_from(len).ok())
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention sequence extent overflow"))?;
        let c = MultiSeqCtx::new(
            self,
            ctx,
            hidden,
            residual,
            num_seqs,
            block_size,
            stream,
            max_seq_len_host,
        );
        let qkv_extent =
            c.n.checked_mul(c.per_seq_qkv)
                .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention QKV extent overflow"))?;
        let query_extent =
            c.n.checked_mul(c.q_dim as usize)
                .and_then(|elements| elements.checked_mul(c.bf16))
                .ok_or_else(|| anyhow::anyhow!("native Qwen4 attention query extent overflow"))?;
        anyhow::ensure!(
            qkv_extent <= arena.qkv_output,
            "native Qwen4 attention QKV rows exceed the QKV arena"
        );
        anyhow::ensure!(
            query_extent <= arena.ssm_qkvz && query_extent <= arena.attn_output,
            "native Qwen4 attention query rows exceed attention scratch"
        );
        anyhow::ensure!(
            row_extents.core <= arena.moe_output,
            "native Qwen4 attention O rows exceed the MoE-output arena"
        );

        // HC preparation stores each row's injection scales in its own 4H
        // residual tail.  That lets the attention core batch all rows before
        // one saved-scale injection without cross-sequence aliasing.
        let mixed = attn_hyper.prepare_batched_into(
            hidden,
            residual,
            num_seqs,
            ctx.buffers,
            None,
            ctx.gpu,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        debug_assert_eq!(mixed, c.normed);

        // Never use batch2/batch3 QKV here: C2-C4 run the exact M4 pair and
        // C5-C17 run the exact M17-capacity pair and C18-C32 the M32 pair,
        // all with C1 lane ownership.
        self.ms_phase_qkv_qwen4_native_exact(&c)?;
        self.ms_phase_rope(&c, metadata)?;
        self.ms_phase_cache_write_qwen4_native(&c, kv_cache, metadata)?;
        let attn_out = match qsa_routes {
            Some(routes) if routes.iter().any(|route| route.maintains_index()) => {
                let frame = NativeQsaFrame {
                    rows: c.n,
                    normed: c.normed,
                    qkv: c.qkv_buf,
                    per_seq_qkv: c.per_seq_qkv,
                    nq: c.nq,
                    nkv: c.nkv,
                    hd: c.hd,
                    bs: c.bs,
                };
                match self.qwen4_native_qsa_rows(
                    frame, routes, seq_lens, kv_cache, metadata, ctx, stream,
                )? {
                    Some(attn_out) => attn_out,
                    None => self.ms_phase_paged_decode(&c, kv_cache, metadata)?,
                }
            }
            _ => self.ms_phase_paged_decode(&c, kv_cache, metadata)?,
        };
        let o_out = self.ms_phase_o_proj(&c, attn_out)?;
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(o_out.0, num_seqs * ctx.config.hidden_size * 2, stream)?;
        }
        attn_hyper.inject_saved_batched(hidden, o_out, residual, num_seqs, ctx.gpu, stream)?;
        Ok(())
    }

    /// Finish one native attention layer with its MLP hyperconnection and
    /// MoE.  Keeping this as a separate hook makes the attention seam useful
    /// in isolation while giving the model loop an explicit, auditable proof
    /// that it did not omit half of the transformer block.
    pub(in crate::layers::qwen3_attention) fn decode_qwen4_native_target_mlp_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.preflight_qwen4_native_target_core(num_seqs, ctx)?;
        anyhow::ensure!(
            (2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&num_seqs),
            "native Qwen4 target MLP admits only C2-C{}",
            crate::layers::QWEN4_NATIVE_MAX_ROWS
        );
        anyhow::ensure!(
            ctx.config.is_qwen4_exp()
                && ctx.config.hc_count == 4
                && ctx.config.hidden_size == 2_560
                && ctx.config.residual_width() == 10_240
                && !ctx.config.use_fp32_residual(),
            "native Qwen4 target MLP requires the canonical BF16 Flash-Next geometry"
        );
        anyhow::ensure!(
            !ctx.graph_capture
                && ctx.config.tp_world_size <= 4
                && (ctx.config.ep_world_size <= 1
                    || ctx.config.ep_world_size == ctx.config.tp_world_size)
                && ctx.self_spec_sparse_draft.is_none()
                && ctx.ffn_defer.is_none(),
            "native Qwen4 target MLP is eager TP<=4 (+ matching EP) target-only"
        );
        anyhow::ensure!(
            matches!(self.ffn, crate::layers::FfnComponent::Moe(_)),
            "native Qwen4 target MLP requires the Flash-Next MoE"
        );
        let mlp_hyper = self.qwen4_mlp_hyper.as_ref().ok_or_else(|| {
            anyhow::anyhow!("native Qwen4 target MLP is missing its hyperconnection")
        })?;
        anyhow::ensure!(
            mlp_hyper.residual_width() == ctx.config.residual_width(),
            "native Qwen4 target MLP hyperconnection geometry mismatch"
        );
        let core_bytes = num_seqs
            .checked_mul(ctx.config.hidden_size)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 target MLP core extent overflow"))?;
        let hyper_bytes = num_seqs
            .checked_mul(ctx.config.residual_width())
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 target MLP hyper extent overflow"))?;
        anyhow::ensure!(
            hyper_bytes <= ctx.buffers.sizes().hidden_states
                && hyper_bytes <= ctx.buffers.sizes().residual
                && core_bytes <= ctx.buffers.sizes().norm_output
                && core_bytes <= ctx.buffers.sizes().moe_output,
            "native Qwen4 target MLP rows exceed the serving arenas"
        );

        let ffn_input = mlp_hyper.prepare_batched(
            hidden,
            residual,
            num_seqs,
            ctx.buffers,
            ctx.gpu,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        self.ffn
            .forward_qwen4_native_target_rows(ffn_input, num_seqs, ctx, stream)?;
        mlp_hyper.inject_saved_batched(
            hidden,
            ctx.buffers.moe_output(),
            residual,
            num_seqs,
            ctx.gpu,
            stream,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::validate_native_paged_decode_selectors;

    #[test]
    fn independent_native_rows_reject_shared_history_bf16_kgamma() {
        assert!(validate_native_paged_decode_selectors(false).is_ok());
        assert!(
            validate_native_paged_decode_selectors(true)
                .unwrap_err()
                .contains("shared-history")
        );
    }
}
