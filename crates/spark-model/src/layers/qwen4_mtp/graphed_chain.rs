// SPDX-License-Identifier: AGPL-3.0-only

//! Capture-safe batched MTP multi-step CUDA graph replay and coalesced FC/argmax helpers.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy};
use spark_runtime::kv_cache::PagedKvCache;

use super::graph_cache::{MtpBatchedChainLaunch, MtpGraphKey};
use super::init::Qwen4MtpHead;
use super::policy::{
    MTP_FC_COALESCED_MAX_HC_ROWS, mtp_lm_head_tc_pipe_enabled, should_use_batched_argmax_prob,
    should_use_mtp_fc_coalesced, should_use_mtp_lm_head_coalesced, should_use_mtp_lm_head_pipe,
};
use super::state::Qwen4MtpState;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;
use crate::speculative::ProposerState;

impl Qwen4MtpHead {
    /// Launch coalesced NVFP4 `fc_hidden` for `hc_rows` up to
    /// [`MTP_FC_COALESCED_MAX_HC_ROWS`] (`256`, i.e. `batch_rows <= 64`) by
    /// delegating directly to `qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm`,
    /// which handles `1..=256` rows directly via SM-aware `w4a16_gemm_coalesced*`
    /// tiles.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_coalesced_mtp_fc_hidden(
        &self,
        gpu: &dyn GpuBackend,
        normed_hidden: DevicePtr,
        hidden: DevicePtr,
        hc_rows: usize,
        h: usize,
        stream: u64,
    ) -> Result<bool> {
        if hc_rows > MTP_FC_COALESCED_MAX_HC_ROWS {
            return Ok(false);
        }
        crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
            gpu,
            normed_hidden,
            &self.fc_hidden,
            hidden,
            hc_rows,
            h,
            h,
            stream,
        )
    }

    /// Launch batched 0-ULP `argmax_prob_batched_bf16` (`grid = [rows, 1, 1]`,
    /// `block = [1024, 1, 1]`) when loaded and `rows > 1`, falling back to
    /// sequential per-row `argmax_prob_bf16` launches otherwise.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_argmax_prob_batch(
        &self,
        gpu: &dyn GpuBackend,
        logits: DevicePtr,
        out_ids: DevicePtr,
        out_probs: DevicePtr,
        rows: usize,
        vocab: usize,
        stream: u64,
    ) -> Result<()> {
        if should_use_batched_argmax_prob(rows, self.argmax_prob_batched_k.0 != 0) {
            spark_runtime::kernel_args::KernelLaunch::new(gpu, self.argmax_prob_batched_k)
                .grid([rows as u32, 1, 1])
                .block([1024, 1, 1])
                .arg_ptr(logits)
                .arg_ptr(out_ids)
                .arg_ptr(out_probs)
                .arg_u32(vocab as u32)
                .arg_u32(vocab as u32)
                .launch(stream)?;
        } else {
            for row in 0..rows {
                spark_runtime::kernel_args::KernelLaunch::new(gpu, self.argmax_prob_k)
                    .grid([1, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(logits.offset(row * vocab * 2))
                    .arg_ptr(out_ids.offset(row * 4))
                    .arg_ptr(out_probs.offset(row * 4))
                    .arg_u32(vocab as u32)
                    .launch(stream)?;
            }
        }
        Ok(())
    }

    /// Capture-safe multi-step batched MTP proposer body: reads all per-step
    /// inputs (`last_tokens`, `target_hiddens` pointer table, `positions`,
    /// `kv_slots`, `seq_lens`, `block_tables`) strictly from pre-staged device
    /// memory (`spec.meta_base`), executes all `spec.step_count` steps (dual
    /// norm -> FC -> MTP layer -> final mixer -> LM head -> batched argmax/prob)
    /// with zero `gpu.alloc()` and zero host-value kernel args, leaving results
    /// in `spec.argmax_base` (and `spec.prob_base`) for ONE trailing D2H copy.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_batched_mtp_graph_body(
        &self,
        spec: &MtpBatchedChainLaunch<'_>,
        active_lanes: Option<&[usize]>,
        states: &mut [&mut dyn ProposerState],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
        capturing: bool,
    ) -> Result<()> {
        let b = spec.batch_size;
        let h = ctx.config.hidden_size;
        let r = ctx.config.residual_width();
        let hc = ctx.config.hc_count;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;

        let embed = ctx.buffers.ssm_qkvz();
        let normed_embed = ctx.buffers.ssm_deinterleaved();
        let normed_hidden = ctx.buffers.residual();
        let embed_proj = ctx.buffers.qkv_output();
        let hidden = ctx.buffers.hidden_states();
        let logits = ctx.buffers.logits();

        let tc_dense_scope =
            crate::layers::qwen4_hyper::batched_verify_tc_dense_active(b, capturing);
        let _tc_dense_guard =
            crate::layers::qwen4_hyper::BatchedVerifyTcDenseGuard::enter(tc_dense_scope);
        let coalesced_lm_head_eligible = should_use_mtp_lm_head_coalesced(
            crate::layers::qwen4_hyper::batched_verify_tc_component_active(
                crate::layers::qwen4_hyper::BatchedVerifyTcComponent::LmHead,
                b,
                capturing,
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

        let mut step_seq_lens = vec![0usize; b];
        for step in 0..spec.step_count {
            let step_meta = spec.meta_base.offset(step * spec.stride);
            if step == 0 {
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.embed_from_argmax_k)
                    .grid([(h as u32).div_ceil(256), b as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(spec.meta_base.offset(spec.last_tok_off))
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_ptr(embed)
                    .arg_ptr(DevicePtr(0))
                    .arg_u32(h as u32)
                    .launch(stream)?;
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.pre_fc_dual_norm_k)
                    .grid([b as u32, 2, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(embed)
                    .arg_ptr(self.pre_fc_norm_embedding.weight)
                    .arg_ptr(normed_embed)
                    .arg_ptr(DevicePtr::NULL)
                    .arg_ptr(self.pre_fc_norm_hidden.weight)
                    .arg_ptr(normed_hidden)
                    .arg_u32(h as u32)
                    .arg_u32(r as u32)
                    .arg_f32(eps)
                    .arg_ptr(spec.meta_base.offset(spec.hidden_ptrs_off))
                    .launch(stream)?;
            } else {
                let prev_argmax = spec.argmax_base.offset((step - 1) * b * 4);
                let token_out = spec
                    .token_out_base
                    .map_or(DevicePtr(0), |base| base.offset((step - 1) * b * 4));
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.embed_from_argmax_k)
                    .grid([(h as u32).div_ceil(256), b as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(prev_argmax)
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_ptr(embed)
                    .arg_ptr(token_out)
                    .arg_u32(h as u32)
                    .launch(stream)?;
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
                    .arg_ptr(DevicePtr::NULL)
                    .launch(stream)?;
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

            let metadata = AttnMetadataDev {
                qwen4_qsa_required: self.mtp_qsa_active(),
                positions: step_meta,
                positions_h: step_meta,
                positions_w: step_meta,
                slot: step_meta.offset(spec.slot_off),
                seq_len: step_meta.offset(spec.seq_len_off),
                block_table: step_meta.offset(spec.bt_off),
                max_blocks_per_seq: spec.bt_blocks_cap as u32,
                num_seqs: b as u32,
            };
            let layer_ctx = ForwardContext {
                attn_metadata: Some(metadata),
                comm: None,
                graph_capture: capturing,
                ..*ctx
            };
            for (dst, &base_len) in step_seq_lens.iter_mut().zip(spec.base_seq_lens.iter()) {
                *dst = base_len + step;
            }
            {
                let mut typed_states: Vec<&mut Qwen4MtpState> = Vec::with_capacity(b);
                if let Some(lanes) = active_lanes {
                    let mut cursor = 0usize;
                    for (orig_s, st) in states.iter_mut().enumerate() {
                        if cursor < lanes.len() && lanes[cursor] == orig_s {
                            typed_states.push(
                                st.as_any_mut()
                                    .downcast_mut::<Qwen4MtpState>()
                                    .expect("validated above"),
                            );
                            cursor += 1;
                        }
                    }
                } else {
                    for st in states.iter_mut() {
                        typed_states.push(
                            st.as_any_mut()
                                .downcast_mut::<Qwen4MtpState>()
                                .expect("validated above"),
                        );
                    }
                }
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
                    kv_cache,
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
                crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                    ctx.gpu,
                    sample_hiddens,
                    &self.lm_head_nvfp4,
                    logits,
                    b,
                    spec.vocab,
                    h,
                    stream,
                )?
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
                        spec.vocab as u32,
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
                        b as u32,
                        spec.vocab as u32,
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

            let step_argmax = spec.argmax_base.offset(step * b * 4);
            let step_prob = spec.prob_base.offset(step * b * 4);
            if spec.want_prob {
                self.launch_argmax_prob_batch(
                    ctx.gpu,
                    logits,
                    step_argmax,
                    step_prob,
                    b,
                    spec.vocab,
                    stream,
                )?;
            } else if self.argmax_rows_k.0 != 0 {
                ops::argmax_bf16_rows(
                    ctx.gpu,
                    self.argmax_rows_k,
                    logits,
                    step_argmax,
                    spec.vocab as u32,
                    spec.vocab as u32,
                    b as u32,
                    stream,
                )?;
            } else {
                for s in 0..b {
                    ops::argmax_bf16(
                        ctx.gpu,
                        self.argmax_k,
                        logits.offset(s * spec.vocab * 2),
                        step_argmax.offset(s * 4),
                        spec.vocab as u32,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Execute a uniform batched MTP multi-step chain via the cached CUDA graph
    /// (`ATLAS_QWEN4_MTP_GRAPH`), warming up lazily on hit 1, capturing on hit 2
    /// with fail-closed fallback to eager on any capture error, and replaying in
    /// 1 graph launch on hits 3+. Returns `Ok(true)` if the chain was executed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_batched_mtp_chain_graphed(
        &self,
        spec: &MtpBatchedChainLaunch<'_>,
        active_lanes: Option<&[usize]>,
        states: &mut [&mut dyn ProposerState],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let b = spec.batch_size;
        let h = ctx.config.hidden_size;
        let hc = ctx.config.hc_count;
        let tc_dense_scope = crate::layers::qwen4_hyper::batched_verify_tc_dense_active(b, true);
        let coalesced_lm_head = should_use_mtp_lm_head_coalesced(
            crate::layers::qwen4_hyper::batched_verify_tc_component_active(
                crate::layers::qwen4_hyper::BatchedVerifyTcComponent::LmHead,
                b,
                true,
            ),
            b,
            h,
        );
        let coalesced_fc = should_use_mtp_fc_coalesced(
            tc_dense_scope,
            b,
            hc,
            h,
            ctx.buffers.sizes().qkv_output,
            self.residual_add_broadcast_hc_k.0 != 0,
        );
        let key = MtpGraphKey {
            batch_size: b,
            step_count: spec.step_count,
            want_prob: spec.want_prob,
            bt_blocks_cap: spec.bt_blocks_cap as u32,
            vocab: spec.vocab,
            write_step_token_out: spec.token_out_base.is_some(),
            tc_dense_scope,
            coalesced_fc,
            coalesced_lm_head,
            scratch_ptr: ctx.buffers.scratch().0,
            hidden_ptr: ctx.buffers.hidden_states().0,
            residual_ptr: ctx.buffers.residual().0,
            logits_ptr: ctx.buffers.logits().0,
            k_pool_ptr: kv_cache.k_pool_ptr(0).0,
            v_pool_ptr: kv_cache.v_pool_ptr(0).0,
            stream,
        };

        let (cached_graph, is_poisoned, is_warmed) = {
            let cache = self.graph_cache.lock();
            (
                cache.get(&key),
                cache.is_poisoned(&key),
                cache.warmed.contains(&key),
            )
        };
        if let Some(graph) = cached_graph {
            ctx.gpu.launch_graph(graph, stream)?;
            return Ok(true);
        }
        if is_poisoned {
            return Ok(false);
        }

        crate::layers::qwen4_fast_proj::warm_coalesced_nvfp4_kernels(ctx.gpu);
        if !is_warmed {
            self.launch_batched_mtp_graph_body(
                spec,
                active_lanes,
                states,
                kv_cache,
                ctx,
                stream,
                false,
            )?;
            self.graph_cache.lock().mark_warmed(key);
            return Ok(true);
        }

        let captured = (|| -> Result<GraphHandle> {
            let capture = crate::model::trait_impl::capture_guard::CaptureGuard::begin(
                ctx.gpu,
                stream,
                "qwen4_mtp_graph",
            )?;
            self.launch_batched_mtp_graph_body(
                spec,
                active_lanes,
                states,
                kv_cache,
                ctx,
                stream,
                true,
            )?;
            capture.end()
        })();
        match captured {
            Ok(graph) if graph.0 != 0 => {
                self.graph_cache.lock().insert(ctx.gpu, key, graph);
                ctx.gpu.launch_graph(graph, stream)?;
                static GRAPH_ENGAGED: std::sync::Once = std::sync::Once::new();
                GRAPH_ENGAGED.call_once(|| {
                    tracing::info!(
                        batch_size = b,
                        step_count = spec.step_count,
                        want_prob = spec.want_prob,
                        bt_blocks_cap = spec.bt_blocks_cap,
                        vocab = spec.vocab,
                        tc_dense_scope,
                        coalesced_fc,
                        coalesced_lm_head,
                        "QWEN4_MTP_GRAPH_ENGAGED"
                    );
                });
                Ok(true)
            }
            Ok(_) => {
                // Mock/no-op capture backends executed the body eagerly during capture.
                Ok(true)
            }
            Err(capture_err) => {
                self.graph_cache.lock().mark_poisoned(key);
                static CAPTURE_FAIL_WARN: std::sync::Once = std::sync::Once::new();
                CAPTURE_FAIL_WARN.call_once(|| {
                    tracing::warn!(
                        batch_size = b,
                        step_count = spec.step_count,
                        want_prob = spec.want_prob,
                        bt_blocks_cap = spec.bt_blocks_cap,
                        error = %capture_err,
                        "QWEN4_MTP_GRAPH capture failed; falling back to eager"
                    );
                });
                self.launch_batched_mtp_graph_body(
                    spec,
                    active_lanes,
                    states,
                    kv_cache,
                    ctx,
                    stream,
                    false,
                )?;
                Ok(true)
            }
        }
    }

    /// Stage `copies` into the MTP head's dedicated page-locked ring and enqueue
    /// async stream-ordered `copy_h2d_pinned_async` transfers with 0 stream drains.
    /// Falls back to `gpu.copy_h2d_group_on_stream` if disabled (`ATLAS_QWEN4_MTP_PINNED_H2D=0`)
    /// or if a payload exceeds the ring capacity.
    pub(super) fn copy_h2d_group_fast(
        &self,
        gpu: &dyn GpuBackend,
        copies: &[HostToDeviceCopy<'_>],
        stream: u64,
    ) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let enabled = *ENABLED.get_or_init(|| {
            std::env::var("ATLAS_QWEN4_MTP_PINNED_H2D").ok().as_deref() != Some("0")
        });
        if enabled {
            let mut ring_guard = self.pinned_h2d_ring.lock();
            if let Some(ring) = ring_guard.as_mut()
                && ring.copy_h2d_group(gpu, copies, stream)?
            {
                return Ok(());
            }
        }
        gpu.copy_h2d_group_on_stream(copies, stream)
    }
}
