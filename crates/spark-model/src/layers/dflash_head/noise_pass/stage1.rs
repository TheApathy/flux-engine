// SPDX-License-Identifier: AGPL-3.0-only

//! Two-stage CUDA-graphed DFlash noise pass coordinator and Stage-1 incremental context KV append.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::super::{
    BlockDiffusionDraftHead, DflashLayerQuantWeights, DflashProposerState, DraftRowLayout,
};
use super::eager::NoisePassArgs;
use super::graph_state::{
    DflashLayerQuantMode, DflashNoiseGraphMeta, NoiseGraphCaptureGuard, RingMetaDevicePtrs,
    invalidate_dstate_ctx_kv_ranges, noise_graph_head_state, read_noise_positions_from_pinned,
};
use super::lm_head_path::{LmHeadPath, resolve_lm_head_path};
use super::policy::{
    CtxCacheInvalidationReason, CtxKvCacheRepr, DFLASH_SPLIT_KV_H128_MAX_SPLITS,
    DflashNoiseGraphKey, LmHeadNvfp4Mode, NoiseGraphRuntimeKey, REJECTION_SLOT_STRIDE,
    STAGE1_GRAPH_MAX_ROWS, Stage2CtxAttnMode, all_layers_cover_old_fc_prefix,
    dflash_noise_graph_enabled, dflash_split_kv_h128_workspace_bytes, dflash_stage1_graph_enabled,
    lm_head_path_tag, noise_graph_stochastic_draft_sampling_active, noise_graph_swa_enabled,
    plan_ctx_kv_cache_transition, plan_stage1_ring_append, resolve_c1_lm_head_nvfp4_mode,
    resolve_dflash_split_kv_splits_for_sm, resolve_dflash2_selector_topk,
    resolve_rejection_seq_staging, resolve_rejection_shortlist_k,
    resolve_stage2_ctx_attn_mode_for_sm,
};
use crate::layer::ForwardContext;

#[derive(Clone, Copy)]
pub(super) struct NoiseGraphDebugFlags {
    pub(super) debug_dump: bool,
    pub(super) force_noise_pattern: bool,
    pub(super) dump_all_layers: bool,
    pub(super) kprofile: bool,
}

#[derive(Clone, Copy)]
pub(super) struct SplitKvH128Plan {
    pub(super) split_kernel: KernelHandle,
    pub(super) reduce_kernel: KernelHandle,
    pub(super) workspace: DevicePtr,
    pub(super) num_splits: u32,
}

#[derive(Clone, Copy)]
pub(super) enum Stage2CtxAttnDispatch {
    Linear {
        kernel: KernelHandle,
        split_kv: Option<SplitKvH128Plan>,
    },
    CircularRingH128 {
        kernel: KernelHandle,
        ring_meta: RingMetaDevicePtrs,
        ring_cap: u32,
        split_kv: Option<SplitKvH128Plan>,
    },
}

#[derive(Clone, Copy)]
pub(super) enum Stage2LayerCtxAttn {
    Linear {
        kernel: KernelHandle,
        ctx_k: DevicePtr,
        ctx_v: DevicePtr,
        eff_ctx_dev: DevicePtr,
        split_kv: Option<SplitKvH128Plan>,
    },
    CircularRingH128 {
        kernel: KernelHandle,
        k_ctx_ptr_dev: DevicePtr,
        v_ctx_ptr_dev: DevicePtr,
        eff_ctx_dev: DevicePtr,
        ring_start_dev: DevicePtr,
        ring_cap: u32,
        split_kv: Option<SplitKvH128Plan>,
    },
}

impl BlockDiffusionDraftHead {
    pub(in super::super) fn layers_have_dflash2_conv(&self) -> bool {
        self.layers.iter().any(|layer| match layer {
            DflashLayerQuantWeights::Bf16(l) => l.attention_conv.is_some() || l.mlp_conv.is_some(),
            DflashLayerQuantWeights::Fp8(l) => l.attention_conv.is_some() || l.mlp_conv.is_some(),
            DflashLayerQuantWeights::Nvfp4(l) => l.attention_conv.is_some() || l.mlp_conv.is_some(),
        })
    }

    fn noise_graph_any_layer_swa(&self) -> bool {
        noise_graph_swa_enabled()
            && self
                .layer_window_sizes
                .iter()
                .take(self.layers.len())
                .any(|&w| w > 0)
    }

    fn noise_graph_structural_ok(
        &self,
        dstate: &DflashProposerState,
        needed_start: usize,
        eff_ctx: usize,
        gamma_eff: usize,
        flags: NoiseGraphDebugFlags,
    ) -> bool {
        let row_layout = DraftRowLayout::for_family(self.checkpoint_family, gamma_eff);
        let noise_rows = row_layout.query_rows;
        let sampled_active = noise_graph_stochastic_draft_sampling_active(
            self.draft_sample_method,
            dstate.active_sampling_request.as_ref(),
        );
        let w4a4_ok = self.w4a4_shadow.is_none()
            || (noise_rows == 16 && self.gamma == 15 && self.physical_verify_k == 16);
        let conv_ok = !self.layers_have_dflash2_conv()
            || (self.kernels.dflash2_conv_prepare.0 != 0
                && self.kernels.dflash2_conv_finish.0 != 0
                && !self.scratch.conv_dyn.is_null()
                && !self.scratch.conv_dyn1.is_null()
                && !self.scratch.conv_out.is_null());
        let num_layers = self.layers.len();
        let caches_ready = num_layers > 0
            && dstate.ctx_k_cache.len() == num_layers
            && dstate.ctx_v_cache.len() == num_layers
            && dstate.cache_k_start.len() == num_layers
            && dstate.cache_k_end.len() == num_layers
            && dstate.cache_v_start.len() == num_layers
            && dstate.cache_v_end.len() == num_layers
            && (eff_ctx == 0
                || (!dstate.ctx_k_cache.iter().any(|p| p.is_null())
                    && !dstate.ctx_v_cache.iter().any(|p| p.is_null())));
        let window_ok = self.ctx_window > 0
            && u32::try_from(self.ctx_window).is_ok()
            && eff_ctx <= self.ctx_window
            && needed_start.checked_add(eff_ctx).is_some()
            && (eff_ctx == 0 || needed_start >= dstate.ctx_valid_from_abs);
        matches!(self.head_dim, 128 | 256)
            && window_ok
            && (1..=DflashNoiseGraphMeta::MAX_NOISE_POSITIONS).contains(&noise_rows)
            && !flags.debug_dump
            && !flags.force_noise_pattern
            && !flags.dump_all_layers
            && !flags.kprofile
            && self.dspine.is_none()
            && !sampled_active
            && caches_ready
            && self.num_kv_heads > 0
            && self.num_q_heads > 0
            && self.num_q_heads.is_multiple_of(self.num_kv_heads)
            && w4a4_ok
            && conv_ok
    }

    /// Whether the 2-stage noise graph will keep all cached context KV rows
    /// (`[needed_start .. needed_start + old_fc_count)`) in circular `PostRope`
    /// representation so `scratch.fc_proj[0..old_fc_count)` is completely dead
    /// and `forward_block` can skip copying `old_fc_count` rows from `ctx_fc_cache`.
    pub(in super::super) fn noise_graph_can_skip_old_fc_copy(
        &self,
        dstate: &DflashProposerState,
        policy: &super::super::proposal_policy::ProposalPolicy,
        needed_start: usize,
        old_fc_count: usize,
        eff_ctx: usize,
        gamma_eff: usize,
    ) -> bool {
        if !dflash_noise_graph_enabled() || old_fc_count == 0 || policy.denoise_steps != 1 {
            return false;
        }
        if !self.noise_graph_structural_ok(
            dstate,
            needed_start,
            eff_ctx,
            gamma_eff,
            NoiseGraphDebugFlags {
                debug_dump: policy.debug_dump,
                force_noise_pattern: policy.force_noise_pattern,
                dump_all_layers: policy.dump_all_layers,
                kprofile: false,
            },
        ) {
            return false;
        }
        if dstate.ctx_kv_repr != Some(CtxKvCacheRepr::PostRope)
            || !all_layers_cover_old_fc_prefix(
                needed_start,
                eff_ctx,
                old_fc_count,
                self.ctx_window,
                &dstate.cache_k_start,
                &dstate.cache_k_end,
                &dstate.cache_v_start,
                &dstate.cache_v_end,
            )
        {
            return false;
        }
        let head_state_arc = noise_graph_head_state(self as *const Self as usize);
        let has_linear_kernel = {
            let state = head_state_arc
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.split_ctx_kernel.is_some_and(|k| k.0 != 0)
        };
        let has_ring_h128_kernel =
            self.head_dim == 128 && self.kernels.split_ctx_h128_batched.0 != 0;
        resolve_stage2_ctx_attn_mode_for_sm(
            self.head_dim,
            needed_start,
            eff_ctx,
            self.ctx_window,
            has_linear_kernel,
            has_ring_h128_kernel,
            self.noise_graph_any_layer_swa(),
            atlas_core::device::active_sm_count(),
        )
        .is_some()
    }

    /// Attempt to run the 2-stage CUDA-graphed DFlash2 noise-block path when
    /// `ATLAS_DFLASH_NOISE_GRAPH=1` and all admission invariants hold.
    ///
    /// Returns `Ok(true)` if the pass was executed via Stage-1 + Stage-2, or
    /// `Ok(false)` to fall back cleanly to the eager path (after invalidating
    /// any cached `PostRope` context K rows if transitioning back to eager).
    pub(super) fn try_run_two_stage_noise_graph(
        &self,
        a: &NoisePassArgs,
        committed: &[Option<u32>],
        loop_pass: usize,
        gumbel_spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
        rejection_slot: Option<usize>,
        ctx: &ForwardContext,
        dstate: &mut DflashProposerState,
    ) -> Result<bool> {
        if !dflash_noise_graph_enabled() {
            // A sequence previously processed by `propose_batch_dflash2` leaves
            // circular PostRope K in `ctx_k_cache`. Invalidate before eager
            // `forward_block_layer` runs so it never applies k_norm/RoPE twice.
            if dstate.ctx_kv_repr == Some(CtxKvCacheRepr::PostRope) {
                let cache_end = dstate
                    .cache_k_end
                    .iter()
                    .chain(dstate.cache_v_end.iter())
                    .copied()
                    .max()
                    .unwrap_or(0);
                if cache_end > 0 {
                    invalidate_dstate_ctx_kv_ranges(dstate);
                }
                dstate.ctx_kv_repr = Some(CtxKvCacheRepr::PreRope);
            }
            return Ok(false);
        }
        let NoisePassArgs {
            last_token,
            eff_ctx,
            gamma_eff,
            mask_id,
            needed_start,
            stream,
            debug_dump,
            force_noise_pattern,
            lm_head_nvfp4,
            dump_all_layers,
            kprofile,
            ..
        } = *a;
        let row_layout = DraftRowLayout::for_family(self.checkpoint_family, gamma_eff);
        let noise_rows = row_layout.query_rows;
        let structural_ok = loop_pass == 0
            && committed.is_empty()
            && self.noise_graph_structural_ok(
                dstate,
                needed_start,
                eff_ctx,
                gamma_eff,
                NoiseGraphDebugFlags {
                    debug_dump,
                    force_noise_pattern,
                    dump_all_layers,
                    kprofile,
                },
            );
        let has_ring_h128_kernel =
            self.head_dim == 128 && self.kernels.split_ctx_h128_batched.0 != 0;
        let any_layer_swa = self.noise_graph_any_layer_swa();

        let head_state_arc = noise_graph_head_state(self as *const Self as usize);
        let (split_ctx_kernel, attn_mode) = {
            let mut state = head_state_arc
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let linear_kernel = if structural_ok && (needed_start == 0 || !has_ring_h128_kernel) {
                state.resolve_split_ctx_kernel(ctx.gpu, self.head_dim)
            } else {
                state.split_ctx_kernel.unwrap_or(KernelHandle(0))
            };
            let mode = if structural_ok {
                resolve_stage2_ctx_attn_mode_for_sm(
                    self.head_dim,
                    needed_start,
                    eff_ctx,
                    self.ctx_window,
                    linear_kernel.0 != 0,
                    has_ring_h128_kernel,
                    any_layer_swa,
                    atlas_core::device::active_sm_count(),
                )
            } else {
                None
            };
            let eligible = mode.is_some();
            let cache_end = dstate
                .cache_k_end
                .iter()
                .chain(dstate.cache_v_end.iter())
                .copied()
                .max()
                .unwrap_or(0);
            let prev_repr = dstate.ctx_kv_repr;
            let (next_repr, invalidation) =
                plan_ctx_kv_cache_transition(prev_repr, eligible, needed_start, cache_end);
            if invalidation != CtxCacheInvalidationReason::None {
                tracing::debug!(
                    "DFlash noise graph cache invalidation: reason={invalidation:?} \
                     needed_start={needed_start} eff_ctx={eff_ctx} cache_end={cache_end}"
                );
                invalidate_dstate_ctx_kv_ranges(dstate);
            }
            dstate.ctx_kv_repr = Some(next_repr);
            (linear_kernel, mode)
        };
        let Some(attn_mode) = attn_mode else {
            return Ok(false);
        };

        let lm_head_path = resolve_lm_head_path(
            lm_head_nvfp4,
            self.lm_head_shared_t.is_some(),
            self.kernels.w4a16_gemm_t_m32_n64.0 != 0,
            self.lm_head_shared_fp8.is_some(),
        )
        .map_err(anyhow::Error::msg)?;
        let h = self.hidden_size as u32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;

        // ── Stage 1 (Eager): incremental post-k_norm/post-RoPE circular KV append ──
        if eff_ctx > 0 {
            self.stage1_append_post_rope_ctx_kv(
                needed_start,
                eff_ctx,
                h,
                kv_dim,
                stream,
                ctx,
                dstate,
            )?;
        }

        // ── Stage 1 (Eager): async pinned H2D metadata upload ──
        let noise_positions =
            read_noise_positions_from_pinned(&dstate.pos_pinned, eff_ctx, noise_rows)?;
        let meta =
            DflashNoiseGraphMeta::try_new(eff_ctx, last_token, &noise_positions, self.ctx_window)?;
        let ring_start = (needed_start % self.ctx_window) as u32;
        let lm_stride = self.target_vocab_size.min(self.vocab_size);
        let rejection_env = spark_runtime::sampler::gumbel::rejection_sampling_enabled();
        let rejection_slot_valid = rejection_slot.filter(|&slot| {
            gamma_eff > 0
                && gamma_eff <= REJECTION_SLOT_STRIDE
                && slot
                    .checked_mul(REJECTION_SLOT_STRIDE)
                    .and_then(|base| base.checked_add(gamma_eff))
                    .is_some_and(|end| end <= self.scratch.shared_q_row_cap)
        });
        let use_rejection_graph_selector = rejection_env
            && self.selector.is_some()
            && self.markov.is_none()
            && self.rejection_selector_ready()
            && self.scratch.shared_q_stride
                == crate::model::trait_impl::device_topk::REJECTION_Q_STRIDE
            && rejection_slot_valid.is_some();
        let (staged_spec, q_row_base_upload, valid_q_record) =
            if use_rejection_graph_selector && let Some(slot) = rejection_slot_valid {
                let temp_scale = spark_runtime::sampler::gumbel::rejection_draft_temp_scale();
                let (eff_spec, spec_row_base, record) = resolve_rejection_seq_staging(
                    gumbel_spec,
                    Some(slot),
                    gamma_eff,
                    self.scratch.shared_q_row_cap,
                    temp_scale,
                );
                let q_row_base = if eff_spec.is_some() {
                    spec_row_base
                } else {
                    (slot * REJECTION_SLOT_STRIDE) as u32
                };
                (
                    eff_spec,
                    Some((q_row_base, self.scratch.q_row_bases)),
                    record,
                )
            } else if rejection_env {
                (None, None, None)
            } else {
                (gumbel_spec, None, None)
            };
        let prefer_tc_lm_head =
            resolve_c1_lm_head_nvfp4_mode(staged_spec.is_some()) == LmHeadNvfp4Mode::CoalescedTc;
        let use_ngram_graph_selector = self.selector.is_some()
            && super::super::ngram_bonus::admit_c1_ngram_selector(
                super::super::ngram_bonus::C1NgramAdmission {
                    ngram_enabled: super::super::ngram_bonus::ngram_bonus_enabled(),
                    ngram_ready: self.ngram_selector_ready(),
                    use_rejection_selector: use_rejection_graph_selector,
                    is_probabilistic_sampled: false,
                    markov_active: self.markov.is_some(),
                    probabilistic_method: self.draft_sample_method
                        == crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic,
                    raw_gumbel_spec: gumbel_spec,
                },
            );
        let use_k32_graph_selector = !use_rejection_graph_selector
            && !use_ngram_graph_selector
            && self.selector.is_some()
            && self.markov.is_none()
            && self.gumbel_selector_k32_ready();
        let use_gumbel_graph_selector = !use_rejection_graph_selector
            && !use_ngram_graph_selector
            && (use_k32_graph_selector
                || (self.selector.is_some()
                    && self.markov.is_none()
                    && self.kernels.dflash2_selector_walk_gumbel_batched.0 != 0
                    && self.gumbel_selector_scratch_ready()));
        let (shortlist_k, use_radix_topk, use_k32_selector) = if use_rejection_graph_selector {
            let (k, use_radix) = resolve_rejection_shortlist_k(
                lm_stride,
                self.kernels.topk_bf16_radix.0 != 0,
                self.scratch.shared_q_stride,
            );
            (k, use_radix, false)
        } else if let Some(sel) = &self.selector {
            resolve_dflash2_selector_topk(
                sel.top_k,
                lm_stride,
                staged_spec.is_some(),
                self.kernels.topk_bf16_radix.0 != 0,
                use_k32_graph_selector,
            )
        } else {
            (0, false, false)
        };
        let logical_key = DflashNoiseGraphKey {
            gamma_eff,
            quant_mode: DflashLayerQuantMode::from_layers(&self.layers),
            markov_active: self.markov.is_some(),
        };

        let desired_split_kv = resolve_dflash_split_kv_splits_for_sm(
            self.head_dim,
            eff_ctx,
            self.num_q_heads,
            1,
            atlas_core::device::active_sm_count(),
        );
        let (bundle_ptrs, split_kv_plan, runtime_key, cached_graph) = {
            let mut state = head_state_arc
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let ring_upload = match attn_mode {
                Stage2CtxAttnMode::CircularRingH128 => Some((
                    ring_start,
                    dstate.ctx_k_cache.as_slice(),
                    dstate.ctx_v_cache.as_slice(),
                )),
                Stage2CtxAttnMode::Linear => None,
            };
            let q_row_base_val = q_row_base_upload.map_or(0u32, |(row_base, _)| row_base);
            let bundle_ptrs = state.upload_c1_stage2_bundle(
                ctx.gpu,
                &meta,
                mask_id,
                ring_upload,
                staged_spec,
                q_row_base_val,
                self.layers.len(),
                stream,
            )?;
            if use_ngram_graph_selector {
                state.upload_ngram_meta_c1(
                    ctx.gpu,
                    &dstate.pld_tokens,
                    last_token,
                    self.scratch.ngram_tables_dev,
                    self.scratch.ngram_prev3_dev,
                    None,
                    stream,
                )?;
            }
            let split_kv_plan = if desired_split_kv > 1
                && let Some((split_kernel, reduce_kernel)) =
                    state.resolve_split_kv_h128_kernels(ctx.gpu, attn_mode)
                && let Some(needed_bytes) = dflash_split_kv_h128_workspace_bytes(
                    1,
                    desired_split_kv,
                    self.num_q_heads,
                    noise_rows,
                )
                && let Some(alloc_bytes) = dflash_split_kv_h128_workspace_bytes(
                    1,
                    DFLASH_SPLIT_KV_H128_MAX_SPLITS,
                    self.num_q_heads,
                    DflashNoiseGraphMeta::MAX_NOISE_POSITIONS,
                )
                && let Some(workspace) =
                    state.ensure_split_kv_workspace(ctx.gpu, alloc_bytes, needed_bytes)?
            {
                Some(SplitKvH128Plan {
                    split_kernel,
                    reduce_kernel,
                    workspace,
                    num_splits: desired_split_kv,
                })
            } else {
                None
            };
            let runtime_key = NoiseGraphRuntimeKey {
                logical: logical_key,
                attn_mode,
                split_kv_splits: split_kv_plan.map_or(1u8, |p| p.num_splits as u8),
                lm_head_path_tag: lm_head_path_tag(lm_head_path),
                prefer_tc_lm_head,
                rejection_mode: use_rejection_graph_selector,
                ngram_mode: use_ngram_graph_selector,
                shortlist_k: shortlist_k as u8,
                mask_id,
                stream,
                ctx_kv_ptrs: match bundle_ptrs.ring_meta_ptrs {
                    Some(ptrs) => vec![(ptrs.base.0, self.ctx_window as u64)],
                    None => dstate
                        .ctx_k_cache
                        .iter()
                        .copied()
                        .zip(dstate.ctx_v_cache.iter().copied())
                        .map(|(k, v)| (k.0, v.0))
                        .collect(),
                },
            };
            let cached = state.graphs.get(&runtime_key).copied();
            (bundle_ptrs, split_kv_plan, runtime_key, cached)
        };
        let meta_dev = bundle_ptrs.meta_dev;
        let mask_dev = bundle_ptrs.mask_dev;
        let ctx_attn_dispatch = match attn_mode {
            Stage2CtxAttnMode::Linear => Stage2CtxAttnDispatch::Linear {
                kernel: split_ctx_kernel,
                split_kv: split_kv_plan,
            },
            Stage2CtxAttnMode::CircularRingH128 => Stage2CtxAttnDispatch::CircularRingH128 {
                kernel: self.kernels.split_ctx_h128_batched,
                ring_meta: bundle_ptrs
                    .ring_meta_ptrs
                    .context("DFlash C=1 ring metadata pointers missing")?,
                ring_cap: self.ctx_window as u32,
                split_kv: split_kv_plan,
            },
        };

        // ── Stage 2: fixed-shape m_rows noise pass (CUDA graph replay/capture) ──
        if let Some(graph) = cached_graph {
            ctx.gpu.launch_graph(graph, stream)?;
        } else if stream != 0 {
            crate::layers::qwen4_fast_proj::warm_coalesced_nvfp4_kernels(ctx.gpu);
            let bf16_lm_head_fallback = (lm_head_path == LmHeadPath::Bf16
                && self.lm_head_shared_nvfp4_exact.is_none())
            .then_some((self.scratch.logits, lm_stride as u32));
            self.prewarm_stage2_cublas(
                noise_rows as u32,
                gamma_eff as u32,
                h,
                (self.num_q_heads * self.head_dim) as u32,
                kv_dim,
                self.intermediate_size as u32,
                bf16_lm_head_fallback,
                stream,
            )?;
            let capture = NoiseGraphCaptureGuard::begin(ctx.gpu, stream)?;
            self.launch_stage2_noise_block(
                ctx_attn_dispatch,
                meta_dev,
                mask_dev,
                bundle_ptrs.gumbel_ptrs,
                bundle_ptrs.q_row_base_dev,
                gamma_eff,
                row_layout,
                lm_head_path,
                prefer_tc_lm_head,
                shortlist_k,
                use_radix_topk,
                use_k32_selector,
                use_rejection_graph_selector,
                use_ngram_graph_selector,
                dstate,
                ctx,
                stream,
            )?;
            let graph = capture.end()?;
            if graph.0 != 0 {
                {
                    let mut state = head_state_arc
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    state.insert_graph(ctx.gpu, runtime_key, graph);
                }
                ctx.gpu.launch_graph(graph, stream)?;
            }
        } else {
            self.launch_stage2_noise_block(
                ctx_attn_dispatch,
                meta_dev,
                mask_dev,
                bundle_ptrs.gumbel_ptrs,
                bundle_ptrs.q_row_base_dev,
                gamma_eff,
                row_layout,
                lm_head_path,
                prefer_tc_lm_head,
                shortlist_k,
                use_radix_topk,
                use_k32_selector,
                use_rejection_graph_selector,
                use_ngram_graph_selector,
                dstate,
                ctx,
                stream,
            )?;
        }

        if let Some(plan) = split_kv_plan {
            static SPLIT_KV_RECEIPT: std::sync::Once = std::sync::Once::new();
            SPLIT_KV_RECEIPT.call_once(|| {
                tracing::info!(
                    attn_mode = ?attn_mode,
                    eff_ctx,
                    num_q_heads = self.num_q_heads,
                    num_splits = plan.num_splits,
                    ctas = self.num_q_heads * plan.num_splits as usize,
                    sm_count = atlas_core::device::active_sm_count(),
                    "DFLASH_SPLIT_KV_H128_ENGAGED"
                );
            });
        }

        if attn_mode == Stage2CtxAttnMode::CircularRingH128 {
            static RING_GRAPH_RECEIPT: std::sync::Once = std::sync::Once::new();
            RING_GRAPH_RECEIPT.call_once(|| {
                tracing::info!(
                    needed_start,
                    eff_ctx,
                    ring_start,
                    ctx_window = self.ctx_window,
                    gamma = gamma_eff,
                    graphed = stream != 0,
                    "DFLASH_NOISE_GRAPH_RING_ENGAGED"
                );
            });
        }

        if (use_k32_selector || use_rejection_graph_selector || use_ngram_graph_selector)
            && self.markov.is_none()
        {
            dstate.pending_draft_probs_gamma = gamma_eff;
        }
        if use_rejection_graph_selector {
            dstate.set_last_rejection_q(valid_q_record);
        }

        if self.markov.is_some() {
            super::super::markov::mark_markov_completed_in_noise_graph(
                self as *const Self as usize,
                stream,
                last_token,
                gamma_eff,
            );
        }

        if let Some(spec) = staged_spec
            && use_gumbel_graph_selector
        {
            static C1_GUMBEL_RECEIPT: std::sync::Once = std::sync::Once::new();
            C1_GUMBEL_RECEIPT.call_once(|| {
                tracing::info!(
                    gamma = gamma_eff,
                    shortlist_k,
                    temp = spec.temperature,
                    top_k = spec.top_k,
                    top_p = spec.top_p,
                    min_p = spec.min_p,
                    step0_abs_pos = spec.step0_abs_pos,
                    "DFLASH2_COUPLED_GUMBEL_ENGAGED mode=c1_graph"
                );
            });
        }

        // DFlash2 selector walk: when `dflash2_selector_walk_probabilistic_batched`,
        // `dflash2_selector_walk_gumbel_batched`, or `dflash2_selector_walk_batched`
        // is available (and Markov is inactive), Stage 2 already ran it inside the
        // captured CUDA graph reading `last_token` from `meta_dev + 4`. Fall back
        // to the post-graph scalar walk only when all batched symbols are absent.
        let selector_walk_ran_in_graph = self.markov.is_none()
            && (use_rejection_graph_selector
                || use_ngram_graph_selector
                || use_gumbel_graph_selector
                || self.kernels.dflash2_selector_walk_batched.0 != 0);
        if let Some(sel) = &self.selector
            && !selector_walk_ran_in_graph
        {
            crate::layers::ops::dflash2_selector_walk(
                ctx.gpu,
                self.kernels.dflash2_selector_walk,
                self.scratch.topk_logits_dev,
                self.scratch.topk_tokens_dev,
                self.scratch.selector_hidden,
                sel.predecessor_codebook.weight,
                sel.successor_codebook.weight,
                self.scratch.draft_tokens_dev,
                last_token,
                gamma_eff as i32,
                sel.rank as i32,
                stream,
            )?;
        }

        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_stage1_layer_kv(
        &self,
        layer: &DflashLayerQuantWeights,
        fc_src: DevicePtr,
        dst_k: DevicePtr,
        dst_v: DevicePtr,
        pos_dev: DevicePtr,
        span_count: usize,
        h: u32,
        kv_dim: u32,
        prefill_pipe_requested: bool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;
        let k_norm = match layer {
            DflashLayerQuantWeights::Bf16(l) => {
                self.project_bf16_dense(
                    gpu,
                    fc_src,
                    &l.k_proj,
                    dst_k,
                    span_count as u32,
                    kv_dim,
                    h,
                    stream,
                )?;
                self.project_bf16_dense(
                    gpu,
                    fc_src,
                    &l.v_proj,
                    dst_v,
                    span_count as u32,
                    kv_dim,
                    h,
                    stream,
                )?;
                &l.k_norm
            }
            DflashLayerQuantWeights::Fp8(l) => {
                self.stage_fp8_project(
                    gpu,
                    fc_src,
                    &l.k_proj,
                    dst_k,
                    span_count as u32,
                    kv_dim,
                    h,
                    stream,
                )?;
                self.stage_fp8_project(
                    gpu,
                    fc_src,
                    &l.v_proj,
                    dst_v,
                    span_count as u32,
                    kv_dim,
                    h,
                    stream,
                )?;
                &l.k_norm
            }
            DflashLayerQuantWeights::Nvfp4(l) => {
                if super::super::proposal_policy::use_prefill_dual_pipe(
                    prefill_pipe_requested,
                    self.kernels.w4a16_gemm_pipe_dual.0 != 0,
                    span_count,
                    h as usize,
                )
                .map_err(anyhow::Error::msg)?
                {
                    ops::w4a16_gemm_pipe_dual(
                        gpu,
                        self.kernels.w4a16_gemm_pipe_dual,
                        fc_src,
                        &l.k_proj,
                        &l.v_proj,
                        dst_k,
                        dst_v,
                        false,
                        span_count as u32,
                        kv_dim,
                        h,
                        stream,
                    )?;
                } else if let (Some(k_t), Some(v_t)) = (l.k_proj_t.as_ref(), l.v_proj_t.as_ref())
                    && l.q_proj_t.is_some()
                    && l.o_proj_t.is_some()
                    && self.kernels.w4a16_gemm_t_m16.0 != 0
                {
                    let allow_m32 = span_count <= 32;
                    self.draft_gemm_t(
                        gpu,
                        fc_src,
                        k_t,
                        dst_k,
                        span_count as u32,
                        kv_dim,
                        h,
                        allow_m32,
                        stream,
                    )?;
                    self.draft_gemm_t(
                        gpu,
                        fc_src,
                        v_t,
                        dst_v,
                        span_count as u32,
                        kv_dim,
                        h,
                        allow_m32,
                        stream,
                    )?;
                } else {
                    self.nvfp4_project_coalesced_or_gemm(
                        gpu,
                        fc_src,
                        &l.k_proj,
                        dst_k,
                        span_count as u32,
                        kv_dim,
                        h,
                        stream,
                    )?;
                    self.nvfp4_project_coalesced_or_gemm(
                        gpu,
                        fc_src,
                        &l.v_proj,
                        dst_v,
                        span_count as u32,
                        kv_dim,
                        h,
                        stream,
                    )?;
                }
                &l.k_norm
            }
        };
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            dst_k,
            k_norm,
            dst_k,
            span_count as u32 * self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        ops::rope_yarn_scaled(
            gpu,
            self.kernels.rope_qwen3,
            DevicePtr::NULL,
            dst_k,
            pos_dev,
            span_count as u32,
            0,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            self.rope_attention_factor,
            stream,
        )
    }

    /// Stage 1 helper: incrementally project ONLY the newly captured context
    /// rows per layer directly into circular physical slots `abs_pos % ctx_window`
    /// of persistent `dstate.ctx_k_cache` / `dstate.ctx_v_cache` (splitting into
    /// two spans when crossing `ctx_window`), and apply `k_norm` + YaRN RoPE in
    /// place on those new K rows only.
    ///
    /// When all layers share the same `new_ctx_count in 1..=8` and `stream != 0`,
    /// replays (or captures) a bucketed Stage-1 CUDA graph executing all layers'
    /// `{k_proj, v_proj, k_norm, rope}` in 1 launch, then scatters the tiny
    /// `[new_ctx_count, kv_dim]` outputs into circular `ctx_k_cache`/`ctx_v_cache`.
    #[allow(clippy::too_many_arguments)]
    fn stage1_append_post_rope_ctx_kv(
        &self,
        needed_start: usize,
        eff_ctx: usize,
        h: u32,
        kv_dim: u32,
        stream: u64,
        ctx: &ForwardContext,
        dstate: &mut DflashProposerState,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let bf16 = 2usize;
        let window = self.ctx_window;
        let needed_end = needed_start
            .checked_add(eff_ctx)
            .context("DFlash Stage-1 needed_end overflow")?;
        let prefill_pipe_requested = self
            .proposal_policy
            .get_or_init(super::super::proposal_policy::ProposalPolicy::from_env)
            .prefill_pipe;

        let mut plans = Vec::with_capacity(self.layers.len());
        for layer_idx in 0..self.layers.len() {
            // `forward_block` skips copying `[0..old_fc_count)` from `ctx_fc_cache`
            // into `scratch.fc_proj` whenever every layer's `plan.old_ctx_count >= old_fc_count`
            // in circular `PostRope` layout, so `fc_src` strictly reads rows at
            // `span.linear_row >= old_ctx_count >= old_fc_count`.
            let plan = plan_stage1_ring_append(
                needed_start,
                eff_ctx,
                window,
                dstate.cache_k_start[layer_idx],
                dstate.cache_k_end[layer_idx],
                dstate.cache_v_start[layer_idx],
                dstate.cache_v_end[layer_idx],
            )?;
            ensure!(
                plan.new_abs_start.checked_add(plan.new_ctx_count) == Some(needed_end),
                "DFlash Stage-1 ring append cursor mismatch: new_abs_start={} new_ctx_count={} needed_end={needed_end}",
                plan.new_abs_start,
                plan.new_ctx_count
            );
            plans.push(plan);
        }

        // Fast path: uniform `1..=STAGE1_GRAPH_MAX_ROWS` across all layers on a
        // non-default stream -> replay/capture 1 Stage-1 CUDA graph across all layers.
        if dflash_stage1_graph_enabled()
            && stream != 0
            && !plans.is_empty()
            && (1..=STAGE1_GRAPH_MAX_ROWS).contains(&plans[0].new_ctx_count)
            && plans.iter().all(|p| {
                p.new_ctx_count == plans[0].new_ctx_count
                    && p.new_abs_start == plans[0].new_abs_start
                    && p.spans().first().map(|s| s.linear_row)
                        == plans[0].spans().first().map(|s| s.linear_row)
            })
        {
            let plan0 = plans[0];
            let new_count = plan0.new_ctx_count;
            let linear_start = plan0.spans()[0].linear_row;
            let head_state_arc = noise_graph_head_state(self as *const Self as usize);
            let prep = {
                let mut state = head_state_arc
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if state.stage1_capture_poisoned {
                    None
                } else {
                    let (fc_dev, pos_dev, k_dev, v_dev) = state.ensure_stage1_buffers(
                        gpu,
                        self.layers.len(),
                        self.hidden_size,
                        kv_dim as usize,
                    )?;
                    state.upload_stage1_positions(gpu, plan0.new_abs_start, new_count, stream)?;
                    let cached = state.stage1_graphs.get(&(new_count, stream)).copied();
                    Some((fc_dev, pos_dev, k_dev, v_dev, cached))
                }
            };
            if let Some((fc_dev, pos_dev, k_dev, v_dev, cached_s1)) = prep {
                let fc_src = self
                    .scratch
                    .fc_proj
                    .offset(linear_start * self.hidden_size * bf16);
                gpu.copy_d2d_async(fc_src, fc_dev, new_count * self.hidden_size * bf16, stream)?;
                let layer_kv_stride = STAGE1_GRAPH_MAX_ROWS * kv_dim as usize * bf16;
                let ran_graphed = if let Some(graph) = cached_s1 {
                    gpu.launch_graph(graph, stream)?;
                    true
                } else {
                    crate::layers::qwen4_fast_proj::warm_coalesced_nvfp4_kernels(gpu);
                    // Pre-warm BF16 cuBLASLt handle/workspace/plans for `m = new_count`
                    // strictly outside stream capture into dead scratch `k_dev`/`v_dev`.
                    for layer in &self.layers {
                        if let DflashLayerQuantWeights::Bf16(l) = layer {
                            self.project_bf16_dense(
                                gpu,
                                fc_dev,
                                &l.k_proj,
                                k_dev,
                                new_count as u32,
                                kv_dim,
                                h,
                                stream,
                            )?;
                            self.project_bf16_dense(
                                gpu,
                                fc_dev,
                                &l.v_proj,
                                v_dev,
                                new_count as u32,
                                kv_dim,
                                h,
                                stream,
                            )?;
                            break;
                        }
                    }
                    let cap_res: Result<GraphHandle> = (|| {
                        let capture = NoiseGraphCaptureGuard::begin(gpu, stream)?;
                        for (layer_idx, layer) in self.layers.iter().enumerate() {
                            let dst_k = k_dev.offset(layer_idx * layer_kv_stride);
                            let dst_v = v_dev.offset(layer_idx * layer_kv_stride);
                            self.launch_stage1_layer_kv(
                                layer,
                                fc_dev,
                                dst_k,
                                dst_v,
                                pos_dev,
                                new_count,
                                h,
                                kv_dim,
                                prefill_pipe_requested,
                                gpu,
                                stream,
                            )?;
                        }
                        capture.end()
                    })();
                    match cap_res {
                        Ok(graph) if graph.0 != 0 => {
                            {
                                let mut state = head_state_arc
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                state.stage1_graphs.insert((new_count, stream), graph);
                            }
                            gpu.launch_graph(graph, stream)?;
                            static S1_LOGGED: std::sync::Once = std::sync::Once::new();
                            S1_LOGGED.call_once(|| {
                                tracing::info!(
                                    new_ctx_count = new_count,
                                    num_layers = self.layers.len(),
                                    "DFLASH_STAGE1_KV_GRAPH_CAPTURED"
                                );
                            });
                            true
                        }
                        Ok(_) => true,
                        Err(err) => {
                            tracing::warn!(
                                new_ctx_count = new_count,
                                "DFLASH_STAGE1_KV_GRAPH capture failed ({err:#}); falling back to eager Stage-1"
                            );
                            let mut state = head_state_arc
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            state.stage1_capture_poisoned = true;
                            false
                        }
                    }
                };
                if ran_graphed {
                    let row_kv_bytes = kv_dim as usize * bf16;
                    for (layer_idx, plan) in plans.iter().enumerate() {
                        let src_k_base = k_dev.offset(layer_idx * layer_kv_stride);
                        let src_v_base = v_dev.offset(layer_idx * layer_kv_stride);
                        let mut local_row = 0usize;
                        for span in plan.spans() {
                            let copy_bytes = span.count * row_kv_bytes;
                            let src_off = local_row * row_kv_bytes;
                            let dst_off = span.physical_slot * row_kv_bytes;
                            gpu.copy_d2d_async(
                                src_k_base.offset(src_off),
                                dstate.ctx_k_cache[layer_idx].offset(dst_off),
                                copy_bytes,
                                stream,
                            )?;
                            gpu.copy_d2d_async(
                                src_v_base.offset(src_off),
                                dstate.ctx_v_cache[layer_idx].offset(dst_off),
                                copy_bytes,
                                stream,
                            )?;
                            local_row += span.count;
                        }
                        dstate.cache_k_start[layer_idx] = needed_start;
                        dstate.cache_k_end[layer_idx] = needed_end;
                        dstate.cache_v_start[layer_idx] = needed_start;
                        dstate.cache_v_end[layer_idx] = needed_end;
                    }
                    return Ok(());
                }
            }
        }

        for (layer_idx, (layer, plan)) in self.layers.iter().zip(plans.iter()).enumerate() {
            if plan.new_ctx_count == 0 {
                continue;
            }
            for span in plan.spans() {
                let fc_src = self
                    .scratch
                    .fc_proj
                    .offset(span.linear_row * self.hidden_size * bf16);
                let kv_byte_offset = span.physical_slot * kv_dim as usize * bf16;
                let dst_k = dstate.ctx_k_cache[layer_idx].offset(kv_byte_offset);
                let dst_v = dstate.ctx_v_cache[layer_idx].offset(kv_byte_offset);
                let pos_dev = self
                    .scratch
                    .position_ids
                    .offset(span.linear_row * std::mem::size_of::<u32>());
                self.launch_stage1_layer_kv(
                    layer,
                    fc_src,
                    dst_k,
                    dst_v,
                    pos_dev,
                    span.count,
                    h,
                    kv_dim,
                    prefill_pipe_requested,
                    gpu,
                    stream,
                )?;
            }

            dstate.cache_k_start[layer_idx] = needed_start;
            dstate.cache_k_end[layer_idx] = needed_end;
            dstate.cache_v_start[layer_idx] = needed_start;
            dstate.cache_v_end[layer_idx] = needed_end;
        }
        Ok(())
    }
}
