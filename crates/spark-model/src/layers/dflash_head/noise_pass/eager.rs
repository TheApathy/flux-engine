// SPDX-License-Identifier: AGPL-3.0-only

//! Eager DFlash noise pass execution, LM head dispatch, and argmax/selector tail.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::{BlockDiffusionDraftHead, DflashProposerState, DraftRowLayout};
use super::graph_state::{GumbelScratchPtrs, noise_graph_head_state};
use super::lm_head_path::{LmHeadPath, resolve_lm_head_path};
use super::policy::{
    LmHeadNvfp4Mode, REJECTION_SLOT_STRIDE, resolve_c1_lm_head_nvfp4_mode,
    resolve_dflash2_selector_topk, resolve_rejection_seq_staging, resolve_rejection_shortlist_k,
    should_use_coalesced_nvfp4_lm_head, validate_selector_shortlist,
};
use crate::layer::ForwardContext;

/// Pass-invariant inputs to [`BlockDiffusionDraftHead::run_noise_pass`].
/// Mirrors the locals computed once per `forward_block` call.
pub(in super::super) struct NoisePassArgs {
    pub last_token: u32,
    pub eff_ctx: usize,
    pub gamma_eff: usize,
    pub n_attn: u32,
    pub mask_id: u32,
    pub needed_start: usize,
    pub stream: u64,
    pub debug_dump: bool,
    pub force_noise_pattern: bool,
    pub lm_head_nvfp4: bool,
    pub dump_all_layers: bool,
    pub kprofile: bool,
}

impl BlockDiffusionDraftHead {
    /// True when the opt-in 4-gate in-context n-gram selector kernel and its
    /// device staging buffers are all initialized and ready to dispatch.
    #[inline]
    #[must_use]
    pub(in super::super) fn ngram_selector_ready(&self) -> bool {
        self.kernels.dflash2_selector_walk_ngram_batched.0 != 0
            && !self.scratch.ngram_tables_dev.is_null()
            && !self.scratch.ngram_prev3_dev.is_null()
            && !self.scratch.batch_anchor_ids.is_null()
    }

    /// Run one γ-block noise pass. `committed[i]` (length `gamma_eff`),
    /// when `Some(tok)`, embeds `tok` at noise row i+1 instead of the
    /// mask token (multi-step denoise feedback).
    ///
    /// On return, `self.scratch.logits` holds the pass's `[γ_eff, vocab]`
    /// logits and `self.scratch.draft_tokens_dev[0..γ_eff]` holds the
    /// per-row argmax tokens.
    ///
    /// Cache safety for repeated calls within one propose: the per-layer
    /// ctx K/V caches are append-once with explicit range tracking — on
    /// passes ≥1 `old_ctx_count == eff_ctx` and `new_ctx_count == 0`, so
    /// the layer loop only COPIES from the caches (never re-appends) and
    /// `cache_{k,v}_start/end` stay consistent. The fc_proj cache is not
    /// touched at all here (step 0 runs once in `forward_block`).
    pub(in super::super) fn run_noise_pass(
        &self,
        a: &NoisePassArgs,
        committed: &[Option<u32>],
        loop_pass: usize,
        ctx: &ForwardContext,
        dstate: &mut DflashProposerState,
    ) -> Result<()> {
        use crate::layers::ops;

        let NoisePassArgs {
            last_token,
            eff_ctx,
            gamma_eff,
            n_attn,
            mask_id,
            needed_start,
            stream,
            debug_dump,
            force_noise_pattern,
            lm_head_nvfp4,
            dump_all_layers,
            kprofile,
        } = *a;
        let h = self.hidden_size as u32;
        let q_dim = (self.num_q_heads * self.head_dim) as u32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let inter = self.intermediate_size as u32;
        let bf16 = 2usize;
        let inv_sqrt_d = 1.0f32 / (self.head_dim as f32).sqrt();
        let gpu = ctx.gpu;
        let row_layout = DraftRowLayout::for_family(self.checkpoint_family, gamma_eff);
        let noise_rows = row_layout.query_rows;
        let output_rows = gamma_eff;
        debug_assert_eq!(eff_ctx + noise_rows, n_attn as usize);

        super::super::markov::clear_markov_completed_in_noise_graph();
        dstate.pending_draft_probs_gamma = 0;
        let clamped_needed_start = needed_start.max(dstate.ctx_valid_from_abs);
        ensure!(
            eff_ctx == 0 || clamped_needed_start == needed_start,
            "DFlash noise pass context start {needed_start} precedes valid prefix {}",
            dstate.ctx_valid_from_abs
        );
        let gumbel_spec = if loop_pass == 0 {
            dstate.take_gumbel_spec(dstate.ctx_len)
        } else {
            None
        };
        let rejection_slot = if loop_pass == 0 {
            dstate.take_pending_rejection_slot()
        } else {
            None
        };
        if self.try_run_two_stage_noise_graph(
            a,
            committed,
            loop_pass,
            gumbel_spec,
            rejection_slot,
            ctx,
            dstate,
        )? {
            return Ok(());
        }

        let dump_bf16 = |label: &str, ptr: spark_runtime::gpu::DevicePtr, n: usize| -> Result<()> {
            if !debug_dump {
                return Ok(());
            }
            let mut buf = vec![0u8; n * 2];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(ptr, &mut buf)?;
            let vals: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            tracing::info!("DFLASH DUMP {label} [{n}]: {:?}", &vals);
            Ok(())
        };

        // Drain `stream` on any early `Err` return after a page-locked async H2D
        // transfer has been enqueued from `dstate` so the caller cannot mutate
        // or drop `dstate`'s pinned host buffers while DMA is still reading them.
        struct PinnedH2dErrSync<'a> {
            gpu: &'a dyn GpuBackend,
            stream: u64,
            armed: bool,
        }
        impl Drop for PinnedH2dErrSync<'_> {
            fn drop(&mut self) {
                if self.armed {
                    let _ = self.gpu.synchronize(self.stream);
                }
            }
        }
        let mut pinned_h2d_guard = PinnedH2dErrSync {
            gpu,
            stream,
            armed: false,
        };

        // ── Step 2: stream_buf layout ──
        // First eff_ctx rows: zero (Q-side ctx is zero; K/V-side gets
        // overwritten in step 3b' below).
        // Next noise_rows: embed of [last_token (bonus), mask, mask, ..., mask].
        // The drafter is trained with query = [next_token_id, mask × (γ-1)]
        // per vLLM (qwen3_dflash.py + dflash.py:set_inputs_first_pass: "Q from
        // query embeddings (bonus + mask tokens)"). Without the bonus token at
        // position 0, the drafter has no anchor and produces a constant
        // high-frequency token (`,`, `<|im_end|>`) for every position.
        // Multi-step denoise (passes ≥1): rows with a committed prediction
        // embed the predicted token instead of the mask.
        // Total stream_buf width = n_attn rows.
        // [eff_ctx zeros, last_token (bonus), per-row mask-or-committed × γ_eff].
        //
        // Pass ≥1 with DeepLoop enabled: use the GPU-side commit-all path.
        // Stage pass N's argmax from draft_tokens_dev[0..gamma_eff] into
        // topk_tokens_dev (non-overlapping), then reconstruct draft_tokens_dev
        // on-stream via dflash_token_recommit — no host D2H, async-safe.
        if loop_pass > 0 && super::super::dflash_deeploop_enabled() {
            gpu.copy_d2d_async(
                self.scratch.draft_tokens_dev,
                self.scratch.topk_tokens_dev,
                gamma_eff * 4,
                stream,
            )?;
            ops::token_recommit(
                gpu,
                self.kernels.token_recommit,
                self.scratch.draft_tokens_dev,
                self.scratch.topk_tokens_dev,
                last_token,
                eff_ctx as u32,
                n_attn,
                stream,
            )?;
        } else {
            let token_bytes = super::super::proposal_inputs::encode_noise_token_ids(
                dstate.token_ids_pinned.as_mut_slice(),
                eff_ctx,
                last_token,
                row_layout.feedback_rows(),
                committed,
                mask_id,
            )?;
            if debug_dump {
                let token_ids_host: Vec<u32> = std::iter::repeat_n(0u32, eff_ctx)
                    .chain(std::iter::once(last_token))
                    .chain(
                        (0..row_layout.feedback_rows())
                            .map(|i| committed.get(i).copied().flatten().unwrap_or(mask_id)),
                    )
                    .collect();
                tracing::info!(
                    "DFLASH DUMP token_ids_host: mask={} eff_ctx={} ids[0..8]={:?}",
                    mask_id,
                    eff_ctx,
                    &token_ids_host[..token_ids_host.len().min(8)],
                );
            }
            let pinned = dstate.token_ids_pinned.pinned_slice(token_bytes)?;
            // SAFETY: token_ids_pinned is page-locked, owned by dstate, and
            // separate from the concurrently pending position-id upload. On Ok,
            // the caller drains this proposal stream before reusing dstate; on
            // Err, `pinned_h2d_guard` drains `stream` before returning.
            unsafe {
                gpu.copy_h2d_pinned_async(pinned, self.scratch.draft_tokens_dev, stream)?;
            }
            pinned_h2d_guard.armed = true;
        }
        ops::batched_embed(
            gpu,
            self.kernels.batched_embed,
            self.scratch.draft_tokens_dev,
            self.embed_tokens_shared,
            self.scratch.stream_buf,
            n_attn,
            h,
            stream,
        )?;
        // Zero ctx slots (batched_embed wrote token-0 embedding to them).
        if eff_ctx > 0 {
            // Stream-ordered on `stream` so the zero runs strictly after
            // `batched_embed` without synchronizing the host under async propose.
            gpu.memset_async(
                self.scratch.stream_buf,
                0,
                eff_ctx * self.hidden_size * bf16,
                stream,
            )?;
        }
        // ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN=1: overwrite noise rows
        // [eff_ctx..n_attn) with a deterministic pattern matching the
        // PyTorch reference. Lets us compare layer-0 q/k/v post-projection
        // when both Flux and PyTorch see identical input.
        if force_noise_pattern {
            let mut bytes = Vec::with_capacity(noise_rows * self.hidden_size * bf16);
            for t in 0..noise_rows {
                for j in 0..self.hidden_size {
                    let v =
                        0.001_f32 * ((t + 1) as f32) * ((j + 1) as f32) / (self.hidden_size as f32);
                    let bf16_bits = (v.to_bits() >> 16) as u16;
                    bytes.extend_from_slice(&bf16_bits.to_le_bytes());
                }
            }
            gpu.synchronize(stream)?;
            gpu.copy_h2d(
                &bytes,
                self.scratch
                    .stream_buf
                    .offset(eff_ctx * self.hidden_size * bf16),
            )?;
        }

        // ── Step 3: 8 drafter layers ──
        //
        // All compute runs on `n_attn = eff_ctx + γ` rows. Slots [0..eff_ctx]
        // are CTX (Q-zero / KV from fc_proj projection) and slots
        // [eff_ctx..n_attn] are NOISE (full Q/K/V from embeddings).
        // Per-layer flow follows `dflash.py:Qwen3DFlashDecoderLayer.forward`.
        // Body extracted to `forward_block_layer.rs` for the 500-LoC budget.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let args = super::super::forward_block_layer::LayerArgs {
                layer_idx,
                n_attn,
                eff_ctx,
                h,
                q_dim,
                kv_dim,
                inter,
                bf16,
                inv_sqrt_d,
                stream,
                needed_start,
                window: self.ctx_window,
                loop_pass,
            };
            self.forward_block_layer(layer, &args, ctx, debug_dump, dstate, kprofile)?;
        }

        // ── Step 4: family-specific final RMSNorm + LM head rows ──
        // Generic DFlash skips the anchor and reads γ MASK rows. DSpark
        // starts at the anchor and reads all γ query rows; dropping that
        // row shifts every proposal one position into the future.
        let noise_byte_offset = (eff_ctx + row_layout.output_start) * self.hidden_size * bf16;
        let stream_noise = self.scratch.stream_buf.offset(noise_byte_offset);
        let norm_noise = self.scratch.norm_buf.offset(noise_byte_offset);
        let lm_head_path = resolve_lm_head_path(
            lm_head_nvfp4,
            self.lm_head_shared_t.is_some(),
            self.kernels.w4a16_gemm_t_m32_n64.0 != 0,
            self.lm_head_shared_fp8.is_some(),
        )
        .map_err(anyhow::Error::msg)?;
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
        if self.dspine.is_none() || lm_head_path.uses_fp8_input_norm() {
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
        }
        // Capped vocab: the shared lm_head may have fewer rows than the
        // drafter's vocab_size (e.g. target capped 248320→248077).
        let lm_vocab = self.target_vocab_size.min(self.vocab_size) as u32;
        // PERF NOTE (2026-05-19): tried switching to `dense_gemm_tc` here —
        // the shape (M=γ_eff≤16, N=vocab=248k, K=2048) looks ideal for the
        // tensor-core kernel (M_TILE=16, m16n8k16 MMA). But A/B benchmark
        // showed zero throughput change (8.70 vs 8.72 mean tok/s). The
        // lm_head GEMM is bandwidth-bound on weight read (~1GB at 273GB/s
        // ≈ 4ms) — TC compute doesn't shrink the floor. Left as scalar
        // dense_gemm to keep the dispatch simple.
        //
        // ATLAS_DFLASH_LM_HEAD_NVFP4=1 (2026-07-31, propose-to-floor): the
        // floor is the weight READ, so shrink the read — the target's
        // NVFP4-T lm_head (shared allocation, ATLAS_LM_HEAD_T) holds the
        // same rows at 1/4 the bytes; the 96k --mtp-vocab prefix is a
        // column slice of the T layout read via ldb. ~1 GB BF16 → ~0.26 GB.
        // Drafter-only (verify commits the target's own argmax) — gated on
        // accept-rate hold + md5 constitution, not bit-parity of drafts.
        // ATLAS_DFLASH_LM_HEAD_FP8 (2026-07-31, second attempt): halve the
        // read instead of quartering it — pre-scaled E4M3 slice (built at
        // load; see lm_head_shared_fp8 field doc) keeps ~3 mantissa bits of
        // logit fidelity where the NVFP4 slice's E2M1 measured accepted
        // 5.88→5.36 and lost more than the bandwidth bought. Its paired
        // input norm carries the 1/s compensation; `self.norm` stays canonical
        // for the NVFP4/BF16 heads and the candidate selector.
        // The NVFP4-T fast path must be checked FIRST: when an experimental
        // FP8 head is also present, the old else-if order shadowed NVFP4 — the
        // NVFP4-T lm_head was built (~0.26 GB vs the FP8 slice's ~1.24 GB) but
        // never used, so ATLAS_DFLASH_LM_HEAD_NVFP4=1 silently measured FP8.
        match lm_head_path {
            LmHeadPath::Nvfp4 => {
                crate::kprof!(gpu, stream, "draft_lm_head_nvfp4_t", {
                    self.launch_nvfp4_lm_head(
                        gpu,
                        norm_noise,
                        self.scratch.logits,
                        gamma_eff as u32,
                        lm_vocab,
                        h,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
            }
            LmHeadPath::Fp8 => {
                let fp8 = self
                    .lm_head_shared_fp8
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("DFlash FP8 path lost its weight/norm pair"))?;
                crate::kprof!(gpu, stream, "draft_lm_head_fp8", {
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
                    anyhow::Result::<()>::Ok(())
                })?;
            }
            LmHeadPath::Bf16 => self.launch_lm_head_bf16_or_shared_nvfp4_exact(
                gpu,
                norm_noise,
                self.scratch.logits,
                gamma_eff,
                lm_vocab,
                h,
                resolve_c1_lm_head_nvfp4_mode(gumbel_spec.is_some()),
                stream,
            )?,
        }

        // The DFlash2 selector projects the ordinary final-normalized hidden,
        // not the FP8 head's compensated input. Restore that canonical value
        // on the same stream after the FP8 logits no longer read this buffer.
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

        // Optional full-stream dump after lm_head (and FP8 unscaled-norm restore).
        if debug_dump {
            dump_bf16("final.norm_buf[noise0]", norm_noise, 10)?;
            // Sanity-check: dump first 10 BF16 values of target's lm_head_shared.
            // If this returns zeros or garbage, the BF16 lm_head was freed by
            // factory.rs's NVFP4 quantization step.
            dump_bf16("final.lm_head_shared[0..10]", self.lm_head_shared, 10)?;
        }

        // ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS=1: final norm/logits dumps.
        if dump_all_layers {
            let norm_bytes = output_rows * self.hidden_size * bf16;
            let mut buf = vec![0u8; norm_bytes];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(norm_noise, &mut buf)?;
            let path = "/tmp/atlas_final_norm_buf.bin";
            if !std::path::Path::new(path).exists() {
                let _ = std::fs::write(path, &buf);
                tracing::info!("DFLASH DUMP_ALL: wrote {norm_bytes}B to {path}");
            }
        }

        // ── Step 5: per-row argmax (generic) OR DFlash 2 candidate-selector
        // greedy walk (selector present) → γ_eff token ids ──
        let lm_stride = self.target_vocab_size.min(self.vocab_size);
        if let Some(sel) = &self.selector {
            // CandidateSelector.select at T=0 (reference: z-lab/dflash
            // dflash/model.py): top-k per position, then a sequential greedy
            // walk scoring each candidate against the predecessor via the
            // low-rank codebooks — replaces the per-row argmax. The walk is
            // drafter-only; the target verify still commits its own greedy
            // token, so this only changes which tokens are *proposed*.
            let sampled_request = dstate
                .active_sampling_request
                .as_ref()
                .filter(|request| request.request_sampler.temperature > 0.0);
            let is_probabilistic_sampled = sampled_request.is_some();
            let rejection_env = spark_runtime::sampler::gumbel::rejection_sampling_enabled();
            let rejection_slot_valid = rejection_slot.filter(|&slot| {
                gamma_eff > 0
                    && gamma_eff <= REJECTION_SLOT_STRIDE
                    && slot
                        .checked_mul(REJECTION_SLOT_STRIDE)
                        .and_then(|base| base.checked_add(gamma_eff))
                        .is_some_and(|end| end <= self.scratch.shared_q_row_cap)
            });
            let use_rejection_selector = rejection_env
                && !is_probabilistic_sampled
                && self.markov.is_none()
                && self.draft_sample_method
                    != crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic
                && self.rejection_selector_ready()
                && self.scratch.shared_q_stride
                    == crate::model::trait_impl::device_topk::REJECTION_Q_STRIDE
                && !self.scratch.batch_anchor_ids.is_null()
                && rejection_slot_valid.is_some();
            let effective_gumbel_spec = if rejection_env { None } else { gumbel_spec };
            let use_ngram_selector = super::super::ngram_bonus::admit_c1_ngram_selector(
                super::super::ngram_bonus::C1NgramAdmission {
                    ngram_enabled: super::super::ngram_bonus::ngram_bonus_enabled(),
                    ngram_ready: self.ngram_selector_ready(),
                    use_rejection_selector,
                    is_probabilistic_sampled,
                    markov_active: self.markov.is_some(),
                    probabilistic_method: self.draft_sample_method
                        == crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic,
                    raw_gumbel_spec: gumbel_spec,
                },
            );
            let can_use_k32 = !is_probabilistic_sampled
                && !use_rejection_selector
                && !use_ngram_selector
                && self.markov.is_none()
                && self.draft_sample_method
                    != crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic
                && self.gumbel_selector_k32_ready()
                && !self.scratch.batch_anchor_ids.is_null();
            let (shortlist_k, use_radix_topk, use_k32_selector) = if is_probabilistic_sampled {
                (sel.top_k, false, false)
            } else if use_rejection_selector {
                let (k, use_radix) = resolve_rejection_shortlist_k(
                    lm_stride,
                    self.kernels.topk_bf16_radix.0 != 0,
                    self.scratch.shared_q_stride,
                );
                (k, use_radix, false)
            } else {
                resolve_dflash2_selector_topk(
                    sel.top_k,
                    lm_stride,
                    effective_gumbel_spec.is_some(),
                    self.kernels.topk_bf16_radix.0 != 0,
                    can_use_k32,
                )
            };
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
                let layout = super::super::forward_block::checked_topk_layout(
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
            // hidden = hidden_projection(final_norm(noise)) → [γ, rank].
            // Reference: `select` applies `hidden_projection` to the
            // (already final-normed) draft hidden — `norm_noise` here.
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
            if let Some(request) = sampled_request {
                request.validate(gamma_eff)?;
                anyhow::ensure!(
                    self.draft_sample_method
                        == crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic,
                    "sampled DFlash request reached a greedy proposer"
                );
                anyhow::ensure!(
                    self.kernels.dflash2_selector_walk_probabilistic.0 != 0
                        && !self.scratch.selector_uniforms.is_null()
                        && !self.scratch.selector_q_modifiers.is_null()
                        && !self.scratch.selector_q_probabilities.is_null()
                        && !self.scratch.selector_q_candidates.is_null(),
                    "sampled DFlash selector kernel or scratch is unavailable"
                );
                // Validate and encode both pinned tables before launching either
                // async H2D transfer so a validation failure never leaves a
                // partial transfer in flight.
                let (uniform_pinned, modifier_pinned, modifier_count) = {
                    let uniform_staging =
                        dstate.selector_uniforms_pinned.as_mut().ok_or_else(|| {
                            anyhow::anyhow!("sampled DFlash uniform staging is missing")
                        })?;
                    let uniform_bytes = super::super::proposal_inputs::encode_selector_uniforms(
                        uniform_staging.as_mut_slice(),
                        &request.draft_uniforms,
                    )?;
                    let uniform_pinned = uniform_staging.pinned_slice(uniform_bytes)?;
                    let modifier_staging =
                        dstate.selector_q_modifiers_pinned.as_mut().ok_or_else(|| {
                            anyhow::anyhow!("sampled DFlash q modifier staging is missing")
                        })?;
                    let count = super::super::selector_policy::encode_selector_q_modifiers(
                        modifier_staging.as_mut_slice(),
                        &request.request_sampler.logit_bias,
                        &request.target_forbidden_token_ids,
                    )?;
                    let modifier_count = i32::try_from(count)
                        .map_err(|_| anyhow::anyhow!("DFlash q modifier count exceeds i32 ABI"))?;
                    let modifier_pinned = modifier_staging
                        .pinned_slice(super::super::selector_policy::SELECTOR_Q_MODIFIER_BYTES)?;
                    (uniform_pinned, modifier_pinned, modifier_count)
                };
                // SAFETY: both page-locked buffers belong to this sequence, are
                // distinct allocations, and remain alive until the producer
                // stream drains (`pinned_h2d_guard` drains on any `Err` path).
                unsafe {
                    gpu.copy_h2d_pinned_async(
                        uniform_pinned,
                        self.scratch.selector_uniforms,
                        stream,
                    )?;
                    pinned_h2d_guard.armed = true;
                    gpu.copy_h2d_pinned_async(
                        modifier_pinned,
                        self.scratch.selector_q_modifiers,
                        stream,
                    )?;
                }
                ops::dflash2_selector_walk_probabilistic(
                    gpu,
                    self.kernels.dflash2_selector_walk_probabilistic,
                    self.scratch.topk_logits_dev,
                    self.scratch.topk_tokens_dev,
                    self.scratch.selector_hidden,
                    sel.predecessor_codebook.weight,
                    sel.successor_codebook.weight,
                    self.scratch.draft_tokens_dev,
                    self.scratch.selector_uniforms,
                    self.scratch.selector_q_modifiers,
                    self.scratch
                        .selector_q_modifiers
                        .offset(super::super::selector_policy::SELECTOR_Q_MODIFIER_VALUES_OFFSET),
                    modifier_count,
                    request.request_sampler.temperature,
                    request.request_sampler.top_k,
                    request.request_sampler.min_p,
                    request.request_sampler.top_p,
                    self.scratch.selector_q_probabilities,
                    self.scratch.selector_q_candidates,
                    last_token,
                    gamma_eff as i32,
                    sel.rank as i32,
                    stream,
                )?;
            } else if use_rejection_selector && let Some(slot) = rejection_slot_valid {
                let temp_scale = spark_runtime::sampler::gumbel::rejection_draft_temp_scale();
                let (staged_spec, spec_row_base, valid_q_record) = resolve_rejection_seq_staging(
                    gumbel_spec,
                    Some(slot),
                    gamma_eff,
                    self.scratch.shared_q_row_cap,
                    temp_scale,
                );
                let q_row_base = if staged_spec.is_some() {
                    spec_row_base
                } else {
                    (slot * REJECTION_SLOT_STRIDE) as u32
                };
                let head_state_arc = noise_graph_head_state(self as *const Self as usize);
                {
                    let mut state = head_state_arc
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    state.upload_gumbel_meta_c1_with_q_base(
                        gpu,
                        staged_spec,
                        GumbelScratchPtrs::from_scratch(&self.scratch),
                        Some((last_token, self.scratch.batch_anchor_ids)),
                        Some((q_row_base, self.scratch.q_row_bases)),
                        stream,
                    )?;
                }
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
                    Some(self.scratch.q_row_bases),
                    self.scratch.batch_anchor_ids,
                    self.scratch.batch_gumbel_seeds,
                    self.scratch.batch_gumbel_step0_pos,
                    self.scratch.batch_gumbel_temps,
                    self.scratch.batch_gumbel_top_ks,
                    self.scratch.batch_gumbel_top_ps,
                    self.scratch.batch_gumbel_min_ps,
                    shortlist_k,
                    self.scratch.shared_q_stride,
                    gamma_eff,
                    sel.rank,
                    1,
                    stream,
                )?;
                if loop_pass == 0 {
                    dstate.pending_draft_probs_gamma = gamma_eff;
                    dstate.set_last_rejection_q(valid_q_record);
                }
            } else if use_ngram_selector {
                let head_state_arc = noise_graph_head_state(self as *const Self as usize);
                {
                    let mut state = head_state_arc
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    state.upload_ngram_meta_c1(
                        gpu,
                        &dstate.pld_tokens,
                        last_token,
                        self.scratch.ngram_tables_dev,
                        self.scratch.ngram_prev3_dev,
                        Some(self.scratch.batch_anchor_ids),
                        stream,
                    )?;
                }
                let cfg = super::super::ngram_bonus::NgramBonusConfig::default();
                ops::dflash2_selector_walk_ngram_batched(
                    gpu,
                    self.kernels.dflash2_selector_walk_ngram_batched,
                    &ops::Dflash2NgramWalkArgs {
                        unary: self.scratch.topk_logits_dev,
                        candidates: self.scratch.topk_tokens_dev,
                        hidden: self.scratch.selector_hidden,
                        pred: sel.predecessor_codebook.weight,
                        succ: sel.successor_codebook.weight,
                        out_tokens: self.scratch.draft_tokens_dev,
                        out_probs: (!self.scratch.batch_out_probs.is_null())
                            .then_some(self.scratch.batch_out_probs),
                        anchor_ids: self.scratch.batch_anchor_ids,
                        prev3_tokens: Some(self.scratch.ngram_prev3_dev),
                        ngram_tables: Some(self.scratch.ngram_tables_dev),
                        temperatures: None,
                        margin_thresh: cfg.margin_thresh,
                        max_bonus: cfg.effective_max_bonus(),
                        top_k_shortlist: shortlist_k,
                        gamma: gamma_eff,
                        rank: sel.rank,
                        batch_size: 1,
                    },
                    stream,
                )?;
                if loop_pass == 0 && !self.scratch.batch_out_probs.is_null() {
                    dstate.pending_draft_probs_gamma = gamma_eff;
                }
            } else if use_k32_selector {
                let head_state_arc = noise_graph_head_state(self as *const Self as usize);
                {
                    let mut state = head_state_arc
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    state.upload_gumbel_meta_c1(
                        gpu,
                        effective_gumbel_spec,
                        GumbelScratchPtrs::from_scratch(&self.scratch),
                        Some((last_token, self.scratch.batch_anchor_ids)),
                        stream,
                    )?;
                }
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
                    self.scratch.batch_anchor_ids,
                    self.scratch.batch_gumbel_seeds,
                    self.scratch.batch_gumbel_step0_pos,
                    self.scratch.batch_gumbel_temps,
                    self.scratch.batch_gumbel_top_ks,
                    self.scratch.batch_gumbel_top_ps,
                    self.scratch.batch_gumbel_min_ps,
                    shortlist_k,
                    gamma_eff,
                    sel.rank,
                    1,
                    stream,
                )?;
                if loop_pass == 0 {
                    dstate.pending_draft_probs_gamma = gamma_eff;
                }
                if let Some(spec) = effective_gumbel_spec {
                    static EAGER_GUMBEL_K32_RECEIPT: std::sync::Once = std::sync::Once::new();
                    EAGER_GUMBEL_K32_RECEIPT.call_once(|| {
                        tracing::info!(
                            gamma = gamma_eff,
                            shortlist_k,
                            temp = spec.temperature,
                            top_k = spec.top_k,
                            top_p = spec.top_p,
                            min_p = spec.min_p,
                            step0_abs_pos = spec.step0_abs_pos,
                            "DFLASH2_COUPLED_GUMBEL_ENGAGED mode=c1_eager_k32"
                        );
                    });
                }
            } else if let Some(spec) = effective_gumbel_spec
                && self.kernels.dflash2_selector_walk_gumbel_batched.0 != 0
                && self.gumbel_selector_scratch_ready()
                && !self.scratch.batch_anchor_ids.is_null()
                && self.draft_sample_method
                    != crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic
            {
                let head_state_arc = noise_graph_head_state(self as *const Self as usize);
                {
                    let mut state = head_state_arc
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    state.upload_gumbel_meta_c1(
                        gpu,
                        Some(spec),
                        GumbelScratchPtrs::from_scratch(&self.scratch),
                        Some((last_token, self.scratch.batch_anchor_ids)),
                        stream,
                    )?;
                }
                ops::dflash2_selector_walk_gumbel_batched(
                    gpu,
                    self.kernels.dflash2_selector_walk_gumbel_batched,
                    self.scratch.topk_logits_dev,
                    self.scratch.topk_tokens_dev,
                    self.scratch.selector_hidden,
                    sel.predecessor_codebook.weight,
                    sel.successor_codebook.weight,
                    self.scratch.draft_tokens_dev,
                    self.scratch.batch_anchor_ids,
                    self.scratch.batch_gumbel_seeds,
                    self.scratch.batch_gumbel_step0_pos,
                    self.scratch.batch_gumbel_temps,
                    self.scratch.batch_gumbel_top_ks,
                    self.scratch.batch_gumbel_top_ps,
                    self.scratch.batch_gumbel_min_ps,
                    gamma_eff,
                    sel.rank,
                    1,
                    stream,
                )?;
                {
                    static EAGER_GUMBEL_RECEIPT: std::sync::Once = std::sync::Once::new();
                    EAGER_GUMBEL_RECEIPT.call_once(|| {
                        tracing::info!(
                            gamma = gamma_eff,
                            temp = spec.temperature,
                            top_k = spec.top_k,
                            top_p = spec.top_p,
                            min_p = spec.min_p,
                            step0_abs_pos = spec.step0_abs_pos,
                            "DFLASH2_COUPLED_GUMBEL_ENGAGED mode=c1_eager"
                        );
                    });
                }
            } else {
                // Greedy walk seeded at the last verified token (`last_token`, the
                // bonus/root row's token id): draft_tokens_dev[i] = chosen
                // candidate at position i.
                ops::dflash2_selector_walk(
                    gpu,
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
            {
                static SEL_SEEN: std::sync::Once = std::sync::Once::new();
                SEL_SEEN.call_once(|| {
                    tracing::info!("DFLASH2_EXEC: selector_walk RAN");
                });
            }
        } else if let Some(dspine) = &self.dspine {
            let sampled_request = dstate
                .active_sampling_request
                .as_ref()
                .filter(|request| request.request_sampler.temperature > 0.0);
            anyhow::ensure!(
                sampled_request.is_none(),
                "DSpine serial_table refinement supports greedy proposal only (temperature=0)"
            );
            self.dspine_refine_tail(gpu, dspine, eff_ctx, gamma_eff, lm_stride, stream)?;
            {
                static DSP_SEEN: std::sync::Once = std::sync::Once::new();
                DSP_SEEN.call_once(|| {
                    tracing::info!("DSPINE_EXEC: serial_table + serial_walk RAN");
                });
            }
        } else {
            self.launch_argmax_rows_or_loop(gpu, gamma_eff, lm_stride, stream)?;
        }
        if debug_dump {
            dump_bf16("final.logits[noise0]", self.scratch.logits, 10)?;
        }
        if dump_all_layers {
            // The lm-head output is compact with `lm_stride`, which can be
            // smaller than the drafter checkpoint vocabulary after target
            // vocab capping. Copy only rows and columns written this pass.
            let logits_bytes = output_rows * lm_stride * bf16;
            let mut buf = vec![0u8; logits_bytes];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(self.scratch.logits, &mut buf)?;
            let path = "/tmp/atlas_final_logits.bin";
            if !std::path::Path::new(path).exists() {
                let _ = std::fs::write(path, &buf);
                tracing::info!("DFLASH DUMP_ALL: wrote {logits_bytes}B to {path}");
            }
        }

        pinned_h2d_guard.armed = false;
        Ok(())
    }

    /// Launch the default `LmHeadPath::Bf16` projection using the target's
    /// already-resident untransposed NVFP4 `lm_head`:
    ///
    /// - When `should_use_coalesced_nvfp4_lm_head(nvfp4_mode, gamma_eff)` holds
    ///   (`CoalescedTc` for `gamma_eff >= 2`, or any `gamma_eff > 32` where the
    ///   scalar exact multi-row tiers stop), dispatches through Tensor Core
    ///   coalesced NVFP4 W4A16 GEMM (`try_coalesced_nvfp4_w4a16_gemm`, zero
    ///   extra allocation, 100% CUDA-graph capture-safe once warmed).
    /// - Keeps `LmHeadNvfp4Mode::Exact` with `gamma_eff <= 32` (including C=1
    ///   T=0 greedy) on the existing `exact_kernels` path for 0-ULP parity.
    /// - Falls back to BF16 `dense_gemm` only when no NVFP4 LM head or matching
    ///   kernel is available.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn launch_lm_head_bf16_or_shared_nvfp4_exact(
        &self,
        gpu: &dyn GpuBackend,
        norm_noise: DevicePtr,
        logits_out: DevicePtr,
        gamma_eff: usize,
        lm_vocab: u32,
        h: u32,
        nvfp4_mode: LmHeadNvfp4Mode,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;

        if let Some(nvfp4) = self.lm_head_shared_nvfp4_exact.as_ref() {
            if should_use_coalesced_nvfp4_lm_head(nvfp4_mode, gamma_eff)
                && crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                    gpu,
                    norm_noise,
                    nvfp4,
                    logits_out,
                    gamma_eff,
                    lm_vocab as usize,
                    h as usize,
                    stream,
                )?
            {
                static COALESCED_LM_HEAD_RECEIPT: std::sync::Once = std::sync::Once::new();
                COALESCED_LM_HEAD_RECEIPT.call_once(|| {
                    tracing::info!(
                        rows = gamma_eff,
                        vocab = lm_vocab,
                        hidden = h,
                        mode = ?nvfp4_mode,
                        "DFLASH_LM_HEAD_NVFP4_COALESCED_ENGAGED"
                    );
                });
                return Ok(());
            }
            if let Some(exact_kernels) = self.lm_head_exact_kernels
                && h.is_multiple_of(16)
            {
                if gamma_eff == 1 {
                    static EXACT_M1_RECEIPT: std::sync::Once = std::sync::Once::new();
                    EXACT_M1_RECEIPT.call_once(|| {
                        tracing::info!(
                            vocab = lm_vocab,
                            hidden = h,
                            "DFLASH_LM_HEAD_NVFP4_EXACT_ENGAGED mode=m1_or_sw4"
                        );
                    });
                    return ops::w4a16_gemv_or_sw4(
                        gpu,
                        self.kernels.w4a16_gemv,
                        exact_kernels.sw4_m8(),
                        norm_noise,
                        nvfp4,
                        logits_out,
                        lm_vocab,
                        h,
                        stream,
                    );
                } else if gamma_eff >= 2
                    && matches!(
                        exact_kernels.route_for_rows(gamma_eff as u32),
                        Some(ops::ExactLmHeadRoute::Exact(_))
                    )
                {
                    static EXACT_BATCH_RECEIPT: std::sync::Once = std::sync::Once::new();
                    EXACT_BATCH_RECEIPT.call_once(|| {
                        tracing::info!(
                            rows = gamma_eff,
                            vocab = lm_vocab,
                            hidden = h,
                            "DFLASH_LM_HEAD_NVFP4_EXACT_ENGAGED mode=batch_exact"
                        );
                    });
                    return ops::w4a16_gemv_batch_logits_exact(
                        gpu,
                        exact_kernels,
                        norm_noise,
                        nvfp4,
                        logits_out,
                        gamma_eff as u32,
                        lm_vocab,
                        h,
                        stream,
                    );
                }
            }
        }
        self.project_bf16_dense(
            gpu,
            norm_noise,
            &crate::weight_map::DenseWeight {
                weight: self.lm_head_shared,
            },
            logits_out,
            gamma_eff as u32,
            lm_vocab,
            h,
            stream,
        )
    }

    /// Check whether the requested LM-head path can be resolved cleanly.
    pub(in super::super) fn lm_head_path_valid(&self, lm_head_nvfp4: bool) -> bool {
        resolve_lm_head_path(
            lm_head_nvfp4,
            self.lm_head_shared_t.is_some(),
            self.kernels.w4a16_gemm_t_m32_n64.0 != 0,
            self.lm_head_shared_fp8.is_some(),
        )
        .is_ok()
    }

    /// Launch the drafter NVFP4 LM head. On wide discrete GPUs (`sm_count > 64`)
    /// at `rows <= 16`, prefers the untransposed BF16×BF16 MMA coalesced v2
    /// kernel (`w4a16_gemm_coalesced_v2_n64_m16`: `TILE_M=16`, 2-stage K=256
    /// prefetch, zero `smem_B_fp8`, zero E4M3 clipping) when the shared
    /// untransposed NVFP4 LM head is wired; otherwise uses transposed
    /// `w4a16_gemm_t_m32_n64` with `ldb`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_nvfp4_lm_head(
        &self,
        gpu: &dyn GpuBackend,
        norm_noise: DevicePtr,
        logits_out: DevicePtr,
        rows: u32,
        lm_vocab: u32,
        h: u32,
        stream: u64,
    ) -> Result<()> {
        if rows <= 16
            && let Some(nvfp4_exact) = self.lm_head_shared_nvfp4_exact.as_ref()
            && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                gpu,
                norm_noise,
                nvfp4_exact,
                logits_out,
                rows,
                lm_vocab,
                h,
                16,
                stream,
            )?
        {
            return Ok(());
        }
        let nvfp4 = self
            .lm_head_shared_t
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DFlash NVFP4 path lost its transposed LM head"))?;
        crate::layers::ops::w4a16_gemm_n64_m32_ldb(
            gpu,
            self.kernels.w4a16_gemm_t_m32_n64,
            norm_noise,
            nvfp4,
            logits_out,
            rows,
            lm_vocab,
            h,
            self.lm_head_shared_t_ldb,
            stream,
        )
    }

    /// Launch SM121a coalesced / Warp-Split-K NVFP4 W4A16 GEMM
    /// (`w4a16_gemm_coalesced*`, zero extra transpose memory, CUDA-graph-safe)
    /// for `m in 1..=128`, falling back to `ops::w4a16_gemm` when ineligible or
    /// unavailable.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn nvfp4_project_coalesced_or_gemm(
        &self,
        gpu: &dyn GpuBackend,
        act: DevicePtr,
        weight: &crate::weight_map::QuantizedWeight,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
            gpu, act, weight, out, m as usize, n as usize, k as usize, stream,
        )? {
            return Ok(());
        }
        crate::layers::ops::w4a16_gemm(
            gpu,
            self.kernels.w4a16_gemm,
            act,
            weight,
            out,
            m,
            n,
            k,
            stream,
        )
    }

    /// Run final RMSNorm + LM head into an explicit `logits_out` buffer
    /// (allowing cross-sequence batched propose to reuse `ctx.buffers.logits()`
    /// with zero extra VRAM) and restore unscaled `norm_out` when `LmHeadPath::Fp8`
    /// feeds a candidate selector.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn run_final_norm_and_lm_head(
        &self,
        gpu: &dyn GpuBackend,
        stream_in: DevicePtr,
        norm_out: DevicePtr,
        logits_out: DevicePtr,
        rows: usize,
        lm_vocab: u32,
        h: u32,
        lm_head_nvfp4: bool,
        nvfp4_mode: LmHeadNvfp4Mode,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;

        let lm_head_path = resolve_lm_head_path(
            lm_head_nvfp4,
            self.lm_head_shared_t.is_some(),
            self.kernels.w4a16_gemm_t_m32_n64.0 != 0,
            self.lm_head_shared_fp8.is_some(),
        )
        .map_err(anyhow::Error::msg)?;
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
            stream_in,
            lm_head_input_norm,
            norm_out,
            rows as u32,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        match lm_head_path {
            LmHeadPath::Nvfp4 => {
                self.launch_nvfp4_lm_head(
                    gpu,
                    norm_out,
                    logits_out,
                    rows as u32,
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
                    norm_out,
                    fp8.weight,
                    logits_out,
                    rows as u32,
                    lm_vocab,
                    h,
                    stream,
                )?;
            }
            LmHeadPath::Bf16 => {
                self.launch_lm_head_bf16_or_shared_nvfp4_exact(
                    gpu, norm_out, logits_out, rows, lm_vocab, h, nvfp4_mode, stream,
                )?;
            }
        }
        if lm_head_path.restores_unscaled_norm_for_selector(self.selector.is_some()) {
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                stream_in,
                &self.norm,
                norm_out,
                rows as u32,
                h,
                self.rms_norm_eps,
                stream,
            )?;
        }
        Ok(())
    }

    /// Launch 1 row-parallel `argmax_bf16_rows` across all `gamma_eff` logit
    /// rows when the kernel is present, else fall back to the per-row
    /// `argmax_bf16` loop.
    pub(super) fn launch_argmax_rows_or_loop(
        &self,
        gpu: &dyn GpuBackend,
        gamma_eff: usize,
        lm_stride: usize,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;

        let argmax_vocab = lm_stride as u32;
        if self.kernels.argmax_rows.0 != 0 && gamma_eff > 0 {
            return ops::argmax_bf16_rows(
                gpu,
                self.kernels.argmax_rows,
                self.scratch.logits,
                self.scratch.draft_tokens_dev,
                argmax_vocab,
                argmax_vocab,
                gamma_eff as u32,
                stream,
            );
        }
        let bf16 = 2usize;
        for i in 0..gamma_eff {
            let logits_row = self.scratch.logits.offset(i * lm_stride * bf16);
            let token_slot = self.scratch.draft_tokens_dev.offset(i * 4);
            ops::argmax_bf16(
                gpu,
                self.kernels.argmax,
                logits_row,
                token_slot,
                argmax_vocab,
                stream,
            )?;
        }
        Ok(())
    }
}
