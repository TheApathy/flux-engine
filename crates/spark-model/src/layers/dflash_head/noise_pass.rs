// SPDX-License-Identifier: AGPL-3.0-only

//! One denoise pass of the DFlash γ-block forward: steps 2–5 of the
//! original `forward_block` body (noise-embedding build → 8 drafter
//! layers → final norm/lm_head → per-row argmax), extracted verbatim so
//! `forward_block` can iterate the pass for multi-step denoise drafting
//! (`ATLAS_DFLASH_DENOISE_STEPS=N`). Pass k+1 re-embeds pass k's argmax
//! at rows whose prediction was confident; the remaining masked rows are
//! then conditioned on partial predictions (DiffusionGemma-style
//! iterative refinement).
//!
//! With `committed` all-`None` (the single-pass default) this reproduces
//! the pre-extraction behavior bit-for-bit.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle, PinnedHostBuffer};
use spark_runtime::kernel_args::KernelLaunch;

#[path = "lm_head_path.rs"]
mod lm_head_path;

use lm_head_path::{LmHeadPath, resolve_lm_head_path};

/// Controls whether the shared untransposed NVFP4 LM head stays on the exact
/// scalar multi-row GEMV path (`Exact`, preserving 0-ULP C=1 T=0 greedy parity)
/// or dispatches through Tensor Core coalesced NVFP4 W4A16 GEMM (`CoalescedTc`,
/// used for C=1 T>0, C>=2 batched cohorts, and always when `rows > 32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum LmHeadNvfp4Mode {
    Exact,
    CoalescedTc,
}

/// SM count threshold above which `C=1` DFlash2 draft `lm_head` engages Tensor
/// Core coalesced NVFP4 (`CoalescedTc`) even in greedy mode (GB10 `<= 64` SMs
/// stays on `Exact` in greedy mode). Target verification strictly validates
/// every drafted token, so draft `lm_head` rounding never alters committed output.
const DFLASH_C1_TC_LM_HEAD_HIGH_SM_THRESHOLD: u32 = 64;

/// Resolve `C=1` DFlash2 draft `lm_head` NVFP4 mode for a given GPU SM count:
/// - Sampled (`sampled == true`): always [`LmHeadNvfp4Mode::CoalescedTc`].
/// - Greedy (`sampled == false`): [`LmHeadNvfp4Mode::CoalescedTc`] on high-SM
///   Blackwell GPUs (`sm_count > 64`, e.g. SM120/SM103); [`LmHeadNvfp4Mode::Exact`]
///   on GB10 (`sm_count <= 64`).
#[inline]
#[must_use]
pub(super) const fn resolve_c1_lm_head_nvfp4_mode_for_sm(
    sampled: bool,
    sm_count: u32,
) -> LmHeadNvfp4Mode {
    if sampled || sm_count > DFLASH_C1_TC_LM_HEAD_HIGH_SM_THRESHOLD {
        LmHeadNvfp4Mode::CoalescedTc
    } else {
        LmHeadNvfp4Mode::Exact
    }
}

/// Resolve `C=1` DFlash2 draft `lm_head` NVFP4 mode for the active GPU.
#[inline]
#[must_use]
pub(super) fn resolve_c1_lm_head_nvfp4_mode(sampled: bool) -> LmHeadNvfp4Mode {
    resolve_c1_lm_head_nvfp4_mode_for_sm(sampled, atlas_core::device::active_sm_count())
}

/// Whether `launch_lm_head_bf16_or_shared_nvfp4_exact` should try Tensor Core
/// coalesced NVFP4 W4A16 GEMM (`try_coalesced_nvfp4_w4a16_gemm`) before the
/// scalar `exact_kernels` / BF16 fallback.
///
/// - `LmHeadNvfp4Mode::Exact` keeps `rows <= 32` (including C=1 T=0 greedy) on
///   the 0-ULP exact scalar kernels, while still routing `rows > 32` (where
///   `exact_kernels` has no tier and previously fell back to 512 MB BF16 GEMM)
///   through coalesced NVFP4.
/// - `LmHeadNvfp4Mode::CoalescedTc` engages coalesced NVFP4 for all `rows >= 2`
///   (`rows == 1` remains on the single-row NVFP4 GEMV).
#[inline]
#[must_use]
pub(super) const fn should_use_coalesced_nvfp4_lm_head(mode: LmHeadNvfp4Mode, rows: usize) -> bool {
    match mode {
        LmHeadNvfp4Mode::CoalescedTc => rows >= 2,
        LmHeadNvfp4Mode::Exact => rows > 32,
    }
}

/// Resolve `(shortlist_k, use_radix_topk, use_k32_selector)` for DFlash2
/// candidate selection.
///
/// - Keeps the checkpoint's trained `sel_top_k` (e.g. `16`) so the bilinear
///   Markov pair head never scores out-of-distribution tail candidates, while
///   still engaging the 23x faster 2-pass `topk_bf16_fast` kernel and the
///   `k32`-capable selector walk (which emits `batch_out_probs` for `p_cum`
///   confidence truncation on both greedy and CRN coupled Gumbel-Max paths).
#[inline]
#[must_use]
pub(super) fn resolve_dflash2_selector_topk(
    sel_top_k: usize,
    lm_stride: usize,
    _widened_for_sampling: bool,
    has_topk_radix: bool,
    k32_selector_ready: bool,
) -> (usize, bool, bool) {
    let base_k = sel_top_k.min(lm_stride);
    (base_k, has_topk_radix, k32_selector_ready)
}

/// Per-slot row stride in the shared GPU `(q_cand_ids, q_probs)` buffer (`slot_idx * 8`).
pub(super) const REJECTION_SLOT_STRIDE: usize = 8;

/// Resolve `(shortlist_k, use_radix_topk)` for maximal-coupling rejection
/// sampling (`dflash2_selector_walk_probabilistic_batched`), clamping
/// `rejection_shortlist_k()` to `1..=min(REJECTION_Q_STRIDE, shared_q_stride, max_topk, lm_stride)`.
#[inline]
#[must_use]
pub(super) fn resolve_rejection_shortlist_k(
    lm_stride: usize,
    has_topk_radix: bool,
    shared_q_stride: usize,
) -> (usize, bool) {
    let max_topk = if has_topk_radix {
        super::DFLASH2_GUMBEL_SHORTLIST_K
    } else {
        super::DDTREE_TOP_K_MAX
    };
    let cap = crate::model::trait_impl::device_topk::REJECTION_Q_STRIDE
        .min(shared_q_stride)
        .min(max_topk)
        .min(lm_stride);
    let k = if cap == 0 {
        0
    } else {
        spark_runtime::sampler::gumbel::rejection_shortlist_k().clamp(1, cap)
    };
    (k, has_topk_radix)
}

/// Resolve per-sequence `(staged_spec, q_row_base, valid_q_record)` for
/// maximal-coupling rejection sampling:
/// - Scales active draft temperatures by `(spec.temperature * temp_scale).max(GUMBEL_MIN_TEMPERATURE)`.
/// - Validates `slot_idx` and `slot_idx * 8 + gamma_eff <= shared_q_row_cap` with `1 <= gamma_eff <= 8`.
/// - Returns `(None, 0, None)` whenever the sequence is greedy/inactive or the
///   slot/gamma bounds do not fit, so the kernel runs exact greedy (`temp = 0.0`)
///   and `last_rejection_q` is not marked valid.
#[must_use]
pub(super) fn resolve_rejection_seq_staging(
    spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
    slot_idx: Option<usize>,
    gamma_eff: usize,
    shared_q_row_cap: usize,
    temp_scale: f32,
) -> (
    Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
    u32,
    Option<(usize, u64, usize)>,
) {
    let Some(mut s) = spec.filter(|s| s.is_active()) else {
        return (None, 0, None);
    };
    let Some(slot) = slot_idx else {
        return (None, 0, None);
    };
    if gamma_eff == 0 || gamma_eff > REJECTION_SLOT_STRIDE {
        return (None, 0, None);
    }
    let Some(row_base) = slot.checked_mul(REJECTION_SLOT_STRIDE) else {
        return (None, 0, None);
    };
    let Some(row_end) = row_base.checked_add(gamma_eff) else {
        return (None, 0, None);
    };
    if row_end > shared_q_row_cap {
        return (None, 0, None);
    }
    let Ok(row_base_u32) = u32::try_from(row_base) else {
        return (None, 0, None);
    };
    let scale = if temp_scale.is_finite() && temp_scale > 0.0 {
        temp_scale
    } else {
        1.0
    };
    s.temperature =
        (s.temperature * scale).max(spark_runtime::sampler::gumbel::GUMBEL_MIN_TEMPERATURE);
    (
        Some(s),
        row_base_u32,
        Some((slot, s.step0_abs_pos, gamma_eff)),
    )
}

/// Validate `(num_rows, vocab, k)` against selector top-K scratch capacity
/// (`max_k` is `16` for legacy `topk_bf16` and `32` for `topk_bf16_radix`).
pub(super) fn validate_selector_shortlist(
    num_rows: usize,
    scratch_rows: usize,
    vocab: usize,
    k: usize,
    max_k: usize,
) -> Result<(u32, u32, u32)> {
    ensure!(
        num_rows <= scratch_rows,
        "selector top-K rows {num_rows} exceed scratch capacity {scratch_rows}"
    );
    ensure!(vocab > 0, "selector top-K requires vocab > 0");
    let vocab_u32 = u32::try_from(vocab).context("selector top-K vocab exceeds u32")?;
    ensure!(k >= 1, "selector top-K requires k >= 1");
    ensure!(k <= max_k, "selector top-K requires k <= {max_k}, got {k}");
    ensure!(
        k <= vocab,
        "selector top-K requires k <= vocab ({vocab}), got {k}"
    );
    let rows_u32 = u32::try_from(num_rows).context("selector top-K rows exceed u32")?;
    Ok((rows_u32, vocab_u32, k as u32))
}

use super::{
    BlockDiffusionDraftHead, DflashLayer, DflashLayerFp8Weights, DflashLayerNvfp4,
    DflashLayerQuantWeights, DflashProposerState, DraftRowLayout,
};
use crate::layer::ForwardContext;

/// Pass-invariant inputs to [`BlockDiffusionDraftHead::run_noise_pass`].
/// Mirrors the locals computed once per `forward_block` call.
pub(super) struct NoisePassArgs {
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
    pub(super) fn run_noise_pass(
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

        super::markov::clear_markov_completed_in_noise_graph();
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
        if loop_pass > 0 && super::dflash_deeploop_enabled() {
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
            let token_bytes = super::proposal_inputs::encode_noise_token_ids(
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
            let args = super::forward_block_layer::LayerArgs {
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
            let can_use_k32 = !is_probabilistic_sampled
                && !use_rejection_selector
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
                    super::DFLASH2_GUMBEL_SHORTLIST_K,
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
                let layout = super::forward_block::checked_topk_layout(
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
                    let uniform_bytes = super::proposal_inputs::encode_selector_uniforms(
                        uniform_staging.as_mut_slice(),
                        &request.draft_uniforms,
                    )?;
                    let uniform_pinned = uniform_staging.pinned_slice(uniform_bytes)?;
                    let modifier_staging =
                        dstate.selector_q_modifiers_pinned.as_mut().ok_or_else(|| {
                            anyhow::anyhow!("sampled DFlash q modifier staging is missing")
                        })?;
                    let count = super::selector_policy::encode_selector_q_modifiers(
                        modifier_staging.as_mut_slice(),
                        &request.request_sampler.logit_bias,
                        &request.target_forbidden_token_ids,
                    )?;
                    let modifier_count = i32::try_from(count)
                        .map_err(|_| anyhow::anyhow!("DFlash q modifier count exceeds i32 ABI"))?;
                    let modifier_pinned = modifier_staging
                        .pinned_slice(super::selector_policy::SELECTOR_Q_MODIFIER_BYTES)?;
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
                        .offset(super::selector_policy::SELECTOR_Q_MODIFIER_VALUES_OFFSET),
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
    pub(super) fn launch_lm_head_bf16_or_shared_nvfp4_exact(
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
    pub(super) fn lm_head_path_valid(&self, lm_head_nvfp4: bool) -> bool {
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
    fn launch_nvfp4_lm_head(
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
    pub(super) fn nvfp4_project_coalesced_or_gemm(
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
    pub(super) fn run_final_norm_and_lm_head(
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
    fn launch_argmax_rows_or_loop(
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

/// Environment selector enabling the 2-stage CUDA-graphed DFlash2 noise-block
/// path (`Stage 1`: eager incremental post-`k_norm`/post-RoPE KV append + 80 B
/// metadata H2D; `Stage 2`: fixed-shape `m_rows` graphed noise pass using
/// `inferspark_prefill_split_ctx_h128` / `inferspark_prefill_split_ctx_h256`
/// and `enqueue_markov_sequential_dev` / `dflash2_selector_walk_batched`).
pub const DFLASH_NOISE_GRAPH_SELECTOR: &str = "ATLAS_DFLASH_NOISE_GRAPH";

const MAX_CACHED_NOISE_GRAPHS: usize = 32;
const NOISE_META_PINNED_SLOTS: usize = 4;
const GUMBEL_META_C1_SLOT_BYTES: usize = 40;
const GUMBEL_META_SEED_OFF: usize = 0;
const GUMBEL_META_STEP0_OFF: usize = 8;
const GUMBEL_META_TEMP_OFF: usize = 16;
const GUMBEL_META_TOPK_OFF: usize = 20;
const GUMBEL_META_TOPP_OFF: usize = 24;
const GUMBEL_META_MINP_OFF: usize = 28;
const GUMBEL_META_ANCHOR_OFF: usize = 32;
const GUMBEL_META_Q_ROW_BASE_OFF: usize = 36;

/// Canonical per-sequence scalar tuple `(seed, step0_abs_pos, temperature, top_k, top_p, min_p)`
/// uploaded to `dflash2_selector_walk_gumbel_batched`. Greedy / inactive specs emit
/// `temperature = 0.0` so the kernel takes its exact greedy argmax branch.
pub(super) fn gumbel_spec_fields(
    spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
) -> (u64, u64, f32, u32, f32, f32) {
    match spec {
        Some(s) if s.is_active() => (
            s.seed,
            s.step0_abs_pos,
            s.temperature,
            s.top_k,
            s.top_p,
            s.min_p,
        ),
        _ => (0u64, 0u64, 0.0f32, 0u32, 1.0f32, 0.0f32),
    }
}

#[derive(Clone, Copy)]
struct GumbelScratchPtrs {
    seeds: DevicePtr,
    step0_pos: DevicePtr,
    temps: DevicePtr,
    top_ks: DevicePtr,
    top_ps: DevicePtr,
    min_ps: DevicePtr,
}

impl GumbelScratchPtrs {
    fn from_scratch(scratch: &super::DflashScratch) -> Self {
        Self {
            seeds: scratch.batch_gumbel_seeds,
            step0_pos: scratch.batch_gumbel_step0_pos,
            temps: scratch.batch_gumbel_temps,
            top_ks: scratch.batch_gumbel_top_ks,
            top_ps: scratch.batch_gumbel_top_ps,
            min_ps: scratch.batch_gumbel_min_ps,
        }
    }
}

/// Fixed-layout 80-byte device metadata uploaded once per draft step in
/// Stage 1 of the 2-stage CUDA-graphed DFlash2 noise pass.
///
/// Stage 2 kernels read `eff_ctx` (`+0`), `last_token` (`+4`), and
/// `noise_positions[0..noise_len]` (`+16..+80`) directly from device memory
/// so the captured CUDA graph contains zero per-step host scalars.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DflashNoiseGraphMeta {
    pub eff_ctx: u32,
    pub last_token: u32,
    pub noise_len: u32,
    pub _pad: u32,
    pub noise_positions: [u32; 16],
}

impl DflashNoiseGraphMeta {
    pub const SIZE_BYTES: usize = 80;
    pub const MAX_NOISE_POSITIONS: usize = 16;
    pub const EFF_CTX_OFFSET: usize = 0;
    pub const LAST_TOKEN_OFFSET: usize = 4;
    #[cfg(test)]
    pub const NOISE_LEN_OFFSET: usize = 8;
    pub const NOISE_POSITIONS_OFFSET: usize = 16;

    /// Validate `eff_ctx` and `positions` bounds and build a zero-padded
    /// 80-byte metadata block ready for little-endian H2D serialization.
    pub fn try_new(
        eff_ctx: usize,
        last_token: u32,
        positions: &[u32],
        max_ctx_window: usize,
    ) -> Result<Self> {
        ensure!(
            eff_ctx <= max_ctx_window,
            "DFlash noise graph eff_ctx={eff_ctx} exceeds ctx_window={max_ctx_window}"
        );
        let eff_ctx_u32 =
            u32::try_from(eff_ctx).context("DFlash noise graph eff_ctx exceeds u32")?;
        ensure!(
            !positions.is_empty() && positions.len() <= Self::MAX_NOISE_POSITIONS,
            "DFlash noise graph noise_len={} outside 1..={}",
            positions.len(),
            Self::MAX_NOISE_POSITIONS
        );
        ensure!(
            positions[0] >= eff_ctx_u32,
            "DFlash noise graph first noise position {} precedes eff_ctx={eff_ctx_u32}",
            positions[0]
        );
        for pair in positions.windows(2) {
            ensure!(
                pair[1] > pair[0],
                "DFlash noise graph noise_positions must be strictly increasing, got {} then {}",
                pair[0],
                pair[1]
            );
        }
        let mut noise_positions = [0u32; Self::MAX_NOISE_POSITIONS];
        noise_positions[..positions.len()].copy_from_slice(positions);
        Ok(Self {
            eff_ctx: eff_ctx_u32,
            last_token,
            noise_len: positions.len() as u32,
            _pad: 0,
            noise_positions,
        })
    }

    /// Serialize into its exact 80-byte little-endian device wire layout.
    pub fn to_le_bytes(self) -> [u8; Self::SIZE_BYTES] {
        let mut out = [0u8; Self::SIZE_BYTES];
        out[0..4].copy_from_slice(&self.eff_ctx.to_le_bytes());
        out[4..8].copy_from_slice(&self.last_token.to_le_bytes());
        out[8..12].copy_from_slice(&self.noise_len.to_le_bytes());
        out[12..16].copy_from_slice(&self._pad.to_le_bytes());
        for (i, pos) in self.noise_positions.iter().enumerate() {
            let off = Self::NOISE_POSITIONS_OFFSET + i * 4;
            out[off..off + 4].copy_from_slice(&pos.to_le_bytes());
        }
        out
    }

    /// Write the 80-byte little-endian layout into `dst[..80]`.
    pub fn write_le_bytes(&self, dst: &mut [u8]) -> Result<usize> {
        ensure!(
            dst.len() >= Self::SIZE_BYTES,
            "DFlashNoiseGraphMeta staging slice too small: {} < {}",
            dst.len(),
            Self::SIZE_BYTES
        );
        let encoded = self.to_le_bytes();
        dst[..Self::SIZE_BYTES].copy_from_slice(&encoded);
        Ok(Self::SIZE_BYTES)
    }

    /// Decode an 80-byte little-endian buffer back into [`DflashNoiseGraphMeta`].
    #[cfg(test)]
    pub fn from_le_bytes(src: &[u8]) -> Result<Self> {
        ensure!(
            src.len() == Self::SIZE_BYTES,
            "DFlashNoiseGraphMeta requires exactly {} bytes, got {}",
            Self::SIZE_BYTES,
            src.len()
        );
        let read_u32 = |off: usize| -> u32 {
            u32::from_le_bytes([src[off], src[off + 1], src[off + 2], src[off + 3]])
        };
        let mut noise_positions = [0u32; Self::MAX_NOISE_POSITIONS];
        for (i, slot) in noise_positions.iter_mut().enumerate() {
            *slot = read_u32(Self::NOISE_POSITIONS_OFFSET + i * 4);
        }
        Ok(Self {
            eff_ctx: read_u32(0),
            last_token: read_u32(4),
            noise_len: read_u32(8),
            _pad: read_u32(12),
            noise_positions,
        })
    }
}

/// Precision class of the drafter transformer layers for graph-key separation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DflashLayerQuantMode {
    Bf16,
    Fp8,
    Nvfp4,
    Mixed,
}

impl DflashLayerQuantMode {
    fn from_layers(layers: &[DflashLayerQuantWeights]) -> Self {
        let mut saw_bf16 = false;
        let mut saw_fp8 = false;
        let mut saw_nvfp4 = false;
        for layer in layers {
            match layer {
                DflashLayerQuantWeights::Bf16(_) => saw_bf16 = true,
                DflashLayerQuantWeights::Fp8(_) => saw_fp8 = true,
                DflashLayerQuantWeights::Nvfp4(_) => saw_nvfp4 = true,
            }
        }
        match (saw_bf16, saw_fp8, saw_nvfp4) {
            (true, false, false) => Self::Bf16,
            (false, true, false) => Self::Fp8,
            (false, false, true) => Self::Nvfp4,
            _ => Self::Mixed,
        }
    }
}

/// How Stage-2 split-context attention indexes `dstate.ctx_{k,v}_cache`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Stage2CtxAttnMode {
    /// Linear `[0..eff_ctx)` lookup via `inferspark_prefill_split_ctx_h{128,256}`.
    /// Preserves the exact `needed_start == 0` path bit-for-bit, and also serves
    /// non-SWA full-ring (`eff_ctx == ctx_window` or aligned `needed_start % ctx_window == 0`)
    /// models when no circular-ring kernel exists (e.g. `head_dim == 256`).
    Linear,
    /// Circular ring `[needed_start % ctx_window .. +eff_ctx)` lookup via
    /// `inferspark_prefill_split_ctx_h128_batched` (`batch_size = 1`), reading
    /// per-layer `ctx_{k,v}_cache` base pointers and `ring_start` from the head's
    /// fixed Stage-2 ring metadata slab. Exact chronological order for SWA,
    /// causal masking, partial rings (prefix-cache skips), and > 4K decode.
    CircularRingH128,
}

/// Resolve which Stage-2 split-context attention mode can safely execute for
/// `(head_dim, needed_start, eff_ctx, ctx_window, has_linear_kernel, has_ring_h128_kernel, any_layer_swa)`
/// on a GPU with `sm_count` streaming multiprocessors.
///
/// On high-SM GPUs (`sm_count > 64`, e.g. SM120), `CircularRingH128` is preferred
/// even when `needed_start == 0` so Stage-2 CUDA graphs read `ctx_{k,v}_cache`
/// pointers through the fixed ring metadata slab instead of baking per-request
/// pointers into `NoiseGraphRuntimeKey.ctx_kv_ptrs` (avoiding per-request graph
/// re-capture). On GB10 (`sm_count <= 64`), `needed_start == 0` preserves `Linear`
/// bit-for-bit.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "pure Stage-2 attention mode geometry resolver mirroring resolve_stage2_ctx_attn_mode"
)]
pub(super) fn resolve_stage2_ctx_attn_mode_for_sm(
    head_dim: usize,
    needed_start: usize,
    eff_ctx: usize,
    ctx_window: usize,
    has_linear_kernel: bool,
    has_ring_h128_kernel: bool,
    any_layer_swa: bool,
    sm_count: u32,
) -> Option<Stage2CtxAttnMode> {
    if ctx_window == 0 || eff_ctx > ctx_window || !matches!(head_dim, 128 | 256) {
        return None;
    }
    if needed_start == 0 {
        if sm_count > 64 && head_dim == 128 && has_ring_h128_kernel {
            return Some(Stage2CtxAttnMode::CircularRingH128);
        }
        if has_linear_kernel {
            return Some(Stage2CtxAttnMode::Linear);
        }
        if head_dim == 128 && has_ring_h128_kernel {
            return Some(Stage2CtxAttnMode::CircularRingH128);
        }
        return None;
    }
    if head_dim == 128 && has_ring_h128_kernel {
        return Some(Stage2CtxAttnMode::CircularRingH128);
    }
    if has_linear_kernel
        && (eff_ctx == 0
            || (!any_layer_swa
                && (eff_ctx == ctx_window || needed_start.is_multiple_of(ctx_window))))
    {
        return Some(Stage2CtxAttnMode::Linear);
    }
    None
}

/// Resolve which Stage-2 split-context attention mode can safely execute on the
/// default (<= 64 SM) path.
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(super) fn resolve_stage2_ctx_attn_mode(
    head_dim: usize,
    needed_start: usize,
    eff_ctx: usize,
    ctx_window: usize,
    has_linear_kernel: bool,
    has_ring_h128_kernel: bool,
    any_layer_swa: bool,
) -> Option<Stage2CtxAttnMode> {
    resolve_stage2_ctx_attn_mode_for_sm(
        head_dim,
        needed_start,
        eff_ctx,
        ctx_window,
        has_linear_kernel,
        has_ring_h128_kernel,
        any_layer_swa,
        0,
    )
}

/// One contiguous physical write span for Stage-1 incremental circular KV append.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Stage1RingSpan {
    /// Row index into `scratch.fc_proj` and `scratch.position_ids`.
    pub linear_row: usize,
    /// Physical slot index into `dstate.ctx_{k,v}_cache[l]` (`abs_pos % window`).
    pub physical_slot: usize,
    /// Number of contiguous rows in this span.
    pub count: usize,
}

/// Pure per-layer plan for Stage-1 incremental post-`k_norm`/post-RoPE circular
/// KV append over `[needed_start .. needed_start + eff_ctx)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Stage1RingAppendPlan {
    /// Leading rows `[needed_start .. needed_start + old_ctx_count)` already
    /// valid in both `ctx_k_cache` and `ctx_v_cache`.
    pub old_ctx_count: usize,
    /// First absolute position to project (`needed_start + old_ctx_count`).
    pub new_abs_start: usize,
    /// Number of new rows to project, `k_norm`, RoPE, and write (`eff_ctx - old_ctx_count`).
    pub new_ctx_count: usize,
    spans: [Stage1RingSpan; 2],
    span_len: usize,
}

impl Stage1RingAppendPlan {
    #[inline]
    pub(super) fn spans(&self) -> &[Stage1RingSpan] {
        &self.spans[..self.span_len]
    }
}

/// Plan one layer's Stage-1 circular `PostRope` KV append over
/// `[needed_start .. needed_start + eff_ctx)`.
///
/// Reuses the cached prefix only when BOTH K and V cover `[needed_start .. valid_end)`
/// without exceeding `window` and without `cache_end > needed_end` (rollback).
/// Splits newly appended rows into at most two contiguous physical ring spans
/// when `(new_abs_start % window) + new_ctx_count > window`.
pub(super) fn plan_stage1_ring_append(
    needed_start: usize,
    eff_ctx: usize,
    window: usize,
    cache_k_start: usize,
    cache_k_end: usize,
    cache_v_start: usize,
    cache_v_end: usize,
) -> Result<Stage1RingAppendPlan> {
    ensure!(
        window > 0 && eff_ctx <= window,
        "DFlash Stage-1 ring append invalid geometry: eff_ctx={eff_ctx} window={window}"
    );
    let needed_end = needed_start
        .checked_add(eff_ctx)
        .context("DFlash Stage-1 ring append needed_end overflow")?;
    if eff_ctx == 0 {
        return Ok(Stage1RingAppendPlan {
            old_ctx_count: 0,
            new_abs_start: needed_end,
            new_ctx_count: 0,
            spans: [Stage1RingSpan::default(); 2],
            span_len: 0,
        });
    }
    let k_range_ok = cache_k_start <= cache_k_end && cache_k_end - cache_k_start <= window;
    let v_range_ok = cache_v_start <= cache_v_end && cache_v_end - cache_v_start <= window;
    let valid_start = cache_k_start.max(cache_v_start);
    let valid_end = cache_k_end.min(cache_v_end);
    let no_rollback = cache_k_end <= needed_end && cache_v_end <= needed_end;
    let old_ctx_count = if k_range_ok
        && v_range_ok
        && no_rollback
        && valid_start <= needed_start
        && valid_end > needed_start
    {
        valid_end - needed_start
    } else {
        0
    };
    let new_abs_start = needed_start + old_ctx_count;
    let new_ctx_count = eff_ctx - old_ctx_count;
    if new_ctx_count == 0 {
        return Ok(Stage1RingAppendPlan {
            old_ctx_count,
            new_abs_start,
            new_ctx_count: 0,
            spans: [Stage1RingSpan::default(); 2],
            span_len: 0,
        });
    }
    let phys_start = new_abs_start % window;
    let first_count = new_ctx_count.min(window - phys_start);
    let second_count = new_ctx_count - first_count;
    let spans = [
        Stage1RingSpan {
            linear_row: old_ctx_count,
            physical_slot: phys_start,
            count: first_count,
        },
        Stage1RingSpan {
            linear_row: old_ctx_count + first_count,
            physical_slot: 0,
            count: second_count,
        },
    ];
    let span_len = if second_count == 0 { 1 } else { 2 };
    Ok(Stage1RingAppendPlan {
        old_ctx_count,
        new_abs_start,
        new_ctx_count,
        spans,
        span_len,
    })
}

/// Whether every layer's circular `PostRope` K/V cache already covers at least
/// `[needed_start .. needed_start + old_fc_count)` without rollback or range
/// errors, guaranteeing `scratch.fc_proj[0..old_fc_count)` is never read by Stage 1.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub(super) fn all_layers_cover_old_fc_prefix(
    needed_start: usize,
    eff_ctx: usize,
    old_fc_count: usize,
    window: usize,
    cache_k_start: &[usize],
    cache_k_end: &[usize],
    cache_v_start: &[usize],
    cache_v_end: &[usize],
) -> bool {
    let num_layers = cache_k_start.len();
    if num_layers == 0
        || cache_k_end.len() != num_layers
        || cache_v_start.len() != num_layers
        || cache_v_end.len() != num_layers
        || old_fc_count > eff_ctx
    {
        return false;
    }
    (0..num_layers).all(|l| {
        plan_stage1_ring_append(
            needed_start,
            eff_ctx,
            window,
            cache_k_start[l],
            cache_k_end[l],
            cache_v_start[l],
            cache_v_end[l],
        )
        .is_ok_and(|plan| plan.old_ctx_count >= old_fc_count)
    })
}

/// Minimum `eff_ctx` for wide-SM DFlash2 Split-KV H=128 attention.
/// Below 256 rows (`<= 8` KV tiles), non-split single-launch attention avoids
/// reduce-pass overhead.
const DFLASH_SPLIT_KV_H128_MIN_EFF_CTX: usize = 256;
/// SM-count threshold above which DFlash2 Split-KV H=128 attention engages
/// (`<= 64` keeps 48-SM GB10 on the single-launch kernel).
const DFLASH_SPLIT_KV_H128_HIGH_SM_THRESHOLD: u32 = 64;
/// Minimum Split-KV partition count when Split-KV engages (`4..=8`).
const DFLASH_SPLIT_KV_H128_MIN_SPLITS: u32 = 4;
/// Maximum Split-KV partition count (`4..=8`, bounds per-head workspace to ~2 MiB).
const DFLASH_SPLIT_KV_H128_MAX_SPLITS: u32 = 8;
const DFLASH_SPLIT_KV_H128_LINEAR_KERNEL: &str = "inferspark_prefill_split_ctx_h128_splitkv";
const DFLASH_SPLIT_KV_H128_BATCHED_KERNEL: &str =
    "inferspark_prefill_split_ctx_h128_batched_splitkv";
const DFLASH_SPLIT_KV_H128_REDUCE_KERNEL: &str = "inferspark_prefill_split_ctx_h128_splitkv_reduce";

/// Compute Split-KV partition count for DFlash2 H=128 attention.
///
/// Engages `num_splits in 4..=8` only when `head_dim == 128`, `sm_count > 64`,
/// `eff_ctx >= 256`, and `num_q_heads * batch_size < sm_count` (lifting `C=1..4`
/// on 188-SM GPUs from 32–128 CTAs to 192–512 CTAs). Returns `1` otherwise.
#[inline]
#[must_use]
pub(super) fn resolve_dflash_split_kv_splits_for_sm(
    head_dim: usize,
    eff_ctx: usize,
    num_q_heads: usize,
    batch_size: usize,
    sm_count: u32,
) -> u32 {
    if head_dim != 128
        || sm_count <= DFLASH_SPLIT_KV_H128_HIGH_SM_THRESHOLD
        || eff_ctx < DFLASH_SPLIT_KV_H128_MIN_EFF_CTX
        || num_q_heads == 0
        || batch_size == 0
    {
        return 1;
    }
    let base_ctas = num_q_heads.saturating_mul(batch_size);
    if base_ctas >= sm_count as usize {
        return 1;
    }
    (sm_count as usize).div_ceil(base_ctas).clamp(
        DFLASH_SPLIT_KV_H128_MIN_SPLITS as usize,
        DFLASH_SPLIT_KV_H128_MAX_SPLITS as usize,
    ) as u32
}

/// Workspace bytes required for Split-KV H=128 partial FP32 outputs and `(m, l)` metadata:
/// `batch_size * num_splits * num_q_heads * m_rows * (128 + 2) * sizeof(f32)`.
#[inline]
#[must_use]
pub(super) fn dflash_split_kv_h128_workspace_bytes(
    batch_size: usize,
    num_splits: u32,
    num_q_heads: usize,
    m_rows: usize,
) -> Option<usize> {
    batch_size
        .checked_mul(num_splits as usize)?
        .checked_mul(num_q_heads)?
        .checked_mul(m_rows)?
        .checked_mul(128 + 2)?
        .checked_mul(std::mem::size_of::<f32>())
}

/// Logical 2-stage DFlash noise-block CUDA graph key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DflashNoiseGraphKey {
    pub gamma_eff: usize,
    pub quant_mode: DflashLayerQuantMode,
    pub markov_active: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NoiseGraphRuntimeKey {
    logical: DflashNoiseGraphKey,
    attn_mode: Stage2CtxAttnMode,
    split_kv_splits: u8,
    lm_head_path_tag: u8,
    prefer_tc_lm_head: bool,
    rejection_mode: bool,
    shortlist_k: u8,
    mask_id: u32,
    stream: u64,
    ctx_kv_ptrs: Vec<(u64, u64)>,
}

pub(super) const LM_HEAD_PATH_TAG_BF16: u8 = 2;

fn lm_head_path_tag(path: LmHeadPath) -> u8 {
    match path {
        LmHeadPath::Nvfp4 => 0,
        LmHeadPath::Fp8 => 1,
        LmHeadPath::Bf16 => LM_HEAD_PATH_TAG_BF16,
    }
}

/// Whether a dense BF16 projection `[M, K] @ [N, K]^T -> [M, N]` satisfies
/// cuBLASLt's non-null 16-byte BF16 alignment requirements (`N % 8 == 0`,
/// `K % 8 == 0`, and 16-byte aligned operand/output device pointers).
#[inline]
#[must_use]
pub(super) fn bf16_proj_cublas_aligned(
    act: DevicePtr,
    weight: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> bool {
    m > 0
        && n > 0
        && k > 0
        && n.is_multiple_of(8)
        && k.is_multiple_of(8)
        && !act.is_null()
        && !weight.is_null()
        && !out.is_null()
        && act.0.is_multiple_of(16)
        && weight.0.is_multiple_of(16)
        && out.0.is_multiple_of(16)
}

pub(super) fn resolve_lm_head_path_tag(
    nvfp4_requested: bool,
    nvfp4_weight_present: bool,
    nvfp4_kernel_present: bool,
    fp8_weight_present: bool,
) -> Result<u8> {
    let path = resolve_lm_head_path(
        nvfp4_requested,
        nvfp4_weight_present,
        nvfp4_kernel_present,
        fp8_weight_present,
    )
    .map_err(anyhow::Error::msg)?;
    Ok(lm_head_path_tag(path))
}

/// Representation currently stored in a sequence's `dstate.ctx_k_cache`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CtxKvCacheRepr {
    /// Eager `forward_block_layer` format: pre-`k_norm`, pre-RoPE ring buffer.
    PreRope,
    /// Stage-1 / batched DFlash2 format: circular `abs_pos % ctx_window`
    /// post-`k_norm`, post-RoPE ring buffer.
    PostRope,
}

/// Why a sequence's `ctx_{k,v}_cache` range counters must be invalidated before
/// reading cached entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CtxCacheInvalidationReason {
    None,
    RingWrapAround,
    ReprTransition {
        from: CtxKvCacheRepr,
        to: CtxKvCacheRepr,
    },
}

/// Pure transition planner for per-sequence `ctx_k_cache` representation and
/// invalidation.
///
/// Circular `PostRope` caches stay hot across the `ctx_window` wrap boundary
/// whenever `want_post_rope` remains true. Whenever a sequence transitions
/// between `PostRope` and eager `PreRope` (e.g. a non-ring-capable model
/// wrapping past `ctx_window` or an eager fallback), any non-empty cached rows
/// MUST be invalidated so eager `forward_block_layer` never applies `k_norm`
/// and RoPE twice and Stage 1 never mixes pre-RoPE rows into `PostRope` slots.
pub(crate) fn plan_ctx_kv_cache_transition(
    prev_repr: Option<CtxKvCacheRepr>,
    want_post_rope: bool,
    needed_start: usize,
    cache_end: usize,
) -> (CtxKvCacheRepr, CtxCacheInvalidationReason) {
    let target = if want_post_rope {
        CtxKvCacheRepr::PostRope
    } else {
        CtxKvCacheRepr::PreRope
    };
    let reason = match prev_repr {
        Some(from) if from != target && cache_end > 0 => {
            if from == CtxKvCacheRepr::PostRope && needed_start > 0 {
                CtxCacheInvalidationReason::RingWrapAround
            } else {
                CtxCacheInvalidationReason::ReprTransition { from, to: target }
            }
        }
        _ => CtxCacheInvalidationReason::None,
    };
    (target, reason)
}

/// SM-count threshold above which `ATLAS_DFLASH_NOISE_GRAPH` auto-enables when
/// unset (`<= 64` keeps 48-SM GB10 off by default). Allowed by release-flat
/// contract (`cli_dflash_environment_rules.rs`: `ZERO_OR_ONE`).
const AUTO_DFLASH_NOISE_GRAPH_SM_THRESHOLD: u32 = 64;

/// Strict parser for `ATLAS_DFLASH_NOISE_GRAPH` with SM-aware default.
///
/// Explicit `"0"` -> `Ok(false)`, `"1"` -> `Ok(true)`, any other value returns
/// an error (fail-closed off). When unset (`None`), auto-enables on wide GPUs
/// (`sm_count > 64`) and stays off on `<= 64`-SM GPUs (GB10).
pub fn parse_dflash_noise_graph_env_for_sm(raw: Option<&str>, sm_count: u32) -> Result<bool> {
    match raw.map(str::trim) {
        None => Ok(sm_count > AUTO_DFLASH_NOISE_GRAPH_SM_THRESHOLD),
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => {
            anyhow::bail!(
                "invalid {DFLASH_NOISE_GRAPH_SELECTOR}={other:?}: expected \"0\" or \"1\""
            )
        }
    }
}

/// Strict parser for `ATLAS_DFLASH_NOISE_GRAPH`. Unset or `"0"` -> `Ok(false)`;
/// `"1"` -> `Ok(true)`; any other value returns an error (fail-closed off).
#[cfg(test)]
pub fn parse_dflash_noise_graph_env(raw: Option<&str>) -> Result<bool> {
    parse_dflash_noise_graph_env_for_sm(raw, 0)
}

pub(super) fn dflash_noise_graph_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let raw = std::env::var(DFLASH_NOISE_GRAPH_SELECTOR).ok();
        let sm_count = atlas_core::device::active_sm_count();
        match parse_dflash_noise_graph_env_for_sm(raw.as_deref(), sm_count) {
            Ok(on) => {
                if on {
                    let source = if raw.is_some() { "env" } else { "auto_wide_sm" };
                    tracing::info!(
                        "DFLASH_NOISE_GRAPH_ENGAGED selector={DFLASH_NOISE_GRAPH_SELECTOR}=1 \
                         source={source} sm_count={sm_count}"
                    );
                }
                on
            }
            Err(err) => {
                tracing::warn!(
                    "DFLASH_NOISE_GRAPH_REFUSED reason=invalid_selector ({err}); falling back to eager"
                );
                false
            }
        }
    })
}

pub(super) fn noise_graph_swa_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let raw = std::env::var("ATLAS_DFLASH_SWA").ok();
        !matches!(
            raw.as_deref(),
            Some("0" | "false" | "FALSE" | "off" | "OFF")
        )
    })
}

pub(super) fn noise_graph_attn_kgamma_disable_o() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_ATTN_KGAMMA_DISABLE_O")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// Fixed byte offsets for the C=1 `CircularRingH128` Stage-2 device metadata
/// slab (`[ring_start: u32, _pad: u32, ctx_k_ptrs: [u64; L], ctx_v_ptrs: [u64; L]]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RingMetaLayout {
    num_layers: usize,
    ring_start_off: usize,
    ctx_k_ptrs_off: usize,
    ctx_v_ptrs_off: usize,
    total_bytes: usize,
}

impl RingMetaLayout {
    fn new(num_layers: usize) -> Result<Self> {
        ensure!(
            num_layers > 0,
            "DFlash C=1 ring metadata requires num_layers > 0"
        );
        let ptr_bytes = num_layers
            .checked_mul(std::mem::size_of::<u64>())
            .context("DFlash C=1 ring metadata pointer bytes overflow")?;
        let ring_start_off = 0usize;
        // Keep u64 pointer arrays 8-byte aligned right after ring_start (4B) + pad (4B).
        let ctx_k_ptrs_off = 8usize;
        let ctx_v_ptrs_off = ctx_k_ptrs_off
            .checked_add(ptr_bytes)
            .context("DFlash C=1 ring metadata V pointer offset overflow")?;
        let total_bytes = ctx_v_ptrs_off
            .checked_add(ptr_bytes)
            .context("DFlash C=1 ring metadata total bytes overflow")?;
        Ok(Self {
            num_layers,
            ring_start_off,
            ctx_k_ptrs_off,
            ctx_v_ptrs_off,
            total_bytes,
        })
    }

    #[inline]
    fn bind_device(&self, base: DevicePtr) -> RingMetaDevicePtrs {
        RingMetaDevicePtrs {
            base,
            ring_start_dev: base.offset(self.ring_start_off),
            ctx_k_ptrs_base: base.offset(self.ctx_k_ptrs_off),
            ctx_v_ptrs_base: base.offset(self.ctx_v_ptrs_off),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RingMetaDevicePtrs {
    base: DevicePtr,
    ring_start_dev: DevicePtr,
    ctx_k_ptrs_base: DevicePtr,
    ctx_v_ptrs_base: DevicePtr,
}

impl RingMetaDevicePtrs {
    #[inline]
    fn layer_k_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.ctx_k_ptrs_base
            .offset(layer_idx * std::mem::size_of::<u64>())
    }

    #[inline]
    fn layer_v_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.ctx_v_ptrs_base
            .offset(layer_idx * std::mem::size_of::<u64>())
    }
}

/// Environment selector for C=1 DFlash2 Stage-1 CUDA graph capture
/// (`5x {k_proj, v_proj, k_norm, rope}` bucketed by `new_ctx_count in 1..=8`).
/// Default ON unless `"0"` / `"off"` / `"false"`.
const DFLASH_STAGE1_GRAPH_ENV: &str = "ATLAS_DFLASH_STAGE1_GRAPH";
const STAGE1_GRAPH_MAX_ROWS: usize = 8;

#[must_use]
fn parse_dflash_stage1_graph_enabled(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(str::trim),
        Some(v) if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")
    )
}

#[must_use]
fn dflash_stage1_graph_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_dflash_stage1_graph_enabled(std::env::var(DFLASH_STAGE1_GRAPH_ENV).ok().as_deref())
    })
}

/// Contiguous C=1 Stage-2 metadata bundle layout (`216 B` for 5 layers) collapsing
/// `DflashNoiseGraphMeta` (80 B) + 6 Gumbel scalars + `q_row_base` + `RingMeta`
/// from 9 separate H2D calls into 1 single pinned H2D per round:
/// - `[0..80)`: `DflashNoiseGraphMeta`
/// - `[80..128)`: Gumbel scalars (`seeds:8, step0_pos:8, temps:4, top_ks:4, top_ps:4, min_ps:4, q_row_base:4`, padded to 16-byte alignment)
/// - `[128..128+ring_bytes)`: `RingMetaLayout` (`ring_start` + layer K/V pointers)
const C1_BUNDLE_META_OFF: usize = 0;
const C1_BUNDLE_GUMBEL_OFF: usize = 80;
const C1_BUNDLE_RING_OFF: usize = 128;

const _: () = {
    assert!(DflashNoiseGraphMeta::SIZE_BYTES == C1_BUNDLE_GUMBEL_OFF);
    assert!(C1_BUNDLE_GUMBEL_OFF.is_multiple_of(16));
    assert!(C1_BUNDLE_RING_OFF.is_multiple_of(16));
    assert!(C1_BUNDLE_GUMBEL_OFF + 36 <= C1_BUNDLE_RING_OFF);
};

#[derive(Clone, Copy)]
struct C1Stage2BundlePtrs {
    meta_dev: DevicePtr,
    mask_dev: DevicePtr,
    gumbel_ptrs: GumbelScratchPtrs,
    q_row_base_dev: DevicePtr,
    ring_meta_ptrs: Option<RingMetaDevicePtrs>,
}

struct NoiseGraphHeadState {
    split_ctx_kernel: Option<KernelHandle>,
    split_ctx_splitkv_kernel: Option<KernelHandle>,
    split_ctx_batched_splitkv_kernel: Option<KernelHandle>,
    split_ctx_splitkv_reduce_kernel: Option<KernelHandle>,
    split_kv_ws_dev: DevicePtr,
    split_kv_ws_cap: usize,
    meta_dev: DevicePtr,
    meta_pinned: Option<PinnedHostBuffer>,
    meta_slot: usize,
    #[cfg(test)]
    ring_meta_dev: DevicePtr,
    #[cfg(test)]
    ring_meta_pinned: Option<PinnedHostBuffer>,
    #[cfg(test)]
    ring_meta_slot: usize,
    #[cfg(test)]
    ring_meta_layers: usize,
    #[cfg(test)]
    ring_meta_slot_bytes: usize,
    /// Coalesced C=1 Stage-2 metadata bundle (`meta + gumbel + q_row_base + ring_meta`).
    c1_bundle_dev: DevicePtr,
    c1_bundle_pinned: Option<PinnedHostBuffer>,
    c1_bundle_slot: usize,
    c1_bundle_layers: usize,
    c1_bundle_slot_bytes: usize,
    gumbel_pinned: Option<PinnedHostBuffer>,
    gumbel_slot: usize,
    mask_tokens_dev: DevicePtr,
    mask_tokens_pinned: Option<PinnedHostBuffer>,
    uploaded_mask_id: Option<u32>,
    /// Fixed staging buffers + bucketed CUDA graphs (`new_ctx_count in 1..=8`)
    /// for Stage 1 (`5x {k_proj, v_proj, k_norm, rope}`).
    stage1_fc_dev: DevicePtr,
    stage1_pos_dev: DevicePtr,
    stage1_pos_pinned: Option<PinnedHostBuffer>,
    stage1_pos_slot: usize,
    stage1_k_dev: DevicePtr,
    stage1_v_dev: DevicePtr,
    stage1_cfg: Option<(usize, usize, usize)>,
    stage1_graphs: HashMap<(usize, u64), GraphHandle>,
    stage1_capture_poisoned: bool,
    graphs: HashMap<NoiseGraphRuntimeKey, GraphHandle>,
    graph_order: Vec<NoiseGraphRuntimeKey>,
}

impl NoiseGraphHeadState {
    fn new() -> Self {
        Self {
            split_ctx_kernel: None,
            split_ctx_splitkv_kernel: None,
            split_ctx_batched_splitkv_kernel: None,
            split_ctx_splitkv_reduce_kernel: None,
            split_kv_ws_dev: DevicePtr::NULL,
            split_kv_ws_cap: 0,
            meta_dev: DevicePtr::NULL,
            meta_pinned: None,
            meta_slot: 0,
            #[cfg(test)]
            ring_meta_dev: DevicePtr::NULL,
            #[cfg(test)]
            ring_meta_pinned: None,
            #[cfg(test)]
            ring_meta_slot: 0,
            #[cfg(test)]
            ring_meta_layers: 0,
            #[cfg(test)]
            ring_meta_slot_bytes: 0,
            c1_bundle_dev: DevicePtr::NULL,
            c1_bundle_pinned: None,
            c1_bundle_slot: 0,
            c1_bundle_layers: 0,
            c1_bundle_slot_bytes: 0,
            gumbel_pinned: None,
            gumbel_slot: 0,
            mask_tokens_dev: DevicePtr::NULL,
            mask_tokens_pinned: None,
            uploaded_mask_id: None,
            stage1_fc_dev: DevicePtr::NULL,
            stage1_pos_dev: DevicePtr::NULL,
            stage1_pos_pinned: None,
            stage1_pos_slot: 0,
            stage1_k_dev: DevicePtr::NULL,
            stage1_v_dev: DevicePtr::NULL,
            stage1_cfg: None,
            stage1_graphs: HashMap::new(),
            stage1_capture_poisoned: false,
            graphs: HashMap::new(),
            graph_order: Vec::new(),
        }
    }

    fn resolve_split_ctx_kernel(&mut self, gpu: &dyn GpuBackend, head_dim: usize) -> KernelHandle {
        if let Some(handle) = self.split_ctx_kernel {
            return handle;
        }
        let handle = match head_dim {
            128 => crate::layers::try_kernel(
                gpu,
                "inferspark_prefill_h128",
                "inferspark_prefill_split_ctx_h128",
            ),
            256 => crate::layers::try_kernel(
                gpu,
                "inferspark_prefill",
                "inferspark_prefill_split_ctx_h256",
            ),
            _ => KernelHandle(0),
        };
        self.split_ctx_kernel = Some(handle);
        handle
    }

    fn resolve_split_kv_h128_kernels(
        &mut self,
        gpu: &dyn GpuBackend,
        mode: Stage2CtxAttnMode,
    ) -> Option<(KernelHandle, KernelHandle)> {
        let split_k = match mode {
            Stage2CtxAttnMode::Linear => *self.split_ctx_splitkv_kernel.get_or_insert_with(|| {
                crate::layers::try_kernel(
                    gpu,
                    "inferspark_prefill_h128",
                    DFLASH_SPLIT_KV_H128_LINEAR_KERNEL,
                )
            }),
            Stage2CtxAttnMode::CircularRingH128 => *self
                .split_ctx_batched_splitkv_kernel
                .get_or_insert_with(|| {
                    crate::layers::try_kernel(
                        gpu,
                        "inferspark_prefill_h128",
                        DFLASH_SPLIT_KV_H128_BATCHED_KERNEL,
                    )
                }),
        };
        let reduce_k = *self.split_ctx_splitkv_reduce_kernel.get_or_insert_with(|| {
            crate::layers::try_kernel(
                gpu,
                "inferspark_prefill_h128",
                DFLASH_SPLIT_KV_H128_REDUCE_KERNEL,
            )
        });
        (split_k.0 != 0 && reduce_k.0 != 0).then_some((split_k, reduce_k))
    }

    fn ensure_split_kv_workspace(
        &mut self,
        gpu: &dyn GpuBackend,
        alloc_bytes: usize,
        needed_bytes: usize,
    ) -> Result<Option<DevicePtr>> {
        if needed_bytes == 0 {
            return Ok(None);
        }
        if self.split_kv_ws_dev.is_null() {
            let bytes = alloc_bytes.max(needed_bytes);
            self.split_kv_ws_dev = gpu.alloc(bytes)?;
            self.split_kv_ws_cap = bytes;
        }
        if self.split_kv_ws_cap >= needed_bytes && self.split_kv_ws_dev.0.is_multiple_of(16) {
            Ok(Some(self.split_kv_ws_dev))
        } else {
            Ok(None)
        }
    }

    fn ensure_device_buffers(&mut self, gpu: &dyn GpuBackend) -> Result<(DevicePtr, DevicePtr)> {
        if self.meta_dev.is_null() {
            self.meta_dev = gpu.alloc(DflashNoiseGraphMeta::SIZE_BYTES)?;
        }
        if self.meta_pinned.is_none() {
            self.meta_pinned = Some(
                gpu.alloc_host_pinned(DflashNoiseGraphMeta::SIZE_BYTES * NOISE_META_PINNED_SLOTS)?,
            );
        }
        if self.mask_tokens_dev.is_null() {
            self.mask_tokens_dev =
                gpu.alloc(DflashNoiseGraphMeta::MAX_NOISE_POSITIONS * std::mem::size_of::<u32>())?;
        }
        if self.mask_tokens_pinned.is_none() {
            self.mask_tokens_pinned = Some(gpu.alloc_host_pinned(
                DflashNoiseGraphMeta::MAX_NOISE_POSITIONS * std::mem::size_of::<u32>(),
            )?);
        }
        Ok((self.meta_dev, self.mask_tokens_dev))
    }

    #[cfg(test)]
    fn ensure_ring_meta_buffers(
        &mut self,
        gpu: &dyn GpuBackend,
        num_layers: usize,
    ) -> Result<RingMetaLayout> {
        let layout = RingMetaLayout::new(num_layers)?;
        if self.ring_meta_dev.is_null() {
            self.ring_meta_dev = gpu.alloc(layout.total_bytes)?;
            self.ring_meta_layers = layout.num_layers;
            self.ring_meta_slot_bytes = layout.total_bytes;
        } else {
            ensure!(
                self.ring_meta_layers == layout.num_layers
                    && layout.total_bytes <= self.ring_meta_slot_bytes,
                "DFlash C=1 ring metadata layer count changed: expected {}, got {}",
                self.ring_meta_layers,
                layout.num_layers
            );
        }
        if self.ring_meta_pinned.is_none() {
            self.ring_meta_pinned =
                Some(gpu.alloc_host_pinned(self.ring_meta_slot_bytes * NOISE_META_PINNED_SLOTS)?);
        }
        Ok(layout)
    }

    #[cfg(test)]
    fn upload_ring_meta(
        &mut self,
        gpu: &dyn GpuBackend,
        ring_start: u32,
        ctx_k_cache: &[DevicePtr],
        ctx_v_cache: &[DevicePtr],
        stream: u64,
    ) -> Result<RingMetaDevicePtrs> {
        ensure!(
            !ctx_k_cache.is_empty() && ctx_k_cache.len() == ctx_v_cache.len(),
            "DFlash C=1 ring metadata KV cache length mismatch: k={} v={}",
            ctx_k_cache.len(),
            ctx_v_cache.len()
        );
        let layout = self.ensure_ring_meta_buffers(gpu, ctx_k_cache.len())?;
        let slot = self.ring_meta_slot;
        self.ring_meta_slot = (self.ring_meta_slot + 1) % NOISE_META_PINNED_SLOTS;
        let offset = slot * self.ring_meta_slot_bytes;
        let pinned = self
            .ring_meta_pinned
            .as_mut()
            .context("DFlash C=1 ring metadata staging buffer missing")?;
        let dst = &mut pinned.as_mut_slice()[offset..offset + layout.total_bytes];
        dst[layout.ring_start_off..layout.ring_start_off + 4]
            .copy_from_slice(&ring_start.to_le_bytes());
        dst[layout.ring_start_off + 4..layout.ctx_k_ptrs_off].copy_from_slice(&0u32.to_le_bytes());
        for (l, (&k_ptr, &v_ptr)) in ctx_k_cache.iter().zip(ctx_v_cache.iter()).enumerate() {
            let k_off = layout.ctx_k_ptrs_off + l * std::mem::size_of::<u64>();
            let v_off = layout.ctx_v_ptrs_off + l * std::mem::size_of::<u64>();
            dst[k_off..k_off + 8].copy_from_slice(&k_ptr.0.to_le_bytes());
            dst[v_off..v_off + 8].copy_from_slice(&v_ptr.0.to_le_bytes());
        }
        let slice = pinned.pinned_slice_range(offset, layout.total_bytes)?;
        // SAFETY: `ring_meta_pinned` is page-locked and owned by the static
        // per-head state; rotating across `NOISE_META_PINNED_SLOTS` disjoint
        // slots ensures no in-flight async H2D transfer is overwritten.
        unsafe {
            gpu.copy_h2d_pinned_async(slice, self.ring_meta_dev, stream)?;
        }
        Ok(layout.bind_device(self.ring_meta_dev))
    }

    #[allow(dead_code)]
    fn upload_meta_and_masks(
        &mut self,
        gpu: &dyn GpuBackend,
        meta: &DflashNoiseGraphMeta,
        mask_id: u32,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let (meta_dev, mask_dev) = self.ensure_device_buffers(gpu)?;
        self.ensure_mask_tokens_uploaded(gpu, mask_dev, mask_id, stream)?;
        let slot = self.meta_slot;
        self.meta_slot = (self.meta_slot + 1) % NOISE_META_PINNED_SLOTS;
        let offset = slot * DflashNoiseGraphMeta::SIZE_BYTES;
        let meta_pinned = self
            .meta_pinned
            .as_mut()
            .context("DFlash noise graph meta staging buffer missing")?;
        meta.write_le_bytes(
            &mut meta_pinned.as_mut_slice()[offset..offset + DflashNoiseGraphMeta::SIZE_BYTES],
        )?;
        let slice = meta_pinned.pinned_slice_range(offset, DflashNoiseGraphMeta::SIZE_BYTES)?;
        // SAFETY: `meta_pinned` is page-locked and owned by the static per-head
        // state; rotating across `NOISE_META_PINNED_SLOTS` disjoint 80-byte
        // ranges ensures no in-flight async H2D transfer is overwritten.
        unsafe {
            gpu.copy_h2d_pinned_async(slice, meta_dev, stream)?;
        }
        Ok((meta_dev, mask_dev))
    }

    fn ensure_mask_tokens_uploaded(
        &mut self,
        gpu: &dyn GpuBackend,
        mask_dev: DevicePtr,
        mask_id: u32,
        stream: u64,
    ) -> Result<()> {
        if self.uploaded_mask_id != Some(mask_id) {
            if self.uploaded_mask_id.is_some() {
                // Single-slot page-locked mask buffer: drain prior async H2D
                // transfers across streams before mutating `mask_tokens_pinned`
                // in place for a new `mask_id`.
                gpu.synchronize_device()?;
            }
            let mask_pinned = self
                .mask_tokens_pinned
                .as_mut()
                .context("DFlash noise graph mask staging buffer missing")?;
            let mask_bytes = DflashNoiseGraphMeta::MAX_NOISE_POSITIONS * std::mem::size_of::<u32>();
            for chunk in mask_pinned.as_mut_slice()[..mask_bytes].chunks_exact_mut(4) {
                chunk.copy_from_slice(&mask_id.to_le_bytes());
            }
            let slice = mask_pinned.pinned_slice(mask_bytes)?;
            // SAFETY: `mask_tokens_pinned` is page-locked, owned by the static
            // per-head state, and any prior H2D copy was drained above before
            // mutating the single-slot buffer.
            unsafe {
                gpu.copy_h2d_pinned_async(slice, mask_dev, stream)?;
            }
            self.uploaded_mask_id = Some(mask_id);
        }
        Ok(())
    }

    fn ensure_c1_bundle_buffers(
        &mut self,
        gpu: &dyn GpuBackend,
        num_layers: usize,
    ) -> Result<(DevicePtr, DevicePtr, RingMetaLayout)> {
        let (_, mask_dev) = self.ensure_device_buffers(gpu)?;
        let ring_layout = RingMetaLayout::new(num_layers.max(1))?;
        let slot_bytes = C1_BUNDLE_RING_OFF
            .checked_add(ring_layout.total_bytes)
            .context("DFlash C=1 metadata bundle size overflow")?;
        if self.c1_bundle_dev.is_null() {
            self.c1_bundle_dev = gpu.alloc(slot_bytes)?;
            self.c1_bundle_layers = num_layers.max(1);
            self.c1_bundle_slot_bytes = slot_bytes;
        } else {
            ensure!(
                self.c1_bundle_layers == num_layers.max(1)
                    && slot_bytes <= self.c1_bundle_slot_bytes,
                "DFlash C=1 metadata bundle layer count changed: expected {}, got {}",
                self.c1_bundle_layers,
                num_layers.max(1)
            );
        }
        if self.c1_bundle_pinned.is_none() {
            self.c1_bundle_pinned =
                Some(gpu.alloc_host_pinned(self.c1_bundle_slot_bytes * NOISE_META_PINNED_SLOTS)?);
        }
        Ok((self.c1_bundle_dev, mask_dev, ring_layout))
    }

    /// Pack all per-round C=1 Stage-2 metadata (`DflashNoiseGraphMeta`, 6 Gumbel
    /// scalars, `q_row_base`, and optional `RingMeta` start + K/V pointers) into
    /// ONE contiguous rotating pinned slot and issue ONE single `copy_h2d_pinned_async`.
    #[allow(clippy::too_many_arguments)]
    fn upload_c1_stage2_bundle(
        &mut self,
        gpu: &dyn GpuBackend,
        meta: &DflashNoiseGraphMeta,
        mask_id: u32,
        ring_upload: Option<(u32, &[DevicePtr], &[DevicePtr])>,
        gumbel_spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
        q_row_base: u32,
        num_layers: usize,
        stream: u64,
    ) -> Result<C1Stage2BundlePtrs> {
        let (bundle_dev, mask_dev, ring_layout) = self.ensure_c1_bundle_buffers(gpu, num_layers)?;
        self.ensure_mask_tokens_uploaded(gpu, mask_dev, mask_id, stream)?;

        let slot = self.c1_bundle_slot;
        self.c1_bundle_slot = (self.c1_bundle_slot + 1) % NOISE_META_PINNED_SLOTS;
        let slot_off = slot * self.c1_bundle_slot_bytes;
        let (seed, step0_abs_pos, temp, top_k, top_p, min_p) = gumbel_spec_fields(gumbel_spec);

        let upload_bytes = if ring_upload.is_some() {
            C1_BUNDLE_RING_OFF + ring_layout.total_bytes
        } else {
            C1_BUNDLE_RING_OFF
        };
        let pinned = self
            .c1_bundle_pinned
            .as_mut()
            .context("DFlash C=1 metadata bundle pinned staging missing")?;
        let dst = &mut pinned.as_mut_slice()[slot_off..slot_off + upload_bytes];
        meta.write_le_bytes(
            &mut dst[C1_BUNDLE_META_OFF..C1_BUNDLE_META_OFF + DflashNoiseGraphMeta::SIZE_BYTES],
        )?;
        let g = &mut dst[C1_BUNDLE_GUMBEL_OFF..C1_BUNDLE_RING_OFF];
        g[0..8].copy_from_slice(&seed.to_le_bytes());
        g[8..16].copy_from_slice(&step0_abs_pos.to_le_bytes());
        g[16..20].copy_from_slice(&temp.to_le_bytes());
        g[20..24].copy_from_slice(&top_k.to_le_bytes());
        g[24..28].copy_from_slice(&top_p.to_le_bytes());
        g[28..32].copy_from_slice(&min_p.to_le_bytes());
        g[32..36].copy_from_slice(&q_row_base.to_le_bytes());
        g[36..48].fill(0);

        let ring_meta_ptrs = if let Some((ring_start, ctx_k_cache, ctx_v_cache)) = ring_upload {
            ensure!(
                !ctx_k_cache.is_empty()
                    && ctx_k_cache.len() == ctx_v_cache.len()
                    && ctx_k_cache.len() == ring_layout.num_layers,
                "DFlash C=1 bundle ring KV length mismatch: k={} v={} layout={}",
                ctx_k_cache.len(),
                ctx_v_cache.len(),
                ring_layout.num_layers
            );
            let r = &mut dst[C1_BUNDLE_RING_OFF..C1_BUNDLE_RING_OFF + ring_layout.total_bytes];
            r[ring_layout.ring_start_off..ring_layout.ring_start_off + 4]
                .copy_from_slice(&ring_start.to_le_bytes());
            r[ring_layout.ring_start_off + 4..ring_layout.ctx_k_ptrs_off]
                .copy_from_slice(&0u32.to_le_bytes());
            for (l, (&k_ptr, &v_ptr)) in ctx_k_cache.iter().zip(ctx_v_cache.iter()).enumerate() {
                let k_off = ring_layout.ctx_k_ptrs_off + l * std::mem::size_of::<u64>();
                let v_off = ring_layout.ctx_v_ptrs_off + l * std::mem::size_of::<u64>();
                r[k_off..k_off + 8].copy_from_slice(&k_ptr.0.to_le_bytes());
                r[v_off..v_off + 8].copy_from_slice(&v_ptr.0.to_le_bytes());
            }
            Some(ring_layout.bind_device(bundle_dev.offset(C1_BUNDLE_RING_OFF)))
        } else {
            None
        };

        let slice = pinned.pinned_slice_range(slot_off, upload_bytes)?;
        // SAFETY: `c1_bundle_pinned` is page-locked and owned by the static
        // per-head state; rotating across `NOISE_META_PINNED_SLOTS` disjoint
        // ranges ensures no in-flight async H2D transfer is overwritten.
        unsafe {
            gpu.copy_h2d_pinned_async(slice, bundle_dev, stream)?;
        }
        let g_base = bundle_dev.offset(C1_BUNDLE_GUMBEL_OFF);
        Ok(C1Stage2BundlePtrs {
            meta_dev: bundle_dev.offset(C1_BUNDLE_META_OFF),
            mask_dev,
            gumbel_ptrs: GumbelScratchPtrs {
                seeds: g_base,
                step0_pos: g_base.offset(8),
                temps: g_base.offset(16),
                top_ks: g_base.offset(20),
                top_ps: g_base.offset(24),
                min_ps: g_base.offset(28),
            },
            q_row_base_dev: g_base.offset(32),
            ring_meta_ptrs,
        })
    }

    fn ensure_stage1_buffers(
        &mut self,
        gpu: &dyn GpuBackend,
        num_layers: usize,
        hidden: usize,
        kv_dim: usize,
    ) -> Result<(DevicePtr, DevicePtr, DevicePtr, DevicePtr)> {
        let cfg = (num_layers, hidden, kv_dim);
        if let Some(prev) = self.stage1_cfg {
            ensure!(
                prev == cfg,
                "DFlash Stage-1 buffer geometry changed: expected {prev:?}, got {cfg:?}"
            );
            return Ok((
                self.stage1_fc_dev,
                self.stage1_pos_dev,
                self.stage1_k_dev,
                self.stage1_v_dev,
            ));
        }
        let bf16 = 2usize;
        let fc_bytes = STAGE1_GRAPH_MAX_ROWS * hidden * bf16;
        let pos_bytes = STAGE1_GRAPH_MAX_ROWS * std::mem::size_of::<u32>();
        let kv_bytes = num_layers * STAGE1_GRAPH_MAX_ROWS * kv_dim * bf16;
        let fc_dev = gpu.alloc(fc_bytes)?;
        let pos_dev = match gpu.alloc(pos_bytes) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.free(fc_dev);
                return Err(e);
            }
        };
        let k_dev = match gpu.alloc(kv_bytes) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.free(pos_dev);
                let _ = gpu.free(fc_dev);
                return Err(e);
            }
        };
        let v_dev = match gpu.alloc(kv_bytes) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.free(k_dev);
                let _ = gpu.free(pos_dev);
                let _ = gpu.free(fc_dev);
                return Err(e);
            }
        };
        let pos_pinned = match gpu.alloc_host_pinned(pos_bytes * NOISE_META_PINNED_SLOTS) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.free(v_dev);
                let _ = gpu.free(k_dev);
                let _ = gpu.free(pos_dev);
                let _ = gpu.free(fc_dev);
                return Err(e);
            }
        };
        self.stage1_fc_dev = fc_dev;
        self.stage1_pos_dev = pos_dev;
        self.stage1_pos_pinned = Some(pos_pinned);
        self.stage1_k_dev = k_dev;
        self.stage1_v_dev = v_dev;
        self.stage1_cfg = Some(cfg);
        Ok((fc_dev, pos_dev, k_dev, v_dev))
    }

    fn upload_stage1_positions(
        &mut self,
        gpu: &dyn GpuBackend,
        start_abs_pos: usize,
        count: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (1..=STAGE1_GRAPH_MAX_ROWS).contains(&count),
            "DFlash Stage-1 position count {count} outside 1..={STAGE1_GRAPH_MAX_ROWS}"
        );
        let slot_bytes = STAGE1_GRAPH_MAX_ROWS * std::mem::size_of::<u32>();
        let slot = self.stage1_pos_slot;
        self.stage1_pos_slot = (self.stage1_pos_slot + 1) % NOISE_META_PINNED_SLOTS;
        let slot_off = slot * slot_bytes;
        let bytes = count * std::mem::size_of::<u32>();
        let pinned = self
            .stage1_pos_pinned
            .as_mut()
            .context("DFlash Stage-1 position staging buffer missing")?;
        let dst = &mut pinned.as_mut_slice()[slot_off..slot_off + bytes];
        for i in 0..count {
            let pos = u32::try_from(start_abs_pos + i)
                .context("DFlash Stage-1 absolute position overflow u32")?;
            dst[i * 4..i * 4 + 4].copy_from_slice(&pos.to_le_bytes());
        }
        let slice = pinned.pinned_slice_range(slot_off, bytes)?;
        // SAFETY: `stage1_pos_pinned` is page-locked and owned by the static per-head
        // state; rotating across `NOISE_META_PINNED_SLOTS` disjoint slots ensures no
        // in-flight async H2D transfer is overwritten.
        unsafe {
            gpu.copy_h2d_pinned_async(slice, self.stage1_pos_dev, stream)?;
        }
        Ok(())
    }

    fn upload_gumbel_meta_c1(
        &mut self,
        gpu: &dyn GpuBackend,
        spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
        ptrs: GumbelScratchPtrs,
        anchor_upload: Option<(u32, DevicePtr)>,
        stream: u64,
    ) -> Result<()> {
        self.upload_gumbel_meta_c1_with_q_base(gpu, spec, ptrs, anchor_upload, None, stream)
    }

    fn upload_gumbel_meta_c1_with_q_base(
        &mut self,
        gpu: &dyn GpuBackend,
        spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
        ptrs: GumbelScratchPtrs,
        anchor_upload: Option<(u32, DevicePtr)>,
        q_row_base_upload: Option<(u32, DevicePtr)>,
        stream: u64,
    ) -> Result<()> {
        if self.gumbel_pinned.is_none() {
            self.gumbel_pinned =
                Some(gpu.alloc_host_pinned(GUMBEL_META_C1_SLOT_BYTES * NOISE_META_PINNED_SLOTS)?);
        }
        let slot = self.gumbel_slot;
        self.gumbel_slot = (self.gumbel_slot + 1) % NOISE_META_PINNED_SLOTS;
        let base = slot * GUMBEL_META_C1_SLOT_BYTES;
        let (seed, step0_abs_pos, temp, top_k, top_p, min_p) = gumbel_spec_fields(spec);
        let anchor_id = anchor_upload.map_or(0u32, |(tok, _)| tok);
        let q_row_base = q_row_base_upload.map_or(0u32, |(row_base, _)| row_base);

        let gumbel_pinned = self
            .gumbel_pinned
            .as_mut()
            .context("DFlash C=1 Gumbel metadata staging buffer missing")?;
        let dst = &mut gumbel_pinned.as_mut_slice()[base..base + GUMBEL_META_C1_SLOT_BYTES];
        dst[GUMBEL_META_SEED_OFF..GUMBEL_META_SEED_OFF + 8].copy_from_slice(&seed.to_le_bytes());
        dst[GUMBEL_META_STEP0_OFF..GUMBEL_META_STEP0_OFF + 8]
            .copy_from_slice(&step0_abs_pos.to_le_bytes());
        dst[GUMBEL_META_TEMP_OFF..GUMBEL_META_TEMP_OFF + 4].copy_from_slice(&temp.to_le_bytes());
        dst[GUMBEL_META_TOPK_OFF..GUMBEL_META_TOPK_OFF + 4].copy_from_slice(&top_k.to_le_bytes());
        dst[GUMBEL_META_TOPP_OFF..GUMBEL_META_TOPP_OFF + 4].copy_from_slice(&top_p.to_le_bytes());
        dst[GUMBEL_META_MINP_OFF..GUMBEL_META_MINP_OFF + 4].copy_from_slice(&min_p.to_le_bytes());
        dst[GUMBEL_META_ANCHOR_OFF..GUMBEL_META_ANCHOR_OFF + 4]
            .copy_from_slice(&anchor_id.to_le_bytes());
        dst[GUMBEL_META_Q_ROW_BASE_OFF..GUMBEL_META_Q_ROW_BASE_OFF + 4]
            .copy_from_slice(&q_row_base.to_le_bytes());

        for (field_off, len, dev_ptr) in [
            (GUMBEL_META_SEED_OFF, 8usize, ptrs.seeds),
            (GUMBEL_META_STEP0_OFF, 8usize, ptrs.step0_pos),
            (GUMBEL_META_TEMP_OFF, 4usize, ptrs.temps),
            (GUMBEL_META_TOPK_OFF, 4usize, ptrs.top_ks),
            (GUMBEL_META_TOPP_OFF, 4usize, ptrs.top_ps),
            (GUMBEL_META_MINP_OFF, 4usize, ptrs.min_ps),
        ] {
            let slice = gumbel_pinned.pinned_slice_range(base + field_off, len)?;
            // SAFETY: `gumbel_pinned` is page-locked and owned by the static
            // per-head state; rotating across `NOISE_META_PINNED_SLOTS` disjoint
            // slots ensures no in-flight async H2D transfer is overwritten.
            unsafe {
                gpu.copy_h2d_pinned_async(slice, dev_ptr, stream)?;
            }
        }
        if let Some((_, anchor_dev)) = anchor_upload {
            let slice = gumbel_pinned.pinned_slice_range(base + GUMBEL_META_ANCHOR_OFF, 4)?;
            // SAFETY: same rotating page-locked slot invariant as above.
            unsafe {
                gpu.copy_h2d_pinned_async(slice, anchor_dev, stream)?;
            }
        }
        if let Some((_, q_row_base_dev)) = q_row_base_upload {
            let slice = gumbel_pinned.pinned_slice_range(base + GUMBEL_META_Q_ROW_BASE_OFF, 4)?;
            // SAFETY: same rotating page-locked slot invariant as above.
            unsafe {
                gpu.copy_h2d_pinned_async(slice, q_row_base_dev, stream)?;
            }
        }
        Ok(())
    }

    fn insert_graph(
        &mut self,
        gpu: &dyn GpuBackend,
        key: NoiseGraphRuntimeKey,
        graph: GraphHandle,
    ) {
        if let Some(old) = self.graphs.insert(key.clone(), graph)
            && old.0 != 0
        {
            let _ = gpu.destroy_graph(old);
        }
        self.graph_order.retain(|existing| existing != &key);
        self.graph_order.push(key);
        while self.graph_order.len() > MAX_CACHED_NOISE_GRAPHS {
            let evicted = self.graph_order.remove(0);
            if let Some(old) = self.graphs.remove(&evicted)
                && old.0 != 0
            {
                let _ = gpu.destroy_graph(old);
            }
        }
    }
}

type NoiseGraphRegistry = Mutex<HashMap<usize, Arc<Mutex<NoiseGraphHeadState>>>>;

static NOISE_GRAPH_REGISTRY: OnceLock<NoiseGraphRegistry> = OnceLock::new();

fn noise_graph_head_state(head_ptr: usize) -> Arc<Mutex<NoiseGraphHeadState>> {
    let map = NOISE_GRAPH_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard
        .entry(head_ptr)
        .or_insert_with(|| Arc::new(Mutex::new(NoiseGraphHeadState::new())))
        .clone()
}

/// Destroy all cached C=1 Stage-2 CUDA graphs and device metadata buffers for
/// `head_ptr` before the parent model frees shared rejection buffers in `Drop`.
pub(super) fn clear_noise_graph_head_state(head_ptr: usize, gpu: &dyn GpuBackend) {
    let Some(map) = NOISE_GRAPH_REGISTRY.get() else {
        return;
    };
    let entry = map
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&head_ptr);
    if let Some(head_state_arc) = entry {
        let mut state = head_state_arc
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.graph_order.clear();
        for (_, graph) in state.graphs.drain() {
            if graph.0 != 0 {
                let _ = gpu.destroy_graph(graph);
            }
        }
        for (_, graph) in state.stage1_graphs.drain() {
            if graph.0 != 0 {
                let _ = gpu.destroy_graph(graph);
            }
        }
        if !state.c1_bundle_dev.is_null() {
            let _ = gpu.free(state.c1_bundle_dev);
            state.c1_bundle_dev = DevicePtr::NULL;
        }
        state.c1_bundle_pinned = None;
        state.c1_bundle_layers = 0;
        state.c1_bundle_slot_bytes = 0;
        let s = &mut *state;
        for ptr in [
            &mut s.stage1_fc_dev,
            &mut s.stage1_pos_dev,
            &mut s.stage1_k_dev,
            &mut s.stage1_v_dev,
        ] {
            if !ptr.is_null() {
                let _ = gpu.free(*ptr);
                *ptr = DevicePtr::NULL;
            }
        }
        state.stage1_pos_pinned = None;
        state.stage1_cfg = None;
        state.stage1_capture_poisoned = false;
        if !state.meta_dev.is_null() {
            let _ = gpu.free(state.meta_dev);
            state.meta_dev = DevicePtr::NULL;
        }
        #[cfg(test)]
        {
            if !state.ring_meta_dev.is_null() {
                let _ = gpu.free(state.ring_meta_dev);
                state.ring_meta_dev = DevicePtr::NULL;
            }
            state.ring_meta_pinned = None;
            state.ring_meta_layers = 0;
            state.ring_meta_slot_bytes = 0;
        }
        if !state.mask_tokens_dev.is_null() {
            let _ = gpu.free(state.mask_tokens_dev);
            state.mask_tokens_dev = DevicePtr::NULL;
        }
        if !state.split_kv_ws_dev.is_null() {
            let _ = gpu.free(state.split_kv_ws_dev);
            state.split_kv_ws_dev = DevicePtr::NULL;
            state.split_kv_ws_cap = 0;
        }
    }
}

/// Remove any remaining `NOISE_GRAPH_REGISTRY` entry for `head_ptr` when
/// [`BlockDiffusionDraftHead`] drops, preventing stale pointer-keyed entries
/// across address reuse if the caller dropped the head without invoking
/// `release_rejection_resources`.
///
/// Returns `true` if a stale entry was present and removed.
pub(super) fn drop_noise_graph_head_entry(head_ptr: usize) -> bool {
    let Some(map) = NOISE_GRAPH_REGISTRY.get() else {
        return false;
    };
    let removed = map
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&head_ptr);
    removed.is_some()
}

pub(super) struct NoiseGraphCaptureGuard<'a>(
    crate::model::trait_impl::capture_guard::CaptureGuard<'a>,
);

impl<'a> NoiseGraphCaptureGuard<'a> {
    pub(super) fn begin(gpu: &'a dyn GpuBackend, stream: u64) -> Result<Self> {
        crate::model::trait_impl::capture_guard::CaptureGuard::begin(
            gpu,
            stream,
            "dflash_noise_graph",
        )
        .map(Self)
    }

    pub(super) fn end(self) -> Result<GraphHandle> {
        self.0.end()
    }
}

pub(super) fn invalidate_dstate_ctx_kv_ranges(dstate: &mut DflashProposerState) {
    dstate.cache_k_start.fill(0);
    dstate.cache_k_end.fill(0);
    dstate.cache_v_start.fill(0);
    dstate.cache_v_end.fill(0);
}

fn read_noise_positions_from_pinned(
    pos_pinned: &PinnedHostBuffer,
    eff_ctx: usize,
    noise_rows: usize,
) -> Result<Vec<u32>> {
    let start_byte = eff_ctx
        .checked_mul(std::mem::size_of::<u32>())
        .context("DFlash noise_positions byte offset overflow")?;
    let byte_len = noise_rows
        .checked_mul(std::mem::size_of::<u32>())
        .context("DFlash noise_positions byte length overflow")?;
    let end_byte = start_byte
        .checked_add(byte_len)
        .context("DFlash noise_positions byte end overflow")?;
    ensure!(
        end_byte <= pos_pinned.len(),
        "DFlash pinned position buffer too small: need {end_byte}, have {}",
        pos_pinned.len()
    );
    // SAFETY: `pos_pinned` is owned by `dstate`, `end_byte <= pos_pinned.len()`
    // was verified above, and this read-only borrow never mutates bytes being
    // read by the pending stream-ordered H2D transfer.
    let bytes =
        unsafe { std::slice::from_raw_parts(pos_pinned.as_ptr().add(start_byte), byte_len) };
    let mut positions = Vec::with_capacity(noise_rows);
    for chunk in bytes.chunks_exact(4) {
        positions.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(positions)
}

/// Whether stochastic draft sampling runs inside the noise pass itself and
/// therefore requires bypassing the deterministic Stage-2 CUDA graph.
///
/// Greedy drafting paired with target Gumbel verification (`temperature > 0.0`
/// on the target request while `draft_sample_method == Greedy`) keeps the
/// noise pass completely deterministic, so Stage-2 CUDA graph capture/replay
/// remains valid.
#[inline]
#[must_use]
fn noise_graph_stochastic_draft_sampling_active(
    draft_sample_method: crate::dflash_sampling::DflashDraftSampleMethod,
    active_sampling_request: Option<&crate::dflash_sampling::DflashDraftSamplingRequest>,
) -> bool {
    draft_sample_method == crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic
        && active_sampling_request.is_some_and(|request| request.request_sampler.temperature > 0.0)
}

#[derive(Clone, Copy)]
struct NoiseGraphDebugFlags {
    debug_dump: bool,
    force_noise_pattern: bool,
    dump_all_layers: bool,
    kprofile: bool,
}

#[derive(Clone, Copy)]
struct SplitKvH128Plan {
    split_kernel: KernelHandle,
    reduce_kernel: KernelHandle,
    workspace: DevicePtr,
    num_splits: u32,
}

#[derive(Clone, Copy)]
enum Stage2CtxAttnDispatch {
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
enum Stage2LayerCtxAttn {
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
    pub(super) fn layers_have_dflash2_conv(&self) -> bool {
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
    pub(super) fn noise_graph_can_skip_old_fc_copy(
        &self,
        dstate: &DflashProposerState,
        policy: &super::proposal_policy::ProposalPolicy,
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
    fn try_run_two_stage_noise_graph(
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
        let use_k32_graph_selector = !use_rejection_graph_selector
            && self.selector.is_some()
            && self.markov.is_none()
            && self.gumbel_selector_k32_ready();
        let use_gumbel_graph_selector = !use_rejection_graph_selector
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

        if (use_k32_selector || use_rejection_graph_selector) && self.markov.is_none() {
            dstate.pending_draft_probs_gamma = gamma_eff;
        }
        if use_rejection_graph_selector {
            dstate.set_last_rejection_q(valid_q_record);
        }

        if self.markov.is_some() {
            super::markov::mark_markov_completed_in_noise_graph(
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
                if super::proposal_policy::use_prefill_dual_pipe(
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
            .get_or_init(super::proposal_policy::ProposalPolicy::from_env)
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage_fp8_project(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &crate::weight_map::Fp8DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;
        if m == 1 && self.kernels.dense_gemv_fp8w.0 != 0 {
            ops::dense_gemv_fp8w(
                gpu,
                self.kernels.dense_gemv_fp8w,
                input,
                weight,
                output,
                n,
                k,
                stream,
            )
        } else {
            ops::fp8_gemm_dflash_rowscale(
                gpu,
                self.kernels.fp8_gemm_dflash,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        }
    }

    /// Execute a dense BF16 projection `out[M, N] = act[M, K] @ weight[N, K]^T`
    /// via cuBLASLt (`ops::cublas_bf16_proj_dense`), falling back to scalar
    /// `ops::dense_gemm` only when dimensions or pointers violate cuBLASLt's
    /// 16-byte BF16 alignment contract (e.g. odd vocabulary sizes).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_bf16_dense(
        &self,
        gpu: &dyn GpuBackend,
        act: DevicePtr,
        weight: &crate::weight_map::DenseWeight,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if bf16_proj_cublas_aligned(act, weight.weight, out, m, n, k) {
            crate::layers::ops::cublas_bf16_proj_dense(act, weight, out, m, n, k, stream)
        } else {
            crate::layers::ops::dense_gemm(
                gpu,
                self.kernels.dense_gemm,
                act,
                weight,
                out,
                m,
                n,
                k,
                stream,
            )
        }
    }

    /// Pre-warm cuBLASLt handle, 64 MB workspace, tuned plans (when
    /// `ATLAS_CUBLAS_TUNED=1`), and lazy kernel cubins for every BF16
    /// projection shape executed inside Stage-2 CUDA graph capture.
    ///
    /// Must run before `NoiseGraphCaptureGuard::begin` so `cublasLtCreate`,
    /// workspace `cuMemAlloc_v2`, and any first-use plan tuning never execute
    /// during stream capture. Every scratch output written here is dead prior
    /// to Stage 2 and overwritten before any Stage-2 consumer reads it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prewarm_stage2_cublas(
        &self,
        m_layer: u32,
        m_out: u32,
        h: u32,
        q_dim: u32,
        kv_dim: u32,
        inter: u32,
        bf16_lm_head_fallback: Option<(DevicePtr, u32)>,
        stream: u64,
    ) -> Result<()> {
        let mut warmed_conv_widths = Vec::new();
        let mut warmed_bf16_layer = false;
        for layer in &self.layers {
            let (attn_conv, mlp_conv) = match layer {
                DflashLayerQuantWeights::Bf16(l) => {
                    if !warmed_bf16_layer {
                        for (act, weight, out, n, k) in [
                            (
                                self.scratch.norm_buf,
                                &l.q_proj,
                                self.scratch.q_buf,
                                q_dim,
                                h,
                            ),
                            (
                                self.scratch.norm_buf,
                                &l.k_proj,
                                self.scratch.k_buf,
                                kv_dim,
                                h,
                            ),
                            (
                                self.scratch.attn_out,
                                &l.o_proj,
                                self.scratch.stream_acc,
                                h,
                                q_dim,
                            ),
                            (
                                self.scratch.norm_buf,
                                &l.gate_proj,
                                self.scratch.mlp_intermediate,
                                inter,
                                h,
                            ),
                            (
                                self.scratch.mlp_intermediate,
                                &l.down_proj,
                                self.scratch.stream_acc,
                                h,
                                inter,
                            ),
                        ] {
                            if bf16_proj_cublas_aligned(act, weight.weight, out, m_layer, n, k) {
                                crate::layers::ops::cublas_bf16_proj_dense(
                                    act, weight, out, m_layer, n, k, stream,
                                )?;
                            }
                        }
                        warmed_bf16_layer = true;
                    }
                    (&l.attention_conv, &l.mlp_conv)
                }
                DflashLayerQuantWeights::Fp8(l) => (&l.attention_conv, &l.mlp_conv),
                DflashLayerQuantWeights::Nvfp4(l) => (&l.attention_conv, &l.mlp_conv),
            };
            for conv in [attn_conv.as_ref(), mlp_conv.as_ref()]
                .into_iter()
                .flatten()
            {
                let dyn_width = (2 * conv.kernel_size * conv.groups) as u32;
                if !warmed_conv_widths.contains(&dyn_width) {
                    if bf16_proj_cublas_aligned(
                        self.scratch.norm_buf,
                        conv.kernel_projection.weight,
                        self.scratch.conv_dyn,
                        m_layer,
                        dyn_width,
                        h,
                    ) {
                        crate::layers::ops::cublas_bf16_proj_dense(
                            self.scratch.norm_buf,
                            &conv.kernel_projection,
                            self.scratch.conv_dyn,
                            m_layer,
                            dyn_width,
                            h,
                            stream,
                        )?;
                    }
                    warmed_conv_widths.push(dyn_width);
                }
            }
        }
        if let Some(sel) = &self.selector {
            let rank = sel.rank as u32;
            if bf16_proj_cublas_aligned(
                self.scratch.norm_buf,
                sel.hidden_projection.weight,
                self.scratch.selector_hidden,
                m_out,
                rank,
                h,
            ) {
                crate::layers::ops::cublas_bf16_proj_dense(
                    self.scratch.norm_buf,
                    &sel.hidden_projection,
                    self.scratch.selector_hidden,
                    m_out,
                    rank,
                    h,
                    stream,
                )?;
            }
        }
        if let Some((logits_out, lm_vocab)) = bf16_lm_head_fallback
            && bf16_proj_cublas_aligned(
                self.scratch.norm_buf,
                self.lm_head_shared,
                logits_out,
                m_out,
                lm_vocab,
                h,
            )
        {
            let lm_head = crate::weight_map::DenseWeight {
                weight: self.lm_head_shared,
            };
            crate::layers::ops::cublas_bf16_proj_dense(
                self.scratch.norm_buf,
                &lm_head,
                logits_out,
                m_out,
                lm_vocab,
                h,
                stream,
            )?;
        }
        Ok(())
    }

    fn stage2_conv_prepare_noise(
        &self,
        conv: &crate::weight_loader::DflashConvWeights,
        noise_count: u32,
        h: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<DevicePtr> {
        use crate::layers::ops;
        let dyn_width = (2 * conv.kernel_size * conv.groups) as u32;
        self.project_bf16_dense(
            gpu,
            self.scratch.norm_buf,
            &conv.kernel_projection,
            self.scratch.conv_dyn,
            noise_count,
            dyn_width,
            h,
            stream,
        )?;
        ops::dflash2_conv_prepare(
            gpu,
            self.kernels.dflash2_conv_prepare,
            self.scratch.norm_buf,
            self.scratch.conv_dyn,
            conv.base_kernel.weight,
            self.scratch.conv_out,
            self.scratch.conv_dyn1,
            noise_count,
            conv.groups as u32,
            stream,
        )?;
        Ok(self.scratch.conv_out)
    }

    fn stage2_conv_finish_noise(
        &self,
        conv: &crate::weight_loader::DflashConvWeights,
        noise_count: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<DevicePtr> {
        use crate::layers::ops;
        ops::dflash2_conv_finish(
            gpu,
            self.kernels.dflash2_conv_finish,
            self.scratch.stream_acc,
            self.scratch.conv_dyn1,
            conv.base_kernel.weight,
            self.scratch.conv_out,
            noise_count,
            conv.groups as u32,
            stream,
        )?;
        Ok(self.scratch.conv_out)
    }

    /// Stage 2: fixed-shape `m_rows` noise-block forward (CUDA-graph-capture-safe:
    /// zero host scalars that vary per step, zero `eff_ctx`-dependent offsets).
    #[allow(clippy::too_many_arguments)]
    fn launch_stage2_noise_block(
        &self,
        ctx_attn_dispatch: Stage2CtxAttnDispatch,
        meta_dev: DevicePtr,
        mask_dev: DevicePtr,
        gumbel_ptrs: GumbelScratchPtrs,
        q_row_base_dev: DevicePtr,
        gamma_eff: usize,
        row_layout: DraftRowLayout,
        lm_head_path: LmHeadPath,
        prefer_tc_lm_head: bool,
        shortlist_k: usize,
        use_radix_topk: bool,
        use_k32_selector: bool,
        rejection_mode: bool,
        dstate: &DflashProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;

        let gpu = ctx.gpu;
        let bf16 = 2usize;
        let m_rows = row_layout.query_rows as u32;
        let h = self.hidden_size as u32;
        let q_dim = (self.num_q_heads * self.head_dim) as u32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let inter = self.intermediate_size as u32;
        let inv_sqrt_d = 1.0f32 / (self.head_dim as f32).sqrt();
        let eff_ctx_dev = meta_dev.offset(DflashNoiseGraphMeta::EFF_CTX_OFFSET);
        let last_token_dev = meta_dev.offset(DflashNoiseGraphMeta::LAST_TOKEN_OFFSET);
        let noise_positions_dev = meta_dev.offset(DflashNoiseGraphMeta::NOISE_POSITIONS_OFFSET);
        let swa_on = noise_graph_swa_enabled();

        // 1. Noise embeddings at offset 0 of `stream_buf`:
        //    row 0 = embed[meta.last_token], rows 1..m_rows = embed[mask_id].
        ops::batched_embed(
            gpu,
            self.kernels.batched_embed,
            last_token_dev,
            self.embed_tokens_shared,
            self.scratch.stream_buf,
            1,
            h,
            stream,
        )?;
        if m_rows > 1 {
            ops::batched_embed(
                gpu,
                self.kernels.batched_embed,
                mask_dev,
                self.embed_tokens_shared,
                self.scratch.stream_buf.offset(self.hidden_size * bf16),
                m_rows - 1,
                h,
                stream,
            )?;
        }

        // 2. Fixed-shape m_rows drafter layers with split-KV attention.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_window = if swa_on {
                self.layer_window_sizes.get(layer_idx).copied().unwrap_or(0)
            } else {
                0
            };
            let layer_causal = if swa_on {
                self.layer_causal.get(layer_idx).copied().unwrap_or(false)
            } else {
                false
            };
            let layer_ctx_attn = match ctx_attn_dispatch {
                Stage2CtxAttnDispatch::Linear { kernel, split_kv } => Stage2LayerCtxAttn::Linear {
                    kernel,
                    ctx_k: dstate.ctx_k_cache[layer_idx],
                    ctx_v: dstate.ctx_v_cache[layer_idx],
                    eff_ctx_dev,
                    split_kv,
                },
                Stage2CtxAttnDispatch::CircularRingH128 {
                    kernel,
                    ring_meta,
                    ring_cap,
                    split_kv,
                } => Stage2LayerCtxAttn::CircularRingH128 {
                    kernel,
                    k_ctx_ptr_dev: ring_meta.layer_k_ptr(layer_idx),
                    v_ctx_ptr_dev: ring_meta.layer_v_ptr(layer_idx),
                    eff_ctx_dev,
                    ring_start_dev: ring_meta.ring_start_dev,
                    ring_cap,
                    split_kv,
                },
            };
            match layer {
                DflashLayerQuantWeights::Bf16(l) => self.stage2_layer_bf16(
                    l,
                    layer_ctx_attn,
                    noise_positions_dev,
                    m_rows,
                    h,
                    q_dim,
                    kv_dim,
                    inter,
                    inv_sqrt_d,
                    layer_causal,
                    layer_window,
                    gpu,
                    stream,
                )?,
                DflashLayerQuantWeights::Fp8(l) => self.stage2_layer_fp8(
                    l,
                    layer_ctx_attn,
                    noise_positions_dev,
                    m_rows,
                    h,
                    q_dim,
                    kv_dim,
                    inter,
                    inv_sqrt_d,
                    layer_causal,
                    layer_window,
                    gpu,
                    stream,
                )?,
                DflashLayerQuantWeights::Nvfp4(l) => self.stage2_layer_nvfp4(
                    l,
                    layer_ctx_attn,
                    noise_positions_dev,
                    m_rows,
                    h,
                    q_dim,
                    kv_dim,
                    inter,
                    inv_sqrt_d,
                    layer_causal,
                    layer_window,
                    gpu,
                    stream,
                )?,
            }
        }

        // 3. Final RMSNorm + LM head over compact noise output slice.
        let output_byte_offset = row_layout.output_start * self.hidden_size * bf16;
        let stream_noise = self.scratch.stream_buf.offset(output_byte_offset);
        let norm_noise = self.scratch.norm_buf.offset(output_byte_offset);
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
            stream_noise,
            lm_head_input_norm,
            norm_noise,
            gamma_eff as u32,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let lm_vocab = self.target_vocab_size.min(self.vocab_size) as u32;
        match lm_head_path {
            LmHeadPath::Nvfp4 => {
                self.launch_nvfp4_lm_head(
                    gpu,
                    norm_noise,
                    self.scratch.logits,
                    gamma_eff as u32,
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
                    norm_noise,
                    fp8.weight,
                    self.scratch.logits,
                    gamma_eff as u32,
                    lm_vocab,
                    h,
                    stream,
                )?;
            }
            LmHeadPath::Bf16 => {
                self.launch_lm_head_bf16_or_shared_nvfp4_exact(
                    gpu,
                    norm_noise,
                    self.scratch.logits,
                    gamma_eff,
                    lm_vocab,
                    h,
                    if prefer_tc_lm_head {
                        LmHeadNvfp4Mode::CoalescedTc
                    } else {
                        LmHeadNvfp4Mode::Exact
                    },
                    stream,
                )?;
            }
        }
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

        // 4. Tail: selector topk + hidden_projection (+ graphed walk when
        //    available), device-pointer Markov, or per-row argmax.
        let lm_stride = self.target_vocab_size.min(self.vocab_size);
        if let Some(sel) = &self.selector {
            if use_radix_topk {
                let (num_rows, vocab_u32, k_u32) = validate_selector_shortlist(
                    gamma_eff,
                    self.gamma,
                    lm_stride,
                    shortlist_k,
                    super::DFLASH2_GUMBEL_SHORTLIST_K,
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
                let layout = super::forward_block::checked_topk_layout(
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
            if self.markov.is_none() && rejection_mode {
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
                    Some(q_row_base_dev),
                    last_token_dev,
                    gumbel_ptrs.seeds,
                    gumbel_ptrs.step0_pos,
                    gumbel_ptrs.temps,
                    gumbel_ptrs.top_ks,
                    gumbel_ptrs.top_ps,
                    gumbel_ptrs.min_ps,
                    shortlist_k,
                    self.scratch.shared_q_stride,
                    gamma_eff,
                    sel.rank,
                    1,
                    stream,
                )?;
            } else if self.markov.is_none() && use_k32_selector {
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
                    last_token_dev,
                    gumbel_ptrs.seeds,
                    gumbel_ptrs.step0_pos,
                    gumbel_ptrs.temps,
                    gumbel_ptrs.top_ks,
                    gumbel_ptrs.top_ps,
                    gumbel_ptrs.min_ps,
                    shortlist_k,
                    gamma_eff,
                    sel.rank,
                    1,
                    stream,
                )?;
            } else if self.markov.is_none()
                && self.kernels.dflash2_selector_walk_gumbel_batched.0 != 0
                && self.gumbel_selector_scratch_ready()
            {
                ops::dflash2_selector_walk_gumbel_batched(
                    gpu,
                    self.kernels.dflash2_selector_walk_gumbel_batched,
                    self.scratch.topk_logits_dev,
                    self.scratch.topk_tokens_dev,
                    self.scratch.selector_hidden,
                    sel.predecessor_codebook.weight,
                    sel.successor_codebook.weight,
                    self.scratch.draft_tokens_dev,
                    last_token_dev,
                    gumbel_ptrs.seeds,
                    gumbel_ptrs.step0_pos,
                    gumbel_ptrs.temps,
                    gumbel_ptrs.top_ks,
                    gumbel_ptrs.top_ps,
                    gumbel_ptrs.min_ps,
                    gamma_eff,
                    sel.rank,
                    1,
                    stream,
                )?;
            } else if self.markov.is_none() && self.kernels.dflash2_selector_walk_batched.0 != 0 {
                ops::dflash2_selector_walk_batched(
                    gpu,
                    self.kernels.dflash2_selector_walk_batched,
                    self.scratch.topk_logits_dev,
                    self.scratch.topk_tokens_dev,
                    self.scratch.selector_hidden,
                    sel.predecessor_codebook.weight,
                    sel.successor_codebook.weight,
                    self.scratch.draft_tokens_dev,
                    last_token_dev,
                    gamma_eff as i32,
                    sel.rank as i32,
                    1,
                    stream,
                )?;
            }
        }
        if self.markov.is_some() {
            self.enqueue_markov_sequential_dev(gpu, stream, last_token_dev, gamma_eff)?;
        } else if self.selector.is_none() {
            self.launch_argmax_rows_or_loop(gpu, gamma_eff, lm_stride, stream)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_split_ctx_attention(
        &self,
        ctx_attn: Stage2LayerCtxAttn,
        m_rows: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        match ctx_attn {
            Stage2LayerCtxAttn::Linear {
                kernel,
                ctx_k,
                ctx_v,
                eff_ctx_dev,
                split_kv: Some(plan),
            } => {
                let _ = kernel;
                KernelLaunch::new(gpu, plan.split_kernel)
                    .grid([
                        self.num_q_heads as u32,
                        m_rows.div_ceil(32),
                        plan.num_splits,
                    ])
                    .block([128, 1, 1])
                    .arg_ptr(self.scratch.q_buf)
                    .arg_ptr(ctx_k)
                    .arg_ptr(ctx_v)
                    .arg_ptr(self.scratch.k_buf)
                    .arg_ptr(self.scratch.v_buf)
                    .arg_ptr(plan.workspace)
                    .arg_ptr(eff_ctx_dev)
                    .arg_u32(m_rows)
                    .arg_u32(self.num_q_heads as u32)
                    .arg_u32(self.num_kv_heads as u32)
                    .arg_u32(plan.num_splits)
                    .arg_f32(inv_sqrt_d)
                    .arg_u32(u32::from(causal))
                    .arg_u32(sliding_window)
                    .launch(stream)?;
                KernelLaunch::new(gpu, plan.reduce_kernel)
                    .grid([self.num_q_heads as u32, m_rows, 1])
                    .block([64, 1, 1])
                    .arg_ptr(plan.workspace)
                    .arg_ptr(self.scratch.attn_out)
                    .arg_u32(m_rows)
                    .arg_u32(self.num_q_heads as u32)
                    .arg_u32(1)
                    .arg_u32(plan.num_splits)
                    .launch(stream)
            }
            Stage2LayerCtxAttn::Linear {
                kernel,
                ctx_k,
                ctx_v,
                eff_ctx_dev,
                split_kv: None,
            } => KernelLaunch::new(gpu, kernel)
                .grid([self.num_q_heads as u32, m_rows.div_ceil(32), 1])
                .block([128, 1, 1])
                .arg_ptr(self.scratch.q_buf)
                .arg_ptr(ctx_k)
                .arg_ptr(ctx_v)
                .arg_ptr(self.scratch.k_buf)
                .arg_ptr(self.scratch.v_buf)
                .arg_ptr(self.scratch.attn_out)
                .arg_ptr(eff_ctx_dev)
                .arg_u32(m_rows)
                .arg_u32(self.num_q_heads as u32)
                .arg_u32(self.num_kv_heads as u32)
                .arg_f32(inv_sqrt_d)
                .arg_u32(u32::from(causal))
                .arg_u32(sliding_window)
                .launch(stream),
            Stage2LayerCtxAttn::CircularRingH128 {
                kernel,
                k_ctx_ptr_dev,
                v_ctx_ptr_dev,
                eff_ctx_dev,
                ring_start_dev,
                ring_cap,
                split_kv: Some(plan),
            } => {
                let _ = kernel;
                KernelLaunch::new(gpu, plan.split_kernel)
                    .grid([
                        self.num_q_heads as u32,
                        m_rows.div_ceil(32),
                        plan.num_splits,
                    ])
                    .block([128, 1, 1])
                    .arg_ptr(self.scratch.q_buf)
                    .arg_ptr(k_ctx_ptr_dev)
                    .arg_ptr(v_ctx_ptr_dev)
                    .arg_ptr(self.scratch.k_buf)
                    .arg_ptr(self.scratch.v_buf)
                    .arg_ptr(plan.workspace)
                    .arg_ptr(eff_ctx_dev)
                    .arg_ptr(ring_start_dev)
                    .arg_u32(ring_cap)
                    .arg_u32(m_rows)
                    .arg_u32(self.num_q_heads as u32)
                    .arg_u32(self.num_kv_heads as u32)
                    .arg_u32(1)
                    .arg_u32(plan.num_splits)
                    .arg_f32(inv_sqrt_d)
                    .arg_u32(u32::from(causal))
                    .arg_u32(sliding_window)
                    .launch(stream)?;
                KernelLaunch::new(gpu, plan.reduce_kernel)
                    .grid([self.num_q_heads as u32, m_rows, 1])
                    .block([64, 1, 1])
                    .arg_ptr(plan.workspace)
                    .arg_ptr(self.scratch.attn_out)
                    .arg_u32(m_rows)
                    .arg_u32(self.num_q_heads as u32)
                    .arg_u32(1)
                    .arg_u32(plan.num_splits)
                    .launch(stream)
            }
            Stage2LayerCtxAttn::CircularRingH128 {
                kernel,
                k_ctx_ptr_dev,
                v_ctx_ptr_dev,
                eff_ctx_dev,
                ring_start_dev,
                ring_cap,
                split_kv: None,
            } => KernelLaunch::new(gpu, kernel)
                .grid([self.num_q_heads as u32, m_rows.div_ceil(32), 1])
                .block([128, 1, 1])
                .arg_ptr(self.scratch.q_buf)
                .arg_ptr(k_ctx_ptr_dev)
                .arg_ptr(v_ctx_ptr_dev)
                .arg_ptr(self.scratch.k_buf)
                .arg_ptr(self.scratch.v_buf)
                .arg_ptr(self.scratch.attn_out)
                .arg_ptr(eff_ctx_dev)
                .arg_ptr(ring_start_dev)
                .arg_u32(ring_cap)
                .arg_u32(m_rows)
                .arg_u32(self.num_q_heads as u32)
                .arg_u32(self.num_kv_heads as u32)
                .arg_f32(inv_sqrt_d)
                .arg_u32(u32::from(causal))
                .arg_u32(sliding_window)
                .launch(stream),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn stage2_qk_norm_rope_and_attn(
        &self,
        q_norm: &crate::weight_map::DenseWeight,
        k_norm: &crate::weight_map::DenseWeight,
        ctx_attn: Stage2LayerCtxAttn,
        noise_positions_dev: DevicePtr,
        m_rows: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.q_buf,
            q_norm,
            self.scratch.q_buf,
            m_rows * self.num_q_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.k_buf,
            k_norm,
            self.scratch.k_buf,
            m_rows * self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        ops::rope_yarn_scaled(
            gpu,
            self.kernels.rope_qwen3,
            self.scratch.q_buf,
            self.scratch.k_buf,
            noise_positions_dev,
            m_rows,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            self.rope_attention_factor,
            stream,
        )?;
        self.launch_split_ctx_attention(
            ctx_attn,
            m_rows,
            inv_sqrt_d,
            causal,
            sliding_window,
            gpu,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn stage2_layer_bf16(
        &self,
        layer: &DflashLayer,
        ctx_attn: Stage2LayerCtxAttn,
        noise_positions_dev: DevicePtr,
        m_rows: u32,
        h: u32,
        q_dim: u32,
        kv_dim: u32,
        inter: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;

        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.input_layernorm,
            self.scratch.norm_buf,
            m_rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let attn_in = if let Some(conv) = &layer.attention_conv {
            self.stage2_conv_prepare_noise(conv, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };
        self.project_bf16_dense(
            gpu,
            attn_in,
            &layer.q_proj,
            self.scratch.q_buf,
            m_rows,
            q_dim,
            h,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            attn_in,
            &layer.k_proj,
            self.scratch.k_buf,
            m_rows,
            kv_dim,
            h,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            attn_in,
            &layer.v_proj,
            self.scratch.v_buf,
            m_rows,
            kv_dim,
            h,
            stream,
        )?;
        self.stage2_qk_norm_rope_and_attn(
            &layer.q_norm,
            &layer.k_norm,
            ctx_attn,
            noise_positions_dev,
            m_rows,
            inv_sqrt_d,
            causal,
            sliding_window,
            gpu,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            self.scratch.attn_out,
            &layer.o_proj,
            self.scratch.stream_acc,
            m_rows,
            h,
            q_dim,
            stream,
        )?;
        let attn_residual = if let Some(conv) = &layer.attention_conv {
            self.stage2_conv_finish_noise(conv, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            attn_residual,
            m_rows * h,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.post_attention_layernorm,
            self.scratch.norm_buf,
            m_rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let mlp_in = if let Some(conv) = &layer.mlp_conv {
            self.stage2_conv_prepare_noise(conv, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };
        self.project_bf16_dense(
            gpu,
            mlp_in,
            &layer.gate_proj,
            self.scratch.mlp_intermediate,
            m_rows,
            inter,
            h,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            mlp_in,
            &layer.up_proj,
            self.scratch.mlp_up,
            m_rows,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            gpu,
            self.kernels.silu_mul,
            self.scratch.mlp_intermediate,
            self.scratch.mlp_up,
            self.scratch.mlp_intermediate,
            m_rows * inter,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            self.scratch.mlp_intermediate,
            &layer.down_proj,
            self.scratch.stream_acc,
            m_rows,
            h,
            inter,
            stream,
        )?;
        let mlp_residual = if let Some(conv) = &layer.mlp_conv {
            self.stage2_conv_finish_noise(conv, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            mlp_residual,
            m_rows * h,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn stage2_layer_fp8(
        &self,
        layer: &DflashLayerFp8Weights,
        ctx_attn: Stage2LayerCtxAttn,
        noise_positions_dev: DevicePtr,
        m_rows: u32,
        h: u32,
        q_dim: u32,
        kv_dim: u32,
        inter: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;

        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.input_layernorm,
            self.scratch.norm_buf,
            m_rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let attn_in = if let Some(conv) = &layer.attention_conv {
            self.stage2_conv_prepare_noise(conv, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };
        self.stage_fp8_project(
            gpu,
            attn_in,
            &layer.q_proj,
            self.scratch.q_buf,
            m_rows,
            q_dim,
            h,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            attn_in,
            &layer.k_proj,
            self.scratch.k_buf,
            m_rows,
            kv_dim,
            h,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            attn_in,
            &layer.v_proj,
            self.scratch.v_buf,
            m_rows,
            kv_dim,
            h,
            stream,
        )?;
        self.stage2_qk_norm_rope_and_attn(
            &layer.q_norm,
            &layer.k_norm,
            ctx_attn,
            noise_positions_dev,
            m_rows,
            inv_sqrt_d,
            causal,
            sliding_window,
            gpu,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            self.scratch.attn_out,
            &layer.o_proj,
            self.scratch.stream_acc,
            m_rows,
            h,
            q_dim,
            stream,
        )?;
        let attn_residual = if let Some(conv) = &layer.attention_conv {
            self.stage2_conv_finish_noise(conv, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            attn_residual,
            m_rows * h,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.post_attention_layernorm,
            self.scratch.norm_buf,
            m_rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let mlp_in = if let Some(conv) = &layer.mlp_conv {
            self.stage2_conv_prepare_noise(conv, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };
        self.stage_fp8_project(
            gpu,
            mlp_in,
            &layer.gate_proj,
            self.scratch.mlp_intermediate,
            m_rows,
            inter,
            h,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            mlp_in,
            &layer.up_proj,
            self.scratch.mlp_up,
            m_rows,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            gpu,
            self.kernels.silu_mul,
            self.scratch.mlp_intermediate,
            self.scratch.mlp_up,
            self.scratch.mlp_intermediate,
            m_rows * inter,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            self.scratch.mlp_intermediate,
            &layer.down_proj,
            self.scratch.stream_acc,
            m_rows,
            h,
            inter,
            stream,
        )?;
        let mlp_residual = if let Some(conv) = &layer.mlp_conv {
            self.stage2_conv_finish_noise(conv, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            mlp_residual,
            m_rows * h,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn stage2_layer_nvfp4(
        &self,
        layer: &DflashLayerNvfp4,
        ctx_attn: Stage2LayerCtxAttn,
        noise_positions_dev: DevicePtr,
        m_rows: u32,
        h: u32,
        q_dim: u32,
        kv_dim: u32,
        inter: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;

        let w4a4 = self.w4a4_shadow.as_ref();
        let w4a4_scales = layer.input_scales;
        ensure!(
            w4a4.is_some() == w4a4_scales.is_some(),
            "DFlash2 W4A4 runtime state and calibrated layer receipt disagree"
        );
        let attn_t = if self.kernels.w4a16_gemm_t_m16.0 != 0 {
            match (
                layer.q_proj_t.as_ref(),
                layer.k_proj_t.as_ref(),
                layer.v_proj_t.as_ref(),
                layer.o_proj_t.as_ref(),
            ) {
                (Some(q), Some(k), Some(v), Some(o)) => Some((q, k, v, o)),
                _ => None,
            }
        } else {
            None
        };

        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.input_layernorm,
            self.scratch.norm_buf,
            m_rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let attn_in = if let Some(conv) = &layer.attention_conv {
            self.stage2_conv_prepare_noise(conv, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };

        if let (Some(w4a4), Some(scales)) = (w4a4, w4a4_scales) {
            w4a4.quantize(gpu, attn_in, scales.qkv(), m_rows, h, stream)?;
            w4a4.project_prepared(
                gpu,
                scales.qkv(),
                &layer.q_proj,
                self.scratch.q_buf,
                m_rows,
                q_dim,
                h,
                stream,
            )?;
            w4a4.project_prepared(
                gpu,
                scales.qkv(),
                &layer.k_proj,
                self.scratch.k_buf,
                m_rows,
                kv_dim,
                h,
                stream,
            )?;
            w4a4.project_prepared(
                gpu,
                scales.qkv(),
                &layer.v_proj,
                self.scratch.v_buf,
                m_rows,
                kv_dim,
                h,
                stream,
            )?;
        } else if m_rows <= 16
            && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                gpu,
                attn_in,
                &layer.q_proj,
                self.scratch.q_buf,
                m_rows,
                q_dim,
                h,
                16,
                stream,
            )?
        {
            if !crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                gpu,
                attn_in,
                &layer.k_proj,
                self.scratch.k_buf,
                m_rows,
                kv_dim,
                h,
                16,
                stream,
            )? {
                self.nvfp4_project_coalesced_or_gemm(
                    gpu,
                    attn_in,
                    &layer.k_proj,
                    self.scratch.k_buf,
                    m_rows,
                    kv_dim,
                    h,
                    stream,
                )?;
            }
            if !crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                gpu,
                attn_in,
                &layer.v_proj,
                self.scratch.v_buf,
                m_rows,
                kv_dim,
                h,
                16,
                stream,
            )? {
                self.nvfp4_project_coalesced_or_gemm(
                    gpu,
                    attn_in,
                    &layer.v_proj,
                    self.scratch.v_buf,
                    m_rows,
                    kv_dim,
                    h,
                    stream,
                )?;
            }
        } else if let Some((q_t, k_t, v_t, _)) = attn_t {
            self.draft_gemm_t(
                gpu,
                attn_in,
                q_t,
                self.scratch.q_buf,
                m_rows,
                q_dim,
                h,
                true,
                stream,
            )?;
            self.draft_gemm_t(
                gpu,
                attn_in,
                k_t,
                self.scratch.k_buf,
                m_rows,
                kv_dim,
                h,
                true,
                stream,
            )?;
            self.draft_gemm_t(
                gpu,
                attn_in,
                v_t,
                self.scratch.v_buf,
                m_rows,
                kv_dim,
                h,
                true,
                stream,
            )?;
        } else {
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                attn_in,
                &layer.q_proj,
                self.scratch.q_buf,
                m_rows,
                q_dim,
                h,
                stream,
            )?;
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                attn_in,
                &layer.k_proj,
                self.scratch.k_buf,
                m_rows,
                kv_dim,
                h,
                stream,
            )?;
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                attn_in,
                &layer.v_proj,
                self.scratch.v_buf,
                m_rows,
                kv_dim,
                h,
                stream,
            )?;
        }

        self.stage2_qk_norm_rope_and_attn(
            &layer.q_norm,
            &layer.k_norm,
            ctx_attn,
            noise_positions_dev,
            m_rows,
            inv_sqrt_d,
            causal,
            sliding_window,
            gpu,
            stream,
        )?;

        let disable_o = noise_graph_attn_kgamma_disable_o();
        if let (Some(w4a4), Some(scales)) = (w4a4, w4a4_scales) {
            w4a4.quantize(
                gpu,
                self.scratch.attn_out,
                scales.output(),
                m_rows,
                q_dim,
                stream,
            )?;
            w4a4.project_prepared(
                gpu,
                scales.output(),
                &layer.o_proj,
                self.scratch.stream_acc,
                m_rows,
                h,
                q_dim,
                stream,
            )?;
        } else if m_rows <= 16
            && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                gpu,
                self.scratch.attn_out,
                &layer.o_proj,
                self.scratch.stream_acc,
                m_rows,
                h,
                q_dim,
                16,
                stream,
            )?
        {
            // Dispatched via SM120 BF16xBF16 MMA coalesced warp-split-K.
        } else if let Some((_, _, _, o_t)) = attn_t
            && !disable_o
        {
            self.draft_gemm_t(
                gpu,
                self.scratch.attn_out,
                o_t,
                self.scratch.stream_acc,
                m_rows,
                h,
                q_dim,
                true,
                stream,
            )?;
        } else {
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                self.scratch.attn_out,
                &layer.o_proj,
                self.scratch.stream_acc,
                m_rows,
                h,
                q_dim,
                stream,
            )?;
        }

        let attn_residual = if let Some(conv) = &layer.attention_conv {
            self.stage2_conv_finish_noise(conv, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            attn_residual,
            m_rows * h,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.post_attention_layernorm,
            self.scratch.norm_buf,
            m_rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let mlp_in = if let Some(conv) = &layer.mlp_conv {
            self.stage2_conv_prepare_noise(conv, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };

        let ffn_t = if self.kernels.w4a16_gemm_t_m16.0 != 0 {
            match (
                layer.gate_proj_t.as_ref(),
                layer.up_proj_t.as_ref(),
                layer.down_proj_t.as_ref(),
            ) {
                (Some(g), Some(u), Some(d)) => Some((g, u, d)),
                _ => None,
            }
        } else {
            None
        };
        let mut fused_gateup = w4a4.is_none()
            && ffn_t.is_some()
            && self.kernels.w4a16_gemm_t_m32_n64_gateup_silu.0 != 0
            && self.kernels.w4a16_gemm_t_m32_n64.0 != 0;

        if let (Some(w4a4), Some(scales)) = (w4a4, w4a4_scales) {
            w4a4.quantize(gpu, mlp_in, scales.gate_up(), m_rows, h, stream)?;
            w4a4.project_prepared(
                gpu,
                scales.gate_up(),
                &layer.gate_proj,
                self.scratch.mlp_intermediate,
                m_rows,
                inter,
                h,
                stream,
            )?;
            w4a4.project_prepared(
                gpu,
                scales.gate_up(),
                &layer.up_proj,
                self.scratch.mlp_up,
                m_rows,
                inter,
                h,
                stream,
            )?;
        } else if m_rows <= 16
            && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m_gateup_silu(
                gpu,
                mlp_in,
                &layer.gate_proj,
                &layer.up_proj,
                self.scratch.mlp_intermediate,
                m_rows,
                inter,
                h,
                16,
                stream,
            )?
        {
            fused_gateup = true;
        } else if let Some((gate_t, up_t, _)) = ffn_t
            && fused_gateup
        {
            if atlas_core::device::active_sm_count() > 64
                && self.kernels.w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64.0 != 0
                && h.is_multiple_of(64)
                && inter.is_multiple_of(32)
            {
                ops::w4a16_gemm_n32_m32_gateup_silu(
                    gpu,
                    self.kernels.w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64,
                    mlp_in,
                    gate_t,
                    up_t,
                    self.scratch.mlp_intermediate,
                    m_rows,
                    inter,
                    h,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n64_m32_gateup_silu(
                    gpu,
                    self.kernels.w4a16_gemm_t_m32_n64_gateup_silu,
                    mlp_in,
                    gate_t,
                    up_t,
                    self.scratch.mlp_intermediate,
                    m_rows,
                    inter,
                    h,
                    stream,
                )?;
            }
        } else if let Some((gate_t, up_t, _)) = ffn_t {
            self.draft_gemm_t(
                gpu,
                mlp_in,
                gate_t,
                self.scratch.mlp_intermediate,
                m_rows,
                inter,
                h,
                true,
                stream,
            )?;
            self.draft_gemm_t(
                gpu,
                mlp_in,
                up_t,
                self.scratch.mlp_up,
                m_rows,
                inter,
                h,
                true,
                stream,
            )?;
        } else {
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                mlp_in,
                &layer.gate_proj,
                self.scratch.mlp_intermediate,
                m_rows,
                inter,
                h,
                stream,
            )?;
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                mlp_in,
                &layer.up_proj,
                self.scratch.mlp_up,
                m_rows,
                inter,
                h,
                stream,
            )?;
        }

        if !fused_gateup {
            ops::silu_mul(
                gpu,
                self.kernels.silu_mul,
                self.scratch.mlp_intermediate,
                self.scratch.mlp_up,
                self.scratch.mlp_intermediate,
                m_rows * inter,
                stream,
            )?;
        }

        if let (Some(w4a4), Some(scales)) = (w4a4, w4a4_scales) {
            w4a4.quantize(
                gpu,
                self.scratch.mlp_intermediate,
                scales.down(),
                m_rows,
                inter,
                stream,
            )?;
            w4a4.project_prepared(
                gpu,
                scales.down(),
                &layer.down_proj,
                self.scratch.stream_acc,
                m_rows,
                h,
                inter,
                stream,
            )?;
        } else if m_rows <= 16
            && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                gpu,
                self.scratch.mlp_intermediate,
                &layer.down_proj,
                self.scratch.stream_acc,
                m_rows,
                h,
                inter,
                16,
                stream,
            )?
        {
            // Dispatched via SM120 BF16xBF16 MMA coalesced warp-split-K.
        } else if let Some((_, _, down_t)) = ffn_t {
            self.draft_gemm_t(
                gpu,
                self.scratch.mlp_intermediate,
                down_t,
                self.scratch.stream_acc,
                m_rows,
                h,
                inter,
                true,
                stream,
            )?;
        } else {
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                self.scratch.mlp_intermediate,
                &layer.down_proj,
                self.scratch.stream_acc,
                m_rows,
                h,
                inter,
                stream,
            )?;
        }

        let mlp_residual = if let Some(conv) = &layer.mlp_conv {
            self.stage2_conv_finish_noise(conv, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            mlp_residual,
            m_rows * h,
            stream,
        )
    }
}

#[cfg(test)]
mod noise_graph_tests {
    use std::collections::HashSet;

    use super::{
        CtxCacheInvalidationReason, CtxKvCacheRepr, DflashLayerQuantMode, DflashNoiseGraphKey,
        DflashNoiseGraphMeta, noise_graph_stochastic_draft_sampling_active,
        parse_dflash_noise_graph_env, plan_ctx_kv_cache_transition,
    };

    #[test]
    fn noise_graph_meta_abi_layout_alignment_and_roundtrip() {
        assert_eq!(std::mem::size_of::<DflashNoiseGraphMeta>(), 80);
        assert_eq!(std::mem::align_of::<DflashNoiseGraphMeta>(), 16);
        assert_eq!(
            std::mem::offset_of!(DflashNoiseGraphMeta, eff_ctx),
            DflashNoiseGraphMeta::EFF_CTX_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(DflashNoiseGraphMeta, last_token),
            DflashNoiseGraphMeta::LAST_TOKEN_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(DflashNoiseGraphMeta, noise_len),
            DflashNoiseGraphMeta::NOISE_LEN_OFFSET
        );
        assert_eq!(std::mem::offset_of!(DflashNoiseGraphMeta, _pad), 12);
        assert_eq!(
            std::mem::offset_of!(DflashNoiseGraphMeta, noise_positions),
            DflashNoiseGraphMeta::NOISE_POSITIONS_OFFSET
        );

        let positions = [128u32, 129, 130, 131, 132, 133, 134, 135];
        let meta = DflashNoiseGraphMeta::try_new(128, 42_001, &positions, 512).unwrap();
        assert_eq!(meta.eff_ctx, 128);
        assert_eq!(meta.last_token, 42_001);
        assert_eq!(meta.noise_len, 8);
        assert_eq!(meta._pad, 0);
        assert_eq!(&meta.noise_positions[..8], &positions);
        assert_eq!(&meta.noise_positions[8..], &[0u32; 8]);

        let bytes = meta.to_le_bytes();
        assert_eq!(bytes.len(), 80);
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 128);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 42_001);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 8);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 128);
        assert_eq!(u32::from_le_bytes(bytes[44..48].try_into().unwrap()), 135);
        assert_eq!(DflashNoiseGraphMeta::from_le_bytes(&bytes).unwrap(), meta);

        let mut staging = [0xAAu8; 96];
        assert_eq!(meta.write_le_bytes(&mut staging).unwrap(), 80);
        assert_eq!(&staging[..80], &bytes);
        assert!(meta.write_le_bytes(&mut staging[..79]).is_err());
        assert!(DflashNoiseGraphMeta::from_le_bytes(&bytes[..79]).is_err());
    }

    #[test]
    fn noise_graph_meta_rejects_out_of_bounds_eff_ctx_and_noise_positions() {
        // eff_ctx > ctx_window
        assert!(DflashNoiseGraphMeta::try_new(513, 1, &[513, 514], 512).is_err());
        // empty noise_positions
        assert!(DflashNoiseGraphMeta::try_new(16, 1, &[], 512).is_err());
        // > 16 noise_positions
        let too_many: Vec<u32> = (16..33).collect();
        assert!(DflashNoiseGraphMeta::try_new(16, 1, &too_many, 512).is_err());
        // first noise position < eff_ctx
        assert!(DflashNoiseGraphMeta::try_new(16, 1, &[15, 16], 512).is_err());
        // non-increasing noise positions
        assert!(DflashNoiseGraphMeta::try_new(16, 1, &[16, 16, 17], 512).is_err());
        assert!(DflashNoiseGraphMeta::try_new(16, 1, &[16, 18, 17], 512).is_err());
        // max valid boundary: 16 positions at eff_ctx = ctx_window
        let exact_max: Vec<u32> = (512..528).collect();
        assert!(DflashNoiseGraphMeta::try_new(512, 7, &exact_max, 512).is_ok());
    }

    #[test]
    fn ring_wrap_around_and_repr_transitions_invalidate_cached_ctx_kv() {
        // Cold start into PostRope: no invalidation needed.
        assert_eq!(
            plan_ctx_kv_cache_transition(None, true, 0, 0),
            (CtxKvCacheRepr::PostRope, CtxCacheInvalidationReason::None)
        );
        // Steady linear growth in PostRope (needed_start == 0): keeps cache hot.
        assert_eq!(
            plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PostRope), true, 0, 256),
            (CtxKvCacheRepr::PostRope, CtxCacheInvalidationReason::None)
        );
        // Ring-native PostRope wrap-around (needed_start > 0, want_post_rope == true):
        // keeps circular PostRope cache hot across the ctx_window boundary.
        assert_eq!(
            plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PostRope), true, 1, 4096),
            (CtxKvCacheRepr::PostRope, CtxCacheInvalidationReason::None)
        );
        // Ring wrap-around fallback to eager (needed_start > 0, want_post_rope == false)
        // while holding PostRope cache: MUST invalidate before eager PreRope runs.
        assert_eq!(
            plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PostRope), false, 1, 512),
            (
                CtxKvCacheRepr::PreRope,
                CtxCacheInvalidationReason::RingWrapAround
            )
        );
        // Subsequent wrapped eager steps stay in PreRope without repeated clears.
        assert_eq!(
            plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PreRope), false, 8, 520),
            (CtxKvCacheRepr::PreRope, CtxCacheInvalidationReason::None)
        );
        // Transition from wrapped PreRope -> ring PostRope invalidates cached PreRope K.
        assert_eq!(
            plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PreRope), true, 8, 520),
            (
                CtxKvCacheRepr::PostRope,
                CtxCacheInvalidationReason::ReprTransition {
                    from: CtxKvCacheRepr::PreRope,
                    to: CtxKvCacheRepr::PostRope,
                }
            )
        );
        // Non-wrapped fallback from PostRope -> PreRope (e.g. sampled request)
        // invalidates cached PostRope K so eager won't double-RoPE it.
        assert_eq!(
            plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PostRope), false, 0, 64),
            (
                CtxKvCacheRepr::PreRope,
                CtxCacheInvalidationReason::ReprTransition {
                    from: CtxKvCacheRepr::PostRope,
                    to: CtxKvCacheRepr::PreRope,
                }
            )
        );
        // Transition from PreRope -> PostRope with non-empty cache invalidates
        // so Stage 1 recomputes clean PostRope K.
        assert_eq!(
            plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PreRope), true, 0, 64),
            (
                CtxKvCacheRepr::PostRope,
                CtxCacheInvalidationReason::ReprTransition {
                    from: CtxKvCacheRepr::PreRope,
                    to: CtxKvCacheRepr::PostRope,
                }
            )
        );
        // Empty cache transitions need no invalidation.
        assert_eq!(
            plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PostRope), false, 0, 0),
            (CtxKvCacheRepr::PreRope, CtxCacheInvalidationReason::None)
        );
    }

    #[test]
    fn noise_graph_key_separates_gamma_quant_and_markov_and_env_fails_closed() {
        let mut keys = HashSet::new();
        for gamma_eff in [7usize, 8, 15] {
            for quant_mode in [
                DflashLayerQuantMode::Bf16,
                DflashLayerQuantMode::Fp8,
                DflashLayerQuantMode::Nvfp4,
                DflashLayerQuantMode::Mixed,
            ] {
                for markov_active in [false, true] {
                    assert!(
                        keys.insert(DflashNoiseGraphKey {
                            gamma_eff,
                            quant_mode,
                            markov_active,
                        }),
                        "duplicate noise graph key for ({gamma_eff}, {quant_mode:?}, {markov_active})"
                    );
                }
            }
        }
        assert_eq!(keys.len(), 3 * 4 * 2);

        assert!(!parse_dflash_noise_graph_env(None).unwrap());
        assert!(!parse_dflash_noise_graph_env(Some("0")).unwrap());
        assert!(parse_dflash_noise_graph_env(Some("1")).unwrap());
        assert!(parse_dflash_noise_graph_env(Some(" 1 ")).unwrap());
        for bad in ["", "2", "true", "yes", "-1"] {
            assert!(parse_dflash_noise_graph_env(Some(bad)).is_err());
        }

        // SM-aware auto-enable: <= 64 SMs (GB10) stays off when unset; > 64 SMs
        // (SM120/SM103) auto-enables when unset, while "0" explicitly opts out.
        for sm in [0u32, 48, 64] {
            assert!(
                !super::parse_dflash_noise_graph_env_for_sm(None, sm).unwrap(),
                "sm={sm} must stay off when unset"
            );
            assert!(
                !super::parse_dflash_noise_graph_env_for_sm(Some("0"), sm).unwrap(),
                "sm={sm}"
            );
            assert!(
                super::parse_dflash_noise_graph_env_for_sm(Some("1"), sm).unwrap(),
                "sm={sm}"
            );
        }
        for sm in [65u32, 110, 170, 188] {
            assert!(
                super::parse_dflash_noise_graph_env_for_sm(None, sm).unwrap(),
                "sm={sm} must auto-enable when unset"
            );
            assert!(
                !super::parse_dflash_noise_graph_env_for_sm(Some("0"), sm).unwrap(),
                "sm={sm} must honor explicit opt-out"
            );
            assert!(
                super::parse_dflash_noise_graph_env_for_sm(Some("1"), sm).unwrap(),
                "sm={sm}"
            );
            for bad in ["", "2", "true", "off"] {
                assert!(
                    super::parse_dflash_noise_graph_env_for_sm(Some(bad), sm).is_err(),
                    "sm={sm} bad={bad:?}"
                );
            }
        }
    }

    #[test]
    fn noise_graph_allows_greedy_draft_with_positive_target_temperature() {
        use crate::dflash_sampling::{DflashDraftSampleMethod, DflashDraftSamplingRequest};
        use spark_runtime::sampler::SamplingParams;

        let mut sampled_params = SamplingParams::greedy(64);
        sampled_params.temperature = 0.7;
        sampled_params.seed = Some(1337);
        let sampled_req =
            DflashDraftSamplingRequest::from_seeded_sampling_params(&sampled_params, &[], 0, 15)
                .unwrap();

        let mut zero_temp_params = SamplingParams::greedy(64);
        zero_temp_params.seed = Some(1337);
        assert!(
            DflashDraftSamplingRequest::from_seeded_sampling_params(&zero_temp_params, &[], 0, 15)
                .is_err(),
            "zero-temperature request must not construct a stochastic draft sampling request"
        );

        // Greedy drafting paired with target Gumbel verify (temperature > 0.0)
        // keeps the noise pass deterministic and must allow Stage-2 CUDA graph replay.
        assert!(!noise_graph_stochastic_draft_sampling_active(
            DflashDraftSampleMethod::Greedy,
            None,
        ));
        assert!(!noise_graph_stochastic_draft_sampling_active(
            DflashDraftSampleMethod::Greedy,
            Some(&sampled_req),
        ));
        assert!(!noise_graph_stochastic_draft_sampling_active(
            DflashDraftSampleMethod::Probabilistic,
            None,
        ));
        // Stochastic draft sampling inside the noise pass must still bypass the graph.
        assert!(noise_graph_stochastic_draft_sampling_active(
            DflashDraftSampleMethod::Probabilistic,
            Some(&sampled_req),
        ));
    }

    #[test]
    fn c1_gumbel_meta_upload_encodes_active_greedy_and_rotating_slots() {
        use super::{GumbelScratchPtrs, NoiseGraphHeadState, gumbel_spec_fields};
        use spark_runtime::gpu::GpuBackend;
        use spark_runtime::gpu::mock::MockGpuBackend;
        use spark_runtime::sampler::gumbel::GumbelDraftSpec;

        let gpu = MockGpuBackend::new();
        let ptrs = GumbelScratchPtrs {
            seeds: gpu.alloc(64 * 8).unwrap(),
            step0_pos: gpu.alloc(64 * 8).unwrap(),
            temps: gpu.alloc(64 * 4).unwrap(),
            top_ks: gpu.alloc(64 * 4).unwrap(),
            top_ps: gpu.alloc(64 * 4).unwrap(),
            min_ps: gpu.alloc(64 * 4).unwrap(),
        };
        let anchor_dev = gpu.alloc(64 * 4).unwrap();
        let mut state = NoiseGraphHeadState::new();

        let active = GumbelDraftSpec {
            seed: 0xDEAD_BEEF_CAFE_0001,
            temperature: 0.75,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.05,
            step0_abs_pos: 1234,
        };
        assert_eq!(
            gumbel_spec_fields(Some(active)),
            (0xDEAD_BEEF_CAFE_0001, 1234, 0.75, 20, 0.95, 0.05)
        );
        assert_eq!(gumbel_spec_fields(None), (0, 0, 0.0, 0, 1.0, 0.0));
        assert_eq!(
            gumbel_spec_fields(Some(GumbelDraftSpec {
                temperature: 0.0,
                ..active
            })),
            (0, 0, 0.0, 0, 1.0, 0.0)
        );

        // Rotate past NOISE_META_PINNED_SLOTS to verify slot wraparound.
        for step in 0..6u64 {
            let spec = GumbelDraftSpec {
                seed: active.seed + step,
                step0_abs_pos: active.step0_abs_pos + step,
                ..active
            };
            state
                .upload_gumbel_meta_c1(
                    &gpu,
                    Some(spec),
                    ptrs,
                    Some((700 + step as u32, anchor_dev)),
                    0,
                )
                .unwrap();
        }
        assert_eq!(
            u64::from_le_bytes(gpu.read_alloc(ptrs.seeds).unwrap()[..8].try_into().unwrap()),
            active.seed + 5
        );
        assert_eq!(
            u64::from_le_bytes(
                gpu.read_alloc(ptrs.step0_pos).unwrap()[..8]
                    .try_into()
                    .unwrap()
            ),
            active.step0_abs_pos + 5
        );
        assert_eq!(
            f32::from_le_bytes(gpu.read_alloc(ptrs.temps).unwrap()[..4].try_into().unwrap()),
            0.75
        );
        assert_eq!(
            u32::from_le_bytes(
                gpu.read_alloc(ptrs.top_ks).unwrap()[..4]
                    .try_into()
                    .unwrap()
            ),
            20
        );
        assert_eq!(
            f32::from_le_bytes(
                gpu.read_alloc(ptrs.top_ps).unwrap()[..4]
                    .try_into()
                    .unwrap()
            ),
            0.95
        );
        assert_eq!(
            f32::from_le_bytes(
                gpu.read_alloc(ptrs.min_ps).unwrap()[..4]
                    .try_into()
                    .unwrap()
            ),
            0.05
        );
        assert_eq!(
            u32::from_le_bytes(gpu.read_alloc(anchor_dev).unwrap()[..4].try_into().unwrap()),
            705
        );

        // Switching back to greedy resets temp to 0.0 without touching anchor_dev when None.
        state
            .upload_gumbel_meta_c1(&gpu, None, ptrs, None, 0)
            .unwrap();
        assert_eq!(
            f32::from_le_bytes(gpu.read_alloc(ptrs.temps).unwrap()[..4].try_into().unwrap()),
            0.0
        );
        assert_eq!(
            u32::from_le_bytes(gpu.read_alloc(anchor_dev).unwrap()[..4].try_into().unwrap()),
            705
        );

        // Rejection staging scales draft temp, validates slot*8+gamma bounds, and uploads q_row_base.
        let q_row_base_dev = gpu.alloc(64 * 4).unwrap();
        let (scaled, base_row, rec) =
            super::resolve_rejection_seq_staging(Some(active), Some(5), 4, 512, 0.5);
        assert_eq!(base_row, 40);
        assert_eq!(rec, Some((5, active.step0_abs_pos, 4)));
        assert!((scaled.unwrap().temperature - 0.375).abs() < 1.0e-6);
        // Out-of-bounds slot or missing slot degrades to greedy + None record.
        assert_eq!(
            super::resolve_rejection_seq_staging(Some(active), Some(64), 4, 512, 1.0),
            (None, 0, None)
        );
        assert_eq!(
            super::resolve_rejection_seq_staging(Some(active), None, 4, 512, 1.0),
            (None, 0, None)
        );
        state
            .upload_gumbel_meta_c1_with_q_base(
                &gpu,
                scaled,
                ptrs,
                None,
                Some((base_row, q_row_base_dev)),
                0,
            )
            .unwrap();
        assert_eq!(
            u32::from_le_bytes(
                gpu.read_alloc(q_row_base_dev).unwrap()[..4]
                    .try_into()
                    .unwrap()
            ),
            40
        );
    }

    #[test]
    fn c1_stage2_graph_key_isolates_shortlist_k_and_tc_lm_head_modes() {
        use super::{
            LmHeadNvfp4Mode, NoiseGraphRuntimeKey, resolve_c1_lm_head_nvfp4_mode_for_sm,
            resolve_dflash2_selector_topk, should_use_coalesced_nvfp4_lm_head,
            validate_selector_shortlist,
        };

        // GB10 (<= 64 SMs) preserves Exact in greedy mode and CoalescedTc in sampled mode;
        // SM120/SM103 (> 64 SMs) uses CoalescedTc even in C=1 greedy mode (lossless under
        // target verification).
        for gb10_sms in [0u32, 48, 64] {
            assert_eq!(
                resolve_c1_lm_head_nvfp4_mode_for_sm(false, gb10_sms),
                LmHeadNvfp4Mode::Exact,
                "GB10 sm_count={gb10_sms} greedy must stay Exact"
            );
            assert_eq!(
                resolve_c1_lm_head_nvfp4_mode_for_sm(true, gb10_sms),
                LmHeadNvfp4Mode::CoalescedTc,
                "GB10 sm_count={gb10_sms} sampled must use CoalescedTc"
            );
        }
        for high_sms in [65u32, 110, 170, 188] {
            assert_eq!(
                resolve_c1_lm_head_nvfp4_mode_for_sm(false, high_sms),
                LmHeadNvfp4Mode::CoalescedTc,
                "SM120/SM103 sm_count={high_sms} greedy must use CoalescedTc"
            );
            assert_eq!(
                resolve_c1_lm_head_nvfp4_mode_for_sm(true, high_sms),
                LmHeadNvfp4Mode::CoalescedTc,
                "SM120/SM103 sm_count={high_sms} sampled must use CoalescedTc"
            );
        }

        // C=1 T=0 greedy stays on exact scalar LM head for gamma_eff <= 32,
        // while rows > 32 or CoalescedTc (rows >= 2) route to coalesced NVFP4 TC.
        assert!(!should_use_coalesced_nvfp4_lm_head(
            LmHeadNvfp4Mode::Exact,
            1
        ));
        assert!(!should_use_coalesced_nvfp4_lm_head(
            LmHeadNvfp4Mode::Exact,
            7
        ));
        assert!(!should_use_coalesced_nvfp4_lm_head(
            LmHeadNvfp4Mode::Exact,
            15
        ));
        assert!(!should_use_coalesced_nvfp4_lm_head(
            LmHeadNvfp4Mode::Exact,
            32
        ));
        assert!(should_use_coalesced_nvfp4_lm_head(
            LmHeadNvfp4Mode::Exact,
            48
        ));
        assert!(!should_use_coalesced_nvfp4_lm_head(
            LmHeadNvfp4Mode::CoalescedTc,
            1
        ));
        assert!(should_use_coalesced_nvfp4_lm_head(
            LmHeadNvfp4Mode::CoalescedTc,
            2
        ));
        assert!(should_use_coalesced_nvfp4_lm_head(
            LmHeadNvfp4Mode::CoalescedTc,
            7
        ));
        assert!(should_use_coalesced_nvfp4_lm_head(
            LmHeadNvfp4Mode::CoalescedTc,
            48
        ));

        // Keeps checkpoint-trained selector_top_k (16) while enabling fast radix top-k + out_probs.
        assert_eq!(
            resolve_dflash2_selector_topk(16, 248_320, true, true, true),
            (16, true, true)
        );
        assert_eq!(
            resolve_dflash2_selector_topk(16, 248_320, false, true, true),
            (16, true, true)
        );
        assert_eq!(
            resolve_dflash2_selector_topk(16, 248_320, true, false, true),
            (16, false, true)
        );
        assert_eq!(
            resolve_dflash2_selector_topk(16, 248_320, true, true, false),
            (16, true, false)
        );
        // Clamps to small vocab cleanly.
        assert_eq!(
            resolve_dflash2_selector_topk(16, 12, true, true, true),
            (12, true, true)
        );

        assert_eq!(
            validate_selector_shortlist(7, 15, 248_320, 32, 32).unwrap(),
            (7, 248_320, 32)
        );
        assert!(validate_selector_shortlist(7, 15, 248_320, 33, 32).is_err());

        let base = NoiseGraphRuntimeKey {
            logical: DflashNoiseGraphKey {
                gamma_eff: 7,
                quant_mode: DflashLayerQuantMode::Nvfp4,
                markov_active: false,
            },
            attn_mode: super::Stage2CtxAttnMode::Linear,
            split_kv_splits: 1,
            lm_head_path_tag: 2,
            prefer_tc_lm_head: false,
            rejection_mode: false,
            shortlist_k: 16,
            mask_id: 248_077,
            stream: 7,
            ctx_kv_ptrs: vec![(0x1000, 0x2000)],
        };
        let split_kv_mode = NoiseGraphRuntimeKey {
            split_kv_splits: 6,
            ..base.clone()
        };
        let ring_mode = NoiseGraphRuntimeKey {
            attn_mode: super::Stage2CtxAttnMode::CircularRingH128,
            ..base.clone()
        };
        let sampled = NoiseGraphRuntimeKey {
            prefer_tc_lm_head: true,
            shortlist_k: 32,
            ..base.clone()
        };
        let tc_only = NoiseGraphRuntimeKey {
            prefer_tc_lm_head: true,
            ..base.clone()
        };
        let k32_only = NoiseGraphRuntimeKey {
            shortlist_k: 32,
            ..base.clone()
        };
        let rejection_only = NoiseGraphRuntimeKey {
            rejection_mode: true,
            shortlist_k: 20,
            ..base.clone()
        };
        let set: HashSet<NoiseGraphRuntimeKey> = [
            base,
            split_kv_mode,
            ring_mode,
            sampled,
            tc_only,
            k32_only,
            rejection_only,
        ]
        .into_iter()
        .collect();
        assert_eq!(set.len(), 7);
    }

    #[test]
    fn dflash_split_kv_policy_saturates_wide_sm_and_preserves_gb10_and_short_ctx() {
        use super::{
            DFLASH_SPLIT_KV_H128_MAX_SPLITS, DflashNoiseGraphMeta, NoiseGraphHeadState,
            dflash_split_kv_h128_workspace_bytes, resolve_dflash_split_kv_splits_for_sm,
        };
        use spark_runtime::gpu::mock::MockGpuBackend;

        // GB10 (<= 64 SMs), short context (< 256), and non-H128 stay single-split (1).
        for sm in [0u32, 48, 64] {
            for eff_ctx in [0usize, 255, 256, 2048, 4096] {
                assert_eq!(
                    resolve_dflash_split_kv_splits_for_sm(128, eff_ctx, 32, 1, sm),
                    1
                );
            }
        }
        assert_eq!(
            resolve_dflash_split_kv_splits_for_sm(128, 255, 32, 1, 188),
            1
        );
        assert_eq!(
            resolve_dflash_split_kv_splits_for_sm(256, 2048, 32, 1, 188),
            1
        );
        assert_eq!(
            resolve_dflash_split_kv_splits_for_sm(128, 2048, 32, 6, 188),
            1
        );

        // Wide-SM (> 64 SMs, e.g. 188-SM SM120) lifts C=1..4 from 32..128 CTAs to 192..512 CTAs.
        let s_c1 = resolve_dflash_split_kv_splits_for_sm(128, 256, 32, 1, 188);
        assert_eq!(s_c1, 6);
        assert_eq!(32 * s_c1, 192);
        let s_c2 = resolve_dflash_split_kv_splits_for_sm(128, 2048, 32, 2, 188);
        assert_eq!(s_c2, 4);
        assert_eq!(32 * 2 * s_c2, 256);
        let s_c4 = resolve_dflash_split_kv_splits_for_sm(128, 2048, 32, 4, 188);
        assert_eq!(s_c4, 4);
        assert_eq!(32 * 4 * s_c4, 512);
        // Smaller head counts clamp cleanly at 8 splits.
        assert_eq!(
            resolve_dflash_split_kv_splits_for_sm(128, 2048, 16, 1, 188),
            8
        );

        // Workspace allocation is fixed-pointer capture-safe and fails closed on oversize demand.
        let alloc_bytes = dflash_split_kv_h128_workspace_bytes(
            1,
            DFLASH_SPLIT_KV_H128_MAX_SPLITS,
            32,
            DflashNoiseGraphMeta::MAX_NOISE_POSITIONS,
        )
        .expect("workspace bytes");
        assert_eq!(alloc_bytes, 8 * 32 * 16 * 130 * 4);
        let needed_bytes = dflash_split_kv_h128_workspace_bytes(1, 6, 32, 8).expect("needed");
        let gpu = MockGpuBackend::new();
        let mut state = NoiseGraphHeadState::new();
        let ptr1 = state
            .ensure_split_kv_workspace(&gpu, alloc_bytes, needed_bytes)
            .expect("first alloc")
            .expect("workspace ptr");
        assert_eq!(gpu.alloc_count(), 1);
        let ptr2 = state
            .ensure_split_kv_workspace(&gpu, alloc_bytes, alloc_bytes)
            .expect("reuse alloc")
            .expect("same ptr");
        assert_eq!(ptr1, ptr2);
        assert_eq!(gpu.alloc_count(), 1);
        assert_eq!(
            state
                .ensure_split_kv_workspace(&gpu, alloc_bytes, alloc_bytes + 4)
                .expect("oversize fail-closed"),
            None
        );
    }

    #[test]
    fn stage1_ring_append_plan_covers_linear_wrapped_two_span_prefill_and_rollback() {
        use super::{Stage1RingSpan, all_layers_cover_old_fc_prefix, plan_stage1_ring_append};

        // 1. needed_start == 0 cold start and steady incremental append.
        let cold = plan_stage1_ring_append(0, 128, 4096, 0, 0, 0, 0).unwrap();
        assert_eq!(cold.old_ctx_count, 0);
        assert_eq!(cold.new_abs_start, 0);
        assert_eq!(cold.new_ctx_count, 128);
        assert_eq!(
            cold.spans(),
            &[Stage1RingSpan {
                linear_row: 0,
                physical_slot: 0,
                count: 128,
            }]
        );

        let linear_step = plan_stage1_ring_append(0, 134, 4096, 0, 128, 0, 128).unwrap();
        assert_eq!(linear_step.old_ctx_count, 128);
        assert_eq!(linear_step.new_abs_start, 128);
        assert_eq!(linear_step.new_ctx_count, 6);
        assert_eq!(
            linear_step.spans(),
            &[Stage1RingSpan {
                linear_row: 128,
                physical_slot: 128,
                count: 6,
            }]
        );

        // 2. Initial post-prefill fill at 136.6K (new_count == window, 2 spans).
        //    needed=[132574..136670), 132574 % 4096 == 1502 -> [1502..4096) + [0..1502).
        let prefill_136k = plan_stage1_ring_append(132_574, 4096, 4096, 0, 0, 0, 0).unwrap();
        assert_eq!(prefill_136k.old_ctx_count, 0);
        assert_eq!(prefill_136k.new_abs_start, 132_574);
        assert_eq!(prefill_136k.new_ctx_count, 4096);
        assert_eq!(
            prefill_136k.spans(),
            &[
                Stage1RingSpan {
                    linear_row: 0,
                    physical_slot: 1502,
                    count: 2594,
                },
                Stage1RingSpan {
                    linear_row: 2594,
                    physical_slot: 0,
                    count: 1502,
                },
            ]
        );

        // 3. Steady decode at 136.6K (needed_start > 0, single physical span).
        let decode_136k =
            plan_stage1_ring_append(132_579, 4096, 4096, 132_574, 136_670, 132_574, 136_670)
                .unwrap();
        assert_eq!(decode_136k.old_ctx_count, 4091);
        assert_eq!(decode_136k.new_abs_start, 136_670);
        assert_eq!(decode_136k.new_ctx_count, 5);
        assert_eq!(
            decode_136k.spans(),
            &[Stage1RingSpan {
                linear_row: 4091,
                physical_slot: 1502,
                count: 5,
            }]
        );

        // 4. Crossing the 4096 boundary from 4095 -> 4099 (2-span wrap-around).
        let wrap_4k = plan_stage1_ring_append(3, 4096, 4096, 0, 4095, 0, 4095).unwrap();
        assert_eq!(wrap_4k.old_ctx_count, 4092);
        assert_eq!(wrap_4k.new_abs_start, 4095);
        assert_eq!(wrap_4k.new_ctx_count, 4);
        assert_eq!(
            wrap_4k.spans(),
            &[
                Stage1RingSpan {
                    linear_row: 4092,
                    physical_slot: 4095,
                    count: 1,
                },
                Stage1RingSpan {
                    linear_row: 4093,
                    physical_slot: 0,
                    count: 3,
                },
            ]
        );

        // 5. High-context 2-span wrap crossing 4096 * 34 = 139264.
        let wrap_139k =
            plan_stage1_ring_append(135_170, 4096, 4096, 135_166, 139_262, 135_166, 139_262)
                .unwrap();
        assert_eq!(wrap_139k.old_ctx_count, 4092);
        assert_eq!(wrap_139k.new_abs_start, 139_262);
        assert_eq!(wrap_139k.new_ctx_count, 4);
        assert_eq!(
            wrap_139k.spans(),
            &[
                Stage1RingSpan {
                    linear_row: 4092,
                    physical_slot: 4094,
                    count: 2,
                },
                Stage1RingSpan {
                    linear_row: 4094,
                    physical_slot: 0,
                    count: 2,
                },
            ]
        );

        // 6. Rollback invalidation (cache_end > needed_end) forces full recompute.
        let rollback_wrapped =
            plan_stage1_ring_append(132_570, 4096, 4096, 132_574, 136_670, 132_574, 136_670)
                .unwrap();
        assert_eq!(rollback_wrapped.old_ctx_count, 0);
        assert_eq!(rollback_wrapped.new_ctx_count, 4096);

        let rollback_linear = plan_stage1_ring_append(0, 400, 4096, 0, 500, 0, 500).unwrap();
        assert_eq!(rollback_linear.old_ctx_count, 0);
        assert_eq!(rollback_linear.new_ctx_count, 400);

        // 7. Forward jump exceeding window forces full recompute.
        let jump = plan_stage1_ring_append(10_000, 4096, 4096, 0, 4096, 0, 4096).unwrap();
        assert_eq!(jump.old_ctx_count, 0);
        assert_eq!(jump.new_ctx_count, 4096);

        // 8. Exact cache hit emits zero spans.
        let hit = plan_stage1_ring_append(132_574, 4096, 4096, 132_574, 136_670, 132_574, 136_670)
            .unwrap();
        assert_eq!(hit.old_ctx_count, 4096);
        assert_eq!(hit.new_ctx_count, 0);
        assert!(hit.spans().is_empty());

        // 9. Multi-layer `skip_old_fc_copy` coverage predicate across wrapped and rollback states.
        assert!(all_layers_cover_old_fc_prefix(
            132_578,
            4096,
            4092,
            4096,
            &[132_574; 8],
            &[136_670; 8],
            &[132_574; 8],
            &[136_670; 8],
        ));
        // One lagging layer must refuse skip.
        let mut lagging_v_end = [136_670; 8];
        lagging_v_end[5] = 136_669;
        assert!(!all_layers_cover_old_fc_prefix(
            132_578,
            4096,
            4092,
            4096,
            &[132_574; 8],
            &[136_670; 8],
            &[132_574; 8],
            &lagging_v_end,
        ));
        // Rollback in any layer must refuse skip.
        assert!(!all_layers_cover_old_fc_prefix(
            132_574,
            4096,
            4092,
            4096,
            &[132_574; 8],
            &[136_675; 8],
            &[132_574; 8],
            &[136_675; 8],
        ));
    }

    #[test]
    fn stage1_ring_simulation_preserves_exact_chronological_slots_across_wraps() {
        use super::plan_stage1_ring_append;

        const WINDOW: usize = 64;
        let mut ring_slots = vec![usize::MAX; WINDOW];
        let mut cache_start = 0usize;
        let mut cache_end = 0usize;
        let mut pos = 0usize;

        // Walk from cold start through prefill + dozens of variable-width decode
        // steps (1..=8 tokens) well past multiple ring wraps.
        let deltas = [
            20usize, 5, 7, 3, 8, 6, 8, 5, 4, 3, 1, 8, 7, 6, 5, 8, 2, 4, 7, 8, 3, 6, 8, 7,
        ];
        for &delta in &deltas {
            pos += delta;
            let eff_ctx = pos.min(WINDOW);
            let needed_start = pos - eff_ctx;
            let plan = plan_stage1_ring_append(
                needed_start,
                eff_ctx,
                WINDOW,
                cache_start,
                cache_end,
                cache_start,
                cache_end,
            )
            .unwrap();
            for span in plan.spans() {
                for j in 0..span.count {
                    let abs_pos = needed_start + span.linear_row + j;
                    let phys = span.physical_slot + j;
                    assert_eq!(phys, abs_pos % WINDOW);
                    ring_slots[phys] = abs_pos;
                }
            }
            cache_start = needed_start;
            cache_end = pos;

            // Verify Stage-2 circular attention indexing `(ring_start + k_row) % WINDOW`
            // recovers exact chronological tokens `[needed_start..pos)`.
            let ring_start = needed_start % WINDOW;
            for k_row in 0..eff_ctx {
                let mut phys = ring_start + k_row;
                if phys >= WINDOW {
                    phys -= WINDOW;
                }
                assert_eq!(
                    ring_slots[phys],
                    needed_start + k_row,
                    "pos={pos} needed_start={needed_start} k_row={k_row}"
                );
            }
        }
    }

    #[test]
    fn stage2_ctx_attn_mode_preserves_linear_under_window_and_engages_ring_past_window() {
        use super::{Stage2CtxAttnMode, resolve_stage2_ctx_attn_mode};

        // needed_start == 0 stays Linear (0-ULP parity for seq_len <= 4096) even when ring kernel exists.
        assert_eq!(
            resolve_stage2_ctx_attn_mode(128, 0, 256, 4096, true, true, true),
            Some(Stage2CtxAttnMode::Linear)
        );
        assert_eq!(
            resolve_stage2_ctx_attn_mode(128, 0, 4096, 4096, true, true, true),
            Some(Stage2CtxAttnMode::Linear)
        );
        // needed_start > 0 engages CircularRingH128 for HD=128 with SWA, full ring, or partial ring.
        assert_eq!(
            resolve_stage2_ctx_attn_mode(128, 1, 4096, 4096, true, true, true),
            Some(Stage2CtxAttnMode::CircularRingH128)
        );
        assert_eq!(
            resolve_stage2_ctx_attn_mode(128, 132_574, 4096, 4096, false, true, true),
            Some(Stage2CtxAttnMode::CircularRingH128)
        );
        // Partial ring (e.g. prefix-cache skip with eff_ctx < window) works with CircularRingH128.
        assert_eq!(
            resolve_stage2_ctx_attn_mode(128, 10_000, 50, 4096, true, true, true),
            Some(Stage2CtxAttnMode::CircularRingH128)
        );
        // HD=256 (no ring kernel): non-SWA full ring allowed on Linear; SWA or partial misaligned ring fails closed.
        assert_eq!(
            resolve_stage2_ctx_attn_mode(256, 132_574, 4096, 4096, true, false, false),
            Some(Stage2CtxAttnMode::Linear)
        );
        assert_eq!(
            resolve_stage2_ctx_attn_mode(256, 132_574, 4096, 4096, true, false, true),
            None
        );
        assert_eq!(
            resolve_stage2_ctx_attn_mode(256, 10_000, 50, 4096, true, false, false),
            None
        );
        // Aligned partial ring on non-SWA Linear is valid because slots [0..eff_ctx) match.
        assert_eq!(
            resolve_stage2_ctx_attn_mode(256, 8192, 50, 4096, true, false, false),
            Some(Stage2CtxAttnMode::Linear)
        );
        // Invalid geometry rejected.
        assert_eq!(
            resolve_stage2_ctx_attn_mode(128, 0, 4097, 4096, true, true, false),
            None
        );
        assert_eq!(
            resolve_stage2_ctx_attn_mode(64, 0, 128, 4096, true, true, false),
            None
        );
        // High-SM GPUs (SM120, >64 SMs) prefer pointer-agnostic CircularRingH128
        // even at needed_start == 0 so Stage-2 CUDA graphs reuse cleanly across
        // requests; GB10 (<=64 SMs) preserves Linear bit-for-bit.
        assert_eq!(
            super::resolve_stage2_ctx_attn_mode_for_sm(128, 0, 256, 4096, true, true, true, 188),
            Some(Stage2CtxAttnMode::CircularRingH128)
        );
        assert_eq!(
            super::resolve_stage2_ctx_attn_mode_for_sm(128, 0, 256, 4096, true, true, true, 48),
            Some(Stage2CtxAttnMode::Linear)
        );
        assert_eq!(
            super::resolve_stage2_ctx_attn_mode_for_sm(256, 0, 256, 4096, true, false, true, 188),
            Some(Stage2CtxAttnMode::Linear)
        );
    }

    #[test]
    fn c1_ring_meta_upload_encodes_aligned_ring_start_and_per_layer_kv_ptrs() {
        use super::{NoiseGraphHeadState, RingMetaLayout};
        use spark_runtime::gpu::DevicePtr;
        use spark_runtime::gpu::mock::MockGpuBackend;

        let gpu = MockGpuBackend::new();
        let mut state = NoiseGraphHeadState::new();
        let k_ptrs = [
            DevicePtr(0x10_0000),
            DevicePtr(0x11_0000),
            DevicePtr(0x12_0000),
            DevicePtr(0x13_0000),
        ];
        let v_ptrs = [
            DevicePtr(0x20_0000),
            DevicePtr(0x21_0000),
            DevicePtr(0x22_0000),
            DevicePtr(0x23_0000),
        ];
        let layout = RingMetaLayout::new(k_ptrs.len()).unwrap();
        assert_eq!(layout.ring_start_off, 0);
        assert_eq!(layout.ctx_k_ptrs_off, 8);
        assert_eq!(layout.ctx_v_ptrs_off, 8 + 4 * 8);
        assert_eq!(layout.total_bytes, 8 + 8 * 8);

        // Rotate past NOISE_META_PINNED_SLOTS to verify slot wraparound and fixed device ptr.
        let mut first_base = DevicePtr::NULL;
        for step in 0..6u32 {
            let ptrs = state
                .upload_ring_meta(&gpu, 1502 + step, &k_ptrs, &v_ptrs, 0)
                .unwrap();
            if step == 0 {
                first_base = ptrs.base;
            } else {
                assert_eq!(ptrs.base, first_base);
            }
            assert_eq!(ptrs.ring_start_dev, ptrs.base);
            assert_eq!(ptrs.layer_k_ptr(2).0, ptrs.base.0 + 8 + 16);
            assert_eq!(ptrs.layer_v_ptr(3).0, ptrs.base.0 + 8 + 32 + 24);
        }

        let dev_bytes = gpu.read_alloc(first_base).unwrap();
        assert_eq!(
            u32::from_le_bytes(dev_bytes[0..4].try_into().unwrap()),
            1507
        );
        assert_eq!(u32::from_le_bytes(dev_bytes[4..8].try_into().unwrap()), 0);
        for (l, (&k_ptr, &v_ptr)) in k_ptrs.iter().zip(v_ptrs.iter()).enumerate() {
            let k_off = layout.ctx_k_ptrs_off + l * 8;
            let v_off = layout.ctx_v_ptrs_off + l * 8;
            assert_eq!(
                u64::from_le_bytes(dev_bytes[k_off..k_off + 8].try_into().unwrap()),
                k_ptr.0
            );
            assert_eq!(
                u64::from_le_bytes(dev_bytes[v_off..v_off + 8].try_into().unwrap()),
                v_ptr.0
            );
        }
        // Layer count mismatch fails closed instead of overflowing the fixed slab.
        assert!(
            state
                .upload_ring_meta(&gpu, 0, &k_ptrs[..2], &v_ptrs[..2], 0)
                .is_err()
        );
    }

    #[test]
    fn bf16_proj_cublas_alignment_accepts_conv_selector_and_rejects_odd_or_misaligned() {
        use super::bf16_proj_cublas_aligned;
        use spark_runtime::gpu::DevicePtr;

        let p = DevicePtr(0x1_0000);
        // DFlash2 conv kernel_projection ([M, 2560] x [640, 2560]^T) and
        // selector hidden_projection ([M, 2560] x [256, 2560]^T) across M=1..128.
        for m in [1u32, 3, 4, 8, 16, 48, 64, 128] {
            assert!(bf16_proj_cublas_aligned(p, p, p, m, 640, 2560));
            assert!(bf16_proj_cublas_aligned(p, p, p, m, 256, 2560));
        }
        // Odd vocab (248_077), zero dims, null pointers, and unaligned pointers
        // fail closed to scalar dense_gemm.
        assert!(!bf16_proj_cublas_aligned(p, p, p, 4, 248_077, 2560));
        assert!(!bf16_proj_cublas_aligned(p, p, p, 0, 640, 2560));
        assert!(!bf16_proj_cublas_aligned(p, p, p, 4, 0, 2560));
        assert!(!bf16_proj_cublas_aligned(p, p, p, 4, 640, 0));
        assert!(!bf16_proj_cublas_aligned(p, p, p, 4, 640, 2564));
        assert!(!bf16_proj_cublas_aligned(
            DevicePtr::NULL,
            p,
            p,
            4,
            640,
            2560
        ));
        assert!(!bf16_proj_cublas_aligned(
            p,
            DevicePtr::NULL,
            p,
            4,
            640,
            2560
        ));
        assert!(!bf16_proj_cublas_aligned(
            p,
            p,
            DevicePtr::NULL,
            4,
            640,
            2560
        ));
        assert!(!bf16_proj_cublas_aligned(
            DevicePtr(0x1_0008),
            p,
            p,
            4,
            640,
            2560
        ));
    }
}

#[cfg(test)]
mod debug_row_count_tests {
    use super::DraftRowLayout;
    use crate::weight_loader::DrafterCheckpointFamily;

    #[test]
    fn generic_dflash_keeps_anchor_plus_gamma_masks_and_drops_anchor_output() {
        let layout = DraftRowLayout::for_family(DrafterCheckpointFamily::Dflash, 15);
        assert_eq!(layout.query_rows, 16);
        assert_eq!(layout.output_start, 1);
        assert_eq!(layout.feedback_rows(), 15);
    }

    #[test]
    fn dspark_uses_gamma_total_rows_and_includes_anchor_output() {
        let layout = DraftRowLayout::for_family(DrafterCheckpointFamily::Dspark, 7);
        assert_eq!(layout.query_rows, 7);
        assert_eq!(layout.output_start, 0);
        assert_eq!(layout.feedback_rows(), 6);
    }

    #[test]
    fn dspark_k14_is_one_anchor_plus_thirteen_masks_not_two_blocks() {
        let layout = DraftRowLayout::for_family(DrafterCheckpointFamily::Dspark, 14);
        assert_eq!(layout.query_rows, 14);
        assert_eq!(layout.output_start, 0);
        assert_eq!(layout.feedback_rows(), 13);
    }

    #[test]
    fn c1_stage2_bundle_packs_meta_gumbel_q_base_and_ring_into_one_h2d_and_stage1_flag() {
        use super::{DflashNoiseGraphMeta, NoiseGraphHeadState, parse_dflash_stage1_graph_enabled};
        use spark_runtime::gpu::mock::MockGpuBackend;
        use spark_runtime::gpu::{DevicePtr, GpuBackend};
        use spark_runtime::sampler::gumbel::GumbelDraftSpec;

        assert!(parse_dflash_stage1_graph_enabled(None));
        assert!(parse_dflash_stage1_graph_enabled(Some("1")));
        assert!(!parse_dflash_stage1_graph_enabled(Some("0")));
        assert!(!parse_dflash_stage1_graph_enabled(Some("off")));
        assert!(!parse_dflash_stage1_graph_enabled(Some("FALSE")));

        let gpu = MockGpuBackend::new();
        let mut state = NoiseGraphHeadState::new();
        let meta = DflashNoiseGraphMeta::try_new(128, 42, &[128, 129, 130, 131], 4096).unwrap();
        let k_ptrs: Vec<DevicePtr> = (0..5).map(|i| DevicePtr(0x10_0000 + i * 0x1000)).collect();
        let v_ptrs: Vec<DevicePtr> = (0..5).map(|i| DevicePtr(0x20_0000 + i * 0x1000)).collect();
        let spec = GumbelDraftSpec {
            seed: 0x0102_0304_0506_0708,
            step0_abs_pos: 128,
            temperature: 0.7,
            top_k: 32,
            top_p: 0.95,
            min_p: 0.05,
        };

        // Warm mask upload once; subsequent steady-state bundle upload must issue 1 H2D.
        let _ = state
            .upload_c1_stage2_bundle(
                &gpu,
                &meta,
                999,
                Some((64, &k_ptrs, &v_ptrs)),
                Some(spec),
                17,
                5,
                0,
            )
            .unwrap();
        let ptrs = state
            .upload_c1_stage2_bundle(
                &gpu,
                &meta,
                999,
                Some((72, &k_ptrs, &v_ptrs)),
                Some(spec),
                23,
                5,
                0,
            )
            .unwrap();

        let mut u32_buf = [0u8; 4];
        let mut u64_buf = [0u8; 8];
        gpu.copy_d2h(
            ptrs.meta_dev.offset(DflashNoiseGraphMeta::EFF_CTX_OFFSET),
            &mut u32_buf,
        )
        .unwrap();
        assert_eq!(u32::from_le_bytes(u32_buf), 128);
        gpu.copy_d2h(
            ptrs.meta_dev
                .offset(DflashNoiseGraphMeta::LAST_TOKEN_OFFSET),
            &mut u32_buf,
        )
        .unwrap();
        assert_eq!(u32::from_le_bytes(u32_buf), 42);
        gpu.copy_d2h(ptrs.gumbel_ptrs.seeds, &mut u64_buf).unwrap();
        assert_eq!(u64::from_le_bytes(u64_buf), spec.seed);
        gpu.copy_d2h(ptrs.gumbel_ptrs.step0_pos, &mut u64_buf)
            .unwrap();
        assert_eq!(u64::from_le_bytes(u64_buf), 128);
        gpu.copy_d2h(ptrs.gumbel_ptrs.temps, &mut u32_buf).unwrap();
        assert_eq!(f32::from_le_bytes(u32_buf).to_bits(), 0.7f32.to_bits());
        gpu.copy_d2h(ptrs.gumbel_ptrs.top_ks, &mut u32_buf).unwrap();
        assert_eq!(i32::from_le_bytes(u32_buf), 32);
        gpu.copy_d2h(ptrs.q_row_base_dev, &mut u32_buf).unwrap();
        assert_eq!(u32::from_le_bytes(u32_buf), 23);
        let ring = ptrs.ring_meta_ptrs.expect("ring_meta_ptrs present");
        gpu.copy_d2h(ring.ring_start_dev, &mut u32_buf).unwrap();
        assert_eq!(u32::from_le_bytes(u32_buf), 72);
        gpu.copy_d2h(ring.layer_k_ptr(4), &mut u64_buf).unwrap();
        assert_eq!(u64::from_le_bytes(u64_buf), k_ptrs[4].0);
        gpu.copy_d2h(ring.layer_v_ptr(4), &mut u64_buf).unwrap();
        assert_eq!(u64::from_le_bytes(u64_buf), v_ptrs[4].0);
    }
}
