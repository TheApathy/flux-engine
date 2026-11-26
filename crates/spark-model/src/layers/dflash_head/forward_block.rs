// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash γ-block forward (Phase 2 kernel chain). Split out of
//! `dflash_head.rs` for file-size budget — body still exceeds the
//! 500 LoC target because the per-step kernel chain (fc → pos →
//! 8 drafter layers → final norm/lm_head/argmax → D2H) shares
//! many locals with no clean extraction boundary.

use anyhow::{Context, Result};

use super::{BlockDiffusionDraftHead, DflashProposerState};
use crate::layer::ForwardContext;

pub(super) fn checked_topk_difference(left: f32, right: f32) -> Result<f32> {
    if left.is_nan() || right.is_nan() {
        anyhow::bail!("top-K difference contains NaN");
    }
    if left == right {
        return Ok(0.0);
    }
    let difference = left - right;
    if difference.is_nan() {
        anyhow::bail!("top-K difference produced NaN");
    }
    Ok(difference)
}

pub(super) fn validate_topk_request(k: usize, vocab: usize) -> Result<usize> {
    if vocab == 0 {
        anyhow::bail!("top-K requires vocab > 0");
    }
    if vocab > u32::MAX as usize {
        anyhow::bail!("top-K vocab exceeds u32: {vocab}");
    }
    if k == 0 {
        anyhow::bail!("top-K requires k >= 1");
    }
    if k > super::DDTREE_TOP_K_MAX {
        anyhow::bail!("top-K requires k <= {}, got {k}", super::DDTREE_TOP_K_MAX);
    }
    if k > vocab {
        anyhow::bail!("top-K requires k <= vocab ({vocab}), got {k}");
    }
    Ok(k)
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct TopkLayout {
    pub(super) num_rows: u32,
    pub(super) vocab: u32,
    pub(super) k: usize,
    pub(super) bytes: usize,
}

pub(super) fn checked_topk_layout(
    num_rows: usize,
    scratch_rows: usize,
    vocab: usize,
    k: usize,
) -> Result<TopkLayout> {
    if num_rows > scratch_rows {
        anyhow::bail!("top-K rows {num_rows} exceed scratch capacity {scratch_rows}");
    }
    let k = validate_topk_request(k, vocab)?;
    let num_rows = u32::try_from(num_rows).context("top-K row count exceeds u32")?;
    let bytes = (num_rows as usize)
        .checked_mul(k)
        .and_then(|elements| elements.checked_mul(4))
        .context("top-K result byte length overflow")?;
    Ok(TopkLayout {
        num_rows,
        vocab: vocab as u32,
        k,
        bytes,
    })
}

impl BlockDiffusionDraftHead {
    /// `async_launch` (ATLAS_DFLASH_ASYNC, task #20): when true, every op is
    /// enqueued on `stream` (the dedicated propose stream, already ordered
    /// after the default stream via an event) and the function returns
    /// IMMEDIATELY after the noise-pass loop — the final synchronize + drafts
    /// D2H (and any host-side post-processing) is deferred to
    /// `collect_async_drafts_impl` at the top of the next scheduler step.
    /// The launch-eligibility gate guarantees none of the host-interactive
    /// features (markov / denoise>1 / margin gate / topk builders / debug
    /// dumps) are active on this path. Returns an empty Vec sentinel.
    pub(super) fn forward_block(
        &self,
        last_token: u32,
        position: usize,
        gamma_eff: usize,
        ctx: &ForwardContext,
        stream: u64,
        dstate: &mut DflashProposerState,
        async_launch: bool,
    ) -> Result<Vec<u32>> {
        use crate::layers::ops;

        let h = self.hidden_size as u32;
        let bf16 = 2usize;
        let gpu = ctx.gpu;

        // ── Kernel profiler (ATLAS_DFLASH_KERNEL_PROFILE=1 or ATLAS_FULL_PROFILE=1) ──
        // Lightweight per-phase timing using synchronize-and-Instant. Matches
        // `qwen3_ssm/ssm_forward.rs:prof!` pattern. Aggregates per-kernel μs
        // sums across all drafter layers via a thread-local accumulator,
        // logged at the end of forward_block. Adds ~20μs/sync × ~14 syncs/layer
        // when enabled — only use for measurement, not production.
        let kprofile = super::kernel_profile_enabled();
        let policy = self
            .proposal_policy
            .get_or_init(super::proposal_policy::ProposalPolicy::from_env);
        let t_total = std::time::Instant::now();
        if kprofile {
            gpu.synchronize(stream)?;
        }
        let t_pre_layers = std::time::Instant::now();
        super::kprof_reset_layers();

        // Determine effective ctx_len: capped by the configured ctx_window
        // and the accumulator's actual fill. Use the LAST `eff_ctx` ctx
        // positions (most recent) — drafter trained on locally recent
        // context, distant history adds noise to attention.
        // ATLAS_DFLASH_DEBUG_CTX_OFF=1 disables ctx entirely (eff_ctx=0)
        // for A/B testing whether the drafter actually responds to ctx.
        let force_no_ctx = policy.force_no_ctx;
        let force_ctx_used = policy.force_ctx_used;
        let ring = dstate.ctx_ring_state()?;
        anyhow::ensure!(
            position == ring.absolute_len,
            "DFlash proposal position {position} does not match target-hidden absolute cursor {}",
            ring.absolute_len,
        );
        let (ctx_base_ptr, ctx_total, eff_ctx) = if ring.resident_len > 0 && !force_no_ctx {
            let n = ring.resident_len;
            let eff = match force_ctx_used {
                Some(forced) => forced.min(n).min(self.ctx_window).min(dstate.ctx_capacity),
                None => n.min(self.ctx_window).min(dstate.ctx_capacity),
            };
            (
                (eff > 0).then_some(dstate.ctx_hidden_acc),
                ring.absolute_len,
                eff,
            )
        } else {
            (None, ring.absolute_len, 0)
        };
        // Noise block layout (vLLM PR #40898 alignment): (γ_eff+1) query
        // tokens = 1 bonus (last_token) + γ_eff MASK rows. Drafts read from
        // rows [1..γ_eff+1] → γ_eff drafts.
        //
        // `gamma_eff` is resolved exactly once by the outer proposal budget.
        // Re-reading env/runtime state here could exceed the caller's public
        // maximum or the verify buffers after an async launch.
        if gamma_eff == 0 || gamma_eff > self.gamma {
            anyhow::bail!(
                "DFlash forward width {gamma_eff} is outside trained capacity 1..={}",
                self.gamma
            )
        }
        if gamma_eff
            .checked_add(1)
            .is_none_or(|verify_k| verify_k > self.physical_verify_k)
        {
            anyhow::bail!(
                "DFlash forward width {gamma_eff} exceeds physical verify K={}",
                self.physical_verify_k
            )
        }
        let logits_layout = super::logits_layout::LogitsLayout::new(
            self.gamma,
            self.target_vocab_size,
            self.vocab_size,
        )?;
        let _active_logits_bytes = logits_layout.active_bytes(gamma_eff)?;
        let row_layout = super::DraftRowLayout::for_family(self.checkpoint_family, gamma_eff);
        let noise_count = row_layout.query_rows;
        let n_attn = (eff_ctx + noise_count) as u32;

        // Scratch buffers are producer-owned. The active ranges are fully
        // overwritten by the embed, cache-copy/GEMM, attention, LM-head, or
        // argmax producers below; context rows that must be zero are cleared
        // immediately after their producer. Clearing the full allocations
        // here only adds launches and global-memory traffic to every propose.

        let target_hidden_dim = self.target_layer_ids.len() * self.target_hidden_size;
        let ctx_slot_bytes = target_hidden_dim * bf16;

        // Debug dump gated by env var: prints first 10 BF16 floats of key
        // intermediates so a Python reference run on the same checkpoint
        // can be compared element-wise. Use ATLAS_DFLASH_DEBUG_DUMP=1.
        let debug_dump = policy.debug_dump;
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

        // ── Step 0: fc projection of captured target hiddens ──
        // For each of the `eff_ctx` most-recent ctx positions, run a GEMV
        // through `self.fc` (input: 10240 BF16 → output: 2048 BF16) and
        // then per-row RMSNorm through `self.hidden_norm`. Results land
        // contiguously in `scratch.fc_proj` shaped `[eff_ctx, hidden]`.
        let mut pos_suffix_only: Option<(usize, usize)> = None;
        if let Some(base) = ctx_base_ptr {
            let needed_start = ctx_total
                .checked_sub(eff_ctx)
                .ok_or_else(|| anyhow::anyhow!("DFlash resident tail underflowed"))?;
            let ctx_gather = ring.plan_gather(needed_start, eff_ctx)?;
            // ATLAS_DFLASH_ZERO_LATE_LAYERS=N zeros out the LAST N capture
            // layer slots per ctx position before the fc projection. This
            // is a workaround for SSM kernel numerical drift that
            // compounds layer-by-layer in Atlas vs HF transformers
            // reference: by L61 (the 5th capture layer of [1,16,31,46,61]),
            // cosine similarity drops to 0.86 — drafter sees OOD input.
            // Zeroing zeros out the corresponding fc input rows so the
            // drafter only conditions on the cleaner early-layer captures.
            // Effective drafter input dim drops from 5*hidden to (5-N)*hidden.
            let zero_late = policy.zero_late;
            // Optional diagnostic overrides for context hidden captures (layout
            // must be [n_ctx, 5*HIDDEN] BF16, matching `ctx_hidden_acc` slot
            // semantics):
            // - `ATLAS_DFLASH_HF_OVERRIDE=<path>`: load reference context
            //   hiddens from a local binary file for A/B comparison against
            //   native captures.
            // - `ATLAS_DFLASH_FLA_SIDECAR=http://host:port`: fetch reference
            //   context hiddens from an external HF/FLA HTTP sidecar.
            if eff_ctx > 0
                && let Some(p) = policy.hf_override.as_deref()
            {
                let needed = eff_ctx * ctx_slot_bytes;
                match std::fs::read(p) {
                    Ok(hf_bytes) if hf_bytes.len() >= needed => {
                        for span in ctx_gather.spans() {
                            let src_start = span.dst_slot * ctx_slot_bytes;
                            let byte_count = span.slot_count * ctx_slot_bytes;
                            gpu.copy_h2d(
                                &hf_bytes[src_start..src_start + byte_count],
                                base.offset(span.src_slot * ctx_slot_bytes),
                            )?;
                        }
                        dstate.cache_fc_start = needed_start;
                        dstate.cache_fc_end = needed_start;
                        if debug_dump {
                            tracing::info!(
                                "DFLASH HF_OVERRIDE: loaded {} bytes from {}",
                                needed,
                                p,
                            );
                        }
                    }
                    Ok(hf_bytes) => tracing::warn!(
                        "DFLASH HF_OVERRIDE file too small ({} < needed {})",
                        hf_bytes.len(),
                        needed,
                    ),
                    Err(e) => tracing::warn!("DFLASH HF_OVERRIDE read failed: {e}"),
                }
            }
            if zero_late > 0 && eff_ctx > 0 {
                // NOT converted to memset_async, unlike every other memset in
                // this file. Two reasons:
                //
                //  1. `base` is `dstate.ctx_hidden_acc`, which is written by
                //     the TARGET model's capture hook (`try_dflash_capture`,
                //     impl_b3.rs:863) on the target's stream — not by us on
                //     `stream`. Making this stream-ordered would order it
                //     against drafter work while leaving it unordered against
                //     the producer, which is the wrong guarantee to keep. The
                //     blocking form at least serialises against everything.
                //  2. ATLAS_DFLASH_ZERO_LATE_LAYERS defaults to 0, so this
                //     never runs in production and the conversion buys nothing.
                //
                // SEPARATE HAZARD if anyone does enable it: this is
                // `eff_ctx * n_zero` BLOCKING host syncs — at the champion's
                // ctx_window=4096 with n_zero=2 that is ~8k host round-trips
                // in one propose step. Batch it (one memset per contiguous
                // run, or a strided-zero kernel) before using it for anything
                // beyond a one-off A/B.
                let n_capture = self.target_layer_ids.len();
                let n_zero = zero_late.min(n_capture);
                let h_bytes = self.target_hidden_size * bf16;
                for slot_i in 0..eff_ctx {
                    let absolute_position = needed_start + slot_i;
                    let physical_slot = ring.slot_for(absolute_position)?;
                    let slot_base = base.offset(physical_slot * ctx_slot_bytes);
                    // Zero the LAST n_zero layer slices (indices n_capture-n_zero .. n_capture)
                    for layer_i in (n_capture - n_zero)..n_capture {
                        let layer_ptr = slot_base.offset(layer_i * h_bytes);
                        gpu.memset(layer_ptr, 0, h_bytes)?;
                    }
                }
                dstate.cache_fc_start = needed_start;
                dstate.cache_fc_end = needed_start;
                if debug_dump {
                    tracing::info!(
                        "DFLASH ZERO_LATE: zeroed last {} of {} capture layers across {} ctx slots",
                        n_zero,
                        n_capture,
                        eff_ctx
                    );
                }
            }
            // ATLAS_DFLASH_DEBUG_FORCE_PATTERN=1 overwrites the captured
            // target_hidden_stack with a deterministic test pattern so a
            // PyTorch reference run on the same input produces directly
            // comparable intermediates. Pattern: row i, col j contains
            // `0.01 * (i+1) * (j+1) / target_hidden` BF16. Mirrors
            // `dflash_pytorch_reference.py:make_input_target_hidden_stack`.
            let force_pattern = policy.force_pattern;
            if force_pattern && eff_ctx > 0 {
                let n_rows = self.target_layer_ids.len();
                let n_cols = self.target_hidden_size;
                let mut bytes = Vec::with_capacity(n_rows * n_cols * 2);
                for i in 0..n_rows {
                    for j in 0..n_cols {
                        let v = 0.01_f32 * ((i + 1) as f32) * ((j + 1) as f32) / (n_cols as f32);
                        // f32 → bf16 (truncate-to-zero of low 16 bits).
                        let bits = v.to_bits();
                        let bf16_bits = (bits >> 16) as u16;
                        bytes.extend_from_slice(&bf16_bits.to_le_bytes());
                    }
                }
                let physical_slot = ring.slot_for(needed_start)?;
                gpu.copy_h2d(&bytes, base.offset(physical_slot * ctx_slot_bytes))?;
                dstate.cache_fc_start = needed_start;
                dstate.cache_fc_end = needed_start;
            }
            // Dump the FIRST ctx slot's input target_hidden_stack (first 10 floats).
            if eff_ctx > 0 {
                let physical_slot = ring.slot_for(needed_start)?;
                dump_bf16(
                    "step0.input.target_hidden_stack[0]",
                    base.offset(physical_slot * ctx_slot_bytes),
                    10,
                )?;
            }
            // ATLAS_DFLASH_DEBUG_DUMP_FULL=1: write the full 10240-element
            // target_hidden_stack (one ctx slot) to /tmp/atlas_target_hidden.bin
            // so a Python reference can run dflash.py forward on the same
            // input and compare predicted draft tokens vs Atlas drafts.
            // Also dumps last_token + drafter outputs separately for the
            // bisect script. ONE-SHOT: writes only the first propose() call.
            static FULL_DUMP_DONE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if eff_ctx > 0
                && !FULL_DUMP_DONE.load(std::sync::atomic::Ordering::Relaxed)
                && policy.debug_dump_full
                // Mirror the tokens-dump gate: defer until position >= N.
                && position >= policy.dump_min_pos
            {
                // Dump ALL eff_ctx slots — needed to reproduce the
                // multi-token ctx in PyTorch reference. Layout:
                // contiguous BF16, eff_ctx slots × 5 layers × 2048 dims.
                let n_bytes = eff_ctx * ctx_slot_bytes;
                let mut buf = vec![0u8; n_bytes];
                gpu.synchronize(stream)?;
                for span in ctx_gather.spans() {
                    let dst_start = span.dst_slot * ctx_slot_bytes;
                    let byte_count = span.slot_count * ctx_slot_bytes;
                    gpu.copy_d2h(
                        base.offset(span.src_slot * ctx_slot_bytes),
                        &mut buf[dst_start..dst_start + byte_count],
                    )?;
                }
                if let Err(e) = std::fs::write("/tmp/atlas_target_hidden.bin", &buf) {
                    tracing::warn!("DFLASH DUMP_FULL: target_hidden write failed: {e}");
                } else {
                    tracing::info!(
                        "DFLASH DUMP_FULL: wrote {} bytes ({} ctx slots × {} BF16 elements) to /tmp/atlas_target_hidden.bin (last_token={}, position={}, eff_ctx={})",
                        n_bytes,
                        eff_ctx,
                        ctx_slot_bytes / 2,
                        last_token,
                        position,
                        eff_ctx,
                    );
                }
                // Sibling metadata JSON so the Python reference can
                // reconstruct the exact propose() inputs without parsing
                // tracing output. Same one-shot guard as the binary.
                let meta = format!(
                    "{{\"last_token\":{},\"position\":{},\"eff_ctx\":{},\"ctx_total\":{},\"gamma\":{},\"mask_token_id\":{},\"hidden\":{},\"target_hidden_size\":{},\"n_target_layers\":{}}}",
                    last_token,
                    position,
                    eff_ctx,
                    ctx_total,
                    self.gamma,
                    self.mask_token_id,
                    self.hidden_size,
                    self.target_hidden_size,
                    self.target_layer_ids.len(),
                );
                if let Err(e) = std::fs::write("/tmp/atlas_dump_meta.json", &meta) {
                    tracing::warn!("DFLASH DUMP_FULL: meta write failed: {e}");
                }
                FULL_DUMP_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            // Persistent fc_proj cache: copy old positions, compute new ones.
            let resident_start = ring.resident_start();
            let mut resident_cache_start = dstate.cache_fc_start.max(resident_start);
            let mut resident_cache_end = dstate.cache_fc_end.min(ring.absolute_len);
            if dstate.cache_fc_end > ring.absolute_len
                || resident_cache_start > resident_cache_end
                || resident_cache_end - resident_cache_start > self.ctx_window
            {
                resident_cache_start = needed_start;
                resident_cache_end = needed_start;
            }
            dstate.cache_fc_start = resident_cache_start;
            dstate.cache_fc_end = resident_cache_end;
            let old_fc_end =
                if resident_cache_start <= needed_start && resident_cache_end >= needed_start {
                    (needed_start + eff_ctx).min(resident_cache_end)
                } else {
                    needed_start
                };
            let old_fc_count = old_fc_end.saturating_sub(needed_start);
            let new_fc_count = eff_ctx.saturating_sub(old_fc_count);
            // PERF (2026-05-19): demoted from tracing::info! — fired on every
            // propose() call (~80×/s during decode), drowning real diagnostics.
            // Re-enable via RUST_LOG=spark_model::layers::dflash_head=debug.
            tracing::debug!(
                "DFlash fc_proj cache: needed=[{}..{}), cached=[{}..{}), old={}, new={}",
                needed_start,
                needed_start + eff_ctx,
                dstate.cache_fc_start,
                dstate.cache_fc_end,
                old_fc_count,
                new_fc_count,
            );

            let skip_old_fc_copy = old_fc_count > 0
                && !kprofile
                && self.noise_graph_can_skip_old_fc_copy(
                    dstate,
                    policy,
                    needed_start,
                    old_fc_count,
                    eff_ctx,
                    gamma_eff,
                );
            if skip_old_fc_copy {
                pos_suffix_only = Some((old_fc_count, new_fc_count));
            }
            if old_fc_count > 0 && !skip_old_fc_copy {
                self.cache_copy_range(
                    gpu,
                    dstate.ctx_fc_cache,
                    dstate.cache_fc_start,
                    dstate.cache_fc_end,
                    self.ctx_window,
                    needed_start,
                    old_fc_end,
                    self.scratch.fc_proj,
                    self.hidden_size * bf16,
                    stream,
                )?;
            }
            // EAGLE-3.1 per-layer FC-normalization (vLLM 2026-05-26):
            // ATLAS_DFLASH_FC_LAYERNORM=1 (default OFF). When the fc layer
            // consumes a CONCATENATION of multiple target-layer hidden states,
            // "the fused input becomes increasingly imbalanced as higher-layer
            // hidden states dominate" (larger magnitude). The fix is to
            // normalize EACH captured target hidden state independently
            // BEFORE the fc layer. DFlash concatenates the 5 raw captures
            // [layers 1,16,31,46,61] into one [5*2048] BF16 stack feeding ONE
            // fc GEMV — exactly the broken thing. Here we apply a per-slice
            // unit-variance RMSNorm (BF16 `1.0` weights for `rms_norm_vanilla`
            // → `x / RMS(x)`, no learned gamma) to each of the n_capture layer
            // slices so no single late-layer capture's magnitude dominates the
            // fused fc input. Drafter-side only → target verify unchanged →
            // token-exact; raises ACCEPTANCE.
            //
            // OOD caveat: self.fc was TRAINED on UN-normalized concat, so this
            // is out-of-distribution and may help or hurt — measured A/B.
            // Variant (a): plain unit-variance (no learned norm weight).
            let fc_layernorm = policy.fc_layernorm;
            if new_fc_count > 0 {
                // Batched fast path (task #64): when NOT applying per-slice
                // FC-norm, every new context position reads a contiguous
                // [target_hidden_dim] source row from `base` (stride
                // ctx_slot_bytes = target_hidden_dim*bf16 = the GEMM K) and
                // writes a contiguous [hidden_size] dst row into `fc_proj`
                // (stride h*bf16 = the GEMM N). So the whole per-position GEMV
                // loop — one ~105MB fc-weight read PER position — collapses to a
                // SINGLE w4a16_gemm over M=new_fc_count rows: one weight read.
                // Bit-exact vs the loop (same weight, same (n=h, k=K) mapping,
                // M_TILE=64 handles arbitrary M with `if (r < M)` guards); the
                // gemv's n/k args map to the gemm's n/k unchanged. The
                // fc_layernorm path (default OFF) still needs per-position
                // RMSNorm into a 1-position scratch buffer, so it keeps the
                // loop below.
                let batched_nvfp4 = !fc_layernorm
                    && matches!(self.quant, super::DflashQuantization::Nvfp4)
                    && self.fc_nvfp4.is_some();
                let batched_fp8 = !fc_layernorm
                    && matches!(self.quant, super::DflashQuantization::Fp8)
                    && self.fc_fp8.is_some();
                if batched_nvfp4 {
                    let fc_q = self
                        .fc_nvfp4
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("DFlash NVFP4 path missing fc_nvfp4"))?;
                    let new_gather = ring.plan_gather(old_fc_end, new_fc_count)?;
                    let mut projected = 0usize;
                    for span in new_gather.spans() {
                        let src = base.offset(span.src_slot * ctx_slot_bytes);
                        let dst = self
                            .scratch
                            .fc_proj
                            .offset((old_fc_count + span.dst_slot) * self.hidden_size * bf16);
                        if super::proposal_policy::use_prefill_pipe(
                            policy.prefill_pipe,
                            self.kernels.w4a16_gemm_pipe.0 != 0,
                            span.slot_count,
                            target_hidden_dim,
                        )
                        .map_err(anyhow::Error::msg)?
                        {
                            static FC_PIPE_SEEN: std::sync::Once = std::sync::Once::new();
                            FC_PIPE_SEEN.call_once(|| {
                                tracing::info!(
                                    "DFLASH_PREFILL_PIPE engaged for V3 FC context ingestion: \
                                     M={}, N={}, K={}",
                                    span.slot_count,
                                    h,
                                    target_hidden_dim,
                                );
                            });
                            ops::w4a16_gemm_pipe(
                                gpu,
                                self.kernels.w4a16_gemm_pipe,
                                src,
                                fc_q,
                                dst,
                                span.slot_count as u32,
                                h,
                                target_hidden_dim as u32,
                                stream,
                            )?;
                        } else if crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                            gpu,
                            src,
                            fc_q,
                            dst,
                            span.slot_count as u32,
                            h,
                            target_hidden_dim as u32,
                            32,
                            stream,
                        )? {
                            static FC_COALESCED_SM120_SEEN: std::sync::Once =
                                std::sync::Once::new();
                            FC_COALESCED_SM120_SEEN.call_once(|| {
                                tracing::info!(
                                    "DFLASH_FC_COALESCED_BF16 engaged for small-M FC context ingestion: \
                                     M={}, N={}, K={}",
                                    span.slot_count,
                                    h,
                                    target_hidden_dim,
                                );
                            });
                        } else if atlas_core::device::active_sm_count() <= 64
                            && should_use_fc_nvfp4_t(
                                span.slot_count,
                                self.hidden_size,
                                target_hidden_dim,
                                FcTransposeKernelCaps {
                                    has_m32_n64: self.kernels.w4a16_gemm_t_m32_n64.0 != 0,
                                    has_m16_n128: self.kernels.w4a16_gemm_t_m16.0 != 0,
                                },
                            )
                            && let Some(fc_t) = super::from_weights::lookup_fc_nvfp4_t(fc_q.weight)
                        {
                            static FC_KGAMMA_T_SEEN: std::sync::Once = std::sync::Once::new();
                            FC_KGAMMA_T_SEEN.call_once(|| {
                                tracing::info!(
                                    "DFLASH_FC_KGAMMA_T engaged for small-M FC context ingestion: \
                                     M={}, N={}, K={}",
                                    span.slot_count,
                                    h,
                                    target_hidden_dim,
                                );
                            });
                            self.draft_gemm_t(
                                gpu,
                                src,
                                &fc_t,
                                dst,
                                span.slot_count as u32,
                                h,
                                target_hidden_dim as u32,
                                true,
                                stream,
                            )?;
                        } else {
                            self.nvfp4_project_coalesced_or_gemm(
                                gpu,
                                src,
                                fc_q,
                                dst,
                                span.slot_count as u32,
                                h,
                                target_hidden_dim as u32,
                                stream,
                            )?;
                        }
                        projected += span.slot_count;
                    }
                    if projected != new_fc_count {
                        anyhow::bail!(
                            "DFlash FC projection range retained {projected} of {new_fc_count} slots"
                        );
                    }
                } else if batched_fp8 {
                    let fc_q = self
                        .fc_fp8
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("DFlash FP8 path missing fc_fp8"))?;
                    let new_gather = ring.plan_gather(old_fc_end, new_fc_count)?;
                    let mut projected = 0usize;
                    for span in new_gather.spans() {
                        let src = base.offset(span.src_slot * ctx_slot_bytes);
                        let dst = self
                            .scratch
                            .fc_proj
                            .offset((old_fc_count + span.dst_slot) * self.hidden_size * bf16);
                        if span.slot_count == 1 && self.kernels.dense_gemv_fp8w.0 != 0 {
                            ops::dense_gemv_fp8w(
                                gpu,
                                self.kernels.dense_gemv_fp8w,
                                src,
                                fc_q,
                                dst,
                                h,
                                target_hidden_dim as u32,
                                stream,
                            )?;
                        } else {
                            ops::fp8_gemm_dflash_rowscale(
                                gpu,
                                self.kernels.fp8_gemm_dflash,
                                src,
                                fc_q,
                                dst,
                                span.slot_count as u32,
                                h,
                                target_hidden_dim as u32,
                                stream,
                            )?;
                        }
                        projected += span.slot_count;
                    }
                    if projected != new_fc_count {
                        anyhow::bail!(
                            "DFlash FP8 FC projection range retained {projected} of {new_fc_count} slots"
                        );
                    }
                }
                // Per-position fallback: fc_layernorm path (per-slice RMSNorm)
                // and the dense (BF16) path. Skipped when batched NVFP4/FP8
                // already computed fc_proj.
                for i in 0..new_fc_count {
                    if batched_nvfp4 || batched_fp8 {
                        break;
                    }
                    let abs_pos = old_fc_end + i;
                    let physical_slot = ring.slot_for(abs_pos)?;
                    let raw_slot = base.offset(physical_slot * ctx_slot_bytes);
                    // Per-layer FC-norm: copy each of the n_capture target-layer
                    // slices [target_hidden_size] into fc_norm_in, unit-variance
                    // RMS-normalized independently, then feed the normalized
                    // concat to the fc GEMV instead of the raw slot. Mirrors the
                    // ZERO_LATE_LAYERS per-layer slicing of `ctx_hidden_acc`.
                    let src_slot = if fc_layernorm {
                        let n_capture = self.target_layer_ids.len();
                        let h_elems = self.target_hidden_size as u32;
                        for layer_i in 0..n_capture {
                            let in_ptr = raw_slot.offset(layer_i * self.target_hidden_size * bf16);
                            let out_ptr = self
                                .scratch
                                .fc_norm_in
                                .offset(layer_i * self.target_hidden_size * bf16);
                            ops::rms_norm(
                                gpu,
                                self.kernels.rms_norm,
                                in_ptr,
                                &crate::weight_map::DenseWeight {
                                    weight: self.scratch.fc_norm_ones_w,
                                },
                                out_ptr,
                                1,
                                h_elems,
                                self.rms_norm_eps,
                                stream,
                            )?;
                        }
                        self.scratch.fc_norm_in
                    } else {
                        raw_slot
                    };
                    let dst_slot = self
                        .scratch
                        .fc_proj
                        .offset((old_fc_count + i) * self.hidden_size * bf16);
                    match (self.quant, self.fc_nvfp4.as_ref(), self.fc_fp8.as_ref()) {
                        (super::DflashQuantization::Nvfp4, Some(fc_q), _) => ops::w4a16_gemv(
                            gpu,
                            self.kernels.w4a16_gemv,
                            src_slot,
                            fc_q,
                            dst_slot,
                            h,
                            target_hidden_dim as u32,
                            stream,
                        )?,
                        (super::DflashQuantization::Fp8, _, Some(fc_q)) => ops::dense_gemv_fp8w(
                            gpu,
                            self.kernels.dense_gemv_fp8w,
                            src_slot,
                            fc_q,
                            dst_slot,
                            h,
                            target_hidden_dim as u32,
                            stream,
                        )?,
                        _ => ops::dense_gemv(
                            gpu,
                            self.kernels.dense_gemv,
                            src_slot,
                            &self.fc,
                            dst_slot,
                            h,
                            target_hidden_dim as u32,
                            stream,
                        )?,
                    }
                }
                if old_fc_count == 0 {
                    dump_bf16("step0.fc_proj.pre_norm[0]", self.scratch.fc_proj, 10)?;
                } else {
                    dump_bf16(
                        "step0.fc_proj.pre_norm[new0]",
                        self.scratch
                            .fc_proj
                            .offset(old_fc_count * self.hidden_size * bf16),
                        10,
                    )?;
                }
                // RMSNorm on the newly-computed slice.
                ops::rms_norm(
                    gpu,
                    self.kernels.rms_norm,
                    self.scratch
                        .fc_proj
                        .offset(old_fc_count * self.hidden_size * bf16),
                    &self.hidden_norm,
                    self.scratch
                        .fc_proj
                        .offset(old_fc_count * self.hidden_size * bf16),
                    new_fc_count as u32,
                    h,
                    self.rms_norm_eps,
                    stream,
                )?;
                // Write new fc_proj into persistent cache.
                // Note: keep this small (10-80 KB) async D2D write active whenever
                // `dstate.ctx_fc_cache` is allocated (including PostRope mode):
                // `invalidate_dstate_ctx_kv_ranges` invalidates only K/V caches on
                // PostRope -> PreRope fallback and relies on `ctx_fc_cache` staying
                // populated. When `skip_ctx_fc_cache` is configured, `ctx_fc_cache`
                // is NULL and `cache_write_range` already no-ops.
                let (new_fc_start, new_fc_end) = self.cache_write_range(
                    gpu,
                    self.scratch
                        .fc_proj
                        .offset(old_fc_count * self.hidden_size * bf16),
                    old_fc_end,
                    new_fc_count,
                    dstate.ctx_fc_cache,
                    dstate.cache_fc_start,
                    dstate.cache_fc_end,
                    self.ctx_window,
                    self.hidden_size * bf16,
                    stream,
                )?;
                dstate.cache_fc_start = new_fc_start;
                dstate.cache_fc_end = new_fc_end;
            }
            // RMSNorm on the cached slice (if not already normalized).
            // The cache stores post-norm fc_proj, so cached positions are
            // already normalized. Only the new slice needs norm above.
            if eff_ctx > 0 {
                dump_bf16(
                    "step0.fc_proj.post_hidden_norm[0]",
                    self.scratch.fc_proj,
                    10,
                )?;
            }
        }

        // ── Step 1: build position ids ──
        // Layout: [ctx_pos_0, ..., ctx_pos_{eff_ctx-1}, seq_pos, ..., seq_pos+γ-1].
        // ctx_pos_i = position - eff_ctx + i — the absolute target indices
        // of the captured positions in chronological order.
        // kprofile sub-phase marker: end of Step 0 (fc_proj) host enqueue.
        let t_pre_fc_done = std::time::Instant::now();
        // Stream-ordered pinned H2D instead of `copy_h2d`: the sync variant
        // drains `default_stream` (the target verify stream) on every propose
        // — measured ~8.7 ms of hidden serialization per cycle. The async
        // copy enqueues after earlier work on `stream`, and `pos_pinned` is
        // only rewritten by the next propose after this pass has been
        // synchronized (sync path) or drained (async path), so ordering is
        // preserved without any host-side wait.
        struct PinnedH2dDrainGuard<'a> {
            gpu: &'a dyn spark_runtime::gpu::GpuBackend,
            stream: u64,
            armed: bool,
        }
        impl Drop for PinnedH2dDrainGuard<'_> {
            fn drop(&mut self) {
                if self.armed {
                    let _ = self.gpu.synchronize(self.stream);
                }
            }
        }
        if let Some((old_fc_count, new_fc_count)) = pos_suffix_only {
            // Two-stage noise graph keeps `[0..old_fc_count)` cached in circular
            // PostRope KV: Stage 1 reads only `position_ids[old_fc_count..eff_ctx]`
            // (`<= 8` u32s) and Stage 2 reads `pos_pinned[eff_ctx..eff_ctx+noise_count]`.
            // Encode only `[old_fc_count..eff_ctx+noise_count)` and upload only the
            // newly appended context suffix (`0..=32 B` instead of full `eff_ctx`).
            let ctx_start = position
                .checked_sub(eff_ctx)
                .ok_or_else(|| anyhow::anyhow!("DFlash position {position} < eff_ctx {eff_ctx}"))?;
            let total_positions = eff_ctx
                .checked_add(noise_count)
                .ok_or_else(|| anyhow::anyhow!("DFlash position count overflow"))?;
            let end_bytes = total_positions
                .checked_mul(size_of::<u32>())
                .ok_or_else(|| anyhow::anyhow!("DFlash position byte count overflow"))?;
            let dst = dstate.pos_pinned.as_mut_slice();
            anyhow::ensure!(
                end_bytes <= dst.len(),
                "DFlash pos_pinned too small: need {end_bytes} B, have {} B",
                dst.len()
            );
            let last_pos = position
                .checked_add(noise_count.saturating_sub(1))
                .ok_or_else(|| anyhow::anyhow!("DFlash noise position overflow"))?;
            anyhow::ensure!(
                last_pos <= i32::MAX as usize,
                "DFlash position {last_pos} exceeds i32::MAX"
            );
            for idx in old_fc_count..eff_ctx {
                let off = idx * size_of::<u32>();
                let val = (ctx_start + idx) as u32;
                dst[off..off + 4].copy_from_slice(&val.to_le_bytes());
            }
            for j in 0..noise_count {
                let off = (eff_ctx + j) * size_of::<u32>();
                let val = (position + j) as u32;
                dst[off..off + 4].copy_from_slice(&val.to_le_bytes());
            }
            if new_fc_count > 0 {
                let byte_off = old_fc_count * size_of::<u32>();
                let byte_len = new_fc_count * size_of::<u32>();
                let pinned = dstate.pos_pinned.pinned_slice_range(byte_off, byte_len)?;
                // SAFETY: `pinned` borrows the page-locked `pos_pinned` buffer owned by
                // `dstate`, which outlives this copy because `pinned_h2d_guard` drains
                // `stream` on any `Err`/unwind return, the synchronous `Ok` path drains
                // `stream` in Step 6 before disarming the guard, and the `async_launch`
                // path transfers stream-drain ownership to `AsyncInflight`. The copy is
                // stream-ordered on `stream`, so the driver reads the pinned bytes only
                // after preceding work on that stream, never concurrently with this
                // host-side write.
                unsafe {
                    gpu.copy_h2d_pinned_async(
                        pinned,
                        self.scratch.position_ids.offset(byte_off),
                        stream,
                    )?;
                }
            }
        } else {
            let pos_n_bytes = super::proposal_inputs::encode_position_ids(
                dstate.pos_pinned.as_mut_slice(),
                position,
                eff_ctx,
                noise_count,
            )?;
            let pinned = dstate.pos_pinned.pinned_slice(pos_n_bytes)?;
            // SAFETY: `pinned` borrows the page-locked `pos_pinned` buffer owned by
            // `dstate`, which outlives this copy because `pinned_h2d_guard` drains
            // `stream` on any `Err`/unwind return, the synchronous `Ok` path drains
            // `stream` in Step 6 before disarming the guard, and the `async_launch`
            // path transfers stream-drain ownership to `AsyncInflight`. The copy is
            // stream-ordered on `stream`, so the driver reads the pinned bytes only
            // after preceding work on that stream, never concurrently with this
            // host-side write.
            unsafe {
                gpu.copy_h2d_pinned_async(pinned, self.scratch.position_ids, stream)?;
            }
        }
        let mut pinned_h2d_guard = PinnedH2dDrainGuard {
            gpu,
            stream,
            armed: true,
        };
        if debug_dump {
            let ctx_start = position.saturating_sub(eff_ctx);
            let pos_preview: Vec<usize> = (0..eff_ctx)
                .map(|i| ctx_start + i)
                .chain((0..noise_count).map(|i| position + i))
                .take(8)
                .collect();
            tracing::info!(
                "DFLASH DUMP positions: eff_ctx={} ctx_total={} position={} pos_ids[0..min(8,n_attn)]={:?}",
                eff_ctx,
                ctx_total,
                position,
                pos_preview
            );
        }

        // ── Steps 2–5: noise-block forward — possibly iterated ──
        //
        // The per-pass body (noise-embedding build → 8 drafter layers →
        // final norm/lm_head → per-row argmax) lives in
        // `noise_pass.rs:run_noise_pass`, extracted verbatim from here.
        //
        // ATLAS_DFLASH_DENOISE_STEPS=N (default 1 = single pass, the
        // original behavior with zero added work) runs the γ-block noise
        // forward N times per propose. After pass k, rows whose argmax is
        // confident (top1−top2 logit margin ≥ ATLAS_DFLASH_DENOISE_MARGIN,
        // default 1.0) are "committed": pass k+1 embeds the predicted token
        // at that row instead of the mask embedding, so the still-masked
        // rows are conditioned on partial predictions (DiffusionGemma-style
        // iterative refinement). Committed rows are frozen into the final
        // drafts (ATLAS_DFLASH_DENOISE_FREEZE=0 takes the last pass's
        // argmax everywhere instead). Early exits: a pass that commits no
        // new rows would re-run on identical input (deterministic → same
        // output), and a pass with ALL rows already committed has nothing
        // masked left to refine — both break out of the loop, so uniformly
        // high pass-1 confidence (counting workloads) pays for at most one
        // extra pass.
        //
        // Cache safety: the per-layer ctx K/V caches and the fc_proj cache
        // are append-once with explicit range tracking. Step 0 (fc
        // projection) runs once above; passes ≥1 re-enter the layer loop
        // where old_ctx_count == eff_ctx and new_ctx_count == 0, so caches
        // are only COPIED from, never re-appended — cache_{k,v,fc}_start/
        // end stay consistent across passes.
        //
        // OOD caveat: the drafter is TRAINED with all-mask noise rows;
        // real-token embeddings at committed rows are out-of-distribution.
        // Measure acceptance A/B before enabling in production.
        //
        // ATLAS_DFLASH_MASK_OVERRIDE: env var override for the mask token ID.
        let mask_id = policy.mask_override.unwrap_or(self.mask_token_id);
        let denoise_steps = policy.denoise_steps;
        if self.checkpoint_family == crate::weight_loader::DrafterCheckpointFamily::Dspark
            && denoise_steps != 1
        {
            anyhow::bail!(
                "DSpark anchor-output layout currently supports exactly one denoise pass"
            );
        }
        let denoise_margin = policy.denoise_margin;
        let denoise_freeze = policy.denoise_freeze;
        let argmax_vocab = self.target_vocab_size.min(self.vocab_size);
        let needed_start = ctx_total
            .checked_sub(eff_ctx)
            .ok_or_else(|| anyhow::anyhow!("DFlash attention tail underflowed"))?;
        let pass_args = super::noise_pass::NoisePassArgs {
            last_token,
            eff_ctx,
            gamma_eff,
            n_attn,
            mask_id,
            needed_start,
            stream,
            debug_dump,
            force_noise_pattern: policy.force_noise_pattern,
            lm_head_nvfp4: policy.lm_head_nvfp4,
            dump_all_layers: policy.dump_all_layers,
            kprofile,
        };
        // A single pass feeds all masks and never consumes committed rows.
        // Keep that production default allocation-free; multi-pass denoise
        // still owns explicit per-row feedback state.
        let mut committed: Vec<Option<u32>> = if denoise_steps == 1 {
            Vec::new()
        } else {
            vec![None; gamma_eff]
        };
        let t_layers = std::time::Instant::now();
        for pass in 0..denoise_steps {
            self.run_noise_pass(&pass_args, &committed, pass, ctx, dstate)?;
            if pass + 1 == denoise_steps {
                break;
            }
            // DeepLoop GPU-side commit-all: run_noise_pass stages argmax and
            // reconstructs draft_tokens_dev on-stream for pass N+1 — no host
            // D2H, async-eligible. Skip the margin-gated D2H path entirely.
            if super::dflash_deeploop_enabled() {
                continue;
            }
            // Confidence feedback: top-2 over this pass's logits gives the
            // per-row top1−top2 margin; the argmax tokens are already in
            // draft_tokens_dev. Cost: one topk launch + ~γ·12 bytes D2H.
            let top2 = checked_topk_layout(gamma_eff, self.gamma, argmax_vocab, 2)?;
            let used_bytes = top2.bytes;
            // Stream-ordered: only consumer is the `topk_bf16` launch on
            // `stream` immediately below, and the host D2H is gated behind
            // the `gpu.synchronize(stream)` that follows it.
            gpu.memset_async(self.scratch.topk_tokens_dev, 0, used_bytes, stream)?;
            gpu.memset_async(self.scratch.topk_logits_dev, 0, used_bytes, stream)?;
            ops::topk_bf16(
                gpu,
                self.kernels.topk,
                self.scratch.logits,
                self.scratch.topk_tokens_dev,
                self.scratch.topk_logits_dev,
                top2.num_rows,
                top2.vocab,
                top2.k as u32,
                stream,
            )?;
            gpu.synchronize(stream)?;
            let mut tok_bytes = vec![0u8; gamma_eff * 4];
            gpu.copy_d2h(self.scratch.draft_tokens_dev, &mut tok_bytes)?;
            let mut top2_token_bytes = vec![0u8; used_bytes];
            let mut top2_logit_bytes = vec![0u8; used_bytes];
            gpu.copy_d2h(self.scratch.topk_tokens_dev, &mut top2_token_bytes)?;
            gpu.copy_d2h(self.scratch.topk_logits_dev, &mut top2_logit_bytes)?;
            let pass_tokens: Vec<u32> = tok_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let (_, top2) = match Self::decode_topk_bytes(
                top2_token_bytes,
                top2_logit_bytes,
                gamma_eff,
                self.gamma,
                top2.k,
                argmax_vocab,
            ) {
                Ok(decoded) => decoded,
                Err(error) => {
                    tracing::warn!(
                        "DFlash denoise top-2 output is invalid ({error:#}); degrading to bootstrap"
                    );
                    return Ok(Vec::new());
                }
            };
            let mut newly_committed = 0usize;
            for i in 0..gamma_eff {
                if committed[i].is_some() {
                    continue;
                }
                let margin = match checked_topk_difference(top2[i * 2], top2[i * 2 + 1]) {
                    Ok(margin) => margin,
                    Err(error) => {
                        tracing::warn!(
                            "DFlash denoise top-2 row {i} is invalid ({error:#}); degrading to bootstrap"
                        );
                        return Ok(Vec::new());
                    }
                };
                if margin >= denoise_margin {
                    committed[i] = Some(pass_tokens[i]);
                    newly_committed += 1;
                }
            }
            let n_committed = committed.iter().filter(|c| c.is_some()).count();
            tracing::debug!(
                "DFlash denoise pass {}: +{} committed ({}/{} rows, margin≥{})",
                pass,
                newly_committed,
                n_committed,
                gamma_eff,
                denoise_margin,
            );
            if newly_committed == 0 || n_committed == gamma_eff {
                break;
            }
        }
        let layers_us = if kprofile {
            gpu.synchronize(stream)?;
            t_layers.elapsed().as_micros()
        } else {
            0
        };

        // ── ATLAS_DFLASH_ASYNC deferred tail ──
        // All drafter kernels (incl. the per-row argmax into
        // `scratch.draft_tokens_dev`) are enqueued on the propose stream.
        // Return now; the synchronize + D2H below runs at collect time.
        // Eligibility guarantees the skipped host-side post-processing
        // (margin gate / markov / denoise freeze / dumps) is inactive.
        if async_launch {
            // Drain ownership transfers to `AsyncInflight` on the propose stream.
            pinned_h2d_guard.armed = false;
            return Ok(Vec::new());
        }

        let t_tail = std::time::Instant::now();
        let dump_all_layers = policy.dump_all_layers;

        // ── Step 5b: optional logit-margin gate (top-1 vs top-2) ──
        //
        // ATLAS_DFLASH_MARGIN_GATE=<float>  (default 0.0 = disabled)
        //   When > 0, runs top-K=2 on the same logits and replaces the draft
        //   token at any row whose (logit_top1 - logit_top2) margin is below
        //   the threshold with `mask_token_id`. Mask is guaranteed to mismatch
        //   the target's argmax → forces the chain to terminate at the first
        //   low-confidence draft. The drafter is essentially saying "I'm
        //   guessing here" — passing the guess to target just to be rejected
        //   wastes verify cycles. Truncating early lets the verifier emit a
        //   bonus token (its own argmax) at the truncation point.
        //
        //   Recommended values: 0.3 (mild), 0.6 (aggressive), 1.0 (very strict).
        //   BF16 logits are O(10-30) for top tokens so margins of 0.3-1.0 are
        //   the meaningful range.
        //
        // ATLAS_DFLASH_ADAPTIVE_GAMMA=1  (default 0 = disabled)
        //   Tracks a moving window of accept counts (last 8 verifies on the
        //   sequence's DflashProposerState) and shrinks the effective draft
        //   count to `clamp(mean + ADAPTIVE_SLACK, ADAPTIVE_MIN, gamma_eff)`.
        //   Excess drafter positions get replaced with `mask_token_id` so the
        //   verifier rejects them and stops the chain — same mechanism as the
        //   margin gate. Saves verifier compute when the drafter is wasted.
        //
        // ATLAS_DFLASH_ADAPTIVE_MIN=<usize>  (default 4)
        //   Lower bound for adaptive gamma. Below 4 the K=γ verify graph
        //   doesn't cache well (too many distinct shapes).
        //
        // ATLAS_DFLASH_ADAPTIVE_SLACK=<usize>  (default 2)
        //   Added to the rolling mean accept count to give the drafter a
        //   little headroom past its recent average — accept rate is bursty.
        let margin_gate = policy.margin_gate;
        let adaptive_gamma = policy.adaptive_gamma;
        let tps_router = policy.tps_router;
        let adaptive_min = policy.adaptive_min;
        let adaptive_slack = policy.adaptive_slack;
        // ATLAS_DFLASH_ADAPTIVE_MAX=<usize> caps the adaptive cutoff so the
        // K=γ verify graph never fires when its wall-time exceeds the
        // throughput benefit. K=γ=16 verify on GB10 is ~2.8s/call vs
        // K=γ=4's 421ms — at 38% accept (γ=16 prose), K=16 emits only
        // ~7 tokens in 2.8s = 2.5 tok/s, dragging mean throughput far
        // below the K=4 path. ADAPTIVE_MAX=4 hard-caps cutoff so the
        // scheduler always lands on a faster verify graph. Default 0 =
        // unbounded (legacy behavior, allows up to gamma_eff).
        let adaptive_max = policy.adaptive_max;
        // ATLAS_DFLASH_ADAPTIVE_PROBE_INTERVAL=<usize> (default 0 = off)
        //   Every N adaptive-engaged steps, force cutoff = gamma_eff
        //   (= no truncation, no masking from adaptive shrink) so the
        //   verifier observes the TRUE accept ceiling for the current
        //   context. Without this, adaptive truncate is self-limiting:
        //   a low-accept burst on prose shrinks γ_eff, which fills
        //   `accept_history` with small accept counts from the truncated
        //   steps, which keeps the cutoff small forever — even after
        //   the content turns predictable (counting, lists, repeating
        //   structure). Periodic γ_max reprobes break the trap: if the
        //   content is still hard, the reprobe step contributes one
        //   small accept count and we truncate again next step; if the
        //   content has become easy, the reprobe lands a long accept
        //   chain and the moving mean climbs back. The probe step
        //   itself bypasses both the adaptive shrink AND the
        //   `ATLAS_DFLASH_ADAPTIVE_MAX` ceiling — its whole purpose is
        //   to measure ceiling, not to be capped by one. Margin-gate
        //   cuts still apply (those are per-step confidence signals
        //   independent of history).
        let adaptive_probe_interval = policy.adaptive_probe_interval;

        // Run top-2 if the margin gate is active, so we have the second-best
        // logit value per row. Cost: one extra kernel launch over the same
        // logits buffer plus a γ_eff×2×4 byte D2H. Tiny vs the layer chain.
        let margin_top2: Option<Vec<f32>> = if margin_gate > 0.0 {
            let top2 = checked_topk_layout(gamma_eff, self.gamma, argmax_vocab, 2)?;
            let used_bytes = top2.bytes;
            // Stream-ordered — same argument as the denoise-pass top-2 above.
            gpu.memset_async(self.scratch.topk_tokens_dev, 0, used_bytes, stream)?;
            gpu.memset_async(self.scratch.topk_logits_dev, 0, used_bytes, stream)?;
            ops::topk_bf16(
                gpu,
                self.kernels.topk,
                self.scratch.logits,
                self.scratch.topk_tokens_dev,
                self.scratch.topk_logits_dev,
                top2.num_rows,
                top2.vocab,
                top2.k as u32,
                stream,
            )?;
            gpu.synchronize(stream)?;
            let mut token_bytes = vec![0u8; used_bytes];
            let mut logits_bytes = vec![0u8; used_bytes];
            gpu.copy_d2h(self.scratch.topk_tokens_dev, &mut token_bytes)?;
            gpu.copy_d2h(self.scratch.topk_logits_dev, &mut logits_bytes)?;
            match Self::decode_topk_bytes(
                token_bytes,
                logits_bytes,
                gamma_eff,
                self.gamma,
                top2.k,
                argmax_vocab,
            ) {
                Ok((_, logits)) => Some(logits),
                Err(error) => {
                    tracing::warn!(
                        "DFlash adaptive top-2 output is invalid ({error:#}); degrading to bootstrap"
                    );
                    // `stream` was synchronized immediately above before top-2 D2H.
                    pinned_h2d_guard.armed = false;
                    return Ok(Vec::new());
                }
            }
        } else {
            None
        };

        // ── DSpark VanillaMarkov head (auto-on when checkpoint ships it) ──
        // Keep the full correction chain device-resident: each chosen u32
        // feeds the next position's W1 gather on this same stream. This must
        // happen before the only D2H below. A loaded head is fail-closed; a
        // kernel/launch failure cannot silently substitute plain DFlash.
        if self.markov.is_some() {
            anyhow::ensure!(
                !dstate
                    .active_sampling_request
                    .as_ref()
                    .is_some_and(|request| { request.request_sampler.temperature > 0.0 }),
                "probabilistic DFlash2 q would be invalidated by the DSpark Markov rewrite"
            );
            self.enqueue_markov_sequential(gpu, stream, last_token, gamma_eff)
                .context("DSpark device-resident Markov correction")?;
        }

        // ── Step 6: one final D2H of gamma token IDs ──
        // ATLAS_DFLASH_ASYNC_PROBE=1: split the propose wall into CPU enqueue
        // time (fn entry → here, all kernels launched) vs GPU drain time (the
        // synchronize below). The drain is the part an async second-stream
        // launch could overlap with the step's CPU tail. Measurement-only.
        let async_probe = crate::layers::dflash_async_probe();
        let probe_enqueue_us = if async_probe {
            t_total.elapsed().as_micros()
        } else {
            0
        };
        let host_bytes = gamma_eff
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| anyhow::anyhow!("DFlash draft output byte count overflow"))?;
        let sampled_request = dstate
            .active_sampling_request
            .as_ref()
            .filter(|request| request.request_sampler.temperature > 0.0)
            .cloned();
        let mut drafts = if let Some(request) = sampled_request {
            let sparse_bytes = gamma_eff
                .checked_mul(crate::dflash_sampling::DFLASH2_SELECTOR_TOP_K)
                .and_then(|elements| elements.checked_mul(size_of::<u32>()))
                .ok_or_else(|| anyhow::anyhow!("DFlash sparse-q byte count overflow"))?;
            // BOUNDED_SAMPLED_READBACK: collect all three bounded outputs
            // through the producer-stream APIs. The pair fails before q is
            // attempted; q fails before any typed frame is published.
            let DflashProposerState {
                draft_output_pinned,
                selector_candidates_pinned,
                selector_q_pinned,
                pending_sampled_proposal,
                ..
            } = dstate;
            let host_buf = &mut draft_output_pinned.as_mut_slice()[..host_bytes];
            let candidate_buf = &mut selector_candidates_pinned
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("DFlash sampled candidate staging is missing"))?
                .as_mut_slice()[..sparse_bytes];
            let q_buf = &mut selector_q_pinned
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("DFlash sampled q staging is missing"))?
                .as_mut_slice()[..sparse_bytes];
            let drafts =
                super::proposal_inputs::publish_after_success(pending_sampled_proposal, || {
                    gpu.copy_d2h_pair_on_stream(
                        self.scratch.draft_tokens_dev,
                        host_buf,
                        self.scratch.selector_q_candidates,
                        candidate_buf,
                        stream,
                    )?;
                    gpu.copy_d2h_on_stream(self.scratch.selector_q_probabilities, q_buf, stream)?;
                    let drafts = super::proposal_inputs::decode_token_ids(host_buf, gamma_eff)?;
                    let sampled = crate::dflash_sampling::decode_dflash2_device_proposal(
                        request,
                        last_token,
                        drafts.clone(),
                        candidate_buf,
                        q_buf,
                        argmax_vocab,
                    )?;
                    Ok((drafts, sampled))
                })?;
            // BOUNDED_SAMPLED_READBACK_END
            drafts
        } else {
            let host_buf = &mut dstate.draft_output_pinned.as_mut_slice()[..host_bytes];
            // When `p_cum` is active and draft probabilities were produced into
            // `batch_out_probs`, coalesce draft tokens + draft probs into 1
            // `copy_d2h_pair_on_stream` call so `propose()` avoids a second D2H sync.
            let pack_probs = dstate.pending_tree_payload.is_none()
                && dstate.pending_draft_probs_gamma > 0
                && gamma_eff <= dstate.pending_draft_probs_gamma
                && gamma_eff <= 64
                && !self.scratch.batch_out_probs.is_null()
                && super::dflash_pcum_enabled_for_target_env(self.target_is_moe);
            if pack_probs {
                let mut probs_buf = [0u8; 64 * size_of::<f32>()];
                let probs_bytes = gamma_eff * size_of::<f32>();
                gpu.copy_d2h_pair_on_stream(
                    self.scratch.draft_tokens_dev,
                    host_buf,
                    self.scratch.batch_out_probs,
                    &mut probs_buf[..probs_bytes],
                    stream,
                )?;
                dstate.last_draft_probs.clear();
                dstate.last_draft_probs.reserve(gamma_eff);
                for chunk in probs_buf[..probs_bytes].chunks_exact(size_of::<f32>()) {
                    dstate
                        .last_draft_probs
                        .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                dstate.pending_draft_probs_gamma = 0;
            } else {
                // One ordered D2H + trailing sync.  The old pair (`synchronize`
                // followed by `copy_d2h`) paid two host-blocking stream syncs and the
                // copy ran on the default stream.  This preserves the same completion
                // guarantee while coalescing it to one sync on the producer stream.
                gpu.copy_d2h_on_stream(self.scratch.draft_tokens_dev, host_buf, stream)?;
            }
            super::proposal_inputs::decode_token_ids(host_buf, gamma_eff)?
        };
        // Step 6's stream-ordered D2H completed, so all in-flight pinned H2D
        // transfers on `stream` have drained.
        pinned_h2d_guard.armed = false;
        if async_probe {
            tracing::info!(
                "ASYNC_PROBE propose: enqueue={probe_enqueue_us}μs gpu_total={}μs \
                 (eff_ctx={eff_ctx} n_attn={n_attn})",
                t_total.elapsed().as_micros(),
            );
        }
        // Multi-step denoise: committed rows freeze their commit-time
        // prediction. On the final pass those rows were embedded as REAL
        // tokens, so the row's own re-prediction is out-of-distribution
        // for the drafter (trained to predict at MASK rows only) — keep
        // the token that earned the confidence commit instead. No-op when
        // ATLAS_DFLASH_DENOISE_STEPS=1 (committed is all-None) or when
        // ATLAS_DFLASH_DENOISE_FREEZE=0 (take last-pass argmax verbatim).
        if denoise_freeze {
            for (d, c) in drafts.iter_mut().zip(committed.iter()) {
                if let Some(t) = c {
                    *d = *t;
                }
            }
        }

        // Apply adaptive gamma + margin gate. Two behaviors are available
        // via ATLAS_DFLASH_ADAPTIVE_MODE:
        //   "mask"     (default): replace excess drafts with mask_token_id
        //              and KEEP drafts.len() == gamma_eff. The verify kernel
        //              still runs K=γ+1, the CUDA graph stays hot, and the
        //              accept-prefix logic terminates at the first mask
        //              position. Use this when the K=γ verify cost is fixed
        //              and you only want to stop wasted ctx pollution.
        //   "truncate": physically shorten the drafts vector to `cutoff`
        //              entries. The dispatcher in mtp_step.rs then routes
        //              to a smaller K verify path (K=2/3/4 graphed) when
        //              drafts.len() < 4, or to eager K=cutoff+1 path. May
        //              save substantial verify cycles when accept is low,
        //              at the cost of dropping the K=γ CUDA graph cache.
        //
        // Replaced positions get `mask_token_id` which is guaranteed to
        // mismatch the target's argmax → accept-prefix terminates at the
        // first replacement (mask mode), or the truncation point (truncate
        // mode — no replacement needed, the position just disappears).
        // The throughput router must change the physical verify shape; masking
        // would retain the wide verifier cost and defeat its objective.
        let adaptive_mode = if policy.truncate_mode {
            "truncate"
        } else {
            "mask"
        };
        let climbdrop_mode = policy.climbdrop_mode;
        let truncate_mode = policy.truncate_mode || tps_router || climbdrop_mode;
        if adaptive_gamma || margin_gate > 0.0 || tps_router || climbdrop_mode {
            let mask = self.mask_token_id;
            // Bump the monotonic step counter once per adaptive-engaged call.
            // Used below to decide whether this step is a γ_max reprobe.
            dstate.propose_steps = dstate.propose_steps.wrapping_add(1);
            // A "probe step" forces cutoff = gamma_eff so the verifier sees
            // the true accept ceiling, bypassing both the history-mean shrink
            // and the ATLAS_DFLASH_ADAPTIVE_MAX ceiling. Margin-gate cuts
            // still apply later (per-step confidence, not history). We trigger
            // a probe every `adaptive_probe_interval` adaptive-engaged steps;
            // a value of 0 disables reprobing (legacy behavior). We also
            // require `accept_history_count >= 4` so the warmup phase (where
            // we're already running γ_max naturally) doesn't waste probes.
            let is_probe_step = adaptive_probe_interval > 0
                && dstate.accept_history_count >= 4
                && dstate.propose_steps.is_multiple_of(adaptive_probe_interval);
            // Compute the adaptive cutoff (= number of drafts to keep as-is).
            // gamma_eff is the post-cap drafter output size. cutoff <= gamma_eff.
            let mut cutoff = gamma_eff;
            if climbdrop_mode {
                // Climb/drop controller (llama.cpp PR #27210 algorithm): the
                // depth is the controller's current state; clamp to the
                // physical max. It climbs on consecutive full accepts and
                // falls back on weighted misses, so no probing is needed.
                let floor = policy.climbdrop_floor;
                let cap = policy.climbdrop_cap.unwrap_or(gamma_eff);
                if dstate.climbdrop_router.score() == 0 || dstate.climbdrop_router.decisions() == 0
                {
                    dstate
                        .climbdrop_router
                        .reset(cap.min(gamma_eff), floor.min(gamma_eff));
                }
                cutoff = dstate.climbdrop_router.choose().min(gamma_eff);
            } else if tps_router {
                let widths = super::throughput_router::parse_widths(
                    policy.router_widths.as_deref(),
                    gamma_eff,
                );
                let probe_interval = policy.router_probe_interval;
                cutoff = dstate
                    .throughput_router
                    .choose(&widths, gamma_eff, probe_interval);
            }
            if !is_probe_step && adaptive_gamma && dstate.accept_history_count >= 4 {
                let n = dstate.accept_history_count.min(dstate.accept_history.len());
                let sum: usize = dstate
                    .accept_history
                    .iter()
                    .take(n)
                    .map(|&v| v as usize)
                    .sum();
                let mean = sum / n; // integer mean is fine — we want conservative shrink
                let target = (mean + adaptive_slack).max(adaptive_min);
                cutoff = cutoff.min(target);
            }
            // Hard ceiling: K=γ=16 verify wall-time (2.8s/call on GB10) is
            // a worse deal than K=γ=4 even at high accept rates. When
            // ATLAS_DFLASH_ADAPTIVE_MAX > 0, never allow cutoff above it.
            // Independent of accept_history — applies even on the first
            // few calls (where history is still warming up). Skipped on
            // probe steps so we can actually measure γ_max behavior.
            if !is_probe_step && adaptive_max > 0 {
                cutoff = cutoff.min(adaptive_max);
            }
            // Margin gate: walk drafts in order, replace with mask at the first
            // low-confidence position (and all subsequent positions, since the
            // chain terminates at the first reject anyway — no point computing
            // attention for tokens we know will be discarded).
            let mut invalid_margin = false;
            if let Some(ref logits) = margin_top2 {
                for i in 0..gamma_eff {
                    let base = i * 2;
                    let top1 = logits[base];
                    let top2 = logits[base + 1];
                    let margin = match checked_topk_difference(top1, top2) {
                        Ok(margin) => margin,
                        Err(error) => {
                            tracing::warn!(
                                "DFlash adaptive top-2 row {i} is invalid ({error:#}); cutting before row"
                            );
                            cutoff = cutoff.min(i);
                            invalid_margin = true;
                            break;
                        }
                    };
                    if margin < margin_gate {
                        cutoff = cutoff.min(i);
                        break;
                    }
                }
            }
            if tps_router || climbdrop_mode {
                // Attribute the observation to the actual physical width after
                // all safety/confidence caps, not merely the router's request.
                dstate.throughput_last_width = cutoff.max(1);
            }
            if truncate_mode {
                // Shrink the vector so the scheduler downgrades the verify K.
                drafts.truncate(if invalid_margin {
                    cutoff
                } else {
                    cutoff.max(1)
                });
            } else {
                // Mask mode (default): replace positions [cutoff..gamma_eff].
                for i in cutoff..gamma_eff {
                    drafts[i] = mask;
                }
            }
            tracing::debug!(
                "DFlash adaptive: gamma_eff={} cutoff={} accept_count={} margin_gate={} mode={} probe={} step={}",
                gamma_eff,
                cutoff,
                dstate.accept_history_count,
                margin_gate,
                adaptive_mode,
                is_probe_step,
                dstate.propose_steps,
            );
        }
        // ATLAS_DFLASH_DEBUG_DUMP_FULL=1 (one-shot): log all γ drafts so
        // we can compare against the PyTorch reference run on the same
        // captured target_hidden. Static guard mirrors the input dump.
        static DRAFTS_DUMP_DONE: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if eff_ctx > 0
            && !DRAFTS_DUMP_DONE.load(std::sync::atomic::Ordering::Relaxed)
            && policy.debug_dump_full
            && position >= policy.dump_min_pos
        {
            tracing::info!(
                "DFLASH DUMP_FULL drafts (γ={}, γ_eff={}, last_token={}, position={}, eff_ctx={}): {:?}",
                self.gamma,
                gamma_eff,
                last_token,
                position,
                eff_ctx,
                drafts,
            );
            DRAFTS_DUMP_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if dump_all_layers {
            let path = "/tmp/atlas_final_drafts.bin";
            if !std::path::Path::new(path).exists() {
                let host_buf = &dstate.draft_output_pinned.as_mut_slice()[..host_bytes];
                let _ = std::fs::write(path, host_buf);
                tracing::info!("DFLASH DUMP_ALL: drafts {drafts:?} → {path}");
            }
        }
        if kprofile {
            gpu.synchronize(stream)?;
            let tail_us = t_tail.elapsed().as_micros();
            let pre_us = t_pre_layers.duration_since(t_total).as_micros()
                + t_layers.duration_since(t_pre_layers).as_micros();
            let pre_fc_us = t_pre_fc_done.duration_since(t_pre_layers).as_micros();
            let pre_pos_us = t_layers.duration_since(t_pre_fc_done).as_micros();
            let fc_is_nvfp4 = self.fc_nvfp4.is_some();
            let total_us = t_total.elapsed().as_micros();
            let agg = super::kprof_snapshot_layers();
            // Publish into the ATLAS_FULL_PROFILE table so a single
            // `ATLAS_FULL_PROFILE=1` run reports drafter kernels alongside the
            // verify path's, under `draft_*` labels.
            for (label, us) in agg.labelled() {
                crate::full_profile::record(label, (us as u64).saturating_mul(1_000));
            }
            // The residue is the diagnostic that matters: `layers_us` is the
            // whole noise pass (embed, 6 layers, final norm, lm_head, argmax)
            // while the accumulator covers only the layer-internal launches.
            // Anything large here is propose time that is still unattributed —
            // report it explicitly rather than leaving it to subtraction.
            let attributed_us = agg.attributed_us();
            let unattributed_us = layers_us.saturating_sub(attributed_us);
            crate::full_profile::record(
                "draft_unattributed",
                (unattributed_us as u64).saturating_mul(1_000),
            );
            tracing::info!(
                "DFLASH_KP residue: layers={}μs attributed={}μs unattributed={}μs \
                 (embed + final_norm + lm_head + argmax + any unlabelled launch)",
                layers_us,
                attributed_us,
                unattributed_us,
            );
            tracing::info!(
                "DFLASH_KP propose: total={:.2}ms pre+steps0-2={:.0}μs (fc={:.0}μs nvfp4_fc={}, pos={:.0}μs) layers={:.2}ms tail={:.0}μs \
                 n_attn={} eff_ctx={} γ_eff={} | per-kernel-sum-over-{}-layers (μs): \
                 input_norm={} q_proj={} kv_ctx_copy={} kv_ctx_new={} kv_noise={} \
                 qk_norm={} rope={} cache_write={} prefill_attn={} \
                 o_proj={} resid1={} post_norm={} gate_up={} silu_mul={} down_proj={} resid2={} \
                 conv_prepare={} conv_finish={}",
                total_us as f32 / 1000.0,
                pre_us,
                pre_fc_us,
                fc_is_nvfp4,
                pre_pos_us,
                layers_us as f32 / 1000.0,
                tail_us,
                n_attn,
                eff_ctx,
                gamma_eff,
                self.layers.len(),
                agg.input_norm_us,
                agg.q_proj_us,
                agg.kv_ctx_copy_us,
                agg.kv_ctx_new_us,
                agg.kv_noise_us,
                agg.qk_norm_us,
                agg.rope_us,
                agg.cache_write_us,
                agg.prefill_attn_us,
                agg.o_proj_us,
                agg.resid1_us,
                agg.post_norm_us,
                agg.gate_up_us,
                agg.silu_mul_us,
                agg.down_proj_us,
                agg.resid2_us,
                agg.conv_prepare_us,
                agg.conv_finish_us,
            );
        }
        Ok(drafts)
    }

    /// DDTree M4B v2: extract top-K tokens + logit values per drafter
    /// MASK row from the just-computed `self.scratch.logits` buffer.
    ///
    /// Must be called immediately after [`forward_block`] (before any
    /// subsequent call overwrites the logits scratch). Returns
    /// `(tokens, logits)` host vectors where each row of length `k`
    /// is sorted by logit descending.
    ///
    /// `gamma_eff` = the number of MASK rows that produced drafts (matches
    /// `drafts.len()` from forward_block). Invalid `k` fails without launch.
    #[allow(dead_code)]
    pub(super) fn extract_topk_from_logits(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        stream: u64,
        gamma_eff: usize,
        k: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let lm_vocab = self.target_vocab_size.min(self.vocab_size);
        let layout = checked_topk_layout(gamma_eff, self.gamma, lm_vocab, k)?;
        self.enqueue_topk_on_stream(gpu, stream, gamma_eff, layout.k)?;
        let mut tokens_bytes = vec![0u8; layout.bytes];
        let mut logits_bytes = vec![0u8; layout.bytes];
        // Drain top-K once, then copy both results through the pageable-safe
        // synchronous D2H pair boundary. This retains two copies plus one
        // producer-stream sync without requiring page-locked Vec storage.
        gpu.copy_d2h_pair_on_stream(
            self.scratch.topk_tokens_dev,
            &mut tokens_bytes,
            self.scratch.topk_logits_dev,
            &mut logits_bytes,
            stream,
        )?;
        Self::decode_topk_bytes(
            tokens_bytes,
            logits_bytes,
            gamma_eff,
            self.gamma,
            layout.k,
            lm_vocab,
        )
    }

    /// Enqueue the top-K extraction kernel on `stream` without syncing or D2H.
    /// Used by the async propose path to defer collection until `collect_topk_d2h`.
    /// Logits must already be populated at `self.scratch.logits` by `forward_block`.
    pub(super) fn enqueue_topk_on_stream(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        stream: u64,
        gamma_eff: usize,
        k: usize,
    ) -> Result<()> {
        use crate::layers::ops;
        let lm_vocab = self.target_vocab_size.min(self.vocab_size);
        let layout = checked_topk_layout(gamma_eff, self.gamma, lm_vocab, k)?;
        // Stream-ordered. This function's contract is already "enqueue on
        // `stream`, do not sync" — both callers sync before reading
        // (`extract_topk_from_logits` immediately below; the async propose
        // path via `collect_async_drafts_impl`'s stream sync), so a blocking
        // memset here was contradicting the function's own doc comment.
        gpu.memset_async(self.scratch.topk_tokens_dev, 0, layout.bytes, stream)?;
        gpu.memset_async(self.scratch.topk_logits_dev, 0, layout.bytes, stream)?;
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
        )
    }

    /// D2H the top-K results from `scratch.topk_tokens_dev`/`topk_logits_dev`.
    /// The stream holding the kernel must already be synced by the caller
    /// (e.g. via `collect_async_drafts_impl`'s stream sync).
    pub(super) fn collect_topk_d2h(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        gamma_eff: usize,
        k: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let lm_vocab = self.target_vocab_size.min(self.vocab_size);
        let layout = checked_topk_layout(gamma_eff, self.gamma, lm_vocab, k)?;
        let mut tokens_bytes = vec![0u8; layout.bytes];
        let mut logits_bytes = vec![0u8; layout.bytes];
        gpu.copy_d2h(self.scratch.topk_tokens_dev, &mut tokens_bytes)?;
        gpu.copy_d2h(self.scratch.topk_logits_dev, &mut logits_bytes)?;
        Self::decode_topk_bytes(
            tokens_bytes,
            logits_bytes,
            gamma_eff,
            self.gamma,
            layout.k,
            lm_vocab,
        )
    }

    fn decode_topk_bytes(
        tokens_bytes: Vec<u8>,
        logits_bytes: Vec<u8>,
        num_rows: usize,
        scratch_rows: usize,
        k: usize,
        vocab: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let layout = checked_topk_layout(num_rows, scratch_rows, vocab, k)?;
        let expected_bytes = layout.bytes;
        if tokens_bytes.len() != expected_bytes || logits_bytes.len() != expected_bytes {
            anyhow::bail!(
                "top-K result byte length mismatch: tokens={} logits={} expected={expected_bytes}",
                tokens_bytes.len(),
                logits_bytes.len(),
            );
        }
        let tokens: Vec<u32> = tokens_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let logits: Vec<f32> = logits_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for row in 0..num_rows {
            let start = row * layout.k;
            let row_tokens = &tokens[start..start + layout.k];
            let row_logits = &logits[start..start + layout.k];
            let marker_tokens = row_tokens
                .iter()
                .filter(|&&token| token == u32::MAX)
                .count();
            if marker_tokens > 0 {
                let exact_markers = row_tokens
                    .iter()
                    .zip(row_logits)
                    .filter(|&(token, score)| *token == u32::MAX && score.is_nan())
                    .count();
                if marker_tokens == layout.k && exact_markers == layout.k {
                    anyhow::bail!("top-K row {row} contains invalid marker");
                }
                anyhow::bail!("top-K row {row} contains partial marker");
            }
            if row_logits.iter().any(|score| score.is_nan()) {
                anyhow::bail!("top-K row {row} contains NaN");
            }
            if !row_logits.iter().any(|score| *score > f32::NEG_INFINITY) {
                anyhow::bail!("top-K row {row} has no usable score");
            }

            let mut seen = std::collections::HashSet::with_capacity(layout.k);
            for column in 0..layout.k {
                let token = row_tokens[column];
                let score = row_logits[column];
                if token >= layout.vocab {
                    anyhow::bail!(
                        "top-K row {row} token {token} is out of range for vocab {}",
                        layout.vocab
                    );
                }
                if !seen.insert(token) {
                    anyhow::bail!("top-K row {row} contains duplicate token {token}");
                }
                if column == 0 {
                    continue;
                }
                let previous_score = row_logits[column - 1];
                let previous_token = row_tokens[column - 1];
                if previous_score < score {
                    anyhow::bail!("top-K row {row} violates score order");
                }
                if previous_score == score && previous_token > token {
                    anyhow::bail!("top-K row {row} violates token order for equal scores");
                }
            }
        }

        Ok((tokens, logits))
    }
}

/// Available transposed W4A16 GEMM kernel handles for `fc_nvfp4_t` dispatch.
#[derive(Debug, Clone, Copy)]
struct FcTransposeKernelCaps {
    has_m32_n64: bool,
    has_m16_n128: bool,
}

/// Returns `true` when an NVFP4 `fc` projection span on `<= 64`-SM GPUs (GB10)
/// should route through the transposed `fc_nvfp4_t` fallback via
/// `self.draft_gemm_t` (`w4a16_gemm_t_m32_n64_splitk` / `m32_n64` / `m16`).
///
/// Typical DFlash2 propose spans have `M = 1..=8` (`<= 32`); wider GPUs
/// (`> 64` SMs, e.g. SM120) route through `try_coalesced_nvfp4_sm120_small_m`
/// or `nvfp4_project_coalesced_or_gemm` before reaching this fallback, and
/// larger spans (`> 32`) use `w4a16_gemm_pipe` or standard `w4a16_gemm`.
fn should_use_fc_nvfp4_t(
    span_slots: usize,
    hidden_size: usize,
    target_hidden_dim: usize,
    caps: FcTransposeKernelCaps,
) -> bool {
    const FC_NVFP4_T_MAX_M: usize = 32;
    if span_slots == 0
        || span_slots > FC_NVFP4_T_MAX_M
        || hidden_size == 0
        || target_hidden_dim == 0
        || !target_hidden_dim.is_multiple_of(32)
    {
        return false;
    }
    (caps.has_m32_n64 && hidden_size.is_multiple_of(64))
        || (caps.has_m16_n128 && hidden_size.is_multiple_of(128))
}

#[cfg(test)]
#[path = "topk_contract_tests.rs"]
mod topk_contract_tests;

#[cfg(test)]
mod fc_nvfp4_t_tests {
    use super::{FcTransposeKernelCaps, should_use_fc_nvfp4_t};

    #[test]
    fn fc_nvfp4_t_dispatch_bounds_match_small_m_contract() {
        let caps = FcTransposeKernelCaps {
            has_m32_n64: true,
            has_m16_n128: true,
        };
        assert!(!should_use_fc_nvfp4_t(0, 5120, 25600, caps));
        assert!(should_use_fc_nvfp4_t(1, 5120, 25600, caps));
        assert!(should_use_fc_nvfp4_t(7, 5120, 25600, caps));
        assert!(should_use_fc_nvfp4_t(32, 5120, 25600, caps));
        assert!(!should_use_fc_nvfp4_t(33, 5120, 25600, caps));
        assert!(!should_use_fc_nvfp4_t(8, 5120, 25616, caps));
        assert!(!should_use_fc_nvfp4_t(
            8,
            5120,
            25600,
            FcTransposeKernelCaps {
                has_m32_n64: false,
                has_m16_n128: false,
            }
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fc_nvfp4_t_splitk_and_m32_match_w4a16_gemm_on_gpu() -> anyhow::Result<()> {
        use crate::layers::ops;
        use crate::weight_map::{DenseWeight, quantize_to_nvfp4};
        use half::bf16;
        use spark_runtime::gpu::GpuBackend;

        let Some(ptx) = atlas_kernels::ptx_for_model("qwen3.8-27b")
            .or_else(|| atlas_kernels::all_ptx_sets().into_iter().next())
        else {
            return Ok(());
        };
        let Ok(gpu) = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &ptx.modules) else {
            return Ok(());
        };
        let stream = gpu.default_stream();

        let absmax_k = crate::layers::try_kernel(&gpu, "quantize_nvfp4", "nvfp4_global_absmax");
        let quantize_k =
            crate::layers::try_kernel(&gpu, "quantize_nvfp4", "quantize_bf16_to_nvfp4");
        let gemm_k = crate::layers::try_kernel(&gpu, "w4a16", "w4a16_gemm");
        let gemm_t_m32_k = crate::layers::try_kernel(&gpu, "w4a16", "w4a16_gemm_t_m32_n64");
        let gemm_t_splitk_k =
            crate::layers::try_kernel(&gpu, "w4a16", "w4a16_gemm_t_m32_n64_splitk");
        let reduce_k = crate::layers::try_kernel(&gpu, "w4a16", "reduce_splitk_f32_to_bf16");
        if absmax_k.0 == 0
            || quantize_k.0 == 0
            || gemm_k.0 == 0
            || gemm_t_m32_k.0 == 0
            || gemm_t_splitk_k.0 == 0
            || reduce_k.0 == 0
        {
            return Ok(());
        }

        // Exact Qwen3.8-27B DFlash2 `fc` shape: M=7, N=5120, K=25600.
        let m = 7usize;
        let n = 5120usize;
        let k = 25600usize;
        let k_splits = 8u32;

        // Deterministic E4M3-exact activations [M, K] and weights [N, K] in BF16:
        // `w4a16_gemm_t_m32_n64(_splitk)` uses FP8 E4M3 tensor-core MMA while
        // `w4a16_gemm` uses BF16 MMA. Choosing E4M3-exact inputs (`{-0.5, -0.25, 0.25, 0.5}`)
        // eliminates E4M3 input rounding noise so any transpose/scale/split-K bug
        // surfaces immediately!
        const E4M3_VALS: [f32; 4] = [-0.5, -0.25, 0.25, 0.5];
        let mut x_bytes = Vec::with_capacity(m * k * 2);
        for row in 0..m {
            for col in 0..k {
                let idx = row
                    .wrapping_mul(1315423911)
                    .wrapping_add(col.wrapping_mul(2654435761));
                let v = E4M3_VALS[(idx >> 16) & 3];
                x_bytes.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
            }
        }
        let mut w_bytes = Vec::with_capacity(n * k * 2);
        for row in 0..n {
            for col in 0..k {
                let idx = row
                    .wrapping_mul(2654435761)
                    .wrapping_add(col.wrapping_mul(40503));
                let v = E4M3_VALS[(idx >> 12) & 3] * 0.5;
                w_bytes.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
            }
        }

        let x_dev = gpu.alloc(x_bytes.len())?;
        let w_dev = gpu.alloc(w_bytes.len())?;
        gpu.copy_h2d(&x_bytes, x_dev)?;
        gpu.copy_h2d(&w_bytes, w_dev)?;

        let fc_q = quantize_to_nvfp4(
            &DenseWeight { weight: w_dev },
            n,
            k,
            &gpu,
            absmax_k,
            quantize_k,
            stream,
        )?;
        let fc_t = fc_q.transpose_for_gemm(&gpu, n, k)?;

        let out_bytes = m * n * 2;
        let out_ref = gpu.alloc(out_bytes)?;
        let out_m32 = gpu.alloc(out_bytes)?;
        let out_splitk = gpu.alloc(out_bytes)?;
        let ws = gpu.alloc((k_splits as usize) * m * n * 4)?;

        ops::w4a16_gemm(
            &gpu, gemm_k, x_dev, &fc_q, out_ref, m as u32, n as u32, k as u32, stream,
        )?;
        ops::w4a16_gemm_n64_m32(
            &gpu,
            gemm_t_m32_k,
            x_dev,
            &fc_t,
            out_m32,
            m as u32,
            n as u32,
            k as u32,
            stream,
        )?;
        ops::w4a16_gemm_n64_m32_splitk(
            &gpu,
            gemm_t_splitk_k,
            reduce_k,
            x_dev,
            &fc_t,
            out_splitk,
            ws,
            m as u32,
            n as u32,
            k as u32,
            n as u32,
            k_splits,
            stream,
        )?;
        gpu.synchronize(stream)?;

        let read_bf16 = |ptr| -> anyhow::Result<Vec<f32>> {
            let mut raw = vec![0u8; out_bytes];
            gpu.copy_d2h(ptr, &mut raw)?;
            Ok(raw
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        };
        let ref_v = read_bf16(out_ref)?;
        let m32_v = read_bf16(out_m32)?;
        let splitk_v = read_bf16(out_splitk)?;

        let mut max_err_m32 = 0.0f32;
        let mut max_err_splitk = 0.0f32;
        let mut max_err_m32_vs_splitk = 0.0f32;
        let mut rms_err_m32 = 0.0f64;
        let mut rms_ref = 0.0f64;
        let mut max_abs_ref = 0.0f32;
        for i in 0..ref_v.len() {
            assert!(m32_v[i].is_finite() && splitk_v[i].is_finite());
            max_abs_ref = max_abs_ref.max(ref_v[i].abs());
            let em = (ref_v[i] - m32_v[i]).abs();
            let es = (ref_v[i] - splitk_v[i]).abs();
            max_err_m32 = max_err_m32.max(em);
            max_err_splitk = max_err_splitk.max(es);
            max_err_m32_vs_splitk = max_err_m32_vs_splitk.max((m32_v[i] - splitk_v[i]).abs());
            rms_err_m32 += f64::from(em) * f64::from(em);
            rms_ref += f64::from(ref_v[i]) * f64::from(ref_v[i]);
        }
        let rel_l2 = (rms_err_m32 / rms_ref.max(1e-12)).sqrt();
        // With E4M3-exact inputs & weights, FP8 MMA (`w4a16_gemm_t_m32_n64(_splitk)`)
        // must closely match BF16 MMA (`w4a16_gemm`), and M32 vs SplitK(8) must match
        // within 1-2 BF16 ULPs across K=25600.
        let tol = (max_abs_ref * 0.02).max(0.05);
        assert!(
            max_err_m32_vs_splitk <= 0.02,
            "fc_nvfp4_t M32 vs SplitK(8) diverged: max_err={max_err_m32_vs_splitk}"
        );
        assert!(
            max_err_m32 <= tol && rel_l2 <= 0.02,
            "fc_nvfp4_t M32 parity failed: max_err={max_err_m32}, tol={tol}, rel_l2={rel_l2}, max_abs_ref={max_abs_ref}"
        );
        assert!(
            max_err_splitk <= tol,
            "fc_nvfp4_t SplitK(8) parity failed: max_err={max_err_splitk}, tol={tol}, max_abs_ref={max_abs_ref}"
        );
        eprintln!(
            "PARITY OK fc_nvfp4_t (M={m}, N={n}, K={k}): max_err_m32={max_err_m32:.6}, max_err_splitk8={max_err_splitk:.6}, m32_vs_splitk={max_err_m32_vs_splitk:.6}, rel_l2={rel_l2:.6}, max_abs_ref={max_abs_ref:.4}"
        );

        gpu.free(ws)?;
        gpu.free(out_splitk)?;
        gpu.free(out_m32)?;
        gpu.free(out_ref)?;
        gpu.free(fc_t.weight)?;
        gpu.free(fc_t.weight_scale)?;
        gpu.free(fc_q.weight)?;
        gpu.free(fc_q.weight_scale)?;
        gpu.free(w_dev)?;
        gpu.free(x_dev)?;
        Ok(())
    }
}
