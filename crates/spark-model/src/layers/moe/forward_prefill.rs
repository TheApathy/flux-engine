// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward_prefill.

use super::*;

/// Environment override for dual-stream shared+routed MoE prefill overlap.
/// When unset, defaults to enabled on high-SM discrete GPUs
/// (`active_sm_count() >= 64` and `active_device_is_integrated() == Some(false)`)
/// and disabled on integrated UMA parts like 48-SM GB10 (where dual-stream
/// contention worsens LPDDR5X bandwidth). Explicit `"0"`/`"false"`/`"off"`
/// disables it everywhere; `"1"`/`"true"`/`"on"` forces it on.
const PREFILL_SHARED_OVERLAP_ENV: &str = "ATLAS_MOE_PREFILL_OVERLAP";
const PREFILL_SHARED_OVERLAP_MIN_SMS: u32 = 64;

#[must_use]
fn resolve_prefill_shared_overlap(
    raw: Option<&str>,
    sm_count: u32,
    is_integrated: Option<bool>,
) -> bool {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off") => {
            false
        }
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on") => {
            true
        }
        _ => sm_count >= PREFILL_SHARED_OVERLAP_MIN_SMS && is_integrated == Some(false),
    }
}

#[must_use]
fn prefill_shared_overlap_enabled() -> bool {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var(PREFILL_SHARED_OVERLAP_ENV).ok());
    resolve_prefill_shared_overlap(
        raw.as_deref(),
        atlas_core::device::active_sm_count(),
        atlas_core::device::active_device_is_integrated(),
    )
}

/// Ensures an overlapped `run_shared_expert_prefill` launch on `prefill_stream`
/// is joined back onto `stream` via `event_b` even if an intermediate routed
/// step fails early before step 8.
struct PrefillOverlapJoinGuard<'a> {
    gpu: &'a dyn GpuBackend,
    stream: u64,
    prefill_stream: u64,
    event_b: u64,
    armed: bool,
}

impl<'a> PrefillOverlapJoinGuard<'a> {
    fn new(gpu: &'a dyn GpuBackend, stream: u64, prefill_stream: u64, event_b: u64) -> Self {
        Self {
            gpu,
            stream,
            prefill_stream,
            event_b,
            armed: false,
        }
    }

    fn arm(&mut self) {
        self.armed = true;
    }

    fn join(&mut self) -> Result<()> {
        if self.armed {
            self.armed = false;
            self.gpu.stream_wait_event(self.stream, self.event_b)?;
        }
        Ok(())
    }
}

impl Drop for PrefillOverlapJoinGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.armed = false;
            if let Err(err) = self.gpu.stream_wait_event(self.stream, self.event_b) {
                tracing::error!(
                    "PrefillOverlapJoinGuard::drop stream_wait_event failed (falling back to synchronize): {err:#}"
                );
                if let Err(sync_err) = self.gpu.synchronize(self.prefill_stream) {
                    tracing::error!(
                        "PrefillOverlapJoinGuard::drop fallback synchronize(prefill_stream) failed: {sync_err:#}"
                    );
                }
            }
        }
    }
}

pub(super) fn router_bf16_tk64_selected(rows: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ON.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_ROUTER_BF16_TK64")
            .ok()
            .as_deref()
            == Some("1")
    });
    router_bf16_tk64_eligible(enabled, rows)
}

fn router_bf16_tk64_eligible(enabled: bool, rows: usize) -> bool {
    enabled && rows >= 32
}

fn validate_router_bf16_tk64(
    rows: usize,
    hidden: usize,
    experts: usize,
    bf16: bool,
    kernel: KernelHandle,
) -> Result<()> {
    anyhow::ensure!(
        bf16 && hidden == 2560
            && experts == 512
            && (32..=1_048_560).contains(&rows)
            && kernel.0 != 0,
        "ATLAS_QWEN4_ROUTER_BF16_TK64=1 requires a resident BF16 router, H2560/E512, 32<=M<=1048560 (grid.y<=65535) and loaded dense_gemm_bf16_tk64 (M={rows}, H={hidden}, E={experts}, bf16={bf16}, kernel_loaded={})",
        kernel.0 != 0
    );
    Ok(())
}

impl MoeLayer {
    /// N-token prefill via grouped GEMM: sort-by-expert → tensor-core GEMM per expert.
    ///
    /// Each expert's weight matrix is loaded once (not per-token), cutting LPDDR5X
    /// reads from ~6 GB (GEMV) to ~150 MB (grouped GEMM) at N=1024.
    ///
    /// Pipeline: gate → topK → sort → grouped gate/up GEMM → SiLU → grouped down GEMM
    ///           → unpermute + weighted reduce → shared expert blend.
    /// Shared expert uses standard w4a16_gemm (single-expert, M=N_tokens).
    #[allow(unused_assignments)]
    pub fn forward_prefill(
        &self,
        input: DevicePtr, // [num_tokens, H] BF16 — normed MoE input
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Small-batch optimization (M <= 16):
        // Bypass the 8-stage padding and launch overhead of grouped GEMM (~260 µs floor)
        // for single-token decode, speculative verify (2 <= M <= 16), and 1-token tail crops.
        let frankenstein_decode = std::env::var("ATLAS_FRANKENSTEIN_DECODE_VIA_PREFILL")
            .ok()
            .as_deref()
            == Some("1");
        let prefill_native_or_r2 =
            crate::layers::qwen4_prefill_native::router_bf16(self.weights.gate.weight)
                || crate::layers::qwen4_prefill_native::shared::lookup(
                    self.weights.shared_expert_gate.weight,
                )
                .is_some()
                || crate::model::qwen4_prefill_engagement::slot_r2_active_for(num_tokens);
        let ep_multi_gpu =
            ctx.config.ep_world_size > 1 && ctx.comm.as_ref().is_some_and(|c| c.world_size() > 1);
        match trait_decode::decide_small_batch_prefill_with_policy(
            num_tokens,
            frankenstein_decode,
            prefill_native_or_r2,
            ep_multi_gpu,
            self.t_lanes_rows_route(),
            self.gate_nvfp4.is_some(),
            crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
        ) {
            trait_decode::SmallBatchPrefillAction::ScalarDecodeM1 => {
                self.forward(input, ctx, stream)?;
                return Ok(());
            }
            trait_decode::SmallBatchPrefillAction::ExactRows => {
                return self.forward_qwen4_exact(input, num_tokens, ctx, stream);
            }
            trait_decode::SmallBatchPrefillAction::GroupedPrefill => {}
        }

        let native_router_bf16 =
            crate::layers::qwen4_prefill_native::router_bf16(self.weights.gate.weight);
        let router_tk64 = router_bf16_tk64_selected(num_tokens);
        if router_tk64 {
            let bf16_router_resident = self.fp8_gate_weight_ptrs.is_none()
                && !self.weights.gate.weight.is_null()
                && (native_router_bf16 || (self.gate_fp8.is_none() && self.gate_nvfp4.is_none()));
            validate_router_bf16_tk64(
                num_tokens,
                ctx.config.hidden_size,
                ctx.config.num_experts,
                bf16_router_resident,
                self.router_bf16_tk64,
            )?;
        }

        // FP8 experts: use grouped GEMM for long prefills (>64 tokens),
        // fall back to per-token fused GEMV for short prefills where
        // the GEMM launch overhead exceeds the bandwidth savings.
        if self.fp8_gate_weight_ptrs.is_some() {
            if self.moe_fp8_grouped_gemm_k.0 != 0 && num_tokens > 64 {
                return self.forward_prefill_fp8(input, num_tokens, ctx, stream);
            }
            return self.forward_batched(input, num_tokens, ctx, stream);
        }

        // Lazy down_proj transpose: synchronous on the compute stream.
        // Overlapping this transpose on a secondary stream regressed cold TTFT
        // on GB10 due to SM contention, so synchronous execution is used.
        let _t_xpose = if ctx.profile && self.down_t_scratch_packed.is_some() {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        self.transpose_down_into_scratch(ctx, stream)?;
        if let Some(t0) = _t_xpose {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "  MoE prefill [lazy_transpose_down] N={}: {}µs",
                num_tokens,
                t0.elapsed().as_micros(),
            );
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = num_tokens as u32;
        let total_expanded = n * top_k;

        // Profile helper macro
        #[allow(unused_macros)]
        macro_rules! prof {
            ($label:expr) => {
                if ctx.profile {
                    ctx.gpu.synchronize(stream)?;
                    let _t = std::time::Instant::now();
                    tracing::info!("  MoE prefill [{}] N={}", $label, num_tokens);
                }
            };
        }
        #[allow(unused_assignments)]
        let mut t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        macro_rules! prof_step {
            ($label:expr) => {
                spark_runtime::cublaslt::stage_timer::mark($label, stream);
                if let Some(t) = t0.take() {
                    ctx.gpu.synchronize(stream)?;
                    let elapsed = t.elapsed().as_micros();
                    tracing::info!("  MoE prefill [{}] N={}: {}µs", $label, num_tokens, elapsed);
                    t0 = Some(std::time::Instant::now());
                }
            };
        }

        // ── Shared expert on secondary stream (overlaps with routed path) ──
        // Shared expert only reads `input` and writes to separate buffers
        // (ssm_deinterleaved, ssm_qkvz, attn_output) — no data conflict
        // with the routed expert path.  In profile mode, run sequentially
        // on the default stream for accurate per-step timing.
        //
        // Skip entirely when shared_inter == 0 (models without a shared expert,
        // e.g. Qwen3-VL-30B which has no shared_expert_intermediate_size).
        // Launching kernels with N=0 produces CUDA_ERROR_INVALID_VALUE (grid.x=0).
        let has_shared = shared_inter > 0;
        let overlap_eligible = has_shared
            && !ctx.profile
            && !ctx.graph_capture
            && self.pre_expert_norm.is_none()
            && self.prefill_stream != 0
            && self.prefill_stream != stream
            && self.event_a != 0
            && self.event_b != 0
            && prefill_shared_overlap_enabled();
        let mut use_overlap = false;
        let mut overlap_join_guard =
            PrefillOverlapJoinGuard::new(ctx.gpu, stream, self.prefill_stream, self.event_b);

        // A shared W4A8 dispatch below may leave this layer's MXFP8 x (and router FP8 act) for this layer only.
        super::qwen4_compact_ops::forget_xq();
        crate::layers::qwen4_fast_proj::clear_prequant_fp8_act();
        let qwen4_compact_active =
            super::qwen4_prefill_compact::request_active(self.qwen4_compact.is_some());
        // RAII guard for Wave-17 full shared-W4A8 overlap: arms `AfterRouted` deferral across
        // `run_routed_grouped_gemm` and guarantees any side-stream `shared_w4a8` event is joined
        // back onto `stream` before `shared_out` (`attn_output`) is consumed or on any early
        // error / unwind exit.
        let shared_overlap_guard = super::qwen4_shared_v7::SharedOverlapJoinGuard::enter(ctx.gpu);
        let mut deferred_shared_w4a8 = None;
        if has_shared {
            if let Some(native) = crate::layers::qwen4_prefill_native::shared::lookup(
                self.weights.shared_expert_gate.weight,
            ) {
                // ATLAS_QWEN4_PREFILL_NATIVE=1: exact-MXFP8 BF16 shared expert (decode keeps NVFP4).
                use crate::layers::qwen4_prefill_native::shared as native_shared;
                native_shared::refuse_conflict(
                    crate::layers::qwen4_fakequant::current()?.moe_shared_v7,
                )?;
                let scratch = native_shared::Scratch {
                    gate_out: ctx.buffers.ssm_deinterleaved(),
                    up_out: ctx.buffers.ssm_qkvz(),
                    down_out: ctx.buffers.attn_output(),
                };
                native_shared::run(
                    ctx.gpu,
                    &native,
                    self.moe_act_mul,
                    input,
                    num_tokens,
                    scratch,
                    stream,
                )?;
            } else if qwen4_compact_active
                && (super::qwen4_shared_v7::selected()
                    || crate::layers::qwen4_fakequant::current()?.moe_shared_v7)
            {
                // Default-off: the shared expert through the V7 routed kernels.
                deferred_shared_w4a8 =
                    self.run_qwen4_shared_v7_prefill(input, num_tokens, ctx, stream)?;
            } else {
                use_overlap = overlap_eligible;
                let aux = if use_overlap {
                    self.prefill_stream
                } else {
                    stream
                };
                self.run_shared_expert_prefill(
                    input,
                    n,
                    h,
                    shared_inter,
                    aux,
                    stream,
                    use_overlap,
                    ctx,
                )?;
                if use_overlap {
                    overlap_join_guard.arm();
                }
            }
        }
        prof_step!("shared_expert");

        // ── Routed expert path on default stream ──

        // Gemma-4 router pre-norm (no-op for other models).
        let router_in = self.router_input(input, n, h, ctx, stream)?;
        super::dump::dump_gate_input(ctx.gpu, stream, router_in, n, h)?;
        // 1. Gate GEMM: [N, H] × [H, num_experts] → [N, num_experts]
        let gate_logits = ctx.buffers.gate_logits();
        if router_tk64 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.router_bf16_tk64)
                .grid([num_experts.div_ceil(16), n.div_ceil(16), 1])
                .block([16, 16, 1])
                .arg_ptr(router_in)
                .arg_ptr(self.weights.gate.weight)
                .arg_ptr(gate_logits)
                .arg_u32(n)
                .arg_u32(num_experts)
                .arg_u32(h)
                .launch(stream)?;
            if native_router_bf16 {
                crate::layers::qwen4_prefill_native::note_call(
                    crate::layers::qwen4_prefill_native::Module::Router,
                    "bf16_dense_gemm_tk64",
                    num_tokens,
                    num_experts as usize,
                    h as usize,
                    1.0,
                );
            }
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::info!(
                    "ENGAGED ATLAS_QWEN4_ROUTER_BF16_TK64: BF16 router prefill H2560/E512 M>=32"
                );
            });
        } else if native_router_bf16 {
            // ATLAS_QWEN4_PREFILL_NATIVE=1: the shipped BF16 router (decode keeps NVFP4).
            anyhow::ensure!(
                !self.weights.gate.weight.is_null(),
                "ATLAS_QWEN4_PREFILL_NATIVE BF16 router was released"
            );
            crate::layers::qwen4_prefill_native::router_dense_gemm(
                ctx.gpu,
                self.dense_gemm,
                router_in,
                &self.weights.gate,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        } else if num_tokens >= 1024
            && crate::layers::qwen4_fast_proj::fp8_tensor_scales()
            && let Some(ref nvfp4) = self.gate_nvfp4
        {
            let _router_pad_guard = crate::layers::qwen4_fast_proj::PadRowsGuard::enter(
                ctx.buffers
                    .sizes()
                    .gate_logits
                    .checked_div(num_experts as usize * 2)
                    .unwrap_or(0),
            );
            crate::layers::qwen4_fast_proj::fp8_gemm_ex(
                ctx.gpu,
                router_in,
                nvfp4,
                gate_logits,
                num_tokens,
                num_experts as usize,
                h as usize,
                num_experts as usize,
                Some((0, 0)),
                stream,
            )?;
        } else if let Some(fp8) = self.gate_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                router_in,
                fp8,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        } else if let Some(ref nvfp4) = self.gate_nvfp4 {
            if !super::qwen4_router_pipe::try_launch_router_gemm(
                ctx.gpu,
                router_in,
                nvfp4,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )? {
                ops::w4a16_gemm(
                    ctx.gpu,
                    self.w4a16_gemm,
                    router_in,
                    nvfp4,
                    gate_logits,
                    n,
                    num_experts,
                    h,
                    stream,
                )?;
            }
        } else {
            anyhow::ensure!(
                !self.weights.gate.weight.is_null(),
                "MoE BF16 router was released (ATLAS_QWEN4_DENSE_DEDUP); this route needs the NVFP4 router"
            );
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm,
                router_in,
                &self.weights.gate,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        }
        crate::layers::qwen4_fast_proj::clear_prequant_fp8_act();
        super::dump::dump_gate_logits(ctx.gpu, stream, gate_logits, n, num_experts)?;
        prof_step!("gate_gemm");
        if let Some(deferred) = deferred_shared_w4a8.take() {
            self.fork_qwen4_shared_w4a8_overlap(deferred, ctx.gpu, stream)?;
        }

        // 2. Batched topK dispatch. DeepSeek-V3 / MiniMax-M2 use sigmoid
        //    + correction bias (detected via `correction_bias_dev`);
        //    every other model takes the softmax path (no behavior
        //    change — this is additive).
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(total_expanded as usize * 4);
        if let Some(bias) = self.correction_bias_dev {
            ops::moe_topk_sigmoid_batched(
                ctx.gpu,
                self.moe_topk_sigmoid_batched_k,
                gate_logits,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                1.0,
                n,
                stream,
            )?;
        } else if !super::qwen4_topk_warp::try_launch(
            ctx.gpu,
            gate_logits,
            indices_dev,
            weights_dev,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            n,
            stream,
        )? {
            // ATLAS_QWEN4_PREFILL_MOE_TOPK_WARP (default off) takes the branch above with the
            // bit-identical warp-per-token twin; this is the shipping block-per-token kernel.
            ops::moe_topk_softmax_batched(
                ctx.gpu,
                self.moe_topk_batched,
                gate_logits,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                n,
                stream,
            )?;
        }
        super::dump::dump_expert_ids(ctx.gpu, stream, indices_dev, weights_dev, n, top_k)?;
        prof_step!("topk");

        let is_ep_prefill_early =
            ctx.config.ep_world_size > 1 && ctx.comm.is_some_and(|c| c.world_size() > 1);
        if is_ep_prefill_early {
            let (ep_start, ep_end) = ctx.config.local_expert_range();
            super::qwen4_compact_ops::launch_ep_mask_remote_weights(
                ctx.gpu,
                indices_dev,
                weights_dev,
                ep_start,
                ep_end,
                total_expanded,
                stream,
            )?;
        }

        // 3. Sort tokens by expert → L2-optimized ordering.
        let te = total_expanded as usize;
        let ne = num_experts as usize;
        let sorted_token_ids = gate_logits;
        let sorted_expert_ids = gate_logits.offset(te * 4);
        let expert_offsets = gate_logits.offset(te * 4 * 2);
        let base_perm_off = te * 4 * 2 + (ne + 1) * 4;
        let gatepre_perm_off = base_perm_off.max(num_tokens * 640);
        let perm_off = if crate::layers::qwen4_hyper::prefill_inject::selection()?.gatepre
            && gatepre_perm_off + te * 4 <= num_tokens * ne * 2
        {
            gatepre_perm_off
        } else {
            base_perm_off
        };
        let token_to_perm = gate_logits.offset(perm_off);
        if is_ep_prefill_early
            || super::dump::enabled()
            || !super::qwen4_router_pipe::fused_sort_plan_active(
                qwen4_compact_active,
                self.qwen4_compact,
                num_tokens,
            )?
        {
            ops::moe_sort_by_expert(
                ctx.gpu,
                self.moe_sort_by_expert,
                indices_dev,
                sorted_token_ids,
                sorted_expert_ids,
                expert_offsets,
                token_to_perm,
                total_expanded,
                num_experts,
                top_k,
                stream,
            )?;
        }
        prof_step!("sort");

        // 3.5. Pre-expert norm: norm the input for expert dispatch (Gemma-4 26B).
        // Router already used the raw input for routing; now norm for experts.
        // IMPORTANT: write to scratch (ssm_deinterleaved), NOT in-place — `input` is
        // the residual and must be preserved for the subsequent residual add.
        let expert_input = if let Some(ref norm_w) = self.pre_expert_norm {
            let normed_buf = ctx.buffers.ssm_deinterleaved();
            let n_tokens = num_tokens as u32;
            let eps = ctx.config.rms_norm_eps as f32;
            ops::rms_norm(
                ctx.gpu,
                self.pre_expert_norm_k,
                input,
                norm_w,
                normed_buf,
                n_tokens,
                h,
                eps,
                stream,
            )?;
            normed_buf
        } else {
            input
        };
        prof_step!("pre_expert_norm");

        // 4-6. Routed grouped-GEMM phase (grid sizing → grouped gate+up
        // GEMM → SiLU → grouped down GEMM). Hoisted to forward_prefill_routed.rs
        // to keep this file under the 500 LoC cap; behavior identical.
        let routed_result = self.run_routed_grouped_gemm(
            expert_input,
            expert_offsets,
            sorted_token_ids,
            n,
            h,
            inter,
            num_experts,
            top_k,
            num_tokens,
            ne,
            &mut t0,
            ctx,
            stream,
        );
        // Join any deferred side-stream `shared_w4a8` launch back onto `stream` before
        // `shared_out` (`attn_output`) is consumed, deferred into `PendingInject::combine`,
        // or dropped on a routed error.
        let join_result = shared_overlap_guard.join();
        routed_result?;
        join_result?;
        let expert_down_out = ctx.buffers.expert_down_out();
        // ATLAS_QWEN4_PREFILL_MOE_SLOT_MXFP8: this dispatch may have written
        // payload + scales, optionally followed by one packed residual-u32 per
        // MX32 group. Carry that explicit ABI through both combine paths.
        let slot_storage = super::qwen4_compact_ops::take_slot_storage();
        let slot_layout =
            super::qwen4_prefill_compact::slot_layout(n as usize * top_k as usize, slot_storage)?;
        let slot_scales = slot_layout
            .scales_offset
            .map_or(spark_runtime::gpu::DevicePtr::NULL, |offset| {
                expert_down_out.offset(offset)
            });
        let slot_corrections = slot_layout
            .corrections_offset
            .map_or(spark_runtime::gpu::DevicePtr::NULL, |offset| {
                expert_down_out.offset(offset)
            });

        // A Qwen4 MLP hyperconnection may take the output combine (steps 7-8)
        // into its next preparation (ATLAS_QWEN4_PREFILL_MHC_FUSED=combine).
        let is_ep_prefill =
            ctx.config.ep_world_size > 1 && ctx.comm.is_some_and(|c| c.world_size() > 1);
        let gate_scalars = if crate::layers::qwen4_hyper::prefill_inject::selection()?.gatepre {
            super::qwen4_compact_ops::take_gate_scalars(
                input,
                self.weights.shared_expert_gate.weight,
                num_tokens,
            )
        } else {
            spark_runtime::gpu::DevicePtr::NULL
        };
        if has_shared
            && !is_ep_prefill
            && !use_overlap
            && !super::dump::enabled()
            && crate::layers::qwen4_hyper::pending::record_combine(
                crate::layers::qwen4_hyper::pending::MoeCombine {
                    slot_scales,
                    slot_corrections,
                    expert_out: expert_down_out,
                    token_to_perm,
                    topk_weights: weights_dev,
                    shared_out: ctx.buffers.attn_output(),
                    gate_input: input,
                    shared_gate: self.weights.shared_expert_gate.weight,
                    gate_scalars,
                    topk: top_k,
                },
            )
        {
            return Ok(());
        }

        // 7 + 8. Unpermute + weighted reduce + shared expert blend.
        let output = ctx.buffers.moe_output();
        let fused_unpermute_blend = has_shared
            && !is_ep_prefill
            && !use_overlap
            && !super::dump::enabled()
            && slot_storage.is_mxfp8()
            && !slot_storage.has_corrections()
            && !gate_scalars.is_null()
            && top_k == 10
            && h == 2560
            && n > 0;
        let ran_fused_unpermute_blend = if fused_unpermute_blend
            && let Some(k) = self
                .qwen4_compact
                .and_then(|c| c.wide)
                .and_then(|w| w.unpermute_blend_mxfp8)
        {
            super::qwen4_compact_ops::launch_unpermute_reduce_blend_mxfp8_bf16(
                super::qwen4_compact_ops::UnpermuteBlendArgs {
                    gpu: ctx.gpu,
                    legacy_kernel: k,
                    slot_q: expert_down_out,
                    slot_s: slot_scales,
                    shared_out: ctx.buffers.attn_output(),
                    ss_ptr: gate_scalars,
                    output,
                    token_to_perm,
                    topk_weights: weights_dev,
                    hidden: h,
                    rows: n as usize,
                    top_k,
                    routed_status: spark_runtime::gpu::DevicePtr::NULL,
                    shared_status: spark_runtime::gpu::DevicePtr::NULL,
                    sticky_err: spark_runtime::gpu::DevicePtr::NULL,
                    stream,
                },
            )?;
            true
        } else {
            false
        };
        if !ran_fused_unpermute_blend && slot_storage.is_mxfp8() {
            let unpermute = anyhow::Context::context(
                self.qwen4_compact
                    .and_then(|k| k.wide)
                    .and_then(|w| w.unpermute_mxfp8),
                "MXFP8 slot rows without the MXFP8 unpermute kernel",
            )?;
            if slot_storage.has_corrections() {
                crate::model::qwen4_prefill_engagement::preflight_slot_r2(
                    crate::model::qwen4_prefill_engagement::MoeSlotR2Consumer::Direct,
                    num_tokens,
                )?;
            }
            let launch = spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, unpermute)
                .grid([n, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(expert_down_out)
                .arg_ptr(slot_scales);
            let launch = if slot_storage.has_corrections() {
                launch.arg_ptr(slot_corrections)
            } else {
                launch
            };
            launch
                .arg_ptr(output)
                .arg_ptr(token_to_perm)
                .arg_ptr(weights_dev)
                .arg_u32(h)
                .arg_u32(n)
                .arg_u32(top_k)
                .launch(stream)?;
            if slot_storage.has_corrections() {
                crate::model::qwen4_prefill_engagement::commit_slot_r2(
                    crate::model::qwen4_prefill_engagement::MoeSlotR2Consumer::Direct,
                );
            }
        } else if !ran_fused_unpermute_blend {
            ops::moe_unpermute_reduce_indexed(
                ctx.gpu,
                self.moe_unpermute_reduce,
                expert_down_out,
                output,
                token_to_perm,
                weights_dev,
                h,
                n,
                top_k,
                stream,
            )?;
        }

        // 8. Blend shared expert: output += sigmoid(dot(input, gate)) * shared
        // Skip when has_shared == false (no shared expert in this model config)
        // or when the fused v17 unpermute+blend kernel already blended using gate_scalars.
        // EP fix: defer shared expert blend until AFTER all-reduce to avoid doubling.
        if has_shared && !is_ep_prefill && !ran_fused_unpermute_blend {
            let shared_down_out = ctx.buffers.attn_output();
            if use_overlap {
                overlap_join_guard.join()?;
            }
            super::dump::dump_routed_only(ctx.gpu, stream, output, n, h)?;
            super::dump::dump_shared_out(ctx.gpu, stream, shared_down_out, n, h)?;
            super::dump::dump_shared_gate(
                ctx.gpu,
                stream,
                input,
                self.weights.shared_expert_gate.weight,
                n,
                h,
            )?;
            ops::moe_batched_blend(
                ctx.gpu,
                self.moe_batched_blend,
                output,
                shared_down_out,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                n,
                stream,
            )?;
        }
        super::dump::dump_moe_out(ctx.gpu, stream, output, n, h)?;
        prof_step!("unpermute_blend");

        // EP all-reduce
        if ctx.config.ep_world_size > 1
            && let Some(comm) = ctx.comm
            && comm.world_size() > 1
        {
            let _t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            if ctx.graph_capture {
                comm.all_reduce(output.0, num_tokens * h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, num_tokens * h as usize * 2, stream)?;
            }
            if let Some(t0) = _t0 {
                ctx.gpu.synchronize(stream)?;
                tracing::info!(
                    "  EP allreduce (moe out) N={}: {}µs",
                    num_tokens,
                    t0.elapsed().as_micros(),
                );
            }
            // Add shared expert ONCE after all-reduce (prevents EP doubling)
            if has_shared {
                let shared_down_out = ctx.buffers.attn_output();
                if use_overlap {
                    overlap_join_guard.join()?;
                }
                ops::moe_batched_blend(
                    ctx.gpu,
                    self.moe_batched_blend,
                    output,
                    shared_down_out,
                    input,
                    self.weights.shared_expert_gate.weight,
                    h,
                    n,
                    stream,
                )?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod router_tk64_tests {
    use super::{KernelHandle, router_bf16_tk64_eligible, validate_router_bf16_tk64};

    #[test]
    fn small_tail_uses_incumbent_route() {
        assert!(!router_bf16_tk64_eligible(true, 1));
        assert!(!router_bf16_tk64_eligible(true, 31));
        assert!(router_bf16_tk64_eligible(true, 32));
        assert!(!router_bf16_tk64_eligible(false, 8192));
    }

    #[test]
    fn accepts_minimum_and_ragged_token_counts() {
        for rows in [32, 33, 4096, 8192] {
            assert!(validate_router_bf16_tk64(rows, 2560, 512, true, KernelHandle(1)).is_ok());
        }
    }

    #[test]
    fn enforces_cuda_grid_y_limit() {
        assert!(validate_router_bf16_tk64(1_048_560, 2560, 512, true, KernelHandle(1)).is_ok());
        assert!(validate_router_bf16_tk64(1_048_561, 2560, 512, true, KernelHandle(1)).is_err());
    }

    #[test]
    fn refuses_incompatible_route_shape_or_missing_kernel() {
        for (rows, hidden, experts, bf16, kernel) in [
            (31, 2560, 512, true, 1),
            (32, 2048, 512, true, 1),
            (32, 2560, 256, true, 1),
            (32, 2560, 512, false, 1),
            (32, 2560, 512, true, 0),
        ] {
            assert!(
                validate_router_bf16_tk64(rows, hidden, experts, bf16, KernelHandle(kernel))
                    .is_err()
            );
        }
        if let Some(rows) = (u32::MAX as usize).checked_add(1) {
            assert!(validate_router_bf16_tk64(rows, 2560, 512, true, KernelHandle(1)).is_err());
        }
    }

    #[test]
    fn prefill_shared_overlap_enables_on_high_sm_discrete_and_respects_overrides() {
        use super::resolve_prefill_shared_overlap;

        // GB10 (48 SMs, integrated) and unset unit-test state stay disabled.
        assert!(!resolve_prefill_shared_overlap(None, 48, Some(true)));
        assert!(!resolve_prefill_shared_overlap(None, 48, None));
        assert!(!resolve_prefill_shared_overlap(None, 128, Some(true)));
        assert!(!resolve_prefill_shared_overlap(None, 128, None));
        assert!(!resolve_prefill_shared_overlap(None, 48, Some(false)));

        // Discrete GPUs with >= 64 SMs enable dual-stream shared+routed overlap by default.
        for sm in [64u32, 110, 170, 188] {
            assert!(
                resolve_prefill_shared_overlap(None, sm, Some(false)),
                "discrete sm={sm} should default enabled"
            );
            assert!(
                !resolve_prefill_shared_overlap(Some("0"), sm, Some(false)),
                "explicit =0 must disable on sm={sm}"
            );
            assert!(
                !resolve_prefill_shared_overlap(Some("off"), sm, Some(false)),
                "explicit =off must disable on sm={sm}"
            );
        }

        // Explicit =1 overrides even on 48-SM integrated devices.
        assert!(resolve_prefill_shared_overlap(Some("1"), 48, Some(true)));
    }
}
