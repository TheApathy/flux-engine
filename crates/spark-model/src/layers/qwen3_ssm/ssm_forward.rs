// SPDX-License-Identifier: AGPL-3.0-only

//! ssm_forward inherent helper.

use super::*;

const GDN_FUSED_K1_SM_THRESHOLD: u32 = 64;

/// Whether a model may engage the fused K=1 GDN decode kernel: always allowed
/// for `qwen4_exp`, and also allowed on high-SM discrete GPUs (`sm_count > 64`)
/// for matching `(nk=16, nv=48, kd=128, vd=128, d_conv=4)` sigmoid-gated
/// architectures (preserving GB10 `<= 64` defaults).
const fn fused_k1_model_eligible(
    is_qwen4_exp: bool,
    sm_count: u32,
    nk: usize,
    nv: usize,
    kd: usize,
    vd: usize,
    d_conv: usize,
) -> bool {
    is_qwen4_exp
        || (sm_count > GDN_FUSED_K1_SM_THRESHOLD
            && nk == 16
            && nv == 48
            && kd == 128
            && vd == 128
            && d_conv == 4)
}

/// Whether single-token (`C=1`) FP32 GDN decode should route through the
/// 192-CTA Split-V=4 kernel (`gated_delta_rule_decode_f32_sequence_lazyfinal_splitv4`,
/// 4 CTAs/value-head, 0-ULP identical to `gated_delta_rule_decode_f32`).
const fn single_seq_splitv4_gdn_eligible(
    use_f32_gdn: bool,
    split_v_enabled: bool,
    h_is_f16: bool,
    kd: usize,
    vd: usize,
    has_splitv4_kernel: bool,
) -> bool {
    use_f32_gdn && split_v_enabled && !h_is_f16 && kd == 128 && vd == 128 && has_splitv4_kernel
}

impl Qwen3SsmLayer {
    pub(super) fn ssm_forward(
        &self,
        normed: DevicePtr,
        state: &mut SsmLayerState,
        ctx: &ForwardContext,
        stream: u64,
        trace: bool,
    ) -> Result<DevicePtr> {
        self.ssm_forward_with_fuse(normed, None, normed, stream, state, ctx, trace)
    }

    /// Same as `ssm_forward` but with an optional fused rms_norm_residual entry point.
    ///
    /// When `fuse_input` is `Some((hidden, residual))`, the kernel computes
    /// `rms_norm_residual(hidden, input_norm) -> {normed, residual}` AND the
    /// QKVZ projection in a single launch. `normed` must point at the buffer
    /// that downstream consumers (ba_gates, etc.) will read from.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ssm_forward_with_fuse(
        &self,
        normed: DevicePtr,
        fuse_input: Option<(DevicePtr, DevicePtr)>,
        _normed_alias: DevicePtr,
        stream: u64,
        state: &mut SsmLayerState,
        ctx: &ForwardContext,
        trace: bool,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk; // vheads_per_group = 2
        // Explicit opt-in, not the ambient tracing level — see
        // `crate::layer::decode_debug_dumps_enabled`.
        let debug = crate::layer::decode_debug_dumps_enabled();
        let profile = ctx.profile;
        // See `crate::layer::debug_sync_allowed`: `debug`/`trace` request a
        // blocking host sync to dump a kernel's output, which is illegal
        // while `stream` is mid CUDA-graph-capture and, if allowed, aborts
        // the capture with no `end_capture` — wedging the stream for the
        // rest of the process (the Qwen4 PLE suffix-graph bug). Gate both
        // kinds of sync on this call's `graph_capture` intent and the
        // backend's live capture state.
        let live_capturing = ctx.gpu.stream_is_capturing(stream);
        let debug_sync_ok =
            crate::layer::debug_sync_allowed(debug, false, ctx.graph_capture) && !live_capturing;
        let trace_sync_ok =
            crate::layer::debug_sync_allowed(false, trace, ctx.graph_capture) && !live_capturing;

        macro_rules! prof {
            ($label:expr, $body:expr) => {{
                if profile {
                    let t = std::time::Instant::now();
                    let r = $body;
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!("    SSM {}: {:.0}μs", $label, t.elapsed().as_micros());
                    r
                } else {
                    let r = $body;
                    spark_runtime::cublaslt::stage_timer::mark($label, stream);
                    r
                }
            }};
        }

        // ── 1+2. QKVZ projection (+ deinterleave if needed) ──
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let qkvz_size = ctx.config.ssm_qkvz_size() as u32;
        prof!("qkvz", {
            if let Some(fp8_decode) =
                self.qwen4_fp8_qkvz(ctx.gpu, normed, deinterleaved, qkvz_size, h, stream)
            {
                fp8_decode
            } else if let Some(ref fp8) = self.qkvz_fp8w {
                // FP8 native: w8a16_gemv + deinterleave (no fused QKVZ variant yet)
                if self.sequential_qkvz {
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        normed,
                        fp8.weight,
                        fp8.row_scale,
                        deinterleaved,
                        qkvz_size,
                        h,
                        stream,
                    )
                } else {
                    let qkvz_out = ctx.buffers.ssm_qkvz();
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        normed,
                        fp8.weight,
                        fp8.row_scale,
                        qkvz_out,
                        qkvz_size,
                        h,
                        stream,
                    )?;
                    ops::deinterleave_qkvz(
                        ctx.gpu,
                        self.deinterleave_k,
                        qkvz_out,
                        deinterleaved,
                        1,
                        nk as u32,
                        kd as u32,
                        vpg as u32,
                        vd as u32,
                        stream,
                    )
                }
            } else if self.sequential_qkvz {
                // Qwen3.5: QKVZ weight is pre-concatenated [Q|K|V|Z] sequential.
                // Plain GEMV writes directly to deinterleaved buffer.
                if let Some(dense) = self.qkvz_dense_decode_only.as_ref() {
                    // `ATLAS_QWEN4_GDN_BF16=1` (decode-only). `qkvz_nvfp4`
                    // is untouched, so every prefill/verify consumer that
                    // asserts it directly keeps working unchanged.
                    // `trait_decode.rs` keeps `fuse_qkvz` disabled when this
                    // weight is installed (the fused kernel is NVFP4-only), so
                    // `rms_norm_residual` has already produced `normed`.
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed,
                        dense,
                        deinterleaved,
                        qkvz_size,
                        h,
                        stream,
                    )
                } else if let (Some((fuse_hidden, fuse_residual)), Some(nvfp4)) =
                    (fuse_input, self.qkvz_nvfp4.as_ref())
                {
                    // Fused path: rms_norm_residual + w4a16_gemv in one launch.
                    // Writes `normed` (consumed by ba_gates below), `residual` (consumed
                    // by the post-attn residual_add_rms_norm), and the QKVZ output.
                    ops::rms_norm_residual_w4a16_gemv(
                        ctx.gpu,
                        self.fused_rms_qkvz_k,
                        fuse_hidden,
                        &self.input_norm,
                        normed,
                        fuse_residual,
                        nvfp4,
                        deinterleaved,
                        qkvz_size,
                        h,
                        ctx.config.rms_norm_eps as f32,
                        stream,
                    )
                } else if let Some((proj_t, ws)) = self.decode_tc_parity_qkvz() {
                    self.decode_tc_parity_splitk(
                        ctx,
                        normed,
                        proj_t,
                        deinterleaved,
                        ws,
                        qkvz_size,
                        h,
                        super::super::ssm_qkvz_splitk(),
                        stream,
                    )
                } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                    ops::w4a16_decode_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        self.w4a16_gemv_sw_k,
                        self.gemv_sw,
                        normed,
                        nvfp4,
                        deinterleaved,
                        qkvz_size,
                        h,
                        stream,
                    )
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed,
                        self.bf16_qkvz()?,
                        deinterleaved,
                        qkvz_size,
                        h,
                        stream,
                    )
                }
            } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                // 80B: Fused QKVZ GEMV writes deinterleaved output directly
                ops::w4a16_gemv_qkvz(
                    ctx.gpu,
                    self.w4a16_gemv_qkvz_k,
                    normed,
                    nvfp4,
                    deinterleaved,
                    qkvz_size,
                    h,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )
            } else {
                // 80B fallback: interleaved GEMV + separate deinterleave
                let qkvz_out = ctx.buffers.ssm_qkvz();
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    self.bf16_qkvz()?,
                    qkvz_out,
                    qkvz_size,
                    h,
                    stream,
                )?;
                ops::deinterleave_qkvz(
                    ctx.gpu,
                    self.deinterleave_k,
                    qkvz_out,
                    deinterleaved,
                    1,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )
            }
        })?;
        if trace_sync_ok {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at qkvz_proj");
            })?;
        }
        if debug_sync_ok {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "deinterleaved-Q", deinterleaved, 4);
        }

        // Sequential offsets into deinterleaved buffer: [Q_2048 | K_2048 | V_4096 | Z_4096]
        let key_dim = nk * kd; // 2048
        let value_dim = nv * vd; // 4096
        // Conv1d input starts at Q (first element of deinterleaved)
        let qkv_ptr = deinterleaved;
        // Z gate for gated norm (after Q+K+V)
        let z_ptr = deinterleaved.offset((key_dim * 2 + value_dim) * 2);

        // ── 3. Fused BA projection + GDN gates (single GEMV + inline transforms) ──
        let ba_size = ctx.config.ssm_ba_size() as u32;
        let gates = ctx.buffers.ssm_gates();
        let beta_fp32 = gates.offset(nv * 4); // FP32, after gate[nv]
        prof!("ba_gates", {
            ops::dense_gemv_ba_gates(
                ctx.gpu,
                self.ba_gates_k,
                normed,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gates,
                beta_fp32,
                ba_size,
                h,
                vpg as u32,
                stream,
            )
        })?;
        if trace_sync_ok {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at ba_gates");
            })?;
        }
        if debug_sync_ok {
            ctx.gpu.synchronize(stream)?;
            Self::debug_f32(ctx.gpu, "gate", gates, 4);
            Self::debug_f32(ctx.gpu, "beta", beta_fp32, 4);
        }

        let normed_out = ctx.buffers.ssm_qkvz();
        let lane = crate::layers::qwen4_verify_lanes::current_lane();
        let max_lanes = crate::layers::qwen4_verify_lanes::MAX_LANES;
        let use_fused_k1 = super::qwen4_gdn_verify_fused_enabled()
            && fused_k1_model_eligible(
                ctx.config.is_qwen4_exp(),
                atlas_core::device::active_sm_count(),
                nk,
                nv,
                kd,
                vd,
                ctx.config.linear_conv_kernel_dim,
            )
            && self.qwen4_gdn_decode_fused_k1_k.0 != 0
            && !self.gdn_fused_group_done.is_null()
            && self.conv1d_l2norm_f32_k.0 != 0
            && self.gdn_f32_k.0 != 0
            && self.gated_rms_norm_f32_k.0 != 0
            && !state.h_is_f16
            && kd == 128
            && vd == 128
            && ctx.config.linear_conv_kernel_dim == 4
            && nk > 0
            && nk <= 32
            && nv >= nk
            && nv.is_multiple_of(nk)
            && lane < max_lanes
            && ctx.config.output_gate_type == "sigmoid"
            && !debug_sync_ok;

        if use_fused_k1 {
            let counter_slot = if stream == ctx.gpu.default_stream() {
                lane
            } else {
                max_lanes + lane
            };
            let group_done = self.gdn_fused_group_done.offset(counter_slot * 32 * 4);
            let splitv4_ready = super::qwen4_gdn_split_v_enabled_for_single_seq()
                && self.qwen4_gdn_decode_fused_k1_splitv4_k.0 != 0;
            if splitv4_ready
                && let (Some((scratch_gdn, scratch_ss)), Some(split_done)) = (
                    Self::gdn_split_v_scratch_ptrs(ctx, 1, nv),
                    self.gdn_split_done_ptr(counter_slot, nv),
                )
            {
                static SPLITV4_K1_LOGGED: std::sync::Once = std::sync::Once::new();
                SPLITV4_K1_LOGGED.call_once(|| {
                    tracing::info!(
                        nv,
                        ctas = nv * ops::GDN_SPLIT_V_FACTOR as usize,
                        "ENGAGED ATLAS_QWEN4_GDN_SPLIT_V: Split-V=4 fused K=1 GDN decode"
                    );
                });
                ops::qwen4_gdn_decode_fused_k1_splitv4(
                    ctx.gpu,
                    self.qwen4_gdn_decode_fused_k1_splitv4_k,
                    state.conv_state,
                    deinterleaved,
                    &self.ssm.conv1d,
                    gates,
                    beta_fp32,
                    state.h_state,
                    &self.ssm.norm,
                    normed_out,
                    group_done,
                    nk as u32,
                    nv as u32,
                    1e-6,
                    ctx.config.rms_norm_eps as f32,
                    scratch_gdn,
                    scratch_ss,
                    split_done,
                    stream,
                )?;
            } else {
                ops::qwen4_gdn_decode_fused_k1(
                    ctx.gpu,
                    self.qwen4_gdn_decode_fused_k1_k,
                    state.conv_state,
                    deinterleaved,
                    &self.ssm.conv1d,
                    gates,
                    beta_fp32,
                    state.h_state,
                    &self.ssm.norm,
                    normed_out,
                    group_done,
                    nk as u32,
                    nv as u32,
                    1e-6,
                    ctx.config.rms_norm_eps as f32,
                    stream,
                )?;
            }
            if trace_sync_ok {
                ctx.gpu.synchronize(stream).inspect_err(|_e| {
                    tracing::error!("CRASH at qwen4_gdn_decode_fused_k1");
                })?;
            }
            spark_runtime::cublaslt::stage_timer::mark("gdn", stream);
        } else {
            // ── 4+5b. Fused conv1d update + SiLU + L2 norm on Q/K ──
            // Combines causal_conv1d_update + l2_norm_bf16 into one kernel launch.
            // Q+K channels (0..4096) get L2-normalized per head after SiLU.
            // V channels (4096..8192) get SiLU only.
            let conv_dim = (key_dim * 2 + value_dim) as u32;
            let d_conv = ctx.config.linear_conv_kernel_dim as u32;
            let qk_channels = (key_dim * 2) as u32;
            // Use FP32 conv output to prevent recurrent precision drift at 8k+ tokens.
            // The conv→GDN path is the SSM recurrent path; BF16 truncation at each token
            // compounds to ~noise after 8000 iterations.
            let (conv_out, use_f32_conv) = if self.conv1d_l2norm_f32_k.0 != 0 {
                (ctx.buffers.ssm_conv_out_f32(), true)
            } else {
                (ctx.buffers.ssm_qkvz(), false)
            };
            if use_f32_conv {
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_f32_k,
                    state.conv_state,
                    qkv_ptr,
                    &self.ssm.conv1d,
                    conv_out,
                    conv_dim,
                    d_conv,
                    1,
                    qk_channels,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
            } else {
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
                    state.conv_state,
                    qkv_ptr,
                    &self.ssm.conv1d,
                    conv_out,
                    conv_dim,
                    d_conv,
                    1,
                    qk_channels,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
            }
            if trace_sync_ok {
                ctx.gpu.synchronize(stream).inspect_err(|_e| {
                    tracing::error!("CRASH at conv1d_l2norm");
                })?;
            }
            if debug_sync_ok {
                ctx.gpu.synchronize(stream)?;
                if use_f32_conv {
                    Self::debug_f32(ctx.gpu, "conv1d-l2norm-out", conv_out, 4);
                } else {
                    Self::debug_bf16(ctx.gpu, "conv1d-l2norm-out", conv_out, 4);
                }
            }

            // ── 5. Split conv output → Q', K', V' ──
            spark_runtime::cublaslt::stage_timer::mark("conv", stream);
            // Bytes per element: 4 if FP32 conv output, 2 if BF16
            let elem_size = if use_f32_conv { 4 } else { 2 };
            let q_conv = conv_out;
            let k_conv = conv_out.offset(key_dim * elem_size);
            let v_conv = conv_out.offset(key_dim * 2 * elem_size);

            // ── 6. GDN decode ──
            // Kernel handles GQA: maps value_head → key_head via kh = vh / head_repeat
            // Q/K: [num_k_heads, k_dim], V: [num_v_heads, v_dim]
            // Use FP32 output variant when available to prevent BF16 truncation that
            // compounds to precision drift at 15K+ tokens in multi-turn conversations.
            let use_f32_gdn = self.gdn_f32_k.0 != 0 && self.gated_rms_norm_f32_k.0 != 0;
            let gdn_out = if use_f32_gdn {
                // Write FP32 GDN output to the unused tail of ssm_conv_out_f32.
                // Buffer layout: [Q|K|V](conv data) [Z-region](unused during decode).
                // Conv uses (2*key_dim + value_dim)*4 bytes; GDN output needs value_dim*4
                // bytes which fits in the remaining Z-region.
                ctx.buffers
                    .ssm_conv_out_f32()
                    .offset((key_dim * 2 + value_dim) * 4)
            } else {
                ctx.buffers.attn_output() // BF16 fallback
            };
            if single_seq_splitv4_gdn_eligible(
                use_f32_gdn,
                super::qwen4_gdn_split_v_enabled_for_single_seq(),
                state.h_is_f16,
                kd,
                vd,
                self.gdn_f32_sequence_lazyfinal_splitv4_k.0 != 0,
            ) {
                static SPLITV4_PLAIN_LOGGED: std::sync::Once = std::sync::Once::new();
                SPLITV4_PLAIN_LOGGED.call_once(|| {
                    tracing::info!(
                        nv,
                        ctas = nv * ops::GDN_SPLIT_V_FACTOR as usize,
                        "ENGAGED Split-V=4 plain C=1 FP32 GDN decode (192 CTAs, 0-ULP exact)"
                    );
                });
                // LAZY_FINAL=true with num_tokens=1 reads each column of H into
                // registers before the single-step update and writes the final
                // column back once to `h_state_inter == state.h_state` (disjoint
                // columns across the 4*32 threads per value head).
                ops::gdn_decode_f32_sequence_persistent_splitv4(
                    ctx.gpu,
                    self.gdn_f32_sequence_lazyfinal_splitv4_k,
                    state.h_state,
                    q_conv,
                    k_conv,
                    v_conv,
                    gates,
                    beta_fp32,
                    gdn_out,
                    state.h_state,
                    1,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    0,
                    0,
                    0,
                    0,
                    0,
                    stream,
                )?;
            } else {
                let gdn_kernel = if use_f32_gdn {
                    self.gdn_f32_k
                } else {
                    self.gdn_k
                };
                ops::gdn_decode(
                    ctx.gpu,
                    gdn_kernel,
                    state.h_state,
                    q_conv,
                    k_conv,
                    v_conv,
                    gates,
                    beta_fp32,
                    gdn_out,
                    1,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    stream,
                )?;
            }
            if trace_sync_ok {
                ctx.gpu.synchronize(stream).inspect_err(|_e| {
                    tracing::error!("CRASH at gdn_decode");
                })?;
            }

            // ── 7. Gated RMS norm with Z gate — per-head normalization ──
            // GDN output is [nv * vd], norm weight is [vd], applied as [nv, vd]
            spark_runtime::cublaslt::stage_timer::mark("gdn", stream);
            let norm_kernel = if use_f32_gdn {
                self.gated_rms_norm_f32_k
            } else {
                self.gated_rms_norm_k
            };
            ops::gated_rms_norm(
                ctx.gpu,
                norm_kernel,
                gdn_out,
                z_ptr,
                &self.ssm.norm,
                normed_out,
                nv as u32,
                vd as u32,
                vd as u32,
                ctx.config.rms_norm_eps as f32,
                vd as u32,
                stream,
            )?;
            if trace_sync_ok {
                ctx.gpu.synchronize(stream).inspect_err(|_e| {
                    tracing::error!("CRASH at gated_rms_norm");
                })?;
            }
            if debug_sync_ok {
                ctx.gpu.synchronize(stream)?;
                Self::debug_bf16(ctx.gpu, "gated-norm-out", normed_out, 4);
            }
        }

        // ── 8. Output projection: [value_dim → hidden_size] ──
        let out = ctx.buffers.moe_output();
        if let Some(fp8_decode) =
            self.qwen4_fp8_out_proj(ctx.gpu, normed_out, out, h, value_dim as u32, stream)
        {
            fp8_decode?;
        } else if let Some(ref dense_out) = self.out_proj_dense_decode_only {
            // `ATLAS_QWEN4_GDN_BF16=1`: decode-only BF16 out_proj (checked with
            // higher priority than out_proj_fp8w/out_proj_dense/NVFP4 per mod.rs).
            // Separate field from `out_proj_dense` so prefill (which reads that
            // field with higher priority than FP8 in
            // `trait_prefill_phase3.rs`) is untouched.
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                normed_out,
                dense_out,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        } else if let Some(ref fp8) = self.out_proj_fp8w {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed_out,
                fp8.weight,
                fp8.row_scale,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        } else if let Some(ref dense_out) = self.out_proj_dense {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                normed_out,
                dense_out,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        } else if let Some((proj_t, ws)) = self.decode_tc_parity_out() {
            self.decode_tc_parity_splitk(
                ctx,
                normed_out,
                proj_t,
                out,
                ws,
                h,
                value_dim as u32,
                super::super::ssm_out_splitk(),
                stream,
            )?;
        } else {
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                self.gemv_sw,
                normed_out,
                &self.ssm.out_proj,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        }
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(out.0, (h as usize) * 2, stream)?;
        }
        if trace_sync_ok {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at out_proj");
            })?;
        }
        if debug_sync_ok {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "out-proj", out, 4);
        }

        spark_runtime::cublaslt::stage_timer::mark("norm_out", stream);
        Ok(out)
    }

    /// `ATLAS_DECODE_TC_PARITY=1`: the transposed QKVZ weight + split-K
    /// workspace when single-token decode must mirror the K=γ verify's
    /// `ATLAS_SSM_PROJ_TC=1` split-K route (see `trait_decode_batched.rs`).
    /// `None` keeps the ordinary K1 GEMV.
    fn decode_tc_parity_qkvz(&self) -> Option<(&QuantizedWeight, DevicePtr)> {
        if !self.decode_tc_parity_splitk_ready(super::super::ssm_qkvz_splitk())
            || self.qkvz_fp8.is_some()
            || super::super::tc_nvfp4_m16_enabled()
        {
            return None;
        }
        let ws = (*self.ssm_splitk_workspace.lock().unwrap())?;
        Some((self.qkvz_nvfp4_t.as_ref()?, ws))
    }

    /// Output-projection counterpart of [`Self::decode_tc_parity_qkvz`].
    fn decode_tc_parity_out(&self) -> Option<(&QuantizedWeight, DevicePtr)> {
        if !self.decode_tc_parity_splitk_ready(super::super::ssm_out_splitk())
            || self.out_proj_dense.is_some()
        {
            return None;
        }
        let ws = (*self.ssm_splitk_workspace.lock().unwrap())?;
        Some((self.out_proj_nvfp4_t.as_ref()?, ws))
    }

    fn decode_tc_parity_splitk_ready(&self, splits: u32) -> bool {
        super::super::decode_tc_parity_enabled()
            && super::super::ssm_proj_tc_enabled()
            && splits > 0
            && self.w4a16_gemm_t_m32_n64_splitk_k.0 != 0
            && self.reduce_splitk_k.0 != 0
    }

    /// One row through the verify's split-K tensor-core GEMM. The kernel is
    /// row-independent, so this equals row r of an M=k verify bit-for-bit.
    #[allow(clippy::too_many_arguments)]
    fn decode_tc_parity_splitk(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        weight_t: &QuantizedWeight,
        output: DevicePtr,
        workspace: DevicePtr,
        n: u32,
        k: u32,
        splits: u32,
        stream: u64,
    ) -> Result<()> {
        ops::w4a16_gemm_n64_m32_splitk(
            ctx.gpu,
            self.w4a16_gemm_t_m32_n64_splitk_k,
            self.reduce_splitk_k,
            input,
            weight_t,
            output,
            workspace,
            1,
            n,
            k,
            n, // ldb == N for tightly-packed T-weight
            splits,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{fused_k1_model_eligible, single_seq_splitv4_gdn_eligible};

    #[test]
    fn fused_k1_lifts_qwen4_guard_for_matching_geometry_above_64_sms_only() {
        // Qwen4 always eligible regardless of SM count.
        assert!(fused_k1_model_eligible(true, 48, 16, 48, 128, 128, 4));
        // Non-Qwen4 matching geometry (nk=16, nv=48, kd=128, vd=128, d_conv=4):
        // OFF on GB10 (<=64 SMs), ON on discrete SM120/SM103 (>64 SMs).
        assert!(!fused_k1_model_eligible(false, 48, 16, 48, 128, 128, 4));
        assert!(!fused_k1_model_eligible(false, 64, 16, 48, 128, 128, 4));
        assert!(fused_k1_model_eligible(false, 110, 16, 48, 128, 128, 4));
        assert!(fused_k1_model_eligible(false, 188, 16, 48, 128, 128, 4));
        // Non-matching geometry stays off.
        assert!(!fused_k1_model_eligible(false, 188, 16, 32, 128, 128, 4));
        assert!(!fused_k1_model_eligible(false, 188, 16, 48, 128, 128, 3));
    }

    #[test]
    fn single_seq_splitv4_gdn_requires_fp32_hd128_and_policy() {
        assert!(single_seq_splitv4_gdn_eligible(
            true, true, false, 128, 128, true
        ));
        assert!(!single_seq_splitv4_gdn_eligible(
            true, false, false, 128, 128, true
        ));
        assert!(!single_seq_splitv4_gdn_eligible(
            false, true, false, 128, 128, true
        ));
        assert!(!single_seq_splitv4_gdn_eligible(
            true, true, true, 128, 128, true
        ));
        assert!(!single_seq_splitv4_gdn_eligible(
            true, true, false, 64, 128, true
        ));
        assert!(!single_seq_splitv4_gdn_eligible(
            true, true, false, 128, 128, false
        ));
    }
}
