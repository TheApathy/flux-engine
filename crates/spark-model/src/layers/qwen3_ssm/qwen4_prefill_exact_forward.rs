// SPDX-License-Identifier: AGPL-3.0-only

use super::qwen4_prefill_check::Snapshot;
use super::qwen4_prefill_exact_plan::{Plan, grid32_partition};
use super::qwen4_prefill_gemm::Projection;
use super::*;

const GDN_LAZYFINAL_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_GDN_LAZYFINAL";

const CONV_PARALLEL_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_CONV_PARALLEL";

fn conv_parallel_selected() -> bool {
    static SEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SEL.get_or_init(|| std::env::var(CONV_PARALLEL_SELECTOR).ok().as_deref() == Some("1"))
}

fn conv_parallel_kernels(gpu: &dyn GpuBackend) -> Result<(KernelHandle, KernelHandle)> {
    static K: std::sync::OnceLock<Result<(KernelHandle, KernelHandle), String>> =
        std::sync::OnceLock::new();
    K.get_or_init(|| {
        let a = gpu
            .kernel(
                "causal_conv1d_prefill_parallel",
                "causal_conv1d_update_l2norm_f32_prefill_parallel",
            )
            .map_err(|e| e.to_string())?;
        let b = gpu
            .kernel(
                "causal_conv1d_prefill_parallel",
                "causal_conv1d_prefill_state_commit",
            )
            .map_err(|e| e.to_string())?;
        Ok((a, b))
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{CONV_PARALLEL_SELECTOR}: kernel unavailable: {e}"))
}

fn gdn_lazyfinal_selected() -> bool {
    gdn_lazyfinal_kernel_name().is_some()
}

/// 1 = register twin, 2 = its prefetch twin, 3 = H truly in registers (the
/// `_regfinal` H array sits in a local-memory stack frame), 4/5 = 3 plus the
/// token stream staged 15/8 tokens at a time through shared memory. All five
/// share the arithmetic order of the nosnap kernel.
fn gdn_lazyfinal_kernel_name() -> Option<&'static str> {
    static SEL: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    *SEL.get_or_init(
        || match std::env::var(GDN_LAZYFINAL_SELECTOR).ok().as_deref() {
            Some("1") => Some("gated_delta_rule_prefill_f32_sequence_regfinal"),
            Some("2") => Some("gated_delta_rule_prefill_f32_sequence_regfinal_pf"),
            Some("3") => Some("gated_delta_rule_prefill_f32_sequence_regfinal_full"),
            Some("4") => Some("gated_delta_rule_prefill_f32_sequence_regfinal_chunk15"),
            Some("5") => Some("gated_delta_rule_prefill_f32_sequence_regfinal_chunk8"),
            _ => None,
        },
    )
}

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_qwen4_ssm_exact(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        rows: usize,
        state: &mut SsmLayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let p = self.preflight_qwen4_prefill_exact(hidden, residual, rows, state, ctx)?;
        let s = p.shape;
        let projection_mode = qwen4_prefill_gemm::mode()?;
        let check = if qwen4_prefill_exact::check_selected()? {
            Some(Snapshot::capture(rows, hidden, state, ctx, stream)?)
        } else {
            None
        };
        let attn = self.qwen4_attn_hyper.as_ref().expect("admitted attn HC");
        let eps = ctx.config.rms_norm_eps as f32;
        use spark_runtime::cublaslt::stage_timer::mark;
        mark("entry", stream);
        // ATLAS_QWEN4_PREFILL_FP8_ACT_AMAX_FUSED: the preparation reduces the
        // absmax the tensor-scale QKVZ projection below would read its rows for.
        let amax_hand_off = {
            use crate::layers::qwen4_fast_proj as fp;
            crate::layers::qwen4_fp8_act_amax::Arm::enter(
                check.is_none()
                    && fp::selection()?.ssm
                    && fp::fp8_selection()?.ssm
                    && fp::fp8_tensor_scales(),
            )?
        };
        // The full-row GDN core reads only the packed rows in norm_output;
        // the check replays from the staged residual rows.
        let input = if check.is_some() {
            attn.prepare_prefill_exact(hidden, residual, rows, ctx.buffers, ctx.gpu, eps, stream)?
        } else {
            attn.prepare_prefill_exact_packed(
                hidden,
                residual,
                rows,
                ctx.buffers,
                ctx.gpu,
                eps,
                stream,
            )?
        };
        mark("hc_prepare", stream);
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let gates = ctx.buffers.ssm_gates();
        let fast = crate::layers::qwen4_fast_proj::selection()?;
        let fp8_ssm = fast.ssm && crate::layers::qwen4_fast_proj::fp8_selection()?.ssm;
        let fused_ba = fp8_ssm
            && s.gates == Plan::GATES
            && crate::layers::qwen4_fast_proj::fp8_tensor_scales()
            && super::qwen4_ba_gemm::fused_act_selected();
        let qkvz_pad_guard = crate::layers::qwen4_fast_proj::PadRowsGuard::enter(
            ctx.buffers.sizes().ssm_deinterleaved / (Plan::QKVZ * 2),
        );
        if fused_ba {
            static LOGGED_FUSED_BA: std::sync::Once = std::sync::Once::new();
            LOGGED_FUSED_BA.call_once(|| {
                tracing::info!(
                    "SSM_PREFILL_FUSED_QKVZ_BA_ENGAGED selector={} rows={rows}",
                    super::qwen4_ba_gemm::FUSED_ACT_SELECTOR
                );
            });
            let planar = super::qwen4_ba_gemm::uses_planar_fi_gates(rows);
            crate::layers::qwen4_fast_proj::fp8_qkvz_and_ba_gates_fused(
                ctx.gpu,
                input,
                self.qkvz_nvfp4.as_ref().expect("admitted NVFP4 QKVZ"),
                deinterleaved,
                self.ssm.in_proj_ba.weight,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gates,
                rows,
                s.qkvz,
                Plan::H,
                planar,
                stream,
            )?;
        } else if fast.ssm {
            let projection = if fp8_ssm {
                "fp8_cublaslt_non_bit_exact"
            } else {
                "dequant_cublaslt_bf16_non_bit_exact"
            };
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::info!(
                    "SSM_PREFILL_FAST_ENGAGED selector={} rows={rows} projection={projection}",
                    crate::layers::qwen4_fast_proj::SELECTOR
                );
            });
            crate::layers::qwen4_fast_proj::proj_gemm_ex(
                ctx.gpu,
                fp8_ssm,
                input,
                self.qkvz_nvfp4.as_ref().expect("admitted NVFP4 QKVZ"),
                deinterleaved,
                rows,
                s.qkvz,
                Plan::H,
                s.qkvz,
                None,
                stream,
            )?;
        } else if projection_mode.uses_gemm(Projection::Qkvz) {
            self.project_prefill_gemm(Projection::Qkvz, input, deinterleaved, rows, ctx, stream)?;
        } else {
            self.project_prefill_exact_tiles(
                input,
                self.qkvz_nvfp4.as_ref().expect("admitted NVFP4 QKVZ"),
                deinterleaved,
                &p,
                s.qkvz,
                Plan::H,
                ctx,
                stream,
            )?;
        }
        drop(qkvz_pad_guard);
        mark("proj_qkvz", stream);
        drop(amax_hand_off);
        if !fused_ba {
            if super::qwen4_ba_gemm::selected() {
                self.qwen4_ba_gates_gemm(input, gates, rows, ctx, stream)?;
            } else {
                ops::dense_gemv_ba_gates_batchn(
                    ctx.gpu,
                    self.ba_gates_batchn_exact_k,
                    input,
                    &self.ssm.in_proj_ba,
                    self.ssm.a_log.weight,
                    self.ssm.dt_bias.weight,
                    gates,
                    gates.offset(s.nv * 4),
                    rows as u32,
                    s.gates as u32,
                    Plan::H as u32,
                    (s.nv / s.nk) as u32,
                    s.gates as u32,
                    stream,
                )?;
            }
        }

        mark("ba_gates", stream);
        super::qwen4_prefill_gdn_fi_dump::maybe_dump_inputs(
            state,
            deinterleaved,
            gates,
            rows,
            ctx,
            stream,
        )?;
        let normed = ctx.buffers.ssm_qkvz();
        // FlashInfer's asynchronous state path is not qualified at <=64
        // rows.  Decide the GDN route before selecting either fused norm
        // consumer so the exact fallback always materializes BF16 `normed`.
        let gdn_fi_selected = super::qwen4_gdn_fi::selected();
        let use_fi = super::qwen4_gdn_fi_policy::use_fi(gdn_fi_selected, rows);
        if gdn_fi_selected && !use_fi {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::info!(
                    "SSM_PREFILL_GDN_FI_SHORT_FALLBACK_ENGAGED rows={rows} threshold={} reason=async_fi_gdn_state selector={}",
                    super::qwen4_gdn_fi_policy::SHORT_FALLBACK_MAX_ROWS,
                    super::qwen4_gdn_fi::SELECTOR,
                );
            });
        }
        // Default-off: the gated norm runs fused with the out-projection FP8
        // quantization after the stage mark (qwen4_gdn_norm_fp8.rs). The check
        // replays the BF16 normed rows, so it keeps the unfused path.
        let selected_norm_fp8 = if check.is_none() {
            super::qwen4_gdn_norm_fp8::mode()?
        } else {
            None
        };
        let norm_fp8 = if use_fi { selected_norm_fp8 } else { None };
        // =3: the FI kernel itself writes the e4m3 out-projection rows.
        let nf8 = match norm_fp8 {
            Some(super::qwen4_gdn_norm_fp8::Mode::FiStatic) => {
                Some(self.qwen4_gdn_nf8_scratch(rows, ctx)?)
            }
            _ => None,
        };
        let output = ctx.buffers.moe_output();
        let mut did_strip_out_proj = false;
        if use_fi && norm_fp8 == Some(super::qwen4_gdn_norm_fp8::Mode::Static) {
            let out_pad_guard = crate::layers::qwen4_fast_proj::PadRowsGuard::enter(
                ctx.buffers.sizes().moe_output / (Plan::H * 2),
            );
            did_strip_out_proj = self.try_qwen4_gdn_fi_strip_norm_fp8_out_proj(
                state,
                deinterleaved,
                gates,
                ctx.buffers.ssm_conv_out_f32(),
                output,
                rows,
                eps,
                ctx,
                stream,
            )?;
            drop(out_pad_guard);
        }
        if !did_strip_out_proj {
            if use_fi {
                self.qwen4_gdn_fi_core(
                    state,
                    deinterleaved,
                    gates,
                    ctx.buffers.ssm_conv_out_f32(),
                    norm_fp8.is_none().then_some(normed),
                    nf8.map(|s| s.fi_outputs()),
                    rows,
                    eps,
                    ctx,
                    stream,
                )?;
            } else {
                // Each FP32 row holds [Q|K|V|GDN output]. Conv and H state are disjoint;
                // preserve token order inside both kernels, without verify snapshots.
                let conv_rows = ctx.buffers.ssm_conv_out_f32();
                if conv_parallel_selected() {
                    self.conv_prefill_parallel(
                        state.conv_state,
                        deinterleaved,
                        conv_rows,
                        rows,
                        ctx,
                        stream,
                    )?;
                } else {
                    ops::conv1d_update_l2norm_f32_sequence(
                        ctx.gpu,
                        self.conv1d_l2norm_f32_sequence_k,
                        state.conv_state,
                        deinterleaved,
                        &self.ssm.conv1d,
                        conv_rows,
                        DevicePtr::NULL,
                        rows as u32,
                        s.conv_dim as u32,
                        4,
                        (s.key_dim * 2) as u32,
                        128,
                        1e-6,
                        s.qkvz as u32,
                        s.qkvz as u32,
                        0,
                        stream,
                    )?;
                }
                let gdn_rows = conv_rows.offset(s.conv_dim * 4);
                // Default-off FP32 TensorFold-style chain (qwen4_prefill_gdn_tf.rs)
                // on the same buffers; otherwise falls back to nosnap (or regfinal
                // when ATLAS_QWEN4_PREFILL_GDN_LAZYFINAL is set).
                if !self.try_prefill_qwen4_gdn_tf(
                    state, conv_rows, gates, gdn_rows, rows, ctx, stream,
                )? {
                    // Default-off: the register-resident lazy-commit twin of the nosnap
                    // kernel (identical arithmetic order, no per-token snapshot writes,
                    // H kept in registers instead of 2 global passes per token).
                    let gdn_kernel = if gdn_lazyfinal_selected() {
                        static K: std::sync::OnceLock<Result<KernelHandle, String>> =
                            std::sync::OnceLock::new();
                        let k = K
                            .get_or_init(|| {
                                ctx.gpu
                                    .kernel(
                                        "gated_delta_rule_prefill_regfinal",
                                        gdn_lazyfinal_kernel_name().expect("selected"),
                                    )
                                    .map_err(|e| e.to_string())
                            })
                            .clone()
                            .map_err(|e| {
                                anyhow::anyhow!("{GDN_LAZYFINAL_SELECTOR}: kernel unavailable: {e}")
                            })?;
                        static LOGGED: std::sync::Once = std::sync::Once::new();
                        LOGGED.call_once(|| {
                            tracing::info!(
                                "SSM_PREFILL_GDN_REGFINAL_ENGAGED selector={GDN_LAZYFINAL_SELECTOR} kernel={} rows={rows}",
                                gdn_lazyfinal_kernel_name().unwrap_or("?")
                            );
                        });
                        k
                    } else {
                        self.gdn_f32_sequence_nosnap_k
                    };
                    ops::gdn_decode_f32_sequence(
                        ctx.gpu,
                        gdn_kernel,
                        state.h_state,
                        conv_rows,
                        conv_rows.offset(s.key_dim * 4),
                        conv_rows.offset(s.key_dim * 2 * 4),
                        gates,
                        gates.offset(s.nv * 4),
                        gdn_rows,
                        DevicePtr::NULL,
                        rows as u32,
                        s.nk as u32,
                        s.nv as u32,
                        128,
                        128,
                        s.qkvz as u32,
                        s.qkvz as u32,
                        s.gates as u32,
                        s.qkvz as u32,
                        0,
                        stream,
                    )?;
                }
                ops::gated_rms_norm_f32_multi_seq(
                    ctx.gpu,
                    self.gated_rms_norm_f32_multi_seq_k,
                    gdn_rows,
                    deinterleaved.offset(s.conv_dim * 2),
                    &self.ssm.norm,
                    normed,
                    s.nv as u32,
                    rows as u32,
                    128,
                    eps,
                    s.qkvz as u32,
                    s.qkvz as u32,
                    s.value_dim as u32,
                    stream,
                )?;
            }
            mark("gdn_conv_norm", stream);
            let out_pad_guard = crate::layers::qwen4_fast_proj::PadRowsGuard::enter(
                ctx.buffers.sizes().moe_output / (Plan::H * 2),
            );
            if let Some(s_nf8) = nf8 {
                self.qwen4_gdn_nf8_out_proj(s_nf8, output, rows, ctx, stream)?;
            } else if let Some(mode) = norm_fp8 {
                self.qwen4_gdn_norm_fp8_out_proj(
                    mode,
                    ctx.buffers.ssm_conv_out_f32().offset(rows * s.conv_dim * 2),
                    deinterleaved.offset(s.conv_dim * 2),
                    output,
                    rows,
                    eps,
                    ctx,
                    stream,
                )?;
            } else if fast.ssm {
                crate::layers::qwen4_fast_proj::proj_gemm_ex(
                    ctx.gpu,
                    crate::layers::qwen4_fast_proj::fp8_selection()?.ssm,
                    normed,
                    &self.ssm.out_proj,
                    output,
                    rows,
                    Plan::H,
                    s.value_dim,
                    Plan::H,
                    None,
                    stream,
                )?;
            } else if projection_mode.uses_gemm(Projection::Output) {
                self.project_prefill_gemm(Projection::Output, normed, output, rows, ctx, stream)?;
            } else {
                self.project_prefill_exact_tiles(
                    normed,
                    &self.ssm.out_proj,
                    output,
                    &p,
                    Plan::H,
                    s.value_dim,
                    ctx,
                    stream,
                )?;
            }
            drop(out_pad_guard);
        }
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(output.0, rows * Plan::H * 2, stream)?;
        }
        mark("proj_out", stream);
        if let Some(check) = check {
            attn.inject_saved_batched(hidden, output, residual, rows, ctx.gpu, stream)?;
            check.verify(self, hidden, residual, rows, state, ctx, stream)?;
        } else {
            // May stay pending for the MLP hyperconnection's normalization.
            attn.inject_saved_or_defer(
                crate::layers::qwen4_hyper::prefill_inject::Boundary::Core,
                hidden,
                output,
                residual,
                rows,
                None,
                ctx.gpu,
                stream,
            )?;
        }
        Ok(())
    }

    /// Token-parallel exact conv1d (+ per-head L2) and shift-register commit.
    fn conv_prefill_parallel(
        &self,
        conv_state: DevicePtr,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
        let s = super::qwen4_prefill_exact_plan::SsmShape::from_heads(
            ctx.config.linear_num_key_heads,
            ctx.config.linear_num_value_heads,
        );
        let (conv_k, commit_k) = conv_parallel_kernels(ctx.gpu)?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "SSM_PREFILL_CONV_PARALLEL_ENGAGED selector={CONV_PARALLEL_SELECTOR} rows={rows}"
            );
        });
        KernelLaunch::new(ctx.gpu, conv_k)
            .grid([div_ceil(s.conv_dim as u32, 256), rows as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(conv_state)
            .arg_ptr(input)
            .arg_ptr(self.ssm.conv1d.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(output)
            .arg_u32(rows as u32)
            .arg_u32(s.conv_dim as u32)
            .arg_u32(4)
            .arg_u32((s.key_dim * 2) as u32)
            .arg_u32(128)
            .arg_f32(1e-6)
            .arg_u32(s.qkvz as u32)
            .arg_u32(s.qkvz as u32)
            .launch(stream)?;
        KernelLaunch::new(ctx.gpu, commit_k)
            .grid([div_ceil(s.conv_dim as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(conv_state)
            .arg_ptr(input)
            .arg_u32(rows as u32)
            .arg_u32(s.conv_dim as u32)
            .arg_u32(4)
            .arg_u32(s.qkvz as u32)
            .launch(stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn project_prefill_exact_tiles(
        &self,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        plan: &Plan,
        n: usize,
        k: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // GRID32 changes only submission geometry: blockIdx.y selects one
        // disjoint M32 slab while every block preserves the incumbent K1
        // accumulation and BF16 projection output. Incomplete tails retain the
        // exact register tier (or shipping single-row decode dispatcher).
        let grid32 = qwen4_prefill_exact::grid32_selected()?;
        let (full_rows, _) = if grid32 {
            grid32_partition(plan.rows)
        } else {
            (0, plan.rows)
        };
        if full_rows != 0 {
            ops::w4a16_gemv_batch_logits_exact_rt2_m32_grid(
                ctx.gpu,
                self.w4a16_exact_projection_kernels,
                input,
                weight,
                output,
                full_rows as u32,
                n as u32,
                k as u32,
                stream,
            )?;
            static SSM_GRID32_ENGAGED: std::sync::Once = std::sync::Once::new();
            SSM_GRID32_ENGAGED.call_once(|| {
                tracing::info!(
                    "SSM_PREFILL_EXACT_GRID32_ENGAGED selector=ATLAS_QWEN4_PREFILL_SSM_GRID32 rows={full_rows}"
                );
            });
        }
        for (start, count) in plan.tiles_from(full_rows) {
            let a = input.offset(start * k * 2);
            let c = output.offset(start * n * 2);
            if count == 1 {
                ops::w4a16_decode_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    self.w4a16_gemv_sw_k,
                    self.gemv_sw,
                    a,
                    weight,
                    c,
                    n as u32,
                    k as u32,
                    stream,
                )?;
            } else {
                let use_rt2 = ops::w4a16_gemv_rt2_enabled();
                if use_rt2 {
                    static SSM_RT2_ENGAGED: std::sync::Once = std::sync::Once::new();
                    SSM_RT2_ENGAGED.call_once(|| {
                        tracing::info!(
                            "SSM_PREFILL_EXACT_RT2_ENGAGED selector=ATLAS_W4A16_GEMV_RT2 rows={count}"
                        );
                    });
                }
                ops::w4a16_gemv_batch_logits_exact_with(
                    ctx.gpu,
                    self.w4a16_exact_projection_kernels,
                    a,
                    weight,
                    c,
                    count as u32,
                    n as u32,
                    k as u32,
                    stream,
                    // The register-tiled exact family is explicitly selected
                    // by ATLAS_W4A16_GEMV_RT2=1 and preflighted above.
                    use_rt2,
                )?;
            }
        }
        Ok(())
    }
}
