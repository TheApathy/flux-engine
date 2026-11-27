// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::prefill.

use spark_runtime::kernel_args::KernelLaunch;

use super::*;

const PREFILL_CONV_FAST_SELECTOR: &str = "ATLAS_PREFILL_CONV_FAST";
const PREFILL_CONV_FAST_SM_THRESHOLD: u32 = 64;

/// Resolve `ATLAS_PREFILL_CONV_FAST`: explicit `"1"`/`"true"` enables,
/// explicit `"0"`/`"false"` disables, and unset auto-enables on discrete
/// high-SM GPUs (`sm_count > 64`) while staying default-OFF on GB10 (`<= 64`).
#[cfg(test)]
fn resolve_prefill_conv_fast(raw: Option<&str>, sm_count: u32) -> bool {
    resolve_prefill_conv_fast_with_gb10(raw, sm_count, false)
}

/// Same resolver as [`resolve_prefill_conv_fast`], additionally auto-enabling
/// when unset under the opt-in `ATLAS_GB10_FAST_PREFILL=1` umbrella while still
/// honoring explicit `"0"`/`"false"` opt-out.
fn resolve_prefill_conv_fast_with_gb10(
    raw: Option<&str>,
    sm_count: u32,
    gb10_fast_prefill: bool,
) -> bool {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
        Some(_) => false,
        None => sm_count > PREFILL_CONV_FAST_SM_THRESHOLD || gb10_fast_prefill,
    }
}

fn prefill_conv_fast_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        let raw = std::env::var(PREFILL_CONV_FAST_SELECTOR).ok();
        resolve_prefill_conv_fast_with_gb10(
            raw.as_deref(),
            atlas_core::device::active_sm_count(),
            ops::gb10_fast_prefill_enabled(),
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefillConvFastRoute {
    Disabled,
    FusedV2Commit,
    ParallelF32ToBf16,
}

#[expect(
    clippy::too_many_arguments,
    reason = "pure route selector mirroring every conv1d+L2 geometry and kernel invariant"
)]
fn select_prefill_conv_fast_route(
    enabled: bool,
    rows: u32,
    conv_dim: u32,
    qk_channels: u32,
    kd: u32,
    d_conv: u32,
    qkvz_size: u32,
    v2_refusal: Option<&'static str>,
    conv_state_aligned_16: bool,
    has_v2_commit_k: bool,
    has_parallel_f32_pipeline: bool,
    conv_f32_bytes_fit: bool,
) -> PrefillConvFastRoute {
    if !enabled
        || rows < 4
        || kd != 128
        || d_conv != 4
        || !conv_dim.is_multiple_of(256)
        || !qk_channels.is_multiple_of(256)
        || qk_channels > conv_dim
        || qkvz_size < conv_dim
    {
        return PrefillConvFastRoute::Disabled;
    }
    if has_v2_commit_k && conv_state_aligned_16 && v2_refusal.is_none() {
        return PrefillConvFastRoute::FusedV2Commit;
    }
    if has_parallel_f32_pipeline && conv_f32_bytes_fit {
        return PrefillConvFastRoute::ParallelF32ToBf16;
    }
    PrefillConvFastRoute::Disabled
}

impl Qwen3SsmLayer {
    #[expect(
        clippy::too_many_arguments,
        reason = "batched prefill conv1d+L2 fast-path launcher matching CUDA operand arity"
    )]
    fn try_prefill_conv1d_l2norm_fast(
        &self,
        ssm_state: &mut SsmLayerState,
        deinterleaved: DevicePtr,
        conv_out_buf: DevicePtr,
        k: u32,
        conv_dim: u32,
        qk_channels: u32,
        kd: u32,
        d_conv: u32,
        qkvz_size: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        use super::qwen4_gdn_conv_v2 as v2;

        if !prefill_conv_fast_enabled() || k < 4 {
            return Ok(false);
        }
        let v2_commit_k = super::super::try_kernel(
            ctx.gpu,
            "qwen4_gdn_fi_glue",
            "qwen4_conv1d_l2norm_bf16_prefill_v2_commit",
        );
        let parallel_k = super::super::try_kernel(
            ctx.gpu,
            "causal_conv1d_prefill_parallel",
            "causal_conv1d_update_l2norm_f32_prefill_parallel",
        );
        let commit_k = super::super::try_kernel(
            ctx.gpu,
            "causal_conv1d_prefill_parallel",
            "causal_conv1d_prefill_state_commit",
        );
        let rne_k = super::super::try_kernel(ctx.gpu, "quantize_nvfp4", "f32_to_bf16_rne");
        let conv_f32_bytes_needed = (k as usize)
            .checked_mul(conv_dim as usize)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()));
        let conv_f32_bytes_fit =
            conv_f32_bytes_needed.is_some_and(|need| need <= ctx.buffers.sizes().ssm_conv_out_f32);
        let v2_refusal = v2::refusal(
            k,
            conv_dim,
            qk_channels,
            kd,
            d_conv,
            qkvz_size,
            conv_dim,
            [deinterleaved.0, conv_out_buf.0, self.ssm.conv1d.weight.0],
        );
        let route = select_prefill_conv_fast_route(
            true,
            k,
            conv_dim,
            qk_channels,
            kd,
            d_conv,
            qkvz_size,
            v2_refusal,
            ssm_state.conv_state.0.is_multiple_of(16),
            v2_commit_k.0 != 0,
            parallel_k.0 != 0 && commit_k.0 != 0 && rne_k.0 != 0,
            conv_f32_bytes_fit,
        );
        match route {
            PrefillConvFastRoute::Disabled => Ok(false),
            PrefillConvFastRoute::FusedV2Commit => {
                KernelLaunch::new(ctx.gpu, v2_commit_k)
                    .grid(v2::grid(conv_dim, k))
                    .block([v2::BLOCK_THREADS, 1, 1])
                    .arg_ptr(ssm_state.conv_state)
                    .arg_ptr(deinterleaved)
                    .arg_ptr(self.ssm.conv1d.weight)
                    .arg_ptr(DevicePtr::NULL)
                    .arg_ptr(conv_out_buf)
                    .arg_u32(k)
                    .arg_u32(conv_dim)
                    .arg_u32(d_conv)
                    .arg_u32(qk_channels)
                    .arg_u32(kd)
                    .arg_f32(1e-6)
                    .arg_u32(qkvz_size)
                    .arg_u32(conv_dim)
                    .launch(stream)?;
                static LOGGED_V2: std::sync::Once = std::sync::Once::new();
                LOGGED_V2.call_once(|| {
                    tracing::info!(
                        "SSM_PREFILL_CONV_FAST_ENGAGED selector={PREFILL_CONV_FAST_SELECTOR} backend=conv1d_l2norm_bf16_v2_commit rows={k} dim={conv_dim}"
                    );
                });
                Ok(true)
            }
            PrefillConvFastRoute::ParallelF32ToBf16 => {
                let conv_f32 = ctx.buffers.ssm_conv_out_f32();
                let total_elems = k
                    .checked_mul(conv_dim)
                    .ok_or_else(|| anyhow::anyhow!("conv1d prefill element count overflow"))?;
                KernelLaunch::new(ctx.gpu, parallel_k)
                    .grid([conv_dim.div_ceil(256), k, 1])
                    .block([256, 1, 1])
                    .arg_ptr(ssm_state.conv_state)
                    .arg_ptr(deinterleaved)
                    .arg_ptr(self.ssm.conv1d.weight)
                    .arg_ptr(DevicePtr::NULL)
                    .arg_ptr(conv_f32)
                    .arg_u32(k)
                    .arg_u32(conv_dim)
                    .arg_u32(d_conv)
                    .arg_u32(qk_channels)
                    .arg_u32(kd)
                    .arg_f32(1e-6)
                    .arg_u32(qkvz_size)
                    .arg_u32(conv_dim)
                    .launch(stream)?;
                KernelLaunch::new(ctx.gpu, commit_k)
                    .grid([conv_dim.div_ceil(256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(ssm_state.conv_state)
                    .arg_ptr(deinterleaved)
                    .arg_u32(k)
                    .arg_u32(conv_dim)
                    .arg_u32(d_conv)
                    .arg_u32(qkvz_size)
                    .launch(stream)?;
                KernelLaunch::new(ctx.gpu, rne_k)
                    .grid([total_elems.div_ceil(256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(conv_f32)
                    .arg_ptr(conv_out_buf)
                    .arg_u32(total_elems)
                    .launch(stream)?;
                static LOGGED_PAR: std::sync::Once = std::sync::Once::new();
                LOGGED_PAR.call_once(|| {
                    tracing::info!(
                        "SSM_PREFILL_CONV_FAST_ENGAGED selector={PREFILL_CONV_FAST_SELECTOR} backend=conv1d_l2norm_f32_parallel_rne rows={k} dim={conv_dim}"
                    );
                });
                Ok(true)
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize, // SSM layers ignore — recurrent state requires all tokens
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if crate::layers::qwen4_prefill_moe::selected()? && num_tokens > 1 {
            return self.prefill_qwen4_moe_batch(
                hidden,
                residual,
                num_tokens,
                state,
                _seq_len_start,
                ctx,
                stream,
            );
        }
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;
        let fp32 = 4usize;

        // Qualification switch used to bisect Qwen4 recurrent prefill from
        // full-attention prefill without changing either implementation.
        // ATLAS_QWEN4_SSM_PREFILL_ROWWISE also leaves this per-token route: it
        // keeps the same exact recurrence but hoists the token-parallel work
        // (QKVZ GEMM, out-proj, MoE FFN) out of the per-token loop. See the
        // rowwise branch below.
        if self.qwen4_attn_hyper.is_some()
            && std::env::var("ATLAS_QWEN4_SSM_PREFILL_BATCH")
                .ok()
                .as_deref()
                != Some("1")
            && std::env::var("ATLAS_QWEN4_SSM_PREFILL_ROWWISE")
                .ok()
                .as_deref()
                != Some("1")
        {
            let row_bytes = ctx.config.residual_width() * 2;
            // K=3 stays on the exact hyper/QKVZ/MoE families already
            // qualified by speculative verification. K>=4 enters grouped
            // prefill math and remains explicit opt-in; set the tile to 1 for
            // the fully serialized diagnostic baseline.
            let default_tile = if crate::layers::qwen4_hybrid_rows(8) {
                8
            } else if crate::layers::qwen4_hybrid_rows(5) {
                5
            } else {
                1
            };
            let tile = std::env::var("ATLAS_QWEN4_SSM_PREFILL_TILE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| (1..=4096).contains(&v))
                .unwrap_or(default_tile);
            let staging_rows = ctx.buffers.sizes().gate_logits / (ctx.config.hidden_size * 2);
            anyhow::ensure!(
                staging_rows > 0,
                "Qwen4 recurrent prefill has no gate-logit staging rows"
            );
            let mut t = 0;
            while t < num_tokens {
                let width = (num_tokens - t).min(tile).min(staging_rows);
                if width == 1 {
                    self.decode_inner(
                        hidden.offset(t * row_bytes),
                        residual.offset(t * row_bytes),
                        state,
                        _kv_cache,
                        _seq_len_start + t,
                        _block_table,
                        _disk_block_ids,
                        _disk_last_offloaded_per_layer,
                        ctx,
                        stream,
                    )?;
                } else {
                    self.decode_qwen4_batched_inner(
                        hidden.offset(t * row_bytes),
                        residual.offset(t * row_bytes),
                        width,
                        state,
                        _kv_cache,
                        _seq_len_start + t,
                        _block_table,
                        _disk_block_ids,
                        _disk_last_offloaded_per_layer,
                        DevicePtr::NULL,
                        DevicePtr::NULL,
                        0,
                        0,
                        ctx,
                        stream,
                    )?;
                }
                t += width;
            }
            return Ok(());
        }

        let qwen4_hyper = match (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper) {
            (Some(attn), Some(mlp)) => Some((attn, mlp)),
            (None, None) => None,
            _ => anyhow::bail!("incomplete Qwen4 SSM hyperconnection pair"),
        };

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd; // 2048
        let value_dim = nv * vd; // Qwen3.8: 6144
        let conv_dim = key_dim * 2 + value_dim; // Qwen3.8: 10240
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size(); // Qwen3.8: 16384
        let normed = ctx.buffers.norm_output();
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        let out_proj_buf = ctx.buffers.moe_output();
        let gdn_c143 = self.preflight_gdn_c143_prefill(
            ctx,
            num_tokens,
            nk,
            nv,
            kd,
            vd,
            conv_dim,
            nv * 2,
            qkvz_size,
            ssm_state.h_state,
            ctx.buffers.ssm_qkvz(),
            ctx.buffers.ssm_gates(),
            ctx.buffers.attn_output(),
            stream,
        )?;
        let flashinfer_ssm = self.preflight_flashinfer_ssm_prefill(
            ctx,
            num_tokens,
            h,
            qkvz_size,
            value_dim,
            normed,
            proj_dst,
            ctx.buffers.ssm_qkvz(),
            out_proj_buf,
            stream,
        )?;

        // Profiling helper: sync + timestamp when ATLAS_PROFILE=1
        macro_rules! prof {
            ($label:expr, $t0:expr) => {
                if ctx.profile {
                    if let Some(t0) = $t0 {
                        ctx.gpu.synchronize(stream)?;
                        let elapsed = t0.elapsed().as_micros();
                        tracing::info!("  SSM prefill [{}] N={}: {}µs", $label, k, elapsed);
                    }
                }
            };
        }
        let mut t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 1. Input preparation for N tokens ──
        let hyper_gemm = std::env::var("ATLAS_QWEN4_HYPER_PREFILL_GEMM")
            .ok()
            .as_deref()
            == Some("1");
        let normed = if let Some((attn_hyper, _)) = qwen4_hyper {
            if hyper_gemm {
                attn_hyper.prepare_prefill(
                    hidden,
                    residual,
                    num_tokens,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?
            } else {
                attn_hyper.prepare_prefill_exact(
                    hidden,
                    residual,
                    num_tokens,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?
            }
        } else {
            let normed = ctx.buffers.norm_output();
            let used_fused_absmax = if k >= 64
                && !ctx.config.use_fp32_residual()
                && ctx.config.model_type != "gemma4"
                && super::super::dense_ffn::prefill_fuse_absmax_enabled(ctx.gpu)
                && !ctx.gpu.stream_is_capturing(stream)
            {
                let absmax_k = super::super::try_kernel(ctx.gpu, "norm", "rms_norm_absmax");
                if absmax_k.0 != 0
                    && let Some(global_max) =
                        super::super::dense_ffn::prepare_precomputed_proj_absmax(
                            ctx.gpu, normed, k, h as u32, stream,
                        )?
                {
                    // Non-hyper SSM prefill never reads `residual`, so skip the
                    // dead 84 MB residual copy and fuse `nvfp4_global_absmax`
                    // directly into the norm pass.
                    ops::rms_norm_absmax(
                        ctx.gpu,
                        absmax_k,
                        hidden,
                        &self.input_norm,
                        normed,
                        global_max,
                        k,
                        h as u32,
                        eps,
                        stream,
                    )?;
                    true
                } else {
                    false
                }
            } else {
                false
            };
            if !used_fused_absmax {
                ops::rms_norm_residual(
                    ctx.gpu,
                    self.rms_norm_residual_k,
                    hidden,
                    &self.input_norm,
                    normed,
                    residual,
                    k,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
            normed
        };
        prof!("rms_norm_residual", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 2+3. QKVZ GEMM (+ deinterleave if needed) ──
        if flashinfer_ssm {
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            self.launch_flashinfer_ssm_qkvz(ctx, k, normed, proj_dst, stream)?;
            #[cfg(not(all(feature = "cuda", target_os = "linux")))]
            unreachable!("non-CUDA FlashInfer SSM route passed preflight");
        } else if let Some(fp8) = self.qkvz_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                normed,
                fp8,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm prefill: QKVZ FP8 GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else if let Some(ref nvfp4_t) = self.qkvz_nvfp4_t {
            if k > 128 {
                self.qkvz_prefill_m128_dispatch(
                    ctx,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "ssm prefill: QKVZ m128 GEMM failed (M={k}, N={qkvz_size}): {e}"
                    )
                })?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("ssm prefill: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
                })?;
            }
        } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                nvfp4,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm prefill: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                self.bf16_qkvz()?,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        }
        if !self.sequential_qkvz {
            ops::deinterleave_qkvz(
                ctx.gpu,
                self.deinterleave_k,
                proj_dst,
                deinterleaved,
                k,
                nk as u32,
                kd as u32,
                vpg as u32,
                vd as u32,
                stream,
            )?;
        }

        prof!("qkvz_gemm", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 4+5. Fused BA GEMM + GDN gates (token-parallel) ──
        // Replaces dense_gemm([M,K]×[N,K]) + compute_gdn_gates.
        // Vectorized uint4 loads, warp shuffle reduction, inline sigmoid/exp.
        // gate_out layout: [gate(nv), beta(nv)] per token, gate_stride = 2*nv FP32.
        let ba_size = ctx.config.ssm_ba_size(); // 64 (or 96 on Qwen3.8-27B)
        let gates_buf = ctx.buffers.ssm_gates();
        let gate_stride = nv * 2; // FP32 elements per token
        let ba_used_tc = if qwen4_hyper.is_none() {
            self.try_dense_ba_gates_gemm(normed, gates_buf, num_tokens, ctx, stream)?
        } else {
            false
        };
        if !ba_used_tc {
            ops::dense_gemm_ba_gates_prefill(
                ctx.gpu,
                self.ba_gates_prefill_k,
                normed,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gates_buf,
                k,
                ba_size as u32,
                h as u32,
                h as u32,
                gate_stride as u32,
                nv as u32,
                vpg as u32,
                stream,
            )?;
        }
        prof!("ba+gates", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 6. Batched conv1d for all N tokens (sequential per-channel in registers) ──
        // Reuse ssm_qkvz buffer for conv output (safe: deinterleave is done)
        let conv_out_buf = ctx.buffers.ssm_qkvz();
        let gdn_out_buf = ctx.buffers.attn_output();
        let normed_out_buf = conv_out_buf;
        let z_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
        let exact_f32_prefill = qwen4_hyper.is_some()
            && std::env::var("ATLAS_QWEN4_SSM_PREFILL_FP32")
                .ok()
                .as_deref()
                == Some("1")
            && self.conv1d_l2norm_f32_sequence_k.0 != 0
            && self.gdn_f32_sequence_nosnap_k.0 != 0
            && self.gated_rms_norm_f32_multi_seq_k.0 != 0;

        // Row-wise exact recurrence inside an otherwise fully batched prefill.
        //
        // Motivation: the two batched recurrent routes below (the WY/chunked
        // `gdn_prefill_*` kernels, and the FP32 sequence kernels behind
        // ATLAS_QWEN4_SSM_PREFILL_FP32) both produce incoherent output — and
        // they do so at a 23-token prompt, so the defect is wiring, not an
        // accumulation that only shows up on long prompts. The alternative that
        // *is* correct today is the per-token route at the top of this function,
        // but it pays for correctness by running the ENTIRE layer — norm, QKVZ
        // GEMM, out-proj and the MoE FFN — once per prompt token, which is why
        // measured prefill (~40 tok/s) is indistinguishable from decode.
        //
        // Only conv1d and the GDN update are actually sequential; everything
        // else in this function is already token-parallel. So keep the batched
        // work batched and step ONLY the recurrence row by row, using the exact
        // single-token kernels — the same three calls, in the same order, as the
        // row-by-row oracle in `trait_decode_batched.rs`, which is qualified by
        // speculative verification. Prefill has no rollback, so the per-row
        // state snapshots that oracle takes are omitted here.
        //
        // conv_f32 holds ONE row (reused each step); gdn_f32 sits in the tail of
        // that row, matching the decode oracle's layout exactly.
        let exact_rowwise_prefill = qwen4_hyper.is_some()
            && std::env::var("ATLAS_QWEN4_SSM_PREFILL_ROWWISE")
                .ok()
                .as_deref()
                == Some("1")
            && self.conv1d_l2norm_f32_k.0 != 0
            && self.gdn_f32_k.0 != 0
            && self.gated_rms_norm_f32_k.0 != 0;

        let conv_seq_rowwise = exact_rowwise_prefill
            && std::env::var("ATLAS_QWEN4_SSM_PREFILL_CONVSEQ")
                .ok()
                .as_deref()
                == Some("1")
            && self.conv1d_l2norm_f32_sequence_k.0 != 0
            && self.gated_rms_norm_f32_multi_seq_k.0 != 0;

        // ── Fail closed on the two unqualified batched recurrent routes ──
        //
        // Both produce FLUENT, PLAUSIBLE, WRONG output on Qwen4 — a 22-token
        // "What is 2+2?" returns unrelated prose rather than "4" (measured;
        // route receipts `batched_f32_sequence` and `batched_bf16_wy`). That is
        // the worst failure mode available: nothing errors, nothing looks
        // malformed, and the wrong answer silently becomes the context.
        //
        // The FP32 sequence route is not merely unlucky, it is used outside its
        // qualified window. `exact_flat_ssm_route` admits the very same
        // sequence kernels ONLY for `4 <= rows <= 32` (the speculative-verify
        // width band); prefill calls them with hundreds to thousands of rows,
        // where they have never been qualified. The bf16 WY route is
        // independently broken (F31/F32).
        //
        // One real defect was found and fixed on the way here — the conv1d
        // `qk_channels` argument was passing `nk * 2` (heads) instead of
        // `key_dim * 2` (channels), 32 instead of 4,096, so ~4,064 of the Q+K
        // channels skipped their L2 normalisation. Fixing it moved the failure
        // from empty output to coherent-but-off-topic, which proves it was real
        // and also proves it was not the only defect.
        //
        // Until the remaining defect is found, refuse rather than corrupt.
        // `ATLAS_QWEN4_SSM_PREFILL_ROWWISE=1` is the qualified fast path and is
        // named in the error so the operator has somewhere to go.
        if qwen4_hyper.is_some() && !exact_rowwise_prefill {
            let route = if exact_f32_prefill {
                "batched_f32_sequence"
            } else {
                "batched_bf16_wy"
            };
            // Only cite the row-window when it actually applies: this route is
            // measured wrong even at k=25, which is inside 4..=32, so claiming
            // the window as the cause there would be a false explanation.
            let window = if exact_f32_prefill && !(4..=32).contains(&num_tokens) {
                format!(
                    " Its FP32 sequence kernels are additionally used outside \
                     their qualified 4..=32 row window (see exact_flat_ssm_route); \
                     this call has {num_tokens} rows."
                )
            } else {
                String::new()
            };
            anyhow::bail!(
                "Qwen4 batched SSM prefill is unqualified and produces incorrect \
                 output (route={route}, k={k}): measured to answer a 22-token \
                 \"what is 2+2\" with unrelated prose.{window} Use \
                 ATLAS_QWEN4_SSM_PREFILL_ROWWISE=1 — batched everything except the \
                 recurrence, exact against the per-token oracle, and 2.6x faster — \
                 or unset ATLAS_QWEN4_SSM_PREFILL_BATCH/_FP32 for the per-token path."
            );
        }

        // ATLAS_QWEN4_SSM_PREFILL_CONVSEQ=1 was a bisect experiment that
        // localized the batched-prefill defect to the conv1d sequence kernel:
        // fail closed BEFORE printing the route engagement receipt so we never
        // claim a rejected diagnostic route engaged.
        if conv_seq_rowwise {
            anyhow::bail!(
                "ATLAS_QWEN4_SSM_PREFILL_CONVSEQ is a diagnostic and produces \
                 incorrect output (k={k}): the conv1d sequence kernel is the \
                 remaining defect — with it, a 22-token \"what is 2+2\" answers \
                 with unrelated prose, while the otherwise-identical per-token \
                 conv answers \"4\". It is also not worth fixing for speed alone: \
                 it removes two thirds of the kernel launches for ~3%. Use \
                 ATLAS_QWEN4_SSM_PREFILL_ROWWISE=1 without it."
            );
        }

        // One-shot receipt naming the recurrent branch actually taken. Twice in
        // this investigation a conclusion was drawn about a branch that was not
        // running (a silently-unqualified flag, and a mistyped kernel symbol),
        // so the branch now identifies itself rather than being inferred.
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static SAID: AtomicBool = AtomicBool::new(false);
            if !SAID.swap(true, Ordering::Relaxed) {
                let which = if exact_rowwise_prefill {
                    "rowwise_exact"
                } else {
                    "batched_bf16_wy"
                };
                tracing::info!(
                    "ATLAS_SSM_PREFILL_ROUTE ENGAGED route={which} k={k} \
                     conv_seq_k={} gdn_nosnap_k={} gnorm_multi_k={}",
                    self.conv1d_l2norm_f32_sequence_k.0 != 0,
                    self.gdn_f32_sequence_nosnap_k.0 != 0,
                    self.gated_rms_norm_f32_multi_seq_k.0 != 0,
                );
            }
        }

        // Input: deinterleaved [N, qkvz_size], output: conv_out [N, conv_dim]
        // Conv1d processes QKV channels (first conv_dim of each token's qkvz_size)
        if exact_rowwise_prefill {
            let conv_f32 = ctx.buffers.ssm_conv_out_f32();
            let gdn_f32 = conv_f32.offset(conv_dim * fp32);
            let qk_ch = (key_dim * 2) as u32;
            let gate_beta_stride = nv * 2 * fp32; // bytes per token
            for t in 0..num_tokens {
                let qkv_t = deinterleaved.offset(t * qkvz_size * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_f32_k,
                    ssm_state.conv_state,
                    qkv_t,
                    &self.ssm.conv1d,
                    conv_f32,
                    conv_dim as u32,
                    d_conv as u32,
                    1,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
                let gate_t = gates_buf.offset(t * gate_beta_stride);
                ops::gdn_decode(
                    ctx.gpu,
                    self.gdn_f32_k,
                    ssm_state.h_state,
                    conv_f32,
                    conv_f32.offset(key_dim * fp32),
                    conv_f32.offset(key_dim * 2 * fp32),
                    gate_t,
                    gate_t.offset(nv * fp32),
                    gdn_f32,
                    1,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    stream,
                )?;
                let z_t = qkv_t.offset((key_dim * 2 + value_dim) * bf16);
                ops::gated_rms_norm(
                    ctx.gpu,
                    self.gated_rms_norm_f32_k,
                    gdn_f32,
                    z_t,
                    &self.ssm.norm,
                    normed_out_buf.offset(t * value_dim * bf16),
                    nv as u32,
                    vd as u32,
                    vd as u32,
                    eps,
                    vd as u32,
                    stream,
                )?;
            }
            prof!("ssm_rowwise_exact", t0);
            t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
        } else {
            let l2_policy = super::gdn_l2_window::gdn_l2_window_policy()?;
            let l2_geom = super::gdn_l2_window::GdnL2WindowGeometry {
                num_key_heads: nk,
                key_head_dim: kd,
                num_value_heads: nv,
                value_head_dim: vd,
                d_conv,
            };
            let l2_plan =
                super::gdn_l2_window::plan_gdn_l2_windows(num_tokens, l2_geom, l2_policy)?;
            if gdn_c143.is_none() && l2_plan.is_windowed() {
                super::gdn_l2_window::log_gdn_l2_window_engaged(&l2_plan, "prefill_inner");
                let gb_stride = (nv * 2) as u32;
                let qkvz_row_bytes = qkvz_size * bf16;
                for win in l2_plan.windows() {
                    let win_k = u32::try_from(win.token_len).map_err(|_| {
                        anyhow::anyhow!("GDN L2 window token count {} exceeds u32", win.token_len)
                    })?;
                    let deinterleaved_w = deinterleaved.offset(win.token_start * qkvz_row_bytes);
                    let conv_out_w = conv_out_buf.offset(win.qkv_byte_offset);
                    let gates_w = gates_buf.offset(win.gate_beta_byte_offset);
                    let gdn_out_w = gdn_out_buf.offset(win.output_byte_offset);

                    let conv_fast_used = if qwen4_hyper.is_none() {
                        self.try_prefill_conv1d_l2norm_fast(
                            ssm_state,
                            deinterleaved_w,
                            conv_out_w,
                            win_k,
                            conv_dim as u32,
                            (key_dim * 2) as u32,
                            kd as u32,
                            d_conv as u32,
                            qkvz_size as u32,
                            ctx,
                            stream,
                        )?
                    } else {
                        false
                    };
                    if !conv_fast_used {
                        ops::conv1d_update_prefill(
                            ctx.gpu,
                            self.conv1d_prefill_k,
                            ssm_state.conv_state,
                            deinterleaved_w,
                            &self.ssm.conv1d,
                            DevicePtr::NULL,
                            conv_out_w,
                            conv_dim as u32,
                            d_conv as u32,
                            win_k,
                            qkvz_size as u32,
                            conv_dim as u32,
                            stream,
                        )?;
                        ops::l2_norm(
                            ctx.gpu,
                            self.l2_norm_k,
                            conv_out_w,
                            (nk * 2) as u32,
                            kd as u32,
                            1e-6,
                            win_k,
                            conv_dim as u32,
                            stream,
                        )?;
                    }
                    let q_ptr = conv_out_w;
                    let k_ptr = conv_out_w.offset(key_dim * bf16);
                    let v_ptr = conv_out_w.offset(key_dim * 2 * bf16);
                    let gate_ptr = gates_w;
                    let beta_ptr = gates_w.offset(nv * fp32);
                    self.launch_gdn_prefill_window(
                        ssm_state, conv_out_w, q_ptr, k_ptr, v_ptr, gate_ptr, beta_ptr, gdn_out_w,
                        win_k, nk, nv, kd, vd, conv_dim, value_dim, gb_stride, ctx, stream,
                    )?;
                }
                prof!("conv1d+gdn_l2_windowed", t0);
                t0 = if ctx.profile {
                    ctx.gpu.synchronize(stream)?;
                    Some(std::time::Instant::now())
                } else {
                    None
                };
            } else {
                let conv_fast_used = if qwen4_hyper.is_none() {
                    self.try_prefill_conv1d_l2norm_fast(
                        ssm_state,
                        deinterleaved,
                        conv_out_buf,
                        k,
                        conv_dim as u32,
                        (key_dim * 2) as u32,
                        kd as u32,
                        d_conv as u32,
                        qkvz_size as u32,
                        ctx,
                        stream,
                    )?
                } else {
                    false
                };
                if conv_fast_used {
                    prof!("conv1d+l2_norm_fast", t0);
                    t0 = if ctx.profile {
                        ctx.gpu.synchronize(stream)?;
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                } else {
                    ops::conv1d_update_prefill(
                        ctx.gpu,
                        self.conv1d_prefill_k,
                        ssm_state.conv_state,
                        deinterleaved,
                        &self.ssm.conv1d,
                        DevicePtr::NULL,
                        conv_out_buf,
                        conv_dim as u32,
                        d_conv as u32,
                        k,
                        qkvz_size as u32,
                        conv_dim as u32,
                        stream,
                    )?;
                    prof!("conv1d", t0);
                    t0 = if ctx.profile {
                        ctx.gpu.synchronize(stream)?;
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };

                    // ── 7. Batched L2 norm on Q,K for all N tokens ──
                    // Q,K are the first 2*key_dim elements of each token's conv_out.
                    // Stride between tokens in conv_out = conv_dim.
                    ops::l2_norm(
                        ctx.gpu,
                        self.l2_norm_k,
                        conv_out_buf,
                        (nk * 2) as u32,
                        kd as u32,
                        1e-6,
                        k,
                        conv_dim as u32,
                        stream,
                    )?;
                    prof!("l2_norm", t0);
                    t0 = if ctx.profile {
                        ctx.gpu.synchronize(stream)?;
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                }

                // ── 8. GDN prefill via WY4-persistent kernel ──
                // Processes 4 tokens per iteration with WY algebraic correction, keeping
                // H state in shared memory for the entire sequence. 4× fewer sequential
                // state multiplications vs single-token kernel, preventing precision
                // drift at long context (28K+). Falls back to single-token persistent,
                // then split4 for unsupported configurations.
                let q_ptr = conv_out_buf;
                let k_ptr = conv_out_buf.offset(key_dim * bf16);
                let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
                let gb_stride = (nv * 2) as u32;

                // GDN profile gate (env: ATLAS_GDN_PROFILE=1). Sync + Instant before kernel
                // launch, paired with sync + record after.
                let __gdn_prof = {
                    use std::sync::OnceLock;
                    static C: OnceLock<bool> = OnceLock::new();
                    *C.get_or_init(|| {
                        std::env::var("ATLAS_GDN_PROFILE").ok().as_deref() == Some("1")
                    })
                };
                let __gdn_t = if __gdn_prof {
                    ctx.gpu.synchronize(stream)?;
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let __gdn_path: &'static str;

                // Ported from the pre-merge dispatch: the explicitly gated WY32
                // gate-cache shadow is byte-identical to the parent and uses the same
                // selector and dynamic-smem sizing as the two-phase path. Resolve it
                // before the default tuned route so an eligible explicit request
                // cannot silently fall back to the parent. Both this and the c143_v3
                // route below default OFF, so the default path is unchanged; they are
                // ordered ahead of `wy32_warp` (also default OFF) only to give the
                // explicitly requested kernel precedence when several are enabled.
                let gatecache = super::trait_prefill_gdn::use_wy32_gatecache(
                    crate::layers::gdn_prefill_gatecache_enabled(),
                    self.gdn_prefill_wy32_gatecache_k.0 != 0,
                    k,
                    kd,
                    vd,
                )
                .map_err(anyhow::Error::msg)?;
                if let Some(gdn_c143) = gdn_c143.as_ref() {
                    __gdn_path = "c143_v3";
                    gdn_c143.launch(ctx.gpu)?;
                } else if super::qwen4_gdn_fi::try_dense_gdn_fi_prefill(
                    conv_out_buf,
                    gates_buf,
                    gdn_out_buf,
                    ssm_state.h_state,
                    ssm_state.h_is_f16,
                    k,
                    nk,
                    nv,
                    kd,
                    vd,
                    conv_dim,
                    stream,
                )? {
                    __gdn_path = "fi_bf16";
                } else if gatecache {
                    __gdn_path = "wy32_gatecache";
                    let smem_v1 = super::trait_prefill_gdn::wy32_dynamic_smem_bytes(kd, vd, true)
                        .ok_or_else(|| {
                        anyhow::anyhow!("WY32 gate-cache shared-memory size overflow")
                    })?;
                    let smem_v2 =
                        super::trait_prefill_gdn::wy32_gatecache_v2_dynamic_smem_bytes(kd, vd)
                            .ok_or_else(|| {
                                anyhow::anyhow!("WY32 gate-cache v2 shared-memory size overflow")
                            })?;
                    let (gatecache_k, smem, _) = super::trait_prefill_gdn::gatecache_kernel_choice(
                        crate::layers::gdn_prefill_gatecache_v2_enabled(),
                        self.gdn_prefill_wy32_gatecache_k,
                        self.gdn_prefill_wy32_gatecache_v2_k,
                        smem_v1,
                        smem_v2,
                    )
                    .map_err(anyhow::Error::msg)?;
                    super::trait_prefill_gdn::log_wy32_gatecache_engaged(k);
                    ops::gdn_prefill_persistent_smem(
                        ctx.gpu,
                        gatecache_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gates_buf,
                        gates_buf.offset(nv * fp32),
                        gdn_out_buf,
                        1,
                        k,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        gb_stride,
                        smem,
                        stream,
                    )?;
                } else if std::env::var("ATLAS_GDN_WY32_WARP").ok().as_deref() == Some("1")
                    && self.gdn_prefill_wy32_warp_k.0 != 0
                    && k > 32
                {
                    __gdn_path = "wy32_warp";
                    let smem_bytes = kd * vd * 4
                        + 32 * kd * 2
                        + 32 * kd * 2
                        + 4 * 4
                        + 32 * 32 * 4
                        + 32 * 4
                        + 32 * 4;
                    let smem = (smem_bytes.div_ceil(256) * 256) as u32;
                    ops::gdn_prefill_persistent_smem(
                        ctx.gpu,
                        self.gdn_prefill_wy32_warp_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gates_buf,
                        gates_buf.offset(nv * fp32),
                        gdn_out_buf,
                        1,
                        k,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        gb_stride,
                        smem,
                        stream,
                    )?;
                // wy32 (32 tokens/WY iteration) is the default GDN prefill kernel
                // when available and no higher-priority route (`gdn_c143`,
                // `try_dense_gdn_fi_prefill`, `gatecache`, or `ATLAS_GDN_WY32_WARP`)
                // took precedence — matching the two-phase `prefill_gdn_full_inner`
                // ladder in `trait_prefill_gdn.rs`. Opt out with
                // `ATLAS_GDN_PREFILL_TUNED=0` to fall through to `wy4`.
                } else if std::env::var("ATLAS_GDN_PREFILL_TUNED").ok().as_deref() != Some("0")
                    && self.gdn_prefill_wy32_k.0 != 0
                    && k > 32
                {
                    __gdn_path = "wy32_tuned";
                    // SMEM layout per kernel:
                    //   H[kd*vd]FP32 + smem_k[32*kd]BF16 + smem_q[32*kd]BF16
                    //   + smem_warp[4]FP32 + smem_kd[32*32]FP32
                    //   + smem_g[32]FP32 + smem_bt[32]FP32
                    // Round up to 256 B for alignment.
                    let smem_bytes = kd * vd * 4
                        + 32 * kd * 2
                        + 32 * kd * 2
                        + 4 * 4
                        + 32 * 32 * 4
                        + 32 * 4
                        + 32 * 4;
                    let smem = (smem_bytes.div_ceil(256) * 256) as u32;
                    ops::gdn_prefill_persistent_smem(
                        ctx.gpu,
                        self.gdn_prefill_wy32_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gates_buf,
                        gates_buf.offset(nv * fp32),
                        gdn_out_buf,
                        1,
                        k,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        gb_stride,
                        smem,
                        stream,
                    )?;
                } else if self.gdn_prefill_persistent_wy4_k.0 != 0 {
                    __gdn_path = "wy4";
                    // WY4-persistent: H in shared memory, 4 tokens per iteration
                    // smem = H[K_DIM*V_DIM] + 8*k/q buffers + warp sums + WY scalars
                    let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
                    ops::gdn_prefill_persistent_smem(
                        ctx.gpu,
                        self.gdn_prefill_persistent_wy4_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gates_buf,
                        gates_buf.offset(nv * fp32),
                        gdn_out_buf,
                        1,
                        k,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        gb_stride,
                        smem,
                        stream,
                    )?;
                } else if (256..=4096).contains(&k) && self.gdn_prefill_persistent_k.0 != 0 {
                    __gdn_path = "persistent";
                    ops::gdn_prefill_persistent(
                        ctx.gpu,
                        self.gdn_prefill_persistent_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gates_buf,
                        gates_buf.offset(nv * fp32),
                        gdn_out_buf,
                        1,
                        k,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        gb_stride,
                        stream,
                    )?;
                } else {
                    __gdn_path = "split4";
                    ops::gdn_prefill_split4(
                        ctx.gpu,
                        self.gdn_prefill_split4_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gates_buf,
                        gates_buf.offset(nv * fp32),
                        gdn_out_buf,
                        1,
                        k,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        gb_stride,
                        stream,
                    )?;
                }

                if let Some(__t) = __gdn_t {
                    ctx.gpu.synchronize(stream)?;
                    let __ns = __t.elapsed().as_nanos() as u64;
                    tracing::info!(
                        "GDN_PROF call total={k} us={} path={}",
                        __ns / 1000,
                        __gdn_path
                    );
                }

                prof!("gdn_prefill", t0);
                t0 = if ctx.profile {
                    ctx.gpu.synchronize(stream)?;
                    Some(std::time::Instant::now())
                } else {
                    None
                };
            }

            // ── 9. Gated RMS norm (batched: all tokens × heads in one launch) ──
            ops::gated_rms_norm_prefill(
                ctx.gpu,
                self.gated_rms_norm_prefill_k,
                gdn_out_buf,
                z_base,
                &self.ssm.norm,
                normed_out_buf,
                nv as u32,
                vd as u32,
                eps,
                k,
                value_dim as u32,
                qkvz_size as u32,
                stream,
            )?;
            prof!("gated_rms_norm", t0);
            t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
        }

        // ── 10. Output projection GEMM: Qwen3.8 [M,6144] × [5120,6144]ᵀ ──
        if flashinfer_ssm {
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            self.launch_flashinfer_ssm_output(ctx, k, normed_out_buf, out_proj_buf, stream)?;
            #[cfg(not(all(feature = "cuda", target_os = "linux")))]
            unreachable!("non-CUDA FlashInfer SSM route passed preflight");
        } else {
            self.prefill_out_proj_dispatch(
                ctx,
                normed_out_buf,
                out_proj_buf,
                k,
                h,
                value_dim,
                stream,
            )?;
        }
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(out_proj_buf.0, num_tokens * h * bf16, stream)?;
        }

        prof!("out_proj", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 11. Batched residual/hyper injection + MoE ──
        if let Some((attn_hyper, mlp_hyper)) = qwen4_hyper {
            attn_hyper.inject_saved_batched(
                hidden,
                out_proj_buf,
                residual,
                num_tokens,
                ctx.gpu,
                stream,
            )?;
            let ffn_inputs = if hyper_gemm {
                mlp_hyper.prepare_prefill(
                    hidden,
                    residual,
                    num_tokens,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?
            } else {
                mlp_hyper.prepare_prefill_exact(
                    hidden,
                    residual,
                    num_tokens,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?
            };
            self.ffn
                .forward_prefill(ffn_inputs, num_tokens, ctx, stream)?;
            mlp_hyper.inject_saved_batched(
                hidden,
                ctx.buffers.moe_output(),
                residual,
                num_tokens,
                ctx.gpu,
                stream,
            )?;
        } else {
            let fused_ffn = match &self.ffn {
                crate::layers::FfnComponent::Dense(dense_ffn) => dense_ffn
                    .try_forward_prefill_fused_residual_norm(
                        hidden,
                        out_proj_buf,
                        &self.post_attn_norm,
                        num_tokens,
                        eps,
                        ctx,
                        stream,
                    )?,
                _ => false,
            };
            if !fused_ffn {
                ops::residual_add_rms_norm(
                    ctx.gpu,
                    self.residual_add_rms_norm_k,
                    hidden,
                    out_proj_buf,
                    &self.post_attn_norm,
                    ctx.buffers.norm_output(),
                    residual,
                    num_tokens as u32,
                    h as u32,
                    eps,
                    stream,
                )?;
                self.ffn
                    .forward_prefill(ctx.buffers.norm_output(), num_tokens, ctx, stream)?;
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden,
                    ctx.buffers.moe_output(),
                    (num_tokens * h) as u32,
                    stream,
                )?;
            }
        }

        prof!("moe_ffn", t0);

        if qwen4_hyper.is_some() {
            crate::model::qwen4_prefill_engagement::engage(
                crate::model::qwen4_prefill_engagement::PrefillPath::Ssm,
                num_tokens,
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PrefillConvFastRoute, resolve_prefill_conv_fast, resolve_prefill_conv_fast_with_gb10,
        select_prefill_conv_fast_route,
    };

    #[test]
    fn prefill_conv_fast_auto_enables_above_64_sms_and_honors_opt_out() {
        assert!(!resolve_prefill_conv_fast(None, 48));
        assert!(!resolve_prefill_conv_fast(None, 64));
        assert!(resolve_prefill_conv_fast(Some("1"), 48));
        assert!(resolve_prefill_conv_fast(Some("true"), 48));

        for sm in [65u32, 110, 170, 188] {
            assert!(resolve_prefill_conv_fast(None, sm), "sm={sm}");
            assert!(!resolve_prefill_conv_fast(Some("0"), sm), "sm={sm}");
            assert!(!resolve_prefill_conv_fast(Some("false"), sm), "sm={sm}");
        }

        // ATLAS_GB10_FAST_PREFILL=1 enables conv fast on GB10 (<= 64 SMs) when
        // ATLAS_PREFILL_CONV_FAST is unset, while honoring explicit =0 opt-out.
        assert!(resolve_prefill_conv_fast_with_gb10(None, 48, true));
        assert!(resolve_prefill_conv_fast_with_gb10(None, 64, true));
        assert!(!resolve_prefill_conv_fast_with_gb10(Some("0"), 48, true));
        assert!(!resolve_prefill_conv_fast_with_gb10(
            Some("false"),
            48,
            true
        ));
        assert!(!resolve_prefill_conv_fast_with_gb10(None, 48, false));
    }

    #[test]
    fn prefill_conv_fast_selects_v2_commit_then_parallel_f32_for_qwen38_27b() {
        // Qwen3.8-27B geometry: conv_dim=10240, qk=4096, kd=128, d_conv=4, qkvz=16384.
        assert_eq!(
            select_prefill_conv_fast_route(
                true, 8192, 10240, 4096, 128, 4, 16384, None, true, true, true, true
            ),
            PrefillConvFastRoute::FusedV2Commit
        );
        // Standard qwen3.8-27b bundle without qwen4_gdn_fi_glue falls cleanly to common parallel+rne.
        assert_eq!(
            select_prefill_conv_fast_route(
                true, 8192, 10240, 4096, 128, 4, 16384, None, true, false, true, true
            ),
            PrefillConvFastRoute::ParallelF32ToBf16
        );
        // Short prompts (<4), disabled policy, or non-matching geometry keep legacy path.
        assert_eq!(
            select_prefill_conv_fast_route(
                true, 3, 10240, 4096, 128, 4, 16384, None, true, true, true, true
            ),
            PrefillConvFastRoute::Disabled
        );
        assert_eq!(
            select_prefill_conv_fast_route(
                false, 8192, 10240, 4096, 128, 4, 16384, None, true, true, true, true
            ),
            PrefillConvFastRoute::Disabled
        );
        assert_eq!(
            select_prefill_conv_fast_route(
                true, 8192, 10240, 4096, 64, 4, 16384, None, true, true, true, true
            ),
            PrefillConvFastRoute::Disabled
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_parity_fused_v2_commit_matches_parallel_f32_at_27b_geometry() -> anyhow::Result<()> {
        use super::super::qwen4_gdn_conv_v2 as v2;
        use half::bf16;
        use spark_runtime::gpu::{DevicePtr, GpuBackend};
        use spark_runtime::kernel_args::KernelLaunch;

        let Some(ptx) = atlas_kernels::ptx_for_model("qwen3.8-27b")
            .or_else(|| atlas_kernels::all_ptx_sets().into_iter().next())
        else {
            return Ok(());
        };
        let Ok(gpu) = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &ptx.modules) else {
            return Ok(());
        };
        let stream = gpu.default_stream();

        let v2_commit_k = super::super::super::try_kernel(
            &gpu,
            "qwen4_gdn_fi_glue",
            "qwen4_conv1d_l2norm_bf16_prefill_v2_commit",
        );
        let parallel_k = super::super::super::try_kernel(
            &gpu,
            "causal_conv1d_prefill_parallel",
            "causal_conv1d_update_l2norm_f32_prefill_parallel",
        );
        let commit_k = super::super::super::try_kernel(
            &gpu,
            "causal_conv1d_prefill_parallel",
            "causal_conv1d_prefill_state_commit",
        );
        let rne_k = super::super::super::try_kernel(&gpu, "quantize_nvfp4", "f32_to_bf16_rne");
        if v2_commit_k.0 == 0 || parallel_k.0 == 0 || commit_k.0 == 0 || rne_k.0 == 0 {
            return Ok(());
        }

        // Exact Qwen3.8-27B 8K prefill geometry:
        // nk=16, nv=48, kd=128, vd=128 -> qk_channels=4096, conv_dim=10240, qkvz_size=16384, d_conv=4, rows=8192.
        let rows = 8192u32;
        let conv_dim = 10240u32;
        let qk_channels = 4096u32;
        let kd = 128u32;
        let d_conv = 4u32;
        let qkvz_size = 16384u32;
        let eps = 1e-6f32;

        // `conv_state` is FP32 `[conv_dim, d_conv]`.
        let mut init_state_bytes = Vec::with_capacity((conv_dim * d_conv * 4) as usize);
        for i in 0..(conv_dim * d_conv) as usize {
            let v = (((i.wrapping_mul(2654435761) >> 20) & 0xFF) as f32 / 128.0 - 1.0) * 0.5;
            init_state_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut weight_bytes = Vec::with_capacity((conv_dim * d_conv * 2) as usize);
        for i in 0..(conv_dim * d_conv) as usize {
            let v = (((i.wrapping_mul(1315423911) >> 20) & 0xFF) as f32 / 128.0 - 1.0) * 0.3;
            weight_bytes.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
        }
        // Populate [rows, qkvz_size] BF16 (first `conv_dim` cols per row are QKV).
        let input_elems = (rows as usize) * (qkvz_size as usize);
        let mut input_u16 = vec![0u16; input_elems];
        for t in 0..(rows as usize) {
            let row_off = t * (qkvz_size as usize);
            for c in 0..(conv_dim as usize) {
                let h = t
                    .wrapping_mul(1315423911)
                    .wrapping_add(c.wrapping_mul(40503));
                let v = (((h >> 16) & 0xFF) as f32 / 128.0 - 1.0) * 0.5;
                input_u16[row_off + c] = bf16::from_f32(v).to_bits();
            }
        }
        let input_bytes: Vec<u8> = input_u16.iter().flat_map(|v| v.to_le_bytes()).collect();

        let state_ref = gpu.alloc(init_state_bytes.len())?;
        let state_v2 = gpu.alloc(init_state_bytes.len())?;
        let weight_dev = gpu.alloc(weight_bytes.len())?;
        let input_dev = gpu.alloc(input_bytes.len())?;
        let out_bytes = (rows as usize) * (conv_dim as usize) * 2;
        let out_f32_dev = gpu.alloc((rows as usize) * (conv_dim as usize) * 4)?;
        let out_ref_dev = gpu.alloc(out_bytes)?;
        let out_v2_dev = gpu.alloc(out_bytes)?;

        gpu.copy_h2d(&init_state_bytes, state_ref)?;
        gpu.copy_h2d(&init_state_bytes, state_v2)?;
        gpu.copy_h2d(&weight_bytes, weight_dev)?;
        gpu.copy_h2d(&input_bytes, input_dev)?;

        assert_eq!(
            v2::refusal(
                rows,
                conv_dim,
                qk_channels,
                kd,
                d_conv,
                qkvz_size,
                conv_dim,
                [input_dev.0, out_v2_dev.0, weight_dev.0],
            ),
            None,
            "Qwen3.8-27B geometry must satisfy FusedV2Commit alignment/shape contract"
        );

        // Reference: ParallelF32ToBf16
        let total_elems = rows * conv_dim;
        KernelLaunch::new(&gpu, parallel_k)
            .grid([conv_dim.div_ceil(256), rows, 1])
            .block([256, 1, 1])
            .arg_ptr(state_ref)
            .arg_ptr(input_dev)
            .arg_ptr(weight_dev)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(out_f32_dev)
            .arg_u32(rows)
            .arg_u32(conv_dim)
            .arg_u32(d_conv)
            .arg_u32(qk_channels)
            .arg_u32(kd)
            .arg_f32(eps)
            .arg_u32(qkvz_size)
            .arg_u32(conv_dim)
            .launch(stream)?;
        KernelLaunch::new(&gpu, commit_k)
            .grid([conv_dim.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(state_ref)
            .arg_ptr(input_dev)
            .arg_u32(rows)
            .arg_u32(conv_dim)
            .arg_u32(d_conv)
            .arg_u32(qkvz_size)
            .launch(stream)?;
        KernelLaunch::new(&gpu, rne_k)
            .grid([total_elems.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(out_f32_dev)
            .arg_ptr(out_ref_dev)
            .arg_u32(total_elems)
            .launch(stream)?;

        // Candidate: FusedV2Commit
        KernelLaunch::new(&gpu, v2_commit_k)
            .grid(v2::grid(conv_dim, rows))
            .block([v2::BLOCK_THREADS, 1, 1])
            .arg_ptr(state_v2)
            .arg_ptr(input_dev)
            .arg_ptr(weight_dev)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(out_v2_dev)
            .arg_u32(rows)
            .arg_u32(conv_dim)
            .arg_u32(d_conv)
            .arg_u32(qk_channels)
            .arg_u32(kd)
            .arg_f32(eps)
            .arg_u32(qkvz_size)
            .arg_u32(conv_dim)
            .launch(stream)?;
        gpu.synchronize(stream)?;

        // Verify committed conv_state is bit-for-bit identical.
        let mut st_ref = vec![0u8; init_state_bytes.len()];
        let mut st_v2 = vec![0u8; init_state_bytes.len()];
        gpu.copy_d2h(state_ref, &mut st_ref)?;
        gpu.copy_d2h(state_v2, &mut st_v2)?;
        assert_eq!(
            st_ref, st_v2,
            "FusedV2Commit committed conv_state mismatch at 27B geometry"
        );

        // Verify conv+l2norm BF16 output matches ParallelF32ToBf16 (<= 1 BF16 ULP).
        let mut o_ref = vec![0u8; out_bytes];
        let mut o_v2 = vec![0u8; out_bytes];
        gpu.copy_d2h(out_ref_dev, &mut o_ref)?;
        gpu.copy_d2h(out_v2_dev, &mut o_v2)?;
        let mut max_abs_err = 0.0f32;
        let mut max_ulp = 0u16;
        for (a, b) in o_ref.chunks_exact(2).zip(o_v2.chunks_exact(2)) {
            let ua = u16::from_le_bytes([a[0], a[1]]);
            let ub = u16::from_le_bytes([b[0], b[1]]);
            let fa = bf16::from_bits(ua).to_f32();
            let fb = bf16::from_bits(ub).to_f32();
            assert!(fb.is_finite(), "FusedV2Commit emitted non-finite output");
            max_abs_err = max_abs_err.max((fa - fb).abs());
            if (ua ^ ub) & 0x8000 == 0 {
                max_ulp = max_ulp.max(ua.abs_diff(ub));
            }
        }
        assert!(
            max_ulp <= 1 && max_abs_err <= 2e-2,
            "FusedV2Commit parity failed at 27B 8K geometry: max_ulp={max_ulp}, max_abs_err={max_abs_err}"
        );
        eprintln!(
            "PARITY OK FusedV2Commit 27B 8K (rows={rows}, conv_dim={conv_dim}): state=0-ULP-exact, max_ulp={max_ulp}, max_abs_err={max_abs_err:.6}"
        );

        for ptr in [
            out_v2_dev,
            out_ref_dev,
            out_f32_dev,
            input_dev,
            weight_dev,
            state_v2,
            state_ref,
        ] {
            let _ = gpu.free(ptr);
        }
        Ok(())
    }
}
