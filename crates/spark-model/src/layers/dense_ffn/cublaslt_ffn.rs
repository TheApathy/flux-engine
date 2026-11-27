// SPDX-License-Identifier: AGPL-3.0-only

//! cuBLASLt NVFP4 prefill, K-gamma, and fused residual-norm dense FFN execution paths.

use anyhow::Result;
#[cfg(feature = "cuda")]
use anyhow::{Context, ensure};
use spark_runtime::gpu::DevicePtr;
#[cfg(feature = "cuda")]
use spark_runtime::gpu::GpuBackend;

#[cfg(feature = "cuda")]
use super::cublaslt_proj::{
    CUBLASLT_DYNAMIC_SCALARS_BYTES, CublasLtAlphaSlot, CublasLtInputScale, CublasLtNvfp4Scales,
    DynamicAbsmaxState, clear_precomputed_proj_absmax, ensure_decode_ffn_e2m1_scratch,
    ensure_shared_merged_gate_up_out, launch_quantize_bf16_to_nvfp4_128x4_zero_sync,
    nvfp4_weight_ready, resolve_cublaslt_scalars, upload_swizzled_merged_gate_up_scales_128x4,
    upload_swizzled_nvfp4_scales_128x4,
};
use super::layer::DenseFfnLayer;
#[cfg(feature = "cuda")]
use super::layer::FfnActivation;
#[cfg(feature = "cuda")]
use super::policy::{
    CublasLtMergedGateUpPolicy, FlashInferFfnAvailability, cublaslt_fused_silu_quantize_scale,
    cublaslt_merged_gate_up_allow_materialize, cublaslt_merged_gate_up_policy,
    cublaslt_merged_gate_up_vram_ok_for_sm, cublaslt_nvfp4_route_enabled,
    ffn_prefill_fused_norm_enabled, ffn_prefill_l2_strip_rows, gpu_sm_count,
    prefill_ffn_cublaslt_nvfp4_policy, select_cublaslt_quantize_128x4_kernel,
    select_merged_silu_quant_kernel,
};
use crate::layer::ForwardContext;
#[cfg(feature = "cuda")]
use crate::layers::ops;
use crate::weight_map::DenseWeight;

impl DenseFfnLayer {
    /// Quantize a BF16 activation matrix `[m, k]` into row-major packed NVFP4
    /// (`[m, k/2]`) plus `128x4`-swizzled `UE4M3` block scales for cuBLASLt.
    ///
    /// Uses static checkpoint `input_scale` when present; otherwise runs the
    /// zero-sync device dynamic-absmax chain (`nvfp4_global_absmax` +
    /// `quantize_bf16_to_nvfp4_atlas_128x4_from_absmax`) paired with
    /// `BlockScaleScalars::device_alpha`, eliminating all host syncs even
    /// without static scales.
    #[cfg(feature = "cuda")]
    #[expect(
        clippy::too_many_arguments,
        reason = "activation quantizer forwarding scratch, dynamic scalars, shape, scale, and stream"
    )]
    fn prepare_e2m1_input_128x4(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        dynamic_scalars: DevicePtr,
        m: u32,
        k: u32,
        checkpoint_scale2: Option<f32>,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr, CublasLtInputScale)> {
        let (a_packed, a_scale, a_max) =
            self.ensure_e2m1_scratch(ctx.gpu, m as usize, k as usize)?;
        let quant_k = select_cublaslt_quantize_128x4_kernel(
            self.nvfp4_quantize_128x4_ptx_k,
            self.nvfp4_quantize_128x4_k,
        );
        let input_scale = launch_quantize_bf16_to_nvfp4_128x4_zero_sync(
            ctx.gpu,
            quant_k,
            self.nvfp4_absmax_k,
            self.nvfp4_quantize_128x4_from_absmax_k,
            self.nvfp4_dynamic_combined_alpha_k,
            input,
            a_packed,
            a_scale,
            dynamic_scalars,
            a_max,
            m,
            k,
            checkpoint_scale2,
            DynamicAbsmaxState::NeedsScan,
            stream,
        )?;
        Ok((a_packed, a_scale, input_scale))
    }

    /// Lazily swizzle and cache `128x4` `UE4M3` block scales for `gate_proj`,
    /// `up_proj`, and `down_proj`, plus a 32-byte zero-sync dynamic scalar
    /// buffer and (when permitted without unsafe VRAM duplication) a merged
    /// `[gate; up]` operand.
    #[cfg(feature = "cuda")]
    pub(super) fn ensure_cublaslt_nvfp4_scales(
        &self,
        gpu: &dyn GpuBackend,
        h: u32,
        inter: u32,
        want_merged_gate_up: bool,
        stream: u64,
    ) -> Result<(
        DevicePtr,
        DevicePtr,
        DevicePtr,
        DevicePtr,
        Option<(DevicePtr, DevicePtr)>,
    )> {
        let mut guard = self
            .cublaslt_nvfp4_scales
            .lock()
            .expect("cublaslt_nvfp4_scales lock poisoned");
        let gate_src = self.weights.gate_proj.weight_scale;
        let up_src = self.weights.up_proj.weight_scale;
        let down_src = self.weights.down_proj.weight_scale;
        if let Some(ref cached) = *guard
            && cached.gate_src == gate_src
            && cached.up_src == up_src
            && cached.down_src == down_src
        {
            let merged = cached.merged_weight.zip(cached.merged_scales);
            return Ok((
                cached.gate,
                cached.up,
                cached.down,
                cached.dynamic_scalars,
                merged,
            ));
        }

        let gate_packed_bytes = (inter as usize)
            .checked_mul((h as usize) / 2)
            .context("cuBLASLt NVFP4 gate packed byte length overflow")?;
        let merged_packed_bytes = gate_packed_bytes
            .checked_mul(2)
            .context("cuBLASLt NVFP4 merged gate/up byte length overflow")?;
        let policy = cublaslt_merged_gate_up_policy();
        let can_try_merged = want_merged_gate_up && policy != CublasLtMergedGateUpPolicy::Disabled;

        // 1. Resolve zero-copy merged [gate; up] weight first (FlashInfer merged
        // operand or already-contiguous gate+up allocations: 0 extra VRAM).
        let mut merged_weight: Option<DevicePtr> = None;
        let mut owns_merged_weight = false;
        let mut borrowed_merged_scales: Option<DevicePtr> = None;
        if can_try_merged {
            #[cfg(target_os = "linux")]
            if let Some(ref fi) = self.flashinfer_ffn_prefill
                && fi.merged_gate_up.source_gate_weight == self.weights.gate_proj.weight
                && fi.merged_gate_up.source_up_weight == self.weights.up_proj.weight
            {
                merged_weight = Some(fi.merged_gate_up.weight);
                borrowed_merged_scales = Some(fi.merged_gate_up.weight_scales_128x4);
            }
            if merged_weight.is_none()
                && self
                    .weights
                    .gate_proj
                    .weight
                    .0
                    .checked_add(gate_packed_bytes as u64)
                    == Some(self.weights.up_proj.weight.0)
            {
                merged_weight = Some(self.weights.gate_proj.weight);
            }
            // Materialize a separate merged weight copy when policy allows
            // (`Auto` on discrete `sm_count > 64` or explicit `Enabled`) AND
            // VRAM headroom meets the platform reserve (`4 GiB` discrete / `12 GiB` UMA).
            let sm_count = gpu_sm_count(gpu);
            if merged_weight.is_none()
                && cublaslt_merged_gate_up_allow_materialize(policy, sm_count)
                && cublaslt_merged_gate_up_vram_ok_for_sm(
                    gpu.free_memory().ok(),
                    merged_packed_bytes,
                    sm_count,
                )
            {
                let d_merged = gpu
                    .alloc(merged_packed_bytes)
                    .context("alloc cuBLASLt NVFP4 merged gate/up weight")?;
                let copy_res = (|| -> Result<()> {
                    gpu.copy_d2d_async(
                        self.weights.gate_proj.weight,
                        d_merged,
                        gate_packed_bytes,
                        stream,
                    )?;
                    gpu.copy_d2d_async(
                        self.weights.up_proj.weight,
                        d_merged.offset(gate_packed_bytes),
                        gate_packed_bytes,
                        stream,
                    )?;
                    gpu.synchronize(stream)?;
                    Ok(())
                })();
                if let Err(err) = copy_res {
                    let _ = gpu.free(d_merged);
                    return Err(err);
                }
                merged_weight = Some(d_merged);
                owns_merged_weight = true;
            }
        }

        let (gate, up, merged_scales, owns_merged_scales) = if let Some(borrowed_ms) =
            borrowed_merged_scales
        {
            let gate = upload_swizzled_nvfp4_scales_128x4(gpu, &self.weights.gate_proj, inter, h)?;
            let up = match upload_swizzled_nvfp4_scales_128x4(gpu, &self.weights.up_proj, inter, h)
            {
                Ok(ptr) => ptr,
                Err(err) => {
                    let _ = gpu.free(gate);
                    return Err(err);
                }
            };
            (gate, up, Some(borrowed_ms), false)
        } else if merged_weight.is_some() && inter.is_multiple_of(128) {
            // When inter % 128 == 0 (e.g. Qwen3.8-27B inter=17408), the 128x4
            // swizzled scales for [gate; up] are the exact concatenation of
            // gate and up 128x4 tiles: ONE allocation serves both merged and
            // split views with ZERO extra scale VRAM.
            let merged_s = match upload_swizzled_merged_gate_up_scales_128x4(
                gpu,
                &self.weights.gate_proj,
                &self.weights.up_proj,
                inter,
                h,
            ) {
                Ok(ptr) => ptr,
                Err(err) => {
                    if owns_merged_weight && let Some(mw) = merged_weight {
                        let _ = gpu.free(mw);
                    }
                    return Err(err);
                }
            };
            let gate_scale_bytes = spark_runtime::cublaslt::block_scale_padded_bytes(
                inter,
                h,
                spark_runtime::cublaslt::NVFP4_BLOCK_SIZE,
            );
            (
                merged_s,
                merged_s.offset(gate_scale_bytes),
                Some(merged_s),
                true,
            )
        } else {
            let gate =
                match upload_swizzled_nvfp4_scales_128x4(gpu, &self.weights.gate_proj, inter, h) {
                    Ok(ptr) => ptr,
                    Err(err) => {
                        if owns_merged_weight && let Some(mw) = merged_weight {
                            let _ = gpu.free(mw);
                        }
                        return Err(err);
                    }
                };
            let up = match upload_swizzled_nvfp4_scales_128x4(gpu, &self.weights.up_proj, inter, h)
            {
                Ok(ptr) => ptr,
                Err(err) => {
                    let _ = gpu.free(gate);
                    if owns_merged_weight && let Some(mw) = merged_weight {
                        let _ = gpu.free(mw);
                    }
                    return Err(err);
                }
            };
            let merged_s = if merged_weight.is_some() {
                match upload_swizzled_merged_gate_up_scales_128x4(
                    gpu,
                    &self.weights.gate_proj,
                    &self.weights.up_proj,
                    inter,
                    h,
                ) {
                    Ok(ptr) => Some(ptr),
                    Err(err) => {
                        let _ = gpu.free(gate);
                        let _ = gpu.free(up);
                        if owns_merged_weight && let Some(mw) = merged_weight {
                            let _ = gpu.free(mw);
                        }
                        return Err(err);
                    }
                }
            } else {
                None
            };
            let owns_ms = merged_s.is_some();
            (gate, up, merged_s, owns_ms)
        };

        let cleanup_gate_up = |gpu: &dyn GpuBackend| {
            if owns_merged_scales && merged_scales == Some(gate) {
                let _ = gpu.free(gate);
            } else {
                let _ = gpu.free(gate);
                let _ = gpu.free(up);
                if owns_merged_scales && let Some(ms) = merged_scales {
                    let _ = gpu.free(ms);
                }
            }
            if owns_merged_weight && let Some(mw) = merged_weight {
                let _ = gpu.free(mw);
            }
        };

        let down = match upload_swizzled_nvfp4_scales_128x4(gpu, &self.weights.down_proj, h, inter)
        {
            Ok(ptr) => ptr,
            Err(err) => {
                cleanup_gate_up(gpu);
                return Err(err);
            }
        };
        let dynamic_scalars = match gpu.alloc(CUBLASLT_DYNAMIC_SCALARS_BYTES) {
            Ok(ptr) => ptr,
            Err(err) => {
                cleanup_gate_up(gpu);
                let _ = gpu.free(down);
                return Err(err).context("alloc cuBLASLt NVFP4 dynamic scalar scratch");
            }
        };

        if let Some(old) = guard.replace(CublasLtNvfp4Scales {
            gate,
            up,
            down,
            dynamic_scalars,
            merged_weight,
            owns_merged_weight,
            merged_scales,
            owns_merged_scales,
            gate_src,
            up_src,
            down_src,
        }) {
            if old.owns_merged_scales && old.merged_scales == Some(old.gate) {
                let _ = gpu.free(old.gate);
            } else {
                let _ = gpu.free(old.gate);
                let _ = gpu.free(old.up);
                if old.owns_merged_scales
                    && let Some(ms) = old.merged_scales
                {
                    let _ = gpu.free(ms);
                }
            }
            if old.owns_merged_weight
                && let Some(mw) = old.merged_weight
            {
                let _ = gpu.free(mw);
            }
            let _ = gpu.free(old.down);
            let _ = gpu.free(old.dynamic_scalars);
        }
        Ok((
            gate,
            up,
            down,
            dynamic_scalars,
            merged_weight.zip(merged_scales),
        ))
    }

    /// Native cuBLASLt block-scaled NVFP4 (`VEC16_UE4M3`) prefill route.
    ///
    /// Auto-enabled on Blackwell datacenter (`SM100`/`SM103`,
    /// `compute_capability == Some((10, 0) | (10, 3))`)
    /// and Blackwell desktop/workstation (`SM120`) when FlashInfer FFN is absent, or via
    /// `ATLAS_CUBLASLT_NVFP4=1`. Returns `Ok(false)` when disabled, geometrically
    /// ineligible, or missing the `128x4` activation quantizer so prefill falls back
    /// cleanly to `w4a16_gemm.cu`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_forward_cublaslt_nvfp4_prefill(
        &self,
        input: DevicePtr,
        m: u32,
        h: u32,
        inter: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let cc = ctx.gpu.compute_capability();
        #[cfg(target_os = "linux")]
        let flashinfer_ffn = if self.flashinfer_ffn_prefill.is_some() {
            FlashInferFfnAvailability::Prepared
        } else {
            FlashInferFfnAvailability::Absent
        };
        #[cfg(not(target_os = "linux"))]
        let flashinfer_ffn = FlashInferFfnAvailability::Absent;
        let checkpoint_scales = self
            .cublaslt_checkpoint_scales
            .or(self.e2m1_checkpoint_scales);
        let has_scale_source = checkpoint_scales.is_some() || self.nvfp4_absmax_k.0 != 0;
        let has_quantizer_128x4 = select_cublaslt_quantize_128x4_kernel(
            self.nvfp4_quantize_128x4_ptx_k,
            self.nvfp4_quantize_128x4_k,
        )
        .0 != 0;
        if !cublaslt_nvfp4_route_enabled(
            prefill_ffn_cublaslt_nvfp4_policy(),
            cc,
            flashinfer_ffn,
            m,
            h,
            inter,
            has_quantizer_128x4,
            has_scale_source,
        ) {
            return Ok(false);
        }

        let weights_ready = [
            &self.weights.gate_proj,
            &self.weights.up_proj,
            &self.weights.down_proj,
        ]
        .into_iter()
        .all(nvfp4_weight_ready);
        if !weights_ready {
            return Ok(false);
        }

        clear_precomputed_proj_absmax();
        let sm_count = gpu_sm_count(ctx.gpu);
        let merged_silu_k = select_merged_silu_quant_kernel(
            self.nvfp4_merged_silu_quantize_128x4_ptx_k,
            self.nvfp4_merged_silu_quantize_128x4_k,
            sm_count,
        );
        let down_scale = checkpoint_scales.map(|s| s.down);
        let merged_eligible = self.activation == FfnActivation::SiLU
            && self.weights.gate_proj.weight_scale_2.to_bits()
                == self.weights.up_proj.weight_scale_2.to_bits()
            && down_scale.is_some()
            && merged_silu_k.0 != 0;

        let l2_strip_opt = ffn_prefill_l2_strip_rows(ctx.gpu);
        let _scratch_use = self.lock_e2m1_use(stream)?;
        let (gate_s128x4, up_s128x4, down_s128x4, dynamic_scalars, merged_gate_up) =
            self.ensure_cublaslt_nvfp4_scales(ctx.gpu, h, inter, merged_eligible, stream)?;

        let l2_strip_active = l2_strip_opt.filter(|&strip| {
            m > strip && merged_eligible && merged_gate_up.is_some() && checkpoint_scales.is_some()
        });
        let scratch_rows = l2_strip_active.unwrap_or(m);
        // Pre-size activation scratch for `scratch_rows` × `max(h, inter)` so
        // down_proj never reallocates while gate/up remain in flight on `stream`.
        let _ = self.ensure_e2m1_scratch(ctx.gpu, scratch_rows as usize, h.max(inter) as usize)?;

        let merged_out_opt =
            if let (true, Some((merged_w, merged_s))) = (merged_eligible, merged_gate_up) {
                let proj_bytes = (scratch_rows as usize)
                    .checked_mul(inter as usize)
                    .and_then(|e| e.checked_mul(2))
                    .context("cuBLASLt merged gate/up projection byte overflow")?;
                let merged_bytes = proj_bytes
                    .checked_mul(2)
                    .context("cuBLASLt merged gate/up output byte overflow")?;
                let buf = if ctx.buffers.sizes().ffn_gate_up_bf16 >= merged_bytes {
                    Some(ctx.buffers.ffn_gate_up_bf16())
                } else if gate_out.0.checked_add(proj_bytes as u64) == Some(up_out.0)
                    && l2_strip_active.is_none()
                {
                    Some(gate_out)
                } else {
                    ensure_shared_merged_gate_up_out(ctx.gpu, merged_bytes)?
                };
                buf.map(|out_buf| (merged_w, merged_s, out_buf))
            } else {
                None
            };

        let zero_sync_dynamic = checkpoint_scales.is_none()
            && self.nvfp4_quantize_128x4_from_absmax_k.0 != 0
            && self.nvfp4_dynamic_combined_alpha_k.0 != 0;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                cc = ?cc,
                m,
                h,
                inter,
                static_scale = checkpoint_scales.is_some(),
                zero_sync_dynamic,
                merged_gate_up = merged_out_opt.is_some(),
                l2_strip_rows = ?l2_strip_active,
                fused_silu_quant = cublaslt_fused_silu_quantize_scale(
                    self.activation,
                    down_scale,
                    self.nvfp4_silu_quantize_128x4_k,
                )
                .is_some(),
                "ENGAGED cuBLASLt NVFP4 (VEC16_UE4M3) dense FFN prefill route"
            );
        });

        let output = ctx.buffers.moe_output();
        // L2-resident strip-mined FFN pass for SM120 (`M > strip`, static scales +
        // merged `[gate; up]`): keeps `merged_out` (`[strip, 2*inter]` BF16 = 71.3 MB
        // at `strip = 1024, inter = 17408`) hot inside SM120's 96 MiB L2 cache instead
        // of writing/reading 570 MB through DRAM. Handles non-1024 remainders cleanly
        // because each strip pads its `128x4` scales independently.
        if let (Some(strip), Some((merged_w, merged_s128x4, merged_out)), Some(scales)) =
            (l2_strip_active, merged_out_opt, checkpoint_scales)
        {
            let merged_n = inter
                .checked_mul(2)
                .context("cuBLASLt merged gate/up N overflow")?;
            let quant_k = select_cublaslt_quantize_128x4_kernel(
                self.nvfp4_quantize_128x4_ptx_k,
                self.nvfp4_quantize_128x4_k,
            );
            let merged_scalars = resolve_cublaslt_scalars(
                ctx.gpu,
                self.nvfp4_dynamic_combined_alpha_k,
                CublasLtInputScale::Static(scales.gate_up),
                self.weights.gate_proj.weight_scale_2,
                CublasLtAlphaSlot::Primary,
                stream,
            )?;
            let down_scalars = resolve_cublaslt_scalars(
                ctx.gpu,
                self.nvfp4_dynamic_combined_alpha_k,
                CublasLtInputScale::Static(scales.down),
                self.weights.down_proj.weight_scale_2,
                CublasLtAlphaSlot::Primary,
                stream,
            )?;
            let (a_packed, a_scale, _) =
                self.ensure_e2m1_scratch(ctx.gpu, strip as usize, h.max(inter) as usize)?;
            let row_bytes_bf16 = (h as usize) * 2;
            let mut row0 = 0u32;
            while row0 < m {
                let m_s = (m - row0).min(strip);
                let in_s = input.offset((row0 as usize) * row_bytes_bf16);
                let out_s = output.offset((row0 as usize) * row_bytes_bf16);
                ops::quantize_bf16_to_nvfp4_atlas_128x4(
                    ctx.gpu,
                    quant_k,
                    in_s,
                    a_packed,
                    a_scale,
                    scales.gate_up,
                    m_s,
                    h,
                    stream,
                )?;
                spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                    a_packed.0,
                    a_scale.0,
                    merged_w.0,
                    merged_s128x4.0,
                    merged_out.0,
                    m_s,
                    merged_n,
                    h,
                    merged_n,
                    merged_scalars,
                    stream,
                )?;
                if ops::fused_rmsnorm_nvfp4_policy()? == ops::FusedRmsNormNvfp4Policy::Enabled {
                    self.launch_fused_silu_nvfp4(
                        ctx.gpu,
                        merged_out,
                        merged_out.offset((inter as usize) * 2),
                        a_packed,
                        a_scale,
                        scales.down,
                        m_s,
                        inter,
                        ops::SwiGluInputLayout::MergedGateUp,
                        ops::Nvfp4ScaleLayout::Swizzled128x4,
                        stream,
                    )?;
                } else {
                    ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                        ctx.gpu,
                        merged_silu_k,
                        merged_out,
                        a_packed,
                        a_scale,
                        scales.down,
                        m_s,
                        inter,
                        stream,
                    )?;
                }
                spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                    a_packed.0,
                    a_scale.0,
                    self.weights.down_proj.weight.0,
                    down_s128x4.0,
                    out_s.0,
                    m_s,
                    h,
                    inter,
                    h,
                    down_scalars,
                    stream,
                )?;
                row0 += m_s;
            }
            return Ok(true);
        }

        // If L2 strip-mining was initially planned but fell back (e.g. VRAM-tight
        // merged buffer unavailable), grow scratch to full `m` × `max(h, inter)`
        // before launching gate/up so `down_proj` never reallocates mid-stream.
        if l2_strip_active.is_some() {
            let _ = self.ensure_e2m1_scratch(ctx.gpu, m as usize, h.max(inter) as usize)?;
        }

        let gate_up_scale = checkpoint_scales.map(|s| s.gate_up);
        let (a_packed, a_scale, gate_up_input_scale) = self.prepare_e2m1_input_128x4(
            ctx,
            input,
            dynamic_scalars,
            m,
            h,
            gate_up_scale,
            stream,
        )?;

        let (down_packed, down_scale_buf, down_input_scale) =
            if let (Some((merged_w, merged_s128x4, merged_out)), Some(scale2)) =
                (merged_out_opt, down_scale)
            {
                let merged_n = inter
                    .checked_mul(2)
                    .context("cuBLASLt merged gate/up N overflow")?;
                let merged_scalars = resolve_cublaslt_scalars(
                    ctx.gpu,
                    self.nvfp4_dynamic_combined_alpha_k,
                    gate_up_input_scale,
                    self.weights.gate_proj.weight_scale_2,
                    CublasLtAlphaSlot::Primary,
                    stream,
                )?;
                spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                    a_packed.0,
                    a_scale.0,
                    merged_w.0,
                    merged_s128x4.0,
                    merged_out.0,
                    m,
                    merged_n,
                    h,
                    merged_n,
                    merged_scalars,
                    stream,
                )?;
                let (down_a_packed, down_a_scale, _) =
                    self.ensure_e2m1_scratch(ctx.gpu, m as usize, inter as usize)?;
                if ops::fused_rmsnorm_nvfp4_policy()? == ops::FusedRmsNormNvfp4Policy::Enabled {
                    self.launch_fused_silu_nvfp4(
                        ctx.gpu,
                        merged_out,
                        merged_out.offset((inter as usize) * 2),
                        down_a_packed,
                        down_a_scale,
                        scale2,
                        m,
                        inter,
                        ops::SwiGluInputLayout::MergedGateUp,
                        ops::Nvfp4ScaleLayout::Swizzled128x4,
                        stream,
                    )?;
                } else {
                    ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                        ctx.gpu,
                        merged_silu_k,
                        merged_out,
                        down_a_packed,
                        down_a_scale,
                        scale2,
                        m,
                        inter,
                        stream,
                    )?;
                }
                (
                    down_a_packed,
                    down_a_scale,
                    CublasLtInputScale::Static(scale2),
                )
            } else {
                let gate_scalars = resolve_cublaslt_scalars(
                    ctx.gpu,
                    self.nvfp4_dynamic_combined_alpha_k,
                    gate_up_input_scale,
                    self.weights.gate_proj.weight_scale_2,
                    CublasLtAlphaSlot::Primary,
                    stream,
                )?;
                let up_scalars = resolve_cublaslt_scalars(
                    ctx.gpu,
                    self.nvfp4_dynamic_combined_alpha_k,
                    gate_up_input_scale,
                    self.weights.up_proj.weight_scale_2,
                    CublasLtAlphaSlot::Secondary,
                    stream,
                )?;
                spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                    a_packed.0,
                    a_scale.0,
                    self.weights.gate_proj.weight.0,
                    gate_s128x4.0,
                    gate_out.0,
                    m,
                    inter,
                    h,
                    inter,
                    gate_scalars,
                    stream,
                )?;
                spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                    a_packed.0,
                    a_scale.0,
                    self.weights.up_proj.weight.0,
                    up_s128x4.0,
                    up_out.0,
                    m,
                    inter,
                    h,
                    inter,
                    up_scalars,
                    stream,
                )?;

                if let Some(scale2) = cublaslt_fused_silu_quantize_scale(
                    self.activation,
                    down_scale,
                    self.nvfp4_silu_quantize_128x4_k,
                ) {
                    let (down_a_packed, down_a_scale, _) =
                        self.ensure_e2m1_scratch(ctx.gpu, m as usize, inter as usize)?;
                    ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(
                        ctx.gpu,
                        self.nvfp4_silu_quantize_128x4_k,
                        gate_out,
                        up_out,
                        down_a_packed,
                        down_a_scale,
                        scale2,
                        m,
                        inter,
                        stream,
                    )?;
                    (
                        down_a_packed,
                        down_a_scale,
                        CublasLtInputScale::Static(scale2),
                    )
                } else {
                    ops::silu_mul(
                        ctx.gpu,
                        self.act_mul,
                        gate_out,
                        up_out,
                        gate_out,
                        m * inter,
                        stream,
                    )?;
                    self.prepare_e2m1_input_128x4(
                        ctx,
                        gate_out,
                        dynamic_scalars,
                        m,
                        inter,
                        down_scale,
                        stream,
                    )?
                }
            };

        let down_scalars = resolve_cublaslt_scalars(
            ctx.gpu,
            self.nvfp4_dynamic_combined_alpha_k,
            down_input_scale,
            self.weights.down_proj.weight_scale_2,
            CublasLtAlphaSlot::Primary,
            stream,
        )?;
        spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
            down_packed.0,
            down_scale_buf.0,
            self.weights.down_proj.weight.0,
            down_s128x4.0,
            output.0,
            m,
            h,
            inter,
            h,
            down_scalars,
            stream,
        )?;

        Ok(true)
    }

    #[cfg(feature = "cuda")]
    #[expect(
        clippy::type_complexity,
        reason = "internal tuple returning cached cuBLASLt 128x4 scale pointers and optional merged gate/up buffers"
    )]
    fn cached_cublaslt_nvfp4_scales(
        &self,
    ) -> Option<(
        DevicePtr,
        DevicePtr,
        DevicePtr,
        Option<(DevicePtr, DevicePtr)>,
    )> {
        let guard = self
            .cublaslt_nvfp4_scales
            .lock()
            .expect("cublaslt_nvfp4_scales lock poisoned");
        let gate_src = self.weights.gate_proj.weight_scale;
        let up_src = self.weights.up_proj.weight_scale;
        let down_src = self.weights.down_proj.weight_scale;
        let cached = guard.as_ref()?;
        if cached.gate_src == gate_src && cached.up_src == up_src && cached.down_src == down_src {
            Some((
                cached.gate,
                cached.up,
                cached.down,
                cached.merged_weight.zip(cached.merged_scales),
            ))
        } else {
            None
        }
    }

    /// Multi-seq decode (`forward_kgamma`) cuBLASLt NVFP4 (`VEC16_UE4M3`) FFN
    /// route for `M >= 32` using a dedicated per-`(gpu, stream)` decode scratch
    /// (`ensure_decode_ffn_e2m1_scratch`) so it never collides with prefill's
    /// stream-bound `e2m1_scratch` and stays 100% CUDA-graph capture safe once
    /// warmed on step 1.
    #[cfg(feature = "cuda")]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors forward_kgamma FFN entry parameters plus scratch outputs and CUDA stream"
    )]
    pub(super) fn try_forward_cublaslt_nvfp4_kgamma(
        &self,
        input: DevicePtr,
        m: u32,
        h: u32,
        inter: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(scales) = self.cublaslt_checkpoint_scales else {
            return Ok(false);
        };
        let cc = ctx.gpu.compute_capability();
        #[cfg(target_os = "linux")]
        let flashinfer_ffn = if self.flashinfer_ffn_prefill.is_some() {
            FlashInferFfnAvailability::Prepared
        } else {
            FlashInferFfnAvailability::Absent
        };
        #[cfg(not(target_os = "linux"))]
        let flashinfer_ffn = FlashInferFfnAvailability::Absent;
        let has_quantizer_128x4 = select_cublaslt_quantize_128x4_kernel(
            self.nvfp4_quantize_128x4_ptx_k,
            self.nvfp4_quantize_128x4_k,
        )
        .0 != 0;
        if !cublaslt_nvfp4_route_enabled(
            prefill_ffn_cublaslt_nvfp4_policy(),
            cc,
            flashinfer_ffn,
            m,
            h,
            inter,
            has_quantizer_128x4,
            true,
        ) {
            return Ok(false);
        }
        let weights_ready = [
            &self.weights.gate_proj,
            &self.weights.up_proj,
            &self.weights.down_proj,
        ]
        .into_iter()
        .all(nvfp4_weight_ready);
        if !weights_ready {
            return Ok(false);
        }

        let sm_count = gpu_sm_count(ctx.gpu);
        let merged_silu_k = select_merged_silu_quant_kernel(
            self.nvfp4_merged_silu_quantize_128x4_ptx_k,
            self.nvfp4_merged_silu_quantize_128x4_k,
            sm_count,
        );
        let merged_eligible = self.activation == FfnActivation::SiLU
            && self.weights.gate_proj.weight_scale_2.to_bits()
                == self.weights.up_proj.weight_scale_2.to_bits()
            && merged_silu_k.0 != 0;

        let (gate_s128x4, up_s128x4, down_s128x4, merged_gate_up) = if ctx.graph_capture {
            let Some(cached) = self.cached_cublaslt_nvfp4_scales() else {
                return Ok(false);
            };
            cached
        } else {
            let (g, u, d, _, merged) =
                self.ensure_cublaslt_nvfp4_scales(ctx.gpu, h, inter, merged_eligible, stream)?;
            (g, u, d, merged)
        };

        let want_merged_bytes = if merged_eligible && merged_gate_up.is_some() {
            (m as usize)
                .checked_mul(inter as usize)
                .and_then(|e| e.checked_mul(4))
                .context("cuBLASLt kgamma merged gate/up output byte overflow")?
        } else {
            0
        };
        let Some((a_packed, a_scale, decode_merged_out, decode_merged_cap)) =
            ensure_decode_ffn_e2m1_scratch(
                ctx.gpu,
                stream,
                m as usize,
                h.max(inter) as usize,
                want_merged_bytes,
                !ctx.graph_capture,
            )?
        else {
            return Ok(false);
        };

        let merged_out_opt =
            if let (true, Some((merged_w, merged_s))) = (merged_eligible, merged_gate_up) {
                let proj_bytes = (m as usize)
                    .checked_mul(inter as usize)
                    .and_then(|e| e.checked_mul(2))
                    .context("cuBLASLt kgamma merged gate/up projection byte overflow")?;
                let merged_bytes = proj_bytes
                    .checked_mul(2)
                    .context("cuBLASLt kgamma merged gate/up output byte overflow")?;
                if ctx.buffers.sizes().ffn_gate_up_bf16 >= merged_bytes {
                    Some((merged_w, merged_s, ctx.buffers.ffn_gate_up_bf16()))
                } else if !decode_merged_out.is_null() && decode_merged_cap >= merged_bytes {
                    Some((merged_w, merged_s, decode_merged_out))
                } else if gate_out.0.checked_add(proj_bytes as u64) == Some(up_out.0) {
                    Some((merged_w, merged_s, gate_out))
                } else {
                    None
                }
            } else {
                None
            };

        let quant_k = select_cublaslt_quantize_128x4_kernel(
            self.nvfp4_quantize_128x4_ptx_k,
            self.nvfp4_quantize_128x4_k,
        );
        ops::quantize_bf16_to_nvfp4_atlas_128x4(
            ctx.gpu,
            quant_k,
            input,
            a_packed,
            a_scale,
            scales.gate_up,
            m,
            h,
            stream,
        )?;

        if let Some((merged_w, merged_s128x4, merged_out)) = merged_out_opt {
            let merged_n = inter
                .checked_mul(2)
                .context("cuBLASLt kgamma merged gate/up N overflow")?;
            let merged_scalars = resolve_cublaslt_scalars(
                ctx.gpu,
                self.nvfp4_dynamic_combined_alpha_k,
                CublasLtInputScale::Static(scales.gate_up),
                self.weights.gate_proj.weight_scale_2,
                CublasLtAlphaSlot::Primary,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                merged_w.0,
                merged_s128x4.0,
                merged_out.0,
                m,
                merged_n,
                h,
                merged_n,
                merged_scalars,
                stream,
            )?;
            ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                ctx.gpu,
                merged_silu_k,
                merged_out,
                a_packed,
                a_scale,
                scales.down,
                m,
                inter,
                stream,
            )?;
        } else {
            let gate_scalars = resolve_cublaslt_scalars(
                ctx.gpu,
                self.nvfp4_dynamic_combined_alpha_k,
                CublasLtInputScale::Static(scales.gate_up),
                self.weights.gate_proj.weight_scale_2,
                CublasLtAlphaSlot::Primary,
                stream,
            )?;
            let up_scalars = resolve_cublaslt_scalars(
                ctx.gpu,
                self.nvfp4_dynamic_combined_alpha_k,
                CublasLtInputScale::Static(scales.gate_up),
                self.weights.up_proj.weight_scale_2,
                CublasLtAlphaSlot::Secondary,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                self.weights.gate_proj.weight.0,
                gate_s128x4.0,
                gate_out.0,
                m,
                inter,
                h,
                inter,
                gate_scalars,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                self.weights.up_proj.weight.0,
                up_s128x4.0,
                up_out.0,
                m,
                inter,
                h,
                inter,
                up_scalars,
                stream,
            )?;
            if let Some(scale2) = cublaslt_fused_silu_quantize_scale(
                self.activation,
                Some(scales.down),
                self.nvfp4_silu_quantize_128x4_k,
            ) {
                ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(
                    ctx.gpu,
                    self.nvfp4_silu_quantize_128x4_k,
                    gate_out,
                    up_out,
                    a_packed,
                    a_scale,
                    scale2,
                    m,
                    inter,
                    stream,
                )?;
            } else {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    m * inter,
                    stream,
                )?;
                ops::quantize_bf16_to_nvfp4_atlas_128x4(
                    ctx.gpu,
                    quant_k,
                    gate_out,
                    a_packed,
                    a_scale,
                    scales.down,
                    m,
                    inter,
                    stream,
                )?;
            }
        }

        let output = ctx.buffers.moe_output();
        let down_scalars = resolve_cublaslt_scalars(
            ctx.gpu,
            self.nvfp4_dynamic_combined_alpha_k,
            CublasLtInputScale::Static(scales.down),
            self.weights.down_proj.weight_scale_2,
            CublasLtAlphaSlot::Primary,
            stream,
        )?;
        spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
            a_packed.0,
            a_scale.0,
            self.weights.down_proj.weight.0,
            down_s128x4.0,
            output.0,
            m,
            h,
            inter,
            h,
            down_scalars,
            stream,
        )?;

        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                m,
                h,
                inter,
                merged_gate_up = merged_out_opt.is_some(),
                graph_capture = ctx.graph_capture,
                "ENGAGED cuBLASLt NVFP4 (VEC16_UE4M3) multi-seq decode FFN route"
            );
        });
        Ok(true)
    }

    /// Fused prefill post-mixer block for single-GPU static-scale NVFP4 FFN
    /// (`sm_count > 64` auto-ON, `ATLAS_FFN_PREFILL_FUSED_NORM`):
    ///
    /// Replaces:
    /// 1. `residual_add_rms_norm(hidden += src -> norm_output, residual)`
    /// 2. `quantize_bf16_to_nvfp4_atlas_128x4(norm_output -> a_packed)`
    /// 3. `gate_up` + `silu_mul_quant` + `down_proj(-> moe_output)`
    /// 4. `bf16_residual_add(hidden += moe_output)`
    ///
    /// with a strip-mined L2-resident sequence per `strip = 1024` rows:
    /// - `residual_add_rms_norm_quantize_nvfp4_128x4(hidden_s += src_s, residual=NULL -> a_packed)`
    /// - `merged_gate_up(a_packed -> merged_out)` (`71.3 MB < 96 MiB L2`)
    /// - `quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4_ptx(merged_out -> a_packed)`
    /// - `down_proj(a_packed -> hidden_s, beta = 1.0)`
    ///
    /// Returns `Ok(true)` when executed, or `Ok(false)` to let the caller run
    /// the standard unfused sequence cleanly.
    #[cfg(feature = "cuda")]
    #[expect(
        clippy::too_many_arguments,
        reason = "fused layer-tail prefill block accepting hidden, residual src, norm weight, shape, context, and stream"
    )]
    pub(crate) fn try_forward_prefill_fused_residual_norm(
        &self,
        hidden: DevicePtr,
        src: DevicePtr,
        norm_weight: &DenseWeight,
        num_tokens: usize,
        eps: f32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if self.try_forward_prefill_fused_residual_rmsnorm_nvfp4(
            hidden,
            src,
            norm_weight,
            num_tokens,
            eps,
            ctx,
            stream,
        )? {
            return Ok(true);
        }
        let Ok(m) = u32::try_from(num_tokens) else {
            return Ok(false);
        };
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        if !ffn_prefill_fused_norm_enabled(ctx.gpu)
            || ctx.graph_capture
            || ctx.gpu.stream_is_capturing(stream)
            || ctx.config.use_fp32_residual()
            || ctx.config.model_type == "gemma4"
            || ctx.config.tp_world_size > 1
            || self.bf16_weights.is_some()
            || self.activation != FfnActivation::SiLU
            || self.nvfp4_residual_add_rms_norm_quant_128x4_k.0 == 0
            || hidden.is_null()
            || src.is_null()
            || norm_weight.weight.is_null()
            || !hidden.0.is_multiple_of(16)
            || !src.0.is_multiple_of(16)
            || !norm_weight.weight.0.is_multiple_of(16)
        {
            return Ok(false);
        }

        let cc = ctx.gpu.compute_capability();
        #[cfg(target_os = "linux")]
        let flashinfer_ffn = if self.flashinfer_ffn_prefill.is_some() {
            FlashInferFfnAvailability::Prepared
        } else {
            FlashInferFfnAvailability::Absent
        };
        #[cfg(not(target_os = "linux"))]
        let flashinfer_ffn = FlashInferFfnAvailability::Absent;
        let Some(scales) = self
            .cublaslt_checkpoint_scales
            .or(self.e2m1_checkpoint_scales)
        else {
            return Ok(false);
        };
        let sm_count = gpu_sm_count(ctx.gpu);
        let merged_silu_k = select_merged_silu_quant_kernel(
            self.nvfp4_merged_silu_quantize_128x4_ptx_k,
            self.nvfp4_merged_silu_quantize_128x4_k,
            sm_count,
        );
        if merged_silu_k.0 == 0
            || self.weights.gate_proj.weight_scale_2.to_bits()
                != self.weights.up_proj.weight_scale_2.to_bits()
        {
            return Ok(false);
        }
        if !cublaslt_nvfp4_route_enabled(
            prefill_ffn_cublaslt_nvfp4_policy(),
            cc,
            flashinfer_ffn,
            m,
            h,
            inter,
            true,
            true,
        ) {
            return Ok(false);
        }
        let weights_ready = [
            &self.weights.gate_proj,
            &self.weights.up_proj,
            &self.weights.down_proj,
        ]
        .into_iter()
        .all(nvfp4_weight_ready);
        if !weights_ready {
            return Ok(false);
        }

        let _scratch_use = self.lock_e2m1_use(stream)?;
        let (_gate_s, _up_s, down_s128x4, _dyn, merged_gate_up) =
            self.ensure_cublaslt_nvfp4_scales(ctx.gpu, h, inter, true, stream)?;
        let Some((merged_w, merged_s128x4)) = merged_gate_up else {
            return Ok(false);
        };

        let strip = ffn_prefill_l2_strip_rows(ctx.gpu).unwrap_or(m).min(m);
        let scratch_rows = strip.max(1);
        let proj_bytes = (scratch_rows as usize)
            .checked_mul(inter as usize)
            .and_then(|e| e.checked_mul(2))
            .context("fused prefill FFN merged projection byte overflow")?;
        let merged_bytes = proj_bytes
            .checked_mul(2)
            .context("fused prefill FFN merged output byte overflow")?;
        let merged_out = if ctx.buffers.sizes().ffn_gate_up_bf16 >= merged_bytes {
            ctx.buffers.ffn_gate_up_bf16()
        } else {
            let Some(ptr) = ensure_shared_merged_gate_up_out(ctx.gpu, merged_bytes)? else {
                return Ok(false);
            };
            ptr
        };

        let (a_packed, a_scale, _) =
            self.ensure_e2m1_scratch(ctx.gpu, scratch_rows as usize, h.max(inter) as usize)?;
        let merged_n = inter
            .checked_mul(2)
            .context("fused prefill FFN merged gate/up N overflow")?;
        let merged_alpha = scales.gate_up * self.weights.gate_proj.weight_scale_2;
        let down_alpha = scales.down * self.weights.down_proj.weight_scale_2;
        ensure!(
            merged_alpha.is_finite() && merged_alpha > 0.0,
            "fused prefill FFN invalid merged alpha {merged_alpha}"
        );
        ensure!(
            down_alpha.is_finite() && down_alpha > 0.0,
            "fused prefill FFN invalid down alpha {down_alpha}"
        );
        let merged_scalars = spark_runtime::cublaslt::BlockScaleScalars::host(merged_alpha, 0.0);
        let down_scalars = spark_runtime::cublaslt::BlockScaleScalars::host(down_alpha, 1.0);

        clear_precomputed_proj_absmax();
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                cc = ?cc,
                m,
                h,
                inter,
                strip_rows = strip,
                "ENGAGED fused residual_add_rms_norm_quantize_nvfp4_128x4 + L2-strip + beta=1.0 FFN prefill"
            );
        });

        let row_bytes_bf16 = (h as usize) * 2;
        let mut row0 = 0u32;
        while row0 < m {
            let m_s = (m - row0).min(strip);
            let h_s = hidden.offset((row0 as usize) * row_bytes_bf16);
            let s_s = src.offset((row0 as usize) * row_bytes_bf16);
            ops::residual_add_rms_norm_quantize_nvfp4_128x4(
                ctx.gpu,
                self.nvfp4_residual_add_rms_norm_quant_128x4_k,
                h_s,
                s_s,
                norm_weight,
                DevicePtr::NULL,
                a_packed,
                a_scale,
                scales.gate_up,
                eps,
                m_s,
                h,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                merged_w.0,
                merged_s128x4.0,
                merged_out.0,
                m_s,
                merged_n,
                h,
                merged_n,
                merged_scalars,
                stream,
            )?;
            ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                ctx.gpu,
                merged_silu_k,
                merged_out,
                a_packed,
                a_scale,
                scales.down,
                m_s,
                inter,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                self.weights.down_proj.weight.0,
                down_s128x4.0,
                h_s.0,
                m_s,
                h,
                inter,
                h,
                down_scalars,
                stream,
            )?;
            row0 += m_s;
        }
        Ok(true)
    }

    #[cfg(not(feature = "cuda"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "non-CUDA stub matching fused prefill FFN signature"
    )]
    pub(crate) fn try_forward_prefill_fused_residual_norm(
        &self,
        hidden: DevicePtr,
        src: DevicePtr,
        norm_weight: &DenseWeight,
        num_tokens: usize,
        eps: f32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        self.try_forward_prefill_fused_residual_rmsnorm_nvfp4(
            hidden,
            src,
            norm_weight,
            num_tokens,
            eps,
            ctx,
            stream,
        )
    }
}
