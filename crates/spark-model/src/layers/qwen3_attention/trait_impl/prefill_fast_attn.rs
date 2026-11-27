// SPDX-License-Identifier: AGPL-3.0-only
//! Full-row Qwen4 projections with a continuation-safe paged-KV flash core.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use super::prefill_moe_attn16_device::{DeviceKernels, mrope_or_scalar};
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::qwen4_prefill_moe::last_layer_crop::{
    self, AttnCropPlan, AttnCropRequest, TailAttnCoreContext, TailAttnCoreGuard, TailAttnPlan,
};
use crate::layers::qwen4_qsa::{prefill_fp8, prefill_sparse};
use crate::layers::{
    Qwen4HyperConnection, ops, qwen4_attn_prep, qwen4_fast_proj, qwen4_fi_attn, qwen4_flash_attn,
    qwen4_prefill_moe,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QkvProjMode {
    /// Project all `rows` of Q, K, and V.
    All,
    /// `Crop::Dead`: project all `rows` of K and V; skip Q completely.
    KvOnly,
    /// `Crop::Tail`: project all `rows` of K and V, and project Q only for the
    /// trailing `attn_rows` starting at `skip_rows` using the full-chunk E4M3
    /// activation `a8` and full-chunk tensor scale `a_scale`.
    KvAllQTail { skip_rows: usize, attn_rows: usize },
}

#[derive(Clone, Copy)]
struct CroppedFp8PrepKernels {
    kv_only: KernelHandle,
    kv_only_k8: KernelHandle,
    q_only: KernelHandle,
}

fn cropped_fp8_prep_kernels(gpu: &dyn GpuBackend) -> Result<CroppedFp8PrepKernels> {
    static K: OnceLock<Mutex<HashMap<usize, CroppedFp8PrepKernels>>> = OnceLock::new();
    let key = std::ptr::from_ref(gpu).cast::<()>() as usize;
    let mut guard = K
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| anyhow::anyhow!("cropped FP8 attention prep kernel cache poisoned"))?;
    if let Some(&cached) = guard.get(&key) {
        return Ok(cached);
    }
    const MODULE: &str = "qwen4_attn_prep_fused_fp8";
    let load = |f: &str| {
        gpu.kernel(MODULE, f).map_err(|e| {
            anyhow::anyhow!("cropped FP8 attention prep kernels unavailable: {MODULE}::{f}: {e}")
        })
    };
    let loaded = CroppedFp8PrepKernels {
        kv_only: load("qwen4_attn_prep_fused_fp8_kv_only")?,
        kv_only_k8: load("qwen4_attn_prep_fused_fp8_kv_only_k8")?,
        q_only: load("qwen4_attn_prep_fused_fp8_q_only")?,
    };
    guard.insert(key, loaded);
    Ok(loaded)
}

impl Qwen3AttentionLayer {
    /// Projects packed inputs into `qkv_output` rows of
    /// `[q (nq*hd) | gate (nq*hd) | k (nkv*hd) | v (nkv*hd)]`, with Q/K norms
    /// applied unless `norms` is false (the fused prep applies them).
    /// Under [`QkvProjMode::KvOnly`] (`Crop::Dead`), `q_proj` and `q_norm` are
    /// skipped while `k_proj` and `v_proj` still cover every row `0..rows`.
    /// Under [`QkvProjMode::KvAllQTail`] (`Crop::Tail`), `k_proj`/`v_proj` run
    /// on all `0..rows` and `q_proj` runs only on `skip_rows..rows`, reusing
    /// the full-chunk E4M3 activation `a8` and full-chunk tensor scale.
    fn qwen4_fast_project_qkv_mode(
        &self,
        normed: DevicePtr,
        rows: usize,
        norms: bool,
        mode: QkvProjMode,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, usize)> {
        ensure!(self.gated, "fast attention QKV requires gated attention");
        ensure!(
            self.attn.q_norm_full.is_none()
                && self.attn.k_norm_full.is_none()
                && self.v_norm_weight.is_none(),
            "fast attention QKV requires per-head Q/K norms and no V norm"
        );
        let q_weight = self
            .q_weight
            .as_ref()
            .and_then(|w| w.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("fast attention Q requires ordinary NVFP4"))?;
        let k_weight = self
            .k_weight
            .as_ref()
            .and_then(|w| w.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("fast attention K requires ordinary NVFP4"))?;
        let v_weight = self
            .v_weight
            .as_ref()
            .and_then(|w| w.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("fast attention V requires ordinary NVFP4"))?;
        let h = ctx.config.hidden_size;
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads);
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads);
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim);
        let q_dim = nq * hd;
        let q_proj_dim = q_dim * 2;
        let kv_dim = nkv * hd;
        let row_stride = q_proj_dim + 2 * kv_dim;
        let row_bytes = row_stride * 2;
        let qkv = ctx.buffers.qkv_output();
        ensure!(
            rows * row_bytes <= ctx.buffers.sizes().qkv_output,
            "fast attention QKV staging exceeds QKV arena"
        );
        let gpu = ctx.gpu;
        let fp8 = qwen4_fast_proj::fp8_selection()?.attn;
        let shared_act = fp8
            && qwen4_fast_proj::fp8_shared_act_selected()
            && qwen4_fast_proj::fp8_tensor_scales();
        let qkv_pad_guard =
            qwen4_fast_proj::PadRowsGuard::enter(ctx.buffers.sizes().qkv_output / row_bytes);
        match mode {
            QkvProjMode::All => {
                if shared_act {
                    // Quantize `normed` once for all three projections.
                    use qwen4_fast_proj::SharedActOut;
                    qwen4_fast_proj::fp8_gemm_shared_act(
                        gpu,
                        normed,
                        &[
                            SharedActOut {
                                weight: k_weight,
                                out: qkv.offset(q_proj_dim * 2),
                                n: kv_dim,
                                qg: None,
                            },
                            SharedActOut {
                                weight: v_weight,
                                out: qkv.offset((q_proj_dim + kv_dim) * 2),
                                n: kv_dim,
                                qg: None,
                            },
                            SharedActOut {
                                weight: q_weight,
                                out: qkv,
                                n: q_proj_dim,
                                qg: Some((nq, hd)),
                            },
                        ],
                        rows,
                        h,
                        row_stride,
                        stream,
                    )?;
                } else {
                    qwen4_fast_proj::proj_gemm_ex(
                        gpu,
                        fp8,
                        normed,
                        q_weight,
                        qkv,
                        rows,
                        q_proj_dim,
                        h,
                        row_stride,
                        Some((nq, hd)),
                        stream,
                    )?;
                    qwen4_fast_proj::proj_gemm_ex(
                        gpu,
                        fp8,
                        normed,
                        k_weight,
                        qkv.offset(q_proj_dim * 2),
                        rows,
                        kv_dim,
                        h,
                        row_stride,
                        None,
                        stream,
                    )?;
                    qwen4_fast_proj::proj_gemm_ex(
                        gpu,
                        fp8,
                        normed,
                        v_weight,
                        qkv.offset((q_proj_dim + kv_dim) * 2),
                        rows,
                        kv_dim,
                        h,
                        row_stride,
                        None,
                        stream,
                    )?;
                }
            }
            QkvProjMode::KvOnly => {
                if shared_act {
                    use qwen4_fast_proj::SharedActOut;
                    qwen4_fast_proj::fp8_gemm_shared_act(
                        gpu,
                        normed,
                        &[
                            SharedActOut {
                                weight: k_weight,
                                out: qkv.offset(q_proj_dim * 2),
                                n: kv_dim,
                                qg: None,
                            },
                            SharedActOut {
                                weight: v_weight,
                                out: qkv.offset((q_proj_dim + kv_dim) * 2),
                                n: kv_dim,
                                qg: None,
                            },
                        ],
                        rows,
                        h,
                        row_stride,
                        stream,
                    )?;
                } else {
                    qwen4_fast_proj::proj_gemm_ex(
                        gpu,
                        fp8,
                        normed,
                        k_weight,
                        qkv.offset(q_proj_dim * 2),
                        rows,
                        kv_dim,
                        h,
                        row_stride,
                        None,
                        stream,
                    )?;
                    qwen4_fast_proj::proj_gemm_ex(
                        gpu,
                        fp8,
                        normed,
                        v_weight,
                        qkv.offset((q_proj_dim + kv_dim) * 2),
                        rows,
                        kv_dim,
                        h,
                        row_stride,
                        None,
                        stream,
                    )?;
                }
            }
            QkvProjMode::KvAllQTail {
                skip_rows,
                attn_rows,
            } => {
                ensure!(
                    shared_act && !norms && attn_rows > 0 && skip_rows + attn_rows == rows,
                    "Crop::Tail Q-slice projection requires shared-act FP8 tensor scales and fused prep"
                );
                ensure!(
                    q_proj_dim > 0
                        && h > 0
                        && h.is_multiple_of(16)
                        && q_proj_dim.is_multiple_of(16)
                        && !q_weight.weight.is_null()
                        && !q_weight.weight_scale.is_null()
                        && q_weight.weight_scale_2.is_finite(),
                    "Crop::Tail Q-slice projection requires valid aligned NVFP4 Q weight"
                );
                // Step 1: quantize all `rows` into `a8` with the full-chunk tensor scale
                // (`ws + 12`, consuming any fused HC-upmix absmax hand-off) and project
                // full `0..rows` K and V identically to the uncropped chunk.
                use qwen4_fast_proj::SharedActOut;
                qwen4_fast_proj::fp8_gemm_shared_act(
                    gpu,
                    normed,
                    &[
                        SharedActOut {
                            weight: k_weight,
                            out: qkv.offset(q_proj_dim * 2),
                            n: kv_dim,
                            qg: None,
                        },
                        SharedActOut {
                            weight: v_weight,
                            out: qkv.offset((q_proj_dim + kv_dim) * 2),
                            n: kv_dim,
                            qg: None,
                        },
                    ],
                    rows,
                    h,
                    row_stride,
                    stream,
                )?;
                // Step 2: project Q only for `skip_rows..rows`, reading the exact
                // full-chunk `a8[skip_rows..rows]` slice and `a_scale` (`ws + 12`).
                // `fp8_scratch` caps `n` at >= 16384 and `k` at >= 6144, so this
                // returns the exact same `(w8, ws, a8)` populated by Step 1.
                let tk = qwen4_fast_proj::fp8_tensor_kernels(gpu)?;
                let (w8, ws, a8, _) = qwen4_fast_proj::fp8_scratch(gpu, q_proj_dim, rows, h)?;
                qwen4_fast_proj::fp8_weight_tensor_gemm(
                    gpu,
                    &tk,
                    q_weight,
                    qkv.offset(skip_rows * row_bytes),
                    attn_rows,
                    q_proj_dim,
                    h,
                    row_stride,
                    Some((nq, hd)),
                    w8,
                    ws,
                    a8.offset(skip_rows * h),
                    false,
                    stream,
                )?;
            }
        }
        drop(qkv_pad_guard);
        let eps = ctx.config.rms_norm_eps as f32;
        if norms && mode == QkvProjMode::All && !self.attn.q_norm.weight.is_null() {
            qwen4_fast_proj::qk_norm_strided(
                gpu,
                qkv,
                self.attn.q_norm.weight,
                rows,
                nq,
                hd,
                row_stride,
                eps,
                stream,
            )?;
        }
        if norms && !self.attn.k_norm.weight.is_null() {
            qwen4_fast_proj::qk_norm_strided(
                gpu,
                qkv.offset(q_proj_dim * 2),
                self.attn.k_norm.weight,
                rows,
                nkv,
                hd,
                row_stride,
                eps,
                stream,
            )?;
        }
        Ok((qkv, row_bytes))
    }

    /// Cropped twin of [`Self::prefill_moe_attn_fullrow_prep`] for Layer 47:
    /// - Always populates QSA indexer compressed keys + raw ring for ALL `0..rows`.
    /// - Always populates BF16/NVFP4 paged K/V cache and (when FP8 QSA is active)
    ///   E4M3 K8/(needed V8) shadow pages for ALL `0..rows`, skipping Q heads.
    /// - When `tail_plan` is `Some(plan)` (`Crop::Tail`), also runs Q-only norm +
    ///   MRoPE + E4M3 quantization for only the trailing `plan.attn_rows` rows into
    ///   `q8[0..plan.attn_rows]`.
    #[allow(clippy::too_many_arguments)]
    fn prefill_moe_attn_fullrow_prep_cropped(
        &self,
        packed_inputs: DevicePtr,
        qkv: DevicePtr,
        qkv_row_bytes: usize,
        rows: usize,
        seq_len_start: usize,
        tail_plan: Option<TailAttnPlan>,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
        device: DeviceKernels,
        prep_fused: bool,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= qwen4_prefill_moe::max_chunk_rows(),
            "cropped full-row attention prep requires 1..=chunk_cap rows (got {rows})"
        );
        let metadata = ctx
            .attn_metadata
            .ok_or_else(|| anyhow::anyhow!("cropped attention prep requires paged metadata"))?;
        ensure!(
            matches!(self.kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Nvfp4)
                && metadata.num_seqs == 1,
            "cropped attention prep requires C1 BF16 or NVFP4 paged metadata"
        );
        ensure!(
            self.gated && self.mrope_interleaved && self.rope_mrope_interleaved_k.0 != 0,
            "cropped attention prep requires gated Qwen4 MRoPE attention"
        );
        let h = ctx.config.hidden_size;
        let nq = ctx.config.num_attention_heads as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let hd = ctx.config.head_dim as u32;
        let q_proj_dim = nq * hd * 2;
        let kv_dim = nkv * hd;
        let row_stride = u32::try_from(qkv_row_bytes / 2)?;
        ensure!(
            row_stride == q_proj_dim + 2 * kv_dim,
            "cropped attention prep QKV stride mismatch"
        );
        let rotary_dim = ctx.config.rotary_dim() as u32;
        let theta = ctx.config.rope_theta as f32;
        let eps = ctx.config.rms_norm_eps as f32;
        if metadata.qwen4_qsa_required {
            let qsa = self
                .qwen4_qsa
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("cropped attention prep requires QSA"))?;
            qsa.validate_exact_group4()?;
            qsa.update_prefill_exact_tile(
                packed_inputs,
                seq_len_start,
                rows,
                kv_cache,
                metadata,
                h as u32,
                eps,
                theta,
                rotary_dim,
                ctx.gpu,
                stream,
            )?;
        }
        if !prep_fused {
            ensure!(
                tail_plan.is_none(),
                "Crop::Tail attention prep requires fused FP8 prep"
            );
            ensure!(
                !self.yarn_scaling_active(),
                "cropped non-fused attention prep does not support YaRN scaling"
            );
            mrope_or_scalar(
                Some(device),
                ctx.gpu,
                self.rope_mrope_interleaved_k,
                qkv,
                qkv_row_bytes,
                metadata.positions,
                metadata.positions_h,
                metadata.positions_w,
                row_stride,
                q_proj_dim,
                nq,
                nkv,
                hd,
                rotary_dim,
                theta,
                rows as u32,
                stream,
            )?;
            let k = qkv.offset(q_proj_dim as usize * 2);
            let v = k.offset(kv_dim as usize * 2);
            return self.write_kv_cache(
                ctx.gpu,
                k,
                v,
                kv_cache,
                metadata.slot,
                rows as u32,
                nkv,
                hd,
                kv_cache.block_size() as u32,
                row_stride,
                row_stride,
                stream,
                false,
            );
        }

        ensure!(
            hd as usize == qwen4_attn_prep::HEAD_DIM && row_stride == 2 * nq * hd + 2 * nkv * hd,
            "{} requires head_dim 256 and fused [q | gate | k | v] rows",
            qwen4_attn_prep::SELECTOR
        );
        ensure!(
            !self.attn.q_norm.weight.is_null() && !self.attn.k_norm.weight.is_null(),
            "{} requires per-head Q and K norms",
            qwen4_attn_prep::SELECTOR
        );
        let gpu = ctx.gpu;
        let layer = self.attn_layer_idx;
        let block_size = kv_cache.block_size();
        let layer_kv_dtype = kv_cache.dtype_for_layer(layer);
        let k_stride_bytes = kv_cache.k_block_stride_bytes_for_layer(layer);
        ensure!(
            k_stride_bytes == kv_cache.v_block_stride_bytes_for_layer(layer)
                && match layer_kv_dtype {
                    KvCacheDtype::Bf16 => {
                        k_stride_bytes == block_size * nkv as usize * qwen4_attn_prep::HEAD_DIM * 2
                    }
                    KvCacheDtype::Nvfp4 => {
                        let elems = block_size * nkv as usize * qwen4_attn_prep::HEAD_DIM;
                        k_stride_bytes == elems / 2 + elems / 16
                    }
                    _ => false,
                },
            "{} requires a dense BF16 or NVFP4 paged cache (got {layer_kv_dtype})",
            qwen4_attn_prep::SELECTOR
        );
        let yarn_active = !self.qwen4_yarn_inv_freq.is_null();
        let freq = if yarn_active {
            ensure!(
                (self.qwen4_yarn_attention_factor - 1.0).abs() <= f32::EPSILON,
                "{} requires unit YaRN attention_factor (1.0), got {}",
                qwen4_attn_prep::SELECTOR,
                self.qwen4_yarn_attention_factor
            );
            self.qwen4_yarn_inv_freq
        } else {
            qwen4_attn_prep::freq_table(gpu, rotary_dim, theta, stream)?
        };
        let k_offset = 2 * nq * hd;
        if layer_kv_dtype == KvCacheDtype::Bf16
            && !yarn_active
            && nq == 24
            && nkv == 2
            && let Some(fp8_mode) = prefill_fp8::mode()?
        {
            ensure!(
                !metadata.block_table.is_null(),
                "{}: requires C1 paged metadata",
                prefill_fp8::SELECTOR
            );
            let sc = prefill_fp8::layer_scales(
                gpu,
                layer,
                self.attn.q_norm.weight,
                self.attn.k_norm.weight,
            )?;
            let (k8, v8) = prefill_fp8::shadow(gpu, layer, kv_cache)?;
            let v8_shared = prefill_fp8::v8_shared()?;
            // On Crop::Dead with shared V8, no consumer reads this layer's V8
            // plane (this layer's core is skipped and later chunks use Mode::Qk);
            // with per-layer V8 or Crop::Tail, write V8 for all 0..rows.
            let write_v8 =
                fp8_mode == prefill_fp8::Mode::QkPv && (tail_plan.is_some() || !v8_shared);
            if rotary_dim == 64 && prefill_sparse::prep_fused_selected()? {
                let ck = cropped_fp8_prep_kernels(gpu)?;
                let kv_kernel = if write_v8 { ck.kv_only } else { ck.kv_only_k8 };
                KernelLaunch::new(gpu, kv_kernel)
                    .grid([u32::try_from(rows)?, 1, 1])
                    .block([128, 1, 1])
                    .arg_ptr(qkv)
                    .arg_ptr(self.attn.k_norm.weight)
                    .arg_ptr(metadata.positions)
                    .arg_ptr(metadata.positions_h)
                    .arg_ptr(metadata.positions_w)
                    .arg_ptr(freq)
                    .arg_ptr(kv_cache.k_pool_ptr(layer))
                    .arg_ptr(kv_cache.v_pool_ptr(layer))
                    .arg_ptr(metadata.slot)
                    .arg_u32(row_stride)
                    .arg_u32(k_offset)
                    .arg_u32(k_offset + nkv * hd)
                    .arg_f32(eps)
                    .arg_ptr(k8)
                    .arg_ptr(if write_v8 { v8 } else { DevicePtr::NULL })
                    .arg_f32(sc.k[0])
                    .arg_f32(sc.k[1])
                    .arg_f32(sc.v[0])
                    .arg_f32(sc.v[1])
                    .launch(stream)?;
            } else {
                ensure!(
                    tail_plan.is_none(),
                    "Crop::Tail FP8 attention prep requires rotary_dim == 64"
                );
                KernelLaunch::new(gpu, prefill_fp8::kernels(gpu)?.prep)
                    .grid([u32::try_from(rows)? * nkv, 1, 1])
                    .block([qwen4_attn_prep::HEAD_DIM as u32, 1, 1])
                    .arg_ptr(qkv)
                    .arg_ptr(self.attn.q_norm.weight)
                    .arg_ptr(self.attn.k_norm.weight)
                    .arg_ptr(metadata.positions)
                    .arg_ptr(metadata.positions_h)
                    .arg_ptr(metadata.positions_w)
                    .arg_ptr(freq)
                    .arg_ptr(kv_cache.k_pool_ptr(layer))
                    .arg_ptr(kv_cache.v_pool_ptr(layer))
                    .arg_ptr(metadata.slot)
                    .arg_u32(0)
                    .arg_u32(nkv)
                    .arg_u32(row_stride)
                    .arg_u32(k_offset)
                    .arg_u32(k_offset + nkv * hd)
                    .arg_u32(rotary_dim)
                    .arg_u32(block_size as u32)
                    .arg_f32(eps)
                    .arg_ptr(DevicePtr::NULL)
                    .arg_ptr(k8)
                    .arg_ptr(if write_v8 { v8 } else { DevicePtr::NULL })
                    .arg_f32(sc.q[0])
                    .arg_f32(sc.q[1])
                    .arg_f32(sc.k[0])
                    .arg_f32(sc.k[1])
                    .arg_f32(sc.v[0])
                    .arg_f32(sc.v[1])
                    .launch(stream)?;
            }
            let refill = prefill_fp8::note_prepped(kv_cache, layer, seq_len_start, rows)?;
            prefill_fp8::refill_shadow(gpu, kv_cache, layer, metadata.block_table, refill, stream)?;
            if let Some(plan) = tail_plan {
                let ck = cropped_fp8_prep_kernels(gpu)?;
                let q8 = prefill_fp8::q8_scratch(gpu, plan.attn_rows)?;
                KernelLaunch::new(gpu, ck.q_only)
                    .grid([u32::try_from(plan.attn_rows)?, 1, 1])
                    .block([qwen4_attn_prep::HEAD_DIM as u32, 1, 1])
                    .arg_ptr(qkv.offset(plan.skip_rows * qkv_row_bytes))
                    .arg_ptr(self.attn.q_norm.weight)
                    .arg_ptr(metadata.positions.offset(plan.skip_rows * 4))
                    .arg_ptr(metadata.positions_h.offset(plan.skip_rows * 4))
                    .arg_ptr(metadata.positions_w.offset(plan.skip_rows * 4))
                    .arg_ptr(freq)
                    .arg_u32(row_stride)
                    .arg_f32(eps)
                    .arg_ptr(q8)
                    .arg_f32(sc.q[0])
                    .arg_f32(sc.q[1])
                    .launch(stream)?;
            }
            return Ok(());
        }

        ensure!(
            tail_plan.is_none(),
            "Crop::Tail attention prep requires BF16 unscaled FP8 QSA prep"
        );
        // Non-FP8 Crop::Dead KV-only prep: `nq = 0` skips all Q heads in
        // `qwen4_attn_prep_fused[_yarn|_nvfp4]` while preserving `k_offset` / `v_offset`.
        let k = qwen4_attn_prep::kernels(gpu)?;
        match layer_kv_dtype {
            KvCacheDtype::Nvfp4 => {
                ensure!(
                    self.active_bitfp4_kv(hd, u32::try_from(block_size)?)
                        .is_none(),
                    "ATLAS_BITFP4_KV=1 is incompatible with qwen4_attn_prep_fused_nvfp4 \
                     (legacy token-major NVFP4 KV writer, layer {}); unset ATLAS_BITFP4_KV",
                    layer
                );
                ensure!(
                    k.fused_nvfp4.0 != 0,
                    "{}: qwen4_attn_prep_fused_nvfp4 kernel unavailable",
                    qwen4_attn_prep::SELECTOR
                );
                KernelLaunch::new(gpu, k.fused_nvfp4)
                    .grid([u32::try_from(rows)? * nkv, 1, 1])
                    .block([qwen4_attn_prep::HEAD_DIM as u32, 1, 1])
                    .arg_ptr(qkv)
                    .arg_ptr(self.attn.q_norm.weight)
                    .arg_ptr(self.attn.k_norm.weight)
                    .arg_ptr(metadata.positions)
                    .arg_ptr(metadata.positions_h)
                    .arg_ptr(metadata.positions_w)
                    .arg_ptr(freq)
                    .arg_ptr(kv_cache.k_pool_ptr(layer))
                    .arg_ptr(kv_cache.v_pool_ptr(layer))
                    .arg_ptr(metadata.slot)
                    .arg_u32(0)
                    .arg_u32(nkv)
                    .arg_u32(row_stride)
                    .arg_u32(k_offset)
                    .arg_u32(k_offset + nkv * hd)
                    .arg_u32(rotary_dim)
                    .arg_u32(block_size as u32)
                    .arg_u64(k_stride_bytes as u64)
                    .arg_u64(kv_cache.nvfp4_data_bytes() as u64)
                    .arg_f32(eps)
                    .arg_f32(if yarn_active {
                        self.qwen4_yarn_attention_factor
                    } else {
                        1.0
                    })
                    .launch(stream)?;
            }
            KvCacheDtype::Bf16 if yarn_active => {
                ensure!(
                    k.fused_yarn.0 != 0,
                    "{}: qwen4_attn_prep_fused_yarn kernel unavailable",
                    qwen4_attn_prep::SELECTOR
                );
                KernelLaunch::new(gpu, k.fused_yarn)
                    .grid([u32::try_from(rows)? * nkv, 1, 1])
                    .block([qwen4_attn_prep::HEAD_DIM as u32, 1, 1])
                    .arg_ptr(qkv)
                    .arg_ptr(self.attn.q_norm.weight)
                    .arg_ptr(self.attn.k_norm.weight)
                    .arg_ptr(metadata.positions)
                    .arg_ptr(metadata.positions_h)
                    .arg_ptr(metadata.positions_w)
                    .arg_ptr(freq)
                    .arg_ptr(kv_cache.k_pool_ptr(layer))
                    .arg_ptr(kv_cache.v_pool_ptr(layer))
                    .arg_ptr(metadata.slot)
                    .arg_u32(0)
                    .arg_u32(nkv)
                    .arg_u32(row_stride)
                    .arg_u32(k_offset)
                    .arg_u32(k_offset + nkv * hd)
                    .arg_u32(rotary_dim)
                    .arg_u32(block_size as u32)
                    .arg_f32(eps)
                    .arg_f32(self.qwen4_yarn_attention_factor)
                    .launch(stream)?;
            }
            KvCacheDtype::Bf16 => {
                KernelLaunch::new(gpu, k.fused)
                    .grid([u32::try_from(rows)? * nkv, 1, 1])
                    .block([qwen4_attn_prep::HEAD_DIM as u32, 1, 1])
                    .arg_ptr(qkv)
                    .arg_ptr(self.attn.q_norm.weight)
                    .arg_ptr(self.attn.k_norm.weight)
                    .arg_ptr(metadata.positions)
                    .arg_ptr(metadata.positions_h)
                    .arg_ptr(metadata.positions_w)
                    .arg_ptr(freq)
                    .arg_ptr(kv_cache.k_pool_ptr(layer))
                    .arg_ptr(kv_cache.v_pool_ptr(layer))
                    .arg_ptr(metadata.slot)
                    .arg_u32(0)
                    .arg_u32(nkv)
                    .arg_u32(row_stride)
                    .arg_u32(k_offset)
                    .arg_u32(k_offset + nkv * hd)
                    .arg_u32(rotary_dim)
                    .arg_u32(block_size as u32)
                    .arg_f32(eps)
                    .launch(stream)?;
            }
            _ => unreachable!("dtype checked above"),
        }
        Ok(())
    }

    /// Run Layer-47 `Crop::Tail` QSA sparse attention core (with static-scale
    /// E4M3 direct output), `o_proj`, and attention HC injection on only the
    /// trailing `plan.attn_rows` rows, placing outputs at `skip_rows` offsets
    /// expected by `last_layer_crop::finish_tail`.
    #[allow(clippy::too_many_arguments)]
    fn prefill_fast_attn_tail_core_and_oproj(
        &self,
        attn_hyper: &Qwen4HyperConnection,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        seq_len_start: usize,
        packed_inputs: DevicePtr,
        raw_outputs: DevicePtr,
        qkv: DevicePtr,
        qkv_row_bytes: usize,
        core_bytes: usize,
        attn_width: usize,
        plan: TailAttnPlan,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::cublaslt::stage_timer::mark;
        let metadata = ctx.attn_metadata.ok_or_else(|| {
            anyhow::anyhow!("{}: requires paged metadata", prefill_sparse::SELECTOR)
        })?;
        let qsa = self.qwen4_qsa.as_ref().ok_or_else(|| {
            anyhow::anyhow!("{}: requires the QSA indexer", prefill_sparse::SELECTOR)
        })?;
        let nq = ctx.config.num_attention_heads;
        let hd = ctx.config.head_dim;
        let q_dim = nq * hd;
        let row_stride = qkv_row_bytes / 2;
        let row_bytes = ctx.config.residual_width() * 2;
        let skip = plan.skip_rows;
        let tail = plan.attn_rows;
        let oproj_amax_on = crate::layers::qwen4_fp8_act_amax::oproj_active()?
            && qwen4_fast_proj::fp8_selection()?.attn
            && qwen4_fast_proj::fp8_tensor_scales();
        let oproj_amax_arm = crate::layers::qwen4_fp8_act_amax::Arm::enter_selected(oproj_amax_on);
        let qsa_e4m3_out_arm = prefill_fp8::E4m3OutArm::arm();
        let tail_core_guard = TailAttnCoreGuard::enter(TailAttnCoreContext {
            chunk_start: seq_len_start,
            chunk_rows: num_tokens,
        });
        let tail_meta = AttnMetadataDev {
            positions: metadata.positions.offset(skip * 4),
            positions_h: metadata.positions_h.offset(skip * 4),
            positions_w: metadata.positions_w.offset(skip * 4),
            slot: metadata.slot.offset(skip * 8),
            ..metadata
        };
        let qkv_tail = qkv.offset(skip * qkv_row_bytes);
        let chunk = prefill_sparse::SparseChunk {
            hidden: packed_inputs.offset(skip * core_bytes),
            q: qkv_tail,
            q_stride: u32::try_from(row_stride)?,
            out: raw_outputs,
            gate: qkv_tail.offset(q_dim * 2),
            gate_stride: u32::try_from(row_stride)?,
            rows: tail,
            start: plan.first_pos,
            attn_layer_idx: self.attn_layer_idx,
            hidden_size: ctx.config.hidden_size as u32,
            eps: ctx.config.rms_norm_eps as f32,
            rope_theta: ctx.config.rope_theta as f32,
            rotary_dim: ctx.config.rotary_dim() as u32,
            softmax_scale: self.effective_attn_scale(hd as u32),
            e4m3: true,
            num_query_heads: nq,
        };
        qsa.prefill_sparse_attention(&chunk, kv_cache, tail_meta, ctx.gpu, stream)?;
        drop(tail_core_guard);
        mark("core", stream);
        let o_out_tail = packed_inputs.offset(skip * core_bytes);
        let oproj_pad_guard = qwen4_fast_proj::PadRowsGuard::enter(tail);
        qwen4_fast_proj::proj_gemm_ex(
            ctx.gpu,
            true,
            raw_outputs,
            &self.attn.o_proj,
            o_out_tail,
            tail,
            ctx.config.hidden_size,
            attn_width,
            ctx.config.hidden_size,
            None,
            stream,
        )?;
        drop(oproj_pad_guard);
        drop(qsa_e4m3_out_arm);
        drop(oproj_amax_arm);
        mark("proj_o", stream);
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(o_out_tail.0, tail * ctx.config.hidden_size * 2, stream)?;
        }
        attn_hyper.inject_saved_or_defer(
            crate::layers::qwen4_hyper::prefill_inject::Boundary::Core,
            hidden.offset(skip * row_bytes),
            o_out_tail,
            residual.offset(skip * row_bytes),
            tail,
            None,
            ctx.gpu,
            stream,
        )?;
        static TAIL_LOGGED: std::sync::Once = std::sync::Once::new();
        TAIL_LOGGED.call_once(|| {
            tracing::info!(
                "ATTN_PREFILL_LAST_LAYER_ATTN_CROP_ENGAGED selector={} chunk_start={seq_len_start} chunk_rows={num_tokens} skip_rows={skip} attn_rows={tail} first_pos={} hc_rows={}",
                last_layer_crop::ATTN_CROP_SELECTOR,
                plan.first_pos,
                plan.hc_rows
            );
        });
        Ok(())
    }

    /// Evaluate the Layer-47 attention crop plan for this call without side effects.
    #[allow(clippy::too_many_arguments)]
    fn resolve_last_layer_attn_crop_plan(
        &self,
        num_tokens: usize,
        seq_len_start: usize,
        attn_width: usize,
        flash: bool,
        fullrow: bool,
        prep_fused: bool,
        kv_only: bool,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
    ) -> Result<AttnCropPlan> {
        let effective_crop = last_layer_crop::active()
            .or_else(|| kv_only.then_some(last_layer_crop::Crop::Dead { rows_after: 0 }));
        match effective_crop {
            None => Ok(AttnCropPlan::Full),
            Some(crop @ last_layer_crop::Crop::Dead { .. }) => {
                Ok(last_layer_crop::plan_attn_crop(AttnCropRequest {
                    attn_crop_enabled: true,
                    crop: Some(crop),
                    chunk_rows: num_tokens,
                    chunk_start: seq_len_start,
                    flash,
                    fullrow,
                    static_qsa_e4m3_out_active: false,
                    fp8_prep_fused_active: false,
                    finish_tail_ready: false,
                    qsa_sparse_active: false,
                }))
            }
            Some(crop @ last_layer_crop::Crop::Tail { .. }) => {
                let attn_crop_enabled = last_layer_crop::attn_crop_selected()?;
                let fp8_attn = qwen4_fast_proj::fp8_selection()?.attn;
                let tensor_scales = qwen4_fast_proj::fp8_tensor_scales();
                let shared_act = qwen4_fast_proj::fp8_shared_act_selected();
                let nq = self
                    .num_q_heads_override
                    .unwrap_or(ctx.config.num_attention_heads);
                let nkv = self
                    .num_kv_heads_override
                    .unwrap_or(ctx.config.num_key_value_heads);
                let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim);
                let qsa_sparse_active = flash
                    && self.qsa_sparse_prefill_selected(seq_len_start, num_tokens)?
                    && prefill_sparse::v2_selected()?;
                let configured_fp8_mode = prefill_fp8::mode()?;
                let fp8_prep_fused_active = prep_fused
                    && prefill_sparse::prep_fused_selected()?
                    && !prefill_sparse::dump_enabled()
                    && fp8_attn
                    && tensor_scales
                    && shared_act
                    && self.gated
                    && self.attn.q_norm_full.is_none()
                    && self.attn.k_norm_full.is_none()
                    && self.v_norm_weight.is_none()
                    && !self.attn.q_norm.weight.is_null()
                    && !self.attn.k_norm.weight.is_null()
                    && nq == 24
                    && nkv == 2
                    && hd == qwen4_attn_prep::HEAD_DIM
                    && ctx.config.rotary_dim() == 64
                    && kv_cache.dtype_for_layer(self.attn_layer_idx) == KvCacheDtype::Bf16
                    && self.qwen4_yarn_inv_freq.is_null()
                    && configured_fp8_mode.is_some()
                    && ctx.attn_metadata.is_some_and(|m| {
                        m.num_seqs == 1
                            && m.qwen4_qsa_required
                            && !m.block_table.is_null()
                            && !m.positions.is_null()
                    });
                let e4m3_out_selected = prefill_fp8::e4m3_out()?;
                let static_qsa_e4m3_out_active = if let Some(cfg_mode) = configured_fp8_mode
                    && fp8_prep_fused_active
                    && qsa_sparse_active
                    && e4m3_out_selected
                    && attn_width == prefill_fp8::E4M3_OUT_COLS
                    && ctx.config.hidden_size > 0
                    && ctx.config.hidden_size.is_multiple_of(16)
                    && !self.attn.o_proj.weight.is_null()
                    && !self.attn.o_proj.weight_scale.is_null()
                    && self.attn.o_proj.weight_scale_2.is_finite()
                    && !crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active()
                {
                    let _ = prefill_fp8::v8_shared()?;
                    let effective_mode = prefill_fp8::effective_chunk_mode(cfg_mode, seq_len_start);
                    let pipe = prefill_fp8::pipe()?;
                    let wdec_active = prefill_fp8::warp_decouple()?.is_some();
                    let sub_cta_active = prefill_fp8::sub_cta()?.is_some();
                    let diagnostics_active = prefill_fp8::e4m3_out_diagnostics_active()?;
                    prefill_fp8::e4m3_out_eligible(prefill_fp8::E4m3OutEligibility {
                        selector_enabled: e4m3_out_selected,
                        armed_for_fp8_oproj: true,
                        mode: effective_mode,
                        pipe,
                        wdec_active,
                        sub_cta_active,
                        rows: num_tokens,
                        hidden_size: ctx.config.hidden_size,
                        diagnostics_active,
                    })
                } else {
                    false
                };
                let finish_tail_ready = !qwen4_prefill_moe::expert_major_mixer_slice()
                    && qwen4_prefill_moe::hyper_selected()?
                    && crate::layers::qwen4_hyper::prefill_fused::level() == 2;
                Ok(last_layer_crop::plan_attn_crop(AttnCropRequest {
                    attn_crop_enabled,
                    crop: Some(crop),
                    chunk_rows: num_tokens,
                    chunk_start: seq_len_start,
                    flash,
                    fullrow,
                    static_qsa_e4m3_out_active,
                    fp8_prep_fused_active,
                    finish_tail_ready,
                    qsa_sparse_active,
                }))
            }
        }
    }

    /// Full-row attention sublayer; the caller has already run HC preparation
    /// (packed inputs for every row live in `packed_inputs` = norm_output).
    /// `kv_only` (a dead final layer, `ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP`):
    /// skips `q_proj` and Q-side prep on the full-row flash path and stops once
    /// every KV/QSA write is done; returns whether the core, o_proj and
    /// injection were skipped.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_moe_only_fast_attn(
        &self,
        attn_hyper: &Qwen4HyperConnection,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        seq_len_start: usize,
        packed_inputs: DevicePtr,
        raw_outputs: DevicePtr,
        core_bytes: usize,
        attn_width: usize,
        attn_row_bytes: usize,
        block_table: &mut Vec<u32>,
        kv_cache: &mut PagedKvCache,
        attn16_device: Option<DeviceKernels>,
        attn32: bool,
        kv_only: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let flash = qwen4_flash_attn::selected() && self.prefill_attn_k.0 != 0;
        let fullrow =
            flash && attn16_device.is_some() && super::prefill_moe_attn16::fullrow_selected();
        ensure!(
            (num_tokens.is_multiple_of(16) || fullrow)
                && num_tokens > 0
                && num_tokens <= qwen4_prefill_moe::max_chunk_rows(),
            "fast attention requires a 16-row multiple or full-row flash prefill (got {num_tokens})"
        );
        ensure!(
            num_tokens * attn_row_bytes <= ctx.buffers.sizes().ssm_qkvz
                && num_tokens * core_bytes <= ctx.buffers.sizes().norm_output,
            "fast attention full-row scratch is undersized"
        );
        use spark_runtime::cublaslt::stage_timer::mark;
        mark("hc_prepare", stream);
        // The fused prep applies the Q/K norms together with MRoPE and the KV write.
        let prep_fused = fullrow && super::prefill_attn_prep_fused::selected()?;
        let attn_crop_plan = self.resolve_last_layer_attn_crop_plan(
            num_tokens,
            seq_len_start,
            attn_width,
            flash,
            fullrow,
            prep_fused,
            kv_only,
            kv_cache,
            ctx,
        )?;
        let proj_mode = match attn_crop_plan {
            AttnCropPlan::Full => QkvProjMode::All,
            AttnCropPlan::DeadKvOnly => QkvProjMode::KvOnly,
            AttnCropPlan::Tail(plan) => QkvProjMode::KvAllQTail {
                skip_rows: plan.skip_rows,
                attn_rows: plan.attn_rows,
            },
        };
        // Overlap QSA `iq_proj` + `iq_norm_rope` onto the secondary stream concurrently
        // with `qkv_proj` + `kv_prep` (`ATLAS_QWEN4_PREFILL_QSA_IQ_OVERLAP`, auto-ON on
        // `active_sm_count() > 64`, OFF on GB10 `<= 64`). Skipped for `kv_only`,
        // `Crop::DeadKvOnly`, and `Crop::Tail` (which needs exact ref-M heuristic pinning).
        let qsa_iq_overlap_guard =
            if matches!(attn_crop_plan, AttnCropPlan::Full) && flash && !kv_only {
                self.try_start_prefill_qsa_iq_overlap(
                    packed_inputs,
                    num_tokens,
                    seq_len_start,
                    ctx,
                    stream,
                )?
            } else {
                None
            };
        let (qkv, qkv_row_bytes) = self.qwen4_fast_project_qkv_mode(
            packed_inputs,
            num_tokens,
            !prep_fused,
            proj_mode,
            ctx,
            stream,
        )?;
        mark("proj_qkv", stream);
        let mut tile_start = 0;
        if fullrow && let Some(device) = attn16_device {
            match attn_crop_plan {
                AttnCropPlan::Full => {
                    self.prefill_moe_attn_fullrow_prep(
                        packed_inputs,
                        qkv,
                        qkv_row_bytes,
                        num_tokens,
                        seq_len_start,
                        kv_cache,
                        ctx,
                        stream,
                        device,
                        prep_fused,
                    )?;
                }
                AttnCropPlan::DeadKvOnly => {
                    self.prefill_moe_attn_fullrow_prep_cropped(
                        packed_inputs,
                        qkv,
                        qkv_row_bytes,
                        num_tokens,
                        seq_len_start,
                        None,
                        kv_cache,
                        ctx,
                        stream,
                        device,
                        prep_fused,
                    )?;
                    last_layer_crop::note_dead_q_skipped();
                }
                AttnCropPlan::Tail(plan) => {
                    self.prefill_moe_attn_fullrow_prep_cropped(
                        packed_inputs,
                        qkv,
                        qkv_row_bytes,
                        num_tokens,
                        seq_len_start,
                        Some(plan),
                        kv_cache,
                        ctx,
                        stream,
                        device,
                        prep_fused,
                    )?;
                }
            }
            tile_start = num_tokens;
        }
        while tile_start < num_tokens {
            let tile_rows = if attn32 && num_tokens - tile_start >= 32 {
                32
            } else {
                16
            };
            self.prefill_moe_attn16(
                packed_inputs.offset(tile_start * core_bytes),
                qkv.offset(tile_start * qkv_row_bytes),
                qkv_row_bytes,
                raw_outputs.offset(tile_start * attn_row_bytes),
                seq_len_start,
                tile_start,
                tile_rows,
                block_table,
                kv_cache,
                ctx,
                stream,
                attn16_device,
                flash,
            )?;
            tile_start += tile_rows;
        }
        mark("qsa_rope_kv", stream);
        if (kv_only || attn_crop_plan == AttnCropPlan::DeadKvOnly) && flash {
            return Ok(true);
        }
        if let AttnCropPlan::Tail(plan) = attn_crop_plan {
            self.prefill_fast_attn_tail_core_and_oproj(
                attn_hyper,
                hidden,
                residual,
                num_tokens,
                seq_len_start,
                packed_inputs,
                raw_outputs,
                qkv,
                qkv_row_bytes,
                core_bytes,
                attn_width,
                plan,
                kv_cache,
                ctx,
                stream,
            )?;
            last_layer_crop::note_tail_attn_cropped(plan.attn_rows);
            let _ = qwen4_prefill_moe::attn16::SELECTOR;
            return Ok(false);
        }
        // ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED (default off): the E4M3 QSA core reduces the absmax
        // of the gated output it writes into raw_outputs, and the o_proj below (a tensor-scale FP8
        // projection of exactly those rows) skips its own absmax pass; byte-identical (max is
        // order-free, same rule). The arm is scoped to the core + o_proj and one-shot.
        let oproj_amax_on = crate::layers::qwen4_fp8_act_amax::oproj_active()?
            && qwen4_fast_proj::fp8_selection()?.attn
            && qwen4_fast_proj::fp8_tensor_scales();
        let oproj_amax_arm = crate::layers::qwen4_fp8_act_amax::Arm::enter_selected(oproj_amax_on);
        // ATLAS_QWEN4_PREFILL_QSA_E4M3_OUT (default on): when QSA `qkpv_pipe` feeds a
        // tensor-scale FP8 `o_proj` directly, emit static-prescaled E4M3 rows into
        // `qwen4_fast_proj`'s activation scratch, skipping BF16 `raw_outputs` writes and
        // `qwen4_fp8_act_rows_tensor_v2`. Scoped one-shot arm, cleared on drop.
        let qsa_e4m3_out_on = flash
            && qwen4_fast_proj::fp8_selection()?.attn
            && qwen4_fast_proj::fp8_tensor_scales()
            && num_tokens >= crate::layers::qwen4_qsa::prefill_fp8::E4M3_OUT_MIN_ROWS
            && attn_width == crate::layers::qwen4_qsa::prefill_fp8::E4M3_OUT_COLS
            && ctx.config.hidden_size > 0
            && ctx.config.hidden_size.is_multiple_of(16)
            && !self.attn.o_proj.weight.is_null()
            && !self.attn.o_proj.weight_scale.is_null()
            && self.attn.o_proj.weight_scale_2.is_finite()
            && !crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active()
            && crate::layers::qwen4_qsa::prefill_fp8::e4m3_out()?;
        let qsa_e4m3_out_arm = if qsa_e4m3_out_on {
            crate::layers::qwen4_qsa::prefill_fp8::E4m3OutArm::arm()
        } else {
            crate::layers::qwen4_qsa::prefill_fp8::E4m3OutArm::disarm()
        };
        if flash {
            let nq = ctx.config.num_attention_heads;
            let nkv = ctx.config.num_key_value_heads;
            let hd = ctx.config.head_dim;
            let q_dim = nq * hd;
            let row_stride = qkv_row_bytes / 2;
            let qsa_sparse = self.qsa_sparse_prefill_selected(seq_len_start, num_tokens)?;
            if qsa_sparse {
                self.prefill_qsa_sparse_core(
                    packed_inputs,
                    qkv,
                    row_stride,
                    raw_outputs,
                    attn_row_bytes / 2,
                    num_tokens,
                    seq_len_start,
                    prep_fused,
                    kv_cache,
                    ctx,
                    stream,
                )?;
            } else if qwen4_fi_attn::selected()
                && kv_cache.dtype_for_layer(self.attn_layer_idx)
                    == spark_runtime::kv_cache::KvCacheDtype::Bf16
            {
                let metadata = ctx.attn_metadata.ok_or_else(|| {
                    anyhow::anyhow!("{} requires paged metadata", qwen4_fi_attn::SELECTOR)
                })?;
                ensure!(
                    seq_len_start == 0 || !metadata.block_table.is_null(),
                    "{} continuation requires the paged block table",
                    qwen4_fi_attn::SELECTOR
                );
                let (q_c, k_c, v_c) = qwen4_flash_attn::compact_qkv(
                    ctx.gpu,
                    qkv,
                    row_stride,
                    q_dim * 2,
                    q_dim * 2 + nkv * hd,
                    num_tokens,
                    nq,
                    nkv,
                    hd,
                    stream,
                )?;
                qwen4_fi_attn::run(
                    &qwen4_fi_attn::Call {
                        q: q_c,
                        k: k_c,
                        v: v_c,
                        k_pool: kv_cache.k_pool_ptr(self.attn_layer_idx),
                        v_pool: kv_cache.v_pool_ptr(self.attn_layer_idx),
                        block_table: metadata.block_table,
                        block_size: kv_cache.block_size(),
                        start: seq_len_start,
                        rows: num_tokens,
                        num_q_heads: nq,
                        num_kv_heads: nkv,
                        head_dim: hd,
                        sm_scale: self.effective_attn_scale(hd as u32),
                        output: raw_outputs,
                    },
                    stream,
                )?;
            } else if seq_len_start == 0 {
                qwen4_flash_attn::run(
                    ctx.gpu,
                    self.prefill_attn_k,
                    qkv,
                    row_stride,
                    q_dim * 2,
                    q_dim * 2 + nkv * hd,
                    raw_outputs,
                    attn_row_bytes / 2,
                    num_tokens,
                    nq,
                    nkv,
                    hd,
                    self.effective_attn_scale(hd as u32),
                    stream,
                )?;
            } else {
                let metadata = ctx.attn_metadata.ok_or_else(|| {
                    anyhow::anyhow!("fast continuation attention requires paged metadata")
                })?;
                ensure!(
                    !metadata.block_table.is_null(),
                    "fast continuation attention requires the paged block table"
                );
                let (q_contiguous, _, _) = qwen4_flash_attn::compact_qkv(
                    ctx.gpu,
                    qkv,
                    row_stride,
                    q_dim * 2,
                    q_dim * 2 + nkv * hd,
                    num_tokens,
                    nq,
                    nkv,
                    hd,
                    stream,
                )?;
                let kv_len = u32::try_from(
                    seq_len_start
                        .checked_add(num_tokens)
                        .ok_or_else(|| anyhow::anyhow!("fast continuation length overflow"))?,
                )?;
                let layer_kv_dtype = kv_cache.dtype_for_layer(self.attn_layer_idx);
                if layer_kv_dtype == spark_runtime::kv_cache::KvCacheDtype::Nvfp4 {
                    ensure!(
                        self.active_bitfp4_kv(
                            u32::try_from(hd)?,
                            u32::try_from(kv_cache.block_size())?,
                        )
                        .is_none(),
                        "ATLAS_BITFP4_KV=1 fast continuation prefill over block-16 transposed V^T \
                         KV cache is not yet supported (layer {}); use single-chunk prefill or \
                         unset ATLAS_BITFP4_KV",
                        self.attn_layer_idx
                    );
                    ensure!(
                        self.prefill_attn_paged_nvfp4_64_k.0 != 0,
                        "fast NVFP4 continuation attention requires prefill_attn_paged_nvfp4_64"
                    );
                    ops::prefill_attention_paged_nvfp4_64(
                        ctx.gpu,
                        self.prefill_attn_paged_nvfp4_64_k,
                        q_contiguous,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        raw_outputs,
                        metadata.block_table,
                        u32::try_from(num_tokens)?,
                        kv_len,
                        u32::try_from(seq_len_start)?,
                        u32::try_from(nq)?,
                        u32::try_from(nkv)?,
                        u32::try_from(hd)?,
                        u32::try_from(kv_cache.block_size())?,
                        self.sliding_window.unwrap_or(0),
                        self.effective_attn_scale(hd as u32),
                        kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                        kv_cache.nvfp4_data_bytes() as u64,
                        stream,
                    )?;
                } else {
                    ensure!(
                        self.prefill_attn_paged_64_k.0 != 0,
                        "fast continuation attention requires the BF16 paged-prefill kernel"
                    );
                    ops::prefill_attention_paged_64(
                        ctx.gpu,
                        self.prefill_attn_paged_64_k,
                        q_contiguous,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        raw_outputs,
                        metadata.block_table,
                        u32::try_from(num_tokens)?,
                        kv_len,
                        u32::try_from(seq_len_start)?,
                        u32::try_from(nq)?,
                        u32::try_from(nkv)?,
                        u32::try_from(hd)?,
                        u32::try_from(kv_cache.block_size())?,
                        self.sliding_window.unwrap_or(0),
                        self.effective_attn_scale(hd as u32),
                        stream,
                    )?;
                }
                static CONTINUATION_LOGGED: std::sync::Once = std::sync::Once::new();
                CONTINUATION_LOGGED.call_once(|| tracing::warn!("ATTN_PREFILL_FLASH_CONTINUATION_ENGAGED q_offset={seq_len_start} rows={num_tokens} kv_dtype={layer_kv_dtype}"));
            }
            // The QSA-sparse core fuses the output gate.
            if !qsa_sparse {
                ops::sigmoid_gate_mul_batched(
                    ctx.gpu,
                    self.sigmoid_gate_mul_batched_k,
                    raw_outputs,
                    qkv.offset(q_dim * 2),
                    raw_outputs,
                    q_dim as u32,
                    row_stride as u32,
                    num_tokens as u32,
                    stream,
                )?;
            }
        }
        // Ensure any overlapped `iq_proj` read of `packed_inputs` has joined onto `stream`
        // strictly before `o_proj` overwrites `packed_inputs` below.
        drop(qsa_iq_overlap_guard);
        mark("core", stream);
        if oproj_amax_on && crate::layers::qwen4_fp8_act_amax::is_armed() {
            // Selected, but the core that wrote raw_outputs was not the E4M3 QSA core: the o_proj
            // runs its own absmax pass (exact either way).
            crate::layers::qwen4_fp8_act_amax::note_oproj_refused();
            static OPROJ_REFUSED: std::sync::Once = std::sync::Once::new();
            OPROJ_REFUSED.call_once(|| {
                tracing::warn!(
                    "OPROJ_AMAX_FUSED_REFUSED selector={} reason=core_not_e4m3_qsa rows={num_tokens} start={seq_len_start} fallback=qwen4_fp8_act_amax",
                    crate::layers::qwen4_fp8_act_amax::OPROJ_SELECTOR
                );
            });
        }
        // O projection for every row lands in norm_output (the packed inputs
        // are no longer needed), then one batched saved-scale injection.
        let oproj_pad_guard =
            qwen4_fast_proj::PadRowsGuard::enter(ctx.buffers.sizes().norm_output / core_bytes);
        qwen4_fast_proj::proj_gemm_ex(
            ctx.gpu,
            qwen4_fast_proj::fp8_selection()?.attn,
            raw_outputs,
            &self.attn.o_proj,
            packed_inputs,
            num_tokens,
            ctx.config.hidden_size,
            attn_width,
            ctx.config.hidden_size,
            None,
            stream,
        )?;
        drop(oproj_pad_guard);
        drop(qsa_e4m3_out_arm);
        drop(oproj_amax_arm);
        mark("proj_o", stream);
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(
                packed_inputs.0,
                num_tokens * ctx.config.hidden_size * 2,
                stream,
            )?;
        }
        // May stay pending for the MLP hyperconnection's normalization.
        attn_hyper.inject_saved_or_defer(
            crate::layers::qwen4_hyper::prefill_inject::Boundary::Core,
            hidden,
            packed_inputs,
            residual,
            num_tokens,
            None,
            ctx.gpu,
            stream,
        )?;
        let projection = if qwen4_fast_proj::fp8_selection()?.attn {
            "fp8_cublaslt_non_bit_exact"
        } else {
            "dequant_cublaslt_bf16_non_bit_exact"
        };
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "ATTN_PREFILL_FAST_ENGAGED selector={} rows={num_tokens} projection={projection}",
                qwen4_fast_proj::SELECTOR
            );
        });
        let _ = qwen4_prefill_moe::attn16::SELECTOR;
        Ok(false)
    }
}
