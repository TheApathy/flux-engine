// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen4-Exp gated residual (four-stream hyperconnection) dispatch.

use anyhow::{Result, bail};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::ops;
use crate::weight_map::{DenseWeight, QuantizedWeight};

pub(crate) mod pending;
mod prefill_exact_admission;
mod prefill_exact_check;
mod prefill_exact_plan;
mod prefill_fast;
pub(crate) mod prefill_fused;
pub(crate) mod prefill_inject;
mod prefill_inject_gemm;
mod prefill_inject_gemm_admit;
mod prefill_inject_launch;
mod prefill_tile;
mod prefill_upmix;
mod prefill_upmix_roof;
mod prefill_upmix_roof_plan;

pub struct Qwen4HyperConnection {
    pub norm: DenseWeight,
    pub down: QuantizedWeight,
    pub up: QuantizedWeight,
    pub inject: Option<QuantizedWeight>,
    /// `ATLAS_QWEN4_HC_BF16=1`: the loader's BF16 dense copies of `down` /
    /// `up` / `inject`, kept resident instead of dropped once the NVFP4
    /// copies above are built. `None` (default) is byte-identical to
    /// today — the NVFP4 copies above are untouched either way, so prefill
    /// never changes. Only `prepare_decode_post_norm` (M=1) reads these.
    pub down_bf16: Option<DenseWeight>,
    pub up_bf16: Option<DenseWeight>,
    pub inject_bf16: Option<DenseWeight>,
    /// `gemv/dense_gemv_bf16`; resolved lazily, zero when no `*_bf16` field
    /// above is set.
    dense_gemv_k: KernelHandle,
    group_norm_k: KernelHandle,
    silu_div_k: KernelHandle,
    mix_k: KernelHandle,
    mix_save_inject_k: KernelHandle,
    inject_k: KernelHandle,
    save_inject_k: KernelHandle,
    inject_saved_k: KernelHandle,
    inject_group_norm_k: KernelHandle,
    /// Vectorized 4-CTA/token `uint4` twin of `qwen4_hc_inject_group_norm`
    /// (`qwen4_hyper::qwen4_hc_inject_group_norm_v2`, 0-ULP bit-identical).
    inject_group_norm_v2_k: KernelHandle,
    stage_mixed_k: KernelHandle,
    pack_mixed_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    /// Exact latency variants for the M=1 decode mixer (ATLAS_QWEN4_HC_GEMV_V2=1):
    /// `w4a16_gemv_pf` (down, inject) and `w4a16_gemv_w32` (up, K <= 512). NULL when off.
    gemv_pf_k: KernelHandle,
    gemv_w32_k: KernelHandle,
    /// Down + inject GEMVs (same input and K) in one launch; NULL when off.
    gemv_pf_dual_k: KernelHandle,
    /// silu_div + up GEMV + mix in one launch; NULL when off.
    up_mix_fused_k: KernelHandle,
    /// Batched-verify read-out tail (silu_div + up + mix + inject transform +
    /// save_inject) for up to HC_TAIL_MAX_ROWS rows in one launch
    /// (`ATLAS_QWEN4_HC_TAIL_ROWS=1`); NULL when off.
    up_mix_tail_rows_k: KernelHandle,
    /// Warp-per-2-outputs batched-verify `up` GEMV + `mix_save_inject` for `rows in 1..=5`.
    up_mix_sw_m5_o2_k: KernelHandle,
    /// Batched-verify down + inject GEMVs in one prefetching exact launch for
    /// up to HC_DUAL_MAX_ROWS rows (`ATLAS_QWEN4_HC_DUAL_ROWS=1`); NULL when off.
    exact_pf_dual_rows_k: KernelHandle,
    exact_pf_dual_rows_m10_k: KernelHandle,
    #[allow(dead_code)]
    exact_pf_dual_rows_m25_k: KernelHandle,
    exact_pf_dual_silu_rows_k: KernelHandle,
    exact_pf_dual_silu_rows_m10_k: KernelHandle,
    #[allow(dead_code)]
    exact_pf_dual_silu_rows_m25_k: KernelHandle,
    w4a16_gemm_k: KernelHandle,
    w4a16_gemm_pipe_k: KernelHandle,
    w4a16_gemm_pipe_m128n128_k: KernelHandle,
    /// Fused Warp-Split-K K-coalesced NVFP4 `hc.down` + exact `silu_div` epilogue
    /// kernels, eagerly loaded at init so CUDA graph capture does zero lookups.
    coalesced_silu_div_k: CoalescedSiluDivKernels,
    /// Single-sweep fused `hc.down + silu_div + hc.inject` Warp-Split-K kernel
    /// for `rows = 7..=24` (`w4a16_gemm_coalesced_n8_sk8_m32_dual_silu_div`,
    /// `ATLAS_QWEN4_HC_DUAL_COALESCED != 0`).
    coalesced_dual_silu_div_k: KernelHandle,
    /// Single-sweep v2 fused `hc.down + silu_div + hc.inject` Warp-Split-K
    /// kernel for `rows = 25..=128` (`w4a16_gemm_coalesced_n16_sk4_m16_dual_silu_div_v2`,
    /// `21` N-blocks x `ceil(M/16)` saturating all 48 SMs, 2-stage K-prefetch,
    /// `ATLAS_QWEN4_HC_DUAL_N32 != 0`).
    coalesced_dual_n16_v2_silu_div_k: KernelHandle,
    /// Fallback single-sweep v1 fused `hc.down + silu_div + hc.inject` kernel
    /// for `rows = 25..=128` (`w4a16_gemm_coalesced_n32_sk4_m16_dual_silu_div`,
    /// `ATLAS_QWEN4_HC_DUAL_N32 != 0`) when the v2 symbol is absent.
    coalesced_dual_n32_silu_div_k: KernelHandle,
    /// High-occupancy SM120/SM103 grid-Split-K dual `hc.down + silu_div + hc.inject`
    /// v2 kernels (`TILE_M = 32/64`), eagerly loaded at init so CUDA graph
    /// capture performs zero symbol lookups.
    coalesced_dual_high_sm_k: DualHighSmKernels,
    /// Pre-allocated shared L2-hot FP32 partial/counter workspace for grid-Split-K
    /// dual launches (`active_sm_count() > 64` only; `None` on GB10).
    dual_splitk_scratch: Option<DualSplitKScratch>,
    /// Fused coalesced NVFP4 `hc.up + mix_save_inject` kernels (`M = 4..=256`,
    /// `ATLAS_QWEN4_HC_UPMIX_COALESCED != 0`), eagerly loaded at init.
    coalesced_upmix_k: CoalescedUpmixKernels,
    w4a16_gemv_exact_m4_k: KernelHandle,
    w4a16_gemv_exact_m8_k: KernelHandle,
    w4a16_gemv_exact_m17_k: KernelHandle,
    w4a16_gemv_exact_m32_k: KernelHandle,
    /// Row-parallel exact K1 GEMV (`ATLAS_W4A16_EXACT_ROWPAR=1`); zero if absent.
    w4a16_gemv_rows_k: KernelHandle,
    /// One-warp-per-output small-K exact twins (`ATLAS_W4A16_EXACT_SWK=1`).
    w4a16_gemv_sw_m4_k: KernelHandle,
    w4a16_gemv_sw_m8_k: KernelHandle,
    w4a16_gemv_sw_m10_k: KernelHandle,
    #[allow(dead_code)]
    w4a16_gemv_sw_m25_k: KernelHandle,
    /// Fused per-stream reciprocal-RMS reduction (`qwen4_hc_fused::qwen4_hc_rstd`)
    /// paired with direct NVFP4 Tensor-Core `up+mix`.
    hc_rstd_k: KernelHandle,
    /// Direct NVFP4 Tensor-Core fused `up+mix` roof kernel
    /// (`qwen4_hc_upmix_roof::qwen4_hc_upmix_roof_t128_w4`, 49,184 B smem).
    upmix_roof_t128_w4_k: KernelHandle,
    /// Row-count bitset (`1..=128`) tracking widths whose `self.down` FP8
    /// weight/amax cache and cuBLASLt plan have warmed outside graph capture.
    fp8_down_warmed: WarmedRowsMask,
    /// Same for `self.up` (non-canonical fallback when `upmix_roof_t128_w4_k`
    /// is unavailable).
    fp8_up_warmed: WarmedRowsMask,
    hidden_size: usize,
    hc_count: usize,
    rank: usize,
}

impl Qwen4HyperConnection {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        norm: DenseWeight,
        down: QuantizedWeight,
        up: QuantizedWeight,
        inject: Option<QuantizedWeight>,
        hidden_size: usize,
        hc_count: usize,
        rank: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        Self::new_with_bf16(
            norm,
            down,
            up,
            inject,
            None,
            None,
            None,
            hidden_size,
            hc_count,
            rank,
            gpu,
        )
    }

    /// Same as [`Self::new`], additionally taking the loader's retained
    /// BF16 copies (`ATLAS_QWEN4_HC_BF16=1`). Every `*_bf16` argument is
    /// `None` on the default path, which behaves exactly as [`Self::new`].
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_bf16(
        norm: DenseWeight,
        down: QuantizedWeight,
        up: QuantizedWeight,
        inject: Option<QuantizedWeight>,
        down_bf16: Option<DenseWeight>,
        up_bf16: Option<DenseWeight>,
        inject_bf16: Option<DenseWeight>,
        hidden_size: usize,
        hc_count: usize,
        rank: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        if hidden_size == 0 || hc_count <= 1 || rank == 0 {
            bail!(
                "invalid Qwen4 hyperconnection geometry h={hidden_size} hc={hc_count} rank={rank}"
            );
        }
        let bf16_decode = down_bf16.is_some() || up_bf16.is_some() || inject_bf16.is_some();
        let hc_rstd_k = crate::layers::try_kernel(gpu, "qwen4_hc_fused", "qwen4_hc_rstd");
        let upmix_roof_t128_w4_k =
            match gpu.kernel("qwen4_hc_upmix_roof", "qwen4_hc_upmix_roof_t128_w4") {
                Ok(k)
                    if gpu
                        .set_kernel_max_dynamic_shared_memory(k, UPMIX_ROOF_T128_W4_SMEM_BYTES)
                        .is_ok() =>
                {
                    k
                }
                _ => KernelHandle(0),
            };
        Ok(Self {
            norm,
            down,
            up,
            inject,
            down_bf16,
            up_bf16,
            inject_bf16,
            // A silent fallback to `try_kernel` here would leave a layer on
            // the NVFP4 arm while the load-time receipt claims BF16
            // engaged; a missing kernel under `ATLAS_QWEN4_HC_BF16=1` is a
            // hard load error instead.
            dense_gemv_k: if bf16_decode {
                gpu.kernel("gemv", "dense_gemv_bf16")?
            } else {
                KernelHandle(0)
            },
            group_norm_k: gpu.kernel("qwen4_hyper", "qwen4_hc_group_norm")?,
            silu_div_k: gpu.kernel("qwen4_hyper", "qwen4_hc_silu_div")?,
            mix_k: gpu.kernel("qwen4_hyper", "qwen4_hc_mix")?,
            mix_save_inject_k: crate::layers::try_kernel(
                gpu,
                "qwen4_hyper",
                "qwen4_hc_mix_save_inject",
            ),
            inject_k: gpu.kernel("qwen4_hyper", "qwen4_hc_inject")?,
            save_inject_k: gpu.kernel("qwen4_hyper", "qwen4_hc_save_inject")?,
            inject_saved_k: gpu.kernel("qwen4_hyper", "qwen4_hc_inject_saved")?,
            inject_group_norm_k: crate::layers::try_kernel(
                gpu,
                "qwen4_hyper",
                "qwen4_hc_inject_group_norm",
            ),
            inject_group_norm_v2_k: crate::layers::try_kernel(
                gpu,
                "qwen4_hyper",
                "qwen4_hc_inject_group_norm_v2",
            ),
            stage_mixed_k: gpu.kernel("qwen4_hyper", "qwen4_hc_stage_mixed")?,
            pack_mixed_k: gpu.kernel("qwen4_hyper", "qwen4_hc_pack_mixed")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            gemv_pf_k: if hc_gemv_v2_enabled() {
                crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_pf")
            } else {
                KernelHandle(0)
            },
            up_mix_fused_k: if hc_gemv_v2_enabled() {
                crate::layers::try_kernel(gpu, "w4a16_gemv", "qwen4_hc_up_mix_fused")
            } else {
                KernelHandle(0)
            },
            up_mix_tail_rows_k: if hc_tail_rows_enabled() {
                crate::layers::try_kernel(gpu, "w4a16_gemv", "qwen4_hc_up_mix_tail_rows")
            } else {
                KernelHandle(0)
            },
            up_mix_sw_m5_o2_k: if hc_tail_rows_enabled() {
                crate::layers::try_kernel(gpu, "w4a16_gemv", "qwen4_hc_up_mix_sw_m5_o2")
            } else {
                KernelHandle(0)
            },
            exact_pf_dual_rows_k: if hc_dual_rows_enabled() {
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv",
                    "w4a16_gemv_batch_logits_exact_pf_dual_m8",
                )
            } else {
                KernelHandle(0)
            },
            exact_pf_dual_rows_m10_k: if hc_dual_rows_enabled() {
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv",
                    "w4a16_gemv_batch_logits_exact_pf_dual_m10",
                )
            } else {
                KernelHandle(0)
            },
            exact_pf_dual_rows_m25_k: if hc_dual_rows_enabled() {
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv",
                    "w4a16_gemv_batch_logits_exact_pf_dual_m25",
                )
            } else {
                KernelHandle(0)
            },
            exact_pf_dual_silu_rows_k: if hc_dual_rows_enabled() {
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv",
                    "w4a16_gemv_batch_logits_exact_pf_dual_silu_m8",
                )
            } else {
                KernelHandle(0)
            },
            exact_pf_dual_silu_rows_m10_k: if hc_dual_rows_enabled() {
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv",
                    "w4a16_gemv_batch_logits_exact_pf_dual_silu_m10",
                )
            } else {
                KernelHandle(0)
            },
            exact_pf_dual_silu_rows_m25_k: if hc_dual_rows_enabled() {
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv",
                    "w4a16_gemv_batch_logits_exact_pf_dual_silu_m25",
                )
            } else {
                KernelHandle(0)
            },
            gemv_pf_dual_k: if hc_gemv_v2_enabled() {
                crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_pf_dual")
            } else {
                KernelHandle(0)
            },
            gemv_w32_k: if hc_gemv_v2_enabled() {
                crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_w32")
            } else {
                KernelHandle(0)
            },
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_pipe_k: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe"),
            w4a16_gemm_pipe_m128n128_k: crate::layers::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_pipe_m128n128",
            ),
            coalesced_silu_div_k: CoalescedSiluDivKernels::load(gpu),
            coalesced_dual_silu_div_k: if hc_dual_coalesced_enabled() {
                try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n8_sk8_m32_dual_silu_div")
            } else {
                KernelHandle(0)
            },
            coalesced_dual_n16_v2_silu_div_k: if hc_dual_coalesced_enabled()
                && hc_dual_n32_enabled()
            {
                try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n16_sk4_m16_dual_silu_div_v2")
            } else {
                KernelHandle(0)
            },
            coalesced_dual_n32_silu_div_k: if hc_dual_coalesced_enabled() && hc_dual_n32_enabled() {
                try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n32_sk4_m16_dual_silu_div")
            } else {
                KernelHandle(0)
            },
            coalesced_dual_high_sm_k: if hc_dual_coalesced_enabled() && hc_dual_n32_enabled() {
                DualHighSmKernels::load(gpu)
            } else {
                DualHighSmKernels::empty()
            },
            dual_splitk_scratch: if hc_dual_coalesced_enabled()
                && hc_dual_n32_enabled()
                && atlas_core::device::active_sm_count() > DUAL_SPLITK_HIGH_SM_THRESHOLD
            {
                dual_splitk_scratch_for_backend(gpu)
            } else {
                None
            },
            coalesced_upmix_k: if hc_upmix_coalesced_enabled() {
                CoalescedUpmixKernels::load(gpu)
            } else {
                CoalescedUpmixKernels::empty()
            },
            w4a16_gemv_exact_m4_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m4")?,
            w4a16_gemv_exact_m8_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m8")?,
            w4a16_gemv_exact_m17_k: gpu
                .kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m17")?,
            w4a16_gemv_exact_m32_k: gpu
                .kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m32")?,
            w4a16_gemv_rows_k: gpu
                .kernel("w4a16_gemv", "w4a16_gemv_rows")
                .unwrap_or(KernelHandle(0)),
            w4a16_gemv_sw_m4_k: gpu
                .kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_sw_m4_o2")
                .unwrap_or(KernelHandle(0)),
            w4a16_gemv_sw_m8_k: gpu
                .kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_sw_m8_o1")
                .unwrap_or(KernelHandle(0)),
            w4a16_gemv_sw_m10_k: gpu
                .kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_sw_m10_o1")
                .unwrap_or(KernelHandle(0)),
            w4a16_gemv_sw_m25_k: gpu
                .kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_sw_m25_o1")
                .unwrap_or(KernelHandle(0)),
            hc_rstd_k,
            upmix_roof_t128_w4_k,
            fp8_down_warmed: WarmedRowsMask::new(),
            fp8_up_warmed: WarmedRowsMask::new(),
            hidden_size,
            hc_count,
            rank,
        })
    }

    fn pf_or_base(&self) -> KernelHandle {
        if self.gemv_pf_k.0 != 0 {
            self.gemv_pf_k
        } else {
            self.w4a16_gemv_k
        }
    }

    pub fn residual_width(&self) -> usize {
        self.hidden_size * self.hc_count
    }

    /// Mix one hyperconnection row into a hidden-size core input. The
    /// normalized four-stream row is staged in `residual`; projection
    /// temporaries use arena buffers that are idle between sublayers.
    pub fn prepare_decode(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<(DevicePtr, Option<DevicePtr>)> {
        self.prepare_decode_with_save(hyper, residual, buffers, gpu, eps, stream, true, true)
    }

    /// Same as [`Self::prepare_decode`], but never takes the
    /// `ATLAS_QWEN4_HC_BF16` arm even when it is engaged. For the handful of
    /// callers that process one prefill row at a time through the M=1
    /// mixer (`prepare_prefill_exact_staged`'s tail tile,
    /// `prefill_qwen4_moe_batch`, `prefill_moe_only`, the exact-prefill
    /// checks) — the BF16 arm is qualified for decode only, and this
    /// control has no BF16 prefill arm, so prefill must never silently
    /// change when it is engaged.
    pub fn prepare_decode_nvfp4(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<(DevicePtr, Option<DevicePtr>)> {
        self.prepare_decode_with_save(hyper, residual, buffers, gpu, eps, stream, true, false)
    }

    /// Mix-only single-row 4H → H contraction (NVFP4 precision matching
    /// [`Self::prepare_decode_nvfp4`]): reads `hyper` strictly read-only,
    /// leaves `buffers.residual()` untouched (staging the normalized 4H row in
    /// `buffers.ssm_qkvz()`), skips the `inject` projection, and writes the
    /// H-wide mixed row into `out_mixed`.
    #[allow(clippy::too_many_arguments)]
    pub fn mix_only_decode_nvfp4(
        &self,
        hyper: DevicePtr,
        out_mixed: DevicePtr,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        pending::flush(gpu, stream)?;
        let r = self.residual_width();
        let row_4h_bytes = r * 2;
        anyhow::ensure!(
            buffers.sizes().ssm_qkvz >= row_4h_bytes,
            "Qwen4 mix_only_decode_nvfp4 requires ssm_qkvz >= {row_4h_bytes} B (got {})",
            buffers.sizes().ssm_qkvz
        );
        anyhow::ensure!(
            buffers.sizes().qkv_output >= row_4h_bytes,
            "Qwen4 mix_only_decode_nvfp4 requires qkv_output >= {row_4h_bytes} B (got {})",
            buffers.sizes().qkv_output
        );
        let down_bytes = self.rank * 2;
        let down_out = if buffers.sizes().ssm_ba >= down_bytes {
            buffers.ssm_ba()
        } else {
            anyhow::ensure!(
                buffers.sizes().gate_logits >= down_bytes,
                "Qwen4 mix_only_decode_nvfp4 requires ssm_ba or gate_logits >= {down_bytes} B"
            );
            buffers.gate_logits()
        };
        let norm_buf = buffers.ssm_qkvz();
        KernelLaunch::new(gpu, self.group_norm_k)
            .grid([1, self.hc_count as u32, 1])
            .block([self.hidden_size.min(1024) as u32, 1, 1])
            .arg_ptr(hyper)
            .arg_ptr(self.norm.weight)
            .arg_ptr(norm_buf)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .arg_f32(eps)
            .launch(stream)?;

        ops::w4a16_gemv(
            gpu,
            self.pf_or_base(),
            norm_buf,
            &self.down,
            down_out,
            self.rank as u32,
            r as u32,
            stream,
        )?;
        KernelLaunch::new(gpu, self.silu_div_k)
            .grid([div_ceil(self.rank as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(down_out)
            .arg_u32(self.rank as u32)
            .arg_f32(self.hc_count as f32)
            .launch(stream)?;

        let mix_logits = buffers.qkv_output();
        if self.gemv_w32_k.0 != 0 && self.rank <= 512 && self.rank.is_multiple_of(16) {
            KernelLaunch::new(gpu, self.gemv_w32_k)
                .grid([div_ceil(r as u32, 8), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(down_out)
                .arg_ptr(self.up.weight)
                .arg_ptr(self.up.weight_scale)
                .arg_f32(self.up.weight_scale_2)
                .arg_ptr(mix_logits)
                .arg_u32(r as u32)
                .arg_u32(self.rank as u32)
                .launch(stream)?;
        } else {
            ops::w4a16_gemv(
                gpu,
                self.w4a16_gemv_k,
                down_out,
                &self.up,
                mix_logits,
                r as u32,
                self.rank as u32,
                stream,
            )?;
        }

        KernelLaunch::new(gpu, self.mix_k)
            .grid([1, div_ceil(self.hidden_size as u32, 256), 1])
            .block([256, 1, 1])
            .arg_ptr(norm_buf)
            .arg_ptr(mix_logits)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(out_mixed)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)?;
        Ok(out_mixed)
    }

    /// Mix-only multi-row 4H → H contraction for prefill or batched verify
    /// capture: exact same NVFP4 per-row precision as
    /// [`Self::prepare_decode_nvfp4`] / [`Self::prepare_prefill_exact`],
    /// without modifying `hyper` or `buffers.residual()` and without running
    /// `inject`. Tiles over `num_tokens` in chunks bounded by idle scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn mix_only_rows_nvfp4(
        &self,
        hyper: DevicePtr,
        num_tokens: usize,
        out_mixed: DevicePtr,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        anyhow::ensure!(num_tokens > 0, "empty Qwen4 mix_only_rows_nvfp4");
        if num_tokens == 1 {
            return self.mix_only_decode_nvfp4(hyper, out_mixed, buffers, gpu, eps, stream);
        }
        pending::flush(gpu, stream)?;
        let r = self.residual_width();
        let row_4h_bytes = r * 2;
        let down_row_bytes = self.rank * 2;
        let norm_cap = buffers.sizes().ssm_qkvz / row_4h_bytes;
        let mix_cap = buffers.sizes().qkv_output / row_4h_bytes;
        let (down_buf, down_cap) = if buffers.sizes().gate_logits >= down_row_bytes {
            (
                buffers.gate_logits(),
                buffers.sizes().gate_logits / down_row_bytes,
            )
        } else {
            (buffers.ssm_ba(), buffers.sizes().ssm_ba / down_row_bytes)
        };
        let tile_cap = norm_cap.min(mix_cap).min(down_cap).min(32);
        anyhow::ensure!(
            tile_cap >= 1,
            "Qwen4 mix_only_rows_nvfp4 scratch too small (norm_cap={norm_cap}, mix_cap={mix_cap}, down_cap={down_cap})"
        );

        let exact_kernels = ops::W4a16ExactLmHeadKernels::new(
            self.w4a16_gemv_exact_m4_k,
            self.w4a16_gemv_exact_m8_k,
            self.w4a16_gemv_exact_m17_k,
            self.w4a16_gemv_exact_m32_k,
        )
        .with_rowpar(self.w4a16_gemv_rows_k)
        .with_small_k_sw(self.w4a16_gemv_sw_m4_k, self.w4a16_gemv_sw_m8_k)
        .with_small_k_sw_m10(self.w4a16_gemv_sw_m10_k)
        .with_small_k_sw_m25(self.w4a16_gemv_sw_m25_k);

        let norm_buf = buffers.ssm_qkvz();
        let mix_logits = buffers.qkv_output();
        let mut row_off = 0usize;
        while row_off < num_tokens {
            let chunk = (num_tokens - row_off).min(tile_cap);
            let hyper_chunk = hyper.offset(row_off * row_4h_bytes);
            let out_chunk = out_mixed.offset(row_off * self.hidden_size * 2);
            if chunk == 1 {
                self.mix_only_decode_nvfp4(hyper_chunk, out_chunk, buffers, gpu, eps, stream)?;
                row_off += 1;
                continue;
            }
            KernelLaunch::new(gpu, self.group_norm_k)
                .grid([chunk as u32, self.hc_count as u32, 1])
                .block([self.hidden_size.min(1024) as u32, 1, 1])
                .arg_ptr(hyper_chunk)
                .arg_ptr(self.norm.weight)
                .arg_ptr(norm_buf)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .arg_f32(eps)
                .launch(stream)?;

            ops::w4a16_gemv_batch_logits_exact_with(
                gpu,
                exact_kernels,
                norm_buf,
                &self.down,
                down_buf,
                chunk as u32,
                self.rank as u32,
                r as u32,
                stream,
                false,
            )?;

            KernelLaunch::new(gpu, self.silu_div_k)
                .grid([div_ceil((chunk * self.rank) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(down_buf)
                .arg_u32((chunk * self.rank) as u32)
                .arg_f32(self.hc_count as f32)
                .launch(stream)?;

            ops::w4a16_gemv_batch_logits_exact_with(
                gpu,
                exact_kernels,
                down_buf,
                &self.up,
                mix_logits,
                chunk as u32,
                r as u32,
                self.rank as u32,
                stream,
                false,
            )?;

            KernelLaunch::new(gpu, self.mix_k)
                .grid([chunk as u32, div_ceil(self.hidden_size as u32, 256), 1])
                .block([256, 1, 1])
                .arg_ptr(norm_buf)
                .arg_ptr(mix_logits)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(out_chunk)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .launch(stream)?;

            row_off += chunk;
        }
        Ok(out_mixed)
    }

    /// Same as [`Self::prepare_decode`], returning the in-place `inject_logits`
    /// pointer directly without copying it to the residual tail. Safe for
    /// single-token decode where `inject_decode` consumes the returned pointer
    /// before the next mixer runs.
    pub fn prepare_decode_nosave(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<(DevicePtr, Option<DevicePtr>)> {
        self.prepare_decode_with_save(hyper, residual, buffers, gpu, eps, stream, false, true)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_decode_after_inject(
        &self,
        hyper: DevicePtr,
        core: DevicePtr,
        inject_opt: DevicePtr,
        residual: DevicePtr,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<(DevicePtr, Option<DevicePtr>)> {
        self.prepare_decode_after_inject_with_save(
            hyper, core, inject_opt, residual, buffers, gpu, eps, stream, true, true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_decode_after_inject_nosave(
        &self,
        hyper: DevicePtr,
        core: DevicePtr,
        inject_opt: DevicePtr,
        residual: DevicePtr,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<(DevicePtr, Option<DevicePtr>)> {
        self.prepare_decode_after_inject_with_save(
            hyper, core, inject_opt, residual, buffers, gpu, eps, stream, false, true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_decode_after_inject_with_save(
        &self,
        hyper: DevicePtr,
        core: DevicePtr,
        inject_opt: DevicePtr,
        residual: DevicePtr,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
        save_to_residual: bool,
        allow_bf16: bool,
    ) -> Result<(DevicePtr, Option<DevicePtr>)> {
        pending::flush(gpu, stream)?;
        if self.inject_group_norm_k.0 != 0 && self.hc_count == 4 && self.hidden_size <= 3072 {
            KernelLaunch::new(gpu, self.inject_group_norm_k)
                .grid([1, 1, 1])
                .block([1024, 1, 1])
                .arg_ptr(hyper)
                .arg_ptr(core)
                .arg_ptr(inject_opt)
                .arg_ptr(self.norm.weight)
                .arg_ptr(residual)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .arg_f32(eps)
                .launch(stream)?;
            return self.prepare_decode_post_norm(
                residual,
                buffers,
                gpu,
                stream,
                save_to_residual,
                allow_bf16,
            );
        }
        self.inject_decode(hyper, core, inject_opt, gpu, stream)?;
        self.prepare_decode_with_save(
            hyper,
            residual,
            buffers,
            gpu,
            eps,
            stream,
            save_to_residual,
            allow_bf16,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_decode_with_save(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
        save_to_residual: bool,
        allow_bf16: bool,
    ) -> Result<(DevicePtr, Option<DevicePtr>)> {
        pending::flush(gpu, stream)?;
        KernelLaunch::new(gpu, self.group_norm_k)
            .grid([1, self.hc_count as u32, 1])
            .block([self.hidden_size.min(1024) as u32, 1, 1])
            .arg_ptr(hyper)
            .arg_ptr(self.norm.weight)
            .arg_ptr(residual)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .arg_f32(eps)
            .launch(stream)?;
        self.prepare_decode_post_norm(residual, buffers, gpu, stream, save_to_residual, allow_bf16)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_decode_post_norm(
        &self,
        residual: DevicePtr,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        stream: u64,
        save_to_residual: bool,
        allow_bf16: bool,
    ) -> Result<(DevicePtr, Option<DevicePtr>)> {
        let r = self.residual_width();

        // `scratch` contains live attention metadata (positions, slot map,
        // block table, and sequence lengths) for the whole layer traversal.
        // Reusing its first bytes here silently corrupted paged-attention
        // metadata before the following QSA call. The SSM BA arena is idle in
        // both full-attention and hyperconnection mixing and is amply sized
        // for rank BF16 values plus hc_count injection logits.
        let down_out = buffers.ssm_ba();
        // The fixed 2 KiB offset is aligned and exceeds rank=320 BF16.
        let inject_logits = buffers.ssm_ba().offset(2048);
        if allow_bf16 && self.dense_gemv_k.0 != 0 {
            return self.prepare_decode_post_norm_bf16(
                residual,
                down_out,
                inject_logits,
                buffers,
                gpu,
                stream,
                save_to_residual,
            );
        }
        // Down and inject read the same normed input with the same K and write
        // disjoint buffers, so one launch computes both (each output exactly as
        // its own w4a16_gemv would); the inject no longer runs after the up GEMV.
        let fused_inject = self.gemv_pf_dual_k.0 != 0 && self.inject.is_some();
        if let (true, Some(weight)) = (fused_inject, self.inject.as_ref()) {
            KernelLaunch::new(gpu, self.gemv_pf_dual_k)
                .grid([
                    div_ceil(self.rank as u32, 4) + div_ceil(self.hc_count as u32, 4),
                    1,
                    1,
                ])
                .block([256, 1, 1])
                .arg_ptr(residual)
                .arg_ptr(self.down.weight)
                .arg_ptr(self.down.weight_scale)
                .arg_f32(self.down.weight_scale_2)
                .arg_ptr(down_out)
                .arg_u32(self.rank as u32)
                .arg_ptr(weight.weight)
                .arg_ptr(weight.weight_scale)
                .arg_f32(weight.weight_scale_2)
                .arg_ptr(inject_logits)
                .arg_u32(self.hc_count as u32)
                .arg_u32(r as u32)
                .launch(stream)?;
        } else {
            ops::w4a16_gemv(
                gpu,
                self.pf_or_base(),
                residual,
                &self.down,
                down_out,
                self.rank as u32,
                r as u32,
                stream,
            )?;
            if let Some(ref weight) = self.inject {
                ops::w4a16_gemv(
                    gpu,
                    self.pf_or_base(),
                    residual,
                    weight,
                    inject_logits,
                    self.hc_count as u32,
                    r as u32,
                    stream,
                )?;
            }
        }
        let fused_tail = self.up_mix_fused_k.0 != 0
            && self.rank <= 512
            && self.rank.is_multiple_of(16)
            && matches!(self.hc_count, 1 | 2 | 4 | 8);
        if fused_tail {
            // silu_div + up + mix in one launch, byte-identical to the three
            // kernels below (fn-decode hc harness).
            let mixed = buffers.norm_output();
            let per_block = 8 / self.hc_count as u32;
            KernelLaunch::new(gpu, self.up_mix_fused_k)
                .grid([div_ceil(self.hidden_size as u32, per_block), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(down_out)
                .arg_ptr(self.up.weight)
                .arg_ptr(self.up.weight_scale)
                .arg_f32(self.up.weight_scale_2)
                .arg_ptr(residual)
                .arg_ptr(inject_logits)
                .arg_ptr(mixed)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .arg_u32(self.rank as u32)
                .arg_f32(self.hc_count as f32)
                .launch(stream)?;
            let out_inject = if self.inject.is_some() {
                if save_to_residual {
                    let saved_inject = residual.offset((r - self.hc_count) * 2);
                    gpu.copy_d2d_async(inject_logits, saved_inject, self.hc_count * 2, stream)?;
                    Some(saved_inject)
                } else {
                    Some(inject_logits)
                }
            } else {
                None
            };
            return Ok((mixed, out_inject));
        }
        KernelLaunch::new(gpu, self.silu_div_k)
            .grid([div_ceil(self.rank as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(down_out)
            .arg_u32(self.rank as u32)
            .arg_f32(self.hc_count as f32)
            .launch(stream)?;

        let mix_logits = buffers.qkv_output();
        if self.gemv_w32_k.0 != 0 && self.rank <= 512 && self.rank.is_multiple_of(16) {
            KernelLaunch::new(gpu, self.gemv_w32_k)
                .grid([div_ceil(r as u32, 8), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(down_out)
                .arg_ptr(self.up.weight)
                .arg_ptr(self.up.weight_scale)
                .arg_f32(self.up.weight_scale_2)
                .arg_ptr(mix_logits)
                .arg_u32(r as u32)
                .arg_u32(self.rank as u32)
                .launch(stream)?;
        } else {
            ops::w4a16_gemv(
                gpu,
                self.w4a16_gemv_k,
                down_out,
                &self.up,
                mix_logits,
                r as u32,
                self.rank as u32,
                stream,
            )?;
        }

        let mixed = buffers.norm_output();
        KernelLaunch::new(gpu, self.mix_k)
            .grid([1, div_ceil(self.hidden_size as u32, 256), 1])
            .block([256, 1, 1])
            .arg_ptr(residual)
            .arg_ptr(mix_logits)
            .arg_ptr(inject_logits)
            .arg_ptr(mixed)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)?;
        let out_inject = if self.inject.is_some() {
            if save_to_residual {
                let saved_inject = residual.offset((r - self.hc_count) * 2);
                gpu.copy_d2d_async(inject_logits, saved_inject, self.hc_count * 2, stream)?;
                Some(saved_inject)
            } else {
                Some(inject_logits)
            }
        } else {
            None
        };
        Ok((mixed, out_inject))
    }

    /// `ATLAS_QWEN4_HC_BF16=1` M=1 decode arm: `down`/`up`/`inject` on the
    /// loader's retained BF16 dense copies via `dense_gemv`, instead of the
    /// NVFP4 GEMVs above. Not used by any fused/batched-verify tail — those
    /// routes are refused at load (`qwen4_hc_bf16::admit_launch`), so this
    /// is reached only for ordinary single-row decode. `group_norm`,
    /// `silu_div` and `mix` are shared with the NVFP4 arm unchanged; only
    /// the GEMV calls differ.
    #[allow(clippy::too_many_arguments)]
    fn prepare_decode_post_norm_bf16(
        &self,
        residual: DevicePtr,
        down_out: DevicePtr,
        inject_logits: DevicePtr,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        stream: u64,
        save_to_residual: bool,
    ) -> Result<(DevicePtr, Option<DevicePtr>)> {
        let r = self.residual_width();
        let down_bf16 = self
            .down_bf16
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("QWEN4_HC_BF16 decode reached with no down_bf16"))?;
        ops::dense_gemv(
            gpu,
            self.dense_gemv_k,
            residual,
            down_bf16,
            down_out,
            self.rank as u32,
            r as u32,
            stream,
        )?;
        if let Some(weight) = self.inject_bf16.as_ref() {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                residual,
                weight,
                inject_logits,
                self.hc_count as u32,
                r as u32,
                stream,
            )?;
        }

        KernelLaunch::new(gpu, self.silu_div_k)
            .grid([div_ceil(self.rank as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(down_out)
            .arg_u32(self.rank as u32)
            .arg_f32(self.hc_count as f32)
            .launch(stream)?;

        let mix_logits = buffers.qkv_output();
        let up_bf16 = self
            .up_bf16
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("QWEN4_HC_BF16 decode reached with no up_bf16"))?;
        ops::dense_gemv(
            gpu,
            self.dense_gemv_k,
            down_out,
            up_bf16,
            mix_logits,
            r as u32,
            self.rank as u32,
            stream,
        )?;

        let mixed = buffers.norm_output();
        KernelLaunch::new(gpu, self.mix_k)
            .grid([1, div_ceil(self.hidden_size as u32, 256), 1])
            .block([256, 1, 1])
            .arg_ptr(residual)
            .arg_ptr(mix_logits)
            .arg_ptr(inject_logits)
            .arg_ptr(mixed)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)?;
        let out_inject = if self.inject_bf16.is_some() {
            if save_to_residual {
                let saved_inject = residual.offset((r - self.hc_count) * 2);
                gpu.copy_d2d_async(inject_logits, saved_inject, self.hc_count * 2, stream)?;
                Some(saved_inject)
            } else {
                Some(inject_logits)
            }
        } else {
            None
        };
        Ok((mixed, out_inject))
    }

    pub fn inject_decode(
        &self,
        hyper: DevicePtr,
        core: DevicePtr,
        inject: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        pending::flush(gpu, stream)?;
        KernelLaunch::new(gpu, self.inject_k)
            .grid([1, div_ceil(self.residual_width() as u32, 256), 1])
            .block([256, 1, 1])
            .arg_ptr(hyper)
            .arg_ptr(core)
            .arg_ptr(inject)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)
    }

    /// Inject a contiguous token tile in one launch. The CUDA primitive is
    /// already token-indexed; decode historically launched only grid.x=1.
    pub fn inject_batched(
        &self,
        hyper: DevicePtr,
        core: DevicePtr,
        inject: DevicePtr,
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        pending::flush(gpu, stream)?;
        KernelLaunch::new(gpu, self.inject_k)
            .grid([
                num_tokens as u32,
                div_ceil(self.residual_width() as u32, 256),
                1,
            ])
            .block([256, 1, 1])
            .arg_ptr(hyper)
            .arg_ptr(core)
            .arg_ptr(inject)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)
    }

    /// Inject a full prompt tile using scales scattered into each residual
    /// row by [`prepare_prefill`].
    pub fn inject_saved_batched(
        &self,
        hyper: DevicePtr,
        core: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        pending::flush(gpu, stream)?;
        KernelLaunch::new(gpu, self.inject_saved_k)
            .grid([
                num_tokens as u32,
                div_ceil(self.residual_width() as u32, 256),
                1,
            ])
            .block([256, 1, 1])
            .arg_ptr(hyper)
            .arg_ptr(core)
            .arg_ptr(residual)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)
    }

    /// Full-prompt hyperconnection preparation using NVFP4 GEMM for all
    /// three projections. This is the throughput-owned prefill path; the
    /// small exact-GEMV family remains the decode/speculative path.
    pub fn prepare_prefill(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        pending::flush(gpu, stream)?;
        anyhow::ensure!(num_tokens > 0, "empty Qwen4 hyperconnection prefill");
        let r = self.residual_width();
        let down_bytes = num_tokens * self.rank * 2;
        let inject_bytes = num_tokens * self.hc_count * 2;
        anyhow::ensure!(
            down_bytes + inject_bytes <= buffers.sizes().gate_logits,
            "Qwen4 prefill hyper scratch exceeds gate-logit arena"
        );
        anyhow::ensure!(
            num_tokens * r * 2 <= buffers.sizes().qkv_output,
            "Qwen4 prefill hyper logits exceed QKV arena"
        );
        anyhow::ensure!(
            num_tokens * self.hidden_size * 2 <= buffers.sizes().norm_output,
            "Qwen4 prefill mixed rows exceed norm arena"
        );
        KernelLaunch::new(gpu, self.group_norm_k)
            .grid([num_tokens as u32, self.hc_count as u32, 1])
            .block([self.hidden_size.min(1024) as u32, 1, 1])
            .arg_ptr(hyper)
            .arg_ptr(self.norm.weight)
            .arg_ptr(residual)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .arg_f32(eps)
            .launch(stream)?;

        // The BA arena is sized for the model's 64-wide recurrent gates and
        // is too small for M*rank at long prompt lengths. Gate logits are idle
        // until the MoE and scale linearly with max prefill tokens.
        let down_out = buffers.gate_logits();
        ops::w4a16_gemm(
            gpu,
            self.w4a16_gemm_k,
            residual,
            &self.down,
            down_out,
            num_tokens as u32,
            self.rank as u32,
            r as u32,
            stream,
        )?;
        KernelLaunch::new(gpu, self.silu_div_k)
            .grid([div_ceil((num_tokens * self.rank) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(down_out)
            .arg_u32((num_tokens * self.rank) as u32)
            .arg_f32(self.hc_count as f32)
            .launch(stream)?;

        let mix_logits = buffers.qkv_output();
        let inject_logits = down_out.offset(down_bytes);
        ops::w4a16_gemm(
            gpu,
            self.w4a16_gemm_k,
            down_out,
            &self.up,
            mix_logits,
            num_tokens as u32,
            r as u32,
            self.rank as u32,
            stream,
        )?;
        if let Some(ref weight) = self.inject {
            ops::w4a16_gemm(
                gpu,
                self.w4a16_gemm_k,
                residual,
                weight,
                inject_logits,
                num_tokens as u32,
                self.hc_count as u32,
                r as u32,
                stream,
            )?;
        } else {
            gpu.memset_async(inject_logits, 0, inject_bytes, stream)?;
        }
        let mixed = buffers.norm_output();
        KernelLaunch::new(gpu, self.mix_k)
            .grid([num_tokens as u32, div_ceil(self.hidden_size as u32, 256), 1])
            .block([256, 1, 1])
            .arg_ptr(residual)
            .arg_ptr(mix_logits)
            .arg_ptr(inject_logits)
            .arg_ptr(mixed)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.save_inject_k)
            .grid([div_ceil((num_tokens * self.hc_count) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(residual)
            .arg_ptr(inject_logits)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .arg_u32(num_tokens as u32)
            .launch(stream)?;
        Ok(mixed)
    }

    /// Arbitrary-length, byte-exact prefill preparation. Projection weights
    /// are streamed once per 32 prompt rows, and mixed rows are staged in the
    /// residual workspace before one contiguous pack for the core kernels.
    pub fn prepare_prefill_exact(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prepare_prefill_exact_staged(
            hyper, residual, num_tokens, true, buffers, gpu, eps, stream,
        )
    }

    /// [`Self::prepare_prefill_exact`] for a consumer that only reads the
    /// returned packed rows: with `ATLAS_QWEN4_PREFILL_MHC_FUSED=nostage` the
    /// full-row fast path skips copying them into the residual row fronts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_prefill_exact_packed(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let stage = !prefill_inject::selection()?.nostage;
        self.prepare_prefill_exact_staged(
            hyper, residual, num_tokens, stage, buffers, gpu, eps, stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_prefill_exact_staged(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        stage: bool,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.validate_prefill_exact(hyper, residual, num_tokens, buffers, eps)?;
        if prefill_fused::level() != 0 && num_tokens >= 64 {
            // HC_FUSED recomputes the normalized rows from hyper and rstd; a
            // pending injection is applied by its rstd pass.
            let p = self.take_rstd_fusable_pending(hyper, residual, num_tokens, gpu, stream)?;
            return self.prepare_prefill_fused(
                hyper,
                residual,
                num_tokens,
                p.as_ref(),
                stage,
                buffers,
                gpu,
                eps,
                stream,
            );
        }
        if let Some(p) = self.take_fusable_pending(hyper, residual, num_tokens, gpu, stream)? {
            return self.prepare_prefill_fast(
                p.hyper,
                p.residual,
                p.rows,
                Some(&p),
                stage,
                buffers,
                gpu,
                eps,
                stream,
            );
        }
        if crate::layers::qwen4_fast_proj::selection()?.hc {
            return self.prepare_prefill_fast(
                hyper, residual, num_tokens, None, stage, buffers, gpu, eps, stream,
            );
        }
        crate::layers::qwen4_prefill_native::note_refused(
            crate::layers::qwen4_prefill_native::Family::Hc,
            "hc_exact_tile_route",
            num_tokens,
        );
        let r = self.residual_width();
        let mut start = 0;
        while start < num_tokens {
            let tile = (num_tokens - start).min(32);
            let hyper_tile = hyper.offset(start * r * 2);
            let residual_tile = residual.offset(start * r * 2);
            let mixed = if tile == 1 {
                self.prepare_decode_nvfp4(hyper_tile, residual_tile, buffers, gpu, eps, stream)?
                    .0
            } else {
                self.prepare_batched(hyper_tile, residual_tile, tile, buffers, gpu, eps, stream)?
            };
            KernelLaunch::new(gpu, self.stage_mixed_k)
                .grid([div_ceil((tile * self.hidden_size) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(mixed)
                .arg_ptr(residual)
                .arg_u32(start as u32)
                .arg_u32(tile as u32)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .launch(stream)?;
            start += tile;
        }
        let mixed = buffers.norm_output();
        KernelLaunch::new(gpu, self.pack_mixed_k)
            .grid([div_ceil((num_tokens * self.hidden_size) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(residual)
            .arg_ptr(mixed)
            .arg_u32(num_tokens as u32)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)?;
        self.check_prefill_exact(hyper, residual, num_tokens, buffers, gpu, eps, stream)?;
        Ok(mixed)
    }

    /// Mix up to 32 contiguous hyperconnection rows while streaming each
    /// projection weight once. The exact M4 kernel preserves K1 lane
    /// ownership, operation order, reduction tree, and BF16 rounding, unlike
    /// the numerically different K8 batch2/batch3 or tensor-core kernels.
    pub fn prepare_batched(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prepare_batched_into(hyper, residual, num_tokens, buffers, None, gpu, eps, stream)
    }

    /// Read-only capability proof for the native target-only C2-C256 lane
    /// (`QWEN4_NATIVE_MAX_ROWS`). `prepare_batched_into` requires a real exact
    /// tier (it has no implicit K1 fallback), so resolve that tier and every
    /// selected tail primitive before an earlier layer can mutate request-owned
    /// state.
    pub(crate) fn preflight_qwen4_native_batched(
        &self,
        rows: usize,
        saved_injection: bool,
    ) -> Result<()> {
        anyhow::ensure!(
            (2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&rows),
            "native Qwen4 hyperconnection admits only C2-C{}, got C{rows}",
            crate::layers::QWEN4_NATIVE_MAX_ROWS
        );
        anyhow::ensure!(
            self.hidden_size > 0
                && self.hc_count == 4
                && self.rank > 0
                && !self.norm.weight.is_null()
                && !self.down.is_null()
                && !self.up.is_null(),
            "native Qwen4 hyperconnection geometry or weights are incomplete"
        );
        let exact_kernels = ops::W4a16ExactLmHeadKernels::new(
            self.w4a16_gemv_exact_m4_k,
            self.w4a16_gemv_exact_m8_k,
            self.w4a16_gemv_exact_m17_k,
            self.w4a16_gemv_exact_m32_k,
        );
        anyhow::ensure!(
            matches!(
                exact_kernels.route_for_rows(rows as u32),
                Some(ops::ExactLmHeadRoute::Exact(_))
            ),
            "native Qwen4 hyperconnection is missing its exact C{rows} projection tier"
        );
        anyhow::ensure!(
            self.group_norm_k.0 != 0,
            "native Qwen4 hyperconnection is missing group normalization"
        );
        if self.tail_rows_admits(rows) {
            anyhow::ensure!(
                self.up_mix_tail_rows_k.0 != 0,
                "native Qwen4 hyperconnection selected a missing exact tail kernel"
            );
        } else {
            anyhow::ensure!(
                self.silu_div_k.0 != 0 && self.mix_k.0 != 0,
                "native Qwen4 hyperconnection baseline tail kernels are incomplete"
            );
            if self.inject.is_some() {
                anyhow::ensure!(
                    self.save_inject_k.0 != 0,
                    "native Qwen4 hyperconnection is missing saved-injection staging"
                );
            }
        }
        if saved_injection {
            anyhow::ensure!(
                self.inject.as_ref().is_some_and(|weight| !weight.is_null())
                    && self.inject_saved_k.0 != 0,
                "native Qwen4 layer hyperconnection requires live saved-injection weights and kernel"
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_batched_after_inject(
        &self,
        hyper: DevicePtr,
        core: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prepare_batched_into_with_inject(
            hyper,
            Some(core),
            residual,
            num_tokens,
            buffers,
            None,
            gpu,
            eps,
            stream,
        )
    }

    /// Fused `inject_saved_batched + prepare_batched_into` writing the mixed
    /// rows into `out_mixed` (when `Some`) or `buffers.norm_output()` (`None`).
    /// Used across inter-layer (`MLP -> next layer Attn/SSM`) boundaries so the
    /// preceding layer's trailing residual injection fuses directly into this
    /// hyperconnection's `group_norm` via `qwen4_hc_inject_group_norm_v2`
    /// (`0-ULP` bit-identical, 100% CUDA-graph safe).
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_batched_into_after_inject(
        &self,
        hyper: DevicePtr,
        core: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        buffers: &BufferArena,
        out_mixed: Option<DevicePtr>,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prepare_batched_into_with_inject(
            hyper,
            Some(core),
            residual,
            num_tokens,
            buffers,
            out_mixed,
            gpu,
            eps,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_batched_into(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        buffers: &BufferArena,
        out_mixed: Option<DevicePtr>,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prepare_batched_into_with_inject(
            hyper, None, residual, num_tokens, buffers, out_mixed, gpu, eps, stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_batched_into_with_inject(
        &self,
        hyper: DevicePtr,
        core_to_inject: Option<DevicePtr>,
        residual: DevicePtr,
        num_tokens: usize,
        buffers: &BufferArena,
        out_mixed: Option<DevicePtr>,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        pending::flush(gpu, stream)?;
        anyhow::ensure!(
            (2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&num_tokens),
            "Qwen4 hyperconnection batching supports K=2..{}, got {num_tokens}",
            crate::layers::QWEN4_NATIVE_MAX_ROWS
        );
        let r = self.residual_width();
        let down_bytes = num_tokens * self.rank * 2;
        let inject_bytes = num_tokens * self.hc_count * 2;
        anyhow::ensure!(
            down_bytes + inject_bytes <= buffers.sizes().ssm_ba,
            "Qwen4 batched hyper scratch exceeds SSM BA arena"
        );
        anyhow::ensure!(
            num_tokens * r * 2 <= buffers.sizes().qkv_output,
            "Qwen4 batched hyper logits exceed QKV arena"
        );
        if out_mixed.is_none() {
            anyhow::ensure!(
                num_tokens * self.hidden_size * 2 <= buffers.sizes().norm_output,
                "Qwen4 batched mixed rows exceed norm arena"
            );
        }

        let tc_component = if self.inject.is_some() {
            BatchedVerifyTcComponent::HcPrepare
        } else {
            BatchedVerifyTcComponent::FinalMixer
        };
        if batched_verify_tc_dense_scope_active()
            && batched_verify_tc_component_active(tc_component, num_tokens, false)
            && let Some(mixed) = self.try_prepare_batched_tc_dense(
                hyper,
                core_to_inject,
                residual,
                num_tokens,
                buffers,
                out_mixed,
                gpu,
                eps,
                stream,
            )?
        {
            return Ok(mixed);
        }

        let _ = self.launch_inject_and_group_norm(
            hyper,
            core_to_inject,
            residual,
            DevicePtr::NULL,
            num_tokens,
            gpu,
            eps,
            stream,
        )?;

        let down_out = buffers.ssm_ba();
        let inject_logits = down_out.offset(down_bytes);
        let exact_kernels = ops::W4a16ExactLmHeadKernels::new(
            self.w4a16_gemv_exact_m4_k,
            self.w4a16_gemv_exact_m8_k,
            self.w4a16_gemv_exact_m17_k,
            self.w4a16_gemv_exact_m32_k,
        )
        .with_rowpar(self.w4a16_gemv_rows_k)
        .with_small_k_sw(self.w4a16_gemv_sw_m4_k, self.w4a16_gemv_sw_m8_k)
        .with_small_k_sw_m10(self.w4a16_gemv_sw_m10_k)
        .with_small_k_sw_m25(self.w4a16_gemv_sw_m25_k);
        // Down and inject read the same rows with the same K and write
        // disjoint buffers, so the dual launch computes both, each output with
        // the exact tiers' arithmetic; the inject then no longer waits for the
        // up projection.
        let det_batch = crate::layers::deterministic_batch::deterministic_batch_enabled();
        let dual_inject = self
            .inject
            .as_ref()
            .filter(|_| !det_batch && self.dual_rows_admits(num_tokens, r));
        let use_sw_tail_rows = !det_batch && self.sw_tail_rows_admits(num_tokens);
        let use_tail_rows = !det_batch && !use_sw_tail_rows && self.tail_rows_admits(num_tokens);
        let fused_dual_silu = dual_inject.is_some() && self.exact_pf_dual_silu_rows_k.0 != 0;
        if let Some(weight) = dual_inject {
            HC_DUAL_ROWS_LAUNCHED.call_once(|| {
                tracing::info!(
                    "HC_DUAL_ROWS_ENGAGED selector=ATLAS_QWEN4_HC_DUAL_ROWS launched rows={num_tokens} rank={} hc={}",
                    self.rank,
                    self.hc_count
                );
            });
            if (11..=25).contains(&num_tokens)
                && ((!fused_dual_silu && self.exact_pf_dual_rows_m25_k.0 != 0)
                    || (fused_dual_silu && self.exact_pf_dual_silu_rows_m25_k.0 != 0))
            {
                let grid_x = div_ceil(self.rank as u32, 4) + div_ceil(self.hc_count as u32, 4);
                if fused_dual_silu {
                    KernelLaunch::new(gpu, self.exact_pf_dual_silu_rows_m25_k)
                        .grid([grid_x, 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(residual)
                        .arg_ptr(self.down.weight)
                        .arg_ptr(self.down.weight_scale)
                        .arg_f32(self.down.weight_scale_2)
                        .arg_ptr(down_out)
                        .arg_u32(self.rank as u32)
                        .arg_ptr(weight.weight)
                        .arg_ptr(weight.weight_scale)
                        .arg_f32(weight.weight_scale_2)
                        .arg_ptr(inject_logits)
                        .arg_u32(self.hc_count as u32)
                        .arg_u32(num_tokens as u32)
                        .arg_u32(r as u32)
                        .arg_f32(self.hc_count as f32)
                        .launch(stream)?;
                } else {
                    KernelLaunch::new(gpu, self.exact_pf_dual_rows_m25_k)
                        .grid([grid_x, 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(residual)
                        .arg_ptr(self.down.weight)
                        .arg_ptr(self.down.weight_scale)
                        .arg_f32(self.down.weight_scale_2)
                        .arg_ptr(down_out)
                        .arg_u32(self.rank as u32)
                        .arg_ptr(weight.weight)
                        .arg_ptr(weight.weight_scale)
                        .arg_f32(weight.weight_scale_2)
                        .arg_ptr(inject_logits)
                        .arg_u32(self.hc_count as u32)
                        .arg_u32(num_tokens as u32)
                        .arg_u32(r as u32)
                        .launch(stream)?;
                }
            } else if num_tokens > 8
                && ((!fused_dual_silu && self.exact_pf_dual_rows_m10_k.0 != 0)
                    || (fused_dual_silu && self.exact_pf_dual_silu_rows_m10_k.0 != 0))
            {
                let grid_x = div_ceil(self.rank as u32, 4) + div_ceil(self.hc_count as u32, 4);
                let grid_y = div_ceil(num_tokens as u32, 10);
                if fused_dual_silu {
                    KernelLaunch::new(gpu, self.exact_pf_dual_silu_rows_m10_k)
                        .grid([grid_x, grid_y, 1])
                        .block([256, 1, 1])
                        .arg_ptr(residual)
                        .arg_ptr(self.down.weight)
                        .arg_ptr(self.down.weight_scale)
                        .arg_f32(self.down.weight_scale_2)
                        .arg_ptr(down_out)
                        .arg_u32(self.rank as u32)
                        .arg_ptr(weight.weight)
                        .arg_ptr(weight.weight_scale)
                        .arg_f32(weight.weight_scale_2)
                        .arg_ptr(inject_logits)
                        .arg_u32(self.hc_count as u32)
                        .arg_u32(num_tokens as u32)
                        .arg_u32(r as u32)
                        .arg_f32(self.hc_count as f32)
                        .launch(stream)?;
                } else {
                    KernelLaunch::new(gpu, self.exact_pf_dual_rows_m10_k)
                        .grid([grid_x, grid_y, 1])
                        .block([256, 1, 1])
                        .arg_ptr(residual)
                        .arg_ptr(self.down.weight)
                        .arg_ptr(self.down.weight_scale)
                        .arg_f32(self.down.weight_scale_2)
                        .arg_ptr(down_out)
                        .arg_u32(self.rank as u32)
                        .arg_ptr(weight.weight)
                        .arg_ptr(weight.weight_scale)
                        .arg_f32(weight.weight_scale_2)
                        .arg_ptr(inject_logits)
                        .arg_u32(self.hc_count as u32)
                        .arg_u32(num_tokens as u32)
                        .arg_u32(r as u32)
                        .launch(stream)?;
                }
            } else {
                let has_m10 = if fused_dual_silu {
                    self.exact_pf_dual_silu_rows_m10_k.0 != 0
                } else {
                    self.exact_pf_dual_rows_m10_k.0 != 0
                };
                let max_chunk = if has_m10 { 10 } else { 8 };
                let mut row_off = 0usize;
                while row_off < num_tokens {
                    let chunk = (num_tokens - row_off).min(max_chunk);
                    let res_chunk = residual.offset(row_off * r * 2);
                    let down_chunk = down_out.offset(row_off * self.rank * 2);
                    let inj_chunk = inject_logits.offset(row_off * self.hc_count * 2);
                    if fused_dual_silu {
                        let k = if chunk > 8 && self.exact_pf_dual_silu_rows_m10_k.0 != 0 {
                            self.exact_pf_dual_silu_rows_m10_k
                        } else {
                            self.exact_pf_dual_silu_rows_k
                        };
                        KernelLaunch::new(gpu, k)
                            .grid([
                                div_ceil(self.rank as u32, 4) + div_ceil(self.hc_count as u32, 4),
                                1,
                                1,
                            ])
                            .block([256, 1, 1])
                            .arg_ptr(res_chunk)
                            .arg_ptr(self.down.weight)
                            .arg_ptr(self.down.weight_scale)
                            .arg_f32(self.down.weight_scale_2)
                            .arg_ptr(down_chunk)
                            .arg_u32(self.rank as u32)
                            .arg_ptr(weight.weight)
                            .arg_ptr(weight.weight_scale)
                            .arg_f32(weight.weight_scale_2)
                            .arg_ptr(inj_chunk)
                            .arg_u32(self.hc_count as u32)
                            .arg_u32(chunk as u32)
                            .arg_u32(r as u32)
                            .arg_f32(self.hc_count as f32)
                            .launch(stream)?;
                    } else {
                        let k = if chunk > 8 && self.exact_pf_dual_rows_m10_k.0 != 0 {
                            self.exact_pf_dual_rows_m10_k
                        } else {
                            self.exact_pf_dual_rows_k
                        };
                        KernelLaunch::new(gpu, k)
                            .grid([
                                div_ceil(self.rank as u32, 4) + div_ceil(self.hc_count as u32, 4),
                                1,
                                1,
                            ])
                            .block([256, 1, 1])
                            .arg_ptr(res_chunk)
                            .arg_ptr(self.down.weight)
                            .arg_ptr(self.down.weight_scale)
                            .arg_f32(self.down.weight_scale_2)
                            .arg_ptr(down_chunk)
                            .arg_u32(self.rank as u32)
                            .arg_ptr(weight.weight)
                            .arg_ptr(weight.weight_scale)
                            .arg_f32(weight.weight_scale_2)
                            .arg_ptr(inj_chunk)
                            .arg_u32(self.hc_count as u32)
                            .arg_u32(chunk as u32)
                            .arg_u32(r as u32)
                            .launch(stream)?;
                    }
                    row_off += chunk;
                }
            }
        } else {
            ops::w4a16_gemv_batch_logits_exact_with(
                gpu,
                exact_kernels,
                residual,
                &self.down,
                down_out,
                num_tokens as u32,
                self.rank as u32,
                r as u32,
                stream,
                false,
            )?;
        }
        let mixed = out_mixed.unwrap_or_else(|| buffers.norm_output());
        if use_sw_tail_rows {
            if !fused_dual_silu {
                KernelLaunch::new(gpu, self.silu_div_k)
                    .grid([div_ceil((num_tokens * self.rank) as u32, 256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(down_out)
                    .arg_u32((num_tokens * self.rank) as u32)
                    .arg_f32(self.hc_count as f32)
                    .launch(stream)?;
            }
            if let Some(ref weight) = self.inject
                && dual_inject.is_none()
            {
                ops::w4a16_gemv_batch_logits_exact_with(
                    gpu,
                    exact_kernels,
                    residual,
                    weight,
                    inject_logits,
                    num_tokens as u32,
                    self.hc_count as u32,
                    r as u32,
                    stream,
                    false,
                )?;
            }
            let inject_ptr = if self.inject.is_some() {
                inject_logits
            } else {
                DevicePtr::NULL
            };
            KernelLaunch::new(gpu, self.up_mix_sw_m5_o2_k)
                .grid([(self.hidden_size / 4) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(down_out)
                .arg_ptr(self.up.weight)
                .arg_ptr(self.up.weight_scale)
                .arg_f32(self.up.weight_scale_2)
                .arg_ptr(residual)
                .arg_ptr(inject_ptr)
                .arg_ptr(mixed)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .arg_u32(self.rank as u32)
                .arg_u32(num_tokens as u32)
                .launch(stream)?;
            return Ok(mixed);
        }
        if use_tail_rows {
            // Inject first: the tail transforms its raw logits in place.
            if let Some(ref weight) = self.inject
                && dual_inject.is_none()
            {
                ops::w4a16_gemv_batch_logits_exact_with(
                    gpu,
                    exact_kernels,
                    residual,
                    weight,
                    inject_logits,
                    num_tokens as u32,
                    self.hc_count as u32,
                    r as u32,
                    stream,
                    false,
                )?;
            }
            KernelLaunch::new(gpu, self.up_mix_tail_rows_k)
                .grid([
                    (self.hidden_size / HC_TAIL_HIDDEN_PER_BLOCK) as u32,
                    div_ceil(num_tokens as u32, HC_TAIL_MAX_ROWS as u32),
                    1,
                ])
                .block([256, 1, 1])
                .arg_ptr(down_out)
                .arg_ptr(self.up.weight)
                .arg_ptr(self.up.weight_scale)
                .arg_f32(self.up.weight_scale_2)
                .arg_ptr(residual)
                .arg_ptr(if self.inject.is_some() {
                    inject_logits
                } else {
                    DevicePtr::NULL
                })
                .arg_ptr(mixed)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .arg_u32(self.rank as u32)
                .arg_f32(if fused_dual_silu {
                    0.0f32
                } else {
                    self.hc_count as f32
                })
                .arg_u32(num_tokens as u32)
                .launch(stream)?;
            return Ok(mixed);
        }
        if !fused_dual_silu {
            KernelLaunch::new(gpu, self.silu_div_k)
                .grid([div_ceil((num_tokens * self.rank) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(down_out)
                .arg_u32((num_tokens * self.rank) as u32)
                .arg_f32(self.hc_count as f32)
                .launch(stream)?;
        }

        let mix_logits = buffers.qkv_output();
        let project = |input: DevicePtr,
                       weight: &QuantizedWeight,
                       output: DevicePtr,
                       n: u32,
                       k: u32|
         -> Result<()> {
            ops::w4a16_gemv_batch_logits_exact_with(
                gpu,
                exact_kernels,
                input,
                weight,
                output,
                num_tokens as u32,
                n,
                k,
                stream,
                false,
            )
        };
        project(down_out, &self.up, mix_logits, r as u32, self.rank as u32)?;
        if let Some(ref weight) = self.inject
            && dual_inject.is_none()
        {
            project(
                residual,
                weight,
                inject_logits,
                self.hc_count as u32,
                r as u32,
            )?;
        }

        if self.inject.is_some()
            && self.mix_save_inject_k.0 != 0
            && self.hc_count <= 256
            && (self.hidden_size.is_multiple_of(256) || self.hidden_size % 256 >= self.hc_count)
        {
            KernelLaunch::new(gpu, self.mix_save_inject_k)
                .grid([num_tokens as u32, div_ceil(self.hidden_size as u32, 256), 1])
                .block([256, 1, 1])
                .arg_ptr(residual)
                .arg_ptr(mix_logits)
                .arg_ptr(inject_logits)
                .arg_ptr(mixed)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .launch(stream)?;
        } else {
            KernelLaunch::new(gpu, self.mix_k)
                .grid([num_tokens as u32, div_ceil(self.hidden_size as u32, 256), 1])
                .block([256, 1, 1])
                .arg_ptr(residual)
                .arg_ptr(mix_logits)
                .arg_ptr(inject_logits)
                .arg_ptr(mixed)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .launch(stream)?;
            if self.inject.is_some() {
                KernelLaunch::new(gpu, self.save_inject_k)
                    .grid([div_ceil((num_tokens * self.hc_count) as u32, 256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(residual)
                    .arg_ptr(inject_logits)
                    .arg_u32(self.hidden_size as u32)
                    .arg_u32(self.hc_count as u32)
                    .arg_u32(num_tokens as u32)
                    .launch(stream)?;
            }
        }
        Ok(mixed)
    }

    fn sw_tail_rows_admits(&self, num_tokens: usize) -> bool {
        self.up_mix_sw_m5_o2_k.0 != 0
            && (1..=5).contains(&num_tokens)
            && self.hc_count == 4
            && self.rank <= 512
            && self.rank.is_multiple_of(16)
            && self.hidden_size.is_multiple_of(4)
    }

    /// The batched read-out tail kernel serves this width and geometry
    /// (`qwen4_hc_up_mix_tail_rows`: hc_count 4, rank <= 320 and a multiple
    /// of 16, hidden a multiple of 4, 1..=25 rows via `HC_TAIL_MAX_ROWS`).
    fn tail_rows_admits(&self, num_tokens: usize) -> bool {
        self.up_mix_tail_rows_k.0 != 0
            && (1..=HC_TAIL_MAX_ROWS).contains(&num_tokens)
            && self.hc_count == 4
            && self.rank <= 320
            && self.rank.is_multiple_of(16)
            && self.hidden_size.is_multiple_of(HC_TAIL_HIDDEN_PER_BLOCK)
    }

    /// The batched-verify dual down + inject launch serves this width
    /// (`w4a16_gemv_batch_logits_exact_pf_dual_m8`: 2..=128 rows sliced into
    /// `<= HC_DUAL_MAX_ROWS`-row chunks, K a multiple of 16).
    fn dual_rows_admits(&self, num_tokens: usize, k: usize) -> bool {
        self.exact_pf_dual_rows_k.0 != 0
            && (2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&num_tokens)
            && k.is_multiple_of(16)
    }

    /// Location where [`prepare_decode`] preserves this row's injection
    /// scales. The tail lives in row-specific residual storage, so callers
    /// may prepare several rows before applying their batched core outputs.
    pub fn saved_inject(&self, residual: DevicePtr) -> DevicePtr {
        residual.offset((self.residual_width() - self.hc_count) * 2)
    }

    fn exact_lm_head_kernels(&self) -> ops::W4a16ExactLmHeadKernels {
        ops::W4a16ExactLmHeadKernels::new(
            self.w4a16_gemv_exact_m4_k,
            self.w4a16_gemv_exact_m8_k,
            self.w4a16_gemv_exact_m17_k,
            self.w4a16_gemv_exact_m32_k,
        )
        .with_rowpar(self.w4a16_gemv_rows_k)
        .with_small_k_sw(self.w4a16_gemv_sw_m4_k, self.w4a16_gemv_sw_m8_k)
        .with_small_k_sw_m10(self.w4a16_gemv_sw_m10_k)
        .with_small_k_sw_m25(self.w4a16_gemv_sw_m25_k)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_w4a16_gemm_best(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        num_tokens: usize,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<&'static str> {
        if num_tokens > 64
            && n >= 6144
            && self.w4a16_gemm_pipe_m128n128_k.0 != 0
            && n.is_multiple_of(128)
            && k.is_multiple_of(32)
            && input.0.is_multiple_of(16)
            && weight.weight.0.is_multiple_of(16)
            && output.0.is_multiple_of(16)
        {
            ops::w4a16_gemm_pipe_m128n128(
                gpu,
                self.w4a16_gemm_pipe_m128n128_k,
                input,
                weight,
                output,
                num_tokens as u32,
                n,
                k,
                stream,
            )?;
            Ok("w4a16_gemm_pipe_m128n128")
        } else if self.w4a16_gemm_pipe_k.0 != 0
            && k.is_multiple_of(32)
            && input.0.is_multiple_of(16)
            && weight.weight.0.is_multiple_of(16)
        {
            ops::w4a16_gemm_pipe(
                gpu,
                self.w4a16_gemm_pipe_k,
                input,
                weight,
                output,
                num_tokens as u32,
                n,
                k,
                stream,
            )?;
            Ok("w4a16_gemm_pipe")
        } else {
            ops::w4a16_gemm(
                gpu,
                self.w4a16_gemm_k,
                input,
                weight,
                output,
                num_tokens as u32,
                n,
                k,
                stream,
            )?;
            Ok("w4a16_gemm")
        }
    }

    /// Execute one tensor-scale FP8 projection (`out[rows, n] = act[rows, k] @ weight[n, k]^T`)
    /// using caller-provided arena slices `a8` (`rows * k` bytes) and `ws` (256 bytes) so
    /// `qwen4_fast_proj::fp8_scratch` never grows for `k = 10240`. Reuses
    /// `qwen4_fast_proj`'s FP8 weight cache (bounded by `MemAvailable` stop line) when
    /// enabled and falls back to L2-resident per-call conversion when disabled.
    #[allow(clippy::too_many_arguments)]
    fn launch_fp8_tensor_proj(
        gpu: &dyn GpuBackend,
        act: DevicePtr,
        weight: &QuantizedWeight,
        out: DevicePtr,
        a8: DevicePtr,
        ws: DevicePtr,
        rows: usize,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        let tk = crate::layers::qwen4_fast_proj::fp8_tensor_kernels(gpu)?;
        let a_amax = ws.offset(4);
        let a_scale = ws.offset(12);
        // Zero both `w_amax` (`ws + 0`, required by `fp8_weight_tensor_gemm` on
        // uncached weights) and `a_amax` (`ws + 4`) in one 8-byte async memset.
        gpu.memset_async(ws, 0, 8, stream)?;
        let amax_blocks = div_ceil(((rows * k) / 8) as u32, 256).clamp(1, 256);
        KernelLaunch::new(gpu, tk.act_amax)
            .grid([amax_blocks, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(act)
            .arg_u64((rows * k) as u64)
            .arg_ptr(a_amax)
            .launch(stream)?;
        KernelLaunch::new(gpu, tk.act_rows)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(act)
            .arg_ptr(a_amax)
            .arg_ptr(a8)
            .arg_ptr(a_scale)
            .arg_u32(k as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        crate::layers::qwen4_fast_proj::fp8_weight_tensor_gemm(
            gpu,
            &tk,
            weight,
            out,
            rows,
            n,
            k,
            n,
            None,
            DevicePtr::NULL,
            ws,
            a8,
            true,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_mix_and_save_inject(
        &self,
        gpu: &dyn GpuBackend,
        residual: DevicePtr,
        mix_logits: DevicePtr,
        inject_logits: DevicePtr,
        mixed: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        if self.inject.is_some()
            && self.mix_save_inject_k.0 != 0
            && self.hc_count <= 256
            && (self.hidden_size.is_multiple_of(256) || self.hidden_size % 256 >= self.hc_count)
        {
            KernelLaunch::new(gpu, self.mix_save_inject_k)
                .grid([num_tokens as u32, div_ceil(self.hidden_size as u32, 256), 1])
                .block([256, 1, 1])
                .arg_ptr(residual)
                .arg_ptr(mix_logits)
                .arg_ptr(inject_logits)
                .arg_ptr(mixed)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .launch(stream)?;
        } else {
            let inject_ptr = if self.inject.is_some() {
                inject_logits
            } else {
                DevicePtr::NULL
            };
            KernelLaunch::new(gpu, self.mix_k)
                .grid([num_tokens as u32, div_ceil(self.hidden_size as u32, 256), 1])
                .block([256, 1, 1])
                .arg_ptr(residual)
                .arg_ptr(mix_logits)
                .arg_ptr(inject_ptr)
                .arg_ptr(mixed)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .launch(stream)?;
            if self.inject.is_some() {
                KernelLaunch::new(gpu, self.save_inject_k)
                    .grid([div_ceil((num_tokens * self.hc_count) as u32, 256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(residual)
                    .arg_ptr(inject_logits)
                    .arg_u32(self.hidden_size as u32)
                    .arg_u32(self.hc_count as u32)
                    .arg_u32(num_tokens as u32)
                    .launch(stream)?;
            }
        }
        Ok(())
    }

    fn launch_silu_div(
        &self,
        gpu: &dyn GpuBackend,
        down_out: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.silu_div_k)
            .grid([div_ceil((num_tokens * self.rank) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(down_out)
            .arg_u32((num_tokens * self.rank) as u32)
            .arg_f32(self.hc_count as f32)
            .launch(stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_inject_and_group_norm(
        &self,
        hyper: DevicePtr,
        core_to_inject: Option<DevicePtr>,
        residual: DevicePtr,
        rstd_out: DevicePtr,
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<bool> {
        let pdl = hc_pdl_enabled_for_sm(atlas_core::device::active_sm_count());
        if let Some(core) = core_to_inject {
            if self.inject_group_norm_v2_k.0 != 0
                && inject_group_norm_v2_eligible(
                    self.hidden_size,
                    self.hc_count,
                    hyper.0,
                    core.0,
                    self.norm.weight.0,
                    residual.0,
                )
            {
                KernelLaunch::new(gpu, self.inject_group_norm_v2_k)
                    .grid([num_tokens as u32, 4, 1])
                    .block([320, 1, 1])
                    .pdl(pdl)
                    .arg_ptr(hyper)
                    .arg_ptr(core)
                    .arg_ptr(DevicePtr::NULL)
                    .arg_ptr(self.norm.weight)
                    .arg_ptr(residual)
                    .arg_ptr(rstd_out)
                    .arg_u32(self.hidden_size as u32)
                    .arg_u32(self.hc_count as u32)
                    .arg_f32(eps)
                    .launch(stream)?;
                Ok(!rstd_out.is_null())
            } else if self.inject_group_norm_k.0 != 0
                && self.hc_count == 4
                && self.hidden_size <= 3072
            {
                KernelLaunch::new(gpu, self.inject_group_norm_k)
                    .grid([num_tokens as u32, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(hyper)
                    .arg_ptr(core)
                    .arg_ptr(DevicePtr::NULL)
                    .arg_ptr(self.norm.weight)
                    .arg_ptr(residual)
                    .arg_u32(self.hidden_size as u32)
                    .arg_u32(self.hc_count as u32)
                    .arg_f32(eps)
                    .launch(stream)?;
                Ok(false)
            } else {
                self.inject_saved_batched(hyper, core, residual, num_tokens, gpu, stream)?;
                KernelLaunch::new(gpu, self.group_norm_k)
                    .grid([num_tokens as u32, self.hc_count as u32, 1])
                    .block([self.hidden_size.min(1024) as u32, 1, 1])
                    .arg_ptr(hyper)
                    .arg_ptr(self.norm.weight)
                    .arg_ptr(residual)
                    .arg_u32(self.hidden_size as u32)
                    .arg_u32(self.hc_count as u32)
                    .arg_f32(eps)
                    .launch(stream)?;
                Ok(false)
            }
        } else {
            KernelLaunch::new(gpu, self.group_norm_k)
                .grid([num_tokens as u32, self.hc_count as u32, 1])
                .block([self.hidden_size.min(1024) as u32, 1, 1])
                .arg_ptr(hyper)
                .arg_ptr(self.norm.weight)
                .arg_ptr(residual)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .arg_f32(eps)
                .launch(stream)?;
            Ok(false)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn try_prepare_batched_tc_dense(
        &self,
        hyper: DevicePtr,
        core_to_inject: Option<DevicePtr>,
        residual: DevicePtr,
        num_tokens: usize,
        buffers: &BufferArena,
        out_mixed: Option<DevicePtr>,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let r = self.residual_width();
        let down_bytes = num_tokens * self.rank * 2;
        let sizes = buffers.sizes();
        let capturing = gpu.stream_is_capturing(stream)
            || spark_runtime::gpu_capture_region::active_capture_region().is_some();
        let inject_valid = self.inject.as_ref().is_some_and(|w| {
            !w.weight.is_null() && !w.weight_scale.is_null() && w.weight_scale_2.is_finite()
        });
        let base_weights_valid = !self.norm.weight.is_null()
            && !self.down.weight.is_null()
            && !self.down.weight_scale.is_null()
            && self.down.weight_scale_2.is_finite()
            && !self.up.weight.is_null()
            && !self.up.weight_scale.is_null()
            && self.up.weight_scale_2.is_finite();
        let weights_valid = base_weights_valid && (self.inject.is_none() || inject_valid);
        let w4a16_gemm_ready = weights_valid
            && self.w4a16_gemm_k.0 != 0
            && r.is_multiple_of(16)
            && self.rank.is_multiple_of(16);

        let mixed = out_mixed.unwrap_or_else(|| buffers.norm_output());
        let down_out = buffers.ssm_ba();
        let inject_logits = down_out.offset(down_bytes);
        let qkv = buffers.qkv_output();
        let qkv_layout =
            BatchedHcQkvScratchLayout::compute(num_tokens, r, self.rank, self.hc_count);
        let qkv_non_aliasing = !qkv.is_null()
            && qkv.0.is_multiple_of(16)
            && qkv != hyper
            && qkv != residual
            && qkv != down_out
            && qkv != mixed;

        let sm_count = atlas_core::device::active_sm_count();
        let dual_coalesced_kernel_ready = resolve_hc_dual_coalesced_for_sms(
            num_tokens,
            self.coalesced_dual_silu_div_k,
            self.coalesced_dual_n16_v2_silu_div_k,
            self.coalesced_dual_n32_silu_div_k,
            self.coalesced_dual_high_sm_k,
            hc_dual_n32_enabled(),
            sm_count,
        )
        .is_some();
        let dual_coalesced_ready = hc_dual_coalesced_eligible(
            hc_dual_coalesced_enabled(),
            dual_coalesced_kernel_ready,
            weights_valid,
            num_tokens,
            self.rank,
            self.hc_count,
            r,
            residual.0,
            self.down.weight.0,
            self.down.weight_scale.0,
            down_out.0,
            self.inject.as_ref().map(|w| w.weight.0),
            self.inject.as_ref().map(|w| w.weight_scale.0),
            inject_logits.0,
        );
        let upmix_coalesced_ready = hc_upmix_coalesced_eligible(
            hc_upmix_coalesced_enabled(),
            self.coalesced_upmix_k.resolve(num_tokens).is_some(),
            weights_valid,
            num_tokens,
            self.hidden_size,
            self.hc_count,
            self.rank,
            down_out.0,
            self.up.weight.0,
            self.up.weight_scale.0,
            residual.0,
            self.inject.is_some(),
            inject_logits.0,
            mixed.0,
        );

        // Direct coalesced NVFP4 W4A16 GEMM (`try_coalesced_nvfp4_w4a16_gemm`):
        // 1 launch, 0 B extra VRAM, zero FP8 cache, zero warmup, 100% CUDA-graph safe.
        let coalesced_down_ready = coalesced_nvfp4_hc_proj_eligible(
            weights_valid,
            num_tokens,
            self.rank,
            r,
            residual.0,
            self.down.weight.0,
            self.down.weight_scale.0,
            down_out.0,
        );
        let coalesced_up_ready = coalesced_nvfp4_hc_proj_eligible(
            weights_valid,
            num_tokens,
            r,
            self.rank,
            down_out.0,
            self.up.weight.0,
            self.up.weight_scale.0,
            qkv.0,
        ) && qkv_non_aliasing
            && !residual.is_null()
            && !mixed.is_null()
            && mixed != residual
            && mixed != down_out;

        let upmix_w4_ready = weights_valid
            && (self.hidden_size, self.hc_count, self.rank) == (2560, 4, 320)
            && self.hc_rstd_k.0 != 0
            && self.upmix_roof_t128_w4_k.0 != 0
            && qkv_non_aliasing
            && qkv_layout.is_some_and(|l| l.total_down_rstd_bytes <= sizes.qkv_output)
            && !hyper.is_null()
            && hyper.0.is_multiple_of(16)
            && !down_out.is_null()
            && down_out.0.is_multiple_of(16)
            && !mixed.is_null()
            && mixed.0.is_multiple_of(16)
            && mixed != hyper
            && mixed != down_out
            && self.norm.weight.0.is_multiple_of(4)
            && self.up.weight.0.is_multiple_of(8);

        let fp8_kernels_ok = weights_valid
            && r.is_multiple_of(16)
            && self.rank.is_multiple_of(16)
            && qkv_non_aliasing
            && !residual.is_null()
            && residual.0.is_multiple_of(16)
            && !down_out.is_null()
            && down_out.0.is_multiple_of(16)
            && fp8_tensor_kernels_available(gpu);
        let fp8_down_ready = fp8_kernels_ok
            && qkv_layout.is_some_and(|l| l.total_down_rstd_bytes <= sizes.qkv_output)
            && (!capturing || self.fp8_down_warmed.is_warmed(num_tokens));
        let fp8_up_ready = fp8_kernels_ok
            && qkv_layout.is_some_and(|l| l.total_up_fp8_bytes <= sizes.qkv_output)
            && (!capturing || self.fp8_up_warmed.is_warmed(num_tokens));

        let dual_silu_m10_ready = self.inject.is_some()
            && self.dual_rows_admits(num_tokens, r)
            && num_tokens < HC_TC_WIDE_DOWN_MIN_ROWS
            && ((num_tokens > 8 && self.exact_pf_dual_silu_rows_m10_k.0 != 0)
                || (num_tokens <= 8 && self.exact_pf_dual_silu_rows_k.0 != 0));
        let exact_small_ready = num_tokens < HC_TC_WIDE_DOWN_MIN_ROWS
            && r.is_multiple_of(16)
            && self.w4a16_gemv_exact_m8_k.0 != 0
            && self.w4a16_gemv_exact_m17_k.0 != 0;
        let dual_silu_exact_ready = self.inject.is_some()
            && self.dual_rows_admits(num_tokens, r)
            && ((11..=25).contains(&num_tokens) && self.exact_pf_dual_silu_rows_m25_k.0 != 0
                || num_tokens > 8 && self.exact_pf_dual_silu_rows_m10_k.0 != 0
                || num_tokens <= 8 && self.exact_pf_dual_silu_rows_k.0 != 0);

        let det_batch =
            crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled();
        let down_route = select_batched_hc_down_route_v2_with_policy(
            num_tokens,
            self.inject.is_some(),
            dual_coalesced_ready,
            dual_silu_m10_ready,
            exact_small_ready,
            coalesced_down_ready,
            fp8_down_ready,
            dual_silu_exact_ready,
            w4a16_gemm_ready,
            det_batch,
        );
        let down_fallback = select_batched_hc_down_route(
            num_tokens,
            self.inject.is_some(),
            dual_silu_m10_ready,
            exact_small_ready,
            down_route == Some(BatchedHcDownRoute::DualCoalescedNvfp4) && coalesced_down_ready,
            fp8_down_ready,
            dual_silu_exact_ready,
            w4a16_gemm_ready,
        );
        let up_route = select_batched_hc_up_route_v2_with_policy(
            num_tokens,
            upmix_coalesced_ready,
            coalesced_up_ready,
            upmix_w4_ready,
            fp8_up_ready,
            w4a16_gemm_ready,
            det_batch,
        );
        let up_fallback = select_batched_hc_up_route(
            num_tokens,
            up_route == Some(BatchedHcUpRoute::UpmixCoalesced) && coalesced_up_ready,
            upmix_w4_ready,
            fp8_up_ready,
            w4a16_gemm_ready,
        );
        let fast_tc_pair = match (down_route, up_route, qkv_layout) {
            (Some(d), Some(u), Some(layout)) if is_fast_tc_pair(d, u) => Some((d, u, layout)),
            _ => None,
        };
        let fast_tc_ready = fast_tc_pair.is_some();

        match select_batched_hc_tc_route_with_policy(
            true,
            fast_tc_ready,
            w4a16_gemm_ready,
            det_batch,
        ) {
            BatchedHcTcRoute::Exact => Ok(None),
            BatchedHcTcRoute::FastTc => {
                let (d_route, u_route, layout) =
                    fast_tc_pair.expect("FastTc route requires resolved down/up routes and layout");
                let mixed = self.prepare_batched_tc_fast(
                    d_route,
                    down_fallback,
                    u_route,
                    up_fallback,
                    layout,
                    hyper,
                    core_to_inject,
                    residual,
                    num_tokens,
                    buffers,
                    out_mixed,
                    gpu,
                    eps,
                    capturing,
                    stream,
                )?;
                Ok(Some(mixed))
            }
            BatchedHcTcRoute::W4a16Gemm => {
                let mixed = self.prepare_batched_tc_w4a16_gemm(
                    hyper,
                    core_to_inject,
                    residual,
                    num_tokens,
                    buffers,
                    out_mixed,
                    gpu,
                    eps,
                    stream,
                )?;
                Ok(Some(mixed))
            }
        }
    }

    /// Launch Warp-Split-K K-coalesced NVFP4 `hc.down` with the fused exact
    /// `silu_div` epilogue (`bf16(x / (1 + exp(-x)))`, `x = bf16(raw) / hc_count`),
    /// eliminating the trailing `qwen4_hc_silu_div` kernel launch across all
    /// canonical `hc.down` row counts (`1..=128`).
    fn try_coalesced_nvfp4_down_silu_div(
        &self,
        gpu: &dyn GpuBackend,
        residual: DevicePtr,
        down_out: DevicePtr,
        num_tokens: usize,
        k: usize,
        stream: u64,
    ) -> Result<bool> {
        if k == 0
            || !k.is_multiple_of(32)
            || residual.is_null()
            || down_out.is_null()
            || self.down.weight.is_null()
            || self.down.weight_scale.is_null()
            || !self.down.weight_scale_2.is_finite()
            || !residual.0.is_multiple_of(16)
            || !self.down.weight.0.is_multiple_of(16)
            || !self.down.weight_scale.0.is_multiple_of(2)
            || !down_out.0.is_multiple_of(2)
        {
            return Ok(false);
        }
        let (Ok(m_u32), Ok(n_u32), Ok(k_u32)) = (
            u32::try_from(num_tokens),
            u32::try_from(self.rank),
            u32::try_from(k),
        ) else {
            return Ok(false);
        };
        let Some(tile) = crate::layers::qwen4_fast_proj::select_coalesced_nvfp4_tile_for_shape(
            num_tokens, self.rank, k,
        ) else {
            return Ok(false);
        };
        let kernel = self.coalesced_silu_div_k.resolve(tile);
        if kernel.0 == 0 {
            return Ok(false);
        }
        KernelLaunch::new(gpu, kernel)
            .grid([
                div_ceil(n_u32, tile.co_n()),
                div_ceil(m_u32, tile.tile_m()),
                1,
            ])
            .block([tile.threads(), 1, 1])
            .arg_ptr(residual)
            .arg_ptr(self.down.weight)
            .arg_ptr(self.down.weight_scale)
            .arg_f32(self.down.weight_scale_2)
            .arg_ptr(down_out)
            .arg_u32(m_u32)
            .arg_u32(n_u32)
            .arg_u32(k_u32)
            .arg_f32(self.hc_count as f32)
            .launch(stream)?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_batched_hc_down_route(
        &self,
        route: BatchedHcDownRoute,
        fallback: Option<BatchedHcDownRoute>,
        layout: BatchedHcQkvScratchLayout,
        residual: DevicePtr,
        down_out: DevicePtr,
        inject_logits: DevicePtr,
        qkv: DevicePtr,
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        capturing: bool,
        stream: u64,
    ) -> Result<(BatchedHcDownRoute, bool)> {
        let r = self.residual_width();
        match route {
            BatchedHcDownRoute::DualCoalescedNvfp4 => {
                let sm_count = atlas_core::device::active_sm_count();
                if let Some((tier, kernel)) = resolve_hc_dual_coalesced_for_sms(
                    num_tokens,
                    self.coalesced_dual_silu_div_k,
                    self.coalesced_dual_n16_v2_silu_div_k,
                    self.coalesced_dual_n32_silu_div_k,
                    self.coalesced_dual_high_sm_k,
                    hc_dual_n32_enabled(),
                    sm_count,
                ) {
                    let (b2_w, b2_s, b2_s2, c2, n2) = match self.inject.as_ref() {
                        Some(weight) => (
                            weight.weight,
                            weight.weight_scale,
                            weight.weight_scale_2,
                            inject_logits,
                            self.hc_count as u32,
                        ),
                        None => (
                            DevicePtr::NULL,
                            DevicePtr::NULL,
                            0.0f32,
                            DevicePtr::NULL,
                            0u32,
                        ),
                    };
                    let co_n = tier.co_n();
                    let grid_x = div_ceil(self.rank as u32, co_n)
                        + if n2 > 0 { div_ceil(n2, co_n) } else { 0 };
                    let grid_y = div_ceil(num_tokens as u32, tier.tile_m());
                    let grid_z = select_dual_split_k_grid_z_with_policy(
                        tier,
                        grid_x,
                        grid_y,
                        r,
                        sm_count,
                        self.dual_splitk_scratch.is_some(),
                        crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(
                        ),
                    );
                    let pdl = hc_pdl_enabled_for_sm(sm_count);
                    let launch = KernelLaunch::new(gpu, kernel)
                        .grid([grid_x, grid_y, grid_z])
                        .block([tier.threads(), 1, 1])
                        .pdl(pdl)
                        .arg_ptr(residual)
                        .arg_ptr(self.down.weight)
                        .arg_ptr(self.down.weight_scale)
                        .arg_f32(self.down.weight_scale_2)
                        .arg_ptr(down_out)
                        .arg_u32(self.rank as u32)
                        .arg_ptr(b2_w)
                        .arg_ptr(b2_s)
                        .arg_f32(b2_s2)
                        .arg_ptr(c2)
                        .arg_u32(n2)
                        .arg_u32(num_tokens as u32)
                        .arg_u32(r as u32)
                        .arg_f32(self.hc_count as f32);
                    if tier.supports_grid_split_k() {
                        let (partials_ptr, counters_ptr) = match self.dual_splitk_scratch {
                            Some(sc) if grid_z > 1 => (sc.partials, sc.counters),
                            _ => (DevicePtr::NULL, DevicePtr::NULL),
                        };
                        launch
                            .arg_ptr(partials_ptr)
                            .arg_ptr(counters_ptr)
                            .launch(stream)?;
                    } else {
                        launch.launch(stream)?;
                    }
                    return Ok((BatchedHcDownRoute::DualCoalescedNvfp4, n2 > 0));
                }
                let fb = fallback.ok_or_else(|| {
                    anyhow::anyhow!(
                        "dual coalesced NVFP4 HC down unavailable for rows={num_tokens} rank={} k={r} with no fallback",
                        self.rank
                    )
                })?;
                debug_assert_ne!(fb, BatchedHcDownRoute::DualCoalescedNvfp4);
                self.execute_batched_hc_down_route(
                    fb,
                    None,
                    layout,
                    residual,
                    down_out,
                    inject_logits,
                    qkv,
                    num_tokens,
                    gpu,
                    capturing,
                    stream,
                )
            }
            BatchedHcDownRoute::CoalescedNvfp4 => {
                if self.try_coalesced_nvfp4_down_silu_div(
                    gpu, residual, down_out, num_tokens, r, stream,
                )? {
                    return Ok((BatchedHcDownRoute::CoalescedNvfp4, false));
                }
                if crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                    gpu, residual, &self.down, down_out, num_tokens, self.rank, r, stream,
                )? {
                    self.launch_silu_div(gpu, down_out, num_tokens, stream)?;
                    return Ok((BatchedHcDownRoute::CoalescedNvfp4, false));
                }
                let fb = fallback.ok_or_else(|| {
                    anyhow::anyhow!(
                        "coalesced NVFP4 HC down unavailable for rows={num_tokens} rank={} k={r} with no fallback",
                        self.rank
                    )
                })?;
                debug_assert_ne!(fb, BatchedHcDownRoute::CoalescedNvfp4);
                self.execute_batched_hc_down_route(
                    fb,
                    None,
                    layout,
                    residual,
                    down_out,
                    inject_logits,
                    qkv,
                    num_tokens,
                    gpu,
                    capturing,
                    stream,
                )
            }
            BatchedHcDownRoute::DualSiluM10 => {
                let weight = self
                    .inject
                    .as_ref()
                    .expect("DualSiluM10 requires an injecting hyperconnection");
                let kernel = if num_tokens > 8 {
                    self.exact_pf_dual_silu_rows_m10_k
                } else {
                    self.exact_pf_dual_silu_rows_k
                };
                let grid_x = div_ceil(self.rank as u32, 4) + div_ceil(self.hc_count as u32, 4);
                KernelLaunch::new(gpu, kernel)
                    .grid([grid_x, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(residual)
                    .arg_ptr(self.down.weight)
                    .arg_ptr(self.down.weight_scale)
                    .arg_f32(self.down.weight_scale_2)
                    .arg_ptr(down_out)
                    .arg_u32(self.rank as u32)
                    .arg_ptr(weight.weight)
                    .arg_ptr(weight.weight_scale)
                    .arg_f32(weight.weight_scale_2)
                    .arg_ptr(inject_logits)
                    .arg_u32(self.hc_count as u32)
                    .arg_u32(num_tokens as u32)
                    .arg_u32(r as u32)
                    .arg_f32(self.hc_count as f32)
                    .launch(stream)?;
                Ok((BatchedHcDownRoute::DualSiluM10, true))
            }
            BatchedHcDownRoute::ExactSmallRows => {
                let exact_kernels = self.exact_lm_head_kernels();
                ops::w4a16_gemv_batch_logits_exact_with(
                    gpu,
                    exact_kernels,
                    residual,
                    &self.down,
                    down_out,
                    num_tokens as u32,
                    self.rank as u32,
                    r as u32,
                    stream,
                    false,
                )?;
                self.launch_silu_div(gpu, down_out, num_tokens, stream)?;
                Ok((BatchedHcDownRoute::ExactSmallRows, false))
            }
            BatchedHcDownRoute::Fp8Tensor => {
                Self::launch_fp8_tensor_proj(
                    gpu,
                    residual,
                    &self.down,
                    down_out,
                    qkv.offset(layout.down_a8_off),
                    qkv.offset(layout.down_ws_off),
                    num_tokens,
                    self.rank,
                    r,
                    stream,
                )?;
                if !capturing {
                    self.fp8_down_warmed.mark_warmed(num_tokens);
                }
                self.launch_silu_div(gpu, down_out, num_tokens, stream)?;
                Ok((BatchedHcDownRoute::Fp8Tensor, false))
            }
            BatchedHcDownRoute::DualSiluExact => {
                let weight = self
                    .inject
                    .as_ref()
                    .expect("DualSiluExact requires an injecting hyperconnection");
                let grid_x = div_ceil(self.rank as u32, 4) + div_ceil(self.hc_count as u32, 4);
                let (kernel, grid_y) = if (11..=25).contains(&num_tokens)
                    && self.exact_pf_dual_silu_rows_m25_k.0 != 0
                {
                    (self.exact_pf_dual_silu_rows_m25_k, 1u32)
                } else if num_tokens > 8 && self.exact_pf_dual_silu_rows_m10_k.0 != 0 {
                    (
                        self.exact_pf_dual_silu_rows_m10_k,
                        div_ceil(num_tokens as u32, 10),
                    )
                } else {
                    (self.exact_pf_dual_silu_rows_k, 1u32)
                };
                KernelLaunch::new(gpu, kernel)
                    .grid([grid_x, grid_y, 1])
                    .block([256, 1, 1])
                    .arg_ptr(residual)
                    .arg_ptr(self.down.weight)
                    .arg_ptr(self.down.weight_scale)
                    .arg_f32(self.down.weight_scale_2)
                    .arg_ptr(down_out)
                    .arg_u32(self.rank as u32)
                    .arg_ptr(weight.weight)
                    .arg_ptr(weight.weight_scale)
                    .arg_f32(weight.weight_scale_2)
                    .arg_ptr(inject_logits)
                    .arg_u32(self.hc_count as u32)
                    .arg_u32(num_tokens as u32)
                    .arg_u32(r as u32)
                    .arg_f32(self.hc_count as f32)
                    .launch(stream)?;
                Ok((BatchedHcDownRoute::DualSiluExact, true))
            }
            BatchedHcDownRoute::W4a16Gemm => {
                let _ = self.launch_w4a16_gemm_best(
                    gpu,
                    residual,
                    &self.down,
                    down_out,
                    num_tokens,
                    self.rank as u32,
                    r as u32,
                    stream,
                )?;
                self.launch_silu_div(gpu, down_out, num_tokens, stream)?;
                Ok((BatchedHcDownRoute::W4a16Gemm, false))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_batched_hc_up_route(
        &self,
        route: BatchedHcUpRoute,
        fallback: Option<BatchedHcUpRoute>,
        rstd_computed: bool,
        layout: BatchedHcQkvScratchLayout,
        hyper: DevicePtr,
        residual: DevicePtr,
        down_out: DevicePtr,
        inject_logits: DevicePtr,
        qkv: DevicePtr,
        mixed: DevicePtr,
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        eps: f32,
        capturing: bool,
        stream: u64,
    ) -> Result<BatchedHcUpRoute> {
        let r = self.residual_width();
        match route {
            BatchedHcUpRoute::UpmixCoalesced => {
                if let Some((tile_m, kernel)) = self.coalesced_upmix_k.resolve(num_tokens) {
                    let inject_ptr = if self.inject.is_some() {
                        inject_logits
                    } else {
                        DevicePtr::NULL
                    };
                    let pdl = hc_pdl_enabled_for_sm(atlas_core::device::active_sm_count());
                    KernelLaunch::new(gpu, kernel)
                        .grid([
                            (self.hidden_size / 16) as u32,
                            div_ceil(num_tokens as u32, tile_m),
                            1,
                        ])
                        .block([256, 1, 1])
                        .pdl(pdl)
                        .arg_ptr(down_out)
                        .arg_ptr(self.up.weight)
                        .arg_ptr(self.up.weight_scale)
                        .arg_f32(self.up.weight_scale_2)
                        .arg_ptr(residual)
                        .arg_ptr(inject_ptr)
                        .arg_ptr(mixed)
                        .arg_u32(num_tokens as u32)
                        .arg_u32(self.hidden_size as u32)
                        .arg_u32(self.rank as u32)
                        .launch(stream)?;
                    return Ok(BatchedHcUpRoute::UpmixCoalesced);
                }
                let fb = fallback.ok_or_else(|| {
                    anyhow::anyhow!(
                        "fused coalesced NVFP4 HC upmix unavailable for rows={num_tokens} h={} rank={} with no fallback",
                        self.hidden_size,
                        self.rank
                    )
                })?;
                debug_assert_ne!(fb, BatchedHcUpRoute::UpmixCoalesced);
                self.execute_batched_hc_up_route(
                    fb,
                    None,
                    rstd_computed,
                    layout,
                    hyper,
                    residual,
                    down_out,
                    inject_logits,
                    qkv,
                    mixed,
                    num_tokens,
                    gpu,
                    eps,
                    capturing,
                    stream,
                )
            }
            BatchedHcUpRoute::CoalescedNvfp4 => {
                if crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                    gpu, down_out, &self.up, qkv, num_tokens, r, self.rank, stream,
                )? {
                    self.launch_mix_and_save_inject(
                        gpu,
                        residual,
                        qkv,
                        inject_logits,
                        mixed,
                        num_tokens,
                        stream,
                    )?;
                    return Ok(BatchedHcUpRoute::CoalescedNvfp4);
                }
                let fb = fallback.ok_or_else(|| {
                    anyhow::anyhow!(
                        "coalesced NVFP4 HC up unavailable for rows={num_tokens} n={r} rank={} with no fallback",
                        self.rank
                    )
                })?;
                debug_assert_ne!(fb, BatchedHcUpRoute::CoalescedNvfp4);
                self.execute_batched_hc_up_route(
                    fb,
                    None,
                    rstd_computed,
                    layout,
                    hyper,
                    residual,
                    down_out,
                    inject_logits,
                    qkv,
                    mixed,
                    num_tokens,
                    gpu,
                    eps,
                    capturing,
                    stream,
                )
            }
            BatchedHcUpRoute::UpmixRoofW4 => {
                if !rstd_computed {
                    KernelLaunch::new(gpu, self.hc_rstd_k)
                        .grid([num_tokens as u32, 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(hyper)
                        .arg_ptr(qkv.offset(layout.rstd_off))
                        .arg_u32(self.hidden_size as u32)
                        .arg_f32(eps)
                        .launch(stream)?;
                }
                if self.inject.is_some() {
                    // Transform raw inject logits in place (`2 * sigmoid(x / hc)`) without
                    // touching `normed`/`mix_logits`/`mixed` (`hidden_size = 0` skips the mix
                    // loop while executing the exact inject transform for `h < hc_count`),
                    // then scatter the 4 saved scales into each `residual` row's tail.
                    KernelLaunch::new(gpu, self.mix_k)
                        .grid([num_tokens as u32, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(residual)
                        .arg_ptr(residual)
                        .arg_ptr(inject_logits)
                        .arg_ptr(mixed)
                        .arg_u32(0u32)
                        .arg_u32(self.hc_count as u32)
                        .launch(stream)?;
                    KernelLaunch::new(gpu, self.save_inject_k)
                        .grid([div_ceil((num_tokens * self.hc_count) as u32, 256), 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(residual)
                        .arg_ptr(inject_logits)
                        .arg_u32(self.hidden_size as u32)
                        .arg_u32(self.hc_count as u32)
                        .arg_u32(num_tokens as u32)
                        .launch(stream)?;
                }
                KernelLaunch::new(gpu, self.upmix_roof_t128_w4_k)
                    .grid([
                        (self.hidden_size / 32) as u32,
                        div_ceil(num_tokens as u32, 128),
                        1,
                    ])
                    .block([256, 1, 1])
                    .shared_mem(UPMIX_ROOF_T128_W4_SMEM_BYTES)
                    .arg_ptr(down_out)
                    .arg_ptr(self.up.weight)
                    .arg_ptr(self.up.weight_scale)
                    .arg_f32(self.up.weight_scale_2)
                    .arg_ptr(hyper)
                    .arg_ptr(qkv.offset(layout.rstd_off))
                    .arg_ptr(self.norm.weight)
                    .arg_ptr(mixed)
                    .arg_u32(num_tokens as u32)
                    .launch(stream)?;
                Ok(BatchedHcUpRoute::UpmixRoofW4)
            }
            BatchedHcUpRoute::Fp8Tensor => {
                Self::launch_fp8_tensor_proj(
                    gpu,
                    down_out,
                    &self.up,
                    qkv,
                    qkv.offset(layout.up_a8_off),
                    qkv.offset(layout.up_ws_off),
                    num_tokens,
                    r,
                    self.rank,
                    stream,
                )?;
                if !capturing {
                    self.fp8_up_warmed.mark_warmed(num_tokens);
                }
                self.launch_mix_and_save_inject(
                    gpu,
                    residual,
                    qkv,
                    inject_logits,
                    mixed,
                    num_tokens,
                    stream,
                )?;
                Ok(BatchedHcUpRoute::Fp8Tensor)
            }
            BatchedHcUpRoute::W4a16Gemm => {
                let _ = self.launch_w4a16_gemm_best(
                    gpu,
                    down_out,
                    &self.up,
                    qkv,
                    num_tokens,
                    r as u32,
                    self.rank as u32,
                    stream,
                )?;
                self.launch_mix_and_save_inject(
                    gpu,
                    residual,
                    qkv,
                    inject_logits,
                    mixed,
                    num_tokens,
                    stream,
                )?;
                Ok(BatchedHcUpRoute::W4a16Gemm)
            }
        }
    }

    /// High-throughput Tensor-Core batched HC preparation for both injecting
    /// layer HCs (`HcPrepare`) and `qwen4_final_mixer` (`FinalMixer`):
    /// - `down`: `select_batched_hc_down_route_v2` selects `DualCoalescedNvfp4`
    ///   for `rows = 4..=256` when ready, falling back via
    ///   `select_batched_hc_down_route` to:
    ///   - `rows <= 10`: `DualSiluM10` (injecting) / `ExactSmallRows` ->
    ///     `CoalescedNvfp4` -> `Fp8Tensor` -> `DualSiluExact` -> `W4a16Gemm`;
    ///   - `rows = 11..=25`: `CoalescedNvfp4` -> `DualSiluExact` ->
    ///     `Fp8Tensor` -> `W4a16Gemm`;
    ///   - `rows >= 26`: `CoalescedNvfp4` -> `Fp8Tensor` -> `DualSiluExact` ->
    ///     `W4a16Gemm`.
    /// - `up + mix`: `select_batched_hc_up_route_v2` selects fused
    ///   `UpmixCoalesced` for `rows = 4..=256` when ready, falling back via
    ///   `select_batched_hc_up_route` to `CoalescedNvfp4 -> UpmixRoofW4 ->
    ///   Fp8Tensor -> W4a16Gemm`.
    #[allow(clippy::too_many_arguments)]
    fn prepare_batched_tc_fast(
        &self,
        down_route: BatchedHcDownRoute,
        down_fallback: Option<BatchedHcDownRoute>,
        up_route: BatchedHcUpRoute,
        up_fallback: Option<BatchedHcUpRoute>,
        layout: BatchedHcQkvScratchLayout,
        hyper: DevicePtr,
        core_to_inject: Option<DevicePtr>,
        residual: DevicePtr,
        num_tokens: usize,
        buffers: &BufferArena,
        out_mixed: Option<DevicePtr>,
        gpu: &dyn GpuBackend,
        eps: f32,
        capturing: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        let r = self.residual_width();
        let down_bytes = num_tokens * self.rank * 2;
        let qkv = buffers.qkv_output();
        let down_out = buffers.ssm_ba();
        let inject_logits = down_out.offset(down_bytes);
        let mixed = out_mixed.unwrap_or_else(|| buffers.norm_output());

        let rstd_target = if up_route == BatchedHcUpRoute::UpmixRoofW4 {
            qkv.offset(layout.rstd_off)
        } else {
            DevicePtr::NULL
        };
        let mut rstd_computed = self.launch_inject_and_group_norm(
            hyper,
            core_to_inject,
            residual,
            rstd_target,
            num_tokens,
            gpu,
            eps,
            stream,
        )?;

        if up_route == BatchedHcUpRoute::UpmixRoofW4 && !rstd_computed {
            KernelLaunch::new(gpu, self.hc_rstd_k)
                .grid([num_tokens as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(hyper)
                .arg_ptr(qkv.offset(layout.rstd_off))
                .arg_u32(self.hidden_size as u32)
                .arg_f32(eps)
                .launch(stream)?;
            rstd_computed = true;
        }

        let (actual_down_route, inject_done) = self.execute_batched_hc_down_route(
            down_route,
            down_fallback,
            layout,
            residual,
            down_out,
            inject_logits,
            qkv,
            num_tokens,
            gpu,
            capturing,
            stream,
        )?;

        if !inject_done && let Some(ref weight) = self.inject {
            if self.w4a16_gemv_rows_k.0 != 0
                && self.hc_count <= ops::W4A16_EXACT_ROWPAR_MAX_N as usize
            {
                ops::w4a16_gemv_rows(
                    gpu,
                    self.w4a16_gemv_rows_k,
                    residual,
                    weight,
                    inject_logits,
                    num_tokens as u32,
                    self.hc_count as u32,
                    r as u32,
                    stream,
                )?;
            } else {
                let _ = self.launch_w4a16_gemm_best(
                    gpu,
                    residual,
                    weight,
                    inject_logits,
                    num_tokens,
                    self.hc_count as u32,
                    r as u32,
                    stream,
                )?;
            }
        }

        let actual_up_route = self.execute_batched_hc_up_route(
            up_route,
            up_fallback,
            rstd_computed,
            layout,
            hyper,
            residual,
            down_out,
            inject_logits,
            qkv,
            mixed,
            num_tokens,
            gpu,
            eps,
            capturing,
            stream,
        )?;

        if self.inject.is_some() {
            static HC_FAST_LOGGED: std::sync::Once = std::sync::Once::new();
            HC_FAST_LOGGED.call_once(|| {
                tracing::info!(
                    selector = BATCHED_VERIFY_TC_DENSE_ENV,
                    component = "hc_prepare",
                    down_route = actual_down_route.label(),
                    up_route = actual_up_route.label(),
                    cached_down = actual_down_route == BatchedHcDownRoute::Fp8Tensor
                        && crate::layers::qwen4_fast_proj::has_cached_weight_tensor(
                            &self.down, self.rank, r, None,
                        ),
                    rows = num_tokens,
                    "QWEN4_BATCHED_VERIFY_TC_DENSE_ENGAGED"
                );
            });
        } else {
            static FINAL_FAST_LOGGED: std::sync::Once = std::sync::Once::new();
            FINAL_FAST_LOGGED.call_once(|| {
                tracing::info!(
                    selector = BATCHED_VERIFY_TC_DENSE_ENV,
                    component = "final_mixer",
                    down_route = actual_down_route.label(),
                    up_route = actual_up_route.label(),
                    cached_down = actual_down_route == BatchedHcDownRoute::Fp8Tensor
                        && crate::layers::qwen4_fast_proj::has_cached_weight_tensor(
                            &self.down, self.rank, r, None,
                        ),
                    rows = num_tokens,
                    "QWEN4_BATCHED_VERIFY_TC_DENSE_ENGAGED"
                );
            });
        }
        Ok(mixed)
    }

    /// Zero-staging Tensor-Core W4A16 batched HC preparation fallback (zero
    /// `s.folded` traffic, zero mutexes, zero lazy allocations, 100% CUDA-graph
    /// safe): covers both injecting layer HCs (`HcPrepare`) and
    /// `qwen4_final_mixer` (`FinalMixer`) when `FastTc` kernels are absent.
    #[allow(clippy::too_many_arguments)]
    fn prepare_batched_tc_w4a16_gemm(
        &self,
        hyper: DevicePtr,
        core_to_inject: Option<DevicePtr>,
        residual: DevicePtr,
        num_tokens: usize,
        buffers: &BufferArena,
        out_mixed: Option<DevicePtr>,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let r = self.residual_width();
        let down_bytes = num_tokens * self.rank * 2;
        let _ = self.launch_inject_and_group_norm(
            hyper,
            core_to_inject,
            residual,
            DevicePtr::NULL,
            num_tokens,
            gpu,
            eps,
            stream,
        )?;

        let down_out = buffers.ssm_ba();
        let inject_logits = down_out.offset(down_bytes);
        let dual_silu_down = if let Some(ref weight) = self.inject
            && self.dual_rows_admits(num_tokens, r)
            && ((11..=25).contains(&num_tokens) && self.exact_pf_dual_silu_rows_m25_k.0 != 0
                || num_tokens > 8 && self.exact_pf_dual_silu_rows_m10_k.0 != 0)
        {
            let grid_x = div_ceil(self.rank as u32, 4) + div_ceil(self.hc_count as u32, 4);
            let (kernel, grid_y) =
                if (11..=25).contains(&num_tokens) && self.exact_pf_dual_silu_rows_m25_k.0 != 0 {
                    (self.exact_pf_dual_silu_rows_m25_k, 1u32)
                } else {
                    (
                        self.exact_pf_dual_silu_rows_m10_k,
                        div_ceil(num_tokens as u32, 10),
                    )
                };
            KernelLaunch::new(gpu, kernel)
                .grid([grid_x, grid_y, 1])
                .block([256, 1, 1])
                .arg_ptr(residual)
                .arg_ptr(self.down.weight)
                .arg_ptr(self.down.weight_scale)
                .arg_f32(self.down.weight_scale_2)
                .arg_ptr(down_out)
                .arg_u32(self.rank as u32)
                .arg_ptr(weight.weight)
                .arg_ptr(weight.weight_scale)
                .arg_f32(weight.weight_scale_2)
                .arg_ptr(inject_logits)
                .arg_u32(self.hc_count as u32)
                .arg_u32(num_tokens as u32)
                .arg_u32(r as u32)
                .arg_f32(self.hc_count as f32)
                .launch(stream)?;
            true
        } else {
            let _ = self.launch_w4a16_gemm_best(
                gpu,
                residual,
                &self.down,
                down_out,
                num_tokens,
                self.rank as u32,
                r as u32,
                stream,
            )?;
            KernelLaunch::new(gpu, self.silu_div_k)
                .grid([div_ceil((num_tokens * self.rank) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(down_out)
                .arg_u32((num_tokens * self.rank) as u32)
                .arg_f32(self.hc_count as f32)
                .launch(stream)?;
            false
        };

        let mix_logits = buffers.qkv_output();
        let up_route = self.launch_w4a16_gemm_best(
            gpu,
            down_out,
            &self.up,
            mix_logits,
            num_tokens,
            r as u32,
            self.rank as u32,
            stream,
        )?;
        if !dual_silu_down && let Some(ref weight) = self.inject {
            if self.w4a16_gemv_rows_k.0 != 0
                && self.hc_count <= ops::W4A16_EXACT_ROWPAR_MAX_N as usize
            {
                ops::w4a16_gemv_rows(
                    gpu,
                    self.w4a16_gemv_rows_k,
                    residual,
                    weight,
                    inject_logits,
                    num_tokens as u32,
                    self.hc_count as u32,
                    r as u32,
                    stream,
                )?;
            } else {
                let _ = self.launch_w4a16_gemm_best(
                    gpu,
                    residual,
                    weight,
                    inject_logits,
                    num_tokens,
                    self.hc_count as u32,
                    r as u32,
                    stream,
                )?;
            }
        }

        let mixed = out_mixed.unwrap_or_else(|| buffers.norm_output());
        self.launch_mix_and_save_inject(
            gpu,
            residual,
            mix_logits,
            inject_logits,
            mixed,
            num_tokens,
            stream,
        )?;
        static GEMM_LOGGED: std::sync::Once = std::sync::Once::new();
        GEMM_LOGGED.call_once(|| {
            tracing::info!(
                selector = BATCHED_VERIFY_TC_DENSE_ENV,
                component = if self.inject.is_some() {
                    "hc_prepare"
                } else {
                    "final_mixer"
                },
                route = up_route,
                dual_silu_down,
                inject = self.inject.is_some(),
                rows = num_tokens,
                "QWEN4_BATCHED_VERIFY_TC_DENSE_ENGAGED"
            );
        });
        Ok(mixed)
    }
}

/// Rows and hidden indices per block of `qwen4_hc_up_mix_tail_rows`.
const HC_TAIL_MAX_ROWS: usize = 25;
const HC_TAIL_HIDDEN_PER_BLOCK: usize = 4;

/// ATLAS_QWEN4_HC_TAIL_ROWS=1 (default on): the batched-verify hyper read-out tail in one
/// byte-identical launch (verify-cost hc_tail_bench).
fn hc_tail_rows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_HC_TAIL_ROWS").ok().as_deref() != Some("0");
        if on {
            tracing::info!("HC_TAIL_ROWS_ENGAGED selector=ATLAS_QWEN4_HC_TAIL_ROWS");
        }
        on
    })
}

/// Receipt for the first dual down + inject launch (the env read alone
/// proves nothing).
static HC_DUAL_ROWS_LAUNCHED: std::sync::Once = std::sync::Once::new();

/// ATLAS_QWEN4_HC_DUAL_ROWS=1: the batched-verify hyper down and inject
/// projections in one byte-identical prefetching launch (verify-cost
/// hc_dual_rows_bench).
fn hc_dual_rows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_HC_DUAL_ROWS").is_ok_and(|v| v == "1")
            || crate::layers::qwen4_verify_fast_host_selected();
        if on {
            tracing::info!("HC_DUAL_ROWS_REQUESTED selector=ATLAS_QWEN4_HC_DUAL_ROWS");
        }
        on
    })
}

/// ATLAS_QWEN4_HC_GEMV_V2=1: byte-identical latency variants of the decode
/// hyper-connection GEMVs (fn-decode hc_gemv_test).
fn hc_gemv_v2_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_HC_GEMV_V2").is_ok_and(|v| v == "1")
            || crate::layers::qwen4_verify_fast_host_selected();
        if on {
            tracing::info!("HC_GEMV_V2_ENGAGED selector=ATLAS_QWEN4_HC_GEMV_V2");
        }
        on
    })
}

/// `ATLAS_QWEN4_BATCHED_VERIFY_TC_DENSE` (default ON; `"0"` disables): route
/// batched-verify dense projections onto Tensor Core kernels instead of scalar
/// GEMV tiers once a component reaches its minimum row threshold. Zero-alloc
/// direct NVFP4/BF16 routes (`LmHead`, `PleProj`, `FinalMixer`, `HcPrepare`,
/// and direct NVFP4 `Fp8Dense` for `rows <= 128`) are also safe inside CUDA
/// graph capture once arena buffers are pre-warmed. Single-sequence `C=1 K=5`
/// (`rows <= 5`) always remains 100% on the exact 0-ULP scalar GEMV path.
pub(crate) const BATCHED_VERIFY_TC_DENSE_ENV: &str = "ATLAS_QWEN4_BATCHED_VERIFY_TC_DENSE";

/// `ATLAS_QWEN4_BATCHED_VERIFY_TC_DENSE_MIN_ROWS`: optional uniform operator
/// floor overriding per-component Tensor Core minimum-row defaults when set to
/// a positive integer.
pub(crate) const BATCHED_VERIFY_TC_DENSE_MIN_ROWS_ENV: &str =
    "ATLAS_QWEN4_BATCHED_VERIFY_TC_DENSE_MIN_ROWS";

/// Default minimum rows for direct NVFP4/BF16 Tensor Core components
/// (`LmHead`, `PleProj`, `FinalMixer`, and fused `upmix_coalesced`): zero
/// weight bloat so `C=2` (`rows >= 4`) engages TC without hitting a `rows=4..6`
/// scalar GEMV cliff, while `C=1` remains 100% on strict 0-ULP exact scalar
/// GEMV via `BatchedVerifyTcDenseGuard::enter(false)` (`qwen4_c1_tc_verify_enabled()`
/// defaults OFF).
pub(crate) const DEFAULT_BATCHED_VERIFY_TC_DIRECT_MIN_ROWS: usize = 4;

/// Default minimum rows for layer dense projections (`Fp8Dense`: SSM
/// QKVZ/out_proj, Attention QG/K/V/o_proj, MoE router). `rows <= 3` stays on
/// exact scalar GEMV (`exact_m4`), while `rows = 4..=128` routes via
/// zero-VRAM, CUDA-graph-safe coalesced and Warp-Split-K NVFP4
/// (`w4a16_gemm_coalesced*`) with `attn.qkv > 48` FP8 handoff when warm,
/// eliminating the `C=2 rows=4..6` scalar GEMV cliff and slow `qg_exact_m17`.
pub(crate) const DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS: usize = 4;

/// Minimum row count at which `select_batched_hc_down_route` (the v1 fallback
/// when `DualCoalescedNvfp4` is unavailable) leaves the `rows <= 10` exact
/// small-row tier (`DualSiluM10` / `ExactSmallRows`) and switches to the medium
/// (`11..=25`: `CoalescedNvfp4 -> DualSiluExact -> Fp8Tensor -> W4a16Gemm`) and
/// wide (`>= 26`: `CoalescedNvfp4 -> Fp8Tensor -> DualSiluExact -> W4a16Gemm`)
/// fallback tiers.
const HC_TC_WIDE_DOWN_MIN_ROWS: usize = 11;

/// Maximum row count at which the v1 fallback (`select_batched_hc_down_route`)
/// prioritizes `DualSiluExact` (`exact_pf_dual_silu_rows_m25_k`) ahead of
/// `Fp8Tensor` when neither `DualCoalescedNvfp4` nor `CoalescedNvfp4` is ready.
const HC_TC_DUAL_M25_MAX_ROWS: usize = 25;

/// Dynamic shared memory (`kRoofSmemT128`) required by
/// `qwen4_hc_upmix_roof_t128_w4`.
const UPMIX_ROOF_T128_W4_SMEM_BYTES: u32 = 49_184;

/// 256-byte device scratch block holding `[w_amax: u32, a_amax: u32,
/// w_scale: f32, a_scale: f32]` for tensor-scale FP8 HC projections.
const HC_FP8_WS_BYTES: usize = 256;

/// Default minimum rows for injecting layer hyperconnections (`HcPrepare`):
/// `rows <= 3` stays on the single-sweep exact `DualSiluM10 + up_mix_tail_rows`
/// path; `rows >= 4` (including `C=2 r=4..8`) engages `FastTc` (`DualCoalescedNvfp4`
/// `n8_sk8_m32_dual` for `4..=24` and `n16_sk4_m16_dual_silu_div_v2` for
/// `25..=128`, plus fused `upmix_coalesced` across `4..=128`), with zero FP8
/// cache, zero warmup, and 100% CUDA-graph safety.
pub(crate) const DEFAULT_BATCHED_VERIFY_TC_HC_PREPARE_MIN_ROWS: usize = 4;

/// Batched-verify dense projection component with an independent default
/// Tensor Core minimum-row threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BatchedVerifyTcComponent {
    /// Terminal NVFP4 LM head (`w4a16_gemm_pipe*` / `w4a16_gemm_t_m32_n64`).
    LmHead,
    /// Cross-sequence BF16 PLE key/value projections (`cublas_bf16_proj_dense`).
    PleProj,
    /// Non-injecting terminal hyperconnection (`qwen4_final_mixer`).
    FinalMixer,
    /// Layer dense projections (SSM QKVZ/out_proj, Attention QG/K/V/o_proj):
    /// direct NVFP4 `w4a16_gemm_pipe*` for `rows <= 128`, FP8 cache above 128.
    Fp8Dense,
    /// Injecting layer hyperconnections (`FastTc`: `DualSiluM10` / coalesced
    /// NVFP4 `down` + coalesced NVFP4 `up` + L2-resident `mix_save_inject`,
    /// zero FP8 cache, zero `s.folded` staging, 100% step-0 graph-safe).
    HcPrepare,
}

impl BatchedVerifyTcComponent {
    /// Default minimum row count for this component when
    /// `ATLAS_QWEN4_BATCHED_VERIFY_TC_DENSE_MIN_ROWS` is not explicitly set.
    #[inline]
    #[must_use]
    pub(crate) const fn default_min_rows(self) -> usize {
        match self {
            Self::LmHead | Self::PleProj | Self::FinalMixer => {
                DEFAULT_BATCHED_VERIFY_TC_DIRECT_MIN_ROWS
            }
            Self::Fp8Dense => DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS,
            Self::HcPrepare => DEFAULT_BATCHED_VERIFY_TC_HC_PREPARE_MIN_ROWS,
        }
    }

    /// Whether this component's TC route at `rows` has zero lazy allocations
    /// and zero host-GPU syncs, making it safe inside CUDA graph capture once
    /// arena buffers are pre-warmed.
    #[inline]
    #[must_use]
    pub(crate) const fn graph_capture_safe_for_rows(self, rows: usize) -> bool {
        match self {
            Self::LmHead | Self::PleProj | Self::FinalMixer | Self::HcPrepare => true,
            // `rows <= 128` dispatches directly to zero-alloc NVFP4 `w4a16_gemm_pipe*`;
            // wider rows use the lazy FP8 weight cache / scratch.
            Self::Fp8Dense => rows <= crate::layers::QWEN4_NATIVE_MAX_ROWS,
        }
    }
}

#[inline]
#[must_use]
pub(crate) fn parse_batched_verify_tc_dense(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("0")
}

/// Parse an explicit positive integer override from
/// `ATLAS_QWEN4_BATCHED_VERIFY_TC_DENSE_MIN_ROWS`; returns `None` when unset,
/// zero, or malformed so per-component defaults apply.
#[inline]
#[must_use]
pub(crate) fn parse_batched_verify_tc_dense_min_rows_override(raw: Option<&str>) -> Option<usize> {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
}

#[must_use]
pub(crate) fn batched_verify_tc_dense_enabled() -> bool {
    if crate::layers::deterministic_batch::deterministic_batch_enabled() {
        return false;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        parse_batched_verify_tc_dense(std::env::var(BATCHED_VERIFY_TC_DENSE_ENV).ok().as_deref())
    })
}

#[must_use]
pub(crate) fn batched_verify_tc_dense_min_rows_override() -> Option<usize> {
    static MIN_ROWS_OVERRIDE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *MIN_ROWS_OVERRIDE.get_or_init(|| {
        parse_batched_verify_tc_dense_min_rows_override(
            std::env::var(BATCHED_VERIFY_TC_DENSE_MIN_ROWS_ENV)
                .ok()
                .as_deref(),
        )
    })
}

#[inline]
#[must_use]
pub(crate) const fn batched_verify_tc_dense_admits(
    enabled: bool,
    min_rows: usize,
    rows: usize,
    graph_capture: bool,
) -> bool {
    enabled && !graph_capture && rows >= min_rows
}

#[inline]
#[must_use]
pub(crate) const fn batched_verify_tc_component_admits_with_policy(
    component: BatchedVerifyTcComponent,
    enabled: bool,
    min_rows_override: Option<usize>,
    rows: usize,
    graph_capture: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> bool {
    if det_batch.is_enabled() {
        return false;
    }
    batched_verify_tc_component_admits(component, enabled, min_rows_override, rows, graph_capture)
}

#[inline]
#[must_use]
pub(crate) const fn batched_verify_tc_component_admits(
    component: BatchedVerifyTcComponent,
    enabled: bool,
    min_rows_override: Option<usize>,
    rows: usize,
    graph_capture: bool,
) -> bool {
    let min_rows = match min_rows_override {
        Some(n) => n,
        None => component.default_min_rows(),
    };
    let capture_blocked = graph_capture && !component.graph_capture_safe_for_rows(rows);
    batched_verify_tc_dense_admits(enabled, min_rows, rows, capture_blocked)
}

#[inline]
#[must_use]
pub(crate) fn batched_verify_tc_component_active(
    component: BatchedVerifyTcComponent,
    rows: usize,
    graph_capture: bool,
) -> bool {
    if graph_capture
        && component == BatchedVerifyTcComponent::Fp8Dense
        && !crate::layers::qwen4_fast_proj::direct_nvfp4_dense_enabled()
        && !crate::layers::qwen4_fast_proj::coalesced_nvfp4_dense_enabled()
    {
        return false;
    }
    batched_verify_tc_component_admits_with_policy(
        component,
        batched_verify_tc_dense_enabled(),
        batched_verify_tc_dense_min_rows_override(),
        rows,
        graph_capture,
        crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
    )
}

#[inline]
#[must_use]
pub(crate) fn batched_verify_tc_dense_active(rows: usize, graph_capture: bool) -> bool {
    batched_verify_tc_component_active(BatchedVerifyTcComponent::LmHead, rows, graph_capture)
}

thread_local! {
    static BATCHED_VERIFY_TC_DENSE_SCOPE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Scoped opt-in for [`Qwen4HyperConnection::prepare_batched`] and
/// [`Qwen4HyperConnection::prepare_batched_into`] inside cross-sequence batched
/// verify frames so exact single-sequence / native-decode / prefill callers of
/// `prepare_batched` remain strictly untouched.
pub(crate) struct BatchedVerifyTcDenseGuard(bool);

impl BatchedVerifyTcDenseGuard {
    #[must_use]
    pub(crate) fn enter(active: bool) -> Self {
        let prev = BATCHED_VERIFY_TC_DENSE_SCOPE.with(|cell| cell.replace(active));
        Self(prev)
    }
}

impl Drop for BatchedVerifyTcDenseGuard {
    fn drop(&mut self) {
        BATCHED_VERIFY_TC_DENSE_SCOPE.with(|cell| cell.set(self.0));
    }
}

#[inline]
#[must_use]
pub(crate) fn batched_verify_tc_dense_scope_active() -> bool {
    BATCHED_VERIFY_TC_DENSE_SCOPE.with(std::cell::Cell::get)
}

/// Enable CUDA Programmatic Dependent Launch (`PDL`) across the hot
/// `inject_group_norm_v2 -> hc_dual -> hc_upmix` chain on high-SM Blackwell
/// GPUs (`sm_count > 64`), keeping GB10 (`sm_count <= 64`) unchanged.
#[inline]
#[must_use]
const fn hc_pdl_enabled_for_sm(sm_count: u32) -> bool {
    sm_count > DUAL_SPLITK_HIGH_SM_THRESHOLD
}

/// Deferred single-sequence verify MLP `inject_saved_batched` waiting to fuse
/// into the next layer's `attn_hyper.prepare_batched_into_after_inject` (or
/// `qwen4_final_mixer.prepare_batched_after_inject` after the final layer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PendingVerifyHcInject {
    pub hyper: DevicePtr,
    pub core: DevicePtr,
    pub residual: DevicePtr,
    pub rows: usize,
    pub stream: u64,
}

impl PendingVerifyHcInject {
    #[inline]
    #[must_use]
    pub(crate) const fn matches(
        self,
        hyper: DevicePtr,
        residual: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> bool {
        self.hyper.0 == hyper.0
            && self.residual.0 == residual.0
            && self.rows == rows
            && self.stream == stream
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct VerifyHcFuseState {
    defer_mlp_tail_allowed: bool,
    pending: Option<PendingVerifyHcInject>,
}

thread_local! {
    static VERIFY_HC_FUSE: std::cell::Cell<VerifyHcFuseState> = const {
        std::cell::Cell::new(VerifyHcFuseState {
            defer_mlp_tail_allowed: false,
            pending: None,
        })
    };
}

/// Scoped inter-layer deferred MLP-inject state for single-sequence verify
/// (`verify_d.rs`). Dropping the guard restores the prior state and discards
/// any unconsumed entry so early errors or aborted CUDA graph captures never
/// leak stale device pointers into a subsequent verify round.
pub(crate) struct VerifyHcFuseScope {
    prev: VerifyHcFuseState,
}

impl VerifyHcFuseScope {
    #[must_use]
    pub(crate) fn enter() -> Self {
        let prev = VERIFY_HC_FUSE.with(|cell| {
            cell.replace(VerifyHcFuseState {
                defer_mlp_tail_allowed: false,
                pending: None,
            })
        });
        Self { prev }
    }

    /// Arm or disarm trailing MLP `inject_saved_batched` deferral for the next
    /// layer inside this scope. Must remain `false` for Layer 0 (PLE runs
    /// between Layer 0 and Layer 1) and every DFlash capture tap layer.
    pub(crate) fn set_defer_mlp_tail(&mut self, allow: bool) {
        VERIFY_HC_FUSE.with(|cell| {
            let mut state = cell.get();
            state.defer_mlp_tail_allowed = allow;
            cell.set(state);
        });
    }
}

impl Drop for VerifyHcFuseScope {
    fn drop(&mut self) {
        VERIFY_HC_FUSE.with(|cell| cell.set(self.prev));
    }
}

/// Whether single-sequence verify inter-layer `mlp.inject + next_attn.group_norm`
/// fusion is enabled (shares `ATLAS_QWEN4_BATCHED_HC_FUSE_INJECT_GN`, default ON
/// unless `"0"`).
#[inline]
#[must_use]
pub(crate) fn verify_hc_fuse_inject_gn_enabled() -> bool {
    crate::layers::qwen4_cross_seq_mlp::batched_hc_fuse_inject_gn_enabled()
}

#[inline]
#[must_use]
pub(crate) fn verify_hc_defer_mlp_tail_allowed() -> bool {
    VERIFY_HC_FUSE.with(|cell| cell.get().defer_mlp_tail_allowed)
}

pub(crate) fn defer_pending_verify_hc_inject(entry: PendingVerifyHcInject) -> Result<()> {
    VERIFY_HC_FUSE.with(|cell| {
        let mut state = cell.get();
        if let Some(prior) = state.pending {
            bail!(
                "pending single-seq verify HC inject {prior:?} was not consumed before deferring {entry:?}"
            );
        }
        state.pending = Some(entry);
        cell.set(state);
        Ok(())
    })
}

#[inline]
pub(crate) fn take_pending_verify_hc_inject() -> Option<PendingVerifyHcInject> {
    VERIFY_HC_FUSE.with(|cell| {
        let mut state = cell.get();
        let pending = state.pending.take();
        cell.set(state);
        pending
    })
}

/// Lock-free 256-row bitset (`1..=256`) recording batch widths for which an
/// HC projection has completed an eager FP8 pass (warming cuBLASLt plans,
/// `WEIGHT_AMAX_CACHE`, `fp8_w8_scratch`, and optionally `WEIGHT_CACHE`) so
/// subsequent CUDA graph captures run with zero lazy allocations or syncs.
struct WarmedRowsMask {
    words: [std::sync::atomic::AtomicU64; 4],
}

impl WarmedRowsMask {
    const fn new() -> Self {
        Self {
            words: [
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU64::new(0),
            ],
        }
    }

    #[inline]
    fn is_warmed(&self, rows: usize) -> bool {
        if (1..=256).contains(&rows) {
            let idx = (rows - 1) / 64;
            let bit = 1u64 << ((rows - 1) % 64);
            (self.words[idx].load(std::sync::atomic::Ordering::Acquire) & bit) != 0
        } else {
            false
        }
    }

    #[inline]
    fn mark_warmed(&self, rows: usize) {
        if (1..=256).contains(&rows) {
            let idx = (rows - 1) / 64;
            let bit = 1u64 << ((rows - 1) % 64);
            self.words[idx].fetch_or(bit, std::sync::atomic::Ordering::Release);
        }
    }
}

#[inline]
fn align_up_256(n: usize) -> Option<usize> {
    Some(n.checked_add(255)? & !255)
}

/// Disjoint 256-byte-aligned scratch regions carved out of idle
/// `buffers.qkv_output()` during [`Qwen4HyperConnection::prepare_batched_tc_fast`].
///
/// - `down_a8` (`rows * residual_width` B), `down_ws` (256 B), and `rstd`
///   (`rows * hc_count * 4` B) are mutually disjoint so `qwen4_hc_rstd` can
///   run before `down` and remain intact until `qwen4_hc_upmix_roof_t128_w4`.
/// - `up_a8` (`rows * rank` B) and `up_ws` (256 B) sit strictly after the
///   `rows * residual_width * 2` BF16 `mix_logits` prefix used only by the
///   non-canonical `BatchedHcUpRoute::Fp8Tensor` fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchedHcQkvScratchLayout {
    down_a8_off: usize,
    down_ws_off: usize,
    rstd_off: usize,
    total_down_rstd_bytes: usize,
    up_a8_off: usize,
    up_ws_off: usize,
    total_up_fp8_bytes: usize,
}

impl BatchedHcQkvScratchLayout {
    fn compute(rows: usize, residual_width: usize, rank: usize, hc_count: usize) -> Option<Self> {
        if rows == 0 || residual_width == 0 || rank == 0 || hc_count == 0 {
            return None;
        }
        let down_a8_bytes = rows.checked_mul(residual_width)?;
        let down_ws_off = align_up_256(down_a8_bytes)?;
        let rstd_off = align_up_256(down_ws_off.checked_add(HC_FP8_WS_BYTES)?)?;
        let rstd_bytes = rows.checked_mul(hc_count)?.checked_mul(4)?;
        let total_down_rstd_bytes = rstd_off.checked_add(rstd_bytes)?;

        let mix_logits_bytes = down_a8_bytes.checked_mul(2)?;
        let up_a8_off = align_up_256(mix_logits_bytes)?;
        let up_a8_bytes = rows.checked_mul(rank)?;
        let up_ws_off = align_up_256(up_a8_off.checked_add(up_a8_bytes)?)?;
        let total_up_fp8_bytes = up_ws_off.checked_add(HC_FP8_WS_BYTES)?;

        Some(Self {
            down_a8_off: 0,
            down_ws_off,
            rstd_off,
            total_down_rstd_bytes,
            up_a8_off,
            up_ws_off,
            total_up_fp8_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchedHcTcRoute {
    Exact,
    FastTc,
    W4a16Gemm,
}

#[inline]
#[must_use]
const fn select_batched_hc_tc_route_with_policy(
    tc_dense_active: bool,
    fast_tc_ready: bool,
    w4a16_gemm_ready: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> BatchedHcTcRoute {
    if det_batch.is_enabled() {
        return BatchedHcTcRoute::Exact;
    }
    select_batched_hc_tc_route(tc_dense_active, fast_tc_ready, w4a16_gemm_ready)
}

#[inline]
#[must_use]
const fn select_batched_hc_tc_route(
    tc_dense_active: bool,
    fast_tc_ready: bool,
    w4a16_gemm_ready: bool,
) -> BatchedHcTcRoute {
    if !tc_dense_active {
        return BatchedHcTcRoute::Exact;
    }
    if fast_tc_ready {
        BatchedHcTcRoute::FastTc
    } else if w4a16_gemm_ready {
        BatchedHcTcRoute::W4a16Gemm
    } else {
        BatchedHcTcRoute::Exact
    }
}

pub(crate) const HC_DUAL_COALESCED_ENV: &str = "ATLAS_QWEN4_HC_DUAL_COALESCED";
pub(crate) const HC_DUAL_N32_ENV: &str = "ATLAS_QWEN4_HC_DUAL_N32";
pub(crate) const HC_UPMIX_COALESCED_ENV: &str = "ATLAS_QWEN4_HC_UPMIX_COALESCED";
const HC_DUAL_N8_MAX_ROWS: usize = 24;
const HC_UPMIX_COALESCED_MIN_ROWS: usize = DEFAULT_BATCHED_VERIFY_TC_DIRECT_MIN_ROWS;
const HC_UPMIX_COALESCED_MAX_ROWS: usize = crate::layers::QWEN4_NATIVE_MAX_ROWS;

#[inline]
#[must_use]
pub(crate) fn parse_hc_dual_coalesced(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("0")
}

/// `ATLAS_QWEN4_HC_DUAL_COALESCED` (default ON; `"0"` disables): fuse
/// `hc.down + silu_div + hc.inject` into a single launch across `rows in 7..=128`
/// (`n8_sk8_m32_dual_silu_div` for `rows = 7..=24` and
/// `n16_sk4_m16_dual_silu_div_v2` — degrading to `n32_sk4_m16_dual_silu_div`
/// when absent — for `rows = 25..=128`).
#[must_use]
pub(crate) fn hc_dual_coalesced_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        parse_hc_dual_coalesced(std::env::var(HC_DUAL_COALESCED_ENV).ok().as_deref())
    })
}

#[inline]
#[must_use]
pub(crate) fn parse_hc_dual_n32(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("0")
}

/// `ATLAS_QWEN4_HC_DUAL_N32` (default ON; `"0"` disables): enable the wide-row
/// dual `hc.down + silu_div + hc.inject` kernel (`w4a16_gemm_coalesced_n16_sk4_m16_dual_silu_div_v2`,
/// 21 N-blocks with 2-stage K-prefetch, falling back to `n32_sk4_m16_dual_silu_div`)
/// for `rows in 25..=128` (avoiding 41x activation rereads from `n8_sk8_m32_dual`,
/// which is capped strictly at `rows <= 24`).
#[must_use]
pub(crate) fn hc_dual_n32_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| parse_hc_dual_n32(std::env::var(HC_DUAL_N32_ENV).ok().as_deref()))
}

/// SM count threshold above which high-occupancy `TILE_M = 32/64` grid-Split-K
/// dual `hc.down + hc.inject` kernels engage (SM120/SM103 discrete Blackwell;
/// GB10 with `<= 64` SMs stays on the exact existing `m16`/`m32` tiers).
const DUAL_SPLITK_HIGH_SM_THRESHOLD: u32 = 64;
/// Maximum FP32 partial elements in the shared L2-hot grid-Split-K workspace
/// (`20` K-splits x `4` M-tiles x `6` N-tiles x `64 * 64` = `1,966,080` floats
/// = `7.5 MiB`, covering every high-SM dual tier across `M = 25..=256`).
const DUAL_SPLITK_MAX_FLOATS: usize = 20 * 4 * 6 * 64 * 64;
/// Maximum `(grid_x * grid_y)` tile semaphore counters (`512 * 4 B = 2 KiB`).
const DUAL_SPLITK_MAX_TILES: usize = 512;
/// Maximum inter-CTA K-splits (`grid.z`) admitted by the grid-Split-K planner.
const DUAL_SPLITK_MAX_Z: u32 = 20;

#[derive(Clone, Copy, Debug)]
struct DualHighSmKernels {
    n8_sk8_m32_v2: KernelHandle,
    n16_sk4_m64_v2: KernelHandle,
    n32_sk2_m64_v2: KernelHandle,
    n64_sk1_m64_v2: KernelHandle,
}

impl DualHighSmKernels {
    const fn empty() -> Self {
        Self {
            n8_sk8_m32_v2: KernelHandle(0),
            n16_sk4_m64_v2: KernelHandle(0),
            n32_sk2_m64_v2: KernelHandle(0),
            n64_sk1_m64_v2: KernelHandle(0),
        }
    }

    fn load(gpu: &dyn GpuBackend) -> Self {
        Self {
            n8_sk8_m32_v2: try_w4a16_kernel(
                gpu,
                "w4a16_gemm_coalesced_n8_sk8_m32_dual_silu_div_v2",
            ),
            n16_sk4_m64_v2: try_w4a16_kernel(
                gpu,
                "w4a16_gemm_coalesced_n16_sk4_m64_dual_silu_div_v2",
            ),
            n32_sk2_m64_v2: try_w4a16_kernel(
                gpu,
                "w4a16_gemm_coalesced_n32_sk2_m64_dual_silu_div_v2",
            ),
            n64_sk1_m64_v2: try_w4a16_kernel(
                gpu,
                "w4a16_gemm_coalesced_n64_sk1_m64_dual_silu_div_v2",
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct DualSplitKScratch {
    partials: DevicePtr,
    counters: DevicePtr,
}

fn dual_splitk_scratch_for_backend(gpu: &dyn GpuBackend) -> Option<DualSplitKScratch> {
    static PRIMARY: std::sync::OnceLock<(usize, Option<DualSplitKScratch>)> =
        std::sync::OnceLock::new();
    static FALLBACK: std::sync::OnceLock<
        parking_lot::Mutex<std::collections::HashMap<usize, Option<DualSplitKScratch>>>,
    > = std::sync::OnceLock::new();

    let alloc = || -> Option<DualSplitKScratch> {
        let partials_bytes = DUAL_SPLITK_MAX_FLOATS * std::mem::size_of::<f32>();
        let counters_bytes = DUAL_SPLITK_MAX_TILES * std::mem::size_of::<u32>();
        let partials = gpu.alloc(partials_bytes).ok()?;
        let counters = match gpu.alloc(counters_bytes) {
            Ok(ptr) => ptr,
            Err(_) => {
                let _ = gpu.free(partials);
                return None;
            }
        };
        // Grid-Split-K tile semaphore requires counters initialized strictly to 0
        // regardless of `ATLAS_PREFILL_SCRATCH_INIT`.
        if gpu.memset(counters, 0, counters_bytes).is_err() {
            let _ = gpu.free(counters);
            let _ = gpu.free(partials);
            return None;
        }
        Some(DualSplitKScratch { partials, counters })
    };

    let key = backend_cache_key(gpu);
    let &(primary_key, primary_val) = PRIMARY.get_or_init(|| (key, alloc()));
    if primary_key == key {
        return primary_val;
    }
    let map = FALLBACK.get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
    let mut guard = map.lock();
    *guard.entry(key).or_insert_with(alloc)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DualCoalescedTier {
    /// `CO_N = 8, SPLIT_K = 8, TILE_M = 32` (`blk = 256`), optimal for `rows = 4..=24`.
    N8Sk8M32,
    /// `CO_N = 16, SPLIT_K = 4, TILE_M = 16` (`blk = 256`, v2 2-stage K-prefetch,
    /// `21` N-blocks x `ceil(M/16)` saturating all 48 SMs), optimal on GB10 for `rows = 25..=256`.
    N16Sk4M16V2,
    /// `CO_N = 32, SPLIT_K = 4, TILE_M = 16` (`blk = 512`, v1 K-loop), fallback for `rows = 25..=256`.
    N32Sk4M16,
    /// Grid-Split-K `CO_N = 8, SPLIT_K = 8, TILE_M = 32` (`blk = 256`), SM120
    /// last-resort fallback for `rows = 25..=32` when `N32Sk2M64V2`,
    /// `N16Sk4M64V2`, and `N64Sk1M64V2` are unavailable.
    N8Sk8M32V2,
    /// Grid-Split-K `CO_N = 16, SPLIT_K = 4, TILE_M = 64` (`blk = 256`), SM120
    /// fallback for `rows = 25..=192` (after `N32Sk2M64V2`) and `193..=256`
    /// (after `N64Sk1M64V2` and `N32Sk2M64V2`).
    N16Sk4M64V2,
    /// Grid-Split-K `CO_N = 32, SPLIT_K = 2, TILE_M = 64` (`blk = 256`), SM120
    /// primary for `rows = 25..=192` and fallback for `rows = 193..=256`.
    N32Sk2M64V2,
    /// Grid-Split-K `CO_N = 64, SPLIT_K = 1, TILE_M = 64` (`blk = 256`), SM120
    /// primary for `rows = 193..=256` and fallback for `rows = 25..=192`.
    N64Sk1M64V2,
}

impl DualCoalescedTier {
    #[inline]
    #[must_use]
    const fn co_n(self) -> u32 {
        match self {
            Self::N8Sk8M32 | Self::N8Sk8M32V2 => 8,
            Self::N16Sk4M16V2 | Self::N16Sk4M64V2 => 16,
            Self::N32Sk4M16 | Self::N32Sk2M64V2 => 32,
            Self::N64Sk1M64V2 => 64,
        }
    }

    #[inline]
    #[must_use]
    const fn tile_m(self) -> u32 {
        match self {
            Self::N8Sk8M32 | Self::N8Sk8M32V2 => 32,
            Self::N16Sk4M16V2 | Self::N32Sk4M16 => 16,
            Self::N16Sk4M64V2 | Self::N32Sk2M64V2 | Self::N64Sk1M64V2 => 64,
        }
    }

    #[inline]
    #[must_use]
    const fn threads(self) -> u32 {
        match self {
            Self::N8Sk8M32
            | Self::N16Sk4M16V2
            | Self::N8Sk8M32V2
            | Self::N16Sk4M64V2
            | Self::N32Sk2M64V2
            | Self::N64Sk1M64V2 => 256,
            Self::N32Sk4M16 => 512,
        }
    }

    #[inline]
    #[must_use]
    const fn supports_grid_split_k(self) -> bool {
        matches!(
            self,
            Self::N8Sk8M32V2 | Self::N16Sk4M64V2 | Self::N32Sk2M64V2 | Self::N64Sk1M64V2
        )
    }

    #[inline]
    #[must_use]
    const fn k_step_cta(self) -> u32 {
        match self {
            Self::N8Sk8M32 | Self::N8Sk8M32V2 => 2_048,
            Self::N16Sk4M16V2 | Self::N32Sk4M16 | Self::N16Sk4M64V2 => 1_024,
            Self::N32Sk2M64V2 => 512,
            Self::N64Sk1M64V2 => 256,
        }
    }
}

/// Choose the inter-CTA K-split count (`grid.z`) for a grid-Split-K dual launch.
///
/// Always returns `1` on GB10 (`sm_count <= 64`), on non-grid-Split-K tiers, or
/// when the pre-allocated workspace is absent. On high-SM Blackwell targets
/// (`sm_count > 64`), chooses the largest balanced divisor of `K / k_step_cta`
/// (falling back to divisors of `K / 256` if needed to saturate `sm_count`)
/// whose total CTA count fits within one 2-CTA/SM wave (`<= 2 * sm_count`) and
/// strictly within the pre-allocated FP32 partial workspace.
#[inline]
#[must_use]
const fn select_dual_split_k_grid_z_with_policy(
    tier: DualCoalescedTier,
    grid_x: u32,
    grid_y: u32,
    k: usize,
    sm_count: u32,
    has_scratch: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> u32 {
    if det_batch.is_enabled() {
        return 1;
    }
    select_dual_split_k_grid_z(tier, grid_x, grid_y, k, sm_count, has_scratch)
}

#[inline]
#[must_use]
const fn select_dual_split_k_grid_z(
    tier: DualCoalescedTier,
    grid_x: u32,
    grid_y: u32,
    k: usize,
    sm_count: u32,
    has_scratch: bool,
) -> u32 {
    if !tier.supports_grid_split_k()
        || sm_count <= DUAL_SPLITK_HIGH_SM_THRESHOLD
        || !has_scratch
        || k < 256
        || grid_x == 0
        || grid_y == 0
    {
        return 1;
    }
    let sm_count_usize = sm_count as usize;
    let num_tiles = (grid_x as usize).saturating_mul(grid_y as usize);
    if num_tiles == 0 || num_tiles > DUAL_SPLITK_MAX_TILES {
        return 1;
    }
    let tile_elems = (tier.tile_m() as usize).saturating_mul(tier.co_n() as usize);
    let per_z_floats = num_tiles.saturating_mul(tile_elems);
    if per_z_floats == 0 || per_z_floats > DUAL_SPLITK_MAX_FLOATS {
        return 1;
    }
    let max_z_scratch = (DUAL_SPLITK_MAX_FLOATS / per_z_floats) as u32;
    let max_z = if max_z_scratch < DUAL_SPLITK_MAX_Z {
        max_z_scratch
    } else {
        DUAL_SPLITK_MAX_Z
    };
    let target_max_ctas = sm_count_usize.saturating_mul(2);

    // Pass 1: largest exact divisor of `K / k_step_cta` (0% intra-CTA warp tail).
    let k_step = tier.k_step_cta() as usize;
    let mut best = 1u32;
    if k.is_multiple_of(k_step) {
        let units = (k / k_step) as u32;
        let mut z = if max_z < units { max_z } else { units };
        while z > 1 {
            if units.is_multiple_of(z) && num_tiles.saturating_mul(z as usize) <= target_max_ctas {
                best = z;
                break;
            }
            z -= 1;
        }
    }
    if num_tiles.saturating_mul(best as usize) >= sm_count_usize {
        return best;
    }

    // Pass 2: if Pass 1 still leaves SMs idle (e.g. `n16_sk4_m64` at `M=256`,
    // `84` tiles where `gz=4 -> 336 CTAs`), pick the largest divisor of `K / 256`
    // within `<= 2 * sm_count`.
    if k.is_multiple_of(256) {
        let units256 = (k / 256) as u32;
        let mut z = if max_z < units256 { max_z } else { units256 };
        while z > best {
            if units256.is_multiple_of(z) && num_tiles.saturating_mul(z as usize) <= target_max_ctas
            {
                return z;
            }
            z -= 1;
        }
    }
    best
}

/// Resolve the dual coalesced `hc.down + silu_div + hc.inject` kernel for `rows`
/// on GB10 (`sm_count <= 64`) or SM120/SM103 (`sm_count > 64`):
/// - `rows = 4..=24` on GB10 (`sm_count <= 64`) -> `w4a16_gemm_coalesced_n8_sk8_m32_dual_silu_div`
/// - `rows = 4..=24` on high-SM Blackwell (`sm_count > 64`) -> `n8_sk8_m32_dual_silu_div_v2`
///   (`grid_z = 5` -> `205` CTAs saturating 188 SMs, falling back to `n8_sk8_m32_dual_silu_div`)
/// - `rows = 25..=256` on GB10 (`sm_count <= 64`) -> `n16_sk4_m16_dual_silu_div_v2`
///   (falling back to `n32_sk4_m16_dual_silu_div`).
/// - `rows = 25..=256` on high-SM Blackwell (`sm_count > 64`):
///   - `25..=192`: `n32_sk2_m64_dual_silu_div_v2` -> `n16_sk4_m64_dual_silu_div_v2` -> `n64_sk1_m64_v2` -> `n8_sk8_m32_v2` -> `n16_sk4_m16_v2`
///   - `193..=256`: `n64_sk1_m64_dual_silu_div_v2` -> `n32_sk2_m64_dual_silu_div_v2` -> `n16_sk4_m64_dual_silu_div_v2` -> `n16_sk4_m16_v2`
#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
const fn resolve_hc_dual_coalesced_for_sms(
    rows: usize,
    dual_n8_k: KernelHandle,
    dual_n16_v2_k: KernelHandle,
    dual_n32_k: KernelHandle,
    high_sm_k: DualHighSmKernels,
    dual_wide_enabled: bool,
    sm_count: u32,
) -> Option<(DualCoalescedTier, KernelHandle)> {
    if rows < DEFAULT_BATCHED_VERIFY_TC_HC_PREPARE_MIN_ROWS
        || rows > crate::layers::QWEN4_NATIVE_MAX_ROWS
    {
        return None;
    }
    if rows <= HC_DUAL_N8_MAX_ROWS {
        if sm_count > DUAL_SPLITK_HIGH_SM_THRESHOLD && high_sm_k.n8_sk8_m32_v2.0 != 0 {
            Some((DualCoalescedTier::N8Sk8M32V2, high_sm_k.n8_sk8_m32_v2))
        } else if dual_n8_k.0 != 0 {
            Some((DualCoalescedTier::N8Sk8M32, dual_n8_k))
        } else {
            None
        }
    } else if !dual_wide_enabled {
        None
    } else if sm_count > DUAL_SPLITK_HIGH_SM_THRESHOLD {
        if rows <= 192 {
            if high_sm_k.n32_sk2_m64_v2.0 != 0 {
                return Some((DualCoalescedTier::N32Sk2M64V2, high_sm_k.n32_sk2_m64_v2));
            }
            if high_sm_k.n16_sk4_m64_v2.0 != 0 {
                return Some((DualCoalescedTier::N16Sk4M64V2, high_sm_k.n16_sk4_m64_v2));
            }
            if high_sm_k.n64_sk1_m64_v2.0 != 0 {
                return Some((DualCoalescedTier::N64Sk1M64V2, high_sm_k.n64_sk1_m64_v2));
            }
            if rows <= 32 && high_sm_k.n8_sk8_m32_v2.0 != 0 {
                return Some((DualCoalescedTier::N8Sk8M32V2, high_sm_k.n8_sk8_m32_v2));
            }
        } else {
            if high_sm_k.n64_sk1_m64_v2.0 != 0 {
                return Some((DualCoalescedTier::N64Sk1M64V2, high_sm_k.n64_sk1_m64_v2));
            }
            if high_sm_k.n32_sk2_m64_v2.0 != 0 {
                return Some((DualCoalescedTier::N32Sk2M64V2, high_sm_k.n32_sk2_m64_v2));
            }
            if high_sm_k.n16_sk4_m64_v2.0 != 0 {
                return Some((DualCoalescedTier::N16Sk4M64V2, high_sm_k.n16_sk4_m64_v2));
            }
        }
        if dual_n16_v2_k.0 != 0 {
            Some((DualCoalescedTier::N16Sk4M16V2, dual_n16_v2_k))
        } else if dual_n32_k.0 != 0 {
            Some((DualCoalescedTier::N32Sk4M16, dual_n32_k))
        } else {
            None
        }
    } else if dual_n16_v2_k.0 != 0 {
        Some((DualCoalescedTier::N16Sk4M16V2, dual_n16_v2_k))
    } else if dual_n32_k.0 != 0 {
        Some((DualCoalescedTier::N32Sk4M16, dual_n32_k))
    } else {
        None
    }
}

/// GB10 (`<= 64` SM) wrapper for [`resolve_hc_dual_coalesced_for_sms`].
#[cfg(test)]
#[inline]
#[must_use]
const fn resolve_hc_dual_coalesced(
    rows: usize,
    dual_n8_k: KernelHandle,
    dual_n16_v2_k: KernelHandle,
    dual_n32_k: KernelHandle,
    dual_wide_enabled: bool,
) -> Option<(DualCoalescedTier, KernelHandle)> {
    resolve_hc_dual_coalesced_for_sms(
        rows,
        dual_n8_k,
        dual_n16_v2_k,
        dual_n32_k,
        DualHighSmKernels::empty(),
        dual_wide_enabled,
        48,
    )
}

#[inline]
#[must_use]
pub(crate) fn parse_hc_upmix_coalesced(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("0")
}

/// `ATLAS_QWEN4_HC_UPMIX_COALESCED` (default ON; `"0"` disables): fuse
/// `hc.up + mix_save_inject` into a single coalesced NVFP4 launch
/// (`qwen4_hc_upmix_coalesced_m16/m32/m64/m80`, with `m64` running `grid.y = 2`
/// across `rows in 81..=128`) for `rows in 4..=128`, eliminating `mix_logits`
/// DRAM traffic and the trailing `mix_save_inject` kernel launch.
#[must_use]
pub(crate) fn hc_upmix_coalesced_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        parse_hc_upmix_coalesced(std::env::var(HC_UPMIX_COALESCED_ENV).ok().as_deref())
    })
}

#[inline]
#[must_use]
const fn inject_group_norm_v2_eligible(
    hidden_size: usize,
    hc_count: usize,
    hyper_ptr: u64,
    core_ptr: u64,
    weight_ptr: u64,
    residual_ptr: u64,
) -> bool {
    hidden_size == 2_560
        && hc_count == 4
        && hyper_ptr != 0
        && core_ptr != 0
        && weight_ptr != 0
        && residual_ptr != 0
        && hyper_ptr != residual_ptr
        && core_ptr != residual_ptr
        && ((hyper_ptr | core_ptr | weight_ptr | residual_ptr) & 15) == 0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchedHcDownRoute {
    DualCoalescedNvfp4,
    DualSiluM10,
    ExactSmallRows,
    CoalescedNvfp4,
    Fp8Tensor,
    DualSiluExact,
    W4a16Gemm,
}

impl BatchedHcDownRoute {
    #[inline]
    #[must_use]
    const fn label(self) -> &'static str {
        match self {
            Self::DualCoalescedNvfp4 => "dual_coalesced_nvfp4",
            Self::DualSiluM10 => "dual_silu_m10",
            Self::ExactSmallRows => "exact_small_rows",
            Self::CoalescedNvfp4 => "coalesced_nvfp4",
            Self::Fp8Tensor => "fp8_tensor",
            Self::DualSiluExact => "dual_silu_exact",
            Self::W4a16Gemm => "w4a16_gemm",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchedHcUpRoute {
    UpmixCoalesced,
    CoalescedNvfp4,
    UpmixRoofW4,
    Fp8Tensor,
    W4a16Gemm,
}

impl BatchedHcUpRoute {
    #[inline]
    #[must_use]
    const fn label(self) -> &'static str {
        match self {
            Self::UpmixCoalesced => "upmix_coalesced",
            Self::CoalescedNvfp4 => "coalesced_nvfp4",
            Self::UpmixRoofW4 => "upmix_roof_t128_w4",
            Self::Fp8Tensor => "fp8_tensor",
            Self::W4a16Gemm => "w4a16_gemm",
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
const fn hc_dual_coalesced_eligible(
    enabled: bool,
    kernel_loaded: bool,
    weights_valid: bool,
    rows: usize,
    rank: usize,
    hc_count: usize,
    k: usize,
    act_ptr: u64,
    down_weight_ptr: u64,
    down_scale_ptr: u64,
    down_out_ptr: u64,
    inject_weight_ptr: Option<u64>,
    inject_scale_ptr: Option<u64>,
    inject_out_ptr: u64,
) -> bool {
    let inject_ok = match (inject_weight_ptr, inject_scale_ptr) {
        (Some(iw), Some(is_)) => {
            hc_count > 0
                && iw != 0
                && iw.is_multiple_of(16)
                && is_ != 0
                && is_.is_multiple_of(4)
                && inject_out_ptr != 0
                && inject_out_ptr.is_multiple_of(2)
                && inject_out_ptr != act_ptr
                && inject_out_ptr != down_out_ptr
        }
        (None, None) => true,
        _ => false,
    };
    enabled
        && kernel_loaded
        && weights_valid
        && rows >= DEFAULT_BATCHED_VERIFY_TC_HC_PREPARE_MIN_ROWS
        && rows <= crate::layers::QWEN4_NATIVE_MAX_ROWS
        && rank > 0
        && k > 0
        && k.is_multiple_of(64)
        && act_ptr != 0
        && act_ptr.is_multiple_of(16)
        && down_weight_ptr != 0
        && down_weight_ptr.is_multiple_of(16)
        && down_scale_ptr != 0
        && down_scale_ptr.is_multiple_of(4)
        && down_out_ptr != 0
        && down_out_ptr.is_multiple_of(2)
        && down_out_ptr != act_ptr
        && inject_ok
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
const fn hc_upmix_coalesced_eligible(
    enabled: bool,
    kernel_ready: bool,
    weights_valid: bool,
    rows: usize,
    hidden_size: usize,
    hc_count: usize,
    rank: usize,
    act_ptr: u64,
    up_weight_ptr: u64,
    up_scale_ptr: u64,
    residual_ptr: u64,
    has_inject: bool,
    inject_logits_ptr: u64,
    mixed_ptr: u64,
) -> bool {
    let inject_ok = !has_inject || (inject_logits_ptr != 0 && inject_logits_ptr.is_multiple_of(2));
    enabled
        && kernel_ready
        && weights_valid
        && rows >= HC_UPMIX_COALESCED_MIN_ROWS
        && rows <= HC_UPMIX_COALESCED_MAX_ROWS
        && hc_count == 4
        && hidden_size >= 16
        && hidden_size.is_multiple_of(16)
        && rank > 0
        && rank.is_multiple_of(64)
        && act_ptr != 0
        && act_ptr.is_multiple_of(16)
        && up_weight_ptr != 0
        && up_weight_ptr.is_multiple_of(16)
        && up_scale_ptr != 0
        && up_scale_ptr.is_multiple_of(4)
        && residual_ptr != 0
        && residual_ptr.is_multiple_of(2)
        && mixed_ptr != 0
        && mixed_ptr.is_multiple_of(2)
        && mixed_ptr != residual_ptr
        && mixed_ptr != act_ptr
        && inject_ok
}

/// Whether a batched HC projection (`[rows, k] @ [n, k]^T -> [rows, n]`)
/// meets the shape, alignment, and non-aliasing preconditions of
/// `qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm`. Requires 0 B extra VRAM
/// and no eager warmup, so eligibility is independent of CUDA graph capture.
#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
const fn coalesced_nvfp4_hc_proj_eligible(
    weights_valid: bool,
    rows: usize,
    n: usize,
    k: usize,
    act_ptr: u64,
    weight_ptr: u64,
    scale_ptr: u64,
    out_ptr: u64,
) -> bool {
    weights_valid
        && rows > 0
        && rows <= crate::layers::QWEN4_NATIVE_MAX_ROWS
        && n > 0
        && k > 0
        && k.is_multiple_of(32)
        && act_ptr != 0
        && act_ptr.is_multiple_of(16)
        && weight_ptr != 0
        && weight_ptr.is_multiple_of(16)
        && scale_ptr != 0
        && scale_ptr.is_multiple_of(2)
        && out_ptr != 0
        && out_ptr.is_multiple_of(16)
        && out_ptr != act_ptr
}

#[inline]
#[must_use]
const fn is_fast_tc_pair(down: BatchedHcDownRoute, up: BatchedHcUpRoute) -> bool {
    matches!(
        up,
        BatchedHcUpRoute::UpmixCoalesced
            | BatchedHcUpRoute::CoalescedNvfp4
            | BatchedHcUpRoute::UpmixRoofW4
            | BatchedHcUpRoute::Fp8Tensor
    ) || matches!(
        down,
        BatchedHcDownRoute::DualCoalescedNvfp4
            | BatchedHcDownRoute::CoalescedNvfp4
            | BatchedHcDownRoute::Fp8Tensor
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
const fn select_batched_hc_down_route_v2_with_policy(
    rows: usize,
    has_inject: bool,
    dual_coalesced_ready: bool,
    dual_silu_m10_ready: bool,
    exact_small_ready: bool,
    coalesced_down_ready: bool,
    fp8_down_ready: bool,
    dual_silu_exact_ready: bool,
    w4a16_gemm_ready: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> Option<BatchedHcDownRoute> {
    if det_batch.is_enabled() {
        return None;
    }
    select_batched_hc_down_route_v2(
        rows,
        has_inject,
        dual_coalesced_ready,
        dual_silu_m10_ready,
        exact_small_ready,
        coalesced_down_ready,
        fp8_down_ready,
        dual_silu_exact_ready,
        w4a16_gemm_ready,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
const fn select_batched_hc_down_route_v2(
    rows: usize,
    has_inject: bool,
    dual_coalesced_ready: bool,
    dual_silu_m10_ready: bool,
    exact_small_ready: bool,
    coalesced_down_ready: bool,
    fp8_down_ready: bool,
    dual_silu_exact_ready: bool,
    w4a16_gemm_ready: bool,
) -> Option<BatchedHcDownRoute> {
    if rows >= DEFAULT_BATCHED_VERIFY_TC_HC_PREPARE_MIN_ROWS && dual_coalesced_ready {
        return Some(BatchedHcDownRoute::DualCoalescedNvfp4);
    }
    select_batched_hc_down_route(
        rows,
        has_inject,
        dual_silu_m10_ready,
        exact_small_ready,
        coalesced_down_ready,
        fp8_down_ready,
        dual_silu_exact_ready,
        w4a16_gemm_ready,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
const fn select_batched_hc_down_route(
    rows: usize,
    has_inject: bool,
    dual_silu_m10_ready: bool,
    exact_small_ready: bool,
    coalesced_down_ready: bool,
    fp8_down_ready: bool,
    dual_silu_exact_ready: bool,
    w4a16_gemm_ready: bool,
) -> Option<BatchedHcDownRoute> {
    if rows < HC_TC_WIDE_DOWN_MIN_ROWS {
        if has_inject && dual_silu_m10_ready {
            return Some(BatchedHcDownRoute::DualSiluM10);
        }
        if exact_small_ready {
            return Some(BatchedHcDownRoute::ExactSmallRows);
        }
        if coalesced_down_ready {
            return Some(BatchedHcDownRoute::CoalescedNvfp4);
        }
        if fp8_down_ready {
            return Some(BatchedHcDownRoute::Fp8Tensor);
        }
    } else if rows <= HC_TC_DUAL_M25_MAX_ROWS {
        // Medium rows (11..=25, e.g. C=4): 1-launch zero-VRAM Warp-Split-K
        // `CoalescedNvfp4` (`n8_sk8_m16`/`n16_sk4_m16`, 15.9-18.4 us vs 50.3 us
        // scalar `DualSiluExact` on GB10) takes priority, falling back cleanly
        // to `DualSiluExact -> Fp8Tensor -> W4a16Gemm`.
        if coalesced_down_ready {
            return Some(BatchedHcDownRoute::CoalescedNvfp4);
        }
        if has_inject && dual_silu_exact_ready {
            return Some(BatchedHcDownRoute::DualSiluExact);
        }
        if fp8_down_ready {
            return Some(BatchedHcDownRoute::Fp8Tensor);
        }
    } else {
        // Wide rows (26..=128, e.g. C=8..16): 1-launch zero-VRAM Warp-Split-K
        // `CoalescedNvfp4` (`n16_sk4_m16`/`n32_sk4_m16`/`n64_sk4_m16`) reads
        // 1.84 MB NVFP4 directly without 3.28 MB FP8 weight caching or warmup,
        // falling back cleanly to `Fp8Tensor -> DualSiluExact -> W4a16Gemm`.
        if coalesced_down_ready {
            return Some(BatchedHcDownRoute::CoalescedNvfp4);
        }
        if fp8_down_ready {
            return Some(BatchedHcDownRoute::Fp8Tensor);
        }
    }
    if has_inject && dual_silu_exact_ready {
        Some(BatchedHcDownRoute::DualSiluExact)
    } else if w4a16_gemm_ready {
        Some(BatchedHcDownRoute::W4a16Gemm)
    } else {
        None
    }
}

#[inline]
#[must_use]
const fn select_batched_hc_up_route_v2_with_policy(
    rows: usize,
    upmix_coalesced_ready: bool,
    coalesced_up_ready: bool,
    upmix_w4_ready: bool,
    fp8_up_ready: bool,
    w4a16_gemm_ready: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> Option<BatchedHcUpRoute> {
    if det_batch.is_enabled() {
        return None;
    }
    select_batched_hc_up_route_v2(
        rows,
        upmix_coalesced_ready,
        coalesced_up_ready,
        upmix_w4_ready,
        fp8_up_ready,
        w4a16_gemm_ready,
    )
}

#[inline]
#[must_use]
const fn select_batched_hc_up_route_v2(
    rows: usize,
    upmix_coalesced_ready: bool,
    coalesced_up_ready: bool,
    upmix_w4_ready: bool,
    fp8_up_ready: bool,
    w4a16_gemm_ready: bool,
) -> Option<BatchedHcUpRoute> {
    if rows >= HC_UPMIX_COALESCED_MIN_ROWS
        && rows <= HC_UPMIX_COALESCED_MAX_ROWS
        && upmix_coalesced_ready
    {
        return Some(BatchedHcUpRoute::UpmixCoalesced);
    }
    select_batched_hc_up_route(
        rows,
        coalesced_up_ready,
        upmix_w4_ready,
        fp8_up_ready,
        w4a16_gemm_ready,
    )
}

#[inline]
#[must_use]
const fn select_batched_hc_up_route(
    _rows: usize,
    coalesced_up_ready: bool,
    upmix_w4_ready: bool,
    fp8_up_ready: bool,
    w4a16_gemm_ready: bool,
) -> Option<BatchedHcUpRoute> {
    // GB10 stage profiling proved `CoalescedNvfp4` + L2-resident `mix_save_inject`
    // beats 128-row-padded `UpmixRoofW4` across the entire `6..=128` batched-verify
    // range by `-6.0 to -6.3 ms/step` at `k=40..80`, so keep `CoalescedNvfp4` first
    // and retain `UpmixRoofW4` strictly as a fallback when coalesced kernels are absent.
    if coalesced_up_ready {
        Some(BatchedHcUpRoute::CoalescedNvfp4)
    } else if upmix_w4_ready {
        Some(BatchedHcUpRoute::UpmixRoofW4)
    } else if fp8_up_ready {
        Some(BatchedHcUpRoute::Fp8Tensor)
    } else if w4a16_gemm_ready {
        Some(BatchedHcUpRoute::W4a16Gemm)
    } else {
        None
    }
}

fn try_w4a16_kernel(gpu: &dyn GpuBackend, name: &str) -> KernelHandle {
    let h = crate::layers::try_kernel(gpu, "w4a16", name);
    if h.0 != 0 {
        h
    } else {
        crate::layers::try_kernel(gpu, "w4a16_gemm", name)
    }
}

#[derive(Clone, Copy, Debug)]
struct CoalescedUpmixKernels {
    m16: KernelHandle,
    m32: KernelHandle,
    m64: KernelHandle,
    m80: KernelHandle,
}

impl CoalescedUpmixKernels {
    const fn empty() -> Self {
        Self {
            m16: KernelHandle(0),
            m32: KernelHandle(0),
            m64: KernelHandle(0),
            m80: KernelHandle(0),
        }
    }

    fn load(gpu: &dyn GpuBackend) -> Self {
        Self {
            m16: try_w4a16_kernel(gpu, "qwen4_hc_upmix_coalesced_m16"),
            m32: try_w4a16_kernel(gpu, "qwen4_hc_upmix_coalesced_m32"),
            m64: try_w4a16_kernel(gpu, "qwen4_hc_upmix_coalesced_m64"),
            m80: try_w4a16_kernel(gpu, "qwen4_hc_upmix_coalesced_m80"),
        }
    }

    #[inline]
    #[must_use]
    const fn resolve(self, rows: usize) -> Option<(u32, KernelHandle)> {
        if rows < HC_UPMIX_COALESCED_MIN_ROWS || rows > HC_UPMIX_COALESCED_MAX_ROWS {
            return None;
        }
        if rows <= 16 {
            if self.m16.0 != 0 {
                Some((16, self.m16))
            } else if self.m32.0 != 0 {
                Some((32, self.m32))
            } else {
                None
            }
        } else if rows <= 32 {
            if self.m32.0 != 0 {
                Some((32, self.m32))
            } else {
                None
            }
        } else if rows <= 64 {
            if self.m64.0 != 0 {
                Some((64, self.m64))
            } else {
                None
            }
        } else if rows <= 80 {
            if self.m80.0 != 0 {
                Some((80, self.m80))
            } else if self.m64.0 != 0 {
                Some((64, self.m64))
            } else {
                None
            }
        } else if self.m64.0 != 0 {
            Some((64, self.m64))
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CoalescedSiluDivKernels {
    n8_sk8_m16: KernelHandle,
    n16_sk4_m16: KernelHandle,
    n32_sk4_m16: KernelHandle,
    n64_sk4_m16: KernelHandle,
}

impl CoalescedSiluDivKernels {
    fn load(gpu: &dyn GpuBackend) -> Self {
        Self {
            n8_sk8_m16: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n8_sk8_m16_silu_div"),
            n16_sk4_m16: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n16_sk4_m16_silu_div"),
            n32_sk4_m16: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n32_sk4_m16_silu_div"),
            n64_sk4_m16: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n64_sk4_m16_silu_div"),
        }
    }

    #[inline]
    #[must_use]
    const fn resolve(
        self,
        tile: crate::layers::qwen4_fast_proj::CoalescedNvfp4Tile,
    ) -> KernelHandle {
        use crate::layers::qwen4_fast_proj::{CoalescedNvfp4MTile, CoalescedNvfp4NTile};
        match (tile.n_tile, tile.m_tile) {
            (CoalescedNvfp4NTile::N8Sk8, CoalescedNvfp4MTile::M16) => self.n8_sk8_m16,
            (CoalescedNvfp4NTile::N16Sk4, CoalescedNvfp4MTile::M16) => self.n16_sk4_m16,
            (CoalescedNvfp4NTile::N32Sk4, CoalescedNvfp4MTile::M16) => self.n32_sk4_m16,
            (CoalescedNvfp4NTile::N64Sk4, CoalescedNvfp4MTile::M16) => self.n64_sk4_m16,
            _ => KernelHandle(0),
        }
    }
}

#[inline]
fn backend_cache_key(gpu: &dyn GpuBackend) -> usize {
    std::ptr::from_ref(gpu) as *const () as usize
}

/// Per-backend boolean probe cache with a lock-free `OnceLock` fast-path for
/// the primary production backend and a backend-pointer-keyed fallback map so
/// distinct mock backends in unit tests never poison each other.
pub(crate) struct BackendBoolCache {
    primary: std::sync::OnceLock<(usize, bool)>,
    fallback: std::sync::OnceLock<parking_lot::Mutex<std::collections::HashMap<usize, bool>>>,
}

impl BackendBoolCache {
    pub(crate) const fn new() -> Self {
        Self {
            primary: std::sync::OnceLock::new(),
            fallback: std::sync::OnceLock::new(),
        }
    }

    pub(crate) fn get_or_probe(
        &self,
        gpu: &dyn GpuBackend,
        probe: impl FnOnce(&dyn GpuBackend) -> bool,
    ) -> bool {
        let key = backend_cache_key(gpu);
        if let Some(&(cached_key, cached_val)) = self.primary.get() {
            if cached_key == key {
                return cached_val;
            }
        } else {
            let val = probe(gpu);
            let &(stored_key, stored_val) = self.primary.get_or_init(|| (key, val));
            if stored_key == key {
                return stored_val;
            }
            let map = self
                .fallback
                .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
            return *map.lock().entry(key).or_insert(val);
        }
        let map = self
            .fallback
            .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
        let cached = map.lock().get(&key).copied();
        if let Some(val) = cached {
            return val;
        }
        let val = probe(gpu);
        *map.lock().entry(key).or_insert(val)
    }
}

/// Cached per-backend availability of row-scale FP8 projection kernels
/// (`qwen4_fp8_weight_rows` and `qwen4_fp8_act_rows`).
pub(crate) fn fp8_row_scale_kernels_available(gpu: &dyn GpuBackend) -> bool {
    static CACHE: BackendBoolCache = BackendBoolCache::new();
    CACHE.get_or_probe(gpu, |gpu| {
        gpu.kernel("qwen4_fp8_proj", "qwen4_fp8_weight_rows")
            .is_ok()
            && gpu.kernel("qwen4_fp8_proj", "qwen4_fp8_act_rows").is_ok()
    })
}

/// Cached per-backend availability of tensor-scale FP8 projection kernels
/// (`qwen4_fp8_weight_amax`, `qwen4_fp8_act_amax`,
/// `qwen4_fp8_weight_rows_tensor`, and `qwen4_fp8_act_rows_tensor`).
fn fp8_tensor_kernels_available(gpu: &dyn GpuBackend) -> bool {
    static CACHE: BackendBoolCache = BackendBoolCache::new();
    CACHE.get_or_probe(gpu, |gpu| {
        [
            "qwen4_fp8_weight_amax",
            "qwen4_fp8_act_amax",
            "qwen4_fp8_weight_rows_tensor",
            "qwen4_fp8_act_rows_tensor",
        ]
        .into_iter()
        .all(|name| gpu.kernel("qwen4_fp8_proj", name).is_ok())
    })
}

#[cfg(test)]
mod batched_verify_tc_dense_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use anyhow::{Result, bail};
    use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

    use super::{
        BackendBoolCache, BatchedHcDownRoute, BatchedHcQkvScratchLayout, BatchedHcTcRoute,
        BatchedHcUpRoute, BatchedVerifyTcComponent, BatchedVerifyTcDenseGuard,
        CoalescedSiluDivKernels, CoalescedUpmixKernels, DEFAULT_BATCHED_VERIFY_TC_DIRECT_MIN_ROWS,
        DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS, DEFAULT_BATCHED_VERIFY_TC_HC_PREPARE_MIN_ROWS,
        DUAL_SPLITK_MAX_FLOATS, DualCoalescedTier, DualHighSmKernels, HC_DUAL_N8_MAX_ROWS,
        HC_FP8_WS_BYTES, HC_TC_WIDE_DOWN_MIN_ROWS, WarmedRowsMask,
        batched_verify_tc_component_admits, batched_verify_tc_dense_admits,
        batched_verify_tc_dense_scope_active, coalesced_nvfp4_hc_proj_eligible,
        fp8_row_scale_kernels_available, fp8_tensor_kernels_available, hc_dual_coalesced_eligible,
        hc_upmix_coalesced_eligible, inject_group_norm_v2_eligible, is_fast_tc_pair,
        parse_batched_verify_tc_dense, parse_batched_verify_tc_dense_min_rows_override,
        parse_hc_dual_coalesced, parse_hc_dual_n32, parse_hc_upmix_coalesced,
        resolve_hc_dual_coalesced, resolve_hc_dual_coalesced_for_sms, select_batched_hc_down_route,
        select_batched_hc_down_route_v2, select_batched_hc_tc_route, select_batched_hc_up_route,
        select_batched_hc_up_route_v2, select_dual_split_k_grid_z,
    };

    struct CountingKernelGpu {
        available: bool,
        lookups: AtomicUsize,
    }

    impl CountingKernelGpu {
        fn new(available: bool) -> Self {
            Self {
                available,
                lookups: AtomicUsize::new(0),
            }
        }

        fn lookups(&self) -> usize {
            self.lookups.load(Ordering::SeqCst)
        }
    }

    impl GpuBackend for CountingKernelGpu {
        fn alloc(&self, _bytes: usize) -> Result<DevicePtr> {
            unimplemented!("not exercised by kernel-lookup cache tests")
        }
        fn alloc_managed(&self, _bytes: usize) -> Result<DevicePtr> {
            unimplemented!("not exercised by kernel-lookup cache tests")
        }
        fn free(&self, _ptr: DevicePtr) -> Result<()> {
            unimplemented!("not exercised by kernel-lookup cache tests")
        }
        fn copy_h2d(&self, _src: &[u8], _dst: DevicePtr) -> Result<()> {
            unimplemented!("not exercised by kernel-lookup cache tests")
        }
        fn copy_d2h(&self, _src: DevicePtr, _dst: &mut [u8]) -> Result<()> {
            unimplemented!("not exercised by kernel-lookup cache tests")
        }
        fn copy_d2d(&self, _src: DevicePtr, _dst: DevicePtr, _bytes: usize) -> Result<()> {
            unimplemented!("not exercised by kernel-lookup cache tests")
        }
        fn launch(
            &self,
            _func: KernelHandle,
            _grid: [u32; 3],
            _block: [u32; 3],
            _shared_mem: u32,
            _stream: u64,
            _params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            unimplemented!("not exercised by kernel-lookup cache tests")
        }
        fn synchronize(&self, _stream: u64) -> Result<()> {
            unimplemented!("not exercised by kernel-lookup cache tests")
        }
        fn default_stream(&self) -> u64 {
            0
        }
        fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
            self.lookups.fetch_add(1, Ordering::SeqCst);
            if self.available {
                Ok(KernelHandle(1))
            } else {
                bail!("missing {module}::{func_name}")
            }
        }
        fn memset(&self, _ptr: DevicePtr, _value: u8, _bytes: usize) -> Result<()> {
            unimplemented!("not exercised by kernel-lookup cache tests")
        }
        fn memset_async(
            &self,
            _ptr: DevicePtr,
            _value: u8,
            _bytes: usize,
            _stream: u64,
        ) -> Result<()> {
            unimplemented!("not exercised by kernel-lookup cache tests")
        }
        fn total_memory(&self) -> Result<usize> {
            unimplemented!("not exercised by kernel-lookup cache tests")
        }
        fn free_memory(&self) -> Result<usize> {
            unimplemented!("not exercised by kernel-lookup cache tests")
        }
    }

    #[test]
    fn tc_dense_selector_defaults_on_and_zero_disables() {
        assert!(parse_batched_verify_tc_dense(None));
        assert!(parse_batched_verify_tc_dense(Some("")));
        assert!(parse_batched_verify_tc_dense(Some("1")));
        assert!(parse_batched_verify_tc_dense(Some("true")));
        assert!(!parse_batched_verify_tc_dense(Some("0")));
        assert!(!parse_batched_verify_tc_dense(Some(" 0 ")));
    }

    #[test]
    fn tc_dense_min_rows_override_returns_none_for_unset_zero_or_invalid() {
        assert_eq!(DEFAULT_BATCHED_VERIFY_TC_DIRECT_MIN_ROWS, 4);
        assert_eq!(DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS, 4);
        assert_eq!(DEFAULT_BATCHED_VERIFY_TC_HC_PREPARE_MIN_ROWS, 4);
        assert_eq!(HC_TC_WIDE_DOWN_MIN_ROWS, 11);
        assert_eq!(parse_batched_verify_tc_dense_min_rows_override(None), None);
        assert_eq!(
            parse_batched_verify_tc_dense_min_rows_override(Some("")),
            None
        );
        assert_eq!(
            parse_batched_verify_tc_dense_min_rows_override(Some("0")),
            None
        );
        assert_eq!(
            parse_batched_verify_tc_dense_min_rows_override(Some("bad")),
            None
        );
        assert_eq!(
            parse_batched_verify_tc_dense_min_rows_override(Some("-4")),
            None
        );
        assert_eq!(
            parse_batched_verify_tc_dense_min_rows_override(Some("10")),
            Some(10)
        );
        assert_eq!(
            parse_batched_verify_tc_dense_min_rows_override(Some(" 32 ")),
            Some(32)
        );
    }

    #[test]
    fn tc_dense_component_thresholds_preserve_c1_exact_speed_up_c2_and_admit_graph_capture() {
        let all_components = [
            BatchedVerifyTcComponent::LmHead,
            BatchedVerifyTcComponent::PleProj,
            BatchedVerifyTcComponent::FinalMixer,
            BatchedVerifyTcComponent::Fp8Dense,
            BatchedVerifyTcComponent::HcPrepare,
        ];
        // Rows <= 3 refuse ALL components so tiny frames stay on exact scalar GEMV;
        // C=1 stays 100% exact via `BatchedVerifyTcDenseGuard::enter(false)`.
        for rows in [0, 1, 2, 3] {
            for component in all_components {
                assert!(
                    !batched_verify_tc_component_admits(component, true, None, rows, false),
                    "component={component:?} rows={rows} must stay on exact scalar GEMV"
                );
            }
        }
        // C=2 r=4..8, C=2 K=5, and C=4..64 (rows=4..=256): ALL components engage in both
        // eager and graph_capture; disabled selector refuses ALL.
        for rows in [
            4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 25, 26, 32, 40, 80, 128, 192, 256,
        ] {
            for component in all_components {
                assert!(
                    batched_verify_tc_component_admits(component, true, None, rows, false),
                    "component={component:?} rows={rows} eager must admit TC dense"
                );
                assert!(
                    batched_verify_tc_component_admits(component, true, None, rows, true),
                    "component={component:?} rows={rows} TC must admit graph capture"
                );
                assert!(
                    !batched_verify_tc_component_admits(component, false, None, rows, false),
                    "component={component:?} rows={rows} disabled must refuse TC dense"
                );
            }
        }
        // Above 256 rows, Fp8Dense falls back to the lazy FP8 cache and stays blocked under graph_capture.
        for rows in [257, 512] {
            assert!(batched_verify_tc_component_admits(
                BatchedVerifyTcComponent::Fp8Dense,
                true,
                None,
                rows,
                false
            ));
            assert!(!batched_verify_tc_component_admits(
                BatchedVerifyTcComponent::Fp8Dense,
                true,
                None,
                rows,
                true
            ));
        }
        // Explicit MIN_ROWS override applies uniformly across every component.
        for component in all_components {
            assert!(!batched_verify_tc_component_admits(
                component,
                true,
                Some(10),
                9,
                false
            ));
            assert!(batched_verify_tc_component_admits(
                component,
                true,
                Some(10),
                10,
                false
            ));
        }
        assert!(!batched_verify_tc_dense_admits(true, 16, 15, false));
        assert!(batched_verify_tc_dense_admits(true, 16, 16, false));
    }

    #[test]
    fn tc_dense_scope_guard_nests_and_restores_cleanly() {
        assert!(!batched_verify_tc_dense_scope_active());
        {
            let _outer = BatchedVerifyTcDenseGuard::enter(true);
            assert!(batched_verify_tc_dense_scope_active());
            {
                let _inner = BatchedVerifyTcDenseGuard::enter(false);
                assert!(!batched_verify_tc_dense_scope_active());
            }
            assert!(batched_verify_tc_dense_scope_active());
        }
        assert!(!batched_verify_tc_dense_scope_active());
    }

    #[test]
    fn hc_route_prefers_fast_tc_then_w4a16_gemm_then_exact() {
        assert_eq!(
            select_batched_hc_tc_route(false, true, true),
            BatchedHcTcRoute::Exact
        );
        assert_eq!(
            select_batched_hc_tc_route(true, true, true),
            BatchedHcTcRoute::FastTc
        );
        assert_eq!(
            select_batched_hc_tc_route(true, true, false),
            BatchedHcTcRoute::FastTc
        );
        assert_eq!(
            select_batched_hc_tc_route(true, false, true),
            BatchedHcTcRoute::W4a16Gemm
        );
        assert_eq!(
            select_batched_hc_tc_route(true, false, false),
            BatchedHcTcRoute::Exact
        );
    }

    #[test]
    fn hc_down_and_up_routes_cover_c2_c4_c16_final_mixer_and_cold_capture() {
        // C=1 K=6..8 & C=2 (rows=6..=10): injecting HcPrepare prefers 1-launch DualSiluM10;
        // FinalMixer (or injecting HC with dual-rows disabled) prefers ExactSmallRows;
        // when both exact small tiers are absent, falls back cleanly to CoalescedNvfp4
        // (zero FP8 cache, zero warmup).
        for rows in [6, 7, 8, 9, 10] {
            assert_eq!(
                select_batched_hc_down_route(rows, true, true, true, true, true, true, true),
                Some(BatchedHcDownRoute::DualSiluM10),
                "rows={rows}: injecting HcPrepare must use 1-launch DualSiluM10"
            );
            assert_eq!(
                select_batched_hc_down_route(rows, true, false, true, true, true, false, true),
                Some(BatchedHcDownRoute::ExactSmallRows),
                "rows={rows}: injecting HcPrepare without DualSiluM10 must use ExactSmallRows"
            );
            assert_eq!(
                select_batched_hc_down_route(rows, false, false, true, true, true, false, true),
                Some(BatchedHcDownRoute::ExactSmallRows),
                "rows={rows}: FinalMixer must use ExactSmallRows at rows<=10"
            );
            assert_eq!(
                select_batched_hc_down_route(rows, false, false, false, true, true, false, true),
                Some(BatchedHcDownRoute::CoalescedNvfp4),
                "rows={rows}: missing ExactSmallRows must fall back to CoalescedNvfp4"
            );
            assert_eq!(
                select_batched_hc_up_route(rows, true, true, true, true),
                Some(BatchedHcUpRoute::CoalescedNvfp4),
                "rows={rows}: up below 32 rows must prioritize CoalescedNvfp4 over 128-row UpmixRoofW4"
            );
        }

        // C=4 medium rows (rows=11..=25): 1-launch zero-VRAM Warp-Split-K CoalescedNvfp4
        // takes priority for BOTH HcPrepare and FinalMixer, degrading cleanly to
        // DualSiluExact -> Fp8Tensor -> W4a16Gemm -> None.
        for rows in [11, 12, 20, 25] {
            assert_eq!(
                select_batched_hc_down_route(rows, true, false, false, true, true, true, true),
                Some(BatchedHcDownRoute::CoalescedNvfp4),
                "rows={rows}: eager HcPrepare must select Warp-Split-K CoalescedNvfp4 for down"
            );
            assert_eq!(
                select_batched_hc_down_route(rows, true, false, false, true, false, true, true),
                Some(BatchedHcDownRoute::CoalescedNvfp4),
                "rows={rows}: cold-capture HcPrepare must select Warp-Split-K CoalescedNvfp4 for down"
            );
            assert_eq!(
                select_batched_hc_down_route(rows, true, false, false, false, true, true, true),
                Some(BatchedHcDownRoute::DualSiluExact),
                "rows={rows}: missing CoalescedNvfp4 must fall back to DualSiluExact"
            );
            assert_eq!(
                select_batched_hc_down_route(rows, false, false, false, true, true, false, true),
                Some(BatchedHcDownRoute::CoalescedNvfp4),
                "rows={rows}: eager FinalMixer must select Warp-Split-K CoalescedNvfp4 for down"
            );
            assert_eq!(
                select_batched_hc_down_route(rows, false, false, false, true, false, false, true),
                Some(BatchedHcDownRoute::CoalescedNvfp4),
                "rows={rows}: cold-capture FinalMixer must select CoalescedNvfp4"
            );
            assert_eq!(
                select_batched_hc_down_route(rows, true, false, false, false, true, false, true),
                Some(BatchedHcDownRoute::Fp8Tensor)
            );
            assert_eq!(
                select_batched_hc_down_route(rows, false, false, false, false, false, false, true),
                Some(BatchedHcDownRoute::W4a16Gemm)
            );
            assert_eq!(
                select_batched_hc_down_route(rows, false, false, false, false, false, false, false),
                None
            );
        }

        // C=8..25 wide rows (rows=26..=128): 1-launch zero-VRAM Warp-Split-K CoalescedNvfp4
        // takes top priority for BOTH HcPrepare and FinalMixer (eager and cold capture alike),
        // degrading cleanly to Fp8Tensor -> DualSiluExact -> W4a16Gemm -> None.
        for rows in [26, 32, 40, 80, 128] {
            assert_eq!(
                select_batched_hc_down_route(rows, true, false, false, true, true, true, true),
                Some(BatchedHcDownRoute::CoalescedNvfp4),
                "rows={rows}: eager HcPrepare must select Warp-Split-K CoalescedNvfp4 for down"
            );
            assert_eq!(
                select_batched_hc_down_route(rows, true, false, false, true, false, true, true),
                Some(BatchedHcDownRoute::CoalescedNvfp4),
                "rows={rows}: cold-capture HcPrepare must select CoalescedNvfp4 for down"
            );
            assert_eq!(
                select_batched_hc_down_route(rows, false, false, false, true, true, false, true),
                Some(BatchedHcDownRoute::CoalescedNvfp4),
                "rows={rows}: eager FinalMixer must select Warp-Split-K CoalescedNvfp4 for down"
            );
            assert_eq!(
                select_batched_hc_down_route(rows, false, false, false, true, false, false, true),
                Some(BatchedHcDownRoute::CoalescedNvfp4),
                "rows={rows}: cold-capture FinalMixer must select CoalescedNvfp4 for down"
            );
            assert_eq!(
                select_batched_hc_down_route(rows, true, false, false, false, true, true, true),
                Some(BatchedHcDownRoute::Fp8Tensor),
                "rows={rows}: missing CoalescedNvfp4 must fall back to Fp8Tensor"
            );
            assert_eq!(
                select_batched_hc_down_route(rows, true, false, false, false, false, true, true),
                Some(BatchedHcDownRoute::DualSiluExact)
            );
            assert_eq!(
                select_batched_hc_down_route(rows, false, false, false, false, false, false, true),
                Some(BatchedHcDownRoute::W4a16Gemm)
            );
            assert_eq!(
                select_batched_hc_down_route(rows, false, false, false, false, false, false, false),
                None
            );
        }

        // Up+mix priority chain below 32 rows: CoalescedNvfp4 > UpmixRoofW4 > Fp8Tensor > W4a16Gemm > None.
        for rows in [6usize, 10, 20, 31] {
            assert_eq!(
                select_batched_hc_up_route(rows, true, true, true, true),
                Some(BatchedHcUpRoute::CoalescedNvfp4),
                "rows={rows}"
            );
            assert_eq!(
                select_batched_hc_up_route(rows, true, true, false, true),
                Some(BatchedHcUpRoute::CoalescedNvfp4),
                "rows={rows}: cold capture must keep CoalescedNvfp4"
            );
            assert_eq!(
                select_batched_hc_up_route(rows, false, true, true, true),
                Some(BatchedHcUpRoute::UpmixRoofW4),
                "rows={rows}"
            );
            assert_eq!(
                select_batched_hc_up_route(rows, false, false, true, true),
                Some(BatchedHcUpRoute::Fp8Tensor),
                "rows={rows}"
            );
            assert_eq!(
                select_batched_hc_up_route(rows, false, false, false, true),
                Some(BatchedHcUpRoute::W4a16Gemm),
                "rows={rows}"
            );
            assert_eq!(
                select_batched_hc_up_route(rows, false, false, false, false),
                None,
                "rows={rows}"
            );
        }

        // Up+mix priority chain at rows >= 32: CoalescedNvfp4 > UpmixRoofW4 > Fp8Tensor > W4a16Gemm > None.
        for rows in [32usize, 40, 64, 80, 128] {
            assert_eq!(
                select_batched_hc_up_route(rows, true, true, true, true),
                Some(BatchedHcUpRoute::CoalescedNvfp4),
                "rows={rows}: must prefer CoalescedNvfp4 over 128-row-padded UpmixRoofW4"
            );
            assert_eq!(
                select_batched_hc_up_route(rows, true, true, false, true),
                Some(BatchedHcUpRoute::CoalescedNvfp4),
                "rows={rows}: cold capture must keep zero-alloc CoalescedNvfp4"
            );
            assert_eq!(
                select_batched_hc_up_route(rows, false, true, true, true),
                Some(BatchedHcUpRoute::UpmixRoofW4),
                "rows={rows}: missing CoalescedNvfp4 falls back to UpmixRoofW4"
            );
            assert_eq!(
                select_batched_hc_up_route(rows, false, false, true, true),
                Some(BatchedHcUpRoute::Fp8Tensor),
                "rows={rows}"
            );
            assert_eq!(
                select_batched_hc_up_route(rows, false, false, false, true),
                Some(BatchedHcUpRoute::W4a16Gemm),
                "rows={rows}"
            );
            assert_eq!(
                select_batched_hc_up_route(rows, false, false, false, false),
                None,
                "rows={rows}"
            );
        }

        // FastTc admission covers every CoalescedNvfp4 / UpmixRoofW4 combination (including
        // C=2 DualSiluM10 / ExactSmallRows paired with CoalescedNvfp4 up and wide
        // CoalescedNvfp4 down paired with UpmixRoofW4 up) while leaving pure legacy pairs on
        // W4a16Gemm.
        assert!(is_fast_tc_pair(
            BatchedHcDownRoute::CoalescedNvfp4,
            BatchedHcUpRoute::UpmixRoofW4
        ));
        assert!(is_fast_tc_pair(
            BatchedHcDownRoute::DualSiluM10,
            BatchedHcUpRoute::CoalescedNvfp4
        ));
        assert!(is_fast_tc_pair(
            BatchedHcDownRoute::ExactSmallRows,
            BatchedHcUpRoute::CoalescedNvfp4
        ));
        assert!(is_fast_tc_pair(
            BatchedHcDownRoute::CoalescedNvfp4,
            BatchedHcUpRoute::CoalescedNvfp4
        ));
        assert!(is_fast_tc_pair(
            BatchedHcDownRoute::CoalescedNvfp4,
            BatchedHcUpRoute::W4a16Gemm
        ));
        assert!(!is_fast_tc_pair(
            BatchedHcDownRoute::DualSiluExact,
            BatchedHcUpRoute::W4a16Gemm
        ));
        assert!(!is_fast_tc_pair(
            BatchedHcDownRoute::W4a16Gemm,
            BatchedHcUpRoute::W4a16Gemm
        ));

        assert_eq!(
            BatchedHcDownRoute::CoalescedNvfp4.label(),
            "coalesced_nvfp4"
        );
        assert_eq!(BatchedHcUpRoute::CoalescedNvfp4.label(), "coalesced_nvfp4");
        for route in [
            BatchedHcDownRoute::DualSiluM10,
            BatchedHcDownRoute::ExactSmallRows,
            BatchedHcDownRoute::CoalescedNvfp4,
            BatchedHcDownRoute::Fp8Tensor,
            BatchedHcDownRoute::DualSiluExact,
            BatchedHcDownRoute::W4a16Gemm,
        ] {
            assert!(!route.label().is_empty());
        }
        for route in [
            BatchedHcUpRoute::CoalescedNvfp4,
            BatchedHcUpRoute::UpmixRoofW4,
            BatchedHcUpRoute::Fp8Tensor,
            BatchedHcUpRoute::W4a16Gemm,
        ] {
            assert!(!route.label().is_empty());
        }
    }

    #[test]
    fn coalesced_nvfp4_hc_proj_eligibility_validates_shapes_alignment_and_zero_warmup() {
        let act = 0x1000u64;
        let weight = 0x2000u64;
        let scale = 0x3000u64;
        let out = 0x4000u64;
        // Canonical hc.down (N=320, K=10240) and hc.up (N=10240, K=320) across 1..=256 rows.
        for rows in [1, 6, 7, 8, 10, 11, 12, 20, 40, 80, 128, 192, 256] {
            assert!(
                coalesced_nvfp4_hc_proj_eligible(true, rows, 320, 10_240, act, weight, scale, out),
                "hc.down must be eligible at rows={rows}"
            );
            assert!(
                coalesced_nvfp4_hc_proj_eligible(true, rows, 10_240, 320, act, weight, scale, out),
                "hc.up must be eligible at rows={rows}"
            );
        }
        // Reject invalid weights, out-of-range rows, K not multiple of 32, misaligned/null/aliased ptrs.
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            false, 20, 320, 10_240, act, weight, scale, out
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true, 0, 320, 10_240, act, weight, scale, out
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true, 257, 320, 10_240, act, weight, scale, out
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true, 20, 0, 10_240, act, weight, scale, out
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true, 20, 320, 0, act, weight, scale, out
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true, 20, 320, 16, act, weight, scale, out
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true, 20, 10_240, 48, act, weight, scale, out
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true, 20, 320, 10_240, 0, weight, scale, out
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true,
            20,
            320,
            10_240,
            act + 8,
            weight,
            scale,
            out
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true, 20, 320, 10_240, act, 0, scale, out
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true,
            20,
            320,
            10_240,
            act,
            weight + 8,
            scale,
            out
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true, 20, 320, 10_240, act, weight, 0, out
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true,
            20,
            320,
            10_240,
            act,
            weight,
            scale + 1,
            out
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true, 20, 320, 10_240, act, weight, scale, 0
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true,
            20,
            320,
            10_240,
            act,
            weight,
            scale,
            out + 8
        ));
        assert!(!coalesced_nvfp4_hc_proj_eligible(
            true, 20, 320, 10_240, act, weight, scale, act
        ));
    }

    #[test]
    fn qkv_scratch_layout_is_256_aligned_disjoint_and_fits_qkv_arena() {
        let residual_width = 10_240;
        let rank = 320;
        let hc_count = 4;
        for rows in [1, 8, 10, 12, 20, 40, 80, 128] {
            let l = BatchedHcQkvScratchLayout::compute(rows, residual_width, rank, hc_count)
                .expect("canonical geometry must produce a valid layout");
            assert_eq!(l.down_a8_off, 0);
            assert!(l.down_ws_off.is_multiple_of(256));
            assert!(l.rstd_off.is_multiple_of(256));
            assert!(l.up_a8_off.is_multiple_of(256));
            assert!(l.up_ws_off.is_multiple_of(256));
            // `down_a8`, `down_ws`, and `rstd` are strictly disjoint and fit within
            // the `rows * residual_width * 2` guaranteed qkv_output arena floor.
            assert!(l.down_ws_off >= rows * residual_width);
            assert!(l.rstd_off >= l.down_ws_off + HC_FP8_WS_BYTES);
            assert_eq!(l.total_down_rstd_bytes, l.rstd_off + rows * hc_count * 4);
            assert!(
                l.total_down_rstd_bytes <= rows * residual_width * 2,
                "rows={rows}: down+rstd scratch {} must fit within qkv floor {}",
                l.total_down_rstd_bytes,
                rows * residual_width * 2
            );
            // Non-canonical FP8 `up` scratch sits strictly after BF16 `mix_logits`.
            assert!(l.up_a8_off >= rows * residual_width * 2);
            assert!(l.up_ws_off >= l.up_a8_off + rows * rank);
            assert_eq!(l.total_up_fp8_bytes, l.up_ws_off + HC_FP8_WS_BYTES);
        }
        assert_eq!(BatchedHcQkvScratchLayout::compute(0, 10_240, 320, 4), None);
        assert_eq!(BatchedHcQkvScratchLayout::compute(10, 0, 320, 4), None);
        assert_eq!(BatchedHcQkvScratchLayout::compute(10, 10_240, 0, 4), None);
        assert_eq!(BatchedHcQkvScratchLayout::compute(10, 10_240, 320, 0), None);
        assert_eq!(
            BatchedHcQkvScratchLayout::compute(usize::MAX, 10_240, 320, 4),
            None
        );
    }

    #[test]
    fn warmed_rows_mask_tracks_1_to_256_and_ignores_out_of_range() {
        let mask = WarmedRowsMask::new();
        for rows in [0, 1, 10, 64, 65, 128, 129, 192, 193, 256, 257, usize::MAX] {
            assert!(!mask.is_warmed(rows));
        }
        let warmed = [1, 12, 20, 64, 65, 80, 128, 129, 192, 193, 256];
        for rows in warmed {
            mask.mark_warmed(rows);
        }
        // Out-of-range marks are no-ops and never poison valid bits.
        for rows in [0, 257, 512, usize::MAX] {
            mask.mark_warmed(rows);
            assert!(!mask.is_warmed(rows));
        }
        for rows in warmed {
            assert!(mask.is_warmed(rows), "rows={rows} must report warmed");
        }
        for rows in [2, 11, 13, 19, 21, 63, 66, 79, 81, 127, 130, 191, 194, 255] {
            assert!(!mask.is_warmed(rows), "rows={rows} must stay cold");
        }
    }

    #[test]
    fn backend_kernel_probe_caches_isolate_distinct_backends_and_avoid_repeat_lookups() {
        let gpu_ok = Box::new(CountingKernelGpu::new(true));
        let gpu_missing = Box::new(CountingKernelGpu::new(false));

        // Local BackendBoolCache exercises both primary fast-path and fallback isolation.
        let cache = BackendBoolCache::new();
        assert!(cache.get_or_probe(gpu_ok.as_ref(), |gpu| { gpu.kernel("m", "k").is_ok() }));
        assert_eq!(gpu_ok.lookups(), 1);
        assert!(cache.get_or_probe(gpu_ok.as_ref(), |gpu| { gpu.kernel("m", "k").is_ok() }));
        assert_eq!(
            gpu_ok.lookups(),
            1,
            "primary backend hit must skip driver lookup"
        );

        assert!(!cache.get_or_probe(gpu_missing.as_ref(), |gpu| { gpu.kernel("m", "k").is_ok() }));
        assert_eq!(gpu_missing.lookups(), 1);
        assert!(!cache.get_or_probe(gpu_missing.as_ref(), |gpu| { gpu.kernel("m", "k").is_ok() }));
        assert_eq!(
            gpu_missing.lookups(),
            1,
            "fallback backend hit must skip driver lookup"
        );

        // Process-static caches must also isolate distinct live backends.
        let fp8_ok_gpu = Box::new(CountingKernelGpu::new(true));
        let fp8_missing_gpu = Box::new(CountingKernelGpu::new(false));
        assert!(fp8_row_scale_kernels_available(fp8_ok_gpu.as_ref()));
        assert_eq!(fp8_ok_gpu.lookups(), 2);
        assert!(fp8_row_scale_kernels_available(fp8_ok_gpu.as_ref()));
        assert_eq!(fp8_ok_gpu.lookups(), 2);
        assert!(!fp8_row_scale_kernels_available(fp8_missing_gpu.as_ref()));
        assert_eq!(fp8_missing_gpu.lookups(), 1);
        assert!(!fp8_row_scale_kernels_available(fp8_missing_gpu.as_ref()));
        assert_eq!(fp8_missing_gpu.lookups(), 1);

        let fp8_tensor_ok_gpu = Box::new(CountingKernelGpu::new(true));
        let fp8_tensor_missing_gpu = Box::new(CountingKernelGpu::new(false));
        assert!(fp8_tensor_kernels_available(fp8_tensor_ok_gpu.as_ref()));
        assert_eq!(fp8_tensor_ok_gpu.lookups(), 4);
        assert!(fp8_tensor_kernels_available(fp8_tensor_ok_gpu.as_ref()));
        assert_eq!(fp8_tensor_ok_gpu.lookups(), 4);
        assert!(!fp8_tensor_kernels_available(
            fp8_tensor_missing_gpu.as_ref()
        ));
        assert_eq!(fp8_tensor_missing_gpu.lookups(), 1);
        assert!(!fp8_tensor_kernels_available(
            fp8_tensor_missing_gpu.as_ref()
        ));
        assert_eq!(fp8_tensor_missing_gpu.lookups(), 1);
    }

    #[test]
    fn coalesced_silu_div_kernels_cover_all_canonical_hc_down_rows() {
        use crate::layers::qwen4_fast_proj::select_coalesced_nvfp4_tile_for_shape;

        let kernels = CoalescedSiluDivKernels {
            n8_sk8_m16: KernelHandle(11),
            n16_sk4_m16: KernelHandle(12),
            n32_sk4_m16: KernelHandle(13),
            n64_sk4_m16: KernelHandle(14),
        };
        for rows in 1usize..=128 {
            let tile = select_coalesced_nvfp4_tile_for_shape(rows, 320, 10_240)
                .expect("canonical hc.down must resolve a coalesced tile");
            let expected = if rows <= 20 {
                KernelHandle(11)
            } else if rows < 32 {
                KernelHandle(12)
            } else if rows <= 64 {
                KernelHandle(13)
            } else {
                KernelHandle(14)
            };
            assert_eq!(
                kernels.resolve(tile).0,
                expected.0,
                "rows={rows}: hc.down tile {tile:?} must resolve its fused silu_div kernel"
            );
        }
        // Non-split-K hc.up tiles must return handle 0 so only hc.down uses silu_div.
        let up_tile = select_coalesced_nvfp4_tile_for_shape(80, 10_240, 320)
            .expect("canonical hc.up must resolve a coalesced tile");
        assert_eq!(kernels.resolve(up_tile).0, 0);
    }

    #[test]
    fn inject_group_norm_v2_and_mhc_fused_coalesced_routes_validate_and_fallback() {
        // Vectorized uint4 4-CTA/tok inject_group_norm_v2 requires H=2560, HC=4, 16B aligned, non-aliasing.
        assert!(inject_group_norm_v2_eligible(
            2_560, 4, 0x1000, 0x2000, 0x3000, 0x4000
        ));
        assert!(!inject_group_norm_v2_eligible(
            2_048, 4, 0x1000, 0x2000, 0x3000, 0x4000
        ));
        assert!(!inject_group_norm_v2_eligible(
            2_560, 2, 0x1000, 0x2000, 0x3000, 0x4000
        ));
        assert!(!inject_group_norm_v2_eligible(
            2_560, 4, 0x1008, 0x2000, 0x3000, 0x4000
        ));
        assert!(!inject_group_norm_v2_eligible(
            2_560, 4, 0x1000, 0x2000, 0x3000, 0x1000
        ));
        assert!(!inject_group_norm_v2_eligible(
            2_560, 4, 0x1000, 0x2000, 0x3000, 0x2000
        ));
        assert!(!inject_group_norm_v2_eligible(
            2_560, 4, 0, 0x2000, 0x3000, 0x4000
        ));

        // Env selectors default ON and `"0"` disables cleanly.
        assert!(parse_hc_dual_coalesced(None));
        assert!(parse_hc_dual_coalesced(Some("1")));
        assert!(!parse_hc_dual_coalesced(Some("0")));
        assert!(!parse_hc_dual_coalesced(Some(" 0 ")));
        assert!(parse_hc_dual_n32(None));
        assert!(parse_hc_dual_n32(Some("")));
        assert!(parse_hc_dual_n32(Some("1")));
        assert!(!parse_hc_dual_n32(Some("0")));
        assert!(!parse_hc_dual_n32(Some(" 0 ")));
        assert!(parse_hc_upmix_coalesced(None));
        assert!(parse_hc_upmix_coalesced(Some("1")));
        assert!(!parse_hc_upmix_coalesced(Some("0")));
        assert!(!parse_hc_upmix_coalesced(Some(" 0 ")));

        // Dual coalesced tier split: n8_sk8_m32 strictly capped at rows=4..=24;
        // n16_sk4_m16_dual_silu_div_v2 (21 N-blocks, 2-stage K-prefetch, blk=256) serves
        // rows=25..=128 when enabled & loaded (degrading to v1 n32_sk4_m16 if v2 is absent),
        // and NEVER falls back to n8_sk8_m32 for rows > 24.
        assert_eq!(HC_DUAL_N8_MAX_ROWS, 24);
        let dual_n8_k = KernelHandle(31);
        let dual_n16_v2_k = KernelHandle(33);
        let dual_n32_k = KernelHandle(32);
        let res_dual =
            |rows: usize, n8: KernelHandle, n16_v2: KernelHandle, n32: KernelHandle, on: bool| {
                resolve_hc_dual_coalesced(rows, n8, n16_v2, n32, on).map(|(t, k)| (t, k.0))
            };
        for rows in [0usize, 1, 2, 3, 257, 512] {
            assert_eq!(
                res_dual(rows, dual_n8_k, dual_n16_v2_k, dual_n32_k, true),
                None
            );
        }
        for rows in [4usize, 5, 6, 7, 8, 9, 10, 16, 20, 24] {
            assert_eq!(
                res_dual(rows, dual_n8_k, dual_n16_v2_k, dual_n32_k, true),
                Some((DualCoalescedTier::N8Sk8M32, 31)),
                "rows={rows}: must select n8_sk8_m32_dual_silu_div"
            );
            assert_eq!(
                res_dual(rows, dual_n8_k, KernelHandle(0), KernelHandle(0), false),
                Some((DualCoalescedTier::N8Sk8M32, 31)),
                "rows={rows}: ATLAS_QWEN4_HC_DUAL_N32=0 must keep n8_sk8_m32 for rows<=24"
            );
            assert_eq!(
                res_dual(rows, KernelHandle(0), dual_n16_v2_k, dual_n32_k, true),
                None,
                "rows={rows}: missing n8_sk8_m32 must not launch wide dual at rows<=24"
            );
        }
        assert_eq!(DualCoalescedTier::N8Sk8M32.co_n(), 8);
        assert_eq!(DualCoalescedTier::N8Sk8M32.tile_m(), 32);
        assert_eq!(DualCoalescedTier::N8Sk8M32.threads(), 256);
        assert_eq!(DualCoalescedTier::N16Sk4M16V2.co_n(), 16);
        assert_eq!(DualCoalescedTier::N16Sk4M16V2.tile_m(), 16);
        assert_eq!(DualCoalescedTier::N16Sk4M16V2.threads(), 256);
        assert_eq!(DualCoalescedTier::N32Sk4M16.co_n(), 32);
        assert_eq!(DualCoalescedTier::N32Sk4M16.tile_m(), 16);
        assert_eq!(DualCoalescedTier::N32Sk4M16.threads(), 512);
        for rows in [25usize, 32, 40, 48, 64, 80, 128, 192, 256] {
            assert_eq!(
                res_dual(rows, dual_n8_k, dual_n16_v2_k, dual_n32_k, true),
                Some((DualCoalescedTier::N16Sk4M16V2, 33)),
                "rows={rows}: must select 21-block v2 n16_sk4_m16_dual_silu_div_v2"
            );
            assert_eq!(
                res_dual(rows, dual_n8_k, KernelHandle(0), dual_n32_k, true),
                Some((DualCoalescedTier::N32Sk4M16, 32)),
                "rows={rows}: missing n16_sk4_m16_dual_v2 must fall back to n32_sk4_m16_dual"
            );
            assert_eq!(
                res_dual(rows, dual_n8_k, dual_n16_v2_k, dual_n32_k, false),
                None,
                "rows={rows}: ATLAS_QWEN4_HC_DUAL_N32=0 must refuse dual (never launch n8_sk8_m32)"
            );
            assert_eq!(
                res_dual(rows, dual_n8_k, KernelHandle(0), KernelHandle(0), true),
                None,
                "rows={rows}: missing both wide dual kernels must refuse dual (never launch n8_sk8_m32)"
            );
            // When wide dual is disabled or absent at rows=25..=128, route falls back cleanly
            // to v1 CoalescedNvfp4 (`n32_sk4_m16_silu_div + gemv_rows`).
            let dual_ready_off = hc_dual_coalesced_eligible(
                true,
                res_dual(rows, dual_n8_k, dual_n16_v2_k, dual_n32_k, false).is_some(),
                true,
                rows,
                320,
                4,
                10_240,
                0x1000,
                0x2000,
                0x3000,
                0x4000,
                Some(0x5000),
                Some(0x6000),
                0x7000,
            );
            assert!(!dual_ready_off);
            assert_eq!(
                select_batched_hc_down_route_v2(
                    rows,
                    true,
                    dual_ready_off,
                    false,
                    false,
                    true,
                    true,
                    true,
                    true
                ),
                Some(BatchedHcDownRoute::CoalescedNvfp4),
                "rows={rows}: opted-out dual_n32 must fall back to v1 CoalescedNvfp4"
            );
        }

        // Single-sweep dual hc.down + silu_div + hc.inject covers M=4..=256 (both injecting & FinalMixer).
        for rows in [
            4usize, 5, 6, 7, 8, 9, 10, 16, 20, 24, 25, 40, 48, 64, 80, 128, 192, 256,
        ] {
            assert!(hc_dual_coalesced_eligible(
                true,
                true,
                true,
                rows,
                320,
                4,
                10_240,
                0x1000,
                0x2000,
                0x3000,
                0x4000,
                Some(0x5000),
                Some(0x6000),
                0x7000,
            ));
            assert!(hc_dual_coalesced_eligible(
                true, true, true, rows, 320, 4, 10_240, 0x1000, 0x2000, 0x3000, 0x4000, None, None,
                0x7000,
            ));
            assert_eq!(
                select_batched_hc_down_route_v2(
                    rows, true, true, true, true, true, true, true, true
                ),
                Some(BatchedHcDownRoute::DualCoalescedNvfp4),
                "rows={rows}"
            );
            assert_eq!(
                select_batched_hc_down_route_v2(
                    rows, false, true, false, true, true, true, false, true
                ),
                Some(BatchedHcDownRoute::DualCoalescedNvfp4),
                "rows={rows} FinalMixer"
            );
        }
        // Rows <= 3 refuse DualCoalescedNvfp4 and stay on exact small-row paths.
        for rows in [1usize, 2, 3] {
            assert!(!hc_dual_coalesced_eligible(
                true,
                true,
                true,
                rows,
                320,
                4,
                10_240,
                0x1000,
                0x2000,
                0x3000,
                0x4000,
                Some(0x5000),
                Some(0x6000),
                0x7000,
            ));
            assert_eq!(
                select_batched_hc_down_route_v2(
                    rows, false, true, false, true, true, true, false, true
                ),
                Some(BatchedHcDownRoute::ExactSmallRows),
                "rows={rows}"
            );
        }
        // Reject K % 64 != 0 or 2B-only scale alignment for uint32 scale loads.
        assert!(!hc_dual_coalesced_eligible(
            true,
            true,
            true,
            40,
            320,
            4,
            96,
            0x1000,
            0x2000,
            0x3000,
            0x4000,
            Some(0x5000),
            Some(0x6000),
            0x7000,
        ));
        assert!(!hc_dual_coalesced_eligible(
            true,
            true,
            true,
            40,
            320,
            4,
            10_240,
            0x1000,
            0x2000,
            0x3002,
            0x4000,
            Some(0x5000),
            Some(0x6000),
            0x7000,
        ));

        // Fused coalesced hc.up + mix_save_inject covers M=4..=128 (including C=2 r=4..=8
        // and grid.y=2 m64 for M=81..=128) and degrades cleanly outside 4..=128.
        let upmix_ks = CoalescedUpmixKernels {
            m16: KernelHandle(21),
            m32: KernelHandle(22),
            m64: KernelHandle(23),
            m80: KernelHandle(24),
        };
        let res = |ks: &CoalescedUpmixKernels, r: usize| ks.resolve(r).map(|(m, k)| (m, k.0));
        assert_eq!(res(&upmix_ks, 3), None);
        assert_eq!(res(&upmix_ks, 4), Some((16, 21)));
        assert_eq!(res(&upmix_ks, 5), Some((16, 21)));
        assert_eq!(res(&upmix_ks, 6), Some((16, 21)));
        assert_eq!(res(&upmix_ks, 7), Some((16, 21)));
        assert_eq!(res(&upmix_ks, 8), Some((16, 21)));
        assert_eq!(res(&upmix_ks, 9), Some((16, 21)));
        assert_eq!(res(&upmix_ks, 16), Some((16, 21)));
        assert_eq!(res(&upmix_ks, 17), Some((32, 22)));
        assert_eq!(res(&upmix_ks, 32), Some((32, 22)));
        assert_eq!(res(&upmix_ks, 33), Some((64, 23)));
        assert_eq!(res(&upmix_ks, 64), Some((64, 23)));
        assert_eq!(res(&upmix_ks, 65), Some((80, 24)));
        assert_eq!(res(&upmix_ks, 80), Some((80, 24)));
        assert_eq!(res(&upmix_ks, 81), Some((64, 23)));
        assert_eq!(res(&upmix_ks, 96), Some((64, 23)));
        assert_eq!(res(&upmix_ks, 128), Some((64, 23)));
        assert_eq!(res(&upmix_ks, 129), Some((64, 23)));
        assert_eq!(res(&upmix_ks, 192), Some((64, 23)));
        assert_eq!(res(&upmix_ks, 256), Some((64, 23)));
        assert_eq!(res(&upmix_ks, 257), None);

        // Missing m64 fails closed for M=81..=256; missing m80 falls back to m64 for M=65..=80.
        let upmix_no_m64 = CoalescedUpmixKernels {
            m64: KernelHandle(0),
            ..upmix_ks
        };
        assert_eq!(res(&upmix_no_m64, 80), Some((80, 24)));
        assert_eq!(res(&upmix_no_m64, 81), None);
        assert_eq!(res(&upmix_no_m64, 128), None);
        assert_eq!(res(&upmix_no_m64, 256), None);
        let upmix_no_m80 = CoalescedUpmixKernels {
            m80: KernelHandle(0),
            ..upmix_ks
        };
        assert_eq!(res(&upmix_no_m80, 72), Some((64, 23)));
        assert_eq!(res(&upmix_no_m80, 80), Some((64, 23)));

        // Missing m16 falls back to m32 for M=4..=16.
        let upmix_no_m16 = CoalescedUpmixKernels {
            m16: KernelHandle(0),
            ..upmix_ks
        };
        assert_eq!(res(&upmix_no_m16, 4), Some((32, 22)));
        assert_eq!(res(&upmix_no_m16, 6), Some((32, 22)));
        assert_eq!(res(&upmix_no_m16, 10), Some((32, 22)));

        for rows in [
            4usize, 5, 6, 7, 8, 9, 10, 20, 40, 64, 80, 81, 96, 128, 192, 256,
        ] {
            assert!(hc_upmix_coalesced_eligible(
                true, true, true, rows, 2_560, 4, 320, 0x1000, 0x2000, 0x3000, 0x4000, true,
                0x5000, 0x6000,
            ));
            assert_eq!(
                select_batched_hc_up_route_v2(rows, true, true, true, true, true),
                Some(BatchedHcUpRoute::UpmixCoalesced),
                "rows={rows}"
            );
            assert_eq!(
                select_batched_hc_up_route_v2(rows, false, true, true, true, true),
                Some(BatchedHcUpRoute::CoalescedNvfp4),
                "rows={rows}: disabled upmix_coalesced must fall back to CoalescedNvfp4"
            );
        }
        for rows in [3usize, 257, 512] {
            assert!(!hc_upmix_coalesced_eligible(
                true, true, true, rows, 2_560, 4, 320, 0x1000, 0x2000, 0x3000, 0x4000, true,
                0x5000, 0x6000,
            ));
            assert_eq!(
                select_batched_hc_up_route_v2(rows, true, true, true, true, true),
                Some(BatchedHcUpRoute::CoalescedNvfp4),
                "rows={rows}: out-of-range upmix_coalesced must fall back to CoalescedNvfp4"
            );
        }
        assert!(is_fast_tc_pair(
            BatchedHcDownRoute::DualCoalescedNvfp4,
            BatchedHcUpRoute::UpmixCoalesced
        ));
        assert_eq!(
            BatchedHcDownRoute::DualCoalescedNvfp4.label(),
            "dual_coalesced_nvfp4"
        );
        assert_eq!(BatchedHcUpRoute::UpmixCoalesced.label(), "upmix_coalesced");
    }

    #[test]
    fn high_sm_dual_coalesced_tier_and_split_k_preserve_gb10_and_saturate_sm120() {
        let k_n8 = KernelHandle(1);
        let k_n16_v2 = KernelHandle(2);
        let k_n32 = KernelHandle(3);
        let high_sm = DualHighSmKernels {
            n8_sk8_m32_v2: KernelHandle(10),
            n16_sk4_m64_v2: KernelHandle(11),
            n32_sk2_m64_v2: KernelHandle(12),
            n64_sk1_m64_v2: KernelHandle(13),
        };

        // GB10 (<= 64 SMs): dispatch is 100% unchanged even when high-SM kernels exist.
        for sm in [48u32, 64] {
            for rows in [4usize, 16, 24] {
                assert_eq!(
                    resolve_hc_dual_coalesced_for_sms(
                        rows, k_n8, k_n16_v2, k_n32, high_sm, true, sm
                    )
                    .map(|(t, h)| (t, h.0)),
                    Some((DualCoalescedTier::N8Sk8M32, k_n8.0)),
                    "sm={sm} rows={rows}"
                );
            }
            for rows in [25usize, 32, 64, 96, 128, 192, 256] {
                assert_eq!(
                    resolve_hc_dual_coalesced_for_sms(
                        rows, k_n8, k_n16_v2, k_n32, high_sm, true, sm
                    )
                    .map(|(t, h)| (t, h.0)),
                    Some((DualCoalescedTier::N16Sk4M16V2, k_n16_v2.0)),
                    "sm={sm} rows={rows}"
                );
            }
            assert_eq!(
                select_dual_split_k_grid_z(DualCoalescedTier::N64Sk1M64V2, 6, 4, 10_240, sm, true),
                1
            );
        }

        // High-SM SM120 (188 SMs): selects wider TILE_M=32/64 + CO_N=32/64 tiers
        // and grid.z Split-K saturating all 188 SMs within 2 CTAs/SM and scratch budget.
        let sm120 = 188u32;
        let k = 10_240usize;
        let n1 = 320u32;
        let n2 = 4u32;

        // Without scratch, grid.z cleanly stays at 1 (0-scratch fast path).
        assert_eq!(
            select_dual_split_k_grid_z(DualCoalescedTier::N64Sk1M64V2, 6, 4, k, sm120, false),
            1
        );

        let expected: &[(usize, DualCoalescedTier, u32)] = &[
            (4, DualCoalescedTier::N8Sk8M32V2, 5),
            (8, DualCoalescedTier::N8Sk8M32V2, 5),
            (16, DualCoalescedTier::N8Sk8M32V2, 5),
            (24, DualCoalescedTier::N8Sk8M32V2, 5),
            (32, DualCoalescedTier::N32Sk2M64V2, 20),
            (48, DualCoalescedTier::N32Sk2M64V2, 20),
            (64, DualCoalescedTier::N32Sk2M64V2, 20),
            (96, DualCoalescedTier::N32Sk2M64V2, 10),
            (128, DualCoalescedTier::N32Sk2M64V2, 10),
            (192, DualCoalescedTier::N32Sk2M64V2, 10),
            (256, DualCoalescedTier::N64Sk1M64V2, 10),
        ];
        for &(rows, want_tier, want_gz) in expected {
            let (tier, _) = resolve_hc_dual_coalesced_for_sms(
                rows, k_n8, k_n16_v2, k_n32, high_sm, true, sm120,
            )
            .expect("high-SM tier");
            assert_eq!(tier, want_tier, "rows={rows}");
            let grid_x = n1.div_ceil(tier.co_n()) + n2.div_ceil(tier.co_n());
            let grid_y = (rows as u32).div_ceil(tier.tile_m());
            let gz = select_dual_split_k_grid_z(tier, grid_x, grid_y, k, sm120, true);
            assert_eq!(gz, want_gz, "rows={rows} tier={tier:?}");
            let ctas = (grid_x as usize) * (grid_y as usize) * (gz as usize);
            assert!(
                ctas >= sm120 as usize && ctas <= (sm120 as usize) * 2,
                "rows={rows} ctas={ctas} must saturate 188 SMs within 2 CTAs/SM"
            );
            let floats = ctas * (tier.tile_m() as usize) * (tier.co_n() as usize);
            assert!(
                floats <= DUAL_SPLITK_MAX_FLOATS,
                "rows={rows} floats={floats} exceeds DUAL_SPLITK_MAX_FLOATS"
            );
        }

        // Fallback on SM120 when n8_sk8_m32_v2 is unavailable: degrades cleanly to N8Sk8M32.
        let no_n8_v2 = DualHighSmKernels {
            n8_sk8_m32_v2: KernelHandle(0),
            ..high_sm
        };
        assert_eq!(
            resolve_hc_dual_coalesced_for_sms(8, k_n8, k_n16_v2, k_n32, no_n8_v2, true, sm120)
                .map(|(t, h)| (t, h.0)),
            Some((DualCoalescedTier::N8Sk8M32, k_n8.0))
        );
        // PDL gating keeps GB10 (<= 64 SMs) unchanged and enables on high-SM Blackwell (> 64 SMs).
        assert!(!super::hc_pdl_enabled_for_sm(48));
        assert!(!super::hc_pdl_enabled_for_sm(64));
        assert!(super::hc_pdl_enabled_for_sm(188));

        // Fallback chain on SM120 when n64 / n32 are unavailable: n16_sk4_m64_v2 still saturates SM120.
        let only_n16_m64 = DualHighSmKernels {
            n16_sk4_m64_v2: KernelHandle(11),
            ..DualHighSmKernels::empty()
        };
        for &(rows, want_gz) in &[(64usize, 10u32), (128, 5), (256, 4)] {
            let (tier, handle) = resolve_hc_dual_coalesced_for_sms(
                rows,
                k_n8,
                k_n16_v2,
                k_n32,
                only_n16_m64,
                true,
                sm120,
            )
            .expect("n16_sk4_m64_v2 fallback");
            assert_eq!(tier, DualCoalescedTier::N16Sk4M64V2);
            assert_eq!(handle.0, 11);
            let grid_x = n1.div_ceil(tier.co_n()) + n2.div_ceil(tier.co_n());
            let grid_y = (rows as u32).div_ceil(tier.tile_m());
            let gz = select_dual_split_k_grid_z(tier, grid_x, grid_y, k, sm120, true);
            assert_eq!(gz, want_gz, "n16_sk4_m64_v2 rows={rows}");
            assert!(grid_x * grid_y * gz >= sm120);
        }
    }

    #[test]
    fn verify_hc_fuse_scope_defers_consumes_rejects_double_defer_and_cleans_on_drop() {
        use super::{
            PendingVerifyHcInject, VerifyHcFuseScope, defer_pending_verify_hc_inject,
            take_pending_verify_hc_inject, verify_hc_defer_mlp_tail_allowed,
        };

        assert!(!verify_hc_defer_mlp_tail_allowed());
        assert_eq!(take_pending_verify_hc_inject(), None);

        let entry = PendingVerifyHcInject {
            hyper: DevicePtr(0x1000),
            core: DevicePtr(0x2000),
            residual: DevicePtr(0x3000),
            rows: 8,
            stream: 42,
        };
        assert!(entry.matches(DevicePtr(0x1000), DevicePtr(0x3000), 8, 42));
        assert!(!entry.matches(DevicePtr(0x1004), DevicePtr(0x3000), 8, 42));
        assert!(!entry.matches(DevicePtr(0x1000), DevicePtr(0x3004), 8, 42));
        assert!(!entry.matches(DevicePtr(0x1000), DevicePtr(0x3000), 9, 42));
        assert!(!entry.matches(DevicePtr(0x1000), DevicePtr(0x3000), 8, 43));

        {
            let mut scope = VerifyHcFuseScope::enter();
            assert!(!verify_hc_defer_mlp_tail_allowed());
            assert_eq!(take_pending_verify_hc_inject(), None);

            scope.set_defer_mlp_tail(true);
            assert!(verify_hc_defer_mlp_tail_allowed());
            defer_pending_verify_hc_inject(entry).expect("first defer succeeds");
            assert!(
                defer_pending_verify_hc_inject(entry).is_err(),
                "double defer without consume must fail closed"
            );
            assert_eq!(take_pending_verify_hc_inject(), Some(entry));
            assert_eq!(take_pending_verify_hc_inject(), None);

            defer_pending_verify_hc_inject(entry).expect("re-defer succeeds");
        }
        assert!(!verify_hc_defer_mlp_tail_allowed());
        assert_eq!(
            take_pending_verify_hc_inject(),
            None,
            "scope drop must discard stale deferred inject"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_parity_n8_sk8_m32_v2_and_inject_group_norm_v2_pdl() -> anyhow::Result<()> {
        use super::DUAL_SPLITK_MAX_TILES;
        use spark_runtime::cuda_backend::AtlasCudaBackend;
        use spark_runtime::gpu::GpuBackend;
        use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

        let Some(ptx) = atlas_kernels::ptx_for_model("flash-next") else {
            return Ok(());
        };
        let Ok(gpu) = AtlasCudaBackend::new(0, &ptx.modules) else {
            return Ok(());
        };
        let stream = gpu.default_stream();

        let k_ref = gpu.kernel("w4a16", "w4a16_gemm_coalesced_n8_sk8_m32_dual_silu_div")?;
        let k_v2 = gpu.kernel("w4a16", "w4a16_gemm_coalesced_n8_sk8_m32_dual_silu_div_v2")?;
        let k_inject = gpu.kernel("qwen4_hyper", "qwen4_hc_inject_saved")?;
        let k_gn = gpu.kernel("qwen4_hyper", "qwen4_hc_group_norm")?;
        let k_ign_v2 = gpu.kernel("qwen4_hyper", "qwen4_hc_inject_group_norm_v2")?;
        let k_upmix = gpu.kernel("w4a16", "qwen4_hc_upmix_coalesced_m16")?;

        // 1. Parity & OOB canary check for `N8Sk8M32V2` (`grid_z=5`, 205 CTAs) across M=1..=24.
        let max_m = 24usize;
        let k = 10_240usize;
        let n1 = 320usize;
        let n2 = 4usize;
        let canary_elems = 64usize;
        let canary_u16: u16 = 0x5A3C;

        let mut a_host = vec![0u16; max_m * k];
        for (i, v) in a_host.iter_mut().enumerate() {
            let f = (((i.wrapping_mul(37).wrapping_add(11)) % 200) as f32 - 100.0) * 0.004;
            *v = half::bf16::from_f32(f).to_bits();
        }
        let make_nvfp4 = |n: usize, seed: usize| -> (Vec<u8>, Vec<u8>, f32) {
            let packed = (0..n * (k / 2))
                .map(|i| {
                    let lo = ((i.wrapping_mul(13).wrapping_add(seed)) & 0xF) as u8;
                    let hi = ((i.wrapping_mul(29).wrapping_add(seed * 3)) & 0xF) as u8;
                    lo | (hi << 4)
                })
                .collect();
            let scales = (0..n * (k / 16))
                .map(|i| 0x38u8.wrapping_add(((i + seed) % 8) as u8))
                .collect();
            (packed, scales, 0.0625f32)
        };
        let (b1_p, b1_s, s2_1) = make_nvfp4(n1, 1);
        let (b2_p, b2_s, s2_2) = make_nvfp4(n2, 7);

        let alloc_upload = |bytes: &[u8]| -> anyhow::Result<DevicePtr> {
            let ptr = gpu.alloc(bytes.len())?;
            gpu.copy_h2d(bytes, ptr)?;
            Ok(ptr)
        };
        // SAFETY: `a_host` is an initialized `Vec<u16>` of POD elements (`a_host.len() * 2` bytes).
        let a_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(a_host.as_ptr().cast::<u8>(), a_host.len() * 2) };
        let a_dev = alloc_upload(a_bytes)?;
        let b1_p_dev = alloc_upload(&b1_p)?;
        let b1_s_dev = alloc_upload(&b1_s)?;
        let b2_p_dev = alloc_upload(&b2_p)?;
        let b2_s_dev = alloc_upload(&b2_s)?;

        let c1_total = max_m * n1 + canary_elems;
        let c2_total = max_m * n2 + canary_elems;
        let c1_ref_dev = gpu.alloc(c1_total * 2)?;
        let c2_ref_dev = gpu.alloc(c2_total * 2)?;
        let c1_v2_dev = gpu.alloc(c1_total * 2)?;
        let c2_v2_dev = gpu.alloc(c2_total * 2)?;
        let partials_dev = gpu.alloc(DUAL_SPLITK_MAX_FLOATS * 4)?;
        let counters_dev = gpu.alloc(DUAL_SPLITK_MAX_TILES * 4)?;
        gpu.memset_async(counters_dev, 0, DUAL_SPLITK_MAX_TILES * 4, stream)?;

        let canary_c1 = vec![canary_u16; c1_total];
        let canary_c2 = vec![canary_u16; c2_total];
        // SAFETY: `canary_c1` is an initialized `Vec<u16>` of POD elements (`c1_total * 2` bytes).
        let canary_c1_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(canary_c1.as_ptr().cast::<u8>(), canary_c1.len() * 2)
        };
        // SAFETY: `canary_c2` is an initialized `Vec<u16>` of POD elements (`c2_total * 2` bytes).
        let canary_c2_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(canary_c2.as_ptr().cast::<u8>(), canary_c2.len() * 2)
        };

        let bf16_ulp = |a: u16, b: u16| -> u32 {
            let to_ord = |bits: u16| -> i32 {
                if (bits & 0x8000) != 0 {
                    -((bits & 0x7fff) as i32)
                } else {
                    bits as i32
                }
            };
            (to_ord(a) - to_ord(b)).unsigned_abs()
        };

        for rows in [1usize, 4, 8, 9, 16, 24] {
            gpu.copy_h2d(canary_c1_bytes, c1_ref_dev)?;
            gpu.copy_h2d(canary_c2_bytes, c2_ref_dev)?;
            gpu.copy_h2d(canary_c1_bytes, c1_v2_dev)?;
            gpu.copy_h2d(canary_c2_bytes, c2_v2_dev)?;

            let grid_x = div_ceil(n1 as u32, 8) + div_ceil(n2 as u32, 8);
            assert_eq!(grid_x, 41);
            let gz =
                select_dual_split_k_grid_z(DualCoalescedTier::N8Sk8M32V2, grid_x, 1, k, 188, true);
            assert_eq!(gz, 5);

            KernelLaunch::new(&gpu, k_ref)
                .grid([grid_x, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(a_dev)
                .arg_ptr(b1_p_dev)
                .arg_ptr(b1_s_dev)
                .arg_f32(s2_1)
                .arg_ptr(c1_ref_dev)
                .arg_u32(n1 as u32)
                .arg_ptr(b2_p_dev)
                .arg_ptr(b2_s_dev)
                .arg_f32(s2_2)
                .arg_ptr(c2_ref_dev)
                .arg_u32(n2 as u32)
                .arg_u32(rows as u32)
                .arg_u32(k as u32)
                .arg_f32(4.0f32)
                .launch(stream)?;

            KernelLaunch::new(&gpu, k_v2)
                .grid([grid_x, 1, gz])
                .block([256, 1, 1])
                .pdl(true)
                .arg_ptr(a_dev)
                .arg_ptr(b1_p_dev)
                .arg_ptr(b1_s_dev)
                .arg_f32(s2_1)
                .arg_ptr(c1_v2_dev)
                .arg_u32(n1 as u32)
                .arg_ptr(b2_p_dev)
                .arg_ptr(b2_s_dev)
                .arg_f32(s2_2)
                .arg_ptr(c2_v2_dev)
                .arg_u32(n2 as u32)
                .arg_u32(rows as u32)
                .arg_u32(k as u32)
                .arg_f32(4.0f32)
                .arg_ptr(partials_dev)
                .arg_ptr(counters_dev)
                .launch(stream)?;
            gpu.synchronize(stream)?;

            let mut ref1 = vec![0u16; c1_total];
            let mut v2_1 = vec![0u16; c1_total];
            let mut ref2 = vec![0u16; c2_total];
            let mut v2_2 = vec![0u16; c2_total];
            // SAFETY: each output buffer is an exclusively borrowed initialized `Vec<u16>` of POD
            // elements (`len * 2` bytes); any byte pattern written by `copy_d2h` is a valid `u16`.
            gpu.copy_d2h(c1_ref_dev, unsafe {
                std::slice::from_raw_parts_mut(ref1.as_mut_ptr().cast::<u8>(), ref1.len() * 2)
            })?;
            // SAFETY: see `ref1` above.
            gpu.copy_d2h(c1_v2_dev, unsafe {
                std::slice::from_raw_parts_mut(v2_1.as_mut_ptr().cast::<u8>(), v2_1.len() * 2)
            })?;
            // SAFETY: see `ref1` above.
            gpu.copy_d2h(c2_ref_dev, unsafe {
                std::slice::from_raw_parts_mut(ref2.as_mut_ptr().cast::<u8>(), ref2.len() * 2)
            })?;
            // SAFETY: see `ref1` above.
            gpu.copy_d2h(c2_v2_dev, unsafe {
                std::slice::from_raw_parts_mut(v2_2.as_mut_ptr().cast::<u8>(), v2_2.len() * 2)
            })?;

            let mut max_ulp_1 = 0u32;
            for i in 0..rows * n1 {
                max_ulp_1 = max_ulp_1.max(bf16_ulp(ref1[i], v2_1[i]));
            }
            assert!(
                max_ulp_1 <= 1,
                "N8Sk8M32V2 down rows={rows}: max BF16 ULP={max_ulp_1} > 1"
            );
            for i in rows * n1..c1_total {
                assert_eq!(
                    v2_1[i], canary_u16,
                    "N8Sk8M32V2 down OOB write at elem {i} for rows={rows}"
                );
            }
            let mut max_ulp_2 = 0u32;
            for i in 0..rows * n2 {
                max_ulp_2 = max_ulp_2.max(bf16_ulp(ref2[i], v2_2[i]));
            }
            assert!(
                max_ulp_2 <= 1,
                "N8Sk8M32V2 inject rows={rows}: max BF16 ULP={max_ulp_2} > 1"
            );
            for i in rows * n2..c2_total {
                assert_eq!(
                    v2_2[i], canary_u16,
                    "N8Sk8M32V2 inject OOB write at elem {i} for rows={rows}"
                );
            }
        }

        // 2. Bit-exact 0-ULP parity for deferred `inject_saved + group_norm` vs fused
        //    `qwen4_hc_inject_group_norm_v2` with `.pdl(true)` (both eager AND CUDA-graphed).
        let rows = 8usize;
        let h = 2_560usize;
        let hc = 4usize;
        let r = h * hc;
        let mut hyper_init = vec![0u16; rows * r];
        let mut core_init = vec![0u16; rows * h];
        let mut resid_init = vec![0u16; rows * r];
        let mut norm_w = vec![0u16; r];
        for (i, v) in hyper_init.iter_mut().enumerate() {
            *v = half::bf16::from_f32((((i * 19 + 3) % 120) as f32 - 60.0) * 0.01).to_bits();
        }
        for (i, v) in core_init.iter_mut().enumerate() {
            *v = half::bf16::from_f32((((i * 23 + 5) % 120) as f32 - 60.0) * 0.01).to_bits();
        }
        for t in 0..rows {
            for s in 0..hc {
                resid_init[t * r + (r - hc) + s] =
                    half::bf16::from_f32(0.15 + 0.05 * ((t + s) as f32)).to_bits();
            }
        }
        for (i, v) in norm_w.iter_mut().enumerate() {
            *v = half::bf16::from_f32(0.8 + 0.001 * ((i % 400) as f32)).to_bits();
        }

        let as_bytes = |v: &[u16]| -> &[u8] {
            // SAFETY: `v` is an initialized `&[u16]` POD slice (`v.len() * 2` bytes).
            unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 2) }
        };
        let hyper_ref = alloc_upload(as_bytes(&hyper_init))?;
        let resid_ref = alloc_upload(as_bytes(&resid_init))?;
        let hyper_fused = alloc_upload(as_bytes(&hyper_init))?;
        let resid_fused = alloc_upload(as_bytes(&resid_init))?;
        let core_dev = alloc_upload(as_bytes(&core_init))?;
        let norm_w_dev = alloc_upload(as_bytes(&norm_w))?;

        // Reference: separate inject_saved_batched -> group_norm.
        KernelLaunch::new(&gpu, k_inject)
            .grid([rows as u32, div_ceil(r as u32, 256), 1])
            .block([256, 1, 1])
            .arg_ptr(hyper_ref)
            .arg_ptr(core_dev)
            .arg_ptr(resid_ref)
            .arg_u32(h as u32)
            .arg_u32(hc as u32)
            .launch(stream)?;
        KernelLaunch::new(&gpu, k_gn)
            .grid([rows as u32, hc as u32, 1])
            .block([1024, 1, 1])
            .arg_ptr(hyper_ref)
            .arg_ptr(norm_w_dev)
            .arg_ptr(resid_ref)
            .arg_u32(h as u32)
            .arg_u32(hc as u32)
            .arg_f32(1e-6f32)
            .launch(stream)?;

        // Fused `inject_group_norm_v2` -> `hc_dual_v2` -> `hc_upmix` with `.pdl(true)` inside a CUDA graph.
        let up_p = vec![0x21u8; r * (n1 / 2)];
        let up_s = vec![0x38u8; r * (n1 / 16)];
        let up_p_dev = alloc_upload(&up_p)?;
        let up_s_dev = alloc_upload(&up_s)?;
        let mixed_dev = gpu.alloc(rows * h * 2)?;

        let graph_stream = gpu.create_stream()?;
        gpu.synchronize(stream)?;
        let capture = crate::model::trait_impl::capture_guard::CaptureGuard::begin(
            &gpu,
            graph_stream,
            "qwen4_hyper_pdl_test",
        )?;
        KernelLaunch::new(&gpu, k_ign_v2)
            .grid([rows as u32, 4, 1])
            .block([320, 1, 1])
            .pdl(true)
            .arg_ptr(hyper_fused)
            .arg_ptr(core_dev)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(norm_w_dev)
            .arg_ptr(resid_fused)
            .arg_ptr(DevicePtr::NULL)
            .arg_u32(h as u32)
            .arg_u32(hc as u32)
            .arg_f32(1e-6f32)
            .launch(graph_stream)?;
        KernelLaunch::new(&gpu, k_v2)
            .grid([41, 1, 5])
            .block([256, 1, 1])
            .pdl(true)
            .arg_ptr(resid_fused)
            .arg_ptr(b1_p_dev)
            .arg_ptr(b1_s_dev)
            .arg_f32(s2_1)
            .arg_ptr(c1_v2_dev)
            .arg_u32(n1 as u32)
            .arg_ptr(b2_p_dev)
            .arg_ptr(b2_s_dev)
            .arg_f32(s2_2)
            .arg_ptr(c2_v2_dev)
            .arg_u32(n2 as u32)
            .arg_u32(rows as u32)
            .arg_u32(k as u32)
            .arg_f32(4.0f32)
            .arg_ptr(partials_dev)
            .arg_ptr(counters_dev)
            .launch(graph_stream)?;
        KernelLaunch::new(&gpu, k_upmix)
            .grid([(h / 16) as u32, 1, 1])
            .block([256, 1, 1])
            .pdl(true)
            .arg_ptr(c1_v2_dev)
            .arg_ptr(up_p_dev)
            .arg_ptr(up_s_dev)
            .arg_f32(0.0625f32)
            .arg_ptr(resid_fused)
            .arg_ptr(c2_v2_dev)
            .arg_ptr(mixed_dev)
            .arg_u32(rows as u32)
            .arg_u32(h as u32)
            .arg_u32(n1 as u32)
            .launch(graph_stream)?;
        let graph = capture.end()?;
        gpu.launch_graph(graph, graph_stream)?;
        gpu.synchronize(graph_stream)?;

        let mut h_ref_out = vec![0u16; rows * r];
        let mut h_fus_out = vec![0u16; rows * r];
        // SAFETY: `h_ref_out` is an exclusively borrowed initialized `Vec<u16>` (`len * 2` bytes).
        gpu.copy_d2h(hyper_ref, unsafe {
            std::slice::from_raw_parts_mut(h_ref_out.as_mut_ptr().cast::<u8>(), h_ref_out.len() * 2)
        })?;
        // SAFETY: `h_fus_out` is an exclusively borrowed initialized `Vec<u16>` (`len * 2` bytes).
        gpu.copy_d2h(hyper_fused, unsafe {
            std::slice::from_raw_parts_mut(h_fus_out.as_mut_ptr().cast::<u8>(), h_fus_out.len() * 2)
        })?;
        assert_eq!(
            h_ref_out, h_fus_out,
            "deferred inject + group_norm_v2 must be 0-ULP bit-identical on hyper"
        );

        gpu.destroy_graph(graph)?;
        for ptr in [
            a_dev,
            b1_p_dev,
            b1_s_dev,
            b2_p_dev,
            b2_s_dev,
            c1_ref_dev,
            c2_ref_dev,
            c1_v2_dev,
            c2_v2_dev,
            partials_dev,
            counters_dev,
            hyper_ref,
            resid_ref,
            hyper_fused,
            resid_fused,
            core_dev,
            norm_w_dev,
            up_p_dev,
            up_s_dev,
            mixed_dev,
        ] {
            let _ = gpu.free(ptr);
        }
        Ok(())
    }

    #[test]
    fn deterministic_batch_pins_hyperconnection_routes_across_1_to_128() {
        use super::{
            BatchedHcTcRoute, DualCoalescedTier, select_batched_hc_down_route_v2_with_policy,
            select_batched_hc_tc_route_with_policy, select_batched_hc_up_route_v2_with_policy,
            select_dual_split_k_grid_z_with_policy,
        };
        use crate::layers::deterministic_batch::DeterministicBatchPolicy;

        for m in [1usize, 2, 4, 8, 16, 32, 64, 128] {
            assert_eq!(
                select_batched_hc_tc_route_with_policy(
                    true,
                    true,
                    true,
                    DeterministicBatchPolicy::Enabled,
                ),
                BatchedHcTcRoute::Exact,
                "HC TC route must pin Exact at M={m}"
            );
            assert_eq!(
                select_batched_hc_down_route_v2_with_policy(
                    m,
                    true,
                    true,
                    true,
                    true,
                    true,
                    true,
                    true,
                    true,
                    DeterministicBatchPolicy::Enabled,
                ),
                None,
                "HC down fast-TC route must be disabled at M={m}"
            );
            assert_eq!(
                select_batched_hc_up_route_v2_with_policy(
                    m,
                    true,
                    true,
                    true,
                    true,
                    true,
                    DeterministicBatchPolicy::Enabled,
                ),
                None,
                "HC up fast-TC route must be disabled at M={m}"
            );
            for sm in [48u32, 188u32] {
                assert_eq!(
                    select_dual_split_k_grid_z_with_policy(
                        DualCoalescedTier::N8Sk8M32V2,
                        41,
                        1,
                        10_240,
                        sm,
                        true,
                        DeterministicBatchPolicy::Enabled,
                    ),
                    1,
                    "HC dual split-K grid_z must pin to 1 at M={m} sm={sm}"
                );
            }
        }
        assert_eq!(
            select_batched_hc_tc_route_with_policy(
                true,
                true,
                true,
                DeterministicBatchPolicy::Disabled,
            ),
            BatchedHcTcRoute::FastTc
        );
    }
}
