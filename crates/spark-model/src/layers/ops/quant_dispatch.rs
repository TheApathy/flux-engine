// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Unified GEMV dispatch: select kernel based on weight quantization format.
///
/// Eliminates cascading if/else chains in layer forward methods. The enum
/// branch (~1 cycle) is negligible vs GPU kernel launch overhead (~5μs).
#[allow(clippy::too_many_arguments)]
pub fn quant_gemv(
    gpu: &dyn GpuBackend,
    gemv_nvfp4: KernelHandle,
    gemv_fp8: KernelHandle,
    gemv_dense: KernelHandle,
    input: DevicePtr,
    weight: &crate::weight_map::QuantWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    use crate::weight_map::QuantWeight;
    match weight {
        QuantWeight::Nvfp4(w) => w4a16_gemv(gpu, gemv_nvfp4, input, w, output, n, k, stream),
        QuantWeight::Fp8(w) => w8a16_gemv(
            gpu,
            gemv_fp8,
            input,
            w.weight,
            w.row_scale,
            output,
            n,
            k,
            stream,
        ),
        QuantWeight::Dense(w) => dense_gemv(gpu, gemv_dense, input, w, output, n, k, stream),
    }
}

/// Unified GEMM dispatch: select kernel based on weight quantization format.
///
/// For M>1 prefill projections (Q/K/V/O). Falls back to dense GEMM for BF16.
#[allow(clippy::too_many_arguments)]
pub fn quant_gemm(
    gpu: &dyn GpuBackend,
    gemm_nvfp4: KernelHandle,
    gemm_fp8: KernelHandle,
    gemm_dense: KernelHandle,
    input: DevicePtr,
    weight: &crate::weight_map::QuantWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    use crate::weight_map::QuantWeight;
    match weight {
        QuantWeight::Nvfp4(w) => w4a16_gemm(gpu, gemm_nvfp4, input, w, output, m, n, k, stream),
        QuantWeight::Fp8(w) => w8a16_gemm(
            gpu,
            gemm_fp8,
            input,
            w.weight,
            w.row_scale,
            output,
            m,
            n,
            k,
            stream,
        ),
        QuantWeight::Dense(w) => dense_gemm(gpu, gemm_dense, input, w, output, m, n, k, stream),
    }
}

/// W4A16 GEMV (M=1): C = A @ dequant(B) for single-row activations.
///
/// A: [1, K] BF16, B: NVFP4 packed, C: [1, N] BF16.
/// 4 outputs/block, 64 threads (2 warps) per output. Cross-warp smem reduction.
///
/// Kernel: `w4a16_gemv(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
///
/// ## Tuning sweep 2026-05-19 — keeping baseline
///
/// Profiled at 59-60% LPDDR5X bandwidth on Qwen3.6-27B M=1 small-projection
/// path (SSM qkvz_proj, attn Q/K/V/O). Four alternate block shapes were
/// benchmarked end-to-end on the K=3 verify path (mean tok/s over count /
/// essay / fruits):
///
///   variant            block_shape          mean tok/s   delta vs base
///   baseline           N=4, t=64, blk=256   20.92        —
///   v1                 N=2, t=128, blk=256  20.55        -1.8%
///   v2                 N=1, t=256, blk=256  19.69        -5.9%
///   v3                 N=8, t=32, blk=256   20.73        -0.9%
///   v4                 N=2, t=64, blk=128   20.92        ~0%
///
/// None beat baseline. K dims on these projections (~4096) are too small for
/// the inner loop to fully hide load latency; the existing (N=4, t=64) shape
/// is close to optimal for sm_121 occupancy. The variant kernels remain in
/// kernels/gb10/nvfp4/w4a16_gemv.cu as documented future-work candidates
/// (zero runtime cost when not loaded).
pub fn w4a16_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 double-GEMV (M=2): reads weights once, computes 2 outputs.
///
/// A: [2, K] BF16 contiguous, B: NVFP4 packed, C: [2, N] BF16 contiguous.
/// Same weight bandwidth as single GEMV — eliminates GEMM M=2 tile waste.
///
/// Kernel: `w4a16_gemv_batch2(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Fused RMS Norm + Residual Save + W4A16 GEMV.
///
/// Replaces the `rms_norm_residual` + `w4a16_gemv` pair in front of the SSM
/// QKVZ projection. One CTA cooperates to compute the RMS-normalized hidden
/// state (writing both the normed and residual buffers), then each of the 4
/// outputs per CTA streams the W4A16 weights once and FMAs against the
/// smem-resident normed vector.
///
/// Dynamic shared memory:
///   `K * 2`  bytes for the normed BF16 vector
///   + `16 * 4` bytes for the E2M1 LUT
///   + `(BLOCK_SIZE / WARP_SIZE) * 4 = 32` bytes for warp partial sums
///   + `N_PER_BLOCK * 2 * 4 = 32` bytes for the GEMV cross-warp reduction
///
/// Kernel: `rms_norm_residual_w4a16_gemv(input, gamma, normed, residual,
///                                       B_packed, B_scale, scale2, C, N, K, eps)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_residual_w4a16_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gamma: &DenseWeight,
    normed_out: DevicePtr,
    residual_out: DevicePtr,
    weight: &QuantizedWeight,
    gemv_out: DevicePtr,
    n: u32,
    k: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    // K * sizeof(BF16) + LUT + warp partial sums + GEMV reduction scratch.
    let smem_bytes = k * 2 + 16 * 4 + 32 + 32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(input)
        .arg_ptr(gamma.weight)
        .arg_ptr(normed_out)
        .arg_ptr(residual_out)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(gemv_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_f32(eps)
        .launch(stream)
}

/// Fused RMS Norm + Residual Save + W4A16 batch-3 GEMV.
///
/// K=3 speculative-verify counterpart of `rms_norm_residual_w4a16_gemv`.
/// Inputs `input`, `residual`, `normed_out` are [3, K]; `gemv_out` is [3, N].
///
/// Dynamic shared memory:
///   `3 * K * 2` bytes for 3 normalized BF16 vectors
///   + 16 * 4 (LUT) + 32 (warp partials) + 32 * 3 (M=3 GEMV reduction) bytes
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_residual_w4a16_gemv_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gamma: &DenseWeight,
    normed_out: DevicePtr,
    residual_out: DevicePtr,
    weight: &QuantizedWeight,
    gemv_out: DevicePtr,
    n: u32,
    k: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    let smem_bytes = 3 * k * 2 + 16 * 4 + 32 + 32 * 3;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(input)
        .arg_ptr(gamma.weight)
        .arg_ptr(normed_out)
        .arg_ptr(residual_out)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(gemv_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_f32(eps)
        .launch(stream)
}

/// W4A16 triple-GEMV (M=3): reads weights once, computes 3 outputs.
///
/// A: [3, K] BF16 contiguous, B: NVFP4 packed, C: [3, N] BF16 contiguous.
/// For K=3 speculative verification.
///
/// Kernel: `w4a16_gemv_batch3(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 triple-GEMV (M=3) specialized for the LM head (large N=vocab).
///
/// Same M=3 algorithm as `w4a16_gemv_batch3` — reads each NVFP4 weight row
/// once and FMAs against 3 input rows — but uses `N_PER_BLOCK=8` (vs 4)
/// to halve the grid at the LM-head's huge N (≈248k for Qwen3.6-27B) and
/// 1 warp per output so there is NO cross-warp shared-memory reduce.
///
/// A: [3, K] BF16 contiguous, B: NVFP4 packed, C: [3, N] BF16 contiguous.
///
/// Kernel: `w4a16_gemv_batch3_logits(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/8), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv_batch3_logits(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMV with inline Q/Gate deinterleave on output write.
///
/// Same as `w4a16_gemv` but writes Q and Gate to deinterleaved positions,
/// eliminating the separate `deinterleave_qg` kernel (12 graph nodes saved).
///
/// Kernel: `w4a16_gemv_qg(A, B, S, s2, C, N, K, num_heads, head_dim)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// W4A16 GEMV with inline QKVZ deinterleave on output write.
///
/// Same as `w4a16_gemv` but writes to deinterleaved output locations,
/// eliminating the separate `deinterleave_qkvz` kernel.
///
/// Kernel: `w4a16_gemv_qkvz(A, B, S, s2, C, N, K, ng, kd, vpg, vd)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qkvz(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_groups: u32,
    head_k_dim: u32,
    vheads_per_group: u32,
    head_v_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_groups)
        .arg_u32(head_k_dim)
        .arg_u32(vheads_per_group)
        .arg_u32(head_v_dim)
        .launch(stream)
}

/// Q+Gate GEMV for 2 tokens with inline deinterleave.
///
/// Reads the Q+Gate weight matrix once, produces 2 deinterleaved output
/// vectors (Q|Gate for each token). Replaces 2× `w4a16_gemv_qg` calls.
///
/// Kernel: `w4a16_gemv_qg_batch2(A, B, S, s2, C, N, K, num_heads, head_dim)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
/// Input A: [2, K], Output C: [2, N] deinterleaved [Q|G] per token.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// W4A16 GEMV batch3 with inline Q/Gate deinterleave.
///
/// Reads the Q+Gate weight matrix once, produces 3 deinterleaved output
/// vectors (Q|Gate for each token). For K=3 speculative verification.
///
/// Kernel: `w4a16_gemv_qg_batch3(A, B, S, s2, C, N, K, num_heads, head_dim)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
/// Input A: [3, K], Output C: [3, N] deinterleaved [Q|G] per token.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// Dual-projection GEMV for 3 tokens (K+V or any 2 weight matrices).
///
/// Reads each weight matrix once, produces 3 output vectors per projection.
/// `blockIdx.z` selects projection 0 or 1.
///
/// Kernel: `w4a16_gemv_dual_batch3(A, B0, S0, s2_0, C0, B1, S1, s2_1, C1, N, K)`
/// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
/// Input A: [3, K], Output C0: [3, N], C1: [3, N].
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight0: &QuantizedWeight,
    output0: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight0.weight)
        .arg_ptr(weight0.weight_scale)
        .arg_f32(weight0.weight_scale_2)
        .arg_ptr(output0)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Tuned dual-projection GEMV for 3 tokens — gate+up fused into one CTA.
///
/// Geometry: 8 outputs per CTA (4 gate + 4 up, dispatched by warp), 64
/// threads per output (2 warps), 512 threads/block. Grid drops to
/// (ceil(N/4), 1, 1) — half the CTAs of [`w4a16_gemv_dual_batch3`] which
/// used z=2 to fan out gate/up. Each CTA reads the 3-token activation
/// vector once and reuses it across both projections via L1, halving the
/// L2 A-read traffic. Inner loop processes 16 K-values per iter (2× uint4
/// acts + 8-byte weight load + 1 FP8 scale) so the K-loop trip count and
/// scale-fetch frequency are both halved vs the baseline's K8 stride.
/// Output buffers byte-equivalent to the baseline.
///
/// Kernel: `w4a16_gemv_dual_batch3_tuned(A, B0, S0, s2_0, C0, B1, S1, s2_1, C1, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (512, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_batch3_tuned(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight0: &QuantizedWeight,
    output0: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([512, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight0.weight)
        .arg_ptr(weight0.weight_scale)
        .arg_f32(weight0.weight_scale_2)
        .arg_ptr(output0)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Strided W4A16 GEMV batch3 + inline Q/Gate deinterleave.
///
/// Like [`w4a16_gemv_qg_batch3`] but writes token i to
/// `output + i * out_stride_bf16` (in BF16 elements) so the K=3 verify
/// path can scatter directly into the interleaved `qkv_buf` layout
/// without a follow-up d2d copy.
///
/// Kernel: `w4a16_gemv_qg_batch3_strided(A, B, S, s2, C, N, K, num_heads, head_dim, out_stride)`
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg_batch3_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    out_stride_bf16: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(out_stride_bf16)
        .launch(stream)
}

/// Strided dual-projection GEMV for 3 tokens.
///
/// Like [`w4a16_gemv_dual_batch3`] but writes token i to
/// `output{0,1} + i * out_stride_bf16` (in BF16 elements). Both
/// projections share the same per-token stride.
///
/// Kernel: `w4a16_gemv_dual_batch3_strided(A, B0, S0, s2_0, C0, B1, S1, s2_1, C1, N, K, out_stride)`
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_batch3_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight0: &QuantizedWeight,
    output0: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    n: u32,
    k: u32,
    out_stride_bf16: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight0.weight)
        .arg_ptr(weight0.weight_scale)
        .arg_f32(weight0.weight_scale_2)
        .arg_ptr(output0)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride_bf16)
        .launch(stream)
}

/// Dual-projection GEMV for 2 tokens (K+V or any 2 weight matrices).
///
/// Reads each weight matrix once, produces 2 output vectors per projection.
/// `blockIdx.z` selects projection 0 or 1.
///
/// Kernel: `w4a16_gemv_dual_batch2(A, B0, S0, s2_0, C0, B1, S1, s2_1, C1, N, K)`
/// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
/// Input A: [2, K], Output C0: [2, N], C1: [2, N].
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight0: &QuantizedWeight,
    output0: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight0.weight)
        .arg_ptr(weight0.weight_scale)
        .arg_f32(weight0.weight_scale_2)
        .arg_ptr(output0)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

// ── Multi-seq SM120/SM103 cuBLASLt NVFP4 block-scaled projection dispatch ──

/// Minimum SM count for auto-enabling multi-seq cuBLASLt NVFP4 projections on
/// discrete Blackwell GPUs while keeping GB10 (`<= 64` SMs) default-OFF.
const MULTISEQ_CUBLASLT_NVFP4_SM_THRESHOLD: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MultiseqCublasLtNvfp4Policy {
    Disabled,
    Auto,
    Enabled,
}

#[must_use]
fn parse_multiseq_cublaslt_nvfp4_policy(raw: Option<&str>) -> MultiseqCublasLtNvfp4Policy {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => {
            MultiseqCublasLtNvfp4Policy::Disabled
        }
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => {
            MultiseqCublasLtNvfp4Policy::Enabled
        }
        _ => MultiseqCublasLtNvfp4Policy::Auto,
    }
}

#[must_use]
const fn resolve_multiseq_cublaslt_nvfp4(
    proj_policy: MultiseqCublasLtNvfp4Policy,
    ffn_policy: MultiseqCublasLtNvfp4Policy,
    sm_count: u32,
    cc: Option<(u32, u32)>,
) -> bool {
    match proj_policy {
        MultiseqCublasLtNvfp4Policy::Disabled => false,
        MultiseqCublasLtNvfp4Policy::Enabled => true,
        MultiseqCublasLtNvfp4Policy::Auto => {
            if sm_count <= MULTISEQ_CUBLASLT_NVFP4_SM_THRESHOLD {
                return false;
            }
            match ffn_policy {
                MultiseqCublasLtNvfp4Policy::Disabled => false,
                MultiseqCublasLtNvfp4Policy::Enabled => true,
                MultiseqCublasLtNvfp4Policy::Auto => {
                    matches!(cc, Some((12, 0) | (10, 3)))
                }
            }
        }
    }
}

/// True when `ATLAS_27B_PROJ_CUBLASLT_NVFP4` is enabled (explicitly or auto on SM120/SM103).
#[must_use]
pub fn multiseq_cublaslt_nvfp4_enabled(cc: Option<(u32, u32)>) -> bool {
    static PROJ: std::sync::OnceLock<MultiseqCublasLtNvfp4Policy> = std::sync::OnceLock::new();
    static FFN: std::sync::OnceLock<MultiseqCublasLtNvfp4Policy> = std::sync::OnceLock::new();
    let proj_policy = *PROJ.get_or_init(|| {
        let raw = std::env::var("ATLAS_27B_PROJ_CUBLASLT_NVFP4").ok();
        parse_multiseq_cublaslt_nvfp4_policy(raw.as_deref())
    });
    let ffn_policy = *FFN.get_or_init(|| {
        let raw = std::env::var("ATLAS_PREFILL_FFN_CUBLASLT_NVFP4").ok();
        parse_multiseq_cublaslt_nvfp4_policy(raw.as_deref())
    });
    resolve_multiseq_cublaslt_nvfp4(
        proj_policy,
        ffn_policy,
        atlas_core::device::active_sm_count(),
        cc,
    )
}

#[cfg(feature = "cuda")]
mod multiseq_cublaslt_nvfp4_cuda {
    use anyhow::{Context, Result, ensure};
    use spark_runtime::cublaslt::{
        BlockScaleScalars, NVFP4_BLOCK_SIZE, block_scale_padded_bytes, block_scale_swizzle_offset,
        nvfp4_gemm_block_scaled,
    };
    use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
    use spark_runtime::kernel_args::KernelLaunch;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use crate::layers::ops;
    use crate::weight_map::QuantizedWeight;

    const DYNAMIC_SCALARS_BYTES: usize = 32;
    const DYNAMIC_ZERO_PREFIX_BYTES: usize = 12;
    const DYNAMIC_GLOBAL_MAX_OFF: usize = 0;
    const DYNAMIC_SCALE2_OFF: usize = 4;
    const DYNAMIC_STATUS_OFF: usize = 8;
    const DYNAMIC_ALPHA0_OFF: usize = 12;
    const DYNAMIC_ALPHA1_OFF: usize = 16;
    const DYNAMIC_ALPHA2_OFF: usize = 20;

    #[derive(Clone, Copy)]
    struct CachedWeightScale {
        scales_128x4: DevicePtr,
        input_scale: Option<f32>,
    }

    #[derive(Clone, Copy)]
    struct QuantKernels {
        quant_static: KernelHandle,
        absmax: KernelHandle,
        from_absmax: KernelHandle,
        combined_alpha: KernelHandle,
    }

    struct SharedScratch {
        a_packed: DevicePtr,
        a_scale: DevicePtr,
        dynamic_scalars: DevicePtr,
        cap_m: usize,
        cap_k: usize,
        bound_stream: Option<u64>,
    }

    #[derive(Clone, Copy)]
    enum InputScale {
        Static(f32),
        DynamicDevice {
            scale2_a_f32: DevicePtr,
            status_u32: DevicePtr,
            alpha_slots: [DevicePtr; 3],
        },
    }

    fn resolve_kernels(gpu: &dyn GpuBackend) -> QuantKernels {
        let gpu_id = std::ptr::from_ref(gpu) as *const () as usize;
        static CACHE: OnceLock<Mutex<HashMap<usize, QuantKernels>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Ok(guard) = cache.lock()
            && let Some(&ks) = guard.get(&gpu_id)
        {
            return ks;
        }
        let ptx_k = crate::layers::try_kernel(
            gpu,
            "quantize_bf16_to_nvfp4_cutlass",
            "quantize_bf16_to_nvfp4_atlas_128x4_ptx",
        );
        let legacy_k = crate::layers::try_kernel(
            gpu,
            "quantize_bf16_to_nvfp4_cutlass",
            "quantize_bf16_to_nvfp4_atlas_128x4",
        );
        let quant_static = if ptx_k.0 != 0 { ptx_k } else { legacy_k };
        let from_absmax_ptx = if atlas_core::device::active_sm_count()
            > super::MULTISEQ_CUBLASLT_NVFP4_SM_THRESHOLD
        {
            crate::layers::try_kernel(
                gpu,
                "quantize_nvfp4",
                "quantize_bf16_to_nvfp4_atlas_128x4_from_absmax_ptx",
            )
        } else {
            KernelHandle(0)
        };
        let from_absmax_legacy = crate::layers::try_kernel(
            gpu,
            "quantize_bf16_to_nvfp4_cutlass",
            "quantize_bf16_to_nvfp4_atlas_128x4_from_absmax",
        );
        let ks = QuantKernels {
            quant_static,
            absmax: crate::layers::try_kernel(gpu, "quantize_nvfp4", "nvfp4_global_absmax"),
            from_absmax: if from_absmax_ptx.0 != 0 {
                from_absmax_ptx
            } else {
                from_absmax_legacy
            },
            combined_alpha: crate::layers::try_kernel(
                gpu,
                "quantize_bf16_to_nvfp4_cutlass",
                "nvfp4_dynamic_combined_alpha",
            ),
        };
        if let Ok(mut guard) = cache.lock() {
            guard.insert(gpu_id, ks);
        }
        ks
    }

    fn weight_ready(w: &QuantizedWeight) -> bool {
        !w.weight.is_null()
            && !w.weight_scale.is_null()
            && w.weight_scale_2.is_finite()
            && w.weight_scale_2 > 0.0
    }

    fn shape_ok(m: u32, n: u32, k: u32, ldc: u32) -> bool {
        m > 0
            && n > 0
            && k > 0
            && n.is_multiple_of(8)
            && k.is_multiple_of(64)
            && ldc >= n
            && ldc.is_multiple_of(8)
    }

    fn swizzle_scales_host(logical: &[u8], n: u32, k: u32) -> Result<Vec<u8>> {
        let k_blocks = k / NVFP4_BLOCK_SIZE;
        let expected_len = (n as usize)
            .checked_mul(k_blocks as usize)
            .context("multi-seq cuBLASLt NVFP4 scale length overflow")?;
        ensure!(
            logical.len() == expected_len,
            "multi-seq cuBLASLt NVFP4 scale length mismatch: expected {expected_len}, got {}",
            logical.len()
        );
        let padded_len = block_scale_padded_bytes(n, k, NVFP4_BLOCK_SIZE);
        let mut swizzled = vec![0u8; padded_len];
        for row in 0..n {
            let row_base = (row as usize) * (k_blocks as usize);
            for g in 0..k_blocks {
                let dst = block_scale_swizzle_offset(row, g, k_blocks);
                swizzled[dst] = logical[row_base + (g as usize)];
            }
        }
        Ok(swizzled)
    }

    fn cached_weight_scale(
        gpu: &dyn GpuBackend,
        weight: &QuantizedWeight,
        n: u32,
        k: u32,
        allow_alloc: bool,
    ) -> Result<Option<CachedWeightScale>> {
        static CACHE: OnceLock<Mutex<HashMap<(usize, u64, u32, u32), CachedWeightScale>>> =
            OnceLock::new();
        let gpu_id = std::ptr::from_ref(gpu) as *const () as usize;
        let key = (gpu_id, weight.weight_scale.0, n, k);
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(&entry) = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("multi-seq NVFP4 scale cache lock poisoned"))?
            .get(&key)
        {
            return Ok(Some(entry));
        }
        if !allow_alloc {
            return Ok(None);
        }

        let k_blocks = k / NVFP4_BLOCK_SIZE;
        let logical_len = (n as usize)
            .checked_mul(k_blocks as usize)
            .context("multi-seq NVFP4 weight scale length overflow")?;
        let mut logical = vec![0u8; logical_len];
        gpu.copy_d2h(weight.weight_scale, &mut logical)
            .context("read NVFP4 weight scales for multi-seq 128x4 swizzle")?;
        let swizzled = swizzle_scales_host(&logical, n, k)?;
        let scales_128x4 = gpu
            .alloc(swizzled.len())
            .context("alloc multi-seq NVFP4 128x4 weight scales")?;
        if let Err(err) = gpu.copy_h2d(&swizzled, scales_128x4) {
            let _ = gpu.free(scales_128x4);
            return Err(err).context("upload multi-seq NVFP4 128x4 weight scales");
        }
        let input_scale = if !weight.input_scale.is_null() {
            let mut bytes = [0u8; 4];
            gpu.copy_d2h(weight.input_scale, &mut bytes)
                .ok()
                .map(|()| f32::from_le_bytes(bytes))
                .filter(|v| v.is_finite() && *v > 0.0)
        } else {
            None
        };
        let entry = CachedWeightScale {
            scales_128x4,
            input_scale,
        };
        let mut guard = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("multi-seq NVFP4 scale cache lock poisoned"))?;
        if let Some(&existing) = guard.get(&key) {
            let _ = gpu.free(scales_128x4);
            return Ok(Some(existing));
        }
        guard.insert(key, entry);
        Ok(Some(entry))
    }

    fn scratch_lock() -> &'static Mutex<HashMap<usize, SharedScratch>> {
        static SCRATCH: OnceLock<Mutex<HashMap<usize, SharedScratch>>> = OnceLock::new();
        SCRATCH.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn ensure_scratch(
        guard: &mut HashMap<usize, SharedScratch>,
        gpu: &dyn GpuBackend,
        m: usize,
        k: usize,
        stream: u64,
        allow_alloc: bool,
    ) -> Result<Option<(DevicePtr, DevicePtr, DevicePtr)>> {
        let gpu_id = std::ptr::from_ref(gpu) as *const () as usize;
        let cap_m = m.div_ceil(128) * 128;
        let cap_k = k.div_ceil(64) * 64;
        let needs_realloc = match guard.get(&gpu_id) {
            Some(existing) => existing.cap_m < cap_m || existing.cap_k < cap_k,
            None => true,
        };
        if let Some(existing) = guard.get_mut(&gpu_id)
            && let Some(prev_stream) = existing.bound_stream
            && prev_stream != stream
        {
            if !allow_alloc {
                return Ok(None);
            }
            gpu.synchronize(prev_stream)?;
            existing.bound_stream = Some(stream);
        }
        if needs_realloc {
            if !allow_alloc {
                return Ok(None);
            }
            let target_m = match guard.get(&gpu_id) {
                Some(existing) => existing.cap_m.max(cap_m),
                None => cap_m,
            };
            let target_k = match guard.get(&gpu_id) {
                Some(existing) => existing.cap_k.max(cap_k),
                None => cap_k,
            };
            if let Some(prev) = guard.remove(&gpu_id) {
                gpu.synchronize_device()?;
                let _ = gpu.free(prev.a_packed);
                let _ = gpu.free(prev.a_scale);
                let _ = gpu.free(prev.dynamic_scalars);
            }
            let packed_bytes = target_m * (target_k / 2);
            let scale_bytes =
                block_scale_padded_bytes(target_m as u32, target_k as u32, NVFP4_BLOCK_SIZE);
            let a_packed = gpu
                .alloc(packed_bytes)
                .context("alloc multi-seq NVFP4 a_packed scratch")?;
            let a_scale = match gpu.alloc(scale_bytes) {
                Ok(ptr) => ptr,
                Err(err) => {
                    let _ = gpu.free(a_packed);
                    return Err(err).context("alloc multi-seq NVFP4 a_scale scratch");
                }
            };
            let dynamic_scalars = match gpu.alloc(DYNAMIC_SCALARS_BYTES) {
                Ok(ptr) => ptr,
                Err(err) => {
                    let _ = gpu.free(a_packed);
                    let _ = gpu.free(a_scale);
                    return Err(err).context("alloc multi-seq NVFP4 dynamic scalars");
                }
            };
            guard.insert(
                gpu_id,
                SharedScratch {
                    a_packed,
                    a_scale,
                    dynamic_scalars,
                    cap_m: target_m,
                    cap_k: target_k,
                    bound_stream: Some(stream),
                },
            );
            return Ok(Some((a_packed, a_scale, dynamic_scalars)));
        }
        let s = guard
            .get(&gpu_id)
            .ok_or_else(|| anyhow::anyhow!("multi-seq NVFP4 scratch missing"))?;
        Ok(Some((s.a_packed, s.a_scale, s.dynamic_scalars)))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "forwards activation quantize handles, scratch buffers, and tensor dimensions"
    )]
    fn quantize_act_zero_sync(
        gpu: &dyn GpuBackend,
        ks: QuantKernels,
        act_bf16: DevicePtr,
        a_packed: DevicePtr,
        a_scale: DevicePtr,
        dynamic_scalars: DevicePtr,
        m: u32,
        k: u32,
        static_scale: Option<f32>,
        stream: u64,
    ) -> Result<InputScale> {
        if let Some(scale2) = static_scale {
            ops::quantize_bf16_to_nvfp4_atlas_128x4(
                gpu,
                ks.quant_static,
                act_bf16,
                a_packed,
                a_scale,
                scale2,
                m,
                k,
                stream,
            )?;
            return Ok(InputScale::Static(scale2));
        }

        let total_elements = m
            .checked_mul(k)
            .context("multi-seq NVFP4 dynamic activation element-count overflow")?;
        let padded_rows = m.div_ceil(128) * 128;
        let cta_cap = atlas_core::device::active_sm_count()
            .saturating_mul(2)
            .max(96);
        let global_max = dynamic_scalars.offset(DYNAMIC_GLOBAL_MAX_OFF);
        let scale2_a = dynamic_scalars.offset(DYNAMIC_SCALE2_OFF);
        let status = dynamic_scalars.offset(DYNAMIC_STATUS_OFF);

        gpu.memset_async(dynamic_scalars, 0, DYNAMIC_ZERO_PREFIX_BYTES, stream)?;
        ops::nvfp4_global_absmax(gpu, ks.absmax, act_bf16, global_max, total_elements, stream)?;
        KernelLaunch::new(gpu, ks.from_absmax)
            .grid([padded_rows.min(cta_cap).max(1), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(act_bf16)
            .arg_ptr(a_packed)
            .arg_ptr(a_scale)
            .arg_ptr(global_max)
            .arg_ptr(scale2_a)
            .arg_ptr(status)
            .arg_u32(m)
            .arg_u32(k)
            .launch(stream)?;
        Ok(InputScale::DynamicDevice {
            scale2_a_f32: scale2_a,
            status_u32: status,
            alpha_slots: [
                dynamic_scalars.offset(DYNAMIC_ALPHA0_OFF),
                dynamic_scalars.offset(DYNAMIC_ALPHA1_OFF),
                dynamic_scalars.offset(DYNAMIC_ALPHA2_OFF),
            ],
        })
    }

    fn resolve_scalars(
        gpu: &dyn GpuBackend,
        combined_alpha_k: KernelHandle,
        input_scale: InputScale,
        weight_scale_2: f32,
        slot_idx: usize,
        stream: u64,
    ) -> Result<BlockScaleScalars> {
        match input_scale {
            InputScale::Static(a_scale2) => {
                Ok(BlockScaleScalars::host(a_scale2 * weight_scale_2, 0.0))
            }
            InputScale::DynamicDevice {
                scale2_a_f32,
                status_u32,
                alpha_slots,
            } => {
                let alpha_dst = alpha_slots[slot_idx];
                KernelLaunch::new(gpu, combined_alpha_k)
                    .grid([1, 1, 1])
                    .block([1, 1, 1])
                    .arg_ptr(scale2_a_f32)
                    .arg_f32(weight_scale_2)
                    .arg_ptr(status_u32)
                    .arg_ptr(alpha_dst)
                    .launch(stream)?;
                Ok(BlockScaleScalars::device_alpha(alpha_dst.0))
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "multi-seq cuBLASLt NVFP4 projection dispatcher"
    )]
    pub(super) fn try_single(
        gpu: &dyn GpuBackend,
        act_bf16: DevicePtr,
        weight: &QuantizedWeight,
        out_bf16: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ldc: u32,
        graph_capture: bool,
        label: &'static str,
        stream: u64,
    ) -> Result<bool> {
        if !super::multiseq_cublaslt_nvfp4_enabled(gpu.compute_capability())
            || !shape_ok(m, n, k, ldc)
            || !weight_ready(weight)
        {
            return Ok(false);
        }
        let ks = resolve_kernels(gpu);
        let Some(cached) = cached_weight_scale(gpu, weight, n, k, !graph_capture)? else {
            return Ok(false);
        };
        let can_static = cached.input_scale.is_some() && ks.quant_static.0 != 0;
        let can_dynamic = ks.absmax.0 != 0 && ks.from_absmax.0 != 0 && ks.combined_alpha.0 != 0;
        if !can_static && !can_dynamic {
            return Ok(false);
        }
        let mut scratch_guard = scratch_lock()
            .lock()
            .map_err(|_| anyhow::anyhow!("multi-seq NVFP4 scratch lock poisoned"))?;
        let Some((a_packed, a_scale, dynamic_scalars)) = ensure_scratch(
            &mut scratch_guard,
            gpu,
            m as usize,
            k as usize,
            stream,
            !graph_capture,
        )?
        else {
            return Ok(false);
        };
        let input_scale = quantize_act_zero_sync(
            gpu,
            ks,
            act_bf16,
            a_packed,
            a_scale,
            dynamic_scalars,
            m,
            k,
            if can_static { cached.input_scale } else { None },
            stream,
        )?;
        let scalars = resolve_scalars(
            gpu,
            ks.combined_alpha,
            input_scale,
            weight.weight_scale_2,
            0,
            stream,
        )?;
        nvfp4_gemm_block_scaled(
            a_packed.0,
            a_scale.0,
            weight.weight.0,
            cached.scales_128x4.0,
            out_bf16.0,
            m,
            n,
            k,
            ldc,
            scalars,
            stream,
        )?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                label,
                m,
                n,
                k,
                ldc,
                static_scale = can_static,
                "ENGAGED multi-seq cuBLASLt NVFP4 projection"
            );
        });
        Ok(true)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "shared-quantization Q/K/V cuBLASLt NVFP4 multi-seq dispatcher"
    )]
    pub(super) fn try_qkv(
        gpu: &dyn GpuBackend,
        act_bf16: DevicePtr,
        q_weight: &QuantizedWeight,
        k_weight: &QuantizedWeight,
        v_weight: &QuantizedWeight,
        q_out: DevicePtr,
        k_out: DevicePtr,
        v_out: DevicePtr,
        m: u32,
        q_n: u32,
        kv_n: u32,
        k: u32,
        ldc: u32,
        graph_capture: bool,
        stream: u64,
    ) -> Result<bool> {
        if !super::multiseq_cublaslt_nvfp4_enabled(gpu.compute_capability())
            || !shape_ok(m, q_n, k, ldc)
            || !shape_ok(m, kv_n, k, ldc)
            || !weight_ready(q_weight)
            || !weight_ready(k_weight)
            || !weight_ready(v_weight)
        {
            return Ok(false);
        }
        let ks = resolve_kernels(gpu);
        let (Some(q_cached), Some(k_cached), Some(v_cached)) = (
            cached_weight_scale(gpu, q_weight, q_n, k, !graph_capture)?,
            cached_weight_scale(gpu, k_weight, kv_n, k, !graph_capture)?,
            cached_weight_scale(gpu, v_weight, kv_n, k, !graph_capture)?,
        ) else {
            return Ok(false);
        };
        let shared_static_scale = match (
            q_cached.input_scale,
            k_cached.input_scale,
            v_cached.input_scale,
        ) {
            (Some(qs), Some(ks_s), Some(vs)) if ks.quant_static.0 != 0 => {
                Some(qs.max(ks_s).max(vs))
            }
            _ => None,
        };
        let can_dynamic = ks.absmax.0 != 0 && ks.from_absmax.0 != 0 && ks.combined_alpha.0 != 0;
        if shared_static_scale.is_none() && !can_dynamic {
            return Ok(false);
        }
        let mut scratch_guard = scratch_lock()
            .lock()
            .map_err(|_| anyhow::anyhow!("multi-seq NVFP4 scratch lock poisoned"))?;
        let Some((a_packed, a_scale, dynamic_scalars)) = ensure_scratch(
            &mut scratch_guard,
            gpu,
            m as usize,
            k as usize,
            stream,
            !graph_capture,
        )?
        else {
            return Ok(false);
        };
        let input_scale = quantize_act_zero_sync(
            gpu,
            ks,
            act_bf16,
            a_packed,
            a_scale,
            dynamic_scalars,
            m,
            k,
            shared_static_scale,
            stream,
        )?;

        for (slot_idx, (weight, scales_128x4, out_ptr, n_dim)) in [
            (q_weight, q_cached.scales_128x4, q_out, q_n),
            (k_weight, k_cached.scales_128x4, k_out, kv_n),
            (v_weight, v_cached.scales_128x4, v_out, kv_n),
        ]
        .into_iter()
        .enumerate()
        {
            let scalars = resolve_scalars(
                gpu,
                ks.combined_alpha,
                input_scale,
                weight.weight_scale_2,
                slot_idx,
                stream,
            )?;
            nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                weight.weight.0,
                scales_128x4.0,
                out_ptr.0,
                m,
                n_dim,
                k,
                ldc,
                scalars,
                stream,
            )?;
        }
        static LOGGED_QKV: std::sync::Once = std::sync::Once::new();
        LOGGED_QKV.call_once(|| {
            tracing::info!(
                m,
                q_n,
                kv_n,
                k,
                ldc,
                static_scale = shared_static_scale.is_some(),
                "ENGAGED multi-seq cuBLASLt NVFP4 shared-quant Attention Q/K/V"
            );
        });
        Ok(true)
    }
}

/// Route a multi-seq NVFP4 projection (`ssm_qkvz`, `ssm_out_proj`, `attn_o_proj`,
/// or `lm_head`) through cuBLASLt native block-scaled NVFP4 (`VEC16_UE4M3`) with
/// cached `128x4` weight scales and zero-sync activation quantization.
///
/// Safe under CUDA graph capture: when `graph_capture` is `true`, returns
/// `Ok(false)` rather than allocating or synchronizing if the weight scales or
/// activation scratch were not already warmed outside capture.
#[expect(
    clippy::too_many_arguments,
    reason = "multi-seq cuBLASLt NVFP4 projection dispatcher"
)]
pub fn try_cublaslt_nvfp4_multiseq_single(
    gpu: &dyn GpuBackend,
    act_bf16: DevicePtr,
    weight: &QuantizedWeight,
    out_bf16: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    graph_capture: bool,
    label: &'static str,
    stream: u64,
) -> Result<bool> {
    #[cfg(feature = "cuda")]
    {
        multiseq_cublaslt_nvfp4_cuda::try_single(
            gpu,
            act_bf16,
            weight,
            out_bf16,
            m,
            n,
            k,
            ldc,
            graph_capture,
            label,
            stream,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            gpu,
            act_bf16,
            weight,
            out_bf16,
            m,
            n,
            k,
            ldc,
            graph_capture,
            label,
            stream,
        );
        Ok(false)
    }
}

/// Route multi-seq Attention `q/k/v` projections through cuBLASLt NVFP4 with a
/// single shared activation quantization and direct strided writes (`ldc`) into
/// `qkv_buf` (eliminating all per-row D2D scatter copies).
#[expect(
    clippy::too_many_arguments,
    reason = "shared-quantization Q/K/V cuBLASLt NVFP4 multi-seq dispatcher"
)]
pub fn try_cublaslt_nvfp4_multiseq_qkv(
    gpu: &dyn GpuBackend,
    act_bf16: DevicePtr,
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    q_out: DevicePtr,
    k_out: DevicePtr,
    v_out: DevicePtr,
    m: u32,
    q_n: u32,
    kv_n: u32,
    k: u32,
    ldc: u32,
    graph_capture: bool,
    stream: u64,
) -> Result<bool> {
    #[cfg(feature = "cuda")]
    {
        multiseq_cublaslt_nvfp4_cuda::try_qkv(
            gpu,
            act_bf16,
            q_weight,
            k_weight,
            v_weight,
            q_out,
            k_out,
            v_out,
            m,
            q_n,
            kv_n,
            k,
            ldc,
            graph_capture,
            stream,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            gpu,
            act_bf16,
            q_weight,
            k_weight,
            v_weight,
            q_out,
            k_out,
            v_out,
            m,
            q_n,
            kv_n,
            k,
            ldc,
            graph_capture,
            stream,
        );
        Ok(false)
    }
}

#[cfg(test)]
mod multiseq_cublaslt_nvfp4_policy_tests {
    use super::{
        MultiseqCublasLtNvfp4Policy, parse_multiseq_cublaslt_nvfp4_policy,
        resolve_multiseq_cublaslt_nvfp4,
    };

    #[test]
    fn multiseq_cublaslt_nvfp4_auto_enables_only_on_sm120_and_sm103_above_64_sms() {
        assert!(!resolve_multiseq_cublaslt_nvfp4(
            MultiseqCublasLtNvfp4Policy::Auto,
            MultiseqCublasLtNvfp4Policy::Auto,
            48,
            Some((12, 1)),
        ));
        assert!(!resolve_multiseq_cublaslt_nvfp4(
            MultiseqCublasLtNvfp4Policy::Auto,
            MultiseqCublasLtNvfp4Policy::Auto,
            64,
            Some((12, 0)),
        ));
        assert!(resolve_multiseq_cublaslt_nvfp4(
            MultiseqCublasLtNvfp4Policy::Auto,
            MultiseqCublasLtNvfp4Policy::Auto,
            188,
            Some((12, 0)),
        ));
        assert!(resolve_multiseq_cublaslt_nvfp4(
            MultiseqCublasLtNvfp4Policy::Auto,
            MultiseqCublasLtNvfp4Policy::Auto,
            148,
            Some((10, 3)),
        ));
        assert!(!resolve_multiseq_cublaslt_nvfp4(
            MultiseqCublasLtNvfp4Policy::Disabled,
            MultiseqCublasLtNvfp4Policy::Enabled,
            188,
            Some((12, 0)),
        ));
        assert!(resolve_multiseq_cublaslt_nvfp4(
            MultiseqCublasLtNvfp4Policy::Enabled,
            MultiseqCublasLtNvfp4Policy::Disabled,
            48,
            Some((12, 1)),
        ));
        assert_eq!(
            parse_multiseq_cublaslt_nvfp4_policy(Some("0")),
            MultiseqCublasLtNvfp4Policy::Disabled
        );
        assert_eq!(
            parse_multiseq_cublaslt_nvfp4_policy(Some("1")),
            MultiseqCublasLtNvfp4Policy::Enabled
        );
        assert_eq!(
            parse_multiseq_cublaslt_nvfp4_policy(None),
            MultiseqCublasLtNvfp4Policy::Auto
        );
    }
}

// ── Position embeddings ────────────────────────────────────────────
