// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

// ── Normalization ──────────────────────────────────────────────────

/// RMS normalization: output = rms_norm(input) * weight.
///
/// Kernel: `rms_norm(input, weight, output, hidden_size, eps)`
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
pub fn rms_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    crate::layers::dense_ffn::invalidate_precomputed_proj_absmax(gpu, output);
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// Fused Qwen3 `(1 + w)` RMSNorm + global FP32 absmax (without writing a dead
/// residual copy). `global_max` must be zero-initialized before launch.
///
/// Kernel: `rms_norm_absmax(input, weight, output, global_max, hidden_size, eps)`
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
#[expect(
    clippy::too_many_arguments,
    reason = "CUDA kernel launcher matching rms_norm_absmax ABI"
)]
pub fn rms_norm_absmax(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    global_max: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(global_max)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// Fused RMS norm + residual save + global FP32 absmax.
///
/// Kernel: `rms_norm_residual_absmax(input, weight, output, residual, global_max, hidden_size, eps)`
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
#[expect(
    clippy::too_many_arguments,
    reason = "CUDA kernel launcher matching rms_norm_residual_absmax ABI"
)]
pub fn rms_norm_residual_absmax(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    residual: DevicePtr,
    global_max: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(residual)
        .arg_ptr(global_max)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// Fused RMS norm + residual save: normed = rms_norm(input), residual = input.
///
/// Eliminates a separate D2D copy by writing the raw input to the residual
/// buffer in the same pass as the normalized output write. On wide discrete
/// GPUs (`sm_count > 64`) during prefill (`num_tokens >= 64`), upgrades plain
/// BF16 `rms_norm_residual` to `rms_norm_residual_absmax` so the subsequent
/// dynamic NVFP4 projection skips `nvfp4_global_absmax`.
///
/// Kernel: `rms_norm_residual(input, weight, output, residual, hidden_size, eps)`
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
pub fn rms_norm_residual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    residual: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    if num_tokens >= 64
        && crate::layers::dense_ffn::prefill_fuse_absmax_enabled(gpu)
        && !gpu.stream_is_capturing(stream)
    {
        let plain_k = crate::layers::try_kernel(gpu, "norm", "rms_norm_residual");
        let absmax_k = crate::layers::try_kernel(gpu, "norm", "rms_norm_residual_absmax");
        if kernel.0 == plain_k.0
            && plain_k.0 != 0
            && absmax_k.0 != 0
            && let Some(global_max) = crate::layers::dense_ffn::prepare_precomputed_proj_absmax(
                gpu,
                output,
                num_tokens,
                hidden_size,
                stream,
            )?
        {
            return rms_norm_residual_absmax(
                gpu,
                absmax_k,
                input,
                weight,
                output,
                residual,
                global_max,
                num_tokens,
                hidden_size,
                eps,
                stream,
            );
        }
    }
    crate::layers::dense_ffn::invalidate_precomputed_proj_absmax(gpu, output);
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(residual)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// Fused residual add + RMS norm + residual save + global FP32 absmax.
///
/// Kernel: `residual_add_rms_norm_absmax(hidden, src, weight, output, residual, global_max, hidden_size, eps)`
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
#[expect(
    clippy::too_many_arguments,
    reason = "CUDA kernel launcher matching residual_add_rms_norm_absmax ABI"
)]
pub fn residual_add_rms_norm_absmax(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    src: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    residual: DevicePtr,
    global_max: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(src)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(residual)
        .arg_ptr(global_max)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// Fused residual add + RMS norm + residual save.
///
/// `hidden[i] += src[i]; normed = rms_norm(hidden) * (1+weight); residual = hidden`.
/// Eliminates one kernel launch per fusion site (48 per decode step).
///
/// Kernel: `residual_add_rms_norm(hidden, src, weight, output, residual, hidden_size, eps)`
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn residual_add_rms_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    src: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    residual: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    crate::layers::dense_ffn::invalidate_precomputed_proj_absmax(gpu, output);
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(src)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(residual)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// Batched per-head RMS norm for the K=3 verify path: 3 q_norms + 3 k_norms
/// in a single launch, delegating to [`rms_norm_qk_batched`] with one block
/// per `(head, token, q|k)`.
///
/// `qkv_base` must point at token 0's Q region; subsequent tokens are at
/// stride `qkv_stride_bf16` BF16 elements. `k_offset_bf16` is the offset
/// (in BF16 elements) from each token's start to its K slab — equal to
/// `q_proj_dim` (which is `2*q_dim` for gated layers because of the Gate
/// chunk preceding K).
///
/// Kernel: `rms_norm_qk_batch3(qkv_base, q_weight, k_weight,
///     qkv_stride, q_dim, k_dim, k_offset, head_dim, eps)`
/// Grid: (max_heads, 3, 2)  Block: (min(head_dim, 1024), 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_qk_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    qkv_base: DevicePtr,
    q_weight: &DenseWeight,
    k_weight: &DenseWeight,
    qkv_stride_bf16: u32,
    q_dim: u32,
    k_dim: u32,
    k_offset_bf16: u32,
    head_dim: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    rms_norm_qk_batched(
        gpu,
        kernel,
        qkv_base,
        q_weight,
        k_weight,
        qkv_stride_bf16,
        q_dim,
        k_dim,
        k_offset_bf16,
        head_dim,
        3,
        eps,
        stream,
    )
}

/// Generalised strided per-head q_norm + k_norm over `num_tokens` tokens in a
/// single launch. Reuses the `rms_norm_qk_batch3` kernel body verbatim (it
/// already indexes the token via `blockIdx.y` with no hardcoded count), so the
/// per-head reduction is bit-identical to both the batch3 path and the
/// contiguous per-token `rms_norm` calls it replaces.
///
/// Grid: (max_heads, num_tokens, 2)  Block: (min(head_dim, 1024), 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_qk_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    qkv_base: DevicePtr,
    q_weight: &DenseWeight,
    k_weight: &DenseWeight,
    qkv_stride_bf16: u32,
    q_dim: u32,
    k_dim: u32,
    k_offset_bf16: u32,
    head_dim: u32,
    num_tokens: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    debug_assert!(head_dim > 0, "head_dim must be > 0");
    debug_assert!(q_dim.is_multiple_of(head_dim) && k_dim.is_multiple_of(head_dim));
    let nq_heads = q_dim / head_dim;
    let nkv_heads = k_dim / head_dim;
    let max_heads = nq_heads.max(nkv_heads);
    KernelLaunch::new(gpu, kernel)
        .grid([max_heads, num_tokens, 2])
        .block([head_dim.min(1024), 1, 1])
        .arg_ptr(qkv_base)
        .arg_ptr(q_weight.weight)
        .arg_ptr(k_weight.weight)
        .arg_u32(qkv_stride_bf16)
        .arg_u32(q_dim)
        .arg_u32(k_dim)
        .arg_u32(k_offset_bf16)
        .arg_u32(head_dim)
        .arg_f32(eps)
        .launch(stream)
}

/// Gated RMS norm (norm_before_gate=False, per-group):
///   output = rms_norm_per_group(input * silu(gate), weight, group_size)
///
/// Kernel: `gated_rms_norm(input, gate, weight, output, hidden_size, eps, gate_stride, group_size)`
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
pub fn gated_rms_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    gate_stride: u32,
    eps: f32,
    group_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .arg_u32(gate_stride)
        .arg_u32(group_size)
        .launch(stream)
}

/// Multi-sequence gated RMS norm with FP32 input, PER-HEAD norm, and
/// per-seq strides.
///
/// Mirrors the single-seq `gated_rms_norm_f32_input` semantics (one CTA
/// per (seq, head) pair, norm computed over `head_dim` per head), but
/// the per-seq row strides for input/gate/output are parameterised so
/// the multi-seq decode buffer layout stays in bounds. `input_stride`
/// is FP32 elements, `gate_stride` and `output_stride` are BF16
/// elements.
///
/// Kernel: `gated_rms_norm_f32_multi_seq(input, gate, weight, output,
///   head_dim, eps, input_stride_fp32, gate_stride_bf16,
///   output_stride_bf16)`
/// Grid: (num_v_heads, num_seqs, 1)  Block: (head_dim, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gated_rms_norm_f32_multi_seq(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    num_v_heads: u32,
    num_seqs: u32,
    head_dim: u32,
    eps: f32,
    input_stride: u32,
    gate_stride: u32,
    output_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, num_seqs, 1])
        .block([head_dim.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(head_dim)
        .arg_f32(eps)
        .arg_u32(input_stride)
        .arg_u32(gate_stride)
        .arg_u32(output_stride)
        .launch(stream)
}

/// Batched gated RMS norm + global FP32 absmax for prefill.
///
/// Grid: (heads_per_token, num_actual_tokens, 1)
/// Block: (min(head_dim, 1024), 1, 1)
#[expect(
    clippy::too_many_arguments,
    reason = "CUDA kernel launcher matching gated_rms_norm_prefill_absmax ABI"
)]
pub fn gated_rms_norm_prefill_absmax(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    global_max: DevicePtr,
    heads_per_token: u32,
    head_dim: u32,
    eps: f32,
    num_actual_tokens: u32,
    input_token_stride: u32,
    gate_token_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([heads_per_token, num_actual_tokens, 1])
        .block([head_dim.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(global_max)
        .arg_u32(head_dim)
        .arg_f32(eps)
        .arg_u32(input_token_stride)
        .arg_u32(gate_token_stride)
        .launch(stream)
}

/// Batched gated RMS norm for prefill: all (head, actual_token) pairs in one launch.
///
/// On wide discrete GPUs (`sm_count > 64`) for contiguous prefill tensors
/// (`num_actual_tokens >= 64` and `input_token_stride == heads_per_token * head_dim`),
/// upgrades to `gated_rms_norm_prefill_absmax` so the subsequent dynamic NVFP4
/// `out_proj` skips `nvfp4_global_absmax`.
///
/// Grid: (heads_per_token, num_actual_tokens, 1)
/// Block: (min(head_dim, 1024), 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gated_rms_norm_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    heads_per_token: u32,
    head_dim: u32,
    eps: f32,
    num_actual_tokens: u32,
    input_token_stride: u32,
    gate_token_stride: u32,
    stream: u64,
) -> Result<()> {
    let contiguous_width = heads_per_token.saturating_mul(head_dim);
    if num_actual_tokens >= 64
        && input_token_stride == contiguous_width
        && crate::layers::dense_ffn::prefill_fuse_absmax_enabled(gpu)
        && !gpu.stream_is_capturing(stream)
    {
        let plain_k = crate::layers::try_kernel(gpu, "norm", "gated_rms_norm_prefill");
        let absmax_k = crate::layers::try_kernel(gpu, "norm", "gated_rms_norm_prefill_absmax");
        if kernel.0 == plain_k.0
            && plain_k.0 != 0
            && absmax_k.0 != 0
            && let Some(global_max) = crate::layers::dense_ffn::prepare_precomputed_proj_absmax(
                gpu,
                output,
                num_actual_tokens,
                contiguous_width,
                stream,
            )?
        {
            return gated_rms_norm_prefill_absmax(
                gpu,
                absmax_k,
                input,
                gate,
                weight,
                output,
                global_max,
                heads_per_token,
                head_dim,
                eps,
                num_actual_tokens,
                input_token_stride,
                gate_token_stride,
                stream,
            );
        }
    }
    crate::layers::dense_ffn::invalidate_precomputed_proj_absmax(gpu, output);
    KernelLaunch::new(gpu, kernel)
        .grid([heads_per_token, num_actual_tokens, 1])
        .block([head_dim.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(head_dim)
        .arg_f32(eps)
        .arg_u32(input_token_stride)
        .arg_u32(gate_token_stride)
        .launch(stream)
}

// ── GEMM ───────────────────────────────────────────────────────────
