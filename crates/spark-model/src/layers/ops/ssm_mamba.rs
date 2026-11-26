// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Causal conv1d update (decode step, supports batched sequences).
///
/// Kernel: `causal_conv1d_update(conv_state, new_input, weight, bias,
///          output, batch, dim, d_conv)`
/// Grid: (ceil(dim/256), batch, 1)  Block: (256, 1, 1)
///
/// For batch > 1, conv_state and input must be contiguous [batch, ...].
pub fn conv1d_update(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    batch_size: u32,
    stream: u64,
) -> Result<()> {
    let bias_ptr = DevicePtr::NULL;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), batch_size, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias_ptr)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .launch(stream)
}

/// Fused conv1d update + SiLU + L2 normalization for Q/K channels.
///
/// Combines `causal_conv1d_update` and `l2_norm_bf16` into a single kernel.
/// Q+K channels (0..qk_channels) get L2-normalized per head after SiLU.
/// V channels (qk_channels..d_inner) get SiLU only.
///
/// Saves 1 kernel launch per SSM layer (36 launches/step for 35B/80B).
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_l2norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    batch_size: u32,
    qk_channels: u32,
    head_dim: u32,
    l2_eps: f32,
    stream: u64,
) -> Result<()> {
    let bias_ptr = DevicePtr::NULL;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), batch_size, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias_ptr)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_f32(l2_eps)
        .launch(stream)
}

/// Multi-sequence conv1d update — runs the decode-step state advance
/// for `num_seqs` independent sequences in a SINGLE launch.
///
/// Kernel: `causal_conv1d_update_multi_seq(conv_state_ptrs, new_input,
///          weight, bias, output, num_seqs, dim, d_conv, input_stride,
///          output_stride)`
/// Grid: (ceil(dim/256), num_seqs, 1)  Block: (256, 1, 1)
///
/// `conv_state_ptrs` is a device-resident `[num_seqs]` array of
/// `float*` per-seq state pointers (the caller uploads it before the
/// launch). `input_stride` / `output_stride` are in BF16 ELEMENTS
/// between successive sequences (allows passing strided slabs of
/// wider per-seq tensors).
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_multi_seq(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state_ptrs: DevicePtr, // device array of num_seqs × u64 ptrs
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    num_seqs: u32,
    input_stride: u32,
    output_stride: u32,
    stream: u64,
) -> Result<()> {
    let bias_ptr = DevicePtr::NULL;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), num_seqs, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state_ptrs)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias_ptr)
        .arg_ptr(output)
        .arg_u32(num_seqs)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .launch(stream)
}

/// Multi-sequence fused conv1d + SiLU + L2-norm for Q/K channels.
///
/// Same semantics as `conv1d_update_l2norm` but for c sequences in one
/// launch. State pointers are passed as a device array. Strides are
/// in BF16 ELEMENTS between successive sequences.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_l2norm_multi_seq(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state_ptrs: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    num_seqs: u32,
    qk_channels: u32,
    head_dim: u32,
    l2_eps: f32,
    input_stride: u32,
    output_stride: u32,
    stream: u64,
) -> Result<()> {
    let bias_ptr = DevicePtr::NULL;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), num_seqs, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state_ptrs)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias_ptr)
        .arg_ptr(output)
        .arg_u32(num_seqs)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_f32(l2_eps)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .launch(stream)
}

/// FP32-output multi-sequence fused conv1d + SiLU + L2-norm.
///
/// Same as `conv1d_update_l2norm_multi_seq` but writes FP32 instead of
/// BF16. Required for the production AEON-Q36-27B decode path where the
/// downstream gdn_decode_f32 reads FP32 q/k/v.
///
/// `output_stride` is in FP32 ELEMENTS between successive sequences.
///
/// Kernel: `causal_conv1d_update_l2norm_f32_multi_seq(conv_state_ptrs,
///          new_input, weight, bias, output, num_seqs, dim, d_conv,
///          qk_channels, head_dim, l2_eps, input_stride, output_stride)`
/// Grid: (ceil(dim/256), num_seqs, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_l2norm_f32_multi_seq(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state_ptrs: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    num_seqs: u32,
    qk_channels: u32,
    head_dim: u32,
    l2_eps: f32,
    input_stride: u32,
    output_stride: u32,
    stream: u64,
) -> Result<()> {
    let bias_ptr = DevicePtr::NULL;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), num_seqs, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state_ptrs)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias_ptr)
        .arg_ptr(output)
        .arg_u32(num_seqs)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_f32(l2_eps)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .launch(stream)
}

/// Select CTA block size for `causal_conv1d_update_l2norm_f32_sequence`.
///
/// The kernel's Q/K L2-norm reduction requires every `head_dim`-wide head to
/// sit within one block (`head_dim = 128 -> 4 warps = 128 threads`). On wide
/// discrete GPUs (`sm_count > 64`) when `head_dim == 128` and `qk_channels` is
/// 128-aligned, halving the block size from `256` to `128` doubles CTA count
/// (`40 -> 80` CTAs at `dim = 10240`) with 0-ULP bit-exact arithmetic. GB10
/// (`sm_count <= 64`) keeps `block = 256` unchanged.
#[must_use]
const fn conv1d_f32_sequence_block_size(sm_count: u32, head_dim: u32, qk_channels: u32) -> u32 {
    if sm_count > 64 && head_dim == 128 && qk_channels.is_multiple_of(128) {
        128
    } else {
        256
    }
}

/// Exact flat-chain FP32 conv recurrence for a multi-row verify window.
///
/// Rows advance sequentially inside each CTA and `state_inter` receives one
/// full post-row conv-state snapshot per row. Strides are element counts.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_l2norm_f32_sequence(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    state_inter: DevicePtr,
    rows: u32,
    dim: u32,
    d_conv: u32,
    qk_channels: u32,
    head_dim: u32,
    l2_eps: f32,
    input_stride: u32,
    output_stride: u32,
    state_stride: u32,
    stream: u64,
) -> Result<()> {
    let block_x = conv1d_f32_sequence_block_size(
        atlas_core::device::active_sm_count(),
        head_dim,
        qk_channels,
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(dim, block_x), 1, 1])
        .block([block_x, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(output)
        .arg_ptr(state_inter)
        .arg_u32(rows)
        .arg_u32(dim)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_f32(l2_eps)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .arg_u32(state_stride)
        .launch(stream)
}

/// Wave 2: Fused multi-token BF16 conv1d + SiLU + Q/K L2-norm + state snapshots.
///
/// Replaces the 17-iteration host loop of `conv1d_update_l2norm` + `copy_d2d_async`
/// in `trait_decode_batched_conv_gdn.rs` (`wy17` and `K>=5` chunked paths) with
/// a single launch per GDN layer.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_l2norm_bf16_sequence(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    state_inter: DevicePtr,
    rows: u32,
    dim: u32,
    d_conv: u32,
    qk_channels: u32,
    head_dim: u32,
    l2_eps: f32,
    input_stride: u32,
    output_stride: u32,
    state_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(dim, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(output)
        .arg_ptr(state_inter)
        .arg_u32(rows)
        .arg_u32(dim)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_f32(l2_eps)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .arg_u32(state_stride)
        .launch(stream)
}

/// Wave 2: Tree-aware fused multi-token BF16 conv1d + SiLU + Q/K L2-norm + state snapshots.
///
/// Replaces the 17-iteration host loop of `conv1d_tree_reroot` +
/// `conv1d_update_l2norm` + `copy_d2d_async` in the DDTree verify path with a
/// single launch per GDN layer. Pass `DevicePtr::NULL` for `parent_ids` when
/// tree re-rooting is not requested.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_l2norm_bf16_tree_sequence(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    parent_ids: DevicePtr,
    output: DevicePtr,
    state_inter: DevicePtr,
    rows: u32,
    dim: u32,
    d_conv: u32,
    qk_channels: u32,
    head_dim: u32,
    l2_eps: f32,
    input_stride: u32,
    output_stride: u32,
    state_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(dim, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(parent_ids)
        .arg_ptr(output)
        .arg_ptr(state_inter)
        .arg_u32(rows)
        .arg_u32(dim)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_f32(l2_eps)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .arg_u32(state_stride)
        .launch(stream)
}

/// K=3 fused conv1d-update + SiLU + L2 norm + intermediate-state save.
///
/// Replaces the K=3 verify per-token loop (3 × `conv1d_update_l2norm` +
/// 3 × d2d copy of conv_state) with a single launch. State stays in
/// registers across the 3 sequential updates; intermediates after
/// tokens 0 and 1 are written to `state_inter_0` / `state_inter_1`,
/// and the final state after token 2 lands in `conv_state` (matching
/// the post-state contract of the per-token kernel). `input_stride` and
/// `output_stride` are BF16 elements between successive tokens so the
/// caller can keep the existing `deinterleaved` layout (qkvz_size
/// stride) and the existing `conv_out_buf` layout (conv_dim stride).
///
/// Kernel: `causal_conv1d_update_l2norm_chunk3(conv_state, new_input,
///          weight, bias, output, state_inter_0, state_inter_1, batch,
///          dim, d_conv, qk_channels, head_dim, l2_eps, input_stride,
///          output_stride)`
/// Grid: (ceil(dim/256), batch, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_l2norm_chunk3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    state_inter_0: DevicePtr,
    state_inter_1: DevicePtr,
    batch: u32,
    dim: u32,
    d_conv: u32,
    qk_channels: u32,
    head_dim: u32,
    l2_eps: f32,
    input_stride: u32,
    output_stride: u32,
    stream: u64,
) -> Result<()> {
    let bias_ptr = DevicePtr::NULL;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(dim, 256), batch, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias_ptr)
        .arg_ptr(output)
        .arg_ptr(state_inter_0)
        .arg_ptr(state_inter_1)
        .arg_u32(batch)
        .arg_u32(dim)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_f32(l2_eps)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .launch(stream)
}

/// Multi-token conv1d sliding window update + SiLU for prefill.
///
/// Processes `seq_len` tokens sequentially per channel in registers.
/// Input/output may be non-contiguous (different strides between tokens).
///
/// Kernel: `causal_conv1d_update_prefill(conv_state, input, weight, bias,
///          output, dim, d_conv, seq_len, input_stride, output_stride)`
/// Grid: (ceil(dim/256), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    bias: DevicePtr,
    output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    seq_len: u32,
    input_stride: u32,
    output_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias)
        .arg_ptr(output)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(seq_len)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .launch(stream)
}

/// Exact [`conv1d_update_prefill`] shadow with projected-Z persistence fused
/// into the same per-channel token loop.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_prefill_zcopy(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    bias: DevicePtr,
    output: DevicePtr,
    z_input: DevicePtr,
    z_output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    seq_len: u32,
    input_stride: u32,
    output_stride: u32,
    z_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias)
        .arg_ptr(output)
        .arg_ptr(z_input)
        .arg_ptr(z_output)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(seq_len)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .arg_u32(z_dim)
        .launch(stream)
}

/// Exact two-phase prefill shadow that additionally folds the standalone Q/K
/// L2-normalization launch into the sequential conv token loop.
///
/// The kernel is deliberately restricted to 128-wide heads. It materializes
/// the conv+SiLU result as BF16 in shared memory and duplicates
/// `l2_norm_bf16`'s packed-pair reduction and BF16 output conversion.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_prefill_l2norm_zcopy(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    bias: DevicePtr,
    output: DevicePtr,
    z_input: DevicePtr,
    z_output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    seq_len: u32,
    input_stride: u32,
    output_stride: u32,
    z_dim: u32,
    qk_channels: u32,
    head_dim: u32,
    l2_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, head_dim), 1, 1])
        .block([head_dim, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias)
        .arg_ptr(output)
        .arg_ptr(z_input)
        .arg_ptr(z_output)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(seq_len)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .arg_u32(z_dim)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_f32(l2_eps)
        .launch(stream)
}

/// Mamba-2 SSM prefill: sequential recurrence across `seq_len` tokens in a single kernel.
///
/// Same algorithm as decode but loops over tokens, avoiding per-token launch overhead.
/// Supports non-contiguous layouts via per-tensor strides (BF16 elements between tokens).
///
/// Grid: (num_heads, batch_size, 1)  Block: (state_size, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssm_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    x: DevicePtr,
    b_proj: DevicePtr,
    c_proj: DevicePtr,
    dt_raw: DevicePtr,
    a_log: DevicePtr,
    d_param: DevicePtr,
    dt_bias: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    state_size: u32,
    n_groups: u32,
    dt_min: f32,
    dt_max: f32,
    x_stride: u32,
    bc_stride: u32,
    dt_stride: u32,
    y_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, batch_size, 1])
        .block([state_size, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(x)
        .arg_ptr(b_proj)
        .arg_ptr(c_proj)
        .arg_ptr(dt_raw)
        .arg_ptr(a_log)
        .arg_ptr(d_param)
        .arg_ptr(dt_bias)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(state_size)
        .arg_u32(n_groups)
        .arg_f32(dt_min)
        .arg_f32(dt_max)
        .arg_u32(x_stride)
        .arg_u32(bc_stride)
        .arg_u32(dt_stride)
        .arg_u32(y_stride)
        .launch(stream)
}

/// Tree-aware conv state re-root for DDTree / FREE_SLOTS verify.
///
/// Before processing token `t` in a non-flat tree, copies
/// `conv_inter_base[parent_ids[t] * conv_floats..]` → `conv_state`
/// when `parent_ids[t] != t - 1`. No-op on the root token (t==0) or
/// on any linear-chain step (parent == t-1).
///
/// This is CUDA-graph-safe: `parent_ids` is a persistent device buffer
/// whose content is updated before each graph replay; the pointer is
/// fixed at capture time. `t` is a launch parameter (graph-baked) and
/// does not change within a replay — each per-token kernel invocation
/// in the sequential loop has a distinct graph-baked `t`.
///
/// Kernel: `causal_conv1d_tree_reroot`
/// Grid:   (ceil(conv_floats / 256), 1, 1)
/// Block:  (256, 1, 1)
pub fn conv1d_tree_reroot(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    inter_base: DevicePtr,
    parent_ids_dev: DevicePtr,
    t: u32,
    conv_floats: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(conv_floats, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(inter_base)
        .arg_ptr(parent_ids_dev)
        .arg_u32(t)
        .arg_u32(conv_floats)
        .launch(stream)
}

/// Persistent Mamba-2 SSM prefill: H in shared memory, reduces global traffic.
/// Same parameters and launch config as mamba2_ssm_prefill.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssm_prefill_persistent(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    x: DevicePtr,
    b_proj: DevicePtr,
    c_proj: DevicePtr,
    dt_raw: DevicePtr,
    a_log: DevicePtr,
    d_param: DevicePtr,
    dt_bias: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    state_size: u32,
    n_groups: u32,
    dt_min: f32,
    dt_max: f32,
    x_stride: u32,
    bc_stride: u32,
    dt_stride: u32,
    y_stride: u32,
    stream: u64,
) -> Result<()> {
    // H_smem + smem_x + smem_warp
    let smem = head_dim * state_size * 4 + head_dim * 4 + 4 * head_dim * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, batch_size, 1])
        .block([state_size, 1, 1])
        .shared_mem(smem)
        .arg_ptr(h_state)
        .arg_ptr(x)
        .arg_ptr(b_proj)
        .arg_ptr(c_proj)
        .arg_ptr(dt_raw)
        .arg_ptr(a_log)
        .arg_ptr(d_param)
        .arg_ptr(dt_bias)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(state_size)
        .arg_u32(n_groups)
        .arg_f32(dt_min)
        .arg_f32(dt_max)
        .arg_u32(x_stride)
        .arg_u32(bc_stride)
        .arg_u32(dt_stride)
        .arg_u32(y_stride)
        .launch(stream)
}

/// Cross-sequence twin of [`conv1d_update_l2norm_f32_sequence`]: grid
/// `(ceil(dim/256), num_seqs)`. Sequence `s` reads input rows
/// `s*rows..(s+1)*rows` of `input` and its own conv state, output rows and
/// snapshots from the `num_seqs`-entry u64 device tables.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_l2norm_f32_sequence_multi(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state_ptrs: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output_ptrs: DevicePtr,
    state_inter_ptrs: DevicePtr,
    num_seqs: u32,
    rows: u32,
    dim: u32,
    d_conv: u32,
    qk_channels: u32,
    head_dim: u32,
    l2_eps: f32,
    input_stride: u32,
    output_stride: u32,
    state_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(dim, 256), num_seqs, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state_ptrs)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(output_ptrs)
        .arg_ptr(state_inter_ptrs)
        .arg_u32(rows)
        .arg_u32(dim)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_f32(l2_eps)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .arg_u32(state_stride)
        .launch(stream)
}

/// Fused 1x1x1-CTA Qwen4 GDN single-token decode (conv1d + SiLU + Q/K L2-norm +
/// register-resident GDN recurrence + per-head sigmoid gated RMS norm).
#[allow(clippy::too_many_arguments)]
pub fn qwen4_gdn_decode_fused_k1(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    deinterleaved: DevicePtr,
    conv_weight: &DenseWeight,
    gate: DevicePtr,
    beta: DevicePtr,
    h_state: DevicePtr,
    rms_weight: &DenseWeight,
    normed_out: DevicePtr,
    group_done: DevicePtr,
    num_k_heads: u32,
    num_v_heads: u32,
    l2_eps: f32,
    rms_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(deinterleaved)
        .arg_ptr(conv_weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(h_state)
        .arg_ptr(rms_weight.weight)
        .arg_ptr(normed_out)
        .arg_ptr(group_done)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_f32(l2_eps)
        .arg_f32(rms_eps)
        .launch(stream)
}

/// Fused 1x1x1-CTA Qwen4 GDN speculative verify (lazy-commit mode: leaves
/// `h_state` untouched at step-initial and writes final H to `h_state_inter[rows - 1]`).
#[allow(clippy::too_many_arguments)]
pub fn qwen4_gdn_verify_fused_lazyfinal(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    deinterleaved: DevicePtr,
    conv_weight: &DenseWeight,
    gate: DevicePtr,
    beta: DevicePtr,
    h_state: DevicePtr,
    rms_weight: &DenseWeight,
    normed_out: DevicePtr,
    retain_qkv: DevicePtr,
    conv_inter: DevicePtr,
    h_state_inter: DevicePtr,
    group_done: DevicePtr,
    rows: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    l2_eps: f32,
    rms_eps: f32,
    input_stride: u32,
    gate_beta_stride: u32,
    normed_stride: u32,
    retain_stride: u32,
    conv_inter_stride: u32,
    h_inter_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(deinterleaved)
        .arg_ptr(conv_weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(h_state)
        .arg_ptr(rms_weight.weight)
        .arg_ptr(normed_out)
        .arg_ptr(retain_qkv)
        .arg_ptr(conv_inter)
        .arg_ptr(h_state_inter)
        .arg_ptr(group_done)
        .arg_u32(rows)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_f32(l2_eps)
        .arg_f32(rms_eps)
        .arg_u32(input_stride)
        .arg_u32(gate_beta_stride)
        .arg_u32(normed_stride)
        .arg_u32(retain_stride)
        .arg_u32(conv_inter_stride)
        .arg_u32(h_inter_stride)
        .launch(stream)
}

/// Fused 1x1x1-CTA Qwen4 GDN speculative verify (persistent snapshot mode:
/// writes every post-row H snapshot to `h_state_inter` and final H to `h_state`).
#[allow(clippy::too_many_arguments)]
pub fn qwen4_gdn_verify_fused_persistent(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    deinterleaved: DevicePtr,
    conv_weight: &DenseWeight,
    gate: DevicePtr,
    beta: DevicePtr,
    h_state: DevicePtr,
    rms_weight: &DenseWeight,
    normed_out: DevicePtr,
    conv_inter: DevicePtr,
    h_state_inter: DevicePtr,
    group_done: DevicePtr,
    rows: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    l2_eps: f32,
    rms_eps: f32,
    input_stride: u32,
    gate_beta_stride: u32,
    normed_stride: u32,
    conv_inter_stride: u32,
    h_inter_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(deinterleaved)
        .arg_ptr(conv_weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(h_state)
        .arg_ptr(rms_weight.weight)
        .arg_ptr(normed_out)
        .arg_ptr(conv_inter)
        .arg_ptr(h_state_inter)
        .arg_ptr(group_done)
        .arg_u32(rows)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_f32(l2_eps)
        .arg_f32(rms_eps)
        .arg_u32(input_stride)
        .arg_u32(gate_beta_stride)
        .arg_u32(normed_stride)
        .arg_u32(conv_inter_stride)
        .arg_u32(h_inter_stride)
        .launch(stream)
}

/// Flag on [`GdnFusedMultiSeqDesc::flags`]: copy unclamped `raw_g` and `bt`
/// directly into `retain_qkv + 16 * retain_stride` during lazy-commit verify,
/// eliminating per-sequence D2D gate copies.
pub const GDN_FUSED_FLAG_COPY_GATES_TO_RETAIN: u32 = 1;

/// Flag on [`GdnFusedMultiSeqDesc::flags`]: enable zero-replay circular-ring
/// `KEEP_LATE` commit folding inside `qwen4_gdn_verify_fused_lazyfinal_multi_seq`.
pub const GDN_FUSED_FLAG_KEEP_LATE: u32 = 2;

/// Flag on [`GdnFusedMultiSeqDesc::flags`]: `h_state_inter` holds the
/// slot-constant [`GdnKeepLateCtrl`] device pointer (100% CUDA-graph safe).
pub const GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR: u32 = 4;

/// Flag on [`GdnFusedMultiSeqDesc::flags`]: load per-head `h_state` from
/// in-place Scaled-FP16 (`32 KiB` row-major `__half` + `f32` scale at byte
/// offset `32768`) instead of `64 KiB` FP32.
pub const GDN_FUSED_FLAG_LOAD_SCALED_FP16: u32 = 1 << 4;

/// Flag on [`GdnFusedMultiSeqDesc::flags`]: store flushed per-head `h_state`
/// as in-place Scaled-FP16 (`32 KiB` row-major `__half` + `f32` scale at byte
/// offset `32768`) instead of `64 KiB` FP32.
pub const GDN_FUSED_FLAG_STORE_SCALED_FP16: u32 = 1 << 5;

/// Per-slot staggered circular-ring control block (`32 B`, 16-byte aligned)
/// matching `struct alignas(16) GdnKeepLateCtrl` in `qwen4_gdn_verify_fused.cu`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GdnKeepLateCtrl {
    /// Base device pointer of this slot's compact circular ring buffer.
    pub ring_base: u64,
    /// Ring index of the oldest unmaterialized committed row.
    pub pending_base: u32,
    /// Number of unmaterialized committed rows (`0..=ring_cap`) to replay.
    pub pending_rows: u32,
    /// Ring index where the current verify step writes its `num_tokens` rows.
    pub write_base: u32,
    /// Set to `1` on GPU by `blockIdx.x == 0 && threadIdx.x == 0` upon completion
    /// (read by abort reconciliation to detect whether a scheduled flush ran).
    pub executed_flag: u32,
    /// Total compact-row capacity of the circular ring buffer.
    pub ring_cap: u32,
    /// Non-zero when this launch must flush the replayed `conv_state` and
    /// `h_state` to DRAM; zero defers DRAM writes and skips `atomicInc`.
    pub flush_now: u32,
}

impl GdnKeepLateCtrl {
    /// Exact byte size matching CUDA `sizeof(GdnKeepLateCtrl)`.
    pub const SIZE_BYTES: usize = 32;

    /// Serialize this control block in exact little-endian ABI layout into
    /// `dst` (which must be at least [`Self::SIZE_BYTES`] bytes).
    pub fn write_le_bytes(&self, dst: &mut [u8]) {
        debug_assert!(dst.len() >= Self::SIZE_BYTES);
        dst[0..8].copy_from_slice(&self.ring_base.to_le_bytes());
        dst[8..12].copy_from_slice(&self.pending_base.to_le_bytes());
        dst[12..16].copy_from_slice(&self.pending_rows.to_le_bytes());
        dst[16..20].copy_from_slice(&self.write_base.to_le_bytes());
        dst[20..24].copy_from_slice(&self.executed_flag.to_le_bytes());
        dst[24..28].copy_from_slice(&self.ring_cap.to_le_bytes());
        dst[28..32].copy_from_slice(&self.flush_now.to_le_bytes());
    }
}

/// Per-sequence descriptor (`64 B`, 16-byte aligned) matching
/// `struct alignas(16) GdnFusedMultiSeqDesc` in `qwen4_gdn_verify_fused.cu`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GdnFusedMultiSeqDesc {
    pub conv_state: u64,
    pub h_state: u64,
    pub retain_qkv: u64,
    pub conv_inter: u64,
    pub h_state_inter: u64,
    pub group_done: u64,
    pub row_base: u32,
    pub num_tokens: u32,
    pub flags: u32,
    pub _pad: u32,
}

impl GdnFusedMultiSeqDesc {
    /// Exact byte size matching CUDA `sizeof(GdnFusedMultiSeqDesc)`.
    pub const SIZE_BYTES: usize = 64;

    /// Serialize this descriptor in exact little-endian ABI layout into `dst`
    /// (which must be at least [`Self::SIZE_BYTES`] bytes).
    pub fn write_le_bytes(&self, dst: &mut [u8]) {
        debug_assert!(dst.len() >= Self::SIZE_BYTES);
        dst[0..8].copy_from_slice(&self.conv_state.to_le_bytes());
        dst[8..16].copy_from_slice(&self.h_state.to_le_bytes());
        dst[16..24].copy_from_slice(&self.retain_qkv.to_le_bytes());
        dst[24..32].copy_from_slice(&self.conv_inter.to_le_bytes());
        dst[32..40].copy_from_slice(&self.h_state_inter.to_le_bytes());
        dst[40..48].copy_from_slice(&self.group_done.to_le_bytes());
        dst[48..52].copy_from_slice(&self.row_base.to_le_bytes());
        dst[52..56].copy_from_slice(&self.num_tokens.to_le_bytes());
        dst[56..60].copy_from_slice(&self.flags.to_le_bytes());
        dst[60..64].copy_from_slice(&self._pad.to_le_bytes());
    }
}

/// Multi-sequence fused 1x1x1-CTA Qwen4 GDN single-token decode (`grid=[nv, num_seqs, 1]`).
#[allow(clippy::too_many_arguments)]
pub fn qwen4_gdn_decode_fused_k1_multi_seq(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    seq_descs: DevicePtr,
    deinterleaved: DevicePtr,
    conv_weight: &DenseWeight,
    gate: DevicePtr,
    beta: DevicePtr,
    rms_weight: &DenseWeight,
    normed_out: DevicePtr,
    num_seqs: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    l2_eps: f32,
    rms_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, num_seqs, 1])
        .block([128, 1, 1])
        .arg_ptr(seq_descs)
        .arg_ptr(deinterleaved)
        .arg_ptr(conv_weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(rms_weight.weight)
        .arg_ptr(normed_out)
        .arg_u32(num_seqs)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_f32(l2_eps)
        .arg_f32(rms_eps)
        .launch(stream)
}

/// Multi-sequence fused 1x1x1-CTA Qwen4 GDN speculative verify (lazy-commit mode,
/// `grid=[nv, num_seqs, 1]`): advances all sequences' conv1d + L2-norm + GDN
/// recurrence + sigmoid gated RMS norm in one launch and optionally writes
/// unclamped `raw_g` and `bt` directly into each sequence's retain lane.
#[allow(clippy::too_many_arguments)]
pub fn qwen4_gdn_verify_fused_lazyfinal_multi_seq(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    seq_descs: DevicePtr,
    deinterleaved: DevicePtr,
    conv_weight: &DenseWeight,
    gate: DevicePtr,
    beta: DevicePtr,
    rms_weight: &DenseWeight,
    normed_out: DevicePtr,
    num_seqs: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    l2_eps: f32,
    rms_eps: f32,
    input_stride: u32,
    gate_beta_stride: u32,
    normed_stride: u32,
    retain_stride: u32,
    conv_inter_stride: u32,
    h_inter_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, num_seqs, 1])
        .block([128, 1, 1])
        .arg_ptr(seq_descs)
        .arg_ptr(deinterleaved)
        .arg_ptr(conv_weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(rms_weight.weight)
        .arg_ptr(normed_out)
        .arg_u32(num_seqs)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_f32(l2_eps)
        .arg_f32(rms_eps)
        .arg_u32(input_stride)
        .arg_u32(gate_beta_stride)
        .arg_u32(normed_stride)
        .arg_u32(retain_stride)
        .arg_u32(conv_inter_stride)
        .arg_u32(h_inter_stride)
        .launch(stream)
}

/// Multi-sequence fused 1x1x1-CTA Qwen4 GDN speculative verify (persistent
/// snapshot mode, `grid=[nv, num_seqs, 1]`).
#[allow(clippy::too_many_arguments)]
pub fn qwen4_gdn_verify_fused_persistent_multi_seq(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    seq_descs: DevicePtr,
    deinterleaved: DevicePtr,
    conv_weight: &DenseWeight,
    gate: DevicePtr,
    beta: DevicePtr,
    rms_weight: &DenseWeight,
    normed_out: DevicePtr,
    num_seqs: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    l2_eps: f32,
    rms_eps: f32,
    input_stride: u32,
    gate_beta_stride: u32,
    normed_stride: u32,
    conv_inter_stride: u32,
    h_inter_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, num_seqs, 1])
        .block([128, 1, 1])
        .arg_ptr(seq_descs)
        .arg_ptr(deinterleaved)
        .arg_ptr(conv_weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(rms_weight.weight)
        .arg_ptr(normed_out)
        .arg_u32(num_seqs)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_f32(l2_eps)
        .arg_f32(rms_eps)
        .arg_u32(input_stride)
        .arg_u32(gate_beta_stride)
        .arg_u32(normed_stride)
        .arg_u32(conv_inter_stride)
        .arg_u32(h_inter_stride)
        .launch(stream)
}

/// Number of V-column CTAs per value head for the Split-V fused Qwen4 GDN
/// kernels (`grid = [num_v_heads, GDN_SPLIT_V_FACTOR, num_seqs]`).
pub const GDN_SPLIT_V_FACTOR: u32 = 4;

/// Byte offsets inside the reused `ssm_conv_out_f32` scratch buffer for
/// Split-V fused Qwen4 GDN cross-CTA RMS-norm reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnSplitVScratchLayout {
    /// Byte offset of `scratch_ss` (`[total_rows, num_v_heads, 4]` FP32)
    /// relative to `scratch_gdn` (`[total_rows, num_v_heads, 128]` FP32).
    pub ss_offset_bytes: usize,
    /// Total required scratch bytes (`scratch_gdn` + `scratch_ss`).
    pub total_bytes: usize,
}

/// Compute the byte layout required inside `ssm_conv_out_f32` for Split-V
/// fused Qwen4 GDN kernels. Returns `None` on overflow.
#[must_use]
pub const fn gdn_split_v_scratch_layout(
    total_rows: usize,
    num_v_heads: usize,
) -> Option<GdnSplitVScratchLayout> {
    const V_DIM: usize = 128;
    const FP32_BYTES: usize = 4;
    let Some(head_rows) = total_rows.checked_mul(num_v_heads) else {
        return None;
    };
    let Some(gdn_elems) = head_rows.checked_mul(V_DIM) else {
        return None;
    };
    let Some(ss_offset_bytes) = gdn_elems.checked_mul(FP32_BYTES) else {
        return None;
    };
    let Some(ss_elems) = head_rows.checked_mul(GDN_SPLIT_V_FACTOR as usize) else {
        return None;
    };
    let Some(ss_bytes) = ss_elems.checked_mul(FP32_BYTES) else {
        return None;
    };
    let Some(total_bytes) = ss_offset_bytes.checked_add(ss_bytes) else {
        return None;
    };
    Some(GdnSplitVScratchLayout {
        ss_offset_bytes,
        total_bytes,
    })
}

/// Split-V (`grid=[nv, 4, 1]`) fused Qwen4 GDN single-token decode (K=1).
#[expect(
    clippy::too_many_arguments,
    reason = "matches CUDA kernel ABI directly"
)]
pub fn qwen4_gdn_decode_fused_k1_splitv4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    deinterleaved: DevicePtr,
    conv_weight: &DenseWeight,
    gate: DevicePtr,
    beta: DevicePtr,
    h_state: DevicePtr,
    rms_weight: &DenseWeight,
    normed_out: DevicePtr,
    group_done: DevicePtr,
    num_k_heads: u32,
    num_v_heads: u32,
    l2_eps: f32,
    rms_eps: f32,
    scratch_gdn: DevicePtr,
    scratch_ss: DevicePtr,
    split_done: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, GDN_SPLIT_V_FACTOR, 1])
        .block([128, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(deinterleaved)
        .arg_ptr(conv_weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(h_state)
        .arg_ptr(rms_weight.weight)
        .arg_ptr(normed_out)
        .arg_ptr(group_done)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_f32(l2_eps)
        .arg_f32(rms_eps)
        .arg_ptr(scratch_gdn)
        .arg_ptr(scratch_ss)
        .arg_ptr(split_done)
        .launch(stream)
}

/// Split-V (`grid=[nv, 4, 1]`) fused Qwen4 GDN speculative verify (lazyfinal).
#[expect(
    clippy::too_many_arguments,
    reason = "matches CUDA kernel ABI directly"
)]
pub fn qwen4_gdn_verify_fused_lazyfinal_splitv4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    deinterleaved: DevicePtr,
    conv_weight: &DenseWeight,
    gate: DevicePtr,
    beta: DevicePtr,
    h_state: DevicePtr,
    rms_weight: &DenseWeight,
    normed_out: DevicePtr,
    retain_qkv: DevicePtr,
    conv_inter: DevicePtr,
    h_state_inter: DevicePtr,
    group_done: DevicePtr,
    num_tokens: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    l2_eps: f32,
    rms_eps: f32,
    input_stride: u32,
    gate_beta_stride: u32,
    normed_stride: u32,
    retain_stride: u32,
    conv_inter_stride: u32,
    h_inter_stride: u32,
    scratch_gdn: DevicePtr,
    scratch_ss: DevicePtr,
    split_done: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, GDN_SPLIT_V_FACTOR, 1])
        .block([128, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(deinterleaved)
        .arg_ptr(conv_weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(h_state)
        .arg_ptr(rms_weight.weight)
        .arg_ptr(normed_out)
        .arg_ptr(retain_qkv)
        .arg_ptr(conv_inter)
        .arg_ptr(h_state_inter)
        .arg_ptr(group_done)
        .arg_u32(num_tokens)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_f32(l2_eps)
        .arg_f32(rms_eps)
        .arg_u32(input_stride)
        .arg_u32(gate_beta_stride)
        .arg_u32(normed_stride)
        .arg_u32(retain_stride)
        .arg_u32(conv_inter_stride)
        .arg_u32(h_inter_stride)
        .arg_ptr(scratch_gdn)
        .arg_ptr(scratch_ss)
        .arg_ptr(split_done)
        .launch(stream)
}

/// Split-V (`grid=[nv, 4, 1]`) fused Qwen4 GDN speculative verify (persistent).
#[expect(
    clippy::too_many_arguments,
    reason = "matches CUDA kernel ABI directly"
)]
pub fn qwen4_gdn_verify_fused_persistent_splitv4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    deinterleaved: DevicePtr,
    conv_weight: &DenseWeight,
    gate: DevicePtr,
    beta: DevicePtr,
    h_state: DevicePtr,
    rms_weight: &DenseWeight,
    normed_out: DevicePtr,
    conv_inter: DevicePtr,
    h_state_inter: DevicePtr,
    group_done: DevicePtr,
    num_tokens: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    l2_eps: f32,
    rms_eps: f32,
    input_stride: u32,
    gate_beta_stride: u32,
    normed_stride: u32,
    conv_inter_stride: u32,
    h_inter_stride: u32,
    scratch_gdn: DevicePtr,
    scratch_ss: DevicePtr,
    split_done: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, GDN_SPLIT_V_FACTOR, 1])
        .block([128, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(deinterleaved)
        .arg_ptr(conv_weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(h_state)
        .arg_ptr(rms_weight.weight)
        .arg_ptr(normed_out)
        .arg_ptr(conv_inter)
        .arg_ptr(h_state_inter)
        .arg_ptr(group_done)
        .arg_u32(num_tokens)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_f32(l2_eps)
        .arg_f32(rms_eps)
        .arg_u32(input_stride)
        .arg_u32(gate_beta_stride)
        .arg_u32(normed_stride)
        .arg_u32(conv_inter_stride)
        .arg_u32(h_inter_stride)
        .arg_ptr(scratch_gdn)
        .arg_ptr(scratch_ss)
        .arg_ptr(split_done)
        .launch(stream)
}

/// Split-V (`grid=[nv, 4, num_seqs]`) multi-sequence fused Qwen4 GDN speculative
/// verify (lazyfinal / KEEP_LATE circular ring, FP32 H-state only).
#[expect(
    clippy::too_many_arguments,
    reason = "matches CUDA kernel ABI directly"
)]
pub fn qwen4_gdn_verify_fused_lazyfinal_multi_seq_splitv4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    seq_descs: DevicePtr,
    deinterleaved: DevicePtr,
    conv_weight: &DenseWeight,
    gate: DevicePtr,
    beta: DevicePtr,
    rms_weight: &DenseWeight,
    normed_out: DevicePtr,
    num_seqs: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    l2_eps: f32,
    rms_eps: f32,
    input_stride: u32,
    gate_beta_stride: u32,
    normed_stride: u32,
    retain_stride: u32,
    conv_inter_stride: u32,
    h_inter_stride: u32,
    scratch_gdn: DevicePtr,
    scratch_ss: DevicePtr,
    split_done: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, GDN_SPLIT_V_FACTOR, num_seqs])
        .block([128, 1, 1])
        .arg_ptr(seq_descs)
        .arg_ptr(deinterleaved)
        .arg_ptr(conv_weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(rms_weight.weight)
        .arg_ptr(normed_out)
        .arg_u32(num_seqs)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_f32(l2_eps)
        .arg_f32(rms_eps)
        .arg_u32(input_stride)
        .arg_u32(gate_beta_stride)
        .arg_u32(normed_stride)
        .arg_u32(retain_stride)
        .arg_u32(conv_inter_stride)
        .arg_u32(h_inter_stride)
        .arg_ptr(scratch_gdn)
        .arg_ptr(scratch_ss)
        .arg_ptr(split_done)
        .launch(stream)
}

/// Split-V (`grid=[nv, 4, num_seqs]`) multi-sequence fused Qwen4 GDN speculative
/// verify (persistent snapshot mode, FP32 H-state only).
#[expect(
    clippy::too_many_arguments,
    reason = "matches CUDA kernel ABI directly"
)]
pub fn qwen4_gdn_verify_fused_persistent_multi_seq_splitv4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    seq_descs: DevicePtr,
    deinterleaved: DevicePtr,
    conv_weight: &DenseWeight,
    gate: DevicePtr,
    beta: DevicePtr,
    rms_weight: &DenseWeight,
    normed_out: DevicePtr,
    num_seqs: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    l2_eps: f32,
    rms_eps: f32,
    input_stride: u32,
    gate_beta_stride: u32,
    normed_stride: u32,
    conv_inter_stride: u32,
    h_inter_stride: u32,
    scratch_gdn: DevicePtr,
    scratch_ss: DevicePtr,
    split_done: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, GDN_SPLIT_V_FACTOR, num_seqs])
        .block([128, 1, 1])
        .arg_ptr(seq_descs)
        .arg_ptr(deinterleaved)
        .arg_ptr(conv_weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(rms_weight.weight)
        .arg_ptr(normed_out)
        .arg_u32(num_seqs)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_f32(l2_eps)
        .arg_f32(rms_eps)
        .arg_u32(input_stride)
        .arg_u32(gate_beta_stride)
        .arg_u32(normed_stride)
        .arg_u32(conv_inter_stride)
        .arg_u32(h_inter_stride)
        .arg_ptr(scratch_gdn)
        .arg_ptr(scratch_ss)
        .arg_ptr(split_done)
        .launch(stream)
}

#[cfg(test)]
mod gdn_fused_multi_desc_tests {
    use super::{
        GDN_FUSED_FLAG_COPY_GATES_TO_RETAIN, GDN_FUSED_FLAG_KEEP_LATE,
        GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR, GDN_FUSED_FLAG_LOAD_SCALED_FP16,
        GDN_FUSED_FLAG_STORE_SCALED_FP16, GDN_SPLIT_V_FACTOR, GdnFusedMultiSeqDesc,
        GdnKeepLateCtrl, conv1d_f32_sequence_block_size, gdn_split_v_scratch_layout,
    };

    #[test]
    fn conv1d_f32_sequence_block_size_widens_only_on_wide_gpus_with_hd128() {
        // GB10 (<= 64 SMs) stays at block=256 (0 regression).
        assert_eq!(conv1d_f32_sequence_block_size(48, 128, 4096), 256);
        assert_eq!(conv1d_f32_sequence_block_size(64, 128, 4096), 256);
        // Wide discrete GPUs (SM120 188 SMs / SM103 160 SMs) halve block to 128
        // (doubling CTAs 40 -> 80) when head_dim == 128 and qk_channels % 128 == 0.
        assert_eq!(conv1d_f32_sequence_block_size(188, 128, 4096), 128);
        assert_eq!(conv1d_f32_sequence_block_size(160, 128, 4096), 128);
        // Non-128 head_dim or unaligned qk_channels stays safely at 256.
        assert_eq!(conv1d_f32_sequence_block_size(188, 256, 4096), 256);
        assert_eq!(conv1d_f32_sequence_block_size(188, 64, 4096), 256);
        assert_eq!(conv1d_f32_sequence_block_size(188, 128, 4100), 256);
    }

    #[test]
    fn gdn_split_v_scratch_layout_fits_ssm_conv_out_f32() {
        let layout = gdn_split_v_scratch_layout(8, 48).expect("8x48 scratch fits");
        assert_eq!(layout.ss_offset_bytes, 8 * 48 * 128 * 4);
        assert_eq!(
            layout.total_bytes,
            8 * 48 * (128 + GDN_SPLIT_V_FACTOR as usize) * 4
        );
        // Qwen4 `ssm_conv_out_f32` is rows * qkvz_size * 4 = rows * 16384 * 4.
        assert!(layout.total_bytes < 8 * 16_384 * 4);
        assert_eq!(gdn_split_v_scratch_layout(usize::MAX, 48), None);
    }

    #[test]
    fn gdn_keep_late_ctrl_abi_layout() {
        assert_eq!(std::mem::size_of::<GdnKeepLateCtrl>(), 32);
        assert_eq!(std::mem::align_of::<GdnKeepLateCtrl>(), 16);
        assert_eq!(std::mem::offset_of!(GdnKeepLateCtrl, ring_base), 0);
        assert_eq!(std::mem::offset_of!(GdnKeepLateCtrl, pending_base), 8);
        assert_eq!(std::mem::offset_of!(GdnKeepLateCtrl, pending_rows), 12);
        assert_eq!(std::mem::offset_of!(GdnKeepLateCtrl, write_base), 16);
        assert_eq!(std::mem::offset_of!(GdnKeepLateCtrl, executed_flag), 20);
        assert_eq!(std::mem::offset_of!(GdnKeepLateCtrl, ring_cap), 24);
        assert_eq!(std::mem::offset_of!(GdnKeepLateCtrl, flush_now), 28);
        assert_eq!(GDN_FUSED_FLAG_KEEP_LATE, 2);
        assert_eq!(GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR, 4);
        assert_eq!(GDN_FUSED_FLAG_LOAD_SCALED_FP16, 16);
        assert_eq!(GDN_FUSED_FLAG_STORE_SCALED_FP16, 32);
        let all_flags = [
            GDN_FUSED_FLAG_COPY_GATES_TO_RETAIN,
            GDN_FUSED_FLAG_KEEP_LATE,
            GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR,
            GDN_FUSED_FLAG_LOAD_SCALED_FP16,
            GDN_FUSED_FLAG_STORE_SCALED_FP16,
        ];
        let mut combined = 0u32;
        for flag in all_flags {
            assert_eq!(flag.count_ones(), 1, "flag {flag} must be a single bit");
            assert_eq!(combined & flag, 0, "flag {flag} collides with {combined}");
            combined |= flag;
        }
        assert_eq!(combined, 1 | 2 | 4 | 16 | 32);

        let ctrl = GdnKeepLateCtrl {
            ring_base: 0x1122_3344_5566_7788,
            pending_base: 91,
            pending_rows: 17,
            write_base: 8,
            executed_flag: 0,
            ring_cap: 100,
            flush_now: 1,
        };
        let mut bytes = [0u8; GdnKeepLateCtrl::SIZE_BYTES];
        ctrl.write_le_bytes(&mut bytes);
        assert_eq!(
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            0x1122_3344_5566_7788
        );
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 91);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 17);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 8);
        assert_eq!(u32::from_le_bytes(bytes[20..24].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 100);
        assert_eq!(u32::from_le_bytes(bytes[28..32].try_into().unwrap()), 1);
    }

    #[test]
    fn gdn_fused_multi_seq_desc_abi_layout() {
        assert_eq!(std::mem::size_of::<GdnFusedMultiSeqDesc>(), 64);
        assert_eq!(std::mem::align_of::<GdnFusedMultiSeqDesc>(), 16);
        assert_eq!(std::mem::offset_of!(GdnFusedMultiSeqDesc, conv_state), 0);
        assert_eq!(std::mem::offset_of!(GdnFusedMultiSeqDesc, h_state), 8);
        assert_eq!(std::mem::offset_of!(GdnFusedMultiSeqDesc, retain_qkv), 16);
        assert_eq!(std::mem::offset_of!(GdnFusedMultiSeqDesc, conv_inter), 24);
        assert_eq!(
            std::mem::offset_of!(GdnFusedMultiSeqDesc, h_state_inter),
            32
        );
        assert_eq!(std::mem::offset_of!(GdnFusedMultiSeqDesc, group_done), 40);
        assert_eq!(std::mem::offset_of!(GdnFusedMultiSeqDesc, row_base), 48);
        assert_eq!(std::mem::offset_of!(GdnFusedMultiSeqDesc, num_tokens), 52);
        assert_eq!(std::mem::offset_of!(GdnFusedMultiSeqDesc, flags), 56);
        assert_eq!(std::mem::offset_of!(GdnFusedMultiSeqDesc, _pad), 60);

        let desc = GdnFusedMultiSeqDesc {
            conv_state: 0x1111_2222_3333_4444,
            h_state: 0x5555_6666_7777_8888,
            retain_qkv: 0x9999_aaaa_bbbb_cccc,
            conv_inter: 0xdddd_eeee_ffff_0001,
            h_state_inter: 0x1234_5678_9abc_def0,
            group_done: 0x0fed_cba9_8765_4321,
            row_base: 15,
            num_tokens: 5,
            flags: GDN_FUSED_FLAG_COPY_GATES_TO_RETAIN,
            _pad: 0,
        };
        let mut bytes = [0u8; GdnFusedMultiSeqDesc::SIZE_BYTES];
        desc.write_le_bytes(&mut bytes);
        assert_eq!(
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            desc.conv_state
        );
        assert_eq!(
            u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            desc.group_done
        );
        assert_eq!(u32::from_le_bytes(bytes[48..52].try_into().unwrap()), 15);
        assert_eq!(u32::from_le_bytes(bytes[52..56].try_into().unwrap()), 5);
        assert_eq!(u32::from_le_bytes(bytes[56..60].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(bytes[60..64].try_into().unwrap()), 0);
    }
}
