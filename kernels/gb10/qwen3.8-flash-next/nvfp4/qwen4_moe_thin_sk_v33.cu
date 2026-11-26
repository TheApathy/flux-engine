// SPDX-License-Identifier: AGPL-3.0-only
//
// V33 Thin-M Split-K MoE Verify Kernel (`v33_thin_sk`, `rows = 4..256`):
// Bandwidth-optimal NVFP4 MoE GEMM targeting SM120 / SM121 verify batches.
//
// Modularized manifest: shared PTX/FP4 primitives live in
// `detail/qwen4_nvfp4_common.cuh`, compile-time model shape traits live in
// `qwen4_geometry.cuh`, and templated kernels live under `detail/moe_thin_sk/`.

#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define OI640_STEP_K 32
#define OI640_TRANSPOSED 1
#define flash_next_orig_i640_prefill flash_next_f32_transposed_compact_thin_sk_v33
#include "moe_w4a16_orig_i640_compact_prefill.cuh"
#include "qwen4_geometry.cuh"
#include "detail/qwen4_nvfp4_common.cuh"
#include <cuda_fp16.h>
#include <type_traits>

namespace oi640 = flash_next_f32_transposed_compact_thin_sk_v33;

#include "detail/moe_thin_sk/v33_gemm_impl.cuh"
#include "detail/moe_thin_sk/v33_epilogue_reduce.cuh"

// =========================================================================
// Public V33 Thin-M Split-K Entrypoints (Flat + Atlas-TS4E-128)
// =========================================================================

extern "C" __global__ __launch_bounds__(128, 6) void qwen4_moe_thin_sk_gateup_v33(
    const __nv_bfloat16* __restrict__ X,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ act_out,
    float* __restrict__ partials,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    unsigned int split_k,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out,
    unsigned int sg_rows
) {
    v33_thin_sk_impl<Qwen4FlashNextShape, true, 0, false>(
        X, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals,
        up_packed_ptrs, up_scale_ptrs, up_scale2_vals,
        act_out, nullptr, nullptr, partials,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, split_k,
        shared_gate, ss_out, sg_rows);
}

extern "C" __global__ __launch_bounds__(128, 6) void qwen4_moe_thin_sk_gateup_v33_ts4e(
    const __nv_bfloat16* __restrict__ X,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ act_out,
    float* __restrict__ partials,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    unsigned int split_k,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out,
    unsigned int sg_rows
) {
    v33_thin_sk_impl<Qwen4FlashNextShape, true, 0, true>(
        X, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals,
        up_packed_ptrs, up_scale_ptrs, up_scale2_vals,
        act_out, nullptr, nullptr, partials,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, split_k,
        shared_gate, ss_out, sg_rows);
}

extern "C" __global__ __launch_bounds__(128, 6) void qwen4_moe_thin_sk_down_bf16_v33(
    const __nv_bfloat16* __restrict__ act_in,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ partials,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    unsigned int split_k
) {
    v33_thin_sk_impl<Qwen4FlashNextShape, false, 0, false>(
        act_in, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr,
        C, nullptr, nullptr, partials,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, split_k);
}

extern "C" __global__ __launch_bounds__(128, 6) void qwen4_moe_thin_sk_down_bf16_v33_ts4e(
    const __nv_bfloat16* __restrict__ act_in,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ partials,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    unsigned int split_k
) {
    v33_thin_sk_impl<Qwen4FlashNextShape, false, 0, true>(
        act_in, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr,
        C, nullptr, nullptr, partials,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, split_k);
}

extern "C" __global__ __launch_bounds__(128, 6) void qwen4_moe_thin_sk_down_mxfp8_v33(
    const __nv_bfloat16* __restrict__ act_in,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    unsigned char* __restrict__ slot_q,
    unsigned char* __restrict__ slot_s,
    float* __restrict__ partials,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    unsigned int split_k
) {
    v33_thin_sk_impl<Qwen4FlashNextShape, false, 3, false>(
        act_in, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr,
        nullptr, slot_q, slot_s, partials,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, split_k);
}

extern "C" __global__ __launch_bounds__(128, 6) void qwen4_moe_thin_sk_down_mxfp8_v33_ts4e(
    const __nv_bfloat16* __restrict__ act_in,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    unsigned char* __restrict__ slot_q,
    unsigned char* __restrict__ slot_s,
    float* __restrict__ partials,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    unsigned int split_k
) {
    v33_thin_sk_impl<Qwen4FlashNextShape, false, 3, true>(
        act_in, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr,
        nullptr, slot_q, slot_s, partials,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, split_k);
}

// =========================================================================
// Fused Split-K Partial Reduction & Shared-Gate Entrypoints
// =========================================================================

extern "C" __global__ __launch_bounds__(32, 8) void qwen4_moe_thin_sk_reduce_silu_v33(
    const float* __restrict__ partials,
    __nv_bfloat16* __restrict__ act_out,
    unsigned int pairs,
    unsigned int split_k
) {
    v33_reduce_silu_impl<Qwen4FlashNextShape>(partials, act_out, pairs, split_k);
}

extern "C" __global__ __launch_bounds__(128, 8) void qwen4_moe_thin_sk_down_reduce_bf16_v33(
    const float* __restrict__ partials,
    __nv_bfloat16* __restrict__ C,
    unsigned int pairs,
    unsigned int split_k
) {
    v33_down_reduce_bf16_impl<Qwen4FlashNextShape>(partials, C, pairs, split_k);
}

extern "C" __global__ __launch_bounds__(128, 8) void qwen4_moe_thin_sk_down_reduce_mxfp8_v33(
    const float* __restrict__ partials,
    unsigned char* __restrict__ slot_q,
    unsigned char* __restrict__ slot_s,
    unsigned int pairs,
    unsigned int split_k
) {
    v33_down_reduce_mxfp8_impl<Qwen4FlashNextShape>(partials, slot_q, slot_s, pairs, split_k);
}

extern "C" __global__ __launch_bounds__(256, 4) void qwen4_moe_thin_sk_shared_gate_v33(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out,
    unsigned int rows
) {
    v33_shared_gate_impl<Qwen4FlashNextShape>(input, shared_gate, ss_out, rows);
}
