// SPDX-License-Identifier: AGPL-3.0-only
//
// V32_W4A4: Native Blackwell W4A4 (NVFP4 x NVFP4) persistent MoE prefill kernels
// for Flash-Next.
//
// Modularized manifest: shared PTX/FP4 primitives live in
// `detail/qwen4_nvfp4_common.cuh`, compile-time model shape traits live in
// `qwen4_geometry.cuh`, and templated kernels live under `detail/moe_w4a4/`.

#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define OI640_STEP_K 32
#define OI640_TRANSPOSED 1
#define flash_next_orig_i640_prefill flash_next_w4a4_v32
#include "moe_w4a16_orig_i640_compact_prefill.cuh"
#include "qwen4_geometry.cuh"
#include "detail/qwen4_nvfp4_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace oi640 = flash_next_w4a4_v32;

#include "detail/moe_w4a4/v32_gemm_impl.cuh"
#include "detail/moe_w4a4/v32_quant_gate.cuh"

// =========================================================================
// Single-read fused BF16 -> NVFP4 activation quantizer + shared_gate dot
// =========================================================================

extern "C" __global__ __launch_bounds__(256) void qwen4_nvfp4_quant_rows_v32_gate(
    const __nv_bfloat16* __restrict__ src,
    unsigned char* __restrict__ xq,
    unsigned char* __restrict__ xs,
    float* __restrict__ x_scale2,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out,
    unsigned int rows
) {
    v32_nvfp4_quant_rows_gate_impl<Qwen4FlashNextShape>(
        src, xq, xs, x_scale2, shared_gate, ss_out, rows);
}

// =========================================================================
// V32_W4A4 GATEUP Entry Points (256 threads, 2 CTAs/SM, <= 41.6 KiB SMEM)
// =========================================================================

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_gateup_v32(
    const unsigned char* __restrict__ xq,
    const unsigned char* __restrict__ xs,
    const float* __restrict__ x_scale2,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    unsigned char* __restrict__ out_q,
    unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<Qwen4FlashNextShape, true, false, 0, true, false>(
        xq, xs, x_scale2, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals,
        up_packed_ptrs, up_scale_ptrs, up_scale2_vals, nullptr, out_q, out_s,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        oq_scale2_buf);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_gateup_v32_ts4e(
    const unsigned char* __restrict__ xq,
    const unsigned char* __restrict__ xs,
    const float* __restrict__ x_scale2,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    unsigned char* __restrict__ out_q,
    unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<Qwen4FlashNextShape, true, false, 0, true, true>(
        xq, xs, x_scale2, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals,
        up_packed_ptrs, up_scale_ptrs, up_scale2_vals, nullptr, out_q, out_s,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        oq_scale2_buf);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_gateup_v32_oq4(
    const unsigned char* __restrict__ xq,
    const unsigned char* __restrict__ xs,
    const float* __restrict__ x_scale2,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    unsigned char* __restrict__ out_q,
    unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<Qwen4FlashNextShape, true, true, 0, true, false>(
        xq, xs, x_scale2, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals,
        up_packed_ptrs, up_scale_ptrs, up_scale2_vals, nullptr, out_q, out_s,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        oq_scale2_buf);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_gateup_v32_oq4_ts4e(
    const unsigned char* __restrict__ xq,
    const unsigned char* __restrict__ xs,
    const float* __restrict__ x_scale2,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    unsigned char* __restrict__ out_q,
    unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<Qwen4FlashNextShape, true, true, 0, true, true>(
        xq, xs, x_scale2, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals,
        up_packed_ptrs, up_scale_ptrs, up_scale2_vals, nullptr, out_q, out_s,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        oq_scale2_buf);
}

// =========================================================================
// V32_W4A4 DOWN Entry Points (256 threads, 2 CTAs/SM, <= 45.4 KiB SMEM)
// =========================================================================

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_down_v32(
    const unsigned char* __restrict__ oq,
    const unsigned char* __restrict__ os,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    unsigned char* __restrict__ slot_q,
    unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    const float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<Qwen4FlashNextShape, false, false, 3, true, false>(
        oq, os, nullptr, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        nullptr, oq_scale2_buf, slot_q, slot_s);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_down_v32_ts4e(
    const unsigned char* __restrict__ oq,
    const unsigned char* __restrict__ os,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    unsigned char* __restrict__ slot_q,
    unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    const float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<Qwen4FlashNextShape, false, false, 3, true, true>(
        oq, os, nullptr, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        nullptr, oq_scale2_buf, slot_q, slot_s);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_down_v32_bf16(
    const unsigned char* __restrict__ oq,
    const unsigned char* __restrict__ os,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    const float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<Qwen4FlashNextShape, false, false, 0, true, false>(
        oq, os, nullptr, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        nullptr, oq_scale2_buf);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_down_v32_bf16_ts4e(
    const unsigned char* __restrict__ oq,
    const unsigned char* __restrict__ os,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    const float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<Qwen4FlashNextShape, false, false, 0, true, true>(
        oq, os, nullptr, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        nullptr, oq_scale2_buf);
}

// Dynamic SMEM query helper: [flat_gu, flat_dn, ts4e_gu, ts4e_dn].
extern "C" __global__ void qwen4_moe_w4a4_v32_smem_bytes(unsigned* out) {
    out[0] = V32W4a4Smem<Qwen4FlashNextShape, true, false>::BYTES;
    out[1] = V32W4a4Smem<Qwen4FlashNextShape, false, false>::BYTES;
    out[2] = V32W4a4Smem<Qwen4FlashNextShape, true, true>::BYTES;
    out[3] = V32W4a4Smem<Qwen4FlashNextShape, false, true>::BYTES;
}
