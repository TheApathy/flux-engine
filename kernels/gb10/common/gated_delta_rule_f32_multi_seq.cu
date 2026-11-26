// SPDX-License-Identifier: AGPL-3.0-only
//
// FP32-output variant of the multi-seq gated delta rule decode.
//
// Same compute as `gated_delta_rule_decode_multi_seq` but writes FP32
// output instead of BF16. Required to wire ATLAS_SSM_MULTI_SEQ_KERNEL
// into the production decode path: the single-seq path uses
// `gated_delta_rule_decode_f32` which writes FP32 to maintain precision
// in the gated_rms_norm input downstream. Qwen3.6-27B is calibrated
// for this precision regime; BF16 output would shift the gated_rms
// distribution and degrade output quality at long context.
//
// Grid: (num_v_heads, num_seqs, 1)  Block: (128, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 128

template <bool H_IS_F16>
__device__ __forceinline__ float gdn_ms_load_cs(const void* base, unsigned int idx) {
    if constexpr (H_IS_F16) {
        const unsigned short* p = reinterpret_cast<const unsigned short*>(base) + idx;
        return __half2float(__ushort_as_half(__ldcs(p)));
    } else {
        return __ldcs(reinterpret_cast<const float*>(base) + idx);
    }
}

template <bool H_IS_F16>
__device__ __forceinline__ void gdn_ms_store_cs(void* base, unsigned int idx, float v) {
    if constexpr (H_IS_F16) {
        unsigned short* p = reinterpret_cast<unsigned short*>(base) + idx;
        __half h = __float2half(fminf(fmaxf(v, -65504.0f), 65504.0f));
        __stcs(p, __half_as_ushort(h));
    } else {
        __stcs(reinterpret_cast<float*>(base) + idx, v);
    }
}

// The recurrent-state norm clamp is a Qwen3.6-27B calibration artefact
// (see common/gated_delta_rule.cu, whose single-sequence decode kernels carry
// the same clamp). A model whose single-sequence decode kernel
// (gated_delta_rule_decode_f32) carries NO clamp must compile this twin
// without it, or C1 and C2..C32 rows of the same request advance different
// recurrences once a value head's state norm exceeds SSM_STATE_MAX_NORM. A
// model-dir override defines ATLAS_GDN_MULTI_SEQ_NO_STATE_NORM before
// including this file (kernels/gb10/qwen3.8-flash-next/nvfp4/).
#if !defined(SSM_STATE_NORM_ENABLED) && !defined(ATLAS_GDN_MULTI_SEQ_NO_STATE_NORM)
#define SSM_STATE_NORM_ENABLED
#endif
#ifndef SSM_STATE_MAX_NORM
#define SSM_STATE_MAX_NORM 1000.0f
#endif

// ============================================================
// DECODE multi-seq: gated delta rule recurrent update (FP32 output).
// ============================================================
// Inputs are FP32 (NOT BF16): conv1d_update_l2norm_f32_multi_seq writes
// FP32, and Qwen3.6-27B's calibration relies on FP32 conv→GDN precision.
// Strides for q/k/v are in FP32 elements (qkvz_size FP32 per seq).
// `H_IS_F16=false`: FP32 DRAM h_state (default exact path).
// `H_IS_F16=true`:  FP16 DRAM h_state (`ATLAS_SSM_H_FP16=1`) with
//                   full FP32 SMEM/register recurrence and saturating store.
template <bool H_IS_F16>
__device__ __forceinline__ void gated_delta_rule_decode_multi_seq_body(
    void* const* __restrict__ h_states,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    unsigned int num_seqs,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int gate_beta_stride, // FP32 elements between seqs in gate/beta
    unsigned int qk_stride,        // FP32 elements between seqs in query/key
    unsigned int v_in_stride,      // FP32 elements between seqs in value
    unsigned int v_out_stride      // FP32 elements between seqs in output
) {
    const unsigned int vh  = blockIdx.x;
    const unsigned int seq = blockIdx.y;
    if (vh >= num_v_heads || seq >= num_seqs || k_dim == 0 || k_dim > 128) return;

    const unsigned int tid = threadIdx.x;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    void* H = H_IS_F16
        ? static_cast<void*>(reinterpret_cast<__half*>(h_states[seq]) + ((unsigned long long)vh * k_dim * v_dim))
        : static_cast<void*>(reinterpret_cast<float*>(h_states[seq]) + ((unsigned long long)vh * k_dim * v_dim));

    const float* q_ptr = query + (unsigned long long)seq * qk_stride + kh * k_dim;
    const float* k_ptr = key   + (unsigned long long)seq * qk_stride + kh * k_dim;
    const float* v_ptr = value + (unsigned long long)seq * v_in_stride + vh * v_dim;

    float g_raw = gate[(unsigned long long)seq * gate_beta_stride + vh];
    const float g = fminf(fmaxf(g_raw, 0.0f), 1.0f);
    const float bt = beta[(unsigned long long)seq * gate_beta_stride + vh];

    __shared__ float smem_k[128];
    __shared__ float smem_q[128];
    // Cache the lower 63 rows of H ([63, 128] = 32,256 B; total SMEM = 33,296 B ->
    // 3 CTAs/SM on 99 KiB SMEM = +50% occupancy vs 48 KiB 2 CTAs/SM) and hold the
    // upper 65 rows in registers (0 spills for BOTH f32 and f16), using streaming
    // `__ldcs` / `__stcs` (L2 evict-first) so multi-seq GDN state never pollutes L2.
    __shared__ float smem_h[63][128];
    #ifdef SSM_STATE_NORM_ENABLED
    __shared__ float norm_sums[4];
    #endif

    if (tid < k_dim) {
        smem_k[tid] = k_ptr[tid];
        smem_q[tid] = q_ptr[tid];
    }
    __syncthreads();

    float q_dot = 0.0f;
    #ifdef SSM_STATE_NORM_ENABLED
    float local_sq = 0.0f;
    #endif

    if (k_dim == 128 && v_dim <= 128) {
        float h_hi[65];
        if (tid < v_dim) {
            float v_i = v_ptr[tid];
            float hk_dot = 0.0f;

            // Pass 1a: lower 63 rows -> smem_h[0..63][tid]
            #pragma unroll 1
            for (unsigned int j = 0; j < 60; j += 4) {
                float h0 = gdn_ms_load_cs<H_IS_F16>(H, (j + 0) * v_dim + tid);
                float h1 = gdn_ms_load_cs<H_IS_F16>(H, (j + 1) * v_dim + tid);
                float h2 = gdn_ms_load_cs<H_IS_F16>(H, (j + 2) * v_dim + tid);
                float h3 = gdn_ms_load_cs<H_IS_F16>(H, (j + 3) * v_dim + tid);
                smem_h[j + 0][tid] = h0;
                smem_h[j + 1][tid] = h1;
                smem_h[j + 2][tid] = h2;
                smem_h[j + 3][tid] = h3;
                hk_dot += h0 * smem_k[j] + h1 * smem_k[j + 1]
                        + h2 * smem_k[j + 2] + h3 * smem_k[j + 3];
            }
            {
                float h0 = gdn_ms_load_cs<H_IS_F16>(H, 60 * v_dim + tid);
                float h1 = gdn_ms_load_cs<H_IS_F16>(H, 61 * v_dim + tid);
                float h2 = gdn_ms_load_cs<H_IS_F16>(H, 62 * v_dim + tid);
                smem_h[60][tid] = h0;
                smem_h[61][tid] = h1;
                smem_h[62][tid] = h2;
                hk_dot += h0 * smem_k[60] + h1 * smem_k[61] + h2 * smem_k[62];
            }

            // Pass 1b: upper 65 rows -> registers h_hi[0..65]
            {
                float h0 = gdn_ms_load_cs<H_IS_F16>(H, 63 * v_dim + tid);
                h_hi[0] = h0;
                hk_dot += h0 * smem_k[63];
            }
            #pragma unroll
            for (unsigned int j = 1; j < 65; j += 4) {
                unsigned int row = 63 + j;
                float h0 = gdn_ms_load_cs<H_IS_F16>(H, (row + 0) * v_dim + tid);
                float h1 = gdn_ms_load_cs<H_IS_F16>(H, (row + 1) * v_dim + tid);
                float h2 = gdn_ms_load_cs<H_IS_F16>(H, (row + 2) * v_dim + tid);
                float h3 = gdn_ms_load_cs<H_IS_F16>(H, (row + 3) * v_dim + tid);
                h_hi[j + 0] = h0;
                h_hi[j + 1] = h1;
                h_hi[j + 2] = h2;
                h_hi[j + 3] = h3;
                hk_dot += h0 * smem_k[row] + h1 * smem_k[row + 1]
                        + h2 * smem_k[row + 2] + h3 * smem_k[row + 3];
            }

            float v_new_i = (v_i - g * hk_dot) * bt;

            // Pass 2a: update lower 63 rows from smem_h -> write H once (streaming)
            #pragma unroll 1
            for (unsigned int j = 0; j < 60; j += 4) {
                float h0 = g * smem_h[j + 0][tid] + smem_k[j]     * v_new_i;
                float h1 = g * smem_h[j + 1][tid] + smem_k[j + 1] * v_new_i;
                float h2 = g * smem_h[j + 2][tid] + smem_k[j + 2] * v_new_i;
                float h3 = g * smem_h[j + 3][tid] + smem_k[j + 3] * v_new_i;
                gdn_ms_store_cs<H_IS_F16>(H, (j + 0) * v_dim + tid, h0);
                gdn_ms_store_cs<H_IS_F16>(H, (j + 1) * v_dim + tid, h1);
                gdn_ms_store_cs<H_IS_F16>(H, (j + 2) * v_dim + tid, h2);
                gdn_ms_store_cs<H_IS_F16>(H, (j + 3) * v_dim + tid, h3);
                q_dot += h0 * smem_q[j] + h1 * smem_q[j + 1]
                       + h2 * smem_q[j + 2] + h3 * smem_q[j + 3];
                #ifdef SSM_STATE_NORM_ENABLED
                local_sq += h0 * h0;
                local_sq += h1 * h1;
                local_sq += h2 * h2;
                local_sq += h3 * h3;
                #endif
            }
            {
                float h0 = g * smem_h[60][tid] + smem_k[60] * v_new_i;
                float h1 = g * smem_h[61][tid] + smem_k[61] * v_new_i;
                float h2 = g * smem_h[62][tid] + smem_k[62] * v_new_i;
                gdn_ms_store_cs<H_IS_F16>(H, 60 * v_dim + tid, h0);
                gdn_ms_store_cs<H_IS_F16>(H, 61 * v_dim + tid, h1);
                gdn_ms_store_cs<H_IS_F16>(H, 62 * v_dim + tid, h2);
                q_dot += h0 * smem_q[60] + h1 * smem_q[61] + h2 * smem_q[62];
                #ifdef SSM_STATE_NORM_ENABLED
                local_sq += h0 * h0;
                local_sq += h1 * h1;
                local_sq += h2 * h2;
                #endif
            }

            // Pass 2b: update upper 65 rows from registers h_hi -> write H once
            {
                float h0 = g * h_hi[0] + smem_k[63] * v_new_i;
                gdn_ms_store_cs<H_IS_F16>(H, 63 * v_dim + tid, h0);
                q_dot += h0 * smem_q[63];
                #ifdef SSM_STATE_NORM_ENABLED
                local_sq += h0 * h0;
                #endif
            }
            #pragma unroll
            for (unsigned int j = 1; j < 65; j += 4) {
                unsigned int row = 63 + j;
                float h0 = g * h_hi[j + 0] + smem_k[row]     * v_new_i;
                float h1 = g * h_hi[j + 1] + smem_k[row + 1] * v_new_i;
                float h2 = g * h_hi[j + 2] + smem_k[row + 2] * v_new_i;
                float h3 = g * h_hi[j + 3] + smem_k[row + 3] * v_new_i;
                gdn_ms_store_cs<H_IS_F16>(H, (row + 0) * v_dim + tid, h0);
                gdn_ms_store_cs<H_IS_F16>(H, (row + 1) * v_dim + tid, h1);
                gdn_ms_store_cs<H_IS_F16>(H, (row + 2) * v_dim + tid, h2);
                gdn_ms_store_cs<H_IS_F16>(H, (row + 3) * v_dim + tid, h3);
                q_dot += h0 * smem_q[row] + h1 * smem_q[row + 1]
                       + h2 * smem_q[row + 2] + h3 * smem_q[row + 3];
                #ifdef SSM_STATE_NORM_ENABLED
                local_sq += h0 * h0;
                local_sq += h1 * h1;
                local_sq += h2 * h2;
                local_sq += h3 * h3;
                #endif
            }
        }
    } else {
        if (tid < v_dim) {
            float v_i = v_ptr[tid];

            float hk_dot = 0.0f;
            #pragma unroll 4
            for (unsigned int j = 0; j < k_dim; j += 4) {
                float h0 = gdn_ms_load_cs<H_IS_F16>(H, (j + 0) * v_dim + tid);
                float h1 = gdn_ms_load_cs<H_IS_F16>(H, (j + 1) * v_dim + tid);
                float h2 = gdn_ms_load_cs<H_IS_F16>(H, (j + 2) * v_dim + tid);
                float h3 = gdn_ms_load_cs<H_IS_F16>(H, (j + 3) * v_dim + tid);
                hk_dot += h0 * smem_k[j] + h1 * smem_k[j + 1]
                        + h2 * smem_k[j + 2] + h3 * smem_k[j + 3];
            }

            float v_new_i = (v_i - g * hk_dot) * bt;

            #pragma unroll 4
            for (unsigned int j = 0; j < k_dim; j += 4) {
                float h0 = gdn_ms_load_cs<H_IS_F16>(H, (j + 0) * v_dim + tid);
                float h1 = gdn_ms_load_cs<H_IS_F16>(H, (j + 1) * v_dim + tid);
                float h2 = gdn_ms_load_cs<H_IS_F16>(H, (j + 2) * v_dim + tid);
                float h3 = gdn_ms_load_cs<H_IS_F16>(H, (j + 3) * v_dim + tid);
                h0 = g * h0 + smem_k[j]     * v_new_i;
                h1 = g * h1 + smem_k[j + 1] * v_new_i;
                h2 = g * h2 + smem_k[j + 2] * v_new_i;
                h3 = g * h3 + smem_k[j + 3] * v_new_i;
                gdn_ms_store_cs<H_IS_F16>(H, (j + 0) * v_dim + tid, h0);
                gdn_ms_store_cs<H_IS_F16>(H, (j + 1) * v_dim + tid, h1);
                gdn_ms_store_cs<H_IS_F16>(H, (j + 2) * v_dim + tid, h2);
                gdn_ms_store_cs<H_IS_F16>(H, (j + 3) * v_dim + tid, h3);
                q_dot += h0 * smem_q[j] + h1 * smem_q[j + 1]
                       + h2 * smem_q[j + 2] + h3 * smem_q[j + 3];
                #ifdef SSM_STATE_NORM_ENABLED
                local_sq += h0 * h0;
                local_sq += h1 * h1;
                local_sq += h2 * h2;
                local_sq += h3 * h3;
                #endif
            }
        }
    }

    #ifdef SSM_STATE_NORM_ENABLED
    {
        float warp_sum = local_sq;
        warp_sum += __shfl_down_sync(0xffffffffu, warp_sum, 16);
        warp_sum += __shfl_down_sync(0xffffffffu, warp_sum, 8);
        warp_sum += __shfl_down_sync(0xffffffffu, warp_sum, 4);
        warp_sum += __shfl_down_sync(0xffffffffu, warp_sum, 2);
        warp_sum += __shfl_down_sync(0xffffffffu, warp_sum, 1);

        unsigned int warp_id = tid / 32;
        unsigned int lane_id = tid % 32;
        if (lane_id == 0) norm_sums[warp_id] = warp_sum;
        __syncthreads();

        if (tid < 4) {
            float s = norm_sums[tid];
            s += __shfl_down_sync(0xfu, s, 2, 4);
            s += __shfl_down_sync(0xfu, s, 1, 4);
            if (tid == 0) norm_sums[0] = s;
        }
        __syncthreads();
        float head_norm_sq = norm_sums[0];

        if (tid < v_dim && head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
            float scale = SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq);
            for (unsigned int j = 0; j < k_dim; j++) {
                float hv = gdn_ms_load_cs<H_IS_F16>(H, j * v_dim + tid);
                gdn_ms_store_cs<H_IS_F16>(H, j * v_dim + tid, hv * scale);
            }
        }
    }
    #endif

    if (tid < v_dim) {
        float inv_sqrt_d = rsqrtf((float)k_dim);
        output[(unsigned long long)seq * v_out_stride + vh * v_dim + tid] = q_dot * inv_sqrt_d;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 3) void gated_delta_rule_decode_f32_multi_seq(
    float* const* __restrict__ h_states,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    unsigned int num_seqs,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int gate_beta_stride,
    unsigned int qk_stride,
    unsigned int v_in_stride,
    unsigned int v_out_stride
) {
    gated_delta_rule_decode_multi_seq_body<false>(
        reinterpret_cast<void* const*>(h_states),
        query, key, value, gate, beta, output,
        num_seqs, num_k_heads, num_v_heads, k_dim, v_dim,
        gate_beta_stride, qk_stride, v_in_stride, v_out_stride);
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 3) void gated_delta_rule_decode_f16_multi_seq(
    __half* const* __restrict__ h_states,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    unsigned int num_seqs,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int gate_beta_stride,
    unsigned int qk_stride,
    unsigned int v_in_stride,
    unsigned int v_out_stride
) {
    gated_delta_rule_decode_multi_seq_body<true>(
        reinterpret_cast<void* const*>(h_states),
        query, key, value, gate, beta, output,
        num_seqs, num_k_heads, num_v_heads, k_dim, v_dim,
        gate_beta_stride, qk_stride, v_in_stride, v_out_stride);
}
