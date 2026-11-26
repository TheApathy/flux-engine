// SPDX-License-Identifier: AGPL-3.0-only

// FP16 h-state twin of `gated_delta_rule_wy2` — stage 2 of `ATLAS_SSM_H_FP16`.
//
// Derived from `gated_delta_rule_wy2` (`gated_delta_rule_wy.cu`): arithmetic
// stays FP32 in registers while the h-state and rollback intermediate are
// `__half` in memory, loaded via `__half2float` and stored via `gdn_f16_store`
// (see gdn_f16_state.cuh), plus an extra `state_is_table` argument.
//
// PER-TOKEN ROUND-TRIP. Each state update is rounded to FP16 BEFORE it is
// stored and before it is carried into the next token or into the q-dot
// accumulation, so the value the forward chain uses is bit-for-bit the value a
// rollback restores from the checkpoint.
//
// This is the K=2 MTP-verify FP16 twin (dispatched via `ops::gdn_decode_wy2_f16`).
// Under FP16, the contiguous (`state_is_table == 0`) form should only be
// launched at `batch_size == 1` because consecutive pool slots are spaced by
// FP32 `h_state_bytes` rather than the dense FP16 footprint.

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#include "gdn_f16_state.cuh"
#define BLOCK_SIZE 128

// Reduction primitives provided by gdn_reduce.cuh:
//   atlas_warp_reduce_sum, atlas_block_reduce_sum
// Shared across the WY kernels for deterministic k_i^T @ k_j FP32 reductions.
// Note: unlike gated_delta_rule_decode, wy2_f16 omits SSM_STATE_NORM_ENABLED.

// ============================================================
// WY2: 2-pass K=2 verification (replaces chunk2)
// ============================================================
extern "C" __global__ void gated_delta_rule_wy2_f16(
    __half* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    __half* __restrict__ h_state_intermediate,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    // 0 = the two state args are CONTIGUOUS bases indexed by
    //     (b*num_v_heads+vh) (valid at batch_size == 1);
    // 1 = they are device POINTER TABLES of `batch_size` entries, one per
    //     sequence (required for multi-sequence batched verify so sequence 1's
    //     intermediate does not alias sequence 0's intermediate pool stride).
    unsigned int state_is_table
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    const unsigned int hv_size = k_dim * v_dim;
    const unsigned long long head_off = (unsigned long long)vh * hv_size;
    const unsigned long long flat_off =
        (unsigned long long)(b * num_v_heads + vh) * hv_size;
    __half* H = state_is_table ? ((__half* const*)h_state)[b] + head_off
                              : h_state + flat_off;
    __half* H_inter = state_is_table
                         ? ((__half* const*)h_state_intermediate)[b] + head_off
                         : h_state_intermediate + flat_off;

    // Token pointers
    const __nv_bfloat16* q0 = query + (b * 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k0 = key   + (b * 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v0 = value + (b * 2) * v_stride  + vh * v_dim;
    // Gate clamp matches gated_delta_rule_wy2 and per-token gated_delta_rule_decode ([0.0f, 1.0f]).
    const float g0 = fminf(fmaxf(gate[(b * 2) * gb_stride + vh], 0.0f), 1.0f);
    const float bt0 = beta[(b * 2) * gb_stride + vh];

    const __nv_bfloat16* q1 = query + (b * 2 + 1) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k1 = key   + (b * 2 + 1) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v1 = value + (b * 2 + 1) * v_stride  + vh * v_dim;
    const float g1 = fminf(fmaxf(gate[(b * 2 + 1) * gb_stride + vh], 0.0f), 1.0f);
    const float bt1 = beta[(b * 2 + 1) * gb_stride + vh];

    __shared__ float smem_k0[128], smem_q0[128];
    __shared__ float smem_k1[128], smem_q1[128];
    __shared__ float smem_kdot;
    __shared__ float smem_warp[4];

    if (tid < k_dim) {
        smem_k0[tid] = (float)k0[tid]; smem_q0[tid] = (float)q0[tid];
        smem_k1[tid] = (float)k1[tid]; smem_q1[tid] = (float)q1[tid];
    }
    __syncthreads();

    // ── Compute kdot = k_1^T @ k_0 ──
    {
        float partial = (tid < k_dim) ? smem_k1[tid] * smem_k0[tid] : 0.0f;
        float result = atlas_block_reduce_sum(partial, smem_warp, tid);
        if (tid == 0) smem_kdot = result;
    }
    __syncthreads();

    if (tid < v_dim) {
        float vi0 = (float)v0[tid];
        float vi1 = (float)v1[tid];
        float kdot_10 = smem_kdot;

        // ── PASS 1: Read H once, compute hk_prev[0] and hk_prev[1] ──
        float hk0 = 0.0f, hk1_prev = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = __half2float(H[(j+0) * v_dim + tid]);
            float h1 = __half2float(H[(j+1) * v_dim + tid]);
            float h2 = __half2float(H[(j+2) * v_dim + tid]);
            float h3 = __half2float(H[(j+3) * v_dim + tid]);
            hk0      += h0*smem_k0[j] + h1*smem_k0[j+1] + h2*smem_k0[j+2] + h3*smem_k0[j+3];
            hk1_prev += h0*smem_k1[j] + h1*smem_k1[j+1] + h2*smem_k1[j+2] + h3*smem_k1[j+3];
        }

        // ── WY Correction ──
        float v_new_0 = (vi0 - g0 * hk0) * bt0;
        float hk1_corr = g0 * hk1_prev + kdot_10 * v_new_0;
        float v_new_1 = (vi1 - g1 * hk1_corr) * bt1;

        // ── PASS 2: Read H once, apply updates, write intermediates + final ──
        float q0_dot = 0.0f, q1_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = __half2float(H[(j+0) * v_dim + tid]);
            float h1 = __half2float(H[(j+1) * v_dim + tid]);
            float h2 = __half2float(H[(j+2) * v_dim + tid]);
            float h3 = __half2float(H[(j+3) * v_dim + tid]);

            // Token 0: H_1 = g0*H + k0 ⊗ v_new_0
            h0 = g0*h0 + smem_k0[j]  *v_new_0;
            h1 = g0*h1 + smem_k0[j+1]*v_new_0;
            h2 = g0*h2 + smem_k0[j+2]*v_new_0;
            h3 = g0*h3 + smem_k0[j+3]*v_new_0;
            h0=__half2float(gdn_f16_store(h0)); h1=__half2float(gdn_f16_store(h1));
            H_inter[(j+0)*v_dim+tid]=gdn_f16_store(h0); H_inter[(j+1)*v_dim+tid]=gdn_f16_store(h1);
            h2=__half2float(gdn_f16_store(h2)); h3=__half2float(gdn_f16_store(h3));
            H_inter[(j+2)*v_dim+tid]=gdn_f16_store(h2); H_inter[(j+3)*v_dim+tid]=gdn_f16_store(h3);
            q0_dot += h0*smem_q0[j] + h1*smem_q0[j+1] + h2*smem_q0[j+2] + h3*smem_q0[j+3];

            // Token 1: H_2 = g1*H_1 + k1 ⊗ v_new_1
            h0 = g1*h0 + smem_k1[j]  *v_new_1;
            h1 = g1*h1 + smem_k1[j+1]*v_new_1;
            h2 = g1*h2 + smem_k1[j+2]*v_new_1;
            h3 = g1*h3 + smem_k1[j+3]*v_new_1;
            h0=__half2float(gdn_f16_store(h0)); h1=__half2float(gdn_f16_store(h1));
            H[(j+0)*v_dim+tid]=gdn_f16_store(h0); H[(j+1)*v_dim+tid]=gdn_f16_store(h1);
            h2=__half2float(gdn_f16_store(h2)); h3=__half2float(gdn_f16_store(h3));
            H[(j+2)*v_dim+tid]=gdn_f16_store(h2); H[(j+3)*v_dim+tid]=gdn_f16_store(h3);
            q1_dot += h0*smem_q1[j] + h1*smem_q1[j+1] + h2*smem_q1[j+2] + h3*smem_q1[j+3];
        }

        float inv_sqrt_d = rsqrtf((float)k_dim);
        unsigned int out0 = (b * 2 * num_v_heads + vh) * v_dim;
        unsigned int out1 = ((b * 2 + 1) * num_v_heads + vh) * v_dim;
        output[out0 + tid] = __float2bfloat16(q0_dot * inv_sqrt_d);
        output[out1 + tid] = __float2bfloat16(q1_dot * inv_sqrt_d);
    }
}

