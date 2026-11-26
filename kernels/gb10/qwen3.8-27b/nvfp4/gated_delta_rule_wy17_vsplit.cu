// SPDX-License-Identifier: AGPL-3.0-only

// Atlas WY-Chunkwise Gated Delta Rule — K=17 verify, V-DIM SPLIT variant.
//
// Wave 2 Optimizations (Decode-75 DFlash-27B):
//   1. Eliminates the 208-byte per-thread local-memory stack frame across both
//      `gated_delta_rule_wy17_vsplit` and `gated_delta_rule_wy17_lazy_vsplit`
//      by precomputing CTA-wide scalar WY gate products (`s_lead[17]`,
//      `s_wy_kd[136]`) in shared memory and fully unrolling the K=17 WY
//      recurrence in registers (`0 B stack, 0 spills`).
//   2. Collapses the 136 sequential `atlas_block_reduce_sum` calls in `kd_flat`
//      from 408 `__syncthreads()` barriers down to 2 `__syncthreads()` barriers,
//      removing the per-split `kd_flat` recompute overhead while preserving the
//      exact `(w0 + w2) + (w1 + w3)` FP32 reduction tree.

#include <cuda_bf16.h>
#include "../../common/gdn_reduce.cuh"
#define BLOCK_SIZE 128
#define K_TOKENS 17
#define NUM_PAIRS (K_TOKENS * (K_TOKENS - 1) / 2)

__device__ __forceinline__ bool wy17vs_is_checkpoint(int s, unsigned int lazy_j) {
    if (lazy_j <= 1u) return true;
    return (s == 0) || (s == K_TOKENS - 2) || (((unsigned)(s + 1) % lazy_j) == 0u);
}

__device__ __forceinline__ void gated_delta_rule_wy17_vsplit_impl(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_inter_base,
    unsigned int inter_stride_floats,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int v_split,
    unsigned int lazy_j
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int zsplit = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size || zsplit >= v_split) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31u;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;

    const unsigned int v_chunk = (v_dim + v_split - 1) / v_split;
    const unsigned int v_off = zsplit * v_chunk;
    if (v_off >= v_dim) return;
    const unsigned int vcol = v_off + tid;
    const bool active = (tid < v_chunk) && (vcol < v_dim);

    float* H = h_state + ((b * num_v_heads + vh) * hv);
    float* Hi_base = h_state_inter_base + ((b * num_v_heads + vh) * hv);

    __shared__ float sk[K_TOKENS][128];
    __shared__ float sq[K_TOKENS][128];
    __shared__ float sg[K_TOKENS];
    __shared__ float sbt[K_TOKENS];
    __shared__ float s_lead[K_TOKENS];
    __shared__ float s_wy_kd[NUM_PAIRS];
    __shared__ float smem_pair_warps[NUM_PAIRS * 4];

    if (tid < k_dim) {
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            const __nv_bfloat16* q_t = query + (b * K_TOKENS + t) * qk_stride + kh * k_dim;
            const __nv_bfloat16* k_t = key   + (b * K_TOKENS + t) * qk_stride + kh * k_dim;
            sq[t][tid] = (float)q_t[tid];
            sk[t][tid] = (float)k_t[tid];
        }
    }
    if (tid < K_TOKENS) {
        float g_raw = gate[(b * K_TOKENS + tid) * gb_stride + vh];
        sg[tid] = fminf(fmaxf(g_raw, 0.0f), 1.0f);
        sbt[tid] = beta[(b * K_TOKENS + tid) * gb_stride + vh];
    }
    __syncthreads();

    // ── kd_flat: 136 inter-token k-dots in 2 barriers (was 408 barriers) ──
    #pragma unroll
    for (int t = 1; t < K_TOKENS; t++) {
        #pragma unroll
        for (int s = 0; s < t; s++) {
            const int pair_idx = t * (t - 1) / 2 + s;
            float p = (tid < k_dim) ? sk[t][tid] * sk[s][tid] : 0.0f;
            float wsum = atlas_warp_reduce_sum(p);
            if (lane_id == 0u) {
                smem_pair_warps[pair_idx * 4 + warp_id] = wsum;
            }
        }
    }
    if (tid < K_TOKENS) {
        float lead_prod = 1.0f;
        for (unsigned int u = 0; u < tid; u++) {
            lead_prod *= sg[u];
        }
        s_lead[tid] = lead_prod;
    }
    __syncthreads();

    for (int pair_idx = (int)tid; pair_idx < NUM_PAIRS; pair_idx += BLOCK_SIZE) {
        float w0 = smem_pair_warps[pair_idx * 4 + 0];
        float w1 = smem_pair_warps[pair_idx * 4 + 1];
        float w2 = smem_pair_warps[pair_idx * 4 + 2];
        float w3 = smem_pair_warps[pair_idx * 4 + 3];
        float kd = __fadd_rn(__fadd_rn(w0, w2), __fadd_rn(w1, w3));

        int t = 1;
        while ((t * (t + 1)) / 2 <= pair_idx) {
            t++;
        }
        int s = pair_idx - (t * (t - 1)) / 2;
        float gprod = 1.0f;
        for (int u = s + 1; u < t; u++) {
            gprod *= sg[u];
        }
        s_wy_kd[pair_idx] = gprod * kd;
    }
    __syncthreads();

    if (active) {
        // PASS 1: hk[t] = H · k_t (H column `vcol`).
        float hk[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) hk[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + vcol];
            float h1 = H[(j + 1) * v_dim + vcol];
            float h2 = H[(j + 2) * v_dim + vcol];
            float h3 = H[(j + 3) * v_dim + vcol];
            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                hk[t] += h0 * sk[t][j + 0] + h1 * sk[t][j + 1]
                       + h2 * sk[t][j + 2] + h3 * sk[t][j + 3];
            }
        }

        // WY correction (fully unrolled in registers, 0 stack frame).
        float vn[K_TOKENS];
        {
            const __nv_bfloat16* v_0 = value + (b * K_TOKENS + 0) * v_stride + vh * v_dim;
            vn[0] = ((float)v_0[vcol] - sg[0] * hk[0]) * sbt[0];
        }
        #pragma unroll
        for (int t = 1; t < K_TOKENS; t++) {
            float corrected = s_lead[t] * hk[t];
            #pragma unroll
            for (int s = 0; s < t; s++) {
                corrected += s_wy_kd[t * (t - 1) / 2 + s] * vn[s];
            }
            const __nv_bfloat16* v_t = value + (b * K_TOKENS + t) * v_stride + vh * v_dim;
            vn[t] = ((float)v_t[vcol] - sg[t] * corrected) * sbt[t];
        }

        // PASS 2: apply K state updates, write intermediates + final H + qd.
        float qd[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) qd[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + vcol];
            float h1 = H[(j + 1) * v_dim + vcol];
            float h2 = H[(j + 2) * v_dim + vcol];
            float h3 = H[(j + 3) * v_dim + vcol];

            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                h0 = sg[t] * h0 + sk[t][j + 0] * vn[t];
                h1 = sg[t] * h1 + sk[t][j + 1] * vn[t];
                h2 = sg[t] * h2 + sk[t][j + 2] * vn[t];
                h3 = sg[t] * h3 + sk[t][j + 3] * vn[t];
                if (t < K_TOKENS - 1) {
                    if (wy17vs_is_checkpoint(t, lazy_j)) {
                        float* Hi_t = Hi_base + t * inter_stride_floats;
                        Hi_t[(j + 0) * v_dim + vcol] = h0;
                        Hi_t[(j + 1) * v_dim + vcol] = h1;
                        Hi_t[(j + 2) * v_dim + vcol] = h2;
                        Hi_t[(j + 3) * v_dim + vcol] = h3;
                    }
                } else {
                    H[(j + 0) * v_dim + vcol] = h0;
                    H[(j + 1) * v_dim + vcol] = h1;
                    H[(j + 2) * v_dim + vcol] = h2;
                    H[(j + 3) * v_dim + vcol] = h3;
                }
                qd[t] += h0 * sq[t][j + 0] + h1 * sq[t][j + 1]
                       + h2 * sq[t][j + 2] + h3 * sq[t][j + 3];
            }
        }

        float s = rsqrtf((float)k_dim);
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            output[((b * K_TOKENS + t) * num_v_heads + vh) * v_dim + vcol] =
                __float2bfloat16(qd[t] * s);
        }
    }
}

extern "C" __global__ void gated_delta_rule_wy17_vsplit(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_inter_base,
    unsigned int inter_stride_floats,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int v_split
) {
    gated_delta_rule_wy17_vsplit_impl(h_state, query, key, value, gate, beta, output,
        h_state_inter_base, inter_stride_floats, batch_size, num_k_heads,
        num_v_heads, k_dim, v_dim, qk_stride, v_stride, gb_stride, v_split, /*lazy_j=*/1u);
}

extern "C" __global__ void gated_delta_rule_wy17_lazy_vsplit(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_inter_base,
    unsigned int inter_stride_floats,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int v_split,
    unsigned int lazy_j
) {
    gated_delta_rule_wy17_vsplit_impl(h_state, query, key, value, gate, beta, output,
        h_state_inter_base, inter_stride_floats, batch_size, num_k_heads,
        num_v_heads, k_dim, v_dim, qk_stride, v_stride, gb_stride, v_split, lazy_j);
}
