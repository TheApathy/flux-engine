// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused Gate Logit GEMV + Top-K Selection + Softmax.
//
// Replaces 2 separate kernels (gate GEMV + topk) with 1 fused kernel.
// Eliminates 48 kernel launches per decode step (one per MoE layer).
//
// Single block of 256 threads — strided over up to MAX_EXPERTS (512) experts
// to compute gate logits, then cooperative top-K selection via warp-level max
// reduction + full-expert softmax.
//
// Gate weight is NVFP4 (runtime-quantized from BF16 at model load).
//
// Grid: (1, 1, 1)  Block: (256, 1, 1)  Dynamic smem: K * sizeof(__nv_bfloat16) bytes

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 256
#define MAX_EXPERTS 512
#define MAX_TOP_K 32
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_GATE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void moe_gate_topk_fused(
    const __nv_bfloat16* __restrict__ A,         // [K] input activation
    const unsigned char* __restrict__ B_packed,   // [num_experts, K/2] NVFP4 gate weight
    const unsigned char* __restrict__ B_scale,    // [num_experts, K/GROUP_SIZE] FP8 scales
    float scale2,                                 // per-tensor scale
    unsigned int* __restrict__ expert_indices,    // [top_k] output
    float* __restrict__ expert_weights,           // [top_k] output
    unsigned int num_experts,
    unsigned int K,
    unsigned int top_k,
    unsigned int normalize                        // 1 = normalize softmax to sum=1
) {
    extern __shared__ __nv_bfloat16 s_A[];
    __shared__ float s_lut[16];
    __shared__ float s_vals[MAX_EXPERTS];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int num_warps = BLOCK_SIZE / WARP_SIZE;

    const unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;
    const unsigned int actual_k = top_k < actual_n
        ? (top_k < MAX_TOP_K ? top_k : MAX_TOP_K)
        : (actual_n < MAX_TOP_K ? actual_n : MAX_TOP_K);
    if (actual_k == 0) return;

    // Load E2M1 LUT into shared memory
    if (tid < 16) s_lut[tid] = E2M1_LUT_GATE[tid];

    // Cooperatively load A into shared memory
    for (unsigned int i = tid; i < K; i += BLOCK_SIZE) {
        s_A[i] = A[i];
    }
    __syncthreads();

    // Phase 1: Cooperatively compute each expert's gate logit (strided up to MAX_EXPERTS).
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    for (unsigned int e = tid; e < actual_n; e += BLOCK_SIZE) {
        float acc = 0.0f;

        for (unsigned int k8 = 0; k8 < K8; k8++) {
            unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)e * half_K + k8 * 4);
            unsigned int sg = (k8 * 8) / GROUP_SIZE;
            unsigned char sb = B_scale[(unsigned long long)e * num_groups + sg];
            __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
            float sc = (float)fp8 * scale2;

            #pragma unroll
            for (int b = 0; b < 4; b++) {
                unsigned char bv = (packed4 >> (b * 8)) & 0xFF;
                float wl = s_lut[bv & 0xF] * sc, wh = s_lut[bv >> 4] * sc;
                float al = __bfloat162float(s_A[k8 * 8 + b * 2]);
                float ah = __bfloat162float(s_A[k8 * 8 + b * 2 + 1]);
                acc += al * wl + ah * wh;
            }
        }

        s_vals[e] = acc;
    }
    __syncthreads();

    // Phase 2: Parallel top-K via iterative max reduction.
    for (unsigned int t = 0; t < actual_k; t++) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_vals[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            unsigned int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        if (lane == 0) {
            s_warp_val[warp_id] = local_max;
            s_warp_idx[warp_id] = local_idx;
        }
        __syncthreads();

        if (tid == 0) {
            float best_val = s_warp_val[0];
            unsigned int best_idx = s_warp_idx[0];
            for (unsigned int w = 1; w < num_warps; w++) {
                if (s_warp_val[w] > best_val) {
                    best_val = s_warp_val[w];
                    best_idx = s_warp_idx[w];
                }
            }
            s_top_vals[t] = best_val;
            s_top_idxs[t] = best_idx;
            s_vals[best_idx] = -1e30f;
        }
        __syncthreads();
    }

    // Phase 3: Compute softmax over ALL experts, then extract top-K weights
    // (matches moe_topk.cu: full softmax denominator; renormalize when normalize != 0).
    const float global_max = s_top_vals[0];

    {
        float local_sum = 0.0f;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            local_sum += __expf(s_vals[i] - global_max);
        }
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0) s_warp_val[warp_id] = local_sum;
        __syncthreads();

        if (tid == 0) {
            float total = 0.0f;
            for (unsigned int w = 0; w < num_warps; w++) {
                total += s_warp_val[w];
            }
            // Add back top-K contributions (their s_vals[] slots were overwritten with -1e30f)
            for (unsigned int t = 0; t < actual_k; t++) {
                total += __expf(s_top_vals[t] - global_max);
            }
            s_warp_val[0] = total;
        }
        __syncthreads();
    }

    const float exp_sum = s_warp_val[0];

    if (tid == 0) {
        for (unsigned int t = 0; t < actual_k; t++) {
            expert_indices[t] = s_top_idxs[t];
            float softmax_weight = __expf(s_top_vals[t] - global_max) / exp_sum;

            if (normalize) {
                s_top_vals[t] = softmax_weight;
            } else {
                expert_weights[t] = softmax_weight;
            }
        }

        if (normalize) {
            float topk_sum = 0.0f;
            for (unsigned int t = 0; t < actual_k; t++) {
                topk_sum += s_top_vals[t];
            }
            for (unsigned int t = 0; t < actual_k; t++) {
                expert_weights[t] = s_top_vals[t] / topk_sum;
            }
        }
    }
}
