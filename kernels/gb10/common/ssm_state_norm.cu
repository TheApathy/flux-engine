// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused SSM State Normalization kernel.
//
// Normalizes the Frobenius norm of each head's h_state matrix across ALL
// SSM layers in a single kernel launch to prevent catastrophic state
// explosion during long chunked prefill sequences.
//
// Without normalization, SSM h_state grows unbounded during prefill
// (which lacks the per-token norm clamping of the decode GDN kernel).
// After 30K+ tokens of chunked prefill, the state saturates and the
// model "forgets" recent tokens — the SSM forgetting bug (Stuffed Mamba).
//
// Called between prefill chunks (every 8K tokens) to keep state bounded.
//
// State layout per layer: [num_heads, k_dim, v_dim] FP32 (v_dim contiguous)
// Pointer table: [num_layers] pointers to each layer's h_state
//
// Grid: (num_heads, num_layers, 1)   — fused across all SSM layers
// Block: (v_dim, 1, 1)              — typically 128 threads

#include <cuda_bf16.h>

// Softer clamp for inter-chunk prefill: allows h_state to carry full context
// between chunks without catastrophic loss. The decode GDN kernel has its own
// per-token norm clamping (50 model-specific, 1000 common) to prevent explosion
// during generation. This only affects the prefill→prefill boundary.
// Old value (50.0) destroyed ~50% of context at chunk boundaries, causing
// the model to output </think> immediately after >8K token chunked prefill.
#define MAX_NORM 200.0f

extern "C" __global__ void ssm_state_clamp_norm_fused(
    float** __restrict__ h_state_ptrs,  // [num_layers] device pointers
    unsigned int num_heads,
    unsigned int k_dim,
    unsigned int v_dim
) {
    const unsigned int head = blockIdx.x;
    const unsigned int layer = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (head >= num_heads) return;  // block-uniform

    float* H = h_state_ptrs[layer] + (unsigned long long)head * k_dim * v_dim;

    // Compute per-column sum of squares (each thread strides over v_dim columns)
    float local_sq = 0.0f;
    for (unsigned int col = tid; col < v_dim; col += blockDim.x) {
        for (unsigned int j = 0; j < k_dim; j++) {
            float v = H[(unsigned long long)j * v_dim + col];
            local_sq += v * v;
        }
    }

    // Block-wide reduction to get full head Frobenius norm²
    float warp_sum = local_sq;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        warp_sum += __shfl_down_sync(0xffffffffu, warp_sum, off);
    }

    __shared__ float norm_sums[32];
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31u;
    const unsigned int num_warps = (blockDim.x + 31u) >> 5;
    if (lane_id == 0u) norm_sums[warp_id] = warp_sum;
    __syncthreads();

    if (warp_id == 0u) {
        float s = (lane_id < num_warps) ? norm_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s += __shfl_down_sync(0xffffffffu, s, off);
        }
        if (lane_id == 0u) norm_sums[0] = s;
    }
    __syncthreads();
    const float head_norm_sq = norm_sums[0];

    // Clamp: if ||h[head]||_F > MAX_NORM, scale down
    if (head_norm_sq > MAX_NORM * MAX_NORM) {
        const float scale = MAX_NORM * rsqrtf(head_norm_sq);
        for (unsigned int col = tid; col < v_dim; col += blockDim.x) {
            for (unsigned int j = 0; j < k_dim; j++) {
                H[(unsigned long long)j * v_dim + col] *= scale;
            }
        }
    }
}
