// SPDX-License-Identifier: AGPL-3.0-only
#include "../../common/shared/rms_norm_standard_core.cuh"

// Fused RMS Norm + Gated variant (for Mamba layers).
// out = rms_norm(x) * SiLU(gate)   where SiLU(x) = x * sigmoid(x)
//
// Uses STANDARD weight (no 1+ offset) for Qwen3-VL.
// Note: Qwen3-VL has no Mamba layers, but this is included for completeness
// since the module is shadowed entirely.
extern "C" __global__ void gated_rms_norm(
    const __nv_bfloat16* __restrict__ input,   // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ gate,    // [num_tokens, gate_stride]
    const __nv_bfloat16* __restrict__ weight,  // [hidden_size]
    __nv_bfloat16* __restrict__ output,         // [num_tokens, hidden_size]
    unsigned int hidden_size,
    float eps,
    unsigned int gate_stride
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + token * hidden_size;
    const __nv_bfloat16* g = gate + (unsigned long long)token * gate_stride;
    __nv_bfloat16* out = output + token * hidden_size;

    const unsigned int quad_size = hidden_size / 4;
    const unsigned long long* x64 = (const unsigned long long*)x;

    float x_cache[16];
    float sum_sq = 0.0f;
    unsigned int n_cached = 0;

    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned long long v = x64[i];
        float f0, f1, f2, f3;
        unpack_bf16x2((unsigned int)v, f0, f1);
        unpack_bf16x2((unsigned int)(v >> 32), f2, f3);
        x_cache[n_cached]     = f0;
        x_cache[n_cached + 1] = f1;
        x_cache[n_cached + 2] = f2;
        x_cache[n_cached + 3] = f3;
        n_cached += 4;
        sum_sq += f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    const unsigned long long* g64 = (const unsigned long long*)g;
    const unsigned long long* w64 = (const unsigned long long*)weight;
    unsigned long long* out64 = (unsigned long long*)out;

    unsigned int ci = 0;
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        float f0 = x_cache[ci];
        float f1 = x_cache[ci + 1];
        float f2 = x_cache[ci + 2];
        float f3 = x_cache[ci + 3];
        ci += 4;

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        unpack_bf16x2((unsigned int)wv, w0, w1);
        unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        unpack_bf16x2((unsigned int)gv, g0, g1);
        unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = g0 / (1.0f + __expf(-g0));
        float s1 = g1 / (1.0f + __expf(-g1));
        float s2 = g2 / (1.0f + __expf(-g2));
        float s3 = g3 / (1.0f + __expf(-g3));

        unsigned int lo = pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}

// L2 Normalization (in-place): x[i] = x[i] / sqrt(sum(x^2) + eps)
//
// Used for Q/K normalization.
//
// Grid: (num_heads, num_tokens, 1)
// Block: (min(head_dim, 1024), 1, 1)
extern "C" __global__ void l2_norm_bf16(
    __nv_bfloat16* __restrict__ data,
    unsigned int head_dim,
    float eps,
    unsigned int stride
) {
    unsigned int head = blockIdx.x;
    unsigned int token = blockIdx.y;
    unsigned int tid = threadIdx.x;

    __nv_bfloat16* x = data + (unsigned long long)token * stride + head * head_dim;

    float sum_sq = 0.0f;
    const unsigned int half_size = head_dim / 2;
    const unsigned int* x32 = (const unsigned int*)x;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((head_dim & 1) && tid == 0) {
        float val = __bfloat162float(x[head_dim - 1]);
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float inv_norm = rsqrtf(warp_sums[0] + eps);

    unsigned int* out32 = (unsigned int*)x;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        out32[i] = pack_bf16x2(v0 * inv_norm, v1 * inv_norm);
    }
    if ((head_dim & 1) && tid == 0) {
        float val = __bfloat162float(x[head_dim - 1]);
        x[head_dim - 1] = __float2bfloat16(val * inv_norm);
    }
}

// FP32-input variant: accepts GDN output in FP32 (no BF16 truncation in the
// recurrent path). Gate is still BF16 (from Z projection), weight is BF16,
// output is BF16 (feeds into the BF16 output projection).
extern "C" __global__ void gated_rms_norm_f32_input(
    const float* __restrict__ input,              // [num_tokens, hidden_size] FP32
    const __nv_bfloat16* __restrict__ gate,       // [num_tokens, gate_stride]
    const __nv_bfloat16* __restrict__ weight,     // [hidden_size]
    __nv_bfloat16* __restrict__ output,            // [num_tokens, hidden_size]
    unsigned int hidden_size,
    float eps,
    unsigned int gate_stride,
    unsigned int group_size
) {
    (void)group_size;
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const float* x = input + token * hidden_size;
    const __nv_bfloat16* g = gate + (unsigned long long)token * gate_stride;
    __nv_bfloat16* out = output + token * hidden_size;

    // Pass 1: compute sum of squares (FP32 input — no BF16 unpack needed)
    float sum_sq = 0.0f;

    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float f = x[i];
        sum_sq += f * f;
    }

    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    // Pass 2: Apply normalization + gate
    const unsigned long long* g64 = (const unsigned long long*)g;
    const unsigned long long* w64 = (const unsigned long long*)weight;
    unsigned long long* out64 = (unsigned long long*)out;

    const unsigned int quad_size = hidden_size / 4;
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned int base = i * 4;
        float f0 = x[base];
        float f1 = x[base + 1];
        float f2 = x[base + 2];
        float f3 = x[base + 3];

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        unpack_bf16x2((unsigned int)wv, w0, w1);
        unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        unpack_bf16x2((unsigned int)gv, g0, g1);
        unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = g0 / (1.0f + __expf(-g0));
        float s1 = g1 / (1.0f + __expf(-g1));
        float s2 = g2 / (1.0f + __expf(-g2));
        float s3 = g3 / (1.0f + __expf(-g3));

        unsigned int lo = pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}

