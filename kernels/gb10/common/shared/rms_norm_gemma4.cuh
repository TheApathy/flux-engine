// SPDX-License-Identifier: AGPL-3.0-only
#include "rms_norm_standard_core.cuh"

// ═══════════════════════════════════════════════════════════════════
// FP32 residual variants — absolute formula `out = x * rms * w`
// (matching HF `Gemma4RMSNorm`, no offset-from-1).
//
// Purpose: keep the residual stream in FP32 across all 60 layers of
// Gemma-4-31B so cumulative early-layer `layer_scalar` values (L0=0.089,
// L1=0.065 ...) don't underflow BF16 and collapse activations into
// repetition-attractor tokens. Qwen3-Next has equivalent `_f32` variants
// in `kernels/gb10/common/rms_norm.cu` but with the offset formula
// `out = x * rms * (1+w)`, which does not apply to Gemma-4.
//
// These kernels are named `_f32_abs` so the Rust dispatch can pick them
// up for Gemma-4 via `gpu.kernel("norm", "<name>_f32_abs")` in
// `crates/spark-model/src/layers/qwen3_attention/mod.rs`.
// ═══════════════════════════════════════════════════════════════════

// rms_norm_residual but: input=FP32, residual=FP32, output=BF16, formula=absolute.
extern "C" __global__ void rms_norm_residual_f32_abs(
    const float* __restrict__ input,             // [num_tokens, hidden_size] FP32
    const __nv_bfloat16* __restrict__ weight,    // [hidden_size]
    __nv_bfloat16* __restrict__ output,           // [num_tokens, hidden_size] BF16
    float* __restrict__ residual,                 // [num_tokens, hidden_size] FP32
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const float* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    float* res = residual + token * hidden_size;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
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

    const unsigned int half_size = hidden_size / 2;
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float xv0 = x[base];
        float xv1 = x[base + 1];
        float wv0, wv1;
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);  // ABSOLUTE
        res[base]     = xv0;
        res[base + 1] = xv1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = x[hidden_size - 1];
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
        res[hidden_size - 1] = val;
    }
}

// residual_add_rms_norm but: hidden=FP32, residual=FP32, src=BF16, output=BF16, formula=absolute.
extern "C" __global__ void residual_add_rms_norm_f32_abs(
    float* __restrict__ hidden,              // [num_tokens, hidden_size] FP32 in/out
    const __nv_bfloat16* __restrict__ src,   // [num_tokens, hidden_size] BF16 layer output
    const __nv_bfloat16* __restrict__ weight, // [hidden_size]
    __nv_bfloat16* __restrict__ output,       // [num_tokens, hidden_size] BF16
    float* __restrict__ residual,             // [num_tokens, hidden_size] FP32
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    float* h = hidden + token * hidden_size;
    const __nv_bfloat16* s = src + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    float* res = residual + token * hidden_size;

    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    const unsigned int* s32 = (const unsigned int*)s;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float sv0, sv1;
        unpack_bf16x2(s32[i], sv0, sv1);
        float new0 = h[base]     + sv0;
        float new1 = h[base + 1] + sv1;
        h[base]     = new0;
        h[base + 1] = new1;
        sum_sq += new0 * new0 + new1 * new1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float hv = h[hidden_size - 1];
        float sv = __bfloat162float(s[hidden_size - 1]);
        float nv = hv + sv;
        h[hidden_size - 1] = nv;
        sum_sq += nv * nv;
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

    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float xv0 = h[base];
        float xv1 = h[base + 1];
        float wv0, wv1;
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);  // ABSOLUTE
        res[base]     = xv0;
        res[base + 1] = xv1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = h[hidden_size - 1];
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
        res[hidden_size - 1] = val;
    }
}

// Plain rms_norm variant but reading FP32 input instead of BF16.
// Used for the final norm before LM head when hidden is kept in FP32
// for the Gemma-4 residual stream. Absolute formula.
extern "C" __global__ void rms_norm_f32(
    const float* __restrict__ input,             // [num_tokens, hidden_size] FP32
    const __nv_bfloat16* __restrict__ weight,    // [hidden_size]
    __nv_bfloat16* __restrict__ output,           // [num_tokens, hidden_size] BF16
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const float* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
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

    const unsigned int half_size = hidden_size / 2;
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float xv0 = x[base];
        float xv1 = x[base + 1];
        float wv0, wv1;
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);  // ABSOLUTE
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = x[hidden_size - 1];
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
    }
}

// Simple FP32 += BF16 residual accumulator for post-FFN residual add.
// Used by `residual_add_k` when FP32 residual is active for Gemma-4.
extern "C" __global__ void f32_residual_add(
    float* __restrict__ residual,
    const __nv_bfloat16* __restrict__ src,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        residual[i] += __bfloat162float(src[i]);
    }
}

// NOTE: `bf16_to_f32` is available via the common `residual_add` module
// (kernels/gb10/common/residual_add.cu) which is NOT shadowed by Gemma-4.
// The Rust-side lookup `gpu.kernel("residual_add", "bf16_to_f32")` therefore
// resolves correctly without needing a Gemma-4 duplicate.
