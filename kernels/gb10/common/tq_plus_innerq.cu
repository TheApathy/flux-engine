// SPDX-License-Identifier: AGPL-3.0-only
// Upstream attribution: Incorporates TurboQuant+ / Sparse-V / two-sided WHT
// components derived from TheTom/llama-cpp-turboquant (MIT License) and
// TheTom/turboquant_plus (Apache-2.0 License). See docs/CITATIONS.md and
// THIRD_PARTY_NOTICES.md for full upstream copyright and license notices.
//
// TurboQuant+ InnerQ — device-side state definitions + host-side calibration
// controller.
//
// Header: tq_plus_innerq.cuh (declarations).
// See docs/CITATIONS.md and THIRD_PARTY_NOTICES.md for prior-art chain.

#include "tq_plus_innerq.cuh"
#include <cmath>

namespace tq_plus {

// Storage for InnerQ per-channel scales. Identity (1.0) by default; finalize
// uploads the calibrated values via cudaMemcpyToSymbol.
__device__ float d_innerq_scale[INNERQ_MAX_CHANNELS] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
};
__device__ float d_innerq_scale_inv[INNERQ_MAX_CHANNELS] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
};
__device__ float d_innerq_sq_accum[INNERQ_MAX_CHANNELS] = {0};
__device__ int   d_innerq_count = 0;
__device__ int   d_innerq_active = 0;
__device__ int   d_innerq_calibrating = 0;

// Host-side controller. Call once at startup with target_tokens > 0 to reset
// calibration accumulators and set d_innerq_calibrating = 1. Note: target_tokens
// and strength are validated here but not stored on device; the host caller
// polls d_innerq_count >= target_tokens and passes group_size & strength to
// turbo_innerq_finalize.
//
// strength: exponent on the rms-ratio scale (0.5 = sqrt / geometric mean,
//           1.0 = full equalization). Must be in (0, 1]; rejected otherwise.
// Returns 0 on success, positive on invalid argument, negative on CUDA error.
extern "C" int turbo_innerq_start_calibration(int target_tokens, float strength) {
    if (target_tokens <= 0) return 1;
    if (!(strength > 0.0f && strength <= 1.0f)) return 2;

    int zero = 0, one = 1;
    float zeros[INNERQ_MAX_CHANNELS] = {0};
    if (cudaMemcpyToSymbol(d_innerq_sq_accum, zeros, sizeof(zeros)) != cudaSuccess) return -1;
    if (cudaMemcpyToSymbol(d_innerq_count, &zero, sizeof(int)) != cudaSuccess) return -1;
    if (cudaMemcpyToSymbol(d_innerq_active, &zero, sizeof(int)) != cudaSuccess) return -1;
    if (cudaMemcpyToSymbol(d_innerq_calibrating, &one, sizeof(int)) != cudaSuccess) return -1;
    return 0;
}

// Host-side finalize. Reads sq_accum + count from device, computes per-channel
// RMS, scale[i] = (mean_rms/rms[i])^strength clamped to [0.5, 2.0].
// Auto-disables if max_ratio < 1.2 && min_ratio > (1.0 / 1.2) (all channels
// already balanced within +/-20%).
// Returns 1 if scales were activated, 0 if auto-disabled or count <= 0,
// negative on invalid arguments or CUDA error.
extern "C" int turbo_innerq_finalize(int group_size, float strength) {
    if (group_size <= 0 || group_size > INNERQ_MAX_CHANNELS
        || !(strength > 0.0f && strength <= 1.0f)) {
        return -1;
    }

    float sq_accum[INNERQ_MAX_CHANNELS] = {0};
    int count = 0;
    if (cudaMemcpyFromSymbol(sq_accum, d_innerq_sq_accum, (size_t)group_size * sizeof(float)) != cudaSuccess) {
        return -2;
    }
    if (cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(int)) != cudaSuccess) {
        return -2;
    }

    if (count <= 0) {
        int zero = 0;
        if (cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int)) != cudaSuccess) {
            return -2;
        }
        return 0;
    }

    float rms[INNERQ_MAX_CHANNELS];
    float mean_rms = 0.0f;
    for (int i = 0; i < group_size; i++) {
        rms[i] = sqrtf(sq_accum[i] / (float)count);
        mean_rms += rms[i];
    }
    mean_rms /= (float)group_size;

    float scale[INNERQ_MAX_CHANNELS];
    float scale_inv[INNERQ_MAX_CHANNELS];
    float max_ratio = 0.0f, min_ratio = 1e30f;
    for (int i = 0; i < group_size; i++) {
        float ratio = (rms[i] > 1e-10f) ? (mean_rms / rms[i]) : 1.0f;
        float s = powf(ratio, strength);
        if (s < 0.5f) s = 0.5f;
        if (s > 2.0f) s = 2.0f;
        scale[i] = s;
        scale_inv[i] = 1.0f / s;
        if (ratio > max_ratio) max_ratio = ratio;
        if (ratio < min_ratio) min_ratio = ratio;
    }

    int zero = 0, one = 1;
    if (cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int)) != cudaSuccess) {
        return -2;
    }
    if (max_ratio < 1.2f && min_ratio > (1.0f / 1.2f)) {
        return 0;  // auto-disabled
    }
    if (cudaMemcpyToSymbol(d_innerq_scale, scale, (size_t)group_size * sizeof(float)) != cudaSuccess) {
        return -2;
    }
    if (cudaMemcpyToSymbol(d_innerq_scale_inv, scale_inv, (size_t)group_size * sizeof(float)) != cudaSuccess) {
        return -2;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return -2;
    }
    if (cudaMemcpyToSymbol(d_innerq_active, &one, sizeof(int)) != cudaSuccess) {
        return -2;
    }
    return 1;
}

}  // namespace tq_plus

// ── Single-TU InnerQ application kernels (same PTX module as d_innerq_*) ──
//
// Both kernels fire AFTER WHT on their respective tensors (post-WHT Q and
// post-WHT K) so that <S^{-1}(W Q), S(W K)> = <W Q, W K> = <Q, K>.
// Defining them in this TU avoids cross-PTX `.extern .global` linkage failures
// under per-file `nvcc --ptx` + `cuModuleLoadData`.

// Apply post-WHT scale_inv to Q (one bf16 vector per warp, hd=128).
// Grid: (num_q_heads, 1, 1)   Block: (32, 1, 1)
extern "C" __global__ void tq_plus_innerq_apply_q(
    __nv_bfloat16* __restrict__ data,
    const unsigned int head_dim
) {
    if (tq_plus::d_innerq_active == 0) return;
    if (head_dim != INNERQ_MAX_CHANNELS) return;

    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (lane >= 32) return;

    __nv_bfloat16* head_data = data + (unsigned long long)head * head_dim;

    #pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        float v = __bfloat162float(head_data[ch]);
        v *= tq_plus::d_innerq_scale_inv[ch];
        head_data[ch] = __float2bfloat16(v);
    }
}

// Apply post-WHT scale to K and accumulate post-WHT K^2 stats during calibration.
// Grid: (num_kv_heads, 1, 1)   Block: (32, 1, 1)
extern "C" __global__ void tq_plus_innerq_apply_k(
    __nv_bfloat16* __restrict__ data,
    const unsigned int head_dim
) {
    if (head_dim != INNERQ_MAX_CHANNELS) return;

    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (lane >= 32) return;

    __nv_bfloat16* head_data = data + (unsigned long long)head * head_dim;

    if (tq_plus::d_innerq_calibrating == 1 && head == 0) {
        #pragma unroll
        for (unsigned int i = 0; i < 4; i++) {
            unsigned int ch = lane * 4 + i;
            float v = __bfloat162float(head_data[ch]);
            atomicAdd(&tq_plus::d_innerq_sq_accum[ch], v * v);
        }
        if (lane == 0) {
            atomicAdd(&tq_plus::d_innerq_count, 1);
        }
    }

    if (tq_plus::d_innerq_active == 0) return;

    #pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        float v = __bfloat162float(head_data[ch]);
        v *= tq_plus::d_innerq_scale[ch];
        head_data[ch] = __float2bfloat16(v);
    }
}
