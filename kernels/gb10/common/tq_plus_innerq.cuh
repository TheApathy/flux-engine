// SPDX-License-Identifier: AGPL-3.0-only
// Upstream attribution: Incorporates TurboQuant+ / Sparse-V / two-sided WHT
// components derived from TheTom/llama-cpp-turboquant (MIT License) and
// TheTom/turboquant_plus (Apache-2.0 License). See docs/CITATIONS.md and
// THIRD_PARTY_NOTICES.md for full upstream copyright and license notices.
//
// TurboQuant+ InnerQ — per-channel Q/K equalization.
//
// See docs/CITATIONS.md and THIRD_PARTY_NOTICES.md for prior-art chain.
//
// Equalizes per-channel variance of post-WHT K vectors before quantization.
// Reduces the dynamic range each Lloyd-Max bin must cover by smoothing
// high-variance channels into the available codebook range. Preserves
// attention dot products via the identity:
//   <S^{-1}(W Q), S(W K)> = (W Q)^T S^{-1} S (W K) = <W Q, W K> = <Q, K>
// Both Q (multiplied by scale_inv = 1/s) and K (multiplied by scale = s) MUST
// be scaled on the SAME side of the orthogonal WHT W (here both post-WHT);
// scaling on opposite sides of W breaks the identity because diag(s) does not
// commute with W.
//
// Two-phase operation:
//   1. Calibration (first target_tokens K vectors): accumulate per-channel
//      post-WHT sum-of-squares in d_innerq_sq_accum. d_innerq_calibrating = 1.
//   2. Finalize (host calls turbo_innerq_finalize or InnerQDriver::maybe_finalize
//      once d_innerq_count >= target_tokens): compute per-channel RMS,
//      scale[i] = (mean_rms/rms[i])^strength clamped to [0.5, 2.0]. Auto-disable
//      if max_ratio < 1.2 && min_ratio > 1/1.2 (channels already balanced).
//      Upload to d_innerq_scale / d_innerq_scale_inv, set d_innerq_active = 1.
//   3. Active (post-finalize, all subsequent tokens): BOTH Q (by
//      d_innerq_scale_inv) and K (by d_innerq_scale) are scaled AFTER WHT and
//      before quantization/attention, preserving <Q, K> exactly.
//
// PTX linking note: Under per-file `nvcc --ptx` + `cuModuleLoadData` (no
// `-rdc=true` / `cuLink`), cross-TU `extern __device__` symbols cannot be
// resolved across separate PTX modules. Therefore the global-state apply
// kernels (`tq_plus_innerq_apply_q`, `tq_plus_innerq_apply_k`) live in the
// same TU as these definitions (`tq_plus_innerq.cu`), while standalone
// `tq_plus_innerq_apply.cu` exposes explicit-pointer variants (`*_ptrs`).

#pragma once
#include <cuda_bf16.h>

namespace tq_plus {

// Max per-head channel count. Qwen3.6-A3B uses head_dim=128, so 128 is enough.
// Bump if porting to head_dim > 128 models without splitting the array.
#define INNERQ_MAX_CHANNELS 128

// Device state — driven by host-side calibration controller.
// Defined in tq_plus_innerq.cu (one TU only).
extern __device__ float d_innerq_scale[INNERQ_MAX_CHANNELS];
extern __device__ float d_innerq_scale_inv[INNERQ_MAX_CHANNELS];
extern __device__ float d_innerq_sq_accum[INNERQ_MAX_CHANNELS];
extern __device__ int   d_innerq_count;
extern __device__ int   d_innerq_active;       // 0 = identity scales, 1 = post-finalize
extern __device__ int   d_innerq_calibrating;  // 1 = accumulating K² stats

// Apply per-channel scale_inv to Q-side values. No-op when d_innerq_active=0.
// Caller: each thread holds 4 floats for hd=128 (lane*4 + i indexing).
__device__ __forceinline__ void apply_innerq_scale_inv_128(float vals[4], unsigned int lane) {
    if (d_innerq_active == 0) return;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        vals[i] *= d_innerq_scale_inv[ch];
    }
}

// Apply per-channel scale to K-side values. No-op when d_innerq_active=0.
__device__ __forceinline__ void apply_innerq_scale_128(float vals[4], unsigned int lane) {
    if (d_innerq_active == 0) return;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        vals[i] *= d_innerq_scale[ch];
    }
}

// Accumulate K² into d_innerq_sq_accum during calibration. Each thread adds
// its 4 values squared into the corresponding channel slot. Atomic adds
// because every token's 32 lanes write to the same 128 channel slots in
// parallel across many warps.
__device__ __forceinline__ void accumulate_innerq_calibration_128(float vals[4], unsigned int lane) {
    if (d_innerq_calibrating == 0) return;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        atomicAdd(&d_innerq_sq_accum[ch], vals[i] * vals[i]);
    }
    if (lane == 0 && threadIdx.y == 0) {
        atomicAdd(&d_innerq_count, 1);
    }
}

}  // namespace tq_plus
