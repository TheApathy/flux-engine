// SPDX-License-Identifier: AGPL-3.0-only
// Upstream attribution: Incorporates TurboQuant+ / Sparse-V / two-sided WHT
// components derived from TheTom/llama-cpp-turboquant (MIT License) and
// TheTom/turboquant_plus (Apache-2.0 License). See docs/CITATIONS.md and
// THIRD_PARTY_NOTICES.md for full upstream copyright and license notices.
//
// TurboQuant+ InnerQ application kernels — per-channel Q/K equalization
// (explicit-pointer ABI, zero cross-PTX `.extern .global` dependencies).
//
// Two tiny element-wise multiply kernels that fire AFTER WHT(Q) and AFTER
// WHT(K) when InnerQ is active (*active_ptr != 0). When inactive
// (*active_ptr == 0), the kernels return immediately — zero scaling cost.
//
// Math: Let W be the orthogonal WHT (W^T W = I) and S = diag(s).
//   <S^{-1}(W Q), S(W K)> = (W Q)^T S^{-1} S (W K) = Q^T W^T W K = <Q, K>.
// Both Q (by 1/s) and K (by s) MUST be scaled on the SAME side of W (both
// post-WHT here); scaling on opposite sides of W breaks the identity because
// non-scalar diagonal S does not commute with W.
//
// Note: The 2-argument global-state variants (`tq_plus_innerq_apply_q` and
// `tq_plus_innerq_apply_k`) live in `tq_plus_innerq.cu` (same PTX module as
// `tq_plus::d_innerq_*`), because standalone per-file `nvcc --ptx` +
// `cuModuleLoadData` cannot link `.extern .global` symbols across modules.

#include <cuda_bf16.h>

static constexpr unsigned int TQ_PLUS_INNERQ_MAX_CHANNELS = 128u;

// Apply post-WHT scale_inv to Q using explicit device pointers (hd=128).
// Grid: (num_q_heads, 1, 1)   Block: (32, 1, 1) — one warp per head.
extern "C" __global__ void tq_plus_innerq_apply_q_ptrs(
    __nv_bfloat16* __restrict__ data,
    const unsigned int head_dim,
    const float* __restrict__ scale_inv,
    const int* __restrict__ active_ptr
) {
    if (active_ptr == nullptr || *active_ptr == 0) return;
    if (scale_inv == nullptr || head_dim != TQ_PLUS_INNERQ_MAX_CHANNELS) return;

    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (lane >= 32) return;

    __nv_bfloat16* head_data = data + (unsigned long long)head * head_dim;

    // 32 threads × 4 elements = 128.
    #pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        float v = __bfloat162float(head_data[ch]);
        v *= scale_inv[ch];
        head_data[ch] = __float2bfloat16(v);
    }
}

// Apply post-WHT scale to K using explicit device pointers (hd=128).
// Also accumulates post-WHT K^2 stats into sq_accum when *calibrating_ptr == 1.
// Grid: (num_kv_heads, 1, 1)   Block: (32, 1, 1) — one warp per head.
extern "C" __global__ void tq_plus_innerq_apply_k_ptrs(
    __nv_bfloat16* __restrict__ data,
    const unsigned int head_dim,
    const float* __restrict__ scale,
    float* __restrict__ sq_accum,
    int* __restrict__ count_ptr,
    const int* __restrict__ active_ptr,
    const int* __restrict__ calibrating_ptr
) {
    if (head_dim != TQ_PLUS_INNERQ_MAX_CHANNELS) return;

    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (lane >= 32) return;

    __nv_bfloat16* head_data = data + (unsigned long long)head * head_dim;

    // Calibration: accumulate post-WHT K^2 before scaling (head 0 only).
    if (calibrating_ptr != nullptr && *calibrating_ptr == 1 && head == 0
        && sq_accum != nullptr && count_ptr != nullptr) {
        #pragma unroll
        for (unsigned int i = 0; i < 4; i++) {
            unsigned int ch = lane * 4 + i;
            float v = __bfloat162float(head_data[ch]);
            atomicAdd(&sq_accum[ch], v * v);
        }
        if (lane == 0) {
            atomicAdd(count_ptr, 1);
        }
    }

    if (active_ptr == nullptr || *active_ptr == 0 || scale == nullptr) return;

    // Apply scale (post-WHT).
    #pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        float v = __bfloat162float(head_data[ch]);
        v *= scale[ch];
        head_data[ch] = __float2bfloat16(v);
    }
}
