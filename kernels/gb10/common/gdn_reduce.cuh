// SPDX-License-Identifier: AGPL-3.0-only
//
// Shared reduction primitives for Gated Delta Net (GDN) WY kernels.
//
// Purpose: guarantee deterministic, identical FP32 reduction order across the
// WY-chunkwise and tree-WY kernels (`gated_delta_rule_wy{,3,4}.cu`,
// `gated_delta_rule_wy*_f16.cu`, `gated_delta_rule_tree*.cu`). FP32
// floating-point addition is non-associative, so differing reduction trees
// produce a few-ulp drift in `k_i^T @ k_j` and `gdn_out` that can flip greedy
// argmax on boundary tokens during MTP verify.
//
// Every WY kernel calls the exact same primitives declared here so the
// compiler emits the identical PTX reduction sequence across variants.
//
// Declared `static __device__ __forceinline__` to give internal linkage per
// translation unit — no ODR violations when included from multiple .cu
// files.

#ifndef ATLAS_GDN_REDUCE_CUH
#define ATLAS_GDN_REDUCE_CUH

// ── Warp-level tree reduction (32 lanes) ──
// Pattern: __shfl_down_sync(0xFFFFFFFFu, ...) with offsets 16, 8, 4, 2, 1.
static __device__ __forceinline__ float atlas_warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFFu, val, 16);
    val += __shfl_down_sync(0xFFFFFFFFu, val,  8);
    val += __shfl_down_sync(0xFFFFFFFFu, val,  4);
    val += __shfl_down_sync(0xFFFFFFFFu, val,  2);
    val += __shfl_down_sync(0xFFFFFFFFu, val,  1);
    return val;
}

// ── Block-level tree reduction for 128-thread blocks (4 warps) ──
//
// Reduction order:
//   1. Each warp reduces its 32 lanes via `atlas_warp_reduce_sum`.
//   2. Lane 0 of each warp writes to smem_warp[warp_id] (4 slots).
//   3. Lanes 0..3 of warp 0 do a width-4 shuffle tree: each lane loads its
//      slot, then `s += __shfl_down_sync(0xfu, s, 2, 4)` followed by
//      `s += __shfl_down_sync(0xfu, s, 1, 4)`. On lane 0 this yields
//      `(a+c) + (b+d)` where a,b,c,d = smem_warp[0..3]. Thread 0 stores the
//      result back to smem_warp[0] and all threads read `smem_warp[0]` after
//      a second __syncthreads().
//
// `smem_warp` must have at least 4 slots and is reused as the return buffer.
static __device__ __forceinline__ float atlas_block_reduce_sum(
    float val,
    float* smem_warp,
    unsigned int tid
) {
    val = atlas_warp_reduce_sum(val);
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid & 31;
    if (lane_id == 0) smem_warp[warp_id] = val;
    __syncthreads();
    if (tid < 4) {
        float s = smem_warp[tid];
        s += __shfl_down_sync(0xfu, s, 2, 4);
        s += __shfl_down_sync(0xfu, s, 1, 4);
        if (tid == 0) smem_warp[0] = s;
    }
    __syncthreads();
    return smem_warp[0];
}

#endif // ATLAS_GDN_REDUCE_CUH
