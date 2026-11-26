// SPDX-License-Identifier: AGPL-3.0-only

// Atlas MoE Top-K Softmax kernel for SM121 (GB10).
//
// GPU-side replacement for CPU top-K routing.
// Eliminates D2H copy of gate logits + CPU sort + CPU softmax.
//
// Input:  gate_logits[num_experts] BF16 on device
// Output: expert_indices[top_k] u32 (sorted by logit value, descending)
//         expert_weights[top_k] f32 (softmax probabilities)
//
// Single block, 256 threads. Supports up to 512 experts (2 loads per thread).

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define MAX_EXPERTS 512
#define MAX_TOP_K 32  // Must be >= num_experts_per_tok (22 for Super 120B)
#define WARP_SIZE 32

// moe_topk_softmax: find top-K experts and compute softmax weights.
//
// Grid: (1, 1, 1)   Block: (256, 1, 1)
//
// Algorithm:
// 1. Each thread loads up to 2 gate_logits (BF16->f32) for num_experts <= 512
// 2. Block-wide find top-K via parallel warp-shuffle + shared memory reduction
// 3. Compute softmax over all experts, extract top-K weights, and optionally renormalize
// 4. Write expert_indices and expert_weights
extern "C" __global__ void moe_topk_softmax(
    const __nv_bfloat16* __restrict__ gate_logits,  // [num_experts] BF16
    unsigned int* __restrict__ expert_indices,       // [top_k] output
    float* __restrict__ expert_weights,              // [top_k] output
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize  // 1 = normalize softmax weights to sum to 1
) {
    __shared__ float s_vals[MAX_EXPERTS];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = BLOCK_SIZE / 32;

    // Phase 1: Load gate logits (BF16 -> f32) — 2 elements per thread for 512 experts
    unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;
    unsigned int actual_k = top_k < actual_n ? top_k : actual_n;
    if (actual_k > MAX_TOP_K) actual_k = MAX_TOP_K;
    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        s_vals[i] = __bfloat162float(gate_logits[i]);
    }
    // Initialize remaining slots to -inf
    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_vals[i] = -1e30f;
    }
    __syncthreads();

    // Phase 2: Parallel top-K via iterative warp-shuffle max reduction
    for (unsigned int t = 0; t < actual_k; t++) {
        // Each thread finds local max across its portion of experts
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_vals[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }

        // Warp-level reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            unsigned int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        // Cross-warp reduction via shared memory
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
            // Invalidate winner for next iteration
            s_vals[best_idx] = -1e30f;
        }
        __syncthreads();
    }

    // Phase 3: Compute softmax over ALL experts, then extract top-K weights.
    //
    // HF/vLLM compute softmax(all_logits)[top_k_indices], NOT softmax(top_k_logits).
    // The difference: with full softmax, each weight is divided by sum of exp over
    // ALL experts (not just top-K), producing smaller absolute weights but correct
    // relative magnitudes.
    //
    // Optimization: s_top_vals[0] IS the global max (first top-K = largest logit).
    // Skip Phase 3a (restore) and Phase 3b (find max). Compute exp+sum directly,
    // adding back top-K contributions that are currently -1e30 in s_vals[].

    float global_max = s_top_vals[0];  // First top-K IS the global max

    // Phase 3c: Compute exp(x - max) for non-top-K experts and parallel sum.
    // Top-K slots in s_vals[] are -1e30, so exp(-1e30 - max) ≈ 0 — correct.
    {
        float local_sum = 0.0f;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            local_sum += __expf(s_vals[i] - global_max);
        }
        // Warp-level sum reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0) s_warp_val[warp_id] = local_sum;
        __syncthreads();

        // Cross-warp sum + add back top-K contributions (thread 0)
        if (tid == 0) {
            float total = 0.0f;
            for (unsigned int w = 0; w < num_warps; w++) {
                total += s_warp_val[w];
            }
            // Add correct exp values for top-K (replacing the ~0 from -1e30)
            for (unsigned int t = 0; t < actual_k; t++) {
                total += __expf(s_top_vals[t] - global_max);
            }
            s_warp_val[0] = total;
        }
        __syncthreads();
    }

    float exp_sum = s_warp_val[0];

    // Phase 3d: Extract top-K weights from full softmax and write results
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

// Batched variant: process N tokens in parallel, one block per token.
//
// Grid: (N, 1, 1)   Block: (256, 1, 1)
//
// gate_logits:    [N, num_experts] BF16
// expert_indices: [N, top_k] u32
// expert_weights: [N, top_k] f32
extern "C" __global__ void moe_topk_softmax_batched(
    const __nv_bfloat16* __restrict__ gate_logits,  // [N, num_experts]
    unsigned int* __restrict__ expert_indices,       // [N, top_k]
    float* __restrict__ expert_weights,              // [N, top_k]
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize
) {
    __shared__ float s_vals[MAX_EXPERTS];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = BLOCK_SIZE / 32;

    // Per-token pointer offsets
    const __nv_bfloat16* my_gate = gate_logits + token * num_experts;
    unsigned int* my_indices = expert_indices + token * top_k;
    float* my_weights = expert_weights + token * top_k;

    unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;
    unsigned int actual_k = top_k < actual_n ? top_k : actual_n;
    if (actual_k > MAX_TOP_K) actual_k = MAX_TOP_K;
    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        s_vals[i] = __bfloat162float(my_gate[i]);
    }
    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_vals[i] = -1e30f;
    }
    __syncthreads();

    // Phase 2: Parallel top-K
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
        for (int offset = 16; offset > 0; offset >>= 1) {
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

    // Phase 3: Full softmax
    float global_max = s_top_vals[0];
    {
        float local_sum = 0.0f;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            local_sum += __expf(s_vals[i] - global_max);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0) s_warp_val[warp_id] = local_sum;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (unsigned int w = 0; w < num_warps; w++) {
                total += s_warp_val[w];
            }
            for (unsigned int t = 0; t < actual_k; t++) {
                total += __expf(s_top_vals[t] - global_max);
            }
            s_warp_val[0] = total;
        }
        __syncthreads();
    }

    float exp_sum = s_warp_val[0];
    if (tid == 0) {
        for (unsigned int t = 0; t < actual_k; t++) {
            my_indices[t] = s_top_idxs[t];
            float softmax_weight = __expf(s_top_vals[t] - global_max) / exp_sum;
            if (normalize) {
                s_top_vals[t] = softmax_weight;
            } else {
                my_weights[t] = softmax_weight;
            }
        }
        if (normalize) {
            float topk_sum = 0.0f;
            for (unsigned int t = 0; t < actual_k; t++) {
                topk_sum += s_top_vals[t];
            }
            for (unsigned int t = 0; t < actual_k; t++) {
                my_weights[t] = s_top_vals[t] / topk_sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Warp-per-token twin of moe_topk_softmax_batched for num_experts == 512 and top_k <= 16
// (ATLAS_QWEN4_PREFILL_MOE_TOPK_WARP=1; Flash-Next prefill router). Same outputs bit for bit:
//
// Selection. The block kernel picks, per round, the maximum s_vals over the candidates
// (v > -1e30f: NaN and already-picked slots, which hold -1e30f, never qualify) and breaks
// ties by its reduction order: thread tid scans i = tid then tid + 256 (strict >, so j = i / 256
// = 0 first), the shfl_down tournament keeps the receiving lane on ties (lane order = the
// bit-reversed lane id: even lanes before odd, then bit 1, ...), and thread 0 folds warps 0..7
// with strict > (lower warp first). So among equal values the winner is the lowest
//   rank(i) = warp(i) << 6 | bitrev5(lane(i)) << 1 | j(i),  warp = (i & 255) >> 5, lane = i & 31.
// "Equal" is the float == of the same compiled comparisons (FTZ under the target's fast-math
// flags, so +-0 and subnormals tie exactly as in the block kernel). With no candidate the block
// kernel returns (-1e30f, expert 0) and writes -1e30f to s_vals[0]; so does this kernel.
// Here a warp owns one token: lane L holds experts 8L..8L+7 (j = 0) and 256+8L..256+8L+7 (j = 1)
// from two 16-byte loads, keeps its best (value, rank), and a 5-step xor butterfly with the
// comparator (v > v' or (v == v' and rank < rank')) gives every lane the same winner.
//
// Softmax. Block thread (w, l) summed 0.0f + exp(s[i]) + exp(s[i + 256]) for i = 32w + l, then
// shfl_down 16, 8, 4, 2, 1 to lane 0, then thread 0 added warps 0..7 in order and the top-k
// exps in order. Lane L holds those pairs for l = 8(L & 3) + c, c = 0..7, w = L >> 2, so the
// same tree is: lanes L, L+2 (step 16), lanes L, L+1 (step 8), then c with c+4, c+2, c+1 in
// registers; warp w's sum lands in lane 4w. Same expressions, same order, same rounding.
// Grid: (ceil(N / 8), 1, 1)  Block: (256, 1, 1)  (one warp per token; any N).
#define MOE_TOPK_W_EXPERTS 512
#define MOE_TOPK_W_MAX_TOP_K 16

__device__ __forceinline__ unsigned int moe_topk_w_bitrev5(unsigned int x) {
    return __brev(x) >> 27;
}

extern "C" __global__ void __launch_bounds__(256) moe_topk_softmax_batched_w32(
    const __nv_bfloat16* __restrict__ gate_logits,  // [N, 512], rows 16-byte aligned
    unsigned int* __restrict__ expert_indices,       // [N, top_k]
    float* __restrict__ expert_weights,              // [N, top_k]
    unsigned int num_tokens,
    unsigned int top_k,                              // 1..16
    unsigned int normalize
) {
    __shared__ float s_top_vals[8][MOE_TOPK_W_MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[8][MOE_TOPK_W_MAX_TOP_K];
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int token = blockIdx.x * 8u + warp_id;
    if (token >= num_tokens) return;  // whole warp: token is warp-uniform

    // x[8j + c] = expert 256 j + 8 lane + c.
    float x[16];
    {
        const uint4* row = reinterpret_cast<const uint4*>(gate_logits + (unsigned long long)token * MOE_TOPK_W_EXPERTS);
        const uint4 w0 = row[lane];
        const uint4 w1 = row[32u + lane];
        const unsigned int u[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[2 * i] = __uint_as_float(u[i] << 16);
            x[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
        }
    }
    // rank of x[8j + c]: (lane >> 2) << 6 | bitrev5(8 (lane & 3) + c) << 1 | j.
    const unsigned int rank_hi = (lane >> 2) << 6;
    const unsigned int l_base = (lane & 3u) << 3;

    if (top_k == 10u && normalize != 0u) {
        const unsigned int rank_lane = rank_hi | (((lane & 1u) << 2) | (lane & 2u));
        constexpr unsigned int kRankImm[16] = {
            0u, 32u, 16u, 48u, 8u, 40u, 24u, 56u,
            1u, 33u, 17u, 49u, 9u, 41u, 25u, 57u
        };
        float global_max = -1e30f;
        float top_val = -1e30f;
        unsigned int top_idx = 0u;
#pragma unroll
        for (unsigned int t = 0; t < 10u; ++t) {
            float bv = -1e30f;
            unsigned int br = 0xffffffffu;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const float v = x[k];
                const unsigned int r = rank_lane | kRankImm[k];
                if (v > -1e30f && (v > bv || (v == bv && r < br))) {
                    bv = v;
                    br = r;
                }
            }
#pragma unroll
            for (int d = 16; d > 0; d >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, d);
                const unsigned int orr = __shfl_xor_sync(0xffffffffu, br, d);
                if (ov > bv || (ov == bv && orr < br)) {
                    bv = ov;
                    br = orr;
                }
            }
            unsigned int idx, owner, ksel;
            if (br == 0xffffffffu) {
                bv = -1e30f;
                idx = 0u;
                owner = 0u;
                ksel = 0u;
            } else {
                const unsigned int w = br >> 6;
                const unsigned int l = moe_topk_w_bitrev5((br >> 1) & 31u);
                const unsigned int j = br & 1u;
                idx = (j << 8) | (w << 5) | l;
                owner = (w << 2) | (l >> 3);
                ksel = (j << 3) | (l & 7u);
            }
            if (lane == owner) {
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    if ((unsigned int)k == ksel) x[k] = -1e30f;
                }
            }
            if (t == 0u) global_max = bv;
            if (lane == t) {
                top_val = bv;
                top_idx = idx;
            }
        }

        float ls[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            float local_sum = 0.0f;
            local_sum += __expf(x[c] - global_max);
            local_sum += __expf(x[8 + c] - global_max);
            ls[c] = local_sum;
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) ls[c] += __shfl_down_sync(0xffffffffu, ls[c], 2);
#pragma unroll
        for (int c = 0; c < 8; ++c) ls[c] += __shfl_down_sync(0xffffffffu, ls[c], 1);
#pragma unroll
        for (int c = 0; c < 4; ++c) ls[c] += ls[c + 4];
#pragma unroll
        for (int c = 0; c < 2; ++c) ls[c] += ls[c + 2];
        ls[0] += ls[1];

        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            total += __shfl_sync(0xffffffffu, ls[0], w * 4);
        }
        const float top_exp = (lane < 10u) ? __expf(top_val - global_max) : 0.0f;
#pragma unroll
        for (unsigned int t = 0; t < 10u; ++t) {
            total += __shfl_sync(0xffffffffu, top_exp, t);
        }
        const float softmax_weight = (lane < 10u) ? (top_exp / total) : 0.0f;
        float topk_sum = 0.0f;
#pragma unroll
        for (unsigned int t = 0; t < 10u; ++t) {
            topk_sum += __shfl_sync(0xffffffffu, softmax_weight, t);
        }
        if (lane < 10u) {
            unsigned int* my_indices = expert_indices + (unsigned long long)token * 10ull;
            float* my_weights = expert_weights + (unsigned long long)token * 10ull;
            my_indices[lane] = top_idx;
            my_weights[lane] = softmax_weight / topk_sum;
        }
        return;
    }

    const unsigned int rounds = top_k < MOE_TOPK_W_MAX_TOP_K ? top_k : MOE_TOPK_W_MAX_TOP_K;
    for (unsigned int t = 0; t < rounds; ++t) {
        float bv = -1e30f;
        unsigned int br = 0xffffffffu;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float v = x[k];
            const unsigned int r = rank_hi | (moe_topk_w_bitrev5(l_base + (k & 7)) << 1) | (unsigned int)(k >> 3);
            if (v > -1e30f && (v > bv || (v == bv && r < br))) {
                bv = v;
                br = r;
            }
        }
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, d);
            const unsigned int orr = __shfl_xor_sync(0xffffffffu, br, d);
            if (ov > bv || (ov == bv && orr < br)) {
                bv = ov;
                br = orr;
            }
        }
        unsigned int idx, owner, ksel;
        if (br == 0xffffffffu) {
            // No candidate left: the block kernel's (-1e30f, expert 0), and s_vals[0] = -1e30f.
            bv = -1e30f;
            idx = 0;
            owner = 0;
            ksel = 0;
        } else {
            const unsigned int w = br >> 6;
            const unsigned int l = moe_topk_w_bitrev5((br >> 1) & 31u);
            const unsigned int j = br & 1u;
            idx = (j << 8) | (w << 5) | l;
            owner = (w << 2) | (l >> 3);
            ksel = (j << 3) | (l & 7u);
        }
        if (lane == owner) {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                if ((unsigned int)k == ksel) x[k] = -1e30f;
            }
        }
        if (lane == 0) {
            s_top_vals[warp_id][t] = bv;
            s_top_idxs[warp_id][t] = idx;
        }
    }
    __syncwarp();

    // Full softmax over the post-selection values, in the block kernel's order.
    const float global_max = s_top_vals[warp_id][0];
    float ls[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        float local_sum = 0.0f;
        local_sum += __expf(x[c] - global_max);
        local_sum += __expf(x[8 + c] - global_max);
        ls[c] = local_sum;
    }
#pragma unroll
    for (int c = 0; c < 8; ++c) ls[c] += __shfl_down_sync(0xffffffffu, ls[c], 2);
#pragma unroll
    for (int c = 0; c < 8; ++c) ls[c] += __shfl_down_sync(0xffffffffu, ls[c], 1);
#pragma unroll
    for (int c = 0; c < 4; ++c) ls[c] += ls[c + 4];
#pragma unroll
    for (int c = 0; c < 2; ++c) ls[c] += ls[c + 2];
    ls[0] += ls[1];
    float warp_sums[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) warp_sums[w] = __shfl_sync(0xffffffffu, ls[0], w * 4);

    if (lane == 0) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            total += warp_sums[w];
        }
        for (unsigned int t = 0; t < rounds; t++) {
            total += __expf(s_top_vals[warp_id][t] - global_max);
        }
        const float exp_sum = total;
        unsigned int* my_indices = expert_indices + (unsigned long long)token * top_k;
        float* my_weights = expert_weights + (unsigned long long)token * top_k;
        for (unsigned int t = 0; t < rounds; t++) {
            my_indices[t] = s_top_idxs[warp_id][t];
            float softmax_weight = __expf(s_top_vals[warp_id][t] - global_max) / exp_sum;
            if (normalize) {
                s_top_vals[warp_id][t] = softmax_weight;
            } else {
                my_weights[t] = softmax_weight;
            }
        }
        if (normalize) {
            float topk_sum = 0.0f;
            for (unsigned int t = 0; t < rounds; t++) {
                topk_sum += s_top_vals[warp_id][t];
            }
            for (unsigned int t = 0; t < rounds; t++) {
                my_weights[t] = s_top_vals[warp_id][t] / topk_sum;
            }
        }
    }
}

// Single-warp-per-CTA twin of moe_topk_softmax_batched_w32 for small batches (rows <= 256)
// on high-SM GPUs (SM120/SM103): launches 1 warp (32 threads) per token so `grid = num_tokens`
// yields 8x more active SMs than the 8-warp/CTA entrypoint while preserving bit-identical
// selection order, full-softmax summation tree, and normalization.
// Grid: (N, 1, 1)  Block: (32, 1, 1)  (one warp per CTA, one token per CTA).
extern "C" __global__ void __launch_bounds__(32) moe_topk_softmax_batched_w32_1w(
    const __nv_bfloat16* __restrict__ gate_logits,  // [N, 512], rows 16-byte aligned
    unsigned int* __restrict__ expert_indices,       // [N, top_k]
    float* __restrict__ expert_weights,              // [N, top_k]
    unsigned int num_tokens,
    unsigned int top_k,                              // 1..16
    unsigned int normalize
) {
    __shared__ float s_top_vals[MOE_TOPK_W_MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[MOE_TOPK_W_MAX_TOP_K];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;

    // x[8j + c] = expert 256 j + 8 lane + c.
    float x[16];
    {
        const uint4* row = reinterpret_cast<const uint4*>(gate_logits + (unsigned long long)token * MOE_TOPK_W_EXPERTS);
        const uint4 w0 = row[lane];
        const uint4 w1 = row[32u + lane];
        const unsigned int u[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[2 * i] = __uint_as_float(u[i] << 16);
            x[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
        }
    }
    const unsigned int rank_hi = (lane >> 2) << 6;
    const unsigned int l_base = (lane & 3u) << 3;

    if (top_k == 10u && normalize != 0u) {
        const unsigned int rank_lane = rank_hi | (((lane & 1u) << 2) | (lane & 2u));
        constexpr unsigned int kRankImm[16] = {
            0u, 32u, 16u, 48u, 8u, 40u, 24u, 56u,
            1u, 33u, 17u, 49u, 9u, 41u, 25u, 57u
        };
        float global_max = -1e30f;
        float top_val = -1e30f;
        unsigned int top_idx = 0u;
#pragma unroll
        for (unsigned int t = 0; t < 10u; ++t) {
            float bv = -1e30f;
            unsigned int br = 0xffffffffu;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const float v = x[k];
                const unsigned int r = rank_lane | kRankImm[k];
                if (v > -1e30f && (v > bv || (v == bv && r < br))) {
                    bv = v;
                    br = r;
                }
            }
#pragma unroll
            for (int d = 16; d > 0; d >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, d);
                const unsigned int orr = __shfl_xor_sync(0xffffffffu, br, d);
                if (ov > bv || (ov == bv && orr < br)) {
                    bv = ov;
                    br = orr;
                }
            }
            unsigned int idx, owner, ksel;
            if (br == 0xffffffffu) {
                bv = -1e30f;
                idx = 0u;
                owner = 0u;
                ksel = 0u;
            } else {
                const unsigned int w = br >> 6;
                const unsigned int l = moe_topk_w_bitrev5((br >> 1) & 31u);
                const unsigned int j = br & 1u;
                idx = (j << 8) | (w << 5) | l;
                owner = (w << 2) | (l >> 3);
                ksel = (j << 3) | (l & 7u);
            }
            if (lane == owner) {
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    if ((unsigned int)k == ksel) x[k] = -1e30f;
                }
            }
            if (t == 0u) global_max = bv;
            if (lane == t) {
                top_val = bv;
                top_idx = idx;
            }
        }

        float ls[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            float local_sum = 0.0f;
            local_sum += __expf(x[c] - global_max);
            local_sum += __expf(x[8 + c] - global_max);
            ls[c] = local_sum;
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) ls[c] += __shfl_down_sync(0xffffffffu, ls[c], 2);
#pragma unroll
        for (int c = 0; c < 8; ++c) ls[c] += __shfl_down_sync(0xffffffffu, ls[c], 1);
#pragma unroll
        for (int c = 0; c < 4; ++c) ls[c] += ls[c + 4];
#pragma unroll
        for (int c = 0; c < 2; ++c) ls[c] += ls[c + 2];
        ls[0] += ls[1];

        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            total += __shfl_sync(0xffffffffu, ls[0], w * 4);
        }
        const float top_exp = (lane < 10u) ? __expf(top_val - global_max) : 0.0f;
#pragma unroll
        for (unsigned int t = 0; t < 10u; ++t) {
            total += __shfl_sync(0xffffffffu, top_exp, t);
        }
        const float softmax_weight = (lane < 10u) ? (top_exp / total) : 0.0f;
        float topk_sum = 0.0f;
#pragma unroll
        for (unsigned int t = 0; t < 10u; ++t) {
            topk_sum += __shfl_sync(0xffffffffu, softmax_weight, t);
        }
        if (lane < 10u) {
            unsigned int* my_indices = expert_indices + (unsigned long long)token * 10ull;
            float* my_weights = expert_weights + (unsigned long long)token * 10ull;
            my_indices[lane] = top_idx;
            my_weights[lane] = softmax_weight / topk_sum;
        }
        return;
    }

    const unsigned int rounds = top_k < MOE_TOPK_W_MAX_TOP_K ? top_k : MOE_TOPK_W_MAX_TOP_K;
    for (unsigned int t = 0; t < rounds; ++t) {
        float bv = -1e30f;
        unsigned int br = 0xffffffffu;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float v = x[k];
            const unsigned int r = rank_hi | (moe_topk_w_bitrev5(l_base + (k & 7)) << 1) | (unsigned int)(k >> 3);
            if (v > -1e30f && (v > bv || (v == bv && r < br))) {
                bv = v;
                br = r;
            }
        }
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, d);
            const unsigned int orr = __shfl_xor_sync(0xffffffffu, br, d);
            if (ov > bv || (ov == bv && orr < br)) {
                bv = ov;
                br = orr;
            }
        }
        unsigned int idx, owner, ksel;
        if (br == 0xffffffffu) {
            bv = -1e30f;
            idx = 0;
            owner = 0;
            ksel = 0;
        } else {
            const unsigned int w = br >> 6;
            const unsigned int l = moe_topk_w_bitrev5((br >> 1) & 31u);
            const unsigned int j = br & 1u;
            idx = (j << 8) | (w << 5) | l;
            owner = (w << 2) | (l >> 3);
            ksel = (j << 3) | (l & 7u);
        }
        if (lane == owner) {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                if ((unsigned int)k == ksel) x[k] = -1e30f;
            }
        }
        if (lane == 0) {
            s_top_vals[t] = bv;
            s_top_idxs[t] = idx;
        }
    }
    __syncwarp();

    const float global_max = s_top_vals[0];
    float ls[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        float local_sum = 0.0f;
        local_sum += __expf(x[c] - global_max);
        local_sum += __expf(x[8 + c] - global_max);
        ls[c] = local_sum;
    }
#pragma unroll
    for (int c = 0; c < 8; ++c) ls[c] += __shfl_down_sync(0xffffffffu, ls[c], 2);
#pragma unroll
    for (int c = 0; c < 8; ++c) ls[c] += __shfl_down_sync(0xffffffffu, ls[c], 1);
#pragma unroll
    for (int c = 0; c < 4; ++c) ls[c] += ls[c + 4];
#pragma unroll
    for (int c = 0; c < 2; ++c) ls[c] += ls[c + 2];
    ls[0] += ls[1];
    float warp_sums[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) warp_sums[w] = __shfl_sync(0xffffffffu, ls[0], w * 4);

    if (lane == 0) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            total += warp_sums[w];
        }
        for (unsigned int t = 0; t < rounds; t++) {
            total += __expf(s_top_vals[t] - global_max);
        }
        const float exp_sum = total;
        unsigned int* my_indices = expert_indices + (unsigned long long)token * top_k;
        float* my_weights = expert_weights + (unsigned long long)token * top_k;
        for (unsigned int t = 0; t < rounds; t++) {
            my_indices[t] = s_top_idxs[t];
            float softmax_weight = __expf(s_top_vals[t] - global_max) / exp_sum;
            if (normalize) {
                s_top_vals[t] = softmax_weight;
            } else {
                my_weights[t] = softmax_weight;
            }
        }
        if (normalize) {
            float topk_sum = 0.0f;
            for (unsigned int t = 0; t < rounds; t++) {
                topk_sum += s_top_vals[t];
            }
            for (unsigned int t = 0; t < rounds; t++) {
                my_weights[t] = s_top_vals[t] / topk_sum;
            }
        }
    }
}
