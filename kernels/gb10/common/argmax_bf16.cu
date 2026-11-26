// SPDX-License-Identifier: AGPL-3.0-only

// Argmax over BF16 logits — single-block tree reduction.
//
// Finds the index of the maximum BF16 value in an array of `n` elements.
// Writes a single u32 token ID to `out`.
//
// Grid: (1, 1, 1)  Block: (1024, 1, 1)
// For vocab_size ≤ ~200K, a single block with 1024 threads is sufficient
// (each thread handles ceil(n/1024) elements).

#include <cuda_bf16.h>
#include <math_constants.h>

extern "C" __global__ void argmax_bf16(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    // Phase 1: each thread finds its local max
    float local_max = -CUDART_INF_F;
    unsigned int local_idx = 0xFFFFFFFFu;

    for (unsigned int i = tid; i < n; i += stride) {
        float v = __bfloat162float(logits[i]);
        if (v > local_max || (v == local_max && i < local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }

    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Phase 2: tree reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid] ||
                (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid])) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    // Phase 3: thread 0 writes result
    if (tid == 0) {
        out[0] = s_idx[0];
    }
}

// Argmax over the elementwise sum of two BF16 vectors.
//
// DSpark uses this for an exact full-vocabulary greedy choice without copying
// either vector to the host:
//   token = argmax(base_logits + markov_bias)
//
// Ties select the lowest token ID, matching a left-to-right host scan with a
// strict `>` comparison. This explicit secondary key matters because the
// ordinary tree reduction's lane order does not imply global token order.
extern "C" __global__ void argmax_add_bf16(
    const __nv_bfloat16* __restrict__ base_logits,
    const __nv_bfloat16* __restrict__ bias,
    unsigned int* __restrict__ out,
    unsigned int n
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    float local_max = -CUDART_INF_F;
    unsigned int local_idx = 0xFFFFFFFFu;

    for (unsigned int i = tid; i < n; i += stride) {
        const float value = __bfloat162float(base_logits[i]) +
                            __bfloat162float(bias[i]);
        if (value > local_max || (value == local_max && i < local_idx)) {
            local_max = value;
            local_idx = i;
        }
    }

    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float candidate = s_val[tid + s];
            const unsigned int candidate_idx = s_idx[tid + s];
            if (candidate > s_val[tid] ||
                (candidate == s_val[tid] && candidate_idx < s_idx[tid])) {
                s_val[tid] = candidate;
                s_idx[tid] = candidate_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = s_idx[0];
    }
}

// Argmax over FP32 logits — used when LM head outputs FP32 for sampling quality.
extern "C" __global__ void argmax_fp32(
    const float* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    float local_max = -CUDART_INF_F;
    unsigned int local_idx = 0xFFFFFFFFu;
    for (unsigned int i = tid; i < n; i += stride) {
        float v = logits[i];
        if (v > local_max || (v == local_max && i < local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid] ||
                (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid])) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = s_idx[0];
}

// Top-K over BF16 logits, batched across multiple rows.
//
// For each row in [0, num_rows), finds the top-K maximum values in a vector
// of `vocab` BF16 logits and writes:
//   - top_indices[row, 0..K]  : u32 token IDs sorted by score descending
//   - top_logits[row, 0..K]   : f32 logit values (NOT log-probs; caller can
//                                normalize externally if needed)
//
// Algorithm: K-pass argmax with masking. K is small (≤ 16 typical), so this
// is O(K * vocab / threads) which beats heap-based per-thread top-K for
// the M4B v2 DDTree case (K=8, vocab=248K). Valid rows sort numeric score
// descending, then token ID ascending. Any NaN or all--Inf row writes the
// invalid pair (UINT_MAX, NaN) to every requested output slot.
//
// Grid: (num_rows, 1, 1)  Block: (1024, 1, 1)
// Each block handles one row independently.
//
// Compile-time bound: MAX_TOP_K caps shared-mem usage. The checked Rust
// wrapper is the release authority for vocab/K bounds; raw invalid launches
// defensively return without writing.

#define MAX_TOP_K 16

__device__ __forceinline__ bool topk_candidate_better(
    float candidate,
    unsigned int candidate_idx,
    float current,
    unsigned int current_idx
) {
    return candidate_idx != 0xFFFFFFFFu &&
           (current_idx == 0xFFFFFFFFu ||
            candidate > current ||
            (candidate == current && candidate_idx < current_idx));
}

extern "C" __global__ void topk_bf16(
    const __nv_bfloat16* __restrict__ logits,  // [num_rows, vocab] BF16
    unsigned int* __restrict__ top_indices,    // [num_rows, k] u32
    float* __restrict__ top_logits,            // [num_rows, k] f32
    unsigned int num_rows,
    unsigned int vocab,
    unsigned int k
) {
    const unsigned int row = blockIdx.x;
    if (row >= num_rows) return;
    if (vocab == 0 || k == 0 || k > MAX_TOP_K || k > vocab) return;

    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];
    __shared__ unsigned int s_taken[MAX_TOP_K]; // already-selected indices
    __shared__ unsigned int s_row_invalid;
    __shared__ unsigned int s_row_usable;

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    const __nv_bfloat16* row_logits = logits + (size_t)row * vocab;

    // Full-row validity scan. NaN outside K is still invalid, so it cannot be
    // discovered by inspecting only the selected values on the host.
    if (tid == 0) {
        s_row_invalid = 0;
        s_row_usable = 0;
    }
    __syncthreads();
    unsigned int local_invalid = 0;
    unsigned int local_usable = 0;
    for (unsigned int i = tid; i < vocab; i += stride) {
        const float v = __bfloat162float(row_logits[i]);
        if (isnan(v)) {
            local_invalid = 1;
        } else if (v > -CUDART_INF_F) {
            local_usable = 1;
        }
    }
    if (local_invalid != 0) atomicOr(&s_row_invalid, 1u);
    if (local_usable != 0) atomicOr(&s_row_usable, 1u);
    __syncthreads();

    if (s_row_invalid != 0 || s_row_usable == 0) {
        if (tid < k) {
            const size_t out = (size_t)row * k + tid;
            top_indices[out] = 0xFFFFFFFFu;
            top_logits[out] = CUDART_NAN_F;
        }
        return;
    }

    // K-pass argmax. Each pass excludes previously selected indices.
    for (unsigned int pass = 0; pass < k; ++pass) {
        // Phase 1: per-thread local max, skipping already-selected indices.
        float local_max = -CUDART_INF_F;
        unsigned int local_idx = 0xFFFFFFFFu;

        for (unsigned int i = tid; i < vocab; i += stride) {
            bool skip = false;
            // Linear scan of already-selected indices — for K ≤ 16 this is
            // negligible vs the vocab loop body.
            for (unsigned int p = 0; p < pass; ++p) {
                if (s_taken[p] == i) { skip = true; break; }
            }
            if (skip) continue;
            const float v = __bfloat162float(row_logits[i]);
            if (topk_candidate_better(v, i, local_max, local_idx)) {
                local_max = v;
                local_idx = i;
            }
        }

        s_val[tid] = local_max;
        s_idx[tid] = local_idx;
        __syncthreads();

        // Phase 2: tree reduction
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (topk_candidate_better(
                        s_val[tid + s], s_idx[tid + s],
                        s_val[tid], s_idx[tid])) {
                    s_val[tid] = s_val[tid + s];
                    s_idx[tid] = s_idx[tid + s];
                }
            }
            __syncthreads();
        }

        // Phase 3: thread 0 records the winner.
        if (tid == 0) {
            s_taken[pass] = s_idx[0];
            top_indices[(size_t)row * k + pass] = s_idx[0];
            top_logits[(size_t)row * k + pass] = s_val[0];
        }
        __syncthreads();
    }
}

// argmax_bf16 plus the winner's softmax probability (fn-decode MTP draft
// confidence). Phases 1-2 are argmax_bf16's verbatim, so out[0] is the same
// token; phase 3 adds p = 1 / sum_i exp(x_i - max). Grid (1,1,1) Block (1024,1,1).
extern "C" __global__ void argmax_prob_bf16(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    float* __restrict__ out_prob,
    unsigned int n
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];
    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    float local_max = -CUDART_INF_F;
    unsigned int local_idx = 0xFFFFFFFFu;
    for (unsigned int i = tid; i < n; i += stride) {
        float v = __bfloat162float(logits[i]);
        if (v > local_max || (v == local_max && i < local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid] ||
                (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid])) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    const float vmax = s_val[0];
    const unsigned int winner = s_idx[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < n; i += stride) {
        local_sum += __expf(__bfloat162float(logits[i]) - vmax);
    }
    s_val[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_val[tid] += s_val[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = winner;
        out_prob[0] = s_val[0] > 0.0f ? 1.0f / s_val[0] : 0.0f;
    }
}

// Batched 0-ULP bit-identical sibling of `argmax_prob_bf16`: one 1024-thread
// block per row (`grid = (rows, 1, 1)`, `block = (1024, 1, 1)`), indexing
// `logits + (unsigned long long)blockIdx.x * vocab_stride`, `out[blockIdx.x]`,
// and `out_prob[blockIdx.x]`. Preserves the exact per-thread scan and tree
// reduction order of `argmax_prob_bf16` so both the winning token ID and its
// FP32 softmax probability are 0-ULP bit-identical to per-row launches.
extern "C" __global__ void __launch_bounds__(1024) argmax_prob_batched_bf16(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    float* __restrict__ out_prob,
    unsigned int n,
    unsigned int vocab_stride
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];
    const unsigned int row = blockIdx.x;
    const __nv_bfloat16* __restrict__ row_logits =
        logits + (unsigned long long)row * (unsigned long long)vocab_stride;
    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    float local_max = -CUDART_INF_F;
    unsigned int local_idx = 0xFFFFFFFFu;
    for (unsigned int i = tid; i < n; i += stride) {
        float v = __bfloat162float(row_logits[i]);
        if (v > local_max || (v == local_max && i < local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid] ||
                (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid])) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    const float vmax = s_val[0];
    const unsigned int winner = s_idx[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < n; i += stride) {
        local_sum += __expf(__bfloat162float(row_logits[i]) - vmax);
    }
    s_val[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_val[tid] += s_val[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        out[row] = winner;
        out_prob[row] = s_val[0] > 0.0f ? 1.0f / s_val[0] : 0.0f;
    }
}

// Row-parallel argmax_bf16 for verify (verify-cost, ATLAS_VERIFY_ARGMAX_ROWS=1):
// one 1024-thread block per row (`rows` rows `row_stride` elements apart), each
// thread scanning 8-element uint4 chunks with all loads independent instead of
// argmax_bf16's single block walking one element per thread per step.
// Every comparison is argmax_bf16's (v > best, or v == best at a lower index;
// NaN never wins; -inf wins only over the 0xFFFFFFFF sentinel), which selects
// the lowest index of the row maximum whatever the scan and tree order, so
// each row's index is the one argmax_bf16 returns for that row.
// Grid: (rows, 1, 1)  Block: (1024, 1, 1)
extern "C" __global__ void __launch_bounds__(1024) argmax_bf16_rows(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n,
    unsigned int row_stride
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    const __nv_bfloat16* __restrict__ x = logits + (unsigned long long)blockIdx.x * row_stride;

    float best = -CUDART_INF_F;
    unsigned int best_idx = 0xFFFFFFFFu;
    const unsigned int n8 = n / 8u;
    const bool vec = ((unsigned long long)x & 15ull) == 0ull;
    const unsigned int scalar_from = vec ? n8 * 8u : 0u;
    if (vec) {
        const uint4* __restrict__ x4 = (const uint4*)x;
        #pragma unroll 4
        for (unsigned int c = tid; c < n8; c += stride) {
            const uint4 w = x4[c];
            const unsigned int words[4] = {w.x, w.y, w.z, w.w};
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    __nv_bfloat16 b;
                    *(unsigned short*)&b = (unsigned short)(words[q] >> (16 * h));
                    const float v = __bfloat162float(b);
                    const unsigned int i = c * 8u + 2u * q + h;
                    if (v > best || (v == best && i < best_idx)) {
                        best = v;
                        best_idx = i;
                    }
                }
            }
        }
    }
    for (unsigned int i = scalar_from + tid; i < n; i += stride) {
        const float v = __bfloat162float(x[i]);
        if (v > best || (v == best && i < best_idx)) {
            best = v;
            best_idx = i;
        }
    }

    s_val[tid] = best;
    s_idx[tid] = best_idx;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid] ||
                (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid])) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = s_idx[0];
    }
}

// ── Fast Zero-Spill 2-Pass Vectorized Radix Top-K (1 <= k <= 32) ────────────
//
// Batched across `num_rows` independent blocks (`grid = (num_rows, 1, 1)`,
// `block = (512, 1, 1)`). Replaces K-pass argmax with 2 `uint4`-vectorized
// passes for nearly all rows (and an exact bounded 3rd-pass fallback only when
// the winning upper-byte bucket exceeds the 2,560-entry SMEM gather capacity).
//
// Exact BF16 total order:
//   - NaN (`(bits & 0x7FFFu) > 0x7F80u`) is excluded; if fewer than `k` valid
//     elements exist, remaining output slots receive `(0u, -INFINITY)`.
//   - `-0.0` (`0x8000`) folds to `+0.0` (`0x0000`) before ordering so equal
//     values break ties strictly by smaller token index (`id_a < id_b`).
//   - Outputs are sorted strictly descending by `(val, ~id)` and `top_logits`
//     stores `__bfloat162float(logits[row * vocab + idx])` (0-ULP exact).

#define TOPK_RADIX_MAX_K 32u
#define TOPK_RADIX_SMEM_CAND_CAP 2560u

__device__ __forceinline__ bool topk_radix_is_valid_u16(unsigned short raw) {
    return (raw & 0x7FFFu) <= 0x7F80u;
}

__device__ __forceinline__ unsigned short topk_radix_to_ordered_u16(unsigned short raw) {
    raw = (raw == 0x8000u) ? 0x0000u : raw;
    return ((raw & 0x8000u) != 0u)
        ? (unsigned short)(~raw & 0xFFFFu)
        : (unsigned short)(raw | 0x8000u);
}

__device__ __forceinline__ void topk_radix_pass1_pair(
    unsigned int* __restrict__ warp_hist,
    unsigned int packed2
) {
    const unsigned short r0 = (unsigned short)(packed2 & 0xFFFFu);
    if (topk_radix_is_valid_u16(r0)) {
        atomicAdd(&warp_hist[(unsigned int)topk_radix_to_ordered_u16(r0) >> 8], 1u);
    }
    const unsigned short r1 = (unsigned short)(packed2 >> 16);
    if (topk_radix_is_valid_u16(r1)) {
        atomicAdd(&warp_hist[(unsigned int)topk_radix_to_ordered_u16(r1) >> 8], 1u);
    }
}

__device__ __forceinline__ void topk_radix_emit_valid(
    unsigned short raw,
    unsigned int idx,
    unsigned int* __restrict__ s_above_cnt,
    unsigned short* __restrict__ s_above_ord,
    unsigned int* __restrict__ s_above_idx
) {
    if (!topk_radix_is_valid_u16(raw)) {
        return;
    }
    const unsigned int slot = atomicAdd(s_above_cnt, 1u);
    if (slot < TOPK_RADIX_MAX_K) {
        s_above_ord[slot] = topk_radix_to_ordered_u16(raw);
        s_above_idx[slot] = idx;
    }
}

__device__ __forceinline__ void topk_radix_pass2_gather_elem(
    unsigned short raw,
    unsigned int idx,
    unsigned int thresh_hi,
    unsigned int* __restrict__ s_above_cnt,
    unsigned short* __restrict__ s_above_ord,
    unsigned int* __restrict__ s_above_idx,
    unsigned int* __restrict__ s_cand_cnt,
    unsigned char* __restrict__ s_cand_lo,
    unsigned int* __restrict__ s_cand_idx,
    unsigned int* __restrict__ s_fine
) {
    if (!topk_radix_is_valid_u16(raw)) {
        return;
    }
    const unsigned short ord = topk_radix_to_ordered_u16(raw);
    const unsigned int hi = (unsigned int)(ord >> 8);
    if (hi < thresh_hi) {
        return;
    }
    if (hi > thresh_hi) {
        const unsigned int slot = atomicAdd(s_above_cnt, 1u);
        if (slot < TOPK_RADIX_MAX_K) {
            s_above_ord[slot] = ord;
            s_above_idx[slot] = idx;
        }
    } else {
        const unsigned int lo = (unsigned int)(ord & 0xFFu);
        const unsigned int slot = atomicAdd(s_cand_cnt, 1u);
        if (slot < TOPK_RADIX_SMEM_CAND_CAP) {
            s_cand_lo[slot] = (unsigned char)lo;
            s_cand_idx[slot] = idx;
        }
        atomicAdd(&s_fine[lo], 1u);
    }
}

__device__ __forceinline__ void topk_radix_pass2_hist_elem(
    unsigned int* __restrict__ warp_hist,
    unsigned short raw,
    unsigned int idx,
    unsigned int thresh_hi,
    unsigned int* __restrict__ s_above_cnt,
    unsigned short* __restrict__ s_above_ord,
    unsigned int* __restrict__ s_above_idx
) {
    if (!topk_radix_is_valid_u16(raw)) {
        return;
    }
    const unsigned short ord = topk_radix_to_ordered_u16(raw);
    const unsigned int hi = (unsigned int)(ord >> 8);
    if (hi < thresh_hi) {
        return;
    }
    if (hi > thresh_hi) {
        const unsigned int slot = atomicAdd(s_above_cnt, 1u);
        if (slot < TOPK_RADIX_MAX_K) {
            s_above_ord[slot] = ord;
            s_above_idx[slot] = idx;
        }
    } else {
        atomicAdd(&warp_hist[(unsigned int)ord & 0xFFu], 1u);
    }
}

__device__ __forceinline__ void topk_radix_pass3_notie_elem(
    unsigned short raw,
    unsigned int idx,
    unsigned int thresh_hi,
    unsigned int thresh_lo,
    unsigned int* __restrict__ s_above_cnt,
    unsigned short* __restrict__ s_above_ord,
    unsigned int* __restrict__ s_above_idx
) {
    if (!topk_radix_is_valid_u16(raw)) {
        return;
    }
    const unsigned short ord = topk_radix_to_ordered_u16(raw);
    if ((unsigned int)(ord >> 8) == thresh_hi && ((unsigned int)ord & 0xFFu) >= thresh_lo) {
        const unsigned int slot = atomicAdd(s_above_cnt, 1u);
        if (slot < TOPK_RADIX_MAX_K) {
            s_above_ord[slot] = ord;
            s_above_idx[slot] = idx;
        }
    }
}

extern "C" __global__ void __launch_bounds__(512, 3) topk_bf16_radix(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ top_indices,
    float* __restrict__ top_logits,
    unsigned int num_rows,
    unsigned int vocab,
    unsigned int k
) {
    const unsigned int row = blockIdx.x;
    if (row >= num_rows || vocab == 0u || k == 0u || k > TOPK_RADIX_MAX_K || k > vocab) {
        return;
    }

    __shared__ union {
        unsigned int hist[16][256];
        struct {
            unsigned int cand_idx[TOPK_RADIX_SMEM_CAND_CAP];
            unsigned int fine[256];
            unsigned char cand_lo[TOPK_RADIX_SMEM_CAND_CAP];
        } gather;
    } s_work;

    __shared__ unsigned short s_above_ord[TOPK_RADIX_MAX_K];
    __shared__ unsigned int s_above_idx[TOPK_RADIX_MAX_K];
    __shared__ unsigned int s_warp_id[16];
    __shared__ unsigned int s_warp_pos[16];
    __shared__ unsigned int s_total_valid;
    __shared__ unsigned int s_thresh_hi;
    __shared__ unsigned int s_thresh_hi_cnt;
    __shared__ unsigned int s_need_in_hi;
    __shared__ unsigned int s_above_cnt;
    __shared__ unsigned int s_cand_cnt;
    __shared__ unsigned int s_thresh_lo;
    __shared__ unsigned int s_above_lo;
    __shared__ unsigned int s_need_eq;
    __shared__ unsigned int s_count_eq;
    __shared__ unsigned int s_eq_cnt;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;

    const __nv_bfloat16* __restrict__ row_logits = logits + (size_t)row * (size_t)vocab;
    const unsigned short* __restrict__ row_raw =
        reinterpret_cast<const unsigned short*>(row_logits);

    // Align `row_raw + prologue` to 16 bytes for `uint4` loads (8 BF16/load).
    const unsigned int misalign_bytes = (unsigned int)((unsigned long long)row_raw & 15ull);
    const bool can_vec = ((misalign_bytes & 1u) == 0u);
    unsigned int prologue = 0u;
    unsigned int n_vec = 0u;
    unsigned int vec_end = 0u;
    if (can_vec) {
        prologue = (misalign_bytes == 0u) ? 0u : ((16u - misalign_bytes) >> 1);
        if (prologue > vocab) {
            prologue = vocab;
        }
        const unsigned int rem = vocab - prologue;
        n_vec = rem >> 3;
        vec_end = prologue + (n_vec << 3);
    }

    // Zero per-warp coarse histogram (16 * 256 = 4096 u32, 8 stores/thread).
    #pragma unroll
    for (unsigned int i = tid; i < 4096u; i += 512u) {
        s_work.hist[i >> 8][i & 255u] = 0u;
    }
    if (tid == 0u) {
        s_above_cnt = 0u;
        s_cand_cnt = 0u;
        s_eq_cnt = 0u;
    }
    __syncthreads();

    // Pass 1: upper-byte (ord >> 8) per-warp privatized histogram.
    unsigned int* __restrict__ warp_hist = s_work.hist[warp];
    if (tid < prologue) {
        const unsigned short raw = row_raw[tid];
        if (topk_radix_is_valid_u16(raw)) {
            atomicAdd(&warp_hist[(unsigned int)topk_radix_to_ordered_u16(raw) >> 8], 1u);
        }
    }
    if (n_vec > 0u) {
        const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
        for (unsigned int idx = tid; idx < n_vec; idx += 512u) {
            const uint4 v4 = row_v4[idx];
            topk_radix_pass1_pair(warp_hist, v4.x);
            topk_radix_pass1_pair(warp_hist, v4.y);
            topk_radix_pass1_pair(warp_hist, v4.z);
            topk_radix_pass1_pair(warp_hist, v4.w);
        }
    }
    for (unsigned int c = vec_end + tid; c < vocab; c += 512u) {
        const unsigned short raw = row_raw[c];
        if (topk_radix_is_valid_u16(raw)) {
            atomicAdd(&warp_hist[(unsigned int)topk_radix_to_ordered_u16(raw) >> 8], 1u);
        }
    }
    __syncthreads();

    // Merge 16 warp histograms into `s_work.hist[0][0..255]`.
    if (tid < 256u) {
        unsigned int sum = 0u;
        #pragma unroll
        for (unsigned int w = 0u; w < 16u; ++w) {
            sum += s_work.hist[w][tid];
        }
        s_work.hist[0][tid] = sum;
    }
    __syncthreads();

    if (tid == 0u) {
        unsigned int total_valid = 0u;
        for (int b = 255; b >= 0; --b) {
            total_valid += s_work.hist[0][b];
        }
        s_total_valid = total_valid;
        if (total_valid > k) {
            unsigned int above = 0u;
            int hi = 255;
            for (; hi > 0; --hi) {
                if (above + s_work.hist[0][hi] >= k) {
                    break;
                }
                above += s_work.hist[0][hi];
            }
            s_thresh_hi = (unsigned int)hi;
            s_thresh_hi_cnt = s_work.hist[0][hi];
            s_need_in_hi = k - above;
        } else {
            s_thresh_hi = 0u;
            s_thresh_hi_cnt = 0u;
            s_need_in_hi = 0u;
        }
    }
    __syncthreads();

    const unsigned int total_valid = s_total_valid;
    if (total_valid == 0u) {
        if (tid < k) {
            const size_t out_pos = (size_t)row * (size_t)k + tid;
            top_indices[out_pos] = 0u;
            top_logits[out_pos] = -CUDART_INF_F;
        }
        return;
    }

    if (total_valid <= k) {
        // Every valid element belongs to top-K (at most 32 total).
        if (tid < prologue) {
            topk_radix_emit_valid(row_raw[tid], tid, &s_above_cnt, s_above_ord, s_above_idx);
        }
        if (n_vec > 0u) {
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
            for (unsigned int idx = tid; idx < n_vec; idx += 512u) {
                const uint4 v4 = row_v4[idx];
                const unsigned int c0 = prologue + (idx << 3);
                topk_radix_emit_valid((unsigned short)(v4.x & 0xFFFFu), c0 + 0u, &s_above_cnt, s_above_ord, s_above_idx);
                topk_radix_emit_valid((unsigned short)(v4.x >> 16),     c0 + 1u, &s_above_cnt, s_above_ord, s_above_idx);
                topk_radix_emit_valid((unsigned short)(v4.y & 0xFFFFu), c0 + 2u, &s_above_cnt, s_above_ord, s_above_idx);
                topk_radix_emit_valid((unsigned short)(v4.y >> 16),     c0 + 3u, &s_above_cnt, s_above_ord, s_above_idx);
                topk_radix_emit_valid((unsigned short)(v4.z & 0xFFFFu), c0 + 4u, &s_above_cnt, s_above_ord, s_above_idx);
                topk_radix_emit_valid((unsigned short)(v4.z >> 16),     c0 + 5u, &s_above_cnt, s_above_ord, s_above_idx);
                topk_radix_emit_valid((unsigned short)(v4.w & 0xFFFFu), c0 + 6u, &s_above_cnt, s_above_ord, s_above_idx);
                topk_radix_emit_valid((unsigned short)(v4.w >> 16),     c0 + 7u, &s_above_cnt, s_above_ord, s_above_idx);
            }
        }
        for (unsigned int c = vec_end + tid; c < vocab; c += 512u) {
            topk_radix_emit_valid(row_raw[c], c, &s_above_cnt, s_above_ord, s_above_idx);
        }
        __syncthreads();
    } else if (s_thresh_hi_cnt <= TOPK_RADIX_SMEM_CAND_CAP) {
        // Fast 2-pass path: gather `hi > thresh_hi` directly into `s_above` and
        // `hi == thresh_hi` (<= 2560 items) into SMEM along with `fine[256]`.
        const unsigned int thresh_hi = s_thresh_hi;
        if (tid < 256u) {
            s_work.gather.fine[tid] = 0u;
        }
        __syncthreads();

        if (tid < prologue) {
            topk_radix_pass2_gather_elem(
                row_raw[tid], tid, thresh_hi,
                &s_above_cnt, s_above_ord, s_above_idx,
                &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine
            );
        }
        if (n_vec > 0u) {
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
            for (unsigned int idx = tid; idx < n_vec; idx += 512u) {
                const uint4 v4 = row_v4[idx];
                const unsigned int c0 = prologue + (idx << 3);
                topk_radix_pass2_gather_elem((unsigned short)(v4.x & 0xFFFFu), c0 + 0u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                topk_radix_pass2_gather_elem((unsigned short)(v4.x >> 16),     c0 + 1u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                topk_radix_pass2_gather_elem((unsigned short)(v4.y & 0xFFFFu), c0 + 2u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                topk_radix_pass2_gather_elem((unsigned short)(v4.y >> 16),     c0 + 3u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                topk_radix_pass2_gather_elem((unsigned short)(v4.z & 0xFFFFu), c0 + 4u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                topk_radix_pass2_gather_elem((unsigned short)(v4.z >> 16),     c0 + 5u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                topk_radix_pass2_gather_elem((unsigned short)(v4.w & 0xFFFFu), c0 + 6u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                topk_radix_pass2_gather_elem((unsigned short)(v4.w >> 16),     c0 + 7u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
            }
        }
        for (unsigned int c = vec_end + tid; c < vocab; c += 512u) {
            topk_radix_pass2_gather_elem(
                row_raw[c], c, thresh_hi,
                &s_above_cnt, s_above_ord, s_above_idx,
                &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine
            );
        }
        __syncthreads();

        if (tid == 0u) {
            const unsigned int need = s_need_in_hi;
            unsigned int above_lo = 0u;
            int lo = 255;
            for (; lo > 0; --lo) {
                if (above_lo + s_work.gather.fine[lo] >= need) {
                    break;
                }
                above_lo += s_work.gather.fine[lo];
            }
            s_thresh_lo = (unsigned int)lo;
            s_above_lo = above_lo;
            s_need_eq = need - above_lo;
            s_count_eq = s_work.gather.fine[lo];
        }
        __syncthreads();

        const unsigned int n_cand =
            (s_cand_cnt < TOPK_RADIX_SMEM_CAND_CAP) ? s_cand_cnt : TOPK_RADIX_SMEM_CAND_CAP;
        const unsigned int thresh_lo = s_thresh_lo;
        const unsigned int need_eq = s_need_eq;
        const unsigned int count_eq = s_count_eq;

        // Collect strictly-above-`thresh_lo` (and `== thresh_lo` if no tie-break).
        const bool take_all_eq = (count_eq <= need_eq);
        for (unsigned int i = tid; i < n_cand; i += 512u) {
            const unsigned int lo = (unsigned int)s_work.gather.cand_lo[i];
            if (lo > thresh_lo || (take_all_eq && lo == thresh_lo)) {
                const unsigned int slot = atomicAdd(&s_above_cnt, 1u);
                if (slot < TOPK_RADIX_MAX_K) {
                    s_above_ord[slot] = (unsigned short)((thresh_hi << 8) | lo);
                    s_above_idx[slot] = s_work.gather.cand_idx[i];
                }
            }
        }
        __syncthreads();

        // Exact tie-breaking inside SMEM: extract `need_eq` smallest token IDs
        // among `cand_lo[i] == thresh_lo` without touching global memory again.
        if (!take_all_eq) {
            const unsigned short thresh_key = (unsigned short)((thresh_hi << 8) | thresh_lo);
            for (unsigned int rep = 0u; rep < need_eq; ++rep) {
                unsigned int best_id = 0xFFFFFFFFu;
                unsigned int best_pos = 0xFFFFFFFFu;
                for (unsigned int i = tid; i < n_cand; i += 512u) {
                    if ((unsigned int)s_work.gather.cand_lo[i] == thresh_lo) {
                        const unsigned int cid = s_work.gather.cand_idx[i];
                        if (best_pos == 0xFFFFFFFFu || cid < best_id) {
                            best_id = cid;
                            best_pos = i;
                        }
                    }
                }
                #pragma unroll
                for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
                    const unsigned int other_id = __shfl_down_sync(0xFFFFFFFFu, best_id, offset);
                    const unsigned int other_pos = __shfl_down_sync(0xFFFFFFFFu, best_pos, offset);
                    if (other_pos != 0xFFFFFFFFu &&
                        (best_pos == 0xFFFFFFFFu || other_id < best_id)) {
                        best_id = other_id;
                        best_pos = other_pos;
                    }
                }
                if (lane == 0u) {
                    s_warp_id[warp] = best_id;
                    s_warp_pos[warp] = best_pos;
                }
                __syncthreads();
                if (warp == 0u) {
                    unsigned int w_id = (lane < 16u) ? s_warp_id[lane] : 0xFFFFFFFFu;
                    unsigned int w_pos = (lane < 16u) ? s_warp_pos[lane] : 0xFFFFFFFFu;
                    #pragma unroll
                    for (unsigned int offset = 8u; offset > 0u; offset >>= 1u) {
                        const unsigned int other_id = __shfl_down_sync(0xFFFFFFFFu, w_id, offset);
                        const unsigned int other_pos = __shfl_down_sync(0xFFFFFFFFu, w_pos, offset);
                        if (other_pos != 0xFFFFFFFFu &&
                            (w_pos == 0xFFFFFFFFu || other_id < w_id)) {
                            w_id = other_id;
                            w_pos = other_pos;
                        }
                    }
                    if (lane == 0u && w_pos != 0xFFFFFFFFu) {
                        const unsigned int slot = s_above_cnt;
                        if (slot < TOPK_RADIX_MAX_K) {
                            s_above_ord[slot] = thresh_key;
                            s_above_idx[slot] = w_id;
                            s_above_cnt = slot + 1u;
                        }
                        s_work.gather.cand_lo[w_pos] = (unsigned char)(thresh_lo ^ 1u);
                    }
                }
                __syncthreads();
            }
        }
    } else {
        // Heavy-bucket fallback (`s_thresh_hi_cnt > 2560`, e.g. all-equal logits):
        // Pass 2 builds fine histogram in `s_work.hist[16][256]` (per-warp privatized shared-memory atomics)
        // and collects `hi > thresh_hi` into `s_above`.
        const unsigned int thresh_hi = s_thresh_hi;
        #pragma unroll
        for (unsigned int i = tid; i < 4096u; i += 512u) {
            s_work.hist[i >> 8][i & 255u] = 0u;
        }
        __syncthreads();

        if (tid < prologue) {
            topk_radix_pass2_hist_elem(
                warp_hist, row_raw[tid], tid, thresh_hi,
                &s_above_cnt, s_above_ord, s_above_idx
            );
        }
        if (n_vec > 0u) {
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
            for (unsigned int idx = tid; idx < n_vec; idx += 512u) {
                const uint4 v4 = row_v4[idx];
                const unsigned int c0 = prologue + (idx << 3);
                topk_radix_pass2_hist_elem(warp_hist, (unsigned short)(v4.x & 0xFFFFu), c0 + 0u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx);
                topk_radix_pass2_hist_elem(warp_hist, (unsigned short)(v4.x >> 16),     c0 + 1u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx);
                topk_radix_pass2_hist_elem(warp_hist, (unsigned short)(v4.y & 0xFFFFu), c0 + 2u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx);
                topk_radix_pass2_hist_elem(warp_hist, (unsigned short)(v4.y >> 16),     c0 + 3u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx);
                topk_radix_pass2_hist_elem(warp_hist, (unsigned short)(v4.z & 0xFFFFu), c0 + 4u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx);
                topk_radix_pass2_hist_elem(warp_hist, (unsigned short)(v4.z >> 16),     c0 + 5u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx);
                topk_radix_pass2_hist_elem(warp_hist, (unsigned short)(v4.w & 0xFFFFu), c0 + 6u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx);
                topk_radix_pass2_hist_elem(warp_hist, (unsigned short)(v4.w >> 16),     c0 + 7u, thresh_hi, &s_above_cnt, s_above_ord, s_above_idx);
            }
        }
        for (unsigned int c = vec_end + tid; c < vocab; c += 512u) {
            topk_radix_pass2_hist_elem(
                warp_hist, row_raw[c], c, thresh_hi,
                &s_above_cnt, s_above_ord, s_above_idx
            );
        }
        __syncthreads();

        if (tid < 256u) {
            unsigned int sum = 0u;
            #pragma unroll
            for (unsigned int w = 0u; w < 16u; ++w) {
                sum += s_work.hist[w][tid];
            }
            s_work.hist[0][tid] = sum;
        }
        __syncthreads();

        if (tid == 0u) {
            const unsigned int need = s_need_in_hi;
            unsigned int above_lo = 0u;
            int lo = 255;
            for (; lo > 0; --lo) {
                if (above_lo + s_work.hist[0][lo] >= need) {
                    break;
                }
                above_lo += s_work.hist[0][lo];
            }
            s_thresh_lo = (unsigned int)lo;
            s_above_lo = above_lo;
            s_need_eq = need - above_lo;
            s_count_eq = s_work.hist[0][lo];
        }
        __syncthreads();

        const unsigned int thresh_lo = s_thresh_lo;
        const unsigned int above_lo = s_above_lo;
        const unsigned int need_eq = s_need_eq;
        const unsigned int count_eq = s_count_eq;

        // Pass 3 (rare fallback only when `s_thresh_hi_cnt > 2560`).
        if (count_eq <= need_eq) {
            if (tid < prologue) {
                topk_radix_pass3_notie_elem(
                    row_raw[tid], tid, thresh_hi, thresh_lo,
                    &s_above_cnt, s_above_ord, s_above_idx
                );
            }
            if (n_vec > 0u) {
                const uint4* __restrict__ row_v4 =
                    reinterpret_cast<const uint4*>(row_raw + prologue);
                for (unsigned int idx = tid; idx < n_vec; idx += 512u) {
                    const uint4 v4 = row_v4[idx];
                    const unsigned int c0 = prologue + (idx << 3);
                    topk_radix_pass3_notie_elem((unsigned short)(v4.x & 0xFFFFu), c0 + 0u, thresh_hi, thresh_lo, &s_above_cnt, s_above_ord, s_above_idx);
                    topk_radix_pass3_notie_elem((unsigned short)(v4.x >> 16),     c0 + 1u, thresh_hi, thresh_lo, &s_above_cnt, s_above_ord, s_above_idx);
                    topk_radix_pass3_notie_elem((unsigned short)(v4.y & 0xFFFFu), c0 + 2u, thresh_hi, thresh_lo, &s_above_cnt, s_above_ord, s_above_idx);
                    topk_radix_pass3_notie_elem((unsigned short)(v4.y >> 16),     c0 + 3u, thresh_hi, thresh_lo, &s_above_cnt, s_above_ord, s_above_idx);
                    topk_radix_pass3_notie_elem((unsigned short)(v4.z & 0xFFFFu), c0 + 4u, thresh_hi, thresh_lo, &s_above_cnt, s_above_ord, s_above_idx);
                    topk_radix_pass3_notie_elem((unsigned short)(v4.z >> 16),     c0 + 5u, thresh_hi, thresh_lo, &s_above_cnt, s_above_ord, s_above_idx);
                    topk_radix_pass3_notie_elem((unsigned short)(v4.w & 0xFFFFu), c0 + 6u, thresh_hi, thresh_lo, &s_above_cnt, s_above_ord, s_above_idx);
                    topk_radix_pass3_notie_elem((unsigned short)(v4.w >> 16),     c0 + 7u, thresh_hi, thresh_lo, &s_above_cnt, s_above_ord, s_above_idx);
                }
            }
            for (unsigned int c = vec_end + tid; c < vocab; c += 512u) {
                topk_radix_pass3_notie_elem(
                    row_raw[c], c, thresh_hi, thresh_lo,
                    &s_above_cnt, s_above_ord, s_above_idx
                );
            }
            __syncthreads();
        } else {
            // Exact ascending-index scan for `ord == thresh_key` ties (early exit
            // as soon as `above_lo == 0 && s_eq_cnt >= need_eq`).
            const unsigned short thresh_key = (unsigned short)((thresh_hi << 8) | thresh_lo);
            for (unsigned int base = 0u; base < vocab; base += 512u) {
                if (above_lo == 0u && s_eq_cnt >= need_eq) {
                    break;
                }
                const unsigned int c = base + tid;
                bool hit_eq = false;
                if (c < vocab) {
                    const unsigned short raw = row_raw[c];
                    if (topk_radix_is_valid_u16(raw)) {
                        const unsigned short ord = topk_radix_to_ordered_u16(raw);
                        if ((unsigned int)(ord >> 8) == thresh_hi) {
                            const unsigned int lo = (unsigned int)(ord & 0xFFu);
                            if (lo > thresh_lo) {
                                const unsigned int slot = atomicAdd(&s_above_cnt, 1u);
                                if (slot < TOPK_RADIX_MAX_K) {
                                    s_above_ord[slot] = ord;
                                    s_above_idx[slot] = c;
                                }
                            } else if (lo == thresh_lo && s_eq_cnt < need_eq) {
                                hit_eq = true;
                            }
                        }
                    }
                }
                if (__syncthreads_or(hit_eq)) {
                    const unsigned int wm = __ballot_sync(0xFFFFFFFFu, hit_eq);
                    if (lane == 0u) {
                        s_warp_pos[warp] = (unsigned int)__popc(wm);
                    }
                    __syncthreads();
                    const unsigned int prev_eq = s_eq_cnt;
                    if (hit_eq) {
                        unsigned int offset = 0u;
                        #pragma unroll
                        for (unsigned int w = 0u; w < 16u; ++w) {
                            if (w < warp) {
                                offset += s_warp_pos[w];
                            }
                        }
                        const unsigned int lane_mask = (1u << lane) - 1u;
                        const unsigned int rank =
                            prev_eq + offset + (unsigned int)__popc(wm & lane_mask);
                        if (rank < need_eq) {
                            const unsigned int slot = atomicAdd(&s_above_cnt, 1u);
                            if (slot < TOPK_RADIX_MAX_K) {
                                s_above_ord[slot] = thresh_key;
                                s_above_idx[slot] = c;
                            }
                        }
                    }
                    __syncthreads();
                    if (tid == 0u) {
                        unsigned int chunk_eq = 0u;
                        #pragma unroll
                        for (unsigned int w = 0u; w < 16u; ++w) {
                            chunk_eq += s_warp_pos[w];
                        }
                        s_eq_cnt = prev_eq + chunk_eq;
                    }
                    __syncthreads();
                }
            }
            __syncthreads();
        }
    }

    // Sort <= 32 collected winners strictly descending by `(ord, ~idx)`.
    if (tid == 0u) {
        const unsigned int count = (s_above_cnt < k) ? s_above_cnt : k;
        for (unsigned int i = 1u; i < count; ++i) {
            const unsigned short cur_ord = s_above_ord[i];
            const unsigned int cur_idx = s_above_idx[i];
            int j = (int)i - 1;
            while (j >= 0 &&
                   (s_above_ord[j] < cur_ord ||
                    (s_above_ord[j] == cur_ord && s_above_idx[j] > cur_idx))) {
                s_above_ord[j + 1] = s_above_ord[j];
                s_above_idx[j + 1] = s_above_idx[j];
                --j;
            }
            s_above_ord[j + 1] = cur_ord;
            s_above_idx[j + 1] = cur_idx;
        }
        s_above_cnt = count;
    }
    __syncthreads();

    if (tid < k) {
        const size_t out_pos = (size_t)row * (size_t)k + tid;
        if (tid < s_above_cnt) {
            const unsigned int idx = s_above_idx[tid];
            top_indices[out_pos] = idx;
            top_logits[out_pos] = __bfloat162float(row_logits[idx]);
        } else {
            top_indices[out_pos] = 0u;
            top_logits[out_pos] = -CUDART_INF_F;
        }
    }
}
