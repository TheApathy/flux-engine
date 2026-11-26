// SPDX-License-Identifier: AGPL-3.0-only
// GLM-5.3 DSA kpool4 selector by radix select (ATLAS_GLM53_DSA_TOPK_RADIX=1).
//
// Bit-identical replacement for atlas_glm53_dsa_topk_k4 (same ABI, grid and
// output). The reference keeps a running top-512 through a 1024-wide bitonic
// network per 512-pool chunk (pool_count/512 full sorts per row). This kernel
// maps every admitted pool to a unique 64-bit key whose descending order is
// exactly glm53_dsa_better's order:
//   hi = 0xffffffff for NaN (NaN beats every number), otherwise the
//        order-preserving image of the score with -0.0 folded onto +0.0
//        (they compare equal in the reference, which then breaks the tie on
//        the pool index);
//   lo = 0xffffffff - pool (a smaller pool wins a tie).
// It finds the 512th largest key with an 8-bit radix select, gathers the keys
// at or above it (exactly min(512, admitted) of them, keys are unique), sorts
// them descending with a 512-wide bitonic network, and writes the same
// expanded indices and raw tail as the reference.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define GLM53_DSA_RX_TOPK 2048U
#define GLM53_DSA_RX_KPOOL 4U
#define GLM53_DSA_RX_SELECTED 512U
#define GLM53_DSA_RX_OUTPUT_WIDTH 2051U
#define GLM53_DSA_RX_MAX_QUERIES 8192U
#define GLM53_DSA_RX_MAX_POOLS 262144U
#define GLM53_DSA_RX_MAX_POSITIONS 1048576U
#define GLM53_DSA_RX_THREADS 256U
#define GLM53_DSA_RX_MAX_GRID_YZ 65535ULL

__device__ __forceinline__ unsigned long long glm53_rx_key(float score, unsigned pool) {
    unsigned hi;
    if (isnan(score)) {
        hi = 0xffffffffU;
    } else {
        unsigned bits = __float_as_uint(score);
        if (bits == 0x80000000U) bits = 0U;  // -0.0 == +0.0 in the reference
        hi = (bits & 0x80000000U) ? ~bits : (bits | 0x80000000U);
    }
    return ((unsigned long long)hi << 32) | (unsigned long long)(0xffffffffU - pool);
}

extern "C" __global__ void __launch_bounds__(GLM53_DSA_RX_THREADS, 1)
atlas_glm53_dsa_topk_k4_radix(
        const float * __restrict__ scores,
        const unsigned char * __restrict__ pool_validity,
        const unsigned int * __restrict__ sequence_lengths,
        const unsigned int * __restrict__ query_positions,
        const unsigned char * __restrict__ query_validity,
        const unsigned char * __restrict__ tail_validity,
        int * __restrict__ output_indices,
        unsigned int batch, unsigned int query_count,
        unsigned int pool_count, unsigned int kv_capacity) {
    const unsigned long long rows = (unsigned long long)batch * query_count;
    const unsigned long long expected_y =
        rows < GLM53_DSA_RX_MAX_GRID_YZ ? rows : GLM53_DSA_RX_MAX_GRID_YZ;
    const unsigned long long expected_z = expected_y == 0ULL ? 0ULL :
        (rows + expected_y - 1ULL) / expected_y;
    if (scores == nullptr || pool_validity == nullptr ||
        sequence_lengths == nullptr || query_positions == nullptr ||
        query_validity == nullptr || tail_validity == nullptr ||
        output_indices == nullptr ||
        batch == 0U || query_count == 0U ||
        query_count > GLM53_DSA_RX_MAX_QUERIES || pool_count == 0U ||
        pool_count > GLM53_DSA_RX_MAX_POOLS || kv_capacity == 0U ||
        kv_capacity > GLM53_DSA_RX_MAX_POSITIONS ||
        pool_count > (kv_capacity + GLM53_DSA_RX_KPOOL - 1U) / GLM53_DSA_RX_KPOOL ||
        blockDim.x != GLM53_DSA_RX_THREADS || gridDim.x != 1U ||
        (unsigned long long)gridDim.y != expected_y ||
        (unsigned long long)gridDim.z != expected_z ||
        expected_z > GLM53_DSA_RX_MAX_GRID_YZ) {
        return;
    }
    const unsigned long long row =
        (unsigned long long)blockIdx.y + (unsigned long long)gridDim.y * blockIdx.z;
    if (row >= rows) {
        return;
    }
    const unsigned lane = threadIdx.x;
    const unsigned long long output_base = row * GLM53_DSA_RX_OUTPUT_WIDTH;
    for (unsigned index = lane; index < GLM53_DSA_RX_OUTPUT_WIDTH; index += GLM53_DSA_RX_THREADS) {
        output_indices[output_base + index] = -1;
    }
    if (query_validity[row] == 0U) {
        return;
    }
    const unsigned long long batch_index = row / query_count;
    const unsigned sequence_length = sequence_lengths[batch_index];
    const unsigned query_position = query_positions[row];
    if (sequence_length == 0U || sequence_length > kv_capacity ||
        query_position >= sequence_length) {
        return;
    }
    const unsigned complete_pools = sequence_length / GLM53_DSA_RX_KPOOL;
    if (complete_pools > pool_count) {
        return;
    }
    // Admitted pools are a prefix: pool < complete_pools and pool*4+3 <= query
    // position, then the per-pool validity byte.
    const unsigned visible_end = (query_position + 1U) / GLM53_DSA_RX_KPOOL;
    const unsigned limit = min(min(pool_count, complete_pools), visible_end);
    const float * row_scores = scores + row * pool_count;
    const unsigned char * validity = pool_validity + batch_index * pool_count;

    __shared__ unsigned histogram[256];
    __shared__ unsigned long long prefix_shared;
    __shared__ unsigned remaining_shared;
    __shared__ unsigned admitted_shared;
    __shared__ unsigned long long selected_keys[GLM53_DSA_RX_SELECTED];
    __shared__ unsigned selected_count_shared;

    if (lane == 0U) {
        prefix_shared = 0ULL;
        admitted_shared = 0U;
        selected_count_shared = 0U;
    }
    __syncthreads();
    {
        unsigned local = 0U;
        for (unsigned pool = lane; pool < limit; pool += GLM53_DSA_RX_THREADS) {
            local += validity[pool] != 0U ? 1U : 0U;
        }
        if (local != 0U) atomicAdd(&admitted_shared, local);
    }
    __syncthreads();
    const unsigned admitted = admitted_shared;
    const unsigned want = min(admitted, GLM53_DSA_RX_SELECTED);
    // Radix select: find the want-th largest key. The bits of `prefix` above
    // `shift + 8` are fixed; `remaining` counts how many keys still to take
    // from the bucket that matches the prefix.
    unsigned long long threshold = 0ULL;
    if (want != 0U && want < admitted) {
        if (lane == 0U) remaining_shared = want;
        unsigned long long mask = 0ULL;
        for (int shift = 56; shift >= 0; shift -= 8) {
            histogram[lane] = 0U;
            __syncthreads();
            const unsigned long long prefix = prefix_shared;
            for (unsigned pool = lane; pool < limit; pool += GLM53_DSA_RX_THREADS) {
                if (validity[pool] == 0U) continue;
                const unsigned long long key = glm53_rx_key(row_scores[pool], pool);
                if ((key & mask) == prefix) {
                    atomicAdd(&histogram[(unsigned)(key >> shift) & 255U], 1U);
                }
            }
            __syncthreads();
            if (lane == 0U) {
                unsigned need = remaining_shared;
                unsigned digit = 255U;
                for (;; --digit) {
                    const unsigned count = histogram[digit];
                    if (count >= need) break;
                    need -= count;
                    if (digit == 0U) break;
                }
                prefix_shared = prefix | ((unsigned long long)digit << shift);
                remaining_shared = need;
            }
            __syncthreads();
            mask |= 255ULL << shift;
        }
        threshold = prefix_shared;  // the want-th largest key itself
    }
    // Gather every admitted key >= threshold (exactly `want` keys).
    for (unsigned pool = lane; pool < limit; pool += GLM53_DSA_RX_THREADS) {
        if (validity[pool] == 0U) continue;
        const unsigned long long key = glm53_rx_key(row_scores[pool], pool);
        if (key >= threshold) {
            const unsigned slot = atomicAdd(&selected_count_shared, 1U);
            if (slot < GLM53_DSA_RX_SELECTED) selected_keys[slot] = key;
        }
    }
    __syncthreads();
    for (unsigned slot = want + lane; slot < GLM53_DSA_RX_SELECTED; slot += GLM53_DSA_RX_THREADS) {
        selected_keys[slot] = 0ULL;  // below every admitted key (lo != 0 needs pool < 2^32-1)
    }
    __syncthreads();
    // Descending bitonic sort of the 512 keys.
    for (unsigned width = 2U; width <= GLM53_DSA_RX_SELECTED; width <<= 1U) {
        for (unsigned stride = width >> 1U; stride != 0U; stride >>= 1U) {
            for (unsigned item = lane; item < GLM53_DSA_RX_SELECTED; item += GLM53_DSA_RX_THREADS) {
                const unsigned partner = item ^ stride;
                if (partner > item) {
                    const bool descending = (item & width) == 0U;
                    const unsigned long long a = selected_keys[item];
                    const unsigned long long b = selected_keys[partner];
                    if (descending ? (a < b) : (a > b)) {
                        selected_keys[item] = b;
                        selected_keys[partner] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (unsigned output = lane; output < want * GLM53_DSA_RX_KPOOL; output += GLM53_DSA_RX_THREADS) {
        const unsigned rank = output / GLM53_DSA_RX_KPOOL;
        const unsigned offset = output % GLM53_DSA_RX_KPOOL;
        const unsigned pool = 0xffffffffU - (unsigned)(selected_keys[rank] & 0xffffffffULL);
        output_indices[output_base + output] = (int)(pool * GLM53_DSA_RX_KPOOL + offset);
    }
    if (lane == 0U) {
        // Raw tail: identical to the reference.
        const unsigned selected_count = want;
        const unsigned tail_output = selected_count * GLM53_DSA_RX_KPOOL;
        const unsigned visible_count = query_position + 1U;
        const unsigned tail_count = visible_count % GLM53_DSA_RX_KPOOL;
        const unsigned tail_start = visible_count - tail_count;
        #pragma unroll
        for (unsigned offset = 0U; offset < GLM53_DSA_RX_KPOOL - 1U; ++offset) {
            const unsigned index = tail_start + offset;
            const unsigned pool = index / GLM53_DSA_RX_KPOOL;
            const unsigned slot = index % GLM53_DSA_RX_KPOOL;
            const bool valid = pool < complete_pools
                ? pool_validity[batch_index * pool_count + pool] != 0U
                : (slot < GLM53_DSA_RX_KPOOL - 1U &&
                   tail_validity[batch_index * (GLM53_DSA_RX_KPOOL - 1U) + slot] != 0U);
            if (offset < tail_count && index < sequence_length && valid) {
                output_indices[output_base + tail_output + offset] = (int)index;
            }
        }
    }
}
