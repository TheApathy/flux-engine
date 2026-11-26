// SPDX-License-Identifier: AGPL-3.0-only
// GLM-5.3 DSA index score on tensor cores (ATLAS_GLM53_DSA_SCORE_MMA=1).
//
// Same ABI and semantics as atlas_glm53_dsa_score_rows_bf16:
//   score[row][pool] = sum_h (w[row][h] / sqrt(32)) * relu((q[row][h] . k[pool]) / sqrt(128))
// but NOT bit-identical: the 128-channel dots are mma.sync bf16 with fp32
// accumulation and the 32-head sum is a shuffle tree. Quality-gated: the
// top-k selection downstream can flip near-tied pools.
//
// One CTA = 4 query rows (ragged last CTA allowed when B == 1) (M = 4 x 32 heads = 128) x all pools, streamed in
// 64-pool key tiles through a 3-slot cp.async ring. Warp w owns row w>>1,
// heads 16*(w&1)..+15, with its Q fragments in registers.

#include <cuda_bf16.h>
#include <float.h>
#include <stdint.h>

#define GLM53_DSA_SMMA_HEADS 32U
#define GLM53_DSA_SMMA_DIM 128U
#define GLM53_DSA_SMMA_MAX_QUERIES 8192U
#define GLM53_DSA_SMMA_MAX_POOLS 262144U
#define GLM53_DSA_SMMA_ROWS 4U
#define GLM53_DSA_SMMA_THREADS 256U
#define GLM53_DSA_SMMA_BN 64U
#define GLM53_DSA_SMMA_STAGES 3U
#define GLM53_DSA_SMMA_INV_SQRT_DIM 0.08838834764831844055f
#define GLM53_DSA_SMMA_INV_SQRT_HEADS 0.17677669529663688110f
#define GLM53_DSA_SMMA_TILE_ELEMS (GLM53_DSA_SMMA_BN * GLM53_DSA_SMMA_DIM)
#define GLM53_DSA_SMMA_OFF_RED (GLM53_DSA_SMMA_STAGES * GLM53_DSA_SMMA_TILE_ELEMS * 2U)
#define GLM53_DSA_SMMA_SHARED (GLM53_DSA_SMMA_OFF_RED + GLM53_DSA_SMMA_ROWS * GLM53_DSA_SMMA_BN * 4U)

// 128-element (256 B) key rows: 16-byte chunk c of row r lives at c ^ (r & 7).
__device__ __forceinline__ unsigned glm53_smma_swz(unsigned row, unsigned col) {
    return row * GLM53_DSA_SMMA_DIM + ((((col >> 3U) ^ (row & 7U)) << 3U) | (col & 7U));
}

__device__ __forceinline__ void glm53_smma_cp16(void * shared, const void * global, bool fill) {
    const unsigned address = (unsigned)__cvta_generic_to_shared(shared);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(global), "r"(fill ? 16 : 0));
}

__device__ __forceinline__ void glm53_smma_stage(
        __nv_bfloat16 * tile, const __nv_bfloat16 * keys, unsigned first, unsigned pool_count,
        unsigned thread) {
    // 64 pools x 16 chunks = 1024 chunks; 8 lanes cover 128 contiguous bytes.
    #pragma unroll
    for (unsigned i = 0U; i < 4U; ++i) {
        const unsigned chunk = thread + i * GLM53_DSA_SMMA_THREADS;
        const unsigned pool = chunk >> 4U;
        const unsigned col = (chunk & 15U) * 8U;
        const bool inside = first + pool < pool_count;
        glm53_smma_cp16(tile + glm53_smma_swz(pool, col),
                        keys + (unsigned long long)(inside ? first + pool : 0U) * GLM53_DSA_SMMA_DIM + col,
                        inside);
    }
    asm volatile("cp.async.commit_group;");
}

extern "C" __global__ void __launch_bounds__(GLM53_DSA_SMMA_THREADS)
atlas_glm53_dsa_score_mma_bf16(
        const __nv_bfloat16 * __restrict__ queries,
        const __nv_bfloat16 * __restrict__ raw_head_weights,
        const __nv_bfloat16 * __restrict__ pool_keys,
        const unsigned char * __restrict__ pool_validity,
        float * __restrict__ output_scores,
        unsigned int batch, unsigned int query_count,
        unsigned int pool_count) {
    const unsigned long long rows = (unsigned long long)batch * query_count;
    if (batch == 0U || query_count == 0U || query_count > GLM53_DSA_SMMA_MAX_QUERIES ||
        pool_count == 0U || pool_count > GLM53_DSA_SMMA_MAX_POOLS ||
        (batch != 1U && query_count % GLM53_DSA_SMMA_ROWS != 0U) ||
        blockDim.x != GLM53_DSA_SMMA_THREADS ||
        (unsigned long long)gridDim.x != (rows + GLM53_DSA_SMMA_ROWS - 1ULL) / GLM53_DSA_SMMA_ROWS ||
        gridDim.y == 0U || gridDim.y > (pool_count + GLM53_DSA_SMMA_BN - 1U) / GLM53_DSA_SMMA_BN ||
        gridDim.z != 1U) {
        return;
    }
    extern __shared__ __align__(128) unsigned char smma_shared[];
    __nv_bfloat16 * ring = (__nv_bfloat16 *)smma_shared;
    float * red = (float *)(smma_shared + GLM53_DSA_SMMA_OFF_RED);

    const unsigned thread = threadIdx.x;
    const unsigned lane = thread & 31U;
    const unsigned warp = thread >> 5U;
    const unsigned local_row = warp >> 1U;
    const unsigned hg = warp & 1U;
    const unsigned g = lane >> 2U;
    const unsigned q4 = lane & 3U;
    // A ragged last CTA (B1 only) computes a clamped row and stores nothing for it.
    const unsigned long long raw_row = (unsigned long long)blockIdx.x * GLM53_DSA_SMMA_ROWS + local_row;
    const bool live = raw_row < rows;
    const unsigned long long row = live ? raw_row : rows - 1ULL;
    // B > 1 requires query_count % 4 == 0, so a CTA never straddles two batch entries.
    const unsigned long long batch_index = row / query_count;
    const __nv_bfloat16 * keys = pool_keys + batch_index * pool_count * (unsigned long long)GLM53_DSA_SMMA_DIM;
    const unsigned char * validity = pool_validity + batch_index * pool_count;

    // gridDim.y splits the pool tiles (decode: one row spread over many SMs).
    // Scores are per pool, so a split never changes any value.
    const unsigned all_tiles = (pool_count + GLM53_DSA_SMMA_BN - 1U) / GLM53_DSA_SMMA_BN;
    const unsigned per_split = (all_tiles + gridDim.y - 1U) / gridDim.y;
    const unsigned tile0 = blockIdx.y * per_split;
    const unsigned tiles = tile0 < all_tiles ? min(per_split, all_tiles - tile0) : 0U;
    for (unsigned s = 0U; s + 1U < GLM53_DSA_SMMA_STAGES; ++s) {
        if (s < tiles) {
            glm53_smma_stage(ring + s * GLM53_DSA_SMMA_TILE_ELEMS, keys, (tile0 + s) * GLM53_DSA_SMMA_BN,
                             pool_count, thread);
        } else {
            asm volatile("cp.async.commit_group;");
        }
    }
    // Q fragments: heads 16hg + g (+8), 8 k-steps of 16 channels.
    unsigned qa[8][4];
    {
        const unsigned * q_lo = (const unsigned *)(queries +
            (row * GLM53_DSA_SMMA_HEADS + hg * 16U + g) * GLM53_DSA_SMMA_DIM + 2U * q4);
        const unsigned * q_hi = q_lo + 8U * GLM53_DSA_SMMA_DIM / 2U;
        #pragma unroll
        for (unsigned kk = 0U; kk < 8U; ++kk) {
            qa[kk][0] = __ldg(q_lo + kk * 8U);
            qa[kk][1] = __ldg(q_hi + kk * 8U);
            qa[kk][2] = __ldg(q_lo + kk * 8U + 4U);
            qa[kk][3] = __ldg(q_hi + kk * 8U + 4U);
        }
    }
    const float w_lo = __bfloat162float(raw_head_weights[row * GLM53_DSA_SMMA_HEADS + hg * 16U + g]) *
                       GLM53_DSA_SMMA_INV_SQRT_HEADS;
    const float w_hi = __bfloat162float(raw_head_weights[row * GLM53_DSA_SMMA_HEADS + hg * 16U + g + 8U]) *
                       GLM53_DSA_SMMA_INV_SQRT_HEADS;
    float * out = output_scores + row * pool_count;

    for (unsigned t = 0U; t < tiles; ++t) {
        asm volatile("cp.async.wait_group %0;" :: "n"(GLM53_DSA_SMMA_STAGES - 2U));
        __syncthreads();
        // Refill the slot consumed in iteration t-1 (all warps passed the barrier).
        if (t + GLM53_DSA_SMMA_STAGES - 1U < tiles) {
            glm53_smma_stage(ring + ((t + GLM53_DSA_SMMA_STAGES - 1U) % GLM53_DSA_SMMA_STAGES) * GLM53_DSA_SMMA_TILE_ELEMS,
                             keys, (tile0 + t + GLM53_DSA_SMMA_STAGES - 1U) * GLM53_DSA_SMMA_BN, pool_count, thread);
        } else {
            asm volatile("cp.async.commit_group;");
        }
        const __nv_bfloat16 * tile = ring + (t % GLM53_DSA_SMMA_STAGES) * GLM53_DSA_SMMA_TILE_ELEMS;
        float part[8][2];
        #pragma unroll
        for (unsigned nt = 0U; nt < 8U; ++nt) {
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (unsigned kk = 0U; kk < 8U; kk += 2U) {
                unsigned b[4];
                const unsigned pool = nt * 8U + (lane & 7U);
                const unsigned col = kk * 16U + ((lane >> 3U) << 3U);
                const unsigned address = (unsigned)__cvta_generic_to_shared(tile + glm53_smma_swz(pool, col));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(address));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                    : "r"(qa[kk][0]), "r"(qa[kk][1]), "r"(qa[kk][2]), "r"(qa[kk][3]), "r"(b[0]), "r"(b[1]));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                    : "r"(qa[kk + 1U][0]), "r"(qa[kk + 1U][1]), "r"(qa[kk + 1U][2]), "r"(qa[kk + 1U][3]),
                      "r"(b[2]), "r"(b[3]));
            }
            // Heads g and g+8 for pools 8nt + 2q4 + {0,1}; then sum the 8 g-lanes.
            #pragma unroll
            for (unsigned j = 0U; j < 2U; ++j) {
                float v = w_lo * fmaxf(c[j] * GLM53_DSA_SMMA_INV_SQRT_DIM, 0.0f) +
                          w_hi * fmaxf(c[j + 2U] * GLM53_DSA_SMMA_INV_SQRT_DIM, 0.0f);
                v += __shfl_xor_sync(0xffffffffU, v, 4U);
                v += __shfl_xor_sync(0xffffffffU, v, 8U);
                v += __shfl_xor_sync(0xffffffffU, v, 16U);
                part[nt][j] = v;
            }
        }
        // Combine the two head halves of each row through shared memory.
        if (hg == 1U && g == 0U) {
            #pragma unroll
            for (unsigned nt = 0U; nt < 8U; ++nt) {
                red[local_row * GLM53_DSA_SMMA_BN + nt * 8U + 2U * q4] = part[nt][0];
                red[local_row * GLM53_DSA_SMMA_BN + nt * 8U + 2U * q4 + 1U] = part[nt][1];
            }
        }
        __syncthreads();
        if (hg == 0U && g == 0U && live) {
            const unsigned first = (tile0 + t) * GLM53_DSA_SMMA_BN;
            #pragma unroll
            for (unsigned nt = 0U; nt < 8U; ++nt) {
                #pragma unroll
                for (unsigned j = 0U; j < 2U; ++j) {
                    const unsigned n = nt * 8U + 2U * q4 + j;
                    const unsigned pool = first + n;
                    if (pool < pool_count) {
                        out[pool] = validity[pool] != 0U
                            ? part[nt][j] + red[local_row * GLM53_DSA_SMMA_BN + n] : -FLT_MAX;
                    }
                }
            }
        }
    }
}
