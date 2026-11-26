// SPDX-License-Identifier: AGPL-3.0-only
// GLM-5.3 DSA selected attention on tensor cores (ATLAS_GLM53_DSA_SELECTED_MMA=1).
//
// Same ABI and semantics as atlas_glm53_dsa_selected_attention_rows_bf16, but
// NOT bit-identical: scores and the weighted latent are mma.sync bf16 products
// with fp32 accumulation in tensor-core order, and the softmax is online
// (flash) instead of two-pass. Quality-gated, not exact.
//
// Shape: per query row, 64 heads share one selected set of <= 2051 latent rows
// (512-d, K == V). One CTA = one row x 32 heads; 8 warps. Warp w owns heads
// 16*(w&1)..+15 of the CTA and (a) scores keys 8*(w>>1)..+7 of each 32-key
// tile over all 512 dims, (b) accumulates output dims 128*(w>>1)..+127.
// Probabilities are exchanged through shared memory as bf16.
//
// Precondition (guaranteed by atlas_glm53_dsa_topk_k4): valid entries of a
// row's index list are distinct. Invalid entries (<0, >= sequence length,
// > query position, >= capacity) are masked. The reference's sort+dedupe is
// therefore not repeated here.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#define GLM53_DSA_MMA_HEADS 64U
#define GLM53_DSA_MMA_LATENT 512U
#define GLM53_DSA_MMA_SELECTED 2051U
#define GLM53_DSA_MMA_MAX_QUERIES 8192U
#define GLM53_DSA_MMA_MAX_POSITIONS 1048576U
#define GLM53_DSA_MMA_THREADS 256U
#define GLM53_DSA_MMA_CTA_HEADS 32U
#define GLM53_DSA_MMA_BK 32U
#define GLM53_DSA_MMA_TILES ((GLM53_DSA_MMA_SELECTED + GLM53_DSA_MMA_BK - 1U) / GLM53_DSA_MMA_BK)
// log2(e) / sqrt(256): scores live in the exp2 domain.
#define GLM53_DSA_MMA_SCALE_LOG2 0.09016844005556021f

#define GLM53_DSA_MMA_Q_BYTES (GLM53_DSA_MMA_CTA_HEADS * GLM53_DSA_MMA_LATENT * 2U)
#define GLM53_DSA_MMA_KV_BYTES (GLM53_DSA_MMA_BK * GLM53_DSA_MMA_LATENT * 2U)
#define GLM53_DSA_MMA_P_BYTES (GLM53_DSA_MMA_CTA_HEADS * GLM53_DSA_MMA_BK * 2U)
#define GLM53_DSA_MMA_RED_BYTES (4U * GLM53_DSA_MMA_CTA_HEADS * 4U)
#define GLM53_DSA_MMA_IDX_BYTES (2U * GLM53_DSA_MMA_BK * 4U)
#define GLM53_DSA_MMA_OFF_KV GLM53_DSA_MMA_Q_BYTES
#define GLM53_DSA_MMA_OFF_P (GLM53_DSA_MMA_OFF_KV + 2U * GLM53_DSA_MMA_KV_BYTES)
#define GLM53_DSA_MMA_OFF_RED (GLM53_DSA_MMA_OFF_P + GLM53_DSA_MMA_P_BYTES)
#define GLM53_DSA_MMA_OFF_IDX (GLM53_DSA_MMA_OFF_RED + GLM53_DSA_MMA_RED_BYTES)
#define GLM53_DSA_MMA_SHARED (GLM53_DSA_MMA_OFF_IDX + GLM53_DSA_MMA_IDX_BYTES)

static_assert(GLM53_DSA_MMA_SHARED <= 101376U, "GB10 opt-in shared memory is 99 KB");

// 512-element (1024 B) rows: 16-byte chunk c of row r lives at chunk c ^ (r & 7).
__device__ __forceinline__ unsigned glm53_mma_swz512(unsigned row, unsigned col) {
    return row * GLM53_DSA_MMA_LATENT + ((((col >> 3U) ^ (row & 7U)) << 3U) | (col & 7U));
}
// 32-element (64 B) P rows: chunk c (0..3) of row r lives at c ^ ((r >> 1) & 3).
__device__ __forceinline__ unsigned glm53_mma_swz32(unsigned row, unsigned col) {
    return row * GLM53_DSA_MMA_BK + ((((col >> 3U) ^ ((row >> 1U) & 3U)) << 3U) | (col & 7U));
}

__device__ __forceinline__ void glm53_mma_cp16(void * shared, const void * global, bool fill) {
    const unsigned address = (unsigned)__cvta_generic_to_shared(shared);
    const int bytes = fill ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(global), "r"(bytes));
}

__device__ __forceinline__ void glm53_mma_ldsm_x4(unsigned (&r)[4], const void * shared) {
    const unsigned address = (unsigned)__cvta_generic_to_shared(shared);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

__device__ __forceinline__ void glm53_mma_ldsm_x4_t(unsigned (&r)[4], const void * shared) {
    const unsigned address = (unsigned)__cvta_generic_to_shared(shared);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

__device__ __forceinline__ void glm53_mma_bf16(float (&c)[4], const unsigned (&a)[4],
                                                unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Stage key tile `tile` (32 index slots) into `kv`, recording each slot's
// admitted index (or -1) in `idx`. Threads 8k..8k+7 copy key k, 128 contiguous
// bytes per instruction.
__device__ __forceinline__ void glm53_mma_stage(
        __nv_bfloat16 * kv, int * idx, const __nv_bfloat16 * cache,
        const int * row_indices, unsigned tile, unsigned sequence_length,
        unsigned query_position, unsigned kv_capacity, unsigned thread) {
    const unsigned key = thread >> 3U;
    const unsigned part = thread & 7U;
    const unsigned slot = tile * GLM53_DSA_MMA_BK + key;
    int admitted = -1;
    if (slot < GLM53_DSA_MMA_SELECTED) {
        const int raw = __ldg(row_indices + slot);
        if (raw >= 0 && (unsigned)raw < sequence_length && (unsigned)raw <= query_position &&
            (unsigned)raw < kv_capacity) {
            admitted = raw;
        }
    }
    const __nv_bfloat16 * source =
        cache + (unsigned long long)(admitted < 0 ? 0 : admitted) * GLM53_DSA_MMA_LATENT;
    #pragma unroll
    for (unsigned i = 0U; i < 8U; ++i) {
        // Lanes part 0..7 of a key read 128 contiguous bytes per instruction.
        const unsigned col = (i * 8U + part) * 8U;
        glm53_mma_cp16(kv + glm53_mma_swz512(key, col), source + col, admitted >= 0);
    }
    if (part == 0U) idx[key] = admitted;
    asm volatile("cp.async.commit_group;");
}

extern "C" __global__ void __launch_bounds__(GLM53_DSA_MMA_THREADS, 1)
atlas_glm53_dsa_selected_attention_mma_bf16(
        const __nv_bfloat16 * __restrict__ absorbed_query,
        const __nv_bfloat16 * __restrict__ latent_cache,
        const int * __restrict__ selected_indices,
        const unsigned int * __restrict__ sequence_lengths,
        const unsigned int * __restrict__ query_positions,
        const unsigned char * __restrict__ query_validity,
        __nv_bfloat16 * __restrict__ output_weighted_latent,
        unsigned int batch, unsigned int query_count,
        unsigned int kv_capacity) {
    const unsigned long long rows = (unsigned long long)batch * query_count;
    if (absorbed_query == nullptr || latent_cache == nullptr ||
        selected_indices == nullptr || sequence_lengths == nullptr ||
        query_positions == nullptr || query_validity == nullptr ||
        output_weighted_latent == nullptr || batch == 0U ||
        query_count == 0U || query_count > GLM53_DSA_MMA_MAX_QUERIES ||
        kv_capacity == 0U || kv_capacity > GLM53_DSA_MMA_MAX_POSITIONS ||
        blockDim.x != GLM53_DSA_MMA_THREADS ||
        (unsigned long long)gridDim.x != rows * 2ULL || gridDim.y != 1U ||
        gridDim.z != 1U) {
        return;
    }
    extern __shared__ __align__(128) unsigned char mma_shared[];
    __nv_bfloat16 * s_q = (__nv_bfloat16 *)mma_shared;
    __nv_bfloat16 * s_kv = (__nv_bfloat16 *)(mma_shared + GLM53_DSA_MMA_OFF_KV);
    __nv_bfloat16 * s_p = (__nv_bfloat16 *)(mma_shared + GLM53_DSA_MMA_OFF_P);
    float * s_red = (float *)(mma_shared + GLM53_DSA_MMA_OFF_RED);
    int * s_idx = (int *)(mma_shared + GLM53_DSA_MMA_OFF_IDX);

    const unsigned long long row = blockIdx.x >> 1U;
    const unsigned head0 = (blockIdx.x & 1U) * GLM53_DSA_MMA_CTA_HEADS;
    const unsigned thread = threadIdx.x;
    const unsigned lane = thread & 31U;
    const unsigned warp = thread >> 5U;
    const unsigned hg = warp & 1U;
    const unsigned slice = warp >> 1U;
    const unsigned long long out_base =
        (row * GLM53_DSA_MMA_HEADS + head0) * GLM53_DSA_MMA_LATENT;

    const unsigned long long batch_index = row / query_count;
    const unsigned sequence_length = sequence_lengths[batch_index];
    const unsigned query_position = query_positions[row];
    if (query_validity[row] == 0U || sequence_length == 0U ||
        sequence_length > kv_capacity || query_position >= sequence_length) {
        for (unsigned i = thread; i < GLM53_DSA_MMA_CTA_HEADS * GLM53_DSA_MMA_LATENT;
             i += GLM53_DSA_MMA_THREADS) {
            output_weighted_latent[out_base + i] = __float2bfloat16_rn(0.0f);
        }
        return;
    }
    const __nv_bfloat16 * cache =
        latent_cache + batch_index * (unsigned long long)kv_capacity * GLM53_DSA_MMA_LATENT;
    const int * row_indices = selected_indices + row * GLM53_DSA_MMA_SELECTED;

    // Q for this CTA's 32 heads (contiguous 32 KB) + the first key tile.
    {
        const __nv_bfloat16 * query = absorbed_query + out_base;
        #pragma unroll
        for (unsigned i = 0U; i < 8U; ++i) {
            const unsigned chunk = thread * 8U + i;
            const unsigned head = chunk >> 6U;
            const unsigned col = (chunk & 63U) * 8U;
            glm53_mma_cp16(s_q + glm53_mma_swz512(head, col),
                           query + head * GLM53_DSA_MMA_LATENT + col, true);
        }
    }
    glm53_mma_stage(s_kv, s_idx, cache, row_indices, 0U, sequence_length,
                    query_position, kv_capacity, thread);

    float acc[16][4];
    #pragma unroll
    for (unsigned n = 0U; n < 16U; ++n) {
        acc[n][0] = 0.0f; acc[n][1] = 0.0f; acc[n][2] = 0.0f; acc[n][3] = 0.0f;
    }
    float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0.0f, l_hi = 0.0f;
    const unsigned g = lane >> 2U;
    const unsigned q4 = lane & 3U;
    const unsigned head_row = hg * 16U;

    for (unsigned t = 0U; t < GLM53_DSA_MMA_TILES; ++t) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        const unsigned buf = t & 1U;
        if (t + 1U < GLM53_DSA_MMA_TILES) {
            glm53_mma_stage(s_kv + (buf ^ 1U) * (GLM53_DSA_MMA_BK * GLM53_DSA_MMA_LATENT),
                            s_idx + (buf ^ 1U) * GLM53_DSA_MMA_BK, cache, row_indices,
                            t + 1U, sequence_length, query_position, kv_capacity, thread);
        }
        const __nv_bfloat16 * kv = s_kv + buf * (GLM53_DSA_MMA_BK * GLM53_DSA_MMA_LATENT);
        const int * idx = s_idx + buf * GLM53_DSA_MMA_BK;

        // S[16 heads x 8 keys] over 512 dims.
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll 4
        for (unsigned k0 = 0U; k0 < GLM53_DSA_MMA_LATENT; k0 += 32U) {
            unsigned a0[4], a1[4], b[4];
            {
                const unsigned mi = lane >> 3U;
                const unsigned r = head_row + (lane & 7U) + ((mi & 1U) << 3U);
                const unsigned c = k0 + ((mi >> 1U) << 3U);
                glm53_mma_ldsm_x4(a0, s_q + glm53_mma_swz512(r, c));
                glm53_mma_ldsm_x4(a1, s_q + glm53_mma_swz512(r, c + 16U));
            }
            {
                const unsigned key = slice * 8U + (lane & 7U);
                const unsigned c = k0 + ((lane >> 3U) << 3U);
                glm53_mma_ldsm_x4(b, kv + glm53_mma_swz512(key, c));
            }
            glm53_mma_bf16(s, a0, b[0], b[1]);
            glm53_mma_bf16(s, a1, b[2], b[3]);
        }
        // Mask and scale; this thread holds keys slice*8 + 2*q4 + {0,1}
        // for head rows g and g + 8.
        const unsigned key0 = slice * 8U + 2U * q4;
        const bool v0 = idx[key0] >= 0;
        const bool v1 = idx[key0 + 1U] >= 0;
        s[0] = v0 ? s[0] * GLM53_DSA_MMA_SCALE_LOG2 : -INFINITY;
        s[1] = v1 ? s[1] * GLM53_DSA_MMA_SCALE_LOG2 : -INFINITY;
        s[2] = v0 ? s[2] * GLM53_DSA_MMA_SCALE_LOG2 : -INFINITY;
        s[3] = v1 ? s[3] * GLM53_DSA_MMA_SCALE_LOG2 : -INFINITY;
        float t_lo = fmaxf(s[0], s[1]);
        float t_hi = fmaxf(s[2], s[3]);
        t_lo = fmaxf(t_lo, __shfl_xor_sync(0xffffffffU, t_lo, 1U));
        t_lo = fmaxf(t_lo, __shfl_xor_sync(0xffffffffU, t_lo, 2U));
        t_hi = fmaxf(t_hi, __shfl_xor_sync(0xffffffffU, t_hi, 1U));
        t_hi = fmaxf(t_hi, __shfl_xor_sync(0xffffffffU, t_hi, 2U));
        if (q4 == 0U) {
            s_red[slice * GLM53_DSA_MMA_CTA_HEADS + head_row + g] = t_lo;
            s_red[slice * GLM53_DSA_MMA_CTA_HEADS + head_row + g + 8U] = t_hi;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned j = 0U; j < 4U; ++j) {
            t_lo = fmaxf(t_lo, s_red[j * GLM53_DSA_MMA_CTA_HEADS + head_row + g]);
            t_hi = fmaxf(t_hi, s_red[j * GLM53_DSA_MMA_CTA_HEADS + head_row + g + 8U]);
        }
        const float n_lo = fmaxf(m_lo, t_lo);
        const float n_hi = fmaxf(m_hi, t_hi);
        // A row with no admitted key so far keeps max -inf: its p are 0.
        const float base_lo = n_lo == -INFINITY ? 0.0f : n_lo;
        const float base_hi = n_hi == -INFINITY ? 0.0f : n_hi;
        const float alpha_lo = exp2f(m_lo - base_lo);
        const float alpha_hi = exp2f(m_hi - base_hi);
        m_lo = n_lo;
        m_hi = n_hi;
        const float p0 = exp2f(s[0] - base_lo), p1 = exp2f(s[1] - base_lo);
        const float p2 = exp2f(s[2] - base_hi), p3 = exp2f(s[3] - base_hi);
        const __nv_bfloat162 pk_lo = __floats2bfloat162_rn(p0, p1);
        const __nv_bfloat162 pk_hi = __floats2bfloat162_rn(p2, p3);
        // The denominator sums the same bf16-rounded p the numerator uses.
        l_lo = l_lo * alpha_lo + __low2float(pk_lo) + __high2float(pk_lo);
        l_hi = l_hi * alpha_hi + __low2float(pk_hi) + __high2float(pk_hi);
        *(__nv_bfloat162 *)(s_p + glm53_mma_swz32(head_row + g, key0)) = pk_lo;
        *(__nv_bfloat162 *)(s_p + glm53_mma_swz32(head_row + g + 8U, key0)) = pk_hi;
        #pragma unroll
        for (unsigned n = 0U; n < 16U; ++n) {
            acc[n][0] *= alpha_lo; acc[n][1] *= alpha_lo;
            acc[n][2] *= alpha_hi; acc[n][3] *= alpha_hi;
        }
        __syncthreads();
        // O[16 heads x 128 dims] += P[16 x 32] V[32 x 128].
        #pragma unroll
        for (unsigned kk = 0U; kk < GLM53_DSA_MMA_BK; kk += 16U) {
            unsigned a[4];
            {
                const unsigned mi = lane >> 3U;
                const unsigned r = head_row + (lane & 7U) + ((mi & 1U) << 3U);
                const unsigned c = kk + ((mi >> 1U) << 3U);
                glm53_mma_ldsm_x4(a, s_p + glm53_mma_swz32(r, c));
            }
            #pragma unroll
            for (unsigned n = 0U; n < 16U; n += 2U) {
                unsigned b[4];
                const unsigned key = kk + (lane & 7U) + (((lane >> 3U) & 1U) << 3U);
                const unsigned col = slice * 128U + n * 8U + ((lane >> 4U) << 3U);
                glm53_mma_ldsm_x4_t(b, kv + glm53_mma_swz512(key, col));
                glm53_mma_bf16(acc[n], a, b[0], b[1]);
                glm53_mma_bf16(acc[n + 1U], a, b[2], b[3]);
            }
        }
    }

    // Row sums: quad, then the four key slices of this head group.
    l_lo += __shfl_xor_sync(0xffffffffU, l_lo, 1U);
    l_lo += __shfl_xor_sync(0xffffffffU, l_lo, 2U);
    l_hi += __shfl_xor_sync(0xffffffffU, l_hi, 1U);
    l_hi += __shfl_xor_sync(0xffffffffU, l_hi, 2U);
    __syncthreads();
    if (q4 == 0U) {
        s_red[slice * GLM53_DSA_MMA_CTA_HEADS + head_row + g] = l_lo;
        s_red[slice * GLM53_DSA_MMA_CTA_HEADS + head_row + g + 8U] = l_hi;
    }
    __syncthreads();
    float d_lo = 0.0f, d_hi = 0.0f;
    #pragma unroll
    for (unsigned j = 0U; j < 4U; ++j) {
        d_lo += s_red[j * GLM53_DSA_MMA_CTA_HEADS + head_row + g];
        d_hi += s_red[j * GLM53_DSA_MMA_CTA_HEADS + head_row + g + 8U];
    }
    const float r_lo = d_lo > 0.0f ? 1.0f / d_lo : 0.0f;
    const float r_hi = d_hi > 0.0f ? 1.0f / d_hi : 0.0f;
    __nv_bfloat16 * out_lo = output_weighted_latent + out_base +
        (unsigned long long)(head_row + g) * GLM53_DSA_MMA_LATENT;
    __nv_bfloat16 * out_hi = out_lo + 8ULL * GLM53_DSA_MMA_LATENT;
    #pragma unroll
    for (unsigned n = 0U; n < 16U; ++n) {
        const unsigned col = slice * 128U + n * 8U + 2U * q4;
        *(__nv_bfloat162 *)(out_lo + col) =
            __floats2bfloat162_rn(acc[n][0] * r_lo, acc[n][1] * r_lo);
        *(__nv_bfloat162 *)(out_hi + col) =
            __floats2bfloat162_rn(acc[n][2] * r_hi, acc[n][3] * r_hi);
    }
}

// ---------------------------------------------------------------------------
// v2: QK split by latent dimension. Warp w (heads 16*(w&1)..+15, slice
// j = w>>1) keeps its Q quarter (dims 128j..+127) in registers, scores all 32
// keys of a tile over that quarter, and publishes the partial through shared
// memory. Every warp of a head group then sums the four partials in the same
// order (identical values), runs the online softmax for all 32 keys itself and
// feeds P straight from registers into the PV MMA for its 128 output dims.
// Two barriers per tile, no Q or P in shared memory.
#define GLM53_DSA_MMA2_PART_BYTES (2U * 4U * 16U * 32U * 4U)
#define GLM53_DSA_MMA2_OFF_PART (2U * GLM53_DSA_MMA_KV_BYTES)
#define GLM53_DSA_MMA2_OFF_IDX (GLM53_DSA_MMA2_OFF_PART + GLM53_DSA_MMA2_PART_BYTES)
#define GLM53_DSA_MMA2_SHARED (GLM53_DSA_MMA2_OFF_IDX + GLM53_DSA_MMA_IDX_BYTES)
static_assert(GLM53_DSA_MMA2_SHARED <= 101376U, "GB10 opt-in shared memory is 99 KB");

extern "C" __global__ void __launch_bounds__(GLM53_DSA_MMA_THREADS, 1)
atlas_glm53_dsa_selected_attention_mma2_bf16(
        const __nv_bfloat16 * __restrict__ absorbed_query,
        const __nv_bfloat16 * __restrict__ latent_cache,
        const int * __restrict__ selected_indices,
        const unsigned int * __restrict__ sequence_lengths,
        const unsigned int * __restrict__ query_positions,
        const unsigned char * __restrict__ query_validity,
        __nv_bfloat16 * __restrict__ output_weighted_latent,
        unsigned int batch, unsigned int query_count,
        unsigned int kv_capacity) {
    const unsigned long long rows = (unsigned long long)batch * query_count;
    if (absorbed_query == nullptr || latent_cache == nullptr ||
        selected_indices == nullptr || sequence_lengths == nullptr ||
        query_positions == nullptr || query_validity == nullptr ||
        output_weighted_latent == nullptr || batch == 0U ||
        query_count == 0U || query_count > GLM53_DSA_MMA_MAX_QUERIES ||
        kv_capacity == 0U || kv_capacity > GLM53_DSA_MMA_MAX_POSITIONS ||
        blockDim.x != GLM53_DSA_MMA_THREADS ||
        (unsigned long long)gridDim.x != rows * 2ULL || gridDim.y != 1U ||
        gridDim.z != 1U) {
        return;
    }
    extern __shared__ __align__(128) unsigned char mma2_shared[];
    __nv_bfloat16 * s_kv = (__nv_bfloat16 *)mma2_shared;
    float2 * s_part = (float2 *)(mma2_shared + GLM53_DSA_MMA2_OFF_PART);
    int * s_idx = (int *)(mma2_shared + GLM53_DSA_MMA2_OFF_IDX);

    const unsigned long long row = blockIdx.x >> 1U;
    const unsigned head0 = (blockIdx.x & 1U) * GLM53_DSA_MMA_CTA_HEADS;
    const unsigned thread = threadIdx.x;
    const unsigned lane = thread & 31U;
    const unsigned warp = thread >> 5U;
    const unsigned hg = warp & 1U;
    const unsigned slice = warp >> 1U;
    const unsigned g = lane >> 2U;
    const unsigned q4 = lane & 3U;
    const unsigned long long out_base =
        (row * GLM53_DSA_MMA_HEADS + head0) * GLM53_DSA_MMA_LATENT;

    const unsigned long long batch_index = row / query_count;
    const unsigned sequence_length = sequence_lengths[batch_index];
    const unsigned query_position = query_positions[row];
    if (query_validity[row] == 0U || sequence_length == 0U ||
        sequence_length > kv_capacity || query_position >= sequence_length) {
        for (unsigned i = thread; i < GLM53_DSA_MMA_CTA_HEADS * GLM53_DSA_MMA_LATENT;
             i += GLM53_DSA_MMA_THREADS) {
            output_weighted_latent[out_base + i] = __float2bfloat16_rn(0.0f);
        }
        return;
    }
    const __nv_bfloat16 * cache =
        latent_cache + batch_index * (unsigned long long)kv_capacity * GLM53_DSA_MMA_LATENT;
    const int * row_indices = selected_indices + row * GLM53_DSA_MMA_SELECTED;
    glm53_mma_stage(s_kv, s_idx, cache, row_indices, 0U, sequence_length,
                    query_position, kv_capacity, thread);

    // Q quarter as A fragments: 8 k-steps of 16 dims.
    unsigned qa[8][4];
    {
        const unsigned * q_lo = (const unsigned *)(absorbed_query + out_base +
            (unsigned long long)(hg * 16U + g) * GLM53_DSA_MMA_LATENT + slice * 128U + 2U * q4);
        const unsigned * q_hi = q_lo + 8U * GLM53_DSA_MMA_LATENT / 2U;
        #pragma unroll
        for (unsigned kk = 0U; kk < 8U; ++kk) {
            qa[kk][0] = __ldg(q_lo + kk * 8U);
            qa[kk][1] = __ldg(q_hi + kk * 8U);
            qa[kk][2] = __ldg(q_lo + kk * 8U + 4U);
            qa[kk][3] = __ldg(q_hi + kk * 8U + 4U);
        }
    }
    float acc[16][4];
    #pragma unroll
    for (unsigned n = 0U; n < 16U; ++n) {
        acc[n][0] = 0.0f; acc[n][1] = 0.0f; acc[n][2] = 0.0f; acc[n][3] = 0.0f;
    }
    float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0.0f, l_hi = 0.0f;
    float2 * my_part = s_part + ((hg * 4U + slice) * 8U) * 32U + lane;
    const float2 * group_part = s_part + (hg * 4U * 8U) * 32U + lane;

    for (unsigned t = 0U; t < GLM53_DSA_MMA_TILES; ++t) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        const unsigned buf = t & 1U;
#ifdef GLM53_DSA_MMA2_EXPERIMENT_NOLOAD
        if (t == 0U) {
#else
        if (t + 1U < GLM53_DSA_MMA_TILES) {
#endif
            glm53_mma_stage(s_kv + (buf ^ 1U) * (GLM53_DSA_MMA_BK * GLM53_DSA_MMA_LATENT),
                            s_idx + (buf ^ 1U) * GLM53_DSA_MMA_BK, cache, row_indices,
                            t + 1U, sequence_length, query_position, kv_capacity, thread);
        }
        const __nv_bfloat16 * kv = s_kv + buf * (GLM53_DSA_MMA_BK * GLM53_DSA_MMA_LATENT);
        const int * idx = s_idx + buf * GLM53_DSA_MMA_BK;

#ifdef GLM53_DSA_MMA2_EXPERIMENT_NOCOMPUTE
        if (idx[lane] == -7) acc[0][0] += 1.0f;
        continue;
#endif
        // Partial S[16 heads x 32 keys] over this warp's 128 dims.
        float sp[4][4];
        #pragma unroll
        for (unsigned nt = 0U; nt < 4U; ++nt) {
            sp[nt][0] = 0.0f; sp[nt][1] = 0.0f; sp[nt][2] = 0.0f; sp[nt][3] = 0.0f;
            #pragma unroll
            for (unsigned kk = 0U; kk < 8U; kk += 2U) {
                unsigned b[4];
                const unsigned key = nt * 8U + (lane & 7U);
                const unsigned col = slice * 128U + kk * 16U + ((lane >> 3U) << 3U);
                glm53_mma_ldsm_x4(b, kv + glm53_mma_swz512(key, col));
                glm53_mma_bf16(sp[nt], qa[kk], b[0], b[1]);
                glm53_mma_bf16(sp[nt], qa[kk + 1U], b[2], b[3]);
            }
        }
        #pragma unroll
        for (unsigned nt = 0U; nt < 4U; ++nt) {
            my_part[(nt * 2U) * 32U] = make_float2(sp[nt][0], sp[nt][1]);
            my_part[(nt * 2U + 1U) * 32U] = make_float2(sp[nt][2], sp[nt][3]);
        }
        __syncthreads();
        float s[4][4];
        #pragma unroll
        for (unsigned nt = 0U; nt < 4U; ++nt) {
            float2 lo = group_part[(nt * 2U) * 32U];
            float2 hi = group_part[(nt * 2U + 1U) * 32U];
            #pragma unroll
            for (unsigned j = 1U; j < 4U; ++j) {
                const float2 a = group_part[(j * 8U + nt * 2U) * 32U];
                const float2 c = group_part[(j * 8U + nt * 2U + 1U) * 32U];
                lo.x += a.x; lo.y += a.y; hi.x += c.x; hi.y += c.y;
            }
            const unsigned key0 = nt * 8U + 2U * q4;
            const bool v0 = idx[key0] >= 0;
            const bool v1 = idx[key0 + 1U] >= 0;
            s[nt][0] = v0 ? lo.x * GLM53_DSA_MMA_SCALE_LOG2 : -INFINITY;
            s[nt][1] = v1 ? lo.y * GLM53_DSA_MMA_SCALE_LOG2 : -INFINITY;
            s[nt][2] = v0 ? hi.x * GLM53_DSA_MMA_SCALE_LOG2 : -INFINITY;
            s[nt][3] = v1 ? hi.y * GLM53_DSA_MMA_SCALE_LOG2 : -INFINITY;
        }
        float t_lo = -INFINITY, t_hi = -INFINITY;
        #pragma unroll
        for (unsigned nt = 0U; nt < 4U; ++nt) {
            t_lo = fmaxf(t_lo, fmaxf(s[nt][0], s[nt][1]));
            t_hi = fmaxf(t_hi, fmaxf(s[nt][2], s[nt][3]));
        }
        t_lo = fmaxf(t_lo, __shfl_xor_sync(0xffffffffU, t_lo, 1U));
        t_lo = fmaxf(t_lo, __shfl_xor_sync(0xffffffffU, t_lo, 2U));
        t_hi = fmaxf(t_hi, __shfl_xor_sync(0xffffffffU, t_hi, 1U));
        t_hi = fmaxf(t_hi, __shfl_xor_sync(0xffffffffU, t_hi, 2U));
        const float n_lo = fmaxf(m_lo, t_lo);
        const float n_hi = fmaxf(m_hi, t_hi);
        const float base_lo = n_lo == -INFINITY ? 0.0f : n_lo;
        const float base_hi = n_hi == -INFINITY ? 0.0f : n_hi;
        const float alpha_lo = exp2f(m_lo - base_lo);
        const float alpha_hi = exp2f(m_hi - base_hi);
        m_lo = n_lo;
        m_hi = n_hi;
        unsigned pa[4][2];
        float sum_lo = 0.0f, sum_hi = 0.0f;
        #pragma unroll
        for (unsigned nt = 0U; nt < 4U; ++nt) {
            const __nv_bfloat162 lo = __floats2bfloat162_rn(exp2f(s[nt][0] - base_lo),
                                                            exp2f(s[nt][1] - base_lo));
            const __nv_bfloat162 hi = __floats2bfloat162_rn(exp2f(s[nt][2] - base_hi),
                                                            exp2f(s[nt][3] - base_hi));
            sum_lo += __low2float(lo) + __high2float(lo);
            sum_hi += __low2float(hi) + __high2float(hi);
            pa[nt][0] = *(const unsigned *)&lo;
            pa[nt][1] = *(const unsigned *)&hi;
        }
        sum_lo += __shfl_xor_sync(0xffffffffU, sum_lo, 1U);
        sum_lo += __shfl_xor_sync(0xffffffffU, sum_lo, 2U);
        sum_hi += __shfl_xor_sync(0xffffffffU, sum_hi, 1U);
        sum_hi += __shfl_xor_sync(0xffffffffU, sum_hi, 2U);
        l_lo = l_lo * alpha_lo + sum_lo;
        l_hi = l_hi * alpha_hi + sum_hi;
        #pragma unroll
        for (unsigned n = 0U; n < 16U; ++n) {
            acc[n][0] *= alpha_lo; acc[n][1] *= alpha_lo;
            acc[n][2] *= alpha_hi; acc[n][3] *= alpha_hi;
        }
        // O[16 x 128] += P[16 x 32] V[32 x 128]; P from registers.
        #pragma unroll
        for (unsigned ks = 0U; ks < 2U; ++ks) {
            const unsigned a[4] = {pa[2U * ks][0], pa[2U * ks][1],
                                   pa[2U * ks + 1U][0], pa[2U * ks + 1U][1]};
            #pragma unroll
            for (unsigned n = 0U; n < 16U; n += 2U) {
                unsigned b[4];
                const unsigned key = ks * 16U + (lane & 7U) + (((lane >> 3U) & 1U) << 3U);
                const unsigned col = slice * 128U + n * 8U + ((lane >> 4U) << 3U);
                glm53_mma_ldsm_x4_t(b, kv + glm53_mma_swz512(key, col));
                glm53_mma_bf16(acc[n], a, b[0], b[1]);
                glm53_mma_bf16(acc[n + 1U], a, b[2], b[3]);
            }
        }
    }
    const float r_lo = l_lo > 0.0f ? 1.0f / l_lo : 0.0f;
    const float r_hi = l_hi > 0.0f ? 1.0f / l_hi : 0.0f;
    __nv_bfloat16 * out_lo = output_weighted_latent + out_base +
        (unsigned long long)(hg * 16U + g) * GLM53_DSA_MMA_LATENT;
    __nv_bfloat16 * out_hi = out_lo + 8ULL * GLM53_DSA_MMA_LATENT;
    #pragma unroll
    for (unsigned n = 0U; n < 16U; ++n) {
        const unsigned col = slice * 128U + n * 8U + 2U * q4;
        *(__nv_bfloat162 *)(out_lo + col) =
            __floats2bfloat162_rn(acc[n][0] * r_lo, acc[n][1] * r_lo);
        *(__nv_bfloat162 *)(out_hi + col) =
            __floats2bfloat162_rn(acc[n][2] * r_hi, acc[n][3] * r_hi);
    }
}

// ---------------------------------------------------------------------------
// v3: v2 with a 3-slot 32-key ring (two tiles of gathers in flight). The
// partial scores of tile t live in the slot that held tile t-1 (consumed by
// the previous PV); tile t+2 is issued into that slot once every warp has read
// the partials. Three barriers per tile.
#define GLM53_DSA_MMA3_OFF_IDX (3U * GLM53_DSA_MMA_KV_BYTES)
#define GLM53_DSA_MMA3_SHARED (GLM53_DSA_MMA3_OFF_IDX + 3U * GLM53_DSA_MMA_BK * 4U)
static_assert(GLM53_DSA_MMA3_SHARED <= 101376U, "GB10 opt-in shared memory is 99 KB");
static_assert(GLM53_DSA_MMA2_PART_BYTES <= GLM53_DSA_MMA_KV_BYTES, "partials fit one ring slot");

extern "C" __global__ void __launch_bounds__(GLM53_DSA_MMA_THREADS, 1)
atlas_glm53_dsa_selected_attention_mma3_bf16(
        const __nv_bfloat16 * __restrict__ absorbed_query,
        const __nv_bfloat16 * __restrict__ latent_cache,
        const int * __restrict__ selected_indices,
        const unsigned int * __restrict__ sequence_lengths,
        const unsigned int * __restrict__ query_positions,
        const unsigned char * __restrict__ query_validity,
        __nv_bfloat16 * __restrict__ output_weighted_latent,
        unsigned int batch, unsigned int query_count,
        unsigned int kv_capacity) {
    const unsigned long long rows = (unsigned long long)batch * query_count;
    if (absorbed_query == nullptr || latent_cache == nullptr ||
        selected_indices == nullptr || sequence_lengths == nullptr ||
        query_positions == nullptr || query_validity == nullptr ||
        output_weighted_latent == nullptr || batch == 0U ||
        query_count == 0U || query_count > GLM53_DSA_MMA_MAX_QUERIES ||
        kv_capacity == 0U || kv_capacity > GLM53_DSA_MMA_MAX_POSITIONS ||
        blockDim.x != GLM53_DSA_MMA_THREADS ||
        (unsigned long long)gridDim.x != rows * 2ULL || gridDim.y != 1U ||
        gridDim.z != 1U) {
        return;
    }
    extern __shared__ __align__(128) unsigned char mma3_shared[];
    __nv_bfloat16 * s_kv = (__nv_bfloat16 *)mma3_shared;
    int * s_idx = (int *)(mma3_shared + GLM53_DSA_MMA3_OFF_IDX);

    const unsigned long long row = blockIdx.x >> 1U;
    const unsigned head0 = (blockIdx.x & 1U) * GLM53_DSA_MMA_CTA_HEADS;
    const unsigned thread = threadIdx.x;
    const unsigned lane = thread & 31U;
    const unsigned warp = thread >> 5U;
    const unsigned hg = warp & 1U;
    const unsigned slice = warp >> 1U;
    const unsigned g = lane >> 2U;
    const unsigned q4 = lane & 3U;
    const unsigned long long out_base =
        (row * GLM53_DSA_MMA_HEADS + head0) * GLM53_DSA_MMA_LATENT;

    const unsigned long long batch_index = row / query_count;
    const unsigned sequence_length = sequence_lengths[batch_index];
    const unsigned query_position = query_positions[row];
    if (query_validity[row] == 0U || sequence_length == 0U ||
        sequence_length > kv_capacity || query_position >= sequence_length) {
        for (unsigned i = thread; i < GLM53_DSA_MMA_CTA_HEADS * GLM53_DSA_MMA_LATENT;
             i += GLM53_DSA_MMA_THREADS) {
            output_weighted_latent[out_base + i] = __float2bfloat16_rn(0.0f);
        }
        return;
    }
    const __nv_bfloat16 * cache =
        latent_cache + batch_index * (unsigned long long)kv_capacity * GLM53_DSA_MMA_LATENT;
    const int * row_indices = selected_indices + row * GLM53_DSA_MMA_SELECTED;
    glm53_mma_stage(s_kv, s_idx, cache, row_indices, 0U, sequence_length,
                    query_position, kv_capacity, thread);
    glm53_mma_stage(s_kv + GLM53_DSA_MMA_BK * GLM53_DSA_MMA_LATENT, s_idx + GLM53_DSA_MMA_BK,
                    cache, row_indices, 1U, sequence_length, query_position, kv_capacity,
                    thread);

    // Q quarter as A fragments: 8 k-steps of 16 dims.
    unsigned qa[8][4];
    {
        const unsigned * q_lo = (const unsigned *)(absorbed_query + out_base +
            (unsigned long long)(hg * 16U + g) * GLM53_DSA_MMA_LATENT + slice * 128U + 2U * q4);
        const unsigned * q_hi = q_lo + 8U * GLM53_DSA_MMA_LATENT / 2U;
        #pragma unroll
        for (unsigned kk = 0U; kk < 8U; ++kk) {
            qa[kk][0] = __ldg(q_lo + kk * 8U);
            qa[kk][1] = __ldg(q_hi + kk * 8U);
            qa[kk][2] = __ldg(q_lo + kk * 8U + 4U);
            qa[kk][3] = __ldg(q_hi + kk * 8U + 4U);
        }
    }
    float acc[16][4];
    #pragma unroll
    for (unsigned n = 0U; n < 16U; ++n) {
        acc[n][0] = 0.0f; acc[n][1] = 0.0f; acc[n][2] = 0.0f; acc[n][3] = 0.0f;
    }
    float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0.0f, l_hi = 0.0f;
    for (unsigned t = 0U; t < GLM53_DSA_MMA_TILES; ++t) {
        // Tile t+1 may stay in flight; tile t must have landed.
        if (t + 1U < GLM53_DSA_MMA_TILES) {
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        const unsigned buf = t % 3U;
        const unsigned spare = (t + 2U) % 3U;  // held tile t-1: consumed, reusable
        const __nv_bfloat16 * kv = s_kv + buf * (GLM53_DSA_MMA_BK * GLM53_DSA_MMA_LATENT);
        const int * idx = s_idx + buf * GLM53_DSA_MMA_BK;
        float2 * s_part = (float2 *)(s_kv + spare * (GLM53_DSA_MMA_BK * GLM53_DSA_MMA_LATENT));
        float2 * my_part = s_part + ((hg * 4U + slice) * 8U) * 32U + lane;
        const float2 * group_part = s_part + (hg * 4U * 8U) * 32U + lane;

        // Partial S[16 heads x 32 keys] over this warp's 128 dims.
        float sp[4][4];
        #pragma unroll
        for (unsigned nt = 0U; nt < 4U; ++nt) {
            sp[nt][0] = 0.0f; sp[nt][1] = 0.0f; sp[nt][2] = 0.0f; sp[nt][3] = 0.0f;
            #pragma unroll
            for (unsigned kk = 0U; kk < 8U; kk += 2U) {
                unsigned b[4];
                const unsigned key = nt * 8U + (lane & 7U);
                const unsigned col = slice * 128U + kk * 16U + ((lane >> 3U) << 3U);
                glm53_mma_ldsm_x4(b, kv + glm53_mma_swz512(key, col));
                glm53_mma_bf16(sp[nt], qa[kk], b[0], b[1]);
                glm53_mma_bf16(sp[nt], qa[kk + 1U], b[2], b[3]);
            }
        }
        #pragma unroll
        for (unsigned nt = 0U; nt < 4U; ++nt) {
            my_part[(nt * 2U) * 32U] = make_float2(sp[nt][0], sp[nt][1]);
            my_part[(nt * 2U + 1U) * 32U] = make_float2(sp[nt][2], sp[nt][3]);
        }
        __syncthreads();
        float s[4][4];
        #pragma unroll
        for (unsigned nt = 0U; nt < 4U; ++nt) {
            float2 lo = group_part[(nt * 2U) * 32U];
            float2 hi = group_part[(nt * 2U + 1U) * 32U];
            #pragma unroll
            for (unsigned j = 1U; j < 4U; ++j) {
                const float2 a = group_part[(j * 8U + nt * 2U) * 32U];
                const float2 c = group_part[(j * 8U + nt * 2U + 1U) * 32U];
                lo.x += a.x; lo.y += a.y; hi.x += c.x; hi.y += c.y;
            }
            const unsigned key0 = nt * 8U + 2U * q4;
            const bool v0 = idx[key0] >= 0;
            const bool v1 = idx[key0 + 1U] >= 0;
            s[nt][0] = v0 ? lo.x * GLM53_DSA_MMA_SCALE_LOG2 : -INFINITY;
            s[nt][1] = v1 ? lo.y * GLM53_DSA_MMA_SCALE_LOG2 : -INFINITY;
            s[nt][2] = v0 ? hi.x * GLM53_DSA_MMA_SCALE_LOG2 : -INFINITY;
            s[nt][3] = v1 ? hi.y * GLM53_DSA_MMA_SCALE_LOG2 : -INFINITY;
        }
        float t_lo = -INFINITY, t_hi = -INFINITY;
        #pragma unroll
        for (unsigned nt = 0U; nt < 4U; ++nt) {
            t_lo = fmaxf(t_lo, fmaxf(s[nt][0], s[nt][1]));
            t_hi = fmaxf(t_hi, fmaxf(s[nt][2], s[nt][3]));
        }
        t_lo = fmaxf(t_lo, __shfl_xor_sync(0xffffffffU, t_lo, 1U));
        t_lo = fmaxf(t_lo, __shfl_xor_sync(0xffffffffU, t_lo, 2U));
        t_hi = fmaxf(t_hi, __shfl_xor_sync(0xffffffffU, t_hi, 1U));
        t_hi = fmaxf(t_hi, __shfl_xor_sync(0xffffffffU, t_hi, 2U));
        const float n_lo = fmaxf(m_lo, t_lo);
        const float n_hi = fmaxf(m_hi, t_hi);
        const float base_lo = n_lo == -INFINITY ? 0.0f : n_lo;
        const float base_hi = n_hi == -INFINITY ? 0.0f : n_hi;
        const float alpha_lo = exp2f(m_lo - base_lo);
        const float alpha_hi = exp2f(m_hi - base_hi);
        m_lo = n_lo;
        m_hi = n_hi;
        unsigned pa[4][2];
        float sum_lo = 0.0f, sum_hi = 0.0f;
        #pragma unroll
        for (unsigned nt = 0U; nt < 4U; ++nt) {
            const __nv_bfloat162 lo = __floats2bfloat162_rn(exp2f(s[nt][0] - base_lo),
                                                            exp2f(s[nt][1] - base_lo));
            const __nv_bfloat162 hi = __floats2bfloat162_rn(exp2f(s[nt][2] - base_hi),
                                                            exp2f(s[nt][3] - base_hi));
            sum_lo += __low2float(lo) + __high2float(lo);
            sum_hi += __low2float(hi) + __high2float(hi);
            pa[nt][0] = *(const unsigned *)&lo;
            pa[nt][1] = *(const unsigned *)&hi;
        }
        // Every warp has read the partials: the spare slot can take tile t+2.
        __syncthreads();
        if (t + 2U < GLM53_DSA_MMA_TILES) {
            glm53_mma_stage(s_kv + spare * (GLM53_DSA_MMA_BK * GLM53_DSA_MMA_LATENT),
                            s_idx + spare * GLM53_DSA_MMA_BK, cache, row_indices, t + 2U,
                            sequence_length, query_position, kv_capacity, thread);
        } else {
            asm volatile("cp.async.commit_group;");
        }
        sum_lo += __shfl_xor_sync(0xffffffffU, sum_lo, 1U);
        sum_lo += __shfl_xor_sync(0xffffffffU, sum_lo, 2U);
        sum_hi += __shfl_xor_sync(0xffffffffU, sum_hi, 1U);
        sum_hi += __shfl_xor_sync(0xffffffffU, sum_hi, 2U);
        l_lo = l_lo * alpha_lo + sum_lo;
        l_hi = l_hi * alpha_hi + sum_hi;
        #pragma unroll
        for (unsigned n = 0U; n < 16U; ++n) {
            acc[n][0] *= alpha_lo; acc[n][1] *= alpha_lo;
            acc[n][2] *= alpha_hi; acc[n][3] *= alpha_hi;
        }
        // O[16 x 128] += P[16 x 32] V[32 x 128]; P from registers.
        #pragma unroll
        for (unsigned ks = 0U; ks < 2U; ++ks) {
            const unsigned a[4] = {pa[2U * ks][0], pa[2U * ks][1],
                                   pa[2U * ks + 1U][0], pa[2U * ks + 1U][1]};
            #pragma unroll
            for (unsigned n = 0U; n < 16U; n += 2U) {
                unsigned b[4];
                const unsigned key = ks * 16U + (lane & 7U) + (((lane >> 3U) & 1U) << 3U);
                const unsigned col = slice * 128U + n * 8U + ((lane >> 4U) << 3U);
                glm53_mma_ldsm_x4_t(b, kv + glm53_mma_swz512(key, col));
                glm53_mma_bf16(acc[n], a, b[0], b[1]);
                glm53_mma_bf16(acc[n + 1U], a, b[2], b[3]);
            }
        }
    }
    const float r_lo = l_lo > 0.0f ? 1.0f / l_lo : 0.0f;
    const float r_hi = l_hi > 0.0f ? 1.0f / l_hi : 0.0f;
    __nv_bfloat16 * out_lo = output_weighted_latent + out_base +
        (unsigned long long)(hg * 16U + g) * GLM53_DSA_MMA_LATENT;
    __nv_bfloat16 * out_hi = out_lo + 8ULL * GLM53_DSA_MMA_LATENT;
    #pragma unroll
    for (unsigned n = 0U; n < 16U; ++n) {
        const unsigned col = slice * 128U + n * 8U + 2U * q4;
        *(__nv_bfloat162 *)(out_lo + col) =
            __floats2bfloat162_rn(acc[n][0] * r_lo, acc[n][1] * r_lo);
        *(__nv_bfloat162 *)(out_hi + col) =
            __floats2bfloat162_rn(acc[n][2] * r_hi, acc[n][3] * r_hi);
    }
}
