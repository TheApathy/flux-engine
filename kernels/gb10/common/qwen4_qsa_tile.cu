// SPDX-License-Identifier: AGPL-3.0-only
//
// Tile-batched QSA prefill pooling: pools each complete 4-token group straight
// from the projected raw keys (same values the ring copy would hold), in the
// same row order and with the same FP32 sum * 0.25 -> BF16 rounding as
// qwen4_qsa_pool_prefill. first_positions[g] = positions[4g+3] - 3.
// Grid: (num_groups), Block: (INDEX_DIM).
#include <cuda_bf16.h>
#ifndef QSA_TILE_INDEX_DIM
#define QSA_TILE_INDEX_DIM 128
#endif
#define QSA_TILE_RATIO 4

extern "C" __global__ void qwen4_qsa_pool_prefill_from_raw(
    const __nv_bfloat16* __restrict__ raw_keys,
    __nv_bfloat16* __restrict__ pooled_keys,
    unsigned int* __restrict__ first_positions,
    const unsigned int* __restrict__ positions,
    unsigned int num_groups) {
    const unsigned int group = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (group >= num_groups || d >= QSA_TILE_INDEX_DIM) return;
    float sum = 0.0f;
#pragma unroll
    for (int row = 0; row < QSA_TILE_RATIO; ++row) {
        sum += __bfloat162float(
            raw_keys[((unsigned long long)group * QSA_TILE_RATIO + row) * QSA_TILE_INDEX_DIM + d]);
    }
    pooled_keys[(unsigned long long)group * QSA_TILE_INDEX_DIM + d] = __float2bfloat16_rn(sum * 0.25f);
    if (d == 0) first_positions[group] = positions[group * QSA_TILE_RATIO + QSA_TILE_RATIO - 1] - (QSA_TILE_RATIO - 1);
}

namespace {

__device__ __forceinline__ unsigned int ld_global_u32_evict_first_tile(const unsigned int* p) {
    unsigned int v;
    asm volatile(
        "{\n\t"
        ".reg .b64 policy;\n\t"
        "createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n\t"
        "ld.global.L2::cache_hint.u32 %0, [%1], policy;\n\t"
        "}\n"
        : "=r"(v)
        : "l"(p));
    return v;
}

__device__ __forceinline__ void st_global_u32_evict_last_tile(unsigned int* p, unsigned int v) {
    asm volatile(
        "{\n\t"
        ".reg .b64 policy;\n\t"
        "createpolicy.fractional.L2::evict_last.b64 policy, 1.0;\n\t"
        "st.global.L2::cache_hint.u32 [%0], %1, policy;\n\t"
        "}\n"
        :
        : "l"(p), "r"(v)
        : "memory");
}

__device__ __forceinline__ void unpack_bf16x2_tile(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFFu)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int pack_bf16x2_tile(float v0, float v1) {
    const unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    const unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

__device__ __forceinline__ float warp_reduce_sum_tile(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

}  // namespace

// Fused 5-stage QSA prefill key transaction (grid = ceil(num_groups/4), block = 128, 4 warps = 4 groups/CTA):
//   1. qwen4_qsa_pool_prefill_from_raw (average 4 raw keys -> BF16)
//   2. rms_norm (d=128, zero-centered weight 1+w -> BF16, identical reduction order to rms_norm.cu)
//   3. qwen4_rope_table / rope_forward (rotary_dim=64, first_pos = pos[4g+3]-3)
//   4. qwen4_qsa_store_prefill_compressed (store to physical compressed_cache)
//   5. qwen4_qsa_stage_prefill_raw (last group stages its 4 raw keys to raw_ring)
// 100% bit-identical (0 ULP) to the 5-kernel chain.
extern "C" __global__ void __launch_bounds__(128) qwen4_qsa_pool_norm_rope_store_fused(
    const __nv_bfloat16* __restrict__ raw_keys,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    __nv_bfloat16* __restrict__ compressed_cache,
    __nv_bfloat16* __restrict__ raw_ring,
    const unsigned int* __restrict__ positions,
    const long long* __restrict__ slots,
    unsigned int num_groups,
    unsigned int block_size,
    float eps,
    float theta) {
    __shared__ __nv_bfloat16 s_rot_all[4][64];
    __shared__ float s_freq[32];
    __shared__ unsigned int s_w[64];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    if (tid < 64u) {
        s_w[tid] = reinterpret_cast<const unsigned int*>(k_norm_weight)[tid];
    }
    if (warp == 0u) {
        const double freq_exp_d = (double)(2u * lane) / 64.0;
        s_freq[lane] = (float)(1.0 / pow((double)theta, freq_exp_d));
    }
    __syncthreads();

    const unsigned int group = blockIdx.x * 4u + warp;
    if (group >= num_groups) return;

    __nv_bfloat16* s_rot = s_rot_all[warp];
    const unsigned int* raw32 = reinterpret_cast<const unsigned int*>(raw_keys);
    const bool is_last_group = (group + 1u == num_groups);

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
#pragma unroll
    for (int row = 0; row < QSA_TILE_RATIO; ++row) {
        const unsigned int tok = group * QSA_TILE_RATIO + (unsigned int)row;
        const unsigned int p_lo = ld_global_u32_evict_first_tile(raw32 + (unsigned long long)tok * 64ull + lane);
        const unsigned int p_hi = ld_global_u32_evict_first_tile(raw32 + (unsigned long long)tok * 64ull + 32ull + lane);
        float r0, r1, r2, r3;
        unpack_bf16x2_tile(p_lo, r0, r1);
        unpack_bf16x2_tile(p_hi, r2, r3);
        sum0 += r0;
        sum1 += r1;
        sum2 += r2;
        sum3 += r3;
        if (is_last_group && block_size >= QSA_TILE_RATIO && (block_size % QSA_TILE_RATIO == 0u)) {
            const long long s = slots[tok];
            if (s >= 0) {
                const unsigned long long ps = static_cast<unsigned long long>(s);
                const unsigned long long pb = ps / block_size;
                const unsigned int rr = static_cast<unsigned int>(ps % block_size) % QSA_TILE_RATIO;
                unsigned int* ring32 = reinterpret_cast<unsigned int*>(
                    raw_ring + (pb * QSA_TILE_RATIO + rr) * QSA_TILE_INDEX_DIM);
                ring32[lane] = p_lo;
                ring32[32u + lane] = p_hi;
            }
        }
    }

    const float xv0 = __bfloat162float(__float2bfloat16_rn(sum0 * 0.25f));
    const float xv1 = __bfloat162float(__float2bfloat16_rn(sum1 * 0.25f));
    const float xv2 = __bfloat162float(__float2bfloat16_rn(sum2 * 0.25f));
    const float xv3 = __bfloat162float(__float2bfloat16_rn(sum3 * 0.25f));

    const float s0 = warp_reduce_sum_tile(xv0 * xv0 + xv1 * xv1);
    const float s1 = warp_reduce_sum_tile(xv2 * xv2 + xv3 * xv3);
    const float rms = rsqrtf((s0 + s1) / (float)QSA_TILE_INDEX_DIM + eps);

    float wv0, wv1, wv2, wv3;
    unpack_bf16x2_tile(s_w[lane], wv0, wv1);
    unpack_bf16x2_tile(s_w[32u + lane], wv2, wv3);

    s_rot[2u * lane] = __float2bfloat16(xv0 * rms * (1.0f + wv0));
    s_rot[2u * lane + 1u] = __float2bfloat16(xv1 * rms * (1.0f + wv1));
    const unsigned int out_hi = pack_bf16x2_tile(xv2 * rms * (1.0f + wv2), xv3 * rms * (1.0f + wv3));
    __syncwarp();

    const unsigned int first_pos =
        positions[group * QSA_TILE_RATIO + QSA_TILE_RATIO - 1u] - (QSA_TILE_RATIO - 1u);
    const float freq_val = s_freq[lane];
    const float angle = (float)first_pos * freq_val;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const float x0 = __bfloat162float(s_rot[lane]);
    const float x1 = __bfloat162float(s_rot[lane + 32u]);
    s_rot[lane] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    s_rot[lane + 32u] = __float2bfloat16(x1 * cos_val + x0 * sin_val);
    __syncwarp();

    const unsigned int out_lo = reinterpret_cast<const unsigned int*>(s_rot)[lane];
    if (block_size >= QSA_TILE_RATIO) {
        const unsigned int token = (QSA_TILE_RATIO - 1u) + group * QSA_TILE_RATIO;
        const long long slot = slots[token];
        if (slot >= 0) {
            const unsigned long long physical_block = (unsigned long long)slot / block_size;
            const unsigned int group_in_block =
                ((unsigned int)slot % block_size) / QSA_TILE_RATIO;
            const unsigned long long dst =
                (physical_block * (block_size / QSA_TILE_RATIO) + group_in_block) *
                QSA_TILE_INDEX_DIM;
            unsigned int* dst32 = reinterpret_cast<unsigned int*>(compressed_cache + dst);
            st_global_u32_evict_last_tile(dst32 + lane, out_lo);
            st_global_u32_evict_last_tile(dst32 + 32u + lane, out_hi);
        }
    }
}

// Fused rms_norm (d=128, zero-centered weight 1+w) + qwen4_rope_table / rope_forward
// (rotary_dim=64, NeoX pairs (d, d+32)) in place on iq = [rows, 4, 128] bf16.
// Grid: (rows, 1, 1), Block: (128, 1, 1) — 4 warps = 4 heads per row, 1 warp per head.
// 100% bit-identical (0 ULP) to running rms_norm followed by qwen4_rope_table.
extern "C" __global__ void __launch_bounds__(128) qwen4_qsa_iq_norm_rope_fused(
    __nv_bfloat16* __restrict__ iq,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ freq,
    unsigned int rows,
    float eps) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;

    __shared__ __nv_bfloat16 s_rot[4][64];

    const unsigned long long row_head = (unsigned long long)row * 4ull + warp;
    unsigned int* x32 = reinterpret_cast<unsigned int*>(iq + row_head * QSA_TILE_INDEX_DIM);
    const unsigned int* w32 = reinterpret_cast<const unsigned int*>(q_norm_weight);

    float xv0, xv1, xv2, xv3;
    unpack_bf16x2_tile(x32[lane], xv0, xv1);
    unpack_bf16x2_tile(x32[32u + lane], xv2, xv3);

    const float s0 = warp_reduce_sum_tile(xv0 * xv0 + xv1 * xv1);
    const float s1 = warp_reduce_sum_tile(xv2 * xv2 + xv3 * xv3);
    const float rms = rsqrtf((s0 + s1) / (float)QSA_TILE_INDEX_DIM + eps);

    float wv0, wv1, wv2, wv3;
    unpack_bf16x2_tile(w32[lane], wv0, wv1);
    unpack_bf16x2_tile(w32[32u + lane], wv2, wv3);

    st_global_u32_evict_last_tile(
        x32 + 32u + lane,
        pack_bf16x2_tile(xv2 * rms * (1.0f + wv2), xv3 * rms * (1.0f + wv3)));
    s_rot[warp][2u * lane] = __float2bfloat16(xv0 * rms * (1.0f + wv0));
    s_rot[warp][2u * lane + 1u] = __float2bfloat16(xv1 * rms * (1.0f + wv1));
    __syncwarp();

    const unsigned int abs_pos = positions[row];
    const float angle = (float)abs_pos * freq[lane];
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const float x0 = __bfloat162float(s_rot[warp][lane]);
    const float x1 = __bfloat162float(s_rot[warp][lane + 32u]);
    s_rot[warp][lane] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    s_rot[warp][lane + 32u] = __float2bfloat16(x1 * cos_val + x0 * sin_val);
    __syncwarp();

    st_global_u32_evict_last_tile(
        x32 + lane,
        reinterpret_cast<const unsigned int*>(s_rot[warp])[lane]);
}
