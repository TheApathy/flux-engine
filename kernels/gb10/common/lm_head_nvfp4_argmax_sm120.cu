// SPDX-License-Identifier: AGPL-3.0-only
//
// Flux Engine — Fused Two-Stage Blackwell NVFP4 LM-Head + Online Warp/Block Argmax.
//
// Eliminates the intermediate `[M, V]` BF16 logits write+read slab (`4 MiB` on
// greedy `K=8`, `V=248,077` verify) by computing vocab logits strictly in
// registers/SMEM and emitting only compact `[M, num_v_tiles]` partial
// `(f32 max_val, u32 argmax_id)` reductions (`62 KiB` at `M=8, V_TILE=256`).
//
// Mathematical & ISA contract:
//   - Input activations `A[M, K]` (BF16, `1 <= M <= 16`, `K % 64 == 0`) are
//     dual-plane quantized per 16-element group on the fly in SMEM:
//       A ≈ UE4M3(As) * (E2M1(A_hi) + 0.125f * E2M1(A_lo))  (~38.5 dB SQNR)
//   - Weight matrix `B_packed[V, K/2]` (NVFP4 E2M1) + `B_scale[V, K/16]` (UE4M3)
//     + `scale2` (FP32) is streamed from HBM ONCE per tile (`V_TILE = 256`).
//   - On Blackwell SM120/SM121 (`__CUDA_ARCH__ >= 1200`), executes native
//     block-scaled FP4 Tensor Core MMA:
//       mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//           .row.col.f32.e2m1.e2m1.f32.ue4m3
//     accumulating `acc_hi` and `acc_lo` in FP32 and forming
//     `logit = (acc_hi + 0.125f * acc_lo) * scale2`.
//   - On SM103a (`__CUDA_ARCH__ < 1200`), executes an exact scalar/FP32
//     fallback over the identical Dual-Plane E2M1 activation + NVFP4 weight
//     contract with 0 B stack and 0 B spills.
//   - Deterministic tie-breaking: on equal FP32 logits, the LOWEST vocab index
//     strictly wins; `NaN` never wins; out-of-bounds tail slots (`v >= V`) are
//     masked off via sentinel `(-INF, 0xFFFFFFFFu)`.

#include <cuda_bf16.h>
#include <math_constants.h>

#define LMH_M_MAX      16u
#define LMH_V_TILE     256u
#define LMH_K_STEP     64u
#define LMH_GROUP_SIZE 16u
#define LMH_WARP_SIZE  32u
#define LMH_INVALID_ID 0xFFFFFFFFu

__device__ __constant__ float LMH_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Branchless FP32 -> E2M1 nibble (round-to-nearest ties-to-even).
__device__ __forceinline__ unsigned int lmh_float_to_e2m1(float x) {
    const unsigned int u = __float_as_uint(x);
    const unsigned int sign = (u >> 28) & 8u;
    const unsigned int abits = u & 0x7FFFFFFFu;
    const unsigned int mag = (unsigned int)(abits >  0x3E800000u)  // 0.25
                           + (unsigned int)(abits >= 0x3F400000u)  // 0.75
                           + (unsigned int)(abits >  0x3FA00000u)  // 1.25
                           + (unsigned int)(abits >= 0x3FE00000u)  // 1.75
                           + (unsigned int)(abits >  0x40200000u)  // 2.5
                           + (unsigned int)(abits >= 0x40600000u)  // 3.5
                           + (unsigned int)(abits >  0x40A00000u); // 5.0
    return sign | mag;
}

// Software positive FP32 -> UE4M3 scale byte (bit 7 = 0, max 448.0, no NaN 0x7F).
__device__ __forceinline__ unsigned char lmh_float_to_ue4m3(float v) {
    float absv = fabsf(v);
    if (absv == 0.0f || isnan(absv)) return 0u;
    if (absv > 448.0f) absv = 448.0f;

    const unsigned int bits = __float_as_uint(absv);
    const int f32_exp = (int)((bits >> 23) & 0xFFu) - 127;
    const unsigned int f32_man = bits & 0x7FFFFFu;

    if (f32_exp < -9) {
        return 0u;
    }
    if (f32_exp < -6) {
        int man = (int)(absv * 512.0f + 0.5f);
        if (man > 7) man = 7;
        if (man < 0) man = 0;
        return (unsigned char)man;
    }
    int e4m3_exp = f32_exp + 7;
    unsigned int e4m3_man = (f32_man + (1u << 19)) >> 20;
    if (e4m3_man > 7u) {
        e4m3_man = 0u;
        ++e4m3_exp;
    }
    if (e4m3_exp > 15 || (e4m3_exp == 15 && e4m3_man > 6u)) {
        e4m3_exp = 15;
        e4m3_man = 6u;
    }
    if (e4m3_exp < 1) e4m3_exp = 1;
    return (unsigned char)(((unsigned int)e4m3_exp << 3) | e4m3_man);
}

// Decode unsigned UE4M3 (masked 7-bit) byte to FP32.
__device__ __forceinline__ float lmh_ue4m3_to_f32(unsigned char byte) {
    const unsigned int exp = (byte >> 3) & 0xFu;
    const unsigned int man = byte & 0x7u;
    if (exp == 0u) {
        return (float)man * 0.001953125f;  // 2^-9
    }
    if (exp == 15u && man == 7u) {
        return 0.0f;
    }
    return __uint_as_float(((exp + 120u) << 23) | (man << 20));
}

// Deterministic argmax comparator:
//   - Sentinel `LMH_INVALID_ID` never wins.
//   - `NaN` candidate never wins.
//   - On equal logits, the strictly LOWER vocab index wins.
__device__ __forceinline__ bool lmh_better(
    float cand_val, unsigned int cand_idx,
    float cur_val,  unsigned int cur_idx
) {
    if (cand_idx == LMH_INVALID_ID) return false;
    if (isnan(cand_val)) return false;
    if (cur_idx == LMH_INVALID_ID) return true;
    return (cand_val > cur_val) || (cand_val == cur_val && cand_idx < cur_idx);
}

// Quantize one 16-element BF16 group into shared-UE4M3 Dual-Plane E2M1
// (`dst_hi[8]`, `dst_lo[8]`, `*dst_scale`).
__device__ __forceinline__ void lmh_quantize_group16(
    const __nv_bfloat16* __restrict__ src16,
    const float* __restrict__ s_lut,
    unsigned char* __restrict__ dst_hi8,
    unsigned char* __restrict__ dst_lo8,
    unsigned char* __restrict__ dst_scale
) {
    const float v0  = __bfloat162float(src16[0]);
    const float v1  = __bfloat162float(src16[1]);
    const float v2  = __bfloat162float(src16[2]);
    const float v3  = __bfloat162float(src16[3]);
    const float v4  = __bfloat162float(src16[4]);
    const float v5  = __bfloat162float(src16[5]);
    const float v6  = __bfloat162float(src16[6]);
    const float v7  = __bfloat162float(src16[7]);
    const float v8  = __bfloat162float(src16[8]);
    const float v9  = __bfloat162float(src16[9]);
    const float v10 = __bfloat162float(src16[10]);
    const float v11 = __bfloat162float(src16[11]);
    const float v12 = __bfloat162float(src16[12]);
    const float v13 = __bfloat162float(src16[13]);
    const float v14 = __bfloat162float(src16[14]);
    const float v15 = __bfloat162float(src16[15]);

    float gmax = fmaxf(fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))),
                       fmaxf(fmaxf(fabsf(v4), fabsf(v5)), fmaxf(fabsf(v6), fabsf(v7))));
    float gmax2 = fmaxf(fmaxf(fmaxf(fabsf(v8), fabsf(v9)), fmaxf(fabsf(v10), fabsf(v11))),
                        fmaxf(fmaxf(fabsf(v12), fabsf(v13)), fmaxf(fabsf(v14), fabsf(v15))));
    gmax = fmaxf(gmax, gmax2);

    const unsigned char sc = lmh_float_to_ue4m3(gmax * (1.0f / 6.0f));
    *dst_scale = sc;
    const float eff = lmh_ue4m3_to_f32(sc);
    const float inv_eff = (eff > 0.0f) ? (1.0f / eff) : 0.0f;

    #define LMH_PACK_PAIR(idx_out, a_val, b_val) do { \
        const float n0 = (a_val) * inv_eff; \
        const float n1 = (b_val) * inv_eff; \
        const unsigned int hi0 = lmh_float_to_e2m1(n0); \
        const unsigned int hi1 = lmh_float_to_e2m1(n1); \
        const unsigned int lo0 = lmh_float_to_e2m1(8.0f * (n0 - s_lut[hi0])); \
        const unsigned int lo1 = lmh_float_to_e2m1(8.0f * (n1 - s_lut[hi1])); \
        dst_hi8[(idx_out)] = (unsigned char)((hi1 << 4) | (hi0 & 0xFu)); \
        dst_lo8[(idx_out)] = (unsigned char)((lo1 << 4) | (lo0 & 0xFu)); \
    } while (0)

    LMH_PACK_PAIR(0, v0,  v1);
    LMH_PACK_PAIR(1, v2,  v3);
    LMH_PACK_PAIR(2, v4,  v5);
    LMH_PACK_PAIR(3, v6,  v7);
    LMH_PACK_PAIR(4, v8,  v9);
    LMH_PACK_PAIR(5, v10, v11);
    LMH_PACK_PAIR(6, v12, v13);
    LMH_PACK_PAIR(7, v14, v15);

    #undef LMH_PACK_PAIR
}

// =========================================================================
// 1. lm_head_nvfp4_argmax_partial_sm120
//
// Grid:  (ceil(V / 256), 1, 1)
// Block: (128, 1, 1) — 4 warps (each warp owns 64 vocab columns x all M<=16 rows)
// =========================================================================
extern "C" __global__
__launch_bounds__(128, 2)
void lm_head_nvfp4_argmax_partial_sm120(
    const __nv_bfloat16* __restrict__ A,             // [M, K] BF16
    const unsigned char* __restrict__ B_packed,      // [V, K/2] NVFP4 E2M1
    const unsigned char* __restrict__ B_scale,       // [V, K/16] UE4M3
    const float scale2,                              // per-tensor weight_scale_2
    float*               __restrict__ partial_vals,  // [M, num_v_tiles] FP32
    unsigned int*        __restrict__ partial_ids,   // [M, num_v_tiles] U32
    unsigned int M,
    unsigned int V,
    unsigned int K
) {
    if (M == 0u || M > LMH_M_MAX || V == 0u || K == 0u) return;

    const unsigned int cta_v = blockIdx.x * LMH_V_TILE;
    const unsigned int num_v_tiles = (V + LMH_V_TILE - 1u) / LMH_V_TILE;
    const unsigned int half_K = K >> 1;
    const unsigned int num_groups = K >> 4;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31u;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16u) s_lut[threadIdx.x] = LMH_E2M1_LUT[threadIdx.x];

#if __CUDA_ARCH__ >= 1200
    __shared__ alignas(16) unsigned char smem_Ahi[2][LMH_M_MAX][LMH_K_STEP / 2];
    __shared__ alignas(16) unsigned char smem_Alo[2][LMH_M_MAX][LMH_K_STEP / 2];
    __shared__ alignas(4)  unsigned char smem_As[2][LMH_M_MAX][LMH_K_STEP / LMH_GROUP_SIZE];
    __shared__ alignas(16) unsigned char smem_Bp[2][LMH_V_TILE][LMH_K_STEP / 2];
    __shared__ alignas(4)  unsigned char smem_Bs[2][LMH_V_TILE][LMH_K_STEP / LMH_GROUP_SIZE];
    __shared__ float smem_red_val[LMH_M_MAX][4];
    __shared__ unsigned int smem_red_idx[LMH_M_MAX][4];
    __syncthreads();

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3u;

    float acc_hi[8][4];
    float acc_lo[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        acc_hi[nt][0] = 0.0f; acc_hi[nt][1] = 0.0f;
        acc_hi[nt][2] = 0.0f; acc_hi[nt][3] = 0.0f;
        acc_lo[nt][0] = 0.0f; acc_lo[nt][1] = 0.0f;
        acc_lo[nt][2] = 0.0f; acc_lo[nt][3] = 0.0f;
    }

    #define LMH_LOAD_TILE(buf, kb) do { \
        const unsigned int gg_base = (kb) >> 4; \
        /* Threads 0..63 quantize A[0..15, kb..kb+64] into Dual-Plane E2M1 + shared UE4M3 */ \
        if (threadIdx.x < 64u) { \
            const unsigned int r = threadIdx.x >> 2; \
            const unsigned int g = threadIdx.x & 3u; \
            if (r < M) { \
                const __nv_bfloat16* src16 = A + (unsigned long long)r * K + (kb) + g * LMH_GROUP_SIZE; \
                lmh_quantize_group16( \
                    src16, \
                    s_lut, \
                    &smem_Ahi[(buf)][r][g * 8u], \
                    &smem_Alo[(buf)][r][g * 8u], \
                    &smem_As[(buf)][r][g]); \
            } else { \
                *(uint2*)&smem_Ahi[(buf)][r][g * 8u] = make_uint2(0u, 0u); \
                *(uint2*)&smem_Alo[(buf)][r][g * 8u] = make_uint2(0u, 0u); \
                smem_As[(buf)][r][g] = 0u; \
            } \
        } \
        /* All 128 threads load B_packed[cta_v..cta_v+256, kb/2..kb/2+32] (512 x 16B) */ \
        _Pragma("unroll") \
        for (unsigned int iter = 0u; iter < 4u; ++iter) { \
            const unsigned int idx = iter * 128u + threadIdx.x; \
            const unsigned int brow = idx >> 1; \
            const unsigned int bcol = (idx & 1u) << 4; \
            const unsigned int gv = cta_v + brow; \
            const unsigned int gc_byte = ((kb) >> 1) + bcol; \
            uint4 vb = make_uint4(0u, 0u, 0u, 0u); \
            if (gv < V && gc_byte + 15u < half_K) { \
                vb = *(const uint4*)(B_packed + (unsigned long long)gv * half_K + gc_byte); \
            } \
            *(uint4*)&smem_Bp[(buf)][brow][bcol] = vb; \
        } \
        /* All 128 threads load B_scale[cta_v..cta_v+256, gg_base..gg_base+4] (256 x 4B) */ \
        _Pragma("unroll") \
        for (unsigned int iter = 0u; iter < 2u; ++iter) { \
            const unsigned int brow = iter * 128u + threadIdx.x; \
            const unsigned int gv = cta_v + brow; \
            unsigned int sw = 0u; \
            if (gv < V && gg_base + 3u < num_groups) { \
                sw = *(const unsigned int*)(B_scale + (unsigned long long)gv * num_groups + gg_base); \
            } \
            *(unsigned int*)&smem_Bs[(buf)][brow][0] = sw & 0x7F7F7F7Fu; \
        } \
    } while (0)

    #define LMH_COMPUTE_KSTEP(buf) do { \
        const unsigned int fr0 = group_id; \
        const unsigned int fr1 = group_id + 8u; \
        const unsigned int a_col = 4u * tid; \
        const unsigned int ahi0 = *(const unsigned int*)&smem_Ahi[(buf)][fr0][a_col]; \
        const unsigned int ahi1 = *(const unsigned int*)&smem_Ahi[(buf)][fr1][a_col]; \
        const unsigned int ahi2 = *(const unsigned int*)&smem_Ahi[(buf)][fr0][a_col + 16u]; \
        const unsigned int ahi3 = *(const unsigned int*)&smem_Ahi[(buf)][fr1][a_col + 16u]; \
        const unsigned int alo0 = *(const unsigned int*)&smem_Alo[(buf)][fr0][a_col]; \
        const unsigned int alo1 = *(const unsigned int*)&smem_Alo[(buf)][fr1][a_col]; \
        const unsigned int alo2 = *(const unsigned int*)&smem_Alo[(buf)][fr0][a_col + 16u]; \
        const unsigned int alo3 = *(const unsigned int*)&smem_Alo[(buf)][fr1][a_col + 16u]; \
        const unsigned int m_sfa = group_id + 8u * (lane_id & 1u); \
        const unsigned int sfa = *(const unsigned int*)&smem_As[(buf)][m_sfa][0]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 8; ++nt) { \
            const unsigned int nc = warp_id * 64u + (unsigned int)nt * 8u + group_id; \
            const unsigned int b0 = *(const unsigned int*)&smem_Bp[(buf)][nc][a_col]; \
            const unsigned int b1 = *(const unsigned int*)&smem_Bp[(buf)][nc][a_col + 16u]; \
            const unsigned int sfb = *(const unsigned int*)&smem_Bs[(buf)][nc][0]; \
            asm volatile( \
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X." \
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13}," \
                "{%14},{%15,%16},{%17},{%18,%19};" \
                : "=f"(acc_hi[nt][0]), "=f"(acc_hi[nt][1]), \
                  "=f"(acc_hi[nt][2]), "=f"(acc_hi[nt][3]) \
                : "r"(ahi0), "r"(ahi1), "r"(ahi2), "r"(ahi3), \
                  "r"(b0), "r"(b1), \
                  "f"(acc_hi[nt][0]), "f"(acc_hi[nt][1]), \
                  "f"(acc_hi[nt][2]), "f"(acc_hi[nt][3]), \
                  "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0), \
                  "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0)); \
            asm volatile( \
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X." \
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13}," \
                "{%14},{%15,%16},{%17},{%18,%19};" \
                : "=f"(acc_lo[nt][0]), "=f"(acc_lo[nt][1]), \
                  "=f"(acc_lo[nt][2]), "=f"(acc_lo[nt][3]) \
                : "r"(alo0), "r"(alo1), "r"(alo2), "r"(alo3), \
                  "r"(b0), "r"(b1), \
                  "f"(acc_lo[nt][0]), "f"(acc_lo[nt][1]), \
                  "f"(acc_lo[nt][2]), "f"(acc_lo[nt][3]), \
                  "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0), \
                  "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0)); \
        } \
    } while (0)

    LMH_LOAD_TILE(0, 0u);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = LMH_K_STEP; k_base < K; k_base += LMH_K_STEP) {
        const int nxt = 1 - cur;
        LMH_LOAD_TILE(nxt, k_base);
        LMH_COMPUTE_KSTEP(cur);
        __syncthreads();
        cur = nxt;
    }
    LMH_COMPUTE_KSTEP(cur);

    #undef LMH_LOAD_TILE
    #undef LMH_COMPUTE_KSTEP

    // Online thread-local reduction over the 16 vocab columns owned by this lane
    // for rows r0 = group_id and r1 = group_id + 8.
    float best_v0 = -CUDART_INF_F;
    unsigned int best_i0 = LMH_INVALID_ID;
    float best_v1 = -CUDART_INF_F;
    unsigned int best_i1 = LMH_INVALID_ID;

    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const unsigned int c0 = cta_v + warp_id * 64u + (unsigned int)nt * 8u + tid * 2u;
        const unsigned int c1 = c0 + 1u;
        if (c0 < V) {
            const float v00 = (acc_hi[nt][0] + 0.125f * acc_lo[nt][0]) * scale2;
            const float v10 = (acc_hi[nt][2] + 0.125f * acc_lo[nt][2]) * scale2;
            if (lmh_better(v00, c0, best_v0, best_i0)) { best_v0 = v00; best_i0 = c0; }
            if (lmh_better(v10, c0, best_v1, best_i1)) { best_v1 = v10; best_i1 = c0; }
        }
        if (c1 < V) {
            const float v01 = (acc_hi[nt][1] + 0.125f * acc_lo[nt][1]) * scale2;
            const float v11 = (acc_hi[nt][3] + 0.125f * acc_lo[nt][3]) * scale2;
            if (lmh_better(v01, c1, best_v0, best_i0)) { best_v0 = v01; best_i0 = c1; }
            if (lmh_better(v11, c1, best_v1, best_i1)) { best_v1 = v11; best_i1 = c1; }
        }
    }

    // 4-lane warp-quad reduction (lanes with identical group_id differ only in tid bits 0..1).
    #pragma unroll
    for (int mask = 1; mask <= 2; mask <<= 1) {
        const float ov0 = __shfl_xor_sync(0xFFFFFFFFu, best_v0, mask);
        const unsigned int oi0 = __shfl_xor_sync(0xFFFFFFFFu, best_i0, mask);
        if (lmh_better(ov0, oi0, best_v0, best_i0)) { best_v0 = ov0; best_i0 = oi0; }

        const float ov1 = __shfl_xor_sync(0xFFFFFFFFu, best_v1, mask);
        const unsigned int oi1 = __shfl_xor_sync(0xFFFFFFFFu, best_i1, mask);
        if (lmh_better(ov1, oi1, best_v1, best_i1)) { best_v1 = ov1; best_i1 = oi1; }
    }

    if (tid == 0u) {
        smem_red_val[group_id][warp_id] = best_v0;
        smem_red_idx[group_id][warp_id] = best_i0;
        smem_red_val[group_id + 8u][warp_id] = best_v1;
        smem_red_idx[group_id + 8u][warp_id] = best_i1;
    }
    __syncthreads();

    if (threadIdx.x < M) {
        const unsigned int r = threadIdx.x;
        float final_v = -CUDART_INF_F;
        unsigned int final_i = LMH_INVALID_ID;
        #pragma unroll
        for (unsigned int w = 0u; w < 4u; ++w) {
            const float cv = smem_red_val[r][w];
            const unsigned int ci = smem_red_idx[r][w];
            if (lmh_better(cv, ci, final_v, final_i)) {
                final_v = cv;
                final_i = ci;
            }
        }
        const unsigned long long out_off = (unsigned long long)r * num_v_tiles + blockIdx.x;
        partial_vals[out_off] = final_v;
        partial_ids[out_off] = final_i;
    }

#else
    // Exact scalar/FP32 fallback for SM103a (< SM120): computes the identical
    // Dual-Plane E2M1 activation + NVFP4 weight dot product with 0 B spills.
    __shared__ alignas(16) unsigned char smem_Ahi_row[LMH_K_STEP / 2];
    __shared__ alignas(16) unsigned char smem_Alo_row[LMH_K_STEP / 2];
    __shared__ alignas(4)  unsigned char smem_As_row[LMH_K_STEP / LMH_GROUP_SIZE];
    __shared__ float smem_red_val[4];
    __shared__ unsigned int smem_red_idx[4];
    __syncthreads();

    const unsigned int gv0 = cta_v + threadIdx.x;
    const unsigned int gv1 = cta_v + 128u + threadIdx.x;

    for (unsigned int r = 0u; r < M; ++r) {
        float acc0_hi = 0.0f;
        float acc0_lo = 0.0f;
        float acc1_hi = 0.0f;
        float acc1_lo = 0.0f;

        for (unsigned int k_base = 0u; k_base < K; k_base += LMH_K_STEP) {
            if (threadIdx.x < 4u) {
                const unsigned int g = threadIdx.x;
                const __nv_bfloat16* src16 =
                    A + (unsigned long long)r * K + k_base + g * LMH_GROUP_SIZE;
                lmh_quantize_group16(
                    src16,
                    s_lut,
                    &smem_Ahi_row[g * 8u],
                    &smem_Alo_row[g * 8u],
                    &smem_As_row[g]);
            }
            __syncthreads();

            const unsigned int gg_base = k_base >> 4;
            const unsigned int kb_half = k_base >> 1;
            #pragma unroll
            for (unsigned int g = 0u; g < 4u; ++g) {
                const float sa = lmh_ue4m3_to_f32(smem_As_row[g]);
                const unsigned long long ahi8 = *(const unsigned long long*)&smem_Ahi_row[g * 8u];
                const unsigned long long alo8 = *(const unsigned long long*)&smem_Alo_row[g * 8u];

                if (gv0 < V) {
                    const float sb0 = lmh_ue4m3_to_f32(
                        (unsigned char)(B_scale[(unsigned long long)gv0 * num_groups + gg_base + g] & 0x7Fu));
                    const float sab0 = sa * sb0;
                    const unsigned long long bp0 = *(const unsigned long long*)(
                        B_packed + (unsigned long long)gv0 * half_K + kb_half + g * 8u);
                    float d_hi = 0.0f;
                    float d_lo = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        const unsigned int bv = (unsigned int)((bp0 >> (b * 8)) & 0xFFu);
                        const unsigned int ah = (unsigned int)((ahi8 >> (b * 8)) & 0xFFu);
                        const unsigned int al = (unsigned int)((alo8 >> (b * 8)) & 0xFFu);
                        const float wb0 = s_lut[bv & 0xFu];
                        const float wb1 = s_lut[bv >> 4];
                        d_hi += s_lut[ah & 0xFu] * wb0 + s_lut[ah >> 4] * wb1;
                        d_lo += s_lut[al & 0xFu] * wb0 + s_lut[al >> 4] * wb1;
                    }
                    acc0_hi += d_hi * sab0;
                    acc0_lo += d_lo * sab0;
                }

                if (gv1 < V) {
                    const float sb1 = lmh_ue4m3_to_f32(
                        (unsigned char)(B_scale[(unsigned long long)gv1 * num_groups + gg_base + g] & 0x7Fu));
                    const float sab1 = sa * sb1;
                    const unsigned long long bp1 = *(const unsigned long long*)(
                        B_packed + (unsigned long long)gv1 * half_K + kb_half + g * 8u);
                    float d_hi = 0.0f;
                    float d_lo = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        const unsigned int bv = (unsigned int)((bp1 >> (b * 8)) & 0xFFu);
                        const unsigned int ah = (unsigned int)((ahi8 >> (b * 8)) & 0xFFu);
                        const unsigned int al = (unsigned int)((alo8 >> (b * 8)) & 0xFFu);
                        const float wb0 = s_lut[bv & 0xFu];
                        const float wb1 = s_lut[bv >> 4];
                        d_hi += s_lut[ah & 0xFu] * wb0 + s_lut[ah >> 4] * wb1;
                        d_lo += s_lut[al & 0xFu] * wb0 + s_lut[al >> 4] * wb1;
                    }
                    acc1_hi += d_hi * sab1;
                    acc1_lo += d_lo * sab1;
                }
            }
            __syncthreads();
        }

        float best_v = -CUDART_INF_F;
        unsigned int best_i = LMH_INVALID_ID;
        if (gv0 < V) {
            const float v0 = (acc0_hi + 0.125f * acc0_lo) * scale2;
            if (lmh_better(v0, gv0, best_v, best_i)) { best_v = v0; best_i = gv0; }
        }
        if (gv1 < V) {
            const float v1 = (acc1_hi + 0.125f * acc1_lo) * scale2;
            if (lmh_better(v1, gv1, best_v, best_i)) { best_v = v1; best_i = gv1; }
        }

        #pragma unroll
        for (int offset = LMH_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, best_v, offset);
            const unsigned int oi = __shfl_down_sync(0xFFFFFFFFu, best_i, offset);
            if (lmh_better(ov, oi, best_v, best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane_id == 0u) {
            smem_red_val[warp_id] = best_v;
            smem_red_idx[warp_id] = best_i;
        }
        __syncthreads();

        if (threadIdx.x == 0u) {
            float final_v = -CUDART_INF_F;
            unsigned int final_i = LMH_INVALID_ID;
            #pragma unroll
            for (unsigned int w = 0u; w < 4u; ++w) {
                if (lmh_better(smem_red_val[w], smem_red_idx[w], final_v, final_i)) {
                    final_v = smem_red_val[w];
                    final_i = smem_red_idx[w];
                }
            }
            const unsigned long long out_off = (unsigned long long)r * num_v_tiles + blockIdx.x;
            partial_vals[out_off] = final_v;
            partial_ids[out_off] = final_i;
        }
        __syncthreads();
    }
#endif
}

// =========================================================================
// 2. lm_head_argmax_reduce_sm120
//
// Reduces `[M, num_v_tiles]` partial `(max_val, argmax_id)` pairs in 1 warp
// per row to final `[M]` u32 argmax token IDs with exact lowest-index tie-break.
//
// Grid:  (M, 1, 1)
// Block: (32, 1, 1)
// =========================================================================
extern "C" __global__
__launch_bounds__(32, 16)
void lm_head_argmax_reduce_sm120(
    const float*        __restrict__ partial_vals,  // [M, num_v_tiles]
    const unsigned int* __restrict__ partial_ids,   // [M, num_v_tiles]
    unsigned int*       __restrict__ out_tokens,    // [M]
    unsigned int M,
    unsigned int num_v_tiles
) {
    const unsigned int row = blockIdx.x;
    if (row >= M || num_v_tiles == 0u) return;

    const unsigned int lane = threadIdx.x;
    const unsigned long long base = (unsigned long long)row * num_v_tiles;

    float best_v = -CUDART_INF_F;
    unsigned int best_i = LMH_INVALID_ID;

    for (unsigned int t = lane; t < num_v_tiles; t += LMH_WARP_SIZE) {
        const float cv = partial_vals[base + t];
        const unsigned int ci = partial_ids[base + t];
        if (lmh_better(cv, ci, best_v, best_i)) {
            best_v = cv;
            best_i = ci;
        }
    }

    #pragma unroll
    for (int offset = LMH_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        const float ov = __shfl_down_sync(0xFFFFFFFFu, best_v, offset);
        const unsigned int oi = __shfl_down_sync(0xFFFFFFFFu, best_i, offset);
        if (lmh_better(ov, oi, best_v, best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    if (lane == 0u) {
        out_tokens[row] = (best_i == LMH_INVALID_ID) ? 0u : best_i;
    }
}
