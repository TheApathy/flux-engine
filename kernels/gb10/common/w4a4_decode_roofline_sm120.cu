// SPDX-License-Identifier: AGPL-3.0-only
//
// Flux Engine — 48-SM Roofline W4A4 Decode GEMM (`split_k=1..=8`, default 3 on GB10)
// with Deterministic Fixed-Order Reduction & `W=1 == W=8 == W=16` Invariance.
//
// Kernels:
//   1. `w4a4_gemm_m16_sk3_sm120`:
//      Native Blackwell SM120/SM121 block-scaled FP4 Tensor Core decode GEMM
//      (`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//        .row.col.f32.e2m1.e2m1.f32.ue4m3`) for `M <= 16` (`W=1`, `W=8`, `W=16`
//      all fit in ONE `m16` tile height with N_TILE=64 -> 240 CTAs = 5.0 CTAs/SM
//      at N=5120, split_k=3 on 48-SM GB10).
//      - Weights (`B_packed`, `B_scale`) streamed via `LDG.E.EF.128` (`L2::evict_first`)
//      - Activations (`A_packed`, `A_scale`) loaded via `LDG.E.128.CONSTANT` (`ld.global.nc`)
//      - 2-stage register pipeline overlapping next-K global prefetch with MMA
//      - Branchless 3-op SWAR E2M1 `-0.0` (0x8) -> `+0.0` (0x0) canonicalization
//      - Deterministic K-slice partition independent of `M` (0 `atomicAdd`)
//      - Explicit inline PTX `cvt.rn.bf16.f32` in epilogue (`split_k == 1`)
//      - Exact scalar FP32 fallback on `< sm_120` (e.g. `sm_103a`)
//   2. `reduce_w4a4_sk3_bf16_sm120`:
//      Fixed ascending-order FP32 split-K workspace reduction (`s = 0..split_k-1`)
//      followed by `scale2_ab` multiply and explicit inline PTX `cvt.rn.bf16.f32`.
//      Guarantees bit-exact `W=1 == W=8 == W=16` and `B=1 == B=4/8` invariance.

#include <cuda_bf16.h>
#include <stdint.h>

#define W4SK_M_TILE     16u
#define W4SK_N_TILE     64u
#define W4SK_K_STEP     64u
#define W4SK_GROUP_SIZE 16u
#define W4SK_MAX_SPLITK 8u

// 3-op branchless SWAR E2M1 `-0.0` (0x8) -> `+0.0` (0x0) canonicalization
// across all 8 packed 4-bit nibbles with zero cross-nibble carry.
__device__ __forceinline__ unsigned int w4sk_canon_e2m1_u32(unsigned int w) {
    const unsigned int m = w & 0x77777777u;
    return w & ((m + 0x77777777u) | 0x77777777u);
}

__device__ __forceinline__ uint4 w4sk_canon_e2m1_u4(uint4 v) {
    return make_uint4(
        w4sk_canon_e2m1_u32(v.x),
        w4sk_canon_e2m1_u32(v.y),
        w4sk_canon_e2m1_u32(v.z),
        w4sk_canon_e2m1_u32(v.w));
}

// Explicit inline PTX FP32 -> BF16 round-to-nearest-even (prevents FMA drift).
__device__ __forceinline__ unsigned short w4sk_f32_to_bf16_rn(float x) {
    unsigned short bits;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(x));
    return bits;
}

__device__ __forceinline__ unsigned long long w4sk_l2_evict_first_policy() {
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    return pol;
}

// 128-bit streaming weight load with cache-streaming + L2::evict_first
// -> SASS `LDG.E.EF.128` with L2::evict_first policy descriptor.
__device__ __forceinline__ uint4 w4sk_ldg_ef_128(
    const void* ptr,
    unsigned long long pol
) {
    uint4 v;
    asm volatile(
        "ld.global.cs.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(ptr), "l"(pol));
    return v;
}

// 32-bit streaming weight-scale load with cache-streaming + L2::evict_first
// -> SASS `LDG.E.EF` with L2::evict_first policy descriptor.
__device__ __forceinline__ unsigned int w4sk_ldg_ef_32(
    const void* ptr,
    unsigned long long pol
) {
    unsigned int v;
    asm volatile(
        "ld.global.cs.L2::cache_hint.u32 %0, [%1], %2;"
        : "=r"(v)
        : "l"(ptr), "l"(pol));
    return v;
}

// 128-bit read-only activation load -> SASS `LDG.E.128.CONSTANT`.
__device__ __forceinline__ uint4 w4sk_ldg_nc_128(const void* ptr) {
    uint4 v;
    asm volatile(
        "ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(ptr));
    return v;
}

// 32-bit read-only activation-scale load -> SASS `LDG.E.CONSTANT`.
__device__ __forceinline__ unsigned int w4sk_ldg_nc_32(const void* ptr) {
    unsigned int v;
    asm volatile(
        "ld.global.nc.u32 %0, [%1];"
        : "=r"(v)
        : "l"(ptr));
    return v;
}

#if __CUDA_ARCH__ < 1200
__device__ __constant__ float W4SK_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float w4sk_ue4m3_to_f32(unsigned char byte) {
    const unsigned int exp = (byte >> 3) & 0xFu;
    const unsigned int man = byte & 0x7u;
    if (exp == 0u) return (float)man * 0.001953125f;
    if (exp == 15u && man == 7u) return 0.0f;
    return __uint_as_float(((exp + 120u) << 23) | (man << 20));
}
#endif

// =========================================================================
// 1. w4a4_gemm_m16_sk3_sm120
//
// Grid:  (ceil(N / 64), 1, split_k)
// Block: (128, 1, 1) — 4 warps (each warp owns 16 N-cols x all M <= 16 rows)
// SMEM:  2,880 B (Ap 512B + As 64B + Bp 2048B + Bs 256B)
// =========================================================================
extern "C" __global__
__launch_bounds__(128, 4)
void w4a4_gemm_m16_sk3_sm120(
    const unsigned char* __restrict__ A_packed,       // [M, K/2] E2M1
    const unsigned char* __restrict__ A_scale,        // [M, K/16] UE4M3
    const unsigned char* __restrict__ B_packed,       // [N, K/2] E2M1
    const unsigned char* __restrict__ B_scale,        // [N, K/16] UE4M3
    float*               __restrict__ workspace_f32,  // [split_k, M, N] FP32 (when split_k > 1)
    __nv_bfloat16*       __restrict__ C_bf16,         // [M, N] BF16 (when split_k == 1)
    const float scale2_ab,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int split_k
) {
    if (M == 0u || M > W4SK_M_TILE || N == 0u || K == 0u || (K & 63u) != 0u) return;
    if (split_k == 0u || split_k > W4SK_MAX_SPLITK) return;

    const unsigned int cta_n = blockIdx.x * W4SK_N_TILE;
    const unsigned int sk = blockIdx.z;
    if (cta_n >= N || sk >= split_k) return;

    const unsigned int num_k_tiles = K >> 6;
    const unsigned int tile_begin = (sk * num_k_tiles) / split_k;
    const unsigned int tile_end = ((sk + 1u) * num_k_tiles) / split_k;
    const unsigned int k_begin = tile_begin << 6;
    const unsigned int k_end = tile_end << 6;

    const int64_t half_K = (int64_t)(K >> 1);
    const int64_t num_groups = (int64_t)(K >> 4);
    const unsigned long long pol_ef = w4sk_l2_evict_first_policy();

#if __CUDA_ARCH__ >= 1200
    __shared__ alignas(16) unsigned char smem_Ap[W4SK_M_TILE][W4SK_K_STEP / 2];
    __shared__ alignas(4)  unsigned char smem_As[W4SK_M_TILE][W4SK_K_STEP / W4SK_GROUP_SIZE];
    __shared__ alignas(16) unsigned char smem_Bp[W4SK_N_TILE][W4SK_K_STEP / 2];
    __shared__ alignas(4)  unsigned char smem_Bs[W4SK_N_TILE][W4SK_K_STEP / W4SK_GROUP_SIZE];

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31u;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3u;

    float acc[2][4];
    acc[0][0] = 0.0f; acc[0][1] = 0.0f; acc[0][2] = 0.0f; acc[0][3] = 0.0f;
    acc[1][0] = 0.0f; acc[1][1] = 0.0f; acc[1][2] = 0.0f; acc[1][3] = 0.0f;

    // Precompute warp-uniform per-thread load coordinates for the 2-stage register pipeline:
    //   - Threads  0..127 -> B_packed (1 x uint4 each, LDG.E.EF.128)
    //   - Threads  0..63  -> B_scale  (1 x u32   each, LDG.E.EF)
    //   - Threads 64..95  -> A_packed (1 x uint4 each, LDG.E.128.CONSTANT)
    //   - Threads 96..111 -> A_scale  (1 x u32   each, LDG.E.CONSTANT)
    const unsigned int bp_row = threadIdx.x >> 1;
    const unsigned int bp_col = (threadIdx.x & 1u) << 4;
    const unsigned int bp_gn = cta_n + bp_row;
    const bool bp_valid = (bp_gn < N);

    const unsigned int bs_row = threadIdx.x;
    const unsigned int bs_gn = cta_n + bs_row;
    const bool bs_valid = (threadIdx.x < W4SK_N_TILE) && (bs_gn < N);

    const unsigned int ap_idx = threadIdx.x - 64u;
    const unsigned int ap_row = ap_idx >> 1;
    const unsigned int ap_col = (ap_idx & 1u) << 4;
    const bool ap_thread = (threadIdx.x >= 64u) && (threadIdx.x < 96u);
    const bool ap_valid = ap_thread && (ap_row < M);

    const unsigned int as_row = threadIdx.x - 96u;
    const bool as_thread = (threadIdx.x >= 96u) && (threadIdx.x < 112u);
    const bool as_valid = as_thread && (as_row < M);

    #define W4SK_PREFETCH_REGS(kb, r_bp, r_bs, r_ap, r_as) do { \
        const int64_t gc_byte = (int64_t)((kb) >> 1); \
        const int64_t gg_base = (int64_t)((kb) >> 4); \
        (r_bp) = make_uint4(0u, 0u, 0u, 0u); \
        if (bp_valid) { \
            (r_bp) = w4sk_ldg_ef_128( \
                B_packed + (int64_t)bp_gn * half_K + gc_byte + (int64_t)bp_col, pol_ef); \
        } \
        (r_bs) = 0u; \
        if (bs_valid) { \
            (r_bs) = w4sk_ldg_ef_32( \
                B_scale + (int64_t)bs_gn * num_groups + gg_base, pol_ef) & 0x7F7F7F7Fu; \
        } \
        (r_ap) = make_uint4(0u, 0u, 0u, 0u); \
        if (ap_valid) { \
            (r_ap) = w4sk_ldg_nc_128( \
                A_packed + (int64_t)ap_row * half_K + gc_byte + (int64_t)ap_col); \
        } \
        (r_as) = 0u; \
        if (as_valid) { \
            (r_as) = w4sk_ldg_nc_32( \
                A_scale + (int64_t)as_row * num_groups + gg_base) & 0x7F7F7F7Fu; \
        } \
    } while (0)

    #define W4SK_COMMIT_SMEM(r_bp, r_bs, r_ap, r_as) do { \
        *(uint4*)&smem_Bp[bp_row][bp_col] = w4sk_canon_e2m1_u4(r_bp); \
        if (threadIdx.x < W4SK_N_TILE) { \
            *(unsigned int*)&smem_Bs[bs_row][0] = (r_bs); \
        } \
        if (ap_thread) { \
            *(uint4*)&smem_Ap[ap_row][ap_col] = w4sk_canon_e2m1_u4(r_ap); \
        } \
        if (as_thread) { \
            *(unsigned int*)&smem_As[as_row][0] = (r_as); \
        } \
    } while (0)

    #define W4SK_COMPUTE_MMA() do { \
        const unsigned int fr0 = group_id; \
        const unsigned int fr1 = group_id + 8u; \
        const unsigned int a_col = 4u * tid; \
        const unsigned int a0 = *(const unsigned int*)&smem_Ap[fr0][a_col]; \
        const unsigned int a1 = *(const unsigned int*)&smem_Ap[fr1][a_col]; \
        const unsigned int a2 = *(const unsigned int*)&smem_Ap[fr0][a_col + 16u]; \
        const unsigned int a3 = *(const unsigned int*)&smem_Ap[fr1][a_col + 16u]; \
        const unsigned int m_sfa = group_id + 8u * (lane_id & 1u); \
        const unsigned int sfa = *(const unsigned int*)&smem_As[m_sfa][0]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 2; ++nt) { \
            const unsigned int nc = warp_id * 16u + (unsigned int)nt * 8u + group_id; \
            const unsigned int b0 = *(const unsigned int*)&smem_Bp[nc][a_col]; \
            const unsigned int b1 = *(const unsigned int*)&smem_Bp[nc][a_col + 16u]; \
            const unsigned int sfb = *(const unsigned int*)&smem_Bs[nc][0]; \
            asm volatile( \
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X." \
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13}," \
                "{%14},{%15,%16},{%17},{%18,%19};" \
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]), \
                  "=f"(acc[nt][2]), "=f"(acc[nt][3]) \
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), \
                  "r"(b0), "r"(b1), \
                  "f"(acc[nt][0]), "f"(acc[nt][1]), \
                  "f"(acc[nt][2]), "f"(acc[nt][3]), \
                  "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0), \
                  "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0)); \
        } \
    } while (0)

    if (k_begin < k_end) {
        uint4 cur_bp, cur_ap;
        unsigned int cur_bs, cur_as;
        W4SK_PREFETCH_REGS(k_begin, cur_bp, cur_bs, cur_ap, cur_as);
        W4SK_COMMIT_SMEM(cur_bp, cur_bs, cur_ap, cur_as);
        __syncthreads();

        for (unsigned int k_next = k_begin + W4SK_K_STEP; k_next < k_end; k_next += W4SK_K_STEP) {
            uint4 nxt_bp, nxt_ap;
            unsigned int nxt_bs, nxt_as;
            W4SK_PREFETCH_REGS(k_next, nxt_bp, nxt_bs, nxt_ap, nxt_as);
            W4SK_COMPUTE_MMA();
            __syncthreads();
            W4SK_COMMIT_SMEM(nxt_bp, nxt_bs, nxt_ap, nxt_as);
            __syncthreads();
        }
        W4SK_COMPUTE_MMA();
    }

    #undef W4SK_PREFETCH_REGS
    #undef W4SK_COMMIT_SMEM
    #undef W4SK_COMPUTE_MMA

    const unsigned int r0 = group_id;
    const unsigned int r1 = group_id + 8u;
    const int64_t ws_slice_base = (int64_t)sk * (int64_t)M * (int64_t)N;

    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const unsigned int c0 = cta_n + warp_id * 16u + (unsigned int)nt * 8u + tid * 2u;
        const unsigned int c1 = c0 + 1u;
        if (split_k == 1u) {
            if (r0 < M && c0 < N) {
                *(unsigned short*)&C_bf16[(int64_t)r0 * N + c0] =
                    w4sk_f32_to_bf16_rn(acc[nt][0] * scale2_ab);
            }
            if (r0 < M && c1 < N) {
                *(unsigned short*)&C_bf16[(int64_t)r0 * N + c1] =
                    w4sk_f32_to_bf16_rn(acc[nt][1] * scale2_ab);
            }
            if (r1 < M && c0 < N) {
                *(unsigned short*)&C_bf16[(int64_t)r1 * N + c0] =
                    w4sk_f32_to_bf16_rn(acc[nt][2] * scale2_ab);
            }
            if (r1 < M && c1 < N) {
                *(unsigned short*)&C_bf16[(int64_t)r1 * N + c1] =
                    w4sk_f32_to_bf16_rn(acc[nt][3] * scale2_ab);
            }
        } else {
            if (r0 < M && c0 < N) workspace_f32[ws_slice_base + (int64_t)r0 * N + c0] = acc[nt][0];
            if (r0 < M && c1 < N) workspace_f32[ws_slice_base + (int64_t)r0 * N + c1] = acc[nt][1];
            if (r1 < M && c0 < N) workspace_f32[ws_slice_base + (int64_t)r1 * N + c0] = acc[nt][2];
            if (r1 < M && c1 < N) workspace_f32[ws_slice_base + (int64_t)r1 * N + c1] = acc[nt][3];
        }
    }

#else
    // Exact scalar FP32 fallback for SM103a (< SM120):
    // 128 threads = 16 rows x 8 column strips of 8 columns each (64 N cols/CTA).
    // Preserves identical K-slice partition, 3-op SWAR E2M1 canonicalization,
    // fixed reduction order, and 0 B stack/spills.
    __shared__ float s_lut[16];
    if (threadIdx.x < 16u) s_lut[threadIdx.x] = W4SK_E2M1_LUT[threadIdx.x];
    __syncthreads();

    const unsigned int r = threadIdx.x & 15u;
    const unsigned int c_base = cta_n + (threadIdx.x >> 4) * 8u;
    if (r >= M) return;

    float acc[8];
    #pragma unroll
    for (int ci = 0; ci < 8; ++ci) acc[ci] = 0.0f;

    for (unsigned int kb = k_begin; kb < k_end; kb += W4SK_K_STEP) {
        const int64_t gc_byte = (int64_t)(kb >> 1);
        const int64_t gg_base = (int64_t)(kb >> 4);
        const uint4 ap_lo = w4sk_canon_e2m1_u4(
            w4sk_ldg_nc_128(A_packed + (int64_t)r * half_K + gc_byte));
        const uint4 ap_hi = w4sk_canon_e2m1_u4(
            w4sk_ldg_nc_128(A_packed + (int64_t)r * half_K + gc_byte + 16));
        const unsigned int as_w =
            w4sk_ldg_nc_32(A_scale + (int64_t)r * num_groups + gg_base) & 0x7F7F7F7Fu;

        const unsigned int a_words[8] = {
            ap_lo.x, ap_lo.y, ap_lo.z, ap_lo.w,
            ap_hi.x, ap_hi.y, ap_hi.z, ap_hi.w
        };

        #pragma unroll
        for (int ci = 0; ci < 8; ++ci) {
            const unsigned int col = c_base + (unsigned int)ci;
            if (col >= N) continue;
            const uint4 bp_lo = w4sk_canon_e2m1_u4(
                w4sk_ldg_ef_128(B_packed + (int64_t)col * half_K + gc_byte, pol_ef));
            const uint4 bp_hi = w4sk_canon_e2m1_u4(
                w4sk_ldg_ef_128(B_packed + (int64_t)col * half_K + gc_byte + 16, pol_ef));
            const unsigned int bs_w =
                w4sk_ldg_ef_32(B_scale + (int64_t)col * num_groups + gg_base, pol_ef) & 0x7F7F7F7Fu;

            const unsigned int b_words[8] = {
                bp_lo.x, bp_lo.y, bp_lo.z, bp_lo.w,
                bp_hi.x, bp_hi.y, bp_hi.z, bp_hi.w
            };

            float tile_sum = 0.0f;
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const float sa = w4sk_ue4m3_to_f32((unsigned char)((as_w >> (g * 8)) & 0x7Fu));
                const float sb = w4sk_ue4m3_to_f32((unsigned char)((bs_w >> (g * 8)) & 0x7Fu));
                float g_dot = 0.0f;
                #pragma unroll
                for (int w_idx = 0; w_idx < 2; ++w_idx) {
                    const unsigned int aw = a_words[g * 2 + w_idx];
                    const unsigned int bw = b_words[g * 2 + w_idx];
                    #pragma unroll
                    for (int nib = 0; nib < 8; ++nib) {
                        const unsigned int an = (aw >> (nib * 4)) & 0xFu;
                        const unsigned int bn = (bw >> (nib * 4)) & 0xFu;
                        g_dot += s_lut[an] * s_lut[bn];
                    }
                }
                tile_sum += g_dot * (sa * sb);
            }
            acc[ci] += tile_sum;
        }
    }

    const int64_t ws_slice_base = (int64_t)sk * (int64_t)M * (int64_t)N;
    #pragma unroll
    for (int ci = 0; ci < 8; ++ci) {
        const unsigned int col = c_base + (unsigned int)ci;
        if (col >= N) continue;
        if (split_k == 1u) {
            *(unsigned short*)&C_bf16[(int64_t)r * N + col] =
                w4sk_f32_to_bf16_rn(acc[ci] * scale2_ab);
        } else {
            workspace_f32[ws_slice_base + (int64_t)r * N + col] = acc[ci];
        }
    }
#endif
}

// =========================================================================
// 2. reduce_w4a4_sk3_bf16_sm120
//
// Deterministic fixed-order split-K workspace reduction (`s = 0..split_k-1`)
// with explicit inline PTX `cvt.rn.bf16.f32` epilogue (ZERO atomicAdd).
//
// Grid:  (ceil(M * N / 256), 1, 1)
// Block: (256, 1, 1)
// =========================================================================
extern "C" __global__
__launch_bounds__(256, 4)
void reduce_w4a4_sk3_bf16_sm120(
    const float*   __restrict__ workspace_f32,  // [split_k, M, N] FP32
    __nv_bfloat16* __restrict__ C_bf16,         // [M, N] BF16
    const float scale2_ab,
    unsigned int M,
    unsigned int N,
    unsigned int split_k
) {
    if (M == 0u || N == 0u || split_k == 0u || split_k > W4SK_MAX_SPLITK) return;

    const int64_t total = (int64_t)M * (int64_t)N;
    const int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (idx >= total) return;

    float sum = 0.0f;
    for (unsigned int s = 0u; s < split_k; ++s) {
        sum += workspace_f32[(int64_t)s * total + idx];
    }
    *(unsigned short*)&C_bf16[idx] = w4sk_f32_to_bf16_rn(sum * scale2_ab);
}
