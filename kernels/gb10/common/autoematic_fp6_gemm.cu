// SPDX-License-Identifier: AGPL-3.0-only
//
// Flux Engine — AutoEMatic (+0.4 bpw) AutoFP6 Extended-K Blackwell GEMM & GEMV.
//
// Implements runtime-adaptive NVFP4 -> AutoFP6 mantissa refinement using
// Blackwell's native block-scaled E2M1xE2M1 Tensor Core MMA instruction:
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//       .row.col.f32.e2m1.e2m1.f32.ue4m3
//
// Mathematical contract (AutoFP6, arXiv:2611.10165 §3.2 & §4.1):
//   - Base weight: NVFP4 E2M1 codes B_packed[N, K/2] with per-16 UE4M3 group
//     scales B_scale[N, K/16] and per-tensor FP32 scale B_scale2.
//   - Sparse refinement sidecar (+0.4 bpw at rho=20% subgroup budget):
//       * Each 16-element group splits into four 4-element subgroups (SG=4).
//       * B_refine_mask[K/16] holds a 4-bit row-shared mask (low nibble) where
//         bit sg in {0,1,2,3} is 1 iff subgroup sg is refined.
//       * B_group_prefix[K/16] holds the exclusive prefix sum of popcount(mask)
//         so group g's refined subgroup bytes start at B_group_prefix[g].
//       * B_refine_codes[N, num_refined_sg] stores 1 byte per refined subgroup:
//         four 2-bit two's-complement signed offsets d in {-2, -1, 0, +1}
//         packed little-endian (2 bits/elem):
//           code 0 (00) -> d =  0 -> 4*delta =  0.0 -> E2M1 0x0
//           code 1 (01) -> d = +1 -> 4*delta = +0.5 -> E2M1 0x1
//           code 2 (10) -> d = -2 -> 4*delta = -1.0 -> E2M1 0xA
//           code 3 (11) -> d = -1 -> 4*delta = -0.5 -> E2M1 0x9
//       * Because delta = d * 2^-3 = (1/4) * (4*delta) and 4*delta is exact
//         in E2M1, the refined weight is W_hat = s * (W_q + 0.25 * W_delta4).
//   - Epilogue accumulates `scale2_ab * (acc_base + 0.25f * acc_ref)` in FP32
//     so subnormal UE4M3 group scales never underflow when scaled by 0.25.
//   - When `enable_refine == 0` (low-entropy tokens) or a K=64 tile has zero
//     refined subgroups, the refinement load and second MMA are skipped
//     uniformly across the CTA (100% native NVFP4 fast path).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define AE_M_TILE     64
#define AE_N_TILE     64
#define AE_K_STEP     64
#define AE_GROUP_SIZE 16
#define AE_WARP_SIZE  32

__device__ __constant__ float AE_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Branchless 2-bit signed offset -> 4-bit E2M1 code for 4*delta:
//   0 (00) -> 0x0 (+0.0), 1 (01) -> 0x1 (+0.5),
//   2 (10) -> 0xA (-1.0), 3 (11) -> 0x9 (-0.5).
__device__ __forceinline__ unsigned int ae_code2_to_e2m1(unsigned int c) {
    const unsigned int sign = (c & 2u) << 2;
    const unsigned int mag  = (c & 1u) | ((c & 2u) & (~(c << 1) & 2u));
    return sign | mag;
}

// Expand 1 sparse refinement byte (4 x 2-bit offsets) into 2 packed E2M1 bytes
// (4 nibbles representing 4*delta in {-1.0, -0.5, 0.0, +0.5}).
__device__ __forceinline__ unsigned short ae_expand_sg_to_e2m1x4(unsigned char raw) {
    const unsigned int n0 = ae_code2_to_e2m1((unsigned int)(raw & 3u));
    const unsigned int n1 = ae_code2_to_e2m1((unsigned int)((raw >> 2) & 3u));
    const unsigned int n2 = ae_code2_to_e2m1((unsigned int)((raw >> 4) & 3u));
    const unsigned int n3 = ae_code2_to_e2m1((unsigned int)((raw >> 6) & 3u));
    const unsigned int b0 = (n1 << 4) | n0;
    const unsigned int b1 = (n3 << 4) | n2;
    return (unsigned short)((b1 << 8) | b0);
}

// =========================================================================
// autoematic_fp6_gemm_sm120 — Extended-K native Blackwell FP4 TC AutoFP6 GEMM
//
// Grid:  (ceil(N/64), ceil(M/64), 1)
// Block: (128, 1, 1) — 4 warps (each warp owns 16 M rows x 64 N cols)
// SMEM:  double-buffered Ap(4KB) + Bp(4KB) + Br(4KB) + As(512B) + Bs(512B) = 13 KB
// =========================================================================
extern "C" __global__
__launch_bounds__(128, 4)
void autoematic_fp6_gemm_sm120(
    const unsigned char* __restrict__ A_packed,        // [M, K/2]
    const unsigned char* __restrict__ A_scale,         // [M, K/16]
    const unsigned char* __restrict__ B_packed,        // [N, K/2]
    const unsigned char* __restrict__ B_scale,         // [N, K/16]
    const unsigned char* __restrict__ B_refine_codes,  // [N, num_refined_sg]
    const unsigned char* __restrict__ B_refine_mask,   // [K/16]
    const unsigned int*  __restrict__ B_group_prefix,  // [K/16]
    const unsigned int num_refined_sg,
    const float scale2_ab,
    const unsigned int enable_refine,
    __nv_bfloat16* __restrict__ C,                     // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
#if __CUDA_ARCH__ >= 1200
    const unsigned int cta_n = blockIdx.x * AE_N_TILE;
    const unsigned int cta_m = blockIdx.y * AE_M_TILE;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31u;
    const unsigned int warp_m_offset = warp_id * 16u;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3u;

    __shared__ alignas(16) unsigned char smem_Ap[2][AE_M_TILE][AE_K_STEP / 2];
    __shared__ alignas(16) unsigned char smem_Bp[2][AE_N_TILE][AE_K_STEP / 2];
    __shared__ alignas(16) unsigned char smem_Br[2][AE_N_TILE][AE_K_STEP / 2];
    __shared__ alignas(4)  unsigned char smem_As[2][AE_M_TILE][AE_K_STEP / AE_GROUP_SIZE];
    __shared__ alignas(4)  unsigned char smem_Bs[2][AE_N_TILE][AE_K_STEP / AE_GROUP_SIZE];

    float acc_base[8][4];
    float acc_ref[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        acc_base[nt][0] = 0.0f; acc_base[nt][1] = 0.0f;
        acc_base[nt][2] = 0.0f; acc_base[nt][3] = 0.0f;
        acc_ref[nt][0]  = 0.0f; acc_ref[nt][1]  = 0.0f;
        acc_ref[nt][2]  = 0.0f; acc_ref[nt][3]  = 0.0f;
    }

    const unsigned int half_K = K >> 1;
    const unsigned int num_groups = K >> 4;
    const bool refine_enabled = (enable_refine != 0u) && (B_refine_codes != nullptr) &&
                                (B_refine_mask != nullptr) && (B_group_prefix != nullptr);

    #define AE_LOAD_TILE(buf, kb, has_ref_out) do { \
        const unsigned int gg_base = (kb) >> 4; \
        /* A_packed: 64 rows x 32 B = 2048 B -> 128 threads x 16 B */ \
        { \
            const unsigned int row = threadIdx.x >> 1; \
            const unsigned int col = (threadIdx.x & 1u) << 4; \
            const unsigned int gr = cta_m + row; \
            const unsigned int gc_byte = ((kb) >> 1) + col; \
            uint4 v = make_uint4(0u, 0u, 0u, 0u); \
            if (gr < M && gc_byte + 15u < half_K) { \
                v = *(const uint4*)(A_packed + (unsigned long long)gr * half_K + gc_byte); \
            } \
            *(uint4*)&smem_Ap[(buf)][row][col] = v; \
        } \
        /* B_packed: 64 rows x 32 B = 2048 B -> 128 threads x 16 B */ \
        { \
            const unsigned int row = threadIdx.x >> 1; \
            const unsigned int col = (threadIdx.x & 1u) << 4; \
            const unsigned int gn = cta_n + row; \
            const unsigned int gc_byte = ((kb) >> 1) + col; \
            uint4 v = make_uint4(0u, 0u, 0u, 0u); \
            if (gn < N && gc_byte + 15u < half_K) { \
                v = *(const uint4*)(B_packed + (unsigned long long)gn * half_K + gc_byte); \
            } \
            *(uint4*)&smem_Bp[(buf)][row][col] = v; \
        } \
        /* A_scale & B_scale: 64 rows x 4 B each */ \
        if (threadIdx.x < AE_M_TILE) { \
            const unsigned int row = threadIdx.x; \
            const unsigned int gr = cta_m + row; \
            unsigned int s_word = 0u; \
            if (gr < M && gg_base + 3u < num_groups) { \
                s_word = *(const unsigned int*)(A_scale + (unsigned long long)gr * num_groups + gg_base); \
            } \
            *(unsigned int*)&smem_As[(buf)][row][0] = s_word & 0x7F7F7F7Fu; \
        } else { \
            const unsigned int row = threadIdx.x - AE_M_TILE; \
            const unsigned int gn = cta_n + row; \
            unsigned int s_word = 0u; \
            if (gn < N && gg_base + 3u < num_groups) { \
                s_word = *(const unsigned int*)(B_scale + (unsigned long long)gn * num_groups + gg_base); \
            } \
            *(unsigned int*)&smem_Bs[(buf)][row][0] = s_word & 0x7F7F7F7Fu; \
        } \
        /* Check row-shared masks for the 4 groups in this K=64 tile */ \
        unsigned int m_tile = 0u; \
        if (refine_enabled && gg_base + 3u < num_groups) { \
            const unsigned int m0 = (unsigned int)(B_refine_mask[gg_base + 0u] & 0x0Fu); \
            const unsigned int m1 = (unsigned int)(B_refine_mask[gg_base + 1u] & 0x0Fu); \
            const unsigned int m2 = (unsigned int)(B_refine_mask[gg_base + 2u] & 0x0Fu); \
            const unsigned int m3 = (unsigned int)(B_refine_mask[gg_base + 3u] & 0x0Fu); \
            m_tile = m0 | (m1 << 4) | (m2 << 8) | (m3 << 12); \
        } \
        (has_ref_out) = (m_tile != 0u); \
        if (m_tile != 0u) { \
            /* 128 threads: each thread handles 1 row x 2 groups (16 bytes of smem_Br) */ \
            const unsigned int row = threadIdx.x >> 1; \
            const unsigned int half_sel = threadIdx.x & 1u; /* 0: groups 0,1; 1: groups 2,3 */ \
            const unsigned int gn = cta_n + row; \
            const unsigned char* ref_row = \
                (gn < N) ? (B_refine_codes + (unsigned long long)gn * num_refined_sg) : nullptr; \
            _Pragma("unroll") \
            for (unsigned int gi = 0u; gi < 2u; ++gi) { \
                const unsigned int g_local = half_sel * 2u + gi; \
                const unsigned int mask4 = (m_tile >> (g_local * 4u)) & 0x0Fu; \
                unsigned int sg_idx = B_group_prefix[gg_base + g_local]; \
                unsigned short* dst16 = (unsigned short*)&smem_Br[(buf)][row][g_local * 8u]; \
                _Pragma("unroll") \
                for (unsigned int sg = 0u; sg < 4u; ++sg) { \
                    if ((mask4 & (1u << sg)) != 0u) { \
                        unsigned char code_byte = (ref_row != nullptr) ? ref_row[sg_idx] : (unsigned char)0; \
                        dst16[sg] = ae_expand_sg_to_e2m1x4(code_byte); \
                        ++sg_idx; \
                    } else { \
                        dst16[sg] = 0u; \
                    } \
                } \
            } \
        } \
    } while (0)

    #define AE_COMPUTE_KSTEP(buf, has_ref) do { \
        const unsigned int fr0 = warp_m_offset + group_id; \
        const unsigned int fr1 = fr0 + 8u; \
        const unsigned int a_col = 4u * tid; \
        const unsigned int a0 = *(const unsigned int*)&smem_Ap[(buf)][fr0][a_col]; \
        const unsigned int a1 = *(const unsigned int*)&smem_Ap[(buf)][fr1][a_col]; \
        const unsigned int a2 = *(const unsigned int*)&smem_Ap[(buf)][fr0][a_col + 16u]; \
        const unsigned int a3 = *(const unsigned int*)&smem_Ap[(buf)][fr1][a_col + 16u]; \
        const unsigned int m_sfa = warp_m_offset + group_id + 8u * (lane_id & 1u); \
        const unsigned int sfa = *(const unsigned int*)&smem_As[(buf)][m_sfa][0]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 8; ++nt) { \
            const unsigned int nc = (unsigned int)nt * 8u + group_id; \
            const unsigned int b0 = *(const unsigned int*)&smem_Bp[(buf)][nc][a_col]; \
            const unsigned int b1 = *(const unsigned int*)&smem_Bp[(buf)][nc][a_col + 16u]; \
            const unsigned int sfb = *(const unsigned int*)&smem_Bs[(buf)][nc][0]; \
            asm volatile( \
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X." \
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13}," \
                "{%14},{%15,%16},{%17},{%18,%19};" \
                : "=f"(acc_base[nt][0]), "=f"(acc_base[nt][1]), \
                  "=f"(acc_base[nt][2]), "=f"(acc_base[nt][3]) \
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), \
                  "r"(b0), "r"(b1), \
                  "f"(acc_base[nt][0]), "f"(acc_base[nt][1]), \
                  "f"(acc_base[nt][2]), "f"(acc_base[nt][3]), \
                  "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0), \
                  "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0)); \
            if (has_ref) { \
                const unsigned int br0 = *(const unsigned int*)&smem_Br[(buf)][nc][a_col]; \
                const unsigned int br1 = *(const unsigned int*)&smem_Br[(buf)][nc][a_col + 16u]; \
                asm volatile( \
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X." \
                    "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13}," \
                    "{%14},{%15,%16},{%17},{%18,%19};" \
                    : "=f"(acc_ref[nt][0]), "=f"(acc_ref[nt][1]), \
                      "=f"(acc_ref[nt][2]), "=f"(acc_ref[nt][3]) \
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), \
                      "r"(br0), "r"(br1), \
                      "f"(acc_ref[nt][0]), "f"(acc_ref[nt][1]), \
                      "f"(acc_ref[nt][2]), "f"(acc_ref[nt][3]), \
                      "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0), \
                      "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0)); \
            } \
        } \
    } while (0)

    if (K == 0u) return;

    bool cur_has_ref = false;
    AE_LOAD_TILE(0, 0u, cur_has_ref);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = AE_K_STEP; k_base < K; k_base += AE_K_STEP) {
        const int nxt = 1 - cur;
        bool nxt_has_ref = false;
        AE_LOAD_TILE(nxt, k_base, nxt_has_ref);
        AE_COMPUTE_KSTEP(cur, cur_has_ref);
        __syncthreads();
        cur = nxt;
        cur_has_ref = nxt_has_ref;
    }
    AE_COMPUTE_KSTEP(cur, cur_has_ref);

    #undef AE_LOAD_TILE
    #undef AE_COMPUTE_KSTEP

    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const unsigned int c0 = cta_n + (unsigned int)nt * 8u + tid * 2u;
        const unsigned int c1 = c0 + 1u;
        const unsigned int r0 = cta_m + warp_m_offset + group_id;
        const unsigned int r1 = r0 + 8u;
        const float v00 = (acc_base[nt][0] + 0.25f * acc_ref[nt][0]) * scale2_ab;
        const float v01 = (acc_base[nt][1] + 0.25f * acc_ref[nt][1]) * scale2_ab;
        const float v10 = (acc_base[nt][2] + 0.25f * acc_ref[nt][2]) * scale2_ab;
        const float v11 = (acc_base[nt][3] + 0.25f * acc_ref[nt][3]) * scale2_ab;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 * N + c0] = __float2bfloat16(v00);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 * N + c1] = __float2bfloat16(v01);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 * N + c0] = __float2bfloat16(v10);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 * N + c1] = __float2bfloat16(v11);
    }
#else
    (void)A_packed; (void)A_scale; (void)B_packed; (void)B_scale;
    (void)B_refine_codes; (void)B_refine_mask; (void)B_group_prefix;
    (void)num_refined_sg; (void)scale2_ab; (void)enable_refine;
    (void)C; (void)M; (void)N; (void)K;
    __trap();
#endif
}

// =========================================================================
// autoematic_fp6_gemv_batch — Small-M (1..8) fused BF16-activation AutoFP6 GEMV
//
// Grid:  (ceil(N/4), 1, 1)
// Block: (256, 1, 1) — 4 output rows/block, 64 threads (2 warps) per row.
// =========================================================================
extern "C" __global__
__launch_bounds__(256, 4)
void autoematic_fp6_gemv_batch(
    const __nv_bfloat16* __restrict__ A,               // [M, K] BF16
    const unsigned char* __restrict__ B_packed,        // [N, K/2] E2M1
    const unsigned char* __restrict__ B_scale,         // [N, K/16] UE4M3
    const unsigned char* __restrict__ B_refine_codes,  // [N, num_refined_sg]
    const unsigned char* __restrict__ B_refine_mask,   // [K/16]
    const unsigned int*  __restrict__ B_group_prefix,  // [K/16]
    const unsigned int num_refined_sg,
    const float scale2,
    const unsigned int enable_refine,
    __nv_bfloat16* __restrict__ C,                     // [M, N] BF16
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    constexpr unsigned int N_PER_BLOCK = 4u;
    constexpr unsigned int THREADS_PER_OUT = 64u;
    constexpr unsigned int MAX_M = 8u;

    const unsigned int local_out = threadIdx.x / THREADS_PER_OUT;
    const unsigned int lane = threadIdx.x % THREADS_PER_OUT;
    const unsigned int warp_lane = lane % AE_WARP_SIZE;
    const unsigned int warp_idx = lane / AE_WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = (M > 0u) && (M <= MAX_M) && (n < N);

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2u * MAX_M];
    if (threadIdx.x < 16u) s_lut[threadIdx.x] = AE_E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc[MAX_M];
    #pragma unroll
    for (unsigned int r = 0u; r < MAX_M; ++r) acc[r] = 0.0f;

    if (valid) {
        const unsigned int half_K = K >> 1;
        const unsigned int num_groups = K >> 4;
        const unsigned char* b_row = B_packed + (unsigned long long)n * half_K;
        const unsigned char* s_row = B_scale + (unsigned long long)n * num_groups;
        const bool do_ref = (enable_refine != 0u) && (B_refine_codes != nullptr) &&
                            (B_refine_mask != nullptr) && (B_group_prefix != nullptr);
        const unsigned char* ref_row = do_ref
            ? (B_refine_codes + (unsigned long long)n * num_refined_sg)
            : nullptr;

        for (unsigned int k16 = lane; k16 < num_groups; k16 += THREADS_PER_OUT) {
            const unsigned long long packed8 = *(const unsigned long long*)(b_row + k16 * 8u);
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = (unsigned char)(s_row[k16] & 0x7Fu);
            const float scale = (float)fp8 * scale2;

            // Expand sparse refinement offsets for this 16-element group into 8 bytes (16 nibbles)
            unsigned long long ref_nibbles = 0ull;
            if (do_ref) {
                const unsigned int mask4 = (unsigned int)(B_refine_mask[k16] & 0x0Fu);
                if (mask4 != 0u) {
                    unsigned int sg_idx = B_group_prefix[k16];
                    #pragma unroll
                    for (unsigned int sg = 0u; sg < 4u; ++sg) {
                        if ((mask4 & (1u << sg)) != 0u) {
                            const unsigned short e4 = ae_expand_sg_to_e2m1x4(ref_row[sg_idx]);
                            ref_nibbles |= ((unsigned long long)e4) << (sg * 16u);
                            ++sg_idx;
                        }
                    }
                }
            }

            for (unsigned int row = 0u; row < M; ++row) {
                const __nv_bfloat16* A_row = A + (unsigned long long)row * K;
                const uint4 a_lo = __ldg(&((const uint4*)A_row)[k16 * 2u]);
                const uint4 a_hi = __ldg(&((const uint4*)A_row)[k16 * 2u + 1u]);
                const unsigned int a_raw[8] = {
                    a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                    a_hi.x, a_hi.y, a_hi.z, a_hi.w
                };

                float row_sum = 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; ++b) {
                    const unsigned int bv = (unsigned int)((packed8 >> (b * 8)) & 0xFFu);
                    const unsigned int rv = (unsigned int)((ref_nibbles >> (b * 8)) & 0xFFu);
                    const float w_lo = (s_lut[bv & 0xFu] + 0.25f * s_lut[rv & 0xFu]) * scale;
                    const float w_hi = (s_lut[bv >> 4]   + 0.25f * s_lut[rv >> 4])   * scale;

                    __nv_bfloat16 al, ah;
                    *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFFu);
                    *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
                    row_sum += __bfloat162float(al) * w_lo;
                    row_sum += __bfloat162float(ah) * w_hi;
                }
                acc[row] += row_sum;
            }
        }
    }

    #pragma unroll
    for (unsigned int r = 0u; r < MAX_M; ++r) {
        float val = acc[r];
        #pragma unroll
        for (int offset = AE_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
        }
        if (warp_lane == 0u) {
            smem[(local_out * 2u + warp_idx) * MAX_M + r] = val;
        }
    }
    __syncthreads();

    if (valid && lane == 0u) {
        for (unsigned int r = 0u; r < M; ++r) {
            const float sum = smem[(local_out * 2u) * MAX_M + r]
                            + smem[(local_out * 2u + 1u) * MAX_M + r];
            C[(unsigned long long)r * N + n] = __float2bfloat16(sum);
        }
    }
}
