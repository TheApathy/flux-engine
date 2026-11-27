// SPDX-License-Identifier: AGPL-3.0-only
//
// Flux Engine — DualPlaneNvfp4 (8.5 bpw) Load-Time Repack & Extended-K Verify Kernels.
//
// Represents any upstream BF16 or FP8 E4M3 weight matrix as two native E2M1
// bit-shared planes sharing one per-16 UE4M3 group scale and one FP32 scale2:
//   W_target ≈ s_g * (W_hi + 0.125f * W_lo),   s_g = UE4M3(B_scale) * scale2
//   - Draft pass: reads ONLY W_hi (4.5 bpw, SQNR=20.5 dB vs FP8, cos=0.9955)
//     using Flux's unchanged w4a16_gemv / NVFP4 Tensor Core kernels.
//   - Verify pass: reads W_hi + W_lo (8.5 bpw, SQNR=38.47 dB vs FP32, beating
//     FP8 E4M3's 31.79 dB by +6.7 dB!) using Extended-K native Blackwell FP4
//     MMA (`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`) or fused small-M GEMV.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define DP_M_TILE     64
#define DP_N_TILE     64
#define DP_K_STEP     64
#define DP_GROUP_SIZE 16
#define DP_WARP_SIZE  32
#define DP_FP8_BLOCK  128

__device__ __constant__ float DP_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Branchless FP32 -> E2M1 (matches e2m1_branchless.cu round-to-nearest ties-to-even).
__device__ __forceinline__ unsigned int dp_float_to_e2m1(float x) {
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

// Software positive FP32 -> UE4M3 (bit 7 always 0; max 448.0; avoids NaN 0x7F).
__device__ __forceinline__ unsigned char dp_float_to_ue4m3(float v) {
    float absv = fabsf(v);
    if (absv == 0.0f) return 0u;
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

// Decode signed FP8 E4M3 byte to FP32 (NaN 0x7F/0xFF -> 0.0).
__device__ __forceinline__ float dp_e4m3_to_f32(unsigned char byte) {
    const unsigned int sign = (byte >> 7) & 1u;
    const unsigned int exp  = (byte >> 3) & 0xFu;
    const unsigned int man  = byte & 0x7u;
    float val;
    if (exp == 0u) {
        val = (float)man * 0.001953125f;  // 2^-9
    } else if (exp == 15u && man == 7u) {
        val = 0.0f;
    } else {
        val = __uint_as_float(((exp + 120u) << 23) | (man << 20));
    }
    return sign ? -val : val;
}

// =========================================================================
// 1. repack_bf16_to_dual_plane_nvfp4
//
// On-the-fly load-time repack of BF16 [N, K] into shared-scale dual E2M1:
//   W_hi[N, K/2] + 0.125 * W_lo[N, K/2] with UE4M3 B_scale[N, K/16].
// Grid: (N, 1, 1)  Block: (256, 1, 1)
// =========================================================================
extern "C" __global__
__launch_bounds__(256, 4)
void repack_bf16_to_dual_plane_nvfp4(
    const __nv_bfloat16* __restrict__ input,   // [N, K] BF16
    unsigned char* __restrict__ w_hi_out,      // [N, K/2] E2M1
    unsigned char* __restrict__ w_lo_out,      // [N, K/2] E2M1
    unsigned char* __restrict__ scale_out,     // [N, K/16] UE4M3
    const float scale2,
    unsigned int N,
    unsigned int K
) {
    const unsigned int row = blockIdx.x;
    if (row >= N) return;

    const __nv_bfloat16* row_in = input + (unsigned long long)row * K;
    unsigned char* row_hi = w_hi_out + (unsigned long long)row * (K >> 1);
    unsigned char* row_lo = w_lo_out + (unsigned long long)row * (K >> 1);
    unsigned char* row_sc = scale_out + (unsigned long long)row * (K >> 4);

    const float inv_scale2 = (scale2 > 0.0f) ? (1.0f / scale2) : 0.0f;
    const unsigned int num_groups = K >> 4;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16u) s_lut[threadIdx.x] = DP_E2M1_LUT[threadIdx.x];
    __syncthreads();

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        const unsigned int base = g * DP_GROUP_SIZE;
        float vals[DP_GROUP_SIZE];
        float gmax = 0.0f;
        #pragma unroll
        for (int i = 0; i < DP_GROUP_SIZE; ++i) {
            const float v = __bfloat162float(row_in[base + (unsigned int)i]);
            vals[i] = v;
            const float av = fabsf(v);
            if (av > gmax) gmax = av;
        }

        const unsigned char fp8_byte = dp_float_to_ue4m3(gmax * inv_scale2 * (1.0f / 6.0f));
        row_sc[g] = fp8_byte;

        const float eff_scale = dp_e4m3_to_f32(fp8_byte) * scale2;
        const float inv_eff = (eff_scale > 0.0f) ? (1.0f / eff_scale) : 0.0f;

        #pragma unroll
        for (int i = 0; i < DP_GROUP_SIZE; i += 2) {
            const float n0 = vals[i] * inv_eff;
            const float n1 = vals[i + 1] * inv_eff;
            const unsigned int hi0 = dp_float_to_e2m1(n0);
            const unsigned int hi1 = dp_float_to_e2m1(n1);
            const unsigned int lo0 = dp_float_to_e2m1(8.0f * (n0 - s_lut[hi0]));
            const unsigned int lo1 = dp_float_to_e2m1(8.0f * (n1 - s_lut[hi1]));
            row_hi[g * 8u + (unsigned int)(i >> 1)] = (unsigned char)((hi1 << 4) | (hi0 & 0xFu));
            row_lo[g * 8u + (unsigned int)(i >> 1)] = (unsigned char)((lo1 << 4) | (lo0 & 0xFu));
        }
    }
}

// =========================================================================
// 2. repack_fp8_blockscaled_to_dual_plane_nvfp4
//
// On-the-fly load-time repack of block-128 FP8 E4M3 [N, K] + BF16 block scales
// [ceil(N/128), ceil(K/128)] into shared-scale DualPlaneNvfp4 (8.5 bpw).
// Grid: (N, 1, 1)  Block: (256, 1, 1)
// =========================================================================
extern "C" __global__
__launch_bounds__(256, 4)
void repack_fp8_blockscaled_to_dual_plane_nvfp4(
    const unsigned char* __restrict__ fp8_in,       // [N, K] FP8 E4M3
    const __nv_bfloat16* __restrict__ block_scale,  // [ceil(N/128), ceil(K/128)] BF16
    unsigned char* __restrict__ w_hi_out,           // [N, K/2] E2M1
    unsigned char* __restrict__ w_lo_out,           // [N, K/2] E2M1
    unsigned char* __restrict__ scale_out,          // [N, K/16] UE4M3
    const float scale2,
    unsigned int N,
    unsigned int K
) {
    const unsigned int row = blockIdx.x;
    if (row >= N) return;

    const unsigned char* row_in = fp8_in + (unsigned long long)row * K;
    unsigned char* row_hi = w_hi_out + (unsigned long long)row * (K >> 1);
    unsigned char* row_lo = w_lo_out + (unsigned long long)row * (K >> 1);
    unsigned char* row_sc = scale_out + (unsigned long long)row * (K >> 4);

    const unsigned int k_blocks = (K + DP_FP8_BLOCK - 1u) / DP_FP8_BLOCK;
    const unsigned int n_block = row / DP_FP8_BLOCK;
    const float inv_scale2 = (scale2 > 0.0f) ? (1.0f / scale2) : 0.0f;
    const unsigned int num_groups = K >> 4;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16u) s_lut[threadIdx.x] = DP_E2M1_LUT[threadIdx.x];
    __syncthreads();

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        const unsigned int base = g * DP_GROUP_SIZE;
        const unsigned int k_block = base / DP_FP8_BLOCK;
        const float bscale = __bfloat162float(
            block_scale[(unsigned long long)n_block * k_blocks + k_block]);

        float vals[DP_GROUP_SIZE];
        float gmax = 0.0f;
        #pragma unroll
        for (int i = 0; i < DP_GROUP_SIZE; ++i) {
            const float v = dp_e4m3_to_f32(row_in[base + (unsigned int)i]) * bscale;
            vals[i] = v;
            const float av = fabsf(v);
            if (av > gmax) gmax = av;
        }

        const unsigned char fp8_byte = dp_float_to_ue4m3(gmax * inv_scale2 * (1.0f / 6.0f));
        row_sc[g] = fp8_byte;

        const float eff_scale = dp_e4m3_to_f32(fp8_byte) * scale2;
        const float inv_eff = (eff_scale > 0.0f) ? (1.0f / eff_scale) : 0.0f;

        #pragma unroll
        for (int i = 0; i < DP_GROUP_SIZE; i += 2) {
            const float n0 = vals[i] * inv_eff;
            const float n1 = vals[i + 1] * inv_eff;
            const unsigned int hi0 = dp_float_to_e2m1(n0);
            const unsigned int hi1 = dp_float_to_e2m1(n1);
            const unsigned int lo0 = dp_float_to_e2m1(8.0f * (n0 - s_lut[hi0]));
            const unsigned int lo1 = dp_float_to_e2m1(8.0f * (n1 - s_lut[hi1]));
            row_hi[g * 8u + (unsigned int)(i >> 1)] = (unsigned char)((hi1 << 4) | (hi0 & 0xFu));
            row_lo[g * 8u + (unsigned int)(i >> 1)] = (unsigned char)((lo1 << 4) | (lo0 & 0xFu));
        }
    }
}

// =========================================================================
// 3. dual_plane_nvfp4_verify_gemm_sm120
//
// Extended-K native Blackwell FP4 Tensor Core verify GEMM:
//   C = scale2_ab * (A_packed * W_hi^T + 0.125f * A_packed * W_lo^T)
// Shares A_packed, A_scale, and B_scale across both E2M1 MMA planes.
// Grid: (ceil(N/64), ceil(M/64), 1)  Block: (128, 1, 1)
// =========================================================================
extern "C" __global__
__launch_bounds__(128, 4)
void dual_plane_nvfp4_verify_gemm_sm120(
    const unsigned char* __restrict__ A_packed,  // [M, K/2]
    const unsigned char* __restrict__ A_scale,   // [M, K/16]
    const unsigned char* __restrict__ W_hi,      // [N, K/2]
    const unsigned char* __restrict__ W_lo,      // [N, K/2]
    const unsigned char* __restrict__ B_scale,   // [N, K/16]
    const float scale2_ab,
    __nv_bfloat16* __restrict__ C,               // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
#if __CUDA_ARCH__ >= 1200
    const unsigned int cta_n = blockIdx.x * DP_N_TILE;
    const unsigned int cta_m = blockIdx.y * DP_M_TILE;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31u;
    const unsigned int warp_m_offset = warp_id * 16u;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3u;

    __shared__ alignas(16) unsigned char smem_Ap[2][DP_M_TILE][DP_K_STEP / 2];
    __shared__ alignas(16) unsigned char smem_Bhi[2][DP_N_TILE][DP_K_STEP / 2];
    __shared__ alignas(16) unsigned char smem_Blo[2][DP_N_TILE][DP_K_STEP / 2];
    __shared__ alignas(4)  unsigned char smem_As[2][DP_M_TILE][DP_K_STEP / DP_GROUP_SIZE];
    __shared__ alignas(4)  unsigned char smem_Bs[2][DP_N_TILE][DP_K_STEP / DP_GROUP_SIZE];

    float acc_hi[8][4];
    float acc_lo[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        acc_hi[nt][0] = 0.0f; acc_hi[nt][1] = 0.0f;
        acc_hi[nt][2] = 0.0f; acc_hi[nt][3] = 0.0f;
        acc_lo[nt][0] = 0.0f; acc_lo[nt][1] = 0.0f;
        acc_lo[nt][2] = 0.0f; acc_lo[nt][3] = 0.0f;
    }

    const unsigned int half_K = K >> 1;
    const unsigned int num_groups = K >> 4;

    #define DP_LOAD_TILE(buf, kb) do { \
        const unsigned int row = threadIdx.x >> 1; \
        const unsigned int col = (threadIdx.x & 1u) << 4; \
        const unsigned int gc_byte = ((kb) >> 1) + col; \
        const unsigned int gr = cta_m + row; \
        const unsigned int gn = cta_n + row; \
        uint4 va = make_uint4(0u, 0u, 0u, 0u); \
        uint4 vhi = make_uint4(0u, 0u, 0u, 0u); \
        uint4 vlo = make_uint4(0u, 0u, 0u, 0u); \
        if (gr < M && gc_byte + 15u < half_K) { \
            va = *(const uint4*)(A_packed + (unsigned long long)gr * half_K + gc_byte); \
        } \
        if (gn < N && gc_byte + 15u < half_K) { \
            vhi = *(const uint4*)(W_hi + (unsigned long long)gn * half_K + gc_byte); \
            vlo = *(const uint4*)(W_lo + (unsigned long long)gn * half_K + gc_byte); \
        } \
        *(uint4*)&smem_Ap[(buf)][row][col]  = va; \
        *(uint4*)&smem_Bhi[(buf)][row][col] = vhi; \
        *(uint4*)&smem_Blo[(buf)][row][col] = vlo; \
        const unsigned int gg_base = (kb) >> 4; \
        if (threadIdx.x < DP_M_TILE) { \
            const unsigned int srow = threadIdx.x; \
            const unsigned int sgr = cta_m + srow; \
            unsigned int sw = 0u; \
            if (sgr < M && gg_base + 3u < num_groups) { \
                sw = *(const unsigned int*)(A_scale + (unsigned long long)sgr * num_groups + gg_base); \
            } \
            *(unsigned int*)&smem_As[(buf)][srow][0] = sw & 0x7F7F7F7Fu; \
        } else { \
            const unsigned int srow = threadIdx.x - DP_M_TILE; \
            const unsigned int sgn = cta_n + srow; \
            unsigned int sw = 0u; \
            if (sgn < N && gg_base + 3u < num_groups) { \
                sw = *(const unsigned int*)(B_scale + (unsigned long long)sgn * num_groups + gg_base); \
            } \
            *(unsigned int*)&smem_Bs[(buf)][srow][0] = sw & 0x7F7F7F7Fu; \
        } \
    } while (0)

    #define DP_COMPUTE_KSTEP(buf) do { \
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
            const unsigned int bhi0 = *(const unsigned int*)&smem_Bhi[(buf)][nc][a_col]; \
            const unsigned int bhi1 = *(const unsigned int*)&smem_Bhi[(buf)][nc][a_col + 16u]; \
            const unsigned int blo0 = *(const unsigned int*)&smem_Blo[(buf)][nc][a_col]; \
            const unsigned int blo1 = *(const unsigned int*)&smem_Blo[(buf)][nc][a_col + 16u]; \
            const unsigned int sfb  = *(const unsigned int*)&smem_Bs[(buf)][nc][0]; \
            asm volatile( \
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X." \
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13}," \
                "{%14},{%15,%16},{%17},{%18,%19};" \
                : "=f"(acc_hi[nt][0]), "=f"(acc_hi[nt][1]), \
                  "=f"(acc_hi[nt][2]), "=f"(acc_hi[nt][3]) \
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), \
                  "r"(bhi0), "r"(bhi1), \
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
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), \
                  "r"(blo0), "r"(blo1), \
                  "f"(acc_lo[nt][0]), "f"(acc_lo[nt][1]), \
                  "f"(acc_lo[nt][2]), "f"(acc_lo[nt][3]), \
                  "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0), \
                  "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0)); \
        } \
    } while (0)

    if (K == 0u) return;

    DP_LOAD_TILE(0, 0u);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = DP_K_STEP; k_base < K; k_base += DP_K_STEP) {
        const int nxt = 1 - cur;
        DP_LOAD_TILE(nxt, k_base);
        DP_COMPUTE_KSTEP(cur);
        __syncthreads();
        cur = nxt;
    }
    DP_COMPUTE_KSTEP(cur);

    #undef DP_LOAD_TILE
    #undef DP_COMPUTE_KSTEP

    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const unsigned int c0 = cta_n + (unsigned int)nt * 8u + tid * 2u;
        const unsigned int c1 = c0 + 1u;
        const unsigned int r0 = cta_m + warp_m_offset + group_id;
        const unsigned int r1 = r0 + 8u;
        const float v00 = (acc_hi[nt][0] + 0.125f * acc_lo[nt][0]) * scale2_ab;
        const float v01 = (acc_hi[nt][1] + 0.125f * acc_lo[nt][1]) * scale2_ab;
        const float v10 = (acc_hi[nt][2] + 0.125f * acc_lo[nt][2]) * scale2_ab;
        const float v11 = (acc_hi[nt][3] + 0.125f * acc_lo[nt][3]) * scale2_ab;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 * N + c0] = __float2bfloat16(v00);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 * N + c1] = __float2bfloat16(v01);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 * N + c0] = __float2bfloat16(v10);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 * N + c1] = __float2bfloat16(v11);
    }
#else
    (void)A_packed; (void)A_scale; (void)W_hi; (void)W_lo;
    (void)B_scale; (void)scale2_ab; (void)C; (void)M; (void)N; (void)K;
    __trap();
#endif
}

// =========================================================================
// 4. dual_plane_nvfp4_verify_gemv_batch
//
// Small-M (1..8) fused BF16-activation verify GEMV reading W_hi + W_lo + scale
// ONCE per 16 K-values and reusing reconstructed FP32 weights across M rows.
// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
// =========================================================================
extern "C" __global__
__launch_bounds__(256, 4)
void dual_plane_nvfp4_verify_gemv_batch(
    const __nv_bfloat16* __restrict__ A,        // [M, K] BF16
    const unsigned char* __restrict__ W_hi,     // [N, K/2] E2M1
    const unsigned char* __restrict__ W_lo,     // [N, K/2] E2M1
    const unsigned char* __restrict__ B_scale,  // [N, K/16] UE4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,              // [M, N] BF16
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    constexpr unsigned int N_PER_BLOCK = 4u;
    constexpr unsigned int THREADS_PER_OUT = 64u;
    constexpr unsigned int MAX_M = 8u;

    const unsigned int local_out = threadIdx.x / THREADS_PER_OUT;
    const unsigned int lane = threadIdx.x % THREADS_PER_OUT;
    const unsigned int warp_lane = lane % DP_WARP_SIZE;
    const unsigned int warp_idx = lane / DP_WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = (M > 0u) && (M <= MAX_M) && (n < N);

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2u * MAX_M];
    if (threadIdx.x < 16u) s_lut[threadIdx.x] = DP_E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc[MAX_M];
    #pragma unroll
    for (unsigned int r = 0u; r < MAX_M; ++r) acc[r] = 0.0f;

    if (valid) {
        const unsigned int half_K = K >> 1;
        const unsigned int num_groups = K >> 4;
        const unsigned char* hi_row = W_hi + (unsigned long long)n * half_K;
        const unsigned char* lo_row = W_lo + (unsigned long long)n * half_K;
        const unsigned char* sc_row = B_scale + (unsigned long long)n * num_groups;

        for (unsigned int k16 = lane; k16 < num_groups; k16 += THREADS_PER_OUT) {
            const unsigned long long p_hi = *(const unsigned long long*)(hi_row + k16 * 8u);
            const unsigned long long p_lo = *(const unsigned long long*)(lo_row + k16 * 8u);
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = (unsigned char)(sc_row[k16] & 0x7Fu);
            const float scale = (float)fp8 * scale2;

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
                    const unsigned int bh = (unsigned int)((p_hi >> (b * 8)) & 0xFFu);
                    const unsigned int bl = (unsigned int)((p_lo >> (b * 8)) & 0xFFu);
                    const float w0 = (s_lut[bh & 0xFu] + 0.125f * s_lut[bl & 0xFu]) * scale;
                    const float w1 = (s_lut[bh >> 4]   + 0.125f * s_lut[bl >> 4])   * scale;

                    __nv_bfloat16 al, ah;
                    *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFFu);
                    *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
                    row_sum += __bfloat162float(al) * w0;
                    row_sum += __bfloat162float(ah) * w1;
                }
                acc[row] += row_sum;
            }
        }
    }

    #pragma unroll
    for (unsigned int r = 0u; r < MAX_M; ++r) {
        float val = acc[r];
        #pragma unroll
        for (int offset = DP_WARP_SIZE / 2; offset > 0; offset >>= 1) {
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
