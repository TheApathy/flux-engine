// SPDX-License-Identifier: AGPL-3.0-only

// Small-M row-scaled FP8-weight GEMM for DFlash / DSpark drafter projections.
//
// Computes:
//   C[m, n] = bf16((sum_{k=0}^{K-1} bf16_to_f32(A[m, k]) * fp8_e4m3(B[n, k])) * row_scale[n])
// where:
//   A:         [M, K] BF16 (row-major activations)
//   B:         [N, K] FP8 E4M3 bytes (row-major weights from quantize_bf16_to_fp8)
//   row_scale: [N] FP32 (per-row dequantization scales from quantize_bf16_to_fp8)
//   C:         [M, N] BF16 (row-major output)
//
// Optimized for M = 1..=16 (DFlash/DSpark block sizes W=7/8/16) and tiles
// arbitrarily across M via grid.y = div_ceil(M, 16) for prompt-context ingestion.
// Every finite FP8 E4M3 value is representable exactly in BF16, so dequantizing
// B tiles to BF16 in SMEM and accumulating via mma.sync.aligned.m16n8k16 in FP32
// preserves full FP8 weight fidelity with half the BF16 weight bandwidth.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define DFLASH_FP8_BM 16
#define DFLASH_FP8_BN 64
#define DFLASH_FP8_BK 32
#define DFLASH_FP8_PAD 8
#define DFLASH_FP8_STRIDE (DFLASH_FP8_BK + DFLASH_FP8_PAD)

__device__ __forceinline__ __nv_bfloat16 fp8_e4m3_byte_to_bf16(unsigned char byte) {
    __nv_fp8_e4m3 v;
    *reinterpret_cast<unsigned char*>(&v) = byte;
    return __float2bfloat16(static_cast<float>(v));
}

__device__ __forceinline__ unsigned int fp8_e4m3_pair_to_bf16x2_bits(unsigned int two_bytes) {
    __nv_fp8_e4m3 lo_fp8, hi_fp8;
    *reinterpret_cast<unsigned char*>(&lo_fp8) = static_cast<unsigned char>(two_bytes & 0xFFu);
    *reinterpret_cast<unsigned char*>(&hi_fp8) = static_cast<unsigned char>((two_bytes >> 8) & 0xFFu);
    __nv_bfloat162 packed = __floats2bfloat162_rn(
        static_cast<float>(lo_fp8),
        static_cast<float>(hi_fp8));
    return *reinterpret_cast<const unsigned int*>(&packed);
}

// Grid:  (div_ceil(N, 64), div_ceil(M, 16), 1)
// Block: (128, 1, 1) — 4 warps, each warp owns 16 output columns x 16 rows.
extern "C" __global__ void __launch_bounds__(128, 4) fp8_gemm_t_m16_n64_rowscale(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * DFLASH_FP8_BN;
    const unsigned int cta_m = blockIdx.y * DFLASH_FP8_BM;
    if (cta_m >= M || cta_n >= N) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31u;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int lane_pair = lane_id & 3u;

    __shared__ __align__(16) __nv_bfloat16 smem_A[DFLASH_FP8_BM][DFLASH_FP8_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 smem_B[DFLASH_FP8_BN][DFLASH_FP8_STRIDE];

    float acc[2][4];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        acc[nt][0] = 0.0f;
        acc[nt][1] = 0.0f;
        acc[nt][2] = 0.0f;
        acc[nt][3] = 0.0f;
    }

    const bool fast_k16 = ((K & 15u) == 0u)
        && ((((unsigned long long)A | (unsigned long long)B) & 15ull) == 0ull);
    const unsigned short* sA = reinterpret_cast<const unsigned short*>(&smem_A[0][0]);
    const unsigned short* sB = reinterpret_cast<const unsigned short*>(&smem_B[0][0]);

    for (unsigned int k_base = 0u; k_base < K; k_base += DFLASH_FP8_BK) {
        const bool full_k_tile = fast_k16 && (k_base + DFLASH_FP8_BK <= K);

        // 1. Load A tile [16, 32] BF16 -> smem_A.
        if (tid < 64u) {
            const unsigned int m_local = tid >> 2;
            const unsigned int k_local = (tid & 3u) << 3;
            const unsigned int gm = cta_m + m_local;
            if (full_k_tile) {
                uint4 v = (gm < M)
                    ? *reinterpret_cast<const uint4*>(
                          A + (unsigned long long)gm * K + k_base + k_local)
                    : make_uint4(0u, 0u, 0u, 0u);
                *reinterpret_cast<uint4*>(&smem_A[m_local][k_local]) = v;
            } else {
                #pragma unroll
                for (unsigned int j = 0u; j < 8u; ++j) {
                    const unsigned int gk = k_base + k_local + j;
                    smem_A[m_local][k_local + j] = (gm < M && gk < K)
                        ? A[(unsigned long long)gm * K + gk]
                        : __float2bfloat16(0.0f);
                }
            }
        }

        // 2. Load & dequantize B tile [64, 32] FP8 E4M3 -> smem_B [64, 32] BF16.
        {
            const unsigned int n_local = tid >> 1;
            const unsigned int k_local = (tid & 1u) << 4;
            const unsigned int gn = cta_n + n_local;
            if (full_k_tile) {
                if (gn < N) {
                    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(
                        B + (unsigned long long)gn * K + k_base + k_local));
                    uint4 out0;
                    out0.x = fp8_e4m3_pair_to_bf16x2_bits(raw.x);
                    out0.y = fp8_e4m3_pair_to_bf16x2_bits(raw.x >> 16);
                    out0.z = fp8_e4m3_pair_to_bf16x2_bits(raw.y);
                    out0.w = fp8_e4m3_pair_to_bf16x2_bits(raw.y >> 16);
                    uint4 out1;
                    out1.x = fp8_e4m3_pair_to_bf16x2_bits(raw.z);
                    out1.y = fp8_e4m3_pair_to_bf16x2_bits(raw.z >> 16);
                    out1.z = fp8_e4m3_pair_to_bf16x2_bits(raw.w);
                    out1.w = fp8_e4m3_pair_to_bf16x2_bits(raw.w >> 16);
                    *reinterpret_cast<uint4*>(&smem_B[n_local][k_local]) = out0;
                    *reinterpret_cast<uint4*>(&smem_B[n_local][k_local + 8u]) = out1;
                } else {
                    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
                    *reinterpret_cast<uint4*>(&smem_B[n_local][k_local]) = z;
                    *reinterpret_cast<uint4*>(&smem_B[n_local][k_local + 8u]) = z;
                }
            } else {
                #pragma unroll
                for (unsigned int j = 0u; j < 16u; ++j) {
                    const unsigned int gk = k_base + k_local + j;
                    smem_B[n_local][k_local + j] = (gn < N && gk < K)
                        ? fp8_e4m3_byte_to_bf16(__ldg(&B[(unsigned long long)gn * K + gk]))
                        : __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();

        // 3. Tensor Core MMA: two k16 slices per BK=32 tile.
        const unsigned int r0 = group_id;
        const unsigned int r1 = group_id + 8u;
        #pragma unroll
        for (unsigned int kk = 0u; kk < DFLASH_FP8_BK; kk += 16u) {
            const unsigned int k0 = kk + lane_pair * 2u;
            const unsigned int k1 = k0 + 8u;
            const unsigned int a0 = *reinterpret_cast<const unsigned int*>(&sA[r0 * DFLASH_FP8_STRIDE + k0]);
            const unsigned int a1 = *reinterpret_cast<const unsigned int*>(&sA[r1 * DFLASH_FP8_STRIDE + k0]);
            const unsigned int a2 = *reinterpret_cast<const unsigned int*>(&sA[r0 * DFLASH_FP8_STRIDE + k1]);
            const unsigned int a3 = *reinterpret_cast<const unsigned int*>(&sA[r1 * DFLASH_FP8_STRIDE + k1]);

            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const unsigned int n_col = warp_id * 16u + (unsigned int)nt * 8u + group_id;
                const unsigned int b0 =
                    *reinterpret_cast<const unsigned int*>(&sB[n_col * DFLASH_FP8_STRIDE + k0]);
                const unsigned int b1 =
                    *reinterpret_cast<const unsigned int*>(&sB[n_col * DFLASH_FP8_STRIDE + k1]);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%10, %11, %12, %13};"
                    : "=f"(acc[nt][0]), "=f"(acc[nt][1]),
                      "=f"(acc[nt][2]), "=f"(acc[nt][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1),
                      "f"(acc[nt][0]), "f"(acc[nt][1]),
                      "f"(acc[nt][2]), "f"(acc[nt][3])
                );
            }
        }

        __syncthreads();
    }

    // 4. Per-row scale epilogue + BF16 store.
    const unsigned int row0 = cta_m + group_id;
    const unsigned int row1 = row0 + 8u;
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const unsigned int col0 = cta_n + warp_id * 16u + (unsigned int)nt * 8u + lane_pair * 2u;
        const unsigned int col1 = col0 + 1u;
        const float s0 = (col0 < N) ? __ldg(&row_scale[col0]) : 0.0f;
        const float s1 = (col1 < N) ? __ldg(&row_scale[col1]) : 0.0f;
        if (row0 < M && col0 < N) {
            C[(unsigned long long)row0 * N + col0] = __float2bfloat16(acc[nt][0] * s0);
        }
        if (row0 < M && col1 < N) {
            C[(unsigned long long)row0 * N + col1] = __float2bfloat16(acc[nt][1] * s1);
        }
        if (row1 < M && col0 < N) {
            C[(unsigned long long)row1 * N + col0] = __float2bfloat16(acc[nt][2] * s0);
        }
        if (row1 < M && col1 < N) {
            C[(unsigned long long)row1 * N + col1] = __float2bfloat16(acc[nt][3] * s1);
        }
    }
}
