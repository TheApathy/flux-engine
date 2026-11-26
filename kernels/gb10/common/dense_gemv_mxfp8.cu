// SPDX-License-Identifier: AGPL-3.0-only

// Atlas MXFP8 block-32 GEMV (M=1 decode) for SM121 (GB10).
//
// out[n] = sum_k A[k] * W[n,k],  W[n,k] = bf16(E4M3(B[n,k]) * 2^(S[n,k/32]-127))
//
//   A: [1, K] BF16 activations
//   B: [N, K] E4M3 codes, exactly as the checkpoint ships them
//   S: [N, K/32] UE8M0 block-32 shared exponents, exactly as shipped
//   C: [1, N] BF16
//
// Every weight is formed in-register with the same operations as
// `dequant_mxfp8_bf16` (software E4M3 decode with NaN/-0 -> 0, `mul.rn`
// that keeps subnormal products under FTZ, truncation to BF16), so the
// weight a row sees is bit-identical to the exact MXFP8 -> BF16 dequant.
// Products accumulate in FP32; only the reduction order differs from a
// BF16 dense GEMV over the dequantized matrix.
//
// Layout and launch mirror dense_gemv_fp8w: 4 outputs per 256-thread block,
// 64 threads per output, 16 weights (one uint4) per lane iteration, so each
// iteration reads one UE8M0 byte (16 weights = half a 32-wide block).
// Requires K % 32 == 0, scale_cols == K / 32, A and every B row 16-byte aligned.
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
//
// Not the DeepSeek V4.1 kernels. dsv41_fp8_gemv_m1 / dsv41_fp8_gemv_m8
// (kernels/gb10/deepseek-v4.1/cb3/dsv41_decode.cu) and dsv41_fp8_gemm_nt_v2
// also decode E4M3 x 2^(ue8m0-127), but their scale grid is 2-D 32x32
// blocks, ue8m0 [ceil(N/32), ceil(K/32)] indexed by (n/32, k/32)
// (weight_loader/deepseek_v41/ops.rs Fp8Linear::load). Qwen4 MXFP8 ships one
// exponent per row per 32 columns, [N, K/32] indexed by (n, k/32); rows in a
// 32-row band carry different exponents, so a 32x32 grid cannot represent it
// exactly. Those kernels also live in the DSV4.1 model directory, which the
// Flash-Next target does not compile.

#include <cuda_bf16.h>

#define MX_BLOCK_SIZE 256
#define MX_N_PER_BLOCK 4
#define MX_WARP_SIZE 32
#define MX_VEC 16

__device__ __forceinline__ float mxgemv_e4m3_to_f32(unsigned int bits) {
    const unsigned int sign = (bits >> 7) & 1u;
    const unsigned int exponent = (bits >> 3) & 0x0fu;
    const unsigned int mantissa = bits & 0x07u;
    float value;
    if ((exponent == 0u && mantissa == 0u) || (exponent == 0x0fu && mantissa == 0x07u)) {
        value = 0.0f;
    } else if (exponent == 0u) {
        value = static_cast<float>(mantissa) * 0.001953125f;
    } else {
        value = __uint_as_float(((exponent + 120u) << 23) | (mantissa << 20));
    }
    return sign != 0u ? -value : value;
}

__device__ __forceinline__ float mxgemv_e8m0_to_f32(unsigned int exponent) {
    if (exponent == 255u) return 0.0f;
    if (exponent == 0u) return __uint_as_float(1u << 22);  // 2^-127
    return __uint_as_float(exponent << 23);
}

// Identical to dequant_mxfp8_bf16: exact product (no FTZ), then BF16 truncation.
__device__ __forceinline__ float mxgemv_weight(unsigned int code, float scale) {
    float product;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(product) : "f"(mxgemv_e4m3_to_f32(code)), "f"(scale));
    return __uint_as_float(__float_as_uint(product) & 0xffff0000u);
}

extern "C" __global__ void dense_gemv_mxfp8(
    const __nv_bfloat16* __restrict__ A,     // [1, K]
    const unsigned char* __restrict__ B,     // [N, K] E4M3
    const unsigned char* __restrict__ S,     // [N, K/32] UE8M0
    __nv_bfloat16* __restrict__ C,           // [1, N]
    unsigned int N,
    unsigned int K,
    unsigned int scale_cols
) {
    const unsigned int threads_per_out = MX_BLOCK_SIZE / MX_N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * MX_N_PER_BLOCK + local_out;
    const bool valid = n < N;

    float acc = 0.0f;
    if (valid) {
        const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);
        const unsigned char* S_row = S + (unsigned long long)n * scale_cols;
        const uint4* A_vec = (const uint4*)A;
        const unsigned int K_VEC = K / MX_VEC;
        for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
            const float scale = mxgemv_e8m0_to_f32(S_row[kv >> 1]);
            const uint4 b = B_vec[kv];
            const uint4 a0 = A_vec[kv * 2];
            const uint4 a1 = A_vec[kv * 2 + 1];
            const unsigned int bw[4] = {b.x, b.y, b.z, b.w};
            const unsigned int aw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            #pragma unroll
            for (int i = 0; i < MX_VEC; i++) {
                const unsigned int code = (bw[i >> 2] >> ((i & 3) * 8)) & 0xffu;
                const unsigned int abits = (aw[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
                acc += __uint_as_float(abits << 16) * mxgemv_weight(code, scale);
            }
        }
    }

    #pragma unroll
    for (int offset = MX_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    __shared__ float smem[MX_N_PER_BLOCK * 2];
    const unsigned int warp_lane = threadIdx.x % MX_WARP_SIZE;
    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / MX_WARP_SIZE)] = acc;
    }
    __syncthreads();
    if (valid && lane == 0) {
        C[n] = __float2bfloat16(smem[local_out * 2] + smem[local_out * 2 + 1]);
    }
}

// ---------------------------------------------------------------------------
// fused_v2: bit-identical rewrite of dense_gemv_mxfp8 (kept above as the
// reference). Three changes, none of which alters any rounded operation:
//
// (a) Fast weight decode. For a 16-weight chunk whose UE8M0 byte e is in
//     [10, 246] and whose codes contain no E4M3 NaN (0x7f/0xff), every weight
//     is E4M3(code) * 2^(e-127) with |E4M3| in {0} U [2^-9, 448]. The product
//     is then 0 or a NORMAL f32 in [2^-126, 1.75 * 2^127] with <= 4 significant
//     bits, so `mul.rn` is exact and the BF16 truncation is a no-op: the weight
//     equals the reference weight bit for bit (signed zeros included). The
//     E4M3 value comes from the hardware cvt.rn.f16x2.e4m3x2 (every E4M3 value
//     is exact in f16) and cvt.f32.f16 (exact). Chunks outside that range
//     (e <= 9: subnormal or flushed products; e >= 247: overflow; e == 255)
//     or holding a NaN code run the reference decode verbatim. The
//     activation-times-weight product and the running add are the SAME two
//     separately rounded f32 operations in the same element order as the
//     reference (this file builds with --fmad=false, no FTZ). The scale is
//     NOT factored out of the sum: hoisting 2^k out of a running f32 sum is
//     exact only while every partial sum and product stays normal in both
//     domains, which a per-element check would have to prove; the decode is
//     where the reference spends its instructions, so only it changes.
// (b) threads_per_out = 32 (TPO32): one warp per row emulates the reference's
//     64 lanes exactly. Lane l keeps two accumulators, for reference lanes l
//     and l+32, each fed the same chunk sequence kv = vlane + 64*i; each set
//     is reduced by the reference's shuffle tree and the row result is
//     s(lanes 0-31) + s(lanes 32-63), the reference's smem[0] + smem[1]
//     (an idle reference warp contributes +0.0, which this reproduces).
//     TPO64 is the reference mapping. 8 rows per block for TPO32, 4 for TPO64.
// (c) Fusion. dense_gemv_mxfp8_v2_* takes two row segments (qkv|z, k|v) in
//     one launch; each row's reduction is unchanged, so the bytes equal two
//     launches. dense_gemv_mxfp8_v2_gus_* computes gate and up rows of the
//     same index in one block and applies silu_mul_separate's exact
//     expression on the BF16-rounded results, writing act (and up) as the
//     gate GEMV + up GEMV + silu_mul_separate chain does.
// ---------------------------------------------------------------------------

#define MXV2_REF_LANES 64

__device__ __forceinline__ unsigned int mxv2_nan_bytes(unsigned int w) {
    const unsigned int t = w & 0x7f7f7f7fu;    // byte == 0x7f iff +1 sets bit 7
    return (t + 0x01010101u) & 0x80808080u;    // no carry crosses a byte
}

__device__ __forceinline__ void mxv2_e4m3x2(unsigned int pair, float& lo, float& hi) {
    unsigned int h2;
    const unsigned short p = (unsigned short)pair;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(p));
    const unsigned short hl = (unsigned short)(h2 & 0xffffu);
    const unsigned short hh = (unsigned short)(h2 >> 16);
    asm("cvt.f32.f16 %0, %1;" : "=f"(lo) : "h"(hl));
    asm("cvt.f32.f16 %0, %1;" : "=f"(hi) : "h"(hh));
}

// One 16-weight chunk added onto `acc` in the reference element order.
__device__ __forceinline__ float mxv2_chunk(
    float acc, const uint4 b, const uint4 a0, const uint4 a1, unsigned int e
) {
    const unsigned int bw[4] = {b.x, b.y, b.z, b.w};
    const unsigned int aw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const unsigned int nan = mxv2_nan_bytes(b.x) | mxv2_nan_bytes(b.y)
                           | mxv2_nan_bytes(b.z) | mxv2_nan_bytes(b.w);
    if (e >= 10u && e <= 246u && nan == 0u) {
        const float scale = __uint_as_float(e << 23);
        #pragma unroll
        for (int j = 0; j < MX_VEC / 2; j++) {
            float e0, e1;
            mxv2_e4m3x2((bw[j >> 1] >> ((j & 1) * 16)) & 0xffffu, e0, e1);
            const float w0 = e0 * scale;   // exact, == mxgemv_weight
            const float w1 = e1 * scale;
            acc += __uint_as_float((aw[j] & 0xffffu) << 16) * w0;
            acc += __uint_as_float(aw[j] & 0xffff0000u) * w1;
        }
    } else {
        const float scale = mxgemv_e8m0_to_f32(e);
        #pragma unroll
        for (int i = 0; i < MX_VEC; i++) {
            const unsigned int code = (bw[i >> 2] >> ((i & 3) * 8)) & 0xffu;
            const unsigned int abits = (aw[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
            acc += __uint_as_float(abits << 16) * mxgemv_weight(code, scale);
        }
    }
    return acc;
}

__device__ __forceinline__ float mxv2_warp_sum(float acc) {
    #pragma unroll
    for (int offset = MX_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    return acc;
}

// Row result (valid on the row's thread 0) for TPO in {32, 64}. Every thread
// of the block must call it (TPO64 synchronises the block).
template <int TPO>
__device__ __forceinline__ float mxv2_row(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_row,
    const unsigned char* __restrict__ S_row,
    bool valid,
    unsigned int K,
    float* smem   // [256 / TPO * 2], TPO64 only
) {
    const unsigned int lane = threadIdx.x % TPO;
    const unsigned int K_VEC = K / MX_VEC;
    const uint4* A_vec = (const uint4*)A;
    const uint4* B_vec = (const uint4*)B_row;
    if (TPO == 32) {
        float acc0 = 0.0f, acc1 = 0.0f;
        if (valid) {
            #pragma unroll 2
            for (unsigned int kv = lane; kv < K_VEC; kv += MXV2_REF_LANES) {
                acc0 = mxv2_chunk(acc0, B_vec[kv], A_vec[kv * 2], A_vec[kv * 2 + 1], S_row[kv >> 1]);
                const unsigned int kv1 = kv + MX_WARP_SIZE;
                if (kv1 < K_VEC) {
                    acc1 = mxv2_chunk(acc1, B_vec[kv1], A_vec[kv1 * 2], A_vec[kv1 * 2 + 1], S_row[kv1 >> 1]);
                }
            }
        }
        const float s0 = mxv2_warp_sum(acc0);
        const float s1 = mxv2_warp_sum(acc1);
        return s0 + s1;
    } else {
        float acc = 0.0f;
        if (valid) {
            #pragma unroll 2
            for (unsigned int kv = lane; kv < K_VEC; kv += MXV2_REF_LANES) {
                acc = mxv2_chunk(acc, B_vec[kv], A_vec[kv * 2], A_vec[kv * 2 + 1], S_row[kv >> 1]);
            }
        }
        acc = mxv2_warp_sum(acc);
        const unsigned int local_out = threadIdx.x / TPO;
        if (threadIdx.x % MX_WARP_SIZE == 0) {
            smem[local_out * 2 + (lane / MX_WARP_SIZE)] = acc;
        }
        __syncthreads();
        return smem[local_out * 2] + smem[local_out * 2 + 1];
    }
}

// Two row segments in one launch: rows [0, N0) from (B0, S0) into C0, rows
// [N0, N0+N1) from (B1, S1) into C1. N1 may be 0.
// Grid: (ceil((N0+N1) / (256/TPO)), 1, 1)   Block: (256, 1, 1)
template <int TPO>
__device__ __forceinline__ void mxv2_seg(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B0, const unsigned char* __restrict__ S0,
    __nv_bfloat16* __restrict__ C0, unsigned int N0,
    const unsigned char* __restrict__ B1, const unsigned char* __restrict__ S1,
    __nv_bfloat16* __restrict__ C1, unsigned int N1,
    unsigned int K, unsigned int scale_cols
) {
    __shared__ float smem[(MX_BLOCK_SIZE / TPO) * 2];
    const unsigned int rows_per_block = MX_BLOCK_SIZE / TPO;
    const unsigned int r = blockIdx.x * rows_per_block + threadIdx.x / TPO;
    const bool first = r < N0;
    const bool valid = first || (r - N0) < N1;
    const unsigned int row = first ? r : r - N0;
    const unsigned char* B = first ? B0 : B1;
    const unsigned char* S = first ? S0 : S1;
    const unsigned long long safe_row = valid ? row : 0ull;
    const float sum = mxv2_row<TPO>(
        A, B + safe_row * K, S + safe_row * scale_cols, valid, K, smem);
    if (valid && threadIdx.x % TPO == 0) {
        (first ? C0 : C1)[row] = __float2bfloat16(sum);
    }
}

#define MXV2_SEG_KERNEL(NAME, TPO)                                                  \
extern "C" __global__ void __launch_bounds__(MX_BLOCK_SIZE) NAME(                   \
    const __nv_bfloat16* __restrict__ A,                                            \
    const unsigned char* __restrict__ B0, const unsigned char* __restrict__ S0,     \
    __nv_bfloat16* __restrict__ C0, unsigned int N0,                                \
    const unsigned char* __restrict__ B1, const unsigned char* __restrict__ S1,     \
    __nv_bfloat16* __restrict__ C1, unsigned int N1,                                \
    unsigned int K, unsigned int scale_cols) {                                      \
    mxv2_seg<TPO>(A, B0, S0, C0, N0, B1, S1, C1, N1, K, scale_cols);                \
}
MXV2_SEG_KERNEL(dense_gemv_mxfp8_v2_t32, 32)
MXV2_SEG_KERNEL(dense_gemv_mxfp8_v2_t64, 64)

// act[n] = bf16(silu(g) * u), g = bf16(gate row n), u = bf16(up row n); also
// writes up_out[n] = u. Block = (256/TPO)/2 gate rows then as many up rows.
// Grid: (ceil(N / ((256/TPO)/2)), 1, 1)   Block: (256, 1, 1)
template <int TPO>
__device__ __forceinline__ void mxv2_gus(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bg, const unsigned char* __restrict__ Sg,
    const unsigned char* __restrict__ Bu, const unsigned char* __restrict__ Su,
    __nv_bfloat16* __restrict__ act, __nv_bfloat16* __restrict__ up_out,
    unsigned int N, unsigned int K, unsigned int scale_cols
) {
    __shared__ float smem[(MX_BLOCK_SIZE / TPO) * 2];
    __shared__ float sums[MX_BLOCK_SIZE / TPO];
    const unsigned int half = (MX_BLOCK_SIZE / TPO) / 2;
    const unsigned int local_out = threadIdx.x / TPO;
    const bool is_up = local_out >= half;
    const unsigned int n = blockIdx.x * half + (is_up ? local_out - half : local_out);
    const bool valid = n < N;
    const unsigned long long safe_n = valid ? n : 0ull;
    const float sum = mxv2_row<TPO>(
        A, (is_up ? Bu : Bg) + safe_n * K, (is_up ? Su : Sg) + safe_n * scale_cols,
        valid, K, smem);
    if (threadIdx.x % TPO == 0) {
        sums[local_out] = sum;
    }
    __syncthreads();
    if (threadIdx.x < half) {
        const unsigned int m = blockIdx.x * half + threadIdx.x;
        if (m < N) {
            const __nv_bfloat16 g_bf = __float2bfloat16(sums[threadIdx.x]);
            const __nv_bfloat16 u_bf = __float2bfloat16(sums[half + threadIdx.x]);
            up_out[m] = u_bf;
            // silu_mul_separate (residual_add.cu), verbatim.
            float g = __bfloat162float(g_bf);
            float u = __bfloat162float(u_bf);
            float silu_g = g / (1.0f + __expf(-g));
            act[m] = __float2bfloat16(silu_g * u);
        }
    }
}

#define MXV2_GUS_KERNEL(NAME, TPO)                                                  \
extern "C" __global__ void __launch_bounds__(MX_BLOCK_SIZE) NAME(                   \
    const __nv_bfloat16* __restrict__ A,                                            \
    const unsigned char* __restrict__ Bg, const unsigned char* __restrict__ Sg,     \
    const unsigned char* __restrict__ Bu, const unsigned char* __restrict__ Su,     \
    __nv_bfloat16* __restrict__ act, __nv_bfloat16* __restrict__ up_out,            \
    unsigned int N, unsigned int K, unsigned int scale_cols) {                      \
    mxv2_gus<TPO>(A, Bg, Sg, Bu, Su, act, up_out, N, K, scale_cols);                \
}
MXV2_GUS_KERNEL(dense_gemv_mxfp8_v2_gus_t32, 32)
MXV2_GUS_KERNEL(dense_gemv_mxfp8_v2_gus_t64, 64)
