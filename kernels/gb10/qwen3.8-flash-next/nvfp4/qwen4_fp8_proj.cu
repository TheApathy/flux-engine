// SPDX-License-Identifier: AGPL-3.0-only
//
// FP8 (e4m3) operands for the Flash-Next fast prefill projections
// (`ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8`, default off): cuBLASLt e4m3 x e4m3 ->
// BF16 with per-output-channel weight scales and per-token activation scales
// (CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F). Each row is scaled so its
// absolute maximum maps to 448.
//
// Weights: the NVFP4 value LUT[nibble] * fp8 block scale (exact in BF16) is
// re-rounded to e4m3 (<= 2^-4 relative per element); the per-tensor scale2
// stays the GEMM alpha. Activations: BF16 -> e4m3 (<= 2^-4 relative).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

__device__ __constant__ float QWEN4_FP8_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float block_max(float v, float* sm) {
    for (int d = 16; d; d >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, d));
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float w = threadIdx.x < (blockDim.x >> 5) ? sm[threadIdx.x] : 0.0f;
        for (int d = 16; d; d >>= 1) w = fmaxf(w, __shfl_xor_sync(0xffffffff, w, d));
        if (threadIdx.x == 0) sm[0] = w;
    }
    __syncthreads();
    const float r = sm[0];
    __syncthreads();
    return r;
}

// One block (256 threads) per NVFP4 weight row n. Optional gated-Q row
// permutation identical to w4a16_dequant_bf16 (qg_heads != 0).
extern "C" __global__ void qwen4_fp8_weight_rows(
    const unsigned char* __restrict__ packed,   // [N, K/2]
    const unsigned char* __restrict__ scales,   // [N, K/16] e4m3
    __nv_fp8_e4m3* __restrict__ out,            // [N, K] (permuted rows)
    float* __restrict__ row_scale,              // [N] (permuted rows)
    unsigned int K,
    unsigned int qg_heads,
    unsigned int qg_head_dim) {
    __shared__ float sm[32];
    const unsigned long long n = blockIdx.x;
    const unsigned int groups = K / 16u;
    const unsigned char* srow = scales + n * groups;
    float m = 0.0f;
    for (unsigned int g = threadIdx.x; g < groups; g += blockDim.x) {
        __nv_fp8_e4m3 f;
        *reinterpret_cast<unsigned char*>(&f) = srow[g];
        m = fmaxf(m, fabsf((float)f));
    }
    const float amax = fmaxf(block_max(m, sm) * 6.0f, 1e-30f);
    const float inv = 448.0f / amax;
    unsigned long long out_row = n;
    if (qg_heads != 0u) {
        const unsigned long long group_dim = 2ull * qg_head_dim;
        const unsigned long long h = n / group_dim;
        const unsigned long long idx = n - h * group_dim;
        out_row = idx < qg_head_dim
            ? h * qg_head_dim + idx
            : (unsigned long long)qg_heads * qg_head_dim + h * qg_head_dim + (idx - qg_head_dim);
    }
    const unsigned char* prow = packed + n * (K / 2u);
    __nv_fp8_e4m3* orow = out + out_row * K;
    for (unsigned int g = threadIdx.x; g < groups; g += blockDim.x) {
        __nv_fp8_e4m3 f;
        *reinterpret_cast<unsigned char*>(&f) = srow[g];
        const float s = (float)f * inv;
        const unsigned long long p8 = *reinterpret_cast<const unsigned long long*>(prow + g * 8u);
        uint32_t words[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t word = 0;
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const unsigned char byte_val = (unsigned char)(p8 >> ((w * 2 + b) * 8));
                const __nv_fp8_e4m3 lo(QWEN4_FP8_E2M1_LUT[byte_val & 0xF] * s);
                const __nv_fp8_e4m3 hi(QWEN4_FP8_E2M1_LUT[byte_val >> 4] * s);
                word |= ((uint32_t)*reinterpret_cast<const unsigned char*>(&lo)) << (b * 16);
                word |= ((uint32_t)*reinterpret_cast<const unsigned char*>(&hi)) << (b * 16 + 8);
            }
            words[w] = word;
        }
        *reinterpret_cast<uint4*>(orow + g * 16u) = make_uint4(words[0], words[1], words[2], words[3]);
    }
    if (threadIdx.x == 0) row_scale[out_row] = amax / 448.0f;
}

// One block (256 threads) per activation row: per-token absmax scaling.
extern "C" __global__ void qwen4_fp8_act_rows(
    const __nv_bfloat16* __restrict__ x,  // [M, lda]
    __nv_fp8_e4m3* __restrict__ out,      // [M, K]
    float* __restrict__ row_scale,        // [M]
    unsigned int K,
    unsigned int lda) {
    __shared__ float sm[32];
    const unsigned long long r = blockIdx.x;
    const uint4* xr = reinterpret_cast<const uint4*>(x + r * lda);
    const unsigned int vecs = K / 8u;
    float m = 0.0f;
    for (unsigned int i = threadIdx.x; i < vecs; i += blockDim.x) {
        const uint4 v = xr[i];
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(p[j]);
            m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    const float amax = fmaxf(block_max(m, sm), 1e-30f);
    const float inv = 448.0f / amax;
    uint2* orow = reinterpret_cast<uint2*>(out + r * K);
    for (unsigned int i = threadIdx.x; i < vecs; i += blockDim.x) {
        const uint4 v = xr[i];
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
        uint32_t words[2] = {0, 0};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(p[j]);
            const __nv_fp8_e4m3 a(f.x * inv);
            const __nv_fp8_e4m3 b(f.y * inv);
            const uint32_t pair = (uint32_t)*reinterpret_cast<const unsigned char*>(&a) |
                                  ((uint32_t)*reinterpret_cast<const unsigned char*>(&b) << 8);
            words[j >> 1] |= pair << ((j & 1) * 16);
        }
        orow[i] = make_uint2(words[0], words[1]);
    }
    if (threadIdx.x == 0) row_scale[r] = amax / 448.0f;
}

// ---------------------------------------------------------------------------
// Per-tensor (SCALAR_32F) fallback for parts where cuBLASLt rejects
// OUTER_VEC_32F: one absmax per tensor (atomicMax on the float bits, valid
// because the values are non-negative), then the same row kernels' math with a
// uniform scale. `amax_bits` must be zeroed before the reduction.
extern "C" __global__ void qwen4_fp8_weight_amax(
    const unsigned char* __restrict__ scales,   // [N, K/16] e4m3 block scales
    unsigned long long count,
    unsigned int* __restrict__ amax_bits) {
    __shared__ float sm[32];
    float m = 0.0f;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        __nv_fp8_e4m3 f;
        *reinterpret_cast<unsigned char*>(&f) = scales[i];
        m = fmaxf(m, fabsf((float)f));
    }
    m = block_max(m, sm) * 6.0f;
    if (threadIdx.x == 0) atomicMax(amax_bits, __float_as_uint(m));
}

extern "C" __global__ void qwen4_fp8_act_amax(
    const __nv_bfloat16* __restrict__ x, unsigned long long count, unsigned int* __restrict__ amax_bits) {
    __shared__ float sm[32];
    float m = 0.0f;
    const uint4* v = reinterpret_cast<const uint4*>(x);
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < count / 8;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const uint4 w = v[i];
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&w);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(p[j]);
            m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    m = block_max(m, sm);
    if (threadIdx.x == 0) atomicMax(amax_bits, __float_as_uint(m));
}

// Build the 8-entry E4M3 magnitude table for one 16-element NVFP4 group scale `sv`.
// Byte j of {t0, t1} = cvt.rn.satfinite.e4m3(QWEN4_FP8_E2M1_LUT[j] * sv), j = 0..7.
// Bit-for-bit identical to per-element `(__nv_fp8_e4m3)(QWEN4_FP8_E2M1_LUT[nib] * sv)`.
// Uses .reg .b16 + mov.b32 {p01, p23} to avoid 4 redundant 16->32 zero-extension
// LOP3.LUT 0xffff instructions per group.
__device__ __forceinline__ void fp8_build_e4m3_table(float sv, uint32_t& t0, uint32_t& t1) {
    asm("{\n"
        "  .reg .b16 p01, p23, p45, p67;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 p01, %2, %3;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 p23, %4, %5;\n"
        "  mov.b32 %0, {p01, p23};\n"
        "  cvt.rn.satfinite.e4m3x2.f32 p45, %6, %7;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 p67, %8, %9;\n"
        "  mov.b32 %1, {p45, p67};\n"
        "}\n"
        : "=r"(t0), "=r"(t1)
        : "f"(0.5f * sv), "f"(0.0f * sv),
          "f"(1.5f * sv), "f"(1.0f * sv),
          "f"(3.0f * sv), "f"(2.0f * sv),
          "f"(6.0f * sv), "f"(4.0f * sv));
}

// Convert 4 packed NVFP4 bytes (8 E2M1 nibbles in K-ascending order) into 8 E4M3 bytes
// in K-ascending order via register PRMT byte-select + sign XOR (0 constant-memory LDC).
__device__ __forceinline__ uint2 fp8_prmt_dequant8(uint32_t w, uint32_t t0, uint32_t t1) {
    const uint32_t mag = w & 0x77777777u;
    const uint32_t sgn = (w >> 3) & 0x11111111u;
    const uint32_t kSignLut = 0x00008000u;
    uint32_t mag_lo, mag_hi, sgn_lo, sgn_hi;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(mag_lo) : "r"(t0), "r"(t1), "r"(mag));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(mag_hi) : "r"(t0), "r"(t1), "r"(mag >> 16));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(sgn_lo) : "r"(kSignLut), "r"(0u), "r"(sgn));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(sgn_hi) : "r"(kSignLut), "r"(0u), "r"(sgn >> 16));
    return make_uint2(mag_lo ^ sgn_lo, mag_hi ^ sgn_hi);
}

__device__ __forceinline__ uint4 fp8_dequant_group16(uint2 p8, unsigned char scale_byte, float inv) {
    __nv_fp8_e4m3 f;
    *reinterpret_cast<unsigned char*>(&f) = scale_byte;
    const float s = (float)f * inv;
    uint32_t t0, t1;
    fp8_build_e4m3_table(s, t0, t1);
    const uint2 lo8 = fp8_prmt_dequant8(p8.x, t0, t1);
    const uint2 hi8 = fp8_prmt_dequant8(p8.y, t0, t1);
    return make_uint4(lo8.x, lo8.y, hi8.x, hi8.y);
}

__device__ __forceinline__ float fp8_div_448(float amax) {
    float q;
    asm("div.approx.ftz.f32 %0, 0f43e00000, %1;" : "=f"(q) : "f"(amax));
    return q;
}

// 8 weight rows per CTA (`grid = [N / 8, 1, 1]`, `block = [256, 1, 1]`), uniform scale from
// amax_bits[0]; writes the tensor scale (amax/448) to scale_out[0].
// For K=2560 (groups=160), 8*160 = 1280 = 5*256 exact full-CTA iterations (0% idle threads);
// for K=6144 (groups=384), 8*384 = 3072 = 12*256 exact full-CTA iterations (0% idle threads).
extern "C" __global__ void __launch_bounds__(256) qwen4_fp8_weight_rows_tensor(
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales,
    const unsigned int* __restrict__ amax_bits,
    __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ scale_out,
    unsigned int K,
    unsigned int qg_heads,
    unsigned int qg_head_dim) {
    const unsigned int n0 = blockIdx.x * 8u;
    const unsigned int groups = K >> 4;
    const float amax = fmaxf(__uint_as_float(amax_bits[0]), 1e-30f);
    const float inv = fp8_div_448(amax);
    unsigned int out_row0 = n0;
    if (qg_heads != 0u) {
        const unsigned int group_dim = 2u * qg_head_dim;
        const unsigned int h = n0 / group_dim;
        const unsigned int idx = n0 - h * group_dim;
        out_row0 = idx < qg_head_dim
            ? h * qg_head_dim + idx
            : qg_heads * qg_head_dim + h * qg_head_dim + (idx - qg_head_dim);
    }
    uint64_t pol_ef;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));
    const unsigned char* srow = scales + (unsigned long long)n0 * groups;
    const uint2* prow = reinterpret_cast<const uint2*>(packed + (unsigned long long)n0 * (K >> 1));
    uint4* orow = reinterpret_cast<uint4*>(out + (unsigned long long)out_row0 * K);

    if (K == 2560u) {
        uint2 p8[5];
        unsigned char sb[5];
#pragma unroll
        for (int u = 0; u < 5; ++u) {
            const unsigned int g = threadIdx.x + (unsigned int)(u * 256);
            unsigned int s_u32;
            asm volatile("ld.global.nc.L2::cache_hint.u8 %0, [%1], %2;\n"
                         : "=r"(s_u32) : "l"(srow + g), "l"(pol_ef));
            sb[u] = (unsigned char)s_u32;
            asm volatile("ld.global.nc.L2::cache_hint.v2.u32 {%0, %1}, [%2], %3;\n"
                         : "=r"(p8[u].x), "=r"(p8[u].y) : "l"(prow + g), "l"(pol_ef));
        }
#pragma unroll
        for (int u = 0; u < 5; ++u) {
            const unsigned int g = threadIdx.x + (unsigned int)(u * 256);
            orow[g] = fp8_dequant_group16(p8[u], sb[u], inv);
        }
    } else if (K == 6144u) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
            uint2 p8[4];
            unsigned char sb[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const unsigned int g = threadIdx.x + (unsigned int)((4 * b + u) * 256);
                unsigned int s_u32;
                asm volatile("ld.global.nc.L2::cache_hint.u8 %0, [%1], %2;\n"
                             : "=r"(s_u32) : "l"(srow + g), "l"(pol_ef));
                sb[u] = (unsigned char)s_u32;
                asm volatile("ld.global.nc.L2::cache_hint.v2.u32 {%0, %1}, [%2], %3;\n"
                             : "=r"(p8[u].x), "=r"(p8[u].y) : "l"(prow + g), "l"(pol_ef));
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const unsigned int g = threadIdx.x + (unsigned int)((4 * b + u) * 256);
                orow[g] = fp8_dequant_group16(p8[u], sb[u], inv);
            }
        }
    } else {
        const unsigned int total = 8u * groups;
#pragma unroll 4
        for (unsigned int g = threadIdx.x; g < total; g += 256u) {
            unsigned int s_u32;
            uint2 p8;
            asm volatile("ld.global.nc.L2::cache_hint.u8 %0, [%1], %2;\n"
                         : "=r"(s_u32) : "l"(srow + g), "l"(pol_ef));
            asm volatile("ld.global.nc.L2::cache_hint.v2.u32 {%0, %1}, [%2], %3;\n"
                         : "=r"(p8.x), "=r"(p8.y) : "l"(prow + g), "l"(pol_ef));
            orow[g] = fp8_dequant_group16(p8, (unsigned char)s_u32, inv);
        }
    }
    if (blockIdx.x == 0u && threadIdx.x == 0u) scale_out[0] = amax * (1.0f / 448.0f);
}

// Activation rows with the tensor scale from amax_bits[0].
extern "C" __global__ void qwen4_fp8_act_rows_tensor(
    const __nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ amax_bits,
    __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ scale_out,
    unsigned int K,
    unsigned int lda) {
    const unsigned long long r = blockIdx.x;
    const float amax = fmaxf(__uint_as_float(amax_bits[0]), 1e-30f);
    const float inv = fp8_div_448(amax);
    const uint4* xr = reinterpret_cast<const uint4*>(x + r * lda);
    uint2* orow = reinterpret_cast<uint2*>(out + r * K);
    for (unsigned int i = threadIdx.x; i < K / 8u; i += blockDim.x) {
        const uint4 v = xr[i];
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
        uint32_t words[2] = {0, 0};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(p[j]);
            const __nv_fp8_e4m3 a(f.x * inv);
            const __nv_fp8_e4m3 b(f.y * inv);
            const uint32_t pair = (uint32_t)*reinterpret_cast<const unsigned char*>(&a) |
                                  ((uint32_t)*reinterpret_cast<const unsigned char*>(&b) << 8);
            words[j >> 1] |= pair << ((j & 1) * 16);
        }
        orow[i] = make_uint2(words[0], words[1]);
    }
    if (r == 0 && threadIdx.x == 0) scale_out[0] = amax * (1.0f / 448.0f);
}

__device__ __forceinline__ uint2 fp8_quant_vec8(uint4 v, float inv) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
    const float2 f0 = __bfloat1622float2(p[0]);
    const float2 f1 = __bfloat1622float2(p[1]);
    const float2 f2 = __bfloat1622float2(p[2]);
    const float2 f3 = __bfloat1622float2(p[3]);
    uint32_t w0, w1;
    asm("{\n"
        "  .reg .b16 q0, q1, q2, q3;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 q0, %2, %3;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 q1, %4, %5;\n"
        "  mov.b32 %0, {q0, q1};\n"
        "  cvt.rn.satfinite.e4m3x2.f32 q2, %6, %7;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 q3, %8, %9;\n"
        "  mov.b32 %1, {q2, q3};\n"
        "}\n"
        : "=r"(w0), "=r"(w1)
        : "f"(f0.y * inv), "f"(f0.x * inv),
          "f"(f1.y * inv), "f"(f1.x * inv),
          "f"(f2.y * inv), "f"(f2.x * inv),
          "f"(f3.y * inv), "f"(f3.x * inv));
    return make_uint2(w0, w1);
}

// Branchless 0-ULP fast-math reciprocal of (1.0f + __expf(-x)) without CALL.REL.NOINC.
__device__ __forceinline__ float fp8_fast_rcp_rn_1p_exp(float x) {
    float y;
    asm("{\n\t"
        ".reg .f32 z, t, den;\n\t"
        "mul.rn.ftz.f32 z, %1, 0fbfb8aa3b;\n\t"
        "ex2.approx.ftz.f32 t, z;\n\t"
        "add.rn.ftz.f32 den, t, 0f3f800000;\n\t"
        "rcp.approx.ftz.f32 %0, den;\n\t"
        "}"
        : "=f"(y) : "f"(x));
    return y;
}

// 4-row-per-CTA twin of qwen4_fp8_act_rows_tensor (`ATLAS_QWEN4_PREFILL_FP8_ACT_ROWS_V2`):
// for K=2560 (k8=320), 4*320 = 1280 = 5*256 exact full-CTA iterations with 0 idle warps
// (vs 192/256 = 75% idle threads on iteration 1 in the 1-row-per-CTA v1 kernel); also exact
// full-CTA iterations for K=4096 (4*512 = 8*256) and K=6144 (4*768 = 12*256).
// Uses L2::evict_first on BF16 input loads (`x`) and L2::evict_last on FP8 output stores (`out`)
// so the quantized E4M3 activation tensor stays resident in L2 for cuBLASLt FP8 GEMMs.
// DESTRUCTIVE INPUT CONTRACT: when `K == 6144 && lda == K` and `x` is 128-byte aligned,
// `discard.global.L2` drops dirty L2 lines of `x` without writeback after quantization;
// callers MUST treat `x` as dead scratch memory after launch.
// Grid: (ceil(rows / 4), 1, 1), Block: 256. 100% bit-for-bit identical to v1 in `out`/`scale_out`.
extern "C" __global__ void __launch_bounds__(256) qwen4_fp8_act_rows_tensor_v2(
    const __nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ amax_bits,
    __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ scale_out,
    unsigned int K,
    unsigned int lda,
    unsigned int rows) {
    const float amax = fmaxf(__uint_as_float(amax_bits[0]), 1e-30f);
    const float inv = fp8_div_448(amax);
    if (blockIdx.x == 0 && threadIdx.x == 0) scale_out[0] = amax * (1.0f / 448.0f);
    // For K>=4096 (o_proj K=4096 after qsa_core, or K=6144), reverse CTA order consumes hot high rows first.
    // For K=2560 (qkv_proj after upmix_roof, which writes rows in reverse order down to 0),
    // forward CTA order consumes the hot low rows from L2 first.
    const unsigned int cta_idx = (K >= 4096u) ? (gridDim.x - 1u - blockIdx.x) : blockIdx.x;
    const unsigned long long r0 = (unsigned long long)cta_idx * 4ull;
    if (r0 >= rows) return;
    const unsigned int k8 = K >> 3;
    const unsigned int rem = (unsigned int)((rows - r0) < 4ull ? (rows - r0) : 4ull);
    const unsigned int total = rem * k8;
    uint64_t pol_ef, pol_el;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));
    if (r0 * K < (16ull << 20)) {
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));
    } else {
        pol_el = pol_ef;
    }

    if (lda == K) {
        const uint4* xr = reinterpret_cast<const uint4*>(x + r0 * K);
        uint2* orow = reinterpret_cast<uint2*>(out + r0 * K);
        if (rem == 4u && K == 2560u) {
            uint4 v[5];
#pragma unroll
            for (int u = 0; u < 5; ++u) {
                const unsigned int i = threadIdx.x + u * 256u;
                asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                             : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w)
                             : "l"(xr + i), "l"(pol_ef));
            }
#pragma unroll
            for (int u = 0; u < 5; ++u) {
                const unsigned int i = threadIdx.x + u * 256u;
                const uint2 q = fp8_quant_vec8(v[u], inv);
                asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                             :: "l"(orow + i), "r"(q.x), "r"(q.y), "l"(pol_el) : "memory");
            }
        } else if (rem == 4u && K == 6144u) {
#pragma unroll
            for (int b = 0; b < 3; ++b) {
                uint4 v[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const unsigned int i = threadIdx.x + (unsigned int)((4 * b + u) * 256);
                    asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                                 : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w)
                                 : "l"(xr + i), "l"(pol_ef));
                }
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const unsigned int i = threadIdx.x + (unsigned int)((4 * b + u) * 256);
                    const uint2 q = fp8_quant_vec8(v[u], inv);
                    asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                                 :: "l"(orow + i), "r"(q.x), "r"(q.y), "l"(pol_el) : "memory");
                }
            }
        } else {
#pragma unroll 4
            for (unsigned int i = threadIdx.x; i < total; i += 256u) {
                uint4 v;
                asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                             : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                             : "l"(xr + i), "l"(pol_ef));
                const uint2 q = fp8_quant_vec8(v, inv);
                asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                             :: "l"(orow + i), "r"(q.x), "r"(q.y), "l"(pol_el) : "memory");
            }
        }
        if (K == 6144u) {
            __syncthreads();
            const unsigned int total_lines = total >> 3;
            const unsigned char* base_discard = reinterpret_cast<const unsigned char*>(xr);
            if ((reinterpret_cast<unsigned long long>(base_discard) & 127ull) == 0ull) {
                for (unsigned int line = threadIdx.x; line < total_lines; line += 256u) {
                    const unsigned char* p = base_discard + (unsigned long long)line * 128ull;
                    asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
                }
            }
        }
    } else {
        for (unsigned int idx = threadIdx.x; idx < total; idx += 256u) {
            const unsigned int dr = idx / k8;
            const unsigned int i = idx - dr * k8;
            const unsigned long long r = r0 + dr;
            const uint4* xptr = reinterpret_cast<const uint4*>(x + r * lda) + i;
            uint2* optr = reinterpret_cast<uint2*>(out + r * K) + i;
            uint4 v;
            asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                         : "l"(xptr), "l"(pol_ef));
            const uint2 q = fp8_quant_vec8(v, inv);
            asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                         :: "l"(optr), "r"(q.x), "r"(q.y), "l"(pol_el) : "memory");
        }
    }
}

// Fused BF16->FP8 activation quantization + GDN (b, a) gate projection via BF16 MMA.
// DESTRUCTIVE INPUT CONTRACT: after consuming each 64-row tile of `x` (K=2560), this
// kernel executes `discard.global.L2` on `x` (when 128-byte aligned), dropping dirty L2
// lines without writeback; callers MUST treat `x` as dead scratch memory after launch.
extern "C" __global__ void __launch_bounds__(256, 3) qwen4_fp8_act_and_ba_gates_mma(
    const __nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ amax_bits,
    __nv_fp8_e4m3* __restrict__ a8_out,
    float* __restrict__ scale_out,
    const __nv_bfloat16* __restrict__ ba_weight,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ gates_out,
    unsigned int rows,
    unsigned int vheads_per_group,
    unsigned int planar) {
    __shared__ union {
        struct {
            uint4 a[64][8];
            uint4 b[2][96][8];
        } kv;
        struct {
            float ba[32][100];
            float a_coeff[48];
            float dt_b[48];
        } ep;
    } smem;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int warp_m = warp >> 1;
    const unsigned int warp_n = warp & 1u;
    const unsigned int row_base = blockIdx.x * 64u;

    const float amax = fmaxf(__uint_as_float(amax_bits[0]), 1e-30f);
    const float inv = fp8_div_448(amax);
    if (blockIdx.x == 0u && tid == 0u) {
        scale_out[0] = amax * (1.0f / 448.0f);
    }
    if (row_base >= rows) return;

    uint64_t pol_ef, pol_el;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));
    const uint64_t pol_a8 = (row_base < 6400u) ? pol_el : pol_ef;

    float acc[6][4];
#pragma unroll
    for (int nt = 0; nt < 6; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[nt][e] = 0.0f;
    }

    unsigned int s_b_dst[3];
    const uint4* w_base[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const unsigned int idx = tid + (unsigned int)(j * 256);
        const unsigned int n = idx >> 3;
        const unsigned int c8 = idx & 7u;
        s_b_dst[j] = static_cast<unsigned int>(__cvta_generic_to_shared(&smem.kv.b[0][n][c8 ^ (n & 7u)]));
        w_base[j] = reinterpret_cast<const uint4*>(ba_weight + (unsigned long long)n * 2560ull) + c8;
    }

#pragma unroll
    for (int j = 0; j < 3; ++j) {
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, 16, %2;\n"
                     :: "r"(s_b_dst[j]), "l"(w_base[j]), "l"(pol_el));
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    if (row_base + 64u <= rows) {
        // Fast full-tile path (taken by 100% of CTAs when rows is a multiple of 64):
        // zero branch reconvergence (`BSSY`/`BSYNC`) and zero zero-init `MOV`s in K-loop.
        const unsigned int r0_ld = tid >> 3;
        const unsigned int c8_ld = tid & 7u;
        const unsigned long long vb0 = (unsigned long long)(row_base + r0_ld) * 320ull + c8_ld;
        const uint4* x_ptr0 = reinterpret_cast<const uint4*>(x) + vb0;
        uint2* a8_ptr0 = reinterpret_cast<uint2*>(a8_out) + vb0;
        const unsigned int s_a_dst0 =
            static_cast<unsigned int>(__cvta_generic_to_shared(&smem.kv.a[r0_ld][c8_ld ^ (r0_ld & 7u)]));

        uint4 xv[2];
        asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                     : "=r"(xv[0].x), "=r"(xv[0].y), "=r"(xv[0].z), "=r"(xv[0].w)
                     : "l"(x_ptr0), "l"(pol_ef));
        asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4 + 163840], %5;\n"
                     : "=r"(xv[1].x), "=r"(xv[1].y), "=r"(xv[1].z), "=r"(xv[1].w)
                     : "l"(x_ptr0), "l"(pol_ef));

        for (unsigned int k0 = 0u; k0 < 2560u; k0 += 64u) {
            const unsigned long long k_vec = k0 >> 3;
            const uint2 q0 = fp8_quant_vec8(xv[0], inv);
            const uint2 q1 = fp8_quant_vec8(xv[1], inv);
            uint2* optr0 = a8_ptr0 + k_vec;
            asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                         :: "l"(optr0), "r"(q0.x), "r"(q0.y), "l"(pol_a8) : "memory");
            asm volatile("st.global.L2::cache_hint.v2.u32 [%0 + 81920], {%1, %2}, %3;\n"
                         :: "l"(optr0), "r"(q1.x), "r"(q1.y), "l"(pol_a8) : "memory");
            asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
                         :: "r"(s_a_dst0), "r"(xv[0].x), "r"(xv[0].y), "r"(xv[0].z), "r"(xv[0].w)
                         : "memory");
            asm volatile("st.shared.v4.u32 [%0 + 4096], {%1, %2, %3, %4};\n"
                         :: "r"(s_a_dst0), "r"(xv[1].x), "r"(xv[1].y), "r"(xv[1].z), "r"(xv[1].w)
                         : "memory");

            const unsigned int stage = (k0 >> 6) & 1u;
            if (k0 + 64u < 2560u) {
                const unsigned long long next_k_vec = (k0 + 64u) >> 3;
                const unsigned int next_b_off = (stage ^ 1u) * 12288u;
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, 16, %2;\n"
                                 :: "r"(s_b_dst[j] + next_b_off), "l"(w_base[j] + next_k_vec), "l"(pol_el));
                }
                asm volatile("cp.async.commit_group;\n" ::: "memory");
                const uint4* xptr_next = x_ptr0 + next_k_vec;
                asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                             : "=r"(xv[0].x), "=r"(xv[0].y), "=r"(xv[0].z), "=r"(xv[0].w)
                             : "l"(xptr_next), "l"(pol_ef));
                asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4 + 163840], %5;\n"
                             : "=r"(xv[1].x), "=r"(xv[1].y), "=r"(xv[1].z), "=r"(xv[1].w)
                             : "l"(xptr_next), "l"(pol_ef));
                asm volatile("cp.async.wait_group 1;\n" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            }
            __syncthreads();

#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const unsigned int c_base = (unsigned int)(ks * 2);
                const unsigned int r_a = warp_m * 16u + (lane & 15u);
                const unsigned int c_a = (c_base + (lane >> 4)) ^ (r_a & 7u);
                const unsigned int s_a =
                    static_cast<unsigned int>(__cvta_generic_to_shared(&smem.kv.a[r_a][c_a]));
                unsigned int a_frag[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                             : "r"(s_a));

                unsigned int b_pair[3][4];
#pragma unroll
                for (int p = 0; p < 3; ++p) {
                    const unsigned int r_b =
                        warp_n * 48u + (unsigned int)(p * 16) + ((lane >> 4) * 8u) + (lane & 7u);
                    const unsigned int c_b = (c_base + ((lane >> 3) & 1u)) ^ (r_b & 7u);
                    const unsigned int s_b =
                        static_cast<unsigned int>(__cvta_generic_to_shared(&smem.kv.b[stage][r_b][c_b]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(b_pair[p][0]), "=r"(b_pair[p][1]), "=r"(b_pair[p][2]), "=r"(b_pair[p][3])
                                 : "r"(s_b));
                }
#pragma unroll
                for (int p = 0; p < 3; ++p) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[2 * p][0]), "+f"(acc[2 * p][1]),
                          "+f"(acc[2 * p][2]), "+f"(acc[2 * p][3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                          "r"(b_pair[p][0]), "r"(b_pair[p][1]));
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[2 * p + 1][0]), "+f"(acc[2 * p + 1][1]),
                          "+f"(acc[2 * p + 1][2]), "+f"(acc[2 * p + 1][3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                          "r"(b_pair[p][2]), "r"(b_pair[p][3]));
                }
            }
            __syncthreads();
        }
    } else {
        bool row_ok[2];
        unsigned long long vec_base[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const unsigned int idx = tid + (unsigned int)(j * 256);
            const unsigned int r = idx >> 3;
            const unsigned int c8 = idx & 7u;
            const unsigned int global_r = row_base + r;
            row_ok[j] = global_r < rows;
            vec_base[j] = (unsigned long long)global_r * 320ull + c8;
        }

        uint4 xv[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            xv[j] = make_uint4(0u, 0u, 0u, 0u);
            if (row_ok[j]) {
                const uint4* xptr = reinterpret_cast<const uint4*>(x) + vec_base[j];
                asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                             : "=r"(xv[j].x), "=r"(xv[j].y), "=r"(xv[j].z), "=r"(xv[j].w)
                             : "l"(xptr), "l"(pol_ef));
            }
        }

        for (unsigned int k0 = 0u; k0 < 2560u; k0 += 64u) {
            const unsigned long long k_vec = k0 >> 3;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const unsigned int idx = tid + (unsigned int)(j * 256);
                const unsigned int r = idx >> 3;
                const unsigned int c8 = idx & 7u;
                if (row_ok[j]) {
                    uint2* optr = reinterpret_cast<uint2*>(a8_out) + vec_base[j] + k_vec;
                    const uint2 q = fp8_quant_vec8(xv[j], inv);
                    asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                                 :: "l"(optr), "r"(q.x), "r"(q.y), "l"(pol_a8) : "memory");
                }
                smem.kv.a[r][c8 ^ (r & 7u)] = xv[j];
            }

            const unsigned int stage = (k0 >> 6) & 1u;
            if (k0 + 64u < 2560u) {
                const unsigned long long next_k_vec = (k0 + 64u) >> 3;
                const unsigned int next_b_off = (stage ^ 1u) * 12288u;
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, 16, %2;\n"
                                 :: "r"(s_b_dst[j] + next_b_off), "l"(w_base[j] + next_k_vec), "l"(pol_el));
                }
                asm volatile("cp.async.commit_group;\n" ::: "memory");
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    xv[j] = make_uint4(0u, 0u, 0u, 0u);
                    if (row_ok[j]) {
                        const uint4* xptr = reinterpret_cast<const uint4*>(x) + vec_base[j] + next_k_vec;
                        asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                                     : "=r"(xv[j].x), "=r"(xv[j].y), "=r"(xv[j].z), "=r"(xv[j].w)
                                     : "l"(xptr), "l"(pol_ef));
                    }
                }
                asm volatile("cp.async.wait_group 1;\n" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            }
            __syncthreads();

#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const unsigned int c_base = (unsigned int)(ks * 2);
                const unsigned int r_a = warp_m * 16u + (lane & 15u);
                const unsigned int c_a = (c_base + (lane >> 4)) ^ (r_a & 7u);
                const unsigned int s_a =
                    static_cast<unsigned int>(__cvta_generic_to_shared(&smem.kv.a[r_a][c_a]));
                unsigned int a_frag[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                             : "r"(s_a));

                unsigned int b_pair[3][4];
#pragma unroll
                for (int p = 0; p < 3; ++p) {
                    const unsigned int r_b =
                        warp_n * 48u + (unsigned int)(p * 16) + ((lane >> 4) * 8u) + (lane & 7u);
                    const unsigned int c_b = (c_base + ((lane >> 3) & 1u)) ^ (r_b & 7u);
                    const unsigned int s_b =
                        static_cast<unsigned int>(__cvta_generic_to_shared(&smem.kv.b[stage][r_b][c_b]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(b_pair[p][0]), "=r"(b_pair[p][1]), "=r"(b_pair[p][2]), "=r"(b_pair[p][3])
                                 : "r"(s_b));
                }
#pragma unroll
                for (int p = 0; p < 3; ++p) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[2 * p][0]), "+f"(acc[2 * p][1]),
                          "+f"(acc[2 * p][2]), "+f"(acc[2 * p][3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                          "r"(b_pair[p][0]), "r"(b_pair[p][1]));
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[2 * p + 1][0]), "+f"(acc[2 * p + 1][1]),
                          "+f"(acc[2 * p + 1][2]), "+f"(acc[2 * p + 1][3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                          "r"(b_pair[p][2]), "r"(b_pair[p][3]));
                }
            }
            __syncthreads();
        }
    }

    if (tid < 48u) {
        smem.ep.a_coeff[tid] = __expf(fminf(A_log[tid], 20.0f));
        smem.ep.dt_b[tid] = dt_bias[tid];
    }
    {
        const unsigned int valid_rows = (rows - row_base) < 64u ? (rows - row_base) : 64u;
        const unsigned int total_lines = valid_rows * 40u;
        const unsigned char* base_discard =
            reinterpret_cast<const unsigned char*>(x + (unsigned long long)row_base * 2560ull);
        if ((reinterpret_cast<unsigned long long>(base_discard) & 127ull) == 0ull) {
            for (unsigned int line = tid; line < total_lines; line += 256u) {
                const unsigned char* p = base_discard + (unsigned long long)line * 128ull;
                asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
            }
        }
    }

    const unsigned int group_id = lane >> 2;
    const unsigned int tid_in_group = lane & 3u;
    const unsigned int r0 = (warp_m & 1u) * 16u + group_id;
    const unsigned int r1 = r0 + 8u;

#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const unsigned int sub_base = row_base + (unsigned int)pass * 32u;
        if (sub_base >= rows) break;
        if ((warp_m >> 1) == (unsigned int)pass) {
#pragma unroll
            for (int nt = 0; nt < 6; ++nt) {
                const unsigned int c0 = warp_n * 48u + (unsigned int)(nt * 8) + tid_in_group * 2u;
                *reinterpret_cast<float2*>(&smem.ep.ba[r0][c0]) = make_float2(acc[nt][0], acc[nt][1]);
                *reinterpret_cast<float2*>(&smem.ep.ba[r1][c0]) = make_float2(acc[nt][2], acc[nt][3]);
            }
        }
        __syncthreads();

        const unsigned int r_limit = (rows - sub_base) < 32u ? (rows - sub_base) : 32u;
        const unsigned int total_el = r_limit * 96u;
        for (unsigned int idx = tid; idx < total_el; idx += 256u) {
            const unsigned int r = (idx * 21846u) >> 21;
            const unsigned int c = idx - r * 96u;
            const unsigned int global_r = sub_base + r;
            const bool is_gate = (c < 48u);
            const unsigned int vh = is_gate ? c : (c - 48u);
            const unsigned int group = (vheads_per_group == 3u) ? ((vh * 21846u) >> 16) : (vh / vheads_per_group);
            const unsigned int ba_col =
                is_gate ? (vh + (group + 1u) * vheads_per_group) : (vh + group * vheads_per_group);
            const float x_val = smem.ep.ba[r][ba_col];
            float val;
            unsigned long long dst_idx;
            if (is_gate) {
                const float dt = __logf(1.0f + __expf(fminf(x_val + smem.ep.dt_b[vh], 20.0f)));
                val = __expf(-smem.ep.a_coeff[vh] * dt);
                dst_idx = planar ? ((unsigned long long)global_r * 48ull + vh)
                                 : ((unsigned long long)global_r * 96ull + c);
            } else {
                val = fp8_fast_rcp_rn_1p_exp(x_val);
                dst_idx = planar ? (((unsigned long long)rows + global_r) * 48ull + vh)
                                 : ((unsigned long long)global_r * 96ull + c);
            }
            if (rows <= 4096u) {
                asm volatile("st.global.L2::cache_hint.f32 [%0], %1, %2;\n"
                             :: "l"(gates_out + dst_idx), "f"(val), "l"(pol_el) : "memory");
            } else {
                gates_out[dst_idx] = val;
            }
        }
        if (pass == 0) __syncthreads();
    }
}
