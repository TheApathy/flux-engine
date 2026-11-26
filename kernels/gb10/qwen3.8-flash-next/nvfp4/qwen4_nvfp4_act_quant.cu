// SPDX-License-Identifier: AGPL-3.0-only
//
// Zero-sync, CUDA-graph-safe NVFP4 activation quantizers and 128x4 weight-scale
// swizzler for native CUDA 13.0 cuBLASLt NVFP4 (`CUDA_R_4F_E2M1` +
// `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`) projections in Flash-Next.
//
// All kernels operate purely on device pointers (zero host syncs, zero D2H
// scalar reads). Device-side `alpha = act_scale2 * weight_scale_2` is produced
// directly on stream for `CUBLASLT_POINTER_MODE_DEVICE`.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float mul_rn(float lhs, float rhs) {
    float result;
    asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(lhs), "f"(rhs));
    return result;
}

__device__ __forceinline__ float rcp_approx_ftz(float value) {
    float result;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

// Blackwell hardware E2M1x2 rounding conversion: packs 8 FP32 values into one
// 32-bit word (low nibble = even index, high nibble = odd index, matching
// cuBLASLt `CUDA_R_4F_E2M1` row-major K-packed layout).
__device__ __forceinline__ unsigned int cvt_f32x8_to_e2m1x8(const float* values, float scale) {
    unsigned int result;
    const float v0 = mul_rn(values[0], scale);
    const float v1 = mul_rn(values[1], scale);
    const float v2 = mul_rn(values[2], scale);
    const float v3 = mul_rn(values[3], scale);
    const float v4 = mul_rn(values[4], scale);
    const float v5 = mul_rn(values[5], scale);
    const float v6 = mul_rn(values[6], scale);
    const float v7 = mul_rn(values[7], scale);
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "}"
        : "=r"(result)
        : "f"(v0), "f"(v1), "f"(v2), "f"(v3), "f"(v4), "f"(v5), "f"(v6), "f"(v7));
    return result;
}

// CUDA 13.0 cuBLASLt `128x4` swizzled block-scale byte offset:
// matches `spark_runtime::cublaslt::block_scale_swizzle_offset`.
__device__ __forceinline__ unsigned long long scale_offset_128x4(
    unsigned int row,
    unsigned int group,
    unsigned int k_tiles
) {
    const unsigned long long tile_idx =
        static_cast<unsigned long long>(row >> 7) * k_tiles + (group >> 2);
    const unsigned int r = row & 127u;
    return (tile_idx << 9) + ((r & 31u) << 4) + ((r >> 5) << 2) + (group & 3u);
}

__device__ __forceinline__ void load_bf16_8_ef(
    const __nv_bfloat16* __restrict__ ptr,
    uint64_t pol_ef,
    float* __restrict__ out
) {
    uint4 raw;
    asm volatile(
        "ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
        : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
        : "l"(ptr), "l"(pol_ef));
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
    const float2 f0 = __bfloat1622float2(p[0]);
    const float2 f1 = __bfloat1622float2(p[1]);
    const float2 f2 = __bfloat1622float2(p[2]);
    const float2 f3 = __bfloat1622float2(p[3]);
    out[0] = f0.x;
    out[1] = f0.y;
    out[2] = f1.x;
    out[3] = f1.y;
    out[4] = f2.x;
    out[5] = f2.y;
    out[6] = f3.x;
    out[7] = f3.y;
}

__device__ __forceinline__ float decode_e4m3_finite(unsigned char b) {
    if ((b & 0x7fu) == 0x7fu) return 0.0f;
    __nv_fp8_e4m3 v;
    v.__x = b;
    return static_cast<float>(v);
}

__device__ __forceinline__ float block_max_256(float v, float* sm) {
    for (int d = 16; d; d >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, d));
    }
    if ((threadIdx.x & 31u) == 0u) {
        sm[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    if (threadIdx.x < 32u) {
        float w = (threadIdx.x < (blockDim.x >> 5)) ? sm[threadIdx.x] : 0.0f;
        for (int d = 16; d; d >>= 1) {
            w = fmaxf(w, __shfl_xor_sync(0xffffffffu, w, d));
        }
        if (threadIdx.x == 0u) sm[0] = w;
    }
    __syncthreads();
    return sm[0];
}

}  // namespace

// Self-contained BF16 global absmax reduction (identical `atomicMax` u32 float-bits
// contract to `qwen4_fp8_proj::qwen4_fp8_act_amax` and fused producer amax slots).
// Caller must zero `amax_bits` (4 bytes) before launch. `count` must be a multiple of 8.
extern "C" __global__ void __launch_bounds__(256) qwen4_nvfp4_act_amax_bf16(
    const __nv_bfloat16* __restrict__ x,
    unsigned long long count,
    unsigned int* __restrict__ amax_bits
) {
    __shared__ float sm[32];
    float m = 0.0f;
    const uint4* v = reinterpret_cast<const uint4*>(x);
    const unsigned long long vecs = count >> 3;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < vecs;
         i += stride) {
        const uint4 w = v[i];
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&w);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(p[j]);
            m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    m = block_max_256(m, sm);
    if (threadIdx.x == 0u) {
        atomicMax(amax_bits, __float_as_uint(m));
    }
}

// Vectorized Blackwell PTX BF16 -> NVFP4 E2M1 + 128x4 UE4M3 block-scale quantizer.
//
// Reads global absmax from `amax_bits[0]` (device u32 float bits), writes:
// - `packed_out`: row-major `[rows, cols / 2]` packed E2M1 bytes,
// - `scale_out`: `128x4`-swizzled UE4M3 block scales with every padded row
//   (`rows..padded_rows`) and padded K-group (`groups..padded_groups`) zeroed,
// - `act_scale2_out[0]` (when non-null): `act_scale2 = amax / 2688.0f` (or `1.0f` if zero),
// - `alpha_out[0]` (when non-null): `act_scale2 * weight_scale_2` for direct
//   `CUBLASLT_POINTER_MODE_DEVICE` matmul consumption without a separate alpha launch.
//
// Grid: `[div_ceil(padded_groups, 128), min(padded_rows, 32768), 1]`, Block: `[128, 1, 1]`.
extern "C" __global__ void __launch_bounds__(128) qwen4_nvfp4_quant_bf16_128x4_from_amax(
    const __nv_bfloat16* __restrict__ input,
    const unsigned int* __restrict__ amax_bits,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float* __restrict__ act_scale2_out,
    float* __restrict__ alpha_out,
    float weight_scale_2,
    unsigned int rows,
    unsigned int cols
) {
    const unsigned int group = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int groups = cols >> 4;
    const unsigned int k_tiles = (groups + 3u) >> 2;
    const unsigned int padded_groups = k_tiles << 2;
    if (group >= padded_groups) return;

    const float amax = __uint_as_float(amax_bits[0]);
    const bool positive = isfinite(amax) && (amax > 1e-30f);
    const float scale2 = positive ? __fdiv_rn(amax, 2688.0f) : 1.0f;
    const float inv_scale2_over_6 = positive ? __fdiv_rn(448.0f, amax) : 0.0f;

    if (blockIdx.x == 0u && blockIdx.y == 0u && threadIdx.x == 0u) {
        if (act_scale2_out != nullptr) {
            act_scale2_out[0] = scale2;
        }
        if (alpha_out != nullptr) {
            alpha_out[0] = mul_rn(scale2, weight_scale_2);
        }
    }

    const unsigned int padded_rows = ((rows + 127u) >> 7) << 7;
    const unsigned long long half_cols = static_cast<unsigned long long>(cols >> 1);
    uint64_t pol_ef;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));

    for (unsigned int row = blockIdx.y; row < padded_rows; row += gridDim.y) {
        if (row >= rows || group >= groups) {
            scale_out[scale_offset_128x4(row, group, k_tiles)] = 0u;
            continue;
        }

        const __nv_bfloat16* group_in =
            input + static_cast<unsigned long long>(row) * cols + (static_cast<unsigned long long>(group) << 4);
        float values[16];
        load_bf16_8_ef(group_in, pol_ef, values);
        load_bf16_8_ef(group_in + 8u, pol_ef, values + 8);

        float group_max = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            group_max = fmaxf(group_max, fabsf(values[i]));
        }

        const float fp8_float =
            (group_max > 0.0f) ? fminf(mul_rn(group_max, inv_scale2_over_6), 448.0f) : 0.0f;
        const __nv_fp8_e4m3 encoded_scale = __nv_fp8_e4m3(fp8_float);
        scale_out[scale_offset_128x4(row, group, k_tiles)] = encoded_scale.__x;

        const float decoded = static_cast<float>(encoded_scale);
        const float effective_scale = mul_rn(decoded, scale2);
        const float inv_eff = (effective_scale > 0.0f) ? rcp_approx_ftz(effective_scale) : 0.0f;

        uint2 packed;
        packed.x = cvt_f32x8_to_e2m1x8(values, inv_eff);
        packed.y = cvt_f32x8_to_e2m1x8(values + 8, inv_eff);
        *reinterpret_cast<uint2*>(
            packed_out + static_cast<unsigned long long>(row) * half_cols
            + (static_cast<unsigned long long>(group) << 3)) = packed;
    }
}

// Single-pass FP8 E4M3 -> NVFP4 E2M1 + 128x4 UE4M3 block-scale requantizer
// for fused E4M3 producers (`GDN_NORM_FP8`, `QSA_PREFILL_E4M3_OUT`).
//
// Because each E4M3 value satisfies `|q| <= 448.0` and `x = q * a_scale`, setting
// `act_scale2 = a_scale / 6.0` makes every group scale `sf = max(|q|) <= 448.0`
// exact in `UE4M3` with zero absmax reduction pass and zero scale-rounding error.
//
// Grid: `[div_ceil(padded_groups, 128), min(padded_rows, 32768), 1]`, Block: `[128, 1, 1]`.
extern "C" __global__ void __launch_bounds__(128) qwen4_nvfp4_quant_e4m3_128x4(
    const unsigned char* __restrict__ a8,
    const float* __restrict__ a_scale_ptr,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float* __restrict__ act_scale2_out,
    float* __restrict__ alpha_out,
    float weight_scale_2,
    unsigned int rows,
    unsigned int cols
) {
    const unsigned int group = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int groups = cols >> 4;
    const unsigned int k_tiles = (groups + 3u) >> 2;
    const unsigned int padded_groups = k_tiles << 2;
    if (group >= padded_groups) return;

    if (blockIdx.x == 0u && blockIdx.y == 0u && threadIdx.x == 0u) {
        const float a_scale = a_scale_ptr[0];
        const float act_scale2 = __fdiv_rn(a_scale, 6.0f);
        if (act_scale2_out != nullptr) {
            act_scale2_out[0] = act_scale2;
        }
        if (alpha_out != nullptr) {
            alpha_out[0] = mul_rn(act_scale2, weight_scale_2);
        }
    }

    const unsigned int padded_rows = ((rows + 127u) >> 7) << 7;
    const unsigned long long half_cols = static_cast<unsigned long long>(cols >> 1);
    uint64_t pol_ef;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));

    for (unsigned int row = blockIdx.y; row < padded_rows; row += gridDim.y) {
        if (row >= rows || group >= groups) {
            scale_out[scale_offset_128x4(row, group, k_tiles)] = 0u;
            continue;
        }

        const unsigned char* group_ptr =
            a8 + static_cast<unsigned long long>(row) * cols + (static_cast<unsigned long long>(group) << 4);
        uint4 raw;
        asm volatile(
            "ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
            : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
            : "l"(group_ptr), "l"(pol_ef));

        const unsigned int words[4] = {raw.x, raw.y, raw.z, raw.w};
        float values[16];
        float group_max = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const unsigned int word = words[w];
            const float v0 = decode_e4m3_finite(static_cast<unsigned char>(word & 0xffu));
            const float v1 = decode_e4m3_finite(static_cast<unsigned char>((word >> 8) & 0xffu));
            const float v2 = decode_e4m3_finite(static_cast<unsigned char>((word >> 16) & 0xffu));
            const float v3 = decode_e4m3_finite(static_cast<unsigned char>((word >> 24) & 0xffu));
            values[w * 4 + 0] = v0;
            values[w * 4 + 1] = v1;
            values[w * 4 + 2] = v2;
            values[w * 4 + 3] = v3;
            group_max = fmaxf(group_max, fmaxf(fabsf(v0), fabsf(v1)));
            group_max = fmaxf(group_max, fmaxf(fabsf(v2), fabsf(v3)));
        }

        const __nv_fp8_e4m3 encoded_scale = __nv_fp8_e4m3(fminf(group_max, 448.0f));
        scale_out[scale_offset_128x4(row, group, k_tiles)] = encoded_scale.__x;

        const float decoded = static_cast<float>(encoded_scale);
        const float inv_eff = (decoded > 0.0f) ? mul_rn(6.0f, rcp_approx_ftz(decoded)) : 0.0f;

        uint2 packed;
        packed.x = cvt_f32x8_to_e2m1x8(values, inv_eff);
        packed.y = cvt_f32x8_to_e2m1x8(values + 8, inv_eff);
        *reinterpret_cast<uint2*>(
            packed_out + static_cast<unsigned long long>(row) * half_cols
            + (static_cast<unsigned long long>(group) << 3)) = packed;
    }
}

// One-time weight preparation: swizzles flat `[N, K/16]` UE4M3 scales into
// cuBLASLt `128x4` layout (zeroing all padded rows/groups), and when
// `qg_heads != 0` also row-permutes `[Q_h, G_h]` -> `[Q_all | G_all]` into
// `packed_out` so `q_proj` writes deinterleaved outputs directly.
//
// Grid: `[div_ceil(padded_groups, 128), min(padded_N, 32768), 1]`, Block: `[128, 1, 1]`.
extern "C" __global__ void __launch_bounds__(128) qwen4_nvfp4_swizzle_weight_scales_128x4(
    const unsigned char* __restrict__ packed_in,
    const unsigned char* __restrict__ scale_in,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    unsigned int N,
    unsigned int K,
    unsigned int qg_heads,
    unsigned int qg_head_dim
) {
    const unsigned int group = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int groups = K >> 4;
    const unsigned int k_tiles = (groups + 3u) >> 2;
    const unsigned int padded_groups = k_tiles << 2;
    if (group >= padded_groups) return;

    const unsigned int padded_n = ((N + 127u) >> 7) << 7;
    const unsigned long long half_k = static_cast<unsigned long long>(K >> 1);
    const unsigned int group_dim = qg_head_dim << 1;

    for (unsigned int row = blockIdx.y; row < padded_n; row += gridDim.y) {
        if (row >= N || group >= groups) {
            scale_out[scale_offset_128x4(row, group, k_tiles)] = 0u;
            continue;
        }

        unsigned int dst_row = row;
        if (qg_heads != 0u) {
            const unsigned int h = row / group_dim;
            const unsigned int idx = row - h * group_dim;
            dst_row = (idx < qg_head_dim)
                ? (h * qg_head_dim + idx)
                : (qg_heads * qg_head_dim + h * qg_head_dim + (idx - qg_head_dim));
        }

        scale_out[scale_offset_128x4(dst_row, group, k_tiles)] =
            scale_in[static_cast<unsigned long long>(row) * groups + group];

        if (qg_heads != 0u && packed_in != nullptr && packed_out != nullptr) {
            const uint2 p8 = *reinterpret_cast<const uint2*>(
                packed_in + static_cast<unsigned long long>(row) * half_k
                + (static_cast<unsigned long long>(group) << 3));
            *reinterpret_cast<uint2*>(
                packed_out + static_cast<unsigned long long>(dst_row) * half_k
                + (static_cast<unsigned long long>(group) << 3)) = p8;
        }
    }
}

// Compute up to 4 independent device-resident FP32 `alpha[i] = act_scale2[0] * ws2[i]`
// values in a single tiny launch for shared-activation projections (Q/K/V).
// Grid: `[1, 1, 1]`, Block: `[32, 1, 1]`.
extern "C" __global__ void __launch_bounds__(32) qwen4_nvfp4_compute_alphas4(
    const float* __restrict__ act_scale2,
    float w0,
    float w1,
    float w2,
    float w3,
    unsigned int count,
    float* __restrict__ alphas_out
) {
    const unsigned int i = threadIdx.x;
    if (blockIdx.x == 0u && i < count && i < 4u) {
        const float ws2 = (i == 0u) ? w0 : ((i == 1u) ? w1 : ((i == 2u) ? w2 : w3));
        alphas_out[i] = mul_rn(act_scale2[0], ws2);
    }
}
