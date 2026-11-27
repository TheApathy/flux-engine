// SPDX-License-Identifier: AGPL-3.0-only

// Single-pass Blackwell (SM120/SM121 + SM103 fallback) fused activation
// preparation kernels emitting NVFP4 (block-16 UE4M3 scales + packed E2M1):
//
// 1. `rms_norm_nvfp4_sm120`:
//    BF16 input + BF16 weight -> warp/block RMSNorm -> block-16 absmax ->
//    UE4M3 scales (linear or 128x4 swizzled) + packed E2M1 output.
// 2. `rms_norm_residual_nvfp4_sm120`:
//    Fused `hidden += residual` (writes updated BF16 hidden in-place) +
//    RMSNorm + block-16 NVFP4 quantization in ONE global memory pass
//    (for hidden_size <= 8192; clean streaming fallback for larger shapes).
// 3. `silu_mul_nvfp4_sm120`:
//    Fused SwiGLU `silu(gate) * up` -> block-16 NVFP4 quantization for both
//    split `(gate, up)` and merged `[gate | up]` dense FFN down_proj inputs.
//
// Hardware invariants:
// - Uses `cvt.rn.satfinite.e2m1x2.f32` under `__CUDA_ARCH__ >= 1200` with
//   branchless 2-op SWAR `-0.0` (0x8) -> `+0.0` (0x0) canonicalization.
// - Uses an exact IEEE-754 RNE scalar E2M1 fallback on `< 1200` (e.g. SM103a)
//   matching hardware RNE + `-0.0` canonicalization bit-for-bit.
// - Zero `.local` stack frames, zero register spills, zero STL/LDL, and zero
//   odd uniform descriptors across `sm_121a`, `sm_120a`, and `sm_103a`.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

constexpr unsigned int NVFP4_GROUP_SIZE = 16u;

struct Float8 {
    float v0;
    float v1;
    float v2;
    float v3;
    float v4;
    float v5;
    float v6;
    float v7;
};

__device__ __forceinline__ float sanitize_finite(float x) {
    return isfinite(x) ? x : 0.0f;
}

__device__ __forceinline__ float2 unpack_bf16x2_f32(unsigned int packed) {
    const __nv_bfloat16 lo = __ushort_as_bfloat16(static_cast<unsigned short>(packed & 0xffffu));
    const __nv_bfloat16 hi = __ushort_as_bfloat16(static_cast<unsigned short>(packed >> 16));
    return make_float2(__bfloat162float(lo), __bfloat162float(hi));
}

__device__ __forceinline__ unsigned int pack_bf16x2_f32(float a, float b) {
    const unsigned int lo = static_cast<unsigned int>(__bfloat16_as_ushort(__float2bfloat16(a)));
    const unsigned int hi = static_cast<unsigned int>(__bfloat16_as_ushort(__float2bfloat16(b)));
    return lo | (hi << 16);
}

__device__ __forceinline__ Float8 unpack_uint4_bf16(uint4 raw) {
    const float2 f0 = unpack_bf16x2_f32(raw.x);
    const float2 f1 = unpack_bf16x2_f32(raw.y);
    const float2 f2 = unpack_bf16x2_f32(raw.z);
    const float2 f3 = unpack_bf16x2_f32(raw.w);
    return Float8{f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y};
}

__device__ __forceinline__ float sum_sq_float8(Float8 x) {
    return x.v0 * x.v0 + x.v1 * x.v1 + x.v2 * x.v2 + x.v3 * x.v3
         + x.v4 * x.v4 + x.v5 * x.v5 + x.v6 * x.v6 + x.v7 * x.v7;
}

__device__ __forceinline__ float max_abs_float8(Float8 x) {
    float m = fmaxf(fabsf(x.v0), fabsf(x.v1));
    m = fmaxf(m, fmaxf(fabsf(x.v2), fabsf(x.v3)));
    m = fmaxf(m, fmaxf(fabsf(x.v4), fabsf(x.v5)));
    m = fmaxf(m, fmaxf(fabsf(x.v6), fabsf(x.v7)));
    return m;
}

__device__ __forceinline__ Float8 apply_rmsnorm_float8(
    Float8 x,
    Float8 w,
    float rms,
    float weight_offset
) {
    return Float8{
        sanitize_finite(__fmul_rn(__fmul_rn(sanitize_finite(x.v0), rms), weight_offset + sanitize_finite(w.v0))),
        sanitize_finite(__fmul_rn(__fmul_rn(sanitize_finite(x.v1), rms), weight_offset + sanitize_finite(w.v1))),
        sanitize_finite(__fmul_rn(__fmul_rn(sanitize_finite(x.v2), rms), weight_offset + sanitize_finite(w.v2))),
        sanitize_finite(__fmul_rn(__fmul_rn(sanitize_finite(x.v3), rms), weight_offset + sanitize_finite(w.v3))),
        sanitize_finite(__fmul_rn(__fmul_rn(sanitize_finite(x.v4), rms), weight_offset + sanitize_finite(w.v4))),
        sanitize_finite(__fmul_rn(__fmul_rn(sanitize_finite(x.v5), rms), weight_offset + sanitize_finite(w.v5))),
        sanitize_finite(__fmul_rn(__fmul_rn(sanitize_finite(x.v6), rms), weight_offset + sanitize_finite(w.v6))),
        sanitize_finite(__fmul_rn(__fmul_rn(sanitize_finite(x.v7), rms), weight_offset + sanitize_finite(w.v7))),
    };
}

__device__ __forceinline__ uint4 add_bf16_uint4(uint4 h_raw, uint4 r_raw) {
    const Float8 hv = unpack_uint4_bf16(h_raw);
    const Float8 rv = unpack_uint4_bf16(r_raw);
    uint4 out;
    out.x = pack_bf16x2_f32(hv.v0 + rv.v0, hv.v1 + rv.v1);
    out.y = pack_bf16x2_f32(hv.v2 + rv.v2, hv.v3 + rv.v3);
    out.z = pack_bf16x2_f32(hv.v4 + rv.v4, hv.v5 + rv.v5);
    out.w = pack_bf16x2_f32(hv.v6 + rv.v6, hv.v7 + rv.v7);
    return out;
}

__device__ __forceinline__ float silu_mul_f32(float g, float u) {
    if (!isfinite(g) || !isfinite(u)) {
        return 0.0f;
    }
    const float denom = 1.0f + expf(-g);
    const float sig = isfinite(denom) ? __fdiv_rn(g, denom) : 0.0f;
    return sanitize_finite(__fmul_rn(sig, u));
}

__device__ __forceinline__ Float8 silu_mul_uint4(uint4 g_raw, uint4 u_raw) {
    const Float8 gv = unpack_uint4_bf16(g_raw);
    const Float8 uv = unpack_uint4_bf16(u_raw);
    return Float8{
        silu_mul_f32(gv.v0, uv.v0),
        silu_mul_f32(gv.v1, uv.v1),
        silu_mul_f32(gv.v2, uv.v2),
        silu_mul_f32(gv.v3, uv.v3),
        silu_mul_f32(gv.v4, uv.v4),
        silu_mul_f32(gv.v5, uv.v5),
        silu_mul_f32(gv.v6, uv.v6),
        silu_mul_f32(gv.v7, uv.v7),
    };
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

__device__ __forceinline__ unsigned long long nvfp4_scale_offset(
    unsigned int row,
    unsigned int group,
    unsigned int groups,
    unsigned int layout_128x4
) {
    if (!layout_128x4) {
        return static_cast<unsigned long long>(row) * groups + group;
    }
    const unsigned int group_blocks = (groups + 3u) / 4u;
    return (((((static_cast<unsigned long long>(row / 128u) * group_blocks + (group / 4u)) * 32u
                + (row % 32u)) * 4u + ((row % 128u) / 32u)) * 4u) + (group % 4u));
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
// Branchless SWAR canonicalization of `-0.0` (0x8) -> `+0.0` (0x0) across
// 8 packed E2M1 nibbles in one 32-bit word.
__device__ __forceinline__ unsigned int canonicalize_e2m1x8_neg_zero(unsigned int packed) {
    const unsigned int neg_zero =
        packed & ~((packed & 0x77777777u) + 0x77777777u) & 0x88888888u;
    return packed ^ neg_zero;
}
#else
// Exact IEEE-754 round-to-nearest-even E2M1 encoder with `-0.0` -> `+0.0`
// canonicalization and non-finite clamp to `0x0`, matching SM120+ hardware
// `cvt.rn.satfinite.e2m1x2.f32` + SWAR canonicalization bit-for-bit.
__device__ __forceinline__ unsigned int quantize_e2m1_rne_scalar(float x) {
    if (!isfinite(x)) {
        return 0u;
    }
    const float a = fabsf(x);
    unsigned int idx;
    if (a <= 0.25f) {
        idx = 0u;
    } else if (a < 0.75f) {
        idx = 1u;
    } else if (a <= 1.25f) {
        idx = 2u;
    } else if (a < 1.75f) {
        idx = 3u;
    } else if (a <= 2.5f) {
        idx = 4u;
    } else if (a < 3.5f) {
        idx = 5u;
    } else if (a <= 5.0f) {
        idx = 6u;
    } else {
        idx = 7u;
    }
    const unsigned int sign = (idx != 0u && signbit(x)) ? 8u : 0u;
    return sign | idx;
}
#endif

__device__ __forceinline__ unsigned int e2m1_pack_eight(Float8 vals, float inv_eff) {
    const float s0 = __fmul_rn(vals.v0, inv_eff);
    const float s1 = __fmul_rn(vals.v1, inv_eff);
    const float s2 = __fmul_rn(vals.v2, inv_eff);
    const float s3 = __fmul_rn(vals.v3, inv_eff);
    const float s4 = __fmul_rn(vals.v4, inv_eff);
    const float s5 = __fmul_rn(vals.v5, inv_eff);
    const float s6 = __fmul_rn(vals.v6, inv_eff);
    const float s7 = __fmul_rn(vals.v7, inv_eff);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    unsigned int packed;
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "}"
        : "=r"(packed)
        : "f"(s0), "f"(s1), "f"(s2), "f"(s3), "f"(s4), "f"(s5), "f"(s6), "f"(s7));
    return canonicalize_e2m1x8_neg_zero(packed);
#else
    const unsigned int b0 =
        (quantize_e2m1_rne_scalar(s1) << 4) | quantize_e2m1_rne_scalar(s0);
    const unsigned int b1 =
        (quantize_e2m1_rne_scalar(s3) << 4) | quantize_e2m1_rne_scalar(s2);
    const unsigned int b2 =
        (quantize_e2m1_rne_scalar(s5) << 4) | quantize_e2m1_rne_scalar(s4);
    const unsigned int b3 =
        (quantize_e2m1_rne_scalar(s7) << 4) | quantize_e2m1_rne_scalar(s6);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
#endif
}

__device__ __forceinline__ void store_group16_nvfp4(
    Float8 lo,
    Float8 hi,
    float group_max,
    float global_scale,
    float inv_scale2_over_6,
    unsigned char* __restrict__ row_packed,
    unsigned char* __restrict__ scale_out,
    unsigned int row,
    unsigned int group,
    unsigned int groups,
    unsigned int layout_128x4
) {
    const float safe_max = (isfinite(group_max) && group_max > 0.0f) ? group_max : 0.0f;
    const float fp8_raw =
        safe_max > 0.0f ? fminf(__fmul_rn(safe_max, inv_scale2_over_6), 448.0f) : 0.0f;
    const __nv_fp8_e4m3 encoded_scale = __nv_fp8_e4m3(fp8_raw);
    const unsigned char raw_byte = encoded_scale.__x;
    const unsigned char scale_byte =
        (raw_byte == 0x80u || raw_byte == 0x7Fu || raw_byte == 0xFFu) ? 0u : raw_byte;
    scale_out[nvfp4_scale_offset(row, group, groups, layout_128x4)] = scale_byte;

    const float decoded = (scale_byte == 0u) ? 0.0f : static_cast<float>(encoded_scale);
    const float effective_scale = __fmul_rn(decoded, global_scale);
    const float inv_eff =
        (isfinite(effective_scale) && effective_scale > 0.0f)
            ? __fdiv_rn(1.0f, effective_scale)
            : 0.0f;

    uint2 packed;
    packed.x = e2m1_pack_eight(lo, inv_eff);
    packed.y = e2m1_pack_eight(hi, inv_eff);
    *reinterpret_cast<uint2*>(row_packed + static_cast<unsigned long long>(group) * 8ull) = packed;
}

__device__ __forceinline__ float block_reduce_rms_inv(
    float thread_sum_sq,
    unsigned int cols,
    float eps,
    unsigned int warp_id,
    unsigned int lane_id,
    unsigned int num_warps,
    float* __restrict__ warp_sums
) {
    float sum_sq = warp_reduce_sum(thread_sum_sq);
    if (lane_id == 0u) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();
    if (warp_id == 0u) {
        float val = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0u) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    const float total_sq = warp_sums[0];
    const float denom =
        (isfinite(total_sq) && total_sq >= 0.0f)
            ? (__fdiv_rn(total_sq, static_cast<float>(cols)) + eps)
            : 0.0f;
    return (isfinite(denom) && denom > 0.0f) ? rsqrtf(denom) : 0.0f;
}

}  // namespace

// 1. Fused RMSNorm -> block-16 NVFP4 activation quantizer.
//    Single global input pass for cols <= 8192 (groups <= blockDim.x <= 512).
extern "C" __global__ void __launch_bounds__(512)
rms_norm_nvfp4_sm120(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float global_scale,
    float eps,
    float weight_offset,
    unsigned int rows,
    unsigned int cols,
    unsigned int layout_128x4
) {
    if (rows == 0u || cols == 0u) {
        return;
    }
    const unsigned int total_rows = layout_128x4 ? (((rows + 127u) / 128u) * 128u) : rows;
    const unsigned int groups = cols / NVFP4_GROUP_SIZE;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32u;
    const unsigned int lane_id = tid % 32u;
    const unsigned int num_warps = (blockDim.x + 31u) / 32u;
    const float scale_denom = __fmul_rn(global_scale, 6.0f);
    const float inv_scale2_over_6 =
        (isfinite(scale_denom) && scale_denom > 0.0f) ? __fdiv_rn(1.0f, scale_denom) : 0.0f;
    const bool single_pass = (groups <= blockDim.x);

    __shared__ float warp_sums[32];

    for (unsigned int row = blockIdx.x; row < total_rows; row += gridDim.x) {
        if (row >= rows) {
            for (unsigned int g = tid; g < groups; g += blockDim.x) {
                scale_out[nvfp4_scale_offset(row, g, groups, layout_128x4)] = 0u;
            }
            continue;
        }

        const __nv_bfloat16* x_row = input + static_cast<unsigned long long>(row) * cols;
        unsigned char* row_packed =
            packed_out + static_cast<unsigned long long>(row) * (cols / 2u);

        uint4 x_cached_lo = make_uint4(0u, 0u, 0u, 0u);
        uint4 x_cached_hi = make_uint4(0u, 0u, 0u, 0u);
        float sum_sq = 0.0f;

        if (single_pass) {
            if (tid < groups) {
                const unsigned int base = tid * NVFP4_GROUP_SIZE;
                x_cached_lo = *reinterpret_cast<const uint4*>(x_row + base);
                x_cached_hi = *reinterpret_cast<const uint4*>(x_row + base + 8u);
                sum_sq = sum_sq_float8(unpack_uint4_bf16(x_cached_lo))
                       + sum_sq_float8(unpack_uint4_bf16(x_cached_hi));
            }
        } else {
            for (unsigned int g = tid; g < groups; g += blockDim.x) {
                const unsigned int base = g * NVFP4_GROUP_SIZE;
                const uint4 x0 = *reinterpret_cast<const uint4*>(x_row + base);
                const uint4 x1 = *reinterpret_cast<const uint4*>(x_row + base + 8u);
                sum_sq += sum_sq_float8(unpack_uint4_bf16(x0))
                        + sum_sq_float8(unpack_uint4_bf16(x1));
            }
        }

        const float rms = block_reduce_rms_inv(
            sum_sq, cols, eps, warp_id, lane_id, num_warps, warp_sums);

        if (single_pass) {
            if (tid < groups) {
                const unsigned int base = tid * NVFP4_GROUP_SIZE;
                const uint4 w0 = *reinterpret_cast<const uint4*>(weight + base);
                const uint4 w1 = *reinterpret_cast<const uint4*>(weight + base + 8u);
                const Float8 lo = apply_rmsnorm_float8(
                    unpack_uint4_bf16(x_cached_lo), unpack_uint4_bf16(w0), rms, weight_offset);
                const Float8 hi = apply_rmsnorm_float8(
                    unpack_uint4_bf16(x_cached_hi), unpack_uint4_bf16(w1), rms, weight_offset);
                const float group_max = fmaxf(max_abs_float8(lo), max_abs_float8(hi));
                store_group16_nvfp4(
                    lo,
                    hi,
                    group_max,
                    global_scale,
                    inv_scale2_over_6,
                    row_packed,
                    scale_out,
                    row,
                    tid,
                    groups,
                    layout_128x4);
            }
        } else {
            for (unsigned int g = tid; g < groups; g += blockDim.x) {
                const unsigned int base = g * NVFP4_GROUP_SIZE;
                const uint4 x0 = *reinterpret_cast<const uint4*>(x_row + base);
                const uint4 x1 = *reinterpret_cast<const uint4*>(x_row + base + 8u);
                const uint4 w0 = *reinterpret_cast<const uint4*>(weight + base);
                const uint4 w1 = *reinterpret_cast<const uint4*>(weight + base + 8u);
                const Float8 lo = apply_rmsnorm_float8(
                    unpack_uint4_bf16(x0), unpack_uint4_bf16(w0), rms, weight_offset);
                const Float8 hi = apply_rmsnorm_float8(
                    unpack_uint4_bf16(x1), unpack_uint4_bf16(w1), rms, weight_offset);
                const float group_max = fmaxf(max_abs_float8(lo), max_abs_float8(hi));
                store_group16_nvfp4(
                    lo,
                    hi,
                    group_max,
                    global_scale,
                    inv_scale2_over_6,
                    row_packed,
                    scale_out,
                    row,
                    g,
                    groups,
                    layout_128x4);
            }
        }
        __syncthreads();
    }
}

// 2. Fused in-place `hidden += residual` + RMSNorm -> block-16 NVFP4 quantizer.
//    Updates `hidden` in BF16 in-place and normalizes over the exact BF16-updated
//    residual values so single-pass (cols <= 8192) and multi-pass (cols > 8192)
//    are bit-identical to `residual_add` followed by `rms_norm_nvfp4_sm120`.
extern "C" __global__ void __launch_bounds__(512)
rms_norm_residual_nvfp4_sm120(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float global_scale,
    float eps,
    float weight_offset,
    unsigned int rows,
    unsigned int cols,
    unsigned int layout_128x4
) {
    if (rows == 0u || cols == 0u) {
        return;
    }
    const unsigned int total_rows = layout_128x4 ? (((rows + 127u) / 128u) * 128u) : rows;
    const unsigned int groups = cols / NVFP4_GROUP_SIZE;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32u;
    const unsigned int lane_id = tid % 32u;
    const unsigned int num_warps = (blockDim.x + 31u) / 32u;
    const float scale_denom = __fmul_rn(global_scale, 6.0f);
    const float inv_scale2_over_6 =
        (isfinite(scale_denom) && scale_denom > 0.0f) ? __fdiv_rn(1.0f, scale_denom) : 0.0f;
    const bool single_pass = (groups <= blockDim.x);

    __shared__ float warp_sums[32];

    for (unsigned int row = blockIdx.x; row < total_rows; row += gridDim.x) {
        if (row >= rows) {
            for (unsigned int g = tid; g < groups; g += blockDim.x) {
                scale_out[nvfp4_scale_offset(row, g, groups, layout_128x4)] = 0u;
            }
            continue;
        }

        const unsigned long long row_base = static_cast<unsigned long long>(row) * cols;
        __nv_bfloat16* h_row = hidden + row_base;
        const __nv_bfloat16* r_row = residual + row_base;
        unsigned char* row_packed =
            packed_out + static_cast<unsigned long long>(row) * (cols / 2u);

        uint4 h_cached_lo = make_uint4(0u, 0u, 0u, 0u);
        uint4 h_cached_hi = make_uint4(0u, 0u, 0u, 0u);
        float sum_sq = 0.0f;

        if (single_pass) {
            if (tid < groups) {
                const unsigned int base = tid * NVFP4_GROUP_SIZE;
                const uint4 h0 = *reinterpret_cast<const uint4*>(h_row + base);
                const uint4 h1 = *reinterpret_cast<const uint4*>(h_row + base + 8u);
                const uint4 r0 = *reinterpret_cast<const uint4*>(r_row + base);
                const uint4 r1 = *reinterpret_cast<const uint4*>(r_row + base + 8u);
                h_cached_lo = add_bf16_uint4(h0, r0);
                h_cached_hi = add_bf16_uint4(h1, r1);
                *reinterpret_cast<uint4*>(h_row + base) = h_cached_lo;
                *reinterpret_cast<uint4*>(h_row + base + 8u) = h_cached_hi;
                sum_sq = sum_sq_float8(unpack_uint4_bf16(h_cached_lo))
                       + sum_sq_float8(unpack_uint4_bf16(h_cached_hi));
            }
        } else {
            for (unsigned int g = tid; g < groups; g += blockDim.x) {
                const unsigned int base = g * NVFP4_GROUP_SIZE;
                const uint4 h0 = *reinterpret_cast<const uint4*>(h_row + base);
                const uint4 h1 = *reinterpret_cast<const uint4*>(h_row + base + 8u);
                const uint4 r0 = *reinterpret_cast<const uint4*>(r_row + base);
                const uint4 r1 = *reinterpret_cast<const uint4*>(r_row + base + 8u);
                const uint4 o0 = add_bf16_uint4(h0, r0);
                const uint4 o1 = add_bf16_uint4(h1, r1);
                *reinterpret_cast<uint4*>(h_row + base) = o0;
                *reinterpret_cast<uint4*>(h_row + base + 8u) = o1;
                sum_sq += sum_sq_float8(unpack_uint4_bf16(o0))
                        + sum_sq_float8(unpack_uint4_bf16(o1));
            }
        }

        const float rms = block_reduce_rms_inv(
            sum_sq, cols, eps, warp_id, lane_id, num_warps, warp_sums);

        if (single_pass) {
            if (tid < groups) {
                const unsigned int base = tid * NVFP4_GROUP_SIZE;
                const uint4 w0 = *reinterpret_cast<const uint4*>(weight + base);
                const uint4 w1 = *reinterpret_cast<const uint4*>(weight + base + 8u);
                const Float8 lo = apply_rmsnorm_float8(
                    unpack_uint4_bf16(h_cached_lo), unpack_uint4_bf16(w0), rms, weight_offset);
                const Float8 hi = apply_rmsnorm_float8(
                    unpack_uint4_bf16(h_cached_hi), unpack_uint4_bf16(w1), rms, weight_offset);
                const float group_max = fmaxf(max_abs_float8(lo), max_abs_float8(hi));
                store_group16_nvfp4(
                    lo,
                    hi,
                    group_max,
                    global_scale,
                    inv_scale2_over_6,
                    row_packed,
                    scale_out,
                    row,
                    tid,
                    groups,
                    layout_128x4);
            }
        } else {
            for (unsigned int g = tid; g < groups; g += blockDim.x) {
                const unsigned int base = g * NVFP4_GROUP_SIZE;
                const uint4 h0 = *reinterpret_cast<const uint4*>(h_row + base);
                const uint4 h1 = *reinterpret_cast<const uint4*>(h_row + base + 8u);
                const uint4 w0 = *reinterpret_cast<const uint4*>(weight + base);
                const uint4 w1 = *reinterpret_cast<const uint4*>(weight + base + 8u);
                const Float8 lo = apply_rmsnorm_float8(
                    unpack_uint4_bf16(h0), unpack_uint4_bf16(w0), rms, weight_offset);
                const Float8 hi = apply_rmsnorm_float8(
                    unpack_uint4_bf16(h1), unpack_uint4_bf16(w1), rms, weight_offset);
                const float group_max = fmaxf(max_abs_float8(lo), max_abs_float8(hi));
                store_group16_nvfp4(
                    lo,
                    hi,
                    group_max,
                    global_scale,
                    inv_scale2_over_6,
                    row_packed,
                    scale_out,
                    row,
                    g,
                    groups,
                    layout_128x4);
            }
        }
        __syncthreads();
    }
}

// 3. Fused SwiGLU `silu(gate) * up` -> block-16 NVFP4 quantizer.
//    Supports both split `(gate[rows, cols], up[rows, cols])` with
//    `row_stride_elems = cols` and merged `[gate | up]` (`[rows, 2 * cols]`)
//    with `up = gate + cols` and `row_stride_elems = 2 * cols`.
extern "C" __global__ void __launch_bounds__(256)
silu_mul_nvfp4_sm120(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float global_scale,
    unsigned int rows,
    unsigned int cols,
    unsigned int row_stride_elems,
    unsigned int layout_128x4
) {
    if (rows == 0u || cols == 0u) {
        return;
    }
    const unsigned int total_rows = layout_128x4 ? (((rows + 127u) / 128u) * 128u) : rows;
    const unsigned int groups = cols / NVFP4_GROUP_SIZE;
    const float scale_denom = __fmul_rn(global_scale, 6.0f);
    const float inv_scale2_over_6 =
        (isfinite(scale_denom) && scale_denom > 0.0f) ? __fdiv_rn(1.0f, scale_denom) : 0.0f;

    for (unsigned int row = blockIdx.x; row < total_rows; row += gridDim.x) {
        if (row >= rows) {
            for (unsigned int g = threadIdx.x; g < groups; g += blockDim.x) {
                scale_out[nvfp4_scale_offset(row, g, groups, layout_128x4)] = 0u;
            }
            continue;
        }

        const unsigned long long row_offset =
            static_cast<unsigned long long>(row) * row_stride_elems;
        const __nv_bfloat16* row_gate = gate + row_offset;
        const __nv_bfloat16* row_up = up + row_offset;
        unsigned char* row_packed =
            packed_out + static_cast<unsigned long long>(row) * (cols / 2u);

        for (unsigned int g = threadIdx.x; g < groups; g += blockDim.x) {
            const unsigned int base = g * NVFP4_GROUP_SIZE;
            const uint4 g0 = *reinterpret_cast<const uint4*>(row_gate + base);
            const uint4 u0 = *reinterpret_cast<const uint4*>(row_up + base);
            const uint4 g1 = *reinterpret_cast<const uint4*>(row_gate + base + 8u);
            const uint4 u1 = *reinterpret_cast<const uint4*>(row_up + base + 8u);

            const Float8 lo = silu_mul_uint4(g0, u0);
            const Float8 hi = silu_mul_uint4(g1, u1);
            const float group_max = fmaxf(max_abs_float8(lo), max_abs_float8(hi));
            store_group16_nvfp4(
                lo,
                hi,
                group_max,
                global_scale,
                inv_scale2_over_6,
                row_packed,
                scale_out,
                row,
                g,
                groups,
                layout_128x4);
        }
    }
}
