// SPDX-License-Identifier: AGPL-3.0-only
//
// Flash-Next GDN gated RMS norm fused with the FP8 (tensor-scale) quantization
// of the out-projection activation (ATLAS_QWEN4_PREFILL_GDN_NORM_FP8).
//
// The FI route writes the GDN output in 16 bits; today the gated norm writes
// BF16 rows, qwen4_fp8_act_amax re-reads them for the tensor absmax, and
// qwen4_fp8_act_rows_tensor re-reads them again to write e4m3. Here:
//   qwen4_gdn_norm_amax_*  computes the same BF16-rounded normalized values
//                          and only reduces their absmax (atomicMax on the
//                          float bits, values are non-negative);
//   qwen4_gdn_norm_fp8_*   recomputes them and writes e4m3 = fp8(bf16(v) *
//                          448 / amax) directly, plus the scale amax / 448.
// With amax_bits from the first kernel (mode 1) the e4m3 rows and the scale
// are bit-identical to norm -> act_amax -> act_rows_tensor (same per-element
// arithmetic as qwen4_gated_rms_norm_16in_body, same rounding, same scale).
// With amax_bits == nullptr (mode 2) the scale is the analytic bound
// sqrt(head_dim) * max|w| (|x_i| * rsqrt(mean(x^2) + eps) <= sqrt(head_dim),
// sigmoid <= 1), so the absmax pass is skipped; only the scale differs.
// Grid: (num_v_heads, rows)   Block: (head_dim = 128, 1, 1)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ float qwen4_nf8_load(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float qwen4_nf8_load(__half v) { return __half2float(v); }

// 0-ULP fast-math num / (1.0f + expf(-g)) without slowpath subroutine call (`CALL.REL.NOINC`),
// matching build_target.rs `--use_fast_math --fmad=false` bit-for-bit across all inputs.
__device__ __forceinline__ float qwen4_nf8_div_rn_1p_exp(float num, float g) {
    float q;
    asm("{\n\t"
        ".reg .f32 z, t, den;\n\t"
        "mul.rn.ftz.f32 z, %2, 0fbfb8aa3b;\n\t"
        "ex2.approx.ftz.f32 t, z;\n\t"
        "add.rn.ftz.f32 den, t, 0f3f800000;\n\t"
        "div.approx.ftz.f32 %0, %1, den;\n\t"
        "}"
        : "=f"(q) : "f"(num), "f"(g));
    return q;
}

__device__ __forceinline__ float qwen4_nf8_div_448(float amax) {
    float q;
    asm("div.approx.ftz.f32 %0, 0f43e00000, %1;" : "=f"(q) : "f"(amax));
    return q;
}

__device__ __forceinline__ float qwen4_nf8_div_approx(float a, float b) {
    float q;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(q) : "f"(a), "f"(b));
    return q;
}

__device__ __forceinline__ float qwen4_nf8_sqrt_approx(float a) {
    float s;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(s) : "f"(a));
    return s;
}

// Block-wide sum with the reduction order of qwen4_gated_rms_norm_16in_body.
__device__ __forceinline__ float qwen4_nf8_block_sum(float v, float* sm) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    const unsigned int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    if (lane_id == 0) sm[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? sm[lane_id] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) val += __shfl_xor_sync(0xffffffff, val, off);
        if (lane_id == 0) sm[0] = val;
    }
    __syncthreads();
    const float r = sm[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float qwen4_nf8_block_max(float v, float* sm) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    const unsigned int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    if (lane_id == 0) sm[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? sm[lane_id] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
        if (lane_id == 0) sm[0] = val;
    }
    __syncthreads();
    const float r = sm[0];
    __syncthreads();
    return r;
}

// One element per thread (head_dim == blockDim.x == 128): the BF16-rounded
// normalized value the unfused norm writes.
template <typename InT>
__device__ __forceinline__ float qwen4_nf8_value(
    const InT* input, const __nv_bfloat16* gate, const __nv_bfloat16* weight,
    unsigned int head_dim, float eps, unsigned int input_stride, unsigned int gate_stride,
    float* sm) {
    const unsigned int head = blockIdx.x, row = blockIdx.y, i = threadIdx.x;
    const InT* x = input + (unsigned long long)row * input_stride + head * head_dim;
    const __nv_bfloat16* g = gate + (unsigned long long)row * gate_stride + head * head_dim;
    const float f = qwen4_nf8_load(x[i]);
    const float total = qwen4_nf8_block_sum(f * f, sm);
    const float rms = rsqrtf(qwen4_nf8_div_approx(total, (float)head_dim) + eps);
    const float w = __bfloat162float(weight[i]);
    const float gv = __bfloat162float(g[i]);
    return __bfloat162float(__float2bfloat16(qwen4_nf8_div_rn_1p_exp(f * rms * w, gv)));
}

template <typename InT>
__device__ __forceinline__ void qwen4_nf8_amax_body(
    const InT* input, const __nv_bfloat16* gate, const __nv_bfloat16* weight,
    unsigned int* amax_bits, unsigned int head_dim, float eps, unsigned int input_stride,
    unsigned int gate_stride) {
    __shared__ float sm[32];
    const float v = qwen4_nf8_value(input, gate, weight, head_dim, eps, input_stride,
                                    gate_stride, sm);
    const float m = qwen4_nf8_block_max(fabsf(v), sm);
    if (threadIdx.x == 0) atomicMax(amax_bits, __float_as_uint(m));
}

template <typename InT>
__device__ __forceinline__ void qwen4_nf8_quant_body(
    const InT* input, const __nv_bfloat16* gate, const __nv_bfloat16* weight,
    const unsigned int* amax_bits, __nv_fp8_e4m3* out, float* scale_out,
    unsigned int head_dim, float eps, unsigned int input_stride, unsigned int gate_stride,
    unsigned int output_stride) {
    __shared__ float sm[32];
    float amax;
    if (amax_bits != nullptr) {
        amax = fmaxf(__uint_as_float(amax_bits[0]), 1e-30f);
    } else {
        const float wmax = qwen4_nf8_block_max(fabsf(__bfloat162float(weight[threadIdx.x])), sm);
        amax = fmaxf(qwen4_nf8_sqrt_approx((float)head_dim) * wmax, 1e-30f);
    }
    const float inv = qwen4_nf8_div_448(amax);
    const float v = qwen4_nf8_value(input, gate, weight, head_dim, eps, input_stride,
                                    gate_stride, sm);
    out[(unsigned long long)blockIdx.y * output_stride + blockIdx.x * head_dim + threadIdx.x] =
        __nv_fp8_e4m3(v * inv);
    if (scale_out != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        scale_out[0] = amax * (1.0f / 448.0f);
    }
}

#define QWEN4_NF8_ENTRIES(SUFFIX, IN_T)                                                   \
    extern "C" __global__ void __launch_bounds__(128) qwen4_gdn_norm_amax_##SUFFIX(       \
        const IN_T* input, const __nv_bfloat16* gate, const __nv_bfloat16* weight,       \
        unsigned int* amax_bits, unsigned int head_dim, float eps,                       \
        unsigned int input_stride, unsigned int gate_stride) {                           \
        qwen4_nf8_amax_body<IN_T>(input, gate, weight, amax_bits, head_dim, eps,         \
                                  input_stride, gate_stride);                            \
    }                                                                                    \
    extern "C" __global__ void __launch_bounds__(128) qwen4_gdn_norm_fp8_##SUFFIX(        \
        const IN_T* input, const __nv_bfloat16* gate, const __nv_bfloat16* weight,       \
        const unsigned int* amax_bits, __nv_fp8_e4m3* out, float* scale_out,             \
        unsigned int head_dim, float eps, unsigned int input_stride,                     \
        unsigned int gate_stride, unsigned int output_stride) {                          \
        qwen4_nf8_quant_body<IN_T>(input, gate, weight, amax_bits, out, scale_out,      \
                                   head_dim, eps, input_stride, gate_stride,            \
                                   output_stride);                                       \
    }

QWEN4_NF8_ENTRIES(bf16, __nv_bfloat16)
QWEN4_NF8_ENTRIES(f16, __half)

// ---------------------------------------------------------------------------
// K11-G: Vectorized 2-heads/warp (16 heads = 2048 cols/CTA) GDN gated RMS norm + FP8 quant (`*_v2`).
// - Launch geometry: grid (3, rows_or_grid_y, 1), block (256, 1, 1) -> 8 warps = 16 heads per CTA
//   (24,576 CTAs at 8192 rows instead of 393,216 CTAs).
// - Each half-warp (q = lane & 15, sub = lane >> 4) processes one head of 128 elements:
//   each lane loads 8 contiguous elements i = 8 * q + c (c = 0..7) via 16-byte uint4 loads
//   (512B coalesced per warp) and writes 8 contiguous e4m3 bytes via an 8-byte uint2 store
//   (or 16B uint4 for BF16 in the Producer::GdnNorm twin).
// - Preserves v1's exact reduction tree (`qwen4_nf8_block_sum`):
//   v1 thread i has warp w' = i / 32 = q / 4 and lane l' = i % 32 = 8 * (q % 4) + c.
//   So v1's intra-warp shfl_xor(16, 8, 4, 2, 1) maps to shfl_xor(2, 1) across lanes,
//   followed by in-register (c ^ 4), (c ^ 2), (c ^ 1), and v1's inter-warp shfl_xor(2, 1)
//   maps to shfl_xor(8, 4) across lanes — zero shared memory, zero __syncthreads(), 0 ULP.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float qwen4_nf8_warp_sum8(float (&s)[8]) {
    #pragma unroll
    for (int off = 2; off >= 1; off >>= 1) {
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            s[c] += __shfl_xor_sync(0xffffffffu, s[c], off);
        }
    }
    #pragma unroll
    for (int c = 0; c < 4; ++c) s[c] += s[c + 4];
    #pragma unroll
    for (int c = 0; c < 2; ++c) s[c] += s[c + 2];
    float ws = s[0] + s[1];
    ws += __shfl_xor_sync(0xffffffffu, ws, 8);
    return ws + __shfl_xor_sync(0xffffffffu, ws, 4);
}

__device__ __forceinline__ float qwen4_nf8_halfwarp_max8(const float (&v)[8]) {
    float m0 = fmaxf(fmaxf(v[0], v[4]), fmaxf(v[2], v[6]));
    float m1 = fmaxf(fmaxf(v[1], v[5]), fmaxf(v[3], v[7]));
    float wm = fmaxf(m0, m1);
    #pragma unroll
    for (int off = 8; off >= 1; off >>= 1) {
        wm = fmaxf(wm, __shfl_xor_sync(0xffffffffu, wm, off));
    }
    return wm;
}

__device__ __forceinline__ float qwen4_nf8_warp_max32(float wm) {
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        wm = fmaxf(wm, __shfl_xor_sync(0xffffffffu, wm, off));
    }
    return wm;
}

template <typename InT>
__device__ __forceinline__ void qwen4_nf8_amax_v2_body(
    const InT* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    unsigned int* __restrict__ amax_bits,
    unsigned int head_dim,
    float eps,
    unsigned int input_stride,
    unsigned int gate_stride,
    unsigned int rows) {
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int sub = lane >> 4;
    const unsigned int q = lane & 15u;
    const unsigned int head = blockIdx.x * 16u + warp * 2u + sub;
    const unsigned int row = blockIdx.y;
    if (head >= 48u || head_dim != 128u || row >= rows) return;

    const InT* x = input + (unsigned long long)row * input_stride + head * 128u;
    const __nv_bfloat16* g = gate + (unsigned long long)row * gate_stride + head * 128u;
    const uint4 x_u4 = *reinterpret_cast<const uint4*>(x + q * 8u);
    const uint4 g_u4 = *reinterpret_cast<const uint4*>(g + q * 8u);
    const uint4 w_u4 = *reinterpret_cast<const uint4*>(weight + q * 8u);
    const InT* xp = reinterpret_cast<const InT*>(&x_u4);
    const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_u4);
    const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&w_u4);

    float f[8], gv[8], w[8], sq[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        f[c] = qwen4_nf8_load(xp[c]);
        gv[c] = __bfloat162float(gp[c]);
        w[c] = __bfloat162float(wp[c]);
        sq[c] = f[c] * f[c];
    }
    const float total = qwen4_nf8_warp_sum8(sq);
    const float rms = rsqrtf(total * (1.0f / 128.0f) + eps);
    float lane_max = 0.0f;
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        const float v = __bfloat162float(__float2bfloat16(
            qwen4_nf8_div_rn_1p_exp(f[c] * rms * w[c], gv[c])));
        lane_max = fmaxf(lane_max, fabsf(v));
    }
    const float warp_max = qwen4_nf8_warp_max32(lane_max);
    if (lane == 0u) atomicMax(amax_bits, __float_as_uint(warp_max));
}

__device__ __forceinline__ uint4 qwen4_nf8_ld_ef(const void* p, unsigned long long pol_ef) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p), "l"(pol_ef));
    return v;
}

__device__ __forceinline__ void qwen4_nf8_st_u2_el(void* p, uint2 v, unsigned long long pol_el) {
    asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                 :: "l"(p), "r"(v.x), "r"(v.y), "l"(pol_el) : "memory");
}

__device__ __forceinline__ void qwen4_nf8_st_u4_el(void* p, uint4 v, unsigned long long pol_el) {
    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(pol_el) : "memory");
}

__device__ __forceinline__ uint2 qwen4_nf8_pack8_e4m3(const float (&vs)[8]) {
    uint2 fp8_u2;
    asm("{\n\t"
        ".reg .b16 h0, h1, h2, h3;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h0, %3, %2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h1, %5, %4;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h2, %7, %6;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h3, %9, %8;\n\t"
        "mov.b32 %0, {h0, h1};\n\t"
        "mov.b32 %1, {h2, h3};\n\t"
        "}"
        : "=r"(fp8_u2.x), "=r"(fp8_u2.y)
        : "f"(vs[0]), "f"(vs[1]), "f"(vs[2]), "f"(vs[3]),
          "f"(vs[4]), "f"(vs[5]), "f"(vs[6]), "f"(vs[7]));
    return fp8_u2;
}

// Bitmask for per-row / per-strip L2 cache line discard (`discard.global.L2 [addr], 128`).
// Each bit is additionally guarded in-kernel by exact 128-byte base and row-stride alignment.
#define QWEN4_NF8_DISC_X   1u  // input (`gdn_out_strip` / `gdn_out`)
#define QWEN4_NF8_DISC_G   2u  // gate (`z` columns of `deinterleaved`)
#define QWEN4_NF8_DISC_QKV 4u  // `qkv_strip` scratch (strip entrypoint only)

template <typename InT, bool STRIP = false, unsigned int ROWS_PER_CTA = (STRIP ? 8u : 4u)>
__device__ __forceinline__ void qwen4_nf8_quant_v2_body(
    const InT* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    const unsigned int* __restrict__ amax_bits,
    __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ scale_out,
    unsigned int head_dim,
    float eps,
    unsigned int input_stride,
    unsigned int gate_stride,
    unsigned int output_stride,
    unsigned int rows,
    const unsigned char* __restrict__ qkv_discard = nullptr,
    unsigned int discard_flags = QWEN4_NF8_DISC_X) {
    static_assert(ROWS_PER_CTA >= 1u && ROWS_PER_CTA <= 8u, "ROWS_PER_CTA must be in 1..=8");
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int sub = lane >> 4;
    const unsigned int q = lane & 15u;
    const unsigned int head = blockIdx.x * 16u + warp * 2u + sub;
    // Reverse CTA order so CTAs first consume the last rows written by GDN FI while still hot in L2.
    const unsigned int row0 = (gridDim.y - 1u - blockIdx.y) * ROWS_PER_CTA;
    if (head >= 48u || head_dim != 128u || row0 >= rows) return;

    const unsigned int r_end = min(ROWS_PER_CTA, rows - row0);
    // Warp-uniform alignment and opt-out guards for 128-byte L2 discard safety.
    const bool x_aligned =
        ((reinterpret_cast<unsigned long long>(input) & 127ull) == 0ull)
        && (((input_stride * (unsigned int)sizeof(InT)) & 127u) == 0u);
    const bool can_disc_x =
        ((discard_flags & QWEN4_NF8_DISC_X) != 0u)
        && x_aligned
        && (STRIP || (row0 + 2048u >= rows));
    const bool can_disc_g = STRIP
        && ((discard_flags & QWEN4_NF8_DISC_G) != 0u)
        && ((reinterpret_cast<unsigned long long>(gate) & 127ull) == 0ull)
        && (((gate_stride * (unsigned int)sizeof(__nv_bfloat16)) & 127u) == 0u);
    const bool can_disc_qkv = STRIP
        && ((discard_flags & QWEN4_NF8_DISC_QKV) != 0u)
        && qkv_discard != nullptr
        && gridDim.x == 3u
        && ((reinterpret_cast<unsigned long long>(qkv_discard) & 127ull) == 0ull);

    if (can_disc_qkv) {
        // Each qkv_strip row is CONV_DIM=10240 FP16 = 20,480 B = 160 * 128 B.
        // Across blockIdx.x in {0, 1, 2} and threadIdx.x in 0..255, tid_grp spans 0..767,
        // bijectively discarding all r_end * 160 (<= 1280 for 8 rows/CTA) dead 128B lines
        // of [row0, row0 + r_end) in <= 2 passes of 768 lines.
        const unsigned int tid_grp = blockIdx.x * 256u + threadIdx.x;
        const unsigned int total_lines = r_end * 160u;
        if (tid_grp < total_lines) {
            const unsigned char* qd =
                qkv_discard + ((unsigned long long)row0 * 160ull + tid_grp) * 128ull;
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(qd) : "memory");
            if constexpr (ROWS_PER_CTA > 4u) {
                if (tid_grp + 768u < total_lines) {
                    const unsigned char* qd2 = qd + 768ull * 128ull;
                    asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(qd2) : "memory");
                }
            }
        }
    }

    unsigned long long pol_ef, pol_el;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));
    if (scale_out != nullptr && row0 < 2560u) {
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));
    } else {
        pol_el = pol_ef;
    }

    const uint4 w_u4 = *reinterpret_cast<const uint4*>(weight + q * 8u);
    const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&w_u4);
    float w[8], aw[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        w[c] = __bfloat162float(wp[c]);
        aw[c] = fabsf(w[c]);
    }

    float amax;
    if (amax_bits != nullptr) {
        amax = fmaxf(__uint_as_float(amax_bits[0]), 1e-30f);
    } else {
        const float wmax = qwen4_nf8_halfwarp_max8(aw);
        amax = fmaxf(11.31370849898476f * wmax, 1e-30f);
    }
    const float inv = qwen4_nf8_div_448(amax);
    if (scale_out != nullptr && blockIdx.x == 0u && row0 == 0u && threadIdx.x == 0u) {
        scale_out[0] = amax * (1.0f / 448.0f);
    }

    const bool disc_lane = (q == 0u || q == 8u);
    const InT* x_ptr = input + (unsigned long long)row0 * input_stride + head * 128u + q * 8u;
    const __nv_bfloat16* g_ptr = gate + (unsigned long long)row0 * gate_stride + head * 128u + q * 8u;
    __nv_fp8_e4m3* out_ptr = out + (unsigned long long)row0 * output_stride + head * 128u + q * 8u;

    uint4 x_cur = qwen4_nf8_ld_ef(x_ptr, pol_ef);
    uint4 g_cur = qwen4_nf8_ld_ef(g_ptr, pol_ef);

    #pragma unroll 1
    for (unsigned int r = 0u; r < r_end; ++r) {
        uint4 x_nxt, g_nxt;
        if (r + 1u < r_end) {
            x_nxt = qwen4_nf8_ld_ef(x_ptr + input_stride, pol_ef);
            g_nxt = qwen4_nf8_ld_ef(g_ptr + gate_stride, pol_ef);
        }
        const InT* xp = reinterpret_cast<const InT*>(&x_cur);
        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_cur);

        float f[8], gv[8], sq[8];
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            f[c] = qwen4_nf8_load(xp[c]);
            gv[c] = __bfloat162float(gp[c]);
            sq[c] = f[c] * f[c];
        }
        x_cur = x_nxt;
        g_cur = g_nxt;

        const float total = qwen4_nf8_warp_sum8(sq);
        if (can_disc_x) {
            // Every lane in the half-warp has loaded x_cur into registers and consumed it into
            // sq[0..7] before the warp-wide shfl.sync in qwen4_nf8_warp_sum8; __syncwarp()
            // anchors the 128-byte L2 discard strictly after all 8 lanes finish reading the line.
            __syncwarp();
            if (disc_lane) {
                asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(x_ptr) : "memory");
            }
        }
        const float rms = rsqrtf(total * (1.0f / 128.0f) + eps);
        float vs[8];
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const float v = __bfloat162float(__float2bfloat16(
                qwen4_nf8_div_rn_1p_exp(f[c] * rms * w[c], gv[c])));
            vs[c] = v * inv;
        }
        const uint2 fp8_u2 = qwen4_nf8_pack8_e4m3(vs);
        qwen4_nf8_st_u2_el(out_ptr, fp8_u2, pol_el);
        if (can_disc_g) {
            // Every lane has consumed g_cur into gv[0..7] -> fp8_u2 -> st.global before sync.
            __syncwarp();
            if (disc_lane) {
                asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(g_ptr) : "memory");
            }
        }
        x_ptr += input_stride;
        g_ptr += gate_stride;
        out_ptr += output_stride;
    }
}

// Producer::GdnNorm: writes BF16 normed output AND reduces tensor-wide absmax into `amax_bits`
// so `ssm out_proj` skips the standalone `qwen4_fp8_act_amax` pass (`-15.5 ms/8K` across 36 layers).
template <typename InT>
__device__ __forceinline__ void qwen4_gated_rms_norm_16in_amax_body(
    const InT* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int* __restrict__ amax_bits,
    unsigned int head_dim,
    float eps,
    unsigned int input_stride,
    unsigned int gate_stride,
    unsigned int output_stride) {
    __shared__ float sm[32];
    const float v = qwen4_nf8_value(input, gate, weight, head_dim, eps, input_stride,
                                    gate_stride, sm);
    output[(unsigned long long)blockIdx.y * output_stride + blockIdx.x * head_dim + threadIdx.x] =
        __float2bfloat16(v);
    const float m = qwen4_nf8_block_max(fabsf(v), sm);
    if (threadIdx.x == 0 && amax_bits != nullptr) atomicMax(amax_bits, __float_as_uint(m));
}

template <typename InT>
__device__ __forceinline__ void qwen4_gated_rms_norm_16in_amax_v2_body(
    const InT* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int* __restrict__ amax_bits,
    unsigned int head_dim,
    float eps,
    unsigned int input_stride,
    unsigned int gate_stride,
    unsigned int output_stride,
    unsigned int rows) {
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int sub = lane >> 4;
    const unsigned int q = lane & 15u;
    const unsigned int head = blockIdx.x * 16u + warp * 2u + sub;
    const unsigned int row = blockIdx.y;
    if (head >= 48u || head_dim != 128u || row >= rows) return;

    unsigned long long pol_ef, pol_el;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));

    const InT* x = input + (unsigned long long)row * input_stride + head * 128u;
    const __nv_bfloat16* g = gate + (unsigned long long)row * gate_stride + head * 128u;
    const uint4 x_u4 = qwen4_nf8_ld_ef(x + q * 8u, pol_ef);
    const uint4 g_u4 = qwen4_nf8_ld_ef(g + q * 8u, pol_ef);
    const uint4 w_u4 = *reinterpret_cast<const uint4*>(weight + q * 8u);
    const InT* xp = reinterpret_cast<const InT*>(&x_u4);
    const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_u4);
    const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&w_u4);

    float f[8], gv[8], w[8], sq[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        f[c] = qwen4_nf8_load(xp[c]);
        gv[c] = __bfloat162float(gp[c]);
        w[c] = __bfloat162float(wp[c]);
        sq[c] = f[c] * f[c];
    }
    const float total = qwen4_nf8_warp_sum8(sq);
    const float rms = rsqrtf(total * (1.0f / 128.0f) + eps);
    __nv_bfloat16 bf[8];
    float lane_max = 0.0f;
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        const __nv_bfloat16 b = __float2bfloat16(
            qwen4_nf8_div_rn_1p_exp(f[c] * rms * w[c], gv[c]));
        bf[c] = b;
        lane_max = fmaxf(lane_max, fabsf(__bfloat162float(b)));
    }
    qwen4_nf8_st_u4_el(
        output + (unsigned long long)row * output_stride + head * 128u + q * 8u,
        *reinterpret_cast<const uint4*>(bf),
        pol_el
    );
    if (amax_bits != nullptr) {
        const float warp_max = qwen4_nf8_warp_max32(lane_max);
        if (lane == 0u) atomicMax(amax_bits, __float_as_uint(warp_max));
    }
}

#define QWEN4_NF8_V2_ENTRIES(SUFFIX, IN_T)                                                \
    extern "C" __global__ void __launch_bounds__(256) qwen4_gdn_norm_amax_##SUFFIX##_v2(   \
        const IN_T* input, const __nv_bfloat16* gate, const __nv_bfloat16* weight,        \
        unsigned int* amax_bits, unsigned int head_dim, float eps,                        \
        unsigned int input_stride, unsigned int gate_stride, unsigned int rows) {         \
        qwen4_nf8_amax_v2_body<IN_T>(input, gate, weight, amax_bits, head_dim, eps,       \
                                     input_stride, gate_stride, rows);                    \
    }                                                                                     \
    extern "C" __global__ void __launch_bounds__(256, 5) qwen4_gdn_norm_fp8_##SUFFIX##_v2( \
        const IN_T* input, const __nv_bfloat16* gate, const __nv_bfloat16* weight,        \
        const unsigned int* amax_bits, __nv_fp8_e4m3* out, float* scale_out,              \
        unsigned int head_dim, float eps, unsigned int input_stride,                      \
        unsigned int gate_stride, unsigned int output_stride, unsigned int rows,          \
        unsigned int discard_flags) {                                                     \
        qwen4_nf8_quant_v2_body<IN_T, false>(input, gate, weight, amax_bits, out,         \
                                             scale_out, head_dim, eps, input_stride,      \
                                             gate_stride, output_stride, rows,            \
                                             nullptr, discard_flags);                     \
    }                                                                                     \
    extern "C" __global__ void __launch_bounds__(256, 5)                                  \
        qwen4_gdn_norm_fp8_##SUFFIX##_v2_strip(                                           \
            const IN_T* input, const __nv_bfloat16* gate, const __nv_bfloat16* weight,    \
            const unsigned int* amax_bits, __nv_fp8_e4m3* out, float* scale_out,          \
            unsigned int head_dim, float eps, unsigned int input_stride,                  \
            unsigned int gate_stride, unsigned int output_stride, unsigned int rows,      \
            const unsigned char* qkv_discard, unsigned int discard_flags) {               \
        qwen4_nf8_quant_v2_body<IN_T, true>(input, gate, weight, amax_bits, out,          \
                                            scale_out, head_dim, eps, input_stride,       \
                                            gate_stride, output_stride, rows,             \
                                            qkv_discard, discard_flags);                  \
    }                                                                                     \
    extern "C" __global__ void __launch_bounds__(128)                                     \
        qwen4_gated_rms_norm_##SUFFIX##in_amax(                                           \
            const IN_T* input, const __nv_bfloat16* gate, const __nv_bfloat16* weight,    \
            __nv_bfloat16* output, unsigned int* amax_bits, unsigned int head_dim,        \
            float eps, unsigned int input_stride, unsigned int gate_stride,               \
            unsigned int output_stride) {                                                 \
        qwen4_gated_rms_norm_16in_amax_body<IN_T>(input, gate, weight, output, amax_bits, \
                                                  head_dim, eps, input_stride,            \
                                                  gate_stride, output_stride);            \
    }                                                                                     \
    extern "C" __global__ void __launch_bounds__(256)                                     \
        qwen4_gated_rms_norm_##SUFFIX##in_amax_v2(                                        \
            const IN_T* input, const __nv_bfloat16* gate, const __nv_bfloat16* weight,    \
            __nv_bfloat16* output, unsigned int* amax_bits, unsigned int head_dim,        \
            float eps, unsigned int input_stride, unsigned int gate_stride,               \
            unsigned int output_stride, unsigned int rows) {                              \
        qwen4_gated_rms_norm_16in_amax_v2_body<IN_T>(input, gate, weight, output,         \
                                                     amax_bits, head_dim, eps,            \
                                                     input_stride, gate_stride,           \
                                                     output_stride, rows);                \
    }

QWEN4_NF8_V2_ENTRIES(bf16, __nv_bfloat16)
QWEN4_NF8_V2_ENTRIES(f16, __half)

// ---------------------------------------------------------------------------
// Wave 4B D2: Vectorized 2-heads/warp (16 heads = 2048 cols/CTA) GDN gated RMS norm
// fused directly with cuBLASLt NVFP4 (`CUDA_R_4F_E2M1` + `128x4` swizzled `VEC16_UE4M3`)
// activation quantization (`qwen4_gdn_norm_nvfp4_{bf16,f16}_v2_strip`).
//
// - Same launch geometry as `qwen4_gdn_norm_fp8_*_v2_strip`:
//   grid = (3, ceil(strip_rows / 8), 1), block = (256, 1, 1) -> 8 warps = 16 heads/CTA.
// - Each half-warp (q = lane & 15, sub = lane >> 4) owns one 128-element head (8 NVFP4
//   groups of 16 elements). Adjacent lanes `(q, q ^ 1)` hold the two 8-element halves of
//   each 16-element NVFP4 group and exchange `local_max` via a single `shfl_xor(..., 1)`.
// - Uses the exact analytic static upper bound `amax = sqrt(128) * max|weight|`:
//   `|v_i| <= sqrt(128) * max|w|` strictly for all inputs, so `scale2 = amax / 2688.0f`
//   makes every 16-element group scale fit UE4M3 (`<= 448.0f`) in a single pass with zero
//   absmax pass, zero FP8 intermediate buffer, and bit-identical NVFP4 output to
//   `qwen4_gated_rms_norm_*in_multi_seq -> qwen4_nvfp4_quant_bf16_128x4_from_amax` when given
//   the same analytic amax (differs from dynamic amax only in global scale2).
// - On the final strip (`row_offset + rows == total_rows`), CTA `blockIdx.y == gridDim.y - 1`
//   (`row0 == 0`) zeroes padded `128x4` scale rows `[total_rows, next_multiple_of(total_rows, 128))`
//   in-kernel.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float qwen4_nvfp4_mul_rn(float lhs, float rhs) {
    float result;
    asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(lhs), "f"(rhs));
    return result;
}

__device__ __forceinline__ float qwen4_nvfp4_rcp_approx_ftz(float value) {
    float result;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ unsigned int qwen4_nvfp4_cvt_f32x8_to_e2m1x8(
    const float (&values)[8],
    float scale
) {
    unsigned int result;
    const float v0 = qwen4_nvfp4_mul_rn(values[0], scale);
    const float v1 = qwen4_nvfp4_mul_rn(values[1], scale);
    const float v2 = qwen4_nvfp4_mul_rn(values[2], scale);
    const float v3 = qwen4_nvfp4_mul_rn(values[3], scale);
    const float v4 = qwen4_nvfp4_mul_rn(values[4], scale);
    const float v5 = qwen4_nvfp4_mul_rn(values[5], scale);
    const float v6 = qwen4_nvfp4_mul_rn(values[6], scale);
    const float v7 = qwen4_nvfp4_mul_rn(values[7], scale);
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

__device__ __forceinline__ unsigned long long qwen4_nvfp4_scale_offset_128x4(
    unsigned int row,
    unsigned int group,
    unsigned int k_tiles
) {
    const unsigned long long tile_idx =
        static_cast<unsigned long long>(row >> 7) * k_tiles + (group >> 2);
    const unsigned int r = row & 127u;
    return (tile_idx << 9) + ((r & 31u) << 4) + ((r >> 5) << 2) + (group & 3u);
}

template <typename InT, unsigned int ROWS_PER_CTA = 8u>
__device__ __forceinline__ void qwen4_nf4_quant_v2_strip_body(
    const InT* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float* __restrict__ alpha_out,
    float weight_scale_2,
    unsigned int head_dim,
    float eps,
    unsigned int input_stride,
    unsigned int gate_stride,
    unsigned int output_stride,
    unsigned int strip_rows,
    unsigned int row_offset,
    unsigned int total_rows,
    const unsigned char* __restrict__ qkv_discard,
    unsigned int discard_flags
) {
    static_assert(ROWS_PER_CTA >= 1u && ROWS_PER_CTA <= 8u, "ROWS_PER_CTA must be in 1..=8");
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int sub = lane >> 4;
    const unsigned int q = lane & 15u;
    const unsigned int head = blockIdx.x * 16u + warp * 2u + sub;
    // Reverse CTA order so CTAs consume the hottest tail rows written by GDN FI first.
    const unsigned int row0 = (gridDim.y - 1u - blockIdx.y) * ROWS_PER_CTA;
    if (head >= 48u || head_dim != 128u || row0 >= strip_rows) return;

    const unsigned int r_end = min(ROWS_PER_CTA, strip_rows - row0);
    const bool x_aligned =
        ((reinterpret_cast<unsigned long long>(input) & 127ull) == 0ull)
        && (((input_stride * (unsigned int)sizeof(InT)) & 127u) == 0u);
    const bool can_disc_x = ((discard_flags & QWEN4_NF8_DISC_X) != 0u) && x_aligned;
    const bool can_disc_g =
        ((discard_flags & QWEN4_NF8_DISC_G) != 0u)
        && ((reinterpret_cast<unsigned long long>(gate) & 127ull) == 0ull)
        && (((gate_stride * (unsigned int)sizeof(__nv_bfloat16)) & 127u) == 0u);
    const bool can_disc_qkv =
        ((discard_flags & QWEN4_NF8_DISC_QKV) != 0u)
        && qkv_discard != nullptr
        && gridDim.x == 3u
        && ((reinterpret_cast<unsigned long long>(qkv_discard) & 127ull) == 0ull);

    if (can_disc_qkv) {
        const unsigned int tid_grp = blockIdx.x * 256u + threadIdx.x;
        const unsigned int total_lines = r_end * 160u;
        if (tid_grp < total_lines) {
            const unsigned char* qd =
                qkv_discard + ((unsigned long long)row0 * 160ull + tid_grp) * 128ull;
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(qd) : "memory");
            if constexpr (ROWS_PER_CTA > 4u) {
                if (tid_grp + 768u < total_lines) {
                    const unsigned char* qd2 = qd + 768ull * 128ull;
                    asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(qd2) : "memory");
                }
            }
        }
    }

    unsigned long long pol_ef;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));

    const uint4 w_u4 = *reinterpret_cast<const uint4*>(weight + q * 8u);
    const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&w_u4);
    float w[8], aw[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        w[c] = __bfloat162float(wp[c]);
        aw[c] = fabsf(w[c]);
    }

    const float wmax = qwen4_nf8_halfwarp_max8(aw);
    const float amax = fmaxf(11.31370849898476f * wmax, 1e-30f);
    const float scale2 = __fdiv_rn(amax, 2688.0f);
    const float inv_scale2_over_6 = __fdiv_rn(448.0f, amax);
    if (alpha_out != nullptr && blockIdx.x == 0u && row0 == 0u && threadIdx.x == 0u) {
        alpha_out[0] = qwen4_nvfp4_mul_rn(scale2, weight_scale_2);
    }

    const unsigned int groups = output_stride >> 4;
    const unsigned int k_tiles = (groups + 3u) >> 2;
    const unsigned int group = head * 8u + (q >> 1);
    const unsigned long long half_out_stride = static_cast<unsigned long long>(output_stride >> 1);
    const bool disc_lane = (q == 0u || q == 8u);

    const InT* x_ptr = input + (unsigned long long)row0 * input_stride + head * 128u + q * 8u;
    const __nv_bfloat16* g_ptr = gate + (unsigned long long)row0 * gate_stride + head * 128u + q * 8u;
    unsigned char* p_out_ptr =
        packed_out + (unsigned long long)(row_offset + row0) * half_out_stride
        + (unsigned long long)head * 64ull + (unsigned long long)q * 4ull;

    uint4 x_cur = qwen4_nf8_ld_ef(x_ptr, pol_ef);
    uint4 g_cur = qwen4_nf8_ld_ef(g_ptr, pol_ef);

    #pragma unroll 1
    for (unsigned int r = 0u; r < r_end; ++r) {
        uint4 x_nxt, g_nxt;
        if (r + 1u < r_end) {
            x_nxt = qwen4_nf8_ld_ef(x_ptr + input_stride, pol_ef);
            g_nxt = qwen4_nf8_ld_ef(g_ptr + gate_stride, pol_ef);
        }
        const InT* xp = reinterpret_cast<const InT*>(&x_cur);
        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_cur);

        float f[8], gv[8], sq[8];
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            f[c] = qwen4_nf8_load(xp[c]);
            gv[c] = __bfloat162float(gp[c]);
            sq[c] = f[c] * f[c];
        }
        x_cur = x_nxt;
        g_cur = g_nxt;

        const float total = qwen4_nf8_warp_sum8(sq);
        if (can_disc_x) {
            __syncwarp();
            if (disc_lane) {
                asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(x_ptr) : "memory");
            }
        }
        const float rms = rsqrtf(total * (1.0f / 128.0f) + eps);
        float vs[8];
        float local_max = 0.0f;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const float v = __bfloat162float(__float2bfloat16(
                qwen4_nf8_div_rn_1p_exp(f[c] * rms * w[c], gv[c])));
            vs[c] = v;
            local_max = fmaxf(local_max, fabsf(v));
        }

        // Pair adjacent lanes (q, q ^ 1) to form the exact 16-element NVFP4 group absmax.
        const float partner_max = __shfl_xor_sync(0xffffffffu, local_max, 1);
        const float group_max = fmaxf(local_max, partner_max);
        const float fp8_float =
            (group_max > 0.0f) ? fminf(qwen4_nvfp4_mul_rn(group_max, inv_scale2_over_6), 448.0f) : 0.0f;
        const __nv_fp8_e4m3 encoded_scale = __nv_fp8_e4m3(fp8_float);
        const float decoded = static_cast<float>(encoded_scale);
        const float effective_scale = qwen4_nvfp4_mul_rn(decoded, scale2);
        const float inv_eff =
            (effective_scale > 0.0f) ? qwen4_nvfp4_rcp_approx_ftz(effective_scale) : 0.0f;

        const unsigned int packed4 = qwen4_nvfp4_cvt_f32x8_to_e2m1x8(vs, inv_eff);
        *reinterpret_cast<unsigned int*>(p_out_ptr) = packed4;

        // Pack 4 adjacent group scales (q = 0,2,4,6 or q = 8,10,12,14) into one 4-byte aligned
        // 128x4 swizzled store per k_tile (4x fewer scale stores than byte-wise writes).
        const unsigned int s0 = static_cast<unsigned int>(encoded_scale.__x);
        const unsigned int s1 = __shfl_xor_sync(0xffffffffu, s0, 2);
        const unsigned int s2 = __shfl_xor_sync(0xffffffffu, s0, 4);
        const unsigned int s3 = __shfl_xor_sync(0xffffffffu, s1, 4);
        if ((q & 7u) == 0u) {
            const unsigned int abs_row = row_offset + row0 + r;
            const unsigned int sf4 = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
            *reinterpret_cast<unsigned int*>(
                scale_out + qwen4_nvfp4_scale_offset_128x4(abs_row, group, k_tiles)) = sf4;
        }

        if (can_disc_g) {
            __syncwarp();
            if (disc_lane) {
                asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(g_ptr) : "memory");
            }
        }
        x_ptr += input_stride;
        g_ptr += gate_stride;
        p_out_ptr += half_out_stride;
    }

    // On the final strip, zero padded 128x4 scale rows `[total_rows, padded_rows)` so cuBLASLt
    // TMA loads never see uninitialized bytes in the trailing partial 128-row scale tile.
    if (row_offset + strip_rows == total_rows && row0 == 0u && (q & 7u) == 0u) {
        const unsigned int padded_rows = ((total_rows + 127u) >> 7) << 7;
        for (unsigned int pad_row = total_rows; pad_row < padded_rows; ++pad_row) {
            *reinterpret_cast<unsigned int*>(
                scale_out + qwen4_nvfp4_scale_offset_128x4(pad_row, group, k_tiles)) = 0u;
        }
    }
}

#define QWEN4_NF4_V2_STRIP_ENTRY(SUFFIX, IN_T)                                                \
    extern "C" __global__ void __launch_bounds__(256, 4)                                      \
        qwen4_gdn_norm_nvfp4_##SUFFIX##_v2_strip(                                             \
            const IN_T* input, const __nv_bfloat16* gate, const __nv_bfloat16* weight,        \
            unsigned char* packed_out, unsigned char* scale_out, float* alpha_out,            \
            float weight_scale_2, unsigned int head_dim, float eps,                           \
            unsigned int input_stride, unsigned int gate_stride, unsigned int output_stride,  \
            unsigned int strip_rows, unsigned int row_offset, unsigned int total_rows,        \
            const unsigned char* qkv_discard, unsigned int discard_flags) {                   \
        qwen4_nf4_quant_v2_strip_body<IN_T>(                                                  \
            input, gate, weight, packed_out, scale_out, alpha_out, weight_scale_2, head_dim,  \
            eps, input_stride, gate_stride, output_stride, strip_rows, row_offset,            \
            total_rows, qkv_discard, discard_flags);                                          \
    }

QWEN4_NF4_V2_STRIP_ENTRY(bf16, __nv_bfloat16)
QWEN4_NF4_V2_STRIP_ENTRY(f16, __half)
