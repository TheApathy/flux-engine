// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// Shared low-level PTX / FP4 / FP8 / BF16 / MMA / async-copy helpers for
// Blackwell NVFP4 MoE kernels (`qwen4_moe_thin_sk_v33`, `qwen4_moe_w4a4_v32`).

namespace qwen4_nvfp4_detail {

__device__ __forceinline__ unsigned prmt_b32(unsigned a, unsigned b, unsigned sel) {
    unsigned d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void cp16_cg(void* smem_ptr, const void* gmem_ptr) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_ptr));
}

__device__ __forceinline__ void ldsm4(
    unsigned smem_addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3
) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(smem_addr));
}

__device__ __forceinline__ void ldsm4_trans(
    unsigned smem_addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3
) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(smem_addr));
}

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ unsigned pack_bf162(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    unsigned u;
    memcpy(&u, &v, sizeof(u));
    return u;
}

__device__ __forceinline__ unsigned bf162_to_f162_clamped(unsigned bf2) {
    const float f0 = fminf(fmaxf(__uint_as_float(bf2 << 16), -65504.0f), 65504.0f);
    const float f1 = fminf(fmaxf(__uint_as_float(bf2 & 0xffff0000u), -65504.0f), 65504.0f);
    const __half2 h = __floats2half2_rn(f0, f1);
    unsigned u;
    memcpy(&u, &h, sizeof(u));
    return u;
}

__device__ __forceinline__ unsigned char cvt_f32_e4m3(float v) {
    unsigned short p = 0u;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p) : "f"(v), "f"(0.0f));
    return static_cast<unsigned char>(p & 0xffu);
}

__device__ __forceinline__ unsigned pack_e4m3x4_ptx(float a, float b, float c, float d) {
    unsigned short p0, p1;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p0) : "f"(a), "f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p1) : "f"(c), "f"(d));
    return static_cast<unsigned>(p0) | (static_cast<unsigned>(p1) << 16);
}

__device__ __forceinline__ unsigned pack_e4m3x4_nv(float a, float b, float c, float d) {
    const unsigned b0 = static_cast<unsigned>(__nv_cvt_float_to_fp8(a, __NV_SATFINITE, __NV_E4M3));
    const unsigned b1 = static_cast<unsigned>(__nv_cvt_float_to_fp8(b, __NV_SATFINITE, __NV_E4M3));
    const unsigned b2 = static_cast<unsigned>(__nv_cvt_float_to_fp8(c, __NV_SATFINITE, __NV_E4M3));
    const unsigned b3 = static_cast<unsigned>(__nv_cvt_float_to_fp8(d, __NV_SATFINITE, __NV_E4M3));
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

__device__ __forceinline__ float e4m3_to_float(unsigned b) {
    if ((b & 0x7fu) == 0x7fu) return 0.0f;
    unsigned h;
    const unsigned short pair = static_cast<unsigned short>(b & 0xffu);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(pair));
    return __low2float(*reinterpret_cast<const __half2*>(&h));
}

// Unbiased round-to-nearest UE4M3 conversion (sign bit masked to 0).
__device__ __forceinline__ unsigned rn_ue4m3(float sf_f) {
    return static_cast<unsigned>(__nv_cvt_float_to_fp8(sf_f, __NV_SATFINITE, __NV_E4M3)) & 0x7fu;
}

// Hardware SM120/SM121 E2M1x2 (single byte -> __half2 {w_even, w_odd}).
__device__ __forceinline__ unsigned e2m1x2_to_h2(unsigned byte_val) {
#if defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || \
    defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_SPECIFIC__)
    unsigned hb;
    const unsigned short b16 = static_cast<unsigned short>(byte_val & 0xffu);
    asm("{ .reg .b8 bx, bz; mov.b16 {bx, bz}, %1; cvt.rn.f16x2.e2m1x2 %0, bx; }"
        : "=r"(hb) : "h"(b16));
    return hb;
#else
    static constexpr float LUT[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    const __half2 h = __floats2half2_rn(LUT[byte_val & 15u], LUT[(byte_val >> 4) & 15u]);
    unsigned u;
    memcpy(&u, &h, sizeof(u));
    return u;
#endif
}

// Hardware SM120/SM121 E4M3 single byte -> broadcast __half2 {s, s}.
__device__ __forceinline__ __half2 e4m3_to_h2_bcast(unsigned s_byte) {
    unsigned hs;
    const unsigned short sb2 = static_cast<unsigned short>((s_byte & 0xffu) | ((s_byte & 0xffu) << 8));
    asm("{ .reg .b16 hsb; mov.b16 hsb, %1; cvt.rn.f16x2.e4m3x2 %0, hsb; }"
        : "=r"(hs) : "h"(sb2));
    return *reinterpret_cast<const __half2*>(&hs);
}

// Hardware SM120/SM121 E4M3x4 -> 2 packed __half2 pairs {s0,s1} and {s2,s3}.
__device__ __forceinline__ void decode_e4m3x4_to_h2_pair(
    unsigned s4, __half2& h01, __half2& h23
) {
    unsigned hs01, hs23;
    asm("{ .reg .b16 s01, s23;\n"
        "  mov.b32 {s01, s23}, %2;\n"
        "  cvt.rn.f16x2.e4m3x2 %0, s01;\n"
        "  cvt.rn.f16x2.e4m3x2 %1, s23; }\n"
        : "=r"(hs01), "=r"(hs23)
        : "r"(s4));
    h01 = *reinterpret_cast<const __half2*>(&hs01);
    h23 = *reinterpret_cast<const __half2*>(&hs23);
}

// Hardware SM120/SM121 E2M1x4 (2 bytes from low or high half of packed4) -> 2 __half2 pairs.
template <bool HI_HALF>
__device__ __forceinline__ void decode_e2m1x4_half_to_h2(
    unsigned packed4, unsigned& h0, unsigned& h1
) {
#if defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || \
    defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_SPECIFIC__)
    if constexpr (!HI_HALF) {
        asm("{ .reg .b16 lo16, hi16; .reg .b8 b0, b1;\n"
            "  mov.b32 {lo16, hi16}, %2;\n"
            "  mov.b16 {b0, b1}, lo16;\n"
            "  cvt.rn.f16x2.e2m1x2 %0, b0;\n"
            "  cvt.rn.f16x2.e2m1x2 %1, b1; }\n"
            : "=r"(h0), "=r"(h1) : "r"(packed4));
    } else {
        asm("{ .reg .b16 lo16, hi16; .reg .b8 b2, b3;\n"
            "  mov.b32 {lo16, hi16}, %2;\n"
            "  mov.b16 {b2, b3}, hi16;\n"
            "  cvt.rn.f16x2.e2m1x2 %0, b2;\n"
            "  cvt.rn.f16x2.e2m1x2 %1, b3; }\n"
            : "=r"(h0), "=r"(h1) : "r"(packed4));
    }
#else
    const unsigned shift = HI_HALF ? 16u : 0u;
    h0 = e2m1x2_to_h2((packed4 >> shift) & 0xffu);
    h1 = e2m1x2_to_h2((packed4 >> (shift + 8u)) & 0xffu);
#endif
}

__device__ __forceinline__ unsigned cvt_f32x8_to_e2m1x8(const float (&v)[8], float inv_eff) {
    const float x0 = __fmul_rn(v[0], inv_eff);
    const float x1 = __fmul_rn(v[1], inv_eff);
    const float x2 = __fmul_rn(v[2], inv_eff);
    const float x3 = __fmul_rn(v[3], inv_eff);
    const float x4 = __fmul_rn(v[4], inv_eff);
    const float x5 = __fmul_rn(v[5], inv_eff);
    const float x6 = __fmul_rn(v[6], inv_eff);
    const float x7 = __fmul_rn(v[7], inv_eff);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    unsigned result;
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "}\n"
        : "=r"(result)
        : "f"(x0), "f"(x1), "f"(x2), "f"(x3), "f"(x4), "f"(x5), "f"(x6), "f"(x7));
    return result;
#else
    auto q_nib = [](float x) -> unsigned {
        const float a = fabsf(x);
        const unsigned s = (x < 0.0f) ? 8u : 0u;
        unsigned idx;
        if      (a < 0.25f) idx = 0u;
        else if (a <= 0.75f) idx = 1u;
        else if (a < 1.25f) idx = 2u;
        else if (a <= 1.75f) idx = 3u;
        else if (a < 2.5f) idx = 4u;
        else if (a <= 3.5f) idx = 5u;
        else if (a < 5.0f) idx = 6u;
        else idx = 7u;
        return s | idx;
    };
    const unsigned b0 = q_nib(x0) | (q_nib(x1) << 4);
    const unsigned b1 = q_nib(x2) | (q_nib(x3) << 4);
    const unsigned b2 = q_nib(x4) | (q_nib(x5) << 4);
    const unsigned b3 = q_nib(x6) | (q_nib(x7) << 4);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
#endif
}

__device__ __forceinline__ void mma_f16(
    float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Native Blackwell W4A4 block-scaled NVFP4 MMA:
//   D[16x8] += (A[16x64] * SFA[16x4]) x (B[8x64]^T * SFB[8x4])
__device__ __forceinline__ void mma_nvfp4(
    float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1,
    unsigned sfa, unsigned sfb
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
#else
    (void)d; (void)a; (void)b0; (void)b1; (void)sfa; (void)sfb;
    __trap();
#endif
}

// 4x4 word transpose across `q4 = (lane >> 2) & 3` (exchanges lanes ^4 and ^8).
__device__ __forceinline__ void transpose_4x4_words(
    uint4 raw, unsigned q4, unsigned& p0, unsigned& p1, unsigned& p2, unsigned& p3
) {
    const bool odd1 = (q4 & 1u) != 0u;
    const unsigned s0 = __shfl_xor_sync(0xffffffffu, odd1 ? raw.x : raw.y, 4);
    const unsigned s2 = __shfl_xor_sync(0xffffffffu, odd1 ? raw.z : raw.w, 4);
    const unsigned t0 = odd1 ? s0 : raw.x;
    const unsigned t1 = odd1 ? raw.y : s0;
    const unsigned t2 = odd1 ? s2 : raw.z;
    const unsigned t3 = odd1 ? raw.w : s2;
    const bool odd2 = (q4 & 2u) != 0u;
    const unsigned u0 = __shfl_xor_sync(0xffffffffu, odd2 ? t0 : t2, 8);
    const unsigned u1 = __shfl_xor_sync(0xffffffffu, odd2 ? t1 : t3, 8);
    p0 = odd2 ? u0 : t0;
    p1 = odd2 ? u1 : t1;
    p2 = odd2 ? t2 : u0;
    p3 = odd2 ? t3 : u1;
}

}  // namespace qwen4_nvfp4_detail
