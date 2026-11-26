// SPDX-License-Identifier: AGPL-3.0-only
//
// V32_W4A4: Native Blackwell W4A4 (NVFP4 x NVFP4) persistent MoE prefill kernels
// for Flash-Next (H=2560, I=640, E=512, top_k=10).
//
// Architecture highlights:
// - Native PTX `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.
//   m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3` (2x peak Tensor Core throughput
//   vs FP8 m16n8k32, zero runtime E2M1->E4M3 transcode, zero sBt, zero LUT).
// - K_STEP = 128 double-buffered async pipeline (DA=2, DB=2 -> 32 back-to-back
//   OMMAs per K_STEP=128 stage, 2 `__syncthreads()` per stage), direct
//   `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` from 128B-swizzled
//   `sB[2][2][32][128]` (ZERO `sBk`, 0 bank conflicts) + 0-conflict `LDS.128`
//   scale reads from pre-packed `sSF[2][2][128]`.
// - 256 threads (8 warps = 4 M x 2 N), TILE_M=128, BC=128 (TN=64 GATEUP /
//   TN=128 DOWN), <= 40.63 KiB (GATEUP) / <= 44.31 KiB (DOWN) dynamic SMEM ->
//   guaranteed 2 CTAs/SM (376 resident CTAs on 188-SM SM120, 96 on 48-SM GB10)
//   with <= 128 regs/thread and 0 register spills.
// - Race-free self-resetting atomic work-stealing via `workspace->next_task_gu`
//   and `workspace->next_task_dn`, plus next-task L2 weight prefetch.
// - Single-read fused BF16->NVFP4 + shared-gate dot quantizer
//   (`qwen4_nvfp4_quant_rows_v32_gate`) with per-row `x_scale2`.
// - Two GATEUP epilogue modes (`OQ_NVFP4 = false`: exact MXFP8 `oq`/`os` for
//   hybrid `gateup` mode feeding V31_WS DOWN; `OQ_NVFP4 = true`: direct NVFP4
//   `oq`/`os` for full `all` W4A4 mode feeding `qwen4_moe_w4a4_down_v32`).
// - Both flat UE4M3 and Atlas-TS4E-128 compressed weight-scale layouts.

#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define OI640_STEP_K 32
#define OI640_TRANSPOSED 1
#define flash_next_orig_i640_prefill flash_next_w4a4_v32
#include "moe_w4a16_orig_i640_compact_prefill.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace oi640 = flash_next_w4a4_v32;

namespace {

constexpr unsigned V32_TILE_M = 128u;
constexpr unsigned V32_BC = 128u;
constexpr unsigned V32_THREADS = 256u;
constexpr unsigned V32_K_STEP = 128u;
constexpr unsigned V32_STAGES = 2u;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1000)
constexpr unsigned V32_XQ_PIN_MAX_ROWS = 20480u;
constexpr unsigned V32_OQ_NVFP4_PIN_MAX_ROWS = 20480u;
constexpr unsigned V32_OQ_MXFP8_KEEP_WINDOW = 960u;
constexpr unsigned long long V32_KEEP_SLOT_BYTES = 64ull << 20;
#else
constexpr unsigned V32_XQ_PIN_MAX_ROWS = 8192u;
constexpr unsigned V32_OQ_NVFP4_PIN_MAX_ROWS = 4096u;
constexpr unsigned V32_OQ_MXFP8_KEEP_WINDOW = 256u;
constexpr unsigned long long V32_KEEP_SLOT_BYTES = 12ull << 20;
#endif

template <bool GATEUP, bool TS4E> struct V32W4a4Smem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V32_BC / BMAT; // 64 (GU) or 128 (DN)
    static constexpr unsigned TS4E_KS = (GATEUP ? oi640::HIDDEN : oi640::INTER) / V32_K_STEP;
    static constexpr unsigned TS4E_B_PER_KS = GATEUP ? 8u : 4u; // 4 k32-steps x BMAT
    static constexpr unsigned TS4E_BKTS = TS4E_KS * TS4E_B_PER_KS;
    static constexpr unsigned TS4E_EXC_CAP = 32u;

    static constexpr unsigned A = V32_STAGES * 2u * V32_TILE_M * 32u;  // 16,384 B ([2][2][128][32])
    static constexpr unsigned AS = V32_STAGES * V32_TILE_M * 8u;       //  2,048 B ([2][128][8])
    static constexpr unsigned B = V32_STAGES * 2u * 32u * V32_BC;      // 16,384 B ([2][2][32][128])
    static constexpr unsigned S = V32_STAGES * 2u * 4u * V32_BC;       //  2,048 B ([2][2][4][128])
    static constexpr unsigned SF = V32_STAGES * 2u * V32_BC * 4u;      //  2,048 B ([2][2][128] u32)
    static constexpr unsigned XS2 = GATEUP ? (V32_TILE_M * 4u) : (11u * V32_TILE_M * 4u); // 512 B (GU) / 5,632 B (DN: 10 ratio + 1 final_s2)
    static constexpr unsigned PTRS = 8u * 8u;                          //     64 B (sPtrs[0..5], sTask[2])

    static constexpr unsigned OFF_AS = A;
    static constexpr unsigned OFF_B = OFF_AS + AS;
    static constexpr unsigned OFF_S = OFF_B + B;
    static constexpr unsigned OFF_SF = OFF_S + S;
    static constexpr unsigned OFF_XS2 = OFF_SF + SF;
    static constexpr unsigned OFF_PTRS = OFF_XS2 + XS2;
    static constexpr unsigned OFF_TASK = OFF_PTRS + 48u;
    static constexpr unsigned OFF_BASE = OFF_PTRS + PTRS;
    static constexpr unsigned OFF_EXC = OFF_BASE + (TS4E ? V32_BC : 0u);
    static constexpr unsigned OFF_BKT_BEG =
        OFF_EXC + (TS4E ? (V32_STAGES * TS4E_B_PER_KS * TS4E_EXC_CAP * 2u) : 0u);
    static constexpr unsigned OFF_BKT_CNT = OFF_BKT_BEG + (TS4E ? (TS4E_BKTS * 4u) : 0u);
    static constexpr unsigned RAW_BYTES = OFF_BKT_CNT + (TS4E ? (TS4E_BKTS * 2u) : 0u);
    static constexpr unsigned BYTES = (RAW_BYTES + 15u) & ~15u;

    static_assert(
        !TS4E ? (GATEUP ? (BYTES == 39488u) : (BYTES == 44608u))
              : (GATEUP ? (BYTES == 41600u) : (BYTES == 45376u)),
        "V32W4a4Smem byte contract"
    );
    static_assert(BYTES <= 49152u, "V32W4a4Smem must stay <= 48 KiB for 2 CTAs/SM");
};

// 32-byte row swizzle for sA (two 16-byte chunks per row): zero bank conflicts
// for both cp.async.16 writes and ldmatrix.m8n8.x4 reads.
__device__ __forceinline__ unsigned v32_swz32(unsigned row, unsigned chunk) {
    return row * 32u + ((chunk ^ ((row >> 2) & 1u)) << 4);
}

__device__ __forceinline__ unsigned v32_prmt(unsigned a, unsigned b, unsigned sel) {
    unsigned d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
}

__device__ __forceinline__ void v32_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void v32_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void v32_cp16_pol_s32(unsigned s_addr, const void* gmem_ptr,
                                                 unsigned long long pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, 16, %2;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "l"(pol));
}

__device__ __forceinline__ void v32_cp16_pol_pred_s32(unsigned s_addr, const void* gmem_ptr, bool pred,
                                                      unsigned long long pol) {
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "r"(bytes), "l"(pol));
}

__device__ __forceinline__ void v32_cp16_s32(unsigned s_addr, const void* gmem_ptr, bool pred) {
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "r"(bytes));
}

__device__ __forceinline__ void v32_cp8_s32(unsigned s_addr, const void* gmem_ptr, bool pred) {
    const int bytes = pred ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "r"(bytes));
}

__device__ __forceinline__ void v32_ldsm4(unsigned addr, unsigned& r0, unsigned& r1,
                                          unsigned& r2, unsigned& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void v32_ldsm4_trans(unsigned addr, unsigned& r0, unsigned& r1,
                                                unsigned& r2, unsigned& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ unsigned v32_ld_su32(unsigned s_addr) {
    unsigned v;
    asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(s_addr));
    return v;
}

__device__ __forceinline__ uint4 v32_ld_sv4(unsigned s_addr) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(s_addr));
    return v;
}

__device__ __forceinline__ void v32_st_su32(unsigned s_addr, unsigned v) {
    asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(s_addr), "r"(v) : "memory");
}

__device__ __forceinline__ const unsigned char* v32_ld_sptr(unsigned ptrs_base, unsigned idx) {
    unsigned long long p;
    asm volatile("ld.shared.u64 %0, [%1];\n" : "=l"(p) : "r"(ptrs_base + idx * 8u));
    return reinterpret_cast<const unsigned char*>(p);
}

__device__ __forceinline__ float v32_ld_sxs2(unsigned xs2_base, unsigned row) {
    float v;
    asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(v) : "r"(xs2_base + row * 4u));
    return v;
}

__device__ __forceinline__ void v32_prefetch_l2(const void* gmem_ptr) {
    asm volatile("prefetch.global.L2 [%0];\n" :: "l"(gmem_ptr));
}

__device__ __forceinline__ void v32_st4_l2_el(void* dst, unsigned v, unsigned long long pol_el) {
    asm volatile("st.global.L2::cache_hint.u32 [%0], %1, %2;\n"
                 :: "l"(dst), "r"(v), "l"(pol_el) : "memory");
}

__device__ __forceinline__ void v32_st4_ef(void* dst, unsigned v) {
    asm volatile("st.global.cs.u32 [%0], %1;\n" :: "l"(dst), "r"(v) : "memory");
}

__device__ __forceinline__ void v32_st8_l2_el(void* dst, uint2 v, unsigned long long pol_el) {
    asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y), "l"(pol_el) : "memory");
}

__device__ __forceinline__ void v32_st8_ef(void* dst, uint2 v) {
    asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y) : "memory");
}

__device__ __forceinline__ void v32_st16_stream(void* dst, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ float v32_e4m3_to_float(unsigned b) {
    if ((b & 0x7fu) == 0x7fu) return 0.0f;
    unsigned h;
    const unsigned short pair = static_cast<unsigned short>(b & 0xffu);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(pair));
    return __low2float(*reinterpret_cast<const __half2*>(&h));
}

// Unbiased round-to-nearest UE4M3 conversion (sign bit masked to 0).
// Note: ceiling UE4M3 introduces up to +12.5% one-sided positive amplification
// bias on E2M1 peak elements (4.0..6.0 midpoint 5.0) across 96 quantize stages;
// round-to-nearest matches NVIDIA ModelOpt NVFP4 activation quantization and
// preserves 100% exact needle recall across 4K..32K contexts.
__device__ __forceinline__ unsigned v32_rn_ue4m3(float sf_f) {
    return static_cast<unsigned>(__nv_cvt_float_to_fp8(sf_f, __NV_SATFINITE, __NV_E4M3)) & 0x7fu;
}

__device__ __forceinline__ unsigned v32_pack_e4m3x4(float a, float b, float c, float d) {
    const unsigned b0 = static_cast<unsigned>(__nv_cvt_float_to_fp8(a, __NV_SATFINITE, __NV_E4M3));
    const unsigned b1 = static_cast<unsigned>(__nv_cvt_float_to_fp8(b, __NV_SATFINITE, __NV_E4M3));
    const unsigned b2 = static_cast<unsigned>(__nv_cvt_float_to_fp8(c, __NV_SATFINITE, __NV_E4M3));
    const unsigned b3 = static_cast<unsigned>(__nv_cvt_float_to_fp8(d, __NV_SATFINITE, __NV_E4M3));
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

__device__ __forceinline__ unsigned v32_pack_bf162(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    unsigned u;
    memcpy(&u, &v, sizeof(u));
    return u;
}

__device__ __forceinline__ unsigned v32_cvt_f32x8_to_e2m1x8(const float (&v)[8], float inv_eff) {
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

// Native Blackwell W4A4 block-scaled NVFP4 MMA:
//   D[16x8] += (A[16x64] * SFA[16x4]) x (B[8x64]^T * SFB[8x4])
//   A, B: packed E2M1 nibbles; SFA, SFB: 4x UE4M3 block-16 scales packed in u32.
__device__ __forceinline__ void v32_mma_nvfp4(
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

// Conflict-free sSF u32 index for even/odd paired LDS.128 reads.
// Layout: sSF[sl][kh][wn (0..1)][half (0..1)][gid (0..7)][4 u32: p0_e, p0_o, p1_e, p1_o].
template <bool GATEUP>
__device__ __forceinline__ unsigned v32_sf_eo_u32_idx(unsigned col) {
    if constexpr (GATEUP) {
        const unsigned mtx = col >> 6;
        const unsigned lc = col & 63u;
        const unsigned w = (lc >> 5) & 1u;
        const unsigned p = (lc >> 4) & 1u;
        const unsigned g = (lc >> 1) & 7u;
        const unsigned eo = lc & 1u;
        return ((((w * 2u + mtx) * 8u) + g) << 2) + (p << 1) + eo;
    } else {
        const unsigned w = col >> 6;
        const unsigned h = (col >> 5) & 1u;
        const unsigned p = (col >> 4) & 1u;
        const unsigned g = (col >> 1) & 7u;
        const unsigned eo = col & 1u;
        return ((((w * 2u + h) * 8u) + g) << 2) + (p << 1) + eo;
    }
}

// Register-direct 4-lane rearrange for even/odd paired MMA outputs (0 SMEM).
__device__ __forceinline__ uint4 v32_bf16_reg_quad_eo(
    unsigned u0, unsigned u1, unsigned u2, unsigned u3, unsigned tig
) {
    const bool hi2 = (tig & 2u) != 0u;
    const unsigned s0 = __shfl_xor_sync(0xffffffffu, hi2 ? u0 : u2, 2);
    const unsigned s1 = __shfl_xor_sync(0xffffffffu, hi2 ? u1 : u3, 2);
    const unsigned t0 = hi2 ? s0 : u0;
    const unsigned t1 = hi2 ? s1 : u1;
    const unsigned t2 = hi2 ? u2 : s0;
    const unsigned t3 = hi2 ? u3 : s1;
    const bool odd1 = (tig & 1u) != 0u;
    const unsigned r0 = __shfl_xor_sync(0xffffffffu, odd1 ? t0 : t2, 1);
    const unsigned r1 = __shfl_xor_sync(0xffffffffu, odd1 ? t1 : t3, 1);
    return make_uint4(
        odd1 ? r0 : t0,
        odd1 ? r1 : t1,
        odd1 ? t2 : r0,
        odd1 ? t3 : r1);
}

// Quad-lane MXFP8 quantizer (bit-identical to v28_mxfp8_reg_quad / v16q4).
__device__ __forceinline__ void v32_mxfp8_reg_quad(
    unsigned u0, unsigned u1, unsigned u2, unsigned u3,
    unsigned tig, unsigned (&qq)[2], unsigned& ss
) {
    const uint4 w = v32_bf16_reg_quad_eo(u0, u1, u2, u3, tig);
    const unsigned u[4] = {w.x, w.y, w.z, w.w};
    float v[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[2 * i] = __uint_as_float(u[i] << 16);
        v[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
    }
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    qq[0] = 0u;
    qq[1] = 0u;
    ss = 0u;
    if (amax > 0.0f) {
        int e;
        const float m = frexpf(__fmul_rn(amax, 1.0f / 448.0f), &e);
        int k = (m == 0.5f) ? e - 1 : e;
        k = max(-127, min(127, k));
        const float inv = __int_as_float((127 - k) << 23);
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            qq[i] = v32_pack_e4m3x4(
                v[4 * i + 0] * inv,
                v[4 * i + 1] * inv,
                v[4 * i + 2] * inv,
                v[4 * i + 3] * inv);
        }
        ss = static_cast<unsigned>(k + 127);
    }
}

template <bool GATEUP, bool OQ_NVFP4, int EPI, bool PREFETCH, bool TS4E>
__device__ __forceinline__ void v32_w4a4_impl(
    const unsigned char* __restrict__ aq,
    const unsigned char* __restrict__ as,
    const float* __restrict__ x_scale2,
    const unsigned long long* __restrict__ pk0,
    const unsigned long long* __restrict__ sc0,
    const float* __restrict__ s20,
    const unsigned long long* __restrict__ pk1,
    const unsigned long long* __restrict__ sc1,
    const float* __restrict__ s21,
    __nv_bfloat16* __restrict__ C,
    unsigned char* __restrict__ oq,
    unsigned char* __restrict__ os,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned N_in,
    unsigned K_in,
    float oq_scale2,
    float* __restrict__ oq_scale2_out = nullptr,
    const float* __restrict__ oq_scale2_in = nullptr,
    unsigned char* __restrict__ slot_q = nullptr,
    unsigned char* __restrict__ slot_s = nullptr
) {
    using L = V32W4a4Smem<GATEUP, TS4E>;
    constexpr unsigned N = GATEUP ? oi640::INTER : oi640::HIDDEN;
    constexpr unsigned K = GATEUP ? oi640::HIDDEN : oi640::INTER;
    constexpr unsigned TN = GATEUP ? 64u : 128u;
    constexpr unsigned BCOLS = L::BCOLS;
    constexpr unsigned CH = BCOLS / 16u; // 4 (GU) or 8 (DN)
    constexpr unsigned TS4E_NUM_K32 = K >> 5;
    constexpr unsigned TS4E_N_TILES = N >> 7;
    constexpr unsigned TS4E_NUM_BKTS = TS4E_NUM_K32 * TS4E_N_TILES;
    constexpr unsigned TS4E_NIBBLES_BYTES = TS4E_NUM_K32 * N;
    constexpr unsigned TS4E_PAGE_BASES_LEN = ((TS4E_NUM_BKTS >> 7) + 4u) & ~3u;
    constexpr unsigned TS4E_PAGE_BASES_OFF = TS4E_NIBBLES_BYTES + N;
    constexpr unsigned TS4E_BKT_OFF_OFF = TS4E_PAGE_BASES_OFF + TS4E_PAGE_BASES_LEN * 4u;
    constexpr unsigned TS4E_BKT_OFF_BYTES_ALIGNED = (((TS4E_NUM_BKTS + 1u) * 2u) + 15u) & ~15u;
    constexpr unsigned TS4E_EXC_OFF = TS4E_BKT_OFF_OFF + TS4E_BKT_OFF_BYTES_ALIGNED;
    constexpr unsigned TS4E_B_PER_KS = L::TS4E_B_PER_KS; // 8 (GU) or 4 (DN)
    constexpr unsigned n_tiles = N / TN;                 // 10 (GU) or 20 (DN)
    constexpr unsigned KS = K / V32_K_STEP;              // 20 (GU) or 5 (DN)
    constexpr unsigned K_HALF = K / 2u;                  // 1280 (GU) or 320 (DN)
    constexpr unsigned K_GRP = K / 16u;                  // 160 (GU) or 40 (DN)
    static_assert(KS >= 2u, "v32_w4a4 requires KS >= 2");

    unsigned dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (gridDim.x == 0u || gridDim.x > oi640::FIXED_GRID || blockDim.x != V32_THREADS ||
        dyn_smem < L::BYTES) {
        if (threadIdx.x == 0u) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }
    const bool shape = (N_in == N && K_in == K);
    const bool s2_gu_ok = (oq_scale2_out != nullptr) || (isfinite(oq_scale2) && oq_scale2 > 0.0f);
    const bool s2_dn_ok = (oq_scale2_in != nullptr) || (isfinite(oq_scale2) && oq_scale2 > 0.0f);
    const bool outs = GATEUP
        ? (oq != nullptr && os != nullptr && (!OQ_NVFP4 || s2_gu_ok))
        : ((EPI == 3 ? (slot_q != nullptr && slot_s != nullptr) : (C != nullptr)) && s2_dn_ok);
    const bool tbl_ok = pk0 != nullptr && sc0 != nullptr && s20 != nullptr &&
                        (!GATEUP || (x_scale2 != nullptr && pk1 != nullptr && sc1 != nullptr && s21 != nullptr));
    if (!shape || !outs || !tbl_ok || aq == nullptr || as == nullptr || expert_offsets == nullptr ||
        sorted_token_ids == nullptr || !oi640::contract_ok(contract) ||
        !oi640::workspace_ok_worksteal(workspace, contract) ||
        status == nullptr || status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0u) oi640::fail(status, oi640::ERR_ABI);
        return;
    }

    const unsigned wc = workspace->work_count;
    const unsigned task_count = wc * n_tiles;
    if (blockIdx.x >= task_count) return;

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    // Keep `sbase` in a general register (not UR0) to avoid the CUDA 13.0 ptxas
    // uniform-register allocator bug that emits odd `desc[UR1]` on `cp.async.L2::cache_hint`.
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n"
                 : "+r"(sbase));
    unsigned long long pol_el, pol_ef;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));

    uint32_t* const next_task_ptr = GATEUP ? &workspace->next_task_gu : &workspace->next_task_dn;
    unsigned task;
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(task));
    unsigned task_slot = 0u;

    #pragma unroll 1
    while (task < task_count) {
        const unsigned raw_wi = task / n_tiles;
        const unsigned wi = (!GATEUP) ? (wc - 1u - raw_wi) : raw_wi;
        const uint2 item_u2 = *reinterpret_cast<const uint2*>(&workspace->items[wi]);
        const unsigned item_expert = item_u2.x;
        const unsigned item_m_tile = item_u2.y;
        const unsigned nt = task % n_tiles;
        const unsigned cta_n = nt * TN;
        if (item_expert >= oi640::EXPERTS) {
            if (threadIdx.x == 0u) oi640::fail(status, oi640::ERR_PLAN);
            return;
        }
        const int m_start = expert_offsets[item_expert];
        const int M_expert = expert_offsets[item_expert + 1] - m_start;
        const int local = static_cast<int>(item_m_tile * V32_TILE_M);
        const bool range_ok = (m_start >= 0 && M_expert > 0 && local >= 0 && local < M_expert);
        const unsigned cta_m = range_ok ? static_cast<unsigned>(m_start + local) : 0u;
        const int valid_rows = range_ok ? min(M_expert - local, static_cast<int>(V32_TILE_M)) : 0;

        if (threadIdx.x == 0u) {
            const unsigned long long pB0 = pk0[item_expert];
            const unsigned long long pS0 = sc0[item_expert];
            const unsigned long long pB1 = GATEUP ? pk1[item_expert] : pB0;
            const unsigned long long pS1 = GATEUP ? sc1[item_expert] : pS0;
            const float f_s2a = s20[item_expert];
            const float f_s2b = GATEUP ? s21[item_expert] : 0.0f;
            const bool ptr_bad = !range_ok || pB0 == 0ull || pS0 == 0ull || pB1 == 0ull || pS1 == 0ull ||
                                 !isfinite(f_s2a) || !isfinite(f_s2b);
            unsigned long long* const sPtrs = reinterpret_cast<unsigned long long*>(smem + L::OFF_PTRS);
            sPtrs[0] = pB0;
            sPtrs[1] = pS0;
            sPtrs[2] = pB1;
            sPtrs[3] = pS1;
            sPtrs[4] = ptr_bad ? 1ull : 0ull;
        }
        if constexpr (GATEUP) {
            const unsigned t0 = threadIdx.x;
            if (t0 < V32_TILE_M) {
                const bool ok_row = static_cast<int>(t0) < valid_rows;
                const unsigned tok_row = ok_row ? static_cast<unsigned>(sorted_token_ids[cta_m + t0]) : 0u;
                reinterpret_cast<float*>(smem + L::OFF_XS2)[t0] = ok_row ? x_scale2[tok_row] : 0.0f;
            }
        } else {
            // Precompute overflow-safe online rescaling ratios `s_os2[0..9][row]` and
            // final scale `s_os2[10][row]` once per task (1 thread per row, 0 divisions
            // and 0 extra persistent registers in the K-loop).
            const unsigned t0 = threadIdx.x;
            float* const s_os2 = reinterpret_cast<float*>(smem + L::OFF_XS2);
            if (t0 < V32_TILE_M) {
                const bool ok_row = static_cast<int>(t0) < valid_rows;
                if (oq_scale2_in != nullptr) {
                    float s_vals[10];
                    float max_s2 = 0.0f;
                    if (ok_row) {
                        const float* row_ptr =
                            oq_scale2_in + static_cast<unsigned long long>(cta_m + t0) * 10u;
                        #pragma unroll
                        for (unsigned sl_i = 0u; sl_i < 10u; ++sl_i) {
                            const float sv = __ldg(&row_ptr[sl_i]);
                            const float clean = (isfinite(sv) && sv > 0.0f) ? sv : 0.0f;
                            s_vals[sl_i] = clean;
                            max_s2 = fmaxf(max_s2, clean);
                        }
                    } else {
                        #pragma unroll
                        for (unsigned sl_i = 0u; sl_i < 10u; ++sl_i) s_vals[sl_i] = 0.0f;
                    }
                    // Skip slices <= 1e-10 * max_s2 (< 0.001 ULP of FP32) to guarantee
                    // `prev / cur <= 1e10` (zero FP32 overflow on extreme bimodal rows).
                    const float thresh = __fmul_rn(max_s2, 1e-10f);
                    float prev = 0.0f;
                    #pragma unroll
                    for (unsigned sl_i = 0u; sl_i < 10u; ++sl_i) {
                        const float cur = s_vals[sl_i];
                        if (cur > thresh) {
                            s_os2[sl_i * V32_TILE_M + t0] =
                                (prev > 0.0f) ? __fdiv_rn(prev, cur) : 0.0f;
                            prev = cur;
                        } else {
                            s_os2[sl_i * V32_TILE_M + t0] = -1.0f;
                        }
                    }
                    s_os2[10u * V32_TILE_M + t0] = prev;
                } else {
                    s_os2[0u * V32_TILE_M + t0] = ok_row ? 0.0f : -1.0f;
                    #pragma unroll
                    for (unsigned sl_i = 1u; sl_i < 10u; ++sl_i) {
                        s_os2[sl_i * V32_TILE_M + t0] = ok_row ? 1.0f : -1.0f;
                    }
                    s_os2[10u * V32_TILE_M + t0] = ok_row ? oq_scale2 : 0.0f;
                }
            }
        }
        __syncthreads();
        if (reinterpret_cast<const unsigned long long*>(smem + L::OFF_PTRS)[4] != 0ull) {
            if (threadIdx.x == 0u) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }

        const int active_a_rows = (valid_rows + 15) & ~15;
        const bool b_reuse = (!GATEUP)
            ? (item_m_tile > 0u)
            : (local + static_cast<int>(V32_TILE_M) < M_expert);
        const unsigned long long pol_b = b_reuse ? pol_el : pol_ef;
        const bool a_evict_last_nt = (!GATEUP) && (nt + 1u == n_tiles);
        const unsigned col_half = cta_n & 64u;

        // Resolve all TS4E buckets up front (160 for GU, 20 for DN -> <= 1 per thread!).
        if constexpr (TS4E) {
            unsigned t;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
            if (t < L::TS4E_BKTS) {
                const unsigned ks_i = t / TS4E_B_PER_KS;
                const unsigned b = t & (TS4E_B_PER_KS - 1u);
                const unsigned kh = GATEUP ? (b >> 2) : (b >> 1);
                const unsigned sub = GATEUP ? ((b >> 1) & 1u) : 0u;
                const unsigned tt = b & 1u;
                const unsigned char* blob = v32_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                const unsigned bkt = (4u * ks_i + 2u * kh + tt) * TS4E_N_TILES + (cta_n >> 7);
                const unsigned* page_bases = reinterpret_cast<const unsigned*>(blob + TS4E_PAGE_BASES_OFF);
                const unsigned short* bkt_off = reinterpret_cast<const unsigned short*>(blob + TS4E_BKT_OFF_OFF);
                const unsigned pb0 = __ldg(&page_bases[bkt >> 7]);
                const unsigned pb1 = ((bkt & 127u) == 127u) ? __ldg(&page_bases[(bkt + 1u) >> 7]) : pb0;
                const unsigned beg = pb0 + static_cast<unsigned>(__ldg(&bkt_off[bkt]));
                const unsigned end = pb1 + static_cast<unsigned>(__ldg(&bkt_off[bkt + 1u]));
                reinterpret_cast<unsigned*>(smem + L::OFF_BKT_BEG)[t] = beg;
                reinterpret_cast<unsigned short*>(smem + L::OFF_BKT_CNT)[t] =
                    static_cast<unsigned short>(end - beg);
            }
            if (t < 8u) {
                const unsigned bm = GATEUP ? (t >> 2) : 0u;
                const unsigned bc = GATEUP ? (t & 3u) : t;
                const unsigned char* bs = v32_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && bm) ? 3u : 1u);
                v32_cp16_pol_s32(
                    sbase + L::OFF_BASE + t * 16u,
                    bs + static_cast<unsigned long long>(TS4E_NIBBLES_BYTES) + cta_n + bc * 16u,
                    pol_b);
            }
            __syncthreads();
        }

        // Load K_STEP = 128 slice into stage `ks & 1` (two k64 halves: kh=0, kh=1).
        auto load_a = [&](unsigned ks) {
            unsigned t;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
            const unsigned sl = ks & 1u;
            const unsigned row = t >> 1;
            const unsigned ch = t & 1u;
            if (static_cast<int>(row) < active_a_rows) {
                const bool ok = static_cast<int>(row) < valid_rows;
                const unsigned src_row = ok
                    ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + row]) : (cta_m + row))
                    : 0u;
                const unsigned dst_a0 = sbase + sl * 8192u + v32_swz32(row, ch);
                const unsigned char* g_a0 =
                    aq + static_cast<unsigned long long>(src_row) * K_HALF + ks * 64u + ch * 16u;
                if constexpr (!GATEUP) {
                    if (a_evict_last_nt) {
                        v32_cp16_pol_pred_s32(dst_a0, g_a0, ok, pol_ef);
                        v32_cp16_pol_pred_s32(dst_a0 + 4096u, g_a0 + 32u, ok, pol_ef);
                    } else {
                        v32_cp16_s32(dst_a0, g_a0, ok);
                        v32_cp16_s32(dst_a0 + 4096u, g_a0 + 32u, ok);
                    }
                } else {
                    v32_cp16_s32(dst_a0, g_a0, ok);
                    v32_cp16_s32(dst_a0 + 4096u, g_a0 + 32u, ok);
                }
            }
            if (t < V32_TILE_M && static_cast<int>(t) < active_a_rows) {
                const bool ok = static_cast<int>(t) < valid_rows;
                const unsigned src_row = ok
                    ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + t]) : (cta_m + t))
                    : 0u;
                v32_cp8_s32(
                    sbase + L::OFF_AS + (sl * V32_TILE_M + t) * 8u,
                    as + static_cast<unsigned long long>(src_row) * K_GRP + ks * 8u,
                    ok);
            }
        };

        auto load_b = [&](unsigned ks) {
            unsigned t;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
            const unsigned sl = ks & 1u;
            const unsigned r = t >> 3;
            const unsigned c = t & 7u;
            const unsigned c_swz = c ^ ((r >> 1) & 7u);
            const unsigned mtx = GATEUP ? (c >> 2) : 0u;
            const unsigned lc = GATEUP ? (c & 3u) : c;
            const unsigned char* bp = v32_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && mtx) ? 2u : 0u);
            const unsigned dst_b0 = sbase + L::OFF_B + sl * 8192u + r * 128u + (c_swz << 4);
            const unsigned char* g_b0 =
                bp + static_cast<unsigned long long>(ks * 64u + r) * N + cta_n + lc * 16u;
            v32_cp16_pol_s32(dst_b0, g_b0, pol_b);
            v32_cp16_pol_s32(dst_b0 + 4096u, g_b0 + static_cast<unsigned long long>(32u) * N, pol_b);

            if constexpr (TS4E) {
                const unsigned wid = t >> 5;
                const unsigned ln = t & 31u;
                if (wid < TS4E_B_PER_KS) {
                    const unsigned kh = GATEUP ? (wid >> 2) : (wid >> 1);
                    const unsigned sub = GATEUP ? ((wid >> 1) & 1u) : 0u;
                    const unsigned tt = wid & 1u;
                    const unsigned col_off_s = GATEUP ? (sub << 6) : 0u;
                    if (ln < CH) {
                        const unsigned char* s = v32_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                        const void* g_s =
                            s + static_cast<unsigned long long>(ks * 4u + kh * 2u + tt) * N + cta_n + ln * 16u;
                        const unsigned dst_s =
                            sbase + L::OFF_S + (sl * 2u + kh) * 512u + (2u * tt + 1u) * 128u + col_off_s + ln * 16u;
                        v32_cp16_pol_s32(dst_s, g_s, pol_b);
                    } else if (ln < CH + 4u) {
                        const unsigned ci = ln - CH;
                        const unsigned idx = ks * TS4E_B_PER_KS + wid;
                        const unsigned cnt = static_cast<unsigned>(
                            reinterpret_cast<const unsigned short*>(smem + L::OFF_BKT_CNT)[idx]);
                        if (cnt > 0u) {
                            const unsigned beg = reinterpret_cast<const unsigned*>(smem + L::OFF_BKT_BEG)[idx];
                            const unsigned pad = beg & 7u;
                            if (pad + cnt <= L::TS4E_EXC_CAP && (ci << 3) < (pad + cnt)) {
                                const unsigned char* blob =
                                    v32_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                                const unsigned short* exc =
                                    reinterpret_cast<const unsigned short*>(blob + TS4E_EXC_OFF);
                                const unsigned dst_e =
                                    sbase + L::OFF_EXC + ((sl * TS4E_B_PER_KS + wid) * L::TS4E_EXC_CAP + (ci << 3)) * 2u;
                                v32_cp16_pol_s32(dst_e, exc + (beg & ~7u) + (ci << 3), pol_b);
                            }
                        }
                    }
                }
            } else {
                if (t < 64u) {
                    const unsigned kh = t >> 5;
                    const unsigned r2 = (t >> 3) & 3u;
                    const unsigned c2 = t & 7u;
                    const unsigned m2 = GATEUP ? (c2 >> 2) : 0u;
                    const unsigned lc2 = GATEUP ? (c2 & 3u) : c2;
                    const unsigned char* sp = v32_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && m2) ? 3u : 1u);
                    v32_cp16_pol_s32(
                        sbase + L::OFF_S + (sl * 2u + kh) * 512u + r2 * 128u + c2 * 16u,
                        sp + static_cast<unsigned long long>(ks * 8u + kh * 4u + r2) * N + cta_n + lc2 * 16u,
                        pol_b);
                }
            }
        };

        auto decode_ts4e_stage = [&](unsigned ks_dec) {
            unsigned t;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
            const unsigned wid = t >> 5;
            if (wid >= TS4E_B_PER_KS) return;
            const unsigned ln = t & 31u;
            __syncwarp();
            unsigned char (*const sS)[2][4][V32_BC] =
                reinterpret_cast<unsigned char (*)[2][4][V32_BC]>(smem + L::OFF_S);
            const unsigned char* const sBase = smem + L::OFF_BASE;
            const unsigned short (*const sExc)[TS4E_B_PER_KS][L::TS4E_EXC_CAP] =
                reinterpret_cast<const unsigned short (*)[TS4E_B_PER_KS][L::TS4E_EXC_CAP]>(smem + L::OFF_EXC);
            const unsigned sl = ks_dec & 1u;
            const unsigned kh = GATEUP ? (wid >> 2) : (wid >> 1);
            const unsigned sub = GATEUP ? ((wid >> 1) & 1u) : 0u;
            const unsigned tt = wid & 1u;
            const unsigned col_off_s = GATEUP ? (sub << 6) : 0u;
            constexpr unsigned LANES_PER_SWAR = BCOLS / 4u; // 16 (GU) or 32 (DN)
            const unsigned idx = ks_dec * TS4E_B_PER_KS + wid;
            const unsigned cnt = static_cast<unsigned>(
                reinterpret_cast<const unsigned short*>(smem + L::OFF_BKT_CNT)[idx]);
            const unsigned beg = reinterpret_cast<const unsigned*>(smem + L::OFF_BKT_BEG)[idx];
            const unsigned pad = beg & 7u;
            const bool fast_exc = (cnt != 0u) && (pad + cnt <= L::TS4E_EXC_CAP);
            const unsigned e_fast = (fast_exc && ln < cnt)
                ? static_cast<unsigned>(sExc[sl][wid][pad + ln])
                : 0u;
            if (ln < LANES_PER_SWAR) {
                const unsigned lc4 = col_off_s + (ln << 2);
                const unsigned base_w = *reinterpret_cast<const unsigned*>(&sBase[lc4]);
                const unsigned nib = *reinterpret_cast<const unsigned*>(&sS[sl][kh][2u * tt + 1u][lc4]);

                const unsigned c0 = nib & 0x0F0F0F0Fu;
                const unsigned e0 = (c0 + 0x01010101u) & 0x10101010u;
                *reinterpret_cast<unsigned*>(&sS[sl][kh][2u * tt][lc4]) = base_w + (c0 & ~(e0 - (e0 >> 4)));

                const unsigned c1 = (nib >> 4) & 0x0F0F0F0Fu;
                const unsigned e1 = (c1 + 0x01010101u) & 0x10101010u;
                *reinterpret_cast<unsigned*>(&sS[sl][kh][2u * tt + 1u][lc4]) = base_w + (c1 & ~(e1 - (e1 >> 4)));
            }
            __syncwarp();
            if (cnt != 0u) {
                if (fast_exc) {
                    if (ln < cnt) {
                        const unsigned col128 = e_fast & 127u;
                        if constexpr (GATEUP) {
                            if ((col128 & 64u) == col_half) {
                                sS[sl][kh][2u * tt + ((e_fast >> 7) & 1u)][col_off_s + (col128 & 63u)] =
                                    static_cast<unsigned char>(e_fast >> 8);
                            }
                        } else {
                            sS[sl][kh][2u * tt + ((e_fast >> 7) & 1u)][col128] =
                                static_cast<unsigned char>(e_fast >> 8);
                        }
                    }
                } else {
                    const unsigned char* blob = v32_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                    const unsigned short* exc =
                        reinterpret_cast<const unsigned short*>(blob + TS4E_EXC_OFF);
                    for (unsigned i = ln; i < cnt; i += 32u) {
                        const unsigned e = static_cast<unsigned>(__ldg(&exc[beg + i]));
                        const unsigned col128 = e & 127u;
                        if constexpr (GATEUP) {
                            if ((col128 & 64u) == col_half) {
                                sS[sl][kh][2u * tt + ((e >> 7) & 1u)][col_off_s + (col128 & 63u)] =
                                    static_cast<unsigned char>(e >> 8);
                            }
                        } else {
                            sS[sl][kh][2u * tt + ((e >> 7) & 1u)][col128] =
                                static_cast<unsigned char>(e >> 8);
                        }
                    }
                }
            }
        };

        // Pack sS[sl][2][4][128] -> conflict-free sSF[sl][2][128] (6 insns across all 256 threads).
        auto pack_sf_stage = [&](unsigned sl) {
            unsigned t;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
            const unsigned kh = t >> 7;
            const unsigned col = t & 127u;
            const unsigned q = col & 3u;
            const unsigned sel1 = (q & 1u) ? 0x3715u : 0x6240u;
            const unsigned sel2 = (q & 2u) ? 0x3276u : 0x5410u;
            const unsigned su = v32_ld_su32(
                sbase + L::OFF_S + (sl * 2u + kh) * 512u + q * 128u + (col & ~3u));
            const unsigned sp = v32_prmt(su, __shfl_xor_sync(0xffffffffu, su, 1), sel1);
            const unsigned sc = v32_prmt(sp, __shfl_xor_sync(0xffffffffu, sp, 2), sel2) & 0x7f7f7f7fu;
            v32_st_su32(
                sbase + L::OFF_SF + (sl * 2u + kh) * 512u + v32_sf_eo_u32_idx<GATEUP>(col) * 4u,
                sc);
        };

        auto prefetch_b = [&](unsigned pf_k128) {
            if (task_count <= gridDim.x) return;
            unsigned raw_peek;
            asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(raw_peek) : "l"(next_task_ptr));
            if (raw_peek >= task_count - gridDim.x) return;
            const unsigned pf_task = gridDim.x + raw_peek;
            const unsigned pf_raw_wi = pf_task / n_tiles;
            const unsigned pf_cta_n = (pf_task % n_tiles) * TN;
            const unsigned pf_wi = (!GATEUP) ? (wc - 1u - pf_raw_wi) : pf_raw_wi;
            const oi640::WorkItem pf_item = workspace->items[pf_wi];
            if (pf_item.expert >= oi640::EXPERTS || pf_item.expert == item_expert) return;
            if constexpr (!GATEUP) {
                if (pf_wi + 1u < wc && workspace->items[pf_wi + 1u].expert == pf_item.expert) return;
            } else {
                if (pf_item.m_tile != 0u) return;
            }
            unsigned t;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
            const unsigned r = t >> 3;
            const unsigned c = t & 7u;
            if ((c & 1u) == 0u) {
                const unsigned mtx = GATEUP ? (c >> 2) : 0u;
                const unsigned lc = GATEUP ? (c & 3u) : c;
                const unsigned char* pB = reinterpret_cast<const unsigned char*>(
                    (GATEUP && mtx) ? pk1[pf_item.expert] : pk0[pf_item.expert]);
                if (pB != nullptr) {
                    const unsigned char* g0 =
                        pB + static_cast<unsigned long long>(pf_k128 * 64u + r) * N + pf_cta_n + lc * 16u;
                    v32_prefetch_l2(g0);
                    v32_prefetch_l2(g0 + static_cast<unsigned long long>(32u) * N);
                }
            }
            constexpr unsigned S_ROWS_LD = TS4E ? 4u : 8u;
            if (t < S_ROWS_LD * 4u) {
                const unsigned r2 = t >> 2;
                const unsigned c32 = (t & 3u) << 5;
                const unsigned m2 = GATEUP ? (c32 >> 6) : 0u;
                const unsigned lc32 = GATEUP ? (c32 & 63u) : c32;
                const unsigned char* pS = reinterpret_cast<const unsigned char*>(
                    (GATEUP && m2) ? sc1[pf_item.expert] : sc0[pf_item.expert]);
                if (pS != nullptr) {
                    v32_prefetch_l2(pS + static_cast<unsigned long long>(pf_k128 * S_ROWS_LD + r2) * N + pf_cta_n + lc32);
                }
            }
        };

        float acc[2][8][4];
        #pragma unroll
        for (int r = 0; r < 2; ++r)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.0f;

        // K_STEP = 128 MMA step: 32 back-to-back OMMAs (kh=0, kh=1) per iteration!
        // Uses ldmatrix.m8n8.x4.trans directly from sB + conflict-free LDS.128 from sSF.
        // In DOWN (`!GATEUP`), applies FlashAttention-style online accumulator rescaling
        // using precomputed overflow-safe ratios from `s_os2[kh_slice][row]` before each
        // `kh64` MMA (0 extra registers live during OMMAs, 0 divisions in K-loop).
        auto mma_step_k128 = [&](unsigned ks, unsigned sl) {
            unsigned t;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
            const unsigned wid = t >> 5;
            const unsigned wm = (wid >> 1) * 32u;
            const int wv = valid_rows - static_cast<int>(wm);
            if (wv <= 0) return;
            const bool two_rt = (wv > 16);

            const unsigned ln = t & 31u;
            const unsigned gid = ln >> 2;
            const unsigned tig = ln & 3u;
            const unsigned wn = wid & 1u;

            const unsigned as_addr = sbase + L::OFF_AS + (sl * V32_TILE_M + wm + gid + (tig & 1u) * 8u) * 8u;
            const unsigned a_base = sbase + sl * 8192u + v32_swz32(wm + (ln & 15u), ln >> 4);
            const unsigned m_ld = ln >> 3;
            const unsigned i_ld = ln & 7u;
            const unsigned r_ld = ((m_ld & 2u) << 3) + (i_ld << 1) + (m_ld & 1u);
            const unsigned b_base = sbase + L::OFF_B + sl * 8192u + r_ld * 128u;
            const unsigned sf_base0 = sbase + L::OFF_SF + (sl * 2u) * 512u + (wn * 16u + gid) * 16u;

            const unsigned c0 = GATEUP ? (wn << 1) : (wn << 2);
            const unsigned o0 = (c0 ^ ((r_ld >> 1) & 7u)) << 4;
            constexpr unsigned XOR_PAIR2 = GATEUP ? 64u : 32u;

            unsigned nr0, nr1, nr2, nr3;
            v32_ldsm4_trans(b_base + o0, nr0, nr1, nr2, nr3);

            auto run_half = [&](unsigned kh_off_a, unsigned kh_off_b, unsigned kh_off_sf,
                                unsigned kh_off_as, unsigned kh_slice, bool prefetch_next_kh, bool do_r1) {
                bool skip_sfa0 = false;
                bool skip_sfa1 = false;
                if constexpr (!GATEUP) {
                    const unsigned s2_row0 = kh_slice * V32_TILE_M + wm + gid;
                    const float r00 = v32_ld_sxs2(sbase + L::OFF_XS2, s2_row0);
                    const float r01 = v32_ld_sxs2(sbase + L::OFF_XS2, s2_row0 + 8u);
                    if (r00 > 0.0f && r00 != 1.0f) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            acc[0][j][0] *= r00;
                            acc[0][j][1] *= r00;
                        }
                    }
                    if (r01 > 0.0f && r01 != 1.0f) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            acc[0][j][2] *= r01;
                            acc[0][j][3] *= r01;
                        }
                    }
                    skip_sfa0 = ((tig & 1u) == 0u) ? (r00 < 0.0f) : (r01 < 0.0f);
                    if (do_r1) {
                        const float r10 = v32_ld_sxs2(sbase + L::OFF_XS2, s2_row0 + 16u);
                        const float r11 = v32_ld_sxs2(sbase + L::OFF_XS2, s2_row0 + 24u);
                        if (r10 > 0.0f && r10 != 1.0f) {
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                acc[1][j][0] *= r10;
                                acc[1][j][1] *= r10;
                            }
                        }
                        if (r11 > 0.0f && r11 != 1.0f) {
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                acc[1][j][2] *= r11;
                                acc[1][j][3] *= r11;
                            }
                        }
                        skip_sfa1 = ((tig & 1u) == 0u) ? (r10 < 0.0f) : (r11 < 0.0f);
                    }
                } else {
                    (void)kh_slice;
                }

                const unsigned sfa0 = skip_sfa0 ? 0u : v32_ld_su32(as_addr + kh_off_as);
                const unsigned sfa1 = (do_r1 && !skip_sfa1) ? v32_ld_su32(as_addr + kh_off_as + 16u * 8u) : 0u;
                unsigned afr0[4], afr1[4];
                v32_ldsm4(a_base + kh_off_a, afr0[0], afr0[1], afr0[2], afr0[3]);
                if (do_r1) v32_ldsm4(a_base + kh_off_a + 512u, afr1[0], afr1[1], afr1[2], afr1[3]);

                const uint4 sfb0 = v32_ld_sv4(sf_base0 + kh_off_sf);

                // Pair 0 (j = 0, 1)
                {
                    const unsigned b0e = v32_prmt(nr0, nr1, 0x6240u);
                    const unsigned b0o = v32_prmt(nr0, nr1, 0x7351u);
                    const unsigned b1e = v32_prmt(nr2, nr3, 0x6240u);
                    const unsigned b1o = v32_prmt(nr2, nr3, 0x7351u);
                    v32_ldsm4_trans(b_base + kh_off_b + (o0 ^ 16u), nr0, nr1, nr2, nr3);
                    v32_mma_nvfp4(acc[0][0], afr0, b0e, b1e, sfa0, sfb0.x);
                    v32_mma_nvfp4(acc[0][1], afr0, b0o, b1o, sfa0, sfb0.y);
                    if (do_r1) {
                        v32_mma_nvfp4(acc[1][0], afr1, b0e, b1e, sfa1, sfb0.x);
                        v32_mma_nvfp4(acc[1][1], afr1, b0o, b1o, sfa1, sfb0.y);
                    }
                }
                // Pair 1 (j = 2, 3)
                {
                    const unsigned b0e = v32_prmt(nr0, nr1, 0x6240u);
                    const unsigned b0o = v32_prmt(nr0, nr1, 0x7351u);
                    const unsigned b1e = v32_prmt(nr2, nr3, 0x6240u);
                    const unsigned b1o = v32_prmt(nr2, nr3, 0x7351u);
                    v32_ldsm4_trans(b_base + kh_off_b + (o0 ^ XOR_PAIR2), nr0, nr1, nr2, nr3);
                    v32_mma_nvfp4(acc[0][2], afr0, b0e, b1e, sfa0, sfb0.z);
                    v32_mma_nvfp4(acc[0][3], afr0, b0o, b1o, sfa0, sfb0.w);
                    if (do_r1) {
                        v32_mma_nvfp4(acc[1][2], afr1, b0e, b1e, sfa1, sfb0.z);
                        v32_mma_nvfp4(acc[1][3], afr1, b0o, b1o, sfa1, sfb0.w);
                    }
                }

                const uint4 sfb1 = v32_ld_sv4(sf_base0 + kh_off_sf + 128u);

                // Pair 2 (j = 4, 5)
                {
                    const unsigned b0e = v32_prmt(nr0, nr1, 0x6240u);
                    const unsigned b0o = v32_prmt(nr0, nr1, 0x7351u);
                    const unsigned b1e = v32_prmt(nr2, nr3, 0x6240u);
                    const unsigned b1o = v32_prmt(nr2, nr3, 0x7351u);
                    v32_ldsm4_trans(b_base + kh_off_b + ((o0 ^ XOR_PAIR2) ^ 16u), nr0, nr1, nr2, nr3);
                    v32_mma_nvfp4(acc[0][4], afr0, b0e, b1e, sfa0, sfb1.x);
                    v32_mma_nvfp4(acc[0][5], afr0, b0o, b1o, sfa0, sfb1.y);
                    if (do_r1) {
                        v32_mma_nvfp4(acc[1][4], afr1, b0e, b1e, sfa1, sfb1.x);
                        v32_mma_nvfp4(acc[1][5], afr1, b0o, b1o, sfa1, sfb1.y);
                    }
                }
                // Pair 3 (j = 6, 7)
                {
                    const unsigned b0e = v32_prmt(nr0, nr1, 0x6240u);
                    const unsigned b0o = v32_prmt(nr0, nr1, 0x7351u);
                    const unsigned b1e = v32_prmt(nr2, nr3, 0x6240u);
                    const unsigned b1o = v32_prmt(nr2, nr3, 0x7351u);
                    if (prefetch_next_kh) {
                        v32_ldsm4_trans(b_base + 4096u + o0, nr0, nr1, nr2, nr3);
                    }
                    v32_mma_nvfp4(acc[0][6], afr0, b0e, b1e, sfa0, sfb1.z);
                    v32_mma_nvfp4(acc[0][7], afr0, b0o, b1o, sfa0, sfb1.w);
                    if (do_r1) {
                        v32_mma_nvfp4(acc[1][6], afr1, b0e, b1e, sfa1, sfb1.z);
                        v32_mma_nvfp4(acc[1][7], afr1, b0o, b1o, sfa1, sfb1.w);
                    }
                }
            };

            const unsigned kh0 = ks * 2u;
            if (two_rt) {
                run_half(0u,    0u,    0u,   0u, kh0,      true,  true);
                run_half(4096u, 4096u, 512u, 4u, kh0 + 1u, false, true);
            } else {
                run_half(0u,    0u,    0u,   0u, kh0,      true,  false);
                run_half(4096u, 4096u, 512u, 4u, kh0 + 1u, false, false);
            }
        };

        // Prologue: issue stage 0 (K = 0..127).
        load_a(0u);
        load_b(0u);
        v32_commit();

        // Steady-state K_STEP = 128 hot loop (ks = 0 .. KS - 2):
        // Starts next stage cp.async before pack_sf_stage so DRAM/L2 loads overlap
        // both scale packing and all 32 back-to-back OMMAs.
        #pragma unroll 1
        for (unsigned ks = 0u; ks < KS - 1u; ++ks) {
            const unsigned sl = ks & 1u;
            v32_wait<0>();
            if constexpr (TS4E) {
                if (ks == 0u) __syncthreads();
                decode_ts4e_stage(ks);
            }
            __syncthreads();
            load_a(ks + 1u);
            load_b(ks + 1u);
            v32_commit();
            pack_sf_stage(sl);
            __syncthreads();
            mma_step_k128(ks, sl);
        }

        // Final stage (ks = KS - 1) with next-task L2 prefetch.
        {
            const unsigned k_last = KS - 1u;
            const unsigned sl = k_last & 1u;
            v32_wait<0>();
            if constexpr (TS4E) {
                decode_ts4e_stage(k_last);
            }
            __syncthreads();
            if constexpr (PREFETCH) prefetch_b(0u);
            pack_sf_stage(sl);
            __syncthreads();
            mma_step_k128(k_last, sl);
        }

        // 0-SMEM Register-Direct Epilogue (even/odd paired MMA outputs -> contiguous quads).
        unsigned ep_tid;
        asm volatile("mov.u32 %0, %%tid.x;" : "=r"(ep_tid));
        const unsigned ep_wid = ep_tid >> 5;
        const unsigned ep_lane = ep_tid & 31u;
        const unsigned ep_gid = ep_lane >> 2;
        const unsigned ep_tig = ep_lane & 3u;
        const unsigned ep_wm = (ep_wid >> 1) * 32u;
        const unsigned ep_wn = (ep_wid & 1u) * (GATEUP ? 32u : 64u);
        const int ep_wv = valid_rows - static_cast<int>(ep_wm);
        const int ep_rt = ep_wv <= 0 ? 0 : min(2, (ep_wv + 15) >> 4);
        const float ep_s2a = s20[item_expert];
        const float ep_s2b = GATEUP ? s21[item_expert] : 0.0f;

        if constexpr (GATEUP) {
            const unsigned col = cta_n + ep_wn;
            auto pack_silu_quad = [&](int r, int q_lo, int q_hi, float ga, float gb, unsigned (&u)[4]) {
                #pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const int je = 2 * p;
                    const int jo = je + 1;
                    const float ge0 = acc[r][je][q_lo] * ga, ue0 = acc[r][je + 4][q_lo] * gb;
                    const float go0 = acc[r][jo][q_lo] * ga, uo0 = acc[r][jo + 4][q_lo] * gb;
                    const float ve0 = ge0 * (1.0f / (1.0f + __expf(-ge0))) * ue0;
                    const float vo0 = go0 * (1.0f / (1.0f + __expf(-go0))) * uo0;
                    u[2 * p + 0] = v32_pack_bf162(ve0, vo0);

                    const float ge1 = acc[r][je][q_hi] * ga, ue1 = acc[r][je + 4][q_hi] * gb;
                    const float go1 = acc[r][jo][q_hi] * ga, uo1 = acc[r][jo + 4][q_hi] * gb;
                    const float ve1 = ge1 * (1.0f / (1.0f + __expf(-ge1))) * ue1;
                    const float vo1 = go1 * (1.0f / (1.0f + __expf(-go1))) * uo1;
                    u[2 * p + 1] = v32_pack_bf162(ve1, vo1);
                }
            };

            if constexpr (OQ_NVFP4) {
                const bool keep_oq = (workspace->rows <= V32_OQ_NVFP4_PIN_MAX_ROWS) ||
                                     (raw_wi + V32_OQ_MXFP8_KEEP_WINDOW >= wc);
                const bool dyn_s2_active = (oq_scale2_out != nullptr);
                const float inv_oq_s2_over_6_scalar =
                    dyn_s2_active ? 0.0f : __fdiv_rn(1.0f, __fmul_rn(oq_scale2, 6.0f));
                const unsigned col_byte4 = (col >> 1) + ep_tig * 4u;
                const unsigned grp_pair = col >> 4;
                if (dyn_s2_active) {
                    // Wait for all warps to finish reading sA before reusing sA[0..1024)
                    // as s_half_amax[2][128] for 2-warp slice_amax reduction.
                    __syncthreads();
                }
                auto emit_oq4_row = [&](int r, int q_lo, int q_hi, unsigned lr) {
                    const bool active = (r < ep_rt);
                    uint4 w = make_uint4(0u, 0u, 0u, 0u);
                    float gmax = 0.0f;
                    float half_max = 0.0f;
                    if (active) {
                        const float xs2 = v32_ld_sxs2(sbase + L::OFF_XS2, lr);
                        unsigned u[4];
                        pack_silu_quad(r, q_lo, q_hi, __fmul_rn(ep_s2a, xs2), __fmul_rn(ep_s2b, xs2), u);
                        w = v32_bf16_reg_quad_eo(u[0], u[1], u[2], u[3], ep_tig);
                        const unsigned uw[4] = {w.x, w.y, w.z, w.w};
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            gmax = fmaxf(gmax, fmaxf(fabsf(__uint_as_float(uw[i] << 16)),
                                                     fabsf(__uint_as_float(uw[i] & 0xffff0000u))));
                        }
                        gmax = fmaxf(gmax, __shfl_xor_sync(0xffffffffu, gmax, 1));
                        if (dyn_s2_active) {
                            half_max = fmaxf(gmax, __shfl_xor_sync(0xffffffffu, gmax, 2));
                        }
                    }
                    float row_s2 = oq_scale2;
                    float inv_s2_over_6 = inv_oq_s2_over_6_scalar;
                    if (dyn_s2_active) {
                        float* const s_half = reinterpret_cast<float*>(smem);
                        const unsigned wn_bit = ep_wid & 1u;
                        if (ep_tig == 0u) {
                            s_half[wn_bit * V32_TILE_M + lr] = half_max;
                        }
                        __syncthreads();
                        if (active) {
                            const float other_max = s_half[(wn_bit ^ 1u) * V32_TILE_M + lr];
                            const float slice_amax = fmaxf(half_max, other_max);
                            const bool pos = isfinite(slice_amax) && (slice_amax > 1e-28f);
                            row_s2 = pos ? __fdiv_rn(slice_amax, 2688.0f) : 0.0f;
                            inv_s2_over_6 = pos ? __fdiv_rn(448.0f, slice_amax) : 0.0f;
                        }
                    }
                    if (active) {
                        const unsigned uw[4] = {w.x, w.y, w.z, w.w};
                        float v[8];
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            v[2 * i] = __uint_as_float(uw[i] << 16);
                            v[2 * i + 1] = __uint_as_float(uw[i] & 0xffff0000u);
                        }
                        const float sf_f = (gmax > 0.0f && inv_s2_over_6 > 0.0f)
                            ? fminf(__fmul_rn(gmax, inv_s2_over_6), 448.0f)
                            : 0.0f;
                        const unsigned sf_byte = v32_rn_ue4m3(sf_f);
                        const float eff = __fmul_rn(v32_e4m3_to_float(sf_byte), row_s2);
                        const float inv_eff = (eff > 0.0f) ? __fdiv_rn(1.0f, eff) : 0.0f;
                        const unsigned packed = v32_cvt_f32x8_to_e2m1x8(v, inv_eff);
                        const unsigned sf_other = __shfl_xor_sync(0xffffffffu, sf_byte, 2);
                        const unsigned short sf_pair =
                            static_cast<unsigned short>((sf_byte & 0xffu) | ((sf_other & 0xffu) << 8));
                        if (static_cast<int>(lr) < valid_rows) {
                            const unsigned long long orow = cta_m + lr;
                            if (keep_oq) v32_st4_l2_el(oq + orow * (N / 2u) + col_byte4, packed, pol_el);
                            else         v32_st4_ef(oq + orow * (N / 2u) + col_byte4, packed);
                            if (ep_tig == 0u) {
                                *reinterpret_cast<unsigned short*>(os + orow * (N / 16u) + grp_pair) = sf_pair;
                                if (dyn_s2_active && (ep_wid & 1u) == 0u) {
                                    if (keep_oq) {
                                        v32_st4_l2_el(oq_scale2_out + orow * n_tiles + nt,
                                                      __float_as_uint(row_s2), pol_el);
                                    } else {
                                        oq_scale2_out[orow * n_tiles + nt] = row_s2;
                                    }
                                }
                            }
                        }
                    }
                };
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const unsigned lr0 = ep_wm + r * 16u + ep_gid;
                    const unsigned lr1 = lr0 + 8u;
                    emit_oq4_row(r, 0, 1, lr0);
                    asm volatile("" ::: "memory");
                    emit_oq4_row(r, 2, 3, lr1);
                    asm volatile("" ::: "memory");
                }
            } else {
                const bool keep_oq = (workspace->rows <= 4096u) ||
                                     (raw_wi + V32_OQ_MXFP8_KEEP_WINDOW >= wc);
                const unsigned col8 = col + ep_tig * 8u;
                const unsigned blk = col >> 5;
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    if (r >= ep_rt) break;
                    const unsigned lr0 = ep_wm + r * 16u + ep_gid;
                    const unsigned lr1 = lr0 + 8u;
                    {
                        const float xs2_0 = v32_ld_sxs2(sbase + L::OFF_XS2, lr0);
                        unsigned u0[4];
                        pack_silu_quad(r, 0, 1, __fmul_rn(ep_s2a, xs2_0), __fmul_rn(ep_s2b, xs2_0), u0);
                        unsigned qq0[2], ss0;
                        v32_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], ep_tig, qq0, ss0);
                        if (static_cast<int>(lr0) < valid_rows) {
                            const unsigned long long orow = cta_m + lr0;
                            if (keep_oq) v32_st8_l2_el(oq + orow * N + col8, make_uint2(qq0[0], qq0[1]), pol_el);
                            else         v32_st8_ef(oq + orow * N + col8, make_uint2(qq0[0], qq0[1]));
                            if (ep_tig == 0u) os[orow * (N / 32u) + blk] = static_cast<unsigned char>(ss0);
                        }
                    }
                    asm volatile("" ::: "memory");
                    {
                        const float xs2_1 = v32_ld_sxs2(sbase + L::OFF_XS2, lr1);
                        unsigned u1[4];
                        pack_silu_quad(r, 2, 3, __fmul_rn(ep_s2a, xs2_1), __fmul_rn(ep_s2b, xs2_1), u1);
                        unsigned qq1[2], ss1;
                        v32_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], ep_tig, qq1, ss1);
                        if (static_cast<int>(lr1) < valid_rows) {
                            const unsigned long long orow = cta_m + lr1;
                            if (keep_oq) v32_st8_l2_el(oq + orow * N + col8, make_uint2(qq1[0], qq1[1]), pol_el);
                            else         v32_st8_ef(oq + orow * N + col8, make_uint2(qq1[0], qq1[1]));
                            if (ep_tig == 0u) os[orow * (N / 32u) + blk] = static_cast<unsigned char>(ss1);
                        }
                    }
                    asm volatile("" ::: "memory");
                }
            }
        } else {
            const bool keep_slot =
                (static_cast<unsigned long long>(workspace->expanded) * 2640ull) <= V32_KEEP_SLOT_BYTES;
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= ep_rt) break;
                const unsigned lr0 = ep_wm + r * 16u + ep_gid;
                const unsigned lr1 = lr0 + 8u;
                const float dn_scale0 =
                    __fmul_rn(ep_s2a, v32_ld_sxs2(sbase + L::OFF_XS2, 10u * V32_TILE_M + lr0));
                const float dn_scale1 =
                    __fmul_rn(ep_s2a, v32_ld_sxs2(sbase + L::OFF_XS2, 10u * V32_TILE_M + lr1));
                #pragma unroll
                for (unsigned b = 0u; b < 2u; ++b) {
                    const unsigned col = cta_n + ep_wn + b * 32u;
                    const unsigned col8 = col + ep_tig * 8u;
                    const int jb = 4 * static_cast<int>(b);
                    {
                        unsigned u0[4];
                        u0[0] = v32_pack_bf162(acc[r][jb + 0][0] * dn_scale0, acc[r][jb + 1][0] * dn_scale0);
                        u0[1] = v32_pack_bf162(acc[r][jb + 0][1] * dn_scale0, acc[r][jb + 1][1] * dn_scale0);
                        u0[2] = v32_pack_bf162(acc[r][jb + 2][0] * dn_scale0, acc[r][jb + 3][0] * dn_scale0);
                        u0[3] = v32_pack_bf162(acc[r][jb + 2][1] * dn_scale0, acc[r][jb + 3][1] * dn_scale0);
                        if constexpr (EPI == 3) {
                            unsigned qq0[2], ss0;
                            v32_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], ep_tig, qq0, ss0);
                            if (static_cast<int>(lr0) < valid_rows) {
                                const unsigned long long prow = cta_m + lr0;
                                if (keep_slot) {
                                    v32_st8_l2_el(slot_q + prow * N + col8, make_uint2(qq0[0], qq0[1]), pol_el);
                                } else {
                                    v32_st8_ef(slot_q + prow * N + col8, make_uint2(qq0[0], qq0[1]));
                                }
                                if (ep_tig == 0u) slot_s[prow * (N / 32u) + (col >> 5)] = static_cast<unsigned char>(ss0);
                            }
                        } else {
                            const uint4 w0 = v32_bf16_reg_quad_eo(u0[0], u0[1], u0[2], u0[3], ep_tig);
                            if (static_cast<int>(lr0) < valid_rows) {
                                v32_st16_stream(C + static_cast<unsigned long long>(cta_m + lr0) * N + col8, w0);
                            }
                        }
                    }
                    asm volatile("" ::: "memory");
                    {
                        unsigned u1[4];
                        u1[0] = v32_pack_bf162(acc[r][jb + 0][2] * dn_scale1, acc[r][jb + 1][2] * dn_scale1);
                        u1[1] = v32_pack_bf162(acc[r][jb + 0][3] * dn_scale1, acc[r][jb + 1][3] * dn_scale1);
                        u1[2] = v32_pack_bf162(acc[r][jb + 2][2] * dn_scale1, acc[r][jb + 3][2] * dn_scale1);
                        u1[3] = v32_pack_bf162(acc[r][jb + 2][3] * dn_scale1, acc[r][jb + 3][3] * dn_scale1);
                        if constexpr (EPI == 3) {
                            unsigned qq1[2], ss1;
                            v32_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], ep_tig, qq1, ss1);
                            if (static_cast<int>(lr1) < valid_rows) {
                                const unsigned long long prow = cta_m + lr1;
                                if (keep_slot) {
                                    v32_st8_l2_el(slot_q + prow * N + col8, make_uint2(qq1[0], qq1[1]), pol_el);
                                } else {
                                    v32_st8_ef(slot_q + prow * N + col8, make_uint2(qq1[0], qq1[1]));
                                }
                                if (ep_tig == 0u) slot_s[prow * (N / 32u) + (col >> 5)] = static_cast<unsigned char>(ss1);
                            }
                        } else {
                            const uint4 w1 = v32_bf16_reg_quad_eo(u1[0], u1[1], u1[2], u1[3], ep_tig);
                            if (static_cast<int>(lr1) < valid_rows) {
                                v32_st16_stream(C + static_cast<unsigned long long>(cta_m + lr1) * N + col8, w1);
                            }
                        }
                    }
                    asm volatile("" ::: "memory");
                }
            }
        }

        // Race-free self-resetting atomic work-stealing across all resident CTAs.
        task_slot ^= 1u;
        if (ep_tid == 0u) {
            unsigned next_t;
            if (task_count <= gridDim.x) {
                next_t = task_count;
            } else {
                const unsigned raw = atomicAdd(next_task_ptr, 1u);
                next_t = gridDim.x + raw;
                if (raw == task_count - 1u) {
                    atomicExch(next_task_ptr, 0u);
                }
            }
            v32_st_su32(sbase + L::OFF_TASK + task_slot * 4u, next_t);
        }
        __syncthreads();
        task = v32_ld_su32(sbase + L::OFF_TASK + task_slot * 4u);
    }
}

} // namespace

// =========================================================================
// Single-read fused BF16 -> NVFP4 activation quantizer + shared_gate dot
// =========================================================================
// Quantizes BF16 `src [rows, 2560]` into:
// - `xq [rows, 1280]` packed E2M1 nibbles (pinned in L2 via `L2::evict_last`)
// - `xs [rows, 160]` flat UE4M3 block-16 scales
// - `x_scale2 [rows]` per-row FP32 scale2 = `row_amax / 2688.0f`
// Simultaneously computes `ss_out[r] = sigmoid(dot(src[r], shared_gate))`
// (exact 256-thread strided order matching `qwen4_mxfp8_quant_rows_v16q4_gate`)
// in a single global read of BF16 `src`.
// Launch: grid (min(rows, cap), 1, 1), block (256, 1, 1).
extern "C" __global__ __launch_bounds__(256) void qwen4_nvfp4_quant_rows_v32_gate(
    const __nv_bfloat16* __restrict__ src,
    unsigned char* __restrict__ xq,
    unsigned char* __restrict__ xs,
    float* __restrict__ x_scale2,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out,
    unsigned int rows
) {
    constexpr unsigned kHidden = oi640::HIDDEN;              // 2560
    constexpr unsigned kHalfHidden = kHidden / 2u;           // 1280
    constexpr unsigned kGroupsPerRow = kHidden / 16u;        // 160
    constexpr unsigned kBlendThreads = 256u;
    constexpr unsigned kGateTerms = kHidden / kBlendThreads; // 10

    const unsigned bt = threadIdx.x;
    const unsigned warp = bt >> 5;
    const unsigned lane = bt & 31u;

    uint64_t pol_el = 0;
    uint64_t pol_ef = 0;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));

    const bool gate_aligned16 = (shared_gate != nullptr) &&
                                ((reinterpret_cast<uintptr_t>(shared_gate) & 15u) == 0u);
    auto load_bf16x8 = [](const __nv_bfloat16* p, bool aligned16) -> uint4 {
        if (aligned16) {
            return __ldg(reinterpret_cast<const uint4*>(p));
        }
        unsigned short h[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            h[i] = *reinterpret_cast<const unsigned short*>(p + i);
        }
        return make_uint4(
            static_cast<unsigned>(h[0]) | (static_cast<unsigned>(h[1]) << 16),
            static_cast<unsigned>(h[2]) | (static_cast<unsigned>(h[3]) << 16),
            static_cast<unsigned>(h[4]) | (static_cast<unsigned>(h[5]) << 16),
            static_cast<unsigned>(h[6]) | (static_cast<unsigned>(h[7]) << 16));
    };

    __shared__ __align__(16) float s_prod[kGateTerms][kBlendThreads];
    __shared__ float s_amax[8];
    __shared__ float s_dot[2][8];

    const unsigned prod_b0 = (lane * 8u) ^ (((lane >> 2) & 1u) << 2);
    const unsigned prod_b1 = prod_b0 ^ 4u;
    const unsigned swz_bt = bt ^ (((bt >> 5) & 1u) << 2);
    unsigned row_parity = 0u;
    const bool pin_xq = (rows <= V32_XQ_PIN_MAX_ROWS);

    auto stage_gate_prod = [&](unsigned pass, const float (&v)[8]) {
        const unsigned term_idx = pass * 8u + warp;
        const uint4 g_u4 = load_bf16x8(shared_gate + pass * 2048u + bt * 8u, gate_aligned16);
        *reinterpret_cast<float4*>(&s_prod[term_idx][prod_b0]) = make_float4(
            v[0] * __uint_as_float(g_u4.x << 16),
            v[1] * __uint_as_float(g_u4.x & 0xffff0000u),
            v[2] * __uint_as_float(g_u4.y << 16),
            v[3] * __uint_as_float(g_u4.y & 0xffff0000u));
        *reinterpret_cast<float4*>(&s_prod[term_idx][prod_b1]) = make_float4(
            v[4] * __uint_as_float(g_u4.z << 16),
            v[5] * __uint_as_float(g_u4.z & 0xffff0000u),
            v[6] * __uint_as_float(g_u4.w << 16),
            v[7] * __uint_as_float(g_u4.w & 0xffff0000u));
    };

    for (unsigned r = blockIdx.x; r < rows; r += gridDim.x) {
        const __nv_bfloat16* x_row = src + static_cast<unsigned long long>(r) * kHidden;
        const uint4* xr4 = reinterpret_cast<const uint4*>(x_row);

        // Pass 0: all 256 threads load x_row[bt*8 .. +7] (128 NVFP4 groups).
        uint4 w0;
        asm volatile("ld.global.ca.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                     : "=r"(w0.x), "=r"(w0.y), "=r"(w0.z), "=r"(w0.w)
                     : "l"(xr4 + bt), "l"(pol_ef));
        const unsigned u0[4] = {w0.x, w0.y, w0.z, w0.w};
        float v0[8];
        float gmax0 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float a = __uint_as_float(u0[i] << 16);
            const float b = __uint_as_float(u0[i] & 0xffff0000u);
            v0[2 * i] = a;
            v0[2 * i + 1] = b;
            gmax0 = fmaxf(gmax0, fmaxf(fabsf(a), fabsf(b)));
        }
        if (shared_gate != nullptr && ss_out != nullptr) {
            stage_gate_prod(0u, v0);
        }
        gmax0 = fmaxf(gmax0, __shfl_xor_sync(0xffffffffu, gmax0, 1));

        // Pass 1: threads 0..63 load x_row[2048 + bt*8 .. +7] (remaining 32 NVFP4 groups).
        float v1[8];
        float gmax1 = 0.0f;
        if (bt < 64u) {
            uint4 w1;
            asm volatile("ld.global.ca.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                         : "=r"(w1.x), "=r"(w1.y), "=r"(w1.z), "=r"(w1.w)
                         : "l"(xr4 + 256u + bt), "l"(pol_ef));
            const unsigned u1[4] = {w1.x, w1.y, w1.z, w1.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float a = __uint_as_float(u1[i] << 16);
                const float b = __uint_as_float(u1[i] & 0xffff0000u);
                v1[2 * i] = a;
                v1[2 * i + 1] = b;
                gmax1 = fmaxf(gmax1, fmaxf(fabsf(a), fabsf(b)));
            }
            if (shared_gate != nullptr && ss_out != nullptr) {
                stage_gate_prod(1u, v1);
            }
            gmax1 = fmaxf(gmax1, __shfl_xor_sync(0xffffffffu, gmax1, 1));
        }

        // Single CTA-wide reduction for per-row absmax (also publishes s_prod!).
        float wmax = fmaxf(gmax0, gmax1);
        #pragma unroll
        for (int d = 16; d >= 2; d >>= 1) {
            wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffffu, wmax, d));
        }
        if (lane == 0u) s_amax[warp] = wmax;
        __syncthreads();

        float row_amax = 0.0f;
        #pragma unroll
        for (unsigned w = 0u; w < 8u; ++w) {
            row_amax = fmaxf(row_amax, s_amax[w]);
        }
        const bool positive = isfinite(row_amax) && (row_amax > 1e-30f);
        const float row_scale2 = positive ? __fdiv_rn(row_amax, 2688.0f) : 1.0f;
        const float inv_s2_over_6 = positive ? __fdiv_rn(448.0f, row_amax) : 0.0f;
        if (bt == 0u) {
            x_scale2[r] = row_scale2;
        }

        unsigned char* const xq_row = xq + static_cast<unsigned long long>(r) * kHalfHidden;
        unsigned char* const xs_row = xs + static_cast<unsigned long long>(r) * kGroupsPerRow;

        // Quantize Pass 0 groups (0..127) directly from registers.
        {
            const float sf_f = (gmax0 > 0.0f) ? fminf(__fmul_rn(gmax0, inv_s2_over_6), 448.0f) : 0.0f;
            const unsigned sf_b = v32_rn_ue4m3(sf_f);
            const float eff = __fmul_rn(v32_e4m3_to_float(sf_b), row_scale2);
            const float inv_eff = (eff > 0.0f) ? __fdiv_rn(1.0f, eff) : 0.0f;
            const unsigned packed0 = v32_cvt_f32x8_to_e2m1x8(v0, inv_eff);
            unsigned* dst0 = reinterpret_cast<unsigned*>(xq_row) + bt;
            if (pin_xq) v32_st4_l2_el(dst0, packed0, pol_el);
            else        *dst0 = packed0;

            const unsigned s1 = __shfl_down_sync(0xffffffffu, sf_b, 2);
            const unsigned s2 = __shfl_down_sync(0xffffffffu, sf_b, 4);
            const unsigned s3 = __shfl_down_sync(0xffffffffu, sf_b, 6);
            if ((lane & 7u) == 0u) {
                const unsigned sf4 = (sf_b & 0xffu) | ((s1 & 0xffu) << 8) |
                                     ((s2 & 0xffu) << 16) | ((s3 & 0xffu) << 24);
                unsigned* sdst0 = reinterpret_cast<unsigned*>(xs_row) + (bt >> 3);
                if (pin_xq) v32_st4_l2_el(sdst0, sf4, pol_el);
                else        *sdst0 = sf4;
            }
        }

        // Quantize Pass 1 groups (128..159) directly from registers.
        if (bt < 64u) {
            const float sf_f = (gmax1 > 0.0f) ? fminf(__fmul_rn(gmax1, inv_s2_over_6), 448.0f) : 0.0f;
            const unsigned sf_b = v32_rn_ue4m3(sf_f);
            const float eff = __fmul_rn(v32_e4m3_to_float(sf_b), row_scale2);
            const float inv_eff = (eff > 0.0f) ? __fdiv_rn(1.0f, eff) : 0.0f;
            const unsigned packed1 = v32_cvt_f32x8_to_e2m1x8(v1, inv_eff);
            unsigned* dst1 = reinterpret_cast<unsigned*>(xq_row) + 256u + bt;
            if (pin_xq) v32_st4_l2_el(dst1, packed1, pol_el);
            else        *dst1 = packed1;

            const unsigned s1 = __shfl_down_sync(0xffffffffu, sf_b, 2);
            const unsigned s2 = __shfl_down_sync(0xffffffffu, sf_b, 4);
            const unsigned s3 = __shfl_down_sync(0xffffffffu, sf_b, 6);
            if ((lane & 7u) == 0u) {
                const unsigned sf4 = (sf_b & 0xffu) | ((s1 & 0xffu) << 8) |
                                     ((s2 & 0xffu) << 16) | ((s3 & 0xffu) << 24);
                unsigned* sdst1 = reinterpret_cast<unsigned*>(xs_row) + 32u + (bt >> 3);
                if (pin_xq) v32_st4_l2_el(sdst1, sf4, pol_el);
                else        *sdst1 = sf4;
            }
        }

        // Finish exact 0-ULP shared-gate dot product from staged s_prod.
        if (ss_out != nullptr && shared_gate != nullptr) {
            float d = 0.0f;
            #pragma unroll
            for (unsigned i = 0u; i < kGateTerms; ++i) {
                d += s_prod[i][swz_bt];
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                d += __shfl_down_sync(0xffffffffu, d, offset);
            }
            if (lane == 0u) s_dot[row_parity][warp] = d;
            __syncthreads();
            if (bt == 0u) {
                float total = 0.0f;
                #pragma unroll
                for (unsigned w = 0u; w < 8u; ++w) total += s_dot[row_parity][w];
                ss_out[r] = 1.0f / (1.0f + __expf(-total));
            }
            row_parity ^= 1u;
        } else {
            if (ss_out != nullptr && bt == 0u) {
                ss_out[r] = 1.0f;
            }
            if (r + gridDim.x < rows) {
                __syncthreads();
            }
        }
    }
}

// =========================================================================
// V32_W4A4 GATEUP Entry Points (256 threads, 2 CTAs/SM, <= 41.6 KiB SMEM)
// =========================================================================

// Hybrid `gateup` mode (flat scales): W4A4 GATEUP -> MXFP8 `oq`/`os` for V31_WS DOWN.
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_gateup_v32(
    const unsigned char* __restrict__ xq,
    const unsigned char* __restrict__ xs,
    const float* __restrict__ x_scale2,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    unsigned char* __restrict__ out_q,
    unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<true, false, 0, true, false>(
        xq, xs, x_scale2, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals,
        up_packed_ptrs, up_scale_ptrs, up_scale2_vals, nullptr, out_q, out_s,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        oq_scale2_buf);
}

// Hybrid `gateup` mode (TS4E-128 scales): W4A4 GATEUP -> MXFP8 `oq`/`os` for V31_WS DOWN.
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_gateup_v32_ts4e(
    const unsigned char* __restrict__ xq,
    const unsigned char* __restrict__ xs,
    const float* __restrict__ x_scale2,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    unsigned char* __restrict__ out_q,
    unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<true, false, 0, true, true>(
        xq, xs, x_scale2, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals,
        up_packed_ptrs, up_scale_ptrs, up_scale2_vals, nullptr, out_q, out_s,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        oq_scale2_buf);
}

// Full `all` W4A4 mode (flat scales): W4A4 GATEUP -> NVFP4 `oq [expanded, 320]` + `os [expanded, 40]`
// + optional per-(orow, nt) dynamic `oq_scale2_buf [expanded, 10]`.
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_gateup_v32_oq4(
    const unsigned char* __restrict__ xq,
    const unsigned char* __restrict__ xs,
    const float* __restrict__ x_scale2,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    unsigned char* __restrict__ out_q,
    unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<true, true, 0, true, false>(
        xq, xs, x_scale2, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals,
        up_packed_ptrs, up_scale_ptrs, up_scale2_vals, nullptr, out_q, out_s,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        oq_scale2_buf);
}

// Full `all` W4A4 mode (TS4E-128 scales): W4A4 GATEUP -> NVFP4 `oq` + `os` + dynamic `oq_scale2_buf`.
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_gateup_v32_oq4_ts4e(
    const unsigned char* __restrict__ xq,
    const unsigned char* __restrict__ xs,
    const float* __restrict__ x_scale2,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    unsigned char* __restrict__ out_q,
    unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<true, true, 0, true, true>(
        xq, xs, x_scale2, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals,
        up_packed_ptrs, up_scale_ptrs, up_scale2_vals, nullptr, out_q, out_s,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        oq_scale2_buf);
}

// =========================================================================
// V32_W4A4 DOWN Entry Points (256 threads, 2 CTAs/SM, <= 44.9 KiB SMEM)
// =========================================================================

// Full `all` W4A4 DOWN (flat scales): NVFP4 `oq`/`os` + `oq_scale2_buf` -> MXFP8 `slot_q`/`slot_s`.
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_down_v32(
    const unsigned char* __restrict__ oq,
    const unsigned char* __restrict__ os,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    unsigned char* __restrict__ slot_q,
    unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    const float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<false, false, 3, true, false>(
        oq, os, nullptr, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        nullptr, oq_scale2_buf, slot_q, slot_s);
}

// Full `all` W4A4 DOWN (TS4E-128 scales): NVFP4 `oq`/`os` + `oq_scale2_buf` -> MXFP8 `slot_q`/`slot_s`.
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_down_v32_ts4e(
    const unsigned char* __restrict__ oq,
    const unsigned char* __restrict__ os,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    unsigned char* __restrict__ slot_q,
    unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    const float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<false, false, 3, true, true>(
        oq, os, nullptr, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        nullptr, oq_scale2_buf, slot_q, slot_s);
}

// Full `all` W4A4 DOWN (flat scales): NVFP4 `oq`/`os` + `oq_scale2_buf` -> BF16 `C`.
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_down_v32_bf16(
    const unsigned char* __restrict__ oq,
    const unsigned char* __restrict__ os,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    const float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<false, false, 0, true, false>(
        oq, os, nullptr, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        nullptr, oq_scale2_buf);
}

// Full `all` W4A4 DOWN (TS4E-128 scales): NVFP4 `oq`/`os` + `oq_scale2_buf` -> BF16 `C`.
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_w4a4_down_v32_bf16_ts4e(
    const unsigned char* __restrict__ oq,
    const unsigned char* __restrict__ os,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    float oq_scale2,
    const float* __restrict__ oq_scale2_buf
) {
    v32_w4a4_impl<false, false, 0, true, true>(
        oq, os, nullptr, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, oq_scale2,
        nullptr, oq_scale2_buf);
}

// Dynamic SMEM query helper: [flat_gu, flat_dn, ts4e_gu, ts4e_dn].
extern "C" __global__ void qwen4_moe_w4a4_v32_smem_bytes(unsigned* out) {
    out[0] = V32W4a4Smem<true, false>::BYTES;
    out[1] = V32W4a4Smem<false, false>::BYTES;
    out[2] = V32W4a4Smem<true, true>::BYTES;
    out[3] = V32W4a4Smem<false, true>::BYTES;
}
