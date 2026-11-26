// SPDX-License-Identifier: AGPL-3.0-only
//
// V17: W4A8 compact routed-MoE down-projection and unpermute-reduce/blend suite (128 x 256 B-column
// tile per 512-thread CTA, dynamic smem, 3-deep A and raw-B rings, k64 stage cadence, QMMA.SF e4m3):
//  * cooperative integer PRMT-LUT weight transcode runs once per CTA k64 stage (one __syncthreads()
//    barrier per k64 stage) with register-direct epilogues;
//  * exports MXFP8 slot-compressed down kernels (`v17q`, residual-two `v17qr2`, plus BF16/FP32-reduce
//    down variants), `moe_unpermute_reduce_mxfp8*` (standalone and fused shared-expert blend), and
//    the graph-capture-safe `moe_status_accumulate_err` helper.
// There is no exact-weight W4A8 on this MMA: e2m1 x e4m3 needs up to 6 significant bits.
#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define OI640_STEP_K 32
#define OI640_TRANSPOSED 1
#define flash_next_orig_i640_prefill flash_next_f32_transposed_compact_k32_v17
#include "moe_w4a16_orig_i640_compact_prefill.cuh"
#include <type_traits>
namespace oi640 = flash_next_f32_transposed_compact_k32_v17;

namespace {
constexpr unsigned V10_TILE_M = 128;

__device__ __forceinline__ void v10_cp16(void* smem, const void* gmem, bool pred) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void v10_cp4(void* smem, const void* gmem, bool pred) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(s), "l"(gmem), "r"(pred ? 4 : 0));
}
__device__ __forceinline__ void v10_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void v10_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ unsigned v10_prmt(unsigned a, unsigned b, unsigned sel) {
    unsigned d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
}
__device__ __forceinline__ unsigned v10_pack_e4m3x4(float a, float b, float c, float d) {
    unsigned short p0, p1;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p0) : "f"(a), "f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p1) : "f"(c), "f"(d));
    return v10_prmt(static_cast<unsigned>(p0), static_cast<unsigned>(p1), 0x5410u);
}
__device__ __forceinline__ int v10_floor_log2_e4m3(unsigned b) {
    const unsigned e = (b >> 3) & 0xfu, m = b & 7u;
    if (e != 0) return static_cast<int>(e) - 7;
    if (m == 0) return -(1 << 30);
    return (31 - __clz(static_cast<int>(m))) - 9;
}
__device__ __forceinline__ float v10_e4m3_to_float(unsigned b) {
    unsigned h;
    const unsigned short pair = static_cast<unsigned short>(b & 0xffu);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(pair));
    return __low2float(*reinterpret_cast<const __half2*>(&h));
}
__device__ __forceinline__ float2 v17_e4m3x2_to_float2(unsigned short pair) {
    unsigned h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(pair));
    return __half22float2(*reinterpret_cast<const __half2*>(&h));
}
// Four E2M1 codes (x: 16 bits, code i in nibble i) -> four E4M3 bytes via the 8-byte magnitude
// table {tlo, thi}; the sign bit of code i becomes bit 7 of byte i.
__device__ __forceinline__ unsigned v10_e2m1x4_to_e4m3x4(unsigned x, unsigned tlo, unsigned thi) {
    const unsigned mag = v10_prmt(tlo, thi, x & 0x7777u);
    const unsigned sgn = v10_prmt(x, x << 4, 0x9D8Cu);
    return mag | (sgn & 0x80808080u);
}

// b12x quantize_block_fp8_mx on 32 values (bit-identical to qwen4_mxfp8_quant_rows).
__device__ __forceinline__ void v10_mxfp8_block(const float (&v)[32], unsigned (&q)[8], unsigned& s) {
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
    if (!(amax > 0.0f)) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) q[i] = 0u;
        s = 0;
        return;
    }
    int e;
    const float m = frexpf(__fmul_rn(amax, 1.0f / 448.0f), &e);
    int k = (m == 0.5f) ? e - 1 : e;
    k = max(-127, min(127, k));
    const float inv = __int_as_float((127 - k) << 23);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        q[i] = v10_pack_e4m3x4(
            v[4 * i + 0] * inv,
            v[4 * i + 1] * inv,
            v[4 * i + 2] * inv,
            v[4 * i + 3] * inv);
    }
    s = static_cast<unsigned>(k + 127);
}

// MXFP8 plus two deterministic residual corrections for one MX32 block. Each
// u16 is lane[4:0] + residual-e4m3[12:5]; the low record is the largest
// absolute normalized quantization error and ties retain the lower lane. The
// residual shares the block's existing UE8M0 scale. Quantization and residual
// selection share one ascending pass so the fused epilogue does not rescan the
// 32 inputs or re-extract its eight packed payload words.
__device__ __forceinline__ void v17_mxfp8_block_r2(
    const float (&v)[32], unsigned (&q)[8], unsigned& s, unsigned& r2) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
    if (!(amax > 0.0f)) {
#pragma unroll
        for (int i = 0; i < 8; ++i) q[i] = 0u;
        s = 0;
        // Valid producer inputs are finite, so this branch contains only
        // signed zeros. Preserve the old scan's lane-0/lane-1 FP8 sign bits.
        const unsigned e0 = (__float_as_uint(v[0]) >> 24) & 0x80u;
        const unsigned e1 = (__float_as_uint(v[1]) >> 24) & 0x80u;
        const unsigned rec0 = e0 << 5;
        const unsigned rec1 = 1u | (e1 << 5);
        r2 = rec0 | (rec1 << 16);
        return;
    }
    int e;
    const float m = frexpf(__fmul_rn(amax, 1.0f / 448.0f), &e);
    int k = (m == 0.5f) ? e - 1 : e;
    k = max(-127, min(127, k));
    const float inv = __int_as_float((127 - k) << 23);
    int best0 = 0, best1 = 1;
    float err0 = -1.0f, err1 = -1.0f;
    float residual0 = 0.0f, residual1 = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned w = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = 4 * i + j;
            const float normalized = __fmul_rn(v[idx], inv);
            const unsigned qb = static_cast<unsigned>(
                __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3));
            w |= qb << (8 * j);
            const float residual = __fsub_rn(normalized, v10_e4m3_to_float(qb));
            const float error = fabsf(residual);
            if (error > err0) {
                err1 = err0;
                best1 = best0;
                residual1 = residual0;
                err0 = error;
                best0 = idx;
                residual0 = residual;
            } else if (error > err1) {
                err1 = error;
                best1 = idx;
                residual1 = residual;
            }
        }
        q[i] = w;
    }
    s = static_cast<unsigned>(k + 127);
    const unsigned e0 = static_cast<unsigned>(
        __nv_cvt_float_to_fp8(residual0, __NV_SATFINITE, __NV_E4M3));
    const unsigned e1 = static_cast<unsigned>(
        __nv_cvt_float_to_fp8(residual1, __NV_SATFINITE, __NV_E4M3));
    const unsigned rec0 = static_cast<unsigned>(best0) | ((e0 & 0xffu) << 5);
    const unsigned rec1 = static_cast<unsigned>(best1) | ((e1 & 0xffu) << 5);
    r2 = rec0 | (rec1 << 16);
}

struct v18_r2_candidate {
    float error;
    float residual;
    unsigned index;
};

__device__ __forceinline__ void v18_r2_insert(
    v18_r2_candidate value, v18_r2_candidate& first, v18_r2_candidate& second) {
    // Equal-error candidates arrive in ascending source-index order, so
    // strict greater-than retains the legacy lower-index tie winner.
    if (value.error > first.error) {
        second = first;
        first = value;
    } else if (value.error > second.error) {
        second = value;
    }
}

// One adjacent lane pair encodes one MX32 block. Each lane handles a contiguous
// 16-value half; XOR-1 exchanges the maximum and the two local residual
// winners. The even lane performs the stable four-candidate merge.
__device__ __forceinline__ void v18_mxfp8_half_r2(
    const float (&v)[16], unsigned half, unsigned (&q)[4], unsigned& s, unsigned& r2) {
    float local_amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) local_amax = fmaxf(local_amax, fabsf(v[i]));
    const float amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffffu, local_amax, 1));
    v18_r2_candidate first = {-1.0f, 0.0f, 32u};
    v18_r2_candidate second = {-1.0f, 0.0f, 32u};
    if (!(amax > 0.0f)) {
#pragma unroll
        for (int i = 0; i < 4; ++i) q[i] = 0u;
        s = 0;
        first = {0.0f, v[0], 16u * half};
        second = {0.0f, v[1], 16u * half + 1u};
    } else {
        int e;
        const float m = frexpf(__fmul_rn(amax, 1.0f / 448.0f), &e);
        int k = (m == 0.5f) ? e - 1 : e;
        k = max(-127, min(127, k));
        const float inv = __int_as_float((127 - k) << 23);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned w = 0;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int local = 4 * i + 2 * j;
                const float2 normalized = make_float2(
                    __fmul_rn(v[local], inv), __fmul_rn(v[local + 1], inv));
                const unsigned pair = static_cast<unsigned>(
                    __nv_cvt_float2_to_fp8x2(normalized, __NV_SATFINITE, __NV_E4M3));
                w |= pair << (16 * j);
                const __half2_raw decoded_raw = __nv_cvt_fp8x2_to_halfraw2(
                    static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
                const float2 decoded = __half22float2(
                    *reinterpret_cast<const __half2*>(&decoded_raw));
                const float residual0 = __fsub_rn(normalized.x, decoded.x);
                const float residual1 = __fsub_rn(normalized.y, decoded.y);
                v18_r2_insert(
                    {fabsf(residual0), residual0, 16u * half + static_cast<unsigned>(local)},
                    first, second);
                v18_r2_insert(
                    {fabsf(residual1), residual1, 16u * half + static_cast<unsigned>(local + 1)},
                    first, second);
            }
            q[i] = w;
        }
        s = static_cast<unsigned>(k + 127);
    }
    const v18_r2_candidate peer_first = {
        __shfl_xor_sync(0xffffffffu, first.error, 1),
        __shfl_xor_sync(0xffffffffu, first.residual, 1),
        __shfl_xor_sync(0xffffffffu, first.index, 1)};
    const v18_r2_candidate peer_second = {
        __shfl_xor_sync(0xffffffffu, second.error, 1),
        __shfl_xor_sync(0xffffffffu, second.residual, 1),
        __shfl_xor_sync(0xffffffffu, second.index, 1)};
    r2 = 0;
    if (half == 0) {
        v18_r2_candidate merged0 = {-1.0f, 0.0f, 32u};
        v18_r2_candidate merged1 = {-1.0f, 0.0f, 32u};
        v18_r2_insert(first, merged0, merged1);
        v18_r2_insert(second, merged0, merged1);
        v18_r2_insert(peer_first, merged0, merged1);
        v18_r2_insert(peer_second, merged0, merged1);
        const unsigned e0 = static_cast<unsigned>(
            __nv_cvt_float_to_fp8(merged0.residual, __NV_SATFINITE, __NV_E4M3));
        const unsigned e1 = static_cast<unsigned>(
            __nv_cvt_float_to_fp8(merged1.residual, __NV_SATFINITE, __NV_E4M3));
        const unsigned rec0 = merged0.index | ((e0 & 0xffu) << 5);
        const unsigned rec1 = merged1.index | ((e1 & 0xffu) << 5);
        r2 = rec0 | (rec1 << 16);
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 1200
__device__ __forceinline__ float v17_scale_ue8m0_pair(float t, unsigned ea, unsigned eb) {
    const unsigned s = ea + eb;
    const unsigned h1 = s >> 1;
    const unsigned h2 = s - h1;
    const unsigned fa_bits = (ea == 255u || eb == 255u) ? 0x7fc00000u : (h1 << 23);
    return (t * __uint_as_float(fa_bits)) * __uint_as_float(h2 << 23);
}
#endif

template <int BID>
__device__ __forceinline__ void v10_qmma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1,
                                         unsigned sa, unsigned sb) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%14, %13};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb),
          "n"(BID), "n"(0), "n"(0));
#else
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(t0), "=f"(t1), "=f"(t2), "=f"(t3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "f"(t0), "f"(t1), "f"(t2), "f"(t3));
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    const unsigned grp = lane & ~3u;
    const unsigned tig = lane & 3u;
    const unsigned sa_b = (sa >> (BID * 8)) & 0xffu;
    const unsigned sb_b = sb & 0xffu;
    const unsigned ea_lo = __shfl_sync(0xffffffffu, sa_b, grp);
    const unsigned ea_hi = __shfl_sync(0xffffffffu, sa_b, grp | 1u);
    const unsigned eb_even = __shfl_sync(0xffffffffu, sb_b, tig << 3);
    const unsigned eb_odd = __shfl_sync(0xffffffffu, sb_b, (tig << 3) | 4u);
    d[0] += v17_scale_ue8m0_pair(t0, ea_lo, eb_even);
    d[1] += v17_scale_ue8m0_pair(t1, ea_lo, eb_odd);
    d[2] += v17_scale_ue8m0_pair(t2, ea_hi, eb_even);
    d[3] += v17_scale_ue8m0_pair(t3, ea_hi, eb_odd);
#endif
}
template <int BID, int SFI_B = 0>
__device__ __forceinline__ void v17_qmma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1,
                                         unsigned sa, unsigned sb) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%14, %13};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb),
          "n"(BID), "n"(0), "n"(SFI_B));
#else
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(t0), "=f"(t1), "=f"(t2), "=f"(t3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "f"(t0), "f"(t1), "f"(t2), "f"(t3));
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    const unsigned grp = lane & ~3u;
    const unsigned tig = lane & 3u;
    const unsigned sa_b = (sa >> (BID * 8)) & 0xffu;
    const unsigned sb_b = (sb >> (SFI_B * 8)) & 0xffu;
    const unsigned ea_lo = __shfl_sync(0xffffffffu, sa_b, grp);
    const unsigned ea_hi = __shfl_sync(0xffffffffu, sa_b, grp | 1u);
    const unsigned eb_even = __shfl_sync(0xffffffffu, sb_b, tig << 3);
    const unsigned eb_odd = __shfl_sync(0xffffffffu, sb_b, (tig << 3) | 4u);
    d[0] += v17_scale_ue8m0_pair(t0, ea_lo, eb_even);
    d[1] += v17_scale_ue8m0_pair(t1, ea_lo, eb_odd);
    d[2] += v17_scale_ue8m0_pair(t2, ea_hi, eb_even);
    d[3] += v17_scale_ue8m0_pair(t3, ea_hi, eb_odd);
#endif
}
template <int BID>
__device__ __forceinline__ void v17_qmma8(float (&d)[8][4], const unsigned (&a)[4], const unsigned (&bfr)[8][2],
                                          unsigned sa, uint2 sb8) {
    v17_qmma<BID, 0>(d[0], a, bfr[0][0], bfr[0][1], sa, sb8.x);
    v17_qmma<BID, 1>(d[1], a, bfr[1][0], bfr[1][1], sa, sb8.x);
    v17_qmma<BID, 2>(d[2], a, bfr[2][0], bfr[2][1], sa, sb8.x);
    v17_qmma<BID, 3>(d[3], a, bfr[3][0], bfr[3][1], sa, sb8.x);
    v17_qmma<BID, 0>(d[4], a, bfr[4][0], bfr[4][1], sa, sb8.y);
    v17_qmma<BID, 1>(d[5], a, bfr[5][0], bfr[5][1], sa, sb8.y);
    v17_qmma<BID, 2>(d[6], a, bfr[6][0], bfr[6][1], sa, sb8.y);
    v17_qmma<BID, 3>(d[7], a, bfr[7][0], bfr[7][1], sa, sb8.y);
}
template <int BID>
__device__ __forceinline__ void v17_qmma4_lo(float (&d)[8][4], const unsigned (&a)[4], const unsigned (&bfr)[8][2],
                                             unsigned sa, unsigned lo) {
    v17_qmma<BID, 0>(d[0], a, bfr[0][0], bfr[0][1], sa, lo);
    v17_qmma<BID, 1>(d[1], a, bfr[1][0], bfr[1][1], sa, lo);
    v17_qmma<BID, 2>(d[2], a, bfr[2][0], bfr[2][1], sa, lo);
    v17_qmma<BID, 3>(d[3], a, bfr[3][0], bfr[3][1], sa, lo);
}
template <int BID>
__device__ __forceinline__ void v17_qmma4_hi(float (&d)[8][4], const unsigned (&a)[4], const unsigned (&bfr)[8][2],
                                             unsigned sa, unsigned hi) {
    v17_qmma<BID, 0>(d[4], a, bfr[4][0], bfr[4][1], sa, hi);
    v17_qmma<BID, 1>(d[5], a, bfr[5][0], bfr[5][1], sa, hi);
    v17_qmma<BID, 2>(d[6], a, bfr[6][0], bfr[6][1], sa, hi);
    v17_qmma<BID, 3>(d[7], a, bfr[7][0], bfr[7][1], sa, hi);
}
__device__ __forceinline__ void v10_ldsm4(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void v17_st8_evict_last(void* dst, uint2 v) {
    asm volatile("st.global.wb.v2.u32 [%0], {%1, %2};\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y)
                 : "memory");
}
__device__ __forceinline__ void v17_st8_evict_first(void* dst, uint2 v) {
    asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y)
                 : "memory");
}
__device__ __forceinline__ void v17_st16_stream(void* dst, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}
__device__ __forceinline__ void v17_cp16_plain(void* smem_ptr, const void* gmem_ptr) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void v17_mxfp8_quad(const uint4 w, unsigned (&qq)[2], unsigned& ss) {
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
            qq[i] = v10_pack_e4m3x4(
                v[4 * i + 0] * inv,
                v[4 * i + 1] * inv,
                v[4 * i + 2] * inv,
                v[4 * i + 3] * inv);
        }
        ss = static_cast<unsigned>(k + 127);
    }
}
__device__ __forceinline__ unsigned v17_pack_bf162(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    unsigned u;
    memcpy(&u, &v, sizeof(u));
    return u;
}
__device__ __forceinline__ void v17_mxfp8_reg_quad(
    unsigned u0, unsigned u1, unsigned u2, unsigned u3,
    unsigned tig, unsigned (&qq)[2], unsigned& ss
) {
    const bool odd1 = (tig & 1u) != 0u;
    const unsigned s0 = __shfl_xor_sync(0xffffffffu, odd1 ? u0 : u1, 1);
    const unsigned s2 = __shfl_xor_sync(0xffffffffu, odd1 ? u2 : u3, 1);
    const unsigned t0 = odd1 ? s0 : u0;
    const unsigned t1 = odd1 ? u1 : s0;
    const unsigned t2 = odd1 ? s2 : u2;
    const unsigned t3 = odd1 ? u3 : s2;
    const bool odd2 = (tig & 2u) != 0u;
    const unsigned r0 = __shfl_xor_sync(0xffffffffu, odd2 ? t0 : t2, 2);
    const unsigned r1 = __shfl_xor_sync(0xffffffffu, odd2 ? t1 : t3, 2);
    const uint4 w = make_uint4(
        odd2 ? r0 : t0,
        odd2 ? r1 : t1,
        odd2 ? t2 : r0,
        odd2 ? t3 : r1
    );
    v17_mxfp8_quad(w, qq, ss);
}
template <bool GATEUP>
__device__ __forceinline__ unsigned v17_sf_index(unsigned col) {
    const unsigned gid = col & 7u;
    if constexpr (GATEUP) {
        const unsigned mtx = col >> 7;
        const unsigned wn = (col >> 5) & 3u;
        const unsigned p_low = (col >> 4) & 1u;
        const unsigned sub = (col >> 3) & 1u;
        const unsigned j = (mtx << 2) | (p_low << 1) | sub;
        return (wn << 6) | (gid << 3) | j;
    } else {
        const unsigned wn = col >> 6;
        const unsigned j = (col >> 3) & 7u;
        return (wn << 6) | (gid << 3) | j;
    }
}
} // namespace

// MXFP8 rows: q[r][k] = e4m3_rn(x / 2^E), s[r][k/32] = E + 127 with 2^E = pow2ceil(amax/448)
// per 32-element block (b12x quantize_block_fp8_mx; all-zero block -> byte 0, payload 0).
// One thread per block, 16-byte loads and stores. `cols` = K (multiple of 32).
extern "C" __global__ void qwen4_mxfp8_quant_rows_v17(
    const __nv_bfloat16* __restrict__ src, unsigned char* __restrict__ q, unsigned char* __restrict__ s,
    unsigned int rows, unsigned int cols
) {
    const unsigned long long blocks = static_cast<unsigned long long>(rows) * (cols / 32);
    for (unsigned long long b = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
         b < blocks; b += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
        const uint4* x = reinterpret_cast<const uint4*>(src + b * 32);
        float v[32];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint4 w = x[c];
            const unsigned u[4] = {w.x, w.y, w.z, w.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[8 * c + 2 * i] = __uint_as_float(u[i] << 16);
                v[8 * c + 2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
            }
        }
        unsigned qq[8], ss;
        v10_mxfp8_block(v, qq, ss);
        uint4* out = reinterpret_cast<uint4*>(q + b * 32);
        out[0] = make_uint4(qq[0], qq[1], qq[2], qq[3]);
        out[1] = make_uint4(qq[4], qq[5], qq[6], qq[7]);
        s[b] = static_cast<unsigned char>(ss);
    }
}

// Standalone/capture twin of the fused v17qr2 epilogue. It is the raw parity
// oracle for the exact producer encoding.
extern "C" __global__ void qwen4_mxfp8_quant_rows_v17r2(
    const __nv_bfloat16* __restrict__ src, unsigned char* __restrict__ q,
    unsigned char* __restrict__ s, unsigned* __restrict__ r2,
    unsigned int rows, unsigned int cols
) {
    if (src == nullptr || q == nullptr || s == nullptr || r2 == nullptr || cols % 32 != 0 ||
        (reinterpret_cast<unsigned long long>(src) & 15ull) != 0 ||
        (reinterpret_cast<unsigned long long>(q) & 15ull) != 0 ||
        (reinterpret_cast<unsigned long long>(r2) & 3ull) != 0) __trap();
    const unsigned long long blocks = static_cast<unsigned long long>(rows) * (cols / 32);
    for (unsigned long long b = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
         b < blocks; b += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
        const uint4* x = reinterpret_cast<const uint4*>(src + b * 32);
        float v[32];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint4 w = x[c];
            const unsigned u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[8 * c + 2 * i] = __uint_as_float(u[i] << 16);
                v[8 * c + 2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
            }
        }
        unsigned qq[8], ss, rr;
        v17_mxfp8_block_r2(v, qq, ss, rr);
        uint4* out = reinterpret_cast<uint4*>(q + b * 32);
        out[0] = make_uint4(qq[0], qq[1], qq[2], qq[3]);
        out[1] = make_uint4(qq[4], qq[5], qq[6], qq[7]);
        s[b] = static_cast<unsigned char>(ss);
        r2[b] = rr;
    }
}

// Standalone twin of the paired-lane fused epilogue. Qualification compares
// this path byte-for-byte with the serial v17r2 encoder before timing.
extern "C" __global__ void qwen4_mxfp8_quant_rows_v18pair(
    const __nv_bfloat16* __restrict__ src, unsigned char* __restrict__ q,
    unsigned char* __restrict__ s, unsigned* __restrict__ r2,
    unsigned int rows, unsigned int cols
) {
    if (src == nullptr || q == nullptr || s == nullptr || r2 == nullptr || cols % 32 != 0 ||
        blockDim.x % 32 != 0 || (reinterpret_cast<unsigned long long>(src) & 15ull) != 0 ||
        (reinterpret_cast<unsigned long long>(q) & 15ull) != 0 ||
        (reinterpret_cast<unsigned long long>(r2) & 3ull) != 0) __trap();
    const unsigned long long blocks = static_cast<unsigned long long>(rows) * (cols / 32);
    const unsigned lane = threadIdx.x & 31u, pair = lane >> 1, half = lane & 1;
    const unsigned long long warp =
        (blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x) >> 5;
    const unsigned long long warp_stride =
        (gridDim.x * static_cast<unsigned long long>(blockDim.x)) >> 5;
    for (unsigned long long base = warp * 16; base < blocks; base += warp_stride * 16) {
        const unsigned long long b = base + pair;
        const bool valid = b < blocks;
        float v[16];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const uint4 w = valid
                ? reinterpret_cast<const uint4*>(src + b * 32 + half * 16)[c]
                : make_uint4(0, 0, 0, 0);
            const unsigned u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[8 * c + 2 * i] = __uint_as_float(u[i] << 16);
                v[8 * c + 2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
            }
        }
        unsigned qq[4], ss, rr;
        v18_mxfp8_half_r2(v, half, qq, ss, rr);
        if (valid) {
            *reinterpret_cast<uint4*>(q + b * 32 + half * 16) =
                make_uint4(qq[0], qq[1], qq[2], qq[3]);
            if (half == 0) {
                s[b] = static_cast<unsigned char>(ss);
                r2[b] = rr;
            }
        }
    }
}

// Raw-gate decoder for the standalone producer above. Serving uses the
// unpermute consumers below; this kernel makes the encoded row itself directly
// comparable with a CPU reconstruction and is not selected by the engine.
extern "C" __global__ void qwen4_mxfp8_dequant_rows_v17r2(
    const unsigned char* __restrict__ q, const unsigned char* __restrict__ s,
    const unsigned* __restrict__ r2, __nv_bfloat16* __restrict__ out,
    unsigned int rows, unsigned int cols
) {
    if (q == nullptr || s == nullptr || r2 == nullptr || out == nullptr || cols % 32 != 0 ||
        (reinterpret_cast<unsigned long long>(r2) & 3ull) != 0) __trap();
    const unsigned long long values = static_cast<unsigned long long>(rows) * cols;
    const unsigned int groups_per_row = cols / 32;
    for (unsigned long long i = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
         i < values; i += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
        const unsigned long long row = i / cols;
        const unsigned int col = static_cast<unsigned int>(i - row * cols);
        const unsigned long long group = row * groups_per_row + col / 32;
        const unsigned int lane = col & 31u;
        float normalized = v10_e4m3_to_float(q[i]);
        const unsigned packed = r2[group];
        const unsigned rec0 = packed & 0xffffu;
        const unsigned rec1 = packed >> 16;
        if ((rec0 & 31u) == lane && (((rec0 >> 5) & 0xffu) & 0x7fu) != 0)
            normalized = __fadd_rn(normalized, v10_e4m3_to_float((rec0 >> 5) & 0xffu));
        if ((rec1 & 31u) == lane && (((rec1 >> 5) & 0xffu) & 0x7fu) != 0)
            normalized = __fadd_rn(normalized, v10_e4m3_to_float((rec1 >> 5) & 0xffu));
        const float scale = __int_as_float(static_cast<int>(s[group]) << 23);
        out[i] = __float2bfloat16(normalized * scale);
    }
}



// 64-byte smem rows (four 16-byte chunks): XOR the chunk with bits 1..2 of the row (ldmatrix 8-row
// phases conflict-free).
__device__ __forceinline__ unsigned v14_swz(unsigned row, unsigned chunk) {
    return row * 64u + ((chunk ^ ((row >> 1) & 3u)) << 4);
}

constexpr unsigned V14_BC = 256;          // transcoded B columns per CTA (gate/up: 128 gate + 128 up)
constexpr unsigned V14_DA = 3;            // A ring depth (k64 stages)
constexpr unsigned V14_DB = 3;            // raw B ring depth
constexpr unsigned V14_E_STRIDE = 136;    // bf16 per staging row in the gate/up epilogue (128 + 8 pad)
constexpr unsigned V14_THREADS = 512;

template <bool GATEUP> struct V14Smem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V14_BC / BMAT;
    static constexpr unsigned A = V14_DA * V10_TILE_M * 64;
    static constexpr unsigned B = V14_DB * BMAT * 32 * BCOLS;
    static constexpr unsigned S = V14_DB * BMAT * 4 * BCOLS;
    static constexpr unsigned AS = 2 * V10_TILE_M * 4;
    static constexpr unsigned BT = 2 * V14_BC * 64;
    static constexpr unsigned SF = 2 * 2 * V14_BC;
    static constexpr unsigned OFF_B = A, OFF_S = OFF_B + B, OFF_AS = OFF_S + S, OFF_BT = OFF_AS + AS,
                              OFF_SF = OFF_BT + BT, OFF_LUT = OFF_SF + SF, BYTES = OFF_LUT + 256 * 8;
    static_assert(!GATEUP || V10_TILE_M * V14_E_STRIDE * 2 <= OFF_AS, "epilogue staging aliases A/B/S only");
};

// Iteration i: wait (A i, raw B i+1 landed) -> barrier -> issue A i+2 + raw B i+3 -> transcode B i+1 -> MMA i.
// 16 warps = 4 (M, 32 rows; or 16 rows when valid_rows <= 32 so Warps 0..7 split rows 0..31 in parallel)
//          x 4 (N, 64 B columns: gate/up 32 gate + 32 up).
template <bool GATEUP, int EPI = 0>
__device__ __forceinline__ void v14_impl(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ pk0, const unsigned long long* __restrict__ sc0,
    const float* __restrict__ s20,
    const unsigned long long* __restrict__ pk1, const unsigned long long* __restrict__ sc1,
    const float* __restrict__ s21,
    __nv_bfloat16* __restrict__ C, unsigned char* __restrict__ oq, unsigned char* __restrict__ os,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned N_in, unsigned K_in,
    float* __restrict__ out32 = nullptr, const float* __restrict__ wsorted = nullptr,
    unsigned char* __restrict__ slot_q = nullptr, unsigned char* __restrict__ slot_s = nullptr,
    unsigned* __restrict__ slot_r2 = nullptr
) {
    using L = V14Smem<GATEUP>;
    constexpr unsigned N = GATEUP ? oi640::INTER : oi640::HIDDEN;
    constexpr unsigned K = GATEUP ? oi640::HIDDEN : oi640::INTER;
    constexpr unsigned TN = GATEUP ? 128u : 256u;
    constexpr unsigned BMAT = L::BMAT, BCOLS = L::BCOLS;
    constexpr unsigned n_tiles = N / TN;
    constexpr unsigned KS = K / 64u;
    constexpr unsigned kblocks = K / 32u;
    unsigned dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (gridDim.x == 0 || gridDim.x > oi640::FIXED_GRID || blockDim.x != V14_THREADS || dyn_smem < L::BYTES) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }
    const bool shape = (N_in == N && K_in == K);
    const bool outs = GATEUP ? (oq != nullptr && os != nullptr)
                             : (EPI == 0 ? C != nullptr
                                         : EPI == 3 ? (slot_q != nullptr && slot_s != nullptr)
                                         : EPI == 4 ? (slot_q != nullptr && slot_s != nullptr && slot_r2 != nullptr &&
                                                       (reinterpret_cast<unsigned long long>(slot_r2) & 3ull) == 0)
                                                    : (out32 != nullptr && wsorted != nullptr));
    const bool tbl_ok = pk0 != nullptr && sc0 != nullptr && s20 != nullptr &&
        (!GATEUP || (pk1 != nullptr && sc1 != nullptr && s21 != nullptr));
    if (!shape || !outs || !tbl_ok || aq == nullptr || as == nullptr || expert_offsets == nullptr ||
        sorted_token_ids == nullptr || !oi640::contract_ok(contract) ||
        !oi640::workspace_ok(workspace, contract) || status == nullptr || status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_ABI);
        return;
    }
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* const sA = smem;                                                     // [DA][128][64] swz
    unsigned char (*const sB)[BMAT][32][BCOLS] = reinterpret_cast<unsigned char (*)[BMAT][32][BCOLS]>(smem + L::OFF_B);
    unsigned char (*const sS)[BMAT][4][BCOLS] = reinterpret_cast<unsigned char (*)[BMAT][4][BCOLS]>(smem + L::OFF_S);
    unsigned (*const sAS)[V10_TILE_M] = reinterpret_cast<unsigned (*)[V10_TILE_M]>(smem + L::OFF_AS);
    unsigned char* const sBt = smem + L::OFF_BT;                                        // [2][256][64] swz
    unsigned char (*const sSF)[2][V14_BC] = reinterpret_cast<unsigned char (*)[2][V14_BC]>(smem + L::OFF_SF);
    uint2* const lut = reinterpret_cast<uint2*>(smem + L::OFF_LUT);
    __nv_bfloat16* const sE = reinterpret_cast<__nv_bfloat16*>(smem);

    const unsigned tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    const unsigned task_count = workspace->work_count * n_tiles;
    if (blockIdx.x >= task_count) return;
    if (tid < 256) {
        const float r = v10_e4m3_to_float(tid);
        const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        unsigned w[2] = {0u, 0u};
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            w[i >> 2] |= static_cast<unsigned>(__nv_cvt_float_to_fp8(mags[i] * r, __NV_SATFINITE, __NV_E4M3)) << (8 * (i & 3));
        lut[tid] = make_uint2(w[0], w[1]);
    }

    const unsigned warp_n = (warp_id & 3u) * (GATEUP ? 32u : 64u);
    const unsigned gid = lane >> 2, tig = lane & 3u;
    // Thread-invariant load_b and transcode coordinates hoisted outside the task/K loops.
    constexpr unsigned CH = BCOLS / 16u;
    const unsigned b_mtx = tid / (32u * CH), b_r = (tid / CH) % 32u, b_cc = tid % CH;
    const unsigned s_m2 = tid / (4u * CH), s_r2 = (tid / CH) % 4u, s_c2 = tid % CH;
    const unsigned t_col = tid & 255u, t_h = tid >> 8;
    const unsigned t_mtx = t_col / BCOLS, t_lc = t_col % BCOLS;
    const unsigned t_sf_idx = v17_sf_index<GATEUP>(t_col);

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const oi640::WorkItem item = workspace->items[task / n_tiles];
        const unsigned cta_n = (task % n_tiles) * TN;
        if (item.expert >= oi640::EXPERTS) { if (tid == 0) oi640::fail(status, oi640::ERR_PLAN); return; }
        const int m_start = expert_offsets[item.expert];
        const int M_expert = expert_offsets[item.expert + 1] - m_start;
        const int local = static_cast<int>(item.m_tile * V10_TILE_M);
        const unsigned char* B0 = reinterpret_cast<const unsigned char*>(pk0[item.expert]);
        const unsigned char* S0 = reinterpret_cast<const unsigned char*>(sc0[item.expert]);
        const unsigned char* B1 = GATEUP ? reinterpret_cast<const unsigned char*>(pk1[item.expert]) : B0;
        const unsigned char* S1 = GATEUP ? reinterpret_cast<const unsigned char*>(sc1[item.expert]) : S0;
        const float s2a = s20[item.expert];
        const float s2b = GATEUP ? s21[item.expert] : 0.0f;
        if (m_start < 0 || M_expert < 0 || local >= M_expert || B0 == nullptr || S0 == nullptr ||
            B1 == nullptr || S1 == nullptr || !isfinite(s2a) || !isfinite(s2b)) {
            if (tid == 0) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const unsigned cta_m = static_cast<unsigned>(m_start + local);
        const int valid_rows = min(M_expert - local, static_cast<int>(V10_TILE_M));
        const unsigned active_rows_pad16 = (static_cast<unsigned>(valid_rows) + 15u) & ~15u;
        // When valid_rows <= 32, Warps 0..3 cover rows 0..15 and Warps 4..7 cover rows 16..31 (rt_active <= 1),
        // doubling active MMA/epilogue warps while keeping Warps 8..15 (tid >= 256) at rt_active == 0 for transcode.
        const bool m32 = (valid_rows <= 32);
        const unsigned warp_m = (warp_id >> 2) * (m32 ? 16u : 32u);
        const int warp_valid = valid_rows - static_cast<int>(warp_m);
        const int rt_active = warp_valid <= 0 ? 0 : (m32 ? 1 : min(2, (warp_valid + 15) >> 4));

        // A: 128 rows x 64 bytes per stage = 512 chunks, one per thread.
        const unsigned a_row = tid >> 2, a_ch = tid & 3u;
        const bool a_ok = static_cast<int>(a_row) < valid_rows;
        const unsigned a_src_row = a_ok ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + a_row]) : cta_m + a_row) : 0u;
        const unsigned char* a_src = aq + static_cast<unsigned long long>(a_src_row) * K + a_ch * 16u;
        const unsigned a_dst = v14_swz(a_row, a_ch);
        const unsigned sc_row = tid - 384u;
        const bool sc_ok = tid >= 384u && static_cast<int>(sc_row) < valid_rows;
        const unsigned sc_src_row = sc_ok ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + sc_row]) : cta_m + sc_row) : 0u;
        const unsigned char* sc_src = as + static_cast<unsigned long long>(sc_src_row) * kblocks;

        const unsigned char* const b_src_base = (b_mtx ? B1 : B0) + b_r * N + cta_n + b_cc * 16u;
        const unsigned char* const s_src_base = (s_m2 ? S1 : S0) + s_r2 * N + cta_n + s_c2 * 16u;

        auto load_a = [&](unsigned ks) {
            if (a_row < active_rows_pad16)
                v10_cp16(sA + (ks % V14_DA) * (V10_TILE_M * 64) + a_dst, a_src + ks * 64u, a_ok);
            if (tid >= 384u && (ks & 1u) == 0u && sc_row < active_rows_pad16)       // four k32 scale bytes cover stages ks, ks+1
                v10_cp4(&sAS[(ks >> 1) & 1][sc_row], sc_src + ks * 2u, sc_ok);
        };
        auto load_b = [&](unsigned ks) {
            v17_cp16_plain(&sB[ks % V14_DB][b_mtx][b_r][b_cc * 16u], b_src_base + ks * (32u * N));
            if (tid < BMAT * 4u * CH) {
                v17_cp16_plain(&sS[ks % V14_DB][s_m2][s_r2][s_c2 * 16u], s_src_base + ks * (4u * N));
            }
        };
        // 512-thread cooperative transcode (1 half per thread, constant-offset sS & sB loads).
        auto transcode_512 = [&](unsigned sl, unsigned tb) {
            const unsigned char (*const sb_base)[BCOLS] = &sB[sl][t_mtx][8u * t_h];
            #pragma unroll
            for (unsigned t = 0; t < 2; ++t) {
                const unsigned s0 = sS[sl][t_mtx][2 * t][t_lc];
                const unsigned s1 = sS[sl][t_mtx][2 * t + 1][t_lc];
                const unsigned s_own = t_h ? s1 : s0;
                const unsigned s_max = max(s0 & 0x7fu, s1 & 0x7fu);
                const int e32 = (s_max >= 8u) ? (static_cast<int>(s_max >> 3) - 7)
                              : (s_max != 0u) ? (22 - __clz(static_cast<int>(s_max)))
                                              : 0;
                const float inv = __int_as_float((127 - e32) << 23);
                const unsigned r = __nv_cvt_float_to_fp8(v10_e4m3_to_float(s_own) * inv, __NV_SATFINITE, __NV_E4M3);
                const uint2 tt = lut[r];
                unsigned w[4];
                #pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const unsigned x = static_cast<unsigned>(sb_base[16 * t + 2 * g][t_lc]) |
                                       (static_cast<unsigned>(sb_base[16 * t + 2 * g + 1][t_lc]) << 8);
                    w[g] = v10_e2m1x4_to_e4m3x4(x, tt.x, tt.y);
                }
                *reinterpret_cast<uint4*>(sBt + tb * (V14_BC * 64) + v14_swz(t_col, 2 * t + t_h)) =
                    make_uint4(w[0], w[1], w[2], w[3]);
                if (t_h == 0) sSF[tb][t][t_sf_idx] = static_cast<unsigned char>((e32 + 127) & 0xff);
            }
        };
        // Adaptive in-loop transcode:
        // - valid_rows <= 64: Warps 8..15 (tid >= 256, rt_active == 0) load s0/s1 and compute e32/inv ONCE per t,
        //   then transcode both h=0 and h=1 while Warps 0..7 start MMAs immediately.
        // - valid_rows > 64: All 16 warps have active MMAs, so all 512 threads cooperate via transcode_512.
        auto transcode = [&](unsigned sl, unsigned tb) {
            if (valid_rows <= 64) {
                if (tid < V14_BC) return;
                #pragma unroll
                for (unsigned t = 0; t < 2; ++t) {
                    const unsigned s0 = sS[sl][t_mtx][2 * t][t_lc];
                    const unsigned s1 = sS[sl][t_mtx][2 * t + 1][t_lc];
                    const unsigned s_max = max(s0 & 0x7fu, s1 & 0x7fu);
                    const int e32 = (s_max >= 8u) ? (static_cast<int>(s_max >> 3) - 7)
                                  : (s_max != 0u) ? (22 - __clz(static_cast<int>(s_max)))
                                                  : 0;
                    const float inv = __int_as_float((127 - e32) << 23);
                    sSF[tb][t][t_sf_idx] = static_cast<unsigned char>((e32 + 127) & 0xff);
                    #pragma unroll
                    for (unsigned h = 0; h < 2; ++h) {
                        const unsigned s_own = h ? s1 : s0;
                        const unsigned r = __nv_cvt_float_to_fp8(v10_e4m3_to_float(s_own) * inv, __NV_SATFINITE, __NV_E4M3);
                        const uint2 tt = lut[r];
                        unsigned w[4];
                        #pragma unroll
                        for (int g = 0; g < 4; ++g) {
                            const unsigned x = static_cast<unsigned>(sB[sl][t_mtx][16 * t + 8 * h + 2 * g][t_lc]) |
                                               (static_cast<unsigned>(sB[sl][t_mtx][16 * t + 8 * h + 2 * g + 1][t_lc]) << 8);
                            w[g] = v10_e2m1x4_to_e4m3x4(x, tt.x, tt.y);
                        }
                        *reinterpret_cast<uint4*>(sBt + tb * (V14_BC * 64) + v14_swz(t_col, 2 * t + h)) =
                            make_uint4(w[0], w[1], w[2], w[3]);
                    }
                }
            } else {
                transcode_512(sl, tb);
            }
        };

        float acc[2][8][4];
        #pragma unroll
        for (int r = 0; r < 2; ++r)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.0f;

        // prologue: Gp0 = {B0}, Gp1 = {A0, B1}, Gp2 = {A1, B2}
        load_b(0);
        v10_commit();
        load_a(0);
        if (KS > 1) load_b(1);
        v10_commit();
        if (KS > 1) load_a(1);
        if (KS > 2) load_b(2);
        v10_commit();
        v10_wait<2>();
        __syncthreads();
        transcode_512(0, 0);

        unsigned sfa[2] = {0u, 0u};
        auto step = [&](auto par_tag, unsigned ks) {
            constexpr unsigned PAR = decltype(par_tag)::value;   // ks & 1
            v10_wait<1>();                                       // A ks, raw B ks+1 landed
            __syncthreads();
            if (ks + 2 < KS) load_a(ks + 2);
            if (ks + 3 < KS) load_b(ks + 3);
            v10_commit();
            if (ks + 1 < KS) transcode((ks + 1) % V14_DB, PAR ^ 1);
            if (rt_active == 0) return;
            if constexpr (PAR == 0) {
                const unsigned as_buf = (ks >> 1) & 1;
                sfa[0] = sAS[as_buf][warp_m + gid + (tig & 1) * 8];
                if (rt_active > 1) sfa[1] = sAS[as_buf][warp_m + 16 + gid + (tig & 1) * 8];
            }
            const unsigned a_slot = ks % V14_DA;
            const unsigned row_a0 = warp_m + (lane & 15u);
            const unsigned row_b0 = warp_n + ((lane >> 4) << 3) + (lane & 7u);
            #pragma unroll
            for (unsigned t = 0; t < 2; ++t) {
                unsigned afr0[4], afr1[4], bfr[8][2];
                const unsigned addr_a0 = static_cast<unsigned>(__cvta_generic_to_shared(
                    sA + a_slot * (V10_TILE_M * 64) + v14_swz(row_a0, 2 * t + (lane >> 4))));
                v10_ldsm4(addr_a0, afr0[0], afr0[1], afr0[2], afr0[3]);
                const uint2 sb8 = *reinterpret_cast<const uint2*>(
                    &sSF[PAR][t][((warp_id & 3u) << 6) | (gid << 3)]);
                const unsigned addr_b0 = static_cast<unsigned>(__cvta_generic_to_shared(
                    sBt + PAR * (V14_BC * 64) + v14_swz(row_b0, 2 * t + ((lane >> 3) & 1u))));
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    constexpr unsigned off_p0 = 0u;
                    (void)off_p0;
                    const unsigned off_p = GATEUP ? ((p >> 1) * (BCOLS * 64u) + (p & 1) * 1024u)
                                                  : (static_cast<unsigned>(p) * 1024u);
                    v10_ldsm4(addr_b0 + off_p, bfr[2 * p][0], bfr[2 * p][1], bfr[2 * p + 1][0], bfr[2 * p + 1][1]);
                }
                if (rt_active > 1) {
                    v10_ldsm4(addr_a0 + 1024u, afr1[0], afr1[1], afr1[2], afr1[3]);
                }
                if (t == 0) {
                    v17_qmma4_lo<2 * PAR>(acc[0], afr0, bfr, sfa[0], sb8.x);
                    v17_qmma4_hi<2 * PAR>(acc[0], afr0, bfr, sfa[0], sb8.y);
                    if (rt_active > 1) {
                        v17_qmma4_lo<2 * PAR>(acc[1], afr1, bfr, sfa[1], sb8.x);
                        v17_qmma4_hi<2 * PAR>(acc[1], afr1, bfr, sfa[1], sb8.y);
                    }
                } else {
                    v17_qmma4_lo<2 * PAR + 1>(acc[0], afr0, bfr, sfa[0], sb8.x);
                    v17_qmma4_hi<2 * PAR + 1>(acc[0], afr0, bfr, sfa[0], sb8.y);
                    if (rt_active > 1) {
                        v17_qmma4_lo<2 * PAR + 1>(acc[1], afr1, bfr, sfa[1], sb8.x);
                        v17_qmma4_hi<2 * PAR + 1>(acc[1], afr1, bfr, sfa[1], sb8.y);
                    }
                }
            }
        };
        #pragma unroll 1
        for (unsigned ks = 0; ks < KS; ks += 2) {
            step(std::integral_constant<unsigned, 0>{}, ks);
            step(std::integral_constant<unsigned, 1>{}, ks + 1);
        }
        v10_wait<0>();

        if (GATEUP) {
            (void)sE;
            const unsigned blk = warp_id & 3u;
            const unsigned col8 = warp_n + tig * 8u;
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= rt_active) break;
                const unsigned lr0 = warp_m + r * 16 + gid, lr1 = lr0 + 8;
                unsigned u0[4], u1[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float v[4];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const float g = acc[r][j][q] * s2a;
                        const float up = acc[r][j + 4][q] * s2b;
                        v[q] = g * (1.0f / (1.0f + __expf(-g))) * up;
                    }
                    u0[j] = v17_pack_bf162(v[0], v[1]);
                    u1[j] = v17_pack_bf162(v[2], v[3]);
                }
                unsigned qq0[2], ss0, qq1[2], ss1;
                v17_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], tig, qq0, ss0);
                if (static_cast<int>(lr0) < valid_rows) {
                    const unsigned long long orow = cta_m + lr0;
                    v17_st8_evict_last(oq + orow * N + cta_n + col8, make_uint2(qq0[0], qq0[1]));
                    if (tig == 0u) os[orow * (N / 32) + cta_n / 32 + blk] = static_cast<unsigned char>(ss0);
                }
                v17_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], tig, qq1, ss1);
                if (static_cast<int>(lr1) < valid_rows) {
                    const unsigned long long orow = cta_m + lr1;
                    v17_st8_evict_last(oq + orow * N + cta_n + col8, make_uint2(qq1[0], qq1[1]));
                    if (tig == 0u) os[orow * (N / 32) + cta_n / 32 + blk] = static_cast<unsigned char>(ss1);
                }
            }
        } else if (EPI == 3) {
            // Register-direct 4x4 quad transpose + MXFP8 quantizer (0 SMEM, 0 __syncthreads barriers).
            (void)sE;
            const bool keep_slot = (task + 320u >= task_count);
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= rt_active) break;
                const unsigned lr0 = warp_m + r * 16 + gid, lr1 = lr0 + 8;
                #pragma unroll
                for (unsigned b = 0; b < 2; ++b) {
                    unsigned u0[4], u1[4];
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const int j = 4 * static_cast<int>(b) + jj;
                        u0[jj] = v17_pack_bf162(acc[r][j][0] * s2a, acc[r][j][1] * s2a);
                        u1[jj] = v17_pack_bf162(acc[r][j][2] * s2a, acc[r][j][3] * s2a);
                    }
                    const unsigned blk = (warp_id & 3u) * 2u + b;
                    const unsigned col8 = warp_n + b * 32u + tig * 8u;
                    unsigned qq0[2], ss0, qq1[2], ss1;
                    v17_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], tig, qq0, ss0);
                    if (static_cast<int>(lr0) < valid_rows) {
                        const unsigned long long orow = cta_m + lr0;
                        if (keep_slot) v17_st8_evict_last(slot_q + orow * N + cta_n + col8, make_uint2(qq0[0], qq0[1]));
                        else v17_st8_evict_first(slot_q + orow * N + cta_n + col8, make_uint2(qq0[0], qq0[1]));
                        if (tig == 0u) slot_s[orow * (N / 32) + cta_n / 32 + blk] = static_cast<unsigned char>(ss0);
                    }
                    v17_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], tig, qq1, ss1);
                    if (static_cast<int>(lr1) < valid_rows) {
                        const unsigned long long orow = cta_m + lr1;
                        if (keep_slot) v17_st8_evict_last(slot_q + orow * N + cta_n + col8, make_uint2(qq1[0], qq1[1]));
                        else v17_st8_evict_first(slot_q + orow * N + cta_n + col8, make_uint2(qq1[0], qq1[1]));
                        if (tig == 0u) slot_s[orow * (N / 32) + cta_n / 32 + blk] = static_cast<unsigned char>(ss1);
                    }
                }
            }
        } else if (EPI == 4) {
            __syncthreads();
            unsigned char* const stage = sBt + warp_id * (8 * 144);
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= rt_active) break;
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        *reinterpret_cast<__nv_bfloat162*>(stage + gid * 144 + (j * 8 + tig * 2) * 2) =
                            __floats2bfloat162_rn(acc[r][j][2 * hh] * s2a, acc[r][j][2 * hh + 1] * s2a);
                    __syncwarp();
                    const unsigned block = lane >> 1, half = lane & 1;
                    const unsigned row = block >> 1, blk = block & 1;
                    const unsigned lr = warp_m + r * 16 + hh * 8 + row;
                    const bool valid = static_cast<int>(lr) < valid_rows;
                    float v[16];
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const uint4 w = valid
                            ? *reinterpret_cast<const uint4*>(
                                  stage + row * 144 + blk * 64 + half * 32 + c * 16)
                            : make_uint4(0, 0, 0, 0);
                        const unsigned u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            v[8 * c + 2 * i] = __uint_as_float(u[i] << 16);
                            v[8 * c + 2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
                        }
                    }
                    unsigned qq[4], ss, rr;
                    v18_mxfp8_half_r2(v, half, qq, ss, rr);
                    if (valid) {
                        const unsigned long long prow = cta_m + lr;
                        const unsigned col = cta_n + warp_n + blk * 32;
                        v17_st16_stream(slot_q + prow * N + col + half * 16,
                                        make_uint4(qq[0], qq[1], qq[2], qq[3]));
                        if (half == 0) {
                            const unsigned long long group = prow * (N / 32) + col / 32;
                            slot_s[group] = static_cast<unsigned char>(ss);
                            slot_r2[group] = rr;
                        }
                    }
                    __syncwarp();
                }
            }
        } else if (EPI != 0) {
            // v17: out32[token][col] += router_weight * float(bf16(scale2 * acc)) (fp32 vector reductions); no slot rows.
            // Each term equals today's bf16 slot value times its router weight (fn-mhc's combine contract); only the
            // fp32 order of the 10-term sum is free (atomics).
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= rt_active) break;
                const unsigned lr0 = warp_m + r * 16 + gid, lr1 = lr0 + 8;
                const bool r0v = static_cast<int>(lr0) < valid_rows, r1v = static_cast<int>(lr1) < valid_rows;
                const unsigned t0 = r0v ? static_cast<unsigned>(sorted_token_ids[cta_m + lr0]) : 0u;
                const unsigned t1 = r1v ? static_cast<unsigned>(sorted_token_ids[cta_m + lr1]) : 0u;
                const float w0 = r0v ? wsorted[cta_m + lr0] : 0.0f, w1 = r1v ? wsorted[cta_m + lr1] : 0.0f;
                float* const o0 = out32 + static_cast<unsigned long long>(t0) * N + cta_n + warp_n;
                float* const o1 = out32 + static_cast<unsigned long long>(t1) * N + cta_n + warp_n;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const __nv_bfloat162 b01 = __floats2bfloat162_rn(acc[r][j][0] * s2a, acc[r][j][1] * s2a);
                    const __nv_bfloat162 b23 = __floats2bfloat162_rn(acc[r][j][2] * s2a, acc[r][j][3] * s2a);
                    const float a0 = __low2float(b01) * w0, a1 = __high2float(b01) * w0;
                    const float a2 = __low2float(b23) * w1, a3 = __high2float(b23) * w1;
                    if (EPI == 1) {
                        if (r0v) asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(o0 + j * 8 + tig * 2), "f"(a0), "f"(a1) : "memory");
                        if (r1v) asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(o1 + j * 8 + tig * 2), "f"(a2), "f"(a3) : "memory");
                    } else {
                        // even lane: row lr0 cols 2tig..2tig+3; odd lane: row lr1 cols 2(tig-1)..+3
                        const bool even = (tig & 1) == 0;
                        const float x = __shfl_xor_sync(0xffffffffu, even ? a2 : a0, 1);
                        const float y = __shfl_xor_sync(0xffffffffu, even ? a3 : a1, 1);
                        if (even && r0v)
                            asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(o0 + j * 8 + tig * 2), "f"(a0), "f"(a1), "f"(x), "f"(y) : "memory");
                        if (!even && r1v)
                            asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(o1 + j * 8 + (tig - 1) * 2), "f"(x), "f"(y), "f"(a2), "f"(a3) : "memory");
                    }
                }
            }
        } else {
            // v16d: stage each warp's 8-row x 64-column bf16 slices in the (now idle) transcoded-B tile and
            // write 16-byte coalesced stores (v14: 32 scattered 4-byte stores per thread per tile).
            __syncthreads();                                      // every warp's last MMA read of the tile is done
            unsigned char* const stage = sBt + warp_id * (8 * 144);
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= rt_active) break;
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        *reinterpret_cast<__nv_bfloat162*>(stage + gid * 144 + (j * 8 + tig * 2) * 2) =
                            __floats2bfloat162_rn(acc[r][j][2 * hh] * s2a, acc[r][j][2 * hh + 1] * s2a);
                    __syncwarp();
                    #pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const unsigned chunk = lane + 32 * q, row = chunk >> 3, cc = chunk & 7;
                        const unsigned lr = warp_m + r * 16 + hh * 8 + row;
                        if (static_cast<int>(lr) < valid_rows)
                            v17_st16_stream(
                                C + static_cast<unsigned long long>(cta_m + lr) * N + cta_n + warp_n + cc * 8,
                                *reinterpret_cast<const uint4*>(stage + row * 144 + cc * 16));
                    }
                    __syncwarp();
                }
            }
        }
        if (task + gridDim.x < task_count) __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v17(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v14_impl<true>(aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
                   up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
                   status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v17(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v14_impl<false>(aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
                    expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

// Dynamic shared memory each kernel needs (host side: set MAX_DYNAMIC_SHARED_SIZE_BYTES and launch with it).
extern "C" __global__ void qwen4_moe_compact_v17_smem_bytes(unsigned* out) {
    out[0] = V14Smem<true>::BYTES;
    out[1] = V14Smem<false>::BYTES;
}




extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v17q(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K) {
    v14_impl<false, 3>(aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K, nullptr, nullptr,
                       slot_q, slot_s);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v17qr2(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    unsigned* __restrict__ slot_r2,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K) {
    v14_impl<false, 4>(aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K, nullptr, nullptr,
                       slot_q, slot_s, slot_r2);
}

// MXFP8 twin of moe_unpermute_reduce_indexed (common/moe_permute.cu): slot rows are e4m3 payload [pairs][hidden] + UE8M0
// [pairs][hidden/32]; value = e4m3 * 2^(s-127). Same arithmetic otherwise: acc += w * val in fp32, k ascending, bf16 out.
// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void moe_unpermute_reduce_mxfp8(
    const unsigned char* __restrict__ slot_q, const unsigned char* __restrict__ slot_s,
    __nv_bfloat16* __restrict__ output, const int* __restrict__ token_to_perm, const float* __restrict__ topk_weights,
    unsigned int hidden_size, unsigned int num_tokens, unsigned int topk
) {
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    if (hidden_size == 2560u && blockDim.x == 256u &&
        (reinterpret_cast<unsigned long long>(slot_q) & 3ull) == 0ull &&
        (reinterpret_cast<unsigned long long>(output) & 7ull) == 0ull) {
        const unsigned int tid = threadIdx.x;
        const unsigned int c0 = tid * 4u, g0 = tid >> 3;
        const unsigned int c1 = 1024u + tid * 4u, g1 = 32u + (tid >> 3);
        const unsigned int c2 = 2048u + tid * 2u, g2 = 64u + (tid >> 4);
        float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float acc2[2] = {0.0f, 0.0f};
        for (unsigned int k = 0; k < topk; ++k) {
            const float w = topk_weights[token * topk + k];
            if (w == 0.0f) continue;
            const unsigned long long row = static_cast<unsigned long long>(token_to_perm[token * topk + k]);
            const unsigned char* const q_row = slot_q + row * 2560ull;
            const unsigned char* const s_row = slot_s + row * 80ull;

            const unsigned q0 = *reinterpret_cast<const unsigned*>(q_row + c0);
            const float sc0 = __int_as_float(static_cast<int>(s_row[g0]) << 23);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float val = v10_e4m3_to_float((q0 >> (8 * j)) & 0xffu) * sc0;
                acc0[j] += w * val;
            }

            const unsigned q1 = *reinterpret_cast<const unsigned*>(q_row + c1);
            const float sc1 = __int_as_float(static_cast<int>(s_row[g1]) << 23);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float val = v10_e4m3_to_float((q1 >> (8 * j)) & 0xffu) * sc1;
                acc1[j] += w * val;
            }

            const unsigned q2 = static_cast<unsigned>(*reinterpret_cast<const unsigned short*>(q_row + c2));
            const float sc2 = __int_as_float(static_cast<int>(s_row[g2]) << 23);
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float val = v10_e4m3_to_float((q2 >> (8 * j)) & 0xffu) * sc2;
                acc2[j] += w * val;
            }
        }
        __nv_bfloat16* const out_row = output + static_cast<unsigned long long>(token) * 2560ull;
        *reinterpret_cast<uint2*>(out_row + c0) = make_uint2(
            v17_pack_bf162(acc0[0], acc0[1]), v17_pack_bf162(acc0[2], acc0[3]));
        *reinterpret_cast<uint2*>(out_row + c1) = make_uint2(
            v17_pack_bf162(acc1[0], acc1[1]), v17_pack_bf162(acc1[2], acc1[3]));
        *reinterpret_cast<unsigned*>(out_row + c2) = v17_pack_bf162(acc2[0], acc2[1]);
        return;
    }
    for (unsigned int c = threadIdx.x; c < hidden_size; c += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < topk; k++) {
            const float w = topk_weights[token * topk + k];
            if (w == 0.0f) continue;
            const unsigned long long row = static_cast<unsigned long long>(token_to_perm[token * topk + k]);
            const float scale = __int_as_float(static_cast<int>(slot_s[row * (hidden_size / 32) + c / 32]) << 23);
            const float val = v10_e4m3_to_float(slot_q[row * hidden_size + c]) * scale;
            acc += w * val;
        }
        output[token * hidden_size + c] = __float2bfloat16(acc);
    }
}

// Residual-two twin. Each group record stores two lane-plus-E4M3 residuals;
// reconstruction adds a matching residual in normalized space, then applies
// the same UE8M0 scale before router weighting.
extern "C" __global__ void moe_unpermute_reduce_mxfp8_r2(
    const unsigned char* __restrict__ slot_q, const unsigned char* __restrict__ slot_s,
    const unsigned* __restrict__ slot_r2, __nv_bfloat16* __restrict__ output,
    const int* __restrict__ token_to_perm, const float* __restrict__ topk_weights,
    unsigned int hidden_size, unsigned int num_tokens, unsigned int topk
) {
    if (hidden_size != 2560 || blockDim.x != 256 || gridDim.x != num_tokens ||
        slot_q == nullptr || slot_s == nullptr || slot_r2 == nullptr || output == nullptr ||
        token_to_perm == nullptr || topk_weights == nullptr ||
        (reinterpret_cast<unsigned long long>(slot_r2) & 3ull) != 0) __trap();
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    if ((reinterpret_cast<unsigned long long>(slot_q) & 3ull) == 0ull &&
        (reinterpret_cast<unsigned long long>(output) & 7ull) == 0ull) {
        const unsigned int tid = threadIdx.x;
        const unsigned int c0 = tid * 4u, g0 = tid >> 3, l0 = (tid & 7u) * 4u;
        const unsigned int c1 = 1024u + tid * 4u, g1 = 32u + (tid >> 3);
        const unsigned int c2 = 2048u + tid * 2u, g2 = 64u + (tid >> 4), l2 = (tid & 15u) * 2u;
        float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float acc2[2] = {0.0f, 0.0f};
        for (unsigned int k = 0; k < topk; ++k) {
            const float w = topk_weights[token * topk + k];
            if (w == 0.0f) continue;
            const unsigned long long row = static_cast<unsigned long long>(token_to_perm[token * topk + k]);
            const unsigned char* const q_row = slot_q + row * 2560ull;
            const unsigned char* const s_row = slot_s + row * 80ull;
            const unsigned* const r2_row = slot_r2 + row * 80ull;

            {
                const unsigned q0 = *reinterpret_cast<const unsigned*>(q_row + c0);
                const float sc0 = __int_as_float(static_cast<int>(s_row[g0]) << 23);
                const unsigned packed = r2_row[g0];
                const unsigned rec0 = packed & 0xffffu, rec1 = packed >> 16;
                const unsigned r0_lane = rec0 & 31u, r0_code = (rec0 >> 5) & 0xffu;
                const unsigned r1_lane = rec1 & 31u, r1_code = (rec1 >> 5) & 0xffu;
                const float r0_val = (r0_code & 0x7fu) != 0u ? v10_e4m3_to_float(r0_code) : 0.0f;
                const float r1_val = (r1_code & 0x7fu) != 0u ? v10_e4m3_to_float(r1_code) : 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned int lane = l0 + static_cast<unsigned int>(j);
                    float normalized = v10_e4m3_to_float((q0 >> (8 * j)) & 0xffu);
                    if (r0_lane == lane && (r0_code & 0x7fu) != 0u)
                        normalized = __fadd_rn(normalized, r0_val);
                    if (r1_lane == lane && (r1_code & 0x7fu) != 0u)
                        normalized = __fadd_rn(normalized, r1_val);
                    acc0[j] += w * (normalized * sc0);
                }
            }
            {
                const unsigned q1 = *reinterpret_cast<const unsigned*>(q_row + c1);
                const float sc1 = __int_as_float(static_cast<int>(s_row[g1]) << 23);
                const unsigned packed = r2_row[g1];
                const unsigned rec0 = packed & 0xffffu, rec1 = packed >> 16;
                const unsigned r0_lane = rec0 & 31u, r0_code = (rec0 >> 5) & 0xffu;
                const unsigned r1_lane = rec1 & 31u, r1_code = (rec1 >> 5) & 0xffu;
                const float r0_val = (r0_code & 0x7fu) != 0u ? v10_e4m3_to_float(r0_code) : 0.0f;
                const float r1_val = (r1_code & 0x7fu) != 0u ? v10_e4m3_to_float(r1_code) : 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned int lane = l0 + static_cast<unsigned int>(j);
                    float normalized = v10_e4m3_to_float((q1 >> (8 * j)) & 0xffu);
                    if (r0_lane == lane && (r0_code & 0x7fu) != 0u)
                        normalized = __fadd_rn(normalized, r0_val);
                    if (r1_lane == lane && (r1_code & 0x7fu) != 0u)
                        normalized = __fadd_rn(normalized, r1_val);
                    acc1[j] += w * (normalized * sc1);
                }
            }
            {
                const unsigned q2 = static_cast<unsigned>(*reinterpret_cast<const unsigned short*>(q_row + c2));
                const float sc2 = __int_as_float(static_cast<int>(s_row[g2]) << 23);
                const unsigned packed = r2_row[g2];
                const unsigned rec0 = packed & 0xffffu, rec1 = packed >> 16;
                const unsigned r0_lane = rec0 & 31u, r0_code = (rec0 >> 5) & 0xffu;
                const unsigned r1_lane = rec1 & 31u, r1_code = (rec1 >> 5) & 0xffu;
                const float r0_val = (r0_code & 0x7fu) != 0u ? v10_e4m3_to_float(r0_code) : 0.0f;
                const float r1_val = (r1_code & 0x7fu) != 0u ? v10_e4m3_to_float(r1_code) : 0.0f;
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const unsigned int lane = l2 + static_cast<unsigned int>(j);
                    float normalized = v10_e4m3_to_float((q2 >> (8 * j)) & 0xffu);
                    if (r0_lane == lane && (r0_code & 0x7fu) != 0u)
                        normalized = __fadd_rn(normalized, r0_val);
                    if (r1_lane == lane && (r1_code & 0x7fu) != 0u)
                        normalized = __fadd_rn(normalized, r1_val);
                    acc2[j] += w * (normalized * sc2);
                }
            }
        }
        __nv_bfloat16* const out_row = output + static_cast<unsigned long long>(token) * 2560ull;
        *reinterpret_cast<uint2*>(out_row + c0) = make_uint2(
            v17_pack_bf162(acc0[0], acc0[1]), v17_pack_bf162(acc0[2], acc0[3]));
        *reinterpret_cast<uint2*>(out_row + c1) = make_uint2(
            v17_pack_bf162(acc1[0], acc1[1]), v17_pack_bf162(acc1[2], acc1[3]));
        *reinterpret_cast<unsigned*>(out_row + c2) = v17_pack_bf162(acc2[0], acc2[1]);
        return;
    }
    const unsigned int groups_per_row = hidden_size / 32;
    for (unsigned int c = threadIdx.x; c < hidden_size; c += blockDim.x) {
        float acc = 0.0f;
        const unsigned int lane = c & 31u;
        for (unsigned int k = 0; k < topk; k++) {
            const float w = topk_weights[token * topk + k];
            if (w == 0.0f) continue;
            const unsigned long long row = static_cast<unsigned long long>(token_to_perm[token * topk + k]);
            const unsigned long long group = row * groups_per_row + c / 32;
            const float scale = __int_as_float(static_cast<int>(slot_s[group]) << 23);
            float normalized = v10_e4m3_to_float(slot_q[row * hidden_size + c]);
            const unsigned packed = slot_r2[group];
            const unsigned rec0 = packed & 0xffffu;
            const unsigned rec1 = packed >> 16;
            if ((rec0 & 31u) == lane && (((rec0 >> 5) & 0xffu) & 0x7fu) != 0)
                normalized = __fadd_rn(normalized, v10_e4m3_to_float((rec0 >> 5) & 0xffu));
            if ((rec1 & 31u) == lane && (((rec1 >> 5) & 0xffu) & 0x7fu) != 0)
                normalized = __fadd_rn(normalized, v10_e4m3_to_float((rec1 >> 5) & 0xffu));
            acc += w * (normalized * scale);
        }
        output[token * hidden_size + c] = __float2bfloat16(acc);
    }
}

// Graph-capture-safe 1-thread sticky error status accumulator for fallback paths where
// the fused unpermute-reduce+blend kernel does not run (e.g. BF16 slots or absent gate scalars).
// Maps non-PLANNED status codes in 1..30 to bit (1u << code) and all other non-PLANNED codes
// (e.g. PENDING=-1, BUILDING=-2, 0) to bit 31 via atomicOr into a pre-allocated sticky device word.
extern "C" __global__ void moe_status_accumulate_err(
    const int* __restrict__ routed_status,
    const int* __restrict__ shared_status,
    unsigned int* __restrict__ sticky_err
) {
    if (blockIdx.x == 0u && threadIdx.x == 0u && sticky_err != nullptr) {
        unsigned int mask = 0u;
        if (routed_status != nullptr) {
            const int st = routed_status[0];
            if (st != oi640::PLANNED) {
                mask |= (st > 0 && st < 31) ? (1u << static_cast<unsigned int>(st)) : (1u << 31);
            }
        }
        if (shared_status != nullptr) {
            const int st = shared_status[0];
            if (st != oi640::PLANNED) {
                mask |= (st > 0 && st < 31) ? (1u << static_cast<unsigned int>(st)) : (1u << 31);
            }
        }
        if (mask != 0u) atomicOr(sticky_err, mask);
    }
}

// Fused vectorized 128-bit MXFP8 slot unpermute-reduce + shared-expert blend reusing precomputed
// `ss_ptr[token] = sigmoid(dot(input[token], shared_gate))` from `qwen4_mxfp8_quant_rows_v16q4_gate`.
// Also folds optional `routed_status` and `shared_status` checks into `sticky_err` on block (0,0)
// thread 0 with zero extra kernel launches or per-layer D2D copies.
//
// Arithmetic matches `moe_unpermute_reduce_mxfp8` + `moe_batched_blend` bit-for-bit:
//   routed_bf16 = bf16(sum_{k=0..topk-1} w_k * (e4m3_to_f32(slot_q) * 2^(slot_s - 127)))
//   output      = bf16(f32(routed_bf16) + ss_ptr[token] * f32(shared_out))
//
// Fast-path launch geometry (hidden_size=2560, topk=10):
//   Grid: (num_tokens, 10, 1)  Block: (32, 1, 1)
//   Each 1-warp CTA owns 256 contiguous columns (8 cols/lane = 1x uint2 slot_q, 1x uint4 shared_out,
//   1x uint4 output), saturating all 48 GB10 SMs from rows >= 6 (60+ CTAs) with 0 SMEM and 0 barriers.
extern "C" __global__ void moe_unpermute_reduce_blend_mxfp8_bf16(
    const unsigned char* __restrict__ slot_q,
    const unsigned char* __restrict__ slot_s,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ ss_ptr,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    unsigned int hidden_size,
    unsigned int num_tokens,
    unsigned int topk,
    const int* __restrict__ routed_status,
    const int* __restrict__ shared_status,
    unsigned int* __restrict__ sticky_err
) {
    if (blockIdx.x == 0u && blockIdx.y == 0u && threadIdx.x == 0u && sticky_err != nullptr) {
        unsigned int mask = 0u;
        if (routed_status != nullptr) {
            const int st = routed_status[0];
            if (st != oi640::PLANNED) {
                mask |= (st > 0 && st < 31) ? (1u << static_cast<unsigned int>(st)) : (1u << 31);
            }
        }
        if (shared_status != nullptr) {
            const int st = shared_status[0];
            if (st != oi640::PLANNED) {
                mask |= (st > 0 && st < 31) ? (1u << static_cast<unsigned int>(st)) : (1u << 31);
            }
        }
        if (mask != 0u) atomicOr(sticky_err, mask);
    }

    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;

    const float gate = (ss_ptr != nullptr) ? ss_ptr[token] : 1.0f;

    if (hidden_size == 2560u && topk == 10u && blockDim.x == 32u && gridDim.y == 10u &&
        (reinterpret_cast<unsigned long long>(slot_q) & 7ull) == 0ull &&
        (reinterpret_cast<unsigned long long>(shared_out) & 15ull) == 0ull &&
        (reinterpret_cast<unsigned long long>(output) & 15ull) == 0ull) {
        const unsigned int lane = threadIdx.x;
        const unsigned int seg = blockIdx.y;
        const unsigned int col = seg * 256u + lane * 8u;
        const unsigned int g = (seg << 3) + (lane >> 2);

        int my_perm = 0;
        float my_w = 0.0f;
        if (lane < 10u) {
            my_perm = token_to_perm[token * 10u + lane];
            my_w = topk_weights[token * 10u + lane];
        }

        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (unsigned int k = 0u; k < 10u; ++k) {
            const float w = __shfl_sync(0xffffffffu, my_w, static_cast<int>(k));
            if (w == 0.0f) continue;
            const unsigned long long row =
                static_cast<unsigned long long>(__shfl_sync(0xffffffffu, my_perm, static_cast<int>(k)));
            const unsigned char* const q_row = slot_q + row * 2560ull;
            const unsigned char* const s_row = slot_s + row * 80ull;
            const uint2 q8 = *reinterpret_cast<const uint2*>(q_row + col);
            const float sc = __int_as_float(static_cast<int>(s_row[g]) << 23);

            const float2 f0 = v17_e4m3x2_to_float2(static_cast<unsigned short>(q8.x & 0xffffu));
            const float2 f1 = v17_e4m3x2_to_float2(static_cast<unsigned short>(q8.x >> 16));
            const float2 f2 = v17_e4m3x2_to_float2(static_cast<unsigned short>(q8.y & 0xffffu));
            const float2 f3 = v17_e4m3x2_to_float2(static_cast<unsigned short>(q8.y >> 16));

            acc[0] += w * (f0.x * sc);
            acc[1] += w * (f0.y * sc);
            acc[2] += w * (f1.x * sc);
            acc[3] += w * (f1.y * sc);
            acc[4] += w * (f2.x * sc);
            acc[5] += w * (f2.y * sc);
            acc[6] += w * (f3.x * sc);
            acc[7] += w * (f3.y * sc);
        }

        const __nv_bfloat16* const shared_row =
            shared_out + static_cast<unsigned long long>(token) * 2560ull;
        __nv_bfloat16* const out_row =
            output + static_cast<unsigned long long>(token) * 2560ull;
        const uint4 sh_raw = *reinterpret_cast<const uint4*>(shared_row + col);
        const unsigned sh_u[4] = {sh_raw.x, sh_raw.y, sh_raw.z, sh_raw.w};
        unsigned out_u[4];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float s0 = __uint_as_float(sh_u[p] << 16);
            const float s1 = __uint_as_float(sh_u[p] & 0xffff0000u);
            const float o0 = __bfloat162float(__float2bfloat16(acc[2 * p + 0]));
            const float o1 = __bfloat162float(__float2bfloat16(acc[2 * p + 1]));
            out_u[p] = v17_pack_bf162(o0 + gate * s0, o1 + gate * s1);
        }
        *reinterpret_cast<uint4*>(out_row + col) =
            make_uint4(out_u[0], out_u[1], out_u[2], out_u[3]);
        return;
    }

    const unsigned int stride = gridDim.y * blockDim.x;
    const unsigned int start_c = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned int groups_per_row = hidden_size / 32u;
    for (unsigned int c = start_c; c < hidden_size; c += stride) {
        float acc = 0.0f;
        for (unsigned int k = 0u; k < topk; ++k) {
            const float w = topk_weights[token * topk + k];
            if (w == 0.0f) continue;
            const unsigned long long row =
                static_cast<unsigned long long>(token_to_perm[token * topk + k]);
            const float scale =
                __int_as_float(static_cast<int>(slot_s[row * groups_per_row + c / 32u]) << 23);
            const float val = v10_e4m3_to_float(slot_q[row * hidden_size + c]) * scale;
            acc += w * val;
        }
        const float o = __bfloat162float(__float2bfloat16(acc));
        const float s = __bfloat162float(shared_out[static_cast<unsigned long long>(token) * hidden_size + c]);
        output[static_cast<unsigned long long>(token) * hidden_size + c] =
            __float2bfloat16(o + gate * s);
    }
}

// 4-warp coalesced SM120 twin of `moe_unpermute_reduce_blend_mxfp8_bf16` (`block = 128`).
// Loads `token_to_perm[0..9]`, `topk_weights[0..9]`, and `ss_ptr[token]` ONCE per 4-warp CTA
// into SMEM, eliminating redundant per-warp metadata reads while eliminating 1-warp CTA cap
// starvation on wide-SM GPUs (188 SMs).
//
// Supported high-throughput vectorized geometries for `hidden_size == 2560, topk == 10, block == 128`:
//   * `grid = (num_tokens, 20, 1)`: 128 cols/CTA (1 col/thread, 1 MX32 block/warp) -> 160-320 CTAs at rows=8..16
//   * `grid = (num_tokens, 10, 1)`: 256 cols/CTA (2 cols/thread, 2 MX32 blocks/warp) -> 200-390 CTAs at rows=20..39
//   * `grid = (num_tokens,  5, 1)`: 512 cols/CTA (4 cols/thread, 128B slot_q/warp) -> 200-1280 CTAs at rows=40..256
//
// Strictly 0-ULP bit-identical to `moe_unpermute_reduce_blend_mxfp8_bf16` (exact same ascending
// k = 0..topk-1 FP32 accumulation order, BF16 routed rounding, and shared-expert blend under --fmad=false).
extern "C" __global__ __launch_bounds__(128, 8) void moe_unpermute_reduce_blend_mxfp8_bf16_w4(
    const unsigned char* __restrict__ slot_q,
    const unsigned char* __restrict__ slot_s,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ ss_ptr,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    unsigned int hidden_size,
    unsigned int num_tokens,
    unsigned int topk,
    const int* __restrict__ routed_status,
    const int* __restrict__ shared_status,
    unsigned int* __restrict__ sticky_err
) {
    if (blockIdx.x == 0u && blockIdx.y == 0u && threadIdx.x == 0u && sticky_err != nullptr) {
        unsigned int mask = 0u;
        if (routed_status != nullptr) {
            const int st = routed_status[0];
            if (st != oi640::PLANNED) {
                mask |= (st > 0 && st < 31) ? (1u << static_cast<unsigned int>(st)) : (1u << 31);
            }
        }
        if (shared_status != nullptr) {
            const int st = shared_status[0];
            if (st != oi640::PLANNED) {
                mask |= (st > 0 && st < 31) ? (1u << static_cast<unsigned int>(st)) : (1u << 31);
            }
        }
        if (mask != 0u) atomicOr(sticky_err, mask);
    }

    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;

    const unsigned int tid = threadIdx.x;

    if (hidden_size == 2560u && topk == 10u && blockDim.x == 128u &&
        (gridDim.y == 5u || gridDim.y == 10u || gridDim.y == 20u)) {
        __shared__ int s_perm[10];
        __shared__ float s_w[10];
        __shared__ float s_gate;

        if (tid < 10u) {
            s_perm[tid] = token_to_perm[token * 10u + tid];
            s_w[tid] = topk_weights[token * 10u + tid];
        }
        if (tid == 0u) {
            s_gate = (ss_ptr != nullptr) ? ss_ptr[token] : 1.0f;
        }
        __syncthreads();

        const float gate = s_gate;
        const unsigned int seg = blockIdx.y;
        const __nv_bfloat16* const shared_row =
            shared_out + static_cast<unsigned long long>(token) * 2560ull;
        __nv_bfloat16* const out_row =
            output + static_cast<unsigned long long>(token) * 2560ull;

        if (gridDim.y == 5u &&
            (reinterpret_cast<unsigned long long>(slot_q) & 3ull) == 0ull &&
            (reinterpret_cast<unsigned long long>(shared_out) & 7ull) == 0ull &&
            (reinterpret_cast<unsigned long long>(output) & 7ull) == 0ull) {
            const unsigned int col = seg * 512u + tid * 4u;
            const unsigned int g = (seg << 4) + (tid >> 3);
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            #pragma unroll
            for (unsigned int k = 0u; k < 10u; ++k) {
                const float w = s_w[k];
                if (w == 0.0f) continue;
                const unsigned long long row = static_cast<unsigned long long>(s_perm[k]);
                const unsigned char* const q_row = slot_q + row * 2560ull;
                const unsigned char* const s_row = slot_s + row * 80ull;
                const unsigned q4 = *reinterpret_cast<const unsigned*>(q_row + col);
                const float sc = __int_as_float(static_cast<int>(s_row[g]) << 23);
                const float2 f0 = v17_e4m3x2_to_float2(static_cast<unsigned short>(q4 & 0xffffu));
                const float2 f1 = v17_e4m3x2_to_float2(static_cast<unsigned short>(q4 >> 16));
                acc0 += w * (f0.x * sc);
                acc1 += w * (f0.y * sc);
                acc2 += w * (f1.x * sc);
                acc3 += w * (f1.y * sc);
            }
            const uint2 sh_raw = *reinterpret_cast<const uint2*>(shared_row + col);
            const float s0 = __uint_as_float(sh_raw.x << 16);
            const float s1 = __uint_as_float(sh_raw.x & 0xffff0000u);
            const float s2 = __uint_as_float(sh_raw.y << 16);
            const float s3 = __uint_as_float(sh_raw.y & 0xffff0000u);
            const float o0 = __bfloat162float(__float2bfloat16(acc0));
            const float o1 = __bfloat162float(__float2bfloat16(acc1));
            const float o2 = __bfloat162float(__float2bfloat16(acc2));
            const float o3 = __bfloat162float(__float2bfloat16(acc3));
            *reinterpret_cast<uint2*>(out_row + col) = make_uint2(
                v17_pack_bf162(o0 + gate * s0, o1 + gate * s1),
                v17_pack_bf162(o2 + gate * s2, o3 + gate * s3));
            return;
        }

        if (gridDim.y == 10u &&
            (reinterpret_cast<unsigned long long>(slot_q) & 1ull) == 0ull &&
            (reinterpret_cast<unsigned long long>(shared_out) & 3ull) == 0ull &&
            (reinterpret_cast<unsigned long long>(output) & 3ull) == 0ull) {
            const unsigned int col = seg * 256u + tid * 2u;
            const unsigned int g = (seg << 3) + (tid >> 4);
            float acc0 = 0.0f, acc1 = 0.0f;
            #pragma unroll
            for (unsigned int k = 0u; k < 10u; ++k) {
                const float w = s_w[k];
                if (w == 0.0f) continue;
                const unsigned long long row = static_cast<unsigned long long>(s_perm[k]);
                const unsigned char* const q_row = slot_q + row * 2560ull;
                const unsigned char* const s_row = slot_s + row * 80ull;
                const unsigned short q2 = *reinterpret_cast<const unsigned short*>(q_row + col);
                const float sc = __int_as_float(static_cast<int>(s_row[g]) << 23);
                const float2 f0 = v17_e4m3x2_to_float2(q2);
                acc0 += w * (f0.x * sc);
                acc1 += w * (f0.y * sc);
            }
            const unsigned sh_u = *reinterpret_cast<const unsigned*>(shared_row + col);
            const float s0 = __uint_as_float(sh_u << 16);
            const float s1 = __uint_as_float(sh_u & 0xffff0000u);
            const float o0 = __bfloat162float(__float2bfloat16(acc0));
            const float o1 = __bfloat162float(__float2bfloat16(acc1));
            *reinterpret_cast<unsigned*>(out_row + col) =
                v17_pack_bf162(o0 + gate * s0, o1 + gate * s1);
            return;
        }

        if (gridDim.y == 20u) {
            const unsigned int col = seg * 128u + tid;
            const unsigned int g = (seg << 2) + (tid >> 5);
            float acc = 0.0f;
            #pragma unroll
            for (unsigned int k = 0u; k < 10u; ++k) {
                const float w = s_w[k];
                if (w == 0.0f) continue;
                const unsigned long long row = static_cast<unsigned long long>(s_perm[k]);
                const float sc = __int_as_float(static_cast<int>(slot_s[row * 80ull + g]) << 23);
                const float val = v10_e4m3_to_float(slot_q[row * 2560ull + col]) * sc;
                acc += w * val;
            }
            const float o = __bfloat162float(__float2bfloat16(acc));
            const float s = __bfloat162float(shared_row[col]);
            out_row[col] = __float2bfloat16(o + gate * s);
            return;
        }
    }

    const float gate = (ss_ptr != nullptr) ? ss_ptr[token] : 1.0f;
    const unsigned int stride = gridDim.y * blockDim.x;
    const unsigned int start_c = blockIdx.y * blockDim.x + tid;
    const unsigned int groups_per_row = hidden_size / 32u;
    for (unsigned int c = start_c; c < hidden_size; c += stride) {
        float acc = 0.0f;
        for (unsigned int k = 0u; k < topk; ++k) {
            const float w = topk_weights[token * topk + k];
            if (w == 0.0f) continue;
            const unsigned long long row =
                static_cast<unsigned long long>(token_to_perm[token * topk + k]);
            const float scale =
                __int_as_float(static_cast<int>(slot_s[row * groups_per_row + c / 32u]) << 23);
            const float val = v10_e4m3_to_float(slot_q[row * hidden_size + c]) * scale;
            acc += w * val;
        }
        const float o = __bfloat162float(__float2bfloat16(acc));
        const float s = __bfloat162float(shared_out[static_cast<unsigned long long>(token) * hidden_size + c]);
        output[static_cast<unsigned long long>(token) * hidden_size + c] =
            __float2bfloat16(o + gate * s);
    }
}
