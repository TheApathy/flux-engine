// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// BM=64 cluster-fused inject + inline-W4 upmix + MXFP8/gate kernels (95,744 B SMEM).

// ============================================================================
// Single-Launch BM=64 Cluster-Fused Inject + Inline-W4 Upmix + MXFP8/Gate
// (`qwen4_hc_inject_upmix_bm64_w4[_mxfp8_gate][_amax]`, 4-CTA cluster,
// 512 threads, `kSmemBytes = 95,744 B`):
//   - Phase 1 (BM=64, BK=32): reads BF16 `folded` via normal `cp.async`
//     (preserving L2 quota for `hyper'` + NVFP4 `up`), reads `hyper` normal and
//     `core` with `L2::evict_first`, writes `hyper'` with `L2::evict_last` on
//     SM120/SM103 (`normal` on SM121) so all 64 rows stay L2-hot for Phase 2,
//     discards dead `core` lines, and stages `down` (10 KiB/rank) + `rstd`
//     (256 B/rank) in DSMEM without writing DRAM.
//   - Phase 2 (BM=64, BH=32, BK=64): copies `down` + `rstd` across the 4-CTA
//     cluster into swizzled `sA_all` (40 KiB) + `sRstd2` (1 KiB), then loops
//     over 20 strided H-tiles/rank (`ht = (s + 4*g) + 8*ti`, pairing adjacent
//     CTAs onto the same 128B `hyper'` cache lines and matching the exact warp
//     reduction order of `moe_batched_blend` for zero-reread SMEM `ss_out`).
//   - Dequantizes `1.76 MiB` NVFP4 `up` (`L2::evict_last`) directly into 2-stage
//     swizzled `sB2` (1 scale group = 16 elements per thread), overlaps L2
//     loads with MMAs and MUFU sigmoids with `sH` `cp.async` (`L2::evict_first`),
//     and keeps trailing `kMixedPinWindow` (10,240 on SM120/SM103; 2,048 on
//     SM121) rows of `mixed` L2-normal for downstream FP8 act.
//   - Strict 0-ULP bit-identical to `inject_gemm_ws_l2h_gatepre` +
//     `qwen4_hc_prep_up` + `qwen4_hc_upmix_roof_t128_r4[_mxfp8_gate][_amax]`.
// ============================================================================
namespace hc_bm64_upmix {

constexpr int kRank2 = Qwen4FlashNextShape::kMhcRank;
constexpr int kBH2 = 32;
constexpr int kBK2 = 64;
constexpr int kKSteps2 = kRank2 / kBK2;  // 5
constexpr int kBN2 = kStreams * kBH2;    // 128

// Phase 1 DSMEM handoff regions (after sQ [0..82944), before sRstd [94720..94976))
constexpr int kOffDownOwn = hc_w4::kSmemBytesW4;
constexpr int kDownOwnBytes = (kBM / kStreams) * kRank2 * 2;  // 16 * 320 * 2 = 10,240 B
static_assert(kOffDownOwn >= kSmemQ && kOffDownOwn + kDownOwnBytes <= kOffRstd,
              "sDownOwn fits cleanly between sQ and sRstd");

// Phase 2 SMEM layout (active after DSMEM copy + cluster.sync())
constexpr int kOffA2 = 0;
constexpr int kSizeA2 = kKSteps2 * kBM * kBK2 * 2;       // 40,960 B
constexpr int kOffRstd2 = kOffA2 + kSizeA2;              // 40,960 B
constexpr int kSizeRstd2 = kBM * kStreams * 4;           // 1,024 B
constexpr int kOffB2 = kOffRstd2 + kSizeRstd2;           // 41,984 B
constexpr int kStageB2 = kBN2 * kBK2 * 2;                // 16,384 B
constexpr int kSizeB2 = 2 * kStageB2;                    // 32,768 B
constexpr int kOffH2 = kOffB2 + kSizeB2;                 // 74,752 B
constexpr int kSizeH2 = kBM * kStreams * kBH2 * 2;       // 16,384 B (XOR-swizzled)
constexpr int kOffO2 = kOffH2 + kSizeH2;                 // 91,136 B
constexpr int kSizeO2 = kBM * kBH2 * 2;                  // 4,096 B (XOR-swizzled)
constexpr int kOffWarpDot = kOffO2 + kSizeO2;            // 95,232 B
constexpr int kSizeWarpDot = 2 * kBM * 4;                // 512 B
constexpr int kSmemBytes = kOffWarpDot + kSizeWarpDot;   // 95,744 B
static_assert(kOffRstd2 + kSizeRstd2 <= kOffDownOwn,
              "Phase 2 DSMEM copy destination is strictly disjoint from Phase 1 sDownOwn/sRstd");
static_assert(kSmemBytes == 95744, "launcher passes 95744 bytes of dynamic smem for bm64_w4");
static_assert(kSmemBytes >= kOffRstd + kBM * 4, "covers Phase 1 sB + sA + sRstd");

__device__ __forceinline__ void check_bm64_smem() {
    unsigned int dyn;
    asm("mov.u32 %0, %%dynamic_smem_size;\n" : "=r"(dyn));
    if (dyn < (unsigned int)kSmemBytes) __trap();
}

__device__ __forceinline__ void cp_async16_u32(unsigned int dst, const void* gmem, bool pred) {
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem), "r"(bytes));
}

__device__ __forceinline__ void preload_b_u32_stage(
    unsigned char* smem, const __nv_bfloat16* __restrict__ folded, int s, int q, int bslot) {
    const unsigned int sbase = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    const int mt = threadIdx.x - kProducers;
    const int n0 = mt >> 2;
    const int part8 = (mt & 3) * 8;
    const __nv_bfloat16* src =
        folded + ((size_t)s * kNTotal + (size_t)n0) * kHidden + part8 + q * kBK;
    const bool ok5 = (n0 + kRank2) < kNTotal;
    unsigned int dst =
        sbase + (unsigned int)((n0 * kLds + part8) * 2) + (unsigned int)(bslot * (kNPad * kLds * 2));
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        cp_async16_u32(dst, src, true);
        dst += (unsigned int)(64 * kLds * 2);
        if (j < 4 || ok5) {
            asm("add.s64 %0, %0, %1;" : "+l"(src) : "l"((long long)(64 * kHidden * sizeof(__nv_bfloat16))));
        }
    }
    if (n0 + kRank2 < kNPad) {
        cp_async16_u32(dst, src, ok5);
    }
    cp_async_commit();
}

// Phase 1 MMA warps (8-15): 32-bit shared offsets keep register usage <= 124
// with 0 B stack / 0 spills even when followed by Phase 2 in the same kernel.
template <bool kPreloadedB = false>
__device__ __forceinline__ void multiply_rows_u32(
    unsigned char* smem, const __nv_bfloat16* __restrict__ folded, int s, float (&acc)[kFrags][4]) {
    const unsigned int sbase = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    const int mt = threadIdx.x - kProducers;
    const int mw = mt >> 5;
    const int lane = mt & 31;
    const int wm = mw >> 1;
    const int wn = mw & 1;
    const int n0 = mt >> 2;
    const int part8 = (mt & 3) * 8;
    const __nv_bfloat16* const w_base0 =
        folded + ((size_t)s * kNTotal + (size_t)n0) * kHidden + part8;
    const bool ok5 = (n0 + kRank2) < kNTotal;
    const bool pad5 = (n0 + kRank2) < kNPad;
    const unsigned int sb_base_t = sbase + (unsigned int)((n0 * kLds + part8) * 2);
    const unsigned int sa_ld_t =
        sbase + (unsigned int)kSmemB +
        (unsigned int)(((wm * 16 + (lane & 15)) * kLds + (lane >> 4) * 8) * 2);
    const unsigned int sb_ld_base =
        sbase + (unsigned int)(((wn * (kFrags * 8) + (lane & 7) + ((lane >> 4) << 3)) * kLds +
                                ((lane >> 3) & 1) * 8) *
                               2);

    auto load_b = [&](int q, int bslot) {
        const __nv_bfloat16* src = w_base0 + q * kBK;
        unsigned int dst = sb_base_t + (unsigned int)(bslot * (kNPad * kLds * 2));
#pragma unroll
        for (int j = 0; j < 5; ++j) {
            cp_async16_u32(dst, src, true);
            dst += (unsigned int)(64 * kLds * 2);
            if (j < 4 || ok5) {
                asm("add.s64 %0, %0, %1;" : "+l"(src) : "l"((long long)(64 * kHidden * sizeof(__nv_bfloat16))));
            }
        }
        if (pad5) {
            cp_async16_u32(dst, src, ok5);
        }
        cp_async_commit();
    };

#pragma unroll
    for (int f = 0; f < kFrags; ++f)
#pragma unroll
        for (int x = 0; x < 4; ++x) acc[f][x] = 0.0f;

    if constexpr (!kPreloadedB) {
        load_b(0, 0);
        load_b(1, 1);
    }

    auto step_mma = [&](int q, int buf, int bslot, int next_bslot) {
        cp_async_wait_prev();
        pipe_barrier();
        if (q + 2 < kSteps) {
            load_b(q + 2, next_bslot);
        } else {
            cp_async_commit();
        }
        const unsigned int sa_buf = sa_ld_t + (unsigned int)(buf * (kBM * kLds * 2));
        const unsigned int sb_buf = sb_ld_base + (unsigned int)(bslot * (kNPad * kLds * 2));
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 16) {
            unsigned int a[4];
            hc_w4::ldsm_x4_u32(a, sa_buf + (unsigned int)(kk * 2));
#pragma unroll
            for (int f = 0; f < kFrags; f += 2) {
                unsigned int r[4];
                hc_w4::ldsm_x4_u32(r, sb_buf + (unsigned int)((f * 8 * kLds + kk) * 2));
                mma_bf16(acc[f], a, r[0], r[1]);
                mma_bf16(acc[f + 1], a, r[2], r[3]);
            }
        }
    };

    static_assert(kSteps == 80 && kStagesB == 3, "period-6 unroll assumes kSteps=80, kStagesB=3");
    for (int q0 = 0; q0 < 78; q0 += 6) {
#pragma unroll
        for (int u = 0; u < 6; ++u) {
            step_mma(q0 + u, u & 1, u % kStagesB, (u + 2) % kStagesB);
        }
    }
    step_mma(78, 0, 0, 2);
    step_mma(79, 1, 1, 0);
}

__device__ __forceinline__ unsigned int swz128(int row, int c) {
    return static_cast<unsigned int>(row * 128 + ((c ^ (row & 7)) << 4));
}

__device__ __forceinline__ unsigned int swz_h2(int row, int st, int piece) {
    return static_cast<unsigned int>(row * 256 + (((st * 4 + piece) ^ (row & 7)) << 4));
}

__device__ __forceinline__ unsigned int swz_o2(int row, int piece) {
    return static_cast<unsigned int>(row * 64 + ((piece ^ ((row >> 1) & 3)) << 4));
}

__device__ __forceinline__ uint2 ld_u64_ca_hint(const void* p, uint64_t pol) {
    uint2 v;
    asm volatile("ld.global.ca.L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;\n"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(p), "l"(pol));
    return v;
}

__device__ __forceinline__ uint32_t ld_u8_ca_hint(const void* p, uint64_t pol) {
    uint32_t v;
    asm volatile("ld.global.ca.L2::cache_hint.u8 %0, [%1], %2;\n" : "=r"(v) : "l"(p), "l"(pol));
    return v;
}

__device__ __forceinline__ void cp16_hint_u32(unsigned int dst, const void* src, bool pred, uint64_t pol) {
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                 :: "r"(dst), "l"(src), "r"(bytes), "l"(pol));
}

__device__ __forceinline__ void store_row8_hint(void* g, uint2 v, uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1,%2}, %3;\n"
                 :: "l"(g), "r"(v.x), "r"(v.y), "l"(pol) : "memory");
}

// Bit-exact (0-ULP) NVFP4 E2M1x16 + single E4M3 scale -> two uint4 (BF16x8 each).
__device__ __forceinline__ void decode16_nvfp4_bf16(uint2 p8, uint32_t s_byte, uint4& out0, uint4& out1) {
#if Q4_HC_HAS_E2M1_CVT
    uint32_t s_f16x2;
    const unsigned short sb2 = static_cast<unsigned short>((s_byte & 0xffu) | ((s_byte & 0xffu) << 8));
    asm("{ .reg .b16 hs; mov.b16 hs, %1; cvt.rn.f16x2.e4m3x2 %0, hs; }" : "=r"(s_f16x2) : "h"(sb2));
    auto dec8 = [&](uint32_t p4) -> uint4 {
        uint32_t h[4];
        asm("{ .reg .b8 b0, b1, b2, b3;\n"
            "  mov.b32 {b0, b1, b2, b3}, %4;\n"
            "  cvt.rn.f16x2.e2m1x2 %0, b0;\n"
            "  cvt.rn.f16x2.e2m1x2 %1, b1;\n"
            "  cvt.rn.f16x2.e2m1x2 %2, b2;\n"
            "  cvt.rn.f16x2.e2m1x2 %3, b3; }\n"
            : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
            : "r"(p4));
        uint32_t o[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const __half2 prod = __hmul2(*reinterpret_cast<const __half2*>(&h[b]),
                                         *reinterpret_cast<const __half2*>(&s_f16x2));
            const float2 f = __half22float2(prod);
            const __nv_bfloat162 bf = __floats2bfloat162_rn(f.x, f.y);
            o[b] = *reinterpret_cast<const uint32_t*>(&bf);
        }
        return make_uint4(o[0], o[1], o[2], o[3]);
    };
    out0 = dec8(p8.x);
    out1 = dec8(p8.y);
#else
    __nv_fp8_e4m3 fp8;
    *reinterpret_cast<unsigned char*>(&fp8) = static_cast<unsigned char>(s_byte);
    const __nv_bfloat162 s2 = __bfloat162bfloat162(__float2bfloat16((float)fp8));
    auto dec8 = [&](uint32_t p4) -> uint4 {
        const uint32_t mag = p4 & 0x77777777u;
        const uint32_t sgn = (p4 >> 3) & 0x11111111u;
        constexpr uint32_t kHi0 = 0x3f3f3f00u, kHi1 = 0x40404040u;
        constexpr uint32_t kLo0 = 0xc0800000u, kLo1 = 0xc0804000u;
        constexpr uint32_t kSgn = 0x00008000u;
        uint32_t hi_a, lo_a, sg_a, hi_b, lo_b, sg_b;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_a) : "r"(kHi0), "r"(kHi1), "r"(mag));
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_a) : "r"(kLo0), "r"(kLo1), "r"(mag));
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(sg_a) : "r"(kSgn), "r"(0u), "r"(sgn));
        hi_a ^= sg_a;
        uint32_t w0, w1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w0) : "r"(lo_a), "r"(hi_a));
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w1) : "r"(lo_a), "r"(hi_a));
        const uint32_t mag_hi = mag >> 16;
        const uint32_t sgn_hi = sgn >> 16;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_b) : "r"(kHi0), "r"(kHi1), "r"(mag_hi));
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_b) : "r"(kLo0), "r"(kLo1), "r"(mag_hi));
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(sg_b) : "r"(kSgn), "r"(0u), "r"(sgn_hi));
        hi_b ^= sg_b;
        uint32_t w2, w3;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w2) : "r"(lo_b), "r"(hi_b));
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w3) : "r"(lo_b), "r"(hi_b));
        const __nv_bfloat162 r0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&w0), s2);
        const __nv_bfloat162 r1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&w1), s2);
        const __nv_bfloat162 r2 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&w2), s2);
        const __nv_bfloat162 r3 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&w3), s2);
        return make_uint4(*reinterpret_cast<const uint32_t*>(&r0),
                          *reinterpret_cast<const uint32_t*>(&r1),
                          *reinterpret_cast<const uint32_t*>(&r2),
                          *reinterpret_cast<const uint32_t*>(&r3));
    };
    out0 = dec8(p8.x);
    out1 = dec8(p8.y);
#endif
}

__device__ __forceinline__ unsigned pack_e4m3x4(float a, float b, float c, float d) {
    unsigned short p0, p1;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p0) : "f"(a), "f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p1) : "f"(c), "f"(d));
    unsigned r;
    asm("prmt.b32 %0, %1, %2, 0x5410;"
        : "=r"(r)
        : "r"(static_cast<unsigned>(p0)), "r"(static_cast<unsigned>(p1)));
    return r;
}

__device__ __forceinline__ void mxfp8_quant_quad(
    const float (&fv)[8], uint2& q_out, unsigned char& s_out) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(fv[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    unsigned qq0 = 0u, qq1 = 0u, ss = 0u;
    if (amax > 0.0f) {
        int e;
        const float m = frexpf(__fmul_rn(amax, 1.0f / 448.0f), &e);
        int k = (m == 0.5f) ? e - 1 : e;
        k = max(-127, min(127, k));
        const float inv = __int_as_float((127 - k) << 23);
        qq0 = pack_e4m3x4(fv[0] * inv, fv[1] * inv, fv[2] * inv, fv[3] * inv);
        qq1 = pack_e4m3x4(fv[4] * inv, fv[5] * inv, fv[6] * inv, fv[7] * inv);
        ss = static_cast<unsigned>(k + 127);
    }
    q_out = make_uint2(qq0, qq1);
    s_out = static_cast<unsigned char>(ss);
}

// Phase 1 producer warps (0-7): normal L2 read/write on in-place `hyper`/`hyper'`
// (preserving L2 residency for Phase 2); reads `core` with `L2::evict_first`
// for external core outputs, or normal L2 when `core` is the L2-hot dirty
// `moe_rows` buffer formed in Phase 0 (discarded after Phase 1 `cluster.sync()`).
template <bool kCoreNormalL2 = false, bool kPreloadedH = false>
__device__ __forceinline__ float produce_rows_inject_bm64(
    unsigned char* smem,
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    const __nv_bfloat16* __restrict__ residual,
    int s,
    int t0,
    unsigned int num_tokens,
    float eps,
    float pre_scale = 0.0f,
    const Vec8* pre_rh = nullptr) {
    auto sA = reinterpret_cast<__nv_bfloat16(*)[kBM][kLds]>(smem + kSmemB);
    const int pr = threadIdx.x >> 2;
    const int pv = threadIdx.x & 3;
    const int ptoken = t0 + pr;
    const bool prow = ptoken < (int)num_tokens;
    const size_t row_idx = prow ? (size_t)ptoken : 0;
    __nv_bfloat16* hrow = hyper + row_idx * kWidth + s * kHidden + pv * 8;
    const __nv_bfloat16* crow = core + row_idx * kHidden + pv * 8;
    const float scale = kPreloadedH
                            ? pre_scale
                            : (prow ? __bfloat162float(residual[row_idx * kWidth + (kWidth - kStreams) + s]) : 0.0f);
    const uint64_t pol_ef = kCoreNormalL2 ? 0ull : policy_evict_first();
    const uint64_t pol_el = kBm64HyperEvictLast ? policy_evict_last() : 0ull;
    float sums[kSlots];
#pragma unroll
    for (int j = 0; j < kSlots; ++j) sums[j] = 0.0f;
    Vec8 rh[4], rc[4];
    const uint4 zero = make_uint4(0, 0, 0, 0);
    auto load_stage = [&](int q, int stg) {
        rh[stg].raw = prow ? load_row16<true>(hrow + q * kBK) : zero;
        if constexpr (kCoreNormalL2) {
            rc[stg].raw = prow ? load_row16<true>(crow + q * kBK) : zero;
        } else {
            rc[stg].raw = prow ? load_row16_stream<true>(crow + q * kBK, pol_ef) : zero;
        }
    };
    auto produce_stage = [&](int q, int slot, int buf, int stg) {
        Vec8 out;
        const __nv_bfloat162* rc_p = reinterpret_cast<const __nv_bfloat162*>(&rc[stg]);
        const __nv_bfloat162* rh_p = reinterpret_cast<const __nv_bfloat162*>(&rh[stg]);
        __nv_bfloat162* out_p = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = __bfloat1622float2(rc_p[j]);
            const float2 h = __bfloat1622float2(rh_p[j]);
            out_p[j] = __floats2bfloat162_rn(h.x + a.x * scale, h.y + a.y * scale);
        }
        if (prow) {
            if constexpr (kBm64HyperEvictLast) {
                store_row16_stream(hrow + q * kBK, out.raw, pol_el);
            } else {
                *reinterpret_cast<uint4*>(hrow + q * kBK) = out.raw;
            }
        }
        *reinterpret_cast<uint4*>(&sA[buf][pr][pv * 8]) = out.raw;
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&out);
        float acc = sums[slot];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(p[j]);
            acc += f.x * f.x + f.y * f.y;
        }
        sums[slot] = acc;
    };
    if constexpr (kPreloadedH) {
#pragma unroll
        for (int stg = 0; stg < 4; ++stg) {
            rh[stg] = pre_rh[stg];
            if constexpr (kCoreNormalL2) {
                rc[stg].raw = prow ? load_row16<true>(crow + stg * kBK) : zero;
            } else {
                rc[stg].raw = prow ? load_row16_stream<true>(crow + stg * kBK, pol_ef) : zero;
            }
        }
    } else {
        load_stage(0, 0);
        load_stage(1, 1);
        load_stage(2, 2);
        load_stage(3, 3);
    }
    produce_stage(0, 0, 0, 0);
    load_stage(4, 0);
    for (int q0 = 0; q0 < kSteps; q0 += kSlots) {
#pragma unroll
        for (int u = 0; u < kSlots; ++u) {
            const int q = q0 + u;
            const int stg = (u + 1) & 3;
            pipe_barrier();
            if (q + 1 < kSteps) {
                produce_stage(q + 1, (u + 1) % kSlots, (u + 1) & 1, stg);
                if (q + 5 < kSteps) load_stage(q + 5, stg);
            }
        }
    }
    float half[2];
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const float* v = sums + 8 * g;
        const float x0 = v[0] + v[4], x1 = v[1] + v[5], x2 = v[2] + v[6], x3 = v[3] + v[7];
        const float y0 = x0 + x2, y1 = x1 + x3;
        float wsum = y0 + y1;
        wsum += __shfl_xor_sync(0xffffffffu, wsum, 2);
        wsum += __shfl_xor_sync(0xffffffffu, wsum, 1);
        half[g] = wsum;
    }
    return rsqrtf(hc_div_approx(half[0] + half[1], (float)kHidden) + eps);
}

__device__ __forceinline__ __nv_bfloat162 mix_pair_gated(
    const float (&gate)[kStreams][4], int half,
    const __nv_bfloat162 (&h2)[kStreams], const float (&r)[kStreams],
    const float (&w1)[kStreams][2]) {
    float pair[2][2];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        pair[p][0] = 0.0f;
        pair[p][1] = 0.0f;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int st = 2 * p + q;
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const float g = gate[st][half * 2 + e];
                const float x = __bfloat162float(e == 0 ? h2[st].x : h2[st].y) * r[st] * w1[st][e];
                pair[p][e] += g * __bfloat162float(__float2bfloat16(x));
            }
        }
    }
    __nv_bfloat162 out;
    out.x = __float2bfloat16((pair[0][0] + pair[1][0]) * 0.25f);
    out.y = __float2bfloat16((pair[0][1] + pair[1][1]) * 0.25f);
    return out;
}

template <bool kAmax, bool kMxfp8Gate, bool kCoreNormalL2 = false, bool kPreloadedB = false, bool kPreloadedH = false,
          typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void inject_upmix_bm64_w4_body(
    unsigned char* smem,
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    __nv_bfloat16* residual,
    const __nv_bfloat16* __restrict__ folded,
    const uint8_t* __restrict__ up_packed,
    const uint8_t* __restrict__ up_scales,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int num_tokens,
    float scale2_down,
    float scale2_inject,
    float alpha_up,
    float eps,
    unsigned int* __restrict__ amax_bits,
    unsigned char* __restrict__ xq,
    unsigned char* __restrict__ xs,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out,
    float pre_scale = 0.0f,
    const Vec8* pre_rh = nullptr) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    // ---- Phase 1: Inject + rstd + BF16 folded GEMM (normal L2) ----
    {
        const int tid = threadIdx.x;
        float* srstd = reinterpret_cast<float*>(smem + kOffRstd);
        float p1_acc[kFrags][4];
        if (tid < kProducers) {
            const float r = produce_rows_inject_bm64<kCoreNormalL2, kPreloadedH>(
                smem, hyper, core, residual, blockIdx.x, blockIdx.y * kBM, num_tokens, eps, pre_scale, pre_rh);
            if ((tid & 3) == 0) {
                srstd[tid >> 2] = r;
            }
        } else {
            multiply_rows_u32<kPreloadedB>(smem, folded, blockIdx.x, p1_acc);
        }
        __syncthreads();

        float* __restrict__ sq = reinterpret_cast<float*>(smem);
        if (tid >= kProducers) {
            const int mt = tid - kProducers;
            const int lane = mt & 31;
            const int wm = (mt >> 5) >> 1;
            const int wn = (mt >> 5) & 1;
            const int row0 = wm * 16 + (lane >> 2);
            const int row1 = row0 + 8;
            const float rs0 = srstd[row0];
            const float rs1 = srstd[row1];
#pragma unroll
            for (int f = 0; f < kFrags; ++f) {
                const int n = wn * (kFrags * 8) + f * 8 + 2 * (lane & 3);
                if (n >= kNTotal) continue;
                *reinterpret_cast<float2*>(&sq[row0 * kNTotal + n]) =
                    make_float2(rs0 * p1_acc[f][0], rs0 * p1_acc[f][1]);
                *reinterpret_cast<float2*>(&sq[row1 * kNTotal + n]) =
                    make_float2(rs1 * p1_acc[f][2], rs1 * p1_acc[f][3]);
            }
        }
        cluster.sync();  // all 4 ranks' sQ and sRstd complete

        const int s = blockIdx.x;
        const int t0 = blockIdx.y * kBM;

        const float2* const sq2 = reinterpret_cast<const float2*>(sq);
        const float2* const rq1 = reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 1) & 3));
        const float2* const rq2 = reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 2) & 3));
        const float2* const rq3 = reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 3) & 3));
        __nv_bfloat16* const sdown_own = reinterpret_cast<__nv_bfloat16*>(smem + kOffDownOwn);
        constexpr int kRowsPerRank = kBM / kStreams;                // 16
        constexpr int kPairsPerRow = kNTotal / 2;                   // 162
        constexpr int kPairsPerRank = kRowsPerRank * kPairsPerRow;  // 2592
        for (int idx = tid; idx < kPairsPerRank; idx += kThreads) {
            const int r_local = idx / kPairsPerRow;
            const int p = idx - r_local * kPairsPerRow;
            const int row = s * kRowsPerRank + r_local;
            const int t = t0 + row;
            const unsigned int n = (unsigned int)(2 * p);
            const int q_pair_idx = row * kPairsPerRow + p;
            const float2 vs = sq2[q_pair_idx];
            const float2 v1 = rq1[q_pair_idx];
            const float2 v2 = rq2[q_pair_idx];
            const float2 v3 = rq3[q_pair_idx];
            float2 q0, q1, q2, q3;
            if (s == 0) {
                q0 = vs; q1 = v1; q2 = v2; q3 = v3;
            } else if (s == 1) {
                q0 = v3; q1 = vs; q2 = v1; q3 = v2;
            } else if (s == 2) {
                q0 = v2; q1 = v3; q2 = vs; q3 = v1;
            } else {
                q0 = v1; q1 = v2; q2 = v3; q3 = vs;
            }
            const float sum0 = (((0.0f + q0.x) + q1.x) + q2.x) + q3.x;
            const float sum1 = (((0.0f + q0.y) + q1.y) + q2.y) + q3.y;
            if (n < (unsigned int)kRank2) {
                const float x0 = (scale2_down * sum0) * (1.0f / (float)kStreams);
                const float x1 = (scale2_down * sum1) * (1.0f / (float)kStreams);
                const __nv_bfloat162 dval =
                    __floats2bfloat162_rn(hc_fast_div_rn_1p_exp(x0), hc_fast_div_rn_1p_exp(x1));
                *reinterpret_cast<__nv_bfloat162*>(&sdown_own[r_local * kRank2 + n]) =
                    (t < (int)num_tokens) ? dval : __floats2bfloat162_rn(0.0f, 0.0f);
            } else if (t < (int)num_tokens) {
                const float x0 = (scale2_inject * sum0) * (1.0f / (float)kStreams);
                const float x1 = (scale2_inject * sum1) * (1.0f / (float)kStreams);
                const float v0 = 2.0f * hc_fast_rcp_rn_1p_exp(x0);
                const float v1 = 2.0f * hc_fast_rcp_rn_1p_exp(x1);
                *reinterpret_cast<__nv_bfloat162*>(
                    &residual[(size_t)t * kWidth + (kWidth - kStreams) + (n - kRank2)]) =
                    __floats2bfloat162_rn(v0, v1);
            }
        }

        // Early L2 discard of dead `core` lines (all 4 ranks already finished Phase 1 reads).
        if (core != mixed && ((reinterpret_cast<uintptr_t>(core) & 127u) == 0u)) {
            const int row0_disc = t0 + s * kRowsPerRank;
            const int valid_rows = min(kRowsPerRank, max(0, (int)num_tokens - row0_disc));
            const unsigned char* base_disc =
                reinterpret_cast<const unsigned char*>(core + (size_t)row0_disc * kHidden);
            const int total_lines = valid_rows * (kHidden * (int)sizeof(__nv_bfloat16) / 128);
            for (int line = tid; line < total_lines; line += kThreads) {
                const unsigned char* p = base_disc + (size_t)line * 128;
                asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
            }
        }

        // Make Phase 1 generic global stores to `hyper'` visible to Phase 2 async cp.async across ranks.
        asm volatile("fence.proxy.async.global;\n\tfence.acq_rel.gpu;\n" ::: "memory");
        cluster.sync();  // sDownOwn, sRstd, and hyper' visible across all 4 ranks
    }

    // ---- Phase 2 Setup: DSMEM -> swizzled local sA_all [0..40960) + sRstd2 [40960..41984) ----
    const int s = blockIdx.x;
    const int t0 = blockIdx.y * kBM;
    int tid;
    asm volatile("mov.u32 %0, %%tid.x;\n" : "=r"(tid));
    unsigned int sbase = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n"
                 : "+r"(sbase));
    constexpr int kRowsPerRank = kBM / kStreams;  // 16
    {
        __nv_bfloat16* const sdown_own = reinterpret_cast<__nv_bfloat16*>(smem + kOffDownOwn);
        float* const srstd = reinterpret_cast<float*>(smem + kOffRstd);
        const uint4* const rd0 = reinterpret_cast<const uint4*>(cluster.map_shared_rank(sdown_own, 0));
        const uint4* const rd1 = reinterpret_cast<const uint4*>(cluster.map_shared_rank(sdown_own, 1));
        const uint4* const rd2 = reinterpret_cast<const uint4*>(cluster.map_shared_rank(sdown_own, 2));
        const uint4* const rd3 = reinterpret_cast<const uint4*>(cluster.map_shared_rank(sdown_own, 3));
        const int row = tid >> 3;       // 0..63
        const int c = tid & 7;          // 0..7
        const int src_rank = row >> 4;  // 0..3
        const int r_local = row & 15;   // 0..15
        const uint4* const rsrc =
            (src_rank == 0) ? rd0 : ((src_rank == 1) ? rd1 : ((src_rank == 2) ? rd2 : rd3));
        const unsigned int dst_base = sbase + kOffA2 + swz128(row, c);
#pragma unroll
        for (int kt = 0; kt < kKSteps2; ++kt) {
            const uint4 v = rsrc[r_local * 40 + kt * 8 + c];
            const unsigned int dst = dst_base + (unsigned int)(kt * (kBM * kBK2 * 2));
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                         :: "r"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
        }
        if (tid < kBM * kStreams) {
            const float* const rs0 = reinterpret_cast<const float*>(cluster.map_shared_rank(srstd, 0));
            const float* const rs1 = reinterpret_cast<const float*>(cluster.map_shared_rank(srstd, 1));
            const float* const rs2 = reinterpret_cast<const float*>(cluster.map_shared_rank(srstd, 2));
            const float* const rs3 = reinterpret_cast<const float*>(cluster.map_shared_rank(srstd, 3));
            const int rrow = tid >> 2;
            const int rst = tid & 3;
            const float* const rsrc_f =
                (rst == 0) ? rs0 : ((rst == 1) ? rs1 : ((rst == 2) ? rs2 : rs3));
            reinterpret_cast<float*>(smem + kOffRstd2)[rrow * kStreams + rst] = rsrc_f[rrow];
        }
    }
    cluster.sync();  // all ranks finished reading DSMEM sDownOwn/sRstd; [41984..95744) now free

    // ---- Phase 2: 20 strided H-tiles/rank (BH=32, BK=64, inline NVFP4 `up`) ----
    auto load_up_b2 = [&](int ht_idx, int kt_idx, uint2& p8_out, uint32_t& sc_out, uint64_t pol) {
        const int b_row = tid >> 2;
        const int b_st = b_row >> 5;
        const int b_hloc = b_row & 31;
        const int grp16 = tid & 3;
        const size_t row_id = (size_t)b_st * kHidden + (size_t)(ht_idx * kBH2) + (size_t)b_hloc;
        p8_out = ld_u64_ca_hint(
            up_packed + row_id * (kRank2 / 2) + (size_t)(kt_idx * (kBK2 / 2) + grp16 * 8), pol);
        sc_out = ld_u8_ca_hint(
            up_scales + row_id * (kRank2 / 16) + (size_t)(kt_idx * (kBK2 / 16) + grp16), pol);
    };

    auto store_b2_decoded = [&](int slot, uint2 p8, uint32_t sc) {
        const int b_row = tid >> 2;
        const int grp16 = tid & 3;
        const unsigned int sb_c0 = swz128(b_row, grp16 * 2);
        uint4 d0, d1;
        decode16_nvfp4_bf16(p8, sc, d0, d1);
        const unsigned int slot_base = sbase + kOffB2 + (unsigned int)(slot * kStageB2);
        const unsigned int dst0 = slot_base + sb_c0;
        const unsigned int dst1 = slot_base + (sb_c0 ^ 16u);
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     :: "r"(dst0), "r"(d0.x), "r"(d0.y), "r"(d0.z), "r"(d0.w) : "memory");
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     :: "r"(dst1), "r"(d1.x), "r"(d1.y), "r"(d1.z), "r"(d1.w) : "memory");
    };

    // Prime sB2[0] with the first H-tile (`ht = s`, `kt = 0`).
    {
        const uint64_t pol_el = policy_evict_last();
        uint2 p8;
        uint32_t sc;
        load_up_b2(s, 0, p8, sc, pol_el);
        store_b2_decoded(0, p8, sc);
        __syncthreads();
    }

    int cur_slot = 0;
    float amax = 0.0f;

#pragma unroll 1
    for (int g = 0; g < 2; ++g) {
        const int warp_id = s + 4 * g;  // matches moe_batched_blend warp 0..7
        float dot_acc[8];
        if constexpr (kMxfp8Gate) {
            if (tid >= 256) {
#pragma unroll
                for (int rr = 0; rr < 8; ++rr) dot_acc[rr] = 0.0f;
            }
        }

#pragma unroll 1
        for (int ti = 0; ti < 10; ++ti) {
            const int warp = tid >> 5;   // 0..15
            const int lane = tid & 31;   // 0..31
            const int wm = warp >> 2;    // 0..3 (16-row slab)
            const int wh = warp & 3;     // 0..3 (8 h-cols)
            const int ht = warp_id + 8 * ti;

            float acc[kStreams][4];
#pragma unroll
            for (int st = 0; st < kStreams; ++st)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[st][e] = 0.0f;

            uint2 p8_next = make_uint2(0u, 0u);
            uint32_t sc_next = 0u;
            bool has_next_ht = false;

#pragma unroll 1
            for (int kt = 0; kt < kKSteps2; ++kt) {
                bool has_next = false;
                int next_ht = ht;
                int next_kt = kt + 1;
                if (kt + 1 < kKSteps2) {
                    has_next = true;
                } else if (ti + 1 < 10) {
                    has_next = true;
                    next_ht = warp_id + 8 * (ti + 1);
                    next_kt = 0;
                } else if (g == 0) {
                    has_next = true;
                    next_ht = s + 4;
                    next_kt = 0;
                }
                if (has_next) {
                    load_up_b2(next_ht, next_kt, p8_next, sc_next, policy_evict_last());
                }
                if (kt == kKSteps2 - 1) has_next_ht = has_next;

                const unsigned int sa_kt = sbase + kOffA2 + (unsigned int)(kt * (kBM * kBK2 * 2));
                const unsigned int sb_kt = sbase + kOffB2 + (unsigned int)(cur_slot * kStageB2);
#pragma unroll
                for (int kk = 0; kk < kBK2; kk += 16) {
                    unsigned int a[4];
                    const int a_row = wm * 16 + (lane & 15);
                    const int a_c = (kk >> 3) + (lane >> 4);
                    hc_w4::ldsm_x4_u32(a, sa_kt + swz128(a_row, a_c));
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        const int st = 2 * p + (lane >> 4);
                        const int br = st * kBH2 + wh * 8 + (lane & 7);
                        const int bc = (kk >> 3) + ((lane >> 3) & 1);
                        unsigned int r[4];
                        hc_w4::ldsm_x4_u32(r, sb_kt + swz128(br, bc));
                        mma_bf16(acc[2 * p], a, r[0], r[1]);
                        mma_bf16(acc[2 * p + 1], a, r[2], r[3]);
                    }
                }
                if (kt + 1 < kKSteps2) {
                    store_b2_decoded(cur_slot ^ 1, p8_next, sc_next);
                    __syncthreads();
                    cur_slot ^= 1;
                }
            }

            // Issue sH cp.async (L2::evict_first) overlapped with norm_w loads, MUFU sigmoids,
            // and next H-tile's NVFP4 `up` decode into `sB2[cur_slot ^ 1]`.
            const int h0 = ht * kBH2;
            const bool do_gate_dot = kMxfp8Gate && (ss_out != nullptr) && (shared_gate != nullptr);
            const bool full_tile = ((unsigned int)(t0 + kBM) <= num_tokens);
            const uint64_t pol_ef = policy_evict_first();
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int idx = tid + j * kThreads;
                const int piece = idx & 3;
                const int st = (idx >> 2) & 3;
                const int row = idx >> 4;
                const int t = t0 + row;
                const bool ok = full_tile || (t < (int)num_tokens);
                const unsigned int dst = sbase + kOffH2 + swz_h2(row, st, piece);
                const __nv_bfloat16* src =
                    hyper + (size_t)(ok ? t : 0) * kWidth + st * kHidden + h0 + piece * 8;
                cp16_hint_u32(dst, src, ok, pol_ef);
            }
            cp_async_commit();

            float w1[kStreams][2];
            const int hl = wh * 8 + 2 * (lane & 3);
#pragma unroll
            for (int st = 0; st < kStreams; ++st) {
                const __nv_bfloat162 w2 =
                    *reinterpret_cast<const __nv_bfloat162*>(norm_w + (size_t)st * kHidden + h0 + hl);
                w1[st][0] = 1.0f + __bfloat162float(w2.x);
                w1[st][1] = 1.0f + __bfloat162float(w2.y);
            }
#pragma unroll
            for (int st = 0; st < kStreams; ++st) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[st][e] = hc_fast_rcp_rn_1p_exp(acc[st][e] * alpha_up);
                }
            }
            if (has_next_ht) {
                store_b2_decoded(cur_slot ^ 1, p8_next, sc_next);
            }
            cur_slot ^= 1;

            asm volatile("cp.async.wait_group 0;\n\tbar.sync 0;\n"
                         : "+f"(acc[kStreams - 1][3]) :: "memory");

            // Compute mixed BF16 pairs into XOR-swizzled sO
            const unsigned int in_piece = (unsigned int)((lane & 3) * 4);
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = wm * 16 + (lane >> 2) + half * 8;
                const bool ok_row = full_tile || (t0 + row < (int)num_tokens);
                if (ok_row) {
                    const float4 r4 = *reinterpret_cast<const float4*>(smem + kOffRstd2 + row * 16);
                    const float r[kStreams] = {r4.x, r4.y, r4.z, r4.w};
                    __nv_bfloat162 h2[kStreams];
#pragma unroll
                    for (int st = 0; st < kStreams; ++st) {
                        const unsigned int h_addr = sbase + kOffH2 + swz_h2(row, st, wh) + in_piece;
                        unsigned int raw_h;
                        asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(raw_h) : "r"(h_addr));
                        h2[st] = *reinterpret_cast<const __nv_bfloat162*>(&raw_h);
                    }
                    const __nv_bfloat162 out = mix_pair_gated(acc, half, h2, r, w1);
                    const unsigned int o_addr = sbase + kOffO2 + swz_o2(row, wh) + in_piece;
                    const unsigned int raw_o = *reinterpret_cast<const unsigned int*>(&out);
                    asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(o_addr), "r"(raw_o) : "memory");
                    if constexpr (kAmax) {
                        const float2 f = __bfloat1622float2(out);
                        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
                    }
                } else {
                    const unsigned int o_addr = sbase + kOffO2 + swz_o2(row, wh) + in_piece;
                    asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(o_addr) : "memory");
                }
            }
            __syncthreads();

            // Coalesced `mixed` stores:
            // - `kMxfp8Gate`: warps 0..7 quantize & store 16B `mixed` + 8B `xq` + 1B `xs`
            //   while warps 8..15 accumulate zero-reread `ss_out` dot products.
            // - `!kMxfp8Gate`: all 512 threads participate in conflict-free 8B (`uint2`)
            //   stores across all 16 warps (0-ULP bit-identical, 2x LSU concurrency).
            if constexpr (kMxfp8Gate) {
                if (tid < 256) {
                    const int row = tid >> 2;
                    const int piece = tid & 3;
                    const int t = t0 + row;
                    const bool ok = full_tile || (t < (int)num_tokens);
                    const unsigned int o_addr = sbase + kOffO2 + swz_o2(row, piece);
                    uint4 v;
                    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                                 : "r"(o_addr));
                    const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&v);
                    float fv[8];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f2 = __bfloat1622float2(p2[j]);
                        fv[2 * j] = f2.x;
                        fv[2 * j + 1] = f2.y;
                    }
                    uint2 q_out;
                    unsigned char s_out;
                    mxfp8_quant_quad(fv, q_out, s_out);
                    if (ok) {
                        __nv_bfloat16* dst = mixed + (size_t)t * kHidden + h0 + piece * 8;
                        if (t + kMixedPinWindow >= (int)num_tokens) {
                            *reinterpret_cast<uint4*>(dst) = v;
                        } else {
                            store_row16_stream(dst, v, pol_ef);
                        }
                        uint2* qptr = reinterpret_cast<uint2*>(xq + (size_t)t * kHidden + h0 + piece * 8);
                        store_row8_hint(qptr, q_out, pol_ef);
                        if (piece == 0) {
                            xs[(size_t)t * (kHidden / kBH2) + ht] = s_out;
                        }
                    }
                } else if (do_gate_dot) {
                    const int warp_dot = (tid - 256) >> 5;  // 0..7
                    const float g_val = __bfloat162float(shared_gate[h0 + lane]);
                    const int piece = lane >> 3;
                    const unsigned int byte_in_piece = (unsigned int)((lane & 7) << 1);
#pragma unroll
                    for (int rr = 0; rr < 8; ++rr) {
                        const int row = warp_dot * 8 + rr;
                        const unsigned int o_addr = sbase + kOffO2 + swz_o2(row, piece) + byte_in_piece;
                        unsigned short raw_bf;
                        asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(raw_bf) : "r"(o_addr));
                        const __nv_bfloat16 bv = *reinterpret_cast<const __nv_bfloat16*>(&raw_bf);
                        dot_acc[rr] += __bfloat162float(bv) * g_val;
                    }
                }
            } else {
                const int row = tid >> 3;
                const int hp = tid & 7;
                const int piece = hp >> 1;
                const int sub = hp & 1;
                const int t = t0 + row;
                if (full_tile || (t < (int)num_tokens)) {
                    const unsigned int o_addr = sbase + kOffO2 + swz_o2(row, piece) + (unsigned int)(sub * 8);
                    uint2 v;
                    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n"
                                 : "=r"(v.x), "=r"(v.y)
                                 : "r"(o_addr));
                    __nv_bfloat16* dst = mixed + (size_t)t * kHidden + h0 + hp * 4;
                    if (t + kMixedPinWindow >= (int)num_tokens) {
                        *reinterpret_cast<uint2*>(dst) = v;
                    } else {
                        store_row8_hint(dst, v, pol_ef);
                    }
                }
            }
        }

        if constexpr (kMxfp8Gate) {
            if (ss_out != nullptr && shared_gate != nullptr && tid >= 256) {
                float* const s_warp_dot = reinterpret_cast<float*>(smem + kOffWarpDot);
                const int warp_dot = (tid - 256) >> 5;
                const int lane = tid & 31;
#pragma unroll
                for (int rr = 0; rr < 8; ++rr) {
                    float d = dot_acc[rr];
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        d += __shfl_down_sync(0xffffffffu, d, off);
                    }
                    if (lane == 0) {
                        s_warp_dot[g * kBM + warp_dot * 8 + rr] = d;
                    }
                }
            }
        }
    }

    if constexpr (kAmax) {
        __syncthreads();  // all warps finished reading sO before reusing sO as sMax
        const int warp = tid >> 5;
        const int lane = tid & 31;
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
        }
        float* const s_max = reinterpret_cast<float*>(smem + kOffO2);
        if (lane == 0) s_max[warp] = amax;
        __syncthreads();
        if (tid == 0) {
            float blk = s_max[0];
#pragma unroll
            for (int w = 1; w < kThreads / 32; ++w) blk = fmaxf(blk, s_max[w]);
            const unsigned int bits = __float_as_uint(blk);
            if (blk > 0.0f && bits > *reinterpret_cast<volatile unsigned int*>(amax_bits)) {
                atomicMax(amax_bits, bits);
            }
        }
    }

    if constexpr (kMxfp8Gate) {
        if (ss_out != nullptr) {
            if (shared_gate == nullptr) {
                if (tid < kRowsPerRank) {
                    const int t = t0 + s * kRowsPerRank + tid;
                    if (t < (int)num_tokens) ss_out[t] = 1.0f;
                }
            } else {
                float* const s_warp_dot = reinterpret_cast<float*>(smem + kOffWarpDot);
                cluster.sync();  // all 4 ranks' sWarpDot[0..1][0..63] complete
                if (tid < kRowsPerRank) {
                    const int row = s * kRowsPerRank + tid;
                    const int t = t0 + row;
                    if (t < (int)num_tokens) {
                        const float* const wd0 =
                            reinterpret_cast<const float*>(cluster.map_shared_rank(s_warp_dot, 0));
                        const float* const wd1 =
                            reinterpret_cast<const float*>(cluster.map_shared_rank(s_warp_dot, 1));
                        const float* const wd2 =
                            reinterpret_cast<const float*>(cluster.map_shared_rank(s_warp_dot, 2));
                        const float* const wd3 =
                            reinterpret_cast<const float*>(cluster.map_shared_rank(s_warp_dot, 3));
                        float total = 0.0f;
                        total += wd0[row];
                        total += wd1[row];
                        total += wd2[row];
                        total += wd3[row];
                        total += wd0[kBM + row];
                        total += wd1[kBM + row];
                        total += wd2[kBM + row];
                        total += wd3[kBM + row];
                        ss_out[t] = 1.0f / (1.0f + __expf(-total));
                    }
                }
                cluster.sync();  // no CTA leaves while another reads its sWarpDot
            }
        }
    }
}

__device__ __forceinline__ int cta_row0_bm64() {
    int bx, by;
    asm volatile("mov.u32 %0, %%ctaid.x;\n\tmov.u32 %1, %%ctaid.y;\n" : "=r"(bx), "=r"(by));
    return by * kBM + bx * kMoeRows;
}

// Single-Launch BM=64 Cluster-Fused MoE Combine + Inject + Inline-W4 Upmix:
//   - Phase 0 (0 B SMEM): overlaps Phase-1 `sB[0..1]` `cp.async` preload on
//     warps 8..15 with `moe_prologue_gatepre<true>` (`moe_form_rows_mx16<true, true>`),
//     forming 16 MoE rows/rank into L2-hot `moe_rows` (`L2::evict_last`) while
//     discarding consumed MXFP8 `expert_output` slot lines.
//   - Preloads the first 4 K-steps of `hyper` (`pre_rh[0..3]`) + `pre_scale`
//     on warps 0..7 behind `cluster.sync()`, then discards dead `shared_out`
//     lines (skipping if aliased with `mixed`) before Phase 1 starts.
//   - Phase 1 + Phase 2: executes `inject_upmix_bm64_w4_body<kAmax, false, true, true, true>`
//     with `moe_rows` as `core` (normal L2 read across all 4 ranks, discarded
//     in L2 right after Phase 1 `cluster.sync()` before Phase 2 starts).
//   - Strict 0-ULP bit-identical to `qwen4_hc_combine_moe_gemm_mx_gatepre` +
//     `qwen4_hc_prep_up` + `qwen4_hc_upmix_roof_t128_r4[_amax]`.
template <bool kAmax, typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void combine_inject_upmix_bm64_w4_body(
    unsigned char* smem,
    __nv_bfloat16* __restrict__ hyper,
    __nv_bfloat16* residual,
    __nv_bfloat16* moe_rows,
    const uint8_t* __restrict__ expert_output,
    const uint8_t* __restrict__ slot_scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    const __nv_bfloat16* __restrict__ folded,
    const uint8_t* __restrict__ up_packed,
    const uint8_t* __restrict__ up_scales,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int num_tokens,
    float scale2_down,
    float scale2_inject,
    float alpha_up,
    float eps,
    unsigned int* __restrict__ amax_bits) {
    if (threadIdx.x >= kProducers) {
        const int s_pre = blockIdx.x;
        preload_b_u32_stage(smem, folded, s_pre, 0, 0);
        preload_b_u32_stage(smem, folded, s_pre, 1, 1);
    }
    const MoeRows m{expert_output, slot_scales, token_to_perm, topk_weights, shared_out, nullptr, nullptr, moe_rows};
    const MoeScratch sc{nullptr, nullptr, nullptr, nullptr};
    moe_prologue_gatepre<true>(m, gate_scalars, sc, cta_row0_bm64(), num_tokens);

    float pre_scale = 0.0f;
    Vec8 pre_rh[4];
    if (threadIdx.x < kProducers) {
        const int s = blockIdx.x;
        const int t0 = blockIdx.y * kBM;
        const int pr = threadIdx.x >> 2;
        const int pv = threadIdx.x & 3;
        const int ptoken = t0 + pr;
        const bool prow = ptoken < (int)num_tokens;
        const size_t row_idx = prow ? (size_t)ptoken : 0;
        const __nv_bfloat16* hrow = hyper + row_idx * kWidth + s * kHidden + pv * 8;
        pre_scale = prow ? __bfloat162float(residual[row_idx * kWidth + (kWidth - kStreams) + s]) : 0.0f;
        const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int stg = 0; stg < 4; ++stg) {
            pre_rh[stg].raw = prow ? load_row16<true>(hrow + stg * kBK) : zero;
        }
    }

    cooperative_groups::this_cluster().sync();  // all 64 formed moe_rows visible; shared_out reads done

    if (shared_out != nullptr && shared_out != mixed &&
        ((reinterpret_cast<uintptr_t>(shared_out) & 127u) == 0u)) {
        const int row0_sh = cta_row0_bm64();
        const int valid_sh = min(kMoeRows, max(0, (int)num_tokens - row0_sh));
        const unsigned char* base_sh =
            reinterpret_cast<const unsigned char*>(shared_out + (size_t)row0_sh * kHidden);
        const int sh_lines = valid_sh * (kHidden * (int)sizeof(__nv_bfloat16) / 128);
        for (int line = threadIdx.x; line < sh_lines; line += kThreads) {
            const unsigned char* p = base_sh + (size_t)line * 128;
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
        }
    }

    inject_upmix_bm64_w4_body<kAmax, false, true, true, true>(
        smem, hyper, moe_rows, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        amax_bits, nullptr, nullptr, nullptr, nullptr, pre_scale, pre_rh);
}

}  // namespace hc_bm64_upmix

#define QWEN4_HC_BM64_UPMIX_W4_PARAMS                                                                   \
    __nv_bfloat16 *__restrict__ hyper, const __nv_bfloat16 *__restrict__ core, __nv_bfloat16 *residual, \
        const __nv_bfloat16 *__restrict__ folded, const uint8_t *__restrict__ up_packed,                \
        const uint8_t *__restrict__ up_scales, const __nv_bfloat16 *__restrict__ norm_w,                \
        __nv_bfloat16 *__restrict__ mixed, unsigned int num_tokens, unsigned int rank,                  \
        unsigned int hidden_size, unsigned int hc, float scale2_down, float scale2_inject,              \
        float alpha_up, float eps

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_upmix_bm64_w4(QWEN4_HC_BM64_UPMIX_W4_PARAMS) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core == nullptr) __trap();
    hc_bm64_upmix::check_bm64_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm64_upmix::inject_upmix_bm64_w4_body<false, false>(
        smem, hyper, core, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        nullptr, nullptr, nullptr, nullptr, nullptr);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_upmix_bm64_w4_amax(
    QWEN4_HC_BM64_UPMIX_W4_PARAMS,
    unsigned int* __restrict__ amax_bits) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core == nullptr) __trap();
    hc_bm64_upmix::check_bm64_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm64_upmix::inject_upmix_bm64_w4_body<true, false>(
        smem, hyper, core, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        amax_bits, nullptr, nullptr, nullptr, nullptr);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_upmix_bm64_w4_mxfp8_gate(
    QWEN4_HC_BM64_UPMIX_W4_PARAMS,
    unsigned char* __restrict__ xq,
    unsigned char* __restrict__ xs,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core == nullptr) __trap();
    hc_bm64_upmix::check_bm64_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm64_upmix::inject_upmix_bm64_w4_body<false, true>(
        smem, hyper, core, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        nullptr, xq, xs, shared_gate, ss_out);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_upmix_bm64_w4_mxfp8_gate_amax(
    QWEN4_HC_BM64_UPMIX_W4_PARAMS,
    unsigned int* __restrict__ amax_bits,
    unsigned char* __restrict__ xq,
    unsigned char* __restrict__ xs,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core == nullptr) __trap();
    hc_bm64_upmix::check_bm64_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm64_upmix::inject_upmix_bm64_w4_body<true, true>(
        smem, hyper, core, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        amax_bits, xq, xs, shared_gate, ss_out);
}

#define QWEN4_HC_BM64_COMBINE_UPMIX_W4_PARAMS                                                                    \
    __nv_bfloat16 *__restrict__ hyper, __nv_bfloat16 *residual, __nv_bfloat16 *moe_rows,                         \
        const uint8_t *__restrict__ expert_output, const uint8_t *__restrict__ slot_scales,                      \
        const int *__restrict__ token_to_perm, const float *__restrict__ topk_weights,                           \
        const __nv_bfloat16 *__restrict__ shared_out, const float *__restrict__ gate_scalars,                    \
        const __nv_bfloat16 *__restrict__ folded, const uint8_t *__restrict__ up_packed,                         \
        const uint8_t *__restrict__ up_scales, const __nv_bfloat16 *__restrict__ norm_w,                         \
        __nv_bfloat16 *__restrict__ mixed, unsigned int num_tokens, unsigned int rank,                           \
        unsigned int hidden_size, unsigned int hc, unsigned int topk, float scale2_down,                         \
        float scale2_inject, float alpha_up, float eps

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_combine_inject_upmix_bm64_w4_mx_gatepre(QWEN4_HC_BM64_COMBINE_UPMIX_W4_PARAMS) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || topk != kTopk ||
        moe_rows == nullptr || expert_output == nullptr || slot_scales == nullptr ||
        ((reinterpret_cast<uintptr_t>(slot_scales) & 3u) != 0u) ||
        token_to_perm == nullptr || topk_weights == nullptr || shared_out == nullptr ||
        gate_scalars == nullptr) {
        __trap();
    }
    hc_bm64_upmix::check_bm64_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm64_upmix::combine_inject_upmix_bm64_w4_body<false>(
        smem, hyper, residual, moe_rows, expert_output, slot_scales, token_to_perm, topk_weights,
        shared_out, gate_scalars, folded, up_packed, up_scales, norm_w, mixed, num_tokens,
        scale2_down, scale2_inject, alpha_up, eps, nullptr);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_combine_inject_upmix_bm64_w4_mx_gatepre_amax(
    QWEN4_HC_BM64_COMBINE_UPMIX_W4_PARAMS,
    unsigned int* __restrict__ amax_bits) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || topk != kTopk ||
        moe_rows == nullptr || expert_output == nullptr || slot_scales == nullptr ||
        ((reinterpret_cast<uintptr_t>(slot_scales) & 3u) != 0u) ||
        token_to_perm == nullptr || topk_weights == nullptr || shared_out == nullptr ||
        gate_scalars == nullptr || amax_bits == nullptr) {
        __trap();
    }
    hc_bm64_upmix::check_bm64_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm64_upmix::combine_inject_upmix_bm64_w4_body<true>(
        smem, hyper, residual, moe_rows, expert_output, slot_scales, token_to_perm, topk_weights,
        shared_out, gate_scalars, folded, up_packed, up_scales, norm_w, mixed, num_tokens,
        scale2_down, scale2_inject, alpha_up, eps, amax_bits);
}

