// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// BM=32 2-CTA/SM native Blackwell NVFP4 TC cluster-fused mHC kernels (47,232 B SMEM).

// ============================================================================
// Opt-In 2-CTA/SM (`47,232 B` SMEM) Native Blackwell NVFP4 TC Cluster-Fused
// mHC (`qwen4_hc_[combine_]inject_upmix_bm32_nvfp4_2cta[_mxfp8_gate][_amax]`):
//   - SM120 / SM121 ONLY (`__CUDA_ARCH__ >= 1200` with arch/family-specific
//     target; traps on SM103 and generic PTX because NVFP4 block-scaled MMA
//     requires SM120/SM121).
//   - 4-CTA cluster, 256 threads/CTA, `47,232 B` dynamic SMEM -> 2 CTAs/SM
//     (`94` clusters/wave on 188-SM SM120, doubling cluster occupancy vs
//     1-CTA/SM `bm64_w4` and hiding barrier/DRAM stalls across resident CTAs).
//   - Phase 1 (`kBM2 = 32`, `hc_2cta`): strict 0-ULP `hyper'` + `rstd` +
//     folded BF16 GEMM, pinning `hyper'` with `L2::evict_last` on SM120
//     (`normal` on SM121).
//   - Phase 1 -> 2 handoff: 8 warps quantize their 8 owned `down` rows in-place
//     to NVFP4 (`E2M1` + block-16 `UE4M3` + per-row FP32 `scale2_a`), shrinking
//     DSMEM activation transfer 3.5x.
//   - Phase 2: native `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.
//     m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3` (zero `decode16_nvfp4_bf16`,
//     zero inner-K barriers across all 5 K-steps, overlapping next H-tile
//     `cp.async` with `sO` + `mixed` stores via `cp.async.wait_group 1`).
// ============================================================================
namespace hc_bm32_nvfp4 {

constexpr int kBM32 = hc_2cta::kBM2;            // 32
constexpr int kThreads32 = hc_2cta::kThreads2;  // 256
constexpr int kBH32 = 32;
constexpr int kRank32 = Qwen4FlashNextShape::kMhcRank;
constexpr int kRankHalf = Qwen4FlashNextShape::kMhcRankHalf;
constexpr int kRankGroups = Qwen4FlashNextShape::kMhcRankNvfp4Groups;
constexpr int kSmemBytes = hc_2cta::kSmemBytes2;  // 47,232 B (2 CTAs/SM exact)

// Phase 1 DSMEM handoff inside [41472 .. 47104) (between sQ [0..41472) and sRstd [47104..47232))
constexpr int kOffDownOwn = kBM32 * kNTotal * 4;
constexpr int kOffDownFp4 = kOffDownOwn;
constexpr int kOffDownSc = kOffDownFp4 + (kBM32 / kStreams) * kRankHalf;
constexpr int kOffDownS2 = kOffDownSc + (kBM32 / kStreams) * kRankGroups;
static_assert(kOffDownOwn + (kBM32 / kStreams) * kRank32 * 2 <= hc_2cta::kOffRstd2,
              "bm32_nvfp4 temp sDownOwn fits between sQ and sRstd");

// Phase 2 SMEM layout (active after DSMEM copy + cluster.sync())
constexpr int kOffA4 = 0;
constexpr int kOffSA4 = kOffA4 + kBM32 * kRankHalf;
constexpr int kOffS2A4 = kOffSA4 + kBM32 * kRankGroups;
constexpr int kOffRstd4 = kOffS2A4 + kBM32 * 4;
constexpr int kOffBp4 = kOffRstd4 + kBM32 * kStreams * 4;
constexpr int kOffBs4 = kOffBp4 + kStreams * kBH32 * kRankHalf;
constexpr int kOffH4 = kOffBs4 + kStreams * kBH32 * kRankGroups;
constexpr int kOffO4 = kOffH4 + kBM32 * kStreams * kBH32 * 2;
constexpr int kOffWarpDot4 = kOffO4 + kBM32 * kBH32 * 2;
constexpr int kOffDotAcc4 = kOffWarpDot4 + 2 * kBM32 * 4;
static_assert(kOffRstd4 + kBM32 * kStreams * 4 <= kOffDownFp4,
              "bm32_nvfp4 Phase 2 DSMEM dst disjoint from Phase 1 handoff");
static_assert(kOffDotAcc4 + 4 * 8 * 32 * 4 <= kSmemBytes,
              "bm32_nvfp4 Phase 2 SMEM fits within 47,232 B 2-CTA/SM budget");
static_assert(kSmemBytes == 47232, "launcher passes 47232 bytes of dynamic smem for bm32_nvfp4");

__device__ __forceinline__ void check_bm32_nvfp4_smem() {
    unsigned int dyn;
    asm("mov.u32 %0, %%dynamic_smem_size;\n" : "=r"(dyn));
    if (dyn < (unsigned int)kSmemBytes) __trap();
}

__device__ __forceinline__ unsigned int quant8_e2m1(const float* v, float q_inv) {
#if Q4_HC_HAS_E2M1_CVT
    unsigned int r;
    const float v0 = v[0] * q_inv, v1 = v[1] * q_inv, v2 = v[2] * q_inv, v3 = v[3] * q_inv;
    const float v4 = v[4] * q_inv, v5 = v[5] * q_inv, v6 = v[6] * q_inv, v7 = v[7] * q_inv;
    asm("{\n\t"
        ".reg .b8 b0, b1, b2, b3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n\t"
        "mov.b32 %0, {b0, b1, b2, b3};\n\t"
        "}"
        : "=r"(r)
        : "f"(v0), "f"(v1), "f"(v2), "f"(v3), "f"(v4), "f"(v5), "f"(v6), "f"(v7));
    return r;
#else
    (void)v;
    (void)q_inv;
    __trap();
    return 0u;
#endif
}

__device__ __forceinline__ unsigned char cvt_f32_to_e4m3(float x) {
    unsigned short p;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p) : "f"(x), "f"(0.0f));
    return static_cast<unsigned char>(p & 0xffu);
}

__device__ __forceinline__ float cvt_e4m3_to_f32(unsigned char s) {
#if Q4_HC_HAS_E2M1_CVT
    uint32_t h2;
    const unsigned short sb2 = static_cast<unsigned short>(s);
    asm("{ .reg .b16 hs; mov.b16 hs, %1; cvt.rn.f16x2.e4m3x2 %0, hs; }" : "=r"(h2) : "h"(sb2));
    return __half2float(*reinterpret_cast<const __half*>(&h2));
#else
    __nv_fp8_e4m3 fp8;
    *reinterpret_cast<unsigned char*>(&fp8) = s;
    return static_cast<float>(fp8);
#endif
}

__device__ __forceinline__ void mma_nvfp4_m16n8k64(
    float (&d)[4],
    unsigned int a0,
    unsigned int a1,
    unsigned int a2,
    unsigned int a3,
    unsigned int b0,
    unsigned int b1,
    unsigned int sfa,
    unsigned int sfb) {
#if Q4_HC_HAS_SM120_MX_MMA
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
        "{%10},{%11,%12},{%13},{%14,%15};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0),
          "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0));
#else
    (void)d;
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)b0;
    (void)b1;
    (void)sfa;
    (void)sfb;
    __trap();
#endif
}

template <bool kCoreNormalL2 = false, bool kPreloadedH = false>
__device__ __forceinline__ float produce_rows_inject_bm32(
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
    const unsigned int offA = hc_2cta::kSmemB2;
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
        *reinterpret_cast<uint4*>(smem + offA + buf * (kBM32 * 64) + hc_2cta::swz64(pr, pv * 8)) = out.raw;
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
            hc_2cta::pipe_barrier_256();
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
        float wsum = (x0 + x2) + (x1 + x3);
        wsum += __shfl_xor_sync(0xffffffffu, wsum, 2);
        wsum += __shfl_xor_sync(0xffffffffu, wsum, 1);
        half[g] = wsum;
    }
    return rsqrtf(hc_div_approx(half[0] + half[1], (float)kHidden) + eps);
}

template <bool kAmax, bool kMxfp8Gate, bool kCoreNormalL2 = false, bool kPreloadedB = false, bool kPreloadedH = false,
          typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void inject_upmix_bm32_nvfp4_body(
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
    constexpr int kRowsPerRank = kBM32 / kStreams;  // 8

    // ---- Phase 1: 2-CTA/SM Inject + rstd + BF16 folded GEMM + in-place NVFP4 down quant ----
    {
        const int s = blockIdx.x;
        const int t0 = blockIdx.y * kBM32;
        const int tid = threadIdx.x;
        unsigned int sbase1 = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
        asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n"
                     : "+r"(sbase1));
        float* srstd = reinterpret_cast<float*>(smem + hc_2cta::kOffRstd2);
        float p1_acc[hc_2cta::kFrags2][4];
        if (tid < hc_2cta::kProd2) {
            const float r = produce_rows_inject_bm32<kCoreNormalL2, kPreloadedH>(
                smem, hyper, core, residual, s, t0, num_tokens, eps, pre_scale, pre_rh);
            if ((tid & 3) == 0) {
                srstd[tid >> 2] = r;
            }
        } else {
            hc_2cta::multiply_rows_2cta<kPreloadedB>(sbase1, folded, s, p1_acc);
        }
        __syncthreads();

        float* __restrict__ sq = reinterpret_cast<float*>(smem);
        if (tid >= hc_2cta::kProd2) {
            const int mt = tid - hc_2cta::kProd2;
            const int lane = mt & 31;
            const int wm = (mt >> 5) >> 1;
            const int wn = (mt >> 5) & 1;
            const int row0 = wm * 16 + (lane >> 2);
            const int row1 = row0 + 8;
            const float rs0 = srstd[row0];
            const float rs1 = srstd[row1];
#pragma unroll
            for (int f = 0; f < hc_2cta::kFrags2; ++f) {
                const int n = wn * (hc_2cta::kFrags2 * 8) + f * 8 + 2 * (lane & 3);
                if (n >= kNTotal) continue;
                *reinterpret_cast<float2*>(&sq[row0 * kNTotal + n]) =
                    make_float2(rs0 * p1_acc[f][0], rs0 * p1_acc[f][1]);
                *reinterpret_cast<float2*>(&sq[row1 * kNTotal + n]) =
                    make_float2(rs1 * p1_acc[f][2], rs1 * p1_acc[f][3]);
            }
        }
        cluster.sync();

        const float2* const sq2 = reinterpret_cast<const float2*>(sq);
        const float2* const rq1 = reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 1) & 3));
        const float2* const rq2 = reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 2) & 3));
        const float2* const rq3 = reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 3) & 3));
        __nv_bfloat16* const sdown_own = reinterpret_cast<__nv_bfloat16*>(smem + kOffDownOwn);
        constexpr int kPairsPerRow = kNTotal / 2;
        constexpr int kPairsPerRank = kRowsPerRank * kPairsPerRow;
        for (int idx = tid; idx < kPairsPerRank; idx += kThreads32) {
            const int r_local = idx / kPairsPerRow;
            const int p = idx - r_local * kPairsPerRow;
            const int row = s * kRowsPerRank + r_local;
            const int t = t0 + row;
            const unsigned int n = (unsigned int)(2 * p);
            const int q_pair_idx = row * kPairsPerRow + p;
            const float2 vs = sq2[q_pair_idx], v1 = rq1[q_pair_idx], v2 = rq2[q_pair_idx], v3 = rq3[q_pair_idx];
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
            if (n < (unsigned int)kRank32) {
                const float x0 = (scale2_down * sum0) * (1.0f / (float)kStreams);
                const float x1 = (scale2_down * sum1) * (1.0f / (float)kStreams);
                const __nv_bfloat162 dval =
                    __floats2bfloat162_rn(hc_fast_div_rn_1p_exp(x0), hc_fast_div_rn_1p_exp(x1));
                *reinterpret_cast<__nv_bfloat162*>(&sdown_own[r_local * kRank32 + n]) =
                    (t < (int)num_tokens) ? dval : __floats2bfloat162_rn(0.0f, 0.0f);
            } else if (t < (int)num_tokens) {
                const float x0 = (scale2_inject * sum0) * (1.0f / (float)kStreams);
                const float x1 = (scale2_inject * sum1) * (1.0f / (float)kStreams);
                *reinterpret_cast<__nv_bfloat162*>(
                    &residual[(size_t)t * kWidth + (kWidth - kStreams) + (n - kRank32)]) =
                    __floats2bfloat162_rn(2.0f * hc_fast_rcp_rn_1p_exp(x0), 2.0f * hc_fast_rcp_rn_1p_exp(x1));
            }
        }
        __syncthreads();

        // 8 warps quantize 8 owned `down` rows in-place -> NVFP4 (E2M1 + UE4M3 + per-row scale2_a)
        {
            const int r_local = tid >> 5;
            const int lane = tid & 31;
            float fv[16];
            float gmax = 0.0f;
            if (lane < 20) {
                const __nv_bfloat162* src =
                    reinterpret_cast<const __nv_bfloat162*>(sdown_own + r_local * kRank32 + lane * 16);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float2 f = __bfloat1622float2(src[j]);
                    fv[2 * j] = f.x;
                    fv[2 * j + 1] = f.y;
                    gmax = fmaxf(gmax, fmaxf(fabsf(f.x), fabsf(f.y)));
                }
            }
            float rmax = gmax;
#pragma unroll
            for (int d = 16; d > 0; d >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, d));
            const bool pos = isfinite(rmax) && (rmax > 1e-28f);
            const float inv_s2_over_6 = pos ? __fdiv_rn(448.0f, rmax) : 0.0f;
            const float s2_row = pos ? __fdiv_rn(rmax, 6.0f * 448.0f) : 0.0f;
            unsigned int q0 = 0u, q1 = 0u;
            unsigned char sc_byte = 0u;
            if (lane < 20 && pos && gmax > 0.0f) {
                const float sf_f = fminf(__fmul_rn(gmax, inv_s2_over_6), 448.0f);
                unsigned int sf_b = static_cast<unsigned int>(cvt_f32_to_e4m3(sf_f)) & 0x7fu;
                if (cvt_e4m3_to_f32(static_cast<unsigned char>(sf_b)) < sf_f && sf_b < 0x7eu) ++sf_b;
                sc_byte = static_cast<unsigned char>(sf_b);
                const float eff = __fmul_rn(cvt_e4m3_to_f32(sc_byte), s2_row);
                const float q_inv = (eff > 0.0f) ? __fdiv_rn(1.0f, eff) : 0.0f;
                q0 = quant8_e2m1(fv, q_inv);
                q1 = quant8_e2m1(fv + 8, q_inv);
            }
            __syncthreads();
            if (lane < 20) {
                reinterpret_cast<uint2*>(smem + kOffDownFp4)[r_local * 20 + lane] = make_uint2(q0, q1);
                (smem + kOffDownSc)[r_local * 20 + lane] = sc_byte;
            }
            if (lane == 0) {
                reinterpret_cast<float*>(smem + kOffDownS2)[r_local] = s2_row;
            }
        }

        if (core != mixed && ((reinterpret_cast<uintptr_t>(core) & 127u) == 0u)) {
            const int row0_disc = t0 + s * kRowsPerRank;
            const int valid_rows = min(kRowsPerRank, max(0, (int)num_tokens - row0_disc));
            const unsigned char* base_disc =
                reinterpret_cast<const unsigned char*>(core + (size_t)row0_disc * kHidden);
            const int total_lines = valid_rows * (kHidden * (int)sizeof(__nv_bfloat16) / 128);
            for (int line = tid; line < total_lines; line += kThreads32) {
                const unsigned char* p = base_disc + (size_t)line * 128;
                asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
            }
        }

        asm volatile("fence.proxy.async.global;\n\tfence.acq_rel.gpu;\n" ::: "memory");
        cluster.sync();
    }

    // ---- Phase 2 Setup: DSMEM copy quantized A [32,160] + sSA [32,20] + sScale2A [32] + sRstd [32,4] ----
    const int s = blockIdx.x;
    const int t0 = blockIdx.y * kBM32;
    const int tid = threadIdx.x;
    unsigned int sbase = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n"
                 : "+r"(sbase));
    {
        const uint4* fp0 = reinterpret_cast<const uint4*>(cluster.map_shared_rank(smem + kOffDownFp4, 0));
        const uint4* fp1 = reinterpret_cast<const uint4*>(cluster.map_shared_rank(smem + kOffDownFp4, 1));
        const uint4* fp2 = reinterpret_cast<const uint4*>(cluster.map_shared_rank(smem + kOffDownFp4, 2));
        const uint4* fp3 = reinterpret_cast<const uint4*>(cluster.map_shared_rank(smem + kOffDownFp4, 3));
        for (int idx = tid; idx < kBM32 * (kRankHalf / 16); idx += kThreads32) {
            const int row = idx / 10;
            const int c = idx - row * 10;
            const int sr = row >> 3;
            const int rl = row & 7;
            const uint4* src = (sr == 0) ? fp0 : ((sr == 1) ? fp1 : ((sr == 2) ? fp2 : fp3));
            reinterpret_cast<uint4*>(smem + kOffA4)[idx] = src[rl * 10 + c];
        }
        if (tid < kBM32 * (kRankGroups / 4)) {
            const uint32_t* sc0 = reinterpret_cast<const uint32_t*>(cluster.map_shared_rank(smem + kOffDownSc, 0));
            const uint32_t* sc1 = reinterpret_cast<const uint32_t*>(cluster.map_shared_rank(smem + kOffDownSc, 1));
            const uint32_t* sc2 = reinterpret_cast<const uint32_t*>(cluster.map_shared_rank(smem + kOffDownSc, 2));
            const uint32_t* sc3 = reinterpret_cast<const uint32_t*>(cluster.map_shared_rank(smem + kOffDownSc, 3));
            const int row = tid / 5;
            const int c = tid - row * 5;
            const int sr = row >> 3;
            const int rl = row & 7;
            const uint32_t* src = (sr == 0) ? sc0 : ((sr == 1) ? sc1 : ((sr == 2) ? sc2 : sc3));
            reinterpret_cast<uint32_t*>(smem + kOffSA4)[tid] = src[rl * 5 + c];
        }
        if (tid < kBM32) {
            const float* s20 = reinterpret_cast<const float*>(cluster.map_shared_rank(smem + kOffDownS2, 0));
            const float* s21 = reinterpret_cast<const float*>(cluster.map_shared_rank(smem + kOffDownS2, 1));
            const float* s22 = reinterpret_cast<const float*>(cluster.map_shared_rank(smem + kOffDownS2, 2));
            const float* s23 = reinterpret_cast<const float*>(cluster.map_shared_rank(smem + kOffDownS2, 3));
            const int sr = tid >> 3;
            const int rl = tid & 7;
            const float* src = (sr == 0) ? s20 : ((sr == 1) ? s21 : ((sr == 2) ? s22 : s23));
            reinterpret_cast<float*>(smem + kOffS2A4)[tid] = src[rl] * alpha_up;
        }
        if (tid < kBM32 * kStreams) {
            const float* rs0 = reinterpret_cast<const float*>(cluster.map_shared_rank(smem + hc_2cta::kOffRstd2, 0));
            const float* rs1 = reinterpret_cast<const float*>(cluster.map_shared_rank(smem + hc_2cta::kOffRstd2, 1));
            const float* rs2 = reinterpret_cast<const float*>(cluster.map_shared_rank(smem + hc_2cta::kOffRstd2, 2));
            const float* rs3 = reinterpret_cast<const float*>(cluster.map_shared_rank(smem + hc_2cta::kOffRstd2, 3));
            const int rrow = tid >> 2;
            const int rst = tid & 3;
            const float* src = (rst == 0) ? rs0 : ((rst == 1) ? rs1 : ((rst == 2) ? rs2 : rs3));
            reinterpret_cast<float*>(smem + kOffRstd4)[rrow * kStreams + rst] = src[rrow];
        }
    }
    cluster.sync();

    // ---- Phase 2: 20 strided H-tiles/rank with native Blackwell OMMA.SF.16864 ----
    const int warp = tid >> 5;   // 0..7
    const int lane = tid & 31;   // 0..31
    const int wm = warp >> 2;    // 0..1 (16-row slab)
    const int wh = warp & 3;     // 0..3 (8 h-cols)
    const int group_id = lane >> 2;
    const int tid4 = lane & 3;

    auto issue_up_tile = [&](int ht) {
        const uint64_t pol_el = policy_evict_last();
        const size_t h0 = (size_t)ht * kBH32;
        for (int idx = tid; idx < kStreams * kBH32 * (kRankHalf / 16); idx += kThreads32) {
            const int br = idx / (kRankHalf / 16);
            const int c = idx - br * (kRankHalf / 16);
            const int bst = br >> 5;
            const int bh = br & 31;
            const uint8_t* src =
                up_packed + ((size_t)bst * kHidden + h0 + (size_t)bh) * (size_t)kRankHalf + (size_t)(c * 16);
            hc_bm64_upmix::cp16_hint_u32(sbase + kOffBp4 + (unsigned int)(idx * 16), src, true, pol_el);
        }
        cp_async_commit();
    };
    auto load_up_scales = [&](int ht) {
        const uint64_t pol_el = policy_evict_last();
        const size_t h0 = (size_t)ht * kBH32;
        for (int idx = tid; idx < kStreams * kBH32 * (kRankGroups / 4); idx += kThreads32) {
            const int br = idx / (kRankGroups / 4);
            const int c = idx - br * (kRankGroups / 4);
            const int bst = br >> 5;
            const int bh = br & 31;
            const uint32_t* src = reinterpret_cast<const uint32_t*>(
                up_scales + ((size_t)bst * kHidden + h0 + (size_t)bh) * (size_t)kRankGroups + (size_t)(c * 4));
            uint32_t sv;
            asm volatile("ld.global.ca.L2::cache_hint.u32 %0, [%1], %2;\n" : "=r"(sv) : "l"(src), "l"(pol_el));
            asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(sbase + kOffBs4 + (unsigned int)(idx * 4)), "r"(sv) : "memory");
        }
    };

    issue_up_tile(s);
    load_up_scales(s);

    float amax = 0.0f;
    const bool do_gate_dot = kMxfp8Gate && (ss_out != nullptr) && (shared_gate != nullptr);
    const bool full_tile = ((unsigned int)(t0 + kBM32) <= num_tokens);

#pragma unroll 1
    for (int g = 0; g < 2; ++g) {
        const int warp_id = s + 4 * g;
        if constexpr (kMxfp8Gate) {
            if (do_gate_dot && tid >= 128) {
                const int warp_dot = (tid - 128) >> 5;
                float* const s_dot = reinterpret_cast<float*>(smem + kOffDotAcc4) + (warp_dot * 8 * 32) + lane;
#pragma unroll
                for (int rr = 0; rr < 8; ++rr) s_dot[rr * 32] = 0.0f;
            }
        }

#pragma unroll 1
        for (int ti = 0; ti < 10; ++ti) {
            const int ht = warp_id + 8 * ti;
            const int h0 = ht * kBH32;
            bool has_next = false;
            int next_ht = ht;
            if (ti + 1 < 10) {
                has_next = true;
                next_ht = warp_id + 8 * (ti + 1);
            } else if (g == 0) {
                has_next = true;
                next_ht = s + 4;
            }

            asm volatile("cp.async.wait_group 0;\n\tbar.sync 0;\n" ::: "memory");

            float acc[kStreams][4];
#pragma unroll
            for (int st = 0; st < kStreams; ++st)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[st][e] = 0.0f;

            const int fr0 = wm * 16 + group_id;
            const int fr1 = fr0 + 8;
            const int m_sfa = wm * 16 + group_id + 8 * (lane & 1);
#pragma unroll 1
            for (int kt = 0; kt < 5; ++kt) {
                const int byte_base = kt * 32 + tid4 * 4;
                const int sc_base = kt * 4;
                const unsigned int a0 = *reinterpret_cast<const unsigned int*>(smem + kOffA4 + fr0 * kRankHalf + byte_base);
                const unsigned int a1 = *reinterpret_cast<const unsigned int*>(smem + kOffA4 + fr1 * kRankHalf + byte_base);
                const unsigned int a2 = *reinterpret_cast<const unsigned int*>(smem + kOffA4 + fr0 * kRankHalf + byte_base + 16);
                const unsigned int a3 = *reinterpret_cast<const unsigned int*>(smem + kOffA4 + fr1 * kRankHalf + byte_base + 16);
                const unsigned int sfa = *reinterpret_cast<const unsigned int*>(smem + kOffSA4 + m_sfa * kRankGroups + sc_base);
#pragma unroll
                for (int st = 0; st < kStreams; ++st) {
                    const int nc = st * 32 + wh * 8 + group_id;
                    const unsigned int b0 = *reinterpret_cast<const unsigned int*>(smem + kOffBp4 + nc * kRankHalf + byte_base);
                    const unsigned int b1 = *reinterpret_cast<const unsigned int*>(smem + kOffBp4 + nc * kRankHalf + byte_base + 16);
                    const unsigned int sfb = *reinterpret_cast<const unsigned int*>(smem + kOffBs4 + nc * kRankGroups + sc_base);
                    mma_nvfp4_m16n8k64(acc[st], a0, a1, a2, a3, b0, b1, sfa, sfb);
                }
            }
            __syncthreads();

            // Issue sH cp.async (group A), then next H-tile `up` cp.async (group B) overlapped with epilogue!
            {
                const uint64_t pol_ef_h = policy_evict_first();
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int idx = tid + j * kThreads32;
                    const int piece = idx & 3;
                    const int st = (idx >> 2) & 3;
                    const int row = idx >> 4;
                    const int t = t0 + row;
                    const bool ok = full_tile || (t < (int)num_tokens);
                    const unsigned int dst = sbase + kOffH4 + hc_bm64_upmix::swz_h2(row, st, piece);
                    const __nv_bfloat16* src = hyper + (size_t)(ok ? t : 0) * kWidth + st * kHidden + h0 + piece * 8;
                    hc_bm64_upmix::cp16_hint_u32(dst, src, ok, pol_ef_h);
                }
            }
            cp_async_commit();
            if (has_next) {
                issue_up_tile(next_ht);
                load_up_scales(next_ht);
            }

            float w1[kStreams][2];
            const int hl = wh * 8 + 2 * (lane & 3);
#pragma unroll
            for (int st = 0; st < kStreams; ++st) {
                const __nv_bfloat162 w2 =
                    *reinterpret_cast<const __nv_bfloat162*>(norm_w + (size_t)st * kHidden + h0 + hl);
                w1[st][0] = 1.0f + __bfloat162float(w2.x);
                w1[st][1] = 1.0f + __bfloat162float(w2.y);
            }
            const float s2_r0 = reinterpret_cast<const float*>(smem + kOffS2A4)[wm * 16 + group_id];
            const float s2_r1 = reinterpret_cast<const float*>(smem + kOffS2A4)[wm * 16 + group_id + 8];
#pragma unroll
            for (int st = 0; st < kStreams; ++st) {
                acc[st][0] = hc_fast_rcp_rn_1p_exp(acc[st][0] * s2_r0);
                acc[st][1] = hc_fast_rcp_rn_1p_exp(acc[st][1] * s2_r0);
                acc[st][2] = hc_fast_rcp_rn_1p_exp(acc[st][2] * s2_r1);
                acc[st][3] = hc_fast_rcp_rn_1p_exp(acc[st][3] * s2_r1);
            }
            if (has_next) {
                asm volatile("cp.async.wait_group 1;\n\tbar.sync 0;\n" : "+f"(acc[kStreams - 1][3]) :: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;\n\tbar.sync 0;\n" : "+f"(acc[kStreams - 1][3]) :: "memory");
            }

            const unsigned int in_piece = (unsigned int)(tid4 * 4);
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = wm * 16 + group_id + half * 8;
                const bool ok_row = full_tile || (t0 + row < (int)num_tokens);
                const unsigned int o_addr = sbase + kOffO4 + hc_bm64_upmix::swz_o2(row, wh) + in_piece;
                if (ok_row) {
                    const float4 r4 = *reinterpret_cast<const float4*>(smem + kOffRstd4 + row * 16);
                    const float r[kStreams] = {r4.x, r4.y, r4.z, r4.w};
                    __nv_bfloat162 h2[kStreams];
#pragma unroll
                    for (int st = 0; st < kStreams; ++st) {
                        unsigned int raw_h;
                        asm volatile("ld.shared.u32 %0, [%1];\n"
                                     : "=r"(raw_h)
                                     : "r"(sbase + kOffH4 + hc_bm64_upmix::swz_h2(row, st, wh) + in_piece));
                        h2[st] = *reinterpret_cast<const __nv_bfloat162*>(&raw_h);
                    }
                    const __nv_bfloat162 out = hc_bm64_upmix::mix_pair_gated(acc, half, h2, r, w1);
                    const unsigned int raw_o = *reinterpret_cast<const unsigned int*>(&out);
                    asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(o_addr), "r"(raw_o) : "memory");
                    if constexpr (kAmax) {
                        const float2 f = __bfloat1622float2(out);
                        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
                    }
                } else {
                    asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(o_addr) : "memory");
                }
            }
            __syncthreads();

            if constexpr (kMxfp8Gate) {
                if (tid < 128) {
                    const int row = tid >> 2;
                    const int piece = tid & 3;
                    const int t = t0 + row;
                    const bool ok = full_tile || (t < (int)num_tokens);
                    const uint64_t pol_ef = policy_evict_first();
                    const unsigned int o_addr = sbase + kOffO4 + hc_bm64_upmix::swz_o2(row, piece);
                    uint4 v;
                    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                                 : "r"(o_addr));
                    if (ok) {
                        __nv_bfloat16* dst = mixed + (size_t)t * kHidden + h0 + piece * 8;
                        if (t + kMixedPinWindow >= (int)num_tokens) {
                            *reinterpret_cast<uint4*>(dst) = v;
                        } else {
                            store_row16_stream(dst, v, pol_ef);
                        }
                    }
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
                    hc_bm64_upmix::mxfp8_quant_quad(fv, q_out, s_out);
                    if (ok) {
                        uint2* qptr = reinterpret_cast<uint2*>(xq + (size_t)t * kHidden + h0 + piece * 8);
                        hc_bm64_upmix::store_row8_hint(qptr, q_out, pol_ef);
                        if (piece == 0) {
                            xs[(size_t)t * (kHidden / kBH32) + ht] = s_out;
                        }
                    }
                } else if (do_gate_dot) {
                    const int warp_dot = (tid - 128) >> 5;  // 0..3
                    const float g_val = __bfloat162float(shared_gate[h0 + lane]);
                    const int piece = lane >> 3;
                    const unsigned int byte_in_piece = (unsigned int)((lane & 7) << 1);
                    float* const s_dot = reinterpret_cast<float*>(smem + kOffDotAcc4) + (warp_dot * 8 * 32) + lane;
#pragma unroll
                    for (int rr = 0; rr < 8; ++rr) {
                        const int row = warp_dot * 8 + rr;
                        const unsigned int o_addr =
                            sbase + kOffO4 + hc_bm64_upmix::swz_o2(row, piece) + byte_in_piece;
                        unsigned short raw_bf;
                        asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(raw_bf) : "r"(o_addr));
                        const __nv_bfloat16 bv = *reinterpret_cast<const __nv_bfloat16*>(&raw_bf);
                        s_dot[rr * 32] += __bfloat162float(bv) * g_val;
                    }
                }
            } else {
                const int row = tid >> 3;
                const int hp = tid & 7;
                const int piece = hp >> 1;
                const int sub = hp & 1;
                const int t = t0 + row;
                if (full_tile || (t < (int)num_tokens)) {
                    const unsigned int o_addr =
                        sbase + kOffO4 + hc_bm64_upmix::swz_o2(row, piece) + (unsigned int)(sub * 8);
                    uint2 v;
                    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n"
                                 : "=r"(v.x), "=r"(v.y)
                                 : "r"(o_addr));
                    __nv_bfloat16* dst = mixed + (size_t)t * kHidden + h0 + hp * 4;
                    if (t + kMixedPinWindow >= (int)num_tokens) {
                        *reinterpret_cast<uint2*>(dst) = v;
                    } else {
                        hc_bm64_upmix::store_row8_hint(dst, v, policy_evict_first());
                    }
                }
            }
        }

        if constexpr (kMxfp8Gate) {
            if (do_gate_dot && tid >= 128) {
                float* const s_warp_dot = reinterpret_cast<float*>(smem + kOffWarpDot4);
                const int warp_dot = (tid - 128) >> 5;
                const float* const s_dot = reinterpret_cast<const float*>(smem + kOffDotAcc4) + (warp_dot * 8 * 32) + lane;
#pragma unroll
                for (int rr = 0; rr < 8; ++rr) {
                    float d = s_dot[rr * 32];
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        d += __shfl_down_sync(0xffffffffu, d, off);
                    }
                    if (lane == 0) {
                        s_warp_dot[g * kBM32 + warp_dot * 8 + rr] = d;
                    }
                }
            }
        }
    }

    if constexpr (kAmax) {
        __syncthreads();
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
        }
        float* const s_max = reinterpret_cast<float*>(smem + kOffO4);
        if (lane == 0) s_max[warp] = amax;
        __syncthreads();
        if (tid == 0) {
            float blk = s_max[0];
#pragma unroll
            for (int w = 1; w < kThreads32 / 32; ++w) blk = fmaxf(blk, s_max[w]);
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
                float* const s_warp_dot = reinterpret_cast<float*>(smem + kOffWarpDot4);
                cluster.sync();
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
                        total += wd0[kBM32 + row];
                        total += wd1[kBM32 + row];
                        total += wd2[kBM32 + row];
                        total += wd3[kBM32 + row];
                        ss_out[t] = 1.0f / (1.0f + __expf(-total));
                    }
                }
                cluster.sync();
            }
        }
    }
}

template <bool kAmax, typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void combine_inject_upmix_bm32_nvfp4_body(
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
    {
        if (threadIdx.x >= hc_2cta::kProd2) {
            unsigned int sbase_pre = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
            asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n" : "+r"(sbase_pre));
            const __nv_bfloat16* wsrc = folded + (size_t)blockIdx.x * kNTotal * kHidden;
            hc_2cta::load_b_2cta(sbase_pre, wsrc, threadIdx.x - hc_2cta::kProd2, 0, 0, policy_evict_last());
        }
        const MoeRows m{expert_output, slot_scales, token_to_perm, topk_weights, shared_out, nullptr, nullptr, moe_rows};
        hc_2cta::moe_form_rows_mx16_2cta(m, gate_scalars, hc_2cta::cta_row0_2cta(), num_tokens);
    }

    cooperative_groups::this_cluster().sync();

    if (shared_out != nullptr && shared_out != mixed &&
        ((reinterpret_cast<uintptr_t>(shared_out) & 127u) == 0u)) {
        const int row0_sh = hc_2cta::cta_row0_2cta();
        const int valid_sh = min(hc_2cta::kMoeRows2, max(0, (int)num_tokens - row0_sh));
        const unsigned char* base_sh =
            reinterpret_cast<const unsigned char*>(shared_out + (size_t)row0_sh * kHidden);
        const int sh_lines = valid_sh * (kHidden * (int)sizeof(__nv_bfloat16) / 128);
        for (int line = threadIdx.x; line < sh_lines; line += kThreads32) {
            const unsigned char* p = base_sh + (size_t)line * 128;
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
        }
    }

    inject_upmix_bm32_nvfp4_body<kAmax, false, true, true, false>(
        smem, hyper, moe_rows, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        amax_bits, nullptr, nullptr, nullptr, nullptr);
}

}  // namespace hc_bm32_nvfp4

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_bm32_nvfp4::kThreads32, 2)
qwen4_hc_inject_upmix_bm32_nvfp4_2cta(QWEN4_HC_BM64_UPMIX_W4_PARAMS) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core == nullptr ||
        up_packed == nullptr || up_scales == nullptr ||
        ((reinterpret_cast<uintptr_t>(up_packed) & 15u) != 0u) ||
        ((reinterpret_cast<uintptr_t>(up_scales) & 3u) != 0u)) {
        __trap();
    }
    hc_bm32_nvfp4::check_bm32_nvfp4_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm32_nvfp4::inject_upmix_bm32_nvfp4_body<false, false>(
        smem, hyper, core, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        nullptr, nullptr, nullptr, nullptr, nullptr);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_bm32_nvfp4::kThreads32, 2)
qwen4_hc_inject_upmix_bm32_nvfp4_2cta_amax(
    QWEN4_HC_BM64_UPMIX_W4_PARAMS,
    unsigned int* __restrict__ amax_bits) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core == nullptr ||
        up_packed == nullptr || up_scales == nullptr ||
        ((reinterpret_cast<uintptr_t>(up_packed) & 15u) != 0u) ||
        ((reinterpret_cast<uintptr_t>(up_scales) & 3u) != 0u)) {
        __trap();
    }
    hc_bm32_nvfp4::check_bm32_nvfp4_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm32_nvfp4::inject_upmix_bm32_nvfp4_body<true, false>(
        smem, hyper, core, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        amax_bits, nullptr, nullptr, nullptr, nullptr);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_bm32_nvfp4::kThreads32, 2)
qwen4_hc_inject_upmix_bm32_nvfp4_2cta_mxfp8_gate(
    QWEN4_HC_BM64_UPMIX_W4_PARAMS,
    unsigned char* __restrict__ xq,
    unsigned char* __restrict__ xs,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core == nullptr ||
        up_packed == nullptr || up_scales == nullptr ||
        ((reinterpret_cast<uintptr_t>(up_packed) & 15u) != 0u) ||
        ((reinterpret_cast<uintptr_t>(up_scales) & 3u) != 0u)) {
        __trap();
    }
    hc_bm32_nvfp4::check_bm32_nvfp4_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm32_nvfp4::inject_upmix_bm32_nvfp4_body<false, true>(
        smem, hyper, core, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        nullptr, xq, xs, shared_gate, ss_out);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_bm32_nvfp4::kThreads32, 2)
qwen4_hc_inject_upmix_bm32_nvfp4_2cta_mxfp8_gate_amax(
    QWEN4_HC_BM64_UPMIX_W4_PARAMS,
    unsigned int* __restrict__ amax_bits,
    unsigned char* __restrict__ xq,
    unsigned char* __restrict__ xs,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core == nullptr ||
        up_packed == nullptr || up_scales == nullptr ||
        ((reinterpret_cast<uintptr_t>(up_packed) & 15u) != 0u) ||
        ((reinterpret_cast<uintptr_t>(up_scales) & 3u) != 0u)) {
        __trap();
    }
    hc_bm32_nvfp4::check_bm32_nvfp4_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm32_nvfp4::inject_upmix_bm32_nvfp4_body<true, true>(
        smem, hyper, core, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        amax_bits, xq, xs, shared_gate, ss_out);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_bm32_nvfp4::kThreads32, 2)
qwen4_hc_combine_inject_upmix_bm32_nvfp4_2cta_mx_gatepre(QWEN4_HC_BM64_COMBINE_UPMIX_W4_PARAMS) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || topk != kTopk ||
        moe_rows == nullptr || expert_output == nullptr || slot_scales == nullptr ||
        ((reinterpret_cast<uintptr_t>(slot_scales) & 3u) != 0u) ||
        token_to_perm == nullptr || topk_weights == nullptr || shared_out == nullptr ||
        gate_scalars == nullptr || up_packed == nullptr || up_scales == nullptr ||
        ((reinterpret_cast<uintptr_t>(up_packed) & 15u) != 0u) ||
        ((reinterpret_cast<uintptr_t>(up_scales) & 3u) != 0u)) {
        __trap();
    }
    hc_bm32_nvfp4::check_bm32_nvfp4_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm32_nvfp4::combine_inject_upmix_bm32_nvfp4_body<false>(
        smem, hyper, residual, moe_rows, expert_output, slot_scales, token_to_perm, topk_weights,
        shared_out, gate_scalars, folded, up_packed, up_scales, norm_w, mixed, num_tokens,
        scale2_down, scale2_inject, alpha_up, eps, nullptr);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_bm32_nvfp4::kThreads32, 2)
qwen4_hc_combine_inject_upmix_bm32_nvfp4_2cta_mx_gatepre_amax(
    QWEN4_HC_BM64_COMBINE_UPMIX_W4_PARAMS,
    unsigned int* __restrict__ amax_bits) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || topk != kTopk ||
        moe_rows == nullptr || expert_output == nullptr || slot_scales == nullptr ||
        ((reinterpret_cast<uintptr_t>(slot_scales) & 3u) != 0u) ||
        token_to_perm == nullptr || topk_weights == nullptr || shared_out == nullptr ||
        gate_scalars == nullptr || amax_bits == nullptr ||
        up_packed == nullptr || up_scales == nullptr ||
        ((reinterpret_cast<uintptr_t>(up_packed) & 15u) != 0u) ||
        ((reinterpret_cast<uintptr_t>(up_scales) & 3u) != 0u)) {
        __trap();
    }
    hc_bm32_nvfp4::check_bm32_nvfp4_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm32_nvfp4::combine_inject_upmix_bm32_nvfp4_body<true>(
        smem, hyper, residual, moe_rows, expert_output, slot_scales, token_to_perm, topk_weights,
        shared_out, gate_scalars, folded, up_packed, up_scales, norm_w, mixed, num_tokens,
        scale2_down, scale2_inject, alpha_up, eps, amax_bits);
}

