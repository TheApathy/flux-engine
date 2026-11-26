// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// BM=32 2-CTA/SM W4A8 MXFP8 TC cluster-fused mHC kernels (47,232 B SMEM).

// ============================================================================
// 2-CTA/SM (`47,232 B` SMEM) W4A8 MXFP8 TC Cluster-Fused mHC
// (`qwen4_hc_[combine_]inject_upmix_bm32_mxfp8_2cta[_mxfp8_gate][_amax]`):
//   - 4-CTA cluster, 256 threads/CTA, `47,232 B` dynamic SMEM -> 2 CTAs/SM.
//   - Phase 1 (`kBM2 = 32`, `hc_2cta`): strict 0-ULP `hyper'` + `rstd` +
//     folded BF16 GEMM, followed by lossy in-place MXFP8 (`E4M3` + block-32
//     `UE8M0`) quantization of the 8 owned `down` rows (trading BF16's 7-bit
//     mantissa for E4M3's 3-bit mantissa; higher fidelity than E2M1 across
//     mHC routing-gate boundaries).
//   - Phase 2: tiles `up` in 2 K-halves (`K = 320 = 2 x 160`) using
//     zero-raw-staging register-direct NVFP4 -> W4A8 (`E4M3` + block-32
//     `UE8M0`) transcoding into `21,504 B` `sB8`/`sSB8` SMEM (synchronized
//     before each K-half's MMA loop), native SM120/SM121 `m16n8k32`
//     `scale_vec::1X` UE8M0 Tensor-Core MMA (with FP8 MMA + UE8M0 scale
//     fallback on SM103/generic), and `sH` `cp.async` overlapped with `norm_w`
//     loads and MUFU sigmoids while staying exact at `47,232 B`.
// ============================================================================
namespace hc_bm32_mxfp8 {

constexpr int kBM32 = hc_2cta::kBM2;               // 32
constexpr int kThreads32 = hc_2cta::kThreads2;     // 256
constexpr int kBH32 = 32;
constexpr int kRank32 = Qwen4FlashNextShape::kMhcRank;
constexpr int kRankHalf = Qwen4FlashNextShape::kMhcRankHalf;
constexpr int kRankGroups4 = Qwen4FlashNextShape::kMhcRankNvfp4Groups;
constexpr int kRankGroups8 = Qwen4FlashNextShape::kMhcRankMxfp8Groups;
constexpr int kSmemBytes = hc_2cta::kSmemBytes2;   // 47,232 B (2 CTAs/SM exact)

// Phase 1 DSMEM handoff inside [41472 .. 47104) (between sQ [0..41472) and sRstd [47104..47232))
constexpr int kOffDownOwn = kBM32 * kNTotal * 4;
constexpr int kOffDownFp8 = kOffDownOwn;
constexpr int kOffDownSc8 = kOffDownFp8 + (kBM32 / kStreams) * kRank32;
static_assert(kOffDownOwn + (kBM32 / kStreams) * kRank32 * 2 <= hc_2cta::kOffRstd2,
              "bm32_mxfp8 temp sDownOwn fits between sQ and sRstd");
static_assert(kOffDownSc8 + (kBM32 / kStreams) * 12 <= hc_2cta::kOffRstd2,
              "bm32_mxfp8 quantized handoff fits between sQ and sRstd");

// Phase 2 SMEM layout (active after DSMEM copy + cluster.sync(); exact 47,232 B!)
constexpr int kOffA8 = 0;
constexpr int kOffSA8 = kOffA8 + kBM32 * kRank32;
constexpr int kOffRstd8 = kOffSA8 + kBM32 * 12;
constexpr int kOffB8 = kOffRstd8 + kBM32 * kStreams * 4;
constexpr int kOffSB8 = kOffB8 + kStreams * kBH32 * kRankHalf;
constexpr int kOffH8 = kOffSB8 + kStreams * kBH32 * 8;
constexpr int kOffO8 = kOffH8 + kBM32 * kStreams * kBH32 * 2;
constexpr int kOffWarpDot8 = kOffO8 + kBM32 * kBH32 * 2;
constexpr int kOffDotAcc8 = kOffWarpDot8 + 2 * kBM32 * 4;
static_assert(kOffRstd8 + kBM32 * kStreams * 4 <= kOffDownFp8,
              "bm32_mxfp8 Phase 2 DSMEM dst disjoint from Phase 1 handoff");
static_assert(kOffDotAcc8 + 4 * 8 * 32 * 4 <= kSmemBytes,
              "bm32_mxfp8 Phase 2 SMEM fits within 47,232 B 2-CTA/SM budget");
static_assert(kSmemBytes == 47232, "launcher passes 47232 bytes of dynamic smem for bm32_mxfp8");

__device__ __forceinline__ void check_bm32_mxfp8_smem() {
    unsigned int dyn;
    asm("mov.u32 %0, %%dynamic_smem_size;\n" : "=r"(dyn));
    if (dyn < (unsigned int)kSmemBytes) __trap();
}

__device__ __forceinline__ unsigned int pack_e4m3x4_f32(float a, float b, float c, float d) {
    unsigned short lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(a), "f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(c), "f"(d));
    return static_cast<unsigned int>(lo) | (static_cast<unsigned int>(hi) << 16);
}

__device__ __forceinline__ unsigned int norm_e4m3_scale(unsigned int s_own, int e32) {
    const unsigned int mag = s_own & 0x7fu;
    const bool norm = (mag >= 8u);
    const int lz = __clz(static_cast<int>(mag));
    const int exp_b = norm ? static_cast<int>(mag >> 3) : (29 - lz);
    const unsigned int sig = norm ? (8u | (mag & 7u)) : (mag << ((lz - 28) & 3));
    const int new_exp = exp_b - e32;
    const unsigned int normal_res = (static_cast<unsigned int>(new_exp) << 3) | (sig & 7u);
    const unsigned int shift = static_cast<unsigned int>(min(max(1, 1 - new_exp), 5));
    const unsigned int q = sig >> shift;
    const unsigned int half = 1u << (shift - 1u);
    const unsigned int sub_res = (sig + half - 1u + (q & 1u)) >> shift;
    unsigned int res = (new_exp >= 1) ? normal_res : sub_res;
    res = (mag == 0u) ? 0u : res;
    res = (mag == 0x7fu) ? 0x7fu : res;
    return res;
}

__device__ __forceinline__ uint2 e4m3_lut_ptx(unsigned int r_byte) {
    const unsigned short r_pair =
        static_cast<unsigned short>((r_byte & 0xffu) | ((r_byte & 0xffu) << 8));
    unsigned int h_r, m0, m1, m2, m3;
    unsigned short p0, p1, p2, p3;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(h_r) : "h"(r_pair));
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(m0) : "r"(h_r), "r"(0x38000000u));
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(m1) : "r"(h_r), "r"(0x3e003c00u));
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(m2) : "r"(h_r), "r"(0x42004000u));
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(m3) : "r"(h_r), "r"(0x46004400u));
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(p0) : "r"(m0));
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(p1) : "r"(m1));
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(p2) : "r"(m2));
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(p3) : "r"(m3));
    unsigned int lo, hi;
    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo) : "r"(static_cast<unsigned int>(p0)), "r"(static_cast<unsigned int>(p1)));
    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(hi) : "r"(static_cast<unsigned int>(p2)), "r"(static_cast<unsigned int>(p3)));
    return make_uint2(lo, hi);
}

__device__ __forceinline__ uint2 decode8_nvfp4_e4m3(unsigned int p4, uint2 lut) {
    const unsigned int lo = p4 & 0xffffu;
    const unsigned int hi = p4 >> 16;
    unsigned int w0, w1;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(w0) : "r"(lut.x), "r"(lut.y), "r"(lo & 0x7777u));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(w1) : "r"(lut.x), "r"(lut.y), "r"(hi & 0x7777u));
    const unsigned int s0 = ((lo & 0x0008u) << 4) | ((lo & 0x0080u) << 8) |
                            ((lo & 0x0800u) << 12) | ((lo & 0x8000u) << 16);
    const unsigned int s1 = ((hi & 0x0008u) << 4) | ((hi & 0x0080u) << 8) |
                            ((hi & 0x0800u) << 12) | ((hi & 0x8000u) << 16);
    return make_uint2(w0 ^ s0, w1 ^ s1);
}

__device__ __forceinline__ void mma_mxfp8_m16n8k32(
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
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X."
        "f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{0,0},%11,{0,0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
#else
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(t0), "=f"(t1), "=f"(t2), "=f"(t3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(t0), "f"(t1), "f"(t2), "f"(t3));
    unsigned int lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    const unsigned int tig = lane & 3u;
    const unsigned int ea_lo = sfa & 0xffu;
    const unsigned int ea_hi = (sfa >> 8) & 0xffu;
    const unsigned int sb_b = sfb & 0xffu;
    auto scale_pair = [](float t, unsigned int ea, unsigned int eb) -> float {
        const unsigned int s = ea + eb;
        const unsigned int h1 = s >> 1;
        const unsigned int h2 = s - h1;
        const unsigned int fa_bits = (ea == 255u || eb == 255u) ? 0x7fc00000u : (h1 << 23);
        return (t * __uint_as_float(fa_bits)) * __uint_as_float(h2 << 23);
    };
    const unsigned int eb_even = __shfl_sync(0xffffffffu, sb_b, tig << 3);
    const unsigned int eb_odd = __shfl_sync(0xffffffffu, sb_b, (tig << 3) | 4u);
    d[0] += scale_pair(t0, ea_lo, eb_even);
    d[1] += scale_pair(t1, ea_lo, eb_odd);
    d[2] += scale_pair(t2, ea_hi, eb_even);
    d[3] += scale_pair(t3, ea_hi, eb_odd);
#endif
}

template <bool kAmax, bool kMxfp8Gate, bool kCoreNormalL2 = false, bool kPreloadedB = false, bool kPreloadedH = false,
          typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void inject_upmix_bm32_mxfp8_body(
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

    // ---- Phase 1: 2-CTA/SM Inject + rstd + BF16 folded GEMM + in-place MXFP8 down quant ----
    {
        const int tid = threadIdx.x;
        unsigned int sbase1 = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
        asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n"
                     : "+r"(sbase1));
        float* srstd = reinterpret_cast<float*>(smem + hc_2cta::kOffRstd2);
        float p1_acc[hc_2cta::kFrags2][4];
        if (tid < hc_2cta::kProd2) {
            const float r = hc_bm32_nvfp4::produce_rows_inject_bm32<kCoreNormalL2, kPreloadedH>(
                smem, hyper, core, residual, blockIdx.x, blockIdx.y * kBM32, num_tokens, eps, pre_scale, pre_rh);
            if ((tid & 3) == 0) {
                srstd[tid >> 2] = r;
            }
        } else {
            hc_2cta::multiply_rows_2cta<kPreloadedB>(sbase1, folded, blockIdx.x, p1_acc);
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

        const int s = blockIdx.x;
        const int t0 = blockIdx.y * kBM32;

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

        // 8 warps quantize 8 owned `down` rows in-place -> MXFP8 (E4M3 + block-32 UE8M0).
        // Lanes 0..19 handle 2 lanes per 32-element block (16 elems/lane).
        {
            const int r_local = tid >> 5;
            const int lane = tid & 31;
            float fv[16];
            float amax16 = 0.0f;
            if (lane < 20) {
                const __nv_bfloat162* src =
                    reinterpret_cast<const __nv_bfloat162*>(sdown_own + r_local * kRank32 + lane * 16);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float2 f = __bfloat1622float2(src[j]);
                    fv[2 * j] = f.x;
                    fv[2 * j + 1] = f.y;
                    amax16 = fmaxf(amax16, fmaxf(fabsf(f.x), fabsf(f.y)));
                }
            }
            const float amax32 = fmaxf(amax16, __shfl_xor_sync(0xffffffffu, amax16, 1));
            uint4 q16 = make_uint4(0u, 0u, 0u, 0u);
            unsigned char ss = 0u;
            if (lane < 20 && amax32 > 0.0f) {
                int e;
                const float m = frexpf(__fmul_rn(amax32, 1.0f / 448.0f), &e);
                int k = (m == 0.5f) ? e - 1 : e;
                k = max(-127, min(127, k));
                const float inv = __int_as_float((127 - k) << 23);
                q16.x = pack_e4m3x4_f32(fv[0] * inv, fv[1] * inv, fv[2] * inv, fv[3] * inv);
                q16.y = pack_e4m3x4_f32(fv[4] * inv, fv[5] * inv, fv[6] * inv, fv[7] * inv);
                q16.z = pack_e4m3x4_f32(fv[8] * inv, fv[9] * inv, fv[10] * inv, fv[11] * inv);
                q16.w = pack_e4m3x4_f32(fv[12] * inv, fv[13] * inv, fv[14] * inv, fv[15] * inv);
                ss = static_cast<unsigned char>(k + 127);
            }
            __syncthreads();
            if (lane < 20) {
                reinterpret_cast<uint4*>(smem + kOffDownFp8)[r_local * 20 + lane] = q16;
                if ((lane & 1) == 0) {
                    (smem + kOffDownSc8)[r_local * 12 + (lane >> 1)] = ss;
                }
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

    // ---- Phase 2 Setup: DSMEM copy MXFP8 A [32,320] + sSA [32,12] + sRstd [32,4] ----
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    unsigned int sbase = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n"
                 : "+r"(sbase));
    {
        const uint4* fp0 = reinterpret_cast<const uint4*>(cluster.map_shared_rank(smem + kOffDownFp8, 0));
        const uint4* fp1 = reinterpret_cast<const uint4*>(cluster.map_shared_rank(smem + kOffDownFp8, 1));
        const uint4* fp2 = reinterpret_cast<const uint4*>(cluster.map_shared_rank(smem + kOffDownFp8, 2));
        const uint4* fp3 = reinterpret_cast<const uint4*>(cluster.map_shared_rank(smem + kOffDownFp8, 3));
        for (int idx = tid; idx < kBM32 * (kRank32 / 16); idx += kThreads32) {
            const int row = idx / (kRank32 / 16);
            const int c = idx - row * (kRank32 / 16);
            const int sr = row >> 3;
            const int rl = row & 7;
            const uint4* src = (sr == 0) ? fp0 : ((sr == 1) ? fp1 : ((sr == 2) ? fp2 : fp3));
            reinterpret_cast<uint4*>(smem + kOffA8)[idx] = src[rl * (kRank32 / 16) + c];
        }
        if (tid < 96) {
            const uint32_t* sc0 = reinterpret_cast<const uint32_t*>(cluster.map_shared_rank(smem + kOffDownSc8, 0));
            const uint32_t* sc1 = reinterpret_cast<const uint32_t*>(cluster.map_shared_rank(smem + kOffDownSc8, 1));
            const uint32_t* sc2 = reinterpret_cast<const uint32_t*>(cluster.map_shared_rank(smem + kOffDownSc8, 2));
            const uint32_t* sc3 = reinterpret_cast<const uint32_t*>(cluster.map_shared_rank(smem + kOffDownSc8, 3));
            const int row = tid / 3;
            const int c = tid - row * 3;
            const int sr = row >> 3;
            const int rl = row & 7;
            const uint32_t* src = (sr == 0) ? sc0 : ((sr == 1) ? sc1 : ((sr == 2) ? sc2 : sc3));
            reinterpret_cast<uint32_t*>(smem + kOffSA8)[tid] = src[rl * 3 + c];
        }
        if (tid < kBM32 * kStreams) {
            const float* rs0 = reinterpret_cast<const float*>(cluster.map_shared_rank(smem + hc_2cta::kOffRstd2, 0));
            const float* rs1 = reinterpret_cast<const float*>(cluster.map_shared_rank(smem + hc_2cta::kOffRstd2, 1));
            const float* rs2 = reinterpret_cast<const float*>(cluster.map_shared_rank(smem + hc_2cta::kOffRstd2, 2));
            const float* rs3 = reinterpret_cast<const float*>(cluster.map_shared_rank(smem + hc_2cta::kOffRstd2, 3));
            const int rrow = tid >> 2;
            const int rst = tid & 3;
            const float* src = (rst == 0) ? rs0 : ((rst == 1) ? rs1 : ((rst == 2) ? rs2 : rs3));
            reinterpret_cast<float*>(smem + kOffRstd8)[rrow * kStreams + rst] = src[rrow];
        }
    }
    cluster.sync();

    // ---- Phase 2: 20 strided H-tiles/rank with W4A8 MXFP8 TC (2 K-halves of 160) ----
    // Register-direct NVFP4 -> W4A8 (E4M3 + UE8M0) load/transcode for one K-half (K=160 = 5x32).
    auto load_transcode_up_half = [&](int ht, int kh) {
        const uint64_t pol_el = policy_evict_last();
        const size_t h0 = (size_t)ht * kBH32;
        const size_t pk_kh_off = (size_t)kh * (size_t)(kRankHalf / 2);
        const size_t sc_kh_off = (size_t)kh * (size_t)kRankGroups8;
#pragma unroll 1
        for (int idx = tid; idx < kStreams * kBH32 * (kRankGroups8 / 2); idx += kThreads32) {
            const int br = idx / (kRankGroups8 / 2);
            const int kb = idx - br * (kRankGroups8 / 2);
            const int bst = br >> 5;
            const int bh = br & 31;
            const size_t row_id = (size_t)bst * kHidden + h0 + (size_t)bh;
            const uint8_t* p_src = up_packed + row_id * (size_t)kRankHalf + pk_kh_off + (size_t)kb * 16u;
            const uint8_t* s_src = up_scales + row_id * (size_t)kRankGroups4 + sc_kh_off + (size_t)kb * 2u;
            const uint2 p_lo = hc_bm64_upmix::ld_u64_ca_hint(p_src, pol_el);
            const uint2 p_hi = hc_bm64_upmix::ld_u64_ca_hint(p_src + 8u, pol_el);
            const unsigned int s0 = hc_bm64_upmix::ld_u8_ca_hint(s_src, pol_el);
            const unsigned int s1 = hc_bm64_upmix::ld_u8_ca_hint(s_src + 1u, pol_el);
            const unsigned int s_max = max(s0 & 0x7fu, s1 & 0x7fu);
            const int e32 = (s_max >= 8u) ? (static_cast<int>(s_max >> 3) - 7)
                          : (s_max != 0u) ? (22 - __clz(static_cast<int>(s_max)))
                                          : 0;
            const unsigned int r0 = norm_e4m3_scale(s0, e32);
            const unsigned int r1 = norm_e4m3_scale(s1, e32);
            const uint2 lut0 = e4m3_lut_ptx(r0);
            const uint2 lut1 = e4m3_lut_ptx(r1);
            const uint2 d0 = decode8_nvfp4_e4m3(p_lo.x, lut0);
            const uint2 d1 = decode8_nvfp4_e4m3(p_lo.y, lut0);
            const uint2 d2 = decode8_nvfp4_e4m3(p_hi.x, lut1);
            const uint2 d3 = decode8_nvfp4_e4m3(p_hi.y, lut1);
            const unsigned int dst_b = sbase + kOffB8 + (unsigned int)(br * kRankHalf + kb * 32);
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                         :: "r"(dst_b), "r"(d0.x), "r"(d0.y), "r"(d1.x), "r"(d1.y) : "memory");
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                         :: "r"(dst_b + 16u), "r"(d2.x), "r"(d2.y), "r"(d3.x), "r"(d3.y) : "memory");
            (smem + kOffSB8)[br * 8 + kb] = static_cast<unsigned char>((e32 + 127) & 0xffu);
        }
    };

    float amax = 0.0f;

#pragma unroll 1
    for (int g = 0; g < 2; ++g) {
        const int warp_id = s + 4 * g;
        if constexpr (kMxfp8Gate) {
            if ((ss_out != nullptr) && (shared_gate != nullptr) && tid >= 128) {
                const int warp_dot = (tid - 128) >> 5;
                float* const s_dot = reinterpret_cast<float*>(smem + kOffDotAcc8) + (warp_dot * 8 * 32) + (tid & 31);
#pragma unroll
                for (int rr = 0; rr < 8; ++rr) s_dot[rr * 32] = 0.0f;
            }
        }

#pragma unroll 1
        for (int ti = 0; ti < 10; ++ti) {
            const int ht = warp_id + 8 * ti;

            float acc[kStreams][4];
#pragma unroll
            for (int st = 0; st < kStreams; ++st)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[st][e] = 0.0f;

#pragma unroll 1
            for (int kh = 0; kh < 2; ++kh) {
                load_transcode_up_half(ht, kh);
                __syncthreads();
                const int warp = tid >> 5;
                const int lane = tid & 31;
                const int wm = warp >> 2;
                const int wh = warp & 3;
                const int group_id = lane >> 2;
                const int tid4 = lane & 3;
                const int fr0 = wm * 16 + group_id;
                const int fr1 = fr0 + 8;
                const int m_sfa = fr0 + 8 * (lane & 1);
                const int a_k_base = kh * kRankHalf;
                const int sa_k_base = kh * (kRankGroups8 / 2);
#pragma unroll 1
                for (int kb = 0; kb < (kRankGroups8 / 2); ++kb) {
                    const int b_byte = kb * 32 + tid4 * 4;
                    const int a_byte = a_k_base + b_byte;
                    const unsigned int a0 = *reinterpret_cast<const unsigned int*>(smem + kOffA8 + fr0 * kRank32 + a_byte);
                    const unsigned int a1 = *reinterpret_cast<const unsigned int*>(smem + kOffA8 + fr1 * kRank32 + a_byte);
                    const unsigned int a2 = *reinterpret_cast<const unsigned int*>(smem + kOffA8 + fr0 * kRank32 + a_byte + 16);
                    const unsigned int a3 = *reinterpret_cast<const unsigned int*>(smem + kOffA8 + fr1 * kRank32 + a_byte + 16);
#if Q4_HC_HAS_SM120_MX_MMA
                    const unsigned int sfa = static_cast<unsigned int>((smem + kOffSA8)[m_sfa * 12 + sa_k_base + kb]);
#else
                    const unsigned int sfa =
                        static_cast<unsigned int>((smem + kOffSA8)[fr0 * 12 + sa_k_base + kb]) |
                        (static_cast<unsigned int>((smem + kOffSA8)[fr1 * 12 + sa_k_base + kb]) << 8);
#endif
#pragma unroll
                    for (int st = 0; st < kStreams; ++st) {
                        const int nc = st * 32 + wh * 8 + group_id;
                        const unsigned int b0 = *reinterpret_cast<const unsigned int*>(smem + kOffB8 + nc * kRankHalf + b_byte);
                        const unsigned int b1 = *reinterpret_cast<const unsigned int*>(smem + kOffB8 + nc * kRankHalf + b_byte + 16);
                        const unsigned int sfb = static_cast<unsigned int>((smem + kOffSB8)[nc * 8 + kb]);
                        mma_mxfp8_m16n8k32(acc[st], a0, a1, a2, a3, b0, b1, sfa, sfb);
                    }
                }
                __syncthreads();
            }

            // Issue sH cp.async overlapped with norm_w loads and MUFU sigmoids.
            const int t0 = blockIdx.y * kBM32;
            const int h0 = ht * kBH32;
            const bool full_tile = ((unsigned int)(t0 + kBM32) <= num_tokens);
            const bool do_gate_dot = kMxfp8Gate && (ss_out != nullptr) && (shared_gate != nullptr);
            const int warp = tid >> 5;
            const int lane = tid & 31;
            const int wm = warp >> 2;
            const int wh = warp & 3;
            const int group_id = lane >> 2;
            const int tid4 = lane & 3;
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
                    const unsigned int dst = sbase + kOffH8 + hc_bm64_upmix::swz_h2(row, st, piece);
                    const __nv_bfloat16* src = hyper + (size_t)(ok ? t : 0) * kWidth + st * kHidden + h0 + piece * 8;
                    hc_bm64_upmix::cp16_hint_u32(dst, src, ok, pol_ef_h);
                }
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

            asm volatile("cp.async.wait_group 0;\n\tbar.sync 0;\n"
                         : "+f"(acc[kStreams - 1][3]) :: "memory");

            const unsigned int in_piece = (unsigned int)(tid4 * 4);
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = wm * 16 + group_id + half * 8;
                const bool ok_row = full_tile || (t0 + row < (int)num_tokens);
                const unsigned int o_addr = sbase + kOffO8 + hc_bm64_upmix::swz_o2(row, wh) + in_piece;
                if (ok_row) {
                    const float4 r4 = *reinterpret_cast<const float4*>(smem + kOffRstd8 + row * 16);
                    const float r[kStreams] = {r4.x, r4.y, r4.z, r4.w};
                    __nv_bfloat162 h2[kStreams];
#pragma unroll
                    for (int st = 0; st < kStreams; ++st) {
                        unsigned int raw_h;
                        asm volatile("ld.shared.u32 %0, [%1];\n"
                                     : "=r"(raw_h)
                                     : "r"(sbase + kOffH8 + hc_bm64_upmix::swz_h2(row, st, wh) + in_piece));
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
                    const unsigned int o_addr = sbase + kOffO8 + hc_bm64_upmix::swz_o2(row, piece);
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
                    float* const s_dot = reinterpret_cast<float*>(smem + kOffDotAcc8) + (warp_dot * 8 * 32) + lane;
#pragma unroll
                    for (int rr = 0; rr < 8; ++rr) {
                        const int row = warp_dot * 8 + rr;
                        const unsigned int o_addr =
                            sbase + kOffO8 + hc_bm64_upmix::swz_o2(row, piece) + byte_in_piece;
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
                        sbase + kOffO8 + hc_bm64_upmix::swz_o2(row, piece) + (unsigned int)(sub * 8);
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
            if ((ss_out != nullptr) && (shared_gate != nullptr) && tid >= 128) {
                const int lane = tid & 31;
                float* const s_warp_dot = reinterpret_cast<float*>(smem + kOffWarpDot8);
                const int warp_dot = (tid - 128) >> 5;
                const float* const s_dot = reinterpret_cast<const float*>(smem + kOffDotAcc8) + (warp_dot * 8 * 32) + lane;
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
        const int warp = tid >> 5;
        const int lane = tid & 31;
        __syncthreads();
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
        }
        float* const s_max = reinterpret_cast<float*>(smem + kOffO8);
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
            const int t0 = blockIdx.y * kBM32;
            if (shared_gate == nullptr) {
                if (tid < kRowsPerRank) {
                    const int t = t0 + s * kRowsPerRank + tid;
                    if (t < (int)num_tokens) ss_out[t] = 1.0f;
                }
            } else {
                float* const s_warp_dot = reinterpret_cast<float*>(smem + kOffWarpDot8);
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
__device__ __forceinline__ void combine_inject_upmix_bm32_mxfp8_body(
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

    inject_upmix_bm32_mxfp8_body<kAmax, false, true, true, false>(
        smem, hyper, moe_rows, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        amax_bits, nullptr, nullptr, nullptr, nullptr);
}

}  // namespace hc_bm32_mxfp8

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_bm32_mxfp8::kThreads32, 2)
qwen4_hc_inject_upmix_bm32_mxfp8_2cta(QWEN4_HC_BM64_UPMIX_W4_PARAMS) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core == nullptr) __trap();
    hc_bm32_mxfp8::check_bm32_mxfp8_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm32_mxfp8::inject_upmix_bm32_mxfp8_body<false, false>(
        smem, hyper, core, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        nullptr, nullptr, nullptr, nullptr, nullptr);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_bm32_mxfp8::kThreads32, 2)
qwen4_hc_inject_upmix_bm32_mxfp8_2cta_amax(
    QWEN4_HC_BM64_UPMIX_W4_PARAMS,
    unsigned int* __restrict__ amax_bits) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core == nullptr) __trap();
    hc_bm32_mxfp8::check_bm32_mxfp8_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm32_mxfp8::inject_upmix_bm32_mxfp8_body<true, false>(
        smem, hyper, core, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        amax_bits, nullptr, nullptr, nullptr, nullptr);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_bm32_mxfp8::kThreads32, 2)
qwen4_hc_inject_upmix_bm32_mxfp8_2cta_mxfp8_gate(
    QWEN4_HC_BM64_UPMIX_W4_PARAMS,
    unsigned char* __restrict__ xq,
    unsigned char* __restrict__ xs,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core == nullptr) __trap();
    hc_bm32_mxfp8::check_bm32_mxfp8_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm32_mxfp8::inject_upmix_bm32_mxfp8_body<false, true>(
        smem, hyper, core, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        nullptr, xq, xs, shared_gate, ss_out);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_bm32_mxfp8::kThreads32, 2)
qwen4_hc_inject_upmix_bm32_mxfp8_2cta_mxfp8_gate_amax(
    QWEN4_HC_BM64_UPMIX_W4_PARAMS,
    unsigned int* __restrict__ amax_bits,
    unsigned char* __restrict__ xq,
    unsigned char* __restrict__ xs,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core == nullptr) __trap();
    hc_bm32_mxfp8::check_bm32_mxfp8_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm32_mxfp8::inject_upmix_bm32_mxfp8_body<true, true>(
        smem, hyper, core, residual, folded, up_packed, up_scales, norm_w, mixed,
        num_tokens, scale2_down, scale2_inject, alpha_up, eps,
        amax_bits, xq, xs, shared_gate, ss_out);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_bm32_mxfp8::kThreads32, 2)
qwen4_hc_combine_inject_upmix_bm32_mxfp8_2cta_mx_gatepre(QWEN4_HC_BM64_COMBINE_UPMIX_W4_PARAMS) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || topk != kTopk ||
        moe_rows == nullptr || expert_output == nullptr || slot_scales == nullptr ||
        ((reinterpret_cast<uintptr_t>(slot_scales) & 3u) != 0u) ||
        token_to_perm == nullptr || topk_weights == nullptr || shared_out == nullptr ||
        gate_scalars == nullptr) {
        __trap();
    }
    hc_bm32_mxfp8::check_bm32_mxfp8_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm32_mxfp8::combine_inject_upmix_bm32_mxfp8_body<false>(
        smem, hyper, residual, moe_rows, expert_output, slot_scales, token_to_perm, topk_weights,
        shared_out, gate_scalars, folded, up_packed, up_scales, norm_w, mixed, num_tokens,
        scale2_down, scale2_inject, alpha_up, eps, nullptr);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_bm32_mxfp8::kThreads32, 2)
qwen4_hc_combine_inject_upmix_bm32_mxfp8_2cta_mx_gatepre_amax(
    QWEN4_HC_BM64_COMBINE_UPMIX_W4_PARAMS,
    unsigned int* __restrict__ amax_bits) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || topk != kTopk ||
        moe_rows == nullptr || expert_output == nullptr || slot_scales == nullptr ||
        ((reinterpret_cast<uintptr_t>(slot_scales) & 3u) != 0u) ||
        token_to_perm == nullptr || topk_weights == nullptr || shared_out == nullptr ||
        gate_scalars == nullptr || amax_bits == nullptr) {
        __trap();
    }
    hc_bm32_mxfp8::check_bm32_mxfp8_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    hc_bm32_mxfp8::combine_inject_upmix_bm32_mxfp8_body<true>(
        smem, hyper, residual, moe_rows, expert_output, slot_scales, token_to_perm, topk_weights,
        shared_out, gate_scalars, folded, up_packed, up_scales, norm_w, mixed, num_tokens,
        scale2_down, scale2_inject, alpha_up, eps, amax_bits);
}




