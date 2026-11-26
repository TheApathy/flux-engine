// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// Inline NVFP4 folded strip-mined mHC kernels (BM=64, 83,200 B SMEM).

// ============================================================================
// Inline NVFP4 `folded` Strip-Mined mHC Kernels (`BM=64`, 512 threads,
// 4-CTA cluster, `kSmemBytesW4 = 83,200 B`, leaving ~45 KiB L1/SM):
//   - Eliminates `qwen4_hc_prep_weights` + 6.33 MiB BF16 `s.folded` writes/reads;
//     reads packed NVFP4 `down`+`inject` + FP8 scales + `norm_w` (`1.80 MiB`
//     total) via `ld.global.ca.L2::cache_hint` (`L2::evict_last`) directly into
//     2-stage `sB`, fitting `hyper'[768]` (15.0 MiB) + NVFP4 `folded` (1.80 MiB)
//     + NVFP4 `up` (1.76 MiB) + `down`/`rstd` (0.48 MiB) <= 24.0 MiB L2!
//   - Strict 0-ULP bit-identical to `qwen4_hc_prep_weights` + BF16 `inject_gemm`
//     / `combine_moe_gemm`: matches `bf16(bf16(e2m1 * s_fp8) * (1 + norm_w))`
//     exact rounding on every finite `(E4M3, E2M1)` pair.
// ============================================================================
namespace hc_w4 {

constexpr int kStagesBW4 = 2;
constexpr int kSmemBW4 = kStagesBW4 * kNPad * kLds * 2;  // 2 * 352 * 40 * 2 = 56,320 B
constexpr int kSmemRingW4 = (kSmemBW4 + kSmemA) > kSmemQ ? (kSmemBW4 + kSmemA) : kSmemQ;  // 82,944 B
constexpr int kOffRstdW4 = kSmemRingW4;                  // 82,944 B
constexpr int kSmemBytesW4 = kOffRstdW4 + kBM * 4;       // 83,200 B
static_assert(kSmemBytesW4 == 83200, "W4 strip launcher passes 83200 bytes of dynamic smem");

struct FoldedW4 {
    const uint8_t* down_packed;    // [320, 5120]
    const uint8_t* down_scales;    // [320, 640]
    const uint8_t* inject_packed;  // [4, 5120]
    const uint8_t* inject_scales;  // [4, 640]
    const __nv_bfloat16* norm_w;   // [10240]
};

__device__ __forceinline__ uint32_t ld_u32_ca_keep(const void* p) {
    uint32_t v;
    asm volatile(
        "{ .reg .b64 pol;\n"
        "  createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n"
        "  ld.global.ca.L2::cache_hint.u32 %0, [%1], pol; }\n"
        : "=r"(v)
        : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t ld_u8_ca_keep(const void* p) {
    uint32_t v;
    asm volatile(
        "{ .reg .b64 pol;\n"
        "  createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n"
        "  ld.global.ca.L2::cache_hint.u8 %0, [%1], pol; }\n"
        : "=r"(v)
        : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 ld_u128_ca_keep(const void* p) {
    uint4 v;
    asm volatile(
        "{ .reg .b64 pol;\n"
        "  createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n"
        "  ld.global.ca.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], pol; }\n"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
    return v;
}

__device__ __forceinline__ void check_w4_smem() {
    unsigned int dyn;
    asm("mov.u32 %0, %%dynamic_smem_size;\n" : "=r"(dyn));
    if (dyn < (unsigned int)kSmemBytesW4) __trap();
}

// Bit-exact (0-ULP) NVFP4 E2M1x8 * FP8_E4M3 scale * (1 + norm_w) -> BF16x8 written to shared `dst_s`.
// Matches `qwen4_hc_prep_weights`: `bf16(float(bf16(lut[nib] * float(fp8))) * (1.0f + float(norm_w)))`.
__device__ __forceinline__ void decode8_fold_bf16_store(
    unsigned int dst_s, uint32_t p4, uint32_t s_byte, uint4 nw_raw) {
    const unsigned int nw_w[4] = {nw_raw.x, nw_raw.y, nw_raw.z, nw_raw.w};
#if Q4_HC_HAS_E2M1_CVT
    uint32_t s_f16x2;
    const unsigned short sb2 = static_cast<unsigned short>((s_byte & 0xffu) | ((s_byte & 0xffu) << 8));
    asm("{ .reg .b16 hs; mov.b16 hs, %1; cvt.rn.f16x2.e4m3x2 %0, hs; }" : "=r"(s_f16x2) : "h"(sb2));
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        uint32_t hb;
        const unsigned short byte_b = static_cast<unsigned short>((p4 >> (b * 8)) & 0xffu);
        asm("{ .reg .b8 bx, bz; mov.b16 {bx, bz}, %1; cvt.rn.f16x2.e2m1x2 %0, bx; }"
            : "=r"(hb) : "h"(byte_b));
        const __half2 prod = __hmul2(*reinterpret_cast<const __half2*>(&hb),
                                     *reinterpret_cast<const __half2*>(&s_f16x2));
        const float2 w = __half22float2(prod);
        const float2 nwf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&nw_w[b]));
        const __nv_bfloat162 f01 = __floats2bfloat162_rn(w.x * (1.0f + nwf.x), w.y * (1.0f + nwf.y));
        const uint32_t out_w = *reinterpret_cast<const uint32_t*>(&f01);
        asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dst_s + (unsigned int)(b * 4)), "r"(out_w) : "memory");
    }
#else
    __nv_fp8_e4m3 fp8;
    *reinterpret_cast<unsigned char*>(&fp8) = static_cast<unsigned char>(s_byte);
    const __nv_bfloat162 s2 = __bfloat162bfloat162(__float2bfloat16((float)fp8));
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
    const __nv_bfloat162 r[4] = {
        __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&w0), s2),
        __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&w1), s2),
        __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&w2), s2),
        __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&w3), s2),
    };
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const float2 w = __bfloat1622float2(r[b]);
        const float2 nwf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&nw_w[b]));
        const __nv_bfloat162 f01 = __floats2bfloat162_rn(w.x * (1.0f + nwf.x), w.y * (1.0f + nwf.y));
        const uint32_t out_w = *reinterpret_cast<const uint32_t*>(&f01);
        asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dst_s + (unsigned int)(b * 4)), "r"(out_w) : "memory");
    }
#endif
}

__device__ __forceinline__ void ldsm_x4_u32(unsigned int (&r)[4], unsigned int s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

// Warps 8-15: inline NVFP4 `down|inject` + `norm_w` fold into 2-stage sB and
// every MMA of the 64x324 tile. Synchronizes with producer warps 0-7 via the
// single `pipe_barrier()` (`bar.sync 1, 512`) per K step.
// Keeps weight base pointers in constant-bank params (not live across the 88-reg
// `acc` MMA loop) and uses 32-bit shared offsets (`sbase`) for 0-spill <=128 regs.
__device__ __forceinline__ void multiply_rows_w4(
    unsigned int sbase, const FoldedW4& fw, int s, float (&acc)[kFrags][4]) {
    const int mt = threadIdx.x - kProducers;

    // Zero padding rows [324..351] once across both sB stages.
    {
        const int n0 = mt >> 2;
        const int part8 = (mt & 3) * 8;
        if (n0 >= kStreams && (n0 + kRankBase) < kNPad) {
            const unsigned int z0 = sbase + (unsigned int)(((n0 + kRankBase) * kLds + part8) * 2);
            const unsigned int z1 = z0 + (unsigned int)(kNPad * kLds * 2);
            asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};\n" :: "r"(z0) : "memory");
            asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};\n" :: "r"(z1) : "memory");
        }
    }

    auto load_b_w4 = [&](int q, int bslot) {
        const int n0 = mt >> 2;
        const int part8 = (mt & 3) * 8;
        const uint4 nw_raw = ld_u128_ca_keep(fw.norm_w + (size_t)s * kHidden + q * kBK + part8);
        const unsigned int p_col = (unsigned int)s * (kHidden / 2) + (unsigned int)q * (kBK / 2) + (part8 >> 1);
        const unsigned int s_col = (unsigned int)s * (kHidden / 16) + (unsigned int)q * (kBK / 16) + (part8 >> 4);
        const unsigned int dst_base = sbase + (unsigned int)(bslot * (kNPad * kLds * 2) + (n0 * kLds + part8) * 2);
        const int j_end = (n0 < kStreams) ? 6 : 5;
#pragma unroll 1
        for (int j = 0; j < j_end; ++j) {
            const bool inj = (j == 5);
            const unsigned int r_idx = inj ? (unsigned int)n0 : (unsigned int)(n0 + j * 64);
            const uint8_t* pk = inj ? fw.inject_packed : fw.down_packed;
            const uint8_t* sc_ptr = inj ? fw.inject_scales : fw.down_scales;
            const uint32_t p4 = ld_u32_ca_keep(pk + (size_t)r_idx * (kWidth / 2) + p_col);
            const uint32_t sc = ld_u8_ca_keep(sc_ptr + (size_t)r_idx * (kWidth / 16) + s_col);
            decode8_fold_bf16_store(dst_base + (unsigned int)(j * 64 * kLds * 2), p4, sc, nw_raw);
        }
    };

    load_b_w4(0, 0);

#pragma unroll
    for (int f = 0; f < kFrags; ++f)
#pragma unroll
        for (int x = 0; x < 4; ++x) acc[f][x] = 0.0f;

    static_assert(kSteps == 80 && kStagesBW4 == 2, "2-stage ring assumes kSteps=80, kStagesBW4=2");
#pragma unroll 1
    for (int q = 0; q < kSteps; ++q) {
        const int buf = q & 1;
        pipe_barrier();  // step q's sA[buf] and sB[buf] are ready; (buf ^ 1) is free
        const int mw = mt >> 5;
        const int lane = mt & 31;
        const int wm = mw >> 1;  // 16-row slab
        const int wn = mw & 1;   // 176-output half
        const unsigned int sa_buf = sbase + (unsigned int)(kSmemBW4 + buf * (kBM * kLds * 2));
        const unsigned int sb_buf = sbase + (unsigned int)(buf * (kNPad * kLds * 2));
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 16) {
            unsigned int a[4];
            const unsigned int a_row = (unsigned int)(wm * 16 + (lane & 15));
            const unsigned int a_col = (unsigned int)(kk + (lane >> 4) * 8);
            ldsm_x4_u32(a, sa_buf + (a_row * kLds + a_col) * 2u);
#pragma unroll
            for (int f = 0; f < kFrags; f += 2) {
                const unsigned int n = (unsigned int)(wn * (kFrags * 8) + f * 8 + (lane & 7) + ((lane >> 4) << 3));
                const unsigned int b_col = (unsigned int)(kk + ((lane >> 3) & 1) * 8);
                unsigned int r[4];
                ldsm_x4_u32(r, sb_buf + (n * kLds + b_col) * 2u);
                mma_bf16(acc[f], a, r[0], r[1]);
                mma_bf16(acc[f + 1], a, r[2], r[3]);
            }
        }
        if (q + 1 < kSteps) {
            load_b_w4(q + 1, buf ^ 1);
        }
    }
}

template <bool kL2Hint, bool kPreloadedH = false, typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void prepare_tile_w4(
    unsigned char* smem, const Operands& op, const FoldedW4& fw, const Outputs& out,
    unsigned int num_tokens, float eps, float pre_scale = 0.0f, const Vec8* pre_rh = nullptr) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    const int s = blockIdx.x;
    const int t0 = blockIdx.y * kBM;
    const int tid = threadIdx.x;
    const unsigned int sbase = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    float* srstd = reinterpret_cast<float*>(smem + kOffRstdW4);
    float acc[kFrags][4];
    if (tid < kProducers) {
        const float r = produce_rows<kL2Hint, true, kPreloadedH, kSmemBW4>(
            smem, op, s, t0, num_tokens, kHidden, eps, pre_scale, pre_rh);
        if ((tid & 3) == 0) {
            const int row = tid >> 2;
            srstd[row] = r;
            if (t0 + row < (int)num_tokens) out.rstd[(size_t)(t0 + row) * kStreams + s] = r;
        }
    } else {
        multiply_rows_w4(sbase, fw, s, acc);
    }
    __syncthreads();  // operand ring idle, srstd visible
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
                make_float2(rs0 * acc[f][0], rs0 * acc[f][1]);
            *reinterpret_cast<float2*>(&sq[row1 * kNTotal + n]) =
                make_float2(rs1 * acc[f][2], rs1 * acc[f][3]);
        }
    }
    cluster.sync();  // every stream's rstd * P tile is complete
    const float2* const sq2 = reinterpret_cast<const float2*>(sq);
    const float2* const rq1 =
        reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 1) & 3));
    const float2* const rq2 =
        reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 2) & 3));
    const float2* const rq3 =
        reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 3) & 3));
    const unsigned long long rw = (unsigned long long)kHidden * out.hc;
    constexpr int kRowsPerRank = kBM / kStreams;                // 16
    constexpr int kPairsPerRow = kNTotal / 2;                   // 162
    constexpr int kPairsPerRank = kRowsPerRank * kPairsPerRow;  // 2592
    for (int idx = tid; idx < kPairsPerRank; idx += kThreads) {
        const int r_local = idx / kPairsPerRow;
        const int p = idx - r_local * kPairsPerRow;
        const int row = s * kRowsPerRank + r_local;
        const int t = t0 + row;
        if (t >= (int)num_tokens) continue;
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
        if (n < out.rank) {
            const float x0 = (out.scale2_down * sum0) * (1.0f / (float)kStreams);
            const float x1 = (out.scale2_down * sum1) * (1.0f / (float)kStreams);
            const float v0 = hc_fast_div_rn_1p_exp(x0);
            const float v1 = hc_fast_div_rn_1p_exp(x1);
            *reinterpret_cast<__nv_bfloat162*>(&out.down[(unsigned long long)t * out.rank + n]) =
                __floats2bfloat162_rn(v0, v1);
        } else {
            const float x0 = (out.scale2_inject * sum0) * (1.0f / (float)kStreams);
            const float x1 = (out.scale2_inject * sum1) * (1.0f / (float)kStreams);
            const float v0 = 2.0f * hc_fast_rcp_rn_1p_exp(x0);
            const float v1 = 2.0f * hc_fast_rcp_rn_1p_exp(x1);
            *reinterpret_cast<__nv_bfloat162*>(
                &out.residual[(unsigned long long)t * rw + (rw - out.hc) + (n - out.rank)]) =
                __floats2bfloat162_rn(v0, v1);
        }
    }
    cluster.sync();  // no CTA leaves while another reads its tile
}

__device__ __forceinline__ int cta_row0_w4() {
    int bx, by;
    asm volatile("mov.u32 %0, %%ctaid.x;\n\tmov.u32 %1, %%ctaid.y;\n" : "=r"(bx), "=r"(by));
    return by * kBM + bx * kMoeRows;
}

template <bool kMx, typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void combine_moe_tile_gatepre_w4(
    unsigned char* smem, const MoeRows& m, const float* gate_scalars,
    __nv_bfloat16* hyper, __nv_bfloat16* residual, const FoldedW4& fw,
    const Outputs& out, unsigned int num_tokens, float eps) {
    // BF16 prologue scratch (1,344 B) sits at [0..1344) before prepare_tile_w4 touches sB;
    // MXFP8 prologue (`moe_form_rows_mx16`) uses 0 SMEM.
    const MoeScratch sc{reinterpret_cast<int*>(smem),
                        reinterpret_cast<float*>(smem + kMoeRows * kTopk * 4),
                        reinterpret_cast<float*>(smem + 2 * kMoeRows * kTopk * 4),
                        nullptr};
    moe_prologue_gatepre<kMx>(m, gate_scalars, sc, cta_row0_w4(), num_tokens);
    const Operands op{hyper, m.out, residual, nullptr, nullptr};
    if constexpr (kMx) {
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
            const uint64_t stream_pol = policy_evict_first();
            const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int stg = 0; stg < 4; ++stg) {
                pre_rh[stg].raw = prow ? load_row16_stream<true>(hrow + stg * kBK, stream_pol) : zero;
            }
        }
        cooperative_groups::this_cluster().sync();
        prepare_tile_w4<true, true>(smem, op, fw, out, num_tokens, eps, pre_scale, pre_rh);
    } else {
        cooperative_groups::this_cluster().sync();
        prepare_tile_w4<true, false>(smem, op, fw, out, num_tokens, eps);
    }
    const int row0_discard = cta_row0_w4();
    const int valid_rows = min(kMoeRows, max(0, (int)num_tokens - row0_discard));
    unsigned char* base_discard = reinterpret_cast<unsigned char*>(m.out + (size_t)row0_discard * kHidden);
    const unsigned char* sh_discard =
        reinterpret_cast<const unsigned char*>(m.shared_out + (size_t)row0_discard * kHidden);
    const bool disc_out = ((reinterpret_cast<uintptr_t>(m.out) & 127u) == 0u);
    const bool disc_sh = (m.shared_out != nullptr) && ((reinterpret_cast<uintptr_t>(m.shared_out) & 127u) == 0u);
    const int total_lines = valid_rows * (kHidden * (int)sizeof(__nv_bfloat16) / 128);
    for (int line = threadIdx.x; line < total_lines; line += kThreads) {
        if (disc_out) {
            unsigned char* p = base_discard + (size_t)line * 128;
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
        }
        if (disc_sh) {
            const unsigned char* ps = sh_discard + (size_t)line * 128;
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(ps) : "memory");
        }
    }
}

}  // namespace hc_w4

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_gemm_ws_gatepre_w4(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    __nv_bfloat16* residual,
    const uint8_t* __restrict__ down_packed,
    const uint8_t* __restrict__ down_scales,
    const uint8_t* __restrict__ inject_packed,
    const uint8_t* __restrict__ inject_scales,
    const __nv_bfloat16* __restrict__ norm_w,
    float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ down,
    unsigned int num_tokens,
    unsigned int rank,
    unsigned int hidden_size,
    unsigned int hc,
    float scale2_down,
    float scale2_inject,
    float eps) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal) __trap();
    hc_w4::check_w4_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    const Operands op{hyper, core, residual, nullptr, nullptr};
    const hc_w4::FoldedW4 fw{down_packed, down_scales, inject_packed, inject_scales, norm_w};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    hc_w4::prepare_tile_w4<false>(smem, op, fw, out, num_tokens, eps);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_gemm_ws_l2h_gatepre_w4(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    __nv_bfloat16* residual,
    const uint8_t* __restrict__ down_packed,
    const uint8_t* __restrict__ down_scales,
    const uint8_t* __restrict__ inject_packed,
    const uint8_t* __restrict__ inject_scales,
    const __nv_bfloat16* __restrict__ norm_w,
    float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ down,
    unsigned int num_tokens,
    unsigned int rank,
    unsigned int hidden_size,
    unsigned int hc,
    float scale2_down,
    float scale2_inject,
    float eps) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal) __trap();
    hc_w4::check_w4_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    const Operands op{hyper, core, residual, nullptr, nullptr};
    const hc_w4::FoldedW4 fw{down_packed, down_scales, inject_packed, inject_scales, norm_w};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    hc_w4::prepare_tile_w4<true>(smem, op, fw, out, num_tokens, eps);
    if (core != nullptr && ((reinterpret_cast<uintptr_t>(core) & 127u) == 0u)) {
        const int row0_discard = (int)(blockIdx.y * kBM + blockIdx.x * kMoeRows);
        const int valid_rows = min(kMoeRows, max(0, (int)num_tokens - row0_discard));
        const unsigned char* base_discard =
            reinterpret_cast<const unsigned char*>(core + (size_t)row0_discard * kHidden);
        const int total_lines = valid_rows * (kHidden * (int)sizeof(__nv_bfloat16) / 128);
        for (int line = threadIdx.x; line < total_lines; line += kThreads) {
            const unsigned char* p = base_discard + (size_t)line * 128;
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
        }
    }
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_combine_moe_gemm_gatepre_w4(
    __nv_bfloat16* __restrict__ hyper,
    __nv_bfloat16* residual,
    const uint8_t* __restrict__ down_packed,
    const uint8_t* __restrict__ down_scales,
    const uint8_t* __restrict__ inject_packed,
    const uint8_t* __restrict__ inject_scales,
    const __nv_bfloat16* __restrict__ norm_w,
    float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ down,
    __nv_bfloat16* moe_rows,
    const __nv_bfloat16* __restrict__ expert_output,
    const uint8_t* __restrict__ slot_scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int num_tokens,
    unsigned int rank,
    unsigned int hidden_size,
    unsigned int hc,
    unsigned int topk,
    float scale2_down,
    float scale2_inject,
    float eps) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || topk != kTopk) __trap();
    hc_w4::check_w4_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    const MoeRows m{expert_output, slot_scales, token_to_perm, topk_weights, shared_out, nullptr, nullptr, moe_rows};
    const hc_w4::FoldedW4 fw{down_packed, down_scales, inject_packed, inject_scales, norm_w};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    hc_w4::combine_moe_tile_gatepre_w4<false>(smem, m, gate_scalars, hyper, residual, fw, out, num_tokens, eps);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_combine_moe_gemm_mx_gatepre_w4(
    __nv_bfloat16* __restrict__ hyper,
    __nv_bfloat16* residual,
    const uint8_t* __restrict__ down_packed,
    const uint8_t* __restrict__ down_scales,
    const uint8_t* __restrict__ inject_packed,
    const uint8_t* __restrict__ inject_scales,
    const __nv_bfloat16* __restrict__ norm_w,
    float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ down,
    __nv_bfloat16* moe_rows,
    const uint8_t* __restrict__ expert_output,
    const uint8_t* __restrict__ slot_scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int num_tokens,
    unsigned int rank,
    unsigned int hidden_size,
    unsigned int hc,
    unsigned int topk,
    float scale2_down,
    float scale2_inject,
    float eps) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || topk != kTopk ||
        slot_scales == nullptr || ((reinterpret_cast<uintptr_t>(slot_scales) & 3u) != 0u)) {
        __trap();
    }
    hc_w4::check_w4_smem();
    extern __shared__ __align__(16) unsigned char smem[];
    const MoeRows m{expert_output, slot_scales, token_to_perm, topk_weights, shared_out, nullptr, nullptr, moe_rows};
    const hc_w4::FoldedW4 fw{down_packed, down_scales, inject_packed, inject_scales, norm_w};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    hc_w4::combine_moe_tile_gatepre_w4<true>(smem, m, gate_scalars, hyper, residual, fw, out, num_tokens, eps);
}

