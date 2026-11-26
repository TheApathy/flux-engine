// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// 2-CTA/SM (47,232 B SMEM, BM=32) cluster inject/combine GEMM kernels.

namespace hc_2cta {

template <typename Shape = Qwen4FlashNextShape>
struct Hc2CtaLayout {
    static constexpr int kBM2 = 32;
    static constexpr int kThreads2 = 256;
    static constexpr int kProd2 = 128;
    static constexpr int kFrags2 = (Shape::kMhcProjN + 15) / 16;
    static constexpr int kNPad2 = 2 * kFrags2 * 8;
    static constexpr int kStagesB2 = 2;
    static constexpr int kRowBytes2 = 64;
    static constexpr int kSmemB2 = kStagesB2 * kNPad2 * kRowBytes2;
    static constexpr int kSmemA2 = 2 * kBM2 * kRowBytes2;
    static constexpr int kOffRstd2 = kSmemB2 + kSmemA2;
    static constexpr int kSmemBytes2 = kOffRstd2 + kBM2 * 4;
    static constexpr int kMoeRows2 = kBM2 / Shape::kMhcStreams;
};

using DefaultLayout2 = Hc2CtaLayout<Qwen4FlashNextShape>;
constexpr int kBM2 = DefaultLayout2::kBM2;
constexpr int kThreads2 = DefaultLayout2::kThreads2;
constexpr int kProd2 = DefaultLayout2::kProd2;
constexpr int kFrags2 = DefaultLayout2::kFrags2;
constexpr int kNPad2 = DefaultLayout2::kNPad2;
constexpr int kStagesB2 = DefaultLayout2::kStagesB2;
constexpr int kRowBytes2 = DefaultLayout2::kRowBytes2;
constexpr int kSmemB2 = DefaultLayout2::kSmemB2;
constexpr int kSmemA2 = DefaultLayout2::kSmemA2;
constexpr int kOffRstd2 = DefaultLayout2::kOffRstd2;
constexpr int kSmemBytes2 = DefaultLayout2::kSmemBytes2;
constexpr int kMoeRows2 = DefaultLayout2::kMoeRows2;
static_assert(kSmemBytes2 == 47232, "2-CTA/SM PostPreMx dynamic SMEM budget");

__device__ __forceinline__ unsigned int swz64(int row, int col_bf16) {
    const int chunk = col_bf16 >> 3;
    return static_cast<unsigned int>(row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4));
}

__device__ __forceinline__ void pipe_barrier_256() { asm volatile("bar.sync 1, 256;\n" ::: "memory"); }

__device__ __forceinline__ uint4 ld16_ef_plain(const __nv_bfloat16* p, uint64_t pol) {
    uint4 v;
    asm volatile("ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p), "l"(pol));
    return v;
}

__device__ __forceinline__ void cp16_keep_u32(unsigned int s, const void* gmem, bool pred, uint64_t pol) {
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                 :: "r"(s), "l"(gmem), "r"(bytes), "l"(pol));
}

template <bool kL2Hint, bool kPreloaded = false, bool kAlwaysInject = false>
__device__ __forceinline__ float produce_rows_2cta(
    unsigned char* smem, const Operands& op, int s, int t0,
    unsigned int num_tokens, float eps, float pre_scale = 0.0f, const Vec8* pre_rh = nullptr
) {
    const unsigned int offA = kSmemB2;
    const int pr = threadIdx.x >> 2;
    const int pv = threadIdx.x & 3;
    const int ptoken = t0 + pr;
    const bool prow = ptoken < (int)num_tokens;
    const size_t row_idx = prow ? (size_t)ptoken : 0;
    const bool inject = kAlwaysInject || (op.core != nullptr);
    __nv_bfloat16* hrow = op.hyper + row_idx * kWidth + s * kHidden + pv * 8;
    const float scale = kPreloaded
        ? pre_scale
        : (prow && inject ? __bfloat162float(op.residual[row_idx * kWidth + (kWidth - kStreams) + s]) : 0.0f);
    const __nv_bfloat16* crow = inject ? (op.core + row_idx * kHidden + pv * 8) : nullptr;

    float sums[kSlots];
#pragma unroll
    for (int j = 0; j < kSlots; ++j) sums[j] = 0.0f;
    const bool keep_hyper = (num_tokens <= kKeepHyperMaxSmallTokens) || (t0 + kKeepHyperWindow >= (int)num_tokens);
    const uint64_t pol_ef = policy_evict_first();
    const uint64_t pol_el = policy_evict_last();

    Vec8 rh[4], rc[4];
    auto load_stage = [&](int q, int stg) {
        const uint4 zero = make_uint4(0, 0, 0, 0);
        if constexpr (kAlwaysInject) {
            rh[stg].raw = prow ? ld16_ef_plain(hrow + q * kBK, pol_ef) : zero;
        } else {
            const uint64_t h_load_pol = (!inject && keep_hyper) ? pol_el : pol_ef;
            rh[stg].raw = prow ? ld16_ef_plain(hrow + q * kBK, h_load_pol) : zero;
        }
        rc[stg].raw = prow && inject ? load_row16<kL2Hint>(crow + q * kBK) : zero;
    };
    auto produce_stage = [&](int q, int slot, int buf, int stg) {
        Vec8 out;
        if (inject) {
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
                if (keep_hyper) store_row16_stream(hrow + q * kBK, out.raw, pol_el);
                else store_row16_stream(hrow + q * kBK, out.raw, pol_ef);
            }
        } else {
            out.raw = rh[stg].raw;
        }
        *reinterpret_cast<uint4*>(smem + offA + buf * (kBM2 * 64) + swz64(pr, pv * 8)) = out.raw;
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&out);
        float acc = sums[slot];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(p[j]);
            acc += f.x * f.x + f.y * f.y;
        }
        sums[slot] = acc;
    };

    if constexpr (kPreloaded) {
        const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int stg = 0; stg < 4; ++stg) {
            rh[stg] = pre_rh[stg];
            rc[stg].raw = prow && inject ? load_row16<kL2Hint>(crow + stg * kBK) : zero;
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
            pipe_barrier_256();
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
        wsum += __shfl_xor_sync(0xffffffff, wsum, 2);
        wsum += __shfl_xor_sync(0xffffffff, wsum, 1);
        half[g] = wsum;
    }
    return rsqrtf(hc_div_approx(half[0] + half[1], (float)kHidden) + eps);
}

__device__ __forceinline__ void load_b_2cta(unsigned int sbase, const __nv_bfloat16* wsrc, int mt, int q, int bslot,
                                            uint64_t pol_el) {
    const int n0 = mt >> 2;
    const int part8 = (mt & 3) * 8;
    const unsigned int sb_slot = sbase + bslot * (kNPad2 * 64);
#pragma unroll
    for (int j = 0; j < 10; ++j) {
        const int n = n0 + j * 32;
        const __nv_bfloat16* src = wsrc + (size_t)n * kHidden + part8 + q * kBK;
        cp16_keep_u32(sb_slot + swz64(n, part8), src, true, pol_el);
    }
    {
        const int n = n0 + kRankBase;
        const bool ok = n < kNTotal;
        const __nv_bfloat16* src = wsrc + (size_t)(ok ? n : 0) * kHidden + part8 + q * kBK;
        if (n < kNPad2) {
            cp16_keep_u32(sb_slot + swz64(n, part8), src, ok, pol_el);
        }
    }
    cp_async_commit();
}

template <bool kPreloaded = false>
__device__ __forceinline__ void multiply_rows_2cta(
    unsigned int sbase, const __nv_bfloat16* __restrict__ folded,
    int s, float (&acc)[kFrags2][4]
) {
    const int mt = threadIdx.x - kProd2;
    const int mw = mt >> 5;
    const int lane = mt & 31;
    const int wm = mw >> 1;
    const int wn = mw & 1;
    const __nv_bfloat16* wsrc = folded + (size_t)s * kNTotal * kHidden;
    const uint64_t pol_el = policy_evict_last();

#pragma unroll
    for (int f = 0; f < kFrags2; ++f) {
#pragma unroll
        for (int x = 0; x < 4; ++x) acc[f][x] = 0.0f;
    }

    if constexpr (!kPreloaded) {
        load_b_2cta(sbase, wsrc, mt, 0, 0, pol_el);
    }

    for (int q = 0; q < kSteps; ++q) {
        const int bslot = q & 1;
        if (q + 1 < kSteps) {
            load_b_2cta(sbase, wsrc, mt, q + 1, bslot ^ 1, pol_el);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        pipe_barrier_256();
        const unsigned int sa_buf = sbase + kSmemB2 + (q & 1) * (kBM2 * 64);
        const unsigned int sb_buf = sbase + bslot * (kNPad2 * 64);
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 16) {
            unsigned int a[4];
            const int ar = wm * 16 + (lane & 15);
            const int ac = kk + (lane >> 4) * 8;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                         : "r"(sa_buf + swz64(ar, ac)));
#pragma unroll
            for (int f = 0; f < 20; f += 2) {
                const int n = wn * (kFrags2 * 8) + f * 8 + (lane & 7) + ((lane >> 4) << 3);
                const int bc = kk + ((lane >> 3) & 1) * 8;
                unsigned int r[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                             : "r"(sb_buf + swz64(n, bc)));
                mma_bf16(acc[f], a, r[0], r[1]);
                mma_bf16(acc[f + 1], a, r[2], r[3]);
            }
            {
                const int n = wn * (kFrags2 * 8) + 20 * 8 + (lane & 7);
                const int bc = kk + ((lane >> 3) & 1) * 8;
                unsigned int r0, r1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(r0), "=r"(r1)
                             : "r"(sb_buf + swz64(n, bc)));
                mma_bf16(acc[20], a, r0, r1);
            }
        }
        // All 4 consumer warps must finish reading `sB[q & 1]` before any
        // consumer warp loops to `q + 1` and overwrites `sB[q & 1]` via `load_b_2cta`.
        asm volatile("bar.sync 2, 128;\n" ::: "memory");
    }
}

template <bool kL2Hint, bool kPreloadedB = false, bool kPreloadedH = false, bool kAlwaysInject = false,
          typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void prepare_tile_2cta(
    unsigned char* smem, unsigned int sbase, const Operands& op,
    const __nv_bfloat16* __restrict__ folded, const Outputs& out,
    unsigned int num_tokens, float eps, float pre_scale = 0.0f, const Vec8* pre_rh = nullptr
) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    const int s = blockIdx.x;
    const int t0 = blockIdx.y * kBM2;
    const int tid = threadIdx.x;
    float* srstd = reinterpret_cast<float*>(smem + kOffRstd2);
    float acc[kFrags2][4];
    if (tid < kProd2) {
        const float r = produce_rows_2cta<kL2Hint, kPreloadedH, kAlwaysInject>(
            smem, op, s, t0, num_tokens, eps, pre_scale, pre_rh);
        if ((tid & 3) == 0) {
            const int row = tid >> 2;
            srstd[row] = r;
            if (t0 + row < (int)num_tokens) out.rstd[(size_t)(t0 + row) * kStreams + s] = r;
        }
    } else {
        multiply_rows_2cta<kPreloadedB>(sbase, folded, s, acc);
    }
    __syncthreads();
    float* __restrict__ sq = reinterpret_cast<float*>(smem);
    if (tid >= kProd2) {
        const int mt = tid - kProd2;
        const int lane = mt & 31;
        const int wm = (mt >> 5) >> 1;
        const int wn = (mt >> 5) & 1;
        const int row0 = wm * 16 + (lane >> 2);
        const int row1 = row0 + 8;
        const float rs0 = srstd[row0];
        const float rs1 = srstd[row1];
#pragma unroll
        for (int f = 0; f < kFrags2; ++f) {
            const int n = wn * (kFrags2 * 8) + f * 8 + 2 * (lane & 3);
            if (n >= kNTotal) continue;
            *reinterpret_cast<float2*>(&sq[row0 * kNTotal + n]) = make_float2(rs0 * acc[f][0], rs0 * acc[f][1]);
            *reinterpret_cast<float2*>(&sq[row1 * kNTotal + n]) = make_float2(rs1 * acc[f][2], rs1 * acc[f][3]);
        }
    }
    cluster.sync();
    const float2* const sq2 = reinterpret_cast<const float2*>(sq);
    const float2* const rq1 = reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 1) & 3));
    const float2* const rq2 = reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 2) & 3));
    const float2* const rq3 = reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 3) & 3));
    const unsigned long long rw = (unsigned long long)kHidden * out.hc;
    constexpr int kRowsPerRank = kBM2 / kStreams;
    constexpr int kPairsPerRow = kNTotal / 2;
    constexpr int kPairsPerRank = kRowsPerRank * kPairsPerRow;
    for (int idx = tid; idx < kPairsPerRank; idx += kThreads2) {
        const int r_local = idx / kPairsPerRow;
        const int p = idx - r_local * kPairsPerRow;
        const int row = s * kRowsPerRank + r_local;
        const int t = t0 + row;
        if (t >= (int)num_tokens) continue;
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
    cluster.sync();
}

__device__ __forceinline__ void moe_form_rows_mx16_2cta(const MoeRows& m, const float* gate_scalars, int row0,
                                                        unsigned int num_tokens) {
    const int r = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int t = row0 + r;
    if (r >= kMoeRows2 || t >= (int)num_tokens) return;
    const uint64_t pol = policy_evict_first();
    const uint64_t keep_pol = policy_evict_last();
    const float gate = gate_scalars[t];
    const __nv_bfloat16* sh_row = m.shared_out + (size_t)t * kHidden;
    __nv_bfloat16* out_row = m.out + (size_t)t * kHidden;
    const int my_perm = (lane < kTopk) ? m.token_to_perm[(size_t)t * kTopk + lane] : 0;
    const float my_w = (lane < kTopk) ? m.topk_weights[(size_t)t * kTopk + lane] : 0.0f;
    uint32_t s_word[kTopk];
#pragma unroll
    for (int k = 0; k < kTopk; ++k) {
        const size_t sk = (size_t)(uint32_t)__shfl_sync(0xffffffffu, my_perm, k);
        s_word[k] = (lane < (kMxScalesPerRow / 4))
                        ? reinterpret_cast<const uint32_t*>(m.slot_scales + sk * kMxScalesPerRow)[lane]
                        : 0u;
    }
    const int byte_shift = ((lane >> 1) & 3) * 8;
#pragma unroll 1
    for (int u = 0; u < 5; ++u) {
        const int h0 = (u * 32 + lane) * 16;
        const int word_lane = u * 4 + (lane >> 3);
        Vec8 sh0, sh1, out0, out1;
        float acc0[8], acc1[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] = 0.0f;
            acc1[i] = 0.0f;
        }
        uint4 raw[kTopk];
#pragma unroll
        for (int k = 0; k < kTopk; ++k) {
            const size_t sk = (size_t)(uint32_t)__shfl_sync(0xffffffffu, my_perm, k);
            raw[k] = ld16_ef_plain(
                reinterpret_cast<const __nv_bfloat16*>(static_cast<const uint8_t*>(m.slots) + sk * kHidden + h0), pol);
        }
#pragma unroll
        for (int k = 0; k < kTopk; ++k) {
            if (k == 5) {
                sh0.raw = ld16_ef_plain(sh_row + h0, pol);
                sh1.raw = ld16_ef_plain(sh_row + h0 + 8, pol);
            }
            const uint32_t ek = (__shfl_sync(0xffffffffu, s_word[k], word_lane) >> byte_shift) & 0xffu;
            const float wk = __shfl_sync(0xffffffffu, my_w, k);
            const float ws = wk * __uint_as_float(ek << 23);
            decode_slot8_acc_ws<true>(make_uint4(raw[k].x, raw[k].y, 0, 0), ws, acc0);
            decode_slot8_acc_ws<true>(make_uint4(raw[k].z, raw[k].w, 0, 0), ws, acc1);
        }
        const __nv_bfloat162* sh0_p = reinterpret_cast<const __nv_bfloat162*>(&sh0);
        const __nv_bfloat162* sh1_p = reinterpret_cast<const __nv_bfloat162*>(&sh1);
        __nv_bfloat162* out0_p = reinterpret_cast<__nv_bfloat162*>(&out0);
        __nv_bfloat162* out1_p = reinterpret_cast<__nv_bfloat162*>(&out1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 s0 = __bfloat1622float2(sh0_p[j]), s1 = __bfloat1622float2(sh1_p[j]);
            const float2 o0 = __bfloat1622float2(__floats2bfloat162_rn(acc0[2 * j], acc0[2 * j + 1]));
            const float2 o1 = __bfloat1622float2(__floats2bfloat162_rn(acc1[2 * j], acc1[2 * j + 1]));
            out0_p[j] = __floats2bfloat162_rn(o0.x + gate * s0.x, o0.y + gate * s0.y);
            out1_p[j] = __floats2bfloat162_rn(o1.x + gate * s1.x, o1.y + gate * s1.y);
        }
        store_row16_stream(out_row + h0, out0.raw, keep_pol);
        store_row16_stream(out_row + h0 + 8, out1.raw, keep_pol);
    }
    if ((reinterpret_cast<uintptr_t>(m.slots) & 127u) == 0u) {
        __syncwarp();
#pragma unroll
        for (int k = 0; k < kTopk; ++k) {
            const size_t sk = (size_t)(uint32_t)__shfl_sync(0xffffffffu, my_perm, k);
            if (lane < 20) {
                const unsigned char* p =
                    static_cast<const unsigned char*>(m.slots) + sk * kHidden + (size_t)lane * 128;
                asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
            }
        }
    }
}

__device__ __forceinline__ int cta_row0_2cta() {
    int bx, by;
    asm volatile("mov.u32 %0, %%ctaid.x;\n\tmov.u32 %1, %%ctaid.y;\n" : "=r"(bx), "=r"(by));
    return by * kBM2 + bx * kMoeRows2;
}

}  // namespace hc_2cta

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_2cta::kThreads2, 2)
qwen4_hc_combine_moe_gemm_mx_gatepre_2cta(
    __nv_bfloat16* __restrict__ hyper,
    __nv_bfloat16* residual,
    const __nv_bfloat16* __restrict__ folded,
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
    check_dyn_smem<hc_2cta::kSmemBytes2>();
    extern __shared__ __align__(16) unsigned char smem[];
    if (threadIdx.x >= hc_2cta::kProd2) {
        unsigned int sbase_pre = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
        asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n" : "+r"(sbase_pre));
        const __nv_bfloat16* wsrc = folded + (size_t)blockIdx.x * kNTotal * kHidden;
        hc_2cta::load_b_2cta(sbase_pre, wsrc, threadIdx.x - hc_2cta::kProd2, 0, 0, policy_evict_last());
    }
    const MoeRows m{expert_output, slot_scales, token_to_perm, topk_weights, shared_out, nullptr, nullptr, moe_rows};
    hc_2cta::moe_form_rows_mx16_2cta(m, gate_scalars, hc_2cta::cta_row0_2cta(), num_tokens);

    unsigned int sbase = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n" : "+r"(sbase));

    float pre_scale = 0.0f;
    Vec8 pre_rh[4];
    if (threadIdx.x < hc_2cta::kProd2) {
        const int s = blockIdx.x;
        const int t0 = blockIdx.y * hc_2cta::kBM2;
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
            pre_rh[stg].raw = prow ? hc_2cta::ld16_ef_plain(hrow + stg * kBK, stream_pol) : zero;
        }
    }
    cooperative_groups::this_cluster().sync();
    if (shared_out != nullptr && ((reinterpret_cast<uintptr_t>(shared_out) & 127u) == 0u)) {
        const int row0_sh = hc_2cta::cta_row0_2cta();
        const int valid_sh = min(hc_2cta::kMoeRows2, max(0, (int)num_tokens - row0_sh));
        const unsigned char* sh_discard = reinterpret_cast<const unsigned char*>(shared_out + (size_t)row0_sh * kHidden);
        const int sh_lines = valid_sh * (kHidden * (int)sizeof(__nv_bfloat16) / 128);
        for (int line = threadIdx.x; line < sh_lines; line += hc_2cta::kThreads2) {
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(sh_discard + (size_t)line * 128) : "memory");
        }
    }
    const Operands op{hyper, moe_rows, residual, nullptr, nullptr};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    hc_2cta::prepare_tile_2cta<true, true, true, true>(smem, sbase, op, folded, out, num_tokens, eps, pre_scale, pre_rh);

    if (moe_rows != nullptr && ((reinterpret_cast<uintptr_t>(moe_rows) & 127u) == 0u)) {
        const int row0_discard = hc_2cta::cta_row0_2cta();
        const int valid_rows = min(hc_2cta::kMoeRows2, max(0, (int)num_tokens - row0_discard));
        unsigned char* base_discard = reinterpret_cast<unsigned char*>(moe_rows + (size_t)row0_discard * kHidden);
        const int total_lines = valid_rows * (kHidden * (int)sizeof(__nv_bfloat16) / 128);
        for (int line = threadIdx.x; line < total_lines; line += hc_2cta::kThreads2) {
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(base_discard + (size_t)line * 128) : "memory");
        }
    }
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_2cta::kThreads2, 2)
qwen4_hc_inject_gemm_ws_gatepre_2cta(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    __nv_bfloat16* residual,
    const __nv_bfloat16* __restrict__ folded,
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
    check_dyn_smem<hc_2cta::kSmemBytes2>();
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned int sbase = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n" : "+r"(sbase));
    const Operands op{hyper, core, residual, nullptr, nullptr};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    hc_2cta::prepare_tile_2cta<false, false, false>(smem, sbase, op, folded, out, num_tokens, eps);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(hc_2cta::kThreads2, 2)
qwen4_hc_inject_gemm_ws_l2h_gatepre_2cta(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    __nv_bfloat16* residual,
    const __nv_bfloat16* __restrict__ folded,
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
    check_dyn_smem<hc_2cta::kSmemBytes2>();
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned int sbase = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n" : "+r"(sbase));
    const Operands op{hyper, core, residual, nullptr, nullptr};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    hc_2cta::prepare_tile_2cta<true, false, false>(smem, sbase, op, folded, out, num_tokens, eps);
    if (core != nullptr && ((reinterpret_cast<uintptr_t>(core) & 127u) == 0u)) {
        const int row0_discard = (int)(blockIdx.y * hc_2cta::kBM2 + blockIdx.x * hc_2cta::kMoeRows2);
        const int valid_rows = min(hc_2cta::kMoeRows2, max(0, (int)num_tokens - row0_discard));
        const unsigned char* base_discard = reinterpret_cast<const unsigned char*>(core + (size_t)row0_discard * kHidden);
        const int total_lines = valid_rows * (kHidden * (int)sizeof(__nv_bfloat16) / 128);
        for (int line = threadIdx.x; line < total_lines; line += hc_2cta::kThreads2) {
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(base_discard + (size_t)line * 128) : "memory");
        }
    }
}

