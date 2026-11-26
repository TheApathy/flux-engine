// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// 64-row warp-specialized cluster inject/combine GEMM kernels and MoE row formation.

namespace {

// Warps 0-7: hyper' of K step after K step into the operand ring (and global
// memory when injecting); returns this producer row's rstd in
// qwen4_hc_rstd's order.
template <bool kL2Hint, bool kPipe3 = false, bool kPreloaded = false, int kOffA = kSmemB>
__device__ __forceinline__ float produce_rows(unsigned char* smem, const Operands& op, int s, int t0,
                                             unsigned int num_tokens, unsigned int hidden_size,
                                             float eps, float pre_scale = 0.0f,
                                             const Vec8* pre_rh = nullptr) {
    auto sA = reinterpret_cast<__nv_bfloat16(*)[kBM][kLds]>(smem + kOffA);
    const int pr = threadIdx.x >> 2;
    const int pv = threadIdx.x & 3;
    const int ptoken = t0 + pr;
    const bool prow = ptoken < (int)num_tokens;
    const size_t row_idx = prow ? (size_t)ptoken : 0;
    const bool inject = op.core != nullptr;
    __nv_bfloat16* hrow = op.hyper + row_idx * kWidth + s * kHidden + pv * 8;
    const float scale = kPreloaded
                            ? pre_scale
                            : (prow && inject
                                   ? __bfloat162float(op.residual[row_idx * kWidth + (kWidth - kStreams) + s])
                                   : 0.0f);
    const __nv_bfloat16* crow = inject ? (op.core + row_idx * kHidden + pv * 8) : nullptr;

    if constexpr (kPipe3) {
        float sums[kSlots];
#pragma unroll
        for (int j = 0; j < kSlots; ++j) sums[j] = 0.0f;
        const bool keep_hyper = (num_tokens <= kKeepHyperMaxSmallTokens) || (t0 + kKeepHyperWindow >= (int)num_tokens);
        const uint64_t pol_ef = policy_evict_first();
        const uint64_t pol_el = policy_evict_last();
        const uint64_t h_load_pol = (!inject && keep_hyper) ? pol_el : pol_ef;
        const uint64_t h_store_pol = keep_hyper ? pol_el : pol_ef;
        Vec8 rh[4], rc[4];
        auto load_stage = [&](int q, int stg) {
            const uint4 zero = make_uint4(0, 0, 0, 0);
            rh[stg].raw = prow ? load_row16_stream<kL2Hint>(hrow + q * kBK, h_load_pol) : zero;
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
                    store_row16_stream(hrow + q * kBK, out.raw, h_store_pol);
                }
            } else {
                out.raw = rh[stg].raw;
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
                pipe_barrier();  // step q is the MMA warps'; ring slot (u + 1) & 1 is free
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
        return rsqrtf(hc_div_approx(half[0] + half[1], (float)hidden_size) + eps);
    } else {
        Vec8 h, c;
        const bool keep_hyper = (num_tokens <= kKeepHyperMaxSmallTokens) || (t0 + kKeepHyperWindow >= (int)num_tokens);
        const uint64_t stream_pol = policy_evict_first();
        const uint64_t keep_pol = policy_evict_last();
        const uint64_t h_load_pol = (!inject && keep_hyper) ? keep_pol : stream_pol;
        const uint64_t store_pol = keep_hyper ? keep_pol : stream_pol;
        auto load = [&](int q) {
            const uint4 zero = make_uint4(0, 0, 0, 0);
            h.raw = prow ? load_row16_stream<kL2Hint>(hrow + q * kBK, h_load_pol) : zero;
            c.raw = prow && inject ? load_row16<kL2Hint>(crow + q * kBK) : zero;
        };
        float sums[kSlots];
#pragma unroll
        for (int j = 0; j < kSlots; ++j) sums[j] = 0.0f;
        auto produce = [&](int q, int slot, int buf) {
            Vec8 out;
            if (inject) {
#pragma unroll
                for (int x = 0; x < 8; ++x) {
                    const float a = __bfloat162float(c.h[x]);
                    out.h[x] = __float2bfloat16(__bfloat162float(h.h[x]) + a * scale);
                }
                if (prow) {
                    store_row16_stream(hrow + q * kBK, out.raw, store_pol);
                }
            } else {
                out.raw = h.raw;
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

        load(0);
        produce(0, 0, 0);
        load(1);
        for (int q0 = 0; q0 < kSteps; q0 += kSlots) {
#pragma unroll
            for (int u = 0; u < kSlots; ++u) {
                const int q = q0 + u;
                pipe_barrier();  // step q is the MMA warps'; ring slot (u + 1) & 1 is free
                if (q + 1 < kSteps) {
                    produce(q + 1, (u + 1) % kSlots, (u + 1) & 1);
                    if (q + 2 < kSteps) load(q + 2);
                }
            }
        }
        // qwen4_hc_rstd's butterflies for this row's two warps (its threads 0-31
        // are slots 0-7, 32-63 slots 8-15): xor 16, 8, 4 combine slots, xor 2, 1
        // this row's four producers.
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
        return rsqrtf(hc_div_approx(half[0] + half[1], (float)hidden_size) + eps);
    }
}

__device__ __forceinline__ void preload_b_stage(unsigned char* smem, const __nv_bfloat16* __restrict__ folded,
                                                int s, int q, int bslot) {
    auto sB = reinterpret_cast<__nv_bfloat16(*)[kNPad][kLds]>(smem);
    const int mt = threadIdx.x - kProducers;
    const __nv_bfloat16* wsrc = folded + (size_t)s * kNTotal * kHidden;
    
    const int n0 = mt >> 2;
    const int part8 = (mt & 3) * 8;
    const __nv_bfloat16* w_q0 = wsrc + (size_t)n0 * kHidden + part8 + q * kBK;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const int n = n0 + j * 64;
        const __nv_bfloat16* src = w_q0 + (size_t)(j * 64 * kHidden);
        cp_async16_keep(&sB[bslot][n][part8], src, true, policy_evict_last());
    }
    if (n0 + kRankBase < kNPad) {
        const int n = n0 + kRankBase;
        const bool ok5 = n < kNTotal;
        const __nv_bfloat16* src = wsrc + (size_t)(ok5 ? n : 0) * kHidden + part8 + q * kBK;
        cp_async16_keep(&sB[bslot][n][part8], src, ok5, policy_evict_last());
    }
    cp_async_commit();
}

// Warps 8-15: the folded weights of stream s and every MMA of the tile.
template <bool kPreloaded = false>
__device__ __forceinline__ void multiply_rows(unsigned char* smem, const __nv_bfloat16* __restrict__ folded,
                                              int s, float (&acc)[kFrags][4]) {
    auto sB = reinterpret_cast<__nv_bfloat16(*)[kNPad][kLds]>(smem);
    auto sA = reinterpret_cast<__nv_bfloat16(*)[kBM][kLds]>(smem + kSmemB);
    const int mt = threadIdx.x - kProducers;
    const int mw = mt >> 5;
    const int lane = mt & 31;
    const int wm = mw >> 1;  // 16-row slab
    const int wn = mw & 1;   // 176-output half
    const __nv_bfloat16* wsrc = folded + (size_t)s * kNTotal * kHidden;
    
    const int n0 = mt >> 2;
    const int part8 = (mt & 3) * 8;
    const __nv_bfloat16* w_base0 = wsrc + (size_t)n0 * kHidden + part8;
    const bool ok5 = (n0 + kRankBase) < kNTotal;
    auto load_b = [&](int q, int bslot) {
        const __nv_bfloat16* src = w_base0 + q * kBK;
#pragma unroll
        for (int j = 0; j < 5; ++j) {
            const int n = n0 + j * 64;
            cp_async16_keep(&sB[bslot][n][part8], src, true, policy_evict_last());
            if (j < 4 || ok5) {
                asm("add.s64 %0, %0, %1;" : "+l"(src) : "l"((long long)(64 * kHidden * sizeof(__nv_bfloat16))));
            }
        }
        if (n0 + kRankBase < kNPad) {
            const int n = n0 + kRankBase;
            cp_async16_keep(&sB[bslot][n][part8], src, ok5, policy_evict_last());
        }
        cp_async_commit();
    };
#pragma unroll
    for (int f = 0; f < kFrags; ++f)
#pragma unroll
        for (int x = 0; x < 4; ++x) acc[f][x] = 0.0f;
    if constexpr (!kPreloaded) {
        load_b(0, 0);
        load_b(1, 1);
    }
    auto step_mma = [&](int q, int buf, int bslot, int next_bslot) {
        cp_async_wait_prev();
        pipe_barrier();  // step q's operands complete; B slot (q + 2) % 3 free
        if (q + 2 < kSteps) {
            load_b(q + 2, next_bslot);
        } else {
            cp_async_commit();  // empty group: wait_prev keeps meaning "step q landed"
        }
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 16) {
            unsigned int a[4];
            ldmatrix_x4(a, &sA[buf][wm * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
            for (int f = 0; f < kFrags; f += 2) {
                // n8 fragments f and f+1: lanes 0-15 address f's k-halves, 16-31 f+1's.
                const int n = wn * (kFrags * 8) + f * 8 + (lane & 7) + ((lane >> 4) << 3);
                unsigned int r[4];
                ldmatrix_x4(r, &sB[bslot][n][kk + ((lane >> 3) & 1) * 8]);
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

struct Outputs {
    float* rstd;
    __nv_bfloat16* down;
    __nv_bfloat16* residual;  // tail: this preparation's inject scales
    unsigned int rank;
    unsigned int hc;
    float scale2_down;
    float scale2_inject;
};

template <bool kL2Hint, bool kPipe3 = false, bool kPreloadedB = false, bool kPreloadedH = kPreloadedB,
          typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void prepare_tile(unsigned char* smem, const Operands& op,
                                             const __nv_bfloat16* __restrict__ folded, const Outputs& out,
                                             unsigned int num_tokens, unsigned int hidden_size, float eps,
                                             float pre_scale = 0.0f, const Vec8* pre_rh = nullptr) {
    static_assert(Shape::kHidden == kHidden && Shape::kMhcStreams == kStreams && Shape::kMhcProjN == kNTotal,
                  "prepare_tile Shape matches HcBaseGeom");
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    const int s = blockIdx.x;
    const int t0 = blockIdx.y * kBM;
    const int tid = threadIdx.x;
    float* srstd = reinterpret_cast<float*>(smem + kOffRstd);
    float acc[kFrags][4];
    if (tid < kProducers) {
        const float r = produce_rows<kL2Hint, kPipe3, kPreloadedH>(
            smem, op, s, t0, num_tokens, hidden_size, eps, pre_scale, pre_rh);
        if ((tid & 3) == 0) {
            const int row = tid >> 2;
            srstd[row] = r;
            if (t0 + row < (int)num_tokens) out.rstd[(size_t)(t0 + row) * kStreams + s] = r;
        }
    } else {
        multiply_rows<kPreloadedB>(smem, folded, s, acc);
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
    // qwen4_hc_combine: the four streams in order from 0, partitioned by
    // 16 rows x 162 float2 column-pairs per cluster rank for 8B DSMEM reads
    // and 4B bfloat162 global stores.
    const float2* const sq2 = reinterpret_cast<const float2*>(sq);
    const float2* const rq1 =
        reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 1) & 3));
    const float2* const rq2 =
        reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 2) & 3));
    const float2* const rq3 =
        reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 3) & 3));
    const unsigned long long rw = (unsigned long long)hidden_size * out.hc;
    constexpr int kRowsPerRank = kBM / kStreams;       // 16
    constexpr int kPairsPerRow = kNTotal / 2;          // 162
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

// Levers `postpre` / `postsplit` of ATLAS_QWEN4_PREFILL_MHC_FUSED: a pending
// MoE output formed once per row. Each stream CTA of a tile forms 16 of its
// 64 rows from contiguous slot rows into a [T, 2560] bf16 buffer; after a
// cluster sync the tile runs the inject pass of qwen4_hc_inject_gemm_ws_l2h
// with that buffer as its core rows (from L2).
// A row is formed with qwen4_hc_combine[_mx]_inject_rstd's expressions
// (moe_unpermute_reduce_indexed's k order, moe_batched_blend's gate tree) and
// injected as that kernel injects it, so hyper', rstd, down and the inject
// tail match the AfterCombine sequence (combine_inject_rstd + the plain pass).
constexpr int kMoeRowVecs = kHidden / 8;             // 16-byte vectors per row
constexpr int kGatePasses = kMoeRows / 2;            // one row per 256-thread half
constexpr int kGateTerms = kHidden / kBlendThreads;  // strided dot terms per thread
constexpr int kMxBlock = 32;
constexpr int kMxScalesPerRow = kHidden / kMxBlock;
static_assert(kThreads == 2 * kBlendThreads && kMoeRows % 2 == 0, "two blend blocks per pass");
static_assert(kOffWts - kOffPerm >= kMoeRows * kTopk * 4 && kSmemBytes - kOffGate >= kMoeRows * 4,
              "the tile's routing slots hold the formed rows' routing and gates");

struct MoeRows {
    const void* slots;           // bf16 [pairs][2560], or the MXFP8 e4m3 payload
    const uint8_t* slot_scales;  // MXFP8 e8m0 [pairs][80]; unused for bf16 slots
    const int* token_to_perm;
    const float* topk_weights;
    const __nv_bfloat16* shared_out;
    const __nv_bfloat16* gate_input;
    const __nv_bfloat16* shared_gate;  // null: gate 1
    __nv_bfloat16* out;                // [T][2560] MoE output rows
};

// Block scratch: perm / wts [kMoeRows][kTopk], gate [kMoeRows], dot [kMoeRows][8].
struct MoeScratch {
    int* perm;
    float* wts;
    float* gate;
    float* dot;
};

__device__ __forceinline__ uint2 load8_stream(const void* p) {
    uint2 v;
    asm volatile(
        "{ .reg .b64 pol;\n"
        "  createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n"
        "  ld.global.L2::cache_hint.v2.u32 {%0,%1}, [%2], pol; }\n"
        : "=r"(v.x), "=r"(v.y)
        : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 load16_evict_first(const void* p) {
    uint4 v;
    asm volatile(
        "{ .reg .b64 pol;\n"
        "  createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n"
        "  ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], pol; }\n"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
    return v;
}

__device__ __forceinline__ void store16_evict_last(void* p, uint4 v) {
    asm volatile(
        "{ .reg .b64 pol;\n"
        "  createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n"
        "  st.global.L2::cache_hint.v4.u32 [%0], {%1,%2,%3,%4}, pol; }\n" ::"l"(p), "r"(v.x), "r"(v.y),
        "r"(v.z), "r"(v.w)
        : "memory");
}

__device__ __forceinline__ void moe_routing(const MoeRows& m, const MoeScratch& sc, int row0,
                                            unsigned int num_tokens) {
    for (int i = threadIdx.x; i < kMoeRows * kTopk; i += kThreads) {
        const int t = row0 + i / kTopk;
        const bool ok = t < (int)num_tokens;
        sc.perm[i] = ok ? m.token_to_perm[(size_t)t * kTopk + i % kTopk] : 0;
        sc.wts[i] = ok ? m.topk_weights[(size_t)t * kTopk + i % kTopk] : 0.0f;
    }
}

// moe_batched_blend's shared-expert gate for rows row0 .. row0 + 15: each
// 256-thread half of the block is one blend block (strided dot in ascending
// i, shfl_down tree, warps summed in order) for rows 2j + half; every pass's
// loads are issued before the trees.
__device__ __forceinline__ void moe_gates(const MoeRows& m, const MoeScratch& sc, int row0,
                                          unsigned int num_tokens) {
    const int tid = threadIdx.x;
    const int half = tid / kBlendThreads;
    const int bt = tid % kBlendThreads;
    float dots[kGatePasses];
#pragma unroll
    for (int j = 0; j < kGatePasses; ++j) dots[j] = 0.0f;
    if (m.shared_gate != nullptr) {
        float g[kGateTerms];
#pragma unroll
        for (int i = 0; i < kGateTerms; ++i) g[i] = __bfloat162float(m.shared_gate[bt + i * kBlendThreads]);
#pragma unroll
        for (int j = 0; j < kGatePasses; ++j) {
            const int t = row0 + 2 * j + half;
            if (t < (int)num_tokens) {
                const __nv_bfloat16* x = m.gate_input + (size_t)t * kHidden;
#pragma unroll
                for (int i = 0; i < kGateTerms; ++i) {
                    const float n = __bfloat162float(x[bt + i * kBlendThreads]);
                    dots[j] += n * g[i];
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kGatePasses; ++j) {
        float d = dots[j];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) d += __shfl_down_sync(0xFFFFFFFF, d, offset);
        if ((bt & 31) == 0) sc.dot[(2 * j + half) * (kBlendThreads / 32) + bt / 32] = d;
    }
    __syncthreads();
    if (tid < kMoeRows) {
        float gate = 1.0f;
        if (m.shared_gate != nullptr) {
            float total = 0.0f;
            for (int w = 0; w < kBlendThreads / 32; w++) total += sc.dot[tid * (kBlendThreads / 32) + w];
            gate = hc_fast_rcp_rn_1p_exp(total);
        }
        sc.gate[tid] = gate;
    }
}

// Columns h0 .. h0 + 7 of one slot row pre-multiplied by `ws` (`wk * scale`
// for MXFP8, `wk` for bf16) and accumulated directly into `acc`. Because
// 2^(e - 127) is an exact power of two, hoisting `ws = wk * scale` out of the
// 8-element decode loop is bit-identical (0-ULP) while doing 1 scale multiply
// instead of 8 (or 16 for mx16) per slot and avoiding 8-float temporaries.
template <bool kMx>
__device__ __forceinline__ void decode_slot8_acc_ws(uint4 raw, float ws, float (&acc)[8]) {
    if constexpr (kMx) {
        const uint32_t words[2] = {raw.x, raw.y};
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(words[w] & 0xffffu), __NV_E4M3);
            const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(words[w] >> 16), __NV_E4M3);
            const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&lo));
            const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&hi));
            acc[4 * w + 0] += a.x * ws;
            acc[4 * w + 1] += a.y * ws;
            acc[4 * w + 2] += b.x * ws;
            acc[4 * w + 3] += b.y * ws;
        }
    } else {
        Vec8 v;
        v.raw = raw;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] += ws * __bfloat162float(v.h[i]);
    }
}

template <bool kMx>
__device__ __forceinline__ void decode_slot8(uint4 raw, uint32_t e, float wk, float (&acc)[8]) {
    const float ws = kMx ? (wk * __uint_as_float(e << 23)) : wk;
    decode_slot8_acc_ws<kMx>(raw, ws, acc);
}

// The MoE output rows row0 .. row0 + 15 into m.out, one 8-column vector per
// thread step: all ten slot vectors are loaded, then summed in k order.
template <bool kMx, bool kKeep = false>
__device__ __forceinline__ void moe_form_rows(const MoeRows& m, const MoeScratch& sc, int row0,
                                              unsigned int num_tokens) {
    for (int v = threadIdx.x; v < kMoeRows * kMoeRowVecs; v += kThreads) {
        const int r = v / kMoeRowVecs;
        const int h0 = (v % kMoeRowVecs) * 8;
        const int t = row0 + r;
        if (t >= (int)num_tokens) continue;
        Vec8 sh, out;
        sh.raw = load16_evict_first(m.shared_out + (size_t)t * kHidden + h0);
        float acc[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            uint4 raw[5];
            uint32_t e[5];
#pragma unroll
            for (int j = 0; j < 5; ++j) {
                const int k = half * 5 + j;
                const size_t slot = (size_t)sc.perm[r * kTopk + k];
                if constexpr (kMx) {
                    const uint2 q = load8_stream(static_cast<const uint8_t*>(m.slots) + slot * kHidden + h0);
                    raw[j] = make_uint4(q.x, q.y, 0, 0);
                    e[j] = m.slot_scales[slot * kMxScalesPerRow + h0 / kMxBlock];
                } else {
                    raw[j] = load16_evict_first(static_cast<const __nv_bfloat16*>(m.slots) + slot * kHidden + h0);
                    e[j] = 0;
                }
            }
#pragma unroll
            for (int j = 0; j < 5; ++j) {
                const int k = half * 5 + j;
                const float w = sc.wts[r * kTopk + k];
                decode_slot8<kMx>(raw[j], e[j], w, acc);
            }
        }
        const float gate = sc.gate[r];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float o = __bfloat162float(__float2bfloat16(acc[i]));
            out.h[i] = __float2bfloat16(o + gate * __bfloat162float(sh.h[i]));
        }
        if constexpr (kKeep) {
            store16_evict_last(m.out + (size_t)t * kHidden + h0, out.raw);
        } else {
            *reinterpret_cast<uint4*>(m.out + (size_t)t * kHidden + h0) = out.raw;
        }
    }
}

template <bool kMx>
__device__ __forceinline__ void moe_prologue(const MoeRows& m, const MoeScratch& sc, int row0,
                                             unsigned int num_tokens) {
    moe_routing(m, sc, row0, num_tokens);
    moe_gates(m, sc, row0, num_tokens);
    __syncthreads();  // routing and gates visible
    moe_form_rows<kMx>(m, sc, row0, num_tokens);
}

// postpre: rank s forms rows 16 s .. 16 s + 15 of the tile, then the tile runs
// the inject pass on the formed rows. The prologue's scratch sits in the
// tile's routing slots and (dot) the operand ring, idle until prepare_tile.
template <bool kMx, typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void combine_moe_tile(unsigned char* smem, const MoeRows& m,
                                                 __nv_bfloat16* hyper, __nv_bfloat16* residual,
                                                 const __nv_bfloat16* folded, const Outputs& out,
                                                 unsigned int num_tokens, float eps) {
    static_assert(Shape::kHidden == kHidden && Shape::kTopK == kTopk && Shape::kMhcProjN == kNTotal,
                  "combine_moe_tile Shape matches HcBaseGeom");
    const MoeScratch sc{reinterpret_cast<int*>(smem + kOffPerm), reinterpret_cast<float*>(smem + kOffWts),
                        reinterpret_cast<float*>(smem + kOffGate), reinterpret_cast<float*>(smem)};
    moe_prologue<kMx>(m, sc, (int)(blockIdx.y * kBM + blockIdx.x * kMoeRows), num_tokens);
    // Release / acquire at cluster scope: the tile's formed rows (global
    // memory) are visible to all four ranks; the ring is free again.
    cooperative_groups::this_cluster().sync();
    if (threadIdx.x >= kProducers) {
        preload_b_stage(smem, folded, blockIdx.x, 0, 0);
        preload_b_stage(smem, folded, blockIdx.x, 1, 1);
    }
    const Operands op{hyper, m.out, residual, nullptr, nullptr};
    prepare_tile<true, true, true, false>(smem, op, folded, out, num_tokens, kHidden, eps);
}

}  // namespace

// Pending core injection (or none when `core` is null) on this
// warp-specialised pipeline, plain loads (the read-only pass: micro4 1.25 ms
// @8192 vs 1.39 with the L2 hint), with L2 eviction priority (micro12: 1.32 ->
// 1.23 ms).
extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_gemm_ws(
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
    check_dyn_smem<kSmemBytes>();
    extern __shared__ __align__(16) unsigned char smem[];
    const Operands op{hyper, core, residual, nullptr, nullptr};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    prepare_tile<false>(smem, op, folded, out, num_tokens, hidden_size, eps);
}

// The same kernel with the L2::256B row prefetch, for a pass that also reads
// core rows and writes hyper' (micro4: 2.94 -> 2.62 ms @8192), with L2
// eviction priority (micro12: 2.55 -> 2.24 ms). A separate
// kernel on purpose: one kernel choosing between both instantiations at run
// time measured 7.36 ms in inject mode (micro9; fnwide15's -6.5%).
extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_gemm_ws_l2h(
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
    check_dyn_smem<kSmemBytes>();
    extern __shared__ __align__(16) unsigned char smem[];
    const Operands op{hyper, core, residual, nullptr, nullptr};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    prepare_tile<true>(smem, op, folded, out, num_tokens, hidden_size, eps);
}

// Lever `postpre` (bf16 slot rows): the pending MoE output formed once per row
// into `moe_rows` ([T, 2560] bf16 scratch), then injected, normalised and
// projected by the inject pass in the same cluster kernel. `slot_scales` is
// unused here so both bf16 and MXFP8 slot formats share one launcher.
// Byte-identical to qwen4_hc_combine_inject_rstd + qwen4_hc_inject_gemm_ws.
extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_combine_moe_gemm(
    __nv_bfloat16* __restrict__ hyper,
    __nv_bfloat16* residual,
    const __nv_bfloat16* __restrict__ folded,
    float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ down,
    __nv_bfloat16* moe_rows,
    const __nv_bfloat16* __restrict__ expert_output,
    const uint8_t* __restrict__ slot_scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ gate_input,
    const __nv_bfloat16* __restrict__ shared_gate,
    unsigned int num_tokens,
    unsigned int rank,
    unsigned int hidden_size,
    unsigned int hc,
    unsigned int topk,
    float scale2_down,
    float scale2_inject,
    float eps) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || topk != kTopk) __trap();
    check_dyn_smem<kSmemBytes>();
    extern __shared__ __align__(16) unsigned char smem[];
    const MoeRows m{expert_output, slot_scales, token_to_perm, topk_weights, shared_out, gate_input, shared_gate,
                    moe_rows};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    combine_moe_tile<false>(smem, m, hyper, residual, folded, out, num_tokens, eps);
}

// qwen4_hc_combine_moe_gemm reading MXFP8 slot rows (`expert_output` is the
// e4m3 payload, `slot_scales` the e8m0 scales). Byte-identical to
// qwen4_hc_combine_mx_inject_rstd + qwen4_hc_inject_gemm_ws.
extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_combine_moe_gemm_mx(
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
    const __nv_bfloat16* __restrict__ gate_input,
    const __nv_bfloat16* __restrict__ shared_gate,
    unsigned int num_tokens,
    unsigned int rank,
    unsigned int hidden_size,
    unsigned int hc,
    unsigned int topk,
    float scale2_down,
    float scale2_inject,
    float eps) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || topk != kTopk) __trap();
    check_dyn_smem<kSmemBytes>();
    extern __shared__ __align__(16) unsigned char smem[];
    const MoeRows m{expert_output, slot_scales, token_to_perm, topk_weights, shared_out, gate_input, shared_gate,
                    moe_rows};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    combine_moe_tile<true>(smem, m, hyper, residual, folded, out, num_tokens, eps);
}

// Lever `postsplit`: only the row formation, 16 rows per block (grid
// ceil(T / 16), block 512); qwen4_hc_inject_gemm_ws_l2h then injects
// `moe_rows` as a pending core output. `slot_scales` null: bf16 slot rows.
template <bool kMx, typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void moe_rows_block(__nv_bfloat16* moe_rows, const void* slots,
                                               const uint8_t* slot_scales, const int* token_to_perm,
                                               const float* topk_weights, const __nv_bfloat16* shared_out,
                                               const __nv_bfloat16* gate_input,
                                               const __nv_bfloat16* shared_gate, unsigned int num_tokens) {
    static_assert(Shape::kHidden == kHidden && Shape::kTopK == kTopk, "moe_rows_block Shape matches HcBaseGeom");
    __shared__ int perm[kMoeRows * kTopk];
    __shared__ float wts[kMoeRows * kTopk];
    __shared__ float gate[kMoeRows];
    __shared__ float dot[kMoeRows * (kBlendThreads / 32)];
    const MoeRows m{slots, slot_scales, token_to_perm, topk_weights, shared_out, gate_input, shared_gate, moe_rows};
    moe_prologue<kMx>(m, MoeScratch{perm, wts, gate, dot}, (int)(blockIdx.x * kMoeRows), num_tokens);
}

extern "C" __global__ void __launch_bounds__(kThreads, 2)
qwen4_hc_moe_rows(
    __nv_bfloat16* __restrict__ moe_rows,
    const __nv_bfloat16* __restrict__ expert_output,
    const uint8_t* __restrict__ slot_scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ gate_input,
    const __nv_bfloat16* __restrict__ shared_gate,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk) {
    if (hidden_size != kHidden || topk != kTopk) __trap();
    moe_rows_block<false>(moe_rows, expert_output, slot_scales, token_to_perm, topk_weights, shared_out, gate_input,
                          shared_gate, num_tokens);
}

extern "C" __global__ void __launch_bounds__(kThreads, 2)
qwen4_hc_moe_rows_mx(
    __nv_bfloat16* __restrict__ moe_rows,
    const uint8_t* __restrict__ expert_output,
    const uint8_t* __restrict__ slot_scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ gate_input,
    const __nv_bfloat16* __restrict__ shared_gate,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk) {
    if (hidden_size != kHidden || topk != kTopk) __trap();
    moe_rows_block<true>(moe_rows, expert_output, slot_scales, token_to_perm, topk_weights, shared_out, gate_input,
                         shared_gate, num_tokens);
}

