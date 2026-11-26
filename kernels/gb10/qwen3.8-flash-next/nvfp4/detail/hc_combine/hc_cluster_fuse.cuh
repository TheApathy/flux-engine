// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// Single-launch cluster-fused inject + upmix kernels (BM=32, 92,672 B SMEM).

// ============================================================================
// Single-Kernel Fused 4-CTA Cluster mHC v2 (Phase 1 inject/down + Phase 2
// disjoint-H upmix): consumes pre-dequantized BF16 `folded` + `up` from a
// single `qwen4_hc_prep_weights(..., up_out=s.up)` launch via `cp.async.cg`
// (zero inline NVFP4 dequant, zero LUT shuffles), keeps Phase-1 `hyper'`
// L2-hot for Phase 2 (`<= 22 MiB` working set across 12 active 4-CTA clusters),
// and runs Phase 2 over 10 disjoint H-tiles/rank with 0 `cluster.sync()`s and
// tail-normal `mixed` stores. Strict 0-ULP bit-exact to
// `prep_weights + inject_gemm + prep_up + upmix_roof`.
// Grid (4, ceil(T / 32)), block 512, cluster (4, 1, 1), 1 CTA/SM,
// dynamic smem hc_cluster_fuse::kSmemBytesF (92,672 B <= 101,376 B).
// ============================================================================
namespace hc_cluster_fuse {

constexpr int kRank = Qwen4FlashNextShape::kMhcRank;
constexpr int kBMF = 32;
constexpr int kMoeRowsF = kBMF / kStreams;                     // 8 rows formed per CTA in PostPreMx
constexpr int kBK1 = 64;                                       // 128 B/row (full L2 cache line per 8 lanes)
constexpr int kSteps1 = kHidden / kBK1;                        // 40 K-steps
constexpr int kSlots1 = 8;                                     // 8 producer threads/row x 8 slots = 64 rstd lanes
constexpr int kNPad1 = ((kNTotal + 7) / 8) * 8;                // 328 (>= kNTotal=324, mult of 8)
constexpr int kRowBytes1 = kBK1 * 2;                           // 128 B
constexpr int kSmemB1 = 2 * kNPad1 * kRowBytes1;               // 83,968 B (2 stages swz128)
constexpr int kSmemA1 = 2 * kBMF * kRowBytes1;                 //  8,192 B (2 stages swz128)
constexpr int kOffRstdF = kSmemB1 + kSmemA1;                   // 92,160 B (sRstd[32][4] persists across Phase 1->2)
constexpr int kSmemBytesF = kOffRstdF + kBMF * kStreams * 4;   // 92,672 B (<= 101,376 B)

// Phase 1 epilogue / handoff (over idle sB1+sA1 [0 .. 92,160)):
constexpr int kSmemQF = kBMF * kNTotal * 4;                    // 41,472 B ([0 .. 41,472))
constexpr int kOffDownPartF = kSmemQF;                         // 41,472 B
constexpr int kSmemDownPartF = kMoeRowsF * kRank * 2;          //  5,120 B ([41,472 .. 46,592))
constexpr int kOffRstdSelfF = kOffDownPartF + kSmemDownPartF;  // 46,592 B
constexpr int kSmemRstdSelfF = kBMF * 4;                       //    128 B ([46,592 .. 46,720))

// Phase 2 layout (over [0 .. 92,160); sRstd[32][4] remains at kOffRstdF = 92,160):
constexpr int kBH2 = 64;
constexpr int kBN2 = kBH2 * kStreams;                          // 256 (stream-major: c = s*64 + hl)
constexpr int kBK2 = 32;
constexpr int kKSteps2 = kRank / kBK2;                         // 10
constexpr int kHTilesPerRank = (kHidden / kBH2) / kStreams;    // 10 disjoint H-tiles per CTA rank
constexpr int kStagesB2 = 3;
constexpr int kSmemDownF = kBMF * kRank * 2;                   // 20,480 B ([0 .. 20,480), 10 slabs x 2,048 B swz64)
constexpr int kOffB2F = kSmemDownF;                            // 20,480 B
constexpr int kStageB2F = kBN2 * (kBK2 * 2);                   // 16,384 B
constexpr int kSmemB2F = kStagesB2 * kStageB2F;                // 49,152 B ([20,480 .. 69,632))
constexpr int kHRow2 = kStreams * kBH2 * 2 + 16;               //    528 B (conflict-free 4B reads)
constexpr int kOffHyperEpiF = kOffB2F + kSmemB2F;              // 69,632 B
constexpr int kSmemHyperEpiF = kBMF * kHRow2;                  // 16,896 B ([69,632 .. 86,528))
constexpr int kORow2 = kBH2 * 2 + 16;                          //    144 B (conflict-free 4B writes)
constexpr int kOffOutEpiF = kOffHyperEpiF + kSmemHyperEpiF;    // 86,528 B
constexpr int kSmemOutEpiF = kBMF * kORow2;                    //  4,608 B ([86,528 .. 91,136))
constexpr int kOffWarpMaxF = kOffOutEpiF + kSmemOutEpiF;       // 91,136 B
constexpr int kSmemWarpMaxF = (kThreads / 32) * 4;             //     64 B ([91,136 .. 91,200))

// Last N rows keep `mixed` in L2 (normal store policy) so downstream FP8 quant
// / gate kernels hit L2 warm without evicting active Phase-1/2 working sets.
constexpr int kMixedTailKeepRows = 1200;

static_assert(kSteps1 == 40 && kSteps1 % kSlots1 == 0, "Phase 1 K-steps and rstd slots");
static_assert(kOffRstdSelfF + kSmemRstdSelfF <= kOffRstdF, "Phase 1 handoff fits under sRstd");
static_assert(kSmemDownF <= kOffDownPartF, "sDown gather disjoint from sDownPart and sRstdSelf");
static_assert(kOffWarpMaxF + kSmemWarpMaxF <= kOffRstdF, "Phase 2 buffers disjoint from sRstd");
static_assert(kOffB2F % 16 == 0 && kOffHyperEpiF % 16 == 0 && kOffOutEpiF % 16 == 0 && kOffRstdF % 16 == 0,
              "16-byte aligned SMEM regions");
static_assert(kSmemBytesF == 92672 && kSmemBytesF <= 101376, "fits 1 CTA/SM on sm_121a");

__device__ __forceinline__ unsigned int swz128(int row, int col_bf16) {
    const int chunk = col_bf16 >> 3;
    return static_cast<unsigned int>(row * 128 + ((chunk ^ (row & 7)) << 4));
}

__device__ __forceinline__ uint64_t pol_evict_last_local() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}

__device__ __forceinline__ uint64_t pol_evict_first_local() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}

__device__ __forceinline__ void cp16_hint_u32(unsigned int s, const void* gmem, bool pred, uint64_t pol) {
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                 :: "r"(s), "l"(gmem), "r"(bytes), "l"(pol) : "memory");
}

__device__ __forceinline__ void cp_async_commit_f() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_f() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void ldsm_x4(unsigned int (&r)[4], unsigned int s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

__device__ __forceinline__ void ldsm_x2(unsigned int& r0, unsigned int& r1, unsigned int s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(s));
}

__device__ __forceinline__ void discard_rows_128(const __nv_bfloat16* base, int row0, int valid_rows) {
    const unsigned char* p0 = reinterpret_cast<const unsigned char*>(base + (size_t)row0 * kHidden);
    const int total_lines = valid_rows * (kHidden * (int)sizeof(__nv_bfloat16) / 128);
    for (int line = threadIdx.x; line < total_lines; line += kThreads) {
        asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p0 + (size_t)line * 128) : "memory");
    }
}

__device__ __forceinline__ int cta_row0_fused() {
    int bx, by;
    asm volatile("mov.u32 %0, %%ctaid.x;\n\tmov.u32 %1, %%ctaid.y;\n" : "=r"(bx), "=r"(by));
    return by * kBMF + bx * kMoeRowsF;
}

// 2 warps per row across 8 rows/CTA (all 16 warps active): each warp covers 80
// 16-element chunks (1280 cols = 10 scale words) and discards the 10 cache lines
// of each slot row it consumed. Bit-identical (0-ULP) to moe_form_rows_mx16.
__device__ __forceinline__ void moe_form_rows_mx16_2w(
    const MoeRows& m,
    const float* __restrict__ gate_scalars,
    int row0,
    unsigned int num_tokens) {
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int r = warp >> 1;      // 0..7
    const int w_sub = warp & 1;   // 0..1 (cols [0..1279] or [1280..2559])
    const int t = row0 + r;
    if (t >= (int)num_tokens) return;

    const uint64_t pol = pol_evict_first_local();
    const uint64_t keep_pol = pol_evict_last_local();
    const float gate = gate_scalars[t];
    const __nv_bfloat16* sh_row = m.shared_out + (size_t)t * kHidden;
    __nv_bfloat16* out_row = m.out + (size_t)t * kHidden;
    const int my_perm = (lane < kTopk) ? m.token_to_perm[(size_t)t * kTopk + lane] : 0;
    const float my_w = (lane < kTopk) ? m.topk_weights[(size_t)t * kTopk + lane] : 0.0f;

    uint32_t s_word[kTopk];
#pragma unroll
    for (int k = 0; k < kTopk; ++k) {
        const size_t sk = (size_t)(uint32_t)__shfl_sync(0xffffffffu, my_perm, k);
        s_word[k] = (lane < 10)
                        ? reinterpret_cast<const uint32_t*>(m.slot_scales + sk * kMxScalesPerRow)[w_sub * 10 + lane]
                        : 0u;
    }

    const int byte_shift = ((lane >> 1) & 3) * 8;
    const int base_h0 = w_sub * (kHidden / 2);
#pragma unroll 1
    for (int u = 0; u < 3; ++u) {
        const bool active = (u < 2) || (lane < 16);
        const int h0 = base_h0 + (u * 32 + lane) * 16;
        const int word_lane = u * 4 + (lane >> 3);
        Vec8 sh0, sh1, out0, out1;
        sh0.raw = active ? load_row16_stream<false>(sh_row + h0, pol) : make_uint4(0, 0, 0, 0);
        sh1.raw = active ? load_row16_stream<false>(sh_row + h0 + 8, pol) : make_uint4(0, 0, 0, 0);
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
            raw[k] = active ? load_row16_stream<false>(
                                  reinterpret_cast<const __nv_bfloat16*>(
                                      static_cast<const uint8_t*>(m.slots) + sk * kHidden + h0),
                                  pol)
                            : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int k = 0; k < kTopk; ++k) {
            const uint32_t ek = (__shfl_sync(0xffffffffu, s_word[k], word_lane) >> byte_shift) & 0xffu;
            const float wk = __shfl_sync(0xffffffffu, my_w, k);
            if (active) {
                const float ws = wk * __uint_as_float(ek << 23);
                decode_slot8_acc_ws<true>(make_uint4(raw[k].x, raw[k].y, 0, 0), ws, acc0);
                decode_slot8_acc_ws<true>(make_uint4(raw[k].z, raw[k].w, 0, 0), ws, acc1);
            }
        }
        if (active) {
            const __nv_bfloat162* sh0_p = reinterpret_cast<const __nv_bfloat162*>(&sh0);
            const __nv_bfloat162* sh1_p = reinterpret_cast<const __nv_bfloat162*>(&sh1);
            __nv_bfloat162* out0_p = reinterpret_cast<__nv_bfloat162*>(&out0);
            __nv_bfloat162* out1_p = reinterpret_cast<__nv_bfloat162*>(&out1);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 s0 = __bfloat1622float2(sh0_p[j]);
                const float2 s1 = __bfloat1622float2(sh1_p[j]);
                const float2 o0 = __bfloat1622float2(__floats2bfloat162_rn(acc0[2 * j], acc0[2 * j + 1]));
                const float2 o1 = __bfloat1622float2(__floats2bfloat162_rn(acc1[2 * j], acc1[2 * j + 1]));
                out0_p[j] = __floats2bfloat162_rn(o0.x + gate * s0.x, o0.y + gate * s0.y);
                out1_p[j] = __floats2bfloat162_rn(o1.x + gate * s1.x, o1.y + gate * s1.y);
            }
            store_row16_stream(out_row + h0, out0.raw, keep_pol);
            store_row16_stream(out_row + h0 + 8, out1.raw, keep_pol);
        }
    }

    if ((reinterpret_cast<uintptr_t>(m.slots) & 127u) == 0u) {
        __syncwarp();
        const size_t line_base = (size_t)(w_sub * 10 + lane) * 128;
#pragma unroll
        for (int k = 0; k < kTopk; ++k) {
            const size_t sk = (size_t)(uint32_t)__shfl_sync(0xffffffffu, my_perm, k);
            if (lane < 10) {
                const unsigned char* p = static_cast<const unsigned char*>(m.slots) + sk * kHidden + line_base;
                asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
            }
        }
    }
}

// Phase 1 Producer Warps 0-7: BM=32, BK=64 (8 threads/row), 4-deep software
// prefetch, hyper' written L2::evict_last for Phase 2, exact 0-ULP
// qwen4_hc_rstd reduction order in 8 register slots.
template <bool kInject, bool kL2Hint, bool kPreloadedH = false>
__device__ __forceinline__ float produce_rows_v2(
    unsigned char* smem,
    const Operands& op,
    int s,
    int t0,
    unsigned int num_tokens,
    float eps,
    float pre_scale = 0.0f,
    const Vec8* pre_rh = nullptr) {
    const int tid = threadIdx.x;  // 0..255
    const int pr = tid >> 3;      // 0..31 (row)
    const int pv = tid & 7;       // 0..7  (16B chunk in kBK1=64)
    const int ptoken = t0 + pr;
    const bool prow = ptoken < (int)num_tokens;
    const size_t row_idx = prow ? (size_t)ptoken : 0;
    __nv_bfloat16* hrow = op.hyper + row_idx * kWidth + s * kHidden + pv * 8;
    const float scale =
        kPreloadedH
            ? pre_scale
            : ((prow && kInject) ? __bfloat162float(op.residual[row_idx * kWidth + (kWidth - kStreams) + s]) : 0.0f);
    const __nv_bfloat16* crow = kInject ? (op.core + row_idx * kHidden + pv * 8) : nullptr;

    float sums[kSlots1];
#pragma unroll
    for (int j = 0; j < kSlots1; ++j) sums[j] = 0.0f;

    const uint64_t pol_ef = pol_evict_first_local();
    const uint64_t pol_el = pol_evict_last_local();
    const uint64_t h_load_pol = kInject ? pol_ef : pol_el;

    Vec8 rh[4], rc[4];
    auto load_stage = [&](int q, int stg) {
        const uint4 zero = make_uint4(0, 0, 0, 0);
        rh[stg].raw = prow ? load_row16_stream<kL2Hint>(hrow + q * kBK1, h_load_pol) : zero;
        if constexpr (kInject) {
            rc[stg].raw = prow ? load_row16<kL2Hint>(crow + q * kBK1) : zero;
        }
    };
    auto produce_stage = [&](int q, int slot, int buf, int stg) {
        Vec8 out;
        if constexpr (kInject) {
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
                store_row16_stream(hrow + q * kBK1, out.raw, pol_el);
            }
        } else {
            out.raw = rh[stg].raw;
        }
        *reinterpret_cast<uint4*>(smem + kSmemB1 + buf * (kBMF * kRowBytes1) + swz128(pr, pv * 8)) = out.raw;
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
        const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int stg = 0; stg < 4; ++stg) {
            rh[stg] = pre_rh[stg];
            if constexpr (kInject) {
                rc[stg].raw = prow ? load_row16<kL2Hint>(crow + stg * kBK1) : zero;
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
    for (int q0 = 0; q0 < kSteps1; q0 += kSlots1) {
#pragma unroll
        for (int u = 0; u < kSlots1; ++u) {
            const int q = q0 + u;
            const int stg = (u + 1) & 3;
            pipe_barrier();  // step q handed to consumers; slot (u + 1) & 1 free
            if (q + 1 < kSteps1) {
                produce_stage(q + 1, (u + 1) & 7, (u + 1) & 1, stg);
                if (q + 5 < kSteps1) load_stage(q + 5, stg);
            }
        }
    }
    // Exact qwen4_hc_rstd reduction order: 64 lanes = 2 warps x 32 lanes.
    // Within half g in {0, 1}, sums[4*g + 0..3] are slot_in_g 0..3 for pv 0..7:
    // xor 16 -> v[0]+v[2], v[1]+v[3]; xor 8 -> x0+x1; xor 4, 2, 1 across pv.
    float half[2];
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const float* v = sums + 4 * g;
        const float x0 = v[0] + v[2];
        const float x1 = v[1] + v[3];
        float wsum = x0 + x1;
        wsum += __shfl_xor_sync(0xffffffffu, wsum, 4);
        wsum += __shfl_xor_sync(0xffffffffu, wsum, 2);
        wsum += __shfl_xor_sync(0xffffffffu, wsum, 1);
        half[g] = wsum;
    }
    return rsqrtf(hc_div_approx(half[0] + half[1], (float)kHidden) + eps);
}

// Phase 1 Consumer cp.async loader for pre-dequantized BF16 `folded` [4][324][2560]:
// 256 consumer threads load 324 rows x 64 bf16 (128 B/row, full cache line per 8 lanes).
__device__ __forceinline__ void load_b1(
    unsigned int sbase,
    const __nv_bfloat16* __restrict__ wsrc,
    int mt,
    int q,
    int bslot,
    uint64_t pol_el) {
    const int n0 = mt >> 3;          // 0..31
    const int part8 = (mt & 7) * 8;  // 0, 8, ..., 56
    const unsigned int sb_slot = sbase + (unsigned int)(bslot * (kNPad1 * kRowBytes1));
    const __nv_bfloat16* src = wsrc + (size_t)n0 * kHidden + q * kBK1 + part8;
#pragma unroll
    for (int j = 0; j < 10; ++j) {
        const int n = n0 + j * 32;
        cp16_hint_u32(sb_slot + swz128(n, part8), src, true, pol_el);
        if (j < 9 || n0 < 4) {
            asm("add.s64 %0, %0, %1;" : "+l"(src) : "l"((long long)(32 * kHidden * sizeof(__nv_bfloat16))));
        }
    }
    if (n0 < 8) {
        const int n = n0 + kRank;
        const bool ok = n < kNTotal;
        const __nv_bfloat16* tail_src = ok ? src : wsrc;
        cp16_hint_u32(sb_slot + swz128(n, part8), tail_src, ok, pol_el);
    }
    cp_async_commit_f();
}

// Phase 1 Consumer Warps 8-15: 2 M-tiles x 4 N-quarters (10, 10, 10, 11 n8 frags),
// pure cp.async.cg + ldmatrix + mma_bf16 over pre-dequantized BF16 `folded`.
template <bool kPreloadedB = false>
__device__ __forceinline__ void multiply_rows_v2(
    unsigned int sbase,
    const __nv_bfloat16* __restrict__ folded,
    int s,
    float (&acc)[11][4]) {
    const int mt = threadIdx.x - kProducers;  // 0..255
    const int mw = mt >> 5;                   // 0..7
    const int lane = mt & 31;
    const int wm = mw >> 2;                   // 0..1 (16-row slab)
    const int wn = mw & 3;                    // 0..3 (80/80/80/88-col quarter)
    const int n_base = wn * (kRank / 4);
    const __nv_bfloat16* wsrc = folded + (size_t)s * kNTotal * kHidden;
    const uint64_t pol_el = pol_evict_last_local();

#pragma unroll
    for (int f = 0; f < 11; ++f)
#pragma unroll
        for (int x = 0; x < 4; ++x) acc[f][x] = 0.0f;

    if constexpr (!kPreloadedB) {
        load_b1(sbase, wsrc, mt, 0, 0, pol_el);
    }

#pragma unroll 1
    for (int q = 0; q < kSteps1; ++q) {
        const int bslot = q & 1;
        cp_async_wait_f<0>();
        pipe_barrier();  // step q operands landed; all warps finished step q-1 on sB1[bslot^1]
        if (q + 1 < kSteps1) {
            load_b1(sbase, wsrc, mt, q + 1, bslot ^ 1, pol_el);
        }
        const unsigned int sa_buf = sbase + kSmemB1 + (unsigned int)(bslot * (kBMF * kRowBytes1));
        const unsigned int sb_buf = sbase + (unsigned int)(bslot * (kNPad1 * kRowBytes1));
#pragma unroll
        for (int kk = 0; kk < kBK1; kk += 16) {
            unsigned int a[4];
            const int ar = wm * 16 + (lane & 15);
            const int ac = kk + (lane >> 4) * 8;
            ldsm_x4(a, sa_buf + swz128(ar, ac));
#pragma unroll
            for (int f = 0; f < 10; f += 2) {
                const int bn = n_base + f * 8 + (lane & 7) + ((lane >> 4) << 3);
                const int bc = kk + ((lane >> 3) & 1) * 8;
                unsigned int r[4];
                ldsm_x4(r, sb_buf + swz128(bn, bc));
                mma_bf16(acc[f], a, r[0], r[1]);
                mma_bf16(acc[f + 1], a, r[2], r[3]);
            }
            if (wn == 3) {
                const int bn = kRank + (lane & 7);
                const int bc = kk + ((lane >> 3) & 1) * 8;
                unsigned int r0, r1;
                ldsm_x2(r0, r1, sb_buf + swz128(bn, bc));
                mma_bf16(acc[10], a, r0, r1);
            }
        }
    }
}

// Exact upmix_roof epilogue pair-tree for one (row, hl..hl+1) output pair.
__device__ __forceinline__ __nv_bfloat162 mix_pair_gated_v2(
    const float (&gate_i)[kStreams][4],
    int half,
    const __nv_bfloat162 (&h2)[kStreams],
    const float (&r)[kStreams],
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
                const float gate = gate_i[st][half * 2 + e];
                const float x = (__bfloat162float(e == 0 ? h2[st].x : h2[st].y) * r[st]) * w1[st][e];
                pair[p][e] += gate * __bfloat162float(__float2bfloat16(x));
            }
        }
    }
    __nv_bfloat162 out;
    out.x = __float2bfloat16((pair[0][0] + pair[1][0]) * 0.25f);
    out.y = __float2bfloat16((pair[0][1] + pair[1][1]) * 0.25f);
    return out;
}

// Phase 2: Disjoint-H upmix over 10 H-tiles/rank (BH=64, BK=32, 0 cluster.sync(),
// 0 DSMEM, 0 inline NVFP4 dequant). Reads sDown[10][32][32] + sRstd[32][4] from
// SMEM, streams pre-dequantized `up` via 3-stage cp.async ring overlapped across
// H-tile epilogues, stages L2-hot hyper' via cp.async evict_first overlapped with
// MUFU sigmoids, and writes coalesced 16B `mixed` stores with tail-normal policy.
template <bool kAmax>
__device__ __forceinline__ void upmix_phase2_v2(
    unsigned char* smem,
    unsigned int sbase,
    const __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ up,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int* __restrict__ amax_bits,
    int s,
    int t0,
    unsigned int num_tokens,
    float alpha_up) {
    int tid;
    asm volatile("mov.u32 %0, %%tid.x;\n" : "=r"(tid));
    const int warp = tid >> 5;   // 0..15
    const int lane = tid & 31;
    const int wm = warp >> 3;    // 0..1 (16-row slab)
    const int wh = warp & 7;     // 0..7 (8 h-cols x 4 streams)
    const int hl = wh * 8 + 2 * (lane & 3);

    const uint64_t pol_el = pol_evict_last_local();
    const uint64_t pol_ef = pol_evict_first_local();

    const int h0_rank = s * (kHTilesPerRank * kBH2);  // s * 640
    const int full_tile = ((unsigned int)(t0 + kBMF) <= num_tokens) ? 1 : 0;

    // Precompute thread-invariant sB2 cp.async coordinates (2 chunks/thread).
    const int b_part = tid & 3;
    const __nv_bfloat16* up_base[2];
    unsigned int sb_dst_rel[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int idx = tid + j * kThreads;
        const int c = idx >> 2;       // 0..255
        const int st = c >> 6;        // 0..3
        const int h_loc = c & 63;     // 0..63
        up_base[j] = up + ((size_t)st * kHidden + (size_t)(h0_rank + h_loc)) * kRank + b_part * 8;
        sb_dst_rel[j] = sbase + kOffB2F + hc_2cta::swz64(c, b_part * 8);
    }

    // Keep the 3-stage sB2 write address directly in thread registers (no uniform
    // `+ wr_off` operand on cp.async): on sm_121a (CUDA 13.0 ptxas), folding a
    // uniform offset (`[R+UR0]`) into `LDGSTS ... desc[UR]` triggers a ptxas
    // register-allocator bug that emits an illegal odd descriptor pair (`desc[UR1]`,
    // CUDA error 715).
    auto load_b2_at = [&](const __nv_bfloat16* (&tile_base)[2], unsigned int dst0, unsigned int dst1, int elem_off) {
        cp16_hint_u32(dst0, tile_base[0] + elem_off, true, pol_el);
        cp16_hint_u32(dst1, tile_base[1] + elem_off, true, pol_el);
        cp_async_commit_f();
    };

    // Precompute sHyperEpi cp.async coordinates (2 chunks/thread, 8 consecutive lanes = 128B line).
    bool epi_ok[2];
    unsigned int epi_dst[2];
    const __nv_bfloat16* epi_h_base[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int idx = tid + j * kThreads;
        const int piece = idx & 7;
        const int st = (idx >> 3) & 3;
        const int row = idx >> 5;
        const int t = t0 + row;
        epi_ok[j] = t < (int)num_tokens;
        epi_dst[j] = sbase + kOffHyperEpiF + (unsigned int)(row * kHRow2 + st * (kBH2 * 2) + piece * 16);
        epi_h_base[j] = hyper + (size_t)(epi_ok[j] ? t : 0) * kWidth + st * kHidden + piece * 8;
    }

    // Prime the 3-stage sB2 ring with tile 0's first 2 K-stages.
    constexpr unsigned int kMaxStageOff = (unsigned int)((kStagesB2 - 1) * kStageB2F);
    load_b2_at(up_base, sb_dst_rel[0], sb_dst_rel[1], 0);
    load_b2_at(up_base, sb_dst_rel[0] + (unsigned int)kStageB2F, sb_dst_rel[1] + (unsigned int)kStageB2F, kBK2);
    unsigned int rd_off = 0u;
    const unsigned int sb_max[2] = {sb_dst_rel[0] + kMaxStageOff, sb_dst_rel[1] + kMaxStageOff};
    unsigned int sb_wr[2] = {sb_max[0], sb_max[1]};
    constexpr int kUpTileElems = kBH2 * kRank;  // 20,480 bf16

    float amax = 0.0f;

#pragma unroll 1
    for (int ht_local = 0; ht_local < kHTilesPerRank; ++ht_local) {
        const int h0 = h0_rank + ht_local * kBH2;
        const __nv_bfloat16* up_tile[2] = {
            up_base[0] + (size_t)ht_local * kUpTileElems,
            up_base[1] + (size_t)ht_local * kUpTileElems,
        };

        float acc[kStreams][4];
#pragma unroll
        for (int st = 0; st < kStreams; ++st)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[st][e] = 0.0f;

#pragma unroll
        for (int kt = 0; kt < kKSteps2; ++kt) {
            cp_async_wait_f<1>();
            __syncthreads();
            const int nx = kt + 2;
            if (nx < kKSteps2) {
                load_b2_at(up_tile, sb_wr[0], sb_wr[1], nx * kBK2);
            } else if (ht_local + 1 < kHTilesPerRank) {
                load_b2_at(up_tile, sb_wr[0], sb_wr[1], kUpTileElems + (nx - kKSteps2) * kBK2);
            } else {
                cp_async_commit_f();
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                sb_wr[j] = (sb_wr[j] == sb_max[j]) ? sb_dst_rel[j] : (sb_wr[j] + (unsigned int)kStageB2F);
            }

            const unsigned int sa_kt = sbase + (unsigned int)(kt * (kBMF * (kBK2 * 2)));
            const unsigned int sb_slot = sbase + kOffB2F + rd_off;
#pragma unroll
            for (int kk = 0; kk < kBK2; kk += 16) {
                unsigned int a[4];
                ldsm_x4(a, sa_kt + hc_2cta::swz64(wm * 16 + (lane & 15), kk + (lane >> 4) * 8));
                unsigned int b[kStreams][2];
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const int st = 2 * p + (lane >> 4);
                    const int c = st * kBH2 + wh * 8 + (lane & 7);
                    unsigned int r4[4];
                    ldsm_x4(r4, sb_slot + hc_2cta::swz64(c, kk + ((lane >> 3) & 1) * 8));
                    b[2 * p][0] = r4[0];
                    b[2 * p][1] = r4[1];
                    b[2 * p + 1][0] = r4[2];
                    b[2 * p + 1][1] = r4[3];
                }
#pragma unroll
                for (int st = 0; st < kStreams; ++st) {
                    mma_bf16(acc[st], a, b[st][0], b[st][1]);
                }
            }
            rd_off = (rd_off == kMaxStageOff) ? 0u : (rd_off + (unsigned int)kStageB2F);
        }

        // Stage hyper'[t0..t0+31, 0..3, h0..h0+63] into disjoint sHyperEpi with
        // evict_first, overlapped with norm_w loads, MUFU sigmoids, AND ht+1's s.up!
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            cp16_hint_u32(epi_dst[j], epi_h_base[j] + h0, epi_ok[j], pol_ef);
        }
        cp_async_commit_f();

        float w1[kStreams][2];
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
            for (int idx = 0; idx < 4; ++idx) {
                const float logit = acc[st][idx] * alpha_up;
                acc[st][idx] = hc_fast_rcp_rn_1p_exp(logit);
            }
        }
        asm volatile("cp.async.wait_group 0;\n"
                     "bar.sync 0;\n"
                     : "+f"(acc[kStreams - 1][3]) :: "memory");

        if (full_tile) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = wm * 16 + (lane >> 2) + half * 8;
                const float4 r4 = *reinterpret_cast<const float4*>(smem + kOffRstdF + row * 16);
                const float r[kStreams] = {r4.x, r4.y, r4.z, r4.w};
                __nv_bfloat162 h2[kStreams];
#pragma unroll
                for (int st = 0; st < kStreams; ++st) {
                    h2[st] = *reinterpret_cast<const __nv_bfloat162*>(
                        smem + kOffHyperEpiF + row * kHRow2 + st * (kBH2 * 2) + hl * 2);
                }
                const __nv_bfloat162 out = mix_pair_gated_v2(acc, half, h2, r, w1);
                *reinterpret_cast<__nv_bfloat162*>(smem + kOffOutEpiF + row * kORow2 + hl * 2) = out;
                if constexpr (kAmax) {
                    const float2 f = __bfloat1622float2(out);
                    amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
                }
            }
        } else {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = wm * 16 + (lane >> 2) + half * 8;
                const unsigned int token = (unsigned int)(t0 + row);
                if (token >= num_tokens) continue;
                const float4 r4 = *reinterpret_cast<const float4*>(smem + kOffRstdF + row * 16);
                const float r[kStreams] = {r4.x, r4.y, r4.z, r4.w};
                __nv_bfloat162 h2[kStreams];
#pragma unroll
                for (int st = 0; st < kStreams; ++st) {
                    h2[st] = *reinterpret_cast<const __nv_bfloat162*>(
                        smem + kOffHyperEpiF + row * kHRow2 + st * (kBH2 * 2) + hl * 2);
                }
                const __nv_bfloat162 out = mix_pair_gated_v2(acc, half, h2, r, w1);
                *reinterpret_cast<__nv_bfloat162*>(smem + kOffOutEpiF + row * kORow2 + hl * 2) = out;
                if constexpr (kAmax) {
                    const float2 f = __bfloat1622float2(out);
                    amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
                }
            }
        }

        __syncthreads();
        // Coalesced 16B stores (8 lanes = 128B full cache line), tail-normal policy:
        // keep trailing kMixedTailKeepRows in L2 for downstream FP8 quant/gates.
        if (tid < 256) {
            const int row = tid >> 3;
            const int piece = tid & 7;
            const int t = t0 + row;
            if (t < (int)num_tokens) {
                const uint4 v = *reinterpret_cast<const uint4*>(smem + kOffOutEpiF + row * kORow2 + piece * 16);
                __nv_bfloat16* dst = mixed + (size_t)t * kHidden + h0 + piece * 8;
                if (t + kMixedTailKeepRows >= (int)num_tokens) {
                    *reinterpret_cast<uint4*>(dst) = v;
                } else {
                    store_row16_stream(dst, v, pol_ef);
                }
            }
        }
    }

    if constexpr (kAmax) {
        float* warp_max = reinterpret_cast<float*>(smem + kOffWarpMaxF);
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
        if (lane == 0) warp_max[warp] = amax;
        __syncthreads();
        if (tid == 0) {
            float block_max = warp_max[0];
#pragma unroll
            for (int wi = 1; wi < kThreads / 32; ++wi) block_max = fmaxf(block_max, warp_max[wi]);
            const unsigned int bits = __float_as_uint(block_max);
            if (block_max > 0.0f && bits > *reinterpret_cast<volatile unsigned int*>(amax_bits)) {
                atomicMax(amax_bits, bits);
            }
        }
    }
}

// Complete Phase 1 -> SMEM/DSMEM handoff (3 cluster.sync()s) -> Phase 2 (0 cluster.sync()s).
template <bool kInject, bool kL2Hint, bool kAmax, bool kPreloadedB = false, bool kPreloadedH = false,
          typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void prepare_and_upmix_tile_v2(
    unsigned char* smem,
    unsigned int sbase,
    const Operands& op,
    const __nv_bfloat16* __restrict__ folded,
    const __nv_bfloat16* __restrict__ up,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* residual,
    __nv_bfloat16* mixed,
    unsigned int* amax_bits,
    unsigned int num_tokens,
    float scale2_down,
    float scale2_inject,
    float alpha_up,
    float eps,
    float pre_scale = 0.0f,
    const Vec8* pre_rh = nullptr) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    const int s = blockIdx.x;
    const int t0 = blockIdx.y * kBMF;
    const int tid = threadIdx.x;
    float* srstd = reinterpret_cast<float*>(smem + kOffRstdF);
    float acc[11][4];
    if (tid < kProducers) {
        const float r = produce_rows_v2<kInject, kL2Hint, kPreloadedH>(
            smem, op, s, t0, num_tokens, eps, pre_scale, pre_rh);
        if ((tid & 7) == 0) {
            srstd[(tid >> 3) * kStreams + s] = r;
        }
    } else {
        multiply_rows_v2<kPreloadedB>(sbase, folded, s, acc);
    }
    __syncthreads();  // Phase-1 sB1+sA1 ring idle, srstd[*][s] visible

    float* __restrict__ sq = reinterpret_cast<float*>(smem);
    float* __restrict__ srstd_self = reinterpret_cast<float*>(smem + kOffRstdSelfF);
    if (tid < kBMF) {
        srstd_self[tid] = srstd[tid * kStreams + s];
    } else if (tid >= kProducers) {
        const int mt = tid - kProducers;
        const int mw = mt >> 5;
        const int lane = mt & 31;
        const int wm = mw >> 2;
        const int wn = mw & 3;
        const int row0 = wm * 16 + (lane >> 2);
        const int row1 = row0 + 8;
        const float rs0 = srstd[row0 * kStreams + s];
        const float rs1 = srstd[row1 * kStreams + s];
        const int n_base = wn * (kRank / 4);
#pragma unroll
        for (int f = 0; f < 10; ++f) {
            const int n = n_base + f * 8 + 2 * (lane & 3);
            *reinterpret_cast<float2*>(&sq[row0 * kNTotal + n]) =
                make_float2(rs0 * acc[f][0], rs0 * acc[f][1]);
            *reinterpret_cast<float2*>(&sq[row1 * kNTotal + n]) =
                make_float2(rs1 * acc[f][2], rs1 * acc[f][3]);
        }
        if (wn == 3) {
            const int n = kRank + 2 * (lane & 3);
            if (n < kNTotal) {
                *reinterpret_cast<float2*>(&sq[row0 * kNTotal + n]) =
                    make_float2(rs0 * acc[10][0], rs0 * acc[10][1]);
                *reinterpret_cast<float2*>(&sq[row1 * kNTotal + n]) =
                    make_float2(rs1 * acc[10][2], rs1 * acc[10][3]);
            }
        }
    }
    cluster.sync();  // #1: all 4 ranks' sq & srstd_self complete; all 4 ranks done reading op.core

    // Early discard of dead Phase-1 core / moe_rows before Phase 2 starts
    // (skip when aliased with mixed, which Phase 2 is about to overwrite).
    if constexpr (kInject) {
        if (op.core != nullptr && op.core != mixed && ((reinterpret_cast<uintptr_t>(op.core) & 127u) == 0u)) {
            const int row0_d = t0 + s * kMoeRowsF;
            const int valid_d = min(kMoeRowsF, max(0, (int)num_tokens - row0_d));
            discard_rows_128(op.core, row0_d, valid_d);
        }
    }

    // Cross-rank combine: down -> SMEM sDownPart[8][320], inject scales -> residual tail.
    const float2* const sq2 = reinterpret_cast<const float2*>(sq);
    const float2* const rq1 = reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 1) & 3));
    const float2* const rq2 = reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 2) & 3));
    const float2* const rq3 = reinterpret_cast<const float2*>(cluster.map_shared_rank(sq, (s + 3) & 3));
    __nv_bfloat162* const sDownPart = reinterpret_cast<__nv_bfloat162*>(smem + kOffDownPartF);

    constexpr int kPairsPerRow = kNTotal / 2;                   // 162
    constexpr int kPairsPerRank = kMoeRowsF * kPairsPerRow;     // 1296
    for (int idx = tid; idx < kPairsPerRank; idx += kThreads) {
        const int r_local = idx / kPairsPerRow;
        const int p = idx - r_local * kPairsPerRow;
        const int row = s * kMoeRowsF + r_local;
        const int t = t0 + row;
        const unsigned int n = (unsigned int)(2 * p);
        if (t >= (int)num_tokens) {
            if (n < (unsigned int)kRank) {
                *reinterpret_cast<uint32_t*>(&sDownPart[r_local * (kRank / 2) + p]) = 0u;
            }
            continue;
        }
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
        if (n < (unsigned int)kRank) {
            const float x0 = (scale2_down * sum0) * (1.0f / (float)kStreams);
            const float x1 = (scale2_down * sum1) * (1.0f / (float)kStreams);
            const float v0 = hc_fast_div_rn_1p_exp(x0);
            const float v1 = hc_fast_div_rn_1p_exp(x1);
            sDownPart[r_local * (kRank / 2) + p] = __floats2bfloat162_rn(v0, v1);
        } else {
            const float x0 = (scale2_inject * sum0) * (1.0f / (float)kStreams);
            const float x1 = (scale2_inject * sum1) * (1.0f / (float)kStreams);
            const float v0 = 2.0f * hc_fast_rcp_rn_1p_exp(x0);
            const float v1 = 2.0f * hc_fast_rcp_rn_1p_exp(x1);
            *reinterpret_cast<__nv_bfloat162*>(
                &residual[(unsigned long long)t * kWidth + (kWidth - kStreams) + (n - (unsigned int)kRank)]) =
                __floats2bfloat162_rn(v0, v1);
        }
    }
    cluster.sync();  // #2: sq is dead; all 4 ranks' sDownPart and srstd_self are complete

    // Gather all 4 ranks' sDownPart (8x320 bf16 each) into swizzled sDown[10][32][32] at [0..20,480)
    // and all 4 ranks' srstd_self (32 floats each) into sRstd[32][4] at kOffRstdF.
    const uint4* dp_ranks[4];
    dp_ranks[0] = reinterpret_cast<const uint4*>(smem + kOffDownPartF);
    dp_ranks[1] = reinterpret_cast<const uint4*>(cluster.map_shared_rank(smem + kOffDownPartF, (s + 1) & 3));
    dp_ranks[2] = reinterpret_cast<const uint4*>(cluster.map_shared_rank(smem + kOffDownPartF, (s + 2) & 3));
    dp_ranks[3] = reinterpret_cast<const uint4*>(cluster.map_shared_rank(smem + kOffDownPartF, (s + 3) & 3));
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        const int src_rank = (s + rr) & 3;
        const uint4* src_dp = dp_ranks[rr];
        for (int idx = tid; idx < kMoeRowsF * (kRank / 8); idx += kThreads) {
            const int r_local = idx / (kRank / 8);
            const int chunk40 = idx - r_local * (kRank / 8);
            const int row = src_rank * kMoeRowsF + r_local;
            const int ks = chunk40 >> 2;
            const int c4 = chunk40 & 3;
            *reinterpret_cast<uint4*>(smem + ks * (kBMF * (kBK2 * 2)) + hc_2cta::swz64(row, c4 * 8)) = src_dp[idx];
        }
    }
    if (tid < kBMF * kStreams) {
        const int row = tid >> 2;
        const int st = tid & 3;
        const float* src_rstd = reinterpret_cast<const float*>(cluster.map_shared_rank(srstd_self, st));
        srstd[tid] = (t0 + row < (int)num_tokens) ? src_rstd[row] : 0.0f;
    }

    // Make all 4 ranks' Phase-1 hyper' stores in L2 globally visible to every rank's
    // Phase-2 cp.async loads before the final cluster.sync().
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
    asm volatile("fence.proxy.async.global;\n" ::: "memory");
    cluster.sync();  // #3 (FINAL cluster.sync()): sDown & sRstd ready; hyper' visible across cluster

    upmix_phase2_v2<kAmax>(smem, sbase, op.hyper, up, norm_w, mixed, amax_bits, s, t0, num_tokens, alpha_up);
}

template <bool kAmax>
__device__ __forceinline__ void combine_moe_upmix_tile_mx_v2(
    unsigned char* smem,
    unsigned int sbase,
    const MoeRows& m,
    const float* gate_scalars,
    __nv_bfloat16* hyper,
    __nv_bfloat16* residual,
    const __nv_bfloat16* folded,
    const __nv_bfloat16* up,
    const __nv_bfloat16* norm_w,
    __nv_bfloat16* mixed,
    unsigned int* amax_bits,
    unsigned int num_tokens,
    float scale2_down,
    float scale2_inject,
    float alpha_up,
    float eps) {
    // Overlap Phase-1 sB1[0] cp.async with MoE row formation (moe_form_rows_mx16_2w uses 0 SMEM).
    if (threadIdx.x >= kProducers) {
        const __nv_bfloat16* wsrc = folded + (size_t)blockIdx.x * kNTotal * kHidden;
        load_b1(sbase, wsrc, threadIdx.x - kProducers, 0, 0, pol_evict_last_local());
    }
    const int row0_moe = cta_row0_fused();
    moe_form_rows_mx16_2w(m, gate_scalars, row0_moe, num_tokens);

    float pre_scale = 0.0f;
    Vec8 pre_rh[4];
    if (threadIdx.x < kProducers) {
        const int s = blockIdx.x;
        const int t0 = blockIdx.y * kBMF;
        const int pr = threadIdx.x >> 3;
        const int pv = threadIdx.x & 7;
        const int ptoken = t0 + pr;
        const bool prow = ptoken < (int)num_tokens;
        const size_t row_idx = prow ? (size_t)ptoken : 0;
        const __nv_bfloat16* hrow = hyper + row_idx * kWidth + s * kHidden + pv * 8;
        pre_scale = prow ? __bfloat162float(residual[row_idx * kWidth + (kWidth - kStreams) + s]) : 0.0f;
        const uint64_t pol_ef = pol_evict_first_local();
        const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int stg = 0; stg < 4; ++stg) {
            pre_rh[stg].raw = prow ? load_row16_stream<true>(hrow + stg * kBK1, pol_ef) : zero;
        }
    }

    cooperative_groups::this_cluster().sync();  // formed moe_rows visible; all warps done reading shared_out

    // Discard dead shared_out rows after cluster sync (skip if aliased with mixed).
    if (m.shared_out != nullptr && m.shared_out != mixed &&
        ((reinterpret_cast<uintptr_t>(m.shared_out) & 127u) == 0u)) {
        const int valid_sh = min(kMoeRowsF, max(0, (int)num_tokens - row0_moe));
        discard_rows_128(m.shared_out, row0_moe, valid_sh);
    }

    const Operands op{hyper, m.out, residual, nullptr, nullptr};
    prepare_and_upmix_tile_v2<true, true, kAmax, true, true>(
        smem, sbase, op, folded, up, norm_w, residual, mixed, amax_bits, num_tokens,
        scale2_down, scale2_inject, alpha_up, eps, pre_scale, pre_rh);
}

__device__ __forceinline__ unsigned int check_fused_smem(unsigned char* smem) {
    unsigned int dyn;
    asm("mov.u32 %0, %%dynamic_smem_size;\n" : "=r"(dyn));
    if (dyn < (unsigned int)kSmemBytesF) __trap();
    return static_cast<unsigned int>(__cvta_generic_to_shared(smem));
}

}  // namespace hc_cluster_fuse

#define QWEN4_HC_CLUSTER_UPMIX_INJECT_PARAMS                                                                       \
    __nv_bfloat16 *__restrict__ hyper, const __nv_bfloat16 *__restrict__ core, __nv_bfloat16 *residual,            \
        const __nv_bfloat16 *__restrict__ folded, const __nv_bfloat16 *__restrict__ up,                            \
        const __nv_bfloat16 *__restrict__ norm_w, __nv_bfloat16 *__restrict__ mixed, unsigned int num_tokens,      \
        unsigned int rank, unsigned int hidden_size, unsigned int hc, float scale2_down, float scale2_inject,      \
        float alpha_up, float eps

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_upmix_ws_gatepre(QWEN4_HC_CLUSTER_UPMIX_INJECT_PARAMS) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core != nullptr) __trap();
    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned int sbase = hc_cluster_fuse::check_fused_smem(smem);
    const Operands op{hyper, nullptr, residual, nullptr, nullptr};
    hc_cluster_fuse::prepare_and_upmix_tile_v2<false, false, false>(
        smem, sbase, op, folded, up, norm_w, residual, mixed, nullptr, num_tokens,
        scale2_down, scale2_inject, alpha_up, eps);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_upmix_ws_gatepre_amax(QWEN4_HC_CLUSTER_UPMIX_INJECT_PARAMS, unsigned int* __restrict__ amax_bits) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core != nullptr) __trap();
    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned int sbase = hc_cluster_fuse::check_fused_smem(smem);
    const Operands op{hyper, nullptr, residual, nullptr, nullptr};
    hc_cluster_fuse::prepare_and_upmix_tile_v2<false, false, true>(
        smem, sbase, op, folded, up, norm_w, residual, mixed, amax_bits, num_tokens,
        scale2_down, scale2_inject, alpha_up, eps);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_upmix_ws_l2h_gatepre(QWEN4_HC_CLUSTER_UPMIX_INJECT_PARAMS) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core == nullptr) __trap();
    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned int sbase = hc_cluster_fuse::check_fused_smem(smem);
    const Operands op{hyper, core, residual, nullptr, nullptr};
    hc_cluster_fuse::prepare_and_upmix_tile_v2<true, true, false>(
        smem, sbase, op, folded, up, norm_w, residual, mixed, nullptr, num_tokens,
        scale2_down, scale2_inject, alpha_up, eps);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_upmix_ws_l2h_gatepre_amax(QWEN4_HC_CLUSTER_UPMIX_INJECT_PARAMS, unsigned int* __restrict__ amax_bits) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || core == nullptr) __trap();
    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned int sbase = hc_cluster_fuse::check_fused_smem(smem);
    const Operands op{hyper, core, residual, nullptr, nullptr};
    hc_cluster_fuse::prepare_and_upmix_tile_v2<true, true, true>(
        smem, sbase, op, folded, up, norm_w, residual, mixed, amax_bits, num_tokens,
        scale2_down, scale2_inject, alpha_up, eps);
}

#define QWEN4_HC_CLUSTER_UPMIX_MOE_PARAMS                                                                          \
    __nv_bfloat16 *__restrict__ hyper, __nv_bfloat16 *residual, __nv_bfloat16 *moe_rows,                           \
        const uint8_t *__restrict__ expert_output, const uint8_t *__restrict__ slot_scales,                        \
        const int *__restrict__ token_to_perm, const float *__restrict__ topk_weights,                             \
        const __nv_bfloat16 *__restrict__ shared_out, const float *__restrict__ gate_scalars,                      \
        const __nv_bfloat16 *__restrict__ folded, const __nv_bfloat16 *__restrict__ up,                            \
        const __nv_bfloat16 *__restrict__ norm_w, __nv_bfloat16 *__restrict__ mixed, unsigned int num_tokens,      \
        unsigned int rank, unsigned int hidden_size, unsigned int hc, unsigned int topk, float scale2_down,        \
        float scale2_inject, float alpha_up, float eps

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_combine_moe_upmix_mx_gatepre(QWEN4_HC_CLUSTER_UPMIX_MOE_PARAMS) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || topk != kTopk ||
        slot_scales == nullptr || ((reinterpret_cast<uintptr_t>(slot_scales) & 3u) != 0u)) {
        __trap();
    }
    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned int sbase = hc_cluster_fuse::check_fused_smem(smem);
    const MoeRows m{expert_output, slot_scales, token_to_perm, topk_weights, shared_out, nullptr, nullptr, moe_rows};
    hc_cluster_fuse::combine_moe_upmix_tile_mx_v2<false>(
        smem, sbase, m, gate_scalars, hyper, residual, folded, up, norm_w, mixed, nullptr, num_tokens,
        scale2_down, scale2_inject, alpha_up, eps);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_combine_moe_upmix_mx_gatepre_amax(QWEN4_HC_CLUSTER_UPMIX_MOE_PARAMS, unsigned int* __restrict__ amax_bits) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || topk != kTopk ||
        slot_scales == nullptr || ((reinterpret_cast<uintptr_t>(slot_scales) & 3u) != 0u)) {
        __trap();
    }
    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned int sbase = hc_cluster_fuse::check_fused_smem(smem);
    const MoeRows m{expert_output, slot_scales, token_to_perm, topk_weights, shared_out, nullptr, nullptr, moe_rows};
    hc_cluster_fuse::combine_moe_upmix_tile_mx_v2<true>(
        smem, sbase, m, gate_scalars, hyper, residual, folded, up, norm_w, mixed, amax_bits, num_tokens,
        scale2_down, scale2_inject, alpha_up, eps);
}

