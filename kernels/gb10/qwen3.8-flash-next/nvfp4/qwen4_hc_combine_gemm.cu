// SPDX-License-Identifier: AGPL-3.0-only

// HC_FUSED's preparation for pending core or MoE outputs (levers `injgemm`,
// `postpre`, `postsplit`, and single-launch cluster-fused upmix of
// ATLAS_QWEN4_PREFILL_MHC_FUSED): deferred MoE combine (formed once per row
// into `moe_rows`), injection, rstd, per-stream down|inject projections and
// their combine in one 4-CTA cluster kernel, replacing
// qwen4_hc_combine_inject_rstd + 4 cuBLASLt GEMMs + qwen4_hc_combine.
// Byte-identical to that sequence: the MoE output row is formed with those
// kernels' expressions (this target builds with --fmad=false) and the shared
// expert gate with moe_batched_blend's 256-thread dot tree.
//
// Tiling: one CTA per (stream, 64-row tile), the four stream CTAs of a tile
// form a cluster, warp-specialised: warps 0-7 form the hyper' vectors (one
// 8-element vector per thread and K step, loads pipelined ahead), warps 8-15
// stream the folded weights and run every MMA (4 row slabs x 2 output halves
// of 22 n8 fragments). Named barrier 1 hands each K step from one group to
// the other. Grid (4, ceil(T / 64)), block 512, dynamic smem >= kSmemBytes.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// Centralized Blackwell ISA feature guards:
// - Q4_HC_HAS_E2M1_CVT: hardware E2M1/E4M3 cvt requires SM100+ with an
//   arch-specific or family-specific target (excludes Hopper sm_90a and generic PTX).
// - Q4_HC_HAS_SM120_MX_MMA: block-scaled MMA (mxf4nvf4 / mxf8f6f4) requires
//   SM120+ with an arch-specific or family-specific target (excludes SM103 and
//   generic sm_120/sm_121).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    (defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || \
     defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_SPECIFIC__))
#define Q4_HC_HAS_E2M1_CVT 1
#else
#define Q4_HC_HAS_E2M1_CVT 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && \
    (defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || \
     defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_SPECIFIC__))
#define Q4_HC_HAS_SM120_MX_MMA 1
#else
#define Q4_HC_HAS_SM120_MX_MMA 0
#endif

namespace {

constexpr int kHidden = 2560;
constexpr int kStreams = 4;
constexpr int kWidth = kHidden * kStreams;
constexpr int kNTotal = 324;
constexpr int kBM = 64;
constexpr int kBK = 32;
constexpr int kLds = kBK + 8;  // 80-byte rows: conflict-free ldmatrix
constexpr int kSteps = kHidden / kBK;
constexpr int kThreads = 512;
constexpr int kProducers = kBM * (kBK / 8);
constexpr int kFrags = 22;  // per MMA warp: 176 outputs
constexpr int kNPad = 2 * kFrags * 8;
// qwen4_hc_rstd's 64 threads per stream: vector 4q + v (step q, producer v)
// belongs to its thread 4 * (q % 16) + v; the loops are unrolled by 16.
constexpr int kSlots = 16;
constexpr int kTopk = 10;  // Flash-Next routes every token to 10 experts
[[maybe_unused]] constexpr int kColsPerRank = kNTotal / kStreams;
constexpr int kBlendThreads = 256;  // moe_batched_blend's block

static_assert(kProducers == 256 && kThreads - kProducers == 256, "two 8-warp groups");
static_assert(kNPad >= kNTotal && kNTotal % kStreams == 0, "output split");
static_assert(kSteps % kSlots == 0, "the unrolled bodies keep slots and buffers static");

// Folded weights come from L2 two K steps ahead (a three-slot ring); the
// operand rows are produced one step ahead (two slots).
constexpr int kStagesB = 3;
constexpr int kMoeRows = kBM / kStreams;  // rows formed per CTA in postpre/postsplit

// L2 residency windows sized for each Blackwell target's L2 cache:
// - SM120 / SM103 (96 MiB L2): keep up to 4096 full rows (~80 MiB) or 3584
//   trailing rows (~70 MiB) of `hyper'` hot, pin `bm64_w4` and `bm32_*` Phase-1
//   `hyper'` stores with `L2::evict_last` for Phase-2 reuse, and keep 10240
//   trailing `mixed` rows (~50 MiB) L2-normal for downstream projections.
// - SM121 (GB10, 24 MiB L2): retain exact calibrated windows (1024 / 768 /
//   2048 / normal Phase-1 store) unchanged.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1030)) || \
    defined(Q4_HC_FORCE_SM120_L2_TUNING)
constexpr unsigned int kKeepHyperMaxSmallTokens = 4096u;
constexpr int kKeepHyperWindow = 3584;
constexpr int kMixedPinWindow = 10240;
constexpr bool kBm64HyperEvictLast = true;
#else
constexpr unsigned int kKeepHyperMaxSmallTokens = 1024u;
constexpr int kKeepHyperWindow = 768;
constexpr int kMixedPinWindow = 2048;
constexpr bool kBm64HyperEvictLast = false;
#endif
constexpr int kSmemB = kStagesB * kNPad * kLds * 2;
constexpr int kSmemA = 2 * kBM * kLds * 2;
constexpr int kSmemQ = kBM * kNTotal * 4;  // rstd * P tile, over the idle ring
constexpr int kSmemRing = kSmemB + kSmemA > kSmemQ ? kSmemB + kSmemA : kSmemQ;
constexpr int kOffRstd = kSmemRing;
constexpr int kOffPerm = kOffRstd + kBM * 4;
constexpr int kOffWts = kOffPerm + kMoeRows * kTopk * 4;
constexpr int kOffGate = kOffWts + kMoeRows * kTopk * 4;
constexpr int kSmemBytes = kOffGate + kMoeRows * 4;
static_assert(kSmemBytes == 96320, "dynamic smem minimum for 64-row cluster kernels");

template <int kMinBytes>
__device__ __forceinline__ void check_dyn_smem() {
    unsigned int dyn;
    asm("mov.u32 %0, %%dynamic_smem_size;\n" : "=r"(dyn));
    if (dyn < static_cast<unsigned int>(kMinBytes)) __trap();
}

union Vec8 {
    uint4 raw;
    __nv_bfloat16 h[8];
};

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

// All but the newest committed group have landed.
__device__ __forceinline__ void cp_async_wait_prev() { asm volatile("cp.async.wait_group 1;\n" ::); }

// 16 bytes of a hyper / core / expert row; with kL2Hint the load also pulls
// the surrounding 256 bytes of the row into L2 (the next K steps' vectors).
template <bool kL2Hint>
__device__ __forceinline__ uint4 load_row16(const __nv_bfloat16* p) {
    if constexpr (kL2Hint) {
        uint4 v;
        asm volatile("ld.global.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(p));
        return v;
    } else {
        return *reinterpret_cast<const uint4*>(p);
    }
}

// L2 eviction priority: the folded weights (6.33 MiB, re-read by
// every CTA) are kept, the once-read / once-written hyper, core and expert
// rows are evicted first, so the stream stops pushing the weights out to
// DRAM (micro11: +41% DRAM reads without it). Changes no value.
__device__ __forceinline__ uint64_t policy_evict_last() {
    uint64_t p;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}

__device__ __forceinline__ uint64_t policy_evict_first() {
    uint64_t p;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}

__device__ __forceinline__ void cp_async16_keep(void* smem, const void* gmem, bool pred, uint64_t pol) {
    const unsigned int s = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n" ::"r"(s), "l"(gmem),
                 "r"(bytes), "l"(pol));
}

__device__ __forceinline__ float hc_div_approx(float a, float b) {
    float q;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(q) : "f"(a), "f"(b));
    return q;
}

__device__ __forceinline__ float hc_fast_rcp_rn_1p_exp(float x) {
    float y;
    asm("{\n\t"
        ".reg .f32 z, t, den;\n\t"
        "mul.rn.ftz.f32 z, %1, 0fbfb8aa3b;\n\t"
        "ex2.approx.ftz.f32 t, z;\n\t"
        "add.rn.ftz.f32 den, t, 0f3f800000;\n\t"
        "rcp.approx.ftz.f32 %0, den;\n\t"
        "}"
        : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float hc_fast_div_rn_1p_exp(float x) {
    float q;
    asm("{\n\t"
        ".reg .f32 z, t, den;\n\t"
        "mul.rn.ftz.f32 z, %1, 0fbfb8aa3b;\n\t"
        "ex2.approx.ftz.f32 t, z;\n\t"
        "add.rn.ftz.f32 den, t, 0f3f800000;\n\t"
        "div.approx.ftz.f32 %0, %1, den;\n\t"
        "}"
        : "=f"(q) : "f"(x));
    return q;
}

template <bool kL2Hint>
__device__ __forceinline__ uint4 load_row16_stream(const __nv_bfloat16* p, uint64_t pol) {
    uint4 v;
    if constexpr (kL2Hint) {
        asm volatile("ld.global.L2::cache_hint.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(p), "l"(pol));
    } else {
        asm volatile("ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(p), "l"(pol));
    }
    return v;
}

__device__ __forceinline__ void store_row16_stream(__nv_bfloat16* p, uint4 v, uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1,%2,%3,%4}, %5;\n" ::"l"(p), "r"(v.x), "r"(v.y),
                 "r"(v.z), "r"(v.w), "l"(pol)
                 : "memory");
}

// One K step handed from the producers to the MMA warps (all 512 threads).
__device__ __forceinline__ void pipe_barrier() { asm volatile("bar.sync 1, 512;\n" ::: "memory"); }

__device__ __forceinline__ void ldmatrix_x4(unsigned int (&r)[4], const void* p) {
    const unsigned int s = static_cast<unsigned int>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const unsigned int (&a)[4],
                                         unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

struct Operands {
    __nv_bfloat16* hyper;
    const __nv_bfloat16* core;  // pending core injection (inject mode), or null
    const __nv_bfloat16* residual;
    const __nv_bfloat16* expert_output;
    const __nv_bfloat16* shared_out;
};

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
    if (n0 + 320 < kNPad) {
        const int n = n0 + 320;
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
    const bool ok5 = (n0 + 320) < kNTotal;
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
        if (n0 + 320 < kNPad) {
            const int n = n0 + 320;
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

template <bool kL2Hint, bool kPipe3 = false, bool kPreloadedB = false, bool kPreloadedH = kPreloadedB>
__device__ __forceinline__ void prepare_tile(unsigned char* smem, const Operands& op,
                                             const __nv_bfloat16* __restrict__ folded, const Outputs& out,
                                             unsigned int num_tokens, unsigned int hidden_size, float eps,
                                             float pre_scale = 0.0f, const Vec8* pre_rh = nullptr) {
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
template <bool kMx>
__device__ __forceinline__ void combine_moe_tile(unsigned char* smem, const MoeRows& m,
                                                 __nv_bfloat16* hyper, __nv_bfloat16* residual,
                                                 const __nv_bfloat16* folded, const Outputs& out,
                                                 unsigned int num_tokens, float eps) {
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
template <bool kMx>
__device__ __forceinline__ void moe_rows_block(__nv_bfloat16* moe_rows, const void* slots,
                                               const uint8_t* slot_scales, const int* token_to_perm,
                                               const float* topk_weights, const __nv_bfloat16* shared_out,
                                               const __nv_bfloat16* gate_input,
                                               const __nv_bfloat16* shared_gate, unsigned int num_tokens) {
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

namespace {

template <bool kFullUnroll = false, bool kDiscardSlots = false>
__device__ __forceinline__ void moe_form_rows_mx16(const MoeRows& m, const float* gate_scalars, int row0,
                                                   unsigned int num_tokens) {
    const int r = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int t = row0 + r;
    if (r >= kMoeRows || t >= (int)num_tokens) return;
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
        sh0.raw = load_row16_stream<false>(sh_row + h0, pol);
        sh1.raw = load_row16_stream<false>(sh_row + h0 + 8, pol);
        float acc0[8], acc1[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] = 0.0f;
            acc1[i] = 0.0f;
        }
        if constexpr (kFullUnroll) {
            uint4 raw[kTopk];
#pragma unroll
            for (int k = 0; k < kTopk; ++k) {
                const size_t sk = (size_t)(uint32_t)__shfl_sync(0xffffffffu, my_perm, k);
                raw[k] = load_row16_stream<false>(
                    reinterpret_cast<const __nv_bfloat16*>(static_cast<const uint8_t*>(m.slots) + sk * kHidden + h0),
                    pol);
            }
#pragma unroll
            for (int k = 0; k < kTopk; ++k) {
                const uint32_t ek = (__shfl_sync(0xffffffffu, s_word[k], word_lane) >> byte_shift) & 0xffu;
                const float wk = __shfl_sync(0xffffffffu, my_w, k);
                const float ws = wk * __uint_as_float(ek << 23);
                decode_slot8_acc_ws<true>(make_uint4(raw[k].x, raw[k].y, 0, 0), ws, acc0);
                decode_slot8_acc_ws<true>(make_uint4(raw[k].z, raw[k].w, 0, 0), ws, acc1);
            }
        } else {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                uint4 raw[5];
#pragma unroll
                for (int j = 0; j < 5; ++j) {
                    const int k = half * 5 + j;
                    const size_t sk = (size_t)(uint32_t)__shfl_sync(0xffffffffu, my_perm, k);
                    raw[j] = load_row16_stream<false>(
                        reinterpret_cast<const __nv_bfloat16*>(static_cast<const uint8_t*>(m.slots) + sk * kHidden + h0),
                        pol);
                }
#pragma unroll
                for (int j = 0; j < 5; ++j) {
                    const int k = half * 5 + j;
                    const uint32_t ek = (__shfl_sync(0xffffffffu, s_word[k], word_lane) >> byte_shift) & 0xffu;
                    const float wk = __shfl_sync(0xffffffffu, my_w, k);
                    const float ws = wk * __uint_as_float(ek << 23);
                    decode_slot8_acc_ws<true>(make_uint4(raw[j].x, raw[j].y, 0, 0), ws, acc0);
                    decode_slot8_acc_ws<true>(make_uint4(raw[j].z, raw[j].w, 0, 0), ws, acc1);
                }
            }
        }
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
    if constexpr (kDiscardSlots) {
        if ((reinterpret_cast<uintptr_t>(m.slots) & 127u) == 0u) {
            __syncwarp();
            int dlane;
            asm volatile("mov.u32 %0, %%laneid;\n" : "=r"(dlane));
#pragma unroll
            for (int k = 0; k < kTopk; ++k) {
                const size_t sk = (size_t)(uint32_t)__shfl_sync(0xffffffffu, my_perm, k);
                if (dlane < 20) {
                    const unsigned char* p =
                        static_cast<const unsigned char*>(m.slots) + sk * kHidden + (size_t)dlane * 128;
                    asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
                }
            }
        }
    }
}

template <bool kMx>
__device__ __forceinline__ void moe_prologue_gatepre(const MoeRows& m, const float* gate_scalars,
                                                     const MoeScratch& sc, int row0,
                                                     unsigned int num_tokens) {
    if constexpr (kMx) {
        (void)sc;
        moe_form_rows_mx16<true, true>(m, gate_scalars, row0, num_tokens);
    } else {
        moe_routing(m, sc, row0, num_tokens);
        if (threadIdx.x >= kMoeRows * kTopk && threadIdx.x < kMoeRows * kTopk + kMoeRows) {
            const int r = (int)threadIdx.x - kMoeRows * kTopk;
            const int t = row0 + r;
            sc.gate[r] = (t < (int)num_tokens) ? gate_scalars[t] : 0.0f;
        }
        __syncthreads();  // routing and gates visible
        moe_form_rows<false, true>(m, sc, row0, num_tokens);
    }
}

template <bool kMx>
__device__ __forceinline__ void combine_moe_tile_gatepre(unsigned char* smem, const MoeRows& m,
                                                         const float* gate_scalars, __nv_bfloat16* hyper,
                                                         __nv_bfloat16* residual, const __nv_bfloat16* folded,
                                                         const Outputs& out, unsigned int num_tokens, float eps) {
    if (threadIdx.x >= kProducers) {
        preload_b_stage(smem, folded, blockIdx.x, 0, 0);
        preload_b_stage(smem, folded, blockIdx.x, 1, 1);
    }
    const MoeScratch sc{reinterpret_cast<int*>(smem + kOffPerm), reinterpret_cast<float*>(smem + kOffWts),
                        reinterpret_cast<float*>(smem + kOffGate), reinterpret_cast<float*>(smem)};
    moe_prologue_gatepre<kMx>(m, gate_scalars, sc, (int)(blockIdx.y * kBM + blockIdx.x * kMoeRows), num_tokens);
    float pre_scale = 0.0f;
    Vec8 pre_rh[4];
    if constexpr (kMx) {
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
    }
    cooperative_groups::this_cluster().sync();
    const Operands op{hyper, m.out, residual, nullptr, nullptr};
    prepare_tile<true, true, true, kMx>(smem, op, folded, out, num_tokens, kHidden, eps, pre_scale, pre_rh);
    const int row0_discard = (int)(blockIdx.y * kBM + blockIdx.x * kMoeRows);
    const int valid_rows = min(kMoeRows, max(0, (int)num_tokens - row0_discard));
    unsigned char* base_discard = reinterpret_cast<unsigned char*>(m.out + (size_t)row0_discard * kHidden);
    const unsigned char* sh_discard = reinterpret_cast<const unsigned char*>(m.shared_out + (size_t)row0_discard * kHidden);
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

template <bool kMx>
__device__ __forceinline__ void moe_rows_block_gatepre(__nv_bfloat16* moe_rows, const void* slots,
                                                       const uint8_t* slot_scales, const int* token_to_perm,
                                                       const float* topk_weights, const __nv_bfloat16* shared_out,
                                                       const float* gate_scalars, unsigned int num_tokens) {
    const MoeRows m{slots, slot_scales, token_to_perm, topk_weights, shared_out, nullptr, nullptr, moe_rows};
    if constexpr (kMx) {
        moe_form_rows_mx16(m, gate_scalars, (int)(blockIdx.x * kMoeRows), num_tokens);
    } else {
        __shared__ int perm[kMoeRows * kTopk];
        __shared__ float wts[kMoeRows * kTopk];
        __shared__ float gate[kMoeRows];
        moe_prologue_gatepre<false>(m, gate_scalars, MoeScratch{perm, wts, gate, nullptr}, (int)(blockIdx.x * kMoeRows),
                                    num_tokens);
    }
}

}  // namespace

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_gemm_ws_gatepre(
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
    prepare_tile<false, true>(smem, op, folded, out, num_tokens, hidden_size, eps);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_gemm_ws_l2h_gatepre(
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
    prepare_tile<true, true>(smem, op, folded, out, num_tokens, hidden_size, eps);
    if (core != nullptr && ((reinterpret_cast<uintptr_t>(core) & 127u) == 0u)) {
        const int row0_discard = (int)(blockIdx.y * kBM + blockIdx.x * kMoeRows);
        const int valid_rows = min(kMoeRows, max(0, (int)num_tokens - row0_discard));
        const unsigned char* base_discard = reinterpret_cast<const unsigned char*>(core + (size_t)row0_discard * kHidden);
        const int total_lines = valid_rows * (kHidden * (int)sizeof(__nv_bfloat16) / 128);
        for (int line = threadIdx.x; line < total_lines; line += kThreads) {
            const unsigned char* p = base_discard + (size_t)line * 128;
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
        }
    }
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_combine_moe_gemm_gatepre(
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
    check_dyn_smem<kSmemBytes>();
    extern __shared__ __align__(16) unsigned char smem[];
    const MoeRows m{expert_output, slot_scales, token_to_perm, topk_weights, shared_out, nullptr, nullptr, moe_rows};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    combine_moe_tile_gatepre<false>(smem, m, gate_scalars, hyper, residual, folded, out, num_tokens, eps);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_combine_moe_gemm_mx_gatepre(
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
    check_dyn_smem<kSmemBytes>();
    extern __shared__ __align__(16) unsigned char smem[];
    const MoeRows m{expert_output, slot_scales, token_to_perm, topk_weights, shared_out, nullptr, nullptr, moe_rows};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    combine_moe_tile_gatepre<true>(smem, m, gate_scalars, hyper, residual, folded, out, num_tokens, eps);
}

extern "C" __global__ void __launch_bounds__(kThreads, 2)
qwen4_hc_moe_rows_gatepre(
    __nv_bfloat16* __restrict__ moe_rows,
    const __nv_bfloat16* __restrict__ expert_output,
    const uint8_t* __restrict__ slot_scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk) {
    if (hidden_size != kHidden || topk != kTopk) __trap();
    moe_rows_block_gatepre<false>(moe_rows, expert_output, slot_scales, token_to_perm, topk_weights, shared_out,
                                  gate_scalars, num_tokens);
}

extern "C" __global__ void __launch_bounds__(kThreads, 2)
qwen4_hc_moe_rows_mx_gatepre(
    __nv_bfloat16* __restrict__ moe_rows,
    const uint8_t* __restrict__ expert_output,
    const uint8_t* __restrict__ slot_scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk) {
    if (hidden_size != kHidden || topk != kTopk ||
        slot_scales == nullptr || ((reinterpret_cast<uintptr_t>(slot_scales) & 3u) != 0u)) {
        __trap();
    }
    moe_rows_block_gatepre<true>(moe_rows, expert_output, slot_scales, token_to_perm, topk_weights, shared_out,
                                 gate_scalars, num_tokens);
}

namespace hc_2cta {

constexpr int kBM2 = 32;
constexpr int kThreads2 = 256;
constexpr int kProd2 = 128;
constexpr int kFrags2 = 21;
constexpr int kNPad2 = 2 * kFrags2 * 8;                    // 336
constexpr int kStagesB2 = 2;
constexpr int kRowBytes2 = 64;
constexpr int kSmemB2 = kStagesB2 * kNPad2 * kRowBytes2;   // 43,008 B
constexpr int kSmemA2 = 2 * kBM2 * kRowBytes2;             //  4,096 B
constexpr int kOffRstd2 = kSmemB2 + kSmemA2;               // 47,104 B
constexpr int kSmemBytes2 = kOffRstd2 + kBM2 * 4;          // 47,232 B
constexpr int kMoeRows2 = kBM2 / kStreams;                 // 8
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
        const int n = n0 + 320;
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

template <bool kL2Hint, bool kPreloadedB = false, bool kPreloadedH = false, bool kAlwaysInject = false>
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

constexpr int kRank = 320;
constexpr int kBMF = 32;
constexpr int kMoeRowsF = kBMF / kStreams;                     // 8 rows formed per CTA in PostPreMx
constexpr int kBK1 = 64;                                       // 128 B/row (full L2 cache line per 8 lanes)
constexpr int kSteps1 = kHidden / kBK1;                        // 40 K-steps
constexpr int kSlots1 = 8;                                     // 8 producer threads/row x 8 slots = 64 rstd lanes
constexpr int kNPad1 = 328;                                    // 3 x 80 + 88 = 328 (>= kNTotal=324, mult of 8)
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
    const int base_h0 = w_sub * 1280;
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
        const int n = n0 + 320;
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
    const int n_base = wn * 80;
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
                const int bn = 320 + (lane & 7);
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
template <bool kInject, bool kL2Hint, bool kAmax, bool kPreloadedB = false, bool kPreloadedH = false>
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
        const int n_base = wn * 80;
#pragma unroll
        for (int f = 0; f < 10; ++f) {
            const int n = n_base + f * 8 + 2 * (lane & 3);
            *reinterpret_cast<float2*>(&sq[row0 * kNTotal + n]) =
                make_float2(rs0 * acc[f][0], rs0 * acc[f][1]);
            *reinterpret_cast<float2*>(&sq[row1 * kNTotal + n]) =
                make_float2(rs1 * acc[f][2], rs1 * acc[f][3]);
        }
        if (wn == 3) {
            const int n = 320 + 2 * (lane & 3);
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
        if (n0 >= kStreams && (n0 + 320) < kNPad) {
            const unsigned int z0 = sbase + (unsigned int)(((n0 + 320) * kLds + part8) * 2);
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

template <bool kL2Hint, bool kPreloadedH = false>
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

template <bool kMx>
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

constexpr int kRank2 = 320;
constexpr int kBH2 = 32;
constexpr int kBK2 = 64;
constexpr int kKSteps2 = kRank2 / kBK2;  // 5
constexpr int kBN2 = kStreams * kBH2;    // 128

// Phase 1 DSMEM handoff regions (after sQ [0..82944), before sRstd [94720..94976))
constexpr int kOffDownOwn = 83200;
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
    const bool ok5 = (n0 + 320) < kNTotal;
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
    if (n0 + 320 < kNPad) {
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
    const bool ok5 = (n0 + 320) < kNTotal;
    const bool pad5 = (n0 + 320) < kNPad;
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

template <bool kAmax, bool kMxfp8Gate, bool kCoreNormalL2 = false, bool kPreloadedB = false, bool kPreloadedH = false>
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
template <bool kAmax>
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
constexpr int kRank32 = 320;
constexpr int kSmemBytes = hc_2cta::kSmemBytes2;  // 47,232 B (2 CTAs/SM exact)

// Phase 1 DSMEM handoff inside [41472 .. 47104) (between sQ [0..41472) and sRstd [47104..47232))
constexpr int kOffDownOwn = 41472;    // temp BF16: 8 * 320 * 2 = 5,120 B -> 46,592 <= 47,104
constexpr int kOffDownFp4 = 41472;    // quantized E2M1: 8 * 160 = 1,280 B -> 42,752
constexpr int kOffDownSc = 42752;     // quantized UE4M3: 8 * 20 = 160 B -> 42,912
constexpr int kOffDownS2 = 42912;     // per-row FP32 scale2_a: 8 * 4 = 32 B -> 42,944
static_assert(kOffDownOwn + (kBM32 / kStreams) * kRank32 * 2 <= hc_2cta::kOffRstd2,
              "bm32_nvfp4 temp sDownOwn fits between sQ and sRstd");

// Phase 2 SMEM layout (active after DSMEM copy + cluster.sync())
constexpr int kOffA4 = 0;                 // 32 * 160 = 5,120 B
constexpr int kOffSA4 = 5120;             // 32 * 20  = 640 B   -> 5,760
constexpr int kOffS2A4 = 5760;            // 32 * 4   = 128 B   -> 5,888
constexpr int kOffRstd4 = 5888;           // 32 * 16  = 512 B   -> 6,400
constexpr int kOffBp4 = 6400;             // 128 * 160 = 20,480 B -> 26,880
constexpr int kOffBs4 = 26880;            // 128 * 20  = 2,560 B  -> 29,440
constexpr int kOffH4 = 29440;             // 32 * 4 * 32 * 2 = 8,192 B -> 37,632
constexpr int kOffO4 = 37632;             // 32 * 32 * 2 = 2,048 B -> 39,680
constexpr int kOffWarpDot4 = 39680;       // 2 * 32 * 4 = 256 B -> 39,936
constexpr int kOffDotAcc4 = 39936;        // 4 warps * 8 rows * 32 lanes * 4 = 4,096 B -> 44,032 <= 47,232
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

template <bool kAmax, bool kMxfp8Gate, bool kCoreNormalL2 = false, bool kPreloadedB = false, bool kPreloadedH = false>
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
        for (int idx = tid; idx < 320; idx += kThreads32) {
            const int row = idx / 10;
            const int c = idx - row * 10;
            const int sr = row >> 3;
            const int rl = row & 7;
            const uint4* src = (sr == 0) ? fp0 : ((sr == 1) ? fp1 : ((sr == 2) ? fp2 : fp3));
            reinterpret_cast<uint4*>(smem + kOffA4)[idx] = src[rl * 10 + c];
        }
        if (tid < 160) {
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
        for (int idx = tid; idx < 1280; idx += kThreads32) {
            const int br = idx / 10;
            const int c = idx - br * 10;
            const int bst = br >> 5;
            const int bh = br & 31;
            const uint8_t* src = up_packed + ((size_t)bst * kHidden + h0 + (size_t)bh) * 160u + (size_t)(c * 16);
            hc_bm64_upmix::cp16_hint_u32(sbase + kOffBp4 + (unsigned int)(idx * 16), src, true, pol_el);
        }
        cp_async_commit();
    };
    auto load_up_scales = [&](int ht) {
        const uint64_t pol_el = policy_evict_last();
        const size_t h0 = (size_t)ht * kBH32;
        for (int idx = tid; idx < 640; idx += kThreads32) {
            const int br = idx / 5;
            const int c = idx - br * 5;
            const int bst = br >> 5;
            const int bh = br & 31;
            const uint32_t* src = reinterpret_cast<const uint32_t*>(
                up_scales + ((size_t)bst * kHidden + h0 + (size_t)bh) * 20u + (size_t)(c * 4));
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
                const unsigned int a0 = *reinterpret_cast<const unsigned int*>(smem + kOffA4 + fr0 * 160 + byte_base);
                const unsigned int a1 = *reinterpret_cast<const unsigned int*>(smem + kOffA4 + fr1 * 160 + byte_base);
                const unsigned int a2 = *reinterpret_cast<const unsigned int*>(smem + kOffA4 + fr0 * 160 + byte_base + 16);
                const unsigned int a3 = *reinterpret_cast<const unsigned int*>(smem + kOffA4 + fr1 * 160 + byte_base + 16);
                const unsigned int sfa = *reinterpret_cast<const unsigned int*>(smem + kOffSA4 + m_sfa * 20 + sc_base);
#pragma unroll
                for (int st = 0; st < kStreams; ++st) {
                    const int nc = st * 32 + wh * 8 + group_id;
                    const unsigned int b0 = *reinterpret_cast<const unsigned int*>(smem + kOffBp4 + nc * 160 + byte_base);
                    const unsigned int b1 = *reinterpret_cast<const unsigned int*>(smem + kOffBp4 + nc * 160 + byte_base + 16);
                    const unsigned int sfb = *reinterpret_cast<const unsigned int*>(smem + kOffBs4 + nc * 20 + sc_base);
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

template <bool kAmax>
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
constexpr int kRank32 = 320;
constexpr int kSmemBytes = hc_2cta::kSmemBytes2;   // 47,232 B (2 CTAs/SM exact)

// Phase 1 DSMEM handoff inside [41472 .. 47104) (between sQ [0..41472) and sRstd [47104..47232))
constexpr int kOffDownOwn = 41472;    // temp BF16: 8 * 320 * 2 = 5,120 B -> 46,592 <= 47,104
constexpr int kOffDownFp8 = 41472;    // quantized E4M3: 8 * 320 = 2,560 B -> 44,032
constexpr int kOffDownSc8 = 44032;    // quantized UE8M0 (10 padded to 12 B/row): 8 * 12 = 96 B -> 44,128
static_assert(kOffDownOwn + (kBM32 / kStreams) * kRank32 * 2 <= hc_2cta::kOffRstd2,
              "bm32_mxfp8 temp sDownOwn fits between sQ and sRstd");
static_assert(kOffDownSc8 + (kBM32 / kStreams) * 12 <= hc_2cta::kOffRstd2,
              "bm32_mxfp8 quantized handoff fits between sQ and sRstd");

// Phase 2 SMEM layout (active after DSMEM copy + cluster.sync(); exact 47,232 B!)
constexpr int kOffA8 = 0;             // 32 * 320 = 10,240 B -> 10,240
constexpr int kOffSA8 = 10240;        // 32 * 12  = 384 B    -> 10,624
constexpr int kOffRstd8 = 10624;      // 32 * 16  = 512 B    -> 11,136
constexpr int kOffB8 = 11136;         // 128 * 160 = 20,480 B (1 K-half of up in E4M3) -> 31,616
constexpr int kOffSB8 = 31616;        // 128 * 8  = 1,024 B  (5 UE8M0 padded to 8 B/row) -> 32,640
constexpr int kOffH8 = 32640;         // 32 * 4 * 32 * 2 = 8,192 B -> 40,832
constexpr int kOffO8 = 40832;         // 32 * 32 * 2 = 2,048 B -> 42,880
constexpr int kOffWarpDot8 = 42880;   // 2 * 32 * 4 = 256 B -> 43,136
constexpr int kOffDotAcc8 = 43136;    // 4 warps * 8 rows * 32 lanes * 4 = 4,096 B -> 47,232
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

template <bool kAmax, bool kMxfp8Gate, bool kCoreNormalL2 = false, bool kPreloadedB = false, bool kPreloadedH = false>
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
        for (int idx = tid; idx < 640; idx += kThreads32) {
            const int row = idx / 20;
            const int c = idx - row * 20;
            const int sr = row >> 3;
            const int rl = row & 7;
            const uint4* src = (sr == 0) ? fp0 : ((sr == 1) ? fp1 : ((sr == 2) ? fp2 : fp3));
            reinterpret_cast<uint4*>(smem + kOffA8)[idx] = src[rl * 20 + c];
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
        const size_t pk_kh_off = (size_t)kh * 80u;
        const size_t sc_kh_off = (size_t)kh * 10u;
#pragma unroll 1
        for (int idx = tid; idx < 640; idx += kThreads32) {
            const int br = idx / 5;
            const int kb = idx - br * 5;
            const int bst = br >> 5;
            const int bh = br & 31;
            const size_t row_id = (size_t)bst * kHidden + h0 + (size_t)bh;
            const uint8_t* p_src = up_packed + row_id * 160u + pk_kh_off + (size_t)kb * 16u;
            const uint8_t* s_src = up_scales + row_id * 20u + sc_kh_off + (size_t)kb * 2u;
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
            const unsigned int dst_b = sbase + kOffB8 + (unsigned int)(br * 160 + kb * 32);
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
                const int a_k_base = kh * 160;
                const int sa_k_base = kh * 5;
#pragma unroll 1
                for (int kb = 0; kb < 5; ++kb) {
                    const int b_byte = kb * 32 + tid4 * 4;
                    const int a_byte = a_k_base + b_byte;
                    const unsigned int a0 = *reinterpret_cast<const unsigned int*>(smem + kOffA8 + fr0 * 320 + a_byte);
                    const unsigned int a1 = *reinterpret_cast<const unsigned int*>(smem + kOffA8 + fr1 * 320 + a_byte);
                    const unsigned int a2 = *reinterpret_cast<const unsigned int*>(smem + kOffA8 + fr0 * 320 + a_byte + 16);
                    const unsigned int a3 = *reinterpret_cast<const unsigned int*>(smem + kOffA8 + fr1 * 320 + a_byte + 16);
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
                        const unsigned int b0 = *reinterpret_cast<const unsigned int*>(smem + kOffB8 + nc * 160 + b_byte);
                        const unsigned int b1 = *reinterpret_cast<const unsigned int*>(smem + kOffB8 + nc * 160 + b_byte + 16);
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

template <bool kAmax>
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




