// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// Shared ISA guards, geometry/SMEM layout, PTX helpers, and Operands struct for mHC cluster GEMM kernels.

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

template <typename Shape = Qwen4FlashNextShape>
struct HcBaseGeom {
    static constexpr int kHidden = Shape::kHidden;
    static constexpr int kStreams = Shape::kMhcStreams;
    static constexpr int kWidth = Shape::kMhcWidth;
    static constexpr int kRank = Shape::kMhcRank;
    static constexpr int kNTotal = Shape::kMhcProjN;
    static constexpr int kBM = 64;
    static constexpr int kBK = 32;
    static constexpr int kLds = kBK + 8;  // 80-byte rows: conflict-free ldmatrix
    static constexpr int kSteps = kHidden / kBK;
    static constexpr int kThreads = 512;
    static constexpr int kProducers = kBM * (kBK / 8);
    static constexpr int kFrags = ((kNTotal + 31) / 32) * 2;  // per MMA warp: 176 outputs for N=324
    static constexpr int kNPad = 2 * kFrags * 8;
    // qwen4_hc_rstd's 64 threads per stream: vector 4q + v (step q, producer v)
    // belongs to its thread 4 * (q % 16) + v; the loops are unrolled by 16.
    static constexpr int kSlots = 16;
    static constexpr int kTopk = Shape::kTopK;
    static constexpr int kColsPerRank = kNTotal / kStreams;
    static constexpr int kBlendThreads = 256;  // moe_batched_blend's block
    static constexpr int kStagesB = 3;
    static constexpr int kMoeRows = kBM / kStreams;  // rows formed per CTA in postpre/postsplit

    static_assert(kProducers == 256 && kThreads - kProducers == 256, "two 8-warp groups");
    static_assert(kNPad >= kNTotal && kNTotal % kStreams == 0, "output split");
    static_assert(kSteps % kSlots == 0, "the unrolled bodies keep slots and buffers static");

    static constexpr int kSmemB = kStagesB * kNPad * kLds * 2;
    static constexpr int kSmemA = 2 * kBM * kLds * 2;
    static constexpr int kSmemQ = kBM * kNTotal * 4;  // rstd * P tile, over the idle ring
    static constexpr int kSmemRing = kSmemB + kSmemA > kSmemQ ? kSmemB + kSmemA : kSmemQ;
    static constexpr int kOffRstd = kSmemRing;
    static constexpr int kOffPerm = kOffRstd + kBM * 4;
    static constexpr int kOffWts = kOffPerm + kMoeRows * kTopk * 4;
    static constexpr int kOffGate = kOffWts + kMoeRows * kTopk * 4;
    static constexpr int kSmemBytes = kOffGate + kMoeRows * 4;
};

using HcDefaultG = HcBaseGeom<Qwen4FlashNextShape>;

constexpr int kHidden = HcDefaultG::kHidden;
constexpr int kStreams = HcDefaultG::kStreams;
constexpr int kWidth = HcDefaultG::kWidth;
constexpr int kRankBase = HcDefaultG::kRank;
constexpr int kNTotal = HcDefaultG::kNTotal;
constexpr int kBM = HcDefaultG::kBM;
constexpr int kBK = HcDefaultG::kBK;
constexpr int kLds = HcDefaultG::kLds;
constexpr int kSteps = HcDefaultG::kSteps;
constexpr int kThreads = HcDefaultG::kThreads;
constexpr int kProducers = HcDefaultG::kProducers;
constexpr int kFrags = HcDefaultG::kFrags;
constexpr int kNPad = HcDefaultG::kNPad;
constexpr int kSlots = HcDefaultG::kSlots;
constexpr int kTopk = HcDefaultG::kTopk;
[[maybe_unused]] constexpr int kColsPerRank = HcDefaultG::kColsPerRank;
constexpr int kBlendThreads = HcDefaultG::kBlendThreads;

// Folded weights come from L2 two K steps ahead (a three-slot ring); the
// operand rows are produced one step ahead (two slots).
constexpr int kStagesB = HcDefaultG::kStagesB;
constexpr int kMoeRows = HcDefaultG::kMoeRows;

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
constexpr int kSmemB = HcDefaultG::kSmemB;
constexpr int kSmemA = HcDefaultG::kSmemA;
constexpr int kSmemQ = HcDefaultG::kSmemQ;
constexpr int kSmemRing = HcDefaultG::kSmemRing;
constexpr int kOffRstd = HcDefaultG::kOffRstd;
constexpr int kOffPerm = HcDefaultG::kOffPerm;
constexpr int kOffWts = HcDefaultG::kOffWts;
constexpr int kOffGate = HcDefaultG::kOffGate;
constexpr int kSmemBytes = HcDefaultG::kSmemBytes;
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
}  // namespace
