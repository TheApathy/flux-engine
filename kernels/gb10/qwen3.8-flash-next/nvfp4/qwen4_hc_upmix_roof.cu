// SPDX-License-Identifier: AGPL-3.0-only

// HC_FUSED's up+mix (qwen4_hc_upmix_hcfused[_amax], qwen4_hc_upmix.cu) with
// the same arithmetic and different data movement (K9 item Y,
// ATLAS_QWEN4_PREFILL_MHC_ROOF, default off). NEW module: no existing kernel
// changes.
//
// Byte-identical to qwen4_hc_upmix_hcfused[_amax] by construction:
//   - every logit is one fp32 register chain of mma.sync.m16n8k16 over the
//     k16 steps 0..19 in ascending order, starting from 0, with the same A
//     (act) and B (up) fragments as upmix_mainloop: the K loop is that loop's
//     instruction for instruction, only the shared-memory placement of the
//     operand tiles (and, for t128, the number of rows a CTA owns) changes;
//   - the epilogue is hc_upmix_hcfused_body's expression for expression:
//     x = bf16(hyper * rstd * (1 + w)), gate = 1 / (1 + __expf(-(acc * alpha))),
//     pair_p = (0 + g_2p x_2p) + g_2p+1 x_2p+1, mixed = bf16((pair_0 + pair_1) * 0.25f)
//     (this target builds with --fmad=false); the absmax is the same max of
//     |mixed| (order-free) into the same atomicMax on the float bits.
// What moves the bytes faster (the K8 item F floor micro measured the
// incumbent at 1.50 ms per 8192 rows, 67% of the 213 GB/s 2R1W roof, co-limited
// by its K loop's L2 operand stream (~1.0 ms) and its epilogue's DRAM reads of
// hyper' (~1.05 ms), which overlap only across CTAs):
//   1. PF: the CTA requests its hyper' tile into L2 when it starts, so the
//      DRAM read overlaps its own K loop instead of following it (no
//      registers, no shared memory held across the K loop);
//   2. EPI: the epilogue copies the hyper' tile and the rstd rows into shared
//      memory with 16-byte cp.async (full sectors, a quarter of the requests of
//      the incumbent's 4-byte loads) and writes `mixed` through a shared staging
//      tile with 16-byte stores;
//   3. HINT: the up weights (6.5 MB, re-read by every row tile) are kept in L2
//      (evict_last), hyper' is marked evict_first once read;
//   4. t128: 128-row CTA tiles halve the up-weight re-reads from L2 (839 ->
//      419 MB per 8192 rows) on a 3-stage swizzled cp.async ring with one
//      barrier per K stage (2 CTAs/SM).
//
// Grid (H / 32, ceil(T / BM)), row tiles last-first (the preceding HC pass
// wrote the last rows last). Block 256 = 2 warps along rows x 4 along h.
// Dynamic shared memory: kRoofSmemT64 / kRoofSmemT128 bytes (the kernel traps
// on less).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint32_t ld_u32_ca_hint(const void* p, uint64_t pol) {
    uint32_t v;
    asm volatile("ld.global.ca.L2::cache_hint.u32 %0, [%1], %2;\n" : "=r"(v) : "l"(p), "l"(pol));
    return v;
}

__device__ __forceinline__ uint32_t ld_u8_ca_hint(const void* p, uint64_t pol) {
    uint32_t v;
    asm volatile("ld.global.ca.L2::cache_hint.u8 %0, [%1], %2;\n" : "=r"(v) : "l"(p), "l"(pol));
    return v;
}

// Fast register-only bit-exact (0-ULP) NVFP4 E2M1x8 + E4M3 scale -> BF16x8 decode.
// Exhaustively verified 0 mismatched bits across all 4,064 finite (E4M3, E2M1) pairs
// against `__float2bfloat16(E2M1_LUT[nib] * float(fp8_scale))`:
//   - On sm_121a/sm_121f: native `cvt.rn.f16x2.e2m1x2` + `cvt.rn.f16x2.e4m3x2` +
//     exact FP16 HMUL2 (<=6-bit product, exponent in [-10,11]) -> BF16x2.
//   - On generic sm_120+: register PRMT magnitude/sign LUT + native `HMUL2.BF16`.
__device__ __forceinline__ uint4 decode8_nvfp4_bf16(uint32_t p4, uint32_t s_byte) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    (defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || \
     defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_SPECIFIC__))
    uint32_t s_f16x2;
    const unsigned short sb2 = static_cast<unsigned short>((s_byte & 0xffu) | ((s_byte & 0xffu) << 8));
    asm("{ .reg .b16 hs; mov.b16 hs, %1; cvt.rn.f16x2.e4m3x2 %0, hs; }" : "=r"(s_f16x2) : "h"(sb2));
    uint32_t h[4];
    asm("{ .reg .b8 b0, b1, b2, b3;\n"
        "  mov.b32 {b0, b1, b2, b3}, %4;\n"
        "  cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "  cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "  cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "  cvt.rn.f16x2.e2m1x2 %3, b3; }\n"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(p4));
    uint32_t out[4];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const __half2 prod = __hmul2(*reinterpret_cast<const __half2*>(&h[b]),
                                     *reinterpret_cast<const __half2*>(&s_f16x2));
        const float2 f = __half22float2(prod);
        const __nv_bfloat162 bf = __floats2bfloat162_rn(f.x, f.y);
        out[b] = *reinterpret_cast<const uint32_t*>(&bf);
    }
    return make_uint4(out[0], out[1], out[2], out[3]);
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
    const __nv_bfloat162 r0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&w0), s2);
    const __nv_bfloat162 r1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&w1), s2);
    const __nv_bfloat162 r2 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&w2), s2);
    const __nv_bfloat162 r3 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&w3), s2);
    return make_uint4(*reinterpret_cast<const uint32_t*>(&r0),
                      *reinterpret_cast<const uint32_t*>(&r1),
                      *reinterpret_cast<const uint32_t*>(&r2),
                      *reinterpret_cast<const uint32_t*>(&r3));
#endif
}

[[maybe_unused]] __device__ __forceinline__ uint2 quantize_bf16x8_to_e4m3(uint4 v, float inv) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
    uint16_t q[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(p[j]);
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(q[j]) : "f"(f.y * inv), "f"(f.x * inv));
    }
    uint32_t w0, w1;
    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(w0) : "r"((uint32_t)q[0]), "r"((uint32_t)q[1]));
    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(w1) : "r"((uint32_t)q[2]), "r"((uint32_t)q[3]));
    return make_uint2(w0, w1);
}

constexpr int kRank = 320;
constexpr int kHidden = 2560;
constexpr int kStreams = 4;
constexpr int kWidth = kHidden * kStreams;
constexpr int kBH = 32;
constexpr int kBN = kBH * kStreams;  // logit columns per CTA, stream-major (c = s * 32 + h)
constexpr int kBK = 32;
constexpr int kKSteps = kRank / kBK;
constexpr int kWarpsH = 4;
// Epilogue staging rows (bytes): hyper' 4 streams x 32 h bf16 padded to 272
// (4-byte reads of 8 rows x 4 words hit 32 distinct banks), rstd 4 floats,
// mixed 32 h bf16 padded to 80 (the same for the 4-byte writes).
constexpr int kHRow = kStreams * kBH * 2 + 16;
constexpr int kRRow = kStreams * 4;
constexpr int kORow = kBH * 2 + 16;

static_assert(kRank % kBK == 0 && kHidden % kBH == 0, "whole K stages and h tiles");

template <int BM, int WM, int STAGES, bool SWZ>
struct RoofTile {
    static constexpr int kThreads = WM * kWarpsH * 32;
    static constexpr int kWarpRows = BM / WM;
    static constexpr int kMT = kWarpRows / 16;  // m16 tiles per warp
    // Operand rows: 32 bf16 of K. Padded to 80 bytes (the incumbent's layout)
    // or 64 bytes with the 16-byte chunk index XOR-swizzled by (row >> 1) & 3.
    static constexpr int kRowBytes = SWZ ? kBK * 2 : (kBK + 8) * 2;
    static constexpr int kStageA = BM * kRowBytes;
    static constexpr int kStageB = kBN * kRowBytes;
    static constexpr int kStage = kStageA + kStageB;
    static constexpr int kRing = STAGES * kStage;
    static constexpr int kOffR = BM * kHRow;
    static constexpr int kOffO = kOffR + BM * kRRow;
    static constexpr int kEpi = kOffO + BM * kORow;
    static constexpr int kBody = kRing > kEpi ? kRing : kEpi;
    static constexpr int kOffMax = kBody;  // per-warp absmax
    static constexpr int kSmem = kBody + (kThreads / 32) * 4;
    static_assert(BM % (WM * 16) == 0, "whole m16 tiles per warp");
    static_assert((BM * 4) % kThreads == 0 && (kBN * 4) % kThreads == 0, "whole operand copies per thread");
    static_assert((BM * 16) % kThreads == 0 && (BM * 8) % kThreads == 0, "whole epilogue copies per thread");
    static_assert(STAGES >= 2 && STAGES - 1 <= kKSteps, "ring depth");
    static_assert(kOffR % 16 == 0 && kOffO % 16 == 0 && kStage % 16 == 0, "16-byte aligned regions");
};

// The two production tiles (the launcher passes these byte counts).
constexpr int kRoofSmemT64 = RoofTile<64, 2, 2, false>::kSmem;
constexpr int kRoofSmemT128 = RoofTile<128, 2, 3, true>::kSmem;
static_assert(kRoofSmemT64 == 30752, "the launcher passes 30752 bytes for t64");
static_assert(kRoofSmemT128 == 49184, "the launcher passes 49184 bytes for t128");

// Byte offset of 16-byte chunk c (K elements 8c..8c+7) of operand row `row`.
template <bool SWZ>
__device__ __forceinline__ unsigned int chunk_off(int row, int c) {
    if constexpr (SWZ) {
        return static_cast<unsigned int>(row * (kBK * 2) + ((c ^ ((row >> 1) & 3)) << 4));
    } else {
        return static_cast<unsigned int>(row * ((kBK + 8) * 2) + (c << 4));
    }
}

__device__ __forceinline__ void cp_async16(unsigned int s, const void* g, bool pred) {
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(bytes));
}

__device__ __forceinline__ void cp_async16_hint(unsigned int s, const void* g, bool pred, uint64_t pol) {
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n" ::"r"(s), "l"(g),
                 "r"(bytes), "l"(pol));
}

__device__ __forceinline__ void store_row16_hint(void* g, uint4 v, uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1,%2,%3,%4}, %5;\n"
                 :: "l"(g), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(pol) : "memory");
}

__device__ __forceinline__ void store_row8_hint(void* g, uint2 v, uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1,%2}, %3;\n"
                 :: "l"(g), "r"(v.x), "r"(v.y), "l"(pol) : "memory");
}

// Identical bit-for-bit to `v10_pack_e4m3x4` in `qwen4_moe_compact_t_k32_v16.cu`.
__device__ __forceinline__ unsigned upmix_pack_e4m3x4(float a, float b, float c, float d) {
    unsigned short p0, p1;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p0) : "f"(a), "f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p1) : "f"(c), "f"(d));
    unsigned r;
    asm("prmt.b32 %0, %1, %2, 0x5410;"
        : "=r"(r)
        : "r"(static_cast<unsigned>(p0)), "r"(static_cast<unsigned>(p1)));
    return r;
}

// Quad-lane MXFP8 block-32 quantization (4 consecutive lanes = 32 contiguous BF16 values).
// Instruction-for-instruction 0-ULP identical to `qwen4_mxfp8_quant_rows_v16q4_gate`.
// Caller must execute warp-convergently so all 32 lanes enter `__shfl_xor_sync`.
__device__ __forceinline__ void upmix_mxfp8_quant_quad(
    const float (&fv)[8],
    uint2& q_out,
    unsigned char& s_out) {
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
        qq0 = upmix_pack_e4m3x4(fv[0] * inv, fv[1] * inv, fv[2] * inv, fv[3] * inv);
        qq1 = upmix_pack_e4m3x4(fv[4] * inv, fv[5] * inv, fv[6] * inv, fv[7] * inv);
        ss = static_cast<unsigned>(k + 127);
    }
    q_out = make_uint2(qq0, qq1);
    s_out = static_cast<unsigned char>(ss);
}

__device__ __forceinline__ float roof_fast_rcp_rn_1p_exp(float x) {
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

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

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

__device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p));
}

#ifdef K9Y_UPMIX_ROOF_VARIANTS
__device__ __forceinline__ void bulk_prefetch_l2_64(const void* p) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(64u) : "memory");
}
#endif

__device__ __forceinline__ void ldsm_x4(unsigned int (&r)[4], unsigned int s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

// qwen4_hc_upmix.cu's mma_bf16, verbatim.
__device__ __forceinline__ void mma_bf16(float (&d)[4], const unsigned int (&a)[4],
                                         unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// PF: request the CTA's hyper' tile (BM rows x 4 streams x 64 bytes) into L2.
// 1 = one prefetch.global.L2 per 32-byte sector; 2 (bench variants only) =
// one 64-byte cp.async.bulk.prefetch.L2 per (row, stream).
template <int BM, int THREADS, int PF>
__device__ __forceinline__ void prefetch_tile(const __nv_bfloat16* __restrict__ hyper, int t0, int h0,
                                              unsigned int num_tokens) {
    if constexpr (PF == 1) {
#pragma unroll
        for (int j = 0; j < BM * 8 / THREADS; ++j) {
            const int idx = threadIdx.x + j * THREADS;
            const int row = idx >> 3;
            const int s = (idx >> 1) & 3;
            const int sec = idx & 1;
            const int t = t0 + row;
            if (t < (int)num_tokens) prefetch_l2(hyper + (size_t)t * kWidth + s * kHidden + h0 + sec * 16);
        }
    }
#ifdef K9Y_UPMIX_ROOF_VARIANTS
    if constexpr (PF == 2) {
#pragma unroll
        for (int j = 0; j < BM * 4 / THREADS; ++j) {
            const int idx = threadIdx.x + j * THREADS;
            const int row = idx >> 2;
            const int s = idx & 3;
            const int t = t0 + row;
            if (t < (int)num_tokens) bulk_prefetch_l2_64(hyper + (size_t)t * kWidth + s * kHidden + h0);
        }
    }
#endif
}

// The K loop: acc[i][s][e] holds logit columns s*2560 + h0 + wh*8 + 2*(lane%4)
// (+1) of tile rows wm*kWarpRows + i*16 + lane/4 (+8), each one fp32 chain
// over the k16 steps 0..19 in ascending order (upmix_mainloop's chain).
// ABL (bench only, timing): bit 2 = no operand copies.
template <int BM, int WM, int STAGES, bool SWZ, bool HINT, int ABL, bool W4 = false>
__device__ __forceinline__ void roof_mainloop(unsigned char* smem,
                                              unsigned int sbase,
                                              const __nv_bfloat16* __restrict__ act,
                                              const __nv_bfloat16* __restrict__ up,
                                              const unsigned char* __restrict__ up_packed,
                                              const unsigned char* __restrict__ up_scales,
                                              unsigned int num_tokens, int t0, int h0,
                                              float (&acc)[RoofTile<BM, WM, STAGES, SWZ>::kMT][kStreams][4]) {
    using T = RoofTile<BM, WM, STAGES, SWZ>;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp >> 2;
    const int wh = warp & 3;
    uint64_t keep = 0;
    if constexpr (HINT) keep = policy_evict_last();

    constexpr int kCopiesA = BM * 4 / T::kThreads;
    constexpr int kCopiesB = kBN * 4 / T::kThreads;
    const int part = tid & 3;
    unsigned int sa_off[kCopiesA];
    const __nv_bfloat16* act_base[kCopiesA];
    bool act_ok[kCopiesA];
#pragma unroll
    for (int j = 0; j < kCopiesA; ++j) {
        const int row = (tid + j * T::kThreads) >> 2;
        const int t = t0 + row;
        act_ok[j] = t < (int)num_tokens;
        sa_off[j] = sbase + chunk_off<SWZ>(row, part);
        act_base[j] = act + (size_t)(act_ok[j] ? t : 0) * kRank + part * 8;
    }
    unsigned int sb_off[kCopiesB];
    const __nv_bfloat16* up_base[kCopiesB];
    const unsigned char* up_p_base[kCopiesB];
    const unsigned char* up_s_base[kCopiesB];
#pragma unroll
    for (int j = 0; j < kCopiesB; ++j) {
        const int c = (tid + j * T::kThreads) >> 2;
        const int n = (c / kBH) * kHidden + h0 + (c % kBH);
        sb_off[j] = sbase + T::kStageA + chunk_off<SWZ>(c, part);
        if constexpr (W4) {
            up_p_base[j] = up_packed + (size_t)n * (kRank / 2u) + (size_t)(part * 4);
            up_s_base[j] = up_scales + (size_t)n * (kRank / 16u) + (size_t)(part >> 1);
            up_base[j] = nullptr;
        } else {
            up_base[j] = up + (size_t)n * kRank + part * 8;
            up_p_base[j] = nullptr;
            up_s_base[j] = nullptr;
        }
    }

    auto load_stage = [&](int slot, int k0) {
        if constexpr ((ABL & 2) == 0) {
            const unsigned int stg_off = static_cast<unsigned int>(slot * T::kStage);
#pragma unroll
            for (int j = 0; j < kCopiesA; ++j) {
                const __nv_bfloat16* src_a = act_base[j] + k0;
                if constexpr (HINT) {
                    if (num_tokens <= 4096u) {
                        cp_async16_hint(sa_off[j] + stg_off, src_a, act_ok[j], keep);
                    } else {
                        cp_async16(sa_off[j] + stg_off, src_a, act_ok[j]);
                    }
                } else {
                    cp_async16(sa_off[j] + stg_off, src_a, act_ok[j]);
                }
            }
            if constexpr (W4) {
                cp_async_commit();
                uint32_t p4[kCopiesB];
                uint32_t sb_val[kCopiesB];
#pragma unroll
                for (int j = 0; j < kCopiesB; ++j) {
                    if constexpr (HINT) {
                        p4[j] = ld_u32_ca_hint(up_p_base[j] + (k0 >> 1), keep);
                        sb_val[j] = ld_u8_ca_hint(up_s_base[j] + (k0 >> 4), keep);
                    } else {
                        p4[j] = __ldg(reinterpret_cast<const uint32_t*>(up_p_base[j] + (k0 >> 1)));
                        sb_val[j] = __ldg(up_s_base[j] + (k0 >> 4));
                    }
                }
#pragma unroll
                for (int j = 0; j < kCopiesB; ++j) {
                    const uint4 dec = decode8_nvfp4_bf16(p4[j], sb_val[j]);
                    const unsigned int dst_s = sb_off[j] + stg_off;
                    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                                 :: "r"(dst_s), "r"(dec.x), "r"(dec.y), "r"(dec.z), "r"(dec.w)
                                 : "memory");
                }
                return;
            } else {
#pragma unroll
                for (int j = 0; j < kCopiesB; ++j) {
                    if constexpr (HINT) {
                        cp_async16_hint(sb_off[j] + stg_off, up_base[j] + k0, true, keep);
                    } else {
                        cp_async16(sb_off[j] + stg_off, up_base[j] + k0, true);
                    }
                }
            }
        }
        cp_async_commit();
    };

    auto compute = [&](int slot) {
        const unsigned int sa = sbase + slot * T::kStage;
        const unsigned int sb = sa + T::kStageA;
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 16) {
            unsigned int a[T::kMT][4];
#pragma unroll
            for (int i = 0; i < T::kMT; ++i) {
                ldsm_x4(a[i], sa + chunk_off<SWZ>(wm * T::kWarpRows + i * 16 + (lane & 15), (kk >> 3) + (lane >> 4)));
            }
            unsigned int b[kStreams][2];
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                // Lanes 0-15 address stream 2p, lanes 16-31 stream 2p+1; each
                // 16-lane half gives the fragment's k and k+8 halves.
                const int s = 2 * p + (lane >> 4);
                const int c = s * kBH + wh * 8 + (lane & 7);
                unsigned int r[4];
                ldsm_x4(r, sb + chunk_off<SWZ>(c, (kk >> 3) + ((lane >> 3) & 1)));
                b[2 * p][0] = r[0];
                b[2 * p][1] = r[1];
                b[2 * p + 1][0] = r[2];
                b[2 * p + 1][1] = r[3];
            }
#pragma unroll
            for (int i = 0; i < T::kMT; ++i)
#pragma unroll
                for (int s = 0; s < kStreams; ++s) mma_bf16(acc[i][s], a[i], b[s][0], b[s][1]);
        }
    };

#pragma unroll
    for (int i = 0; i < T::kMT; ++i)
#pragma unroll
        for (int s = 0; s < kStreams; ++s)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][s][e] = 0.0f;

    if constexpr (STAGES == 2) {
        // upmix_mainloop's schedule: issue stage+1, wait for stage, two barriers.
        load_stage(0, 0);
#pragma unroll
        for (int stage = 0; stage < kKSteps; ++stage) {
            const int buf = stage & 1;
            if (stage + 1 < kKSteps) {
                load_stage(buf ^ 1, (stage + 1) * kBK);
                cp_async_wait<1>();
            } else {
                cp_async_wait<0>();
            }
            __syncthreads();
            compute(buf);
            __syncthreads();
        }
    } else {
        // STAGES-deep ring, one barrier per stage: after the barrier of stage k
        // every warp has finished stage k-1, whose slot then takes stage
        // k + STAGES - 1 (an empty group past the last stage keeps the count).
#pragma unroll
        for (int st = 0; st < STAGES - 1; ++st) load_stage(st, st * kBK);
        int rd = 0;
        int wr = STAGES - 1;
#pragma unroll
        for (int stage = 0; stage < kKSteps; ++stage) {
            cp_async_wait<STAGES - 2>();
            __syncthreads();
            const int nx = stage + STAGES - 1;
            if (nx < kKSteps) {
                load_stage(wr, nx * kBK);
            } else {
                cp_async_commit();
            }
            wr = wr + 1 == STAGES ? 0 : wr + 1;
            compute(rd);
            rd = rd + 1 == STAGES ? 0 : rd + 1;
        }
        cp_async_wait<0>();
        __syncthreads();  // the ring is idle for the epilogue
    }
}

// hc_upmix_hcfused_body's epilogue for one output pair: tile row `row`
// (token t0 + row < num_tokens), the four stream logits acc_i[s][half*2 + e],
// hyper' pairs h2[s], rstd r[s], (1 + w) w1[s][e]. Expression for expression.
__device__ __forceinline__ __nv_bfloat162 mix_pair(const float (&acc_i)[kStreams][4], int half, float alpha,
                                                   const __nv_bfloat162 (&h2)[kStreams], const float (&r)[kStreams],
                                                   const float (&w1)[kStreams][2]) {
    float pair[2][2];  // [stream pair][e]
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        pair[p][0] = 0.0f;
        pair[p][1] = 0.0f;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int s = 2 * p + q;
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const float logit = acc_i[s][half * 2 + e] * alpha;
                const float gate = roof_fast_rcp_rn_1p_exp(logit);
                const float x = __bfloat162float(e == 0 ? h2[s].x : h2[s].y) * r[s] * w1[s][e];
                pair[p][e] += gate * __bfloat162float(__float2bfloat16(x));
            }
        }
    }
    __nv_bfloat162 out;
    out.x = __float2bfloat16((pair[0][0] + pair[1][0]) * 0.25f);
    out.y = __float2bfloat16((pair[0][1] + pair[1][1]) * 0.25f);
    return out;
}

__device__ __forceinline__ __nv_bfloat162 mix_pair_gated(const float (&gate_i)[kStreams][4], int half,
                                                         const __nv_bfloat162 (&h2)[kStreams],
                                                         const float (&r)[kStreams],
                                                         const float (&w1)[kStreams][2]) {
    float pair[2][2];  // [stream pair][e]
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        pair[p][0] = 0.0f;
        pair[p][1] = 0.0f;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int s = 2 * p + q;
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const float gate = gate_i[s][half * 2 + e];
                const float x = __bfloat162float(e == 0 ? h2[s].x : h2[s].y) * r[s] * w1[s][e];
                pair[p][e] += gate * __bfloat162float(__float2bfloat16(x));
            }
        }
    }
    __nv_bfloat162 out;
    out.x = __float2bfloat16((pair[0][0] + pair[1][0]) * 0.25f);
    out.y = __float2bfloat16((pair[0][1] + pair[1][1]) * 0.25f);
    return out;
}

// ABL (bench only, timing): bit 1 = no hyper' / rstd reads in the epilogue.
// STREAM_OUT: mark `mixed` stores evict_first so strip-mined wide prefill
// keeps `folded`, `up`, `down[strip]`, and `hyper'[strip]` resident in L2.
// MXFP8: also emit block-32 MXFP8 `(xq, xs)` directly from the staged BF16
// `mixed` values (0-ULP bit-identical to `qwen4_mxfp8_quant_rows_v16q4_gate`).
template <int BM, int WM, int STAGES, bool SWZ, int PF, bool EPI, bool HINT, bool AMAX, int ABL, bool OVERLAP = false, bool W4 = false, bool STREAM_OUT = false, bool MXFP8 = false>
__device__ __forceinline__ void upmix_roof_body(unsigned char* smem,
                                                const __nv_bfloat16* __restrict__ act,
                                                const __nv_bfloat16* __restrict__ up,
                                                const unsigned char* __restrict__ up_packed,
                                                const unsigned char* __restrict__ up_scales,
                                                float alpha,
                                                const __nv_bfloat16* __restrict__ hyper,
                                                const float* __restrict__ rstd,
                                                const __nv_bfloat16* __restrict__ norm_w,
                                                __nv_bfloat16* __restrict__ mixed,
                                                unsigned int num_tokens,
                                                unsigned int* __restrict__ amax_bits,
                                                unsigned char* __restrict__ xq = nullptr,
                                                unsigned char* __restrict__ xs = nullptr) {
    using T = RoofTile<BM, WM, STAGES, SWZ>;
    {
        unsigned int dyn;
        asm("mov.u32 %0, %%dynamic_smem_size;\n" : "=r"(dyn));
        if (dyn < (unsigned int)T::kSmem) __trap();
    }
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp >> 2;
    const int wh = warp & 3;
    const int h0 = blockIdx.x * kBH;
    // Row tiles last-first, as the incumbent.
    const int t0 = (int)(gridDim.y - 1 - blockIdx.y) * BM;
    const unsigned int sbase = static_cast<unsigned int>(__cvta_generic_to_shared(smem));

    if constexpr (PF != 0 && (ABL & 1) == 0) prefetch_tile<BM, T::kThreads, PF>(hyper, t0, h0, num_tokens);

    float acc[T::kMT][kStreams][4];
    roof_mainloop<BM, WM, STAGES, SWZ, HINT, ABL, W4>(smem, sbase, act, up, up_packed, up_scales, num_tokens, t0, h0, acc);

    const int hl = wh * 8 + 2 * (lane & 3);
    if constexpr (OVERLAP) {
        float amax = 0.0f;
        if constexpr (EPI) {
            uint64_t first = 0;
            if constexpr (HINT) first = policy_evict_first();
            const bool full_tile = (unsigned int)(t0 + BM) <= num_tokens;
            if constexpr ((ABL & 1) == 0) {
                if (full_tile) {
                    const int row0 = tid >> 4;
                    const int s = (tid >> 2) & 3;
                    const int piece = tid & 3;
                    const __nv_bfloat16* src_base =
                        hyper + (size_t)(t0 + row0) * kWidth + s * kHidden + h0 + piece * 8;
                    const unsigned int dst_base = sbase + row0 * kHRow + s * (kBH * 2) + piece * 16;
                    constexpr int kRowStep = T::kThreads >> 4;
#pragma unroll
                    for (int j = 0; j < BM * 16 / T::kThreads; ++j) {
                        const __nv_bfloat16* src = src_base + (size_t)(j * kRowStep) * kWidth;
                        const unsigned int dst = dst_base + j * kRowStep * kHRow;
                        if constexpr (HINT) {
                            cp_async16_hint(dst, src, true, first);
                        } else {
                            cp_async16(dst, src, true);
                        }
                    }
                    if (tid < BM) {
                        cp_async16(sbase + T::kOffR + tid * kRRow, rstd + (size_t)(t0 + tid) * kStreams, true);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < BM * 16 / T::kThreads; ++j) {
                        const int idx = tid + j * T::kThreads;
                        const int row = idx >> 4;
                        const int s = (idx >> 2) & 3;
                        const int piece = idx & 3;
                        const int t = t0 + row;
                        const bool ok = t < (int)num_tokens;
                        const __nv_bfloat16* src = hyper + (size_t)(ok ? t : 0) * kWidth + s * kHidden + h0 + piece * 8;
                        const unsigned int dst = sbase + row * kHRow + s * (kBH * 2) + piece * 16;
                        if constexpr (HINT) {
                            cp_async16_hint(dst, src, ok, first);
                        } else {
                            cp_async16(dst, src, ok);
                        }
                    }
                    for (int row = tid; row < BM; row += T::kThreads) {
                        const int t = t0 + row;
                        const bool ok = t < (int)num_tokens;
                        cp_async16(sbase + T::kOffR + row * kRRow, rstd + (size_t)(ok ? t : 0) * kStreams, ok);
                    }
                }
            }
            cp_async_commit();
            // Overlap norm_w loads and all 64 MUFU sigmoid evaluations with the 34 KiB cp.async transfer.
            float w1[kStreams][2];
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                const __nv_bfloat162 w2 =
                    *reinterpret_cast<const __nv_bfloat162*>(norm_w + (size_t)s * kHidden + h0 + hl);
                w1[s][0] = 1.0f + __bfloat162float(w2.x);
                w1[s][1] = 1.0f + __bfloat162float(w2.y);
            }
#pragma unroll
            for (int i = 0; i < T::kMT; ++i) {
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
#pragma unroll
                    for (int idx = 0; idx < 4; ++idx) {
                        const float logit = acc[i][s][idx] * alpha;
                        acc[i][s][idx] = roof_fast_rcp_rn_1p_exp(logit);
                    }
                }
            }
            asm volatile("cp.async.wait_group 0;\n"
                         "bar.sync 0;\n"
                         : "+f"(acc[T::kMT - 1][kStreams - 1][3]) :: "memory");
            if (full_tile) {
#pragma unroll
                for (int i = 0; i < T::kMT; ++i) {
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int row = wm * T::kWarpRows + i * 16 + (lane >> 2) + half * 8;
                        const float4 r4 = *reinterpret_cast<const float4*>(smem + T::kOffR + row * kRRow);
                        const float r[kStreams] = {r4.x, r4.y, r4.z, r4.w};
                        __nv_bfloat162 h2[kStreams];
#pragma unroll
                        for (int s = 0; s < kStreams; ++s) {
                            h2[s] = *reinterpret_cast<const __nv_bfloat162*>(smem + row * kHRow + s * (kBH * 2) + hl * 2);
                        }
                        const __nv_bfloat162 out = mix_pair_gated(acc[i], half, h2, r, w1);
                        *reinterpret_cast<__nv_bfloat162*>(smem + T::kOffO + row * kORow + hl * 2) = out;
                        if constexpr (AMAX) {
                            const float2 f = __bfloat1622float2(out);
                            amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
                        }
                    }
                }
            } else {
#pragma unroll
                for (int i = 0; i < T::kMT; ++i) {
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int row = wm * T::kWarpRows + i * 16 + (lane >> 2) + half * 8;
                        const unsigned int token = t0 + row;
                        if (token >= num_tokens) continue;
                        const float4 r4 = *reinterpret_cast<const float4*>(smem + T::kOffR + row * kRRow);
                        const float r[kStreams] = {r4.x, r4.y, r4.z, r4.w};
                        __nv_bfloat162 h2[kStreams];
#pragma unroll
                        for (int s = 0; s < kStreams; ++s) {
                            h2[s] = *reinterpret_cast<const __nv_bfloat162*>(smem + row * kHRow + s * (kBH * 2) + hl * 2);
                        }
                        const __nv_bfloat162 out = mix_pair_gated(acc[i], half, h2, r, w1);
                        *reinterpret_cast<__nv_bfloat162*>(smem + T::kOffO + row * kORow + hl * 2) = out;
                        if constexpr (AMAX) {
                            const float2 f = __bfloat1622float2(out);
                            amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
                        }
                    }
                }
            }
            __syncthreads();
            const bool hint_out = HINT && (STREAM_OUT || num_tokens <= 4096u);
            const uint64_t out_pol =
                hint_out ? (STREAM_OUT ? policy_evict_first() : policy_evict_last()) : 0;
            uint64_t xq_pol = 0;
            if constexpr (MXFP8) {
                xq_pol = (STREAM_OUT || num_tokens > 4096u) ? policy_evict_first() : policy_evict_last();
            }
            if (full_tile) {
                const int row0 = tid >> 2;
                const int piece = tid & 3;
                __nv_bfloat16* dst_base = mixed + (size_t)(t0 + row0) * kHidden + h0 + piece * 8;
                const unsigned char* src_base = smem + T::kOffO + row0 * kORow + piece * 16;
                constexpr int kOutRowStep = T::kThreads >> 2;
#pragma unroll
                for (int j = 0; j < BM * 4 / T::kThreads; ++j) {
                    const uint4 v = *reinterpret_cast<const uint4*>(src_base + j * kOutRowStep * kORow);
                    __nv_bfloat16* dst = dst_base + (size_t)(j * kOutRowStep) * kHidden;
                    if (hint_out) {
                        store_row16_hint(dst, v, out_pol);
                    } else {
                        *reinterpret_cast<uint4*>(dst) = v;
                    }
                    if constexpr (MXFP8) {
                        const unsigned u[4] = {v.x, v.y, v.z, v.w};
                        float fv[8];
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            fv[2 * k] = __uint_as_float(u[k] << 16);
                            fv[2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u);
                        }
                        uint2 q_out;
                        unsigned char s_out;
                        upmix_mxfp8_quant_quad(fv, q_out, s_out);
                        const int t = t0 + row0 + j * kOutRowStep;
                        uint2* qptr = reinterpret_cast<uint2*>(xq + (size_t)t * kHidden + h0 + piece * 8);
                        store_row8_hint(qptr, q_out, xq_pol);
                        if (piece == 0) {
                            xs[(size_t)t * (kHidden / kBH) + blockIdx.x] = s_out;
                        }
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < BM * 4 / T::kThreads; ++j) {
                    const int idx = tid + j * T::kThreads;
                    const int row = idx >> 2;
                    const int piece = idx & 3;
                    const int t = t0 + row;
                    const bool ok = t < (int)num_tokens;
                    const uint4 v = ok
                        ? *reinterpret_cast<const uint4*>(smem + T::kOffO + row * kORow + piece * 16)
                        : make_uint4(0u, 0u, 0u, 0u);
                    if constexpr (MXFP8) {
                        const unsigned u[4] = {v.x, v.y, v.z, v.w};
                        float fv[8];
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            fv[2 * k] = __uint_as_float(u[k] << 16);
                            fv[2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u);
                        }
                        uint2 q_out;
                        unsigned char s_out;
                        upmix_mxfp8_quant_quad(fv, q_out, s_out);
                        if (ok) {
                            __nv_bfloat16* dst = mixed + (size_t)t * kHidden + h0 + piece * 8;
                            if (hint_out) {
                                store_row16_hint(dst, v, out_pol);
                            } else {
                                *reinterpret_cast<uint4*>(dst) = v;
                            }
                            uint2* qptr = reinterpret_cast<uint2*>(xq + (size_t)t * kHidden + h0 + piece * 8);
                            store_row8_hint(qptr, q_out, xq_pol);
                            if (piece == 0) {
                                xs[(size_t)t * (kHidden / kBH) + blockIdx.x] = s_out;
                            }
                        }
                    } else if (ok) {
                        __nv_bfloat16* dst = mixed + (size_t)t * kHidden + h0 + piece * 8;
                        if (hint_out) {
                            store_row16_hint(dst, v, out_pol);
                        } else {
                            *reinterpret_cast<uint4*>(dst) = v;
                        }
                    }
                }
            }
        }
        if constexpr (AMAX) {
            float* warp_max = reinterpret_cast<float*>(smem + T::kOffMax);
            for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, d));
            if (lane == 0) warp_max[warp] = amax;
            __syncthreads();
            if (tid == 0) {
                float block = warp_max[0];
                for (int w = 1; w < T::kThreads / 32; ++w) block = fmaxf(block, warp_max[w]);
                const unsigned int bits = __float_as_uint(block);
                if (block > 0.0f && bits > *reinterpret_cast<volatile unsigned int*>(amax_bits)) {
                    atomicMax(amax_bits, bits);
                }
            }
        }
    } else {
        float w1[kStreams][2];
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            const __nv_bfloat162 w2 =
                *reinterpret_cast<const __nv_bfloat162*>(norm_w + (size_t)s * kHidden + h0 + hl);
            w1[s][0] = 1.0f + __bfloat162float(w2.x);
            w1[s][1] = 1.0f + __bfloat162float(w2.y);
        }
        float amax = 0.0f;

        if constexpr (EPI) {
            // The ring is idle (every warp is past the K loop's last barrier):
            // hyper' tile and rstd rows into shared memory, 16 bytes per copy.
            if constexpr ((ABL & 1) == 0) {
                uint64_t first = 0;
                if constexpr (HINT) first = policy_evict_first();
#pragma unroll
                for (int j = 0; j < BM * 16 / T::kThreads; ++j) {
                    const int idx = tid + j * T::kThreads;
                    const int row = idx >> 4;
                    const int s = (idx >> 2) & 3;
                    const int piece = idx & 3;
                    const int t = t0 + row;
                    const bool ok = t < (int)num_tokens;
                    const __nv_bfloat16* src = hyper + (size_t)(ok ? t : 0) * kWidth + s * kHidden + h0 + piece * 8;
                    const unsigned int dst = sbase + row * kHRow + s * (kBH * 2) + piece * 16;
                    if constexpr (HINT) {
                        cp_async16_hint(dst, src, ok, first);
                    } else {
                        cp_async16(dst, src, ok);
                    }
                }
                for (int row = tid; row < BM; row += T::kThreads) {
                    const int t = t0 + row;
                    const bool ok = t < (int)num_tokens;
                    cp_async16(sbase + T::kOffR + row * kRRow, rstd + (size_t)(ok ? t : 0) * kStreams, ok);
                }
            }
            cp_async_commit();
            cp_async_wait<0>();
            __syncthreads();
#pragma unroll
            for (int i = 0; i < T::kMT; ++i) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int row = wm * T::kWarpRows + i * 16 + (lane >> 2) + half * 8;
                    const unsigned int token = t0 + row;
                    if (token >= num_tokens) continue;
                    const float4 r4 = *reinterpret_cast<const float4*>(smem + T::kOffR + row * kRRow);
                    const float r[kStreams] = {r4.x, r4.y, r4.z, r4.w};
                    __nv_bfloat162 h2[kStreams];
#pragma unroll
                    for (int s = 0; s < kStreams; ++s) {
                        h2[s] = *reinterpret_cast<const __nv_bfloat162*>(smem + row * kHRow + s * (kBH * 2) + hl * 2);
                    }
                    const __nv_bfloat162 out = mix_pair(acc[i], half, alpha, h2, r, w1);
                    *reinterpret_cast<__nv_bfloat162*>(smem + T::kOffO + row * kORow + hl * 2) = out;
                    if constexpr (AMAX) {
                        const float2 f = __bfloat1622float2(out);
                        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
                    }
                }
            }
            __syncthreads();
            // `mixed`: 16-byte stores, a row's 32 h positions in four.
            const uint64_t out_pol = (HINT && STREAM_OUT) ? policy_evict_first() : 0;
            const uint64_t xq_pol = MXFP8
                ? ((STREAM_OUT || num_tokens > 4096u) ? policy_evict_first() : policy_evict_last())
                : 0;
#pragma unroll
            for (int j = 0; j < BM * 4 / T::kThreads; ++j) {
                const int idx = tid + j * T::kThreads;
                const int row = idx >> 2;
                const int piece = idx & 3;
                const int t = t0 + row;
                const bool ok = t < (int)num_tokens;
                if constexpr (MXFP8) {
                    const uint4 v = ok
                        ? *reinterpret_cast<const uint4*>(smem + T::kOffO + row * kORow + piece * 16)
                        : make_uint4(0u, 0u, 0u, 0u);
                    const unsigned u[4] = {v.x, v.y, v.z, v.w};
                    float fv[8];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        fv[2 * k] = __uint_as_float(u[k] << 16);
                        fv[2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u);
                    }
                    uint2 q_out;
                    unsigned char s_out;
                    upmix_mxfp8_quant_quad(fv, q_out, s_out);
                    if (ok) {
                        __nv_bfloat16* dst = mixed + (size_t)t * kHidden + h0 + piece * 8;
                        if constexpr (HINT && STREAM_OUT) {
                            store_row16_hint(dst, v, out_pol);
                        } else {
                            *reinterpret_cast<uint4*>(dst) = v;
                        }
                        uint2* qptr = reinterpret_cast<uint2*>(xq + (size_t)t * kHidden + h0 + piece * 8);
                        store_row8_hint(qptr, q_out, xq_pol);
                        if (piece == 0) {
                            xs[(size_t)t * (kHidden / kBH) + blockIdx.x] = s_out;
                        }
                    }
                } else if (ok) {
                    const uint4 v = *reinterpret_cast<const uint4*>(smem + T::kOffO + row * kORow + piece * 16);
                    __nv_bfloat16* dst = mixed + (size_t)t * kHidden + h0 + piece * 8;
                    if constexpr (HINT && STREAM_OUT) {
                        store_row16_hint(dst, v, out_pol);
                    } else {
                        *reinterpret_cast<uint4*>(dst) = v;
                    }
                }
            }
        } else {
            // The incumbent's epilogue (4-byte loads and stores), for the bench.
#pragma unroll
            for (int i = 0; i < T::kMT; ++i) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const unsigned int token = t0 + wm * T::kWarpRows + i * 16 + (lane >> 2) + half * 8;
                    if (token >= num_tokens) continue;
                    const size_t row = (size_t)token * kWidth;
                    float r[kStreams];
                    __nv_bfloat162 h2[kStreams];
#pragma unroll
                    for (int s = 0; s < kStreams; ++s) {
                        if constexpr ((ABL & 1) == 0) {
                            r[s] = rstd[(size_t)token * kStreams + s];
                            h2[s] = *reinterpret_cast<const __nv_bfloat162*>(hyper + row + (size_t)s * kHidden + h0 + hl);
                        } else {
                            r[s] = 1.0f;
                            h2[s] = __floats2bfloat162_rn(0.5f, 0.25f);
                        }
                    }
                    const __nv_bfloat162 out = mix_pair(acc[i], half, alpha, h2, r, w1);
                    *reinterpret_cast<__nv_bfloat162*>(mixed + (size_t)token * kHidden + h0 + hl) = out;
                    if constexpr (AMAX) {
                        const float2 f = __bfloat1622float2(out);
                        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
                    }
                }
            }
        }
        if constexpr (AMAX) {
            float* warp_max = reinterpret_cast<float*>(smem + T::kOffMax);
            for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, d));
            if (lane == 0) warp_max[warp] = amax;
            __syncthreads();
            if (tid == 0) {
                float block = warp_max[0];
                for (int w = 1; w < T::kThreads / 32; ++w) block = fmaxf(block, warp_max[w]);
                const unsigned int bits = __float_as_uint(block);
                if (block > 0.0f && bits > *reinterpret_cast<volatile unsigned int*>(amax_bits)) {
                    atomicMax(amax_bits, bits);
                }
            }
        }
    }
}

__device__ __forceinline__ unsigned int atomic_add_acq_rel_gpu(unsigned int* ptr, unsigned int val) {
    unsigned int old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;\n"
                 : "=r"(old)
                 : "l"(ptr), "r"(val)
                 : "memory");
    return old;
}

__device__ __forceinline__ __nv_bfloat16 ld_cg_bf16(const __nv_bfloat16* p) {
    unsigned short v;
    asm volatile("ld.global.cg.u16 %0, [%1];\n" : "=h"(v) : "l"(p));
    return *reinterpret_cast<const __nv_bfloat16*>(&v);
}

__device__ __forceinline__ __nv_bfloat16 ld_cg_bf16_ef(const __nv_bfloat16* p, uint64_t pol) {
    unsigned short v;
    asm volatile("ld.global.cg.L2::cache_hint.u16 %0, [%1], %2;\n" : "=h"(v) : "l"(p), "l"(pol));
    return *reinterpret_cast<const __nv_bfloat16*>(&v);
}

// Single-kernel fused up+mix + MXFP8 `(xq, xs)` + Last-CTA L2-hot shared-expert
// gate scalar reduction (`ss_out[rows]`).
//
// Grid = `(80, ceil(T / 128), 1)`, block = `256`, SMEM = `49,184 B` (2 CTAs/SM):
//   - All 80 CTAs of row-tile `blockIdx.y` execute the exact 3-stage `t128_r4`/
//     `t128_w4` K-loop and epilogue, writing `mixed` (BF16), `xq` (E4M3), `xs`
//     (UE8M0), and optional `amax_bits` directly from staged BF16 values.
//   - When `ss_out != nullptr`, each CTA arrives on `tile_done[blockIdx.y]` via
//     `atom.acq_rel.gpu.global.add.u32`. Exactly the 80th (last) CTA resets
//     `tile_done[blockIdx.y] = 0` for the next launch and immediately computes
//     `ss_out[t0..row_end)` from the 640 KiB L2-hot `mixed` tile (0 DRAM reads,
//     2-row-per-barrier double-buffered `s_dot[2][16]`), matching the exact
//     256-thread strided reduction tree of `qwen4_mxfp8_quant_rows_v16q4_gate` /
//     `moe_batched_blend` bit-for-bit (0-ULP under `--fmad=false`).
//   - When `ss_out == nullptr`, skips the Last-CTA gate path (0 atomics).
template <bool AMAX, bool W4, bool R4 = true, bool STREAM_OUT = false>
__device__ __forceinline__ void upmix_roof_mxfp8_gate_body(
    unsigned char* smem,
    const __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ up,
    const unsigned char* __restrict__ up_packed,
    const unsigned char* __restrict__ up_scales,
    float alpha,
    const __nv_bfloat16* __restrict__ hyper,
    const float* __restrict__ rstd,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int num_tokens,
    unsigned int* __restrict__ amax_bits,
    unsigned char* __restrict__ xq,
    unsigned char* __restrict__ xs,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out,
    unsigned int* __restrict__ tile_done) {
    upmix_roof_body<128, 2, 3, true, 1, true, true, AMAX, 0, R4, W4, STREAM_OUT, true>(
        smem, act, up, up_packed, up_scales, alpha, hyper, rstd, norm_w, mixed,
        num_tokens, amax_bits, xq, xs);

    if (ss_out == nullptr) return;
    if (gridDim.x != static_cast<unsigned int>(kHidden / kBH) || tile_done == nullptr) __trap();
    __threadfence();
    __syncthreads();

    const int tid = threadIdx.x;
    unsigned int* flag_smem = reinterpret_cast<unsigned int*>(smem);
    if (tid == 0) {
        const unsigned int old = atomic_add_acq_rel_gpu(tile_done + blockIdx.y, 1u);
        const unsigned int is_last =
            (old + 1u == static_cast<unsigned int>(kHidden / kBH)) ? 1u : 0u;
        if (is_last) {
            tile_done[blockIdx.y] = 0u;  // self-reset for next launch / graph replay
        }
        flag_smem[0] = is_last;
    }
    __syncthreads();
    if (flag_smem[0] == 0u) return;

    const int t0 = (int)(gridDim.y - 1 - blockIdx.y) * 128;
    const int row_end = min(t0 + 128, (int)num_tokens);
    if (shared_gate == nullptr) {
        if (tid < 128 && (t0 + tid) < row_end) {
            ss_out[t0 + tid] = 1.0f;
        }
        return;
    }

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const uint64_t mixed_ef = STREAM_OUT ? policy_evict_first() : 0ull;
    float g[10];
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        g[i] = __bfloat162float(shared_gate[tid + i * 256]);
    }
    // Double-buffered 2-row-per-barrier warp sums (`s_dot[2][16]`, 128 B at smem+16).
    float* s_dot = reinterpret_cast<float*>(smem + 16);
    unsigned int parity = 0u;
    int r = t0;
    for (; r + 1 < row_end; r += 2) {
        const __nv_bfloat16* x0 = mixed + (size_t)r * kHidden;
        const __nv_bfloat16* x1 = x0 + kHidden;
        float d0 = 0.0f;
        float d1 = 0.0f;
#pragma unroll
        for (int i = 0; i < 10; ++i) {
            const __nv_bfloat16 v0 =
                STREAM_OUT ? ld_cg_bf16_ef(x0 + tid + i * 256, mixed_ef) : ld_cg_bf16(x0 + tid + i * 256);
            const __nv_bfloat16 v1 =
                STREAM_OUT ? ld_cg_bf16_ef(x1 + tid + i * 256, mixed_ef) : ld_cg_bf16(x1 + tid + i * 256);
            const float n0 = __bfloat162float(v0);
            const float n1 = __bfloat162float(v1);
            d0 += n0 * g[i];
            d1 += n1 * g[i];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            d0 += __shfl_down_sync(0xffffffffu, d0, offset);
            d1 += __shfl_down_sync(0xffffffffu, d1, offset);
        }
        if (lane == 0) {
            s_dot[parity * 16u + warp] = d0;
            s_dot[parity * 16u + 8u + warp] = d1;
        }
        __syncthreads();
        if (tid == 0) {
            float total0 = 0.0f;
            float total1 = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                total0 += s_dot[parity * 16u + w];
                total1 += s_dot[parity * 16u + 8u + w];
            }
            ss_out[r] = 1.0f / (1.0f + __expf(-total0));
            ss_out[r + 1] = 1.0f / (1.0f + __expf(-total1));
        }
        parity ^= 1u;
    }
    if (r < row_end) {
        const __nv_bfloat16* x0 = mixed + (size_t)r * kHidden;
        float d0 = 0.0f;
#pragma unroll
        for (int i = 0; i < 10; ++i) {
            const __nv_bfloat16 v0 =
                STREAM_OUT ? ld_cg_bf16_ef(x0 + tid + i * 256, mixed_ef) : ld_cg_bf16(x0 + tid + i * 256);
            const float n0 = __bfloat162float(v0);
            d0 += n0 * g[i];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            d0 += __shfl_down_sync(0xffffffffu, d0, offset);
        }
        if (lane == 0) s_dot[parity * 16u + warp] = d0;
        __syncthreads();
        if (tid == 0) {
            float total0 = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) total0 += s_dot[parity * 16u + w];
            ss_out[r] = 1.0f / (1.0f + __expf(-total0));
        }
    }
}

}  // namespace

// One entry (plain, or with the absmax of `mixed` into *amax_bits, zeroed by
// the caller) per tile configuration. Arguments are qwen4_hc_upmix_hcfused's.
#define QWEN4_HC_UPMIX_ROOF_PARAMS                                                                       \
    const __nv_bfloat16 *__restrict__ act, const __nv_bfloat16 *__restrict__ up, float alpha,          \
        const __nv_bfloat16 *__restrict__ hyper, const float *__restrict__ rstd,                       \
        const __nv_bfloat16 *__restrict__ norm_w, __nv_bfloat16 *__restrict__ mixed, unsigned int num_tokens

#define QWEN4_HC_UPMIX_ROOF_ENTRY(NAME, MINB, BM, WM, ST, SWZ, PF, EPI, HINT, ABL)                       \
    extern "C" __global__ void __launch_bounds__((WM) * 128, MINB) NAME(QWEN4_HC_UPMIX_ROOF_PARAMS) {    \
        extern __shared__ __align__(16) unsigned char smem[];                                            \
        upmix_roof_body<BM, WM, ST, SWZ, PF, EPI, HINT, false, ABL, false, false>(smem, act, up, nullptr, nullptr, alpha, hyper, rstd,   \
                                                                    norm_w, mixed, num_tokens, nullptr); \
    }

#define QWEN4_HC_UPMIX_ROOF_ENTRY_AMAX(NAME, MINB, BM, WM, ST, SWZ, PF, EPI, HINT, ABL)                  \
    extern "C" __global__ void __launch_bounds__((WM) * 128, MINB)                                       \
        NAME(QWEN4_HC_UPMIX_ROOF_PARAMS, unsigned int* __restrict__ amax_bits) {                         \
        extern __shared__ __align__(16) unsigned char smem[];                                            \
        upmix_roof_body<BM, WM, ST, SWZ, PF, EPI, HINT, true, ABL, false, false>(smem, act, up, nullptr, nullptr, alpha, hyper, rstd,    \
                                                                   norm_w, mixed, num_tokens, amax_bits); \
    }

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_r4(QWEN4_HC_UPMIX_ROOF_PARAMS) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_body<128, 2, 3, true, 1, true, true, false, 0, true, false>(smem, act, up, nullptr, nullptr, alpha, hyper, rstd,
                                                                    norm_w, mixed, num_tokens, nullptr);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_r4_amax(QWEN4_HC_UPMIX_ROOF_PARAMS, unsigned int* __restrict__ amax_bits) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_body<128, 2, 3, true, 1, true, true, true, 0, true, false>(smem, act, up, nullptr, nullptr, alpha, hyper, rstd,
                                                                   norm_w, mixed, num_tokens, amax_bits);
}

// Strip-mined wide-prefill twins (`STREAM_OUT = true`): `mixed` stores carry
// `L2::evict_first` so 80 MiB of output rows never evict `folded`, `up`, or
// `hyper'[strip]` from L2 across strips. Bit-identical (0-ULP) to the base entries.
extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_r4_stream(QWEN4_HC_UPMIX_ROOF_PARAMS) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_body<128, 2, 3, true, 1, true, true, false, 0, true, false, true>(
        smem, act, up, nullptr, nullptr, alpha, hyper, rstd, norm_w, mixed, num_tokens, nullptr);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_r4_stream_amax(QWEN4_HC_UPMIX_ROOF_PARAMS, unsigned int* __restrict__ amax_bits) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_body<128, 2, 3, true, 1, true, true, true, 0, true, false, true>(
        smem, act, up, nullptr, nullptr, alpha, hyper, rstd, norm_w, mixed, num_tokens, amax_bits);
}

// t128_w4: on-the-fly NVFP4 dequantization into shared memory + permanent L2 pinning
extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_w4(
        const __nv_bfloat16* __restrict__ act,
        const unsigned char* __restrict__ up_packed,
        const unsigned char* __restrict__ up_scales,
        float alpha,
        const __nv_bfloat16* __restrict__ hyper,
        const float* __restrict__ rstd,
        const __nv_bfloat16* __restrict__ norm_w,
        __nv_bfloat16* __restrict__ mixed,
        unsigned int num_tokens
    ) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_body<128, 2, 3, true, 1, true, true, false, 0, true, true>(
        smem, act, nullptr, up_packed, up_scales, alpha, hyper, rstd, norm_w, mixed, num_tokens, nullptr
    );
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_w4_amax(
        const __nv_bfloat16* __restrict__ act,
        const unsigned char* __restrict__ up_packed,
        const unsigned char* __restrict__ up_scales,
        float alpha,
        const __nv_bfloat16* __restrict__ hyper,
        const float* __restrict__ rstd,
        const __nv_bfloat16* __restrict__ norm_w,
        __nv_bfloat16* __restrict__ mixed,
        unsigned int num_tokens,
        unsigned int* __restrict__ amax_bits
    ) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_body<128, 2, 3, true, 1, true, true, true, 0, true, true>(
        smem, act, nullptr, up_packed, up_scales, alpha, hyper, rstd, norm_w, mixed, num_tokens, amax_bits
    );
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_w4_stream(
        const __nv_bfloat16* __restrict__ act,
        const unsigned char* __restrict__ up_packed,
        const unsigned char* __restrict__ up_scales,
        float alpha,
        const __nv_bfloat16* __restrict__ hyper,
        const float* __restrict__ rstd,
        const __nv_bfloat16* __restrict__ norm_w,
        __nv_bfloat16* __restrict__ mixed,
        unsigned int num_tokens
    ) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_body<128, 2, 3, true, 1, true, true, false, 0, true, true, true>(
        smem, act, nullptr, up_packed, up_scales, alpha, hyper, rstd, norm_w, mixed, num_tokens, nullptr
    );
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_w4_stream_amax(
        const __nv_bfloat16* __restrict__ act,
        const unsigned char* __restrict__ up_packed,
        const unsigned char* __restrict__ up_scales,
        float alpha,
        const __nv_bfloat16* __restrict__ hyper,
        const float* __restrict__ rstd,
        const __nv_bfloat16* __restrict__ norm_w,
        __nv_bfloat16* __restrict__ mixed,
        unsigned int num_tokens,
        unsigned int* __restrict__ amax_bits
    ) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_body<128, 2, 3, true, 1, true, true, true, 0, true, true, true>(
        smem, act, nullptr, up_packed, up_scales, alpha, hyper, rstd, norm_w, mixed, num_tokens, amax_bits
    );
}

// t64: the incumbent's tile and K loop (3 CTAs/SM) + PF + EPI + HINT.
// Grid (80, ceil(T / 64)), block 256, dynamic smem 30752 bytes.
QWEN4_HC_UPMIX_ROOF_ENTRY(qwen4_hc_upmix_roof_t64, 3, 64, 2, 2, false, 1, true, true, 0)
QWEN4_HC_UPMIX_ROOF_ENTRY_AMAX(qwen4_hc_upmix_roof_t64_amax, 3, 64, 2, 2, false, 1, true, true, 0)

// t128: 128-row tiles on a 3-stage swizzled ring (2 CTAs/SM) + PF + EPI + HINT.
// Grid (80, ceil(T / 128)), block 256, dynamic smem 49184 bytes.
QWEN4_HC_UPMIX_ROOF_ENTRY(qwen4_hc_upmix_roof_t128, 2, 128, 2, 3, true, 1, true, true, 0)
QWEN4_HC_UPMIX_ROOF_ENTRY_AMAX(qwen4_hc_upmix_roof_t128_amax, 2, 128, 2, 3, true, 1, true, true, 0)

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_stream(QWEN4_HC_UPMIX_ROOF_PARAMS) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_body<128, 2, 3, true, 1, true, true, false, 0, false, false, true>(
        smem, act, up, nullptr, nullptr, alpha, hyper, rstd, norm_w, mixed, num_tokens, nullptr);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_stream_amax(QWEN4_HC_UPMIX_ROOF_PARAMS, unsigned int* __restrict__ amax_bits) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_body<128, 2, 3, true, 1, true, true, true, 0, false, false, true>(
        smem, act, up, nullptr, nullptr, alpha, hyper, rstd, norm_w, mixed, num_tokens, amax_bits);
}

// Single-kernel fused up+mix + MXFP8 `(xq, xs)` + optional Last-CTA L2-hot
// shared-expert gate scalars `ss_out` (`grid = (80, ceil(T / 128))`, 49,184 B
// SMEM, 2 CTAs/SM). Strict 0-ULP parity on `mixed`, `amax_bits`, `xq`, `xs`,
// and `ss_out`. Pass `ss_out = nullptr` to emit MXFP8 without gate scalars.
extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_mxfp8_gate(
        QWEN4_HC_UPMIX_ROOF_PARAMS,
        unsigned char* __restrict__ xq,
        unsigned char* __restrict__ xs,
        const __nv_bfloat16* __restrict__ shared_gate,
        float* __restrict__ ss_out,
        unsigned int* __restrict__ tile_done) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_mxfp8_gate_body<false, false, false>(
        smem, act, up, nullptr, nullptr, alpha, hyper, rstd, norm_w, mixed, num_tokens,
        nullptr, xq, xs, shared_gate, ss_out, tile_done);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_mxfp8_gate_amax(
        QWEN4_HC_UPMIX_ROOF_PARAMS,
        unsigned int* __restrict__ amax_bits,
        unsigned char* __restrict__ xq,
        unsigned char* __restrict__ xs,
        const __nv_bfloat16* __restrict__ shared_gate,
        float* __restrict__ ss_out,
        unsigned int* __restrict__ tile_done) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_mxfp8_gate_body<true, false, false>(
        smem, act, up, nullptr, nullptr, alpha, hyper, rstd, norm_w, mixed, num_tokens,
        amax_bits, xq, xs, shared_gate, ss_out, tile_done);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_r4_mxfp8_gate(
        QWEN4_HC_UPMIX_ROOF_PARAMS,
        unsigned char* __restrict__ xq,
        unsigned char* __restrict__ xs,
        const __nv_bfloat16* __restrict__ shared_gate,
        float* __restrict__ ss_out,
        unsigned int* __restrict__ tile_done) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_mxfp8_gate_body<false, false>(
        smem, act, up, nullptr, nullptr, alpha, hyper, rstd, norm_w, mixed, num_tokens,
        nullptr, xq, xs, shared_gate, ss_out, tile_done);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_r4_mxfp8_gate_amax(
        QWEN4_HC_UPMIX_ROOF_PARAMS,
        unsigned int* __restrict__ amax_bits,
        unsigned char* __restrict__ xq,
        unsigned char* __restrict__ xs,
        const __nv_bfloat16* __restrict__ shared_gate,
        float* __restrict__ ss_out,
        unsigned int* __restrict__ tile_done) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_mxfp8_gate_body<true, false>(
        smem, act, up, nullptr, nullptr, alpha, hyper, rstd, norm_w, mixed, num_tokens,
        amax_bits, xq, xs, shared_gate, ss_out, tile_done);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_w4_mxfp8_gate(
        const __nv_bfloat16* __restrict__ act,
        const unsigned char* __restrict__ up_packed,
        const unsigned char* __restrict__ up_scales,
        float alpha,
        const __nv_bfloat16* __restrict__ hyper,
        const float* __restrict__ rstd,
        const __nv_bfloat16* __restrict__ norm_w,
        __nv_bfloat16* __restrict__ mixed,
        unsigned int num_tokens,
        unsigned char* __restrict__ xq,
        unsigned char* __restrict__ xs,
        const __nv_bfloat16* __restrict__ shared_gate,
        float* __restrict__ ss_out,
        unsigned int* __restrict__ tile_done) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_mxfp8_gate_body<false, true>(
        smem, act, nullptr, up_packed, up_scales, alpha, hyper, rstd, norm_w, mixed, num_tokens,
        nullptr, xq, xs, shared_gate, ss_out, tile_done);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_w4_mxfp8_gate_amax(
        const __nv_bfloat16* __restrict__ act,
        const unsigned char* __restrict__ up_packed,
        const unsigned char* __restrict__ up_scales,
        float alpha,
        const __nv_bfloat16* __restrict__ hyper,
        const float* __restrict__ rstd,
        const __nv_bfloat16* __restrict__ norm_w,
        __nv_bfloat16* __restrict__ mixed,
        unsigned int num_tokens,
        unsigned int* __restrict__ amax_bits,
        unsigned char* __restrict__ xq,
        unsigned char* __restrict__ xs,
        const __nv_bfloat16* __restrict__ shared_gate,
        float* __restrict__ ss_out,
        unsigned int* __restrict__ tile_done) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_mxfp8_gate_body<true, true>(
        smem, act, nullptr, up_packed, up_scales, alpha, hyper, rstd, norm_w, mixed, num_tokens,
        amax_bits, xq, xs, shared_gate, ss_out, tile_done);
}

// Strip-mined streaming-output (`L2::evict_first` on `mixed` + `xq`) twins of the
// fused MXFP8 + Last-CTA gate-scalar entries so strip outputs never evict `folded`,
// `up`, or `hyper'[strip]` from L2 across strips. Strict 0-ULP identical.
extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_stream_mxfp8_gate(
        QWEN4_HC_UPMIX_ROOF_PARAMS,
        unsigned char* __restrict__ xq,
        unsigned char* __restrict__ xs,
        const __nv_bfloat16* __restrict__ shared_gate,
        float* __restrict__ ss_out,
        unsigned int* __restrict__ tile_done) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_mxfp8_gate_body<false, false, false, true>(
        smem, act, up, nullptr, nullptr, alpha, hyper, rstd, norm_w, mixed, num_tokens,
        nullptr, xq, xs, shared_gate, ss_out, tile_done);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_stream_mxfp8_gate_amax(
        QWEN4_HC_UPMIX_ROOF_PARAMS,
        unsigned int* __restrict__ amax_bits,
        unsigned char* __restrict__ xq,
        unsigned char* __restrict__ xs,
        const __nv_bfloat16* __restrict__ shared_gate,
        float* __restrict__ ss_out,
        unsigned int* __restrict__ tile_done) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_mxfp8_gate_body<true, false, false, true>(
        smem, act, up, nullptr, nullptr, alpha, hyper, rstd, norm_w, mixed, num_tokens,
        amax_bits, xq, xs, shared_gate, ss_out, tile_done);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_r4_stream_mxfp8_gate(
        QWEN4_HC_UPMIX_ROOF_PARAMS,
        unsigned char* __restrict__ xq,
        unsigned char* __restrict__ xs,
        const __nv_bfloat16* __restrict__ shared_gate,
        float* __restrict__ ss_out,
        unsigned int* __restrict__ tile_done) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_mxfp8_gate_body<false, false, true, true>(
        smem, act, up, nullptr, nullptr, alpha, hyper, rstd, norm_w, mixed, num_tokens,
        nullptr, xq, xs, shared_gate, ss_out, tile_done);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_r4_stream_mxfp8_gate_amax(
        QWEN4_HC_UPMIX_ROOF_PARAMS,
        unsigned int* __restrict__ amax_bits,
        unsigned char* __restrict__ xq,
        unsigned char* __restrict__ xs,
        const __nv_bfloat16* __restrict__ shared_gate,
        float* __restrict__ ss_out,
        unsigned int* __restrict__ tile_done) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_mxfp8_gate_body<true, false, true, true>(
        smem, act, up, nullptr, nullptr, alpha, hyper, rstd, norm_w, mixed, num_tokens,
        amax_bits, xq, xs, shared_gate, ss_out, tile_done);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_w4_stream_mxfp8_gate(
        const __nv_bfloat16* __restrict__ act,
        const unsigned char* __restrict__ up_packed,
        const unsigned char* __restrict__ up_scales,
        float alpha,
        const __nv_bfloat16* __restrict__ hyper,
        const float* __restrict__ rstd,
        const __nv_bfloat16* __restrict__ norm_w,
        __nv_bfloat16* __restrict__ mixed,
        unsigned int num_tokens,
        unsigned char* __restrict__ xq,
        unsigned char* __restrict__ xs,
        const __nv_bfloat16* __restrict__ shared_gate,
        float* __restrict__ ss_out,
        unsigned int* __restrict__ tile_done) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_mxfp8_gate_body<false, true, true, true>(
        smem, act, nullptr, up_packed, up_scales, alpha, hyper, rstd, norm_w, mixed, num_tokens,
        nullptr, xq, xs, shared_gate, ss_out, tile_done);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    qwen4_hc_upmix_roof_t128_w4_stream_mxfp8_gate_amax(
        const __nv_bfloat16* __restrict__ act,
        const unsigned char* __restrict__ up_packed,
        const unsigned char* __restrict__ up_scales,
        float alpha,
        const __nv_bfloat16* __restrict__ hyper,
        const float* __restrict__ rstd,
        const __nv_bfloat16* __restrict__ norm_w,
        __nv_bfloat16* __restrict__ mixed,
        unsigned int num_tokens,
        unsigned int* __restrict__ amax_bits,
        unsigned char* __restrict__ xq,
        unsigned char* __restrict__ xs,
        const __nv_bfloat16* __restrict__ shared_gate,
        float* __restrict__ ss_out,
        unsigned int* __restrict__ tile_done) {
    extern __shared__ __align__(16) unsigned char smem[];
    upmix_roof_mxfp8_gate_body<true, true, true, true>(
        smem, act, nullptr, up_packed, up_scales, alpha, hyper, rstd, norm_w, mixed, num_tokens,
        amax_bits, xq, xs, shared_gate, ss_out, tile_done);
}

// Bit-identical (0-ULP) router tensor-scaled FP8 E4M3 activation quantization
// matching `qwen4_mxfp8_quant_rows_v16q4_gate`'s `a8_out` path (exact IEEE
// `448.0f / amax`, `amax / 448.0f`, `cvt.rn.satfinite.e4m3x2.f32`, L2 hints,
// and L2 discard on dead BF16 `src`), without rewriting `(xq, xs)` or `ss_out`.
// Launch: grid (min(rows, 1024), 1, 1), block (256, 1, 1).
extern "C" __global__ void __launch_bounds__(256) qwen4_hc_upmix_fp8_act_from_bf16(
    const __nv_bfloat16* __restrict__ src,
    const unsigned int* __restrict__ amax_bits,
    unsigned char* __restrict__ a8_out,
    float* __restrict__ a_scale_out,
    unsigned int rows) {
    constexpr unsigned int kCols = 2560u;
    const unsigned int bt = threadIdx.x;
    const float fp8_amax = fmaxf(__uint_as_float(amax_bits[0]), 1e-30f);
    const float fp8_inv = 448.0f / fp8_amax;
    if (a_scale_out != nullptr && blockIdx.x == 0u && bt == 0u) {
        a_scale_out[0] = fp8_amax / 448.0f;
    }
    const uint64_t pol_el = policy_evict_last();
    const uint64_t pol_ef = policy_evict_first();
    const bool src_aligned128 = ((reinterpret_cast<uintptr_t>(src) & 127u) == 0u);

    for (unsigned int r = blockIdx.x; r < rows; r += gridDim.x) {
        const __nv_bfloat16* x_row = src + static_cast<unsigned long long>(r) * kCols;
        const uint4* xr4 = reinterpret_cast<const uint4*>(x_row);
        uint2* out2 = reinterpret_cast<uint2*>(
            a8_out + static_cast<unsigned long long>(r) * kCols);
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const unsigned int idx = pass * 256u + bt;
            if (pass == 0 || idx < (kCols / 8u)) {
                uint4 w;
                asm volatile("ld.global.ca.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                             : "=r"(w.x), "=r"(w.y), "=r"(w.z), "=r"(w.w)
                             : "l"(xr4 + idx), "l"(pol_ef));
                const unsigned u[4] = {w.x, w.y, w.z, w.w};
                float v[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    v[2 * i] = __uint_as_float(u[i] << 16);
                    v[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
                }
                const unsigned wa0 = upmix_pack_e4m3x4(
                    v[0] * fp8_inv, v[1] * fp8_inv, v[2] * fp8_inv, v[3] * fp8_inv);
                const unsigned wa1 = upmix_pack_e4m3x4(
                    v[4] * fp8_inv, v[5] * fp8_inv, v[6] * fp8_inv, v[7] * fp8_inv);
                if (r < 6400u) {
                    store_row8_hint(out2 + idx, make_uint2(wa0, wa1), pol_el);
                } else {
                    out2[idx] = make_uint2(wa0, wa1);
                }
            }
        }
        if (src_aligned128) {
            __syncthreads();
            if (bt < 40u) {
                asm volatile("discard.global.L2 [%0], 128;\n"
                             :: "l"(reinterpret_cast<const unsigned char*>(x_row) + bt * 128u)
                             : "memory");
            }
        }
    }
}

// Post-upmix L2 discard for dead scratch buffers (s.down, s.rstd, s.folded, s.up)
// on the 2-kernel fallback path. Launched on the same stream after upmix_roof
// finishes so all 80 H-tiles have consumed the buffers before any line is discarded.
// Only discards complete 128-byte aligned cache lines strictly inside [ptr, ptr + bytes).
extern "C" __global__ void __launch_bounds__(256)
    qwen4_hc_discard_l2_ranges(
        const unsigned char* p0,
        unsigned long long bytes0,
        const unsigned char* p1,
        unsigned long long bytes1,
        const unsigned char* p2,
        unsigned long long bytes2,
        const unsigned char* p3,
        unsigned long long bytes3) {
    auto discard_range = [](const unsigned char* ptr, unsigned long long bytes) {
        if (ptr == nullptr || bytes < 128ull) return;
        const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
        const uintptr_t start = (base + 127u) & ~uintptr_t(127u);
        const uintptr_t end = (base + bytes) & ~uintptr_t(127u);
        if (end <= start) return;
        const unsigned long long lines = (unsigned long long)((end - start) >> 7);
        const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
        for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
             i < lines; i += stride) {
            const unsigned char* line = reinterpret_cast<const unsigned char*>(start + (uintptr_t)(i << 7));
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(line) : "memory");
        }
    };
    discard_range(p0, bytes0);
    discard_range(p1, bytes1);
    discard_range(p2, bytes2);
    discard_range(p3, bytes3);
}
