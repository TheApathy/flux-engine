// SPDX-License-Identifier: AGPL-3.0-only
//
// Native Blackwell (SM120/SM121) BitDecoding-FP4 + 6-op SWAR RaZeR-KV Paged
// Decode Attention & Block-16 KV Quantization Kernels.
//
// Design overview:
//   1. Self-contained `block_size = 16` KV layout (identical pool byte footprint
//      to Flux's existing NVFP4 `PagedKvCache`, zero extra pool allocation,
//      directly compatible with GateSight 16-token block eviction):
//      - K block (token-wise, raw-copyable):
//          data  [16, num_kv_heads, HDIM/2]  packed E2M1 nibbles
//          scale [16, num_kv_heads, HDIM/16] UE4M3 micro-block scales
//        Each key is transformed via optional static mean subtraction +
//        orthogonal 256-point Walsh-Hadamard rotation `K' = H(K - mu_k)` and
//        quantized with Four-Over-Six (`{amax/6, amax/4}`) MSE scale search
//        (or RaZeR-KV `±5.0` codebook with `neg_sv` in UE4M3 bit 7).
//      - V block (block-16 channel-transposed `V^T`, self-contained per block):
//          data  [num_kv_heads, HDIM, 8]     packed E2M1 nibbles (16 toks/ch)
//          scale [num_kv_heads, HDIM, 1]     UE4M3 channel scale across 16 toks
//        Matches `NVFP4_GROUP_SIZE = 16` and feeds `O^T = V^T @ P^T` directly
//        into `m16n8k64` `scale_vec::4X` FP4 Tensor Cores with zero inner-loop
//        transpose.
//   2. Native SM120/SM121 FP4 Tensor Core decode attention (`m16n8k64`,
//      `kind::mxf4nvf4.block_scale.scale_vec::4X`, `.ue4m3` scales):
//      - Dual-Residual Stacked Query Quantization (`Q^(1)` + `Q^(2)` packed into
//        adjacent columns of `B` in `m16n8k64` so every lane sums `d[0]+d[1]`
//        and `d[2]+d[3]` in registers with zero shuffles).
//      - Transposed Softmax (`S^T = K @ Q^T`) with parallel warp butterfly
//        reductions (`__shfl_xor_sync` XOR 4/8/16) across 16-key blocks.
//      - Underflow-Aware Probability Scaling (`c = 8` folded as `+3.0f` in
//        `exp2f`, `kappa = 2` when `c * p_max <= 21/1024`, else `6`), pushing
//        the E4M3 underflow floor 24x lower to `max(p_b) < 2^-12`.
//      - Optional 6-op branchless SWAR RaZeR-KV (`-0.0` (`0x8`) -> `±4.0 + ±1.0`)
//        reading `neg_sv` from bit 7 of `.ue4m3` scale bytes (zero range loss).
//      - Optional fused BF16 attention-sink sidecar merge in prologue (split 0).
//      - 2-stage `cp.async` double-buffered `B_k = 64` (4 blocks) SMEM pipeline
//        fitting inside 42 KiB static SMEM -> 2 CTAs/SM on GB10 & SM120.
//   3. Multi-arch safety:
//      - Block-scaled `mma.sync...kind::mxf4nvf4` is guarded under
//        `__CUDA_ARCH__ >= 1200` (arch/family-specific SM120/SM121 targets)
//        with a clean `__trap()` fallback on SM103a/generic PTX so every target
//        in `atlas-kernels` compiles with 0 B stack frame and 0 B spills.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// ============================================================================
// Blackwell ISA feature guards (matching Flux `hc_common.cuh` contract)
// ============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    (defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || \
     defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_SPECIFIC__))
#define BITFP4_HAS_E2M1_CVT 1
#else
#define BITFP4_HAS_E2M1_CVT 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && \
    (defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || \
     defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_SPECIFIC__))
#define BITFP4_HAS_SM120_MX_MMA 1
#else
#define BITFP4_HAS_SM120_MX_MMA 0
#endif

namespace bitfp4 {

constexpr unsigned int kHdim = 256u;
constexpr unsigned int kHdimHalf = kHdim / 2u;         // 128 bytes
constexpr unsigned int kHdimGroups = kHdim / 16u;      // 16 groups
constexpr unsigned int kBlockSize = 16u;               // Flux NVFP4 block_size=16
constexpr unsigned int kBlocksPerTile = 4u;            // B_k = 64 = 4 * 16
constexpr unsigned int kTileK = kBlocksPerTile * kBlockSize;  // 64
constexpr unsigned int kStages = 2u;
constexpr unsigned int kWarps = 4u;
constexpr unsigned int kThreads = kWarps * 32u;        // 128
constexpr float kLog2E = 1.44269504088896340736f;
constexpr float kNegInf = -1.0e30f;
// BitDecoding-FP4 Eq. 4 underflow-aware probability threshold:
//   c = 8.0f (folded as +3.0f in exp2f), tau = 21/1024.
//   Since p_max_c = c * p_max, `p_max <= tau / c` <=> `p_max_c <= tau`.
constexpr float kUnderflowTauC = 21.0f / 1024.0f;

// Runtime flags bitmask shared by Rust host and CUDA kernels.
constexpr unsigned int kFlagApplyWht = 1u << 0;
constexpr unsigned int kFlagFourOverSix = 1u << 1;
constexpr unsigned int kFlagRazerKv = 1u << 2;

// ============================================================================
// Low-level Blackwell PTX conversion, MMA, SWAR RaZeR, and WHT primitives
// ============================================================================

__device__ __forceinline__ float e2m1_nibble_to_f32(unsigned int nib) {
    const unsigned int mag = nib & 7u;
    float val = (mag == 0u) ? 0.0f
              : (mag == 1u) ? 0.5f
              : static_cast<float>(2u | (mag & 1u)) *
                static_cast<float>(1u << ((mag >> 1) - 1u)) * 0.5f;
    return (nib & 8u) ? -val : val;
}

__device__ __forceinline__ float razer_nibble_to_f32(unsigned int nib, unsigned int neg_sv) {
    if (nib == 8u) {
        return neg_sv ? -5.0f : 5.0f;
    }
    return e2m1_nibble_to_f32(nib);
}

__device__ __forceinline__ unsigned char cvt_f32_to_ue4m3(float x) {
    // Clamp to non-negative finite UE4M3 range [0, 448.0f]; bit 7 is always 0.
    const float clamped = fminf(fmaxf(x, 0.0f), 448.0f);
#if BITFP4_HAS_E2M1_CVT
    unsigned short p;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p) : "f"(clamped), "f"(0.0f));
    return static_cast<unsigned char>(p & 0x7fu);
#else
    __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(clamped, __NV_SATFINITE, __NV_E4M3);
    return static_cast<unsigned char>(s & 0x7fu);
#endif
}

__device__ __forceinline__ float cvt_ue4m3_to_f32(unsigned char s) {
    const unsigned char clean = static_cast<unsigned char>(s & 0x7fu);
#if BITFP4_HAS_E2M1_CVT
    unsigned int h2;
    const unsigned short sb2 = static_cast<unsigned short>(clean);
    asm("{ .reg .b16 hs; mov.b16 hs, %1; cvt.rn.f16x2.e4m3x2 %0, hs; }" : "=r"(h2) : "h"(sb2));
    return __half2float(*reinterpret_cast<const __half*>(&h2));
#else
    return __half2float(__nv_cvt_fp8_to_halfraw(clean, __NV_E4M3));
#endif
}

__device__ __forceinline__ unsigned char quant2_e2m1(float v0, float v1) {
#if BITFP4_HAS_E2M1_CVT
    unsigned int r;
    asm("{\n\t"
        ".reg .b8 b0;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n\t"
        "mov.b32 %0, {b0, 0, 0, 0};\n\t"
        "}"
        : "=r"(r)
        : "f"(v0), "f"(v1));
    return static_cast<unsigned char>(r & 0xffu);
#else
    auto q1 = [](float x) -> unsigned int {
        const unsigned int sign = (x < 0.0f) ? 8u : 0u;
        const float a = fabsf(x);
        unsigned int mag;
        if (a < 0.25f) mag = 0u;
        else if (a <= 0.75f) mag = 1u;
        else if (a < 1.25f) mag = 2u;
        else if (a <= 1.75f) mag = 3u;
        else if (a < 2.5f) mag = 4u;
        else if (a <= 3.5f) mag = 5u;
        else if (a < 5.0f) mag = 6u;
        else mag = 7u;
        return (mag == 0u) ? 0u : (sign | mag);
    };
    return static_cast<unsigned char>(q1(v0) | (q1(v1) << 4));
#endif
}

__device__ __forceinline__ unsigned int quant8_e2m1(const float v[8], float q_inv) {
#if BITFP4_HAS_E2M1_CVT
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
    const unsigned int b0 = quant2_e2m1(v[0] * q_inv, v[1] * q_inv);
    const unsigned int b1 = quant2_e2m1(v[2] * q_inv, v[3] * q_inv);
    const unsigned int b2 = quant2_e2m1(v[4] * q_inv, v[5] * q_inv);
    const unsigned int b3 = quant2_e2m1(v[6] * q_inv, v[7] * q_inv);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
#endif
}

// Quantize a single float value `x` into E2M1 or RaZeR-KV (where `0x8` maps to
// `+5.0` when `neg_sv == 0` or `-5.0` when `neg_sv == 1`).
__device__ __forceinline__ unsigned int quant1_razer_nibble(float x, unsigned int neg_sv) {
    const unsigned int sign = (x < 0.0f) ? 8u : 0u;
    const float a = fabsf(x);
    // Check if value matches the special `±5.0` RaZeR code (same sign as special value).
    const bool same_sign_as_sv = ((sign != 0u) == (neg_sv != 0u));
    if (same_sign_as_sv && a >= 4.5f && a <= 5.5f) {
        return 8u;  // Remapped `-0.0` -> ±5.0
    }
    unsigned int mag;
    if (a < 0.25f) mag = 0u;
    else if (a <= 0.75f) mag = 1u;
    else if (a < 1.25f) mag = 2u;
    else if (a <= 1.75f) mag = 3u;
    else if (a < 2.5f) mag = 4u;
    else if (a <= 3.5f) mag = 5u;
    else if (a < 5.0f) mag = 6u;
    else mag = 7u;
    // Never emit `-0.0` (0x8) for ordinary values; zero maps to `+0.0` (0x0).
    return (mag == 0u) ? 0u : (sign | mag);
}

// 6-op branchless SWAR RaZeR-KV nibble splitter:
//   For every 4-bit nibble in `x`:
//     - If nibble != 0x8: `x_main` keeps nibble unchanged, `x_comp` = 0x0 (+0.0).
//     - If nibble == 0x8 (`-0.0`):
//         neg_sv == 0 -> `x_main` = 0x6 (+4.0), `x_comp` = 0x2 (+1.0) => +5.0
//         neg_sv == 1 -> `x_main` = 0xE (-4.0), `x_comp` = 0xA (-1.0) => -5.0
// No cross-nibble carry can occur because `(x & 0x7) + 0x7 <= 0xE < 0x10`.
__device__ __forceinline__ void razer_swar_split_u32(
    unsigned int x,
    unsigned int neg_sv,
    unsigned int& x_main,
    unsigned int& x_comp
) {
    const unsigned int neg_mask = 0u - (neg_sv & 1u);
    const unsigned int lo3 = x & 0x77777777u;
    const unsigned int m = x & ~(lo3 + 0x77777777u) & 0x88888888u;
    const unsigned int m2 = m >> 2;
    const unsigned int m4 = m >> 1;
    x_main = (x ^ (m & ~neg_mask)) | m4 | m2;
    x_comp = (m & neg_mask) | m2;
}

__device__ __forceinline__ void mma_nvfp4_m16n8k64_raw(
    float (&d)[4],
    unsigned int a0,
    unsigned int a1,
    unsigned int a2,
    unsigned int a3,
    unsigned int b0,
    unsigned int b1,
    unsigned int sfa,
    unsigned int sfb
) {
#if BITFP4_HAS_SM120_MX_MMA
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
    (void)d; (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1; (void)sfa; (void)sfb;
    __trap();
#endif
}

template <bool kRazerKv>
__device__ __forceinline__ void mma_nvfp4_m16n8k64(
    float (&d)[4],
    unsigned int a0,
    unsigned int a1,
    unsigned int a2,
    unsigned int a3,
    unsigned int b0,
    unsigned int b1,
    unsigned int sfa_raw,
    unsigned int sfb,
    unsigned int lane
) {
    if constexpr (!kRazerKv) {
        mma_nvfp4_m16n8k64_raw(d, a0, a1, a2, a3, b0, b1, sfa_raw & 0x7f7f7f7fu, sfb);
    } else {
        // Every lane holds `sfa_raw` for `m_sfa = group_id + 8*(lane & 1)`, while
        // `a0, a2` come from `fr0 = group_id` and `a1, a3` come from `fr1 = group_id + 8`.
        // One XOR-1 warp shuffle gives every lane both `sfa_fr0` and `sfa_fr1`.
        const unsigned int sfa_other = __shfl_xor_sync(0xffffffffu, sfa_raw, 1);
        const unsigned int sfa_fr0 = (lane & 1u) ? sfa_other : sfa_raw;
        const unsigned int sfa_fr1 = (lane & 1u) ? sfa_raw : sfa_other;
        const unsigned int blk_lo = (lane & 3u) >> 1;  // 0 for tid4 in {0,1}; 1 for {2,3}
        const unsigned int shift_lo = 7u + (blk_lo << 3);
        const unsigned int shift_hi = 23u + (blk_lo << 3);

        unsigned int a0_m, a0_c, a1_m, a1_c, a2_m, a2_c, a3_m, a3_c;
        razer_swar_split_u32(a0, (sfa_fr0 >> shift_lo) & 1u, a0_m, a0_c);
        razer_swar_split_u32(a1, (sfa_fr1 >> shift_lo) & 1u, a1_m, a1_c);
        razer_swar_split_u32(a2, (sfa_fr0 >> shift_hi) & 1u, a2_m, a2_c);
        razer_swar_split_u32(a3, (sfa_fr1 >> shift_hi) & 1u, a3_m, a3_c);

        const unsigned int sfa_clean = sfa_raw & 0x7f7f7f7fu;
        mma_nvfp4_m16n8k64_raw(d, a0_m, a1_m, a2_m, a3_m, b0, b1, sfa_clean, sfb);
        if (__any_sync(0xffffffffu, (a0_c | a1_c | a2_c | a3_c) != 0u)) {
            mma_nvfp4_m16n8k64_raw(d, a0_c, a1_c, a2_c, a3_c, b0, b1, sfa_clean, sfb);
        }
    }
}

// Plain normalized 256-point orthogonal Walsh-Hadamard transform (H^T H = I).
// 32 lanes x 8 floats = 256 elements; normalization factor 1/sqrt(256) = 1/16.
__device__ __forceinline__ void wht256_warp_plain(float vals[8], unsigned int lane) {
    #pragma unroll
    for (int stride = 1; stride <= 4; stride <<= 1) {
        #pragma unroll
        for (int i = 0; i < 8; i += stride * 2) {
            #pragma unroll
            for (int j = 0; j < stride; ++j) {
                const float a = vals[i + j];
                const float b = vals[i + j + stride];
                vals[i + j] = a + b;
                vals[i + j + stride] = a - b;
            }
        }
    }
    #pragma unroll
    for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float other = __shfl_xor_sync(0xffffffffu, vals[i], xor_mask);
            vals[i] = (lane & static_cast<unsigned int>(xor_mask)) ? (other - vals[i]) : (vals[i] + other);
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        vals[i] *= 0.0625f;
    }
}

// Plain `cp.async.cg.shared.global` with zero-fill predicate (no L2::cache_hint
// descriptor -> immune to CUDA 13.0 ptxas odd `desc[UR]` allocator bug).
__device__ __forceinline__ void cp_async_cg_16_z(
    unsigned int smem_addr,
    const void* gmem_ptr,
    bool pred
) {
    const int src_bytes = pred ? 16 : 0;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_bytes)
        : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_prev() {
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}

// ============================================================================
// Block-16 MSE evaluation helpers for Four-Over-Six and RaZeR-KV quantization
// ============================================================================

// Compute reconstruction MSE of 8 elements (half of a 16-elem group) under a
// candidate UE4M3 scale `sc_f` and optional RaZeR `±5.0` sign `neg_sv`.
// The two lanes holding a 16-element group sum their halves via XOR-1 shuffle.
__device__ __forceinline__ float half_group_mse_e2m1(
    const float v[8],
    float sc_f,
    float inv_sc,
    unsigned int& packed_word
) {
    packed_word = quant8_e2m1(v, inv_sc);
    float err = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int nib = (packed_word >> (i * 4)) & 0xfu;
        const float diff = v[i] - e2m1_nibble_to_f32(nib) * sc_f;
        err += diff * diff;
    }
    return err;
}

__device__ __forceinline__ float half_group_mse_razer(
    const float v[8],
    float sc_f,
    float inv_sc,
    unsigned int neg_sv,
    unsigned int& packed_word
) {
    unsigned int w = 0u;
    float err = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int nib = quant1_razer_nibble(v[i] * inv_sc, neg_sv);
        w |= (nib << (i * 4));
        const float diff = v[i] - razer_nibble_to_f32(nib, neg_sv) * sc_f;
        err += diff * diff;
    }
    packed_word = w;
    return err;
}

// ============================================================================
// Core SM120/SM121 BitDecoding-FP4 Decode Kernel Body (GQA & K=gamma unified)
// ============================================================================
//
// `kQSub` = 8 for decode (up to 8 GQA heads/CTA), 16 for K=gamma verify
// (up to 16 stacked `(q_row, q_head)` queries per CTA).
//
// Memory & occupancy profile (HDIM=256, B_k=64, 2 stages, 128 threads):
//   - kQSub =  8: 40,240 B static SMEM (39.3 KiB) -> 2 CTAs/SM on GB10 & SM120.
//   - kQSub = 16: 42,832 B static SMEM (41.8 KiB) -> 2 CTAs/SM on GB10 & SM120.

template <bool kRazerKv, unsigned int kQSub>
struct BitFp4Smem {
    alignas(16) unsigned char sK_data[kStages][kTileK][kHdimHalf];           // 16,384 B
    alignas(16) unsigned char sK_scale[kStages][kTileK][kHdimGroups];        //  2,048 B
    alignas(16) unsigned char sV_data[kStages][kBlocksPerTile][kHdim][8];    // 16,384 B
    alignas(16) unsigned char sV_scale[kStages][kBlocksPerTile][kHdim];      //  2,048 B
    alignas(16) unsigned char sQ_data[2 * kQSub][kHdimHalf];                 //  2,048 / 4,096 B
    alignas(16) unsigned char sQ_scale[2 * kQSub][kHdimGroups];              //    256 /   512 B
    alignas(16) unsigned char sP_data[kQSub][kTileK / 2u];                   //    256 /   512 B
    alignas(16) unsigned char sP_scale[kQSub][kBlocksPerTile];               //     32 /    64 B
    float sQ_logit_scale[kQSub];                                             //     32 /    64 B
    float sBlkMax[kWarps][kQSub];                                            //    128 /   256 B
    float sBlkSum[kWarps][kQSub];                                            //    128 /   256 B
    float sRunningM[kQSub];                                                  //     32 /    64 B
    float sRunningL[kQSub];                                                  //     32 /    64 B
    float sAlpha[kQSub];                                                     //     32 /    64 B
    unsigned int sQSeqLen[kQSub];                                            //     32 /    64 B
};

template <bool kRazerKv, unsigned int kQSub, bool kIsKgamma>
__device__ __forceinline__ void paged_decode_bitfp4_core(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ workspace,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int num_qtile,            // K=gamma query rows (unused when !kIsKgamma)
    const unsigned int flags,                // BITFP4_FLAG_*
    const __nv_bfloat16* __restrict__ K_sink_bf16,   // Optional [num_sink, nkv, HDIM] BF16
    const __nv_bfloat16* __restrict__ V_sink_bf16,   // Optional [num_sink, nkv, HDIM] BF16
    const unsigned int num_sink_tokens               // 0 or 16 (block 0)
) {
    static_assert(kQSub == 8u || kQSub == 16u, "kQSub must be 8 or 16");
    constexpr unsigned int kQHalves = kQSub / 4u;  // 2 for decode, 4 for kgamma
    constexpr unsigned int kQTile8 = kQSub / 8u;   // 1 for decode, 2 for kgamma

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int group_id = lane >> 2;  // 0..7
    const unsigned int tid4 = lane & 3u;      // 0..3

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    unsigned int kv_head = 0u;
    unsigned int split_id = 0u;
    unsigned int seq_idx = 0u;
    unsigned int q_chunk_base = 0u;
    unsigned int total_q_for_kv = 0u;

    if constexpr (!kIsKgamma) {
        // Grid: (num_kv_heads * q_groups_per_kv, num_splits, num_seqs)
        const unsigned int q_groups_per_kv = (gqa_ratio + kQSub - 1u) / kQSub;
        kv_head = blockIdx.x / q_groups_per_kv;
        const unsigned int q_group = blockIdx.x - kv_head * q_groups_per_kv;
        split_id = blockIdx.y;
        seq_idx = blockIdx.z;
        q_chunk_base = q_group * kQSub;
        total_q_for_kv = gqa_ratio;
    } else {
        // Grid: (num_kv_heads, num_q_chunks, num_splits)
        kv_head = blockIdx.x;
        q_chunk_base = blockIdx.y * kQSub;
        split_id = blockIdx.z;
        seq_idx = 0u;
        total_q_for_kv = num_qtile * gqa_ratio;
    }

    if (kv_head >= num_kv_heads || split_id >= num_splits || q_chunk_base >= total_q_for_kv) {
        return;
    }

    const unsigned int num_q_active =
        (total_q_for_kv - q_chunk_base < kQSub) ? (total_q_for_kv - q_chunk_base) : kQSub;

    __shared__ BitFp4Smem<kRazerKv, kQSub> sm;

    // Resolve per-query causal seq_len and max seq_len across active queries.
    if (tid < kQSub) {
        unsigned int sl = 0u;
        if (tid < num_q_active) {
            if constexpr (!kIsKgamma) {
                sl = static_cast<unsigned int>(seq_lens[seq_idx]);
            } else {
                const unsigned int q_flat = q_chunk_base + tid;
                const unsigned int q_row = q_flat / gqa_ratio;
                sl = static_cast<unsigned int>(seq_lens[q_row]);
            }
        }
        sm.sQSeqLen[tid] = sl;
        sm.sRunningM[tid] = kNegInf;
        sm.sRunningL[tid] = 0.0f;
        sm.sAlpha[tid] = 0.0f;
        sm.sQ_logit_scale[tid] = 0.0f;
    }
    __syncthreads();

    unsigned int max_seq_len = 0u;
    #pragma unroll
    for (unsigned int r = 0u; r < kQSub; ++r) {
        max_seq_len = max(max_seq_len, sm.sQSeqLen[r]);
    }
    if (max_seq_len == 0u) {
        return;
    }

    // All queries in a CTA share the same sequence block table (`seq_idx` for
    // decode; row 0's flat chain block table for K=gamma verify).
    const int* my_block_table =
        block_tables + static_cast<unsigned long long>(seq_idx) * max_blocks_per_seq;

    // Determine FP4 block range for this split. When BF16 sinks cover block 0
    // (`num_sink_tokens > 0` and non-null sink pointers), FP4 blocks begin at
    // `sink_blocks = (num_sink_tokens + 15) / 16`, and split 0 fuses the BF16
    // sink tokens in the prologue below.
    const bool use_bf16_sink =
        (K_sink_bf16 != nullptr) && (V_sink_bf16 != nullptr) && (num_sink_tokens > 0u);
    const unsigned int first_fp4_blk = use_bf16_sink ? ((num_sink_tokens + kBlockSize - 1u) / kBlockSize) : 0u;
    const unsigned int total_blocks = (max_seq_len + kBlockSize - 1u) / kBlockSize;
    const unsigned int fp4_blocks = (total_blocks > first_fp4_blk) ? (total_blocks - first_fp4_blk) : 0u;
    const unsigned int blocks_per_split = (fp4_blocks + num_splits - 1u) / num_splits;
    const unsigned int blk_start = first_fp4_blk + split_id * blocks_per_split;
    const unsigned int blk_end = min(total_blocks, blk_start + blocks_per_split);

    // Output accumulators in MMA `m16n8k64` layout:
    //   warp `w` owns channels `w*64 .. w*64 + 63` (4 tiles of 16 channels).
    //   `o_acc[qt][dt][0..3]` for query tile `qt` (8 queries/tile) and channel tile `dt`.
    float o_acc[kQTile8][4][4];
    #pragma unroll
    for (unsigned int qt = 0u; qt < kQTile8; ++qt) {
        #pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
            o_acc[qt][dt][0] = 0.0f;
            o_acc[qt][dt][1] = 0.0f;
            o_acc[qt][dt][2] = 0.0f;
            o_acc[qt][dt][3] = 0.0f;
        }
    }

    // ------------------------------------------------------------------------
    // Prologue: optional BF16 attention-sink fusion (split 0) + optional WHT
    //           + Dual-Residual Stacked Query Quantization (`Q^(1)`, `Q^(2)`)
    // ------------------------------------------------------------------------
    const bool apply_wht = (flags & kFlagApplyWht) != 0u;
    const bool run_sink_now = use_bf16_sink && (split_id == 0u);
    // Reuse `sK_data` (16,384 B) as temporary FP32 sink output buffer `[kQSub, 256]`
    // before the FP4 pipeline starts.
    static_assert(kQSub * kHdim * sizeof(float) <= sizeof(sm.sK_data),
                  "BF16 sink scratch must fit inside sK_data");
    float* sink_o_scratch = reinterpret_cast<float*>(sm.sK_data);

    for (unsigned int r = warp_id; r < kQSub; r += kWarps) {
        if (r < num_q_active) {
            unsigned int q_row = seq_idx;
            unsigned int q_head_local = 0u;
            if constexpr (!kIsKgamma) {
                q_head_local = q_chunk_base + r;
            } else {
                const unsigned int q_flat = q_chunk_base + r;
                q_row = q_flat / gqa_ratio;
                q_head_local = q_flat - q_row * gqa_ratio;
            }
            const unsigned int q_head = kv_head * gqa_ratio + q_head_local;
            const __nv_bfloat16* q_ptr =
                Q + static_cast<unsigned long long>(q_row) * q_stride
                  + static_cast<unsigned long long>(q_head) * kHdim
                  + lane * 8u;

            float qv[8];
            const uint4 q_raw16 = *reinterpret_cast<const uint4*>(q_ptr);
            const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(&q_raw16);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(q2[j]);
                qv[2 * j] = f.x;
                qv[2 * j + 1] = f.y;
            }

            // Optional BF16 sink attention on raw unrotated Q (split 0 only).
            if (run_sink_now) {
                const unsigned int sink_lim = min(num_sink_tokens, sm.sQSeqLen[r]);
                float m_s = kNegInf;
                float l_s = 0.0f;
                float os[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                const float scale_log2 = inv_sqrt_d * kLog2E;
                for (unsigned int ts = 0u; ts < sink_lim; ++ts) {
                    const unsigned long long kv_off =
                        (static_cast<unsigned long long>(ts) * num_kv_heads + kv_head) * kHdim + lane * 8u;
                    const uint4 k_raw16 = *reinterpret_cast<const uint4*>(K_sink_bf16 + kv_off);
                    const uint4 v_raw16 = *reinterpret_cast<const uint4*>(V_sink_bf16 + kv_off);
                    const __nv_bfloat162* k2 = reinterpret_cast<const __nv_bfloat162*>(&k_raw16);
                    const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&v_raw16);
                    float dot = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 kf = __bfloat1622float2(k2[j]);
                        dot += qv[2 * j] * kf.x + qv[2 * j + 1] * kf.y;
                    }
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        dot += __shfl_xor_sync(0xffffffffu, dot, off);
                    }
                    const float s_log2 = dot * scale_log2;
                    const float m_new = fmaxf(m_s, s_log2);
                    const float a_old = (m_s > -1.0e29f) ? exp2f(m_s - m_new) : 0.0f;
                    // Match `c = 8.0f` (+3.0f in exp2f) convention of FP4 loop.
                    const float p_c = exp2f(s_log2 - m_new + 3.0f);
                    l_s = l_s * a_old + p_c;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 vf = __bfloat1622float2(v2[j]);
                        os[2 * j] = os[2 * j] * a_old + p_c * vf.x;
                        os[2 * j + 1] = os[2 * j + 1] * a_old + p_c * vf.y;
                    }
                    m_s = m_new;
                }
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sink_o_scratch[r * kHdim + lane * 8u + j] = os[j];
                }
                if (lane == 0u) {
                    sm.sRunningM[r] = m_s;
                    sm.sRunningL[r] = l_s;
                }
            }

            if (apply_wht) {
                wht256_warp_plain(qv, lane);
            }

            // Dynamic per-query normalization so both Q^(1) and residual Q^(2)
            // sit in the full-precision normal UE4M3 range (zero underflow).
            float q_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                q_amax = fmaxf(q_amax, fabsf(qv[j]));
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                q_amax = fmaxf(q_amax, __shfl_xor_sync(0xffffffffu, q_amax, off));
            }
            const bool q_nz = (q_amax > 1.0e-30f) && isfinite(q_amax);
            const float q_norm_scale = q_nz ? __fdiv_rn(256.0f, q_amax) : 0.0f;
            if (lane == 0u) {
                sm.sQ_logit_scale[r] = q_nz ? (inv_sqrt_d * kLog2E * __fdiv_rn(q_amax, 256.0f)) : 0.0f;
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                qv[j] *= q_norm_scale;
            }

            // Dual-Residual Query Quantization (BitDecoding-FP4 Eq. 3):
            //   lane pairs `(2*g, 2*g+1)` own 16-element micro-block `g`.
            float lmax1 = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                lmax1 = fmaxf(lmax1, fabsf(qv[j]));
            }
            const float gmax1 = fmaxf(lmax1, __shfl_xor_sync(0xffffffffu, lmax1, 1));
            const unsigned char s1_u8 = cvt_f32_to_ue4m3(gmax1 * (1.0f / 6.0f));
            const float s1_f = cvt_ue4m3_to_f32(s1_u8);
            const float inv_s1 = (s1_f > 0.0f) ? __fdiv_rn(1.0f, s1_f) : 0.0f;
            const unsigned int q1_word = quant8_e2m1(qv, inv_s1);

            // Residual `res = q - q^(1)`
            float res[8];
            float lmax2 = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned int nib = (q1_word >> (j * 4)) & 0xfu;
                res[j] = qv[j] - e2m1_nibble_to_f32(nib) * s1_f;
                lmax2 = fmaxf(lmax2, fabsf(res[j]));
            }
            const float gmax2 = fmaxf(lmax2, __shfl_xor_sync(0xffffffffu, lmax2, 1));
            const unsigned char s2_u8 = cvt_f32_to_ue4m3(gmax2 * (1.0f / 6.0f));
            const float s2_f = cvt_ue4m3_to_f32(s2_u8);
            const float inv_s2 = (s2_f > 0.0f) ? __fdiv_rn(1.0f, s2_f) : 0.0f;
            const unsigned int q2_word = quant8_e2m1(res, inv_s2);

            // Store stacked `(Q^(1), Q^(2))` in adjacent rows `2*r` and `2*r + 1`.
            reinterpret_cast<unsigned int*>(sm.sQ_data[2u * r])[lane] = q1_word;
            reinterpret_cast<unsigned int*>(sm.sQ_data[2u * r + 1u])[lane] = q2_word;
            if ((lane & 1u) == 0u) {
                sm.sQ_scale[2u * r][lane >> 1] = s1_u8;
                sm.sQ_scale[2u * r + 1u][lane >> 1] = s2_u8;
            }
        } else {
            reinterpret_cast<unsigned int*>(sm.sQ_data[2u * r])[lane] = 0u;
            reinterpret_cast<unsigned int*>(sm.sQ_data[2u * r + 1u])[lane] = 0u;
            if ((lane & 1u) == 0u) {
                sm.sQ_scale[2u * r][lane >> 1] = 0u;
                sm.sQ_scale[2u * r + 1u][lane >> 1] = 0u;
            }
        }
    }
    __syncthreads();

    // Load BF16 sink accumulators into MMA register layout before `sK_data` is
    // overwritten by the FP4 KV pipeline.
    if (run_sink_now) {
        #pragma unroll
        for (unsigned int qt = 0u; qt < kQTile8; ++qt) {
            const unsigned int c0 = qt * 8u + tid4 * 2u;
            const unsigned int c1 = c0 + 1u;
            #pragma unroll
            for (int dt = 0; dt < 4; ++dt) {
                const unsigned int d0 = warp_id * 64u + dt * 16u + group_id;
                const unsigned int d1 = d0 + 8u;
                o_acc[qt][dt][0] = (c0 < num_q_active) ? sink_o_scratch[c0 * kHdim + d0] : 0.0f;
                o_acc[qt][dt][1] = (c1 < num_q_active) ? sink_o_scratch[c1 * kHdim + d0] : 0.0f;
                o_acc[qt][dt][2] = (c0 < num_q_active) ? sink_o_scratch[c0 * kHdim + d1] : 0.0f;
                o_acc[qt][dt][3] = (c1 < num_q_active) ? sink_o_scratch[c1 * kHdim + d1] : 0.0f;
            }
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------------
    // Main Double-Buffered FP4 KV Tile Loop (`B_k = 64` = 4 blocks / tile)
    // ------------------------------------------------------------------------
    const unsigned int num_tiles =
        (blk_end > blk_start) ? ((blk_end - blk_start + kBlocksPerTile - 1u) / kBlocksPerTile) : 0u;

    auto issue_kv_tile = [&](unsigned int tile_idx, unsigned int stage) {
        const unsigned int tile_blk0 = blk_start + tile_idx * kBlocksPerTile;
        const unsigned int valid_blks = min(kBlocksPerTile, blk_end - tile_blk0);
        const int bid0 = (0u < valid_blks) ? my_block_table[tile_blk0 + 0u] : 0;
        const int bid1 = (1u < valid_blks) ? my_block_table[tile_blk0 + 1u] : 0;
        const int bid2 = (2u < valid_blks) ? my_block_table[tile_blk0 + 2u] : 0;
        const int bid3 = (3u < valid_blks) ? my_block_table[tile_blk0 + 3u] : 0;
        auto bid_of = [&](unsigned int b) -> int {
            return (b == 0u) ? bid0 : ((b == 1u) ? bid1 : ((b == 2u) ? bid2 : bid3));
        };

        // 1. K data: 64 keys x 128 B = 8,192 B = 512 x 16 B (4 per thread)
        #pragma unroll
        for (unsigned int it = 0u; it < 4u; ++it) {
            const unsigned int idx = it * kThreads + tid;  // 0..511
            const unsigned int tok_in_tile = idx >> 3;     // 0..63
            const unsigned int b = tok_in_tile >> 4;       // 0..3
            const unsigned int tok_in_blk = tok_in_tile & 15u;
            const unsigned int chunk16 = idx & 7u;         // 0..7
            const bool pred = (b < valid_blks);
            const unsigned long long phys = pred ? static_cast<unsigned long long>(bid_of(b)) : 0ull;
            const unsigned char* gptr =
                K_cache + phys * block_stride_bytes
                        + (static_cast<unsigned long long>(tok_in_blk) * num_kv_heads + kv_head) * kHdimHalf
                        + chunk16 * 16u;
            const unsigned int saddr =
                static_cast<unsigned int>(__cvta_generic_to_shared(&sm.sK_data[stage][tok_in_tile][chunk16 * 16u]));
            cp_async_cg_16_z(saddr, gptr, pred);
        }

        // 2. V^T data: 4 blocks x 256 ch x 8 B = 8,192 B (128 threads x 16 B per block)
        #pragma unroll
        for (unsigned int b = 0u; b < kBlocksPerTile; ++b) {
            const bool pred = (b < valid_blks);
            const unsigned long long phys = pred ? static_cast<unsigned long long>(bid_of(b)) : 0ull;
            const unsigned char* gptr =
                V_cache + phys * block_stride_bytes
                        + static_cast<unsigned long long>(kv_head) * (kHdim * 8u)
                        + tid * 16u;
            const unsigned int saddr =
                static_cast<unsigned int>(__cvta_generic_to_shared(&sm.sV_data[stage][b][0][0])) + tid * 16u;
            cp_async_cg_16_z(saddr, gptr, pred);
        }

        // 3. K scales (threads 0..63) + V^T scales (threads 64..127), 16 B each
        if (tid < 64u) {
            const unsigned int tok_in_tile = tid;
            const unsigned int b = tok_in_tile >> 4;
            const unsigned int tok_in_blk = tok_in_tile & 15u;
            const bool pred = (b < valid_blks);
            const unsigned long long phys = pred ? static_cast<unsigned long long>(bid_of(b)) : 0ull;
            const unsigned char* gptr =
                K_cache + phys * block_stride_bytes + data_section_bytes
                        + (static_cast<unsigned long long>(tok_in_blk) * num_kv_heads + kv_head) * kHdimGroups;
            const unsigned int saddr =
                static_cast<unsigned int>(__cvta_generic_to_shared(&sm.sK_scale[stage][tok_in_tile][0]));
            cp_async_cg_16_z(saddr, gptr, pred);
        } else {
            const unsigned int vidx = tid - 64u;     // 0..63
            const unsigned int b = vidx >> 4;        // 0..3
            const unsigned int chunk16 = vidx & 15u; // 0..15 (256 B / 16 B)
            const bool pred = (b < valid_blks);
            const unsigned long long phys = pred ? static_cast<unsigned long long>(bid_of(b)) : 0ull;
            const unsigned char* gptr =
                V_cache + phys * block_stride_bytes + data_section_bytes
                        + static_cast<unsigned long long>(kv_head) * kHdim
                        + chunk16 * 16u;
            const unsigned int saddr =
                static_cast<unsigned int>(__cvta_generic_to_shared(&sm.sV_scale[stage][b][chunk16 * 16u]));
            cp_async_cg_16_z(saddr, gptr, pred);
        }

        cp_async_commit();
    };

    if (num_tiles > 0u) {
        issue_kv_tile(0u, 0u);
    }

    for (unsigned int tile = 0u; tile < num_tiles; ++tile) {
        const unsigned int stage = tile & 1u;
        if (tile + 1u < num_tiles) {
            issue_kv_tile(tile + 1u, stage ^ 1u);
            cp_async_wait_prev();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();

        const unsigned int tile_tok0 = (blk_start + tile * kBlocksPerTile) * kBlockSize;
        const unsigned int warp_tok0 = tile_tok0 + warp_id * kBlockSize;
        const unsigned int key_pos0 = warp_tok0 + group_id;
        const unsigned int key_pos1 = key_pos0 + 8u;

        // --------------------------------------------------------------------
        // Stage 1: Native FP4 QK MMA (`S^T = K @ Q^T`) with Dual-Residual Q
        //   Warp `w` computes scores for the 16 keys of block `w` against all
        //   `kQSub` stacked queries (`kQHalves` tiles of 4 real queries each).
        // --------------------------------------------------------------------
        float s0[kQHalves];
        float s1[kQHalves];
        float m_blk[kQHalves];

        const unsigned int k_row0 = warp_id * kBlockSize + group_id;
        const unsigned int k_row1 = k_row0 + 8u;
        const unsigned int k_sfa_row = warp_id * kBlockSize + group_id + 8u * (lane & 1u);

        #pragma unroll
        for (unsigned int qh = 0u; qh < kQHalves; ++qh) {
            float acc_qk[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const unsigned int q_nc = qh * 8u + group_id;  // 8 stacked cols = 4 (Q1, Q2) pairs
            #pragma unroll
            for (unsigned int kt = 0u; kt < 4u; ++kt) {
                const unsigned int byte_base = kt * 32u + tid4 * 4u;
                const unsigned int sc_base = kt * 4u;
                const unsigned int a0 = *reinterpret_cast<const unsigned int*>(&sm.sK_data[stage][k_row0][byte_base]);
                const unsigned int a1 = *reinterpret_cast<const unsigned int*>(&sm.sK_data[stage][k_row1][byte_base]);
                const unsigned int a2 = *reinterpret_cast<const unsigned int*>(&sm.sK_data[stage][k_row0][byte_base + 16u]);
                const unsigned int a3 = *reinterpret_cast<const unsigned int*>(&sm.sK_data[stage][k_row1][byte_base + 16u]);
                const unsigned int sfa = *reinterpret_cast<const unsigned int*>(&sm.sK_scale[stage][k_sfa_row][sc_base]);

                const unsigned int b0 = *reinterpret_cast<const unsigned int*>(&sm.sQ_data[q_nc][byte_base]);
                const unsigned int b1 = *reinterpret_cast<const unsigned int*>(&sm.sQ_data[q_nc][byte_base + 16u]);
                const unsigned int sfb = *reinterpret_cast<const unsigned int*>(&sm.sQ_scale[q_nc][sc_base]);

                mma_nvfp4_m16n8k64<kRazerKv>(acc_qk, a0, a1, a2, a3, b0, b1, sfa, sfb, lane);
            }

            const unsigned int r = qh * 4u + tid4;
            const unsigned int q_sl = sm.sQSeqLen[r];
            const float lscale = sm.sQ_logit_scale[r];
            // Recombine Dual-Residual `K·Q^(1) + K·Q^(2)` directly in registers (0 shuffles).
            const float v0 = (key_pos0 < q_sl) ? ((acc_qk[0] + acc_qk[1]) * lscale) : kNegInf;
            const float v1 = (key_pos1 < q_sl) ? ((acc_qk[2] + acc_qk[3]) * lscale) : kNegInf;
            s0[qh] = v0;
            s1[qh] = v1;

            // Transposed Softmax butterfly max across all 16 keys of block `warp_id`
            // (lanes with fixed `tid4` span `group_id = 0..7` -> XOR 4, 8, 16).
            float mb = fmaxf(v0, v1);
            mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 4));
            mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 8));
            mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 16));
            m_blk[qh] = mb;
            if (group_id == 0u) {
                sm.sBlkMax[warp_id][r] = mb;
            }
        }
        __syncthreads();

        // Cross-warp 4-block max merge -> unified `m_new` and `alpha` per query.
        if (tid < kQSub) {
            const float mt = fmaxf(
                fmaxf(sm.sBlkMax[0][tid], sm.sBlkMax[1][tid]),
                fmaxf(sm.sBlkMax[2][tid], sm.sBlkMax[3][tid]));
            const float m_old = sm.sRunningM[tid];
            const float m_new = fmaxf(m_old, mt);
            const float alpha = (m_old > -1.0e29f && m_new > -1.0e29f) ? exp2f(m_old - m_new) : 0.0f;
            sm.sRunningM[tid] = m_new;
            sm.sAlpha[tid] = alpha;
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // Stage 2: Underflow-Aware Probability Scaling + Fused E2M1 Quantization
        //   Computes `p_c = exp2f(s - m_new + 3.0f)` (`c = 8` folded for free),
        //   switches `kappa = 2` iff `p_max_c <= 21/1024`, else `6`, and packs
        //   `P^T` directly into `sP_data[r][warp_id*8 .. +7]` + `sP_scale[r][warp_id]`.
        // --------------------------------------------------------------------
        #pragma unroll
        for (unsigned int qh = 0u; qh < kQHalves; ++qh) {
            const unsigned int r = qh * 4u + tid4;
            const unsigned int q_sl = sm.sQSeqLen[r];
            const float m_new = sm.sRunningM[r];
            const float mb = m_blk[qh];

            const float p0_c = (key_pos0 < q_sl && m_new > -1.0e29f) ? exp2f(s0[qh] - m_new + 3.0f) : 0.0f;
            const float p1_c = (key_pos1 < q_sl && m_new > -1.0e29f) ? exp2f(s1[qh] - m_new + 3.0f) : 0.0f;

            // Exact block sum via 3 transposed butterfly shuffles (keeps `c = 8` factor,
            // which cancels identically in `O_acc / l_acc`).
            float l_b = p0_c + p1_c;
            l_b += __shfl_xor_sync(0xffffffffu, l_b, 4);
            l_b += __shfl_xor_sync(0xffffffffu, l_b, 8);
            l_b += __shfl_xor_sync(0xffffffffu, l_b, 16);
            if (group_id == 0u) {
                sm.sBlkSum[warp_id][r] = l_b;
            }

            // Underflow-aware E4M3 probability block scale (Eq. 4 of 2611.09825).
            const float p_max_c = (mb > -1.0e29f && m_new > -1.0e29f) ? exp2f(mb - m_new + 3.0f) : 0.0f;
            const float kappa = (p_max_c <= kUnderflowTauC) ? 2.0f : 6.0f;
            const unsigned char sp_u8 = cvt_f32_to_ue4m3(__fdiv_rn(p_max_c, kappa));
            const float sp_f = cvt_ue4m3_to_f32(sp_u8);
            const float inv_sp = (sp_f > 0.0f) ? __fdiv_rn(1.0f, sp_f) : 0.0f;

            // Gather odd-key neighbor (`group_id ^ 1` -> `lane ^ 4`) so even `group_id`
            // lanes emit exact 2-nibble E2M1 bytes via hardware `cvt.rn.satfinite.e2m1x2.f32`.
            const float p0_odd = __shfl_xor_sync(0xffffffffu, p0_c, 4);
            const float p1_odd = __shfl_xor_sync(0xffffffffu, p1_c, 4);
            if ((group_id & 1u) == 0u) {
                const unsigned int pair = group_id >> 1;  // 0..3
                sm.sP_data[r][warp_id * 8u + pair] = quant2_e2m1(p0_c * inv_sp, p0_odd * inv_sp);
                sm.sP_data[r][warp_id * 8u + 4u + pair] = quant2_e2m1(p1_c * inv_sp, p1_odd * inv_sp);
            }
            if (group_id == 0u) {
                sm.sP_scale[r][warp_id] = sp_u8;
            }
        }
        __syncthreads();

        // Update running softmax denominator `l_new = l_old * alpha + sum_w l_b`.
        if (tid < kQSub) {
            const float l_tile =
                sm.sBlkSum[0][tid] + sm.sBlkSum[1][tid] + sm.sBlkSum[2][tid] + sm.sBlkSum[3][tid];
            sm.sRunningL[tid] = sm.sRunningL[tid] * sm.sAlpha[tid] + l_tile;
        }

        // --------------------------------------------------------------------
        // Stage 3: Native FP4 PV MMA (`O^T = V^T @ P^T`)
        //   Warp `w` computes 64 output channels (`4` x `m16n8k64` tiles) across
        //   all `kQTile8` 8-query blocks using pre-transposed `sV_data[stage]`.
        // --------------------------------------------------------------------
        const unsigned int blk_lo = tid4 >> 1;    // 0 or 1
        const unsigned int half_w = tid4 & 1u;    // 0 (bytes 0..3) or 1 (bytes 4..7)
        const unsigned int blk_hi = 2u + blk_lo;  // 2 or 3

        #pragma unroll
        for (unsigned int qt = 0u; qt < kQTile8; ++qt) {
            const unsigned int q_base = qt * 8u;
            const float alpha0 = sm.sAlpha[q_base + tid4 * 2u];
            const float alpha1 = sm.sAlpha[q_base + tid4 * 2u + 1u];

            const unsigned int p_nc = q_base + group_id;
            const unsigned int b0 = *reinterpret_cast<const unsigned int*>(&sm.sP_data[p_nc][tid4 * 4u]);
            const unsigned int b1 = *reinterpret_cast<const unsigned int*>(&sm.sP_data[p_nc][16u + tid4 * 4u]);
            const unsigned int sfb = *reinterpret_cast<const unsigned int*>(&sm.sP_scale[p_nc][0]);

            #pragma unroll
            for (int dt = 0; dt < 4; ++dt) {
                // Rescale running output accumulator for new tile max.
                o_acc[qt][dt][0] *= alpha0;
                o_acc[qt][dt][1] *= alpha1;
                o_acc[qt][dt][2] *= alpha0;
                o_acc[qt][dt][3] *= alpha1;

                const unsigned int d_base = warp_id * 64u + static_cast<unsigned int>(dt) * 16u;
                const unsigned int fr0 = d_base + group_id;
                const unsigned int fr1 = fr0 + 8u;
                const unsigned int m_sfa = d_base + group_id + 8u * (lane & 1u);

                const unsigned int a0 = *reinterpret_cast<const unsigned int*>(&sm.sV_data[stage][blk_lo][fr0][half_w * 4u]);
                const unsigned int a1 = *reinterpret_cast<const unsigned int*>(&sm.sV_data[stage][blk_lo][fr1][half_w * 4u]);
                const unsigned int a2 = *reinterpret_cast<const unsigned int*>(&sm.sV_data[stage][blk_hi][fr0][half_w * 4u]);
                const unsigned int a3 = *reinterpret_cast<const unsigned int*>(&sm.sV_data[stage][blk_hi][fr1][half_w * 4u]);
                const unsigned int sfa_v =
                    (static_cast<unsigned int>(sm.sV_scale[stage][0][m_sfa])) |
                    (static_cast<unsigned int>(sm.sV_scale[stage][1][m_sfa]) << 8) |
                    (static_cast<unsigned int>(sm.sV_scale[stage][2][m_sfa]) << 16) |
                    (static_cast<unsigned int>(sm.sV_scale[stage][3][m_sfa]) << 24);

                mma_nvfp4_m16n8k64<kRazerKv>(o_acc[qt][dt], a0, a1, a2, a3, b0, b1, sfa_v, sfb, lane);
            }
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------------
    // Epilogue: Direct BF16 O store (`num_splits == 1`) or F32 split-K partial
    // ------------------------------------------------------------------------
    if (num_splits == 1u) {
        #pragma unroll
        for (unsigned int qt = 0u; qt < kQTile8; ++qt) {
            const unsigned int c0 = qt * 8u + tid4 * 2u;
            const unsigned int c1 = c0 + 1u;
            const float l0 = (c0 < num_q_active) ? sm.sRunningL[c0] : 0.0f;
            const float l1 = (c1 < num_q_active) ? sm.sRunningL[c1] : 0.0f;
            const float inv_l0 = (l0 > 0.0f) ? __fdiv_rn(1.0f, l0) : 0.0f;
            const float inv_l1 = (l1 > 0.0f) ? __fdiv_rn(1.0f, l1) : 0.0f;

            if (c0 < num_q_active) {
                unsigned int q_row0 = seq_idx;
                unsigned int q_hl0 = q_chunk_base + c0;
                if constexpr (kIsKgamma) {
                    q_row0 = q_hl0 / gqa_ratio;
                    q_hl0 = q_hl0 - q_row0 * gqa_ratio;
                }
                const unsigned int q_head0 = kv_head * gqa_ratio + q_hl0;
                __nv_bfloat16* o_ptr0 =
                    O + (static_cast<unsigned long long>(q_row0) * num_q_heads + q_head0) * kHdim;
                #pragma unroll
                for (int dt = 0; dt < 4; ++dt) {
                    const unsigned int d0 = warp_id * 64u + static_cast<unsigned int>(dt) * 16u + group_id;
                    o_ptr0[d0] = __float2bfloat16(o_acc[qt][dt][0] * inv_l0);
                    o_ptr0[d0 + 8u] = __float2bfloat16(o_acc[qt][dt][2] * inv_l0);
                }
            }
            if (c1 < num_q_active) {
                unsigned int q_row1 = seq_idx;
                unsigned int q_hl1 = q_chunk_base + c1;
                if constexpr (kIsKgamma) {
                    q_row1 = q_hl1 / gqa_ratio;
                    q_hl1 = q_hl1 - q_row1 * gqa_ratio;
                }
                const unsigned int q_head1 = kv_head * gqa_ratio + q_hl1;
                __nv_bfloat16* o_ptr1 =
                    O + (static_cast<unsigned long long>(q_row1) * num_q_heads + q_head1) * kHdim;
                #pragma unroll
                for (int dt = 0; dt < 4; ++dt) {
                    const unsigned int d0 = warp_id * 64u + static_cast<unsigned int>(dt) * 16u + group_id;
                    o_ptr1[d0] = __float2bfloat16(o_acc[qt][dt][1] * inv_l1);
                    o_ptr1[d0 + 8u] = __float2bfloat16(o_acc[qt][dt][3] * inv_l1);
                }
            }
        }
    } else {
        const unsigned int ws_stride = kHdim + 2u;
        #pragma unroll
        for (unsigned int qt = 0u; qt < kQTile8; ++qt) {
            const unsigned int c0 = qt * 8u + tid4 * 2u;
            const unsigned int c1 = c0 + 1u;
            if (c0 < num_q_active) {
                unsigned int q_row0 = seq_idx;
                unsigned int q_hl0 = q_chunk_base + c0;
                if constexpr (kIsKgamma) {
                    q_row0 = q_hl0 / gqa_ratio;
                    q_hl0 = q_hl0 - q_row0 * gqa_ratio;
                }
                const unsigned int q_head0 = kv_head * gqa_ratio + q_hl0;
                float* ws0 = workspace
                    + ((static_cast<unsigned long long>(q_row0) * num_q_heads + q_head0) * num_splits + split_id)
                      * ws_stride;
                #pragma unroll
                for (int dt = 0; dt < 4; ++dt) {
                    const unsigned int d0 = warp_id * 64u + static_cast<unsigned int>(dt) * 16u + group_id;
                    ws0[d0] = o_acc[qt][dt][0];
                    ws0[d0 + 8u] = o_acc[qt][dt][2];
                }
            }
            if (c1 < num_q_active) {
                unsigned int q_row1 = seq_idx;
                unsigned int q_hl1 = q_chunk_base + c1;
                if constexpr (kIsKgamma) {
                    q_row1 = q_hl1 / gqa_ratio;
                    q_hl1 = q_hl1 - q_row1 * gqa_ratio;
                }
                const unsigned int q_head1 = kv_head * gqa_ratio + q_hl1;
                float* ws1 = workspace
                    + ((static_cast<unsigned long long>(q_row1) * num_q_heads + q_head1) * num_splits + split_id)
                      * ws_stride;
                #pragma unroll
                for (int dt = 0; dt < 4; ++dt) {
                    const unsigned int d0 = warp_id * 64u + static_cast<unsigned int>(dt) * 16u + group_id;
                    ws1[d0] = o_acc[qt][dt][1];
                    ws1[d0 + 8u] = o_acc[qt][dt][3];
                }
            }
        }
        if (tid < num_q_active) {
            unsigned int q_row = seq_idx;
            unsigned int q_hl = q_chunk_base + tid;
            if constexpr (kIsKgamma) {
                q_row = q_hl / gqa_ratio;
                q_hl = q_hl - q_row * gqa_ratio;
            }
            const unsigned int q_head = kv_head * gqa_ratio + q_hl;
            float* ws = workspace
                + ((static_cast<unsigned long long>(q_row) * num_q_heads + q_head) * num_splits + split_id)
                  * ws_stride;
            ws[kHdim] = sm.sRunningM[tid];
            ws[kHdim + 1u] = sm.sRunningL[tid];
        }
    }
}

}  // namespace bitfp4

// ============================================================================
// Exported Decode Attention Entrypoints
// ============================================================================

extern "C" __global__ __launch_bounds__(128, 2) void paged_decode_attn_bitfp4_sm120(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ workspace,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int flags,
    const __nv_bfloat16* __restrict__ K_sink_bf16,
    const __nv_bfloat16* __restrict__ V_sink_bf16,
    const unsigned int num_sink_tokens
) {
    bitfp4::paged_decode_bitfp4_core<false, 8u, false>(
        Q, K_cache, V_cache, O, workspace, block_tables, seq_lens,
        max_blocks_per_seq, num_q_heads, num_kv_heads, inv_sqrt_d,
        num_splits, q_stride, block_stride_bytes, data_section_bytes,
        1u, flags, K_sink_bf16, V_sink_bf16, num_sink_tokens);
}

extern "C" __global__ __launch_bounds__(128, 2) void paged_decode_attn_bitfp4_razer_sm120(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ workspace,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int flags,
    const __nv_bfloat16* __restrict__ K_sink_bf16,
    const __nv_bfloat16* __restrict__ V_sink_bf16,
    const unsigned int num_sink_tokens
) {
    bitfp4::paged_decode_bitfp4_core<true, 8u, false>(
        Q, K_cache, V_cache, O, workspace, block_tables, seq_lens,
        max_blocks_per_seq, num_q_heads, num_kv_heads, inv_sqrt_d,
        num_splits, q_stride, block_stride_bytes, data_section_bytes,
        1u, flags, K_sink_bf16, V_sink_bf16, num_sink_tokens);
}

extern "C" __global__ __launch_bounds__(128, 2) void paged_decode_attn_bitfp4_kgamma_sm120(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ workspace,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int num_qtile,
    const unsigned int flags,
    const __nv_bfloat16* __restrict__ K_sink_bf16,
    const __nv_bfloat16* __restrict__ V_sink_bf16,
    const unsigned int num_sink_tokens
) {
    bitfp4::paged_decode_bitfp4_core<false, 16u, true>(
        Q, K_cache, V_cache, O, workspace, block_tables, seq_lens,
        max_blocks_per_seq, num_q_heads, num_kv_heads, inv_sqrt_d,
        num_splits, q_stride, block_stride_bytes, data_section_bytes,
        num_qtile, flags, K_sink_bf16, V_sink_bf16, num_sink_tokens);
}

extern "C" __global__ __launch_bounds__(128, 2) void paged_decode_attn_bitfp4_kgamma_razer_sm120(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ workspace,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int num_qtile,
    const unsigned int flags,
    const __nv_bfloat16* __restrict__ K_sink_bf16,
    const __nv_bfloat16* __restrict__ V_sink_bf16,
    const unsigned int num_sink_tokens
) {
    bitfp4::paged_decode_bitfp4_core<true, 16u, true>(
        Q, K_cache, V_cache, O, workspace, block_tables, seq_lens,
        max_blocks_per_seq, num_q_heads, num_kv_heads, inv_sqrt_d,
        num_splits, q_stride, block_stride_bytes, data_section_bytes,
        num_qtile, flags, K_sink_bf16, V_sink_bf16, num_sink_tokens);
}

// ============================================================================
// Base-2 Split-K Reduce Kernel (`exp2f`, matches BitDecoding-FP4 log2 softmax)
// Grid: (num_q_heads, num_rows, 1)   Block: (32, 1, 1)
// ============================================================================

extern "C" __global__ void paged_decode_attn_bitfp4_reduce(
    const float* __restrict__ workspace,  // [num_rows, num_q_heads, num_splits, HDIM+2]
    __nv_bfloat16* __restrict__ O,        // [num_rows, num_q_heads, HDIM]
    const int* __restrict__ seq_lens,     // [num_rows]
    const unsigned int num_q_heads,
    const unsigned int num_splits
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int row_idx = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    if (q_head >= num_q_heads) return;
    if (seq_lens[row_idx] <= 0) return;

    const unsigned int ws_stride = bitfp4::kHdim + 2u;
    const float* ws_base =
        workspace + (static_cast<unsigned long long>(row_idx) * num_q_heads + q_head) * num_splits * ws_stride;
    const unsigned int vec_off = lane * 8u;

    float m = bitfp4::kNegInf;
    float l = 0.0f;
    float o_reg[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned int s = 0u; s < num_splits; ++s) {
        const float* ws = ws_base + static_cast<unsigned long long>(s) * ws_stride;
        const float ms = ws[bitfp4::kHdim];
        const float ls = ws[bitfp4::kHdim + 1u];
        if (ls <= 0.0f) continue;

        const float m_new = fmaxf(m, ms);
        const float scale_me = (l > 0.0f) ? exp2f(m - m_new) : 0.0f;
        const float scale_s = exp2f(ms - m_new);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            o_reg[i] = o_reg[i] * scale_me + ws[vec_off + i] * scale_s;
        }
        l = l * scale_me + ls * scale_s;
        m = m_new;
    }

    const float inv_l = (l > 0.0f) ? __fdiv_rn(1.0f, l) : 0.0f;
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(
        O + (static_cast<unsigned long long>(row_idx) * num_q_heads + q_head) * bitfp4::kHdim + vec_off);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        o2[i] = __floats2bfloat162_rn(o_reg[2 * i] * inv_l, o_reg[2 * i + 1] * inv_l);
    }
}

// ============================================================================
// Companion Block-16 K Write Kernel (`K' = H(K - mu_k)` + 4/6 + optional RaZeR)
// Grid: (num_tokens, num_kv_heads, 1)   Block: (32, 1, 1) — 1 warp per (tok, kv_head)
// ============================================================================

extern "C" __global__ void reshape_and_cache_bitfp4_k_block16(
    const __nv_bfloat16* __restrict__ key,         // [num_tokens, key_stride] BF16
    const __nv_bfloat16* __restrict__ mu_k,        // Optional [num_kv_heads, HDIM] BF16 static mean
    unsigned char* __restrict__ k_cache,           // [num_blocks, block_stride_bytes]
    const int64_t* __restrict__ slot_mapping,      // [num_tokens]
    const unsigned int num_kv_heads,
    const unsigned int key_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int flags                       // BITFP4_FLAG_*
) {
    const unsigned int tok_idx = blockIdx.x;
    const unsigned int kv_head = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    if (kv_head >= num_kv_heads) return;

    const int64_t slot = slot_mapping[tok_idx];
    if (slot < 0) return;
    const unsigned long long block_idx = static_cast<unsigned long long>(slot) >> 4;
    const unsigned int tok_in_blk = static_cast<unsigned int>(slot) & 15u;

    const __nv_bfloat16* k_src =
        key + static_cast<unsigned long long>(tok_idx) * key_stride
            + static_cast<unsigned long long>(kv_head) * bitfp4::kHdim
            + lane * 8u;
    const uint4 raw16 = *reinterpret_cast<const uint4*>(k_src);
    const __nv_bfloat162* k2 = reinterpret_cast<const __nv_bfloat162*>(&raw16);

    float kv[8];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(k2[j]);
        kv[2 * j] = f.x;
        kv[2 * j + 1] = f.y;
    }

    if (mu_k != nullptr) {
        const uint4 mu16 = *reinterpret_cast<const uint4*>(
            mu_k + static_cast<unsigned long long>(kv_head) * bitfp4::kHdim + lane * 8u);
        const __nv_bfloat162* m2 = reinterpret_cast<const __nv_bfloat162*>(&mu16);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 mf = __bfloat1622float2(m2[j]);
            kv[2 * j] -= mf.x;
            kv[2 * j + 1] -= mf.y;
        }
    }

    if ((flags & bitfp4::kFlagApplyWht) != 0u) {
        bitfp4::wht256_warp_plain(kv, lane);
    }

    // Group max across the 16-element micro-block held by lane pair `(2*g, 2*g+1)`.
    float lmax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        lmax = fmaxf(lmax, fabsf(kv[j]));
    }
    const float gmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 1));

    unsigned int best_word = 0u;
    unsigned char best_sc = 0u;

    if (gmax > 0.0f) {
        const bool use_razer = (flags & bitfp4::kFlagRazerKv) != 0u;
        const bool use_4o6 = (flags & bitfp4::kFlagFourOverSix) != 0u;

        // Candidate 0: standard max-6 scale
        const unsigned char s6_u8 = bitfp4::cvt_f32_to_ue4m3(gmax * (1.0f / 6.0f));
        const float s6_f = bitfp4::cvt_ue4m3_to_f32(s6_u8);
        const float inv_s6 = (s6_f > 0.0f) ? __fdiv_rn(1.0f, s6_f) : 0.0f;
        unsigned int w6 = 0u;
        float e6 = bitfp4::half_group_mse_e2m1(kv, s6_f, inv_s6, w6);
        e6 += __shfl_xor_sync(0xffffffffu, e6, 1);

        best_word = w6;
        best_sc = s6_u8;
        float best_err = e6;

        // Candidate 1: Four-Over-Six max-4 scale (Cook et al. 2025 / BitDecoding-FP4 §3.3)
        if (use_4o6) {
            const unsigned char s4_u8 = bitfp4::cvt_f32_to_ue4m3(gmax * 0.25f);
            const float s4_f = bitfp4::cvt_ue4m3_to_f32(s4_u8);
            const float inv_s4 = (s4_f > 0.0f) ? __fdiv_rn(1.0f, s4_f) : 0.0f;
            unsigned int w4 = 0u;
            float e4 = bitfp4::half_group_mse_e2m1(kv, s4_f, inv_s4, w4);
            e4 += __shfl_xor_sync(0xffffffffu, e4, 1);
            if (e4 < best_err) {
                best_err = e4;
                best_word = w4;
                best_sc = s4_u8;
            }
        }

        // Candidates 2-3: RaZeR-KV ±5.0 remapping (neg_sv stored in UE4M3 bit 7)
        if (use_razer) {
            // Test both +5.0 (neg_sv=0) and -5.0 (neg_sv=1) with max-6 scale (fills the 4..6 gap)
            // and max-5 scale (exact peak landing on ±5.0).
            #pragma unroll
            for (unsigned int neg_sv = 0u; neg_sv < 2u; ++neg_sv) {
                unsigned int wr6 = 0u;
                float er6 = bitfp4::half_group_mse_razer(kv, s6_f, inv_s6, neg_sv, wr6);
                er6 += __shfl_xor_sync(0xffffffffu, er6, 1);
                if (er6 < best_err) {
                    best_err = er6;
                    best_word = wr6;
                    best_sc = static_cast<unsigned char>(s6_u8 | (neg_sv << 7));
                }

                const unsigned char s5_u8 = bitfp4::cvt_f32_to_ue4m3(gmax * 0.2f);
                const float s5_f = bitfp4::cvt_ue4m3_to_f32(s5_u8);
                const float inv_s5 = (s5_f > 0.0f) ? __fdiv_rn(1.0f, s5_f) : 0.0f;
                unsigned int wr5 = 0u;
                float er5 = bitfp4::half_group_mse_razer(kv, s5_f, inv_s5, neg_sv, wr5);
                er5 += __shfl_xor_sync(0xffffffffu, er5, 1);
                if (er5 < best_err) {
                    best_err = er5;
                    best_word = wr5;
                    best_sc = static_cast<unsigned char>(s5_u8 | (neg_sv << 7));
                }
            }
        }
    }

    unsigned char* blk_base = k_cache + block_idx * block_stride_bytes;
    unsigned int* dst_data = reinterpret_cast<unsigned int*>(
        blk_base + (static_cast<unsigned long long>(tok_in_blk) * num_kv_heads + kv_head) * bitfp4::kHdimHalf);
    dst_data[lane] = best_word;

    if ((lane & 1u) == 0u) {
        unsigned char* dst_sc =
            blk_base + data_section_bytes
            + (static_cast<unsigned long long>(tok_in_blk) * num_kv_heads + kv_head) * bitfp4::kHdimGroups;
        dst_sc[lane >> 1] = best_sc;
    }
}

// ============================================================================
// Companion Block-16 Transposed V^T Write / Promote Kernel
//   Stores self-contained channel-wise `V^T` per physical block:
//     data  [num_kv_heads, HDIM, 8]  (2,048 * nkv B = exact nvfp4_data_bytes)
//     scale [num_kv_heads, HDIM, 1]  (  256 * nkv B = exact nvfp4_scale_bytes)
//
// Race-free for ANY `num_tokens` (single-token decode, K=gamma verify, or full
// prefill): only the first token `t` of each contiguous same-block run executes
// and commits all tokens targeting that physical block in one atomic pass.
// Uses static SMEM `sV[16][256]` (16 KiB) -> 0 B `.local` stack frame & spills!
//
// Grid: (num_tokens, num_kv_heads, 1)   Block: (256, 1, 1)
// ============================================================================

extern "C" __global__ void promote_block16_bitfp4_v_transposed(
    const __nv_bfloat16* __restrict__ value,       // [num_tokens, value_stride] BF16
    unsigned char* __restrict__ v_cache,           // [num_blocks, block_stride_bytes]
    const int64_t* __restrict__ slot_mapping,      // [num_tokens]
    const unsigned int num_tokens,
    const unsigned int num_kv_heads,
    const unsigned int value_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int flags                       // BITFP4_FLAG_*
) {
    const unsigned int t0 = blockIdx.x;
    const unsigned int kv_head = blockIdx.y;
    const unsigned int d = threadIdx.x;  // 0..255
    if (t0 >= num_tokens || kv_head >= num_kv_heads) return;

    const int64_t slot0 = slot_mapping[t0];
    if (slot0 < 0) return;
    const int64_t blk_id = slot0 >> 4;

    // Only the first token of a contiguous same-block run handles this block.
    if (t0 > 0u) {
        const int64_t prev_slot = slot_mapping[t0 - 1u];
        if (prev_slot >= 0 && (prev_slot >> 4) == blk_id) {
            return;
        }
    }

    // Find end of contiguous run targeting `blk_id` (at most 16 tokens).
    unsigned int t_end = t0 + 1u;
    while (t_end < num_tokens) {
        const int64_t s = slot_mapping[t_end];
        if (s < 0 || (s >> 4) != blk_id) break;
        ++t_end;
    }

    // Bank-conflict-free SMEM staging (`[16][256]`, 16 KiB) -> zero local spills.
    __shared__ float sV[bitfp4::kBlockSize][bitfp4::kHdim];

    unsigned char* blk_base = v_cache + static_cast<unsigned long long>(blk_id) * block_stride_bytes;
    uint2* v_data_ptr = reinterpret_cast<uint2*>(
        blk_base + static_cast<unsigned long long>(kv_head) * (bitfp4::kHdim * 8u) + d * 8u);
    unsigned char* v_sc_ptr =
        blk_base + data_section_bytes + static_cast<unsigned long long>(kv_head) * bitfp4::kHdim + d;

    const unsigned int first_off = static_cast<unsigned int>(slot0) & 15u;
    const bool use_razer = (flags & bitfp4::kFlagRazerKv) != 0u;

    // Initialize the 16-slot channel vector: zero for fresh block (`first_off == 0`),
    // or decode existing slots `[0..first_off)` for incremental append into a partial block.
    if (first_off == 0u) {
        #pragma unroll
        for (unsigned int k = 0u; k < bitfp4::kBlockSize; ++k) {
            sV[k][d] = 0.0f;
        }
    } else {
        const uint2 old_words = *v_data_ptr;
        const unsigned char old_sc_raw = *v_sc_ptr;
        const float old_sc = bitfp4::cvt_ue4m3_to_f32(old_sc_raw);
        const unsigned int old_neg = (old_sc_raw >> 7) & 1u;
        #pragma unroll
        for (unsigned int k = 0u; k < 8u; ++k) {
            const unsigned int nib = (old_words.x >> (k * 4u)) & 0xfu;
            const float val = use_razer ? bitfp4::razer_nibble_to_f32(nib, old_neg)
                                        : bitfp4::e2m1_nibble_to_f32(nib);
            sV[k][d] = (k < first_off) ? (val * old_sc) : 0.0f;
        }
        #pragma unroll
        for (unsigned int k = 0u; k < 8u; ++k) {
            const unsigned int nib = (old_words.y >> (k * 4u)) & 0xfu;
            const float val = use_razer ? bitfp4::razer_nibble_to_f32(nib, old_neg)
                                        : bitfp4::e2m1_nibble_to_f32(nib);
            sV[8u + k][d] = ((8u + k) < first_off) ? (val * old_sc) : 0.0f;
        }
    }

    // Insert all new BF16 token values from this run into their exact slot positions.
    for (unsigned int u = t0; u < t_end; ++u) {
        const unsigned int off = static_cast<unsigned int>(slot_mapping[u]) & 15u;
        const float val = __bfloat162float(
            value[static_cast<unsigned long long>(u) * value_stride
                  + static_cast<unsigned long long>(kv_head) * bitfp4::kHdim + d]);
        sV[off][d] = val;
    }

    // Load the 16-element channel vector into two 8-element register arrays (zero spills).
    float lo[8], hi[8];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        lo[i] = sV[i][d];
        hi[i] = sV[8 + i][d];
        amax = fmaxf(amax, fmaxf(fabsf(lo[i]), fabsf(hi[i])));
    }

    unsigned int best_w0 = 0u;
    unsigned int best_w1 = 0u;
    unsigned char best_sc = 0u;

    if (amax > 0.0f) {
        const bool use_4o6 = (flags & bitfp4::kFlagFourOverSix) != 0u;

        // Candidate 0: standard max-6 scale
        const unsigned char s6_u8 = bitfp4::cvt_f32_to_ue4m3(amax * (1.0f / 6.0f));
        const float s6_f = bitfp4::cvt_ue4m3_to_f32(s6_u8);
        const float inv_s6 = (s6_f > 0.0f) ? __fdiv_rn(1.0f, s6_f) : 0.0f;
        unsigned int w6_0 = 0u, w6_1 = 0u;
        float best_err = bitfp4::half_group_mse_e2m1(lo, s6_f, inv_s6, w6_0)
                       + bitfp4::half_group_mse_e2m1(hi, s6_f, inv_s6, w6_1);
        best_w0 = w6_0;
        best_w1 = w6_1;
        best_sc = s6_u8;

        // Candidate 1: Four-Over-Six max-4 scale
        if (use_4o6) {
            const unsigned char s4_u8 = bitfp4::cvt_f32_to_ue4m3(amax * 0.25f);
            const float s4_f = bitfp4::cvt_ue4m3_to_f32(s4_u8);
            const float inv_s4 = (s4_f > 0.0f) ? __fdiv_rn(1.0f, s4_f) : 0.0f;
            unsigned int w4_0 = 0u, w4_1 = 0u;
            const float e4 = bitfp4::half_group_mse_e2m1(lo, s4_f, inv_s4, w4_0)
                           + bitfp4::half_group_mse_e2m1(hi, s4_f, inv_s4, w4_1);
            if (e4 < best_err) {
                best_err = e4;
                best_w0 = w4_0;
                best_w1 = w4_1;
                best_sc = s4_u8;
            }
        }

        // Candidates 2-3: RaZeR-KV ±5.0 remapping
        if (use_razer) {
            #pragma unroll
            for (unsigned int neg_sv = 0u; neg_sv < 2u; ++neg_sv) {
                unsigned int wr6_0 = 0u, wr6_1 = 0u;
                const float er6 = bitfp4::half_group_mse_razer(lo, s6_f, inv_s6, neg_sv, wr6_0)
                                + bitfp4::half_group_mse_razer(hi, s6_f, inv_s6, neg_sv, wr6_1);
                if (er6 < best_err) {
                    best_err = er6;
                    best_w0 = wr6_0;
                    best_w1 = wr6_1;
                    best_sc = static_cast<unsigned char>(s6_u8 | (neg_sv << 7));
                }

                const unsigned char s5_u8 = bitfp4::cvt_f32_to_ue4m3(amax * 0.2f);
                const float s5_f = bitfp4::cvt_ue4m3_to_f32(s5_u8);
                const float inv_s5 = (s5_f > 0.0f) ? __fdiv_rn(1.0f, s5_f) : 0.0f;
                unsigned int wr5_0 = 0u, wr5_1 = 0u;
                const float er5 = bitfp4::half_group_mse_razer(lo, s5_f, inv_s5, neg_sv, wr5_0)
                                + bitfp4::half_group_mse_razer(hi, s5_f, inv_s5, neg_sv, wr5_1);
                if (er5 < best_err) {
                    best_err = er5;
                    best_w0 = wr5_0;
                    best_w1 = wr5_1;
                    best_sc = static_cast<unsigned char>(s5_u8 | (neg_sv << 7));
                }
            }
        }
    }

    *v_data_ptr = make_uint2(best_w0, best_w1);
    *v_sc_ptr = best_sc;
}
