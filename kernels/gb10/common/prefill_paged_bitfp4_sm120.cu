// SPDX-License-Identifier: AGPL-3.0-only
//
// Native Blackwell (SM120/SM121) BitDecoding-FP4 + 6-op SWAR RaZeR-KV Paged
// Chunked-Continuation Prefill Attention Kernels (HDIM=256, BLOCK_SIZE=16).
//
// Design overview:
//   1. Directly reads the self-contained block-16 BitDecoding-FP4 KV cache
//      written by `reshape_and_cache_bitfp4_k_block16` and
//      `promote_block16_bitfp4_v_transposed` (zero intermediate BF16 dequant,
//      zero pool layout change, full 64K-512K chunked continuation support):
//      - K block (token-wise): data [16, nkv, 128] E2M1 + scale [16, nkv, 16] UE4M3
//        (post-WHT + Four-Over-Six + optional RaZeR-KV bit-7 `neg_sv`).
//      - V block (block-16 channel-transposed `V^T`):
//        data [nkv, 256, 8] E2M1 + scale [nkv, 256, 1] UE4M3.
//   2. Per-Q-head `B_q = 32` x `B_k = 64` (4 KV blocks) 2-stage `cp.async`
//      pipeline with 8 warps (256 threads) fitting inside 48,896 B (< 48 KiB)
//      static shared memory (no dynamic SMEM attribute needed):
//      - Warps split 2x4 (`warp_q = warp_id >> 2` owns 16 queries;
//        `warp_k = warp_id & 3` owns 16-key KV block in QK/softmax and
//        64 output channels in PV).
//      - Native SM120/SM121 FP4 Tensor Cores (`m16n8k64`, `scale_vec::4X`,
//        `.ue4m3`) with Dual-Residual stacked `Q^(1) + Q^(2)`, transposed
//        butterfly softmax, Underflow-Aware Probability Scaling (`c = 8`,
//        `kappa = 2` iff `c * p_max <= 21/1024`, else `6`), and hardware
//        `-0.0` (0x8) -> `+0.0` (0x0) E2M1 canonicalization.
//      - Optional 6-op branchless SWAR RaZeR-KV (`0x8` -> `±4.0 + ±1.0`)
//        reading `neg_sv` from bit 7 of `.ue4m3` scale bytes.
//      - Optional fused BF16 attention-sink prologue (arbitrary `num_sink_tokens`,
//        typically block 0 = 16 tokens) respecting causal masking even inside
//        sink positions (`q_offset == 0`).
//      - Supports both single-stream (`block_table`) and same-chunk-len batched
//        (`block_table_ptrs[blockIdx.z]`) continuation prefill for arbitrary
//        `q_offset` (`seq_len_start`), `q_len`, and `kv_len`.
//   3. Multi-arch safety:
//      - Block-scaled `mma.sync...kind::mxf4nvf4` is guarded under
//        `__CUDA_ARCH__ >= 1200` (arch/family-specific SM120/SM121 targets)
//        with a clean `__trap()` fallback on SM103a/generic PTX so every target
//        compiles with 0 B stack frame, 0 B spills, and 0 odd `desc[UR]`.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// ============================================================================
// Blackwell ISA feature guards (matching Flux `hc_common.cuh` contract)
// ============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    (defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || \
     defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_SPECIFIC__))
#define BITFP4_PF_HAS_E2M1_CVT 1
#else
#define BITFP4_PF_HAS_E2M1_CVT 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && \
    (defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || \
     defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_SPECIFIC__))
#define BITFP4_PF_HAS_SM120_MX_MMA 1
#else
#define BITFP4_PF_HAS_SM120_MX_MMA 0
#endif

namespace bitfp4_prefill {

constexpr unsigned int kHdim = 256u;
constexpr unsigned int kHdimHalf = kHdim / 2u;         // 128 bytes
constexpr unsigned int kHdimGroups = kHdim / 16u;      // 16 groups
constexpr unsigned int kBlockSize = 16u;               // Flux NVFP4 block_size=16
constexpr unsigned int kBlocksPerTile = 4u;            // B_k = 64 = 4 * 16
constexpr unsigned int kTileK = kBlocksPerTile * kBlockSize;  // 64
constexpr unsigned int kTileQ = 32u;                   // B_q = 32 query rows / CTA
constexpr unsigned int kQPerWarpQ = 16u;               // 16 queries per warp_q half
constexpr unsigned int kQHalves = kQPerWarpQ / 4u;     // 4 x (4-query dual-residual MMA)
constexpr unsigned int kQTile8 = kQPerWarpQ / 8u;      // 2 x (8-query PV MMA)
constexpr unsigned int kStages = 2u;
constexpr unsigned int kWarps = 8u;
constexpr unsigned int kThreads = kWarps * 32u;        // 256
constexpr float kLog2E = 1.44269504088896340736f;
constexpr float kNegInf = -1.0e30f;
// BitDecoding-FP4 Eq. 4 underflow-aware probability threshold:
//   c = 8.0f (folded as +3.0f in exp2f), tau = 21/1024.
constexpr float kUnderflowTauC = 21.0f / 1024.0f;

// Runtime flags bitmask shared by Rust host and CUDA kernels.
constexpr unsigned int kFlagApplyWht = 1u << 0;

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

__device__ __forceinline__ unsigned char cvt_f32_to_ue4m3(float x) {
    const float clamped = fminf(fmaxf(x, 0.0f), 448.0f);
#if BITFP4_PF_HAS_E2M1_CVT
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
#if BITFP4_PF_HAS_E2M1_CVT
    unsigned int h2;
    const unsigned short sb2 = static_cast<unsigned short>(clean);
    asm("{ .reg .b16 hs; mov.b16 hs, %1; cvt.rn.f16x2.e4m3x2 %0, hs; }" : "=r"(h2) : "h"(sb2));
    return __half2float(*reinterpret_cast<const __half*>(&h2));
#else
    return __half2float(__nv_cvt_fp8_to_halfraw(clean, __NV_E4M3));
#endif
}

__device__ __forceinline__ unsigned char quant2_e2m1(float v0, float v1) {
#if BITFP4_PF_HAS_E2M1_CVT
    unsigned int r;
    asm("{\n\t"
        ".reg .b8 b0;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n\t"
        "mov.b32 %0, {b0, 0, 0, 0};\n\t"
        "}"
        : "=r"(r)
        : "f"(v0), "f"(v1));
    // Canonicalize hardware `-0.0` (0x8) -> `+0.0` (0x0) with zero cross-nibble carry.
    const unsigned int neg0 = r & ~((r & 0x77777777u) + 0x77777777u) & 0x88888888u;
    r ^= neg0;
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
#if BITFP4_PF_HAS_E2M1_CVT
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
    // Canonicalize hardware `-0.0` (0x8) -> `+0.0` (0x0) with zero cross-nibble carry.
    const unsigned int neg0 = r & ~((r & 0x77777777u) + 0x77777777u) & 0x88888888u;
    r ^= neg0;
    return r;
#else
    const unsigned int b0 = quant2_e2m1(v[0] * q_inv, v[1] * q_inv);
    const unsigned int b1 = quant2_e2m1(v[2] * q_inv, v[3] * q_inv);
    const unsigned int b2 = quant2_e2m1(v[4] * q_inv, v[5] * q_inv);
    const unsigned int b3 = quant2_e2m1(v[6] * q_inv, v[7] * q_inv);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
#endif
}

// 6-op branchless SWAR RaZeR-KV nibble splitter (`0x8` -> `±4.0 + ±1.0`).
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
#if BITFP4_PF_HAS_SM120_MX_MMA
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
        const unsigned int sfa_other = __shfl_xor_sync(0xffffffffu, sfa_raw, 1);
        const unsigned int sfa_fr0 = (lane & 1u) ? sfa_other : sfa_raw;
        const unsigned int sfa_fr1 = (lane & 1u) ? sfa_raw : sfa_other;
        const unsigned int blk_lo = (lane & 3u) >> 1;
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
// Shared Memory Layout (48,896 B <= 49,152 B static SMEM limit)
// ============================================================================

struct KvPipeSmem {
    alignas(16) unsigned char sK_data[kStages][kTileK][kHdimHalf];           // 16,384 B
    alignas(16) unsigned char sK_scale[kStages][kTileK][kHdimGroups];        //  2,048 B
    alignas(16) unsigned char sV_data[kStages][kBlocksPerTile][kHdim][8];    // 16,384 B
    alignas(16) unsigned char sV_scale[kStages][kBlocksPerTile][kHdim];      //  2,048 B
};

union KvOrSinkSmem {
    KvPipeSmem pipe;                                                         // 36,864 B
    alignas(16) float sink_o[kTileQ][kHdim];                                 // 32,768 B
};

struct PrefillBitFp4Smem {
    KvOrSinkSmem kv_or_sink;                                                 // 36,864 B
    alignas(16) unsigned char sQ_data[2u * kTileQ][kHdimHalf];               //  8,192 B
    alignas(16) unsigned char sQ_scale[2u * kTileQ][kHdimGroups];            //  1,024 B
    alignas(16) unsigned char sP_data[kTileQ][kTileK / 2u];                  //  1,024 B
    alignas(16) unsigned char sP_scale[kTileQ][kBlocksPerTile];              //    128 B
    float sQ_logit_scale[kTileQ];                                            //    128 B
    float sBlkMax[kBlocksPerTile][kTileQ];                                   //    512 B
    float sBlkSum[kBlocksPerTile][kTileQ];                                   //    512 B
    float sRunningM[kTileQ];                                                 //    128 B
    float sRunningL[kTileQ];                                                 //    128 B
    float sAlpha[kTileQ];                                                    //    128 B
    unsigned int sQKvLim[kTileQ];                                            //    128 B
};

static_assert(sizeof(PrefillBitFp4Smem) <= 49152u,
              "PrefillBitFp4Smem must fit inside 48 KiB static SMEM");

// ============================================================================
// Core SM120/SM121 BitDecoding-FP4 Chunked Continuation Prefill Body
// ============================================================================

template <bool kRazerKv>
__device__ __forceinline__ void prefill_paged_bitfp4_core(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ my_block_table,
    const unsigned int q_len,
    const unsigned int kv_len,
    const unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned int o_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int causal_mask_enabled,
    const unsigned int flags,
    const __nv_bfloat16* __restrict__ K_sink_bf16,
    const __nv_bfloat16* __restrict__ V_sink_bf16,
    const unsigned int num_sink_tokens
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_start = blockIdx.y * kTileQ;
    if (q_head >= num_q_heads || q_start >= q_len) {
        return;
    }

    const unsigned int q_tile_len =
        (q_len - q_start < kTileQ) ? (q_len - q_start) : kTileQ;
    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;   // 0..7
    const unsigned int lane = tid & 31u;     // 0..31
    const unsigned int group_id = lane >> 2; // 0..7
    const unsigned int tid4 = lane & 3u;     // 0..3
    const unsigned int warp_q = warp_id >> 2;  // 0 (q 0..15) or 1 (q 16..31)
    const unsigned int warp_k = warp_id & 3u;  // 0..3 (KV block / output channel quarter)

    __shared__ PrefillBitFp4Smem sm;

    // Resolve per-query exclusive KV visibility limit `[0, sQKvLim[r])`.
    if (tid < kTileQ) {
        unsigned int qlim = 0u;
        if (tid < q_tile_len) {
            if (causal_mask_enabled != 0u) {
                const unsigned int causal_end = q_offset + q_start + tid + 1u;
                qlim = (causal_end < kv_len) ? causal_end : kv_len;
            } else {
                qlim = kv_len;
            }
        }
        sm.sQKvLim[tid] = qlim;
        sm.sRunningM[tid] = kNegInf;
        sm.sRunningL[tid] = 0.0f;
        sm.sAlpha[tid] = 0.0f;
        sm.sQ_logit_scale[tid] = 0.0f;
    }
    __syncthreads();

    // Since `sQKvLim` is monotonically non-decreasing across `0..q_tile_len`,
    // `sQKvLim[q_tile_len - 1]` is the exact CTA-wide KV upper bound.
    const unsigned int max_kv_lim = sm.sQKvLim[q_tile_len - 1u];

    // Output accumulators in MMA `m16n8k64` layout:
    //   warp `(warp_q, warp_k)` owns queries `warp_q*16 .. +15` (2 tiles of 8)
    //   and output channels `warp_k*64 .. +63` (4 tiles of 16).
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

    if (max_kv_lim > 0u) {
        const bool use_bf16_sink =
            (K_sink_bf16 != nullptr) && (V_sink_bf16 != nullptr) && (num_sink_tokens > 0u);
        const unsigned int sink_skip_tokens = use_bf16_sink ? num_sink_tokens : 0u;
        // Floor division ensures partial block-0 sinks (e.g. num_sink_tokens=4)
        // still process tokens [num_sink_tokens..16) from FP4 block 0 while
        // full block-0 sinks (num_sink_tokens=16) skip block 0 completely.
        const unsigned int first_fp4_blk = use_bf16_sink ? (num_sink_tokens / kBlockSize) : 0u;
        const unsigned int total_blocks = (max_kv_lim + kBlockSize - 1u) / kBlockSize;
        const unsigned int fp4_blocks =
            (total_blocks > first_fp4_blk) ? (total_blocks - first_fp4_blk) : 0u;
        const unsigned int num_tiles =
            (fp4_blocks > 0u) ? ((fp4_blocks + kBlocksPerTile - 1u) / kBlocksPerTile) : 0u;
        const bool apply_wht = (flags & kFlagApplyWht) != 0u;

        // --------------------------------------------------------------------
        // Prologue: optional BF16 sink attention + WHT + Dual-Residual Q quant
        //   8 warps cooperate over 32 query rows (4 rows per warp).
        // --------------------------------------------------------------------
        for (unsigned int r = warp_id; r < kTileQ; r += kWarps) {
            if (r < q_tile_len) {
                const __nv_bfloat16* q_ptr =
                    Q + static_cast<unsigned long long>(q_start + r) * q_stride
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

                // Optional BF16 attention-sink fusion on raw unrotated Q, respecting
                // per-row causal visibility `sQKvLim[r]`.
                if (use_bf16_sink) {
                    const unsigned int sink_lim = min(num_sink_tokens, sm.sQKvLim[r]);
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
                        // Carry `c = 8.0f` (+3.0f in exp2f) to match FP4 loop identically.
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
                        sm.kv_or_sink.sink_o[r][lane * 8u + j] = os[j];
                    }
                    if (lane == 0u) {
                        sm.sRunningM[r] = m_s;
                        sm.sRunningL[r] = l_s;
                    }
                }

                if (num_tiles > 0u) {
                    if (apply_wht) {
                        wht256_warp_plain(qv, lane);
                    }

                    // Dynamic per-query normalization into full-precision UE4M3 range.
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
                        sm.sQ_logit_scale[r] =
                            q_nz ? (inv_sqrt_d * kLog2E * __fdiv_rn(q_amax, 256.0f)) : 0.0f;
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

                    reinterpret_cast<unsigned int*>(sm.sQ_data[2u * r])[lane] = q1_word;
                    reinterpret_cast<unsigned int*>(sm.sQ_data[2u * r + 1u])[lane] = q2_word;
                    if ((lane & 1u) == 0u) {
                        sm.sQ_scale[2u * r][lane >> 1] = s1_u8;
                        sm.sQ_scale[2u * r + 1u][lane >> 1] = s2_u8;
                    }
                }
            } else if (num_tiles > 0u) {
                reinterpret_cast<unsigned int*>(sm.sQ_data[2u * r])[lane] = 0u;
                reinterpret_cast<unsigned int*>(sm.sQ_data[2u * r + 1u])[lane] = 0u;
                if ((lane & 1u) == 0u) {
                    sm.sQ_scale[2u * r][lane >> 1] = 0u;
                    sm.sQ_scale[2u * r + 1u][lane >> 1] = 0u;
                }
            }
        }
        __syncthreads();

        // Load BF16 sink accumulators into MMA registers before the FP4 pipeline
        // reuses `sm.kv_or_sink.pipe`.
        if (use_bf16_sink) {
            #pragma unroll
            for (unsigned int qt = 0u; qt < kQTile8; ++qt) {
                const unsigned int c0 = warp_q * kQPerWarpQ + qt * 8u + tid4 * 2u;
                const unsigned int c1 = c0 + 1u;
                #pragma unroll
                for (int dt = 0; dt < 4; ++dt) {
                    const unsigned int d0 = warp_k * 64u + static_cast<unsigned int>(dt) * 16u + group_id;
                    const unsigned int d1 = d0 + 8u;
                    o_acc[qt][dt][0] = (c0 < q_tile_len) ? sm.kv_or_sink.sink_o[c0][d0] : 0.0f;
                    o_acc[qt][dt][1] = (c1 < q_tile_len) ? sm.kv_or_sink.sink_o[c1][d0] : 0.0f;
                    o_acc[qt][dt][2] = (c0 < q_tile_len) ? sm.kv_or_sink.sink_o[c0][d1] : 0.0f;
                    o_acc[qt][dt][3] = (c1 < q_tile_len) ? sm.kv_or_sink.sink_o[c1][d1] : 0.0f;
                }
            }
            __syncthreads();
        }

        // --------------------------------------------------------------------
        // Main Double-Buffered FP4 KV Tile Loop (`B_k = 64` = 4 blocks / tile)
        // --------------------------------------------------------------------
        auto issue_kv_tile = [&](unsigned int tile_idx, unsigned int stage) {
            const unsigned int tile_blk0 = first_fp4_blk + tile_idx * kBlocksPerTile;
            const unsigned int valid_blks = min(kBlocksPerTile, total_blocks - tile_blk0);
            // Keep block IDs in scalar registers (zero local stack frame).
            const int bid0 = (0u < valid_blks) ? my_block_table[tile_blk0 + 0u] : -1;
            const int bid1 = (1u < valid_blks) ? my_block_table[tile_blk0 + 1u] : -1;
            const int bid2 = (2u < valid_blks) ? my_block_table[tile_blk0 + 2u] : -1;
            const int bid3 = (3u < valid_blks) ? my_block_table[tile_blk0 + 3u] : -1;
            auto bid_of = [&](unsigned int b) -> int {
                return (b == 0u) ? bid0 : ((b == 1u) ? bid1 : ((b == 2u) ? bid2 : bid3));
            };

            // 1. K data: 64 keys x 128 B = 8,192 B = 512 x 16 B (2 per thread)
            #pragma unroll
            for (unsigned int it = 0u; it < 2u; ++it) {
                const unsigned int idx = it * kThreads + tid;  // 0..511
                const unsigned int tok_in_tile = idx >> 3;     // 0..63
                const unsigned int b = tok_in_tile >> 4;       // 0..3
                const unsigned int tok_in_blk = tok_in_tile & 15u;
                const unsigned int chunk16 = idx & 7u;         // 0..7
                const int bid = bid_of(b);
                const bool pred = (b < valid_blks) && (bid >= 0);
                const unsigned long long phys = pred ? static_cast<unsigned long long>(bid) : 0ull;
                const unsigned char* gptr =
                    K_cache + phys * block_stride_bytes
                            + (static_cast<unsigned long long>(tok_in_blk) * num_kv_heads + kv_head) * kHdimHalf
                            + chunk16 * 16u;
                const unsigned int saddr = static_cast<unsigned int>(
                    __cvta_generic_to_shared(&sm.kv_or_sink.pipe.sK_data[stage][tok_in_tile][chunk16 * 16u]));
                cp_async_cg_16_z(saddr, gptr, pred);
            }

            // 2. V^T data: 4 blocks x 256 ch x 8 B = 8,192 B = 512 x 16 B (2 per thread)
            #pragma unroll
            for (unsigned int b_pair = 0u; b_pair < 2u; ++b_pair) {
                const unsigned int b = b_pair * 2u + (tid >> 7);  // 0..3
                const unsigned int off16 = (tid & 127u) * 16u;    // 0..2032
                const int bid = bid_of(b);
                const bool pred = (b < valid_blks) && (bid >= 0);
                const unsigned long long phys = pred ? static_cast<unsigned long long>(bid) : 0ull;
                const unsigned char* gptr =
                    V_cache + phys * block_stride_bytes
                            + static_cast<unsigned long long>(kv_head) * (kHdim * 8u)
                            + off16;
                const unsigned int saddr = static_cast<unsigned int>(
                    __cvta_generic_to_shared(&sm.kv_or_sink.pipe.sV_data[stage][b][0][0])) + off16;
                cp_async_cg_16_z(saddr, gptr, pred);
            }

            // 3. K scales (threads 0..63) + V^T scales (threads 64..127), 16 B each
            if (tid < 64u) {
                const unsigned int tok_in_tile = tid;
                const unsigned int b = tok_in_tile >> 4;
                const unsigned int tok_in_blk = tok_in_tile & 15u;
                const int bid = bid_of(b);
                const bool pred = (b < valid_blks) && (bid >= 0);
                const unsigned long long phys = pred ? static_cast<unsigned long long>(bid) : 0ull;
                const unsigned char* gptr =
                    K_cache + phys * block_stride_bytes + data_section_bytes
                            + (static_cast<unsigned long long>(tok_in_blk) * num_kv_heads + kv_head) * kHdimGroups;
                const unsigned int saddr = static_cast<unsigned int>(
                    __cvta_generic_to_shared(&sm.kv_or_sink.pipe.sK_scale[stage][tok_in_tile][0]));
                cp_async_cg_16_z(saddr, gptr, pred);
            } else if (tid < 128u) {
                const unsigned int vidx = tid - 64u;     // 0..63
                const unsigned int b = vidx >> 4;        // 0..3
                const unsigned int chunk16 = vidx & 15u; // 0..15 (256 B / 16 B)
                const int bid = bid_of(b);
                const bool pred = (b < valid_blks) && (bid >= 0);
                const unsigned long long phys = pred ? static_cast<unsigned long long>(bid) : 0ull;
                const unsigned char* gptr =
                    V_cache + phys * block_stride_bytes + data_section_bytes
                            + static_cast<unsigned long long>(kv_head) * kHdim
                            + chunk16 * 16u;
                const unsigned int saddr = static_cast<unsigned int>(
                    __cvta_generic_to_shared(&sm.kv_or_sink.pipe.sV_scale[stage][b][chunk16 * 16u]));
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

            const unsigned int tile_tok0 = (first_fp4_blk + tile * kBlocksPerTile) * kBlockSize;
            const unsigned int warp_tok0 = tile_tok0 + warp_k * kBlockSize;
            const unsigned int key_pos0 = warp_tok0 + group_id;
            const unsigned int key_pos1 = key_pos0 + 8u;

            // ----------------------------------------------------------------
            // Stage 1: Native FP4 QK MMA (`S^T = K @ Q^T`) with Dual-Residual Q
            //   Warp `(warp_q, warp_k)` scores KV block `warp_k` (16 keys)
            //   against the 16 queries in half `warp_q`.
            // ----------------------------------------------------------------
            float s0[kQHalves];
            float s1[kQHalves];
            float m_blk[kQHalves];

            const unsigned int k_row0 = warp_k * kBlockSize + group_id;
            const unsigned int k_row1 = k_row0 + 8u;
            const unsigned int k_sfa_row = warp_k * kBlockSize + group_id + 8u * (lane & 1u);

            #pragma unroll
            for (unsigned int qh = 0u; qh < kQHalves; ++qh) {
                float acc_qk[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const unsigned int q_nc = (warp_q * kQPerWarpQ + qh * 4u) * 2u + group_id;
                #pragma unroll
                for (unsigned int kt = 0u; kt < 4u; ++kt) {
                    const unsigned int byte_base = kt * 32u + tid4 * 4u;
                    const unsigned int sc_base = kt * 4u;
                    const unsigned int a0 = *reinterpret_cast<const unsigned int*>(
                        &sm.kv_or_sink.pipe.sK_data[stage][k_row0][byte_base]);
                    const unsigned int a1 = *reinterpret_cast<const unsigned int*>(
                        &sm.kv_or_sink.pipe.sK_data[stage][k_row1][byte_base]);
                    const unsigned int a2 = *reinterpret_cast<const unsigned int*>(
                        &sm.kv_or_sink.pipe.sK_data[stage][k_row0][byte_base + 16u]);
                    const unsigned int a3 = *reinterpret_cast<const unsigned int*>(
                        &sm.kv_or_sink.pipe.sK_data[stage][k_row1][byte_base + 16u]);
                    const unsigned int sfa = *reinterpret_cast<const unsigned int*>(
                        &sm.kv_or_sink.pipe.sK_scale[stage][k_sfa_row][sc_base]);

                    const unsigned int b0 = *reinterpret_cast<const unsigned int*>(
                        &sm.sQ_data[q_nc][byte_base]);
                    const unsigned int b1 = *reinterpret_cast<const unsigned int*>(
                        &sm.sQ_data[q_nc][byte_base + 16u]);
                    const unsigned int sfb = *reinterpret_cast<const unsigned int*>(
                        &sm.sQ_scale[q_nc][sc_base]);

                    mma_nvfp4_m16n8k64<kRazerKv>(acc_qk, a0, a1, a2, a3, b0, b1, sfa, sfb, lane);
                }

                const unsigned int r = warp_q * kQPerWarpQ + qh * 4u + tid4;
                const unsigned int q_sl = sm.sQKvLim[r];
                const float lscale = sm.sQ_logit_scale[r];
                const bool vis0 = (key_pos0 >= sink_skip_tokens) && (key_pos0 < q_sl);
                const bool vis1 = (key_pos1 >= sink_skip_tokens) && (key_pos1 < q_sl);
                const float v0 = vis0 ? ((acc_qk[0] + acc_qk[1]) * lscale) : kNegInf;
                const float v1 = vis1 ? ((acc_qk[2] + acc_qk[3]) * lscale) : kNegInf;
                s0[qh] = v0;
                s1[qh] = v1;

                // Transposed Softmax butterfly max across 16 keys of block `warp_k`.
                float mb = fmaxf(v0, v1);
                mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 4));
                mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 8));
                mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 16));
                m_blk[qh] = mb;
                if (group_id == 0u) {
                    sm.sBlkMax[warp_k][r] = mb;
                }
            }
            __syncthreads();

            // Cross-warp 4-block max merge -> unified `m_new` and `alpha` per query.
            if (tid < kTileQ) {
                const float mt = fmaxf(
                    fmaxf(sm.sBlkMax[0][tid], sm.sBlkMax[1][tid]),
                    fmaxf(sm.sBlkMax[2][tid], sm.sBlkMax[3][tid]));
                const float m_old = sm.sRunningM[tid];
                const float m_new = fmaxf(m_old, mt);
                const float alpha =
                    (m_old > -1.0e29f && m_new > -1.0e29f) ? exp2f(m_old - m_new) : 0.0f;
                sm.sRunningM[tid] = m_new;
                sm.sAlpha[tid] = alpha;
            }
            __syncthreads();

            // ----------------------------------------------------------------
            // Stage 2: Underflow-Aware Probability Scaling + E2M1 Quantization
            // ----------------------------------------------------------------
            #pragma unroll
            for (unsigned int qh = 0u; qh < kQHalves; ++qh) {
                const unsigned int r = warp_q * kQPerWarpQ + qh * 4u + tid4;
                const unsigned int q_sl = sm.sQKvLim[r];
                const float m_new = sm.sRunningM[r];
                const float mb = m_blk[qh];

                const bool vis0 = (key_pos0 >= sink_skip_tokens) && (key_pos0 < q_sl) && (m_new > -1.0e29f);
                const bool vis1 = (key_pos1 >= sink_skip_tokens) && (key_pos1 < q_sl) && (m_new > -1.0e29f);
                const float p0_c = vis0 ? exp2f(s0[qh] - m_new + 3.0f) : 0.0f;
                const float p1_c = vis1 ? exp2f(s1[qh] - m_new + 3.0f) : 0.0f;

                float l_b = p0_c + p1_c;
                l_b += __shfl_xor_sync(0xffffffffu, l_b, 4);
                l_b += __shfl_xor_sync(0xffffffffu, l_b, 8);
                l_b += __shfl_xor_sync(0xffffffffu, l_b, 16);
                if (group_id == 0u) {
                    sm.sBlkSum[warp_k][r] = l_b;
                }

                const float p_max_c =
                    (mb > -1.0e29f && m_new > -1.0e29f) ? exp2f(mb - m_new + 3.0f) : 0.0f;
                const float kappa = (p_max_c <= kUnderflowTauC) ? 2.0f : 6.0f;
                const unsigned char sp_u8 = cvt_f32_to_ue4m3(__fdiv_rn(p_max_c, kappa));
                const float sp_f = cvt_ue4m3_to_f32(sp_u8);
                const float inv_sp = (sp_f > 0.0f) ? __fdiv_rn(1.0f, sp_f) : 0.0f;

                const float p0_odd = __shfl_xor_sync(0xffffffffu, p0_c, 4);
                const float p1_odd = __shfl_xor_sync(0xffffffffu, p1_c, 4);
                if ((group_id & 1u) == 0u) {
                    const unsigned int pair = group_id >> 1;  // 0..3
                    sm.sP_data[r][warp_k * 8u + pair] = quant2_e2m1(p0_c * inv_sp, p0_odd * inv_sp);
                    sm.sP_data[r][warp_k * 8u + 4u + pair] = quant2_e2m1(p1_c * inv_sp, p1_odd * inv_sp);
                }
                if (group_id == 0u) {
                    sm.sP_scale[r][warp_k] = sp_u8;
                }
            }
            __syncthreads();

            // Update running softmax denominator `l_new = l_old * alpha + sum_w l_b`.
            if (tid < kTileQ) {
                const float l_tile =
                    sm.sBlkSum[0][tid] + sm.sBlkSum[1][tid] + sm.sBlkSum[2][tid] + sm.sBlkSum[3][tid];
                sm.sRunningL[tid] = sm.sRunningL[tid] * sm.sAlpha[tid] + l_tile;
            }

            // ----------------------------------------------------------------
            // Stage 3: Native FP4 PV MMA (`O^T = V^T @ P^T`)
            //   Warp `(warp_q, warp_k)` updates 16 queries x 64 output channels.
            // ----------------------------------------------------------------
            const unsigned int blk_lo = tid4 >> 1;
            const unsigned int half_w = tid4 & 1u;
            const unsigned int blk_hi = 2u + blk_lo;

            #pragma unroll
            for (unsigned int qt = 0u; qt < kQTile8; ++qt) {
                const unsigned int q_base = warp_q * kQPerWarpQ + qt * 8u;
                const float alpha0 = sm.sAlpha[q_base + tid4 * 2u];
                const float alpha1 = sm.sAlpha[q_base + tid4 * 2u + 1u];

                const unsigned int p_nc = q_base + group_id;
                const unsigned int b0 = *reinterpret_cast<const unsigned int*>(&sm.sP_data[p_nc][tid4 * 4u]);
                const unsigned int b1 = *reinterpret_cast<const unsigned int*>(&sm.sP_data[p_nc][16u + tid4 * 4u]);
                const unsigned int sfb = *reinterpret_cast<const unsigned int*>(&sm.sP_scale[p_nc][0]);

                #pragma unroll
                for (int dt = 0; dt < 4; ++dt) {
                    o_acc[qt][dt][0] *= alpha0;
                    o_acc[qt][dt][1] *= alpha1;
                    o_acc[qt][dt][2] *= alpha0;
                    o_acc[qt][dt][3] *= alpha1;

                    const unsigned int d_base = warp_k * 64u + static_cast<unsigned int>(dt) * 16u;
                    const unsigned int fr0 = d_base + group_id;
                    const unsigned int fr1 = fr0 + 8u;
                    const unsigned int m_sfa = d_base + group_id + 8u * (lane & 1u);

                    const unsigned int a0 = *reinterpret_cast<const unsigned int*>(
                        &sm.kv_or_sink.pipe.sV_data[stage][blk_lo][fr0][half_w * 4u]);
                    const unsigned int a1 = *reinterpret_cast<const unsigned int*>(
                        &sm.kv_or_sink.pipe.sV_data[stage][blk_lo][fr1][half_w * 4u]);
                    const unsigned int a2 = *reinterpret_cast<const unsigned int*>(
                        &sm.kv_or_sink.pipe.sV_data[stage][blk_hi][fr0][half_w * 4u]);
                    const unsigned int a3 = *reinterpret_cast<const unsigned int*>(
                        &sm.kv_or_sink.pipe.sV_data[stage][blk_hi][fr1][half_w * 4u]);
                    const unsigned int sfa_v =
                        (static_cast<unsigned int>(sm.kv_or_sink.pipe.sV_scale[stage][0][m_sfa])) |
                        (static_cast<unsigned int>(sm.kv_or_sink.pipe.sV_scale[stage][1][m_sfa]) << 8) |
                        (static_cast<unsigned int>(sm.kv_or_sink.pipe.sV_scale[stage][2][m_sfa]) << 16) |
                        (static_cast<unsigned int>(sm.kv_or_sink.pipe.sV_scale[stage][3][m_sfa]) << 24);

                    mma_nvfp4_m16n8k64<kRazerKv>(o_acc[qt][dt], a0, a1, a2, a3, b0, b1, sfa_v, sfb, lane);
                }
            }
            __syncthreads();
        }
    }

    // ------------------------------------------------------------------------
    // Epilogue: Normalize `O_acc / l` and store BF16 directly to `O`
    // ------------------------------------------------------------------------
    #pragma unroll
    for (unsigned int qt = 0u; qt < kQTile8; ++qt) {
        const unsigned int c0 = warp_q * kQPerWarpQ + qt * 8u + tid4 * 2u;
        const unsigned int c1 = c0 + 1u;
        const float l0 = (c0 < q_tile_len) ? sm.sRunningL[c0] : 0.0f;
        const float l1 = (c1 < q_tile_len) ? sm.sRunningL[c1] : 0.0f;
        const float inv_l0 = (l0 > 0.0f) ? __fdiv_rn(1.0f, l0) : 0.0f;
        const float inv_l1 = (l1 > 0.0f) ? __fdiv_rn(1.0f, l1) : 0.0f;

        if (c0 < q_tile_len) {
            __nv_bfloat16* o_ptr0 =
                O + static_cast<unsigned long long>(q_start + c0) * o_stride
                  + static_cast<unsigned long long>(q_head) * kHdim;
            #pragma unroll
            for (int dt = 0; dt < 4; ++dt) {
                const unsigned int d0 = warp_k * 64u + static_cast<unsigned int>(dt) * 16u + group_id;
                o_ptr0[d0] = __float2bfloat16(o_acc[qt][dt][0] * inv_l0);
                o_ptr0[d0 + 8u] = __float2bfloat16(o_acc[qt][dt][2] * inv_l0);
            }
        }
        if (c1 < q_tile_len) {
            __nv_bfloat16* o_ptr1 =
                O + static_cast<unsigned long long>(q_start + c1) * o_stride
                  + static_cast<unsigned long long>(q_head) * kHdim;
            #pragma unroll
            for (int dt = 0; dt < 4; ++dt) {
                const unsigned int d0 = warp_k * 64u + static_cast<unsigned int>(dt) * 16u + group_id;
                o_ptr1[d0] = __float2bfloat16(o_acc[qt][dt][1] * inv_l1);
                o_ptr1[d0 + 8u] = __float2bfloat16(o_acc[qt][dt][3] * inv_l1);
            }
        }
    }
}

}  // namespace bitfp4_prefill

// ============================================================================
// Exported Chunked Continuation Prefill Entrypoints
// ============================================================================

extern "C" __global__ __launch_bounds__(256, 1) void prefill_paged_bitfp4_sm120(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_table,
    const unsigned int q_len,
    const unsigned int kv_len,
    const unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned int o_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int causal_mask_enabled,
    const unsigned int flags,
    const __nv_bfloat16* __restrict__ K_sink_bf16,
    const __nv_bfloat16* __restrict__ V_sink_bf16,
    const unsigned int num_sink_tokens
) {
    bitfp4_prefill::prefill_paged_bitfp4_core<false>(
        Q, K_cache, V_cache, O, block_table,
        q_len, kv_len, q_offset, num_q_heads, num_kv_heads,
        inv_sqrt_d, q_stride, o_stride, block_stride_bytes, data_section_bytes,
        causal_mask_enabled, flags, K_sink_bf16, V_sink_bf16, num_sink_tokens);
}

extern "C" __global__ __launch_bounds__(256, 1) void prefill_paged_bitfp4_razer_sm120(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_table,
    const unsigned int q_len,
    const unsigned int kv_len,
    const unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned int o_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int causal_mask_enabled,
    const unsigned int flags,
    const __nv_bfloat16* __restrict__ K_sink_bf16,
    const __nv_bfloat16* __restrict__ V_sink_bf16,
    const unsigned int num_sink_tokens
) {
    bitfp4_prefill::prefill_paged_bitfp4_core<true>(
        Q, K_cache, V_cache, O, block_table,
        q_len, kv_len, q_offset, num_q_heads, num_kv_heads,
        inv_sqrt_d, q_stride, o_stride, block_stride_bytes, data_section_bytes,
        causal_mask_enabled, flags, K_sink_bf16, V_sink_bf16, num_sink_tokens);
}

extern "C" __global__ __launch_bounds__(256, 1) void prefill_paged_bitfp4_batched_sm120(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* const* __restrict__ block_table_ptrs,
    const unsigned int batch_size,
    const unsigned int q_len,
    const unsigned int kv_len,
    const unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned int o_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int causal_mask_enabled,
    const unsigned int flags,
    const __nv_bfloat16* __restrict__ K_sink_bf16,
    const __nv_bfloat16* __restrict__ V_sink_bf16,
    const unsigned int num_sink_tokens
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch_size) return;
    const unsigned long long q_batch_off =
        static_cast<unsigned long long>(b) * q_len * q_stride;
    const unsigned long long o_batch_off =
        static_cast<unsigned long long>(b) * q_len * o_stride;
    const unsigned long long sink_batch_off =
        static_cast<unsigned long long>(b) * num_sink_tokens * num_kv_heads * bitfp4_prefill::kHdim;
    const __nv_bfloat16* k_sink_b =
        (K_sink_bf16 != nullptr) ? (K_sink_bf16 + sink_batch_off) : nullptr;
    const __nv_bfloat16* v_sink_b =
        (V_sink_bf16 != nullptr) ? (V_sink_bf16 + sink_batch_off) : nullptr;
    bitfp4_prefill::prefill_paged_bitfp4_core<false>(
        Q + q_batch_off, K_cache, V_cache, O + o_batch_off, block_table_ptrs[b],
        q_len, kv_len, q_offset, num_q_heads, num_kv_heads,
        inv_sqrt_d, q_stride, o_stride, block_stride_bytes, data_section_bytes,
        causal_mask_enabled, flags, k_sink_b, v_sink_b, num_sink_tokens);
}

extern "C" __global__ __launch_bounds__(256, 1) void prefill_paged_bitfp4_batched_razer_sm120(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* const* __restrict__ block_table_ptrs,
    const unsigned int batch_size,
    const unsigned int q_len,
    const unsigned int kv_len,
    const unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned int o_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int causal_mask_enabled,
    const unsigned int flags,
    const __nv_bfloat16* __restrict__ K_sink_bf16,
    const __nv_bfloat16* __restrict__ V_sink_bf16,
    const unsigned int num_sink_tokens
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch_size) return;
    const unsigned long long q_batch_off =
        static_cast<unsigned long long>(b) * q_len * q_stride;
    const unsigned long long o_batch_off =
        static_cast<unsigned long long>(b) * q_len * o_stride;
    const unsigned long long sink_batch_off =
        static_cast<unsigned long long>(b) * num_sink_tokens * num_kv_heads * bitfp4_prefill::kHdim;
    const __nv_bfloat16* k_sink_b =
        (K_sink_bf16 != nullptr) ? (K_sink_bf16 + sink_batch_off) : nullptr;
    const __nv_bfloat16* v_sink_b =
        (V_sink_bf16 != nullptr) ? (V_sink_bf16 + sink_batch_off) : nullptr;
    bitfp4_prefill::prefill_paged_bitfp4_core<true>(
        Q + q_batch_off, K_cache, V_cache, O + o_batch_off, block_table_ptrs[b],
        q_len, kv_len, q_offset, num_q_heads, num_kv_heads,
        inv_sqrt_d, q_stride, o_stride, block_stride_bytes, data_section_bytes,
        causal_mask_enabled, flags, k_sink_b, v_sink_b, num_sink_tokens);
}
