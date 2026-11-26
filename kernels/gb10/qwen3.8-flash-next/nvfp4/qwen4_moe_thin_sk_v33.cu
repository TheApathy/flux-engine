// SPDX-License-Identifier: AGPL-3.0-only
//
// V33 Thin-M Split-K MoE Verify Kernel (`v33_thin_sk`, `rows = 4..256`):
// Bandwidth-optimal NVFP4 MoE GEMM targeting SM120 / SM121 verify batches.
//
// Hybrid per-expert architecture inside a single 4-warp (`block = 128`),
// `<= 9.72 KiB` SMEM (`9,824 B` flat / `9,952 B` TS4E; `<= 12 KiB`, 6 resident CTAs/SM = `1,128` CTAs on SM120)
// persistent atomic work-stealing kernel:
//   * Thin experts (`M_expert <= 4`, >90% of experts at `C=1..16`):
//     Zero-padding register-streamed BF16 -> FP32 path (`RM = 1, 2, 4`) using
//     coalesced 128-bit `ld.global.cs.v4.u32` (streaming L2 bypass) + 4-shuffle
//     in-warp 4x4 transpose + SM120 hardware `cvt.rn.f16x2.e2m1x2` &
//     `cvt.rn.f16x2.e4m3x2` + exact FP16 `__hmul2` + FP32 `fmaf`.
//   * Hot experts (`M_expert >= 5`, `rows = 16..256`):
//     Zero-LUT, zero-`sBt`, zero-SWAR `TILE_M = 32` (`M=5..16` runs 1 M-tile;
//     `M=17..128` loops in 32-row chunks) native `mma.sync.aligned.m16n8k16`
//     (`f32.f16.f16.f32`) feeding hardware-decoded `cvt.rn.f16x2.e2m1x2 * sc`
//     directly into MMA B-fragments (2x fewer weight reads than `v28_m16`,
//     zero `xq`/`oq` MXFP8 quantization round-trips!).
//   * Dynamic Split-K on GATEUP (`K=2560`, `SK in {1,2,4}`) and DOWN (`K=640`,
//     `SK in {1,2}`) with zero-contention first-wave indexing + atomic work-
//     stealing over `(work_item, n_tile, k_slice)` via `Workspace::next_task_*`.
//   * Full support for both flat `[K/16, N]` E4M3 scales and Atlas-TS4E-128
//     compressed scale blobs, and exact BF16 or MXFP8 (`slot_q`/`slot_s`) output.

#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define OI640_STEP_K 32
#define OI640_TRANSPOSED 1
#define flash_next_orig_i640_prefill flash_next_f32_transposed_compact_thin_sk_v33
#include "moe_w4a16_orig_i640_compact_prefill.cuh"
#include <cuda_fp16.h>
#include <type_traits>

namespace oi640 = flash_next_f32_transposed_compact_thin_sk_v33;

namespace {

constexpr unsigned V33_ITEM_TILE_M = 128u;
constexpr unsigned V33_THREADS = 128u;
constexpr unsigned V33_MAX_RM = 4u;
constexpr unsigned V33_MMA_TILE_M = 32u;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1030)
constexpr unsigned long long V33_KEEP_SLOT_BYTES = 64ull << 20;
#else
constexpr unsigned long long V33_KEEP_SLOT_BYTES = 12ull << 20;
#endif

__device__ __forceinline__ uint4 v33_ld_cs_v4(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint4 v33_ld_ca_v4(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr));
    return v;
}

__device__ __forceinline__ unsigned v33_ld_cs_u32(const void* ptr) {
    unsigned v;
    asm volatile("ld.global.cs.u32 %0, [%1];\n" : "=r"(v) : "l"(ptr));
    return v;
}

__device__ __forceinline__ unsigned v33_ld_ca_u32(const void* ptr) {
    unsigned v;
    asm volatile("ld.global.ca.u32 %0, [%1];\n" : "=r"(v) : "l"(ptr));
    return v;
}

__device__ __forceinline__ void v33_cp16_cg(void* smem_ptr, const void* gmem_ptr) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_ptr));
}

__device__ __forceinline__ void v33_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void v33_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void v33_ldsm4(unsigned smem_addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(smem_addr));
}

__device__ __forceinline__ void v33_mma_f16(
    float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void v33_st_wb_f4(void* ptr, float4 v) {
    asm volatile("st.global.wb.v4.f32 [%0], {%1, %2, %3, %4};\n"
                 :: "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
}

__device__ __forceinline__ void v33_st_wb_f2(void* ptr, float2 v) {
    asm volatile("st.global.wb.v2.f32 [%0], {%1, %2};\n"
                 :: "l"(ptr), "f"(v.x), "f"(v.y)
                 : "memory");
}

__device__ __forceinline__ void v33_st_wb_u2(void* ptr, uint2 v) {
    asm volatile("st.global.wb.v2.u32 [%0], {%1, %2};\n"
                 :: "l"(ptr), "r"(v.x), "r"(v.y)
                 : "memory");
}

__device__ __forceinline__ void v33_st_l2_evict_last_u2(void* ptr, uint2 v) {
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "l"(pol)
                 : "memory");
}

__device__ __forceinline__ void v33_pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void v33_pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ unsigned v33_pack_bf162(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    unsigned u;
    memcpy(&u, &v, sizeof(u));
    return u;
}

__device__ __forceinline__ unsigned v33_bf162_to_f162_clamped(unsigned bf2) {
    const float f0 = fminf(fmaxf(__uint_as_float(bf2 << 16), -65504.0f), 65504.0f);
    const float f1 = fminf(fmaxf(__uint_as_float(bf2 & 0xffff0000u), -65504.0f), 65504.0f);
    const __half2 h = __floats2half2_rn(f0, f1);
    unsigned u;
    memcpy(&u, &h, sizeof(u));
    return u;
}

__device__ __forceinline__ unsigned char v33_cvt_f32_e4m3(float v) {
    unsigned short p = 0u;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p) : "f"(v), "f"(0.0f));
    return static_cast<unsigned char>(p & 0xffu);
}

__device__ __forceinline__ unsigned v33_pack_e4m3x4(float a, float b, float c, float d) {
    unsigned short p0, p1;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p0) : "f"(a), "f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p1) : "f"(c), "f"(d));
    return static_cast<unsigned>(p0) | (static_cast<unsigned>(p1) << 16);
}

// Hardware SM120/SM121 E2M1x2 (single byte -> __half2 {w_even, w_odd}).
__device__ __forceinline__ unsigned v33_e2m1x2_to_h2(unsigned byte_val) {
#if defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || \
    defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_SPECIFIC__)
    unsigned hb;
    const unsigned short b16 = static_cast<unsigned short>(byte_val & 0xffu);
    asm("{ .reg .b8 bx, bz; mov.b16 {bx, bz}, %1; cvt.rn.f16x2.e2m1x2 %0, bx; }"
        : "=r"(hb) : "h"(b16));
    return hb;
#else
    static constexpr float LUT[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    const __half2 h = __floats2half2_rn(LUT[byte_val & 15u], LUT[(byte_val >> 4) & 15u]);
    unsigned u;
    memcpy(&u, &h, sizeof(u));
    return u;
#endif
}

// Hardware SM120/SM121 E4M3 single byte -> broadcast __half2 {s, s}.
__device__ __forceinline__ __half2 v33_e4m3_to_h2_bcast(unsigned s_byte) {
    unsigned hs;
    const unsigned short sb2 = static_cast<unsigned short>((s_byte & 0xffu) | ((s_byte & 0xffu) << 8));
    asm("{ .reg .b16 hsb; mov.b16 hsb, %1; cvt.rn.f16x2.e4m3x2 %0, hsb; }"
        : "=r"(hs) : "h"(sb2));
    return *reinterpret_cast<const __half2*>(&hs);
}

// Hardware SM120/SM121 E4M3x4 -> 2 packed __half2 pairs {s0,s1} and {s2,s3}.
__device__ __forceinline__ void v33_decode_e4m3x4_to_h2_pair(
    unsigned s4, __half2& h01, __half2& h23
) {
    unsigned hs01, hs23;
    asm("{ .reg .b16 s01, s23;\n"
        "  mov.b32 {s01, s23}, %2;\n"
        "  cvt.rn.f16x2.e4m3x2 %0, s01;\n"
        "  cvt.rn.f16x2.e4m3x2 %1, s23; }\n"
        : "=r"(hs01), "=r"(hs23)
        : "r"(s4));
    h01 = *reinterpret_cast<const __half2*>(&hs01);
    h23 = *reinterpret_cast<const __half2*>(&hs23);
}

// Hardware SM120/SM121 E2M1x4 (2 bytes from low or high half of packed4) -> 2 __half2 pairs.
template <bool HI_HALF>
__device__ __forceinline__ void v33_decode_e2m1x4_half_to_h2(
    unsigned packed4, unsigned& h0, unsigned& h1
) {
#if defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || \
    defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_SPECIFIC__)
    if constexpr (!HI_HALF) {
        asm("{ .reg .b16 lo16, hi16; .reg .b8 b0, b1;\n"
            "  mov.b32 {lo16, hi16}, %2;\n"
            "  mov.b16 {b0, b1}, lo16;\n"
            "  cvt.rn.f16x2.e2m1x2 %0, b0;\n"
            "  cvt.rn.f16x2.e2m1x2 %1, b1; }\n"
            : "=r"(h0), "=r"(h1) : "r"(packed4));
    } else {
        asm("{ .reg .b16 lo16, hi16; .reg .b8 b2, b3;\n"
            "  mov.b32 {lo16, hi16}, %2;\n"
            "  mov.b16 {b2, b3}, hi16;\n"
            "  cvt.rn.f16x2.e2m1x2 %0, b2;\n"
            "  cvt.rn.f16x2.e2m1x2 %1, b3; }\n"
            : "=r"(h0), "=r"(h1) : "r"(packed4));
    }
#else
    const unsigned shift = HI_HALF ? 16u : 0u;
    h0 = v33_e2m1x2_to_h2((packed4 >> shift) & 0xffu);
    h1 = v33_e2m1x2_to_h2((packed4 >> (shift + 8u)) & 0xffu);
#endif
}

// 4x4 word transpose across `q4 = (lane >> 2) & 3` (exchanges lanes ^4 and ^8).
__device__ __forceinline__ void v33_transpose_4x4_words(
    uint4 raw, unsigned q4, unsigned& p0, unsigned& p1, unsigned& p2, unsigned& p3
) {
    const bool odd1 = (q4 & 1u) != 0u;
    const unsigned s0 = __shfl_xor_sync(0xffffffffu, odd1 ? raw.x : raw.y, 4);
    const unsigned s2 = __shfl_xor_sync(0xffffffffu, odd1 ? raw.z : raw.w, 4);
    const unsigned t0 = odd1 ? s0 : raw.x;
    const unsigned t1 = odd1 ? raw.y : s0;
    const unsigned t2 = odd1 ? s2 : raw.z;
    const unsigned t3 = odd1 ? raw.w : s2;
    const bool odd2 = (q4 & 2u) != 0u;
    const unsigned u0 = __shfl_xor_sync(0xffffffffu, odd2 ? t0 : t2, 8);
    const unsigned u1 = __shfl_xor_sync(0xffffffffu, odd2 ? t1 : t3, 8);
    p0 = odd2 ? u0 : t0;
    p1 = odd2 ? u1 : t1;
    p2 = odd2 ? t2 : u0;
    p3 = odd2 ? t3 : u1;
}

// Register-direct 4x4 quad transpose + MXFP8 quantizer (0 SMEM, 0 barriers).
// Bit-identical to v28_mxfp8_reg_quad.
__device__ __forceinline__ void v33_mxfp8_reg_quad(
    unsigned u0, unsigned u1, unsigned u2, unsigned u3,
    unsigned tig, unsigned (&qq)[2], unsigned& ss
) {
    const bool odd1 = (tig & 1u) != 0u;
    const unsigned s0 = __shfl_xor_sync(0xffffffffu, odd1 ? u0 : u1, 1);
    const unsigned s2 = __shfl_xor_sync(0xffffffffu, odd1 ? u2 : u3, 1);
    const unsigned t0 = odd1 ? s0 : u0;
    const unsigned t1 = odd1 ? u1 : s0;
    const unsigned t2 = odd1 ? s2 : u2;
    const unsigned t3 = odd1 ? u3 : s2;
    const bool odd2 = (tig & 2u) != 0u;
    const unsigned r0 = __shfl_xor_sync(0xffffffffu, odd2 ? t0 : t2, 2);
    const unsigned r1 = __shfl_xor_sync(0xffffffffu, odd2 ? t1 : t3, 2);
    const unsigned u[4] = {
        odd2 ? r0 : t0,
        odd2 ? r1 : t1,
        odd2 ? t2 : r0,
        odd2 ? t3 : r1
    };
    float v[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[2 * i] = __uint_as_float(u[i] << 16);
        v[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
    }
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    qq[0] = 0u;
    qq[1] = 0u;
    ss = 0u;
    if (amax > 0.0f) {
        int e;
        const float m = frexpf(__fmul_rn(amax, 1.0f / 448.0f), &e);
        int k = (m == 0.5f) ? e - 1 : e;
        k = max(-127, min(127, k));
        const float inv = __int_as_float((127 - k) << 23);
        qq[0] = v33_pack_e4m3x4(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv);
        qq[1] = v33_pack_e4m3x4(v[4] * inv, v[5] * inv, v[6] * inv, v[7] * inv);
        ss = static_cast<unsigned>(k + 127);
    }
}

// Per-task + per-launch metadata stored in SMEM so both thin and MMA loops stay spill-free (0 B stack frame).
struct alignas(16) V33TaskMeta {
    const unsigned char* B0;
    const unsigned char* B1;
    const unsigned char* S0;
    const unsigned char* S1;
    uint32_t* next_task_ptr;
    float s2a;
    float s2b;
    unsigned task;
    unsigned task_count;
    unsigned tasks_per_wi;
    unsigned pairs;
    unsigned ks_per_slice;
    unsigned cta_m;
    unsigned cta_n;
    unsigned sk_idx;
    unsigned ks_beg;
    unsigned ks_end;
    unsigned item_rows;
};

// Unified Dynamic SMEM layout (<= 9,952 B = 9.72 KiB <= 12 KiB -> 6-8 CTAs/SM):
//   Stage payload (8,192 B):
//     * Thin path (`M <= 4`): `sX[2][4][64]` FP32 (2,048 B)
//     * Hot MMA path (`M >= 5`): `sA[2][32][32]` FP16 (4,096 B) + `sB[2][16][128]` raw NVFP4 (4,096 B)
//       (and aliases `sGateStage[32][64]` FP32 = 8,192 B in SK=1 GATEUP epilogue!)
//   sScale[2][512] : 1,024 B (2 stages x 4 scale groups x 128 B for thin TS4E; 2 stages x 2 groups x 128 B for MMA)
//   sRowIds[128]   :   512 B (cached sorted_token_ids for the active 128-row work item)
//   sBase[128]     :   128 B (TS4E only: per-column base scales for active N-tile)
//   sMeta          :    96 B (launch + task broadcast + pointers + scale2 values)
template <bool TS4E> struct V33Smem {
    static constexpr unsigned PAYLOAD_BYTES = 8192u; // 4,096 B sA + 4,096 B sB (or 2,048 B sX)
    static constexpr unsigned SCALE_BYTES = 1024u;   // Used by MMA (flat & TS4E) and Thin TS4E
    static constexpr unsigned ROW_IDS_BYTES = V33_ITEM_TILE_M * 4u; // 512 B
    static constexpr unsigned BASE_BYTES = TS4E ? 128u : 0u;
    static constexpr unsigned META_BYTES = (sizeof(V33TaskMeta) + 15u) & ~15u;

    static constexpr unsigned OFF_PAYLOAD = 0u;
    static constexpr unsigned OFF_A = 0u;
    static constexpr unsigned OFF_B = 4096u;
    static constexpr unsigned OFF_SCALE = OFF_PAYLOAD + PAYLOAD_BYTES;
    static constexpr unsigned OFF_ROW_IDS = OFF_SCALE + SCALE_BYTES;
    static constexpr unsigned OFF_BASE = OFF_ROW_IDS + ROW_IDS_BYTES;
    static constexpr unsigned OFF_META = OFF_BASE + BASE_BYTES;
    static constexpr unsigned BYTES = (OFF_META + META_BYTES + 15u) & ~15u;

    static_assert(!TS4E ? (BYTES == 9824u) : (BYTES == 9952u), "V33Smem byte contract");
    static_assert(BYTES <= 12288u, "V33Smem must stay <= 12 KiB for 6-8 CTAs/SM");
};

template <int RM>
__device__ __forceinline__ void v33_load_x_k64(
    float (*sX)[V33_MAX_RM][64],
    unsigned stage,
    const __nv_bfloat16* __restrict__ X,
    const unsigned* sRowIds,
    unsigned row_base,
    unsigned valid_r,
    unsigned k_base,
    unsigned K,
    unsigned tid,
    bool indirect_rows
) {
    if (tid < static_cast<unsigned>(RM) * 8u) {
        const unsigned r = tid >> 3;
        const unsigned c8 = tid & 7u;
        float* const dst = &sX[stage][r][c8 << 3];
        if (r < valid_r) {
            const unsigned src_row = indirect_rows ? sRowIds[row_base + r] : (row_base + r);
            const uint4 raw = v33_ld_ca_v4(
                X + static_cast<unsigned long long>(src_row) * K + k_base + (c8 << 3));
            *reinterpret_cast<float4*>(dst) = make_float4(
                __uint_as_float(raw.x << 16),
                __uint_as_float(raw.x & 0xffff0000u),
                __uint_as_float(raw.y << 16),
                __uint_as_float(raw.y & 0xffff0000u));
            *reinterpret_cast<float4*>(dst + 4) = make_float4(
                __uint_as_float(raw.z << 16),
                __uint_as_float(raw.z & 0xffff0000u),
                __uint_as_float(raw.w << 16),
                __uint_as_float(raw.w & 0xffff0000u));
        } else {
            const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            *reinterpret_cast<float4*>(dst) = z;
            *reinterpret_cast<float4*>(dst + 4) = z;
        }
    }
}

template <int RM>
__device__ __forceinline__ void v33_accum_group16(
    float (&acc)[RM][4],
    uint4 raw_b,
    unsigned s4,
    unsigned q4,
    const float (*sX)[V33_MAX_RM][64],
    unsigned stage,
    unsigned k_in_64
) {
    unsigned p0, p1, p2, p3;
    v33_transpose_4x4_words(raw_b, q4, p0, p1, p2, p3);
    __half2 s01, s23;
    v33_decode_e4m3x4_to_h2_pair(s4, s01, s23);

    const unsigned pw[4] = {p0, p1, p2, p3};
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        {
            unsigned hw0, hw1;
            v33_decode_e2m1x4_half_to_h2<false>(pw[p], hw0, hw1);
            const float2 w0 = __half22float2(__hmul2(*reinterpret_cast<const __half2*>(&hw0), __low2half2(s01)));
            const float2 w1 = __half22float2(__hmul2(*reinterpret_cast<const __half2*>(&hw1), __high2half2(s01)));
            #pragma unroll
            for (int r = 0; r < RM; ++r) {
                const float2 xv = *reinterpret_cast<const float2*>(&sX[stage][r][k_in_64 + static_cast<unsigned>(2 * p)]);
                acc[r][0] = fmaf(w0.x, xv.x, fmaf(w0.y, xv.y, acc[r][0]));
                acc[r][1] = fmaf(w1.x, xv.x, fmaf(w1.y, xv.y, acc[r][1]));
            }
        }
        {
            unsigned hw2, hw3;
            v33_decode_e2m1x4_half_to_h2<true>(pw[p], hw2, hw3);
            const float2 w2 = __half22float2(__hmul2(*reinterpret_cast<const __half2*>(&hw2), __low2half2(s23)));
            const float2 w3 = __half22float2(__hmul2(*reinterpret_cast<const __half2*>(&hw3), __high2half2(s23)));
            #pragma unroll
            for (int r = 0; r < RM; ++r) {
                const float2 xv = *reinterpret_cast<const float2*>(&sX[stage][r][k_in_64 + static_cast<unsigned>(2 * p)]);
                acc[r][2] = fmaf(w2.x, xv.x, fmaf(w2.y, xv.y, acc[r][2]));
                acc[r][3] = fmaf(w3.x, xv.x, fmaf(w3.y, xv.y, acc[r][3]));
            }
        }
    }
}

// Decode ONE k32 TS4E-128 bucket whose raw nibbles were already prefetched into `dst_even + STRIDE` via cp.async!
// Overwrites `dst_even` (`[0..STRIDE-1]`) and `dst_odd` (`[STRIDE..2*STRIDE-1]`) in place with exact E4M3 scales.
template <bool GATEUP, unsigned N, unsigned K>
__device__ __forceinline__ void v33_decode_ts4e_k32_from_smem_warp(
    const unsigned char* blob,
    const unsigned char* sBase_sub,
    unsigned char* dst_even,
    unsigned k32,
    unsigned cta_n,
    unsigned lane
) {
    constexpr unsigned STRIDE = GATEUP ? 64u : 128u;
    constexpr unsigned TS4E_NUM_K32 = K >> 5;
    constexpr unsigned TS4E_N_TILES = N >> 7;
    constexpr unsigned TS4E_NUM_BKTS = TS4E_NUM_K32 * TS4E_N_TILES;
    constexpr unsigned TS4E_NIBBLES_BYTES = TS4E_NUM_K32 * N;
    constexpr unsigned TS4E_PAGE_BASES_LEN = ((TS4E_NUM_BKTS >> 7) + 4u) & ~3u;
    constexpr unsigned TS4E_PAGE_BASES_OFF = TS4E_NIBBLES_BYTES + N;
    constexpr unsigned TS4E_BKT_OFF_OFF = TS4E_PAGE_BASES_OFF + TS4E_PAGE_BASES_LEN * 4u;
    constexpr unsigned TS4E_BKT_OFF_BYTES_ALIGNED = (((TS4E_NUM_BKTS + 1u) * 2u) + 15u) & ~15u;
    constexpr unsigned TS4E_EXC_OFF = TS4E_BKT_OFF_OFF + TS4E_BKT_OFF_BYTES_ALIGNED;

    unsigned char* const dst_odd = dst_even + STRIDE;
    constexpr unsigned SWAR_LANES = GATEUP ? 16u : 32u;
    if (lane < SWAR_LANES) {
        const unsigned off = lane << 2;
        const unsigned base_w = *reinterpret_cast<const unsigned*>(sBase_sub + off);
        const unsigned nib = *reinterpret_cast<const unsigned*>(dst_odd + off);
        const unsigned c0 = nib & 0x0F0F0F0Fu;
        const unsigned e0 = (c0 + 0x01010101u) & 0x10101010u;
        *reinterpret_cast<unsigned*>(dst_even + off) = base_w + (c0 & ~(e0 - (e0 >> 4)));
        const unsigned c1 = (nib >> 4) & 0x0F0F0F0Fu;
        const unsigned e1 = (c1 + 0x01010101u) & 0x10101010u;
        *reinterpret_cast<unsigned*>(dst_odd + off) = base_w + (c1 & ~(e1 - (e1 >> 4)));
    }
    __syncwarp();
    unsigned beg = 0u, cnt = 0u;
    if (lane == 0u) {
        const unsigned bkt = k32 * TS4E_N_TILES + (cta_n >> 7);
        const unsigned* const page_bases =
            reinterpret_cast<const unsigned*>(blob + TS4E_PAGE_BASES_OFF);
        const unsigned short* const bkt_off =
            reinterpret_cast<const unsigned short*>(blob + TS4E_BKT_OFF_OFF);
        beg = __ldg(&page_bases[bkt >> 7]) + static_cast<unsigned>(__ldg(&bkt_off[bkt]));
        cnt = (__ldg(&page_bases[(bkt + 1u) >> 7]) + static_cast<unsigned>(__ldg(&bkt_off[bkt + 1u]))) - beg;
    }
    beg = __shfl_sync(0xffffffffu, beg, 0);
    cnt = __shfl_sync(0xffffffffu, cnt, 0);
    const unsigned short* const exc =
        reinterpret_cast<const unsigned short*>(blob + TS4E_EXC_OFF);
    const unsigned col_half = cta_n & 64u;
    for (unsigned i = lane; i < cnt; i += 32u) {
        const unsigned e = static_cast<unsigned>(__ldg(&exc[beg + i]));
        if constexpr (GATEUP) {
            if ((e & 64u) == col_half) {
                dst_even[((e & 128u) >> 1) | (e & 63u)] = static_cast<unsigned char>(e >> 8);
            }
        } else {
            dst_even[e & 255u] = static_cast<unsigned char>(e >> 8);
        }
    }
}

template <bool GATEUP, int EPI, bool TS4E>
__device__ __forceinline__ void v33_thin_sk_impl(
    const __nv_bfloat16* __restrict__ X,
    const unsigned long long* __restrict__ pk0,
    const unsigned long long* __restrict__ sc0,
    const float* __restrict__ s20,
    const unsigned long long* __restrict__ pk1,
    const unsigned long long* __restrict__ sc1,
    const float* __restrict__ s21,
    __nv_bfloat16* __restrict__ out_bf16,
    unsigned char* __restrict__ slot_q,
    unsigned char* __restrict__ slot_s,
    float* __restrict__ partials,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned N_in,
    unsigned K_in,
    unsigned split_k,
    const __nv_bfloat16* __restrict__ shared_gate = nullptr,
    float* __restrict__ ss_out = nullptr,
    unsigned sg_rows = 0u
) {
    using L = V33Smem<TS4E>;
    constexpr unsigned N = GATEUP ? oi640::INTER : oi640::HIDDEN;
    constexpr unsigned K = GATEUP ? oi640::HIDDEN : oi640::INTER;
    constexpr unsigned TN = GATEUP ? 64u : 128u;
    constexpr unsigned N_TILES = N / TN; // 10 (GU) or 20 (DN)
    constexpr unsigned KS_TOTAL = K / 64u; // 40 (GU) or 10 (DN)
    constexpr unsigned TS4E_NIBBLES_BYTES = (K >> 5) * N;

    unsigned dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    const bool valid_sk = GATEUP ? (split_k == 1u || split_k == 2u || split_k == 4u)
                                 : (split_k == 1u || split_k == 2u);
    if (gridDim.x == 0u || gridDim.x > oi640::FIXED_GRID || blockDim.x != V33_THREADS ||
        dyn_smem < L::BYTES || !valid_sk) {
        if (threadIdx.x == 0u) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }

    // Ensure preceding stream-ordered producer (fused_sort_plan / reduce_silu /
    // gateup) has flushed its writes before reading status/workspace/X under PDL.
    v33_pdl_wait();

    const bool shape_ok = (N_in == N && K_in == K);
    const bool out_ok = (split_k > 1u)
        ? (partials != nullptr)
        : (GATEUP ? (out_bf16 != nullptr)
                  : (EPI == 3 ? (slot_q != nullptr && slot_s != nullptr) : (out_bf16 != nullptr)));
    if (!shape_ok || !out_ok || X == nullptr || pk0 == nullptr || sc0 == nullptr || s20 == nullptr ||
        (GATEUP && (pk1 == nullptr || sc1 == nullptr || s21 == nullptr)) ||
        expert_offsets == nullptr || sorted_token_ids == nullptr ||
        !oi640::contract_ok(contract) || !oi640::workspace_ok_worksteal(workspace, contract) ||
        status == nullptr || status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0u) oi640::fail(status, oi640::ERR_ABI);
        return;
    }

    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned tid = threadIdx.x;

    // Optional fused shared-expert gate scalar precompute (`0-ULP` bit-identical
    // to `qwen4_moe_thin_sk_shared_gate_v33`): CTAs `0..sg_rows-1` compute
    // `ss_out[r] = sigmoid(dot(X[r], shared_gate))` using the idle dynamic SMEM
    // before entering the routed work-stealing loop, saving 1 launch per MoE layer.
    if constexpr (GATEUP) {
        if (ss_out != nullptr && blockIdx.x < sg_rows) {
            const unsigned r = blockIdx.x;
            if (shared_gate == nullptr) {
                if (tid == 0u) ss_out[r] = 1.0f;
            } else {
                float* const s_dot = reinterpret_cast<float*>(smem);
                const __nv_bfloat16* const x_row =
                    X + static_cast<unsigned long long>(r) * 2560ull;
                float d0 = 0.0f;
                float d1 = 0.0f;
                #pragma unroll
                for (unsigned i = 0u; i < 10u; ++i) {
                    const unsigned c0 = tid + i * 256u;
                    const unsigned c1 = c0 + 128u;
                    d0 += __bfloat162float(x_row[c0]) * __bfloat162float(shared_gate[c0]);
                    d1 += __bfloat162float(x_row[c1]) * __bfloat162float(shared_gate[c1]);
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    d0 += __shfl_down_sync(0xffffffffu, d0, offset);
                    d1 += __shfl_down_sync(0xffffffffu, d1, offset);
                }
                if ((tid & 31u) == 0u) {
                    const unsigned w = tid >> 5;
                    s_dot[w] = d0;
                    s_dot[w + 4u] = d1;
                }
                __syncthreads();
                if (tid == 0u) {
                    float total = 0.0f;
                    #pragma unroll
                    for (unsigned w = 0u; w < 8u; ++w) total += s_dot[w];
                    ss_out[r] = 1.0f / (1.0f + __expf(-total));
                }
                __syncthreads();
            }
        }
    }

    const unsigned wc = workspace->work_count;
    const unsigned pairs = workspace->expanded;
    const unsigned tasks_per_wi = N_TILES * split_k;
    const unsigned task_count = wc * tasks_per_wi;
    if (blockIdx.x >= task_count) return;

    unsigned* const sRowIds = reinterpret_cast<unsigned*>(smem + L::OFF_ROW_IDS);
    unsigned char* const sBase = smem + L::OFF_BASE;
    V33TaskMeta* const sMeta = reinterpret_cast<V33TaskMeta*>(smem + L::OFF_META);

    const unsigned warp_id = tid >> 5;
    const unsigned lane = tid & 31u;

    if (tid == 0u) {
        sMeta->task = blockIdx.x;
        sMeta->task_count = task_count;
        sMeta->tasks_per_wi = tasks_per_wi;
        sMeta->pairs = pairs;
        sMeta->ks_per_slice = KS_TOTAL / split_k;
        sMeta->next_task_ptr = GATEUP ? &workspace->next_task_gu : &workspace->next_task_dn;
    }
    __syncthreads();

    #pragma unroll 1
    while (sMeta->task < sMeta->task_count) {
        const unsigned task_cur = sMeta->task;
        const unsigned tpw = sMeta->tasks_per_wi;
        const unsigned wi = task_cur / tpw;
        const unsigned rem = task_cur % tpw;
        // Group adjacent N-tiles together within the same K-slice so even/odd 64B GATEUP
        // tiles hit the same 128B DRAM cache line concurrently across neighboring CTAs.
        const unsigned sk_idx = rem / N_TILES;
        const unsigned nt = rem % N_TILES;
        const unsigned cta_n = nt * TN;
        const unsigned ks_beg = sk_idx * sMeta->ks_per_slice;
        const unsigned ks_end = ks_beg + sMeta->ks_per_slice;

        const oi640::WorkItem item = workspace->items[wi];
        if (item.expert >= oi640::EXPERTS) {
            if (tid == 0u) oi640::fail(status, oi640::ERR_PLAN);
            return;
        }
        const int m_start = expert_offsets[item.expert];
        const int M_expert = expert_offsets[item.expert + 1] - m_start;
        const int local = static_cast<int>(item.m_tile * V33_ITEM_TILE_M);
        const unsigned char* const B0 = reinterpret_cast<const unsigned char*>(pk0[item.expert]);
        const unsigned char* const S0 = reinterpret_cast<const unsigned char*>(sc0[item.expert]);
        const unsigned char* const B1 = GATEUP ? reinterpret_cast<const unsigned char*>(pk1[item.expert]) : B0;
        const unsigned char* const S1 = GATEUP ? reinterpret_cast<const unsigned char*>(sc1[item.expert]) : S0;
        const float s2a = s20[item.expert];
        const float s2b = GATEUP ? s21[item.expert] : 0.0f;
        if (m_start < 0 || M_expert <= 0 || local < 0 || local >= M_expert ||
            B0 == nullptr || S0 == nullptr || B1 == nullptr || S1 == nullptr ||
            !isfinite(s2a) || (GATEUP && !isfinite(s2b))) {
            if (tid == 0u) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const unsigned cta_m = static_cast<unsigned>(m_start + local);
        const unsigned item_rows = static_cast<unsigned>(
            min(M_expert - local, static_cast<int>(V33_ITEM_TILE_M)));

        if (tid == 0u) {
            sMeta->B0 = B0;
            sMeta->B1 = B1;
            sMeta->S0 = S0;
            sMeta->S1 = S1;
            sMeta->s2a = s2a;
            sMeta->s2b = s2b;
            sMeta->cta_m = cta_m;
            sMeta->cta_n = cta_n;
            sMeta->sk_idx = sk_idx;
            sMeta->ks_beg = ks_beg;
            sMeta->ks_end = ks_end;
            sMeta->item_rows = item_rows;
        }
        if constexpr (GATEUP) {
            if (tid < item_rows) {
                sRowIds[tid] = static_cast<unsigned>(sorted_token_ids[cta_m + tid]);
            }
        }
        if constexpr (TS4E) {
            if (tid < 32u) {
                if constexpr (GATEUP) {
                    const unsigned bm = tid >> 4;
                    const unsigned off = (tid & 15u) << 2;
                    const unsigned char* const bs = bm ? S1 : S0;
                    *reinterpret_cast<unsigned*>(&sBase[tid << 2]) =
                        v33_ld_ca_u32(bs + static_cast<unsigned long long>(TS4E_NIBBLES_BYTES) + cta_n + off);
                } else {
                    const unsigned off = tid << 2;
                    *reinterpret_cast<unsigned*>(&sBase[off]) =
                        v33_ld_ca_u32(S0 + static_cast<unsigned long long>(TS4E_NIBBLES_BYTES) + cta_n + off);
                }
            }
        }
        __syncthreads();

        const unsigned total_r = sMeta->item_rows;

        if (total_r <= V33_MAX_RM) {
            // =================================================================
            // PATH A: Thin Experts (`M_expert <= 4`): Zero-padding register-
            // streamed BF16 -> FP32 path (`RM = 1, 2, 4`), single weight read!
            // =================================================================
            float (*const sX)[V33_MAX_RM][64] =
                reinterpret_cast<float (*)[V33_MAX_RM][64]>(smem + L::OFF_PAYLOAD);
            unsigned char* const sScaleRaw = smem + L::OFF_SCALE;
            const unsigned n_blk = lane & 3u;
            const unsigned k_q = lane >> 2;
            const unsigned k_half = lane >> 4;
            const unsigned q4 = k_q & 3u;
            const unsigned col4_in_64 = (n_blk << 4) | (q4 << 2);
            const unsigned my_mtx = GATEUP ? (warp_id >> 1) : 0u;
            const unsigned my_n_half = GATEUP ? 0u : (warp_id >> 1);
            const unsigned my_k_par = warp_id & 1u;
            const unsigned my_warp_n_off = my_n_half << 6;
            const unsigned my_col_in_cta = my_warp_n_off | col4_in_64;

            // Warp-local async prefetch of 2 k32 TS4E nibble rows into dst_odd slots
            // (each decoding warp issues its own cp.async so cp_wait<0> + __syncwarp is race-free!).
            auto prefetch_ts4e_k64 = [&](unsigned stage, unsigned ks) {
                if constexpr (!TS4E) {
                    (void)stage; (void)ks; return;
                } else {
                    unsigned char* const sSt = sScaleRaw + stage * 512u;
                    const unsigned m_cta_n = sMeta->cta_n;
                    if constexpr (GATEUP) {
                        const unsigned sub = warp_id >> 1;
                        const unsigned t = warp_id & 1u;
                        if (lane < 4u) {
                            const unsigned char* const* const slot = sub ? &sMeta->S1 : &sMeta->S0;
                            const unsigned char* const blob = *reinterpret_cast<const unsigned char* const volatile*>(slot);
                            v33_cp16_cg(
                                sSt + ((sub * 4u + 2u * t + 1u) << 6) + (lane << 4),
                                blob + static_cast<unsigned long long>(2u * ks + t) * N + m_cta_n + (lane << 4));
                        }
                    } else if ((tid & 63u) < 8u) {
                        const unsigned t = tid >> 6;
                        const unsigned c = tid & 7u;
                        const unsigned char* const blob = *reinterpret_cast<const unsigned char* const volatile*>(&sMeta->S0);
                        v33_cp16_cg(
                            sSt + ((2u * t + 1u) << 7) + (c << 4),
                            blob + static_cast<unsigned long long>(2u * ks + t) * N + m_cta_n + (c << 4));
                    }
                    v33_cp_commit();
                }
            };

            auto decode_ts4e_k64 = [&](unsigned stage, unsigned ks) {
                if constexpr (!TS4E) {
                    (void)stage; (void)ks; return;
                } else {
                    unsigned char* const sSt = sScaleRaw + stage * 512u;
                    const unsigned m_cta_n = sMeta->cta_n;
                    if constexpr (GATEUP) {
                        const unsigned sub = warp_id >> 1;
                        const unsigned t = warp_id & 1u;
                        const unsigned char* const* const slot = sub ? &sMeta->S1 : &sMeta->S0;
                        const unsigned char* const blob = *reinterpret_cast<const unsigned char* const volatile*>(slot);
                        v33_decode_ts4e_k32_from_smem_warp<true, N, K>(
                            blob,
                            sBase + (sub << 6),
                            sSt + ((sub * 4u + 2u * t) << 6),
                            2u * ks + t, m_cta_n, lane);
                    } else if ((warp_id & 1u) == 0u) {
                        const unsigned t = warp_id >> 1;
                        const unsigned char* const blob = *reinterpret_cast<const unsigned char* const volatile*>(&sMeta->S0);
                        v33_decode_ts4e_k32_from_smem_warp<false, N, K>(
                            blob, sBase,
                            sSt + ((2u * t) << 7),
                            2u * ks + t, m_cta_n, lane);
                    }
                }
            };

            auto load_b_group = [&](unsigned kg) -> uint4 {
                const unsigned char* const* const slot = (GATEUP && my_mtx) ? &sMeta->B1 : &sMeta->B0;
                const unsigned char* const base_b = *reinterpret_cast<const unsigned char* const volatile*>(slot);
                const unsigned char* const b_ptr = base_b + sMeta->cta_n + my_warp_n_off + (n_blk << 4);
                return v33_ld_cs_v4(b_ptr + static_cast<unsigned long long>(kg * 8u + k_q) * N);
            };

            auto load_s_group = [&](unsigned stage, unsigned kg, unsigned g_in_64) -> unsigned {
                if constexpr (TS4E) {
                    const unsigned char* const sSt = sScaleRaw + stage * 512u;
                    if constexpr (GATEUP) {
                        return *reinterpret_cast<const unsigned*>(
                            &sSt[((my_mtx * 4u + g_in_64) << 6) + col4_in_64]);
                    } else {
                        return *reinterpret_cast<const unsigned*>(
                            &sSt[(g_in_64 << 7) + my_col_in_cta]);
                    }
                } else {
                    (void)stage; (void)g_in_64;
                    unsigned s4 = 0u;
                    if (lane < 16u) {
                        const unsigned char* const* const slot = (GATEUP && my_mtx) ? &sMeta->S1 : &sMeta->S0;
                        const unsigned char* const base_s = *reinterpret_cast<const unsigned char* const volatile*>(slot);
                        const unsigned char* const s_ptr = base_s + sMeta->cta_n + my_col_in_cta;
                        s4 = v33_ld_cs_u32(s_ptr + static_cast<unsigned long long>(kg) * N);
                    }
                    return __shfl_sync(0xffffffffu, s4, static_cast<int>(lane & 15u));
                }
            };

            auto run_thin = [&](auto rm_tag) {
                constexpr int RM = decltype(rm_tag)::value;
                const unsigned m_ks_end = sMeta->ks_end;
                float acc[RM][4];
                #pragma unroll
                for (int r = 0; r < RM; ++r)
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) acc[r][c] = 0.0f;

                const unsigned row_base = GATEUP ? 0u : sMeta->cta_m;
                const unsigned ks0 = sMeta->ks_beg;
                prefetch_ts4e_k64(0u, ks0);
                v33_load_x_k64<RM>(sX, 0u, X, sRowIds, row_base, total_r, ks0 * 64u, K, tid, GATEUP);
                uint4 b_cur = load_b_group(ks0 * 4u + my_k_par);
                if constexpr (TS4E) { v33_cp_wait<0>(); __syncwarp(); }
                decode_ts4e_k64(0u, ks0);
                __syncthreads();

                unsigned cur_st = 0u;
                #pragma unroll 1
                for (unsigned ks = ks0; ks < m_ks_end; ++ks, cur_st ^= 1u) {
                    const unsigned nxt_st = cur_st ^ 1u;
                    const bool has_next = (ks + 1u < m_ks_end);
                    const unsigned kg0 = ks * 4u + my_k_par;
                    const unsigned kg1 = kg0 + 2u;

                    // Issue async TS4E nibble prefetch for ks+1 BEFORE computing ks so nibbles overlap compute!
                    if (has_next) {
                        prefetch_ts4e_k64(nxt_st, ks + 1u);
                    }
                    uint4 b_nxt = load_b_group(kg1);
                    const unsigned s0 = load_s_group(cur_st, kg0, my_k_par);
                    v33_accum_group16<RM>(acc, b_cur, s0, q4, sX, cur_st, my_k_par * 16u + k_half * 8u);

                    if (has_next) {
                        b_cur = load_b_group((ks + 1u) * 4u + my_k_par);
                    }
                    const unsigned s1 = load_s_group(cur_st, kg1, 2u + my_k_par);
                    v33_accum_group16<RM>(acc, b_nxt, s1, q4, sX, cur_st, (2u + my_k_par) * 16u + k_half * 8u);

                    if (has_next) {
                        v33_load_x_k64<RM>(
                            sX, nxt_st, X, sRowIds, row_base, total_r, (ks + 1u) * 64u, K, tid, GATEUP);
                        if constexpr (TS4E) { v33_cp_wait<0>(); __syncwarp(); }
                        decode_ts4e_k64(nxt_st, ks + 1u);
                        __syncthreads();
                    }
                }

                #pragma unroll
                for (int r = 0; r < RM; ++r)
                    #pragma unroll
                    for (int c = 0; c < 4; ++c)
                        acc[r][c] += __shfl_xor_sync(0xffffffffu, acc[r][c], 16);

                __syncthreads();
                const unsigned ep_tid = threadIdx.x;
                const unsigned ep_col4 = ((ep_tid & 3u) << 4) | (ep_tid & 12u);
                if (((ep_tid ^ 32u) & 63u) < 16u) {
                    const unsigned slot_m = ep_tid >> 6;
                    #pragma unroll
                    for (int r = 0; r < RM; ++r)
                        *reinterpret_cast<float4*>(&sX[slot_m][r][ep_col4]) =
                            make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
                }
                __syncthreads();

                const unsigned base_m = sMeta->cta_m;
                const unsigned m_cta_n = sMeta->cta_n;
                const unsigned m_sk_idx = sMeta->sk_idx;
                if ((ep_tid & 63u) < 16u) {
                    const unsigned slot_m = ep_tid >> 6;
                    const float scale2 = (GATEUP && slot_m == 1u) ? sMeta->s2b : sMeta->s2a;
                    #pragma unroll
                    for (int r = 0; r < RM; ++r) {
                        const float4 oth = *reinterpret_cast<const float4*>(&sX[slot_m][r][ep_col4]);
                        acc[r][0] = (acc[r][0] + oth.x) * scale2;
                        acc[r][1] = (acc[r][1] + oth.y) * scale2;
                        acc[r][2] = (acc[r][2] + oth.z) * scale2;
                        acc[r][3] = (acc[r][3] + oth.w) * scale2;
                    }
                    if (split_k > 1u) {
                        const unsigned m_pairs = sMeta->pairs;
                        #pragma unroll
                        for (int r = 0; r < RM; ++r) {
                            if (static_cast<unsigned>(r) < total_r) {
                                const unsigned long long prow = base_m + static_cast<unsigned>(r);
                                if constexpr (GATEUP) {
                                    const unsigned long long off =
                                        (static_cast<unsigned long long>(m_sk_idx) * m_pairs + prow) * 1280ull +
                                        static_cast<unsigned long long>(slot_m) * 640ull + m_cta_n + ep_col4;
                                    v33_st_wb_f4(partials + off, make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]));
                                } else {
                                    const unsigned long long off =
                                        (static_cast<unsigned long long>(m_sk_idx) * m_pairs + prow) * 2560ull +
                                        m_cta_n + (slot_m << 6) + ep_col4;
                                    v33_st_wb_f4(partials + off, make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]));
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int r = 0; r < RM; ++r)
                            *reinterpret_cast<float4*>(&sX[slot_m][r][ep_col4]) =
                                make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
                    }
                }

                if (split_k == 1u) {
                    __syncthreads();
                    if constexpr (GATEUP) {
                        if (warp_id < total_r) {
                            const unsigned c = lane * 2u;
                            const float g0 = sX[0][warp_id][c], g1 = sX[0][warp_id][c + 1u];
                            const float u0 = sX[1][warp_id][c], u1 = sX[1][warp_id][c + 1u];
                            const float a0 = g0 * (1.0f / (1.0f + __expf(-g0))) * u0;
                            const float a1 = g1 * (1.0f / (1.0f + __expf(-g1))) * u1;
                            *reinterpret_cast<unsigned*>(out_bf16 + (base_m + warp_id) * 640ull + m_cta_n + c) =
                                v33_pack_bf162(a0, a1);
                        }
                    } else {
                        const unsigned half_idx = warp_id >> 1;
                        const unsigned col_in_64 = ((warp_id & 1u) << 5) | lane;
                        const unsigned g_col = m_cta_n + (warp_id << 5) + lane;
                        for (unsigned r = 0u; r < total_r; ++r) {
                            const unsigned long long prow = base_m + r;
                            const float v = sX[half_idx][r][col_in_64];
                            if constexpr (EPI == 3) {
                                const bool keep_slot =
                                    (static_cast<unsigned long long>(sMeta->pairs) * 2640ull) <= V33_KEEP_SLOT_BYTES;
                                const unsigned g_blk = (m_cta_n >> 5) + warp_id;
                                const float vb = __bfloat162float(__float2bfloat16(v));
                                float amax = fabsf(vb);
                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 16));
                                unsigned char qb = 0u, sb = 0u;
                                if (amax > 0.0f) {
                                    int e;
                                    const float m = frexpf(__fmul_rn(amax, 1.0f / 448.0f), &e);
                                    int k = (m == 0.5f) ? e - 1 : e;
                                    k = max(-127, min(127, k));
                                    const float inv = __int_as_float((127 - k) << 23);
                                    qb = v33_cvt_f32_e4m3(vb * inv);
                                    sb = static_cast<unsigned char>(k + 127);
                                }
                                if (keep_slot) {
                                    unsigned long long pol;
                                    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
                                    asm volatile("st.global.L2::cache_hint.u8 [%0], %1, %2;\n"
                                                 :: "l"(slot_q + prow * 2560ull + g_col),
                                                    "r"(static_cast<unsigned>(qb)), "l"(pol)
                                                 : "memory");
                                } else {
                                    slot_q[prow * 2560ull + g_col] = qb;
                                }
                                if (lane == 0u) slot_s[prow * 80ull + g_blk] = sb;
                            } else {
                                out_bf16[prow * 2560ull + g_col] = __float2bfloat16(v);
                            }
                        }
                    }
                }
            };

            if (total_r == 1u) {
                run_thin(std::integral_constant<int, 1>{});
            } else if (total_r == 2u) {
                run_thin(std::integral_constant<int, 2>{});
            } else {
                run_thin(std::integral_constant<int, 4>{});
            }
        } else {
            // =================================================================
            // PATH B: Hot Experts (`M_expert >= 5`): Native `m16n8k16` FP16
            // Tensor Core (`TILE_M = 32`, `BC = 128`, `k_step = 32`).
            // Feeds `cvt.rn.f16x2.e2m1x2 * sc` directly into MMA B-fragments
            // (zero LUT, zero sBt, zero SWAR transcode, zero xq/oq MXFP8!).
            // =================================================================
            unsigned char* const sA_raw = smem + L::OFF_A;
            unsigned char* const sB_raw = smem + L::OFF_B;
            unsigned char* const sScaleRaw = smem + L::OFF_SCALE;

            // Async cp.async issue for stage `stage` (`sB` + `sS` flat OR `TS4E` nibbles) — 0 stalls!
            auto issue_b_and_s_mma = [&](unsigned stage, unsigned k32) {
                const unsigned m_cta_n = sMeta->cta_n;
                const unsigned kp_r = tid >> 3; // 0..15
                const unsigned cc = tid & 7u;   // 0..7 (16B chunk)
                const unsigned dst_col = (cc << 4) ^ ((kp_r & 3u) << 4);
                const unsigned char* b_src;
                if constexpr (GATEUP) {
                    const unsigned bm = cc >> 2;
                    const unsigned c4 = cc & 3u;
                    b_src = (bm ? sMeta->B1 : sMeta->B0) +
                            static_cast<unsigned long long>(k32 * 16u + kp_r) * N + m_cta_n + (c4 << 4);
                } else {
                    b_src = sMeta->B0 +
                            static_cast<unsigned long long>(k32 * 16u + kp_r) * N + m_cta_n + (cc << 4);
                }
                v33_cp16_cg(sB_raw + stage * 2048u + (kp_r << 7) + dst_col, b_src);

                unsigned char* const sSt = sScaleRaw + stage * 256u;
                if constexpr (!TS4E) {
                    if (tid < 16u) {
                        const unsigned kh = tid >> 3;
                        const unsigned sc_c = tid & 7u;
                        if constexpr (GATEUP) {
                            const unsigned bm = sc_c >> 2;
                            const unsigned c4 = sc_c & 3u;
                            const unsigned char* const s_src = (bm ? sMeta->S1 : sMeta->S0) +
                                static_cast<unsigned long long>(k32 * 2u + kh) * N + m_cta_n + (c4 << 4);
                            v33_cp16_cg(sSt + (bm << 7) + (kh << 6) + (c4 << 4), s_src);
                        } else {
                            const unsigned char* const s_src = sMeta->S0 +
                                static_cast<unsigned long long>(k32 * 2u + kh) * N + m_cta_n + (sc_c << 4);
                            v33_cp16_cg(sSt + (kh << 7) + (sc_c << 4), s_src);
                        }
                    }
                } else {
                    // Warp-local async prefetch of TS4E nibbles into `dst_even + STRIDE`.
                    if constexpr (GATEUP) {
                        if (warp_id < 2u && lane < 4u) {
                            const unsigned char* const n_src =
                                (warp_id ? sMeta->S1 : sMeta->S0) +
                                static_cast<unsigned long long>(k32) * N + m_cta_n + (lane << 4);
                            v33_cp16_cg(sSt + (warp_id << 7) + 64u + (lane << 4), n_src);
                        }
                    } else if (warp_id == 0u && lane < 8u) {
                        const unsigned char* const n_src =
                            sMeta->S0 + static_cast<unsigned long long>(k32) * N + m_cta_n + (lane << 4);
                        v33_cp16_cg(sSt + 128u + (lane << 4), n_src);
                    }
                }
                v33_cp_commit();
            };

            auto finish_stage_mma = [&](unsigned stage, unsigned k32, unsigned m_sub, unsigned valid_r) {
                // Load 32x32 BF16 activations (L2-resident) -> clamped FP16 in XOR-swizzled sA[stage].
                const unsigned a_row = tid >> 2; // 0..31
                const unsigned a_ch = tid & 3u;  // 0..3 (8 elements = 16 B)
                uint4 raw = make_uint4(0u, 0u, 0u, 0u);
                if (a_row < valid_r) {
                    const unsigned src_row = GATEUP ? sRowIds[m_sub + a_row] : (sMeta->cta_m + m_sub + a_row);
                    raw = v33_ld_ca_v4(
                        X + static_cast<unsigned long long>(src_row) * K + k32 * 32u + (a_ch << 3));
                    raw.x = v33_bf162_to_f162_clamped(raw.x);
                    raw.y = v33_bf162_to_f162_clamped(raw.y);
                    raw.z = v33_bf162_to_f162_clamped(raw.z);
                    raw.w = v33_bf162_to_f162_clamped(raw.w);
                }
                const unsigned a_swz = (a_ch ^ (a_row & 3u)) << 4;
                *reinterpret_cast<uint4*>(sA_raw + stage * 2048u + (a_row << 6) + a_swz) = raw;

                v33_cp_wait<0>();
                if constexpr (TS4E) {
                    __syncwarp();
                    unsigned char* const sSt = sScaleRaw + stage * 256u;
                    const unsigned m_cta_n = sMeta->cta_n;
                    if constexpr (GATEUP) {
                        if (warp_id < 2u) {
                            v33_decode_ts4e_k32_from_smem_warp<true, N, K>(
                                warp_id ? sMeta->S1 : sMeta->S0,
                                sBase + (warp_id << 6),
                                sSt + (warp_id << 7),
                                k32, m_cta_n, lane);
                        }
                    } else if (warp_id == 0u) {
                        v33_decode_ts4e_k32_from_smem_warp<false, N, K>(
                            sMeta->S0, sBase, sSt, k32, m_cta_n, lane);
                    }
                }
                __syncthreads();
            };

            const unsigned m_step = (total_r <= 16u) ? 16u : V33_MMA_TILE_M;

            #pragma unroll 1
            for (unsigned m_sub = 0u; m_sub < total_r; m_sub += m_step) {
                const unsigned valid_r = min(total_r - m_sub, m_step);
                const bool has_m1 = (valid_r > 16u);
                const unsigned k32_0 = sMeta->ks_beg * 2u;
                const unsigned k32_end = sMeta->ks_end * 2u;

                float acc0[4][4];
                float acc1[4][4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        acc0[j][q] = 0.0f;
                        acc1[j][q] = 0.0f;
                    }
                }

                // Prologue: stage 0 async B + S/nibbles, then sA + TS4E decode.
                issue_b_and_s_mma(0u, k32_0);
                finish_stage_mma(0u, k32_0, m_sub, valid_r);

                unsigned cur_st = 0u;
                #pragma unroll 1
                for (unsigned k32 = k32_0; k32 < k32_end; ++k32, cur_st ^= 1u) {
                    const unsigned nxt_st = cur_st ^ 1u;
                    const bool has_next = (k32 + 1u < k32_end);
                    // Non-blocking cp.async for k32+1 runs 100% overlapped with k32 MMA compute!
                    if (has_next) {
                        issue_b_and_s_mma(nxt_st, k32 + 1u);
                    }

                    const unsigned char* const sB_st = sB_raw + cur_st * 2048u;
                    const unsigned char* const sS_st = sScaleRaw + cur_st * 256u;
                    const unsigned char* const sA_st = sA_raw + cur_st * 2048u;
                    const unsigned a_ld_row = lane & 15u;
                    const unsigned a_ld_ch_lo = lane >> 4;
                    const unsigned gid_l = lane >> 2;
                    const unsigned tig_l = lane & 3u;
                    const unsigned warp_bc = warp_id << 5;

                    #pragma unroll
                    for (unsigned kh = 0u; kh < 2u; ++kh) {
                        unsigned af0[4], af1[4];
                        const unsigned ch_kh = (kh << 1) | a_ld_ch_lo;
                        const unsigned addr0 = static_cast<unsigned>(__cvta_generic_to_shared(
                            sA_st + (a_ld_row << 6) + ((ch_kh ^ (a_ld_row & 3u)) << 4)));
                        v33_ldsm4(addr0, af0[0], af0[1], af0[2], af0[3]);
                        if (has_m1) {
                            const unsigned r1 = a_ld_row + 16u;
                            const unsigned addr1 = static_cast<unsigned>(__cvta_generic_to_shared(
                                sA_st + (r1 << 6) + ((ch_kh ^ (r1 & 3u)) << 4)));
                            v33_ldsm4(addr1, af1[0], af1[1], af1[2], af1[3]);
                        }

                        const unsigned kp0 = (kh << 3) + tig_l;
                        const unsigned kp1 = kp0 + 4u;
                        const unsigned swz0 = (kp0 & 3u) << 4;
                        const unsigned swz1 = (kp1 & 3u) << 4;

                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const unsigned col = warp_bc + static_cast<unsigned>(j) * 8u + gid_l;
                            const unsigned sc_off = GATEUP
                                ? (((warp_id >> 1) << 7) + (kh << 6) + (col & 63u))
                                : ((kh << 7) + col);
                            const __half2 sc_h2 = v33_e4m3_to_h2_bcast(sS_st[sc_off]);
                            const unsigned raw0 = sB_st[(kp0 << 7) + (col ^ swz0)];
                            const unsigned raw1 = sB_st[(kp1 << 7) + (col ^ swz1)];
                            const unsigned h0 = v33_e2m1x2_to_h2(raw0);
                            const unsigned h1 = v33_e2m1x2_to_h2(raw1);
                            const __half2 b0_h = __hmul2(*reinterpret_cast<const __half2*>(&h0), sc_h2);
                            const __half2 b1_h = __hmul2(*reinterpret_cast<const __half2*>(&h1), sc_h2);
                            const unsigned b0 = *reinterpret_cast<const unsigned*>(&b0_h);
                            const unsigned b1 = *reinterpret_cast<const unsigned*>(&b1_h);
                            v33_mma_f16(acc0[j], af0, b0, b1);
                            if (has_m1) {
                                v33_mma_f16(acc1[j], af1, b0, b1);
                            }
                        }
                    }

                    if (has_next) {
                        finish_stage_mma(nxt_st, k32 + 1u, m_sub, valid_r);
                    }
                }

                // Epilogue for this `m_sub` slice (valid_r rows <= 32).
                const unsigned gid = lane >> 2;
                const unsigned tig = lane & 3u;
                const unsigned base_m = sMeta->cta_m + m_sub;
                const unsigned m_cta_n = sMeta->cta_n;
                const unsigned m_sk_idx = sMeta->sk_idx;
                const float scale2 = (GATEUP && (warp_id >= 2u)) ? sMeta->s2b : sMeta->s2a;

                if (split_k > 1u) {
                    // Direct coalesced float2 stores to Split-K partials (0 SMEM, 0 barriers!).
                    const unsigned m_pairs = sMeta->pairs;
                    const unsigned local_col_base = GATEUP ? ((warp_id & 1u) * 32u) : (warp_id * 32u);
                    const unsigned slot_m = GATEUP ? (warp_id >> 1) : 0u;
                    auto store_m16_partials = [&](float (&acc)[4][4], unsigned m_off) {
                        const unsigned r0 = m_off + gid;
                        const unsigned r1 = r0 + 8u;
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const unsigned c = m_cta_n + local_col_base + static_cast<unsigned>(j) * 8u + tig * 2u;
                            if (r0 < valid_r) {
                                const unsigned long long prow = base_m + r0;
                                const unsigned long long off = GATEUP
                                    ? ((static_cast<unsigned long long>(m_sk_idx) * m_pairs + prow) * 1280ull +
                                       static_cast<unsigned long long>(slot_m) * 640ull + c)
                                    : ((static_cast<unsigned long long>(m_sk_idx) * m_pairs + prow) * 2560ull + c);
                                v33_st_wb_f2(partials + off, make_float2(acc[j][0] * scale2, acc[j][1] * scale2));
                            }
                            if (r1 < valid_r) {
                                const unsigned long long prow = base_m + r1;
                                const unsigned long long off = GATEUP
                                    ? ((static_cast<unsigned long long>(m_sk_idx) * m_pairs + prow) * 1280ull +
                                       static_cast<unsigned long long>(slot_m) * 640ull + c)
                                    : ((static_cast<unsigned long long>(m_sk_idx) * m_pairs + prow) * 2560ull + c);
                                v33_st_wb_f2(partials + off, make_float2(acc[j][2] * scale2, acc[j][3] * scale2));
                            }
                        }
                    };
                    store_m16_partials(acc0, 0u);
                    if (has_m1) store_m16_partials(acc1, 16u);
                } else if constexpr (GATEUP) {
                    // Reuse payload SMEM (8,192 B = 32 x 64 FP32) to pass `gate * s2a` from Warps 0..1 to Warps 2..3!
                    float (*const sGate)[64] = reinterpret_cast<float (*)[64]>(smem + L::OFF_PAYLOAD);
                    __syncthreads();
                    if (warp_id < 2u) {
                        const unsigned col_base = warp_id * 32u;
                        auto stage_gate = [&](float (&acc)[4][4], unsigned m_off) {
                            const unsigned r0 = m_off + gid, r1 = r0 + 8u;
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const unsigned c = col_base + static_cast<unsigned>(j) * 8u + tig * 2u;
                                *reinterpret_cast<float2*>(&sGate[r0][c]) =
                                    make_float2(acc[j][0] * scale2, acc[j][1] * scale2);
                                *reinterpret_cast<float2*>(&sGate[r1][c]) =
                                    make_float2(acc[j][2] * scale2, acc[j][3] * scale2);
                            }
                        };
                        stage_gate(acc0, 0u);
                        if (has_m1) stage_gate(acc1, 16u);
                    }
                    __syncthreads();
                    if (warp_id >= 2u) {
                        const unsigned col_base = (warp_id - 2u) * 32u;
                        auto finish_silu = [&](float (&acc)[4][4], unsigned m_off) {
                            const unsigned r0 = m_off + gid, r1 = r0 + 8u;
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const unsigned c = col_base + static_cast<unsigned>(j) * 8u + tig * 2u;
                                if (r0 < valid_r) {
                                    const float2 gv = *reinterpret_cast<const float2*>(&sGate[r0][c]);
                                    const float u0 = acc[j][0] * scale2, u1 = acc[j][1] * scale2;
                                    const float a0 = gv.x * (1.0f / (1.0f + __expf(-gv.x))) * u0;
                                    const float a1 = gv.y * (1.0f / (1.0f + __expf(-gv.y))) * u1;
                                    *reinterpret_cast<unsigned*>(
                                        out_bf16 + (base_m + r0) * 640ull + m_cta_n + c) = v33_pack_bf162(a0, a1);
                                }
                                if (r1 < valid_r) {
                                    const float2 gv = *reinterpret_cast<const float2*>(&sGate[r1][c]);
                                    const float u0 = acc[j][2] * scale2, u1 = acc[j][3] * scale2;
                                    const float a0 = gv.x * (1.0f / (1.0f + __expf(-gv.x))) * u0;
                                    const float a1 = gv.y * (1.0f / (1.0f + __expf(-gv.y))) * u1;
                                    *reinterpret_cast<unsigned*>(
                                        out_bf16 + (base_m + r1) * 640ull + m_cta_n + c) = v33_pack_bf162(a0, a1);
                                }
                            }
                        };
                        finish_silu(acc0, 0u);
                        if (has_m1) finish_silu(acc1, 16u);
                    }
                } else {
                    // DOWN SK=1 epilogue: 100% register-direct (0 SMEM, 0 barriers!)
                    const unsigned col = m_cta_n + warp_id * 32u;
                    const unsigned col8 = col + tig * 8u;
                    auto finish_down = [&](float (&acc)[4][4], unsigned m_off) {
                        const unsigned lr0 = m_off + gid, lr1 = lr0 + 8u;
                        if constexpr (EPI == 3) {
                            const bool keep_slot =
                                (static_cast<unsigned long long>(sMeta->pairs) * 2640ull) <= V33_KEEP_SLOT_BYTES;
                            {
                                unsigned u0[4];
                                #pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    u0[j] = v33_pack_bf162(acc[j][0] * scale2, acc[j][1] * scale2);
                                }
                                unsigned qq[2], ss;
                                v33_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], tig, qq, ss);
                                if (lr0 < valid_r) {
                                    const unsigned long long prow = base_m + lr0;
                                    if (keep_slot) v33_st_l2_evict_last_u2(slot_q + prow * 2560ull + col8, make_uint2(qq[0], qq[1]));
                                    else           v33_st_wb_u2(slot_q + prow * 2560ull + col8, make_uint2(qq[0], qq[1]));
                                    if (tig == 0u) slot_s[prow * 80ull + (col >> 5)] = static_cast<unsigned char>(ss);
                                }
                            }
                            {
                                unsigned u1[4];
                                #pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    u1[j] = v33_pack_bf162(acc[j][2] * scale2, acc[j][3] * scale2);
                                }
                                unsigned qq[2], ss;
                                v33_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], tig, qq, ss);
                                if (lr1 < valid_r) {
                                    const unsigned long long prow = base_m + lr1;
                                    if (keep_slot) v33_st_l2_evict_last_u2(slot_q + prow * 2560ull + col8, make_uint2(qq[0], qq[1]));
                                    else           v33_st_wb_u2(slot_q + prow * 2560ull + col8, make_uint2(qq[0], qq[1]));
                                    if (tig == 0u) slot_s[prow * 80ull + (col >> 5)] = static_cast<unsigned char>(ss);
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const unsigned c = col + static_cast<unsigned>(j) * 8u + tig * 2u;
                                if (lr0 < valid_r) {
                                    *reinterpret_cast<unsigned*>(out_bf16 + (base_m + lr0) * 2560ull + c) =
                                        v33_pack_bf162(acc[j][0] * scale2, acc[j][1] * scale2);
                                }
                                if (lr1 < valid_r) {
                                    *reinterpret_cast<unsigned*>(out_bf16 + (base_m + lr1) * 2560ull + c) =
                                        v33_pack_bf162(acc[j][2] * scale2, acc[j][3] * scale2);
                                }
                            }
                        }
                    };
                    finish_down(acc0, 0u);
                    if (has_m1) finish_down(acc1, 16u);
                }

                if (m_sub + m_step < total_r) __syncthreads();
            }
        }

        // Dynamic atomic work-stealing after the zero-contention first wave.
        __syncthreads();
        if (tid == 0u) {
            sMeta->task = gridDim.x + atomicAdd(sMeta->next_task_ptr, 1u);
        }
        __syncthreads();
    }

    // Signal PDL dependents ONLY when the immediate successor executes
    // `griddepcontrol.wait` (GATEUP -> reduce_silu/down; DOWN sk>1 -> down_reduce).
    // When DOWN sk==1 feeds `moe_unpermute_reduce_blend_mxfp8_bf16_w4` (no PDL
    // wait), let natural grid completion serialize cleanly.
    if (GATEUP || split_k > 1u) {
        v33_pdl_launch_dependents();
    }
}

} // namespace

// =========================================================================
// Public V33 Thin-M Split-K Entrypoints (Flat + Atlas-TS4E-128)
// =========================================================================

extern "C" __global__ __launch_bounds__(128, 6) void qwen4_moe_thin_sk_gateup_v33(
    const __nv_bfloat16* __restrict__ X,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ act_out,
    float* __restrict__ partials,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    unsigned int split_k,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out,
    unsigned int sg_rows
) {
    v33_thin_sk_impl<true, 0, false>(
        X, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals,
        up_packed_ptrs, up_scale_ptrs, up_scale2_vals,
        act_out, nullptr, nullptr, partials,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, split_k,
        shared_gate, ss_out, sg_rows);
}

extern "C" __global__ __launch_bounds__(128, 6) void qwen4_moe_thin_sk_gateup_v33_ts4e(
    const __nv_bfloat16* __restrict__ X,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ act_out,
    float* __restrict__ partials,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    unsigned int split_k,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out,
    unsigned int sg_rows
) {
    v33_thin_sk_impl<true, 0, true>(
        X, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals,
        up_packed_ptrs, up_scale_ptrs, up_scale2_vals,
        act_out, nullptr, nullptr, partials,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, split_k,
        shared_gate, ss_out, sg_rows);
}

extern "C" __global__ __launch_bounds__(128, 6) void qwen4_moe_thin_sk_down_bf16_v33(
    const __nv_bfloat16* __restrict__ act_in,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ partials,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    unsigned int split_k
) {
    v33_thin_sk_impl<false, 0, false>(
        act_in, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr,
        C, nullptr, nullptr, partials,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, split_k);
}

extern "C" __global__ __launch_bounds__(128, 6) void qwen4_moe_thin_sk_down_bf16_v33_ts4e(
    const __nv_bfloat16* __restrict__ act_in,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ partials,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    unsigned int split_k
) {
    v33_thin_sk_impl<false, 0, true>(
        act_in, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr,
        C, nullptr, nullptr, partials,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, split_k);
}

extern "C" __global__ __launch_bounds__(128, 6) void qwen4_moe_thin_sk_down_mxfp8_v33(
    const __nv_bfloat16* __restrict__ act_in,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    unsigned char* __restrict__ slot_q,
    unsigned char* __restrict__ slot_s,
    float* __restrict__ partials,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    unsigned int split_k
) {
    v33_thin_sk_impl<false, 3, false>(
        act_in, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr,
        nullptr, slot_q, slot_s, partials,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, split_k);
}

extern "C" __global__ __launch_bounds__(128, 6) void qwen4_moe_thin_sk_down_mxfp8_v33_ts4e(
    const __nv_bfloat16* __restrict__ act_in,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    unsigned char* __restrict__ slot_q,
    unsigned char* __restrict__ slot_s,
    float* __restrict__ partials,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K,
    unsigned int split_k
) {
    v33_thin_sk_impl<false, 3, true>(
        act_in, packed_ptrs, scale_ptrs, scale2_vals,
        nullptr, nullptr, nullptr,
        nullptr, slot_q, slot_s, partials,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K, split_k);
}

// =========================================================================
// Fused Split-K Partial Reduction Kernels (L2-resident, 0 SMEM)
// =========================================================================

extern "C" __global__ __launch_bounds__(32, 8) void qwen4_moe_thin_sk_reduce_silu_v33(
    const float* __restrict__ partials,
    __nv_bfloat16* __restrict__ act_out,
    unsigned int pairs,
    unsigned int split_k
) {
    const unsigned prow = blockIdx.x;
    const unsigned seg = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (prow >= pairs || seg >= 5u) return;

    const unsigned col = seg * 128u + lane * 4u;
    const unsigned long long slice_stride = static_cast<unsigned long long>(pairs) * 1280ull;
    const unsigned long long base_g = static_cast<unsigned long long>(prow) * 1280ull + col;
    const unsigned long long base_u = base_g + 640ull;

    v33_pdl_wait();

    float4 g = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 u = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (unsigned s = 0u; s < split_k; ++s) {
        const float4 pg = *reinterpret_cast<const float4*>(partials + s * slice_stride + base_g);
        const float4 pu = *reinterpret_cast<const float4*>(partials + s * slice_stride + base_u);
        g.x += pg.x; g.y += pg.y; g.z += pg.z; g.w += pg.w;
        u.x += pu.x; u.y += pu.y; u.z += pu.z; u.w += pu.w;
    }

    const float a0 = g.x * (1.0f / (1.0f + __expf(-g.x))) * u.x;
    const float a1 = g.y * (1.0f / (1.0f + __expf(-g.y))) * u.y;
    const float a2 = g.z * (1.0f / (1.0f + __expf(-g.z))) * u.z;
    const float a3 = g.w * (1.0f / (1.0f + __expf(-g.w))) * u.w;

    const uint2 packed = make_uint2(v33_pack_bf162(a0, a1), v33_pack_bf162(a2, a3));
    v33_st_wb_u2(act_out + static_cast<unsigned long long>(prow) * 640ull + col, packed);
    v33_pdl_launch_dependents();
}

extern "C" __global__ __launch_bounds__(128, 8) void qwen4_moe_thin_sk_down_reduce_bf16_v33(
    const float* __restrict__ partials,
    __nv_bfloat16* __restrict__ C,
    unsigned int pairs,
    unsigned int split_k
) {
    const unsigned prow = blockIdx.x;
    const unsigned seg = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (prow >= pairs || seg >= 20u) return;

    const unsigned col = seg * 128u + tid;
    const unsigned long long slice_stride = static_cast<unsigned long long>(pairs) * 2560ull;
    const unsigned long long base = static_cast<unsigned long long>(prow) * 2560ull + col;

    v33_pdl_wait();

    float sum = 0.0f;
    for (unsigned s = 0u; s < split_k; ++s) {
        sum += partials[s * slice_stride + base];
    }
    C[base] = __float2bfloat16(sum);
}

extern "C" __global__ __launch_bounds__(128, 8) void qwen4_moe_thin_sk_down_reduce_mxfp8_v33(
    const float* __restrict__ partials,
    unsigned char* __restrict__ slot_q,
    unsigned char* __restrict__ slot_s,
    unsigned int pairs,
    unsigned int split_k
) {
    const unsigned prow = blockIdx.x;
    const unsigned seg = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (prow >= pairs || seg >= 20u) return;

    const unsigned warp_id = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned col = seg * 128u + tid;
    const unsigned g_blk = seg * 4u + warp_id;
    const unsigned long long slice_stride = static_cast<unsigned long long>(pairs) * 2560ull;
    const unsigned long long base = static_cast<unsigned long long>(prow) * 2560ull + col;

    v33_pdl_wait();

    float sum = 0.0f;
    for (unsigned s = 0u; s < split_k; ++s) {
        sum += partials[s * slice_stride + base];
    }
    const float vb = __bfloat162float(__float2bfloat16(sum));
    float amax = fabsf(vb);
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 16));

    unsigned char qb = 0u;
    unsigned char sb = 0u;
    if (amax > 0.0f) {
        int e;
        const float m = frexpf(__fmul_rn(amax, 1.0f / 448.0f), &e);
        int k = (m == 0.5f) ? e - 1 : e;
        k = max(-127, min(127, k));
        const float inv = __int_as_float((127 - k) << 23);
        qb = v33_cvt_f32_e4m3(vb * inv);
        sb = static_cast<unsigned char>(k + 127);
    }
    const bool keep_slot =
        (static_cast<unsigned long long>(pairs) * 2640ull) <= V33_KEEP_SLOT_BYTES;
    if (keep_slot) {
        unsigned long long pol;
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
        asm volatile("st.global.L2::cache_hint.u8 [%0], %1, %2;\n"
                     :: "l"(slot_q + base), "r"(static_cast<unsigned>(qb)), "l"(pol)
                     : "memory");
    } else {
        slot_q[base] = qb;
    }
    if (lane == 0u) {
        slot_s[static_cast<unsigned long long>(prow) * 80ull + g_blk] = sb;
    }
}

extern "C" __global__ __launch_bounds__(256, 4) void qwen4_moe_thin_sk_shared_gate_v33(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out,
    unsigned int rows
) {
    const unsigned r = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (r >= rows) return;
    if (shared_gate == nullptr) {
        if (tid == 0u) ss_out[r] = 1.0f;
        return;
    }
    __shared__ float s_dot[8];
    const __nv_bfloat16* const x_row = input + static_cast<unsigned long long>(r) * 2560ull;
    float d = 0.0f;
    #pragma unroll
    for (unsigned i = 0u; i < 10u; ++i) {
        const unsigned c = tid + i * 256u;
        d += __bfloat162float(x_row[c]) * __bfloat162float(shared_gate[c]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        d += __shfl_down_sync(0xffffffffu, d, offset);
    }
    if ((tid & 31u) == 0u) s_dot[tid >> 5] = d;
    __syncthreads();
    if (tid == 0u) {
        float total = 0.0f;
        #pragma unroll
        for (unsigned w = 0u; w < 8u; ++w) total += s_dot[w];
        ss_out[r] = 1.0f / (1.0f + __expf(-total));
    }
}
