// SPDX-License-Identifier: AGPL-3.0-only
//
// Flux Engine — Native Blackwell SM120/SM121 Fused INT8 Block-32 BF16-Scale
// (8.50 bpw) LM-Head GEMM + Online Warp/Block Argmax & Weight Quantizer.
//
// Kernels:
//   1. `lm_head_int8_block32_gemm_sm120`:
//      Full BF16 logit GEMM (`M = 1..64`, optimal for `M = 1..32`, `K % 32 == 0`,
//      arbitrary `N`, 128 threads = 4 warps) using:
//      - `LDG.E.EF.128` (`L2::evict_first`) for `int8` weight vectors
//      - Coalesced `u32` `LDG.E.CONSTANT` scale-pair loads (2x BF16)
//      - `LDG.E.128.CONSTANT` (`ld.global.nc`) for BF16 activations
//      - 0-ULP lossless `int8 (-127..127) -> BF16` unpack into portable
//        `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` with per-block-32
//        FP32 scale factorization (zero BF16 dequant rounding loss)
//      - 2-stage register pipeline overlapping global loads with Tensor Core MMA
//      - Explicit inline PTX `cvt.rn.bf16.f32` on BF16 logit stores (0 atomicAdd)
//   1b. `lm_head_int8_block32_gemm_m64_sm120`:
//      Specialized 256-thread (8-warp: `2 M-warps x 4 N-warps`) single-CTA
//      `grid.y == 1` full BF16 logit GEMM for `M = 33..64` (accepting all
//      `M = 1..64` with 0-ULP bit-exact identity against `gemm_sm120`):
//      - 1x `W_i8` (`LDG.E.EF.128`, 1 vector/thread) + 1x `S_bf16` read per CTA
//      - Bank-conflict-free padded SMEM (`smem_A[64][72]`, `smem_W[64][80]`)
//      - Half the accumulator pressure (`acc[2][2][4]` = 16 regs/thread) and
//        half the inner MMA loop per warp vs 128-thread `gemm_sm120`
//      - Full 2-stage `A + W + S` register prefetch overlapping global/L2 loads
//        with zero-CVT `0x4300` magic-bias BF16x2 unpack + Tensor Core MMA.
//   2. `lm_head_int8_block32_argmax_stage1_sm120`:
//      Fused `M = 1..16`, `V_TILE = 256` INT8 Block-32 LM-head + online warp/block
//      argmax skipping the full `[M, V]` BF16 logit write and emitting only compact
//      `[M, ceil(V/256)]` `(val_f32, idx_u32)` partials (`<= 62 KiB` at M=8, V=248k),
//      with strict deterministic lowest-vocab-index tie-break and NaN exclusion.
//   3. `lm_head_int8_block32_argmax_stage2_sm120`:
//      1-warp-per-row deterministic final reduction over partial tiles -> `[M]` u32 IDs.
//   4. `quantize_bf16_to_int8_block32_sm120`:
//      Coalesced 1-warp-per-block-32 BF16 -> INT8 (`[-127, 127]`) + BF16 scale packer
//      with explicit inline PTX `cvt.rn.bf16.f32` and non-finite sanitization.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#define W8_M_TILE      16u
#define W8_N_TILE_GEMM 64u
#define W8_V_TILE_ARG  256u
#define W8_K_STEP      64u
#define W8_BLOCK_SIZE  32u
#define W8_MAX_M_GEMM        64u
#define W8_MAX_M_ARG         16u
#define W8_INVALID_ID        0xFFFFFFFFu
#define W8_M64_SMEM_A_STRIDE 72u
#define W8_M64_SMEM_W_STRIDE 80u

// Explicit inline PTX FP32 -> BF16 round-to-nearest-even.
__device__ __forceinline__ unsigned short w8_f32_to_bf16_rn(float x) {
    unsigned short bits;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(x));
    return bits;
}

__device__ __forceinline__ float w8_bf16_bits_to_f32(unsigned short bits) {
    return __uint_as_float(((unsigned int)bits) << 16);
}

// 0-ULP 256/256 bit-exact ZERO-CVT conversion of two packed signed int8
// weights in [-128, 127] (`u16_pair = (u8(q1) << 8) | u8(q0)`) into packed
// `bf16x2` (`low 16 bits = bf16(q0)`, `high 16 bits = bf16(q1)`).
//
// Uses the IEEE-754 BF16 2^7 = 128.0 (`0x4300`) magic-bias identity:
//   hi   = 0x4300 | (u & 0x7F) = 128.0 + (u & 0x7F)
//   bias = 0x4300 | (u & 0x80) = 128.0 (q >= 0) or 256.0 (q < 0)
//   bf16(q) == sub.bf16x2(hi, bias) — ZERO I2F, ZERO F2FP!
__device__ __forceinline__ unsigned int w8_u16_to_bf16x2_zero_cvt(unsigned int u16_pair) {
    unsigned int merged;
    asm("prmt.b32 %0, %1, 0x00000043, 0x4140;" : "=r"(merged) : "r"(u16_pair));
    const unsigned int hi = merged & 0xFF7FFF7Fu;
    const unsigned int bias = merged & 0xFF80FF80u;
    unsigned int packed;
    asm("sub.bf16x2 %0, %1, %2;" : "=r"(packed) : "r"(hi), "r"(bias));
    return packed;
}

__device__ __forceinline__ unsigned long long w8_l2_evict_first_policy() {
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    return pol;
}

// 128-bit streaming weight load -> SASS `LDG.E.EF.128` + L2::evict_first desc.
__device__ __forceinline__ uint4 w8_ldg_ef_128(
    const void* ptr,
    unsigned long long pol
) {
    uint4 v;
    asm volatile(
        "ld.global.cs.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(ptr), "l"(pol));
    return v;
}

// 32-bit L1-cached coalesced BF16 scale-pair load -> SASS `LDG.E.CONSTANT`.
__device__ __forceinline__ unsigned int w8_ldg_nc_32(const void* ptr) {
    unsigned int v;
    asm volatile(
        "ld.global.nc.u32 %0, [%1];"
        : "=r"(v)
        : "l"(ptr));
    return v;
}

// 16-bit L1-cached BF16 scale load for odd-block K % 64 == 32 tail -> SASS `LDG.E.U16.CONSTANT`.
__device__ __forceinline__ unsigned short w8_ldg_nc_16(const void* ptr) {
    unsigned short v;
    asm volatile(
        "ld.global.nc.u16 %0, [%1];"
        : "=h"(v)
        : "l"(ptr));
    return v;
}

// 128-bit read-only BF16 activation load -> SASS `LDG.E.128.CONSTANT`.
__device__ __forceinline__ uint4 w8_ldg_nc_128(const void* ptr) {
    uint4 v;
    asm volatile(
        "ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(ptr));
    return v;
}

// Deterministic strict lowest-vocab-index comparator with NaN exclusion.
__device__ __forceinline__ bool w8_better(
    float val,
    unsigned int idx,
    float best_val,
    unsigned int best_idx
) {
    if (idx == W8_INVALID_ID || isnan(val)) return false;
    if (best_idx == W8_INVALID_ID) return true;
    if (val > best_val) return true;
    if (val == best_val && idx < best_idx) return true;
    return false;
}

// =========================================================================
// 1. lm_head_int8_block32_gemm_sm120
//
// Single-CTA-per-N-tile full BF16 logit GEMM (`grid.y == 1` for ALL M <= 64!):
//   C[M, N] = A[M, K] @ dequant(W_i8[N, K], S_bf16[N, K/32])^T
// Reads each int8 weight & BF16 scale ONCE across all `m_blocks = ceil(M/16)`
// strips (`1..=4`), unpacking `int8 -> bf16x2` once per weight via 0-CVT
// `0x4300` magic-bias `sub.bf16x2` (`I2F = 0`) and guaranteeing 0-ULP
// `M=1 == M=8 == M=16 == M=32 == M=64` batch-invariance.
//
// Grid:  (ceil(N / 64), 1, 1)
// Block: (128, 1, 1) — 4 warps (each warp owns 16 N-cols x up to 64 M-rows)
// SMEM:  12,544 B (smem_A 8192B + smem_W 4096B + smem_S 256B)
// =========================================================================
extern "C" __global__
__launch_bounds__(128, 4)
void lm_head_int8_block32_gemm_sm120(
    const __nv_bfloat16*  __restrict__ A_bf16,   // [M, K] BF16
    const signed char*    __restrict__ W_i8,     // [N, K] int8 (-127..127)
    const unsigned short* __restrict__ S_bf16,   // [N, K/32] BF16 scales (raw u16 bits)
    __nv_bfloat16*        __restrict__ C_bf16,   // [M, N] BF16 logits
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    if (M == 0u || M > W8_MAX_M_GEMM || N == 0u || K < W8_BLOCK_SIZE || (K & 31u) != 0u) return;

    const unsigned int cta_n = blockIdx.x * W8_N_TILE_GEMM;
    if (cta_n >= N) return;

    const unsigned int m_blocks = (M + W8_M_TILE - 1u) >> 4; // 1..=4
    const int64_t K64 = (int64_t)K;
    const int64_t num_blocks = (int64_t)(K >> 5);
    const bool k_even_blocks = ((K & 63u) == 0u);
    const unsigned long long pol_ef = w8_l2_evict_first_policy();

    __shared__ alignas(16) unsigned short smem_A[W8_MAX_M_GEMM][W8_K_STEP];
    __shared__ alignas(16) signed char    smem_W[W8_N_TILE_GEMM][W8_K_STEP];
    __shared__ alignas(4)  unsigned short smem_S[W8_N_TILE_GEMM][2];

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31u;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3u;

    float acc[4][2][4];
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb) {
        acc[mb][0][0] = 0.0f; acc[mb][0][1] = 0.0f; acc[mb][0][2] = 0.0f; acc[mb][0][3] = 0.0f;
        acc[mb][1][0] = 0.0f; acc[mb][1][1] = 0.0f; acc[mb][1][2] = 0.0f; acc[mb][1][3] = 0.0f;
    }

    // Warp-uniform per-thread load coordinates:
    //   A_bf16: 128 threads x m_blocks uint4 (L1-cached LDG.E.128.CONSTANT)
    //   W_i8:   128 threads x 2 uint4        (streaming LDG.E.EF.128)
    //   S_bf16:  64 threads x 1 u32          (L1-cached LDG.E.CONSTANT)
    const unsigned int a_row_local = threadIdx.x >> 3;
    const unsigned int a_col = (threadIdx.x & 7u) << 3;

    const unsigned int w0_row = threadIdx.x >> 2;
    const unsigned int w0_col = (threadIdx.x & 3u) << 4;
    const unsigned int w0_gn = cta_n + w0_row;
    const bool w0_row_valid = (w0_gn < N);

    const unsigned int w1_idx = threadIdx.x + 128u;
    const unsigned int w1_row = w1_idx >> 2;
    const unsigned int w1_col = (w1_idx & 3u) << 4;
    const unsigned int w1_gn = cta_n + w1_row;
    const bool w1_row_valid = (w1_gn < N);

    const unsigned int s_row = threadIdx.x;
    const unsigned int s_gn = cta_n + s_row;
    const bool s_valid = (threadIdx.x < W8_N_TILE_GEMM) && (s_gn < N);

    #define W8_GEMM_LOAD_A(kb) do { \
        const bool a_col_valid = ((kb) + a_col < K); \
        _Pragma("unroll") \
        for (unsigned int mb = 0u; mb < 4u; ++mb) { \
            if (mb < m_blocks) { \
                const unsigned int gm = mb * W8_M_TILE + a_row_local; \
                uint4 av = make_uint4(0u, 0u, 0u, 0u); \
                if (gm < M && a_col_valid) { \
                    av = w8_ldg_nc_128(A_bf16 + (int64_t)gm * K64 + (int64_t)((kb) + a_col)); \
                } \
                *(uint4*)&smem_A[gm][a_col] = av; \
            } \
        } \
    } while (0)

    #define W8_GEMM_PREFETCH_WS(kb, r_w0, r_w1, r_s) do { \
        (r_w0) = make_uint4(0u, 0u, 0u, 0u); \
        if (w0_row_valid && ((kb) + w0_col < K)) { \
            (r_w0) = w8_ldg_ef_128(W_i8 + (int64_t)w0_gn * K64 + (int64_t)((kb) + w0_col), pol_ef); \
        } \
        (r_w1) = make_uint4(0u, 0u, 0u, 0u); \
        if (w1_row_valid && ((kb) + w1_col < K)) { \
            (r_w1) = w8_ldg_ef_128(W_i8 + (int64_t)w1_gn * K64 + (int64_t)((kb) + w1_col), pol_ef); \
        } \
        (r_s) = 0u; \
        if (s_valid) { \
            const int64_t b_idx = (int64_t)((kb) >> 5); \
            if (k_even_blocks) { \
                (r_s) = w8_ldg_nc_32(S_bf16 + (int64_t)s_gn * num_blocks + b_idx); \
            } else { \
                const unsigned short s0 = w8_ldg_nc_16(S_bf16 + (int64_t)s_gn * num_blocks + b_idx); \
                unsigned short s1 = 0u; \
                if (b_idx + 1 < num_blocks) { \
                    s1 = w8_ldg_nc_16(S_bf16 + (int64_t)s_gn * num_blocks + b_idx + 1); \
                } \
                (r_s) = (unsigned int)s0 | ((unsigned int)s1 << 16); \
            } \
        } \
    } while (0)

    #define W8_GEMM_COMMIT_WS(r_w0, r_w1, r_s) do { \
        *(uint4*)&smem_W[w0_row][w0_col] = (r_w0); \
        *(uint4*)&smem_W[w1_row][w1_col] = (r_w1); \
        if (threadIdx.x < W8_N_TILE_GEMM) { \
            *(unsigned int*)&smem_S[s_row][0] = (r_s); \
        } \
    } while (0)

    #define W8_GEMM_COMPUTE() do { \
        const unsigned int k_tid = tid * 2u; \
        _Pragma("unroll") \
        for (int blk = 0; blk < 2; ++blk) { \
            const unsigned int kb_off = (unsigned int)blk * 32u; \
            _Pragma("unroll") \
            for (int nt = 0; nt < 2; ++nt) { \
                const unsigned int n_base = warp_id * 16u + (unsigned int)nt * 8u; \
                const unsigned int nc = n_base + group_id; \
                /* Unpack int8 -> bf16x2 ONCE per weight and reuse across all m_blocks! */ \
                const unsigned int b0_k0 = w8_u16_to_bf16x2_zero_cvt( \
                    (unsigned int)*(const unsigned short*)&smem_W[nc][kb_off + k_tid]); \
                const unsigned int b1_k0 = w8_u16_to_bf16x2_zero_cvt( \
                    (unsigned int)*(const unsigned short*)&smem_W[nc][kb_off + 8u + k_tid]); \
                const unsigned int b0_k1 = w8_u16_to_bf16x2_zero_cvt( \
                    (unsigned int)*(const unsigned short*)&smem_W[nc][kb_off + 16u + k_tid]); \
                const unsigned int b1_k1 = w8_u16_to_bf16x2_zero_cvt( \
                    (unsigned int)*(const unsigned short*)&smem_W[nc][kb_off + 24u + k_tid]); \
                const float sc0 = w8_bf16_bits_to_f32(smem_S[n_base + tid * 2u][blk]); \
                const float sc1 = w8_bf16_bits_to_f32(smem_S[n_base + tid * 2u + 1u][blk]); \
                _Pragma("unroll") \
                for (unsigned int mb = 0u; mb < 4u; ++mb) { \
                    if (mb < m_blocks) { \
                        const unsigned int r0 = mb * W8_M_TILE + group_id; \
                        const unsigned int r1 = r0 + 8u; \
                        float bacc0 = 0.0f, bacc1 = 0.0f, bacc2 = 0.0f, bacc3 = 0.0f; \
                        { \
                            const unsigned int a0 = *(const unsigned int*)&smem_A[r0][kb_off + k_tid]; \
                            const unsigned int a1 = *(const unsigned int*)&smem_A[r1][kb_off + k_tid]; \
                            const unsigned int a2 = *(const unsigned int*)&smem_A[r0][kb_off + 8u + k_tid]; \
                            const unsigned int a3 = *(const unsigned int*)&smem_A[r1][kb_off + 8u + k_tid]; \
                            asm volatile( \
                                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
                                : "+f"(bacc0), "+f"(bacc1), "+f"(bacc2), "+f"(bacc3) \
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_k0), "r"(b1_k0)); \
                        } \
                        { \
                            const unsigned int a0 = *(const unsigned int*)&smem_A[r0][kb_off + 16u + k_tid]; \
                            const unsigned int a1 = *(const unsigned int*)&smem_A[r1][kb_off + 16u + k_tid]; \
                            const unsigned int a2 = *(const unsigned int*)&smem_A[r0][kb_off + 24u + k_tid]; \
                            const unsigned int a3 = *(const unsigned int*)&smem_A[r1][kb_off + 24u + k_tid]; \
                            asm volatile( \
                                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
                                : "+f"(bacc0), "+f"(bacc1), "+f"(bacc2), "+f"(bacc3) \
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_k1), "r"(b1_k1)); \
                        } \
                        acc[mb][nt][0] += bacc0 * sc0; \
                        acc[mb][nt][1] += bacc1 * sc1; \
                        acc[mb][nt][2] += bacc2 * sc0; \
                        acc[mb][nt][3] += bacc3 * sc1; \
                    } \
                } \
            } \
        } \
    } while (0)

    {
        uint4 cur_w0, cur_w1;
        unsigned int cur_s;
        W8_GEMM_LOAD_A(0u);
        W8_GEMM_PREFETCH_WS(0u, cur_w0, cur_w1, cur_s);
        W8_GEMM_COMMIT_WS(cur_w0, cur_w1, cur_s);
        __syncthreads();

        for (unsigned int k_next = W8_K_STEP; k_next < K; k_next += W8_K_STEP) {
            uint4 nxt_w0, nxt_w1;
            unsigned int nxt_s;
            W8_GEMM_PREFETCH_WS(k_next, nxt_w0, nxt_w1, nxt_s);
            W8_GEMM_COMPUTE();
            __syncthreads();
            W8_GEMM_LOAD_A(k_next);
            W8_GEMM_COMMIT_WS(nxt_w0, nxt_w1, nxt_s);
            __syncthreads();
        }
        W8_GEMM_COMPUTE();
    }

    #undef W8_GEMM_LOAD_A
    #undef W8_GEMM_PREFETCH_WS
    #undef W8_GEMM_COMMIT_WS
    #undef W8_GEMM_COMPUTE

    #pragma unroll
    for (unsigned int mb = 0u; mb < 4u; ++mb) {
        if (mb < m_blocks) {
            const unsigned int gr0 = mb * W8_M_TILE + group_id;
            const unsigned int gr1 = gr0 + 8u;
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const unsigned int gc0 = cta_n + warp_id * 16u + (unsigned int)nt * 8u + tid * 2u;
                const unsigned int gc1 = gc0 + 1u;
                if (gr0 < M && gc0 < N) {
                    *(unsigned short*)&C_bf16[(int64_t)gr0 * N + gc0] =
                        w8_f32_to_bf16_rn(acc[mb][nt][0]);
                }
                if (gr0 < M && gc1 < N) {
                    *(unsigned short*)&C_bf16[(int64_t)gr0 * N + gc1] =
                        w8_f32_to_bf16_rn(acc[mb][nt][1]);
                }
                if (gr1 < M && gc0 < N) {
                    *(unsigned short*)&C_bf16[(int64_t)gr1 * N + gc0] =
                        w8_f32_to_bf16_rn(acc[mb][nt][2]);
                }
                if (gr1 < M && gc1 < N) {
                    *(unsigned short*)&C_bf16[(int64_t)gr1 * N + gc1] =
                        w8_f32_to_bf16_rn(acc[mb][nt][3]);
                }
            }
        }
    }
}

// =========================================================================
// 1b. lm_head_int8_block32_gemm_m64_sm120
//
// Specialized 256-thread (8-warp: `2 M-warps x 4 N-warps`) single-CTA
// `grid.y == 1` full BF16 logit GEMM for `M = 33..64` (also valid and 0-ULP
// bit-identical to `lm_head_int8_block32_gemm_sm120` across all `M = 1..64`):
//   C[M, N] = A[M, K] @ dequant(W_i8[N, K], S_bf16[N, K/32])^T
//
// Key SM121 roofline optimizations vs 4-warp `gemm_sm120`:
// - 8 warps = `2 M-warps (32 rows/warp) x 4 N-warps (16 cols/warp)`:
//   each thread holds only `acc[2][2][4]` (16 accumulators vs 32) and runs
//   half the inner MMA loop per warp.
// - Bank-conflict-free padded SMEM:
//   `smem_A[64][72]` (144 B/row = 36 banks -> 0-way conflict on `LDS.32` + 16B `STS.128`)
//   `smem_W[64][80]` ( 80 B/row = 20 banks -> 0-way conflict on `LDS.U16` + 16B `STS.128`)
//   `smem_S[64][2]`  (  4 B/row = broadcast, 0-way conflict)
// - 256 threads cooperatively issue 1 `LDG.E.EF.128` per thread for `W_i8`
//   and 2 `LDG.E.128.CONSTANT` per thread for `A_bf16`, fully overlapped with
//   zero-CVT `0x4300` magic-bias BF16x2 unpack + `HMMA.16816` in a 2-stage
//   register pipeline (zero global loads between the two `__syncthreads()`).
//
// Grid:  (ceil(N / 64), 1, 1)
// Block: (256, 1, 1) — 8 warps (`2 M-warps x 4 N-warps`)
// SMEM:  14,592 B (`smem_A` 9,216 B + `smem_W` 5,120 B + `smem_S` 256 B)
// =========================================================================
extern "C" __global__
__launch_bounds__(256, 3)
void lm_head_int8_block32_gemm_m64_sm120(
    const __nv_bfloat16*  __restrict__ A_bf16,   // [M, K] BF16
    const signed char*    __restrict__ W_i8,     // [N, K] int8 (-127..127)
    const unsigned short* __restrict__ S_bf16,   // [N, K/32] BF16 scales (raw u16 bits)
    __nv_bfloat16*        __restrict__ C_bf16,   // [M, N] BF16 logits
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    if (M == 0u || M > W8_MAX_M_GEMM || N == 0u || K < W8_BLOCK_SIZE || (K & 31u) != 0u) return;

    const unsigned int cta_n = blockIdx.x * W8_N_TILE_GEMM;
    if (cta_n >= N) return;

    const int64_t K64 = (int64_t)K;
    const int64_t num_blocks = (int64_t)(K >> 5);
    const bool k_even_blocks = ((K & 63u) == 0u);
    const unsigned long long pol_ef = w8_l2_evict_first_policy();

    __shared__ alignas(16) unsigned short smem_A[W8_MAX_M_GEMM][W8_M64_SMEM_A_STRIDE];
    __shared__ alignas(16) signed char    smem_W[W8_N_TILE_GEMM][W8_M64_SMEM_W_STRIDE];
    __shared__ alignas(4)  unsigned short smem_S[W8_N_TILE_GEMM][2];

    const unsigned int warp_id  = threadIdx.x >> 5;   // 0..7
    const unsigned int m_warp   = warp_id >> 2;       // 0 (rows 0..31) or 1 (rows 32..63)
    const unsigned int n_warp   = warp_id & 3u;       // 0..3 (16 N-cols each)
    const unsigned int lane_id  = threadIdx.x & 31u;
    const unsigned int group_id = lane_id >> 2;       // 0..7
    const unsigned int tid      = lane_id & 3u;       // 0..3
    const unsigned int m_base   = m_warp * 32u;       // 0 or 32
    const bool mb0_active = (m_base < M);
    const bool mb1_active = (m_base + W8_M_TILE < M);

    float acc[2][2][4];
    #pragma unroll
    for (int mb = 0; mb < 2; ++mb) {
        acc[mb][0][0] = 0.0f; acc[mb][0][1] = 0.0f; acc[mb][0][2] = 0.0f; acc[mb][0][3] = 0.0f;
        acc[mb][1][0] = 0.0f; acc[mb][1][1] = 0.0f; acc[mb][1][2] = 0.0f; acc[mb][1][3] = 0.0f;
    }

    // Cooperative 256-thread load coordinates:
    //   A_bf16: 256 threads x 2 uint4 (rows 0..31 & 32..63, LDG.E.128.CONSTANT)
    //   W_i8:   256 threads x 1 uint4 (rows 0..63,          LDG.E.EF.128)
    //   S_bf16:  64 threads x 1 u32   (rows 0..63,          LDG.E.CONSTANT)
    const unsigned int a_row0 = threadIdx.x >> 3;        // 0..31
    const unsigned int a_row1 = a_row0 + 32u;            // 32..63
    const unsigned int a_col  = (threadIdx.x & 7u) << 3; // 0, 8, ..., 56
    const bool a0_row_valid = (a_row0 < M);
    const bool a1_row_valid = (a_row1 < M);

    const unsigned int w_row = threadIdx.x >> 2;         // 0..63
    const unsigned int w_col = (threadIdx.x & 3u) << 4;  // 0, 16, 32, 48
    const unsigned int w_gn  = cta_n + w_row;
    const bool w_row_valid = (w_gn < N);

    const unsigned int s_row = threadIdx.x;
    const unsigned int s_gn  = cta_n + s_row;
    const bool s_valid = (threadIdx.x < W8_N_TILE_GEMM) && (s_gn < N);

    #define W8_M64_PREFETCH(kb, r_a0, r_a1, r_w, r_s) do { \
        const bool a_col_valid = ((kb) + a_col < K); \
        (r_a0) = make_uint4(0u, 0u, 0u, 0u); \
        if (a0_row_valid && a_col_valid) { \
            (r_a0) = w8_ldg_nc_128(A_bf16 + (int64_t)a_row0 * K64 + (int64_t)((kb) + a_col)); \
        } \
        (r_a1) = make_uint4(0u, 0u, 0u, 0u); \
        if (a1_row_valid && a_col_valid) { \
            (r_a1) = w8_ldg_nc_128(A_bf16 + (int64_t)a_row1 * K64 + (int64_t)((kb) + a_col)); \
        } \
        (r_w) = make_uint4(0u, 0u, 0u, 0u); \
        if (w_row_valid && ((kb) + w_col < K)) { \
            (r_w) = w8_ldg_ef_128(W_i8 + (int64_t)w_gn * K64 + (int64_t)((kb) + w_col), pol_ef); \
        } \
        (r_s) = 0u; \
        if (s_valid) { \
            const int64_t b_idx = (int64_t)((kb) >> 5); \
            if (k_even_blocks) { \
                (r_s) = w8_ldg_nc_32(S_bf16 + (int64_t)s_gn * num_blocks + b_idx); \
            } else { \
                const unsigned short s0 = w8_ldg_nc_16(S_bf16 + (int64_t)s_gn * num_blocks + b_idx); \
                unsigned short s1 = 0u; \
                if (b_idx + 1 < num_blocks) { \
                    s1 = w8_ldg_nc_16(S_bf16 + (int64_t)s_gn * num_blocks + b_idx + 1); \
                } \
                (r_s) = (unsigned int)s0 | ((unsigned int)s1 << 16); \
            } \
        } \
    } while (0)

    #define W8_M64_COMMIT(r_a0, r_a1, r_w, r_s) do { \
        *(uint4*)&smem_A[a_row0][a_col] = (r_a0); \
        *(uint4*)&smem_A[a_row1][a_col] = (r_a1); \
        *(uint4*)&smem_W[w_row][w_col]  = (r_w); \
        if (threadIdx.x < W8_N_TILE_GEMM) { \
            *(unsigned int*)&smem_S[s_row][0] = (r_s); \
        } \
    } while (0)

    #define W8_M64_COMPUTE() do { \
        if (mb0_active) { \
            const unsigned int k_tid = tid * 2u; \
            const unsigned int n0_base = n_warp * 16u; \
            const unsigned int n1_base = n0_base + 8u; \
            const unsigned int nc0 = n0_base + group_id; \
            const unsigned int nc1 = n1_base + group_id; \
            _Pragma("unroll") \
            for (int blk = 0; blk < 2; ++blk) { \
                const unsigned int kb_off = (unsigned int)blk * 32u; \
                /* Unpack int8 -> bf16x2 ONCE per weight and reuse across mb=0 and mb=1! */ \
                const unsigned int b0_n0_k0 = w8_u16_to_bf16x2_zero_cvt( \
                    (unsigned int)*(const unsigned short*)&smem_W[nc0][kb_off + k_tid]); \
                const unsigned int b1_n0_k0 = w8_u16_to_bf16x2_zero_cvt( \
                    (unsigned int)*(const unsigned short*)&smem_W[nc0][kb_off + 8u + k_tid]); \
                const unsigned int b0_n0_k1 = w8_u16_to_bf16x2_zero_cvt( \
                    (unsigned int)*(const unsigned short*)&smem_W[nc0][kb_off + 16u + k_tid]); \
                const unsigned int b1_n0_k1 = w8_u16_to_bf16x2_zero_cvt( \
                    (unsigned int)*(const unsigned short*)&smem_W[nc0][kb_off + 24u + k_tid]); \
                const unsigned int b0_n1_k0 = w8_u16_to_bf16x2_zero_cvt( \
                    (unsigned int)*(const unsigned short*)&smem_W[nc1][kb_off + k_tid]); \
                const unsigned int b1_n1_k0 = w8_u16_to_bf16x2_zero_cvt( \
                    (unsigned int)*(const unsigned short*)&smem_W[nc1][kb_off + 8u + k_tid]); \
                const unsigned int b0_n1_k1 = w8_u16_to_bf16x2_zero_cvt( \
                    (unsigned int)*(const unsigned short*)&smem_W[nc1][kb_off + 16u + k_tid]); \
                const unsigned int b1_n1_k1 = w8_u16_to_bf16x2_zero_cvt( \
                    (unsigned int)*(const unsigned short*)&smem_W[nc1][kb_off + 24u + k_tid]); \
                const float sc0_n0 = w8_bf16_bits_to_f32(smem_S[n0_base + tid * 2u][blk]); \
                const float sc1_n0 = w8_bf16_bits_to_f32(smem_S[n0_base + tid * 2u + 1u][blk]); \
                const float sc0_n1 = w8_bf16_bits_to_f32(smem_S[n1_base + tid * 2u][blk]); \
                const float sc1_n1 = w8_bf16_bits_to_f32(smem_S[n1_base + tid * 2u + 1u][blk]); \
                /* mb = 0: load A ONCE per kh and reuse across BOTH nt=0 and nt=1 (dual-acc ILP). */ \
                { \
                    const unsigned int r0 = m_base + group_id; \
                    const unsigned int r1 = r0 + 8u; \
                    float ba0_0 = 0.0f, ba0_1 = 0.0f, ba0_2 = 0.0f, ba0_3 = 0.0f; \
                    float ba1_0 = 0.0f, ba1_1 = 0.0f, ba1_2 = 0.0f, ba1_3 = 0.0f; \
                    { \
                        const unsigned int a0 = *(const unsigned int*)&smem_A[r0][kb_off + k_tid]; \
                        const unsigned int a1 = *(const unsigned int*)&smem_A[r1][kb_off + k_tid]; \
                        const unsigned int a2 = *(const unsigned int*)&smem_A[r0][kb_off + 8u + k_tid]; \
                        const unsigned int a3 = *(const unsigned int*)&smem_A[r1][kb_off + 8u + k_tid]; \
                        asm volatile( \
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
                            : "+f"(ba0_0), "+f"(ba0_1), "+f"(ba0_2), "+f"(ba0_3) \
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_n0_k0), "r"(b1_n0_k0)); \
                        asm volatile( \
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
                            : "+f"(ba1_0), "+f"(ba1_1), "+f"(ba1_2), "+f"(ba1_3) \
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_n1_k0), "r"(b1_n1_k0)); \
                    } \
                    { \
                        const unsigned int a0 = *(const unsigned int*)&smem_A[r0][kb_off + 16u + k_tid]; \
                        const unsigned int a1 = *(const unsigned int*)&smem_A[r1][kb_off + 16u + k_tid]; \
                        const unsigned int a2 = *(const unsigned int*)&smem_A[r0][kb_off + 24u + k_tid]; \
                        const unsigned int a3 = *(const unsigned int*)&smem_A[r1][kb_off + 24u + k_tid]; \
                        asm volatile( \
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
                            : "+f"(ba0_0), "+f"(ba0_1), "+f"(ba0_2), "+f"(ba0_3) \
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_n0_k1), "r"(b1_n0_k1)); \
                        asm volatile( \
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
                            : "+f"(ba1_0), "+f"(ba1_1), "+f"(ba1_2), "+f"(ba1_3) \
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_n1_k1), "r"(b1_n1_k1)); \
                    } \
                    acc[0][0][0] += ba0_0 * sc0_n0; \
                    acc[0][0][1] += ba0_1 * sc1_n0; \
                    acc[0][0][2] += ba0_2 * sc0_n0; \
                    acc[0][0][3] += ba0_3 * sc1_n0; \
                    acc[0][1][0] += ba1_0 * sc0_n1; \
                    acc[0][1][1] += ba1_1 * sc1_n1; \
                    acc[0][1][2] += ba1_2 * sc0_n1; \
                    acc[0][1][3] += ba1_3 * sc1_n1; \
                } \
                /* mb = 1 (active when m_base + 16 < M) */ \
                if (mb1_active) { \
                    const unsigned int r0 = m_base + W8_M_TILE + group_id; \
                    const unsigned int r1 = r0 + 8u; \
                    float ba0_0 = 0.0f, ba0_1 = 0.0f, ba0_2 = 0.0f, ba0_3 = 0.0f; \
                    float ba1_0 = 0.0f, ba1_1 = 0.0f, ba1_2 = 0.0f, ba1_3 = 0.0f; \
                    { \
                        const unsigned int a0 = *(const unsigned int*)&smem_A[r0][kb_off + k_tid]; \
                        const unsigned int a1 = *(const unsigned int*)&smem_A[r1][kb_off + k_tid]; \
                        const unsigned int a2 = *(const unsigned int*)&smem_A[r0][kb_off + 8u + k_tid]; \
                        const unsigned int a3 = *(const unsigned int*)&smem_A[r1][kb_off + 8u + k_tid]; \
                        asm volatile( \
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
                            : "+f"(ba0_0), "+f"(ba0_1), "+f"(ba0_2), "+f"(ba0_3) \
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_n0_k0), "r"(b1_n0_k0)); \
                        asm volatile( \
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
                            : "+f"(ba1_0), "+f"(ba1_1), "+f"(ba1_2), "+f"(ba1_3) \
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_n1_k0), "r"(b1_n1_k0)); \
                    } \
                    { \
                        const unsigned int a0 = *(const unsigned int*)&smem_A[r0][kb_off + 16u + k_tid]; \
                        const unsigned int a1 = *(const unsigned int*)&smem_A[r1][kb_off + 16u + k_tid]; \
                        const unsigned int a2 = *(const unsigned int*)&smem_A[r0][kb_off + 24u + k_tid]; \
                        const unsigned int a3 = *(const unsigned int*)&smem_A[r1][kb_off + 24u + k_tid]; \
                        asm volatile( \
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
                            : "+f"(ba0_0), "+f"(ba0_1), "+f"(ba0_2), "+f"(ba0_3) \
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_n0_k1), "r"(b1_n0_k1)); \
                        asm volatile( \
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
                            : "+f"(ba1_0), "+f"(ba1_1), "+f"(ba1_2), "+f"(ba1_3) \
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_n1_k1), "r"(b1_n1_k1)); \
                    } \
                    acc[1][0][0] += ba0_0 * sc0_n0; \
                    acc[1][0][1] += ba0_1 * sc1_n0; \
                    acc[1][0][2] += ba0_2 * sc0_n0; \
                    acc[1][0][3] += ba0_3 * sc1_n0; \
                    acc[1][1][0] += ba1_0 * sc0_n1; \
                    acc[1][1][1] += ba1_1 * sc1_n1; \
                    acc[1][1][2] += ba1_2 * sc0_n1; \
                    acc[1][1][3] += ba1_3 * sc1_n1; \
                } \
            } \
        } \
    } while (0)

    {
        uint4 cur_a0, cur_a1, cur_w;
        unsigned int cur_s;
        W8_M64_PREFETCH(0u, cur_a0, cur_a1, cur_w, cur_s);
        W8_M64_COMMIT(cur_a0, cur_a1, cur_w, cur_s);
        __syncthreads();

        for (unsigned int k_next = W8_K_STEP; k_next < K; k_next += W8_K_STEP) {
            uint4 nxt_a0, nxt_a1, nxt_w;
            unsigned int nxt_s;
            W8_M64_PREFETCH(k_next, nxt_a0, nxt_a1, nxt_w, nxt_s);
            W8_M64_COMPUTE();
            __syncthreads();
            W8_M64_COMMIT(nxt_a0, nxt_a1, nxt_w, nxt_s);
            __syncthreads();
        }
        W8_M64_COMPUTE();
    }

    #undef W8_M64_PREFETCH
    #undef W8_M64_COMMIT
    #undef W8_M64_COMPUTE

    if (mb0_active) {
        #pragma unroll
        for (unsigned int mb = 0u; mb < 2u; ++mb) {
            if (mb == 0u || mb1_active) {
                const unsigned int gr0 = m_base + mb * W8_M_TILE + group_id;
                const unsigned int gr1 = gr0 + 8u;
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const unsigned int gc0 = cta_n + n_warp * 16u + (unsigned int)nt * 8u + tid * 2u;
                    const unsigned int gc1 = gc0 + 1u;
                    if (gr0 < M && gc0 < N) {
                        *(unsigned short*)&C_bf16[(int64_t)gr0 * (int64_t)N + (int64_t)gc0] =
                            w8_f32_to_bf16_rn(acc[mb][nt][0]);
                    }
                    if (gr0 < M && gc1 < N) {
                        *(unsigned short*)&C_bf16[(int64_t)gr0 * (int64_t)N + (int64_t)gc1] =
                            w8_f32_to_bf16_rn(acc[mb][nt][1]);
                    }
                    if (gr1 < M && gc0 < N) {
                        *(unsigned short*)&C_bf16[(int64_t)gr1 * (int64_t)N + (int64_t)gc0] =
                            w8_f32_to_bf16_rn(acc[mb][nt][2]);
                    }
                    if (gr1 < M && gc1 < N) {
                        *(unsigned short*)&C_bf16[(int64_t)gr1 * (int64_t)N + (int64_t)gc1] =
                            w8_f32_to_bf16_rn(acc[mb][nt][3]);
                    }
                }
            }
        }
    }
}

// =========================================================================
// 2. lm_head_int8_block32_argmax_stage1_sm120
//
// Fused INT8 Block-32 LM-head + online warp/block argmax (`M = 1..16`).
// Skips the full `[M, V]` BF16 logit write and writes only compact
// `[M, ceil(V/256)]` `(val_f32, idx_u32)` partials (`<= 62 KiB` at M=8, V=248k).
//
// Grid:  (ceil(V / 256), 1, 1)
// Block: (128, 1, 1) — 4 warps x 4 N-subtiles of 64 = 256 vocab cols/CTA
// SMEM:  6,912 B
// =========================================================================
extern "C" __global__
__launch_bounds__(128, 2)
void lm_head_int8_block32_argmax_stage1_sm120(
    const __nv_bfloat16*  __restrict__ A_bf16,       // [M, K] BF16
    const signed char*    __restrict__ W_i8,         // [V, K] int8 (-127..127)
    const unsigned short* __restrict__ S_bf16,       // [V, K/32] BF16 scales (raw u16 bits)
    float*                __restrict__ partial_vals, // [M, num_v_tiles] FP32
    unsigned int*         __restrict__ partial_ids,  // [M, num_v_tiles] U32
    unsigned int M,
    unsigned int V,
    unsigned int K
) {
    if (M == 0u || M > W8_MAX_M_ARG || V == 0u || K < W8_BLOCK_SIZE || (K & 31u) != 0u) return;

    const unsigned int cta_v = blockIdx.x * W8_V_TILE_ARG;
    const unsigned int num_v_tiles = (V + W8_V_TILE_ARG - 1u) / W8_V_TILE_ARG;
    if (cta_v >= V) return;

    const int64_t K64 = (int64_t)K;
    const int64_t num_blocks = (int64_t)(K >> 5);
    const bool k_even_blocks = ((K & 63u) == 0u);
    const unsigned long long pol_ef = w8_l2_evict_first_policy();

    __shared__ alignas(16) unsigned short smem_A[W8_M_TILE][W8_K_STEP];
    __shared__ alignas(16) signed char    smem_W[W8_N_TILE_GEMM][W8_K_STEP];
    __shared__ alignas(4)  unsigned short smem_S[W8_N_TILE_GEMM][2];
    __shared__ float                      smem_red_val[W8_M_TILE][4];
    __shared__ unsigned int               smem_red_idx[W8_M_TILE][4];

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31u;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3u;

    float acc[8][4];
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        acc[t][0] = 0.0f; acc[t][1] = 0.0f; acc[t][2] = 0.0f; acc[t][3] = 0.0f;
    }

    const unsigned int a_row = threadIdx.x >> 3;
    const unsigned int a_col = (threadIdx.x & 7u) << 3;
    const bool a_row_valid = (a_row < M);

    const unsigned int w0_row = threadIdx.x >> 2;
    const unsigned int w0_col = (threadIdx.x & 3u) << 4;
    const unsigned int w1_idx = threadIdx.x + 128u;
    const unsigned int w1_row = w1_idx >> 2;
    const unsigned int w1_col = (w1_idx & 3u) << 4;
    const unsigned int s_row = threadIdx.x;

    #define W8_ARG_PREFETCH_W(kb, sub_v, r_w0, r_w1, r_s) do { \
        const unsigned int gn0 = (sub_v) + w0_row; \
        (r_w0) = make_uint4(0u, 0u, 0u, 0u); \
        if (gn0 < V && ((kb) + w0_col < K)) { \
            (r_w0) = w8_ldg_ef_128(W_i8 + (int64_t)gn0 * K64 + (int64_t)((kb) + w0_col), pol_ef); \
        } \
        const unsigned int gn1 = (sub_v) + w1_row; \
        (r_w1) = make_uint4(0u, 0u, 0u, 0u); \
        if (gn1 < V && ((kb) + w1_col < K)) { \
            (r_w1) = w8_ldg_ef_128(W_i8 + (int64_t)gn1 * K64 + (int64_t)((kb) + w1_col), pol_ef); \
        } \
        (r_s) = 0u; \
        if (threadIdx.x < W8_N_TILE_GEMM) { \
            const unsigned int gns = (sub_v) + s_row; \
            if (gns < V) { \
                const int64_t b_idx = (int64_t)((kb) >> 5); \
                if (k_even_blocks) { \
                    (r_s) = w8_ldg_nc_32(S_bf16 + (int64_t)gns * num_blocks + b_idx); \
                } else { \
                    const unsigned short s0 = w8_ldg_nc_16(S_bf16 + (int64_t)gns * num_blocks + b_idx); \
                    unsigned short s1 = 0u; \
                    if (b_idx + 1 < num_blocks) { \
                        s1 = w8_ldg_nc_16(S_bf16 + (int64_t)gns * num_blocks + b_idx + 1); \
                    } \
                    (r_s) = (unsigned int)s0 | ((unsigned int)s1 << 16); \
                } \
            } \
        } \
    } while (0)

    #define W8_ARG_COMMIT_W(r_w0, r_w1, r_s) do { \
        *(uint4*)&smem_W[w0_row][w0_col] = (r_w0); \
        *(uint4*)&smem_W[w1_row][w1_col] = (r_w1); \
        if (threadIdx.x < W8_N_TILE_GEMM) { \
            *(unsigned int*)&smem_S[s_row][0] = (r_s); \
        } \
    } while (0)

    #define W8_ARG_COMPUTE_SUB(sub_idx) do { \
        const unsigned int r0 = group_id; \
        const unsigned int r1 = group_id + 8u; \
        const unsigned int k_tid = tid * 2u; \
        _Pragma("unroll") \
        for (int blk = 0; blk < 2; ++blk) { \
            const unsigned int kb_off = (unsigned int)blk * 32u; \
            float bacc[2][4]; \
            bacc[0][0] = 0.0f; bacc[0][1] = 0.0f; bacc[0][2] = 0.0f; bacc[0][3] = 0.0f; \
            bacc[1][0] = 0.0f; bacc[1][1] = 0.0f; bacc[1][2] = 0.0f; bacc[1][3] = 0.0f; \
            _Pragma("unroll") \
            for (int kh = 0; kh < 2; ++kh) { \
                const unsigned int kh_off = kb_off + (unsigned int)kh * 16u; \
                const unsigned int a0 = *(const unsigned int*)&smem_A[r0][kh_off + k_tid]; \
                const unsigned int a1 = *(const unsigned int*)&smem_A[r1][kh_off + k_tid]; \
                const unsigned int a2 = *(const unsigned int*)&smem_A[r0][kh_off + 8u + k_tid]; \
                const unsigned int a3 = *(const unsigned int*)&smem_A[r1][kh_off + 8u + k_tid]; \
                _Pragma("unroll") \
                for (int nt = 0; nt < 2; ++nt) { \
                    const unsigned int nc = warp_id * 16u + (unsigned int)nt * 8u + group_id; \
                    const unsigned int p0 = (unsigned int)*(const unsigned short*)&smem_W[nc][kh_off + k_tid]; \
                    const unsigned int p1 = (unsigned int)*(const unsigned short*)&smem_W[nc][kh_off + 8u + k_tid]; \
                    const unsigned int b0 = w8_u16_to_bf16x2_zero_cvt(p0); \
                    const unsigned int b1 = w8_u16_to_bf16x2_zero_cvt(p1); \
                    asm volatile( \
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
                        : "+f"(bacc[nt][0]), "+f"(bacc[nt][1]), "+f"(bacc[nt][2]), "+f"(bacc[nt][3]) \
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)); \
                } \
            } \
            _Pragma("unroll") \
            for (int nt = 0; nt < 2; ++nt) { \
                const int acc_i = (sub_idx) * 2 + nt; \
                const unsigned int n_base = warp_id * 16u + (unsigned int)nt * 8u; \
                const float sc0 = w8_bf16_bits_to_f32(smem_S[n_base + tid * 2u][blk]); \
                const float sc1 = w8_bf16_bits_to_f32(smem_S[n_base + tid * 2u + 1u][blk]); \
                acc[acc_i][0] += bacc[nt][0] * sc0; \
                acc[acc_i][1] += bacc[nt][1] * sc1; \
                acc[acc_i][2] += bacc[nt][2] * sc0; \
                acc[acc_i][3] += bacc[nt][3] * sc1; \
            } \
        } \
    } while (0)

    for (unsigned int kb = 0u; kb < K; kb += W8_K_STEP) {
        // Load A tile once per 256 vocab rows (LDG.E.128.CONSTANT).
        uint4 a_vec = make_uint4(0u, 0u, 0u, 0u);
        if (a_row_valid && (kb + a_col < K)) {
            a_vec = w8_ldg_nc_128(A_bf16 + (int64_t)a_row * K64 + (int64_t)(kb + a_col));
        }
        *(uint4*)&smem_A[a_row][a_col] = a_vec;

        uint4 cur_w0, cur_w1;
        unsigned int cur_s;
        W8_ARG_PREFETCH_W(kb, cta_v, cur_w0, cur_w1, cur_s);
        W8_ARG_COMMIT_W(cur_w0, cur_w1, cur_s);
        __syncthreads();

        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            uint4 nxt_w0, nxt_w1;
            unsigned int nxt_s = 0u;
            if (sub < 3) {
                W8_ARG_PREFETCH_W(kb, cta_v + (unsigned int)(sub + 1) * 64u, nxt_w0, nxt_w1, nxt_s);
            }
            W8_ARG_COMPUTE_SUB(sub);
            __syncthreads();
            if (sub < 3) {
                W8_ARG_COMMIT_W(nxt_w0, nxt_w1, nxt_s);
                __syncthreads();
            }
        }
    }

    #undef W8_ARG_PREFETCH_W
    #undef W8_ARG_COMMIT_W
    #undef W8_ARG_COMPUTE_SUB

    // Thread-local online argmax reduction over this lane's 16 vocab columns
    // for rows r0 = group_id and r1 = group_id + 8.
    float best_v0 = -INFINITY;
    unsigned int best_i0 = W8_INVALID_ID;
    float best_v1 = -INFINITY;
    unsigned int best_i1 = W8_INVALID_ID;

    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int acc_i = sub * 2 + nt;
            const unsigned int c0 =
                cta_v + (unsigned int)sub * 64u + warp_id * 16u + (unsigned int)nt * 8u + tid * 2u;
            const unsigned int c1 = c0 + 1u;
            if (c0 < V) {
                if (w8_better(acc[acc_i][0], c0, best_v0, best_i0)) {
                    best_v0 = acc[acc_i][0];
                    best_i0 = c0;
                }
                if (w8_better(acc[acc_i][2], c0, best_v1, best_i1)) {
                    best_v1 = acc[acc_i][2];
                    best_i1 = c0;
                }
            }
            if (c1 < V) {
                if (w8_better(acc[acc_i][1], c1, best_v0, best_i0)) {
                    best_v0 = acc[acc_i][1];
                    best_i0 = c1;
                }
                if (w8_better(acc[acc_i][3], c1, best_v1, best_i1)) {
                    best_v1 = acc[acc_i][3];
                    best_i1 = c1;
                }
            }
        }
    }

    // 4-lane warp-quad reduction (lanes sharing group_id differ only in tid bits 0..1).
    #pragma unroll
    for (int mask = 1; mask <= 2; mask <<= 1) {
        const float ov0 = __shfl_xor_sync(0xFFFFFFFFu, best_v0, mask);
        const unsigned int oi0 = __shfl_xor_sync(0xFFFFFFFFu, best_i0, mask);
        if (w8_better(ov0, oi0, best_v0, best_i0)) {
            best_v0 = ov0;
            best_i0 = oi0;
        }

        const float ov1 = __shfl_xor_sync(0xFFFFFFFFu, best_v1, mask);
        const unsigned int oi1 = __shfl_xor_sync(0xFFFFFFFFu, best_i1, mask);
        if (w8_better(ov1, oi1, best_v1, best_i1)) {
            best_v1 = ov1;
            best_i1 = oi1;
        }
    }

    if (tid == 0u) {
        smem_red_val[group_id][warp_id] = best_v0;
        smem_red_idx[group_id][warp_id] = best_i0;
        smem_red_val[group_id + 8u][warp_id] = best_v1;
        smem_red_idx[group_id + 8u][warp_id] = best_i1;
    }
    __syncthreads();

    if (threadIdx.x < M) {
        const unsigned int r = threadIdx.x;
        float final_v = -INFINITY;
        unsigned int final_i = W8_INVALID_ID;
        #pragma unroll
        for (unsigned int w = 0u; w < 4u; ++w) {
            const float cv = smem_red_val[r][w];
            const unsigned int ci = smem_red_idx[r][w];
            if (w8_better(cv, ci, final_v, final_i)) {
                final_v = cv;
                final_i = ci;
            }
        }
        const int64_t out_off = (int64_t)r * (int64_t)num_v_tiles + (int64_t)blockIdx.x;
        partial_vals[out_off] = final_v;
        partial_ids[out_off] = final_i;
    }
}

// =========================================================================
// 3. lm_head_int8_block32_argmax_stage2_sm120
//
// Final 1-warp-per-row deterministic reduction over stage-1 partial tiles.
// Grid:  (M, 1, 1)
// Block: (32, 1, 1)
// =========================================================================
extern "C" __global__
__launch_bounds__(32, 4)
void lm_head_int8_block32_argmax_stage2_sm120(
    const float*        __restrict__ partial_vals, // [M, num_v_tiles] FP32
    const unsigned int* __restrict__ partial_ids,  // [M, num_v_tiles] U32
    unsigned int*       __restrict__ out_tokens,   // [M] U32
    unsigned int M,
    unsigned int num_v_tiles
) {
    const unsigned int r = blockIdx.x;
    if (r >= M || num_v_tiles == 0u) return;

    const unsigned int lane = threadIdx.x & 31u;
    const int64_t row_off = (int64_t)r * (int64_t)num_v_tiles;

    float best_v = -INFINITY;
    unsigned int best_i = W8_INVALID_ID;
    for (unsigned int t = lane; t < num_v_tiles; t += 32u) {
        const float v = partial_vals[row_off + (int64_t)t];
        const unsigned int id = partial_ids[row_off + (int64_t)t];
        if (w8_better(v, id, best_v, best_i)) {
            best_v = v;
            best_i = id;
        }
    }

    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, best_v, mask);
        const unsigned int oi = __shfl_xor_sync(0xFFFFFFFFu, best_i, mask);
        if (w8_better(ov, oi, best_v, best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    if (lane == 0u) {
        out_tokens[r] = (best_i == W8_INVALID_ID) ? 0u : best_i;
    }
}

// =========================================================================
// 4. quantize_bf16_to_int8_block32_sm120
//
// Coalesced 1-warp-per-block-32 BF16 -> INT8 (`[-127, 127]`) + BF16 scale
// weight packer matching `quantize_bf16_weight_to_int8_block32` on CPU.
// Grid:  (N, 1, 1)
// Block: (256, 1, 1) — 8 warps/CTA
// =========================================================================
extern "C" __global__
__launch_bounds__(256, 4)
void quantize_bf16_to_int8_block32_sm120(
    const __nv_bfloat16* __restrict__ src_bf16,    // [N, K] BF16
    signed char*         __restrict__ dst_i8,      // [N, K] int8 (-127..127)
    unsigned short*      __restrict__ dst_scales,  // [N, K/32] BF16 scale bits
    unsigned int N,
    unsigned int K
) {
    const unsigned int row = blockIdx.x;
    if (row >= N || K < W8_BLOCK_SIZE || (K & 31u) != 0u) return;

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int num_blocks = K >> 5;
    const int64_t row_off = (int64_t)row * (int64_t)K;
    const int64_t scale_row_off = (int64_t)row * (int64_t)num_blocks;

    for (unsigned int b = warp_id; b < num_blocks; b += 8u) {
        const int64_t elem_idx = row_off + (int64_t)b * 32 + (int64_t)lane;
        float val = __bfloat162float(src_bf16[elem_idx]);
        if (!isfinite(val)) val = 0.0f;

        float amax = fabsf(val);
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFFu, amax, mask);
            if (other > amax) amax = other;
        }

        const float raw_scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
        const unsigned short scale_bits = w8_f32_to_bf16_rn(raw_scale);
        const float scale_rt = w8_bf16_bits_to_f32(scale_bits);

        if (lane == 0u) {
            dst_scales[scale_row_off + (int64_t)b] = scale_bits;
        }

        int q = 0;
        if (scale_rt > 0.0f) {
            float scaled = rintf(val / scale_rt);
            if (scaled > 127.0f) scaled = 127.0f;
            if (scaled < -127.0f) scaled = -127.0f;
            q = (int)scaled;
        }
        dst_i8[elem_idx] = (signed char)q;
    }
}
