// SPDX-License-Identifier: AGPL-3.0-only
//
// Flux Engine — 48-SM Roofline W4A4 Decode GEMM (`split_k=1..=8`, default 3 on GB10)
// with Deterministic Fixed-Order Reduction & `W=1 == W=8 == W=16` Invariance.
//
// Kernels:
//   1. `w4a4_gemm_m16_sk3_sm120`:
//      Native Blackwell SM120/SM121 Register-Direct block-scaled FP4 Tensor Core
//      decode GEMM (`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.
//      m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`) for `M <= 16` (`W=1`, `W=8`,
//      `W=16` all fit in ONE `m16` tile height with N_TILE=64 -> 240 CTAs =
//      5.0 CTAs/SM at N=5120, split_k=3 on 48-SM GB10):
//      - ZERO B-SMEM staging (`0 B` SMEM) & ZERO per-K-step `__syncthreads()`!
//        Every warp streams independently straight from global memory into
//        registers via `K_SUPER = 128` (2x `K=64` tiles) coalesced 64B quads +
//        exact 2-stage 4x4 intra-quad `SHFL.BFLY` register transpose (plus a
//        clean `K=64` odd-tile tail).
//      - 3-stage register-direct B pipeline (`2` `K_SUPER=128` tiles = `2,048 B`
//        per warp = `40.96 KB/SM` at 5 CTAs/SM in flight simultaneously).
//      - Weights (`B_packed`) streamed via `LDG.E.EF.128` (`L2::evict_first`,
//        `.cs` L1-bypass so weights never evict L1 scales/activations).
//      - Weight scales (`B_scale`) loaded via `LDG.E.CONSTANT` with
//        `L2::evict_first` (`ld.global.nc.L2::cache_hint.u32`): L1-cached across
//        4 consecutive `K_SUPER=128` steps (eliminates 4x-8x 32B DRAM sector
//        re-fetch amplification while keeping L2 clean).
//      - Activations (`A_packed`, `A_scale`) loaded via `LDG.E.128.CONSTANT` /
//        `LDG.E.CONSTANT` (`ld.global.nc`, L1/L2 resident; rows 8..15 skipped
//        uniformly when `M <= 8`).
//      - Branchless 3-op SWAR E2M1 `-0.0` (0x8) -> `+0.0` (0x0) canonicalization
//      - Deterministic K-slice partition independent of `M` (0 `atomicAdd`)
//      - Explicit inline PTX `cvt.rn.bf16.f32` in epilogue (`split_k == 1`)
//      - Exact scalar FP32 fallback on `< sm_120` (e.g. `sm_103a`)
//   2. `reduce_w4a4_sk3_bf16_sm120`:
//      Fixed ascending-order FP32 split-K workspace reduction (`s = 0..split_k-1`)
//      followed by `scale2_ab` multiply and explicit inline PTX `cvt.rn.bf16.f32`.
//      Guarantees bit-exact `W=1 == W=8 == W=16` and `B=1 == B=4/8` invariance.

#include <cuda_bf16.h>
#include <stdint.h>

#define W4SK_M_TILE     16u
#define W4SK_N_TILE     64u
#define W4SK_K_STEP     64u
#define W4SK_K_SUPER    128u
#define W4SK_GROUP_SIZE 16u
#define W4SK_MAX_SPLITK 8u

#if __CUDA_ARCH__ >= 1200
#define W4SK_MIN_CTAS_PER_SM 6
#else
#define W4SK_MIN_CTAS_PER_SM 4
#endif

// 3-op branchless SWAR E2M1 `-0.0` (0x8) -> `+0.0` (0x0) canonicalization
// across all 8 packed 4-bit nibbles with zero cross-nibble carry.
__device__ __forceinline__ unsigned int w4sk_canon_e2m1_u32(unsigned int w) {
    const unsigned int m = w & 0x77777777u;
    return w & ((m + 0x77777777u) | 0x77777777u);
}

__device__ __forceinline__ uint4 w4sk_canon_e2m1_u4(uint4 v) {
    return make_uint4(
        w4sk_canon_e2m1_u32(v.x),
        w4sk_canon_e2m1_u32(v.y),
        w4sk_canon_e2m1_u32(v.z),
        w4sk_canon_e2m1_u32(v.w));
}

// Explicit inline PTX FP32 -> BF16 round-to-nearest-even (prevents FMA drift).
__device__ __forceinline__ unsigned short w4sk_f32_to_bf16_rn(float x) {
    unsigned short bits;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(x));
    return bits;
}

__device__ __forceinline__ unsigned long long w4sk_l2_evict_first_policy() {
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    return pol;
}

// 128-bit streaming weight load with cache-streaming + L2::evict_first
// -> SASS `LDG.E.EF.128` with L2::evict_first policy descriptor.
__device__ __forceinline__ uint4 w4sk_ldg_ef_128(
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

// 32-bit L1-cached read-only weight-scale load with L2::evict_first policy
// -> SASS `LDG.E.CONSTANT` with `desc[UR]` L2::evict_first policy descriptor.
// Caches the 32B scale sector in L1 across 4 consecutive K_SUPER=128 steps
// (eliminating 4x-8x DRAM sector re-fetch amplification) without polluting L2.
__device__ __forceinline__ unsigned int w4sk_ldg_nc_ef_32(
    const void* ptr,
    unsigned long long pol
) {
    unsigned int v;
    asm volatile(
        "ld.global.nc.L2::cache_hint.u32 %0, [%1], %2;"
        : "=r"(v)
        : "l"(ptr), "l"(pol));
    return v;
}

// 128-bit read-only activation load -> SASS `LDG.E.128.CONSTANT`.
__device__ __forceinline__ uint4 w4sk_ldg_nc_128(const void* ptr) {
    uint4 v;
    asm volatile(
        "ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(ptr));
    return v;
}

// 32-bit read-only activation-scale load -> SASS `LDG.E.CONSTANT`.
__device__ __forceinline__ unsigned int w4sk_ldg_nc_32(const void* ptr) {
    unsigned int v;
    asm volatile(
        "ld.global.nc.u32 %0, [%1];"
        : "=r"(v)
        : "l"(ptr));
    return v;
}

#if __CUDA_ARCH__ >= 1200
// Exact 2-stage 4x4 intra-quad register transpose across lanes tid = lane_id & 3.
//
// Each 4-lane quad (group_id = lane_id >> 2, tid = 0..3) loads 64 contiguous
// bytes (K_SUPER = 128 E2M1 elements) of one matrix row via 4 x LDG.128:
//   lane t holds row t = (M[t][0], M[t][1], M[t][2], M[t][3]) = (v.x, v.y, v.z, v.w).
// This 4x SHFL.BFLY butterfly transposes the 4x4 matrix directly in registers
// (0 SMEM, 0 BAR.SYNC) so lane t receives column t = (M[0][t], M[1][t], M[2][t], M[3][t]):
//   c0 = K[  0.. 32) word t  (b0 / a0 for K-step 0)
//   c1 = K[ 32.. 64) word t  (b1 / a2 for K-step 0)
//   c2 = K[ 64.. 96) word t  (b0 / a0 for K-step 1)
//   c3 = K[ 96..128) word t  (b1 / a2 for K-step 1)
// Also serves the K=64 odd-tile tail identically when lanes tid>=2 supply zero
// (producing c0 = K[0..32) word t, c1 = K[32..64) word t across all tid=0..3).
__device__ __forceinline__ void w4sk_quad_transpose_4x4(
    uint4 v,
    bool p1,
    bool p2,
    unsigned int& c0,
    unsigned int& c1,
    unsigned int& c2,
    unsigned int& c3
) {
    const unsigned int e0 = p1 ? v.x : v.y;
    const unsigned int e1 = p1 ? v.z : v.w;
    const unsigned int k0 = p1 ? v.y : v.x;
    const unsigned int k1 = p1 ? v.w : v.z;
    const unsigned int r0 = __shfl_xor_sync(0xFFFFFFFFu, e0, 1);
    const unsigned int r1 = __shfl_xor_sync(0xFFFFFFFFu, e1, 1);
    const unsigned int t00 = p1 ? r0 : k0;
    const unsigned int t10 = p1 ? k0 : r0;
    const unsigned int t01 = p1 ? r1 : k1;
    const unsigned int t11 = p1 ? k1 : r1;

    const unsigned int s0 = p2 ? t00 : t01;
    const unsigned int s1 = p2 ? t10 : t11;
    const unsigned int q0 = __shfl_xor_sync(0xFFFFFFFFu, s0, 2);
    const unsigned int q1 = __shfl_xor_sync(0xFFFFFFFFu, s1, 2);
    c0 = p2 ? q0 : t00;
    c1 = p2 ? q1 : t10;
    c2 = p2 ? t01 : q0;
    c3 = p2 ? t11 : q1;
}

__device__ __forceinline__ void w4sk_mma_m16n8k64(
    float (&d)[4],
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    unsigned int sfa, unsigned int sfb
) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]),
          "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0),
          "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0));
}

struct W4SkBStage {
    uint4 b0;
    uint4 b1;
    unsigned int bs;
};
#else
__device__ __constant__ float W4SK_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float w4sk_ue4m3_to_f32(unsigned char byte) {
    const unsigned int exp = (byte >> 3) & 0xFu;
    const unsigned int man = byte & 0x7u;
    if (exp == 0u) return (float)man * 0.001953125f;
    if (exp == 15u && man == 7u) return 0.0f;
    return __uint_as_float(((exp + 120u) << 23) | (man << 20));
}
#endif

// =========================================================================
// 1. w4a4_gemm_m16_sk3_sm120
//
// Grid:  (ceil(N / 64), 1, split_k)
// Block: (128, 1, 1) — 4 independent warps (each owns 16 N-cols x M<=16 rows)
// SMEM:  0 B on SM120+ (ZERO B-SMEM staging, ZERO __syncthreads()!)
// =========================================================================
extern "C" __global__
__launch_bounds__(128, W4SK_MIN_CTAS_PER_SM)
void w4a4_gemm_m16_sk3_sm120(
    const unsigned char* __restrict__ A_packed,       // [M, K/2] E2M1
    const unsigned char* __restrict__ A_scale,        // [M, K/16] UE4M3
    const unsigned char* __restrict__ B_packed,       // [N, K/2] E2M1
    const unsigned char* __restrict__ B_scale,        // [N, K/16] UE4M3
    float*               __restrict__ workspace_f32,  // [split_k, M, N] FP32 (when split_k > 1)
    __nv_bfloat16*       __restrict__ C_bf16,         // [M, N] BF16 (when split_k == 1)
    const float scale2_ab,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int split_k
) {
    if (M == 0u || M > W4SK_M_TILE || N == 0u || K == 0u || (K & 63u) != 0u) return;
    if (split_k == 0u || split_k > W4SK_MAX_SPLITK) return;

    const unsigned int cta_n = blockIdx.x * W4SK_N_TILE;
    const unsigned int sk = blockIdx.z;
    if (cta_n >= N || sk >= split_k) return;

    const unsigned int num_k_tiles = K >> 6;
    const unsigned int tile_begin = (sk * num_k_tiles) / split_k;
    const unsigned int tile_end = ((sk + 1u) * num_k_tiles) / split_k;
    const unsigned int k_begin = tile_begin << 6;

    const int64_t half_K = (int64_t)(K >> 1);
    const int64_t num_groups = (int64_t)(K >> 4);
    const unsigned long long pol_ef = w4sk_l2_evict_first_policy();

#if __CUDA_ARCH__ >= 1200
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31u;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3u;
    const unsigned int q0 = group_id << 2;
    const bool p1 = (tid & 1u) != 0u;
    const bool p2 = (tid & 2u) != 0u;

    float acc[2][4];
    acc[0][0] = 0.0f; acc[0][1] = 0.0f; acc[0][2] = 0.0f; acc[0][3] = 0.0f;
    acc[1][0] = 0.0f; acc[1][1] = 0.0f; acc[1][2] = 0.0f; acc[1][3] = 0.0f;

    const unsigned int slice_tiles = tile_end - tile_begin;
    const unsigned int num_super = slice_tiles >> 1;
    const bool has_tail = (slice_tiles & 1u) != 0u;
    const bool has_hi_rows = (M > 8u);

    // Precompute warp-independent direct global pointers (ZERO SMEM, ZERO BAR.SYNC):
    //   - B_packed: each 4-lane quad (group_id, tid=0..3) streams 64 contiguous
    //     bytes (2 x 32B sectors = K_SUPER=128) per N-subtile via LDG.E.EF.128.
    //   - B_scale:  tid={0,1,2,3} load (n0 k0, n0 k1, n1 k0, n1 k1) via ONE
    //     L1-cached L2::evict_first u32 load (100% lane utilization, 0 wasted
    //     DRAM sector re-fetches). Masking (& 0x7F7F7F7Fu) is deferred to
    //     consumption so ZERO ALU stalls on the freshly issued LDG.
    //   - A_packed: each 4-lane quad loads 64B of row fr0=group_id (and fr1=
    //     group_id+8 when M>8) via L1/L2-resident LDG.E.128.CONSTANT.
    //   - A_scale:  tid={0,1,2,3} load (fr0 k0, fr1 k0, fr0 k1, fr1 k1) via ONE
    //     LDG.E.CONSTANT load + 1x SHFL.BFLY(2).
    const unsigned int warp_n = cta_n + warp_id * 16u;
    const unsigned int b0_gn = warp_n + group_id;
    const unsigned int b1_gn = b0_gn + 8u;
    const bool b0_valid = (b0_gn < N);
    const bool b1_valid = (b1_gn < N);

    const unsigned int bs_gn = warp_n + ((tid >> 1) << 3) + group_id;
    const bool bs_valid = (bs_gn < N);

    const unsigned int fr0 = group_id;
    const unsigned int fr1 = group_id + 8u;
    const bool a0_valid = (fr0 < M);
    const bool a1_valid = (fr1 < M);

    const unsigned int as_row = group_id + ((tid & 1u) << 3);
    const bool as_valid = (as_row < M);

    const int64_t k0_byte = (int64_t)(k_begin >> 1) + (int64_t)(tid << 4);
    const int64_t g0_base = (int64_t)(k_begin >> 4);
    const int64_t row8_stride = half_K << 3;

    // Since b1_gn = b0_gn + 8 and fr1 = fr0 + 8, b1_valid => b0_valid and
    // a1_valid => a0_valid. Sharing b0_ptr + row8_stride and a0_ptr + row8_stride
    // (row8_stride lives in uniform register UR) frees 4 vector registers for
    // the 3-stage pipeline scheduler.
    const unsigned char* b0_ptr = b0_valid ? (B_packed + (int64_t)b0_gn * half_K + k0_byte) : B_packed;
    const unsigned char* bs_ptr = bs_valid ? (B_scale + (int64_t)bs_gn * num_groups + g0_base + (int64_t)((tid & 1u) << 2)) : B_scale;

    const unsigned char* a0_ptr = a0_valid ? (A_packed + (int64_t)fr0 * half_K + k0_byte) : A_packed;
    const unsigned char* as_ptr = as_valid ? (A_scale + (int64_t)as_row * num_groups + g0_base + (int64_t)((tid & 2u) << 1)) : A_scale;

    #define W4SK_LOAD_B_SUPER(st) do { \
        (st).b0 = b0_valid ? w4sk_ldg_ef_128(b0_ptr, pol_ef) : make_uint4(0u, 0u, 0u, 0u); \
        (st).b1 = b1_valid ? w4sk_ldg_ef_128(b0_ptr + row8_stride, pol_ef) : make_uint4(0u, 0u, 0u, 0u); \
        (st).bs = bs_valid ? w4sk_ldg_nc_ef_32(bs_ptr, pol_ef) : 0u; \
        b0_ptr += 64; \
        bs_ptr += 8; \
    } while (0)

    #define W4SK_LOAD_B_TAIL(st) do { \
        (st).b0 = (b0_valid && !p2) ? w4sk_ldg_ef_128(b0_ptr, pol_ef) : make_uint4(0u, 0u, 0u, 0u); \
        (st).b1 = (b1_valid && !p2) ? w4sk_ldg_ef_128(b0_ptr + row8_stride, pol_ef) : make_uint4(0u, 0u, 0u, 0u); \
        (st).bs = (bs_valid && !p1) ? w4sk_ldg_nc_ef_32(bs_ptr, pol_ef) : 0u; \
    } while (0)

    #define W4SK_COMPUTE_SUPER(st) do { \
        const uint4 ra0 = a0_valid ? w4sk_ldg_nc_128(a0_ptr) : make_uint4(0u, 0u, 0u, 0u); \
        const unsigned int ras_raw = as_valid ? w4sk_ldg_nc_32(as_ptr) : 0u; \
        uint4 ra1 = make_uint4(0u, 0u, 0u, 0u); \
        if (has_hi_rows && a1_valid) { \
            ra1 = w4sk_ldg_nc_128(a0_ptr + row8_stride); \
        } \
        a0_ptr += 64; \
        as_ptr += 8; \
        /* Hide L1 latency of A behind SWAR + quad-transpose of B[nt=0] */ \
        unsigned int b0_k0, b1_k0, b0_k1, b1_k1; \
        w4sk_quad_transpose_4x4(w4sk_canon_e2m1_u4((st).b0), p1, p2, b0_k0, b1_k0, b0_k1, b1_k1); \
        const unsigned int bs_clean = (st).bs & 0x7F7F7F7Fu; \
        const unsigned int sfb0_k0 = __shfl_sync(0xFFFFFFFFu, bs_clean, q0 + 0u); \
        const unsigned int sfb0_k1 = __shfl_sync(0xFFFFFFFFu, bs_clean, q0 + 1u); \
        unsigned int a0_k0, a2_k0, a0_k1, a2_k1; \
        w4sk_quad_transpose_4x4(w4sk_canon_e2m1_u4(ra0), p1, p2, a0_k0, a2_k0, a0_k1, a2_k1); \
        unsigned int a1_k0 = 0u, a3_k0 = 0u, a1_k1 = 0u, a3_k1 = 0u; \
        if (has_hi_rows) { \
            w4sk_quad_transpose_4x4(w4sk_canon_e2m1_u4(ra1), p1, p2, a1_k0, a3_k0, a1_k1, a3_k1); \
        } \
        const unsigned int ras = ras_raw & 0x7F7F7F7Fu; \
        const unsigned int as_peer = __shfl_xor_sync(0xFFFFFFFFu, ras, 2); \
        const unsigned int sfa_k0 = p2 ? as_peer : ras; \
        const unsigned int sfa_k1 = p2 ? ras : as_peer; \
        w4sk_mma_m16n8k64(acc[0], a0_k0, a1_k0, a2_k0, a3_k0, b0_k0, b1_k0, sfa_k0, sfb0_k0); \
        w4sk_mma_m16n8k64(acc[0], a0_k1, a1_k1, a2_k1, a3_k1, b0_k1, b1_k1, sfa_k1, sfb0_k1); \
        w4sk_quad_transpose_4x4(w4sk_canon_e2m1_u4((st).b1), p1, p2, b0_k0, b1_k0, b0_k1, b1_k1); \
        const unsigned int sfb1_k0 = __shfl_sync(0xFFFFFFFFu, bs_clean, q0 + 2u); \
        const unsigned int sfb1_k1 = __shfl_sync(0xFFFFFFFFu, bs_clean, q0 + 3u); \
        w4sk_mma_m16n8k64(acc[1], a0_k0, a1_k0, a2_k0, a3_k0, b0_k0, b1_k0, sfa_k0, sfb1_k0); \
        w4sk_mma_m16n8k64(acc[1], a0_k1, a1_k1, a2_k1, a3_k1, b0_k1, b1_k1, sfa_k1, sfb1_k1); \
    } while (0)

    #define W4SK_COMPUTE_TAIL(st) do { \
        const uint4 ra0 = (a0_valid && !p2) ? w4sk_ldg_nc_128(a0_ptr) : make_uint4(0u, 0u, 0u, 0u); \
        const unsigned int ras_raw = (as_valid && !p2) ? w4sk_ldg_nc_32(as_ptr) : 0u; \
        uint4 ra1 = make_uint4(0u, 0u, 0u, 0u); \
        if (has_hi_rows && a1_valid && !p2) { \
            ra1 = w4sk_ldg_nc_128(a0_ptr + row8_stride); \
        } \
        unsigned int b0_k0, b1_k0, u_b2, u_b3; \
        w4sk_quad_transpose_4x4(w4sk_canon_e2m1_u4((st).b0), p1, p2, b0_k0, b1_k0, u_b2, u_b3); \
        const unsigned int bs_clean = (st).bs & 0x7F7F7F7Fu; \
        const unsigned int sfb0 = __shfl_sync(0xFFFFFFFFu, bs_clean, q0 + 0u); \
        unsigned int a0_k0, a2_k0, u_a2, u_a3; \
        w4sk_quad_transpose_4x4(w4sk_canon_e2m1_u4(ra0), p1, p2, a0_k0, a2_k0, u_a2, u_a3); \
        unsigned int a1_k0 = 0u, a3_k0 = 0u; \
        if (has_hi_rows) { \
            w4sk_quad_transpose_4x4(w4sk_canon_e2m1_u4(ra1), p1, p2, a1_k0, a3_k0, u_a2, u_a3); \
        } \
        const unsigned int ras = ras_raw & 0x7F7F7F7Fu; \
        const unsigned int as_peer = __shfl_xor_sync(0xFFFFFFFFu, ras, 2); \
        const unsigned int sfa = p2 ? as_peer : ras; \
        w4sk_mma_m16n8k64(acc[0], a0_k0, a1_k0, a2_k0, a3_k0, b0_k0, b1_k0, sfa, sfb0); \
        w4sk_quad_transpose_4x4(w4sk_canon_e2m1_u4((st).b1), p1, p2, b0_k0, b1_k0, u_b2, u_b3); \
        const unsigned int sfb1 = __shfl_sync(0xFFFFFFFFu, bs_clean, q0 + 2u); \
        w4sk_mma_m16n8k64(acc[1], a0_k0, a1_k0, a2_k0, a3_k0, b0_k0, b1_k0, sfa, sfb1); \
    } while (0)

    // Static 3-stage register ring (s0, s1, s2): keeps 2 full K_SUPER=128
    // weight tiles (2,048 B/warp = 40.96 KB/SM at 5 CTAs/SM) concurrently in
    // flight with ZERO register-shift MOVs and ZERO scoreboard stalls between
    // in-flight stages, followed by a compact 0..2-iteration cleanup loop.
    if (num_super >= 2u) {
        W4SkBStage s0, s1, s2;
        W4SK_LOAD_B_SUPER(s0);
        W4SK_LOAD_B_SUPER(s1);
        unsigned int it = 2u;
        for (; it + 3u <= num_super; it += 3u) {
            W4SK_LOAD_B_SUPER(s2);
            W4SK_COMPUTE_SUPER(s0);
            W4SK_LOAD_B_SUPER(s0);
            W4SK_COMPUTE_SUPER(s1);
            W4SK_LOAD_B_SUPER(s1);
            W4SK_COMPUTE_SUPER(s2);
        }
        for (; it < num_super; ++it) {
            W4SK_LOAD_B_SUPER(s2);
            W4SK_COMPUTE_SUPER(s0);
            s0 = s1;
            s1 = s2;
        }
        if (has_tail) {
            W4SK_LOAD_B_TAIL(s2);
            W4SK_COMPUTE_SUPER(s0);
            W4SK_COMPUTE_SUPER(s1);
            W4SK_COMPUTE_TAIL(s2);
        } else {
            W4SK_COMPUTE_SUPER(s0);
            W4SK_COMPUTE_SUPER(s1);
        }
    } else if (num_super == 1u) {
        W4SkBStage s0;
        W4SK_LOAD_B_SUPER(s0);
        if (has_tail) {
            W4SkBStage stail;
            W4SK_LOAD_B_TAIL(stail);
            W4SK_COMPUTE_SUPER(s0);
            W4SK_COMPUTE_TAIL(stail);
        } else {
            W4SK_COMPUTE_SUPER(s0);
        }
    } else if (has_tail) {
        W4SkBStage stail;
        W4SK_LOAD_B_TAIL(stail);
        W4SK_COMPUTE_TAIL(stail);
    }

    #undef W4SK_LOAD_B_SUPER
    #undef W4SK_LOAD_B_TAIL
    #undef W4SK_COMPUTE_SUPER
    #undef W4SK_COMPUTE_TAIL

    const unsigned int r0 = group_id;
    const unsigned int r1 = group_id + 8u;
    const int64_t ws_slice_base = (int64_t)sk * (int64_t)M * (int64_t)N;

    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const unsigned int c0 = cta_n + warp_id * 16u + (unsigned int)nt * 8u + tid * 2u;
        const unsigned int c1 = c0 + 1u;
        if (split_k == 1u) {
            if (r0 < M && c0 < N) {
                *(unsigned short*)&C_bf16[(int64_t)r0 * N + c0] =
                    w4sk_f32_to_bf16_rn(acc[nt][0] * scale2_ab);
            }
            if (r0 < M && c1 < N) {
                *(unsigned short*)&C_bf16[(int64_t)r0 * N + c1] =
                    w4sk_f32_to_bf16_rn(acc[nt][1] * scale2_ab);
            }
            if (r1 < M && c0 < N) {
                *(unsigned short*)&C_bf16[(int64_t)r1 * N + c0] =
                    w4sk_f32_to_bf16_rn(acc[nt][2] * scale2_ab);
            }
            if (r1 < M && c1 < N) {
                *(unsigned short*)&C_bf16[(int64_t)r1 * N + c1] =
                    w4sk_f32_to_bf16_rn(acc[nt][3] * scale2_ab);
            }
        } else {
            if (r0 < M && c0 < N) workspace_f32[ws_slice_base + (int64_t)r0 * N + c0] = acc[nt][0];
            if (r0 < M && c1 < N) workspace_f32[ws_slice_base + (int64_t)r0 * N + c1] = acc[nt][1];
            if (r1 < M && c0 < N) workspace_f32[ws_slice_base + (int64_t)r1 * N + c0] = acc[nt][2];
            if (r1 < M && c1 < N) workspace_f32[ws_slice_base + (int64_t)r1 * N + c1] = acc[nt][3];
        }
    }

#else
    const unsigned int k_end = tile_end << 6;
    // Exact scalar FP32 fallback for SM103a (< SM120):
    // 128 threads = 16 rows x 8 column strips of 8 columns each (64 N cols/CTA).
    // Preserves identical K-slice partition, 3-op SWAR E2M1 canonicalization,
    // fixed reduction order, and 0 B stack/spills.
    __shared__ float s_lut[16];
    if (threadIdx.x < 16u) s_lut[threadIdx.x] = W4SK_E2M1_LUT[threadIdx.x];
    __syncthreads();

    const unsigned int r = threadIdx.x & 15u;
    const unsigned int c_base = cta_n + (threadIdx.x >> 4) * 8u;
    if (r >= M) return;

    float acc[8];
    #pragma unroll
    for (int ci = 0; ci < 8; ++ci) acc[ci] = 0.0f;

    for (unsigned int kb = k_begin; kb < k_end; kb += W4SK_K_STEP) {
        const int64_t gc_byte = (int64_t)(kb >> 1);
        const int64_t gg_base = (int64_t)(kb >> 4);
        const uint4 ap_lo = w4sk_canon_e2m1_u4(
            w4sk_ldg_nc_128(A_packed + (int64_t)r * half_K + gc_byte));
        const uint4 ap_hi = w4sk_canon_e2m1_u4(
            w4sk_ldg_nc_128(A_packed + (int64_t)r * half_K + gc_byte + 16));
        const unsigned int as_w =
            w4sk_ldg_nc_32(A_scale + (int64_t)r * num_groups + gg_base) & 0x7F7F7F7Fu;

        const unsigned int a_words[8] = {
            ap_lo.x, ap_lo.y, ap_lo.z, ap_lo.w,
            ap_hi.x, ap_hi.y, ap_hi.z, ap_hi.w
        };

        #pragma unroll
        for (int ci = 0; ci < 8; ++ci) {
            const unsigned int col = c_base + (unsigned int)ci;
            if (col >= N) continue;
            const uint4 bp_lo = w4sk_canon_e2m1_u4(
                w4sk_ldg_ef_128(B_packed + (int64_t)col * half_K + gc_byte, pol_ef));
            const uint4 bp_hi = w4sk_canon_e2m1_u4(
                w4sk_ldg_ef_128(B_packed + (int64_t)col * half_K + gc_byte + 16, pol_ef));
            const unsigned int bs_w =
                w4sk_ldg_nc_ef_32(B_scale + (int64_t)col * num_groups + gg_base, pol_ef) & 0x7F7F7F7Fu;

            const unsigned int b_words[8] = {
                bp_lo.x, bp_lo.y, bp_lo.z, bp_lo.w,
                bp_hi.x, bp_hi.y, bp_hi.z, bp_hi.w
            };

            float tile_sum = 0.0f;
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const float sa = w4sk_ue4m3_to_f32((unsigned char)((as_w >> (g * 8)) & 0x7Fu));
                const float sb = w4sk_ue4m3_to_f32((unsigned char)((bs_w >> (g * 8)) & 0x7Fu));
                float g_dot = 0.0f;
                #pragma unroll
                for (int w_idx = 0; w_idx < 2; ++w_idx) {
                    const unsigned int aw = a_words[g * 2 + w_idx];
                    const unsigned int bw = b_words[g * 2 + w_idx];
                    #pragma unroll
                    for (int nib = 0; nib < 8; ++nib) {
                        const unsigned int an = (aw >> (nib * 4)) & 0xFu;
                        const unsigned int bn = (bw >> (nib * 4)) & 0xFu;
                        g_dot += s_lut[an] * s_lut[bn];
                    }
                }
                tile_sum += g_dot * (sa * sb);
            }
            acc[ci] += tile_sum;
        }
    }

    const int64_t ws_slice_base = (int64_t)sk * (int64_t)M * (int64_t)N;
    #pragma unroll
    for (int ci = 0; ci < 8; ++ci) {
        const unsigned int col = c_base + (unsigned int)ci;
        if (col >= N) continue;
        if (split_k == 1u) {
            *(unsigned short*)&C_bf16[(int64_t)r * N + col] =
                w4sk_f32_to_bf16_rn(acc[ci] * scale2_ab);
        } else {
            workspace_f32[ws_slice_base + (int64_t)r * N + col] = acc[ci];
        }
    }
#endif
}

// =========================================================================
// 2. reduce_w4a4_sk3_bf16_sm120
//
// Deterministic fixed-order split-K workspace reduction (`s = 0..split_k-1`)
// with explicit inline PTX `cvt.rn.bf16.f32` epilogue (ZERO atomicAdd).
//
// Grid:  (ceil(M * N / 256), 1, 1)
// Block: (256, 1, 1)
// =========================================================================
extern "C" __global__
__launch_bounds__(256, 4)
void reduce_w4a4_sk3_bf16_sm120(
    const float*   __restrict__ workspace_f32,  // [split_k, M, N] FP32
    __nv_bfloat16* __restrict__ C_bf16,         // [M, N] BF16
    const float scale2_ab,
    unsigned int M,
    unsigned int N,
    unsigned int split_k
) {
    if (M == 0u || N == 0u || split_k == 0u || split_k > W4SK_MAX_SPLITK) return;

    const int64_t total = (int64_t)M * (int64_t)N;
    const int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (idx >= total) return;

    float sum = 0.0f;
    for (unsigned int s = 0u; s < split_k; ++s) {
        sum += workspace_f32[(int64_t)s * total + idx];
    }
    *(unsigned short*)&C_bf16[idx] = w4sk_f32_to_bf16_rn(sum * scale2_ab);
}
