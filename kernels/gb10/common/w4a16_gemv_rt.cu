// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 register-tiled exact multi-row GEMV (T=2 outputs per lane group).
//
// This is a bandwidth transform of `w4a16_gemv_batch_logits_exact_body` in
// w4a16_gemv.cu, NOT a new numerical formulation. Each 64-lane group covers T
// ADJACENT output rows instead of one, so the activation slab for a given
// (k16, row) is loaded into registers ONCE and feeds T independent accumulator
// chains. Grid shrinks to ceil(N / (N_PER_BLOCK * T)).
//
// Bit-exactness contract — every one of these must hold, or the transform is
// not a drop-in replacement:
//   * K16 lane ownership is unchanged: k16 = lane, stride 64, ascending.
//   * Weights are pre-scaled per element (LUT[nibble] * scale) exactly as the
//     baseline does, and the two accumulator updates per packed byte stay in
//     (low, high) order. Under the common/ `--fmad=false` build this is a
//     genuine MUL then ADD, so it must not be rewritten as fmaf().
//   * The per-output five-step shuffle tree and the ordered two-warp cross-warp
//     add are replayed independently per (output, row).
//   * Tail groups never return early; they enter both barriers with zero
//     accumulators while every global load/store stays predicated.
// Nothing above depends on T, so acc[o][row] sees the identical operand
// sequence it would see in the baseline kernel for output row n0 + o.
//
// This deliberately preserves the per-element scale multiplication rather than
// hoisting `fmaf(scale, partial, acc)` out of the inner loop, keeping results
// bit-identical against the exact batch family used for speculative verify.
//
// A:         [M, K] BF16 contiguous
// B_packed:  [N, K/2] NVFP4 packed, row-major
// B_scale:   [N, K/16] FP8-E4M3
// C:         [M, N] BF16 contiguous
// Grid:      (ceil(N / (N_PER_BLOCK * T)), 1, 1)
// Block:     (256, 1, 1), 64 threads / lane group, T outputs / lane group

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16
#define RT_ASTAGE_ROWS 9
#define RT_ASTAGE_U4_PER_ROW 128

// Must stay byte-identical to the table in w4a16_gemv.cu.
__device__ __constant__ float E2M1_LUT_RT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

template <int MAX_M, int T, bool STAGE_ACTIVATIONS = false>
__device__ __forceinline__ void w4a16_gemv_batch_logits_exact_rt_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    float* s_lut,
    float* smem,
    uint4* s_a = nullptr)
{
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = lane % WARP_SIZE;
    const unsigned int warp_idx = lane / WARP_SIZE;
    // This lane group owns T adjacent output rows n0 .. n0 + T - 1.
    const unsigned int n0 = (blockIdx.x * N_PER_BLOCK + local_out) * T;
    const bool rows_valid = M > 0 && M <= (unsigned int)MAX_M;

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_RT[threadIdx.x];
    __syncthreads();

    float acc[T][MAX_M];
    #pragma unroll
    for (int o = 0; o < T; ++o) {
        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) acc[o][row] = 0.0f;
    }

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    if constexpr (STAGE_ACTIVATIONS) {
        const unsigned int K8 = K / 8u;
        for (unsigned int wave_k16 = 0; wave_k16 < K16; wave_k16 += 64u) {
            const unsigned int k16 = wave_k16 + lane;
            const bool k_valid = k16 < K16;
            const unsigned int base_k = k16 * 16;
            const unsigned int scale_group = base_k / GROUP_SIZE;

            float w_lo[T][8];
            float w_hi[T][8];
            #pragma unroll
            for (int o = 0; o < T; ++o) {
                const unsigned int n = n0 + (unsigned int)o;
                if (rows_valid && k_valid && n < N) {
                    const unsigned long long weight_row =
                        (unsigned long long)n * half_K;
                    const unsigned long long scale_row =
                        (unsigned long long)n * num_groups;
                    const unsigned long long packed8 =
                        *(const unsigned long long*)(B_packed + weight_row + k16 * 8);
                    const unsigned char scale_byte = B_scale[scale_row + scale_group];
                    __nv_fp8_e4m3 fp8;
                    *(unsigned char*)&fp8 = scale_byte;
                    const float scale = (float)fp8 * scale2;
                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        const unsigned char byte_val =
                            (unsigned char)(packed8 >> (b * 8));
                        w_lo[o][b] = s_lut[byte_val & 0xF] * scale;
                        w_hi[o][b] = s_lut[byte_val >> 4] * scale;
                    }
                } else {
                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        w_lo[o][b] = 0.0f;
                        w_hi[o][b] = 0.0f;
                    }
                }
            }

            const unsigned int wave_u4_base = wave_k16 * 2u;
            constexpr int NUM_TILES = (MAX_M + RT_ASTAGE_ROWS - 1) / RT_ASTAGE_ROWS;
            #pragma unroll
            for (int tile = 0; tile < NUM_TILES; ++tile) {
                const int row_base = tile * RT_ASTAGE_ROWS;
                const int tile_cap =
                    (MAX_M - row_base < RT_ASTAGE_ROWS) ? (MAX_M - row_base) : RT_ASTAGE_ROWS;
                if (rows_valid && (unsigned int)row_base < M) {
                    const unsigned int rem_m = M - (unsigned int)row_base;
                    const unsigned int tile_rows =
                        rem_m < (unsigned int)tile_cap ? rem_m : (unsigned int)tile_cap;
                    const unsigned int total_u4 = tile_rows * RT_ASTAGE_U4_PER_ROW;
                    for (unsigned int idx = threadIdx.x; idx < total_u4; idx += BLOCK_SIZE) {
                        const unsigned int r_local = idx >> 7;
                        const unsigned int u4_off = idx & 127u;
                        const unsigned int g_u4 = wave_u4_base + u4_off;
                        const __nv_bfloat16* __restrict__ A_row =
                            A + (unsigned long long)((unsigned int)row_base + r_local) * K;
                        s_a[idx] = (g_u4 < K8)
                            ? __ldg(&((const uint4*)A_row)[g_u4])
                            : make_uint4(0u, 0u, 0u, 0u);
                    }
                    __syncthreads();

                    if (k_valid && n0 < N) {
                        #pragma unroll
                        for (int r = 0; r < tile_cap; ++r) {
                            if ((unsigned int)(row_base + r) < M) {
                                const uint4 a_lo =
                                    s_a[(unsigned int)r * RT_ASTAGE_U4_PER_ROW + lane * 2u];
                                const uint4 a_hi =
                                    s_a[(unsigned int)r * RT_ASTAGE_U4_PER_ROW + lane * 2u + 1u];
                                const unsigned int a_raw[8] = {
                                    a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                    a_hi.x, a_hi.y, a_hi.z, a_hi.w
                                };
                                #pragma unroll
                                for (int o = 0; o < T; ++o) {
                                    #pragma unroll
                                    for (int b = 0; b < 8; ++b) {
                                        __nv_bfloat16 a_lo_bf, a_hi_bf;
                                        *(unsigned short*)&a_lo_bf =
                                            (unsigned short)(a_raw[b] & 0xFFFF);
                                        *(unsigned short*)&a_hi_bf =
                                            (unsigned short)(a_raw[b] >> 16);
                                        acc[o][row_base + r] +=
                                            __bfloat162float(a_lo_bf) * w_lo[o][b];
                                        acc[o][row_base + r] +=
                                            __bfloat162float(a_hi_bf) * w_hi[o][b];
                                    }
                                }
                            }
                        }
                    }
                    __syncthreads();
                }
            }
        }
    } else {
        // Keep the ordinary K=1 lane ownership exactly: k16 = lane, stride 64.
        for (unsigned int k16 = lane; k16 < K16; k16 += 64u) {
            const unsigned int base_k = k16 * 16;
            const unsigned int scale_group = base_k / GROUP_SIZE;

            // T weight chunks, one per adjacent output row. Pre-scaled per element
            // so the inner accumulator update matches the baseline operand for
            // operand.
            float w_lo[T][8];
            float w_hi[T][8];
            #pragma unroll
            for (int o = 0; o < T; ++o) {
                const unsigned int n = n0 + (unsigned int)o;
                if (rows_valid && n < N) {
                    const unsigned long long weight_row =
                        (unsigned long long)n * half_K;
                    const unsigned long long scale_row =
                        (unsigned long long)n * num_groups;
                    const unsigned long long packed8 =
                        *(const unsigned long long*)(B_packed + weight_row + k16 * 8);
                    const unsigned char scale_byte = B_scale[scale_row + scale_group];
                    __nv_fp8_e4m3 fp8;
                    *(unsigned char*)&fp8 = scale_byte;
                    const float scale = (float)fp8 * scale2;
                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        const unsigned char byte_val =
                            (unsigned char)(packed8 >> (b * 8));
                        w_lo[o][b] = s_lut[byte_val & 0xF] * scale;
                        w_hi[o][b] = s_lut[byte_val >> 4] * scale;
                    }
                } else {
                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        w_lo[o][b] = 0.0f;
                        w_hi[o][b] = 0.0f;
                    }
                }
            }

            #pragma unroll
            for (int row = 0; row < MAX_M; ++row) {
                if (rows_valid && (unsigned int)row < M && n0 < N) {
                    const __nv_bfloat16* __restrict__ A_row =
                        A + (unsigned long long)row * K;
                    // ONE activation slab per (k16, row), reused by all T chains.
                    const uint4 a_lo = ((const uint4*)A_row)[k16 * 2];
                    const uint4 a_hi = ((const uint4*)A_row)[k16 * 2 + 1];
                    const unsigned int a_raw[8] = {
                        a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                        a_hi.x, a_hi.y, a_hi.z, a_hi.w
                    };

                    #pragma unroll
                    for (int o = 0; o < T; ++o) {
                        #pragma unroll
                        for (int b = 0; b < 8; ++b) {
                            __nv_bfloat16 a_lo_bf, a_hi_bf;
                            *(unsigned short*)&a_lo_bf =
                                (unsigned short)(a_raw[b] & 0xFFFF);
                            *(unsigned short*)&a_hi_bf =
                                (unsigned short)(a_raw[b] >> 16);
                            acc[o][row] += __bfloat162float(a_lo_bf) * w_lo[o][b];
                            acc[o][row] += __bfloat162float(a_hi_bf) * w_hi[o][b];
                        }
                    }
                }
            }
        }
    }

    // Reproduce the ordinary K=1 five-step tree independently for every
    // (output, row) pair.
    #pragma unroll
    for (int o = 0; o < T; ++o) {
        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) {
            if ((unsigned int)row < M) {
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                    acc[o][row] += __shfl_down_sync(0xFFFFFFFF, acc[o][row], offset);
                }
                if (warp_lane == 0) {
                    smem[(((unsigned int)row * N_PER_BLOCK + local_out) *
                          (unsigned int)T + (unsigned int)o) * 2 + warp_idx] =
                        acc[o][row];
                }
            }
        }
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int o = 0; o < T; ++o) {
            const unsigned int n = n0 + (unsigned int)o;
            if (!rows_valid || n >= N) continue;
            #pragma unroll
            for (int row = 0; row < MAX_M; ++row) {
                if ((unsigned int)row < M) {
                    const unsigned int base =
                        (((unsigned int)row * N_PER_BLOCK + local_out) *
                         (unsigned int)T + (unsigned int)o) * 2;
                    C[(unsigned long long)row * N + n] =
                        __float2bfloat16(smem[base] + smem[base + 1]);
                }
            }
        }
    }
}

#define DEFINE_W4A16_GEMV_BATCH_LOGITS_EXACT_RT(NAME, MAX_ROWS, TILE)         \
extern "C" __global__ void NAME(                                             \
    const __nv_bfloat16* __restrict__ A,                                     \
    const unsigned char* __restrict__ B_packed,                              \
    const unsigned char* __restrict__ B_scale,                               \
    const float scale2,                                                       \
    __nv_bfloat16* __restrict__ C,                                            \
    unsigned int M, unsigned int N, unsigned int K)                           \
{                                                                            \
    __shared__ float s_lut[16];                                               \
    __shared__ float smem[(MAX_ROWS) * N_PER_BLOCK * (TILE) * 2];             \
    w4a16_gemv_batch_logits_exact_rt_body<MAX_ROWS, TILE>(                    \
        A, B_packed, B_scale, scale2, C, M, N, K, s_lut, smem);               \
}

DEFINE_W4A16_GEMV_BATCH_LOGITS_EXACT_RT(
    w4a16_gemv_batch_logits_exact_rt2_m4, 4, 2)
DEFINE_W4A16_GEMV_BATCH_LOGITS_EXACT_RT(
    w4a16_gemv_batch_logits_exact_rt2_m8, 8, 2)

extern "C" __global__ void __launch_bounds__(256, 2) w4a16_gemv_batch_logits_exact_rt2_m17(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K)
{
    __shared__ float s_lut[16];
    __shared__ union {
        uint4 s_a[RT_ASTAGE_ROWS * RT_ASTAGE_U4_PER_ROW];
        float smem[17 * N_PER_BLOCK * 2 * 2];
    } shared_ws;
    w4a16_gemv_batch_logits_exact_rt_body<17, 2, true>(
        A, B_packed, B_scale, scale2, C, M, N, K, s_lut, shared_ws.smem, shared_ws.s_a);
}

extern "C" __global__ void __launch_bounds__(256, 2) w4a16_gemv_batch_logits_exact_rt2_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K)
{
    __shared__ float s_lut[16];
    __shared__ union {
        uint4 s_a[RT_ASTAGE_ROWS * RT_ASTAGE_U4_PER_ROW];
        float smem[32 * N_PER_BLOCK * 2 * 2];
    } shared_ws;
    w4a16_gemv_batch_logits_exact_rt_body<32, 2, true>(
        A, B_packed, B_scale, scale2, C, M, N, K, s_lut, shared_ws.smem, shared_ws.s_a);
}

extern "C" __global__ void w4a16_gemv_batch_logits_exact_rt2_m32_grid(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K)
{
    const unsigned int row_start = blockIdx.y * 32u;
    if (row_start >= M) return;
    const unsigned int tile_rows = min(32u, M - row_start);
    __shared__ float s_lut[16];
    __shared__ float smem[32 * N_PER_BLOCK * 2 * 2];
    w4a16_gemv_batch_logits_exact_rt_body<32, 2>(
        A + (unsigned long long)row_start * K,
        B_packed,
        B_scale,
        scale2,
        C + (unsigned long long)row_start * N,
        tile_rows,
        N,
        K,
        s_lut,
        smem);
}

#undef DEFINE_W4A16_GEMV_BATCH_LOGITS_EXACT_RT
