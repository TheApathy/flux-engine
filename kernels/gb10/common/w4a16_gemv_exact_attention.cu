// SPDX-License-Identifier: AGPL-3.0-only

// Exact dynamic-M attention projections for the ordinary gated-Q and dual-KV
// NVFP4 K1 kernels. Each row retains the K1 K8 lane assignment, ordered
// low/high FP32 updates, shuffle tree, cross-warp add, and BF16 rounding.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16
#define ASTAGE_K8_PER_WAVE 64

__device__ __constant__ float E2M1_LUT_EXACT_ATTN[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

template <int MAX_M, bool STAGE_ACTIVATIONS, int ASTAGE_CAP = 22>
__device__ __forceinline__ void w4a16_attention_exact_accumulate_k8_prefetched(
    const __nv_bfloat16* __restrict__ A,
    unsigned int packed4,
    unsigned char scale_byte,
    const float scale2,
    unsigned int M,
    unsigned int K,
    unsigned int k8,
    unsigned int lane,
    const float* s_lut,
    const uint4* s_a,
    float* acc)
{
    __nv_fp8_e4m3 fp8;
    *(unsigned char*)&fp8 = scale_byte;
    const float scale = (float)fp8 * scale2;

    float w_lo[4];
    float w_hi[4];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        const unsigned char byte_val =
            (unsigned char)(packed4 >> (b * 8));
        w_lo[b] = s_lut[byte_val & 0xF] * scale;
        w_hi[b] = s_lut[byte_val >> 4] * scale;
    }

    constexpr int ASTAGE_MAX_ROWS = (MAX_M > ASTAGE_CAP) ? ASTAGE_CAP : MAX_M;
    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) {
        if ((unsigned int)row < M) {
            const uint4 a_data = (STAGE_ACTIVATIONS && row < ASTAGE_MAX_ROWS)
                ? s_a[row * ASTAGE_K8_PER_WAVE + lane]
                : __ldg(&((const uint4*)(A + (unsigned long long)row * K))[k8]);
            const unsigned int a_raw[4] = {
                a_data.x, a_data.y, a_data.z, a_data.w
            };
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                __nv_bfloat16 a_lo, a_hi;
                *(unsigned short*)&a_lo =
                    (unsigned short)(a_raw[b] & 0xFFFF);
                *(unsigned short*)&a_hi =
                    (unsigned short)(a_raw[b] >> 16);
                acc[row] += __bfloat162float(a_lo) * w_lo[b];
                acc[row] += __bfloat162float(a_hi) * w_hi[b];
            }
        }
    }
}

template <int MAX_M, bool STAGE_ACTIVATIONS, int ASTAGE_CAP = 22>
__device__ __forceinline__ void w4a16_attention_exact_accumulate_k8(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    unsigned int M,
    unsigned int K,
    unsigned int n,
    unsigned int k8,
    unsigned int lane,
    const float* s_lut,
    const uint4* s_a,
    float* acc)
{
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int base_k = k8 * 8;
    const unsigned long long weight_row = (unsigned long long)n * half_K;
    const unsigned long long scale_row = (unsigned long long)n * num_groups;
    const unsigned int packed4 = *(const unsigned int*)(
        B_packed + weight_row + k8 * 4);
    const unsigned char scale_byte =
        B_scale[scale_row + base_k / GROUP_SIZE];
    w4a16_attention_exact_accumulate_k8_prefetched<MAX_M, STAGE_ACTIVATIONS, ASTAGE_CAP>(
        A, packed4, scale_byte, scale2, M, K, k8, lane, s_lut, s_a, acc);
}

template <int MAX_M, bool QG_LAYOUT, bool STAGE_ACTIVATIONS, unsigned int ASTAGE_CAP = 22u>
__device__ __forceinline__ void w4a16_attention_exact_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride,
    float* s_lut,
    float* smem,
    uint4* s_a)
{
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = lane % WARP_SIZE;
    const unsigned int warp_idx = lane / WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = M > 0u && M <= (unsigned int)MAX_M && n < N;

    if (threadIdx.x < 16) {
        s_lut[threadIdx.x] = E2M1_LUT_EXACT_ATTN[threadIdx.x];
    }
    __syncthreads();

    float acc[MAX_M];
    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) acc[row] = 0.0f;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;
    const unsigned long long weight_row = (unsigned long long)n * half_K;
    const unsigned long long scale_row = (unsigned long long)n * num_groups;
    if (STAGE_ACTIVATIONS) {
        constexpr unsigned int ASTAGE_MAX_ROWS = ((unsigned int)MAX_M > ASTAGE_CAP) ? ASTAGE_CAP : (unsigned int)MAX_M;
        const unsigned int stage_rows = (M < ASTAGE_MAX_ROWS) ? M : ASTAGE_MAX_ROWS;
        const unsigned int waves =
            (K8 + ASTAGE_K8_PER_WAVE - 1) / ASTAGE_K8_PER_WAVE;
        const unsigned int local_k8 = threadIdx.x & 63u;
        const unsigned int r_init = threadIdx.x >> 6;
        for (unsigned int wave = 0; wave < waves; ++wave) {
            if (wave > 0u) __syncthreads();
            const unsigned int global_k8 =
                wave * ASTAGE_K8_PER_WAVE + lane;
            unsigned int packed4 = 0u;
            unsigned char scale_byte = 0u;
            if (valid && global_k8 < K8) {
                packed4 = *(const unsigned int*)(
                    B_packed + weight_row + (unsigned long long)global_k8 * 4u);
                scale_byte = B_scale[scale_row + (global_k8 * 8u) / GROUP_SIZE];
            }
            const unsigned int global_k8_stage = wave * ASTAGE_K8_PER_WAVE + local_k8;
            if (global_k8_stage < K8) {
                for (unsigned int row = r_init; row < stage_rows; row += 4u) {
                    const __nv_bfloat16* A_row = A + (unsigned long long)row * K;
                    s_a[row * ASTAGE_K8_PER_WAVE + local_k8] = __ldg(&((const uint4*)A_row)[global_k8_stage]);
                }
            }
            __syncthreads();

            if (valid) {
                if (global_k8 < K8) {
                    w4a16_attention_exact_accumulate_k8_prefetched<MAX_M, true, (int)ASTAGE_CAP>(
                        A, packed4, scale_byte, scale2, M, K, global_k8,
                        lane, s_lut, s_a, acc);
                }
            }
        }
    } else if (valid) {
        const unsigned int K8 = K / 8;

        // Ordinary gated-Q and dual-KV K1 kernels both own K8 chunks by
        // lane=0..63 with stride 64. Keep that association byte-for-byte.
        for (unsigned int k8 = lane; k8 < K8; k8 += 64u) {
            w4a16_attention_exact_accumulate_k8<MAX_M, false>(
                A, B_packed, B_scale, scale2, M, K, n, k8, lane,
                s_lut, s_a, acc);
        }
    }

    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) {
        if ((unsigned int)row < M) {
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                acc[row] += __shfl_down_sync(0xFFFFFFFF, acc[row], offset);
            }
            if (warp_lane == 0) {
                smem[(row * N_PER_BLOCK + local_out) * 2 + warp_idx] = acc[row];
            }
        }
    }
    __syncthreads();

    if (valid && lane == 0) {
        unsigned int out_idx = n;
        if (QG_LAYOUT) {
            const unsigned int group_dim = 2 * head_dim;
            const unsigned int h = n / group_dim;
            const unsigned int idx = n % group_dim;
            const unsigned int q_total = num_heads * head_dim;
            if (idx < head_dim) {
                out_idx = h * head_dim + idx;
            } else {
                out_idx = q_total + h * head_dim + (idx - head_dim);
            }
        }
        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) {
            if ((unsigned int)row < M) {
                const unsigned int base =
                    (row * N_PER_BLOCK + local_out) * 2;
                C[(unsigned long long)row * out_stride + out_idx] =
                    __float2bfloat16(smem[base] + smem[base + 1]);
            }
        }
    }
}

extern "C" __global__ void w4a16_gemv_qg_exact_m4(
    const __nv_bfloat16* A,
    const unsigned char* B_packed,
    const unsigned char* B_scale,
    const float scale2,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride)
{
    __shared__ float s_lut[16];
    __shared__ float smem[4 * N_PER_BLOCK * 2];
    w4a16_attention_exact_body<4, true, false>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        num_heads, head_dim, out_stride, s_lut, smem, nullptr);
}

extern "C" __global__ void w4a16_gemv_dual_kv_exact_m4(
    const __nv_bfloat16* A,
    const unsigned char* K_packed,
    const unsigned char* K_scale,
    const float K_scale2,
    __nv_bfloat16* K_out,
    const unsigned char* V_packed,
    const unsigned char* V_scale,
    const float V_scale2,
    __nv_bfloat16* V_out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride)
{
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? K_packed : V_packed;
    const unsigned char* B_scale = proj == 0 ? K_scale : V_scale;
    const float scale2 = proj == 0 ? K_scale2 : V_scale2;
    __nv_bfloat16* C = proj == 0 ? K_out : V_out;
    __shared__ float s_lut[16];
    __shared__ float smem[4 * N_PER_BLOCK * 2];
    w4a16_attention_exact_body<4, false, false>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        0, 0, out_stride, s_lut, smem, nullptr);
}

extern "C" __global__ void __launch_bounds__(256, 3) w4a16_gemv_qg_exact_m17(
    const __nv_bfloat16* A,
    const unsigned char* B_packed,
    const unsigned char* B_scale,
    const float scale2,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride)
{
    __shared__ float s_lut[16];
    __shared__ float smem[17 * N_PER_BLOCK * 2];
    __shared__ uint4 s_a[16 * ASTAGE_K8_PER_WAVE];
    if (M <= 5u) {
        w4a16_attention_exact_body<5, true, false>(
            A, B_packed, B_scale, scale2, C, M, N, K,
            num_heads, head_dim, out_stride, s_lut, smem, nullptr);
    } else if (M <= 10u) {
        w4a16_attention_exact_body<10, true, true, 10u>(
            A, B_packed, B_scale, scale2, C, M, N, K,
            num_heads, head_dim, out_stride, s_lut, smem, s_a);
    } else if (M <= 15u) {
        w4a16_attention_exact_body<15, true, true, 15u>(
            A, B_packed, B_scale, scale2, C, M, N, K,
            num_heads, head_dim, out_stride, s_lut, smem, s_a);
    } else {
        w4a16_attention_exact_body<17, true, true, 8u>(
            A, B_packed, B_scale, scale2, C, M, N, K,
            num_heads, head_dim, out_stride, s_lut, smem, s_a);
    }
}

extern "C" __global__ void __launch_bounds__(256, 3) w4a16_gemv_dual_kv_exact_m17(
    const __nv_bfloat16* A,
    const unsigned char* K_packed,
    const unsigned char* K_scale,
    const float K_scale2,
    __nv_bfloat16* K_out,
    const unsigned char* V_packed,
    const unsigned char* V_scale,
    const float V_scale2,
    __nv_bfloat16* V_out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride)
{
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? K_packed : V_packed;
    const unsigned char* B_scale = proj == 0 ? K_scale : V_scale;
    const float scale2 = proj == 0 ? K_scale2 : V_scale2;
    __nv_bfloat16* C = proj == 0 ? K_out : V_out;
    __shared__ float s_lut[16];
    __shared__ float smem[17 * N_PER_BLOCK * 2];
    __shared__ uint4 s_a[16 * ASTAGE_K8_PER_WAVE];
    if (M <= 5u) {
        w4a16_attention_exact_body<5, false, false>(
            A, B_packed, B_scale, scale2, C, M, N, K,
            0, 0, out_stride, s_lut, smem, nullptr);
    } else if (M <= 10u) {
        w4a16_attention_exact_body<10, false, true, 10u>(
            A, B_packed, B_scale, scale2, C, M, N, K,
            0, 0, out_stride, s_lut, smem, s_a);
    } else if (M <= 15u) {
        w4a16_attention_exact_body<15, false, true, 15u>(
            A, B_packed, B_scale, scale2, C, M, N, K,
            0, 0, out_stride, s_lut, smem, s_a);
    } else {
        w4a16_attention_exact_body<17, false, true, 8u>(
            A, B_packed, B_scale, scale2, C, M, N, K,
            0, 0, out_stride, s_lut, smem, s_a);
    }
}

// M18..32 exact tier. This deliberately retains the K1 ownership
// and reduction order; MAX_M only adds independent row accumulators.
extern "C" __global__ void __launch_bounds__(256, 3) w4a16_gemv_qg_exact_m32(
    const __nv_bfloat16* A,
    const unsigned char* B_packed,
    const unsigned char* B_scale,
    const float scale2,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride)
{
    __shared__ float s_lut[16];
    __shared__ float smem[32 * N_PER_BLOCK * 2];
    __shared__ uint4 s_a[25 * ASTAGE_K8_PER_WAVE];
    if (M <= 20u) {
        w4a16_attention_exact_body<20, true, true, 20u>(
            A, B_packed, B_scale, scale2, C, M, N, K,
            num_heads, head_dim, out_stride, s_lut, smem, s_a);
    } else if (M <= 25u) {
        w4a16_attention_exact_body<25, true, true, 25u>(
            A, B_packed, B_scale, scale2, C, M, N, K,
            num_heads, head_dim, out_stride, s_lut, smem, s_a);
    } else {
        w4a16_attention_exact_body<32, true, true, 25u>(
            A, B_packed, B_scale, scale2, C, M, N, K,
            num_heads, head_dim, out_stride, s_lut, smem, s_a);
    }
}

extern "C" __global__ void __launch_bounds__(256, 3) w4a16_gemv_dual_kv_exact_m32(
    const __nv_bfloat16* A,
    const unsigned char* K_packed,
    const unsigned char* K_scale,
    const float K_scale2,
    __nv_bfloat16* K_out,
    const unsigned char* V_packed,
    const unsigned char* V_scale,
    const float V_scale2,
    __nv_bfloat16* V_out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride)
{
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? K_packed : V_packed;
    const unsigned char* B_scale = proj == 0 ? K_scale : V_scale;
    const float scale2 = proj == 0 ? K_scale2 : V_scale2;
    __nv_bfloat16* C = proj == 0 ? K_out : V_out;
    __shared__ float s_lut[16];
    __shared__ float smem[32 * N_PER_BLOCK * 2];
    __shared__ uint4 s_a[25 * ASTAGE_K8_PER_WAVE];
    if (M <= 20u) {
        w4a16_attention_exact_body<20, false, true, 20u>(
            A, B_packed, B_scale, scale2, C, M, N, K,
            0, 0, out_stride, s_lut, smem, s_a);
    } else if (M <= 25u) {
        w4a16_attention_exact_body<25, false, true, 25u>(
            A, B_packed, B_scale, scale2, C, M, N, K,
            0, 0, out_stride, s_lut, smem, s_a);
    } else {
        w4a16_attention_exact_body<32, false, true, 25u>(
            A, B_packed, B_scale, scale2, C, M, N, K,
            0, 0, out_stride, s_lut, smem, s_a);
    }
}

// M17-only activation-staging candidates. Each 512-column BF16 input tile is
// loaded once by the CTA and reused by its four independent output groups.
// The compute lanes still visit global k8 = lane + 64*wave in the parent's
// order; staging changes only where the identical 16 input bytes are read.
extern "C" __global__ void __launch_bounds__(256, 3) w4a16_gemv_qg_exact_m17_astage(
    const __nv_bfloat16* A,
    const unsigned char* B_packed,
    const unsigned char* B_scale,
    const float scale2,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride)
{
    __shared__ float s_lut[16];
    __shared__ float smem[17 * N_PER_BLOCK * 2];
    __shared__ uint4 s_a[17 * ASTAGE_K8_PER_WAVE];
    w4a16_attention_exact_body<17, true, true>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        num_heads, head_dim, out_stride, s_lut, smem, s_a);
}

extern "C" __global__ void __launch_bounds__(256, 3) w4a16_gemv_dual_kv_exact_m17_astage(
    const __nv_bfloat16* A,
    const unsigned char* K_packed,
    const unsigned char* K_scale,
    const float K_scale2,
    __nv_bfloat16* K_out,
    const unsigned char* V_packed,
    const unsigned char* V_scale,
    const float V_scale2,
    __nv_bfloat16* V_out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride)
{
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? K_packed : V_packed;
    const unsigned char* B_scale = proj == 0 ? K_scale : V_scale;
    const float scale2 = proj == 0 ? K_scale2 : V_scale2;
    __nv_bfloat16* C = proj == 0 ? K_out : V_out;
    __shared__ float s_lut[16];
    __shared__ float smem[17 * N_PER_BLOCK * 2];
    __shared__ uint4 s_a[17 * ASTAGE_K8_PER_WAVE];
    w4a16_attention_exact_body<17, false, true>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        0, 0, out_stride, s_lut, smem, s_a);
}

// Single-sweep wide-row exact attention body (MAX_M = 64 / 80 / 128).
// Per k8 wave, weights are loaded and unpacked once into w_lo[4]/w_hi[4] and
// reused across 16-row slabs staged cooperatively into s_a[16 * 64] (16 KiB).
// Preserves the exact K1 K8 lane assignment, ordered low/high FP32 updates,
// 5-step warp shuffle tree, cross-warp add, and BF16 rounding (0-ULP).
template <int MAX_M, bool QG_LAYOUT>
__device__ __forceinline__ void w4a16_attention_exact_slab_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride,
    float* s_lut,
    float* smem,
    uint4* s_a)
{
    static_assert(MAX_M % 16 == 0, "MAX_M must be a multiple of 16");
    constexpr int SLAB_ROWS = 16;
    constexpr int NUM_SLABS = MAX_M / SLAB_ROWS;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = lane % WARP_SIZE;
    const unsigned int warp_idx = lane / WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = M > 0u && M <= (unsigned int)MAX_M && n < N;

    if (threadIdx.x < 16) {
        s_lut[threadIdx.x] = E2M1_LUT_EXACT_ATTN[threadIdx.x];
    }
    __syncthreads();

    float acc[MAX_M];
    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) acc[row] = 0.0f;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;
    const unsigned long long weight_row = (unsigned long long)n * half_K;
    const unsigned long long scale_row = (unsigned long long)n * num_groups;
    const unsigned int waves = (K8 + ASTAGE_K8_PER_WAVE - 1u) / ASTAGE_K8_PER_WAVE;
    const unsigned int local_k8 = threadIdx.x & 63u;
    const unsigned int r_init = threadIdx.x >> 6;

    if (M > 0u && M <= (unsigned int)MAX_M) {
        for (unsigned int wave = 0u; wave < waves; ++wave) {
            const unsigned int global_k8 = wave * ASTAGE_K8_PER_WAVE + lane;
            const bool valid_k = valid && global_k8 < K8;
            float w_lo[4];
            float w_hi[4];
            if (valid_k) {
                const unsigned int packed4 = *(const unsigned int*)(
                    B_packed + weight_row + (unsigned long long)global_k8 * 4u);
                const unsigned char scale_byte =
                    B_scale[scale_row + (global_k8 * 8u) / GROUP_SIZE];
                __nv_fp8_e4m3 fp8;
                *(unsigned char*)&fp8 = scale_byte;
                const float scale = (float)fp8 * scale2;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const unsigned char byte_val = (unsigned char)(packed4 >> (b * 8));
                    w_lo[b] = s_lut[byte_val & 0xF] * scale;
                    w_hi[b] = s_lut[byte_val >> 4] * scale;
                }
            }

            const unsigned int global_k8_stage = wave * ASTAGE_K8_PER_WAVE + local_k8;
            const bool stage_k_valid = global_k8_stage < K8;

            #pragma unroll
            for (int slab = 0; slab < NUM_SLABS; ++slab) {
                const unsigned int row_base = (unsigned int)(slab * SLAB_ROWS);
                if (row_base < M) {
                    if (wave > 0u || slab > 0) __syncthreads();
                    const unsigned int rem = M - row_base;
                    const unsigned int slab_rows = rem > (unsigned int)SLAB_ROWS
                        ? (unsigned int)SLAB_ROWS
                        : rem;
                    if (stage_k_valid) {
                        for (unsigned int rl = r_init; rl < slab_rows; rl += 4u) {
                            const __nv_bfloat16* A_row = A + (unsigned long long)(row_base + rl) * K;
                            s_a[rl * ASTAGE_K8_PER_WAVE + local_k8] =
                                __ldg(&((const uint4*)A_row)[global_k8_stage]);
                        }
                    }
                    __syncthreads();

                    if (valid_k) {
                        #pragma unroll
                        for (int rl = 0; rl < SLAB_ROWS; ++rl) {
                            if ((unsigned int)rl < slab_rows) {
                                const uint4 a_data = s_a[rl * ASTAGE_K8_PER_WAVE + lane];
                                const unsigned int a_raw[4] = {
                                    a_data.x, a_data.y, a_data.z, a_data.w
                                };
                                #pragma unroll
                                for (int b = 0; b < 4; ++b) {
                                    __nv_bfloat16 a_lo, a_hi;
                                    *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFFu);
                                    *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
                                    acc[slab * SLAB_ROWS + rl] += __bfloat162float(a_lo) * w_lo[b];
                                    acc[slab * SLAB_ROWS + rl] += __bfloat162float(a_hi) * w_hi[b];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) {
        if ((unsigned int)row < M) {
            float v = acc[row];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
            }
            if (warp_lane == 0u) {
                smem[(row * N_PER_BLOCK + local_out) * 2 + warp_idx] = v;
            }
        }
    }
    __syncthreads();

    if (valid && lane == 0u) {
        unsigned int out_idx = n;
        if (QG_LAYOUT) {
            const unsigned int group_dim = 2 * head_dim;
            const unsigned int h = n / group_dim;
            const unsigned int idx = n % group_dim;
            const unsigned int q_total = num_heads * head_dim;
            if (idx < head_dim) {
                out_idx = h * head_dim + idx;
            } else {
                out_idx = q_total + h * head_dim + (idx - head_dim);
            }
        }
        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) {
            if ((unsigned int)row < M) {
                const unsigned int base = (row * N_PER_BLOCK + local_out) * 2;
                C[(unsigned long long)row * out_stride + out_idx] =
                    __float2bfloat16(smem[base] + smem[base + 1]);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) w4a16_gemv_qg_exact_m64(
    const __nv_bfloat16* A,
    const unsigned char* B_packed,
    const unsigned char* B_scale,
    const float scale2,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride)
{
    __shared__ float s_lut[16];
    __shared__ float smem[64 * N_PER_BLOCK * 2];
    __shared__ __align__(16) uint4 s_a[16 * ASTAGE_K8_PER_WAVE];
    w4a16_attention_exact_slab_body<64, true>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        num_heads, head_dim, out_stride, s_lut, smem, s_a);
}

extern "C" __global__ void __launch_bounds__(256, 2) w4a16_gemv_dual_kv_exact_m64(
    const __nv_bfloat16* A,
    const unsigned char* K_packed,
    const unsigned char* K_scale,
    const float K_scale2,
    __nv_bfloat16* K_out,
    const unsigned char* V_packed,
    const unsigned char* V_scale,
    const float V_scale2,
    __nv_bfloat16* V_out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride)
{
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? K_packed : V_packed;
    const unsigned char* B_scale = proj == 0 ? K_scale : V_scale;
    const float scale2 = proj == 0 ? K_scale2 : V_scale2;
    __nv_bfloat16* C = proj == 0 ? K_out : V_out;
    __shared__ float s_lut[16];
    __shared__ float smem[64 * N_PER_BLOCK * 2];
    __shared__ __align__(16) uint4 s_a[16 * ASTAGE_K8_PER_WAVE];
    w4a16_attention_exact_slab_body<64, false>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        0, 0, out_stride, s_lut, smem, s_a);
}

extern "C" __global__ void __launch_bounds__(256, 2) w4a16_gemv_qg_exact_m80(
    const __nv_bfloat16* A,
    const unsigned char* B_packed,
    const unsigned char* B_scale,
    const float scale2,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride)
{
    __shared__ float s_lut[16];
    __shared__ float smem[80 * N_PER_BLOCK * 2];
    __shared__ __align__(16) uint4 s_a[16 * ASTAGE_K8_PER_WAVE];
    w4a16_attention_exact_slab_body<80, true>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        num_heads, head_dim, out_stride, s_lut, smem, s_a);
}

extern "C" __global__ void __launch_bounds__(256, 2) w4a16_gemv_dual_kv_exact_m80(
    const __nv_bfloat16* A,
    const unsigned char* K_packed,
    const unsigned char* K_scale,
    const float K_scale2,
    __nv_bfloat16* K_out,
    const unsigned char* V_packed,
    const unsigned char* V_scale,
    const float V_scale2,
    __nv_bfloat16* V_out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride)
{
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? K_packed : V_packed;
    const unsigned char* B_scale = proj == 0 ? K_scale : V_scale;
    const float scale2 = proj == 0 ? K_scale2 : V_scale2;
    __nv_bfloat16* C = proj == 0 ? K_out : V_out;
    __shared__ float s_lut[16];
    __shared__ float smem[80 * N_PER_BLOCK * 2];
    __shared__ __align__(16) uint4 s_a[16 * ASTAGE_K8_PER_WAVE];
    w4a16_attention_exact_slab_body<80, false>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        0, 0, out_stride, s_lut, smem, s_a);
}

extern "C" __global__ void __launch_bounds__(256, 1) w4a16_gemv_qg_exact_m128(
    const __nv_bfloat16* A,
    const unsigned char* B_packed,
    const unsigned char* B_scale,
    const float scale2,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride)
{
    __shared__ float s_lut[16];
    __shared__ float smem[128 * N_PER_BLOCK * 2];
    __shared__ __align__(16) uint4 s_a[16 * ASTAGE_K8_PER_WAVE];
    w4a16_attention_exact_slab_body<128, true>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        num_heads, head_dim, out_stride, s_lut, smem, s_a);
}

extern "C" __global__ void __launch_bounds__(256, 1) w4a16_gemv_dual_kv_exact_m128(
    const __nv_bfloat16* A,
    const unsigned char* K_packed,
    const unsigned char* K_scale,
    const float K_scale2,
    __nv_bfloat16* K_out,
    const unsigned char* V_packed,
    const unsigned char* V_scale,
    const float V_scale2,
    __nv_bfloat16* V_out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride)
{
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? K_packed : V_packed;
    const unsigned char* B_scale = proj == 0 ? K_scale : V_scale;
    const float scale2 = proj == 0 ? K_scale2 : V_scale2;
    __nv_bfloat16* C = proj == 0 ? K_out : V_out;
    __shared__ float s_lut[16];
    __shared__ float smem[128 * N_PER_BLOCK * 2];
    __shared__ __align__(16) uint4 s_a[16 * ASTAGE_K8_PER_WAVE];
    w4a16_attention_exact_slab_body<128, false>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        0, 0, out_stride, s_lut, smem, s_a);
}
