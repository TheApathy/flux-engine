// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Dense GEMM kernel for SM121 (GB10).
//
// C = A * B^T  where:
//   A: [M, K] BF16 (activations, row-major)
//   B: [N, K] BF16 (weights, row-major — standard HuggingFace layout)
//   C: [M, N] BF16 (output, row-major)
//
// The kernel reads B transposed: B^T[k,n] = B[n,k] = B[n*K + k].
//
// Phase 1: Correct scalar implementation with shared memory tiling.
// Phase 2: Will add mma.sync.aligned.m16n8k16 BF16 tensor cores.

#include <cuda_bf16.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// Tiled GEMM: C[M,N] = A[M,K] * B[N,K]^T
// All matrices in BF16, accumulation in FP32.
//
// Grid: (ceil(N/TILE_N), ceil(M/TILE_M))
// Block: (TILE_N, TILE_M) — each thread computes one output element
extern "C" __global__ void dense_gemm_bf16(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major (read transposed)
    __nv_bfloat16* __restrict__ C,         // [M, N] row-major
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    // Each thread computes one element of C
    unsigned int row = blockIdx.y * TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * TILE_N + threadIdx.x;

    // Shared memory tiles
    __shared__ __nv_bfloat16 smem_A[TILE_M][TILE_K];
    __shared__ __nv_bfloat16 smem_B[TILE_K][TILE_N];

    float acc = 0.0f;

    // Loop over K in TILE_K chunks
    for (unsigned int k_base = 0; k_base < K; k_base += TILE_K) {
        // Load A tile
        if (row < M && (k_base + threadIdx.x) < K) {
            smem_A[threadIdx.y][threadIdx.x] = A[(unsigned long long)row * K + k_base + threadIdx.x];
        } else {
            smem_A[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        // Load B tile (B is [N,K] row-major, read as B^T[K,N])
        if ((k_base + threadIdx.y) < K && col < N) {
            smem_B[threadIdx.y][threadIdx.x] = B[(unsigned long long)col * K + (k_base + threadIdx.y)];
        } else {
            smem_B[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        __syncthreads();

        // Compute partial dot product
        for (unsigned int kk = 0; kk < TILE_K; kk++) {
            acc += __bfloat162float(smem_A[threadIdx.y][kk])
                 * __bfloat162float(smem_B[kk][threadIdx.x]);
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[(unsigned long long)row * N + col] = __float2bfloat16(acc);
    }
}

// Router TK=64 companion: stage four K subtiles between block barriers while
// retaining dense_gemm_bf16's exact ascending scalar accumulation and 16-value tail.
extern "C" __global__ void dense_gemm_bf16_tk64(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    constexpr unsigned int TK = 64;
    const unsigned int row = blockIdx.y * TILE_M + threadIdx.y;
    const unsigned int col = blockIdx.x * TILE_N + threadIdx.x;
    __shared__ __nv_bfloat16 smem_A[TILE_M][TK];
    __shared__ __nv_bfloat16 smem_B[TK][TILE_N];
    float acc = 0.0f;

    for (unsigned int k_base = 0; k_base < K; k_base += TK) {
        for (unsigned int sub = 0; sub < TK; sub += TILE_K) {
            const unsigned int ak = k_base + sub + threadIdx.x;
            const unsigned int bk = k_base + sub + threadIdx.y;
            smem_A[threadIdx.y][sub + threadIdx.x] =
                row < M && ak < K
                    ? A[static_cast<unsigned long long>(row) * K + ak]
                    : __float2bfloat16(0.0f);
            smem_B[sub + threadIdx.y][threadIdx.x] =
                col < N && bk < K
                    ? B[static_cast<unsigned long long>(col) * K + bk]
                    : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // Skip wholly absent subtiles, preserving exactly ceil(K / 16) * 16
        // products, including dense_gemm_bf16's zero padding in its final tile.
        for (unsigned int sub = 0; sub < TK && k_base + sub < K; sub += TILE_K) {
            for (unsigned int kk = 0; kk < TILE_K; ++kk) {
                acc += __bfloat162float(smem_A[threadIdx.y][sub + kk])
                     * __bfloat162float(smem_B[sub + kk][threadIdx.x]);
            }
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[static_cast<unsigned long long>(row) * N + col] = __float2bfloat16(acc);
    }
}

// Fused SiLU(gate) * up activation — vectorized 2-wide BF16 loads/stores.
// Input: [N, inter_size*2] where first half is gate, second half is up.
// Output: [N, inter_size]
// out[i] = silu(gate[i]) * up[i]  where silu(x) = x * sigmoid(x)
extern "C" __global__ void fused_silu_mul(
    const __nv_bfloat16* __restrict__ gate_up,  // [num_tokens, inter_size * 2]
    __nv_bfloat16* __restrict__ output,          // [num_tokens, inter_size]
    unsigned int num_tokens,
    unsigned int inter_size
) {
    // Each thread processes 2 elements (vectorized BF16x2)
    unsigned long long idx2 = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long half_total = ((unsigned long long)num_tokens * inter_size) / 2;
    if (idx2 >= half_total) return;

    // Map linear index to (token, col_pair)
    unsigned int half_inter = inter_size / 2;
    unsigned long long token = idx2 / half_inter;
    unsigned int col_pair = (unsigned int)(idx2 % half_inter);

    // Vectorized loads: 2 BF16 per 32-bit read
    const unsigned int* gate32 = (const unsigned int*)(gate_up + token * ((unsigned long long)inter_size * 2));
    const unsigned int* up32 = (const unsigned int*)(gate_up + token * ((unsigned long long)inter_size * 2) + inter_size);
    unsigned int* out32 = (unsigned int*)(output + token * inter_size);

    unsigned int g_packed = gate32[col_pair];
    unsigned int u_packed = up32[col_pair];

    float g0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(g_packed & 0xFFFF)));
    float g1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(g_packed >> 16)));
    float u0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(u_packed & 0xFFFF)));
    float u1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(u_packed >> 16)));

    // SiLU(gate) * up for both elements
    float sg0 = 1.0f / (1.0f + __expf(-g0));
    float sg1 = 1.0f / (1.0f + __expf(-g1));
    float r0 = g0 * sg0 * u0;
    float r1 = g1 * sg1 * u1;

    // Vectorized store
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(r0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(r1));
    out32[col_pair] = lo | (hi << 16);
}
