// SPDX-License-Identifier: AGPL-3.0-only

// M8/M17 exact-FFN fast path with CTA-cooperative shared-memory FP32 activation
// staging on M17 down_proj + optional fused residual add (Decode-75 Q2).
//
// Why this matters on Qwen3.8-27B DFlash K=16/17 verify:
// In `w4a16_gemv_f32_input_exact_m17`, the input `A` (`preactivation_f32`) has
// shape `[17, 17408]` FP32 = 1.18 MB. That exceeds L1 (128 KB) and competes
// with the 49.9 MB `down_proj` weight matrix in GB10's 16 MB L2 cache across
// `grid.x = 5120 / 4 = 1,280` CTAs.
// Without shared-memory staging, all 4 output groups (`N_PER_BLOCK = 4`) in each
// CTA independently fetch the 1.18 MB FP32 tensor from L2/DRAM (4x overfetch =
// 6.06 GB/layer = 388 GB across 64 layers per verify step).
//
// By cooperatively staging each 64-chunk wave (`64 * 8 = 512` floats/row,
// `17 * 64 * 2 * sizeof(float4) = 34,816 B` shared memory) into `s_a`, all 4
// output groups share a single coalesced load per CTA while preserving 100%
// bit-identical FP32 FMA order and warp reduction.

#ifndef W4A16_F32_INPUT_K8_STAGE
#define W4A16_F32_INPUT_K8_STAGE 64u
#endif
static_assert(W4A16_F32_INPUT_K8_STAGE == 64u,
              "W4A16_F32_INPUT_K8_STAGE must equal 64u (1 chunk per lane per wave)");

template <int MAX_M>
__device__ __forceinline__ void w4a16_dual_silu_f32_exact_body(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    float* __restrict__ activation,
    unsigned int M, unsigned int K)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = M <= (unsigned int)MAX_M ? M * K : 0u;
    if (idx >= elements) return;

    // These BF16 loads are the required projection-output rounding boundary.
    const __nv_bfloat16 gate_bf16 = gate_out[idx];
    const __nv_bfloat16 up_bf16 = up_out[idx];
    const float gate = __bfloat162float(gate_bf16);
    const float up = __bfloat162float(up_bf16);
    activation[idx] = (gate / (1.0f + __expf(-gate))) * up;
}

template <int MAX_M>
__device__ __forceinline__ void w4a16_f32_input_exact_body(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    float* __restrict__ s_lut,
    float* __restrict__ smem)
{
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = lane % WARP_SIZE;
    const unsigned int warp_idx = lane / WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = M > 0u && M <= (unsigned int)MAX_M && n < N;

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_W4[threadIdx.x];
    __syncthreads();

    float acc[MAX_M];
    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) acc[row] = 0.0f;

    if (valid) {
        const unsigned int half_K = K / 2;
        const unsigned int num_groups = K / GROUP_SIZE;
        const unsigned int K8 = K / 8;
        const unsigned long long weight_row = (unsigned long long)n * half_K;
        const unsigned long long scale_row = (unsigned long long)n * num_groups;

        unsigned int packed4 = 0u;
        unsigned char scale_byte = 0u;
        if (lane < K8) {
            packed4 = *(const unsigned int*)(
                B_packed + weight_row + (unsigned long long)lane * 4);
            scale_byte = B_scale[scale_row
                + ((unsigned long long)lane * 8) / GROUP_SIZE];
        }
        for (unsigned int k8 = lane; k8 < K8; k8 += 64u) {
            const unsigned int next_k8 = k8 + 64u;
            unsigned int next_packed4 = 0u;
            unsigned char next_scale_byte = 0u;
            if (next_k8 < K8) {
                next_packed4 = *(const unsigned int*)(
                    B_packed + weight_row + (unsigned long long)next_k8 * 4);
                next_scale_byte = B_scale[scale_row
                    + ((unsigned long long)next_k8 * 8) / GROUP_SIZE];
            }
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

            #pragma unroll
            for (int row = 0; row < MAX_M; ++row) {
                if ((unsigned int)row < M) {
                    const float* A_row = A + (unsigned long long)row * K;
                    const float4 a_lo = ((const float4*)A_row)[k8 * 2];
                    const float4 a_hi = ((const float4*)A_row)[k8 * 2 + 1];
                    const float a[8] = {
                        a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                        a_hi.x, a_hi.y, a_hi.z, a_hi.w
                    };
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        acc[row] += a[b * 2] * w_lo[b];
                        acc[row] += a[b * 2 + 1] * w_hi[b];
                    }
                }
            }
            packed4 = next_packed4;
            scale_byte = next_scale_byte;
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
        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) {
            if ((unsigned int)row < M) {
                const unsigned int base = (row * N_PER_BLOCK + local_out) * 2;
                C[(unsigned long long)row * N + n] =
                    __float2bfloat16(smem[base] + smem[base + 1]);
            }
        }
    }
}

// CTA-cooperative shared-memory staged FP32 input GEMV for M<=17 (`down_proj`).
// Stages 64 k8-chunks (`512` floats/row = `128` float4s/row x 17 rows = 34 KB)
// per wave so the 4 output groups in the CTA read `A[17, 17408]` once per CTA
// instead of 4x. When `FUSE_RESIDUAL` is true and `residual != nullptr`, adds
// `residual[row * N + n]` (aliasing `residual == C` is supported) without a
// separate elementwise kernel launch.
template <int MAX_M, bool FUSE_RESIDUAL>
__device__ __forceinline__ void w4a16_f32_input_exact_astage_body(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    const __nv_bfloat16* residual,
    __nv_bfloat16* C,
    unsigned int M, unsigned int N, unsigned int K,
    float* __restrict__ s_lut,
    float* __restrict__ smem,
    float4* __restrict__ s_a)
{
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = lane % WARP_SIZE;
    const unsigned int warp_idx = lane / WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool rows_valid = M > 0u && M <= (unsigned int)MAX_M;
    const bool valid = rows_valid && n < N;

    if (threadIdx.x < 16u) s_lut[threadIdx.x] = E2M1_LUT_FUSED_W4[threadIdx.x];
    __syncthreads();

    float acc[MAX_M];
    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) acc[row] = 0.0f;

    const unsigned int half_K = K / 2u;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8u;
    const unsigned long long weight_row = (unsigned long long)n * half_K;
    const unsigned long long scale_row = (unsigned long long)n * num_groups;
    const unsigned int f4_per_row = W4A16_F32_INPUT_K8_STAGE * 2u; // 128 float4s = 512 floats
    const unsigned int stage_f4 = rows_valid ? (M * f4_per_row) : 0u;

    unsigned int packed4 = 0u;
    unsigned char scale_byte = 0u;
    if (valid && lane < K8) {
        packed4 = *(const unsigned int*)(
            B_packed + weight_row + (unsigned long long)lane * 4u);
        scale_byte = B_scale[scale_row
            + ((unsigned long long)lane * 8u) / GROUP_SIZE];
    }

    for (unsigned int k8_base = 0u; k8_base < K8; k8_base += W4A16_F32_INPUT_K8_STAGE) {
        const unsigned int k8 = k8_base + lane;
        const unsigned int next_k8 = k8 + W4A16_F32_INPUT_K8_STAGE;

        unsigned int next_packed4 = 0u;
        unsigned char next_scale_byte = 0u;
        if (valid && next_k8 < K8) {
            next_packed4 = *(const unsigned int*)(
                B_packed + weight_row + (unsigned long long)next_k8 * 4u);
            next_scale_byte = B_scale[scale_row
                + ((unsigned long long)next_k8 * 8u) / GROUP_SIZE];
        }

        // Cooperatively stage M * 128 float4s (34 KB for M=17).
        for (unsigned int idx = threadIdx.x; idx < stage_f4; idx += BLOCK_SIZE) {
            const unsigned int row = idx / f4_per_row;
            const unsigned int f4_idx = idx % f4_per_row;
            const unsigned int gk8 = k8_base + (f4_idx >> 1);
            if (gk8 < K8) {
                const float* A_row = A + (unsigned long long)row * K;
                s_a[idx] = ((const float4*)A_row)[k8_base * 2u + f4_idx];
            } else {
                s_a[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        if (valid && k8 < K8) {
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

            const unsigned int lane_f4 = lane * 2u;
            #pragma unroll
            for (int row = 0; row < MAX_M; ++row) {
                if ((unsigned int)row < M) {
                    const float4 a_lo = s_a[(unsigned int)row * f4_per_row + lane_f4];
                    const float4 a_hi = s_a[(unsigned int)row * f4_per_row + lane_f4 + 1u];
                    acc[row] += a_lo.x * w_lo[0];
                    acc[row] += a_lo.y * w_hi[0];
                    acc[row] += a_lo.z * w_lo[1];
                    acc[row] += a_lo.w * w_hi[1];
                    acc[row] += a_hi.x * w_lo[2];
                    acc[row] += a_hi.y * w_hi[2];
                    acc[row] += a_hi.z * w_lo[3];
                    acc[row] += a_hi.w * w_hi[3];
                }
            }
        }

        packed4 = next_packed4;
        scale_byte = next_scale_byte;
        __syncthreads();
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
        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) {
            if ((unsigned int)row < M) {
                const unsigned int base = (row * N_PER_BLOCK + local_out) * 2;
                const unsigned long long out_idx = (unsigned long long)row * N + n;
                const __nv_bfloat16 down_bf16 =
                    __float2bfloat16(smem[base] + smem[base + 1]);
                if (FUSE_RESIDUAL && residual != nullptr) {
                    const float res_f = __bfloat162float(residual[out_idx]);
                    const float down_f = __bfloat162float(down_bf16);
                    C[out_idx] = __float2bfloat16(res_f + down_f);
                } else {
                    C[out_idx] = down_bf16;
                }
            }
        }
    }
}

#define DEFINE_W4A16_DUAL_SILU_F32_EXACT(NAME, MAX_ROWS)                    \
extern "C" __global__ void NAME(                                             \
    const __nv_bfloat16* gate_out, const __nv_bfloat16* up_out,              \
    float* activation, unsigned int M, unsigned int K)                       \
{                                                                            \
    w4a16_dual_silu_f32_exact_body<MAX_ROWS>(                                \
        gate_out, up_out, activation, M, K);                                 \
}

DEFINE_W4A16_DUAL_SILU_F32_EXACT(
    w4a16_gemv_dual_silu_f32_exact_m8, 8)
DEFINE_W4A16_DUAL_SILU_F32_EXACT(
    w4a16_gemv_dual_silu_f32_exact_m17, 17)

extern "C" __global__ void w4a16_gemv_f32_input_exact_m8(
    const float* A, const unsigned char* B_packed,
    const unsigned char* B_scale, const float scale2,
    __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K)
{
    __shared__ float s_lut[16];
    __shared__ float smem[8 * N_PER_BLOCK * 2];
    w4a16_f32_input_exact_body<8>(
        A, B_packed, B_scale, scale2, C, M, N, K, s_lut, smem);
}

extern "C" __global__ void __launch_bounds__(256)
w4a16_gemv_f32_input_exact_m17(
    const float* A, const unsigned char* B_packed,
    const unsigned char* B_scale, const float scale2,
    __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K)
{
    __shared__ float s_lut[16];
    __shared__ float smem[17 * N_PER_BLOCK * 2];
    __shared__ float4 s_a[17 * W4A16_F32_INPUT_K8_STAGE * 2];
    w4a16_f32_input_exact_astage_body<17, false>(
        A, B_packed, B_scale, scale2, nullptr, C, M, N, K, s_lut, smem, s_a);
}

extern "C" __global__ void __launch_bounds__(256)
w4a16_gemv_f32_input_exact_m17_astage(
    const float* A, const unsigned char* B_packed,
    const unsigned char* B_scale, const float scale2,
    __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K)
{
    __shared__ float s_lut[16];
    __shared__ float smem[17 * N_PER_BLOCK * 2];
    __shared__ float4 s_a[17 * W4A16_F32_INPUT_K8_STAGE * 2];
    w4a16_f32_input_exact_astage_body<17, false>(
        A, B_packed, B_scale, scale2, nullptr, C, M, N, K, s_lut, smem, s_a);
}

extern "C" __global__ void __launch_bounds__(256)
w4a16_gemv_f32_input_exact_residual_m17_astage(
    const float* A, const unsigned char* B_packed,
    const unsigned char* B_scale, const float scale2,
    const __nv_bfloat16* residual,
    __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K)
{
    __shared__ float s_lut[16];
    __shared__ float smem[17 * N_PER_BLOCK * 2];
    __shared__ float4 s_a[17 * W4A16_F32_INPUT_K8_STAGE * 2];
    w4a16_f32_input_exact_astage_body<17, true>(
        A, B_packed, B_scale, scale2, residual, C, M, N, K, s_lut, smem, s_a);
}

#undef DEFINE_W4A16_DUAL_SILU_F32_EXACT
