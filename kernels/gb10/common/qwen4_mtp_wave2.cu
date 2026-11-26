// SPDX-License-Identifier: AGPL-3.0-only
//
// Wave 2 Decode-75 Kernels for Qwen3.8-Flash-Next-NVFP4 MTP Proposer (`qwen4_mtp.rs`):
//   1. `qwen4_mtp_embed_dual_rms_norm_fused`:
//      Fuses token embedding gather (from device argmax_ptr or host token_id) with
//      dual pre-FC RMSNorm (`pre_fc_norm_embedding` [1, H] + `pre_fc_norm_hidden` [1, 4*H])
//      in a single launch `grid=(1, 2, 1), block=(1024, 1, 1)`.
//      Eliminates `embed_from_argmax` / `copy_d2d_async` launch and global write+read of `s.embed`.
//
//   2. `qwen4_mtp_embed_dual_rms_norm_rows_packed`:
//      Multi-row twin for `forward_rows` (`rows = 1..=8`), taking up to 8 token IDs packed
//      in 4 `uint64_t` scalar kernel args (`tok_pack01..67`).
//      Eliminates the host loop of `rows` separate `copy_d2d_async` calls + intermediate `s.embed`
//      writes/reads, collapsing `rows + 1` driver calls into 1 launch `grid=(rows, 2, 1)`.
//
//   3. `qwen4_mtp_fc_dual_broadcast_rows`:
//      Fuses `fc_embedding` W4A16 GEMV (`[rows, 2560 -> 2560]`) +
//      `fc_hidden` W4A16 GEMV (`[rows*4, 2560 -> 2560]`) +
//      `bf16_residual_add_broadcast_hc` (`hidden[row*4 + br, n] += embed_proj[row, n]`)
//      into a SINGLE kernel launch `grid=(ceil(N/4), rows, 1), block=(256, 1, 1)`.
//      Preserves 100% bit-identical FP32/BF16 rounding to the 3-kernel baseline:
//        - Exact 64-lane `k16 = lane + 64*i` loop and `b = 0..7` (lo, hi) FMA sequence
//        - Exact 5-step warp shuffle + 2-warp `smem[0] + smem[1]` reduction
//        - Explicit intermediate `__float2bfloat16` rounding on both `embed_proj` and `hidden`
//          before the BF16 residual broadcast addition.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define GROUP_SIZE 16
#define BLOCK_SIZE 256
#define N_PER_BLOCK 4

__constant__ float E2M1_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

__device__ __forceinline__ void unpack_bf16x2(unsigned int packed, float &f0, float &f1) {
    __nv_bfloat16 b0, b1;
    unsigned short lo = (unsigned short)(packed & 0xFFFFu);
    unsigned short hi = (unsigned short)(packed >> 16);
    memcpy(&b0, &lo, 2);
    memcpy(&b1, &hi, 2);
    f0 = __bfloat162float(b0);
    f1 = __bfloat162float(b1);
}

__device__ __forceinline__ unsigned int pack_bf16x2(float f0, float f1) {
    __nv_bfloat16 b0 = __float2bfloat16(f0);
    __nv_bfloat16 b1 = __float2bfloat16(f1);
    unsigned short lo, hi;
    memcpy(&lo, &b0, 2);
    memcpy(&hi, &b1, 2);
    return ((unsigned int)hi << 16) | (unsigned int)lo;
}

// ============================================================================
// 1. Single-Step Fused Embedding Gather + Dual Pre-FC RMSNorm (`forward_one`)
// ============================================================================
// Grid:  (1, 2, 1)
// Block: (1024, 1, 1)
//   blockIdx.y == 0: gather embed_table[tok, 0..h] and RMSNorm -> normed_embed[0..h]
//   blockIdx.y == 1: RMSNorm hidden_in[0..r] -> normed_hidden[0..r]
extern "C" __global__ void __launch_bounds__(1024) qwen4_mtp_embed_dual_rms_norm_fused(
    const __nv_bfloat16* __restrict__ embed_table,      // [vocab_size, h]
    const unsigned int* __restrict__ token_id_ptr,      // device u32 ptr or nullptr
    unsigned int* __restrict__ token_id_out,            // optional device u32 out or nullptr
    unsigned int host_token_id,                         // used when token_id_ptr == nullptr
    unsigned int vocab_size,
    const __nv_bfloat16* __restrict__ embed_norm_weight, // [h]
    __nv_bfloat16* __restrict__ normed_embed,           // [h]
    const __nv_bfloat16* __restrict__ hidden_in,        // [r] (r = hc * h = 10240)
    const __nv_bfloat16* __restrict__ hidden_norm_weight, // [r]
    __nv_bfloat16* __restrict__ normed_hidden,          // [r]
    unsigned int h,
    unsigned int r,
    float eps
) {
    const unsigned int kind = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x;
    const __nv_bfloat16* weight;
    __nv_bfloat16* out;
    unsigned int width;

    if (kind == 0u) {
        unsigned int tok = (token_id_ptr != nullptr) ? token_id_ptr[0] : host_token_id;
        if (tok >= vocab_size) tok = 0u;
        if (tid == 0u && token_id_out != nullptr) {
            *token_id_out = tok;
        }
        x = embed_table + (unsigned long long)tok * h;
        weight = embed_norm_weight;
        out = normed_embed;
        width = h;
    } else {
        x = hidden_in;
        weight = hidden_norm_weight;
        out = normed_hidden;
        width = r;
    }

    float sum_sq = 0.0f;
    const unsigned int half_size = width / 2u;
    const bool vec_aligned = (((reinterpret_cast<unsigned long long>(x)
                              | reinterpret_cast<unsigned long long>(weight)
                              | reinterpret_cast<unsigned long long>(out)) & 3ull) == 0ull);
    const unsigned int* x32 = (const unsigned int*)x;
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    if (vec_aligned && half_size <= 5120u && blockDim.x == 1024u) {
        float xv0[5], xv1[5];
        #pragma unroll
        for (int k = 0; k < 5; ++k) {
            const unsigned int i = tid + (unsigned int)k * 1024u;
            if (i < half_size) {
                unpack_bf16x2(__ldg(x32 + i), xv0[k], xv1[k]);
                sum_sq += xv0[k] * xv0[k] + xv1[k] * xv1[k];
            } else {
                xv0[k] = 0.0f;
                xv1[k] = 0.0f;
            }
        }
        if ((width & 1u) && tid == 0u) {
            float val = __bfloat162float(__ldg(x + width - 1u));
            sum_sq += val * val;
        }

        sum_sq = warp_reduce_sum(sum_sq);

        __shared__ float warp_sums[32];
        const unsigned int warp_id = tid / 32u;
        const unsigned int lane_id = tid % 32u;

        if (lane_id == 0u) {
            warp_sums[warp_id] = sum_sq;
        }
        __syncthreads();

        if (warp_id == 0u) {
            float val = warp_sums[lane_id];
            val = warp_reduce_sum(val);
            if (lane_id == 0u) {
                warp_sums[0] = val;
            }
        }
        __syncthreads();

        const float rms = rsqrtf(warp_sums[0] / (float)width + eps);
        #pragma unroll
        for (int k = 0; k < 5; ++k) {
            const unsigned int i = tid + (unsigned int)k * 1024u;
            if (i < half_size) {
                float wv0, wv1;
                unpack_bf16x2(__ldg(w32 + i), wv0, wv1);
                out32[i] = pack_bf16x2(xv0[k] * rms * (1.0f + wv0), xv1[k] * rms * (1.0f + wv1));
            }
        }
        if ((width & 1u) && tid == 0u) {
            float val = __bfloat162float(__ldg(x + width - 1u));
            float w = __bfloat162float(__ldg(weight + width - 1u));
            out[width - 1u] = __float2bfloat16(val * rms * (1.0f + w));
        }
        return;
    }

    if (vec_aligned) {
        for (unsigned int i = tid; i < half_size; i += blockDim.x) {
            float v0, v1;
            unpack_bf16x2(__ldg(x32 + i), v0, v1);
            sum_sq += v0 * v0 + v1 * v1;
        }
        if ((width & 1u) && tid == 0u) {
            float val = __bfloat162float(__ldg(x + width - 1u));
            sum_sq += val * val;
        }
    } else {
        for (unsigned int i = tid; i < width; i += blockDim.x) {
            float val = __bfloat162float(__ldg(x + i));
            sum_sq += val * val;
        }
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    const unsigned int warp_id = tid / 32u;
    const unsigned int lane_id = tid % 32u;

    if (lane_id == 0u) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0u) {
        float val = (lane_id < (blockDim.x + 31u) / 32u) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0u) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    const float rms = rsqrtf(warp_sums[0] / (float)width + eps);

    if (vec_aligned) {
        for (unsigned int i = tid; i < half_size; i += blockDim.x) {
            float xv0, xv1, wv0, wv1;
            unpack_bf16x2(__ldg(x32 + i), xv0, xv1);
            unpack_bf16x2(__ldg(w32 + i), wv0, wv1);
            out32[i] = pack_bf16x2(xv0 * rms * (1.0f + wv0), xv1 * rms * (1.0f + wv1));
        }
        if ((width & 1u) && tid == 0u) {
            float val = __bfloat162float(__ldg(x + width - 1u));
            float w = __bfloat162float(__ldg(weight + width - 1u));
            out[width - 1u] = __float2bfloat16(val * rms * (1.0f + w));
        }
    } else {
        for (unsigned int i = tid; i < width; i += blockDim.x) {
            float val = __bfloat162float(__ldg(x + i));
            float w = __bfloat162float(__ldg(weight + i));
            out[i] = __float2bfloat16(val * rms * (1.0f + w));
        }
    }
}

// ============================================================================
// 2. Multi-Row Fused Embedding Gather + Dual Pre-FC RMSNorm (`forward_rows`)
// ============================================================================
// Grid:  (rows, 2, 1)  where rows in 1..=8
// Block: (1024, 1, 1)
extern "C" __global__ void __launch_bounds__(1024) qwen4_mtp_embed_dual_rms_norm_rows_packed(
    const __nv_bfloat16* __restrict__ embed_table,      // [vocab_size, h]
    unsigned long long tok_pack01,                      // u32 tok0 | (u64(tok1) << 32)
    unsigned long long tok_pack23,                      // u32 tok2 | (u64(tok3) << 32)
    unsigned long long tok_pack45,                      // u32 tok4 | (u64(tok5) << 32)
    unsigned long long tok_pack67,                      // u32 tok6 | (u64(tok7) << 32)
    unsigned int vocab_size,
    const __nv_bfloat16* __restrict__ embed_norm_weight, // [h]
    __nv_bfloat16* __restrict__ normed_embed,           // [rows, h]
    const __nv_bfloat16* __restrict__ hidden_in,        // [rows, r]
    const __nv_bfloat16* __restrict__ hidden_norm_weight, // [r]
    __nv_bfloat16* __restrict__ normed_hidden,          // [rows, r]
    unsigned int h,
    unsigned int r,
    float eps,
    const unsigned long long* __restrict__ hidden_in_ptrs
) {
    const unsigned int row = blockIdx.x;
    const unsigned int kind = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x;
    const __nv_bfloat16* weight;
    __nv_bfloat16* out;
    unsigned int width;

    if (kind == 0u) {
        unsigned int tok = 0u;
        switch (row) {
            case 0u: tok = (unsigned int)(tok_pack01 & 0xFFFFFFFFull); break;
            case 1u: tok = (unsigned int)(tok_pack01 >> 32); break;
            case 2u: tok = (unsigned int)(tok_pack23 & 0xFFFFFFFFull); break;
            case 3u: tok = (unsigned int)(tok_pack23 >> 32); break;
            case 4u: tok = (unsigned int)(tok_pack45 & 0xFFFFFFFFull); break;
            case 5u: tok = (unsigned int)(tok_pack45 >> 32); break;
            case 6u: tok = (unsigned int)(tok_pack67 & 0xFFFFFFFFull); break;
            default: tok = (unsigned int)(tok_pack67 >> 32); break;
        }
        if (tok >= vocab_size) tok = 0u;
        x = embed_table + (unsigned long long)tok * h;
        weight = embed_norm_weight;
        out = normed_embed + (unsigned long long)row * h;
        width = h;
    } else {
        x = (hidden_in_ptrs != nullptr)
            ? (const __nv_bfloat16*)(hidden_in_ptrs[row])
            : (hidden_in + (unsigned long long)row * r);
        weight = hidden_norm_weight;
        out = normed_hidden + (unsigned long long)row * r;
        width = r;
    }

    float sum_sq = 0.0f;
    const unsigned int half_size = width / 2u;
    const bool vec_aligned = (((reinterpret_cast<unsigned long long>(x)
                              | reinterpret_cast<unsigned long long>(weight)
                              | reinterpret_cast<unsigned long long>(out)) & 3ull) == 0ull);
    const unsigned int* x32 = (const unsigned int*)x;
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    if (vec_aligned && half_size <= 5120u && blockDim.x == 1024u) {
        float xv0[5], xv1[5];
        #pragma unroll
        for (int k = 0; k < 5; ++k) {
            const unsigned int i = tid + (unsigned int)k * 1024u;
            if (i < half_size) {
                unpack_bf16x2(__ldg(x32 + i), xv0[k], xv1[k]);
                sum_sq += xv0[k] * xv0[k] + xv1[k] * xv1[k];
            } else {
                xv0[k] = 0.0f;
                xv1[k] = 0.0f;
            }
        }
        if ((width & 1u) && tid == 0u) {
            float val = __bfloat162float(__ldg(x + width - 1u));
            sum_sq += val * val;
        }

        sum_sq = warp_reduce_sum(sum_sq);

        __shared__ float warp_sums[32];
        const unsigned int warp_id = tid / 32u;
        const unsigned int lane_id = tid % 32u;

        if (lane_id == 0u) {
            warp_sums[warp_id] = sum_sq;
        }
        __syncthreads();

        if (warp_id == 0u) {
            float val = warp_sums[lane_id];
            val = warp_reduce_sum(val);
            if (lane_id == 0u) {
                warp_sums[0] = val;
            }
        }
        __syncthreads();

        const float rms = rsqrtf(warp_sums[0] / (float)width + eps);
        #pragma unroll
        for (int k = 0; k < 5; ++k) {
            const unsigned int i = tid + (unsigned int)k * 1024u;
            if (i < half_size) {
                float wv0, wv1;
                unpack_bf16x2(__ldg(w32 + i), wv0, wv1);
                out32[i] = pack_bf16x2(xv0[k] * rms * (1.0f + wv0), xv1[k] * rms * (1.0f + wv1));
            }
        }
        if ((width & 1u) && tid == 0u) {
            float val = __bfloat162float(__ldg(x + width - 1u));
            float w = __bfloat162float(__ldg(weight + width - 1u));
            out[width - 1u] = __float2bfloat16(val * rms * (1.0f + w));
        }
        return;
    }

    if (vec_aligned) {
        for (unsigned int i = tid; i < half_size; i += blockDim.x) {
            float v0, v1;
            unpack_bf16x2(__ldg(x32 + i), v0, v1);
            sum_sq += v0 * v0 + v1 * v1;
        }
        if ((width & 1u) && tid == 0u) {
            float val = __bfloat162float(__ldg(x + width - 1u));
            sum_sq += val * val;
        }
    } else {
        for (unsigned int i = tid; i < width; i += blockDim.x) {
            float val = __bfloat162float(__ldg(x + i));
            sum_sq += val * val;
        }
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    const unsigned int warp_id = tid / 32u;
    const unsigned int lane_id = tid % 32u;

    if (lane_id == 0u) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0u) {
        float val = (lane_id < (blockDim.x + 31u) / 32u) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0u) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    const float rms = rsqrtf(warp_sums[0] / (float)width + eps);

    if (vec_aligned) {
        for (unsigned int i = tid; i < half_size; i += blockDim.x) {
            float xv0, xv1, wv0, wv1;
            unpack_bf16x2(__ldg(x32 + i), xv0, xv1);
            unpack_bf16x2(__ldg(w32 + i), wv0, wv1);
            out32[i] = pack_bf16x2(xv0 * rms * (1.0f + wv0), xv1 * rms * (1.0f + wv1));
        }
        if ((width & 1u) && tid == 0u) {
            float val = __bfloat162float(__ldg(x + width - 1u));
            float w = __bfloat162float(__ldg(weight + width - 1u));
            out[width - 1u] = __float2bfloat16(val * rms * (1.0f + w));
        }
    } else {
        for (unsigned int i = tid; i < width; i += blockDim.x) {
            float val = __bfloat162float(__ldg(x + i));
            float w = __bfloat162float(__ldg(weight + i));
            out[i] = __float2bfloat16(val * rms * (1.0f + w));
        }
    }
}

// ============================================================================
// 3. Fused Dual FC W4A16 GEMV + HC=4 Broadcast Residual Add (`rows = 1..=8`)
// ============================================================================
// Computes both:
//   embed_proj[row, n] = w4a16_gemv(normed_embed[row], B_embed, n)
//   hidden_proj[row*4 + br, n] = w4a16_gemv(normed_hidden[row*4 + br], B_hidden, n)
// and writes:
//   hidden_out[(row*4 + br)*N + n] = __float2bfloat16(
//       __bfloat162float(hidden_proj[row*4 + br, n]) + __bfloat162float(embed_proj[row, n]))
//
// Grid:  (ceil(N / 4), rows, 1)
// Block: (256, 1, 1)
extern "C" __global__ void __launch_bounds__(256, 4) qwen4_mtp_fc_dual_broadcast_rows(
    const __nv_bfloat16* __restrict__ normed_embed,     // [rows, K]
    const unsigned char* __restrict__ B_embed_packed,   // [N, K/2]
    const unsigned char* __restrict__ B_embed_scale,    // [N, K/16]
    const float scale2_embed,
    const __nv_bfloat16* __restrict__ normed_hidden,    // [rows * 4, K]
    const unsigned char* __restrict__ B_hidden_packed,  // [N, K/2]
    const unsigned char* __restrict__ B_hidden_scale,   // [N, K/16]
    const float scale2_hidden,
    __nv_bfloat16* __restrict__ hidden_out,             // [rows * 4, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK; // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;  // 0..3
    const unsigned int lane = threadIdx.x % threads_per_out;       // 0..63
    const unsigned int warp_lane = lane % WARP_SIZE;               // 0..31
    const unsigned int warp_idx = lane / WARP_SIZE;                // 0..1
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const unsigned int row = blockIdx.y;
    const bool valid = (n < N);

    __shared__ float s_lut[16];
    __shared__ float smem_e[N_PER_BLOCK * 2];
    __shared__ float smem_h[4 * N_PER_BLOCK * 2];

    if (threadIdx.x < 16u) {
        s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    }
    __syncthreads();

    float acc_e = 0.0f;
    float acc_h0 = 0.0f;
    float acc_h1 = 0.0f;
    float acc_h2 = 0.0f;
    float acc_h3 = 0.0f;

    if (valid) {
        const unsigned int half_K = K / 2u;
        const unsigned int num_groups = K / GROUP_SIZE;
        const unsigned int K16 = K / 16u;
        const unsigned long long w_row_off = (unsigned long long)n * half_K;
        const unsigned long long s_row_off = (unsigned long long)n * num_groups;

        const __nv_bfloat16* __restrict__ A_e = normed_embed + (unsigned long long)row * K;
        const __nv_bfloat16* __restrict__ A_h0 = normed_hidden + (unsigned long long)(row * 4u + 0u) * K;
        const __nv_bfloat16* __restrict__ A_h1 = normed_hidden + (unsigned long long)(row * 4u + 1u) * K;
        const __nv_bfloat16* __restrict__ A_h2 = normed_hidden + (unsigned long long)(row * 4u + 2u) * K;
        const __nv_bfloat16* __restrict__ A_h3 = normed_hidden + (unsigned long long)(row * 4u + 3u) * K;

        for (unsigned int k16 = lane; k16 < K16; k16 += 64u) {
            const unsigned int sg = (k16 * 16u) / GROUP_SIZE;

            // --- 1. fc_embedding accumulate ---
            const unsigned long long pk_e = *(const unsigned long long*)(B_embed_packed + w_row_off + k16 * 8u);
            const unsigned char sb_e = B_embed_scale[s_row_off + sg];
            __nv_fp8_e4m3 fp8_e;
            *(unsigned char*)&fp8_e = sb_e;
            const float scale_e = (float)fp8_e * scale2_embed;

            const uint4 ae_lo = __ldg(&((const uint4*)A_e)[k16 * 2u]);
            const uint4 ae_hi = __ldg(&((const uint4*)A_e)[k16 * 2u + 1u]);
            const unsigned int ae_raw[8] = {
                ae_lo.x, ae_lo.y, ae_lo.z, ae_lo.w,
                ae_hi.x, ae_hi.y, ae_hi.z, ae_hi.w
            };

            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                const unsigned char byte_e = (unsigned char)(pk_e >> (b * 8));
                const float we_lo = s_lut[byte_e & 0xFu] * scale_e;
                const float we_hi = s_lut[byte_e >> 4] * scale_e;
                __nv_bfloat16 a_lo_bf, a_hi_bf;
                *(unsigned short*)&a_lo_bf = (unsigned short)(ae_raw[b] & 0xFFFFu);
                *(unsigned short*)&a_hi_bf = (unsigned short)(ae_raw[b] >> 16);
                acc_e += __bfloat162float(a_lo_bf) * we_lo;
                acc_e += __bfloat162float(a_hi_bf) * we_hi;
            }

            // --- 2. fc_hidden accumulate across 4 hyper-connection branches ---
            const unsigned long long pk_h = *(const unsigned long long*)(B_hidden_packed + w_row_off + k16 * 8u);
            const unsigned char sb_h = B_hidden_scale[s_row_off + sg];
            __nv_fp8_e4m3 fp8_h;
            *(unsigned char*)&fp8_h = sb_h;
            const float scale_h = (float)fp8_h * scale2_hidden;

            float wh_lo[8], wh_hi[8];
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                const unsigned char byte_h = (unsigned char)(pk_h >> (b * 8));
                wh_lo[b] = s_lut[byte_h & 0xFu] * scale_h;
                wh_hi[b] = s_lut[byte_h >> 4] * scale_h;
            }

            #define ACCUM_HIDDEN_BRANCH(A_PTR, ACC_VAR)                                      \
                do {                                                                          \
                    const uint4 ah_lo = __ldg(&((const uint4*)(A_PTR))[k16 * 2u]);            \
                    const uint4 ah_hi = __ldg(&((const uint4*)(A_PTR))[k16 * 2u + 1u]);       \
                    const unsigned int ah_raw[8] = {                                          \
                        ah_lo.x, ah_lo.y, ah_lo.z, ah_lo.w,                                   \
                        ah_hi.x, ah_hi.y, ah_hi.z, ah_hi.w                                    \
                    };                                                                        \
                    _Pragma("unroll")                                                         \
                    for (int b = 0; b < 8; ++b) {                                             \
                        __nv_bfloat16 a_lo_bf, a_hi_bf;                                       \
                        *(unsigned short*)&a_lo_bf = (unsigned short)(ah_raw[b] & 0xFFFFu);   \
                        *(unsigned short*)&a_hi_bf = (unsigned short)(ah_raw[b] >> 16);       \
                        (ACC_VAR) += __bfloat162float(a_lo_bf) * wh_lo[b];                    \
                        (ACC_VAR) += __bfloat162float(a_hi_bf) * wh_hi[b];                    \
                    }                                                                         \
                } while (0)

            ACCUM_HIDDEN_BRANCH(A_h0, acc_h0);
            ACCUM_HIDDEN_BRANCH(A_h1, acc_h1);
            ACCUM_HIDDEN_BRANCH(A_h2, acc_h2);
            ACCUM_HIDDEN_BRANCH(A_h3, acc_h3);

            #undef ACCUM_HIDDEN_BRANCH
        }
    }

    // 5-step warp shuffle reduction (identical to w4a16_gemv_k1 & exact_m4)
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc_e  += __shfl_down_sync(0xFFFFFFFFu, acc_e,  offset);
        acc_h0 += __shfl_down_sync(0xFFFFFFFFu, acc_h0, offset);
        acc_h1 += __shfl_down_sync(0xFFFFFFFFu, acc_h1, offset);
        acc_h2 += __shfl_down_sync(0xFFFFFFFFu, acc_h2, offset);
        acc_h3 += __shfl_down_sync(0xFFFFFFFFu, acc_h3, offset);
    }

    if (warp_lane == 0u) {
        smem_e[local_out * 2u + warp_idx] = acc_e;
        smem_h[(0u * N_PER_BLOCK + local_out) * 2u + warp_idx] = acc_h0;
        smem_h[(1u * N_PER_BLOCK + local_out) * 2u + warp_idx] = acc_h1;
        smem_h[(2u * N_PER_BLOCK + local_out) * 2u + warp_idx] = acc_h2;
        smem_h[(3u * N_PER_BLOCK + local_out) * 2u + warp_idx] = acc_h3;
    }
    __syncthreads();

    if (valid && lane == 0u) {
        // Round embed_proj and hidden_proj to BF16 first so the broadcast addition
        // is bit-for-bit identical to running `bf16_residual_add_broadcast_hc` after
        // separate `fc_embedding` and `fc_hidden` GEMVs.
        const __nv_bfloat16 e_bf16 = __float2bfloat16(
            smem_e[local_out * 2u] + smem_e[local_out * 2u + 1u]);
        const float e_f32 = __bfloat162float(e_bf16);

        #pragma unroll
        for (unsigned int br = 0u; br < 4u; ++br) {
            const unsigned int base = (br * N_PER_BLOCK + local_out) * 2u;
            const __nv_bfloat16 h_bf16 = __float2bfloat16(smem_h[base] + smem_h[base + 1u]);
            hidden_out[(unsigned long long)(row * 4u + br) * N + n] =
                __float2bfloat16(__bfloat162float(h_bf16) + e_f32);
        }
    }
}
