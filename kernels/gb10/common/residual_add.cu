// SPDX-License-Identifier: AGPL-3.0-only

// BF16 residual addition: residual[i] += src[i]
//
// Used in the transformer loop after attention/SSM and FFN blocks.
// Operates in-place on the residual stream.

#include <cuda_bf16.h>

extern "C" __global__ void bf16_residual_add(
    __nv_bfloat16* __restrict__ residual,  // [n] — modified in-place
    const __nv_bfloat16* __restrict__ src,  // [n]
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float r = __bfloat162float(residual[i]);
        float s = __bfloat162float(src[i]);
        residual[i] = __float2bfloat16(r + s);
    }
}

// BF16 → FP32 element-wise conversion.
// Used to promote embedding output (BF16) to FP32 residual stream.
extern "C" __global__ void bf16_to_f32(
    const __nv_bfloat16* __restrict__ src,  // [n] BF16
    float* __restrict__ dst,                 // [n] FP32
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __bfloat162float(src[i]);
    }
}

// Separate-input SiLU multiplication: output = SiLU(gate) * up.
//
// Unlike fused_silu_mul in dense_gemm_bf16.cu which expects interleaved
// gate_up, this takes separate gate and up buffers. Used by MoE shared
// expert which has separate gate_proj and up_proj.
extern "C" __global__ void silu_mul_separate(
    const __nv_bfloat16* __restrict__ gate,   // [n]
    const __nv_bfloat16* __restrict__ up,     // [n]
    __nv_bfloat16* __restrict__ output,        // [n]
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        // SiLU(x) = x * sigmoid(x)
        float silu_g = g / (1.0f + __expf(-g));
        output[i] = __float2bfloat16(silu_g * u);
    }
}

// Scaled accumulate: output[i] += scale * src[i] (BF16 with f32 scale).
//
// Used for MoE routing: accumulate weighted expert outputs.
extern "C" __global__ void bf16_scaled_add(
    __nv_bfloat16* __restrict__ output,    // [n] — modified in-place
    const __nv_bfloat16* __restrict__ src,  // [n]
    float scale,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float o = __bfloat162float(output[i]);
        float s = __bfloat162float(src[i]);
        output[i] = __float2bfloat16(o + scale * s);
    }
}

// Sigmoid-gated blend: output = sigmoid_gate * src + output (BF16).
//
// Used for MoE shared expert: blend shared expert output with routed output.
// sigmoid_gate is a single f32 value applied to all elements.
extern "C" __global__ void bf16_sigmoid_blend(
    __nv_bfloat16* __restrict__ output,    // [n] — routed output, modified in-place
    const __nv_bfloat16* __restrict__ src,  // [n] — shared expert output
    float sigmoid_gate,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float o = __bfloat162float(output[i]);
        float s = __bfloat162float(src[i]);
        output[i] = __float2bfloat16(o + sigmoid_gate * s);
    }
}

// Element-wise sigmoid gate: output[i] = input[i] * sigmoid(gate[i]).
//
// Used for gated attention: attn_output = attn_output * sigmoid(q_gate).
// gate and input are separate BF16 vectors of the same size.
extern "C" __global__ void sigmoid_gate_mul(
    const __nv_bfloat16* __restrict__ input,   // [n]
    const __nv_bfloat16* __restrict__ gate,    // [n]
    __nv_bfloat16* __restrict__ output,        // [n]
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(gate[i]);
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = __float2bfloat16(x * sigmoid_g);
    }
}

// Batched sigmoid gate multiply across multiple tokens.
// Replaces per-token kernel launches with a single launch for all tokens.
//
// input is contiguous [num_tokens, dim]. gate is strided [num_tokens, gate_stride]
// (gate data starts at gate pointer, with gate_stride elements between tokens).
// output[t * dim + d] = input[t * dim + d] * sigmoid(gate[t * gate_stride + d])
//
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void sigmoid_gate_mul_batched(
    const __nv_bfloat16* __restrict__ input,   // [num_tokens, dim] contiguous
    const __nv_bfloat16* __restrict__ gate,    // [num_tokens, gate_stride] strided
    __nv_bfloat16* __restrict__ output,        // [num_tokens, dim] contiguous
    unsigned int dim,
    unsigned int gate_stride,
    unsigned int total_elements               // num_tokens * dim
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        unsigned int t = i / dim;
        unsigned int d = i % dim;
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(gate[(unsigned long long)t * gate_stride + d]);
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = __float2bfloat16(x * sigmoid_g);
    }
}

// BF16 concatenation: out[0..N] = a[0..N], out[N..2N] = b[0..N].
//
// Used by MTP head to concatenate normed embedding + normed hidden state
// before the FC projection. Both inputs are [N] BF16, output is [2N] BF16.
extern "C" __global__ void bf16_concat(
    const __nv_bfloat16* __restrict__ a,      // [N]
    const __nv_bfloat16* __restrict__ b,      // [N]
    __nv_bfloat16* __restrict__ out,          // [2N]
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = a[i];
        out[N + i] = b[i];
    }
}

// DeepLoop token recommit: GPU-side reconstruct of draft_tokens_dev for pass N+1.
//
// After pass N's argmax, staged[0..gamma_eff] holds those argmax tokens
// (D2D-copied to topk_tokens_dev before calling this kernel). Writes to dst:
//   [0]*eff_ctx + [last_token] + staged[0..gamma_eff]
// so that batched_embed can immediately proceed on-stream without host D2H.
//
// Grid: (ceil(n_attn / 256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void dflash_token_recommit(
    int* __restrict__ dst,           // draft_tokens_dev [n_attn] — overwritten
    const int* __restrict__ staged,  // topk_tokens_dev [gamma_eff] — pass-N argmax
    int last_token,
    unsigned int eff_ctx,
    unsigned int n_attn             // = eff_ctx + 1 + gamma_eff
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_attn) return;
    if (i < eff_ctx) {
        dst[i] = 0;
    } else if (i == eff_ctx) {
        dst[i] = last_token;
    } else {
        dst[i] = staged[i - eff_ctx - 1];
    }
}

// Device-pointer sigmoid blend: reads gate scalar from device BF16 pointer.
//
// output[i] += sigmoid(bf16_to_f32(*gate_ptr)) * src[i]
//
// Eliminates the D2H + CPU sigmoid for shared expert gate.
// Each block reads the single BF16 scalar and computes sigmoid in shared mem.
extern "C" __global__ void bf16_sigmoid_blend_device(
    __nv_bfloat16* __restrict__ output,             // [n] — routed output, modified in-place
    const __nv_bfloat16* __restrict__ src,          // [n] — shared expert output
    const __nv_bfloat16* __restrict__ gate_ptr,     // device pointer to single BF16 scalar
    unsigned int n
) {
    __shared__ float sigmoid_val;
    if (threadIdx.x == 0) {
        float g = __bfloat162float(*gate_ptr);
        sigmoid_val = 1.0f / (1.0f + __expf(-g));
    }
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float o = __bfloat162float(output[i]);
        float s = __bfloat162float(src[i]);
        output[i] = __float2bfloat16(o + sigmoid_val * s);
    }
}

// Batched BF16 residual add with per-row broadcast across `hc` hyper-streams:
//   hidden[(row * hc + branch) * h + d] += src[row * h + d]
// Replaces `rows * hc` separate `bf16_residual_add` launches in Qwen4 MTP
// `forward_one` (4 -> 1) and `forward_rows` (4 * rows -> 1) with a single
// launch while keeping the exact per-element FP32 addition + BF16 rounding.
//
// Grid: (ceil(h / 256), rows, 1)  Block: (256, 1, 1)
extern "C" __global__ void bf16_residual_add_broadcast_hc(
    __nv_bfloat16* __restrict__ hidden,     // [rows, hc, h] — modified in-place
    const __nv_bfloat16* __restrict__ src,  // [rows, h]
    unsigned int h,
    unsigned int hc
) {
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    if (d >= h) return;
    float s = __bfloat162float(src[(unsigned long long)row * h + d]);
    __nv_bfloat16* row_hidden = hidden + (unsigned long long)row * hc * h + d;
    for (unsigned int branch = 0; branch < hc; ++branch) {
        float r = __bfloat162float(row_hidden[(unsigned long long)branch * h]);
        row_hidden[(unsigned long long)branch * h] = __float2bfloat16(r + s);
    }
}

// Batched sigmoid gate multiply + per-tensor global absmax for prefill attention
// output gating (`0-ULP` identical output to `sigmoid_gate_mul_batched`).
// Grid: (min(num_tokens, cta_cap), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void __launch_bounds__(256) sigmoid_gate_mul_batched_absmax(
    const __nv_bfloat16* __restrict__ input,   // [num_tokens, dim] contiguous
    const __nv_bfloat16* __restrict__ gate,    // [num_tokens, gate_stride] strided
    __nv_bfloat16* __restrict__ output,        // [num_tokens, dim] contiguous
    float* __restrict__ global_max,            // [1] zero-initialized FP32
    unsigned int dim,
    unsigned int gate_stride,
    unsigned int num_tokens
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int half_dim = dim / 2u;
    const bool even_strides = ((dim & 1u) == 0u) && ((gate_stride & 1u) == 0u)
        && ((reinterpret_cast<unsigned long long>(input) & 3ull) == 0ull)
        && ((reinterpret_cast<unsigned long long>(gate) & 3ull) == 0ull)
        && ((reinterpret_cast<unsigned long long>(output) & 3ull) == 0ull);
    float local_max = 0.0f;

    for (unsigned int t = blockIdx.x; t < num_tokens; t += gridDim.x) {
        const __nv_bfloat16* in_row = input + (unsigned long long)t * dim;
        const __nv_bfloat16* g_row = gate + (unsigned long long)t * gate_stride;
        __nv_bfloat16* out_row = output + (unsigned long long)t * dim;

        if (even_strides) {
            const unsigned int* in32 = reinterpret_cast<const unsigned int*>(in_row);
            const unsigned int* g32 = reinterpret_cast<const unsigned int*>(g_row);
            unsigned int* out32 = reinterpret_cast<unsigned int*>(out_row);
            for (unsigned int i = tid; i < half_dim; i += blockDim.x) {
                unsigned int xv = in32[i];
                unsigned int gv = g32[i];
                float x0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(xv & 0xFFFFu)));
                float x1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(xv >> 16)));
                float g0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(gv & 0xFFFFu)));
                float g1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(gv >> 16)));
                __nv_bfloat16 b0 = __float2bfloat16(x0 * (1.0f / (1.0f + expf(-g0))));
                __nv_bfloat16 b1 = __float2bfloat16(x1 * (1.0f / (1.0f + expf(-g1))));
                out32[i] = (unsigned int)__bfloat16_as_ushort(b0)
                    | ((unsigned int)__bfloat16_as_ushort(b1) << 16);
                float a0 = fabsf(__bfloat162float(b0));
                float a1 = fabsf(__bfloat162float(b1));
                if (a0 > local_max) local_max = a0;
                if (a1 > local_max) local_max = a1;
            }
        } else {
            for (unsigned int d = tid; d < dim; d += blockDim.x) {
                float x = __bfloat162float(in_row[d]);
                float g = __bfloat162float(g_row[d]);
                __nv_bfloat16 ob = __float2bfloat16(x * (1.0f / (1.0f + expf(-g))));
                out_row[d] = ob;
                float a = fabsf(__bfloat162float(ob));
                if (a > local_max) local_max = a;
            }
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFu, local_max, offset);
        if (other > local_max) local_max = other;
    }

    __shared__ float warp_max[32];
    const unsigned int warp_id = tid / 32u;
    const unsigned int lane_id = tid % 32u;
    const unsigned int num_warps = (blockDim.x + 31u) / 32u;
    if (lane_id == 0u) warp_max[warp_id] = local_max;
    __syncthreads();

    if (tid == 0u) {
        float block_max = 0.0f;
        for (unsigned int w = 0; w < num_warps; ++w) {
            if (warp_max[w] > block_max) block_max = warp_max[w];
        }
        if (block_max > 0.0f && isfinite(block_max)) {
            unsigned int bits = __float_as_uint(block_max);
            unsigned int observed = *reinterpret_cast<const volatile unsigned int*>(global_max);
            if (bits > observed) {
                atomicMax(reinterpret_cast<unsigned int*>(global_max), bits);
            }
        }
    }
}


