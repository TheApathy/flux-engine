// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Causal Conv1d — Depthwise temporal convolution + SiLU activation.
//
// For Qwen3-Next Gated Delta Net layers:
//   Input: [batch, dim, seq_len]  BF16 (dim=8192, d_conv=4)
//   Weight: [dim, 1, d_conv]      BF16 (depthwise: groups=dim, converted to FP32 in-register)
//   Output: [batch, dim, seq_len]  BF16
//
// Wave 2 Additions (Decode-75 DFlash-27B):
//   - `causal_conv1d_update_l2norm_bf16_sequence`: Fused multi-token (K=17)
//     sliding-window conv1d + SiLU + Q/K L2-norm + per-token `state_inter`
//     snapshot with register-resident `s[4]` & `w_reg[4]`. Eliminates 33
//     launches per GDN layer (1,584 launches per verify step across 48 layers).
//   - `causal_conv1d_update_l2norm_bf16_tree_sequence`: Tree-aware multi-token
//     variant taking `parent_ids[num_tokens]`, resolving root and linear-chain
//     steps in registers and branch forks from `state_inter[parent]` in-kernel.
//     Eliminates 50 launches per GDN layer (2,400 launches per verify step).
//   - Register-resident sliding window in `causal_conv1d_update_l2norm_f32_sequence`.

#include <cuda_bf16.h>

// ============================================================
// PREFILL: Full-sequence causal depthwise conv1d + SiLU
// ============================================================
extern "C" __global__ void causal_conv1d_fwd(
    const __nv_bfloat16* __restrict__ input,   // [batch, dim, seq_len]
    const __nv_bfloat16* __restrict__ weight,  // [dim, d_conv]  BF16
    const float* __restrict__ bias,             // [dim] or nullptr
    __nv_bfloat16* __restrict__ output,         // [batch, dim, seq_len]
    unsigned int batch,
    unsigned int dim,
    unsigned int seq_len,
    unsigned int d_conv                         // typically 4
) {
    const unsigned int ch = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (ch >= dim || b >= batch) return;

    const unsigned long long base = ((unsigned long long)b * dim + ch) * seq_len;
    const __nv_bfloat16* in_ptr = input + base;
    __nv_bfloat16* out_ptr = output + base;
    const __nv_bfloat16* w = weight + ch * d_conv;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;

    float w_reg[8];
    for (unsigned int i = 0; i < d_conv && i < 8; i++) {
        w_reg[i] = (float)w[i];
    }

    for (unsigned int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float acc = b_val;
        for (unsigned int k = 0; k < d_conv && k < 8; k++) {
            int idx = (int)t - (int)(d_conv - 1) + (int)k;
            if (idx >= 0) {
                acc += (float)in_ptr[idx] * w_reg[k];
            }
        }
        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        float silu = acc * sigmoid_acc;
        out_ptr[t] = __float2bfloat16(silu);
    }
}

// ============================================================
// DECODE (UPDATE): Single-token sliding window update + SiLU
// ============================================================
extern "C" __global__ void causal_conv1d_update(
    float* __restrict__ conv_state,             // [batch, dim, d_conv] FP32 (in/out)
    const __nv_bfloat16* __restrict__ new_input, // [batch, dim] BF16
    const __nv_bfloat16* __restrict__ weight,   // [dim, d_conv] BF16
    const float* __restrict__ bias,             // [dim] or nullptr
    __nv_bfloat16* __restrict__ output,         // [batch, dim] BF16
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    if (ch >= dim || b >= batch) return;

    float* state = conv_state + (b * dim + ch) * d_conv;

    for (unsigned int i = 0; i < d_conv - 1; i++) {
        state[i] = state[i + 1];
    }
    state[d_conv - 1] = (float)new_input[b * dim + ch];

    const __nv_bfloat16* w = weight + ch * d_conv;
    float acc = (bias != nullptr) ? bias[ch] : 0.0f;
    for (unsigned int k = 0; k < d_conv; k++) {
        acc += state[k] * (float)w[k];
    }

    float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
    float silu = acc * sigmoid_acc;

    output[b * dim + ch] = __float2bfloat16(silu);
}

// ============================================================
// FP32 OUTPUT variant — prevents BF16 truncation in SSM recurrent path.
// ============================================================
extern "C" __global__ void causal_conv1d_update_f32(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    if (ch >= dim || b >= batch) return;

    float* state = conv_state + (b * dim + ch) * d_conv;

    for (unsigned int i = 0; i < d_conv - 1; i++) {
        state[i] = state[i + 1];
    }
    state[d_conv - 1] = (float)new_input[b * dim + ch];

    const __nv_bfloat16* w = weight + ch * d_conv;
    float acc = (bias != nullptr) ? bias[ch] : 0.0f;
    for (unsigned int k = 0; k < d_conv; k++) {
        acc += state[k] * (float)w[k];
    }

    float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
    float silu = acc * sigmoid_acc;

    output[b * dim + ch] = silu;
}

// ============================================================
// PREFILL UPDATE: Multi-token sliding window update + SiLU
// Requires d_conv == 4.
// ============================================================
extern "C" __global__ void causal_conv1d_update_prefill(
    float* __restrict__ conv_state,             // [dim, d_conv] FP32 (in/out)
    const __nv_bfloat16* __restrict__ input,    // N tokens, input[t * input_stride + ch]
    const __nv_bfloat16* __restrict__ weight,   // [dim, d_conv] BF16
    const float* __restrict__ bias,             // [dim] or nullptr
    __nv_bfloat16* __restrict__ output,         // N tokens, output[t * output_stride + ch]
    unsigned int dim,
    unsigned int d_conv,
    unsigned int seq_len,
    unsigned int input_stride,
    unsigned int output_stride
) {
    if (d_conv != 4u) return;
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;

    float* state = conv_state + (unsigned long long)ch * 4u;
    const __nv_bfloat16* w = weight + (unsigned long long)ch * 4u;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;

    float w_reg[4];
    for (unsigned int k = 0; k < 4u; k++) {
        w_reg[k] = (float)w[k];
    }

    float s[4];
    for (unsigned int k = 0; k < 4u; k++) {
        s[k] = state[k];
    }

    for (unsigned int t = 0; t < seq_len; t++) {
        float new_val = (float)input[(unsigned long long)t * input_stride + ch];
        s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; s[3] = new_val;
        float acc = b_val + s[0]*w_reg[0] + s[1]*w_reg[1] + s[2]*w_reg[2] + s[3]*w_reg[3];
        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        output[(unsigned long long)t * output_stride + ch] = __float2bfloat16(acc * sigmoid_acc);
    }

    for (unsigned int k = 0; k < 4u; k++) {
        state[k] = s[k];
    }
}

// Requires d_conv == 4.
extern "C" __global__ void causal_conv1d_update_prefill_zcopy(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ z_input,
    __nv_bfloat16* __restrict__ z_output,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int seq_len,
    unsigned int input_stride,
    unsigned int output_stride,
    unsigned int z_dim
) {
    if (d_conv != 4u) return;
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;

    float* state = conv_state + (unsigned long long)ch * 4u;
    const __nv_bfloat16* w = weight + (unsigned long long)ch * 4u;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;

    float w_reg[4];
    for (unsigned int k = 0; k < 4u; k++) {
        w_reg[k] = (float)w[k];
    }

    float s[4];
    for (unsigned int k = 0; k < 4u; k++) {
        s[k] = state[k];
    }

    for (unsigned int t = 0; t < seq_len; t++) {
        float new_val = (float)input[(unsigned long long)t * input_stride + ch];
        s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; s[3] = new_val;
        float acc = b_val + s[0]*w_reg[0] + s[1]*w_reg[1] + s[2]*w_reg[2] + s[3]*w_reg[3];
        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        output[(unsigned long long)t * output_stride + ch] = __float2bfloat16(acc * sigmoid_acc);
        if (ch < z_dim) {
            z_output[(unsigned long long)t * z_dim + ch] =
                z_input[(unsigned long long)t * input_stride + ch];
        }
    }

    for (unsigned int k = 0; k < 4u; k++) {
        state[k] = s[k];
    }
}

__device__ __forceinline__ void conv_l2_unpack_bf16x2(
    unsigned int packed, float& v0, float& v1
) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int conv_l2_pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

__device__ __forceinline__ float conv_l2_warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
    }
    return value;
}

extern "C" __global__ void causal_conv1d_update_prefill_l2norm_zcopy(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ z_input,
    __nv_bfloat16* __restrict__ z_output,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int seq_len,
    unsigned int input_stride,
    unsigned int output_stride,
    unsigned int z_dim,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps
) {
    if (d_conv != 4 || head_dim != 128 || blockDim.x != 128 || (dim & 127) != 0 ||
        (qk_channels & 127) != 0) {
        return;
    }

    const unsigned int tid = threadIdx.x;
    const unsigned int head_start = blockIdx.x * head_dim;
    const unsigned int ch = head_start + tid;
    const bool normalize = head_start < qk_channels;

    float* state = conv_state + (unsigned long long)ch * d_conv;
    const __nv_bfloat16* w = weight + (unsigned long long)ch * d_conv;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;

    float w_reg[4];
    float s[4];
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        w_reg[k] = (float)w[k];
        s[k] = state[k];
    }

    __shared__ __nv_bfloat16 rounded[128];
    __shared__ float warp_sums[4];

    for (unsigned int token = 0; token < seq_len; token++) {
        float new_val = (float)input[(unsigned long long)token * input_stride + ch];
        s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; s[3] = new_val;

        float acc = b_val + s[0]*w_reg[0] + s[1]*w_reg[1]
            + s[2]*w_reg[2] + s[3]*w_reg[3];
        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        __nv_bfloat16 silu_bf16 = __float2bfloat16(acc * sigmoid_acc);

        if (ch < z_dim) {
            z_output[(unsigned long long)token * z_dim + ch] =
                z_input[(unsigned long long)token * input_stride + ch];
        }

        if (normalize) {
            rounded[tid] = silu_bf16;
            __syncthreads();

            float sum_sq = 0.0f;
            if (tid < 64) {
                float v0, v1;
                unsigned int packed = ((const unsigned int*)rounded)[tid];
                conv_l2_unpack_bf16x2(packed, v0, v1);
                sum_sq += v0 * v0 + v1 * v1;
            }
            sum_sq = conv_l2_warp_reduce_sum(sum_sq);

            const unsigned int warp_id = tid / 32;
            const unsigned int lane_id = tid % 32;
            if (lane_id == 0) warp_sums[warp_id] = sum_sq;
            __syncthreads();

            if (warp_id == 0) {
                float value = lane_id < 4 ? warp_sums[lane_id] : 0.0f;
                value = conv_l2_warp_reduce_sum(value);
                if (lane_id == 0) warp_sums[0] = value;
            }
            __syncthreads();

            const float inv_norm = rsqrtf(warp_sums[0] + l2_eps);
            if (tid < 64) {
                float v0, v1;
                unsigned int packed = ((const unsigned int*)rounded)[tid];
                conv_l2_unpack_bf16x2(packed, v0, v1);
                unsigned int* out32 = (unsigned int*)(
                    output + (unsigned long long)token * output_stride + head_start
                );
                out32[tid] = conv_l2_pack_bf16x2(v0 * inv_norm, v1 * inv_norm);
            }
            __syncthreads();
        } else {
            output[(unsigned long long)token * output_stride + ch] = silu_bf16;
        }
    }

    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        state[k] = s[k];
    }
}

// ============================================================
// CHUNK2: Fused 2-token conv1d update + SiLU
// ============================================================
// Requires d_conv == 4.
extern "C" __global__ void causal_conv1d_update_chunk2(
    float* __restrict__ conv_state,              // [batch, dim, d_conv] FP32 (in/out)
    const __nv_bfloat16* __restrict__ new_input, // [batch, 2, dim] BF16
    const __nv_bfloat16* __restrict__ weight,    // [dim, d_conv] BF16
    const float* __restrict__ bias,              // [dim] or nullptr
    __nv_bfloat16* __restrict__ output,          // [batch, 2, dim] BF16
    float* __restrict__ conv_state_intermediate, // [batch, dim, d_conv] FP32 (out)
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv
) {
    if (d_conv != 4u) return;
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    if (ch >= dim || b >= batch) return;

    float* state = conv_state + ((unsigned long long)b * dim + ch) * 4u;
    float* state_inter = conv_state_intermediate + ((unsigned long long)b * dim + ch) * 4u;
    const __nv_bfloat16* w = weight + (unsigned long long)ch * 4u;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;

    float w_reg[4];
    for (unsigned int k = 0; k < 4u; k++) {
        w_reg[k] = (float)w[k];
    }

    float s[4];
    for (unsigned int k = 0; k < 4u; k++) {
        s[k] = state[k];
    }

    float in0 = (float)new_input[b * 2 * dim + ch];
    float s0_0 = s[1], s0_1 = s[2], s0_2 = s[3], s0_3 = in0;

    state_inter[0] = s0_0;
    state_inter[1] = s0_1;
    state_inter[2] = s0_2;
    state_inter[3] = s0_3;

    float acc0 = b_val + s0_0*w_reg[0] + s0_1*w_reg[1] + s0_2*w_reg[2] + s0_3*w_reg[3];
    float sig0 = 1.0f / (1.0f + __expf(-acc0));
    output[b * 2 * dim + ch] = __float2bfloat16(acc0 * sig0);

    float in1 = (float)new_input[(b * 2 + 1) * dim + ch];
    float s1_0 = s0_1, s1_1 = s0_2, s1_2 = s0_3, s1_3 = in1;

    state[0] = s1_0;
    state[1] = s1_1;
    state[2] = s1_2;
    state[3] = s1_3;

    float acc1 = b_val + s1_0*w_reg[0] + s1_1*w_reg[1] + s1_2*w_reg[2] + s1_3*w_reg[3];
    float sig1 = 1.0f / (1.0f + __expf(-acc1));
    output[(b * 2 + 1) * dim + ch] = __float2bfloat16(acc1 * sig1);
}

// ============================================================
// FUSED: Conv1d update + SiLU + L2 normalization (decode)
// ============================================================
extern "C" __global__ void causal_conv1d_update_l2norm(
    float* __restrict__ conv_state,             // [batch, dim, d_conv] FP32 (in/out)
    const __nv_bfloat16* __restrict__ new_input, // [batch, dim] BF16
    const __nv_bfloat16* __restrict__ weight,   // [dim, d_conv] BF16
    const float* __restrict__ bias,             // [dim] or nullptr
    __nv_bfloat16* __restrict__ output,         // [batch, dim] BF16
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);

    const bool valid = (ch < dim && b < batch);
    float silu = 0.0f;

    if (valid) {
        float* state = conv_state + (b * dim + ch) * d_conv;

        for (unsigned int i = 0; i < d_conv - 1; i++)
            state[i] = state[i + 1];
        state[d_conv - 1] = (float)new_input[b * dim + ch];

        const __nv_bfloat16* w = weight + ch * d_conv;
        float acc = (bias != nullptr) ? bias[ch] : 0.0f;
        for (unsigned int k = 0; k < d_conv; k++)
            acc += state[k] * (float)w[k];

        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        silu = acc * sigmoid_acc;
    }

    if (block_needs_l2) {
        float sq = valid ? (silu * silu) : 0.0f;

        const unsigned int warp_id = tid / 32;
        const unsigned int lane = tid % 32;
        for (int offset = 16; offset >= 1; offset >>= 1)
            sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);

        __shared__ float warp_sums[8];
        if (lane == 0) warp_sums[warp_id] = sq;
        __syncthreads();

        const unsigned int head_in_block = tid / head_dim;
        const unsigned int base_warp = head_in_block * (head_dim / 32);

        if (tid == 0 || tid == head_dim) {
            float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                        + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
            warp_sums[base_warp] = rsqrtf(total + l2_eps);
        }
        __syncthreads();

        if (valid) {
            silu *= warp_sums[base_warp];
        }
    }

    if (valid) {
        output[b * dim + ch] = __float2bfloat16(silu);
    }
}

// ============================================================
// FP32 OUTPUT variant of fused conv1d + L2 norm.
// ============================================================
extern "C" __global__ void causal_conv1d_update_l2norm_f32(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);
    const bool valid = (ch < dim && b < batch);
    float silu = 0.0f;

    if (valid) {
        float* state = conv_state + (b * dim + ch) * d_conv;
        for (unsigned int i = 0; i < d_conv - 1; i++)
            state[i] = state[i + 1];
        state[d_conv - 1] = (float)new_input[b * dim + ch];
        const __nv_bfloat16* w = weight + ch * d_conv;
        float acc = (bias != nullptr) ? bias[ch] : 0.0f;
        for (unsigned int k = 0; k < d_conv; k++)
            acc += state[k] * (float)w[k];
        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        silu = acc * sigmoid_acc;
    }

    if (block_needs_l2) {
        float sq = valid ? (silu * silu) : 0.0f;
        const unsigned int warp_id = tid / 32;
        const unsigned int lane = tid % 32;
        for (int offset = 16; offset >= 1; offset >>= 1)
            sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
        __shared__ float warp_sums[8];
        if (lane == 0) warp_sums[warp_id] = sq;
        __syncthreads();
        const unsigned int head_in_block = tid / head_dim;
        const unsigned int base_warp = head_in_block * (head_dim / 32);
        if (tid == 0 || tid == head_dim) {
            float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                        + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
            warp_sums[base_warp] = rsqrtf(total + l2_eps);
        }
        __syncthreads();
        if (valid) {
            silu *= warp_sums[base_warp];
        }
    }

    if (valid) {
        output[b * dim + ch] = silu;
    }
}

// Flat-chain multi-row variant of causal_conv1d_update_l2norm_f32.
// Wave 2: keeps `s[4]` and `w_reg[4]` in registers across all `num_tokens`
// instead of reading/writing `conv_state` in DRAM on every token iteration.
extern "C" __global__ void causal_conv1d_update_l2norm_f32_sequence(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float* __restrict__ state_inter,
    unsigned int num_tokens,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    unsigned int input_stride,
    unsigned int output_stride,
    unsigned int state_stride
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = block_start < qk_channels;
    const bool valid = ch < dim;
    __shared__ float warp_sums[8];

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f;
    float b_val = 0.0f;
    float* state = conv_state + (unsigned long long)ch * d_conv;

    if (valid && d_conv == 4u) {
        const __nv_bfloat16* w = weight + (unsigned long long)ch * 4u;
        s0 = state[0]; s1 = state[1]; s2 = state[2]; s3 = state[3];
        w0 = (float)w[0]; w1 = (float)w[1]; w2 = (float)w[2]; w3 = (float)w[3];
        b_val = (bias != nullptr) ? bias[ch] : 0.0f;
    }

    for (unsigned int token = 0; token < num_tokens; token++) {
        float silu = 0.0f;
        if (valid) {
            if (d_conv == 4u) {
                s0 = s1;
                s1 = s2;
                s2 = s3;
                s3 = (float)new_input[(unsigned long long)token * input_stride + ch];
                float acc = b_val + s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3;
                float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
                silu = acc * sigmoid_acc;

                float* snapshot = state_inter
                    + (unsigned long long)token * state_stride
                    + (unsigned long long)ch * 4u;
                snapshot[0] = s0;
                snapshot[1] = s1;
                snapshot[2] = s2;
                snapshot[3] = s3;
            } else {
                for (unsigned int i = 0; i < d_conv - 1; i++)
                    state[i] = state[i + 1];
                state[d_conv - 1] =
                    (float)new_input[(unsigned long long)token * input_stride + ch];
                const __nv_bfloat16* w = weight + (unsigned long long)ch * d_conv;
                float acc = (bias != nullptr) ? bias[ch] : 0.0f;
                for (unsigned int k = 0; k < d_conv; k++)
                    acc += state[k] * (float)w[k];
                float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
                silu = acc * sigmoid_acc;

                float* snapshot = state_inter
                    + (unsigned long long)token * state_stride
                    + (unsigned long long)ch * d_conv;
                for (unsigned int i = 0; i < d_conv; i++)
                    snapshot[i] = state[i];
            }
        }

        if (block_needs_l2) {
            float sq = valid ? silu * silu : 0.0f;
            const unsigned int warp_id = tid / 32;
            const unsigned int lane = tid % 32;
            for (int offset = 16; offset >= 1; offset >>= 1)
                sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
            if (lane == 0) warp_sums[warp_id] = sq;
            __syncthreads();
            const unsigned int head_in_block = tid / head_dim;
            const unsigned int base_warp = head_in_block * (head_dim / 32);
            if (tid == 0 || tid == head_dim) {
                float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                            + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
                warp_sums[base_warp] = rsqrtf(total + l2_eps);
            }
            __syncthreads();
            if (valid) silu *= warp_sums[base_warp];
            __syncthreads();
        }

        if (valid) {
            output[(unsigned long long)token * output_stride + ch] = silu;
        }
    }

    if (valid && d_conv == 4u && num_tokens > 0u) {
        state[0] = s0; state[1] = s1; state[2] = s2; state[3] = s3;
    }
}

// ============================================================
// WAVE 2: Fused Multi-Token BF16 Conv1d + SiLU + L2-Norm + State Snapshots
// ============================================================
// Replaces the 17-iteration host loop of `causal_conv1d_update_l2norm` +
// `cuMemcpyDtoDAsync(conv_state -> conv_state_intermediates[t])` in
// `trait_decode_batched_conv_gdn.rs` (`wy17` and `K>=5` paths) with a single
// launch per GDN layer. Keeps the 4-tap sliding window and weights in
// registers across all `num_tokens` tokens.
//
// Grid:  (ceil(dim/256), 1, 1)
// Block: (256, 1, 1)
extern "C" __global__ void causal_conv1d_update_l2norm_bf16_sequence(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ state_inter,
    unsigned int num_tokens,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    unsigned int input_stride,
    unsigned int output_stride,
    unsigned int state_stride
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = block_start < qk_channels;
    const bool valid = ch < dim;
    __shared__ float warp_sums[8];

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f;
    float b_val = 0.0f;
    float* state = conv_state + (unsigned long long)ch * d_conv;

    if (valid && d_conv == 4u) {
        const __nv_bfloat16* w = weight + (unsigned long long)ch * 4u;
        s0 = state[0]; s1 = state[1]; s2 = state[2]; s3 = state[3];
        w0 = (float)w[0]; w1 = (float)w[1]; w2 = (float)w[2]; w3 = (float)w[3];
        b_val = (bias != nullptr) ? bias[ch] : 0.0f;
    }

    for (unsigned int token = 0; token < num_tokens; token++) {
        float silu = 0.0f;
        if (valid) {
            if (d_conv == 4u) {
                s0 = s1;
                s1 = s2;
                s2 = s3;
                s3 = (float)new_input[(unsigned long long)token * input_stride + ch];
                float acc = b_val + s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3;
                float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
                silu = acc * sigmoid_acc;

                float* snapshot = state_inter
                    + (unsigned long long)token * state_stride
                    + (unsigned long long)ch * 4u;
                snapshot[0] = s0;
                snapshot[1] = s1;
                snapshot[2] = s2;
                snapshot[3] = s3;
            } else {
                for (unsigned int i = 0; i < d_conv - 1; i++)
                    state[i] = state[i + 1];
                state[d_conv - 1] =
                    (float)new_input[(unsigned long long)token * input_stride + ch];
                const __nv_bfloat16* w = weight + (unsigned long long)ch * d_conv;
                float acc = (bias != nullptr) ? bias[ch] : 0.0f;
                for (unsigned int k = 0; k < d_conv; k++)
                    acc += state[k] * (float)w[k];
                float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
                silu = acc * sigmoid_acc;

                float* snapshot = state_inter
                    + (unsigned long long)token * state_stride
                    + (unsigned long long)ch * d_conv;
                for (unsigned int i = 0; i < d_conv; i++)
                    snapshot[i] = state[i];
            }
        }

        if (block_needs_l2) {
            float sq = valid ? (silu * silu) : 0.0f;
            const unsigned int warp_id = tid / 32;
            const unsigned int lane = tid % 32;
            for (int offset = 16; offset >= 1; offset >>= 1)
                sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
            if (lane == 0) warp_sums[warp_id] = sq;
            __syncthreads();
            const unsigned int head_in_block = tid / head_dim;
            const unsigned int base_warp = head_in_block * (head_dim / 32);
            if (tid == 0 || tid == head_dim) {
                float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                            + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
                warp_sums[base_warp] = rsqrtf(total + l2_eps);
            }
            __syncthreads();
            if (valid) silu *= warp_sums[base_warp];
            __syncthreads();
        }

        if (valid) {
            output[(unsigned long long)token * output_stride + ch] =
                __float2bfloat16(silu);
        }
    }

    if (valid && d_conv == 4u && num_tokens > 0u) {
        state[0] = s0; state[1] = s1; state[2] = s2; state[3] = s3;
    }
}

// ============================================================
// WAVE 2: Tree-Aware Fused Multi-Token BF16 Conv1d + SiLU + L2-Norm
// ============================================================
// Replaces the 17-iteration host loop of `causal_conv1d_tree_reroot` +
// `causal_conv1d_update_l2norm` + `cuMemcpyDtoDAsync` in the DDTree verify
// path (`trait_decode_batched_conv_gdn.rs`) with a single launch per GDN layer.
//
// When `parent_ids != nullptr && token > 0`:
//   - `parent == token - 1`: reuses rolling `s0..s3` in registers (0 DRAM reads)
//   - `parent < 0`:          reuses root `sr0..sr3` in registers (0 DRAM reads)
//   - `0 <= parent < token - 1`: reloads `s0..s3` from `state_inter[parent]`
//     (written earlier by the same thread `ch` in this kernel).
extern "C" __global__ void causal_conv1d_update_l2norm_bf16_tree_sequence(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    const int* __restrict__ parent_ids,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ state_inter,
    unsigned int num_tokens,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    unsigned int input_stride,
    unsigned int output_stride,
    unsigned int state_stride
) {
    if (d_conv == 0u || (d_conv != 4u && d_conv > 8u)) return;
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = block_start < qk_channels;
    const bool valid = ch < dim;
    __shared__ float warp_sums[8];

    float sr0 = 0.0f, sr1 = 0.0f, sr2 = 0.0f, sr3 = 0.0f;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f;
    float b_val = 0.0f;
    float sr_gen[8];
    float* state = conv_state + (unsigned long long)ch * d_conv;

    if (valid && d_conv == 4u) {
        const __nv_bfloat16* w = weight + (unsigned long long)ch * 4u;
        sr0 = state[0]; sr1 = state[1]; sr2 = state[2]; sr3 = state[3];
        s0 = sr0; s1 = sr1; s2 = sr2; s3 = sr3;
        w0 = (float)w[0]; w1 = (float)w[1]; w2 = (float)w[2]; w3 = (float)w[3];
        b_val = (bias != nullptr) ? bias[ch] : 0.0f;
    } else if (valid) {
        for (unsigned int i = 0; i < d_conv; i++) {
            sr_gen[i] = state[i];
        }
    }

    for (unsigned int token = 0; token < num_tokens; token++) {
        float silu = 0.0f;
        if (valid) {
            if (d_conv == 4u) {
                if (token > 0u && parent_ids != nullptr) {
                    const int parent = parent_ids[token];
                    if (parent < 0) {
                        s0 = sr0; s1 = sr1; s2 = sr2; s3 = sr3;
                    } else if (parent != (int)token - 1) {
                        const float* p_snap = state_inter
                            + (unsigned long long)((unsigned int)parent) * state_stride
                            + (unsigned long long)ch * 4u;
                        s0 = p_snap[0]; s1 = p_snap[1]; s2 = p_snap[2]; s3 = p_snap[3];
                    }
                }
                s0 = s1;
                s1 = s2;
                s2 = s3;
                s3 = (float)new_input[(unsigned long long)token * input_stride + ch];
                float acc = b_val + s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3;
                float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
                silu = acc * sigmoid_acc;

                float* snapshot = state_inter
                    + (unsigned long long)token * state_stride
                    + (unsigned long long)ch * 4u;
                snapshot[0] = s0;
                snapshot[1] = s1;
                snapshot[2] = s2;
                snapshot[3] = s3;
            } else {
                if (token > 0u && parent_ids != nullptr) {
                    const int parent = parent_ids[token];
                    if (parent < 0) {
                        for (unsigned int i = 0; i < d_conv; i++)
                            state[i] = sr_gen[i];
                    } else if (parent != (int)token - 1) {
                        const float* p_snap = state_inter
                            + (unsigned long long)((unsigned int)parent) * state_stride
                            + (unsigned long long)ch * d_conv;
                        for (unsigned int i = 0; i < d_conv; i++)
                            state[i] = p_snap[i];
                    }
                }
                for (unsigned int i = 0; i < d_conv - 1; i++)
                    state[i] = state[i + 1];
                state[d_conv - 1] =
                    (float)new_input[(unsigned long long)token * input_stride + ch];
                const __nv_bfloat16* w = weight + (unsigned long long)ch * d_conv;
                float acc = (bias != nullptr) ? bias[ch] : 0.0f;
                for (unsigned int k = 0; k < d_conv; k++)
                    acc += state[k] * (float)w[k];
                float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
                silu = acc * sigmoid_acc;

                float* snapshot = state_inter
                    + (unsigned long long)token * state_stride
                    + (unsigned long long)ch * d_conv;
                for (unsigned int i = 0; i < d_conv; i++)
                    snapshot[i] = state[i];
            }
        }

        if (block_needs_l2) {
            float sq = valid ? (silu * silu) : 0.0f;
            const unsigned int warp_id = tid / 32;
            const unsigned int lane = tid % 32;
            for (int offset = 16; offset >= 1; offset >>= 1)
                sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
            if (lane == 0) warp_sums[warp_id] = sq;
            __syncthreads();
            const unsigned int head_in_block = tid / head_dim;
            const unsigned int base_warp = head_in_block * (head_dim / 32);
            if (tid == 0 || tid == head_dim) {
                float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                            + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
                warp_sums[base_warp] = rsqrtf(total + l2_eps);
            }
            __syncthreads();
            if (valid) silu *= warp_sums[base_warp];
            __syncthreads();
        }

        if (valid) {
            output[(unsigned long long)token * output_stride + ch] =
                __float2bfloat16(silu);
        }
    }

    if (valid && d_conv == 4u && num_tokens > 0u) {
        state[0] = s0; state[1] = s1; state[2] = s2; state[3] = s3;
    }
}

// ============================================================
// TREE-AWARE CONV STATE RE-ROOT (M8A / FREE_SLOTS)
// ============================================================
extern "C" __global__ void causal_conv1d_tree_reroot(
    float* __restrict__ conv_state,
    const float* __restrict__ inter_base,
    const int* __restrict__ parent_ids,
    unsigned int t,
    unsigned int n_floats
) {
    const int parent = parent_ids[t];
    if (t == 0 || parent < 0 || parent == (int)t - 1) return;
    const float* src = inter_base + (long long)parent * n_floats;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_floats; i += gridDim.x * blockDim.x) {
        conv_state[i] = src[i];
    }
}

// ── Cross-sequence twin of causal_conv1d_update_l2norm_f32_sequence ──────
// Grid (ceil(dim/256), num_seqs). Block (x, s) executes, for sequence s,
// the statement sequence of causal_conv1d_update_l2norm_f32_sequence on its
// own conv state, its `num_tokens` contiguous input rows (rows
// s*num_tokens.. of `new_input`), its own output rows and snapshots, so
// every sequence is bit-identical to its single-sequence launch.
extern "C" __global__ void causal_conv1d_update_l2norm_f32_sequence_multi(
    const unsigned long long* __restrict__ conv_state_ptrs,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    const unsigned long long* __restrict__ output_ptrs,
    const unsigned long long* __restrict__ state_inter_ptrs,
    unsigned int num_tokens,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    unsigned int input_stride,
    unsigned int output_stride,
    unsigned int state_stride
) {
    const unsigned int s = blockIdx.y;
    float* conv_state = (float*)conv_state_ptrs[s];
    const __nv_bfloat16* seq_input =
        new_input + (unsigned long long)s * num_tokens * input_stride;
    float* output = (float*)output_ptrs[s];
    float* state_inter = (float*)state_inter_ptrs[s];
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = block_start < qk_channels;
    const bool valid = ch < dim;
    __shared__ float warp_sums[8];

    for (unsigned int token = 0; token < num_tokens; token++) {
        float silu = 0.0f;
        if (valid) {
            float* state = conv_state + (unsigned long long)ch * d_conv;
            for (unsigned int i = 0; i < d_conv - 1; i++)
                state[i] = state[i + 1];
            state[d_conv - 1] =
                (float)seq_input[(unsigned long long)token * input_stride + ch];
            const __nv_bfloat16* w = weight + (unsigned long long)ch * d_conv;
            float acc = (bias != nullptr) ? bias[ch] : 0.0f;
            for (unsigned int k = 0; k < d_conv; k++)
                acc += state[k] * (float)w[k];
            float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
            silu = acc * sigmoid_acc;
        }

        if (block_needs_l2) {
            float sq = valid ? silu * silu : 0.0f;
            const unsigned int warp_id = tid / 32;
            const unsigned int lane = tid % 32;
            for (int offset = 16; offset >= 1; offset >>= 1)
                sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
            if (lane == 0) warp_sums[warp_id] = sq;
            __syncthreads();
            const unsigned int head_in_block = tid / head_dim;
            const unsigned int base_warp = head_in_block * (head_dim / 32);
            if (tid == 0 || tid == head_dim) {
                float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                            + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
                warp_sums[base_warp] = rsqrtf(total + l2_eps);
            }
            __syncthreads();
            if (valid) silu *= warp_sums[base_warp];
        }

        if (valid) {
            output[(unsigned long long)token * output_stride + ch] = silu;
            if (state_inter != nullptr) {
                const float* state = conv_state + (unsigned long long)ch * d_conv;
                float* snapshot = state_inter
                    + (unsigned long long)token * state_stride
                    + (unsigned long long)ch * d_conv;
                for (unsigned int i = 0; i < d_conv; i++)
                    snapshot[i] = state[i];
            }
        }
        __syncthreads();
    }
}
