// SPDX-License-Identifier: AGPL-3.0-only

// Qwen4-Exp four-stream gated residual primitives.
// The projection GEMVs stay on Atlas's tuned dense/NVFP4 kernels; these
// kernels implement the architecture-specific normalization and joins.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __constant__ float QWEN4_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Dequantize the 16 selected PLE records into their concatenated 2560-wide
// embedding. Each record is 80 packed E2M1 bytes followed by ten E4M3
// group-16 scales; scale2 is one f32 per physical source shard.
extern "C" __global__ void qwen4_ple_dequant_rows(
    const unsigned char* __restrict__ records,
    const float* __restrict__ scale2,
    __nv_bfloat16* __restrict__ output) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2560u) return;
    const unsigned int row = blockIdx.y;
    const unsigned int head = i / 160u;
    const unsigned int dim = i - head * 160u;
    const unsigned int rec_idx = row * 16u + head;
    const unsigned char* record = records + rec_idx * 90u;
    const unsigned char packed = __ldg(&record[dim >> 1]);
    const unsigned int code = (dim & 1u) ? (packed >> 4) : (packed & 0x0fu);
    __nv_fp8_e4m3 fp8_scale;
    *(unsigned char*)&fp8_scale = __ldg(&record[80u + dim / 16u]);
    const float value = QWEN4_E2M1_LUT[code] * (float)fp8_scale * __ldg(&scale2[rec_idx]);
    output[row * 2560u + i] = __float2bfloat16(value);
}

// Whole-prompt variant. Records and scale2 values are token-major, then
// head-major; output is a row-major [M, 2560] BF16 activation matrix.
extern "C" __global__ void qwen4_ple_dequant_prefill(
    const unsigned char* __restrict__ records,
    const float* __restrict__ scale2,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_tokens) {
    const unsigned int lane = threadIdx.x & 31u;
    const float lut_lane = QWEN4_E2M1_LUT[lane & 15u];
    const unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = num_tokens * 2560u;
    if (linear >= total) return;
    const unsigned int token = linear / 2560u;
    const unsigned int embedding_dim = linear - token * 2560u;
    const unsigned int head = embedding_dim / 160u;
    const unsigned int dim = embedding_dim - head * 160u;
    const unsigned int record_index = token * 16u + head;
    const unsigned char* record = records + record_index * 90u;
    const unsigned char packed = __ldg(&record[dim >> 1]);
    const unsigned int code = (dim & 1u) ? (packed >> 4) : (packed & 0x0fu);
    __nv_fp8_e4m3 fp8_scale;
    *(unsigned char*)&fp8_scale = __ldg(&record[80u + dim / 16u]);
    const float lut_val = __shfl_sync(0xffffffffu, lut_lane, code);
    const float value = lut_val * (float)fp8_scale
                      * __ldg(&scale2[record_index]);
    output[linear] = __float2bfloat16(value);
}

__device__ __forceinline__ float warp_sum(float v);

// Complete one-token PLE join. The projection GEMVs are dispatched through
// Atlas's tuned NVFP4 kernels; this kernel fuses the three grouped RMS norms,
// signed-sqrt gate, dilated depthwise short convolution, state update, and
// residual injection. `conv_state` is [4H, 9], oldest sample first.
extern "C" __global__ void qwen4_ple_fuse_decode(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ norm_key,
    const __nv_bfloat16* __restrict__ norm_query,
    const __nv_bfloat16* __restrict__ norm_conv,
    const __nv_bfloat16* __restrict__ conv_weight,
    __nv_bfloat16* __restrict__ conv_state,
    unsigned int hidden_size,
    float eps,
    unsigned int reset_state) {
    const unsigned int stream_id = blockIdx.x;
    const unsigned int base = stream_id * hidden_size;
    float q2 = 0.0f, k2 = 0.0f;
    for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const float q = __bfloat162float(hyper[base + h]);
        const float k = __bfloat162float(key[base + h]);
        q2 += q * q;
        k2 += k * k;
    }
    q2 = warp_sum(q2);
    k2 = warp_sum(k2);
    __shared__ float q_parts[32], k_parts[32], shared[4];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    if (lane == 0) {
        q_parts[warp] = q2;
        k_parts[warp] = k2;
    }
    __syncthreads();
    if (warp == 0) {
        const unsigned int warps = (blockDim.x + 31) / 32;
        float q = lane < warps ? q_parts[lane] : 0.0f;
        float k = lane < warps ? k_parts[lane] : 0.0f;
        q = warp_sum(q);
        k = warp_sum(k);
        if (lane == 0) {
            shared[0] = rsqrtf(q / hidden_size + eps);
            shared[1] = rsqrtf(k / hidden_size + eps);
        }
    }
    __syncthreads();

    float dot = 0.0f;
    for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const unsigned int i = base + h;
        const float q = __bfloat162float(hyper[i]) * shared[0]
                      * (1.0f + __bfloat162float(norm_query[i]));
        const float k = __bfloat162float(key[i]) * shared[1]
                      * (1.0f + __bfloat162float(norm_key[i]));
        dot += q * k;
    }
    dot = warp_sum(dot);
    if (lane == 0) q_parts[warp] = dot;
    __syncthreads();
    if (warp == 0) {
        const unsigned int warps = (blockDim.x + 31) / 32;
        float v = lane < warps ? q_parts[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) {
            v /= sqrtf((float)hidden_size);
            const float signed_root = copysignf(sqrtf(fmaxf(fabsf(v), 1.0e-6f)), v);
            shared[2] = 1.0f / (1.0f + expf(-signed_root));
        }
    }
    __syncthreads();

    // The gate is scalar per stream, so the RMS of gate*value is obtained
    // directly without materializing a 4H intermediate.
    float gv2 = 0.0f;
    for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const float gv = shared[2] * __bfloat162float(value[h]);
        gv2 += gv * gv;
    }
    gv2 = warp_sum(gv2);
    if (lane == 0) q_parts[warp] = gv2;
    __syncthreads();
    if (warp == 0) {
        const unsigned int warps = (blockDim.x + 31) / 32;
        float v = lane < warps ? q_parts[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) shared[3] = rsqrtf(v / hidden_size + eps);
    }
    __syncthreads();

    for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const unsigned int i = base + h;
        const float gv = shared[2] * __bfloat162float(value[h]);
        const float x = gv * shared[3] * (1.0f + __bfloat162float(norm_conv[i]));
        __nv_bfloat16* state = conv_state + i * 9u;
        const float s0 = reset_state ? 0.0f : __bfloat162float(state[0]);
        const float s3 = reset_state ? 0.0f : __bfloat162float(state[3]);
        const float s6 = reset_state ? 0.0f : __bfloat162float(state[6]);
        const __nv_bfloat16* w = conv_weight + i * 4u;
        float c = s0 * __bfloat162float(w[0])
                + s3 * __bfloat162float(w[1])
                + s6 * __bfloat162float(w[2])
                + x  * __bfloat162float(w[3]);
        c = c / (1.0f + expf(-c));
        if (!reset_state) {
            #pragma unroll
            for (unsigned int j = 0; j < 8; ++j) state[j] = state[j + 1];
        } else {
            #pragma unroll
            for (unsigned int j = 0; j < 8; ++j) state[j] = __float2bfloat16(0.0f);
        }
        state[8] = __float2bfloat16(x);
        hyper[i] = __float2bfloat16(__bfloat162float(hyper[i]) + gv + c);
    }
}

// Multi-row speculative verify PLE join + intermediate state capture in a single launch per layer.
// Replaces `rows` sequential launches of `qwen4_ple_fuse_decode` + `rows` sequential `cuMemcpyDtoDAsync`
// (`save_intermediate`) calls while preserving the exact per-row reduction tree and BF16 rounding.
extern "C" __global__ void __launch_bounds__(1024, 1) qwen4_ple_fuse_verify_rows(
    __nv_bfloat16* __restrict__ hyper,
    unsigned int hyper_row_elems,
    const __nv_bfloat16* __restrict__ key_rows,
    const __nv_bfloat16* __restrict__ value_rows,
    const __nv_bfloat16* __restrict__ norm_key,
    const __nv_bfloat16* __restrict__ norm_query,
    const __nv_bfloat16* __restrict__ norm_conv,
    const __nv_bfloat16* __restrict__ conv_weight,
    __nv_bfloat16* __restrict__ conv_state_base,
    __nv_bfloat16* __restrict__ conv_inter_base,
    unsigned int hidden_size,
    float eps,
    unsigned int rows,
    unsigned int max_spec_rows,
    unsigned long long packed_slots) {
    const unsigned int stream_id = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int slot_idx = (unsigned int)((packed_slots >> (seq_idx * 8u)) & 0xFFull);
    const unsigned int base = stream_id * hidden_size;
    const unsigned int residual_width = hidden_size * 4u;
    const unsigned int state_stride = residual_width * 9u;
    const unsigned long long row_offset = (unsigned long long)seq_idx * rows;
    hyper += row_offset * hyper_row_elems;
    key_rows += row_offset * residual_width;
    value_rows += row_offset * hidden_size;
    __nv_bfloat16* __restrict__ conv_state =
        conv_state_base + (unsigned long long)slot_idx * state_stride;
    __nv_bfloat16* __restrict__ conv_inter =
        conv_inter_base + (unsigned long long)slot_idx * max_spec_rows * state_stride;
    __shared__ float q_parts[32], k_parts[32], shared[4];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warps = (blockDim.x + 31u) / 32u;

    for (unsigned int r = 0u; r < rows; ++r) {
        __nv_bfloat16* __restrict__ hyper_r = hyper + (unsigned long long)r * hyper_row_elems;
        const __nv_bfloat16* __restrict__ key_r = key_rows + (unsigned long long)r * residual_width;
        const __nv_bfloat16* __restrict__ val_r = value_rows + (unsigned long long)r * hidden_size;
        __nv_bfloat16* __restrict__ inter_r = conv_inter + (unsigned long long)r * state_stride;

        float q2 = 0.0f, k2 = 0.0f;
        for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            const float q = __bfloat162float(hyper_r[base + h]);
            const float k = __bfloat162float(key_r[base + h]);
            q2 += q * q;
            k2 += k * k;
        }
        q2 = warp_sum(q2);
        k2 = warp_sum(k2);
        if (lane == 0u) {
            q_parts[warp] = q2;
            k_parts[warp] = k2;
        }
        __syncthreads();
        if (warp == 0u) {
            float q = lane < warps ? q_parts[lane] : 0.0f;
            float k = lane < warps ? k_parts[lane] : 0.0f;
            q = warp_sum(q);
            k = warp_sum(k);
            if (lane == 0u) {
                shared[0] = rsqrtf(q / hidden_size + eps);
                shared[1] = rsqrtf(k / hidden_size + eps);
            }
        }
        __syncthreads();

        float dot = 0.0f;
        for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            const unsigned int i = base + h;
            const float q = __bfloat162float(hyper_r[i]) * shared[0]
                          * (1.0f + __bfloat162float(norm_query[i]));
            const float k = __bfloat162float(key_r[i]) * shared[1]
                          * (1.0f + __bfloat162float(norm_key[i]));
            dot += q * k;
        }
        dot = warp_sum(dot);
        if (lane == 0u) q_parts[warp] = dot;
        __syncthreads();
        if (warp == 0u) {
            float v = lane < warps ? q_parts[lane] : 0.0f;
            v = warp_sum(v);
            if (lane == 0u) {
                v /= sqrtf((float)hidden_size);
                const float signed_root = copysignf(sqrtf(fmaxf(fabsf(v), 1.0e-6f)), v);
                shared[2] = 1.0f / (1.0f + expf(-signed_root));
            }
        }
        __syncthreads();

        float gv2 = 0.0f;
        for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            const float gv = shared[2] * __bfloat162float(val_r[h]);
            gv2 += gv * gv;
        }
        gv2 = warp_sum(gv2);
        if (lane == 0u) q_parts[warp] = gv2;
        __syncthreads();
        if (warp == 0u) {
            float v = lane < warps ? q_parts[lane] : 0.0f;
            v = warp_sum(v);
            if (lane == 0u) shared[3] = rsqrtf(v / hidden_size + eps);
        }
        __syncthreads();

        const __nv_bfloat16* __restrict__ prev_state_base =
            (r == 0u) ? conv_state : (conv_inter + (unsigned long long)(r - 1u) * state_stride);
        const bool is_last_row = (r + 1u == rows);
        for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            const unsigned int i = base + h;
            const float gv = shared[2] * __bfloat162float(val_r[h]);
            const float x = gv * shared[3] * (1.0f + __bfloat162float(__ldg(&norm_conv[i])));
            const __nv_bfloat16* pstate = prev_state_base + i * 9u;
            __nv_bfloat16* istate = inter_r + i * 9u;
            const float s0 = __bfloat162float(pstate[0]);
            const float s3 = __bfloat162float(pstate[3]);
            const float s6 = __bfloat162float(pstate[6]);
            const __nv_bfloat16* w = conv_weight + i * 4u;
            float c = s0 * __bfloat162float(__ldg(&w[0]))
                    + s3 * __bfloat162float(__ldg(&w[1]))
                    + s6 * __bfloat162float(__ldg(&w[2]))
                    + x  * __bfloat162float(__ldg(&w[3]));
            c = c / (1.0f + expf(-c));
            const __nv_bfloat16 x_bf16 = __float2bfloat16(x);
            if (is_last_row) {
                __nv_bfloat16* state = conv_state + i * 9u;
                #pragma unroll
                for (unsigned int j = 0u; j < 8u; ++j) {
                    const __nv_bfloat16 sv = pstate[j + 1u];
                    istate[j] = sv;
                    state[j] = sv;
                }
                istate[8] = x_bf16;
                state[8] = x_bf16;
            } else {
                #pragma unroll
                for (unsigned int j = 0u; j < 8u; ++j) {
                    istate[j] = pstate[j + 1u];
                }
                istate[8] = x_bf16;
            }
            hyper_r[i] = __float2bfloat16(__bfloat162float(hyper_r[i]) + gv + c);
        }
        __syncthreads();
    }
}

// Ragged multi-sequence speculative verify PLE join + intermediate state capture
// in a single launch (`grid = [hc_count, num_seqs, 1]`, `block = [1024, 1, 1]`).
// Each `seq_descs[seq_idx]` packs (matching `encode_ragged_ple_fuse_desc` in Rust):
//   bits  0..11: `row_offset` (starting row in `hyper`/`key_rows`/`value_rows`, 0..4095)
//   bits 12..15: `rows`       (verify window length K_s for this sequence, 1..15)
//   bits 16..31: `slot_idx`   (sequence-owned persistent PLE state slot, 0..65535)
// Executes the exact same per-row reduction tree, arithmetic, and BF16 rounding
// (`0-ULP` bit-identical) as `qwen4_ple_fuse_verify_rows`.
extern "C" __global__ void __launch_bounds__(1024, 1) qwen4_ple_fuse_verify_rows_ragged(
    __nv_bfloat16* __restrict__ hyper,
    unsigned int hyper_row_elems,
    const __nv_bfloat16* __restrict__ key_rows,
    const __nv_bfloat16* __restrict__ value_rows,
    const __nv_bfloat16* __restrict__ norm_key,
    const __nv_bfloat16* __restrict__ norm_query,
    const __nv_bfloat16* __restrict__ norm_conv,
    const __nv_bfloat16* __restrict__ conv_weight,
    __nv_bfloat16* __restrict__ conv_state_base,
    __nv_bfloat16* __restrict__ conv_inter_base,
    unsigned int hidden_size,
    float eps,
    unsigned int max_spec_rows,
    const unsigned int* __restrict__ seq_descs) {
    const unsigned int stream_id = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int desc = __ldg(&seq_descs[seq_idx]);
    // Packed descriptor layout (matches `encode_ragged_ple_fuse_desc` in Rust):
    //   bits  0..11: row_offset (0..=4095)
    //   bits 12..15: rows       (1..=15)
    //   bits 16..31: slot_idx   (0..=65535)
    const unsigned int row_offset_u32 = desc & 0xFFFu;
    const unsigned int rows = (desc >> 12u) & 0xFu;
    const unsigned int slot_idx = desc >> 16u;
    if (rows == 0u) return;
    const unsigned int base = stream_id * hidden_size;
    const unsigned int residual_width = hidden_size * 4u;
    const unsigned int state_stride = residual_width * 9u;
    const unsigned long long row_offset = (unsigned long long)row_offset_u32;
    hyper += row_offset * hyper_row_elems;
    key_rows += row_offset * residual_width;
    value_rows += row_offset * hidden_size;
    __nv_bfloat16* __restrict__ conv_state =
        conv_state_base + (unsigned long long)slot_idx * state_stride;
    __nv_bfloat16* __restrict__ conv_inter =
        conv_inter_base + (unsigned long long)slot_idx * max_spec_rows * state_stride;
    __shared__ float q_parts[32], k_parts[32], shared[4];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warps = (blockDim.x + 31u) / 32u;

    for (unsigned int r = 0u; r < rows; ++r) {
        __nv_bfloat16* __restrict__ hyper_r = hyper + (unsigned long long)r * hyper_row_elems;
        const __nv_bfloat16* __restrict__ key_r = key_rows + (unsigned long long)r * residual_width;
        const __nv_bfloat16* __restrict__ val_r = value_rows + (unsigned long long)r * hidden_size;
        __nv_bfloat16* __restrict__ inter_r = conv_inter + (unsigned long long)r * state_stride;

        float q2 = 0.0f, k2 = 0.0f;
        for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            const float q = __bfloat162float(hyper_r[base + h]);
            const float k = __bfloat162float(key_r[base + h]);
            q2 += q * q;
            k2 += k * k;
        }
        q2 = warp_sum(q2);
        k2 = warp_sum(k2);
        if (lane == 0u) {
            q_parts[warp] = q2;
            k_parts[warp] = k2;
        }
        __syncthreads();
        if (warp == 0u) {
            float q = lane < warps ? q_parts[lane] : 0.0f;
            float k = lane < warps ? k_parts[lane] : 0.0f;
            q = warp_sum(q);
            k = warp_sum(k);
            if (lane == 0u) {
                shared[0] = rsqrtf(q / hidden_size + eps);
                shared[1] = rsqrtf(k / hidden_size + eps);
            }
        }
        __syncthreads();

        float dot = 0.0f;
        for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            const unsigned int i = base + h;
            const float q = __bfloat162float(hyper_r[i]) * shared[0]
                          * (1.0f + __bfloat162float(norm_query[i]));
            const float k = __bfloat162float(key_r[i]) * shared[1]
                          * (1.0f + __bfloat162float(norm_key[i]));
            dot += q * k;
        }
        dot = warp_sum(dot);
        if (lane == 0u) q_parts[warp] = dot;
        __syncthreads();
        if (warp == 0u) {
            float v = lane < warps ? q_parts[lane] : 0.0f;
            v = warp_sum(v);
            if (lane == 0u) {
                v /= sqrtf((float)hidden_size);
                const float signed_root = copysignf(sqrtf(fmaxf(fabsf(v), 1.0e-6f)), v);
                shared[2] = 1.0f / (1.0f + expf(-signed_root));
            }
        }
        __syncthreads();

        float gv2 = 0.0f;
        for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            const float gv = shared[2] * __bfloat162float(val_r[h]);
            gv2 += gv * gv;
        }
        gv2 = warp_sum(gv2);
        if (lane == 0u) q_parts[warp] = gv2;
        __syncthreads();
        if (warp == 0u) {
            float v = lane < warps ? q_parts[lane] : 0.0f;
            v = warp_sum(v);
            if (lane == 0u) shared[3] = rsqrtf(v / hidden_size + eps);
        }
        __syncthreads();

        const __nv_bfloat16* __restrict__ prev_state_base =
            (r == 0u) ? conv_state : (conv_inter + (unsigned long long)(r - 1u) * state_stride);
        const bool is_last_row = (r + 1u == rows);
        for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            const unsigned int i = base + h;
            const float gv = shared[2] * __bfloat162float(val_r[h]);
            const float x = gv * shared[3] * (1.0f + __bfloat162float(__ldg(&norm_conv[i])));
            const __nv_bfloat16* pstate = prev_state_base + i * 9u;
            __nv_bfloat16* istate = inter_r + i * 9u;
            const float s0 = __bfloat162float(pstate[0]);
            const float s3 = __bfloat162float(pstate[3]);
            const float s6 = __bfloat162float(pstate[6]);
            const __nv_bfloat16* w = conv_weight + i * 4u;
            float c = s0 * __bfloat162float(__ldg(&w[0]))
                    + s3 * __bfloat162float(__ldg(&w[1]))
                    + s6 * __bfloat162float(__ldg(&w[2]))
                    + x  * __bfloat162float(__ldg(&w[3]));
            c = c / (1.0f + expf(-c));
            const __nv_bfloat16 x_bf16 = __float2bfloat16(x);
            if (is_last_row) {
                __nv_bfloat16* state = conv_state + i * 9u;
                #pragma unroll
                for (unsigned int j = 0u; j < 8u; ++j) {
                    const __nv_bfloat16 sv = pstate[j + 1u];
                    istate[j] = sv;
                    state[j] = sv;
                }
                istate[8] = x_bf16;
                state[8] = x_bf16;
            } else {
                #pragma unroll
                for (unsigned int j = 0u; j < 8u; ++j) {
                    istate[j] = pstate[j + 1u];
                }
                istate[8] = x_bf16;
            }
            hyper_r[i] = __float2bfloat16(__bfloat162float(hyper_r[i]) + gv + c);
        }
        __syncthreads();
    }
}

// Compute every token/stream PLE gate and normalized short-conv input in
// parallel. `key` is [M,4H], `value` is [M,H], and all other row-major
// matrices are [M,4H]. The recurrent depthwise convolution is a second pass.
extern "C" __global__ void qwen4_ple_prepare_prefill(
    const __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ norm_key,
    const __nv_bfloat16* __restrict__ norm_query,
    const __nv_bfloat16* __restrict__ norm_conv,
    float* __restrict__ gated_value,
    float* __restrict__ conv_input,
    unsigned int num_tokens,
    unsigned int hidden_size,
    float eps) {
    const unsigned int token = blockIdx.x;
    const unsigned int stream_id = blockIdx.y;
    if (token >= num_tokens || stream_id >= 4u) return;
    const unsigned long long residual_width = (unsigned long long)hidden_size * 4ull;
    const unsigned long long base = (unsigned long long)token * residual_width
                                  + (unsigned long long)stream_id * hidden_size;
    const unsigned int weight_base = stream_id * hidden_size;
    float q2 = 0.0f, k2 = 0.0f;
    for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const float q = __bfloat162float(hyper[base + h]);
        const float k = __bfloat162float(key[base + h]);
        q2 += q * q;
        k2 += k * k;
    }
    q2 = warp_sum(q2);
    k2 = warp_sum(k2);
    __shared__ float q_parts[32], k_parts[32], shared[4];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    if (lane == 0) {
        q_parts[warp] = q2;
        k_parts[warp] = k2;
    }
    __syncthreads();
    if (warp == 0) {
        const unsigned int warps = (blockDim.x + 31u) / 32u;
        float q = lane < warps ? q_parts[lane] : 0.0f;
        float k = lane < warps ? k_parts[lane] : 0.0f;
        q = warp_sum(q);
        k = warp_sum(k);
        if (lane == 0) {
            shared[0] = rsqrtf(q / hidden_size + eps);
            shared[1] = rsqrtf(k / hidden_size + eps);
        }
    }
    __syncthreads();

    float dot = 0.0f;
    for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const unsigned int i = weight_base + h;
        const float q = __bfloat162float(hyper[base + h]) * shared[0]
                      * (1.0f + __bfloat162float(norm_query[i]));
        const float k = __bfloat162float(key[base + h]) * shared[1]
                      * (1.0f + __bfloat162float(norm_key[i]));
        dot += q * k;
    }
    dot = warp_sum(dot);
    if (lane == 0) q_parts[warp] = dot;
    __syncthreads();
    if (warp == 0) {
        const unsigned int warps = (blockDim.x + 31u) / 32u;
        float sum = lane < warps ? q_parts[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) {
            sum /= sqrtf((float)hidden_size);
            const float root = copysignf(sqrtf(fmaxf(fabsf(sum), 1.0e-6f)), sum);
            shared[2] = 1.0f / (1.0f + expf(-root));
        }
    }
    __syncthreads();

    float gv2 = 0.0f;
    const unsigned long long value_base = (unsigned long long)token * hidden_size;
    for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const float gv = shared[2] * __bfloat162float(value[value_base + h]);
        gv2 += gv * gv;
    }
    gv2 = warp_sum(gv2);
    if (lane == 0) q_parts[warp] = gv2;
    __syncthreads();
    if (warp == 0) {
        const unsigned int warps = (blockDim.x + 31u) / 32u;
        float sum = lane < warps ? q_parts[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) shared[3] = rsqrtf(sum / hidden_size + eps);
    }
    __syncthreads();
    for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const unsigned int i = weight_base + h;
        const float gv = shared[2] * __bfloat162float(value[value_base + h]);
        gated_value[base + h] = gv;
        conv_input[base + h] =
            gv * shared[3] * (1.0f + __bfloat162float(norm_conv[i]));
    }
}

__device__ __forceinline__ float qwen4_ple_history_bf16(float value) {
    return __bfloat162float(__float2bfloat16(value));
}

// The PLE convolution stores normalized inputs rather than convolution
// outputs, so all prompt tokens can be evaluated independently from the
// immutable entry history plus the prepared chunk matrix.
extern "C" __global__ void qwen4_ple_conv_inject_prefill(
    __nv_bfloat16* __restrict__ hyper,
    const float* __restrict__ gated_value,
    const float* __restrict__ conv_input,
    const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* __restrict__ entry_state,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int reset_state) {
    const unsigned int token = blockIdx.x;
    const unsigned int stream_id = blockIdx.y;
    const unsigned int h = blockIdx.z * blockDim.x + threadIdx.x;
    if (token >= num_tokens || stream_id >= 4u || h >= hidden_size) return;
    const unsigned long long residual_width = (unsigned long long)hidden_size * 4ull;
    const unsigned int i = stream_id * hidden_size + h;
    const unsigned long long out = (unsigned long long)token * residual_width + i;
    const __nv_bfloat16* state = entry_state + i * 9u;
    const float s0 = token >= 9u
        ? qwen4_ple_history_bf16(
            conv_input[(unsigned long long)(token - 9u) * residual_width + i])
        : (reset_state ? 0.0f : __bfloat162float(state[token]));
    const float s3 = token >= 6u
        ? qwen4_ple_history_bf16(
            conv_input[(unsigned long long)(token - 6u) * residual_width + i])
        : (reset_state ? 0.0f : __bfloat162float(state[token + 3u]));
    const float s6 = token >= 3u
        ? qwen4_ple_history_bf16(
            conv_input[(unsigned long long)(token - 3u) * residual_width + i])
        : (reset_state ? 0.0f : __bfloat162float(state[token + 6u]));
    const float x = conv_input[out];
    const __nv_bfloat16* weight = conv_weight + i * 4u;
    float c = s0 * __bfloat162float(weight[0])
            + s3 * __bfloat162float(weight[1])
            + s6 * __bfloat162float(weight[2])
            + x  * __bfloat162float(weight[3]);
    c = c / (1.0f + expf(-c));
    hyper[out] = __float2bfloat16(
        __bfloat162float(hyper[out]) + gated_value[out] + c);
}

// Commit the last nine prepared inputs after every parallel consumer has
// finished. One thread owns one residual channel and snapshots old history
// before overwriting it, which also handles chunks shorter than nine tokens.
extern "C" __global__ void qwen4_ple_commit_prefill_state(
    const float* __restrict__ conv_input,
    __nv_bfloat16* __restrict__ state,
    unsigned int num_tokens,
    unsigned int residual_width,
    unsigned int reset_state) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= residual_width || num_tokens == 0u) return;
    __nv_bfloat16 old[9];
    #pragma unroll
    for (unsigned int j = 0; j < 9u; ++j) old[j] = state[i * 9u + j];
    #pragma unroll
    for (unsigned int j = 0; j < 9u; ++j) {
        const long long source_token = (long long)num_tokens - 9ll + (long long)j;
        __nv_bfloat16 next;
        if (source_token >= 0) {
            next = __float2bfloat16(
                conv_input[(unsigned long long)source_token * residual_width + i]);
        } else if (reset_state) {
            next = __float2bfloat16(0.0f);
        } else {
            next = old[num_tokens + j];
        }
        state[i * 9u + j] = next;
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(0xffffffff, v, d);
    return v;
}

extern "C" __global__ void qwen4_hc_group_norm(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    unsigned int hc_count,
    float eps) {
    const unsigned int token = blockIdx.x;
    const unsigned int stream_id = blockIdx.y;
    if (stream_id >= hc_count) return;
    const unsigned long long base = ((unsigned long long)token * hc_count + stream_id) * hidden_size;
    if (hidden_size <= 3072u && blockDim.x == 1024u) {
        const unsigned int i0 = threadIdx.x;
        const unsigned int i1 = i0 + 1024u;
        const unsigned int i2 = i0 + 2048u;
        const float x0 = i0 < hidden_size ? __bfloat162float(__ldg(input + base + i0)) : 0.0f;
        const float x1 = i1 < hidden_size ? __bfloat162float(__ldg(input + base + i1)) : 0.0f;
        const float x2 = i2 < hidden_size ? __bfloat162float(__ldg(input + base + i2)) : 0.0f;
        float sum = x0 * x0 + x1 * x1;
        if (i2 < hidden_size) sum += x2 * x2;
        sum = warp_sum(sum);
        __shared__ float partial[32];
        const unsigned int lane = threadIdx.x & 31u;
        const unsigned int warp = threadIdx.x >> 5;
        if (lane == 0u) partial[warp] = sum;
        __syncthreads();
        if (warp == 0u) {
            float v = partial[lane];
            v = warp_sum(v);
            if (lane == 0u) partial[0] = rsqrtf(v / hidden_size + eps);
        }
        __syncthreads();
        const float inv = partial[0];
        const unsigned int wbase = stream_id * hidden_size;
        if (i0 < hidden_size) {
            const float w0 = __bfloat162float(__ldg(weight + wbase + i0));
            output[base + i0] = __float2bfloat16(x0 * inv * (1.0f + w0));
        }
        if (i1 < hidden_size) {
            const float w1 = __bfloat162float(__ldg(weight + wbase + i1));
            output[base + i1] = __float2bfloat16(x1 * inv * (1.0f + w1));
        }
        if (i2 < hidden_size) {
            const float w2 = __bfloat162float(__ldg(weight + wbase + i2));
            output[base + i2] = __float2bfloat16(x2 * inv * (1.0f + w2));
        }
        return;
    }
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float x = __bfloat162float(input[base + i]);
        sum += x * x;
    }
    sum = warp_sum(sum);
    __shared__ float partial[32];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float v = lane < (blockDim.x + 31) / 32 ? partial[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) partial[0] = rsqrtf(v / hidden_size + eps);
    }
    __syncthreads();
    const float inv = partial[0];
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float x = __bfloat162float(input[base + i]);
        const float w = __bfloat162float(weight[stream_id * hidden_size + i]);
        output[base + i] = __float2bfloat16(x * inv * (1.0f + w));
    }
}

// Fuses `qwen4_hc_inject` / `qwen4_hc_inject_saved` with the immediately
// following `qwen4_hc_group_norm` across all 4 hyper-connection streams in one
// CTA per token (`grid = [num_tokens, 1, 1], block = [1024, 1, 1]`).
// Preserves 100% 0-ULP bit-identity by rounding `hyper` to BF16 before
// computing the group-norm sum of squares and scaling.
extern "C" __global__ void __launch_bounds__(1024, 1) qwen4_hc_inject_group_norm(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    const __nv_bfloat16* __restrict__ inject_opt,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ residual,
    unsigned int hidden_size,
    unsigned int hc_count,
    float eps) {
    const unsigned int token = blockIdx.x;
    if (hc_count != 4u || hidden_size > 3072u || blockDim.x != 1024u) return;
    const unsigned long long r = (unsigned long long)hidden_size * 4ull;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;

    float scale[4];
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const __nv_bfloat16 raw = (inject_opt != nullptr)
            ? __ldg(inject_opt + (unsigned long long)token * 4ull + (unsigned int)s)
            : __ldg(residual + (unsigned long long)token * r + (r - 4ull) + (unsigned int)s);
        scale[s] = __bfloat162float(raw);
    }

    const unsigned int i0 = tid;
    const unsigned int i1 = tid + 1024u;
    const unsigned int i2 = tid + 2048u;

    const unsigned long long core_base = (unsigned long long)token * hidden_size;
    const float c0 = i0 < hidden_size ? __bfloat162float(__ldg(core + core_base + i0)) : 0.0f;
    const float c1 = i1 < hidden_size ? __bfloat162float(__ldg(core + core_base + i1)) : 0.0f;
    const float c2 = i2 < hidden_size ? __bfloat162float(__ldg(core + core_base + i2)) : 0.0f;

    float x[4][3];
    float sum[4];
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const unsigned long long base = ((unsigned long long)token * 4ull + (unsigned int)s) * hidden_size;
        if (i0 < hidden_size) {
            const float old0 = __bfloat162float(hyper[base + i0]);
            const __nv_bfloat16 h0 = __float2bfloat16(old0 + c0 * scale[s]);
            hyper[base + i0] = h0;
            x[s][0] = __bfloat162float(h0);
        } else {
            x[s][0] = 0.0f;
        }
        if (i1 < hidden_size) {
            const float old1 = __bfloat162float(hyper[base + i1]);
            const __nv_bfloat16 h1 = __float2bfloat16(old1 + c1 * scale[s]);
            hyper[base + i1] = h1;
            x[s][1] = __bfloat162float(h1);
        } else {
            x[s][1] = 0.0f;
        }
        if (i2 < hidden_size) {
            const float old2 = __bfloat162float(hyper[base + i2]);
            const __nv_bfloat16 h2 = __float2bfloat16(old2 + c2 * scale[s]);
            hyper[base + i2] = h2;
            x[s][2] = __bfloat162float(h2);
        } else {
            x[s][2] = 0.0f;
        }
        float s_sq = x[s][0] * x[s][0] + x[s][1] * x[s][1];
        if (i2 < hidden_size) s_sq += x[s][2] * x[s][2];
        sum[s] = warp_sum(s_sq);
    }

    __shared__ float partial[4][32];
    if (lane == 0u) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) partial[s][warp] = sum[s];
    }
    __syncthreads();
    if (warp < 4u) {
        float v = partial[warp][lane];
        v = warp_sum(v);
        if (lane == 0u) partial[warp][0] = rsqrtf(v / hidden_size + eps);
    }
    __syncthreads();

    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const float inv = partial[s][0];
        const unsigned long long base = ((unsigned long long)token * 4ull + (unsigned int)s) * hidden_size;
        const unsigned long long wbase = (unsigned long long)(unsigned int)s * hidden_size;
        if (i0 < hidden_size) {
            const float w0 = __bfloat162float(__ldg(weight + wbase + i0));
            residual[base + i0] = __float2bfloat16(x[s][0] * inv * (1.0f + w0));
        }
        if (i1 < hidden_size) {
            const float w1 = __bfloat162float(__ldg(weight + wbase + i1));
            residual[base + i1] = __float2bfloat16(x[s][1] * inv * (1.0f + w1));
        }
        if (i2 < hidden_size) {
            const float w2 = __bfloat162float(__ldg(weight + wbase + i2));
            residual[base + i2] = __float2bfloat16(x[s][2] * inv * (1.0f + w2));
        }
    }
}

// Vectorized 4-CTA/token twin of `qwen4_hc_inject_group_norm` for canonical
// `hidden_size == 2560, hc_count == 4` with 16-byte aligned pointers
// (`grid = [num_tokens, 4, 1], block = [320, 1, 1]`, cluster `(1, 4, 1)`).
// Each CTA owns one `(token, stream)` in a single branchless `uint4` (BF16x8)
// pass (`320 * 8 = 2560`), reproduces the exact 1024-thread virtual-warp FP32
// reduction tree (`0-ULP` bit-identical `hyper`, `residual`, and optional
// `rstd_out`), and synchronizes the 4 stream CTAs of each token via a hardware
// cluster barrier so in-place reads of the saved-inject residual tail never
// race with stream 3's normalized output store.
extern "C" __global__ void __cluster_dims__(1, 4, 1) __launch_bounds__(320, 4)
qwen4_hc_inject_group_norm_v2(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    const __nv_bfloat16* __restrict__ inject_opt,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ residual,
    float* __restrict__ rstd_out,
    unsigned int hidden_size,
    unsigned int hc_count,
    float eps) {
    if (hidden_size != 2560u || hc_count != 4u || blockDim.x != 320u) return;
    const unsigned int token = blockIdx.x;
    const unsigned int stream_id = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;

    const unsigned long long stream_base =
        ((unsigned long long)token * 4ull + (unsigned long long)stream_id) * 2560ull;
    const unsigned long long core_base = (unsigned long long)token * 2560ull;
    const unsigned long long weight_base = (unsigned long long)stream_id * 2560ull;

    const __nv_bfloat16 raw_scale = (inject_opt != nullptr)
        ? __ldg(inject_opt + (unsigned long long)token * 4ull + stream_id)
        : __ldg(residual + (unsigned long long)token * 10240ull + 10236ull + stream_id);
    const float scale = __bfloat162float(raw_scale);

    const uint4 h_vec = reinterpret_cast<const uint4*>(hyper + stream_base)[tid];
    const uint4 c_vec = __ldg(&reinterpret_cast<const uint4*>(core + core_base)[tid]);

    const unsigned int h_u[4] = {h_vec.x, h_vec.y, h_vec.z, h_vec.w};
    const unsigned int c_u[4] = {c_vec.x, c_vec.y, c_vec.z, c_vec.w};
    unsigned int h_out_u[4];
    float x[8];
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const __nv_bfloat162 hb = *reinterpret_cast<const __nv_bfloat162*>(&h_u[p]);
        const __nv_bfloat162 cb = *reinterpret_cast<const __nv_bfloat162*>(&c_u[p]);
        const float2 hf = __bfloat1622float2(hb);
        const float2 cf = __bfloat1622float2(cb);
        const __nv_bfloat16 lo = __float2bfloat16(hf.x + cf.x * scale);
        const __nv_bfloat16 hi = __float2bfloat16(hf.y + cf.y * scale);
        const __nv_bfloat162 out_p = __halves2bfloat162(lo, hi);
        h_out_u[p] = *reinterpret_cast<const unsigned int*>(&out_p);
        x[2 * p] = __bfloat162float(lo);
        x[2 * p + 1] = __bfloat162float(hi);
    }
    reinterpret_cast<uint4*>(hyper + stream_base)[tid] =
        make_uint4(h_out_u[0], h_out_u[1], h_out_u[2], h_out_u[3]);

    __shared__ float smem_x[192][9];
    __shared__ float partial[32];
    __shared__ float inv_shared;

    if (tid >= 128u) {
        const unsigned int row = tid - 128u;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            smem_x[row][e] = x[e];
        }
    }
    if (inject_opt == nullptr) {
        asm volatile("barrier.cluster.arrive.aligned;\n" ::: "memory");
    }
    __syncthreads();

    if (tid < 128u) {
        float v[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float x1 = smem_x[tid][e];
            float s = x[e] * x[e] + x1 * x1;
            if (tid < 64u) {
                const float x2 = smem_x[128u + tid][e];
                s += x2 * x2;
            }
            v[e] = s;
        }
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] += __shfl_xor_sync(0xffffffffu, v[e], 2);
        }
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] += __shfl_xor_sync(0xffffffffu, v[e], 1);
        }
        if ((tid & 3u) == 0u) {
            const float y0 = v[0] + v[4];
            const float y1 = v[1] + v[5];
            const float y2 = v[2] + v[6];
            const float y3 = v[3] + v[7];
            const float z0 = y0 + y2;
            const float z1 = y1 + y3;
            partial[tid >> 2] = z0 + z1;
        }
    }
    __syncthreads();

    if (warp == 0u) {
        float tot = partial[lane];
        tot = warp_sum(tot);
        if (lane == 0u) {
            const float inv = rsqrtf(tot / 2560.0f + eps);
            inv_shared = inv;
            if (rstd_out != nullptr) {
                rstd_out[(unsigned long long)token * 4ull + stream_id] = inv;
            }
        }
    }
    __syncthreads();

    const float inv = inv_shared;
    const uint4 w_vec = __ldg(&reinterpret_cast<const uint4*>(weight + weight_base)[tid]);
    const unsigned int w_u[4] = {w_vec.x, w_vec.y, w_vec.z, w_vec.w};
    unsigned int r_out_u[4];
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const __nv_bfloat162 wb = *reinterpret_cast<const __nv_bfloat162*>(&w_u[p]);
        const float2 wf = __bfloat1622float2(wb);
        const __nv_bfloat16 lo = __float2bfloat16(x[2 * p] * inv * (1.0f + wf.x));
        const __nv_bfloat16 hi = __float2bfloat16(x[2 * p + 1] * inv * (1.0f + wf.y));
        const __nv_bfloat162 out_p = __halves2bfloat162(lo, hi);
        r_out_u[p] = *reinterpret_cast<const unsigned int*>(&out_p);
    }
    if (inject_opt == nullptr) {
        asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
    }
    reinterpret_cast<uint4*>(residual + stream_base)[tid] =
        make_uint4(r_out_u[0], r_out_u[1], r_out_u[2], r_out_u[3]);
}

extern "C" __global__ void qwen4_hc_silu_div(
    __nv_bfloat16* values, unsigned int numel, float divisor) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    float x = __bfloat162float(values[i]) / divisor;
    values[i] = __float2bfloat16(x / (1.0f + expf(-x)));
}

extern "C" __global__ void qwen4_hc_mix(
    const __nv_bfloat16* __restrict__ normed,
    __nv_bfloat16* __restrict__ mix_logits,
    __nv_bfloat16* __restrict__ inject_logits,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int hidden_size,
    unsigned int hc_count) {
    const unsigned int token = blockIdx.x;
    const unsigned int h = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned long long r = (unsigned long long)hidden_size * hc_count;
    if (h < hidden_size) {
        float acc = 0.0f;
        for (unsigned int s = 0; s < hc_count; ++s) {
            const unsigned long long i = (unsigned long long)token * r + (unsigned long long)s * hidden_size + h;
            const float g0 = __bfloat162float(mix_logits[i]);
            const float g = 1.0f / (1.0f + expf(-g0));
            acc += g * __bfloat162float(normed[i]);
        }
        mixed[(unsigned long long)token * hidden_size + h] = __float2bfloat16(acc / hc_count);
    }
    if (inject_logits != nullptr && blockIdx.y == 0 && h < hc_count) {
        const unsigned long long i = (unsigned long long)token * hc_count + h;
        const float x = __bfloat162float(inject_logits[i]) / hc_count;
        inject_logits[i] = __float2bfloat16(2.0f / (1.0f + expf(-x)));
    }
}

// Precondition: CTA `blockIdx.y == gridDim.y - 1u` overwrites the tail
// `normed[token * r + (r - hc_count) + 0..hc_count)` after CTA-local
// `__syncthreads()`. To prevent an inter-CTA data race with `gridDim.y - 2u`,
// the final CTA must own all `hc_count` trailing hidden positions:
//   blockDim.x >= hc_count && (hidden_size % blockDim.x == 0 || hidden_size % blockDim.x >= hc_count).
extern "C" __global__ void qwen4_hc_mix_save_inject(
    __nv_bfloat16* __restrict__ normed,
    __nv_bfloat16* __restrict__ mix_logits,
    __nv_bfloat16* __restrict__ inject_logits,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int hidden_size,
    unsigned int hc_count) {
    if (blockDim.x < hc_count
        || ((hidden_size % blockDim.x) != 0u && (hidden_size % blockDim.x) < hc_count)) {
        asm volatile("trap;");
        return;
    }
    const unsigned int token = blockIdx.x;
    const unsigned int h = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned long long r = (unsigned long long)hidden_size * hc_count;
    if (h < hidden_size) {
        float acc = 0.0f;
        for (unsigned int s = 0; s < hc_count; ++s) {
            const unsigned long long i = (unsigned long long)token * r + (unsigned long long)s * hidden_size + h;
            const float g0 = __bfloat162float(mix_logits[i]);
            const float g = 1.0f / (1.0f + expf(-g0));
            acc += g * __bfloat162float(normed[i]);
        }
        mixed[(unsigned long long)token * hidden_size + h] = __float2bfloat16(acc / hc_count);
    }
    if (blockIdx.y == gridDim.y - 1u) {
        __syncthreads();
        if (threadIdx.x < hc_count) {
            const unsigned long long i = (unsigned long long)token * hc_count + threadIdx.x;
            const float x = __bfloat162float(inject_logits[i]) / hc_count;
            const __nv_bfloat16 scale = __float2bfloat16(2.0f / (1.0f + expf(-x)));
            inject_logits[i] = scale;
            normed[(unsigned long long)token * r + (r - hc_count) + threadIdx.x] = scale;
        }
    }
}

extern "C" __global__ void qwen4_hc_inject(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    const __nv_bfloat16* __restrict__ inject,
    unsigned int hidden_size,
    unsigned int hc_count) {
    const unsigned int token = blockIdx.x;
    const unsigned int i = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned long long r = (unsigned long long)hidden_size * hc_count;
    if (i >= r) return;
    const unsigned int s = i / hidden_size;
    const unsigned int h = i - s * hidden_size;
    const unsigned long long base = (unsigned long long)token * r + i;
    const float old = __bfloat162float(hyper[base]);
    const float add = __bfloat162float(core[(unsigned long long)token * hidden_size + h]);
    const float scale = __bfloat162float(inject[(unsigned long long)token * hc_count + s]);
    hyper[base] = __float2bfloat16(old + add * scale);
}

// Preserve the per-token injection scales in the dead tail of each normalized
// hyper row. This avoids thousands of tiny D2D copies during long prefill.
extern "C" __global__ void qwen4_hc_save_inject(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ inject,
    unsigned int hidden_size,
    unsigned int hc_count,
    unsigned int num_tokens) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)num_tokens * hc_count;
    if (i >= total) return;
    const unsigned long long token = i / hc_count;
    const unsigned long long stream_id = i - token * hc_count;
    const unsigned long long r = (unsigned long long)hidden_size * hc_count;
    residual[token * r + (r - hc_count) + stream_id] = inject[i];
}

// Batched injection reading the scales saved by qwen4_hc_save_inject.
extern "C" __global__ void qwen4_hc_inject_saved(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    const __nv_bfloat16* __restrict__ residual,
    unsigned int hidden_size,
    unsigned int hc_count) {
    const unsigned int token = blockIdx.x;
    const unsigned int i = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned long long r = (unsigned long long)hidden_size * hc_count;
    if (i >= r) return;
    const unsigned int stream_id = i / hidden_size;
    const unsigned int h = i - stream_id * hidden_size;
    const unsigned long long row_base = (unsigned long long)token * r;
    const float old = __bfloat162float(hyper[row_base + i]);
    const float add = __bfloat162float(core[(unsigned long long)token * hidden_size + h]);
    const float scale = __bfloat162float(
        residual[row_base + (r - hc_count) + stream_id]);
    hyper[row_base + i] = __float2bfloat16(old + add * scale);
}

// Stage/pack exact small-M mixed rows through the otherwise-dead first H
// elements of each residual row. This permits an arbitrary-length prefill to
// use byte-exact M<=32 projections while presenting a contiguous [M,H] input
// to the tuned attention/SSM cores.
extern "C" __global__ void qwen4_hc_stage_mixed(
    const __nv_bfloat16* __restrict__ mixed,
    __nv_bfloat16* __restrict__ residual,
    unsigned int token_start,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int hc_count) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)num_tokens * hidden_size;
    if (i >= total) return;
    const unsigned long long token = i / hidden_size;
    const unsigned long long h = i - token * hidden_size;
    const unsigned long long r = (unsigned long long)hidden_size * hc_count;
    residual[((unsigned long long)token_start + token) * r + h] = mixed[i];
}

extern "C" __global__ void qwen4_hc_pack_mixed(
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int hc_count) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)num_tokens * hidden_size;
    if (i >= total) return;
    const unsigned long long token = i / hidden_size;
    const unsigned long long h = i - token * hidden_size;
    const unsigned long long r = (unsigned long long)hidden_size * hc_count;
    mixed[i] = residual[token * r + h];
}

extern "C" __global__ void qwen4_hc_expand_embedding(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    unsigned int hc_count) {
    const unsigned int token = blockIdx.x;
    const unsigned int i = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned long long r = (unsigned long long)hidden_size * hc_count;
    if (i < r) output[(unsigned long long)token * r + i] = input[(unsigned long long)token * hidden_size + (i % hidden_size)];
}

template <typename InputT>
__device__ __forceinline__ float qwen4_read(InputT x) { return (float)x; }
template <>
__device__ __forceinline__ float qwen4_read(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename InputT>
__device__ void qwen4_gated_rms_sigmoid_body(
    const InputT* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    unsigned int hidden_size, float eps, unsigned int gate_stride) {
    const unsigned int token = blockIdx.x;
    const InputT* x = input + (unsigned long long)token * hidden_size;
    const __nv_bfloat16* z = gate + (unsigned long long)token * gate_stride;
    __nv_bfloat16* out = output + (unsigned long long)token * hidden_size;
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float v = qwen4_read(x[i]);
        sum += v * v;
    }
    sum = warp_sum(sum);
    __shared__ float parts[32];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    if (lane == 0) parts[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float v = lane < (blockDim.x + 31) / 32 ? parts[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) parts[0] = rsqrtf(v / hidden_size + eps);
    }
    __syncthreads();
    const float inv = parts[0];
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float v = qwen4_read(x[i]);
        const float w = __bfloat162float(weight[i]);
        const float g = __bfloat162float(z[i]);
        out[i] = __float2bfloat16(v * inv * w / (1.0f + expf(-g)));
    }
}

extern "C" __global__ void qwen4_gated_rms_norm_sigmoid(
    const __nv_bfloat16* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    unsigned int hidden_size, float eps, unsigned int gate_stride,
    unsigned int group_size) {
    (void)group_size;
    qwen4_gated_rms_sigmoid_body(input, gate, weight, output, hidden_size, eps, gate_stride);
}

extern "C" __global__ void qwen4_gated_rms_norm_sigmoid_f32(
    const float* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    unsigned int hidden_size, float eps, unsigned int gate_stride,
    unsigned int group_size) {
    (void)group_size;
    qwen4_gated_rms_sigmoid_body(input, gate, weight, output, hidden_size, eps, gate_stride);
}

// ---------------------------------------------------------------------------
// Fused PLE prefill join (ATLAS_QWEN4_PLE_PREFILL_FUSED, default on; set to 0 to disable).
//
// Replaces qwen4_ple_prepare_prefill + qwen4_ple_conv_inject_prefill +
// qwen4_ple_commit_prefill_state (and the block pipeline's history commit)
// without the two fp32 [M, 4H] intermediates. prepare stores them unrounded:
//   gated_value[t, s, h] = g[t, s] * value[t, h]
//   conv_input[t, s, h]  = gated_value[t, s, h] * r[t, s] * (1 + norm_conv[s, h])
// with g = prepare's shared[2] (the sigmoid gate) and r = shared[3] (rsqrt of
// the gated value's mean square). qwen4_ple_gate_prefill reproduces g and r
// with prepare's exact fp32 reduction tree and stores them (8 bytes per token
// and stream); the inject and commit kernels recompute the two elementwise
// terms with the same expressions, so every value the old kernels read from
// DRAM is recomputed bit for bit (fmad=false: no contraction either way).
// The causal conv taps of tokens before the launch's first token come from
// the chunk's own value rows and gate scalars (chunk-relative token >= 0) or
// the slot's entry state (token < 0), which is what the pipeline's history
// commit wrote, so a launch over any token range of the chunk equals the
// whole-chunk launch.
// ---------------------------------------------------------------------------

// The physical block of qwen4_ple_gate_prefill and the block of
// qwen4_ple_prepare_prefill whose reduction tree it emulates.
#define QWEN4_PLE_GATE_THREADS 256u
#define QWEN4_PLE_PREPARE_THREADS 1024u
// Tokens per thread run in qwen4_ple_conv_inject_fused_prefill (rust:
// qwen4_ple_fused::INJECT_RUN must match).
#define QWEN4_PLE_INJECT_RUN 64u

// One 256-thread block per (token, stream): grid.x = 4 * num_tokens, stream
// fastest so the four blocks that read a token's value row run together.
// Physical thread tid owns prepare's virtual threads v = tid + 256 r
// (r = 0..3). Virtual thread v sums h = v, v + 1024, v + 2048 in that order,
// exactly as prepare's thread v; virtual warp v / 32 = warp + 8 r has lane
// tid % 32, so the per-r butterfly is prepare's warp_sum of that warp and
// the 32 virtual-warp partials are combined by warp 0 with the same
// butterfly prepare's warp 0 applies to its 32 warp partials.
extern "C" __global__ void __launch_bounds__(256) qwen4_ple_gate_prefill(
    const __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ norm_key,
    const __nv_bfloat16* __restrict__ norm_query,
    float* __restrict__ gate,
    unsigned int token_start,
    unsigned int num_tokens,
    unsigned int hidden_size,
    float eps) {
    const unsigned int row = blockIdx.x >> 2;
    const unsigned int stream_id = blockIdx.x & 3u;
    if (row >= num_tokens) return;
    const unsigned int token = token_start + row;
    const unsigned long long row_width = (unsigned long long)hidden_size * 4ull;
    const unsigned long long base = (unsigned long long)token * row_width
                                  + (unsigned long long)stream_id * hidden_size;
    const unsigned int weight_base = stream_id * hidden_size;
    __shared__ float q_parts[32], k_parts[32], d_parts[32], shared[4];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;

    float q2v[4], k2v[4];
    float dotv[4];
    if (hidden_size == 2560u) {
        float q_reg[4][3], k_reg[4][3];
        float nq_reg[4][3], nk_reg[4][3];
        #pragma unroll
        for (unsigned int r = 0; r < 4u; ++r) {
            float q2 = 0.0f, k2 = 0.0f;
            const unsigned int h0 = threadIdx.x + r * QWEN4_PLE_GATE_THREADS;
            #pragma unroll
            for (unsigned int s = 0; s < 3u; ++s) {
                const unsigned int h = h0 + s * QWEN4_PLE_PREPARE_THREADS;
                if (h < 2560u) {
                    const float q = __bfloat162float(__ldg(&hyper[base + h]));
                    const float k = __bfloat162float(__ldg(&key[base + h]));
                    q_reg[r][s] = q;
                    k_reg[r][s] = k;
                    q2 += q * q;
                    k2 += k * k;
                }
            }
            q2v[r] = warp_sum(q2);
            k2v[r] = warp_sum(k2);
        }
        if (lane == 0) {
            #pragma unroll
            for (unsigned int r = 0; r < 4u; ++r) {
                q_parts[warp + 8u * r] = q2v[r];
                k_parts[warp + 8u * r] = k2v[r];
            }
        }
        #pragma unroll
        for (unsigned int r = 0; r < 4u; ++r) {
            const unsigned int h0 = threadIdx.x + r * QWEN4_PLE_GATE_THREADS;
            #pragma unroll
            for (unsigned int s = 0; s < 3u; ++s) {
                const unsigned int h = h0 + s * QWEN4_PLE_PREPARE_THREADS;
                if (h < 2560u) {
                    const unsigned int i = weight_base + h;
                    nq_reg[r][s] = __bfloat162float(__ldg(&norm_query[i]));
                    nk_reg[r][s] = __bfloat162float(__ldg(&norm_key[i]));
                }
            }
        }
        __syncthreads();

        float q_tot = __shfl_sync(0xffffffffu, warp_sum(q_parts[lane]), 0);
        float k_tot = __shfl_sync(0xffffffffu, warp_sum(k_parts[lane]), 0);
        const float inv_q = rsqrtf(q_tot / 2560.0f + eps);
        const float inv_k = rsqrtf(k_tot / 2560.0f + eps);

        #pragma unroll
        for (unsigned int r = 0; r < 4u; ++r) {
            float dot = 0.0f;
            const unsigned int h0 = threadIdx.x + r * QWEN4_PLE_GATE_THREADS;
            #pragma unroll
            for (unsigned int s = 0; s < 3u; ++s) {
                const unsigned int h = h0 + s * QWEN4_PLE_PREPARE_THREADS;
                if (h < 2560u) {
                    const float q = q_reg[r][s] * inv_q * (1.0f + nq_reg[r][s]);
                    const float k = k_reg[r][s] * inv_k * (1.0f + nk_reg[r][s]);
                    dot += q * k;
                }
            }
            dotv[r] = warp_sum(dot);
        }

        const unsigned long long value_base = (unsigned long long)token * 2560ull;
        float val_reg[4][3];
        if (lane == 0) {
            #pragma unroll
            for (unsigned int r = 0; r < 4u; ++r) d_parts[warp + 8u * r] = dotv[r];
        }
        #pragma unroll
        for (unsigned int r = 0; r < 4u; ++r) {
            const unsigned int h0 = threadIdx.x + r * QWEN4_PLE_GATE_THREADS;
            #pragma unroll
            for (unsigned int s = 0; s < 3u; ++s) {
                const unsigned int h = h0 + s * QWEN4_PLE_PREPARE_THREADS;
                if (h < 2560u) {
                    val_reg[r][s] = __bfloat162float(__ldg(&value[value_base + h]));
                }
            }
        }
        __syncthreads();

        float sum_dot = __shfl_sync(0xffffffffu, warp_sum(d_parts[lane]), 0);
        sum_dot /= sqrtf(2560.0f);
        const float root = copysignf(sqrtf(fmaxf(fabsf(sum_dot), 1.0e-6f)), sum_dot);
        const float g_val = 1.0f / (1.0f + expf(-root));

        float gv2v[4];
        #pragma unroll
        for (unsigned int r = 0; r < 4u; ++r) {
            float gv2 = 0.0f;
            const unsigned int h0 = threadIdx.x + r * QWEN4_PLE_GATE_THREADS;
            #pragma unroll
            for (unsigned int s = 0; s < 3u; ++s) {
                const unsigned int h = h0 + s * QWEN4_PLE_PREPARE_THREADS;
                if (h < 2560u) {
                    const float gv = g_val * val_reg[r][s];
                    gv2 += gv * gv;
                }
            }
            gv2v[r] = warp_sum(gv2);
        }
        if (lane == 0) {
            #pragma unroll
            for (unsigned int r = 0; r < 4u; ++r) q_parts[warp + 8u * r] = gv2v[r];
        }
        __syncthreads();
        if (warp == 0) {
            float sum = q_parts[lane];
            sum = warp_sum(sum);
            if (lane == 0) {
                const unsigned long long out = ((unsigned long long)token * 4ull + stream_id) * 2ull;
                *reinterpret_cast<float2*>(&gate[out]) = make_float2(g_val, rsqrtf(sum / 2560.0f + eps));
            }
        }
        return;
    }

    #pragma unroll
    for (unsigned int r = 0; r < 4u; ++r) {
        float q2 = 0.0f, k2 = 0.0f;
        for (unsigned int h = threadIdx.x + r * QWEN4_PLE_GATE_THREADS; h < hidden_size;
             h += QWEN4_PLE_PREPARE_THREADS) {
            const float q = __bfloat162float(hyper[base + h]);
            const float k = __bfloat162float(key[base + h]);
            q2 += q * q;
            k2 += k * k;
        }
        q2v[r] = warp_sum(q2);
        k2v[r] = warp_sum(k2);
    }
    if (lane == 0) {
        #pragma unroll
        for (unsigned int r = 0; r < 4u; ++r) {
            q_parts[warp + 8u * r] = q2v[r];
            k_parts[warp + 8u * r] = k2v[r];
        }
    }
    __syncthreads();
    if (warp == 0) {
        float q = q_parts[lane];
        float k = k_parts[lane];
        q = warp_sum(q);
        k = warp_sum(k);
        if (lane == 0) {
            shared[0] = rsqrtf(q / hidden_size + eps);
            shared[1] = rsqrtf(k / hidden_size + eps);
        }
    }
    __syncthreads();

    #pragma unroll
    for (unsigned int r = 0; r < 4u; ++r) {
        float dot = 0.0f;
        for (unsigned int h = threadIdx.x + r * QWEN4_PLE_GATE_THREADS; h < hidden_size;
             h += QWEN4_PLE_PREPARE_THREADS) {
            const unsigned int i = weight_base + h;
            const float q = __bfloat162float(hyper[base + h]) * shared[0]
                          * (1.0f + __bfloat162float(norm_query[i]));
            const float k = __bfloat162float(key[base + h]) * shared[1]
                          * (1.0f + __bfloat162float(norm_key[i]));
            dot += q * k;
        }
        dotv[r] = warp_sum(dot);
    }
    if (lane == 0) {
        #pragma unroll
        for (unsigned int r = 0; r < 4u; ++r) q_parts[warp + 8u * r] = dotv[r];
    }
    __syncthreads();
    if (warp == 0) {
        float sum = q_parts[lane];
        sum = warp_sum(sum);
        if (lane == 0) {
            sum /= sqrtf((float)hidden_size);
            const float root = copysignf(sqrtf(fmaxf(fabsf(sum), 1.0e-6f)), sum);
            shared[2] = 1.0f / (1.0f + expf(-root));
        }
    }
    __syncthreads();

    const unsigned long long value_base = (unsigned long long)token * hidden_size;
    float gv2v[4];
    #pragma unroll
    for (unsigned int r = 0; r < 4u; ++r) {
        float gv2 = 0.0f;
        for (unsigned int h = threadIdx.x + r * QWEN4_PLE_GATE_THREADS; h < hidden_size;
             h += QWEN4_PLE_PREPARE_THREADS) {
            const float gv = shared[2] * __bfloat162float(__ldg(&value[value_base + h]));
            gv2 += gv * gv;
        }
        gv2v[r] = warp_sum(gv2);
    }
    if (lane == 0) {
        #pragma unroll
        for (unsigned int r = 0; r < 4u; ++r) q_parts[warp + 8u * r] = gv2v[r];
    }
    __syncthreads();
    if (warp == 0) {
        float sum = q_parts[lane];
        sum = warp_sum(sum);
        if (lane == 0) {
            const unsigned long long out = ((unsigned long long)token * 4ull + stream_id) * 2ull;
            *reinterpret_cast<float2*>(&gate[out]) = make_float2(shared[2], rsqrtf(sum / hidden_size + eps));
        }
    }
}

// prepare's conv_input of chunk token `token`, channel (stream_id, h):
// gated value * r * (1 + norm_conv), the expression prepare stores.
__device__ __forceinline__ float qwen4_ple_fused_conv_input(
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    unsigned long long token,
    unsigned int stream_id,
    unsigned int h,
    unsigned int hidden_size,
    float conv_scale) {
    const float2 g_pair = __ldg(reinterpret_cast<const float2*>(
        gate + (token * 4ull + stream_id) * 2ull));
    const float gv = g_pair.x * __bfloat162float(__ldg(&value[token * hidden_size + h]));
    return gv * g_pair.y * conv_scale;
}

// Causal conv + residual injection of chunk tokens [token_start,
// token_start + num_tokens). One thread per residual channel and run of
// QWEN4_PLE_INJECT_RUN tokens: grid = (ceil(4H / 256), ceil(num_tokens /
// RUN)), block 256. The BF16-rounded conv inputs of the nine tokens before the
// run start a register ring (chunk token >= 0: recomputed from value + gate;
// else the slot's entry state, zero under reset), then each token's conv
// reads taps t-9, t-6, t-3 from the ring exactly as
// qwen4_ple_conv_inject_prefill reads them from conv_input/entry_state.
extern "C" __global__ void __launch_bounds__(256) qwen4_ple_conv_inject_fused_prefill(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const __nv_bfloat16* __restrict__ norm_conv,
    const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* __restrict__ entry_state,
    unsigned int token_start,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int reset_state) {
    const unsigned int residual_width = hidden_size * 4u;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= residual_width) return;
    const unsigned int run = blockIdx.y * QWEN4_PLE_INJECT_RUN;
    if (run >= num_tokens) return;
    const unsigned int first = token_start + run;
    const unsigned int last = token_start + min(run + QWEN4_PLE_INJECT_RUN, num_tokens);
    const unsigned int stream_id = i / hidden_size;
    const unsigned int h = i - stream_id * hidden_size;
    const float conv_scale = 1.0f + __bfloat162float(__ldg(&norm_conv[i]));
    const __nv_bfloat16* weight = conv_weight + i * 4u;
    float w0, w1, w2, w3;
    if (((unsigned long long)weight & 7ull) == 0ull) {
        const uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(weight));
        const __nv_bfloat162 b01 = *reinterpret_cast<const __nv_bfloat162*>(&w_u2.x);
        const __nv_bfloat162 b23 = *reinterpret_cast<const __nv_bfloat162*>(&w_u2.y);
        const float2 f01 = __bfloat1622float2(b01);
        const float2 f23 = __bfloat1622float2(b23);
        w0 = f01.x; w1 = f01.y; w2 = f23.x; w3 = f23.y;
    } else {
        w0 = __bfloat162float(__ldg(&weight[0]));
        w1 = __bfloat162float(__ldg(&weight[1]));
        w2 = __bfloat162float(__ldg(&weight[2]));
        w3 = __bfloat162float(__ldg(&weight[3]));
    }
    const __nv_bfloat16* state = entry_state + i * 9u;
    float hist[9];
    #pragma unroll
    for (unsigned int j = 0; j < 9u; ++j) {
        const long long t = (long long)first - 9ll + (long long)j;
        hist[j] = t >= 0
            ? qwen4_ple_history_bf16(qwen4_ple_fused_conv_input(
                  value, gate, (unsigned long long)t, stream_id, h, hidden_size, conv_scale))
            : (reset_state ? 0.0f : __bfloat162float(__ldg(&state[t + 9ll])));
    }
    #pragma unroll 4
    for (unsigned int t = first; t < last; ++t) {
        const float2 g_pair = __ldg(reinterpret_cast<const float2*>(
            gate + ((unsigned long long)t * 4ull + stream_id) * 2ull));
        const float gv = g_pair.x
                       * __bfloat162float(__ldg(&value[(unsigned long long)t * hidden_size + h]));
        const float x = gv * g_pair.y * conv_scale;
        float c = hist[0] * w0
                + hist[3] * w1
                + hist[6] * w2
                + x  * w3;
        c = c / (1.0f + expf(-c));
        const unsigned long long out = (unsigned long long)t * residual_width + i;
        hyper[out] = __float2bfloat16(__bfloat162float(__ldg(&hyper[out])) + gv + c);
        #pragma unroll
        for (unsigned int j = 0; j < 8u; ++j) hist[j] = hist[j + 1u];
        hist[8] = qwen4_ple_history_bf16(x);
    }
}

// qwen4_ple_commit_prefill_state with conv_input recomputed: after every
// inject launch of the chunk, commit the last nine BF16 conv inputs (chunks
// shorter than nine tokens keep the tail of the old history, or zeros under
// reset), one thread per residual channel.
extern "C" __global__ void qwen4_ple_commit_fused_prefill_state(
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const __nv_bfloat16* __restrict__ norm_conv,
    __nv_bfloat16* __restrict__ state,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int reset_state) {
    const unsigned int residual_width = hidden_size * 4u;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= residual_width || num_tokens == 0u) return;
    const unsigned int stream_id = i / hidden_size;
    const unsigned int h = i - stream_id * hidden_size;
    const float conv_scale = 1.0f + __bfloat162float(__ldg(&norm_conv[i]));
    __nv_bfloat16 prior[9];
    #pragma unroll
    for (unsigned int j = 0; j < 9u; ++j) prior[j] = __ldg(&state[i * 9u + j]);
    #pragma unroll
    for (unsigned int j = 0; j < 9u; ++j) {
        const long long source_token = (long long)num_tokens - 9ll + (long long)j;
        __nv_bfloat16 next;
        if (source_token >= 0) {
            next = __float2bfloat16(qwen4_ple_fused_conv_input(
                value, gate, (unsigned long long)source_token, stream_id, h, hidden_size,
                conv_scale));
        } else if (reset_state) {
            next = __float2bfloat16(0.0f);
        } else {
            next = prior[num_tokens + j];
        }
        state[i * 9u + j] = next;
    }
}

#include "qwen4_sigmoid_multi_seq.cuh"

// Fused batched embedding gather + Qwen4 hyperconnection stream expansion:
// reads embed_table[token_ids[token], 0..hidden_size] once and replicates it
// across all hc_count residual streams in output[token, 0..hc_count*hidden_size],
// bypassing the [num_tokens, hidden_size] norm_output staging buffer.
extern "C" __global__ void __launch_bounds__(256) qwen4_batched_embed_expand(
    const unsigned int* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embed_table,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    unsigned int hc_count) {
    const unsigned int token = blockIdx.x;
    const unsigned int token_id = __ldg(&token_ids[token]);
    const __nv_bfloat16* __restrict__ src_row =
        embed_table + (unsigned long long)token_id * hidden_size;
    __nv_bfloat16* __restrict__ dst_row =
        output + (unsigned long long)token * hidden_size * hc_count;
    if (hc_count == 4u && (hidden_size & 7u) == 0u
        && (((unsigned long long)src_row | (unsigned long long)dst_row) & 15ull) == 0ull) {
        const unsigned int vec_len = hidden_size >> 3;
        const uint4* __restrict__ src4 = reinterpret_cast<const uint4*>(src_row);
        uint4* __restrict__ dst0 = reinterpret_cast<uint4*>(dst_row);
        uint4* __restrict__ dst1 = dst0 + vec_len;
        uint4* __restrict__ dst2 = dst1 + vec_len;
        uint4* __restrict__ dst3 = dst2 + vec_len;
        for (unsigned int v = threadIdx.x; v < vec_len; v += blockDim.x) {
            const uint4 val = __ldg(&src4[v]);
            dst0[v] = val;
            dst1[v] = val;
            dst2[v] = val;
            dst3[v] = val;
        }
        return;
    }
    for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const __nv_bfloat16 val = __ldg(&src_row[h]);
        for (unsigned int s = 0; s < hc_count; ++s) {
            dst_row[s * hidden_size + h] = val;
        }
    }
}

// Arithmetic mean of the four Qwen4 hyperconnection streams per row:
//   out_h[r, c] = 0.25 * (hyper_4h[r, 0, c] + ... + hyper_4h[r, 3, c])
// Vectorized bf162 fast path (even h, 4-byte aligned); CUDA-graph-safe, 0 B spill.
extern "C" __global__ void __launch_bounds__(256) qwen4_hc_mean_rows(
    const __nv_bfloat16* __restrict__ hyper_4h,
    __nv_bfloat16* __restrict__ out_h,
    int rows,
    int h) {
    const int row = (int)blockIdx.y;
    if (row >= rows || h <= 0) return;
    const long long base_in = (long long)row * 4ll * (long long)h;
    const long long base_out = (long long)row * (long long)h;
    const __nv_bfloat16* __restrict__ s0 = hyper_4h + base_in;
    const __nv_bfloat16* __restrict__ s1 = s0 + h;
    const __nv_bfloat16* __restrict__ s2 = s1 + h;
    const __nv_bfloat16* __restrict__ s3 = s2 + h;
    __nv_bfloat16* __restrict__ dst = out_h + base_out;

    if ((h & 1) == 0
        && ((((unsigned long long)s0 | (unsigned long long)dst) & 3ull) == 0ull)) {
        const int h2 = h >> 1;
        const int idx2 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (idx2 >= h2) return;
        const __nv_bfloat162 v0 = reinterpret_cast<const __nv_bfloat162*>(s0)[idx2];
        const __nv_bfloat162 v1 = reinterpret_cast<const __nv_bfloat162*>(s1)[idx2];
        const __nv_bfloat162 v2 = reinterpret_cast<const __nv_bfloat162*>(s2)[idx2];
        const __nv_bfloat162 v3 = reinterpret_cast<const __nv_bfloat162*>(s3)[idx2];
        const float2 f0 = __bfloat1622float2(v0);
        const float2 f1 = __bfloat1622float2(v1);
        const float2 f2 = __bfloat1622float2(v2);
        const float2 f3 = __bfloat1622float2(v3);
        const float m_x = 0.25f * ((f0.x + f1.x) + (f2.x + f3.x));
        const float m_y = 0.25f * ((f0.y + f1.y) + (f2.y + f3.y));
        reinterpret_cast<__nv_bfloat162*>(dst)[idx2] = __floats2bfloat162_rn(m_x, m_y);
        return;
    }

    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx < h) {
        const float sum0 = (__bfloat162float(s0[idx]) + __bfloat162float(s1[idx]))
                         + (__bfloat162float(s2[idx]) + __bfloat162float(s3[idx]));
        dst[idx] = __float2bfloat16(0.25f * sum0);
    }
    if (idx + 1 < h) {
        const float sum1 = (__bfloat162float(s0[idx + 1]) + __bfloat162float(s1[idx + 1]))
                         + (__bfloat162float(s2[idx + 1]) + __bfloat162float(s3[idx + 1]));
        dst[idx + 1] = __float2bfloat16(0.25f * sum1);
    }
}
