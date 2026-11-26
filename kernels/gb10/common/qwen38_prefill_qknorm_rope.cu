// SPDX-License-Identifier: AGPL-3.0-only

// Exact Qwen3.8 C=1 prefill fusion:
//   deinterleave Q/Gate + per-head Q RMSNorm
//   per-head K RMSNorm
//   interleaved multi-modal RoPE for Q and K
//
// Grid: (num_tokens, 1, 1), block: (256, 1, 1)
// Precondition: head_dim % 32 == 0 (block-uniform early exit otherwise).
// Dynamic shared: num_q_heads * head_dim * 2 * sizeof(BF16) = 24,576 B
// for Q24/HD256. After Q is fully consumed, its storage is reused for the
// K heads plus eight FP32 block-reduction partials.

#include <cuda_bf16.h>
#include <stdint.h>

static constexpr unsigned int Q38_ROPE_CACHE_MAX_PAIRS = 64u;

__device__ __forceinline__ float q38_q_warp_sum(float value) {
    value = __shfl_xor_sync(0xFFFFFFFF, value, 16) + value;
    value = __shfl_xor_sync(0xFFFFFFFF, value, 8) + value;
    value = __shfl_xor_sync(0xFFFFFFFF, value, 4) + value;
    value = __shfl_xor_sync(0xFFFFFFFF, value, 2) + value;
    value = __shfl_xor_sync(0xFFFFFFFF, value, 1) + value;
    return value;
}

__device__ __forceinline__ float q38_k_warp_sum(float value) {
    value += __shfl_xor_sync(0xFFFFFFFF, value, 16);
    value += __shfl_xor_sync(0xFFFFFFFF, value, 8);
    value += __shfl_xor_sync(0xFFFFFFFF, value, 4);
    value += __shfl_xor_sync(0xFFFFFFFF, value, 2);
    value += __shfl_xor_sync(0xFFFFFFFF, value, 1);
    return value;
}

__device__ __forceinline__ void q38_qknorm_unpack_bf16x2(
    unsigned int packed, float& v0, float& v1
) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int q38_qknorm_pack_bf16x2(
    float v0, float v1
) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

__device__ __forceinline__ unsigned int q38_mrope_position(
    unsigned int pair_idx,
    unsigned int token,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w
) {
    const unsigned int section = pair_idx % 3;
    if (section == 0) return pos_t[token];
    if (section == 1) return pos_h[token];
    return pos_w[token];
}

__device__ __forceinline__ float q38_mrope_inv_freq(
    unsigned int pair_idx,
    unsigned int rotary_dim,
    double log2_theta
) {
    const double freq_exp_d = (double)(2 * pair_idx) / (double)rotary_dim;
    return (float)exp2(-freq_exp_d * log2_theta);
}

__device__ __forceinline__ void q38_mrope_rotate_pair(
    const __nv_bfloat16* __restrict__ source,
    __nv_bfloat16* __restrict__ output,
    unsigned int pair_idx,
    unsigned int rotary_dim,
    unsigned int abs_pos,
    double log2_theta
) {
    const float freq = q38_mrope_inv_freq(pair_idx, rotary_dim, log2_theta);
    const float angle = (float)abs_pos * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + half_rot;
    const float x0 = (float)source[d0];
    const float x1 = (float)source[d1];
    const float y0 = x0 * cos_val - x1 * sin_val;
    const float y1 = x1 * cos_val + x0 * sin_val;
    output[d0] = __float2bfloat16(y0);
    output[d1] = __float2bfloat16(y1);
}

__device__ __forceinline__ void q38_mrope_rotate_bf16x2_cached(
    const unsigned int* __restrict__ src_u32,
    unsigned int* __restrict__ dst_u32,
    unsigned int chunk,
    unsigned int pair_chunks,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin
) {
    const unsigned int p0 = chunk * 2u;
    const unsigned int p1 = p0 + 1u;
    float x0_a, x0_b, x1_a, x1_b;
    q38_qknorm_unpack_bf16x2(src_u32[chunk], x0_a, x0_b);
    q38_qknorm_unpack_bf16x2(src_u32[chunk + pair_chunks], x1_a, x1_b);
    const float cos_a = rope_cos[p0];
    const float sin_a = rope_sin[p0];
    const float cos_b = rope_cos[p1];
    const float sin_b = rope_sin[p1];
    const float y0_a = x0_a * cos_a - x1_a * sin_a;
    const float y1_a = x1_a * cos_a + x0_a * sin_a;
    const float y0_b = x0_b * cos_b - x1_b * sin_b;
    const float y1_b = x1_b * cos_b + x0_b * sin_b;
    dst_u32[chunk] = q38_qknorm_pack_bf16x2(y0_a, y0_b);
    dst_u32[chunk + pair_chunks] = q38_qknorm_pack_bf16x2(y1_a, y1_b);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen38_prefill_qknorm_rope(
    __nv_bfloat16* __restrict__ qg_data,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_data,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    const unsigned int num_tokens,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int qg_stride,
    const unsigned int rotary_dim,
    const float eps,
    const float theta
) {
    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (token >= num_tokens || (head_dim & 31u) != 0u) return;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ float rope_cos[Q38_ROPE_CACHE_MAX_PAIRS];
    __shared__ float rope_sin[Q38_ROPE_CACHE_MAX_PAIRS];

    __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    const unsigned int q_total = num_q_heads * head_dim;
    const unsigned int qg_total = q_total * 2;
    const unsigned int group_dim = head_dim * 2;
    const unsigned int pairs = rotary_dim / 2;
    __nv_bfloat16* token_qg = qg_data + (unsigned long long)token * qg_stride;
    __nv_bfloat16* token_q = q_out + (unsigned long long)token * q_total;

    const bool qg_vec16 = (((uintptr_t)token_qg & 15u) == 0u)
        && ((head_dim & 7u) == 0u)
        && ((q_total & 7u) == 0u);
    const bool q_vec16 = (((uintptr_t)token_q & 15u) == 0u)
        && ((head_dim & 7u) == 0u)
        && ((rotary_dim & 7u) == 0u);
    const bool use_rope_cache = (pairs <= Q38_ROPE_CACHE_MAX_PAIRS) && ((pairs & 1u) == 0u);
    const double log2_theta = log2((double)theta);

    // Identical load and Gate destination to deinterleave_qg_split_qnorm.
    uint4* smem_v4 = reinterpret_cast<uint4*>(smem);
    if (qg_vec16) {
        const uint4* token_qg_v4 = reinterpret_cast<const uint4*>(token_qg);
        const unsigned int qg_v4 = qg_total >> 3;
        for (unsigned int i = tid; i < qg_v4; i += blockDim.x) {
            smem_v4[i] = token_qg_v4[i];
        }
    } else {
        for (unsigned int i = tid; i < qg_total; i += blockDim.x) {
            smem[i] = token_qg[i];
        }
    }

    // Precompute the token's RoPE cos/sin table once per CTA so all
    // Q and K heads reuse identical values without repeating transcendentals.
    if (use_rope_cache) {
        for (unsigned int p = tid; p < pairs; p += blockDim.x) {
            const unsigned int abs_pos = q38_mrope_position(p, token, pos_t, pos_h, pos_w);
            const float freq = q38_mrope_inv_freq(p, rotary_dim, log2_theta);
            const float angle = (float)abs_pos * freq;
            rope_cos[p] = cosf(angle);
            rope_sin[p] = sinf(angle);
        }
    }
    __syncthreads();

    if (qg_vec16) {
        uint4* token_qg_v4 = reinterpret_cast<uint4*>(token_qg);
        const unsigned int q_v4 = q_total >> 3;
        const unsigned int head_v4 = head_dim >> 3;
        const unsigned int group_v4 = group_dim >> 3;
        for (unsigned int i = tid; i < q_v4; i += blockDim.x) {
            const unsigned int head = i / head_v4;
            const unsigned int dim_v4 = i % head_v4;
            token_qg_v4[q_v4 + i] = smem_v4[head * group_v4 + head_v4 + dim_v4];
        }
    } else {
        for (unsigned int i = tid; i < q_total; i += blockDim.x) {
            const unsigned int head = i / head_dim;
            const unsigned int dim = i % head_dim;
            token_qg[q_total + i] = smem[head * group_dim + head_dim + dim];
        }
    }

    // Identical per-head/lane accumulation and shuffle order to the parent Q
    // fusion. Store the normalized BF16 boundary back to shared for RoPE.
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = (blockDim.x + 31u) / 32u;
    const unsigned int elems_per_thread = head_dim / 32;
    if (elems_per_thread <= 8u) {
        float w_cache[8];
#pragma unroll
        for (unsigned int e = 0; e < 8u; e++) {
            if (e < elems_per_thread) {
                w_cache[e] = __bfloat162float(q_norm_weight[lane + e * 32u]);
            }
        }
        for (unsigned int head = warp_id; head < num_q_heads; head += num_warps) {
            float x_cache[8];
            float sum_sq = 0.0f;
#pragma unroll
            for (unsigned int e = 0; e < 8u; e++) {
                if (e < elems_per_thread) {
                    const unsigned int dim = lane + e * 32u;
                    const unsigned int source = head * group_dim + dim;
                    const float value = __bfloat162float(smem[source]);
                    x_cache[e] = value;
                    sum_sq += value * value;
                }
            }
            sum_sq = q38_q_warp_sum(sum_sq);
            const float rms = rsqrtf(sum_sq / (float)head_dim + eps);
#pragma unroll
            for (unsigned int e = 0; e < 8u; e++) {
                if (e < elems_per_thread) {
                    const unsigned int dim = lane + e * 32u;
                    const unsigned int source = head * group_dim + dim;
                    const float value = x_cache[e];
                    const float weight = w_cache[e];
                    smem[source] = __float2bfloat16(value * rms * (1.0f + weight));
                }
            }
        }
    } else {
        for (unsigned int head = warp_id; head < num_q_heads; head += num_warps) {
            float sum_sq = 0.0f;
            for (unsigned int e = 0; e < elems_per_thread; e++) {
                const unsigned int dim = lane + e * 32u;
                const unsigned int source = head * group_dim + dim;
                const float value = __bfloat162float(smem[source]);
                sum_sq += value * value;
            }
            sum_sq = q38_q_warp_sum(sum_sq);
            const float rms = rsqrtf(sum_sq / (float)head_dim + eps);
            for (unsigned int e = 0; e < elems_per_thread; e++) {
                const unsigned int dim = lane + e * 32u;
                const unsigned int source = head * group_dim + dim;
                const float value = __bfloat162float(smem[source]);
                const float weight = __bfloat162float(q_norm_weight[dim]);
                smem[source] = __float2bfloat16(value * rms * (1.0f + weight));
            }
        }
    }
    __syncthreads();

    // Preserve the normalized BF16 values outside the rotary prefix.
    if (q_vec16) {
        uint4* token_q_v4 = reinterpret_cast<uint4*>(token_q);
        const unsigned int head_v4 = head_dim >> 3;
        const unsigned int group_v4 = group_dim >> 3;
        const unsigned int rot_v4 = rotary_dim >> 3;
        const unsigned int tail_v4 = head_v4 - rot_v4;
        const unsigned int q_tail_v4 = num_q_heads * tail_v4;
        for (unsigned int i = tid; i < q_tail_v4; i += blockDim.x) {
            const unsigned int head = i / tail_v4;
            const unsigned int dim_v4 = rot_v4 + (i % tail_v4);
            token_q_v4[head * head_v4 + dim_v4] = smem_v4[head * group_v4 + dim_v4];
        }
    } else {
        const unsigned int q_tail = num_q_heads * (head_dim - rotary_dim);
        for (unsigned int i = tid; i < q_tail; i += blockDim.x) {
            const unsigned int head = i / (head_dim - rotary_dim);
            const unsigned int dim = rotary_dim + i % (head_dim - rotary_dim);
            token_q[head * head_dim + dim] = smem[head * group_dim + dim];
        }
    }

    if (use_rope_cache && q_vec16) {
        const unsigned int pair_chunks = pairs >> 1;
        const unsigned int q_pair_chunks = num_q_heads * pair_chunks;
        for (unsigned int i = tid; i < q_pair_chunks; i += blockDim.x) {
            const unsigned int head = i / pair_chunks;
            const unsigned int chunk = i % pair_chunks;
            q38_mrope_rotate_bf16x2_cached(
                reinterpret_cast<const unsigned int*>(smem + head * group_dim),
                reinterpret_cast<unsigned int*>(token_q + head * head_dim),
                chunk,
                pair_chunks,
                rope_cos,
                rope_sin
            );
        }
    } else {
        const unsigned int q_pairs = num_q_heads * pairs;
        for (unsigned int i = tid; i < q_pairs; i += blockDim.x) {
            const unsigned int head = i / pairs;
            const unsigned int pair = i % pairs;
            const unsigned int abs_pos = q38_mrope_position(pair, token, pos_t, pos_h, pos_w);
            q38_mrope_rotate_pair(
                smem + head * group_dim,
                token_q + head * head_dim,
                pair,
                rotary_dim,
                abs_pos,
                log2_theta
            );
        }
    }
    __syncthreads();

    // The Q tile is dead. Reuse its storage for K heads and eight FP32
    // block-reduction partials.
    const unsigned int kv_total = num_kv_heads * head_dim;
    __nv_bfloat16* token_k_base = k_data + (unsigned long long)token * kv_total;
    const bool k_batched_vec16 = (((uintptr_t)token_k_base & 15u) == 0u)
        && ((head_dim & 7u) == 0u)
        && ((rotary_dim & 7u) == 0u)
        && use_rope_cache
        && (kv_total + 16u <= qg_total);

    if (k_batched_vec16) {
        uint4* token_k_v4 = reinterpret_cast<uint4*>(token_k_base);
        const unsigned int kv_v4 = kv_total >> 3;
        for (unsigned int i = tid; i < kv_v4; i += blockDim.x) {
            smem_v4[i] = token_k_v4[i];
        }
        __syncthreads();

        float* k_warp_sums = reinterpret_cast<float*>(smem + kv_total);
        const unsigned int half_size = head_dim / 2;
        if (half_size <= blockDim.x) {
            float w0 = 0.0f, w1 = 0.0f;
            if (tid < half_size) {
                q38_qknorm_unpack_bf16x2(
                    reinterpret_cast<const unsigned int*>(k_norm_weight)[tid], w0, w1
                );
            }
            for (unsigned int head = 0; head < num_kv_heads; head++) {
                __nv_bfloat16* head_smem = smem + head * head_dim;
                float sum_sq = 0.0f;
                float v0 = 0.0f, v1 = 0.0f;
                if (tid < half_size) {
                    q38_qknorm_unpack_bf16x2(
                        reinterpret_cast<const unsigned int*>(head_smem)[tid], v0, v1
                    );
                    sum_sq += v0 * v0 + v1 * v1;
                }
                sum_sq = q38_k_warp_sum(sum_sq);
                if (lane == 0) k_warp_sums[warp_id] = sum_sq;
                __syncthreads();
                if (warp_id == 0) {
                    float value = lane < num_warps ? k_warp_sums[lane] : 0.0f;
                    value = q38_k_warp_sum(value);
                    if (lane == 0) k_warp_sums[0] = value;
                }
                __syncthreads();
                const float rms = rsqrtf(k_warp_sums[0] / (float)head_dim + eps);
                if (tid < half_size) {
                    reinterpret_cast<unsigned int*>(head_smem)[tid] = q38_qknorm_pack_bf16x2(
                        v0 * rms * (1.0f + w0), v1 * rms * (1.0f + w1)
                    );
                }
                __syncthreads();
            }
        } else {
            for (unsigned int head = 0; head < num_kv_heads; head++) {
                __nv_bfloat16* head_smem = smem + head * head_dim;
                float sum_sq = 0.0f;
                for (unsigned int i = tid; i < half_size; i += blockDim.x) {
                    float v0, v1;
                    q38_qknorm_unpack_bf16x2(
                        reinterpret_cast<const unsigned int*>(head_smem)[i], v0, v1
                    );
                    sum_sq += v0 * v0 + v1 * v1;
                }
                sum_sq = q38_k_warp_sum(sum_sq);
                if (lane == 0) k_warp_sums[warp_id] = sum_sq;
                __syncthreads();
                if (warp_id == 0) {
                    float value = lane < num_warps ? k_warp_sums[lane] : 0.0f;
                    value = q38_k_warp_sum(value);
                    if (lane == 0) k_warp_sums[0] = value;
                }
                __syncthreads();
                const float rms = rsqrtf(k_warp_sums[0] / (float)head_dim + eps);
                for (unsigned int i = tid; i < half_size; i += blockDim.x) {
                    float v0, v1, w0, w1;
                    q38_qknorm_unpack_bf16x2(
                        reinterpret_cast<const unsigned int*>(head_smem)[i], v0, v1
                    );
                    q38_qknorm_unpack_bf16x2(
                        reinterpret_cast<const unsigned int*>(k_norm_weight)[i], w0, w1
                    );
                    reinterpret_cast<unsigned int*>(head_smem)[i] = q38_qknorm_pack_bf16x2(
                        v0 * rms * (1.0f + w0), v1 * rms * (1.0f + w1)
                    );
                }
                __syncthreads();
            }
        }

        const unsigned int head_v4 = head_dim >> 3;
        const unsigned int rot_v4 = rotary_dim >> 3;
        const unsigned int tail_v4 = head_v4 - rot_v4;
        const unsigned int k_tail_v4 = num_kv_heads * tail_v4;
        for (unsigned int i = tid; i < k_tail_v4; i += blockDim.x) {
            const unsigned int head = i / tail_v4;
            const unsigned int dim_v4 = rot_v4 + (i % tail_v4);
            token_k_v4[head * head_v4 + dim_v4] = smem_v4[head * head_v4 + dim_v4];
        }
        const unsigned int pair_chunks = pairs >> 1;
        const unsigned int k_pair_chunks = num_kv_heads * pair_chunks;
        for (unsigned int i = tid; i < k_pair_chunks; i += blockDim.x) {
            const unsigned int head = i / pair_chunks;
            const unsigned int chunk = i % pair_chunks;
            q38_mrope_rotate_bf16x2_cached(
                reinterpret_cast<const unsigned int*>(smem + head * head_dim),
                reinterpret_cast<unsigned int*>(token_k_base + head * head_dim),
                chunk,
                pair_chunks,
                rope_cos,
                rope_sin
            );
        }
    } else {
        float* k_warp_sums = reinterpret_cast<float*>(smem + head_dim);
        for (unsigned int head = 0; head < num_kv_heads; head++) {
            __nv_bfloat16* token_k = token_k_base + head * head_dim;
            for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
                smem[dim] = token_k[dim];
            }
            __syncthreads();

            float sum_sq = 0.0f;
            const unsigned int half_size = head_dim / 2;
            for (unsigned int i = tid; i < half_size; i += blockDim.x) {
                float v0, v1;
                q38_qknorm_unpack_bf16x2(
                    reinterpret_cast<const unsigned int*>(smem)[i], v0, v1
                );
                sum_sq += v0 * v0 + v1 * v1;
            }
            sum_sq = q38_k_warp_sum(sum_sq);
            if (lane == 0) k_warp_sums[warp_id] = sum_sq;
            __syncthreads();
            if (warp_id == 0) {
                float value = lane < num_warps ? k_warp_sums[lane] : 0.0f;
                value = q38_k_warp_sum(value);
                if (lane == 0) k_warp_sums[0] = value;
            }
            __syncthreads();
            const float rms = rsqrtf(k_warp_sums[0] / (float)head_dim + eps);
            for (unsigned int i = tid; i < half_size; i += blockDim.x) {
                float x0, x1, w0, w1;
                q38_qknorm_unpack_bf16x2(
                    reinterpret_cast<const unsigned int*>(smem)[i], x0, x1
                );
                q38_qknorm_unpack_bf16x2(
                    reinterpret_cast<const unsigned int*>(k_norm_weight)[i], w0, w1
                );
                reinterpret_cast<unsigned int*>(smem)[i] = q38_qknorm_pack_bf16x2(
                    x0 * rms * (1.0f + w0), x1 * rms * (1.0f + w1)
                );
            }
            __syncthreads();

            for (unsigned int dim = rotary_dim + tid; dim < head_dim; dim += blockDim.x) {
                token_k[dim] = smem[dim];
            }
            for (unsigned int p = tid; p < pairs; p += blockDim.x) {
                const unsigned int abs_pos = q38_mrope_position(p, token, pos_t, pos_h, pos_w);
                q38_mrope_rotate_pair(
                    smem, token_k, p, rotary_dim, abs_pos, log2_theta
                );
            }
            __syncthreads();
        }
    }
}
