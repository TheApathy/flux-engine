// SPDX-License-Identifier: AGPL-3.0-only
// Gemma-4 HDIM=512 Prefill — Scalar reference implementation.
// Correct but slow. Used to validate the pipeline before optimizing.
// One CTA per (q_head, BR=16 q_rows). Each 8-thread group handles one row (64 dims/thread).
// Grid: (num_q_heads, ceil(seq_len/BR), batch)  Block: (128, 1, 1)
//
// ABI: the same 13 parameters as `inferspark_prefill`, because the one host
// launcher (`ops::prefill_attention`) serves both. With the old 11-parameter
// list the driver read `query_start` (always 0) as num_q_heads, so every CTA
// returned at the q_head guard and the output was never written.

#include <cuda_bf16.h>

#define BR 16
#define HDIM 512

extern "C" __global__ void inferspark_prefill_512(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int query_start,
    const unsigned int query_len_total,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window   // Always 0 for full-attention 512-hd layers; param added for signature consistency
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;  // 0..127

    if (q_head >= num_q_heads || head_dim != HDIM) return;

    const unsigned int q_row = query_start + q_block * BR + (tid / 8);  // 128 threads / 8 = 16 rows
    if (q_row >= seq_len || q_row >= query_start + query_len_total) return;

    const unsigned int dim_lane = tid % 8;  // 8 threads per row, each handles 64 dims
    const unsigned int dim_start = dim_lane * 64;
    const unsigned int dim_end = dim_start + 64;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned int q_stride = num_q_heads * head_dim;
    const unsigned int kv_stride = num_kv_heads * head_dim;

    const __nv_bfloat16* Q_row = Q + (unsigned long long)batch * seq_len * q_stride
                                    + (unsigned long long)q_row * q_stride
                                    + (unsigned long long)q_head * head_dim;

    // Phase 1: Compute attention scores (all threads collaborate on the full row)
    // Each of the 8 dim-lanes computes a partial dot product, then reduce

    // Determine KV length for this query row
    unsigned int kv_len = seq_len;
    if (causal) kv_len = min(kv_len, q_row + 1);

    // Online softmax over KV positions
    float m = -1e30f;
    float l = 0.0f;
    float o_acc[64];  // Each thread accumulates 64 output dims
    for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; d++) {
        o_acc[d] = 0.0f;
    }

    for (unsigned int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
        const __nv_bfloat16* K_row = K + (unsigned long long)batch * seq_len * kv_stride
                                        + (unsigned long long)kv_pos * kv_stride
                                        + (unsigned long long)kv_head * head_dim;

        // Dot product Q[q_row] · K[kv_pos] — partial across dim_lane
        float dot = 0.0f;
        for (unsigned int d = dim_start; d < dim_end; d++) {
            dot += __bfloat162float(Q_row[d]) * __bfloat162float(K_row[d]);
        }

        // Reduce across 8 dim-lanes (within the 8-thread group for this row)
        // Warp shuffle within the 8-thread group. The mask names only this
        // row's 8 lanes: the 4 row groups of a warp leave at different times
        // (tail rows return at q_row >= seq_len, causal rows loop over
        // different kv_len), so a full-warp mask would name lanes that have
        // exited or are at another iteration -- undefined on sm_70+.
        const unsigned int group_mask = 0xFFu << ((tid % 32) & ~7u);
        dot += __shfl_xor_sync(group_mask, dot, 1);
        dot += __shfl_xor_sync(group_mask, dot, 2);
        dot += __shfl_xor_sync(group_mask, dot, 4);
        // Now all 8 threads have the full dot product

        float score = dot * inv_sqrt_d;

        // Online softmax update
        float m_new = fmaxf(m, score);
        float exp_old = __expf(m - m_new);
        float exp_new = __expf(score - m_new);

        // Rescale existing accumulator
        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; d++) {
            o_acc[d] *= exp_old;
        }
        l = l * exp_old + exp_new;
        m = m_new;

        // Accumulate V contribution
        const __nv_bfloat16* V_row = V + (unsigned long long)batch * seq_len * kv_stride
                                        + (unsigned long long)kv_pos * kv_stride
                                        + (unsigned long long)kv_head * head_dim;
        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; d++) {
            o_acc[d] += exp_new * __bfloat162float(V_row[dim_start + d]);
        }
    }

    // Normalize and write output
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    __nv_bfloat16* O_row = O + (unsigned long long)batch * seq_len * q_stride
                              + (unsigned long long)q_row * q_stride
                              + (unsigned long long)q_head * head_dim;
    for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; d++) {
        O_row[dim_start + d] = __float2bfloat16(o_acc[d] * inv_l);
    }
}
