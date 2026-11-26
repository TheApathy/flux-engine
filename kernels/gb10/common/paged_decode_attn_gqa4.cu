// SPDX-License-Identifier: AGPL-3.0-only
// GQA-fused twin of paged_decode_attn: one CTA per (group of HPC=4 q heads, row).
// Every per-head operation, operand order, batching (BC), warp split and
// reduction tree is identical to paged_decode_attn; only the K/V loads are
// shared across the HPC heads, so results are bit-identical. Grid: (num_q_heads/4, num_seqs).
#include <cuda_bf16.h>

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 256
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32  (HDIM / (WARP_SIZE * 2))
#define NUM_WARPS 8
#define BC 4            // KV positions batched per loop iteration

__device__ __forceinline__ void unpack2_pd(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

// Helper: compute pointer to K or V for a given position in paged cache
__device__ __forceinline__ const __nv_bfloat16* paged_kv_ptr(
    const __nv_bfloat16* __restrict__ cache,   // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_table,       // [max_blocks_per_seq]
    unsigned int pos,
    unsigned int block_size,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int kv_head
) {
    unsigned int logical_block = pos / block_size;
    unsigned int block_offset = pos % block_size;
    unsigned int physical_block = (unsigned int)block_table[logical_block];
    unsigned long long page_stride = (unsigned long long)block_size * num_kv_heads * head_dim;
    return cache + (unsigned long long)physical_block * page_stride
                 + (unsigned long long)block_offset * num_kv_heads * head_dim
                 + (unsigned long long)kv_head * head_dim;
}

#define HPC 4  // q heads per CTA (all share one kv head)
extern "C" __global__ void __launch_bounds__(256, 1) paged_decode_attn_gqa4(
    const __nv_bfloat16* __restrict__ Q,          // [num_seqs, num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ K_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ V_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ O,                // [num_seqs, num_q_heads, head_dim]
    const int* __restrict__ block_tables,         // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,             // [num_seqs]
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,              // query.stride(0) in elements
    const unsigned int sliding_window         // 0 = full attention; >0 = only attend to last `sliding_window` KV positions (Gemma-4 hybrid attn)
) {
    const unsigned int q_head0 = blockIdx.x * HPC;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head0 + HPC > num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Sliding-window start position. For Gemma-4 sliding layers with
    // window=1024, we mask out KV positions older than seq_len - 1024.
    // When sliding_window == 0 (full attention) or seq_len fits inside
    // the window, window_start = 0 (no masking).
    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head0 / gqa_ratio;
    if ((q_head0 + HPC - 1) / gqa_ratio != kv_head) return;  // all HPC heads must share the kv head
    const unsigned int vec_offset = lane_id * VEC_BF16;

    // Block table for this sequence
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q into registers (strided: Q may be a non-contiguous QKV split view)
    float q_reg[HPC][VEC_BF16];
    #pragma unroll
    for (int h = 0; h < HPC; h++) {
        const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                           + (unsigned long long)(q_head0 + h) * head_dim + vec_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            unpack2_pd(q32[i], q_reg[h][2*i], q_reg[h][2*i+1]);
        }
    }

    // Each warp handles a chunk of the KV sequence. Split across the
    // ATTENDED range [window_start, seq_len) rather than the raw [0, seq_len)
    // so warps aren't wasted on positions masked out by the sliding window.
    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    // Online softmax state
    float m[HPC], l[HPC];
    float o_reg[HPC][VEC_BF16];
    #pragma unroll
    for (int h = 0; h < HPC; h++) {
        m[h] = -1e30f; l[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) o_reg[h][i] = 0.0f;
    }

    // === Main loop: process positions with batched KV loading ===
    // We can batch BC=4 positions when they're in the same physical block.
    // At block boundaries, fall back to single-position processing.
    unsigned int pos = my_start;

    while (pos < my_end) {
        // Check how many consecutive positions share the same physical block
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

        // Get physical block pointer base
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned long long page_stride = (unsigned long long)block_size * num_kv_heads * head_dim;
        unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;
        const __nv_bfloat16* k_block_base = K_cache + (unsigned long long)physical_block * page_stride
                                                     + (unsigned long long)block_offset * head_stride_kv
                                                     + (unsigned long long)kv_head * head_dim;
        const __nv_bfloat16* v_block_base = V_cache + (unsigned long long)physical_block * page_stride
                                                     + (unsigned long long)block_offset * head_stride_kv
                                                     + (unsigned long long)kv_head * head_dim;

        // Process in batches of BC within this physical block
        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;

        // Batched path: BC=4 positions at a time (contiguous in memory)
        for (; processed < aligned_count; processed += BC) {
            // Load BC K vectors
            unsigned int k_packed[BC][VEC_U32];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* k32 = (const unsigned int*)(k_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++)
                    k_packed[b][i] = k32[i];
            }

            // Compute BC dot products for every head (same order per head)
            float scores[HPC][BC];
            #pragma unroll
            for (int h = 0; h < HPC; h++) {
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VEC_U32; i++) {
                        float k0, k1;
                        unpack2_pd(k_packed[b][i], k0, k1);
                        dot += q_reg[h][2*i] * k0 + q_reg[h][2*i+1] * k1;
                    }
                    #pragma unroll
                    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                        dot += __shfl_xor_sync(0xffffffff, dot, offset);
                    scores[h][b] = dot * inv_sqrt_d;
                }
            }

            // Prefetch V
            unsigned int v_packed[BC][VEC_U32];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* v32 = (const unsigned int*)(v_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++)
                    v_packed[b][i] = v32[i];
            }

            #pragma unroll
            for (int h = 0; h < HPC; h++) {
                // Batched softmax
                float m_new = m[h];
                #pragma unroll
                for (int b = 0; b < BC; b++)
                    m_new = fmaxf(m_new, scores[h][b]);

                float exp_old = __expf(m[h] - m_new);
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++)
                    o_reg[h][i] *= exp_old;
                l[h] *= exp_old;

                float exp_factors[BC];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    exp_factors[b] = __expf(scores[h][b] - m_new);
                    l[h] += exp_factors[b];
                }
                m[h] = m_new;

                // V accumulate
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    float ef = exp_factors[b];
                    #pragma unroll
                    for (int i = 0; i < VEC_U32; i++) {
                        float v0, v1;
                        unpack2_pd(v_packed[b][i], v0, v1);
                        o_reg[h][2*i]   += ef * v0;
                        o_reg[h][2*i+1] += ef * v1;
                    }
                }
            }
        }

        // Remainder: single positions
        for (; processed < batch_count; processed++) {
            const unsigned int* k32 = (const unsigned int*)(k_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset);
            unsigned int k1p[VEC_U32], v1p[VEC_U32];
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) k1p[i] = k32[i];
            const unsigned int* v32 = (const unsigned int*)(v_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) v1p[i] = v32[i];
            #pragma unroll
            for (int h = 0; h < HPC; h++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    float k0, k1;
                    unpack2_pd(k1p[i], k0, k1);
                    dot += q_reg[h][2*i] * k0 + q_reg[h][2*i+1] * k1;
                }
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);

                float score = dot * inv_sqrt_d;
                float m_new = fmaxf(m[h], score);
                float exp_old = __expf(m[h] - m_new);
                float exp_new = __expf(score - m_new);
                l[h] = l[h] * exp_old + exp_new;

                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    float v0, v1;
                    unpack2_pd(v1p[i], v0, v1);
                    o_reg[h][2*i]   = o_reg[h][2*i]   * exp_old + exp_new * v0;
                    o_reg[h][2*i+1] = o_reg[h][2*i+1] * exp_old + exp_new * v1;
                }
                m[h] = m_new;
            }
        }

        pos += batch_count;
    }

    // === Tree-based inter-warp reduction (per head, same tree as the base kernel) ===
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    #pragma unroll 1
    for (int h = 0; h < HPC; h++) {
        if (h > 0) __syncthreads();
        if (lane_id == 0) {
            smem_m[warp_id] = m[h];
            smem_l[warp_id] = l[h];
        }
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) {
            smem_o[warp_id][vec_offset + i] = o_reg[h][i];
        }
        __syncthreads();

        #pragma unroll
        for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
            if (warp_id < (unsigned int)stride) {
                unsigned int other = warp_id + stride;
                float lw = smem_l[other];
                if (lw > 0.0f) {
                    float mw = smem_m[other];
                    float my_m = smem_m[warp_id];
                    float my_l = smem_l[warp_id];
                    float m_new = fmaxf(my_m, mw);
                    float scale_me = __expf(my_m - m_new);
                    float scale_w = __expf(mw - m_new);
                    float new_l = my_l * scale_me + lw * scale_w;
                    __syncwarp();
                    if (lane_id == 0) {
                        smem_l[warp_id] = new_l;
                        smem_m[warp_id] = m_new;
                    }
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++) {
                        smem_o[warp_id][vec_offset + i] =
                            smem_o[warp_id][vec_offset + i] * scale_me +
                            smem_o[other][vec_offset + i] * scale_w;
                    }
                }
            }
            __syncthreads();
        }

        if (warp_id == 0) {
            float final_l = smem_l[0];
            float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
            unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                                  + (unsigned long long)(q_head0 + h) * head_dim + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float v0 = smem_o[0][vec_offset + 2*i]     * inv_l;
                float v1 = smem_o[0][vec_offset + 2*i + 1] * inv_l;
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
                o32[i] = lo | (hi << 16);
            }
        }
    }
}

#if HDIM == 256
// ============================================================================
// GQA-Grouped (up to 12 Q heads / 1 KV head, SK=32) + Warp-0 Ragged Partitioner
// (BF16 KV cache, C=1..64).
// ============================================================================

#define GQA12_MAX_HPC 12
#define GQA12_H_SUB 3
#define GQA12_SK 32
#define GQA12_MAX_SEQS 256
#define GQA12_MAX_PARTS 128

__device__ __forceinline__ unsigned int gqa12_bf16_warp_scan_inc_u32(unsigned int val, unsigned int lane) {
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        unsigned int y = __shfl_up_sync(0xffffffffu, val, o);
        if (lane >= (unsigned int)o) val += y;
    }
    return val;
}

extern "C" __global__ void __launch_bounds__(256, 2) paged_decode_attn_gqa12_ragged_bf16(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ workspace,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_seqs,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned int sliding_window
) {
    if (head_dim != HDIM || num_kv_heads == 0u || num_seqs == 0u || num_seqs > GQA12_MAX_SEQS) return;

    const unsigned int part_idx = blockIdx.x;
    const unsigned int group_idx = blockIdx.y;
    const unsigned int G = gridDim.x;
    if (G == 0u || G > GQA12_MAX_PARTS) return;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    if (gqa_ratio == 0u || gqa_ratio * num_kv_heads != num_q_heads) return;
    const unsigned int groups_per_kv = (gqa_ratio + GQA12_MAX_HPC - 1u) / GQA12_MAX_HPC;
    const unsigned int total_groups = num_kv_heads * groups_per_kv;
    if (group_idx >= total_groups) return;

    const unsigned int kv_head = group_idx / groups_per_kv;
    const unsigned int sub_group = group_idx % groups_per_kv;
    const unsigned int q_head0 = kv_head * gqa_ratio + sub_group * GQA12_MAX_HPC;
    const unsigned int kv_q_end = (kv_head + 1u) * gqa_ratio;
    const unsigned int h_count = (kv_q_end - q_head0 < GQA12_MAX_HPC)
        ? (kv_q_end - q_head0)
        : (unsigned int)GQA12_MAX_HPC;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;
    const unsigned int head_grp = warp_id >> 1; // 0..3
    const unsigned int key_sub = warp_id & 1u;  // 0..1
    const unsigned int vec_off = lane_id * VEC_BF16;

    __shared__ __align__(16) union {
        struct {
            __nv_bfloat16 k[GQA12_SK][HDIM];
            __nv_bfloat16 v[GQA12_SK][HDIM];
        } kv;
        float red_o[GQA12_MAX_HPC][HDIM];
    } smem_stage;
    __shared__ float s_red_m[GQA12_MAX_HPC];
    __shared__ float s_red_l[GQA12_MAX_HPC];
    __shared__ unsigned int s_seg_b[GQA12_MAX_SEQS];
    __shared__ unsigned int s_seg_t0[GQA12_MAX_SEQS];
    __shared__ unsigned int s_seg_t1[GQA12_MAX_SEQS];
    __shared__ unsigned int s_seg_win[GQA12_MAX_SEQS];
    __shared__ unsigned int s_seg_len[GQA12_MAX_SEQS];
    __shared__ unsigned char s_seg_split[GQA12_MAX_SEQS];
    __shared__ unsigned char s_seg_subslot[GQA12_MAX_SEQS];
    __shared__ unsigned int s_nseg;

    // Warp-0 prefix-sum ragged tile partitioner across up to 256 sequences.
    // Pass 1 computes total_tiles (seq_lens[0..num_seqs] <= 1 KiB stays hot in L1);
    // Pass 2 replays the L1-hot scan and compacts overlapping segments into s_seg_*.
    if (warp_id == 0u) {
        unsigned int total_tiles = 0u;
        for (unsigned int base = 0u; base < num_seqs; base += 32u) {
            const unsigned int b = base + lane_id;
            const int raw = (b < num_seqs) ? seq_lens[b] : 0;
            const unsigned int len = (raw > 0) ? (unsigned int)raw : 0u;
            const unsigned int win = (sliding_window > 0u && len > sliding_window) ? (len - sliding_window) : 0u;
            const unsigned int tiles = (len - win + GQA12_SK - 1u) / GQA12_SK;
            const unsigned int inc = gqa12_bf16_warp_scan_inc_u32(tiles, lane_id);
            total_tiles += __shfl_sync(0xffffffffu, inc, 31);
        }

        const unsigned int t_lo = (unsigned int)(((unsigned long long)total_tiles * part_idx) / G);
        const unsigned int t_hi = (unsigned int)(((unsigned long long)total_tiles * (part_idx + 1u)) / G);

        unsigned int running_tot = 0u;
        unsigned int cnt_acc = 0u;
        for (unsigned int base = 0u; base < num_seqs; base += 32u) {
            const unsigned int b = base + lane_id;
            const int raw = (b < num_seqs) ? seq_lens[b] : 0;
            const unsigned int len = (raw > 0) ? (unsigned int)raw : 0u;
            const unsigned int win = (sliding_window > 0u && len > sliding_window) ? (len - sliding_window) : 0u;
            const unsigned int tiles = (len - win + GQA12_SK - 1u) / GQA12_SK;
            const unsigned int inc = running_tot + gqa12_bf16_warp_scan_inc_u32(tiles, lane_id);
            running_tot = __shfl_sync(0xffffffffu, inc, 31);

            const unsigned int st = inc - tiles;
            const unsigned int en = inc;
            const unsigned int ov_lo = max(t_lo, st);
            const unsigned int ov_hi = min(t_hi, en);
            const bool meets = (b < num_seqs) && (tiles > 0u) && (ov_lo < ov_hi);
            const unsigned int m = __ballot_sync(0xffffffffu, meets);
            const unsigned int cnt = (unsigned int)__popc(m);
            if (meets) {
                const unsigned int idx = cnt_acc + (unsigned int)__popc(m & ((1u << lane_id) - 1u));
                s_seg_b[idx] = b;
                s_seg_t0[idx] = ov_lo - st;
                s_seg_t1[idx] = ov_hi - st;
                s_seg_win[idx] = win;
                s_seg_len[idx] = len;
                s_seg_split[idx] = (st < t_lo || en > t_hi) ? 1u : 0u;
                s_seg_subslot[idx] = (st >= t_lo) ? 1u : 0u;
            }
            cnt_acc += cnt;
        }
        if (lane_id == 0u) {
            s_nseg = (total_tiles > 0u) ? cnt_acc : 0u;
        }
    }
    __syncthreads();

    const unsigned int nseg = s_nseg;
    if (nseg == 0u) return;

    const unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * HDIM;
    const unsigned long long page_stride = (unsigned long long)block_size * head_stride_kv;

    for (unsigned int si = 0u; si < nseg; ++si) {
        const unsigned int seq_idx = s_seg_b[si];
        const unsigned int seg_t0 = s_seg_t0[si];
        const unsigned int seg_t1 = s_seg_t1[si];
        const unsigned int win_start = s_seg_win[si];
        const unsigned int seq_len = s_seg_len[si];
        const unsigned char seg_split = s_seg_split[si];
        const unsigned char seg_subslot = s_seg_subslot[si];

        const int* my_block_table = block_tables + (unsigned long long)seq_idx * max_blocks_per_seq;

        float q_reg[GQA12_H_SUB][VEC_BF16];
        float o_reg[GQA12_H_SUB][VEC_BF16];
        float m[GQA12_H_SUB];
        float l[GQA12_H_SUB];

        #pragma unroll
        for (int c = 0; c < GQA12_H_SUB; ++c) {
            const unsigned int h_local = head_grp * GQA12_H_SUB + (unsigned int)c;
            m[c] = -1e30f;
            l[c] = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; ++i) {
                o_reg[c][i] = 0.0f;
                q_reg[c][i] = 0.0f;
            }
            if (h_local < h_count) {
                const unsigned int* q32 = (const unsigned int*)(Q
                    + (unsigned long long)seq_idx * q_stride
                    + (unsigned long long)(q_head0 + h_local) * HDIM + vec_off);
                #pragma unroll
                for (int i = 0; i < VEC_U32; ++i) {
                    unpack2_pd(q32[i], q_reg[c][2*i], q_reg[c][2*i+1]);
                    q_reg[c][2*i]   *= inv_sqrt_d;
                    q_reg[c][2*i+1] *= inv_sqrt_d;
                }
            }
        }

        for (unsigned int tile = seg_t0; tile < seg_t1; ++tile) {
            const unsigned int tile_pos0 = win_start + tile * GQA12_SK;
            const unsigned int rem_keys = seq_len - tile_pos0;
            const unsigned int tile_keys = (rem_keys < GQA12_SK) ? rem_keys : (unsigned int)GQA12_SK;

            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const unsigned int k_row = warp_id * 4u + (unsigned int)r;
                if (k_row < tile_keys) {
                    const unsigned int pos = tile_pos0 + k_row;
                    const unsigned int logical_block = pos / block_size;
                    const unsigned int block_offset = pos % block_size;
                    const unsigned int physical_block = (unsigned int)__ldg(&my_block_table[logical_block]);
                    const unsigned long long base_off =
                        (unsigned long long)physical_block * page_stride
                        + (unsigned long long)block_offset * head_stride_kv
                        + (unsigned long long)kv_head * HDIM + vec_off;
                    const uint4 kv_k = __ldg((const uint4*)(K_cache + base_off));
                    const uint4 kv_v = __ldg((const uint4*)(V_cache + base_off));
                    *(uint4*)(&smem_stage.kv.k[k_row][vec_off]) = kv_k;
                    *(uint4*)(&smem_stage.kv.v[k_row][vec_off]) = kv_v;
                }
            }
            __syncthreads();

            const unsigned int sub_start = key_sub * 16u;
            const unsigned int sub_end = (sub_start + 16u < tile_keys)
                ? (sub_start + 16u)
                : ((sub_start < tile_keys) ? tile_keys : sub_start);

            unsigned int kp = sub_start;
            for (; kp + BC <= sub_end; kp += BC) {
                unsigned int k_pack[BC][VEC_U32];
                unsigned int v_pack[BC][VEC_U32];
                #pragma unroll
                for (int b = 0; b < BC; ++b) {
                    const uint4 kk = *(const uint4*)(&smem_stage.kv.k[kp + (unsigned int)b][vec_off]);
                    const uint4 vv = *(const uint4*)(&smem_stage.kv.v[kp + (unsigned int)b][vec_off]);
                    k_pack[b][0] = kk.x; k_pack[b][1] = kk.y; k_pack[b][2] = kk.z; k_pack[b][3] = kk.w;
                    v_pack[b][0] = vv.x; v_pack[b][1] = vv.y; v_pack[b][2] = vv.z; v_pack[b][3] = vv.w;
                }

                #pragma unroll
                for (int c = 0; c < GQA12_H_SUB; ++c) {
                    const unsigned int h_local = head_grp * GQA12_H_SUB + (unsigned int)c;
                    if (h_local < h_count) {
                        float scores[BC];
                        #pragma unroll
                        for (int b = 0; b < BC; ++b) {
                            float dot = 0.0f;
                            #pragma unroll
                            for (int i = 0; i < VEC_U32; ++i) {
                                float k0, k1;
                                unpack2_pd(k_pack[b][i], k0, k1);
                                dot = fmaf(q_reg[c][2*i],   k0, dot);
                                dot = fmaf(q_reg[c][2*i+1], k1, dot);
                            }
                            #pragma unroll
                            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                                dot += __shfl_xor_sync(0xffffffffu, dot, offset);
                            }
                            scores[b] = dot;
                        }

                        float m_new = m[c];
                        #pragma unroll
                        for (int b = 0; b < BC; ++b) m_new = fmaxf(m_new, scores[b]);
                        float exp_old = __expf(m[c] - m_new);
                        l[c] *= exp_old;
                        #pragma unroll
                        for (int i = 0; i < VEC_BF16; ++i) o_reg[c][i] *= exp_old;
                        m[c] = m_new;

                        #pragma unroll
                        for (int b = 0; b < BC; ++b) {
                            float ef = __expf(scores[b] - m_new);
                            l[c] += ef;
                            #pragma unroll
                            for (int i = 0; i < VEC_U32; ++i) {
                                float v0, v1;
                                unpack2_pd(v_pack[b][i], v0, v1);
                                o_reg[c][2*i]   = fmaf(ef, v0, o_reg[c][2*i]);
                                o_reg[c][2*i+1] = fmaf(ef, v1, o_reg[c][2*i+1]);
                            }
                        }
                    }
                }
            }

            for (; kp < sub_end; ++kp) {
                const uint4 kk = *(const uint4*)(&smem_stage.kv.k[kp][vec_off]);
                const uint4 vv = *(const uint4*)(&smem_stage.kv.v[kp][vec_off]);
                unsigned int k1[VEC_U32] = {kk.x, kk.y, kk.z, kk.w};
                unsigned int v1[VEC_U32] = {vv.x, vv.y, vv.z, vv.w};
                #pragma unroll
                for (int c = 0; c < GQA12_H_SUB; ++c) {
                    const unsigned int h_local = head_grp * GQA12_H_SUB + (unsigned int)c;
                    if (h_local < h_count) {
                        float dot = 0.0f;
                        #pragma unroll
                        for (int i = 0; i < VEC_U32; ++i) {
                            float k0, k1v;
                            unpack2_pd(k1[i], k0, k1v);
                            dot = fmaf(q_reg[c][2*i],   k0,  dot);
                            dot = fmaf(q_reg[c][2*i+1], k1v, dot);
                        }
                        #pragma unroll
                        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                            dot += __shfl_xor_sync(0xffffffffu, dot, offset);
                        }
                        float m_new = fmaxf(m[c], dot);
                        float exp_old = __expf(m[c] - m_new);
                        float exp_new = __expf(dot - m_new);
                        l[c] = fmaf(l[c], exp_old, exp_new);
                        m[c] = m_new;
                        #pragma unroll
                        for (int i = 0; i < VEC_U32; ++i) {
                            float v0, v1v;
                            unpack2_pd(v1[i], v0, v1v);
                            o_reg[c][2*i]   = fmaf(exp_new, v0,  o_reg[c][2*i]   * exp_old);
                            o_reg[c][2*i+1] = fmaf(exp_new, v1v, o_reg[c][2*i+1] * exp_old);
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (key_sub == 1u) {
            #pragma unroll
            for (int c = 0; c < GQA12_H_SUB; ++c) {
                const unsigned int h_local = head_grp * GQA12_H_SUB + (unsigned int)c;
                if (h_local < h_count) {
                    if (lane_id == 0u) {
                        s_red_m[h_local] = m[c];
                        s_red_l[h_local] = l[c];
                    }
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; ++i) {
                        smem_stage.red_o[h_local][vec_off + (unsigned int)i] = o_reg[c][i];
                    }
                }
            }
        }
        __syncthreads();

        if (key_sub == 0u) {
            #pragma unroll
            for (int c = 0; c < GQA12_H_SUB; ++c) {
                const unsigned int h_local = head_grp * GQA12_H_SUB + (unsigned int)c;
                if (h_local < h_count) {
                    float my_m = m[c];
                    float my_l = l[c];
                    const float oth_l = s_red_l[h_local];
                    if (oth_l > 0.0f) {
                        const float oth_m = s_red_m[h_local];
                        const float m_new = fmaxf(my_m, oth_m);
                        const float s0 = (my_l > 0.0f) ? __expf(my_m - m_new) : 0.0f;
                        const float s1 = __expf(oth_m - m_new);
                        my_l = my_l * s0 + oth_l * s1;
                        my_m = m_new;
                        #pragma unroll
                        for (int i = 0; i < VEC_BF16; ++i) {
                            o_reg[c][i] = o_reg[c][i] * s0
                                + smem_stage.red_o[h_local][vec_off + (unsigned int)i] * s1;
                        }
                    }

                    if (seg_split == 0u) {
                        const float inv_l = (my_l > 0.0f) ? (1.0f / my_l) : 0.0f;
                        unsigned int* o32 = (unsigned int*)(O
                            + (unsigned long long)seq_idx * num_q_heads * HDIM
                            + (unsigned long long)(q_head0 + h_local) * HDIM + vec_off);
                        #pragma unroll
                        for (int i = 0; i < VEC_U32; ++i) {
                            unsigned int lo = (unsigned int)__bfloat16_as_ushort(
                                __float2bfloat16(o_reg[c][2*i] * inv_l));
                            unsigned int hi = (unsigned int)__bfloat16_as_ushort(
                                __float2bfloat16(o_reg[c][2*i+1] * inv_l));
                            o32[i] = lo | (hi << 16);
                        }
                    } else {
                        const unsigned int slot = 2u * part_idx + (unsigned int)seg_subslot;
                        float* ws = workspace
                            + (((unsigned long long)group_idx * (2u * G) + slot) * GQA12_MAX_HPC + h_local)
                              * (HDIM + 2u);
                        #pragma unroll
                        for (int i = 0; i < VEC_BF16; ++i) {
                            ws[vec_off + (unsigned int)i] = o_reg[c][i];
                        }
                        if (lane_id == 0u) {
                            ws[HDIM] = my_m;
                            ws[HDIM + 1u] = my_l;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ unsigned int gqa12_apply_sigmoid_gate_pair(
    float x0,
    float x1,
    unsigned int g_packed
) {
    float g0, g1;
    unpack2_pd(g_packed, g0, g1);
    const float s0 = 1.0f / (1.0f + expf(-g0));
    const float s1 = 1.0f / (1.0f + expf(-g1));
    const unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(x0 * s0));
    const unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(x1 * s1));
    return lo | (hi << 16);
}

template <bool GATED>
__device__ __forceinline__ void paged_decode_attn_gqa12_ragged_reduce_bf16_body(
    const float* __restrict__ workspace,
    __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ gate,
    const int* __restrict__ seq_lens,
    const unsigned int num_seqs,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int num_partitions,
    const unsigned int sliding_window,
    const unsigned int gate_stride
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int lane_id = threadIdx.x;
    const unsigned int G = num_partitions;

    if (q_head >= num_q_heads || seq_idx >= num_seqs || num_seqs > GQA12_MAX_SEQS || G == 0u || G > GQA12_MAX_PARTS) {
        return;
    }

    unsigned int running_tot = 0u;
    unsigned int st_b = 0u;
    unsigned int en_b = 0u;
    for (unsigned int base = 0u; base < num_seqs; base += 32u) {
        const unsigned int b = base + lane_id;
        const int raw = (b < num_seqs) ? seq_lens[b] : 0;
        const unsigned int len = (raw > 0) ? (unsigned int)raw : 0u;
        const unsigned int win = (sliding_window > 0u && len > sliding_window) ? (len - sliding_window) : 0u;
        const unsigned int tiles = (len - win + GQA12_SK - 1u) / GQA12_SK;
        const unsigned int inc = running_tot + gqa12_bf16_warp_scan_inc_u32(tiles, lane_id);
        running_tot = __shfl_sync(0xffffffffu, inc, 31);
        if (seq_idx >= base && seq_idx < base + 32u) {
            const int src = (int)(seq_idx - base);
            en_b = __shfl_sync(0xffffffffu, inc, src);
            const unsigned int tb = __shfl_sync(0xffffffffu, tiles, src);
            st_b = en_b - tb;
        }
    }
    const unsigned int total_tiles = running_tot;
    if (en_b == st_b || total_tiles == 0u) return;

    const unsigned int vec_off = lane_id * VEC_BF16;
    bool needs_reduce = true;
    unsigned int mask0 = 0u;
    unsigned int mask1 = 0u;
    unsigned int mask2 = 0u;
    unsigned int mask3 = 0u;
    if (needs_reduce) {
        const unsigned int p0 = lane_id;
        const unsigned int L0 = (p0 < G) ? (unsigned int)(((unsigned long long)total_tiles * p0) / G) : 0u;
        const unsigned int R0 = (p0 < G) ? (unsigned int)(((unsigned long long)total_tiles * (p0 + 1u)) / G) : 0u;
        mask0 = __ballot_sync(0xffffffffu, (p0 < G) && (max(L0, st_b) < min(R0, en_b)));

        const unsigned int p1 = 32u + lane_id;
        const unsigned int L1 = (p1 < G) ? (unsigned int)(((unsigned long long)total_tiles * p1) / G) : 0u;
        const unsigned int R1 = (p1 < G) ? (unsigned int)(((unsigned long long)total_tiles * (p1 + 1u)) / G) : 0u;
        mask1 = __ballot_sync(0xffffffffu, (p1 < G) && (max(L1, st_b) < min(R1, en_b)));

        if (G > 64u) {
            const unsigned int p2 = 64u + lane_id;
            const unsigned int L2 = (p2 < G) ? (unsigned int)(((unsigned long long)total_tiles * p2) / G) : 0u;
            const unsigned int R2 = (p2 < G) ? (unsigned int)(((unsigned long long)total_tiles * (p2 + 1u)) / G) : 0u;
            mask2 = __ballot_sync(0xffffffffu, (p2 < G) && (max(L2, st_b) < min(R2, en_b)));

            const unsigned int p3 = 96u + lane_id;
            const unsigned int L3 = (p3 < G) ? (unsigned int)(((unsigned long long)total_tiles * p3) / G) : 0u;
            const unsigned int R3 = (p3 < G) ? (unsigned int)(((unsigned long long)total_tiles * (p3 + 1u)) / G) : 0u;
            mask3 = __ballot_sync(0xffffffffu, (p3 < G) && (max(L3, st_b) < min(R3, en_b)));
        }

        needs_reduce = (__popc(mask0) + __popc(mask1) + __popc(mask2) + __popc(mask3)) > 1;
    }

    if (!needs_reduce) {
        if constexpr (GATED) {
            // Unsplit sequence: main kernel already normalized and stored BF16 to O.
            // Gate in place with exact sigmoid_gate_mul_batched math (0-ULP).
            unsigned int* o32 = (unsigned int*)(O
                + (unsigned long long)seq_idx * num_q_heads * HDIM
                + (unsigned long long)q_head * HDIM + vec_off);
            const unsigned int* g32 = (const unsigned int*)(gate
                + (unsigned long long)seq_idx * gate_stride
                + (unsigned long long)q_head * HDIM + vec_off);
            #pragma unroll
            for (int i = 0; i < VEC_U32; ++i) {
                float x0, x1;
                unpack2_pd(o32[i], x0, x1);
                o32[i] = gqa12_apply_sigmoid_gate_pair(x0, x1, g32[i]);
            }
        }
        return;
    }

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int in_kv = q_head % gqa_ratio;
    const unsigned int groups_per_kv = (gqa_ratio + GQA12_MAX_HPC - 1u) / GQA12_MAX_HPC;
    const unsigned int group_idx = kv_head * groups_per_kv + (in_kv / GQA12_MAX_HPC);
    const unsigned int h_local = in_kv % GQA12_MAX_HPC;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; ++i) o_reg[i] = 0.0f;

    const unsigned long long masks[2] = {
        ((unsigned long long)mask1 << 32) | (unsigned long long)mask0,
        ((unsigned long long)mask3 << 32) | (unsigned long long)mask2,
    };
    #pragma unroll
    for (unsigned int w = 0u; w < 2u; ++w) {
        unsigned long long mask64 = masks[w];
        const unsigned int p_base = w * 64u;
        while (mask64 != 0ull) {
            const unsigned int p = p_base + (unsigned int)(__ffsll((long long)mask64) - 1);
            mask64 &= (mask64 - 1ull);
            const unsigned int L_p = (unsigned int)(((unsigned long long)total_tiles * p) / G);
            const unsigned int sub_slot = (st_b >= L_p) ? 1u : 0u;
            const unsigned int slot = 2u * p + sub_slot;
            const float* ws = workspace
                + (((unsigned long long)group_idx * (2u * G) + slot) * GQA12_MAX_HPC + h_local)
                  * (HDIM + 2u);
            const float ls = ws[HDIM + 1u];
            if (ls > 0.0f) {
                const float ms = ws[HDIM];
                const float m_new = fmaxf(m, ms);
                const float s_me = (l > 0.0f) ? __expf(m - m_new) : 0.0f;
                const float s_ot = __expf(ms - m_new);
                #pragma unroll
                for (int i = 0; i < VEC_BF16; ++i) {
                    o_reg[i] = o_reg[i] * s_me + ws[vec_off + (unsigned int)i] * s_ot;
                }
                l = l * s_me + ls * s_ot;
                m = m_new;
            }
        }
    }

    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    unsigned int* o32 = (unsigned int*)(O
        + (unsigned long long)seq_idx * num_q_heads * HDIM
        + (unsigned long long)q_head * HDIM + vec_off);
    if constexpr (GATED) {
        const unsigned int* g32 = (const unsigned int*)(gate
            + (unsigned long long)seq_idx * gate_stride
            + (unsigned long long)q_head * HDIM + vec_off);
        #pragma unroll
        for (int i = 0; i < VEC_U32; ++i) {
            // Preserve the intermediate BF16 rounding before sigmoid_gate_mul_batched for 0-ULP parity.
            const float x0 = __bfloat162float(__float2bfloat16(o_reg[2*i] * inv_l));
            const float x1 = __bfloat162float(__float2bfloat16(o_reg[2*i+1] * inv_l));
            o32[i] = gqa12_apply_sigmoid_gate_pair(x0, x1, g32[i]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < VEC_U32; ++i) {
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(o_reg[2*i] * inv_l));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(o_reg[2*i+1] * inv_l));
            o32[i] = lo | (hi << 16);
        }
    }
}

extern "C" __global__ void __launch_bounds__(32, 8) paged_decode_attn_gqa12_ragged_reduce_bf16(
    const float* __restrict__ workspace,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ seq_lens,
    const unsigned int num_seqs,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int num_partitions,
    const unsigned int sliding_window
) {
    paged_decode_attn_gqa12_ragged_reduce_bf16_body<false>(
        workspace,
        O,
        nullptr,
        seq_lens,
        num_seqs,
        num_q_heads,
        num_kv_heads,
        num_partitions,
        sliding_window,
        0u);
}

extern "C" __global__ void __launch_bounds__(32, 8) paged_decode_attn_gqa12_ragged_reduce_gated_bf16(
    const float* __restrict__ workspace,
    __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ gate,
    const int* __restrict__ seq_lens,
    const unsigned int num_seqs,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int num_partitions,
    const unsigned int sliding_window,
    const unsigned int gate_stride
) {
    paged_decode_attn_gqa12_ragged_reduce_bf16_body<true>(
        workspace,
        O,
        gate,
        seq_lens,
        num_seqs,
        num_q_heads,
        num_kv_heads,
        num_partitions,
        sliding_window,
        gate_stride);
}
#endif // HDIM == 256

