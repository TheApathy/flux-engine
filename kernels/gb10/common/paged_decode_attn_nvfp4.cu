// SPDX-License-Identifier: AGPL-3.0-only

// Paged Decode Attention — NVFP4 (E2M1 + per-group FP8 scales) KV cache variant.
//
// Memory layout per KV cache block (K or V separately):
//   [data section: block_size * num_kv_heads * head_dim/2 bytes (packed E2M1 nibble pairs)]
//   [scale section: block_size * num_kv_heads * head_dim/GROUP_SIZE bytes (FP8 E4M3 scales)]
//
// GROUP_SIZE = 16 elements share one FP8 E4M3 scale.
// Dequant: f32_val = E2M1_LUT[nibble] * fp8_to_f32(scale_byte)
//
// Same algorithmic structure as paged_decode_attn_fp8.cu:
//   - One CTA per (q_head, seq) pair
//   - 8 warps split KV sequence
//   - Batched loading (BC=4) within physical blocks
//   - Online softmax with tree-based inter-warp reduction
//
// Key differences from FP8:
//   - Cache is byte-addressed with appended per-group FP8 scales
//   - No global k_scale/v_scale params — scales embedded per-group in cache
//   - LUT-based dequant (16-entry E2M1 table in shared memory)
//   - HDIM=256: 1 uint32 load per thread (8 nibbles = 8 elements)
//   - HDIM=128: 1 uint16 load per thread (4 nibbles = 4 elements)
//
// Grid: (num_q_heads, num_seqs, 1)  [splitk: (num_q_heads, num_splits, num_seqs)]
// Block: (256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 256
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32  (HDIM / (WARP_SIZE * 2))
#define NUM_WARPS 8
#define BC 4
#define NVFP4_GROUP_SIZE 16

// ---- Helpers ----------------------------------------------------------------

__device__ __forceinline__ void unpack2_bf16(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ float fp8e4m3_to_f32(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

__device__ __forceinline__ float e2m1_lut_entry(unsigned int idx) {
    unsigned int mag = idx & 7u;
    float val = (mag == 0u) ? 0.0f
              : (mag == 1u) ? 0.5f
              : (float)(2u | (mag & 1u)) * (float)(1u << ((mag >> 1) - 1u)) * 0.5f;
    return copysignf(val, (idx & 8u) ? -1.0f : 1.0f);
}

#if VEC_BF16 == 8
typedef unsigned int nvfp4_packed_t;
#elif VEC_BF16 == 4
typedef unsigned short nvfp4_packed_t;
#else
#error "Unsupported VEC_BF16 (need 4 or 8)"
#endif

// FOSCAR group-scale folding (QK): accumulate raw unscaled group dot product
// across all VEC_BF16 elements in this lane's group, then multiply by gs_k once.
__device__ __forceinline__ float nvfp4_dot_k_packed(
    nvfp4_packed_t pk,
    __nv_fp8_storage_t sc_raw,
    const float* lut,
    const float* q_reg
) {
    float gs_k = fp8e4m3_to_f32(sc_raw);
#if VEC_BF16 == 8
    float grp_dot = q_reg[0] * lut[(pk)       & 0xF];
    grp_dot = fmaf(q_reg[1], lut[(pk >> 4)  & 0xF], grp_dot);
    grp_dot = fmaf(q_reg[2], lut[(pk >> 8)  & 0xF], grp_dot);
    grp_dot = fmaf(q_reg[3], lut[(pk >> 12) & 0xF], grp_dot);
    grp_dot = fmaf(q_reg[4], lut[(pk >> 16) & 0xF], grp_dot);
    grp_dot = fmaf(q_reg[5], lut[(pk >> 20) & 0xF], grp_dot);
    grp_dot = fmaf(q_reg[6], lut[(pk >> 24) & 0xF], grp_dot);
    grp_dot = fmaf(q_reg[7], lut[pk >> 28],         grp_dot);
#elif VEC_BF16 == 4
    float grp_dot = q_reg[0] * lut[(pk)       & 0xF];
    grp_dot = fmaf(q_reg[1], lut[(pk >> 4)  & 0xF], grp_dot);
    grp_dot = fmaf(q_reg[2], lut[(pk >> 8)  & 0xF], grp_dot);
    grp_dot = fmaf(q_reg[3], lut[pk >> 12],         grp_dot);
#endif
    return gs_k * grp_dot;
}

// FOSCAR group-scale folding (PV): fold gs_v into token softmax weight p_tok
// once per group (w_grp = p_tok * gs_v), then FMA raw LUT entries into o_reg.
__device__ __forceinline__ void nvfp4_accum_v_packed(
    nvfp4_packed_t pk,
    __nv_fp8_storage_t sc_raw,
    const float* lut,
    float p_tok,
    float* o_reg
) {
    float gs_v = fp8e4m3_to_f32(sc_raw);
    float w_grp = p_tok * gs_v;
#if VEC_BF16 == 8
    o_reg[0] = fmaf(w_grp, lut[(pk)       & 0xF], o_reg[0]);
    o_reg[1] = fmaf(w_grp, lut[(pk >> 4)  & 0xF], o_reg[1]);
    o_reg[2] = fmaf(w_grp, lut[(pk >> 8)  & 0xF], o_reg[2]);
    o_reg[3] = fmaf(w_grp, lut[(pk >> 12) & 0xF], o_reg[3]);
    o_reg[4] = fmaf(w_grp, lut[(pk >> 16) & 0xF], o_reg[4]);
    o_reg[5] = fmaf(w_grp, lut[(pk >> 20) & 0xF], o_reg[5]);
    o_reg[6] = fmaf(w_grp, lut[(pk >> 24) & 0xF], o_reg[6]);
    o_reg[7] = fmaf(w_grp, lut[pk >> 28],         o_reg[7]);
#elif VEC_BF16 == 4
    o_reg[0] = fmaf(w_grp, lut[(pk)       & 0xF], o_reg[0]);
    o_reg[1] = fmaf(w_grp, lut[(pk >> 4)  & 0xF], o_reg[1]);
    o_reg[2] = fmaf(w_grp, lut[(pk >> 8)  & 0xF], o_reg[2]);
    o_reg[3] = fmaf(w_grp, lut[pk >> 12],         o_reg[3]);
#endif
}

// ============================================================================
// Basic NVFP4 paged decode attention
// ============================================================================

extern "C" __global__ void paged_decode_attn_nvfp4(
    const __nv_bfloat16* __restrict__ Q,          // [num_seqs, num_q_heads, head_dim] BF16
    const unsigned char* __restrict__ K_cache,     // [num_blocks, block_stride_bytes] raw bytes
    const unsigned char* __restrict__ V_cache,     // [num_blocks, block_stride_bytes] raw bytes
    __nv_bfloat16* __restrict__ O,                // [num_seqs, num_q_heads, head_dim] BF16
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    // Optional tree-aware KV indirection (ATLAS_TREE_AWARE_ATTN). When
    // `kv_indirection == nullptr`, kernel behavior is identical to the
    // pre-tree path. When non-null, positions `[kv_indir_base..seq_lens[t])`
    // are remapped via:
    //     actual_pos = kv_indir_base + kv_indirection[seq_idx * kv_indir_stride + (i - kv_indir_base)]
    // Positions below `kv_indir_base` (prior linear context) read normally.
    // Indirection covers the K=γ tree window (≤17 positions per row).
    //
    // CUDA graph fix: `kv_indir_base` lives in a 1×i32 device buffer
    // (`kv_indir_base_ptr`) so captured graphs see the fresh value on
    // each replay. Host writes the current `seq.seq_len` before launch.
    // `nullptr` for legacy / non-tree paths.
    const int* __restrict__ kv_indirection,
    const int* __restrict__ kv_indir_base_ptr,
    const unsigned int kv_indir_stride
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Load tree-window base position from device buffer (graph-safe, volatile).
    const unsigned int kv_indir_base = (kv_indir_base_ptr != nullptr)
        ? (unsigned int)(*((const volatile int*)kv_indir_base_ptr)) : 0u;

    // E2M1 dequant LUT in shared memory (16 floats = 64 bytes, 0 stack frame)
    __shared__ float e2m1_lut[16];
    if (tid < 16) {
        e2m1_lut[tid] = e2m1_lut_entry(tid);
    }
    __syncthreads();

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;

    // NVFP4 cache addressing constants
    const unsigned int head_data_bytes = head_dim / 2;
    const unsigned int head_scale_bytes = head_dim / NVFP4_GROUP_SIZE;
    const unsigned int token_data_stride = num_kv_heads * head_data_bytes;
    const unsigned int token_scale_stride = num_kv_heads * head_scale_bytes;
    const unsigned int kv_data_offset = kv_head * head_data_bytes + lane_id * (VEC_BF16 / 2);
    const unsigned int kv_scale_offset = kv_head * head_scale_bytes + (lane_id * VEC_BF16 / NVFP4_GROUP_SIZE);

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q (BF16, strided) and fold inv_sqrt_d once in prologue (FOSCAR)
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset_bf16);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_bf16(q32[i], q_reg[2*i], q_reg[2*i+1]);
        q_reg[2*i]   *= inv_sqrt_d;
        q_reg[2*i+1] *= inv_sqrt_d;
    }

    unsigned int chunk_size = (seq_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    // Tree-aware indirection: when active and we're past `kv_indir_base`,
    // remap raw position → actual position. Pre-`kv_indir_base` reads use
    // the original batched path (large prior linear context). Tree window
    // is small (≤17 positions) so single-position fallback is acceptable.
    const bool tree_active = (kv_indirection != nullptr);
    const int* my_indir = tree_active
        ? kv_indirection + (unsigned long long)seq_idx * kv_indir_stride
        : nullptr;

    unsigned int pos = my_start;
    while (pos < my_end) {
        // In tree-aware mode, when we're at/past kv_indir_base, take the
        // single-position indirected path (one slot at a time, possibly
        // crossing blocks arbitrarily).
        if (tree_active && pos >= kv_indir_base) {
            unsigned int local_off = pos - kv_indir_base;
            int compact_idx = my_indir[local_off];
            unsigned int actual_pos = kv_indir_base + (unsigned int)compact_idx;
            unsigned int logical_block_t = actual_pos / block_size;
            unsigned int block_offset_t = actual_pos % block_size;
            unsigned int physical_block_t = (unsigned int)my_block_table[logical_block_t];
            const unsigned char* k_block_t = K_cache + (unsigned long long)physical_block_t * block_stride_bytes;
            const unsigned char* v_block_t = V_cache + (unsigned long long)physical_block_t * block_stride_bytes;
            const unsigned char* kd = k_block_t + block_offset_t * token_data_stride + kv_data_offset;
            const unsigned char* ks = k_block_t + data_section_bytes + block_offset_t * token_scale_stride + kv_scale_offset;

            float dot = nvfp4_dot_k_packed(*(const nvfp4_packed_t*)kd, (__nv_fp8_storage_t)*ks, e2m1_lut, q_reg);
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot;
            float p_tok;
            if (score > m) {
                float exp_old = __expf(m - score);
                l = fmaf(l, exp_old, 1.0f);
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++)
                    o_reg[i] *= exp_old;
                m = score;
                p_tok = 1.0f;
            } else {
                p_tok = __expf(score - m);
                l += p_tok;
            }

            const unsigned char* vd = v_block_t + block_offset_t * token_data_stride + kv_data_offset;
            const unsigned char* vs = v_block_t + data_section_bytes + block_offset_t * token_scale_stride + kv_scale_offset;
            nvfp4_accum_v_packed(*(const nvfp4_packed_t*)vd, (__nv_fp8_storage_t)*vs, e2m1_lut, p_tok, o_reg);
            pos += 1;
            continue;
        }

        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

        // Don't let the batched (linear) path bleed into the indirected
        // (tree-window) region — stop short of kv_indir_base when active.
        if (tree_active && pos < kv_indir_base) {
            unsigned int remaining_to_base = kv_indir_base - pos;
            if (batch_count > remaining_to_base) batch_count = remaining_to_base;
        }

        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        const unsigned char* k_block = K_cache + (unsigned long long)physical_block * block_stride_bytes;
        const unsigned char* v_block = V_cache + (unsigned long long)physical_block * block_stride_bytes;

        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;

        // Batched path: BC=4 positions at a time
        for (; processed < aligned_count; processed += BC) {
            // Issue BC packed K + scale loads upfront before LUT/shuffle
            nvfp4_packed_t k_pk[BC];
            __nv_fp8_storage_t k_sc[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                unsigned int p = block_offset + processed + b;
                k_pk[b] = *(const nvfp4_packed_t*)(k_block + p * token_data_stride + kv_data_offset);
                k_sc[b] = (__nv_fp8_storage_t)*(k_block + data_section_bytes + p * token_scale_stride + kv_scale_offset);
            }

            // Compute BC dot products (FOSCAR group-scale folded)
            float scores[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float dot = nvfp4_dot_k_packed(k_pk[b], k_sc[b], e2m1_lut, q_reg);
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                scores[b] = dot;
            }

            // Issue BC packed V + scale loads upfront
            nvfp4_packed_t v_pk[BC];
            __nv_fp8_storage_t v_sc[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                unsigned int p = block_offset + processed + b;
                v_pk[b] = *(const nvfp4_packed_t*)(v_block + p * token_data_stride + kv_data_offset);
                v_sc[b] = (__nv_fp8_storage_t)*(v_block + data_section_bytes + p * token_scale_stride + kv_scale_offset);
            }

            // FA4 conditional batched softmax update: skip rescaling when m_max <= m
            float m_max = fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[2], scores[3]));
            if (m_max > m) {
                float exp_old = __expf(m - m_max);
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++)
                    o_reg[i] *= exp_old;
                l *= exp_old;
                m = m_max;
            }

            float exp_factors[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                exp_factors[b] = __expf(scores[b] - m);
                l += exp_factors[b];
            }

            // V accumulate (FOSCAR: w_grp = ef * gs_v once per group, then FMA)
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                nvfp4_accum_v_packed(v_pk[b], v_sc[b], e2m1_lut, exp_factors[b], o_reg);
            }
        }

        // Remainder: single positions
        for (; processed < batch_count; processed++) {
            unsigned int p = block_offset + processed;
            const unsigned char* kd = k_block + p * token_data_stride + kv_data_offset;
            const unsigned char* ks = k_block + data_section_bytes + p * token_scale_stride + kv_scale_offset;

            float dot = nvfp4_dot_k_packed(*(const nvfp4_packed_t*)kd, (__nv_fp8_storage_t)*ks, e2m1_lut, q_reg);
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot;
            float p_tok;
            if (score > m) {
                float exp_old = __expf(m - score);
                l = fmaf(l, exp_old, 1.0f);
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++)
                    o_reg[i] *= exp_old;
                m = score;
                p_tok = 1.0f;
            } else {
                p_tok = __expf(score - m);
                l += p_tok;
            }

            const unsigned char* vd = v_block + p * token_data_stride + kv_data_offset;
            const unsigned char* vs = v_block + data_section_bytes + p * token_scale_stride + kv_scale_offset;
            nvfp4_accum_v_packed(*(const nvfp4_packed_t*)vd, (__nv_fp8_storage_t)*vs, e2m1_lut, p_tok, o_reg);
        }

        pos += batch_count;
    }

    // Tree-based inter-warp reduction (identical to FP8/BF16)
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset_bf16 + i] = o_reg[i];
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
                    smem_o[warp_id][vec_offset_bf16 + i] =
                        smem_o[warp_id][vec_offset_bf16 + i] * scale_me +
                        smem_o[other][vec_offset_bf16 + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim + vec_offset_bf16);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0 = smem_o[0][vec_offset_bf16 + 2*i]     * inv_l;
            float v1 = smem_o[0][vec_offset_bf16 + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}

// ============================================================================
// Split-K NVFP4 variant for long sequences with few heads
// Grid: (num_q_heads, num_splits, num_seqs)
// ============================================================================

extern "C" __global__ void paged_decode_attn_splitk_nvfp4(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    float* __restrict__ workspace,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    // See `paged_decode_attn_nvfp4` for the indirection contract.
    // `kv_indir_base_ptr` is a 1×i32 device buffer (graph-safe replacement
    // for the prior scalar). Pass `nullptr` for legacy / non-tree paths.
    const int* __restrict__ kv_indirection,
    const int* __restrict__ kv_indir_base_ptr,
    const unsigned int kv_indir_stride
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split_id = blockIdx.y;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Load tree-window base position from device buffer (graph-safe, volatile).
    const unsigned int kv_indir_base = (kv_indir_base_ptr != nullptr)
        ? (unsigned int)(*((const volatile int*)kv_indir_base_ptr)) : 0u;

    // E2M1 dequant LUT (0 stack frame)
    __shared__ float e2m1_lut[16];
    if (tid < 16) {
        e2m1_lut[tid] = e2m1_lut_entry(tid);
    }
    __syncthreads();

    unsigned int split_size = (seq_len + num_splits - 1) / num_splits;
    unsigned int kv_start = split_id * split_size;
    unsigned int kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_start >= seq_len) kv_start = kv_end;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;

    const unsigned int head_data_bytes = head_dim / 2;
    const unsigned int head_scale_bytes = head_dim / NVFP4_GROUP_SIZE;
    const unsigned int token_data_stride = num_kv_heads * head_data_bytes;
    const unsigned int token_scale_stride = num_kv_heads * head_scale_bytes;
    const unsigned int kv_data_offset = kv_head * head_data_bytes + lane_id * (VEC_BF16 / 2);
    const unsigned int kv_scale_offset = kv_head * head_scale_bytes + (lane_id * VEC_BF16 / NVFP4_GROUP_SIZE);

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q (BF16, strided) and fold inv_sqrt_d once in prologue (FOSCAR)
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset_bf16);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_bf16(q32[i], q_reg[2*i], q_reg[2*i+1]);
        q_reg[2*i]   *= inv_sqrt_d;
        q_reg[2*i+1] *= inv_sqrt_d;
    }

    unsigned int local_len = kv_end - kv_start;
    unsigned int chunk_size = (local_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = kv_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;

    float m_val = -1e30f;
    float l_val = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    // Tree-aware indirection (see paged_decode_attn_nvfp4 contract).
    const bool tree_active = (kv_indirection != nullptr);
    const int* my_indir = tree_active
        ? kv_indirection + (unsigned long long)seq_idx * kv_indir_stride
        : nullptr;

    for (unsigned int pos = my_start; pos < my_end; pos++) {
        unsigned int actual_pos = pos;
        if (tree_active && pos >= kv_indir_base) {
            unsigned int local_off = pos - kv_indir_base;
            int compact_idx = my_indir[local_off];
            actual_pos = kv_indir_base + (unsigned int)compact_idx;
        }
        unsigned int logical_block = actual_pos / block_size;
        unsigned int block_offset = actual_pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const unsigned char* k_block = K_cache + (unsigned long long)physical_block * block_stride_bytes;
        const unsigned char* v_block = V_cache + (unsigned long long)physical_block * block_stride_bytes;

        const unsigned char* kd = k_block + block_offset * token_data_stride + kv_data_offset;
        const unsigned char* ks = k_block + data_section_bytes + block_offset * token_scale_stride + kv_scale_offset;

        float dot = nvfp4_dot_k_packed(*(const nvfp4_packed_t*)kd, (__nv_fp8_storage_t)*ks, e2m1_lut, q_reg);
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, offset);

        float score = dot;
        float p_tok;
        if (score > m_val) {
            float exp_old = __expf(m_val - score);
            l_val = fmaf(l_val, exp_old, 1.0f);
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                o_reg[i] *= exp_old;
            m_val = score;
            p_tok = 1.0f;
        } else {
            p_tok = __expf(score - m_val);
            l_val += p_tok;
        }

        const unsigned char* vd = v_block + block_offset * token_data_stride + kv_data_offset;
        const unsigned char* vs = v_block + data_section_bytes + block_offset * token_scale_stride + kv_scale_offset;
        nvfp4_accum_v_packed(*(const nvfp4_packed_t*)vd, (__nv_fp8_storage_t)*vs, e2m1_lut, p_tok, o_reg);
    }

    // Tree merge within CTA
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m_val;
        smem_l[warp_id] = l_val;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset_bf16 + i] = o_reg[i];
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
                    smem_o[warp_id][vec_offset_bf16 + i] =
                        smem_o[warp_id][vec_offset_bf16 + i] * scale_me +
                        smem_o[other][vec_offset_bf16 + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    // Write partial to workspace (F32)
    unsigned int ws_stride = (head_dim + 2);
    float* ws_base = workspace + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride
                   + split_id * ws_stride;

    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) {
            ws_base[vec_offset_bf16 + i] = smem_o[0][vec_offset_bf16 + i];
        }
        if (lane_id == 0) {
            ws_base[head_dim] = smem_m[0];
            ws_base[head_dim + 1] = smem_l[0];
        }
    }
}

// ============================================================================
// Reduction kernel: merge split-K partials → final BF16 output
// Grid: (num_q_heads, num_seqs, 1)  Block: (32, 1, 1)
// ============================================================================

extern "C" __global__ void paged_decode_attn_reduce_nvfp4(
    const float* __restrict__ workspace,    // [num_seqs, num_q_heads, num_splits, (head_dim+2)] F32
    __nv_bfloat16* __restrict__ O,          // [num_seqs, num_q_heads, head_dim] BF16
    const int* __restrict__ seq_lens,
    const unsigned int num_q_heads,
    const unsigned int head_dim,
    const unsigned int num_splits
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int lane_id = threadIdx.x;  // 0..31

    if (q_head >= num_q_heads) return;
    if (seq_lens[seq_idx] == 0) return;

    const unsigned int vec_off = lane_id * VEC_BF16;
    const unsigned int ws_stride = head_dim + 2;
    const float* ws_base = workspace
        + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride;

    // Load split 0
    float m = ws_base[head_dim];
    float l = ws_base[head_dim + 1];
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++)
        o_reg[i] = ws_base[vec_off + i];

    // Merge splits 1..num_splits-1
    for (unsigned int s = 1; s < num_splits; s++) {
        const float* ws = ws_base + s * ws_stride;
        float ms = ws[head_dim];
        float ls = ws[head_dim + 1];

        if (ls <= 0.0f) continue;

        float m_new = fmaxf(m, ms);
        float scale_me = __expf(m - m_new);
        float scale_s = __expf(ms - m_new);

        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            o_reg[i] = o_reg[i] * scale_me + ws[vec_off + i] * scale_s;

        l = l * scale_me + ls * scale_s;
        m = m_new;
    }

    // Normalize and write BF16 output
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                          + (unsigned long long)q_head * head_dim + vec_off);
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        float v0 = o_reg[2*i] * inv_l;
        float v1 = o_reg[2*i + 1] * inv_l;
        unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
        unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
        o32[i] = lo | (hi << 16);
    }
}
