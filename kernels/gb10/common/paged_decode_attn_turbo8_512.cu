// SPDX-License-Identifier: AGPL-3.0-only
// Upstream attribution: Incorporates TurboQuant+ / Sparse-V / two-sided WHT
// components derived from TheTom/llama-cpp-turboquant (MIT License) and
// TheTom/turboquant_plus (Apache-2.0 License). See docs/CITATIONS.md and
// THIRD_PARTY_NOTICES.md for full upstream copyright and license notices.

// Paged Decode Attention — Turbo8 (WHT + FP8 E4M3 + per-group BF16 scales).
// Operates in the WHT domain (Q rotated at load, output iWHT'd).
//
// Memory layout per KV cache block (K or V separately):
//   [data section: block_size * num_kv_heads * head_dim bytes (1 FP8 E4M3 byte per element)]
//   [scale section: block_size * num_kv_heads * (head_dim/GROUP_SIZE) * 2 bytes (BF16 scales)]
//
// GROUP_SIZE = 16 elements share one BF16 scale.
// Dequant: f32_val = fp8_to_f32(data_byte) * bf16_to_f32(group_scale)
//
// Same algorithmic structure as paged_decode_attn_fp8.cu:
//   - One CTA per (q_head, seq) pair
//   - 8 warps split KV sequence
//   - Batched loading (BC=4) within physical blocks
//   - Online softmax with tree-based inter-warp reduction
//
// Key differences from FP8:
//   - Cache is byte-addressed with appended per-group BF16 scales
//   - No global k_scale/v_scale params — scales embedded per-group in cache
//   - Direct FP8 E4M3 conversion multiplied by per-group BF16 scale
//   - HDIM=256: 1 uint32 load per thread (8 nibbles = 8 elements)
//   - HDIM=128: 1 uint16 load per thread (4 nibbles = 4 elements)
//
// Grid: (num_q_heads, num_seqs, 1)  [splitk: (num_q_heads, num_splits, num_seqs)]
// Block: (256, 1, 1)

#include <cuda_bf16.h>

// Sparse V dequant threshold — skip V load + dequant when attn weight is tiny.
#ifndef TQ_PLUS_SPARSE_V_THRESHOLD
#define TQ_PLUS_SPARSE_V_THRESHOLD 1e-3f
#endif

#include <cuda_fp8.h>

#define WARP_SIZE 32
// HDIM=512 variant for Gemma-4 full-attention layers
#undef HDIM
#define HDIM 512
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

#if VEC_BF16 == 8
struct turbo8_packed_t { unsigned long long lo; };
__device__ __forceinline__ turbo8_packed_t load_turbo8_packed(const unsigned char* data_ptr) {
    turbo8_packed_t p;
    p.lo = *(const unsigned long long*)data_ptr;
    return p;
}
#elif VEC_BF16 == 4
struct turbo8_packed_t { unsigned int lo; };
__device__ __forceinline__ turbo8_packed_t load_turbo8_packed(const unsigned char* data_ptr) {
    turbo8_packed_t p;
    p.lo = *(const unsigned int*)data_ptr;
    return p;
}
#elif VEC_BF16 == 16
struct turbo8_packed_t { unsigned long long lo; unsigned long long hi; };
__device__ __forceinline__ turbo8_packed_t load_turbo8_packed(const unsigned char* data_ptr) {
    turbo8_packed_t p;
    p.lo = *(const unsigned long long*)data_ptr;
    p.hi = *(const unsigned long long*)(data_ptr + 8);
    return p;
}
#else
#error "Unsupported VEC_BF16 (need 4, 8, or 16)"
#endif

// FOSCAR group-scale folding (QK) for Turbo8 (FP8 E4M3 data + BF16 group scale):
// accumulate raw group dot product across VEC_BF16 elements, then multiply by gs_k once.
__device__ __forceinline__ float turbo8_dot_k_packed(
    turbo8_packed_t pk,
    __nv_bfloat16 sc_bf16,
    const float* q_reg
) {
    float gs_k = __bfloat162float(sc_bf16);
#if VEC_BF16 == 8
    unsigned long long pk8 = pk.lo;
    float grp_dot = q_reg[0] * fp8e4m3_to_f32((__nv_fp8_storage_t)(pk8 & 0xFF));
    grp_dot = fmaf(q_reg[1], fp8e4m3_to_f32((__nv_fp8_storage_t)((pk8 >> 8)  & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[2], fp8e4m3_to_f32((__nv_fp8_storage_t)((pk8 >> 16) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[3], fp8e4m3_to_f32((__nv_fp8_storage_t)((pk8 >> 24) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[4], fp8e4m3_to_f32((__nv_fp8_storage_t)((pk8 >> 32) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[5], fp8e4m3_to_f32((__nv_fp8_storage_t)((pk8 >> 40) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[6], fp8e4m3_to_f32((__nv_fp8_storage_t)((pk8 >> 48) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[7], fp8e4m3_to_f32((__nv_fp8_storage_t)(pk8 >> 56)),          grp_dot);
#elif VEC_BF16 == 4
    unsigned int pk4 = pk.lo;
    float grp_dot = q_reg[0] * fp8e4m3_to_f32((__nv_fp8_storage_t)(pk4 & 0xFF));
    grp_dot = fmaf(q_reg[1], fp8e4m3_to_f32((__nv_fp8_storage_t)((pk4 >> 8)  & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[2], fp8e4m3_to_f32((__nv_fp8_storage_t)((pk4 >> 16) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[3], fp8e4m3_to_f32((__nv_fp8_storage_t)(pk4 >> 24)),          grp_dot);
#elif VEC_BF16 == 16
    unsigned long long pk_lo = pk.lo;
    unsigned long long pk_hi = pk.hi;
    float grp_dot = q_reg[0]  * fp8e4m3_to_f32((__nv_fp8_storage_t)(pk_lo & 0xFF));
    grp_dot = fmaf(q_reg[1],  fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_lo >> 8)  & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[2],  fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_lo >> 16) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[3],  fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_lo >> 24) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[4],  fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_lo >> 32) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[5],  fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_lo >> 40) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[6],  fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_lo >> 48) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[7],  fp8e4m3_to_f32((__nv_fp8_storage_t)(pk_lo >> 56)),          grp_dot);
    grp_dot = fmaf(q_reg[8],  fp8e4m3_to_f32((__nv_fp8_storage_t)(pk_hi & 0xFF)),         grp_dot);
    grp_dot = fmaf(q_reg[9],  fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_hi >> 8)  & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[10], fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_hi >> 16) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[11], fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_hi >> 24) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[12], fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_hi >> 32) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[13], fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_hi >> 40) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[14], fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_hi >> 48) & 0xFF)), grp_dot);
    grp_dot = fmaf(q_reg[15], fp8e4m3_to_f32((__nv_fp8_storage_t)(pk_hi >> 56)),          grp_dot);
#endif
    return gs_k * grp_dot;
}

// FOSCAR group-scale folding (PV) for Turbo8:
// fold BF16 group scale gs_v into token softmax weight (w_grp = p_tok * gs_v) once per group, then FMA.
__device__ __forceinline__ void turbo8_accum_v_packed(
    turbo8_packed_t pk,
    __nv_bfloat16 sc_bf16,
    float p_tok,
    float* o_reg
) {
    float gs_v = __bfloat162float(sc_bf16);
    float w_grp = p_tok * gs_v;
#if VEC_BF16 == 8
    unsigned long long pk8 = pk.lo;
    o_reg[0] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)(pk8 & 0xFF)),         o_reg[0]);
    o_reg[1] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk8 >> 8)  & 0xFF)), o_reg[1]);
    o_reg[2] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk8 >> 16) & 0xFF)), o_reg[2]);
    o_reg[3] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk8 >> 24) & 0xFF)), o_reg[3]);
    o_reg[4] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk8 >> 32) & 0xFF)), o_reg[4]);
    o_reg[5] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk8 >> 40) & 0xFF)), o_reg[5]);
    o_reg[6] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk8 >> 48) & 0xFF)), o_reg[6]);
    o_reg[7] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)(pk8 >> 56)),          o_reg[7]);
#elif VEC_BF16 == 4
    unsigned int pk4 = pk.lo;
    o_reg[0] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)(pk4 & 0xFF)),         o_reg[0]);
    o_reg[1] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk4 >> 8)  & 0xFF)), o_reg[1]);
    o_reg[2] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk4 >> 16) & 0xFF)), o_reg[2]);
    o_reg[3] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)(pk4 >> 24)),          o_reg[3]);
#elif VEC_BF16 == 16
    unsigned long long pk_lo = pk.lo;
    unsigned long long pk_hi = pk.hi;
    o_reg[0]  = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)(pk_lo & 0xFF)),         o_reg[0]);
    o_reg[1]  = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_lo >> 8)  & 0xFF)), o_reg[1]);
    o_reg[2]  = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_lo >> 16) & 0xFF)), o_reg[2]);
    o_reg[3]  = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_lo >> 24) & 0xFF)), o_reg[3]);
    o_reg[4]  = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_lo >> 32) & 0xFF)), o_reg[4]);
    o_reg[5]  = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_lo >> 40) & 0xFF)), o_reg[5]);
    o_reg[6]  = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_lo >> 48) & 0xFF)), o_reg[6]);
    o_reg[7]  = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)(pk_lo >> 56)),          o_reg[7]);
    o_reg[8]  = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)(pk_hi & 0xFF)),         o_reg[8]);
    o_reg[9]  = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_hi >> 8)  & 0xFF)), o_reg[9]);
    o_reg[10] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_hi >> 16) & 0xFF)), o_reg[10]);
    o_reg[11] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_hi >> 24) & 0xFF)), o_reg[11]);
    o_reg[12] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_hi >> 32) & 0xFF)), o_reg[12]);
    o_reg[13] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_hi >> 40) & 0xFF)), o_reg[13]);
    o_reg[14] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)((pk_hi >> 48) & 0xFF)), o_reg[14]);
    o_reg[15] = fmaf(w_grp, fp8e4m3_to_f32((__nv_fp8_storage_t)(pk_hi >> 56)),          o_reg[15]);
#endif
}

// ============================================================================
// Basic Turbo8 paged decode attention
// ============================================================================

extern "C" __global__ void paged_decode_attn_turbo8(
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
    const unsigned long long data_section_bytes
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;

    // Turbo8 cache addressing constants
    const unsigned int head_data_bytes = head_dim;  // Turbo8: 1 FP8 byte per element
    const unsigned int head_scale_bytes = (head_dim / NVFP4_GROUP_SIZE) * 2;
    const unsigned int token_data_stride = num_kv_heads * head_data_bytes;
    const unsigned int token_scale_stride = num_kv_heads * head_scale_bytes;
    const unsigned int kv_data_offset = kv_head * head_data_bytes + lane_id * VEC_BF16  /* Turbo8: 1 byte per element */;
    const unsigned int kv_scale_offset = kv_head * head_scale_bytes + (lane_id * VEC_BF16 / NVFP4_GROUP_SIZE) * 2;

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

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        const unsigned char* k_block = K_cache + (unsigned long long)physical_block * block_stride_bytes;
        const unsigned char* v_block = V_cache + (unsigned long long)physical_block * block_stride_bytes;

        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;

        // Batched path: BC=4 positions at a time
        for (; processed < aligned_count; processed += BC) {
            turbo8_packed_t k_pk[BC];
            __nv_bfloat16 k_sc[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                unsigned int p = block_offset + processed + b;
                k_pk[b] = load_turbo8_packed(k_block + p * token_data_stride + kv_data_offset);
                k_sc[b] = *(const __nv_bfloat16*)(k_block + data_section_bytes + p * token_scale_stride + kv_scale_offset);
            }

            float scores[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float dot = turbo8_dot_k_packed(k_pk[b], k_sc[b], q_reg);
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                scores[b] = dot;
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

            // Sparse V load + FOSCAR group-scale folded FMA accumulate
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                if (exp_factors[b] > TQ_PLUS_SPARSE_V_THRESHOLD) {
                    unsigned int p = block_offset + processed + b;
                    const unsigned char* vd = v_block + p * token_data_stride + kv_data_offset;
                    const unsigned char* vs = v_block + data_section_bytes + p * token_scale_stride + kv_scale_offset;
                    turbo8_accum_v_packed(load_turbo8_packed(vd), *(const __nv_bfloat16*)vs, exp_factors[b], o_reg);
                }
            }
        }

        // Remainder: single positions
        for (; processed < batch_count; processed++) {
            unsigned int p = block_offset + processed;
            const unsigned char* kd = k_block + p * token_data_stride + kv_data_offset;
            const unsigned char* ks = k_block + data_section_bytes + p * token_scale_stride + kv_scale_offset;

            float dot = turbo8_dot_k_packed(load_turbo8_packed(kd), *(const __nv_bfloat16*)ks, q_reg);
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

            if (p_tok > TQ_PLUS_SPARSE_V_THRESHOLD) {
                const unsigned char* vd = v_block + p * token_data_stride + kv_data_offset;
                const unsigned char* vs = v_block + data_section_bytes + p * token_scale_stride + kv_scale_offset;
                turbo8_accum_v_packed(load_turbo8_packed(vd), *(const __nv_bfloat16*)vs, p_tok, o_reg);
            }
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
// Split-K Turbo8 variant for long sequences with few heads
// Grid: (num_q_heads, num_splits, num_seqs)
// ============================================================================

extern "C" __global__ void paged_decode_attn_splitk_turbo8(
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
    const unsigned long long data_section_bytes
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

    unsigned int split_size = (seq_len + num_splits - 1) / num_splits;
    unsigned int kv_start = split_id * split_size;
    unsigned int kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_start >= seq_len) kv_start = kv_end;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;

    const unsigned int head_data_bytes = head_dim;  // Turbo8: 1 FP8 byte per element
    const unsigned int head_scale_bytes = (head_dim / NVFP4_GROUP_SIZE) * 2;
    const unsigned int token_data_stride = num_kv_heads * head_data_bytes;
    const unsigned int token_scale_stride = num_kv_heads * head_scale_bytes;
    const unsigned int kv_data_offset = kv_head * head_data_bytes + lane_id * VEC_BF16  /* Turbo8: 1 byte per element */;
    const unsigned int kv_scale_offset = kv_head * head_scale_bytes + (lane_id * VEC_BF16 / NVFP4_GROUP_SIZE) * 2;

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

    for (unsigned int pos = my_start; pos < my_end; pos++) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const unsigned char* k_block = K_cache + (unsigned long long)physical_block * block_stride_bytes;
        const unsigned char* v_block = V_cache + (unsigned long long)physical_block * block_stride_bytes;

        const unsigned char* kd = k_block + block_offset * token_data_stride + kv_data_offset;
        const unsigned char* ks = k_block + data_section_bytes + block_offset * token_scale_stride + kv_scale_offset;

        float dot = turbo8_dot_k_packed(load_turbo8_packed(kd), *(const __nv_bfloat16*)ks, q_reg);
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
        turbo8_accum_v_packed(load_turbo8_packed(vd), *(const __nv_bfloat16*)vs, p_tok, o_reg);
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

extern "C" __global__ void paged_decode_attn_reduce_turbo8(
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
