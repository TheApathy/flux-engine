// SPDX-License-Identifier: AGPL-3.0-only

// Paged Decode Attention — FP8 E4M3 KV cache variant.
//
// Reads FP8 K/V from paged cache, dequantizes to F32 in registers, computes
// attention with BF16 query, writes BF16 output.
//
// Same algorithmic structure as paged_decode_attn.cu (BF16):
//   - One CTA per (q_head, seq) pair
//   - 8 warps split KV sequence
//   - Batched loading (BC=4) within physical blocks
//   - Online softmax with tree-based inter-warp reduction
//
// Key differences from BF16:
//   - K/V cache pointers are __nv_fp8_storage_t* (1 byte/elem vs 2)
//   - Each uint32 load yields 4 FP8 values (vs 2 BF16) → VEC_U32_FP8=2
//   - Dequant: f32_val = fp8_to_f32(byte) * scale
//   - Half the memory loads for same 8 elements per thread
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
#define VEC_U32_FP8 (HDIM / (WARP_SIZE * 4))
#define NUM_WARPS 8
#define BC 4

// K-gamma FP8 specialization: Qwen3.8 verify currently uses M=15. Keep the
// bound explicit so an unsupported verify shape cannot silently index past
// the register tile.
#define KGAMMA_FP8_QTILE 15
#define KGAMMA_FP8_QPER_WARP ((KGAMMA_FP8_QTILE + NUM_WARPS - 1) / NUM_WARPS)

// Unpack 2 BF16 from uint32 → 2 F32 (reused for Q loading)
__device__ __forceinline__ void unpack2_bf16(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

// Unpack 4 FP8 E4M3 from uint32 → 4 unscaled F32.
// FOSCAR affine scale folding: k_scale * inv_sqrt_d is folded into q_reg once
// in the prologue, and v_scale is applied once in the epilogue (or split-K
// partial workspace write), eliminating all 512 per-token scale multiplies.
__device__ __forceinline__ void unpack4_fp8_raw(
    unsigned int packed,
    float& v0, float& v1, float& v2, float& v3
) {
    __nv_fp8_storage_t b0 = (__nv_fp8_storage_t)(packed & 0xFF);
    __nv_fp8_storage_t b1 = (__nv_fp8_storage_t)((packed >> 8) & 0xFF);
    __nv_fp8_storage_t b2 = (__nv_fp8_storage_t)((packed >> 16) & 0xFF);
    __nv_fp8_storage_t b3 = (__nv_fp8_storage_t)((packed >> 24) & 0xFF);
    v0 = __half2float(__nv_cvt_fp8_to_halfraw(b0, __NV_E4M3));
    v1 = __half2float(__nv_cvt_fp8_to_halfraw(b1, __NV_E4M3));
    v2 = __half2float(__nv_cvt_fp8_to_halfraw(b2, __NV_E4M3));
    v3 = __half2float(__nv_cvt_fp8_to_halfraw(b3, __NV_E4M3));
}

// Fused-query FP8 paged attention for the deterministic Qwen3.8 K-gamma
// verify shape. Each warp owns up to two queries and scans their shared KV
// history once. KV positions remain grouped in BC=4 batches, with the exact
// max-rescaled online-softmax update used by paged_decode_attn_fp8. Query dot
// products retain the same XOR warp reduction order.
//
// Contract: num_qtile == 15, identical block-table rows, no tree indirection.
extern "C" __global__ void paged_decode_attn_kgamma_fp8_bc4(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_fp8_storage_t* __restrict__ K_cache,
    const __nv_fp8_storage_t* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float k_scale,
    const float v_scale,
    const unsigned int q_stride,
    const unsigned long long cache_stride,
    const unsigned int num_qtile
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;
    if (q_head >= num_q_heads || num_qtile != KGAMMA_FP8_QTILE || head_dim != HDIM) return;

    unsigned int my_q[KGAMMA_FP8_QPER_WARP];
    unsigned int my_count = 0;
    #pragma unroll
    for (int s = 0; s < KGAMMA_FP8_QPER_WARP; ++s) {
        unsigned int q = warp_id + s * NUM_WARPS;
        my_q[s] = q;
        if (q < num_qtile) ++my_count;
    }
    if (my_count == 0) return;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset = lane_id * VEC_BF16;
    const unsigned long long token_stride = (unsigned long long)num_kv_heads * HDIM;
    const int* block_table = block_tables; // all verify rows are identical

    const float q_scale = k_scale * inv_sqrt_d;
    float q_reg[KGAMMA_FP8_QPER_WARP][VEC_BF16];
    float out[KGAMMA_FP8_QPER_WARP][VEC_BF16];
    float m[KGAMMA_FP8_QPER_WARP], l[KGAMMA_FP8_QPER_WARP];
    unsigned int sl[KGAMMA_FP8_QPER_WARP], max_sl = 0;
    #pragma unroll
    for (int s = 0; s < KGAMMA_FP8_QPER_WARP; ++s) {
        m[s] = -1e30f; l[s] = 0.0f;
        sl[s] = s < (int)my_count ? (unsigned int)seq_lens[my_q[s]] : 0;
        max_sl = max(max_sl, sl[s]);
        #pragma unroll
        for (int i = 0; i < VEC_BF16; ++i) out[s][i] = 0.0f;
        if (s < (int)my_count) {
            const unsigned int* q32 = (const unsigned int*)(Q
                + (unsigned long long)my_q[s] * q_stride
                + (unsigned long long)q_head * HDIM + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; ++i) {
                unpack2_bf16(q32[i], q_reg[s][2*i], q_reg[s][2*i+1]);
                q_reg[s][2*i]   *= q_scale;
                q_reg[s][2*i+1] *= q_scale;
            }
        }
    }

    unsigned int pos = 0;
    while (pos < max_sl) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)block_table[logical_block];
        unsigned int count = min(min(BC, block_size - block_offset), max_sl - pos);
        const __nv_fp8_storage_t* kb = K_cache + (unsigned long long)physical_block * cache_stride
            + (unsigned long long)block_offset * token_stride + (unsigned long long)kv_head * HDIM;
        const __nv_fp8_storage_t* vb = V_cache + (unsigned long long)physical_block * cache_stride
            + (unsigned long long)block_offset * token_stride + (unsigned long long)kv_head * HDIM;
        unsigned int kp[BC][VEC_U32_FP8], vp[BC][VEC_U32_FP8];
        #pragma unroll
        for (int b = 0; b < BC; ++b) {
            if (b < (int)count) {
                const unsigned int* k32 = (const unsigned int*)(kb + (unsigned long long)b * token_stride + vec_offset);
                const unsigned int* v32 = (const unsigned int*)(vb + (unsigned long long)b * token_stride + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; ++i) { kp[b][i] = k32[i]; vp[b][i] = v32[i]; }
            }
        }

        #pragma unroll
        for (int s = 0; s < KGAMMA_FP8_QPER_WARP; ++s) {
            if (s >= (int)my_count) break;
            float scores[BC];
            #pragma unroll
            for (int b = 0; b < BC; ++b) {
                scores[b] = -1e30f;
                if (b < (int)count && pos + b < sl[s]) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VEC_U32_FP8; ++i) {
                        float a, b1, c, d;
                        unpack4_fp8_raw(kp[b][i], a, b1, c, d);
                        dot = fmaf(q_reg[s][4*i],   a,  dot);
                        dot = fmaf(q_reg[s][4*i+1], b1, dot);
                        dot = fmaf(q_reg[s][4*i+2], c,  dot);
                        dot = fmaf(q_reg[s][4*i+3], d,  dot);
                    }
                    #pragma unroll
                    for (int off = WARP_SIZE/2; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
                    scores[b] = dot;
                }
            }
            float m_max = fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[2], scores[3]));
            if (m_max > m[s]) {
                float eo = __expf(m[s] - m_max);
                l[s] *= eo;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; ++i) out[s][i] *= eo;
                m[s] = m_max;
            }
            #pragma unroll
            for (int b = 0; b < BC; ++b) {
                if (scores[b] <= -1e29f) continue;
                float en = __expf(scores[b] - m[s]);
                l[s] += en;
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; ++i) {
                    float a, b1, c, d;
                    unpack4_fp8_raw(vp[b][i], a, b1, c, d);
                    out[s][4*i]   = fmaf(en, a,  out[s][4*i]);
                    out[s][4*i+1] = fmaf(en, b1, out[s][4*i+1]);
                    out[s][4*i+2] = fmaf(en, c,  out[s][4*i+2]);
                    out[s][4*i+3] = fmaf(en, d,  out[s][4*i+3]);
                }
            }
        }
        pos += count;
    }

    #pragma unroll
    for (int s = 0; s < KGAMMA_FP8_QPER_WARP; ++s) {
        if (s >= (int)my_count) break;
        float il = l[s] > 0.0f ? (v_scale / l[s]) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)my_q[s] * num_q_heads * HDIM
            + (unsigned long long)q_head * HDIM + vec_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; ++i) {
            unsigned int lo = __bfloat16_as_ushort(__float2bfloat16(out[s][2*i]*il));
            unsigned int hi = __bfloat16_as_ushort(__float2bfloat16(out[s][2*i+1]*il));
            o32[i] = lo | (hi << 16);
        }
    }
}

// Bit-parity candidate: two queries share one 16-warp CTA, but each query
// retains the legacy kernel's complete eight-warp reduction topology.
extern "C" __global__ void paged_decode_attn_kgamma_fp8_bc4_exact(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_fp8_storage_t* __restrict__ K_cache,
    const __nv_fp8_storage_t* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float k_scale,
    const float v_scale,
    const unsigned int q_stride,
    const unsigned long long cache_stride,
    const unsigned int num_qtile
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int global_warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int group = global_warp / NUM_WARPS;
    const unsigned int warp_id = global_warp % NUM_WARPS;
    const unsigned int query = blockIdx.y * 2 + group;
    if (q_head >= num_q_heads || num_qtile != 15 || head_dim != HDIM) return;
    const bool active = query < num_qtile;
    const unsigned int seq_len = active ? (unsigned int)seq_lens[query] : 0u;
    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset = lane_id * VEC_BF16;
    const unsigned long long token_stride = (unsigned long long)num_kv_heads * HDIM;
    const int* table = block_tables + (unsigned long long)(active ? query : 0u) * max_blocks_per_seq;

    const float q_scale = k_scale * inv_sqrt_d;
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)(active ? query : 0u) * q_stride
        + (unsigned long long)q_head * HDIM + vec_offset);
    float qr[VEC_BF16];
    #pragma unroll
    for (int i=0;i<VEC_U32;i++) {
        unpack2_bf16(q32[i], qr[2*i], qr[2*i+1]);
        qr[2*i]   *= q_scale;
        qr[2*i+1] *= q_scale;
    }

    // Identical legacy partition for this query's local warp id.
    unsigned int chunk = (seq_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int begin = min(warp_id * chunk, seq_len);
    unsigned int end = min(begin + chunk, seq_len);
    float m = -1e30f, l = 0.0f, out[VEC_BF16];
    #pragma unroll
    for (int i=0;i<VEC_BF16;i++) out[i]=0.0f;

    unsigned int pos=begin;
    while(pos<end) {
        unsigned int logical=pos/block_size, off=pos%block_size;
        unsigned int count=min(block_size-off,end-pos);
        unsigned int physical=(unsigned int)table[logical];
        const __nv_fp8_storage_t* kb=K_cache+(unsigned long long)physical*cache_stride
            +(unsigned long long)off*token_stride+(unsigned long long)kv_head*HDIM;
        const __nv_fp8_storage_t* vb=V_cache+(unsigned long long)physical*cache_stride
            +(unsigned long long)off*token_stride+(unsigned long long)kv_head*HDIM;
        unsigned int done=0, aligned=(count/BC)*BC;
        for(;done<aligned;done+=BC) {
            unsigned int kp[BC][VEC_U32_FP8],vp[BC][VEC_U32_FP8];
            #pragma unroll
            for(int b=0;b<BC;b++) {
                const unsigned int* k=(const unsigned int*)(kb+(unsigned long long)(done+b)*token_stride+vec_offset);
                const unsigned int* v=(const unsigned int*)(vb+(unsigned long long)(done+b)*token_stride+vec_offset);
                #pragma unroll
                for(int i=0;i<VEC_U32_FP8;i++){kp[b][i]=k[i];vp[b][i]=v[i];}
            }
            float scores[BC];
            #pragma unroll
            for(int b=0;b<BC;b++) {
                float dot=0.0f;
                #pragma unroll
                for(int i=0;i<VEC_U32_FP8;i++) {
                    float a,b1,c,d; unpack4_fp8_raw(kp[b][i],a,b1,c,d);
                    dot=fmaf(qr[4*i],a,dot);
                    dot=fmaf(qr[4*i+1],b1,dot);
                    dot=fmaf(qr[4*i+2],c,dot);
                    dot=fmaf(qr[4*i+3],d,dot);
                }
                #pragma unroll
                for(int x=WARP_SIZE/2;x>0;x>>=1)dot+=__shfl_xor_sync(0xffffffff,dot,x);
                scores[b]=dot;
            }
            float m_max=fmaxf(fmaxf(scores[0],scores[1]),fmaxf(scores[2],scores[3]));
            if(m_max>m) {
                float eo=__expf(m-m_max);l*=eo;
                #pragma unroll
                for(int i=0;i<VEC_BF16;i++)out[i]*=eo;
                m=m_max;
            }
            float ef[BC];
            #pragma unroll
            for(int b=0;b<BC;b++){ef[b]=__expf(scores[b]-m);l+=ef[b];}
            #pragma unroll
            for(int b=0;b<BC;b++){
                #pragma unroll
                for(int i=0;i<VEC_U32_FP8;i++){
                    float a,b1,c,d;unpack4_fp8_raw(vp[b][i],a,b1,c,d);
                    out[4*i]=fmaf(ef[b],a,out[4*i]);
                    out[4*i+1]=fmaf(ef[b],b1,out[4*i+1]);
                    out[4*i+2]=fmaf(ef[b],c,out[4*i+2]);
                    out[4*i+3]=fmaf(ef[b],d,out[4*i+3]);
                }
            }
        }
        for(;done<count;done++) {
            const unsigned int* k=(const unsigned int*)(kb+(unsigned long long)done*token_stride+vec_offset);
            float dot=0.0f;
            #pragma unroll
            for(int i=0;i<VEC_U32_FP8;i++){
                float a,b1,c,d;unpack4_fp8_raw(k[i],a,b1,c,d);
                dot=fmaf(qr[4*i],a,dot);
                dot=fmaf(qr[4*i+1],b1,dot);
                dot=fmaf(qr[4*i+2],c,dot);
                dot=fmaf(qr[4*i+3],d,dot);
            }
            #pragma unroll
            for(int x=WARP_SIZE/2;x>0;x>>=1)dot+=__shfl_xor_sync(0xffffffff,dot,x);
            float score=dot;
            float en;
            if(score>m){
                float eo=__expf(m-score);
                l=fmaf(l,eo,1.0f);
                #pragma unroll
                for(int i=0;i<VEC_BF16;i++)out[i]*=eo;
                m=score;
                en=1.0f;
            }else{
                en=__expf(score-m);
                l+=en;
            }
            const unsigned int* v=(const unsigned int*)(vb+(unsigned long long)done*token_stride+vec_offset);
            #pragma unroll
            for(int i=0;i<VEC_U32_FP8;i++){
                float a,b1,c,d;unpack4_fp8_raw(v[i],a,b1,c,d);
                out[4*i]=fmaf(en,a,out[4*i]);out[4*i+1]=fmaf(en,b1,out[4*i+1]);
                out[4*i+2]=fmaf(en,c,out[4*i+2]);out[4*i+3]=fmaf(en,d,out[4*i+3]);
            }
        }
        pos+=count;
    }

    __shared__ float sm[2][NUM_WARPS],sl[2][NUM_WARPS],so[2][NUM_WARPS][HDIM];
    if(lane_id==0){sm[group][warp_id]=m;sl[group][warp_id]=l;}
    #pragma unroll
    for(int i=0;i<VEC_BF16;i++)so[group][warp_id][vec_offset+i]=out[i];
    __syncthreads();
    #pragma unroll
    for(int stride=NUM_WARPS/2;stride>0;stride>>=1){
        if(warp_id<(unsigned int)stride){
            unsigned int other=warp_id+stride;
            float lw=sl[group][other];
            if(lw>0.0f){
                float mw=sm[group][other],my_m=sm[group][warp_id],my_l=sl[group][warp_id];
                float mn=fmaxf(my_m,mw),sa=__expf(my_m-mn),sb=__expf(mw-mn);
                float new_l=my_l*sa+lw*sb;
                __syncwarp();
                if(lane_id==0){sl[group][warp_id]=new_l;sm[group][warp_id]=mn;}
                #pragma unroll
                for(int i=0;i<VEC_BF16;i++)so[group][warp_id][vec_offset+i]=
                    so[group][warp_id][vec_offset+i]*sa+so[group][other][vec_offset+i]*sb;
            }
        }
        __syncthreads();
    }
    if(active && warp_id==0){
        float final_l=sl[group][0];
        float il=final_l>0.0f?(v_scale/final_l):0.0f;
        unsigned int* o=(unsigned int*)(O+(unsigned long long)query*num_q_heads*HDIM+(unsigned long long)q_head*HDIM+vec_offset);
        #pragma unroll
        for(int i=0;i<VEC_U32;i++){
            unsigned int lo=__bfloat16_as_ushort(__float2bfloat16(so[group][0][vec_offset+2*i]*il));
            unsigned int hi=__bfloat16_as_ushort(__float2bfloat16(so[group][0][vec_offset+2*i+1]*il));o[i]=lo|(hi<<16);
        }
    }
}

// ============================================================================
// Basic FP8 paged decode attention
// ============================================================================

extern "C" __global__ void paged_decode_attn_fp8(
    const __nv_bfloat16* __restrict__ Q,             // [num_seqs, num_q_heads, head_dim] BF16
    const __nv_fp8_storage_t* __restrict__ K_cache,  // [num_blocks, block_size, num_kv_heads, head_dim] FP8
    const __nv_fp8_storage_t* __restrict__ V_cache,  // [num_blocks, block_size, num_kv_heads, head_dim] FP8
    __nv_bfloat16* __restrict__ O,                   // [num_seqs, num_q_heads, head_dim] BF16
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float k_scale,
    const float v_scale,
    const unsigned int q_stride,              // query.stride(0) in elements
    const unsigned long long cache_stride,    // k_cache.stride(0) in elements (block-level stride)
    // Optional tree-aware KV indirection (ATLAS_TREE_AWARE_ATTN). When
    // `kv_indirection == nullptr`, behavior is identical to the pre-tree
    // path. Otherwise positions in `[kv_indir_base..seq_lens[t])` are
    // remapped via `actual_pos = kv_indir_base + kv_indirection[seq_idx*kv_indir_stride + (pos - kv_indir_base)]`.
    //
    // CUDA graph fix: `kv_indir_base` is read from a 1×i32 device buffer
    // (`kv_indir_base_ptr`) instead of a kernel-launch immediate so the
    // captured graph reads the fresh value on each replay. Host writes the
    // current `seq.seq_len` to the buffer before launching. `nullptr` for
    // legacy / non-tree paths (kernel ignores `kv_indir_base` when
    // `kv_indirection == nullptr`).
    const int* __restrict__ kv_indirection,
    const int* __restrict__ kv_indir_base_ptr,
    const unsigned int kv_indir_stride,
    // ATLAS_TREE_KV_PACK fast path: pre-scattered ancestor KV pool. When
    // `K_pack_pool != nullptr`, positions in `[kv_indir_base..seq_lens[t])`
    // read from this contiguous scratch (one block per seq, BC=4 batched)
    // instead of walking the indirection table. Layout matches the regular
    // FP8 paged cache but with `block_size = pack_block_size`,
    // `num_blocks = num_seqs`, identity block table. Linear context
    // (`[0..kv_indir_base)`) still reads from `K_cache` via `block_tables`
    // — both contributions feed the same online-softmax accumulator.
    const __nv_fp8_storage_t* __restrict__ K_pack_pool,
    const __nv_fp8_storage_t* __restrict__ V_pack_pool,
    const unsigned int pack_block_size
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Load tree-window base position from device buffer (graph-safe).
    // `kv_indir_base_ptr` is nullptr for legacy non-tree callers; in that
    // case `kv_indir_base` is unused (gated by `kv_indirection == nullptr`).
    // `volatile` to defeat any aggressive caching of the load — the value
    // is updated each step via cuMemcpyHtoDAsync before launch.
    const unsigned int kv_indir_base = (kv_indir_base_ptr != nullptr)
        ? (unsigned int)(*((const volatile int*)kv_indir_base_ptr)) : 0u;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    // FP8: each lane covers 8 elements, but byte offset = lane_id * 8
    // (vs BF16 where byte offset = lane_id * 8 * 2)
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;  // BF16 element offset for Q/O
    const unsigned int vec_offset_fp8 = lane_id * VEC_BF16;   // FP8 element offset for K/V cache

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const bool pack_active = (K_pack_pool != nullptr) && (V_pack_pool != nullptr);

    // Load Q (BF16, strided: Q may be a non-contiguous QKV split view)
    // FOSCAR: fold k_scale * inv_sqrt_d into q_reg once in prologue.
    const float q_scale = k_scale * inv_sqrt_d;
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset_bf16);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_bf16(q32[i], q_reg[2*i], q_reg[2*i+1]);
        q_reg[2*i]   *= q_scale;
        q_reg[2*i+1] *= q_scale;
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

    // FP8 cache strides: cache_stride (block-level) passed from host to handle
    // non-contiguous layouts (e.g. vLLM BF16-allocated, uint8-viewed FP8 cache).
    // head_stride_kv (within-block) is always contiguous.
    unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;

    // Tree-aware indirection (see kernel docstring).
    const bool tree_active = (kv_indirection != nullptr);
    const int* my_indir = tree_active
        ? kv_indirection + (unsigned long long)seq_idx * kv_indir_stride
        : nullptr;

    unsigned int pos = my_start;
    while (pos < my_end) {
        // ── ATLAS_TREE_KV_PACK fast path ──
        // When `pack_active`, the tree-window slots have been pre-scattered
        // into a contiguous per-seq block in `K_pack_pool` / `V_pack_pool`
        // with `block_size = pack_block_size`, `num_blocks = num_seqs`, and
        // implicit identity block table. Read BC=4 from the pack pool —
        // no indirection lookups, no block_table walk per position.
        if (pack_active && pos >= kv_indir_base) {
            unsigned int local_off = pos - kv_indir_base;
            unsigned int remaining_total = my_end - pos;
            unsigned int batch_count = remaining_total;
            // Reads stay within the one pack block per seq (pack_block_size
            // slots). Already guaranteed by seq_lens[t] ≤ kv_indir_base + pack_block_size.
            const unsigned long long pack_block_stride =
                (unsigned long long)pack_block_size * head_stride_kv;
            const __nv_fp8_storage_t* k_pack_base = K_pack_pool
                + (unsigned long long)seq_idx * pack_block_stride
                + (unsigned long long)local_off * head_stride_kv
                + (unsigned long long)kv_head * head_dim;
            const __nv_fp8_storage_t* v_pack_base = V_pack_pool
                + (unsigned long long)seq_idx * pack_block_stride
                + (unsigned long long)local_off * head_stride_kv
                + (unsigned long long)kv_head * head_dim;

            unsigned int processed = 0;
            unsigned int aligned_count = (batch_count / BC) * BC;
            for (; processed < aligned_count; processed += BC) {
                unsigned int k_packed[BC][VEC_U32_FP8];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    const unsigned int* k32 = (const unsigned int*)(k_pack_base
                        + (unsigned long long)(processed + b) * head_stride_kv + vec_offset_fp8);
                    #pragma unroll
                    for (int i = 0; i < VEC_U32_FP8; i++)
                        k_packed[b][i] = k32[i];
                }
                float scores[BC];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VEC_U32_FP8; i++) {
                        float k0, k1, k2, k3;
                        unpack4_fp8_raw(k_packed[b][i], k0, k1, k2, k3);
                        dot = fmaf(q_reg[4*i],   k0, dot);
                        dot = fmaf(q_reg[4*i+1], k1, dot);
                        dot = fmaf(q_reg[4*i+2], k2, dot);
                        dot = fmaf(q_reg[4*i+3], k3, dot);
                    }
                    #pragma unroll
                    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                        dot += __shfl_xor_sync(0xffffffff, dot, offset);
                    scores[b] = dot;
                }
                unsigned int v_packed[BC][VEC_U32_FP8];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    const unsigned int* v32 = (const unsigned int*)(v_pack_base
                        + (unsigned long long)(processed + b) * head_stride_kv + vec_offset_fp8);
                    #pragma unroll
                    for (int i = 0; i < VEC_U32_FP8; i++)
                        v_packed[b][i] = v32[i];
                }
                float m_max = fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[2], scores[3]));
                if (m_max > m) {
                    float exp_old = __expf(m - m_max);
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++) o_reg[i] *= exp_old;
                    l *= exp_old;
                    m = m_max;
                }
                float exp_factors[BC];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    exp_factors[b] = __expf(scores[b] - m);
                    l += exp_factors[b];
                }
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    float ef = exp_factors[b];
                    #pragma unroll
                    for (int i = 0; i < VEC_U32_FP8; i++) {
                        float v0, v1, v2, v3;
                        unpack4_fp8_raw(v_packed[b][i], v0, v1, v2, v3);
                        o_reg[4*i]   = fmaf(ef, v0, o_reg[4*i]);
                        o_reg[4*i+1] = fmaf(ef, v1, o_reg[4*i+1]);
                        o_reg[4*i+2] = fmaf(ef, v2, o_reg[4*i+2]);
                        o_reg[4*i+3] = fmaf(ef, v3, o_reg[4*i+3]);
                    }
                }
            }
            // Remainder: single-position from pack pool (still no indirection).
            for (; processed < batch_count; processed++) {
                const unsigned int* k32 = (const unsigned int*)(k_pack_base
                    + (unsigned long long)processed * head_stride_kv + vec_offset_fp8);
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++) {
                    float k0, k1, k2, k3;
                    unpack4_fp8_raw(k32[i], k0, k1, k2, k3);
                    dot = fmaf(q_reg[4*i],   k0, dot);
                    dot = fmaf(q_reg[4*i+1], k1, dot);
                    dot = fmaf(q_reg[4*i+2], k2, dot);
                    dot = fmaf(q_reg[4*i+3], k3, dot);
                }
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                float score = dot;
                float exp_new;
                if (score > m) {
                    float exp_old = __expf(m - score);
                    l = fmaf(l, exp_old, 1.0f);
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++) o_reg[i] *= exp_old;
                    m = score;
                    exp_new = 1.0f;
                } else {
                    exp_new = __expf(score - m);
                    l += exp_new;
                }
                const unsigned int* v32 = (const unsigned int*)(v_pack_base
                    + (unsigned long long)processed * head_stride_kv + vec_offset_fp8);
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++) {
                    float v0, v1, v2, v3;
                    unpack4_fp8_raw(v32[i], v0, v1, v2, v3);
                    o_reg[4*i]   = fmaf(exp_new, v0, o_reg[4*i]);
                    o_reg[4*i+1] = fmaf(exp_new, v1, o_reg[4*i+1]);
                    o_reg[4*i+2] = fmaf(exp_new, v2, o_reg[4*i+2]);
                    o_reg[4*i+3] = fmaf(exp_new, v3, o_reg[4*i+3]);
                }
            }
            pos += batch_count;
            continue;
        }

        // Tree-window single-position fallback when active and past the base
        // (only reached when pack_pool is NULL — i.e. ATLAS_TREE_AWARE_ATTN
        // without ATLAS_TREE_KV_PACK).
        if (tree_active && pos >= kv_indir_base) {
            unsigned int local_off = pos - kv_indir_base;
            int compact_idx = my_indir[local_off];
            unsigned int actual_pos = kv_indir_base + (unsigned int)compact_idx;
            unsigned int logical_block_t = actual_pos / block_size;
            unsigned int block_offset_t = actual_pos % block_size;
            unsigned int physical_block_t = (unsigned int)my_block_table[logical_block_t];
            const __nv_fp8_storage_t* k_base_t = K_cache
                + (unsigned long long)physical_block_t * cache_stride
                + (unsigned long long)block_offset_t * head_stride_kv
                + (unsigned long long)kv_head * head_dim;
            const __nv_fp8_storage_t* v_base_t = V_cache
                + (unsigned long long)physical_block_t * cache_stride
                + (unsigned long long)block_offset_t * head_stride_kv
                + (unsigned long long)kv_head * head_dim;

            const unsigned int* k32 = (const unsigned int*)(k_base_t + vec_offset_fp8);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_U32_FP8; i++) {
                float k0, k1, k2, k3;
                unpack4_fp8_raw(k32[i], k0, k1, k2, k3);
                dot = fmaf(q_reg[4*i],   k0, dot);
                dot = fmaf(q_reg[4*i+1], k1, dot);
                dot = fmaf(q_reg[4*i+2], k2, dot);
                dot = fmaf(q_reg[4*i+3], k3, dot);
            }
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot;
            float exp_new;
            if (score > m) {
                float exp_old = __expf(m - score);
                l = fmaf(l, exp_old, 1.0f);
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) o_reg[i] *= exp_old;
                m = score;
                exp_new = 1.0f;
            } else {
                exp_new = __expf(score - m);
                l += exp_new;
            }

            const unsigned int* v32 = (const unsigned int*)(v_base_t + vec_offset_fp8);
            #pragma unroll
            for (int i = 0; i < VEC_U32_FP8; i++) {
                float v0, v1, v2, v3;
                unpack4_fp8_raw(v32[i], v0, v1, v2, v3);
                o_reg[4*i]   = fmaf(exp_new, v0, o_reg[4*i]);
                o_reg[4*i+1] = fmaf(exp_new, v1, o_reg[4*i+1]);
                o_reg[4*i+2] = fmaf(exp_new, v2, o_reg[4*i+2]);
                o_reg[4*i+3] = fmaf(exp_new, v3, o_reg[4*i+3]);
            }
            pos += 1;
            continue;
        }

        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

        // Don't bleed past kv_indir_base into the indirected window
        // (slow fallback) or the packed window (pack_pool fast path).
        if ((tree_active || pack_active) && pos < kv_indir_base) {
            unsigned int remaining_to_base = kv_indir_base - pos;
            if (batch_count > remaining_to_base) batch_count = remaining_to_base;
        }

        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        const __nv_fp8_storage_t* k_block_base = K_cache + (unsigned long long)physical_block * cache_stride
                                                          + (unsigned long long)block_offset * head_stride_kv
                                                          + (unsigned long long)kv_head * head_dim;
        const __nv_fp8_storage_t* v_block_base = V_cache + (unsigned long long)physical_block * cache_stride
                                                          + (unsigned long long)block_offset * head_stride_kv
                                                          + (unsigned long long)kv_head * head_dim;

        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;

        // Batched path: BC=4 positions at a time
        for (; processed < aligned_count; processed += BC) {
            // Load BC K vectors (FP8: 2 uint32 per thread = 8 FP8 elements)
            unsigned int k_packed[BC][VEC_U32_FP8];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* k32 = (const unsigned int*)(k_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset_fp8);
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++)
                    k_packed[b][i] = k32[i];
            }

            // Compute BC dot products (unscaled FP8 + FMA; k_scale*inv_sqrt_d folded in Q)
            float scores[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++) {
                    float k0, k1, k2, k3;
                    unpack4_fp8_raw(k_packed[b][i], k0, k1, k2, k3);
                    dot = fmaf(q_reg[4*i],   k0, dot);
                    dot = fmaf(q_reg[4*i+1], k1, dot);
                    dot = fmaf(q_reg[4*i+2], k2, dot);
                    dot = fmaf(q_reg[4*i+3], k3, dot);
                }
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                scores[b] = dot;
            }

            // Load BC V vectors
            unsigned int v_packed[BC][VEC_U32_FP8];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* v32 = (const unsigned int*)(v_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset_fp8);
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++)
                    v_packed[b][i] = v32[i];
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

            // V accumulate (unscaled FP8 + FMA; v_scale folded into epilogue)
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float ef = exp_factors[b];
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++) {
                    float v0, v1, v2, v3;
                    unpack4_fp8_raw(v_packed[b][i], v0, v1, v2, v3);
                    o_reg[4*i]   = fmaf(ef, v0, o_reg[4*i]);
                    o_reg[4*i+1] = fmaf(ef, v1, o_reg[4*i+1]);
                    o_reg[4*i+2] = fmaf(ef, v2, o_reg[4*i+2]);
                    o_reg[4*i+3] = fmaf(ef, v3, o_reg[4*i+3]);
                }
            }
        }

        // Remainder: single positions
        for (; processed < batch_count; processed++) {
            const unsigned int* k32 = (const unsigned int*)(k_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset_fp8);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_U32_FP8; i++) {
                float k0, k1, k2, k3;
                unpack4_fp8_raw(k32[i], k0, k1, k2, k3);
                dot = fmaf(q_reg[4*i],   k0, dot);
                dot = fmaf(q_reg[4*i+1], k1, dot);
                dot = fmaf(q_reg[4*i+2], k2, dot);
                dot = fmaf(q_reg[4*i+3], k3, dot);
            }
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot;
            float exp_new;
            if (score > m) {
                float exp_old = __expf(m - score);
                l = fmaf(l, exp_old, 1.0f);
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) o_reg[i] *= exp_old;
                m = score;
                exp_new = 1.0f;
            } else {
                exp_new = __expf(score - m);
                l += exp_new;
            }

            const unsigned int* v32 = (const unsigned int*)(v_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset_fp8);
            #pragma unroll
            for (int i = 0; i < VEC_U32_FP8; i++) {
                float v0, v1, v2, v3;
                unpack4_fp8_raw(v32[i], v0, v1, v2, v3);
                o_reg[4*i]   = fmaf(exp_new, v0, o_reg[4*i]);
                o_reg[4*i+1] = fmaf(exp_new, v1, o_reg[4*i+1]);
                o_reg[4*i+2] = fmaf(exp_new, v2, o_reg[4*i+2]);
                o_reg[4*i+3] = fmaf(exp_new, v3, o_reg[4*i+3]);
            }
        }

        pos += batch_count;
    }

    // Tree-based inter-warp reduction (identical to BF16 version)
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
        // FOSCAR: fold v_scale into final normalization (v_scale / final_l)
        float inv_l = (final_l > 0.0f) ? (v_scale / final_l) : 0.0f;
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
// Split-K FP8 variant for long sequences with few heads
// Grid: (num_q_heads, num_splits, num_seqs)
// ============================================================================

extern "C" __global__ void paged_decode_attn_splitk_fp8(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_fp8_storage_t* __restrict__ K_cache,
    const __nv_fp8_storage_t* __restrict__ V_cache,
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
    const float k_scale,
    const float v_scale,
    const unsigned int q_stride,              // query.stride(0) in elements
    const unsigned long long cache_stride,    // k_cache.stride(0) in elements (block-level stride)
    // See `paged_decode_attn_fp8` for the indirection contract.
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

    unsigned int split_size = (seq_len + num_splits - 1) / num_splits;
    unsigned int kv_start = split_id * split_size;
    unsigned int kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_start >= seq_len) kv_start = kv_end;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;
    const unsigned int vec_offset_fp8 = lane_id * VEC_BF16;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q (BF16, strided: Q may be a non-contiguous QKV split view)
    // FOSCAR: fold k_scale * inv_sqrt_d into q_reg once in prologue.
    const float q_scale = k_scale * inv_sqrt_d;
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset_bf16);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_bf16(q32[i], q_reg[2*i], q_reg[2*i+1]);
        q_reg[2*i]   *= q_scale;
        q_reg[2*i+1] *= q_scale;
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

    // cache_stride (block-level) passed from host; head_stride_kv (within-block) always contiguous.
    unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;

    // Tree-aware indirection (see paged_decode_attn_fp8 contract).
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

        const unsigned int* k32 = (const unsigned int*)(K_cache
            + (unsigned long long)physical_block * cache_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset_fp8);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_U32_FP8; i++) {
            float k0, k1, k2, k3;
            unpack4_fp8_raw(k32[i], k0, k1, k2, k3);
            dot = fmaf(q_reg[4*i],   k0, dot);
            dot = fmaf(q_reg[4*i+1], k1, dot);
            dot = fmaf(q_reg[4*i+2], k2, dot);
            dot = fmaf(q_reg[4*i+3], k3, dot);
        }
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, offset);

        float score = dot;
        float exp_new;
        if (score > m_val) {
            float exp_old = __expf(m_val - score);
            l_val = fmaf(l_val, exp_old, 1.0f);
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) o_reg[i] *= exp_old;
            m_val = score;
            exp_new = 1.0f;
        } else {
            exp_new = __expf(score - m_val);
            l_val += exp_new;
        }

        const unsigned int* v32 = (const unsigned int*)(V_cache
            + (unsigned long long)physical_block * cache_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset_fp8);

        #pragma unroll
        for (int i = 0; i < VEC_U32_FP8; i++) {
            float v0, v1, v2, v3;
            unpack4_fp8_raw(v32[i], v0, v1, v2, v3);
            o_reg[4*i]   = fmaf(exp_new, v0, o_reg[4*i]);
            o_reg[4*i+1] = fmaf(exp_new, v1, o_reg[4*i+1]);
            o_reg[4*i+2] = fmaf(exp_new, v2, o_reg[4*i+2]);
            o_reg[4*i+3] = fmaf(exp_new, v3, o_reg[4*i+3]);
        }
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

    // Write partial to workspace (F32); FOSCAR: apply v_scale once per partial
    // output element so paged_decode_attn_reduce_fp8 sees exact v_scale-scaled partials.
    unsigned int ws_stride = (head_dim + 2);
    float* ws_base = workspace + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride
                   + split_id * ws_stride;

    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) {
            ws_base[vec_offset_bf16 + i] = smem_o[0][vec_offset_bf16 + i] * v_scale;
        }
        if (lane_id == 0) {
            ws_base[head_dim] = smem_m[0];
            ws_base[head_dim + 1] = smem_l[0];
        }
    }
}

// ============================================================================
// Reduce split-K partials into final BF16 output (FP8 variant).
//
// Identical to NVFP4 reduce — the workspace format is quantization-agnostic:
//   workspace[seq_idx, q_head, split, :] = [head_dim F32 values, m, l]
//
// Grid: (num_q_heads, num_seqs, 1)  Block: (32, 1, 1)
// ============================================================================

extern "C" __global__ void paged_decode_attn_reduce_fp8(
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

#if HDIM == 256
// ============================================================================
// GQA-Grouped (up to 12 Q heads / 1 KV head, SK=32) + Warp-0 Ragged Partitioner
// (FP8 E4M3 KV cache, C=1..64).
//
// Combines two orthogonal memory/tail-latency reductions:
//   1. 12:1 GQA shared-memory KV staging (`SK=32`, 16 KiB `s_k`+`s_v`): all 8
//      warps cooperatively stage 32 KV vectors from global PagedKV once per
//      KV-head group, then 4 warp-subgroups (`3` Q heads each x `2` key-split
//      subwarps) reuse those staged keys/values from shared memory — cutting
//      KV DRAM reads up to 12x while keeping per-thread register state at
//      only 3 Q heads (`0` stack frame, `0` spills, `>=2` CTAs/SM).
//   2. Warp-0 prefix-sum ragged tile partitioner (`__shfl_up_sync` over
//      `seq_lens[0..num_seqs)`): partitions total `SK=32` KV tiles evenly
//      across `G = gridDim.x` CTAs per KV-head group. Sequences wholly owned
//      by a single partition normalize and write BF16 directly to `O`; only
//      sequences crossing a partition boundary (`<= 2` per CTA, deterministic
//      slot `2 * part_idx + (st_b >= t_lo ? 1 : 0)`) emit exact `(O, m, l)`
//      partials merged by `paged_decode_attn_gqa12_ragged_reduce_fp8`.
// ============================================================================

#define GQA12_MAX_HPC 12
#define GQA12_H_SUB 3
#define GQA12_SK 32
#define GQA12_MAX_SEQS 256
#define GQA12_MAX_PARTS 64

__device__ __forceinline__ unsigned int gqa12_warp_scan_inc_u32(unsigned int val, unsigned int lane) {
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        unsigned int y = __shfl_up_sync(0xffffffffu, val, o);
        if (lane >= (unsigned int)o) val += y;
    }
    return val;
}

extern "C" __global__ void __launch_bounds__(256, 2) paged_decode_attn_gqa12_ragged_fp8(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_fp8_storage_t* __restrict__ K_cache,
    const __nv_fp8_storage_t* __restrict__ V_cache,
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
    const float k_scale,
    const float v_scale,
    const unsigned int q_stride,
    const unsigned long long cache_stride,
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
            unsigned char k[GQA12_SK][HDIM];
            unsigned char v[GQA12_SK][HDIM];
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
            const unsigned int inc = gqa12_warp_scan_inc_u32(tiles, lane_id);
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
            const unsigned int inc = running_tot + gqa12_warp_scan_inc_u32(tiles, lane_id);
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

    const float q_scale = k_scale * inv_sqrt_d;
    const unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * HDIM;

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
                    unpack2_bf16(q32[i], q_reg[c][2*i], q_reg[c][2*i+1]);
                    q_reg[c][2*i]   *= q_scale;
                    q_reg[c][2*i+1] *= q_scale;
                }
            }
        }

        for (unsigned int tile = seg_t0; tile < seg_t1; ++tile) {
            const unsigned int tile_pos0 = win_start + tile * GQA12_SK;
            const unsigned int rem_keys = seq_len - tile_pos0;
            const unsigned int tile_keys = (rem_keys < GQA12_SK) ? rem_keys : (unsigned int)GQA12_SK;

            // All 8 warps cooperatively stage up to 32 K and V vectors (4 rows/warp).
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const unsigned int k_row = warp_id * 4u + (unsigned int)r;
                if (k_row < tile_keys) {
                    const unsigned int pos = tile_pos0 + k_row;
                    const unsigned int logical_block = pos / block_size;
                    const unsigned int block_offset = pos % block_size;
                    const unsigned int physical_block = (unsigned int)__ldg(&my_block_table[logical_block]);
                    const unsigned long long base_off =
                        (unsigned long long)physical_block * cache_stride
                        + (unsigned long long)block_offset * head_stride_kv
                        + (unsigned long long)kv_head * HDIM + vec_off;
                    const uint2 kv_k = __ldg((const uint2*)(K_cache + base_off));
                    const uint2 kv_v = __ldg((const uint2*)(V_cache + base_off));
                    *(uint2*)(&smem_stage.kv.k[k_row][vec_off]) = kv_k;
                    *(uint2*)(&smem_stage.kv.v[k_row][vec_off]) = kv_v;
                }
            }
            __syncthreads();

            const unsigned int sub_start = key_sub * 16u;
            const unsigned int sub_end = (sub_start + 16u < tile_keys)
                ? (sub_start + 16u)
                : ((sub_start < tile_keys) ? tile_keys : sub_start);

            unsigned int kp = sub_start;
            for (; kp + BC <= sub_end; kp += BC) {
                unsigned int k_pack[BC][VEC_U32_FP8];
                unsigned int v_pack[BC][VEC_U32_FP8];
                #pragma unroll
                for (int b = 0; b < BC; ++b) {
                    const uint2 kk = *(const uint2*)(&smem_stage.kv.k[kp + (unsigned int)b][vec_off]);
                    const uint2 vv = *(const uint2*)(&smem_stage.kv.v[kp + (unsigned int)b][vec_off]);
                    k_pack[b][0] = kk.x; k_pack[b][1] = kk.y;
                    v_pack[b][0] = vv.x; v_pack[b][1] = vv.y;
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
                            for (int i = 0; i < VEC_U32_FP8; ++i) {
                                float k0, k1, k2, k3;
                                unpack4_fp8_raw(k_pack[b][i], k0, k1, k2, k3);
                                dot = fmaf(q_reg[c][4*i],   k0, dot);
                                dot = fmaf(q_reg[c][4*i+1], k1, dot);
                                dot = fmaf(q_reg[c][4*i+2], k2, dot);
                                dot = fmaf(q_reg[c][4*i+3], k3, dot);
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
                            for (int i = 0; i < VEC_U32_FP8; ++i) {
                                float v0, v1, v2, v3;
                                unpack4_fp8_raw(v_pack[b][i], v0, v1, v2, v3);
                                o_reg[c][4*i]   = fmaf(ef, v0, o_reg[c][4*i]);
                                o_reg[c][4*i+1] = fmaf(ef, v1, o_reg[c][4*i+1]);
                                o_reg[c][4*i+2] = fmaf(ef, v2, o_reg[c][4*i+2]);
                                o_reg[c][4*i+3] = fmaf(ef, v3, o_reg[c][4*i+3]);
                            }
                        }
                    }
                }
            }

            for (; kp < sub_end; ++kp) {
                const uint2 kk = *(const uint2*)(&smem_stage.kv.k[kp][vec_off]);
                const uint2 vv = *(const uint2*)(&smem_stage.kv.v[kp][vec_off]);
                unsigned int k1[VEC_U32_FP8] = {kk.x, kk.y};
                unsigned int v1[VEC_U32_FP8] = {vv.x, vv.y};
                #pragma unroll
                for (int c = 0; c < GQA12_H_SUB; ++c) {
                    const unsigned int h_local = head_grp * GQA12_H_SUB + (unsigned int)c;
                    if (h_local < h_count) {
                        float dot = 0.0f;
                        #pragma unroll
                        for (int i = 0; i < VEC_U32_FP8; ++i) {
                            float k0, k1v, k2, k3;
                            unpack4_fp8_raw(k1[i], k0, k1v, k2, k3);
                            dot = fmaf(q_reg[c][4*i],   k0,  dot);
                            dot = fmaf(q_reg[c][4*i+1], k1v, dot);
                            dot = fmaf(q_reg[c][4*i+2], k2,  dot);
                            dot = fmaf(q_reg[c][4*i+3], k3,  dot);
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
                        for (int i = 0; i < VEC_U32_FP8; ++i) {
                            float v0, v1v, v2, v3;
                            unpack4_fp8_raw(v1[i], v0, v1v, v2, v3);
                            o_reg[c][4*i]   = fmaf(exp_new, v0,  o_reg[c][4*i]   * exp_old);
                            o_reg[c][4*i+1] = fmaf(exp_new, v1v, o_reg[c][4*i+1] * exp_old);
                            o_reg[c][4*i+2] = fmaf(exp_new, v2,  o_reg[c][4*i+2] * exp_old);
                            o_reg[c][4*i+3] = fmaf(exp_new, v3,  o_reg[c][4*i+3] * exp_old);
                        }
                    }
                }
            }
            __syncthreads();
        }

        // Merge key_sub=1 into key_sub=0 across smem_stage.red_o.
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
                        const float inv_l = (my_l > 0.0f) ? (v_scale / my_l) : 0.0f;
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
                            ws[vec_off + (unsigned int)i] = o_reg[c][i] * v_scale;
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

extern "C" __global__ void __launch_bounds__(32, 8) paged_decode_attn_gqa12_ragged_reduce_fp8(
    const float* __restrict__ workspace,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ seq_lens,
    const unsigned int num_seqs,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int num_partitions,
    const unsigned int sliding_window
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
        const unsigned int inc = running_tot + gqa12_warp_scan_inc_u32(tiles, lane_id);
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

    const unsigned int p0 = lane_id;
    const unsigned int L0 = (p0 < G) ? (unsigned int)(((unsigned long long)total_tiles * p0) / G) : 0u;
    const unsigned int R0 = (p0 < G) ? (unsigned int)(((unsigned long long)total_tiles * (p0 + 1u)) / G) : 0u;
    const unsigned int mask0 = __ballot_sync(0xffffffffu, (p0 < G) && (max(L0, st_b) < min(R0, en_b)));

    const unsigned int p1 = 32u + lane_id;
    const unsigned int L1 = (p1 < G) ? (unsigned int)(((unsigned long long)total_tiles * p1) / G) : 0u;
    const unsigned int R1 = (p1 < G) ? (unsigned int)(((unsigned long long)total_tiles * (p1 + 1u)) / G) : 0u;
    const unsigned int mask1 = __ballot_sync(0xffffffffu, (p1 < G) && (max(L1, st_b) < min(R1, en_b)));

    const int overlap_count = __popc(mask0) + __popc(mask1);
    if (overlap_count <= 1) {
        // Unsplit sequence: already normalized and written directly to O.
        return;
    }

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int in_kv = q_head % gqa_ratio;
    const unsigned int groups_per_kv = (gqa_ratio + GQA12_MAX_HPC - 1u) / GQA12_MAX_HPC;
    const unsigned int group_idx = kv_head * groups_per_kv + (in_kv / GQA12_MAX_HPC);
    const unsigned int h_local = in_kv % GQA12_MAX_HPC;
    const unsigned int vec_off = lane_id * VEC_BF16;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; ++i) o_reg[i] = 0.0f;

    unsigned long long mask64 = ((unsigned long long)mask1 << 32) | (unsigned long long)mask0;
    while (mask64 != 0ull) {
        const unsigned int p = (unsigned int)(__ffsll((long long)mask64) - 1);
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

    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    unsigned int* o32 = (unsigned int*)(O
        + (unsigned long long)seq_idx * num_q_heads * HDIM
        + (unsigned long long)q_head * HDIM + vec_off);
    #pragma unroll
    for (int i = 0; i < VEC_U32; ++i) {
        unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(o_reg[2*i] * inv_l));
        unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(o_reg[2*i+1] * inv_l));
        o32[i] = lo | (hi << 16);
    }
}
#endif // HDIM == 256

