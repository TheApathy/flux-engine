// SPDX-License-Identifier: AGPL-3.0-only
//
// Flash-Next full-row attention prep in one pass (ATLAS_QWEN4_PREFILL_ATTN_PREP_FUSED).
//
// Replaces, over a whole prefill chunk of fused QKV rows [q | gate | k | v]:
//   qwen4_qk_norm_strided (Q heads, then K heads)
//   rope_forward_mrope_interleaved_strided (Q and K)
//   reshape_and_cache_flash (K and V into the paged BF16 cache)
// with the same arithmetic in the same order, so every written byte is
// identical; it only drops the intermediate round trips and the per-element
// FP64 pow of the MRoPE frequency (computed once per launch by
// qwen4_rope_freq_table, with the same expression).
//
// The frequency table keeps the exact expression of the MRoPE kernels.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

constexpr unsigned int kPrepHeadDim = 256u;
constexpr unsigned int kNvfp4Group = 16u;

__device__ __forceinline__ unsigned char prep_nvfp4_float_to_e2m1(float x) {
    const unsigned char sign = (unsigned char)((__float_as_uint(x) >> 28) & 8u);
    const unsigned int abits = __float_as_uint(x) & 0x7FFFFFFFu;
    const unsigned char mag = (abits >  0x3E800000u)
                            + (abits >= 0x3F400000u)
                            + (abits >  0x3FA00000u)
                            + (abits >= 0x3FE00000u)
                            + (abits >  0x40200000u)
                            + (abits >= 0x40600000u)
                            + (abits >  0x40A00000u);
    return sign | mag;
}

__device__ __forceinline__ void prep_nvfp4_quantize_head_256(
    float val,
    unsigned char* __restrict__ data_head_dst,
    __nv_fp8_storage_t* __restrict__ scale_head_dst,
    unsigned int tid) {
    const unsigned int group_idx = tid >> 4;   // 0..15
    const unsigned int sub = tid & 15u;        // 0..15
    float absmax = fabsf(val);
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        absmax = fmaxf(absmax, __shfl_xor_sync(0xFFFFFFFFu, absmax, offset));
    }
    const __nv_fp8_storage_t fp8_scale = __nv_cvt_float_to_fp8(
        absmax * (1.0f / 6.0f), __NV_SATFINITE, __NV_E4M3);
    const float dequant_scale = __half2float(__nv_cvt_fp8_to_halfraw(fp8_scale, __NV_E4M3));
    const float inv_scale = (dequant_scale > 0.0f) ? (1.0f / dequant_scale) : 0.0f;
    const unsigned int nib = (unsigned int)prep_nvfp4_float_to_e2m1(val * inv_scale);
    const unsigned int nib_hi = __shfl_down_sync(0xFFFFFFFFu, nib, 1);
    if ((sub & 1u) == 0u) {
        data_head_dst[group_idx * 8u + (sub >> 1)] = (unsigned char)(nib | (nib_hi << 4));
    }
    if (sub == 0u) {
        scale_head_dst[group_idx] = fp8_scale;
    }
}

__device__ __forceinline__ __nv_bfloat16 prep_norm_mrope_head(
    __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ weight,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    const float* __restrict__ freq,
    unsigned int row,
    unsigned int rotary_dim,
    float eps,
    float attn_factor) {
    const unsigned int tid = threadIdx.x;
    const float val = __bfloat162float(v[tid]);
    float sum_sq = val * val;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFFu, sum_sq, offset);
    __shared__ float warp_sums[32];
    __shared__ __nv_bfloat16 y[kPrepHeadDim];
    const unsigned int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float s = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down_sync(0xFFFFFFFFu, s, offset);
        if (lane_id == 0) warp_sums[0] = s;
    }
    __syncthreads();
    const float rms = rsqrtf(warp_sums[0] / (float)blockDim.x + eps);
    const float w = __bfloat162float(weight[tid]);
    y[tid] = __float2bfloat16(val * rms * (1.0f + w));
    __syncthreads();

    const unsigned int half_rot = rotary_dim / 2;
    if (tid < half_rot) {
        const unsigned int section = tid % 3;
        unsigned int abs_pos;
        if (section == 0) abs_pos = pos_t[row];
        else if (section == 1) abs_pos = pos_h[row];
        else abs_pos = pos_w[row];
        const float angle = (float)abs_pos * freq[tid];
        const float cos_val = cosf(angle) * attn_factor;
        const float sin_val = sinf(angle) * attn_factor;
        const unsigned int d0 = tid;
        const unsigned int d1 = tid + half_rot;
        const float x0 = (float)y[d0];
        const float x1 = (float)y[d1];
        const float y0 = x0 * cos_val - x1 * sin_val;
        const float y1 = x1 * cos_val + x0 * sin_val;
        y[d0] = __float2bfloat16(y0);
        y[d1] = __float2bfloat16(y1);
    }
    __syncthreads();
    const __nv_bfloat16 out = y[tid];
    v[tid] = out;
    return out;
}

}  // namespace

extern "C" __global__ void qwen4_rope_freq_table(float* __restrict__ freq,
                                                 unsigned int rotary_dim, float theta) {
    const unsigned int i = threadIdx.x;
    if (i >= rotary_dim / 2) return;
    const double freq_exp_d = (double)(2 * i) / (double)rotary_dim;
    freq[i] = (float)(1.0 / pow((double)theta, freq_exp_d));
}

// Grid rows * (nq + nkv), block 256 (head_dim == 256, one element per thread).
extern "C" __global__ void __launch_bounds__(256) qwen4_attn_prep_fused(
    __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    const float* __restrict__ freq,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    unsigned int nq,
    unsigned int nkv,
    unsigned int row_stride,
    unsigned int k_offset,
    unsigned int v_offset,
    unsigned int rotary_dim,
    unsigned int block_size,
    float eps) {
    const unsigned int heads = nq + nkv;
    const unsigned int row = blockIdx.x / heads;
    const unsigned int head = blockIdx.x - row * heads;
    const bool is_q = head < nq;
    const unsigned int kvh = head - nq;
    __nv_bfloat16* v = qkv + (unsigned long long)row * row_stride +
                       (is_q ? (unsigned long long)head * kPrepHeadDim
                             : (unsigned long long)k_offset + (unsigned long long)kvh * kPrepHeadDim);
    const __nv_bfloat16 out = prep_norm_mrope_head(
        v, is_q ? q_norm_w : k_norm_w, pos_t, pos_h, pos_w, freq, row, rotary_dim, eps, 1.0f);
    if (is_q) return;

    const long long slot = slot_mapping[row];
    if (slot < 0) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int n_elems = nkv * kPrepHeadDim;
    const unsigned long long dst = (unsigned long long)(slot / block_size) * block_size * n_elems +
                                   (unsigned long long)(slot % block_size) * n_elems +
                                   (unsigned long long)kvh * kPrepHeadDim + tid;
    k_cache[dst] = out;
    v_cache[dst] = qkv[(unsigned long long)row * row_stride + v_offset + kvh * kPrepHeadDim + tid];
}

// YaRN-scaled BF16 variant of qwen4_attn_prep_fused (multiplies cos/sin by attn_factor).
extern "C" __global__ void __launch_bounds__(256) qwen4_attn_prep_fused_yarn(
    __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    const float* __restrict__ freq,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    unsigned int nq,
    unsigned int nkv,
    unsigned int row_stride,
    unsigned int k_offset,
    unsigned int v_offset,
    unsigned int rotary_dim,
    unsigned int block_size,
    float eps,
    float attn_factor) {
    const unsigned int heads = nq + nkv;
    const unsigned int row = blockIdx.x / heads;
    const unsigned int head = blockIdx.x - row * heads;
    const bool is_q = head < nq;
    const unsigned int kvh = head - nq;
    __nv_bfloat16* v = qkv + (unsigned long long)row * row_stride +
                       (is_q ? (unsigned long long)head * kPrepHeadDim
                             : (unsigned long long)k_offset + (unsigned long long)kvh * kPrepHeadDim);
    const __nv_bfloat16 out = prep_norm_mrope_head(
        v, is_q ? q_norm_w : k_norm_w, pos_t, pos_h, pos_w, freq, row, rotary_dim, eps, attn_factor);
    if (is_q) return;

    const long long slot = slot_mapping[row];
    if (slot < 0) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int n_elems = nkv * kPrepHeadDim;
    const unsigned long long dst = (unsigned long long)(slot / block_size) * block_size * n_elems +
                                   (unsigned long long)(slot % block_size) * n_elems +
                                   (unsigned long long)kvh * kPrepHeadDim + tid;
    k_cache[dst] = out;
    v_cache[dst] = qkv[(unsigned long long)row * row_stride + v_offset + kvh * kPrepHeadDim + tid];
}

// NVFP4 paged-KV variant of qwen4_attn_prep_fused (with YaRN attn_factor; 1.0f when unscaled).
// Writes packed E2M1 nibbles + E4M3 group-16 scales byte-identically to reshape_and_cache_flash_nvfp4.
extern "C" __global__ void __launch_bounds__(256) qwen4_attn_prep_fused_nvfp4(
    __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    const float* __restrict__ freq,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    unsigned int nq,
    unsigned int nkv,
    unsigned int row_stride,
    unsigned int k_offset,
    unsigned int v_offset,
    unsigned int rotary_dim,
    unsigned int block_size,
    unsigned long long block_stride_bytes,
    unsigned long long data_section_bytes,
    float eps,
    float attn_factor) {
    const unsigned int heads = nq + nkv;
    const unsigned int row = blockIdx.x / heads;
    const unsigned int head = blockIdx.x - row * heads;
    const bool is_q = head < nq;
    const unsigned int kvh = head - nq;
    __nv_bfloat16* v = qkv + (unsigned long long)row * row_stride +
                       (is_q ? (unsigned long long)head * kPrepHeadDim
                             : (unsigned long long)k_offset + (unsigned long long)kvh * kPrepHeadDim);
    const __nv_bfloat16 out = prep_norm_mrope_head(
        v, is_q ? q_norm_w : k_norm_w, pos_t, pos_h, pos_w, freq, row, rotary_dim, eps, attn_factor);
    if (is_q) return;

    const long long slot = slot_mapping[row];
    if (slot < 0) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = nkv * kPrepHeadDim;
    const unsigned int num_groups = n_elems / kNvfp4Group;
    const unsigned long long data_off =
        (unsigned long long)block_offset * (n_elems / 2u) + (unsigned long long)kvh * (kPrepHeadDim / 2u);
    const unsigned long long scale_off =
        data_section_bytes + (unsigned long long)block_offset * num_groups +
        (unsigned long long)kvh * (kPrepHeadDim / kNvfp4Group);
    unsigned char* block_k = k_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * block_stride_bytes;
    const float v_val = __bfloat162float(
        qkv[(unsigned long long)row * row_stride + v_offset + kvh * kPrepHeadDim + tid]);

    prep_nvfp4_quantize_head_256(
        __bfloat162float(out),
        block_k + data_off,
        reinterpret_cast<__nv_fp8_storage_t*>(block_k + scale_off),
        tid);
    prep_nvfp4_quantize_head_256(
        v_val,
        block_v + data_off,
        reinterpret_cast<__nv_fp8_storage_t*>(block_v + scale_off),
        tid);
}

// rope_forward (common/rope.cu) for one tensor [seq_len][heads][head_dim] with
// the frequency read from qwen4_rope_freq_table: grid (heads, ceil(seq_len /
// (128 / (rotary_dim / 2)))), block 128, the same thread mapping and rotation.
extern "C" __global__ void qwen4_rope_table(__nv_bfloat16* __restrict__ x,
                                            const unsigned int* __restrict__ positions,
                                            const float* __restrict__ freq,
                                            unsigned int seq_len,
                                            unsigned int heads,
                                            unsigned int head_dim,
                                            unsigned int rotary_dim) {
    const unsigned int head = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int pairs_per_pos = rotary_dim / 2;
    const unsigned int pos_per_block = 128 / pairs_per_pos;
    const unsigned int local_pos = tid / pairs_per_pos;
    const unsigned int pair_idx = tid % pairs_per_pos;
    const unsigned int seq_pos = blockIdx.y * pos_per_block + local_pos;
    if (seq_pos >= seq_len || local_pos >= pos_per_block) return;
    const unsigned int abs_pos = positions[seq_pos];
    const float angle = (float)abs_pos * freq[pair_idx];
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    __nv_bfloat16* ptr = x + (unsigned long long)seq_pos * (heads * head_dim) + head * head_dim;
    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + half_rot;
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];
    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x1 * cos_val + x0 * sin_val;
    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}
