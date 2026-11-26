// SPDX-License-Identifier: AGPL-3.0-only
//
// qwen4_attn_prep_fused (common/qwen4_attn_prep_fused.cu) plus the E4M3
// operands of the FP8 QSA prefill core (ATLAS_QWEN4_PREFILL_QSA_FP8).
//
// `qwen4_attn_prep_fused_fp8` computes every BF16 byte (normed + roped Q/K in
// place, paged BF16 K/V) identically to `qwen4_attn_prep_fused` (shfl_down +
// fast-math division). The warp-vectorized variants (`_vec`, `_v2`, `_kv_only`,
// `_q_only`) use an intra-thread + `shfl_xor` reduction with `* (1.0f / 256.0f)`.
// Variants with `kWriteInplaceBf16 == false` (`_v2`, `_v2_k8`, `_kv_only`,
// `_kv_only_k8`, `_q_only`) do NOT write normed+roped Q/K back into `qkv`;
// instead they treat `qkv` as a dead scratch buffer and destructively issue
// `discard.global.L2` on 128-byte-aligned lines after reading.
// In the same pass the kernels also write, from those final BF16 values and a
// static power-of-two prescale:
//   q8[row][head*256 + d]        = e4m3(q * s_q)   (contiguous [rows][nq*256])
//   k8/v8 shadow pages at slot   = e4m3(k * s_k), e4m3(v * s_v)
// The shadow pages have the BF16 cache's [block][block_size][nkv][256] layout at
// one byte per element. Round to nearest even, saturating at +-448. The prescales
// are per (layer, GQA group) for Q and per (layer, KV head) for K and V: head h of
// the nq query heads belongs to group h / (nq / nkv).
//
// qwen4_kv_shadow_e4m3 fills the shadow for tokens [t0, t1) of a sequence from
// its BF16 pages (positions the prep did not write, e.g. after decode).
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint8_t to_e4m3(float x) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

}  // namespace

// Grid rows * (nq + nkv), block 256 (head_dim == 256, one element per thread),
// exactly as qwen4_attn_prep_fused, plus q8 / k8_cache / v8_cache.
extern "C" __global__ void __launch_bounds__(256) qwen4_attn_prep_fused_fp8(
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
    uint8_t* __restrict__ q8,
    uint8_t* __restrict__ k8_cache,
    uint8_t* __restrict__ v8_cache,
    float s_q0,
    float s_q1,
    float s_k0,
    float s_k1,
    float s_v0,
    float s_v1) {
    constexpr unsigned int kHeadDim = 256;
    if (nkv != 2u || (nq != 0u && nq < nkv) || rotary_dim > kHeadDim) __trap();
    (void)block_size;
    const unsigned int heads = nq + nkv;
    const unsigned int row = blockIdx.x / heads;
    const unsigned int head = blockIdx.x - row * heads;
    const bool is_q = head < nq;
    const unsigned int kvh = head - nq;
    __nv_bfloat16* v = qkv + (unsigned long long)row * row_stride +
                       (is_q ? (unsigned long long)head * kHeadDim
                             : (unsigned long long)k_offset + (unsigned long long)kvh * kHeadDim);
    const __nv_bfloat16* weight = is_q ? q_norm_w : k_norm_w;
    const unsigned int tid = threadIdx.x;

    // qwen4_qk_norm_strided, blockDim 256 == head_dim.
    const float val = __bfloat162float(v[tid]);
    float sum_sq = 0.0f;
    sum_sq += val * val;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFFu, sum_sq, offset);
    __shared__ float warp_sums[32];
    __shared__ __nv_bfloat16 y[kHeadDim];
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
    // Runtime divisor, as in qwen4_qk_norm_strided (fast-math division).
    const float rms = rsqrtf(warp_sums[0] / (float)blockDim.x + eps);
    const float w = __bfloat162float(weight[tid]);
    y[tid] = __float2bfloat16(val * rms * (1.0f + w));
    __syncthreads();

    // rope_forward_mrope_interleaved_strided for pair tid.
    const unsigned int half_rot = rotary_dim / 2;
    if (tid < half_rot) {
        const unsigned int section = tid % 3;
        unsigned int abs_pos;
        if (section == 0) abs_pos = pos_t[row];
        else if (section == 1) abs_pos = pos_h[row];
        else abs_pos = pos_w[row];
        const float angle = (float)abs_pos * freq[tid];
        const float cos_val = cosf(angle);
        const float sin_val = sinf(angle);
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
    if (is_q) {
        const float s_q = head / (nq / nkv) ? s_q1 : s_q0;
        q8[((unsigned long long)row * nq + head) * kHeadDim + tid] =
            to_e4m3(__bfloat162float(out) * s_q);
        return;
    }

    // reshape_and_cache_flash for this row's KV head.
    const long long slot = slot_mapping[row];
    if (slot < 0) return;
    const unsigned int n_elems = nkv * kHeadDim;
    const unsigned long long dst =
        (unsigned long long)slot * n_elems + (unsigned long long)kvh * kHeadDim + tid;
    const __nv_bfloat16 vv = qkv[(unsigned long long)row * row_stride + v_offset + kvh * kHeadDim + tid];
    k_cache[dst] = out;
    v_cache[dst] = vv;
    k8_cache[dst] = to_e4m3(__bfloat162float(out) * (kvh ? s_k1 : s_k0));
    if (v8_cache != nullptr) {
        v8_cache[dst] = to_e4m3(__bfloat162float(vv) * (kvh ? s_v1 : s_v0));
    }
}

// Grid ceil((t1 - t0) / 4), block 256: 64 threads per token, 8 elements each,
// over the token's [nkv = 2][256] K and V rows. block_table maps the sequence's
// logical blocks to cache blocks (BF16 pages at 2 bytes, shadow at 1 byte).
extern "C" __global__ void __launch_bounds__(256) qwen4_kv_shadow_e4m3(
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    uint8_t* __restrict__ k8_cache,
    uint8_t* __restrict__ v8_cache,
    const int* __restrict__ block_table,
    unsigned int t0,
    unsigned int t1,
    unsigned int block_size,
    float s_k0,
    float s_k1,
    float s_v0,
    float s_v1) {
    constexpr unsigned int kRowElems = 2 * 256;
    if (block_size == 0u) __trap();
    const unsigned int tok = t0 + blockIdx.x * 4 + threadIdx.x / 64;
    if (tok >= t1) return;
    const int blk = block_table[tok / block_size];
    if (blk < 0) return;
    const unsigned int e = (threadIdx.x % 64) * 8;
    const unsigned long long off =
        (unsigned long long)blk * block_size * kRowElems +
        (unsigned long long)(tok % block_size) * kRowElems + e;
    const uint4 kw = *reinterpret_cast<const uint4*>(k_cache + off);
    const auto pack = [](const uint4 w, float s) {
        const uint32_t in[4] = {w.x, w.y, w.z, w.w};
        uint32_t o[2] = {0u, 0u};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float lo = __uint_as_float(in[i] << 16) * s;
            const float hi = __uint_as_float(in[i] & 0xFFFF0000u) * s;
            const uint32_t b = (uint32_t)to_e4m3(lo) | ((uint32_t)to_e4m3(hi) << 8);
            o[i >> 1] |= b << ((i & 1) * 16);
        }
        return make_uint2(o[0], o[1]);
    };
    const bool h1 = e >= 256;  // this thread's 8 elements lie in one KV head
    *reinterpret_cast<uint2*>(k8_cache + off) = pack(kw, h1 ? s_k1 : s_k0);
    if (v8_cache != nullptr) {
        const uint4 vw = *reinterpret_cast<const uint4*>(v_cache + off);
        *reinterpret_cast<uint2*>(v8_cache + off) = pack(vw, h1 ? s_v1 : s_v0);
    }
}

namespace {

__device__ __forceinline__ uint4 ld_global_v4_u32_evict_first(const void* p) {
    uint4 v;
    asm volatile(
        "{\n\t"
        ".reg .b64 policy;\n\t"
        "createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n\t"
        "ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], policy;\n\t"
        "}\n"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
    return v;
}

__device__ __forceinline__ void st_global_v4_u32_evict_first(void* p, uint4 v) {
    asm volatile(
        "{\n\t"
        ".reg .b64 policy;\n\t"
        "createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n\t"
        "st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, policy;\n\t"
        "}\n"
        :
        : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
        : "memory");
}

__device__ __forceinline__ void st_global_v2_u32_evict_last(void* p, uint2 v) {
    asm volatile(
        "{\n\t"
        ".reg .b64 policy;\n\t"
        "createpolicy.fractional.L2::evict_last.b64 policy, 1.0;\n\t"
        "st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, policy;\n\t"
        "}\n"
        :
        : "l"(p), "r"(v.x), "r"(v.y)
        : "memory");
}

__device__ __forceinline__ void unpack_bf16x8(uint4 u, float (&out)[8]) {
    const uint32_t raw[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&raw[i]);
        const float2 f2 = __bfloat1622float2(b2);
        out[2 * i + 0] = f2.x;
        out[2 * i + 1] = f2.y;
    }
}

template <bool kWriteInplaceBf16>
__device__ __forceinline__ void prep_one_head_warp(
    uint4 v_u4,
    const float (&w_gain)[8],
    const float (&c_rot)[8],
    const float (&s_rot)[8],
    unsigned int lane_id,
    unsigned int e,
    float eps,
    float s_fp8,
    __nv_bfloat16* dst_inplace_bf16,
    __nv_bfloat16* dst_cache_bf16,
    uint8_t* dst_fp8) {
    float val[8];
    unpack_bf16x8(v_u4, val);

    float sq[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sq[i] = val[i] * val[i];
    }
    float sum_sq = ((sq[0] + sq[4]) + (sq[2] + sq[6])) + ((sq[1] + sq[5]) + (sq[3] + sq[7]));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFFu, sum_sq, offset);
    }
    const float rms = rsqrtf(sum_sq * (1.0f / 256.0f) + eps);

    uint32_t n_u32[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_bfloat162 b2 = __floats2bfloat162_rn(
            val[2 * i + 0] * rms * w_gain[2 * i + 0],
            val[2 * i + 1] * rms * w_gain[2 * i + 1]);
        n_u32[i] = *reinterpret_cast<const uint32_t*>(&b2);
    }

    uint32_t p_u32[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        p_u32[i] = __shfl_xor_sync(0xFFFFFFFFu, n_u32[i], 4);
    }

    uint32_t out_u32[4];
    float out_f[8];
    if (lane_id < 8u) {
        const bool first_half = (lane_id < 4u);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 self_f2 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&n_u32[i]));
            const float2 part_f2 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&p_u32[i]));
            const float x0_a = first_half ? self_f2.x : part_f2.x;
            const float x1_a = first_half ? part_f2.x : self_f2.x;
            const float x0_b = first_half ? self_f2.y : part_f2.y;
            const float x1_b = first_half ? part_f2.y : self_f2.y;
            const float r_a = first_half
                                  ? (x0_a * c_rot[2 * i + 0] - x1_a * s_rot[2 * i + 0])
                                  : (x1_a * c_rot[2 * i + 0] + x0_a * s_rot[2 * i + 0]);
            const float r_b = first_half
                                  ? (x0_b * c_rot[2 * i + 1] - x1_b * s_rot[2 * i + 1])
                                  : (x1_b * c_rot[2 * i + 1] + x0_b * s_rot[2 * i + 1]);
            const __nv_bfloat162 ob2 = __floats2bfloat162_rn(r_a, r_b);
            out_u32[i] = *reinterpret_cast<const uint32_t*>(&ob2);
            const float2 of2 = __bfloat1622float2(ob2);
            out_f[2 * i + 0] = of2.x;
            out_f[2 * i + 1] = of2.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out_u32[i] = n_u32[i];
            const float2 of2 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&n_u32[i]));
            out_f[2 * i + 0] = of2.x;
            out_f[2 * i + 1] = of2.y;
        }
    }

    const uint4 out_u4 = make_uint4(out_u32[0], out_u32[1], out_u32[2], out_u32[3]);
    if (kWriteInplaceBf16 && dst_inplace_bf16 != nullptr) {
        st_global_v4_u32_evict_first(dst_inplace_bf16 + e, out_u4);
    } else if (!kWriteInplaceBf16 && dst_inplace_bf16 != nullptr && lane_id < 4u) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(dst_inplace_bf16) + lane_id * 128u;
        if ((reinterpret_cast<uintptr_t>(p) & 127u) == 0u) {
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
        }
    }
    if (dst_cache_bf16 != nullptr) {
        st_global_v4_u32_evict_first(dst_cache_bf16 + e, out_u4);
    }
    if (dst_fp8 != nullptr) {
        const uint32_t fp8_0 = (uint32_t)to_e4m3(out_f[0] * s_fp8) |
                               ((uint32_t)to_e4m3(out_f[1] * s_fp8) << 8) |
                               ((uint32_t)to_e4m3(out_f[2] * s_fp8) << 16) |
                               ((uint32_t)to_e4m3(out_f[3] * s_fp8) << 24);
        const uint32_t fp8_1 = (uint32_t)to_e4m3(out_f[4] * s_fp8) |
                               ((uint32_t)to_e4m3(out_f[5] * s_fp8) << 8) |
                               ((uint32_t)to_e4m3(out_f[6] * s_fp8) << 16) |
                               ((uint32_t)to_e4m3(out_f[7] * s_fp8) << 24);
        st_global_v2_u32_evict_last(dst_fp8 + e, make_uint2(fp8_0, fp8_1));
    }
}

// Grid: rows CTAs, block 256 (8 warps per CTA, 1 warp = 32 lanes x 8 elems = 256-dim head).
// Each CTA processes all 24 Q heads (3 heads/warp), 2 K heads (warps 0-1), and 2 V heads (warps 2-3)
// for a single row, sharing pos/freq/cos/sin and q_norm_w across the row.
template <bool kWriteQkBf16, bool kWriteV8 = true>
__device__ __forceinline__ void attn_prep_fused_fp8_impl(
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
    unsigned int /*block_size*/,
    float eps,
    uint8_t* __restrict__ q8,
    uint8_t* __restrict__ k8_cache,
    uint8_t* __restrict__ v8_cache,
    float s_q0,
    float s_q1,
    float s_k0,
    float s_k1,
    float s_v0,
    float s_v1) {
    constexpr unsigned int kHeadDim = 256;
    constexpr unsigned int kNq = 24;
    constexpr unsigned int kNkv = 2;
    if (nq != kNq || nkv != kNkv || rotary_dim != 64u || blockDim.x != kHeadDim) __trap();
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31u;
    const unsigned int e = lane_id * 8u;

    __nv_bfloat16* row_qkv = qkv + (unsigned long long)row * row_stride;

    // Issue all norm weight and Q/K/V global loads before the single CTA sync.
    const uint4 qw_u4 = __ldg(reinterpret_cast<const uint4*>(q_norm_w + e));
    const uint4 q0_u4 = ld_global_v4_u32_evict_first(row_qkv + (warp_id + 0u) * kHeadDim + e);
    const uint4 q1_u4 = ld_global_v4_u32_evict_first(row_qkv + (warp_id + 8u) * kHeadDim + e);
    const uint4 q2_u4 = ld_global_v4_u32_evict_first(row_qkv + (warp_id + 16u) * kHeadDim + e);

    uint4 kw_u4 = make_uint4(0u, 0u, 0u, 0u);
    uint4 kv_u4 = make_uint4(0u, 0u, 0u, 0u);
    if (warp_id < 2u) {
        kw_u4 = __ldg(reinterpret_cast<const uint4*>(k_norm_w + e));
        kv_u4 = ld_global_v4_u32_evict_first(row_qkv + k_offset + warp_id * kHeadDim + e);
    } else if (warp_id < 4u) {
        const unsigned int kvh = warp_id - 2u;
        kv_u4 = ld_global_v4_u32_evict_first(row_qkv + v_offset + kvh * kHeadDim + e);
    }

    __shared__ float s_cos[32];
    __shared__ float s_sin[32];
    __shared__ long long s_slot;
    if (tid < 32u) {
        const unsigned int section = tid % 3u;
        unsigned int abs_pos;
        if (section == 0u) abs_pos = __ldg(&pos_t[row]);
        else if (section == 1u) abs_pos = __ldg(&pos_h[row]);
        else abs_pos = __ldg(&pos_w[row]);
        const float angle = (float)abs_pos * __ldg(&freq[tid]);
        s_cos[tid] = cosf(angle);
        s_sin[tid] = sinf(angle);
    } else if (tid == 32u) {
        s_slot = __ldg(&slot_mapping[row]);
    }
    __syncthreads();

    float c_rot[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float s_rot[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (lane_id < 8u) {
        const unsigned int pair_base = (lane_id & 3u) * 8u;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c_rot[i] = s_cos[pair_base + i];
            s_rot[i] = s_sin[pair_base + i];
        }
    }

    float qw_gain[8];
    unpack_bf16x8(qw_u4, qw_gain);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        qw_gain[i] += 1.0f;
    }

    uint8_t* row_q8 = q8 + (unsigned long long)row * (kNq * kHeadDim);
    const unsigned int h0 = warp_id + 0u;
    const unsigned int h1 = warp_id + 8u;
    const unsigned int h2 = warp_id + 16u;
    prep_one_head_warp<kWriteQkBf16>(
        q0_u4, qw_gain, c_rot, s_rot, lane_id, e, eps,
        (h0 >= 12u) ? s_q1 : s_q0,
        row_qkv + h0 * kHeadDim, nullptr, row_q8 + h0 * kHeadDim);
    prep_one_head_warp<kWriteQkBf16>(
        q1_u4, qw_gain, c_rot, s_rot, lane_id, e, eps,
        (h1 >= 12u) ? s_q1 : s_q0,
        row_qkv + h1 * kHeadDim, nullptr, row_q8 + h1 * kHeadDim);
    prep_one_head_warp<kWriteQkBf16>(
        q2_u4, qw_gain, c_rot, s_rot, lane_id, e, eps,
        (h2 >= 12u) ? s_q1 : s_q0,
        row_qkv + h2 * kHeadDim, nullptr, row_q8 + h2 * kHeadDim);

    const long long slot = s_slot;
    if (warp_id < 2u) {
        const unsigned int kvh = warp_id;
        float kw_gain[8];
        unpack_bf16x8(kw_u4, kw_gain);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            kw_gain[i] += 1.0f;
        }
        const float sk = kvh ? s_k1 : s_k0;
        const unsigned long long dst_base =
            (slot >= 0) ? ((unsigned long long)slot * (kNkv * kHeadDim) + (unsigned long long)kvh * kHeadDim) : 0ull;
        prep_one_head_warp<kWriteQkBf16>(
            kv_u4, kw_gain, c_rot, s_rot, lane_id, e, eps, sk,
            row_qkv + k_offset + kvh * kHeadDim,
            (slot >= 0) ? (k_cache + dst_base) : nullptr,
            (slot >= 0) ? (k8_cache + dst_base) : nullptr);
    } else if (warp_id < 4u && slot >= 0) {
        const unsigned int kvh = warp_id - 2u;
        const unsigned long long dst_base =
            (unsigned long long)slot * (kNkv * kHeadDim) + (unsigned long long)kvh * kHeadDim;
        st_global_v4_u32_evict_first(v_cache + dst_base + e, kv_u4);
        if constexpr (kWriteV8) {
            if (v8_cache != nullptr) {
                const float sv = kvh ? s_v1 : s_v0;
                float vf[8];
                unpack_bf16x8(kv_u4, vf);
                const uint32_t v8_0 = (uint32_t)to_e4m3(vf[0] * sv) |
                                      ((uint32_t)to_e4m3(vf[1] * sv) << 8) |
                                      ((uint32_t)to_e4m3(vf[2] * sv) << 16) |
                                      ((uint32_t)to_e4m3(vf[3] * sv) << 24);
                const uint32_t v8_1 = (uint32_t)to_e4m3(vf[4] * sv) |
                                      ((uint32_t)to_e4m3(vf[5] * sv) << 8) |
                                      ((uint32_t)to_e4m3(vf[6] * sv) << 16) |
                                      ((uint32_t)to_e4m3(vf[7] * sv) << 24);
                st_global_v2_u32_evict_last(v8_cache + dst_base + e, make_uint2(v8_0, v8_1));
            }
        }
        if (!kWriteQkBf16 && lane_id < 4u) {
            const unsigned char* p =
                reinterpret_cast<const unsigned char*>(row_qkv + v_offset + kvh * kHeadDim) + lane_id * 128u;
            if ((reinterpret_cast<uintptr_t>(p) & 127u) == 0u) {
                asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
            }
        }
    }
}

// Grid: rows CTAs, block 128 (4 warps per CTA: warps 0-1 for 2 K heads, warps 2-3 for 2 V heads).
// Populates paged BF16 K/V and E4M3 K8/(optional V8) shadows identically to
// attn_prep_fused_fp8_impl<false, kWriteV8> while skipping all 24 Q heads.
template <bool kWriteV8>
__device__ __forceinline__ void attn_prep_fused_fp8_kv_only_impl(
    __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    const float* __restrict__ freq,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    unsigned int row_stride,
    unsigned int k_offset,
    unsigned int v_offset,
    float eps,
    uint8_t* __restrict__ k8_cache,
    uint8_t* __restrict__ v8_cache,
    float s_k0,
    float s_k1,
    float s_v0,
    float s_v1) {
    constexpr unsigned int kHeadDim = 256;
    constexpr unsigned int kNkv = 2;
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31u;
    const unsigned int e = lane_id * 8u;

    __nv_bfloat16* row_qkv = qkv + (unsigned long long)row * row_stride;

    uint4 kw_u4 = make_uint4(0u, 0u, 0u, 0u);
    uint4 kv_u4 = make_uint4(0u, 0u, 0u, 0u);
    if (warp_id < 2u) {
        kw_u4 = __ldg(reinterpret_cast<const uint4*>(k_norm_w + e));
        kv_u4 = ld_global_v4_u32_evict_first(row_qkv + k_offset + warp_id * kHeadDim + e);
    } else {
        const unsigned int kvh = warp_id - 2u;
        kv_u4 = ld_global_v4_u32_evict_first(row_qkv + v_offset + kvh * kHeadDim + e);
    }

    __shared__ float s_cos[32];
    __shared__ float s_sin[32];
    __shared__ long long s_slot;
    if (tid < 32u) {
        const unsigned int section = tid % 3u;
        unsigned int abs_pos;
        if (section == 0u) abs_pos = __ldg(&pos_t[row]);
        else if (section == 1u) abs_pos = __ldg(&pos_h[row]);
        else abs_pos = __ldg(&pos_w[row]);
        const float angle = (float)abs_pos * __ldg(&freq[tid]);
        s_cos[tid] = cosf(angle);
        s_sin[tid] = sinf(angle);
    } else if (tid == 32u) {
        s_slot = __ldg(&slot_mapping[row]);
    }
    __syncthreads();

    const long long slot = s_slot;
    if (warp_id < 2u) {
        float c_rot[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float s_rot[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        if (lane_id < 8u) {
            const unsigned int pair_base = (lane_id & 3u) * 8u;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                c_rot[i] = s_cos[pair_base + i];
                s_rot[i] = s_sin[pair_base + i];
            }
        }
        const unsigned int kvh = warp_id;
        float kw_gain[8];
        unpack_bf16x8(kw_u4, kw_gain);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            kw_gain[i] += 1.0f;
        }
        const float sk = kvh ? s_k1 : s_k0;
        const unsigned long long dst_base =
            (slot >= 0) ? ((unsigned long long)slot * (kNkv * kHeadDim) + (unsigned long long)kvh * kHeadDim) : 0ull;
        prep_one_head_warp<false>(
            kv_u4, kw_gain, c_rot, s_rot, lane_id, e, eps, sk,
            row_qkv + k_offset + kvh * kHeadDim,
            (slot >= 0) ? (k_cache + dst_base) : nullptr,
            (slot >= 0) ? (k8_cache + dst_base) : nullptr);
    } else if (slot >= 0) {
        const unsigned int kvh = warp_id - 2u;
        const unsigned long long dst_base =
            (unsigned long long)slot * (kNkv * kHeadDim) + (unsigned long long)kvh * kHeadDim;
        st_global_v4_u32_evict_first(v_cache + dst_base + e, kv_u4);
        if constexpr (kWriteV8) {
            if (v8_cache != nullptr) {
                const float sv = kvh ? s_v1 : s_v0;
                float vf[8];
                unpack_bf16x8(kv_u4, vf);
                const uint32_t v8_0 = (uint32_t)to_e4m3(vf[0] * sv) |
                                      ((uint32_t)to_e4m3(vf[1] * sv) << 8) |
                                      ((uint32_t)to_e4m3(vf[2] * sv) << 16) |
                                      ((uint32_t)to_e4m3(vf[3] * sv) << 24);
                const uint32_t v8_1 = (uint32_t)to_e4m3(vf[4] * sv) |
                                      ((uint32_t)to_e4m3(vf[5] * sv) << 8) |
                                      ((uint32_t)to_e4m3(vf[6] * sv) << 16) |
                                      ((uint32_t)to_e4m3(vf[7] * sv) << 24);
                st_global_v2_u32_evict_last(v8_cache + dst_base + e, make_uint2(v8_0, v8_1));
            }
        }
        if (lane_id < 4u) {
            const unsigned char* p =
                reinterpret_cast<const unsigned char*>(row_qkv + v_offset + kvh * kHeadDim) + lane_id * 128u;
            if ((reinterpret_cast<uintptr_t>(p) & 127u) == 0u) {
                asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
            }
        }
    }
}

// Grid: q_rows CTAs, block 256 (8 warps per CTA, 3 Q heads/warp = 24 Q heads).
// Norms, MRoPEs, and E4M3-quantizes only the Q heads for a cropped tail slice,
// writing contiguous q8[0..q_rows] bit-identically to attn_prep_fused_fp8_impl<false>.
__device__ __forceinline__ void attn_prep_fused_fp8_q_only_impl(
    __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    const float* __restrict__ freq,
    unsigned int row_stride,
    float eps,
    uint8_t* __restrict__ q8,
    float s_q0,
    float s_q1) {
    constexpr unsigned int kHeadDim = 256;
    constexpr unsigned int kNq = 24;
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31u;
    const unsigned int e = lane_id * 8u;

    __nv_bfloat16* row_qkv = qkv + (unsigned long long)row * row_stride;

    const uint4 qw_u4 = __ldg(reinterpret_cast<const uint4*>(q_norm_w + e));
    const uint4 q0_u4 = ld_global_v4_u32_evict_first(row_qkv + (warp_id + 0u) * kHeadDim + e);
    const uint4 q1_u4 = ld_global_v4_u32_evict_first(row_qkv + (warp_id + 8u) * kHeadDim + e);
    const uint4 q2_u4 = ld_global_v4_u32_evict_first(row_qkv + (warp_id + 16u) * kHeadDim + e);

    __shared__ float s_cos[32];
    __shared__ float s_sin[32];
    if (tid < 32u) {
        const unsigned int section = tid % 3u;
        unsigned int abs_pos;
        if (section == 0u) abs_pos = __ldg(&pos_t[row]);
        else if (section == 1u) abs_pos = __ldg(&pos_h[row]);
        else abs_pos = __ldg(&pos_w[row]);
        const float angle = (float)abs_pos * __ldg(&freq[tid]);
        s_cos[tid] = cosf(angle);
        s_sin[tid] = sinf(angle);
    }
    __syncthreads();

    float c_rot[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float s_rot[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (lane_id < 8u) {
        const unsigned int pair_base = (lane_id & 3u) * 8u;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c_rot[i] = s_cos[pair_base + i];
            s_rot[i] = s_sin[pair_base + i];
        }
    }

    float qw_gain[8];
    unpack_bf16x8(qw_u4, qw_gain);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        qw_gain[i] += 1.0f;
    }

    uint8_t* row_q8 = q8 + (unsigned long long)row * (kNq * kHeadDim);
    const unsigned int h0 = warp_id + 0u;
    const unsigned int h1 = warp_id + 8u;
    const unsigned int h2 = warp_id + 16u;
    prep_one_head_warp<false>(
        q0_u4, qw_gain, c_rot, s_rot, lane_id, e, eps,
        (h0 >= 12u) ? s_q1 : s_q0,
        row_qkv + h0 * kHeadDim, nullptr, row_q8 + h0 * kHeadDim);
    prep_one_head_warp<false>(
        q1_u4, qw_gain, c_rot, s_rot, lane_id, e, eps,
        (h1 >= 12u) ? s_q1 : s_q0,
        row_qkv + h1 * kHeadDim, nullptr, row_q8 + h1 * kHeadDim);
    prep_one_head_warp<false>(
        q2_u4, qw_gain, c_rot, s_rot, lane_id, e, eps,
        (h2 >= 12u) ? s_q1 : s_q0,
        row_qkv + h2 * kHeadDim, nullptr, row_q8 + h2 * kHeadDim);
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256) qwen4_attn_prep_fused_fp8_vec(
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
    uint8_t* __restrict__ q8,
    uint8_t* __restrict__ k8_cache,
    uint8_t* __restrict__ v8_cache,
    float s_q0,
    float s_q1,
    float s_k0,
    float s_k1,
    float s_v0,
    float s_v1) {
    attn_prep_fused_fp8_impl<true>(qkv, q_norm_w, k_norm_w, pos_t, pos_h, pos_w, freq, k_cache,
                                   v_cache, slot_mapping, nq, nkv, row_stride, k_offset, v_offset,
                                   rotary_dim, block_size, eps, q8, k8_cache, v8_cache, s_q0, s_q1,
                                   s_k0, s_k1, s_v0, s_v1);
}

extern "C" __global__ void __launch_bounds__(256) qwen4_attn_prep_fused_fp8_v2(
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
    uint8_t* __restrict__ q8,
    uint8_t* __restrict__ k8_cache,
    uint8_t* __restrict__ v8_cache,
    float s_q0,
    float s_q1,
    float s_k0,
    float s_k1,
    float s_v0,
    float s_v1) {
    attn_prep_fused_fp8_impl<false, true>(qkv, q_norm_w, k_norm_w, pos_t, pos_h, pos_w, freq, k_cache,
                                          v_cache, slot_mapping, nq, nkv, row_stride, k_offset, v_offset,
                                          rotary_dim, block_size, eps, q8, k8_cache, v8_cache, s_q0, s_q1,
                                          s_k0, s_k1, s_v0, s_v1);
}

extern "C" __global__ void __launch_bounds__(256) qwen4_attn_prep_fused_fp8_vec_k8(
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
    uint8_t* __restrict__ q8,
    uint8_t* __restrict__ k8_cache,
    uint8_t* __restrict__ v8_cache,
    float s_q0,
    float s_q1,
    float s_k0,
    float s_k1,
    float s_v0,
    float s_v1) {
    attn_prep_fused_fp8_impl<true, false>(qkv, q_norm_w, k_norm_w, pos_t, pos_h, pos_w, freq, k_cache,
                                          v_cache, slot_mapping, nq, nkv, row_stride, k_offset, v_offset,
                                          rotary_dim, block_size, eps, q8, k8_cache, v8_cache, s_q0, s_q1,
                                          s_k0, s_k1, s_v0, s_v1);
}

extern "C" __global__ void __launch_bounds__(256) qwen4_attn_prep_fused_fp8_v2_k8(
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
    uint8_t* __restrict__ q8,
    uint8_t* __restrict__ k8_cache,
    uint8_t* __restrict__ v8_cache,
    float s_q0,
    float s_q1,
    float s_k0,
    float s_k1,
    float s_v0,
    float s_v1) {
    attn_prep_fused_fp8_impl<false, false>(qkv, q_norm_w, k_norm_w, pos_t, pos_h, pos_w, freq, k_cache,
                                           v_cache, slot_mapping, nq, nkv, row_stride, k_offset, v_offset,
                                           rotary_dim, block_size, eps, q8, k8_cache, v8_cache, s_q0, s_q1,
                                           s_k0, s_k1, s_v0, s_v1);
}

extern "C" __global__ void __launch_bounds__(128) qwen4_attn_prep_fused_fp8_kv_only(
    __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    const float* __restrict__ freq,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    unsigned int row_stride,
    unsigned int k_offset,
    unsigned int v_offset,
    float eps,
    uint8_t* __restrict__ k8_cache,
    uint8_t* __restrict__ v8_cache,
    float s_k0,
    float s_k1,
    float s_v0,
    float s_v1) {
    attn_prep_fused_fp8_kv_only_impl<true>(qkv, k_norm_w, pos_t, pos_h, pos_w, freq, k_cache,
                                           v_cache, slot_mapping, row_stride, k_offset, v_offset,
                                           eps, k8_cache, v8_cache, s_k0, s_k1, s_v0, s_v1);
}

extern "C" __global__ void __launch_bounds__(128) qwen4_attn_prep_fused_fp8_kv_only_k8(
    __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    const float* __restrict__ freq,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    unsigned int row_stride,
    unsigned int k_offset,
    unsigned int v_offset,
    float eps,
    uint8_t* __restrict__ k8_cache,
    uint8_t* __restrict__ v8_cache,
    float s_k0,
    float s_k1,
    float s_v0,
    float s_v1) {
    attn_prep_fused_fp8_kv_only_impl<false>(qkv, k_norm_w, pos_t, pos_h, pos_w, freq, k_cache,
                                            v_cache, slot_mapping, row_stride, k_offset, v_offset,
                                            eps, k8_cache, v8_cache, s_k0, s_k1, s_v0, s_v1);
}

extern "C" __global__ void __launch_bounds__(256) qwen4_attn_prep_fused_fp8_q_only(
    __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    const float* __restrict__ freq,
    unsigned int row_stride,
    float eps,
    uint8_t* __restrict__ q8,
    float s_q0,
    float s_q1) {
    attn_prep_fused_fp8_q_only_impl(qkv, q_norm_w, pos_t, pos_h, pos_w, freq, row_stride, eps, q8,
                                    s_q0, s_q1);
}
