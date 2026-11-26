// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// Templatized Split-K partial reduction & shared-gate implementations for V33.
// Included by `qwen4_moe_thin_sk_v33.cu` after `v33_gemm_impl.cuh`.

namespace {

template <typename Shape>
__device__ __forceinline__ void v33_reduce_silu_impl(
    const float* __restrict__ partials,
    __nv_bfloat16* __restrict__ act_out,
    unsigned int pairs,
    unsigned int split_k
) {
    static_assert((Shape::INTER % 128u) == 0u, "v33_reduce_silu requires INTER divisible by 128");
    constexpr unsigned INTER_SEGS = Shape::INTER / 128u;

    const unsigned prow = blockIdx.x;
    const unsigned seg = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (prow >= pairs || seg >= INTER_SEGS) return;

    const unsigned col = seg * 128u + lane * 4u;
    const unsigned long long slice_stride =
        static_cast<unsigned long long>(pairs) * Shape::GATEUP_PAIR_STRIDE_ULL;
    const unsigned long long base_g =
        static_cast<unsigned long long>(prow) * Shape::GATEUP_PAIR_STRIDE_ULL + col;
    const unsigned long long base_u = base_g + Shape::INTER_ULL;

    qwen4_nvfp4_detail::pdl_wait();

    float4 g = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 u = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (unsigned s = 0u; s < split_k; ++s) {
        const float4 pg = *reinterpret_cast<const float4*>(partials + s * slice_stride + base_g);
        const float4 pu = *reinterpret_cast<const float4*>(partials + s * slice_stride + base_u);
        g.x += pg.x; g.y += pg.y; g.z += pg.z; g.w += pg.w;
        u.x += pu.x; u.y += pu.y; u.z += pu.z; u.w += pu.w;
    }

    const float a0 = g.x * (1.0f / (1.0f + __expf(-g.x))) * u.x;
    const float a1 = g.y * (1.0f / (1.0f + __expf(-g.y))) * u.y;
    const float a2 = g.z * (1.0f / (1.0f + __expf(-g.z))) * u.z;
    const float a3 = g.w * (1.0f / (1.0f + __expf(-g.w))) * u.w;

    const uint2 packed = make_uint2(
        qwen4_nvfp4_detail::pack_bf162(a0, a1),
        qwen4_nvfp4_detail::pack_bf162(a2, a3));
    v33_st_wb_u2(act_out + static_cast<unsigned long long>(prow) * Shape::INTER_ULL + col, packed);
    qwen4_nvfp4_detail::pdl_launch_dependents();
}

template <typename Shape>
__device__ __forceinline__ void v33_down_reduce_bf16_impl(
    const float* __restrict__ partials,
    __nv_bfloat16* __restrict__ C,
    unsigned int pairs,
    unsigned int split_k
) {
    static_assert((Shape::HIDDEN % 128u) == 0u, "v33_down_reduce_bf16 requires HIDDEN divisible by 128");
    constexpr unsigned HIDDEN_SEGS = Shape::HIDDEN / 128u;

    const unsigned prow = blockIdx.x;
    const unsigned seg = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (prow >= pairs || seg >= HIDDEN_SEGS) return;

    const unsigned col = seg * 128u + tid;
    const unsigned long long slice_stride =
        static_cast<unsigned long long>(pairs) * Shape::HIDDEN_ULL;
    const unsigned long long base =
        static_cast<unsigned long long>(prow) * Shape::HIDDEN_ULL + col;

    qwen4_nvfp4_detail::pdl_wait();

    float sum = 0.0f;
    for (unsigned s = 0u; s < split_k; ++s) {
        sum += partials[s * slice_stride + base];
    }
    C[base] = __float2bfloat16(sum);
}

template <typename Shape>
__device__ __forceinline__ void v33_down_reduce_mxfp8_impl(
    const float* __restrict__ partials,
    unsigned char* __restrict__ slot_q,
    unsigned char* __restrict__ slot_s,
    unsigned int pairs,
    unsigned int split_k
) {
    static_assert((Shape::HIDDEN % 128u) == 0u, "v33_down_reduce_mxfp8 requires HIDDEN divisible by 128");
    constexpr unsigned HIDDEN_SEGS = Shape::HIDDEN / 128u;

    const unsigned prow = blockIdx.x;
    const unsigned seg = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (prow >= pairs || seg >= HIDDEN_SEGS) return;

    const unsigned warp_id = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned col = seg * 128u + tid;
    const unsigned g_blk = seg * 4u + warp_id;
    const unsigned long long slice_stride =
        static_cast<unsigned long long>(pairs) * Shape::HIDDEN_ULL;
    const unsigned long long base =
        static_cast<unsigned long long>(prow) * Shape::HIDDEN_ULL + col;

    qwen4_nvfp4_detail::pdl_wait();

    float sum = 0.0f;
    for (unsigned s = 0u; s < split_k; ++s) {
        sum += partials[s * slice_stride + base];
    }
    const float vb = __bfloat162float(__float2bfloat16(sum));
    float amax = fabsf(vb);
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 16));

    unsigned char qb = 0u;
    unsigned char sb = 0u;
    if (amax > 0.0f) {
        int e;
        const float m = frexpf(__fmul_rn(amax, 1.0f / 448.0f), &e);
        int k = (m == 0.5f) ? e - 1 : e;
        k = max(-127, min(127, k));
        const float inv = __int_as_float((127 - k) << 23);
        qb = qwen4_nvfp4_detail::cvt_f32_e4m3(vb * inv);
        sb = static_cast<unsigned char>(k + 127);
    }
    const bool keep_slot =
        (static_cast<unsigned long long>(pairs) * Shape::SLOT_MXFP8_BYTES_ULL) <= V33_KEEP_SLOT_BYTES;
    if (keep_slot) {
        unsigned long long pol;
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
        asm volatile("st.global.L2::cache_hint.u8 [%0], %1, %2;\n"
                     :: "l"(slot_q + base), "r"(static_cast<unsigned>(qb)), "l"(pol)
                     : "memory");
    } else {
        slot_q[base] = qb;
    }
    if (lane == 0u) {
        slot_s[static_cast<unsigned long long>(prow) * Shape::HIDDEN_MXFP8_GROUPS_ULL + g_blk] = sb;
    }
}

template <typename Shape>
__device__ __forceinline__ void v33_shared_gate_impl(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out,
    unsigned int rows
) {
    static_assert((Shape::HIDDEN % 256u) == 0u, "v33_shared_gate requires HIDDEN divisible by 256");
    constexpr unsigned SG_ITERS = Shape::HIDDEN / 256u;

    const unsigned r = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (r >= rows) return;
    if (shared_gate == nullptr) {
        if (tid == 0u) ss_out[r] = 1.0f;
        return;
    }
    __shared__ float s_dot[8];
    const __nv_bfloat16* const x_row = input + static_cast<unsigned long long>(r) * Shape::HIDDEN_ULL;
    float d = 0.0f;
    #pragma unroll
    for (unsigned i = 0u; i < SG_ITERS; ++i) {
        const unsigned c = tid + i * 256u;
        d += __bfloat162float(x_row[c]) * __bfloat162float(shared_gate[c]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        d += __shfl_down_sync(0xffffffffu, d, offset);
    }
    if ((tid & 31u) == 0u) s_dot[tid >> 5] = d;
    __syncthreads();
    if (tid == 0u) {
        float total = 0.0f;
        #pragma unroll
        for (unsigned w = 0u; w < 8u; ++w) total += s_dot[w];
        ss_out[r] = 1.0f / (1.0f + __expf(-total));
    }
}

}  // namespace
