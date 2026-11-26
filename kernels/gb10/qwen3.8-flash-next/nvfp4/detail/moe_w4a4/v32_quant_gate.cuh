// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// Single-read fused BF16 -> NVFP4 activation quantizer + shared_gate dot template.
// Included by `qwen4_moe_w4a4_v32.cu` after `v32_gemm_impl.cuh`.

namespace {

template <typename Shape>
__device__ __forceinline__ void v32_nvfp4_quant_rows_gate_impl(
    const __nv_bfloat16* __restrict__ src,
    unsigned char* __restrict__ xq,
    unsigned char* __restrict__ xs,
    float* __restrict__ x_scale2,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out,
    unsigned int rows
) {
    constexpr unsigned kHidden = Shape::HIDDEN;
    constexpr unsigned kHalfHidden = Shape::HIDDEN_HALF;
    constexpr unsigned kGroupsPerRow = Shape::HIDDEN_NVFP4_GROUPS;
    constexpr unsigned kBlendThreads = 256u;
    constexpr unsigned kGateTerms = kHidden / kBlendThreads;
    constexpr unsigned kPass0Elems = kBlendThreads * 8u;             // 2048
    constexpr unsigned kTailElems = kHidden - kPass0Elems;           // 512 for H=2560
    constexpr unsigned kTailThreads = kTailElems / 8u;               // 64 for H=2560
    constexpr unsigned kPass0ScaleWords = kPass0Elems / 64u;         // 32

    static_assert((kHidden % kBlendThreads) == 0u,
                  "v32_nvfp4_quant_rows_gate requires HIDDEN divisible by 256");
    static_assert(kHidden >= kPass0Elems && kHidden <= 2u * kPass0Elems && (kTailElems % 64u) == 0u,
                  "v32_nvfp4_quant_rows_gate supports 2-pass HIDDEN in [2048, 4096]");

    const unsigned bt = threadIdx.x;
    const unsigned warp = bt >> 5;
    const unsigned lane = bt & 31u;

    uint64_t pol_el = 0;
    uint64_t pol_ef = 0;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));

    const bool gate_aligned16 = (shared_gate != nullptr) &&
                                ((reinterpret_cast<uintptr_t>(shared_gate) & 15u) == 0u);
    auto load_bf16x8 = [](const __nv_bfloat16* p, bool aligned16) -> uint4 {
        if (aligned16) {
            return __ldg(reinterpret_cast<const uint4*>(p));
        }
        unsigned short h[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            h[i] = *reinterpret_cast<const unsigned short*>(p + i);
        }
        return make_uint4(
            static_cast<unsigned>(h[0]) | (static_cast<unsigned>(h[1]) << 16),
            static_cast<unsigned>(h[2]) | (static_cast<unsigned>(h[3]) << 16),
            static_cast<unsigned>(h[4]) | (static_cast<unsigned>(h[5]) << 16),
            static_cast<unsigned>(h[6]) | (static_cast<unsigned>(h[7]) << 16));
    };

    __shared__ __align__(16) float s_prod[kGateTerms][kBlendThreads];
    __shared__ float s_amax[8];
    __shared__ float s_dot[2][8];

    const unsigned prod_b0 = (lane * 8u) ^ (((lane >> 2) & 1u) << 2);
    const unsigned prod_b1 = prod_b0 ^ 4u;
    const unsigned swz_bt = bt ^ (((bt >> 5) & 1u) << 2);
    unsigned row_parity = 0u;
    const bool pin_xq = (rows <= V32_XQ_PIN_MAX_ROWS);

    auto stage_gate_prod = [&](unsigned pass, const float (&v)[8]) {
        const unsigned term_idx = pass * 8u + warp;
        const uint4 g_u4 = load_bf16x8(shared_gate + pass * kPass0Elems + bt * 8u, gate_aligned16);
        *reinterpret_cast<float4*>(&s_prod[term_idx][prod_b0]) = make_float4(
            v[0] * __uint_as_float(g_u4.x << 16),
            v[1] * __uint_as_float(g_u4.x & 0xffff0000u),
            v[2] * __uint_as_float(g_u4.y << 16),
            v[3] * __uint_as_float(g_u4.y & 0xffff0000u));
        *reinterpret_cast<float4*>(&s_prod[term_idx][prod_b1]) = make_float4(
            v[4] * __uint_as_float(g_u4.z << 16),
            v[5] * __uint_as_float(g_u4.z & 0xffff0000u),
            v[6] * __uint_as_float(g_u4.w << 16),
            v[7] * __uint_as_float(g_u4.w & 0xffff0000u));
    };

    for (unsigned r = blockIdx.x; r < rows; r += gridDim.x) {
        const __nv_bfloat16* x_row = src + static_cast<unsigned long long>(r) * kHidden;
        const uint4* xr4 = reinterpret_cast<const uint4*>(x_row);

        // Pass 0: all 256 threads load x_row[bt*8 .. +7] (128 NVFP4 groups).
        uint4 w0;
        asm volatile("ld.global.ca.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                     : "=r"(w0.x), "=r"(w0.y), "=r"(w0.z), "=r"(w0.w)
                     : "l"(xr4 + bt), "l"(pol_ef));
        const unsigned u0[4] = {w0.x, w0.y, w0.z, w0.w};
        float v0[8];
        float gmax0 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float a = __uint_as_float(u0[i] << 16);
            const float b = __uint_as_float(u0[i] & 0xffff0000u);
            v0[2 * i] = a;
            v0[2 * i + 1] = b;
            gmax0 = fmaxf(gmax0, fmaxf(fabsf(a), fabsf(b)));
        }
        if (shared_gate != nullptr && ss_out != nullptr) {
            stage_gate_prod(0u, v0);
        }
        gmax0 = fmaxf(gmax0, __shfl_xor_sync(0xffffffffu, gmax0, 1));

        // Pass 1: threads 0..kTailThreads-1 load remaining tail groups.
        float v1[8];
        float gmax1 = 0.0f;
        if (bt < kTailThreads) {
            uint4 w1;
            asm volatile("ld.global.ca.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                         : "=r"(w1.x), "=r"(w1.y), "=r"(w1.z), "=r"(w1.w)
                         : "l"(xr4 + kBlendThreads + bt), "l"(pol_ef));
            const unsigned u1[4] = {w1.x, w1.y, w1.z, w1.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float a = __uint_as_float(u1[i] << 16);
                const float b = __uint_as_float(u1[i] & 0xffff0000u);
                v1[2 * i] = a;
                v1[2 * i + 1] = b;
                gmax1 = fmaxf(gmax1, fmaxf(fabsf(a), fabsf(b)));
            }
            if (shared_gate != nullptr && ss_out != nullptr) {
                stage_gate_prod(1u, v1);
            }
            gmax1 = fmaxf(gmax1, __shfl_xor_sync(0xffffffffu, gmax1, 1));
        }

        // Single CTA-wide reduction for per-row absmax (also publishes s_prod!).
        float wmax = fmaxf(gmax0, gmax1);
        #pragma unroll
        for (int d = 16; d >= 2; d >>= 1) {
            wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffffu, wmax, d));
        }
        if (lane == 0u) s_amax[warp] = wmax;
        __syncthreads();

        float row_amax = 0.0f;
        #pragma unroll
        for (unsigned w = 0u; w < 8u; ++w) {
            row_amax = fmaxf(row_amax, s_amax[w]);
        }
        const bool positive = isfinite(row_amax) && (row_amax > 1e-30f);
        const float row_scale2 = positive ? __fdiv_rn(row_amax, 2688.0f) : 1.0f;
        const float inv_s2_over_6 = positive ? __fdiv_rn(448.0f, row_amax) : 0.0f;
        if (bt == 0u) {
            x_scale2[r] = row_scale2;
        }

        unsigned char* const xq_row = xq + static_cast<unsigned long long>(r) * kHalfHidden;
        unsigned char* const xs_row = xs + static_cast<unsigned long long>(r) * kGroupsPerRow;

        // Quantize Pass 0 groups (0..127) directly from registers.
        {
            const float sf_f = (gmax0 > 0.0f) ? fminf(__fmul_rn(gmax0, inv_s2_over_6), 448.0f) : 0.0f;
            const unsigned sf_b = qwen4_nvfp4_detail::rn_ue4m3(sf_f);
            const float eff = __fmul_rn(qwen4_nvfp4_detail::e4m3_to_float(sf_b), row_scale2);
            const float inv_eff = (eff > 0.0f) ? __fdiv_rn(1.0f, eff) : 0.0f;
            const unsigned packed0 = qwen4_nvfp4_detail::cvt_f32x8_to_e2m1x8(v0, inv_eff);
            unsigned* dst0 = reinterpret_cast<unsigned*>(xq_row) + bt;
            if (pin_xq) v32_st4_l2_el(dst0, packed0, pol_el);
            else        *dst0 = packed0;

            const unsigned s1 = __shfl_down_sync(0xffffffffu, sf_b, 2);
            const unsigned s2 = __shfl_down_sync(0xffffffffu, sf_b, 4);
            const unsigned s3 = __shfl_down_sync(0xffffffffu, sf_b, 6);
            if ((lane & 7u) == 0u) {
                const unsigned sf4 = (sf_b & 0xffu) | ((s1 & 0xffu) << 8) |
                                     ((s2 & 0xffu) << 16) | ((s3 & 0xffu) << 24);
                unsigned* sdst0 = reinterpret_cast<unsigned*>(xs_row) + (bt >> 3);
                if (pin_xq) v32_st4_l2_el(sdst0, sf4, pol_el);
                else        *sdst0 = sf4;
            }
        }

        // Quantize Pass 1 tail groups directly from registers.
        if (bt < kTailThreads) {
            const float sf_f = (gmax1 > 0.0f) ? fminf(__fmul_rn(gmax1, inv_s2_over_6), 448.0f) : 0.0f;
            const unsigned sf_b = qwen4_nvfp4_detail::rn_ue4m3(sf_f);
            const float eff = __fmul_rn(qwen4_nvfp4_detail::e4m3_to_float(sf_b), row_scale2);
            const float inv_eff = (eff > 0.0f) ? __fdiv_rn(1.0f, eff) : 0.0f;
            const unsigned packed1 = qwen4_nvfp4_detail::cvt_f32x8_to_e2m1x8(v1, inv_eff);
            unsigned* dst1 = reinterpret_cast<unsigned*>(xq_row) + kBlendThreads + bt;
            if (pin_xq) v32_st4_l2_el(dst1, packed1, pol_el);
            else        *dst1 = packed1;

            const unsigned s1 = __shfl_down_sync(0xffffffffu, sf_b, 2);
            const unsigned s2 = __shfl_down_sync(0xffffffffu, sf_b, 4);
            const unsigned s3 = __shfl_down_sync(0xffffffffu, sf_b, 6);
            if ((lane & 7u) == 0u) {
                const unsigned sf4 = (sf_b & 0xffu) | ((s1 & 0xffu) << 8) |
                                     ((s2 & 0xffu) << 16) | ((s3 & 0xffu) << 24);
                unsigned* sdst1 = reinterpret_cast<unsigned*>(xs_row) + kPass0ScaleWords + (bt >> 3);
                if (pin_xq) v32_st4_l2_el(sdst1, sf4, pol_el);
                else        *sdst1 = sf4;
            }
        }

        // Finish exact 0-ULP shared-gate dot product from staged s_prod.
        if (ss_out != nullptr && shared_gate != nullptr) {
            float d = 0.0f;
            #pragma unroll
            for (unsigned i = 0u; i < kGateTerms; ++i) {
                d += s_prod[i][swz_bt];
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                d += __shfl_down_sync(0xffffffffu, d, offset);
            }
            if (lane == 0u) s_dot[row_parity][warp] = d;
            __syncthreads();
            if (bt == 0u) {
                float total = 0.0f;
                #pragma unroll
                for (unsigned w = 0u; w < 8u; ++w) total += s_dot[row_parity][w];
                ss_out[r] = 1.0f / (1.0f + __expf(-total));
            }
            row_parity ^= 1u;
        } else {
            if (ss_out != nullptr && bt == 0u) {
                ss_out[r] = 1.0f;
            }
            if (r + gridDim.x < rows) {
                __syncthreads();
            }
        }
    }
}

}  // namespace
