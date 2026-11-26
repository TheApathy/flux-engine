// SPDX-License-Identifier: AGPL-3.0-only
//
// V28-V31: W4A8 MoE kernel family built on V16 (ATLAS_QWEN4_PREFILL_MOE_W4A8_WIDE=7).
//
// Modularized manifest: includes V16 as the arithmetic/layout SSOT, compile-time
// model shape traits in `qwen4_geometry.cuh`, and focused kernel families under
// `detail/moe_compact/`.

#include "qwen4_moe_compact_t_k32_v16.cu"
#include "qwen4_geometry.cuh"

#include "detail/moe_compact/v28_common.cuh"
#include "detail/moe_compact/v28_base_gemm.cuh"

// The V16 MXFP8 row quantiser under the V28 tag (same code; the runtime loads quant + GEMMs by tag).
extern "C" __global__ void qwen4_mxfp8_quant_rows_v28(
    const __nv_bfloat16* __restrict__ src, unsigned char* __restrict__ q, unsigned char* __restrict__ s,
    unsigned int rows, unsigned int cols
) {
    const unsigned long long blocks = static_cast<unsigned long long>(rows) * (cols / 32);
    for (unsigned long long b = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
         b < blocks; b += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
        const uint4* x = reinterpret_cast<const uint4*>(src + b * 32);
        float v[32];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint4 w = x[c];
            const unsigned u[4] = {w.x, w.y, w.z, w.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[8 * c + 2 * i] = __uint_as_float(u[i] << 16);
                v[8 * c + 2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
            }
        }
        unsigned qq[8], ss;
        v10_mxfp8_block(v, qq, ss);
        uint4* out = reinterpret_cast<uint4*>(q + b * 32);
        out[0] = make_uint4(qq[0], qq[1], qq[2], qq[3]);
        out[1] = make_uint4(qq[4], qq[5], qq[6], qq[7]);
        s[b] = static_cast<unsigned char>(ss);
    }
}

#include "detail/moe_compact/v28_shared.cuh"
#include "detail/moe_compact/v28_q4.cuh"
#include "detail/moe_compact/v28_twarp.cuh"
#include "detail/moe_compact/v28_m32.cuh"
#include "detail/moe_compact/v28_plan.cuh"
#include "detail/moe_compact/v29_m256.cuh"
#include "detail/moe_compact/v28_m16.cuh"
#include "detail/moe_compact/v30_2cta.cuh"
#include "detail/moe_compact/v31_ws.cuh"
