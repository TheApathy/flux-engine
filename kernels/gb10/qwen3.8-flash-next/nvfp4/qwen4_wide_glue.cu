// SPDX-License-Identifier: AGPL-3.0-only
//
// Small glue kernels for Flash-Next wide-chunk prefill (fn-wide lane).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// GDN gates from an FP32 [rows, N=2*nv] BA projection produced by one
// cuBLASLt GEMM (`ATLAS_QWEN4_PREFILL_BA_GEMM=1`), replacing the per-row GEMV
// `dense_gemv_ba_gates_batchn` (which re-reads the BA weight once per row).
// Same group layout and transforms as that kernel: per group of 2*vpg
// columns the first vpg are beta logits, the next vpg the alpha logits;
//   beta = sigmoid(b),  gate = exp(-exp(min(A_log, 20)) * softplus(min(a + dt_bias, 20))).
// Output rows [gate(nv) | beta(nv)] at `out_stride` floats per row.
// Grid: (rows), Block: (N) with N <= 1024.
extern "C" __global__ void qwen4_ba_gates_from_f32(
    const float* __restrict__ ba,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ out,
    unsigned int n,
    unsigned int vheads_per_group,
    unsigned int out_stride) {
    const unsigned int row = blockIdx.x;
    const unsigned int c = threadIdx.x;
    if (c >= n) return;
    const unsigned int nv = n / 2;
    const unsigned int group_dim = 2 * vheads_per_group;
    const unsigned int group = c / group_dim;
    const unsigned int within = c % group_dim;
    const float x = ba[(unsigned long long)row * n + c];
    float* o = out + (unsigned long long)row * out_stride;
    if (within < vheads_per_group) {
        const unsigned int vh = group * vheads_per_group + within;
        o[nv + vh] = 1.0f / (1.0f + __expf(-x));
    } else {
        const unsigned int vh = group * vheads_per_group + (within - vheads_per_group);
        const float a = __expf(fminf(A_log[vh], 20.0f));
        const float dt = __logf(1.0f + __expf(fminf(x + dt_bias[vh], 20.0f)));
        o[vh] = __expf(-a * dt);
    }
}

namespace {

__device__ __forceinline__ float ld_global_f32_evict_first_wide(const float* p) {
    float v;
    asm volatile(
        "{\n\t"
        ".reg .b64 policy;\n\t"
        "createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n\t"
        "ld.global.L2::cache_hint.f32 %0, [%1], policy;\n\t"
        "}\n"
        : "=f"(v)
        : "l"(p));
    return v;
}

__device__ __forceinline__ void st_global_f32_evict_last_wide(float* p, float v) {
    asm volatile(
        "{\n\t"
        ".reg .b64 policy;\n\t"
        "createpolicy.fractional.L2::evict_last.b64 policy, 1.0;\n\t"
        "st.global.L2::cache_hint.f32 [%0], %1, policy;\n\t"
        "}\n"
        :
        : "l"(p), "f"(v)
        : "memory");
}

}  // namespace

// Output-head-ordered, L2-hinted v2 twin of `qwen4_ba_gates_from_f32`:
// threads c < nv compute gate[vh] (vh = c) and threads c >= nv compute beta[vh] (vh = c - nv),
// eliminating 3-lane interleaved branch divergence and coalescing row stores.
extern "C" __global__ void qwen4_ba_gates_from_f32_v2(
    const float* __restrict__ ba,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ out,
    unsigned int n,
    unsigned int vheads_per_group,
    unsigned int out_stride) {
    const unsigned int row = blockIdx.x;
    const unsigned int c = threadIdx.x;
    if (c >= n) return;
    const unsigned int nv = n / 2;
    const float* ba_row = ba + (unsigned long long)row * n;
    float val;
    if (c < nv) {
        const unsigned int vh = c;
        const unsigned int group = vh / vheads_per_group;
        const unsigned int ba_col = vh + (group + 1u) * vheads_per_group;
        const float x = ld_global_f32_evict_first_wide(ba_row + ba_col);
        const float a = __expf(fminf(A_log[vh], 20.0f));
        const float dt = __logf(1.0f + __expf(fminf(x + dt_bias[vh], 20.0f)));
        val = __expf(-a * dt);
    } else {
        const unsigned int vh = c - nv;
        const unsigned int group = vh / vheads_per_group;
        const unsigned int ba_col = vh + group * vheads_per_group;
        const float x = ld_global_f32_evict_first_wide(ba_row + ba_col);
        val = 1.0f / (1.0f + __expf(-x));
    }
    st_global_f32_evict_last_wide(out + (unsigned long long)row * out_stride + c, val);
}

// Planar twin of `qwen4_ba_gates_from_f32` for the FlashInfer GDN bridge:
// writes contiguous `gate[num_rows * nv]` at `out` followed by contiguous
// `beta[num_rows * nv]` at `out + num_rows * nv`, eliminating `k_deinterleave_gate_beta`.
// Uses output-head thread order (c < nv -> gate[vh], c >= nv -> beta[vh]) for
// coalesced planar writes and L2::evict_first / L2::evict_last cache hints.
extern "C" __global__ void qwen4_ba_gates_from_f32_planar(
    const float* __restrict__ ba,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ out,
    unsigned int n,
    unsigned int vheads_per_group,
    unsigned int num_rows) {
    const unsigned int row = blockIdx.x;
    const unsigned int c = threadIdx.x;
    if (c >= n || row >= num_rows) return;
    const unsigned int nv = n / 2;
    const float* ba_row = ba + (unsigned long long)row * n;
    float val;
    unsigned long long dst_idx;
    if (c < nv) {
        const unsigned int vh = c;
        const unsigned int group = vh / vheads_per_group;
        const unsigned int ba_col = vh + (group + 1u) * vheads_per_group;
        const float x = ld_global_f32_evict_first_wide(ba_row + ba_col);
        const float a = __expf(fminf(A_log[vh], 20.0f));
        const float dt = __logf(1.0f + __expf(fminf(x + dt_bias[vh], 20.0f)));
        val = __expf(-a * dt);
        dst_idx = (unsigned long long)row * nv + vh;
    } else {
        const unsigned int vh = c - nv;
        const unsigned int group = vh / vheads_per_group;
        const unsigned int ba_col = vh + group * vheads_per_group;
        const float x = ld_global_f32_evict_first_wide(ba_row + ba_col);
        val = 1.0f / (1.0f + __expf(-x));
        dst_idx = ((unsigned long long)num_rows + row) * nv + vh;
    }
    st_global_f32_evict_last_wide(out + dst_idx, val);
}

// Dequantize a TRANSPOSED NVFP4 weight (`QuantizedWeight::transpose_for_gemm`:
// packed [K/2][N] with the even k in the low nibble, e4m3 scales [K/16][N],
// N a multiple of 64 so there is no padding) into row-major BF16 [K][N] with
// the per-tensor scale2 folded in: bf16(LUT[nib] * fp8 * scale2), the operand
// the W4A16 tensor-core GEMMs form. One thread per packed byte.
// Grid: ceil(K/2 * N / 256), Block 256.
__device__ __constant__ float QWEN4_WIDE_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

extern "C" __global__ void qwen4_dequant_t_bf16(
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales,
    float scale2,
    __nv_bfloat16* __restrict__ out,
    unsigned int N,
    unsigned int K) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)(K / 2u) * N;
    if (i >= total) return;
    const unsigned long long r = i / N;
    const unsigned long long n = i - r * N;
    const unsigned int k = (unsigned int)(2ull * r);
    const unsigned char byte = packed[i];
    __nv_fp8_e4m3 fp8;
    *reinterpret_cast<unsigned char*>(&fp8) = scales[(unsigned long long)(k / 16u) * N + n];
    const float s = (float)fp8 * scale2;
    out[(unsigned long long)k * N + n] = __float2bfloat16(QWEN4_WIDE_E2M1_LUT[byte & 15] * s);
    out[(unsigned long long)(k + 1) * N + n] = __float2bfloat16(QWEN4_WIDE_E2M1_LUT[byte >> 4] * s);
}
