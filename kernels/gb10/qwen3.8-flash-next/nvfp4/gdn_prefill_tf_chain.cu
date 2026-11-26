// SPDX-License-Identifier: AGPL-3.0-only
//
// FP32 token-serial GDN prefill recurrence after TensorFold's `chain_kernel`
// (TensorFold 0.6.6 src/tensorfold/cuda/kernels/gdn_prefill.cu,
// https://github.com/ashhart/TensorFold, Copyright 2026 TensorFold
// contributors, Apache-2.0), adapted to the Flash-Next exact prefill route
// (ATLAS_QWEN4_PREFILL_GDN_TF=1):
//   - q/k/v are the exact route's FP32 conv rows (token strides qk_stride /
//     v_stride, key head kh = vh / (HV/HK)); TensorFold reads BF16 v;
//   - the output is FP32 (token stride output_stride), not BF16, so the
//     FP32-input gated norm downstream is unchanged;
//   - the H state is the engine's FP32 [HV][K][V] (TensorFold: [HV][V][K]),
//     read once at the start and written once at the end, in place;
//   - the decay is clamped to [0, 1] like the exact kernels.
// Same math as gated_delta_rule_prefill_f32_sequence_regfinal* (S' = g S;
// d = (v - S'^T k) beta; S = S' + k d; o = S^T q / sqrt(K)); only the FP32
// association differs (64-term half-row FMA chains split 4 ways + one
// shuffle instead of one 128-term chain), so it is exact-class, not bit-exact.
//
// Two threads own one value row (64 FP32 state values each, in registers);
// a 24-token stage of k/q/v/g/beta is prefetched into registers while the
// previous stage computes, then staged through shared memory (37,056 B <= 48 KiB
// static SMEM cap for ROWS=128).
// Grid: (HV * (128 / ROWS), 1, 1)   Block: (2 * ROWS, 1, 1)
#include <cuda_runtime.h>

namespace {

constexpr int TF_DK = 128;
constexpr int TF_DV = 128;
constexpr int TF_STEPS = 24;

template <int ROWS>
__device__ __forceinline__ void qwen4_gdn_tf_chain_body(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    unsigned int num_tokens,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gate_beta_stride,
    unsigned int output_stride
) {
    constexpr int THREADS = 2 * ROWS;
    constexpr int SPLIT = TF_DV / ROWS;
    constexpr int KQ = TF_STEPS * (TF_DK / 4) / THREADS;
    constexpr int VV = TF_STEPS * (ROWS / 4) / THREADS;
    static_assert(KQ * THREADS == TF_STEPS * (TF_DK / 4), "k/q slots per thread");
    static_assert(VV * THREADS == TF_STEPS * (ROWS / 4), "v slots per thread");
    __shared__ float4 ks[TF_STEPS][TF_DK / 4];
    __shared__ float4 qs[TF_STEPS][TF_DK / 4];
    __shared__ __align__(16) float vs[TF_STEPS][ROWS];
    __shared__ float gs[TF_STEPS];
    __shared__ float bs[TF_STEPS];

    const int head = blockIdx.x / SPLIT;
    const int col0 = (blockIdx.x % SPLIT) * ROWS;
    if (head >= (int)num_v_heads) return;
    const int local = threadIdx.x >> 1;
    const int half = threadIdx.x & 1;
    const int row = col0 + local;
    const int key_head = head / ((int)num_v_heads / (int)num_k_heads);
    const int W = (int)num_tokens;

    // s[4j+e] = H[k = 8j + 4*half + e][row]
    float* H = h_state + (unsigned long long)head * TF_DK * TF_DV;
    float s[64];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e)
            s[4 * j + e] = H[(unsigned long long)(8 * j + 4 * half + e) * TF_DV + row];
    }

    float4 kr[KQ], qr[KQ], vr[VV];
    float gr = 0.0f, br = 0.0f;
    auto fetch = [&](int t0) {
#pragma unroll
        for (int it = 0; it < KQ; ++it) {
            const int i = threadIdx.x + it * THREADS;
            const int st = i / (TF_DK / 4), c = i % (TF_DK / 4);
            const unsigned long long t = (unsigned long long)min(t0 + st, W - 1);
            const unsigned long long at = t * qk_stride + key_head * TF_DK + 4 * c;
            kr[it] = *reinterpret_cast<const float4*>(key + at);
            qr[it] = *reinterpret_cast<const float4*>(query + at);
        }
#pragma unroll
        for (int it = 0; it < VV; ++it) {
            const int i = threadIdx.x + it * THREADS;
            const int st = i / (ROWS / 4), c = i % (ROWS / 4);
            const unsigned long long t = (unsigned long long)min(t0 + st, W - 1);
            vr[it] = *reinterpret_cast<const float4*>(
                value + t * v_stride + head * TF_DV + col0 + 4 * c);
        }
        if (threadIdx.x < TF_STEPS) {
            const unsigned long long t =
                (unsigned long long)min(t0 + (int)threadIdx.x, W - 1);
            gr = fminf(fmaxf(gate[t * gate_beta_stride + head], 0.0f), 1.0f);
            br = beta[t * gate_beta_stride + head];
        }
    };
    auto store = [&]() {
#pragma unroll
        for (int it = 0; it < KQ; ++it) {
            const int i = threadIdx.x + it * THREADS;
            ks[i / (TF_DK / 4)][i % (TF_DK / 4)] = kr[it];
            qs[i / (TF_DK / 4)][i % (TF_DK / 4)] = qr[it];
        }
#pragma unroll
        for (int it = 0; it < VV; ++it) {
            const int i = threadIdx.x + it * THREADS;
            *reinterpret_cast<float4*>(&vs[i / (ROWS / 4)][4 * (i % (ROWS / 4))]) = vr[it];
        }
        if (threadIdx.x < TF_STEPS) {
            gs[threadIdx.x] = gr;
            bs[threadIdx.x] = br;
        }
    };

    const float inv_sqrt_d = rsqrtf((float)TF_DK);
    if (W > 0) fetch(0);
    for (int t0 = 0; t0 < W; t0 += TF_STEPS) {
        const int n = min(TF_STEPS, W - t0);
        __syncthreads();
        store();
        __syncthreads();
        if (t0 + TF_STEPS < W) fetch(t0 + TF_STEPS);
        for (int tt = 0; tt < n; ++tt) {
            const float gt = gs[tt], bt = bs[tt];
            const float vt = vs[tt][local];
            float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float4 kk = ks[tt][2 * j + half];
                s[4 * j] = s[4 * j] * gt;
                s[4 * j + 1] = s[4 * j + 1] * gt;
                s[4 * j + 2] = s[4 * j + 2] * gt;
                s[4 * j + 3] = s[4 * j + 3] * gt;
                m[0] = __fmaf_rn(s[4 * j], kk.x, m[0]);
                m[1] = __fmaf_rn(s[4 * j + 1], kk.y, m[1]);
                m[2] = __fmaf_rn(s[4 * j + 2], kk.z, m[2]);
                m[3] = __fmaf_rn(s[4 * j + 3], kk.w, m[3]);
            }
            float mem = (m[0] + m[1]) + (m[2] + m[3]);
            mem = mem + __shfl_xor_sync(0xffffffffu, mem, 1);
            const float delta = (vt - mem) * bt;
            float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float4 kk = ks[tt][2 * j + half], qq = qs[tt][2 * j + half];
                s[4 * j] = __fmaf_rn(kk.x, delta, s[4 * j]);
                s[4 * j + 1] = __fmaf_rn(kk.y, delta, s[4 * j + 1]);
                s[4 * j + 2] = __fmaf_rn(kk.z, delta, s[4 * j + 2]);
                s[4 * j + 3] = __fmaf_rn(kk.w, delta, s[4 * j + 3]);
                o[0] = __fmaf_rn(s[4 * j], qq.x, o[0]);
                o[1] = __fmaf_rn(s[4 * j + 1], qq.y, o[1]);
                o[2] = __fmaf_rn(s[4 * j + 2], qq.z, o[2]);
                o[3] = __fmaf_rn(s[4 * j + 3], qq.w, o[3]);
            }
            float out = (o[0] + o[1]) + (o[2] + o[3]);
            out = out + __shfl_xor_sync(0xffffffffu, out, 1);
            if (half == 0)
                output[(unsigned long long)(t0 + tt) * output_stride + head * TF_DV + row] =
                    out * inv_sqrt_d;
        }
    }

#pragma unroll
    for (int j = 0; j < 16; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e)
            H[(unsigned long long)(8 * j + 4 * half + e) * TF_DV + row] = s[4 * j + e];
    }
}

}  // namespace

#define QWEN4_GDN_TF_ENTRY(NAME, ROWS)                                                   \
    extern "C" __global__ void __launch_bounds__(2 * ROWS) NAME(                         \
        float* __restrict__ h_state, const float* __restrict__ query,                    \
        const float* __restrict__ key, const float* __restrict__ value,                  \
        const float* __restrict__ gate, const float* __restrict__ beta,                  \
        float* __restrict__ output, unsigned int num_tokens, unsigned int num_k_heads,   \
        unsigned int num_v_heads, unsigned int qk_stride, unsigned int v_stride,         \
        unsigned int gate_beta_stride, unsigned int output_stride) {                     \
        qwen4_gdn_tf_chain_body<ROWS>(h_state, query, key, value, gate, beta, output,    \
                                      num_tokens, num_k_heads, num_v_heads, qk_stride,  \
                                      v_stride, gate_beta_stride, output_stride);       \
    }

// 128 rows / 256 threads per CTA, one CTA per value head (TensorFold's choice
// when HV >= SMs); 64 rows / 128 threads, two CTAs per head.
QWEN4_GDN_TF_ENTRY(qwen4_gdn_prefill_tf_chain_r128, 128)
QWEN4_GDN_TF_ENTRY(qwen4_gdn_prefill_tf_chain_r64, 64)
