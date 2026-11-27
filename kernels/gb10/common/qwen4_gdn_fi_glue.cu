// SPDX-License-Identifier: AGPL-3.0-only
//
// Glue for the FlashInfer SM121a chunked GDN route on Flash-Next prefill
// (`ATLAS_QWEN4_PREFILL_GDN_FI=1`). The FI kernel consumes BF16 Q/K/V packed
// as one [T, conv_dim] buffer and writes BF16 [T, HV*D]; Flash-Next's exact
// path keeps both in FP32 rows of stride QKVZ. These are the two BF16 twins of
// the exact kernels at the boundary (same arithmetic, BF16 store / load).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// The FI route's 16-bit format: BF16 export (ATLAS_QWEN4_PREFILL_GDN_FI=1) or
// FP16 export (=2). Stores round to nearest; loads widen exactly.
__device__ __forceinline__ void qwen4_fi_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
__device__ __forceinline__ void qwen4_fi_store(__half* p, float v) { *p = __float2half_rn(v); }
__device__ __forceinline__ float qwen4_fi_load(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float qwen4_fi_load(__half v) { return __half2float(v); }

// 0-ULP fast-math 1.0f / (1.0f + __expf(-x)) without slowpath subroutine call (`CALL.REL.NOINC`),
// matching build_target.rs `--use_fast_math --fmad=false` bit-for-bit across v1, v2, and v3.
__device__ __forceinline__ float qwen4_fast_rcp_rn_1p_exp(float x) {
    float y;
    asm("{\n\t"
        ".reg .f32 z, t, den;\n\t"
        "mul.rn.ftz.f32 z, %1, 0fbfb8aa3b;\n\t"
        "ex2.approx.ftz.f32 t, z;\n\t"
        "add.rn.ftz.f32 den, t, 0f3f800000;\n\t"
        "rcp.approx.ftz.f32 %0, den;\n\t"
        "}"
        : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float qwen4_fi_div_rn_1p_exp(float num, float g) {
    float q;
    asm("{\n\t"
        ".reg .f32 z, t, den;\n\t"
        "mul.rn.ftz.f32 z, %2, 0fbfb8aa3b;\n\t"
        "ex2.approx.ftz.f32 t, z;\n\t"
        "add.rn.ftz.f32 den, t, 0f3f800000;\n\t"
        "div.approx.ftz.f32 %0, %1, den;\n\t"
        "}"
        : "=f"(q) : "f"(num), "f"(g));
    return q;
}

__device__ __forceinline__ float qwen4_fi_div_approx(float a, float b) {
    float q;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(q) : "f"(a), "f"(b));
    return q;
}

// BF16-output twin of `causal_conv1d_update_l2norm_f32_prefill_parallel`
// (common/causal_conv1d_prefill_parallel.cu): identical conv, SiLU and per-head
// L2, then a BF16 store into a [T, output_stride] buffer.
// Grid: (ceil(dim/256), num_tokens)  Block: (256, 1, 1)
template <typename OutT, bool GATESIGHT = false>
__device__ __forceinline__ void qwen4_conv1d_l2norm_16_body(
    const float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    OutT* __restrict__ output,
    unsigned int num_tokens,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    unsigned int input_stride,
    unsigned int output_stride,
    __half* __restrict__ gatesight_out = nullptr,
    unsigned int num_v_heads = 0u
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int token = blockIdx.y;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = block_start < qk_channels;
    const bool valid = ch < dim && token < num_tokens;
    __shared__ float warp_sums[8];

    float silu = 0.0f;
    if (valid) {
        const __nv_bfloat16* w = weight + (unsigned long long)ch * d_conv;
        float acc = (bias != nullptr) ? bias[ch] : 0.0f;
        for (unsigned int k = 0; k < d_conv; k++) {
            const int idx = (int)token - (int)(d_conv - 1) + (int)k;
            const float s = (idx >= 0)
                ? (float)new_input[(unsigned long long)idx * input_stride + ch]
                : conv_state[(unsigned long long)ch * d_conv + (k + token + 1)];
            acc += s * (float)w[k];
        }
        float sigmoid_acc = qwen4_fast_rcp_rn_1p_exp(acc);
        silu = acc * sigmoid_acc;
    }

    if (block_needs_l2) {
        float sq = valid ? silu * silu : 0.0f;
        const unsigned int warp_id = tid / 32;
        const unsigned int lane = tid % 32;
        for (int offset = 16; offset >= 1; offset >>= 1)
            sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
        if (lane == 0) warp_sums[warp_id] = sq;
        __syncthreads();
        const unsigned int head_in_block = tid / head_dim;
        const unsigned int base_warp = head_in_block * (head_dim / 32);
        if (tid == 0 || tid == head_dim) {
            float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                        + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
            warp_sums[base_warp] = rsqrtf(total + l2_eps);
        }
        __syncthreads();
        if (valid) silu *= warp_sums[base_warp];
    } else if constexpr (GATESIGHT) {
        if (gatesight_out != nullptr) {
            float sq = valid ? silu * silu : 0.0f;
            const unsigned int warp_id = tid / 32;
            const unsigned int lane = tid % 32;
            for (int offset = 16; offset >= 1; offset >>= 1)
                sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
            if (lane == 0) warp_sums[warp_id] = sq;
            __syncthreads();
            const unsigned int head_in_block = tid / head_dim;
            const unsigned int base_warp = head_in_block * (head_dim / 32);
            if ((tid == 0 || tid == head_dim) && valid) {
                float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                            + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
                const unsigned int vh = (ch - qk_channels) / head_dim;
                if (vh < num_v_heads) {
                    gatesight_out[((unsigned long long)token * num_v_heads + vh) * 3ull + 2ull] =
                        __float2half_rn(sqrtf(total));
                }
            }
        }
    }

    if (valid) {
        qwen4_fi_store(&output[(unsigned long long)token * output_stride + ch], silu);
    }
}

#define QWEN4_FI_CONV_ENTRY(NAME, OUT_T)                                              \
    extern "C" __global__ void NAME(                                                  \
        const float* __restrict__ conv_state,                                         \
        const __nv_bfloat16* __restrict__ new_input,                                  \
        const __nv_bfloat16* __restrict__ weight, const float* __restrict__ bias,     \
        OUT_T* __restrict__ output, unsigned int num_tokens, unsigned int dim,        \
        unsigned int d_conv, unsigned int qk_channels, unsigned int head_dim,         \
        float l2_eps, unsigned int input_stride, unsigned int output_stride) {        \
        qwen4_conv1d_l2norm_16_body<OUT_T, false>(conv_state, new_input, weight,      \
                                                  bias, output, num_tokens, dim,      \
                                                  d_conv, qk_channels, head_dim,      \
                                                  l2_eps, input_stride,               \
                                                  output_stride, nullptr, 0u);        \
    }

#define QWEN4_FI_CONV_GATESIGHT_ENTRY(NAME, OUT_T)                                    \
    extern "C" __global__ void NAME(                                                  \
        const float* __restrict__ conv_state,                                         \
        const __nv_bfloat16* __restrict__ new_input,                                  \
        const __nv_bfloat16* __restrict__ weight, const float* __restrict__ bias,     \
        OUT_T* __restrict__ output, unsigned int num_tokens, unsigned int dim,        \
        unsigned int d_conv, unsigned int qk_channels, unsigned int head_dim,         \
        float l2_eps, unsigned int input_stride, unsigned int output_stride,          \
        __half* __restrict__ gatesight_out, unsigned int num_v_heads) {               \
        qwen4_conv1d_l2norm_16_body<OUT_T, true>(conv_state, new_input, weight, bias, \
                                                 output, num_tokens, dim, d_conv,     \
                                                 qk_channels, head_dim, l2_eps,       \
                                                 input_stride, output_stride,         \
                                                 gatesight_out, num_v_heads);         \
    }

QWEN4_FI_CONV_ENTRY(qwen4_conv1d_l2norm_bf16_prefill_parallel, __nv_bfloat16)
QWEN4_FI_CONV_ENTRY(qwen4_conv1d_l2norm_f16_prefill_parallel, __half)
QWEN4_FI_CONV_GATESIGHT_ENTRY(qwen4_conv1d_l2norm_bf16_prefill_parallel_gatesight, __nv_bfloat16)
QWEN4_FI_CONV_GATESIGHT_ENTRY(qwen4_conv1d_l2norm_f16_prefill_parallel_gatesight, __half)

// ---------------------------------------------------------------------------
// v2 of the conv/SiLU/L2 kernel above (ATLAS_QWEN4_PREFILL_GDN_CONV_V2=1): byte-identical output, fewer
// and wider memory operations. v1 runs one 256-thread block per (token, 256 channels) with 2-byte loads
// (four taps re-read per row) and two barriers; v2 gives each lane 8 adjacent channels (16-byte loads
// and stores) and a strip of TOKENS tokens, carrying the causal window x[t-3..t-1] in registers, so
// each input row is read once per strip (plus a 3-row halo) and no barrier runs.
// Per element the arithmetic is v1's: the same taps in the same order (bias-or-zero, then
// += x[t-3+k] * w[k] for k = 0..3, unfused under --fmad=false), the same SiLU expression, the same
// rounding at the store. The per-head L2 sum keeps v1's tree: v1 thread (head warp w', lane l') holds
// channel 32 w' + l' of the head and reduces shfl_down 16, 8, 4, 2, 1, then ((S0 + S1) + S2) + S3.
// In v2, lane p holds channels 8 (p % 16) + c of head p / 16, i.e. w' = (p % 16) / 4, l' = 8 (p % 4) + c,
// so the same tree is: lanes p, p + 2 (step 16), lanes p, p + 1 (step 8), then registers c with
// c + 4, c + 2, c + 1; S_w' lands in lane 16 (p / 16) + 4 w'.
// Grid: (dim / 256, ceil(num_tokens / (8 * TOKENS)))  Block: 256 (warp = 256 channels x TOKENS tokens).
// Host-checked: dim % 256 == 0, qk_channels % 256 == 0, head_dim == 128, d_conv == 4, input/output
// rows and weight 16-byte aligned. dim/d_conv/head_dim are taken for signature parity with v1.
template <typename OutT>
__device__ __forceinline__ unsigned int qwen4_fi_pack2(float a, float b);
template <>
__device__ __forceinline__ unsigned int qwen4_fi_pack2<__half>(float a, float b) {
    return (unsigned int)__half_as_ushort(__float2half_rn(a)) |
           ((unsigned int)__half_as_ushort(__float2half_rn(b)) << 16);
}
template <>
__device__ __forceinline__ unsigned int qwen4_fi_pack2<__nv_bfloat16>(float a, float b) {
    return (unsigned int)__bfloat16_as_ushort(__float2bfloat16(a)) |
           ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(b)) << 16);
}

// Eight bf16 -> f32 (exact widening, as v1's (float) cast).
template <bool STREAMING = false>
__device__ __forceinline__ void qwen4_conv_v2_load_raw(const __nv_bfloat16* __restrict__ p, uint4& w) {
    if constexpr (STREAMING) {
        asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(w.x), "=r"(w.y), "=r"(w.z), "=r"(w.w)
                     : "l"(p));
    } else {
        w = __ldg(reinterpret_cast<const uint4*>(p));
    }
}

__device__ __forceinline__ void qwen4_conv_v2_unpack8(const uint4& w, float (&v)[8]) {
    const unsigned int u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[2 * i] = __uint_as_float(u[i] << 16);
        v[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
    }
}

template <bool STREAMING = false>
__device__ __forceinline__ void qwen4_conv_v2_load8(const __nv_bfloat16* __restrict__ p, float (&v)[8]) {
    uint4 w;
    qwen4_conv_v2_load_raw<STREAMING>(p, w);
    qwen4_conv_v2_unpack8(w, v);
}

template <typename OutT, unsigned int TOKENS, bool STREAMING = false>
__device__ __forceinline__ void qwen4_conv1d_l2norm_16_v2_body(
    const float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    OutT* __restrict__ output,
    unsigned int num_tokens,
    unsigned int qk_channels,
    float l2_eps,
    unsigned int input_stride,
    unsigned int output_stride
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int ch0 = blockIdx.x * 256u + lane * 8u;
    // CTA-uniform: a 256-channel block is all Q|K or all V (qk_channels % 256 == 0).
    const bool l2 = blockIdx.x * 256u < qk_channels;
    const unsigned int t_begin = (blockIdx.y * 8u + warp) * TOKENS;
    if (t_begin >= num_tokens) return;  // warp-uniform
    const unsigned int t_end = min(t_begin + TOKENS, num_tokens);

    // w[c][k] for the lane's 8 channels: 32 contiguous bf16 (64 bytes).
    float wt[8][4];
    {
        const uint4* wp = reinterpret_cast<const uint4*>(weight + (unsigned long long)ch0 * 4u);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint4 v = wp[q];
            const unsigned int u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int e = 8 * q + 2 * i;
                wt[e >> 2][e & 3] = __uint_as_float(u[i] << 16);
                wt[(e + 1) >> 2][(e + 1) & 3] = __uint_as_float(u[i] & 0xffff0000u);
            }
        }
    }
    float bias_c[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) bias_c[c] = (bias != nullptr) ? bias[ch0 + c] : 0.0f;

    // win[k] = x[t - 3 + k] for k = 0..2; before the chunk start x[-j] = conv_state[ch * 4 + 4 - j]
    // (v1: idx = t - 3 + k < 0 reads conv_state[ch * d_conv + (k + t + 1)] = [ch * 4 + idx + 4]).
    float win[3][8];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int idx = (int)t_begin - 3 + k;
        if (idx >= 0) {
            qwen4_conv_v2_load8<STREAMING>(new_input + (unsigned long long)idx * input_stride + ch0, win[k]);
        } else {
#pragma unroll
            for (int c = 0; c < 8; ++c)
                win[k][c] = conv_state[(unsigned long long)(ch0 + c) * 4u + (unsigned int)(idx + 4)];
        }
    }

    uint4 raw_cur;
    qwen4_conv_v2_load_raw<STREAMING>(new_input + (unsigned long long)t_begin * input_stride + ch0, raw_cur);

#pragma unroll 2
    for (unsigned int t = t_begin; t < t_end; ++t) {
        uint4 raw_nxt;
        if (t + 1u < t_end) {
            qwen4_conv_v2_load_raw<STREAMING>(new_input + (unsigned long long)(t + 1u) * input_stride + ch0, raw_nxt);
        }
        float cur[8];
        qwen4_conv_v2_unpack8(raw_cur, cur);
        raw_cur = raw_nxt;
        float silu[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            float acc = bias_c[c];
            acc += win[0][c] * wt[c][0];
            acc += win[1][c] * wt[c][1];
            acc += win[2][c] * wt[c][2];
            acc += cur[c] * wt[c][3];
            float sigmoid_acc = qwen4_fast_rcp_rn_1p_exp(acc);
            silu[c] = acc * sigmoid_acc;
        }
        if (l2) {
            float sq[8];
#pragma unroll
            for (int c = 0; c < 8; ++c) sq[c] = silu[c] * silu[c];
#pragma unroll
            for (int c = 0; c < 8; ++c) sq[c] += __shfl_down_sync(0xFFFFFFFF, sq[c], 2);
#pragma unroll
            for (int c = 0; c < 8; ++c) sq[c] += __shfl_down_sync(0xFFFFFFFF, sq[c], 1);
#pragma unroll
            for (int c = 0; c < 4; ++c) sq[c] += sq[c + 4];
#pragma unroll
            for (int c = 0; c < 2; ++c) sq[c] += sq[c + 2];
            sq[0] += sq[1];
            const unsigned int head_lane = lane & 16u;
            const float s0 = __shfl_sync(0xFFFFFFFF, sq[0], head_lane | 0u);
            const float s1 = __shfl_sync(0xFFFFFFFF, sq[0], head_lane | 4u);
            const float s2 = __shfl_sync(0xFFFFFFFF, sq[0], head_lane | 8u);
            const float s3 = __shfl_sync(0xFFFFFFFF, sq[0], head_lane | 12u);
            float total = s0 + s1 + s2 + s3;
            const float inv = rsqrtf(total + l2_eps);
#pragma unroll
            for (int c = 0; c < 8; ++c) silu[c] *= inv;
        }
        uint4 o;
        o.x = qwen4_fi_pack2<OutT>(silu[0], silu[1]);
        o.y = qwen4_fi_pack2<OutT>(silu[2], silu[3]);
        o.z = qwen4_fi_pack2<OutT>(silu[4], silu[5]);
        o.w = qwen4_fi_pack2<OutT>(silu[6], silu[7]);
        if constexpr (STREAMING) {
            asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                         :: "l"(output + (unsigned long long)t * output_stride + ch0),
                            "r"(o.x), "r"(o.y), "r"(o.z), "r"(o.w)
                         : "memory");
        } else {
            *reinterpret_cast<uint4*>(output + (unsigned long long)t * output_stride + ch0) = o;
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            win[0][c] = win[1][c];
            win[1][c] = win[2][c];
            win[2][c] = cur[c];
        }
    }
}

// Tokens per warp strip of the v2 entries (the host grid uses the same constant).
#define QWEN4_FI_CONV_V2_TOKENS 32u

#define QWEN4_FI_CONV_V2_ENTRY(NAME, OUT_T)                                           \
    extern "C" __global__ void __launch_bounds__(256) NAME(                           \
        const float* __restrict__ conv_state,                                         \
        const __nv_bfloat16* __restrict__ new_input,                                  \
        const __nv_bfloat16* __restrict__ weight, const float* __restrict__ bias,     \
        OUT_T* __restrict__ output, unsigned int num_tokens, unsigned int dim,        \
        unsigned int d_conv, unsigned int qk_channels, unsigned int head_dim,         \
        float l2_eps, unsigned int input_stride, unsigned int output_stride) {        \
        (void)dim; (void)d_conv; (void)head_dim;                                      \
        qwen4_conv1d_l2norm_16_v2_body<OUT_T, QWEN4_FI_CONV_V2_TOKENS>(               \
            conv_state, new_input, weight, bias, output, num_tokens, qk_channels,     \
            l2_eps, input_stride, output_stride);                                     \
    }

QWEN4_FI_CONV_V2_ENTRY(qwen4_conv1d_l2norm_bf16_prefill_v2, __nv_bfloat16)
QWEN4_FI_CONV_V2_ENTRY(qwen4_conv1d_l2norm_f16_prefill_v2, __half)

// BF16-input twin of `qwen4_gated_rms_norm_sigmoid_f32_multi_seq`
// (common/qwen4_hyper.cu, selected for output_gate_type "sigmoid"): per
// (row, head) RMS norm of the GDN output, times weight, times sigmoid(z).
// Grid: (num_v_heads, num_rows, 1)   Block: (head_dim, 1, 1)
template <typename InT>
__device__ __forceinline__ void qwen4_gated_rms_norm_16in_body(
    const InT* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int head_dim,
    float eps,
    unsigned int input_stride,
    unsigned int gate_stride,
    unsigned int output_stride
) {
    unsigned int head = blockIdx.x;
    unsigned int row = blockIdx.y;
    unsigned int tid = threadIdx.x;

    const InT* x = input + (unsigned long long)row * input_stride + head * head_dim;
    const __nv_bfloat16* g = gate + (unsigned long long)row * gate_stride + head * head_dim;
    __nv_bfloat16* out = output + (unsigned long long)row * output_stride + head * head_dim;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) {
        float f = qwen4_fi_load(x[i]);
        sum_sq += f * f;
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, off);
    }
    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, off);
        }
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(qwen4_fi_div_approx(warp_sums[0], (float)head_dim) + eps);
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) {
        float f = qwen4_fi_load(x[i]);
        float w = __bfloat162float(weight[i]);
        float gv = __bfloat162float(g[i]);
        // Flash-Next gates the GDN output with sigmoid(z) (output_gate_type
        // "sigmoid"), with the op order of qwen4_gated_rms_sigmoid_body, so this
        // equals qwen4_gated_rms_norm_sigmoid_f32_multi_seq on the widened input.
        out[i] = __float2bfloat16(qwen4_fi_div_rn_1p_exp(f * rms * w, gv));
    }
}

#define QWEN4_FI_NORM_ENTRY(NAME, IN_T)                                               \
    extern "C" __global__ void NAME(                                                  \
        const IN_T* __restrict__ input, const __nv_bfloat16* __restrict__ gate,       \
        const __nv_bfloat16* __restrict__ weight, __nv_bfloat16* __restrict__ output, \
        unsigned int head_dim, float eps, unsigned int input_stride,                  \
        unsigned int gate_stride, unsigned int output_stride) {                       \
        qwen4_gated_rms_norm_16in_body<IN_T>(input, gate, weight, output, head_dim,   \
                                             eps, input_stride, gate_stride,          \
                                             output_stride);                          \
    }

QWEN4_FI_NORM_ENTRY(qwen4_gated_rms_norm_bf16in_multi_seq, __nv_bfloat16)
QWEN4_FI_NORM_ENTRY(qwen4_gated_rms_norm_f16in_multi_seq, __half)

template <typename OutT, unsigned int TOKENS>
__device__ __forceinline__ void qwen4_conv1d_l2norm_16_v2_commit_body(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    OutT* __restrict__ output,
    unsigned int num_tokens,
    unsigned int qk_channels,
    float l2_eps,
    unsigned int input_stride,
    unsigned int output_stride
) {
    qwen4_conv1d_l2norm_16_v2_body<OutT, TOKENS, true>(
        conv_state, new_input, weight, bias, output, num_tokens, qk_channels,
        l2_eps, input_stride, output_stride);
    if (blockIdx.y == 0u && (threadIdx.x >> 5) == 0u && num_tokens >= 4u) {
        const unsigned int lane = threadIdx.x & 31u;
        const unsigned int ch0 = blockIdx.x * 256u + lane * 8u;
        float tail[4][8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const unsigned int tok = num_tokens - 4u + (unsigned int)k;
            qwen4_conv_v2_load8<true>(new_input + (unsigned long long)tok * input_stride + ch0, tail[k]);
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            float4 st;
            st.x = tail[0][c];
            st.y = tail[1][c];
            st.z = tail[2][c];
            st.w = tail[3][c];
            *reinterpret_cast<float4*>(conv_state + (unsigned long long)(ch0 + c) * 4u) = st;
        }
    }
}

#define QWEN4_FI_CONV_V2_COMMIT_ENTRY(NAME, OUT_T)                                    \
    extern "C" __global__ void __launch_bounds__(256) NAME(                           \
        float* __restrict__ conv_state,                                               \
        const __nv_bfloat16* __restrict__ new_input,                                  \
        const __nv_bfloat16* __restrict__ weight, const float* __restrict__ bias,     \
        OUT_T* __restrict__ output, unsigned int num_tokens, unsigned int dim,        \
        unsigned int d_conv, unsigned int qk_channels, unsigned int head_dim,         \
        float l2_eps, unsigned int input_stride, unsigned int output_stride) {        \
        (void)dim; (void)d_conv; (void)head_dim;                                      \
        qwen4_conv1d_l2norm_16_v2_commit_body<OUT_T, QWEN4_FI_CONV_V2_TOKENS>(        \
            conv_state, new_input, weight, bias, output, num_tokens, qk_channels,     \
            l2_eps, input_stride, output_stride);                                     \
    }

QWEN4_FI_CONV_V2_COMMIT_ENTRY(qwen4_conv1d_l2norm_bf16_prefill_v2_commit, __nv_bfloat16)
QWEN4_FI_CONV_V2_COMMIT_ENTRY(qwen4_conv1d_l2norm_f16_prefill_v2_commit, __half)

// ---------------------------------------------------------------------------
// v3 of the conv/SiLU/L2 kernel (ATLAS_QWEN4_PREFILL_GDN_CONV_V3=1):
//   1. Cooperative CTA staging of the 256-channel BF16 conv weights and optional bias into
//      bank-conflict-free shared memory (5 KiB/CTA), cutting weight global traffic 8x
//      (__launch_bounds__(256, 3)).
//   2. Explicit 2-stage software pipeline (#pragma unroll 1): loads the 3-row halo via
//      ld.global.nc.L1::no_allocate.v4.u32, prefetches TWO rows (t_begin and t_begin+1) via
//      ld.global.cs.v4.u32 before the loop, and issues ld.global.cs.v4.u32 for row t+2 at
//      the top of iteration t while evaluating win[0..2] * wt[0..2]. Interior consumed rows
//      (t + 3 < t_end && t + 4 < num_tokens, 128 B aligned) issue `discard.global.L2` on
//      `new_input` (caller-owned scratch buffer consumed solely by this conv pass).
//   3. Output stores use L2::evict_last for t_begin < 768 and streaming st.global.cs.v4.u32
//      for t_begin >= 768 so large QKV streams do not thrash L2 cache.
//   4. 100% bit-identical (0 ULP) to v1 and v2 under --fmad=false.
__device__ __forceinline__ uint4 qwen4_conv_v3_ld128_nc(const __nv_bfloat16* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 qwen4_conv_v3_ld128_cs(const __nv_bfloat16* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ void qwen4_conv_v3_st128_el(void* p, uint4 v) {
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(pol)
                 : "memory");
}

__device__ __forceinline__ void qwen4_conv_v3_st128_cs(void* p, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__device__ __forceinline__ void qwen4_conv_v3_unpack8(uint4 w, float (&v)[8]) {
    const unsigned int u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[2 * i] = __uint_as_float(u[i] << 16);
        v[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
    }
}

template <typename OutT>
__device__ __forceinline__ unsigned int qwen4_conv_v3_pack2(float a, float b);
template <>
__device__ __forceinline__ unsigned int qwen4_conv_v3_pack2<__half>(float a, float b) {
    unsigned int d;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(b), "f"(a));
    return d;
}
template <>
__device__ __forceinline__ unsigned int qwen4_conv_v3_pack2<__nv_bfloat16>(float a, float b) {
    unsigned int d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(b), "f"(a));
    return d;
}

template <typename OutT, unsigned int TOKENS, bool COMMIT = false>
__device__ __forceinline__ void qwen4_conv1d_l2norm_16_v3_body(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    OutT* __restrict__ output,
    unsigned int num_tokens,
    unsigned int qk_channels,
    float l2_eps,
    unsigned int input_stride,
    unsigned int output_stride
) {
    __shared__ float s_wt[4][8][32];
    __shared__ float s_bias[8][32];
    {
        const unsigned int tid = threadIdx.x;
        const unsigned int ch = blockIdx.x * 256u + tid;
        const unsigned int dst_lane = tid >> 3;
        const unsigned int dst_c = tid & 7u;
        const uint2 w2 = *reinterpret_cast<const uint2*>(weight + (unsigned long long)ch * 4u);
        s_wt[0][dst_c][dst_lane] = __uint_as_float(w2.x << 16);
        s_wt[1][dst_c][dst_lane] = __uint_as_float(w2.x & 0xffff0000u);
        s_wt[2][dst_c][dst_lane] = __uint_as_float(w2.y << 16);
        s_wt[3][dst_c][dst_lane] = __uint_as_float(w2.y & 0xffff0000u);
        s_bias[dst_c][dst_lane] = (bias != nullptr) ? bias[ch] : 0.0f;
    }
    __syncthreads();

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int rev_by = gridDim.y - 1u - blockIdx.y;
    const unsigned int t_begin = (rev_by * 8u + warp) * TOKENS;
    if (t_begin >= num_tokens) return;  // warp-uniform
    const unsigned int t_end = min(t_begin + TOKENS, num_tokens);
    const unsigned int ch0 = blockIdx.x * 256u + lane * 8u;
    const bool l2 = blockIdx.x * 256u < qk_channels;

    float win[3][8];
    if (t_begin >= 3u) {
        const uint4 r0 = qwen4_conv_v3_ld128_nc(
            new_input + (unsigned long long)(t_begin - 3u) * input_stride + ch0);
        const uint4 r1 = qwen4_conv_v3_ld128_nc(
            new_input + (unsigned long long)(t_begin - 2u) * input_stride + ch0);
        const uint4 r2 = qwen4_conv_v3_ld128_nc(
            new_input + (unsigned long long)(t_begin - 1u) * input_stride + ch0);
        qwen4_conv_v3_unpack8(r0, win[0]);
        qwen4_conv_v3_unpack8(r1, win[1]);
        qwen4_conv_v3_unpack8(r2, win[2]);
    } else {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const float4 st = *reinterpret_cast<const float4*>(
                conv_state + (unsigned long long)(ch0 + c) * 4u);
            win[0][c] = st.y;
            win[1][c] = st.z;
            win[2][c] = st.w;
        }
    }

    uint4 next0_raw = qwen4_conv_v3_ld128_cs(
        new_input + (unsigned long long)t_begin * input_stride + ch0);
    uint4 next1_raw = (t_begin + 1u < t_end)
        ? qwen4_conv_v3_ld128_cs(
              new_input + (unsigned long long)(t_begin + 1u) * input_stride + ch0)
        : next0_raw;

#pragma unroll 1
    for (unsigned int t = t_begin; t < t_end; ++t) {
        const uint4 cur_raw = next0_raw;
        next0_raw = next1_raw;
        if (t + 2u < t_end) {
            next1_raw = qwen4_conv_v3_ld128_cs(
                new_input + (unsigned long long)(t + 2u) * input_stride + ch0);
        }

        float acc[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            float a = s_bias[c][lane];
            a += win[0][c] * s_wt[0][c][lane];
            a += win[1][c] * s_wt[1][c][lane];
            a += win[2][c] * s_wt[2][c][lane];
            acc[c] = a;
        }

        float cur[8];
        qwen4_conv_v3_unpack8(cur_raw, cur);
        if (t + 3u < t_end && t + 4u < num_tokens) {
            __syncwarp();
            if ((lane & 7u) == 0u) {
                const unsigned char* p_disc = reinterpret_cast<const unsigned char*>(
                    new_input + (unsigned long long)t * input_stride + ch0);
                if ((reinterpret_cast<unsigned long long>(p_disc) & 127ull) == 0ull) {
                    asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p_disc) : "memory");
                }
            }
        }
        float silu[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            float a = acc[c] + cur[c] * s_wt[3][c][lane];
            float sigmoid_acc = qwen4_fast_rcp_rn_1p_exp(a);
            silu[c] = a * sigmoid_acc;
            win[0][c] = win[1][c];
            win[1][c] = win[2][c];
            win[2][c] = cur[c];
        }

        if (l2) {
            float sq[8];
#pragma unroll
            for (int c = 0; c < 8; ++c) sq[c] = silu[c] * silu[c];
#pragma unroll
            for (int c = 0; c < 8; ++c) sq[c] += __shfl_down_sync(0xFFFFFFFF, sq[c], 2);
#pragma unroll
            for (int c = 0; c < 8; ++c) sq[c] += __shfl_down_sync(0xFFFFFFFF, sq[c], 1);
#pragma unroll
            for (int c = 0; c < 4; ++c) sq[c] += sq[c + 4];
#pragma unroll
            for (int c = 0; c < 2; ++c) sq[c] += sq[c + 2];
            sq[0] += sq[1];
            const unsigned int head_lane = lane & 16u;
            const float s0 = __shfl_sync(0xFFFFFFFF, sq[0], head_lane | 0u);
            const float s1 = __shfl_sync(0xFFFFFFFF, sq[0], head_lane | 4u);
            const float s2 = __shfl_sync(0xFFFFFFFF, sq[0], head_lane | 8u);
            const float s3 = __shfl_sync(0xFFFFFFFF, sq[0], head_lane | 12u);
            float total = s0 + s1 + s2 + s3;
            const float inv = rsqrtf(total + l2_eps);
#pragma unroll
            for (int c = 0; c < 8; ++c) silu[c] *= inv;
        }

        uint4 o;
        o.x = qwen4_conv_v3_pack2<OutT>(silu[0], silu[1]);
        o.y = qwen4_conv_v3_pack2<OutT>(silu[2], silu[3]);
        o.z = qwen4_conv_v3_pack2<OutT>(silu[4], silu[5]);
        o.w = qwen4_conv_v3_pack2<OutT>(silu[6], silu[7]);
        OutT* dst = output + (unsigned long long)t * output_stride + ch0;
        if (t_begin < 768u) {
            qwen4_conv_v3_st128_el(dst, o);
        } else {
            qwen4_conv_v3_st128_cs(dst, o);
        }
    }

    if (COMMIT && t_begin == 0u && num_tokens >= 4u) {
        const unsigned int t0 = num_tokens - 4u;
        const uint4 tr0 = qwen4_conv_v3_ld128_nc(new_input + (unsigned long long)(t0 + 0u) * input_stride + ch0);
        const uint4 tr1 = qwen4_conv_v3_ld128_nc(new_input + (unsigned long long)(t0 + 1u) * input_stride + ch0);
        const uint4 tr2 = qwen4_conv_v3_ld128_nc(new_input + (unsigned long long)(t0 + 2u) * input_stride + ch0);
        const uint4 tr3 = qwen4_conv_v3_ld128_nc(new_input + (unsigned long long)(t0 + 3u) * input_stride + ch0);
        float tail[4][8];
        qwen4_conv_v3_unpack8(tr0, tail[0]);
        qwen4_conv_v3_unpack8(tr1, tail[1]);
        qwen4_conv_v3_unpack8(tr2, tail[2]);
        qwen4_conv_v3_unpack8(tr3, tail[3]);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            float4 st;
            st.x = tail[0][c];
            st.y = tail[1][c];
            st.z = tail[2][c];
            st.w = tail[3][c];
            *reinterpret_cast<float4*>(conv_state + (unsigned long long)(ch0 + c) * 4u) = st;
        }
    }
}

#define QWEN4_FI_CONV_V3_ENTRY(NAME, OUT_T, COMMIT)                                   \
    extern "C" __global__ void __launch_bounds__(256, 3) NAME(                        \
        float* __restrict__ conv_state,                                               \
        const __nv_bfloat16* __restrict__ new_input,                                  \
        const __nv_bfloat16* __restrict__ weight, const float* __restrict__ bias,     \
        OUT_T* __restrict__ output, unsigned int num_tokens, unsigned int dim,        \
        unsigned int d_conv, unsigned int qk_channels, unsigned int head_dim,         \
        float l2_eps, unsigned int input_stride, unsigned int output_stride) {        \
        (void)dim; (void)d_conv; (void)head_dim;                                      \
        qwen4_conv1d_l2norm_16_v3_body<OUT_T, QWEN4_FI_CONV_V2_TOKENS, COMMIT>(       \
            conv_state, new_input, weight, bias, output, num_tokens, qk_channels,     \
            l2_eps, input_stride, output_stride);                                     \
    }

QWEN4_FI_CONV_V3_ENTRY(qwen4_conv1d_l2norm_bf16_prefill_v3, __nv_bfloat16, false)
QWEN4_FI_CONV_V3_ENTRY(qwen4_conv1d_l2norm_f16_prefill_v3, __half, false)
QWEN4_FI_CONV_V3_ENTRY(qwen4_conv1d_l2norm_bf16_prefill_v3_commit, __nv_bfloat16, true)
QWEN4_FI_CONV_V3_ENTRY(qwen4_conv1d_l2norm_f16_prefill_v3_commit, __half, true)

__device__ __forceinline__ uint2 qwen4_conv_v3_ld64_nc(const __nv_bfloat16* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint2 qwen4_conv_v3_ld64_cs(const __nv_bfloat16* p) {
    uint2 v;
    asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ void qwen4_conv_v3_st64_el(
    void* p, uint2 v, unsigned long long pol
) {
    asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;"
                 :: "l"(p), "r"(v.x), "r"(v.y), "l"(pol)
                 : "memory");
}

__device__ __forceinline__ void qwen4_conv_v3_unpack4(uint2 w, float (&v)[4]) {
    v[0] = __uint_as_float(w.x << 16);
    v[1] = __uint_as_float(w.x & 0xffff0000u);
    v[2] = __uint_as_float(w.y << 16);
    v[3] = __uint_as_float(w.y & 0xffff0000u);
}

// Strip-mined L2-resident twin of v3: processes `strip_tokens` rows starting at `row_offset`
// within a `total_tokens`-row chunk using 4 warps (128 threads) spanning 128 channels/CTA
// (1 Q/K head = 4 channels/lane = 256 B coalesced per warp). Rows are partitioned evenly across
// `gridDim.y * 4` warps (`grid = (conv_dim / 128, ceil(strip_tokens / 192))` -> `(80, 3) = 240`
// CTAs = 5.00 exact waves on 48 SMs at 512-row strips, ~42-43 tokens/warp -> 7.0% halo).
// Conv weights and optional bias are hoisted into thread registers (0 B SMEM, 0 barriers, <= 64
// regs -> 8 CTAs/SM). Interior input rows stream via `ld.global.cs`, while each tile's 3-row
// halo AND last 3 rows (`t + 3 >= t_end`) use L2-retaining `ld.global.nc` so adjacent tiles'
// halos hit L2 instead of DRAM. `qkv_strip` stores use `L2::evict_last` for FlashInfer GDN.
// Commits `conv_state` once on the final strip (`row_offset + strip_tokens == total_tokens`).
// 100% bit-identical (0 ULP) to v1/v2/v3 under `--fmad=false`.
#define QWEN4_FI_CONV_V3_STRIP_CHANNELS_PER_CTA 128u
#define QWEN4_FI_CONV_V3_STRIP_WARPS 4u
#define QWEN4_FI_CONV_V3_STRIP_TOKENS 48u

template <typename OutT, unsigned int WARPS>
__device__ __forceinline__ void qwen4_conv1d_l2norm_16_v3_strip_body(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    OutT* __restrict__ output,
    unsigned int strip_tokens,
    unsigned int row_offset,
    unsigned int total_tokens,
    unsigned int qk_channels,
    float l2_eps,
    unsigned int input_stride,
    unsigned int output_stride
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int rev_by = gridDim.y - 1u - blockIdx.y;
    const unsigned int total_warps = gridDim.y * WARPS;
    const unsigned int w_idx = rev_by * WARPS + warp;
    const unsigned int base_toks = strip_tokens / total_warps;
    const unsigned int rem_toks = strip_tokens % total_warps;
    const unsigned int t_len = base_toks + (w_idx < rem_toks ? 1u : 0u);
    const unsigned int ch0 = blockIdx.x * QWEN4_FI_CONV_V3_STRIP_CHANNELS_PER_CTA + lane * 4u;
    if (t_len > 0u) {
        const unsigned int t_begin = w_idx * base_toks + min(w_idx, rem_toks);
        const unsigned int t_end = t_begin + t_len;
        const unsigned int abs_t_begin = row_offset + t_begin;
        const bool l2 = blockIdx.x * QWEN4_FI_CONV_V3_STRIP_CHANNELS_PER_CTA < qk_channels;

        // Hoist 4-tap conv weights (4 channels x 4 BF16 = 32 B = 2 x uint4) + optional bias into regs.
        float wt[4][4];
        {
            const uint4* wp = reinterpret_cast<const uint4*>(weight + (unsigned long long)ch0 * 4u);
            const uint4 w0 = __ldg(wp);
            const uint4 w1 = __ldg(wp + 1);
            wt[0][0] = __uint_as_float(w0.x << 16);
            wt[1][0] = __uint_as_float(w0.x & 0xffff0000u);
            wt[2][0] = __uint_as_float(w0.y << 16);
            wt[3][0] = __uint_as_float(w0.y & 0xffff0000u);
            wt[0][1] = __uint_as_float(w0.z << 16);
            wt[1][1] = __uint_as_float(w0.z & 0xffff0000u);
            wt[2][1] = __uint_as_float(w0.w << 16);
            wt[3][1] = __uint_as_float(w0.w & 0xffff0000u);
            wt[0][2] = __uint_as_float(w1.x << 16);
            wt[1][2] = __uint_as_float(w1.x & 0xffff0000u);
            wt[2][2] = __uint_as_float(w1.y << 16);
            wt[3][2] = __uint_as_float(w1.y & 0xffff0000u);
            wt[0][3] = __uint_as_float(w1.z << 16);
            wt[1][3] = __uint_as_float(w1.z & 0xffff0000u);
            wt[2][3] = __uint_as_float(w1.w << 16);
            wt[3][3] = __uint_as_float(w1.w & 0xffff0000u);
        }
        float bias_c[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            bias_c[c] = (bias != nullptr) ? __ldg(bias + ch0 + c) : 0.0f;
        }

        float win[3][4];
        if (abs_t_begin >= 3u) {
            const uint2 r0 = qwen4_conv_v3_ld64_nc(
                new_input + (unsigned long long)(abs_t_begin - 3u) * input_stride + ch0);
            const uint2 r1 = qwen4_conv_v3_ld64_nc(
                new_input + (unsigned long long)(abs_t_begin - 2u) * input_stride + ch0);
            const uint2 r2 = qwen4_conv_v3_ld64_nc(
                new_input + (unsigned long long)(abs_t_begin - 1u) * input_stride + ch0);
            qwen4_conv_v3_unpack4(r0, win[0]);
            qwen4_conv_v3_unpack4(r1, win[1]);
            qwen4_conv_v3_unpack4(r2, win[2]);
        } else {
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                const int idx = static_cast<int>(abs_t_begin) - 3 + k;
                if (idx >= 0) {
                    const uint2 rk = qwen4_conv_v3_ld64_nc(
                        new_input + (unsigned long long)idx * input_stride + ch0);
                    qwen4_conv_v3_unpack4(rk, win[k]);
                } else {
                    const unsigned int slot = static_cast<unsigned int>(idx + 4);
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        win[k][c] = conv_state[(unsigned long long)(ch0 + c) * 4u + slot];
                    }
                }
            }
        }

        unsigned long long pol_el;
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));

        const __nv_bfloat16* p_in0 =
            new_input + (unsigned long long)abs_t_begin * input_stride + ch0;
        uint2 next0_raw = (t_begin + 3u >= t_end)
            ? qwen4_conv_v3_ld64_nc(p_in0)
            : qwen4_conv_v3_ld64_cs(p_in0);
        uint2 next1_raw = next0_raw;
        if (t_begin + 1u < t_end) {
            const __nv_bfloat16* p_in1 = p_in0 + input_stride;
            next1_raw = (t_begin + 4u >= t_end)
                ? qwen4_conv_v3_ld64_nc(p_in1)
                : qwen4_conv_v3_ld64_cs(p_in1);
        }

        const __nv_bfloat16* p_pf = p_in0 + (unsigned long long)2u * input_stride;
        OutT* p_out = output + (unsigned long long)t_begin * output_stride + ch0;

#pragma unroll 1
        for (unsigned int t = t_begin; t < t_end; ++t) {
            const uint2 cur_raw = next0_raw;
            next0_raw = next1_raw;
            if (t + 2u < t_end) {
                next1_raw = (t + 5u >= t_end)
                    ? qwen4_conv_v3_ld64_nc(p_pf)
                    : qwen4_conv_v3_ld64_cs(p_pf);
            }
            p_pf += input_stride;

            float acc[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                float a = bias_c[c];
                a += win[0][c] * wt[0][c];
                a += win[1][c] * wt[1][c];
                a += win[2][c] * wt[2][c];
                acc[c] = a;
            }

            float cur[4];
            qwen4_conv_v3_unpack4(cur_raw, cur);
            float silu[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                float a = acc[c] + cur[c] * wt[3][c];
                float sigmoid_acc = qwen4_fast_rcp_rn_1p_exp(a);
                silu[c] = a * sigmoid_acc;
                win[0][c] = win[1][c];
                win[1][c] = win[2][c];
                win[2][c] = cur[c];
            }

            if (l2) {
                float sq[4];
#pragma unroll
                for (int c = 0; c < 4; ++c) sq[c] = silu[c] * silu[c];
#pragma unroll
                for (int c = 0; c < 4; ++c) sq[c] += __shfl_down_sync(0xFFFFFFFFu, sq[c], 4);
#pragma unroll
                for (int c = 0; c < 4; ++c) sq[c] += __shfl_down_sync(0xFFFFFFFFu, sq[c], 2);
#pragma unroll
                for (int c = 0; c < 4; ++c) sq[c] += __shfl_down_sync(0xFFFFFFFFu, sq[c], 1);
                sq[0] += sq[2];
                sq[1] += sq[3];
                sq[0] += sq[1];
                const float s0 = __shfl_sync(0xFFFFFFFFu, sq[0], 0u);
                const float s1 = __shfl_sync(0xFFFFFFFFu, sq[0], 8u);
                const float s2 = __shfl_sync(0xFFFFFFFFu, sq[0], 16u);
                const float s3 = __shfl_sync(0xFFFFFFFFu, sq[0], 24u);
                float total = ((s0 + s1) + s2) + s3;
                const float inv = rsqrtf(total + l2_eps);
#pragma unroll
                for (int c = 0; c < 4; ++c) silu[c] *= inv;
            }

            uint2 o;
            o.x = qwen4_conv_v3_pack2<OutT>(silu[0], silu[1]);
            o.y = qwen4_conv_v3_pack2<OutT>(silu[2], silu[3]);
            qwen4_conv_v3_st64_el(p_out, o, pol_el);
            p_out += output_stride;
        }
    }

    // Commit conv_state from w_idx == 0u (warp 0 of CTA rev_by == 0u).
    // When row_offset < 3u, every warp that can read entry conv_state
    // (abs_t_begin < 3u) resides in CTA rev_by == 0u (since WARPS == 4 > 3),
    // so a CTA-uniform __syncthreads() in rev_by == 0u eliminates both
    // intra-CTA and inter-CTA races on conv_state during single-strip prefill.
    if (row_offset + strip_tokens == total_tokens && total_tokens >= 4u) {
        if (rev_by == 0u && row_offset < 3u) {
            __syncthreads();
        }
        if (w_idx == 0u) {
            const unsigned int t0 = total_tokens - 4u;
            const uint2 tr0 = qwen4_conv_v3_ld64_nc(new_input + (unsigned long long)(t0 + 0u) * input_stride + ch0);
            const uint2 tr1 = qwen4_conv_v3_ld64_nc(new_input + (unsigned long long)(t0 + 1u) * input_stride + ch0);
            const uint2 tr2 = qwen4_conv_v3_ld64_nc(new_input + (unsigned long long)(t0 + 2u) * input_stride + ch0);
            const uint2 tr3 = qwen4_conv_v3_ld64_nc(new_input + (unsigned long long)(t0 + 3u) * input_stride + ch0);
            float tail[4][4];
            qwen4_conv_v3_unpack4(tr0, tail[0]);
            qwen4_conv_v3_unpack4(tr1, tail[1]);
            qwen4_conv_v3_unpack4(tr2, tail[2]);
            qwen4_conv_v3_unpack4(tr3, tail[3]);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                float4 st;
                st.x = tail[0][c];
                st.y = tail[1][c];
                st.z = tail[2][c];
                st.w = tail[3][c];
                *reinterpret_cast<float4*>(conv_state + (unsigned long long)(ch0 + c) * 4u) = st;
            }
        }
    }
}

#define QWEN4_FI_CONV_V3_STRIP_ENTRY(NAME, OUT_T)                                     \
    extern "C" __global__ void __launch_bounds__(128, 8) NAME(                        \
        float* __restrict__ conv_state,                                               \
        const __nv_bfloat16* __restrict__ new_input,                                  \
        const __nv_bfloat16* __restrict__ weight, const float* __restrict__ bias,     \
        OUT_T* __restrict__ output, unsigned int strip_tokens,                        \
        unsigned int row_offset, unsigned int total_tokens,                           \
        unsigned int qk_channels, float l2_eps,                                       \
        unsigned int input_stride, unsigned int output_stride) {                      \
        qwen4_conv1d_l2norm_16_v3_strip_body<                                         \
            OUT_T, QWEN4_FI_CONV_V3_STRIP_WARPS>(                                     \
            conv_state, new_input, weight, bias, output, strip_tokens, row_offset,    \
            total_tokens, qk_channels, l2_eps, input_stride, output_stride);          \
    }

QWEN4_FI_CONV_V3_STRIP_ENTRY(qwen4_conv1d_l2norm_bf16_prefill_v3_strip, __nv_bfloat16)
QWEN4_FI_CONV_V3_STRIP_ENTRY(qwen4_conv1d_l2norm_f16_prefill_v3_strip, __half)

// ---------------------------------------------------------------------------
// v4 strip-mined conv/SiLU/L2 kernel (`ATLAS_QWEN4_PREFILL_GDN_CONV_V4_STRIP`):
//   1. Upgrades v3_strip's 8B `ld64`/`st64` to 16B `ld128`/`st128` (`uint4`, 8 channels/lane =
//      256 channels/CTA = 2 Q/K heads/warp = 512 B = 4 exact 128 B L2 cache lines per row),
//      cutting LSU issue count by 2x and warp L2-norm SHFL count by 37.5%.
//   2. Cooperative 4-warp SMEM sliding ring buffer (`float4 s_ring[16][2][32]` = 16 KiB/CTA,
//      bank-conflict-free `STS.128`/`LDS.128`, 1 `__syncthreads()` per 4 rows): eliminates
//      intra-CTA warp halos (1 halo/CTA -> 3.5% halo vs 7.0% in v3_strip at 512-row strips)
//      and frees `win[3][8]` registers so 4-tap conv weights hoist into registers (`96` regs,
//      `0` spills, `5` CTAs/SM -> all `(40, 6) = 240` CTAs fit in one resident wave on 48 SMs).
//   3. Emits `discard.global.L2 [p], 128` (`CCTL.E.RML2`) on strictly consumed `new_input` QKV
//      cache lines (guarded by 128B base+stride alignment, `discard_flags & 1`, trailing 3-row
//      halo `t + 3 < cta_t1`, and final 4-row `conv_state` tail `abs_t + 4 < total_tokens`)
//      plus `L2::evict_first` + `L2::128B` interior reads and `L2::evict_last` `qkv_strip`
//      stores so FlashInfer GDN immediately hits L2.
//   4. 100% bit-identical (0 ULP) to v1/v2/v3/v3_strip under `--fmad=false`.
__device__ __forceinline__ float qwen4_conv_v4_rsqrt_approx(float x) {
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint4 qwen4_conv_v4_ld128_nc_ef_128(
    const __nv_bfloat16* p, unsigned long long pol_ef
) {
    uint4 v;
    asm volatile(
        "ld.global.nc.L1::no_allocate.L2::128B.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p), "l"(pol_ef));
    return v;
}

__device__ __forceinline__ void qwen4_conv_v4_st128_el(
    void* p, uint4 v, unsigned long long pol_el
) {
    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(pol_el)
                 : "memory");
}

__device__ __forceinline__ void qwen4_conv_v4_discard_l2(const void* p) {
    asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
}

#define QWEN4_FI_CONV_V4_STRIP_CHANNELS_PER_CTA 256u
#define QWEN4_FI_CONV_V4_STRIP_WARPS 4u
#define QWEN4_FI_CONV_V4_STRIP_ROWS_PER_CTA 96u
#define QWEN4_FI_CONV_V4_STRIP_DISC_INPUT 1u
#define QWEN4_FI_CONV_V4_STRIP_DISC_COARSE 2u
#define QWEN4_FI_CONV_V4_STRIP_DISC_EVERY_ROW 4u

template <typename OutT, bool GATESIGHT = false>
__device__ __forceinline__ void qwen4_conv1d_l2norm_16_v4_strip_body(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    OutT* __restrict__ output,
    unsigned int strip_tokens,
    unsigned int row_offset,
    unsigned int total_tokens,
    unsigned int qk_channels,
    float l2_eps,
    unsigned int input_stride,
    unsigned int output_stride,
    unsigned int discard_flags,
    __half* __restrict__ gatesight_out = nullptr,
    unsigned int num_v_heads = 0u
) {
    // 16-slot FP32 ring buffer (16 KiB/CTA, 2 bank-conflict-free float4s/lane/row).
    // Between barrier m and barrier m+1, active readers touch slots [4m..4m+5] while
    // next-step writers populate [4m+7..4m+10] (span 11 < 16), requiring 1 barrier per 4 rows.
    __shared__ float4 s_ring[16][2][32];

    const unsigned int rev_by = gridDim.y - 1u - blockIdx.y;
    const unsigned int base_cta = strip_tokens / gridDim.y;
    const unsigned int rem_cta = strip_tokens % gridDim.y;
    const unsigned int cta_len = base_cta + (rev_by < rem_cta ? 1u : 0u);
    if (cta_len == 0u) return;  // CTA-uniform

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int cta_t0 = rev_by * base_cta + min(rev_by, rem_cta);
    const unsigned int cta_t1 = cta_t0 + cta_len;
    const unsigned int abs_cta_t0 = row_offset + cta_t0;
    const unsigned int ch0 = blockIdx.x * QWEN4_FI_CONV_V4_STRIP_CHANNELS_PER_CTA + lane * 8u;
    const bool l2 = blockIdx.x * QWEN4_FI_CONV_V4_STRIP_CHANNELS_PER_CTA < qk_channels;
    const bool can_disc_in =
        ((discard_flags & (QWEN4_FI_CONV_V4_STRIP_DISC_INPUT
                           | QWEN4_FI_CONV_V4_STRIP_DISC_COARSE
                           | QWEN4_FI_CONV_V4_STRIP_DISC_EVERY_ROW)) != 0u)
        && ((reinterpret_cast<unsigned long long>(new_input) & 127ull) == 0ull)
        && ((input_stride & 63u) == 0u);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200)
    // SM120 has a 96 MiB L2 and inner-loop loads already use L2::evict_first.
    // Coarsen inner-loop `discard.global.L2` to 1-in-4 outer steps (4x fewer
    // L2 tag invalidations) unless `DISC_EVERY_ROW` explicitly forces per-row.
    const bool coarse_disc_in =
        ((discard_flags & QWEN4_FI_CONV_V4_STRIP_DISC_EVERY_ROW) == 0u);
#else
    const bool coarse_disc_in =
        ((discard_flags & QWEN4_FI_CONV_V4_STRIP_DISC_COARSE) != 0u)
        && ((discard_flags & QWEN4_FI_CONV_V4_STRIP_DISC_EVERY_ROW) == 0u);
#endif
    const bool disc_lane = (lane & 7u) == 0u;

    // Hoist 4-tap BF16 conv weights for the lane's 8 channels (64 B = 4 x uint4) into registers.
    float wt[4][8];
    {
        const uint4* wp = reinterpret_cast<const uint4*>(weight + (unsigned long long)ch0 * 4u);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint4 v = __ldg(wp + q);
            const int c0 = 2 * q;
            wt[0][c0]     = __uint_as_float(v.x << 16);
            wt[1][c0]     = __uint_as_float(v.x & 0xffff0000u);
            wt[2][c0]     = __uint_as_float(v.y << 16);
            wt[3][c0]     = __uint_as_float(v.y & 0xffff0000u);
            wt[0][c0 + 1] = __uint_as_float(v.z << 16);
            wt[1][c0 + 1] = __uint_as_float(v.z & 0xffff0000u);
            wt[2][c0 + 1] = __uint_as_float(v.w << 16);
            wt[3][c0 + 1] = __uint_as_float(v.w & 0xffff0000u);
        }
    }

    unsigned long long pol_el, pol_ef;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));

    // Warps 0, 1, 2 cooperatively load the 3-row CTA halo in parallel into s_ring[0..2].
    if (warp < 3u) {
        const int idx = static_cast<int>(abs_cta_t0) - 3 + static_cast<int>(warp);
        float h[8];
        const __nv_bfloat16* ph = nullptr;
        if (idx >= 0) {
            ph = new_input + (unsigned long long)idx * input_stride + ch0;
            const uint4 rh = qwen4_conv_v3_ld128_nc(ph);
            qwen4_conv_v3_unpack8(rh, h);
        } else {
            const unsigned int slot = static_cast<unsigned int>(idx + 4);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                h[c] = conv_state[(unsigned long long)(ch0 + c) * 4u + slot];
            }
        }
        s_ring[warp][0][lane] = make_float4(h[0], h[1], h[2], h[3]);
        s_ring[warp][1][lane] = make_float4(h[4], h[5], h[6], h[7]);
        // Only CTA rev_by == 0 reads the previous strip's trailing 3-row halo when base_cta >= 3.
        // s_ring write above anchors rh's LDG scoreboard before __syncwarp() + L2 discard.
        if (can_disc_in && rev_by == 0u && base_cta >= 3u && idx >= 0
            && static_cast<unsigned int>(idx) + 4u < total_tokens) {
            __syncwarp();
            if (disc_lane) qwen4_conv_v4_discard_l2(ph);
        }
    }

    const __nv_bfloat16* p_cur =
        new_input + (unsigned long long)(abs_cta_t0 + warp) * input_stride + ch0;
    const __nv_bfloat16* p_pf = p_cur + (unsigned long long)4u * input_stride;
    OutT* p_out = output + (unsigned long long)(cta_t0 + warp) * output_stride + ch0;

    uint4 cur_raw = make_uint4(0u, 0u, 0u, 0u);
    if (cta_t0 + warp < cta_t1) {
        const unsigned int t0 = cta_t0 + warp;
        const unsigned int at0 = row_offset + t0;
        cur_raw = (t0 + 3u >= cta_t1 || at0 + 4u >= total_tokens)
            ? qwen4_conv_v3_ld128_nc(p_cur)
            : qwen4_conv_v4_ld128_nc_ef_128(p_cur, pol_ef);
    }

    unsigned int ring_head = 3u;
#pragma unroll 1
    for (unsigned int step = 0u; step < cta_len; step += 4u) {
        const unsigned int t = cta_t0 + step + warp;
        const bool active = (t < cta_t1);

        // Prefetch next 4-row step while unpacking, synchronizing, and computing the current step.
        uint4 next_raw = cur_raw;
        if (t + 4u < cta_t1) {
            const unsigned int tn = t + 4u;
            const unsigned int atn = row_offset + tn;
            next_raw = (tn + 3u >= cta_t1 || atn + 4u >= total_tokens)
                ? qwen4_conv_v3_ld128_nc(p_pf)
                : qwen4_conv_v4_ld128_nc_ef_128(p_pf, pol_ef);
        }
        p_pf += (unsigned long long)4u * input_stride;

        float cur[8];
        if (active) {
            qwen4_conv_v3_unpack8(cur_raw, cur);
            const unsigned int dst_slot = (ring_head + warp) & 15u;
            s_ring[dst_slot][0][lane] = make_float4(cur[0], cur[1], cur[2], cur[3]);
            s_ring[dst_slot][1][lane] = make_float4(cur[4], cur[5], cur[6], cur[7]);
        }
        __syncthreads();

        if (active) {
            const unsigned int s0 = (ring_head + warp + 13u) & 15u;
            const unsigned int s1 = (ring_head + warp + 14u) & 15u;
            const unsigned int s2 = (ring_head + warp + 15u) & 15u;
            const float4 w0a = s_ring[s0][0][lane], w0b = s_ring[s0][1][lane];
            const float4 w1a = s_ring[s1][0][lane], w1b = s_ring[s1][1][lane];
            const float4 w2a = s_ring[s2][0][lane], w2b = s_ring[s2][1][lane];
            const float win0[8] = {w0a.x, w0a.y, w0a.z, w0a.w, w0b.x, w0b.y, w0b.z, w0b.w};
            const float win1[8] = {w1a.x, w1a.y, w1a.z, w1a.w, w1b.x, w1b.y, w1b.z, w1b.w};
            const float win2[8] = {w2a.x, w2a.y, w2a.z, w2a.w, w2b.x, w2b.y, w2b.z, w2b.w};

            float silu[8];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float a = (bias != nullptr) ? __ldg(bias + ch0 + c) : 0.0f;
                a += win0[c] * wt[0][c];
                a += win1[c] * wt[1][c];
                a += win2[c] * wt[2][c];
                a += cur[c]  * wt[3][c];
                const float sigmoid_acc = qwen4_fast_rcp_rn_1p_exp(a);
                silu[c] = a * sigmoid_acc;
            }

            if (l2) {
                float sq[8];
#pragma unroll
                for (int c = 0; c < 8; ++c) sq[c] = silu[c] * silu[c];
#pragma unroll
                for (int c = 0; c < 8; ++c) sq[c] += __shfl_down_sync(0xFFFFFFFFu, sq[c], 2);
#pragma unroll
                for (int c = 0; c < 8; ++c) sq[c] += __shfl_down_sync(0xFFFFFFFFu, sq[c], 1);
#pragma unroll
                for (int c = 0; c < 4; ++c) sq[c] += sq[c + 4];
#pragma unroll
                for (int c = 0; c < 2; ++c) sq[c] += sq[c + 2];
                sq[0] += sq[1];
                const unsigned int head_lane = lane & 16u;
                const float l0 = __shfl_sync(0xFFFFFFFFu, sq[0], head_lane | 0u);
                const float l1 = __shfl_sync(0xFFFFFFFFu, sq[0], head_lane | 4u);
                const float l2v = __shfl_sync(0xFFFFFFFFu, sq[0], head_lane | 8u);
                const float l3 = __shfl_sync(0xFFFFFFFFu, sq[0], head_lane | 12u);
                const float total = ((l0 + l1) + l2v) + l3;
                const float inv = qwen4_conv_v4_rsqrt_approx(total + l2_eps);
#pragma unroll
                for (int c = 0; c < 8; ++c) silu[c] *= inv;
            } else if constexpr (GATESIGHT) {
                if (gatesight_out != nullptr) {
                    float sq[8];
#pragma unroll
                    for (int c = 0; c < 8; ++c) sq[c] = silu[c] * silu[c];
#pragma unroll
                    for (int c = 0; c < 8; ++c) sq[c] += __shfl_down_sync(0xFFFFFFFFu, sq[c], 2);
#pragma unroll
                    for (int c = 0; c < 8; ++c) sq[c] += __shfl_down_sync(0xFFFFFFFFu, sq[c], 1);
#pragma unroll
                    for (int c = 0; c < 4; ++c) sq[c] += sq[c + 4];
#pragma unroll
                    for (int c = 0; c < 2; ++c) sq[c] += sq[c + 2];
                    sq[0] += sq[1];
                    const unsigned int head_lane = lane & 16u;
                    const float l0 = __shfl_sync(0xFFFFFFFFu, sq[0], head_lane | 0u);
                    const float l1 = __shfl_sync(0xFFFFFFFFu, sq[0], head_lane | 4u);
                    const float l2v = __shfl_sync(0xFFFFFFFFu, sq[0], head_lane | 8u);
                    const float l3 = __shfl_sync(0xFFFFFFFFu, sq[0], head_lane | 12u);
                    const float total = ((l0 + l1) + l2v) + l3;
                    if ((lane & 15u) == 0u) {
                        const unsigned int vh =
                            ((blockIdx.x * QWEN4_FI_CONV_V4_STRIP_CHANNELS_PER_CTA - qk_channels) >> 7)
                            + (lane >> 4);
                        if (vh < num_v_heads) {
                            gatesight_out[
                                ((unsigned long long)(row_offset + t) * num_v_heads + vh) * 3ull + 2ull
                            ] = __float2half_rn(sqrtf(total));
                        }
                    }
                }
            }

            uint4 o;
            o.x = qwen4_conv_v3_pack2<OutT>(silu[0], silu[1]);
            o.y = qwen4_conv_v3_pack2<OutT>(silu[2], silu[3]);
            o.z = qwen4_conv_v3_pack2<OutT>(silu[4], silu[5]);
            o.w = qwen4_conv_v3_pack2<OutT>(silu[6], silu[7]);
            qwen4_conv_v4_st128_el(p_out, o, pol_el);

            // Safe L2 discard: p_cur has already been unpacked -> stored to SMEM ->
            // CTA-synced via __syncthreads() above -> consumed into st128_el, so all
            // 32 lanes finished reading p_cur with zero extra __syncwarp() needed.
            // t + 3 < cta_t1 && abs_t + 4 < total_tokens excludes both the next
            // CTA / next strip 3-row halo and the final 4-row conv_state tail.
            if (can_disc_in && (!coarse_disc_in || (step & 12u) == 0u)
                && t + 3u < cta_t1 && (row_offset + t) + 4u < total_tokens) {
                if (disc_lane) qwen4_conv_v4_discard_l2(p_cur);
            }
        }
        cur_raw = next_raw;
        p_cur += (unsigned long long)4u * input_stride;
        p_out += (unsigned long long)4u * output_stride;
        ring_head = (ring_head + 4u) & 15u;
    }

    // When row_offset < 3u (single-strip prefill), CTA cta_t0 == 0u (rev_by == 0)
    // is the CTA that reads entry conv_state; commit conv_state from it (after
    // __syncthreads()) so no other CTA races with entry conv_state reads.
    // Keep L2 discard of pt0..pt3 exclusively when cta_t1 == strip_tokens.
    if (row_offset + strip_tokens == total_tokens && total_tokens >= 4u) {
        const bool commit_cta = (row_offset < 3u) ? (cta_t0 == 0u) : (cta_t1 == strip_tokens);
        if (commit_cta) {
            __syncthreads();
            if (warp == 0u) {
                const unsigned int t0 = total_tokens - 4u;
                const __nv_bfloat16* pt0 =
                    new_input + (unsigned long long)(t0 + 0u) * input_stride + ch0;
                const __nv_bfloat16* pt1 = pt0 + input_stride;
                const __nv_bfloat16* pt2 = pt1 + input_stride;
                const __nv_bfloat16* pt3 = pt2 + input_stride;
                const uint4 tr0 = qwen4_conv_v3_ld128_nc(pt0);
                const uint4 tr1 = qwen4_conv_v3_ld128_nc(pt1);
                const uint4 tr2 = qwen4_conv_v3_ld128_nc(pt2);
                const uint4 tr3 = qwen4_conv_v3_ld128_nc(pt3);
                float tail[4][8];
                qwen4_conv_v3_unpack8(tr0, tail[0]);
                qwen4_conv_v3_unpack8(tr1, tail[1]);
                qwen4_conv_v3_unpack8(tr2, tail[2]);
                qwen4_conv_v3_unpack8(tr3, tail[3]);
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    float4 st;
                    st.x = tail[0][c];
                    st.y = tail[1][c];
                    st.z = tail[2][c];
                    st.w = tail[3][c];
                    *reinterpret_cast<float4*>(conv_state + (unsigned long long)(ch0 + c) * 4u) = st;
                }
                if (cta_t1 == strip_tokens && can_disc_in && base_cta >= 4u) {
                    __syncwarp();
                    if (disc_lane) {
                        qwen4_conv_v4_discard_l2(pt0);
                        qwen4_conv_v4_discard_l2(pt1);
                        qwen4_conv_v4_discard_l2(pt2);
                        qwen4_conv_v4_discard_l2(pt3);
                    }
                }
            }
        }
    }
}

#define QWEN4_FI_CONV_V4_STRIP_ENTRY(NAME, OUT_T)                                     \
    extern "C" __global__ void __launch_bounds__(128, 5) NAME(                        \
        float* __restrict__ conv_state,                                               \
        const __nv_bfloat16* __restrict__ new_input,                                  \
        const __nv_bfloat16* __restrict__ weight, const float* __restrict__ bias,     \
        OUT_T* __restrict__ output, unsigned int strip_tokens,                        \
        unsigned int row_offset, unsigned int total_tokens,                           \
        unsigned int qk_channels, float l2_eps,                                       \
        unsigned int input_stride, unsigned int output_stride,                        \
        unsigned int discard_flags) {                                                 \
        qwen4_conv1d_l2norm_16_v4_strip_body<OUT_T, false>(                           \
            conv_state, new_input, weight, bias, output, strip_tokens, row_offset,    \
            total_tokens, qk_channels, l2_eps, input_stride, output_stride,           \
            discard_flags, nullptr, 0u);                                              \
    }

#define QWEN4_FI_CONV_V4_STRIP_GATESIGHT_ENTRY(NAME, OUT_T)                           \
    extern "C" __global__ void __launch_bounds__(128, 4) NAME(                        \
        float* __restrict__ conv_state,                                               \
        const __nv_bfloat16* __restrict__ new_input,                                  \
        const __nv_bfloat16* __restrict__ weight, const float* __restrict__ bias,     \
        OUT_T* __restrict__ output, unsigned int strip_tokens,                        \
        unsigned int row_offset, unsigned int total_tokens,                           \
        unsigned int qk_channels, float l2_eps,                                       \
        unsigned int input_stride, unsigned int output_stride,                        \
        unsigned int discard_flags, __half* __restrict__ gatesight_out,               \
        unsigned int num_v_heads) {                                                   \
        qwen4_conv1d_l2norm_16_v4_strip_body<OUT_T, true>(                            \
            conv_state, new_input, weight, bias, output, strip_tokens, row_offset,    \
            total_tokens, qk_channels, l2_eps, input_stride, output_stride,           \
            discard_flags, gatesight_out, num_v_heads);                               \
    }

QWEN4_FI_CONV_V4_STRIP_ENTRY(qwen4_conv1d_l2norm_bf16_prefill_v4_strip, __nv_bfloat16)
QWEN4_FI_CONV_V4_STRIP_ENTRY(qwen4_conv1d_l2norm_f16_prefill_v4_strip, __half)
QWEN4_FI_CONV_V4_STRIP_GATESIGHT_ENTRY(qwen4_conv1d_l2norm_bf16_prefill_v4_strip_gatesight, __nv_bfloat16)
QWEN4_FI_CONV_V4_STRIP_GATESIGHT_ENTRY(qwen4_conv1d_l2norm_f16_prefill_v4_strip_gatesight, __half)
