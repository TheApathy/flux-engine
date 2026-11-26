// SPDX-License-Identifier: AGPL-3.0-only

// Fused Flash-Next hyperconnection prefill glue (ATLAS_QWEN4_PREFILL_MHC_FUSED).
//
// Every kernel here reproduces the arithmetic of the unfused kernels it
// replaces in `common/qwen4_hyper.cu` and `common/moe_permute.cu`
// expression by expression (this target builds with --fmad=false, so each
// product and sum is rounded separately), and every reduction reproduces the
// unfused thread/warp tree. The fused output is therefore byte-identical;
// only the intermediate DRAM round trips disappear.
//
// Geometry is the canonical Flash-Next one: hidden 2560, 4 streams.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

constexpr unsigned int kHidden = 2560;
constexpr unsigned int kStreams = 4;
constexpr unsigned int kWidth = kHidden * kStreams;
constexpr unsigned int kThreads = 256;
// qwen4_hc_group_norm runs one block of hidden.min(1024) threads per
// (token, stream). Its reduction tree is replayed with 4 virtual threads per
// real thread: virtual thread v = t + 256k sits in virtual warp w + 8k at
// lane t % 32, so a butterfly over the real warp's lanes is the virtual
// warp's butterfly.
constexpr unsigned int kNormThreads = 1024;
constexpr unsigned int kVirtualPerThread = kNormThreads / kThreads;
constexpr unsigned int kNormWarps = kNormThreads / 32;
constexpr unsigned int kChunk = 8;  // bf16 per 16-byte vector
constexpr unsigned int kChunksPerStream = kHidden / kChunk;
constexpr unsigned int kChunks = kWidth / kChunk;
constexpr unsigned int kChunksPerThread = kChunks / kThreads;

static_assert(kHidden % kChunk == 0, "stream rows must split into whole vectors");
static_assert(kChunks % kThreads == 0, "row vectors must split evenly over the block");
static_assert(kNormThreads % kThreads == 0, "virtual threads must split evenly");

// Helpers that pass shared-memory rows between threads take plain pointers:
// __restrict__ there lets the compiler cache values across the block barrier
// (it miscompiled the MoE combine row before it was removed).
__device__ __forceinline__ float hc_warp_sum(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(0xffffffff, v, d);
    return v;
}

union Vec8 {
    uint4 raw;
    __nv_bfloat16 h[kChunk];
};

// Per-stream inverse RMS of the bf16 row in `row` (smem), reproducing
// qwen4_hc_group_norm's per-thread order, butterfly and cross-warp sum.
__device__ __forceinline__ void hc_stream_rinv(
    const __nv_bfloat16* row,
    float (*partial)[kStreams],
    float* rinv,
    unsigned int hidden_size,
    float eps) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
#pragma unroll
    for (unsigned int k = 0; k < kVirtualPerThread; ++k) {
        const unsigned int v = threadIdx.x + k * kThreads;
#pragma unroll
        for (unsigned int s = 0; s < kStreams; ++s) {
            float sum = 0.0f;
            for (unsigned int i = v; i < hidden_size; i += kNormThreads) {
                const float x = __bfloat162float(row[s * kHidden + i]);
                sum += x * x;
            }
            sum = hc_warp_sum(sum);
            if (lane == 0) partial[warp + k * (kThreads / 32)][s] = sum;
        }
    }
    __syncthreads();
    if (warp < kStreams) {
        float v = partial[lane][warp];
        v = hc_warp_sum(v);
        if (lane == 0) rinv[warp] = rsqrtf(v / hidden_size + eps);
    }
    __syncthreads();
}

// hyper[t, s, h] = bf16(hyper + core[t, h] * scale[s]) for one token row,
// exactly as qwen4_hc_inject_saved; the updated row also lands in `row`
// (smem) and in `updated` (this thread's vectors, chunk c = tid + j*256).
__device__ __forceinline__ void hc_inject_row(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* core_row,
    const float (&scale)[kStreams],
    unsigned int token,
    __nv_bfloat16* row,
    Vec8 (&updated)[kChunksPerThread]) {
    const size_t base = (size_t)token * kWidth;
#pragma unroll
    for (unsigned int j = 0; j < kChunksPerThread; ++j) {
        const unsigned int c = threadIdx.x + j * kThreads;
        const unsigned int s = c / kChunksPerStream;
        const unsigned int h0 = (c - s * kChunksPerStream) * kChunk;
        Vec8 old, add;
        old.raw = *reinterpret_cast<const uint4*>(hyper + base + c * kChunk);
        add.raw = *reinterpret_cast<const uint4*>(core_row + h0);
#pragma unroll
        for (unsigned int e = 0; e < kChunk; ++e) {
            const float o = __bfloat162float(old.h[e]);
            const float a = __bfloat162float(add.h[e]);
            updated[j].h[e] = __float2bfloat16(o + a * scale[s]);
        }
        *reinterpret_cast<uint4*>(hyper + base + c * kChunk) = updated[j].raw;
        *reinterpret_cast<uint4*>(row + c * kChunk) = updated[j].raw;
    }
}

// The injecting hyperconnection's saved scales: the tail (R-4..R-1) of the
// token's residual row, read before anything overwrites it.
__device__ __forceinline__ void hc_saved_scales(
    const __nv_bfloat16* __restrict__ residual,
    unsigned int token,
    float (&scale)[kStreams]) {
#pragma unroll
    for (unsigned int s = 0; s < kStreams; ++s) {
        scale[s] = __bfloat162float(residual[(size_t)token * kWidth + (kWidth - kStreams) + s]);
    }
}

// rstd in qwen4_hc_rstd's order (the HC_FUSED preparation's own pass, in
// qwen4_hc_fused.cu): 64 threads per stream over 8-element vectors, element
// pairs summed before accumulating, a 32-lane butterfly, then the stream's two
// warps added. A 256-thread block, like that kernel's.
__device__ __forceinline__ void hc_stream_rstd_fused_order(
    const __nv_bfloat16* row,
    float* part8,
    float* rstd_out,
    unsigned int hidden,
    float eps) {
    const unsigned int s = threadIdx.x >> 6;
    const unsigned int l = threadIdx.x & 63;
    const uint4* srow = reinterpret_cast<const uint4*>(row + s * kHidden);
    float acc = 0.0f;
    for (unsigned int i = l; i < hidden / 8; i += 64) {
        const uint4 v = srow[i];
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(p[j]);
            acc += f.x * f.x + f.y * f.y;
        }
    }
    for (int d = 16; d; d >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, d);
    if ((threadIdx.x & 31) == 0) part8[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 4) {
        const float sum = part8[threadIdx.x * 2] + part8[threadIdx.x * 2 + 1];
        rstd_out[threadIdx.x] = rsqrtf(sum / (float)hidden + eps);
    }
}

// residual[t, s, h] = bf16(hyper * rinv[s] * (1 + weight)), qwen4_hc_group_norm's
// final expression, from this thread's updated vectors.
__device__ __forceinline__ void hc_store_normed(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const float* rinv,
    unsigned int token,
    const Vec8 (&updated)[kChunksPerThread]) {
    const size_t base = (size_t)token * kWidth;
#pragma unroll
    for (unsigned int j = 0; j < kChunksPerThread; ++j) {
        const unsigned int c = threadIdx.x + j * kThreads;
        const unsigned int s = c / kChunksPerStream;
        const float inv = rinv[s];
        Vec8 w, out;
        w.raw = *reinterpret_cast<const uint4*>(weight + c * kChunk);
#pragma unroll
        for (unsigned int e = 0; e < kChunk; ++e) {
            const float x = __bfloat162float(updated[j].h[e]);
            const float g = __bfloat162float(w.h[e]);
            out.h[e] = __float2bfloat16(x * inv * (1.0f + g));
        }
        *reinterpret_cast<uint4*>(residual + base + c * kChunk) = out.raw;
    }
}

constexpr unsigned int kMaxTopk = 16;

// moe_batched_blend's shared-expert gate for `token`: the same 256-thread
// strided dot, shfl_down tree and warp order (null gate = 1). `dot_partial`
// is block scratch; contains two block barriers; every thread gets the gate.
__device__ __forceinline__ float hc_blend_gate(
    const __nv_bfloat16* __restrict__ gate_input,
    const __nv_bfloat16* __restrict__ shared_gate,
    unsigned int token,
    unsigned int hidden_size,
    float* dot_partial) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    float local_dot = 0.0f;
    if (shared_gate != nullptr) {
        const __nv_bfloat16* x = gate_input + (size_t)token * hidden_size;
        for (unsigned int i = tid; i < hidden_size; i += kThreads) {
            const float n = __bfloat162float(x[i]);
            const float g = __bfloat162float(shared_gate[i]);
            local_dot += n * g;
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dot += __shfl_down_sync(0xFFFFFFFF, local_dot, offset);
    }
    if (lane == 0) dot_partial[tid / 32] = local_dot;
    __syncthreads();
    if (tid == 0) {
        float gate;
        if (shared_gate == nullptr) {
            gate = 1.0f;
        } else {
            float total = 0.0f;
            for (unsigned int w = 0; w < kThreads / 32; w++) total += dot_partial[w];
            gate = 1.0f / (1.0f + __expf(-total));
        }
        dot_partial[0] = gate;
    }
    __syncthreads();
    return dot_partial[0];
}

// MXFP8 slot rows (ATLAS fn-moe down epilogue, MX mode): payload [slots][2560]
// e4m3 bytes, scales [slots][80] e8m0 bytes (one per 32 columns), value =
// e4m3 * 2^(scale - 127). Four consecutive columns of one slot row as floats
// (exact: an e4m3 value times a power of two).
constexpr unsigned int kMxBlock = 32;
constexpr unsigned int kMxScalesPerRow = kHidden / kMxBlock;

__device__ __forceinline__ float4 hc_mx_load4(
    const uint8_t* __restrict__ payload,
    const uint8_t* __restrict__ scales,
    unsigned int slot,
    unsigned int h) {
    const uint32_t q = *reinterpret_cast<const uint32_t*>(payload + (size_t)slot * kHidden + h);
    const uint32_t e = scales[(size_t)slot * kMxScalesPerRow + h / kMxBlock];
    // 2^(e - 127) by exponent bits (e in [1, 254]; e = 0 is 2^-127, flushed like the epilogue's ftz).
    const float scale = __uint_as_float(e << 23);
    const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(q & 0xffffu), __NV_E4M3);
    const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(q >> 16), __NV_E4M3);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&lo));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&hi));
    return make_float4(a.x * scale, a.y * scale, b.x * scale, b.y * scale);
}

__device__ __forceinline__ float hc_e4m3_to_float(uint8_t q) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)q, __NV_E4M3);
    return __low2float(*reinterpret_cast<const __half2*>(&h));
}

__device__ __forceinline__ float4 hc_mx_load4_r2(
    const uint8_t* __restrict__ payload,
    const uint8_t* __restrict__ scales,
    const uint32_t* __restrict__ corrections,
    unsigned int slot,
    unsigned int h) {
    const uint32_t q = *reinterpret_cast<const uint32_t*>(payload + (size_t)slot * kHidden + h);
    const size_t group = (size_t)slot * kMxScalesPerRow + h / kMxBlock;
    const float scale = __uint_as_float((uint32_t)scales[group] << 23);
    const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(q & 0xffffu), __NV_E4M3);
    const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(q >> 16), __NV_E4M3);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&lo));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&hi));
    float value[4] = {a.x, a.y, b.x, b.y};
    const uint32_t packed = corrections[group];
    const uint32_t rec[2] = {packed & 0xffffu, packed >> 16};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const unsigned int lane = rec[r] & 31u;
        const uint8_t residual = (uint8_t)((rec[r] >> 5) & 0xffu);
        if ((residual & 0x7fu) != 0 && lane >= (h & 31u) && lane < (h & 31u) + 4u)
            value[lane - (h & 31u)] = __fadd_rn(
                value[lane - (h & 31u)], hc_e4m3_to_float(residual));
    }
    return make_float4(value[0] * scale, value[1] * scale, value[2] * scale, value[3] * scale);
}

// The MoE output row of `token` into `core` (smem), exactly as
// moe_unpermute_reduce_indexed then moe_batched_blend form it (the blend's
// 256-thread strided dot, shfl_down tree and warp order). `perm`/`wts`/
// `dot_partial` are block-shared scratch. Ends with a block barrier.
__device__ __forceinline__ void hc_moe_core_row(
    const __nv_bfloat16* __restrict__ expert_output,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ gate_input,
    const __nv_bfloat16* __restrict__ shared_gate,
    unsigned int token,
    unsigned int hidden_size,
    unsigned int topk,
    __nv_bfloat16* core,
    int* perm,
    float* wts,
    float* dot_partial) {
    const unsigned int tid = threadIdx.x;
    if (tid < topk) {
        perm[tid] = token_to_perm[token * topk + tid];
        wts[tid] = topk_weights[token * topk + tid];
    }
    const float gate = hc_blend_gate(gate_input, shared_gate, token, hidden_size, dot_partial);

    // Unpermute + blend into this token's core row (smem), two h per thread step.
    const __nv_bfloat16* shared_row = shared_out + (size_t)token * hidden_size;
    for (unsigned int pair = tid; pair < kHidden / 2; pair += kThreads) {
        const unsigned int h = pair * 2;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (unsigned int k = 0; k < topk; k++) {
            const float w = wts[k];
            const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(
                expert_output + (size_t)perm[k] * hidden_size + h);
            acc0 += w * __bfloat162float(v.x);
            acc1 += w * __bfloat162float(v.y);
        }
        const __nv_bfloat162 sh = *reinterpret_cast<const __nv_bfloat162*>(shared_row + h);
        const float o0 = __bfloat162float(__float2bfloat16(acc0));
        const float o1 = __bfloat162float(__float2bfloat16(acc1));
        __nv_bfloat162 out;
        out.x = __float2bfloat16(o0 + gate * __bfloat162float(sh.x));
        out.y = __float2bfloat16(o1 + gate * __bfloat162float(sh.y));
        *reinterpret_cast<__nv_bfloat162*>(core + h) = out;
    }
    __syncthreads();
}

// The MoE output row of `token` into `core` (smem) from MXFP8 slot rows:
// moe_unpermute_reduce_indexed's weighted sum in its k order over the
// dequantized slot values, then moe_batched_blend. Four columns per thread
// step (32 lanes read 128 contiguous payload bytes of a slot row). Ends with
// a block barrier.
__device__ __forceinline__ void hc_moe_core_row_mx(
    const uint8_t* __restrict__ payload,
    const uint8_t* __restrict__ scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ gate_input,
    const __nv_bfloat16* __restrict__ shared_gate,
    unsigned int token,
    unsigned int hidden_size,
    unsigned int topk,
    __nv_bfloat16* core,
    int* perm,
    float* wts,
    float* dot_partial) {
    const unsigned int tid = threadIdx.x;
    if (tid < topk) {
        perm[tid] = token_to_perm[token * topk + tid];
        wts[tid] = topk_weights[token * topk + tid];
    }
    const float gate = hc_blend_gate(gate_input, shared_gate, token, hidden_size, dot_partial);
    const __nv_bfloat16* shared_row = shared_out + (size_t)token * hidden_size;
    for (unsigned int quad = tid; quad < kHidden / 4; quad += kThreads) {
        const unsigned int h = quad * 4;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned int k = 0; k < topk; k++) {
            const float w = wts[k];
            const float4 v = hc_mx_load4(payload, scales, (unsigned int)perm[k], h);
            acc[0] += w * v.x;
            acc[1] += w * v.y;
            acc[2] += w * v.z;
            acc[3] += w * v.w;
        }
        const uint2 shraw = *reinterpret_cast<const uint2*>(shared_row + h);
        const __nv_bfloat162* sh = reinterpret_cast<const __nv_bfloat162*>(&shraw);
        const float shv[4] = {__low2float(sh[0]), __high2float(sh[0]), __low2float(sh[1]), __high2float(sh[1])};
        __align__(8) __nv_bfloat16 out[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __bfloat162float(__float2bfloat16(acc[j]));
            out[j] = __float2bfloat16(o + gate * shv[j]);
        }
        *reinterpret_cast<uint2*>(core + h) = *reinterpret_cast<const uint2*>(out);
    }
    __syncthreads();
}

__device__ __forceinline__ void hc_moe_core_row_mx_r2(
    const uint8_t* __restrict__ payload,
    const uint8_t* __restrict__ scales,
    const uint32_t* __restrict__ corrections,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ gate_input,
    const __nv_bfloat16* __restrict__ shared_gate,
    unsigned int token,
    unsigned int hidden_size,
    unsigned int topk,
    __nv_bfloat16* core,
    int* perm,
    float* wts,
    float* dot_partial) {
    const unsigned int tid = threadIdx.x;
    if (tid < topk) {
        perm[tid] = token_to_perm[token * topk + tid];
        wts[tid] = topk_weights[token * topk + tid];
    }
    const float gate = hc_blend_gate(gate_input, shared_gate, token, hidden_size, dot_partial);
    const __nv_bfloat16* shared_row = shared_out + (size_t)token * hidden_size;
    for (unsigned int quad = tid; quad < kHidden / 4; quad += kThreads) {
        const unsigned int h = quad * 4;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned int k = 0; k < topk; k++) {
            const float w = wts[k];
            const float4 v = hc_mx_load4_r2(payload, scales, corrections, (unsigned int)perm[k], h);
            acc[0] += w * v.x;
            acc[1] += w * v.y;
            acc[2] += w * v.z;
            acc[3] += w * v.w;
        }
        const uint2 shraw = *reinterpret_cast<const uint2*>(shared_row + h);
        const __nv_bfloat162* sh = reinterpret_cast<const __nv_bfloat162*>(&shraw);
        const float shv[4] = {__low2float(sh[0]), __high2float(sh[0]), __low2float(sh[1]), __high2float(sh[1])};
        __align__(8) __nv_bfloat16 out[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __bfloat162float(__float2bfloat16(acc[j]));
            out[j] = __float2bfloat16(o + gate * shv[j]);
        }
        *reinterpret_cast<uint2*>(core + h) = *reinterpret_cast<const uint2*>(out);
    }
    __syncthreads();
}

}  // namespace

// Pending core injection + the next hyperconnection's group RMS norm:
//   hyper[t, s, h]    = bf16(hyper + core[t, h] * scale[t, s])   (qwen4_hc_inject_saved)
//   residual[t, s, h] = bf16(hyper * rinv[t, s] * (1 + weight))  (qwen4_hc_group_norm)
// Grid: (num_tokens). Block: 256.
extern "C" __global__ void __launch_bounds__(kThreads, 3)
qwen4_hc_inject_norm(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    unsigned int num_tokens,
    unsigned int hidden_size,
    float eps) {
    if (hidden_size != kHidden) __trap();
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    __shared__ __align__(16) __nv_bfloat16 row[kWidth];
    __shared__ float partial[kNormWarps][kStreams];
    __shared__ float rinv[kStreams];

    float scale[kStreams];
    hc_saved_scales(residual, token, scale);
    Vec8 updated[kChunksPerThread];
    hc_inject_row(hyper, core + (size_t)token * kHidden, scale, token, row, updated);
    __syncthreads();
    hc_stream_rinv(row, partial, rinv, hidden_size, eps);

    hc_store_normed(residual, weight, rinv, token, updated);
}

// Pending core injection + only the per-(token, stream) inverse RMS, for the
// HC_FUSED preparation (ATLAS_QWEN4_PREFILL_HC_FUSED), which recomputes the
// normalized rows from hyper and rstd. rstd follows that preparation's own
// qwen4_hc_rstd order, so the composed path is byte-identical to
// qwen4_hc_inject_saved + qwen4_hc_rstd. Grid: (num_tokens). Block: 256.
extern "C" __global__ void __launch_bounds__(kThreads, 3)
qwen4_hc_inject_rstd(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    const __nv_bfloat16* __restrict__ residual,
    float* __restrict__ rstd,
    unsigned int num_tokens,
    unsigned int hidden_size,
    float eps) {
    if (hidden_size != kHidden) __trap();
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    __shared__ __align__(16) __nv_bfloat16 row[kWidth];
    __shared__ float part8[kThreads / 32];

    float scale[kStreams];
    hc_saved_scales(residual, token, scale);
    Vec8 updated[kChunksPerThread];
    hc_inject_row(hyper, core + (size_t)token * kHidden, scale, token, row, updated);
    __syncthreads();
    hc_stream_rstd_fused_order(row, part8, rstd + (size_t)token * kStreams, hidden_size, eps);
}

// Pending MoE output (routed experts not yet unpermuted, shared expert not yet
// blended) + its injection + the next hyperconnection's group RMS norm, one
// pass per token. Reproduces, in order and expression by expression:
//   moe_unpermute_reduce_indexed  out[h] = bf16(sum_k w_k * expert[perm_k, h])
//   moe_batched_blend             out[h] = bf16(out + sigmoid(x . gate_w) * shared[h])
//                                 (its 256-thread shfl_down dot-product tree)
//   qwen4_hc_inject_saved         hyper = bf16(hyper + out * scale[s])
//   qwen4_hc_group_norm           residual = bf16(hyper * rinv[s] * (1 + weight))
// `gate_input` is the MoE input row (the blend's `normed`); `shared_gate` may
// be null (gate 1). Grid: (num_tokens). Block: 256.
extern "C" __global__ void __launch_bounds__(kThreads, 3)
qwen4_hc_combine_inject_norm(
    __nv_bfloat16* __restrict__ hyper,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ expert_output,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ gate_input,
    const __nv_bfloat16* __restrict__ shared_gate,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk,
    float eps) {
    if (hidden_size != kHidden || topk > kMaxTopk) __trap();
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    __shared__ __align__(16) __nv_bfloat16 row[kWidth];
    __shared__ __align__(16) __nv_bfloat16 core[kHidden];
    __shared__ float partial[kNormWarps][kStreams];
    __shared__ float rinv[kStreams];
    __shared__ float dot_partial[kThreads / 32];
    __shared__ int perm[kMaxTopk];
    __shared__ float wts[kMaxTopk];
    hc_moe_core_row(expert_output, token_to_perm, topk_weights, shared_out, gate_input,
                    shared_gate, token, hidden_size, topk, core, perm, wts, dot_partial);

    float scale[kStreams];
    hc_saved_scales(residual, token, scale);
    Vec8 updated[kChunksPerThread];
    hc_inject_row(hyper, core, scale, token, row, updated);
    __syncthreads();
    hc_stream_rinv(row, partial, rinv, hidden_size, eps);
    hc_store_normed(residual, weight, rinv, token, updated);
}

// The same deferred MoE output + injection for the HC_FUSED preparation:
// forms the output rows, injects them, and writes the per-(token, stream)
// rstd in qwen4_hc_rstd's order (no normalized rows), byte-identical to
// unpermute + blend + qwen4_hc_inject_saved + qwen4_hc_rstd.
// Grid: (num_tokens). Block: 256.
extern "C" __global__ void __launch_bounds__(kThreads, 3)
qwen4_hc_combine_inject_rstd(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ residual,
    float* __restrict__ rstd,
    const __nv_bfloat16* __restrict__ expert_output,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ gate_input,
    const __nv_bfloat16* __restrict__ shared_gate,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk,
    float eps) {
    if (hidden_size != kHidden || topk > kMaxTopk) __trap();
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    __shared__ __align__(16) __nv_bfloat16 row[kWidth];
    __shared__ __align__(16) __nv_bfloat16 core[kHidden];
    __shared__ float part8[kThreads / 32];
    __shared__ float dot_partial[kThreads / 32];
    __shared__ int perm[kMaxTopk];
    __shared__ float wts[kMaxTopk];
    hc_moe_core_row(expert_output, token_to_perm, topk_weights, shared_out, gate_input,
                    shared_gate, token, hidden_size, topk, core, perm, wts, dot_partial);

    float scale[kStreams];
    hc_saved_scales(residual, token, scale);
    Vec8 updated[kChunksPerThread];
    hc_inject_row(hyper, core, scale, token, row, updated);
    __syncthreads();
    hc_stream_rstd_fused_order(row, part8, rstd + (size_t)token * kStreams, hidden_size, eps);
}

// qwen4_hc_combine_inject_rstd reading MXFP8 slot rows (half the gather
// bytes). Grid: (num_tokens). Block: 256.
extern "C" __global__ void __launch_bounds__(kThreads, 3)
qwen4_hc_combine_mx_inject_rstd(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ residual,
    float* __restrict__ rstd,
    const uint8_t* __restrict__ slot_payload,
    const uint8_t* __restrict__ slot_scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ gate_input,
    const __nv_bfloat16* __restrict__ shared_gate,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk,
    float eps) {
    if (hidden_size != kHidden || topk > kMaxTopk) __trap();
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    __shared__ __align__(16) __nv_bfloat16 row[kWidth];
    __shared__ __align__(16) __nv_bfloat16 core[kHidden];
    __shared__ float part8[kThreads / 32];
    __shared__ float dot_partial[kThreads / 32];
    __shared__ int perm[kMaxTopk];
    __shared__ float wts[kMaxTopk];
    hc_moe_core_row_mx(slot_payload, slot_scales, token_to_perm, topk_weights, shared_out, gate_input,
                       shared_gate, token, hidden_size, topk, core, perm, wts, dot_partial);

    float scale[kStreams];
    hc_saved_scales(residual, token, scale);
    Vec8 updated[kChunksPerThread];
    hc_inject_row(hyper, core, scale, token, row, updated);
    __syncthreads();
    hc_stream_rstd_fused_order(row, part8, rstd + (size_t)token * kStreams, hidden_size, eps);
}

extern "C" __global__ void __launch_bounds__(kThreads, 3)
qwen4_hc_combine_mx_r2_inject_rstd(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ residual,
    float* __restrict__ rstd,
    const uint8_t* __restrict__ slot_payload,
    const uint8_t* __restrict__ slot_scales,
    const uint32_t* __restrict__ slot_corrections,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ gate_input,
    const __nv_bfloat16* __restrict__ shared_gate,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk,
    float eps) {
    if (hidden_size != kHidden || topk > kMaxTopk || blockDim.x != kThreads ||
        gridDim.x != num_tokens || hyper == nullptr || residual == nullptr || rstd == nullptr ||
        slot_payload == nullptr || slot_scales == nullptr || slot_corrections == nullptr ||
        token_to_perm == nullptr || topk_weights == nullptr || shared_out == nullptr ||
        (shared_gate != nullptr && gate_input == nullptr) ||
        (reinterpret_cast<uintptr_t>(hyper) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(residual) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(shared_out) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(slot_payload) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(slot_corrections) & 3u) != 0) __trap();
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    __shared__ __align__(16) __nv_bfloat16 row[kWidth];
    __shared__ __align__(16) __nv_bfloat16 core[kHidden];
    __shared__ float part8[kThreads / 32];
    __shared__ float dot_partial[kThreads / 32];
    __shared__ int perm[kMaxTopk];
    __shared__ float wts[kMaxTopk];
    hc_moe_core_row_mx_r2(slot_payload, slot_scales, slot_corrections, token_to_perm, topk_weights,
                          shared_out, gate_input, shared_gate, token, hidden_size, topk, core, perm, wts,
                          dot_partial);

    float scale[kStreams];
    hc_saved_scales(residual, token, scale);
    Vec8 updated[kChunksPerThread];
    hc_inject_row(hyper, core, scale, token, row, updated);
    __syncthreads();
    hc_stream_rstd_fused_order(row, part8, rstd + (size_t)token * kStreams, hidden_size, eps);
}

// moe_unpermute_reduce_indexed over MXFP8 slot rows into bf16 token rows
// [num_tokens, 2560] (for consumers that form the MoE output unfused; the
// blend follows with moe_batched_blend). Grid: (num_tokens). Block: 256.
extern "C" __global__ void __launch_bounds__(kThreads)
qwen4_hc_moe_unpermute_mx(
    const uint8_t* __restrict__ slot_payload,
    const uint8_t* __restrict__ slot_scales,
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk) {
    if (hidden_size != kHidden || topk > kMaxTopk) __trap();
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    __shared__ int perm[kMaxTopk];
    __shared__ float wts[kMaxTopk];
    if (threadIdx.x < topk) {
        perm[threadIdx.x] = token_to_perm[token * topk + threadIdx.x];
        wts[threadIdx.x] = topk_weights[token * topk + threadIdx.x];
    }
    __syncthreads();
    for (unsigned int quad = threadIdx.x; quad < kHidden / 4; quad += kThreads) {
        const unsigned int h = quad * 4;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned int k = 0; k < topk; k++) {
            const float w = wts[k];
            const float4 v = hc_mx_load4(slot_payload, slot_scales, (unsigned int)perm[k], h);
            acc[0] += w * v.x;
            acc[1] += w * v.y;
            acc[2] += w * v.z;
            acc[3] += w * v.w;
        }
        __align__(8) __nv_bfloat16 o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = __float2bfloat16(acc[j]);
        *reinterpret_cast<uint2*>(out + (size_t)token * hidden_size + h) = *reinterpret_cast<const uint2*>(o);
    }
}

extern "C" __global__ void __launch_bounds__(kThreads)
qwen4_hc_moe_unpermute_mx_r2(
    const uint8_t* __restrict__ slot_payload,
    const uint8_t* __restrict__ slot_scales,
    const uint32_t* __restrict__ slot_corrections,
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk) {
    if (hidden_size != kHidden || topk > kMaxTopk || blockDim.x != kThreads ||
        gridDim.x != num_tokens || slot_payload == nullptr || slot_scales == nullptr ||
        slot_corrections == nullptr || out == nullptr || token_to_perm == nullptr ||
        topk_weights == nullptr ||
        (reinterpret_cast<uintptr_t>(out) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(slot_payload) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(slot_corrections) & 3u) != 0) __trap();
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    __shared__ int perm[kMaxTopk];
    __shared__ float wts[kMaxTopk];
    if (threadIdx.x < topk) {
        perm[threadIdx.x] = token_to_perm[token * topk + threadIdx.x];
        wts[threadIdx.x] = topk_weights[token * topk + threadIdx.x];
    }
    __syncthreads();
    for (unsigned int quad = threadIdx.x; quad < kHidden / 4; quad += kThreads) {
        const unsigned int h = quad * 4;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned int k = 0; k < topk; k++) {
            const float w = wts[k];
            const float4 v = hc_mx_load4_r2(slot_payload, slot_scales, slot_corrections,
                                            (unsigned int)perm[k], h);
            acc[0] += w * v.x;
            acc[1] += w * v.y;
            acc[2] += w * v.z;
            acc[3] += w * v.w;
        }
        __align__(8) __nv_bfloat16 o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = __float2bfloat16(acc[j]);
        *reinterpret_cast<uint2*>(out + (size_t)token * hidden_size + h) = *reinterpret_cast<const uint2*>(o);
    }
}

namespace {

__device__ __forceinline__ void hc_moe_core_row_gatepre(
    const __nv_bfloat16* __restrict__ expert_output,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int token,
    unsigned int hidden_size,
    unsigned int topk,
    __nv_bfloat16* core,
    int* perm,
    float* wts) {
    const unsigned int tid = threadIdx.x;
    if (tid < topk) {
        perm[tid] = token_to_perm[token * topk + tid];
        wts[tid] = topk_weights[token * topk + tid];
    }
    __syncthreads();
    const float gate = gate_scalars[token];

    const __nv_bfloat16* shared_row = shared_out + (size_t)token * hidden_size;
    for (unsigned int pair = tid; pair < kHidden / 2; pair += kThreads) {
        const unsigned int h = pair * 2;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (unsigned int k = 0; k < topk; k++) {
            const float w = wts[k];
            const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(
                expert_output + (size_t)perm[k] * hidden_size + h);
            acc0 += w * __bfloat162float(v.x);
            acc1 += w * __bfloat162float(v.y);
        }
        const __nv_bfloat162 sh = *reinterpret_cast<const __nv_bfloat162*>(shared_row + h);
        const float o0 = __bfloat162float(__float2bfloat16(acc0));
        const float o1 = __bfloat162float(__float2bfloat16(acc1));
        __nv_bfloat162 out;
        out.x = __float2bfloat16(o0 + gate * __bfloat162float(sh.x));
        out.y = __float2bfloat16(o1 + gate * __bfloat162float(sh.y));
        *reinterpret_cast<__nv_bfloat162*>(core + h) = out;
    }
    __syncthreads();
}

__device__ __forceinline__ void hc_moe_core_row_mx_gatepre(
    const uint8_t* __restrict__ payload,
    const uint8_t* __restrict__ scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int token,
    unsigned int hidden_size,
    unsigned int topk,
    __nv_bfloat16* core,
    int* perm,
    float* wts) {
    const unsigned int tid = threadIdx.x;
    if (tid < topk) {
        perm[tid] = token_to_perm[token * topk + tid];
        wts[tid] = topk_weights[token * topk + tid];
    }
    __syncthreads();
    const float gate = gate_scalars[token];
    const __nv_bfloat16* shared_row = shared_out + (size_t)token * hidden_size;
    for (unsigned int quad = tid; quad < kHidden / 4; quad += kThreads) {
        const unsigned int h = quad * 4;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned int k = 0; k < topk; k++) {
            const float w = wts[k];
            const float4 v = hc_mx_load4(payload, scales, (unsigned int)perm[k], h);
            acc[0] += w * v.x;
            acc[1] += w * v.y;
            acc[2] += w * v.z;
            acc[3] += w * v.w;
        }
        const uint2 shraw = *reinterpret_cast<const uint2*>(shared_row + h);
        const __nv_bfloat162* sh = reinterpret_cast<const __nv_bfloat162*>(&shraw);
        const float shv[4] = {__low2float(sh[0]), __high2float(sh[0]), __low2float(sh[1]), __high2float(sh[1])};
        __align__(8) __nv_bfloat16 out[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __bfloat162float(__float2bfloat16(acc[j]));
            out[j] = __float2bfloat16(o + gate * shv[j]);
        }
        *reinterpret_cast<uint2*>(core + h) = *reinterpret_cast<const uint2*>(out);
    }
    __syncthreads();
}

__device__ __forceinline__ void hc_moe_core_row_mx_r2_gatepre(
    const uint8_t* __restrict__ payload,
    const uint8_t* __restrict__ scales,
    const uint32_t* __restrict__ corrections,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int token,
    unsigned int hidden_size,
    unsigned int topk,
    __nv_bfloat16* core,
    int* perm,
    float* wts) {
    const unsigned int tid = threadIdx.x;
    if (tid < topk) {
        perm[tid] = token_to_perm[token * topk + tid];
        wts[tid] = topk_weights[token * topk + tid];
    }
    __syncthreads();
    const float gate = gate_scalars[token];
    const __nv_bfloat16* shared_row = shared_out + (size_t)token * hidden_size;
    for (unsigned int quad = tid; quad < kHidden / 4; quad += kThreads) {
        const unsigned int h = quad * 4;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned int k = 0; k < topk; k++) {
            const float w = wts[k];
            const float4 v = hc_mx_load4_r2(payload, scales, corrections, (unsigned int)perm[k], h);
            acc[0] += w * v.x;
            acc[1] += w * v.y;
            acc[2] += w * v.z;
            acc[3] += w * v.w;
        }
        const uint2 shraw = *reinterpret_cast<const uint2*>(shared_row + h);
        const __nv_bfloat162* sh = reinterpret_cast<const __nv_bfloat162*>(&shraw);
        const float shv[4] = {__low2float(sh[0]), __high2float(sh[0]), __low2float(sh[1]), __high2float(sh[1])};
        __align__(8) __nv_bfloat16 out[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __bfloat162float(__float2bfloat16(acc[j]));
            out[j] = __float2bfloat16(o + gate * shv[j]);
        }
        *reinterpret_cast<uint2*>(core + h) = *reinterpret_cast<const uint2*>(out);
    }
    __syncthreads();
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads, 3)
qwen4_hc_combine_inject_norm_gatepre(
    __nv_bfloat16* __restrict__ hyper,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ expert_output,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk,
    float eps) {
    if (hidden_size != kHidden || topk > kMaxTopk) __trap();
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    __shared__ __align__(16) __nv_bfloat16 row[kWidth];
    __shared__ __align__(16) __nv_bfloat16 core[kHidden];
    __shared__ float partial[kNormWarps][kStreams];
    __shared__ float rinv[kStreams];
    __shared__ int perm[kMaxTopk];
    __shared__ float wts[kMaxTopk];
    hc_moe_core_row_gatepre(expert_output, token_to_perm, topk_weights, shared_out, gate_scalars,
                            token, hidden_size, topk, core, perm, wts);

    float scale[kStreams];
    hc_saved_scales(residual, token, scale);
    Vec8 updated[kChunksPerThread];
    hc_inject_row(hyper, core, scale, token, row, updated);
    __syncthreads();
    hc_stream_rinv(row, partial, rinv, hidden_size, eps);
    hc_store_normed(residual, weight, rinv, token, updated);
}

extern "C" __global__ void __launch_bounds__(kThreads, 3)
qwen4_hc_combine_inject_rstd_gatepre(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ residual,
    float* __restrict__ rstd,
    const __nv_bfloat16* __restrict__ expert_output,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk,
    float eps) {
    if (hidden_size != kHidden || topk > kMaxTopk) __trap();
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    __shared__ __align__(16) __nv_bfloat16 row[kWidth];
    __shared__ __align__(16) __nv_bfloat16 core[kHidden];
    __shared__ float part8[kThreads / 32];
    __shared__ int perm[kMaxTopk];
    __shared__ float wts[kMaxTopk];
    hc_moe_core_row_gatepre(expert_output, token_to_perm, topk_weights, shared_out, gate_scalars,
                            token, hidden_size, topk, core, perm, wts);

    float scale[kStreams];
    hc_saved_scales(residual, token, scale);
    Vec8 updated[kChunksPerThread];
    hc_inject_row(hyper, core, scale, token, row, updated);
    __syncthreads();
    hc_stream_rstd_fused_order(row, part8, rstd + (size_t)token * kStreams, hidden_size, eps);
}

extern "C" __global__ void __launch_bounds__(kThreads, 3)
qwen4_hc_combine_mx_inject_rstd_gatepre(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ residual,
    float* __restrict__ rstd,
    const uint8_t* __restrict__ slot_payload,
    const uint8_t* __restrict__ slot_scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk,
    float eps) {
    if (hidden_size != kHidden || topk > kMaxTopk) __trap();
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    __shared__ __align__(16) __nv_bfloat16 row[kWidth];
    __shared__ __align__(16) __nv_bfloat16 core[kHidden];
    __shared__ float part8[kThreads / 32];
    __shared__ int perm[kMaxTopk];
    __shared__ float wts[kMaxTopk];
    hc_moe_core_row_mx_gatepre(slot_payload, slot_scales, token_to_perm, topk_weights, shared_out,
                               gate_scalars, token, hidden_size, topk, core, perm, wts);

    float scale[kStreams];
    hc_saved_scales(residual, token, scale);
    Vec8 updated[kChunksPerThread];
    hc_inject_row(hyper, core, scale, token, row, updated);
    __syncthreads();
    hc_stream_rstd_fused_order(row, part8, rstd + (size_t)token * kStreams, hidden_size, eps);
}

extern "C" __global__ void __launch_bounds__(kThreads, 3)
qwen4_hc_combine_mx_r2_inject_rstd_gatepre(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ residual,
    float* __restrict__ rstd,
    const uint8_t* __restrict__ slot_payload,
    const uint8_t* __restrict__ slot_scales,
    const uint32_t* __restrict__ slot_corrections,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk,
    float eps) {
    if (hidden_size != kHidden || topk > kMaxTopk || blockDim.x != kThreads ||
        gridDim.x != num_tokens || hyper == nullptr || residual == nullptr || rstd == nullptr ||
        slot_payload == nullptr || slot_scales == nullptr || slot_corrections == nullptr ||
        token_to_perm == nullptr || topk_weights == nullptr || shared_out == nullptr ||
        gate_scalars == nullptr ||
        (reinterpret_cast<uintptr_t>(hyper) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(residual) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(shared_out) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(slot_payload) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(slot_corrections) & 3u) != 0) __trap();
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    __shared__ __align__(16) __nv_bfloat16 row[kWidth];
    __shared__ __align__(16) __nv_bfloat16 core[kHidden];
    __shared__ float part8[kThreads / 32];
    __shared__ int perm[kMaxTopk];
    __shared__ float wts[kMaxTopk];
    hc_moe_core_row_mx_r2_gatepre(slot_payload, slot_scales, slot_corrections, token_to_perm, topk_weights,
                                  shared_out, gate_scalars, token, hidden_size, topk, core, perm, wts);

    float scale[kStreams];
    hc_saved_scales(residual, token, scale);
    Vec8 updated[kChunksPerThread];
    hc_inject_row(hyper, core, scale, token, row, updated);
    __syncthreads();
    hc_stream_rstd_fused_order(row, part8, rstd + (size_t)token * kStreams, hidden_size, eps);
}

