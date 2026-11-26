// SPDX-License-Identifier: AGPL-3.0-only

// Flash-Next hyperconnection up projection with the mix in its epilogue
// (ATLAS_QWEN4_PREFILL_MHC_FUSED=upmix). Replaces, on the full-row fast HC
// path, the cuBLASLt up GEMM (10240-wide mix logits written to DRAM) and
// qwen4_hc_mix (logits read back):
//
//   logit[t, s*H + h] = bf16(alpha * sum_k act[t, k] * up[s*H + h, k])
//   mixed[t, h]       = bf16(sum_s sigmoid(logit) * normed[t, s*H + h] / hc)
//   inject[t, j]      = bf16(2 sigmoid(inject[t, j] / hc))
//
// The epilogue repeats qwen4_hc_mix expression by expression (this target
// builds with --fmad=false), and the GEMM accumulates each logit in one fp32
// register chain over ascending k16 mma.sync steps. Whether that chain is the
// one cuBLASLt's kernel uses is what the parity harness checks.
//
// CTA tile: 64 tokens x 32 h positions x 4 streams. Logit columns are
// stream-major (c = s*32 + h), so every thread holds all four streams of the
// (token, h) pairs it owns and sums them in stream order, as qwen4_hc_mix.
// Grid (H/32, ceil(T/64)), Block 256: 8 warps = 2 along tokens (32 rows
// each) x 4 along h (8 positions each, one n8 MMA fragment per stream).

#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int kRank = 320;
constexpr int kHidden = 2560;
constexpr int kStreams = 4;
constexpr int kBM = 64;
constexpr int kBH = 32;
constexpr int kBN = kBH * kStreams;
constexpr int kBK = 32;
constexpr int kLds = kBK + 8;  // 80-byte smem rows: conflict-free ldmatrix
constexpr int kStages = kRank / kBK;
constexpr int kThreads = 256;

static_assert(kRank % kBK == 0, "rank must split into whole K stages");
static_assert(kHidden % kBH == 0, "hidden must split into whole h tiles");

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    const unsigned int s = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(unsigned int (&r)[4], const void* p) {
    const unsigned int s = static_cast<unsigned int>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const unsigned int (&a)[4],
                                         unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One CTA tile of the up projection: acc[i][s][e] holds logit columns
// s*2560 + h0 + wh*8 + 2*(lane%4) (+1) of rows wm*32 + i*16 + lane/4 (+8),
// each a single fp32 chain over ascending k16 mma.sync steps (cuBLASLt's
// order for this shape, see the parity harness).
__device__ __forceinline__ void upmix_mainloop(
    const __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ up,
    unsigned int num_tokens,
    int t0,
    int h0,
    __nv_bfloat16 (*sA)[kBM][kLds],
    __nv_bfloat16 (*sB)[kBN][kLds],
    float (&acc)[2][kStreams][4]) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp >> 2;
    const int wh = warp & 3;
    auto load_stage = [&](int buf, int k0) {
        {
            const int row = tid >> 2;  // 64 rows x 4 chunks of 8
            const int part = tid & 3;
            const int t = t0 + row;
            const bool ok = t < (int)num_tokens;
            cp_async16(&sA[buf][row][part * 8],
                       act + (size_t)(ok ? t : 0) * kRank + k0 + part * 8, ok);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int c = (tid >> 2) + j * 64;  // logit column, stream-major
            const int part = tid & 3;
            const int n = (c / kBH) * kHidden + h0 + (c % kBH);
            cp_async16(&sB[buf][c][part * 8], up + (size_t)n * kRank + k0 + part * 8, true);
        }
        cp_async_commit();
    };
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int s = 0; s < kStreams; ++s)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][s][e] = 0.0f;

    load_stage(0, 0);
    for (int stage = 0; stage < kStages; ++stage) {
        const int buf = stage & 1;
        if (stage + 1 < kStages) {
            load_stage(buf ^ 1, (stage + 1) * kBK);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 16) {
            unsigned int a[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                ldmatrix_x4(a[i], &sA[buf][wm * 32 + i * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
            }
            unsigned int b[kStreams][2];
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                // Lanes 0-15 address stream 2p, lanes 16-31 stream 2p+1; each
                // 16-lane half gives the fragment's k and k+8 halves.
                const int s = 2 * p + (lane >> 4);
                const int c = s * kBH + wh * 8 + (lane & 7);
                unsigned int r[4];
                ldmatrix_x4(r, &sB[buf][c][kk + ((lane >> 3) & 1) * 8]);
                b[2 * p][0] = r[0];
                b[2 * p][1] = r[1];
                b[2 * p + 1][0] = r[2];
                b[2 * p + 1][1] = r[3];
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int s = 0; s < kStreams; ++s) mma_bf16(acc[i][s], a[i], b[s][0], b[s][1]);
        }
        __syncthreads();
    }
}

}  // namespace

// Exact lever (`upmix`): the fast path's cuBLASLt up GEMM + qwen4_hc_mix.
// Grid (H/32, ceil(T/64)), Block 256.
extern "C" __global__ void __launch_bounds__(kThreads)
qwen4_hc_upmix(
    const __nv_bfloat16* __restrict__ act,     // [T, 320] silu'd down projection
    const __nv_bfloat16* __restrict__ up,      // [4*2560, 320] dequantized LUT*fp8
    float alpha,                               // up scale2 (the cuBLASLt alpha)
    const __nv_bfloat16* __restrict__ normed,  // [T, 4*2560] group-normalized rows
    __nv_bfloat16* __restrict__ inject,        // [T, 4] logits -> scales, in place
    __nv_bfloat16* __restrict__ mixed,         // [T, 2560]
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int hc_count) {
    if (hidden_size != kHidden || hc_count != kStreams) __trap();
    __shared__ __align__(16) __nv_bfloat16 sA[2][kBM][kLds];
    __shared__ __align__(16) __nv_bfloat16 sB[2][kBN][kLds];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp >> 2;
    const int wh = warp & 3;
    const int h0 = blockIdx.x * kBH;
    const int t0 = blockIdx.y * kBM;

    // qwen4_hc_mix's injection-scale transform, once per token (h tile 0).
    if (blockIdx.x == 0) {
        for (int i = tid; i < kBM * kStreams; i += kThreads) {
            const unsigned int token = t0 + i / kStreams;
            if (token < num_tokens) {
                const unsigned int j = token * hc_count + (i % kStreams);
                const float x = __bfloat162float(inject[j]) / hc_count;
                inject[j] = __float2bfloat16(2.0f / (1.0f + expf(-x)));
            }
        }
    }
    float acc[2][kStreams][4];
    upmix_mainloop(act, up, num_tokens, t0, h0, sA, sB, acc);

    const int hl = wh * 8 + 2 * (lane & 3);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const unsigned int token = t0 + wm * 32 + i * 16 + (lane >> 2) + half * 8;
            if (token >= num_tokens) continue;
            const size_t row = (size_t)token * (kHidden * kStreams);
            float mix[2] = {0.0f, 0.0f};
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                const __nv_bfloat162 n2 = *reinterpret_cast<const __nv_bfloat162*>(
                    normed + row + (size_t)s * kHidden + h0 + hl);
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float logit = __bfloat162float(__float2bfloat16(alpha * acc[i][s][half * 2 + e]));
                    const float g = 1.0f / (1.0f + expf(-logit));
                    mix[e] += g * __bfloat162float(e == 0 ? n2.x : n2.y);
                }
            }
            __nv_bfloat162 out;
            out.x = __float2bfloat16(mix[0] / hc_count);
            out.y = __float2bfloat16(mix[1] / hc_count);
            *reinterpret_cast<__nv_bfloat162*>(mixed + (size_t)token * kHidden + h0 + hl) = out;
        }
    }
}

// HC_FUSED's up+mix (qwen4_hc_up_mix_mma in qwen4_hc_fused.cu) on this
// kernel's tiling: same logits (fp32 acc * alpha, no bf16 rounding), the
// normalized rows recomputed as bf16(hyper * rstd * (1 + w)), streams summed
// in pairs ((s0 + s1) + (s2 + s3)) and scaled by 0.25f, so the output is
// byte-identical to that kernel. Grid (H/32, ceil(T/64)), Block 256.
//
// kAmax: also reduce |mixed| into *amax_bits exactly as qwen4_fp8_act_amax
// (qwen4_fp8_proj.cu, same target and flags) does over the same T x 2560
// values: fmaxf of fabsf, then atomicMax on the float bits. The max is
// order-independent, so the FP8 tensor-scale projection that reads `mixed`
// can skip its own absmax pass (ATLAS_QWEN4_PREFILL_FP8_ACT_AMAX_FUSED).
template <bool kAmax>
__device__ __forceinline__ void hc_upmix_hcfused_body(
    const __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ up,
    float alpha,
    const __nv_bfloat16* __restrict__ hyper,
    const float* __restrict__ rstd,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int num_tokens,
    unsigned int* __restrict__ amax_bits) {
    __shared__ __align__(16) __nv_bfloat16 sA[2][kBM][kLds];
    __shared__ __align__(16) __nv_bfloat16 sB[2][kBN][kLds];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp >> 2;
    const int wh = warp & 3;
    const int h0 = blockIdx.x * kBH;
    // Row tiles last-first: the preceding HC pass wrote the last rows last, so
    // they are the likeliest to still be in L2 (micro4: 1.88 -> 1.65 ms @8192).
    const int t0 = (gridDim.y - 1 - blockIdx.y) * kBM;
    float acc[2][kStreams][4];
    upmix_mainloop(act, up, num_tokens, t0, h0, sA, sB, acc);

    const int hl = wh * 8 + 2 * (lane & 3);
    float amax = 0.0f;
    float w1[kStreams][2];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        const __nv_bfloat162 w2 =
            *reinterpret_cast<const __nv_bfloat162*>(norm_w + (size_t)s * kHidden + h0 + hl);
        w1[s][0] = 1.0f + __bfloat162float(w2.x);
        w1[s][1] = 1.0f + __bfloat162float(w2.y);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const unsigned int token = t0 + wm * 32 + i * 16 + (lane >> 2) + half * 8;
            if (token >= num_tokens) continue;
            const size_t row = (size_t)token * (kHidden * kStreams);
            float pair[2][2];  // [stream pair][e]
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                pair[p][0] = 0.0f;
                pair[p][1] = 0.0f;
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const int s = 2 * p + q;
                    const float r = rstd[(size_t)token * kStreams + s];
                    const __nv_bfloat162 h2 = *reinterpret_cast<const __nv_bfloat162*>(
                        hyper + row + (size_t)s * kHidden + h0 + hl);
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const float logit = acc[i][s][half * 2 + e] * alpha;
                        const float gate = 1.0f / (1.0f + __expf(-logit));
                        const float x = __bfloat162float(e == 0 ? h2.x : h2.y) * r * w1[s][e];
                        pair[p][e] += gate * __bfloat162float(__float2bfloat16(x));
                    }
                }
            }
            __nv_bfloat162 out;
            out.x = __float2bfloat16((pair[0][0] + pair[1][0]) * 0.25f);
            out.y = __float2bfloat16((pair[0][1] + pair[1][1]) * 0.25f);
            *reinterpret_cast<__nv_bfloat162*>(mixed + (size_t)token * kHidden + h0 + hl) = out;
            if (kAmax) {
                const float2 f = __bfloat1622float2(out);
                amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
    }
    if (kAmax) {
        __shared__ float warp_max[kThreads / 32];
        for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, d));
        if (lane == 0) warp_max[warp] = amax;
        __syncthreads();
        if (tid == 0) {
            float block = warp_max[0];
            for (int w = 1; w < kThreads / 32; ++w) block = fmaxf(block, warp_max[w]);
            atomicMax(amax_bits, __float_as_uint(block));
        }
    }
}

extern "C" __global__ void __launch_bounds__(kThreads)
qwen4_hc_upmix_hcfused(
    const __nv_bfloat16* __restrict__ act,     // [T, 320] silu'd down projection
    const __nv_bfloat16* __restrict__ up,      // [4*2560, 320] dequantized
    float alpha,
    const __nv_bfloat16* __restrict__ hyper,   // [T, 4*2560]
    const float* __restrict__ rstd,            // [T, 4]
    const __nv_bfloat16* __restrict__ norm_w,  // [4*2560]
    __nv_bfloat16* __restrict__ mixed,         // [T, 2560]
    unsigned int num_tokens) {
    hc_upmix_hcfused_body<false>(act, up, alpha, hyper, rstd, norm_w, mixed, num_tokens, nullptr);
}

// qwen4_hc_upmix_hcfused plus the absmax of `mixed` into *amax_bits (zeroed
// by the caller); `mixed` is byte-identical to the plain kernel.
extern "C" __global__ void __launch_bounds__(kThreads)
qwen4_hc_upmix_hcfused_amax(
    const __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ up,
    float alpha,
    const __nv_bfloat16* __restrict__ hyper,
    const float* __restrict__ rstd,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int num_tokens,
    unsigned int* __restrict__ amax_bits) {
    hc_upmix_hcfused_body<true>(act, up, alpha, hyper, rstd, norm_w, mixed, num_tokens, amax_bits);
}
