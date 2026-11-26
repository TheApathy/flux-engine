// SPDX-License-Identifier: AGPL-3.0-only
//
// Fused Flash-Next hyperconnection (mHC) preparation for wide prefill
// (`ATLAS_QWEN4_PREFILL_HC_FUSED=1|2`). The fast path materializes the
// group-normalized four-stream row (20 KB/token) and the 10240-wide mix
// logits (20 KB/token) and reads each of them back. Here:
//
//   rstd[t,s]   = rsqrt(mean_h(hyper[t,s,h]^2) + eps)            (qwen4_hc_rstd)
//   P_s[t,n]    = sum_h hyper[t,s,h] * W[n, s*H+h] * (1 + w[s,h])  (cuBLASLt, 4 GEMMs
//                 on the raw hyper rows, folded weights from qwen4_hc_fold)
//   down[t,r]   = silu(scale2_d * sum_s rstd[t,s] P_s[t,r] / hc)    (qwen4_hc_combine)
//   inject[t,j] = 2 sigmoid(scale2_i * sum_s rstd[t,s] P_s[t,R+j] / hc)
//   mixed[t,h]  = (1/hc) sum_s sigmoid(alpha_u * (down[t] . U[s*H+h]))
//                 * bf16(hyper[t,s,h] * rstd[t,s] * (1 + w[s,h]))   (qwen4_hc_up_mix_mma,
//                 or cuBLASLt up GEMM + qwen4_hc_mix_rstd at level 1)
//
// Same math as qwen4_hc_group_norm / silu_div / mix; the normalized row is
// recomputed instead of stored and the norm weight is folded into the
// down/inject weights, so it is not bit-exact (different rounding points).

#include <cuda_bf16.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// rstd: Grid (T), Block (256). 64 threads per stream (hc == 4), 16-byte loads.
extern "C" __global__ void qwen4_hc_rstd(
    const __nv_bfloat16* __restrict__ hyper,
    float* __restrict__ rstd,
    unsigned int hidden,
    float eps) {
    const unsigned int t = blockIdx.x;
    const unsigned int s = threadIdx.x >> 6;
    const unsigned int l = threadIdx.x & 63;
    const uint4* row = reinterpret_cast<const uint4*>(
        hyper + ((unsigned long long)t * 4 + s) * hidden);
    float acc = 0.0f;
    for (unsigned int i = l; i < hidden / 8; i += 64) {
        const uint4 v = row[i];
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(p[j]);
            acc += f.x * f.x + f.y * f.y;
        }
    }
    for (int d = 16; d; d >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, d);
    __shared__ float part[8];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 4) {
        const float sum = part[threadIdx.x * 2] + part[threadIdx.x * 2 + 1];
        rstd[t * 4 + threadIdx.x] = rsqrtf(sum / (float)hidden + eps);
    }
}

// ---------------------------------------------------------------------------
// fold: out[s][n][h] = bf16(W[n][s*H+h] * (1 + w[s*H+h])) with W rows
// [0, rank) from wd and [rank, rank+hc) from wi (both [rows][hc*H] bf16).
// Grid: ceil(hc*(rank+hc)*H / 256), Block 256.
extern "C" __global__ void qwen4_hc_fold(
    const __nv_bfloat16* __restrict__ wd,
    const __nv_bfloat16* __restrict__ wi,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ out,
    unsigned int rank,
    unsigned int hidden,
    unsigned int hc) {
    const unsigned int n_total = rank + hc;
    const unsigned long long total = (unsigned long long)hc * n_total * hidden;
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const unsigned int h = (unsigned int)(i % hidden);
    const unsigned long long rest = i / hidden;
    const unsigned int n = (unsigned int)(rest % n_total);
    const unsigned int s = (unsigned int)(rest / n_total);
    const unsigned long long col = (unsigned long long)s * hidden + h;
    const unsigned long long row_len = (unsigned long long)hc * hidden;
    const float w = n < rank ? __bfloat162float(wd[n * row_len + col])
                             : __bfloat162float(wi[(n - rank) * row_len + col]);
    out[i] = __float2bfloat16(w * (1.0f + __bfloat162float(norm_w[col])));
}

// ---------------------------------------------------------------------------
// combine: Grid (T), Block (128).
// P: [hc][T][rank+hc] f32 (per-stream partial projections on the raw hyper).
extern "C" __global__ void qwen4_hc_combine(
    const float* __restrict__ P,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ down,
    __nv_bfloat16* __restrict__ residual,
    unsigned int num_tokens,
    unsigned int rank,
    unsigned int hidden,
    unsigned int hc,
    float scale2_down,
    float scale2_inject) {
    if (hc == 0u || hc > 4u) __trap();
    const unsigned int t = blockIdx.x;
    const unsigned int n_total = rank + hc;
    float r[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) r[s] = s < (int)hc ? rstd[t * hc + s] : 0.0f;
    const unsigned long long stream_stride = (unsigned long long)num_tokens * n_total;
    const float* row = P + (unsigned long long)t * n_total;
    for (unsigned int n = threadIdx.x; n < n_total; n += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int s = 0; s < hc; ++s) acc += r[s] * row[s * stream_stride + n];
        if (n < rank) {
            const float x = scale2_down * acc / (float)hc;
            down[(unsigned long long)t * rank + n] = __float2bfloat16(x / (1.0f + __expf(-x)));
        } else {
            const float x = scale2_inject * acc / (float)hc;
            const unsigned long long rw = (unsigned long long)hidden * hc;
            residual[(unsigned long long)t * rw + (rw - hc) + (n - rank)] =
                __float2bfloat16(2.0f / (1.0f + __expf(-x)));
        }
    }
}

// ---------------------------------------------------------------------------
// mix (level 1): qwen4_hc_mix with the normalized row recomputed from hyper.
// Grid (T, ceil(H/256)), Block 256.
extern "C" __global__ void qwen4_hc_mix_rstd(
    const __nv_bfloat16* __restrict__ hyper,
    const float* __restrict__ rstd,
    const __nv_bfloat16* __restrict__ norm_w,
    const __nv_bfloat16* __restrict__ mix_logits,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int hidden,
    unsigned int hc) {
    const unsigned int t = blockIdx.x;
    const unsigned int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const unsigned long long r = (unsigned long long)hidden * hc;
    float acc = 0.0f;
    for (unsigned int s = 0; s < hc; ++s) {
        const unsigned long long i = t * r + (unsigned long long)s * hidden + h;
        const float g = 1.0f / (1.0f + __expf(-__bfloat162float(mix_logits[i])));
        const float x = __bfloat162float(hyper[i]) * rstd[t * hc + s] *
                        (1.0f + __bfloat162float(norm_w[s * hidden + h]));
        acc += g * __bfloat162float(__float2bfloat16(x));
    }
    mixed[(unsigned long long)t * hidden + h] = __float2bfloat16(acc / (float)hc);
}

// ---------------------------------------------------------------------------
// up+mix fused (level 2): one tensor-core GEMM tile of the 10240-wide up
// projection whose epilogue applies the mix, so the logits never leave
// registers. CTA tile: 64 tokens x 32 h positions = 128 logit columns, ordered
// c = hl*4 + s so each thread's two accumulator columns are streams (s, s+1)
// of one (token, h) and a lane-pair shuffle completes the 4-stream sum.
// Grid (H/32, ceil(T/64)), Block 256 (8 warps: 2 along M x 4 along N, each
// warp 32x32). K (rank) = 320, pipelined 32 at a time through two stages.
namespace hcmix {
constexpr int BM = 64;
constexpr int BH = 32;
constexpr int BN = 128;  // BH * 4 streams
constexpr int BK = 32;
constexpr int LDS = BK + 8;  // 80-byte rows: conflict-free ldmatrix
constexpr int RANK = 320;
constexpr int HID = 2560;
}  // namespace hcmix

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    const unsigned int s = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
}

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

extern "C" __global__ void __launch_bounds__(256, 2) qwen4_hc_up_mix_mma(
    const __nv_bfloat16* __restrict__ down,    // [T, 320] bf16 (silu'd)
    const __nv_bfloat16* __restrict__ up,      // [4*2560, 320] bf16, row n = s*2560 + h
    float alpha_up,
    const __nv_bfloat16* __restrict__ hyper,   // [T, 4*2560]
    const float* __restrict__ rstd,            // [T, 4]
    const __nv_bfloat16* __restrict__ norm_w,  // [4*2560]
    __nv_bfloat16* __restrict__ mixed,         // [T, 2560]
    unsigned int num_tokens) {
    using namespace hcmix;
    __shared__ __align__(16) __nv_bfloat16 sA[2][BM][LDS];
    __shared__ __align__(16) __nv_bfloat16 sB[2][BN][LDS];
    __shared__ __align__(16) __nv_bfloat16 sH[BM][4][BH];
    __shared__ float sR[BM][4];
    __shared__ float sW[4][BH];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp >> 2;  // 0..1 -> rows wm*32
    const int wn = warp & 3;   // 0..3 -> cols wn*32
    const int h0 = blockIdx.x * BH;
    const int t0 = blockIdx.y * BM;

    // Stage the epilogue operands (hyper tile, rstd, norm weight) up front.
    // sH: 64 tokens x 4 streams x 32 h = 256 rows of 64 B = 1024 16-byte chunks.
    for (int c = tid; c < BM * 4 * (BH / 8); c += 256) {
        const int row = c >> 2;       // (token, stream)
        const int part = c & 3;       // 8 h per chunk
        const int tl = row >> 2;
        const int s = row & 3;
        const int t = t0 + tl;
        const bool ok = t < (int)num_tokens;
        const __nv_bfloat16* g =
            hyper + ((unsigned long long)(ok ? t : 0) * 4 + s) * HID + h0 + part * 8;
        cp_async16(&sH[tl][s][part * 8], g, ok);
    }
    for (int i = tid; i < BM * 4; i += 256) {
        const int t = t0 + (i >> 2);
        sR[i >> 2][i & 3] = t < (int)num_tokens ? rstd[(unsigned long long)t * 4 + (i & 3)] : 0.0f;
    }
    if (tid < 4 * BH) sW[tid / BH][tid % BH] = 1.0f + __bfloat162float(norm_w[(tid / BH) * HID + h0 + tid % BH]);

    auto load_stage = [&](int buf, int k0) {
        // A: 64 rows x 32 k = 256 chunks (one per thread).
        {
            const int r = tid >> 2, part = tid & 3;
            const int t = t0 + r;
            const bool ok = t < (int)num_tokens;
            cp_async16(&sA[buf][r][part * 8],
                       down + (unsigned long long)(ok ? t : 0) * RANK + k0 + part * 8, ok);
        }
        // B: 128 rows (c = hl*4 + s) x 32 k = 512 chunks (two per thread).
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int idx = tid + j * 256;
            const int c = idx >> 2, part = idx & 3;
            const int hl = c >> 2, s = c & 3;
            cp_async16(&sB[buf][c][part * 8],
                       up + ((unsigned long long)s * HID + h0 + hl) * RANK + k0 + part * 8, true);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    constexpr int KT = RANK / BK;  // 10
    load_stage(0, 0);
    for (int kt = 0; kt < KT; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < KT) {
            load_stage(buf ^ 1, (kt + 1) * BK);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            unsigned int a[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                // x4: rows wm*32 + i*16 + (lane & 15), k = kk + (lane >> 4) * 8
                const int r = wm * 32 + i * 16 + (lane & 15);
                ldmatrix_x4(a[i], &sA[buf][r][kk + (lane >> 4) * 8]);
            }
            unsigned int b[4][2];
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                // x4 covers two n8 blocks: matrices (n0..7,k0..7),(n0..7,k8..15),
                // (n8..15,k0..7),(n8..15,k8..15).
                const int n = wn * 32 + jj * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int k = kk + ((lane >> 3) & 1) * 8;
                unsigned int r4[4];
                ldmatrix_x4(r4, &sB[buf][n][k]);
                b[jj * 2][0] = r4[0];
                b[jj * 2][1] = r4[1];
                b[jj * 2 + 1][0] = r4[2];
                b[jj * 2 + 1][1] = r4[3];
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_bf16(acc[i][j], a[i], b[j][0], b[j][1]);
        }
        __syncthreads();
    }

    // Epilogue. Thread fragment (m16n8): rows g and g+8 (g = lane>>2), cols
    // 2*tig and 2*tig+1 (tig = lane&3) of each n8 block. Column c = hl*4 + s,
    // so a thread holds (hl, s) and (hl, s+1) with s = (2*tig) & 3.
    const int g = lane >> 2;
    const int tig = lane & 3;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c0 = wn * 32 + j * 8 + 2 * tig;
            const int hl = c0 >> 2;
            const int s = c0 & 3;  // 0 or 2
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int tl = wm * 32 + i * 16 + g + half * 8;
                float part = 0.0f;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float logit = acc[i][j][half * 2 + e] * alpha_up;
                    const float gate = 1.0f / (1.0f + __expf(-logit));
                    const float x = __bfloat162float(sH[tl][s + e][hl]) * sR[tl][s + e] * sW[s + e][hl];
                    part += gate * __bfloat162float(__float2bfloat16(x));
                }
                part += __shfl_xor_sync(0xffffffff, part, 1);
                const int t = t0 + tl;
                if ((tig & 1) == 0 && t < (int)num_tokens) {
                    mixed[(unsigned long long)t * HID + h0 + hl] = __float2bfloat16(part * 0.25f);
                }
            }
        }
    }
}

#include <cuda_fp8.h>

__device__ __constant__ float HC_DQ_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Fused mHC static-weight preparation (`ATLAS_QWEN4_PREFILL_HC_FUSED_PREP=1`):
// replaces the 4 per-boundary weight kernels (dequant down -> s.weights, dequant inject ->
// s.weights, qwen4_hc_fold -> s.folded, dequant up -> s.up) and the 13.27 MB intermediate
// `s.weights` round-trip with a single 16-element-vectorized kernel launch that produces
// `folded_out` (`[hc][rank+hc][hidden]` bf16) and optionally `up_out` (`[hc*hidden][rank]` bf16)
// with 100% bit-for-bit identical values.
// Launch grid contract (both canonical fast path and generic fallback):
//   `fold_threads = (hc * (rank + hc) * (hidden / 16)) / 4` (4 fold groups/thread;
//   requires `hidden % 16 == 0`, `rank % 16 == 0`, `(rank + hc) % 4 == 0`), plus
//   `up_groups = hc * hidden * (rank / 16)` when `up_out != nullptr`.
extern "C" __global__ void __launch_bounds__(256) qwen4_hc_prep_weights(
    const unsigned char* __restrict__ down_packed,
    const unsigned char* __restrict__ down_scales,
    const unsigned char* __restrict__ inject_packed,
    const unsigned char* __restrict__ inject_scales,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ folded_out,
    const unsigned char* __restrict__ up_packed,
    const unsigned char* __restrict__ up_scales,
    __nv_bfloat16* __restrict__ up_out,
    unsigned int rank,
    unsigned int hidden,
    unsigned int hc
) {
    if (rank == 320u && hidden == 2560u && hc == 4u) {
        const float lut_lane = HC_DQ_E2M1_LUT[threadIdx.x & 15u];
        const unsigned int g32 = blockIdx.x * blockDim.x + threadIdx.x;
        if (g32 < 51840u) {
            const unsigned int n4 = g32 / 640u;
            const unsigned int gi = g32 - n4 * 640u;
            const unsigned int s = gi / 160u;
            const unsigned int hg = gi - s * 160u;
            const unsigned int n_base = n4 * 4u;
            const unsigned int r_base = (n4 < 80u) ? n_base : 0u;
            const unsigned char* pk = (n4 < 80u) ? down_packed : inject_packed;
            const unsigned char* sc = (n4 < 80u) ? down_scales : inject_scales;

            const uint4* nw_vec = reinterpret_cast<const uint4*>(norm_w + (unsigned long long)gi * 16ull);
            const uint4 nw0 = nw_vec[0];
            const uint4 nw1 = nw_vec[1];
            const unsigned int nw_words[8] = {nw0.x, nw0.y, nw0.z, nw0.w, nw1.x, nw1.y, nw1.z, nw1.w};
            float2 nw_scale[8];
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const float2 nwf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&nw_words[b]));
                nw_scale[b] = make_float2(1.0f + nwf.x, 1.0f + nwf.y);
            }

            uint2 p8_k[4];
            unsigned char sc_k[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const unsigned long long grp_idx = (unsigned long long)(r_base + (unsigned int)k) * 640ull + gi;
                p8_k[k] = *reinterpret_cast<const uint2*>(pk + grp_idx * 8ull);
                sc_k[k] = sc[grp_idx];
            }

            uint64_t pol;
            asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                __nv_fp8_e4m3 fp8;
                *reinterpret_cast<unsigned char*>(&fp8) = sc_k[k];
                const float s_fp8 = (float)fp8;
                const uint2 p8 = p8_k[k];
                unsigned int out_words[8];
#pragma unroll
                for (int b = 0; b < 8; ++b) {
                    const unsigned int word = (b < 4) ? p8.x : p8.y;
                    const unsigned int byte_val = (word >> ((b & 3) * 8)) & 0xFFu;
                    const float lut0 = __shfl_sync(0xffffffffu, lut_lane, byte_val & 0xFu);
                    const float lut1 = __shfl_sync(0xffffffffu, lut_lane, byte_val >> 4);
                    const float2 w01 = __bfloat1622float2(__floats2bfloat162_rn(lut0 * s_fp8, lut1 * s_fp8));
                    const __nv_bfloat162 f01 = __floats2bfloat162_rn(w01.x * nw_scale[b].x, w01.y * nw_scale[b].y);
                    out_words[b] = *reinterpret_cast<const unsigned int*>(&f01);
                }
                uint4* dst = reinterpret_cast<uint4*>(
                    folded_out + ((unsigned long long)s * 324ull + n_base + (unsigned int)k) * 2560ull
                    + (unsigned long long)hg * 16ull);
                asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n"
                             :: "l"(dst + 0), "r"(out_words[0]), "r"(out_words[1]), "r"(out_words[2]), "r"(out_words[3]), "l"(pol) : "memory");
                asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n"
                             :: "l"(dst + 1), "r"(out_words[4]), "r"(out_words[5]), "r"(out_words[6]), "r"(out_words[7]), "l"(pol) : "memory");
            }
            return;
        }
        if (up_out != nullptr && g32 < 51840u + 204800u) {
            const unsigned int ug = g32 - 51840u;
            const uint2 p8 = *reinterpret_cast<const uint2*>(up_packed + (unsigned long long)ug * 8ull);
            __nv_fp8_e4m3 fp8;
            *reinterpret_cast<unsigned char*>(&fp8) = up_scales[ug];
            const float s_fp8 = (float)fp8;

            unsigned int out_words[8];
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const unsigned int word = (b < 4) ? p8.x : p8.y;
                const unsigned int byte_val = (word >> ((b & 3) * 8)) & 0xFFu;
                const float lut0 = __shfl_sync(0xffffffffu, lut_lane, byte_val & 0xFu);
                const float lut1 = __shfl_sync(0xffffffffu, lut_lane, byte_val >> 4);
                const __nv_bfloat16 lo = __float2bfloat16(lut0 * s_fp8);
                const __nv_bfloat16 hi = __float2bfloat16(lut1 * s_fp8);
                const unsigned short u0 = *reinterpret_cast<const unsigned short*>(&lo);
                const unsigned short u1 = *reinterpret_cast<const unsigned short*>(&hi);
                out_words[b] = (unsigned int)u0 | ((unsigned int)u1 << 16);
            }
            uint4* dst = reinterpret_cast<uint4*>(up_out + (unsigned long long)ug * 16ull);
            uint64_t pol;
            asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
            asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n"
                         :: "l"(dst + 0), "r"(out_words[0]), "r"(out_words[1]), "r"(out_words[2]), "r"(out_words[3]), "l"(pol) : "memory");
            asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n"
                         :: "l"(dst + 1), "r"(out_words[4]), "r"(out_words[5]), "r"(out_words[6]), "r"(out_words[7]), "l"(pol) : "memory");
            return;
        }
        return;
    }

    if ((hidden & 15u) != 0u || (rank & 15u) != 0u || ((rank + hc) & 3u) != 0u) __trap();
    const unsigned int n_total = rank + hc;
    const unsigned int n4_count = n_total / 4u;
    const unsigned int groups_per_h = hidden / 16u;
    const unsigned int groups_per_row = hc * groups_per_h;
    const unsigned long long fold_threads = (unsigned long long)n4_count * groups_per_row;
    const unsigned int up_groups_per_row = rank / 16u;
    const unsigned long long up_groups = (unsigned long long)(hc * hidden) * up_groups_per_row;
    const unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    if (g < fold_threads) {
        const unsigned int n4 = (unsigned int)(g / groups_per_row);
        const unsigned int gi = (unsigned int)(g - (unsigned long long)n4 * groups_per_row);
        const unsigned int s = gi / groups_per_h;
        const unsigned int hg = gi - s * groups_per_h;
        const unsigned int n_base = n4 * 4u;
        const __nv_bfloat16* nw = norm_w + (unsigned long long)gi * 16ull;
        uint64_t pol;
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));

#pragma unroll
        for (unsigned int k = 0; k < 4u; ++k) {
            const unsigned int n = n_base + k;
            const unsigned long long r_idx = (n < rank) ? n : (n - rank);
            const unsigned char* pk = (n < rank) ? down_packed : inject_packed;
            const unsigned char* sc = (n < rank) ? down_scales : inject_scales;

            const unsigned long long p8 =
                *reinterpret_cast<const unsigned long long*>(pk + r_idx * (groups_per_row * 8ull) + gi * 8ull);
            __nv_fp8_e4m3 fp8;
            *reinterpret_cast<unsigned char*>(&fp8) = sc[r_idx * groups_per_row + gi];
            const float s_fp8 = (float)fp8;

            unsigned short vals[16];
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const unsigned char byte_val = (unsigned char)(p8 >> (b * 8));
                const float w0 = __bfloat162float(__float2bfloat16(HC_DQ_E2M1_LUT[byte_val & 0xF] * s_fp8));
                const float w1 = __bfloat162float(__float2bfloat16(HC_DQ_E2M1_LUT[byte_val >> 4] * s_fp8));
                const __nv_bfloat16 f0 = __float2bfloat16(w0 * (1.0f + __bfloat162float(nw[2 * b])));
                const __nv_bfloat16 f1 = __float2bfloat16(w1 * (1.0f + __bfloat162float(nw[2 * b + 1])));
                vals[2 * b] = *reinterpret_cast<const unsigned short*>(&f0);
                vals[2 * b + 1] = *reinterpret_cast<const unsigned short*>(&f1);
            }
            uint4* dst = reinterpret_cast<uint4*>(
                folded_out + ((unsigned long long)s * n_total + n) * hidden + (unsigned long long)hg * 16ull);
            uint4 v0, v1;
            v0.x = (unsigned)vals[0] | ((unsigned)vals[1] << 16);
            v0.y = (unsigned)vals[2] | ((unsigned)vals[3] << 16);
            v0.z = (unsigned)vals[4] | ((unsigned)vals[5] << 16);
            v0.w = (unsigned)vals[6] | ((unsigned)vals[7] << 16);
            v1.x = (unsigned)vals[8] | ((unsigned)vals[9] << 16);
            v1.y = (unsigned)vals[10] | ((unsigned)vals[11] << 16);
            v1.z = (unsigned)vals[12] | ((unsigned)vals[13] << 16);
            v1.w = (unsigned)vals[14] | ((unsigned)vals[15] << 16);
            asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n"
                         :: "l"(dst + 0), "r"(v0.x), "r"(v0.y), "r"(v0.z), "r"(v0.w), "l"(pol) : "memory");
            asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n"
                         :: "l"(dst + 1), "r"(v1.x), "r"(v1.y), "r"(v1.z), "r"(v1.w), "l"(pol) : "memory");
        }
        return;
    }

    if (up_out != nullptr && g < fold_threads + up_groups) {
        const unsigned long long ug = g - fold_threads;
        const unsigned long long u_row = ug / up_groups_per_row;
        const unsigned long long u_gi = ug - u_row * up_groups_per_row;
        const unsigned long long p8 =
            *reinterpret_cast<const unsigned long long*>(up_packed + u_row * (rank / 2u) + u_gi * 8ull);
        __nv_fp8_e4m3 fp8;
        *reinterpret_cast<unsigned char*>(&fp8) = up_scales[u_row * up_groups_per_row + u_gi];
        const float s_fp8 = (float)fp8;

        unsigned short vals[16];
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const unsigned char byte_val = (unsigned char)(p8 >> (b * 8));
            const __nv_bfloat16 lo = __float2bfloat16(HC_DQ_E2M1_LUT[byte_val & 0xF] * s_fp8);
            const __nv_bfloat16 hi = __float2bfloat16(HC_DQ_E2M1_LUT[byte_val >> 4] * s_fp8);
            vals[2 * b] = *reinterpret_cast<const unsigned short*>(&lo);
            vals[2 * b + 1] = *reinterpret_cast<const unsigned short*>(&hi);
        }
        uint4* dst = reinterpret_cast<uint4*>(up_out + u_row * (unsigned long long)rank + u_gi * 16ull);
        uint4 v0, v1;
        v0.x = (unsigned)vals[0] | ((unsigned)vals[1] << 16);
        v0.y = (unsigned)vals[2] | ((unsigned)vals[3] << 16);
        v0.z = (unsigned)vals[4] | ((unsigned)vals[5] << 16);
        v0.w = (unsigned)vals[6] | ((unsigned)vals[7] << 16);
        v1.x = (unsigned)vals[8] | ((unsigned)vals[9] << 16);
        v1.y = (unsigned)vals[10] | ((unsigned)vals[11] << 16);
        v1.z = (unsigned)vals[12] | ((unsigned)vals[13] << 16);
        v1.w = (unsigned)vals[14] | ((unsigned)vals[15] << 16);
        uint64_t pol;
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
        asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n"
                     :: "l"(dst + 0), "r"(v0.x), "r"(v0.y), "r"(v0.z), "r"(v0.w), "l"(pol) : "memory");
        asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n"
                     :: "l"(dst + 1), "r"(v1.x), "r"(v1.y), "r"(v1.z), "r"(v1.w), "l"(pol) : "memory");
    }
}

extern "C" __global__ void __launch_bounds__(256) qwen4_hc_prep_up(
    const unsigned char* __restrict__ up_packed,
    const unsigned char* __restrict__ up_scales,
    __nv_bfloat16* __restrict__ up_out,
    unsigned int rank,
    unsigned int hidden,
    unsigned int hc,
    unsigned int* __restrict__ amax_zero
) {
    if (amax_zero != nullptr && blockIdx.x == 0u && threadIdx.x == 0u) {
        *amax_zero = 0u;
    }
    if (rank == 320u && hidden == 2560u && hc == 4u) {
        const float lut_lane = HC_DQ_E2M1_LUT[threadIdx.x & 15u];
        const unsigned int ug = blockIdx.x * blockDim.x + threadIdx.x;
        if (ug >= 204800u) return;
        const uint2 p8 = *reinterpret_cast<const uint2*>(up_packed + (unsigned long long)ug * 8ull);
        __nv_fp8_e4m3 fp8;
        *reinterpret_cast<unsigned char*>(&fp8) = up_scales[ug];
        const float s_fp8 = (float)fp8;

        unsigned int out_words[8];
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const unsigned int word = (b < 4) ? p8.x : p8.y;
            const unsigned int byte_val = (word >> ((b & 3) * 8)) & 0xFFu;
            const float lut0 = __shfl_sync(0xffffffffu, lut_lane, byte_val & 0xFu);
            const float lut1 = __shfl_sync(0xffffffffu, lut_lane, byte_val >> 4);
            const __nv_bfloat16 lo = __float2bfloat16(lut0 * s_fp8);
            const __nv_bfloat16 hi = __float2bfloat16(lut1 * s_fp8);
            const unsigned short u0 = *reinterpret_cast<const unsigned short*>(&lo);
            const unsigned short u1 = *reinterpret_cast<const unsigned short*>(&hi);
            out_words[b] = (unsigned int)u0 | ((unsigned int)u1 << 16);
        }
        uint4* dst = reinterpret_cast<uint4*>(up_out + (unsigned long long)ug * 16ull);
        uint64_t pol;
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
        asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n"
                     :: "l"(dst + 0), "r"(out_words[0]), "r"(out_words[1]), "r"(out_words[2]), "r"(out_words[3]), "l"(pol) : "memory");
        asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n"
                     :: "l"(dst + 1), "r"(out_words[4]), "r"(out_words[5]), "r"(out_words[6]), "r"(out_words[7]), "l"(pol) : "memory");
        return;
    }

    const unsigned int up_groups_per_row = rank / 16u;
    const unsigned long long up_groups = (unsigned long long)(hc * hidden) * up_groups_per_row;
    const unsigned long long ug = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (ug >= up_groups) return;
    const unsigned long long u_row = ug / up_groups_per_row;
    const unsigned long long u_gi = ug - u_row * up_groups_per_row;
    const unsigned long long p8 =
        *reinterpret_cast<const unsigned long long*>(up_packed + u_row * (rank / 2u) + u_gi * 8ull);
    __nv_fp8_e4m3 fp8;
    *reinterpret_cast<unsigned char*>(&fp8) = up_scales[u_row * up_groups_per_row + u_gi];
    const float s_fp8 = (float)fp8;

    unsigned short vals[16];
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const unsigned char byte_val = (unsigned char)(p8 >> (b * 8));
        const __nv_bfloat16 lo = __float2bfloat16(HC_DQ_E2M1_LUT[byte_val & 0xF] * s_fp8);
        const __nv_bfloat16 hi = __float2bfloat16(HC_DQ_E2M1_LUT[byte_val >> 4] * s_fp8);
        vals[2 * b] = *reinterpret_cast<const unsigned short*>(&lo);
        vals[2 * b + 1] = *reinterpret_cast<const unsigned short*>(&hi);
    }
    uint4* dst = reinterpret_cast<uint4*>(up_out + u_row * (unsigned long long)rank + u_gi * 16ull);
    uint4 v0, v1;
    v0.x = (unsigned)vals[0] | ((unsigned)vals[1] << 16);
    v0.y = (unsigned)vals[2] | ((unsigned)vals[3] << 16);
    v0.z = (unsigned)vals[4] | ((unsigned)vals[5] << 16);
    v0.w = (unsigned)vals[6] | ((unsigned)vals[7] << 16);
    v1.x = (unsigned)vals[8] | ((unsigned)vals[9] << 16);
    v1.y = (unsigned)vals[10] | ((unsigned)vals[11] << 16);
    v1.z = (unsigned)vals[12] | ((unsigned)vals[13] << 16);
    v1.w = (unsigned)vals[14] | ((unsigned)vals[15] << 16);
    uint64_t pol;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n"
                 :: "l"(dst + 0), "r"(v0.x), "r"(v0.y), "r"(v0.z), "r"(v0.w), "l"(pol) : "memory");
    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n"
                 :: "l"(dst + 1), "r"(v1.x), "r"(v1.y), "r"(v1.z), "r"(v1.w), "l"(pol) : "memory");
}
