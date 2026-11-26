// SPDX-License-Identifier: AGPL-3.0-only
//
// E4M3 variant of the QSA-sparse prefill attention core
// (qwen4_qsa_prefill_attn.cu, T=10 with per-warp tile skipping), selected by
// ATLAS_QWEN4_PREFILL_QSA_FP8. Same CTA shape, tile plan, masking and base-2
// online softmax as qwen4_qsa_prefill_attn_t10_skip.
//
// Q, K (and with PV8 also V) arrive already in E4M3, written once by their
// producers with static power-of-two prescales s_q, s_k, s_v: the kernel does
// no scaling or packing of Q/K/V. Per KV head h (= blockIdx.y), scale_log2_h carries
// log2(e)*softmax_scale/(s_q*s_k) and out_scale_h carries 1/s_v (1 without PV8).
//
// QK^T: m16n8k32 E4M3 MMA, fp32 accumulate. A 256-byte E4M3 row has the same
// ldmatrix.b16 fragment layout as a 16-column BF16 k-step, so Q and K use the
// BF16 kernel's ldmatrix addressing with half the k-steps.
// PV8: P = 256 * exp2(s - m) <= 256 is packed to E4M3 from the thread's own
// accumulators with the k axis permuted within each 32-token tile:
//   k = 4t+i <-> token (i>>1)*8 + 2t + (i&1),  k = 16+4t+i <-> 16 + the same,
// and V fragments for that permutation come from ldmatrix.trans.b16 over
// [token][dim pair] plus two byte permutes: one ldmatrix gives an even-dim
// and an odd-dim n-tile, so a thread owns dims 16c+4t .. 16c+4t+3.
// ACC (PV8 accumulation):
//   0  P*V accumulates in the MMA (C = O, rescaled when a row max moves); l is the
//      fp32 sum of P before rounding (the 22:48Z kernel).
//   1  two-level: each tile's P*V starts from C = 0 and one FFMA folds it into the
//      running O with the online-softmax correction, so O stays fp32 across tiles.
//   2  as 0 for O.
//   With 1 and 2 the row sum l is an MMA of the same E4M3 P against ones, i.e.
//   exactly the P mass the numerator used (an underflowed P drops out of both). With
// qwen4_qsa_prefill_plan_sort_desc (need groups ascending, blocks descending)
// the newest blocks come first and the sink block last, so the running max
// that scales P is the non-sink one for most tiles.
// Without PV8, P*V is the BF16 path of the original kernel on the BF16 V cache.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace {

constexpr int kHd = 256;
constexpr int kGroup = 12;
constexpr int kBc = 32;
constexpr int kTrows = 10;
constexpr int kM = kGroup * kTrows;
constexpr int kWarps = (kM + 15) / 16;
constexpr int kThreads = kWarps * 32;
constexpr int kCap = (kTrows * 513 + 3) / 4 * 4;
constexpr unsigned int kRatio = 4;
constexpr unsigned int kBlockBits = 22;
constexpr unsigned int kBlockMask = (1u << kBlockBits) - 1u;
constexpr int kTile8 = kBc * kHd;           // bytes of one E4M3 32-token tile
constexpr int kTile16 = kBc * kHd * 2;      // bytes of one BF16 32-token tile
constexpr int kQ8Bytes = kWarps * 16 * kHd; // E4M3 Q staging (128 rows)

template <bool PV8>
struct Fp8Cfg {
    static constexpr int kStageBytes = kTile8 + (PV8 ? kTile8 : kTile16);
};

// E4M3 rows: 16-byte chunk XOR-swizzled by row & 7 (256-byte rows).
__device__ __forceinline__ unsigned int swz8(unsigned int row, unsigned int chunk) {
    return row * kHd + ((chunk ^ (row & 7u)) << 4);
}
// BF16 rows (elements), as in qwen4_qsa_prefill_attn.cu.
__device__ __forceinline__ unsigned int swz16(unsigned int row, unsigned int col) {
    return row * kHd + ((((col >> 3) ^ (row & 7u)) << 3) | (col & 7u));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned int sa = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(gmem));
}

__device__ __forceinline__ void ldsm_x4(unsigned int (&r)[4], const void* smem) {
    const unsigned int sa = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sa));
}

__device__ __forceinline__ void ldsm_x4_t(unsigned int (&r)[4], const void* smem) {
    const unsigned int sa = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sa));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const unsigned int (&a)[4],
                                         unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Issued as block-scaled kind::mxf8f6f4 with unit ue8m0 scale 0x7f (1.0): runs at
// 2.0x full tensor-core rate on SM120/SM121 with bit-identical FP32 outputs.
__device__ __forceinline__ void mma_e4m3(float (&d)[4], const unsigned int (&a)[4],
                                         unsigned int b0, unsigned int b1) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%10},{0,0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0x7fu));
#else
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// D = A*B with C = 0 (one tile's partial, folded by the caller).
__device__ __forceinline__ void mma_e4m3_zc(float (&d)[4], const unsigned int (&a)[4],
                                            unsigned int b0, unsigned int b1) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%10,%10,%10},{%11},{0,0},{%11},{0,0};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.0f), "r"(0x7fu));
#else
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%10,%10,%10};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.0f));
#endif
}

// Four E4M3 1.0 (0x38): a B operand of ones turns the MMA into row sums of A.
constexpr unsigned int kE4m3OnesX4 = 0x38383838u;

__device__ __forceinline__ unsigned int pack_bf16(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const unsigned int*>(&v);
}

// Four floats -> four E4M3 bytes, a in the lowest byte (round to nearest, satfinite).
__device__ __forceinline__ unsigned int e4m3x4(float a, float b, float c, float d) {
    const unsigned int lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
    const unsigned int hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
    return lo | (hi << 16);
}

__device__ __forceinline__ unsigned int prmt(unsigned int a, unsigned int b, unsigned int sel) {
    unsigned int d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
}

struct RowInfo {
    bool live;
    unsigned int bit;        // member bit (row within the tile)
    unsigned int pos;        // absolute query position
    unsigned int tail_lo;    // first token of the open tail group
};

__device__ __forceinline__ RowInfo row_info(int m, unsigned int r0, unsigned int rows,
                                            unsigned int first_pos) {
    RowInfo info;
    const unsigned int rt = static_cast<unsigned int>(m) / kGroup;
    info.live = m < kM && r0 + rt < rows;
    info.bit = info.live ? rt : 0u;
    info.pos = first_pos + r0 + rt;
    info.tail_lo = ((info.pos + 1) / kRatio) * kRatio;
    return info;
}

__device__ __forceinline__ bool allowed(const RowInfo& row, unsigned int entry, bool slot_live,
                                        unsigned int in_block) {
    const unsigned int tok = (entry & kBlockMask) * kRatio + in_block;
    const bool member = (entry >> (kBlockBits + row.bit)) & 1u;
    const bool tail = tok >= row.tail_lo && tok <= row.pos;
    return slot_live && row.live && (member || tail);
}

__device__ __forceinline__ unsigned int entry_need(unsigned int e, unsigned int row0_pos) {
    const unsigned int blk = e & kBlockMask;
    unsigned int need = (e >> kBlockBits) & ((1u << kTrows) - 1u);
    const int lo = static_cast<int>(blk * kRatio) - static_cast<int>(row0_pos);
    for (int rt = max(lo, 0); rt < min(lo + 3, kTrows); ++rt) need |= 1u << rt;
    return need;
}

// Gather one 32-token tile: E4M3 K (and E4M3 V with PV8) 16-byte chunks, or
// the BF16 V tile without PV8. Slots past `count` are zero filled.
template <bool PV8>
__device__ __forceinline__ void load_tile(unsigned char* st, const uint8_t* __restrict__ kc8,
                                          const uint8_t* __restrict__ vc,
                                          const unsigned int* ent, unsigned int base,
                                          unsigned int count, const int* __restrict__ bt,
                                          unsigned int page_size,
                                          unsigned long long k_page_stride,
                                          unsigned int k_tok_stride,
                                          unsigned long long v_page_stride,
                                          unsigned int v_tok_stride, unsigned int kvh,
                                          unsigned int tid) {
    unsigned char* ks = st;
    unsigned char* vs = st + kTile8;
    for (unsigned int c = tid; c < kBc * 16; c += kThreads) {
        const unsigned int j = c >> 4;
        const unsigned int ch = c & 15;
        const unsigned int slot = base + (j >> 2);
        unsigned char* kd = ks + swz8(j, ch);
        if (slot < count) {
            const unsigned int tok = (ent[slot] & kBlockMask) * kRatio + (j & 3);
            const unsigned int page = static_cast<unsigned int>(__ldg(bt + tok / page_size));
            const unsigned int in_page = tok % page_size;
            cp_async16(kd, kc8 + (unsigned long long)page * k_page_stride +
                               (unsigned long long)in_page * k_tok_stride + kvh * kHd + ch * 16);
            if (PV8) {
                cp_async16(vs + swz8(j, ch), vc + (unsigned long long)page * v_page_stride +
                                                 (unsigned long long)in_page * v_tok_stride +
                                                 kvh * kHd + ch * 16);
            }
        } else {
            *reinterpret_cast<uint4*>(kd) = make_uint4(0, 0, 0, 0);
            if (PV8) *reinterpret_cast<uint4*>(vs + swz8(j, ch)) = make_uint4(0, 0, 0, 0);
        }
    }
    if (!PV8) {
        __nv_bfloat16* vs16 = reinterpret_cast<__nv_bfloat16*>(vs);
        for (unsigned int c = tid; c < kBc * 32; c += kThreads) {
            const unsigned int j = c >> 5;
            const unsigned int ch = c & 31;
            const unsigned int slot = base + (j >> 2);
            __nv_bfloat16* vd = vs16 + swz16(j, ch * 8);
            if (slot < count) {
                const unsigned int tok = (ent[slot] & kBlockMask) * kRatio + (j & 3);
                const unsigned int page = static_cast<unsigned int>(__ldg(bt + tok / page_size));
                cp_async16(vd, vc + (unsigned long long)page * v_page_stride +
                                   (unsigned long long)(tok % page_size) * v_tok_stride +
                                   (kvh * kHd + ch * 8) * 2);
            } else {
                *reinterpret_cast<uint4*>(vd) = make_uint4(0, 0, 0, 0);
            }
        }
    }
}

// |x| of both bf16 halves of a packed word, folded into amax with qwen4_fp8_act_amax's rule
// (fmaxf of fabsf of the exactly widened bf16; NaN ignored; same target flags).
__device__ __forceinline__ float amax_bf16x2(float amax, unsigned int w) {
    return fmaxf(amax, fmaxf(fabsf(__uint_as_float(w << 16)), fabsf(__uint_as_float(w & 0xffff0000u))));
}

// Normalise and store one MMA row (half 0: row g, half 2: row g+8).
// BF16 PV layout: o[i] (i = 0..31) holds dims 8i + 2t + {0,1} (as the BF16 kernel).
// PV8 layout: o[2c] (even dims) and o[2c+1] (odd dims) hold dims 16c + 4t + {0..3}.
// With a gate the result is bf16(bf16(o) * sigmoid(g)), as the BF16 kernel.
// AMAX (ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED) also folds the stored values' magnitudes into `amax`:
//   1  |bf16| of each packed word (qwen4_fp8_act_amax's widening);
//   2  |x| in fp32, rounded to bf16 once at the end: max_i |rn(x_i)| == rn(max_i |x_i|) because RN is monotone
//      and odd (NaN ignored by fmaxf on both sides; FTZ flushes the same subnormal magnitudes on both sides);
//   3  nothing here: the epilogue re-reads the stored words (reread_row), leaving this loop as shipped.
template <bool PV8, int AMAX>
__device__ __forceinline__ void store_row(__nv_bfloat16* __restrict__ out,
                                          const __nv_bfloat16* __restrict__ gate,
                                          unsigned int row, unsigned int head,
                                          unsigned int o_stride, unsigned int gate_stride,
                                          unsigned int t, float inv_l, const float (&o)[32][4],
                                          int half, float& amax) {
    __nv_bfloat16* dst = out + (size_t)row * o_stride + head * kHd;
    const __nv_bfloat16* gsrc = gate ? gate + (size_t)row * gate_stride + head * kHd : nullptr;
    // bf16(o / l), then (with a gate) bf16(that * sigmoid(g)) at the final pack.
    const auto norm = [&](float acc) { return __bfloat162float(__float2bfloat16(acc * inv_l)); };
    const auto gated = [](float x, __nv_bfloat16 gv) {
        return x * (1.0f / (1.0f + expf(-__bfloat162float(gv))));
    };
    if (PV8) {
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            const unsigned int d0 = c * 16 + t * 4;
            float x0 = norm(o[2 * c][half]);
            float x1 = norm(o[2 * c + 1][half]);
            float x2 = norm(o[2 * c][half + 1]);
            float x3 = norm(o[2 * c + 1][half + 1]);
            if (gsrc) {
                const uint2 gw = *reinterpret_cast<const uint2*>(gsrc + d0);
                const __nv_bfloat162 g01 = *reinterpret_cast<const __nv_bfloat162*>(&gw.x);
                const __nv_bfloat162 g23 = *reinterpret_cast<const __nv_bfloat162*>(&gw.y);
                x0 = gated(x0, g01.x);
                x1 = gated(x1, g01.y);
                x2 = gated(x2, g23.x);
                x3 = gated(x3, g23.y);
            }
            const unsigned int w01 = pack_bf16(x0, x1);
            const unsigned int w23 = pack_bf16(x2, x3);
            *reinterpret_cast<uint2*>(dst + d0) = make_uint2(w01, w23);
            if constexpr (AMAX == 1) amax = amax_bf16x2(amax_bf16x2(amax, w01), w23);
            if constexpr (AMAX == 2)
                amax = fmaxf(fmaxf(fmaxf(fmaxf(amax, fabsf(x0)), fabsf(x1)), fabsf(x2)), fabsf(x3));
        }
    } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const unsigned int d0 = i * 8 + t * 2;
            float x0 = norm(o[i][half]);
            float x1 = norm(o[i][half + 1]);
            if (gsrc) {
                const __nv_bfloat162 gv = *reinterpret_cast<const __nv_bfloat162*>(gsrc + d0);
                x0 = gated(x0, gv.x);
                x1 = gated(x1, gv.y);
            }
            const unsigned int w01 = pack_bf16(x0, x1);
            *reinterpret_cast<unsigned int*>(dst + d0) = w01;
            if constexpr (AMAX == 1) amax = amax_bf16x2(amax, w01);
            if constexpr (AMAX == 2) amax = fmaxf(fmaxf(amax, fabsf(x0)), fabsf(x1));
        }
    }
}

// AMAX mode 3: fold |bf16| of the words this lane stored for one row, re-read with ld.global.cg (opaque to
// the compiler, so the value comes from memory, not from the registers that produced it). The "memory" clobber
// makes the compiler complete every earlier store (store_row's) before the load and keep it after them: without
// it the order of the plain stores against the asm load is the compiler's choice, and a load hoisted above its
// store would fold the previous layer's raw_outputs into the o_proj scale (K8 D review). The same thread stored
// these words, so program order is enough at the hardware level (no fence).
template <bool PV8>
__device__ __forceinline__ float reread_row(const __nv_bfloat16* __restrict__ out, unsigned int row,
                                            unsigned int head, unsigned int o_stride, unsigned int t,
                                            float amax) {
    const __nv_bfloat16* src = out + (size_t)row * o_stride + head * kHd;
    if (PV8) {
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            unsigned int w0, w1;
            asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(w0), "=r"(w1) : "l"(src + c * 16 + t * 4) : "memory");
            amax = amax_bf16x2(amax_bf16x2(amax, w0), w1);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            unsigned int w0;
            asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(w0) : "l"(src + i * 8 + t * 2) : "memory");
            amax = amax_bf16x2(amax, w0);
        }
    }
    return amax;
}

template <bool PV8, int ACC, int STAGES, int AMAX = 0>
__device__ __forceinline__ void qsa_prefill_attn_fp8(
    const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,
    const uint8_t* __restrict__ vc, const int* __restrict__ bt,
    const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
    unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
    unsigned int gate_stride, unsigned int page_size, unsigned long long k_page_stride,
    unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
    float scale_log2_h0, float scale_log2_h1, float out_scale_h0, float out_scale_h1,
    unsigned int* __restrict__ amax_bits = nullptr) {
    using Cfg = Fp8Cfg<PV8>;
    constexpr bool kTwoLevel = PV8 && ACC == 1;
    constexpr bool kQuantL = PV8 && ACC != 0;
    static_assert(STAGES * Cfg::kStageBytes >= kQ8Bytes, "Q staging must fit the tile ring");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned char* ring = smem_raw;
    unsigned int* ent = reinterpret_cast<unsigned int*>(smem_raw + STAGES * Cfg::kStageBytes);

    const unsigned int tile = blockIdx.x;
    const unsigned int kvh = blockIdx.y;
    const float scale_log2 = kvh ? scale_log2_h1 : scale_log2_h0;
    const float out_scale = kvh ? out_scale_h1 : out_scale_h0;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3;
    const unsigned int r0 = tile * kTrows;
    if (r0 >= rows) return;
    const unsigned int count = min(counts[tile], static_cast<unsigned int>(kCap));
    const unsigned int* gent = entries + (size_t)tile * kCap;

    // Stage E4M3 Q (128 rows x 256 B) through the ring, and the tile plan.
    for (unsigned int c = tid; c < kWarps * 16 * 16; c += kThreads) {
        const unsigned int m = c >> 4;
        const unsigned int ch = c & 15;
        unsigned char* dst = ring + swz8(m, ch);
        const unsigned int rt = m / kGroup;
        if (m < static_cast<unsigned int>(kM) && r0 + rt < rows) {
            const unsigned int head = kvh * kGroup + m % kGroup;
            cp_async16(dst, q8 + (size_t)(r0 + rt) * q_stride + head * kHd + ch * 16);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
    for (unsigned int i = tid * 4; i < count; i += kThreads * 4) {
        if (i + 4 <= count) {
            cp_async16(ent + i, gent + i);
        } else {
            for (unsigned int j = i; j < count; ++j) ent[j] = gent[j];
        }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    unsigned int qf[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const unsigned int row = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        ldsm_x4(qf[ks], ring + swz8(row, ks * 2 + (lane >> 4)));
    }
    __syncthreads();

    const RowInfo ra = row_info(warp * 16 + g, r0, rows, first_pos);
    const RowInfo rb = row_info(warp * 16 + g + 8, r0, rows, first_pos);
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float ma = -CUDART_INF_F, mb = -CUDART_INF_F, la = 0.0f, lb = 0.0f;

    const unsigned int ntiles = (count + 7) / 8;
    unsigned int warp_rows = 0;
    {
        const unsigned int m_lo = warp * 16;
        const unsigned int m_hi = min(m_lo + 15, static_cast<unsigned int>(kM) - 1);
        for (unsigned int rt = m_lo / kGroup; rt <= m_hi / kGroup; ++rt) warp_rows |= 1u << rt;
    }
    // Prologue: tiles 0 .. STAGES-2 (one commit group each, empty past ntiles).
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (static_cast<unsigned int>(s) < ntiles) {
            load_tile<PV8>(ring + s * Cfg::kStageBytes, kc8, vc, ent, s * 8, count, bt, page_size,
                           k_page_stride, k_tok_stride, v_page_stride, v_tok_stride, kvh, tid);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    for (unsigned int it = 0; it < ntiles; ++it) {
        // Tile `it` is the oldest of STAGES-1 groups in flight. After this barrier
        // every warp is done with tile it-1, whose buffer takes tile it+STAGES-1.
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2) : "memory");
        __syncthreads();
        {
            const unsigned int nt = it + STAGES - 1;
            if (nt < ntiles) {
                load_tile<PV8>(ring + (nt % STAGES) * Cfg::kStageBytes, kc8, vc, ent, nt * 8, count,
                               bt, page_size, k_page_stride, k_tok_stride, v_page_stride,
                               v_tok_stride, kvh, tid);
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        const unsigned char* ks_s = ring + (it % STAGES) * Cfg::kStageBytes;
        const unsigned char* vs_s = ks_s + kTile8;
        {
            const unsigned int slot = it * 8 + (lane & 7);
            const unsigned int need =
                lane < 8 && slot < count ? entry_need(ent[slot], first_pos + r0) : 0u;
            if ((__reduce_or_sync(0xffffffffu, need) & warp_rows) == 0) continue;
        }

        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
#pragma unroll
        for (int kstep = 0; kstep < 8; ++kstep) {
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned int b[4];
                const unsigned int row = (np * 2 + (lane >> 4)) * 8 + (lane & 7);
                ldsm_x4(b, ks_s + swz8(row, kstep * 2 + ((lane >> 3) & 1)));
                mma_e4m3(s[np * 2], qf[kstep], b[0], b[1]);
                mma_e4m3(s[np * 2 + 1], qf[kstep], b[2], b[3]);
            }
        }

        // Mask to each row's own set, then base-2 online softmax.
        const unsigned int base = it * 8;
        float mxa = -CUDART_INF_F, mxb = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const unsigned int slot = base + nt * 2 + (t >> 1);
            const bool slot_live = slot < count;
            const unsigned int e = slot_live ? ent[slot] : 0u;
#pragma unroll
            for (int x = 0; x < 2; ++x) {
                const unsigned int in_block = (t * 2 + x) & 3u;
                s[nt][x] = allowed(ra, e, slot_live, in_block) ? s[nt][x] * scale_log2
                                                               : -CUDART_INF_F;
                s[nt][2 + x] = allowed(rb, e, slot_live, in_block) ? s[nt][2 + x] * scale_log2
                                                                   : -CUDART_INF_F;
                mxa = fmaxf(mxa, s[nt][x]);
                mxb = fmaxf(mxb, s[nt][2 + x]);
            }
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float na = fmaxf(ma, mxa);
        const float nb = fmaxf(mb, mxb);
        const float alpha_a = na == -CUDART_INF_F ? 1.0f : exp2f(ma - na);
        const float alpha_b = nb == -CUDART_INF_F ? 1.0f : exp2f(mb - nb);
        // PV8: P = 256 * exp2(s - m) so it lands in E4M3's normal range; l carries
        // the same factor, so the normalisation cancels it.
        const float p_log2 = PV8 ? 8.0f : 0.0f;
        const float base_a = (na == -CUDART_INF_F ? 0.0f : na) - p_log2;
        const float base_b = (nb == -CUDART_INF_F ? 0.0f : nb) - p_log2;
        ma = na;
        mb = nb;
        float suma = 0.0f, sumb = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - base_a);
            s[nt][1] = exp2f(s[nt][1] - base_a);
            s[nt][2] = exp2f(s[nt][2] - base_b);
            s[nt][3] = exp2f(s[nt][3] - base_b);
            if (!kQuantL) {
                suma += s[nt][0] + s[nt][1];
                sumb += s[nt][2] + s[nt][3];
            }
        }
        if (!kQuantL) {
            la = la * alpha_a + suma;
            lb = lb * alpha_b + sumb;
        }
        if (!kTwoLevel) {
            if (__any_sync(0xffffffffu, alpha_a != 1.0f || alpha_b != 1.0f)) {
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    o[i][0] *= alpha_a;
                    o[i][1] *= alpha_a;
                    o[i][2] *= alpha_b;
                    o[i][3] *= alpha_b;
                }
            }
        }

        if (PV8) {
            const unsigned int a[4] = {
                e4m3x4(s[0][0], s[0][1], s[1][0], s[1][1]),
                e4m3x4(s[0][2], s[0][3], s[1][2], s[1][3]),
                e4m3x4(s[2][0], s[2][1], s[3][0], s[3][1]),
                e4m3x4(s[2][2], s[2][3], s[3][2], s[3][3]),
            };
            if (kQuantL) {
                // Row sums of this tile's E4M3 P (every column of D equals the row sum).
                float dl[4];
                mma_e4m3_zc(dl, a, kE4m3OnesX4, kE4m3OnesX4);
                la = fmaf(la, alpha_a, dl[0]);
                lb = fmaf(lb, alpha_b, dl[2]);
            }
            if (kTwoLevel) {
                // P*V per 16-dim block from C = 0, folded into O with the correction.
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    unsigned int r[4];
                    ldsm_x4_t(r, vs_s + swz8((lane >> 3) * 8 + (lane & 7), c));
                    float de[4], dd[4];
                    mma_e4m3_zc(de, a, prmt(r[0], r[1], 0x6420u), prmt(r[2], r[3], 0x6420u));
                    mma_e4m3_zc(dd, a, prmt(r[0], r[1], 0x7531u), prmt(r[2], r[3], 0x7531u));
                    o[2 * c][0] = fmaf(o[2 * c][0], alpha_a, de[0]);
                    o[2 * c][1] = fmaf(o[2 * c][1], alpha_a, de[1]);
                    o[2 * c][2] = fmaf(o[2 * c][2], alpha_b, de[2]);
                    o[2 * c][3] = fmaf(o[2 * c][3], alpha_b, de[3]);
                    o[2 * c + 1][0] = fmaf(o[2 * c + 1][0], alpha_a, dd[0]);
                    o[2 * c + 1][1] = fmaf(o[2 * c + 1][1], alpha_a, dd[1]);
                    o[2 * c + 1][2] = fmaf(o[2 * c + 1][2], alpha_b, dd[2]);
                    o[2 * c + 1][3] = fmaf(o[2 * c + 1][3], alpha_b, dd[3]);
                }
            } else {
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    unsigned int r[4];
                    ldsm_x4_t(r, vs_s + swz8((lane >> 3) * 8 + (lane & 7), c));
                    mma_e4m3(o[2 * c], a, prmt(r[0], r[1], 0x6420u), prmt(r[2], r[3], 0x6420u));
                    mma_e4m3(o[2 * c + 1], a, prmt(r[0], r[1], 0x7531u),
                             prmt(r[2], r[3], 0x7531u));
                }
            }
        } else {
            const __nv_bfloat16* vs16 = reinterpret_cast<const __nv_bfloat16*>(vs_s);
#pragma unroll
            for (int kstep = 0; kstep < 2; ++kstep) {
                const unsigned int a[4] = {
                    pack_bf16(s[kstep * 2][0], s[kstep * 2][1]),
                    pack_bf16(s[kstep * 2][2], s[kstep * 2][3]),
                    pack_bf16(s[kstep * 2 + 1][0], s[kstep * 2 + 1][1]),
                    pack_bf16(s[kstep * 2 + 1][2], s[kstep * 2 + 1][3]),
                };
#pragma unroll
                for (int dp = 0; dp < 16; ++dp) {
                    unsigned int b[4];
                    const unsigned int row = kstep * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const unsigned int col = dp * 16 + (lane >> 4) * 8;
                    ldsm_x4_t(b, vs16 + swz16(row, col));
                    mma_bf16(o[dp * 2], a, b[0], b[1]);
                    mma_bf16(o[dp * 2 + 1], a, b[2], b[3]);
                }
            }
        }
    }

    if (!kQuantL) {
        // Per-thread partial row sums -> the row's (the ones-MMA already gives every thread the row).
        la += __shfl_xor_sync(0xffffffffu, la, 1);
        la += __shfl_xor_sync(0xffffffffu, la, 2);
        lb += __shfl_xor_sync(0xffffffffu, lb, 1);
        lb += __shfl_xor_sync(0xffffffffu, lb, 2);
    }
    const float ia = la > 0.0f ? out_scale / la : 0.0f;
    const float ib = lb > 0.0f ? out_scale / lb : 0.0f;
    const unsigned int ma_row = warp * 16 + g;
    const unsigned int mb_row = ma_row + 8;
    float amax = 0.0f;
    if (ra.live) {
        store_row<PV8, AMAX>(out, gate, r0 + ra.bit, kvh * kGroup + ma_row % kGroup, o_stride,
                             gate_stride, t, ia, o, 0, amax);
    }
    if (rb.live) {
        store_row<PV8, AMAX>(out, gate, r0 + rb.bit, kvh * kGroup + mb_row % kGroup, o_stride,
                             gate_stride, t, ib, o, 2, amax);
    }
    if constexpr (AMAX == 2) amax = __bfloat162float(__float2bfloat16(amax));
    if constexpr (AMAX == 3) {
        if (ra.live) amax = reread_row<PV8>(out, r0 + ra.bit, kvh * kGroup + ma_row % kGroup, o_stride, t, amax);
        if (rb.live) amax = reread_row<PV8>(out, r0 + rb.bit, kvh * kGroup + mb_row % kGroup, o_stride, t, amax);
    }
    if constexpr (AMAX != 0) {
        // Every thread of the CTA reaches this point (the early return above is CTA-uniform and
        // the tile skip is warp-uniform). The warp's max of its stored |bf16|, then the CTA's (one
        // barrier), then ONE atomicMax on the float bits (non-negative) per CTA, as
        // qwen4_fp8_act_amax's block_max does. The atomic is skipped when the slot already holds at
        // least this value: the slot only grows, so the skip never loses the max, and a stale read
        // only costs an extra atomic. (One same-address atomic per warp, 13k per 8K chunk, cost
        // +0.39 ms per layer in micro 110 and cancelled the saving.)
        __shared__ float s_amax[kWarps];
        for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
        if (lane == 0) s_amax[warp] = amax;
        __syncthreads();
        if (tid == 0) {
            float m = s_amax[0];
#pragma unroll
            for (int w = 1; w < kWarps; ++w) m = fmaxf(m, s_amax[w]);
            const float seen = __uint_as_float(*reinterpret_cast<volatile unsigned int*>(amax_bits));
            if (m > 0.0f && m > seen) atomicMax(amax_bits, __float_as_uint(m));
        }
    }
}

}  // namespace

// Grid (tiles, num_kv_heads), block 256, dynamic shared memory
// STAGES * stage_bytes + kCap * 4 B (stage_bytes = 16 KiB with PV8, 24 KiB
// without). q8 is [rows][24*256] E4M3 at q_stride bytes; K8 (and V8) pages are
// [page][page_size][kv_heads][256] E4M3 at k_page_stride / k_tok_stride bytes;
// without PV8, vc is the BF16 V cache at v_page_stride / v_tok_stride BYTES.
#define QSA_PREFILL_ATTN_FP8_ENTRY(NAME, PV8, ACC, STAGES)                                     \
    extern "C" __global__ void __launch_bounds__(kThreads, 1) NAME(                             \
        const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,                        \
        const uint8_t* __restrict__ vc, const int* __restrict__ bt,                             \
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,      \
        __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,               \
        unsigned int rows, unsigned int first_pos, unsigned int q_stride,                       \
        unsigned int o_stride, unsigned int gate_stride, unsigned int page_size,                \
        unsigned long long k_page_stride, unsigned int k_tok_stride,                            \
        unsigned long long v_page_stride, unsigned int v_tok_stride, float scale_log2_h0,       \
        float scale_log2_h1, float out_scale_h0, float out_scale_h1) {                          \
        qsa_prefill_attn_fp8<PV8, ACC, STAGES>(                                                 \
            q8, kc8, vc, bt, entries, counts, out, gate, rows, first_pos, q_stride, o_stride,   \
            gate_stride, page_size, k_page_stride, k_tok_stride, v_page_stride, v_tok_stride,   \
            scale_log2_h0, scale_log2_h1, out_scale_h0, out_scale_h1);                          \
    }

// AMAX mode of the shipped _amax entries (see store_row). Micro 110c (2026-10-15 17:31Z, L5 8K/32K, every bit gate
// PASS in each mode), core time vs the shipping qkpv core: mode 1 +0.46/+0.34 ms, mode 2 +0.15/+0.15 ms, mode 3
// -0.01/+0.00 ms against the 0.43 ms qwen4_fp8_act_amax pass it removes. Folding inside store_row perturbs the
// store/gate-load schedule; re-reading the stored words afterwards does not. Mode 3 ships.
#ifndef QSA_OPROJ_AMAX_MODE
#define QSA_OPROJ_AMAX_MODE 3
#endif

// ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED: the same kernels with one more argument, `amax_bits`
// (one u32, zeroed by the host before the launch): the absmax of every stored bf16 output value,
// reduced with qwen4_fp8_act_amax's rule, so the o_proj FP8 tensor-scale projection skips its own
// absmax pass over [rows][24*256]. The stored outputs are the same instructions' results.
#define QSA_PREFILL_ATTN_FP8_AMAX_ENTRY(NAME, PV8, ACC, STAGES)                                \
    extern "C" __global__ void __launch_bounds__(kThreads, 1) NAME(                             \
        const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,                        \
        const uint8_t* __restrict__ vc, const int* __restrict__ bt,                             \
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,      \
        __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,               \
        unsigned int rows, unsigned int first_pos, unsigned int q_stride,                       \
        unsigned int o_stride, unsigned int gate_stride, unsigned int page_size,                \
        unsigned long long k_page_stride, unsigned int k_tok_stride,                            \
        unsigned long long v_page_stride, unsigned int v_tok_stride, float scale_log2_h0,       \
        float scale_log2_h1, float out_scale_h0, float out_scale_h1,                            \
        unsigned int* __restrict__ amax_bits) {                                                 \
        qsa_prefill_attn_fp8<PV8, ACC, STAGES, QSA_OPROJ_AMAX_MODE>(                            \
            q8, kc8, vc, bt, entries, counts, out, gate, rows, first_pos, q_stride, o_stride,   \
            gate_stride, page_size, k_page_stride, k_tok_stride, v_page_stride, v_tok_stride,   \
            scale_log2_h0, scale_log2_h1, out_scale_h0, out_scale_h1, amax_bits);               \
    }

QSA_PREFILL_ATTN_FP8_ENTRY(qwen4_qsa_prefill_attn_fp8_qk, false, 0, 2)
// qkpv ships ACC 2. Measured on real L5 dumps at 8K / 32K (micro-queue 42, 23:39Z), against
// t10_skip: ACC 2 -42.2% / -44.5%, ACC 1 (two-level) -35.7% / -38.2% (36 B of spills at 255
// registers), ACC 0 -41.3% / -43.2%. ACC 1 and ACC 2 have identical error vs fp32 (1.597% /
// 1.407%), so GB10's ~21-bit MMA accumulation gives two-level nothing to recover. The gain over
// ACC 0 (1.636% / 1.439%) comes from taking l from the E4M3 P.
QSA_PREFILL_ATTN_FP8_ENTRY(qwen4_qsa_prefill_attn_fp8_qkpv, true, 2, 2)
QSA_PREFILL_ATTN_FP8_AMAX_ENTRY(qwen4_qsa_prefill_attn_fp8_qk_amax, false, 0, 2)
QSA_PREFILL_ATTN_FP8_AMAX_ENTRY(qwen4_qsa_prefill_attn_fp8_qkpv_amax, true, 2, 2)

// qwen4_qsa_prefill_plan_sort (qwen4_qsa_prefill_select.cu) with blocks DEscending inside
// each need group: the same counting sort by need mask (groups ascending, so the all-rows
// group with the sink block comes last), fed from the end of the ascending plan. The E4M3
// core reads its tiles newest-first with the sink last. Deterministic (match_any ranking).
// Dynamic shared memory: (1024 + min(cap, 5120)) * 4 B.
extern "C" __global__ void __launch_bounds__(32) qwen4_qsa_prefill_plan_sort_desc(
    unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
    unsigned int first_pos, unsigned int tile_rows, unsigned int cap) {
    extern __shared__ unsigned int sm[];
    unsigned int* bins = sm;
    unsigned int* vals = sm + 1024;
    const unsigned int tile = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    const unsigned int count = min(min(counts[tile], cap), 5120u);
    unsigned int* dst = entries + (size_t)tile * cap;
    const unsigned int row0_pos = first_pos + tile * tile_rows;
    const auto need_of = [&](unsigned int e) -> unsigned int {
        const unsigned int blk = e & kBlockMask;
        unsigned int need = (e >> kBlockBits) & ((1u << tile_rows) - 1u);
        const int lo = static_cast<int>(blk * kRatio) - static_cast<int>(row0_pos);
        for (int rt = max(lo, 0); rt < min(lo + 3, static_cast<int>(tile_rows)); ++rt) {
            need |= 1u << rt;
        }
        return need;
    };
    for (unsigned int b = lane; b < 1024; b += 32) bins[b] = 0;
    __syncwarp();
    for (unsigned int i = lane; i < count; i += 32) {
        const unsigned int e = dst[i];
        vals[i] = e;
        atomicAdd(&bins[need_of(e)], 1u);
    }
    __syncwarp();
    unsigned int local = 0;
    for (unsigned int j = 0; j < 32; ++j) local += bins[lane * 32 + j];
    unsigned int incl = local;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const unsigned int v = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= static_cast<unsigned int>(off)) incl += v;
    }
    unsigned int run = incl - local;
    __syncwarp();
    for (unsigned int j = 0; j < 32; ++j) {
        const unsigned int c = bins[lane * 32 + j];
        bins[lane * 32 + j] = run;
        run += c;
    }
    __syncwarp();
    const unsigned int lt = (1u << lane) - 1u;
    for (unsigned int i0 = 0; i0 < count; i0 += 32) {
        const unsigned int j = i0 + lane;
        const bool live = j < count;
        const unsigned int e = live ? vals[count - 1 - j] : 0u;
        const unsigned int key = live ? need_of(e) : 0xffffffffu;
        const unsigned int peers = __match_any_sync(0xffffffffu, key);
        if (live) dst[bins[key] + __popc(peers & lt)] = e;
        __syncwarp();
        if (live && (peers & lt) == 0) bins[key] += __popc(peers);
        __syncwarp();
    }
}

namespace {

// 64-byte row (4 x 16B chunks): XOR-swizzled by (row >> 1) & 3 for bank-conflict-free
// 32x64 quarter-tile staging (2 KiB) across both cp.async and ldmatrix.x4 / ldmatrix.x4.trans.
__device__ __forceinline__ unsigned int swz8_qtr(unsigned int row, unsigned int ch_local) {
    return row * 64u + ((ch_local ^ ((row >> 1) & 3u)) << 4);
}

// 128-byte row (8 x 16B chunks): XOR-swizzled by row & 7 for bank-conflict-free
// 32x128 half-tile staging (4 KiB) across both cp.async and ldmatrix.x4 / ldmatrix.x4.trans.
__device__ __forceinline__ unsigned int swz8_half(unsigned int row, unsigned int ch_local) {
    return row * 128u + ((ch_local ^ (row & 7u)) << 4);
}

__device__ __forceinline__ void cp_async16_pred(void* smem, const void* gmem, bool pred) {
    const unsigned int sa = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    const unsigned int src_size = pred ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sa), "l"(gmem), "r"(src_size));
}

// Decoupled-warp E4M3 PV8 attention core (ATLAS_QWEN4_PREFILL_QSA_WARP_DECOUPLE).
// After staging Q into `qf` registers and `ent` into shared memory, the 8 warps decouple:
// each warp owns a private 2-stage sub-tile ring (`w_ring`, 2 x 2 KiB when !HALF_TILE or
// 2 x 4 KiB when HALF_TILE) and iterates only over the tiles `it` whose 8 entries intersect
// its own `warp_rows`, synchronizing with `__syncwarp()` instead of CTA-wide `__syncthreads()`.
// Within each active tile `it`, the warp executes the exact same QK^T MMAs, softmax, and PV8
// MMAs in the exact same order on the exact same operands as `qsa_prefill_attn_fp8<true, ACC, 2, AMAX>`,
// producing 100% bit-for-bit identical outputs.
template <bool HALF_TILE, int ACC, int AMAX = 0>
__device__ __forceinline__ void qsa_prefill_attn_fp8_wdec(
    const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,
    const uint8_t* __restrict__ vc, const int* __restrict__ bt,
    const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
    unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
    unsigned int gate_stride, unsigned int page_size, unsigned long long k_page_stride,
    unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
    float scale_log2_h0, float scale_log2_h1, float out_scale_h0, float out_scale_h1,
    unsigned int* __restrict__ amax_bits = nullptr) {
    constexpr bool kTwoLevel = ACC == 1;
    constexpr bool kQuantL = ACC != 0;
    constexpr unsigned int kWarpStageBytes = HALF_TILE ? 4096u : 2048u;
    constexpr unsigned int kWarpRingBytes = 2u * kWarpStageBytes;
    constexpr unsigned int kTotalRingBytes = kWarps * kWarpRingBytes;
    constexpr int kSubTiles = HALF_TILE ? 2 : 4;
    static_assert(kTotalRingBytes >= kQ8Bytes, "Q staging must fit the warp rings");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned char* ring = smem_raw;
    unsigned int* ent = reinterpret_cast<unsigned int*>(smem_raw + kTotalRingBytes);

    const unsigned int tile = blockIdx.x;
    const unsigned int kvh = blockIdx.y;
    const float scale_log2 = kvh ? scale_log2_h1 : scale_log2_h0;
    const float out_scale = kvh ? out_scale_h1 : out_scale_h0;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3;
    const unsigned int r0 = tile * kTrows;
    if (r0 >= rows) return /*early*/;
    const unsigned int count = min(counts[tile], static_cast<unsigned int>(kCap));
    const unsigned int* gent = entries + (size_t)tile * kCap;

    // Stage E4M3 Q (128 rows x 256 B) through the ring, and the tile plan.
    for (unsigned int c = tid; c < kWarps * 16 * 16; c += kThreads) {
        const unsigned int m = c >> 4;
        const unsigned int ch = c & 15;
        unsigned char* dst = ring + swz8(m, ch);
        const unsigned int rt = m / kGroup;
        if (m < static_cast<unsigned int>(kM) && r0 + rt < rows) {
            const unsigned int head = kvh * kGroup + m % kGroup;
            cp_async16(dst, q8 + (size_t)(r0 + rt) * q_stride + head * kHd + ch * 16);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
    for (unsigned int i = tid * 4; i < count; i += kThreads * 4) {
        if (i + 4 <= count) {
            cp_async16(ent + i, gent + i);
        } else {
            for (unsigned int j = i; j < count; ++j) ent[j] = gent[j];
        }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    unsigned int qf[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const unsigned int row = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        ldsm_x4(qf[ks], ring + swz8(row, ks * 2 + (lane >> 4)));
    }
    __syncthreads();

    const RowInfo ra = row_info(warp * 16 + g, r0, rows, first_pos);
    const RowInfo rb = row_info(warp * 16 + g + 8, r0, rows, first_pos);
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float ma = -CUDART_INF_F, mb = -CUDART_INF_F, la = 0.0f, lb = 0.0f;

    const unsigned int ntiles = (count + 7) / 8;
    unsigned int warp_rows = 0;
    {
        const unsigned int m_lo = warp * 16;
        const unsigned int m_hi = min(m_lo + 15, static_cast<unsigned int>(kM) - 1);
        for (unsigned int rt = m_lo / kGroup; rt <= m_hi / kGroup; ++rt) warp_rows |= 1u << rt;
    }

    unsigned char* const w_ring = ring + warp * kWarpRingBytes;

    const auto find_next_active = [&](unsigned int start_it) -> unsigned int {
        for (unsigned int cand = start_it; cand < ntiles; ++cand) {
            const unsigned int slot = cand * 8u + (lane & 7u);
            const unsigned int need =
                (lane < 8u && slot < count) ? entry_need(ent[slot], first_pos + r0) : 0u;
            if ((__reduce_or_sync(0xffffffffu, need) & warp_rows) != 0u) return cand;
        }
        return ntiles;
    };

    bool live_g = false;
    unsigned int page_g = 0u;
    unsigned int in_page0_g = 0u;
    const auto resolve_tile = [&](unsigned int cur_it) {
        const unsigned int slot_g = cur_it * 8u + g;
        live_g = slot_g < count;
        const unsigned int blk_g = live_g ? (ent[slot_g] & kBlockMask) : 0u;
        const unsigned int tok0 = blk_g * kRatio;
        page_g = live_g ? static_cast<unsigned int>(__ldg(bt + tok0 / page_size)) : 0u;
        in_page0_g = tok0 % page_size;
    };

    // Load one sub-tile (32 tokens x 64 cols when !HALF_TILE, 32 tokens x 128 cols when HALF_TILE)
    // of K or V using only this warp's 32 lanes (g = 0..7 selects the 4-token block, t = 0..3 selects
    // the 4 contiguous 16B chunks within a 64B group). All 4 tokens of block g lie in `page_g`.
    const auto load_kv_sub = [&](unsigned char* st, const uint8_t* __restrict__ pool,
                                 unsigned long long page_stride, unsigned int tok_stride,
                                 unsigned int sub) {
        if constexpr (!HALF_TILE) {
            const unsigned int ch_base = sub * 4u;
            const uint8_t* gmem_g =
                pool + (unsigned long long)page_g * page_stride +
                (unsigned long long)in_page0_g * tok_stride + kvh * kHd + (ch_base + t) * 16u;
#pragma unroll
            for (unsigned int step = 0; step < 4u; ++step) {
                const unsigned int r = step ^ (g & 1u);
                const unsigned int j = g * 4u + r;
                cp_async16_pred(st + swz8_qtr(j, t),
                                gmem_g + (unsigned long long)r * tok_stride, live_g);
            }
        } else {
            const unsigned int ch_base = sub * 8u;
            const uint8_t* gmem_g =
                pool + (unsigned long long)page_g * page_stride +
                (unsigned long long)in_page0_g * tok_stride + kvh * kHd + (ch_base + t) * 16u;
#pragma unroll
            for (unsigned int r = 0; r < 4u; ++r) {
                const unsigned int j = g * 4u + r;
                const uint8_t* row_ptr = gmem_g + (unsigned long long)r * tok_stride;
                cp_async16_pred(st + swz8_half(j, t), row_ptr, live_g);
                cp_async16_pred(st + swz8_half(j, 4u + t), row_ptr + 64u, live_g);
            }
        }
    };

    unsigned int it = find_next_active(0u);
    if (it < ntiles) {
        resolve_tile(it);
        load_kv_sub(w_ring, kc8, k_page_stride, k_tok_stride, 0u);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }

    while (it < ntiles) {
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;

        // QK^T across kSubTiles sub-tiles, ping-ponging between stage 0 and stage 1.
        // At the last K sub-tile (sub == kSubTiles - 1, which reads stage 1), prefetch V sub-tile 0
        // into stage 0 so it overlaps the last K sub-tile's MMAs and the entire online softmax.
#pragma unroll
        for (int sub = 0; sub < kSubTiles; ++sub) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncwarp();
            unsigned char* const next_st = w_ring + (static_cast<unsigned int>(sub + 1) & 1u) * kWarpStageBytes;
            if (sub + 1 < kSubTiles) {
                load_kv_sub(next_st, kc8, k_page_stride, k_tok_stride, static_cast<unsigned int>(sub + 1));
            } else {
                load_kv_sub(next_st, vc, v_page_stride, v_tok_stride, 0u);
            }
            asm volatile("cp.async.commit_group;" ::: "memory");

            const unsigned char* const ks_s = w_ring + (static_cast<unsigned int>(sub) & 1u) * kWarpStageBytes;
            if constexpr (!HALF_TILE) {
#pragma unroll
                for (int ks_sub = 0; ks_sub < 2; ++ks_sub) {
                    const int kstep = sub * 2 + ks_sub;
#pragma unroll
                    for (int np = 0; np < 2; ++np) {
                        unsigned int b[4];
                        const unsigned int row = (np * 2 + (lane >> 4)) * 8 + (lane & 7);
                        ldsm_x4(b, ks_s + swz8_qtr(row, ks_sub * 2 + ((lane >> 3) & 1)));
                        mma_e4m3(s[np * 2], qf[kstep], b[0], b[1]);
                        mma_e4m3(s[np * 2 + 1], qf[kstep], b[2], b[3]);
                    }
                }
            } else {
#pragma unroll
                for (int ks_sub = 0; ks_sub < 4; ++ks_sub) {
                    const int kstep = sub * 4 + ks_sub;
#pragma unroll
                    for (int np = 0; np < 2; ++np) {
                        unsigned int b[4];
                        const unsigned int row = (np * 2 + (lane >> 4)) * 8 + (lane & 7);
                        ldsm_x4(b, ks_s + swz8_half(row, ks_sub * 2 + ((lane >> 3) & 1)));
                        mma_e4m3(s[np * 2], qf[kstep], b[0], b[1]);
                        mma_e4m3(s[np * 2 + 1], qf[kstep], b[2], b[3]);
                    }
                }
            }
        }

        // Mask to each row's own set, then base-2 online softmax.
        const unsigned int base = it * 8u;
        float mxa = -CUDART_INF_F, mxb = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const unsigned int slot = base + nt * 2 + (t >> 1);
            const bool slot_live = slot < count;
            const unsigned int e = slot_live ? ent[slot] : 0u;
#pragma unroll
            for (int x = 0; x < 2; ++x) {
                const unsigned int in_block = (t * 2 + x) & 3u;
                s[nt][x] = allowed(ra, e, slot_live, in_block) ? s[nt][x] * scale_log2
                                                               : -CUDART_INF_F;
                s[nt][2 + x] = allowed(rb, e, slot_live, in_block) ? s[nt][2 + x] * scale_log2
                                                                   : -CUDART_INF_F;
                mxa = fmaxf(mxa, s[nt][x]);
                mxb = fmaxf(mxb, s[nt][2 + x]);
            }
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float na = fmaxf(ma, mxa);
        const float nb = fmaxf(mb, mxb);
        const float alpha_a = na == -CUDART_INF_F ? 1.0f : exp2f(ma - na);
        const float alpha_b = nb == -CUDART_INF_F ? 1.0f : exp2f(mb - nb);
        const float p_log2 = 8.0f;
        const float base_a = (na == -CUDART_INF_F ? 0.0f : na) - p_log2;
        const float base_b = (nb == -CUDART_INF_F ? 0.0f : nb) - p_log2;
        ma = na;
        mb = nb;
        float suma = 0.0f, sumb = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - base_a);
            s[nt][1] = exp2f(s[nt][1] - base_a);
            s[nt][2] = exp2f(s[nt][2] - base_b);
            s[nt][3] = exp2f(s[nt][3] - base_b);
            if (!kQuantL) {
                suma += s[nt][0] + s[nt][1];
                sumb += s[nt][2] + s[nt][3];
            }
        }
        if (!kQuantL) {
            la = la * alpha_a + suma;
            lb = lb * alpha_b + sumb;
        }
        if (!kTwoLevel) {
            if (__any_sync(0xffffffffu, alpha_a != 1.0f || alpha_b != 1.0f)) {
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    o[i][0] *= alpha_a;
                    o[i][1] *= alpha_a;
                    o[i][2] *= alpha_b;
                    o[i][3] *= alpha_b;
                }
            }
        }

        const unsigned int a[4] = {
            e4m3x4(s[0][0], s[0][1], s[1][0], s[1][1]),
            e4m3x4(s[0][2], s[0][3], s[1][2], s[1][3]),
            e4m3x4(s[2][0], s[2][1], s[3][0], s[3][1]),
            e4m3x4(s[2][2], s[2][3], s[3][2], s[3][3]),
        };
        if (kQuantL) {
            float dl[4];
            mma_e4m3_zc(dl, a, kE4m3OnesX4, kE4m3OnesX4);
            la = fmaf(la, alpha_a, dl[0]);
            lb = fmaf(lb, alpha_b, dl[2]);
        }

        // PV8 across kSubTiles sub-tiles. At the last V sub-tile (v_sub == kSubTiles - 1, which
        // reads stage 1), advance `it` to the next active tile and prefetch its K sub-tile 0 into
        // stage 0 so it overlaps the last V sub-tile's MMAs.
#pragma unroll
        for (int v_sub = 0; v_sub < kSubTiles; ++v_sub) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncwarp();
            unsigned char* const next_st = w_ring + (static_cast<unsigned int>(v_sub + 1) & 1u) * kWarpStageBytes;
            if (v_sub + 1 < kSubTiles) {
                load_kv_sub(next_st, vc, v_page_stride, v_tok_stride, static_cast<unsigned int>(v_sub + 1));
                asm volatile("cp.async.commit_group;" ::: "memory");
            } else {
                it = find_next_active(it + 1u);
                if (it < ntiles) {
                    resolve_tile(it);
                    load_kv_sub(next_st, kc8, k_page_stride, k_tok_stride, 0u);
                    asm volatile("cp.async.commit_group;" ::: "memory");
                }
            }

            const unsigned char* const vs_s = w_ring + (static_cast<unsigned int>(v_sub) & 1u) * kWarpStageBytes;
            if constexpr (!HALF_TILE) {
#pragma unroll
                for (int c_sub = 0; c_sub < 4; ++c_sub) {
                    const int c = v_sub * 4 + c_sub;
                    unsigned int r[4];
                    ldsm_x4_t(r, vs_s + swz8_qtr((lane >> 3) * 8 + (lane & 7), c_sub));
                    if (kTwoLevel) {
                        float de[4], dd[4];
                        mma_e4m3_zc(de, a, prmt(r[0], r[1], 0x6420u), prmt(r[2], r[3], 0x6420u));
                        mma_e4m3_zc(dd, a, prmt(r[0], r[1], 0x7531u), prmt(r[2], r[3], 0x7531u));
                        o[2 * c][0] = fmaf(o[2 * c][0], alpha_a, de[0]);
                        o[2 * c][1] = fmaf(o[2 * c][1], alpha_a, de[1]);
                        o[2 * c][2] = fmaf(o[2 * c][2], alpha_b, de[2]);
                        o[2 * c][3] = fmaf(o[2 * c][3], alpha_b, de[3]);
                        o[2 * c + 1][0] = fmaf(o[2 * c + 1][0], alpha_a, dd[0]);
                        o[2 * c + 1][1] = fmaf(o[2 * c + 1][1], alpha_a, dd[1]);
                        o[2 * c + 1][2] = fmaf(o[2 * c + 1][2], alpha_b, dd[2]);
                        o[2 * c + 1][3] = fmaf(o[2 * c + 1][3], alpha_b, dd[3]);
                    } else {
                        mma_e4m3(o[2 * c], a, prmt(r[0], r[1], 0x6420u), prmt(r[2], r[3], 0x6420u));
                        mma_e4m3(o[2 * c + 1], a, prmt(r[0], r[1], 0x7531u),
                                 prmt(r[2], r[3], 0x7531u));
                    }
                }
            } else {
#pragma unroll
                for (int c_sub = 0; c_sub < 8; ++c_sub) {
                    const int c = v_sub * 8 + c_sub;
                    unsigned int r[4];
                    ldsm_x4_t(r, vs_s + swz8_half((lane >> 3) * 8 + (lane & 7), c_sub));
                    if (kTwoLevel) {
                        float de[4], dd[4];
                        mma_e4m3_zc(de, a, prmt(r[0], r[1], 0x6420u), prmt(r[2], r[3], 0x6420u));
                        mma_e4m3_zc(dd, a, prmt(r[0], r[1], 0x7531u), prmt(r[2], r[3], 0x7531u));
                        o[2 * c][0] = fmaf(o[2 * c][0], alpha_a, de[0]);
                        o[2 * c][1] = fmaf(o[2 * c][1], alpha_a, de[1]);
                        o[2 * c][2] = fmaf(o[2 * c][2], alpha_b, de[2]);
                        o[2 * c][3] = fmaf(o[2 * c][3], alpha_b, de[3]);
                        o[2 * c + 1][0] = fmaf(o[2 * c + 1][0], alpha_a, dd[0]);
                        o[2 * c + 1][1] = fmaf(o[2 * c + 1][1], alpha_a, dd[1]);
                        o[2 * c + 1][2] = fmaf(o[2 * c + 1][2], alpha_b, dd[2]);
                        o[2 * c + 1][3] = fmaf(o[2 * c + 1][3], alpha_b, dd[3]);
                    } else {
                        mma_e4m3(o[2 * c], a, prmt(r[0], r[1], 0x6420u), prmt(r[2], r[3], 0x6420u));
                        mma_e4m3(o[2 * c + 1], a, prmt(r[0], r[1], 0x7531u),
                                 prmt(r[2], r[3], 0x7531u));
                    }
                }
            }
        }
    }

    if (!kQuantL) {
        la += __shfl_xor_sync(0xffffffffu, la, 1);
        la += __shfl_xor_sync(0xffffffffu, la, 2);
        lb += __shfl_xor_sync(0xffffffffu, lb, 1);
        lb += __shfl_xor_sync(0xffffffffu, lb, 2);
    }
    const float ia = la > 0.0f ? out_scale / la : 0.0f;
    const float ib = lb > 0.0f ? out_scale / lb : 0.0f;
    const unsigned int ma_row = warp * 16 + g;
    const unsigned int mb_row = ma_row + 8;
    float amax = 0.0f;
    if (ra.live) {
        store_row<true, AMAX>(out, gate, r0 + ra.bit, kvh * kGroup + ma_row % kGroup, o_stride,
                              gate_stride, t, ia, o, 0, amax);
    }
    if (rb.live) {
        store_row<true, AMAX>(out, gate, r0 + rb.bit, kvh * kGroup + mb_row % kGroup, o_stride,
                              gate_stride, t, ib, o, 2, amax);
    }
    if constexpr (AMAX == 2) amax = __bfloat162float(__float2bfloat16(amax));
    if constexpr (AMAX == 3) {
        if (ra.live) amax = reread_row<true>(out, r0 + ra.bit, kvh * kGroup + ma_row % kGroup, o_stride, t, amax);
        if (rb.live) amax = reread_row<true>(out, r0 + rb.bit, kvh * kGroup + mb_row % kGroup, o_stride, t, amax);
    }
    if constexpr (AMAX != 0) {
        __shared__ float s_amax[kWarps];
        for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
        if (lane == 0) s_amax[warp] = amax;
        __syncthreads();
        if (tid == 0) {
            float m = s_amax[0];
#pragma unroll
            for (int w = 1; w < kWarps; ++w) m = fmaxf(m, s_amax[w]);
            const float seen = __uint_as_float(*reinterpret_cast<volatile unsigned int*>(amax_bits));
            if (m > 0.0f && m > seen) atomicMax(amax_bits, __float_as_uint(m));
        }
    }
}

}  // namespace

#define QSA_PREFILL_ATTN_FP8_WDEC_ENTRY(NAME, HALF_TILE, ACC)                                   \
    extern "C" __global__ void __launch_bounds__(kThreads, 1) NAME(                             \
        const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,                        \
        const uint8_t* __restrict__ vc, const int* __restrict__ bt,                             \
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,      \
        __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,               \
        unsigned int rows, unsigned int first_pos, unsigned int q_stride,                       \
        unsigned int o_stride, unsigned int gate_stride, unsigned int page_size,                \
        unsigned long long k_page_stride, unsigned int k_tok_stride,                            \
        unsigned long long v_page_stride, unsigned int v_tok_stride, float scale_log2_h0,       \
        float scale_log2_h1, float out_scale_h0, float out_scale_h1) {                          \
        qsa_prefill_attn_fp8_wdec<HALF_TILE, ACC, 0>(                                           \
            q8, kc8, vc, bt, entries, counts, out, gate, rows, first_pos, q_stride, o_stride,   \
            gate_stride, page_size, k_page_stride, k_tok_stride, v_page_stride, v_tok_stride,   \
            scale_log2_h0, scale_log2_h1, out_scale_h0, out_scale_h1);                          \
    }

#define QSA_PREFILL_ATTN_FP8_WDEC_AMAX_ENTRY(NAME, HALF_TILE, ACC)                              \
    extern "C" __global__ void __launch_bounds__(kThreads, 1) NAME(                             \
        const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,                        \
        const uint8_t* __restrict__ vc, const int* __restrict__ bt,                             \
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,      \
        __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,               \
        unsigned int rows, unsigned int first_pos, unsigned int q_stride,                       \
        unsigned int o_stride, unsigned int gate_stride, unsigned int page_size,                \
        unsigned long long k_page_stride, unsigned int k_tok_stride,                            \
        unsigned long long v_page_stride, unsigned int v_tok_stride, float scale_log2_h0,       \
        float scale_log2_h1, float out_scale_h0, float out_scale_h1,                            \
        unsigned int* __restrict__ amax_bits) {                                                 \
        qsa_prefill_attn_fp8_wdec<HALF_TILE, ACC, QSA_OPROJ_AMAX_MODE>(                         \
            q8, kc8, vc, bt, entries, counts, out, gate, rows, first_pos, q_stride, o_stride,   \
            gate_stride, page_size, k_page_stride, k_tok_stride, v_page_stride, v_tok_stride,   \
            scale_log2_h0, scale_log2_h1, out_scale_h0, out_scale_h1, amax_bits);               \
    }

#define QSA_QP_AMAX_SYM(suf) qwen4_qsa_prefill_attn_fp8_qkpv_ ## amax_ ## suf
QSA_PREFILL_ATTN_FP8_WDEC_ENTRY(qwen4_qsa_prefill_attn_fp8_qkpv_wdec, false, 2)
QSA_PREFILL_ATTN_FP8_WDEC_AMAX_ENTRY(QSA_QP_AMAX_SYM(wdec), false, 2)
QSA_PREFILL_ATTN_FP8_WDEC_ENTRY(qwen4_qsa_prefill_attn_fp8_qkpv_wdec8k, true, 2)
QSA_PREFILL_ATTN_FP8_WDEC_AMAX_ENTRY(QSA_QP_AMAX_SYM(wdec8k), true, 2)

namespace {

__device__ __forceinline__ void cp_async16_l2_128b(void* smem, const void* gmem) {
    const unsigned int sa = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" ::"r"(sa), "l"(gmem));
}

template <int AMAX>
__device__ __forceinline__ void store_row_pipe(__nv_bfloat16* __restrict__ out,
                                               const __nv_bfloat16* __restrict__ gate,
                                               unsigned int row, unsigned int head,
                                               unsigned int o_stride, unsigned int gate_stride,
                                               unsigned int t, float inv_l,
                                               const float (&o)[32][4], int half, float& amax) {
    __nv_bfloat16* dst = out + (size_t)row * o_stride + head * kHd;
    const __nv_bfloat16* gsrc = gate ? gate + (size_t)row * gate_stride + head * kHd : nullptr;
    const auto norm = [&](float acc) { return __bfloat162float(__float2bfloat16(acc * inv_l)); };
    const auto gated = [](float x, __nv_bfloat16 gv) {
        return x * (1.0f / (1.0f + expf(-__bfloat162float(gv))));
    };
    if (gsrc) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int c0 = b * 4;
            const unsigned int d0 = (c0 + 0) * 16 + t * 4;
            const unsigned int d1 = (c0 + 1) * 16 + t * 4;
            const unsigned int d2 = (c0 + 2) * 16 + t * 4;
            const unsigned int d3 = (c0 + 3) * 16 + t * 4;
            uint2 gw[4];
            asm volatile(
                "{\n\t"
                "ld.global.cs.v2.u32 {%0, %1}, [%8];\n\t"
                "ld.global.cs.v2.u32 {%2, %3}, [%9];\n\t"
                "ld.global.cs.v2.u32 {%4, %5}, [%10];\n\t"
                "ld.global.cs.v2.u32 {%6, %7}, [%11];\n\t"
                "}\n"
                : "=r"(gw[0].x), "=r"(gw[0].y), "=r"(gw[1].x), "=r"(gw[1].y),
                  "=r"(gw[2].x), "=r"(gw[2].y), "=r"(gw[3].x), "=r"(gw[3].y)
                : "l"(gsrc + d0), "l"(gsrc + d1), "l"(gsrc + d2), "l"(gsrc + d3));
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int c = c0 + k;
                const unsigned int dc = c * 16 + t * 4;
                float x0 = norm(o[2 * c][half]);
                float x1 = norm(o[2 * c + 1][half]);
                float x2 = norm(o[2 * c][half + 1]);
                float x3 = norm(o[2 * c + 1][half + 1]);
                const __nv_bfloat162 g01 = *reinterpret_cast<const __nv_bfloat162*>(&gw[k].x);
                const __nv_bfloat162 g23 = *reinterpret_cast<const __nv_bfloat162*>(&gw[k].y);
                x0 = gated(x0, g01.x);
                x1 = gated(x1, g01.y);
                x2 = gated(x2, g23.x);
                x3 = gated(x3, g23.y);
                const unsigned int w01 = pack_bf16(x0, x1);
                const unsigned int w23 = pack_bf16(x2, x3);
                *reinterpret_cast<uint2*>(dst + dc) = make_uint2(w01, w23);
                if constexpr (AMAX == 1) amax = amax_bf16x2(amax_bf16x2(amax, w01), w23);
                if constexpr (AMAX == 2)
                    amax =
                        fmaxf(fmaxf(fmaxf(fmaxf(amax, fabsf(x0)), fabsf(x1)), fabsf(x2)), fabsf(x3));
            }
        }
    } else {
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            const unsigned int d0 = c * 16 + t * 4;
            const float x0 = norm(o[2 * c][half]);
            const float x1 = norm(o[2 * c + 1][half]);
            const float x2 = norm(o[2 * c][half + 1]);
            const float x3 = norm(o[2 * c + 1][half + 1]);
            const unsigned int w01 = pack_bf16(x0, x1);
            const unsigned int w23 = pack_bf16(x2, x3);
            *reinterpret_cast<uint2*>(dst + d0) = make_uint2(w01, w23);
            if constexpr (AMAX == 1) amax = amax_bf16x2(amax_bf16x2(amax, w01), w23);
            if constexpr (AMAX == 2)
                amax = fmaxf(fmaxf(fmaxf(fmaxf(amax, fabsf(x0)), fabsf(x1)), fabsf(x2)), fabsf(x3));
        }
    }
}

__device__ __forceinline__ unsigned int pack_e4m3x4_pipe(float x0, float x1, float x2, float x3) {
    unsigned int packed;
    asm("{\n\t"
        ".reg .b16 h0, h1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h0, %2, %1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h1, %4, %3;\n\t"
        "mov.b32 %0, {h0, h1};\n\t"
        "}"
        : "=r"(packed)
        : "f"(x0), "f"(x1), "f"(x2), "f"(x3));
    return packed;
}

// Direct static-prescaled E4M3 epilogue for ATLAS_QWEN4_PREFILL_QSA_E4M3_OUT:
// inv_l already folds (inv_scale / s_v[h]) / l, so each element lands directly in
// [-448, +448] and saturates cleanly via cvt.rn.satfinite.e4m3x2.f32 with zero NaNs.
__device__ __forceinline__ void store_row_pipe_e4m3(uint8_t* __restrict__ out,
                                                    const __nv_bfloat16* __restrict__ gate,
                                                    unsigned int row, unsigned int head,
                                                    unsigned int o_stride, unsigned int gate_stride,
                                                    unsigned int t, float inv_l,
                                                    const float (&o)[32][4], int half) {
    uint8_t* dst = out + (size_t)row * o_stride + head * kHd;
    const __nv_bfloat16* gsrc = gate ? gate + (size_t)row * gate_stride + head * kHd : nullptr;
    const auto gated = [](float x, __nv_bfloat16 gv) {
        return x * (1.0f / (1.0f + expf(-__bfloat162float(gv))));
    };
    if (gsrc) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int c0 = b * 4;
            const unsigned int d0 = (c0 + 0) * 16 + t * 4;
            const unsigned int d1 = (c0 + 1) * 16 + t * 4;
            const unsigned int d2 = (c0 + 2) * 16 + t * 4;
            const unsigned int d3 = (c0 + 3) * 16 + t * 4;
            uint2 gw[4];
            asm volatile(
                "{\n\t"
                "ld.global.cs.v2.u32 {%0, %1}, [%8];\n\t"
                "ld.global.cs.v2.u32 {%2, %3}, [%9];\n\t"
                "ld.global.cs.v2.u32 {%4, %5}, [%10];\n\t"
                "ld.global.cs.v2.u32 {%6, %7}, [%11];\n\t"
                "}\n"
                : "=r"(gw[0].x), "=r"(gw[0].y), "=r"(gw[1].x), "=r"(gw[1].y),
                  "=r"(gw[2].x), "=r"(gw[2].y), "=r"(gw[3].x), "=r"(gw[3].y)
                : "l"(gsrc + d0), "l"(gsrc + d1), "l"(gsrc + d2), "l"(gsrc + d3));
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int c = c0 + k;
                const unsigned int dc = c * 16 + t * 4;
                const __nv_bfloat162 g01 = *reinterpret_cast<const __nv_bfloat162*>(&gw[k].x);
                const __nv_bfloat162 g23 = *reinterpret_cast<const __nv_bfloat162*>(&gw[k].y);
                const float x0 = gated(o[2 * c][half] * inv_l, g01.x);
                const float x1 = gated(o[2 * c + 1][half] * inv_l, g01.y);
                const float x2 = gated(o[2 * c][half + 1] * inv_l, g23.x);
                const float x3 = gated(o[2 * c + 1][half + 1] * inv_l, g23.y);
                *reinterpret_cast<unsigned int*>(dst + dc) = pack_e4m3x4_pipe(x0, x1, x2, x3);
            }
        }
    } else {
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            const unsigned int d0 = c * 16 + t * 4;
            const float x0 = o[2 * c][half] * inv_l;
            const float x1 = o[2 * c + 1][half] * inv_l;
            const float x2 = o[2 * c][half + 1] * inv_l;
            const float x3 = o[2 * c + 1][half + 1] * inv_l;
            *reinterpret_cast<unsigned int*>(dst + d0) = pack_e4m3x4_pipe(x0, x1, x2, x3);
        }
    }
}

__device__ __forceinline__ float reread_row_pipe(const __nv_bfloat16* __restrict__ out,
                                                 unsigned int row, unsigned int head,
                                                 unsigned int o_stride, unsigned int t,
                                                 float amax) {
    const __nv_bfloat16* src = out + (size_t)row * o_stride + head * kHd;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int c0 = b * 4;
        unsigned int w[8];
        asm volatile(
            "{\n\t"
            "ld.global.cg.v2.u32 {%0, %1}, [%8];\n\t"
            "ld.global.cg.v2.u32 {%2, %3}, [%9];\n\t"
            "ld.global.cg.v2.u32 {%4, %5}, [%10];\n\t"
            "ld.global.cg.v2.u32 {%6, %7}, [%11];\n\t"
            "}\n"
            : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]),
              "=r"(w[4]), "=r"(w[5]), "=r"(w[6]), "=r"(w[7])
            : "l"(src + (c0 + 0) * 16 + t * 4), "l"(src + (c0 + 1) * 16 + t * 4),
              "l"(src + (c0 + 2) * 16 + t * 4), "l"(src + (c0 + 3) * 16 + t * 4)
            : "memory");
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            amax = amax_bf16x2(amax_bf16x2(amax, w[2 * k]), w[2 * k + 1]);
        }
    }
    return amax;
}

__device__ __forceinline__ unsigned int entry_need_pipe(unsigned int e, unsigned int row0_pos) {
    const unsigned int blk = e & kBlockMask;
    unsigned int need = (e >> kBlockBits) & ((1u << kTrows) - 1u);
    const unsigned int s = blk * kRatio + 2u - row0_pos;
    if (s < static_cast<unsigned int>(kTrows + 2)) {
        need |= ((7u << s) >> 2) & ((1u << kTrows) - 1u);
    }
    return need;
}

// Split-sub-CTA E4M3 PV8 attention core (`qsa_prefill_attn_fp8_sub<SUB, AMAX>`).
// Splits the 8 warps of a 10-row tile across `SUB = 2` (Half-CTA: 4 warps = 128 threads/CTA,
// grid.y = num_kv_heads * 2) or `SUB = 4` (Quarter-CTA: 2 warps = 64 threads/CTA,
// grid.y = num_kv_heads * 4), while keeping the exact 10-row `entries[tile, 0..count-1]` plan
// and the exact 16 KiB full-tile swz8 KV staging layout (1 barrier per active tile).
//
// Because `qwen4_qsa_prefill_plan_sort_desc` clusters entries by their 10-bit row-need mask,
// a sub-CTA owning only 2-4 warps (`cta_rows` = 2..6 rows) skips 35%..68% of all 32-token KV
// tiles completely (no cp.async KV load, no pipeline stage, no barrier), and its 32,768 B dynamic
// (+ 64..80 B static) shared memory footprint allows 2 CTAs/SM concurrently on SM121.
// Every warp `warp = sub * kSubWarps + warp_local` visits the exact same active tiles in the
// exact same ascending `it` order and executes the exact same QK^T, online softmax, and PV8 MMA
// sequence as `qsa_prefill_attn_fp8<true, 2, 2, AMAX>`, producing 100% bit-identical outputs.
template <int SUB, int AMAX = 0>
__device__ __forceinline__ void qsa_prefill_attn_fp8_sub(
    const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,
    const uint8_t* __restrict__ vc, const int* __restrict__ bt,
    const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
    unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
    unsigned int gate_stride, unsigned int page_size, unsigned long long k_page_stride,
    unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
    float scale_log2_h0, float scale_log2_h1, float out_scale_h0, float out_scale_h1,
    unsigned int* __restrict__ amax_bits = nullptr) {
    static_assert(SUB == 2 || SUB == 4, "SUB must be 2 (half-CTA) or 4 (quarter-CTA)");
    using Cfg = Fp8Cfg<true>;
    constexpr unsigned int kSubWarps = static_cast<unsigned int>(kWarps / SUB);
    constexpr unsigned int kSubThreads = kSubWarps * 32u;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned char* const ring = smem_raw;
    __shared__ unsigned int s_ent[2][8];

    const unsigned int tile = blockIdx.x;
    const unsigned int sub = blockIdx.y % static_cast<unsigned int>(SUB);
    const unsigned int kvh = blockIdx.y / static_cast<unsigned int>(SUB);
    const float scale_log2 = kvh ? scale_log2_h1 : scale_log2_h0;
    const float out_scale = kvh ? out_scale_h1 : out_scale_h0;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_local = tid >> 5;
    const unsigned int warp = sub * kSubWarps + warp_local;
    const unsigned int lane = tid & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const unsigned int r0 = tile * kTrows;
    if (r0 >= rows || page_size == 0u || (page_size & (page_size - 1u)) != 0u) return /*early*/;
    const unsigned int count = min(counts[tile], static_cast<unsigned int>(kCap));
    const unsigned int* const gent = entries + (size_t)tile * kCap;
    const unsigned int page_shift = static_cast<unsigned int>(__ffs(page_size) - 1);
    const unsigned int page_mask = page_size - 1u;

    // Stage only this sub-CTA's E4M3 Q rows (kSubWarps * 16 rows x 256 B) into `ring`.
    for (unsigned int c = tid; c < kSubWarps * 16u * 16u; c += kSubThreads) {
        const unsigned int m_local = c >> 4;
        const unsigned int m = sub * (kSubWarps * 16u) + m_local;
        const unsigned int ch = c & 15u;
        unsigned char* const dst = ring + swz8(m_local, ch);
        const unsigned int rt = m / kGroup;
        if (m < static_cast<unsigned int>(kM) && r0 + rt < rows) {
            const unsigned int head = kvh * kGroup + m % kGroup;
            cp_async16_l2_128b(dst, q8 + (size_t)(r0 + rt) * q_stride + head * kHd + ch * 16u);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    unsigned int qf[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const unsigned int row_local = warp_local * 16u + (lane & 7u) + ((lane >> 3) & 1u) * 8u;
        ldsm_x4(qf[ks], ring + swz8(row_local, ks * 2 + (lane >> 4)));
    }
    __syncthreads();

    unsigned int cta_rows = 0u;
    {
        const unsigned int m_lo = sub * (kSubWarps * 16u);
        const unsigned int m_hi = min(m_lo + kSubWarps * 16u - 1u, static_cast<unsigned int>(kM) - 1u);
        for (unsigned int rt = m_lo / kGroup; rt <= m_hi / kGroup; ++rt) cta_rows |= 1u << rt;
    }

    const unsigned int ntiles = (count + 7u) / 8u;
    const unsigned int row0_pos = first_pos + r0;

    // Lockstep warp-ballot active-tile iterator with vectorized uint4 entry loads:
    unsigned int scan_it0 = 0u;
    unsigned int scan_mask = 0u;
    const auto pop_next_active_it = [&]() -> unsigned int {
        while (scan_mask == 0u) {
            if (scan_it0 >= ntiles) return 0xffffffffu;
            const unsigned int it = scan_it0 + lane;
            scan_it0 += 32u;
            unsigned int need = 0u;
            if (it < ntiles) {
                const unsigned int base = it * 8u;
                if (base + 8u <= count) {
                    const uint4 v0 = __ldg(reinterpret_cast<const uint4*>(gent + base));
                    const uint4 v1 = __ldg(reinterpret_cast<const uint4*>(gent + base + 4u));
                    need = entry_need_pipe(v0.x, row0_pos) | entry_need_pipe(v0.y, row0_pos) |
                           entry_need_pipe(v0.z, row0_pos) | entry_need_pipe(v0.w, row0_pos) |
                           entry_need_pipe(v1.x, row0_pos) | entry_need_pipe(v1.y, row0_pos) |
                           entry_need_pipe(v1.z, row0_pos) | entry_need_pipe(v1.w, row0_pos);
                } else {
#pragma unroll
                    for (unsigned int j = 0u; j < 8u; ++j) {
                        if (base + j < count) {
                            need |= entry_need_pipe(__ldg(gent + base + j), row0_pos);
                        }
                    }
                }
            }
            scan_mask = __ballot_sync(0xffffffffu, (need & cta_rows) != 0u);
        }
        const unsigned int bit = __ffs(scan_mask) - 1u;
        scan_mask &= scan_mask - 1u;
        return (scan_it0 - 32u) + bit;
    };

    const RowInfo ra = row_info(warp * 16u + g, r0, rows, first_pos);
    const RowInfo rb = row_info(warp * 16u + g + 8u, r0, rows, first_pos);
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float ma = -CUDART_INF_F, mb = -CUDART_INF_F, la = 0.0f, lb = 0.0f;

    unsigned int warp_rows = 0u;
    {
        const unsigned int m_lo = warp * 16u;
        const unsigned int m_hi = min(m_lo + 15u, static_cast<unsigned int>(kM) - 1u);
        for (unsigned int rt = m_lo / kGroup; rt <= m_hi / kGroup; ++rt) warp_rows |= 1u << rt;
    }

    const unsigned int k_row0 = (lane >> 4) * 8u + (lane & 7u);
    const unsigned int k_row1 = k_row0 + 16u;
    const unsigned int k_col_off = (lane >> 3) & 1u;
    const unsigned int vs_row = (lane >> 3) * 8u + (lane & 7u);

    const auto load_tile_sub = [&](unsigned char* st, unsigned int stage_idx, unsigned int base) {
        if (tid < 8u) {
            s_ent[stage_idx][tid] = (base + tid < count) ? __ldg(gent + base + tid) : 0u;
        }
        unsigned char* const ks = st;
        unsigned char* const vs = st + kTile8;
        for (unsigned int c = tid; c < kBc * 16u; c += kSubThreads) {
            const unsigned int j = c >> 4;
            const unsigned int ch = c & 15u;
            const unsigned int slot = base + (j >> 2);
            unsigned char* const kd = ks + swz8(j, ch);
            unsigned char* const vd = vs + swz8(j, ch);
            if (slot < count) {
                const unsigned int tok = (__ldg(gent + slot) & kBlockMask) * kRatio + (j & 3u);
                const unsigned int page = static_cast<unsigned int>(__ldg(bt + (tok >> page_shift)));
                const size_t in_page = tok & page_mask;
                cp_async16_l2_128b(kd, kc8 + page * k_page_stride +
                                           in_page * k_tok_stride + kvh * kHd + ch * 16u);
                cp_async16_l2_128b(vd, vc + page * v_page_stride +
                                           in_page * v_tok_stride + kvh * kHd + ch * 16u);
            } else {
                *reinterpret_cast<uint4*>(kd) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(vd) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    unsigned int cur_it = pop_next_active_it();
    if (cur_it != 0xffffffffu) {
        load_tile_sub(ring, 0u, cur_it * 8u);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");

    for (unsigned int stage = 0u; cur_it != 0xffffffffu; stage ^= 1u) {
        const unsigned int next_it = pop_next_active_it();
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (next_it != 0xffffffffu) {
            load_tile_sub(ring + (stage ^ 1u) * Cfg::kStageBytes, stage ^ 1u, next_it * 8u);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");

        const unsigned int base = cur_it * 8u;
        const unsigned char* const ks_s = ring + stage * Cfg::kStageBytes;
        const unsigned char* const vs_s = ks_s + kTile8;
        cur_it = next_it;

        bool do_h0 = false;
        bool do_h1 = false;
        {
            const unsigned int j = lane & 7u;
            const unsigned int need =
                (lane < 8u && base + j < count) ? entry_need_pipe(s_ent[stage][j], row0_pos) : 0u;
            const unsigned int need01 =
                __reduce_or_sync(0xffffffffu, lane < 4u ? need : (need << 16));
            do_h0 = (need01 & warp_rows) != 0u;
            do_h1 = ((need01 >> 16) & warp_rows) != 0u;
            if (!do_h0 && !do_h1) continue;
        }

        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
        if (do_h0 && do_h1) {
#pragma unroll
            for (int kstep = 0; kstep < 8; ++kstep) {
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    unsigned int b[4];
                    const unsigned int row = np ? k_row1 : k_row0;
                    ldsm_x4(b, ks_s + swz8(row, kstep * 2 + k_col_off));
                    mma_e4m3(s[np * 2], qf[kstep], b[0], b[1]);
                    mma_e4m3(s[np * 2 + 1], qf[kstep], b[2], b[3]);
                }
            }
        } else if (do_h0) {
#pragma unroll
            for (int kstep = 0; kstep < 8; ++kstep) {
                unsigned int b[4];
                ldsm_x4(b, ks_s + swz8(k_row0, kstep * 2 + k_col_off));
                mma_e4m3(s[0], qf[kstep], b[0], b[1]);
                mma_e4m3(s[1], qf[kstep], b[2], b[3]);
            }
        } else {
#pragma unroll
            for (int kstep = 0; kstep < 8; ++kstep) {
                unsigned int b[4];
                ldsm_x4(b, ks_s + swz8(k_row1, kstep * 2 + k_col_off));
                mma_e4m3(s[2], qf[kstep], b[0], b[1]);
                mma_e4m3(s[3], qf[kstep], b[2], b[3]);
            }
        }

        float mxa = -CUDART_INF_F, mxb = -CUDART_INF_F;
        if (do_h0) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const unsigned int j = nt * 2 + (t >> 1);
                const bool slot_live = base + j < count;
                const unsigned int e = s_ent[stage][j];
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    const unsigned int in_block = (t * 2 + x) & 3u;
                    s[nt][x] = allowed(ra, e, slot_live, in_block) ? s[nt][x] * scale_log2
                                                                   : -CUDART_INF_F;
                    s[nt][2 + x] = allowed(rb, e, slot_live, in_block) ? s[nt][2 + x] * scale_log2
                                                                       : -CUDART_INF_F;
                    mxa = fmaxf(mxa, s[nt][x]);
                    mxb = fmaxf(mxb, s[nt][2 + x]);
                }
            }
        }
        if (do_h1) {
#pragma unroll
            for (int nt = 2; nt < 4; ++nt) {
                const unsigned int j = nt * 2 + (t >> 1);
                const bool slot_live = base + j < count;
                const unsigned int e = s_ent[stage][j];
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    const unsigned int in_block = (t * 2 + x) & 3u;
                    s[nt][x] = allowed(ra, e, slot_live, in_block) ? s[nt][x] * scale_log2
                                                                   : -CUDART_INF_F;
                    s[nt][2 + x] = allowed(rb, e, slot_live, in_block) ? s[nt][2 + x] * scale_log2
                                                                       : -CUDART_INF_F;
                    mxa = fmaxf(mxa, s[nt][x]);
                    mxb = fmaxf(mxb, s[nt][2 + x]);
                }
            }
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float na = fmaxf(ma, mxa);
        const float nb = fmaxf(mb, mxb);
        const float alpha_a = na == -CUDART_INF_F ? 1.0f : exp2f(ma - na);
        const float alpha_b = nb == -CUDART_INF_F ? 1.0f : exp2f(mb - nb);
        const float p_log2 = 8.0f;
        const float base_a = (na == -CUDART_INF_F ? 0.0f : na) - p_log2;
        const float base_b = (nb == -CUDART_INF_F ? 0.0f : nb) - p_log2;
        ma = na;
        mb = nb;
        unsigned int a[4] = {0u, 0u, 0u, 0u};
        if (do_h0) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                s[nt][0] = exp2f(s[nt][0] - base_a);
                s[nt][1] = exp2f(s[nt][1] - base_a);
                s[nt][2] = exp2f(s[nt][2] - base_b);
                s[nt][3] = exp2f(s[nt][3] - base_b);
            }
            a[0] = e4m3x4(s[0][0], s[0][1], s[1][0], s[1][1]);
            a[1] = e4m3x4(s[0][2], s[0][3], s[1][2], s[1][3]);
        }
        if (do_h1) {
#pragma unroll
            for (int nt = 2; nt < 4; ++nt) {
                s[nt][0] = exp2f(s[nt][0] - base_a);
                s[nt][1] = exp2f(s[nt][1] - base_a);
                s[nt][2] = exp2f(s[nt][2] - base_b);
                s[nt][3] = exp2f(s[nt][3] - base_b);
            }
            a[2] = e4m3x4(s[2][0], s[2][1], s[3][0], s[3][1]);
            a[3] = e4m3x4(s[2][2], s[2][3], s[3][2], s[3][3]);
        }
        float dl[4];
        mma_e4m3_zc(dl, a, kE4m3OnesX4, kE4m3OnesX4);
        la = fmaf(la, alpha_a, dl[0]);
        lb = fmaf(lb, alpha_b, dl[2]);

        if (__any_sync(0xffffffffu, alpha_a != 1.0f || alpha_b != 1.0f)) {
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                o[i][0] *= alpha_a;
                o[i][1] *= alpha_a;
                o[i][2] *= alpha_b;
                o[i][3] *= alpha_b;
            }
        }
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            unsigned int r[4];
            ldsm_x4_t(r, vs_s + swz8(vs_row, c));
            mma_e4m3(o[2 * c], a, prmt(r[0], r[1], 0x6420u), prmt(r[2], r[3], 0x6420u));
            mma_e4m3(o[2 * c + 1], a, prmt(r[0], r[1], 0x7531u),
                     prmt(r[2], r[3], 0x7531u));
        }
    }

    const float ia = la > 0.0f ? out_scale / la : 0.0f;
    const float ib = lb > 0.0f ? out_scale / lb : 0.0f;
    const unsigned int ma_row = warp * 16u + g;
    const unsigned int mb_row = ma_row + 8u;
    float amax = 0.0f;
    if (ra.live) {
        store_row_pipe<AMAX>(out, gate, r0 + ra.bit, kvh * kGroup + ma_row % kGroup, o_stride,
                             gate_stride, t, ia, o, 0, amax);
    }
    if (rb.live) {
        store_row_pipe<AMAX>(out, gate, r0 + rb.bit, kvh * kGroup + mb_row % kGroup, o_stride,
                             gate_stride, t, ib, o, 2, amax);
    }
    if constexpr (AMAX == 2) amax = __bfloat162float(__float2bfloat16(amax));
    if constexpr (AMAX == 3) {
        if (ra.live) amax = reread_row_pipe(out, r0 + ra.bit, kvh * kGroup + ma_row % kGroup, o_stride, t, amax);
        if (rb.live) amax = reread_row_pipe(out, r0 + rb.bit, kvh * kGroup + mb_row % kGroup, o_stride, t, amax);
    }
    if constexpr (AMAX != 0) {
        __shared__ float s_amax[kSubWarps];
        for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
        if (lane == 0u) s_amax[warp_local] = amax;
        __syncthreads();
        if (tid == 0u) {
            float m = s_amax[0];
#pragma unroll
            for (unsigned int w = 1u; w < kSubWarps; ++w) m = fmaxf(m, s_amax[w]);
            const float seen = __uint_as_float(*reinterpret_cast<volatile unsigned int*>(amax_bits));
            if (m > 0.0f && m > seen) atomicMax(amax_bits, __float_as_uint(m));
        }
    }
}

}  // namespace

#define QSA_PREFILL_ATTN_FP8_SUB_ENTRY(NAME, SUB, THREADS_CTA)                                  \
    extern "C" __global__ void __launch_bounds__(THREADS_CTA, 2) NAME(                          \
        const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,                        \
        const uint8_t* __restrict__ vc, const int* __restrict__ bt,                             \
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,      \
        __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,               \
        unsigned int rows, unsigned int first_pos, unsigned int q_stride,                       \
        unsigned int o_stride, unsigned int gate_stride, unsigned int page_size,                \
        unsigned long long k_page_stride, unsigned int k_tok_stride,                            \
        unsigned long long v_page_stride, unsigned int v_tok_stride, float scale_log2_h0,       \
        float scale_log2_h1, float out_scale_h0, float out_scale_h1) {                          \
        qsa_prefill_attn_fp8_sub<SUB, 0>(                                                       \
            q8, kc8, vc, bt, entries, counts, out, gate, rows, first_pos, q_stride, o_stride,   \
            gate_stride, page_size, k_page_stride, k_tok_stride, v_page_stride, v_tok_stride,   \
            scale_log2_h0, scale_log2_h1, out_scale_h0, out_scale_h1);                          \
    }

#define QSA_PREFILL_ATTN_FP8_SUB_AMAX_ENTRY(NAME, SUB, THREADS_CTA)                             \
    extern "C" __global__ void __launch_bounds__(THREADS_CTA, 2) NAME(                          \
        const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,                        \
        const uint8_t* __restrict__ vc, const int* __restrict__ bt,                             \
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,      \
        __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,               \
        unsigned int rows, unsigned int first_pos, unsigned int q_stride,                       \
        unsigned int o_stride, unsigned int gate_stride, unsigned int page_size,                \
        unsigned long long k_page_stride, unsigned int k_tok_stride,                            \
        unsigned long long v_page_stride, unsigned int v_tok_stride, float scale_log2_h0,       \
        float scale_log2_h1, float out_scale_h0, float out_scale_h1,                            \
        unsigned int* __restrict__ amax_bits) {                                                 \
        qsa_prefill_attn_fp8_sub<SUB, QSA_OPROJ_AMAX_MODE>(                                     \
            q8, kc8, vc, bt, entries, counts, out, gate, rows, first_pos, q_stride, o_stride,   \
            gate_stride, page_size, k_page_stride, k_tok_stride, v_page_stride, v_tok_stride,   \
            scale_log2_h0, scale_log2_h1, out_scale_h0, out_scale_h1, amax_bits);               \
    }

QSA_PREFILL_ATTN_FP8_SUB_ENTRY(qwen4_qsa_prefill_attn_fp8_qkpv_half, 2, 128)
QSA_PREFILL_ATTN_FP8_SUB_AMAX_ENTRY(QSA_QP_AMAX_SYM(half), 2, 128)
QSA_PREFILL_ATTN_FP8_SUB_ENTRY(qwen4_qsa_prefill_attn_fp8_qkpv_qtr, 4, 64)
QSA_PREFILL_ATTN_FP8_SUB_AMAX_ENTRY(QSA_QP_AMAX_SYM(qtr), 4, 64)

// Fused QSA tile plan + descending-block counting sort (`qwen4_qsa_prefill_plan` +
// `qwen4_qsa_prefill_plan_sort_desc` in a single launch).
// Grid (tiles, 1, 1), block (32, 1, 1), dynamic shared memory (1024 + min(cap, 5120)) * 4 B.
// Builds the union of `tile_rows` (<= 10) bitmasks directly into shared memory `vals`
// (clamped to `smem_cap = min(cap, 5120u)` entries) while accumulating the 1024-bin need
// histogram, then writes `counts[tile]` and scatters the sorted entries once to global
// `entries[tile * cap ..]`.
extern "C" __global__ void __launch_bounds__(32) qwen4_qsa_prefill_plan_fused_sort_desc(
    const unsigned int* __restrict__ masks,  // [rows][mask_words]
    unsigned int* __restrict__ entries,      // [tiles][cap]
    unsigned int* __restrict__ counts,       // [tiles]
    unsigned int rows,
    unsigned int first_pos,
    unsigned int mask_words,
    unsigned int tile_rows,
    unsigned int cap,
    unsigned int tiles) {
    extern __shared__ unsigned int sm[];
    unsigned int* const bins = sm;
    unsigned int* const vals = sm + 1024;
    const unsigned int tile = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    (void)tiles;
    const unsigned int r0 = tile * tile_rows;
    if (r0 >= rows) return /*early*/;
    const unsigned int r1 = min(r0 + tile_rows, rows);
    const unsigned int row0_pos = first_pos + r0;
    const unsigned int last_pos = first_pos + r1 - 1;
    const unsigned int words = min(mask_words, ((last_pos + 1) / kRatio) / 32 + 1);
    unsigned int* const dst = entries + (size_t)tile * cap;
    const unsigned int smem_cap = min(cap, 5120u);
    const unsigned int half_cap = smem_cap >> 1;
    const unsigned int first_tail_blk = (row0_pos + 1u) / kRatio;
    const unsigned int tail_w0 = first_tail_blk >> 5;
    const unsigned int row_mask = (1u << tile_rows) - 1u;
    const unsigned int lt = (1u << lane) - 1u;

    const auto fast_need_of = [&](unsigned int e) -> unsigned int {
        const unsigned int blk = e & kBlockMask;
        unsigned int need = (e >> kBlockBits) & row_mask;
        const unsigned int s = blk * kRatio + 2u - row0_pos;
        if (s < tile_rows + 2u) {
            need |= ((7u << s) >> 2) & row_mask;
        }
        return need;
    };

    for (unsigned int b = lane; b < 1024; b += 32) bins[b] = 0;
    __syncwarp();

    unsigned int total = 0;
    for (unsigned int w0 = 0; w0 < words; w0 += 32) {
        const unsigned int w = w0 + lane;
        unsigned int rw[10];
        unsigned int uni = 0;
        if (w < words) {
#pragma unroll
            for (unsigned int i = 0; i < 10; ++i) {
                rw[i] = (r0 + i < r1) ? __ldg(masks + (size_t)(r0 + i) * mask_words + w) : 0u;
                uni |= rw[i];
            }
            if (w >= tail_w0) {
#pragma unroll
                for (unsigned int i = 0; i < 10; ++i) {
                    if (r0 + i < r1) {
                        const unsigned int p1 = row0_pos + i + 1u;
                        const unsigned int tb = p1 / kRatio;
                        if ((p1 & (kRatio - 1u)) != 0u && (tb >> 5) == w) {
                            uni |= 1u << (tb & 31u);
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (unsigned int i = 0; i < 10; ++i) rw[i] = 0u;
        }
        if (__all_sync(0xffffffffu, uni == 0u)) continue;
        const unsigned int c = __popc(uni);
        unsigned int incl = c;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const unsigned int v = __shfl_up_sync(0xffffffffu, incl, off);
            if (lane >= static_cast<unsigned int>(off)) incl += v;
        }
        unsigned int pos = total + incl - c;
        unsigned int bits = uni;
        while (bits) {
            const unsigned int j = __ffs(bits) - 1;
            bits &= bits - 1;
            unsigned int member = 0;
#pragma unroll
            for (unsigned int i = 0; i < 10; ++i) member |= ((rw[i] >> j) & 1u) << i;
            if (pos < smem_cap) {
                const unsigned int e = (member << kBlockBits) | ((w * 32 + j) & kBlockMask);
                vals[pos] = e;
                const unsigned int k = fast_need_of(e);
                const unsigned int active = __activemask();
                const unsigned int peers = __match_any_sync(active, k);
                if ((peers & lt) == 0u) {
                    atomicAdd(&bins[k], static_cast<unsigned int>(__popc(peers)));
                }
            }
            ++pos;
        }
        total += __shfl_sync(0xffffffffu, incl, 31);
    }
    const unsigned int count = min(total, smem_cap);
    if (lane == 0) counts[tile] = count;
    __syncwarp();

    unsigned int local = 0;
    for (unsigned int j = 0; j < 32; ++j) local += bins[lane * 32 + j];
    unsigned int incl = local;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const unsigned int v = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= static_cast<unsigned int>(off)) incl += v;
    }
    unsigned int run = incl - local;
    __syncwarp();
    for (unsigned int j = 0; j < 32; ++j) {
        const unsigned int c = bins[lane * 32 + j];
        bins[lane * 32 + j] = run;
        run += c;
    }
    __syncwarp();
    unsigned int* const scatter_dst = (count <= half_cap) ? (vals + half_cap) : dst;
    for (unsigned int i0 = 0; i0 < count; i0 += 32) {
        const unsigned int j = i0 + lane;
        const bool live = j < count;
        const unsigned int e = live ? vals[count - 1 - j] : 0u;
        const unsigned int key = live ? fast_need_of(e) : 0xffffffffu;
        const unsigned int peers = __match_any_sync(0xffffffffu, key);
        const unsigned int rank = __popc(peers & lt);
        unsigned int base_off = 0u;
        if (live && rank == 0u) {
            base_off = bins[key];
            bins[key] = base_off + __popc(peers);
        }
        base_off = __shfl_sync(peers, base_off, __ffs(peers) - 1);
        if (live) scatter_dst[base_off + rank] = e;
    }
    if (count <= half_cap) {
        __syncwarp();
        for (unsigned int i = lane; i < count; i += 32) {
            dst[i] = scatter_dst[i];
        }
    }
}

// 4-warp (`block=128`) variant of `qwen4_qsa_prefill_plan_fused_sort_desc` for
// wide discrete GPUs (`> 64` SMs). Same dynamic shared memory `(1024 + min(cap, 5120)) * 4 B`
// (entries clamped at `smem_cap = min(cap, 5120u)`), 4x higher warp occupancy and 4x mask-word
// load parallelism, while preserving exact bit-for-bit identical `counts` and descending-block
// `entries`.
extern "C" __global__ void __launch_bounds__(128) qwen4_qsa_prefill_plan_fused_sort_desc_b128(
    const unsigned int* __restrict__ masks,  // [rows][mask_words]
    unsigned int* __restrict__ entries,      // [tiles][cap]
    unsigned int* __restrict__ counts,       // [tiles]
    unsigned int rows,
    unsigned int first_pos,
    unsigned int mask_words,
    unsigned int tile_rows,
    unsigned int cap,
    unsigned int tiles) {
    extern __shared__ unsigned int sm[];
    __shared__ unsigned int warp_tot[4];
    unsigned int* const bins = sm;
    unsigned int* const vals = sm + 1024;
    const unsigned int tile = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    (void)tiles;
    const unsigned int r0 = tile * tile_rows;
    if (r0 >= rows) return /*early*/;
    const unsigned int r1 = min(r0 + tile_rows, rows);
    const unsigned int row0_pos = first_pos + r0;
    const unsigned int last_pos = first_pos + r1 - 1;
    const unsigned int words = min(mask_words, ((last_pos + 1) / kRatio) / 32 + 1);
    unsigned int* const dst = entries + (size_t)tile * cap;
    const unsigned int smem_cap = min(cap, 5120u);
    const unsigned int half_cap = smem_cap >> 1;
    const unsigned int first_tail_blk = (row0_pos + 1u) / kRatio;
    const unsigned int tail_w0 = first_tail_blk >> 5;
    const unsigned int row_mask = (1u << tile_rows) - 1u;
    const unsigned int lt = (1u << lane) - 1u;

    const auto fast_need_of = [&](unsigned int e) -> unsigned int {
        const unsigned int blk = e & kBlockMask;
        unsigned int need = (e >> kBlockBits) & row_mask;
        const unsigned int s = blk * kRatio + 2u - row0_pos;
        if (s < tile_rows + 2u) {
            need |= ((7u << s) >> 2) & row_mask;
        }
        return need;
    };

    for (unsigned int b = tid; b < 1024; b += 128) bins[b] = 0;
    __syncthreads();

    unsigned int total = 0;
    for (unsigned int w0 = 0; w0 < words; w0 += 128) {
        const unsigned int w = w0 + tid;
        unsigned int rw[10];
        unsigned int uni = 0;
        if (w < words) {
#pragma unroll
            for (unsigned int i = 0; i < 10; ++i) {
                rw[i] = (r0 + i < r1) ? __ldg(masks + (size_t)(r0 + i) * mask_words + w) : 0u;
                uni |= rw[i];
            }
            if (w >= tail_w0) {
#pragma unroll
                for (unsigned int i = 0; i < 10; ++i) {
                    if (r0 + i < r1) {
                        const unsigned int p1 = row0_pos + i + 1u;
                        const unsigned int tb = p1 / kRatio;
                        if ((p1 & (kRatio - 1u)) != 0u && (tb >> 5) == w) {
                            uni |= 1u << (tb & 31u);
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (unsigned int i = 0; i < 10; ++i) rw[i] = 0u;
        }
        const unsigned int c = __popc(uni);
        unsigned int incl = c;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const unsigned int v = __shfl_up_sync(0xffffffffu, incl, off);
            if (lane >= static_cast<unsigned int>(off)) incl += v;
        }
        if (lane == 31u) warp_tot[warp] = incl;
        __syncthreads();
        const unsigned int chunk_tot = warp_tot[0] + warp_tot[1] + warp_tot[2] + warp_tot[3];
        if (chunk_tot == 0u) {
            __syncthreads();
            continue;
        }
        unsigned int warp_base = 0u;
        if (warp >= 1u) warp_base += warp_tot[0];
        if (warp >= 2u) warp_base += warp_tot[1];
        if (warp >= 3u) warp_base += warp_tot[2];
        unsigned int pos = total + warp_base + incl - c;
        unsigned int bits = uni;
        while (bits) {
            const unsigned int j = __ffs(bits) - 1;
            bits &= bits - 1;
            unsigned int member = 0;
#pragma unroll
            for (unsigned int i = 0; i < 10; ++i) member |= ((rw[i] >> j) & 1u) << i;
            if (pos < smem_cap) {
                const unsigned int e = (member << kBlockBits) | ((w * 32 + j) & kBlockMask);
                vals[pos] = e;
                const unsigned int k = fast_need_of(e);
                const unsigned int active = __activemask();
                const unsigned int peers = __match_any_sync(active, k);
                if ((peers & lt) == 0u) {
                    atomicAdd(&bins[k], static_cast<unsigned int>(__popc(peers)));
                }
            }
            ++pos;
        }
        total += chunk_tot;
        __syncthreads();
    }
    const unsigned int count = min(total, smem_cap);
    if (tid == 0u) counts[tile] = count;

    unsigned int local = 0u;
#pragma unroll
    for (unsigned int j = 0; j < 8; ++j) local += bins[tid * 8 + j];
    unsigned int incl = local;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const unsigned int v = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= static_cast<unsigned int>(off)) incl += v;
    }
    if (lane == 31u) warp_tot[warp] = incl;
    __syncthreads();
    unsigned int warp_base = 0u;
    if (warp >= 1u) warp_base += warp_tot[0];
    if (warp >= 2u) warp_base += warp_tot[1];
    if (warp >= 3u) warp_base += warp_tot[2];
    unsigned int run = warp_base + incl - local;
#pragma unroll
    for (unsigned int j = 0; j < 8; ++j) {
        const unsigned int c = bins[tid * 8 + j];
        bins[tid * 8 + j] = run;
        run += c;
    }
    __syncthreads();

    unsigned int* const scatter_dst = (count <= half_cap) ? (vals + half_cap) : dst;
    for (unsigned int i0 = 0; i0 < count; i0 += 128) {
        const unsigned int j = i0 + tid;
        const bool live = j < count;
        const unsigned int e = live ? vals[count - 1 - j] : 0u;
        const unsigned int key = live ? fast_need_of(e) : 0xffffffffu;
        const unsigned int peers = __match_any_sync(0xffffffffu, key);
        const unsigned int rank = __popc(peers & lt);
        const unsigned int cnt = __popc(peers);
        unsigned int base_off = 0u;
#pragma unroll
        for (unsigned int w = 0u; w < 4u; ++w) {
            if (warp == w && live && rank == 0u) {
                base_off = bins[key];
                bins[key] = base_off + cnt;
            }
            __syncthreads();
        }
        base_off = __shfl_sync(peers, base_off, __ffs(peers) - 1);
        if (live) scatter_dst[base_off + rank] = e;
    }
    if (count <= half_cap) {
        __syncthreads();
        for (unsigned int i = tid; i < count; i += 128) {
            dst[i] = scatter_dst[i];
        }
    }
}

namespace {

__device__ __forceinline__ uint64_t qsa_l2_policy_evict_last() {
    uint64_t pol;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    return pol;
}

__device__ __forceinline__ uint64_t qsa_l2_policy_evict_first() {
    uint64_t pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
    return pol;
}

__device__ __forceinline__ void cp_async16_l2_128b_hint_sa(
    unsigned int sa, const void* gmem, uint64_t pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::128B [%0], [%1], 16, %2;"
                 :: "r"(sa), "l"(gmem), "l"(pol));
}

__device__ __forceinline__ void cp_async16_l2_128b_hint(
    void* smem, const void* gmem, uint64_t pol) {
    unsigned int sa = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    // Materialize 16B-aligned shared address via volatile mask so sm_121a ptxas never folds
    // `[R+UR0]` into `LDGSTS ... desc[UR]` (which triggers an odd `desc[UR1]` bug).
    asm volatile("and.b32 %0, %0, 0xfffffff0;" : "+r"(sa));
    cp_async16_l2_128b_hint_sa(sa, gmem, pol);
}

// Unrolled 2-step E4M3 K+V tile loader for kBc = 32, kThreads = 256:
// overlaps both block-table __ldg loads before issuing L2::evict_last cp.async.cg.L2::128B
// transfers (keeping the 16.0 MiB 16K K8+V8 working set resident in GB10's 24 MiB L2),
// and uses a CTA-uniform fast path on full 8-block (32-token) tiles.
__device__ __forceinline__ void load_tile_pipe(
    unsigned char* stage, const uint8_t* kc8, const uint8_t* vc, const unsigned int* gent,
    unsigned int base, unsigned int count, const int* bt, unsigned int page_shift,
    unsigned int page_mask, unsigned long long k_page_stride, unsigned int k_tok_stride,
    unsigned long long v_page_stride, unsigned int v_tok_stride, unsigned int kvh,
    unsigned int tid, uint64_t pol_kv) {
    unsigned char* ks = stage;
    unsigned char* vs = stage + kTile8;
    const unsigned int j0 = tid >> 4;
    const unsigned int j1 = j0 + 16u;
    const unsigned int ch = tid & 15u;
    const unsigned int in_b = j0 & 3u;
    const unsigned int slot0 = base + (j0 >> 2);
    const unsigned int slot1 = base + (j1 >> 2);
    // Materialize 16B-aligned SMEM base in a thread register (`sa_k0`) via volatile `and.b32`
    // so ptxas emits `[R+imm], desc[UR<even>]` rather than `[R+UR0], desc[UR1]` on sm_121a.
    // Note swz8(j0+16, ch) == swz8(j0, ch) + 0x1000.
    unsigned int sa_k0 =
        static_cast<unsigned int>(__cvta_generic_to_shared(ks + swz8(j0, ch)));
    asm volatile("and.b32 %0, %0, 0xfffffff0;" : "+r"(sa_k0));
    constexpr unsigned int kHalfTile8 = 16u * kHd;
    if (base + 8u <= count) {
        const unsigned int tok0 = (__ldg(gent + slot0) & kBlockMask) * kRatio + in_b;
        const unsigned int tok1 = (__ldg(gent + slot1) & kBlockMask) * kRatio + in_b;
        const unsigned int page0 = static_cast<unsigned int>(__ldg(bt + (tok0 >> page_shift)));
        const unsigned int page1 = static_cast<unsigned int>(__ldg(bt + (tok1 >> page_shift)));
        const size_t in_page0 = tok0 & page_mask;
        const size_t in_page1 = tok1 & page_mask;
        const size_t k_off0 =
            page0 * k_page_stride + in_page0 * k_tok_stride + kvh * kHd + ch * 16;
        const size_t v_off0 =
            page0 * v_page_stride + in_page0 * v_tok_stride + kvh * kHd + ch * 16;
        const size_t k_off1 =
            page1 * k_page_stride + in_page1 * k_tok_stride + kvh * kHd + ch * 16;
        const size_t v_off1 =
            page1 * v_page_stride + in_page1 * v_tok_stride + kvh * kHd + ch * 16;
        cp_async16_l2_128b_hint_sa(sa_k0, kc8 + k_off0, pol_kv);
        cp_async16_l2_128b_hint_sa(sa_k0 + kTile8, vc + v_off0, pol_kv);
        cp_async16_l2_128b_hint_sa(sa_k0 + kHalfTile8, kc8 + k_off1, pol_kv);
        cp_async16_l2_128b_hint_sa(sa_k0 + kTile8 + kHalfTile8, vc + v_off1, pol_kv);
    } else {
        const bool live0 = slot0 < count;
        const bool live1 = slot1 < count;
        const unsigned int tok0 = live0 ? (__ldg(gent + slot0) & kBlockMask) * kRatio + in_b : 0u;
        const unsigned int tok1 = live1 ? (__ldg(gent + slot1) & kBlockMask) * kRatio + in_b : 0u;
        const unsigned int page0 =
            live0 ? static_cast<unsigned int>(__ldg(bt + (tok0 >> page_shift))) : 0u;
        const unsigned int page1 =
            live1 ? static_cast<unsigned int>(__ldg(bt + (tok1 >> page_shift))) : 0u;
        if (live0) {
            const size_t in_page0 = tok0 & page_mask;
            cp_async16_l2_128b_hint_sa(
                sa_k0,
                kc8 + page0 * k_page_stride + in_page0 * k_tok_stride + kvh * kHd + ch * 16,
                pol_kv);
            cp_async16_l2_128b_hint_sa(
                sa_k0 + kTile8,
                vc + page0 * v_page_stride + in_page0 * v_tok_stride + kvh * kHd + ch * 16,
                pol_kv);
        } else {
            *reinterpret_cast<uint4*>(ks + swz8(j0, ch)) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(vs + swz8(j0, ch)) = make_uint4(0, 0, 0, 0);
        }
        if (live1) {
            const size_t in_page1 = tok1 & page_mask;
            cp_async16_l2_128b_hint_sa(
                sa_k0 + kHalfTile8,
                kc8 + page1 * k_page_stride + in_page1 * k_tok_stride + kvh * kHd + ch * 16,
                pol_kv);
            cp_async16_l2_128b_hint_sa(
                sa_k0 + kTile8 + kHalfTile8,
                vc + page1 * v_page_stride + in_page1 * v_tok_stride + kvh * kHd + ch * 16,
                pol_kv);
        } else {
            *reinterpret_cast<uint4*>(ks + swz8(j1, ch)) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(vs + swz8(j1, ch)) = make_uint4(0, 0, 0, 0);
        }
    }
}

// Intra-warp software-pipelined E4M3 QK+PV core (PV8 = true, ACC = 2, STAGES = 3):
//   1. Drops the 20,528 B full-`ent` SMEM copy in favor of per-tile warp register caching
//      (`e_lane` in lanes 0..7 broadcast via `__shfl_sync`) and upgrades K8+V8 `cp.async` to a
//      3-stage ring (`49,152 B` = 48 KiB dynamic SMEM, <= Mode::QkPv.smem() = 53,296 B) with
//      `L2::evict_last` on K8+V8 (`16.0 MiB <= 24 MiB` GB10 L2) and `L2::evict_first` on Q8.
//   2. Overlaps both block-table __ldg lookups in load_tile_pipe before cp.async.cg.L2::128B.
//   3. Software-pipelines ldsm_x4 ahead of mma_e4m3 in the QK^T loop and skips unused 16-col halves.
//   4. Issues the row-sum mma_e4m3_zc(dl, ...) and c=0 V ldsm_x4_t before the PV loop,
//      interleaving branchless O rescales with mma_e4m3 and software-pipelining
//      ldsm_x4_t(r_next) ahead of mma_e4m3.
//   5. Batches 4x streaming gate loads (ld.global.cs.v2.u32) in store_row_pipe and 4x
//      AMAX mode-3 re-read loads (ld.global.cg.v2.u32) in reread_row_pipe.
// For E4M3_OUT == false, every floating-point operation and accumulation order is bit-for-bit
// identical to qsa_prefill_attn_fp8<true, 2, 2, AMAX>.
template <int AMAX = 0, bool E4M3_OUT = false>
__device__ __forceinline__ void qsa_prefill_attn_fp8_pipe(
    const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,
    const uint8_t* __restrict__ vc, const int* __restrict__ bt,
    const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
    void* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
    unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
    unsigned int gate_stride, unsigned int page_size, unsigned long long k_page_stride,
    unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
    float scale_log2_h0, float scale_log2_h1, float out_scale_h0, float out_scale_h1,
    unsigned int* __restrict__ amax_bits = nullptr, float* __restrict__ a_scale_out = nullptr,
    float a_scale_val = 0.0f) {
    static_assert(!(E4M3_OUT && AMAX != 0), "E4M3_OUT skips amax reduction");
    using Cfg = Fp8Cfg<true>;
    constexpr int STAGES = 3;
    static_assert(STAGES * Cfg::kStageBytes >= kQ8Bytes, "Q staging must fit the tile ring");
    static_assert(STAGES * Cfg::kStageBytes <= 2 * Cfg::kStageBytes + kCap * 4,
                  "3-stage ring must fit inside Mode::QkPv dynamic SMEM budget");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned char* ring = smem_raw;

    const unsigned int tile = blockIdx.x;
    const unsigned int kvh = blockIdx.y;
    const float scale_log2 = kvh ? scale_log2_h1 : scale_log2_h0;
    const float out_scale = kvh ? out_scale_h1 : out_scale_h0;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3;
    const unsigned int r0 = tile * kTrows;
    if (r0 >= rows || page_size == 0u || (page_size & (page_size - 1u)) != 0u) return /*early*/;
    if constexpr (E4M3_OUT) {
        if (tile == 0u && kvh == 0u && tid == 0u && a_scale_out != nullptr) {
            a_scale_out[0] = a_scale_val;
        }
    }
    const unsigned int count = min(counts[tile], static_cast<unsigned int>(kCap));
    const unsigned int* gent = entries + (size_t)tile * kCap;
    const unsigned int page_shift = static_cast<unsigned int>(__ffs(page_size) - 1);
    const unsigned int page_mask = page_size - 1u;

    const uint64_t pol_q = qsa_l2_policy_evict_first();
    for (unsigned int c = tid; c < kWarps * 16 * 16; c += kThreads) {
        const unsigned int m = c >> 4;
        const unsigned int ch = c & 15;
        unsigned char* dst = ring + swz8(m, ch);
        const unsigned int rt = m / kGroup;
        if (m < static_cast<unsigned int>(kM) && r0 + rt < rows) {
            const unsigned int head = kvh * kGroup + m % kGroup;
            cp_async16_l2_128b_hint(
                dst, q8 + (size_t)(r0 + rt) * q_stride + head * kHd + ch * 16, pol_q);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    unsigned int qf[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const unsigned int row = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        ldsm_x4(qf[ks], ring + swz8(row, ks * 2 + (lane >> 4)));
    }
    __syncthreads();

    const RowInfo ra = row_info(warp * 16 + g, r0, rows, first_pos);
    const RowInfo rb = row_info(warp * 16 + g + 8, r0, rows, first_pos);
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float ma = -CUDART_INF_F, mb = -CUDART_INF_F, la = 0.0f, lb = 0.0f;

    const unsigned int ntiles = (count + 7) / 8;
    const unsigned int row0_pos = first_pos + r0;
    unsigned int warp_rows = 0;
    {
        const unsigned int m_lo = warp * 16;
        const unsigned int m_hi = min(m_lo + 15, static_cast<unsigned int>(kM) - 1);
        for (unsigned int rt = m_lo / kGroup; rt <= m_hi / kGroup; ++rt) warp_rows |= 1u << rt;
    }

    const uint64_t pol_kv = qsa_l2_policy_evict_last();
    // Keep prologue non-unrolled so `stage` is materialized in a thread register (`[R]`) rather
    // than `[R+UR0]` on `LDGSTS ... desc[UR]` (which triggers sm_121a ptxas odd `desc[UR1]`).
#pragma unroll 1
    for (unsigned int s = 0u; s < static_cast<unsigned int>(STAGES - 1); ++s) {
        if (s < ntiles) {
            load_tile_pipe(ring + s * Cfg::kStageBytes, kc8, vc, gent, s * 8u, count, bt,
                           page_shift, page_mask, k_page_stride, k_tok_stride,
                           v_page_stride, v_tok_stride, kvh, tid, pol_kv);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }

    const unsigned int k_row0 = (lane >> 4) * 8u + (lane & 7u);
    const unsigned int k_row1 = k_row0 + 16u;
    const unsigned int k_col_off = (lane >> 3) & 1u;
    const unsigned int vs_row = (lane >> 3) * 8u + (lane & 7u);

    for (unsigned int it = 0u, st_cur = 0u; it < ntiles;
         ++it, st_cur = (st_cur == static_cast<unsigned int>(STAGES - 1)) ? 0u : (st_cur + 1u)) {
        const unsigned int base = it * 8u;
        const unsigned int e_lane =
            (lane < 8u && base + lane < count) ? __ldg(gent + base + lane) : 0u;
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2) : "memory");
        __syncthreads();
        {
            const unsigned int nt = it + static_cast<unsigned int>(STAGES - 1);
            if (nt < ntiles) {
                const unsigned int st_load =
                    (st_cur == 0u) ? static_cast<unsigned int>(STAGES - 1) : (st_cur - 1u);
                load_tile_pipe(ring + st_load * Cfg::kStageBytes, kc8, vc, gent, nt * 8u, count,
                               bt, page_shift, page_mask, k_page_stride,
                               k_tok_stride, v_page_stride, v_tok_stride, kvh, tid, pol_kv);
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        const unsigned char* ks_s = ring + st_cur * Cfg::kStageBytes;
        const unsigned char* vs_s = ks_s + kTile8;
        bool do_h0 = false;
        bool do_h1 = false;
        {
            const unsigned int need =
                (lane < 8u && base + lane < count) ? entry_need_pipe(e_lane, row0_pos) : 0u;
            const unsigned int need01 =
                __reduce_or_sync(0xffffffffu, lane < 4u ? need : (need << 16));
            do_h0 = (need01 & warp_rows) != 0u;
            do_h1 = ((need01 >> 16) & warp_rows) != 0u;
            if (!do_h0 && !do_h1) continue;
        }

        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
        if (do_h0 && do_h1) {
            unsigned int b_cur[4], b_next[4];
            ldsm_x4(b_cur, ks_s + swz8(k_row0, k_col_off));
#pragma unroll
            for (int kstep = 0; kstep < 8; ++kstep) {
                ldsm_x4(b_next, ks_s + swz8(k_row1, kstep * 2 + k_col_off));
                mma_e4m3(s[0], qf[kstep], b_cur[0], b_cur[1]);
                mma_e4m3(s[1], qf[kstep], b_cur[2], b_cur[3]);
                if (kstep < 7) {
                    ldsm_x4(b_cur, ks_s + swz8(k_row0, (kstep + 1) * 2 + k_col_off));
                }
                mma_e4m3(s[2], qf[kstep], b_next[0], b_next[1]);
                mma_e4m3(s[3], qf[kstep], b_next[2], b_next[3]);
            }
        } else if (do_h0) {
            unsigned int b_cur[4], b_next[4];
            ldsm_x4(b_cur, ks_s + swz8(k_row0, k_col_off));
#pragma unroll
            for (int kstep = 0; kstep < 8; ++kstep) {
                if (kstep < 7) {
                    ldsm_x4(b_next, ks_s + swz8(k_row0, (kstep + 1) * 2 + k_col_off));
                }
                mma_e4m3(s[0], qf[kstep], b_cur[0], b_cur[1]);
                mma_e4m3(s[1], qf[kstep], b_cur[2], b_cur[3]);
                if (kstep < 7) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) b_cur[k] = b_next[k];
                }
            }
        } else {
            unsigned int b_cur[4], b_next[4];
            ldsm_x4(b_cur, ks_s + swz8(k_row1, k_col_off));
#pragma unroll
            for (int kstep = 0; kstep < 8; ++kstep) {
                if (kstep < 7) {
                    ldsm_x4(b_next, ks_s + swz8(k_row1, (kstep + 1) * 2 + k_col_off));
                }
                mma_e4m3(s[2], qf[kstep], b_cur[0], b_cur[1]);
                mma_e4m3(s[3], qf[kstep], b_cur[2], b_cur[3]);
                if (kstep < 7) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) b_cur[k] = b_next[k];
                }
            }
        }

        float mxa = -CUDART_INF_F, mxb = -CUDART_INF_F;
        if (do_h0) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const unsigned int slot_off = nt * 2u + (t >> 1);
                const bool slot_live = base + slot_off < count;
                const unsigned int e = __shfl_sync(0xffffffffu, e_lane, slot_off);
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    const unsigned int in_block = (t * 2 + x) & 3u;
                    s[nt][x] = allowed(ra, e, slot_live, in_block) ? s[nt][x] * scale_log2
                                                                   : -CUDART_INF_F;
                    s[nt][2 + x] = allowed(rb, e, slot_live, in_block) ? s[nt][2 + x] * scale_log2
                                                                       : -CUDART_INF_F;
                    mxa = fmaxf(mxa, s[nt][x]);
                    mxb = fmaxf(mxb, s[nt][2 + x]);
                }
            }
        }
        if (do_h1) {
#pragma unroll
            for (int nt = 2; nt < 4; ++nt) {
                const unsigned int slot_off = nt * 2u + (t >> 1);
                const bool slot_live = base + slot_off < count;
                const unsigned int e = __shfl_sync(0xffffffffu, e_lane, slot_off);
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    const unsigned int in_block = (t * 2 + x) & 3u;
                    s[nt][x] = allowed(ra, e, slot_live, in_block) ? s[nt][x] * scale_log2
                                                                   : -CUDART_INF_F;
                    s[nt][2 + x] = allowed(rb, e, slot_live, in_block) ? s[nt][2 + x] * scale_log2
                                                                       : -CUDART_INF_F;
                    mxa = fmaxf(mxa, s[nt][x]);
                    mxb = fmaxf(mxb, s[nt][2 + x]);
                }
            }
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float na = fmaxf(ma, mxa);
        const float nb = fmaxf(mb, mxb);
        const bool rescale_a = (na > ma) && (ma != -CUDART_INF_F);
        const bool rescale_b = (nb > mb) && (mb != -CUDART_INF_F);
        const bool any_rescale_a = __any_sync(0xffffffffu, rescale_a);
        const bool any_rescale_b = __any_sync(0xffffffffu, rescale_b);
        float alpha_a = 1.0f;
        if (any_rescale_a) {
            alpha_a = rescale_a ? exp2f(ma - na) : 1.0f;
        }
        float alpha_b = 1.0f;
        if (any_rescale_b) {
            alpha_b = rescale_b ? exp2f(mb - nb) : 1.0f;
        }
        const float p_log2 = 8.0f;
        const float base_a = (na == -CUDART_INF_F ? 0.0f : na) - p_log2;
        const float base_b = (nb == -CUDART_INF_F ? 0.0f : nb) - p_log2;
        ma = na;
        mb = nb;
        unsigned int a[4] = {0u, 0u, 0u, 0u};
        if (do_h0) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                s[nt][0] = exp2f(s[nt][0] - base_a);
                s[nt][1] = exp2f(s[nt][1] - base_a);
                s[nt][2] = exp2f(s[nt][2] - base_b);
                s[nt][3] = exp2f(s[nt][3] - base_b);
            }
            a[0] = e4m3x4(s[0][0], s[0][1], s[1][0], s[1][1]);
            a[1] = e4m3x4(s[0][2], s[0][3], s[1][2], s[1][3]);
        }
        if (do_h1) {
#pragma unroll
            for (int nt = 2; nt < 4; ++nt) {
                s[nt][0] = exp2f(s[nt][0] - base_a);
                s[nt][1] = exp2f(s[nt][1] - base_a);
                s[nt][2] = exp2f(s[nt][2] - base_b);
                s[nt][3] = exp2f(s[nt][3] - base_b);
            }
            a[2] = e4m3x4(s[2][0], s[2][1], s[3][0], s[3][1]);
            a[3] = e4m3x4(s[2][2], s[2][3], s[3][2], s[3][3]);
        }
        float dl[4];
        mma_e4m3_zc(dl, a, kE4m3OnesX4, kE4m3OnesX4);

        unsigned int r_cur[4], r_next[4];
        ldsm_x4_t(r_cur, vs_s + swz8(vs_row, 0));

        // Interleave branchless FP32 accumulator rescales directly with the PV QMMAs
        // so FMULs execute on the FMA pipe concurrently with Tensor Core MMAs (`alpha == 1.0f`
        // is exact IEEE-754 0-ULP under target FTZ), eliminating the 128-FMUL serial stall and
        // branch/I-cache split before PV.
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            if (c < 15) {
                ldsm_x4_t(r_next, vs_s + swz8(vs_row, c + 1));
            }
            o[2 * c][0] *= alpha_a;
            o[2 * c][1] *= alpha_a;
            o[2 * c][2] *= alpha_b;
            o[2 * c][3] *= alpha_b;
            mma_e4m3(o[2 * c], a, prmt(r_cur[0], r_cur[1], 0x6420u),
                     prmt(r_cur[2], r_cur[3], 0x6420u));
            o[2 * c + 1][0] *= alpha_a;
            o[2 * c + 1][1] *= alpha_a;
            o[2 * c + 1][2] *= alpha_b;
            o[2 * c + 1][3] *= alpha_b;
            mma_e4m3(o[2 * c + 1], a, prmt(r_cur[0], r_cur[1], 0x7531u),
                     prmt(r_cur[2], r_cur[3], 0x7531u));
            if (c == 1) {
                la = fmaf(la, alpha_a, dl[0]);
                lb = fmaf(lb, alpha_b, dl[2]);
            }
            if (c < 15) {
#pragma unroll
                for (int k = 0; k < 4; ++k) r_cur[k] = r_next[k];
            }
        }
    }

    const float ia = la > 0.0f ? out_scale / la : 0.0f;
    const float ib = lb > 0.0f ? out_scale / lb : 0.0f;
    const unsigned int ma_row = warp * 16 + g;
    const unsigned int mb_row = ma_row + 8;
    if constexpr (E4M3_OUT) {
        uint8_t* const out8 = static_cast<uint8_t*>(out);
        if (ra.live) {
            store_row_pipe_e4m3(out8, gate, r0 + ra.bit, kvh * kGroup + ma_row % kGroup, o_stride,
                                gate_stride, t, ia, o, 0);
        }
        if (rb.live) {
            store_row_pipe_e4m3(out8, gate, r0 + rb.bit, kvh * kGroup + mb_row % kGroup, o_stride,
                                gate_stride, t, ib, o, 2);
        }
    } else {
        __nv_bfloat16* const out16 = static_cast<__nv_bfloat16*>(out);
        float amax = 0.0f;
        if (ra.live) {
            store_row_pipe<AMAX>(out16, gate, r0 + ra.bit, kvh * kGroup + ma_row % kGroup,
                                 o_stride, gate_stride, t, ia, o, 0, amax);
        }
        if (rb.live) {
            store_row_pipe<AMAX>(out16, gate, r0 + rb.bit, kvh * kGroup + mb_row % kGroup,
                                 o_stride, gate_stride, t, ib, o, 2, amax);
        }
        if constexpr (AMAX == 2) amax = __bfloat162float(__float2bfloat16(amax));
        if constexpr (AMAX == 3) {
            if (ra.live) {
                amax = reread_row_pipe(out16, r0 + ra.bit, kvh * kGroup + ma_row % kGroup,
                                       o_stride, t, amax);
            }
            if (rb.live) {
                amax = reread_row_pipe(out16, r0 + rb.bit, kvh * kGroup + mb_row % kGroup,
                                       o_stride, t, amax);
            }
        }
        if constexpr (AMAX != 0) {
            __shared__ float s_amax[kWarps];
            for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
            if (lane == 0) s_amax[warp] = amax;
            __syncthreads();
            if (tid == 0) {
                float m = s_amax[0];
#pragma unroll
                for (int w = 1; w < kWarps; ++w) m = fmaxf(m, s_amax[w]);
                const float seen =
                    __uint_as_float(*reinterpret_cast<volatile unsigned int*>(amax_bits));
                if (m > 0.0f && m > seen) atomicMax(amax_bits, __float_as_uint(m));
            }
        }
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads, 1) qwen4_qsa_prefill_attn_fp8_qkpv_pipe(
    const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,
    const uint8_t* __restrict__ vc, const int* __restrict__ bt,
    const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
    unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
    unsigned int gate_stride, unsigned int page_size, unsigned long long k_page_stride,
    unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
    float scale_log2_h0, float scale_log2_h1, float out_scale_h0, float out_scale_h1) {
    qsa_prefill_attn_fp8_pipe<0>(q8, kc8, vc, bt, entries, counts, out, gate, rows, first_pos,
                                 q_stride, o_stride, gate_stride, page_size, k_page_stride,
                                 k_tok_stride, v_page_stride, v_tok_stride, scale_log2_h0,
                                 scale_log2_h1, out_scale_h0, out_scale_h1);
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
    QSA_QP_AMAX_SYM(pipe)(
        const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,
        const uint8_t* __restrict__ vc, const int* __restrict__ bt,
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
        __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
        unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
        unsigned int gate_stride, unsigned int page_size, unsigned long long k_page_stride,
        unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
        float scale_log2_h0, float scale_log2_h1, float out_scale_h0, float out_scale_h1,
        unsigned int* __restrict__ amax_bits) {
    qsa_prefill_attn_fp8_pipe<QSA_OPROJ_AMAX_MODE>(
        q8, kc8, vc, bt, entries, counts, out, gate, rows, first_pos, q_stride, o_stride,
        gate_stride, page_size, k_page_stride, k_tok_stride, v_page_stride, v_tok_stride,
        scale_log2_h0, scale_log2_h1, out_scale_h0, out_scale_h1, amax_bits);
}

// Direct static-prescaled E4M3 output variant of qwen4_qsa_prefill_attn_fp8_qkpv_pipe
// (ATLAS_QWEN4_PREFILL_QSA_E4M3_OUT): writes static-prescaled E4M3 gated output rows straight
// into `out_e4m3` ([rows, 6144], stride `o_stride` bytes) and writes the cuBLASLt tensor scale
// `a_scale_val` to `a_scale_out[0]`, skipping BF16 raw_outputs writes and amax reduction.
extern "C" __global__ void __launch_bounds__(kThreads, 1)
    qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_pipe(
        const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,
        const uint8_t* __restrict__ vc, const int* __restrict__ bt,
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
        uint8_t* __restrict__ out_e4m3, const __nv_bfloat16* __restrict__ gate,
        unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
        unsigned int gate_stride, unsigned int page_size, unsigned long long k_page_stride,
        unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
        float scale_log2_h0, float scale_log2_h1, float out_scale_h0, float out_scale_h1,
        float* __restrict__ a_scale_out, float a_scale_val) {
    qsa_prefill_attn_fp8_pipe<0, true>(
        q8, kc8, vc, bt, entries, counts, out_e4m3, gate, rows, first_pos, q_stride, o_stride,
        gate_stride, page_size, k_page_stride, k_tok_stride, v_page_stride, v_tok_stride,
        scale_log2_h0, scale_log2_h1, out_scale_h0, out_scale_h1, nullptr, a_scale_out,
        a_scale_val);
}

namespace {

// Half-D direct static-prescaled E4M3 epilogue for `qkpv_e4m3_pipe_2cta`:
// writes 128 output columns `[c_base * 16 .. (c_base + 8) * 16)` using half-D accumulator `o[16][4]`.
__device__ __forceinline__ void store_row_pipe_e4m3_half(
    uint8_t* __restrict__ out, const __nv_bfloat16* __restrict__ gate, unsigned int row,
    unsigned int head, unsigned int o_stride, unsigned int gate_stride, unsigned int t,
    float inv_l, const float (&o)[16][4], int half, unsigned int c_base) {
    uint8_t* dst = out + (size_t)row * o_stride + head * kHd;
    const __nv_bfloat16* gsrc = gate ? gate + (size_t)row * gate_stride + head * kHd : nullptr;
    const auto gated = [](float x, __nv_bfloat16 gv) {
        return x * (1.0f / (1.0f + expf(-__bfloat162float(gv))));
    };
    if (gsrc) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int c0 = b * 4;
            const unsigned int d0 = (c_base + c0 + 0u) * 16u + t * 4u;
            const unsigned int d1 = (c_base + c0 + 1u) * 16u + t * 4u;
            const unsigned int d2 = (c_base + c0 + 2u) * 16u + t * 4u;
            const unsigned int d3 = (c_base + c0 + 3u) * 16u + t * 4u;
            uint2 gw[4];
            asm volatile(
                "{\n\t"
                "ld.global.cs.v2.u32 {%0, %1}, [%8];\n\t"
                "ld.global.cs.v2.u32 {%2, %3}, [%9];\n\t"
                "ld.global.cs.v2.u32 {%4, %5}, [%10];\n\t"
                "ld.global.cs.v2.u32 {%6, %7}, [%11];\n\t"
                "}\n"
                : "=r"(gw[0].x), "=r"(gw[0].y), "=r"(gw[1].x), "=r"(gw[1].y),
                  "=r"(gw[2].x), "=r"(gw[2].y), "=r"(gw[3].x), "=r"(gw[3].y)
                : "l"(gsrc + d0), "l"(gsrc + d1), "l"(gsrc + d2), "l"(gsrc + d3));
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int c = c0 + k;
                const unsigned int dc = (c_base + c) * 16u + t * 4u;
                const __nv_bfloat162 g01 = *reinterpret_cast<const __nv_bfloat162*>(&gw[k].x);
                const __nv_bfloat162 g23 = *reinterpret_cast<const __nv_bfloat162*>(&gw[k].y);
                const float x0 = gated(o[2 * c][half] * inv_l, g01.x);
                const float x1 = gated(o[2 * c + 1][half] * inv_l, g01.y);
                const float x2 = gated(o[2 * c][half + 1] * inv_l, g23.x);
                const float x3 = gated(o[2 * c + 1][half + 1] * inv_l, g23.y);
                *reinterpret_cast<unsigned int*>(dst + dc) = pack_e4m3x4_pipe(x0, x1, x2, x3);
            }
        }
    } else {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const unsigned int d0 = (c_base + c) * 16u + t * 4u;
            const float x0 = o[2 * c][half] * inv_l;
            const float x1 = o[2 * c + 1][half] * inv_l;
            const float x2 = o[2 * c][half + 1] * inv_l;
            const float x3 = o[2 * c + 1][half + 1] * inv_l;
            *reinterpret_cast<unsigned int*>(dst + d0) = pack_e4m3x4_pipe(x0, x1, x2, x3);
        }
    }
}

// 256-thread overlapped 2-step E4M3 K+V tile loader + 8-entry stage cache for `qkpv_e4m3_pipe_2cta`.
__device__ __forceinline__ void load_tile_pipe_2cta(
    unsigned char* stage, unsigned int* ent_dst, const uint8_t* kc8, const uint8_t* vc,
    const unsigned int* gent, unsigned int base, unsigned int count, const int* bt,
    unsigned int page_shift, unsigned int page_mask, unsigned long long k_page_stride,
    unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
    unsigned int kvh, unsigned int tid) {
    if (tid < 8u) {
        ent_dst[tid] = (base + tid < count) ? __ldg(gent + base + tid) : 0u;
    }
    unsigned char* const ks = stage;
    unsigned char* const vs = stage + kTile8;
    const unsigned int j0 = tid >> 4;
    const unsigned int j1 = j0 + 16u;
    const unsigned int ch = tid & 15u;
    const unsigned int in_b = j0 & 3u;
    const unsigned int slot0 = base + (j0 >> 2);
    const unsigned int slot1 = base + (j1 >> 2);
    if (base + 8u <= count) {
        const unsigned int tok0 = (__ldg(gent + slot0) & kBlockMask) * kRatio + in_b;
        const unsigned int tok1 = (__ldg(gent + slot1) & kBlockMask) * kRatio + in_b;
        const unsigned int page0 = static_cast<unsigned int>(__ldg(bt + (tok0 >> page_shift)));
        const unsigned int page1 = static_cast<unsigned int>(__ldg(bt + (tok1 >> page_shift)));
        const size_t in_page0 = tok0 & page_mask;
        const size_t in_page1 = tok1 & page_mask;
        const size_t k_off0 =
            page0 * k_page_stride + in_page0 * k_tok_stride + kvh * kHd + ch * 16u;
        const size_t v_off0 =
            page0 * v_page_stride + in_page0 * v_tok_stride + kvh * kHd + ch * 16u;
        const size_t k_off1 =
            page1 * k_page_stride + in_page1 * k_tok_stride + kvh * kHd + ch * 16u;
        const size_t v_off1 =
            page1 * v_page_stride + in_page1 * v_tok_stride + kvh * kHd + ch * 16u;
        cp_async16_l2_128b(ks + swz8(j0, ch), kc8 + k_off0);
        cp_async16_l2_128b(vs + swz8(j0, ch), vc + v_off0);
        cp_async16_l2_128b(ks + swz8(j1, ch), kc8 + k_off1);
        cp_async16_l2_128b(vs + swz8(j1, ch), vc + v_off1);
    } else {
        const bool live0 = slot0 < count;
        const bool live1 = slot1 < count;
        const unsigned int tok0 = live0 ? (__ldg(gent + slot0) & kBlockMask) * kRatio + in_b : 0u;
        const unsigned int tok1 = live1 ? (__ldg(gent + slot1) & kBlockMask) * kRatio + in_b : 0u;
        const unsigned int page0 =
            live0 ? static_cast<unsigned int>(__ldg(bt + (tok0 >> page_shift))) : 0u;
        const unsigned int page1 =
            live1 ? static_cast<unsigned int>(__ldg(bt + (tok1 >> page_shift))) : 0u;
        if (live0) {
            const size_t in_page0 = tok0 & page_mask;
            cp_async16_l2_128b(
                ks + swz8(j0, ch),
                kc8 + page0 * k_page_stride + in_page0 * k_tok_stride + kvh * kHd + ch * 16u);
            cp_async16_l2_128b(
                vs + swz8(j0, ch),
                vc + page0 * v_page_stride + in_page0 * v_tok_stride + kvh * kHd + ch * 16u);
        } else {
            *reinterpret_cast<uint4*>(ks + swz8(j0, ch)) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(vs + swz8(j0, ch)) = make_uint4(0, 0, 0, 0);
        }
        if (live1) {
            const size_t in_page1 = tok1 & page_mask;
            cp_async16_l2_128b(
                ks + swz8(j1, ch),
                kc8 + page1 * k_page_stride + in_page1 * k_tok_stride + kvh * kHd + ch * 16u);
            cp_async16_l2_128b(
                vs + swz8(j1, ch),
                vc + page1 * v_page_stride + in_page1 * v_tok_stride + kvh * kHd + ch * 16u);
        } else {
            *reinterpret_cast<uint4*>(ks + swz8(j1, ch)) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(vs + swz8(j1, ch)) = make_uint4(0, 0, 0, 0);
        }
    }
}

// Per-warp-pair 64-thread named barrier with compile-time immediate barrier IDs 1..4 so ptxas
// reports `used 5 barriers` instead of 16 (on sm_121a the SM barrier pool is 24/SM: 24/5 = 4 CTAs/SM
// vs 24/16 = 1 CTA/SM, preserving full 2-CTA/SM occupancy).
__device__ __forceinline__ void warp_pair_barrier_64(unsigned int w_pair) {
    if (w_pair == 0u) {
        asm volatile("bar.sync 1, 64;" ::: "memory");
    } else if (w_pair == 1u) {
        asm volatile("bar.sync 2, 64;" ::: "memory");
    } else if (w_pair == 2u) {
        asm volatile("bar.sync 3, 64;" ::: "memory");
    } else {
        asm volatile("bar.sync 4, 64;" ::: "memory");
    }
}

// 2-CTA/SM intra-CTA D-split + Half-CTA E4M3 QK+PV core (ATLAS_QWEN4_PREFILL_QSA_E4M3_2CTA):
//   - Grid: (tiles, KV_HEADS * 2, 1), Block: (256, 1, 1) = 8 warps = 4 warp-pairs per Half-CTA.
//   - Each Half-CTA covers 64 query rows (4 original warps) and skips ~35% of KV tiles via a
//     lockstep warp-ballot active-tile iterator (`s_ent[2][8]` = 64 B instead of 20,528 B `ent`).
//   - Within each warp-pair `(2*w_pair, 2*w_pair + 1)` sharing 16 query rows:
//       * Even warp (`p_role = 0`): computes QK^T + exp2 for K cols 0..15, accumulates PV cols 0..127.
//       * Odd warp  (`p_role = 1`): computes QK^T + exp2 for K cols 16..31, accumulates PV cols 128..255.
//     Warp-pairs exchange row max (`s_mx`, 512 B) and E4M3 P (`s_a`, 2,048 B) via per-pair 64-thread
//     named barriers (`bar.sync w_pair + 1, 64`) with ZERO QK^T duplication and ZERO exp2 duplication.
//   - Dynamic SMEM = 32,768 B (2-stage K8+V8) + 8,192 B (Q_hi cols 128..255) + 512 B (s_mx) +
//     2,048 B (s_a) + 64 B (s_ent) = 43,584 B <= 49,152 B; registers <= 128 (o[16][4] = 64 regs,
//     qf_lo[4][4] = 16 regs, s[2][4] = 8 regs) -> 2 CTAs/SM = 16 warps/SM on GB10 sm_121a.
//   - 100% bit-for-bit 0-ULP identical to `qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_pipe`.
__device__ __forceinline__ void qsa_prefill_attn_fp8_e4m3_pipe_2cta_body(
    const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,
    const uint8_t* __restrict__ vc, const int* __restrict__ bt,
    const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
    uint8_t* __restrict__ out_e4m3, const __nv_bfloat16* __restrict__ gate,
    unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
    unsigned int gate_stride, unsigned int page_size, unsigned long long k_page_stride,
    unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
    float scale_log2_h0, float scale_log2_h1, float out_scale_h0, float out_scale_h1,
    float* __restrict__ a_scale_out, float a_scale_val) {
    using Cfg = Fp8Cfg<true>;
    constexpr unsigned int kRingBytes = 2u * Cfg::kStageBytes;  // 32,768 B
    constexpr unsigned int kQHiBytes = 64u * 128u;              // 8,192 B
    constexpr unsigned int kSMxBytes = 4u * 2u * 8u * 8u;       // 512 B (float2[4][2][8])
    constexpr unsigned int kSABytes = 4u * 2u * 32u * 8u;       // 2,048 B (uint2[4][2][32])

    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned char* const ring = smem_raw;
    unsigned char* const q_hi = smem_raw + kRingBytes;
    float2* const s_mx = reinterpret_cast<float2*>(q_hi + kQHiBytes);
    uint2* const s_a = reinterpret_cast<uint2*>(q_hi + kQHiBytes + kSMxBytes);
    unsigned int* const s_ent =
        reinterpret_cast<unsigned int*>(q_hi + kQHiBytes + kSMxBytes + kSABytes);

    const unsigned int tile = blockIdx.x;
    const unsigned int sub = blockIdx.y & 1u;
    const unsigned int kvh = blockIdx.y >> 1;
    const float scale_log2 = kvh ? scale_log2_h1 : scale_log2_h0;
    const float out_scale = kvh ? out_scale_h1 : out_scale_h0;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_local = tid >> 5;
    const unsigned int w_pair = warp_local >> 1;
    const unsigned int p_role = warp_local & 1u;
    const unsigned int warp = sub * 4u + w_pair;
    const unsigned int lane = tid & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const unsigned int r0 = tile * kTrows;
    if (r0 >= rows) return /*early*/;
    if (tile == 0u && blockIdx.y == 0u && tid == 0u && a_scale_out != nullptr) {
        a_scale_out[0] = a_scale_val;
    }

    const unsigned int count = min(counts[tile], static_cast<unsigned int>(kCap));
    const unsigned int* const gent = entries + (size_t)tile * kCap;
    const unsigned int page_shift = static_cast<unsigned int>(__ffs(page_size) - 1);
    const unsigned int page_mask = page_size - 1u;

    // Stage Half-CTA's 64 E4M3 Q rows: cols 0..127 into `ring` (ephemeral), cols 128..255 into `q_hi` (persistent).
    for (unsigned int c = tid; c < 64u * 16u; c += kThreads) {
        const unsigned int m_local = c >> 4;
        const unsigned int m = sub * 64u + m_local;
        const unsigned int ch = c & 15u;
        unsigned char* const dst =
            (ch < 8u) ? (ring + swz8_half(m_local, ch)) : (q_hi + swz8_half(m_local, ch - 8u));
        const unsigned int rt = m / kGroup;
        if (m < static_cast<unsigned int>(kM) && r0 + rt < rows) {
            const unsigned int head = kvh * kGroup + m % kGroup;
            cp_async16_l2_128b(dst, q8 + (size_t)(r0 + rt) * q_stride + head * kHd + ch * 16u);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    const unsigned int q_row_local = w_pair * 16u + (lane & 7u) + ((lane >> 3) & 1u) * 8u;
    unsigned int qf_lo[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        ldsm_x4(qf_lo[ks], ring + swz8_half(q_row_local, ks * 2u + (lane >> 4)));
    }
    __syncthreads();

    unsigned int cta_rows = 0u;
    {
        const unsigned int m_lo = sub * 64u;
        const unsigned int m_hi = min(m_lo + 63u, static_cast<unsigned int>(kM) - 1u);
        for (unsigned int rt = m_lo / kGroup; rt <= m_hi / kGroup; ++rt) cta_rows |= 1u << rt;
    }
    unsigned int warp_rows = 0u;
    {
        const unsigned int m_lo = warp * 16u;
        const unsigned int m_hi = min(m_lo + 15u, static_cast<unsigned int>(kM) - 1u);
        for (unsigned int rt = m_lo / kGroup; rt <= m_hi / kGroup; ++rt) warp_rows |= 1u << rt;
    }

    const unsigned int ntiles = (count + 7u) / 8u;
    const unsigned int row0_pos = first_pos + r0;

    unsigned int scan_it0 = 0u;
    unsigned int scan_mask = 0u;
    const auto pop_next_active_it = [&]() -> unsigned int {
        while (scan_mask == 0u) {
            if (scan_it0 >= ntiles) return 0xffffffffu;
            const unsigned int it = scan_it0 + lane;
            scan_it0 += 32u;
            unsigned int need = 0u;
            if (it < ntiles) {
                const unsigned int base = it * 8u;
                if (base + 8u <= count) {
                    const uint4 v0 = __ldg(reinterpret_cast<const uint4*>(gent + base));
                    const uint4 v1 = __ldg(reinterpret_cast<const uint4*>(gent + base + 4u));
                    need = entry_need_pipe(v0.x, row0_pos) | entry_need_pipe(v0.y, row0_pos) |
                           entry_need_pipe(v0.z, row0_pos) | entry_need_pipe(v0.w, row0_pos) |
                           entry_need_pipe(v1.x, row0_pos) | entry_need_pipe(v1.y, row0_pos) |
                           entry_need_pipe(v1.z, row0_pos) | entry_need_pipe(v1.w, row0_pos);
                } else {
#pragma unroll
                    for (unsigned int j = 0u; j < 8u; ++j) {
                        if (base + j < count) {
                            need |= entry_need_pipe(__ldg(gent + base + j), row0_pos);
                        }
                    }
                }
            }
            scan_mask = __ballot_sync(0xffffffffu, (need & cta_rows) != 0u);
        }
        const unsigned int bit = __ffs(scan_mask) - 1u;
        scan_mask &= scan_mask - 1u;
        return (scan_it0 - 32u) + bit;
    };

    const RowInfo ra = row_info(warp * 16u + g, r0, rows, first_pos);
    const RowInfo rb = row_info(warp * 16u + g + 8u, r0, rows, first_pos);
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float ma = -CUDART_INF_F, mb = -CUDART_INF_F, la = 0.0f, lb = 0.0f;

    const unsigned int my_k_row = (lane >> 4) * 8u + (lane & 7u) + p_role * 16u;
    const unsigned int k_col_off = (lane >> 3) & 1u;
    const unsigned int vs_row = (lane >> 3) * 8u + (lane & 7u);
    const unsigned int c_base = p_role * 8u;

    unsigned int cur_it = pop_next_active_it();
    if (cur_it != 0xffffffffu) {
        load_tile_pipe_2cta(ring, s_ent, kc8, vc, gent, cur_it * 8u, count, bt, page_shift,
                            page_mask, k_page_stride, k_tok_stride, v_page_stride, v_tok_stride,
                            kvh, tid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");

    for (unsigned int stage = 0u; cur_it != 0xffffffffu; stage ^= 1u) {
        const unsigned int next_it = pop_next_active_it();
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (next_it != 0xffffffffu) {
            load_tile_pipe_2cta(ring + (stage ^ 1u) * Cfg::kStageBytes,
                                s_ent + (stage ^ 1u) * 8u, kc8, vc, gent, next_it * 8u, count, bt,
                                page_shift, page_mask, k_page_stride, k_tok_stride, v_page_stride,
                                v_tok_stride, kvh, tid);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");

        const unsigned int base = cur_it * 8u;
        const unsigned char* const ks_s = ring + stage * Cfg::kStageBytes;
        const unsigned char* const vs_s = ks_s + kTile8;
        const unsigned int* const ent_s = s_ent + stage * 8u;
        cur_it = next_it;

        bool do_h0 = false;
        bool do_h1 = false;
        {
            const unsigned int j = lane & 7u;
            const unsigned int need =
                (lane < 8u && base + j < count) ? entry_need_pipe(ent_s[j], row0_pos) : 0u;
            const unsigned int need01 =
                __reduce_or_sync(0xffffffffu, lane < 4u ? need : (need << 16));
            do_h0 = (need01 & warp_rows) != 0u;
            do_h1 = ((need01 >> 16) & warp_rows) != 0u;
            if (!do_h0 && !do_h1) continue;
        }

        const bool my_do = p_role ? do_h1 : do_h0;
        float s[2][4];
        float mxa = -CUDART_INF_F, mxb = -CUDART_INF_F;
        if (my_do) {
            s[0][0] = s[0][1] = s[0][2] = s[0][3] = 0.0f;
            s[1][0] = s[1][1] = s[1][2] = s[1][3] = 0.0f;
            unsigned int b_cur[4], b_next[4];
            ldsm_x4(b_cur, ks_s + swz8(my_k_row, k_col_off));
#pragma unroll
            for (int kstep = 0; kstep < 4; ++kstep) {
                ldsm_x4(b_next, ks_s + swz8(my_k_row, (kstep + 1) * 2 + k_col_off));
                mma_e4m3(s[0], qf_lo[kstep], b_cur[0], b_cur[1]);
                mma_e4m3(s[1], qf_lo[kstep], b_cur[2], b_cur[3]);
#pragma unroll
                for (int k = 0; k < 4; ++k) b_cur[k] = b_next[k];
            }
#pragma unroll
            for (int kstep = 0; kstep < 4; ++kstep) {
                unsigned int qf_hi[4];
                ldsm_x4(qf_hi, q_hi + swz8_half(q_row_local, kstep * 2u + (lane >> 4)));
                if (kstep < 3) {
                    ldsm_x4(b_next, ks_s + swz8(my_k_row, (kstep + 5) * 2 + k_col_off));
                }
                mma_e4m3(s[0], qf_hi, b_cur[0], b_cur[1]);
                mma_e4m3(s[1], qf_hi, b_cur[2], b_cur[3]);
                if (kstep < 3) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) b_cur[k] = b_next[k];
                }
            }

            const unsigned int nt_base = p_role * 2u;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const unsigned int slot_off = (nt_base + j) * 2u + (t >> 1);
                const bool slot_live = base + slot_off < count;
                const unsigned int e = slot_live ? ent_s[slot_off] : 0u;
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    const unsigned int in_block = (t * 2u + x) & 3u;
                    s[j][x] = allowed(ra, e, slot_live, in_block) ? s[j][x] * scale_log2
                                                                  : -CUDART_INF_F;
                    s[j][2 + x] = allowed(rb, e, slot_live, in_block) ? s[j][2 + x] * scale_log2
                                                                      : -CUDART_INF_F;
                    mxa = fmaxf(mxa, s[j][x]);
                    mxb = fmaxf(mxb, s[j][2 + x]);
                }
            }
            mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
            mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        }

        if (t == 0u) {
            s_mx[(w_pair * 2u + p_role) * 8u + g] = make_float2(mxa, mxb);
        }
        warp_pair_barrier_64(w_pair);
        const float2 peer_mx = s_mx[(w_pair * 2u + (p_role ^ 1u)) * 8u + g];
        mxa = fmaxf(mxa, peer_mx.x);
        mxb = fmaxf(mxb, peer_mx.y);

        const float na = fmaxf(ma, mxa);
        const float nb = fmaxf(mb, mxb);
        const bool rescale_a = (na > ma) && (ma != -CUDART_INF_F);
        const bool rescale_b = (nb > mb) && (mb != -CUDART_INF_F);
        const bool any_rescale_a = __any_sync(0xffffffffu, rescale_a);
        const bool any_rescale_b = __any_sync(0xffffffffu, rescale_b);
        float alpha_a = 1.0f;
        if (any_rescale_a) {
            alpha_a = rescale_a ? exp2f(ma - na) : 1.0f;
        }
        float alpha_b = 1.0f;
        if (any_rescale_b) {
            alpha_b = rescale_b ? exp2f(mb - nb) : 1.0f;
        }
        const float p_log2 = 8.0f;
        const float base_a = (na == -CUDART_INF_F ? 0.0f : na) - p_log2;
        const float base_b = (nb == -CUDART_INF_F ? 0.0f : nb) - p_log2;
        ma = na;
        mb = nb;

        uint2 my_a = make_uint2(0u, 0u);
        if (my_do) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                s[j][0] = exp2f(s[j][0] - base_a);
                s[j][1] = exp2f(s[j][1] - base_a);
                s[j][2] = exp2f(s[j][2] - base_b);
                s[j][3] = exp2f(s[j][3] - base_b);
            }
            my_a.x = e4m3x4(s[0][0], s[0][1], s[1][0], s[1][1]);
            my_a.y = e4m3x4(s[0][2], s[0][3], s[1][2], s[1][3]);
        }
        s_a[(w_pair * 2u + p_role) * 32u + lane] = my_a;
        warp_pair_barrier_64(w_pair);
        const uint2 peer_a = s_a[(w_pair * 2u + (p_role ^ 1u)) * 32u + lane];
        const unsigned int a[4] = {
            p_role ? peer_a.x : my_a.x,
            p_role ? peer_a.y : my_a.y,
            p_role ? my_a.x : peer_a.x,
            p_role ? my_a.y : peer_a.y,
        };

        float dl[4];
        mma_e4m3_zc(dl, a, kE4m3OnesX4, kE4m3OnesX4);

        unsigned int r_cur[4], r_next[4];
        ldsm_x4_t(r_cur, vs_s + swz8(vs_row, c_base));

        // Guard the 64 FP32 accumulator rescales on warp-uniform `any_rescale_{a,b}`:
        // descending-block order makes `alpha == 1.0f` on ~90%+ of tiles (0-ULP exact skip).
        if (any_rescale_a) {
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                o[c][0] *= alpha_a;
                o[c][1] *= alpha_a;
            }
        }
        if (any_rescale_b) {
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                o[c][2] *= alpha_b;
                o[c][3] *= alpha_b;
            }
        }

#pragma unroll
        for (int c = 0; c < 8; ++c) {
            if (c < 7) {
                ldsm_x4_t(r_next, vs_s + swz8(vs_row, c_base + c + 1u));
            }
            mma_e4m3(o[2 * c], a, prmt(r_cur[0], r_cur[1], 0x6420u),
                     prmt(r_cur[2], r_cur[3], 0x6420u));
            mma_e4m3(o[2 * c + 1], a, prmt(r_cur[0], r_cur[1], 0x7531u),
                     prmt(r_cur[2], r_cur[3], 0x7531u));
            if (c == 1) {
                la = fmaf(la, alpha_a, dl[0]);
                lb = fmaf(lb, alpha_b, dl[2]);
            }
            if (c < 7) {
#pragma unroll
                for (int k = 0; k < 4; ++k) r_cur[k] = r_next[k];
            }
        }
    }

    const float ia = la > 0.0f ? out_scale / la : 0.0f;
    const float ib = lb > 0.0f ? out_scale / lb : 0.0f;
    const unsigned int ma_row = warp * 16u + g;
    const unsigned int mb_row = ma_row + 8u;
    if (ra.live) {
        store_row_pipe_e4m3_half(out_e4m3, gate, r0 + ra.bit, kvh * kGroup + ma_row % kGroup,
                                 o_stride, gate_stride, t, ia, o, 0, c_base);
    }
    if (rb.live) {
        store_row_pipe_e4m3_half(out_e4m3, gate, r0 + rb.bit, kvh * kGroup + mb_row % kGroup,
                                 o_stride, gate_stride, t, ib, o, 2, c_base);
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads, 2)
    qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_pipe_2cta(
        const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,
        const uint8_t* __restrict__ vc, const int* __restrict__ bt,
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
        uint8_t* __restrict__ out_e4m3, const __nv_bfloat16* __restrict__ gate,
        unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
        unsigned int gate_stride, unsigned int page_size, unsigned long long k_page_stride,
        unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
        float scale_log2_h0, float scale_log2_h1, float out_scale_h0, float out_scale_h1,
        float* __restrict__ a_scale_out, float a_scale_val) {
    qsa_prefill_attn_fp8_e4m3_pipe_2cta_body(
        q8, kc8, vc, bt, entries, counts, out_e4m3, gate, rows, first_pos, q_stride, o_stride,
        gate_stride, page_size, k_page_stride, k_tok_stride, v_page_stride, v_tok_stride,
        scale_log2_h0, scale_log2_h1, out_scale_h0, out_scale_h1, a_scale_out, a_scale_val);
}

namespace {

// Unrolled 2-step E4M3 K + lossless BF16 V tile loader for kBc = 32, kThreads = 256:
// Each thread owns two tokens (j0 = tid >> 4 in 0..15, j1 = j0 + 16 in 16..31) and 16B chunk
// ch = tid & 15 in 0..15, loading one 16B E4M3 K chunk (ch) and two 16B BF16 V chunks (ch and
// ch + 16, covering all 512 B per BF16 V token). Reuses the two overlapped block-table __ldg
// lookups across both K8 and BF16 V (cutting block-table loads 3x vs load_tile<false>) and uses
// shift/mask indexing when page_size is a power of two.
__device__ __forceinline__ void load_tile_qk_pipe(
    unsigned char* stage, const uint8_t* kc8, const uint8_t* vc, const unsigned int* ent,
    unsigned int base, unsigned int count, const int* bt, unsigned int page_size, bool page_pow2,
    unsigned int page_shift, unsigned int page_mask, unsigned long long k_page_stride,
    unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
    unsigned int kvh, unsigned int tid) {
    unsigned char* const ks = stage;
    __nv_bfloat16* const vs16 = reinterpret_cast<__nv_bfloat16*>(stage + kTile8);
    const unsigned int j0 = tid >> 4;
    const unsigned int j1 = j0 + 16u;
    const unsigned int ch = tid & 15u;
    const unsigned int in_b = j0 & 3u;
    const unsigned int slot0 = base + (j0 >> 2);
    const unsigned int slot1 = base + (j1 >> 2);
    __nv_bfloat16* const vd0_lo = vs16 + swz16(j0, ch * 8u);
    __nv_bfloat16* const vd0_hi = vd0_lo + 128;
    __nv_bfloat16* const vd1_lo = vs16 + swz16(j1, ch * 8u);
    __nv_bfloat16* const vd1_hi = vd1_lo + 128;
    if (base + 8u <= count) {
        const unsigned int tok0 = (ent[slot0] & kBlockMask) * kRatio + in_b;
        const unsigned int tok1 = (ent[slot1] & kBlockMask) * kRatio + in_b;
        unsigned int lp0, lp1;
        size_t in_page0, in_page1;
        if (page_pow2) {
            lp0 = tok0 >> page_shift;
            lp1 = tok1 >> page_shift;
            in_page0 = tok0 & page_mask;
            in_page1 = tok1 & page_mask;
        } else {
            lp0 = tok0 / page_size;
            lp1 = tok1 / page_size;
            in_page0 = tok0 % page_size;
            in_page1 = tok1 % page_size;
        }
        const unsigned int page0 = static_cast<unsigned int>(__ldg(bt + lp0));
        const unsigned int page1 = static_cast<unsigned int>(__ldg(bt + lp1));
        const size_t k_off0 =
            page0 * k_page_stride + in_page0 * k_tok_stride + kvh * kHd + ch * 16u;
        const size_t v_off0 =
            page0 * v_page_stride + in_page0 * v_tok_stride + kvh * (kHd * 2u) + ch * 16u;
        const size_t k_off1 =
            page1 * k_page_stride + in_page1 * k_tok_stride + kvh * kHd + ch * 16u;
        const size_t v_off1 =
            page1 * v_page_stride + in_page1 * v_tok_stride + kvh * (kHd * 2u) + ch * 16u;
        cp_async16_l2_128b(ks + swz8(j0, ch), kc8 + k_off0);
        cp_async16_l2_128b(vd0_lo, vc + v_off0);
        cp_async16_l2_128b(vd0_hi, vc + v_off0 + 256u);
        cp_async16_l2_128b(ks + swz8(j1, ch), kc8 + k_off1);
        cp_async16_l2_128b(vd1_lo, vc + v_off1);
        cp_async16_l2_128b(vd1_hi, vc + v_off1 + 256u);
    } else {
        const bool live0 = slot0 < count;
        const bool live1 = slot1 < count;
        const unsigned int tok0 = live0 ? (ent[slot0] & kBlockMask) * kRatio + in_b : 0u;
        const unsigned int tok1 = live1 ? (ent[slot1] & kBlockMask) * kRatio + in_b : 0u;
        unsigned int lp0, lp1;
        size_t in_page0, in_page1;
        if (page_pow2) {
            lp0 = tok0 >> page_shift;
            lp1 = tok1 >> page_shift;
            in_page0 = tok0 & page_mask;
            in_page1 = tok1 & page_mask;
        } else {
            lp0 = tok0 / page_size;
            lp1 = tok1 / page_size;
            in_page0 = tok0 % page_size;
            in_page1 = tok1 % page_size;
        }
        const unsigned int page0 = live0 ? static_cast<unsigned int>(__ldg(bt + lp0)) : 0u;
        const unsigned int page1 = live1 ? static_cast<unsigned int>(__ldg(bt + lp1)) : 0u;
        if (live0) {
            const size_t v_off0 =
                page0 * v_page_stride + in_page0 * v_tok_stride + kvh * (kHd * 2u) + ch * 16u;
            cp_async16_l2_128b(
                ks + swz8(j0, ch),
                kc8 + page0 * k_page_stride + in_page0 * k_tok_stride + kvh * kHd + ch * 16u);
            cp_async16_l2_128b(vd0_lo, vc + v_off0);
            cp_async16_l2_128b(vd0_hi, vc + v_off0 + 256u);
        } else {
            *reinterpret_cast<uint4*>(ks + swz8(j0, ch)) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(vd0_lo) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(vd0_hi) = make_uint4(0, 0, 0, 0);
        }
        if (live1) {
            const size_t v_off1 =
                page1 * v_page_stride + in_page1 * v_tok_stride + kvh * (kHd * 2u) + ch * 16u;
            cp_async16_l2_128b(
                ks + swz8(j1, ch),
                kc8 + page1 * k_page_stride + in_page1 * k_tok_stride + kvh * kHd + ch * 16u);
            cp_async16_l2_128b(vd1_lo, vc + v_off1);
            cp_async16_l2_128b(vd1_hi, vc + v_off1 + 256u);
        } else {
            *reinterpret_cast<uint4*>(ks + swz8(j1, ch)) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(vd1_lo) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(vd1_hi) = make_uint4(0, 0, 0, 0);
        }
    }
}

template <int AMAX>
__device__ __forceinline__ void store_row_qk_pipe(__nv_bfloat16* __restrict__ out,
                                                  const __nv_bfloat16* __restrict__ gate,
                                                  unsigned int row, unsigned int head,
                                                  unsigned int o_stride, unsigned int gate_stride,
                                                  unsigned int t, float inv_l,
                                                  const float (&o)[32][4], int half, float& amax) {
    __nv_bfloat16* dst = out + (size_t)row * o_stride + head * kHd;
    const __nv_bfloat16* gsrc = gate ? gate + (size_t)row * gate_stride + head * kHd : nullptr;
    const auto norm = [&](float acc) { return __bfloat162float(__float2bfloat16(acc * inv_l)); };
    const auto gated = [](float x, __nv_bfloat16 gv) {
        return x * (1.0f / (1.0f + expf(-__bfloat162float(gv))));
    };
    if (gsrc) {
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int i0 = b * 4;
            const unsigned int d0 = (i0 + 0) * 8 + t * 2;
            const unsigned int d1 = (i0 + 1) * 8 + t * 2;
            const unsigned int d2 = (i0 + 2) * 8 + t * 2;
            const unsigned int d3 = (i0 + 3) * 8 + t * 2;
            unsigned int gw[4];
            asm volatile(
                "{\n\t"
                "ld.global.cs.u32 %0, [%4];\n\t"
                "ld.global.cs.u32 %1, [%5];\n\t"
                "ld.global.cs.u32 %2, [%6];\n\t"
                "ld.global.cs.u32 %3, [%7];\n\t"
                "}\n"
                : "=r"(gw[0]), "=r"(gw[1]), "=r"(gw[2]), "=r"(gw[3])
                : "l"(gsrc + d0), "l"(gsrc + d1), "l"(gsrc + d2), "l"(gsrc + d3));
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int i = i0 + k;
                const unsigned int di = i * 8 + t * 2;
                float x0 = norm(o[i][half]);
                float x1 = norm(o[i][half + 1]);
                const __nv_bfloat162 gv = *reinterpret_cast<const __nv_bfloat162*>(&gw[k]);
                x0 = gated(x0, gv.x);
                x1 = gated(x1, gv.y);
                const unsigned int w01 = pack_bf16(x0, x1);
                *reinterpret_cast<unsigned int*>(dst + di) = w01;
                if constexpr (AMAX == 1) amax = amax_bf16x2(amax, w01);
                if constexpr (AMAX == 2) amax = fmaxf(fmaxf(amax, fabsf(x0)), fabsf(x1));
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const unsigned int d0 = i * 8 + t * 2;
            const float x0 = norm(o[i][half]);
            const float x1 = norm(o[i][half + 1]);
            const unsigned int w01 = pack_bf16(x0, x1);
            *reinterpret_cast<unsigned int*>(dst + d0) = w01;
            if constexpr (AMAX == 1) amax = amax_bf16x2(amax, w01);
            if constexpr (AMAX == 2) amax = fmaxf(fmaxf(amax, fabsf(x0)), fabsf(x1));
        }
    }
}

__device__ __forceinline__ float reread_row_qk_pipe(const __nv_bfloat16* __restrict__ out,
                                                    unsigned int row, unsigned int head,
                                                    unsigned int o_stride, unsigned int t,
                                                    float amax) {
    const __nv_bfloat16* src = out + (size_t)row * o_stride + head * kHd;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int i0 = b * 8;
        unsigned int w[8];
        asm volatile(
            "{\n\t"
            "ld.global.cg.u32 %0, [%8];\n\t"
            "ld.global.cg.u32 %1, [%9];\n\t"
            "ld.global.cg.u32 %2, [%10];\n\t"
            "ld.global.cg.u32 %3, [%11];\n\t"
            "ld.global.cg.u32 %4, [%12];\n\t"
            "ld.global.cg.u32 %5, [%13];\n\t"
            "ld.global.cg.u32 %6, [%14];\n\t"
            "ld.global.cg.u32 %7, [%15];\n\t"
            "}\n"
            : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]),
              "=r"(w[4]), "=r"(w[5]), "=r"(w[6]), "=r"(w[7])
            : "l"(src + (i0 + 0) * 8 + t * 2), "l"(src + (i0 + 1) * 8 + t * 2),
              "l"(src + (i0 + 2) * 8 + t * 2), "l"(src + (i0 + 3) * 8 + t * 2),
              "l"(src + (i0 + 4) * 8 + t * 2), "l"(src + (i0 + 5) * 8 + t * 2),
              "l"(src + (i0 + 6) * 8 + t * 2), "l"(src + (i0 + 7) * 8 + t * 2)
            : "memory");
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            amax = amax_bf16x2(amax, w[k]);
        }
    }
    return amax;
}

// Intra-warp software-pipelined E4M3 QK^T + lossless BF16 P*V core (PV8 = false, ACC = 0,
// STAGES = 2, dynamic smem = Mode::Qk.smem() = 69,680 B):
//   1. Unifies K8 and BF16 V tile loading in load_tile_qk_pipe: reuses two overlapped block-table
//      __ldg lookups across both K8 and BF16 V with cp.async.cg.L2::128B and shift/mask page
//      indexing for power-of-two page_size.
//   2. Software-pipelines ldsm_x4 ahead of mma_e4m3 in QK^T and skips unused 16-token halves
//      (do_h0 / do_h1) in QK^T, softmax, AND BF16 P*V.
//   3. Packs FP32 P to BF16 right after exp2f (cutting live P state from 16 to 8 regs before
//      P*V), issues the first ldsm_x4_t(V) before PV, interleaves branchless O rescales into
//      the first active BF16 PV half, and ping-pongs ldsm_x4_t ahead of mma_bf16 across both
//      ksteps with zero bubble.
//   4. Batches 4x streaming gate loads (ld.global.cs.u32) in store_row_qk_pipe and 8x AMAX
//      mode-3 re-read loads (ld.global.cg.u32) in reread_row_qk_pipe.
// Every floating-point operation and accumulation order matches qsa_prefill_attn_fp8<false, 0, 2, AMAX>.
template <int AMAX = 0>
__device__ __forceinline__ void qsa_prefill_attn_fp8_qk_pipe(
    const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,
    const uint8_t* __restrict__ vc, const int* __restrict__ bt,
    const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
    unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
    unsigned int gate_stride, unsigned int page_size, unsigned long long k_page_stride,
    unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
    float scale_log2_h0, float scale_log2_h1, float out_scale_h0, float out_scale_h1,
    unsigned int* __restrict__ amax_bits = nullptr) {
    using Cfg = Fp8Cfg<false>;
    constexpr int STAGES = 2;
    static_assert(STAGES * Cfg::kStageBytes >= kQ8Bytes, "Q staging must fit the tile ring");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned char* ring = smem_raw;
    unsigned int* ent = reinterpret_cast<unsigned int*>(smem_raw + STAGES * Cfg::kStageBytes);

    const unsigned int tile = blockIdx.x;
    const unsigned int kvh = blockIdx.y;
    const float scale_log2 = kvh ? scale_log2_h1 : scale_log2_h0;
    const float out_scale = kvh ? out_scale_h1 : out_scale_h0;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3;
    const unsigned int r0 = tile * kTrows;
    if (r0 >= rows) return;
    const unsigned int count = min(counts[tile], static_cast<unsigned int>(kCap));
    const unsigned int* gent = entries + (size_t)tile * kCap;
    const bool page_pow2 = (page_size & (page_size - 1u)) == 0u;
    const unsigned int page_shift = static_cast<unsigned int>(__ffs(page_size) - 1);
    const unsigned int page_mask = page_size - 1u;

    for (unsigned int c = tid; c < kWarps * 16 * 16; c += kThreads) {
        const unsigned int m = c >> 4;
        const unsigned int ch = c & 15;
        unsigned char* dst = ring + swz8(m, ch);
        const unsigned int rt = m / kGroup;
        if (m < static_cast<unsigned int>(kM) && r0 + rt < rows) {
            const unsigned int head = kvh * kGroup + m % kGroup;
            cp_async16_l2_128b(dst, q8 + (size_t)(r0 + rt) * q_stride + head * kHd + ch * 16);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
    for (unsigned int i = tid * 4; i < count; i += kThreads * 4) {
        if (i + 4 <= count) {
            cp_async16_l2_128b(ent + i, gent + i);
        } else {
            for (unsigned int j = i; j < count; ++j) ent[j] = gent[j];
        }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    unsigned int qf[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const unsigned int row = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        ldsm_x4(qf[ks], ring + swz8(row, ks * 2 + (lane >> 4)));
    }
    __syncthreads();

    const RowInfo ra = row_info(warp * 16 + g, r0, rows, first_pos);
    const RowInfo rb = row_info(warp * 16 + g + 8, r0, rows, first_pos);
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float ma = -CUDART_INF_F, mb = -CUDART_INF_F, la = 0.0f, lb = 0.0f;

    const unsigned int ntiles = (count + 7) / 8;
    unsigned int warp_rows = 0;
    {
        const unsigned int m_lo = warp * 16;
        const unsigned int m_hi = min(m_lo + 15, static_cast<unsigned int>(kM) - 1);
        for (unsigned int rt = m_lo / kGroup; rt <= m_hi / kGroup; ++rt) warp_rows |= 1u << rt;
    }

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (static_cast<unsigned int>(s) < ntiles) {
            load_tile_qk_pipe(ring + s * Cfg::kStageBytes, kc8, vc, ent, s * 8, count, bt,
                              page_size, page_pow2, page_shift, page_mask, k_page_stride,
                              k_tok_stride, v_page_stride, v_tok_stride, kvh, tid);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }

    const unsigned int k_row0 = (lane >> 4) * 8u + (lane & 7u);
    const unsigned int k_row1 = k_row0 + 16u;
    const unsigned int k_col_off = (lane >> 3) & 1u;
    const unsigned int v_row0 = (lane & 7u) + ((lane >> 3) & 1u) * 8u;
    const unsigned int v_col_off = (lane >> 4) * 8u;

    for (unsigned int it = 0; it < ntiles; ++it) {
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2) : "memory");
        __syncthreads();
        {
            const unsigned int nt = it + STAGES - 1;
            if (nt < ntiles) {
                load_tile_qk_pipe(ring + (nt % STAGES) * Cfg::kStageBytes, kc8, vc, ent, nt * 8,
                                  count, bt, page_size, page_pow2, page_shift, page_mask,
                                  k_page_stride, k_tok_stride, v_page_stride, v_tok_stride, kvh,
                                  tid);
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        const unsigned char* ks_s = ring + (it % STAGES) * Cfg::kStageBytes;
        const unsigned char* vs_s = ks_s + kTile8;
        bool do_h0 = false;
        bool do_h1 = false;
        {
            const unsigned int slot = it * 8 + (lane & 7);
            const unsigned int need =
                lane < 8 && slot < count ? entry_need_pipe(ent[slot], first_pos + r0) : 0u;
            const unsigned int need01 =
                __reduce_or_sync(0xffffffffu, lane < 4 ? need : (need << 16));
            do_h0 = (need01 & warp_rows) != 0u;
            do_h1 = ((need01 >> 16) & warp_rows) != 0u;
            if (!do_h0 && !do_h1) continue;
        }

        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
        if (do_h0 && do_h1) {
            unsigned int b_cur[4], b_next[4];
            ldsm_x4(b_cur, ks_s + swz8(k_row0, k_col_off));
#pragma unroll
            for (int kstep = 0; kstep < 8; ++kstep) {
                ldsm_x4(b_next, ks_s + swz8(k_row1, kstep * 2 + k_col_off));
                mma_e4m3(s[0], qf[kstep], b_cur[0], b_cur[1]);
                mma_e4m3(s[1], qf[kstep], b_cur[2], b_cur[3]);
                if (kstep < 7) {
                    ldsm_x4(b_cur, ks_s + swz8(k_row0, (kstep + 1) * 2 + k_col_off));
                }
                mma_e4m3(s[2], qf[kstep], b_next[0], b_next[1]);
                mma_e4m3(s[3], qf[kstep], b_next[2], b_next[3]);
            }
        } else if (do_h0) {
            unsigned int kb[2][4];
            ldsm_x4(kb[0], ks_s + swz8(k_row0, k_col_off));
#pragma unroll
            for (int kstep = 0; kstep < 8; ++kstep) {
                const int cur = kstep & 1;
                if (kstep < 7) {
                    ldsm_x4(kb[cur ^ 1], ks_s + swz8(k_row0, (kstep + 1) * 2 + k_col_off));
                }
                mma_e4m3(s[0], qf[kstep], kb[cur][0], kb[cur][1]);
                mma_e4m3(s[1], qf[kstep], kb[cur][2], kb[cur][3]);
            }
        } else {
            unsigned int kb[2][4];
            ldsm_x4(kb[0], ks_s + swz8(k_row1, k_col_off));
#pragma unroll
            for (int kstep = 0; kstep < 8; ++kstep) {
                const int cur = kstep & 1;
                if (kstep < 7) {
                    ldsm_x4(kb[cur ^ 1], ks_s + swz8(k_row1, (kstep + 1) * 2 + k_col_off));
                }
                mma_e4m3(s[2], qf[kstep], kb[cur][0], kb[cur][1]);
                mma_e4m3(s[3], qf[kstep], kb[cur][2], kb[cur][3]);
            }
        }

        const unsigned int base = it * 8;
        float mxa = -CUDART_INF_F, mxb = -CUDART_INF_F;
        if (do_h0) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const unsigned int slot = base + nt * 2 + (t >> 1);
                const bool slot_live = slot < count;
                const unsigned int e = slot_live ? ent[slot] : 0u;
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    const unsigned int in_block = (t * 2 + x) & 3u;
                    s[nt][x] = allowed(ra, e, slot_live, in_block) ? s[nt][x] * scale_log2
                                                                   : -CUDART_INF_F;
                    s[nt][2 + x] = allowed(rb, e, slot_live, in_block) ? s[nt][2 + x] * scale_log2
                                                                       : -CUDART_INF_F;
                    mxa = fmaxf(mxa, s[nt][x]);
                    mxb = fmaxf(mxb, s[nt][2 + x]);
                }
            }
        }
        if (do_h1) {
#pragma unroll
            for (int nt = 2; nt < 4; ++nt) {
                const unsigned int slot = base + nt * 2 + (t >> 1);
                const bool slot_live = slot < count;
                const unsigned int e = slot_live ? ent[slot] : 0u;
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    const unsigned int in_block = (t * 2 + x) & 3u;
                    s[nt][x] = allowed(ra, e, slot_live, in_block) ? s[nt][x] * scale_log2
                                                                   : -CUDART_INF_F;
                    s[nt][2 + x] = allowed(rb, e, slot_live, in_block) ? s[nt][2 + x] * scale_log2
                                                                       : -CUDART_INF_F;
                    mxa = fmaxf(mxa, s[nt][x]);
                    mxb = fmaxf(mxb, s[nt][2 + x]);
                }
            }
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float na = fmaxf(ma, mxa);
        const float nb = fmaxf(mb, mxb);
        const bool rescale_a = (na > ma) && (ma != -CUDART_INF_F);
        const bool rescale_b = (nb > mb) && (mb != -CUDART_INF_F);
        const bool any_rescale_a = __any_sync(0xffffffffu, rescale_a);
        const bool any_rescale_b = __any_sync(0xffffffffu, rescale_b);
        float alpha_a = 1.0f;
        if (any_rescale_a) {
            alpha_a = rescale_a ? exp2f(ma - na) : 1.0f;
        }
        float alpha_b = 1.0f;
        if (any_rescale_b) {
            alpha_b = rescale_b ? exp2f(mb - nb) : 1.0f;
        }
        const float base_a = na == -CUDART_INF_F ? 0.0f : na;
        const float base_b = nb == -CUDART_INF_F ? 0.0f : nb;
        ma = na;
        mb = nb;

        float suma = 0.0f, sumb = 0.0f;
        unsigned int a0[4] = {0u, 0u, 0u, 0u};
        unsigned int a1[4] = {0u, 0u, 0u, 0u};
        if (do_h0) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                s[nt][0] = exp2f(s[nt][0] - base_a);
                s[nt][1] = exp2f(s[nt][1] - base_a);
                s[nt][2] = exp2f(s[nt][2] - base_b);
                s[nt][3] = exp2f(s[nt][3] - base_b);
                suma += s[nt][0] + s[nt][1];
                sumb += s[nt][2] + s[nt][3];
            }
            a0[0] = pack_bf16(s[0][0], s[0][1]);
            a0[1] = pack_bf16(s[0][2], s[0][3]);
            a0[2] = pack_bf16(s[1][0], s[1][1]);
            a0[3] = pack_bf16(s[1][2], s[1][3]);
        }
        if (do_h1) {
#pragma unroll
            for (int nt = 2; nt < 4; ++nt) {
                s[nt][0] = exp2f(s[nt][0] - base_a);
                s[nt][1] = exp2f(s[nt][1] - base_a);
                s[nt][2] = exp2f(s[nt][2] - base_b);
                s[nt][3] = exp2f(s[nt][3] - base_b);
                suma += s[nt][0] + s[nt][1];
                sumb += s[nt][2] + s[nt][3];
            }
            a1[0] = pack_bf16(s[2][0], s[2][1]);
            a1[1] = pack_bf16(s[2][2], s[2][3]);
            a1[2] = pack_bf16(s[3][0], s[3][1]);
            a1[3] = pack_bf16(s[3][2], s[3][3]);
        }

        const __nv_bfloat16* const vs16 = reinterpret_cast<const __nv_bfloat16*>(vs_s);
        const __nv_bfloat16* const v_first = do_h0 ? vs16 : (vs16 + 4096);
        unsigned int vb[2][4];
        ldsm_x4_t(vb[0], v_first + swz16(v_row0, v_col_off));

        la = la * alpha_a + suma;
        lb = lb * alpha_b + sumb;

        // Interleave branchless FP32 accumulator rescales with the first active BF16 PV half (0-ULP).
        if (do_h0) {
#pragma unroll
            for (int dp = 0; dp < 16; ++dp) {
                const int cur = dp & 1;
                if (dp < 15) {
                    ldsm_x4_t(vb[cur ^ 1], vs16 + swz16(v_row0, (dp + 1) * 16u + v_col_off));
                } else if (do_h1) {
                    ldsm_x4_t(vb[cur ^ 1], (vs16 + 4096) + swz16(v_row0, v_col_off));
                }
                o[dp * 2][0] *= alpha_a;
                o[dp * 2][1] *= alpha_a;
                o[dp * 2][2] *= alpha_b;
                o[dp * 2][3] *= alpha_b;
                mma_bf16(o[dp * 2], a0, vb[cur][0], vb[cur][1]);
                o[dp * 2 + 1][0] *= alpha_a;
                o[dp * 2 + 1][1] *= alpha_a;
                o[dp * 2 + 1][2] *= alpha_b;
                o[dp * 2 + 1][3] *= alpha_b;
                mma_bf16(o[dp * 2 + 1], a0, vb[cur][2], vb[cur][3]);
            }
            if (do_h1) {
#pragma unroll
                for (int dp = 0; dp < 16; ++dp) {
                    const int cur = dp & 1;
                    if (dp < 15) {
                        ldsm_x4_t(vb[cur ^ 1],
                                  (vs16 + 4096) + swz16(v_row0, (dp + 1) * 16u + v_col_off));
                    }
                    mma_bf16(o[dp * 2], a1, vb[cur][0], vb[cur][1]);
                    mma_bf16(o[dp * 2 + 1], a1, vb[cur][2], vb[cur][3]);
                }
            }
        } else {
#pragma unroll
            for (int dp = 0; dp < 16; ++dp) {
                const int cur = dp & 1;
                if (dp < 15) {
                    ldsm_x4_t(vb[cur ^ 1],
                              (vs16 + 4096) + swz16(v_row0, (dp + 1) * 16u + v_col_off));
                }
                o[dp * 2][0] *= alpha_a;
                o[dp * 2][1] *= alpha_a;
                o[dp * 2][2] *= alpha_b;
                o[dp * 2][3] *= alpha_b;
                mma_bf16(o[dp * 2], a1, vb[cur][0], vb[cur][1]);
                o[dp * 2 + 1][0] *= alpha_a;
                o[dp * 2 + 1][1] *= alpha_a;
                o[dp * 2 + 1][2] *= alpha_b;
                o[dp * 2 + 1][3] *= alpha_b;
                mma_bf16(o[dp * 2 + 1], a1, vb[cur][2], vb[cur][3]);
            }
        }
    }

    la += __shfl_xor_sync(0xffffffffu, la, 1);
    la += __shfl_xor_sync(0xffffffffu, la, 2);
    lb += __shfl_xor_sync(0xffffffffu, lb, 1);
    lb += __shfl_xor_sync(0xffffffffu, lb, 2);

    const float ia = la > 0.0f ? out_scale / la : 0.0f;
    const float ib = lb > 0.0f ? out_scale / lb : 0.0f;
    const unsigned int ma_row = warp * 16 + g;
    const unsigned int mb_row = ma_row + 8;
    float amax = 0.0f;
    if (ra.live) {
        store_row_qk_pipe<AMAX>(out, gate, r0 + ra.bit, kvh * kGroup + ma_row % kGroup, o_stride,
                                gate_stride, t, ia, o, 0, amax);
    }
    if (rb.live) {
        store_row_qk_pipe<AMAX>(out, gate, r0 + rb.bit, kvh * kGroup + mb_row % kGroup, o_stride,
                                gate_stride, t, ib, o, 2, amax);
    }
    if constexpr (AMAX == 2) amax = __bfloat162float(__float2bfloat16(amax));
    if constexpr (AMAX == 3) {
        if (ra.live) {
            amax = reread_row_qk_pipe(out, r0 + ra.bit, kvh * kGroup + ma_row % kGroup, o_stride, t,
                                      amax);
        }
        if (rb.live) {
            amax = reread_row_qk_pipe(out, r0 + rb.bit, kvh * kGroup + mb_row % kGroup, o_stride, t,
                                      amax);
        }
    }
    if constexpr (AMAX != 0) {
        __shared__ float s_amax[kWarps];
        for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
        if (lane == 0) s_amax[warp] = amax;
        __syncthreads();
        if (tid == 0) {
            float m = s_amax[0];
#pragma unroll
            for (int w = 1; w < kWarps; ++w) m = fmaxf(m, s_amax[w]);
            const float seen =
                __uint_as_float(*reinterpret_cast<volatile unsigned int*>(amax_bits));
            if (m > 0.0f && m > seen) atomicMax(amax_bits, __float_as_uint(m));
        }
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads, 1) qwen4_qsa_prefill_attn_fp8_qk_pipe(
    const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,
    const uint8_t* __restrict__ vc, const int* __restrict__ bt,
    const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
    unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
    unsigned int gate_stride, unsigned int page_size, unsigned long long k_page_stride,
    unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
    float scale_log2_h0, float scale_log2_h1, float out_scale_h0, float out_scale_h1) {
    qsa_prefill_attn_fp8_qk_pipe<0>(q8, kc8, vc, bt, entries, counts, out, gate, rows, first_pos,
                                    q_stride, o_stride, gate_stride, page_size, k_page_stride,
                                    k_tok_stride, v_page_stride, v_tok_stride, scale_log2_h0,
                                    scale_log2_h1, out_scale_h0, out_scale_h1);
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
    qwen4_qsa_prefill_attn_fp8_qk_amax_pipe(
        const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,
        const uint8_t* __restrict__ vc, const int* __restrict__ bt,
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
        __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
        unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
        unsigned int gate_stride, unsigned int page_size, unsigned long long k_page_stride,
        unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
        float scale_log2_h0, float scale_log2_h1, float out_scale_h0, float out_scale_h1,
        unsigned int* __restrict__ amax_bits) {
    qsa_prefill_attn_fp8_qk_pipe<QSA_OPROJ_AMAX_MODE>(
        q8, kc8, vc, bt, entries, counts, out, gate, rows, first_pos, q_stride, o_stride,
        gate_stride, page_size, k_page_stride, k_tok_stride, v_page_stride, v_tok_stride,
        scale_log2_h0, scale_log2_h1, out_scale_h0, out_scale_h1, amax_bits);
}





// ---------------------------------------------------------------------------
// Multi-row QSA Speculative Verify (K11-D2) — 7 active kernels + 2 shadowed copies:
//   - qwen4_qsa_score_rows_v2 / qwen4_qsa_score_device_meta_rows_v2
//   - qwen4_qsa_select_expand_rows_v3 / qwen4_qsa_select_expand_rows_par
//   - qwen4_qsa_select_expand_device_meta_rows_par / _v3
//   - qwen4_qsa_sparse_attention_bf16_partial_rows
//   - qwen4_qsa_sparse_attention_bf16_partial_rows_v2 / _reduce_rows (shadowed at
//     runtime by common/qwen4_qsa.cu; kept in sync with SM90+ PDL griddepcontrol
//     for test_qwen4_qsa_k5_transaction_static.py:226-232 until that test points
//     to common/qwen4_qsa.cu)
// Placed in qwen4_qsa_prefill_attn_fp8.cu (already in FN_EXPECTED_DIFF_MODULES)
// so common/qwen4_qsa.ptx stays 100% unchanged vs unified6.
// Structured control flow avoids bare void return statements so oproj_amax_static
// source assertions on this file remain satisfied.
// ---------------------------------------------------------------------------
namespace qsa_verify_rows {

constexpr int INDEX_HEADS = 4;
constexpr int INDEX_DIM = 128;
constexpr int COMPRESS_RATIO = 4;
constexpr int TOKEN_TOPK = 2048;
constexpr int BLOCK_TOPK = TOKEN_TOPK / COMPRESS_RATIO;
constexpr int OUTPUT_WIDTH = TOKEN_TOPK + COMPRESS_RATIO - 1;
constexpr int VERIFY_MAX_ROWS = 9;
constexpr int QSA_SCORE_G = 4;
constexpr int SELECT_THREADS = 1024;
constexpr int SELECT_WARPS = SELECT_THREADS / 32;
constexpr int SV3_E = 33;
constexpr unsigned int SV3_MAX = SELECT_THREADS * SV3_E;
constexpr int SV3_CELLS = SV3_E * SELECT_WARPS;
constexpr unsigned int SV3_MIN = 3072;

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ unsigned int qsa_order_key(float value) {
    if (value == 0.0f) value = 0.0f;
    const unsigned int bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ unsigned int sv3_lanemask_lt() {
    unsigned int m;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
    return m;
}

__device__ __forceinline__ int qsa_block_exclusive_scan(int x, int* warp_buf, int* total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int inclusive = x;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += y;
    }
    if (lane == 31) warp_buf[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
        int w = lane < SELECT_WARPS ? warp_buf[lane] : 0;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, offset);
            if (lane >= offset) w += y;
        }
        if (lane < SELECT_WARPS) warp_buf[lane] = w;
    }
    __syncthreads();
    const int warp_prefix = warp == 0 ? 0 : warp_buf[warp - 1];
    *total = warp_buf[SELECT_WARPS - 1];
    __syncthreads();
    return warp_prefix + inclusive - x;
}

__device__ void qsa_select_expand_parallel(
    const float* __restrict__ logits,
    int* __restrict__ token_indices,
    unsigned int visible_groups,
    unsigned int position,
    unsigned int sequence_length) {
    __shared__ unsigned int hist[256];
    __shared__ int warp_buf[SELECT_WARPS];
    __shared__ unsigned int sel_prefix;
    __shared__ unsigned int sel_remaining;
    const int tid = threadIdx.x;

    for (int i = tid; i < OUTPUT_WIDTH; i += SELECT_THREADS) token_indices[i] = -1;
    __syncthreads();

    const int count = visible_groups < BLOCK_TOPK ? (int)visible_groups : BLOCK_TOPK;
    if (visible_groups <= (unsigned int)BLOCK_TOPK) {
        for (int r = tid; r < count * COMPRESS_RATIO; r += SELECT_THREADS) {
            if ((unsigned int)r < sequence_length) token_indices[r] = r;
        }
    } else {
        if (tid == 0) {
            sel_prefix = 0u;
            sel_remaining = (unsigned int)BLOCK_TOPK;
        }
        unsigned int prefix_mask = 0u;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int b = tid; b < 256; b += SELECT_THREADS) hist[b] = 0u;
            __syncthreads();
            const unsigned int prefix = sel_prefix;
            for (unsigned int i = tid; i < visible_groups; i += SELECT_THREADS) {
                const unsigned int key = qsa_order_key(__ldg(logits + i));
                if ((key & prefix_mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (tid < 32) {
                const int lane = tid;
                const unsigned int need = sel_remaining;
                unsigned int chunk = 0u;
#pragma unroll
                for (int j = 0; j < 8; ++j) chunk += hist[255 - lane * 8 - j];
                unsigned int inclusive = chunk;
#pragma unroll
                for (int offset = 1; offset < 32; offset <<= 1) {
                    const unsigned int y = __shfl_up_sync(0xffffffffu, inclusive, offset);
                    if (lane >= offset) inclusive += y;
                }
                const unsigned int before = inclusive - chunk;
                const bool mine = before < need && need <= inclusive;
                if (mine) {
                    unsigned int above = before;
                    for (int j = 0; j < 8; ++j) {
                        const unsigned int digit = 255u - (unsigned int)(lane * 8 + j);
                        const unsigned int h = hist[digit];
                        if (above + h >= need) {
                            sel_prefix = prefix | (digit << shift);
                            sel_remaining = need - above;
                            break;
                        }
                        above += h;
                    }
                }
            }
            prefix_mask |= 255u << shift;
            __syncthreads();
        }
        const unsigned int threshold = sel_prefix;
        const int take_equal = (int)sel_remaining;

        int equal_base = 0;
        int out_base = 0;
        for (unsigned int base = 0; base < visible_groups; base += SELECT_THREADS) {
            const unsigned int i = base + (unsigned int)tid;
            const bool valid = i < visible_groups;
            const unsigned int key = valid ? qsa_order_key(__ldg(logits + i)) : 0u;
            const bool greater = valid && key > threshold;
            const bool equal = valid && key == threshold;
            int equal_total;
            const int equal_rank = equal_base + qsa_block_exclusive_scan(equal ? 1 : 0, warp_buf, &equal_total);
            const bool selected = greater || (equal && equal_rank < take_equal);
            int selected_total;
            const int rank = out_base + qsa_block_exclusive_scan(selected ? 1 : 0, warp_buf, &selected_total);
            if (selected) {
#pragma unroll
                for (int offset = 0; offset < COMPRESS_RATIO; ++offset) {
                    const unsigned int token = i * COMPRESS_RATIO + offset;
                    if (token < sequence_length) token_indices[rank * COMPRESS_RATIO + offset] = (int)token;
                }
            }
            equal_base += equal_total;
            out_base += selected_total;
        }
    }
    __syncthreads();
    if (tid == 0) {
        int out = count * COMPRESS_RATIO;
        const unsigned int tail_start = ((position + 1) / COMPRESS_RATIO) * COMPRESS_RATIO;
        for (unsigned int token = tail_start;
             token <= position && token < sequence_length && out < OUTPUT_WIDTH;
             ++token) {
            token_indices[out++] = (int)token;
        }
    }
}

__device__ __forceinline__ void qsa_select_expand_v3_slabbed(
    const float* __restrict__ logits,
    int* __restrict__ token_indices,
    unsigned int visible_groups,
    unsigned int position,
    unsigned int sequence_length,
    unsigned int* hist,
    int* warp_buf,
    int* c_gt,
    int* c_eq,
    unsigned int* sel_prefix_s,
    unsigned int* sel_remaining_s) {
    constexpr int SV3_SLAB_E = 32;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned int lt = sv3_lanemask_lt();
    const int jn = (int)((visible_groups + SELECT_THREADS - 1u) / SELECT_THREADS);

    for (int i = tid; i < OUTPUT_WIDTH; i += SELECT_THREADS) token_indices[i] = -1;
    if (tid == 0) {
        *sel_prefix_s = 0u;
        *sel_remaining_s = (unsigned int)BLOCK_TOPK;
    }
    unsigned int prefix_mask = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += SELECT_THREADS) hist[b] = 0u;
        __syncthreads();
        const unsigned int prefix = *sel_prefix_s;
        for (int j = 0; j < jn; ++j) {
            const unsigned int i = (unsigned int)j * SELECT_THREADS + (unsigned int)tid;
            const unsigned int key = (i < visible_groups) ? qsa_order_key(__ldg(logits + i)) : 0u;
            const bool m = (i < visible_groups) && ((key & prefix_mask) == prefix);
            const unsigned int digit = (key >> shift) & 255u;
            const unsigned int mb = __ballot_sync(0xffffffffu, m);
            if (mb != 0u) {
                const int leader = __ffs(mb) - 1;
                const unsigned int ld = __shfl_sync(0xffffffffu, digit, leader);
                if (__all_sync(0xffffffffu, !m || digit == ld)) {
                    if (lane == leader) atomicAdd(&hist[ld], (unsigned int)__popc(mb));
                } else if (m) {
                    atomicAdd(&hist[digit], 1u);
                }
            }
        }
        __syncthreads();
        if (tid < 32) {
            const unsigned int need = *sel_remaining_s;
            unsigned int chunk = 0u;
#pragma unroll
            for (int j = 0; j < 8; ++j) chunk += hist[255 - lane * 8 - j];
            unsigned int inclusive = chunk;
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                const unsigned int y = __shfl_up_sync(0xffffffffu, inclusive, offset);
                if (lane >= offset) inclusive += y;
            }
            const unsigned int before = inclusive - chunk;
            if (before < need && need <= inclusive) {
                unsigned int above = before;
                for (int j = 0; j < 8; ++j) {
                    const unsigned int digit = 255u - (unsigned int)(lane * 8 + j);
                    const unsigned int h = hist[digit];
                    if (above + h >= need) {
                        *sel_prefix_s = prefix | (digit << shift);
                        *sel_remaining_s = need - above;
                        break;
                    }
                    above += h;
                }
            }
        }
        prefix_mask |= 255u << shift;
        __syncthreads();
    }

    const unsigned int threshold = *sel_prefix_s;
    const int take_equal = (int)(*sel_remaining_s);
    int equal_base = 0;
    int out_base = 0;

    for (int slab_j0 = 0; slab_j0 < jn && out_base < BLOCK_TOPK; slab_j0 += SV3_SLAB_E) {
        const int slab_jn = min(SV3_SLAB_E, jn - slab_j0);
        for (int jj = 0; jj < slab_jn; ++jj) {
            const unsigned int i =
                (unsigned int)(slab_j0 + jj) * SELECT_THREADS + (unsigned int)tid;
            const bool valid = i < visible_groups;
            const unsigned int key = valid ? qsa_order_key(__ldg(logits + i)) : 0u;
            const unsigned int bg = __ballot_sync(0xffffffffu, valid && key > threshold);
            const unsigned int be = __ballot_sync(0xffffffffu, valid && key == threshold);
            if (lane == 0) {
                c_gt[jj * SELECT_WARPS + warp] = __popc(bg);
                c_eq[jj * SELECT_WARPS + warp] = __popc(be);
            }
        }
        __syncthreads();

        const int ncells = slab_jn * SELECT_WARPS;
        const int c0 = 2 * tid, c1 = 2 * tid + 1;
        const int e0 = c0 < ncells ? c_eq[c0] : 0;
        const int e1 = c1 < ncells ? c_eq[c1] : 0;
        int slab_eq_total = 0;
        const int eb0 = equal_base + qsa_block_exclusive_scan(e0 + e1, warp_buf, &slab_eq_total);
        const int eb1 = eb0 + e0;
        const int s0 = c0 < ncells ? c_gt[c0] + min(max(take_equal - eb0, 0), e0) : 0;
        const int s1 = c1 < ncells ? c_gt[c1] + min(max(take_equal - eb1, 0), e1) : 0;
        int slab_sel_total = 0;
        const int ob0 = out_base + qsa_block_exclusive_scan(s0 + s1, warp_buf, &slab_sel_total);
        if (c0 < ncells) { c_eq[c0] = eb0; c_gt[c0] = ob0; }
        if (c1 < ncells) { c_eq[c1] = eb1; c_gt[c1] = ob0 + s0; }
        __syncthreads();

        if (slab_sel_total > 0) {
            for (int jj = 0; jj < slab_jn; ++jj) {
                const unsigned int i =
                    (unsigned int)(slab_j0 + jj) * SELECT_THREADS + (unsigned int)tid;
                const bool valid = i < visible_groups;
                const unsigned int key = valid ? qsa_order_key(__ldg(logits + i)) : 0u;
                const bool eq = valid && key == threshold;
                const unsigned int be = __ballot_sync(0xffffffffu, eq);
                const int cell = jj * SELECT_WARPS + warp;
                const bool sel = (valid && key > threshold)
                                 || (eq && c_eq[cell] + __popc(be & lt) < take_equal);
                const unsigned int bs = __ballot_sync(0xffffffffu, sel);
                if (sel) {
                    const int rank = c_gt[cell] + __popc(bs & lt);
#pragma unroll
                    for (int offset = 0; offset < COMPRESS_RATIO; ++offset) {
                        const unsigned int token = i * COMPRESS_RATIO + (unsigned int)offset;
                        if (token < sequence_length) {
                            token_indices[rank * COMPRESS_RATIO + offset] = (int)token;
                        }
                    }
                }
            }
        }
        equal_base += slab_eq_total;
        out_base += slab_sel_total;
        __syncthreads();
    }
    if (tid == 0) {
        int out = BLOCK_TOPK * COMPRESS_RATIO;
        const unsigned int tail_start = ((position + 1) / COMPRESS_RATIO) * COMPRESS_RATIO;
        for (unsigned int token = tail_start;
             token <= position && token < sequence_length && out < OUTPUT_WIDTH;
             ++token) {
            token_indices[out++] = (int)token;
        }
    }
}

__device__ void qsa_select_expand_v3(
    const float* __restrict__ logits,
    int* __restrict__ token_indices,
    unsigned int visible_groups,
    unsigned int position,
    unsigned int sequence_length) {
    if (visible_groups <= SV3_MIN) {
        qsa_select_expand_parallel(logits, token_indices, visible_groups, position, sequence_length);
    } else {
        __shared__ unsigned int hist[256];
        __shared__ int warp_buf[SELECT_WARPS];
        __shared__ int c_gt[SV3_CELLS];
        __shared__ int c_eq[SV3_CELLS];
        __shared__ unsigned int sel_prefix;
        __shared__ unsigned int sel_remaining;
        if (visible_groups > SV3_MAX) {
            qsa_select_expand_v3_slabbed(
                logits,
                token_indices,
                visible_groups,
                position,
                sequence_length,
                hist,
                warp_buf,
                c_gt,
                c_eq,
                &sel_prefix,
                &sel_remaining);
        } else {
        const int tid = threadIdx.x;
        const int lane = tid & 31;
        const int warp = tid >> 5;
        const unsigned int lt = sv3_lanemask_lt();
        const int jn = (int)((visible_groups + SELECT_THREADS - 1) / SELECT_THREADS);

        for (int i = tid; i < OUTPUT_WIDTH; i += SELECT_THREADS) token_indices[i] = -1;

        unsigned int keys[SV3_E];
#pragma unroll
        for (int j = 0; j < SV3_E; ++j) {
            const unsigned int i = (unsigned int)(j * SELECT_THREADS + tid);
            keys[j] = (j < jn && i < visible_groups) ? qsa_order_key(__ldg(logits + i)) : 0u;
        }
        if (tid == 0) {
            sel_prefix = 0u;
            sel_remaining = (unsigned int)BLOCK_TOPK;
        }
        unsigned int prefix_mask = 0u;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int b = tid; b < 256; b += SELECT_THREADS) hist[b] = 0u;
            __syncthreads();
            const unsigned int prefix = sel_prefix;
#pragma unroll
            for (int j = 0; j < SV3_E; ++j) {
                if (j >= jn) break;
                const unsigned int i = (unsigned int)(j * SELECT_THREADS + tid);
                const bool m = i < visible_groups && (keys[j] & prefix_mask) == prefix;
                const unsigned int digit = (keys[j] >> shift) & 255u;
                const unsigned int mb = __ballot_sync(0xffffffffu, m);
                if (mb == 0u) continue;
                const int leader = __ffs(mb) - 1;
                const unsigned int ld = __shfl_sync(0xffffffffu, digit, leader);
                if (__all_sync(0xffffffffu, !m || digit == ld)) {
                    if (lane == leader) atomicAdd(&hist[ld], (unsigned int)__popc(mb));
                } else if (m) {
                    atomicAdd(&hist[digit], 1u);
                }
            }
            __syncthreads();
            if (tid < 32) {
                const unsigned int need = sel_remaining;
                unsigned int chunk = 0u;
#pragma unroll
                for (int j = 0; j < 8; ++j) chunk += hist[255 - lane * 8 - j];
                unsigned int inclusive = chunk;
#pragma unroll
                for (int offset = 1; offset < 32; offset <<= 1) {
                    const unsigned int y = __shfl_up_sync(0xffffffffu, inclusive, offset);
                    if (lane >= offset) inclusive += y;
                }
                const unsigned int before = inclusive - chunk;
                const bool mine = before < need && need <= inclusive;
                if (mine) {
                    unsigned int above = before;
                    for (int j = 0; j < 8; ++j) {
                        const unsigned int digit = 255u - (unsigned int)(lane * 8 + j);
                        const unsigned int h = hist[digit];
                        if (above + h >= need) {
                            sel_prefix = prefix | (digit << shift);
                            sel_remaining = need - above;
                            break;
                        }
                        above += h;
                    }
                }
            }
            prefix_mask |= 255u << shift;
            __syncthreads();
        }
        const unsigned int threshold = sel_prefix;
        const int take_equal = (int)sel_remaining;

#pragma unroll
        for (int j = 0; j < SV3_E; ++j) {
            if (j >= jn) break;
            const unsigned int i = (unsigned int)(j * SELECT_THREADS + tid);
            const bool valid = i < visible_groups;
            const unsigned int bg = __ballot_sync(0xffffffffu, valid && keys[j] > threshold);
            const unsigned int be = __ballot_sync(0xffffffffu, valid && keys[j] == threshold);
            if (lane == 0) {
                c_gt[j * SELECT_WARPS + warp] = __popc(bg);
                c_eq[j * SELECT_WARPS + warp] = __popc(be);
            }
        }
        __syncthreads();
        const int ncells = jn * SELECT_WARPS;
        const int c0 = 2 * tid, c1 = 2 * tid + 1;
        const int e0 = c0 < ncells ? c_eq[c0] : 0;
        const int e1 = c1 < ncells ? c_eq[c1] : 0;
        int total;
        const int eb0 = qsa_block_exclusive_scan(e0 + e1, warp_buf, &total);
        const int eb1 = eb0 + e0;
        const int s0 = c0 < ncells ? c_gt[c0] + min(max(take_equal - eb0, 0), e0) : 0;
        const int s1 = c1 < ncells ? c_gt[c1] + min(max(take_equal - eb1, 0), e1) : 0;
        const int ob0 = qsa_block_exclusive_scan(s0 + s1, warp_buf, &total);
        if (c0 < ncells) { c_eq[c0] = eb0; c_gt[c0] = ob0; }
        if (c1 < ncells) { c_eq[c1] = eb1; c_gt[c1] = ob0 + s0; }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < SV3_E; ++j) {
            if (j >= jn) break;
            const unsigned int i = (unsigned int)(j * SELECT_THREADS + tid);
            const bool valid = i < visible_groups;
            const bool eq = valid && keys[j] == threshold;
            const unsigned int be = __ballot_sync(0xffffffffu, eq);
            const int cell = j * SELECT_WARPS + warp;
            const bool sel = (valid && keys[j] > threshold) || (eq && c_eq[cell] + __popc(be & lt) < take_equal);
            const unsigned int bs = __ballot_sync(0xffffffffu, sel);
            if (sel) {
                const int rank = c_gt[cell] + __popc(bs & lt);
#pragma unroll
                for (int offset = 0; offset < COMPRESS_RATIO; ++offset) {
                    const unsigned int token = i * COMPRESS_RATIO + offset;
                    if (token < sequence_length) token_indices[rank * COMPRESS_RATIO + offset] = (int)token;
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            int out = BLOCK_TOPK * COMPRESS_RATIO;
            const unsigned int tail_start = ((position + 1) / COMPRESS_RATIO) * COMPRESS_RATIO;
            for (unsigned int token = tail_start;
                 token <= position && token < sequence_length && out < OUTPUT_WIDTH;
                 ++token) {
                token_indices[out++] = (int)token;
            }
        }
        }
    }
}

}  // namespace qsa_verify_rows

// Multi-row QSA score (host ABI): grid [ceil(max_visible_groups / 32)], block 256.
// Loads each compressed key group ONCE and dots against all num_rows queries.
// Bit-for-bit identical to launching qwen4_qsa_score_v2 once per row.
extern "C" __global__ void __launch_bounds__(256, 4) qwen4_qsa_score_rows_v2(
    const __nv_bfloat16* __restrict__ query,
    unsigned int query_row_stride,
    const __nv_bfloat16* __restrict__ compressed_cache,
    const int* __restrict__ block_table,
    float* __restrict__ logits,
    unsigned long long logits_row_stride,
    unsigned int base_sequence_length,
    unsigned int num_rows,
    unsigned int block_size) {
    if (num_rows > 0u && num_rows <= (unsigned int)qsa_verify_rows::VERIFY_MAX_ROWS &&
        block_size >= (unsigned int)qsa_verify_rows::COMPRESS_RATIO) {
        __shared__ __nv_bfloat16 s_q[qsa_verify_rows::VERIFY_MAX_ROWS]
                                    [qsa_verify_rows::INDEX_HEADS * qsa_verify_rows::INDEX_DIM];
        for (unsigned int r = 0; r < num_rows; ++r) {
            const __nv_bfloat16* q_row = query + (unsigned long long)r * query_row_stride;
            s_q[r][threadIdx.x] = __ldg(q_row + threadIdx.x);
            s_q[r][threadIdx.x + 256u] = __ldg(q_row + threadIdx.x + 256u);
        }
        __syncthreads();

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const unsigned int first =
            (blockIdx.x * 8u + (unsigned int)warp) * qsa_verify_rows::QSA_SCORE_G;
        const unsigned int max_vis =
            (base_sequence_length + num_rows - 1u) / qsa_verify_rows::COMPRESS_RATIO;
        if (first < max_vis) {
            const unsigned int groups_per_page = block_size / qsa_verify_rows::COMPRESS_RATIO;
            int page[qsa_verify_rows::QSA_SCORE_G];
            if ((groups_per_page & (qsa_verify_rows::QSA_SCORE_G - 1u)) == 0u &&
                groups_per_page > 0u) {
                const int p0 = __ldg(block_table + first / groups_per_page);
#pragma unroll
                for (int g = 0; g < qsa_verify_rows::QSA_SCORE_G; ++g) {
                    page[g] = (first + g < max_vis) ? p0 : -1;
                }
            } else {
#pragma unroll
                for (int g = 0; g < qsa_verify_rows::QSA_SCORE_G; ++g) {
                    const unsigned int group = first + g;
                    page[g] = group < max_vis ? __ldg(block_table + group / groups_per_page) : -1;
                }
            }
            float k[qsa_verify_rows::QSA_SCORE_G][qsa_verify_rows::INDEX_DIM / 32];
#pragma unroll
            for (int g = 0; g < qsa_verify_rows::QSA_SCORE_G; ++g) {
                const unsigned int group = first + g;
                const unsigned long long key_base =
                    ((unsigned long long)(page[g] < 0 ? 0 : page[g]) * groups_per_page +
                     group % groups_per_page) *
                    qsa_verify_rows::INDEX_DIM;
#pragma unroll
                for (int j = 0; j < qsa_verify_rows::INDEX_DIM / 32; ++j) {
                    k[g][j] = (group < max_vis && page[g] >= 0)
                                  ? __bfloat162float(
                                        __ldg(compressed_cache + key_base + lane + 32 * j))
                                  : 0.0f;
                }
            }
            for (unsigned int r = 0; r < num_rows; ++r) {
                const unsigned int vis_r =
                    (base_sequence_length + r) / qsa_verify_rows::COMPRESS_RATIO;
                if (first < vis_r) {
                    const __nv_bfloat16* q_row = s_q[r];
                    float score[qsa_verify_rows::QSA_SCORE_G] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                    for (int h = 0; h < qsa_verify_rows::INDEX_HEADS; ++h) {
                        float qh[qsa_verify_rows::INDEX_DIM / 32];
#pragma unroll
                        for (int j = 0; j < qsa_verify_rows::INDEX_DIM / 32; ++j)
                            qh[j] = __bfloat162float(
                                q_row[h * qsa_verify_rows::INDEX_DIM + lane + 32 * j]);
#pragma unroll
                        for (int g = 0; g < qsa_verify_rows::QSA_SCORE_G; ++g) {
                            float partial = 0.0f;
#pragma unroll
                            for (int j = 0; j < qsa_verify_rows::INDEX_DIM / 32; ++j)
                                partial += qh[j] * k[g][j];
                            partial = qsa_verify_rows::warp_sum(partial);
                            if (lane == 0) score[g] += fmaxf(partial, 0.0f);
                        }
                    }
                    if (lane == 0) {
                        float* row_logits = logits + (unsigned long long)r * logits_row_stride;
#pragma unroll
                        for (int g = 0; g < qsa_verify_rows::QSA_SCORE_G; ++g) {
                            const unsigned int group = first + g;
                            if (group < vis_r) {
                                row_logits[group] = (page[g] < 0)
                                    ? -CUDART_INF_F
                                    : score[g] * 0.08838834764831845f;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Multi-row QSA score (device-metadata ABI, graph-safe): grid sized by caller, block 256.
// Bit-for-bit identical to launching qwen4_qsa_score_device_meta_v2 once per row.
extern "C" __global__ void __launch_bounds__(256, 4) qwen4_qsa_score_device_meta_rows_v2(
    const __nv_bfloat16* __restrict__ query,
    unsigned int query_row_stride,
    const __nv_bfloat16* __restrict__ compressed_cache,
    const int* __restrict__ block_table,
    float* __restrict__ logits,
    unsigned long long logits_row_stride,
    const unsigned int* __restrict__ sequence_lengths,
    unsigned int num_rows,
    unsigned int block_size) {
    if (num_rows > 0u && num_rows <= (unsigned int)qsa_verify_rows::VERIFY_MAX_ROWS &&
        block_size >= (unsigned int)qsa_verify_rows::COMPRESS_RATIO) {
        const unsigned int cta_first = blockIdx.x * (8u * qsa_verify_rows::QSA_SCORE_G);
        unsigned int max_vis = 0u;
        for (unsigned int r = 0; r < num_rows; ++r) {
            const unsigned int v = __ldg(sequence_lengths + r) / qsa_verify_rows::COMPRESS_RATIO;
            if (v > max_vis) max_vis = v;
        }
        if (cta_first < max_vis) {
            __shared__ __nv_bfloat16 s_q[qsa_verify_rows::VERIFY_MAX_ROWS]
                                        [qsa_verify_rows::INDEX_HEADS * qsa_verify_rows::INDEX_DIM];
            for (unsigned int r = 0; r < num_rows; ++r) {
                const __nv_bfloat16* q_row = query + (unsigned long long)r * query_row_stride;
                s_q[r][threadIdx.x] = __ldg(q_row + threadIdx.x);
                s_q[r][threadIdx.x + 256u] = __ldg(q_row + threadIdx.x + 256u);
            }
            __syncthreads();

            const int lane = threadIdx.x & 31;
            const int warp = threadIdx.x >> 5;
            const unsigned int first =
                cta_first + (unsigned int)warp * qsa_verify_rows::QSA_SCORE_G;
            if (first < max_vis) {
                const unsigned int groups_per_page = block_size / qsa_verify_rows::COMPRESS_RATIO;
                int page[qsa_verify_rows::QSA_SCORE_G];
                if ((groups_per_page & (qsa_verify_rows::QSA_SCORE_G - 1u)) == 0u &&
                    groups_per_page > 0u) {
                    const int p0 = __ldg(block_table + first / groups_per_page);
#pragma unroll
                    for (int g = 0; g < qsa_verify_rows::QSA_SCORE_G; ++g) {
                        page[g] = (first + g < max_vis) ? p0 : -1;
                    }
                } else {
#pragma unroll
                    for (int g = 0; g < qsa_verify_rows::QSA_SCORE_G; ++g) {
                        const unsigned int group = first + g;
                        page[g] =
                            group < max_vis ? __ldg(block_table + group / groups_per_page) : -1;
                    }
                }
                float k[qsa_verify_rows::QSA_SCORE_G][qsa_verify_rows::INDEX_DIM / 32];
#pragma unroll
                for (int g = 0; g < qsa_verify_rows::QSA_SCORE_G; ++g) {
                    const unsigned int group = first + g;
                    const unsigned long long key_base =
                        ((unsigned long long)(page[g] < 0 ? 0 : page[g]) * groups_per_page +
                         group % groups_per_page) *
                        qsa_verify_rows::INDEX_DIM;
#pragma unroll
                    for (int j = 0; j < qsa_verify_rows::INDEX_DIM / 32; ++j) {
                        k[g][j] = (group < max_vis && page[g] >= 0)
                                      ? __bfloat162float(
                                            __ldg(compressed_cache + key_base + lane + 32 * j))
                                      : 0.0f;
                    }
                }
                for (unsigned int r = 0; r < num_rows; ++r) {
                    const unsigned int vis_r =
                        __ldg(sequence_lengths + r) / qsa_verify_rows::COMPRESS_RATIO;
                    if (first < vis_r) {
                        const __nv_bfloat16* q_row = s_q[r];
                        float score[qsa_verify_rows::QSA_SCORE_G] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                        for (int h = 0; h < qsa_verify_rows::INDEX_HEADS; ++h) {
                            float qh[qsa_verify_rows::INDEX_DIM / 32];
#pragma unroll
                            for (int j = 0; j < qsa_verify_rows::INDEX_DIM / 32; ++j)
                                qh[j] = __bfloat162float(
                                    q_row[h * qsa_verify_rows::INDEX_DIM + lane + 32 * j]);
#pragma unroll
                            for (int g = 0; g < qsa_verify_rows::QSA_SCORE_G; ++g) {
                                float partial = 0.0f;
#pragma unroll
                                for (int j = 0; j < qsa_verify_rows::INDEX_DIM / 32; ++j)
                                    partial += qh[j] * k[g][j];
                                partial = qsa_verify_rows::warp_sum(partial);
                                if (lane == 0) score[g] += fmaxf(partial, 0.0f);
                            }
                        }
                        if (lane == 0) {
                            float* row_logits = logits + (unsigned long long)r * logits_row_stride;
#pragma unroll
                            for (int g = 0; g < qsa_verify_rows::QSA_SCORE_G; ++g) {
                                const unsigned int group = first + g;
                                if (group < vis_r) {
                                    row_logits[group] = (page[g] < 0)
                                        ? -CUDART_INF_F
                                        : score[g] * 0.08838834764831845f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Host multi-row ABI of qwen4_qsa_select_expand_par_v3: grid [1, num_rows, 1], block [1024, 1, 1].
extern "C" __global__ void __launch_bounds__(1024) qwen4_qsa_select_expand_rows_v3(
    const float* __restrict__ logits,
    unsigned long long logits_row_stride,
    int* __restrict__ token_indices,
    unsigned int base_position,
    unsigned int base_sequence_length) {
    if (blockIdx.x == 0u) {
        const unsigned int row = blockIdx.y;
        const float* row_logits = logits + (unsigned long long)row * logits_row_stride;
        int* row_indices = token_indices + (unsigned long long)row * qsa_verify_rows::OUTPUT_WIDTH;
        const unsigned int position = base_position + row;
        const unsigned int sequence_length = base_sequence_length + row;
        const unsigned int visible_groups = sequence_length / qsa_verify_rows::COMPRESS_RATIO;
        qsa_verify_rows::qsa_select_expand_v3(
            row_logits, row_indices, visible_groups, position, sequence_length);
    }
}

// Host multi-row ABI of qwen4_qsa_select_expand_par: grid [1, num_rows, 1], block [1024, 1, 1].
extern "C" __global__ void __launch_bounds__(1024) qwen4_qsa_select_expand_rows_par(
    const float* __restrict__ logits,
    unsigned long long logits_row_stride,
    int* __restrict__ token_indices,
    unsigned int base_position,
    unsigned int base_sequence_length) {
    if (blockIdx.x == 0u) {
        const unsigned int row = blockIdx.y;
        const float* row_logits = logits + (unsigned long long)row * logits_row_stride;
        int* row_indices = token_indices + (unsigned long long)row * qsa_verify_rows::OUTPUT_WIDTH;
        const unsigned int position = base_position + row;
        const unsigned int sequence_length = base_sequence_length + row;
        const unsigned int visible_groups = sequence_length / qsa_verify_rows::COMPRESS_RATIO;
        qsa_verify_rows::qsa_select_expand_parallel(
            row_logits, row_indices, visible_groups, position, sequence_length);
    }
}

// Device-metadata multi-row ABI of qwen4_qsa_select_expand_device_meta_par: grid [1, num_rows, 1], block [1024, 1, 1].
extern "C" __global__ void __launch_bounds__(1024) qwen4_qsa_select_expand_device_meta_rows_par(
    const float* __restrict__ logits,
    unsigned long long logits_row_stride,
    int* __restrict__ token_indices,
    const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ sequence_lengths) {
    if (blockIdx.x == 0u) {
        (void)positions;
        const unsigned int row = blockIdx.y;
        const float* row_logits = logits + (unsigned long long)row * logits_row_stride;
        int* row_indices = token_indices + (unsigned long long)row * qsa_verify_rows::OUTPUT_WIDTH;
        const unsigned int sequence_length = sequence_lengths[row];
        if (sequence_length == 0u) {
            for (int i = threadIdx.x; i < qsa_verify_rows::OUTPUT_WIDTH;
                 i += qsa_verify_rows::SELECT_THREADS) {
                row_indices[i] = -1;
            }
        } else {
            const unsigned int position = sequence_length - 1u;
            qsa_verify_rows::qsa_select_expand_parallel(
                row_logits,
                row_indices,
                sequence_length / qsa_verify_rows::COMPRESS_RATIO,
                position,
                sequence_length);
        }
    }
}

// Device-metadata multi-row ABI of qwen4_qsa_select_expand_device_meta_par_v3: grid [1, num_rows, 1], block [1024, 1, 1].
extern "C" __global__ void __launch_bounds__(1024) qwen4_qsa_select_expand_device_meta_rows_par_v3(
    const float* __restrict__ logits,
    unsigned long long logits_row_stride,
    int* __restrict__ token_indices,
    const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ sequence_lengths) {
    if (blockIdx.x == 0u) {
        (void)positions;
        const unsigned int row = blockIdx.y;
        const float* row_logits = logits + (unsigned long long)row * logits_row_stride;
        int* row_indices = token_indices + (unsigned long long)row * qsa_verify_rows::OUTPUT_WIDTH;
        const unsigned int sequence_length = sequence_lengths[row];
        if (sequence_length == 0u) {
            for (int i = threadIdx.x; i < qsa_verify_rows::OUTPUT_WIDTH;
                 i += qsa_verify_rows::SELECT_THREADS) {
                row_indices[i] = -1;
            }
        } else {
            const unsigned int position = sequence_length - 1u;
            qsa_verify_rows::qsa_select_expand_v3(
                row_logits,
                row_indices,
                sequence_length / qsa_verify_rows::COMPRESS_RATIO,
                position,
                sequence_length);
        }
    }
}

// Multi-row BF16 QSA sparse attention partial (v1 counterpart):
// grid [num_query_heads, num_splits, num_rows], block [256, 1, 1].
// Bit-for-bit identical (0 ULP) to launching qwen4_qsa_sparse_attention_bf16_partial per row.
extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_sparse_attention_bf16_partial_rows(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_elems,
    unsigned long long v_block_stride_elems,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits,
    unsigned int query_row_stride,
    unsigned int block_table_row_stride) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split = blockIdx.y;
    const unsigned long long row = blockIdx.z;
    const unsigned int d = threadIdx.x;
    if (q_head < num_query_heads && head_dim == 256u && blockDim.x == 256u) {
        const __nv_bfloat16* row_query = query + row * query_row_stride;
        const int* row_indices = token_indices + row * qsa_verify_rows::OUTPUT_WIDTH;
        const int* row_bt = block_table + row * block_table_row_stride;
        const unsigned int group_size = num_query_heads / num_kv_heads;
        const unsigned int kv_head = q_head / group_size;
        const float qv = __bfloat162float(row_query[(unsigned long long)q_head * head_dim + d]);
        float acc = 0.0f;
        float row_max = -CUDART_INF_F;
        float row_sum = 0.0f;
        __shared__ float reduce[256];
        __shared__ float alpha_shared;
        __shared__ float beta_shared;

        for (int rank = (int)split; rank < qsa_verify_rows::OUTPUT_WIDTH; rank += (int)num_splits) {
            const int logical_token = row_indices[rank];
            if (logical_token < 0) break;
            const unsigned int logical_page = (unsigned int)logical_token / block_size;
            const unsigned int token_in_page = (unsigned int)logical_token % block_size;
            const int physical_page = row_bt[logical_page];
            if (physical_page < 0) continue;
            const unsigned long long k_index =
                (unsigned long long)physical_page * k_block_stride_elems +
                ((unsigned long long)token_in_page * num_kv_heads + kv_head) * head_dim + d;
            reduce[d] = qv * __bfloat162float(k_cache[k_index]);
            __syncthreads();
            for (unsigned int stride = head_dim >> 1; stride > 0; stride >>= 1) {
                if (d < stride) reduce[d] += reduce[d + stride];
                __syncthreads();
            }
            if (d == 0) {
                const float score = reduce[0] * softmax_scale;
                const float next_max = fmaxf(row_max, score);
                const float alpha = row_max == -CUDART_INF_F ? 0.0f : __expf(row_max - next_max);
                const float beta = __expf(score - next_max);
                row_sum = row_sum * alpha + beta;
                row_max = next_max;
                alpha_shared = alpha;
                beta_shared = beta;
            }
            __syncthreads();
            const unsigned long long v_index =
                (unsigned long long)physical_page * v_block_stride_elems +
                ((unsigned long long)token_in_page * num_kv_heads + kv_head) * head_dim + d;
            acc = acc * alpha_shared + beta_shared * __bfloat162float(v_cache[v_index]);
            __syncthreads();
        }
        const unsigned long long part =
            (row * num_query_heads + (unsigned long long)q_head) * num_splits + split;
        partial_output[part * head_dim + d] = acc;
        if (d == 0) {
            partial_max[part] = row_max;
            partial_sum[part] = row_sum;
        }
    }
}

// Multi-row BF16 QSA sparse attention partial v2 (shadowed at runtime by common/qwen4_qsa.cu;
// kept in sync with SM90+ PDL griddepcontrol for test_qwen4_qsa_k5_transaction_static.py):
// grid [num_query_heads, num_splits, num_rows], block [256, 1, 1] (num_splits >= 8).
// Bit-for-bit identical (0 ULP) to launching qwen4_qsa_sparse_attention_bf16_partial_v2 per row,
// with 4,880 B static shared memory (4 CTAs/SM occupancy on GB10).
extern "C" __global__ void __launch_bounds__(256, 4) qwen4_qsa_sparse_attention_bf16_partial_rows_v2(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_elems,
    unsigned long long v_block_stride_elems,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits,
    unsigned int query_row_stride,
    unsigned int block_table_row_stride) {
    constexpr int QSA_V2_MAX_TOKENS = (qsa_verify_rows::OUTPUT_WIDTH + 7) / 8 + 8;
    const unsigned int q_head = blockIdx.x;
    const unsigned int split = blockIdx.y;
    const unsigned long long row = blockIdx.z;
    const unsigned int d = threadIdx.x;
    const unsigned int lane = d & 31u;
    const unsigned int warp = d >> 5;
    if (q_head < num_query_heads && head_dim == 256u && blockDim.x == 256u && num_splits >= 8u) {
        const __nv_bfloat16* row_query = query + row * query_row_stride;
        const int* row_indices = token_indices + row * qsa_verify_rows::OUTPUT_WIDTH;
        const int* row_bt = block_table + row * block_table_row_stride;
        const unsigned int group_size = num_query_heads / num_kv_heads;
        const unsigned int kv_head = q_head / group_size;

        __shared__ float s_q[256];
        __shared__ unsigned int s_phys[QSA_V2_MAX_TOKENS];
        __shared__ unsigned short s_tip[QSA_V2_MAX_TOKENS];
        __shared__ float s_score[QSA_V2_MAX_TOKENS];
        __shared__ float s_beta[QSA_V2_MAX_TOKENS];
        float* const s_alpha = s_score;
        __shared__ unsigned int s_warp_end[8], s_warp_keep[8], s_warp_off[8];
        __shared__ int s_n, s_done;
        __shared__ float s_wmax[8];
        __shared__ float s_row_max, s_row_sum;

        s_q[d] = __bfloat162float(row_query[(unsigned long long)q_head * head_dim + d]);
        if (d == 0) {
            s_n = 0;
            s_done = 0;
        }
        __syncthreads();

        const bool bs_pow2 = block_size != 0u && (block_size & (block_size - 1u)) == 0u;
        const unsigned int bs_shift =
            bs_pow2 ? static_cast<unsigned int>(__ffs(block_size) - 1) : 0u;
        const unsigned int bs_mask = block_size - 1u;

        // Partition every rank exactly once across splits (generalizing split + warp * num_splits
        // across all 32 lanes of each warp in rank order).
        for (unsigned int round = 0;; ++round) {
            const unsigned int idx = round * 256u + warp * 32u + lane;
            const unsigned long long rank =
                (unsigned long long)split + (unsigned long long)idx * num_splits;
            const bool in_range = rank < (unsigned long long)qsa_verify_rows::OUTPUT_WIDTH;
            const int tok = in_range ? __ldg(row_indices + rank) : -1;
            const unsigned int end_mask = __ballot_sync(0xffffffffu, !in_range || tok < 0);
            const unsigned int live = end_mask ? (__ffs(end_mask) - 1) : 32u;
            bool keep = false;
            unsigned int phys_page = 0u;
            unsigned short tip = 0u;
            if (lane < live) {
                const unsigned int utok = static_cast<unsigned int>(tok);
                const unsigned int logical_page =
                    bs_pow2 ? (utok >> bs_shift) : (utok / block_size);
                const unsigned int token_in_page =
                    bs_pow2 ? (utok & bs_mask) : (utok % block_size);
                const int physical_page = __ldg(row_bt + logical_page);
                if (physical_page >= 0) {
                    keep = true;
                    phys_page = static_cast<unsigned int>(physical_page);
                    tip = static_cast<unsigned short>(token_in_page);
                }
            }
            const unsigned int keep_mask = __ballot_sync(0xffffffffu, keep);
            if (lane == 0) {
                s_warp_end[warp] = live;
                s_warp_keep[warp] = __popc(keep_mask);
            }
            __syncthreads();
            if (d == 0) {
                int off = s_n;
                bool ended = false;
                for (int w = 0; w < 8; ++w) {
                    s_warp_off[w] = ended ? 0xffffffffu : (unsigned int)off;
                    if (!ended) {
                        off += (int)s_warp_keep[w];
                        if (s_warp_end[w] < 32u) ended = true;
                    }
                }
                s_n = off;
                s_done = ended ? 1 : 0;
            }
            __syncthreads();
            const unsigned int base_off = s_warp_off[warp];
            if (keep && base_off != 0xffffffffu) {
                const unsigned int slot = base_off + __popc(keep_mask & ((1u << lane) - 1u));
                if (slot < (unsigned int)QSA_V2_MAX_TOKENS) {
                    s_phys[slot] = phys_page;
                    s_tip[slot] = tip;
                }
            }
            const int done = s_done;
            __syncthreads();
            if (done) break;
        }
        const int n = s_n < QSA_V2_MAX_TOKENS ? s_n : QSA_V2_MAX_TOKENS;

        for (int t0 = (int)warp * 4; t0 < n; t0 += 32) {
            float p[4][8];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int t = t0 + u;
                const unsigned long long kb =
                    t < n ? ((unsigned long long)s_phys[t] * k_block_stride_elems +
                             ((unsigned long long)s_tip[t] * num_kv_heads + kv_head) * head_dim)
                          : 0ull;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned int dd = lane + 32u * j;
                    p[u][j] = t < n ? s_q[dd] * __bfloat162float(__ldg(k_cache + kb + dd)) : 0.0f;
                }
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
#pragma unroll
                for (int j = 0; j < 4; ++j) p[u][j] += p[u][j + 4];
#pragma unroll
                for (int j = 0; j < 2; ++j) p[u][j] += p[u][j + 2];
                p[u][0] += p[u][1];
#pragma unroll
                for (int s = 16; s > 0; s >>= 1) {
                    const float other = __shfl_down_sync(0xffffffffu, p[u][0], s);
                    if (lane < (unsigned int)s) p[u][0] += other;
                }
                if (lane == 0 && t0 + u < n) s_score[t0 + u] = p[u][0] * softmax_scale;
            }
        }
        __syncthreads();

        float run = -CUDART_INF_F;
        for (int c0 = 0; c0 < n; c0 += 256) {
            const int t = c0 + (int)d;
            const float sc = t < n ? s_score[t] : -CUDART_INF_F;
            float m = sc;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const float o = __shfl_up_sync(0xffffffffu, m, off);
                if (lane >= (unsigned int)off) m = fmaxf(o, m);
            }
            if (lane == 31) s_wmax[warp] = m;
            __syncthreads();
            float pre = run;
            for (unsigned int w = 0; w < warp; ++w) pre = fmaxf(pre, s_wmax[w]);
            const float next_max = fmaxf(pre, m);
            const float prev_lane = __shfl_up_sync(0xffffffffu, m, 1);
            const float prev_max = lane == 0 ? pre : fmaxf(pre, prev_lane);
            if (t < n) {
                s_alpha[t] = prev_max == -CUDART_INF_F ? 0.0f : __expf(prev_max - next_max);
                s_beta[t] = __expf(sc - next_max);
            }
            for (unsigned int w = 0; w < 8; ++w) run = fmaxf(run, s_wmax[w]);
            __syncthreads();
        }
        if (d == 0) {
            float row_sum = 0.0f;
            for (int t = 0; t < n; ++t) row_sum = row_sum * s_alpha[t] + s_beta[t];
            s_row_max = run;
            s_row_sum = row_sum;
        }
        __syncthreads();

        float acc = 0.0f;
        int t = 0;
        for (; t + 16 <= n; t += 16) {
            float v[16];
#pragma unroll
            for (int u = 0; u < 16; ++u) {
                const unsigned long long vb =
                    (unsigned long long)s_phys[t + u] * v_block_stride_elems +
                    ((unsigned long long)s_tip[t + u] * num_kv_heads + kv_head) * head_dim;
                v[u] = __bfloat162float(__ldg(v_cache + vb + d));
            }
#pragma unroll
            for (int u = 0; u < 16; ++u) acc = acc * s_alpha[t + u] + s_beta[t + u] * v[u];
        }
        for (; t < n; ++t) {
            const unsigned long long vb =
                (unsigned long long)s_phys[t] * v_block_stride_elems +
                ((unsigned long long)s_tip[t] * num_kv_heads + kv_head) * head_dim;
            acc = acc * s_alpha[t] + s_beta[t] * __bfloat162float(__ldg(v_cache + vb + d));
        }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
        const unsigned long long part =
            (row * num_query_heads + (unsigned long long)q_head) * num_splits + split;
        partial_output[part * head_dim + d] = acc;
        if (d == 0) {
            partial_max[part] = s_row_max;
            partial_sum[part] = s_row_sum;
        }
    }
}

// Multi-row BF16 QSA sparse attention split-K reduction (shadowed at runtime by
// common/qwen4_qsa.cu; kept in sync with SM90+ PDL griddepcontrol for
// test_qwen4_qsa_k5_transaction_static.py):
// grid [num_query_heads, num_rows, 1], block [256, 1, 1].
// Bit-for-bit identical (0 ULP) to launching qwen4_qsa_sparse_attention_bf16_reduce per row.
extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_sparse_attention_bf16_reduce_rows(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_query_heads,
    unsigned int head_dim,
    unsigned int num_splits) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int d = threadIdx.x;
    if (q_head < num_query_heads && d < head_dim) {
        const unsigned long long base =
            ((unsigned long long)row * num_query_heads + q_head) * num_splits;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
        float global_max = -CUDART_INF_F;
        for (unsigned int split = 0; split < num_splits; ++split)
            global_max = fmaxf(global_max, partial_max[base + split]);
        float denom = 0.0f;
        float acc = 0.0f;
        if (global_max > -CUDART_INF_F) {
            for (unsigned int split = 0; split < num_splits; ++split) {
                const float scale = __expf(partial_max[base + split] - global_max);
                denom += partial_sum[base + split] * scale;
                acc += partial_output[(base + split) * head_dim + d] * scale;
            }
        }
        output[((unsigned long long)row * num_query_heads + q_head) * head_dim + d] =
            __float2bfloat16_rn(denom > 0.0f ? acc / denom : 0.0f);
    }
}

namespace {

constexpr unsigned int kTrowsT4 = 4u;
constexpr unsigned int kMT4 = kTrowsT4 * static_cast<unsigned int>(kGroup);  // 48 live Q rows
constexpr unsigned int kThreadsT4 = 192u;                                    // 3 warp-pairs = 6 warps
constexpr unsigned int kCapT4 = (kTrowsT4 * 513u + 3u) / 4u * 4u;            // 2052
constexpr unsigned int kBc64 = 64u;
constexpr unsigned int kBlocksPerTile64 = kBc64 / kRatio;                    // 16
constexpr unsigned int kTile8_64 = kBc64 * static_cast<unsigned int>(kHd);   // 16,384 B

__device__ __forceinline__ unsigned int entry_need_t4(unsigned int e, unsigned int row0_pos) {
    const unsigned int blk = e & kBlockMask;
    unsigned int need = (e >> kBlockBits) & ((1u << kTrowsT4) - 1u);
    const unsigned int s = blk * kRatio + 2u - row0_pos;
    if (s < kTrowsT4 + 2u) {
        need |= ((7u << s) >> 2) & ((1u << kTrowsT4) - 1u);
    }
    return need;
}

__device__ __forceinline__ RowInfo row_info_t4(unsigned int m, unsigned int r0, unsigned int rows,
                                               unsigned int first_pos) {
    RowInfo info;
    const unsigned int rt = m / static_cast<unsigned int>(kGroup);
    info.live = (m < kMT4) && (r0 + rt < rows);
    info.bit = rt;
    info.pos = first_pos + r0 + rt;
    info.tail_lo = ((info.pos + 1u) / kRatio) * kRatio;
    return info;
}

__device__ __forceinline__ void warp_pair_barrier_64_t4(unsigned int w_pair) {
    if (w_pair == 0u) {
        asm volatile("bar.sync 1, 64;" ::: "memory");
    } else if (w_pair == 1u) {
        asm volatile("bar.sync 2, 64;" ::: "memory");
    } else {
        asm volatile("bar.sync 3, 64;" ::: "memory");
    }
}

// 192-thread E4M3 K[64, 256] (`16,384 B` = 1024 x 16B) + 16-entry `s_ent` async loader.
__device__ __forceinline__ void load_k_tile_kbc64_192(
    unsigned char* ks, unsigned int* ent_dst, const uint8_t* kc8, const unsigned int* gent,
    unsigned int base, unsigned int count, const int* bt, unsigned int page_shift,
    unsigned int page_mask, unsigned long long k_page_stride, unsigned int k_tok_stride,
    unsigned int kvh, unsigned int tid) {
    if (tid < kBlocksPerTile64) {
        ent_dst[tid] = (base + tid < count) ? __ldg(gent + base + tid) : 0u;
    }
    if (base + kBlocksPerTile64 <= count) {
#pragma unroll
        for (unsigned int c = tid; c < kBc64 * 16u; c += kThreadsT4) {
            const unsigned int j = c >> 4;
            const unsigned int ch = c & 15u;
            const unsigned int slot = base + (j >> 2);
            const unsigned int tok = (__ldg(gent + slot) & kBlockMask) * kRatio + (j & 3u);
            const unsigned int page = static_cast<unsigned int>(__ldg(bt + (tok >> page_shift)));
            const size_t in_page = tok & page_mask;
            const size_t k_off =
                page * k_page_stride + in_page * k_tok_stride + kvh * kHd + ch * 16u;
            cp_async16_l2_128b(ks + swz8(j, ch), kc8 + k_off);
        }
    } else {
#pragma unroll
        for (unsigned int c = tid; c < kBc64 * 16u; c += kThreadsT4) {
            const unsigned int j = c >> 4;
            const unsigned int ch = c & 15u;
            const unsigned int slot = base + (j >> 2);
            if (slot < count) {
                const unsigned int tok = (__ldg(gent + slot) & kBlockMask) * kRatio + (j & 3u);
                const unsigned int page =
                    static_cast<unsigned int>(__ldg(bt + (tok >> page_shift)));
                const size_t in_page = tok & page_mask;
                const size_t k_off =
                    page * k_page_stride + in_page * k_tok_stride + kvh * kHd + ch * 16u;
                cp_async16_l2_128b(ks + swz8(j, ch), kc8 + k_off);
            } else {
                *reinterpret_cast<uint4*>(ks + swz8(j, ch)) = make_uint4(0, 0, 0, 0);
            }
        }
    }
}

// 192-thread E4M3 V[64, 256] (`16,384 B`) async loader reading cached `ent_s[0..15]` from SMEM.
__device__ __forceinline__ void load_v_tile_kbc64_192(
    unsigned char* vs, const unsigned int* ent_s, const uint8_t* vc, unsigned int base,
    unsigned int count, const int* bt, unsigned int page_shift, unsigned int page_mask,
    unsigned long long v_page_stride, unsigned int v_tok_stride, unsigned int kvh,
    unsigned int tid) {
    if (base + kBlocksPerTile64 <= count) {
#pragma unroll
        for (unsigned int c = tid; c < kBc64 * 16u; c += kThreadsT4) {
            const unsigned int j = c >> 4;
            const unsigned int ch = c & 15u;
            const unsigned int tok = (ent_s[j >> 2] & kBlockMask) * kRatio + (j & 3u);
            const unsigned int page = static_cast<unsigned int>(__ldg(bt + (tok >> page_shift)));
            const size_t in_page = tok & page_mask;
            const size_t v_off =
                page * v_page_stride + in_page * v_tok_stride + kvh * kHd + ch * 16u;
            cp_async16_l2_128b(vs + swz8(j, ch), vc + v_off);
        }
    } else {
#pragma unroll
        for (unsigned int c = tid; c < kBc64 * 16u; c += kThreadsT4) {
            const unsigned int j = c >> 4;
            const unsigned int ch = c & 15u;
            const unsigned int slot = base + (j >> 2);
            if (slot < count) {
                const unsigned int tok = (ent_s[j >> 2] & kBlockMask) * kRatio + (j & 3u);
                const unsigned int page =
                    static_cast<unsigned int>(__ldg(bt + (tok >> page_shift)));
                const size_t in_page = tok & page_mask;
                const size_t v_off =
                    page * v_page_stride + in_page * v_tok_stride + kvh * kHd + ch * 16u;
                cp_async16_l2_128b(vs + swz8(j, ch), vc + v_off);
            } else {
                *reinterpret_cast<uint4*>(vs + swz8(j, ch)) = make_uint4(0, 0, 0, 0);
            }
        }
    }
}

__device__ __forceinline__ void pv_half_kbc64(
    float (&o)[16][4], const unsigned int (&a)[4], const unsigned char* vs_half,
    unsigned int vs_row, unsigned int c_base) {
    unsigned int r_cur[4], r_next[4];
    ldsm_x4_t(r_cur, vs_half + swz8(vs_row, c_base));
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        if (c < 7) {
            ldsm_x4_t(r_next, vs_half + swz8(vs_row, c_base + c + 1u));
        }
        mma_e4m3(o[2 * c], a, prmt(r_cur[0], r_cur[1], 0x6420u),
                 prmt(r_cur[2], r_cur[3], 0x6420u));
        mma_e4m3(o[2 * c + 1], a, prmt(r_cur[0], r_cur[1], 0x7531u),
                 prmt(r_cur[2], r_cur[3], 0x7531u));
        if (c < 7) {
#pragma unroll
            for (int k = 0; k < 4; ++k) r_cur[k] = r_next[k];
        }
    }
}

}  // namespace

// 4-warp (`block=128`) token-aligned `TILE_ROWS = 4` fused plan + descending need-bin sort.
// Dynamic shared memory = `(16 + min(cap, 2052)) * 4 B` (`8,272 B` at `cap = 2052`, 3x less
// SMEM than `T=10` and 16-bin warp-shuffle prefix sum, placing newest blocks first and sink last).
extern "C" __global__ void __launch_bounds__(128)
    qwen4_qsa_prefill_plan_fused_sort_desc_t4_b128(
        const unsigned int* __restrict__ masks,  // [rows][mask_words]
        unsigned int* __restrict__ entries,      // [tiles][cap]
        unsigned int* __restrict__ counts,       // [tiles]
        unsigned int rows,
        unsigned int first_pos,
        unsigned int mask_words,
        unsigned int tile_rows,
        unsigned int cap,
        unsigned int tiles) {
    extern __shared__ unsigned int sm[];
    __shared__ unsigned int warp_tot[4];
    unsigned int* const bins = sm;
    unsigned int* const vals = sm + 16;
    const unsigned int tile = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    (void)tiles;
    const unsigned int tr = min(tile_rows, kTrowsT4);
    const unsigned int r0 = tile * tr;
    if (r0 >= rows) return /*early*/;
    const unsigned int r1 = min(r0 + tr, rows);
    const unsigned int row0_pos = first_pos + r0;
    const unsigned int last_pos = first_pos + r1 - 1u;
    const unsigned int words = min(mask_words, ((last_pos + 1u) / kRatio) / 32u + 1u);
    unsigned int* const dst = entries + (size_t)tile * cap;
    const unsigned int smem_cap = min(cap, kCapT4);
    const unsigned int half_cap = smem_cap >> 1;
    const unsigned int first_tail_blk = (row0_pos + 1u) / kRatio;
    const unsigned int tail_w0 = first_tail_blk >> 5;
    const unsigned int row_mask = (1u << tr) - 1u;
    const unsigned int lt = (1u << lane) - 1u;

    const auto fast_need_of = [&](unsigned int e) -> unsigned int {
        const unsigned int blk = e & kBlockMask;
        unsigned int need = (e >> kBlockBits) & row_mask;
        const unsigned int s = blk * kRatio + 2u - row0_pos;
        if (s < tr + 2u) {
            need |= ((7u << s) >> 2) & row_mask;
        }
        return need;
    };

    if (tid < 16u) bins[tid] = 0u;
    __syncthreads();

    unsigned int total = 0u;
    for (unsigned int w0 = 0u; w0 < words; w0 += 128u) {
        const unsigned int w = w0 + tid;
        unsigned int rw[4];
        unsigned int uni = 0u;
        if (w < words) {
#pragma unroll
            for (unsigned int i = 0u; i < 4u; ++i) {
                rw[i] = (r0 + i < r1) ? __ldg(masks + (size_t)(r0 + i) * mask_words + w) : 0u;
                uni |= rw[i];
            }
            if (w >= tail_w0) {
#pragma unroll
                for (unsigned int i = 0u; i < 4u; ++i) {
                    if (r0 + i < r1) {
                        const unsigned int p1 = row0_pos + i + 1u;
                        const unsigned int tb = p1 / kRatio;
                        if ((p1 & (kRatio - 1u)) != 0u && (tb >> 5) == w) {
                            uni |= 1u << (tb & 31u);
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (unsigned int i = 0u; i < 4u; ++i) rw[i] = 0u;
        }
        const unsigned int c = __popc(uni);
        unsigned int incl = c;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const unsigned int v = __shfl_up_sync(0xffffffffu, incl, off);
            if (lane >= static_cast<unsigned int>(off)) incl += v;
        }
        if (lane == 31u) warp_tot[warp] = incl;
        __syncthreads();
        const unsigned int chunk_tot = warp_tot[0] + warp_tot[1] + warp_tot[2] + warp_tot[3];
        if (chunk_tot == 0u) {
            __syncthreads();
            continue;
        }
        unsigned int warp_base = 0u;
        if (warp >= 1u) warp_base += warp_tot[0];
        if (warp >= 2u) warp_base += warp_tot[1];
        if (warp >= 3u) warp_base += warp_tot[2];
        unsigned int pos = total + warp_base + incl - c;
        unsigned int bits = uni;
        while (bits) {
            const unsigned int j = __ffs(bits) - 1u;
            bits &= bits - 1u;
            unsigned int member = 0u;
#pragma unroll
            for (unsigned int i = 0u; i < 4u; ++i) member |= ((rw[i] >> j) & 1u) << i;
            if (pos < smem_cap) {
                const unsigned int e = (member << kBlockBits) | ((w * 32u + j) & kBlockMask);
                vals[pos] = e;
                const unsigned int k = fast_need_of(e);
                const unsigned int active = __activemask();
                const unsigned int peers = __match_any_sync(active, k);
                if ((peers & lt) == 0u) {
                    atomicAdd(&bins[k], static_cast<unsigned int>(__popc(peers)));
                }
            }
            ++pos;
        }
        total += chunk_tot;
        __syncthreads();
    }
    const unsigned int count = min(total, smem_cap);
    if (tid == 0u) counts[tile] = count;

    if (warp == 0u) {
        const unsigned int c = (lane < 16u) ? bins[lane] : 0u;
        unsigned int incl = c;
#pragma unroll
        for (int off = 1; off < 16; off <<= 1) {
            const unsigned int v = __shfl_up_sync(0xffffffffu, incl, off);
            if (lane >= static_cast<unsigned int>(off)) incl += v;
        }
        if (lane < 16u) bins[lane] = incl - c;
    }
    __syncthreads();

    unsigned int* const scatter_dst = (count <= half_cap) ? (vals + half_cap) : dst;
    for (unsigned int i0 = 0u; i0 < count; i0 += 128u) {
        const unsigned int j = i0 + tid;
        const bool live = j < count;
        const unsigned int e = live ? vals[count - 1u - j] : 0u;
        const unsigned int key = live ? fast_need_of(e) : 0xffffffffu;
        const unsigned int peers = __match_any_sync(0xffffffffu, key);
        const unsigned int rank = __popc(peers & lt);
        const unsigned int cnt = __popc(peers);
        unsigned int base_off = 0u;
#pragma unroll
        for (unsigned int w = 0u; w < 4u; ++w) {
            if (warp == w && live && rank == 0u) {
                base_off = bins[key];
                bins[key] = base_off + cnt;
            }
            __syncthreads();
        }
        base_off = __shfl_sync(peers, base_off, __ffs(peers) - 1u);
        if (live) scatter_dst[base_off + rank] = e;
    }
    if (count <= half_cap) {
        __syncthreads();
        for (unsigned int i = tid; i < count; i += 128u) {
            dst[i] = scatter_dst[i];
        }
    }
}

// SM120 `TILE_ROWS = 4`, `kBc = 64` (16 QSA blocks/tile) 192-thread E4M3 QK+PV attention core:
//   - Grid: `(ceil(rows / 4), KV_HEADS, 1)`, Block: `(192, 1, 1)` = 3 warp-pairs (`48/48 = 100%`
//     M-tile Tensor Core fill: `4` tokens x `12` Q heads = exact `3` 16-row MMA warp-pairs!).
//   - At 192 threads & 2 CTAs/SM (384 threads/SM), register budget is 168 regs/thread: ALL 256
//     columns of Q (`qf[8][4]`, 32 regs) stay permanently resident in registers with ZERO Q SMEM
//     reloads per KV tile and double-buffered `b_cur`/`b_next` `ldsm_x4` pipelining!
//   - Alternating K[64,256] (`16,384 B`) / V[64,256] (`16,384 B`) `cp.async` overlap fits in
//     `36,352 B <= 49,152 B` dynamic SMEM (`2 CTAs/SM`, 4 barriers), halving warp-pair barriers
//     (`4 -> 2`) and softmax rescales (`2 -> 1`) per 64 KV tokens with half-tile skipping.
extern "C" __global__ void __launch_bounds__(kThreadsT4, 2)
    qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_kbc64_t4(
        const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc8,
        const uint8_t* __restrict__ vc, const int* __restrict__ bt,
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
        uint8_t* __restrict__ out_e4m3, const __nv_bfloat16* __restrict__ gate,
        unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
        unsigned int gate_stride, unsigned int page_size, unsigned long long k_page_stride,
        unsigned int k_tok_stride, unsigned long long v_page_stride, unsigned int v_tok_stride,
        float scale_log2_h0, float scale_log2_h1, float out_scale_h0, float out_scale_h1,
        float* __restrict__ a_scale_out, float a_scale_val, unsigned int cap) {
    constexpr unsigned int kSMxBytes = 3u * 2u * 8u * 8u;   // 384 B (float2[3][2][8])
    constexpr unsigned int kSABytes = 3u * 2u * 32u * 16u;  // 3,072 B (uint4[3][2][32])

    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned char* const buf_k = smem_raw;
    unsigned char* const buf_v = smem_raw + kTile8_64;
    float2* const s_mx = reinterpret_cast<float2*>(buf_v + kTile8_64);
    uint4* const s_a = reinterpret_cast<uint4*>(buf_v + kTile8_64 + kSMxBytes);
    unsigned int* const s_ent =
        reinterpret_cast<unsigned int*>(buf_v + kTile8_64 + kSMxBytes + kSABytes);

    const unsigned int tile = blockIdx.x;
    const unsigned int kvh = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_local = tid >> 5;
    const unsigned int w_pair = warp_local >> 1;  // 0..2
    const unsigned int p_role = warp_local & 1u;  // 0..1
    const unsigned int lane = tid & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const unsigned int r0 = tile * kTrowsT4;
    if (r0 >= rows) return /*early*/;
    if (tile == 0u && kvh == 0u && tid == 0u && a_scale_out != nullptr) {
        a_scale_out[0] = a_scale_val;
    }

    const unsigned int eff_cap = (cap > 0u) ? cap : kCapT4;
    const unsigned int count = min(counts[tile], eff_cap);
    const unsigned int* const gent = entries + (size_t)tile * eff_cap;
    const unsigned int page_shift = static_cast<unsigned int>(__ffs(page_size) - 1);
    const unsigned int page_mask = page_size - 1u;

    // Stage all 48 Q rows (12,288 B <= 16,384 B `buf_k`) once, then load ALL 256 cols into `qf[8][4]`.
    for (unsigned int c = tid; c < kMT4 * 16u; c += kThreadsT4) {
        const unsigned int m = c >> 4;
        const unsigned int rt = m / static_cast<unsigned int>(kGroup);
        const unsigned int ch = c & 15u;
        unsigned char* const dst = buf_k + swz8(m, ch);
        if (r0 + rt < rows) {
            const unsigned int head = kvh * kGroup + (m % static_cast<unsigned int>(kGroup));
            cp_async16_l2_128b(dst, q8 + (size_t)(r0 + rt) * q_stride + head * kHd + ch * 16u);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    unsigned int qf[8][4];
    {
        const unsigned int q_row = w_pair * 16u + (lane & 7u) + ((lane >> 3) & 1u) * 8u;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            ldsm_x4(qf[ks], buf_k + swz8(q_row, ks * 2u + (lane >> 4)));
        }
    }
    __syncthreads();

    unsigned int warp_rows = 0u;
    {
        const unsigned int m_lo = w_pair * 16u;
        const unsigned int m_hi = m_lo + 15u;
        for (unsigned int rt = m_lo / static_cast<unsigned int>(kGroup);
             rt <= m_hi / static_cast<unsigned int>(kGroup); ++rt) {
            warp_rows |= 1u << rt;
        }
    }

    const unsigned int ntiles = (count + kBlocksPerTile64 - 1u) / kBlocksPerTile64;
    const unsigned int row0_pos = first_pos + r0;
    const RowInfo ra = row_info_t4(w_pair * 16u + g, r0, rows, first_pos);
    const RowInfo rb = row_info_t4(w_pair * 16u + g + 8u, r0, rows, first_pos);
    const float scale_log2 = kvh ? scale_log2_h1 : scale_log2_h0;
    const float out_scale = kvh ? out_scale_h1 : out_scale_h0;

    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float ma = -CUDART_INF_F, mb = -CUDART_INF_F, la = 0.0f, lb = 0.0f;

    if (ntiles > 0u) {
        load_k_tile_kbc64_192(buf_k, s_ent, kc8, gent, 0u, count, bt, page_shift, page_mask,
                              k_page_stride, k_tok_stride, kvh, tid);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }

    for (unsigned int cur_it = 0u; cur_it < ntiles; ++cur_it) {
        const unsigned int ebuf = cur_it & 1u;
        const unsigned int base = cur_it * kBlocksPerTile64;
        const unsigned int* const ent_s = s_ent + ebuf * kBlocksPerTile64;

        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();

        // Overlap V[64,256] async copy into `buf_v` with register-resident QK^T + softmax + alpha rescale.
        load_v_tile_kbc64_192(buf_v, ent_s, vc, base, count, bt, page_shift, page_mask,
                              v_page_stride, v_tok_stride, kvh, tid);
        asm volatile("cp.async.commit_group;" ::: "memory");

        bool do_h0 = false;
        bool do_h1 = false;
        {
            const unsigned int need = (lane < kBlocksPerTile64 && base + lane < count)
                                          ? entry_need_t4(ent_s[lane], row0_pos)
                                          : 0u;
            const unsigned int need01 =
                __reduce_or_sync(0xffffffffu, (lane < 8u) ? need : (need << 16));
            do_h0 = (need01 & warp_rows) != 0u;
            do_h1 = ((need01 >> 16) & warp_rows) != 0u;
        }

        if (do_h0 || do_h1) {
            const bool my_do = p_role ? do_h1 : do_h0;
            float s[4][4];
            float mxa = -CUDART_INF_F, mxb = -CUDART_INF_F;
            if (my_do) {
                unsigned int my_k_row0 = (lane >> 4) * 8u + (lane & 7u) + p_role * 32u;
                unsigned int k_col_off = (lane >> 3) & 1u;
                asm volatile("" : "+r"(my_k_row0), "+r"(k_col_off));
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
                }
                // Software-pipelined `b0`/`b1` double-buffering across all 8 K-steps with
                // ALL 256 Q columns in `qf[8][4]` (zero Q SMEM reloads!).
                unsigned int b0[4], b1[4];
                ldsm_x4(b0, buf_k + swz8(my_k_row0, k_col_off));
#pragma unroll
                for (int kstep = 0; kstep < 8; ++kstep) {
                    ldsm_x4(b1, buf_k + swz8(my_k_row0 + 16u, kstep * 2u + k_col_off));
                    mma_e4m3(s[0], qf[kstep], b0[0], b0[1]);
                    mma_e4m3(s[1], qf[kstep], b0[2], b0[3]);
                    if (kstep < 7) {
                        ldsm_x4(b0, buf_k + swz8(my_k_row0, (kstep + 1u) * 2u + k_col_off));
                    }
                    mma_e4m3(s[2], qf[kstep], b1[0], b1[1]);
                    mma_e4m3(s[3], qf[kstep], b1[2], b1[3]);
                }

                const unsigned int nt_base = p_role * 4u;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned int slot_off = (nt_base + j) * 2u + (t >> 1);
                    const bool slot_live = base + slot_off < count;
                    const unsigned int e = slot_live ? ent_s[slot_off] : 0u;
#pragma unroll
                    for (int x = 0; x < 2; ++x) {
                        const unsigned int in_block = (t * 2u + x) & 3u;
                        s[j][x] = allowed(ra, e, slot_live, in_block) ? s[j][x] * scale_log2
                                                                      : -CUDART_INF_F;
                        s[j][2 + x] = allowed(rb, e, slot_live, in_block)
                                          ? s[j][2 + x] * scale_log2
                                          : -CUDART_INF_F;
                        mxa = fmaxf(mxa, s[j][x]);
                        mxb = fmaxf(mxb, s[j][2 + x]);
                    }
                }
                mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
                mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
                mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
                mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
            }

            if (t == 0u) {
                s_mx[(w_pair * 2u + p_role) * 8u + g] = make_float2(mxa, mxb);
            }
            warp_pair_barrier_64_t4(w_pair);
            const float2 peer_mx = s_mx[(w_pair * 2u + (p_role ^ 1u)) * 8u + g];
            mxa = fmaxf(mxa, peer_mx.x);
            mxb = fmaxf(mxb, peer_mx.y);

            const float na = fmaxf(ma, mxa);
            const float nb = fmaxf(mb, mxb);
            const bool rescale_a = (na > ma) && (ma != -CUDART_INF_F);
            const bool rescale_b = (nb > mb) && (mb != -CUDART_INF_F);
            const bool any_rescale_a = __any_sync(0xffffffffu, rescale_a);
            const bool any_rescale_b = __any_sync(0xffffffffu, rescale_b);
            float alpha_a = 1.0f;
            if (any_rescale_a) {
                alpha_a = rescale_a ? exp2f(ma - na) : 1.0f;
            }
            float alpha_b = 1.0f;
            if (any_rescale_b) {
                alpha_b = rescale_b ? exp2f(mb - nb) : 1.0f;
            }
            const float p_log2 = 8.0f;
            const float base_a = (na == -CUDART_INF_F ? 0.0f : na) - p_log2;
            const float base_b = (nb == -CUDART_INF_F ? 0.0f : nb) - p_log2;
            ma = na;
            mb = nb;

            uint4 my_a = make_uint4(0u, 0u, 0u, 0u);
            if (my_do) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    s[j][0] = exp2f(s[j][0] - base_a);
                    s[j][1] = exp2f(s[j][1] - base_a);
                    s[j][2] = exp2f(s[j][2] - base_b);
                    s[j][3] = exp2f(s[j][3] - base_b);
                }
                my_a.x = e4m3x4(s[0][0], s[0][1], s[1][0], s[1][1]);
                my_a.y = e4m3x4(s[0][2], s[0][3], s[1][2], s[1][3]);
                my_a.z = e4m3x4(s[2][0], s[2][1], s[3][0], s[3][1]);
                my_a.w = e4m3x4(s[2][2], s[2][3], s[3][2], s[3][3]);
            }
            s_a[(w_pair * 2u + p_role) * 32u + lane] = my_a;
            warp_pair_barrier_64_t4(w_pair);

            float dl[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (do_h0) {
                const uint4 a0_v = s_a[(w_pair * 2u + 0u) * 32u + lane];
                const unsigned int a0[4] = {a0_v.x, a0_v.y, a0_v.z, a0_v.w};
                mma_e4m3_zc(dl, a0, kE4m3OnesX4, kE4m3OnesX4);
            }
            if (do_h1) {
                const uint4 a1_v = s_a[(w_pair * 2u + 1u) * 32u + lane];
                const unsigned int a1[4] = {a1_v.x, a1_v.y, a1_v.z, a1_v.w};
                if (do_h0) {
                    mma_e4m3(dl, a1, kE4m3OnesX4, kE4m3OnesX4);
                } else {
                    mma_e4m3_zc(dl, a1, kE4m3OnesX4, kE4m3OnesX4);
                }
            }
            la = fmaf(la, alpha_a, dl[0]);
            lb = fmaf(lb, alpha_b, dl[2]);

            if (any_rescale_a) {
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    o[c][0] *= alpha_a;
                    o[c][1] *= alpha_a;
                }
            }
            if (any_rescale_b) {
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    o[c][2] *= alpha_b;
                    o[c][3] *= alpha_b;
                }
            }
        }

        // Wait for `buf_v` async load; all warps finished reading `buf_k` above.
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();

        // Overlap next tile's K[64,256] async copy into `buf_k` with this tile's PV MMAs.
        const unsigned int next_it = cur_it + 1u;
        if (next_it < ntiles) {
            load_k_tile_kbc64_192(buf_k, s_ent + (ebuf ^ 1u) * kBlocksPerTile64, kc8, gent,
                                  next_it * kBlocksPerTile64, count, bt, page_shift, page_mask,
                                  k_page_stride, k_tok_stride, kvh, tid);
            asm volatile("cp.async.commit_group;" ::: "memory");
        }

        unsigned int vs_row = (lane >> 3) * 8u + (lane & 7u);
        unsigned int c_base = p_role * 8u;
        if (do_h0 || do_h1) {
            asm volatile("" : "+r"(vs_row), "+r"(c_base));
        }
        if (do_h0) {
            const uint4 a0_v = s_a[(w_pair * 2u + 0u) * 32u + lane];
            const unsigned int a0[4] = {a0_v.x, a0_v.y, a0_v.z, a0_v.w};
            pv_half_kbc64(o, a0, buf_v, vs_row, c_base);
        }
        if (do_h1) {
            const uint4 a1_v = s_a[(w_pair * 2u + 1u) * 32u + lane];
            const unsigned int a1[4] = {a1_v.x, a1_v.y, a1_v.z, a1_v.w};
            pv_half_kbc64(o, a1, buf_v + 32u * static_cast<unsigned int>(kHd), vs_row, c_base);
        }
    }

    const float ia = la > 0.0f ? out_scale / la : 0.0f;
    const float ib = lb > 0.0f ? out_scale / lb : 0.0f;
    const unsigned int ma_row = w_pair * 16u + g;
    const unsigned int mb_row = ma_row + 8u;
    const unsigned int c_base = p_role * 8u;
    if (ra.live) {
        store_row_pipe_e4m3_half(out_e4m3, gate, r0 + ra.bit, kvh * kGroup + ma_row % kGroup,
                                 o_stride, gate_stride, t, ia, o, 0, c_base);
    }
    if (rb.live) {
        store_row_pipe_e4m3_half(out_e4m3, gate, r0 + rb.bit, kvh * kGroup + mb_row % kGroup,
                                 o_stride, gate_stride, t, ib, o, 2, c_base);
    }
}

