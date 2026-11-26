// SPDX-License-Identifier: AGPL-3.0-only
//
// QSA-sparse prefill attention core for Qwen3.8-Flash-Next
// (ATLAS_QWEN4_PREFILL_QSA). Consumes the tile plan of
// qwen4_qsa_prefill_select.cu.
//
// One CTA = one tile of TROWS consecutive query rows x the 12 query heads of
// one KV head (M = 12 * TROWS rows of the MMA, 16 per warp). The selection is
// shared by every head of a row, so the CTA streams the UNION of its rows'
// selected 4-token blocks (plus their open tail blocks) from the BF16 paged
// cache in 32-token tiles, and masks each MMA row to its own row's set:
//   allowed(row p, token t) = member(block(t), p) || 4*((p+1)/4) <= t <= p.
// Q fragments live in registers; K/V tiles are double buffered with cp.async
// in an XOR-swizzled layout read by ldmatrix. P*V runs in BF16 with fp32
// accumulation and a base-2 online softmax.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace {

constexpr int kHd = 256;
constexpr int kGroup = 12;
constexpr int kBc = 32;
constexpr unsigned int kRatio = 4;
constexpr unsigned int kBlockBits = 22;
constexpr unsigned int kBlockMask = (1u << kBlockBits) - 1u;
constexpr int kTileElems = kBc * kHd;

template <int TROWS, int GROUP = kGroup>
struct AttnCfg {
    static constexpr int kM = GROUP * TROWS;
    static constexpr int kWarps = (kM + 15) / 16;
    static constexpr int kThreads = kWarps * 32;
    // Union bound: 512 selected + 1 tail block per row; a multiple of 4 keeps
    // every tile's entry slab 16-byte aligned for cp.async.
    static constexpr int kCap = (TROWS * 513 + 3) / 4 * 4;
};

__device__ __forceinline__ unsigned int swz(unsigned int row, unsigned int col) {
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

__device__ __forceinline__ unsigned int pack_bf16(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const unsigned int*>(&v);
}

struct RowInfo {
    bool live;
    unsigned int bit;        // member bit (row within the tile)
    unsigned int pos;        // absolute query position
    unsigned int tail_lo;    // first token of the open tail group
};

template <int GROUP = kGroup>
__device__ __forceinline__ RowInfo row_info(int m, int kM, unsigned int r0, unsigned int rows,
                                            unsigned int first_pos) {
    RowInfo info;
    const unsigned int rt = static_cast<unsigned int>(m) / GROUP;
    info.live = m < kM && r0 + rt < rows;
    info.bit = rt;
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

// Gather one 32-token K and V tile from the paged BF16 cache. Consecutive
// threads take consecutive 16-byte chunks of a token row, so each warp reads
// one contiguous 512-byte row per instruction. Slots past `count` are zero
// filled so P*V never sees garbage.
template <int THREADS>
__device__ __forceinline__ void load_kv_tile(__nv_bfloat16* ks, __nv_bfloat16* vs,
                                             const __nv_bfloat16* __restrict__ kc,
                                             const __nv_bfloat16* __restrict__ vc,
                                             const unsigned int* ent, unsigned int base,
                                             unsigned int count, const int* __restrict__ bt,
                                             unsigned int page_size,
                                             unsigned long long page_stride,
                                             unsigned int tok_stride, unsigned int head_off,
                                             unsigned int tid) {
    for (unsigned int c = tid; c < kBc * 32; c += THREADS) {
        const unsigned int j = c >> 5;
        const unsigned int ch = c & 31;
        const unsigned int slot = base + (j >> 2);
        __nv_bfloat16* kd = ks + swz(j, ch * 8);
        __nv_bfloat16* vd = vs + swz(j, ch * 8);
        if (slot < count) {
            const unsigned int tok = (ent[slot] & kBlockMask) * kRatio + (j & 3);
            const int page = __ldg(bt + tok / page_size);
            if (page >= 0) {
                const unsigned long long off = (unsigned long long)page * page_stride +
                                               (unsigned long long)(tok % page_size) * tok_stride +
                                               head_off + ch * 8;
                cp_async16(kd, kc + off);
                cp_async16(vd, vc + off);
            } else {
                *reinterpret_cast<uint4*>(kd) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(vd) = make_uint4(0, 0, 0, 0);
            }
        } else {
            *reinterpret_cast<uint4*>(kd) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(vd) = make_uint4(0, 0, 0, 0);
        }
    }
}

// Normalise one thread's 2 x 32 output columns of an MMA row and store them.
// With a gate the result is bf16(bf16(o) * sigmoid(g)), exactly the unfused
// sigmoid_gate_mul_batched applied to the stored core output.
__device__ __forceinline__ void store_row(__nv_bfloat16* __restrict__ out,
                                          const __nv_bfloat16* __restrict__ gate,
                                          unsigned int row, unsigned int head,
                                          unsigned int o_stride, unsigned int gate_stride,
                                          unsigned int t, float inv_l, const float (&o)[32][4],
                                          int half) {
    __nv_bfloat16* dst = out + (size_t)row * o_stride + head * kHd + t * 2;
    const __nv_bfloat16* gsrc = gate ? gate + (size_t)row * gate_stride + head * kHd + t * 2 : nullptr;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        float x0 = __bfloat162float(__float2bfloat16(o[i][half] * inv_l));
        float x1 = __bfloat162float(__float2bfloat16(o[i][half + 1] * inv_l));
        if (gsrc) {
            const __nv_bfloat162 gv = *reinterpret_cast<const __nv_bfloat162*>(gsrc + i * 8);
            x0 *= 1.0f / (1.0f + expf(-__bfloat162float(gv.x)));
            x1 *= 1.0f / (1.0f + expf(-__bfloat162float(gv.y)));
        }
        *reinterpret_cast<unsigned int*>(dst + i * 8) = pack_bf16(x0, x1);
    }
}

// Rows of the tile that need entry `e`: its member bits plus the rows whose
// open tail lies in its block (tail tokens are allowed by position, not bit).
template <int TROWS>
__device__ __forceinline__ unsigned int entry_need(unsigned int e, unsigned int row0_pos) {
    const unsigned int blk = e & kBlockMask;
    unsigned int need = (e >> kBlockBits) & ((1u << TROWS) - 1u);
    // Row rt has open tail tokens in blk iff row0_pos + rt + 1 in [4blk + 1, 4blk + 3].
    const int lo = static_cast<int>(blk * kRatio) - static_cast<int>(row0_pos);
    for (int rt = max(lo, 0); rt < min(lo + 3, TROWS); ++rt) need |= 1u << rt;
    return need;
}

template <int TROWS, bool kSkip, int GROUP = kGroup>
__device__ __forceinline__ void qsa_prefill_attn(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ vc, const int* __restrict__ bt,
    const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
    unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
    unsigned int gate_stride, unsigned int page_size, unsigned long long page_stride,
    unsigned int tok_stride, float scale_log2) {
    using Cfg = AttnCfg<TROWS, GROUP>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* stage = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    unsigned int* ent = reinterpret_cast<unsigned int*>(smem_raw + 4 * kTileElems * 2);

    const unsigned int tile = blockIdx.x;
    const unsigned int kvh = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3;
    const unsigned int r0 = tile * TROWS;
    if (r0 >= rows) return;
    const unsigned int count = min(counts[tile], static_cast<unsigned int>(Cfg::kCap));
    const unsigned int* gent = entries + (size_t)tile * Cfg::kCap;

    // Stage Q (M rows x 256) through the K/V buffers, and the tile plan.
    for (unsigned int c = tid; c < Cfg::kWarps * 16 * 32; c += Cfg::kThreads) {
        const unsigned int m = c >> 5;
        const unsigned int ch = c & 31;
        __nv_bfloat16* dst = stage + swz(m, ch * 8);
        const unsigned int rt = m / GROUP;
        if (m < static_cast<unsigned int>(Cfg::kM) && r0 + rt < rows) {
            const unsigned int head = kvh * GROUP + m % GROUP;
            cp_async16(dst, q + (size_t)(r0 + rt) * q_stride + head * kHd + ch * 8);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
    for (unsigned int i = tid * 4; i < count; i += Cfg::kThreads * 4) {
        if (i + 4 <= count) {
            cp_async16(ent + i, gent + i);
        } else {
            for (unsigned int j = i; j < count; ++j) ent[j] = gent[j];
        }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    unsigned int qf[16][4];
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
        const unsigned int row = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const unsigned int col = ks * 16 + (lane >> 4) * 8;
        ldsm_x4(qf[ks], stage + swz(row, col));
    }
    __syncthreads();

    const RowInfo ra = row_info<GROUP>(warp * 16 + g, Cfg::kM, r0, rows, first_pos);
    const RowInfo rb = row_info<GROUP>(warp * 16 + g + 8, Cfg::kM, r0, rows, first_pos);
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float ma = -CUDART_INF_F, mb = -CUDART_INF_F, la = 0.0f, lb = 0.0f;

    const unsigned int head_off = kvh * kHd;
    const unsigned int ntiles = (count + 7) / 8;
    // Tile rows this warp's 16 MMA rows belong to (kSkip: skip 32-token tiles
    // none of them needs; the cooperative loads are still issued).
    unsigned int warp_rows = 0;
    if (kSkip) {
        const unsigned int m_lo = warp * 16;
        const unsigned int m_hi = min(m_lo + 15, static_cast<unsigned int>(Cfg::kM) - 1);
        for (unsigned int rt = m_lo / GROUP; rt <= m_hi / GROUP; ++rt) warp_rows |= 1u << rt;
    }
    if (ntiles > 0) {
        load_kv_tile<Cfg::kThreads>(stage, stage + kTileElems, kc, vc, ent, 0, count, bt,
                                    page_size, page_stride, tok_stride, head_off, tid);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    for (unsigned int it = 0; it < ntiles; ++it) {
        const unsigned int cur = it & 1u;
        // Tile `it` was issued last iteration. After this barrier every warp is
        // done with tile it-1, so its buffer takes tile it+1: one barrier per
        // tile with a one-tile prefetch distance (3-6% faster than waiting
        // before the barrier and fencing again after compute).
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (it + 1 < ntiles) {
            __nv_bfloat16* ns = stage + (cur ^ 1u) * 2 * kTileElems;
            load_kv_tile<Cfg::kThreads>(ns, ns + kTileElems, kc, vc, ent, (it + 1) * 8, count,
                                        bt, page_size, page_stride, tok_stride, head_off, tid);
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        const __nv_bfloat16* ks_s = stage + cur * 2 * kTileElems;
        const __nv_bfloat16* vs_s = ks_s + kTileElems;
        if (kSkip) {
            const unsigned int slot = it * 8 + (lane & 7);
            const unsigned int need =
                lane < 8 && slot < count ? entry_need<TROWS>(ent[slot], first_pos + r0) : 0u;
            if ((__reduce_or_sync(0xffffffffu, need) & warp_rows) == 0) continue;
        }

        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
#pragma unroll
        for (int kstep = 0; kstep < 16; ++kstep) {
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned int b[4];
                const unsigned int row = (np * 2 + (lane >> 4)) * 8 + (lane & 7);
                const unsigned int col = kstep * 16 + ((lane >> 3) & 1) * 8;
                ldsm_x4(b, ks_s + swz(row, col));
                mma_bf16(s[np * 2], qf[kstep], b[0], b[1]);
                mma_bf16(s[np * 2 + 1], qf[kstep], b[2], b[3]);
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
        const float base_a = na == -CUDART_INF_F ? 0.0f : na;
        const float base_b = nb == -CUDART_INF_F ? 0.0f : nb;
        ma = na;
        mb = nb;
        float suma = 0.0f, sumb = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - base_a);
            s[nt][1] = exp2f(s[nt][1] - base_a);
            s[nt][2] = exp2f(s[nt][2] - base_b);
            s[nt][3] = exp2f(s[nt][3] - base_b);
            suma += s[nt][0] + s[nt][1];
            sumb += s[nt][2] + s[nt][3];
        }
        la = la * alpha_a + suma;
        lb = lb * alpha_b + sumb;
        // Rescale O only when some row max of this warp moved (warp-uniform).
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
                ldsm_x4_t(b, vs_s + swz(row, col));
                mma_bf16(o[dp * 2], a, b[0], b[1]);
                mma_bf16(o[dp * 2 + 1], a, b[2], b[3]);
            }
        }
    }

    la += __shfl_xor_sync(0xffffffffu, la, 1);
    la += __shfl_xor_sync(0xffffffffu, la, 2);
    lb += __shfl_xor_sync(0xffffffffu, lb, 1);
    lb += __shfl_xor_sync(0xffffffffu, lb, 2);
    const float ia = la > 0.0f ? 1.0f / la : 0.0f;
    const float ib = lb > 0.0f ? 1.0f / lb : 0.0f;
    const unsigned int ma_row = warp * 16 + g;
    const unsigned int mb_row = ma_row + 8;
    if (ra.live) {
        store_row(out, gate, r0 + ra.bit, kvh * GROUP + ma_row % GROUP, o_stride, gate_stride,
                  t, ia, o, 0);
    }
    if (rb.live) {
        store_row(out, gate, r0 + rb.bit, kvh * GROUP + mb_row % GROUP, o_stride, gate_stride,
                  t, ib, o, 2);
    }
}

__device__ __forceinline__ float qsa_fp8e4m3_to_f32(__nv_fp8_storage_t value) {
    return __half2float(__nv_cvt_fp8_to_halfraw(value, __NV_E4M3));
}

__device__ __forceinline__ float qsa_decode_e2m1(unsigned int nibble) {
    const unsigned int mag = nibble & 7u;
    const float abs_val = (mag <= 4u)
        ? 0.5f * static_cast<float>(mag)
        : ((mag == 5u) ? 3.0f : ((mag == 6u) ? 4.0f : 6.0f));
    return (nibble & 8u) ? -abs_val : abs_val;
}

__device__ __forceinline__ uint4 unpack_nvfp4x8_to_bf16x8(uint32_t packed4, float scale) {
    uint4 out;
    out.x = pack_bf16(qsa_decode_e2m1(packed4 & 0xfu) * scale,
                      qsa_decode_e2m1((packed4 >> 4) & 0xfu) * scale);
    out.y = pack_bf16(qsa_decode_e2m1((packed4 >> 8) & 0xfu) * scale,
                      qsa_decode_e2m1((packed4 >> 12) & 0xfu) * scale);
    out.z = pack_bf16(qsa_decode_e2m1((packed4 >> 16) & 0xfu) * scale,
                      qsa_decode_e2m1((packed4 >> 20) & 0xfu) * scale);
    out.w = pack_bf16(qsa_decode_e2m1((packed4 >> 24) & 0xfu) * scale,
                      qsa_decode_e2m1((packed4 >> 28) & 0xfu) * scale);
    return out;
}

template <int THREADS>
__device__ __forceinline__ void load_kv_tile_nvfp4(
    __nv_bfloat16* ks,
    __nv_bfloat16* vs,
    const unsigned char* __restrict__ kc,
    const unsigned char* __restrict__ vc,
    const unsigned int* ent,
    unsigned int base,
    unsigned int count,
    const int* __restrict__ bt,
    unsigned int page_size,
    unsigned long long page_stride_bytes,
    unsigned long long data_section_bytes,
    unsigned int tok_stride,
    unsigned int head_off,
    unsigned int tid) {
    for (unsigned int c = tid; c < kBc * 32; c += THREADS) {
        const unsigned int j = c >> 5;
        const unsigned int ch = c & 31;
        const unsigned int slot = base + (j >> 2);
        __nv_bfloat16* kd = ks + swz(j, ch * 8);
        __nv_bfloat16* vd = vs + swz(j, ch * 8);
        if (slot < count) {
            const unsigned int tok = (ent[slot] & kBlockMask) * kRatio + (j & 3);
            const int page = __ldg(bt + tok / page_size);
            if (page >= 0) {
                const unsigned int elem0 = (tok % page_size) * tok_stride + head_off + ch * 8u;
                const unsigned int group_idx = elem0 >> 4;
                const unsigned char* k_blk = kc + (unsigned long long)page * page_stride_bytes;
                const unsigned char* v_blk = vc + (unsigned long long)page * page_stride_bytes;
                const uint32_t kp4 =
                    __ldg(reinterpret_cast<const uint32_t*>(k_blk + (elem0 >> 1)));
                const uint32_t vp4 =
                    __ldg(reinterpret_cast<const uint32_t*>(v_blk + (elem0 >> 1)));
                const float ks_f = qsa_fp8e4m3_to_f32(
                    (__nv_fp8_storage_t)__ldg(k_blk + data_section_bytes + group_idx));
                const float vs_f = qsa_fp8e4m3_to_f32(
                    (__nv_fp8_storage_t)__ldg(v_blk + data_section_bytes + group_idx));
                *reinterpret_cast<uint4*>(kd) = unpack_nvfp4x8_to_bf16x8(kp4, ks_f);
                *reinterpret_cast<uint4*>(vd) = unpack_nvfp4x8_to_bf16x8(vp4, vs_f);
            } else {
                *reinterpret_cast<uint4*>(kd) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(vd) = make_uint4(0, 0, 0, 0);
            }
        } else {
            *reinterpret_cast<uint4*>(kd) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(vd) = make_uint4(0, 0, 0, 0);
        }
    }
}

template <int TROWS, bool kSkip, int GROUP = kGroup>
__device__ __forceinline__ void qsa_prefill_attn_nvfp4(
    const __nv_bfloat16* __restrict__ q,
    const unsigned char* __restrict__ kc,
    const unsigned char* __restrict__ vc,
    const int* __restrict__ bt,
    const unsigned int* __restrict__ entries,
    const unsigned int* __restrict__ counts,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate,
    unsigned int rows,
    unsigned int first_pos,
    unsigned int q_stride,
    unsigned int o_stride,
    unsigned int gate_stride,
    unsigned int page_size,
    unsigned long long page_stride_bytes,
    unsigned long long data_section_bytes,
    unsigned int tok_stride,
    float scale_log2) {
    using Cfg = AttnCfg<TROWS, GROUP>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* stage = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    unsigned int* ent = reinterpret_cast<unsigned int*>(smem_raw + 4 * kTileElems * 2);

    const unsigned int tile = blockIdx.x;
    const unsigned int kvh = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3;
    const unsigned int r0 = tile * TROWS;
    if (r0 >= rows) return;
    const unsigned int count = min(counts[tile], static_cast<unsigned int>(Cfg::kCap));
    const unsigned int* gent = entries + (size_t)tile * Cfg::kCap;

    for (unsigned int c = tid; c < Cfg::kWarps * 16 * 32; c += Cfg::kThreads) {
        const unsigned int m = c >> 5;
        const unsigned int ch = c & 31;
        __nv_bfloat16* dst = stage + swz(m, ch * 8);
        const unsigned int rt = m / GROUP;
        if (m < static_cast<unsigned int>(Cfg::kM) && r0 + rt < rows) {
            const unsigned int head = kvh * GROUP + m % GROUP;
            cp_async16(dst, q + (size_t)(r0 + rt) * q_stride + head * kHd + ch * 8);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
    for (unsigned int i = tid * 4; i < count; i += Cfg::kThreads * 4) {
        if (i + 4 <= count) {
            cp_async16(ent + i, gent + i);
        } else {
            for (unsigned int j = i; j < count; ++j) ent[j] = gent[j];
        }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    unsigned int qf[16][4];
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
        const unsigned int row = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const unsigned int col = ks * 16 + (lane >> 4) * 8;
        ldsm_x4(qf[ks], stage + swz(row, col));
    }
    __syncthreads();

    const RowInfo ra = row_info<GROUP>(warp * 16 + g, Cfg::kM, r0, rows, first_pos);
    const RowInfo rb = row_info<GROUP>(warp * 16 + g + 8, Cfg::kM, r0, rows, first_pos);
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float ma = -CUDART_INF_F, mb = -CUDART_INF_F, la = 0.0f, lb = 0.0f;

    const unsigned int head_off = kvh * kHd;
    const unsigned int ntiles = (count + 7) / 8;
    unsigned int warp_rows = 0;
    if (kSkip) {
        const unsigned int m_lo = warp * 16;
        const unsigned int m_hi = min(m_lo + 15, static_cast<unsigned int>(Cfg::kM) - 1);
        for (unsigned int rt = m_lo / GROUP; rt <= m_hi / GROUP; ++rt) warp_rows |= 1u << rt;
    }
    for (unsigned int it = 0; it < ntiles; ++it) {
        __nv_bfloat16* ks_s = stage + (it & 1u) * 2 * kTileElems;
        __nv_bfloat16* vs_s = ks_s + kTileElems;
        load_kv_tile_nvfp4<Cfg::kThreads>(
            ks_s, vs_s, kc, vc, ent, it * 8, count, bt, page_size,
            page_stride_bytes, data_section_bytes, tok_stride, head_off, tid);
        __syncthreads();
        bool skip_tile = false;
        if (kSkip) {
            const unsigned int slot = it * 8 + (lane & 7);
            const unsigned int need =
                lane < 8 && slot < count ? entry_need<TROWS>(ent[slot], first_pos + r0) : 0u;
            skip_tile = (__reduce_or_sync(0xffffffffu, need) & warp_rows) == 0;
        }

        if (!skip_tile) {
            float s[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
#pragma unroll
            for (int kstep = 0; kstep < 16; ++kstep) {
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    unsigned int b[4];
                    const unsigned int row = (np * 2 + (lane >> 4)) * 8 + (lane & 7);
                    const unsigned int col = kstep * 16 + ((lane >> 3) & 1) * 8;
                    ldsm_x4(b, ks_s + swz(row, col));
                    mma_bf16(s[np * 2], qf[kstep], b[0], b[1]);
                    mma_bf16(s[np * 2 + 1], qf[kstep], b[2], b[3]);
                }
            }

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
            const float base_a = na == -CUDART_INF_F ? 0.0f : na;
            const float base_b = nb == -CUDART_INF_F ? 0.0f : nb;
            ma = na;
            mb = nb;
            float suma = 0.0f, sumb = 0.0f;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                s[nt][0] = exp2f(s[nt][0] - base_a);
                s[nt][1] = exp2f(s[nt][1] - base_a);
                s[nt][2] = exp2f(s[nt][2] - base_b);
                s[nt][3] = exp2f(s[nt][3] - base_b);
                suma += s[nt][0] + s[nt][1];
                sumb += s[nt][2] + s[nt][3];
            }
            la = la * alpha_a + suma;
            lb = lb * alpha_b + sumb;
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
                    ldsm_x4_t(b, vs_s + swz(row, col));
                    mma_bf16(o[dp * 2], a, b[0], b[1]);
                    mma_bf16(o[dp * 2 + 1], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }

    la += __shfl_xor_sync(0xffffffffu, la, 1);
    la += __shfl_xor_sync(0xffffffffu, la, 2);
    lb += __shfl_xor_sync(0xffffffffu, lb, 1);
    lb += __shfl_xor_sync(0xffffffffu, lb, 2);
    const float ia = la > 0.0f ? 1.0f / la : 0.0f;
    const float ib = lb > 0.0f ? 1.0f / lb : 0.0f;
    const unsigned int ma_row = warp * 16 + g;
    const unsigned int mb_row = ma_row + 8;
    if (ra.live) {
        store_row(out, gate, r0 + ra.bit, kvh * GROUP + ma_row % GROUP, o_stride, gate_stride,
                  t, ia, o, 0);
    }
    if (rb.live) {
        store_row(out, gate, r0 + rb.bit, kvh * GROUP + mb_row % GROUP, o_stride, gate_stride,
                  t, ib, o, 2);
    }
}

}  // namespace

// Grid (tiles, num_kv_heads), block AttnCfg<T>::kThreads, dynamic shared memory
// 64 KiB + kCap * 4 B. q/out are [rows][24*256] at q_stride/o_stride elements;
// K/V pages are [page][page_size][kv_heads][256] at page_stride elements. A
// non-null gate ([rows] at gate_stride, 24*256 wide) fuses the sigmoid output gate.
#define QSA_PREFILL_ATTN_ENTRY(T)                                                              \
    extern "C" __global__ void __launch_bounds__(AttnCfg<T>::kThreads, 1)                      \
        qwen4_qsa_prefill_attn_t##T(                                                           \
            const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,         \
            const __nv_bfloat16* __restrict__ vc, const int* __restrict__ bt,                  \
            const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts, \
            __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,          \
            unsigned int rows, unsigned int first_pos, unsigned int q_stride,                  \
            unsigned int o_stride, unsigned int gate_stride, unsigned int page_size,           \
            unsigned long long page_stride, unsigned int tok_stride, float scale_log2) {       \
        qsa_prefill_attn<T, false>(q, kc, vc, bt, entries, counts, out, gate, rows, first_pos, \
                                   q_stride, o_stride, gate_stride, page_size, page_stride,    \
                                   tok_stride, scale_log2);                                    \
    }

QSA_PREFILL_ATTN_ENTRY(4)
QSA_PREFILL_ATTN_ENTRY(8)
QSA_PREFILL_ATTN_ENTRY(10)

// T=10 with per-warp skipping of 32-token tiles no row of the warp needs;
// pays off when the plan groups entries by need mask.
extern "C" __global__ void __launch_bounds__(AttnCfg<10>::kThreads, 1)
    qwen4_qsa_prefill_attn_t10_skip(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
        const __nv_bfloat16* __restrict__ vc, const int* __restrict__ bt,
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
        __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
        unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
        unsigned int gate_stride, unsigned int page_size, unsigned long long page_stride,
        unsigned int tok_stride, float scale_log2) {
    qsa_prefill_attn<10, true>(q, kc, vc, bt, entries, counts, out, gate, rows, first_pos,
                               q_stride, o_stride, gate_stride, page_size, page_stride,
                               tok_stride, scale_log2);
}

// NVFP4 paged-KV sparse prefill attention entries (1M-token single-Spark KV cache).
extern "C" __global__ void __launch_bounds__(AttnCfg<10>::kThreads, 1)
    qwen4_qsa_prefill_attn_nvfp4_t10(
        const __nv_bfloat16* __restrict__ q,
        const unsigned char* __restrict__ kc,
        const unsigned char* __restrict__ vc,
        const int* __restrict__ bt,
        const unsigned int* __restrict__ entries,
        const unsigned int* __restrict__ counts,
        __nv_bfloat16* __restrict__ out,
        const __nv_bfloat16* __restrict__ gate,
        unsigned int rows,
        unsigned int first_pos,
        unsigned int q_stride,
        unsigned int o_stride,
        unsigned int gate_stride,
        unsigned int page_size,
        unsigned long long page_stride_bytes,
        unsigned long long data_section_bytes,
        unsigned int tok_stride,
        float scale_log2) {
    qsa_prefill_attn_nvfp4<10, false>(
        q, kc, vc, bt, entries, counts, out, gate, rows, first_pos, q_stride, o_stride,
        gate_stride, page_size, page_stride_bytes, data_section_bytes, tok_stride, scale_log2);
}

extern "C" __global__ void __launch_bounds__(AttnCfg<10>::kThreads, 1)
    qwen4_qsa_prefill_attn_nvfp4_t10_skip(
        const __nv_bfloat16* __restrict__ q,
        const unsigned char* __restrict__ kc,
        const unsigned char* __restrict__ vc,
        const int* __restrict__ bt,
        const unsigned int* __restrict__ entries,
        const unsigned int* __restrict__ counts,
        __nv_bfloat16* __restrict__ out,
        const __nv_bfloat16* __restrict__ gate,
        unsigned int rows,
        unsigned int first_pos,
        unsigned int q_stride,
        unsigned int o_stride,
        unsigned int gate_stride,
        unsigned int page_size,
        unsigned long long page_stride_bytes,
        unsigned long long data_section_bytes,
        unsigned int tok_stride,
        float scale_log2) {
    qsa_prefill_attn_nvfp4<10, true>(
        q, kc, vc, bt, entries, counts, out, gate, rows, first_pos, q_stride, o_stride,
        gate_stride, page_size, page_stride_bytes, data_section_bytes, tok_stride, scale_log2);
}

// TP=4 KV-replicated GQA=6:1 entries (6 local Q heads per 1 local KV head).
extern "C" __global__ void __launch_bounds__(AttnCfg<10, 6>::kThreads, 1)
    qwen4_qsa_prefill_attn_t10_g6(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
        const __nv_bfloat16* __restrict__ vc, const int* __restrict__ bt,
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
        __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
        unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
        unsigned int gate_stride, unsigned int page_size, unsigned long long page_stride,
        unsigned int tok_stride, float scale_log2) {
    qsa_prefill_attn<10, false, 6>(q, kc, vc, bt, entries, counts, out, gate, rows, first_pos,
                                   q_stride, o_stride, gate_stride, page_size, page_stride,
                                   tok_stride, scale_log2);
}

extern "C" __global__ void __launch_bounds__(AttnCfg<10, 6>::kThreads, 1)
    qwen4_qsa_prefill_attn_t10_skip_g6(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
        const __nv_bfloat16* __restrict__ vc, const int* __restrict__ bt,
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
        __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
        unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
        unsigned int gate_stride, unsigned int page_size, unsigned long long page_stride,
        unsigned int tok_stride, float scale_log2) {
    qsa_prefill_attn<10, true, 6>(q, kc, vc, bt, entries, counts, out, gate, rows, first_pos,
                                  q_stride, o_stride, gate_stride, page_size, page_stride,
                                  tok_stride, scale_log2);
}

extern "C" __global__ void __launch_bounds__(AttnCfg<10, 6>::kThreads, 1)
    qwen4_qsa_prefill_attn_nvfp4_t10_g6(
        const __nv_bfloat16* __restrict__ q, const unsigned char* __restrict__ kc,
        const unsigned char* __restrict__ vc, const int* __restrict__ bt,
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
        __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
        unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
        unsigned int gate_stride, unsigned int page_size, unsigned long long page_stride_bytes,
        unsigned long long data_section_bytes, unsigned int tok_stride, float scale_log2) {
    qsa_prefill_attn_nvfp4<10, false, 6>(
        q, kc, vc, bt, entries, counts, out, gate, rows, first_pos, q_stride, o_stride,
        gate_stride, page_size, page_stride_bytes, data_section_bytes, tok_stride, scale_log2);
}

extern "C" __global__ void __launch_bounds__(AttnCfg<10, 6>::kThreads, 1)
    qwen4_qsa_prefill_attn_nvfp4_t10_skip_g6(
        const __nv_bfloat16* __restrict__ q, const unsigned char* __restrict__ kc,
        const unsigned char* __restrict__ vc, const int* __restrict__ bt,
        const unsigned int* __restrict__ entries, const unsigned int* __restrict__ counts,
        __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
        unsigned int rows, unsigned int first_pos, unsigned int q_stride, unsigned int o_stride,
        unsigned int gate_stride, unsigned int page_size, unsigned long long page_stride_bytes,
        unsigned long long data_section_bytes, unsigned int tok_stride, float scale_log2) {
    qsa_prefill_attn_nvfp4<10, true, 6>(
        q, kc, vc, bt, entries, counts, out, gate, rows, first_pos, q_stride, o_stride,
        gate_stride, page_size, page_stride_bytes, data_section_bytes, tok_stride, scale_log2);
}
