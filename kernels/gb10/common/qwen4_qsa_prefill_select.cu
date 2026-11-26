// SPDX-License-Identifier: AGPL-3.0-only
//
// QSA-sparse prefill selection for Qwen3.8-Flash-Next (ATLAS_QWEN4_PREFILL_QSA).
//
// The model attends, for a query at absolute position p, to the top-512 of its
// n_p = (p+1)/4 complete 4-token blocks (all of them while n_p <= 512) plus the
// 0-3 token open tail. Block score (HF Qwen4ExpTextQSAIndexer, Atlas decode
// qwen4_qsa_score):  s = (1/sqrt(128)) * sum_{h<4} relu(q_h . c_b).
//
// Three kernels, one prefill chunk at a time:
//   score  — tensor-core GEMM of the chunk's indexer queries against the paged
//            compressed-key side cache, relu-sum epilogue, fp32 logits.
//   topk   — per-row radix select (k = 512, equal scores prefer the LOWER
//            block, the decode tie rule) into a per-row block bitmask; rows
//            with n_p <= 512 get every complete block.
//   plan   — per tile of consecutive rows: the union of their bitmasks plus
//            their tail blocks, each entry packed as (member_bits << 22 | block).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr unsigned int kRatio = 4;
constexpr unsigned int kBlockTopk = 512;
constexpr unsigned int kIdxHeads = 4;
constexpr unsigned int kIdxDim = 128;
constexpr unsigned int kBlockBits = 22;
constexpr unsigned int kBlockMask = (1u << kBlockBits) - 1u;
constexpr int kScoreRows = 64;     // rows per CTA, 16 per warp
constexpr int kScoreBlocks = 64;   // compressed blocks per CTA
constexpr int kKeyPad = 136;       // 128 + 8 bf16: conflict-free fragment loads

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned int sa = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" ::"r"(sa), "l"(gmem));
}

__device__ __forceinline__ void ldsm_x4_bf16_k(unsigned int (&r)[4], const void* smem) {
    const unsigned int sa = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sa));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], unsigned int a0, unsigned int a1,
                                         unsigned int a2, unsigned int a3, unsigned int b0,
                                         unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned int ld_u32(const __nv_bfloat16* p) {
    unsigned int v;
    asm volatile(
        "{\n\t"
        ".reg .b64 policy;\n\t"
        "createpolicy.fractional.L2::evict_last.b64 policy, 1.0;\n\t"
        "ld.global.ca.L2::cache_hint.b32 %0, [%1], policy;\n\t"
        "}\n"
        : "=r"(v)
        : "l"(p));
    return v;
}

}  // namespace

// Grid (ceil(rows/64), ceil(max_blocks/64)), block 128. Row r is absolute
// position first_pos + r and owns scores[r * ld + b] for b < (first_pos+r+1)/4.
extern "C" __global__ void __launch_bounds__(128) qwen4_qsa_prefill_score(
    const __nv_bfloat16* __restrict__ iq,     // [rows][4][128], normed + roped
    const __nv_bfloat16* __restrict__ ckeys,  // [phys_page][page_size/4][128]
    const int* __restrict__ block_table,      // logical page -> physical page
    float* __restrict__ scores,               // [rows][ld]
    unsigned int rows,
    unsigned int first_pos,
    unsigned int ld,
    unsigned int page_size) {
    __shared__ __align__(16) __nv_bfloat16 sk[kScoreBlocks][kKeyPad];
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3;
    const unsigned int row0 = blockIdx.x * kScoreRows;
    const unsigned int blk0 = blockIdx.y * kScoreBlocks;
    if (row0 >= rows || page_size < kRatio) return;
    const unsigned int last_row = min(row0 + kScoreRows, rows) - 1;
    const unsigned int n_tile = (first_pos + last_row + 1) / kRatio;
    if (blk0 >= n_tile) return;

    const unsigned int gpp = page_size / kRatio;
    const bool gpp_pow2 = (gpp & (gpp - 1u)) == 0u;
    const unsigned int gpp_shift = static_cast<unsigned int>(__ffs(gpp) - 1);
    const unsigned int gpp_mask = gpp - 1u;
    const unsigned int bl0 = tid >> 4;
    const unsigned int ch8 = (tid & 15u) * 8u;
#pragma unroll
    for (unsigned int s = 0; s < 8; ++s) {
        const unsigned int bl = bl0 + s * 8u;
        const unsigned int b = blk0 + bl;
        if (b < n_tile) {
            const unsigned int page = gpp_pow2 ? (b >> gpp_shift) : (b / gpp);
            const int phys = __ldg(block_table + page);
            const size_t blk_idx =
                gpp_pow2 ? ((static_cast<size_t>(phys) << gpp_shift) + (b & gpp_mask))
                         : (static_cast<size_t>(phys) * gpp + (b % gpp));
            const __nv_bfloat16* src = ckeys + blk_idx * kIdxDim + ch8;
            cp_async16(&sk[bl][ch8], src);
        } else {
            *reinterpret_cast<uint4*>(&sk[bl][ch8]) = make_uint4(0, 0, 0, 0);
        }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    const unsigned int wrow = row0 + warp * 16;
    if (wrow >= rows) return;
    const unsigned int ra = min(wrow + g, rows - 1);
    const unsigned int rb = min(wrow + g + 8, rows - 1);
    const __nv_bfloat16* qa = iq + (size_t)ra * (kIdxHeads * kIdxDim);
    const __nv_bfloat16* qb = iq + (size_t)rb * (kIdxHeads * kIdxDim);
    const __nv_bfloat16* l_base = &sk[((lane >> 4) * 8u) + (lane & 7u)][((lane >> 3) & 1u) * 8u];

    float total[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        total[nt][0] = total[nt][1] = total[nt][2] = total[nt][3] = 0.0f;
    }
#pragma unroll 1
    for (unsigned int h = 0; h < kIdxHeads; ++h) {
        unsigned int af[8][4];
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const unsigned int c = h * kIdxDim + ks * 16 + t * 2;
            af[ks][0] = ld_u32(qa + c);
            af[ks][1] = ld_u32(qb + c);
            af[ks][2] = ld_u32(qa + c + 8);
            af[ks][3] = ld_u32(qb + c + 8);
        }
        float acc[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
        }
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                unsigned int b_reg[4];
                ldsm_x4_bf16_k(b_reg, l_base + (np * 16 * kKeyPad + ks * 16));
                mma_bf16(acc[np * 2 + 0], af[ks][0], af[ks][1], af[ks][2], af[ks][3], b_reg[0], b_reg[1]);
                mma_bf16(acc[np * 2 + 1], af[ks][0], af[ks][1], af[ks][2], af[ks][3], b_reg[2], b_reg[3]);
            }
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) total[nt][e] += fmaxf(acc[nt][e], 0.0f);
        }
    }

    // Scores are >= 0 by construction; fabsf only maps -0.0 to +0.0 so the
    // top-k may order IEEE bits as unsigned integers.
    const float scale = 0.08838834764831845f;  // 1/sqrt(128)
    const unsigned int na = (first_pos + wrow + g + 1) / kRatio;
    const unsigned int nb = (first_pos + wrow + g + 9) / kRatio;
    const bool row_a = wrow + g < rows;
    const bool row_b = wrow + g + 8 < rows;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const unsigned int b = blk0 + nt * 8 + t * 2;
        const float a0 = fabsf(total[nt][0] * scale);
        const float a1 = fabsf(total[nt][1] * scale);
        const float c0 = fabsf(total[nt][2] * scale);
        const float c1 = fabsf(total[nt][3] * scale);
        if (row_a && b < na) {
            float* dst = scores + (size_t)(wrow + g) * ld + b;
            if (b + 1 < na) {
                asm volatile("st.global.cs.v2.f32 [%0], {%1, %2};" : : "l"(dst), "f"(a0), "f"(a1) : "memory");
            } else {
                asm volatile("st.global.cs.f32 [%0], %1;" : : "l"(dst), "f"(a0) : "memory");
            }
        }
        if (row_b && b < nb) {
            float* dst = scores + (size_t)(wrow + g + 8) * ld + b;
            if (b + 1 < nb) {
                asm volatile("st.global.cs.v2.f32 [%0], {%1, %2};" : : "l"(dst), "f"(c0), "f"(c1) : "memory");
            } else {
                asm volatile("st.global.cs.f32 [%0], %1;" : : "l"(dst), "f"(c0) : "memory");
            }
        }
    }
}

namespace {

// One CTA (256 threads) per row. Rows [0, sparse_row0) are inside the dense
// window and take every complete block; row r >= sparse_row0 reads its logits
// from scores[(r - sparse_row0) * ld]. kStaged copies the row to dynamic
// shared memory (max_blocks * 4 B) before the four passes; otherwise every
// pass reads the score row in global memory (L2), for rows too long to stage.
template <bool kStaged>
__device__ __forceinline__ void topk_row(const float* __restrict__ scores,
                                         unsigned int* __restrict__ masks,  // [rows][mask_words]
                                         unsigned int rows, unsigned int first_pos,
                                         unsigned int sparse_row0, unsigned int ld,
                                         unsigned int mask_words) {
    extern __shared__ unsigned int skeys[];
    __shared__ unsigned int hist[2048];
    __shared__ unsigned int s_warp[8];
    __shared__ unsigned int s_bin;
    __shared__ unsigned int s_above;
    __shared__ unsigned int s_eq_total;

    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    const unsigned int n = (first_pos + row + 1) / kRatio;
    unsigned int* out = masks + (size_t)row * mask_words;

    if (row < sparse_row0 || n <= kBlockTopk) {
        for (unsigned int w = tid; w < mask_words; w += 256) {
            const unsigned int lo = w * 32;
            unsigned int bits = 0;
            if (lo + 32 <= n) {
                bits = 0xffffffffu;
            } else if (lo < n) {
                bits = (1u << (n - lo)) - 1u;
            }
            out[w] = bits;
        }
        return;
    }

    const float* src = scores + (size_t)(row - sparse_row0) * ld;
    if (kStaged) {
        // All of the row's 16-byte chunks in flight at once (rows are 64-float
        // aligned: ld is a multiple of 64); the scalar tail covers n % 4.
        const unsigned int n4 = n / 4;
        for (unsigned int i = tid; i < n4; i += 256) cp_async16(skeys + i * 4, src + i * 4);
        asm volatile("cp.async.commit_group;" ::: "memory");
        for (unsigned int i = n4 * 4 + tid; i < n; i += 256) {
            skeys[i] = __float_as_uint(__ldg(src + i));
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    const auto key_at = [&](unsigned int i) -> unsigned int {
        return (kStaged ? skeys[i] : __float_as_uint(__ldg(src + i))) & 0x7fffffffu;
    };

    // Scores are sums of relus times a positive scale, so their IEEE bits are
    // monotone as unsigned integers and bit 31 is clear. MSB-first radix
    // select of the k-th largest key (k = 512) over bits 30..20, 19..10, 9..0.
    unsigned int prefix = 0;
    unsigned int pmask = 0;
    unsigned int k = kBlockTopk;
    for (int pass = 0; pass < 3; ++pass) {
        const unsigned int shift = pass == 0 ? 20u : (pass == 1 ? 10u : 0u);
        const unsigned int bins = pass == 0 ? 2048u : 1024u;
        for (unsigned int b = tid; b < bins; b += 256) hist[b] = 0;
        __syncthreads();
        for (unsigned int i = tid; i < n; i += 256) {
            const unsigned int key = key_at(i);
            if ((key & pmask) == prefix) atomicAdd(&hist[(key >> shift) & (bins - 1u)], 1u);
        }
        __syncthreads();
        // Thread t owns the per-thread slice of bins counted from the TOP.
        const unsigned int per = bins / 256;
        const unsigned int hi = bins - tid * per;  // exclusive upper bin
        unsigned int local = 0;
        for (unsigned int j = 1; j <= per; ++j) local += hist[hi - j];
        unsigned int incl = local;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const unsigned int v = __shfl_up_sync(0xffffffffu, incl, off);
            if (lane >= static_cast<unsigned int>(off)) incl += v;
        }
        if (lane == 31) s_warp[warp] = incl;
        __syncthreads();
        for (unsigned int w = 0; w < warp; ++w) incl += s_warp[w];
        const unsigned int excl = incl - local;
        if (excl < k && k <= incl) {
            unsigned int above = excl;
            for (unsigned int j = 1; j <= per; ++j) {
                const unsigned int c = hist[hi - j];
                if (above + c >= k) {
                    s_bin = hi - j;
                    s_above = above;
                    s_eq_total = c;
                    break;
                }
                above += c;
            }
        }
        __syncthreads();
        prefix |= s_bin << shift;
        pmask |= (bins - 1u) << shift;
        k -= s_above;
        __syncthreads();
    }
    // prefix is the exact 512th-largest key; k (>= 1) of the s_eq_total equal
    // keys are taken, lowest block index first. One warp ballots one 32-block
    // word (lane = block): conflict-free shared reads.
    const unsigned int thr = prefix;
    const bool take_all_eq = k >= s_eq_total;
    if (take_all_eq) {
        for (unsigned int w = warp; w < mask_words; w += 8) {
            const unsigned int i = w * 32 + lane;
            const unsigned int key = i < n ? key_at(i) : 0u;
            const unsigned int gt = __ballot_sync(0xffffffffu, i < n && key > thr);
            const unsigned int eq = __ballot_sync(0xffffffffu, i < n && key == thr);
            if (lane == 0) out[w] = gt | eq;
        }
    } else if (warp == 0) {
        // Rare exact tie at the threshold: walk words in order with a budget.
        unsigned int left = k;
        for (unsigned int w = 0; w < mask_words; ++w) {
            const unsigned int i = w * 32 + lane;
            const unsigned int key = i < n ? key_at(i) : 0u;
            const unsigned int gt = __ballot_sync(0xffffffffu, i < n && key > thr);
            unsigned int eq = __ballot_sync(0xffffffffu, i < n && key == thr);
            unsigned int sel = 0;
            while (eq && left) {
                const unsigned int low = eq & (~eq + 1u);
                sel |= low;
                eq ^= low;
                --left;
            }
            if (lane == 0) out[w] = gt | sel;
        }
    }
}

}  // namespace

// Rows of up to MAX_TOPK_SMEM / 4 blocks (90,112 tokens): staged in dynamic
// shared memory, max_blocks * 4 B.
extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_prefill_topk(
    const float* __restrict__ scores, unsigned int* __restrict__ masks, unsigned int rows,
    unsigned int first_pos, unsigned int sparse_row0, unsigned int ld, unsigned int mask_words) {
    topk_row<true>(scores, masks, rows, first_pos, sparse_row0, ld, mask_words);
}

// Longer rows: the same selection read from global memory; no dynamic shared
// memory.
extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_prefill_topk_global(
    const float* __restrict__ scores, unsigned int* __restrict__ masks, unsigned int rows,
    unsigned int first_pos, unsigned int sparse_row0, unsigned int ld, unsigned int mask_words) {
    topk_row<false>(scores, masks, rows, first_pos, sparse_row0, ld, mask_words);
}

// One warp per tile of tile_rows (<= 10) consecutive rows. entries[tile*cap..]
// receives the tile's union of selected blocks and tail blocks, ascending, as
// (member << 22) | block where member bit i = row (tile*tile_rows + i) selected
// the block. Tail tokens are recognised by position in the attention kernel.
extern "C" __global__ void __launch_bounds__(128) qwen4_qsa_prefill_plan(
    const unsigned int* __restrict__ masks,  // [rows][mask_words]
    unsigned int* __restrict__ entries,      // [tiles][cap]
    unsigned int* __restrict__ counts,       // [tiles]
    unsigned int rows,
    unsigned int first_pos,
    unsigned int mask_words,
    unsigned int tile_rows,
    unsigned int cap,
    unsigned int tiles) {
    const unsigned int tile = blockIdx.x * 4 + (threadIdx.x >> 5);
    const unsigned int lane = threadIdx.x & 31;
    if (tile >= tiles || tile_rows == 0 || tile_rows > 10) return;
    const unsigned int r0 = tile * tile_rows;
    const unsigned int r1 = min(r0 + tile_rows, rows);
    const unsigned int last_pos = first_pos + r1 - 1;
    // Highest block any row touches: the last row's tail block (or last complete block).
    const unsigned int words = min(mask_words, ((last_pos + 1) / kRatio) / 32 + 1);
    unsigned int* dst = entries + (size_t)tile * cap;
    unsigned int total = 0;
    for (unsigned int w0 = 0; w0 < words; w0 += 32) {
        const unsigned int w = w0 + lane;
        unsigned int rw[10];
        unsigned int uni = 0;
#pragma unroll
        for (unsigned int i = 0; i < 10; ++i) {
            rw[i] = 0;
            if (i < r1 - r0 && w < words) {
                rw[i] = __ldg(masks + (size_t)(r0 + i) * mask_words + w);
                const unsigned int p = first_pos + r0 + i;
                const unsigned int tb = (p + 1) / kRatio;  // tail block (open group)
                if ((p + 1) % kRatio != 0 && tb / 32 == w) uni |= 1u << (tb % 32);
            }
            uni |= rw[i];
        }
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
            if (pos < cap) dst[pos] = (member << kBlockBits) | ((w * 32 + j) & kBlockMask);
            ++pos;
        }
        total += __shfl_sync(0xffffffffu, incl, 31);
    }
    if (lane == 0) counts[tile] = min(total, cap);
}

// One warp per tile: stable counting sort of the tile's plan entries by their
// need mask (member bits | rows whose open tail lies in the block), in place.
// Entries a warp of the attention kernel does not need then cluster into whole
// 32-token tiles it can skip (qwen4_qsa_prefill_attn_t10_skip). Deterministic:
// equal keys keep plan order (match_any ranking, no atomics on positions).
// Dynamic shared memory: cap * 4 + 1024 * 4 B.
extern "C" __global__ void __launch_bounds__(32) qwen4_qsa_prefill_plan_sort(
    unsigned int* __restrict__ entries,  // [tiles][cap]
    const unsigned int* __restrict__ counts,
    unsigned int first_pos,
    unsigned int tile_rows,
    unsigned int cap) {
    extern __shared__ unsigned int sm[];
    unsigned int* bins = sm;          // [1024]
    unsigned int* vals = sm + 1024;   // [cap]
    const unsigned int tile = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    const unsigned int count = min(counts[tile], cap);
    unsigned int* dst = entries + (size_t)tile * cap;
    const unsigned int row0_pos = first_pos + tile * tile_rows;
    const auto need_of = [&](unsigned int e) -> unsigned int {
        const unsigned int blk = e & kBlockMask;
        unsigned int need = (e >> kBlockBits) & ((1u << tile_rows) - 1u);
        // Rows with open tail tokens in blk: row0_pos + rt + 1 in [4blk + 1, 4blk + 3].
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
    // Exclusive prefix over the 1024 bins: lane l owns bins [32l, 32l + 32).
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
        const unsigned int i = i0 + lane;
        const bool live = i < count;
        const unsigned int e = live ? vals[i] : 0u;
        const unsigned int key = live ? need_of(e) : 0xffffffffu;
        const unsigned int peers = __match_any_sync(0xffffffffu, key);
        if (live) dst[bins[key] + __popc(peers & lt)] = e;
        __syncwarp();
        if (live && (peers & lt) == 0) bins[key] += __popc(peers);
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------
// ATLAS_QWEN4_PREFILL_QSA_FUSED_TOPK (default off): exact group-max select.
//
// The score kernel also writes, per row, the maximum key of every 8-block group
// (one m16n8 n-tile, so a quad shuffle). The select then works on n/8 group
// maxima plus the scores of the candidate groups instead of all n scores:
//   1. tau0 = the 512th-largest group maximum (every group when n/8 <= 512);
//   2. candidates = the groups whose maximum is >= tau0, ascending;
//   3. today's radix select (same passes, same tie rule) over their scores.
// Identity: the group maxima are real scores, so tau0 <= tau, the 512th-largest
// score of the row (the k-th largest of a subset never exceeds that of the
// set). Every score >= tau lies in a group whose maximum is >= tau >= tau0, so
// the candidates hold every key > tau and every key == tau. The k-th largest of
// the candidates is therefore tau, the counts above and at tau are the row's,
// and the lower-block-first walk over equal keys sees them in the same order:
// the mask is byte-identical to qwen4_qsa_prefill_topk[_global]. A row whose
// candidates exceed the shared-memory cap (ties among group maxima) runs the
// unchanged full-row algorithm on its global-memory scores.
// ---------------------------------------------------------------------------

namespace {

constexpr unsigned int kGroup = 8;  // blocks per group = one m16n8 n-tile
constexpr unsigned int kKeyMask = 0x7fffffffu;

// k-th largest of the keys >= floor among `count` keys (bit 31 clear; at least
// k >= 1 of them are >= floor) with topk_row's three MSB-first passes (bits
// 30..20, 19..10, 9..0) and the same bin ownership. Keys below the floor cannot
// be among the top k, so the result is that of all `count` keys. 256 threads;
// every thread returns the same values.
struct Kth {
    unsigned int thr;       // the k-th largest key
    unsigned int k_left;    // keys equal to thr the top k take (>= 1)
    unsigned int eq_total;  // keys equal to thr
};

template <typename KeyAt>
__device__ __forceinline__ Kth radix_kth(const KeyAt& key_at, unsigned int count, unsigned int k,
                                         unsigned int floor, unsigned int* hist,
                                         unsigned int* s_warp, unsigned int* s_sc) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    unsigned int prefix = 0;
    unsigned int pmask = 0;
    for (int pass = 0; pass < 3; ++pass) {
        const unsigned int shift = pass == 0 ? 20u : (pass == 1 ? 10u : 0u);
        const unsigned int bins = pass == 0 ? 2048u : 1024u;
        for (unsigned int b = tid; b < bins; b += 256) hist[b] = 0;
        __syncthreads();
        for (unsigned int i = tid; i < count; i += 256) {
            const unsigned int key = key_at(i);
            if (key >= floor && (key & pmask) == prefix) {
                atomicAdd(&hist[(key >> shift) & (bins - 1u)], 1u);
            }
        }
        __syncthreads();
        const unsigned int per = bins / 256;
        const unsigned int hi = bins - tid * per;
        unsigned int local = 0;
        for (unsigned int j = 1; j <= per; ++j) local += hist[hi - j];
        unsigned int incl = local;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const unsigned int v = __shfl_up_sync(0xffffffffu, incl, off);
            if (lane >= static_cast<unsigned int>(off)) incl += v;
        }
        if (lane == 31) s_warp[warp] = incl;
        __syncthreads();
        for (unsigned int w = 0; w < warp; ++w) incl += s_warp[w];
        const unsigned int excl = incl - local;
        if (excl < k && k <= incl) {
            unsigned int above = excl;
            for (unsigned int j = 1; j <= per; ++j) {
                const unsigned int c = hist[hi - j];
                if (above + c >= k) {
                    s_sc[0] = hi - j;
                    s_sc[1] = above;
                    s_sc[2] = c;
                    break;
                }
                above += c;
            }
        }
        __syncthreads();
        prefix |= s_sc[0] << shift;
        pmask |= (bins - 1u) << shift;
        k -= s_sc[1];
        __syncthreads();
    }
    Kth r;
    r.thr = prefix;
    r.k_left = k;
    r.eq_total = s_sc[2];
    return r;
}

// topk_row's mask writer for a full row of n keys (the fallback path).
template <typename KeyAt>
__device__ __forceinline__ void write_full_mask(const KeyAt& key_at, unsigned int n, const Kth& s,
                                                unsigned int* __restrict__ out,
                                                unsigned int mask_words) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int thr = s.thr;
    if (s.k_left >= s.eq_total) {
        for (unsigned int w = warp; w < mask_words; w += 8) {
            const unsigned int i = w * 32 + lane;
            const unsigned int key = i < n ? key_at(i) : 0u;
            const unsigned int gt = __ballot_sync(0xffffffffu, i < n && key > thr);
            const unsigned int eq = __ballot_sync(0xffffffffu, i < n && key == thr);
            if (lane == 0) out[w] = gt | eq;
        }
    } else if (warp == 0) {
        unsigned int left = s.k_left;
        for (unsigned int w = 0; w < mask_words; ++w) {
            const unsigned int i = w * 32 + lane;
            const unsigned int key = i < n ? key_at(i) : 0u;
            const unsigned int gt = __ballot_sync(0xffffffffu, i < n && key > thr);
            unsigned int eq = __ballot_sync(0xffffffffu, i < n && key == thr);
            unsigned int sel = 0;
            while (eq && left) {
                const unsigned int low = eq & (~eq + 1u);
                sel |= low;
                eq ^= low;
                --left;
            }
            if (lane == 0) out[w] = gt | sel;
        }
    }
}

// Candidate chunk c (entries 32c..32c+31 = candidate groups 4c..4c+3): lanes 0..3
// OR their group's 8 selection bits into the row's shared mask words. Block
// 8g + j is bit (g & 3) * 8 + j of word g >> 2 (groups never straddle words).
__device__ __forceinline__ void or_group_bytes(unsigned int sel, unsigned int c, unsigned int lane,
                                               unsigned int ncand, const unsigned int* cg,
                                               unsigned int* smask) {
    const unsigned int slot = c * 4 + lane;
    if (lane < 4 && slot < ncand) {
        const unsigned int byte = (sel >> (lane * 8)) & 0xffu;
        if (byte) {
            const unsigned int g = cg[slot];
            atomicOr(&smask[g >> 2], byte << ((g & 3u) * 8));
        }
    }
}

}  // namespace

// qwen4_qsa_prefill_score plus the group maxima: same grid, block and
// arithmetic (the MMA body is a verbatim copy; micro-queue 112 checks the
// scores byte for byte against the unfused kernel). gmax[r * ldg + g] is the
// largest key (IEEE bits & 0x7fffffff, the top-k order) among row r's scores
// of blocks 8g .. 8g+7 below its n_p; groups at or past n_p are not written.
extern "C" __global__ void __launch_bounds__(128) qwen4_qsa_prefill_score_gmax(
    const __nv_bfloat16* __restrict__ iq,     // [rows][4][128], normed + roped
    const __nv_bfloat16* __restrict__ ckeys,  // [phys_page][page_size/4][128]
    const int* __restrict__ block_table,      // logical page -> physical page
    float* __restrict__ scores,               // [rows][ld]
    unsigned int* __restrict__ gmax,          // [rows][ldg], ldg = ld / 8
    unsigned int rows,
    unsigned int first_pos,
    unsigned int ld,
    unsigned int ldg,
    unsigned int page_size) {
    __shared__ __align__(16) __nv_bfloat16 sk[kScoreBlocks][kKeyPad];
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3;
    const unsigned int row0 = blockIdx.x * kScoreRows;
    const unsigned int blk0 = blockIdx.y * kScoreBlocks;
    if (row0 >= rows || page_size < kRatio) return;
    const unsigned int last_row = min(row0 + kScoreRows, rows) - 1;
    const unsigned int n_tile = (first_pos + last_row + 1) / kRatio;
    if (blk0 >= n_tile) return;

    const unsigned int gpp = page_size / kRatio;
    const bool gpp_pow2 = (gpp & (gpp - 1u)) == 0u;
    const unsigned int gpp_shift = static_cast<unsigned int>(__ffs(gpp) - 1);
    const unsigned int gpp_mask = gpp - 1u;
    const unsigned int bl0 = tid >> 4;
    const unsigned int ch8 = (tid & 15u) * 8u;
#pragma unroll
    for (unsigned int s = 0; s < 8; ++s) {
        const unsigned int bl = bl0 + s * 8u;
        const unsigned int b = blk0 + bl;
        if (b < n_tile) {
            const unsigned int page = gpp_pow2 ? (b >> gpp_shift) : (b / gpp);
            const int phys = __ldg(block_table + page);
            const size_t blk_idx =
                gpp_pow2 ? ((static_cast<size_t>(phys) << gpp_shift) + (b & gpp_mask))
                         : (static_cast<size_t>(phys) * gpp + (b % gpp));
            const __nv_bfloat16* src = ckeys + blk_idx * kIdxDim + ch8;
            cp_async16(&sk[bl][ch8], src);
        } else {
            *reinterpret_cast<uint4*>(&sk[bl][ch8]) = make_uint4(0, 0, 0, 0);
        }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    const unsigned int wrow = row0 + warp * 16;
    if (wrow >= rows) return;
    const unsigned int ra = min(wrow + g, rows - 1);
    const unsigned int rb = min(wrow + g + 8, rows - 1);
    const __nv_bfloat16* qa = iq + (size_t)ra * (kIdxHeads * kIdxDim);
    const __nv_bfloat16* qb = iq + (size_t)rb * (kIdxHeads * kIdxDim);
    const __nv_bfloat16* l_base = &sk[((lane >> 4) * 8u) + (lane & 7u)][((lane >> 3) & 1u) * 8u];

    float total[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        total[nt][0] = total[nt][1] = total[nt][2] = total[nt][3] = 0.0f;
    }
#pragma unroll 1
    for (unsigned int h = 0; h < kIdxHeads; ++h) {
        unsigned int af[8][4];
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const unsigned int c = h * kIdxDim + ks * 16 + t * 2;
            af[ks][0] = ld_u32(qa + c);
            af[ks][1] = ld_u32(qb + c);
            af[ks][2] = ld_u32(qa + c + 8);
            af[ks][3] = ld_u32(qb + c + 8);
        }
        float acc[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
        }
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                unsigned int b_reg[4];
                ldsm_x4_bf16_k(b_reg, l_base + (np * 16 * kKeyPad + ks * 16));
                mma_bf16(acc[np * 2 + 0], af[ks][0], af[ks][1], af[ks][2], af[ks][3], b_reg[0], b_reg[1]);
                mma_bf16(acc[np * 2 + 1], af[ks][0], af[ks][1], af[ks][2], af[ks][3], b_reg[2], b_reg[3]);
            }
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) total[nt][e] += fmaxf(acc[nt][e], 0.0f);
        }
    }

    const float scale = 0.08838834764831845f;  // 1/sqrt(128)
    const unsigned int na = (first_pos + wrow + g + 1) / kRatio;
    const unsigned int nb = (first_pos + wrow + g + 9) / kRatio;
    const bool row_a = wrow + g < rows;
    const bool row_b = wrow + g + 8 < rows;
    // Group maxima kept by the quad lane that stores them: lane t owns groups 2t and 2t+1 of
    // this 64-block tile (n-tiles 2t, 2t+1), so after the loop each row's eight maxima are one
    // 32-byte store spread over its quad (8 rows x 32 B per warp instruction).
    unsigned int xa0 = 0, xa1 = 0, xb0 = 0, xb1 = 0;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const unsigned int b = blk0 + nt * 8 + t * 2;
        // The stored values, computed once: the stores below write exactly
        // what qwen4_qsa_prefill_score writes, and the maxima read the same bits.
        const float a0 = fabsf(total[nt][0] * scale);
        const float a1 = fabsf(total[nt][1] * scale);
        const float c0 = fabsf(total[nt][2] * scale);
        const float c1 = fabsf(total[nt][3] * scale);
        if (row_a && b < na) {
            float* dst = scores + (size_t)(wrow + g) * ld + b;
            if (b + 1 < na) {
                asm volatile("st.global.cs.v2.f32 [%0], {%1, %2};" : : "l"(dst), "f"(a0), "f"(a1) : "memory");
            } else {
                asm volatile("st.global.cs.f32 [%0], %1;" : : "l"(dst), "f"(a0) : "memory");
            }
        }
        if (row_b && b < nb) {
            float* dst = scores + (size_t)(wrow + g + 8) * ld + b;
            if (b + 1 < nb) {
                asm volatile("st.global.cs.v2.f32 [%0], {%1, %2};" : : "l"(dst), "f"(c0), "f"(c1) : "memory");
            } else {
                asm volatile("st.global.cs.f32 [%0], %1;" : : "l"(dst), "f"(c0) : "memory");
            }
        }
        // Keys below n_p only; scores are >= 0, so 0 is the identity of the max.
        unsigned int ma = b < na ? (__float_as_uint(a0) & kKeyMask) : 0u;
        if (b + 1 < na) ma = max(ma, __float_as_uint(a1) & kKeyMask);
        unsigned int mb = b < nb ? (__float_as_uint(c0) & kKeyMask) : 0u;
        if (b + 1 < nb) mb = max(mb, __float_as_uint(c1) & kKeyMask);
        ma = max(ma, __shfl_xor_sync(0xffffffffu, ma, 1));
        ma = max(ma, __shfl_xor_sync(0xffffffffu, ma, 2));
        mb = max(mb, __shfl_xor_sync(0xffffffffu, mb, 1));
        mb = max(mb, __shfl_xor_sync(0xffffffffu, mb, 2));
        if (t == static_cast<unsigned int>(nt >> 1)) {
            if (nt & 1) {
                xa1 = ma;
                xb1 = mb;
            } else {
                xa0 = ma;
                xb0 = mb;
            }
        }
    }
    // Same values at the same addresses as one store per group; groups at or past n_p are
    // not written (when only the first of the pair is below n_p, a 4-byte store).
    const unsigned int grp = blk0 / kGroup + 2 * t;
    if (row_a && grp * kGroup < na) {
        unsigned int* dst = gmax + (size_t)(wrow + g) * ldg + grp;
        if ((grp + 1) * kGroup < na) {
            *reinterpret_cast<uint2*>(dst) = make_uint2(xa0, xa1);
        } else {
            dst[0] = xa0;
        }
    }
    if (row_b && grp * kGroup < nb) {
        unsigned int* dst = gmax + (size_t)(wrow + g + 8) * ldg + grp;
        if ((grp + 1) * kGroup < nb) {
            *reinterpret_cast<uint2*>(dst) = make_uint2(xb0, xb1);
        } else {
            dst[0] = xb0;
        }
    }
}

// The select of ATLAS_QWEN4_PREFILL_QSA_FUSED_TOPK: same grid (one CTA of 256
// threads per row), arguments and mask output as qwen4_qsa_prefill_topk, plus
// the group maxima. Dynamic shared memory (a_words + 9 * cap_groups) * 4 B:
//   [0, a_words)                   the row's group maxima, then its mask words
//   [a_words, a_words + 8 * cap)   candidate keys, 8 per group, ascending block
//   [.., + cap)                    candidate group indices, ascending
// a_words >= max(ld / 8, mask_words) and a multiple of 4 (16-byte alignment).
// cap_groups = 0 sends every sparse row down the full-row fallback (tests).
extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_prefill_topk_fused(
    const float* __restrict__ scores, const unsigned int* __restrict__ gmax,
    unsigned int* __restrict__ masks, unsigned int rows, unsigned int first_pos,
    unsigned int sparse_row0, unsigned int ld, unsigned int ldg, unsigned int mask_words,
    unsigned int a_words, unsigned int cap_groups) {
    extern __shared__ __align__(16) unsigned int fsm[];
    __shared__ unsigned int hist[2048];
    __shared__ unsigned int s_warp[8];
    __shared__ unsigned int s_sc[4];
    unsigned int* sg = fsm;
    unsigned int* ck = fsm + a_words;
    unsigned int* cg = ck + 8 * cap_groups;

    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    const unsigned int n = (first_pos + row + 1) / kRatio;
    unsigned int* out = masks + (size_t)row * mask_words;

    if (row < sparse_row0 || n <= kBlockTopk) {
        for (unsigned int w = tid; w < mask_words; w += 256) {
            const unsigned int lo = w * 32;
            unsigned int bits = 0;
            if (lo + 32 <= n) {
                bits = 0xffffffffu;
            } else if (lo < n) {
                bits = (1u << (n - lo)) - 1u;
            }
            out[w] = bits;
        }
        return;
    }
    const float* src = scores + (size_t)(row - sparse_row0) * ld;
    const unsigned int* gsrc = gmax + (size_t)(row - sparse_row0) * ldg;
    const unsigned int ng = (n + kGroup - 1) / kGroup;
    const auto global_key = [&](unsigned int i) -> unsigned int {
        return __float_as_uint(__ldg(src + i)) & kKeyMask;
    };
    if (ng > a_words || mask_words > a_words) {
        const Kth s = radix_kth(global_key, n, kBlockTopk, 0u, hist, s_warp, s_sc);
        write_full_mask(global_key, n, s, out, mask_words);
        return;
    }

    // 1. The row's group maxima, then tau0 = the 512th largest (0 = every group).
    for (unsigned int i = tid; i < ng; i += 256) sg[i] = __ldg(gsrc + i);
    __syncthreads();
    unsigned int tau0 = 0;
    unsigned int ncand = ng;
    if (ng > kBlockTopk) {
        const auto group_key = [&](unsigned int i) -> unsigned int { return sg[i]; };
        const Kth gk = radix_kth(group_key, ng, kBlockTopk, 0u, hist, s_warp, s_sc);
        tau0 = gk.thr;
        ncand = (kBlockTopk - gk.k_left) + gk.eq_total;  // groups with max >= tau0
    }
    if (ncand > cap_groups) {
        const Kth s = radix_kth(global_key, n, kBlockTopk, 0u, hist, s_warp, s_sc);
        write_full_mask(global_key, n, s, out, mask_words);
        return;
    }

    // 2. Ordered compaction of the candidate groups: warp w owns the contiguous
    //    slice [w * span, (w + 1) * span) of the groups (span a multiple of 32).
    const unsigned int span = ((ng + 255u) / 256u) * 32u;
    const unsigned int g_lo = min(warp * span, ng);
    const unsigned int g_hi = min(g_lo + span, ng);
    unsigned int cnt = 0;
    for (unsigned int base = g_lo; base < g_hi; base += 32) {
        const unsigned int gi = base + lane;
        cnt += __popc(__ballot_sync(0xffffffffu, gi < g_hi && sg[gi] >= tau0));
    }
    if (lane == 0) s_warp[warp] = cnt;
    __syncthreads();
    unsigned int pos = 0;
    for (unsigned int w = 0; w < warp; ++w) pos += s_warp[w];
    for (unsigned int base = g_lo; base < g_hi; base += 32) {
        const unsigned int gi = base + lane;
        const bool take = gi < g_hi && sg[gi] >= tau0;
        const unsigned int bal = __ballot_sync(0xffffffffu, take);
        if (take) cg[pos + __popc(bal & ((1u << lane) - 1u))] = gi;
        pos += __popc(bal);
    }
    __syncthreads();

    // 3. Gather the candidates' scores (32 B per group). Only the row's last
    //    group can be partial; it is then the last candidate, so the m valid
    //    keys are a prefix of ck.
    const bool last_partial = cg[ncand - 1] == ng - 1;
    const unsigned int m = ncand * kGroup - (last_partial ? ng * kGroup - n : 0u);
    for (unsigned int i = tid; i < ncand * 2; i += 256) {
        const unsigned int slot = i >> 1;
        const unsigned int half = i & 1;
        const float4 v =
            __ldg(reinterpret_cast<const float4*>(src + cg[slot] * kGroup + half * 4));
        *reinterpret_cast<uint4*>(ck + slot * kGroup + half * 4) =
            make_uint4(__float_as_uint(v.x) & kKeyMask, __float_as_uint(v.y) & kKeyMask,
                       __float_as_uint(v.z) & kKeyMask, __float_as_uint(v.w) & kKeyMask);
    }
    __syncthreads();

    // 4. The exact selection over the candidates. Only keys >= tau0 can reach the
    //    top 512 (tau >= tau0), and the candidate groups' maxima are 512 of them.
    const auto cand_key = [&](unsigned int i) -> unsigned int { return ck[i]; };
    const Kth s = radix_kth(cand_key, m, kBlockTopk, tau0, hist, s_warp, s_sc);

    // 5. Mask words: zero (the maxima are dead), OR the selected bits, store.
    for (unsigned int w = tid; w < mask_words; w += 256) sg[w] = 0;
    __syncthreads();
    const unsigned int thr = s.thr;
    const unsigned int chunks = (m + 31) / 32;
    if (s.k_left >= s.eq_total) {
        for (unsigned int c = warp; c < chunks; c += 8) {
            const unsigned int i = c * 32 + lane;
            const unsigned int key = i < m ? ck[i] : 0u;
            const unsigned int sel = __ballot_sync(0xffffffffu, i < m && key >= thr);
            or_group_bytes(sel, c, lane, ncand, cg, sg);
        }
    } else if (warp == 0) {
        // Exact tie at the threshold: lowest block first, as topk_row.
        unsigned int left = s.k_left;
        for (unsigned int c = 0; c < chunks; ++c) {
            const unsigned int i = c * 32 + lane;
            const unsigned int key = i < m ? ck[i] : 0u;
            const unsigned int gt = __ballot_sync(0xffffffffu, i < m && key > thr);
            unsigned int eq = __ballot_sync(0xffffffffu, i < m && key == thr);
            unsigned int sel = gt;
            while (eq && left) {
                const unsigned int low = eq & (~eq + 1u);
                sel |= low;
                eq ^= low;
                --left;
            }
            or_group_bytes(sel, c, lane, ncand, cg, sg);
        }
    }
    __syncthreads();
    for (unsigned int w = tid; w < mask_words; w += 256) out[w] = sg[w];
}


// qwen4_qsa_prefill_topk_fused in 33 KB of shared memory instead of 49 KB, so three CTAs fit an SM
// and a 128-row V2 batch (128K) selects in one wave instead of two: the candidates' keys overwrite
// the staged group maxima (dead after the compaction) and the mask words reuse the radix histogram
// (dead after the last pass). Same algorithm, same passes, same tie walk, same masks.
// Dynamic shared memory (a_words + cap_groups) * 4 B: [0, a_words) the maxima, then the candidate
// keys (a_words >= ng and >= 8 * cap_groups, a multiple of 4); [a_words, + cap) the candidate group
// indices. mask_words must be <= 2048 (the histogram); otherwise, and when 8 * cap_groups > a_words,
// every row takes the full-row path. cap_groups = 0 sends every sparse row there (tests).
extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_prefill_topk_fused_c3(
    const float* __restrict__ scores, const unsigned int* __restrict__ gmax,
    unsigned int* __restrict__ masks, unsigned int rows, unsigned int first_pos,
    unsigned int sparse_row0, unsigned int ld, unsigned int ldg, unsigned int mask_words,
    unsigned int a_words, unsigned int cap_groups) {
    extern __shared__ __align__(16) unsigned int fsm3[];
    __shared__ unsigned int hist[2048];
    __shared__ unsigned int s_warp[8];
    __shared__ unsigned int s_sc[4];
    unsigned int* sg = fsm3;            // the maxima, then (step 3) the candidate keys
    unsigned int* ck = fsm3;
    unsigned int* cg = fsm3 + a_words;
    unsigned int* smask = hist;         // step 5 only

    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    const unsigned int n = (first_pos + row + 1) / kRatio;
    unsigned int* out = masks + (size_t)row * mask_words;

    if (row < sparse_row0 || n <= kBlockTopk) {
        for (unsigned int w = tid; w < mask_words; w += 256) {
            const unsigned int lo = w * 32;
            unsigned int bits = 0;
            if (lo + 32 <= n) {
                bits = 0xffffffffu;
            } else if (lo < n) {
                bits = (1u << (n - lo)) - 1u;
            }
            out[w] = bits;
        }
        return;
    }
    const float* src = scores + (size_t)(row - sparse_row0) * ld;
    const unsigned int* gsrc = gmax + (size_t)(row - sparse_row0) * ldg;
    const unsigned int ng = (n + kGroup - 1) / kGroup;
    const auto global_key = [&](unsigned int i) -> unsigned int {
        return __float_as_uint(__ldg(src + i)) & kKeyMask;
    };
    if (ng > a_words || mask_words > 2048u || 8u * cap_groups > a_words) {
        const Kth s = radix_kth(global_key, n, kBlockTopk, 0u, hist, s_warp, s_sc);
        write_full_mask(global_key, n, s, out, mask_words);
        return;
    }

    // 1. The row's group maxima, then tau0 = the 512th largest (0 = every group).
    for (unsigned int i = tid; i < ng; i += 256) sg[i] = __ldg(gsrc + i);
    __syncthreads();
    unsigned int tau0 = 0;
    unsigned int ncand = ng;
    if (ng > kBlockTopk) {
        const auto group_key = [&](unsigned int i) -> unsigned int { return sg[i]; };
        const Kth gk = radix_kth(group_key, ng, kBlockTopk, 0u, hist, s_warp, s_sc);
        tau0 = gk.thr;
        ncand = (kBlockTopk - gk.k_left) + gk.eq_total;  // groups with max >= tau0
    }
    if (ncand > cap_groups) {
        const Kth s = radix_kth(global_key, n, kBlockTopk, 0u, hist, s_warp, s_sc);
        write_full_mask(global_key, n, s, out, mask_words);
        return;
    }

    // 2. Ordered compaction of the candidate groups (as qwen4_qsa_prefill_topk_fused).
    const unsigned int span = ((ng + 255u) / 256u) * 32u;
    const unsigned int g_lo = min(warp * span, ng);
    const unsigned int g_hi = min(g_lo + span, ng);
    unsigned int cnt = 0;
    for (unsigned int base = g_lo; base < g_hi; base += 32) {
        const unsigned int gi = base + lane;
        cnt += __popc(__ballot_sync(0xffffffffu, gi < g_hi && sg[gi] >= tau0));
    }
    if (lane == 0) s_warp[warp] = cnt;
    __syncthreads();
    unsigned int pos = 0;
    for (unsigned int w = 0; w < warp; ++w) pos += s_warp[w];
    for (unsigned int base = g_lo; base < g_hi; base += 32) {
        const unsigned int gi = base + lane;
        const bool take = gi < g_hi && sg[gi] >= tau0;
        const unsigned int bal = __ballot_sync(0xffffffffu, take);
        if (take) cg[pos + __popc(bal & ((1u << lane) - 1u))] = gi;
        pos += __popc(bal);
    }
    __syncthreads();  // the maxima are dead from here: the candidate keys take their place

    // 3. Gather the candidates' scores (32 B per group) over the maxima. Only the row's last
    //    group can be partial; it is then the last candidate, so the m valid keys are a prefix.
    const bool last_partial = cg[ncand - 1] == ng - 1;
    const unsigned int m = ncand * kGroup - (last_partial ? ng * kGroup - n : 0u);
    for (unsigned int i = tid; i < ncand * 2; i += 256) {
        const unsigned int slot = i >> 1;
        const unsigned int half = i & 1;
        const float4 v =
            __ldg(reinterpret_cast<const float4*>(src + cg[slot] * kGroup + half * 4));
        *reinterpret_cast<uint4*>(ck + slot * kGroup + half * 4) =
            make_uint4(__float_as_uint(v.x) & kKeyMask, __float_as_uint(v.y) & kKeyMask,
                       __float_as_uint(v.z) & kKeyMask, __float_as_uint(v.w) & kKeyMask);
    }
    __syncthreads();

    // 4. The exact selection over the candidates (keys >= tau0 only, as the staged select).
    const auto cand_key = [&](unsigned int i) -> unsigned int { return ck[i]; };
    const Kth s = radix_kth(cand_key, m, kBlockTopk, tau0, hist, s_warp, s_sc);

    // 5. Mask words in the (now free) histogram: zero, OR the selected bits, store.
    for (unsigned int w = tid; w < mask_words; w += 256) smask[w] = 0;
    __syncthreads();
    const unsigned int thr = s.thr;
    const unsigned int chunks = (m + 31) / 32;
    if (s.k_left >= s.eq_total) {
        for (unsigned int c = warp; c < chunks; c += 8) {
            const unsigned int i = c * 32 + lane;
            const unsigned int key = i < m ? ck[i] : 0u;
            const unsigned int sel = __ballot_sync(0xffffffffu, i < m && key >= thr);
            or_group_bytes(sel, c, lane, ncand, cg, smask);
        }
    } else if (warp == 0) {
        // Exact tie at the threshold: lowest block first, as topk_row.
        unsigned int left = s.k_left;
        for (unsigned int c = 0; c < chunks; ++c) {
            const unsigned int i = c * 32 + lane;
            const unsigned int key = i < m ? ck[i] : 0u;
            const unsigned int gt = __ballot_sync(0xffffffffu, i < m && key > thr);
            unsigned int eq = __ballot_sync(0xffffffffu, i < m && key == thr);
            unsigned int sel = gt;
            while (eq && left) {
                const unsigned int low = eq & (~eq + 1u);
                sel |= low;
                eq ^= low;
                --left;
            }
            or_group_bytes(sel, c, lane, ncand, cg, smask);
        }
    }
    __syncthreads();
    for (unsigned int w = tid; w < mask_words; w += 256) out[w] = smask[w];
}

namespace {

__device__ __forceinline__ void hist_add_warp(unsigned int* hist, unsigned int bin) {
    const unsigned int mask = __activemask();
    const unsigned int peers = __match_any_sync(mask, bin);
    const unsigned int leader = static_cast<unsigned int>(__ffs(peers) - 1);
    if ((threadIdx.x & 31u) == leader) {
        atomicAdd(&hist[bin], static_cast<unsigned int>(__popc(peers)));
    }
}

template <bool kStaged>
__device__ __forceinline__ void topk_row_v2(const float* __restrict__ scores,
                                            unsigned int* __restrict__ masks,
                                            unsigned int rows, unsigned int first_pos,
                                            unsigned int sparse_row0, unsigned int ld,
                                            unsigned int mask_words) {
    extern __shared__ unsigned int skeys[];
    __shared__ unsigned int hist[2048];
    __shared__ unsigned int s_warp[8];
    __shared__ unsigned int s_bin;
    __shared__ unsigned int s_above;
    __shared__ unsigned int s_eq_total;

    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    const unsigned int n = (first_pos + row + 1) / kRatio;
    unsigned int* out = masks + (size_t)row * mask_words;

    if (row < sparse_row0 || n <= kBlockTopk) {
        for (unsigned int w = tid; w < mask_words; w += 256) {
            const unsigned int lo = w * 32;
            unsigned int bits = 0;
            if (lo + 32 <= n) {
                bits = 0xffffffffu;
            } else if (lo < n) {
                bits = (1u << (n - lo)) - 1u;
            }
            out[w] = bits;
        }
        return;
    }

    for (unsigned int b = tid; b < 2048u; b += 256) hist[b] = 0;
    __syncthreads();
    const float* src = scores + (size_t)(row - sparse_row0) * ld;
    const auto tail_key_at = [&](unsigned int i) -> unsigned int {
        return (kStaged ? skeys[i - 4096u] : __float_as_uint(__ldg(src + i))) & 0x7fffffffu;
    };

    unsigned int rk[16];
    unsigned int* const sc = hist + 1024;
    unsigned int prefix = 0;
    unsigned int pmask = 0;
    unsigned int k = kBlockTopk;
    unsigned int n_cand = n;
    for (int pass = 0; pass < 3; ++pass) {
        const unsigned int shift = pass == 0 ? 20u : (pass == 1 ? 10u : 0u);
        const unsigned int bins = pass == 0 ? 2048u : 1024u;
        if (pass == 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200)
            // SM120 (96 MiB L2): 4x coalesced 128-bit L2-cached loads (`ld.global.ca.v4.b32`)
            // matching `score_v2`'s `L2::evict_last` stores (`src` and `ld` are 256 B aligned).
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const unsigned int i0 = static_cast<unsigned int>(q) * 1024u + tid * 4u;
                if (i0 + 3u < n) {
                    uint4 v;
                    asm volatile("ld.global.ca.v4.b32 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                                 : "l"(src + i0));
                    rk[q * 4 + 0] = v.x & 0x7fffffffu;
                    rk[q * 4 + 1] = v.y & 0x7fffffffu;
                    rk[q * 4 + 2] = v.z & 0x7fffffffu;
                    rk[q * 4 + 3] = v.w & 0x7fffffffu;
                } else {
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        const unsigned int i = i0 + static_cast<unsigned int>(r);
                        unsigned int key = 0u;
                        if (i < n) {
                            unsigned int raw;
                            asm volatile("ld.global.ca.b32 %0, [%1];" : "=r"(raw) : "l"(src + i));
                            key = raw & 0x7fffffffu;
                        }
                        rk[q * 4 + r] = key;
                    }
                }
            }
#pragma unroll
            for (int s = 0; s < 16; ++s) {
                const unsigned int i =
                    static_cast<unsigned int>(s >> 2) * 1024u + tid * 4u + static_cast<unsigned int>(s & 3);
                if (i < n) {
                    hist_add_warp(hist, (rk[s] >> 20u) & 2047u);
                }
            }
#else
#pragma unroll
            for (int s = 0; s < 16; ++s) {
                const unsigned int i = tid + static_cast<unsigned int>(s) * 256u;
                unsigned int key = 0u;
                if (i < n) {
                    unsigned int raw;
                    asm volatile("ld.global.cs.b32 %0, [%1];" : "=r"(raw) : "l"(src + i));
                    key = raw & 0x7fffffffu;
                }
                rk[s] = key;
            }
#pragma unroll
            for (int s = 0; s < 16; ++s) {
                const unsigned int i = tid + static_cast<unsigned int>(s) * 256u;
                if (i < n) {
                    hist_add_warp(hist, (rk[s] >> 20u) & 2047u);
                }
            }
#endif
            if (n > 4096u) {
                if (kStaged) {
                    const unsigned int tail = n - 4096u;
                    const unsigned int tail4 = tail / 4u;
                    const float* src_tail = src + 4096u;
                    for (unsigned int base = tid; base < tail4; base += 512u) {
                        uint4 v0 = make_uint4(0u, 0u, 0u, 0u);
                        uint4 v1 = make_uint4(0u, 0u, 0u, 0u);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200)
                        asm volatile("ld.global.ca.v4.b32 {%0,%1,%2,%3}, [%4];"
                                     : "=r"(v0.x), "=r"(v0.y), "=r"(v0.z), "=r"(v0.w)
                                     : "l"(src_tail + base * 4u));
                        if (base + 256u < tail4) {
                            asm volatile("ld.global.ca.v4.b32 {%0,%1,%2,%3}, [%4];"
                                         : "=r"(v1.x), "=r"(v1.y), "=r"(v1.z), "=r"(v1.w)
                                         : "l"(src_tail + (base + 256u) * 4u));
                        }
#else
                        asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3}, [%4];"
                                     : "=r"(v0.x), "=r"(v0.y), "=r"(v0.z), "=r"(v0.w)
                                     : "l"(src_tail + base * 4u));
                        if (base + 256u < tail4) {
                            asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3}, [%4];"
                                         : "=r"(v1.x), "=r"(v1.y), "=r"(v1.z), "=r"(v1.w)
                                         : "l"(src_tail + (base + 256u) * 4u));
                        }
#endif
                        v0.x &= 0x7fffffffu;
                        v0.y &= 0x7fffffffu;
                        v0.z &= 0x7fffffffu;
                        v0.w &= 0x7fffffffu;
                        *reinterpret_cast<uint4*>(skeys + base * 4u) = v0;
                        hist_add_warp(hist, (v0.x >> 20u) & 2047u);
                        hist_add_warp(hist, (v0.y >> 20u) & 2047u);
                        hist_add_warp(hist, (v0.z >> 20u) & 2047u);
                        hist_add_warp(hist, (v0.w >> 20u) & 2047u);
                        if (base + 256u < tail4) {
                            v1.x &= 0x7fffffffu;
                            v1.y &= 0x7fffffffu;
                            v1.z &= 0x7fffffffu;
                            v1.w &= 0x7fffffffu;
                            *reinterpret_cast<uint4*>(skeys + (base + 256u) * 4u) = v1;
                            hist_add_warp(hist, (v1.x >> 20u) & 2047u);
                            hist_add_warp(hist, (v1.y >> 20u) & 2047u);
                            hist_add_warp(hist, (v1.z >> 20u) & 2047u);
                            hist_add_warp(hist, (v1.w >> 20u) & 2047u);
                        }
                    }
                    for (unsigned int i = tail4 * 4u + tid; i < tail; i += 256u) {
                        const unsigned int key =
                            __float_as_uint(__ldg(src_tail + i)) & 0x7fffffffu;
                        skeys[i] = key;
                        hist_add_warp(hist, (key >> 20u) & 2047u);
                    }
                } else {
                    for (unsigned int i = 4096u + tid; i < n; i += 256u) {
                        const unsigned int key = tail_key_at(i);
                        hist_add_warp(hist, (key >> 20u) & 2047u);
                    }
                }
            }
        } else if (pass == 1) {
            if (n_cand <= 1024u) {
#pragma unroll
                for (int s = 0; s < 16; ++s) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200)
                    const unsigned int i = static_cast<unsigned int>(s >> 2) * 1024u + tid * 4u
                                           + static_cast<unsigned int>(s & 3);
#else
                    const unsigned int i = tid + static_cast<unsigned int>(s) * 256u;
#endif
                    if (i < n && (rk[s] & pmask) == prefix) {
                        atomicAdd(&hist[(rk[s] >> 10u) & 1023u], 1u);
                        const unsigned int slot = atomicAdd(&s_warp[7], 1u);
                        if (slot < 1024u) sc[slot] = rk[s];
                    }
                }
                for (unsigned int i = 4096u + tid; i < n; i += 256u) {
                    const unsigned int key = tail_key_at(i);
                    if ((key & pmask) == prefix) {
                        atomicAdd(&hist[(key >> 10u) & 1023u], 1u);
                        const unsigned int slot = atomicAdd(&s_warp[7], 1u);
                        if (slot < 1024u) sc[slot] = key;
                    }
                }
            } else {
#pragma unroll
                for (int s = 0; s < 16; ++s) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200)
                    const unsigned int i = static_cast<unsigned int>(s >> 2) * 1024u + tid * 4u
                                           + static_cast<unsigned int>(s & 3);
#else
                    const unsigned int i = tid + static_cast<unsigned int>(s) * 256u;
#endif
                    if (i < n && (rk[s] & pmask) == prefix) {
                        atomicAdd(&hist[(rk[s] >> 10u) & 1023u], 1u);
                    }
                }
                for (unsigned int i = 4096u + tid; i < n; i += 256u) {
                    const unsigned int key = tail_key_at(i);
                    if ((key & pmask) == prefix) atomicAdd(&hist[(key >> 10u) & 1023u], 1u);
                }
            }
        } else {
            if (n_cand <= 1024u) {
                for (unsigned int i = tid; i < n_cand; i += 256) {
                    const unsigned int key = sc[i];
                    if ((key & pmask) == prefix) atomicAdd(&hist[key & 1023u], 1u);
                }
            } else {
#pragma unroll
                for (int s = 0; s < 16; ++s) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200)
                    const unsigned int i = static_cast<unsigned int>(s >> 2) * 1024u + tid * 4u
                                           + static_cast<unsigned int>(s & 3);
#else
                    const unsigned int i = tid + static_cast<unsigned int>(s) * 256u;
#endif
                    if (i < n && (rk[s] & pmask) == prefix) {
                        atomicAdd(&hist[rk[s] & 1023u], 1u);
                    }
                }
                for (unsigned int i = 4096u + tid; i < n; i += 256u) {
                    const unsigned int key = tail_key_at(i);
                    if ((key & pmask) == prefix) atomicAdd(&hist[key & 1023u], 1u);
                }
            }
        }
        __syncthreads();
        const unsigned int per = bins / 256;
        const unsigned int hi = bins - tid * per;
        unsigned int h_local[8];
        unsigned int local = 0;
#pragma unroll
        for (unsigned int j = 0; j < 8; ++j) {
            const unsigned int c = (j < per) ? hist[hi - 1u - j] : 0u;
            h_local[j] = c;
            local += c;
        }
        unsigned int incl = local;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const unsigned int v = __shfl_up_sync(0xffffffffu, incl, off);
            if (lane >= static_cast<unsigned int>(off)) incl += v;
        }
        if (lane == 31 && warp < 7) s_warp[warp] = incl;
        __syncthreads();
        for (unsigned int w = 0; w < warp; ++w) incl += s_warp[w];
        const unsigned int excl = incl - local;
        if (excl < k && k <= incl) {
            unsigned int above = excl;
            for (unsigned int j = 0; j < per; ++j) {
                const unsigned int c = h_local[j];
                if (above + c >= k) {
                    s_bin = hi - 1u - j;
                    s_above = above;
                    s_eq_total = c;
                    break;
                }
                above += c;
            }
        }
        if (pass < 2) {
            for (unsigned int b = tid; b < 1024u; b += 256) hist[b] = 0;
            if (pass == 0 && tid == 0) s_warp[7] = 0;
        }
        __syncthreads();
        prefix |= s_bin << shift;
        pmask |= (bins - 1u) << shift;
        k -= s_above;
        if (pass == 0) n_cand = s_eq_total;
        if (k >= s_eq_total) break;
    }

    const unsigned int thr = prefix;
    const bool take_all_eq = k >= s_eq_total;
    if (take_all_eq) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const unsigned int i0 = static_cast<unsigned int>(q) * 1024u + tid * 4u;
            unsigned int m4 = 0u;
#pragma unroll
            for (unsigned int r = 0u; r < 4u; ++r) {
                const unsigned int idx = i0 + r;
                if (idx < n && rk[q * 4 + static_cast<int>(r)] >= thr) {
                    m4 |= (1u << r);
                }
            }
            unsigned int word = m4 << ((lane & 7u) * 4u);
            word |= __shfl_xor_sync(0xffffffffu, word, 1);
            word |= __shfl_xor_sync(0xffffffffu, word, 2);
            word |= __shfl_xor_sync(0xffffffffu, word, 4);
            if ((lane & 7u) == 0u) {
                const unsigned int w =
                    static_cast<unsigned int>(q) * 32u + warp * 4u + (lane >> 3);
                if (w < mask_words) out[w] = word;
            }
        }
#else
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const unsigned int w = static_cast<unsigned int>(s) * 8u + warp;
            if (w < mask_words) {
                const unsigned int i = tid + static_cast<unsigned int>(s) * 256u;
                const unsigned int gt = __ballot_sync(0xffffffffu, i < n && rk[s] > thr);
                const unsigned int eq = __ballot_sync(0xffffffffu, i < n && rk[s] == thr);
                if (lane == 0) out[w] = gt | eq;
            }
        }
#endif
        if (n <= 4096u) {
            for (unsigned int w = 128u + tid; w < mask_words; w += 256u) {
                out[w] = 0u;
            }
        } else {
            for (unsigned int w = 128u + warp; w < mask_words; w += 8u) {
                const unsigned int i = w * 32u + lane;
                const unsigned int key = i < n ? tail_key_at(i) : 0u;
                const unsigned int gt = __ballot_sync(0xffffffffu, i < n && key > thr);
                const unsigned int eq = __ballot_sync(0xffffffffu, i < n && key == thr);
                if (lane == 0) out[w] = gt | eq;
            }
        }
    } else {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const unsigned int i0 = static_cast<unsigned int>(q) * 1024u + tid * 4u;
            unsigned int gt4 = 0u;
            unsigned int eq4 = 0u;
#pragma unroll
            for (unsigned int r = 0u; r < 4u; ++r) {
                const unsigned int idx = i0 + r;
                const unsigned int key = rk[q * 4 + static_cast<int>(r)];
                if (idx < n) {
                    if (key > thr) {
                        gt4 |= (1u << r);
                    } else if (key == thr) {
                        eq4 |= (1u << r);
                    }
                }
            }
            const unsigned int shift4 = (lane & 7u) * 4u;
            unsigned int gt_w = gt4 << shift4;
            unsigned int eq_w = eq4 << shift4;
            gt_w |= __shfl_xor_sync(0xffffffffu, gt_w, 1);
            gt_w |= __shfl_xor_sync(0xffffffffu, gt_w, 2);
            gt_w |= __shfl_xor_sync(0xffffffffu, gt_w, 4);
            eq_w |= __shfl_xor_sync(0xffffffffu, eq_w, 1);
            eq_w |= __shfl_xor_sync(0xffffffffu, eq_w, 2);
            eq_w |= __shfl_xor_sync(0xffffffffu, eq_w, 4);
            if ((lane & 7u) == 0u) {
                const unsigned int w =
                    static_cast<unsigned int>(q) * 32u + warp * 4u + (lane >> 3);
                hist[w] = gt_w;
                hist[128u + w] = eq_w;
            }
        }
#else
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const unsigned int w = static_cast<unsigned int>(s) * 8u + warp;
            const unsigned int i = tid + static_cast<unsigned int>(s) * 256u;
            const unsigned int gt = __ballot_sync(0xffffffffu, i < n && rk[s] > thr);
            const unsigned int eq = __ballot_sync(0xffffffffu, i < n && rk[s] == thr);
            if (lane == 0) {
                hist[w] = gt;
                hist[128u + w] = eq;
            }
        }
#endif
        __syncthreads();
        if (warp == 0) {
            unsigned int left = k;
            if (lane == 0) {
                const unsigned int w_lim = min(mask_words, 128u);
                for (unsigned int w = 0; w < w_lim; ++w) {
                    const unsigned int gt = hist[w];
                    unsigned int eq = hist[128u + w];
                    unsigned int sel = 0;
                    while (eq && left) {
                        const unsigned int low = eq & (~eq + 1u);
                        sel |= low;
                        eq ^= low;
                        --left;
                    }
                    out[w] = gt | sel;
                }
            }
            if (n > 4096u) {
                left = __shfl_sync(0xffffffffu, left, 0);
                for (unsigned int w = 128u; w < mask_words; ++w) {
                    const unsigned int i = w * 32u + lane;
                    const unsigned int key = i < n ? tail_key_at(i) : 0u;
                    const unsigned int gt = __ballot_sync(0xffffffffu, i < n && key > thr);
                    unsigned int eq = __ballot_sync(0xffffffffu, i < n && key == thr);
                    unsigned int sel = 0;
                    while (eq && left) {
                        const unsigned int low = eq & (~eq + 1u);
                        sel |= low;
                        eq ^= low;
                        --left;
                    }
                    if (lane == 0) out[w] = gt | sel;
                }
            }
        }
        if (n <= 4096u) {
            for (unsigned int w = 128u + tid; w < mask_words; w += 256u) {
                out[w] = 0u;
            }
        }
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_prefill_topk_v2(
    const float* __restrict__ scores, unsigned int* __restrict__ masks, unsigned int rows,
    unsigned int first_pos, unsigned int sparse_row0, unsigned int ld, unsigned int mask_words) {
    topk_row_v2<true>(scores, masks, rows, first_pos, sparse_row0, ld, mask_words);
}

extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_prefill_topk_global_v2(
    const float* __restrict__ scores, unsigned int* __restrict__ masks, unsigned int rows,
    unsigned int first_pos, unsigned int sparse_row0, unsigned int ld, unsigned int mask_words) {
    topk_row_v2<false>(scores, masks, rows, first_pos, sparse_row0, ld, mask_words);
}

// Fused qwen4_qsa_prefill_plan + qwen4_qsa_prefill_plan_sort (1 warp per tile,
// dynamic shared memory (1024 + cap) * 4 B): builds the ascending union of
// selected and tail blocks in vals[0..count), histograms need_of(e) into
// bins[0..1024) on the fly, writes counts[tile] = count, and scatters vals[]
// in ascending order into entries[tile*cap..], bit-identical to the two-kernel
// sequence.
extern "C" __global__ void __launch_bounds__(32) qwen4_qsa_prefill_plan_and_sort(
    const unsigned int* __restrict__ masks,
    unsigned int* __restrict__ entries,
    unsigned int* __restrict__ counts,
    unsigned int rows,
    unsigned int first_pos,
    unsigned int mask_words,
    unsigned int tile_rows,
    unsigned int cap,
    unsigned int tiles) {
    extern __shared__ unsigned int sm[];
    unsigned int* bins = sm;
    unsigned int* vals = sm + 1024;
    const unsigned int tile = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (tile >= tiles || tile_rows == 0 || tile_rows > 10) return;
    const unsigned int r0 = tile * tile_rows;
    const unsigned int r1 = min(r0 + tile_rows, rows);
    const unsigned int last_pos = first_pos + r1 - 1;
    const unsigned int words = min(mask_words, ((last_pos + 1) / kRatio) / 32 + 1);
    unsigned int* dst = entries + (size_t)tile * cap;
    const unsigned int row0_pos = first_pos + r0;
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
    unsigned int total = 0;
    for (unsigned int w0 = 0; w0 < words; w0 += 32) {
        const unsigned int w = w0 + lane;
        unsigned int rw[10];
        unsigned int uni = 0;
#pragma unroll
        for (unsigned int i = 0; i < 10; ++i) {
            rw[i] = 0;
            if (i < r1 - r0 && w < words) {
                rw[i] = __ldg(masks + (size_t)(r0 + i) * mask_words + w);
                const unsigned int p = first_pos + r0 + i;
                const unsigned int tb = (p + 1) / kRatio;
                if ((p + 1) % kRatio != 0 && tb / 32 == w) uni |= 1u << (tb % 32);
            }
            uni |= rw[i];
        }
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
            if (pos < cap) {
                const unsigned int e = (member << kBlockBits) | ((w * 32 + j) & kBlockMask);
                vals[pos] = e;
                atomicAdd(&bins[need_of(e)], 1u);
            }
            ++pos;
        }
        total += __shfl_sync(0xffffffffu, incl, 31);
    }
    const unsigned int count = min(total, cap);
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
    const unsigned int lt = (1u << lane) - 1u;
    for (unsigned int i0 = 0; i0 < count; i0 += 32) {
        const unsigned int i = i0 + lane;
        const bool live = i < count;
        const unsigned int e = live ? vals[i] : 0u;
        const unsigned int key = live ? need_of(e) : 0xffffffffu;
        const unsigned int peers = __match_any_sync(0xffffffffu, key);
        if (live) dst[bins[key] + __popc(peers & lt)] = e;
        __syncwarp();
        if (live && (peers & lt) == 0) bins[key] += __popc(peers);
        __syncwarp();
    }
}

namespace {

__device__ __forceinline__ void ldsm_x4_bf16(unsigned int (&r)[4], const void* smem) {
    const unsigned int sa = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sa));
}

__device__ __forceinline__ uint64_t select_policy_evict_last() {
    uint64_t pol;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    return pol;
}

__device__ __forceinline__ unsigned int ld_u32_evict_last(const void* p) {
    unsigned int v;
    const uint64_t pol = select_policy_evict_last();
    asm volatile("ld.global.ca.L2::cache_hint.b32 %0, [%1], %2;\n"
                 : "=r"(v)
                 : "l"(p), "l"(pol));
    return v;
}

__device__ __forceinline__ void prefetch_ckey_phys(int (&phys)[8], const int* __restrict__ block_table,
                                                   unsigned int blk0, unsigned int n_tile,
                                                   unsigned int gpp, unsigned int gpp_shift,
                                                   bool gpp_pow2, unsigned int tid) {
    const unsigned int bl0 = tid >> 4;
    if (gpp == 16u && blk0 + kScoreBlocks <= n_tile) {
        const unsigned int page0 = blk0 >> 4;
#pragma unroll
        for (unsigned int p = 0; p < 4; ++p) {
            const int pg = static_cast<int>(ld_u32_evict_last(block_table + page0 + p));
            phys[2u * p] = pg;
            phys[2u * p + 1u] = pg;
        }
    } else if (gpp_pow2 && blk0 + kScoreBlocks <= n_tile) {
#pragma unroll
        for (unsigned int s = 0; s < 8; ++s) {
            const unsigned int b = blk0 + bl0 + s * 8u;
            phys[s] = static_cast<int>(ld_u32_evict_last(block_table + (b >> gpp_shift)));
        }
    } else {
#pragma unroll
        for (unsigned int s = 0; s < 8; ++s) {
            const unsigned int b = blk0 + bl0 + s * 8u;
            if (b < n_tile) {
                const unsigned int page = gpp_pow2 ? (b >> gpp_shift) : (b / gpp);
                phys[s] = static_cast<int>(ld_u32_evict_last(block_table + page));
            } else {
                phys[s] = 0;
            }
        }
    }
}

__device__ __forceinline__ void issue_ckey_tile(__nv_bfloat16 (*__restrict__ sk_dst)[kKeyPad],
                                                const __nv_bfloat16* __restrict__ ckeys,
                                                const int (&phys)[8], unsigned int blk0,
                                                unsigned int n_tile, unsigned int gpp,
                                                unsigned int gpp_shift, unsigned int gpp_mask,
                                                bool gpp_pow2, unsigned int tid) {
    const unsigned int bl0 = tid >> 4;
    const unsigned int ch8 = (tid & 15u) * 8u;
    const uint64_t pol = select_policy_evict_last();
    if (gpp == 16u && blk0 + kScoreBlocks <= n_tile) {
#pragma unroll
        for (unsigned int p = 0; p < 4; ++p) {
            const unsigned int bl_even = bl0 + (2u * p) * 8u;
            const size_t blk_even = (static_cast<size_t>(phys[2u * p]) << 4) + bl0;
            const __nv_bfloat16* src_even = ckeys + blk_even * kIdxDim + ch8;
            const __nv_bfloat16* src_odd = ckeys + (blk_even + 8u) * kIdxDim + ch8;
            const unsigned int dst_even =
                static_cast<unsigned int>(__cvta_generic_to_shared(&sk_dst[bl_even][ch8]));
            const unsigned int dst_odd =
                static_cast<unsigned int>(__cvta_generic_to_shared(&sk_dst[bl_even + 8u][ch8]));
            asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::128B [%0], [%1], 16, %2;\n"
                         :
                         : "r"(dst_even), "l"(src_even), "l"(pol));
            asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::128B [%0], [%1], 16, %2;\n"
                         :
                         : "r"(dst_odd), "l"(src_odd), "l"(pol));
        }
    } else if (gpp_pow2 && blk0 + kScoreBlocks <= n_tile) {
#pragma unroll
        for (unsigned int s = 0; s < 8; ++s) {
            const unsigned int bl = bl0 + s * 8u;
            const unsigned int b = blk0 + bl;
            const size_t blk_idx =
                (static_cast<size_t>(phys[s]) << gpp_shift) + (b & gpp_mask);
            const __nv_bfloat16* src = ckeys + blk_idx * kIdxDim + ch8;
            const unsigned int dst_s =
                static_cast<unsigned int>(__cvta_generic_to_shared(&sk_dst[bl][ch8]));
            asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::128B [%0], [%1], 16, %2;\n"
                         :
                         : "r"(dst_s), "l"(src), "l"(pol));
        }
    } else {
#pragma unroll
        for (unsigned int s = 0; s < 8; ++s) {
            const unsigned int bl = bl0 + s * 8u;
            const unsigned int b = blk0 + bl;
            if (b < n_tile) {
                const size_t blk_idx =
                    gpp_pow2 ? ((static_cast<size_t>(phys[s]) << gpp_shift) + (b & gpp_mask))
                             : (static_cast<size_t>(phys[s]) * gpp + (b % gpp));
                const __nv_bfloat16* src = ckeys + blk_idx * kIdxDim + ch8;
                const unsigned int dst_s =
                    static_cast<unsigned int>(__cvta_generic_to_shared(&sk_dst[bl][ch8]));
                asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::128B [%0], [%1], 16, %2;\n"
                             :
                             : "r"(dst_s), "l"(src), "l"(pol));
            } else {
                *reinterpret_cast<uint4*>(&sk_dst[bl][ch8]) = make_uint4(0, 0, 0, 0);
            }
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <bool kWithGmax>
__device__ __forceinline__ void qsa_prefill_score_v2(
    const __nv_bfloat16* __restrict__ iq,     // [rows][4][128], normed + roped
    const __nv_bfloat16* __restrict__ ckeys,  // [phys_page][page_size/4][128]
    const int* __restrict__ block_table,      // logical page -> physical page
    float* __restrict__ scores,               // [rows][ld]
    unsigned int* __restrict__ gmax,          // [rows][ldg], ldg = ld / 8
    unsigned int rows, unsigned int first_pos, unsigned int ld, unsigned int ldg,
    unsigned int page_size) {
    __shared__ __align__(16) __nv_bfloat16 sk[2][kScoreBlocks][kKeyPad];
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3;
    const unsigned int row0 = blockIdx.x * kScoreRows;
    if (row0 >= rows || page_size < kRatio) return;
    const unsigned int last_row = min(row0 + kScoreRows, rows) - 1;
    const unsigned int n_tile = (first_pos + last_row + 1) / kRatio;
    const unsigned int n_max = (first_pos + rows) / kRatio;
    const unsigned int total_tiles = (n_max + kScoreBlocks - 1u) / kScoreBlocks;
    const unsigned int grid_y = max(gridDim.y, 1u);
    const unsigned int tiles_per_cta = max(1u, (total_tiles + grid_y - 1u) / grid_y);
    unsigned int blk0 = blockIdx.y * tiles_per_cta * kScoreBlocks;
    const unsigned int blk_end = min(n_tile, (blockIdx.y + 1u) * tiles_per_cta * kScoreBlocks);
    if (blk0 >= blk_end) return;

    const unsigned int gpp = page_size / kRatio;
    const bool gpp_pow2 = (gpp & (gpp - 1u)) == 0u;
    const unsigned int gpp_shift = static_cast<unsigned int>(__ffs(gpp) - 1);
    const unsigned int gpp_mask = gpp - 1u;

    int next_phys[8];
    prefetch_ckey_phys(next_phys, block_table, blk0, n_tile, gpp, gpp_shift, gpp_pow2, tid);

    const unsigned int wrow = row0 + warp * 16;
    const unsigned int ra = min(wrow + g, rows - 1);
    const unsigned int rb = min(wrow + g + 8, rows - 1);
    const __nv_bfloat16* qa = iq + (size_t)ra * (kIdxHeads * kIdxDim);
    const __nv_bfloat16* qb = iq + (size_t)rb * (kIdxHeads * kIdxDim);

    unsigned int af[kIdxHeads][8][4];
#pragma unroll
    for (unsigned int h = 0; h < kIdxHeads; ++h) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const unsigned int c = h * kIdxDim + ks * 16 + t * 2;
            af[h][ks][0] = ld_u32_evict_last(qa + c);
            af[h][ks][1] = ld_u32_evict_last(qb + c);
            af[h][ks][2] = ld_u32_evict_last(qa + c + 8);
            af[h][ks][3] = ld_u32_evict_last(qb + c + 8);
        }
    }

    issue_ckey_tile(sk[0], ckeys, next_phys, blk0, n_tile, gpp, gpp_shift, gpp_mask, gpp_pow2, tid);
    if (blk0 + kScoreBlocks < blk_end) {
        prefetch_ckey_phys(next_phys, block_table, blk0 + kScoreBlocks, n_tile, gpp, gpp_shift,
                           gpp_pow2, tid);
    }

    const float scale = 0.08838834764831845f;  // 1/sqrt(128)
    const unsigned int na = (first_pos + wrow + g + 1) / kRatio;
    const unsigned int nb = (first_pos + wrow + g + 9) / kRatio;
    const bool row_a = wrow + g < rows;
    const bool row_b = wrow + g + 8 < rows;
    const unsigned int l_row_off = ((lane >> 4) * 8u) + (lane & 7u);
    const unsigned int l_col_off = ((lane >> 3) & 1u) * 8u;

    for (unsigned int stage = 0;; stage ^= 1u) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        const __nv_bfloat16* l_base = &sk[stage][l_row_off][l_col_off];
        unsigned int b_cur[4];
        ldsm_x4_bf16(b_cur, l_base);

        const unsigned int next_blk0 = blk0 + kScoreBlocks;
        if (next_blk0 < blk_end) {
            issue_ckey_tile(sk[stage ^ 1u], ckeys, next_phys, next_blk0, n_tile, gpp, gpp_shift,
                            gpp_mask, gpp_pow2, tid);
            if (next_blk0 + kScoreBlocks < blk_end) {
                prefetch_ckey_phys(next_phys, block_table, next_blk0 + kScoreBlocks, n_tile, gpp,
                                   gpp_shift, gpp_pow2, tid);
            }
        }

        unsigned int xa0 = 0, xa1 = 0, xb0 = 0, xb1 = 0;

#pragma unroll
        for (int np = 0; np < 4; ++np) {
            float acc[kIdxHeads][2][4];
#pragma unroll
            for (unsigned int h = 0; h < kIdxHeads; ++h) {
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    acc[h][s][0] = acc[h][s][1] = acc[h][s][2] = acc[h][s][3] = 0.0f;
                }
            }
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                unsigned int b_next[4];
                if (ks < 7) {
                    ldsm_x4_bf16(b_next, l_base + (np * 16 * kKeyPad + (ks + 1) * 16));
                } else if (np < 3) {
                    ldsm_x4_bf16(b_next, l_base + ((np + 1) * 16 * kKeyPad));
                }
#pragma unroll
                for (unsigned int h = 0; h < kIdxHeads; ++h) {
                    mma_bf16(acc[h][0], af[h][ks][0], af[h][ks][1], af[h][ks][2], af[h][ks][3],
                             b_cur[0], b_cur[1]);
                    mma_bf16(acc[h][1], af[h][ks][0], af[h][ks][1], af[h][ks][2], af[h][ks][3],
                             b_cur[2], b_cur[3]);
                }
                if (ks < 7 || np < 3) {
                    b_cur[0] = b_next[0];
                    b_cur[1] = b_next[1];
                    b_cur[2] = b_next[2];
                    b_cur[3] = b_next[3];
                }
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int nt = np * 2 + s;
                float tot0 = 0.0f, tot1 = 0.0f, tot2 = 0.0f, tot3 = 0.0f;
#pragma unroll
                for (unsigned int h = 0; h < kIdxHeads; ++h) {
                    tot0 += fmaxf(acc[h][s][0], 0.0f);
                    tot1 += fmaxf(acc[h][s][1], 0.0f);
                    tot2 += fmaxf(acc[h][s][2], 0.0f);
                    tot3 += fmaxf(acc[h][s][3], 0.0f);
                }
                const unsigned int b = blk0 + nt * 8 + t * 2;
                const float a0 = fabsf(tot0 * scale);
                const float a1 = fabsf(tot1 * scale);
                const float c0 = fabsf(tot2 * scale);
                const float c1 = fabsf(tot3 * scale);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200)
                const uint64_t pol_scores = select_policy_evict_last();
                if (row_a && b < na) {
                    float* dst = scores + (size_t)(wrow + g) * ld + b;
                    if (b + 1 < na) {
                        asm volatile("st.global.wb.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;"
                                     :
                                     : "l"(dst), "f"(a0), "f"(a1), "l"(pol_scores)
                                     : "memory");
                    } else {
                        asm volatile("st.global.wb.L2::cache_hint.f32 [%0], %1, %2;"
                                     :
                                     : "l"(dst), "f"(a0), "l"(pol_scores)
                                     : "memory");
                    }
                }
                if (row_b && b < nb) {
                    float* dst = scores + (size_t)(wrow + g + 8) * ld + b;
                    if (b + 1 < nb) {
                        asm volatile("st.global.wb.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;"
                                     :
                                     : "l"(dst), "f"(c0), "f"(c1), "l"(pol_scores)
                                     : "memory");
                    } else {
                        asm volatile("st.global.wb.L2::cache_hint.f32 [%0], %1, %2;"
                                     :
                                     : "l"(dst), "f"(c0), "l"(pol_scores)
                                     : "memory");
                    }
                }
#else
                if (row_a && b < na) {
                    float* dst = scores + (size_t)(wrow + g) * ld + b;
                    if (b + 1 < na) {
                        asm volatile("st.global.cs.v2.f32 [%0], {%1, %2};"
                                     :
                                     : "l"(dst), "f"(a0), "f"(a1)
                                     : "memory");
                    } else {
                        asm volatile("st.global.cs.f32 [%0], %1;"
                                     :
                                     : "l"(dst), "f"(a0)
                                     : "memory");
                    }
                }
                if (row_b && b < nb) {
                    float* dst = scores + (size_t)(wrow + g + 8) * ld + b;
                    if (b + 1 < nb) {
                        asm volatile("st.global.cs.v2.f32 [%0], {%1, %2};"
                                     :
                                     : "l"(dst), "f"(c0), "f"(c1)
                                     : "memory");
                    } else {
                        asm volatile("st.global.cs.f32 [%0], %1;"
                                     :
                                     : "l"(dst), "f"(c0)
                                     : "memory");
                    }
                }
#endif
                if (kWithGmax) {
                    unsigned int ma = b < na ? (__float_as_uint(a0) & kKeyMask) : 0u;
                    if (b + 1 < na) ma = max(ma, __float_as_uint(a1) & kKeyMask);
                    unsigned int mb = b < nb ? (__float_as_uint(c0) & kKeyMask) : 0u;
                    if (b + 1 < nb) mb = max(mb, __float_as_uint(c1) & kKeyMask);
                    ma = max(ma, __shfl_xor_sync(0xffffffffu, ma, 1));
                    ma = max(ma, __shfl_xor_sync(0xffffffffu, ma, 2));
                    mb = max(mb, __shfl_xor_sync(0xffffffffu, mb, 1));
                    mb = max(mb, __shfl_xor_sync(0xffffffffu, mb, 2));
                    if (t == static_cast<unsigned int>(np)) {
                        if (s) {
                            xa1 = ma;
                            xb1 = mb;
                        } else {
                            xa0 = ma;
                            xb0 = mb;
                        }
                    }
                }
            }
        }

        if (kWithGmax) {
            const uint64_t pol_el = select_policy_evict_last();
            const unsigned int grp = blk0 / kGroup + 2 * t;
            if (row_a && grp * kGroup < na) {
                unsigned int* dst = gmax + (size_t)(wrow + g) * ldg + grp;
                if ((grp + 1) * kGroup < na) {
                    asm volatile("st.global.wb.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                                 :
                                 : "l"(dst), "r"(xa0), "r"(xa1), "l"(pol_el)
                                 : "memory");
                } else {
                    asm volatile("st.global.wb.L2::cache_hint.u32 [%0], %1, %2;\n"
                                 :
                                 : "l"(dst), "r"(xa0), "l"(pol_el)
                                 : "memory");
                }
            }
            if (row_b && grp * kGroup < nb) {
                unsigned int* dst = gmax + (size_t)(wrow + g + 8) * ldg + grp;
                if ((grp + 1) * kGroup < nb) {
                    asm volatile("st.global.wb.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                                 :
                                 : "l"(dst), "r"(xb0), "r"(xb1), "l"(pol_el)
                                 : "memory");
                } else {
                    asm volatile("st.global.wb.L2::cache_hint.u32 [%0], %1, %2;\n"
                                 :
                                 : "l"(dst), "r"(xb0), "l"(pol_el)
                                 : "memory");
                }
            }
        }

        blk0 = next_blk0;
        if (blk0 >= blk_end) break;
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(128, 2) qwen4_qsa_prefill_score_v2(
    const __nv_bfloat16* __restrict__ iq,
    const __nv_bfloat16* __restrict__ ckeys,
    const int* __restrict__ block_table,
    float* __restrict__ scores,
    unsigned int rows,
    unsigned int first_pos,
    unsigned int ld,
    unsigned int page_size) {
    qsa_prefill_score_v2<false>(iq, ckeys, block_table, scores, nullptr, rows, first_pos, ld, 0,
                                page_size);
}

extern "C" __global__ void __launch_bounds__(128, 2) qwen4_qsa_prefill_score_gmax_v2(
    const __nv_bfloat16* __restrict__ iq,
    const __nv_bfloat16* __restrict__ ckeys,
    const int* __restrict__ block_table,
    float* __restrict__ scores,
    unsigned int* __restrict__ gmax,
    unsigned int rows,
    unsigned int first_pos,
    unsigned int ld,
    unsigned int ldg,
    unsigned int page_size) {
    qsa_prefill_score_v2<true>(iq, ckeys, block_table, scores, gmax, rows, first_pos, ld, ldg,
                               page_size);
}

// K11-R9: Register-resident group maxima + warp-aggregated radix histogram +
// fused Step 3 candidate gather & Step 4 Pass 0 histogram + sc candidate compaction
// + early exit when k >= s_sc[2]. Same signature, shared memory footprint, and
// 0-ULP bit-identical output masks as qwen4_qsa_prefill_topk_fused_c3.
extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_prefill_topk_fused_c3_v2(
    const float* __restrict__ scores, const unsigned int* __restrict__ gmax,
    unsigned int* __restrict__ masks, unsigned int rows, unsigned int first_pos,
    unsigned int sparse_row0, unsigned int ld, unsigned int ldg, unsigned int mask_words,
    unsigned int a_words, unsigned int cap_groups) {
    extern __shared__ __align__(16) unsigned int fsm3[];
    __shared__ unsigned int hist[2048];
    __shared__ unsigned int s_warp[8];
    __shared__ unsigned int s_sc[4];
    unsigned int* sg = fsm3;
    unsigned int* ck = fsm3;
    unsigned int* cg = fsm3 + a_words;
    unsigned int* smask = hist;

    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    const unsigned int n = (first_pos + row + 1) / kRatio;
    unsigned int* out = masks + (size_t)row * mask_words;

    if (row < sparse_row0 || n <= kBlockTopk) {
        for (unsigned int w = tid; w < mask_words; w += 256) {
            const unsigned int lo = w * 32;
            unsigned int bits = 0;
            if (lo + 32 <= n) {
                bits = 0xffffffffu;
            } else if (lo < n) {
                bits = (1u << (n - lo)) - 1u;
            }
            out[w] = bits;
        }
        return;
    }
    const float* src = scores + (size_t)(row - sparse_row0) * ld;
    const unsigned int* gsrc = gmax + (size_t)(row - sparse_row0) * ldg;
    const unsigned int ng = (n + kGroup - 1) / kGroup;
    const auto global_key = [&](unsigned int i) -> unsigned int {
        return __float_as_uint(__ldg(src + i)) & kKeyMask;
    };
    if (ng > a_words || mask_words > 2048u || 8u * cap_groups > a_words) {
        const Kth s = radix_kth(global_key, n, kBlockTopk, 0u, hist, s_warp, s_sc);
        write_full_mask(global_key, n, s, out, mask_words);
        return;
    }

    const unsigned int span = ((ng + 255u) / 256u) * 32u;
    const unsigned int g_lo = min(warp * span, ng);
    const unsigned int g_hi = min(g_lo + span, ng);
    unsigned int rg[16];
    unsigned int tau0 = 0;
    unsigned int ncand = ng;

    // 1. Group maxima + tau0 = 512th largest group max.
    //    For ng <= 4096 (up to 128K tokens), keep group maxima in rg[0..15] mapped
    //    to this warp's [g_lo, g_hi) slice so both Step 1 and Step 2 run from registers.
    if (ng <= 4096u) {
        if (ng > kBlockTopk) {
            for (unsigned int b = tid; b < 2048u; b += 256) hist[b] = 0;
            __syncthreads();
        }
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const unsigned int gi = g_lo + static_cast<unsigned int>(s) * 32u + lane;
            unsigned int key = 0u;
            if (gi < g_hi) {
                asm volatile("ld.global.cs.b32 %0, [%1];" : "=r"(key) : "l"(gsrc + gi));
            }
            rg[s] = key;
        }
        if (ng > kBlockTopk) {
#pragma unroll
            for (int s = 0; s < 16; ++s) {
                const unsigned int gi = g_lo + static_cast<unsigned int>(s) * 32u + lane;
                if (gi < g_hi) {
                    hist_add_warp(hist, (rg[s] >> 20u) & 2047u);
                }
            }
        }
        if (ng > kBlockTopk) {
            unsigned int* const sc = hist + 1024;
            unsigned int prefix = 0;
            unsigned int pmask = 0;
            unsigned int k = kBlockTopk;
            unsigned int g_cand = ng;
            for (int pass = 0; pass < 3; ++pass) {
                const unsigned int shift = pass == 0 ? 20u : (pass == 1 ? 10u : 0u);
                const unsigned int bins = pass == 0 ? 2048u : 1024u;
                if (pass == 1) {
                    if (g_cand <= 1024u) {
#pragma unroll
                        for (int s = 0; s < 16; ++s) {
                            const unsigned int gi = g_lo + static_cast<unsigned int>(s) * 32u + lane;
                            if (gi < g_hi && (rg[s] & pmask) == prefix) {
                                atomicAdd(&hist[(rg[s] >> 10u) & 1023u], 1u);
                                const unsigned int slot = atomicAdd(&s_warp[7], 1u);
                                if (slot < 1024u) sc[slot] = rg[s];
                            }
                        }
                    } else {
#pragma unroll
                        for (int s = 0; s < 16; ++s) {
                            const unsigned int gi = g_lo + static_cast<unsigned int>(s) * 32u + lane;
                            if (gi < g_hi && (rg[s] & pmask) == prefix) {
                                atomicAdd(&hist[(rg[s] >> 10u) & 1023u], 1u);
                            }
                        }
                    }
                } else if (pass == 2) {
                    if (g_cand <= 1024u) {
                        for (unsigned int i = tid; i < g_cand; i += 256) {
                            const unsigned int key = sc[i];
                            if ((key & pmask) == prefix) atomicAdd(&hist[key & 1023u], 1u);
                        }
                    } else {
#pragma unroll
                        for (int s = 0; s < 16; ++s) {
                            const unsigned int gi = g_lo + static_cast<unsigned int>(s) * 32u + lane;
                            if (gi < g_hi && (rg[s] & pmask) == prefix) {
                                atomicAdd(&hist[rg[s] & 1023u], 1u);
                            }
                        }
                    }
                }
                __syncthreads();
                const unsigned int per = bins / 256;
                const unsigned int hi = bins - tid * per;
                unsigned int h_local[8];
                unsigned int local = 0;
#pragma unroll
                for (unsigned int j = 0; j < 8; ++j) {
                    const unsigned int c = (j < per) ? hist[hi - 1u - j] : 0u;
                    h_local[j] = c;
                    local += c;
                }
                unsigned int incl = local;
#pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                    const unsigned int v = __shfl_up_sync(0xffffffffu, incl, off);
                    if (lane >= static_cast<unsigned int>(off)) incl += v;
                }
                if (lane == 31 && warp < 7) s_warp[warp] = incl;
                __syncthreads();
                for (unsigned int w = 0; w < warp; ++w) incl += s_warp[w];
                const unsigned int excl = incl - local;
                if (excl < k && k <= incl) {
                    unsigned int above = excl;
                    for (unsigned int j = 0; j < per; ++j) {
                        const unsigned int c = h_local[j];
                        if (above + c >= k) {
                            s_sc[0] = hi - 1u - j;
                            s_sc[1] = above;
                            s_sc[2] = c;
                            break;
                        }
                        above += c;
                    }
                }
                if (pass < 2) {
                    for (unsigned int b = tid; b < 1024u; b += 256) hist[b] = 0;
                    if (pass == 0 && tid == 0) s_warp[7] = 0;
                }
                __syncthreads();
                prefix |= s_sc[0] << shift;
                pmask |= (bins - 1u) << shift;
                k -= s_sc[1];
                if (pass == 0) g_cand = s_sc[2];
                const unsigned int stop_cap = cap_groups < 544u ? cap_groups : 544u;
                if (k >= s_sc[2] || (kBlockTopk - k) + s_sc[2] <= stop_cap) break;
            }
            tau0 = prefix;
            ncand = (kBlockTopk - k) + s_sc[2];
        }
    } else {
        for (unsigned int i = tid; i < ng; i += 256) sg[i] = __ldg(gsrc + i);
        __syncthreads();
        if (ng > kBlockTopk) {
            const auto group_key = [&](unsigned int i) -> unsigned int { return sg[i]; };
            const Kth gk = radix_kth(group_key, ng, kBlockTopk, 0u, hist, s_warp, s_sc);
            tau0 = gk.thr;
            ncand = (kBlockTopk - gk.k_left) + gk.eq_total;
        }
    }
    if (ncand > cap_groups) {
        const Kth s = radix_kth(global_key, n, kBlockTopk, 0u, hist, s_warp, s_sc);
        write_full_mask(global_key, n, s, out, mask_words);
        return;
    }

    // 2. Ordered compaction of candidate groups into cg[0..ncand), plus pre-clearing
    //    hist[0..2048) before the barrier so Step 3 can fuse Step 4 Pass 0.
    unsigned int cnt = 0;
    if (ng <= 4096u) {
        unsigned int bal_s[16];
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const unsigned int gi = g_lo + static_cast<unsigned int>(s) * 32u + lane;
            const unsigned int bal = __ballot_sync(0xffffffffu, gi < g_hi && rg[s] >= tau0);
            bal_s[s] = bal;
            cnt += __popc(bal);
        }
        if (lane == 0) s_warp[warp] = cnt;
        __syncthreads();
        unsigned int pos = 0;
        for (unsigned int w = 0; w < warp; ++w) pos += s_warp[w];
        const unsigned int lane_lt = (1u << lane) - 1u;
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const unsigned int bal = bal_s[s];
            if (bal) {
                if ((bal >> lane) & 1u) {
                    cg[pos + __popc(bal & lane_lt)] = g_lo + static_cast<unsigned int>(s) * 32u + lane;
                }
                pos += __popc(bal);
            }
        }
    } else {
        for (unsigned int base = g_lo; base < g_hi; base += 32) {
            const unsigned int gi = base + lane;
            cnt += __popc(__ballot_sync(0xffffffffu, gi < g_hi && sg[gi] >= tau0));
        }
        if (lane == 0) s_warp[warp] = cnt;
        __syncthreads();
        unsigned int pos = 0;
        for (unsigned int w = 0; w < warp; ++w) pos += s_warp[w];
        for (unsigned int base = g_lo; base < g_hi; base += 32) {
            const unsigned int gi = base + lane;
            const bool take = gi < g_hi && sg[gi] >= tau0;
            const unsigned int bal = __ballot_sync(0xffffffffu, take);
            if (take) cg[pos + __popc(bal & ((1u << lane) - 1u))] = gi;
            pos += __popc(bal);
        }
    }
    for (unsigned int b = tid; b < 2048u; b += 256) hist[b] = 0;
    __syncthreads();

    // 3. Fused candidate score gather (32 B per group) + Step 4 Pass 0 radix histogram.
    //    Stage up to 4 uint4 loads per thread before storing to ck and histogramming.
    const bool last_partial = cg[ncand - 1] == ng - 1;
    const unsigned int m = ncand * kGroup - (last_partial ? ng * kGroup - n : 0u);
    const unsigned int total_v4 = ncand * 2u;
    for (unsigned int base = tid; base < total_v4; base += 1024u) {
        uint4 rv[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const unsigned int i = base + static_cast<unsigned int>(s) * 256u;
            rv[s] = make_uint4(0u, 0u, 0u, 0u);
            if (i < total_v4) {
                const float* g_ptr = src + cg[i >> 1] * kGroup + (i & 1u) * 4u;
                asm volatile("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(rv[s].x), "=r"(rv[s].y), "=r"(rv[s].z), "=r"(rv[s].w)
                             : "l"(g_ptr));
            }
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const unsigned int i = base + static_cast<unsigned int>(s) * 256u;
            if (i < total_v4) {
                const unsigned int k0 = rv[s].x & kKeyMask;
                const unsigned int k1 = rv[s].y & kKeyMask;
                const unsigned int k2 = rv[s].z & kKeyMask;
                const unsigned int k3 = rv[s].w & kKeyMask;
                const unsigned int base_idx = i * 4u;
                *reinterpret_cast<uint4*>(ck + base_idx) = make_uint4(k0, k1, k2, k3);
                if (base_idx + 0u < m && k0 >= tau0) hist_add_warp(hist, k0 >> 20u);
                if (base_idx + 1u < m && k1 >= tau0) hist_add_warp(hist, k1 >> 20u);
                if (base_idx + 2u < m && k2 >= tau0) hist_add_warp(hist, k2 >> 20u);
                if (base_idx + 3u < m && k3 >= tau0) hist_add_warp(hist, k3 >> 20u);
            }
        }
    }
    __syncthreads();

    // 4. Exact selection over the candidates (Pass 0 histogram is already in hist[0..2048)).
    unsigned int* const sc = hist + 1024;
    unsigned int prefix = 0;
    unsigned int pmask = 0;
    unsigned int k = kBlockTopk;
    unsigned int m_cand = m;
    for (int pass = 0; pass < 3; ++pass) {
        const unsigned int shift = pass == 0 ? 20u : (pass == 1 ? 10u : 0u);
        const unsigned int bins = pass == 0 ? 2048u : 1024u;
        if (pass == 1) {
            if (m_cand <= 1024u) {
                for (unsigned int i = tid; i < m; i += 256) {
                    const unsigned int key = ck[i];
                    if (key >= tau0 && (key & pmask) == prefix) {
                        atomicAdd(&hist[(key >> 10u) & 1023u], 1u);
                        const unsigned int slot = atomicAdd(&s_warp[7], 1u);
                        if (slot < 1024u) sc[slot] = key;
                    }
                }
            } else {
                for (unsigned int i = tid; i < m; i += 256) {
                    const unsigned int key = ck[i];
                    if (key >= tau0 && (key & pmask) == prefix) {
                        atomicAdd(&hist[(key >> 10u) & 1023u], 1u);
                    }
                }
            }
            __syncthreads();
        } else if (pass == 2) {
            if (m_cand <= 1024u) {
                for (unsigned int i = tid; i < m_cand; i += 256) {
                    const unsigned int key = sc[i];
                    if ((key & pmask) == prefix) atomicAdd(&hist[key & 1023u], 1u);
                }
            } else {
                for (unsigned int i = tid; i < m; i += 256) {
                    const unsigned int key = ck[i];
                    if (key >= tau0 && (key & pmask) == prefix) {
                        atomicAdd(&hist[key & 1023u], 1u);
                    }
                }
            }
            __syncthreads();
        }
        const unsigned int per = bins / 256;
        const unsigned int hi = bins - tid * per;
        unsigned int h_local[8];
        unsigned int local = 0;
#pragma unroll
        for (unsigned int j = 0; j < 8; ++j) {
            const unsigned int c = (j < per) ? hist[hi - 1u - j] : 0u;
            h_local[j] = c;
            local += c;
        }
        unsigned int incl = local;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const unsigned int v = __shfl_up_sync(0xffffffffu, incl, off);
            if (lane >= static_cast<unsigned int>(off)) incl += v;
        }
        if (lane == 31 && warp < 7) s_warp[warp] = incl;
        __syncthreads();
        for (unsigned int w = 0; w < warp; ++w) incl += s_warp[w];
        const unsigned int excl = incl - local;
        if (excl < k && k <= incl) {
            unsigned int above = excl;
            for (unsigned int j = 0; j < per; ++j) {
                const unsigned int c = h_local[j];
                if (above + c >= k) {
                    s_sc[0] = hi - 1u - j;
                    s_sc[1] = above;
                    s_sc[2] = c;
                    break;
                }
                above += c;
            }
        }
        if (pass < 2) {
            for (unsigned int b = tid; b < 1024u; b += 256) hist[b] = 0;
            if (pass == 0 && tid == 0) s_warp[7] = 0;
        }
        __syncthreads();
        prefix |= s_sc[0] << shift;
        pmask |= (bins - 1u) << shift;
        k -= s_sc[1];
        if (pass == 0) m_cand = s_sc[2];
        if (k >= s_sc[2]) {
            prefix = max(prefix, tau0);
            break;
        }
    }

    // 5. Mask words in the (now free) histogram: zero, OR the selected bits, store.
    for (unsigned int w = tid; w < mask_words; w += 256) smask[w] = 0;
    __syncthreads();
    const unsigned int thr = prefix;
    const unsigned int chunks = (m + 31) / 32;
    if (k >= s_sc[2]) {
        for (unsigned int c = warp; c < chunks; c += 8) {
            const unsigned int i = c * 32 + lane;
            const unsigned int key = i < m ? ck[i] : 0u;
            const unsigned int sel = __ballot_sync(0xffffffffu, i < m && key >= thr);
            or_group_bytes(sel, c, lane, ncand, cg, smask);
        }
    } else if (warp == 0) {
        unsigned int left = k;
        for (unsigned int c = 0; c < chunks; ++c) {
            const unsigned int i = c * 32 + lane;
            const unsigned int key = i < m ? ck[i] : 0u;
            const unsigned int gt = __ballot_sync(0xffffffffu, i < m && key > thr);
            unsigned int eq = __ballot_sync(0xffffffffu, i < m && key == thr);
            unsigned int sel = gt;
            while (eq && left) {
                const unsigned int low = eq & (~eq + 1u);
                sel |= low;
                eq ^= low;
                --left;
            }
            or_group_bytes(sel, c, lane, ncand, cg, smask);
        }
    }
    __syncthreads();
    for (unsigned int w = tid; w < mask_words; w += 256) out[w] = smask[w];
}

