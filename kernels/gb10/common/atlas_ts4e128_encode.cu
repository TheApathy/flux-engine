// SPDX-License-Identifier: AGPL-3.0-only
//
// Atlas-TS4E-128: Transposed K-Paired 4-Bit Scale Encoding with 0xF Escape.
//
// Lossless bit-exact (0-ULP across all 256 byte values 0x00..0xFF) encoder and
// verification decoder for NVFP4 E4M3 block scales, stored directly at each
// expert's existing scale pointer (`ExpertPtrTable::scale_ptrs[e]`).
//
// Contiguous 16-byte-aligned per-expert blob layout for transposed [K/16, N]
// scales (N % 128 == 0, K % 32 == 0, num_k32 = K / 32, n_tiles = N / 128,
// num_bkts = num_k32 * n_tiles, page_bases_len = ((num_bkts >> 7) + 4) & ~3):
//   Offset 0                                : nibbles[num_k32][N]
//       byte = (hi_code << 4) | lo_code; lo = k16 = 2*k32, hi = k16 = 2*k32 + 1
//   Offset nibbles_bytes                    : bases[N]
//       u8 in [0..=241], per-column 15-code window base (lowest-base tie-break)
//   Offset nibbles_bytes + N                : page_bases[page_bases_len]
//       u32 cumulative exception count at page boundary 128*p (16 B for <=511 bkts)
//   Offset nibbles_bytes + N + 4*pb_len     : bkt_off[num_bkts + 1] (u16, 16B padded)
//       relative to page_bases[b >> 7]; max value <= 127*256 = 32,512 < 65,535
//   Offset aligned_meta_end                 : exceptions[num_exc] (u16, 16B padded)
//       lo_byte = ((k16 & 1) << 7) | (n & 127), hi_byte = exact raw u8 scale;
//       strictly sorted by (bkt, local_pos)

#include <stdint.h>

namespace {

__device__ __forceinline__ unsigned int ts4e128_page_bases_len(unsigned int num_bkts) {
    return ((num_bkts >> 7) + 4u) & ~3u;
}

}  // namespace

// Pass 1: Compute optimal per-column bases[n] in [0..=241] (maximizing count
// within [base, base + 14], tie-breaking to lowest base) and exact per-expert
// exception totals from flat row-major [N, K/16] E4M3 source scales.
//
// Launch: grid=[num_experts, 1, 1], block=[256, 1, 1].
extern "C" __global__ void __launch_bounds__(256, 1) atlas_ts4e128_count_batched(
    const unsigned long long* __restrict__ src_ptrs,
    unsigned char* __restrict__ bases_tmp,
    unsigned int* __restrict__ exc_totals,
    unsigned int K,
    unsigned int N,
    unsigned int num_experts
) {
    const unsigned int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const unsigned int tid = threadIdx.x;
    const unsigned char* src = reinterpret_cast<const unsigned char*>(src_ptrs[expert]);
    if (src == nullptr) {
        if (tid == 0u) exc_totals[expert] = 0u;
        return;
    }

    const unsigned int k16_count = K >> 4;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31u;

    __shared__ unsigned int s_hist[8][256];
    __shared__ unsigned int s_warp_exc[8];

    unsigned int warp_exc_sum = 0u;
    unsigned char* dst_bases = bases_tmp + static_cast<unsigned long long>(expert) * N;

    for (unsigned int n = warp_id; n < N; n += 8u) {
        #pragma unroll
        for (unsigned int b = 0u; b < 256u; b += 32u) {
            s_hist[warp_id][b + lane] = 0u;
        }
        __syncwarp();

        const unsigned char* src_row = src + static_cast<unsigned long long>(n) * k16_count;
        for (unsigned int k16 = lane; k16 < k16_count; k16 += 32u) {
            const unsigned int v = static_cast<unsigned int>(src_row[k16]);
            atomicAdd(&s_hist[warp_id][v], 1u);
        }
        __syncwarp();

        // Each lane tests up to 8 consecutive bases [lane*8 .. min(lane*8+7, 241)].
        const unsigned int b0 = lane << 3;
        unsigned int best_cnt = 0u;
        unsigned int best_base = 255u;
        if (b0 <= 241u) {
            unsigned int win = 0u;
            #pragma unroll
            for (unsigned int j = 0u; j < 15u; ++j) {
                win += s_hist[warp_id][b0 + j];
            }
            best_cnt = win;
            best_base = b0;
            const unsigned int b_end = (b0 + 7u <= 241u) ? (b0 + 7u) : 241u;
            for (unsigned int b = b0 + 1u; b <= b_end; ++b) {
                win = win + s_hist[warp_id][b + 14u] - s_hist[warp_id][b - 1u];
                if (win > best_cnt) {
                    best_cnt = win;
                    best_base = b;
                }
            }
        }

        // Warp-reduce (best_cnt max, tie-break lowest best_base).
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const unsigned int oc = __shfl_down_sync(0xffffffffu, best_cnt, off);
            const unsigned int ob = __shfl_down_sync(0xffffffffu, best_base, off);
            if (oc > best_cnt || (oc == best_cnt && ob < best_base)) {
                best_cnt = oc;
                best_base = ob;
            }
        }

        if (lane == 0u) {
            dst_bases[n] = static_cast<unsigned char>(best_base);
            warp_exc_sum += (k16_count - best_cnt);
        }
        __syncwarp();
    }

    if (lane == 0u) {
        s_warp_exc[warp_id] = warp_exc_sum;
    }
    __syncthreads();

    if (tid == 0u) {
        unsigned int total = 0u;
        #pragma unroll
        for (unsigned int w = 0u; w < 8u; ++w) {
            total += s_warp_exc[w];
        }
        exc_totals[expert] = total;
    }
}

// Pass 2: Pack transposed nibbles[K/32][N], bases[N], page_bases, bkt_off, and
// sorted exceptions into `dst_slab + dst_offsets[expert]`.
//
// Launch: grid=[num_experts, 1, 1], block=[256, 1, 1].
extern "C" __global__ void __launch_bounds__(256, 1) atlas_ts4e128_pack_batched(
    const unsigned long long* __restrict__ src_ptrs,
    const unsigned char* __restrict__ bases_tmp,
    unsigned char* __restrict__ dst_slab,
    const unsigned long long* __restrict__ dst_offsets,
    unsigned int K,
    unsigned int N,
    unsigned int num_experts
) {
    const unsigned int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const unsigned char* src = reinterpret_cast<const unsigned char*>(src_ptrs[expert]);
    if (src == nullptr || dst_slab == nullptr) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int k_bit = tid >> 7;
    const unsigned int ln = tid & 127u;

    const unsigned int k16_count = K >> 4;
    const unsigned int num_k32 = K >> 5;
    const unsigned int n_tiles = N >> 7;
    const unsigned int num_bkts = num_k32 * n_tiles;
    const unsigned int nibbles_bytes = num_k32 * N;
    const unsigned int page_bases_len = ts4e128_page_bases_len(num_bkts);
    const unsigned int page_bases_off = nibbles_bytes + N;
    const unsigned int bkt_off_off = page_bases_off + page_bases_len * 4u;
    const unsigned int bkt_off_bytes_aligned = (((num_bkts + 1u) * 2u) + 15u) & ~15u;
    const unsigned int exc_off = bkt_off_off + bkt_off_bytes_aligned;

    unsigned char* dst = dst_slab + dst_offsets[expert];
    const unsigned char* bases_in = bases_tmp + static_cast<unsigned long long>(expert) * N;
    unsigned char* bases_out = dst + nibbles_bytes;
    unsigned int* page_bases_out = reinterpret_cast<unsigned int*>(dst + page_bases_off);
    unsigned short* bkt_off_out = reinterpret_cast<unsigned short*>(dst + bkt_off_off);
    unsigned short* exc_out = reinterpret_cast<unsigned short*>(dst + exc_off);

    for (unsigned int n = tid; n < N; n += 256u) {
        bases_out[n] = bases_in[n];
    }

    __shared__ unsigned char s_lo_code[2][128];
    __shared__ unsigned int s_warp_cnt[2][8];

    unsigned int exc_cursor = 0u;
    unsigned int page_base = 0u;
    unsigned int k32 = 0u;
    unsigned int nt = 0u;

    for (unsigned int bkt = 0u; bkt < num_bkts; ++bkt) {
        const unsigned int n = (nt << 7) + ln;
        const unsigned int k16 = (k32 << 1) + k_bit;
        const unsigned int val = static_cast<unsigned int>(
            src[static_cast<unsigned long long>(n) * k16_count + k16]
        );
        const unsigned int base = static_cast<unsigned int>(bases_in[n]);
        const bool is_exc = (val < base) || (val > base + 14u);
        const unsigned int code = is_exc ? 15u : (val - base);

        const unsigned int buf = bkt & 1u;
        if (tid < 128u) {
            s_lo_code[buf][ln] = static_cast<unsigned char>(code);
        }

        const unsigned int mask = __ballot_sync(0xffffffffu, is_exc);
        const unsigned int rank = static_cast<unsigned int>(__popc(mask & ((1u << lane) - 1u)));
        if (lane == 0u) {
            s_warp_cnt[buf][warp_id] = static_cast<unsigned int>(__popc(mask));
        }
        __syncthreads();

        if (tid >= 128u) {
            const unsigned int lo = static_cast<unsigned int>(s_lo_code[buf][ln]);
            dst[static_cast<unsigned long long>(k32) * N + n] =
                static_cast<unsigned char>((code << 4) | lo);
        }

        unsigned int warp_off = 0u;
        unsigned int bkt_exc_cnt = 0u;
        #pragma unroll
        for (unsigned int w = 0u; w < 8u; ++w) {
            const unsigned int c = s_warp_cnt[buf][w];
            if (w < warp_id) warp_off += c;
            bkt_exc_cnt += c;
        }

        if ((bkt & 127u) == 0u) {
            page_base = exc_cursor;
            if (tid == 0u) {
                page_bases_out[bkt >> 7] = page_base;
            }
        }
        if (tid == 0u) {
            bkt_off_out[bkt] = static_cast<unsigned short>(exc_cursor - page_base);
        }
        if (is_exc) {
            exc_out[exc_cursor + warp_off + rank] =
                static_cast<unsigned short>((val << 8) | tid);
        }
        exc_cursor += bkt_exc_cnt;

        if (++nt == n_tiles) {
            nt = 0u;
            ++k32;
        }
    }

    if ((num_bkts & 127u) == 0u) {
        page_base = exc_cursor;
        if (tid == 0u) {
            page_bases_out[num_bkts >> 7] = page_base;
        }
    }
    if (tid == 0u) {
        bkt_off_out[num_bkts] = static_cast<unsigned short>(exc_cursor - page_base);
        const unsigned int last_page = num_bkts >> 7;
        for (unsigned int p = last_page + 1u; p < page_bases_len; ++p) {
            page_bases_out[p] = 0u;
        }
        unsigned char* bkt_off_bytes = dst + bkt_off_off;
        for (unsigned int b = (num_bkts + 1u) * 2u; b < bkt_off_bytes_aligned; ++b) {
            bkt_off_bytes[b] = 0u;
        }
        const unsigned int exc_bytes = exc_cursor * 2u;
        const unsigned int exc_bytes_aligned = (exc_bytes + 15u) & ~15u;
        unsigned char* exc_bytes_ptr = dst + exc_off;
        for (unsigned int b = exc_bytes; b < exc_bytes_aligned; ++b) {
            exc_bytes_ptr[b] = 0u;
        }
    }
}

// Verification helper: decode one TS4E-128 blob back to flat transposed [K/16, N]
// E4M3 scales. Launch: grid=[num_bkts, 1, 1], block=[256, 1, 1].
extern "C" __global__ void __launch_bounds__(256, 1) atlas_ts4e128_decode_flat(
    const unsigned char* __restrict__ blob,
    unsigned char* __restrict__ dst_transposed,
    unsigned int K,
    unsigned int N
) {
    const unsigned int num_k32 = K >> 5;
    const unsigned int n_tiles = N >> 7;
    const unsigned int num_bkts = num_k32 * n_tiles;
    const unsigned int bkt = blockIdx.x;
    if (bkt >= num_bkts) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int k_bit = tid >> 7;
    const unsigned int ln = tid & 127u;
    const unsigned int k32 = bkt / n_tiles;
    const unsigned int nt = bkt - k32 * n_tiles;
    const unsigned int n = (nt << 7) + ln;
    const unsigned int k16 = (k32 << 1) + k_bit;

    const unsigned int nibbles_bytes = num_k32 * N;
    const unsigned int nib_pair = static_cast<unsigned int>(
        blob[static_cast<unsigned long long>(k32) * N + n]
    );
    const unsigned int code = k_bit ? (nib_pair >> 4) : (nib_pair & 0xFu);

    unsigned int scale_val;
    if (code != 15u) {
        const unsigned int base = static_cast<unsigned int>(blob[nibbles_bytes + n]);
        scale_val = base + code;
    } else {
        const unsigned int page_bases_len = ts4e128_page_bases_len(num_bkts);
        const unsigned int page_bases_off = nibbles_bytes + N;
        const unsigned int bkt_off_off = page_bases_off + page_bases_len * 4u;
        const unsigned int bkt_off_bytes_aligned = (((num_bkts + 1u) * 2u) + 15u) & ~15u;
        const unsigned int exc_off = bkt_off_off + bkt_off_bytes_aligned;

        const unsigned int* page_bases =
            reinterpret_cast<const unsigned int*>(blob + page_bases_off);
        const unsigned short* bkt_off =
            reinterpret_cast<const unsigned short*>(blob + bkt_off_off);
        const unsigned short* exc =
            reinterpret_cast<const unsigned short*>(blob + exc_off);

        const unsigned int beg = page_bases[bkt >> 7] + static_cast<unsigned int>(bkt_off[bkt]);
        const unsigned int end =
            page_bases[(bkt + 1u) >> 7] + static_cast<unsigned int>(bkt_off[bkt + 1u]);

        unsigned int lo = beg;
        unsigned int hi = end;
        while (lo < hi) {
            const unsigned int mid = lo + ((hi - lo) >> 1);
            const unsigned int pos = static_cast<unsigned int>(exc[mid]) & 0xFFu;
            if (pos < tid) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        scale_val = (lo < end) ? (static_cast<unsigned int>(exc[lo]) >> 8) : 0u;
    }

    dst_transposed[static_cast<unsigned long long>(k16) * N + n] =
        static_cast<unsigned char>(scale_val);
}
