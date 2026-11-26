// SPDX-License-Identifier: AGPL-3.0-only

// Device top-m candidate selection for stochastic sampling (SPEED_PLAN W1,
// ATLAS_SAMPLER_DEVICE_TOPK=1) + 2-stage multi-CTA sliced top-k + in-chain
// CRN Gumbel-Max MTP draft selection. CPU specifications:
//   - spark_runtime::sampler::device_topk::topk_candidates_reference
//   - spark_runtime::sampler::device_topk::topk_sliced_candidates_reference
//   - spark_runtime::sampler::gumbel::select_gumbel_shortlist_bf16
//     (note: `mtp_gumbel_argmax_bf16` caps its filtered shortlist at <= 64
//      candidates; callers must enforce `top_k in 1..=64` on filtered paths or
//      fall back when `top_k == 0` with `top_p < 1`/`min_p > 0`, or `top_k > 64`)
//
// Each finite raw logit v at id i gets the unique 64-bit key
//   hi = order-preserving bits of (v / temperature + 0.0f)   (-0 folds to +0)
//   lo = ~i                                                 (lower id wins ties)
// and the kernel writes the m largest keys' (id, raw v) in unspecified order
// plus their count (every finite entry when there are at most m). Division and
// the +0 fold use explicit round-to-nearest intrinsics so the key is bit-equal
// to the host's `v / temperature`. Non-finite raw values are skipped, as the
// host sampler's is_finite filter skips them.

#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>

__device__ __forceinline__ unsigned long long topk_cand_key_f32(
    float v, unsigned int id, float temperature
) {
    if (!isfinite(v)) {
        return 0ull;
    }
    const float k = __fadd_rn(__fdiv_rn(v, temperature), 0.0f);
    const unsigned int bits = __float_as_uint(k);
    const unsigned int ord = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    return ((unsigned long long)ord << 32) | (unsigned long long)(~id);
}

__device__ __forceinline__ unsigned long long topk_cand_key(
    __nv_bfloat16 b, unsigned int id, float temperature
) {
    return topk_cand_key_f32(__bfloat162float(b), id, temperature);
}

extern "C" __global__ void __launch_bounds__(1024) topk_candidates_bf16(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out_ids,
    float* __restrict__ out_vals,
    unsigned int* __restrict__ out_count,
    unsigned int n,
    unsigned int m,
    float temperature
) {
    __shared__ unsigned int s_hist[256];
    __shared__ unsigned long long s_prefix;
    __shared__ unsigned long long s_mask;
    __shared__ unsigned int s_need;
    __shared__ unsigned int s_valid;
    __shared__ unsigned int s_out;

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    if (tid == 0) {
        s_prefix = 0ull;
        s_mask = 0ull;
        s_need = m;
        s_valid = 0u;
        s_out = 0u;
    }
    __syncthreads();

    unsigned int local_valid = 0u;
    for (unsigned int i = tid; i < n; i += stride) {
        local_valid += topk_cand_key(logits[i], i, temperature) != 0ull ? 1u : 0u;
    }
    atomicAdd(&s_valid, local_valid);
    __syncthreads();

    // Every valid key is > 0, so threshold 1 collects all valid entries.
    unsigned long long threshold = 1ull;
    if (s_valid > m) {
        for (int shift = 56; shift >= 0; shift -= 8) {
            for (unsigned int d = tid; d < 256u; d += stride) {
                s_hist[d] = 0u;
            }
            __syncthreads();
            const unsigned long long prefix = s_prefix;
            const unsigned long long mask = s_mask;
            for (unsigned int i = tid; i < n; i += stride) {
                const unsigned long long key = topk_cand_key(logits[i], i, temperature);
                if (key != 0ull && (key & mask) == prefix) {
                    atomicAdd(&s_hist[(unsigned int)(key >> shift) & 0xFFu], 1u);
                }
            }
            __syncthreads();
            if (tid == 0) {
                const unsigned int need = s_need;
                unsigned int above = 0u;
                int digit = 255;
                for (; digit > 0; digit--) {
                    if (above + s_hist[digit] >= need) {
                        break;
                    }
                    above += s_hist[digit];
                }
                s_need = need - above;
                s_prefix = prefix | ((unsigned long long)digit << shift);
                s_mask = mask | (0xFFull << shift);
            }
            __syncthreads();
        }
        threshold = s_prefix;
    }

    for (unsigned int i = tid; i < n; i += stride) {
        const unsigned long long key = topk_cand_key(logits[i], i, temperature);
        if (key != 0ull && key >= threshold) {
            const unsigned int slot = atomicAdd(&s_out, 1u);
            if (slot < m) {
                out_ids[slot] = i;
                out_vals[slot] = __bfloat162float(logits[i]);
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        out_count[0] = s_out;
    }
}

// ── 2-Stage Multi-CTA Sliced Top-K (SW = 8192 cols/CTA, m <= 64) ────────────
//
// Stage 1 scans 8192-column slices in parallel across CTAs (31 CTAs for 248,320
// vocab in 1 wave across GB10's 48 SMs), reading global HBM exactly ONCE into
// 16 KiB shared memory and emitting up to m <= 64 (key, raw_f32) candidates per
// slice. Stage 2 merges the <= num_slices * m candidates on 1 CTA into the same
// (out_ids, out_vals, out_count) layout as `topk_candidates_bf16`.
#define TOPK_SLICE_SW 8192u
#define TOPK_SLICE_M_MAX 64u
#define TOPK_STAGE2_SMEM_CAP 3072u

extern "C" __global__ void __launch_bounds__(256, 4) topk_slice_stage1_bf16(
    const __nv_bfloat16* __restrict__ logits,
    unsigned long long* __restrict__ slice_keys,
    float* __restrict__ slice_vals,
    unsigned int* __restrict__ slice_counts,
    unsigned int n,
    unsigned int m,
    float temperature
) {
    __shared__ unsigned short s_raw[TOPK_SLICE_SW];
    __shared__ unsigned int s_hist[256];
    __shared__ unsigned long long s_prefix;
    __shared__ unsigned long long s_mask;
    __shared__ unsigned int s_need;
    __shared__ unsigned int s_valid;
    __shared__ unsigned int s_out;

    const unsigned int tid = threadIdx.x;
    const unsigned int cta = blockIdx.x;
    const unsigned int slice_start = cta * TOPK_SLICE_SW;
    if (slice_start >= n) {
        if (tid == 0) {
            slice_counts[cta] = 0u;
        }
        return;
    }
    const unsigned int slice_len =
        (n - slice_start < TOPK_SLICE_SW) ? (n - slice_start) : TOPK_SLICE_SW;
    if (m > TOPK_SLICE_M_MAX) {
        m = TOPK_SLICE_M_MAX;
    }

    s_hist[tid] = 0u;
    if (tid == 0) {
        s_prefix = 0ull;
        s_mask = 0ull;
        s_need = m;
        s_valid = 0u;
        s_out = 0u;
    }
    __syncthreads();

    // Single coalesced HBM read pass + fused top-byte (shift = 56) histogram.
    const unsigned short* __restrict__ raw_in =
        reinterpret_cast<const unsigned short*>(logits) + slice_start;
    unsigned int local_valid = 0u;
    for (unsigned int j = tid; j < slice_len; j += 256u) {
        const unsigned short raw_bits = raw_in[j];
        s_raw[j] = raw_bits;
        const float v = __uint_as_float(((unsigned int)raw_bits) << 16);
        const unsigned long long key = topk_cand_key_f32(v, slice_start + j, temperature);
        if (key != 0ull) {
            local_valid += 1u;
            atomicAdd(&s_hist[(unsigned int)(key >> 56) & 0xFFu], 1u);
        }
    }
    atomicAdd(&s_valid, local_valid);
    __syncthreads();

    unsigned long long threshold = 1ull;
    if (s_valid > m) {
        // Resolve shift = 56 from the fused pass.
        if (tid == 0) {
            const unsigned int need = s_need;
            unsigned int above = 0u;
            int digit = 255;
            for (; digit > 0; digit--) {
                if (above + s_hist[digit] >= need) {
                    break;
                }
                above += s_hist[digit];
            }
            s_need = need - above;
            s_prefix = ((unsigned long long)digit) << 56;
            s_mask = 0xFFull << 56;
        }
        __syncthreads();

        // Remaining 7 radix digits read exclusively from 16 KiB SMEM `s_raw`.
        for (int shift = 48; shift >= 0; shift -= 8) {
            s_hist[tid] = 0u;
            __syncthreads();
            const unsigned long long prefix = s_prefix;
            const unsigned long long mask = s_mask;
            for (unsigned int j = tid; j < slice_len; j += 256u) {
                const float v = __uint_as_float(((unsigned int)s_raw[j]) << 16);
                const unsigned long long key = topk_cand_key_f32(v, slice_start + j, temperature);
                if (key != 0ull && (key & mask) == prefix) {
                    atomicAdd(&s_hist[(unsigned int)(key >> shift) & 0xFFu], 1u);
                }
            }
            __syncthreads();
            if (tid == 0) {
                const unsigned int need = s_need;
                unsigned int above = 0u;
                int digit = 255;
                for (; digit > 0; digit--) {
                    if (above + s_hist[digit] >= need) {
                        break;
                    }
                    above += s_hist[digit];
                }
                s_need = need - above;
                s_prefix = prefix | ((unsigned long long)digit << shift);
                s_mask = mask | (0xFFull << shift);
            }
            __syncthreads();
        }
        threshold = s_prefix;
    }

    unsigned long long* __restrict__ out_k = slice_keys + (size_t)cta * TOPK_SLICE_M_MAX;
    float* __restrict__ out_v = slice_vals + (size_t)cta * TOPK_SLICE_M_MAX;
    for (unsigned int j = tid; j < slice_len; j += 256u) {
        const float v = __uint_as_float(((unsigned int)s_raw[j]) << 16);
        const unsigned long long key = topk_cand_key_f32(v, slice_start + j, temperature);
        if (key != 0ull && key >= threshold) {
            const unsigned int slot = atomicAdd(&s_out, 1u);
            if (slot < m) {
                out_k[slot] = key;
                out_v[slot] = v;
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        slice_counts[cta] = s_out;
    }
}

extern "C" __global__ void __launch_bounds__(256, 4) topk_slice_stage2_merge(
    const unsigned long long* __restrict__ slice_keys,
    const float* __restrict__ slice_vals,
    const unsigned int* __restrict__ slice_counts,
    unsigned int* __restrict__ out_ids,
    float* __restrict__ out_vals,
    unsigned int* __restrict__ out_count,
    unsigned int num_slices,
    unsigned int m
) {
    // 24 KiB keys + 6 KiB BF16 raw bits + 1 KiB histogram = 31 KiB (< 48 KiB static SMEM cap).
    __shared__ unsigned long long s_keys[TOPK_STAGE2_SMEM_CAP];
    __shared__ unsigned short s_raw_bf16[TOPK_STAGE2_SMEM_CAP];
    __shared__ unsigned int s_hist[256];
    __shared__ unsigned long long s_prefix;
    __shared__ unsigned long long s_mask;
    __shared__ unsigned int s_need;
    __shared__ unsigned int s_total;
    __shared__ unsigned int s_out;

    const unsigned int tid = threadIdx.x;
    if (m > TOPK_SLICE_M_MAX) {
        m = TOPK_SLICE_M_MAX;
    }

    if (tid == 0) {
        s_prefix = 0ull;
        s_mask = 0ull;
        s_need = m;
        s_total = 0u;
        s_out = 0u;
    }
    __syncthreads();

    // Compact valid candidates across slices.
    const unsigned int max_slots = num_slices * TOPK_SLICE_M_MAX;
    const bool use_smem = (max_slots <= TOPK_STAGE2_SMEM_CAP);
    if (use_smem) {
        for (unsigned int idx = tid; idx < max_slots; idx += 256u) {
            const unsigned int s = idx / TOPK_SLICE_M_MAX;
            const unsigned int k_idx = idx % TOPK_SLICE_M_MAX;
            if (k_idx < slice_counts[s]) {
                const unsigned long long key = slice_keys[idx];
                if (key != 0ull) {
                    const unsigned int dst = atomicAdd(&s_total, 1u);
                    s_keys[dst] = key;
                    s_raw_bf16[dst] = (unsigned short)(__float_as_uint(slice_vals[idx]) >> 16);
                }
            }
        }
    } else {
        unsigned int local_cnt = 0u;
        for (unsigned int idx = tid; idx < max_slots; idx += 256u) {
            const unsigned int s = idx / TOPK_SLICE_M_MAX;
            const unsigned int k_idx = idx % TOPK_SLICE_M_MAX;
            if (k_idx < slice_counts[s] && slice_keys[idx] != 0ull) {
                local_cnt += 1u;
            }
        }
        atomicAdd(&s_total, local_cnt);
    }
    __syncthreads();

    const unsigned int total = s_total;
    unsigned long long threshold = 1ull;
    if (total > m) {
        for (int shift = 56; shift >= 0; shift -= 8) {
            s_hist[tid] = 0u;
            __syncthreads();
            const unsigned long long prefix = s_prefix;
            const unsigned long long mask = s_mask;
            if (use_smem) {
                for (unsigned int i = tid; i < total; i += 256u) {
                    const unsigned long long key = s_keys[i];
                    if ((key & mask) == prefix) {
                        atomicAdd(&s_hist[(unsigned int)(key >> shift) & 0xFFu], 1u);
                    }
                }
            } else {
                for (unsigned int idx = tid; idx < max_slots; idx += 256u) {
                    const unsigned int s = idx / TOPK_SLICE_M_MAX;
                    const unsigned int k_idx = idx % TOPK_SLICE_M_MAX;
                    if (k_idx < slice_counts[s]) {
                        const unsigned long long key = slice_keys[idx];
                        if (key != 0ull && (key & mask) == prefix) {
                            atomicAdd(&s_hist[(unsigned int)(key >> shift) & 0xFFu], 1u);
                        }
                    }
                }
            }
            __syncthreads();
            if (tid == 0) {
                const unsigned int need = s_need;
                unsigned int above = 0u;
                int digit = 255;
                for (; digit > 0; digit--) {
                    if (above + s_hist[digit] >= need) {
                        break;
                    }
                    above += s_hist[digit];
                }
                s_need = need - above;
                s_prefix = prefix | ((unsigned long long)digit << shift);
                s_mask = mask | (0xFFull << shift);
            }
            __syncthreads();
        }
        threshold = s_prefix;
    }

    if (use_smem) {
        for (unsigned int i = tid; i < total; i += 256u) {
            const unsigned long long key = s_keys[i];
            if (key >= threshold) {
                const unsigned int slot = atomicAdd(&s_out, 1u);
                if (slot < m) {
                    out_ids[slot] = ~((unsigned int)(key & 0xFFFFFFFFull));
                    out_vals[slot] = __uint_as_float(((unsigned int)s_raw_bf16[i]) << 16);
                }
            }
        }
    } else {
        for (unsigned int idx = tid; idx < max_slots; idx += 256u) {
            const unsigned int s = idx / TOPK_SLICE_M_MAX;
            const unsigned int k_idx = idx % TOPK_SLICE_M_MAX;
            if (k_idx < slice_counts[s]) {
                const unsigned long long key = slice_keys[idx];
                if (key != 0ull && key >= threshold) {
                    const unsigned int slot = atomicAdd(&s_out, 1u);
                    if (slot < m) {
                        out_ids[slot] = ~((unsigned int)(key & 0xFFFFFFFFull));
                        out_vals[slot] = slice_vals[idx];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        out_count[0] = s_out;
    }
}

// ── In-Chain CRN Noise-Coupled Gumbel-Max MTP Draft Selection ───────────────
//
// Matches `spark_runtime::sampler::gumbel::{key_bits, gumbel_f32}` bit-for-bit:
//   base(seed, pos) = mix(mix(seed + 0x9E3779B97F4A7C15) ^ (pos * 0xD1B54A32D192ED03))
//   key_bits        = mix(base ^ token_id)
//   u24             = min((key_bits >> 40) * 2^-24 + 2^-25, 1 - 2^-24)
//   gumbel_f32      = -logf(-logf(u24))
__device__ __forceinline__ unsigned long long splitmix64_mix_dev(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ unsigned long long gumbel_pos_base_dev(
    unsigned long long seed, unsigned long long abs_pos
) {
    return splitmix64_mix_dev(
        splitmix64_mix_dev(seed + 0x9E3779B97F4A7C15ull) ^ (abs_pos * 0xD1B54A32D192ED03ull)
    );
}

__device__ __forceinline__ float gumbel_from_base_f32_dev(
    unsigned long long pos_base, unsigned int token_id
) {
    const unsigned long long bits = splitmix64_mix_dev(pos_base ^ (unsigned long long)token_id);
    const float u = fminf((float)(bits >> 40) * 0x1.0p-24f + 0x1.0p-25f, 0x1.fffffep-1f);
    return -logf(-logf(u));
}

// ── Fast 2-Pass 16-Bit BF16 Histogram & Warp-Aggregated Selection Helpers ───
//
// Maps every finite BF16 value onto an order-preserving uint16_t key in
// [0x0080, 0xFF7F] with -0.0 folded onto +0.0 (0x8000). Two 8-bit passes
// (upper byte -> winning bin -> lower byte) find the exact top-k BF16 threshold
// with zero shared-memory atomics using per-warp 256-bin histograms and
// convergent `__ballot_sync` + `__match_any_sync` aggregation.

__device__ __forceinline__ bool bf16_is_finite_u16(uint16_t raw) {
    return (raw & 0x7F80u) != 0x7F80u;
}

__device__ __forceinline__ uint16_t bf16_to_ordered_u16(uint16_t raw) {
    raw = (raw == 0x8000u) ? 0x0000u : raw;
    const uint16_t mask = (raw & 0x8000u) ? 0xFFFFu : 0x8000u;
    return (uint16_t)(raw ^ mask);
}

__device__ __forceinline__ uint16_t ordered_u16_to_bf16(uint16_t ord) {
    return (ord & 0x8000u) ? (uint16_t)(ord ^ 0x8000u) : (uint16_t)(~ord);
}

__device__ __forceinline__ float bf16_u16_to_f32(uint16_t raw) {
    return __uint_as_float(((uint32_t)raw) << 16);
}

__device__ __forceinline__ void hist_inc_warp_convergent(
    unsigned int* __restrict__ warp_hist,
    unsigned int lane,
    bool valid,
    uint32_t bin
) {
    const unsigned int valid_mask = __ballot_sync(0xFFFFFFFFu, valid);
    if (!valid) {
        return;
    }
    const unsigned int peers = __match_any_sync(valid_mask, bin);
    const unsigned int leader = (unsigned int)__ffs((int)peers) - 1u;
    if (lane == leader) {
        warp_hist[bin] += (unsigned int)__popc(peers);
    }
}

__device__ __forceinline__ void hist_pass1_pair_convergent(
    unsigned int* __restrict__ warp_hist,
    unsigned int lane,
    uint32_t packed2,
    bool compute_max,
    float& local_max
) {
    const uint16_t r0 = (uint16_t)(packed2 & 0xFFFFu);
    const bool f0 = bf16_is_finite_u16(r0);
    const uint32_t b0 = f0 ? ((uint32_t)bf16_to_ordered_u16(r0) >> 8) : 0u;
    if (compute_max && f0) {
        local_max = fmaxf(local_max, bf16_u16_to_f32(r0));
    }
    hist_inc_warp_convergent(warp_hist, lane, f0, b0);

    const uint16_t r1 = (uint16_t)(packed2 >> 16);
    const bool f1 = bf16_is_finite_u16(r1);
    const uint32_t b1 = f1 ? ((uint32_t)bf16_to_ordered_u16(r1) >> 8) : 0u;
    if (compute_max && f1) {
        local_max = fmaxf(local_max, bf16_u16_to_f32(r1));
    }
    hist_inc_warp_convergent(warp_hist, lane, f1, b1);
}

__device__ __forceinline__ void hist_pass2_pair_convergent(
    unsigned int* __restrict__ warp_hist,
    unsigned int lane,
    uint32_t packed2,
    uint32_t win_hi
) {
    const uint16_t r0 = (uint16_t)(packed2 & 0xFFFFu);
    const bool f0 = bf16_is_finite_u16(r0);
    const uint16_t k0 = f0 ? bf16_to_ordered_u16(r0) : 0u;
    const bool h0 = f0 && (((uint32_t)k0 >> 8) == win_hi);
    hist_inc_warp_convergent(warp_hist, lane, h0, h0 ? ((uint32_t)k0 & 0xFFu) : 0u);

    const uint16_t r1 = (uint16_t)(packed2 >> 16);
    const bool f1 = bf16_is_finite_u16(r1);
    const uint16_t k1 = f1 ? bf16_to_ordered_u16(r1) : 0u;
    const bool h1 = f1 && (((uint32_t)k1 >> 8) == win_hi);
    hist_inc_warp_convergent(warp_hist, lane, h1, h1 ? ((uint32_t)k1 & 0xFFu) : 0u);
}

__device__ __forceinline__ void block_reduce_argmax_score_id(
    float* __restrict__ s_val,
    unsigned int* __restrict__ s_id,
    unsigned int tid,
    float local_score,
    unsigned int local_id
) {
    s_val[tid] = local_score;
    s_id[tid] = local_id;
    __syncthreads();
    for (unsigned int s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            const float sv = s_val[tid + s];
            const unsigned int sid = s_id[tid + s];
            const float mv = s_val[tid];
            const unsigned int mid = s_id[tid];
            if (sv > mv || (sv == mv && sid < mid)) {
                s_val[tid] = sv;
                s_id[tid] = sid;
            }
        }
        __syncthreads();
    }
}

struct Bf16Topk2PassMeta {
    uint32_t total_valid;
    uint32_t win_hi;
    uint32_t above_hi;
    uint32_t need_in_hi;
    uint32_t thresh_key;
    uint32_t strict_gt;
    uint32_t need_eq;
    uint32_t count_eq;
    uint32_t run_pass2;
    uint32_t topk_max_id;
    uint32_t p_win_hi;
    uint32_t p_valid;
    float row_max;
    float total_prob_sum;
    float p_cum_above_hi;
};

// 2-pass 16-bit BF16 threshold finder for a 256-thread CTA (8 warps).
// Computes `s_meta.{total_valid, thresh_key, strict_gt, need_eq, count_eq}`
// and (when `compute_max == true`) `s_meta.row_max`.
__device__ __forceinline__ void bf16_topk_threshold_2pass(
    const uint16_t* __restrict__ row_raw,
    uint32_t vocab,
    uint32_t k,
    bool compute_max,
    unsigned int (&s_hist)[8][256],
    float (&s_reduce_val)[256],
    Bf16Topk2PassMeta& s_meta
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const bool aligned8 = ((((uintptr_t)row_raw) & 15u) == 0u);

    #pragma unroll
    for (unsigned int w = 0u; w < 8u; w++) {
        s_hist[w][tid] = 0u;
    }
    __syncthreads();

    // Pass 1: upper 8 bits of order-preserving BF16 key (+ optional row_max).
    float local_max = -CUDART_INF_F;
    if (aligned8) {
        const uint32_t n_vec = vocab >> 3;
        const uint32_t vec_end = n_vec << 3;
        const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw);
        for (uint32_t base = 0u; base < n_vec; base += 256u) {
            const uint32_t idx = base + tid;
            const uint4 v4 = (idx < n_vec)
                ? row_v4[idx]
                : make_uint4(0x7FC07FC0u, 0x7FC07FC0u, 0x7FC07FC0u, 0x7FC07FC0u);
            hist_pass1_pair_convergent(s_hist[warp], lane, v4.x, compute_max, local_max);
            hist_pass1_pair_convergent(s_hist[warp], lane, v4.y, compute_max, local_max);
            hist_pass1_pair_convergent(s_hist[warp], lane, v4.z, compute_max, local_max);
            hist_pass1_pair_convergent(s_hist[warp], lane, v4.w, compute_max, local_max);
        }
        if (vec_end < vocab) {
            const uint32_t c = vec_end + tid;
            const uint16_t raw = (c < vocab) ? row_raw[c] : 0x7FC0u;
            const bool f = (c < vocab) && bf16_is_finite_u16(raw);
            const uint32_t b = f ? ((uint32_t)bf16_to_ordered_u16(raw) >> 8) : 0u;
            if (compute_max && f) {
                local_max = fmaxf(local_max, bf16_u16_to_f32(raw));
            }
            hist_inc_warp_convergent(s_hist[warp], lane, f, b);
        }
    } else {
        for (uint32_t base = 0u; base < vocab; base += 256u) {
            const uint32_t c = base + tid;
            const uint16_t raw = (c < vocab) ? row_raw[c] : 0x7FC0u;
            const bool f = (c < vocab) && bf16_is_finite_u16(raw);
            const uint32_t b = f ? ((uint32_t)bf16_to_ordered_u16(raw) >> 8) : 0u;
            if (compute_max && f) {
                local_max = fmaxf(local_max, bf16_u16_to_f32(raw));
            }
            hist_inc_warp_convergent(s_hist[warp], lane, f, b);
        }
    }
    __syncthreads();

    unsigned int bin_total = 0u;
    #pragma unroll
    for (unsigned int w = 0u; w < 8u; w++) {
        bin_total += s_hist[w][tid];
    }
    s_hist[0][tid] = bin_total;
    if (compute_max) {
        s_reduce_val[tid] = local_max;
    }
    __syncthreads();

    if (compute_max) {
        for (unsigned int s = 128u; s > 0u; s >>= 1u) {
            if (tid < s) {
                s_reduce_val[tid] = fmaxf(s_reduce_val[tid], s_reduce_val[tid + s]);
            }
            __syncthreads();
        }
    }

    if (tid == 0u) {
        if (compute_max) {
            s_meta.row_max = s_reduce_val[0];
        }
        uint32_t total = 0u;
        for (int b = 255; b >= 0; b--) {
            total += s_hist[0][b];
        }
        s_meta.total_valid = total;
        if (total == 0u || k == 0u || total <= k) {
            s_meta.win_hi = 0u;
            s_meta.above_hi = 0u;
            s_meta.need_in_hi = 0u;
            s_meta.thresh_key = 0u;
            s_meta.strict_gt = total;
            s_meta.need_eq = 0u;
            s_meta.count_eq = 0u;
            s_meta.run_pass2 = 0u;
        } else {
            uint32_t above = 0u;
            int hi = 255;
            for (; hi > 0; hi--) {
                if (above + s_hist[0][hi] >= k) {
                    break;
                }
                above += s_hist[0][hi];
            }
            s_meta.win_hi = (uint32_t)hi;
            s_meta.above_hi = above;
            s_meta.need_in_hi = k - above;
            if (above + s_hist[0][hi] == k) {
                s_meta.thresh_key = ((uint32_t)hi) << 8;
                s_meta.strict_gt = k;
                s_meta.need_eq = 0u;
                s_meta.count_eq = 0u;
                s_meta.run_pass2 = 0u;
            } else {
                s_meta.run_pass2 = 1u;
            }
        }
    }
    __syncthreads();

    // Pass 2: lower 8 bits inside the winning upper bin (skipped on exact fit / k >= valid).
    if (s_meta.run_pass2 != 0u) {
        const uint32_t win_hi = s_meta.win_hi;
        #pragma unroll
        for (unsigned int w = 0u; w < 8u; w++) {
            s_hist[w][tid] = 0u;
        }
        __syncthreads();

        if (aligned8) {
            const uint32_t n_vec = vocab >> 3;
            const uint32_t vec_end = n_vec << 3;
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw);
            for (uint32_t base = 0u; base < n_vec; base += 256u) {
                const uint32_t idx = base + tid;
                const uint4 v4 = (idx < n_vec)
                    ? row_v4[idx]
                    : make_uint4(0x7FC07FC0u, 0x7FC07FC0u, 0x7FC07FC0u, 0x7FC07FC0u);
                hist_pass2_pair_convergent(s_hist[warp], lane, v4.x, win_hi);
                hist_pass2_pair_convergent(s_hist[warp], lane, v4.y, win_hi);
                hist_pass2_pair_convergent(s_hist[warp], lane, v4.z, win_hi);
                hist_pass2_pair_convergent(s_hist[warp], lane, v4.w, win_hi);
            }
            if (vec_end < vocab) {
                const uint32_t c = vec_end + tid;
                const uint16_t raw = (c < vocab) ? row_raw[c] : 0x7FC0u;
                const bool f = (c < vocab) && bf16_is_finite_u16(raw);
                const uint16_t k16 = f ? bf16_to_ordered_u16(raw) : 0u;
                const bool hit = f && (((uint32_t)k16 >> 8) == win_hi);
                hist_inc_warp_convergent(s_hist[warp], lane, hit, hit ? ((uint32_t)k16 & 0xFFu) : 0u);
            }
        } else {
            for (uint32_t base = 0u; base < vocab; base += 256u) {
                const uint32_t c = base + tid;
                const uint16_t raw = (c < vocab) ? row_raw[c] : 0x7FC0u;
                const bool f = (c < vocab) && bf16_is_finite_u16(raw);
                const uint16_t k16 = f ? bf16_to_ordered_u16(raw) : 0u;
                const bool hit = f && (((uint32_t)k16 >> 8) == win_hi);
                hist_inc_warp_convergent(s_hist[warp], lane, hit, hit ? ((uint32_t)k16 & 0xFFu) : 0u);
            }
        }
        __syncthreads();

        unsigned int lo_total = 0u;
        #pragma unroll
        for (unsigned int w = 0u; w < 8u; w++) {
            lo_total += s_hist[w][tid];
        }
        s_hist[0][tid] = lo_total;
        __syncthreads();

        if (tid == 0u) {
            const uint32_t need = s_meta.need_in_hi;
            uint32_t above_lo = 0u;
            int lo = 255;
            for (; lo > 0; lo--) {
                if (above_lo + s_hist[0][lo] >= need) {
                    break;
                }
                above_lo += s_hist[0][lo];
            }
            s_meta.thresh_key = (s_meta.win_hi << 8) | (uint32_t)lo;
            s_meta.strict_gt = s_meta.above_hi + above_lo;
            s_meta.need_eq = need - above_lo;
            s_meta.count_eq = s_hist[0][lo];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void mtp_emit_cand(
    uint16_t raw,
    uint32_t c,
    const unsigned int* __restrict__ shortlist_ids,
    uint16_t thresh_key,
    float eff_temp,
    uint32_t k_cap,
    unsigned int* __restrict__ s_out,
    unsigned long long* __restrict__ s_cand_key,
    float* __restrict__ s_cand_scaled
) {
    if (!bf16_is_finite_u16(raw)) {
        return;
    }
    const uint16_t k16 = bf16_to_ordered_u16(raw);
    if (k16 >= thresh_key) {
        const unsigned int slot = atomicAdd(s_out, 1u);
        if (slot < k_cap) {
            const float v = bf16_u16_to_f32(raw);
            const unsigned int gid = (shortlist_ids != nullptr) ? shortlist_ids[c] : c;
            s_cand_key[slot] = topk_cand_key_f32(v, gid, eff_temp);
            s_cand_scaled[slot] = __fdiv_rn(v, eff_temp);
        }
    }
}

// Filtered Gumbel-Max argmax over BF16 logits.
// NOTE: The filtered path (`top_k`, `min_p`, or `top_p`) uses a fixed 64-entry
// SMEM shortlist (`k_cap = (top_k > 0u && top_k <= 64u) ? top_k : 64u`),
// diverging from `select_gumbel_shortlist_bf16` whenever `top_k == 0` (with
// `top_p < 1.0` or `min_p > 0.0`) or `top_k > 64`. Callers must restrict
// filtered launches to `1 <= top_k <= 64` or fall back.
extern "C" __global__ void __launch_bounds__(256, 2) mtp_gumbel_argmax_bf16(
    const __nv_bfloat16* __restrict__ logits,
    const unsigned int* __restrict__ shortlist_ids,
    const unsigned long long* __restrict__ seeds,
    const unsigned long long* __restrict__ positions,
    unsigned int* __restrict__ argmax_out,
    float* __restrict__ prob_out,
    unsigned int vocab,
    unsigned long long seed0,
    unsigned long long pos0,
    float temperature,
    unsigned int top_k,
    float top_p,
    float min_p
) {
    // 8 KiB union + 2 KiB reduce + scalars = ~10.2 KiB static SMEM (<= 16 KiB).
    __shared__ union {
        unsigned int hist[8][256];
        float reduce_logit[256];
        struct {
            unsigned long long key[64];
            float scaled[64];
            float weight[64];
        } cand;
    } s_work;
    __shared__ float s_reduce_val[256];
    __shared__ unsigned int s_reduce_id[256];
    __shared__ unsigned int s_warp_cnt[8];
    __shared__ Bf16Topk2PassMeta s_meta;
    __shared__ float s_sum_exp;
    __shared__ unsigned int s_out;
    __shared__ unsigned int s_eq_collected;
    __shared__ unsigned int s_gid_prefix;
    __shared__ unsigned int s_gid_mask;
    __shared__ unsigned int s_gid_need;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int row = blockIdx.x;
    const uint16_t* __restrict__ row_raw =
        reinterpret_cast<const uint16_t*>(logits) + (size_t)row * (size_t)vocab;
    const unsigned long long seed = (seeds != nullptr) ? seeds[row] : seed0;
    const unsigned long long abs_pos = (positions != nullptr) ? positions[row] : pos0;
    const unsigned long long pos_base = gumbel_pos_base_dev(seed, abs_pos);
    const float eff_temp = fmaxf(temperature, 1.0e-6f);

    if (vocab == 0u) {
        if (tid == 0u) {
            argmax_out[row] = 0u;
            if (prob_out != nullptr) {
                prob_out[row] = 0.0f;
            }
        }
        return;
    }

    const bool untruncated =
        ((top_k == 0u || top_k >= vocab) && min_p <= 0.0f && top_p >= 1.0f);
    if (untruncated) {
        if (prob_out != nullptr) {
            float local_max = -CUDART_INF_F;
            for (unsigned int c = tid; c < vocab; c += 256u) {
                const uint16_t raw = row_raw[c];
                if (bf16_is_finite_u16(raw)) {
                    local_max = fmaxf(local_max, bf16_u16_to_f32(raw));
                }
            }
            s_reduce_val[tid] = local_max;
            __syncthreads();
            for (unsigned int s = 128u; s > 0u; s >>= 1u) {
                if (tid < s) {
                    s_reduce_val[tid] = fmaxf(s_reduce_val[tid], s_reduce_val[tid + s]);
                }
                __syncthreads();
            }
            const float row_max = s_reduce_val[0];
            __syncthreads();
            float local_sum = 0.0f;
            float best_score = -CUDART_INF_F;
            float best_v = -CUDART_INF_F;
            unsigned int best_gid = 0xFFFFFFFFu;
            for (unsigned int c = tid; c < vocab; c += 256u) {
                const uint16_t raw = row_raw[c];
                if (!bf16_is_finite_u16(raw)) {
                    continue;
                }
                const float v = bf16_u16_to_f32(raw);
                if (isfinite(row_max)) {
                    local_sum += expf(__fdiv_rn(v - row_max, eff_temp));
                }
                const unsigned int gid = (shortlist_ids != nullptr) ? shortlist_ids[c] : c;
                const float score = __fdiv_rn(v, eff_temp) + gumbel_from_base_f32_dev(pos_base, gid);
                if (score > best_score || (score == best_score && gid < best_gid)) {
                    best_score = score;
                    best_gid = gid;
                    best_v = v;
                }
            }
            s_reduce_val[tid] = local_sum;
            __syncthreads();
            for (unsigned int s = 128u; s > 0u; s >>= 1u) {
                if (tid < s) {
                    s_reduce_val[tid] += s_reduce_val[tid + s];
                }
                __syncthreads();
            }
            if (tid == 0u) {
                s_sum_exp = s_reduce_val[0];
            }
            __syncthreads();
            s_reduce_val[tid] = best_score;
            s_reduce_id[tid] = best_gid;
            s_work.reduce_logit[tid] = best_v;
            __syncthreads();
            for (unsigned int s = 128u; s > 0u; s >>= 1u) {
                if (tid < s) {
                    const float sv = s_reduce_val[tid + s];
                    const unsigned int sid = s_reduce_id[tid + s];
                    const float mv = s_reduce_val[tid];
                    const unsigned int mid = s_reduce_id[tid];
                    if (sv > mv || (sv == mv && sid < mid)) {
                        s_reduce_val[tid] = sv;
                        s_reduce_id[tid] = sid;
                        s_work.reduce_logit[tid] = s_work.reduce_logit[tid + s];
                    }
                }
                __syncthreads();
            }
            if (tid == 0u) {
                const unsigned int winner = s_reduce_id[0];
                argmax_out[row] = (winner == 0xFFFFFFFFu)
                    ? ((shortlist_ids != nullptr) ? shortlist_ids[0] : 0u)
                    : winner;
                const float denom = s_sum_exp;
                if (winner != 0xFFFFFFFFu && denom > 0.0f && isfinite(denom) && isfinite(row_max)) {
                    const float p = __fdiv_rn(
                        expf(__fdiv_rn(s_work.reduce_logit[0] - row_max, eff_temp)),
                        denom
                    );
                    prob_out[row] = isfinite(p) ? fminf(fmaxf(p, 0.0f), 1.0f) : 0.0f;
                } else {
                    prob_out[row] = 0.0f;
                }
            }
            return;
        }

        // Fast single-pass untruncated Gumbel argmax (no histogram, no softmax denom).
        float best_score = -CUDART_INF_F;
        unsigned int best_gid = 0xFFFFFFFFu;
        for (unsigned int c = tid; c < vocab; c += 256u) {
            const uint16_t raw = row_raw[c];
            if (!bf16_is_finite_u16(raw)) {
                continue;
            }
            const float v = bf16_u16_to_f32(raw);
            const unsigned int gid = (shortlist_ids != nullptr) ? shortlist_ids[c] : c;
            const float score = __fdiv_rn(v, eff_temp) + gumbel_from_base_f32_dev(pos_base, gid);
            if (score > best_score || (score == best_score && gid < best_gid)) {
                best_score = score;
                best_gid = gid;
            }
        }
        block_reduce_argmax_score_id(s_reduce_val, s_reduce_id, tid, best_score, best_gid);
        if (tid == 0u) {
            argmax_out[row] = (s_reduce_id[0] == 0xFFFFFFFFu)
                ? ((shortlist_ids != nullptr) ? shortlist_ids[0] : 0u)
                : s_reduce_id[0];
        }
        return;
    }

    // Filtered path (top_k / min_p / top_p): fast 2-pass 16-bit BF16 threshold
    // selecting up to 64 top candidates by exact (value desc, global_id asc).
    const unsigned int k_cap = (top_k > 0u && top_k <= 64u) ? top_k : 64u;
    bf16_topk_threshold_2pass(
        row_raw,
        vocab,
        k_cap,
        prob_out != nullptr,
        s_work.hist,
        s_reduce_val,
        s_meta
    );

    if (prob_out != nullptr) {
        const float row_max = s_meta.row_max;
        float local_sum = 0.0f;
        if (isfinite(row_max)) {
            for (unsigned int c = tid; c < vocab; c += 256u) {
                const uint16_t raw = row_raw[c];
                if (bf16_is_finite_u16(raw)) {
                    local_sum += expf(__fdiv_rn(bf16_u16_to_f32(raw) - row_max, eff_temp));
                }
            }
        }
        s_reduce_val[tid] = local_sum;
        __syncthreads();
        for (unsigned int s = 128u; s > 0u; s >>= 1u) {
            if (tid < s) {
                s_reduce_val[tid] += s_reduce_val[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            s_sum_exp = s_reduce_val[0];
        }
        __syncthreads();
    }

    if (s_meta.total_valid == 0u) {
        if (tid == 0u) {
            argmax_out[row] = (shortlist_ids != nullptr) ? shortlist_ids[0] : 0u;
            if (prob_out != nullptr) {
                prob_out[row] = 0.0f;
            }
        }
        return;
    }

    const uint16_t thresh_key = (uint16_t)s_meta.thresh_key;
    const uint32_t strict_gt = s_meta.strict_gt;
    const uint32_t need_eq = s_meta.need_eq;
    const bool need_tiebreak = (s_meta.count_eq > need_eq);

    // If boundary ties exist with an arbitrary `shortlist_ids` table, resolve exact
    // `need_eq`-th smallest `gid` while `s_work.hist` is still live.
    uint32_t gid_thresh = 0xFFFFFFFFu;
    if (need_tiebreak && shortlist_ids != nullptr) {
        if (tid == 0u) {
            s_gid_prefix = 0u;
            s_gid_mask = 0u;
            s_gid_need = need_eq;
        }
        __syncthreads();
        for (int shift = 24; shift >= 0; shift -= 8) {
            #pragma unroll
            for (unsigned int w = 0u; w < 8u; w++) {
                s_work.hist[w][tid] = 0u;
            }
            __syncthreads();
            const uint32_t prefix = s_gid_prefix;
            const uint32_t mask = s_gid_mask;
            for (uint32_t base = 0u; base < vocab; base += 256u) {
                const uint32_t c = base + tid;
                bool hit = false;
                uint32_t digit = 0u;
                if (c < vocab) {
                    const uint16_t raw = row_raw[c];
                    if (bf16_is_finite_u16(raw) && bf16_to_ordered_u16(raw) == thresh_key) {
                        const uint32_t gid = shortlist_ids[c];
                        if ((gid & mask) == prefix) {
                            hit = true;
                            digit = (gid >> shift) & 0xFFu;
                        }
                    }
                }
                hist_inc_warp_convergent(s_work.hist[warp], lane, hit, digit);
            }
            __syncthreads();
            unsigned int d_total = 0u;
            #pragma unroll
            for (unsigned int w = 0u; w < 8u; w++) {
                d_total += s_work.hist[w][tid];
            }
            s_work.hist[0][tid] = d_total;
            __syncthreads();
            if (tid == 0u) {
                const uint32_t need = s_gid_need;
                uint32_t below = 0u;
                int d = 0;
                for (; d < 255; d++) {
                    if (below + s_work.hist[0][d] >= need) {
                        break;
                    }
                    below += s_work.hist[0][d];
                }
                s_gid_need = need - below;
                s_gid_prefix = prefix | ((uint32_t)d << shift);
                s_gid_mask = mask | (0xFFu << shift);
            }
            __syncthreads();
        }
        gid_thresh = s_gid_prefix;
    }

    // Pass 3: collect exact top `min(total_valid, k_cap)` candidates into `s_work.cand`.
    if (tid == 0u) {
        s_out = 0u;
        s_eq_collected = 0u;
    }
    __syncthreads();

    if (!need_tiebreak) {
        const bool aligned8 = ((((uintptr_t)row_raw) & 15u) == 0u);
        if (aligned8) {
            const uint32_t n_vec = vocab >> 3;
            const uint32_t vec_end = n_vec << 3;
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw);
            for (uint32_t idx = tid; idx < n_vec; idx += 256u) {
                const uint4 v4 = row_v4[idx];
                const uint32_t c0 = idx << 3;
                mtp_emit_cand((uint16_t)(v4.x & 0xFFFFu), c0 + 0u, shortlist_ids, thresh_key, eff_temp, k_cap, &s_out, s_work.cand.key, s_work.cand.scaled);
                mtp_emit_cand((uint16_t)(v4.x >> 16),     c0 + 1u, shortlist_ids, thresh_key, eff_temp, k_cap, &s_out, s_work.cand.key, s_work.cand.scaled);
                mtp_emit_cand((uint16_t)(v4.y & 0xFFFFu), c0 + 2u, shortlist_ids, thresh_key, eff_temp, k_cap, &s_out, s_work.cand.key, s_work.cand.scaled);
                mtp_emit_cand((uint16_t)(v4.y >> 16),     c0 + 3u, shortlist_ids, thresh_key, eff_temp, k_cap, &s_out, s_work.cand.key, s_work.cand.scaled);
                mtp_emit_cand((uint16_t)(v4.z & 0xFFFFu), c0 + 4u, shortlist_ids, thresh_key, eff_temp, k_cap, &s_out, s_work.cand.key, s_work.cand.scaled);
                mtp_emit_cand((uint16_t)(v4.z >> 16),     c0 + 5u, shortlist_ids, thresh_key, eff_temp, k_cap, &s_out, s_work.cand.key, s_work.cand.scaled);
                mtp_emit_cand((uint16_t)(v4.w & 0xFFFFu), c0 + 6u, shortlist_ids, thresh_key, eff_temp, k_cap, &s_out, s_work.cand.key, s_work.cand.scaled);
                mtp_emit_cand((uint16_t)(v4.w >> 16),     c0 + 7u, shortlist_ids, thresh_key, eff_temp, k_cap, &s_out, s_work.cand.key, s_work.cand.scaled);
            }
            for (uint32_t c = vec_end + tid; c < vocab; c += 256u) {
                mtp_emit_cand(row_raw[c], c, shortlist_ids, thresh_key, eff_temp, k_cap, &s_out, s_work.cand.key, s_work.cand.scaled);
            }
        } else {
            for (uint32_t c = tid; c < vocab; c += 256u) {
                mtp_emit_cand(row_raw[c], c, shortlist_ids, thresh_key, eff_temp, k_cap, &s_out, s_work.cand.key, s_work.cand.scaled);
            }
        }
    } else if (shortlist_ids == nullptr) {
        for (uint32_t base = 0u; base < vocab; base += 256u) {
            const uint32_t c = base + tid;
            bool hit_eq = false;
            uint16_t raw = 0u;
            if (c < vocab) {
                raw = row_raw[c];
                if (bf16_is_finite_u16(raw)) {
                    const uint16_t k16 = bf16_to_ordered_u16(raw);
                    if (k16 > thresh_key) {
                        const unsigned int slot = atomicAdd(&s_out, 1u);
                        if (slot < strict_gt) {
                            const float v = bf16_u16_to_f32(raw);
                            s_work.cand.key[slot] = topk_cand_key_f32(v, c, eff_temp);
                            s_work.cand.scaled[slot] = __fdiv_rn(v, eff_temp);
                        }
                    } else if (k16 == thresh_key && s_eq_collected < need_eq) {
                        hit_eq = true;
                    }
                }
            }
            if (__syncthreads_or(hit_eq)) {
                const unsigned int wm = __ballot_sync(0xFFFFFFFFu, hit_eq);
                if (lane == 0u) {
                    s_warp_cnt[warp] = (unsigned int)__popc(wm);
                }
                __syncthreads();
                const unsigned int prev_eq = s_eq_collected;
                if (hit_eq) {
                    unsigned int offset = 0u;
                    #pragma unroll
                    for (unsigned int w = 0u; w < 8u; w++) {
                        if (w < warp) {
                            offset += s_warp_cnt[w];
                        }
                    }
                    const unsigned int lane_mask = (1u << lane) - 1u;
                    const unsigned int r = prev_eq + offset + (unsigned int)__popc(wm & lane_mask);
                    if (r < need_eq) {
                        const unsigned int slot = strict_gt + r;
                        const float v = bf16_u16_to_f32(raw);
                        s_work.cand.key[slot] = topk_cand_key_f32(v, c, eff_temp);
                        s_work.cand.scaled[slot] = __fdiv_rn(v, eff_temp);
                    }
                }
                __syncthreads();
                if (tid == 0u) {
                    unsigned int chunk_eq = 0u;
                    #pragma unroll
                    for (unsigned int w = 0u; w < 8u; w++) {
                        chunk_eq += s_warp_cnt[w];
                    }
                    s_eq_collected = prev_eq + chunk_eq;
                }
                __syncthreads();
            }
        }
    } else {
        for (uint32_t c = tid; c < vocab; c += 256u) {
            const uint16_t raw = row_raw[c];
            if (!bf16_is_finite_u16(raw)) {
                continue;
            }
            const uint16_t k16 = bf16_to_ordered_u16(raw);
            if (k16 > thresh_key) {
                const unsigned int slot = atomicAdd(&s_out, 1u);
                if (slot < k_cap) {
                    const float v = bf16_u16_to_f32(raw);
                    const unsigned int gid = shortlist_ids[c];
                    s_work.cand.key[slot] = topk_cand_key_f32(v, gid, eff_temp);
                    s_work.cand.scaled[slot] = __fdiv_rn(v, eff_temp);
                }
            } else if (k16 == thresh_key) {
                const unsigned int gid = shortlist_ids[c];
                if (gid <= gid_thresh) {
                    const unsigned int slot = atomicAdd(&s_out, 1u);
                    if (slot < k_cap) {
                        const float v = bf16_u16_to_f32(raw);
                        s_work.cand.key[slot] = topk_cand_key_f32(v, gid, eff_temp);
                        s_work.cand.scaled[slot] = __fdiv_rn(v, eff_temp);
                    }
                }
            }
        }
    }
    __syncthreads();

    if (tid == 0u) {
        unsigned int count = (need_tiebreak && shortlist_ids == nullptr)
            ? (strict_gt + (s_eq_collected < need_eq ? s_eq_collected : need_eq))
            : (s_out < k_cap ? s_out : k_cap);
        if (count > k_cap) {
            count = k_cap;
        }
        // Insertion sort <= 64 candidates descending by `s_work.cand.key`.
        for (unsigned int i = 1u; i < count; i++) {
            const unsigned long long kk = s_work.cand.key[i];
            const float sc = s_work.cand.scaled[i];
            int j = (int)i - 1;
            while (j >= 0 && s_work.cand.key[j] < kk) {
                s_work.cand.key[j + 1] = s_work.cand.key[j];
                s_work.cand.scaled[j + 1] = s_work.cand.scaled[j];
                j--;
            }
            s_work.cand.key[j + 1] = kk;
            s_work.cand.scaled[j + 1] = sc;
        }
        if (count == 0u) {
            argmax_out[row] = (shortlist_ids != nullptr) ? shortlist_ids[0] : 0u;
            if (prob_out != nullptr) {
                prob_out[row] = 0.0f;
            }
            return;
        }
        const float max_sc = s_work.cand.scaled[0];
        for (unsigned int i = 0u; i < count; i++) {
            s_work.cand.weight[i] = expf(s_work.cand.scaled[i] - max_sc);
        }
        if (min_p > 0.0f) {
            const float min_thresh = min_p * s_work.cand.weight[0];
            unsigned int kept = 0u;
            for (unsigned int i = 0u; i < count; i++) {
                if (s_work.cand.weight[i] >= min_thresh) {
                    s_work.cand.key[kept] = s_work.cand.key[i];
                    s_work.cand.scaled[kept] = s_work.cand.scaled[i];
                    s_work.cand.weight[kept] = s_work.cand.weight[i];
                    kept++;
                }
            }
            if (kept > 0u) {
                count = kept;
            }
        }
        if (top_p < 1.0f && count > 0u) {
            float sum = 0.0f;
            for (unsigned int i = 0u; i < count; i++) {
                sum += s_work.cand.weight[i];
            }
            if (sum > 0.0f) {
                float cum = 0.0f;
                for (unsigned int i = 0u; i < count; i++) {
                    cum += __fdiv_rn(s_work.cand.weight[i], sum);
                    if (cum >= top_p) {
                        count = i + 1u;
                        break;
                    }
                }
            }
        }
        unsigned int best_gid = ~((unsigned int)(s_work.cand.key[0] & 0xFFFFFFFFull));
        float best_score = -CUDART_INF_F;
        float best_weight = s_work.cand.weight[0];
        for (unsigned int i = 0u; i < count; i++) {
            const unsigned int gid = ~((unsigned int)(s_work.cand.key[i] & 0xFFFFFFFFull));
            const float score = s_work.cand.scaled[i] + gumbel_from_base_f32_dev(pos_base, gid);
            if (score > best_score || (score == best_score && gid < best_gid)) {
                best_score = score;
                best_gid = gid;
                best_weight = s_work.cand.weight[i];
            }
        }
        argmax_out[row] = best_gid;
        if (prob_out != nullptr) {
            const float denom = s_sum_exp;
            if (denom > 0.0f && isfinite(denom)) {
                const float p = __fdiv_rn(best_weight, denom);
                prob_out[row] = isfinite(p) ? fminf(fmaxf(p, 0.0f), 1.0f) : 0.0f;
            } else {
                prob_out[row] = 0.0f;
            }
        }
    }
}

// ── Multi-Row Target Verifier CRN Gumbel-Max Sampler (248K BF16 logits) ─────
//
// Unified 3-way per-row temperature contract (1 CTA of 512 threads / 16 warps per row):
//   - `temp < 0.0f`  : instant skip sentinel (`out_tokens[row] = 0xFFFFFFFFu`, 0 logits reads)
//   - `temp == 0.0f` : fast 1-pass `uint4` exact greedy argmax (matches `argmax_bf16_rows`)
//   - `temp > 0.0f`  : fast 2-pass `uint4` per-warp SMEM-atomic radix top_k + exact top_p +
//                      SplitMix64 keyed Gumbel-Max (with in-SMEM exact tie-breaking).

#define TARGET_VERIFY_SMEM_MAX_K 512u
#define TARGET_VERIFY_GATHER_CAP 2560u
#define TARGET_VERIFY_SKIP_SENTINEL 0xFFFFFFFFu

__device__ __forceinline__ bool bf16_is_non_nan_u16(uint16_t raw) {
    return (raw & 0x7FFFu) <= 0x7F80u;
}

__device__ __forceinline__ bool target_cand_comes_before(
    uint16_t ord_a, uint32_t id_a, uint16_t ord_b, uint32_t id_b
) {
    return (ord_a > ord_b) || (ord_a == ord_b && id_a < id_b);
}

__device__ __forceinline__ void target_greedy_update(
    uint16_t raw,
    uint32_t idx,
    uint32_t& best_ord,
    uint32_t& best_id
) {
    if (bf16_is_non_nan_u16(raw)) {
        const uint32_t ord = (uint32_t)bf16_to_ordered_u16(raw);
        if (ord > best_ord || (ord == best_ord && idx < best_id)) {
            best_ord = ord;
            best_id = idx;
        }
    }
}

__device__ __forceinline__ void target_max_finite_ord_update(
    uint16_t raw,
    uint32_t& best_ord
) {
    if (bf16_is_finite_u16(raw)) {
        const uint32_t ord = (uint32_t)bf16_to_ordered_u16(raw);
        if (ord > best_ord) {
            best_ord = ord;
        }
    }
}

__device__ __forceinline__ void target_score_if_ge(
    uint16_t raw,
    uint32_t c,
    uint16_t thresh_key,
    float eff_temp,
    uint64_t pos_base,
    float& best_score,
    uint32_t& best_id
) {
    if (!bf16_is_finite_u16(raw)) {
        return;
    }
    const uint16_t k16 = bf16_to_ordered_u16(raw);
    if (k16 >= thresh_key) {
        const float v = bf16_u16_to_f32(raw);
        const float score = __fdiv_rn(v, eff_temp) + gumbel_from_base_f32_dev(pos_base, c);
        if (score > best_score || (score == best_score && c < best_id)) {
            best_score = score;
            best_id = c;
        }
    }
}

__device__ __forceinline__ void target_pass1_pair(
    unsigned int* __restrict__ warp_hist,
    uint32_t packed2
) {
    const uint16_t r0 = (uint16_t)(packed2 & 0xFFFFu);
    if (bf16_is_finite_u16(r0)) {
        atomicAdd(&warp_hist[(uint32_t)bf16_to_ordered_u16(r0) >> 8], 1u);
    }
    const uint16_t r1 = (uint16_t)(packed2 >> 16);
    if (bf16_is_finite_u16(r1)) {
        atomicAdd(&warp_hist[(uint32_t)bf16_to_ordered_u16(r1) >> 8], 1u);
    }
}

__device__ __forceinline__ void target_pass2_fine_pair(
    unsigned int* __restrict__ warp_hist,
    uint32_t packed2,
    uint32_t thresh_hi
) {
    const uint16_t r0 = (uint16_t)(packed2 & 0xFFFFu);
    if (bf16_is_finite_u16(r0)) {
        const uint16_t ord0 = bf16_to_ordered_u16(r0);
        if (((uint32_t)ord0 >> 8) == thresh_hi) {
            atomicAdd(&warp_hist[(uint32_t)ord0 & 0xFFu], 1u);
        }
    }
    const uint16_t r1 = (uint16_t)(packed2 >> 16);
    if (bf16_is_finite_u16(r1)) {
        const uint16_t ord1 = bf16_to_ordered_u16(r1);
        if (((uint32_t)ord1 >> 8) == thresh_hi) {
            atomicAdd(&warp_hist[(uint32_t)ord1 & 0xFFu], 1u);
        }
    }
}

__device__ __forceinline__ void target_emit_finite(
    uint16_t raw,
    uint32_t idx,
    uint32_t cap,
    unsigned int* __restrict__ s_above_cnt,
    unsigned short* __restrict__ s_above_ord,
    unsigned int* __restrict__ s_above_idx
) {
    if (!bf16_is_finite_u16(raw)) {
        return;
    }
    const uint32_t slot = atomicAdd(s_above_cnt, 1u);
    if (slot < cap) {
        s_above_ord[slot] = bf16_to_ordered_u16(raw);
        s_above_idx[slot] = idx;
    }
}

__device__ __forceinline__ void target_pass2_gather_elem(
    uint16_t raw,
    uint32_t idx,
    uint32_t thresh_hi,
    uint32_t above_cap,
    unsigned int* __restrict__ s_above_cnt,
    unsigned short* __restrict__ s_above_ord,
    unsigned int* __restrict__ s_above_idx,
    unsigned int* __restrict__ s_cand_cnt,
    unsigned char* __restrict__ s_cand_lo,
    unsigned int* __restrict__ s_cand_idx,
    unsigned int* __restrict__ s_fine
) {
    if (!bf16_is_finite_u16(raw)) {
        return;
    }
    const uint16_t ord = bf16_to_ordered_u16(raw);
    const uint32_t hi = (uint32_t)(ord >> 8);
    if (hi < thresh_hi) {
        return;
    }
    if (hi > thresh_hi) {
        const uint32_t slot = atomicAdd(s_above_cnt, 1u);
        if (slot < above_cap) {
            s_above_ord[slot] = ord;
            s_above_idx[slot] = idx;
        }
    } else {
        const uint32_t lo = (uint32_t)(ord & 0xFFu);
        const uint32_t slot = atomicAdd(s_cand_cnt, 1u);
        if (slot < TARGET_VERIFY_GATHER_CAP) {
            s_cand_lo[slot] = (unsigned char)lo;
            s_cand_idx[slot] = idx;
        }
        atomicAdd(&s_fine[lo], 1u);
    }
}

__device__ __forceinline__ void target_pass2_hist_elem(
    unsigned int* __restrict__ warp_hist,
    uint16_t raw,
    uint32_t idx,
    uint32_t thresh_hi,
    uint32_t above_cap,
    unsigned int* __restrict__ s_above_cnt,
    unsigned short* __restrict__ s_above_ord,
    unsigned int* __restrict__ s_above_idx
) {
    if (!bf16_is_finite_u16(raw)) {
        return;
    }
    const uint16_t ord = bf16_to_ordered_u16(raw);
    const uint32_t hi = (uint32_t)(ord >> 8);
    if (hi < thresh_hi) {
        return;
    }
    if (hi > thresh_hi) {
        const uint32_t slot = atomicAdd(s_above_cnt, 1u);
        if (slot < above_cap) {
            s_above_ord[slot] = ord;
            s_above_idx[slot] = idx;
        }
    } else {
        atomicAdd(&warp_hist[(uint32_t)ord & 0xFFu], 1u);
    }
}

__device__ __forceinline__ void target_pass3_notie_elem(
    uint16_t raw,
    uint32_t idx,
    uint32_t thresh_hi,
    uint32_t thresh_lo,
    uint32_t above_cap,
    unsigned int* __restrict__ s_above_cnt,
    unsigned short* __restrict__ s_above_ord,
    unsigned int* __restrict__ s_above_idx
) {
    if (!bf16_is_finite_u16(raw)) {
        return;
    }
    const uint16_t ord = bf16_to_ordered_u16(raw);
    if (((uint32_t)ord >> 8) == thresh_hi && ((uint32_t)ord & 0xFFu) >= thresh_lo) {
        const uint32_t slot = atomicAdd(s_above_cnt, 1u);
        if (slot < above_cap) {
            s_above_ord[slot] = ord;
            s_above_idx[slot] = idx;
        }
    }
}

__device__ __forceinline__ void target_prob_hi_elem(
    uint16_t raw,
    uint32_t c,
    uint16_t topk_thresh_key,
    uint32_t topk_max_id,
    float eff_temp,
    float max_sc,
    float* __restrict__ s_prob_hi
) {
    if (!bf16_is_finite_u16(raw)) {
        return;
    }
    const uint16_t k16 = bf16_to_ordered_u16(raw);
    if (k16 < topk_thresh_key || (k16 == topk_thresh_key && c > topk_max_id)) {
        return;
    }
    const float w = expf(__fdiv_rn(bf16_u16_to_f32(raw), eff_temp) - max_sc);
    if (w > 0.0f) {
        atomicAdd(&s_prob_hi[(uint32_t)k16 >> 8], w);
    }
}

__device__ __forceinline__ void target_prob_lo_elem(
    unsigned int* __restrict__ warp_hist,
    uint16_t raw,
    uint32_t c,
    uint16_t topk_thresh_key,
    uint32_t topk_max_id,
    uint32_t p_hi
) {
    if (!bf16_is_finite_u16(raw)) {
        return;
    }
    const uint16_t k16 = bf16_to_ordered_u16(raw);
    if ((k16 > topk_thresh_key || (k16 == topk_thresh_key && c <= topk_max_id))
        && (((uint32_t)k16 >> 8) == p_hi)) {
        atomicAdd(&warp_hist[(uint32_t)k16 & 0xFFu], 1u);
    }
}

__device__ __forceinline__ void target_score_masked_ge(
    uint16_t raw,
    uint32_t c,
    uint16_t topk_thresh_key,
    uint32_t topk_max_id,
    uint16_t p_thresh_key,
    float eff_temp,
    uint64_t pos_base,
    float& best_score,
    uint32_t& best_id
) {
    if (!bf16_is_finite_u16(raw)) {
        return;
    }
    const uint16_t k16 = bf16_to_ordered_u16(raw);
    if (k16 < topk_thresh_key || (k16 == topk_thresh_key && c > topk_max_id)) {
        return;
    }
    if (k16 >= p_thresh_key) {
        const float v = bf16_u16_to_f32(raw);
        const float score = __fdiv_rn(v, eff_temp) + gumbel_from_base_f32_dev(pos_base, c);
        if (score > best_score || (score == best_score && c < best_id)) {
            best_score = score;
            best_id = c;
        }
    }
}

__device__ __forceinline__ void block_reduce_argmax_u32_id_512(
    uint32_t* __restrict__ s_warp_ord,
    uint32_t* __restrict__ s_warp_id,
    unsigned int warp,
    unsigned int lane,
    uint32_t& best_ord,
    uint32_t& best_id
) {
    #pragma unroll
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        const uint32_t o_ord = __shfl_down_sync(0xFFFFFFFFu, best_ord, offset);
        const uint32_t o_id = __shfl_down_sync(0xFFFFFFFFu, best_id, offset);
        if (o_ord > best_ord || (o_ord == best_ord && o_id < best_id)) {
            best_ord = o_ord;
            best_id = o_id;
        }
    }
    if (lane == 0u) {
        s_warp_ord[warp] = best_ord;
        s_warp_id[warp] = best_id;
    }
    __syncthreads();
    if (warp == 0u) {
        best_ord = (lane < 16u) ? s_warp_ord[lane] : 0u;
        best_id = (lane < 16u) ? s_warp_id[lane] : 0xFFFFFFFFu;
        #pragma unroll
        for (unsigned int offset = 8u; offset > 0u; offset >>= 1u) {
            const uint32_t o_ord = __shfl_down_sync(0xFFFFFFFFu, best_ord, offset);
            const uint32_t o_id = __shfl_down_sync(0xFFFFFFFFu, best_id, offset);
            if (o_ord > best_ord || (o_ord == best_ord && o_id < best_id)) {
                best_ord = o_ord;
                best_id = o_id;
            }
        }
    }
}

__device__ __forceinline__ void block_reduce_argmax_score_id_512(
    float* __restrict__ s_warp_val,
    uint32_t* __restrict__ s_warp_id,
    unsigned int warp,
    unsigned int lane,
    float& best_score,
    uint32_t& best_id
) {
    #pragma unroll
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        const float o_val = __shfl_down_sync(0xFFFFFFFFu, best_score, offset);
        const uint32_t o_id = __shfl_down_sync(0xFFFFFFFFu, best_id, offset);
        if (o_val > best_score || (o_val == best_score && o_id < best_id)) {
            best_score = o_val;
            best_id = o_id;
        }
    }
    if (lane == 0u) {
        s_warp_val[warp] = best_score;
        s_warp_id[warp] = best_id;
    }
    __syncthreads();
    if (warp == 0u) {
        best_score = (lane < 16u) ? s_warp_val[lane] : -CUDART_INF_F;
        best_id = (lane < 16u) ? s_warp_id[lane] : 0xFFFFFFFFu;
        #pragma unroll
        for (unsigned int offset = 8u; offset > 0u; offset >>= 1u) {
            const float o_val = __shfl_down_sync(0xFFFFFFFFu, best_score, offset);
            const uint32_t o_id = __shfl_down_sync(0xFFFFFFFFu, best_id, offset);
            if (o_val > best_score || (o_val == best_score && o_id < best_id)) {
                best_score = o_val;
                best_id = o_id;
            }
        }
    }
}

__device__ __forceinline__ uint32_t block_reduce_max_u32_512(
    uint32_t* __restrict__ s_warp_ord,
    unsigned int warp,
    unsigned int lane,
    uint32_t local_max
) {
    #pragma unroll
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        const uint32_t other = __shfl_down_sync(0xFFFFFFFFu, local_max, offset);
        if (other > local_max) {
            local_max = other;
        }
    }
    if (lane == 0u) {
        s_warp_ord[warp] = local_max;
    }
    __syncthreads();
    if (warp == 0u) {
        uint32_t w_max = (lane < 16u) ? s_warp_ord[lane] : 0u;
        #pragma unroll
        for (unsigned int offset = 8u; offset > 0u; offset >>= 1u) {
            const uint32_t other = __shfl_down_sync(0xFFFFFFFFu, w_max, offset);
            if (other > w_max) {
                w_max = other;
            }
        }
        if (lane == 0u) {
            s_warp_ord[0] = w_max;
        }
    }
    __syncthreads();
    return s_warp_ord[0];
}

extern "C" __global__ void __launch_bounds__(512, 3) target_gumbel_verify_rows_bf16(
    const uint16_t* __restrict__ logits,       // [num_rows, vocab_stride] BF16
    const uint64_t* __restrict__ seeds,        // [num_rows] per-row base seed
    const uint64_t* __restrict__ positions,    // [num_rows] per-row abs_pos
    const float*    __restrict__ temperatures, // [num_rows] per-row temperature (<0 skip, ==0 greedy, >0 Gumbel)
    const uint32_t* __restrict__ top_ks,       // [num_rows] per-row top_k (0 = full vocab)
    const float*    __restrict__ top_ps,       // [num_rows] per-row top_p (<= 0 or >= 1 -> disabled)
    uint32_t*       __restrict__ out_tokens,   // [num_rows] sampled token IDs (or 0xFFFFFFFFu if skipped)
    uint32_t vocab_size,                       // logical vocab (e.g., 248320)
    uint32_t vocab_stride                      // row stride in BF16 elements
) {
    // 16 KiB union + 3 KiB winners + ~0.4 KiB warp/meta = ~19.4 KiB static SMEM
    // (fits >= 2-3 CTAs/SM on SM121a's 99 KiB SMEM).
    __shared__ union {
        unsigned int hist[16][256];
        struct {
            unsigned int cand_idx[TARGET_VERIFY_GATHER_CAP];
            unsigned int fine[256];
            unsigned char cand_lo[TARGET_VERIFY_GATHER_CAP];
        } gather;
        struct {
            float scaled[TARGET_VERIFY_SMEM_MAX_K];
            float weight[TARGET_VERIFY_SMEM_MAX_K];
        } top_p;
        float prob_hi[256];
    } s_work;
    __shared__ unsigned int s_above_idx[TARGET_VERIFY_SMEM_MAX_K];
    __shared__ unsigned short s_above_ord[TARGET_VERIFY_SMEM_MAX_K];
    __shared__ union {
        float val[16];
        uint32_t ord[16];
    } s_warp_primary;
    __shared__ uint32_t s_warp_id[16];
    __shared__ uint32_t s_warp_cnt[16];
    __shared__ Bf16Topk2PassMeta s_meta;
    __shared__ uint64_t s_pos_base;
    __shared__ uint32_t s_thresh_hi_cnt;
    __shared__ uint32_t s_above_cnt;
    __shared__ uint32_t s_cand_cnt;
    __shared__ uint32_t s_eq_cnt;
    __shared__ uint32_t s_kept;
    __shared__ uint32_t s_id_prefix;
    __shared__ uint32_t s_id_mask;
    __shared__ uint32_t s_id_need;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const uint32_t row = blockIdx.x;
    const float temp_raw = temperatures[row];

    // Mode 1: `temp < 0.0f` -> instant 0-cycle skip sentinel (0 logits reads).
    if (temp_raw < 0.0f) {
        if (tid == 0u) {
            out_tokens[row] = TARGET_VERIFY_SKIP_SENTINEL;
        }
        return;
    }

    if (vocab_size == 0u) {
        if (tid == 0u) {
            out_tokens[row] = 0u;
        }
        return;
    }

    const uint16_t* __restrict__ row_raw = logits + (size_t)row * (size_t)vocab_stride;

    // Align `row_raw + prologue` to 16 bytes for `uint4` loads (8 BF16/load).
    const unsigned int misalign_bytes = (unsigned int)((uintptr_t)row_raw & 15u);
    const bool can_vec = ((misalign_bytes & 1u) == 0u);
    uint32_t prologue = 0u;
    uint32_t n_vec = 0u;
    uint32_t vec_end = 0u;
    if (can_vec) {
        prologue = (misalign_bytes == 0u) ? 0u : ((16u - misalign_bytes) >> 1);
        if (prologue > vocab_size) {
            prologue = vocab_size;
        }
        const uint32_t rem = vocab_size - prologue;
        n_vec = rem >> 3;
        vec_end = prologue + (n_vec << 3);
    }

    // Mode 2: `temp == 0.0f` (or non-finite defensive fallback) -> fast 1-pass `uint4`
    // 512-thread exact greedy argmax (highest BF16 value, lowest token_id on ties, NaN skipped).
    if (!(temp_raw > 0.0f) || !isfinite(temp_raw)) {
        uint32_t best_ord = 0u;
        uint32_t best_id = 0xFFFFFFFFu;
        if (tid < prologue) {
            target_greedy_update(row_raw[tid], tid, best_ord, best_id);
        }
        if (n_vec > 0u) {
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
            for (uint32_t idx = tid; idx < n_vec; idx += 512u) {
                const uint4 v4 = row_v4[idx];
                const uint32_t c0 = prologue + (idx << 3);
                target_greedy_update((uint16_t)(v4.x & 0xFFFFu), c0 + 0u, best_ord, best_id);
                target_greedy_update((uint16_t)(v4.x >> 16),     c0 + 1u, best_ord, best_id);
                target_greedy_update((uint16_t)(v4.y & 0xFFFFu), c0 + 2u, best_ord, best_id);
                target_greedy_update((uint16_t)(v4.y >> 16),     c0 + 3u, best_ord, best_id);
                target_greedy_update((uint16_t)(v4.z & 0xFFFFu), c0 + 4u, best_ord, best_id);
                target_greedy_update((uint16_t)(v4.z >> 16),     c0 + 5u, best_ord, best_id);
                target_greedy_update((uint16_t)(v4.w & 0xFFFFu), c0 + 6u, best_ord, best_id);
                target_greedy_update((uint16_t)(v4.w >> 16),     c0 + 7u, best_ord, best_id);
            }
        }
        for (uint32_t c = vec_end + tid; c < vocab_size; c += 512u) {
            target_greedy_update(row_raw[c], c, best_ord, best_id);
        }
        block_reduce_argmax_u32_id_512(
            s_warp_primary.ord, s_warp_id, warp, lane, best_ord, best_id
        );
        if (tid == 0u) {
            out_tokens[row] = (best_id == 0xFFFFFFFFu) ? 0u : best_id;
        }
        return;
    }

    // Mode 3: `temp > 0.0f` -> keyed Gumbel-Max with exact top_k & top_p.
    if (tid == 0u) {
        s_pos_base = gumbel_pos_base_dev(seeds[row], positions[row]);
    }
    const uint32_t top_k = top_ks[row];
    const float top_p = top_ps[row];
    const float eff_temp = fmaxf(temp_raw, 1.0e-6f);
    const bool has_top_p = (top_p > 0.0f && top_p < 1.0f);
    const bool has_top_k = (top_k > 0u && top_k < vocab_size);

    // Regime A: Full-vocab Gumbel-Max (no top_k, no top_p) -> 1 single `uint4` pass.
    if (!has_top_k && !has_top_p) {
        __syncthreads();
        float best_score = -CUDART_INF_F;
        uint32_t best_id = 0xFFFFFFFFu;
        if (tid < prologue) {
            target_score_if_ge(row_raw[tid], tid, 0u, eff_temp, s_pos_base, best_score, best_id);
        }
        if (n_vec > 0u) {
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
            for (uint32_t idx = tid; idx < n_vec; idx += 512u) {
                const uint4 v4 = row_v4[idx];
                const uint32_t c0 = prologue + (idx << 3);
                #pragma unroll 1
                for (uint32_t h = 0u; h < 8u; ++h) {
                    const uint32_t q = h >> 1;
                    const uint32_t w = (q == 0u) ? v4.x : ((q == 1u) ? v4.y : ((q == 2u) ? v4.z : v4.w));
                    const uint16_t raw = (uint16_t)((h & 1u) ? (w >> 16) : (w & 0xFFFFu));
                    target_score_if_ge(raw, c0 + h, 0u, eff_temp, s_pos_base, best_score, best_id);
                }
            }
        }
        for (uint32_t c = vec_end + tid; c < vocab_size; c += 512u) {
            target_score_if_ge(row_raw[c], c, 0u, eff_temp, s_pos_base, best_score, best_id);
        }
        block_reduce_argmax_score_id_512(
            s_warp_primary.val, s_warp_id, warp, lane, best_score, best_id
        );
        if (tid == 0u) {
            out_tokens[row] = (best_id == 0xFFFFFFFFu) ? 0u : best_id;
        }
        return;
    }

    const uint32_t eff_k_limit = has_top_k ? top_k : vocab_size;
    unsigned int* __restrict__ warp_hist = s_work.hist[warp];

    // Regime B (hot path: `top_k <= 512`, or `vocab_size <= 512` with `top_p`):
    // Fast 2-pass `uint4` 512-thread radix top-k + fused gather + in-SMEM exact tie-break.
    if (eff_k_limit <= TARGET_VERIFY_SMEM_MAX_K) {
        #pragma unroll
        for (unsigned int i = tid; i < 4096u; i += 512u) {
            s_work.hist[i >> 8][i & 255u] = 0u;
        }
        if (tid == 0u) {
            s_above_cnt = 0u;
            s_cand_cnt = 0u;
            s_eq_cnt = 0u;
        }
        __syncthreads();

        // Pass 1: upper-byte (ord >> 8) per-warp privatized SMEM-atomic histogram.
        if (tid < prologue) {
            const uint16_t raw = row_raw[tid];
            if (bf16_is_finite_u16(raw)) {
                atomicAdd(&warp_hist[(uint32_t)bf16_to_ordered_u16(raw) >> 8], 1u);
            }
        }
        if (n_vec > 0u) {
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
            for (uint32_t idx = tid; idx < n_vec; idx += 512u) {
                const uint4 v4 = row_v4[idx];
                target_pass1_pair(warp_hist, v4.x);
                target_pass1_pair(warp_hist, v4.y);
                target_pass1_pair(warp_hist, v4.z);
                target_pass1_pair(warp_hist, v4.w);
            }
        }
        for (uint32_t c = vec_end + tid; c < vocab_size; c += 512u) {
            const uint16_t raw = row_raw[c];
            if (bf16_is_finite_u16(raw)) {
                atomicAdd(&warp_hist[(uint32_t)bf16_to_ordered_u16(raw) >> 8], 1u);
            }
        }
        __syncthreads();

        // Merge 16 warp histograms into `s_work.hist[0][0..255]`.
        if (tid < 256u) {
            unsigned int sum = 0u;
            #pragma unroll
            for (unsigned int w = 0u; w < 16u; ++w) {
                sum += s_work.hist[w][tid];
            }
            s_work.hist[0][tid] = sum;
        }
        __syncthreads();

        if (tid == 0u) {
            uint32_t total = 0u;
            for (int b = 255; b >= 0; --b) {
                total += s_work.hist[0][b];
            }
            s_meta.total_valid = total;
            if (total > eff_k_limit) {
                uint32_t above = 0u;
                int hi = 255;
                for (; hi > 0; --hi) {
                    if (above + s_work.hist[0][hi] >= eff_k_limit) {
                        break;
                    }
                    above += s_work.hist[0][hi];
                }
                s_meta.win_hi = (uint32_t)hi;
                s_meta.above_hi = above;
                s_meta.need_in_hi = eff_k_limit - above;
                s_thresh_hi_cnt = s_work.hist[0][hi];
            } else {
                s_meta.win_hi = 0u;
                s_meta.above_hi = 0u;
                s_meta.need_in_hi = 0u;
                s_thresh_hi_cnt = 0u;
            }
        }
        __syncthreads();

        const uint32_t total_valid = s_meta.total_valid;
        if (total_valid == 0u) {
            if (tid == 0u) {
                out_tokens[row] = 0u;
            }
            return;
        }
        const uint32_t eff_k = (total_valid < eff_k_limit) ? total_valid : eff_k_limit;

        if (total_valid <= eff_k_limit) {
            // Every finite element belongs to top-K (<= 512 total).
            if (tid < prologue) {
                target_emit_finite(row_raw[tid], tid, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
            }
            if (n_vec > 0u) {
                const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
                for (uint32_t idx = tid; idx < n_vec; idx += 512u) {
                    const uint4 v4 = row_v4[idx];
                    const uint32_t c0 = prologue + (idx << 3);
                    target_emit_finite((uint16_t)(v4.x & 0xFFFFu), c0 + 0u, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    target_emit_finite((uint16_t)(v4.x >> 16),     c0 + 1u, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    target_emit_finite((uint16_t)(v4.y & 0xFFFFu), c0 + 2u, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    target_emit_finite((uint16_t)(v4.y >> 16),     c0 + 3u, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    target_emit_finite((uint16_t)(v4.z & 0xFFFFu), c0 + 4u, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    target_emit_finite((uint16_t)(v4.z >> 16),     c0 + 5u, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    target_emit_finite((uint16_t)(v4.w & 0xFFFFu), c0 + 6u, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    target_emit_finite((uint16_t)(v4.w >> 16),     c0 + 7u, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                }
            }
            for (uint32_t c = vec_end + tid; c < vocab_size; c += 512u) {
                target_emit_finite(row_raw[c], c, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
            }
            __syncthreads();
        } else if (s_thresh_hi_cnt <= TARGET_VERIFY_GATHER_CAP) {
            // Fast 2-pass fused path (99.9%+ of rows): gather `hi > thresh_hi` directly
            // into `s_above` and `hi == thresh_hi` (<= 2560 items) into SMEM along with
            // `fine[256]`, then resolve exact ties in SMEM with 0 extra HBM reads.
            const uint32_t thresh_hi = s_meta.win_hi;
            if (tid < 256u) {
                s_work.gather.fine[tid] = 0u;
            }
            __syncthreads();

            if (tid < prologue) {
                target_pass2_gather_elem(
                    row_raw[tid], tid, thresh_hi, eff_k,
                    &s_above_cnt, s_above_ord, s_above_idx,
                    &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine
                );
            }
            if (n_vec > 0u) {
                const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
                for (uint32_t idx = tid; idx < n_vec; idx += 512u) {
                    const uint4 v4 = row_v4[idx];
                    const uint32_t c0 = prologue + (idx << 3);
                    target_pass2_gather_elem((uint16_t)(v4.x & 0xFFFFu), c0 + 0u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                    target_pass2_gather_elem((uint16_t)(v4.x >> 16),     c0 + 1u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                    target_pass2_gather_elem((uint16_t)(v4.y & 0xFFFFu), c0 + 2u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                    target_pass2_gather_elem((uint16_t)(v4.y >> 16),     c0 + 3u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                    target_pass2_gather_elem((uint16_t)(v4.z & 0xFFFFu), c0 + 4u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                    target_pass2_gather_elem((uint16_t)(v4.z >> 16),     c0 + 5u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                    target_pass2_gather_elem((uint16_t)(v4.w & 0xFFFFu), c0 + 6u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                    target_pass2_gather_elem((uint16_t)(v4.w >> 16),     c0 + 7u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                }
            }
            for (uint32_t c = vec_end + tid; c < vocab_size; c += 512u) {
                target_pass2_gather_elem(
                    row_raw[c], c, thresh_hi, eff_k,
                    &s_above_cnt, s_above_ord, s_above_idx,
                    &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine
                );
            }
            __syncthreads();

            if (tid == 0u) {
                const uint32_t need = s_meta.need_in_hi;
                uint32_t above_lo = 0u;
                int lo = 255;
                for (; lo > 0; --lo) {
                    if (above_lo + s_work.gather.fine[lo] >= need) {
                        break;
                    }
                    above_lo += s_work.gather.fine[lo];
                }
                s_meta.thresh_key = (thresh_hi << 8) | (uint32_t)lo;
                s_meta.need_eq = need - above_lo;
                s_meta.count_eq = s_work.gather.fine[lo];
            }
            __syncthreads();

            const uint32_t n_cand =
                (s_cand_cnt < TARGET_VERIFY_GATHER_CAP) ? s_cand_cnt : TARGET_VERIFY_GATHER_CAP;
            const uint32_t thresh_lo = s_meta.thresh_key & 0xFFu;
            const uint32_t need_eq = s_meta.need_eq;
            const uint32_t count_eq = s_meta.count_eq;

            // Exact O(1)-pass in-SMEM radix select for boundary ties (`cand_lo == thresh_lo`):
            // finds the exact `need_eq`-th smallest token ID with zero global memory reads.
            uint32_t eq_id_thresh = 0xFFFFFFFFu;
            if (count_eq > need_eq) {
                if (tid == 0u) {
                    s_id_prefix = 0u;
                    s_id_mask = 0u;
                    s_id_need = need_eq;
                }
                __syncthreads();
                const int start_shift = (vocab_size <= 0x10000u)
                    ? 8
                    : ((vocab_size <= 0x1000000u) ? 16 : 24);
                for (int shift = start_shift; shift >= 0; shift -= 8) {
                    if (tid < 256u) {
                        s_work.gather.fine[tid] = 0u;
                    }
                    __syncthreads();
                    const uint32_t prefix = s_id_prefix;
                    const uint32_t mask = s_id_mask;
                    for (uint32_t i = tid; i < n_cand; i += 512u) {
                        if ((uint32_t)s_work.gather.cand_lo[i] == thresh_lo) {
                            const uint32_t cid = s_work.gather.cand_idx[i];
                            if ((cid & mask) == prefix) {
                                atomicAdd(&s_work.gather.fine[(cid >> shift) & 0xFFu], 1u);
                            }
                        }
                    }
                    __syncthreads();
                    if (tid == 0u) {
                        const uint32_t need = s_id_need;
                        uint32_t below = 0u;
                        int d = 0;
                        for (; d < 255; ++d) {
                            if (below + s_work.gather.fine[d] >= need) {
                                break;
                            }
                            below += s_work.gather.fine[d];
                        }
                        s_id_need = need - below;
                        s_id_prefix = prefix | ((uint32_t)d << shift);
                        s_id_mask = mask | (0xFFu << shift);
                    }
                    __syncthreads();
                }
                eq_id_thresh = s_id_prefix;
            }

            for (uint32_t i = tid; i < n_cand; i += 512u) {
                const uint32_t lo = (uint32_t)s_work.gather.cand_lo[i];
                const uint32_t cid = s_work.gather.cand_idx[i];
                if (lo > thresh_lo || (lo == thresh_lo && cid <= eq_id_thresh)) {
                    const uint32_t slot = atomicAdd(&s_above_cnt, 1u);
                    if (slot < eff_k) {
                        s_above_ord[slot] = (uint16_t)((thresh_hi << 8) | lo);
                        s_above_idx[slot] = cid;
                    }
                }
            }
            __syncthreads();
        } else {
            // Heavy-bucket fallback (`s_thresh_hi_cnt > 2560`, e.g. all-equal logits):
            // Pass 2 builds fine histogram in `s_work.hist[16][256]` while emitting `hi > thresh_hi`.
            const uint32_t thresh_hi = s_meta.win_hi;
            #pragma unroll
            for (unsigned int i = tid; i < 4096u; i += 512u) {
                s_work.hist[i >> 8][i & 255u] = 0u;
            }
            __syncthreads();

            if (tid < prologue) {
                target_pass2_hist_elem(
                    warp_hist, row_raw[tid], tid, thresh_hi, eff_k,
                    &s_above_cnt, s_above_ord, s_above_idx
                );
            }
            if (n_vec > 0u) {
                const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
                for (uint32_t idx = tid; idx < n_vec; idx += 512u) {
                    const uint4 v4 = row_v4[idx];
                    const uint32_t c0 = prologue + (idx << 3);
                    target_pass2_hist_elem(warp_hist, (uint16_t)(v4.x & 0xFFFFu), c0 + 0u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    target_pass2_hist_elem(warp_hist, (uint16_t)(v4.x >> 16),     c0 + 1u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    target_pass2_hist_elem(warp_hist, (uint16_t)(v4.y & 0xFFFFu), c0 + 2u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    target_pass2_hist_elem(warp_hist, (uint16_t)(v4.y >> 16),     c0 + 3u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    target_pass2_hist_elem(warp_hist, (uint16_t)(v4.z & 0xFFFFu), c0 + 4u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    target_pass2_hist_elem(warp_hist, (uint16_t)(v4.z >> 16),     c0 + 5u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    target_pass2_hist_elem(warp_hist, (uint16_t)(v4.w & 0xFFFFu), c0 + 6u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    target_pass2_hist_elem(warp_hist, (uint16_t)(v4.w >> 16),     c0 + 7u, thresh_hi, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                }
            }
            for (uint32_t c = vec_end + tid; c < vocab_size; c += 512u) {
                target_pass2_hist_elem(
                    warp_hist, row_raw[c], c, thresh_hi, eff_k,
                    &s_above_cnt, s_above_ord, s_above_idx
                );
            }
            __syncthreads();

            if (tid < 256u) {
                unsigned int sum = 0u;
                #pragma unroll
                for (unsigned int w = 0u; w < 16u; ++w) {
                    sum += s_work.hist[w][tid];
                }
                s_work.hist[0][tid] = sum;
            }
            __syncthreads();

            if (tid == 0u) {
                const uint32_t need = s_meta.need_in_hi;
                uint32_t above_lo = 0u;
                int lo = 255;
                for (; lo > 0; --lo) {
                    if (above_lo + s_work.hist[0][lo] >= need) {
                        break;
                    }
                    above_lo += s_work.hist[0][lo];
                }
                s_meta.thresh_key = (thresh_hi << 8) | (uint32_t)lo;
                s_meta.strict_gt = above_lo;
                s_meta.need_eq = need - above_lo;
                s_meta.count_eq = s_work.hist[0][lo];
            }
            __syncthreads();

            const uint32_t thresh_lo = s_meta.thresh_key & 0xFFu;
            const uint32_t above_lo = s_meta.strict_gt;
            const uint32_t need_eq = s_meta.need_eq;
            const uint32_t count_eq = s_meta.count_eq;

            if (count_eq <= need_eq) {
                if (tid < prologue) {
                    target_pass3_notie_elem(
                        row_raw[tid], tid, thresh_hi, thresh_lo, eff_k,
                        &s_above_cnt, s_above_ord, s_above_idx
                    );
                }
                if (n_vec > 0u) {
                    const uint4* __restrict__ row_v4 =
                        reinterpret_cast<const uint4*>(row_raw + prologue);
                    for (uint32_t idx = tid; idx < n_vec; idx += 512u) {
                        const uint4 v4 = row_v4[idx];
                        const uint32_t c0 = prologue + (idx << 3);
                        target_pass3_notie_elem((uint16_t)(v4.x & 0xFFFFu), c0 + 0u, thresh_hi, thresh_lo, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                        target_pass3_notie_elem((uint16_t)(v4.x >> 16),     c0 + 1u, thresh_hi, thresh_lo, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                        target_pass3_notie_elem((uint16_t)(v4.y & 0xFFFFu), c0 + 2u, thresh_hi, thresh_lo, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                        target_pass3_notie_elem((uint16_t)(v4.y >> 16),     c0 + 3u, thresh_hi, thresh_lo, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                        target_pass3_notie_elem((uint16_t)(v4.z & 0xFFFFu), c0 + 4u, thresh_hi, thresh_lo, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                        target_pass3_notie_elem((uint16_t)(v4.z >> 16),     c0 + 5u, thresh_hi, thresh_lo, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                        target_pass3_notie_elem((uint16_t)(v4.w & 0xFFFFu), c0 + 6u, thresh_hi, thresh_lo, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                        target_pass3_notie_elem((uint16_t)(v4.w >> 16),     c0 + 7u, thresh_hi, thresh_lo, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    }
                }
                for (uint32_t c = vec_end + tid; c < vocab_size; c += 512u) {
                    target_pass3_notie_elem(
                        row_raw[c], c, thresh_hi, thresh_lo, eff_k,
                        &s_above_cnt, s_above_ord, s_above_idx
                    );
                }
                __syncthreads();
            } else {
                const uint16_t thresh_key = (uint16_t)s_meta.thresh_key;
                for (uint32_t base = 0u; base < vocab_size; base += 512u) {
                    if (above_lo == 0u && s_eq_cnt >= need_eq) {
                        break;
                    }
                    const uint32_t c = base + tid;
                    bool hit_eq = false;
                    if (c < vocab_size) {
                        const uint16_t raw = row_raw[c];
                        if (bf16_is_finite_u16(raw)) {
                            const uint16_t ord = bf16_to_ordered_u16(raw);
                            if (((uint32_t)ord >> 8) == thresh_hi) {
                                const uint32_t lo = (uint32_t)ord & 0xFFu;
                                if (lo > thresh_lo) {
                                    const uint32_t slot = atomicAdd(&s_above_cnt, 1u);
                                    if (slot < eff_k) {
                                        s_above_ord[slot] = ord;
                                        s_above_idx[slot] = c;
                                    }
                                } else if (lo == thresh_lo && s_eq_cnt < need_eq) {
                                    hit_eq = true;
                                }
                            }
                        }
                    }
                    if (__syncthreads_or(hit_eq)) {
                        const unsigned int wm = __ballot_sync(0xFFFFFFFFu, hit_eq);
                        if (lane == 0u) {
                            s_warp_cnt[warp] = (uint32_t)__popc(wm);
                        }
                        __syncthreads();
                        const uint32_t prev_eq = s_eq_cnt;
                        if (hit_eq) {
                            uint32_t offset = 0u;
                            #pragma unroll
                            for (unsigned int w = 0u; w < 16u; ++w) {
                                if (w < warp) {
                                    offset += s_warp_cnt[w];
                                }
                            }
                            const unsigned int lane_mask = (1u << lane) - 1u;
                            const uint32_t r =
                                prev_eq + offset + (uint32_t)__popc(wm & lane_mask);
                            if (r < need_eq) {
                                const uint32_t slot = atomicAdd(&s_above_cnt, 1u);
                                if (slot < eff_k) {
                                    s_above_ord[slot] = thresh_key;
                                    s_above_idx[slot] = c;
                                }
                            }
                        }
                        __syncthreads();
                        if (tid == 0u) {
                            uint32_t chunk_eq = 0u;
                            #pragma unroll
                            for (unsigned int w = 0u; w < 16u; ++w) {
                                chunk_eq += s_warp_cnt[w];
                            }
                            s_eq_cnt = prev_eq + chunk_eq;
                        }
                        __syncthreads();
                    }
                }
                __syncthreads();
            }
        }

        const uint32_t count = (s_above_cnt < eff_k) ? s_above_cnt : eff_k;
        if (count == 0u) {
            if (tid == 0u) {
                out_tokens[row] = 0u;
            }
            return;
        }

        if (!has_top_p) {
            // Top-K only: no sort needed — score all <= 512 winners directly in 1 parallel step!
            float best_score = -CUDART_INF_F;
            uint32_t best_id = 0xFFFFFFFFu;
            if (tid < count) {
                const uint32_t gid = s_above_idx[tid];
                const float v = bf16_u16_to_f32(ordered_u16_to_bf16(s_above_ord[tid]));
                best_score = __fdiv_rn(v, eff_temp) + gumbel_from_base_f32_dev(s_pos_base, gid);
                best_id = gid;
            }
            block_reduce_argmax_score_id_512(
                s_warp_primary.val, s_warp_id, warp, lane, best_score, best_id
            );
            if (tid == 0u) {
                out_tokens[row] = (best_id == 0xFFFFFFFFu) ? s_above_idx[0] : best_id;
            }
            return;
        }

        // Top-K + Top-P: parallel bitonic sort descending by (ord DESC, id ASC) over <= 512 items.
        uint32_t n_pad = 1u;
        while (n_pad < count) {
            n_pad <<= 1u;
        }
        if (tid >= count && tid < n_pad) {
            s_above_ord[tid] = 0u;
            s_above_idx[tid] = 0xFFFFFFFFu;
        }
        __syncthreads();

        for (uint32_t size = 2u; size <= n_pad; size <<= 1u) {
            for (uint32_t stride = size >> 1u; stride > 0u; stride >>= 1u) {
                if (tid < (n_pad >> 1u)) {
                    const uint32_t block_lo = tid & (stride - 1u);
                    const uint32_t i = ((tid - block_lo) << 1u) + block_lo;
                    const uint32_t j = i + stride;
                    const bool want_desc = ((i & size) == 0u);
                    const uint16_t oa = s_above_ord[i];
                    const uint32_t ida = s_above_idx[i];
                    const uint16_t ob = s_above_ord[j];
                    const uint32_t idb = s_above_idx[j];
                    const bool a_before_b = target_cand_comes_before(oa, ida, ob, idb);
                    if (want_desc ? !a_before_b : a_before_b) {
                        s_above_ord[i] = ob;
                        s_above_ord[j] = oa;
                        s_above_idx[i] = idb;
                        s_above_idx[j] = ida;
                    }
                }
                __syncthreads();
            }
        }

        // Softmax weights + exact cumulative top_p cutoff matching `choose_gumbel`.
        const float max_sc = __fdiv_rn(
            bf16_u16_to_f32(ordered_u16_to_bf16(s_above_ord[0])),
            eff_temp
        );
        if (tid < count) {
            const float sc = __fdiv_rn(
                bf16_u16_to_f32(ordered_u16_to_bf16(s_above_ord[tid])),
                eff_temp
            );
            s_work.top_p.scaled[tid] = sc;
            s_work.top_p.weight[tid] = expf(sc - max_sc);
        }
        __syncthreads();

        if (tid == 0u) {
            float sum = 0.0f;
            for (uint32_t i = 0u; i < count; ++i) {
                sum += s_work.top_p.weight[i];
            }
            uint32_t kept = count;
            if (sum > 0.0f) {
                float cum = 0.0f;
                for (uint32_t i = 0u; i < count; ++i) {
                    cum += __fdiv_rn(s_work.top_p.weight[i], sum);
                    if (cum >= top_p) {
                        kept = i + 1u;
                        break;
                    }
                }
            }
            s_kept = kept;
        }
        __syncthreads();

        const uint32_t kept = s_kept;
        float best_score = -CUDART_INF_F;
        uint32_t best_id = 0xFFFFFFFFu;
        if (tid < kept) {
            const uint32_t gid = s_above_idx[tid];
            best_score = s_work.top_p.scaled[tid] + gumbel_from_base_f32_dev(s_pos_base, gid);
            best_id = gid;
        }
        block_reduce_argmax_score_id_512(
            s_warp_primary.val, s_warp_id, warp, lane, best_score, best_id
        );
        if (tid == 0u) {
            out_tokens[row] = (best_id == 0xFFFFFFFFu) ? s_above_idx[0] : best_id;
        }
        return;
    }

    // Regime C: Large `top_k > 512` (with or without `top_p`) or full-vocab `top_p` (`top_k == 0`).
    uint16_t topk_thresh_key = 0u;
    uint32_t topk_need_eq = 0u;
    bool topk_need_tiebreak = false;

    if (has_top_k) {
        #pragma unroll
        for (unsigned int i = tid; i < 4096u; i += 512u) {
            s_work.hist[i >> 8][i & 255u] = 0u;
        }
        __syncthreads();

        if (tid < prologue) {
            const uint16_t raw = row_raw[tid];
            if (bf16_is_finite_u16(raw)) {
                atomicAdd(&warp_hist[(uint32_t)bf16_to_ordered_u16(raw) >> 8], 1u);
            }
        }
        if (n_vec > 0u) {
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
            for (uint32_t idx = tid; idx < n_vec; idx += 512u) {
                const uint4 v4 = row_v4[idx];
                target_pass1_pair(warp_hist, v4.x);
                target_pass1_pair(warp_hist, v4.y);
                target_pass1_pair(warp_hist, v4.z);
                target_pass1_pair(warp_hist, v4.w);
            }
        }
        for (uint32_t c = vec_end + tid; c < vocab_size; c += 512u) {
            const uint16_t raw = row_raw[c];
            if (bf16_is_finite_u16(raw)) {
                atomicAdd(&warp_hist[(uint32_t)bf16_to_ordered_u16(raw) >> 8], 1u);
            }
        }
        __syncthreads();

        if (tid < 256u) {
            unsigned int sum = 0u;
            #pragma unroll
            for (unsigned int w = 0u; w < 16u; ++w) {
                sum += s_work.hist[w][tid];
            }
            s_work.hist[0][tid] = sum;
        }
        __syncthreads();

        if (tid == 0u) {
            uint32_t total = 0u;
            for (int b = 255; b >= 0; --b) {
                total += s_work.hist[0][b];
            }
            s_meta.total_valid = total;
            if (total <= top_k) {
                s_meta.thresh_key = 0u;
                s_meta.need_eq = 0u;
                s_meta.count_eq = 0u;
                s_meta.run_pass2 = 0u;
            } else {
                uint32_t above = 0u;
                int hi = 255;
                for (; hi > 0; --hi) {
                    if (above + s_work.hist[0][hi] >= top_k) {
                        break;
                    }
                    above += s_work.hist[0][hi];
                }
                s_meta.win_hi = (uint32_t)hi;
                s_meta.above_hi = above;
                s_meta.need_in_hi = top_k - above;
                if (above + s_work.hist[0][hi] == top_k) {
                    s_meta.thresh_key = ((uint32_t)hi) << 8;
                    s_meta.need_eq = 0u;
                    s_meta.count_eq = 0u;
                    s_meta.run_pass2 = 0u;
                } else {
                    s_meta.run_pass2 = 1u;
                }
            }
        }
        __syncthreads();

        if (s_meta.total_valid == 0u) {
            if (tid == 0u) {
                out_tokens[row] = 0u;
            }
            return;
        }

        if (s_meta.run_pass2 != 0u) {
            const uint32_t thresh_hi = s_meta.win_hi;
            #pragma unroll
            for (unsigned int i = tid; i < 4096u; i += 512u) {
                s_work.hist[i >> 8][i & 255u] = 0u;
            }
            __syncthreads();

            if (tid < prologue) {
                const uint16_t raw = row_raw[tid];
                if (bf16_is_finite_u16(raw)) {
                    const uint16_t ord = bf16_to_ordered_u16(raw);
                    if (((uint32_t)ord >> 8) == thresh_hi) {
                        atomicAdd(&warp_hist[(uint32_t)ord & 0xFFu], 1u);
                    }
                }
            }
            if (n_vec > 0u) {
                const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
                for (uint32_t idx = tid; idx < n_vec; idx += 512u) {
                    const uint4 v4 = row_v4[idx];
                    target_pass2_fine_pair(warp_hist, v4.x, thresh_hi);
                    target_pass2_fine_pair(warp_hist, v4.y, thresh_hi);
                    target_pass2_fine_pair(warp_hist, v4.z, thresh_hi);
                    target_pass2_fine_pair(warp_hist, v4.w, thresh_hi);
                }
            }
            for (uint32_t c = vec_end + tid; c < vocab_size; c += 512u) {
                const uint16_t raw = row_raw[c];
                if (bf16_is_finite_u16(raw)) {
                    const uint16_t ord = bf16_to_ordered_u16(raw);
                    if (((uint32_t)ord >> 8) == thresh_hi) {
                        atomicAdd(&warp_hist[(uint32_t)ord & 0xFFu], 1u);
                    }
                }
            }
            __syncthreads();

            if (tid < 256u) {
                unsigned int sum = 0u;
                #pragma unroll
                for (unsigned int w = 0u; w < 16u; ++w) {
                    sum += s_work.hist[w][tid];
                }
                s_work.hist[0][tid] = sum;
            }
            __syncthreads();

            if (tid == 0u) {
                const uint32_t need = s_meta.need_in_hi;
                uint32_t above_lo = 0u;
                int lo = 255;
                for (; lo > 0; --lo) {
                    if (above_lo + s_work.hist[0][lo] >= need) {
                        break;
                    }
                    above_lo += s_work.hist[0][lo];
                }
                s_meta.thresh_key = (thresh_hi << 8) | (uint32_t)lo;
                s_meta.need_eq = need - above_lo;
                s_meta.count_eq = s_work.hist[0][lo];
            }
            __syncthreads();
        }

        topk_thresh_key = (uint16_t)s_meta.thresh_key;
        topk_need_eq = s_meta.need_eq;
        topk_need_tiebreak = (s_meta.count_eq > s_meta.need_eq);
    }

    // Regime C1: Large `top_k > 512` without `top_p`.
    if (!has_top_p) {
        float best_score = -CUDART_INF_F;
        uint32_t best_id = 0xFFFFFFFFu;
        if (!topk_need_tiebreak) {
            if (tid < prologue) {
                target_score_if_ge(row_raw[tid], tid, topk_thresh_key, eff_temp, s_pos_base, best_score, best_id);
            }
            if (n_vec > 0u) {
                const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
                for (uint32_t idx = tid; idx < n_vec; idx += 512u) {
                    const uint4 v4 = row_v4[idx];
                    const uint32_t c0 = prologue + (idx << 3);
                    #pragma unroll 1
                    for (uint32_t h = 0u; h < 8u; ++h) {
                        const uint32_t q = h >> 1;
                        const uint32_t w = (q == 0u) ? v4.x : ((q == 1u) ? v4.y : ((q == 2u) ? v4.z : v4.w));
                        const uint16_t raw = (uint16_t)((h & 1u) ? (w >> 16) : (w & 0xFFFFu));
                        target_score_if_ge(raw, c0 + h, topk_thresh_key, eff_temp, s_pos_base, best_score, best_id);
                    }
                }
            }
            for (uint32_t c = vec_end + tid; c < vocab_size; c += 512u) {
                target_score_if_ge(row_raw[c], c, topk_thresh_key, eff_temp, s_pos_base, best_score, best_id);
            }
        } else {
            if (tid == 0u) {
                s_eq_cnt = 0u;
            }
            __syncthreads();
            for (uint32_t base = 0u; base < vocab_size; base += 512u) {
                const uint32_t c = base + tid;
                bool hit_eq = false;
                uint16_t raw = 0u;
                if (c < vocab_size) {
                    raw = row_raw[c];
                    if (bf16_is_finite_u16(raw)) {
                        const uint16_t k16 = bf16_to_ordered_u16(raw);
                        if (k16 > topk_thresh_key) {
                            const float v = bf16_u16_to_f32(raw);
                            const float score =
                                __fdiv_rn(v, eff_temp) + gumbel_from_base_f32_dev(s_pos_base, c);
                            if (score > best_score || (score == best_score && c < best_id)) {
                                best_score = score;
                                best_id = c;
                            }
                        } else if (k16 == topk_thresh_key && s_eq_cnt < topk_need_eq) {
                            hit_eq = true;
                        }
                    }
                }
                if (__syncthreads_or(hit_eq)) {
                    const unsigned int wm = __ballot_sync(0xFFFFFFFFu, hit_eq);
                    if (lane == 0u) {
                        s_warp_cnt[warp] = (uint32_t)__popc(wm);
                    }
                    __syncthreads();
                    const uint32_t prev_eq = s_eq_cnt;
                    if (hit_eq) {
                        uint32_t offset = 0u;
                        #pragma unroll
                        for (unsigned int w = 0u; w < 16u; ++w) {
                            if (w < warp) {
                                offset += s_warp_cnt[w];
                            }
                        }
                        const unsigned int lane_mask = (1u << lane) - 1u;
                        const uint32_t r =
                            prev_eq + offset + (uint32_t)__popc(wm & lane_mask);
                        if (r < topk_need_eq) {
                            const float v = bf16_u16_to_f32(raw);
                            const float score =
                                __fdiv_rn(v, eff_temp) + gumbel_from_base_f32_dev(s_pos_base, c);
                            if (score > best_score || (score == best_score && c < best_id)) {
                                best_score = score;
                                best_id = c;
                            }
                        }
                    }
                    __syncthreads();
                    if (tid == 0u) {
                        uint32_t chunk_eq = 0u;
                        #pragma unroll
                        for (unsigned int w = 0u; w < 16u; ++w) {
                            chunk_eq += s_warp_cnt[w];
                        }
                        s_eq_cnt = prev_eq + chunk_eq;
                    }
                    __syncthreads();
                }
            }
        }
        block_reduce_argmax_score_id_512(
            s_warp_primary.val, s_warp_id, warp, lane, best_score, best_id
        );
        if (tid == 0u) {
            out_tokens[row] = (best_id == 0xFFFFFFFFu) ? 0u : best_id;
        }
        return;
    }

    // Regime C2: `top_p` active with large (`> 512`) or disabled (`0`) `top_k`.
    if (tid == 0u) {
        s_meta.topk_max_id = 0xFFFFFFFFu;
        s_eq_cnt = 0u;
    }
    __syncthreads();

    if (has_top_k && topk_need_tiebreak) {
        for (uint32_t base = 0u; base < vocab_size; base += 512u) {
            if (s_eq_cnt >= topk_need_eq) {
                break;
            }
            const uint32_t c = base + tid;
            bool hit_eq = false;
            if (c < vocab_size) {
                const uint16_t raw = row_raw[c];
                if (bf16_is_finite_u16(raw) && bf16_to_ordered_u16(raw) == topk_thresh_key) {
                    hit_eq = true;
                }
            }
            if (__syncthreads_or(hit_eq)) {
                const unsigned int wm = __ballot_sync(0xFFFFFFFFu, hit_eq);
                if (lane == 0u) {
                    s_warp_cnt[warp] = (uint32_t)__popc(wm);
                }
                __syncthreads();
                const uint32_t prev_eq = s_eq_cnt;
                if (hit_eq) {
                    uint32_t offset = 0u;
                    #pragma unroll
                    for (unsigned int w = 0u; w < 16u; ++w) {
                        if (w < warp) {
                            offset += s_warp_cnt[w];
                        }
                    }
                    const unsigned int lane_mask = (1u << lane) - 1u;
                    const uint32_t r = prev_eq + offset + (uint32_t)__popc(wm & lane_mask);
                    if (r + 1u == topk_need_eq) {
                        s_meta.topk_max_id = c;
                    }
                }
                __syncthreads();
                if (tid == 0u) {
                    uint32_t chunk_eq = 0u;
                    #pragma unroll
                    for (unsigned int w = 0u; w < 16u; ++w) {
                        chunk_eq += s_warp_cnt[w];
                    }
                    s_eq_cnt = prev_eq + chunk_eq;
                }
                __syncthreads();
            }
        }
    }
    const uint32_t topk_max_id = s_meta.topk_max_id;

    // Exact row max over finite BF16 logits (`uint4` 1-pass).
    uint32_t local_max_ord = 0u;
    if (tid < prologue) {
        target_max_finite_ord_update(row_raw[tid], local_max_ord);
    }
    if (n_vec > 0u) {
        const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
        for (uint32_t idx = tid; idx < n_vec; idx += 512u) {
            const uint4 v4 = row_v4[idx];
            target_max_finite_ord_update((uint16_t)(v4.x & 0xFFFFu), local_max_ord);
            target_max_finite_ord_update((uint16_t)(v4.x >> 16),     local_max_ord);
            target_max_finite_ord_update((uint16_t)(v4.y & 0xFFFFu), local_max_ord);
            target_max_finite_ord_update((uint16_t)(v4.y >> 16),     local_max_ord);
            target_max_finite_ord_update((uint16_t)(v4.z & 0xFFFFu), local_max_ord);
            target_max_finite_ord_update((uint16_t)(v4.z >> 16),     local_max_ord);
            target_max_finite_ord_update((uint16_t)(v4.w & 0xFFFFu), local_max_ord);
            target_max_finite_ord_update((uint16_t)(v4.w >> 16),     local_max_ord);
        }
    }
    for (uint32_t c = vec_end + tid; c < vocab_size; c += 512u) {
        target_max_finite_ord_update(row_raw[c], local_max_ord);
    }
    const uint32_t max_ord = block_reduce_max_u32_512(s_warp_primary.ord, warp, lane, local_max_ord);
    if (max_ord == 0u) {
        if (tid == 0u) {
            out_tokens[row] = 0u;
        }
        return;
    }
    const float max_sc = __fdiv_rn(
        bf16_u16_to_f32(ordered_u16_to_bf16((uint16_t)max_ord)),
        eff_temp
    );

    // Pass A: probability-mass histogram over upper 8 bits of BF16 key.
    if (tid < 256u) {
        s_work.prob_hi[tid] = 0.0f;
    }
    __syncthreads();

    if (tid < prologue) {
        target_prob_hi_elem(
            row_raw[tid], tid, topk_thresh_key, topk_max_id, eff_temp, max_sc, s_work.prob_hi
        );
    }
    if (n_vec > 0u) {
        const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
        for (uint32_t idx = tid; idx < n_vec; idx += 512u) {
            const uint4 v4 = row_v4[idx];
            const uint32_t c0 = prologue + (idx << 3);
            #pragma unroll 1
            for (uint32_t h = 0u; h < 8u; ++h) {
                const uint32_t q = h >> 1;
                const uint32_t w = (q == 0u) ? v4.x : ((q == 1u) ? v4.y : ((q == 2u) ? v4.z : v4.w));
                const uint16_t raw = (uint16_t)((h & 1u) ? (w >> 16) : (w & 0xFFFFu));
                target_prob_hi_elem(raw, c0 + h, topk_thresh_key, topk_max_id, eff_temp, max_sc, s_work.prob_hi);
            }
        }
    }
    for (uint32_t c = vec_end + tid; c < vocab_size; c += 512u) {
        target_prob_hi_elem(
            row_raw[c], c, topk_thresh_key, topk_max_id, eff_temp, max_sc, s_work.prob_hi
        );
    }
    __syncthreads();

    if (tid == 0u) {
        float total_sum = 0.0f;
        for (int b = 255; b >= 0; --b) {
            total_sum += s_work.prob_hi[b];
        }
        s_meta.total_prob_sum = total_sum;
        if (total_sum > 0.0f && isfinite(total_sum)) {
            float cum = 0.0f;
            int hi = 255;
            int last_nz = 255;
            for (; hi >= 0; --hi) {
                const float bh = s_work.prob_hi[hi];
                if (bh > 0.0f) {
                    last_nz = hi;
                    if (__fdiv_rn(cum + bh, total_sum) >= top_p) {
                        break;
                    }
                    cum += bh;
                }
            }
            if (hi < 0) {
                hi = last_nz;
                cum = fmaxf(cum - s_work.prob_hi[hi], 0.0f);
            }
            s_meta.p_win_hi = (uint32_t)hi;
            s_meta.p_cum_above_hi = cum;
            s_meta.p_valid = 1u;
        } else {
            s_meta.p_valid = 0u;
        }
    }
    __syncthreads();

    // Pass B: lower-8-bit count histogram inside `p_win_hi` (per-warp SMEM atomics).
    if (s_meta.p_valid != 0u) {
        const uint32_t p_hi = s_meta.p_win_hi;
        #pragma unroll
        for (unsigned int i = tid; i < 4096u; i += 512u) {
            s_work.hist[i >> 8][i & 255u] = 0u;
        }
        __syncthreads();

        if (tid < prologue) {
            target_prob_lo_elem(warp_hist, row_raw[tid], tid, topk_thresh_key, topk_max_id, p_hi);
        }
        if (n_vec > 0u) {
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
            for (uint32_t idx = tid; idx < n_vec; idx += 512u) {
                const uint4 v4 = row_v4[idx];
                const uint32_t c0 = prologue + (idx << 3);
                target_prob_lo_elem(warp_hist, (uint16_t)(v4.x & 0xFFFFu), c0 + 0u, topk_thresh_key, topk_max_id, p_hi);
                target_prob_lo_elem(warp_hist, (uint16_t)(v4.x >> 16),     c0 + 1u, topk_thresh_key, topk_max_id, p_hi);
                target_prob_lo_elem(warp_hist, (uint16_t)(v4.y & 0xFFFFu), c0 + 2u, topk_thresh_key, topk_max_id, p_hi);
                target_prob_lo_elem(warp_hist, (uint16_t)(v4.y >> 16),     c0 + 3u, topk_thresh_key, topk_max_id, p_hi);
                target_prob_lo_elem(warp_hist, (uint16_t)(v4.z & 0xFFFFu), c0 + 4u, topk_thresh_key, topk_max_id, p_hi);
                target_prob_lo_elem(warp_hist, (uint16_t)(v4.z >> 16),     c0 + 5u, topk_thresh_key, topk_max_id, p_hi);
                target_prob_lo_elem(warp_hist, (uint16_t)(v4.w & 0xFFFFu), c0 + 6u, topk_thresh_key, topk_max_id, p_hi);
                target_prob_lo_elem(warp_hist, (uint16_t)(v4.w >> 16),     c0 + 7u, topk_thresh_key, topk_max_id, p_hi);
            }
        }
        for (uint32_t c = vec_end + tid; c < vocab_size; c += 512u) {
            target_prob_lo_elem(warp_hist, row_raw[c], c, topk_thresh_key, topk_max_id, p_hi);
        }
        __syncthreads();

        if (tid < 256u) {
            unsigned int lo_total = 0u;
            #pragma unroll
            for (unsigned int w = 0u; w < 16u; ++w) {
                lo_total += s_work.hist[w][tid];
            }
            s_work.hist[0][tid] = lo_total;
        }
        __syncthreads();

        if (tid == 0u) {
            const float total_sum = s_meta.total_prob_sum;
            float cum = s_meta.p_cum_above_hi;
            int win_lo = 0;
            uint32_t need_eq_p = 1u;
            uint32_t count_eq_p = 1u;
            bool found = false;
            for (int lo = 255; lo >= 0; --lo) {
                const uint32_t cnt = s_work.hist[0][lo];
                if (cnt == 0u) {
                    continue;
                }
                const uint16_t ord_bin = (uint16_t)((p_hi << 8) | (uint32_t)lo);
                const float v_bin = bf16_u16_to_f32(ordered_u16_to_bf16(ord_bin));
                const float w_bin = expf(__fdiv_rn(v_bin, eff_temp) - max_sc);
                const float bin_mass = w_bin * (float)cnt;
                win_lo = lo;
                count_eq_p = cnt;
                if (__fdiv_rn(cum + bin_mass, total_sum) >= top_p) {
                    const float rem = fmaxf(top_p * total_sum - cum, 0.0f);
                    uint32_t n_take =
                        (w_bin > 0.0f) ? (uint32_t)ceilf(__fdiv_rn(rem, w_bin)) : cnt;
                    if (n_take < 1u) {
                        n_take = 1u;
                    }
                    if (n_take > cnt) {
                        n_take = cnt;
                    }
                    while (n_take > 1u
                        && __fdiv_rn(cum + w_bin * (float)(n_take - 1u), total_sum) >= top_p) {
                        --n_take;
                    }
                    while (n_take < cnt
                        && __fdiv_rn(cum + w_bin * (float)n_take, total_sum) < top_p) {
                        ++n_take;
                    }
                    need_eq_p = n_take;
                    found = true;
                    break;
                }
                cum += bin_mass;
            }
            if (!found) {
                need_eq_p = count_eq_p;
            }
            s_meta.thresh_key = (p_hi << 8) | (uint32_t)win_lo;
            s_meta.need_eq = need_eq_p;
            s_meta.count_eq = count_eq_p;
        }
        __syncthreads();
    } else {
        if (tid == 0u) {
            s_meta.thresh_key = topk_thresh_key;
            s_meta.need_eq = 0xFFFFFFFFu;
            s_meta.count_eq = 0u;
        }
        __syncthreads();
    }

    // Final scoring pass over the exact post-(top_k, top_p) support.
    const uint16_t p_thresh_key = (uint16_t)s_meta.thresh_key;
    const uint32_t p_need_eq = s_meta.need_eq;
    const bool p_need_tiebreak = (s_meta.count_eq > p_need_eq);
    float best_score = -CUDART_INF_F;
    uint32_t best_id = 0xFFFFFFFFu;
    if (!p_need_tiebreak) {
        if (tid < prologue) {
            target_score_masked_ge(
                row_raw[tid], tid, topk_thresh_key, topk_max_id, p_thresh_key,
                eff_temp, s_pos_base, best_score, best_id
            );
        }
        if (n_vec > 0u) {
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + prologue);
            for (uint32_t idx = tid; idx < n_vec; idx += 512u) {
                const uint4 v4 = row_v4[idx];
                const uint32_t c0 = prologue + (idx << 3);
                #pragma unroll 1
                for (uint32_t h = 0u; h < 8u; ++h) {
                    const uint32_t q = h >> 1;
                    const uint32_t w = (q == 0u) ? v4.x : ((q == 1u) ? v4.y : ((q == 2u) ? v4.z : v4.w));
                    const uint16_t raw = (uint16_t)((h & 1u) ? (w >> 16) : (w & 0xFFFFu));
                    target_score_masked_ge(raw, c0 + h, topk_thresh_key, topk_max_id, p_thresh_key, eff_temp, s_pos_base, best_score, best_id);
                }
            }
        }
        for (uint32_t c = vec_end + tid; c < vocab_size; c += 512u) {
            target_score_masked_ge(
                row_raw[c], c, topk_thresh_key, topk_max_id, p_thresh_key,
                eff_temp, s_pos_base, best_score, best_id
            );
        }
    } else {
        if (tid == 0u) {
            s_eq_cnt = 0u;
        }
        __syncthreads();
        for (uint32_t base = 0u; base < vocab_size; base += 512u) {
            const uint32_t c = base + tid;
            bool hit_eq = false;
            uint16_t raw = 0u;
            if (c < vocab_size) {
                raw = row_raw[c];
                if (bf16_is_finite_u16(raw)) {
                    const uint16_t k16 = bf16_to_ordered_u16(raw);
                    if (k16 > topk_thresh_key || (k16 == topk_thresh_key && c <= topk_max_id)) {
                        if (k16 > p_thresh_key) {
                            const float v = bf16_u16_to_f32(raw);
                            const float score =
                                __fdiv_rn(v, eff_temp) + gumbel_from_base_f32_dev(s_pos_base, c);
                            if (score > best_score || (score == best_score && c < best_id)) {
                                best_score = score;
                                best_id = c;
                            }
                        } else if (k16 == p_thresh_key && s_eq_cnt < p_need_eq) {
                            hit_eq = true;
                        }
                    }
                }
            }
            if (__syncthreads_or(hit_eq)) {
                const unsigned int wm = __ballot_sync(0xFFFFFFFFu, hit_eq);
                if (lane == 0u) {
                    s_warp_cnt[warp] = (uint32_t)__popc(wm);
                }
                __syncthreads();
                const uint32_t prev_eq = s_eq_cnt;
                if (hit_eq) {
                    uint32_t offset = 0u;
                    #pragma unroll
                    for (unsigned int w = 0u; w < 16u; ++w) {
                        if (w < warp) {
                            offset += s_warp_cnt[w];
                        }
                    }
                    const unsigned int lane_mask = (1u << lane) - 1u;
                    const uint32_t r = prev_eq + offset + (uint32_t)__popc(wm & lane_mask);
                    if (r < p_need_eq) {
                        const float v = bf16_u16_to_f32(raw);
                        const float score =
                            __fdiv_rn(v, eff_temp) + gumbel_from_base_f32_dev(s_pos_base, c);
                        if (score > best_score || (score == best_score && c < best_id)) {
                            best_score = score;
                            best_id = c;
                        }
                    }
                }
                __syncthreads();
                if (tid == 0u) {
                    uint32_t chunk_eq = 0u;
                    #pragma unroll
                    for (unsigned int w = 0u; w < 16u; ++w) {
                        chunk_eq += s_warp_cnt[w];
                    }
                    s_eq_cnt = prev_eq + chunk_eq;
                }
                __syncthreads();
            }
        }
    }
    block_reduce_argmax_score_id_512(
        s_warp_primary.val, s_warp_id, warp, lane, best_score, best_id
    );
    if (tid == 0u) {
        out_tokens[row] = (best_id == 0xFFFFFFFFu) ? 0u : best_id;
    }
}

// ── Wave-15 1-Launch GPU Maximal-Coupling Rejection Verify (248K BF16 logits) ─
//
// Unified 3-way per-row contract (1 CTA of 512 threads / 16 warps per row):
//   - `temp < 0.0f`  : instant skip sentinel (`out_words[row] = 0x7FFFFFFFu`, 0 logits reads)
//   - `temp == 0.0f` : fast 1-pass `uint4` exact greedy argmax (skipping mask_tok0/mask_tok1);
//                      bit 31 = 1 iff draft row (`q_row_idx != 0xFFFFu`) matches `draft_token`
//   - `temp > 0.0f`  : fast 2-pass `uint4` radix top_k + min_p + top_p + renorm, division-free
//                      maximal coupling accept (`p_x >= q_x || u_acc * q_x < p_x`), and exact
//                      residual `norm(max(0, p - q))` (or pure `p` on bonus rows).
//
// Strict RNG domain separation (SplitMix64 bijection, non-colliding with drafter 0xDF2D5A31):
//   - accept uniform `u_acc`: `mix64(pos_base(seed, abs_pos) ^ 0xDF2ACC37ull)`
//   - residual base / `u_res`: `pos_base(seed, abs_pos) ^ 0xDF2BE51Dull`

#define REJ_VERIFY_SMEM_MAX_K 512u
#define REJ_VERIFY_GATHER_CAP 2560u
#define REJ_VERIFY_MAX_Q_STRIDE 64u
#define REJ_VERIFY_SKIP_SENTINEL 0x7FFFFFFFu
#define REJ_VERIFY_BONUS_Q_ROW_IDX 0xFFFFu

__device__ __forceinline__ bool rej_is_masked(uint32_t c, uint32_t m0, uint32_t m1) {
    return (c == m0) || (c == m1);
}

__device__ __forceinline__ void rej_hist_subtract_masked_hi(
    unsigned int* __restrict__ s_hist0,
    const uint16_t* __restrict__ row_raw,
    uint32_t vocab,
    uint32_t m0,
    uint32_t m1
) {
    if (m0 < vocab) {
        const uint16_t raw0 = row_raw[m0];
        if (bf16_is_finite_u16(raw0)) {
            const uint32_t hi0 = (uint32_t)bf16_to_ordered_u16(raw0) >> 8;
            if (s_hist0[hi0] > 0u) {
                s_hist0[hi0] -= 1u;
            }
        }
    }
    if (m1 < vocab && m1 != m0) {
        const uint16_t raw1 = row_raw[m1];
        if (bf16_is_finite_u16(raw1)) {
            const uint32_t hi1 = (uint32_t)bf16_to_ordered_u16(raw1) >> 8;
            if (s_hist0[hi1] > 0u) {
                s_hist0[hi1] -= 1u;
            }
        }
    }
}

__device__ __forceinline__ void rej_hist_subtract_masked_lo(
    unsigned int* __restrict__ s_hist0,
    const uint16_t* __restrict__ row_raw,
    uint32_t vocab,
    uint32_t thresh_hi,
    uint32_t m0,
    uint32_t m1
) {
    if (m0 < vocab) {
        const uint16_t raw0 = row_raw[m0];
        if (bf16_is_finite_u16(raw0)) {
            const uint16_t ord0 = bf16_to_ordered_u16(raw0);
            if (((uint32_t)ord0 >> 8) == thresh_hi) {
                const uint32_t lo0 = (uint32_t)ord0 & 0xFFu;
                if (s_hist0[lo0] > 0u) {
                    s_hist0[lo0] -= 1u;
                }
            }
        }
    }
    if (m1 < vocab && m1 != m0) {
        const uint16_t raw1 = row_raw[m1];
        if (bf16_is_finite_u16(raw1)) {
            const uint16_t ord1 = bf16_to_ordered_u16(raw1);
            if (((uint32_t)ord1 >> 8) == thresh_hi) {
                const uint32_t lo1 = (uint32_t)ord1 & 0xFFu;
                if (s_hist0[lo1] > 0u) {
                    s_hist0[lo1] -= 1u;
                }
            }
        }
    }
}

__device__ __forceinline__ void rej_greedy_update(
    uint16_t raw,
    uint32_t idx,
    uint32_t m0,
    uint32_t m1,
    uint32_t& best_ord,
    uint32_t& best_id
) {
    if (!rej_is_masked(idx, m0, m1) && bf16_is_non_nan_u16(raw)) {
        const uint32_t ord = (uint32_t)bf16_to_ordered_u16(raw);
        if (ord > best_ord || (ord == best_ord && idx < best_id)) {
            best_ord = ord;
            best_id = idx;
        }
    }
}

__device__ __forceinline__ void rej_max_finite_ord_update(
    uint16_t raw,
    uint32_t idx,
    uint32_t m0,
    uint32_t m1,
    uint32_t& best_ord
) {
    if (bf16_is_finite_u16(raw) && !rej_is_masked(idx, m0, m1)) {
        const uint32_t ord = (uint32_t)bf16_to_ordered_u16(raw);
        if (ord > best_ord) {
            best_ord = ord;
        }
    }
}

__device__ __forceinline__ void rej_emit_finite(
    uint16_t raw,
    uint32_t idx,
    uint32_t m0,
    uint32_t m1,
    uint32_t cap,
    unsigned int* __restrict__ s_above_cnt,
    unsigned short* __restrict__ s_above_ord,
    unsigned int* __restrict__ s_above_idx
) {
    if (!bf16_is_finite_u16(raw) || rej_is_masked(idx, m0, m1)) {
        return;
    }
    const uint32_t slot = atomicAdd(s_above_cnt, 1u);
    if (slot < cap) {
        s_above_ord[slot] = bf16_to_ordered_u16(raw);
        s_above_idx[slot] = idx;
    }
}

__device__ __forceinline__ void rej_pass2_gather_elem(
    uint16_t raw,
    uint32_t idx,
    uint32_t thresh_hi,
    uint32_t m0,
    uint32_t m1,
    uint32_t above_cap,
    unsigned int* __restrict__ s_above_cnt,
    unsigned short* __restrict__ s_above_ord,
    unsigned int* __restrict__ s_above_idx,
    unsigned int* __restrict__ s_cand_cnt,
    unsigned char* __restrict__ s_cand_lo,
    unsigned int* __restrict__ s_cand_idx,
    unsigned int* __restrict__ s_fine
) {
    if (!bf16_is_finite_u16(raw)) {
        return;
    }
    const uint16_t ord = bf16_to_ordered_u16(raw);
    const uint32_t hi = (uint32_t)(ord >> 8);
    if (hi < thresh_hi || rej_is_masked(idx, m0, m1)) {
        return;
    }
    if (hi > thresh_hi) {
        const uint32_t slot = atomicAdd(s_above_cnt, 1u);
        if (slot < above_cap) {
            s_above_ord[slot] = ord;
            s_above_idx[slot] = idx;
        }
    } else {
        const uint32_t lo = (uint32_t)(ord & 0xFFu);
        const uint32_t slot = atomicAdd(s_cand_cnt, 1u);
        if (slot < REJ_VERIFY_GATHER_CAP) {
            s_cand_lo[slot] = (unsigned char)lo;
            s_cand_idx[slot] = idx;
        }
        atomicAdd(&s_fine[lo], 1u);
    }
}

__device__ __forceinline__ void rej_pass2_hist_elem(
    unsigned int* __restrict__ warp_hist,
    uint16_t raw,
    uint32_t idx,
    uint32_t thresh_hi,
    uint32_t m0,
    uint32_t m1,
    uint32_t above_cap,
    unsigned int* __restrict__ s_above_cnt,
    unsigned short* __restrict__ s_above_ord,
    unsigned int* __restrict__ s_above_idx
) {
    if (!bf16_is_finite_u16(raw)) {
        return;
    }
    const uint16_t ord = bf16_to_ordered_u16(raw);
    const uint32_t hi = (uint32_t)(ord >> 8);
    if (hi < thresh_hi || rej_is_masked(idx, m0, m1)) {
        return;
    }
    if (hi > thresh_hi) {
        const uint32_t slot = atomicAdd(s_above_cnt, 1u);
        if (slot < above_cap) {
            s_above_ord[slot] = ord;
            s_above_idx[slot] = idx;
        }
    } else {
        atomicAdd(&warp_hist[(uint32_t)ord & 0xFFu], 1u);
    }
}

__device__ __forceinline__ void rej_pass3_notie_elem(
    uint16_t raw,
    uint32_t idx,
    uint32_t thresh_hi,
    uint32_t thresh_lo,
    uint32_t m0,
    uint32_t m1,
    uint32_t above_cap,
    unsigned int* __restrict__ s_above_cnt,
    unsigned short* __restrict__ s_above_ord,
    unsigned int* __restrict__ s_above_idx
) {
    if (!bf16_is_finite_u16(raw) || rej_is_masked(idx, m0, m1)) {
        return;
    }
    const uint16_t ord = bf16_to_ordered_u16(raw);
    if (((uint32_t)ord >> 8) == thresh_hi && ((uint32_t)ord & 0xFFu) >= thresh_lo) {
        const uint32_t slot = atomicAdd(s_above_cnt, 1u);
        if (slot < above_cap) {
            s_above_ord[slot] = ord;
            s_above_idx[slot] = idx;
        }
    }
}

__device__ __forceinline__ float rej_lookup_q(
    uint32_t tok,
    const unsigned int* __restrict__ s_q_ids,
    const float* __restrict__ s_q_probs,
    uint32_t q_len
) {
    for (uint32_t j = 0u; j < q_len; ++j) {
        if (s_q_ids[j] == tok) {
            return s_q_probs[j];
        }
    }
    return 0.0f;
}

__device__ __forceinline__ uint32_t rej_inverse_cdf(
    const unsigned int* __restrict__ ids,
    const float* __restrict__ weights,
    uint32_t count,
    float target
) {
    float cum = 0.0f;
    uint32_t selected = ids[0];
    uint32_t last_nonzero = ids[0];
    for (uint32_t i = 0u; i < count; ++i) {
        const float w = weights[i];
        if (w > 0.0f) {
            last_nonzero = ids[i];
            cum += w;
            if (target < cum) {
                return ids[i];
            }
            selected = last_nonzero;
        }
    }
    return selected;
}

__device__ __forceinline__ void rej_prob_hi_elem(
    uint16_t raw,
    uint32_t c,
    uint32_t m0,
    uint32_t m1,
    uint16_t topk_thresh_key,
    uint32_t topk_max_id,
    float eff_temp,
    float max_sc,
    float min_p_w_thresh,
    float* __restrict__ s_prob_hi
) {
    if (!bf16_is_finite_u16(raw) || rej_is_masked(c, m0, m1)) {
        return;
    }
    const uint16_t k16 = bf16_to_ordered_u16(raw);
    if (k16 < topk_thresh_key || (k16 == topk_thresh_key && c > topk_max_id)) {
        return;
    }
    const float w = expf(__fdiv_rn(bf16_u16_to_f32(raw), eff_temp) - max_sc);
    if (w >= min_p_w_thresh && w > 0.0f) {
        atomicAdd(&s_prob_hi[(uint32_t)k16 >> 8], w);
    }
}

__device__ __forceinline__ void rej_prob_lo_elem(
    unsigned int* __restrict__ warp_hist,
    uint16_t raw,
    uint32_t c,
    uint32_t m0,
    uint32_t m1,
    uint16_t topk_thresh_key,
    uint32_t topk_max_id,
    uint32_t p_hi
) {
    if (!bf16_is_finite_u16(raw) || rej_is_masked(c, m0, m1)) {
        return;
    }
    const uint16_t k16 = bf16_to_ordered_u16(raw);
    if ((k16 > topk_thresh_key || (k16 == topk_thresh_key && c <= topk_max_id))
        && (((uint32_t)k16 >> 8) == p_hi)) {
        atomicAdd(&warp_hist[(uint32_t)k16 & 0xFFu], 1u);
    }
}

__device__ __forceinline__ void rej_stream_score_p_elem(
    uint16_t raw,
    uint32_t c,
    uint32_t m0,
    uint32_t m1,
    uint16_t topk_thresh_key,
    uint32_t topk_max_id,
    uint16_t p_thresh_key,
    uint32_t p_max_id,
    float eff_temp,
    float max_sc,
    float min_p_w_thresh,
    uint64_t res_base,
    float& best_score,
    uint32_t& best_id
) {
    if (!bf16_is_finite_u16(raw) || rej_is_masked(c, m0, m1)) {
        return;
    }
    const uint16_t k16 = bf16_to_ordered_u16(raw);
    if (k16 < topk_thresh_key || (k16 == topk_thresh_key && c > topk_max_id)) {
        return;
    }
    if (k16 < p_thresh_key || (k16 == p_thresh_key && c > p_max_id)) {
        return;
    }
    const float sc = __fdiv_rn(bf16_u16_to_f32(raw), eff_temp);
    if (min_p_w_thresh > 0.0f) {
        const float w = expf(sc - max_sc);
        if (w < min_p_w_thresh || !(w > 0.0f)) {
            return;
        }
    }
    const float score = sc + gumbel_from_base_f32_dev(res_base, c);
    if (score > best_score || (score == best_score && c < best_id)) {
        best_score = score;
        best_id = c;
    }
}

__device__ __forceinline__ void rej_stream_score_res_elem(
    uint16_t raw,
    uint32_t c,
    uint32_t m0,
    uint32_t m1,
    uint16_t topk_thresh_key,
    uint32_t topk_max_id,
    uint16_t p_thresh_key,
    uint32_t p_max_id,
    float eff_temp,
    float max_sc,
    float min_p_w_thresh,
    float kept_sum,
    uint64_t res_base,
    const unsigned int* __restrict__ s_q_ids,
    const float* __restrict__ s_q_probs,
    uint32_t q_len,
    float& best_score,
    uint32_t& best_id
) {
    if (!bf16_is_finite_u16(raw) || rej_is_masked(c, m0, m1)) {
        return;
    }
    const uint16_t k16 = bf16_to_ordered_u16(raw);
    if (k16 < topk_thresh_key || (k16 == topk_thresh_key && c > topk_max_id)) {
        return;
    }
    if (k16 < p_thresh_key || (k16 == p_thresh_key && c > p_max_id)) {
        return;
    }
    const float w = expf(__fdiv_rn(bf16_u16_to_f32(raw), eff_temp) - max_sc);
    if (w < min_p_w_thresh || !(w > 0.0f)) {
        return;
    }
    const float q_c = rej_lookup_q(c, s_q_ids, s_q_probs, q_len);
    const float w_res = fmaxf(0.0f, __fdiv_rn(w, kept_sum) - q_c);
    if (w_res > 0.0f) {
        // Exponential-race identity: argmax(log(w_res) - log(-log(u))) == argmax(w_res / -log(u)).
        const uint64_t bits = splitmix64_mix_dev(res_base ^ (uint64_t)c);
        const float u = fminf((float)(bits >> 40) * 0x1.0p-24f + 0x1.0p-25f, 0x1.fffffep-1f);
        const float score = __fdiv_rn(w_res, -logf(u));
        if (score > best_score || (score == best_score && c < best_id)) {
            best_score = score;
            best_id = c;
        }
    }
}

extern "C" __global__ __launch_bounds__(512, 3) void target_rejection_verify_rows_bf16(
    const __nv_bfloat16* __restrict__ logits,       // [num_rows, vocab_size]
    const unsigned int* __restrict__ q_cand_ids,    // [q_row_cap, q_stride] (0xFFFFFFFFu padded)
    const float* __restrict__ q_probs,              // [q_row_cap, q_stride] (0.0f padded)
    const unsigned long long* __restrict__ seeds,   // [num_rows]
    const unsigned long long* __restrict__ abs_pos, // [num_rows]
    const float* __restrict__ temperatures,         // [num_rows]: >0 rejection/bonus, ==0 greedy, <0 skip
    const unsigned int* __restrict__ top_ks,        // [num_rows]
    const float* __restrict__ top_ps,               // [num_rows]
    const float* __restrict__ min_ps,               // [num_rows]
    const unsigned int* __restrict__ draft_tokens,  // [num_rows]
    const unsigned int* __restrict__ q_row_indices, // [num_rows]: 0xFFFFu = bonus/greedy, else row in q_cand_ids/q_probs
    const unsigned int* __restrict__ mask_tok0,     // [num_rows]: 0xFFFFFFFFu = none
    const unsigned int* __restrict__ mask_tok1,     // [num_rows]: 0xFFFFFFFFu = none
    unsigned int* __restrict__ out_words,           // [num_rows]: (accepted << 31) | token_id, or 0x7FFFFFFFu on skip
    int num_rows,
    int vocab_size,
    int q_row_cap,
    int q_stride
) {
    // 16 KiB union + 3 KiB winners + 0.5 KiB q cache + ~0.4 KiB warp/meta = ~19.8 KiB static SMEM
    // (fits 3 CTAs/SM on SM121a's 99 KiB SMEM).
    __shared__ union {
        unsigned int hist[16][256];
        struct {
            unsigned int cand_idx[REJ_VERIFY_GATHER_CAP];
            unsigned int fine[256];
            unsigned char cand_lo[REJ_VERIFY_GATHER_CAP];
        } gather;
        struct {
            float weight[REJ_VERIFY_SMEM_MAX_K];
            float prob[REJ_VERIFY_SMEM_MAX_K];
            float res[REJ_VERIFY_SMEM_MAX_K];
        } dist;
        float prob_hi[256];
    } s_work;
    __shared__ unsigned int s_above_idx[REJ_VERIFY_SMEM_MAX_K];
    __shared__ unsigned short s_above_ord[REJ_VERIFY_SMEM_MAX_K];
    __shared__ unsigned int s_q_ids[REJ_VERIFY_MAX_Q_STRIDE];
    __shared__ float s_q_probs[REJ_VERIFY_MAX_Q_STRIDE];
    __shared__ union {
        float val[16];
        uint32_t ord[16];
    } s_warp_primary;
    __shared__ uint32_t s_warp_id[16];
    __shared__ uint32_t s_warp_cnt[16];
    __shared__ Bf16Topk2PassMeta s_meta;
    __shared__ uint64_t s_res_base;
    __shared__ float s_u_acc;
    __shared__ float s_u_res;
    __shared__ float s_kept_sum;
    __shared__ float s_q_x;
    __shared__ uint32_t s_m0;
    __shared__ uint32_t s_m1;
    __shared__ uint32_t s_q_len;
    __shared__ uint32_t s_is_draft;
    __shared__ uint32_t s_thresh_hi_cnt;
    __shared__ uint32_t s_above_cnt;
    __shared__ uint32_t s_cand_cnt;
    __shared__ uint32_t s_eq_cnt;
    __shared__ uint32_t s_kept;
    __shared__ uint32_t s_id_prefix;
    __shared__ uint32_t s_id_mask;
    __shared__ uint32_t s_id_need;
    __shared__ uint32_t s_p_max_id;
    __shared__ uint32_t s_accepted;
    __shared__ uint32_t s_topk_tkey;
    __shared__ uint32_t s_prologue;
    __shared__ uint32_t s_n_vec;
    __shared__ uint32_t s_vec_end;
    __shared__ float s_max_sc;
    __shared__ float s_min_p_w;
    __shared__ float s_eff_temp;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const uint32_t row = blockIdx.x;

    if (num_rows <= 0 || row >= (uint32_t)num_rows) {
        return;
    }

    const float temp_raw = temperatures[row];

    // Mode 1: `temp < 0.0f` -> instant 0-cycle skip sentinel (0 logits reads).
    if (temp_raw < 0.0f) {
        if (tid == 0u) {
            out_words[row] = REJ_VERIFY_SKIP_SENTINEL;
        }
        return;
    }

    if (vocab_size <= 0) {
        if (tid == 0u) {
            out_words[row] = 0u;
        }
        return;
    }

    const uint32_t vocab = (uint32_t)vocab_size;
    const uint16_t* __restrict__ row_raw =
        reinterpret_cast<const uint16_t*>(logits) + (size_t)row * (size_t)vocab;

    if (tid == 0u) {
        // Align `row_raw + prologue` to 16 bytes for `uint4` loads (8 BF16/load).
        const unsigned int misalign_bytes = (unsigned int)((uintptr_t)row_raw & 15u);
        uint32_t pr = 0u;
        uint32_t nv = 0u;
        uint32_t ve = 0u;
        if ((misalign_bytes & 1u) == 0u) {
            pr = (misalign_bytes == 0u) ? 0u : ((16u - misalign_bytes) >> 1);
            if (pr > vocab) {
                pr = vocab;
            }
            const uint32_t rem = vocab - pr;
            nv = rem >> 3;
            ve = pr + (nv << 3);
        }
        s_prologue = pr;
        s_n_vec = nv;
        s_vec_end = ve;
        s_m0 = (mask_tok0 != nullptr) ? mask_tok0[row] : 0xFFFFFFFFu;
        s_m1 = (mask_tok1 != nullptr) ? mask_tok1[row] : 0xFFFFFFFFu;
    }

    // Mode 2: `temp == 0.0f` (or non-finite defensive fallback) -> fast 1-pass `uint4`
    // 512-thread exact greedy argmax (skipping mask_tok0/mask_tok1, matching argmax_bf16_rows).
    if (!(temp_raw > 0.0f) || !isfinite(temp_raw)) {
        __syncthreads();
        const uint32_t pr = s_prologue;
        const uint32_t nv = s_n_vec;
        const uint32_t ve = s_vec_end;
        const uint32_t m0 = s_m0;
        const uint32_t m1 = s_m1;
        uint32_t best_ord = 0u;
        uint32_t best_id = 0xFFFFFFFFu;
        if (tid < pr) {
            rej_greedy_update(row_raw[tid], tid, m0, m1, best_ord, best_id);
        }
        if (nv > 0u) {
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + pr);
            for (uint32_t idx = tid; idx < nv; idx += 512u) {
                const uint4 v4 = row_v4[idx];
                const uint32_t c0 = pr + (idx << 3);
                rej_greedy_update((uint16_t)(v4.x & 0xFFFFu), c0 + 0u, m0, m1, best_ord, best_id);
                rej_greedy_update((uint16_t)(v4.x >> 16),     c0 + 1u, m0, m1, best_ord, best_id);
                rej_greedy_update((uint16_t)(v4.y & 0xFFFFu), c0 + 2u, m0, m1, best_ord, best_id);
                rej_greedy_update((uint16_t)(v4.y >> 16),     c0 + 3u, m0, m1, best_ord, best_id);
                rej_greedy_update((uint16_t)(v4.z & 0xFFFFu), c0 + 4u, m0, m1, best_ord, best_id);
                rej_greedy_update((uint16_t)(v4.z >> 16),     c0 + 5u, m0, m1, best_ord, best_id);
                rej_greedy_update((uint16_t)(v4.w & 0xFFFFu), c0 + 6u, m0, m1, best_ord, best_id);
                rej_greedy_update((uint16_t)(v4.w >> 16),     c0 + 7u, m0, m1, best_ord, best_id);
            }
        }
        for (uint32_t c = ve + tid; c < vocab; c += 512u) {
            rej_greedy_update(row_raw[c], c, m0, m1, best_ord, best_id);
        }
        block_reduce_argmax_u32_id_512(
            s_warp_primary.ord, s_warp_id, warp, lane, best_ord, best_id
        );
        if (tid == 0u) {
            const uint32_t chosen = (best_id == 0xFFFFFFFFu) ? 0u : best_id;
            const uint32_t q_row_idx = q_row_indices[row];
            const bool accepted = (q_row_idx != REJ_VERIFY_BONUS_Q_ROW_IDX)
                && (chosen == draft_tokens[row]);
            out_words[row] = (accepted ? (1u << 31) : 0u) | (chosen & 0x7FFFFFFFu);
        }
        return;
    }

    // Mode 3: `temp > 0.0f` -> Maximal-Coupling Rejection Verify / Bonus Sample.
    {
        const uint32_t q_row_idx = q_row_indices[row];
        const bool is_draft = (q_row_idx != REJ_VERIFY_BONUS_Q_ROW_IDX)
            && (q_row_cap > 0)
            && (q_row_idx < (uint32_t)q_row_cap)
            && (q_stride > 0)
            && (q_cand_ids != nullptr)
            && (q_probs != nullptr);
        const uint32_t q_len = is_draft
            ? (((uint32_t)q_stride < REJ_VERIFY_MAX_Q_STRIDE)
                   ? (uint32_t)q_stride
                   : REJ_VERIFY_MAX_Q_STRIDE)
            : 0u;

        if (tid == 0u) {
            const uint64_t pos_base = gumbel_pos_base_dev(seeds[row], abs_pos[row]);
            const uint64_t h_acc = splitmix64_mix_dev(pos_base ^ 0xDF2ACC37ull);
            const uint64_t res_base = pos_base ^ 0xDF2BE51Dull;
            const uint64_t h_res = splitmix64_mix_dev(res_base);
            s_u_acc = (float)(h_acc >> 40) * 0x1.0p-24f;
            s_res_base = res_base;
            s_u_res = (float)(h_res >> 40) * 0x1.0p-24f;
            s_q_len = q_len;
            s_is_draft = is_draft ? 1u : 0u;
            s_eff_temp = fmaxf(temp_raw, 1.0e-6f);
        }

        // Cooperatively load draft candidate distribution `(q_cand_ids, q_probs)` into SMEM.
        if (tid < q_len) {
            const size_t q_off = (size_t)q_row_idx * (size_t)q_stride + (size_t)tid;
            const uint32_t qid = q_cand_ids[q_off];
            const float qp = q_probs[q_off];
            s_q_ids[tid] = qid;
            s_q_probs[tid] = (qid != 0xFFFFFFFFu && qp > 0.0f && isfinite(qp)) ? qp : 0.0f;
        }
    }
    __syncthreads();

    const uint32_t top_k = top_ks[row];
    const bool has_top_k = (top_k > 0u && top_k < vocab);
    const uint32_t eff_k_limit = has_top_k ? top_k : vocab;
    unsigned int* __restrict__ warp_hist = s_work.hist[warp];

    // Regime A: Bonus row with full-vocab untruncated `p` -> 1-pass streaming Gumbel-Max!
    if (s_is_draft == 0u && !has_top_k) {
        const float tp_a = top_ps[row];
        const float mp_a = min_ps[row];
        if (!(tp_a > 0.0f && tp_a < 1.0f) && !(mp_a > 0.0f && isfinite(mp_a))) {
            const float eff_temp_a = s_eff_temp;
            float best_score = -CUDART_INF_F;
            uint32_t best_id = 0xFFFFFFFFu;
            if (tid < s_prologue && !rej_is_masked(tid, s_m0, s_m1)) {
                target_score_if_ge(row_raw[tid], tid, 0u, eff_temp_a, s_res_base, best_score, best_id);
            }
            if (s_n_vec > 0u) {
                const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + s_prologue);
                for (uint32_t idx = tid; idx < s_n_vec; idx += 512u) {
                    const uint4 v4 = row_v4[idx];
                    const uint32_t c0 = s_prologue + (idx << 3);
                    #pragma unroll 1
                    for (uint32_t h = 0u; h < 8u; ++h) {
                        const uint32_t c = c0 + h;
                        if (!rej_is_masked(c, s_m0, s_m1)) {
                            const uint32_t q = h >> 1;
                            const uint32_t w = (q == 0u) ? v4.x : ((q == 1u) ? v4.y : ((q == 2u) ? v4.z : v4.w));
                            const uint16_t raw = (uint16_t)((h & 1u) ? (w >> 16) : (w & 0xFFFFu));
                            target_score_if_ge(raw, c, 0u, eff_temp_a, s_res_base, best_score, best_id);
                        }
                    }
                }
            }
            for (uint32_t c = s_vec_end + tid; c < vocab; c += 512u) {
                if (!rej_is_masked(c, s_m0, s_m1)) {
                    target_score_if_ge(row_raw[c], c, 0u, eff_temp_a, s_res_base, best_score, best_id);
                }
            }
            block_reduce_argmax_score_id_512(
                s_warp_primary.val, s_warp_id, warp, lane, best_score, best_id
            );
            if (tid == 0u) {
                const uint32_t chosen = (best_id == 0xFFFFFFFFu) ? 0u : best_id;
                out_words[row] = chosen & 0x7FFFFFFFu;
            }
            return;
        }
    }

    // Regime B (hot path: `top_k <= 512`, or `vocab <= 512`):
    // Fast 2-pass `uint4` 512-thread radix top-k + fused gather + in-SMEM exact tie-break +
    // exact post-policy renorm + division-free maximal coupling accept/reject + inverse-CDF.
    if (eff_k_limit <= REJ_VERIFY_SMEM_MAX_K) {
        #pragma unroll
        for (unsigned int i = tid; i < 4096u; i += 512u) {
            s_work.hist[i >> 8][i & 255u] = 0u;
        }
        if (tid == 0u) {
            s_above_cnt = 0u;
            s_cand_cnt = 0u;
            s_eq_cnt = 0u;
        }
        __syncthreads();

        // Pass 1: upper-byte (ord >> 8) per-warp privatized SMEM-atomic histogram.
        if (tid < s_prologue) {
            const uint16_t raw = row_raw[tid];
            if (bf16_is_finite_u16(raw)) {
                atomicAdd(&warp_hist[(uint32_t)bf16_to_ordered_u16(raw) >> 8], 1u);
            }
        }
        if (s_n_vec > 0u) {
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + s_prologue);
            for (uint32_t idx = tid; idx < s_n_vec; idx += 512u) {
                const uint4 v4 = row_v4[idx];
                target_pass1_pair(warp_hist, v4.x);
                target_pass1_pair(warp_hist, v4.y);
                target_pass1_pair(warp_hist, v4.z);
                target_pass1_pair(warp_hist, v4.w);
            }
        }
        for (uint32_t c = s_vec_end + tid; c < vocab; c += 512u) {
            const uint16_t raw = row_raw[c];
            if (bf16_is_finite_u16(raw)) {
                atomicAdd(&warp_hist[(uint32_t)bf16_to_ordered_u16(raw) >> 8], 1u);
            }
        }
        __syncthreads();

        // Merge 16 warp histograms into `s_work.hist[0][0..255]`.
        if (tid < 256u) {
            unsigned int sum = 0u;
            #pragma unroll
            for (unsigned int w = 0u; w < 16u; ++w) {
                sum += s_work.hist[w][tid];
            }
            s_work.hist[0][tid] = sum;
        }
        __syncthreads();

        if (tid == 0u) {
            rej_hist_subtract_masked_hi(s_work.hist[0], row_raw, vocab, s_m0, s_m1);
            uint32_t total = 0u;
            for (int b = 255; b >= 0; --b) {
                total += s_work.hist[0][b];
            }
            s_meta.total_valid = total;
            if (total > eff_k_limit) {
                uint32_t above = 0u;
                int hi = 255;
                for (; hi > 0; --hi) {
                    if (above + s_work.hist[0][hi] >= eff_k_limit) {
                        break;
                    }
                    above += s_work.hist[0][hi];
                }
                s_meta.win_hi = (uint32_t)hi;
                s_meta.above_hi = above;
                s_meta.need_in_hi = eff_k_limit - above;
                s_thresh_hi_cnt = s_work.hist[0][hi];
            } else {
                s_meta.win_hi = 0u;
                s_meta.above_hi = 0u;
                s_meta.need_in_hi = 0u;
                s_thresh_hi_cnt = 0u;
            }
        }
        __syncthreads();

        const uint32_t total_valid = s_meta.total_valid;
        if (total_valid == 0u) {
            if (tid == 0u) {
                out_words[row] = 0u;
            }
            return;
        }
        const uint32_t eff_k = (total_valid < eff_k_limit) ? total_valid : eff_k_limit;

        if (total_valid <= eff_k_limit) {
            if (tid < s_prologue) {
                rej_emit_finite(row_raw[tid], tid, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
            }
            if (s_n_vec > 0u) {
                const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + s_prologue);
                for (uint32_t idx = tid; idx < s_n_vec; idx += 512u) {
                    const uint4 v4 = row_v4[idx];
                    const uint32_t c0 = s_prologue + (idx << 3);
                    rej_emit_finite((uint16_t)(v4.x & 0xFFFFu), c0 + 0u, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    rej_emit_finite((uint16_t)(v4.x >> 16),     c0 + 1u, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    rej_emit_finite((uint16_t)(v4.y & 0xFFFFu), c0 + 2u, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    rej_emit_finite((uint16_t)(v4.y >> 16),     c0 + 3u, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    rej_emit_finite((uint16_t)(v4.z & 0xFFFFu), c0 + 4u, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    rej_emit_finite((uint16_t)(v4.z >> 16),     c0 + 5u, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    rej_emit_finite((uint16_t)(v4.w & 0xFFFFu), c0 + 6u, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    rej_emit_finite((uint16_t)(v4.w >> 16),     c0 + 7u, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                }
            }
            for (uint32_t c = s_vec_end + tid; c < vocab; c += 512u) {
                rej_emit_finite(row_raw[c], c, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
            }
            __syncthreads();
        } else if (s_thresh_hi_cnt <= REJ_VERIFY_GATHER_CAP) {
            const uint32_t thresh_hi = s_meta.win_hi;
            if (tid < 256u) {
                s_work.gather.fine[tid] = 0u;
            }
            __syncthreads();

            if (tid < s_prologue) {
                rej_pass2_gather_elem(
                    row_raw[tid], tid, thresh_hi, s_m0, s_m1, eff_k,
                    &s_above_cnt, s_above_ord, s_above_idx,
                    &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine
                );
            }
            if (s_n_vec > 0u) {
                const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + s_prologue);
                for (uint32_t idx = tid; idx < s_n_vec; idx += 512u) {
                    const uint4 v4 = row_v4[idx];
                    const uint32_t c0 = s_prologue + (idx << 3);
                    rej_pass2_gather_elem((uint16_t)(v4.x & 0xFFFFu), c0 + 0u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                    rej_pass2_gather_elem((uint16_t)(v4.x >> 16),     c0 + 1u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                    rej_pass2_gather_elem((uint16_t)(v4.y & 0xFFFFu), c0 + 2u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                    rej_pass2_gather_elem((uint16_t)(v4.y >> 16),     c0 + 3u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                    rej_pass2_gather_elem((uint16_t)(v4.z & 0xFFFFu), c0 + 4u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                    rej_pass2_gather_elem((uint16_t)(v4.z >> 16),     c0 + 5u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                    rej_pass2_gather_elem((uint16_t)(v4.w & 0xFFFFu), c0 + 6u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                    rej_pass2_gather_elem((uint16_t)(v4.w >> 16),     c0 + 7u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx, &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine);
                }
            }
            for (uint32_t c = s_vec_end + tid; c < vocab; c += 512u) {
                rej_pass2_gather_elem(
                    row_raw[c], c, thresh_hi, s_m0, s_m1, eff_k,
                    &s_above_cnt, s_above_ord, s_above_idx,
                    &s_cand_cnt, s_work.gather.cand_lo, s_work.gather.cand_idx, s_work.gather.fine
                );
            }
            __syncthreads();

            if (tid == 0u) {
                const uint32_t need = s_meta.need_in_hi;
                uint32_t above_lo = 0u;
                int lo = 255;
                for (; lo > 0; --lo) {
                    if (above_lo + s_work.gather.fine[lo] >= need) {
                        break;
                    }
                    above_lo += s_work.gather.fine[lo];
                }
                s_meta.thresh_key = (thresh_hi << 8) | (uint32_t)lo;
                s_meta.need_eq = need - above_lo;
                s_meta.count_eq = s_work.gather.fine[lo];
            }
            __syncthreads();

            const uint32_t n_cand =
                (s_cand_cnt < REJ_VERIFY_GATHER_CAP) ? s_cand_cnt : REJ_VERIFY_GATHER_CAP;
            const uint32_t thresh_lo = s_meta.thresh_key & 0xFFu;
            const uint32_t need_eq = s_meta.need_eq;
            const uint32_t count_eq = s_meta.count_eq;

            uint32_t eq_id_thresh = 0xFFFFFFFFu;
            if (count_eq > need_eq) {
                if (tid == 0u) {
                    s_id_prefix = 0u;
                    s_id_mask = 0u;
                    s_id_need = need_eq;
                }
                __syncthreads();
                const int start_shift = (vocab <= 0x10000u)
                    ? 8
                    : ((vocab <= 0x1000000u) ? 16 : 24);
                for (int shift = start_shift; shift >= 0; shift -= 8) {
                    if (tid < 256u) {
                        s_work.gather.fine[tid] = 0u;
                    }
                    __syncthreads();
                    const uint32_t prefix = s_id_prefix;
                    const uint32_t mask = s_id_mask;
                    for (uint32_t i = tid; i < n_cand; i += 512u) {
                        if ((uint32_t)s_work.gather.cand_lo[i] == thresh_lo) {
                            const uint32_t cid = s_work.gather.cand_idx[i];
                            if ((cid & mask) == prefix) {
                                atomicAdd(&s_work.gather.fine[(cid >> shift) & 0xFFu], 1u);
                            }
                        }
                    }
                    __syncthreads();
                    if (tid == 0u) {
                        const uint32_t need = s_id_need;
                        uint32_t below = 0u;
                        int d = 0;
                        for (; d < 255; ++d) {
                            if (below + s_work.gather.fine[d] >= need) {
                                break;
                            }
                            below += s_work.gather.fine[d];
                        }
                        s_id_need = need - below;
                        s_id_prefix = prefix | ((uint32_t)d << shift);
                        s_id_mask = mask | (0xFFu << shift);
                    }
                    __syncthreads();
                }
                eq_id_thresh = s_id_prefix;
            }

            for (uint32_t i = tid; i < n_cand; i += 512u) {
                const uint32_t lo = (uint32_t)s_work.gather.cand_lo[i];
                const uint32_t cid = s_work.gather.cand_idx[i];
                if (lo > thresh_lo || (lo == thresh_lo && cid <= eq_id_thresh)) {
                    const uint32_t slot = atomicAdd(&s_above_cnt, 1u);
                    if (slot < eff_k) {
                        s_above_ord[slot] = (uint16_t)((thresh_hi << 8) | lo);
                        s_above_idx[slot] = cid;
                    }
                }
            }
            __syncthreads();
        } else {
            const uint32_t thresh_hi = s_meta.win_hi;
            #pragma unroll
            for (unsigned int i = tid; i < 4096u; i += 512u) {
                s_work.hist[i >> 8][i & 255u] = 0u;
            }
            __syncthreads();

            if (tid < s_prologue) {
                rej_pass2_hist_elem(
                    warp_hist, row_raw[tid], tid, thresh_hi, s_m0, s_m1, eff_k,
                    &s_above_cnt, s_above_ord, s_above_idx
                );
            }
            if (s_n_vec > 0u) {
                const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + s_prologue);
                for (uint32_t idx = tid; idx < s_n_vec; idx += 512u) {
                    const uint4 v4 = row_v4[idx];
                    const uint32_t c0 = s_prologue + (idx << 3);
                    rej_pass2_hist_elem(warp_hist, (uint16_t)(v4.x & 0xFFFFu), c0 + 0u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    rej_pass2_hist_elem(warp_hist, (uint16_t)(v4.x >> 16),     c0 + 1u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    rej_pass2_hist_elem(warp_hist, (uint16_t)(v4.y & 0xFFFFu), c0 + 2u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    rej_pass2_hist_elem(warp_hist, (uint16_t)(v4.y >> 16),     c0 + 3u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    rej_pass2_hist_elem(warp_hist, (uint16_t)(v4.z & 0xFFFFu), c0 + 4u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    rej_pass2_hist_elem(warp_hist, (uint16_t)(v4.z >> 16),     c0 + 5u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    rej_pass2_hist_elem(warp_hist, (uint16_t)(v4.w & 0xFFFFu), c0 + 6u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    rej_pass2_hist_elem(warp_hist, (uint16_t)(v4.w >> 16),     c0 + 7u, thresh_hi, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                }
            }
            for (uint32_t c = s_vec_end + tid; c < vocab; c += 512u) {
                rej_pass2_hist_elem(
                    warp_hist, row_raw[c], c, thresh_hi, s_m0, s_m1, eff_k,
                    &s_above_cnt, s_above_ord, s_above_idx
                );
            }
            __syncthreads();

            if (tid < 256u) {
                unsigned int sum = 0u;
                #pragma unroll
                for (unsigned int w = 0u; w < 16u; ++w) {
                    sum += s_work.hist[w][tid];
                }
                s_work.hist[0][tid] = sum;
            }
            __syncthreads();

            if (tid == 0u) {
                const uint32_t need = s_meta.need_in_hi;
                uint32_t above_lo = 0u;
                int lo = 255;
                for (; lo > 0; --lo) {
                    if (above_lo + s_work.hist[0][lo] >= need) {
                        break;
                    }
                    above_lo += s_work.hist[0][lo];
                }
                s_meta.thresh_key = (thresh_hi << 8) | (uint32_t)lo;
                s_meta.strict_gt = above_lo;
                s_meta.need_eq = need - above_lo;
                s_meta.count_eq = s_work.hist[0][lo];
            }
            __syncthreads();

            const uint32_t thresh_lo = s_meta.thresh_key & 0xFFu;
            const uint32_t above_lo = s_meta.strict_gt;
            const uint32_t need_eq = s_meta.need_eq;
            const uint32_t count_eq = s_meta.count_eq;

            if (count_eq <= need_eq) {
                if (tid < s_prologue) {
                    rej_pass3_notie_elem(
                        row_raw[tid], tid, thresh_hi, thresh_lo, s_m0, s_m1, eff_k,
                        &s_above_cnt, s_above_ord, s_above_idx
                    );
                }
                if (s_n_vec > 0u) {
                    const uint4* __restrict__ row_v4 =
                        reinterpret_cast<const uint4*>(row_raw + s_prologue);
                    for (uint32_t idx = tid; idx < s_n_vec; idx += 512u) {
                        const uint4 v4 = row_v4[idx];
                        const uint32_t c0 = s_prologue + (idx << 3);
                        rej_pass3_notie_elem((uint16_t)(v4.x & 0xFFFFu), c0 + 0u, thresh_hi, thresh_lo, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                        rej_pass3_notie_elem((uint16_t)(v4.x >> 16),     c0 + 1u, thresh_hi, thresh_lo, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                        rej_pass3_notie_elem((uint16_t)(v4.y & 0xFFFFu), c0 + 2u, thresh_hi, thresh_lo, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                        rej_pass3_notie_elem((uint16_t)(v4.y >> 16),     c0 + 3u, thresh_hi, thresh_lo, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                        rej_pass3_notie_elem((uint16_t)(v4.z & 0xFFFFu), c0 + 4u, thresh_hi, thresh_lo, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                        rej_pass3_notie_elem((uint16_t)(v4.z >> 16),     c0 + 5u, thresh_hi, thresh_lo, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                        rej_pass3_notie_elem((uint16_t)(v4.w & 0xFFFFu), c0 + 6u, thresh_hi, thresh_lo, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                        rej_pass3_notie_elem((uint16_t)(v4.w >> 16),     c0 + 7u, thresh_hi, thresh_lo, s_m0, s_m1, eff_k, &s_above_cnt, s_above_ord, s_above_idx);
                    }
                }
                for (uint32_t c = s_vec_end + tid; c < vocab; c += 512u) {
                    rej_pass3_notie_elem(
                        row_raw[c], c, thresh_hi, thresh_lo, s_m0, s_m1, eff_k,
                        &s_above_cnt, s_above_ord, s_above_idx
                    );
                }
                __syncthreads();
            } else {
                const uint16_t thresh_key = (uint16_t)s_meta.thresh_key;
                for (uint32_t base = 0u; base < vocab; base += 512u) {
                    if (above_lo == 0u && s_eq_cnt >= need_eq) {
                        break;
                    }
                    const uint32_t c = base + tid;
                    bool hit_eq = false;
                    if (c < vocab && !rej_is_masked(c, s_m0, s_m1)) {
                        const uint16_t raw = row_raw[c];
                        if (bf16_is_finite_u16(raw)) {
                            const uint16_t ord = bf16_to_ordered_u16(raw);
                            if (((uint32_t)ord >> 8) == thresh_hi) {
                                const uint32_t lo = (uint32_t)ord & 0xFFu;
                                if (lo > thresh_lo) {
                                    const uint32_t slot = atomicAdd(&s_above_cnt, 1u);
                                    if (slot < eff_k) {
                                        s_above_ord[slot] = ord;
                                        s_above_idx[slot] = c;
                                    }
                                } else if (lo == thresh_lo && s_eq_cnt < need_eq) {
                                    hit_eq = true;
                                }
                            }
                        }
                    }
                    if (__syncthreads_or(hit_eq)) {
                        const unsigned int wm = __ballot_sync(0xFFFFFFFFu, hit_eq);
                        if (lane == 0u) {
                            s_warp_cnt[warp] = (uint32_t)__popc(wm);
                        }
                        __syncthreads();
                        const uint32_t prev_eq = s_eq_cnt;
                        if (hit_eq) {
                            uint32_t offset = 0u;
                            #pragma unroll
                            for (unsigned int w = 0u; w < 16u; ++w) {
                                if (w < warp) {
                                    offset += s_warp_cnt[w];
                                }
                            }
                            const unsigned int lane_mask = (1u << lane) - 1u;
                            const uint32_t r =
                                prev_eq + offset + (uint32_t)__popc(wm & lane_mask);
                            if (r < need_eq) {
                                const uint32_t slot = atomicAdd(&s_above_cnt, 1u);
                                if (slot < eff_k) {
                                    s_above_ord[slot] = thresh_key;
                                    s_above_idx[slot] = c;
                                }
                            }
                        }
                        __syncthreads();
                        if (tid == 0u) {
                            uint32_t chunk_eq = 0u;
                            #pragma unroll
                            for (unsigned int w = 0u; w < 16u; ++w) {
                                chunk_eq += s_warp_cnt[w];
                            }
                            s_eq_cnt = prev_eq + chunk_eq;
                        }
                        __syncthreads();
                    }
                }
                __syncthreads();
            }
        }

        const uint32_t count = (s_above_cnt < eff_k) ? s_above_cnt : eff_k;
        if (count == 0u) {
            if (tid == 0u) {
                out_words[row] = 0u;
            }
            return;
        }

        // Parallel bitonic sort descending by `(ord DESC, id ASC)` over `<= 512` items.
        uint32_t n_pad = 1u;
        while (n_pad < count) {
            n_pad <<= 1u;
        }
        if (tid >= count && tid < n_pad) {
            s_above_ord[tid] = 0u;
            s_above_idx[tid] = 0xFFFFFFFFu;
        }
        __syncthreads();

        for (uint32_t size = 2u; size <= n_pad; size <<= 1u) {
            for (uint32_t stride = size >> 1u; stride > 0u; stride >>= 1u) {
                if (tid < (n_pad >> 1u)) {
                    const uint32_t block_lo = tid & (stride - 1u);
                    const uint32_t i = ((tid - block_lo) << 1u) + block_lo;
                    const uint32_t j = i + stride;
                    const bool want_desc = ((i & size) == 0u);
                    const uint16_t oa = s_above_ord[i];
                    const uint32_t ida = s_above_idx[i];
                    const uint16_t ob = s_above_ord[j];
                    const uint32_t idb = s_above_idx[j];
                    const bool a_before_b = target_cand_comes_before(oa, ida, ob, idb);
                    if (want_desc ? !a_before_b : a_before_b) {
                        s_above_ord[i] = ob;
                        s_above_ord[j] = oa;
                        s_above_idx[i] = idb;
                        s_above_idx[j] = ida;
                    }
                }
                __syncthreads();
            }
        }

        // Compute unnormalized softmax weights in parallel.
        const float eff_temp_b = s_eff_temp;
        const float max_sc = __fdiv_rn(
            bf16_u16_to_f32(ordered_u16_to_bf16(s_above_ord[0])),
            eff_temp_b
        );
        if (tid < count) {
            const float sc = __fdiv_rn(
                bf16_u16_to_f32(ordered_u16_to_bf16(s_above_ord[tid])),
                eff_temp_b
            );
            s_work.dist.weight[tid] = expf(sc - max_sc);
        }
        __syncthreads();

        // Exact `min_p` + `top_p` nucleus cutoff + post-truncation denominator.
        if (tid == 0u) {
            const float top_p_b = top_ps[row];
            const float min_p_b = min_ps[row];
            uint32_t count_minp = count;
            if (min_p_b > 0.0f && isfinite(min_p_b)) {
                const float min_thresh = min_p_b * s_work.dist.weight[0];
                while (count_minp > 1u && s_work.dist.weight[count_minp - 1u] < min_thresh) {
                    --count_minp;
                }
            }
            uint32_t kept = count_minp;
            if (top_p_b > 0.0f && top_p_b < 1.0f) {
                float sum = 0.0f;
                for (uint32_t i = 0u; i < count_minp; ++i) {
                    sum += s_work.dist.weight[i];
                }
                if (sum > 0.0f) {
                    float cum = 0.0f;
                    for (uint32_t i = 0u; i < count_minp; ++i) {
                        cum += __fdiv_rn(s_work.dist.weight[i], sum);
                        if (cum >= top_p_b) {
                            kept = i + 1u;
                            break;
                        }
                    }
                }
            }
            float kept_sum = 0.0f;
            for (uint32_t i = 0u; i < kept; ++i) {
                kept_sum += s_work.dist.weight[i];
            }
            s_kept = kept;
            s_kept_sum = kept_sum;
        }
        __syncthreads();

        // Renormalize `p[0..kept]` and precompute `max(0, p[i] - q[i])` in parallel.
        const uint32_t kept = s_kept;
        const float kept_sum = s_kept_sum;
        const bool valid_sum = (kept_sum > 0.0f) && isfinite(kept_sum);
        if (tid < kept) {
            const float p_i = valid_sum
                ? __fdiv_rn(s_work.dist.weight[tid], kept_sum)
                : ((tid == 0u) ? 1.0f : 0.0f);
            s_work.dist.prob[tid] = p_i;
            if (s_is_draft != 0u) {
                const float q_i = rej_lookup_q(s_above_idx[tid], s_q_ids, s_q_probs, s_q_len);
                s_work.dist.res[tid] = fmaxf(0.0f, p_i - q_i);
            }
        }
        __syncthreads();

        // Lane 0 executes division-free exact accept test + exact inverse-CDF fallback/bonus.
        if (tid == 0u) {
            const float u_res = s_u_res;
            if (s_is_draft != 0u) {
                const uint32_t draft_tok = draft_tokens[row];
                const float q_x = rej_lookup_q(draft_tok, s_q_ids, s_q_probs, s_q_len);
                float p_x = 0.0f;
                for (uint32_t i = 0u; i < kept; ++i) {
                    if (s_above_idx[i] == draft_tok) {
                        p_x = s_work.dist.prob[i];
                        break;
                    }
                }
                if (q_x > 0.0f && p_x > 0.0f && (p_x >= q_x || s_u_acc * q_x < p_x)) {
                    out_words[row] = (1u << 31) | (draft_tok & 0x7FFFFFFFu);
                    return;
                }
                uint32_t sampled = s_above_idx[0];
                if (q_x > 0.0f) {
                    float res_sum = 0.0f;
                    for (uint32_t i = 0u; i < kept; ++i) {
                        res_sum += s_work.dist.res[i];
                    }
                    if (res_sum > 1.0e-12f && isfinite(res_sum)) {
                        sampled = rej_inverse_cdf(s_above_idx, s_work.dist.res, kept, u_res * res_sum);
                    } else {
                        float p_sum = 0.0f;
                        for (uint32_t i = 0u; i < kept; ++i) {
                            p_sum += s_work.dist.prob[i];
                        }
                        sampled = rej_inverse_cdf(s_above_idx, s_work.dist.prob, kept, u_res * p_sum);
                    }
                } else {
                    float p_sum = 0.0f;
                    for (uint32_t i = 0u; i < kept; ++i) {
                        p_sum += s_work.dist.prob[i];
                    }
                    sampled = rej_inverse_cdf(s_above_idx, s_work.dist.prob, kept, u_res * p_sum);
                }
                out_words[row] = sampled & 0x7FFFFFFFu;
            } else {
                float p_sum = 0.0f;
                for (uint32_t i = 0u; i < kept; ++i) {
                    p_sum += s_work.dist.prob[i];
                }
                const uint32_t sampled =
                    rej_inverse_cdf(s_above_idx, s_work.dist.prob, kept, u_res * p_sum);
                out_words[row] = sampled & 0x7FFFFFFFu;
            }
        }
        return;
    }

    // Regime C: General `|S_p| > 512` (`top_k == 0` or `top_k > 512`).
    uint16_t topk_thresh_key = 0u;
    uint32_t topk_need_eq = 0u;
    bool topk_need_tiebreak = false;

    if (has_top_k) {
        #pragma unroll
        for (unsigned int i = tid; i < 4096u; i += 512u) {
            s_work.hist[i >> 8][i & 255u] = 0u;
        }
        __syncthreads();

        if (tid < s_prologue) {
            const uint16_t raw = row_raw[tid];
            if (bf16_is_finite_u16(raw)) {
                atomicAdd(&warp_hist[(uint32_t)bf16_to_ordered_u16(raw) >> 8], 1u);
            }
        }
        if (s_n_vec > 0u) {
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + s_prologue);
            for (uint32_t idx = tid; idx < s_n_vec; idx += 512u) {
                const uint4 v4 = row_v4[idx];
                target_pass1_pair(warp_hist, v4.x);
                target_pass1_pair(warp_hist, v4.y);
                target_pass1_pair(warp_hist, v4.z);
                target_pass1_pair(warp_hist, v4.w);
            }
        }
        for (uint32_t c = s_vec_end + tid; c < vocab; c += 512u) {
            const uint16_t raw = row_raw[c];
            if (bf16_is_finite_u16(raw)) {
                atomicAdd(&warp_hist[(uint32_t)bf16_to_ordered_u16(raw) >> 8], 1u);
            }
        }
        __syncthreads();

        if (tid < 256u) {
            unsigned int sum = 0u;
            #pragma unroll
            for (unsigned int w = 0u; w < 16u; ++w) {
                sum += s_work.hist[w][tid];
            }
            s_work.hist[0][tid] = sum;
        }
        __syncthreads();

        if (tid == 0u) {
            rej_hist_subtract_masked_hi(s_work.hist[0], row_raw, vocab, s_m0, s_m1);
            uint32_t total = 0u;
            for (int b = 255; b >= 0; --b) {
                total += s_work.hist[0][b];
            }
            s_meta.total_valid = total;
            if (total <= top_k) {
                s_meta.thresh_key = 0u;
                s_meta.need_eq = 0u;
                s_meta.count_eq = 0u;
                s_meta.run_pass2 = 0u;
            } else {
                uint32_t above = 0u;
                int hi = 255;
                for (; hi > 0; --hi) {
                    if (above + s_work.hist[0][hi] >= top_k) {
                        break;
                    }
                    above += s_work.hist[0][hi];
                }
                s_meta.win_hi = (uint32_t)hi;
                s_meta.above_hi = above;
                s_meta.need_in_hi = top_k - above;
                if (above + s_work.hist[0][hi] == top_k) {
                    s_meta.thresh_key = ((uint32_t)hi) << 8;
                    s_meta.need_eq = 0u;
                    s_meta.count_eq = 0u;
                    s_meta.run_pass2 = 0u;
                } else {
                    s_meta.run_pass2 = 1u;
                }
            }
        }
        __syncthreads();

        if (s_meta.total_valid == 0u) {
            if (tid == 0u) {
                out_words[row] = 0u;
            }
            return;
        }

        if (s_meta.run_pass2 != 0u) {
            const uint32_t thresh_hi = s_meta.win_hi;
            #pragma unroll
            for (unsigned int i = tid; i < 4096u; i += 512u) {
                s_work.hist[i >> 8][i & 255u] = 0u;
            }
            __syncthreads();

            if (tid < s_prologue) {
                const uint16_t raw = row_raw[tid];
                if (bf16_is_finite_u16(raw)) {
                    const uint16_t ord = bf16_to_ordered_u16(raw);
                    if (((uint32_t)ord >> 8) == thresh_hi) {
                        atomicAdd(&warp_hist[(uint32_t)ord & 0xFFu], 1u);
                    }
                }
            }
            if (s_n_vec > 0u) {
                const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + s_prologue);
                for (uint32_t idx = tid; idx < s_n_vec; idx += 512u) {
                    const uint4 v4 = row_v4[idx];
                    target_pass2_fine_pair(warp_hist, v4.x, thresh_hi);
                    target_pass2_fine_pair(warp_hist, v4.y, thresh_hi);
                    target_pass2_fine_pair(warp_hist, v4.z, thresh_hi);
                    target_pass2_fine_pair(warp_hist, v4.w, thresh_hi);
                }
            }
            for (uint32_t c = s_vec_end + tid; c < vocab; c += 512u) {
                const uint16_t raw = row_raw[c];
                if (bf16_is_finite_u16(raw)) {
                    const uint16_t ord = bf16_to_ordered_u16(raw);
                    if (((uint32_t)ord >> 8) == thresh_hi) {
                        atomicAdd(&warp_hist[(uint32_t)ord & 0xFFu], 1u);
                    }
                }
            }
            __syncthreads();

            if (tid < 256u) {
                unsigned int sum = 0u;
                #pragma unroll
                for (unsigned int w = 0u; w < 16u; ++w) {
                    sum += s_work.hist[w][tid];
                }
                s_work.hist[0][tid] = sum;
            }
            __syncthreads();

            if (tid == 0u) {
                rej_hist_subtract_masked_lo(s_work.hist[0], row_raw, vocab, thresh_hi, s_m0, s_m1);
                const uint32_t need = s_meta.need_in_hi;
                uint32_t above_lo = 0u;
                int lo = 255;
                for (; lo > 0; --lo) {
                    if (above_lo + s_work.hist[0][lo] >= need) {
                        break;
                    }
                    above_lo += s_work.hist[0][lo];
                }
                s_meta.thresh_key = (thresh_hi << 8) | (uint32_t)lo;
                s_meta.need_eq = need - above_lo;
                s_meta.count_eq = s_work.hist[0][lo];
            }
            __syncthreads();
        }

        topk_thresh_key = (uint16_t)s_meta.thresh_key;
        topk_need_eq = s_meta.need_eq;
        topk_need_tiebreak = (s_meta.count_eq > s_meta.need_eq);
    }

    if (tid == 0u) {
        s_meta.topk_max_id = 0xFFFFFFFFu;
        s_eq_cnt = 0u;
        s_p_max_id = 0xFFFFFFFFu;
        s_accepted = 0u;
    }
    __syncthreads();

    if (has_top_k && topk_need_tiebreak) {
        for (uint32_t base = 0u; base < vocab; base += 512u) {
            if (s_eq_cnt >= topk_need_eq) {
                break;
            }
            const uint32_t c = base + tid;
            bool hit_eq = false;
            if (c < vocab && !rej_is_masked(c, s_m0, s_m1)) {
                const uint16_t raw = row_raw[c];
                if (bf16_is_finite_u16(raw) && bf16_to_ordered_u16(raw) == topk_thresh_key) {
                    hit_eq = true;
                }
            }
            if (__syncthreads_or(hit_eq)) {
                const unsigned int wm = __ballot_sync(0xFFFFFFFFu, hit_eq);
                if (lane == 0u) {
                    s_warp_cnt[warp] = (uint32_t)__popc(wm);
                }
                __syncthreads();
                const uint32_t prev_eq = s_eq_cnt;
                if (hit_eq) {
                    uint32_t offset = 0u;
                    #pragma unroll
                    for (unsigned int w = 0u; w < 16u; ++w) {
                        if (w < warp) {
                            offset += s_warp_cnt[w];
                        }
                    }
                    const unsigned int lane_mask = (1u << lane) - 1u;
                    const uint32_t r = prev_eq + offset + (uint32_t)__popc(wm & lane_mask);
                    if (r + 1u == topk_need_eq) {
                        s_meta.topk_max_id = c;
                    }
                }
                __syncthreads();
                if (tid == 0u) {
                    uint32_t chunk_eq = 0u;
                    #pragma unroll
                    for (unsigned int w = 0u; w < 16u; ++w) {
                        chunk_eq += s_warp_cnt[w];
                    }
                    s_eq_cnt = prev_eq + chunk_eq;
                }
                __syncthreads();
            }
        }
    }
    // Exact row max over unmasked finite BF16 logits (`uint4` 1-pass).
    uint32_t local_max_ord = 0u;
    if (tid < s_prologue) {
        rej_max_finite_ord_update(row_raw[tid], tid, s_m0, s_m1, local_max_ord);
    }
    if (s_n_vec > 0u) {
        const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + s_prologue);
        for (uint32_t idx = tid; idx < s_n_vec; idx += 512u) {
            const uint4 v4 = row_v4[idx];
            const uint32_t c0 = s_prologue + (idx << 3);
            rej_max_finite_ord_update((uint16_t)(v4.x & 0xFFFFu), c0 + 0u, s_m0, s_m1, local_max_ord);
            rej_max_finite_ord_update((uint16_t)(v4.x >> 16),     c0 + 1u, s_m0, s_m1, local_max_ord);
            rej_max_finite_ord_update((uint16_t)(v4.y & 0xFFFFu), c0 + 2u, s_m0, s_m1, local_max_ord);
            rej_max_finite_ord_update((uint16_t)(v4.y >> 16),     c0 + 3u, s_m0, s_m1, local_max_ord);
            rej_max_finite_ord_update((uint16_t)(v4.z & 0xFFFFu), c0 + 4u, s_m0, s_m1, local_max_ord);
            rej_max_finite_ord_update((uint16_t)(v4.z >> 16),     c0 + 5u, s_m0, s_m1, local_max_ord);
            rej_max_finite_ord_update((uint16_t)(v4.w & 0xFFFFu), c0 + 6u, s_m0, s_m1, local_max_ord);
            rej_max_finite_ord_update((uint16_t)(v4.w >> 16),     c0 + 7u, s_m0, s_m1, local_max_ord);
        }
    }
    for (uint32_t c = s_vec_end + tid; c < vocab; c += 512u) {
        rej_max_finite_ord_update(row_raw[c], c, s_m0, s_m1, local_max_ord);
    }
    const uint32_t max_ord = block_reduce_max_u32_512(s_warp_primary.ord, warp, lane, local_max_ord);
    if (max_ord == 0u) {
        if (tid == 0u) {
            out_words[row] = 0u;
        }
        return;
    }
    if (tid == 0u) {
        const float min_p_c = min_ps[row];
        s_topk_tkey = (uint32_t)topk_thresh_key;
        s_max_sc = __fdiv_rn(
            bf16_u16_to_f32(ordered_u16_to_bf16((uint16_t)max_ord)),
            s_eff_temp
        );
        s_min_p_w = (min_p_c > 0.0f && isfinite(min_p_c)) ? fminf(min_p_c, 1.0f) : 0.0f;
    }

    // Pass A: probability-mass histogram over upper 8 bits of BF16 key (also computes `Z`).
    if (tid < 256u) {
        s_work.prob_hi[tid] = 0.0f;
    }
    __syncthreads();

    for (uint32_t c = tid; c < vocab; c += 512u) {
        rej_prob_hi_elem(
            row_raw[c], c, s_m0, s_m1, (uint16_t)s_topk_tkey, s_meta.topk_max_id,
            s_eff_temp, s_max_sc, s_min_p_w, s_work.prob_hi
        );
    }
    __syncthreads();

    if (tid == 0u) {
        const float top_p_c = top_ps[row];
        const bool has_top_p_c = (top_p_c > 0.0f && top_p_c < 1.0f);
        float total_sum = 0.0f;
        for (int b = 255; b >= 0; --b) {
            total_sum += s_work.prob_hi[b];
        }
        s_meta.total_prob_sum = total_sum;
        s_kept_sum = total_sum;
        if (has_top_p_c && total_sum > 0.0f && isfinite(total_sum)) {
            float cum = 0.0f;
            int hi = 255;
            int last_nz = 255;
            for (; hi >= 0; --hi) {
                const float bh = s_work.prob_hi[hi];
                if (bh > 0.0f) {
                    last_nz = hi;
                    if (__fdiv_rn(cum + bh, total_sum) >= top_p_c) {
                        break;
                    }
                    cum += bh;
                }
            }
            if (hi < 0) {
                hi = last_nz;
                cum = fmaxf(cum - s_work.prob_hi[hi], 0.0f);
            }
            s_meta.p_win_hi = (uint32_t)hi;
            s_meta.p_cum_above_hi = cum;
            s_meta.p_valid = 1u;
        } else {
            s_meta.p_valid = 0u;
            s_meta.thresh_key = s_topk_tkey;
            s_meta.need_eq = 0xFFFFFFFFu;
            s_meta.count_eq = 0u;
            s_p_max_id = s_meta.topk_max_id;
        }
    }
    __syncthreads();

    // Pass B: lower-8-bit count histogram inside `p_win_hi` (only if `has_top_p` active).
    if (s_meta.p_valid != 0u) {
        const uint32_t p_hi = s_meta.p_win_hi;
        #pragma unroll
        for (unsigned int i = tid; i < 4096u; i += 512u) {
            s_work.hist[i >> 8][i & 255u] = 0u;
        }
        __syncthreads();

        if (tid < s_prologue) {
            rej_prob_lo_elem(warp_hist, row_raw[tid], tid, s_m0, s_m1, (uint16_t)s_topk_tkey, s_meta.topk_max_id, p_hi);
        }
        if (s_n_vec > 0u) {
            const uint4* __restrict__ row_v4 = reinterpret_cast<const uint4*>(row_raw + s_prologue);
            for (uint32_t idx = tid; idx < s_n_vec; idx += 512u) {
                const uint4 v4 = row_v4[idx];
                const uint32_t c0 = s_prologue + (idx << 3);
                #pragma unroll 1
                for (uint32_t h = 0u; h < 8u; ++h) {
                    const uint32_t q = h >> 1;
                    const uint32_t w = (q == 0u) ? v4.x : ((q == 1u) ? v4.y : ((q == 2u) ? v4.z : v4.w));
                    const uint16_t raw = (uint16_t)((h & 1u) ? (w >> 16) : (w & 0xFFFFu));
                    rej_prob_lo_elem(warp_hist, raw, c0 + h, s_m0, s_m1, (uint16_t)s_topk_tkey, s_meta.topk_max_id, p_hi);
                }
            }
        }
        for (uint32_t c = s_vec_end + tid; c < vocab; c += 512u) {
            rej_prob_lo_elem(warp_hist, row_raw[c], c, s_m0, s_m1, (uint16_t)s_topk_tkey, s_meta.topk_max_id, p_hi);
        }
        __syncthreads();

        if (tid < 256u) {
            unsigned int lo_total = 0u;
            #pragma unroll
            for (unsigned int w = 0u; w < 16u; ++w) {
                lo_total += s_work.hist[w][tid];
            }
            s_work.hist[0][tid] = lo_total;
        }
        __syncthreads();

        if (tid == 0u) {
            const float top_p_c = top_ps[row];
            const float total_sum = s_meta.total_prob_sum;
            float cum = s_meta.p_cum_above_hi;
            int win_lo = 0;
            uint32_t need_eq_p = 1u;
            uint32_t count_eq_p = 1u;
            float kept_z = cum;
            bool found = false;
            for (int lo = 255; lo >= 0; --lo) {
                const uint32_t cnt = s_work.hist[0][lo];
                if (cnt == 0u) {
                    continue;
                }
                const uint16_t ord_bin = (uint16_t)((p_hi << 8) | (uint32_t)lo);
                const float v_bin = bf16_u16_to_f32(ordered_u16_to_bf16(ord_bin));
                const float w_bin = expf(__fdiv_rn(v_bin, s_eff_temp) - s_max_sc);
                if (w_bin < s_min_p_w || !(w_bin > 0.0f)) {
                    continue;
                }
                const float bin_mass = w_bin * (float)cnt;
                win_lo = lo;
                count_eq_p = cnt;
                if (__fdiv_rn(cum + bin_mass, total_sum) >= top_p_c) {
                    const float rem = fmaxf(top_p_c * total_sum - cum, 0.0f);
                    uint32_t n_take =
                        (w_bin > 0.0f) ? (uint32_t)ceilf(__fdiv_rn(rem, w_bin)) : cnt;
                    if (n_take < 1u) {
                        n_take = 1u;
                    }
                    if (n_take > cnt) {
                        n_take = cnt;
                    }
                    while (n_take > 1u
                        && __fdiv_rn(cum + w_bin * (float)(n_take - 1u), total_sum) >= top_p_c) {
                        --n_take;
                    }
                    while (n_take < cnt
                        && __fdiv_rn(cum + w_bin * (float)n_take, total_sum) < top_p_c) {
                        ++n_take;
                    }
                    need_eq_p = n_take;
                    kept_z = cum + w_bin * (float)n_take;
                    found = true;
                    break;
                }
                cum += bin_mass;
                kept_z = cum;
            }
            if (!found) {
                need_eq_p = count_eq_p;
            }
            s_meta.thresh_key = (p_hi << 8) | (uint32_t)win_lo;
            s_meta.need_eq = need_eq_p;
            s_meta.count_eq = count_eq_p;
            s_kept_sum = kept_z;
            s_eq_cnt = 0u;
            s_p_max_id = 0xFFFFFFFFu;
        }
        __syncthreads();

        // Resolve exact `p_max_id` when nucleus splits a tied BF16 bucket.
        if (s_meta.count_eq > s_meta.need_eq) {
            const uint16_t p_tkey = (uint16_t)s_meta.thresh_key;
            const uint32_t p_need = s_meta.need_eq;
            for (uint32_t base = 0u; base < vocab; base += 512u) {
                if (s_eq_cnt >= p_need) {
                    break;
                }
                const uint32_t c = base + tid;
                bool hit_eq = false;
                if (c < vocab && !rej_is_masked(c, s_m0, s_m1)) {
                    const uint16_t raw = row_raw[c];
                    if (bf16_is_finite_u16(raw)) {
                        const uint16_t k16 = bf16_to_ordered_u16(raw);
                        if (k16 == p_tkey
                            && (k16 > (uint16_t)s_topk_tkey || (k16 == (uint16_t)s_topk_tkey && c <= s_meta.topk_max_id))) {
                            hit_eq = true;
                        }
                    }
                }
                if (__syncthreads_or(hit_eq)) {
                    const unsigned int wm = __ballot_sync(0xFFFFFFFFu, hit_eq);
                    if (lane == 0u) {
                        s_warp_cnt[warp] = (uint32_t)__popc(wm);
                    }
                    __syncthreads();
                    const uint32_t prev_eq = s_eq_cnt;
                    if (hit_eq) {
                        uint32_t offset = 0u;
                        #pragma unroll
                        for (unsigned int w = 0u; w < 16u; ++w) {
                            if (w < warp) {
                                offset += s_warp_cnt[w];
                            }
                        }
                        const unsigned int lane_mask = (1u << lane) - 1u;
                        const uint32_t r = prev_eq + offset + (uint32_t)__popc(wm & lane_mask);
                        if (r + 1u == p_need) {
                            s_p_max_id = c;
                        }
                    }
                    __syncthreads();
                    if (tid == 0u) {
                        uint32_t chunk_eq = 0u;
                        #pragma unroll
                        for (unsigned int w = 0u; w < 16u; ++w) {
                            chunk_eq += s_warp_cnt[w];
                        }
                        s_eq_cnt = prev_eq + chunk_eq;
                    }
                    __syncthreads();
                }
            }
        }
    }

    // O(1) division-free accept test on `draft_token` before streaming residual scan.
    if (tid == 0u) {
        const uint16_t tk_key = (uint16_t)s_topk_tkey;
        const uint32_t tk_max = s_meta.topk_max_id;
        const uint16_t p_key = (uint16_t)s_meta.thresh_key;
        const uint32_t p_max = s_p_max_id;
        const float ksum = s_kept_sum;
        if (s_is_draft != 0u) {
            const uint32_t draft_tok = draft_tokens[row];
            const float q_x = rej_lookup_q(draft_tok, s_q_ids, s_q_probs, s_q_len);
            s_q_x = q_x;
            float p_x = 0.0f;
            if (draft_tok < vocab && !rej_is_masked(draft_tok, s_m0, s_m1)
                && ksum > 0.0f && isfinite(ksum)) {
                const uint16_t raw_d = row_raw[draft_tok];
                if (bf16_is_finite_u16(raw_d)) {
                    const uint16_t k16_d = bf16_to_ordered_u16(raw_d);
                    const bool in_topk = (k16_d > tk_key)
                        || (k16_d == tk_key && draft_tok <= tk_max);
                    const bool in_topp = (k16_d > p_key)
                        || (k16_d == p_key && draft_tok <= p_max);
                    if (in_topk && in_topp) {
                        const float wd = expf(
                            __fdiv_rn(bf16_u16_to_f32(raw_d), s_eff_temp) - s_max_sc
                        );
                        if (wd >= s_min_p_w && wd > 0.0f) {
                            p_x = __fdiv_rn(wd, ksum);
                        }
                    }
                }
            }
            if (q_x > 0.0f && p_x > 0.0f && (p_x >= q_x || s_u_acc * q_x < p_x)) {
                s_accepted = 1u;
                out_words[row] = (1u << 31) | (draft_tok & 0x7FFFFFFFu);
            }
        } else {
            s_q_x = 0.0f;
        }
    }
    __syncthreads();

    if (s_accepted != 0u) {
        return;
    }

    // Streaming Gumbel-Max over exact residual `w_res = max(0, p - q)` (with pure `p` fallback).
    const bool use_residual =
        (s_is_draft != 0u) && (s_q_x > 0.0f) && (s_kept_sum > 0.0f) && isfinite(s_kept_sum);
    if (use_residual) {
        float best_score = -CUDART_INF_F;
        uint32_t best_id = 0xFFFFFFFFu;
        for (uint32_t c = tid; c < vocab; c += 512u) {
            rej_stream_score_res_elem(
                row_raw[c], c, s_m0, s_m1, (uint16_t)s_topk_tkey, s_meta.topk_max_id,
                (uint16_t)s_meta.thresh_key, s_p_max_id, s_eff_temp, s_max_sc, s_min_p_w,
                s_kept_sum, s_res_base, s_q_ids, s_q_probs, s_q_len, best_score, best_id
            );
        }
        block_reduce_argmax_score_id_512(
            s_warp_primary.val, s_warp_id, warp, lane, best_score, best_id
        );
        if (tid == 0u) {
            s_kept = best_id;
            if (best_id != 0xFFFFFFFFu) {
                out_words[row] = best_id & 0x7FFFFFFFu;
            }
        }
        __syncthreads();
        if (s_kept != 0xFFFFFFFFu) {
            return;
        }
    }

    float best_p_score = -CUDART_INF_F;
    uint32_t best_p_id = 0xFFFFFFFFu;
    for (uint32_t c = tid; c < vocab; c += 512u) {
        rej_stream_score_p_elem(
            row_raw[c], c, s_m0, s_m1, (uint16_t)s_topk_tkey, s_meta.topk_max_id,
            (uint16_t)s_meta.thresh_key, s_p_max_id, s_eff_temp, s_max_sc, s_min_p_w,
            s_res_base, best_p_score, best_p_id
        );
    }
    block_reduce_argmax_score_id_512(
        s_warp_primary.val, s_warp_id, warp, lane, best_p_score, best_p_id
    );
    if (tid == 0u) {
        const uint32_t chosen = (best_p_id != 0xFFFFFFFFu) ? best_p_id : 0u;
        out_words[row] = chosen & 0x7FFFFFFFu;
    }
}

// Certified F2 statistics for one edited BF16 row (SPEED_PLAN W1a2).
// CPU specification:
// spark_runtime::sampler::f2_certify::edited_sum_exp_reference.
//
// Applies `n_edits` host edits in order (kind 0: x -= val, kind 1: x = -inf),
// then writes { f64 sum, f64 weighted, f64 term_err, f32 max, u32 nonfinite }:
//   sum      = sum_i exp(x_i - max) as per-lane f32 sums (lane = i % 1024,
//              increasing i) merged in f64;
//   weighted = sum_i t_i (n - i) (0-based i: term i enters n - i partial sums);  term_err = sum_i t_i (2|x_i - max| + 6);
//   nonfinite = the edited row holds a NaN or +inf.
// The host certifies its own sequential-sum decision from these with a
// rigorous error bound; the weights are bounds, so per-lane f32 accumulation
// is covered by the bound's slack.
// Grid: (1, 1, 1)  Block: (1024, 1, 1)
#define F2_EDITS_MAX 64

__device__ __forceinline__ float f2_edited(
    const __nv_bfloat16* __restrict__ row, unsigned int i,
    const unsigned int* s_ids, const unsigned int* s_kind, const float* s_val,
    unsigned int n_edits
) {
    float v = __bfloat162float(row[i]);
    for (unsigned int e = 0; e < n_edits; e++) {
        if (s_ids[e] == i) {
            v = s_kind[e] == 0u ? __fsub_rn(v, s_val[e]) : -CUDART_INF_F;
        }
    }
    return v;
}

extern "C" __global__ void __launch_bounds__(1024) edited_sum_exp_bf16(
    const __nv_bfloat16* __restrict__ row,
    const unsigned int* __restrict__ edit_ids,
    const unsigned int* __restrict__ edit_kind,
    const float* __restrict__ edit_val,
    unsigned int n_edits,
    unsigned int n,
    unsigned char* __restrict__ out
) {
    __shared__ unsigned int s_ids[F2_EDITS_MAX];
    __shared__ unsigned int s_kind[F2_EDITS_MAX];
    __shared__ float s_val[F2_EDITS_MAX];
    __shared__ float s_max[1024];
    __shared__ double s_sum[1024];
    __shared__ double s_w[1024];
    __shared__ double s_e[1024];
    __shared__ unsigned int s_nonfinite;

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    if (n_edits > F2_EDITS_MAX) {
        n_edits = F2_EDITS_MAX;  // the host refuses more before launch
    }
    for (unsigned int e = tid; e < n_edits; e += stride) {
        s_ids[e] = edit_ids[e];
        s_kind[e] = edit_kind[e];
        s_val[e] = edit_val[e];
    }
    if (tid == 0) {
        s_nonfinite = 0u;
    }
    __syncthreads();

    float local_max = -CUDART_INF_F;
    unsigned int local_bad = 0u;
    for (unsigned int i = tid; i < n; i += stride) {
        const float v = f2_edited(row, i, s_ids, s_kind, s_val, n_edits);
        if (isnan(v) || v == CUDART_INF_F) {
            local_bad = 1u;
        }
        local_max = fmaxf(local_max, v);
    }
    s_max[tid] = local_max;
    if (local_bad) {
        atomicOr(&s_nonfinite, 1u);
    }
    __syncthreads();
    for (unsigned int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    const float mx = s_max[0];

    float lane = 0.0f, w = 0.0f, te = 0.0f;
    for (unsigned int i = tid; i < n; i += stride) {
        const float a = __fsub_rn(f2_edited(row, i, s_ids, s_kind, s_val, n_edits), mx);
        const float t = expf(a);
        lane = __fadd_rn(lane, t);
        if (t > 0.0f) {
            w += t * (float)(n - i);  // term i is in n - i host partial sums
            te += t * (2.0f * fabsf(a) + 6.0f);
        }
    }
    s_sum[tid] = (double)lane;
    s_w[tid] = (double)w;
    s_e[tid] = (double)te;
    __syncthreads();
    for (unsigned int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_w[tid] += s_w[tid + s];
            s_e[tid] += s_e[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        double* o = (double*)out;
        o[0] = s_sum[0];
        o[1] = s_w[0] * 1.001;  // per-lane f32 weight accumulation slack
        o[2] = s_e[0] * 1.001;
        *(float*)(out + 24) = mx;
        *(unsigned int*)(out + 28) = s_nonfinite;
    }
}
