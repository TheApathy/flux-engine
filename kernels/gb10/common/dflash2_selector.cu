// SPDX-License-Identifier: AGPL-3.0-only
//
// DFlash2 `CandidateSelector` greedy walk, ported from
// Avarok-Cybersecurity/atlas PR #648 (`dflash2_selector_walk`).
//
// Upstream launches 512 threads (16 warps, one warp per candidate) and takes
// the predecessor implicitly from `tokens[0]`. This tree's Rust op predates
// that and passes `anchor_id` explicitly with a 16-thread block — one thread
// per candidate — which `examples/dflash2_conv_selector_microtest.rs` already
// checks against a CPU reference. Ported to that ABI so the microtest gates it.
//
// Reference: `CandidateSelector.select` (z-lab/dflash `dflash/model.py`),
// greedy (T=0) branch. Per position:
//   scores[k] = unary[k] + Σ_r (pred[prev][r] * hidden[t][r]) * succ[cand_k][r]
//   index     = argmax_k scores
//   prev      = candidates[index]
// The walk is inherently SEQUENTIAL across positions: position t's predecessor
// is position t-1's choice, which is why this is one block, not a grid.

#include <cuda_bf16.h>

#define DF2_TOPK 16

extern "C" __global__ void dflash2_selector_walk(
    const float* __restrict__ unary,            // [gamma, 16]
    const unsigned int* __restrict__ candidates,// [gamma, 16]
    const __nv_bfloat16* __restrict__ hidden,   // [gamma, rank]
    const __nv_bfloat16* __restrict__ pred,     // [vocab, rank]
    const __nv_bfloat16* __restrict__ succ,     // [vocab, rank]
    unsigned int* __restrict__ path,            // [gamma] out
    unsigned int anchor_id,
    int gamma,
    int rank
) {
    __shared__ float s_score[DF2_TOPK];
    __shared__ unsigned int s_prev;

    const unsigned int k = threadIdx.x;   // one thread per candidate

    if (k == 0) s_prev = anchor_id;
    __syncthreads();

    for (int t = 0; t < gamma; ++t) {
        const __nv_bfloat16* pr = pred + (size_t)s_prev * (size_t)rank;
        const __nv_bfloat16* hr = hidden + (size_t)t * (size_t)rank;

        float acc = 0.0f;
        if (k < DF2_TOPK) {
            const unsigned int cand = candidates[(size_t)t * DF2_TOPK + k];
            const __nv_bfloat16* sr = succ + (size_t)cand * (size_t)rank;
            // gate[r] = pred[prev][r] * hidden[t][r]; score contribution is
            // gate[r] * succ[cand][r]. Recomputed per thread rather than
            // staged in shared: rank is 256, and a shared gate would need a
            // second barrier inside the sequential loop for no bandwidth win.
            for (int r = 0; r < rank; ++r) {
                acc += __bfloat162float(pr[r]) * __bfloat162float(hr[r])
                     * __bfloat162float(sr[r]);
            }
            s_score[k] = unary[(size_t)t * DF2_TOPK + k] + acc;
        }
        __syncthreads();

        if (k == 0) {
            float best = s_score[0];
            int best_k = 0;
            #pragma unroll
            for (int j = 1; j < DF2_TOPK; ++j) {
                if (s_score[j] > best) { best = s_score[j]; best_k = j; }
            }
            const unsigned int chosen =
                candidates[(size_t)t * DF2_TOPK + best_k];
            path[t] = chosen;
            s_prev = chosen;
        }
        __syncthreads();
    }
}

// Sampled DFlash2 selector. Kept as a separate symbol so the released greedy
// ABI and instruction path above remain literally unchanged.
//
// `uniforms[t]` is supplied by the host receipt path and must be in [0,1).
// Candidate membership remains the raw drafter top-16. The kernel applies
// pair adjustment, ordered bias/masks, then request top-k -> softmax -> min-p
// -> top-p. It exports those exact post-policy q rows; the host validates
// every row before publication.
// Grid: [1, 1, 1], Block: [16, 1, 1].
extern "C" __global__ void dflash2_selector_walk_probabilistic(
    const float* __restrict__ unary,             // [gamma, 16]
    const unsigned int* __restrict__ candidates, // [gamma, 16]
    const __nv_bfloat16* __restrict__ hidden,    // [gamma, rank]
    const __nv_bfloat16* __restrict__ pred,      // [vocab, rank]
    const __nv_bfloat16* __restrict__ succ,      // [vocab, rank]
    unsigned int* __restrict__ path,             // [gamma] out
    const float* __restrict__ uniforms,           // [gamma]
    const unsigned int* __restrict__ q_modifier_token_ids, // [modifier_count]
    const float* __restrict__ q_modifier_values,  // finite bias or -inf mask
    int q_modifier_count,
    float temperature,
    unsigned int top_k,
    float min_p,
    float top_p,
    float* __restrict__ q_probabilities,          // [gamma, 16] out
    unsigned int* __restrict__ q_candidates,      // [gamma, 16] out
    unsigned int anchor_id,
    int gamma,
    int rank
) {
    __shared__ float s_score[DF2_TOPK];
    __shared__ float s_probability[DF2_TOPK];
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ int s_order[DF2_TOPK];
    __shared__ int s_order_count;
    __shared__ unsigned int s_keep_mask;
    __shared__ unsigned int s_prev;

    const unsigned int k = threadIdx.x;
    if (k == 0) s_prev = anchor_id;
    __syncthreads();

    for (int t = 0; t < gamma; ++t) {
        if (k < DF2_TOPK) {
            const __nv_bfloat16* pr = pred + (size_t)s_prev * (size_t)rank;
            const __nv_bfloat16* hr = hidden + (size_t)t * (size_t)rank;
            const unsigned int cand = candidates[(size_t)t * DF2_TOPK + k];
            const __nv_bfloat16* sr = succ + (size_t)cand * (size_t)rank;
            float acc = 0.0f;
            for (int r = 0; r < rank; ++r) {
                acc += __bfloat162float(pr[r]) * __bfloat162float(hr[r])
                     * __bfloat162float(sr[r]);
            }
            float score = unary[(size_t)t * DF2_TOPK + k] + acc;
            bool hard_masked = false;
            // Preserve request order: duplicate logit_bias entries use the same
            // sequential f32 addition semantics as the target sampler. Internal
            // negative-infinity entries are hard masks and dominate every bias.
            for (int modifier = 0; modifier < q_modifier_count; ++modifier) {
                if (q_modifier_token_ids[modifier] == cand) {
                    const float value = q_modifier_values[modifier];
                    if (isinf(value) && value < 0.0f) {
                        hard_masked = true;
                    } else {
                        score += value;
                    }
                }
            }
            s_score[k] = hard_masked ? -INFINITY : score;
            q_candidates[(size_t)t * DF2_TOPK + k] = cand;
        }
        __syncthreads();

        // Establish one deterministic score order for top-k and top-p:
        // score descending, then token id ascending. Candidate output remains
        // in its original raw-top16 order; zero q records every filtered slot.
        if (k == 0) {
            unsigned int ordered_mask = 0u;
            int count = 0;
            #pragma unroll
            for (int rank_index = 0; rank_index < DF2_TOPK; ++rank_index) {
                int best = -1;
                #pragma unroll
                for (int j = 0; j < DF2_TOPK; ++j) {
                    if ((ordered_mask & (1u << j)) != 0u || !isfinite(s_score[j])) continue;
                    if (best < 0 || s_score[j] > s_score[best] ||
                        (s_score[j] == s_score[best] &&
                         candidates[(size_t)t * DF2_TOPK + j] <
                             candidates[(size_t)t * DF2_TOPK + best])) {
                        best = j;
                    }
                }
                if (best < 0) break;
                s_order[count++] = best;
                ordered_mask |= 1u << best;
            }
            s_order_count = count;
            if (count == 0) {
                s_max = nanf("");
                s_keep_mask = 0u;
            } else {
                s_max = s_score[s_order[0]];
                const unsigned int limit =
                    top_k == 0u ? (unsigned int)count : min(top_k, (unsigned int)count);
                unsigned int keep_mask = 0u;
                for (unsigned int rank_index = 0; rank_index < limit; ++rank_index) {
                    keep_mask |= 1u << s_order[rank_index];
                }
                s_keep_mask = keep_mask;
            }
        }
        __syncthreads();

        if (k < DF2_TOPK) {
            s_probability[k] = (s_keep_mask & (1u << k)) != 0u
                ? expf((s_score[k] - s_max) / temperature)
                : 0.0f;
        }
        __syncthreads();

        if (k == 0 && s_order_count > 0) {
            unsigned int keep_mask = s_keep_mask;
            const float min_weight = min_p * s_probability[s_order[0]];
            #pragma unroll
            for (int rank_index = 0; rank_index < DF2_TOPK; ++rank_index) {
                if (rank_index >= s_order_count) break;
                const int index = s_order[rank_index];
                if ((keep_mask & (1u << index)) != 0u &&
                    s_probability[index] < min_weight) {
                    keep_mask &= ~(1u << index);
                }
            }

            float pre_top_p_sum = 0.0f;
            #pragma unroll
            for (int rank_index = 0; rank_index < DF2_TOPK; ++rank_index) {
                if (rank_index >= s_order_count) break;
                const int index = s_order[rank_index];
                if ((keep_mask & (1u << index)) != 0u) {
                    pre_top_p_sum += s_probability[index];
                }
            }
            if (top_p < 1.0f) {
                float cumulative = 0.0f;
                unsigned int nucleus_mask = 0u;
                #pragma unroll
                for (int rank_index = 0; rank_index < DF2_TOPK; ++rank_index) {
                    if (rank_index >= s_order_count) break;
                    const int index = s_order[rank_index];
                    if ((keep_mask & (1u << index)) == 0u) continue;
                    nucleus_mask |= 1u << index;
                    cumulative += s_probability[index] / pre_top_p_sum;
                    if (cumulative >= top_p) break;
                }
                keep_mask = nucleus_mask;
            }
            s_keep_mask = keep_mask;

            float final_sum = 0.0f;
            #pragma unroll
            for (int rank_index = 0; rank_index < DF2_TOPK; ++rank_index) {
                if (rank_index >= s_order_count) break;
                const int index = s_order[rank_index];
                if ((keep_mask & (1u << index)) != 0u) {
                    final_sum += s_probability[index];
                }
            }
            s_sum = final_sum;
        } else if (k == 0) {
            s_sum = nanf("");
        }
        __syncthreads();

        if (k < DF2_TOPK) {
            s_probability[k] = (s_keep_mask & (1u << k)) != 0u
                ? s_probability[k] / s_sum
                : (s_order_count > 0 ? 0.0f : nanf(""));
            q_probabilities[(size_t)t * DF2_TOPK + k] = s_probability[k];
        }
        __syncthreads();

        if (k == 0) {
            const float uniform = uniforms[t];
            float cumulative = 0.0f;
            int selected = 0;
            int last_nonzero = 0;
            #pragma unroll
            for (int j = 0; j < DF2_TOPK; ++j) {
                if (s_probability[j] > 0.0f) last_nonzero = j;
                cumulative += s_probability[j];
                if (cumulative > uniform) {
                    selected = j;
                    break;
                }
                selected = last_nonzero;
            }
            const unsigned int chosen =
                candidates[(size_t)t * DF2_TOPK + selected];
            path[t] = chosen;
            s_prev = chosen;
        }
        __syncthreads();
    }
}

// Batched DFlash2 greedy selector walk: one 16-thread CTA per sequence in
// `0..batch_size`, preserving the exact instruction path of
// `dflash2_selector_walk`.
// Grid: [batch_size, 1, 1], Block: [16, 1, 1].
extern "C" __global__ void dflash2_selector_walk_batched(
    const float* __restrict__ unary,             // [batch_size * gamma, 16]
    const unsigned int* __restrict__ candidates, // [batch_size * gamma, 16]
    const __nv_bfloat16* __restrict__ hidden,    // [batch_size * gamma, rank]
    const __nv_bfloat16* __restrict__ pred,      // [vocab, rank]
    const __nv_bfloat16* __restrict__ succ,      // [vocab, rank]
    unsigned int* __restrict__ path,             // [batch_size * gamma] out
    const unsigned int* __restrict__ anchor_ids, // [batch_size]
    int gamma,
    int rank,
    int batch_size
) {
    const int b = (int)blockIdx.x;
    if (b >= batch_size) return;

    unary      += (size_t)b * (size_t)gamma * DF2_TOPK;
    candidates += (size_t)b * (size_t)gamma * DF2_TOPK;
    hidden     += (size_t)b * (size_t)gamma * (size_t)rank;
    path       += (size_t)b * (size_t)gamma;
    const unsigned int anchor_id = anchor_ids[b];

    __shared__ float s_score[DF2_TOPK];
    __shared__ unsigned int s_prev;

    const unsigned int k = threadIdx.x;   // one thread per candidate

    if (k == 0) s_prev = anchor_id;
    __syncthreads();

    for (int t = 0; t < gamma; ++t) {
        const __nv_bfloat16* pr = pred + (size_t)s_prev * (size_t)rank;
        const __nv_bfloat16* hr = hidden + (size_t)t * (size_t)rank;

        float acc = 0.0f;
        if (k < DF2_TOPK) {
            const unsigned int cand = candidates[(size_t)t * DF2_TOPK + k];
            const __nv_bfloat16* sr = succ + (size_t)cand * (size_t)rank;
            for (int r = 0; r < rank; ++r) {
                acc += __bfloat162float(pr[r]) * __bfloat162float(hr[r])
                     * __bfloat162float(sr[r]);
            }
            s_score[k] = unary[(size_t)t * DF2_TOPK + k] + acc;
        }
        __syncthreads();

        if (k == 0) {
            float best = s_score[0];
            int best_k = 0;
            #pragma unroll
            for (int j = 1; j < DF2_TOPK; ++j) {
                if (s_score[j] > best) { best = s_score[j]; best_k = j; }
            }
            const unsigned int chosen =
                candidates[(size_t)t * DF2_TOPK + best_k];
            path[t] = chosen;
            s_prev = chosen;
        }
        __syncthreads();
    }
}

// ── CRN Noise-Coupled Gumbel-Max Batched DFlash2 Selector Walk ───────────────
//
// Matches `kernels/gb10/common/sampler_topk_candidates.cu` SplitMix64 + Gumbel
// helpers bit-for-bit so DFlash2 draft selection shares Common Random Numbers
// with the target verifier at `(seed, abs_pos, token_id)`.

__device__ __forceinline__ unsigned long long df2_splitmix64_mix(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ unsigned long long df2_gumbel_pos_base(
    unsigned long long seed, unsigned long long abs_pos
) {
    return df2_splitmix64_mix(
        df2_splitmix64_mix(seed + 0x9E3779B97F4A7C15ull) ^ (abs_pos * 0xD1B54A32D192ED03ull)
    );
}

__device__ __forceinline__ float df2_gumbel_from_base(
    unsigned long long pos_base, unsigned int token_id
) {
    const unsigned long long bits = df2_splitmix64_mix(pos_base ^ (unsigned long long)token_id);
    const float u = fminf((float)(bits >> 40) * 0x1.0p-24f + 0x1.0p-25f, 0x1.fffffep-1f);
    return -logf(-logf(u));
}

// Batched CRN noise-coupled Gumbel-Max DFlash2 selector walk.
// Grid: [batch_size, 1, 1], Block: [16, 1, 1].
// When `temperatures[b] <= 0.0f` or non-finite, executes the exact greedy
// argmax path of `dflash2_selector_walk_batched`.
extern "C" __global__ void dflash2_selector_walk_gumbel_batched(
    const float* __restrict__ unary,                     // [batch_size * gamma, 16]
    const unsigned int* __restrict__ candidates,         // [batch_size * gamma, 16]
    const __nv_bfloat16* __restrict__ hidden,            // [batch_size * gamma, rank]
    const __nv_bfloat16* __restrict__ pred,              // [vocab, rank]
    const __nv_bfloat16* __restrict__ succ,              // [vocab, rank]
    unsigned int* __restrict__ path,                     // [batch_size * gamma] out
    const unsigned int* __restrict__ anchor_ids,         // [batch_size]
    const unsigned long long* __restrict__ seeds,        // [batch_size]
    const unsigned long long* __restrict__ step0_abs_pos,// [batch_size]
    const float* __restrict__ temperatures,              // [batch_size] (<= 0.0 -> exact greedy argmax)
    const unsigned int* __restrict__ top_ks,             // [batch_size] (0 -> all 16)
    const float* __restrict__ top_ps,                    // [batch_size] (<= 0 or >= 1 -> disabled)
    const float* __restrict__ min_ps,                    // [batch_size] (<= 0 -> disabled)
    int gamma,
    int rank,
    int batch_size
) {
    const int b = (int)blockIdx.x;
    if (b >= batch_size) return;

    unary      += (size_t)b * (size_t)gamma * DF2_TOPK;
    candidates += (size_t)b * (size_t)gamma * DF2_TOPK;
    hidden     += (size_t)b * (size_t)gamma * (size_t)rank;
    path       += (size_t)b * (size_t)gamma;

    const unsigned int anchor_id = anchor_ids[b];
    const float temp = temperatures[b];
    const unsigned long long seed = seeds[b];
    const unsigned long long base_pos = step0_abs_pos[b];
    const unsigned int top_k = top_ks[b];
    const float top_p = top_ps[b];
    const float min_p = min_ps[b];

    const bool use_gumbel = (temp > 0.0f) && isfinite(temp);
    const float eff_temp = use_gumbel ? fmaxf(temp, 1.0e-6f) : 1.0f;

    __shared__ float s_score[DF2_TOPK];
    __shared__ unsigned int s_cand[DF2_TOPK];
    __shared__ float s_scaled[DF2_TOPK];
    __shared__ int s_ord[DF2_TOPK];
    __shared__ float s_w[DF2_TOPK];
    __shared__ unsigned int s_prev;

    const unsigned int k = threadIdx.x;   // one thread per candidate

    if (k == 0) s_prev = anchor_id;
    __syncthreads();

    for (int t = 0; t < gamma; ++t) {
        const __nv_bfloat16* pr = pred + (size_t)s_prev * (size_t)rank;
        const __nv_bfloat16* hr = hidden + (size_t)t * (size_t)rank;

        float acc = 0.0f;
        if (k < DF2_TOPK) {
            const unsigned int cand = candidates[(size_t)t * DF2_TOPK + k];
            const __nv_bfloat16* sr = succ + (size_t)cand * (size_t)rank;
            for (int r = 0; r < rank; ++r) {
                acc += __bfloat162float(pr[r]) * __bfloat162float(hr[r])
                     * __bfloat162float(sr[r]);
            }
            s_score[k] = unary[(size_t)t * DF2_TOPK + k] + acc;
            s_cand[k] = cand;
        }
        __syncthreads();

        if (k == 0) {
            if (!use_gumbel) {
                float best = s_score[0];
                int best_k = 0;
                #pragma unroll
                for (int j = 1; j < DF2_TOPK; ++j) {
                    if (s_score[j] > best) { best = s_score[j]; best_k = j; }
                }
                const unsigned int chosen = s_cand[best_k];
                path[t] = chosen;
                s_prev = chosen;
            } else {
                for (int j = 0; j < DF2_TOPK; ++j) {
                    const float raw_sc = __fdiv_rn(s_score[j], eff_temp);
                    s_scaled[j] = isfinite(raw_sc) ? raw_sc : -1.0e30f;
                    s_ord[j] = j;
                }

                // 16-element insertion sort descending by (s_scaled[idx], ~s_cand[idx]).
                for (int i = 1; i < DF2_TOPK; ++i) {
                    const int idx = s_ord[i];
                    const float sc = s_scaled[idx];
                    const unsigned int cid = s_cand[idx];
                    int j = i - 1;
                    while (j >= 0) {
                        const int prev_idx = s_ord[j];
                        const float prev_sc = s_scaled[prev_idx];
                        const unsigned int prev_cid = s_cand[prev_idx];
                        if (sc > prev_sc || (sc == prev_sc && cid < prev_cid)) {
                            s_ord[j + 1] = prev_idx;
                            --j;
                        } else {
                            break;
                        }
                    }
                    s_ord[j + 1] = idx;
                }

                int k_eff = (top_k > 0u && top_k < (unsigned int)DF2_TOPK)
                    ? (int)top_k
                    : DF2_TOPK;

                const float max_s = s_scaled[s_ord[0]];
                float sum_w = 0.0f;
                for (int j = 0; j < k_eff; ++j) {
                    const float w = expf(s_scaled[s_ord[j]] - max_s);
                    s_w[j] = w;
                    sum_w += w;
                }

                if (min_p > 0.0f && isfinite(min_p)) {
                    const float threshold = min_p * s_w[0];
                    while (k_eff > 1 && s_w[k_eff - 1] < threshold) {
                        --k_eff;
                    }
                }

                int kept = k_eff;
                if (top_p > 0.0f && top_p < 1.0f && sum_w > 0.0f) {
                    float rem_sum = 0.0f;
                    for (int j = 0; j < k_eff; ++j) {
                        rem_sum += s_w[j];
                    }
                    if (rem_sum > 0.0f) {
                        const float inv_sum = __fdiv_rn(1.0f, rem_sum);
                        float cum = 0.0f;
                        for (int j = 0; j < k_eff; ++j) {
                            cum += s_w[j] * inv_sum;
                            if (cum >= top_p) {
                                kept = j + 1;
                                break;
                            }
                        }
                    }
                }

                const unsigned long long pos_base =
                    df2_gumbel_pos_base(seed, base_pos + (unsigned long long)t);
                unsigned int best_cid = s_cand[s_ord[0]];
                float best_g = -INFINITY;
                for (int j = 0; j < kept; ++j) {
                    const int idx = s_ord[j];
                    const unsigned int cid = s_cand[idx];
                    const float g_score = s_scaled[idx] + df2_gumbel_from_base(pos_base, cid);
                    if (j == 0 || g_score > best_g || (g_score == best_g && cid < best_cid)) {
                        best_g = g_score;
                        best_cid = cid;
                    }
                }
                path[t] = best_cid;
                s_prev = best_cid;
            }
        }
        __syncthreads();
    }
}

// ── Widened k <= 32 CRN Gumbel-Max Batched Selector Walk + Confidence Probs ──
//
// Generalized `dflash2_selector_walk_gumbel_batched` supporting arbitrary
// `1 <= top_k_shortlist <= 32` (`DF2_TOPK_MAX`) on a 32-thread CTA (1 warp),
// and exporting per-position post-policy selection confidence `out_probs[t]`.
// Grid: [batch_size, 1, 1], Block: [32, 1, 1].
// When `temperatures[b] <= 0.0f` or non-finite, executes the exact greedy
// argmax path of `dflash2_selector_walk_batched` plus shortlist softmax prob.

#define DF2_TOPK_MAX 32

extern "C" __global__ void dflash2_selector_walk_gumbel_batched_k32(
    const float* __restrict__ unary,                     // [batch_size * gamma, top_k_shortlist]
    const unsigned int* __restrict__ candidates,         // [batch_size * gamma, top_k_shortlist]
    const __nv_bfloat16* __restrict__ hidden,            // [batch_size * gamma, rank]
    const __nv_bfloat16* __restrict__ pred,              // [vocab, rank]
    const __nv_bfloat16* __restrict__ succ,              // [vocab, rank]
    unsigned int* __restrict__ path,                     // [batch_size * gamma] out
    float* __restrict__ out_probs,                       // [batch_size * gamma] out
    const unsigned int* __restrict__ anchor_ids,         // [batch_size]
    const unsigned long long* __restrict__ seeds,        // [batch_size]
    const unsigned long long* __restrict__ step0_abs_pos,// [batch_size]
    const float* __restrict__ temperatures,              // [batch_size] (<= 0.0 -> greedy + softmax prob)
    const unsigned int* __restrict__ top_ks,             // [batch_size] (0 -> all top_k_shortlist)
    const float* __restrict__ top_ps,                    // [batch_size] (<= 0 or >= 1 -> disabled)
    const float* __restrict__ min_ps,                    // [batch_size] (<= 0 -> disabled)
    int top_k_shortlist,                                 // 1..=32 (16 or 32)
    int gamma,
    int rank,
    int batch_size
) {
    const int b = (int)blockIdx.x;
    if (b >= batch_size || top_k_shortlist <= 0 || top_k_shortlist > DF2_TOPK_MAX) return;

    const size_t stride_k = (size_t)top_k_shortlist;
    unary      += (size_t)b * (size_t)gamma * stride_k;
    candidates += (size_t)b * (size_t)gamma * stride_k;
    hidden     += (size_t)b * (size_t)gamma * (size_t)rank;
    path       += (size_t)b * (size_t)gamma;
    out_probs  += (size_t)b * (size_t)gamma;

    const unsigned int anchor_id = anchor_ids[b];
    const float temp = temperatures[b];
    const unsigned long long seed = seeds[b];
    const unsigned long long base_pos = step0_abs_pos[b];
    const unsigned int top_k = top_ks[b];
    const float top_p = top_ps[b];
    const float min_p = min_ps[b];

    const bool use_gumbel = (temp > 0.0f) && isfinite(temp);
    const float eff_temp = use_gumbel ? fmaxf(temp, 1.0e-6f) : 1.0f;

    __shared__ float s_score[DF2_TOPK_MAX];
    __shared__ unsigned int s_cand[DF2_TOPK_MAX];
    __shared__ float s_scaled[DF2_TOPK_MAX];
    __shared__ int s_ord[DF2_TOPK_MAX];
    __shared__ float s_w[DF2_TOPK_MAX];
    __shared__ unsigned int s_prev;

    const unsigned int k = threadIdx.x;   // one thread per candidate

    if (k == 0) s_prev = anchor_id;
    __syncthreads();

    for (int t = 0; t < gamma; ++t) {
        const __nv_bfloat16* pr = pred + (size_t)s_prev * (size_t)rank;
        const __nv_bfloat16* hr = hidden + (size_t)t * (size_t)rank;

        float acc = 0.0f;
        if ((int)k < top_k_shortlist) {
            const unsigned int cand = candidates[(size_t)t * stride_k + k];
            const __nv_bfloat16* sr = succ + (size_t)cand * (size_t)rank;
            for (int r = 0; r < rank; ++r) {
                acc += __bfloat162float(pr[r]) * __bfloat162float(hr[r])
                     * __bfloat162float(sr[r]);
            }
            s_score[k] = unary[(size_t)t * stride_k + k] + acc;
            s_cand[k] = cand;
        }
        __syncthreads();

        if (k == 0) {
            if (!use_gumbel) {
                float best = s_score[0];
                int best_k = 0;
                for (int j = 1; j < top_k_shortlist; ++j) {
                    if (s_score[j] > best) {
                        best = s_score[j];
                        best_k = j;
                    }
                }
                const unsigned int chosen = s_cand[best_k];
                float denom = 0.0f;
                for (int j = 0; j < top_k_shortlist; ++j) {
                    const float term = expf(s_score[j] - best);
                    denom += isfinite(term) ? term : 0.0f;
                }
                out_probs[t] = (denom > 0.0f && isfinite(denom))
                    ? __fdiv_rn(1.0f, denom)
                    : 1.0f;
                path[t] = chosen;
                s_prev = chosen;
            } else {
                for (int j = 0; j < top_k_shortlist; ++j) {
                    const float raw_sc = __fdiv_rn(s_score[j], eff_temp);
                    s_scaled[j] = isfinite(raw_sc) ? raw_sc : -1.0e30f;
                    s_ord[j] = j;
                }

                // Insertion sort descending by (s_scaled[idx], ~s_cand[idx]).
                for (int i = 1; i < top_k_shortlist; ++i) {
                    const int idx = s_ord[i];
                    const float sc = s_scaled[idx];
                    const unsigned int cid = s_cand[idx];
                    int j = i - 1;
                    while (j >= 0) {
                        const int prev_idx = s_ord[j];
                        const float prev_sc = s_scaled[prev_idx];
                        const unsigned int prev_cid = s_cand[prev_idx];
                        if (sc > prev_sc || (sc == prev_sc && cid < prev_cid)) {
                            s_ord[j + 1] = prev_idx;
                            --j;
                        } else {
                            break;
                        }
                    }
                    s_ord[j + 1] = idx;
                }

                int k_eff = (top_k > 0u && top_k < (unsigned int)top_k_shortlist)
                    ? (int)top_k
                    : top_k_shortlist;

                const float max_s = s_scaled[s_ord[0]];
                float sum_w = 0.0f;
                for (int j = 0; j < k_eff; ++j) {
                    const float w = expf(s_scaled[s_ord[j]] - max_s);
                    s_w[j] = w;
                    sum_w += w;
                }

                if (min_p > 0.0f && isfinite(min_p)) {
                    const float threshold = min_p * s_w[0];
                    while (k_eff > 1 && s_w[k_eff - 1] < threshold) {
                        --k_eff;
                    }
                }

                int kept = k_eff;
                if (top_p > 0.0f && top_p < 1.0f && sum_w > 0.0f) {
                    float rem_sum = 0.0f;
                    for (int j = 0; j < k_eff; ++j) {
                        rem_sum += s_w[j];
                    }
                    if (rem_sum > 0.0f) {
                        const float inv_sum = __fdiv_rn(1.0f, rem_sum);
                        float cum = 0.0f;
                        for (int j = 0; j < k_eff; ++j) {
                            cum += s_w[j] * inv_sum;
                            if (cum >= top_p) {
                                kept = j + 1;
                                break;
                            }
                        }
                    }
                }

                const unsigned long long pos_base =
                    df2_gumbel_pos_base(seed, base_pos + (unsigned long long)t);
                unsigned int best_cid = s_cand[s_ord[0]];
                float best_g = -INFINITY;
                int best_j = 0;
                for (int j = 0; j < kept; ++j) {
                    const int idx = s_ord[j];
                    const unsigned int cid = s_cand[idx];
                    const float g_score = s_scaled[idx] + df2_gumbel_from_base(pos_base, cid);
                    if (j == 0 || g_score > best_g || (g_score == best_g && cid < best_cid)) {
                        best_g = g_score;
                        best_cid = cid;
                        best_j = j;
                    }
                }
                float kept_sum = 0.0f;
                for (int j = 0; j < kept; ++j) {
                    kept_sum += s_w[j];
                }
                out_probs[t] = (kept_sum > 0.0f && isfinite(kept_sum))
                    ? __fdiv_rn(s_w[best_j], kept_sum)
                    : 1.0f;
                path[t] = best_cid;
                s_prev = best_cid;
            }
        }
        __syncthreads();
    }
}

// ── Stage-1 Coalesced BF16 Context Gather ───────────────────────────────────
//
// Gathers `total_rows` BF16 rows from per-row device base pointers
// (`src_ptrs[r] + src_row_idx[r] * row_elems`) into contiguous `dst[r, :]`,
// collapsing Stage-1's per-sequence ring-wrap D2D loops into a single launch.
// Grid: [total_rows, 1, 1], Block: [256, 1, 1].
extern "C" __global__ void __launch_bounds__(256) dflash_stage1_gather_ctx_bf16(
    __nv_bfloat16* __restrict__ dst,                 // [total_rows, row_elems]
    const unsigned long long* __restrict__ src_ptrs, // [total_rows] device ptrs to each row's base buffer
    const unsigned int* __restrict__ src_row_idx,    // [total_rows] row offset inside src_ptrs[r]
    int total_rows,
    int row_elems
) {
    const int r = (int)blockIdx.x;
    if (r >= total_rows || row_elems <= 0) return;

    const unsigned long long base_addr = src_ptrs[r];
    if (base_addr == 0ull) return;

    const size_t stride = (size_t)row_elems;
    const __nv_bfloat16* src_row =
        reinterpret_cast<const __nv_bfloat16*>(base_addr)
        + (size_t)src_row_idx[r] * stride;
    __nv_bfloat16* dst_row = dst + (size_t)r * stride;

    const bool aligned16 =
        ((((unsigned long long)(size_t)src_row | (unsigned long long)(size_t)dst_row) & 15ull) == 0ull);
    if (aligned16) {
        const int vec_count = row_elems >> 3;
        const uint4* src_v = reinterpret_cast<const uint4*>(src_row);
        uint4* dst_v = reinterpret_cast<uint4*>(dst_row);
        for (int i = (int)threadIdx.x; i < vec_count; i += (int)blockDim.x) {
            dst_v[i] = src_v[i];
        }
        const int tail_start = vec_count << 3;
        const int tail_rem = row_elems - tail_start;
        for (int t = (int)threadIdx.x; t < tail_rem; t += (int)blockDim.x) {
            dst_row[tail_start + t] = src_row[tail_start + t];
        }
    } else {
        for (int elem = (int)threadIdx.x; elem < row_elems; elem += (int)blockDim.x) {
            dst_row[elem] = src_row[elem];
        }
    }
}

// ── Fused YaRN-Scaled RoPE + Per-Sequence Circular KV-Ring Scatter ──────────
//
// Register-only bit-exact twin of CUDA libdevice's `sinf(x)` + `cosf(x)`.
// Replaces the 28-byte `.local` array in libdevice's Payne-Hanek slowpath with
// scalar registers so `ptxas` compiles with 0 B stack frame and 0 spills while
// matching `rope_forward_yarn_scaled` bit-for-bit.
__device__ __forceinline__ float df2_exact_cuda_trig_poly(float r, int q_poly, int q_sign) {
    const float r2 = __fmul_rn(r, r);
    const bool use_sin_poly = (q_poly & 1) != 0;
    const float f23 = use_sin_poly ? r : 1.0f;
    const float f24 = __fmaf_rn(r2, f23, 0.0f);
    const float f25 = __fmaf_rn(r2, __int_as_float(0x37CBAC00u), __int_as_float(0xBAB607EDu));
    const float f26 = use_sin_poly ? __int_as_float(0xB94D4153u) : f25;
    const float f27 = use_sin_poly ? __int_as_float(0x3C0885E4u) : __int_as_float(0x3D2AAABBu);
    const float f28 = __fmaf_rn(f26, r2, f27);
    const float f29 = use_sin_poly ? __int_as_float(0xBE2AAAA8u) : __int_as_float(0xBEFFFFFFu);
    const float f30 = __fmaf_rn(f28, r2, f29);
    const float f31 = __fmaf_rn(f30, f24, f23);
    return ((q_sign & 2) == 0) ? f31 : (0.0f - f31);
}

__device__ __forceinline__ void df2_exact_cuda_sincosf(float x, float& s_out, float& c_out) {
    int q = 0;
    float r = 0.0f;
    const float ax = fabsf(x);
    if (ax < __int_as_float(0x47CE4780u)) {
        const float prod = __fmul_rn(x, __int_as_float(0x3F22F983u));
        asm("cvt.rni.s32.f32 %0, %1;" : "=r"(q) : "f"(prod));
        const float fn = __int2float_rn(q);
        r = __fmaf_rn(fn, __int_as_float(0xBFC90FDAu), x);
        r = __fmaf_rn(fn, __int_as_float(0xB3A22168u), r);
        r = __fmaf_rn(fn, __int_as_float(0xA7C234C5u), r);
    } else {
        // Register-resident Payne-Hanek 192-bit 2/pi reduction (zero .local stack usage).
        const unsigned int ix = __float_as_uint(x);
        const unsigned int mant = (ix << 8) | 0x80000000u;
        const unsigned int i2opi[6] = {
            0x3c439041u, 0xdb629599u, 0xf534ddc0u,
            0xfc2757d1u, 0x4e441529u, 0xa2f9836eu
        };
        unsigned long long carry = 0ull;
        unsigned int p[7];
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            unsigned long long wide =
                (unsigned long long)i2opi[i] * (unsigned long long)mant + carry;
            p[i] = (unsigned int)wide;
            carry = wide >> 32;
        }
        p[6] = (unsigned int)carry;
        const unsigned int exp = ix >> 23;
        const unsigned int shift = exp & 31u;
        const unsigned int idx = ((exp & 224u) - 128u) >> 5;
        unsigned int hi = (idx == 0u) ? p[6] : (idx == 1u ? p[5] : (idx == 2u ? p[4] : p[3]));
        unsigned int lo = (idx == 0u) ? p[5] : (idx == 1u ? p[4] : (idx == 2u ? p[3] : p[2]));
        unsigned int ex = (idx == 0u) ? p[4] : (idx == 1u ? p[3] : (idx == 2u ? p[2] : p[1]));
        if (shift != 0u) {
            hi = __funnelshift_l(lo, hi, shift);
            lo = __funnelshift_l(ex, lo, shift);
        }
        const unsigned int q_bits = hi >> 30;
        const unsigned int f_hi = __funnelshift_l(lo, hi, 2u);
        const unsigned int f_lo = lo << 2;
        const int q_abs = (int)((f_hi >> 31) + q_bits);
        q = ((int)ix < 0) ? -q_abs : q_abs;
        const int sign_flip = (int)f_hi ^ (int)ix;
        const int mask = ((int)f_hi) >> 31;
        const unsigned long long mag =
            (((unsigned long long)(f_hi ^ (unsigned int)mask)) << 32) |
            ((unsigned long long)(f_lo ^ (unsigned int)mask));
        const double d_red =
            __ll2double_rn((long long)mag) * __longlong_as_double(0x3BF921FB54442D19ull);
        const float f_red = __double2float_rn(d_red);
        r = (sign_flip < 0) ? -f_red : f_red;
    }
    s_out = df2_exact_cuda_trig_poly(r, q + 1, q);
    c_out = df2_exact_cuda_trig_poly(r, q, q + 1);
}

// Replaces per-layer `rope_yarn_scaled` + per-sequence KV ring-buffer D2D loops
// in Stage-1 (`run_stage1_batched_ctx_append` in `batched_propose.rs`) with a
// single fused launch per layer. For each row `r in 0..total_rows`:
//   - Applies exact YaRN-scaled rotate_half RoPE to `k_in[r, :, 0..rotary_dim]`
//     at `row_pos[r]` (and copies any unrotated tail `rotary_dim..head_dim`),
//     writing directly to `dst_k_ptrs[r] + dst_slot[r] * kv_stride`.
//   - Copies `v_in[r, :]` directly to `dst_v_ptrs[r] + dst_slot[r] * kv_stride`
//     via coalesced 128-bit `uint4` loads/stores (with alignment & tail safety).
// Grid: [total_rows, 1, 1], Block: [256, 1, 1].
extern "C" __global__ void __launch_bounds__(256) dflash_rope_yarn_scaled_scatter_ring_bf16(
    const __nv_bfloat16* __restrict__ k_in,            // [total_rows, num_kv_heads * head_dim]
    const __nv_bfloat16* __restrict__ v_in,            // [total_rows, num_kv_heads * head_dim]
    const unsigned long long* __restrict__ dst_k_ptrs, // [total_rows] per-row ctx_k_cache base
    const unsigned long long* __restrict__ dst_v_ptrs, // [total_rows] per-row ctx_v_cache base
    const unsigned int* __restrict__ dst_slot,         // [total_rows] physical slot in KV ring
    const unsigned int* __restrict__ row_pos,          // [total_rows] absolute position for RoPE
    const float* __restrict__ inv_freq,                // [rotary_dim / 2] YaRN inv_freq
    float attention_factor,                            // YaRN cos/sin scale (1.0f for vanilla RoPE)
    int total_rows,
    int num_kv_heads,
    int head_dim,
    int rotary_dim
) {
    const int r = (int)blockIdx.x;
    if (r >= total_rows || num_kv_heads <= 0 || head_dim <= 0) return;
    if (rotary_dim < 0 || rotary_dim > head_dim || (rotary_dim & 1) != 0) return;

    const unsigned int tid = threadIdx.x;
    const size_t kv_stride = (size_t)num_kv_heads * (size_t)head_dim;
    const size_t slot = (size_t)dst_slot[r];

    const unsigned long long dst_k_addr = dst_k_ptrs[r];
    if (dst_k_addr != 0ull) {
        const __nv_bfloat16* src_k = k_in + (size_t)r * kv_stride;
        __nv_bfloat16* dst_k =
            reinterpret_cast<__nv_bfloat16*>(dst_k_addr) + slot * kv_stride;

        const unsigned int abs_pos = row_pos[r];
        const int half_rot = rotary_dim >> 1;

        __shared__ float s_cos[256];
        __shared__ float s_sin[256];

        for (int pair_base = 0; pair_base < half_rot; pair_base += 256) {
            const int chunk_pairs = min(256, half_rot - pair_base);
            if ((int)tid < chunk_pairs) {
                const float angle = (float)abs_pos * inv_freq[pair_base + (int)tid];
                float sin_raw, cos_raw;
                df2_exact_cuda_sincosf(angle, sin_raw, cos_raw);
                s_cos[tid] = cos_raw * attention_factor;
                s_sin[tid] = sin_raw * attention_factor;
            }
            __syncthreads();

            const int total_chunk_work = num_kv_heads * chunk_pairs;
            for (int idx = (int)tid; idx < total_chunk_work; idx += (int)blockDim.x) {
                const int h = idx / chunk_pairs;
                const int local_p = idx - h * chunk_pairs;
                const int pair_idx = pair_base + local_p;
                const size_t head_off = (size_t)h * (size_t)head_dim;
                const size_t d0 = head_off + (size_t)pair_idx;
                const size_t d1 = d0 + (size_t)half_rot;
                const float cos_val = s_cos[local_p];
                const float sin_val = s_sin[local_p];
                const float x0 = __bfloat162float(src_k[d0]);
                const float x1 = __bfloat162float(src_k[d1]);
                dst_k[d0] = __float2bfloat16((x0 * cos_val) - (x1 * sin_val));
                dst_k[d1] = __float2bfloat16((x1 * cos_val) + (x0 * sin_val));
            }
            if (pair_base + 256 < half_rot) {
                __syncthreads();
            }
        }

        const int tail_dim = head_dim - rotary_dim;
        if (tail_dim > 0) {
            const int total_tail = num_kv_heads * tail_dim;
            for (int idx = (int)tid; idx < total_tail; idx += (int)blockDim.x) {
                const int h = idx / tail_dim;
                const int d = rotary_dim + (idx - h * tail_dim);
                const size_t off = (size_t)h * (size_t)head_dim + (size_t)d;
                dst_k[off] = src_k[off];
            }
        }
    }

    const unsigned long long dst_v_addr = dst_v_ptrs[r];
    if (dst_v_addr != 0ull) {
        const __nv_bfloat16* src_v = v_in + (size_t)r * kv_stride;
        __nv_bfloat16* dst_v =
            reinterpret_cast<__nv_bfloat16*>(dst_v_addr) + slot * kv_stride;
        const int kv_elems = num_kv_heads * head_dim;
        const bool v_aligned16 =
            ((((unsigned long long)(size_t)src_v | (unsigned long long)(size_t)dst_v) & 15ull) == 0ull);
        if (v_aligned16) {
            const int vec_count = kv_elems >> 3;
            const uint4* src_vv = reinterpret_cast<const uint4*>(src_v);
            uint4* dst_vv = reinterpret_cast<uint4*>(dst_v);
            for (int i = (int)tid; i < vec_count; i += (int)blockDim.x) {
                dst_vv[i] = src_vv[i];
            }
            const int tail_start = vec_count << 3;
            const int tail_rem = kv_elems - tail_start;
            for (int t = (int)tid; t < tail_rem; t += (int)blockDim.x) {
                dst_v[tail_start + t] = src_v[tail_start + t];
            }
        } else {
            for (int elem = (int)tid; elem < kv_elems; elem += (int)blockDim.x) {
                dst_v[elem] = src_v[elem];
            }
        }
    }
}

// ── Wave-15 Batched Probabilistic DFlash2 Selector Walk (Maximal Coupling) ──
//
// Stateless position-keyed uniform variate in [0, 1) with domain salt
// `0xDF2D5A31ULL`, independent of Gumbel per-token keys and verifier-side RNG.
__device__ __forceinline__ float df2_prob_uniform(
    unsigned long long seed, unsigned long long abs_pos
) {
    const unsigned long long pos_base = df2_gumbel_pos_base(seed, abs_pos);
    const unsigned long long bits = df2_splitmix64_mix(pos_base ^ 0xDF2D5A31ull);
    return fminf((float)(bits >> 40) * 0x1.0p-24f, 0x1.fffffep-1f);
}

// Generalized `1 <= top_k_shortlist <= 32` batched selector walk on a 32-thread
// warp CTA that exports the exact post-policy draft distribution `q[0..kept-1]`
// directly to GPU memory (`out_q_ids`, `out_q_probs`) with zero D2H/H2D.
// Grid: [batch_size, 1, 1], Block: [32, 1, 1].
extern "C" __global__ void dflash2_selector_walk_probabilistic_batched(
    const float* __restrict__ unary,                      // [batch_size * gamma, top_k_shortlist]
    const unsigned int* __restrict__ candidates,          // [batch_size * gamma, top_k_shortlist]
    const __nv_bfloat16* __restrict__ hidden,             // [batch_size * gamma, rank]
    const __nv_bfloat16* __restrict__ pred,               // [vocab, rank]
    const __nv_bfloat16* __restrict__ succ,               // [vocab, rank]
    unsigned int* __restrict__ out_tokens,                // [batch_size * gamma] out
    float* __restrict__ out_max_prob,                     // [batch_size * gamma] out
    unsigned int* __restrict__ out_q_ids,                 // [*, out_q_stride] out
    float* __restrict__ out_q_probs,                      // [*, out_q_stride] out
    const unsigned int* __restrict__ out_q_row_bases,     // [batch_size] optional row base (nullptr -> b * gamma)
    const unsigned int* __restrict__ anchor_ids,          // [batch_size]
    const unsigned long long* __restrict__ seeds,         // [batch_size]
    const unsigned long long* __restrict__ step0_abs_pos, // [batch_size]
    const float* __restrict__ temperatures,               // [batch_size] (<= 0.0 -> greedy 1-hot)
    const unsigned int* __restrict__ top_ks,              // [batch_size] (0 -> all top_k_shortlist)
    const float* __restrict__ top_ps,                     // [batch_size] (<= 0 or >= 1 -> disabled)
    const float* __restrict__ min_ps,                     // [batch_size] (<= 0 -> disabled)
    int top_k_shortlist,                                  // 1..=32 (e.g. 16 or 20)
    int out_q_stride,                                     // >= 1 (e.g. 20 or 32)
    int gamma,
    int rank,
    int batch_size
) {
    const int b = (int)blockIdx.x;
    if (b >= batch_size || top_k_shortlist <= 0 || top_k_shortlist > DF2_TOPK_MAX || out_q_stride <= 0) {
        return;
    }

    const size_t stride_k = (size_t)top_k_shortlist;
    const size_t stride_q = (size_t)out_q_stride;
    const size_t row_base = (size_t)b * (size_t)gamma;
    const size_t q_row_base = (out_q_row_bases != nullptr)
        ? static_cast<size_t>(out_q_row_bases[b])
        : row_base;
    unary        += row_base * stride_k;
    candidates   += row_base * stride_k;
    hidden       += row_base * (size_t)rank;
    out_tokens   += row_base;
    out_max_prob += row_base;
    out_q_ids    += q_row_base * stride_q;
    out_q_probs  += q_row_base * stride_q;

    const unsigned int anchor_id = anchor_ids[b];
    const float temp = temperatures[b];
    const unsigned long long seed = seeds[b];
    const unsigned long long base_pos = step0_abs_pos[b];
    const unsigned int top_k = top_ks[b];
    const float top_p = top_ps[b];
    const float min_p = min_ps[b];

    const bool use_prob = (temp > 0.0f) && isfinite(temp);
    const float eff_temp = use_prob ? fmaxf(temp, 1.0e-6f) : 1.0f;

    __shared__ float s_score[DF2_TOPK_MAX];
    __shared__ unsigned int s_cand[DF2_TOPK_MAX];
    __shared__ float s_scaled[DF2_TOPK_MAX];
    __shared__ int s_ord[DF2_TOPK_MAX];
    __shared__ float s_w[DF2_TOPK_MAX];
    __shared__ int s_kept;
    __shared__ unsigned int s_prev;

    const unsigned int k = threadIdx.x;   // one thread per candidate

    if (k == 0) s_prev = anchor_id;
    __syncthreads();

    for (int t = 0; t < gamma; ++t) {
        const __nv_bfloat16* pr = pred + (size_t)s_prev * (size_t)rank;
        const __nv_bfloat16* hr = hidden + (size_t)t * (size_t)rank;

        float acc = 0.0f;
        if ((int)k < top_k_shortlist) {
            const unsigned int cand = candidates[(size_t)t * stride_k + k];
            const __nv_bfloat16* sr = succ + (size_t)cand * (size_t)rank;
            for (int r = 0; r < rank; ++r) {
                acc += __bfloat162float(pr[r]) * __bfloat162float(hr[r])
                     * __bfloat162float(sr[r]);
            }
            s_score[k] = unary[(size_t)t * stride_k + k] + acc;
            s_cand[k] = cand;
        }
        __syncthreads();

        if (k == 0) {
            if (!use_prob) {
                float best = s_score[0];
                int best_k = 0;
                for (int j = 1; j < top_k_shortlist; ++j) {
                    if (s_score[j] > best) {
                        best = s_score[j];
                        best_k = j;
                    }
                }
                const unsigned int chosen = s_cand[best_k];
                float denom = 0.0f;
                for (int j = 0; j < top_k_shortlist; ++j) {
                    const float term = expf(s_score[j] - best);
                    denom += isfinite(term) ? term : 0.0f;
                }
                out_tokens[t] = chosen;
                out_max_prob[t] = (denom > 0.0f && isfinite(denom))
                    ? __fdiv_rn(1.0f, denom)
                    : 1.0f;
                s_ord[0] = best_k;
                s_w[0] = 1.0f;
                s_kept = 1;
                s_prev = chosen;
            } else {
                for (int j = 0; j < top_k_shortlist; ++j) {
                    const float raw_sc = __fdiv_rn(s_score[j], eff_temp);
                    s_scaled[j] = isfinite(raw_sc) ? raw_sc : -1.0e30f;
                    s_ord[j] = j;
                }

                // Insertion sort descending by (s_scaled[idx], ~s_cand[idx]).
                for (int i = 1; i < top_k_shortlist; ++i) {
                    const int idx = s_ord[i];
                    const float sc = s_scaled[idx];
                    const unsigned int cid = s_cand[idx];
                    int j = i - 1;
                    while (j >= 0) {
                        const int prev_idx = s_ord[j];
                        const float prev_sc = s_scaled[prev_idx];
                        const unsigned int prev_cid = s_cand[prev_idx];
                        if (sc > prev_sc || (sc == prev_sc && cid < prev_cid)) {
                            s_ord[j + 1] = prev_idx;
                            --j;
                        } else {
                            break;
                        }
                    }
                    s_ord[j + 1] = idx;
                }

                int k_eff = (top_k > 0u && top_k < (unsigned int)top_k_shortlist)
                    ? (int)top_k
                    : top_k_shortlist;

                const float max_s = s_scaled[s_ord[0]];
                float sum_w = 0.0f;
                for (int j = 0; j < k_eff; ++j) {
                    const float w = expf(s_scaled[s_ord[j]] - max_s);
                    s_w[j] = w;
                    sum_w += w;
                }

                if (min_p > 0.0f && isfinite(min_p)) {
                    const float threshold = min_p * s_w[0];
                    while (k_eff > 1 && s_w[k_eff - 1] < threshold) {
                        --k_eff;
                    }
                }

                int kept = k_eff;
                if (top_p > 0.0f && top_p < 1.0f && sum_w > 0.0f) {
                    float rem_sum = 0.0f;
                    for (int j = 0; j < k_eff; ++j) {
                        rem_sum += s_w[j];
                    }
                    if (rem_sum > 0.0f) {
                        const float inv_sum = __fdiv_rn(1.0f, rem_sum);
                        float cum = 0.0f;
                        for (int j = 0; j < k_eff; ++j) {
                            cum += s_w[j] * inv_sum;
                            if (cum >= top_p) {
                                kept = j + 1;
                                break;
                            }
                        }
                    }
                }
                if (kept > out_q_stride) {
                    kept = out_q_stride;
                }

                float kept_sum = 0.0f;
                for (int j = 0; j < kept; ++j) {
                    kept_sum += s_w[j];
                }
                const bool valid_sum = (kept_sum > 0.0f) && isfinite(kept_sum);
                for (int j = 0; j < kept; ++j) {
                    s_w[j] = valid_sum
                        ? __fdiv_rn(s_w[j], kept_sum)
                        : (j == 0 ? 1.0f : 0.0f);
                }

                const float u = df2_prob_uniform(seed, base_pos + (unsigned long long)t);
                float cum = 0.0f;
                int selected_j = 0;
                int last_nonzero = 0;
                for (int j = 0; j < kept; ++j) {
                    const float p = s_w[j];
                    if (p > 0.0f) {
                        last_nonzero = j;
                    }
                    cum += p;
                    if (u < cum) {
                        selected_j = j;
                        break;
                    }
                    selected_j = last_nonzero;
                }
                const unsigned int chosen = s_cand[s_ord[selected_j]];
                out_tokens[t] = chosen;
                // Export max_z q_t(z | x_{<t}) (independent of sampled x_t) so
                // forward p_cum truncation stays strictly F_t-measurable and
                // preserves exact maximal-coupling rejection sampling (TV = 0).
                out_max_prob[t] = s_w[0];
                s_kept = kept;
                s_prev = chosen;
            }
        }
        __syncthreads();

        const int kept_out = s_kept;
        unsigned int* row_q_ids = out_q_ids + (size_t)t * stride_q;
        float* row_q_probs = out_q_probs + (size_t)t * stride_q;
        for (int j = (int)k; j < out_q_stride; j += 32) {
            if (j < kept_out) {
                row_q_ids[j] = s_cand[s_ord[j]];
                row_q_probs[j] = s_w[j];
            } else {
                row_q_ids[j] = 0xFFFFFFFFu;
                row_q_probs[j] = 0.0f;
            }
        }
        __syncthreads();
    }
}

// ── 4-Gate Safe In-Context N-Gram Bonus DFlash2 Selector Walk (T=0 ONLY) ─────
//
// Opt-in (`ATLAS_DFLASH_NGRAM_BONUS=1`, default OFF). Appended as a separate
// symbol so every existing selector kernel above remains bit-for-bit unchanged.
//
// Safety invariants enforced in-kernel:
//   Gate 0 (T=0 ONLY): bypassed whenever `temperatures != nullptr &&
//          temperatures[b] > 0.0f && isfinite(temperatures[b])`.
//   Gate 1 (Uncertainty): fires ONLY when `top1 - top2 < margin_thresh`.
//   Gate 2 (Top-4 verbatim 3/4-gram): probes open-addressed sequence table for
//          4-gram first then 3-gram; boosts ONLY if matched `next_token` is
//          already ranked inside top-4 (`0..min(4, top_k_shortlist)`).
//   Gate 3 (Hard logit cap): bonus strictly clamped `<= min(max_bonus, 0.75f)`
//          (`0.50f` for 3-gram, `0.75f` for 4-gram).
//   Gate 4 (Anti-loop <= 64): host table builder excludes all matches inside
//          the trailing 64-token window and any locally repeating 3-gram cycle.

#define DF2_NGRAM_TABLE_SLOTS 512u
#define DF2_NGRAM_TABLE_MASK  (DF2_NGRAM_TABLE_SLOTS - 1u)
#define DF2_NGRAM_MAX_PROBE   8u
#define DF2_NGRAM_TOP_RANK    4
#define DF2_NGRAM_HARD_CAP    0.75f
#define DF2_NGRAM_SALT3       0x9E3779B97F4A7C15ull
#define DF2_NGRAM_SALT4       0xBF58476D1CE4E5B9ull
#define DF2_NGRAM_EMPTY_TOK   0xFFFFFFFFu

struct Df2NgramEntry {
    unsigned long long key;
    unsigned int next_token;
    unsigned int match_len;
};

__device__ __forceinline__ unsigned long long df2_ngram_key3(
    unsigned int a, unsigned int b, unsigned int c
) {
    unsigned long long h = df2_splitmix64_mix((unsigned long long)a ^ DF2_NGRAM_SALT3);
    h = df2_splitmix64_mix(h ^ (unsigned long long)b);
    h = df2_splitmix64_mix(h ^ (unsigned long long)c);
    return (h == 0ull) ? 1ull : h;
}

__device__ __forceinline__ unsigned long long df2_ngram_key4(
    unsigned int a, unsigned int b, unsigned int c, unsigned int d
) {
    unsigned long long h = df2_splitmix64_mix((unsigned long long)a ^ DF2_NGRAM_SALT4);
    h = df2_splitmix64_mix(h ^ (unsigned long long)b);
    h = df2_splitmix64_mix(h ^ (unsigned long long)c);
    h = df2_splitmix64_mix(h ^ (unsigned long long)d);
    return (h == 0ull) ? 1ull : h;
}

__device__ __forceinline__ unsigned int df2_ngram_probe(
    const Df2NgramEntry* __restrict__ table,
    unsigned long long key,
    unsigned int expected_len
) {
    unsigned int slot = (unsigned int)(key & (unsigned long long)DF2_NGRAM_TABLE_MASK);
    #pragma unroll
    for (unsigned int step = 0u; step < DF2_NGRAM_MAX_PROBE; ++step) {
        const Df2NgramEntry entry = table[slot];
        if (entry.key == 0ull) {
            return DF2_NGRAM_EMPTY_TOK;
        }
        if (entry.key == key && entry.match_len == expected_len) {
            return entry.next_token;
        }
        slot = (slot + 1u) & DF2_NGRAM_TABLE_MASK;
    }
    return DF2_NGRAM_EMPTY_TOK;
}

// Grid: [batch_size, 1, 1], Block: [32, 1, 1].
extern "C" __global__ void __launch_bounds__(32) dflash2_selector_walk_ngram_batched(
    const float* __restrict__ unary,               // [batch_size * gamma, top_k_shortlist]
    const unsigned int* __restrict__ candidates,   // [batch_size * gamma, top_k_shortlist]
    const __nv_bfloat16* __restrict__ hidden,      // [batch_size * gamma, rank]
    const __nv_bfloat16* __restrict__ pred,        // [vocab, rank]
    const __nv_bfloat16* __restrict__ succ,        // [vocab, rank]
    unsigned int* __restrict__ out_tokens,         // [batch_size * gamma] out
    float* __restrict__ out_probs,                 // [batch_size * gamma] optional out
    const unsigned int* __restrict__ anchor_ids,   // [batch_size]
    const unsigned int* __restrict__ prev3_tokens, // [batch_size, 3] ([t-3, t-2, t-1], 0xFFFFFFFF if absent)
    const Df2NgramEntry* __restrict__ ngram_tables,// [batch_size, 512]
    const float* __restrict__ temperatures,        // [batch_size] optional (nullptr -> greedy T=0)
    float margin_thresh,
    float max_bonus,
    int top_k_shortlist,
    int gamma,
    int rank,
    int batch_size
) {
    const int b = (int)blockIdx.x;
    if (b >= batch_size || batch_size <= 0 || gamma <= 0 || rank <= 0
        || top_k_shortlist <= 0 || top_k_shortlist > DF2_TOPK_MAX) {
        return;
    }

    const size_t stride_k = (size_t)top_k_shortlist;
    const size_t row_base = (size_t)b * (size_t)gamma;
    unary      += row_base * stride_k;
    candidates += row_base * stride_k;
    hidden     += row_base * (size_t)rank;
    out_tokens += row_base;
    if (out_probs != nullptr) {
        out_probs += row_base;
    }

    const Df2NgramEntry* seq_table =
        (ngram_tables != nullptr) ? (ngram_tables + (size_t)b * DF2_NGRAM_TABLE_SLOTS) : nullptr;
    const float temp = (temperatures != nullptr) ? temperatures[b] : 0.0f;
    // Gate 0: strict T=0 ONLY (fail closed on T > 0 or non-finite temperature).
    const bool gate0_greedy_ok = isfinite(temp) && (temp <= 0.0f);
    const float eff_max_bonus =
        (isfinite(max_bonus) && max_bonus > 0.0f) ? fminf(max_bonus, DF2_NGRAM_HARD_CAP) : 0.0f;
    const bool ngram_active =
        gate0_greedy_ok && (seq_table != nullptr)
        && isfinite(margin_thresh) && (margin_thresh > 0.0f)
        && (eff_max_bonus > 0.0f);

    __shared__ float s_score[DF2_TOPK_MAX];
    __shared__ unsigned int s_cand[DF2_TOPK_MAX];
    __shared__ unsigned int s_hist[4]; // [h_{t-4}, h_{t-3}, h_{t-2}, h_{t-1}=s_prev]

    const unsigned int k = threadIdx.x;
    if (k == 0u) {
        if (prev3_tokens != nullptr) {
            s_hist[0] = prev3_tokens[(size_t)b * 3u + 0u];
            s_hist[1] = prev3_tokens[(size_t)b * 3u + 1u];
            s_hist[2] = prev3_tokens[(size_t)b * 3u + 2u];
        } else {
            s_hist[0] = DF2_NGRAM_EMPTY_TOK;
            s_hist[1] = DF2_NGRAM_EMPTY_TOK;
            s_hist[2] = DF2_NGRAM_EMPTY_TOK;
        }
        s_hist[3] = anchor_ids[b];
    }
    __syncthreads();

    for (int t = 0; t < gamma; ++t) {
        const unsigned int prev_tok = s_hist[3];
        const __nv_bfloat16* pr = pred + (size_t)prev_tok * (size_t)rank;
        const __nv_bfloat16* hr = hidden + (size_t)t * (size_t)rank;

        if ((int)k < top_k_shortlist) {
            const unsigned int cand = candidates[(size_t)t * stride_k + k];
            const __nv_bfloat16* sr = succ + (size_t)cand * (size_t)rank;
            float acc = 0.0f;
            for (int r = 0; r < rank; ++r) {
                acc += __bfloat162float(pr[r]) * __bfloat162float(hr[r])
                     * __bfloat162float(sr[r]);
            }
            s_score[k] = unary[(size_t)t * stride_k + k] + acc;
            s_cand[k] = cand;
        }
        __syncthreads();

        if (k == 0u) {
            // Find top-4 slots descending by (s_score[j], ~s_cand[j]) without sorting all K.
            int top_idx[DF2_NGRAM_TOP_RANK];
            const int rank_cap = (top_k_shortlist < DF2_NGRAM_TOP_RANK)
                ? top_k_shortlist
                : DF2_NGRAM_TOP_RANK;
            unsigned int used_mask = 0u;
            for (int r = 0; r < rank_cap; ++r) {
                int best = -1;
                for (int j = 0; j < top_k_shortlist; ++j) {
                    if ((used_mask & (1u << j)) != 0u) continue;
                    if (best < 0 || s_score[j] > s_score[best]
                        || (s_score[j] == s_score[best] && s_cand[j] < s_cand[best])) {
                        best = j;
                    }
                }
                top_idx[r] = best;
                used_mask |= (1u << best);
            }

            // Gate 1: uncertainty margin `top1 - top2 < margin_thresh`.
            if (ngram_active && rank_cap >= 2) {
                const float margin = s_score[top_idx[0]] - s_score[top_idx[1]];
                if (isfinite(margin) && margin < margin_thresh) {
                    unsigned int matched_tok = DF2_NGRAM_EMPTY_TOK;
                    float bonus = 0.0f;
                    // Gate 2a: verbatim 4-gram first (bonus <= 0.75), then 3-gram (bonus <= 0.50).
                    if (s_hist[0] != DF2_NGRAM_EMPTY_TOK && s_hist[1] != DF2_NGRAM_EMPTY_TOK
                        && s_hist[2] != DF2_NGRAM_EMPTY_TOK) {
                        matched_tok = df2_ngram_probe(
                            seq_table,
                            df2_ngram_key4(s_hist[0], s_hist[1], s_hist[2], s_hist[3]),
                            4u
                        );
                        if (matched_tok != DF2_NGRAM_EMPTY_TOK) {
                            bonus = eff_max_bonus;
                        }
                    }
                    if (matched_tok == DF2_NGRAM_EMPTY_TOK
                        && s_hist[1] != DF2_NGRAM_EMPTY_TOK
                        && s_hist[2] != DF2_NGRAM_EMPTY_TOK) {
                        matched_tok = df2_ngram_probe(
                            seq_table,
                            df2_ngram_key3(s_hist[1], s_hist[2], s_hist[3]),
                            3u
                        );
                        if (matched_tok != DF2_NGRAM_EMPTY_TOK) {
                            bonus = fminf(eff_max_bonus, 0.50f);
                        }
                    }
                    // Gate 2b + Gate 3: apply capped bonus ONLY if matched_tok is inside top-4.
                    if (matched_tok != DF2_NGRAM_EMPTY_TOK && bonus > 0.0f) {
                        for (int r = 0; r < rank_cap; ++r) {
                            const int slot = top_idx[r];
                            if (s_cand[slot] == matched_tok) {
                                s_score[slot] += bonus;
                                break;
                            }
                        }
                    }
                }
            }

            // Final greedy argmax (exact tie-break matching existing selector).
            float best = s_score[0];
            int best_k = 0;
            for (int j = 1; j < top_k_shortlist; ++j) {
                if (s_score[j] > best) {
                    best = s_score[j];
                    best_k = j;
                }
            }
            const unsigned int chosen = s_cand[best_k];
            if (out_probs != nullptr) {
                float denom = 0.0f;
                for (int j = 0; j < top_k_shortlist; ++j) {
                    const float term = expf(s_score[j] - best);
                    denom += isfinite(term) ? term : 0.0f;
                }
                out_probs[t] = (denom > 0.0f && isfinite(denom))
                    ? __fdiv_rn(1.0f, denom)
                    : 1.0f;
            }
            out_tokens[t] = chosen;
            s_hist[0] = s_hist[1];
            s_hist[1] = s_hist[2];
            s_hist[2] = s_hist[3];
            s_hist[3] = chosen;
        }
        __syncthreads();
    }
}
