// SPDX-License-Identifier: AGPL-3.0-only
//
// Hardened On-GPU Exact Speculative Acceptance & Two-Pass Sparse Residual
// Sampling (`ATLAS_SPEC_EXACT_GPU_ACCEPT=1`, default OFF).
//
// Entrypoints:
//   1. `exact_spec_accept_sparse_sm120`:
//      Per-(seq, col) CTA (256 threads = 8 warps) evaluating exact target
//      probabilities `q` over BF16 logits with:
//        - Prefix-aware presence & frequency penalty adjustment across
//          in-window prefix tokens `0..col-1`, guaranteeing exact
//          `W=1 == W=8 == W=16` invariance;
//        - Exact IEEE `__fdiv_rn(logit, temp)` temperature scaling;
//        - Bit-exact Salmon et al. (2011) `Philox4_32_10` counter RNG keyed
//          by `(seed, abs_pos, 0, 0)` yielding exact 24-bit uniforms
//          `(u_accept, u_residual)` in `[0, 1)`;
//        - Fail-closed `|sum(p) - 1.0f| <= 1e-4f` validation on BOTH target
//          `q` and sparse draft support `p_sparse` (`K <= 64`, plus non-finite,
//          negative, duplicate-ID, and vocab-OOB checks) writing an explicit
//          device status bitmask;
//        - Two-pass sparse residual sampling `r(x) = max(q(x) - p_sparse(x), 0)`
//          via 1024-bit SMEM bloom prefilter + coalesced 256-lane tile scan
//          with guaranteed deterministic valid target-support zero-mass
//          fallback (NEVER emits `-1` / OOB on valid target distributions).
//   2. `exact_spec_prefix_reduce_sm120`:
//      Device-side per-sequence prefix scan emitting zero-host-sync
//      `d_num_accepted[seq]`, `d_bonus_token[seq]`, and `d_status_out[seq]`,
//      enforcing fail-closed `Generation` (`slot_generation == expected_generation`)
//      and `sharing <= 1` (`slot_refcount == 1` + cross-batch duplicate
//      physical `slot_id` detection) before emitting non-zero accept counts.
//
// Compiles with 0 B stack frame, 0 B spill stores/loads, 0 STL/LDL, and
// 0 odd `desc[UR]` across sm_121a, sm_121f, sm_120a, and sm_103a.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr uint32_t EXACT_SPEC_MAX_SPARSE_K = 64u;
constexpr uint32_t EXACT_SPEC_MAX_WIDTH = 16u;
constexpr uint32_t EXACT_SPEC_INVALID_TOKEN = 0xFFFFFFFFu;
constexpr float EXACT_SPEC_PROB_SUM_TOL = 1e-4f;

// Explicit device status bitmask flags (bits 0..7 are fail-closed errors;
// bit 8 records deterministic target-support fallback without failing closed).
constexpr uint32_t EXACT_SPEC_STATUS_OK = 0u;
constexpr uint32_t EXACT_SPEC_STATUS_BAD_PARAM = (1u << 0);
constexpr uint32_t EXACT_SPEC_STATUS_TARGET_NONFINITE = (1u << 1);
constexpr uint32_t EXACT_SPEC_STATUS_TARGET_SUM_ERR = (1u << 2);
constexpr uint32_t EXACT_SPEC_STATUS_DRAFT_NONFINITE = (1u << 3);
constexpr uint32_t EXACT_SPEC_STATUS_DRAFT_SUM_ERR = (1u << 4);
constexpr uint32_t EXACT_SPEC_STATUS_DRAFT_OOB = (1u << 5);
constexpr uint32_t EXACT_SPEC_STATUS_GEN_MISMATCH = (1u << 6);
constexpr uint32_t EXACT_SPEC_STATUS_SLOT_SHARED = (1u << 7);
constexpr uint32_t EXACT_SPEC_STATUS_FALLBACK_USED = (1u << 8);
constexpr uint32_t EXACT_SPEC_STATUS_ERROR_MASK = 0x00FFu;

// Canonical Salmon et al. (SC'11) Philox4x32-10 constants.
constexpr uint32_t PHILOX_M4X32_0 = 0xD2511F53u;
constexpr uint32_t PHILOX_M4X32_1 = 0xCD9E8D57u;
constexpr uint32_t PHILOX_W32_0 = 0x9E3779B9u;
constexpr uint32_t PHILOX_W32_1 = 0xBB67AE85u;

__device__ __forceinline__ void philox4x32_10(
    uint64_t seed,
    uint64_t abs_pos,
    uint32_t& out0,
    uint32_t& out1
) {
    uint32_t k0 = (uint32_t)(seed & 0xFFFFFFFFull);
    uint32_t k1 = (uint32_t)(seed >> 32);
    uint32_t c0 = (uint32_t)(abs_pos & 0xFFFFFFFFull);
    uint32_t c1 = (uint32_t)(abs_pos >> 32);
    uint32_t c2 = 0u;
    uint32_t c3 = 0u;

    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        const uint64_t p0 = (uint64_t)PHILOX_M4X32_0 * (uint64_t)c0;
        const uint64_t p1 = (uint64_t)PHILOX_M4X32_1 * (uint64_t)c2;
        const uint32_t hi0 = (uint32_t)(p0 >> 32);
        const uint32_t lo0 = (uint32_t)p0;
        const uint32_t hi1 = (uint32_t)(p1 >> 32);
        const uint32_t lo1 = (uint32_t)p1;
        const uint32_t n0 = hi1 ^ c1 ^ k0;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3 ^ k1;
        const uint32_t n3 = lo0;
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
        if (r < 9) {
            k0 += PHILOX_W32_0;
            k1 += PHILOX_W32_1;
        }
    }
    out0 = c0;
    out1 = c1;
}

// Exact 24-bit mantissa dyadic uniform in [0.0f, 1.0f) (0 ULP vs Rust f32).
__device__ __forceinline__ float u32_to_uniform_f32(uint32_t x) {
    return (float)(x >> 8) * 5.9604644775390625e-8f;
}

__device__ __forceinline__ float compute_scaled_logit(
    __nv_bfloat16 raw_bf16,
    uint32_t v,
    const uint32_t* __restrict__ seq_pre_counts,
    const uint32_t* __restrict__ s_prefix_tokens,
    uint32_t col,
    float presence_penalty,
    float frequency_penalty,
    float temp,
    bool& is_valid,
    bool& is_fatal_nonfinite
) {
    float logit = __bfloat162float(raw_bf16);
    if (isnan(logit) || (isinf(logit) && logit > 0.0f)) {
        is_valid = false;
        is_fatal_nonfinite = true;
        return -INFINITY;
    }
    if (!isfinite(logit)) {
        // Explicit -Inf masks out a token cleanly (q_v = 0).
        is_valid = false;
        is_fatal_nonfinite = false;
        return -INFINITY;
    }
    if (presence_penalty != 0.0f || frequency_penalty != 0.0f) {
        uint32_t total_cnt = (seq_pre_counts != nullptr) ? seq_pre_counts[v] : 0u;
        for (uint32_t p = 0u; p < col; ++p) {
            if (s_prefix_tokens[p] == v) {
                total_cnt += 1u;
            }
        }
        if (total_cnt > 0u) {
            if (presence_penalty != 0.0f) {
                logit = __fsub_rn(logit, presence_penalty);
            }
            if (frequency_penalty != 0.0f) {
                logit = __fsub_rn(logit, __fmul_rn(frequency_penalty, (float)total_cnt));
            }
        }
    }
    const float scaled = __fdiv_rn(logit, temp);
    if (!isfinite(scaled)) {
        is_valid = false;
        is_fatal_nonfinite = true;
        return -INFINITY;
    }
    is_valid = true;
    is_fatal_nonfinite = false;
    return scaled;
}

__device__ __forceinline__ float lookup_sparse_prob(
    uint32_t v,
    const uint32_t* __restrict__ s_bloom,
    const uint32_t* __restrict__ s_p_ids,
    const float* __restrict__ s_p_probs,
    uint32_t active_k
) {
    if (active_k == 0u) {
        return 0.0f;
    }
    const uint32_t word = s_bloom[(v >> 5) & 31u];
    if ((word & (1u << (v & 31u))) == 0u) {
        return 0.0f;
    }
    for (uint32_t j = 0u; j < active_k; ++j) {
        if (s_p_ids[j] == v) {
            return s_p_probs[j];
        }
    }
    return 0.0f;
}

__device__ __forceinline__ float block_reduce_sum_256(
    float val,
    float* __restrict__ s_warp_f,
    uint32_t warp,
    uint32_t lane
) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = __fadd_rn(val, __shfl_down_sync(0xFFFFFFFFu, val, offset));
    }
    if (lane == 0u) {
        s_warp_f[warp] = val;
    }
    __syncthreads();
    if (warp == 0u) {
        float w = (lane < 8u) ? s_warp_f[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            w = __fadd_rn(w, __shfl_down_sync(0xFFFFFFFFu, w, offset));
        }
        if (lane == 0u) {
            s_warp_f[0] = w;
        }
    }
    __syncthreads();
    const float result = s_warp_f[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ void block_reduce_argmax_256(
    float val,
    uint32_t id,
    float* __restrict__ s_warp_f,
    uint32_t* __restrict__ s_warp_u,
    uint32_t warp,
    uint32_t lane,
    float& out_val,
    uint32_t& out_id
) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float o_val = __shfl_down_sync(0xFFFFFFFFu, val, offset);
        const uint32_t o_id = __shfl_down_sync(0xFFFFFFFFu, id, offset);
        if (o_id != EXACT_SPEC_INVALID_TOKEN &&
            (id == EXACT_SPEC_INVALID_TOKEN || o_val > val || (o_val == val && o_id < id))) {
            val = o_val;
            id = o_id;
        }
    }
    if (lane == 0u) {
        s_warp_f[warp] = val;
        s_warp_u[warp] = id;
    }
    __syncthreads();
    if (warp == 0u) {
        float w_val = (lane < 8u) ? s_warp_f[lane] : -INFINITY;
        uint32_t w_id = (lane < 8u) ? s_warp_u[lane] : EXACT_SPEC_INVALID_TOKEN;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            const float o_val = __shfl_down_sync(0xFFFFFFFFu, w_val, offset);
            const uint32_t o_id = __shfl_down_sync(0xFFFFFFFFu, w_id, offset);
            if (o_id != EXACT_SPEC_INVALID_TOKEN &&
                (w_id == EXACT_SPEC_INVALID_TOKEN || o_val > w_val ||
                 (o_val == w_val && o_id < w_id))) {
                w_val = o_val;
                w_id = o_id;
            }
        }
        if (lane == 0u) {
            s_warp_f[0] = w_val;
            s_warp_u[0] = w_id;
        }
    }
    __syncthreads();
    out_val = s_warp_f[0];
    out_id = s_warp_u[0];
    __syncthreads();
}

__device__ __forceinline__ float tile_inclusive_scan_256(
    float val,
    float* __restrict__ s_warp_f,
    uint32_t warp,
    uint32_t lane,
    float& out_tile_total
) {
    float x = val;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float up = __shfl_up_sync(0xFFFFFFFFu, x, offset);
        if ((int)lane >= offset) {
            x = __fadd_rn(x, up);
        }
    }
    if (lane == 31u) {
        s_warp_f[warp] = x;
    }
    __syncthreads();
    if (warp == 0u) {
        float ws = (lane < 8u) ? s_warp_f[lane] : 0.0f;
        #pragma unroll
        for (int offset = 1; offset < 8; offset <<= 1) {
            const float up = __shfl_up_sync(0xFFFFFFFFu, ws, offset);
            if ((int)lane >= offset) {
                ws = __fadd_rn(ws, up);
            }
        }
        if (lane < 8u) {
            s_warp_f[lane] = ws;
        }
    }
    __syncthreads();
    const float warp_base = (warp > 0u) ? s_warp_f[warp - 1u] : 0.0f;
    out_tile_total = s_warp_f[7];
    __syncthreads();
    return __fadd_rn(x, warp_base);
}

}  // namespace

extern "C" __global__ __launch_bounds__(256, 4) void exact_spec_accept_sparse_sm120(
    const __nv_bfloat16* __restrict__ logits,            // [batch_size * verify_width, vocab_size]
    const uint32_t* __restrict__ draft_tokens,           // [batch_size * verify_width] (INVALID on bonus col)
    const uint32_t* __restrict__ sparse_draft_ids,       // [batch_size * verify_width, sparse_k]
    const float* __restrict__ sparse_draft_probs,        // [batch_size * verify_width, sparse_k]
    const uint32_t* __restrict__ verify_prefix_tokens,   // [batch_size, verify_width] (drafts[0..W-2] for prefix penalties)
    const uint32_t* __restrict__ pre_verify_counts,      // optional [batch_size, vocab_size] (nullable)
    const uint64_t* __restrict__ seeds,                  // [batch_size]
    const uint64_t* __restrict__ base_abs_pos,           // [batch_size]
    const float* __restrict__ temperatures,              // [batch_size] (must be finite > 0.0f)
    const float* __restrict__ presence_penalties,        // [batch_size]
    const float* __restrict__ frequency_penalties,       // [batch_size]
    uint32_t* __restrict__ out_accepted,                 // [batch_size * verify_width] (0 or 1)
    uint32_t* __restrict__ out_tokens,                   // [batch_size * verify_width]
    uint32_t* __restrict__ out_status,                   // [batch_size * verify_width]
    uint32_t batch_size,
    uint32_t verify_width,
    uint32_t vocab_size,
    uint32_t sparse_k
) {
    __shared__ uint32_t s_p_ids[EXACT_SPEC_MAX_SPARSE_K];
    __shared__ float s_p_probs[EXACT_SPEC_MAX_SPARSE_K];
    __shared__ uint32_t s_bloom[32];
    __shared__ uint32_t s_prefix_tokens[EXACT_SPEC_MAX_WIDTH];
    __shared__ float s_warp_f[8];
    __shared__ uint32_t s_warp_u[8];
    __shared__ uint32_t s_status;
    __shared__ uint32_t s_active_k;
    __shared__ float s_p_draft_x;
    __shared__ float s_u_accept;
    __shared__ float s_u_residual;
    __shared__ float s_max_scaled;
    __shared__ uint32_t s_argmax_id;
    __shared__ float s_sum_exp;
    __shared__ float s_sum_q;
    __shared__ float s_sum_r;
    __shared__ uint32_t s_accepted;
    __shared__ uint32_t s_sampled_tok;
    __shared__ uint32_t s_found;

    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t col = blockIdx.x;
    const uint32_t seq = blockIdx.y;

    if (seq >= batch_size || col >= verify_width) {
        return;
    }
    if (out_accepted == nullptr || out_tokens == nullptr || out_status == nullptr) {
        return;
    }

    const uint32_t row = seq * verify_width + col;

    // Parameter and pointer validation (fail-closed before reading logits/counts).
    if (verify_width == 0u || verify_width > EXACT_SPEC_MAX_WIDTH ||
        vocab_size == 0u || sparse_k == 0u || sparse_k > EXACT_SPEC_MAX_SPARSE_K ||
        logits == nullptr || seeds == nullptr || base_abs_pos == nullptr ||
        temperatures == nullptr || presence_penalties == nullptr ||
        frequency_penalties == nullptr ||
        (verify_width > 1u &&
         (draft_tokens == nullptr || sparse_draft_ids == nullptr ||
          sparse_draft_probs == nullptr || verify_prefix_tokens == nullptr))) {
        if (tid == 0u) {
            out_accepted[row] = 0u;
            out_tokens[row] = EXACT_SPEC_INVALID_TOKEN;
            out_status[row] = EXACT_SPEC_STATUS_BAD_PARAM;
        }
        return;
    }

    const float temp = temperatures[seq];
    const float pres_pen = presence_penalties[seq];
    const float freq_pen = frequency_penalties[seq];
    if (!isfinite(temp) || !(temp > 0.0f) || !isfinite(pres_pen) || !isfinite(freq_pen)) {
        if (tid == 0u) {
            out_accepted[row] = 0u;
            out_tokens[row] = EXACT_SPEC_INVALID_TOKEN;
            out_status[row] = EXACT_SPEC_STATUS_BAD_PARAM;
        }
        return;
    }

    const bool is_draft_row = (col + 1u < verify_width);
    const uint32_t draft_tok = is_draft_row ? draft_tokens[row] : EXACT_SPEC_INVALID_TOKEN;

    // Stage prefix tokens `0..col-1` and clear 1024-bit bloom mask in SMEM.
    if (tid < 32u) {
        s_bloom[tid] = 0u;
    }
    if (tid < col && verify_prefix_tokens != nullptr) {
        s_prefix_tokens[tid] = verify_prefix_tokens[(size_t)seq * (size_t)verify_width + (size_t)tid];
    } else if (tid < EXACT_SPEC_MAX_WIDTH) {
        s_prefix_tokens[tid] = EXACT_SPEC_INVALID_TOKEN;
    }
    if (tid == 0u) {
        s_status = EXACT_SPEC_STATUS_OK;
        s_active_k = 0u;
        s_p_draft_x = 0.0f;
        s_accepted = 0u;
        s_sampled_tok = EXACT_SPEC_INVALID_TOKEN;
        s_found = 0u;

        uint32_t r0 = 0u;
        uint32_t r1 = 0u;
        philox4x32_10(seeds[seq], base_abs_pos[seq] + (uint64_t)col, r0, r1);
        s_u_accept = u32_to_uniform_f32(r0);
        s_u_residual = u32_to_uniform_f32(r1);
    }
    __syncthreads();

    // Validate & compact sparse draft support `p_sparse` on draft rows (K <= 64).
    if (is_draft_row && tid == 0u) {
        uint32_t st = EXACT_SPEC_STATUS_OK;
        if (draft_tok >= vocab_size) {
            st |= EXACT_SPEC_STATUS_DRAFT_OOB;
        }
        const size_t q_base = (size_t)row * (size_t)sparse_k;
        float sum_p = 0.0f;
        uint32_t active = 0u;
        float p_x = 0.0f;
        for (uint32_t j = 0u; j < sparse_k; ++j) {
            const uint32_t id = sparse_draft_ids[q_base + (size_t)j];
            const float pr = sparse_draft_probs[q_base + (size_t)j];
            if (id == EXACT_SPEC_INVALID_TOKEN) {
                if (!isfinite(pr) || pr != 0.0f) {
                    st |= EXACT_SPEC_STATUS_DRAFT_NONFINITE;
                }
                continue;
            }
            if (id >= vocab_size || !isfinite(pr) || pr < 0.0f ||
                pr > (1.0f + EXACT_SPEC_PROB_SUM_TOL)) {
                st |= EXACT_SPEC_STATUS_DRAFT_NONFINITE;
                continue;
            }
            bool dup = false;
            for (uint32_t prev = 0u; prev < active; ++prev) {
                if (s_p_ids[prev] == id) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                st |= EXACT_SPEC_STATUS_DRAFT_NONFINITE;
                continue;
            }
            s_p_ids[active] = id;
            s_p_probs[active] = pr;
            s_bloom[(id >> 5) & 31u] |= (1u << (id & 31u));
            active += 1u;
            sum_p = __fadd_rn(sum_p, pr);
            if (id == draft_tok) {
                p_x = pr;
            }
        }
        if (!isfinite(sum_p) || fabsf(__fsub_rn(sum_p, 1.0f)) > EXACT_SPEC_PROB_SUM_TOL) {
            st |= EXACT_SPEC_STATUS_DRAFT_SUM_ERR;
        }
        if (!(p_x > 0.0f)) {
            st |= EXACT_SPEC_STATUS_DRAFT_OOB;
        }
        s_status = st;
        s_active_k = active;
        s_p_draft_x = p_x;
    }
    __syncthreads();

    if ((s_status & EXACT_SPEC_STATUS_ERROR_MASK) != 0u) {
        if (tid == 0u) {
            out_accepted[row] = 0u;
            out_tokens[row] = EXACT_SPEC_INVALID_TOKEN;
            out_status[row] = s_status;
        }
        return;
    }

    const __nv_bfloat16* __restrict__ row_logits = logits + (size_t)row * (size_t)vocab_size;
    const uint32_t* __restrict__ seq_pre_counts =
        (pre_verify_counts != nullptr)
            ? (pre_verify_counts + (size_t)seq * (size_t)vocab_size)
            : nullptr;

    // Pass 1: Max scaled logit & deterministic lowest-ID target-support argmax.
    // Any NaN, +Inf, or post-scale non-finite logit fails closed with TARGET_NONFINITE.
    float local_max = -INFINITY;
    uint32_t local_argmax = EXACT_SPEC_INVALID_TOKEN;
    bool local_fatal_nonfinite = false;
    for (uint32_t v = tid; v < vocab_size; v += 256u) {
        bool valid = false;
        bool fatal = false;
        const float sc = compute_scaled_logit(
            row_logits[v],
            v,
            seq_pre_counts,
            s_prefix_tokens,
            col,
            pres_pen,
            freq_pen,
            temp,
            valid,
            fatal
        );
        if (fatal) {
            local_fatal_nonfinite = true;
        } else if (valid &&
                   (local_argmax == EXACT_SPEC_INVALID_TOKEN || sc > local_max ||
                    (sc == local_max && v < local_argmax))) {
            local_max = sc;
            local_argmax = v;
        }
    }
    if (local_fatal_nonfinite) {
        atomicOr(&s_status, EXACT_SPEC_STATUS_TARGET_NONFINITE);
    }
    float max_sc = -INFINITY;
    uint32_t argmax_id = EXACT_SPEC_INVALID_TOKEN;
    block_reduce_argmax_256(local_max, local_argmax, s_warp_f, s_warp_u, warp, lane, max_sc, argmax_id);
    if (tid == 0u) {
        s_max_scaled = max_sc;
        s_argmax_id = argmax_id;
    }
    __syncthreads();

    if ((s_status & EXACT_SPEC_STATUS_ERROR_MASK) != 0u ||
        s_argmax_id == EXACT_SPEC_INVALID_TOKEN || !isfinite(s_max_scaled)) {
        if (tid == 0u) {
            out_accepted[row] = 0u;
            out_tokens[row] = EXACT_SPEC_INVALID_TOKEN;
            out_status[row] = s_status | EXACT_SPEC_STATUS_TARGET_NONFINITE;
        }
        return;
    }

    // Pass 2a: Exact softmax denominator `Z = sum_v expf(scaled(v) - max_scaled)`.
    float local_exp_sum = 0.0f;
    for (uint32_t v = tid; v < vocab_size; v += 256u) {
        bool valid = false;
        bool fatal = false;
        const float sc = compute_scaled_logit(
            row_logits[v],
            v,
            seq_pre_counts,
            s_prefix_tokens,
            col,
            pres_pen,
            freq_pen,
            temp,
            valid,
            fatal
        );
        if (valid && !fatal) {
            local_exp_sum = __fadd_rn(local_exp_sum, expf(__fsub_rn(sc, s_max_scaled)));
        }
    }
    const float sum_exp = block_reduce_sum_256(local_exp_sum, s_warp_f, warp, lane);
    if (tid == 0u) {
        s_sum_exp = sum_exp;
    }
    __syncthreads();

    if (!isfinite(s_sum_exp) || !(s_sum_exp > 0.0f)) {
        if (tid == 0u) {
            out_accepted[row] = 0u;
            out_tokens[row] = EXACT_SPEC_INVALID_TOKEN;
            out_status[row] = s_status | EXACT_SPEC_STATUS_TARGET_NONFINITE;
        }
        return;
    }

    // Pass 2b (Pass 1 of Two-Pass Sparse Residual):
    // Verify `|sum(q) - 1.0f| <= 1e-4f` fail-closed AND compute exact residual
    // mass `Z_r = sum_v max(q(v) - p_sparse(v), 0.0f)` without cancellation.
    const uint32_t active_k = s_active_k;
    float local_sum_q = 0.0f;
    float local_sum_r = 0.0f;
    for (uint32_t v = tid; v < vocab_size; v += 256u) {
        bool valid = false;
        bool fatal = false;
        const float sc = compute_scaled_logit(
            row_logits[v],
            v,
            seq_pre_counts,
            s_prefix_tokens,
            col,
            pres_pen,
            freq_pen,
            temp,
            valid,
            fatal
        );
        if (valid && !fatal) {
            const float q_v = __fdiv_rn(expf(__fsub_rn(sc, s_max_scaled)), s_sum_exp);
            local_sum_q = __fadd_rn(local_sum_q, q_v);
            if (is_draft_row) {
                const float p_v = lookup_sparse_prob(v, s_bloom, s_p_ids, s_p_probs, active_k);
                const float diff = __fsub_rn(q_v, p_v);
                if (diff > 0.0f) {
                    local_sum_r = __fadd_rn(local_sum_r, diff);
                }
            }
        }
    }
    const float sum_q = block_reduce_sum_256(local_sum_q, s_warp_f, warp, lane);
    const float sum_r = is_draft_row
        ? block_reduce_sum_256(local_sum_r, s_warp_f, warp, lane)
        : 0.0f;
    if (tid == 0u) {
        s_sum_q = sum_q;
        s_sum_r = sum_r;
        if (!isfinite(sum_q) || fabsf(__fsub_rn(sum_q, 1.0f)) > EXACT_SPEC_PROB_SUM_TOL) {
            s_status |= EXACT_SPEC_STATUS_TARGET_SUM_ERR;
        }
    }
    __syncthreads();

    if ((s_status & EXACT_SPEC_STATUS_ERROR_MASK) != 0u) {
        if (tid == 0u) {
            out_accepted[row] = 0u;
            out_tokens[row] = EXACT_SPEC_INVALID_TOKEN;
            out_status[row] = s_status;
        }
        return;
    }

    // Acceptance test on draft rows: accept iff `q(x) >= p_sparse(x)` or
    // `u_accept * p_sparse(x) < q(x)` (with `p_sparse(x) > 0` already verified).
    if (is_draft_row && tid == 0u) {
        bool valid_x = false;
        bool fatal_x = false;
        const float sc_x = compute_scaled_logit(
            row_logits[draft_tok],
            draft_tok,
            seq_pre_counts,
            s_prefix_tokens,
            col,
            pres_pen,
            freq_pen,
            temp,
            valid_x,
            fatal_x
        );
        const float q_x = (valid_x && !fatal_x)
            ? __fdiv_rn(expf(__fsub_rn(sc_x, s_max_scaled)), s_sum_exp)
            : 0.0f;
        if (s_p_draft_x > 0.0f && q_x > 0.0f &&
            (q_x >= s_p_draft_x || __fmul_rn(s_u_accept, s_p_draft_x) < q_x)) {
            s_accepted = 1u;
            s_sampled_tok = draft_tok;
            s_found = 1u;
        }
    }
    __syncthreads();

    if (s_accepted != 0u) {
        if (tid == 0u) {
            out_accepted[row] = 1u;
            out_tokens[row] = draft_tok;
            out_status[row] = s_status;
        }
        return;
    }

    // Pass 3 (Pass 2 of Two-Pass Sparse Residual, or Bonus Sampling from `q`):
    // Coalesced 256-lane ascending vocab CDF scan.
    const float mass_total = is_draft_row ? s_sum_r : s_sum_q;
    const float u_draw = is_draft_row ? s_u_residual : s_u_accept;

    if (isfinite(mass_total) && mass_total > 0.0f) {
        const float thresh = __fmul_rn(u_draw, mass_total);
        float running_cdf = 0.0f;
        for (uint32_t base = 0u; base < vocab_size; base += 256u) {
            const uint32_t v = base + tid;
            float w_v = 0.0f;
            if (v < vocab_size) {
                bool valid = false;
                bool fatal = false;
                const float sc = compute_scaled_logit(
                    row_logits[v],
                    v,
                    seq_pre_counts,
                    s_prefix_tokens,
                    col,
                    pres_pen,
                    freq_pen,
                    temp,
                    valid,
                    fatal
                );
                if (valid && !fatal) {
                    const float q_v = __fdiv_rn(expf(__fsub_rn(sc, s_max_scaled)), s_sum_exp);
                    if (is_draft_row) {
                        const float p_v = lookup_sparse_prob(v, s_bloom, s_p_ids, s_p_probs, active_k);
                        const float diff = __fsub_rn(q_v, p_v);
                        w_v = (diff > 0.0f) ? diff : 0.0f;
                    } else {
                        w_v = q_v;
                    }
                }
            }
            float tile_total = 0.0f;
            const float incl = tile_inclusive_scan_256(w_v, s_warp_f, warp, lane, tile_total);
            const float next_cdf = __fadd_rn(running_cdf, tile_total);
            if (next_cdf > thresh) {
                const bool hit = (v < vocab_size) && (w_v > 0.0f) &&
                                 (__fadd_rn(running_cdf, incl) > thresh);
                const uint32_t mask = __ballot_sync(0xFFFFFFFFu, hit);
                if (lane == 0u) {
                    s_warp_u[warp] = (mask != 0u) ? ((warp << 5) + (uint32_t)(__ffs((int)mask) - 1))
                                                  : EXACT_SPEC_INVALID_TOKEN;
                }
                __syncthreads();
                if (tid == 0u) {
                    uint32_t win = EXACT_SPEC_INVALID_TOKEN;
                    for (uint32_t w = 0u; w < 8u; ++w) {
                        if (s_warp_u[w] != EXACT_SPEC_INVALID_TOKEN) {
                            win = s_warp_u[w];
                            break;
                        }
                    }
                    if (win != EXACT_SPEC_INVALID_TOKEN && (base + win) < vocab_size) {
                        s_sampled_tok = base + win;
                        s_found = 1u;
                    }
                }
                __syncthreads();
                if (s_found != 0u) {
                    break;
                }
            }
            running_cdf = next_cdf;
        }
    }

    // Guaranteed valid target-support zero-mass / tail-rounding fallback:
    // NEVER emits `-1` (`EXACT_SPEC_INVALID_TOKEN`) or OOB on valid target `q`.
    if (tid == 0u) {
        uint32_t final_tok = s_sampled_tok;
        uint32_t final_status = s_status;
        if (s_found == 0u || final_tok >= vocab_size) {
            final_tok = s_argmax_id;
            final_status |= EXACT_SPEC_STATUS_FALLBACK_USED;
        }
        out_accepted[row] = 0u;
        out_tokens[row] = final_tok;
        out_status[row] = final_status;
    }
}

extern "C" __global__ __launch_bounds__(32, 4) void exact_spec_prefix_reduce_sm120(
    const uint32_t* __restrict__ row_accepted,         // [batch_size * verify_width]
    const uint32_t* __restrict__ row_tokens,           // [batch_size * verify_width]
    const uint32_t* __restrict__ row_status,           // [batch_size * verify_width]
    const uint32_t* __restrict__ slot_ids,             // [batch_size]
    const uint64_t* __restrict__ slot_generations,     // [batch_size]
    const uint64_t* __restrict__ expected_generations, // [batch_size]
    const uint32_t* __restrict__ slot_refcounts,       // [batch_size] (must be == 1)
    uint32_t* __restrict__ d_num_accepted,             // [batch_size]
    uint32_t* __restrict__ d_bonus_token,              // [batch_size]
    uint32_t* __restrict__ d_status_out,               // [batch_size]
    uint32_t batch_size,
    uint32_t verify_width
) {
    const uint32_t seq = blockIdx.x;
    if (seq >= batch_size || threadIdx.x != 0u) {
        return;
    }
    if (d_num_accepted == nullptr || d_bonus_token == nullptr || d_status_out == nullptr) {
        return;
    }

    uint32_t status = EXACT_SPEC_STATUS_OK;
    if (verify_width == 0u || verify_width > EXACT_SPEC_MAX_WIDTH ||
        row_accepted == nullptr || row_tokens == nullptr || row_status == nullptr ||
        slot_ids == nullptr || slot_generations == nullptr ||
        expected_generations == nullptr || slot_refcounts == nullptr) {
        d_num_accepted[seq] = 0u;
        d_bonus_token[seq] = EXACT_SPEC_INVALID_TOKEN;
        d_status_out[seq] = EXACT_SPEC_STATUS_BAD_PARAM;
        return;
    }

    // Fail-closed Generation and sharing <= 1 (owned refcount == 1 + no
    // duplicate physical slot_id across active batch lanes) guard.
    if (slot_generations[seq] != expected_generations[seq]) {
        status |= EXACT_SPEC_STATUS_GEN_MISMATCH;
    }
    if (slot_refcounts[seq] != 1u) {
        status |= EXACT_SPEC_STATUS_SLOT_SHARED;
    }
    const uint32_t my_slot = slot_ids[seq];
    for (uint32_t other = 0u; other < batch_size; ++other) {
        if (other != seq && slot_ids[other] == my_slot) {
            status |= EXACT_SPEC_STATUS_SLOT_SHARED;
            break;
        }
    }

    if ((status & EXACT_SPEC_STATUS_ERROR_MASK) != 0u) {
        d_num_accepted[seq] = 0u;
        d_bonus_token[seq] = EXACT_SPEC_INVALID_TOKEN;
        d_status_out[seq] = status;
        return;
    }

    const uint32_t base = seq * verify_width;
    const uint32_t num_drafts = verify_width - 1u;
    uint32_t num_accepted = 0u;
    uint32_t bonus_token = EXACT_SPEC_INVALID_TOKEN;
    bool stopped = false;

    for (uint32_t col = 0u; col < num_drafts; ++col) {
        const uint32_t st = row_status[base + col];
        status |= st;
        if ((st & EXACT_SPEC_STATUS_ERROR_MASK) != 0u) {
            d_num_accepted[seq] = 0u;
            d_bonus_token[seq] = EXACT_SPEC_INVALID_TOKEN;
            d_status_out[seq] = status;
            return;
        }
        if (row_accepted[base + col] != 0u) {
            num_accepted += 1u;
        } else {
            bonus_token = row_tokens[base + col];
            stopped = true;
            break;
        }
    }

    if (!stopped) {
        const uint32_t st = row_status[base + num_drafts];
        status |= st;
        if ((st & EXACT_SPEC_STATUS_ERROR_MASK) != 0u) {
            d_num_accepted[seq] = 0u;
            d_bonus_token[seq] = EXACT_SPEC_INVALID_TOKEN;
            d_status_out[seq] = status;
            return;
        }
        bonus_token = row_tokens[base + num_drafts];
    }

    if (bonus_token == EXACT_SPEC_INVALID_TOKEN) {
        d_num_accepted[seq] = 0u;
        d_bonus_token[seq] = EXACT_SPEC_INVALID_TOKEN;
        d_status_out[seq] = status | EXACT_SPEC_STATUS_BAD_PARAM;
        return;
    }

    d_num_accepted[seq] = num_accepted;
    d_bonus_token[seq] = bonus_token;
    d_status_out[seq] = status;
}
