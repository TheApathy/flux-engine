// SPDX-License-Identifier: AGPL-3.0-only
//
// Flux-DCOCO-v2 (Contextual Correction of Parallel Drafts) Blackwell Kernels.
//
// Clean-room implementation of the three GPU-resident stages for parallel
// candidate-grid contextual scoring and Wave-15 maximal-coupling walk:
//
//   1. `dcoco_proj_fuse_bf16`:
//      Reads frozen target `embed_tokens` E[c_ik] and drafter post-norm hidden
//      states h_i (including anchor slot 0), applies non-affine LayerNorm to
//      `W_cand E[c_ik]` and `W_ctx h_i`, fuses through `W_fuse` + affine
//      LayerNorm into `z_out[B, gamma+1, K, 128]`, and computes adjacent pair
//      states `s_out[B, gamma, 128] = W_pair [h_p; h_{p+1}]`.
//
//   2. `dcoco_pair_mlp_bf16`:
//      8.25 KiB input SMEM-resident fused 2-layer GELU MLP (`386 -> 256 -> 1`)
//      with zero intermediate global `[B*gamma*K^2, 386]` materialization.
//      Exploits the exact algebraic factorization of layer 1 across the K x K
//      pair grid:
//        pre[a, b, h] = (b1[h] + W1_s[h] . s_p)
//                     + (W1_a[h] . z_{p,a} + w1_la[h] * ell_p(a))
//                     + (W1_b[h] . z_{p+1,b} + w1_lb[h] * ell_{p+1}(b))
//      computing the K x 128 x 256 projections via native Blackwell/Ampere
//      `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` Tensor Cores in
//      32-channel hidden tiles (total static SMEM ~12.6 KiB, 0 B stack/spills
//      across sm_121a, sm_120a, and sm_103a).
//
//   3. `dcoco_walk_and_export_q`:
//      Single-warp CTA (`Block = 32`) prefix-first walk over precomputed
//      `S[B, gamma, K, K]` supporting:
//        - mode 0: exact Greedy argmax (+ 1-hot q export if pointers non-null)
//        - mode 1: CRN SplitMix64 Gumbel-Max sampling
//        - mode 2: Wave-15 Maximal-Coupling rejection sampling exporting
//          normalized conditional `q_t(. | k_p)` and F_t-measurable
//          `out_max_prob[t] = max_z q_t(z | x_{<t})` with continuous `beta`
//          temperature scaling (`beta -> 0+` cleanly reduces to point-mass).

#include <cuda_bf16.h>
#include <math.h>

#define DCOCO_RANK       128
#define DCOCO_MLP_IN     386
#define DCOCO_MLP_HIDDEN 256
#define DCOCO_MAX_K      16
#define DCOCO_WALK_MAX_K 32
#define DCOCO_H_TILE     32
#define DCOCO_THREADS    256

// ── Shared Register-Only Helpers (0 B .local stack, 0 spills) ────────────────

__device__ __forceinline__ unsigned long long dcoco_splitmix64_mix(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ unsigned long long dcoco_pos_base(
    unsigned long long seed, unsigned long long abs_pos
) {
    return dcoco_splitmix64_mix(
        dcoco_splitmix64_mix(seed + 0x9E3779B97F4A7C15ull) ^ (abs_pos * 0xD1B54A32D192ED03ull)
    );
}

__device__ __forceinline__ float dcoco_gumbel_from_base(
    unsigned long long pos_base, unsigned int token_id
) {
    const unsigned long long bits = dcoco_splitmix64_mix(pos_base ^ (unsigned long long)token_id);
    const float u = fminf((float)(bits >> 40) * 0x1.0p-24f + 0x1.0p-25f, 0x1.fffffep-1f);
    return -logf(-logf(u));
}

__device__ __forceinline__ float dcoco_prob_uniform(
    unsigned long long seed, unsigned long long abs_pos
) {
    const unsigned long long pos_base = dcoco_pos_base(seed, abs_pos);
    const unsigned long long bits = dcoco_splitmix64_mix(pos_base ^ 0xDF2D5A31ull);
    return fminf((float)(bits >> 40) * 0x1.0p-24f, 0x1.fffffep-1f);
}

// Register-only exact erf-GELU via Abramowitz & Stegun 7.1.26 (max |err| < 1.5e-7
// across all finite FP32 inputs, guaranteeing 0 bytes .local stack on sm_121a/120a/103a).
__device__ __forceinline__ float dcoco_gelu(float x) {
    const float z = x * 0.7071067811865475244f;
    const float az = fabsf(z);
    const float t = __fdiv_rn(1.0f, 1.0f + 0.3275911f * az);
    const float poly =
        ((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t
        + 0.254829592f;
    const float erf_abs = 1.0f - (poly * t) * expf(-(az * az));
    const float erf_val = (z < 0.0f) ? -erf_abs : erf_abs;
    return 0.5f * x * (1.0f + erf_val);
}

__device__ __forceinline__ float dcoco_warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

// Parallel 128-element LayerNorm in shared memory (4 warps: threads 0..127 active).
// Non-affine when `gamma == nullptr || beta == nullptr`.
__device__ __forceinline__ void dcoco_layernorm_128_smem(
    float* s_vec,
    float* s_red,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    float eps,
    unsigned int tid
) {
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;

    float v = 0.0f;
    if (tid < DCOCO_RANK) {
        v = s_vec[tid];
        const float wsum = dcoco_warp_sum(v);
        if (lane == 0u) {
            s_red[warp] = wsum;
        }
    }
    __syncthreads();

    float mean = 0.0f;
    if (tid < DCOCO_RANK) {
        mean = (s_red[0] + s_red[1] + s_red[2] + s_red[3]) * (1.0f / (float)DCOCO_RANK);
        const float diff = v - mean;
        const float wvar = dcoco_warp_sum(diff * diff);
        if (lane == 0u) {
            s_red[4u + warp] = wvar;
        }
    }
    __syncthreads();

    if (tid < DCOCO_RANK) {
        const float var =
            (s_red[4] + s_red[5] + s_red[6] + s_red[7]) * (1.0f / (float)DCOCO_RANK);
        const float norm = (v - mean) * rsqrtf(var + eps);
        if (gamma != nullptr && beta != nullptr) {
            s_vec[tid] = norm * __bfloat162float(gamma[tid]) + __bfloat162float(beta[tid]);
        } else {
            s_vec[tid] = norm;
        }
    }
    __syncthreads();
}

// Pack two BF16 elements (lo, hi) from an arbitrary BF16 pointer into one u32
// without assuming 4-byte alignment of `base + idx`.
__device__ __forceinline__ unsigned int dcoco_pack_bf16x2(
    const __nv_bfloat16* base, size_t idx
) {
    const unsigned short lo = __bfloat16_as_ushort(base[idx]);
    const unsigned short hi = __bfloat16_as_ushort(base[idx + 1]);
    return ((unsigned int)hi << 16) | (unsigned int)lo;
}

// ── Kernel 1: Fused Candidate/Context Projection + Pair State ────────────────
//
// Grid:  [batch_size * (gamma + 1), 1, 1]
// Block: [256, 1, 1] (8 warps)
//
// Slot layout per batch item `b`:
//   slot == 0         : verified anchor (`anchor_ids[b]`, 1 candidate at k=0)
//   slot == 1..gamma  : drafted slot `t = slot - 1` (`top_k` candidates)
//
// Outputs:
//   z_out: [batch_size, gamma + 1, top_k, 128] BF16 (slot 0 populates k=0)
//   s_out: [batch_size, gamma, 128] BF16 (emitted by slots 0..gamma-1)
extern "C" __global__ void __launch_bounds__(DCOCO_THREADS) dcoco_proj_fuse_bf16(
    const __nv_bfloat16* __restrict__ embed_tokens,  // [vocab, hidden_dim]
    const __nv_bfloat16* __restrict__ hidden_states, // [batch_size, gamma + 1, hidden_dim]
    const unsigned int* __restrict__ anchor_ids,     // [batch_size]
    const unsigned int* __restrict__ candidates,     // [batch_size, gamma, top_k]
    const __nv_bfloat16* __restrict__ w_cand,        // [128, hidden_dim]
    const __nv_bfloat16* __restrict__ w_ctx,         // [128, hidden_dim]
    const __nv_bfloat16* __restrict__ w_fuse,        // [128, 256]
    const __nv_bfloat16* __restrict__ fuse_ln_w,     // [128]
    const __nv_bfloat16* __restrict__ fuse_ln_b,     // [128]
    const __nv_bfloat16* __restrict__ w_pair,        // [128, 2 * hidden_dim]
    __nv_bfloat16* __restrict__ z_out,               // [batch_size, gamma + 1, top_k, 128]
    __nv_bfloat16* __restrict__ s_out,               // [batch_size, gamma, 128]
    float ln_eps,
    int batch_size,
    int gamma,
    int top_k,
    int hidden_dim,
    int vocab_size
) {
    const int slots_per_seq = gamma + 1;
    const int slot_global = (int)blockIdx.x;
    if (batch_size <= 0 || gamma <= 0 || top_k <= 0 || top_k > DCOCO_MAX_K
        || hidden_dim <= 0 || vocab_size <= 0
        || slot_global >= batch_size * slots_per_seq) {
        return;
    }

    const int b = slot_global / slots_per_seq;
    const int slot = slot_global - b * slots_per_seq;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;

    __shared__ float s_g[DCOCO_RANK];
    __shared__ float s_b[DCOCO_RANK];
    __shared__ float s_u[DCOCO_RANK];
    __shared__ float s_red[8];

    const __nv_bfloat16* h_cur = hidden_states + (size_t)slot_global * (size_t)hidden_dim;

    // 1. g_i = LN_no_affine(W_ctx @ h_i)
    for (int r = (int)warp; r < DCOCO_RANK; r += 8) {
        const __nv_bfloat16* w_row = w_ctx + (size_t)r * (size_t)hidden_dim;
        float acc = 0.0f;
        for (int j = (int)lane; j < hidden_dim; j += 32) {
            acc += __bfloat162float(w_row[j]) * __bfloat162float(h_cur[j]);
        }
        const float sum = dcoco_warp_sum(acc);
        if (lane == 0u) {
            s_g[r] = sum;
        }
    }
    __syncthreads();
    dcoco_layernorm_128_smem(s_g, s_red, nullptr, nullptr, ln_eps, tid);

    // 2. For each candidate k at this slot:
    //    b_ik = LN_no_affine(W_cand @ E[c_ik])
    //    z_ik = LN_{gamma,beta}(W_fuse @ [b_ik; g_i])
    const int slot_k = (slot == 0) ? 1 : top_k;
    for (int k = 0; k < slot_k; ++k) {
        unsigned int cid = (slot == 0)
            ? anchor_ids[b]
            : candidates[((size_t)b * (size_t)gamma + (size_t)(slot - 1)) * (size_t)top_k + (size_t)k];
        if (cid >= (unsigned int)vocab_size) {
            cid = 0u;
        }
        const __nv_bfloat16* emb_row = embed_tokens + (size_t)cid * (size_t)hidden_dim;

        for (int r = (int)warp; r < DCOCO_RANK; r += 8) {
            const __nv_bfloat16* w_row = w_cand + (size_t)r * (size_t)hidden_dim;
            float acc = 0.0f;
            for (int j = (int)lane; j < hidden_dim; j += 32) {
                acc += __bfloat162float(w_row[j]) * __bfloat162float(emb_row[j]);
            }
            const float sum = dcoco_warp_sum(acc);
            if (lane == 0u) {
                s_b[r] = sum;
            }
        }
        __syncthreads();
        dcoco_layernorm_128_smem(s_b, s_red, nullptr, nullptr, ln_eps, tid);

        for (int r = (int)warp; r < DCOCO_RANK; r += 8) {
            const __nv_bfloat16* wf_row = w_fuse + (size_t)r * (size_t)(2 * DCOCO_RANK);
            float acc = 0.0f;
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int j = (int)lane + u * 32;
                acc += __bfloat162float(wf_row[j]) * s_b[j];
                acc += __bfloat162float(wf_row[DCOCO_RANK + j]) * s_g[j];
            }
            const float sum = dcoco_warp_sum(acc);
            if (lane == 0u) {
                s_u[r] = sum;
            }
        }
        __syncthreads();
        dcoco_layernorm_128_smem(s_u, s_red, fuse_ln_w, fuse_ln_b, ln_eps, tid);

        if (tid < DCOCO_RANK) {
            const size_t z_off =
                ((size_t)slot_global * (size_t)top_k + (size_t)k) * (size_t)DCOCO_RANK + tid;
            z_out[z_off] = __float2bfloat16_rn(s_u[tid]);
        }
        __syncthreads();
    }

    // 3. Pair state for adjacent transition (slot -> slot + 1):
    //    s_p = W_pair @ [h_slot; h_{slot+1}]
    if (slot < gamma) {
        const __nv_bfloat16* h_next =
            hidden_states + (size_t)(slot_global + 1) * (size_t)hidden_dim;
        const size_t pair_in_dim = (size_t)2 * (size_t)hidden_dim;
        __nv_bfloat16* s_dst =
            s_out + ((size_t)b * (size_t)gamma + (size_t)slot) * (size_t)DCOCO_RANK;

        for (int r = (int)warp; r < DCOCO_RANK; r += 8) {
            const __nv_bfloat16* wp_row = w_pair + (size_t)r * pair_in_dim;
            float acc = 0.0f;
            for (int j = (int)lane; j < hidden_dim; j += 32) {
                acc += __bfloat162float(wp_row[j]) * __bfloat162float(h_cur[j]);
                acc += __bfloat162float(wp_row[hidden_dim + j]) * __bfloat162float(h_next[j]);
            }
            const float sum = dcoco_warp_sum(acc);
            if (lane == 0u) {
                s_dst[r] = __float2bfloat16_rn(sum);
            }
        }
    }
}

// ── Kernel 2: SMEM-Resident Fused Blackwell Tensor-Core Pair MLP ─────────────
//
// Grid:  [gamma, batch_size, 1]
// Block: [256, 1, 1] (8 warps; thread tid maps to pair (a = tid >> 4, b = tid & 15))
//
// Computes all K x K transition scores for `(b, p)` with ZERO global
// materialization of the `[B*gamma*K^2, 386]` MLP input tensor:
//   S_p(a, b) = ell_{p+1}(b) + b2 + sum_{h=0..255} W2[h] * GELU(
//       b1[h] + W1_s[h].s_p
//       + (W1_a[h].z_{p,a} + w1_la[h]*ell_p(a))
//       + (W1_b[h].z_{p+1,b} + w1_lb[h]*ell_{p+1}(b))
//   )
//
// Input SMEM: s_sp (256 B) + s_za (4 KiB) + s_zb (4 KiB) = 8.25 KiB.
// Tile SMEM:  s_ta (2 KiB) + s_tb (2 KiB) + s_base/s_w2/s_la/s_lb (384 B).
// Total static SMEM = 12.6 KiB (0 B stack frame, 0 B spills on sm_121a/120a/103a).
extern "C" __global__ void __launch_bounds__(DCOCO_THREADS) dcoco_pair_mlp_bf16(
    const __nv_bfloat16* __restrict__ s_pair,        // [batch_size, gamma, 128]
    const __nv_bfloat16* __restrict__ z_cand,        // [batch_size, gamma + 1, top_k, 128]
    const float* __restrict__ unary,                 // [batch_size, gamma, top_k]
    const float* __restrict__ anchor_logits,         // [batch_size] optional (nullptr -> 0.0f)
    const __nv_bfloat16* __restrict__ mlp_w1,        // [256, 386]
    const __nv_bfloat16* __restrict__ mlp_b1,        // [256]
    const __nv_bfloat16* __restrict__ mlp_w2,        // [256]
    const __nv_bfloat16* __restrict__ mlp_b2,        // [1]
    float* __restrict__ out_scores,                  // [batch_size, gamma, top_k, top_k]
    int batch_size,
    int gamma,
    int top_k
) {
    const int p = (int)blockIdx.x;
    const int b = (int)blockIdx.y;
    if (batch_size <= 0 || gamma <= 0 || top_k <= 0 || top_k > DCOCO_MAX_K
        || p >= gamma || b >= batch_size) {
        return;
    }

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int group_id = lane >> 2;
    const unsigned int tid_in_group = lane & 3u;

    // 8.25 KiB input staging (16-byte aligned for alignment-safe 32-bit LDS loads)
    __shared__ __align__(16) __nv_bfloat16 s_sp[DCOCO_RANK];
    __shared__ __align__(16) __nv_bfloat16 s_za[DCOCO_MAX_K][DCOCO_RANK];
    __shared__ __align__(16) __nv_bfloat16 s_zb[DCOCO_MAX_K][DCOCO_RANK];
    __shared__ float s_la[DCOCO_MAX_K];
    __shared__ float s_lb[DCOCO_MAX_K];

    // 4.25 KiB hidden-tile accumulators (H_TILE = 32)
    __shared__ float s_base[DCOCO_H_TILE];
    __shared__ float s_ta[DCOCO_MAX_K][DCOCO_H_TILE];
    __shared__ float s_tb[DCOCO_MAX_K][DCOCO_H_TILE];
    __shared__ float s_w2_tile[DCOCO_H_TILE];

    const int slots_per_seq = gamma + 1;
    const int k_prev = (p == 0) ? 1 : top_k;

    // Load s_p [128]
    if (tid < DCOCO_RANK) {
        s_sp[tid] = s_pair[((size_t)b * (size_t)gamma + (size_t)p) * (size_t)DCOCO_RANK + tid];
    }

    // Load s_za [16, 128] and s_zb [16, 128] (zero-pad rows >= k_prev / top_k)
    const size_t z_base_a =
        ((size_t)b * (size_t)slots_per_seq + (size_t)p) * (size_t)top_k * (size_t)DCOCO_RANK;
    const size_t z_base_b =
        ((size_t)b * (size_t)slots_per_seq + (size_t)(p + 1)) * (size_t)top_k * (size_t)DCOCO_RANK;

    for (int idx = (int)tid; idx < DCOCO_MAX_K * DCOCO_RANK; idx += DCOCO_THREADS) {
        const int row = idx >> 7;
        const int col = idx & 127;
        s_za[row][col] = (row < k_prev)
            ? z_cand[z_base_a + (size_t)row * (size_t)DCOCO_RANK + (size_t)col]
            : __float2bfloat16(0.0f);
        s_zb[row][col] = (row < top_k)
            ? z_cand[z_base_b + (size_t)row * (size_t)DCOCO_RANK + (size_t)col]
            : __float2bfloat16(0.0f);
    }

    // Load scalar logits ell_p(a) and ell_{p+1}(b)
    if (tid < DCOCO_MAX_K) {
        if ((int)tid < k_prev) {
            if (p == 0) {
                s_la[tid] = (anchor_logits != nullptr) ? anchor_logits[b] : 0.0f;
            } else {
                s_la[tid] =
                    unary[((size_t)b * (size_t)gamma + (size_t)(p - 1)) * (size_t)top_k + tid];
            }
        } else {
            s_la[tid] = 0.0f;
        }
        s_lb[tid] = ((int)tid < top_k)
            ? unary[((size_t)b * (size_t)gamma + (size_t)p) * (size_t)top_k + tid]
            : 0.0f;
    }
    __syncthreads();

    const unsigned short* s_za_u16 = reinterpret_cast<const unsigned short*>(&s_za[0][0]);
    const unsigned short* s_zb_u16 = reinterpret_cast<const unsigned short*>(&s_zb[0][0]);

    const unsigned int pair_a = tid >> 4;
    const unsigned int pair_b = tid & 15u;
    float acc_out = 0.0f;

    // Tile across all 256 hidden units in 8 passes of 32 channels.
    // Within each pass:
    //   - All 8 warps compute s_base[0..31] (4 channels/warp) + load s_w2_tile[0..31]
    //   - Warps 0..3 execute 8x m16n8k16 Tensor-Core MMAs for s_ta[0..15][0..31]
    //   - Warps 4..7 execute 8x m16n8k16 Tensor-Core MMAs for s_tb[0..15][0..31]
    //   - All 256 threads accumulate their assigned (a, b) GELU * W2 contribution.
    for (int h_base = 0; h_base < DCOCO_MLP_HIDDEN; h_base += DCOCO_H_TILE) {
        // 1. Pair-state base vector: b1[h] + W1_s[h] . s_p (4 hidden rows per warp)
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int h_local = (int)warp * 4 + c;
            const int h = h_base + h_local;
            const __nv_bfloat16* w1_row = mlp_w1 + (size_t)h * (size_t)DCOCO_MLP_IN;
            float dot = 0.0f;
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int j = (int)lane + u * 32;
                dot += __bfloat162float(w1_row[j]) * __bfloat162float(s_sp[j]);
            }
            const float sum = dcoco_warp_sum(dot);
            if (lane == 0u) {
                s_base[h_local] = __bfloat162float(mlp_b1[h]) + sum;
                s_w2_tile[h_local] = __bfloat162float(mlp_w2[h]);
            }
        }

        // 2. Native BF16 Tensor-Core MMA: m16n8k16 over K=128 (8 k-steps)
        //    Warps 0..3 own term_a (col offset 128); warps 4..7 own term_b (col offset 256).
        const bool is_b_side = (warp >= 4u);
        const unsigned int n_slice = warp & 3u;
        const int n_row = h_base + (int)(n_slice * 8u + group_id);
        const unsigned short* s_z_u16 = is_b_side ? s_zb_u16 : s_za_u16;
        const size_t w_col_base = is_b_side ? (size_t)(2 * DCOCO_RANK) : (size_t)DCOCO_RANK;
        const __nv_bfloat16* w1_n = mlp_w1 + (size_t)n_row * (size_t)DCOCO_MLP_IN + w_col_base;

        float d0 = 0.0f;
        float d1 = 0.0f;
        float d2 = 0.0f;
        float d3 = 0.0f;

        #pragma unroll
        for (int k_step = 0; k_step < DCOCO_RANK; k_step += 16) {
            const unsigned int k0 = (unsigned int)k_step + tid_in_group * 2u;
            const unsigned int k1 = k0 + 8u;

            // Manual SM121-safe 32-bit BF16x2 fragment loads (matches inferspark_prefill.cu)
            const unsigned int a0 =
                *(const unsigned int*)&s_z_u16[(size_t)group_id * DCOCO_RANK + k0];
            const unsigned int a1 =
                *(const unsigned int*)&s_z_u16[(size_t)(group_id + 8u) * DCOCO_RANK + k0];
            const unsigned int a2 =
                *(const unsigned int*)&s_z_u16[(size_t)group_id * DCOCO_RANK + k1];
            const unsigned int a3 =
                *(const unsigned int*)&s_z_u16[(size_t)(group_id + 8u) * DCOCO_RANK + k1];

            const unsigned int b0 = dcoco_pack_bf16x2(w1_n, k0);
            const unsigned int b1 = dcoco_pack_bf16x2(w1_n, k1);

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(d0), "f"(d1), "f"(d2), "f"(d3)
            );
        }

        // Fold scalar logit linear weights w1[h, 384] * ell_p(a) and w1[h, 385] * ell_{p+1}(b)
        const int col0 = (int)(n_slice * 8u + tid_in_group * 2u);
        const int col1 = col0 + 1;
        const size_t logit_col = is_b_side ? 385u : 384u;
        const float wl0 =
            __bfloat162float(mlp_w1[(size_t)(h_base + col0) * (size_t)DCOCO_MLP_IN + logit_col]);
        const float wl1 =
            __bfloat162float(mlp_w1[(size_t)(h_base + col1) * (size_t)DCOCO_MLP_IN + logit_col]);

        const unsigned int r0 = group_id;
        const unsigned int r1 = group_id + 8u;
        const float l_r0 = is_b_side ? s_lb[r0] : s_la[r0];
        const float l_r1 = is_b_side ? s_lb[r1] : s_la[r1];

        if (is_b_side) {
            s_tb[r0][col0] = d0 + wl0 * l_r0;
            s_tb[r0][col1] = d1 + wl1 * l_r0;
            s_tb[r1][col0] = d2 + wl0 * l_r1;
            s_tb[r1][col1] = d3 + wl1 * l_r1;
        } else {
            s_ta[r0][col0] = d0 + wl0 * l_r0;
            s_ta[r0][col1] = d1 + wl1 * l_r0;
            s_ta[r1][col0] = d2 + wl0 * l_r1;
            s_ta[r1][col1] = d3 + wl1 * l_r1;
        }
        __syncthreads();

        // 3. Fused GELU + W2 reduction across the 32 hidden channels
        if ((int)pair_a < k_prev && (int)pair_b < top_k) {
            #pragma unroll
            for (int hh = 0; hh < DCOCO_H_TILE; ++hh) {
                const float pre = s_base[hh] + s_ta[pair_a][hh] + s_tb[pair_b][hh];
                acc_out += s_w2_tile[hh] * dcoco_gelu(pre);
            }
        }
        __syncthreads();
    }

    if ((int)pair_a < top_k && (int)pair_b < top_k) {
        const size_t out_off =
            (((size_t)b * (size_t)gamma + (size_t)p) * (size_t)top_k + (size_t)pair_a)
                * (size_t)top_k
            + (size_t)pair_b;
        if ((int)pair_a < k_prev) {
            const float b2_val = __bfloat162float(mlp_b2[0]);
            out_scores[out_off] = s_lb[pair_b] + acc_out + b2_val;
        } else {
            out_scores[out_off] = -INFINITY;
        }
    }
}

// ── Kernel 3: Single-Warp Prefix-First Walk + Wave-15 Maximal-Coupling Q Export ──
//
// Grid:  [batch_size, 1, 1]
// Block: [32, 1, 1]
//
// Modes (`walk_mode`):
//   0 = Greedy argmax (always used when temperatures[b] <= 0 or non-finite)
//   1 = CRN noise-coupled Gumbel-Max
//   2 = Wave-15 Maximal-Coupling Rejection Sampling with continuous `beta`
//       temperature scaling (`draft_temp_scales[b]`):
//         * beta <= 1e-4 -> exact DCOCO Appendix G point-mass (1-hot q)
//         * beta >  1e-4 -> conditional row softmax at eff_temp = max(T*beta, 1e-6)
//       Exports F_t-measurable `out_max_prob[t] = max_z q_t(z | x_{<t})` and
//       `(out_q_ids, out_q_probs)` at `out_q_row_bases[b]` (or `b * gamma`).
extern "C" __global__ void __launch_bounds__(32) dcoco_walk_and_export_q(
    const float* __restrict__ transition_scores,         // [batch_size, gamma, top_k, top_k]
    const unsigned int* __restrict__ candidates,         // [batch_size, gamma, top_k]
    unsigned int* __restrict__ out_tokens,               // [batch_size, gamma]
    float* __restrict__ out_max_prob,                    // [batch_size, gamma]
    unsigned int* __restrict__ out_q_ids,                // [*, out_q_stride] optional
    float* __restrict__ out_q_probs,                     // [*, out_q_stride] optional
    const unsigned int* __restrict__ out_q_row_bases,    // [batch_size] optional
    const unsigned long long* __restrict__ seeds,        // [batch_size]
    const unsigned long long* __restrict__ step0_abs_pos,// [batch_size]
    const float* __restrict__ temperatures,              // [batch_size]
    const float* __restrict__ draft_temp_scales,         // [batch_size] (beta; nullptr -> 1.0f)
    const unsigned int* __restrict__ top_ks,             // [batch_size]
    const float* __restrict__ top_ps,                    // [batch_size]
    const float* __restrict__ min_ps,                    // [batch_size]
    int walk_mode,                                       // 0=greedy, 1=gumbel, 2=maximal_coupling
    int top_k_shortlist,                                 // 1..=32
    int out_q_stride,                                    // >= 0 (0 disables q export)
    int gamma,
    int batch_size
) {
    const int b = (int)blockIdx.x;
    if (b >= batch_size || batch_size <= 0 || gamma <= 0
        || top_k_shortlist <= 0 || top_k_shortlist > DCOCO_WALK_MAX_K) {
        return;
    }

    const size_t stride_k = (size_t)top_k_shortlist;
    const size_t row_base = (size_t)b * (size_t)gamma;
    const size_t q_row_base = (out_q_row_bases != nullptr)
        ? (size_t)out_q_row_bases[b]
        : row_base;

    transition_scores += row_base * stride_k * stride_k;
    candidates        += row_base * stride_k;
    out_tokens        += row_base;
    out_max_prob      += row_base;

    const bool export_q = (out_q_ids != nullptr) && (out_q_probs != nullptr) && (out_q_stride > 0);
    if (export_q) {
        out_q_ids   += q_row_base * (size_t)out_q_stride;
        out_q_probs += q_row_base * (size_t)out_q_stride;
    }

    const float temp = (temperatures != nullptr) ? temperatures[b] : 0.0f;
    const float raw_beta = (draft_temp_scales != nullptr) ? draft_temp_scales[b] : 1.0f;
    const float beta = (isfinite(raw_beta) && raw_beta > 0.0f) ? raw_beta : 0.0f;
    const unsigned long long seed = (seeds != nullptr) ? seeds[b] : 0ull;
    const unsigned long long base_pos = (step0_abs_pos != nullptr) ? step0_abs_pos[b] : 0ull;
    const unsigned int top_k = (top_ks != nullptr) ? top_ks[b] : 0u;
    const float top_p = (top_ps != nullptr) ? top_ps[b] : 1.0f;
    const float min_p = (min_ps != nullptr) ? min_ps[b] : 0.0f;

    const bool sampled = (walk_mode != 0) && (temp > 0.0f) && isfinite(temp);
    const bool point_mass_beta = (walk_mode == 2) && (beta <= 1.0e-4f);
    const float eff_temp = (sampled && !point_mass_beta)
        ? fmaxf(temp * ((beta > 0.0f) ? beta : 1.0f), 1.0e-6f)
        : 1.0f;

    __shared__ float s_score[DCOCO_WALK_MAX_K];
    __shared__ unsigned int s_cand[DCOCO_WALK_MAX_K];
    __shared__ float s_scaled[DCOCO_WALK_MAX_K];
    __shared__ int s_ord[DCOCO_WALK_MAX_K];
    __shared__ float s_w[DCOCO_WALK_MAX_K];
    __shared__ int s_kept;
    __shared__ int s_prev_slot;

    const unsigned int k = threadIdx.x;
    if (k == 0u) {
        s_prev_slot = 0; // Slot 0 reads row 0 (anchor predecessor)
    }
    __syncthreads();

    for (int t = 0; t < gamma; ++t) {
        const int prev_a = s_prev_slot;
        if ((int)k < top_k_shortlist) {
            const size_t score_off =
                ((size_t)t * stride_k + (size_t)prev_a) * stride_k + (size_t)k;
            s_score[k] = transition_scores[score_off];
            s_cand[k] = candidates[(size_t)t * stride_k + (size_t)k];
        }
        __syncthreads();

        if (k == 0u) {
            if (!sampled || point_mass_beta) {
                float best = s_score[0];
                int best_k = 0;
                unsigned int best_cid = s_cand[0];
                for (int j = 1; j < top_k_shortlist; ++j) {
                    const float sc = s_score[j];
                    const unsigned int cid = s_cand[j];
                    if (sc > best || (sc == best && cid < best_cid)) {
                        best = sc;
                        best_k = j;
                        best_cid = cid;
                    }
                }
                float denom = 0.0f;
                for (int j = 0; j < top_k_shortlist; ++j) {
                    const float term = expf(s_score[j] - best);
                    denom += isfinite(term) ? term : 0.0f;
                }
                out_tokens[t] = best_cid;
                out_max_prob[t] = point_mass_beta
                    ? 1.0f
                    : ((denom > 0.0f && isfinite(denom)) ? __fdiv_rn(1.0f, denom) : 1.0f);
                s_ord[0] = best_k;
                s_w[0] = 1.0f;
                s_kept = 1;
                s_prev_slot = best_k;
            } else {
                for (int j = 0; j < top_k_shortlist; ++j) {
                    const float raw_sc = __fdiv_rn(s_score[j], eff_temp);
                    s_scaled[j] = isfinite(raw_sc) ? raw_sc : -1.0e30f;
                    s_ord[j] = j;
                }

                // Deterministic insertion sort descending by (s_scaled[idx], ~s_cand[idx]).
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
                if (export_q && kept > out_q_stride) {
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
                        : ((j == 0) ? 1.0f : 0.0f);
                }

                int selected_j = 0;
                if (walk_mode == 1) {
                    // CRN SplitMix64 Gumbel-Max
                    const unsigned long long pos_base =
                        dcoco_pos_base(seed, base_pos + (unsigned long long)t);
                    unsigned int best_cid = s_cand[s_ord[0]];
                    float best_g = -INFINITY;
                    for (int j = 0; j < kept; ++j) {
                        const int idx = s_ord[j];
                        const unsigned int cid = s_cand[idx];
                        const float g_score = s_scaled[idx] + dcoco_gumbel_from_base(pos_base, cid);
                        if (j == 0 || g_score > best_g || (g_score == best_g && cid < best_cid)) {
                            best_g = g_score;
                            best_cid = cid;
                            selected_j = j;
                        }
                    }
                    out_max_prob[t] = s_w[selected_j];
                } else {
                    // Wave-15 Maximal Coupling: exact inverse-CDF draw from q_t(. | k_p)
                    const float u = dcoco_prob_uniform(seed, base_pos + (unsigned long long)t);
                    float cum = 0.0f;
                    int last_nonzero = 0;
                    for (int j = 0; j < kept; ++j) {
                        const float p_val = s_w[j];
                        if (p_val > 0.0f) {
                            last_nonzero = j;
                        }
                        cum += p_val;
                        if (u < cum) {
                            selected_j = j;
                            break;
                        }
                        selected_j = last_nonzero;
                    }
                    // Strictly F_t-measurable prefix confidence (independent of sampled x_t)
                    out_max_prob[t] = s_w[0];
                }

                const int chosen_slot = s_ord[selected_j];
                out_tokens[t] = s_cand[chosen_slot];
                s_kept = kept;
                s_prev_slot = chosen_slot;
            }
        }
        __syncthreads();

        if (export_q) {
            const int kept_out = s_kept;
            unsigned int* row_q_ids = out_q_ids + (size_t)t * (size_t)out_q_stride;
            float* row_q_probs = out_q_probs + (size_t)t * (size_t)out_q_stride;
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
}
