// SPDX-License-Identifier: AGPL-3.0-only

// Atlas tree-aware WY-fused Gated Delta Rule — M8A v2 (Wave 2 Optimized).
//
// Wave 2 Optimizations (Decode-75 DFlash-27B):
//   1. Eliminates the 640-byte per-thread local-memory stack frame (`0 B stack,
//      0 B spill stores, 0 B spill loads`):
//      - Replaces the per-thread `int chain[32]` local stack array with a CTA-wide
//        32-bit ancestor bitmask `s_anc_mask[32]` and precomputed WY weights in
//        shared memory (`s_lead[32]`, `kd_flat[496]`). Because `parent_ids[x] < x`
//        is a strict DAG invariant, iterating `s = 0..t-1` under `(mask >> s) & 1u`
//        visits ancestors in the exact oldest-to-newest order (`chain[L-1..0]`).
//      - Dispatches a register-unrolled `<17>` fast path for `T <= 17` (covering
//        all DFlash γ+1 verify steps) and `<32>` for `17 < T <= 32`.
//   2. Collapses the `T*(T-1)/2` sequential `atlas_block_reduce_sum` calls in
//      `kd_flat` (408 `__syncthreads()` barriers at T=17, 1,488 barriers at T=32)
//      into 2 `__syncthreads()` barriers by reusing `sq` as warp-partial staging
//      before loading `q`, preserving the exact `(w0 + w2) + (w1 + w3)` FP32 tree.
//   3. Inverts the PASS 2 loop order (`j` outer, `t` inner) so linear-chain and
//      contiguous branch-tail steps (`parent[t] == t - 1`) keep rolling `h0..h3`
//      in registers and root steps (`parent[t] < 0`) reuse register-cached
//      `hr0..hr3`, eliminating up to 17 redundant 64 KB `H_in` DRAM reads per head
//      (saving ~50 MB of DRAM traffic per GDN layer = ~2.4 GB per verify step).

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#define BLOCK_SIZE 128
#define K_MAX 32
#define MAX_PAIRS (K_MAX * (K_MAX - 1) / 2)

template <int K_UNROLL>
__device__ __forceinline__ void gated_delta_rule_tree_wy_pass(
    const float* __restrict__ H_root,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_inter_base,
    const float (* __restrict__ sk)[128],
    const float (* __restrict__ sq)[128],
    const float* __restrict__ sg,
    const float* __restrict__ sbt,
    const int* __restrict__ sparent,
    const unsigned int* __restrict__ s_anc_mask,
    const float* __restrict__ s_lead,
    const float* __restrict__ kd_flat,
    unsigned int inter_stride_floats,
    unsigned int vh_off,
    unsigned int T,
    unsigned int b,
    unsigned int vh,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int v_stride,
    unsigned int tid
) {
    // ── PASS 1: one read of H_root, compute hk_root[t] for all t ──
    float hk_root[K_UNROLL];
    #pragma unroll
    for (int t = 0; t < K_UNROLL; t++) hk_root[t] = 0.0f;

    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H_root[(j + 0) * v_dim + tid];
        float h1 = H_root[(j + 1) * v_dim + tid];
        float h2 = H_root[(j + 2) * v_dim + tid];
        float h3 = H_root[(j + 3) * v_dim + tid];
        #pragma unroll
        for (int t = 0; t < K_UNROLL; t++) {
            if ((unsigned int)t < T) {
                hk_root[t] += h0 * sk[t][j + 0] + h1 * sk[t][j + 1]
                            + h2 * sk[t][j + 2] + h3 * sk[t][j + 3];
            }
        }
    }

    // ── WY correction (unrolled in registers, ancestor-bitmask gated) ──
    float vn[K_UNROLL];
    #pragma unroll
    for (int t = 0; t < K_UNROLL; t++) vn[t] = 0.0f;

    #pragma unroll
    for (int t = 0; t < K_UNROLL; t++) {
        if ((unsigned int)t < T) {
            float corrected = s_lead[t] * hk_root[t];
            const unsigned int mask = s_anc_mask[t];
            #pragma unroll
            for (int s = 0; s < t; s++) {
                if ((mask >> s) & 1u) {
                    corrected += kd_flat[t * (t - 1) / 2 + s] * vn[s];
                }
            }
            const __nv_bfloat16* vp = value + (b * T + (unsigned int)t) * v_stride + vh * v_dim;
            vn[t] = ((float)vp[tid] - sg[t] * corrected) * sbt[t];
        }
    }

    // ── PASS 2: outer loop over j, inner unrolled loop over t ──
    // Keeps rolling h0..h3 in registers across linear-chain / branch-tail steps
    // (`sparent[t] == t - 1`) and reuses `hr0..hr3` at root steps (`sparent[t] < 0`).
    float qd[K_UNROLL];
    #pragma unroll
    for (int t = 0; t < K_UNROLL; t++) qd[t] = 0.0f;

    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        const float hr0 = H_root[(j + 0) * v_dim + tid];
        const float hr1 = H_root[(j + 1) * v_dim + tid];
        const float hr2 = H_root[(j + 2) * v_dim + tid];
        const float hr3 = H_root[(j + 3) * v_dim + tid];
        float h0 = hr0, h1 = hr1, h2 = hr2, h3 = hr3;

        #pragma unroll
        for (int t = 0; t < K_UNROLL; t++) {
            if ((unsigned int)t < T) {
                const int p = sparent[t];
                if (p < 0) {
                    h0 = hr0; h1 = hr1; h2 = hr2; h3 = hr3;
                } else if (p != t - 1) {
                    const float* H_in =
                        h_state_inter_base + ((unsigned int)p) * inter_stride_floats + vh_off;
                    h0 = H_in[(j + 0) * v_dim + tid];
                    h1 = H_in[(j + 1) * v_dim + tid];
                    h2 = H_in[(j + 2) * v_dim + tid];
                    h3 = H_in[(j + 3) * v_dim + tid];
                }
                h0 = sg[t] * h0 + sk[t][j + 0] * vn[t];
                h1 = sg[t] * h1 + sk[t][j + 1] * vn[t];
                h2 = sg[t] * h2 + sk[t][j + 2] * vn[t];
                h3 = sg[t] * h3 + sk[t][j + 3] * vn[t];

                float* H_out =
                    h_state_inter_base + ((unsigned int)t) * inter_stride_floats + vh_off;
                H_out[(j + 0) * v_dim + tid] = h0;
                H_out[(j + 1) * v_dim + tid] = h1;
                H_out[(j + 2) * v_dim + tid] = h2;
                H_out[(j + 3) * v_dim + tid] = h3;

                qd[t] += h0 * sq[t][j + 0] + h1 * sq[t][j + 1]
                       + h2 * sq[t][j + 2] + h3 * sq[t][j + 3];
            }
        }
    }

    // ── Write outputs ──
    float scale = rsqrtf((float)k_dim);
    #pragma unroll
    for (int t = 0; t < K_UNROLL; t++) {
        if ((unsigned int)t < T) {
            output[((b * T + (unsigned int)t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(qd[t] * scale);
        }
    }
}

extern "C" __global__ void gated_delta_rule_tree_wy(
    float* __restrict__ h_state,                // [batch, vh, k_dim, v_dim] FP32 — RO here
    const __nv_bfloat16* __restrict__ query,    // [batch*T, qk_stride]
    const __nv_bfloat16* __restrict__ key,      // same shape as query
    const __nv_bfloat16* __restrict__ value,    // [batch*T, v_stride]
    const float* __restrict__ gate,             // [batch*T, gb_stride]
    const float* __restrict__ beta,             // [batch*T, gb_stride]
    const int* __restrict__ parent_ids,         // [T] — -1 for root, else < t
    __nv_bfloat16* __restrict__ output,         // [batch*T, num_v_heads, v_dim]
    float* __restrict__ h_state_inter_base,     // [T, batch, vh, hv] stride per inter_stride_floats
    unsigned int inter_stride_floats,
    unsigned int num_tokens,                    // T, 1..K_MAX
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31u;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;
    const unsigned int T = (num_tokens <= K_MAX) ? num_tokens : K_MAX;

    const float* H_root = h_state + ((b * num_v_heads + vh) * hv);
    const unsigned int vh_off = (b * num_v_heads + vh) * hv;

    // ── SMEM ──
    __shared__ float sk[K_MAX][128];
    __shared__ float sq[K_MAX][128];
    __shared__ float sg[K_MAX];
    __shared__ float sbt[K_MAX];
    __shared__ int   sparent[K_MAX];
    __shared__ unsigned int s_anc_mask[K_MAX];
    __shared__ float s_lead[K_MAX];
    __shared__ float kd_flat[MAX_PAIRS];

    // Load k, gate, beta, parent_ids first (`sq` is temporarily used as
    // `smem_pair_warps[496 * 4]` during `kd_flat` reduction before loading `q`).
    if (tid < k_dim) {
        for (unsigned int t = 0; t < T; t++) {
            const __nv_bfloat16* kp = key + (b * T + t) * qk_stride + kh * k_dim;
            sk[t][tid] = (float)kp[tid];
        }
    }
    if (tid < T) {
        float g_raw = gate[(b * T + tid) * gb_stride + vh];
        sg[tid]  = fminf(fmaxf(g_raw, 0.0f), 1.0f);
        sbt[tid] = beta[(b * T + tid) * gb_stride + vh];
        sparent[tid] = parent_ids[tid];
    }
    __syncthreads();

    // ── Batched kd[t][s] warp reductions + CTA-wide ancestor bitmask & lead_prod ──
    float* smem_pair_warps = &sq[0][0];
    for (unsigned int t = 1; t < T; t++) {
        for (unsigned int s = 0; s < t; s++) {
            const unsigned int pair_idx = t * (t - 1u) / 2u + s;
            float p = (tid < k_dim) ? sk[t][tid] * sk[s][tid] : 0.0f;
            float wsum = atlas_warp_reduce_sum(p);
            if (lane_id == 0u) {
                smem_pair_warps[pair_idx * 4u + warp_id] = wsum;
            }
        }
    }
    if (tid < T) {
        unsigned int mask = 0u;
        int cur = sparent[tid];
        int steps = 0;
        while (cur >= 0 && steps < K_MAX) {
            mask |= (1u << (unsigned int)cur);
            cur = sparent[cur];
            steps++;
        }
        s_anc_mask[tid] = mask;

        // Accumulate leading product oldest->newest (ascending index `u`).
        float lead = 1.0f;
        for (unsigned int u = 0; u < tid; u++) {
            if ((mask >> u) & 1u) {
                lead *= sg[u];
            }
        }
        s_lead[tid] = lead;
    }
    __syncthreads();

    // Cross-warp reduction `(w0 + w2) + (w1 + w3)` + pre-multiplication by `gprod`.
    const int active_pairs = (int)(T * (T - 1u) / 2u);
    for (int pair_idx = (int)tid; pair_idx < active_pairs; pair_idx += BLOCK_SIZE) {
        float w0 = smem_pair_warps[pair_idx * 4 + 0];
        float w1 = smem_pair_warps[pair_idx * 4 + 1];
        float w2 = smem_pair_warps[pair_idx * 4 + 2];
        float w3 = smem_pair_warps[pair_idx * 4 + 3];
        float kd = __fadd_rn(__fadd_rn(w0, w2), __fadd_rn(w1, w3));

        int t = 1;
        while ((t * (t + 1)) / 2 <= pair_idx) {
            t++;
        }
        int s = pair_idx - (t * (t - 1)) / 2;
        const unsigned int mask = s_anc_mask[t];
        if ((mask >> s) & 1u) {
            // Multiply closer-than-s ancestors oldest->newest (`u = s+1 .. t-1`).
            float gprod = 1.0f;
            for (int u = s + 1; u < t; u++) {
                if ((mask >> u) & 1u) {
                    gprod *= sg[u];
                }
            }
            kd_flat[pair_idx] = gprod * kd;
        } else {
            kd_flat[pair_idx] = 0.0f;
        }
    }
    __syncthreads();

    // Load `sq` into shared memory now that `smem_pair_warps` has been consumed.
    if (tid < k_dim) {
        for (unsigned int t = 0; t < T; t++) {
            const __nv_bfloat16* qp = query + (b * T + t) * qk_stride + kh * k_dim;
            sq[t][tid] = (float)qp[tid];
        }
    }
    __syncthreads();

    if (tid < v_dim) {
        if (T <= 17u) {
            gated_delta_rule_tree_wy_pass<17>(
                H_root, value, output, h_state_inter_base,
                sk, sq, sg, sbt, sparent, s_anc_mask, s_lead, kd_flat,
                inter_stride_floats, vh_off, T, b, vh, num_v_heads,
                k_dim, v_dim, v_stride, tid);
        } else {
            gated_delta_rule_tree_wy_pass<32>(
                H_root, value, output, h_state_inter_base,
                sk, sq, sg, sbt, sparent, s_anc_mask, s_lead, kd_flat,
                inter_stride_floats, vh_off, T, b, vh, num_v_heads,
                k_dim, v_dim, v_stride, tid);
        }
    }
}
