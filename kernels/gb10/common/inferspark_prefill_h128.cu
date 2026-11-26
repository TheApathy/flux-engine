// SPDX-License-Identifier: AGPL-3.0-only

// Inferspark Prefill Attention v2 — Tensor Core Flash Attention for SM121.
//
// Optimizations over v1:
//   - cp.async.cg for vectorized 16-byte K/V loads (8x fewer instructions)
//   - Separate smem_K / smem_V buffers: V loads overlap with QK^T compute
//   - Register-based softmax with warp shuffle reductions (eliminates smem_S)
//   - PAD=8 for 16-byte row alignment (bank-conflict-free)
//
// Algorithm: Tiled online softmax over sequence length (Flash Attention v2).
// For each query block Br, iterate over all KV blocks Bc:
//   1. S = Q_tile @ K_tile^T / sqrt(d)  (TC GEMM via mma.sync m16n8k16)
//   2. Apply causal mask
//   3. Online softmax in registers: warp-shuffle max/sum reduction
//   4. O += P @ V_tile                  (TC GEMM via mma.sync m16n8k16)
//
// 4 warps (128 threads):
//   QK^T: warps 0-1 each own 16 M-rows, compute S[32,32]
//   PV:   all 4 warps — warps 0,2 share rows 0-15, warps 1,3 share rows 16-31
//         Each warp handles 8 of 16 N-tiles (half of head_dim=128)
//
// Shared memory (PAD=8, stride=136 BF16 = 272 bytes per row):
//   Q:   [32][136] BF16 =  8.5 KB
//   K:   [2][32][136] BF16 = 17.0 KB  (double-buffered: K[i] used, K[i+1] loading)
//   V:   [32][136] BF16 =  8.5 KB  (loaded async during QK^T, used for PV)
//   P:   [32][40]  BF16 =  2.5 KB
//   m/l: [32][2]   FP32 =  0.3 KB
//   Total: ~36.8 KB (within 99 KB/SM)

#include <cuda_bf16.h>

#define BR 32
#define BC 32
#ifndef HDIM
#define HDIM 128
#endif
#define PAD_KV 8           // 16-byte row alignment: (128+8)*2 = 272 bytes
#define HDIM_PAD (HDIM + PAD_KV)  // 136
#define PAD_P 8            // P stride: BC + PAD_P = 40, 40*2=80 bytes (16-byte aligned)
#define N_TILES_PER_WARP ((HDIM / 8) / 2)  // total n-tiles / 2 warp-pairs

// Number of 16-byte (8-element) chunks per tile: 32 rows * (128/8) = 32*16 = 512
#define TILE_CHUNKS (BR * (HDIM / 8))

extern "C" __global__ void inferspark_prefill_h128(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int query_start,
    const unsigned int query_len_total,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window   // 0 = no sliding limit; >0 = mask out k where q - k >= sliding_window
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;

    const unsigned int q_start = query_start + q_block * BR;
    if (q_start >= seq_len || q_start >= query_start + query_len_total) return;
    const unsigned int q_end = min(q_start + BR, min(seq_len, query_start + query_len_total));
    const unsigned int q_len = q_end - q_start;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned long long q_seq_stride = (unsigned long long)num_q_heads * head_dim;
    const unsigned long long kv_seq_stride = (unsigned long long)num_kv_heads * head_dim;

    const __nv_bfloat16* Q_batch = Q + (unsigned long long)batch * seq_len * q_seq_stride;
    const __nv_bfloat16* K_batch = K + (unsigned long long)batch * seq_len * kv_seq_stride;
    const __nv_bfloat16* V_batch = V + (unsigned long long)batch * seq_len * kv_seq_stride;
    __nv_bfloat16* O_batch = O + (unsigned long long)batch * seq_len * q_seq_stride;

    // Shared memory — double-buffered K + separate V for full async overlap
    __shared__ __nv_bfloat16 smem_Q[BR][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[2][BC][HDIM_PAD];  // double-buffered
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P[BR][BC + PAD_P];
    __shared__ float smem_ml[BR][2]; // [row][0]=m, [row][1]=l

    // MMA lane mapping
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;

    // Warp role mapping
    const unsigned int qk_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * N_TILES_PER_WARP;

    // Output accumulators
    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }

    // Per-row online softmax state (each thread owns 2 rows)
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    const unsigned int p_smem_stride = BC + PAD_P;

    // === KV block count (computed early for merged load) ===
    unsigned int num_kv_blocks = (seq_len + BC - 1) / BC;
    if (causal) {
        unsigned int max_kv_block = (q_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1);
    }

    // === Merged Q + K[0] load (single cp.async commit group) ===
    // Saves one commit/wait/sync vs separate Q and K loads.
    {
        const unsigned int chunks_per_row = HDIM / 8;  // 16

        // Q tile
        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
            unsigned int row = idx / chunks_per_row;
            unsigned int chunk = idx % chunks_per_row;
            unsigned int col = chunk * 8;
            unsigned int q_row = q_start + row;
            unsigned int smem_addr = __cvta_generic_to_shared(&smem_Q[row][col]);

            if (q_row < seq_len) {
                const void* gmem = (const void*)&Q_batch[(unsigned long long)q_row * q_seq_stride + (unsigned long long)q_head * head_dim + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }

        // K[0] tile (same commit group — no extra sync)
        if (num_kv_blocks > 0) {
            for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K[0][row][col]);

                if (row < seq_len) {
                    const void* gmem = (const void*)&K_batch[(unsigned long long)row * kv_seq_stride + (unsigned long long)kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_K[0][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, seq_len);
        unsigned int kv_len = kv_end - kv_start;
        unsigned int buf = kv_block & 1;

        // === Start async V load into smem_V (overlaps with QK^T below) ===
        {
            const unsigned int chunks_per_row = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int v_row = kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_V[row][col]);

                if (v_row < seq_len) {
                    const void* gmem = (const void*)&V_batch[(unsigned long long)v_row * kv_seq_stride + (unsigned long long)kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_V[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");  // V tile loading in background
        }

        // K[kv_block] already in smem_K[buf] (preloaded or from prev iteration)

        // === QK^T (warps 0-1, register-based) ===
        float acc_s[4][4];  // [n_tile][{row0_c0, row0_c1, row1_c0, row1_c1}]
        if (warp_id < 2) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                acc_s[i][0] = 0.0f; acc_s[i][1] = 0.0f;
                acc_s[i][2] = 0.0f; acc_s[i][3] = 0.0f;
            }

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM / 16); ks++) {
                unsigned int k_base = ks * 16;

                // SM121 workaround: manual Q register loading
                // (ldmatrix.x4 produces incorrect results on GB10)
                const unsigned short* sQ_u16 = (const unsigned short*)smem_Q;
                unsigned int ar0 = qk_warp_m + group_id;
                unsigned int ar1 = qk_warp_m + group_id + 8;
                unsigned int ac0 = k_base + tid_in_group * 2;
                unsigned int ac1 = k_base + tid_in_group * 2 + 8;
                unsigned int a0 = *(const unsigned int*)&sQ_u16[ar0 * HDIM_PAD + ac0];
                unsigned int a1 = *(const unsigned int*)&sQ_u16[ar1 * HDIM_PAD + ac0];
                unsigned int a2 = *(const unsigned int*)&sQ_u16[ar0 * HDIM_PAD + ac1];
                unsigned int a3 = *(const unsigned int*)&sQ_u16[ar1 * HDIM_PAD + ac1];

                // B fragments: iterate over 4 N-tiles of K^T
                // SM121 workaround: manual B-operand register loading
                // (ldmatrix.trans produces incorrect results on GB10)
                const unsigned short* sK_u16 = (const unsigned short*)smem_K[buf];
                #pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    unsigned int n_col = nt * 8 + group_id;
                    unsigned int k0 = k_base + tid_in_group * 2;
                    unsigned int k1 = k_base + tid_in_group * 2 + 8;
                    unsigned int b0 = ((unsigned int)sK_u16[n_col * HDIM_PAD + k0 + 1] << 16) |
                                      (unsigned int)sK_u16[n_col * HDIM_PAD + k0];
                    unsigned int b1 = ((unsigned int)sK_u16[n_col * HDIM_PAD + k1 + 1] << 16) |
                                      (unsigned int)sK_u16[n_col * HDIM_PAD + k1];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]),
                          "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_s[nt][0]), "f"(acc_s[nt][1]),
                          "f"(acc_s[nt][2]), "f"(acc_s[nt][3])
                    );
                }
            }

            // === Register-based softmax ===
            unsigned int row0 = qk_warp_m + group_id;
            unsigned int row1 = row0 + 8;

            // Scale + causal mask + boundary checks (in registers)
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                acc_s[nt][0] *= inv_sqrt_d;
                acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d;
                acc_s[nt][3] *= inv_sqrt_d;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                unsigned int col1 = col0 + 1;

                if (causal) {
                    unsigned int qr0 = q_start + row0, qr1 = q_start + row1;
                    if (kv_start + col0 > qr0) acc_s[nt][0] = -1e30f;
                    if (kv_start + col1 > qr0) acc_s[nt][1] = -1e30f;
                    if (kv_start + col0 > qr1) acc_s[nt][2] = -1e30f;
                    if (kv_start + col1 > qr1) acc_s[nt][3] = -1e30f;
                }
                // Sliding-window mask: key position k out-of-window for
                // query q when (q - k) >= sliding_window. sliding_window=0
                // (full attn) leaves scores untouched. Applied independently
                // of causal so DFlash bidirectional mode (causal=0) still
                // respects per-layer sliding_window limits.
                if (sliding_window > 0) {
                    unsigned int qr0 = q_start + row0, qr1 = q_start + row1;
                    unsigned int k0 = kv_start + col0, k1 = kv_start + col1;
                    if (qr0 >= k0 && qr0 - k0 >= sliding_window) acc_s[nt][0] = -1e30f;
                    if (qr0 >= k1 && qr0 - k1 >= sliding_window) acc_s[nt][1] = -1e30f;
                    if (qr1 >= k0 && qr1 - k0 >= sliding_window) acc_s[nt][2] = -1e30f;
                    if (qr1 >= k1 && qr1 - k1 >= sliding_window) acc_s[nt][3] = -1e30f;
                }
                if (col0 >= kv_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (col1 >= kv_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (row0 >= q_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (row1 >= q_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }

            // Row max: local max then warp shuffle across tid_in_group (4 threads)
            float rmax0 = -1e30f, rmax1 = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
                rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

            // Online softmax update: rescale previous accumulators.
            // Guard fully-masked tiles (e.g. causal tail or SWA prefix tiles) so
            // `-1e30f - (-1e30f)` never produces `exp(0) = 1.0f`.
            const bool has_valid0 = (rmax0 > -1e29f);
            float m_new0 = has_valid0 ? fmaxf(m_r0, rmax0) : m_r0;
            float exp_old0 = (has_valid0 && m_r0 > -1e29f)
                ? __expf(m_r0 - m_new0)
                : (has_valid0 ? 0.0f : 1.0f);
            l_r0 *= exp_old0;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_old0; acc_o[i][1] *= exp_old0;
            }
            m_r0 = m_new0;

            const bool has_valid1 = (rmax1 > -1e29f);
            float m_new1 = has_valid1 ? fmaxf(m_r1, rmax1) : m_r1;
            float exp_old1 = (has_valid1 && m_r1 > -1e29f)
                ? __expf(m_r1 - m_new1)
                : (has_valid1 ? 0.0f : 1.0f);
            l_r1 *= exp_old1;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][2] *= exp_old1; acc_o[i][3] *= exp_old1;
            }
            m_r1 = m_new1;

            // Compute P = exp(s - m) in registers, write to smem_P
            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                float p00 = (has_valid0 && acc_s[nt][0] > -1e29f) ? __expf(acc_s[nt][0] - m_r0) : 0.0f;
                float p01 = (has_valid0 && acc_s[nt][1] > -1e29f) ? __expf(acc_s[nt][1] - m_r0) : 0.0f;
                float p10 = (has_valid1 && acc_s[nt][2] > -1e29f) ? __expf(acc_s[nt][2] - m_r1) : 0.0f;
                float p11 = (has_valid1 && acc_s[nt][3] > -1e29f) ? __expf(acc_s[nt][3] - m_r1) : 0.0f;
                sum0 += p00 + p01;
                sum1 += p10 + p11;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                smem_P[row0][col0]     = __float2bfloat16(p00);
                smem_P[row0][col0 + 1] = __float2bfloat16(p01);
                smem_P[row1][col0]     = __float2bfloat16(p10);
                smem_P[row1][col0 + 1] = __float2bfloat16(p11);
            }

            // Row sum reduction via warp shuffle
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0;
            l_r1 += sum1;

            // Share m/l with warps 2-3
            if (tid_in_group == 0) {
                smem_ml[row0][0] = m_r0; smem_ml[row0][1] = l_r0;
                smem_ml[row1][0] = m_r1; smem_ml[row1][1] = l_r1;
            }
        }

        // Wait for V tile load (group 1 complete) — was loading during QK^T+softmax
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Warps 2-3: rescale accumulators to match current m
        if (warp_id >= 2) {
            unsigned int row0 = pv_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            float cur_m0 = smem_ml[row0][0];
            float cur_m1 = smem_ml[row1][0];
            float exp_r0 = (cur_m0 > -1e29f && m_r0 > -1e29f) ? __expf(m_r0 - cur_m0) : 0.0f;
            float exp_r1 = (cur_m1 > -1e29f && m_r1 > -1e29f) ? __expf(m_r1 - cur_m1) : 0.0f;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_r0; acc_o[i][1] *= exp_r0;
                acc_o[i][2] *= exp_r1; acc_o[i][3] *= exp_r1;
            }
            m_r0 = cur_m0; m_r1 = cur_m1;
        }

        // === Preload K[i+1] into smem_K[1-buf] (overlaps with PV below) ===
        if (kv_block + 1 < num_kv_blocks) {
            unsigned int next_kv_start = (kv_block + 1) * BC;
            const unsigned int chunks_per_row_k = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
                unsigned int row = idx / chunks_per_row_k;
                unsigned int chunk = idx % chunks_per_row_k;
                unsigned int col = chunk * 8;
                unsigned int k_row = next_kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K[1 - buf][row][col]);

                if (k_row < seq_len) {
                    const void* gmem = (const void*)&K_batch[(unsigned long long)k_row * kv_seq_stride + (unsigned long long)kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_K[1 - buf][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");  // K[i+1] loading in background
        }

        // === PV MMA (all 4 warps, 8 n-tiles each, V from smem_V) ===
        {
            #pragma unroll
            for (unsigned int ks = 0; ks < 2; ks++) {
                unsigned int k_off = ks * 16;

                // SM121 workaround: manual P register loading
                // (ldmatrix.x4 produces incorrect results on GB10)
                const unsigned short* sP_u16 = (const unsigned short*)smem_P;
                unsigned int ar0 = pv_warp_m + group_id;
                unsigned int ar1 = pv_warp_m + group_id + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = k_off + tid_in_group * 2 + 8;
                unsigned int a0 = *(const unsigned int*)&sP_u16[ar0 * p_smem_stride + ac0];
                unsigned int a1 = *(const unsigned int*)&sP_u16[ar1 * p_smem_stride + ac0];
                unsigned int a2 = *(const unsigned int*)&sP_u16[ar0 * p_smem_stride + ac1];
                unsigned int a3 = *(const unsigned int*)&sP_u16[ar1 * p_smem_stride + ac1];

                // SM121 workaround: manual V register loading
                // (ldmatrix.trans produces incorrect results on GB10)
                const unsigned short* sV_u16 = (const unsigned short*)smem_V;

                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
                    unsigned int n_col = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k_off + tid_in_group * 2 + 8;
                    unsigned int b0 = ((unsigned int)sV_u16[(k0 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV_u16[k0 * HDIM_PAD + n_col];
                    unsigned int b1 = ((unsigned int)sV_u16[(k1 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV_u16[k1 * HDIM_PAD + n_col];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]),
                          "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_o[nt][0]), "f"(acc_o[nt][1]),
                          "f"(acc_o[nt][2]), "f"(acc_o[nt][3])
                    );
                }
            }
        }

        // Wait for K[i+1] prefetch to complete before next iteration
        if (kv_block + 1 < num_kv_blocks) {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
    }

    // === Final normalization and store ===
    {
        unsigned int row0 = pv_warp_m + group_id;
        unsigned int row1 = row0 + 8;

        float inv_l0, inv_l1;
        if (warp_id < 2) {
            inv_l0 = (l_r0 > 0.0f) ? (1.0f / l_r0) : 0.0f;
            inv_l1 = (l_r1 > 0.0f) ? (1.0f / l_r1) : 0.0f;
        } else {
            float l0 = smem_ml[row0][1];
            float l1 = smem_ml[row1][1];
            inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
            inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
        }

        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            unsigned int col0 = (pv_n_start + nt) * 8 + tid_in_group * 2;

            unsigned int gr0 = q_start + row0;
            unsigned int gr1 = q_start + row1;
            __nv_bfloat16* o_base = O_batch + q_head * head_dim;

            // Vectorized packed BF16x2 store (32-bit write instead of 2x 16-bit)
            if (gr0 < seq_len && row0 < q_len && col0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * inv_l0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * inv_l0));
                *(unsigned int*)&o_base[(unsigned long long)gr0 * q_seq_stride + col0] = lo | (hi << 16);
            }
            if (gr1 < seq_len && row1 < q_len && col0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * inv_l1));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * inv_l1));
                *(unsigned int*)&o_base[(unsigned long long)gr1 * q_seq_stride + col0] = lo | (hi << 16);
            }
        }
    }
}

// ============================================================================
// BR=64 variant: 8 warps (256 threads) for longer sequences (seq >= 256).
//
// Key differences from BR=32:
//   - 64 Q rows per CTA (halves number of CTAs, halves causal KV iterations)
//   - 256 threads → 2x faster cp.async tile loads
//   - QK^T: warps 0-3, each owns 16 M-rows (0-15, 16-31, 32-47, 48-63)
//   - PV:   all 8 warps in 4 pairs:
//           (0,4)→rows 0-15, (1,5)→rows 16-31,
//           (2,6)→rows 32-47, (3,7)→rows 48-63
//           Each warp handles 8 of 16 N-tiles (64 of 128 head_dim columns)
//
// Shared memory (~48 KB, within 99 KB/SM limit):
//   Q:   [64][136] = 17.0 KB
//   K:   [2][32][136] = 17.0 KB  (double-buffered)
//   V:   [32][136] =  8.5 KB
//   P:   [64][40]  =  5.0 KB
//   m/l: [64][2]   =  0.5 KB
// ============================================================================

#define BR64 64
#define TILE_CHUNKS_Q64 (BR64 * (HDIM / 8))  // 1024
#define TILE_CHUNKS_KV  (BC * (HDIM / 8))     // 512

extern "C" __global__ void inferspark_prefill_h128_64(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int query_start,
    const unsigned int query_len_total,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window   // 0 = no sliding limit
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;

    const unsigned int q_start = query_start + q_block * BR64;
    if (q_start >= seq_len || q_start >= query_start + query_len_total) return;
    const unsigned int q_end = min(q_start + BR64, min(seq_len, query_start + query_len_total));
    const unsigned int q_len = q_end - q_start;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned long long q_seq_stride = (unsigned long long)num_q_heads * head_dim;
    const unsigned long long kv_seq_stride = (unsigned long long)num_kv_heads * head_dim;

    const __nv_bfloat16* Q_batch = Q + (unsigned long long)batch * seq_len * q_seq_stride;
    const __nv_bfloat16* K_batch = K + (unsigned long long)batch * seq_len * kv_seq_stride;
    const __nv_bfloat16* V_batch = V + (unsigned long long)batch * seq_len * kv_seq_stride;
    __nv_bfloat16* O_batch = O + (unsigned long long)batch * seq_len * q_seq_stride;

    __shared__ __nv_bfloat16 smem_Q[BR64][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K64[2][BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V64[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P64[BR64][BC + PAD_P];
    __shared__ float smem_ml64[BR64][2];

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;

    // QK^T: warps 0-3, each handles 16 M-rows
    const unsigned int qk_warp_m = warp_id * 16;  // valid for warp_id < 4

    // PV: all 8 warps, 4 pairs
    const unsigned int pv_warp_m = (warp_id & 3) * 16;
    const unsigned int pv_n_start = (warp_id >> 2) * N_TILES_PER_WARP;

    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }

    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    const unsigned int p_smem_stride64 = BC + PAD_P;

    // === KV block count ===
    unsigned int num_kv_blocks = (seq_len + BC - 1) / BC;
    if (causal) {
        unsigned int max_kv_block = (q_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1);
    }

    // === Merged Q + K[0] load (single cp.async commit group) ===
    {
        const unsigned int chunks_per_row = HDIM / 8;

        // Q tile (64 rows)
        for (unsigned int idx = tid; idx < TILE_CHUNKS_Q64; idx += 256) {
            unsigned int row = idx / chunks_per_row;
            unsigned int chunk = idx % chunks_per_row;
            unsigned int col = chunk * 8;
            unsigned int q_row = q_start + row;
            unsigned int smem_addr = __cvta_generic_to_shared(&smem_Q[row][col]);

            if (q_row < seq_len) {
                const void* gmem = (const void*)&Q_batch[(unsigned long long)q_row * q_seq_stride + (unsigned long long)q_head * head_dim + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }

        // K[0] tile (32 rows)
        if (num_kv_blocks > 0) {
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 256) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K64[0][row][col]);

                if (row < seq_len) {
                    const void* gmem = (const void*)&K_batch[(unsigned long long)row * kv_seq_stride + (unsigned long long)kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_K64[0][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, seq_len);
        unsigned int kv_len = kv_end - kv_start;
        unsigned int buf = kv_block & 1;

        // === Async V load (overlaps with QK^T) ===
        {
            const unsigned int chunks_per_row = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 256) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int v_row = kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_V64[row][col]);

                if (v_row < seq_len) {
                    const void* gmem = (const void*)&V_batch[(unsigned long long)v_row * kv_seq_stride + (unsigned long long)kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_V64[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // === QK^T (warps 0-3, each 16 M-rows) ===
        float acc_s[4][4];
        if (warp_id < 4) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                acc_s[i][0] = 0.0f; acc_s[i][1] = 0.0f;
                acc_s[i][2] = 0.0f; acc_s[i][3] = 0.0f;
            }

            const unsigned short* sQ = (const unsigned short*)smem_Q;
            const unsigned short* sK = (const unsigned short*)smem_K64[buf];

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM / 16); ks++) {
                unsigned int k_off = ks * 16;
                unsigned int ar0 = qk_warp_m + group_id;
                unsigned int ar1 = qk_warp_m + group_id + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = k_off + tid_in_group * 2 + 8;

                unsigned int a0 = *(const unsigned int*)&sQ[ar0 * HDIM_PAD + ac0];
                unsigned int a1 = *(const unsigned int*)&sQ[ar1 * HDIM_PAD + ac0];
                unsigned int a2 = *(const unsigned int*)&sQ[ar0 * HDIM_PAD + ac1];
                unsigned int a3 = *(const unsigned int*)&sQ[ar1 * HDIM_PAD + ac1];

                #pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    unsigned int n_col = nt * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k_off + tid_in_group * 2 + 8;

                    unsigned int b0 = ((unsigned int)sK[n_col * HDIM_PAD + k0 + 1] << 16) |
                                      (unsigned int)sK[n_col * HDIM_PAD + k0];
                    unsigned int b1 = ((unsigned int)sK[n_col * HDIM_PAD + k1 + 1] << 16) |
                                      (unsigned int)sK[n_col * HDIM_PAD + k1];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]),
                          "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_s[nt][0]), "f"(acc_s[nt][1]),
                          "f"(acc_s[nt][2]), "f"(acc_s[nt][3])
                    );
                }
            }

            // === Register-based softmax (warps 0-3) ===
            unsigned int row0 = qk_warp_m + group_id;
            unsigned int row1 = row0 + 8;

            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                acc_s[nt][0] *= inv_sqrt_d;
                acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d;
                acc_s[nt][3] *= inv_sqrt_d;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                unsigned int col1 = col0 + 1;

                if (causal) {
                    unsigned int qr0 = q_start + row0, qr1 = q_start + row1;
                    if (kv_start + col0 > qr0) acc_s[nt][0] = -1e30f;
                    if (kv_start + col1 > qr0) acc_s[nt][1] = -1e30f;
                    if (kv_start + col0 > qr1) acc_s[nt][2] = -1e30f;
                    if (kv_start + col1 > qr1) acc_s[nt][3] = -1e30f;
                }
                // Sliding-window mask: applied independently of causal mode.
                if (sliding_window > 0) {
                    unsigned int qr0 = q_start + row0, qr1 = q_start + row1;
                    unsigned int k0 = kv_start + col0, k1 = kv_start + col1;
                    if (qr0 >= k0 && qr0 - k0 >= sliding_window) acc_s[nt][0] = -1e30f;
                    if (qr0 >= k1 && qr0 - k1 >= sliding_window) acc_s[nt][1] = -1e30f;
                    if (qr1 >= k0 && qr1 - k0 >= sliding_window) acc_s[nt][2] = -1e30f;
                    if (qr1 >= k1 && qr1 - k1 >= sliding_window) acc_s[nt][3] = -1e30f;
                }
                if (col0 >= kv_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (col1 >= kv_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (row0 >= q_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (row1 >= q_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }

            float rmax0 = -1e30f, rmax1 = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
                rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

            // Guard fully-masked tiles (e.g. causal tail or SWA prefix tiles) so
            // `-1e30f - (-1e30f)` never produces `exp(0) = 1.0f`.
            const bool has_valid0 = (rmax0 > -1e29f);
            float m_new0 = has_valid0 ? fmaxf(m_r0, rmax0) : m_r0;
            float exp_old0 = (has_valid0 && m_r0 > -1e29f)
                ? __expf(m_r0 - m_new0)
                : (has_valid0 ? 0.0f : 1.0f);
            l_r0 *= exp_old0;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_old0; acc_o[i][1] *= exp_old0;
            }
            m_r0 = m_new0;

            const bool has_valid1 = (rmax1 > -1e29f);
            float m_new1 = has_valid1 ? fmaxf(m_r1, rmax1) : m_r1;
            float exp_old1 = (has_valid1 && m_r1 > -1e29f)
                ? __expf(m_r1 - m_new1)
                : (has_valid1 ? 0.0f : 1.0f);
            l_r1 *= exp_old1;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][2] *= exp_old1; acc_o[i][3] *= exp_old1;
            }
            m_r1 = m_new1;

            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                float p00 = (has_valid0 && acc_s[nt][0] > -1e29f) ? __expf(acc_s[nt][0] - m_r0) : 0.0f;
                float p01 = (has_valid0 && acc_s[nt][1] > -1e29f) ? __expf(acc_s[nt][1] - m_r0) : 0.0f;
                float p10 = (has_valid1 && acc_s[nt][2] > -1e29f) ? __expf(acc_s[nt][2] - m_r1) : 0.0f;
                float p11 = (has_valid1 && acc_s[nt][3] > -1e29f) ? __expf(acc_s[nt][3] - m_r1) : 0.0f;
                sum0 += p00 + p01;
                sum1 += p10 + p11;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                smem_P64[row0][col0]     = __float2bfloat16(p00);
                smem_P64[row0][col0 + 1] = __float2bfloat16(p01);
                smem_P64[row1][col0]     = __float2bfloat16(p10);
                smem_P64[row1][col0 + 1] = __float2bfloat16(p11);
            }

            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0;
            l_r1 += sum1;

            if (tid_in_group == 0) {
                smem_ml64[row0][0] = m_r0; smem_ml64[row0][1] = l_r0;
                smem_ml64[row1][0] = m_r1; smem_ml64[row1][1] = l_r1;
            }
        }

        // Wait for V tile load
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Warps 4-7: rescale accumulators to match current m
        if (warp_id >= 4) {
            unsigned int row0 = pv_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            float cur_m0 = smem_ml64[row0][0];
            float cur_m1 = smem_ml64[row1][0];
            float exp_r0 = (cur_m0 > -1e29f && m_r0 > -1e29f) ? __expf(m_r0 - cur_m0) : 0.0f;
            float exp_r1 = (cur_m1 > -1e29f && m_r1 > -1e29f) ? __expf(m_r1 - cur_m1) : 0.0f;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_r0; acc_o[i][1] *= exp_r0;
                acc_o[i][2] *= exp_r1; acc_o[i][3] *= exp_r1;
            }
            m_r0 = cur_m0; m_r1 = cur_m1;
        }

        // === Preload K[i+1] (overlaps with PV below) ===
        if (kv_block + 1 < num_kv_blocks) {
            unsigned int next_kv_start = (kv_block + 1) * BC;
            const unsigned int chunks_per_row_k = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 256) {
                unsigned int row = idx / chunks_per_row_k;
                unsigned int chunk = idx % chunks_per_row_k;
                unsigned int col = chunk * 8;
                unsigned int k_row = next_kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K64[1 - buf][row][col]);

                if (k_row < seq_len) {
                    const void* gmem = (const void*)&K_batch[(unsigned long long)k_row * kv_seq_stride + (unsigned long long)kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_K64[1 - buf][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // === PV MMA (all 8 warps, 8 n-tiles each) ===
        {
            const unsigned short* sP = (const unsigned short*)smem_P64;
            const unsigned short* sV = (const unsigned short*)smem_V64;

            #pragma unroll
            for (unsigned int ks = 0; ks < 2; ks++) {
                unsigned int k_off = ks * 16;
                unsigned int ar0 = pv_warp_m + group_id;
                unsigned int ar1 = pv_warp_m + group_id + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = k_off + tid_in_group * 2 + 8;

                unsigned int a0 = *(const unsigned int*)&sP[ar0 * p_smem_stride64 + ac0];
                unsigned int a1 = *(const unsigned int*)&sP[ar1 * p_smem_stride64 + ac0];
                unsigned int a2 = *(const unsigned int*)&sP[ar0 * p_smem_stride64 + ac1];
                unsigned int a3 = *(const unsigned int*)&sP[ar1 * p_smem_stride64 + ac1];

                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
                    unsigned int n_col = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k_off + tid_in_group * 2 + 8;

                    unsigned int b0 = ((unsigned int)sV[(k0 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV[k0 * HDIM_PAD + n_col];
                    unsigned int b1 = ((unsigned int)sV[(k1 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV[k1 * HDIM_PAD + n_col];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]),
                          "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_o[nt][0]), "f"(acc_o[nt][1]),
                          "f"(acc_o[nt][2]), "f"(acc_o[nt][3])
                    );
                }
            }
        }

        // Wait for K[i+1] prefetch
        if (kv_block + 1 < num_kv_blocks) {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
    }

    // === Final normalization and store ===
    {
        unsigned int row0 = pv_warp_m + group_id;
        unsigned int row1 = row0 + 8;

        float inv_l0, inv_l1;
        if (warp_id < 4) {
            inv_l0 = (l_r0 > 0.0f) ? (1.0f / l_r0) : 0.0f;
            inv_l1 = (l_r1 > 0.0f) ? (1.0f / l_r1) : 0.0f;
        } else {
            float l0 = smem_ml64[row0][1];
            float l1 = smem_ml64[row1][1];
            inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
            inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
        }

        __nv_bfloat16* o_base = O_batch + q_head * head_dim;

        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            unsigned int col0 = (pv_n_start + nt) * 8 + tid_in_group * 2;
            unsigned int gr0 = q_start + row0;
            unsigned int gr1 = q_start + row1;

            if (gr0 < seq_len && row0 < q_len && col0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * inv_l0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * inv_l0));
                *(unsigned int*)&o_base[(unsigned long long)gr0 * q_seq_stride + col0] = lo | (hi << 16);
            }
            if (gr1 < seq_len && row1 < q_len && col0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * inv_l1));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * inv_l1));
                *(unsigned int*)&o_base[(unsigned long long)gr1 * q_seq_stride + col0] = lo | (hi << 16);
            }
        }
    }
}

// ============================================================================
// Split-KV H=128 Flash Attention for 2-stage CUDA-graphed DFlash2 drafting.
//
// Reads `eff_ctx = *eff_ctx_dev` from device memory (zero host scalar, 100%
// CUDA-graph-capture-safe) and attends query rows `[0..m_rows)` directly over
// persistent post-k_norm/post-RoPE `K_ctx`/`V_ctx[0..eff_ctx)` concatenated
// virtually with `K_noise`/`V_noise[0..m_rows)`, eliminating both O(eff_ctx)
// K/V D2D copies and O(eff_ctx) re-RoPE on every draft step.
//
// SWA optimization skips strictly-masked prefix KV blocks `[0..kv_block_start)`
// where `kv_block_start = (eff_ctx - sliding_window) / BC` when
// `sliding_window > 0 && eff_ctx > sliding_window`.
//
// Shared memory (BR=32, BC=32, HDIM=128, HDIM_PAD=136):
//   Q:   [32][136]    BF16 =  8.5 KB
//   K:   [2][32][136] BF16 = 17.0 KB
//   V:   [32][136]    BF16 =  8.5 KB
//   P:   [32][40]     BF16 =  2.5 KB
//   m/l: [32][2]      FP32 =  0.25 KB
//   Total: ~36.8 KB
//
// Grid:  [num_q_heads, ceil(m_rows / BR), 1]
// Block: [128, 1, 1]
// ============================================================================
extern "C" __global__ void inferspark_prefill_split_ctx_h128(
    const __nv_bfloat16* __restrict__ Q_noise,
    const __nv_bfloat16* __restrict__ K_ctx,
    const __nv_bfloat16* __restrict__ V_ctx,
    const __nv_bfloat16* __restrict__ K_noise,
    const __nv_bfloat16* __restrict__ V_noise,
    __nv_bfloat16* __restrict__ O_noise,
    const unsigned int* __restrict__ eff_ctx_dev,
    const unsigned int m_rows,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;

    const unsigned int q_tile_start = q_block * BR;
    if (q_tile_start >= m_rows) return;
    const unsigned int q_tile_end = min(q_tile_start + BR, m_rows);
    const unsigned int q_len = q_tile_end - q_tile_start;

    const unsigned int eff_ctx = *eff_ctx_dev;
    const unsigned int total_kv = eff_ctx + m_rows;
    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned long long q_seq_stride = (unsigned long long)num_q_heads * (unsigned long long)HDIM;
    const unsigned long long kv_seq_stride = (unsigned long long)num_kv_heads * (unsigned long long)HDIM;
    const unsigned long long q_head_off = (unsigned long long)q_head * (unsigned long long)HDIM;
    const unsigned long long kv_head_off = (unsigned long long)kv_head * (unsigned long long)HDIM;

    __shared__ __nv_bfloat16 smem_Q[BR][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[2][BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P[BR][BC + PAD_P];
    __shared__ float smem_ml[BR][2];

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;

    const unsigned int qk_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * N_TILES_PER_WARP;

    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }

    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    const unsigned int p_smem_stride = BC + PAD_P;

    unsigned int num_kv_blocks = (total_kv + BC - 1) / BC;
    if (causal) {
        unsigned int max_kv_block = (eff_ctx + q_tile_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1);
    }
    unsigned int kv_block_start = 0;
    if (sliding_window > 0 && eff_ctx > sliding_window) {
        kv_block_start = (eff_ctx - sliding_window) / BC;
    }

    // === Merged Q_noise + K[kv_block_start] load (single cp.async commit group) ===
    {
        const unsigned int chunks_per_row = HDIM / 8;

        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
            unsigned int row = idx / chunks_per_row;
            unsigned int chunk = idx % chunks_per_row;
            unsigned int col = chunk * 8;
            unsigned int q_row = q_tile_start + row;
            unsigned int smem_addr = __cvta_generic_to_shared(&smem_Q[row][col]);

            if (q_row < m_rows) {
                const void* gmem = (const void*)&Q_noise[(unsigned long long)q_row * q_seq_stride + q_head_off + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }

        if (kv_block_start < num_kv_blocks) {
            const unsigned int first_kv_start = kv_block_start * BC;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 128) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int k_row = first_kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K[0][row][col]);

                if (k_row < total_kv) {
                    const __nv_bfloat16* src = (k_row < eff_ctx)
                        ? &K_ctx[(unsigned long long)k_row * kv_seq_stride + kv_head_off + col]
                        : &K_noise[(unsigned long long)(k_row - eff_ctx) * kv_seq_stride + kv_head_off + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"((const void*)src));
                } else {
                    *((uint4*)&smem_K[0][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = kv_block_start; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, total_kv);
        unsigned int kv_len = kv_end - kv_start;
        unsigned int buf = (kv_block - kv_block_start) & 1;

        // === Async V load (overlaps with QK^T) ===
        {
            const unsigned int chunks_per_row = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 128) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int v_row = kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_V[row][col]);

                if (v_row < total_kv) {
                    const __nv_bfloat16* src = (v_row < eff_ctx)
                        ? &V_ctx[(unsigned long long)v_row * kv_seq_stride + kv_head_off + col]
                        : &V_noise[(unsigned long long)(v_row - eff_ctx) * kv_seq_stride + kv_head_off + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"((const void*)src));
                } else {
                    *((uint4*)&smem_V[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // === QK^T (warps 0-1, register-based) ===
        float acc_s[BC / 8][4];
        if (warp_id < 2) {
            #pragma unroll
            for (int i = 0; i < (int)(BC / 8); i++) {
                acc_s[i][0] = 0.0f; acc_s[i][1] = 0.0f;
                acc_s[i][2] = 0.0f; acc_s[i][3] = 0.0f;
            }

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM / 16); ks++) {
                unsigned int k_base = ks * 16;

                const unsigned short* sQ_u16 = (const unsigned short*)smem_Q;
                unsigned int ar0 = qk_warp_m + group_id;
                unsigned int ar1 = qk_warp_m + group_id + 8;
                unsigned int ac0 = k_base + tid_in_group * 2;
                unsigned int ac1 = k_base + tid_in_group * 2 + 8;
                unsigned int a0 = *(const unsigned int*)&sQ_u16[ar0 * HDIM_PAD + ac0];
                unsigned int a1 = *(const unsigned int*)&sQ_u16[ar1 * HDIM_PAD + ac0];
                unsigned int a2 = *(const unsigned int*)&sQ_u16[ar0 * HDIM_PAD + ac1];
                unsigned int a3 = *(const unsigned int*)&sQ_u16[ar1 * HDIM_PAD + ac1];

                const unsigned short* sK_u16 = (const unsigned short*)smem_K[buf];
                #pragma unroll
                for (int nt = 0; nt < (int)(BC / 8); nt++) {
                    unsigned int n_col = nt * 8 + group_id;
                    unsigned int k0 = k_base + tid_in_group * 2;
                    unsigned int k1 = k_base + tid_in_group * 2 + 8;
                    unsigned int b0 = ((unsigned int)sK_u16[n_col * HDIM_PAD + k0 + 1] << 16) |
                                      (unsigned int)sK_u16[n_col * HDIM_PAD + k0];
                    unsigned int b1 = ((unsigned int)sK_u16[n_col * HDIM_PAD + k1 + 1] << 16) |
                                      (unsigned int)sK_u16[n_col * HDIM_PAD + k1];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]),
                          "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_s[nt][0]), "f"(acc_s[nt][1]),
                          "f"(acc_s[nt][2]), "f"(acc_s[nt][3])
                    );
                }
            }

            // === Register-based softmax ===
            unsigned int row0 = qk_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            unsigned int qr0 = eff_ctx + q_tile_start + row0;
            unsigned int qr1 = eff_ctx + q_tile_start + row1;

            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                acc_s[nt][0] *= inv_sqrt_d;
                acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d;
                acc_s[nt][3] *= inv_sqrt_d;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                unsigned int col1 = col0 + 1;

                if (causal) {
                    if (kv_start + col0 > qr0) acc_s[nt][0] = -1e30f;
                    if (kv_start + col1 > qr0) acc_s[nt][1] = -1e30f;
                    if (kv_start + col0 > qr1) acc_s[nt][2] = -1e30f;
                    if (kv_start + col1 > qr1) acc_s[nt][3] = -1e30f;
                }
                if (sliding_window > 0) {
                    unsigned int k0 = kv_start + col0, k1 = kv_start + col1;
                    if (qr0 >= k0 && qr0 - k0 >= sliding_window) acc_s[nt][0] = -1e30f;
                    if (qr0 >= k1 && qr0 - k1 >= sliding_window) acc_s[nt][1] = -1e30f;
                    if (qr1 >= k0 && qr1 - k0 >= sliding_window) acc_s[nt][2] = -1e30f;
                    if (qr1 >= k1 && qr1 - k1 >= sliding_window) acc_s[nt][3] = -1e30f;
                }
                if (col0 >= kv_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (col1 >= kv_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (row0 >= q_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (row1 >= q_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }

            float rmax0 = -1e30f, rmax1 = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
                rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

            // Guard fully-masked tiles (e.g. causal tail or SWA prefix tiles) so
            // `-1e30f - (-1e30f)` never produces `exp(0) = 1.0f`.
            const bool has_valid0 = (rmax0 > -1e29f);
            float m_new0 = has_valid0 ? fmaxf(m_r0, rmax0) : m_r0;
            float exp_old0 = (has_valid0 && m_r0 > -1e29f)
                ? __expf(m_r0 - m_new0)
                : (has_valid0 ? 0.0f : 1.0f);
            l_r0 *= exp_old0;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_old0; acc_o[i][1] *= exp_old0;
            }
            m_r0 = m_new0;

            const bool has_valid1 = (rmax1 > -1e29f);
            float m_new1 = has_valid1 ? fmaxf(m_r1, rmax1) : m_r1;
            float exp_old1 = (has_valid1 && m_r1 > -1e29f)
                ? __expf(m_r1 - m_new1)
                : (has_valid1 ? 0.0f : 1.0f);
            l_r1 *= exp_old1;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][2] *= exp_old1; acc_o[i][3] *= exp_old1;
            }
            m_r1 = m_new1;

            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                float p00 = (has_valid0 && acc_s[nt][0] > -1e29f) ? __expf(acc_s[nt][0] - m_r0) : 0.0f;
                float p01 = (has_valid0 && acc_s[nt][1] > -1e29f) ? __expf(acc_s[nt][1] - m_r0) : 0.0f;
                float p10 = (has_valid1 && acc_s[nt][2] > -1e29f) ? __expf(acc_s[nt][2] - m_r1) : 0.0f;
                float p11 = (has_valid1 && acc_s[nt][3] > -1e29f) ? __expf(acc_s[nt][3] - m_r1) : 0.0f;
                sum0 += p00 + p01;
                sum1 += p10 + p11;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                smem_P[row0][col0]     = __float2bfloat16(p00);
                smem_P[row0][col0 + 1] = __float2bfloat16(p01);
                smem_P[row1][col0]     = __float2bfloat16(p10);
                smem_P[row1][col0 + 1] = __float2bfloat16(p11);
            }

            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0;
            l_r1 += sum1;

            if (tid_in_group == 0) {
                smem_ml[row0][0] = m_r0; smem_ml[row0][1] = l_r0;
                smem_ml[row1][0] = m_r1; smem_ml[row1][1] = l_r1;
            }
        }

        // Wait for V tile load
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Warps 2-3: rescale accumulators to match current m
        if (warp_id >= 2) {
            unsigned int row0 = pv_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            float cur_m0 = smem_ml[row0][0];
            float cur_m1 = smem_ml[row1][0];
            float exp_r0 = (cur_m0 > -1e29f && m_r0 > -1e29f) ? __expf(m_r0 - cur_m0) : 0.0f;
            float exp_r1 = (cur_m1 > -1e29f && m_r1 > -1e29f) ? __expf(m_r1 - cur_m1) : 0.0f;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_r0; acc_o[i][1] *= exp_r0;
                acc_o[i][2] *= exp_r1; acc_o[i][3] *= exp_r1;
            }
            m_r0 = cur_m0; m_r1 = cur_m1;
        }

        // === Preload K[i+1] into smem_K[1-buf] (overlaps with PV) ===
        if (kv_block + 1 < num_kv_blocks) {
            unsigned int next_kv_start = (kv_block + 1) * BC;
            const unsigned int chunks_per_row_k = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 128) {
                unsigned int row = idx / chunks_per_row_k;
                unsigned int chunk = idx % chunks_per_row_k;
                unsigned int col = chunk * 8;
                unsigned int k_row = next_kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K[1 - buf][row][col]);

                if (k_row < total_kv) {
                    const __nv_bfloat16* src = (k_row < eff_ctx)
                        ? &K_ctx[(unsigned long long)k_row * kv_seq_stride + kv_head_off + col]
                        : &K_noise[(unsigned long long)(k_row - eff_ctx) * kv_seq_stride + kv_head_off + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"((const void*)src));
                } else {
                    *((uint4*)&smem_K[1 - buf][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // === PV MMA (all 4 warps, 8 n-tiles each) ===
        {
            #pragma unroll
            for (unsigned int ks = 0; ks < (BC / 16); ks++) {
                unsigned int k_off = ks * 16;

                const unsigned short* sP_u16 = (const unsigned short*)smem_P;
                unsigned int ar0 = pv_warp_m + group_id;
                unsigned int ar1 = pv_warp_m + group_id + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = k_off + tid_in_group * 2 + 8;
                unsigned int a0 = *(const unsigned int*)&sP_u16[ar0 * p_smem_stride + ac0];
                unsigned int a1 = *(const unsigned int*)&sP_u16[ar1 * p_smem_stride + ac0];
                unsigned int a2 = *(const unsigned int*)&sP_u16[ar0 * p_smem_stride + ac1];
                unsigned int a3 = *(const unsigned int*)&sP_u16[ar1 * p_smem_stride + ac1];

                const unsigned short* sV_u16 = (const unsigned short*)smem_V;

                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
                    unsigned int n_col = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k_off + tid_in_group * 2 + 8;
                    unsigned int b0 = ((unsigned int)sV_u16[(k0 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV_u16[k0 * HDIM_PAD + n_col];
                    unsigned int b1 = ((unsigned int)sV_u16[(k1 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV_u16[k1 * HDIM_PAD + n_col];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]),
                          "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_o[nt][0]), "f"(acc_o[nt][1]),
                          "f"(acc_o[nt][2]), "f"(acc_o[nt][3])
                    );
                }
            }
        }

        if (kv_block + 1 < num_kv_blocks) {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
    }

    // === Final normalization and store directly to O_noise[0..m_rows) ===
    {
        unsigned int row0 = pv_warp_m + group_id;
        unsigned int row1 = row0 + 8;

        float inv_l0, inv_l1;
        if (warp_id < 2) {
            inv_l0 = (l_r0 > 0.0f) ? (1.0f / l_r0) : 0.0f;
            inv_l1 = (l_r1 > 0.0f) ? (1.0f / l_r1) : 0.0f;
        } else {
            float l0 = smem_ml[row0][1];
            float l1 = smem_ml[row1][1];
            inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
            inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
        }

        __nv_bfloat16* o_base = O_noise + q_head_off;

        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            unsigned int col0 = (pv_n_start + nt) * 8 + tid_in_group * 2;
            unsigned int nr0 = q_tile_start + row0;
            unsigned int nr1 = q_tile_start + row1;

            if (nr0 < m_rows && row0 < q_len) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * inv_l0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * inv_l0));
                *(unsigned int*)&o_base[(unsigned long long)nr0 * q_seq_stride + col0] = lo | (hi << 16);
            }
            if (nr1 < m_rows && row1 < q_len) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * inv_l1));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * inv_l1));
                *(unsigned int*)&o_base[(unsigned long long)nr1 * q_seq_stride + col0] = lo | (hi << 16);
            }
        }
    }
}

// ============================================================================
// Batched Split-KV H=128 Flash Attention with circular ring context support.
//
// Launches all active sequences in one grid (`blockIdx.z = b`), reading per-
// sequence context cache base pointers (`K_ctx_ptrs[b]`, `V_ctx_ptrs[b]`),
// effective context length (`eff_ctx_dev[b]`), and circular ring head offset
// (`ring_start_dev[b]`) from device memory.
//
// Grid:  [num_q_heads, ceil(m_rows / BR), batch_size]
// Block: [128, 1, 1]
// ============================================================================
extern "C" __global__ void inferspark_prefill_split_ctx_h128_batched(
    const __nv_bfloat16* __restrict__ Q_noise,
    const unsigned long long* __restrict__ K_ctx_ptrs,
    const unsigned long long* __restrict__ V_ctx_ptrs,
    const __nv_bfloat16* __restrict__ K_noise,
    const __nv_bfloat16* __restrict__ V_noise,
    __nv_bfloat16* __restrict__ O_noise,
    const unsigned int* __restrict__ eff_ctx_dev,
    const unsigned int* __restrict__ ring_start_dev,
    const unsigned int ring_cap,
    const unsigned int m_rows,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int b = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;

    const unsigned int q_tile_start = q_block * BR;
    if (q_tile_start >= m_rows) return;
    const unsigned int q_tile_end = min(q_tile_start + BR, m_rows);
    const unsigned int q_len = q_tile_end - q_tile_start;

    const unsigned int eff_ctx = eff_ctx_dev[b];
    const unsigned int ring_start = ring_start_dev[b];
    const unsigned int total_kv = eff_ctx + m_rows;
    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned long long q_seq_stride = (unsigned long long)num_q_heads * (unsigned long long)HDIM;
    const unsigned long long kv_seq_stride = (unsigned long long)num_kv_heads * (unsigned long long)HDIM;
    const unsigned long long q_head_off = (unsigned long long)q_head * (unsigned long long)HDIM;
    const unsigned long long kv_head_off = (unsigned long long)kv_head * (unsigned long long)HDIM;

    const __nv_bfloat16* Q_s = Q_noise + (unsigned long long)b * (unsigned long long)m_rows * q_seq_stride;
    const __nv_bfloat16* K_s = K_noise + (unsigned long long)b * (unsigned long long)m_rows * kv_seq_stride;
    const __nv_bfloat16* V_s = V_noise + (unsigned long long)b * (unsigned long long)m_rows * kv_seq_stride;
    __nv_bfloat16* O_s = O_noise + (unsigned long long)b * (unsigned long long)m_rows * q_seq_stride;
    const __nv_bfloat16* K_ctx = (const __nv_bfloat16*)K_ctx_ptrs[b];
    const __nv_bfloat16* V_ctx = (const __nv_bfloat16*)V_ctx_ptrs[b];

    __shared__ __nv_bfloat16 smem_Q[BR][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[2][BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P[BR][BC + PAD_P];
    __shared__ float smem_ml[BR][2];

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;

    const unsigned int qk_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * N_TILES_PER_WARP;

    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }

    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    const unsigned int p_smem_stride = BC + PAD_P;

    unsigned int num_kv_blocks = (total_kv + BC - 1) / BC;
    if (causal) {
        unsigned int max_kv_block = (eff_ctx + q_tile_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1);
    }
    unsigned int kv_block_start = 0;
    if (sliding_window > 0 && eff_ctx > sliding_window) {
        kv_block_start = (eff_ctx - sliding_window) / BC;
    }

    // === Merged Q_s + K[kv_block_start] load (single cp.async commit group) ===
    {
        const unsigned int chunks_per_row = HDIM / 8;

        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
            unsigned int row = idx / chunks_per_row;
            unsigned int chunk = idx % chunks_per_row;
            unsigned int col = chunk * 8;
            unsigned int q_row = q_tile_start + row;
            unsigned int smem_addr = __cvta_generic_to_shared(&smem_Q[row][col]);

            if (q_row < m_rows) {
                const void* gmem = (const void*)&Q_s[(unsigned long long)q_row * q_seq_stride + q_head_off + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }

        if (kv_block_start < num_kv_blocks) {
            const unsigned int first_kv_start = kv_block_start * BC;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 128) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int k_row = first_kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K[0][row][col]);

                if (k_row < total_kv) {
                    const __nv_bfloat16* src;
                    if (k_row < eff_ctx) {
                        unsigned int phys = ring_start + k_row;
                        if (phys >= ring_cap) phys -= ring_cap;
                        src = &K_ctx[(unsigned long long)phys * kv_seq_stride + kv_head_off + col];
                    } else {
                        src = &K_s[(unsigned long long)(k_row - eff_ctx) * kv_seq_stride + kv_head_off + col];
                    }
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"((const void*)src));
                } else {
                    *((uint4*)&smem_K[0][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = kv_block_start; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, total_kv);
        unsigned int kv_len = kv_end - kv_start;
        unsigned int buf = (kv_block - kv_block_start) & 1;

        // === Async V load (overlaps with QK^T) ===
        {
            const unsigned int chunks_per_row = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 128) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int v_row = kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_V[row][col]);

                if (v_row < total_kv) {
                    const __nv_bfloat16* src;
                    if (v_row < eff_ctx) {
                        unsigned int phys = ring_start + v_row;
                        if (phys >= ring_cap) phys -= ring_cap;
                        src = &V_ctx[(unsigned long long)phys * kv_seq_stride + kv_head_off + col];
                    } else {
                        src = &V_s[(unsigned long long)(v_row - eff_ctx) * kv_seq_stride + kv_head_off + col];
                    }
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"((const void*)src));
                } else {
                    *((uint4*)&smem_V[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // === QK^T (warps 0-1, register-based) ===
        float acc_s[BC / 8][4];
        if (warp_id < 2) {
            #pragma unroll
            for (int i = 0; i < (int)(BC / 8); i++) {
                acc_s[i][0] = 0.0f; acc_s[i][1] = 0.0f;
                acc_s[i][2] = 0.0f; acc_s[i][3] = 0.0f;
            }

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM / 16); ks++) {
                unsigned int k_base = ks * 16;

                const unsigned short* sQ_u16 = (const unsigned short*)smem_Q;
                unsigned int ar0 = qk_warp_m + group_id;
                unsigned int ar1 = qk_warp_m + group_id + 8;
                unsigned int ac0 = k_base + tid_in_group * 2;
                unsigned int ac1 = k_base + tid_in_group * 2 + 8;
                unsigned int a0 = *(const unsigned int*)&sQ_u16[ar0 * HDIM_PAD + ac0];
                unsigned int a1 = *(const unsigned int*)&sQ_u16[ar1 * HDIM_PAD + ac0];
                unsigned int a2 = *(const unsigned int*)&sQ_u16[ar0 * HDIM_PAD + ac1];
                unsigned int a3 = *(const unsigned int*)&sQ_u16[ar1 * HDIM_PAD + ac1];

                const unsigned short* sK_u16 = (const unsigned short*)smem_K[buf];
                #pragma unroll
                for (int nt = 0; nt < (int)(BC / 8); nt++) {
                    unsigned int n_col = nt * 8 + group_id;
                    unsigned int k0 = k_base + tid_in_group * 2;
                    unsigned int k1 = k_base + tid_in_group * 2 + 8;
                    unsigned int b0 = ((unsigned int)sK_u16[n_col * HDIM_PAD + k0 + 1] << 16) |
                                      (unsigned int)sK_u16[n_col * HDIM_PAD + k0];
                    unsigned int b1 = ((unsigned int)sK_u16[n_col * HDIM_PAD + k1 + 1] << 16) |
                                      (unsigned int)sK_u16[n_col * HDIM_PAD + k1];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]),
                          "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_s[nt][0]), "f"(acc_s[nt][1]),
                          "f"(acc_s[nt][2]), "f"(acc_s[nt][3])
                    );
                }
            }

            // === Register-based softmax ===
            unsigned int row0 = qk_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            unsigned int qr0 = eff_ctx + q_tile_start + row0;
            unsigned int qr1 = eff_ctx + q_tile_start + row1;

            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                acc_s[nt][0] *= inv_sqrt_d;
                acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d;
                acc_s[nt][3] *= inv_sqrt_d;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                unsigned int col1 = col0 + 1;

                if (causal) {
                    if (kv_start + col0 > qr0) acc_s[nt][0] = -1e30f;
                    if (kv_start + col1 > qr0) acc_s[nt][1] = -1e30f;
                    if (kv_start + col0 > qr1) acc_s[nt][2] = -1e30f;
                    if (kv_start + col1 > qr1) acc_s[nt][3] = -1e30f;
                }
                if (sliding_window > 0) {
                    unsigned int k0 = kv_start + col0, k1 = kv_start + col1;
                    if (qr0 >= k0 && qr0 - k0 >= sliding_window) acc_s[nt][0] = -1e30f;
                    if (qr0 >= k1 && qr0 - k1 >= sliding_window) acc_s[nt][1] = -1e30f;
                    if (qr1 >= k0 && qr1 - k0 >= sliding_window) acc_s[nt][2] = -1e30f;
                    if (qr1 >= k1 && qr1 - k1 >= sliding_window) acc_s[nt][3] = -1e30f;
                }
                if (col0 >= kv_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (col1 >= kv_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (row0 >= q_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (row1 >= q_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }

            float rmax0 = -1e30f, rmax1 = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
                rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

            // Guard fully-masked tiles (e.g. causal tail or SWA prefix tiles) so
            // `-1e30f - (-1e30f)` never produces `exp(0) = 1.0f`.
            const bool has_valid0 = (rmax0 > -1e29f);
            float m_new0 = has_valid0 ? fmaxf(m_r0, rmax0) : m_r0;
            float exp_old0 = (has_valid0 && m_r0 > -1e29f)
                ? __expf(m_r0 - m_new0)
                : (has_valid0 ? 0.0f : 1.0f);
            l_r0 *= exp_old0;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_old0; acc_o[i][1] *= exp_old0;
            }
            m_r0 = m_new0;

            const bool has_valid1 = (rmax1 > -1e29f);
            float m_new1 = has_valid1 ? fmaxf(m_r1, rmax1) : m_r1;
            float exp_old1 = (has_valid1 && m_r1 > -1e29f)
                ? __expf(m_r1 - m_new1)
                : (has_valid1 ? 0.0f : 1.0f);
            l_r1 *= exp_old1;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][2] *= exp_old1; acc_o[i][3] *= exp_old1;
            }
            m_r1 = m_new1;

            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                float p00 = (has_valid0 && acc_s[nt][0] > -1e29f) ? __expf(acc_s[nt][0] - m_r0) : 0.0f;
                float p01 = (has_valid0 && acc_s[nt][1] > -1e29f) ? __expf(acc_s[nt][1] - m_r0) : 0.0f;
                float p10 = (has_valid1 && acc_s[nt][2] > -1e29f) ? __expf(acc_s[nt][2] - m_r1) : 0.0f;
                float p11 = (has_valid1 && acc_s[nt][3] > -1e29f) ? __expf(acc_s[nt][3] - m_r1) : 0.0f;
                sum0 += p00 + p01;
                sum1 += p10 + p11;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                smem_P[row0][col0]     = __float2bfloat16(p00);
                smem_P[row0][col0 + 1] = __float2bfloat16(p01);
                smem_P[row1][col0]     = __float2bfloat16(p10);
                smem_P[row1][col0 + 1] = __float2bfloat16(p11);
            }

            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0;
            l_r1 += sum1;

            if (tid_in_group == 0) {
                smem_ml[row0][0] = m_r0; smem_ml[row0][1] = l_r0;
                smem_ml[row1][0] = m_r1; smem_ml[row1][1] = l_r1;
            }
        }

        // Wait for V tile load
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Warps 2-3: rescale accumulators to match current m
        if (warp_id >= 2) {
            unsigned int row0 = pv_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            float cur_m0 = smem_ml[row0][0];
            float cur_m1 = smem_ml[row1][0];
            float exp_r0 = (cur_m0 > -1e29f && m_r0 > -1e29f) ? __expf(m_r0 - cur_m0) : 0.0f;
            float exp_r1 = (cur_m1 > -1e29f && m_r1 > -1e29f) ? __expf(m_r1 - cur_m1) : 0.0f;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_r0; acc_o[i][1] *= exp_r0;
                acc_o[i][2] *= exp_r1; acc_o[i][3] *= exp_r1;
            }
            m_r0 = cur_m0; m_r1 = cur_m1;
        }

        // === Preload K[i+1] into smem_K[1-buf] (overlaps with PV) ===
        if (kv_block + 1 < num_kv_blocks) {
            unsigned int next_kv_start = (kv_block + 1) * BC;
            const unsigned int chunks_per_row_k = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 128) {
                unsigned int row = idx / chunks_per_row_k;
                unsigned int chunk = idx % chunks_per_row_k;
                unsigned int col = chunk * 8;
                unsigned int k_row = next_kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K[1 - buf][row][col]);

                if (k_row < total_kv) {
                    const __nv_bfloat16* src;
                    if (k_row < eff_ctx) {
                        unsigned int phys = ring_start + k_row;
                        if (phys >= ring_cap) phys -= ring_cap;
                        src = &K_ctx[(unsigned long long)phys * kv_seq_stride + kv_head_off + col];
                    } else {
                        src = &K_s[(unsigned long long)(k_row - eff_ctx) * kv_seq_stride + kv_head_off + col];
                    }
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"((const void*)src));
                } else {
                    *((uint4*)&smem_K[1 - buf][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // === PV MMA (all 4 warps, 8 n-tiles each) ===
        {
            #pragma unroll
            for (unsigned int ks = 0; ks < (BC / 16); ks++) {
                unsigned int k_off = ks * 16;

                const unsigned short* sP_u16 = (const unsigned short*)smem_P;
                unsigned int ar0 = pv_warp_m + group_id;
                unsigned int ar1 = pv_warp_m + group_id + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = k_off + tid_in_group * 2 + 8;
                unsigned int a0 = *(const unsigned int*)&sP_u16[ar0 * p_smem_stride + ac0];
                unsigned int a1 = *(const unsigned int*)&sP_u16[ar1 * p_smem_stride + ac0];
                unsigned int a2 = *(const unsigned int*)&sP_u16[ar0 * p_smem_stride + ac1];
                unsigned int a3 = *(const unsigned int*)&sP_u16[ar1 * p_smem_stride + ac1];

                const unsigned short* sV_u16 = (const unsigned short*)smem_V;

                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
                    unsigned int n_col = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k_off + tid_in_group * 2 + 8;
                    unsigned int b0 = ((unsigned int)sV_u16[(k0 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV_u16[k0 * HDIM_PAD + n_col];
                    unsigned int b1 = ((unsigned int)sV_u16[(k1 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV_u16[k1 * HDIM_PAD + n_col];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]),
                          "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_o[nt][0]), "f"(acc_o[nt][1]),
                          "f"(acc_o[nt][2]), "f"(acc_o[nt][3])
                    );
                }
            }
        }

        if (kv_block + 1 < num_kv_blocks) {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
    }

    // === Final normalization and store directly to O_s[0..m_rows) ===
    {
        unsigned int row0 = pv_warp_m + group_id;
        unsigned int row1 = row0 + 8;

        float inv_l0, inv_l1;
        if (warp_id < 2) {
            inv_l0 = (l_r0 > 0.0f) ? (1.0f / l_r0) : 0.0f;
            inv_l1 = (l_r1 > 0.0f) ? (1.0f / l_r1) : 0.0f;
        } else {
            float l0 = smem_ml[row0][1];
            float l1 = smem_ml[row1][1];
            inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
            inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
        }

        __nv_bfloat16* o_base = O_s + q_head_off;

        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            unsigned int col0 = (pv_n_start + nt) * 8 + tid_in_group * 2;
            unsigned int nr0 = q_tile_start + row0;
            unsigned int nr1 = q_tile_start + row1;

            if (nr0 < m_rows && row0 < q_len) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * inv_l0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * inv_l0));
                *(unsigned int*)&o_base[(unsigned long long)nr0 * q_seq_stride + col0] = lo | (hi << 16);
            }
            if (nr1 < m_rows && row1 < q_len) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * inv_l1));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * inv_l1));
                *(unsigned int*)&o_base[(unsigned long long)nr1 * q_seq_stride + col0] = lo | (hi << 16);
            }
        }
    }
}

// ============================================================================
// Split-KV H=128 Flash Attention + Lightweight Log-Sum-Exp Reduce for wide-SM
// DFlash2 drafting (saturates 188+ SMs at C=1..4 when num_q_heads * B < SMs).
//
// Workspace layout (`float* workspace`, 16-byte aligned):
//   total_rows = batch_size * num_splits * num_q_heads * m_rows
//   partial_o  = workspace[0 .. total_rows * 128)        (unnormalized FP32 O)
//   partial_ml = float2*(workspace + total_rows * 128)   ((m, l) per row)
//   Total bytes = total_rows * (128 + 2) * sizeof(float).
// ============================================================================
template <bool BATCHED_RING>
__device__ __forceinline__ void inferspark_prefill_split_ctx_h128_splitkv_impl(
    const __nv_bfloat16* __restrict__ Q_noise,
    const __nv_bfloat16* __restrict__ K_ctx_linear,
    const __nv_bfloat16* __restrict__ V_ctx_linear,
    const unsigned long long* __restrict__ K_ctx_ptrs,
    const unsigned long long* __restrict__ V_ctx_ptrs,
    const __nv_bfloat16* __restrict__ K_noise,
    const __nv_bfloat16* __restrict__ V_noise,
    float* __restrict__ workspace,
    const unsigned int* __restrict__ eff_ctx_dev,
    const unsigned int* __restrict__ ring_start_dev,
    const unsigned int ring_cap,
    const unsigned int m_rows,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int batch_size,
    const unsigned int num_splits,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int bz = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads || num_splits == 0u || batch_size == 0u) return;
    const unsigned int b = BATCHED_RING ? (bz / num_splits) : 0u;
    const unsigned int split_idx = BATCHED_RING ? (bz % num_splits) : bz;
    if (b >= batch_size || split_idx >= num_splits) return;

    const unsigned int q_tile_start = q_block * BR;
    if (q_tile_start >= m_rows) return;
    const unsigned int q_tile_end = min(q_tile_start + BR, m_rows);
    const unsigned int q_len = q_tile_end - q_tile_start;

    const unsigned int eff_ctx = BATCHED_RING ? eff_ctx_dev[b] : *eff_ctx_dev;
    const unsigned int ring_start = BATCHED_RING ? ring_start_dev[b] : 0u;
    const unsigned int total_kv = eff_ctx + m_rows;
    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned long long total_rows =
        (unsigned long long)batch_size * (unsigned long long)num_splits *
        (unsigned long long)num_q_heads * (unsigned long long)m_rows;
    float* const partial_o = workspace;
    float2* const partial_ml =
        reinterpret_cast<float2*>(workspace + total_rows * (unsigned long long)HDIM);
    const unsigned long long base_row_tile =
        (((unsigned long long)b * (unsigned long long)num_splits + (unsigned long long)split_idx) *
         (unsigned long long)num_q_heads + (unsigned long long)q_head) *
        (unsigned long long)m_rows + (unsigned long long)q_tile_start;

    unsigned int num_kv_blocks = (total_kv + BC - 1u) / BC;
    if (causal) {
        unsigned int max_kv_block = (eff_ctx + q_tile_end - 1u) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1u);
    }
    unsigned int kv_block_start = 0u;
    if (sliding_window > 0u && eff_ctx > sliding_window) {
        kv_block_start = (eff_ctx - sliding_window) / BC;
    }

    const unsigned int total_blocks =
        (num_kv_blocks > kv_block_start) ? (num_kv_blocks - kv_block_start) : 0u;
    const unsigned int q_blk = total_blocks / num_splits;
    const unsigned int r_blk = total_blocks % num_splits;
    const unsigned int my_block_start = kv_block_start + split_idx * q_blk + min(split_idx, r_blk);
    const unsigned int my_block_end = my_block_start + q_blk + (split_idx < r_blk ? 1u : 0u);

    if (my_block_start >= my_block_end) {
        for (unsigned int r = tid; r < q_len; r += 128u) {
            partial_ml[base_row_tile + (unsigned long long)r] = make_float2(-1e30f, 0.0f);
        }
        return;
    }

    const unsigned long long q_seq_stride = (unsigned long long)num_q_heads * (unsigned long long)HDIM;
    const unsigned long long kv_seq_stride = (unsigned long long)num_kv_heads * (unsigned long long)HDIM;
    const unsigned long long q_head_off = (unsigned long long)q_head * (unsigned long long)HDIM;
    const unsigned long long kv_head_off = (unsigned long long)kv_head * (unsigned long long)HDIM;

    const __nv_bfloat16* const Q_s = BATCHED_RING
        ? (Q_noise + (unsigned long long)b * (unsigned long long)m_rows * q_seq_stride)
        : Q_noise;
    const __nv_bfloat16* const K_s = BATCHED_RING
        ? (K_noise + (unsigned long long)b * (unsigned long long)m_rows * kv_seq_stride)
        : K_noise;
    const __nv_bfloat16* const V_s = BATCHED_RING
        ? (V_noise + (unsigned long long)b * (unsigned long long)m_rows * kv_seq_stride)
        : V_noise;
    const __nv_bfloat16* const K_ctx = BATCHED_RING
        ? reinterpret_cast<const __nv_bfloat16*>(K_ctx_ptrs[b])
        : K_ctx_linear;
    const __nv_bfloat16* const V_ctx = BATCHED_RING
        ? reinterpret_cast<const __nv_bfloat16*>(V_ctx_ptrs[b])
        : V_ctx_linear;

    __shared__ __nv_bfloat16 smem_Q[BR][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[2][BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P[BR][BC + PAD_P];
    __shared__ float smem_ml[BR][2];

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3u;
    const unsigned int qk_warp_m = (warp_id & 1u) * 16u;
    const unsigned int pv_warp_m = (warp_id & 1u) * 16u;
    const unsigned int pv_n_start = (warp_id >> 1) * N_TILES_PER_WARP;

    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }

    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;
    const unsigned int p_smem_stride = BC + PAD_P;

    auto kv_row_ptr = [&](const __nv_bfloat16* ctx_base, const __nv_bfloat16* noise_base,
                          unsigned int row_idx, unsigned int col) -> const __nv_bfloat16* {
        if (row_idx < eff_ctx) {
            unsigned int phys = row_idx;
            if constexpr (BATCHED_RING) {
                phys = ring_start + row_idx;
                if (phys >= ring_cap) phys -= ring_cap;
            }
            return &ctx_base[(unsigned long long)phys * kv_seq_stride + kv_head_off + col];
        }
        return &noise_base[(unsigned long long)(row_idx - eff_ctx) * kv_seq_stride + kv_head_off + col];
    };

    // === Merged Q_s + K[my_block_start] load ===
    {
        const unsigned int chunks_per_row = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128u) {
            unsigned int row = idx / chunks_per_row;
            unsigned int col = (idx % chunks_per_row) * 8u;
            unsigned int q_row = q_tile_start + row;
            unsigned int smem_addr = __cvta_generic_to_shared(&smem_Q[row][col]);
            if (q_row < m_rows) {
                const void* gmem = (const void*)&Q_s[(unsigned long long)q_row * q_seq_stride + q_head_off + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
        const unsigned int first_kv_start = my_block_start * BC;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 128u) {
            unsigned int row = idx / chunks_per_row;
            unsigned int col = (idx % chunks_per_row) * 8u;
            unsigned int k_row = first_kv_start + row;
            unsigned int smem_addr = __cvta_generic_to_shared(&smem_K[0][row][col]);
            if (k_row < total_kv) {
                const __nv_bfloat16* src = kv_row_ptr(K_ctx, K_s, k_row, col);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"((const void*)src));
            } else {
                *((uint4*)&smem_K[0][row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = my_block_start; kv_block < my_block_end; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, total_kv);
        unsigned int kv_len = kv_end - kv_start;
        unsigned int buf = (kv_block - my_block_start) & 1u;

        // === Async V load (overlaps with QK^T) ===
        {
            const unsigned int chunks_per_row = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 128u) {
                unsigned int row = idx / chunks_per_row;
                unsigned int col = (idx % chunks_per_row) * 8u;
                unsigned int v_row = kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_V[row][col]);
                if (v_row < total_kv) {
                    const __nv_bfloat16* src = kv_row_ptr(V_ctx, V_s, v_row, col);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"((const void*)src));
                } else {
                    *((uint4*)&smem_V[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // === QK^T + online softmax (warps 0-1) ===
        float acc_s[BC / 8][4];
        if (warp_id < 2u) {
            #pragma unroll
            for (int i = 0; i < (int)(BC / 8); i++) {
                acc_s[i][0] = 0.0f; acc_s[i][1] = 0.0f;
                acc_s[i][2] = 0.0f; acc_s[i][3] = 0.0f;
            }
            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM / 16); ks++) {
                unsigned int k_base = ks * 16u;
                const unsigned short* sQ_u16 = (const unsigned short*)smem_Q;
                unsigned int ar0 = qk_warp_m + group_id;
                unsigned int ar1 = qk_warp_m + group_id + 8u;
                unsigned int ac0 = k_base + tid_in_group * 2u;
                unsigned int ac1 = k_base + tid_in_group * 2u + 8u;
                unsigned int a0 = *(const unsigned int*)&sQ_u16[ar0 * HDIM_PAD + ac0];
                unsigned int a1 = *(const unsigned int*)&sQ_u16[ar1 * HDIM_PAD + ac0];
                unsigned int a2 = *(const unsigned int*)&sQ_u16[ar0 * HDIM_PAD + ac1];
                unsigned int a3 = *(const unsigned int*)&sQ_u16[ar1 * HDIM_PAD + ac1];

                const unsigned short* sK_u16 = (const unsigned short*)smem_K[buf];
                #pragma unroll
                for (int nt = 0; nt < (int)(BC / 8); nt++) {
                    unsigned int n_col = nt * 8u + group_id;
                    unsigned int k0 = k_base + tid_in_group * 2u;
                    unsigned int k1 = k_base + tid_in_group * 2u + 8u;
                    unsigned int b0 = ((unsigned int)sK_u16[n_col * HDIM_PAD + k0 + 1] << 16) |
                                      (unsigned int)sK_u16[n_col * HDIM_PAD + k0];
                    unsigned int b1 = ((unsigned int)sK_u16[n_col * HDIM_PAD + k1 + 1] << 16) |
                                      (unsigned int)sK_u16[n_col * HDIM_PAD + k1];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]),
                          "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_s[nt][0]), "f"(acc_s[nt][1]),
                          "f"(acc_s[nt][2]), "f"(acc_s[nt][3])
                    );
                }
            }

            unsigned int row0 = qk_warp_m + group_id;
            unsigned int row1 = row0 + 8u;
            unsigned int qr0 = eff_ctx + q_tile_start + row0;
            unsigned int qr1 = eff_ctx + q_tile_start + row1;

            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                acc_s[nt][0] *= inv_sqrt_d;
                acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d;
                acc_s[nt][3] *= inv_sqrt_d;

                unsigned int col0 = nt * 8u + tid_in_group * 2u;
                unsigned int col1 = col0 + 1u;

                if (causal) {
                    if (kv_start + col0 > qr0) acc_s[nt][0] = -1e30f;
                    if (kv_start + col1 > qr0) acc_s[nt][1] = -1e30f;
                    if (kv_start + col0 > qr1) acc_s[nt][2] = -1e30f;
                    if (kv_start + col1 > qr1) acc_s[nt][3] = -1e30f;
                }
                if (sliding_window > 0u) {
                    unsigned int k0 = kv_start + col0, k1 = kv_start + col1;
                    if (qr0 >= k0 && qr0 - k0 >= sliding_window) acc_s[nt][0] = -1e30f;
                    if (qr0 >= k1 && qr0 - k1 >= sliding_window) acc_s[nt][1] = -1e30f;
                    if (qr1 >= k0 && qr1 - k0 >= sliding_window) acc_s[nt][2] = -1e30f;
                    if (qr1 >= k1 && qr1 - k1 >= sliding_window) acc_s[nt][3] = -1e30f;
                }
                if (col0 >= kv_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (col1 >= kv_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (row0 >= q_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (row1 >= q_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }

            float rmax0 = -1e30f, rmax1 = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
                rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

            // Guard fully-masked tiles (e.g. causal tail or SWA prefix split) so
            // `-1e30f - (-1e30f)` never produces `exp(0) = 1.0f`.
            const bool has_valid0 = (rmax0 > -1e29f);
            float m_new0 = has_valid0 ? fmaxf(m_r0, rmax0) : m_r0;
            float exp_old0 = (has_valid0 && m_r0 > -1e29f)
                ? __expf(m_r0 - m_new0)
                : (has_valid0 ? 0.0f : 1.0f);
            l_r0 *= exp_old0;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_old0; acc_o[i][1] *= exp_old0;
            }
            m_r0 = m_new0;

            const bool has_valid1 = (rmax1 > -1e29f);
            float m_new1 = has_valid1 ? fmaxf(m_r1, rmax1) : m_r1;
            float exp_old1 = (has_valid1 && m_r1 > -1e29f)
                ? __expf(m_r1 - m_new1)
                : (has_valid1 ? 0.0f : 1.0f);
            l_r1 *= exp_old1;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][2] *= exp_old1; acc_o[i][3] *= exp_old1;
            }
            m_r1 = m_new1;

            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                float p00 = (has_valid0 && acc_s[nt][0] > -1e29f) ? __expf(acc_s[nt][0] - m_r0) : 0.0f;
                float p01 = (has_valid0 && acc_s[nt][1] > -1e29f) ? __expf(acc_s[nt][1] - m_r0) : 0.0f;
                float p10 = (has_valid1 && acc_s[nt][2] > -1e29f) ? __expf(acc_s[nt][2] - m_r1) : 0.0f;
                float p11 = (has_valid1 && acc_s[nt][3] > -1e29f) ? __expf(acc_s[nt][3] - m_r1) : 0.0f;
                sum0 += p00 + p01;
                sum1 += p10 + p11;

                unsigned int col0 = nt * 8u + tid_in_group * 2u;
                smem_P[row0][col0]     = __float2bfloat16(p00);
                smem_P[row0][col0 + 1] = __float2bfloat16(p01);
                smem_P[row1][col0]     = __float2bfloat16(p10);
                smem_P[row1][col0 + 1] = __float2bfloat16(p11);
            }

            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0;
            l_r1 += sum1;

            if (tid_in_group == 0u) {
                smem_ml[row0][0] = m_r0; smem_ml[row0][1] = l_r0;
                smem_ml[row1][0] = m_r1; smem_ml[row1][1] = l_r1;
            }
        }

        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        if (warp_id >= 2u) {
            unsigned int row0 = pv_warp_m + group_id;
            unsigned int row1 = row0 + 8u;
            float cur_m0 = smem_ml[row0][0];
            float cur_m1 = smem_ml[row1][0];
            float exp_r0 = (cur_m0 > -1e29f && m_r0 > -1e29f) ? __expf(m_r0 - cur_m0) : 0.0f;
            float exp_r1 = (cur_m1 > -1e29f && m_r1 > -1e29f) ? __expf(m_r1 - cur_m1) : 0.0f;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_r0; acc_o[i][1] *= exp_r0;
                acc_o[i][2] *= exp_r1; acc_o[i][3] *= exp_r1;
            }
            m_r0 = cur_m0; m_r1 = cur_m1;
        }

        if (kv_block + 1u < my_block_end) {
            unsigned int next_kv_start = (kv_block + 1u) * BC;
            const unsigned int chunks_per_row_k = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 128u) {
                unsigned int row = idx / chunks_per_row_k;
                unsigned int col = (idx % chunks_per_row_k) * 8u;
                unsigned int k_row = next_kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K[1u - buf][row][col]);
                if (k_row < total_kv) {
                    const __nv_bfloat16* src = kv_row_ptr(K_ctx, K_s, k_row, col);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"((const void*)src));
                } else {
                    *((uint4*)&smem_K[1u - buf][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // === PV MMA (all 4 warps, 8 n-tiles each) ===
        {
            #pragma unroll
            for (unsigned int ks = 0; ks < (BC / 16); ks++) {
                unsigned int k_off = ks * 16u;
                const unsigned short* sP_u16 = (const unsigned short*)smem_P;
                unsigned int ar0 = pv_warp_m + group_id;
                unsigned int ar1 = pv_warp_m + group_id + 8u;
                unsigned int ac0 = k_off + tid_in_group * 2u;
                unsigned int ac1 = k_off + tid_in_group * 2u + 8u;
                unsigned int a0 = *(const unsigned int*)&sP_u16[ar0 * p_smem_stride + ac0];
                unsigned int a1 = *(const unsigned int*)&sP_u16[ar1 * p_smem_stride + ac0];
                unsigned int a2 = *(const unsigned int*)&sP_u16[ar0 * p_smem_stride + ac1];
                unsigned int a3 = *(const unsigned int*)&sP_u16[ar1 * p_smem_stride + ac1];

                const unsigned short* sV_u16 = (const unsigned short*)smem_V;
                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
                    unsigned int n_col = (pv_n_start + nt) * 8u + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2u;
                    unsigned int k1 = k_off + tid_in_group * 2u + 8u;
                    unsigned int b0 = ((unsigned int)sV_u16[(k0 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV_u16[k0 * HDIM_PAD + n_col];
                    unsigned int b1 = ((unsigned int)sV_u16[(k1 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV_u16[k1 * HDIM_PAD + n_col];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]),
                          "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_o[nt][0]), "f"(acc_o[nt][1]),
                          "f"(acc_o[nt][2]), "f"(acc_o[nt][3])
                    );
                }
            }
        }

        if (kv_block + 1u < my_block_end) {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
    }

    // === Write unnormalized FP32 partials + (m, l) for log-sum-exp reduce ===
    {
        unsigned int row0 = pv_warp_m + group_id;
        unsigned int row1 = row0 + 8u;
        if (warp_id < 2u && tid_in_group == 0u) {
            if (row0 < q_len) {
                partial_ml[base_row_tile + (unsigned long long)row0] = make_float2(m_r0, l_r0);
            }
            if (row1 < q_len) {
                partial_ml[base_row_tile + (unsigned long long)row1] = make_float2(m_r1, l_r1);
            }
        }
        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            unsigned int col0 = (pv_n_start + nt) * 8u + tid_in_group * 2u;
            if (row0 < q_len) {
                const unsigned long long r_idx = base_row_tile + (unsigned long long)row0;
                *reinterpret_cast<float2*>(&partial_o[r_idx * (unsigned long long)HDIM + col0]) =
                    make_float2(acc_o[nt][0], acc_o[nt][1]);
            }
            if (row1 < q_len) {
                const unsigned long long r_idx = base_row_tile + (unsigned long long)row1;
                *reinterpret_cast<float2*>(&partial_o[r_idx * (unsigned long long)HDIM + col0]) =
                    make_float2(acc_o[nt][2], acc_o[nt][3]);
            }
        }
    }
}

// Grid:  [num_q_heads, ceil(m_rows / BR), num_splits]
// Block: [128, 1, 1]
extern "C" __global__ __launch_bounds__(128, 2)
void inferspark_prefill_split_ctx_h128_splitkv(
    const __nv_bfloat16* __restrict__ Q_noise,
    const __nv_bfloat16* __restrict__ K_ctx,
    const __nv_bfloat16* __restrict__ V_ctx,
    const __nv_bfloat16* __restrict__ K_noise,
    const __nv_bfloat16* __restrict__ V_noise,
    float* __restrict__ workspace,
    const unsigned int* __restrict__ eff_ctx_dev,
    const unsigned int m_rows,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int num_splits,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window
) {
    inferspark_prefill_split_ctx_h128_splitkv_impl<false>(
        Q_noise, K_ctx, V_ctx, nullptr, nullptr, K_noise, V_noise, workspace,
        eff_ctx_dev, nullptr, 0u, m_rows, num_q_heads, num_kv_heads, 1u, num_splits,
        inv_sqrt_d, causal, sliding_window
    );
}

// Grid:  [num_q_heads, ceil(m_rows / BR), batch_size * num_splits]
// Block: [128, 1, 1]
extern "C" __global__ __launch_bounds__(128, 2)
void inferspark_prefill_split_ctx_h128_batched_splitkv(
    const __nv_bfloat16* __restrict__ Q_noise,
    const unsigned long long* __restrict__ K_ctx_ptrs,
    const unsigned long long* __restrict__ V_ctx_ptrs,
    const __nv_bfloat16* __restrict__ K_noise,
    const __nv_bfloat16* __restrict__ V_noise,
    float* __restrict__ workspace,
    const unsigned int* __restrict__ eff_ctx_dev,
    const unsigned int* __restrict__ ring_start_dev,
    const unsigned int ring_cap,
    const unsigned int m_rows,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int batch_size,
    const unsigned int num_splits,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window
) {
    inferspark_prefill_split_ctx_h128_splitkv_impl<true>(
        Q_noise, nullptr, nullptr, K_ctx_ptrs, V_ctx_ptrs, K_noise, V_noise, workspace,
        eff_ctx_dev, ring_start_dev, ring_cap, m_rows, num_q_heads, num_kv_heads,
        batch_size, num_splits, inv_sqrt_d, causal, sliding_window
    );
}

// Lightweight log-sum-exp reduce for Split-KV H=128 (both linear B=1 and batched B>=1).
// Grid:  [num_q_heads, m_rows, batch_size]
// Block: [64, 1, 1] (each thread reduces 2 columns -> packed 32-bit 2xBF16 store)
extern "C" __global__ __launch_bounds__(64, 4)
void inferspark_prefill_split_ctx_h128_splitkv_reduce(
    const float* __restrict__ workspace,
    __nv_bfloat16* __restrict__ O_noise,
    const unsigned int m_rows,
    const unsigned int num_q_heads,
    const unsigned int batch_size,
    const unsigned int num_splits
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_row = blockIdx.y;
    const unsigned int b = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    if (q_head >= num_q_heads || q_row >= m_rows || b >= batch_size || tid >= (HDIM / 2u)) return;

    const unsigned long long total_rows =
        (unsigned long long)batch_size * (unsigned long long)num_splits *
        (unsigned long long)num_q_heads * (unsigned long long)m_rows;
    const float* const partial_o = workspace;
    const float2* const partial_ml =
        reinterpret_cast<const float2*>(workspace + total_rows * (unsigned long long)HDIM);
    const unsigned long long split_row_stride =
        (unsigned long long)num_q_heads * (unsigned long long)m_rows;
    const unsigned long long base_row =
        (unsigned long long)b * (unsigned long long)num_splits * split_row_stride +
        (unsigned long long)q_head * (unsigned long long)m_rows +
        (unsigned long long)q_row;

    float m_max = -1e30f;
    for (unsigned int s = 0u; s < num_splits; ++s) {
        const float2 ml = __ldg(&partial_ml[base_row + (unsigned long long)s * split_row_stride]);
        if (ml.y > 0.0f && ml.x > m_max) {
            m_max = ml.x;
        }
    }

    float l_total = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;
    const unsigned int col0 = tid * 2u;
    for (unsigned int s = 0u; s < num_splits; ++s) {
        const unsigned long long r_idx = base_row + (unsigned long long)s * split_row_stride;
        const float2 ml = __ldg(&partial_ml[r_idx]);
        if (ml.y > 0.0f) {
            const float w = __expf(ml.x - m_max);
            l_total += ml.y * w;
            const float2 o2 =
                __ldg(reinterpret_cast<const float2*>(&partial_o[r_idx * (unsigned long long)HDIM + col0]));
            acc0 += o2.x * w;
            acc1 += o2.y * w;
        }
    }

    const float inv_l = (l_total > 0.0f) ? (1.0f / l_total) : 0.0f;
    const unsigned long long q_seq_stride = (unsigned long long)num_q_heads * (unsigned long long)HDIM;
    __nv_bfloat16* const o_dst = O_noise +
        ((unsigned long long)b * (unsigned long long)m_rows + (unsigned long long)q_row) * q_seq_stride +
        (unsigned long long)q_head * (unsigned long long)HDIM + col0;
    const unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc0 * inv_l));
    const unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc1 * inv_l));
    *reinterpret_cast<unsigned int*>(o_dst) = lo | (hi << 16);
}
