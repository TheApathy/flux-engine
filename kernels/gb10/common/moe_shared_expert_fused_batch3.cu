// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused MoE Expert+Shared GEMV — multi-token batch variant (arbitrary num_tokens).
//
// Processes num_tokens tokens through MoE in single kernel launches by expanding
// blockIdx.y to accommodate num_tokens sets of (top_k routed + 1 shared) experts.
//
// Token layout in blockIdx.y:
//   y ∈ [0, num_tokens*top_k)                     → routed experts (token = y/top_k, slot = y%top_k)
//   y ∈ [num_tokens*top_k, num_tokens*(top_k+1))  → shared expert  (token = y - num_tokens*top_k)
//
// Grid: gate_up_batch3   (ceil(N/8), num_tokens*(top_k+1), 2)
//       silu_down_batch3 (ceil(N/8), num_tokens*(top_k+1), 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_BATCH3[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ── Fused Gate+Up 2x with shared expert — multi-token batch variant ──
//
// Grid: (ceil(N/8), num_tokens*(top_k+1), 2)  Block: (128, 1, 1)
// blockIdx.y: 0..num_tokens*top_k-1 = routed (token=y/top_k, slot=y%top_k)
//             num_tokens*top_k..num_tokens*(top_k+1)-1 = shared (token=y-num_tokens*top_k)
extern "C" __global__ void moe_expert_gate_up_shared_batch3(
    const __nv_bfloat16* __restrict__ A,       // [num_tokens, H] BF16 input
    // Routed expert tables
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,      // [num_tokens*top_k, inter] BF16
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,        // [num_tokens*top_k, inter] BF16
    const unsigned int* __restrict__ expert_indices,  // [num_tokens*top_k] u32
    // Shared expert direct pointers
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,   // [num_tokens, inter] BF16
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,     // [num_tokens, inter] BF16
    unsigned int N, unsigned int K, unsigned int top_k,
    unsigned int num_tokens                    // number of tokens in batch
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y >= total_routed);

    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;  // 0..num_tokens-1
        expert_slot = 0;
    } else {
        token = y / top_k;
        expert_slot = y % top_k;
    }

    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (is_shared) {
        if (proj == 0) {
            B_packed = sh_gate_packed; B_scale = sh_gate_scale;
            s2 = sh_gate_s2; C = sh_gate_out + (unsigned long long)token * N;
        } else {
            B_packed = sh_up_packed; B_scale = sh_up_scale;
            s2 = sh_up_s2; C = sh_up_out + (unsigned long long)token * N;
        }
        // NULL shared expert: model has no shared expert weights — write zero and return
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id];
            C = gate_out + (unsigned long long)flat_slot * N;
        } else {
            B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id];
            C = up_out + (unsigned long long)flat_slot * N;
        }
        // EP: NULL pointer means remote expert — write zero output and return
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH3[threadIdx.x];
    __syncthreads();

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;
        uint4 a_lo = ((const uint4*)A_token)[k16 * 2];
        uint4 a_hi = ((const uint4*)A_token)[k16 * 2 + 1];
        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                       a_hi.x, a_hi.y, a_hi.z, a_hi.w};

        unsigned long long packed8_1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        __nv_fp8_e4m3 fp8_1; *(unsigned char*)&fp8_1 = sb1;
        float sc1 = (float)fp8_1 * s2;

        unsigned long long packed8_2 = have_n2 ?
            *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        __nv_fp8_e4m3 fp8_2; *(unsigned char*)&fp8_2 = sb2;
        float sc2 = have_n2 ? (float)fp8_2 * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
            __nv_bfloat16 al, ah;
            *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
            float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * w1l + afh * w1h;
            acc2 += afl * w2l + afh * w2h;
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) C[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) C[n2] = __float2bfloat16(acc2);
    }
}

// ── Fused SiLU+Down 2x with shared expert — multi-token batch variant ──
//
// Grid: (ceil(N/8), num_tokens*(top_k+1), 1)  Block: (128, 1, 1)
extern "C" __global__ void moe_expert_silu_down_shared_batch3(
    const __nv_bfloat16* __restrict__ gate_out,  // [num_tokens*top_k, inter] BF16
    const __nv_bfloat16* __restrict__ up_out,    // [num_tokens*top_k, inter] BF16
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,               // [num_tokens*top_k, H] BF16
    const unsigned int* __restrict__ expert_indices,  // [num_tokens*top_k] u32
    // Shared expert
    const __nv_bfloat16* __restrict__ sh_gate_in,  // [num_tokens, inter] BF16
    const __nv_bfloat16* __restrict__ sh_up_in,    // [num_tokens, inter] BF16
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,       // [num_tokens, H] BF16
    unsigned int N, unsigned int K, unsigned int top_k,
    unsigned int num_tokens                        // number of tokens in batch
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);

    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;
        expert_slot = 0;
    } else {
        token = y / top_k;
        expert_slot = y % top_k;
    }

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;

    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in + (unsigned long long)token * K;
        u_ptr = sh_up_in + (unsigned long long)token * K;
        // NULL shared expert: model has no shared expert weights — write zero and return
        if (B_packed == 0) {
            __nv_bfloat16* out = sh_down_out + (unsigned long long)token * N;
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                out[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)flat_slot * K;
        u_ptr = up_out + (unsigned long long)flat_slot * K;
        // EP: NULL pointer means remote expert — write zero output and return
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[(unsigned long long)(token * top_k + expert_slot) * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    __shared__ float s_lut[16];
    // Dynamic: K floats, sized by the launcher (issue #85 -- the old
    // static s_act[1024] overflowed for expert inter dims > 1024,
    // e.g. Mistral-Small-4's expert_hidden_dim=2048, illegal-addressing
    // the batched FFN; matches the extern pattern the _t variant uses).
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH3[threadIdx.x];

    // Phase 1: Precompute SiLU(gate)*up into shared memory
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    // Phase 2: GEMV reading precomputed activation from shared memory
    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        unsigned long long packed8_1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        __nv_fp8_e4m3 fp8_1; *(unsigned char*)&fp8_1 = sb1;
        float sc1 = (float)fp8_1 * s2;

        unsigned long long packed8_2 = have_n2 ?
            *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        __nv_fp8_e4m3 fp8_2; *(unsigned char*)&fp8_2 = sb2;
        float sc2 = have_n2 ? (float)fp8_2 * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];

            unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;

            acc1 += al * w1l + ah * w1h;
            acc2 += al * w2l + ah * w2h;
        }
    }

    // Output: shared at sh_down_out[token*N], routed at C[flat_slot*N]
    __nv_bfloat16* out = is_shared
        ? (sh_down_out + (unsigned long long)token * N)
        : (C + (unsigned long long)(token * top_k + expert_slot) * N);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) out[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) out[n2] = __float2bfloat16(acc2);
    }
}

// ── Weighted sum + sigmoid blend — multi-token batch variant ──
//
// Combines routed expert outputs with shared expert via sigmoid gate.
// blockIdx.y = token index (0..num_tokens-1).
//
// Grid: (ceil(hidden/256), num_tokens, 1)  Block: (256, 1, 1)
extern "C" __global__ void moe_weighted_sum_blend_batch3(
    __nv_bfloat16* __restrict__ output,              // [num_tokens, hidden] BF16
    const __nv_bfloat16* __restrict__ expert_out,    // [num_tokens*top_k, hidden] BF16
    const float* __restrict__ expert_weights,         // [num_tokens*top_k] f32
    const __nv_bfloat16* __restrict__ shared_out,    // [num_tokens, hidden] BF16
    const __nv_bfloat16* __restrict__ input,         // [num_tokens, K] BF16 (MoE input)
    const __nv_bfloat16* __restrict__ gate_weight,   // [1, K] BF16 (shared gate)
    unsigned int hidden,
    unsigned int top_k,
    unsigned int K
) {
    const unsigned int token = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;

    const __nv_bfloat16* my_input = input + (unsigned long long)token * K;
    const float* my_weights = expert_weights + token * top_k;
    const __nv_bfloat16* my_expert_out = expert_out + (unsigned long long)token * top_k * hidden;
    const __nv_bfloat16* my_shared_out = shared_out + (unsigned long long)token * hidden;
    __nv_bfloat16* my_output = output + (unsigned long long)token * hidden;

    // ── Phase 1: Compute gate scalar (dot product + sigmoid) ──
    __shared__ float s_warp_sums[8];
    __shared__ float sigmoid_val;

    if (gate_weight == 0) {
        // No gate: shared expert always included at full weight
        if (tid == 0) sigmoid_val = 1.0f;
        __syncthreads();
    } else {
        float dot_acc = 0.0f;
        unsigned int K8 = K / 8;
        for (unsigned int k8 = tid; k8 < K8; k8 += 256) {
            uint4 a_data = ((const uint4*)my_input)[k8];
            uint4 w_data = ((const uint4*)gate_weight)[k8];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            const unsigned int w_raw[4] = {w_data.x, w_data.y, w_data.z, w_data.w};

            #pragma unroll
            for (int b = 0; b < 4; b++) {
                __nv_bfloat16 a_lo, a_hi, w_lo, w_hi;
                *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
                *(unsigned short*)&w_lo = (unsigned short)(w_raw[b] & 0xFFFF);
                *(unsigned short*)&w_hi = (unsigned short)(w_raw[b] >> 16);
                dot_acc += __bfloat162float(a_lo) * __bfloat162float(w_lo);
                dot_acc += __bfloat162float(a_hi) * __bfloat162float(w_hi);
            }
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            dot_acc += __shfl_down_sync(0xFFFFFFFF, dot_acc, offset);
        }
        if (lane == 0) {
            s_warp_sums[warp_id] = dot_acc;
        }
        __syncthreads();

        if (tid == 0) {
            float gate_scalar = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                gate_scalar += s_warp_sums[w];
            }
            sigmoid_val = 1.0f / (1.0f + __expf(-gate_scalar));
        }
        __syncthreads();
    }

    // ── Phase 2: Weighted sum + blend ──
    unsigned int j = blockIdx.x * blockDim.x + tid;
    if (j >= hidden) return;

    float acc = 0.0f;
    for (unsigned int e = 0; e < top_k; e++) {
        acc += my_weights[e] * __bfloat162float(my_expert_out[(unsigned long long)e * hidden + j]);
    }
    acc += sigmoid_val * __bfloat162float(my_shared_out[j]);
    my_output[j] = __float2bfloat16(acc);
}

// ── Expert-deduplicated twins of the two kernels above ──
//
// Same grid, same per-row arithmetic, but each distinct expert (and the
// shared expert) is streamed from memory ONCE for all rows routed to it.
// The block at the FIRST flat slot routed to an expert is its leader; later
// duplicates exit. The leader dequantizes each weight group once and applies
// it to every one of its rows with exactly the per-row expression, lane
// partition and warp-shuffle order of the kernels above, so each output row
// is byte-identical to theirs (the Flash-Next target builds with
// --fmad=false, so no contraction can differ either).
//
// ROWS_MAX bounds the rows one expert can serve (an expert appears at most
// once per token, so rows <= num_tokens <= ROWS_MAX).

template <int ROWS_MAX>
__device__ __forceinline__ unsigned int moe_dedup_rows(
    const unsigned int* __restrict__ expert_indices,
    unsigned int y, unsigned int total_routed, unsigned int top_k,
    unsigned int num_tokens, bool is_shared,
    unsigned int* row_token, unsigned int* row_slot
) {
    if (is_shared) {
        if (y != total_routed) return 0u;
        const unsigned int n = num_tokens < (unsigned int)ROWS_MAX ? num_tokens : (unsigned int)ROWS_MAX;
        for (unsigned int t = 0; t < n; t++) {
            row_token[t] = t;
            row_slot[t] = 0u;
        }
        return n;
    }
    const unsigned int e = expert_indices[y];
    for (unsigned int j = 0; j < y; j++) {
        if (expert_indices[j] == e) return 0u;
    }
    unsigned int n = 0;
    for (unsigned int j = y; j < total_routed; j++) {
        if (expert_indices[j] == e && n < (unsigned int)ROWS_MAX) {
            row_token[n] = j / top_k;
            row_slot[n] = j;
            n++;
        }
    }
    return n;
}

template <int ROWS_MAX>
__device__ __forceinline__ void moe_gate_up_dedup_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k,
    unsigned int num_tokens
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y >= total_routed);

    unsigned int row_token[ROWS_MAX];
    unsigned int row_slot[ROWS_MAX];
    const unsigned int nrows = moe_dedup_rows<ROWS_MAX>(
        expert_indices, y, total_routed, top_k, num_tokens, is_shared, row_token, row_slot);
    if (nrows == 0u) return;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C_base;
    if (is_shared) {
        if (proj == 0) {
            B_packed = sh_gate_packed; B_scale = sh_gate_scale; s2 = sh_gate_s2; C_base = sh_gate_out;
        } else {
            B_packed = sh_up_packed; B_scale = sh_up_scale; s2 = sh_up_s2; C_base = sh_up_out;
        }
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int r = 0; r < nrows; r++) {
                __nv_bfloat16* C = C_base + (unsigned long long)row_token[r] * N;
                for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                    C[n_base + i] = __float2bfloat16(0.0f);
                }
            }
            return;
        }
    } else {
        const unsigned int expert_id = expert_indices[y];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id];
            C_base = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id];
            C_base = up_out;
        }
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int r = 0; r < nrows; r++) {
                __nv_bfloat16* C = C_base + (unsigned long long)row_slot[r] * N;
                for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                    C[n_base + i] = __float2bfloat16(0.0f);
                }
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH3[threadIdx.x];
    __syncthreads();

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    float acc1[ROWS_MAX], acc2[ROWS_MAX];
    #pragma unroll
    for (int r = 0; r < ROWS_MAX; r++) { acc1[r] = 0.0f; acc2[r] = 0.0f; }

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        unsigned long long packed8_1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        __nv_fp8_e4m3 fp8_1; *(unsigned char*)&fp8_1 = sb1;
        float sc1 = (float)fp8_1 * s2;

        unsigned long long packed8_2 = have_n2 ?
            *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        __nv_fp8_e4m3 fp8_2; *(unsigned char*)&fp8_2 = sb2;
        float sc2 = have_n2 ? (float)fp8_2 * s2 : 0.0f;

        float w1l[8], w1h[8], w2l[8], w2h[8];
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
            w1l[b] = s_lut[bv1 & 0xF] * sc1; w1h[b] = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
            w2l[b] = s_lut[bv2 & 0xF] * sc2; w2h[b] = s_lut[bv2 >> 4] * sc2;
        }

        #pragma unroll
        for (int r = 0; r < ROWS_MAX; r++) {
            if ((unsigned int)r >= nrows) break;
            const __nv_bfloat16* A_token = A + (unsigned long long)row_token[r] * K;
            uint4 a_lo = ((const uint4*)A_token)[k16 * 2];
            uint4 a_hi = ((const uint4*)A_token)[k16 * 2 + 1];
            const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                           a_hi.x, a_hi.y, a_hi.z, a_hi.w};
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                __nv_bfloat16 al, ah;
                *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
                *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
                float afl = __bfloat162float(al), afh = __bfloat162float(ah);
                acc1[r] += afl * w1l[b] + afh * w1h[b];
                acc2[r] += afl * w2l[b] + afh * w2h[b];
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_MAX; r++) {
        if ((unsigned int)r >= nrows) break;
        __nv_bfloat16* C = is_shared
            ? C_base + (unsigned long long)row_token[r] * N
            : C_base + (unsigned long long)row_slot[r] * N;
        float a1 = acc1[r], a2 = acc2[r];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            a1 += __shfl_down_sync(0xFFFFFFFF, a1, offset);
        if (lane == 0) C[n1] = __float2bfloat16(a1);
        if (have_n2) {
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                a2 += __shfl_down_sync(0xFFFFFFFF, a2, offset);
            if (lane == 0) C[n2] = __float2bfloat16(a2);
        }
    }
}

template <int ROWS_MAX>
__device__ __forceinline__ void moe_silu_down_dedup_body(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k,
    unsigned int num_tokens
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);

    unsigned int row_token[ROWS_MAX];
    unsigned int row_slot[ROWS_MAX];
    const unsigned int nrows = moe_dedup_rows<ROWS_MAX>(
        expert_indices, y, total_routed, top_k, num_tokens, is_shared, row_token, row_slot);
    if (nrows == 0u) return;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    if (is_shared) {
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int r = 0; r < nrows; r++) {
                __nv_bfloat16* out = sh_down_out + (unsigned long long)row_token[r] * N;
                for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                    out[n_base + i] = __float2bfloat16(0.0f);
                }
            }
            return;
        }
    } else {
        const unsigned int expert_id = expert_indices[y];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int r = 0; r < nrows; r++) {
                for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                    C[(unsigned long long)row_slot[r] * N + n_base + i] = __float2bfloat16(0.0f);
                }
            }
            return;
        }
    }

    __shared__ float s_lut[16];
    // Dynamic: ROWS_MAX * K floats, sized by the launcher.
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH3[threadIdx.x];

    for (unsigned int r = 0; r < nrows; r++) {
        const __nv_bfloat16* g_ptr = is_shared
            ? sh_gate_in + (unsigned long long)row_token[r] * K
            : gate_out + (unsigned long long)row_slot[r] * K;
        const __nv_bfloat16* u_ptr = is_shared
            ? sh_up_in + (unsigned long long)row_token[r] * K
            : up_out + (unsigned long long)row_slot[r] * K;
        for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
            float gf = __bfloat162float(g_ptr[i]);
            float uf = __bfloat162float(u_ptr[i]);
            s_act[r * K + i] = (gf / (1.0f + __expf(-gf))) * uf;
        }
    }
    __syncthreads();

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    float acc1[ROWS_MAX], acc2[ROWS_MAX];
    #pragma unroll
    for (int r = 0; r < ROWS_MAX; r++) { acc1[r] = 0.0f; acc2[r] = 0.0f; }

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        unsigned long long packed8_1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        __nv_fp8_e4m3 fp8_1; *(unsigned char*)&fp8_1 = sb1;
        float sc1 = (float)fp8_1 * s2;

        unsigned long long packed8_2 = have_n2 ?
            *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        __nv_fp8_e4m3 fp8_2; *(unsigned char*)&fp8_2 = sb2;
        float sc2 = have_n2 ? (float)fp8_2 * s2 : 0.0f;

        float w1l[8], w1h[8], w2l[8], w2h[8];
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
            w1l[b] = s_lut[bv1 & 0xF] * sc1; w1h[b] = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
            w2l[b] = s_lut[bv2 & 0xF] * sc2; w2h[b] = s_lut[bv2 >> 4] * sc2;
        }

        #pragma unroll
        for (int r = 0; r < ROWS_MAX; r++) {
            if ((unsigned int)r >= nrows) break;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                float al = s_act[r * K + base_k + b * 2];
                float ah = s_act[r * K + base_k + b * 2 + 1];
                acc1[r] += al * w1l[b] + ah * w1h[b];
                acc2[r] += al * w2l[b] + ah * w2h[b];
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_MAX; r++) {
        if ((unsigned int)r >= nrows) break;
        __nv_bfloat16* out = is_shared
            ? (sh_down_out + (unsigned long long)row_token[r] * N)
            : (C + (unsigned long long)row_slot[r] * N);
        float a1 = acc1[r], a2 = acc2[r];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            a1 += __shfl_down_sync(0xFFFFFFFF, a1, offset);
        if (lane == 0) out[n1] = __float2bfloat16(a1);
        if (have_n2) {
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                a2 += __shfl_down_sync(0xFFFFFFFF, a2, offset);
            if (lane == 0) out[n2] = __float2bfloat16(a2);
        }
    }
}

#define MOE_GATE_UP_DEDUP_PARAMS \
    const __nv_bfloat16* __restrict__ A, \
    const unsigned long long* __restrict__ gate_packed_ptrs, \
    const unsigned long long* __restrict__ gate_scale_ptrs, \
    const float* __restrict__ gate_scale2_vals, \
    __nv_bfloat16* __restrict__ gate_out, \
    const unsigned long long* __restrict__ up_packed_ptrs, \
    const unsigned long long* __restrict__ up_scale_ptrs, \
    const float* __restrict__ up_scale2_vals, \
    __nv_bfloat16* __restrict__ up_out, \
    const unsigned int* __restrict__ expert_indices, \
    const unsigned char* __restrict__ sh_gate_packed, \
    const unsigned char* __restrict__ sh_gate_scale, \
    float sh_gate_s2, \
    __nv_bfloat16* __restrict__ sh_gate_out, \
    const unsigned char* __restrict__ sh_up_packed, \
    const unsigned char* __restrict__ sh_up_scale, \
    float sh_up_s2, \
    __nv_bfloat16* __restrict__ sh_up_out, \
    unsigned int N, unsigned int K, unsigned int top_k, \
    unsigned int num_tokens
#define MOE_GATE_UP_DEDUP_ARGS \
    A, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, gate_out, \
    up_packed_ptrs, up_scale_ptrs, up_scale2_vals, up_out, expert_indices, \
    sh_gate_packed, sh_gate_scale, sh_gate_s2, sh_gate_out, \
    sh_up_packed, sh_up_scale, sh_up_s2, sh_up_out, N, K, top_k, num_tokens
#define MOE_SILU_DOWN_DEDUP_PARAMS \
    const __nv_bfloat16* __restrict__ gate_out, \
    const __nv_bfloat16* __restrict__ up_out, \
    const unsigned long long* __restrict__ packed_ptrs, \
    const unsigned long long* __restrict__ scale_ptrs, \
    const float* __restrict__ scale2_vals, \
    __nv_bfloat16* __restrict__ C, \
    const unsigned int* __restrict__ expert_indices, \
    const __nv_bfloat16* __restrict__ sh_gate_in, \
    const __nv_bfloat16* __restrict__ sh_up_in, \
    const unsigned char* __restrict__ sh_down_packed, \
    const unsigned char* __restrict__ sh_down_scale, \
    float sh_down_s2, \
    __nv_bfloat16* __restrict__ sh_down_out, \
    unsigned int N, unsigned int K, unsigned int top_k, \
    unsigned int num_tokens
#define MOE_SILU_DOWN_DEDUP_ARGS \
    gate_out, up_out, packed_ptrs, scale_ptrs, scale2_vals, C, expert_indices, \
    sh_gate_in, sh_up_in, sh_down_packed, sh_down_scale, sh_down_s2, sh_down_out, \
    N, K, top_k, num_tokens

extern "C" __global__ void moe_expert_gate_up_shared_dedup_r8(MOE_GATE_UP_DEDUP_PARAMS) {
    moe_gate_up_dedup_body<8>(MOE_GATE_UP_DEDUP_ARGS);
}
extern "C" __global__ void moe_expert_gate_up_shared_dedup_r16(MOE_GATE_UP_DEDUP_PARAMS) {
    moe_gate_up_dedup_body<16>(MOE_GATE_UP_DEDUP_ARGS);
}
extern "C" __global__ void moe_expert_silu_down_shared_dedup_r8(MOE_SILU_DOWN_DEDUP_PARAMS) {
    moe_silu_down_dedup_body<8>(MOE_SILU_DOWN_DEDUP_ARGS);
}
extern "C" __global__ void moe_expert_silu_down_shared_dedup_r16(MOE_SILU_DOWN_DEDUP_PARAMS) {
    moe_silu_down_dedup_body<16>(MOE_SILU_DOWN_DEDUP_ARGS);
}

// ── Dedup down, activation-once variant (fn-mtp, ATLAS_QWEN4_MOE_DOWN_ACT=1) ──
//
// The dedup down kernel above recomputes SiLU(gate)*up for every row of its
// expert into a ROWS_MAX*K dynamic shared-memory slab in each of the
// ceil(N/8) blocks per expert; the 20 KiB slab (r8, K=640) caps occupancy and
// the down GEMV falls from ~130 to ~70 GB/s as rows grow. Here the activation
// of every routed slot and shared row is computed ONCE into an FP32 buffer
// with the identical expression, and the down GEMV reads it from global
// memory. Leader rule, lane partition (32 lanes per output pair, k16 = lane +
// 32 i), per-byte expression acc += al*wl + ah*wh and the warp tree are the
// dedup kernel's, so every output is byte-identical to it. Each warp owns
// OUTP output pairs and loads every weight word of UNR k16 steps first.
//
// act layout: [num_tokens*top_k routed slots | num_tokens shared rows][K] f32.

extern "C" __global__ void moe_dedup_act_rows(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    float* __restrict__ act,
    unsigned int K, unsigned int total_routed, unsigned int num_tokens
) {
    const unsigned int slot = blockIdx.y;
    const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K || slot >= total_routed + num_tokens) return;
    const bool sh = slot >= total_routed;
    const __nv_bfloat16* g = sh ? sh_gate_in + (unsigned long long)(slot - total_routed) * K
                                : gate_out + (unsigned long long)slot * K;
    const __nv_bfloat16* u = sh ? sh_up_in + (unsigned long long)(slot - total_routed) * K
                                : up_out + (unsigned long long)slot * K;
    float gf = __bfloat162float(g[k]);
    float uf = __bfloat162float(u[k]);
    act[(unsigned long long)slot * K + k] = (gf / (1.0f + __expf(-gf))) * uf;
}

#define MOE_DOWN_ACT_WARPS 4

template <int ROWS_MAX, int OUTP, int UNR>
__device__ __forceinline__ void moe_silu_down_dedup_act_body(
    const float* __restrict__ act,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k,
    unsigned int num_tokens
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);
    unsigned int row_token[ROWS_MAX];
    unsigned int row_slot[ROWS_MAX];
    const unsigned int nrows = moe_dedup_rows<ROWS_MAX>(
        expert_indices, y, total_routed, top_k, num_tokens, is_shared, row_token, row_slot);
    if (nrows == 0u) return;
    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    if (is_shared) {
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (MOE_DOWN_ACT_WARPS * 2 * OUTP);
            for (unsigned int r = 0; r < nrows; r++)
                for (unsigned int i = threadIdx.x; i < MOE_DOWN_ACT_WARPS * 2 * OUTP && n_base + i < N; i += blockDim.x)
                    sh_down_out[(unsigned long long)row_token[r] * N + n_base + i] = __float2bfloat16(0.0f);
            return;
        }
    } else {
        const unsigned int expert_id = expert_indices[y];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (MOE_DOWN_ACT_WARPS * 2 * OUTP);
            for (unsigned int r = 0; r < nrows; r++)
                for (unsigned int i = threadIdx.x; i < MOE_DOWN_ACT_WARPS * 2 * OUTP && n_base + i < N; i += blockDim.x)
                    C[(unsigned long long)row_slot[r] * N + n_base + i] = __float2bfloat16(0.0f);
            return;
        }
    }
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH3[threadIdx.x];
    __syncthreads();
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int nb = (blockIdx.x * MOE_DOWN_ACT_WARPS + warp) * (2 * OUTP);
    if (nb >= N) return;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    float acc[2 * OUTP][ROWS_MAX];
    #pragma unroll
    for (int o = 0; o < 2 * OUTP; o++)
        #pragma unroll
        for (int r = 0; r < ROWS_MAX; r++) acc[o][r] = 0.0f;

    for (unsigned int kb = lane; kb < K16; kb += WARP_SIZE * UNR) {
        unsigned long long pk[UNR][2 * OUTP];
        unsigned char sb[UNR][2 * OUTP];
        #pragma unroll
        for (int u = 0; u < UNR; u++) {
            const unsigned int k16 = kb + WARP_SIZE * u;
            #pragma unroll
            for (int o = 0; o < 2 * OUTP; o++) {
                const unsigned int n = nb + o;
                const bool ok = k16 < K16 && n < N;
                pk[u][o] = ok ? *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + k16 * 8) : 0ull;
                sb[u][o] = ok ? B_scale[(unsigned long long)n * num_groups + (k16 * 16) / GROUP_SIZE] : 0;
            }
        }
        #pragma unroll
        for (int u = 0; u < UNR; u++) {
            const unsigned int k16 = kb + WARP_SIZE * u;
            if (k16 >= K16) break;
            const unsigned int base_k = k16 * 16;
            float wl[2 * OUTP][8], wh[2 * OUTP][8];
            #pragma unroll
            for (int o = 0; o < 2 * OUTP; o++) {
                __nv_fp8_e4m3 f; *(unsigned char*)&f = sb[u][o];
                const float sc = (nb + o < N) ? (float)f * s2 : 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    const unsigned char bv = (unsigned char)(pk[u][o] >> (b * 8));
                    wl[o][b] = s_lut[bv & 0xF] * sc;
                    wh[o][b] = s_lut[bv >> 4] * sc;
                }
            }
            #pragma unroll
            for (int r = 0; r < ROWS_MAX; r++) {
                if ((unsigned int)r >= nrows) break;
                const unsigned int arow = is_shared ? total_routed + row_token[r] : row_slot[r];
                const float4* a4 = (const float4*)(act + (unsigned long long)arow * K + base_k);
                const float4 q0 = a4[0], q1 = a4[1], q2 = a4[2], q3 = a4[3];
                const float av[16] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w,
                                      q2.x, q2.y, q2.z, q2.w, q3.x, q3.y, q3.z, q3.w};
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    const float al = av[b * 2];
                    const float ah = av[b * 2 + 1];
                    #pragma unroll
                    for (int o = 0; o < 2 * OUTP; o++) acc[o][r] += al * wl[o][b] + ah * wh[o][b];
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS_MAX; r++) {
        if ((unsigned int)r >= nrows) break;
        __nv_bfloat16* out = is_shared ? (sh_down_out + (unsigned long long)row_token[r] * N)
                                       : (C + (unsigned long long)row_slot[r] * N);
        #pragma unroll
        for (int o = 0; o < 2 * OUTP; o++) {
            float a = acc[o][r];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                a += __shfl_down_sync(0xFFFFFFFF, a, offset);
            if (lane == 0 && nb + o < N) out[nb + o] = __float2bfloat16(a);
        }
    }
}

#define MOE_SILU_DOWN_ACT_PARAMS \
    const float* __restrict__ act, \
    const unsigned long long* __restrict__ packed_ptrs, \
    const unsigned long long* __restrict__ scale_ptrs, \
    const float* __restrict__ scale2_vals, \
    __nv_bfloat16* __restrict__ C, \
    const unsigned int* __restrict__ expert_indices, \
    const unsigned char* __restrict__ sh_down_packed, \
    const unsigned char* __restrict__ sh_down_scale, \
    float sh_down_s2, \
    __nv_bfloat16* __restrict__ sh_down_out, \
    unsigned int N, unsigned int K, unsigned int top_k, \
    unsigned int num_tokens
#define MOE_SILU_DOWN_ACT_ARGS \
    act, packed_ptrs, scale_ptrs, scale2_vals, C, expert_indices, \
    sh_down_packed, sh_down_scale, sh_down_s2, sh_down_out, N, K, top_k, num_tokens

// Grid: (ceil(N / (4 * 2 * OUTP)), num_tokens * (top_k + 1), 1), Block: (128, 1, 1), no dynamic smem.
extern "C" __global__ void __launch_bounds__(128) moe_expert_silu_down_shared_dedup_act_r8_o1(MOE_SILU_DOWN_ACT_PARAMS) {
    moe_silu_down_dedup_act_body<8, 1, 2>(MOE_SILU_DOWN_ACT_ARGS);
}
extern "C" __global__ void __launch_bounds__(128) moe_expert_silu_down_shared_dedup_act_r8_o2(MOE_SILU_DOWN_ACT_PARAMS) {
    moe_silu_down_dedup_act_body<8, 2, 2>(MOE_SILU_DOWN_ACT_ARGS);
}
extern "C" __global__ void __launch_bounds__(128) moe_expert_silu_down_shared_dedup_act_r16_o1(MOE_SILU_DOWN_ACT_PARAMS) {
    moe_silu_down_dedup_act_body<16, 1, 2>(MOE_SILU_DOWN_ACT_ARGS);
}
// o4 (4 output pairs per warp, 32 outputs per block): fastest r8 variant at 2..8 rows in the
// fn-mtp microbench (0.210 vs 0.388 ms/layer at 5 rows, byte-identical).
extern "C" __global__ void __launch_bounds__(128) moe_expert_silu_down_shared_dedup_act_r8_o4(MOE_SILU_DOWN_ACT_PARAMS) {
    moe_silu_down_dedup_act_body<8, 4, 2>(MOE_SILU_DOWN_ACT_ARGS);
}
