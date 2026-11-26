// SPDX-License-Identifier: AGPL-3.0-only
//
// Transposed-layout decode MoE GEMV — Phase 8a unified-layout MoE.
//
// Same semantics as moe_expert_silu_down_shared / moe_expert_gate_up_shared,
// but reads the weight tensor in transposed `[K/2, N]` layout (input-major)
// instead of the current `[N, K/2]` layout (output-major). Goal: a single
// weight layout shared between prefill and decode so we can free the
// untransposed copies (~59 GB on MiniMax M2.7-NVFP4 EP=2) and either
// run the persistent down transpose or use the headroom for KV cache.
//
// Coalescence strategy is inverted vs the original kernel:
//   - original: each thread owns ONE output (sub-group reduces over K)
//     → reads B[n_lane * (K/2) + k_iter*8] = 8 bytes contiguous per lane
//   - transposed: each thread STILL owns one output, but threads in a
//     warp have ADJACENT n's (lane = n within warp). Read B_t[k_half * N + n]:
//     32 bytes contiguous per warp = 1 cache line per K iter. No warp
//     reduction needed (each lane has its own accumulator + own output).
//
// Block: (32, 1, 1)  Grid: gate_up (ceil(N/32), top_k+1, 2); silu_down (ceil(N/32), top_k+1, 1) + K*4 B dynamic smem

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Tuning note: 32-thread blocks (1 warp per block) outperform 128-thread blocks
// on GB10 for the transposed decode silu_down — more blocks → better SM
// occupancy, the s_act shared-mem precompute parallelism is unchanged because
// the K loop dominates (block_size irrelevant once each thread is 1 output).
#define BLOCK_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_T[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Transposed-layout fused gate+up decode kernel.
//
// Single-token GEMV: for each routed expert slot (top_k of them) plus the
// shared expert, compute gate_out = A @ W_gate^T and up_out = A @ W_up^T.
// blockIdx.z selects gate (0) or up (1). Same coalescence strategy as
// silu_down_t — each thread owns one output, lanes within warp adjacent
// in N.
extern "C" __global__ void moe_expert_gate_up_shared_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert direct pointers (transposed).
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (is_shared) {
        if (proj == 0) {
            if (sh_gate_t_packed == 0) {
                const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
                if (n < N) sh_gate_out[n] = __float2bfloat16(0.0f);
                return;
            }
            B_packed = sh_gate_t_packed;
            B_scale = sh_gate_t_scale;
            s2 = sh_gate_s2;
            C = sh_gate_out;
        } else {
            if (sh_up_t_packed == 0) {
                const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
                if (n < N) sh_up_out[n] = __float2bfloat16(0.0f);
                return;
            }
            B_packed = sh_up_t_packed;
            B_scale = sh_up_t_scale;
            s2 = sh_up_s2;
            C = sh_up_out;
        }
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_t_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id];
            C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_t_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id];
            C = up_out;
        }
        if (B_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool valid = (n < N);

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    __syncthreads();

    if (!valid) return;

    // GEMV: C[n] = sum_k A[k] * W[n, k]. With transposed weight stored as
    // [K/2, N] packed: each byte at (k_half, n) holds two consecutive k
    // nibbles for output position n. Iterate by scale-group (16 K) to
    // cache the per-group scale.
    const unsigned int num_groups = K / GROUP_SIZE;
    float acc = 0.0f;
    for (unsigned int sg = 0; sg < num_groups; sg++) {
        unsigned char sb = B_scale[(unsigned long long)sg * N + n];
        __nv_fp8_e4m3 fp8_s; *(unsigned char*)&fp8_s = sb;
        float sc = (float)fp8_s * s2;
        const unsigned int kh_base = sg * 8;
        #pragma unroll
        for (unsigned int kh_off = 0; kh_off < 8; kh_off++) {
            unsigned int k_half = kh_base + kh_off;
            unsigned char byte = B_packed[(unsigned long long)k_half * N + n];
            float a_lo = __bfloat162float(A[k_half * 2]);
            float a_hi = __bfloat162float(A[k_half * 2 + 1]);
            float w_lo = s_lut[byte & 0xFu] * sc;
            float w_hi = s_lut[(byte >> 4) & 0xFu] * sc;
            acc += a_lo * w_lo + a_hi * w_hi;
        }
    }

    if (is_shared) {
        C[n] = __float2bfloat16(acc);
    } else {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}

// Transposed-layout silu_down decode kernel.
//
// Per-expert weight buffers `[K/2, N]` packed NVFP4 + `[K/16, N]` FP8 scales.
// Input gate_out / up_out: `[top_k, K]` BF16 (routed slots); shared-expert
// inputs come from `sh_gate_in` / `sh_up_in`: `[K]` BF16. Output C:
// `[top_k, N]` BF16; shared-expert output goes to `sh_down_out: [N]`.
extern "C" __global__ void moe_expert_silu_down_shared_t(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,   // [num_experts] device ptrs to [K/2 * N] bytes
    const unsigned long long* __restrict__ scale_t_ptrs,    // [num_experts] device ptrs to [K/16 * N] bytes
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert direct pointers (transposed layout)
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;
    if (is_shared) {
        if (sh_down_t_packed == 0) {
            // No shared expert — write zeros.
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) sh_down_out[n] = __float2bfloat16(0.0f);
            return;
        }
        B_packed = sh_down_t_packed;
        B_scale = sh_down_t_scale;
        s2 = sh_down_s2;
        g_ptr = sh_gate_in;
        u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_t_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_t_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        // EP remote expert: NULL pointer → write zero output and return.
        if (B_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[expert_slot * N + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool valid = (n < N);

    // Phase 1: cooperatively precompute s_act[K] = SiLU(gate[K]) * up[K].
    extern __shared__ float s_act[];
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    __syncthreads();

    if (!valid) return;

    // Phase 2: per-thread accumulate over K_half iterations. Each thread
    // owns ONE output position `n`; lanes in a warp have adjacent n's so
    // `B_packed[k_half * N + n]` reads are coalesced (1 byte per lane,
    // 32 bytes contiguous per warp per iter).
    const unsigned int num_groups = K / GROUP_SIZE;
    float acc = 0.0f;

    // Iterate scale groups (8 K_half iters per group). Cache the
    // per-group scale in a register.
    for (unsigned int sg = 0; sg < num_groups; sg++) {
        unsigned char sb = B_scale[(unsigned long long)sg * N + n];
        __nv_fp8_e4m3 fp8_s; *(unsigned char*)&fp8_s = sb;
        float sc = (float)fp8_s * s2;
        const unsigned int kh_base = sg * 8;
        #pragma unroll
        for (unsigned int kh_off = 0; kh_off < 8; kh_off++) {
            unsigned int k_half = kh_base + kh_off;
            unsigned char byte = B_packed[(unsigned long long)k_half * N + n];
            unsigned int nibble_lo = byte & 0xFu;
            unsigned int nibble_hi = (byte >> 4) & 0xFu;
            float w_lo = s_lut[nibble_lo] * sc;
            float w_hi = s_lut[nibble_hi] * sc;
            float a_lo = s_act[k_half * 2];
            float a_hi = s_act[k_half * 2 + 1];
            acc += a_lo * w_lo + a_hi * w_hi;
        }
    }

    // Output offset: routed → C[expert_slot * N + n]; shared → sh_down_out[n].
    if (is_shared) {
        sh_down_out[n] = __float2bfloat16(acc);
    } else {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}

// ── Lane-parallel transposed decode kernels (exact twins of the [N, K/2] ──
// ── decode kernels moe_expert_gate_up_shared / moe_expert_silu_down_shared) ──
//
// The one-thread-per-output kernels above walk all of K serially with
// single-byte loads and are latency bound. These twins reproduce the
// UNTRANSPOSED decode kernels' arithmetic bit for bit on the transposed
// layout: each output's 32 K-lanes (lane l owns k16 = l, l+32, ...) compute the
// same per-lane partial with the same expression, in the same k16 order, and
// the partials are combined by the same tree the reference's __shfl_down_sync
// reduction evaluates (offsets 16, 8, 4, 2, 1 as seen by lane 0).
//
// Block: 32 K-lanes x TPL threads; each thread owns OPT adjacent outputs,
// loaded as one OPT-byte word, so a warp reads 32/TPL rows of TPL*OPT
// contiguous bytes. UNR k16 steps are loaded before any is accumulated, to
// keep more loads in flight; accumulation order is unchanged.
// Grid: gate_up (ceil(N/(TPL*OPT)), top_k+1, 2); silu_down (.., top_k+1, 1).

#define T_LANES_K 32

template <int OPT> struct TLanesWord;
template <> struct TLanesWord<1> { typedef unsigned char T; };
template <> struct TLanesWord<2> { typedef unsigned short T; };
template <> struct TLanesWord<4> { typedef unsigned int T; };

template <int OPT>
__device__ __forceinline__ unsigned int t_lanes_load(const unsigned char* __restrict__ row, unsigned int n0,
                                                     unsigned int N) {
    if (n0 + OPT <= N && (((unsigned long long)(row + n0)) % OPT) == 0)
        return *(const typename TLanesWord<OPT>::T*)(row + n0);
    unsigned int w = 0;
    #pragma unroll
    for (int o = 0; o < OPT; o++)
        if (n0 + o < N) w |= (unsigned int)row[n0 + o] << (8 * o);
    return w;
}

template <int OB>
__device__ __forceinline__ float t_lanes_tree(float (*part)[OB + 1], unsigned int c) {
    float v[T_LANES_K];
    #pragma unroll
    for (int l = 0; l < T_LANES_K; l++) v[l] = part[l][c];
    #pragma unroll
    for (int off = T_LANES_K / 2; off > 0; off >>= 1) {
        #pragma unroll
        for (int l = 0; l < off; l++) v[l] = v[l] + v[l + off];
    }
    return v[0];
}

// One lane's partials for OPT outputs. ACT(k) yields the two activations of
// packed byte b of k16 (as floats) exactly as the reference computes them.
template <int OPT, int UNR, class Act>
__device__ __forceinline__ void t_lanes_accumulate(const unsigned char* __restrict__ B_packed,
                                                   const unsigned char* __restrict__ B_scale, float s2,
                                                   const float* __restrict__ s_lut, unsigned int l,
                                                   unsigned int n0, unsigned int N, unsigned int K,
                                                   Act act, float acc[OPT]) {
    const unsigned int K16 = K / 16;
    for (unsigned int kb = l; kb < K16; kb += T_LANES_K * UNR) {
        unsigned int sw[UNR], pw[UNR][8];
        #pragma unroll
        for (int u = 0; u < UNR; u++) {
            const unsigned int k16 = kb + u * T_LANES_K;
            if (k16 < K16) {
                sw[u] = t_lanes_load<OPT>(B_scale + (unsigned long long)k16 * N, n0, N);
                #pragma unroll
                for (int b = 0; b < 8; b++)
                    pw[u][b] = t_lanes_load<OPT>(B_packed + (unsigned long long)(k16 * 8 + b) * N, n0, N);
            }
        }
        #pragma unroll
        for (int u = 0; u < UNR; u++) {
            const unsigned int k16 = kb + u * T_LANES_K;
            if (k16 >= K16) break;
            float sc[OPT];
            #pragma unroll
            for (int o = 0; o < OPT; o++) {
                __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = (unsigned char)(sw[u] >> (8 * o));
                sc[o] = (float)fp8 * s2;
            }
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                float al, ah;
                act(k16, b, al, ah);
                #pragma unroll
                for (int o = 0; o < OPT; o++) {
                    const unsigned int byte = (pw[u][b] >> (8 * o)) & 0xFF;
                    const float wl = s_lut[byte & 0xF] * sc[o], wh = s_lut[byte >> 4] * sc[o];
                    acc[o] += al * wl + ah * wh;
                }
            }
        }
    }
}

template <int OPT, int TPL, int UNR>
__device__ __forceinline__ void gate_up_t_lanes_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    constexpr unsigned int OB = TPL * OPT;
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);
    const unsigned int q = threadIdx.x % TPL;
    const unsigned int l = threadIdx.x / TPL;
    const unsigned int n_block = blockIdx.x * OB;
    const unsigned int n0 = n_block + q * OPT;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;
    if (is_shared) {
        B_packed = proj == 0 ? sh_gate_t_packed : sh_up_t_packed;
        B_scale = proj == 0 ? sh_gate_t_scale : sh_up_t_scale;
        s2 = proj == 0 ? sh_gate_s2 : sh_up_s2;
        C = proj == 0 ? sh_gate_out : sh_up_out;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)(proj == 0 ? gate_packed_t_ptrs : up_packed_t_ptrs)[expert_id];
        B_scale = (const unsigned char*)(proj == 0 ? gate_scale_t_ptrs : up_scale_t_ptrs)[expert_id];
        s2 = (proj == 0 ? gate_scale2_vals : up_scale2_vals)[expert_id];
        C = (proj == 0 ? gate_out : up_out) + (unsigned long long)expert_slot * N;
    }
    if (B_packed == 0) {
        if (threadIdx.x < OB && n_block + threadIdx.x < N) C[n_block + threadIdx.x] = __float2bfloat16(0.0f);
        return;
    }

    __shared__ float s_lut[16];
    __shared__ float part[T_LANES_K][OB + 1];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    __syncthreads();

    float acc[OPT];
    #pragma unroll
    for (int o = 0; o < OPT; o++) acc[o] = 0.0f;
    if (n0 < N) {
        const unsigned int* A32 = (const unsigned int*)A;
        auto act = [&](unsigned int k16, int b, float& al, float& ah) {
            const unsigned int raw = A32[k16 * 8 + b];
            __nv_bfloat16 lo, hi;
            *(unsigned short*)&lo = (unsigned short)(raw & 0xFFFF);
            *(unsigned short*)&hi = (unsigned short)(raw >> 16);
            al = __bfloat162float(lo);
            ah = __bfloat162float(hi);
        };
        t_lanes_accumulate<OPT, UNR>(B_packed, B_scale, s2, s_lut, l, n0, N, K, act, acc);
    }
    #pragma unroll
    for (int o = 0; o < OPT; o++) part[l][q * OPT + o] = acc[o];
    __syncthreads();
    for (unsigned int c = threadIdx.x; c < OB; c += blockDim.x)
        if (n_block + c < N) C[n_block + c] = __float2bfloat16(t_lanes_tree<OB>(part, c));
}

template <int OPT, int TPL, int UNR>
__device__ __forceinline__ void silu_down_t_lanes_body(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    constexpr unsigned int OB = TPL * OPT;
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);
    const unsigned int q = threadIdx.x % TPL;
    const unsigned int l = threadIdx.x / TPL;
    const unsigned int n_block = blockIdx.x * OB;
    const unsigned int n0 = n_block + q * OPT;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;
    __nv_bfloat16* out;
    if (is_shared) {
        B_packed = sh_down_t_packed; B_scale = sh_down_t_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in; u_ptr = sh_up_in; out = sh_down_out;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_t_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_t_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        out = C + (unsigned long long)expert_slot * N;
    }
    if (B_packed == 0) {
        if (threadIdx.x < OB && n_block + threadIdx.x < N) out[n_block + threadIdx.x] = __float2bfloat16(0.0f);
        return;
    }

    __shared__ float s_lut[16];
    __shared__ float part[T_LANES_K][OB + 1];
    extern __shared__ float s_act[];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    for (unsigned int i = threadIdx.x; i < K; i += blockDim.x) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    float acc[OPT];
    #pragma unroll
    for (int o = 0; o < OPT; o++) acc[o] = 0.0f;
    if (n0 < N) {
        auto act = [&](unsigned int k16, int b, float& al, float& ah) {
            al = s_act[k16 * 16 + b * 2];
            ah = s_act[k16 * 16 + b * 2 + 1];
        };
        t_lanes_accumulate<OPT, UNR>(B_packed, B_scale, s2, s_lut, l, n0, N, K, act, acc);
    }
    #pragma unroll
    for (int o = 0; o < OPT; o++) part[l][q * OPT + o] = acc[o];
    __syncthreads();
    for (unsigned int c = threadIdx.x; c < OB; c += blockDim.x)
        if (n_block + c < N) out[n_block + c] = __float2bfloat16(t_lanes_tree<OB>(part, c));
}

#define T_LANES_GU_PARAMS \
    const __nv_bfloat16* __restrict__ A, const unsigned long long* __restrict__ gate_packed_t_ptrs, \
    const unsigned long long* __restrict__ gate_scale_t_ptrs, const float* __restrict__ gate_scale2_vals, \
    __nv_bfloat16* __restrict__ gate_out, const unsigned long long* __restrict__ up_packed_t_ptrs, \
    const unsigned long long* __restrict__ up_scale_t_ptrs, const float* __restrict__ up_scale2_vals, \
    __nv_bfloat16* __restrict__ up_out, const unsigned int* __restrict__ expert_indices, \
    const unsigned char* __restrict__ sh_gate_t_packed, const unsigned char* __restrict__ sh_gate_t_scale, \
    float sh_gate_s2, __nv_bfloat16* __restrict__ sh_gate_out, const unsigned char* __restrict__ sh_up_t_packed, \
    const unsigned char* __restrict__ sh_up_t_scale, float sh_up_s2, __nv_bfloat16* __restrict__ sh_up_out, \
    unsigned int N, unsigned int K, unsigned int top_k
#define T_LANES_GU_ARGS \
    A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out, up_packed_t_ptrs, up_scale_t_ptrs, \
    up_scale2_vals, up_out, expert_indices, sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2, sh_gate_out, \
    sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out, N, K, top_k
#define T_LANES_DN_PARAMS \
    const __nv_bfloat16* __restrict__ gate_out, const __nv_bfloat16* __restrict__ up_out, \
    const unsigned long long* __restrict__ packed_t_ptrs, const unsigned long long* __restrict__ scale_t_ptrs, \
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C, const unsigned int* __restrict__ expert_indices, \
    const __nv_bfloat16* __restrict__ sh_gate_in, const __nv_bfloat16* __restrict__ sh_up_in, \
    const unsigned char* __restrict__ sh_down_t_packed, const unsigned char* __restrict__ sh_down_t_scale, \
    float sh_down_s2, __nv_bfloat16* __restrict__ sh_down_out, unsigned int N, unsigned int K, unsigned int top_k
#define T_LANES_DN_ARGS \
    gate_out, up_out, packed_t_ptrs, scale_t_ptrs, scale2_vals, C, expert_indices, sh_gate_in, sh_up_in, \
    sh_down_t_packed, sh_down_t_scale, sh_down_s2, sh_down_out, N, K, top_k

// Variant name suffix: o<OPT>t<TPL>u<UNR>. Block = 32*TPL threads.
#define T_LANES_VARIANT(OPT, TPL, UNR) \
    extern "C" __global__ void __launch_bounds__(32 * TPL) \
    moe_expert_gate_up_shared_t_lanes_o##OPT##t##TPL##u##UNR(T_LANES_GU_PARAMS) { \
        gate_up_t_lanes_body<OPT, TPL, UNR>(T_LANES_GU_ARGS); } \
    extern "C" __global__ void __launch_bounds__(32 * TPL) \
    moe_expert_silu_down_shared_t_lanes_o##OPT##t##TPL##u##UNR(T_LANES_DN_PARAMS) { \
        silu_down_t_lanes_body<OPT, TPL, UNR>(T_LANES_DN_ARGS); }

T_LANES_VARIANT(4, 16, 1)
T_LANES_VARIANT(4, 16, 2)
T_LANES_VARIANT(4, 32, 1)
T_LANES_VARIANT(2, 32, 1)
T_LANES_VARIANT(2, 32, 2)

// ---------------------------------------------------------------------------
// Multi-row lane-parallel `_t` MoE for Qwen4 exact K4/K5/K9 verify on the
// unified layout (fn-decode). Same buffer layouts as the batch3/dedup kernels
// (routed at (token*top_k+slot)*N, shared at token*N). grid.y = rows*top_k
// routed units + 1 shared unit. A routed unit owns its expert only if no
// earlier unit selected the same expert (others exit); the owner streams the
// expert's weights ONCE and runs, for every row that selected it, exactly the
// single-row t_lanes arithmetic (same lane l, same k16 order, same per-step
// expression, same tree), so each row's output is byte-identical to
// moe_expert_*_shared_t_lanes_o4t16u1 on that row, which reproduces the
// untransposed K=1 kernels bit for bit.
// ---------------------------------------------------------------------------
// Atlas-TS4E-128 lossless 4-bit scale decode helpers (OPT in {2, 4}).
// Fast path uses carry-free SWAR addition (`base <= 241`, `code <= 14` =>
// `base + code <= 255`) when no lane has the 0xF escape code; cold path
// binary-searches the bucket's sorted exception table with 0-ULP exactness.
template <int OPT, unsigned int KC = 0u>
__device__ __noinline__ unsigned int ts4e_patch_word_noinline(
    const unsigned char* __restrict__ B_scale,
    unsigned int base_word,
    unsigned int codes,
    unsigned int k16_n0,
    unsigned int N_and_k32
) {
    static_assert(OPT == 2 || OPT == 4, "TS4E SWAR decode supports OPT in {2, 4}");
    const unsigned int k16 = k16_n0 >> 16;
    const unsigned int n0 = k16_n0 & 0xFFFFu;
    const unsigned int N = N_and_k32 & 0xFFFFu;
    const unsigned int num_k32 = (KC != 0u) ? (KC >> 5) : (N_and_k32 >> 16);
    const unsigned int n_tiles = N >> 7;
    const unsigned int num_bkts = num_k32 * n_tiles;
    const unsigned int nibbles_bytes = num_k32 * N;
    const unsigned int page_bases_len = ((num_bkts >> 7) + 4u) & ~3u;
    const unsigned int page_bases_off = nibbles_bytes + N;
    const unsigned int bkt_off_off = page_bases_off + page_bases_len * 4u;
    const unsigned int bkt_off_bytes_aligned = (((num_bkts + 1u) * 2u) + 15u) & ~15u;
    const unsigned int exc_off = bkt_off_off + bkt_off_bytes_aligned;

    const unsigned int bkt = (k16 >> 1) * n_tiles + (n0 >> 7);
    const unsigned int* page_bases =
        reinterpret_cast<const unsigned int*>(B_scale + page_bases_off);
    const unsigned short* bkt_off =
        reinterpret_cast<const unsigned short*>(B_scale + bkt_off_off);
    const unsigned short* exc =
        reinterpret_cast<const unsigned short*>(B_scale + exc_off);

    const unsigned int beg = page_bases[bkt >> 7] + static_cast<unsigned int>(bkt_off[bkt]);
    const unsigned int end =
        page_bases[(bkt + 1u) >> 7] + static_cast<unsigned int>(bkt_off[bkt + 1u]);
    const unsigned int k_bit_hi = (k16 & 1u) << 7;

    unsigned int out = 0u;
    #pragma unroll
    for (int o = 0; o < OPT; ++o) {
        const unsigned int c = (codes >> (8 * o)) & 0xFFu;
        unsigned int val;
        if (c != 15u) {
            val = ((base_word >> (8 * o)) & 0xFFu) + c;
        } else {
            const unsigned int target = k_bit_hi | ((n0 + static_cast<unsigned int>(o)) & 127u);
            unsigned int lo = beg;
            unsigned int hi = end;
            while (lo < hi) {
                const unsigned int mid = lo + ((hi - lo) >> 1);
                if ((static_cast<unsigned int>(exc[mid]) & 0xFFu) < target) {
                    lo = mid + 1u;
                } else {
                    hi = mid;
                }
            }
            val = (lo < end) ? (static_cast<unsigned int>(exc[lo]) >> 8) : 0u;
        }
        out |= (val & 0xFFu) << (8 * o);
    }
    return out;
}

template <int OPT, unsigned int KC = 0u>
__device__ __forceinline__ unsigned int ts4e_decode_scale_word(
    const unsigned char* __restrict__ B_scale,
    unsigned int base_word,
    unsigned int raw,
    unsigned int k16,
    unsigned int n0,
    unsigned int N,
    unsigned int K = KC
) {
    static_assert(OPT == 2 || OPT == 4, "TS4E SWAR decode supports OPT in {2, 4}");
    constexpr unsigned int MASK_LO = (OPT == 4) ? 0x0F0F0F0Fu : 0x00000F0Fu;
    constexpr unsigned int ADD1    = (OPT == 4) ? 0x01010101u : 0x00000101u;
    constexpr unsigned int HI4     = (OPT == 4) ? 0x10101010u : 0x00001010u;
    const unsigned int codes = ((k16 & 1u) ? (raw >> 4) : raw) & MASK_LO;
    if (__builtin_expect(((codes + ADD1) & HI4) == 0u, 1)) {
        return base_word + codes;
    }
    const unsigned int k16_n0 = (k16 << 16) | (n0 & 0xFFFFu);
    const unsigned int n_and_k32 = (KC != 0u) ? (N & 0xFFFFu) : (((K >> 5) << 16) | (N & 0xFFFFu));
    return ts4e_patch_word_noinline<OPT, KC>(B_scale, base_word, codes, k16_n0, n_and_k32);
}

template <int OPT, unsigned int KC = 0u>
__device__ __forceinline__ unsigned int ts4e_load_scale_word(
    const unsigned char* __restrict__ B_scale,
    unsigned int base_word,
    unsigned int k16,
    unsigned int n0,
    unsigned int N,
    unsigned int K = KC
) {
    static_assert(OPT == 2 || OPT == 4, "TS4E SWAR decode supports OPT in {2, 4}");
    const unsigned int raw =
        t_lanes_load<OPT>(B_scale + static_cast<unsigned long long>(k16 >> 1) * N, n0, N);
    return ts4e_decode_scale_word<OPT, KC>(B_scale, base_word, raw, k16, n0, N, K);
}

#define T_LANES_RMAX 9

template <int OPT, int RMAX, int ACTIVE_RMAX = RMAX, bool TS4E = false, class ActM>
__device__ __forceinline__ void t_lanes_accumulate_rows(const unsigned char* __restrict__ B_packed,
                                                        const unsigned char* __restrict__ B_scale, float s2,
                                                        const float* __restrict__ s_lut, unsigned int l,
                                                        unsigned int n0, unsigned int N, unsigned int K,
                                                        int cnt, ActM act, float acc[RMAX][OPT],
                                                        const unsigned int* __restrict__ s_base = nullptr,
                                                        unsigned int base_idx = 0u,
                                                        bool use_ts4e = false) {
    const unsigned int K16 = K / 16;
    if constexpr (!TS4E && ACTIVE_RMAX <= 3) {
        if (l < K16) {
            unsigned int sw_cur = t_lanes_load<OPT>(B_scale + (unsigned long long)l * N, n0, N);
            unsigned int pw_cur[8];
            #pragma unroll
            for (int b = 0; b < 8; b++)
                pw_cur[b] = t_lanes_load<OPT>(B_packed + (unsigned long long)(l * 8 + b) * N, n0, N);
            for (unsigned int k16 = l; k16 < K16; k16 += T_LANES_K) {
                const unsigned int next_k16 = k16 + T_LANES_K;
                unsigned int sw_next = 0u;
                unsigned int pw_next[8];
                if (next_k16 < K16) {
                    sw_next = t_lanes_load<OPT>(B_scale + (unsigned long long)next_k16 * N, n0, N);
                    #pragma unroll
                    for (int b = 0; b < 8; b++)
                        pw_next[b] = t_lanes_load<OPT>(B_packed + (unsigned long long)(next_k16 * 8 + b) * N, n0, N);
                }
                float sc[OPT];
                #pragma unroll
                for (int o = 0; o < OPT; o++) {
                    __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = (unsigned char)(sw_cur >> (8 * o));
                    sc[o] = (float)fp8 * s2;
                }
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    float wl[OPT], wh[OPT];
                    #pragma unroll
                    for (int o = 0; o < OPT; o++) {
                        const unsigned int byte = (pw_cur[b] >> (8 * o)) & 0xFF;
                        wl[o] = s_lut[byte & 0xF] * sc[o];
                        wh[o] = s_lut[byte >> 4] * sc[o];
                    }
                    #pragma unroll
                    for (int m = 0; m < ACTIVE_RMAX; m++) {
                        if (ACTIVE_RMAX < RMAX || ACTIVE_RMAX == 1 || m < cnt) {
                            float al, ah;
                            act(m, k16, b, al, ah);
                            #pragma unroll
                            for (int o = 0; o < OPT; o++) acc[m][o] += al * wl[o] + ah * wh[o];
                        }
                    }
                }
                sw_cur = sw_next;
                #pragma unroll
                for (int b = 0; b < 8; b++) pw_cur[b] = pw_next[b];
            }
        }
    } else if constexpr (TS4E && ACTIVE_RMAX <= 3) {
        // Low-register software pipelining for TS4E small-row tiers (ACTIVE_RMAX <= 3):
        // overlap next-k16 raw scale fetch + 2-word double-buffered packed nibble loads
        // (`pw_cur[2]`/`pw_next[2]`, 4 regs total matching `pw[4]` and only 2 live across the
        // cold `ts4e_patch_word_noinline` call) to keep 0 B spills even under 64-reg r2_ts4e.
        auto load_scale_raw = [&](unsigned int k16_idx) -> unsigned int {
            if (use_ts4e) {
                return t_lanes_load<OPT>(B_scale + static_cast<unsigned long long>(k16_idx >> 1) * N, n0, N);
            }
            return t_lanes_load<OPT>(B_scale + static_cast<unsigned long long>(k16_idx) * N, n0, N);
        };
        auto decode_scale_raw = [&](unsigned int raw, unsigned int k16_idx) -> unsigned int {
            if (use_ts4e) {
                return ts4e_decode_scale_word<OPT>(B_scale, s_base[base_idx], raw, k16_idx, n0, N, K);
            }
            return raw;
        };
        if (l < K16) {
            unsigned int raw_cur = load_scale_raw(l);
            unsigned int pw_cur[2];
            pw_cur[0] = t_lanes_load<OPT>(B_packed + static_cast<unsigned long long>(l * 8u) * N, n0, N);
            pw_cur[1] = t_lanes_load<OPT>(B_packed + static_cast<unsigned long long>(l * 8u + 1u) * N, n0, N);
            for (unsigned int k16 = l; k16 < K16; k16 += T_LANES_K) {
                const unsigned int sw = decode_scale_raw(raw_cur, k16);
                const unsigned int next_k16 = k16 + T_LANES_K;
                const unsigned int raw_next = (next_k16 < K16) ? load_scale_raw(next_k16) : 0u;
                float sc[OPT];
                #pragma unroll
                for (int o = 0; o < OPT; o++) {
                    __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = (unsigned char)(sw >> (8 * o));
                    sc[o] = (float)fp8 * s2;
                }
                #pragma unroll 1
                for (int bh = 0; bh < 4; bh++) {
                    unsigned int pw_next[2] = {0u, 0u};
                    const bool has_next_pw = (bh < 3) || (next_k16 < K16);
                    if (has_next_pw) {
                        const unsigned int next_row = (bh < 3)
                            ? (k16 * 8u + static_cast<unsigned int>((bh + 1) * 2))
                            : (next_k16 * 8u);
                        pw_next[0] = t_lanes_load<OPT>(B_packed + static_cast<unsigned long long>(next_row) * N, n0, N);
                        pw_next[1] = t_lanes_load<OPT>(B_packed + static_cast<unsigned long long>(next_row + 1u) * N, n0, N);
                    }
                    #pragma unroll
                    for (int bi = 0; bi < 2; bi++) {
                        const int b = bh * 2 + bi;
                        float wl[OPT], wh[OPT];
                        #pragma unroll
                        for (int o = 0; o < OPT; o++) {
                            const unsigned int byte = (pw_cur[bi] >> (8 * o)) & 0xFF;
                            wl[o] = s_lut[byte & 0xF] * sc[o];
                            wh[o] = s_lut[byte >> 4] * sc[o];
                        }
                        #pragma unroll
                        for (int m = 0; m < ACTIVE_RMAX; m++) {
                            if (ACTIVE_RMAX < RMAX || ACTIVE_RMAX == 1 || m < cnt) {
                                float al, ah;
                                act(m, k16, b, al, ah);
                                #pragma unroll
                                for (int o = 0; o < OPT; o++) acc[m][o] += al * wl[o] + ah * wh[o];
                            }
                        }
                    }
                    if (has_next_pw) {
                        pw_cur[0] = pw_next[0];
                        pw_cur[1] = pw_next[1];
                    }
                }
                raw_cur = raw_next;
            }
        }
    } else {
        auto load_scale = [&](unsigned int k16_idx) -> unsigned int {
            if constexpr (TS4E) {
                if (use_ts4e) {
                    return ts4e_load_scale_word<OPT>(B_scale, s_base[base_idx], k16_idx, n0, N, K);
                }
            }
            return t_lanes_load<OPT>(B_scale + (unsigned long long)k16_idx * N, n0, N);
        };
        for (unsigned int k16 = l; k16 < K16; k16 += T_LANES_K) {
            const unsigned int sw = load_scale(k16);
            float sc[OPT];
            #pragma unroll
            for (int o = 0; o < OPT; o++) {
                __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = (unsigned char)(sw >> (8 * o));
                sc[o] = (float)fp8 * s2;
            }
            if constexpr (ACTIVE_RMAX >= 6 && ACTIVE_RMAX <= 8) {
                #pragma unroll 1
                for (int bh = 0; bh < 4; bh++) {
                    unsigned int pw[2];
                    #pragma unroll
                    for (int bi = 0; bi < 2; bi++)
                        pw[bi] = t_lanes_load<OPT>(B_packed + (unsigned long long)(k16 * 8 + bh * 2 + bi) * N, n0, N);
                    #pragma unroll
                    for (int bi = 0; bi < 2; bi++) {
                        const int b = bh * 2 + bi;
                        float wl[OPT], wh[OPT];
                        #pragma unroll
                        for (int o = 0; o < OPT; o++) {
                            const unsigned int byte = (pw[bi] >> (8 * o)) & 0xFF;
                            wl[o] = s_lut[byte & 0xF] * sc[o];
                            wh[o] = s_lut[byte >> 4] * sc[o];
                        }
                        #pragma unroll
                        for (int m = 0; m < ACTIVE_RMAX; m++) {
                            if (ACTIVE_RMAX < RMAX || ACTIVE_RMAX == 1 || m < cnt) {
                                float al, ah;
                                act(m, k16, b, al, ah);
                                #pragma unroll
                                for (int o = 0; o < OPT; o++) acc[m][o] += al * wl[o] + ah * wh[o];
                            }
                        }
                    }
                }
            } else if constexpr (ACTIVE_RMAX <= 5) {
                #pragma unroll 1
                for (int bh = 0; bh < 2; bh++) {
                    unsigned int pw[4];
                    #pragma unroll
                    for (int bi = 0; bi < 4; bi++)
                        pw[bi] = t_lanes_load<OPT>(B_packed + (unsigned long long)(k16 * 8 + bh * 4 + bi) * N, n0, N);
                    #pragma unroll
                    for (int bi = 0; bi < 4; bi++) {
                        const int b = bh * 4 + bi;
                        float wl[OPT], wh[OPT];
                        #pragma unroll
                        for (int o = 0; o < OPT; o++) {
                            const unsigned int byte = (pw[bi] >> (8 * o)) & 0xFF;
                            wl[o] = s_lut[byte & 0xF] * sc[o];
                            wh[o] = s_lut[byte >> 4] * sc[o];
                        }
                        #pragma unroll
                        for (int m = 0; m < ACTIVE_RMAX; m++) {
                            if (ACTIVE_RMAX < RMAX || ACTIVE_RMAX == 1 || m < cnt) {
                                float al, ah;
                                act(m, k16, b, al, ah);
                                #pragma unroll
                                for (int o = 0; o < OPT; o++) acc[m][o] += al * wl[o] + ah * wh[o];
                            }
                        }
                    }
                }
            } else {
                unsigned int pw[8];
                #pragma unroll
                for (int b = 0; b < 8; b++)
                    pw[b] = t_lanes_load<OPT>(B_packed + (unsigned long long)(k16 * 8 + b) * N, n0, N);
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    float wl[OPT], wh[OPT];
                    #pragma unroll
                    for (int o = 0; o < OPT; o++) {
                        const unsigned int byte = (pw[b] >> (8 * o)) & 0xFF;
                        wl[o] = s_lut[byte & 0xF] * sc[o];
                        wh[o] = s_lut[byte >> 4] * sc[o];
                    }
                    #pragma unroll
                    for (int m = 0; m < ACTIVE_RMAX; m++) {
                        if (ACTIVE_RMAX == 1 || m < cnt) {
                            float al, ah;
                            act(m, k16, b, al, ah);
                            #pragma unroll
                            for (int o = 0; o < OPT; o++) acc[m][o] += al * wl[o] + ah * wh[o];
                        }
                    }
                }
            }
        }
    }
}

// Resolve the unit's members. Returns false if this unit is a duplicate (exit).
template <int RMAX>
__device__ __forceinline__ bool t_lanes_rows_members(const unsigned int* __restrict__ expert_indices,
                                                     unsigned int rows, unsigned int top_k,
                                                     int* s_cnt, unsigned int* s_token, unsigned int* s_flat) {
    const unsigned int total = rows * top_k;
    const unsigned int y = blockIdx.y;
    if (y < total) {
        const unsigned int lane = threadIdx.x & 31u;
        const unsigned int e = expert_indices[y];
        for (unsigned int base = 0; base < y; base += 32u) {
            const unsigned int jj = base + lane;
            if (__ballot_sync(0xFFFFFFFFu, jj < y && expert_indices[jj] == e) != 0u) return false;
        }
        if (threadIdx.x < 32u) {
            int cnt = 0;
            for (unsigned int base = (y & ~31u); base < total && cnt < RMAX; base += 32u) {
                const unsigned int jj = base + lane;
                unsigned int mask = __ballot_sync(0xFFFFFFFFu, jj >= y && jj < total && expert_indices[jj] == e);
                while (mask != 0u && cnt < RMAX) {
                    const int bit = __ffs(mask) - 1;
                    if (lane == 0u) {
                        const unsigned int j = base + (unsigned int)bit;
                        s_token[cnt] = j / top_k;
                        s_flat[cnt] = j;
                    }
                    cnt++;
                    mask &= mask - 1u;
                }
            }
            if (lane == 0u) *s_cnt = cnt;
        }
    } else if (threadIdx.x == 0) {
        *s_cnt = (int)rows;
        for (unsigned int r = 0; r < rows; r++) { s_token[r] = r; s_flat[r] = r; }
    }
    __syncthreads();
    return *s_cnt > 0;
}

template <int OPT, int TPL, int RMAX>
__device__ __forceinline__ void gate_up_t_lanes_rows_body(T_LANES_GU_PARAMS, unsigned int rows) {
    constexpr unsigned int OB = TPL * OPT;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (blockIdx.y == rows * top_k);
    const unsigned int q = threadIdx.x % TPL;
    const unsigned int l = threadIdx.x / TPL;
    const unsigned int n_block = blockIdx.x * OB;
    const unsigned int n0 = n_block + q * OPT;
    __shared__ int s_cnt;
    __shared__ unsigned int s_token[RMAX], s_flat[RMAX];
    if (!t_lanes_rows_members<RMAX>(expert_indices, rows, top_k, &s_cnt, s_token, s_flat)) return;
    const int cnt = s_cnt;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* Cbase;
    if (is_shared) {
        B_packed = proj == 0 ? sh_gate_t_packed : sh_up_t_packed;
        B_scale = proj == 0 ? sh_gate_t_scale : sh_up_t_scale;
        s2 = proj == 0 ? sh_gate_s2 : sh_up_s2;
        Cbase = proj == 0 ? sh_gate_out : sh_up_out;
    } else {
        const unsigned int expert_id = expert_indices[blockIdx.y];
        B_packed = (const unsigned char*)(proj == 0 ? gate_packed_t_ptrs : up_packed_t_ptrs)[expert_id];
        B_scale = (const unsigned char*)(proj == 0 ? gate_scale_t_ptrs : up_scale_t_ptrs)[expert_id];
        s2 = (proj == 0 ? gate_scale2_vals : up_scale2_vals)[expert_id];
        Cbase = proj == 0 ? gate_out : up_out;
    }
    if (B_packed == 0) {
        for (int m = 0; m < cnt; m++)
            if (threadIdx.x < OB && n_block + threadIdx.x < N)
                Cbase[(unsigned long long)s_flat[m] * N + n_block + threadIdx.x] = __float2bfloat16(0.0f);
        return;
    }
    __shared__ float s_lut[16];
    __shared__ float part[T_LANES_K][OB + 1];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    __syncthreads();

    float acc[RMAX][OPT];
    #pragma unroll
    for (int m = 0; m < RMAX; m++)
        #pragma unroll
        for (int o = 0; o < OPT; o++) acc[m][o] = 0.0f;
    if (n0 < N) {
        auto act = [&](int m, unsigned int k16, int b, float& al, float& ah) {
            const unsigned int* A32 = (const unsigned int*)(A + (unsigned long long)s_token[m] * K);
            const unsigned int raw = A32[k16 * 8 + b];
            __nv_bfloat16 lo, hi;
            *(unsigned short*)&lo = (unsigned short)(raw & 0xFFFF);
            *(unsigned short*)&hi = (unsigned short)(raw >> 16);
            al = __bfloat162float(lo);
            ah = __bfloat162float(hi);
        };
        t_lanes_accumulate_rows<OPT, RMAX>(B_packed, B_scale, s2, s_lut, l, n0, N, K, cnt, act, acc);
    }
    #pragma unroll
    for (int m = 0; m < RMAX; m++) {
        if (m >= cnt) break;
        if (m > 0) __syncthreads();
        #pragma unroll
        for (int o = 0; o < OPT; o++) part[l][q * OPT + o] = acc[m][o];
        __syncthreads();
        __nv_bfloat16* C = Cbase + (unsigned long long)s_flat[m] * N;
        if (blockDim.x == 8 * OB) {
            const unsigned int c = threadIdx.x >> 3;
            const unsigned int j = threadIdx.x & 7u;
            const float v0 = part[j][c] + part[j + 16u][c];
            const float v1 = part[j + 8u][c] + part[j + 24u][c];
            float v = v0 + v1;
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 4, 8);
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 2, 8);
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 1, 8);
            if (j == 0u && n_block + c < N) C[n_block + c] = __float2bfloat16(v);
        } else {
            for (unsigned int c = threadIdx.x; c < OB; c += blockDim.x)
                if (n_block + c < N) C[n_block + c] = __float2bfloat16(t_lanes_tree<OB>(part, c));
        }
    }
}

template <int OPT, int TPL, int RMAX>
__device__ __forceinline__ void silu_down_t_lanes_rows_body(T_LANES_DN_PARAMS, unsigned int rows) {
    constexpr unsigned int OB = TPL * OPT;
    const bool is_shared = (blockIdx.y == rows * top_k);
    const unsigned int q = threadIdx.x % TPL;
    const unsigned int l = threadIdx.x / TPL;
    const unsigned int n_block = blockIdx.x * OB;
    const unsigned int n0 = n_block + q * OPT;
    __shared__ int s_cnt;
    __shared__ unsigned int s_token[RMAX], s_flat[RMAX];
    if (!t_lanes_rows_members<RMAX>(expert_indices, rows, top_k, &s_cnt, s_token, s_flat)) return;
    const int cnt = s_cnt;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* out_base;
    if (is_shared) {
        B_packed = sh_down_t_packed; B_scale = sh_down_t_scale; s2 = sh_down_s2; out_base = sh_down_out;
    } else {
        const unsigned int expert_id = expert_indices[blockIdx.y];
        B_packed = (const unsigned char*)packed_t_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_t_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        out_base = C;
    }
    if (B_packed == 0) {
        for (int m = 0; m < cnt; m++)
            if (threadIdx.x < OB && n_block + threadIdx.x < N)
                out_base[(unsigned long long)s_flat[m] * N + n_block + threadIdx.x] = __float2bfloat16(0.0f);
        return;
    }
    __shared__ float s_lut[16];
    __shared__ float part[T_LANES_K][OB + 1];
    extern __shared__ float s_act_rows[];  // [cnt][K]
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    for (int m = 0; m < cnt; m++) {
        const __nv_bfloat16* g_ptr = is_shared ? sh_gate_in + (unsigned long long)s_token[m] * K
                                               : gate_out + (unsigned long long)s_flat[m] * K;
        const __nv_bfloat16* u_ptr = is_shared ? sh_up_in + (unsigned long long)s_token[m] * K
                                               : up_out + (unsigned long long)s_flat[m] * K;
        for (unsigned int i = threadIdx.x; i < K; i += blockDim.x) {
            float gf = __bfloat162float(g_ptr[i]);
            float uf = __bfloat162float(u_ptr[i]);
            s_act_rows[m * K + i] = (gf / (1.0f + __expf(-gf))) * uf;
        }
    }
    __syncthreads();

    float acc[RMAX][OPT];
    #pragma unroll
    for (int m = 0; m < RMAX; m++)
        #pragma unroll
        for (int o = 0; o < OPT; o++) acc[m][o] = 0.0f;
    if (n0 < N) {
        auto act = [&](int m, unsigned int k16, int b, float& al, float& ah) {
            al = s_act_rows[m * K + k16 * 16 + b * 2];
            ah = s_act_rows[m * K + k16 * 16 + b * 2 + 1];
        };
        t_lanes_accumulate_rows<OPT, RMAX>(B_packed, B_scale, s2, s_lut, l, n0, N, K, cnt, act, acc);
    }
    #pragma unroll
    for (int m = 0; m < RMAX; m++) {
        if (m >= cnt) break;
        if (m > 0) __syncthreads();
        #pragma unroll
        for (int o = 0; o < OPT; o++) part[l][q * OPT + o] = acc[m][o];
        __syncthreads();
        __nv_bfloat16* out = out_base + (unsigned long long)s_flat[m] * N;
        for (unsigned int c = threadIdx.x; c < OB; c += blockDim.x)
            if (n_block + c < N) out[n_block + c] = __float2bfloat16(t_lanes_tree<OB>(part, c));
    }
}

// grid (ceil(N/64), rows*top_k + 1, 2), block 512; rows <= T_LANES_RMAX.
extern "C" __global__ void __launch_bounds__(512) moe_expert_gate_up_shared_t_lanes_rows_o4t16(
    T_LANES_GU_PARAMS, unsigned int rows) {
    gate_up_t_lanes_rows_body<4, 16, T_LANES_RMAX>(T_LANES_GU_ARGS, rows);
}
// grid (ceil(N/64), rows*top_k + 1, 1), block 512, dynamic smem rows*K*4.
extern "C" __global__ void __launch_bounds__(512) moe_expert_silu_down_shared_t_lanes_rows_o4t16(
    T_LANES_DN_PARAMS, unsigned int rows) {
    silu_down_t_lanes_rows_body<4, 16, T_LANES_RMAX>(T_LANES_DN_ARGS, rows);
}


// Row-count-specialized twins of the *_t_lanes_rows kernels (fn-decode): RMAX =
// the verify width (2/3/4/5/9) so K5 does not carry 9 predicated accumulator
// rows, and gate/up stages each member row's input in shared memory (raw BF16
// pairs, the same values the global read returns). Arithmetic per row is
// unchanged => byte-identical to the RMAX=9 kernels and to single-row _t_lanes.
template <int OPT, int TPL, int RMAX>
__device__ __forceinline__ void gate_up_t_lanes_rows_staged_body(T_LANES_GU_PARAMS, unsigned int rows) {
    constexpr unsigned int OB = TPL * OPT;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (blockIdx.y == rows * top_k);
    const unsigned int q = threadIdx.x % TPL;
    const unsigned int l = threadIdx.x / TPL;
    const unsigned int n_block = blockIdx.x * OB;
    const unsigned int n0 = n_block + q * OPT;
    __shared__ int s_cnt;
    __shared__ unsigned int s_token[RMAX], s_flat[RMAX];
    if (!t_lanes_rows_members<RMAX>(expert_indices, rows, top_k, &s_cnt, s_token, s_flat)) return;
    const int cnt = s_cnt;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* Cbase;
    if (is_shared) {
        B_packed = proj == 0 ? sh_gate_t_packed : sh_up_t_packed;
        B_scale = proj == 0 ? sh_gate_t_scale : sh_up_t_scale;
        s2 = proj == 0 ? sh_gate_s2 : sh_up_s2;
        Cbase = proj == 0 ? sh_gate_out : sh_up_out;
    } else {
        const unsigned int expert_id = expert_indices[blockIdx.y];
        B_packed = (const unsigned char*)(proj == 0 ? gate_packed_t_ptrs : up_packed_t_ptrs)[expert_id];
        B_scale = (const unsigned char*)(proj == 0 ? gate_scale_t_ptrs : up_scale_t_ptrs)[expert_id];
        s2 = (proj == 0 ? gate_scale2_vals : up_scale2_vals)[expert_id];
        Cbase = proj == 0 ? gate_out : up_out;
    }
    if (B_packed == 0) {
        for (int m = 0; m < cnt; m++)
            if (threadIdx.x < OB && n_block + threadIdx.x < N)
                Cbase[(unsigned long long)s_flat[m] * N + n_block + threadIdx.x] = __float2bfloat16(0.0f);
        return;
    }
    __shared__ float s_lut[16];
    __shared__ float part[T_LANES_K][OB + 1];
    extern __shared__ __align__(16) unsigned int s_a32[];  // [cnt][K/2] raw BF16 pairs
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    const unsigned int K2 = K / 2;
    if ((K & 7u) == 0u) {
        const unsigned int K8 = K >> 3;
        uint4* s_a128 = reinterpret_cast<uint4*>(s_a32);
        for (int m = 0; m < cnt; m++) {
            const uint4* A128 = reinterpret_cast<const uint4*>(A + (unsigned long long)s_token[m] * K);
            for (unsigned int i = threadIdx.x; i < K8; i += blockDim.x) s_a128[m * K8 + i] = __ldg(&A128[i]);
        }
    } else {
        for (int m = 0; m < cnt; m++) {
            const unsigned int* A32 = (const unsigned int*)(A + (unsigned long long)s_token[m] * K);
            for (unsigned int i = threadIdx.x; i < K2; i += blockDim.x) s_a32[m * K2 + i] = __ldg(&A32[i]);
        }
    }
    __syncthreads();

    float acc[RMAX][OPT];
    #pragma unroll
    for (int m = 0; m < RMAX; m++)
        #pragma unroll
        for (int o = 0; o < OPT; o++) acc[m][o] = 0.0f;
    if (n0 < N) {
        auto act = [&](int m, unsigned int k16, int b, float& al, float& ah) {
            const unsigned int raw = s_a32[m * K2 + k16 * 8 + b];
            __nv_bfloat16 lo, hi;
            *(unsigned short*)&lo = (unsigned short)(raw & 0xFFFF);
            *(unsigned short*)&hi = (unsigned short)(raw >> 16);
            al = __bfloat162float(lo);
            ah = __bfloat162float(hi);
        };
        t_lanes_accumulate_rows<OPT, RMAX>(B_packed, B_scale, s2, s_lut, l, n0, N, K, cnt, act, acc);
    }
    #pragma unroll
    for (int m = 0; m < RMAX; m++) {
        if (m >= cnt) break;
        if (m > 0) __syncthreads();
        #pragma unroll
        for (int o = 0; o < OPT; o++) part[l][q * OPT + o] = acc[m][o];
        __syncthreads();
        __nv_bfloat16* C = Cbase + (unsigned long long)s_flat[m] * N;
        if (blockDim.x == 8 * OB) {
            const unsigned int c = threadIdx.x >> 3;
            const unsigned int j = threadIdx.x & 7u;
            const float v0 = part[j][c] + part[j + 16u][c];
            const float v1 = part[j + 8u][c] + part[j + 24u][c];
            float v = v0 + v1;
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 4, 8);
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 2, 8);
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 1, 8);
            if (j == 0u && n_block + c < N) C[n_block + c] = __float2bfloat16(v);
        } else {
            for (unsigned int c = threadIdx.x; c < OB; c += blockDim.x)
                if (n_block + c < N) C[n_block + c] = __float2bfloat16(t_lanes_tree<OB>(part, c));
        }
    }
}

#define T_LANES_ROWS_R(R, MINB) \
    extern "C" __global__ void __launch_bounds__(512, MINB) moe_expert_gate_up_shared_t_lanes_rows_r##R( \
        T_LANES_GU_PARAMS, unsigned int rows) { \
        gate_up_t_lanes_rows_staged_body<4, 16, R>(T_LANES_GU_ARGS, rows); } \
    extern "C" __global__ void __launch_bounds__(512, MINB) moe_expert_silu_down_shared_t_lanes_rows_r##R( \
        T_LANES_DN_PARAMS, unsigned int rows) { \
        silu_down_t_lanes_rows_body<4, 16, R>(T_LANES_DN_ARGS, rows); }
// gate_up: dynamic smem rows*K*2 B (staged input); down: rows*K*4 B (SiLU*up).
T_LANES_ROWS_R(2, 2)
T_LANES_ROWS_R(3, 2)
T_LANES_ROWS_R(4, 2)
T_LANES_ROWS_R(5, 2)
T_LANES_ROWS_R(9, 1)

// Occupancy variants of the row-specialized twins (fn-decode): OPT (outputs per
// thread) and TPL (output groups per block) do not enter any output's
// arithmetic (lane l, its k16 sequence and the 32-lane tree are fixed), so these
// are byte-identical to the o4t16 rows kernels. Block 32*TPL, OB = TPL*OPT.
#define T_LANES_ROWS_RV(R, OPT, TPL, MINB) \
    extern "C" __global__ void __launch_bounds__(32 * TPL, MINB) moe_expert_gate_up_shared_t_lanes_rows_r##R##_o##OPT##t##TPL( \
        T_LANES_GU_PARAMS, unsigned int rows) { \
        gate_up_t_lanes_rows_staged_body<OPT, TPL, R>(T_LANES_GU_ARGS, rows); } \
    extern "C" __global__ void __launch_bounds__(32 * TPL, MINB) moe_expert_silu_down_shared_t_lanes_rows_r##R##_o##OPT##t##TPL( \
        T_LANES_DN_PARAMS, unsigned int rows) { \
        silu_down_t_lanes_rows_body<OPT, TPL, R>(T_LANES_DN_ARGS, rows); }
T_LANES_ROWS_RV(5, 2, 16, 2)
T_LANES_ROWS_RV(5, 2, 32, 1)
T_LANES_ROWS_RV(5, 4, 8, 3)
T_LANES_ROWS_RV(5, 2, 8, 4)
T_LANES_ROWS_RV(3, 2, 16, 2)
T_LANES_ROWS_RV(3, 4, 8, 3)
T_LANES_ROWS_RV(2, 2, 16, 2)
T_LANES_ROWS_RV(2, 4, 8, 3)

// ---------------------------------------------------------------------------
// Eight-threads-per-group K-row down (verify-cost, ATLAS_MOE_T_LANES_ROWS_DOWN_T8=1).
//
// The *_t_lanes_rows down above spreads an output group's 32 K-lanes over 16
// warps, so every row ends in a shared-memory tree with two block barriers,
// and a 512-thread block at ~86 registers leaves one block per SM behind a
// thread-0 leader scan. Here a warp holds 4 groups of OPT adjacent outputs
// x 8 lane-threads: thread j = lane % 8 keeps the incumbent lanes l = 8m + j
// (m = 0..3), each partial the identical per-byte sum acc += al*wl + ah*wh
// over k16 = l, l + 32, ... (same LUT, scale, act expression and order). Each
// weight word is decoded once and applied to every member row. The incumbent
// tree adds lane l + 16, then l + 8, to lane l: both pairs share a residue
// mod 8, so they are register adds in the same order; offsets 4, 2, 1 are
// three width-8 shuffles. Outputs are byte-identical to
// silu_down_t_lanes_rows_body and to the single-row _t_lanes kernel per row.
// The leader rule and member order are the t_lanes_rows ones, resolved by
// ballots in every warp. Act rows are staged padded to 18 floats per k16
// block, so the eight lane-threads' float2 reads are bank-conflict free.
//
// Grid: (ceil(N / T8_OUTS), rows * top_k + 1, 1)  Block: (T8_THREADS, 1, 1)
// Dynamic shared memory: rows * (K / 16) * 18 * 4 bytes. K must equal 16 * K16C.
#define T8_THREADS 256
#define T8_OPT 2
#define T8_OUTS ((T8_THREADS / 8) * T8_OPT)
#define T8_ACT_STRIDE 18

// Steps of a lane-thread: s = 2m + i is lane 8m + j's i-th k16 (k16 = 8m + j + 32i).
template <int K16C>
__host__ __device__ constexpr bool t8_step_valid(int s) {
    return 8 * (s / 2) + T_LANES_K * (s % 2) < K16C;
}
template <int K16C>
__host__ __device__ constexpr int t8_next_step(int s) {
    for (int t = s + 1; t < 8; t++)
        if (t8_step_valid<K16C>(t)) return t;
    return -1;
}

// OPT adjacent outputs per thread (4 = u32 words, 2 = u16), THREADS per block,
// PREFETCH: issue the first step's weight words before the act rows are staged
// and every next step's words before the current step is accumulated. None of
// these enter any output's arithmetic (lane l, its k16 order, the per-byte
// expression and the tree are fixed), so every instantiation is byte-identical.
template <int RMAX, int K16C, int OPT, int THREADS, int PREFETCH>
__device__ __forceinline__ void silu_down_t_lanes_rows_t8_body(T_LANES_DN_PARAMS, unsigned int rows) {
    static_assert(K16C >= T_LANES_K && K16C <= 2 * T_LANES_K, "every lane owns one or two k16 steps");
    // Lane 8m + j owns k16 = 8m + j + 32i; with K16C % 8 == 0 that step exists
    // for every j <= 7 exactly when 8m + 32i < K16C, a compile-time condition.
    static_assert(K16C % 8 == 0, "per-thread step count must not depend on j");
    static_assert(THREADS % 32 == 0 && (OPT == 2 || OPT == 4), "t8 launch shape");
    constexpr int OUTS = (THREADS / 8) * OPT;
    constexpr int KC = 16 * K16C;
    constexpr int ROW_FLOATS = K16C * T8_ACT_STRIDE;
    (void)K;  // the launcher checks K == KC and rows <= RMAX
    const unsigned int total = rows * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y == total);
    const unsigned int lane = threadIdx.x % 32u;

    // Leader rule and member order of t_lanes_rows_members, warp-uniform from ballots.
    unsigned int cnt = 0;
    unsigned int flat[RMAX];
    #pragma unroll
    for (int r = 0; r < RMAX; r++) flat[r] = 0u;
    if (is_shared) {
        cnt = rows < (unsigned int)RMAX ? rows : (unsigned int)RMAX;
        #pragma unroll
        for (int r = 0; r < RMAX; r++) flat[r] = (unsigned int)r;
    } else {
        const unsigned int e = expert_indices[y];
        bool dup = false;
        for (unsigned int base = 0; base < total; base += 32u) {
            const unsigned int jj = base + lane;
            const unsigned int hits = __ballot_sync(0xFFFFFFFFu, jj < total && expert_indices[jj] == e);
            const unsigned int below = y >= base + 32u ? hits : (y > base ? hits & ((1u << (y - base)) - 1u) : 0u);
            dup = dup || below != 0u;
            unsigned int from_y = y <= base ? hits : (y < base + 32u ? hits & ~((1u << (y - base)) - 1u) : 0u);
            while (from_y != 0u && cnt < (unsigned int)RMAX) {
                const unsigned int f_idx = base + (unsigned int)(__ffs(from_y) - 1);
                #pragma unroll
                for (int r = 0; r < RMAX; r++) {
                    if (cnt == (unsigned int)r) flat[r] = f_idx;
                }
                cnt++;
                from_y &= from_y - 1u;
            }
        }
        if (dup) return;
    }

    const unsigned char* B_packed = is_shared
        ? sh_down_t_packed
        : (const unsigned char*)packed_t_ptrs[expert_indices[y]];
    const unsigned int n_block = blockIdx.x * OUTS;
    if (B_packed == 0) {
        __nv_bfloat16* z_base = is_shared ? sh_down_out : C;
        #pragma unroll
        for (int m = 0; m < RMAX; m++) {
            if ((unsigned int)m < cnt && threadIdx.x < OUTS && n_block + threadIdx.x < N)
                z_base[(unsigned long long)flat[m] * N + n_block + threadIdx.x] = __float2bfloat16(0.0f);
        }
        return;
    }

    __shared__ float s_lut[16];
    extern __shared__ __align__(16) float s_act_t8[];  // [cnt][K16C][T8_ACT_STRIDE]
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    #pragma unroll
    for (int m = 0; m < RMAX; m++) {
        if ((unsigned int)m < cnt) {
            const __nv_bfloat16* g_ptr = is_shared ? sh_gate_in + (unsigned long long)flat[m] * KC
                                                   : gate_out + (unsigned long long)flat[m] * KC;
            const __nv_bfloat16* u_ptr = is_shared ? sh_up_in + (unsigned long long)flat[m] * KC
                                                   : up_out + (unsigned long long)flat[m] * KC;
            float* dst = s_act_t8 + m * ROW_FLOATS;
            for (unsigned int i = threadIdx.x; i < (unsigned int)KC; i += blockDim.x) {
                float gf = __bfloat162float(g_ptr[i]);
                float uf = __bfloat162float(u_ptr[i]);
                dst[(i / 16u) * T8_ACT_STRIDE + i % 16u] = (gf / (1.0f + __expf(-gf))) * uf;
            }
        }
    }

    const unsigned char* B_scale = is_shared
        ? sh_down_t_scale
        : (const unsigned char*)scale_t_ptrs[expert_indices[y]];
    const float s2 = is_shared ? sh_down_s2 : scale2_vals[expert_indices[y]];
    __nv_bfloat16* out_base = is_shared ? sh_down_out : C;
    const unsigned int j = lane % 8u;
    const unsigned int n0 = n_block + (threadIdx.x / 8u) * OPT;
    const bool valid = n0 < N;
    // Scale word and eight packed words of one step, OPT outputs per word.
    auto load_step = [&](int s, unsigned int& sw, unsigned int (&pw)[8]) {
        const unsigned int k16 = 8u * (s / 2) + j + T_LANES_K * (s % 2);
        sw = valid ? t_lanes_load<OPT>(B_scale + (unsigned long long)k16 * N, n0, N) : 0u;
        #pragma unroll
        for (int b = 0; b < 8; b++)
            pw[b] = valid ? t_lanes_load<OPT>(B_packed + (unsigned long long)(k16 * 8 + b) * N, n0, N) : 0u;
    };
    unsigned int sw_cur = 0, pw_cur[8];
    if (PREFETCH) load_step(0, sw_cur, pw_cur);
    __syncthreads();
    if (!PREFETCH) load_step(0, sw_cur, pw_cur);

    // pacc[h]: incumbent lane 8h + j after the offset-16 add (h = 0: lanes j and
    // 16 + j; h = 1: lanes 8 + j and 24 + j), per member row and output.
    float pacc[2][RMAX][OPT];
    float acc[RMAX][OPT];
    #pragma unroll
    for (int r = 0; r < RMAX; r++)
        #pragma unroll
        for (int o = 0; o < OPT; o++) acc[r][o] = 0.0f;
    #pragma unroll
    for (int s = 0; s < 8; s++) {
        if (!t8_step_valid<K16C>(s)) continue;
        const int m = s / 2;
        const int ns = t8_next_step<K16C>(s);
        unsigned int sw_next = 0, pw_next[8];
        if (PREFETCH && ns >= 0) load_step(ns, sw_next, pw_next);
        const unsigned int k16 = 8u * m + j + T_LANES_K * (s % 2);
        float sc[OPT];
        #pragma unroll
        for (int o = 0; o < OPT; o++) {
            __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = (unsigned char)(sw_cur >> (8 * o));
            sc[o] = (float)fp8 * s2;
        }
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            float wl[OPT], wh[OPT];
            #pragma unroll
            for (int o = 0; o < OPT; o++) {
                const unsigned int byte = (pw_cur[b] >> (8 * o)) & 0xFF;
                wl[o] = s_lut[byte & 0xF] * sc[o];
                wh[o] = s_lut[byte >> 4] * sc[o];
            }
            #pragma unroll
            for (int r = 0; r < RMAX; r++) {
                if ((unsigned int)r < cnt) {
                    const float2 a = *(const float2*)(s_act_t8 + r * ROW_FLOATS + k16 * T8_ACT_STRIDE + b * 2);
                    #pragma unroll
                    for (int o = 0; o < OPT; o++) acc[r][o] += a.x * wl[o] + a.y * wh[o];
                }
            }
        }
        if (ns < 0 || ns / 2 != m) {
            // Lane 8m + j is final. Incumbent offset 16: lane 8m + j gains lane 8(m + 2) + j.
            #pragma unroll
            for (int r = 0; r < RMAX; r++)
                #pragma unroll
                for (int o = 0; o < OPT; o++) {
                    pacc[m & 1][r][o] = m >= 2 ? pacc[m & 1][r][o] + acc[r][o] : acc[r][o];
                    acc[r][o] = 0.0f;
                }
        }
        if (ns >= 0) {
            if (PREFETCH) {
                sw_cur = sw_next;
                #pragma unroll
                for (int b = 0; b < 8; b++) pw_cur[b] = pw_next[b];
            } else {
                load_step(ns, sw_cur, pw_cur);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < RMAX; r++) {
        if ((unsigned int)r >= cnt) break;  // block-uniform: the shuffles below stay converged
        __nv_bfloat16 res[OPT];
        #pragma unroll
        for (int o = 0; o < OPT; o++) {
            // Offset 8: lane j gains lane 8 + j; offsets 4, 2, 1 inside the group's eight threads.
            float v = pacc[0][r][o] + pacc[1][r][o];
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 4, 8);
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 2, 8);
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 1, 8);
            res[o] = __float2bfloat16(v);
        }
        if (valid && j == 0u) {
            __nv_bfloat16* out = out_base + (unsigned long long)flat[r] * N;
            #pragma unroll
            for (int o = 0; o < OPT; o++)
                if (n0 + o < N) out[n0 + o] = res[o];
        }
    }
}

// Flash-Next moe_intermediate_size 640 (K16C = 40), up to 5 member rows. tlanes_t8_tune_bench-1
// (10-14 17:19Z, 8 variants): two outputs per group, one-step prefetch, three blocks per SM is the
// fastest at 4 and 5 rows, 0.155 vs 0.198 ms (r4) and 0.198 vs 0.259 ms (r5) per layer, against
// 0.165 / 0.213 for the four-output, no-prefetch first cut; the r3 twin still wins at 3 rows
// (0.116 vs 0.121), so the launcher admits it for 4..5 verify rows only.
extern "C" __global__ void __launch_bounds__(T8_THREADS, 3) moe_expert_silu_down_shared_t_lanes_rows_t8_r5_k40(
    T_LANES_DN_PARAMS, unsigned int rows) {
    silu_down_t_lanes_rows_t8_body<5, 40, T8_OPT, T8_THREADS, 1>(T_LANES_DN_ARGS, rows);
}

// Eight-threads-per-group K-row gate/up on the unified (transposed) layout
// (ATLAS_MOE_T_LANES_ROWS_GATE_UP_T8=1).
// Same 8-thread-per-group register + width-8 shuffle tree and warp-ballot
// duplicate exit as silu_down_t_lanes_rows_t8_body, for K = 16 * K16C where
// K16C is a multiple of T_LANES_K = 32 (Flash-Next hidden_size 2560 -> K16C =
// 160, 5 k16 steps per incumbent lane). Each member row's raw BF16 pairs are
// staged in shared memory padded to T8_A32_STRIDE = 9 uint32 words per k16
// block (bank-conflict free across j = 0..7). Byte-identical per row to
// gate_up_t_lanes_rows_staged_body and single-row _t_lanes.
#define T8_A32_STRIDE 9

template <int RMAX, int K16C, int OPT, int THREADS, int PREFETCH>
__device__ __forceinline__ void gate_up_t_lanes_rows_t8_body(T_LANES_GU_PARAMS, unsigned int rows) {
    static_assert(K16C >= T_LANES_K && K16C % T_LANES_K == 0, "every lane owns K16C / 32 k16 steps");
    static_assert(THREADS % 32 == 0 && (OPT == 2 || OPT == 4), "t8 launch shape");
    constexpr int OUTS = (THREADS / 8) * OPT;
    constexpr int KC = 16 * K16C;
    constexpr int ROW_U32 = K16C * T8_A32_STRIDE;
    constexpr int STEPS_PER_LANE = K16C / T_LANES_K;
    constexpr int TOTAL_STEPS = 4 * STEPS_PER_LANE;
    (void)K;
    const unsigned int proj = blockIdx.z;
    const unsigned int total = rows * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y == total);
    const unsigned int lane = threadIdx.x % 32u;

    unsigned int cnt = 0;
    unsigned int flat[RMAX];
    #pragma unroll
    for (int r = 0; r < RMAX; r++) flat[r] = 0u;
    if (is_shared) {
        cnt = rows < (unsigned int)RMAX ? rows : (unsigned int)RMAX;
        #pragma unroll
        for (int r = 0; r < RMAX; r++) flat[r] = (unsigned int)r;
    } else {
        const unsigned int e = expert_indices[y];
        bool dup = false;
        for (unsigned int base = 0; base < total; base += 32u) {
            const unsigned int jj = base + lane;
            const unsigned int hits = __ballot_sync(0xFFFFFFFFu, jj < total && expert_indices[jj] == e);
            const unsigned int below = y >= base + 32u ? hits : (y > base ? hits & ((1u << (y - base)) - 1u) : 0u);
            dup = dup || below != 0u;
            unsigned int from_y = y <= base ? hits : (y < base + 32u ? hits & ~((1u << (y - base)) - 1u) : 0u);
            while (from_y != 0u && cnt < (unsigned int)RMAX) {
                const unsigned int f_idx = base + (unsigned int)(__ffs(from_y) - 1);
                #pragma unroll
                for (int r = 0; r < RMAX; r++) {
                    if (cnt == (unsigned int)r) flat[r] = f_idx;
                }
                cnt++;
                from_y &= from_y - 1u;
            }
        }
        if (dup) return;
    }

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* Cbase;
    if (is_shared) {
        B_packed = proj == 0 ? sh_gate_t_packed : sh_up_t_packed;
        B_scale = proj == 0 ? sh_gate_t_scale : sh_up_t_scale;
        s2 = proj == 0 ? sh_gate_s2 : sh_up_s2;
        Cbase = proj == 0 ? sh_gate_out : sh_up_out;
    } else {
        const unsigned int expert_id = expert_indices[y];
        B_packed = (const unsigned char*)(proj == 0 ? gate_packed_t_ptrs : up_packed_t_ptrs)[expert_id];
        B_scale = (const unsigned char*)(proj == 0 ? gate_scale_t_ptrs : up_scale_t_ptrs)[expert_id];
        s2 = (proj == 0 ? gate_scale2_vals : up_scale2_vals)[expert_id];
        Cbase = proj == 0 ? gate_out : up_out;
    }
    const unsigned int n_block = blockIdx.x * OUTS;
    if (B_packed == 0) {
        #pragma unroll
        for (int m = 0; m < RMAX; m++) {
            if ((unsigned int)m < cnt && threadIdx.x < OUTS && n_block + threadIdx.x < N)
                Cbase[(unsigned long long)flat[m] * N + n_block + threadIdx.x] = __float2bfloat16(0.0f);
        }
        return;
    }

    const unsigned int j = lane % 8u;
    const unsigned int n0 = n_block + (threadIdx.x / 8u) * OPT;
    const bool valid = n0 < N;
    auto load_step = [&](int s, unsigned int& sw, unsigned int (&pw)[8]) {
        const unsigned int k16 = 8u * (s / STEPS_PER_LANE) + j + T_LANES_K * (s % STEPS_PER_LANE);
        sw = valid ? t_lanes_load<OPT>(B_scale + (unsigned long long)k16 * N, n0, N) : 0u;
        #pragma unroll
        for (int b = 0; b < 8; b++)
            pw[b] = valid ? t_lanes_load<OPT>(B_packed + (unsigned long long)(k16 * 8 + b) * N, n0, N) : 0u;
    };
    unsigned int sw_cur = 0, pw_cur[8];
    if (PREFETCH) load_step(0, sw_cur, pw_cur);

    __shared__ float s_lut[16];
    extern __shared__ unsigned int s_a32_t8[];  // [cnt][K16C][T8_A32_STRIDE]
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    #pragma unroll
    for (int m = 0; m < RMAX; m++) {
        if ((unsigned int)m < cnt) {
            const unsigned int tok = is_shared ? flat[m] : (flat[m] / top_k);
            const unsigned int* A32 = (const unsigned int*)(A + (unsigned long long)tok * KC);
            unsigned int* dst = s_a32_t8 + m * ROW_U32;
            for (unsigned int i = threadIdx.x; i < (unsigned int)(KC / 2); i += blockDim.x) {
                dst[(i / 8u) * T8_A32_STRIDE + i % 8u] = A32[i];
            }
        }
    }
    __syncthreads();
    if (!PREFETCH) load_step(0, sw_cur, pw_cur);

    float pacc[2][RMAX][OPT];
    float acc[RMAX][OPT];
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        #pragma unroll
        for (int r = 0; r < RMAX; r++)
            #pragma unroll
            for (int o = 0; o < OPT; o++) acc[r][o] = 0.0f;
        #pragma unroll 1
        for (int i = 0; i < STEPS_PER_LANE; i++) {
            const int s = m * STEPS_PER_LANE + i;
            const int ns = s + 1 < TOTAL_STEPS ? s + 1 : -1;
            unsigned int sw_next = 0, pw_next[8];
            if (PREFETCH && ns >= 0) load_step(ns, sw_next, pw_next);
            const unsigned int k16 = 8u * m + j + T_LANES_K * i;
            float sc[OPT];
            #pragma unroll
            for (int o = 0; o < OPT; o++) {
                __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = (unsigned char)(sw_cur >> (8 * o));
                sc[o] = (float)fp8 * s2;
            }
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                float wl[OPT], wh[OPT];
                #pragma unroll
                for (int o = 0; o < OPT; o++) {
                    const unsigned int byte = (pw_cur[b] >> (8 * o)) & 0xFF;
                    wl[o] = s_lut[byte & 0xF] * sc[o];
                    wh[o] = s_lut[byte >> 4] * sc[o];
                }
                #pragma unroll
                for (int r = 0; r < RMAX; r++) {
                    if ((unsigned int)r < cnt) {
                        const unsigned int raw = s_a32_t8[r * ROW_U32 + k16 * T8_A32_STRIDE + b];
                        __nv_bfloat16 lo, hi;
                        *(unsigned short*)&lo = (unsigned short)(raw & 0xFFFF);
                        *(unsigned short*)&hi = (unsigned short)(raw >> 16);
                        const float al = __bfloat162float(lo);
                        const float ah = __bfloat162float(hi);
                        #pragma unroll
                        for (int o = 0; o < OPT; o++) acc[r][o] += al * wl[o] + ah * wh[o];
                    }
                }
            }
            if (ns >= 0) {
                if (PREFETCH) {
                    sw_cur = sw_next;
                    #pragma unroll
                    for (int b = 0; b < 8; b++) pw_cur[b] = pw_next[b];
                } else {
                    load_step(ns, sw_cur, pw_cur);
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < RMAX; r++)
            #pragma unroll
            for (int o = 0; o < OPT; o++)
                pacc[m & 1][r][o] = m >= 2 ? pacc[m & 1][r][o] + acc[r][o] : acc[r][o];
    }
    #pragma unroll
    for (int r = 0; r < RMAX; r++) {
        if ((unsigned int)r >= cnt) break;
        __nv_bfloat16 res[OPT];
        #pragma unroll
        for (int o = 0; o < OPT; o++) {
            float v = pacc[0][r][o] + pacc[1][r][o];
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 4, 8);
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 2, 8);
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 1, 8);
            res[o] = __float2bfloat16(v);
        }
        if (valid && j == 0u) {
            __nv_bfloat16* out = Cbase + (unsigned long long)flat[r] * N;
            #pragma unroll
            for (int o = 0; o < OPT; o++)
                if (n0 + o < N) out[n0 + o] = res[o];
        }
    }
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 3) moe_expert_gate_up_shared_t_lanes_rows_t8_r5_k160(
    T_LANES_GU_PARAMS, unsigned int rows) {
    gate_up_t_lanes_rows_t8_body<5, 160, T8_OPT, T8_THREADS, 1>(T_LANES_GU_ARGS, rows);
}

// ---------------------------------------------------------------------------
// Eight-threads-per-group single-token (M = 1) decode gate/up and silu_down
// on the unified (transposed) layout (`ATLAS_MOE_T_LANES_GATE_UP_T8=1`,
// `ATLAS_MOE_T_LANES_DOWN_T8=1`).
//
// Warp-coalesced 8-thread-per-group layout (g = tid % 32 across lanes,
// j = tid / 32 across the 8 warps) specialized for M = 1 (called 4 times
// per MTP step in Qwen4MtpHead + single-token decode):
//  - Static shared memory `s_a32_t8[K16C * T8_A32_STRIDE]` /
//    `s_act_t8[K16C * T8_ACT_STRIDE]` (no dynamic smem needed; bank-conflict
//    free across j = 0..7).
//  - Two-barrier shared-memory reduction reusing `s_a32_t8` / `s_act_t8`
//    aliased as `s_red[8 * OUTS]` (`part[32][65]` eliminated).
//  - 256 threads/block (`__launch_bounds__(256, 4)`), 1-step weight prefetch.
//  - 0 ULP bit-for-bit identical to `moe_expert_*_shared_t_lanes_o4t16u1`.
// ---------------------------------------------------------------------------
template <int K16C, int OPT, int THREADS, int PREFETCH, bool TS4E = false>
__device__ __forceinline__ void gate_up_t_lanes_t8_body(T_LANES_GU_PARAMS) {
    static_assert(K16C >= T_LANES_K && K16C % T_LANES_K == 0, "every lane owns K16C / 32 k16 steps");
    static_assert(THREADS % 32 == 0 && (OPT == 2 || OPT == 4), "t8 launch shape");
    constexpr int OUTS = (THREADS / 8) * OPT;
    constexpr int KC = 16 * K16C;
    constexpr int STEPS_PER_LANE = K16C / T_LANES_K;
    constexpr int TOTAL_STEPS = 4 * STEPS_PER_LANE;
    (void)K;

    const unsigned int expert_slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;
    if (is_shared) {
        B_packed = proj == 0 ? sh_gate_t_packed : sh_up_t_packed;
        B_scale = proj == 0 ? sh_gate_t_scale : sh_up_t_scale;
        s2 = proj == 0 ? sh_gate_s2 : sh_up_s2;
        C = proj == 0 ? sh_gate_out : sh_up_out;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)(proj == 0 ? gate_packed_t_ptrs : up_packed_t_ptrs)[expert_id];
        B_scale = (const unsigned char*)(proj == 0 ? gate_scale_t_ptrs : up_scale_t_ptrs)[expert_id];
        s2 = (proj == 0 ? gate_scale2_vals : up_scale2_vals)[expert_id];
        C = (proj == 0 ? gate_out : up_out) + (unsigned long long)expert_slot * N;
    }
    const unsigned int n_block = blockIdx.x * OUTS;
    if (B_packed == 0) {
        if (threadIdx.x < OUTS && n_block + threadIdx.x < N)
            C[n_block + threadIdx.x] = __float2bfloat16(0.0f);
        return;
    }

    const unsigned int g = threadIdx.x % 32u;
    const unsigned int j = threadIdx.x / 32u;
    const unsigned int n0 = n_block + g * OPT;
    const bool valid = n0 < N;
    const unsigned int base_word = (TS4E && !is_shared && valid)
        ? t_lanes_load<OPT>(B_scale + static_cast<unsigned long long>(KC >> 5) * N, n0, N)
        : 0u;
    auto load_step = [&](int s, unsigned int& sw, unsigned int (&pw)[8]) {
        const unsigned int k16 = 8u * (s / STEPS_PER_LANE) + j + T_LANES_K * (s % STEPS_PER_LANE);
        if constexpr (TS4E) {
            sw = valid
                ? (is_shared
                    ? t_lanes_load<OPT>(B_scale + (unsigned long long)k16 * N, n0, N)
                    : ts4e_load_scale_word<OPT, KC>(B_scale, base_word, k16, n0, N))
                : 0u;
        } else {
            sw = valid ? t_lanes_load<OPT>(B_scale + (unsigned long long)k16 * N, n0, N) : 0u;
        }
        #pragma unroll
        for (int b = 0; b < 8; b++)
            pw[b] = valid ? t_lanes_load<OPT>(B_packed + (unsigned long long)(k16 * 8 + b) * N, n0, N) : 0u;
    };
    unsigned int sw_cur = 0, pw_cur[8];
    if (PREFETCH) load_step(0, sw_cur, pw_cur);

    __shared__ float s_lut[16];
    __shared__ unsigned int s_a32_t8[K16C * T8_A32_STRIDE];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    const unsigned int* A32 = (const unsigned int*)A;
    for (unsigned int i = threadIdx.x; i < (unsigned int)(KC / 2); i += blockDim.x) {
        s_a32_t8[(i / 8u) * T8_A32_STRIDE + i % 8u] = A32[i];
    }
    __syncthreads();
    if (!PREFETCH) load_step(0, sw_cur, pw_cur);

    float pacc[2][OPT];
    float acc[OPT];
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        #pragma unroll
        for (int o = 0; o < OPT; o++) acc[o] = 0.0f;
        #pragma unroll
        for (int i = 0; i < STEPS_PER_LANE; i++) {
            const int s = m * STEPS_PER_LANE + i;
            const int ns = s + 1 < TOTAL_STEPS ? s + 1 : -1;
            unsigned int sw_next = 0, pw_next[8];
            if (PREFETCH && ns >= 0) load_step(ns, sw_next, pw_next);
            const unsigned int k16 = 8u * m + j + T_LANES_K * i;
            float sc[OPT];
            #pragma unroll
            for (int o = 0; o < OPT; o++) {
                __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = (unsigned char)(sw_cur >> (8 * o));
                sc[o] = (float)fp8 * s2;
            }
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                const unsigned int raw = s_a32_t8[k16 * T8_A32_STRIDE + b];
                __nv_bfloat16 lo, hi;
                *(unsigned short*)&lo = (unsigned short)(raw & 0xFFFF);
                *(unsigned short*)&hi = (unsigned short)(raw >> 16);
                const float al = __bfloat162float(lo);
                const float ah = __bfloat162float(hi);
                #pragma unroll
                for (int o = 0; o < OPT; o++) {
                    const unsigned int byte = (pw_cur[b] >> (8 * o)) & 0xFF;
                    const float wl = s_lut[byte & 0xF] * sc[o];
                    const float wh = s_lut[byte >> 4] * sc[o];
                    acc[o] += al * wl + ah * wh;
                }
            }
            if (ns >= 0) {
                if (PREFETCH) {
                    sw_cur = sw_next;
                    #pragma unroll
                    for (int b = 0; b < 8; b++) pw_cur[b] = pw_next[b];
                } else {
                    load_step(ns, sw_cur, pw_cur);
                }
            }
        }
        #pragma unroll
        for (int o = 0; o < OPT; o++)
            pacc[m & 1][o] = m >= 2 ? pacc[m & 1][o] + acc[o] : acc[o];
    }
    float* s_red = reinterpret_cast<float*>(s_a32_t8);
    __syncthreads();
    #pragma unroll
    for (int o = 0; o < OPT; o++) {
        s_red[j * OUTS + g * OPT + o] = pacc[0][o] + pacc[1][o];
    }
    __syncthreads();
    if (threadIdx.x < (unsigned int)OUTS && n_block + threadIdx.x < N) {
        const unsigned int c = threadIdx.x;
        const float v04 = s_red[0 * OUTS + c] + s_red[4 * OUTS + c];
        const float v26 = s_red[2 * OUTS + c] + s_red[6 * OUTS + c];
        const float v15 = s_red[1 * OUTS + c] + s_red[5 * OUTS + c];
        const float v37 = s_red[3 * OUTS + c] + s_red[7 * OUTS + c];
        C[n_block + c] = __float2bfloat16((v04 + v26) + (v15 + v37));
    }
}

template <int K16C, int OPT, int THREADS, int PREFETCH, bool TS4E = false>
__device__ __forceinline__ void silu_down_t_lanes_t8_body(T_LANES_DN_PARAMS) {
    static_assert(K16C >= T_LANES_K && K16C <= 2 * T_LANES_K, "every lane owns one or two k16 steps");
    static_assert(K16C % 8 == 0, "per-thread step count must not depend on j");
    static_assert(THREADS % 32 == 0 && (OPT == 2 || OPT == 4), "t8 launch shape");
    constexpr int OUTS = (THREADS / 8) * OPT;
    constexpr int KC = 16 * K16C;
    (void)K;

    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;
    __nv_bfloat16* out;
    if (is_shared) {
        B_packed = sh_down_t_packed; B_scale = sh_down_t_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in; u_ptr = sh_up_in; out = sh_down_out;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_t_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_t_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * KC;
        u_ptr = up_out + (unsigned long long)expert_slot * KC;
        out = C + (unsigned long long)expert_slot * N;
    }
    const unsigned int n_block = blockIdx.x * OUTS;
    if (B_packed == 0) {
        if (threadIdx.x < OUTS && n_block + threadIdx.x < N)
            out[n_block + threadIdx.x] = __float2bfloat16(0.0f);
        return;
    }

    const unsigned int g = threadIdx.x % 32u;
    const unsigned int j = threadIdx.x / 32u;
    const unsigned int n0 = n_block + g * OPT;
    const bool valid = n0 < N;
    const unsigned int base_word = (TS4E && !is_shared && valid)
        ? t_lanes_load<OPT>(B_scale + static_cast<unsigned long long>(KC >> 5) * N, n0, N)
        : 0u;
    auto load_step = [&](int s, unsigned int& sw, unsigned int (&pw)[8]) {
        const unsigned int k16 = 8u * (s / 2) + j + T_LANES_K * (s % 2);
        if constexpr (TS4E) {
            sw = valid
                ? (is_shared
                    ? t_lanes_load<OPT>(B_scale + (unsigned long long)k16 * N, n0, N)
                    : ts4e_load_scale_word<OPT, KC>(B_scale, base_word, k16, n0, N))
                : 0u;
        } else {
            sw = valid ? t_lanes_load<OPT>(B_scale + (unsigned long long)k16 * N, n0, N) : 0u;
        }
        #pragma unroll
        for (int b = 0; b < 8; b++)
            pw[b] = valid ? t_lanes_load<OPT>(B_packed + (unsigned long long)(k16 * 8 + b) * N, n0, N) : 0u;
    };
    unsigned int sw_cur = 0, pw_cur[8];
    if (PREFETCH) load_step(0, sw_cur, pw_cur);

    __shared__ float s_lut[16];
    __shared__ __align__(16) float s_act_t8[K16C * T8_ACT_STRIDE];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    for (unsigned int i = threadIdx.x; i < (unsigned int)KC; i += blockDim.x) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act_t8[(i / 16u) * T8_ACT_STRIDE + i % 16u] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();
    if (!PREFETCH) load_step(0, sw_cur, pw_cur);

    float pacc[2][OPT];
    float acc[OPT];
    #pragma unroll
    for (int o = 0; o < OPT; o++) acc[o] = 0.0f;
    #pragma unroll
    for (int s = 0; s < 8; s++) {
        if (!t8_step_valid<K16C>(s)) continue;
        const int m = s / 2;
        const int ns = t8_next_step<K16C>(s);
        unsigned int sw_next = 0, pw_next[8];
        if (PREFETCH && ns >= 0) load_step(ns, sw_next, pw_next);
        const unsigned int k16 = 8u * m + j + T_LANES_K * (s % 2);
        float sc[OPT];
        #pragma unroll
        for (int o = 0; o < OPT; o++) {
            __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = (unsigned char)(sw_cur >> (8 * o));
            sc[o] = (float)fp8 * s2;
        }
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            const float2 a = *(const float2*)(s_act_t8 + k16 * T8_ACT_STRIDE + b * 2);
            #pragma unroll
            for (int o = 0; o < OPT; o++) {
                const unsigned int byte = (pw_cur[b] >> (8 * o)) & 0xFF;
                const float wl = s_lut[byte & 0xF] * sc[o];
                const float wh = s_lut[byte >> 4] * sc[o];
                acc[o] += a.x * wl + a.y * wh;
            }
        }
        if (ns < 0 || ns / 2 != m) {
            #pragma unroll
            for (int o = 0; o < OPT; o++) {
                pacc[m & 1][o] = m >= 2 ? pacc[m & 1][o] + acc[o] : acc[o];
                acc[o] = 0.0f;
            }
        }
        if (ns >= 0) {
            if (PREFETCH) {
                sw_cur = sw_next;
                #pragma unroll
                for (int b = 0; b < 8; b++) pw_cur[b] = pw_next[b];
            } else {
                load_step(ns, sw_cur, pw_cur);
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int o = 0; o < OPT; o++) {
        s_act_t8[j * OUTS + g * OPT + o] = pacc[0][o] + pacc[1][o];
    }
    __syncthreads();
    if (threadIdx.x < (unsigned int)OUTS && n_block + threadIdx.x < N) {
        const unsigned int c = threadIdx.x;
        const float v04 = s_act_t8[0 * OUTS + c] + s_act_t8[4 * OUTS + c];
        const float v26 = s_act_t8[2 * OUTS + c] + s_act_t8[6 * OUTS + c];
        const float v15 = s_act_t8[1 * OUTS + c] + s_act_t8[5 * OUTS + c];
        const float v37 = s_act_t8[3 * OUTS + c] + s_act_t8[7 * OUTS + c];
        out[n_block + c] = __float2bfloat16((v04 + v26) + (v15 + v37));
    }
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 4) moe_expert_gate_up_shared_t_lanes_t8_k160(
    T_LANES_GU_PARAMS) {
    gate_up_t_lanes_t8_body<160, T8_OPT, T8_THREADS, 1>(T_LANES_GU_ARGS);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 4) moe_expert_silu_down_shared_t_lanes_t8_k40(
    T_LANES_DN_PARAMS) {
    silu_down_t_lanes_t8_body<40, T8_OPT, T8_THREADS, 1>(T_LANES_DN_ARGS);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 4) moe_expert_gate_up_shared_t_lanes_t8_k160_ts4e(
    T_LANES_GU_PARAMS) {
    gate_up_t_lanes_t8_body<160, T8_OPT, T8_THREADS, 1, true>(T_LANES_GU_ARGS);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 4) moe_expert_silu_down_shared_t_lanes_t8_k40_ts4e(
    T_LANES_DN_PARAMS) {
    silu_down_t_lanes_t8_body<40, T8_OPT, T8_THREADS, 1, true>(T_LANES_DN_ARGS);
}

// ===========================================================================
// Decode-75 Wave 2 Extensions (Smoking Gun D1 + D4 + Worklist Gate/Up):
//  1. Complete `t8` 8-thread-per-group warp-shuffle coverage across RMAX={3,5,8}
//     with register-pinned `flat[RMAX]` (0 bytes stack frame, 0 spill stores)
//     so K=2..3 and K=6..8 never fall off the cliff into 512-thread shared-mem
//     tree kernels (`r2/r3/r9`).
//  2. Warp-parallel `moe_build_unique_worklist_r8`: 32-thread ballot compactor
//     that builds `unique_first_slot`, `unique_cnt`, and `unique_flat` in up to 13
//     warp-ballot passes over `s_idx[400]` so neither `gate_up` nor `down_proj`
//     scans `expert_indices[0..rows*top_k]` across 2,400+ CTAs.
//  3. `moe_expert_gate_up_worklist_t8_r{3,5,8}_k160`: Worklist-driven `t8`
//     gate+up kernels with O(1) duplicate/inactive CTA exit, static unpadded
//     shared-memory staging (`s_a32_t8[RMAX * ROW_U32]`), and 0 stack frames.
//  4. `moe_silu_mul_rows_f32` + `moe_expert_down_act_shared_t_lanes_rows_t8_*`:
//     Materializes `SiLU(gate) * up` once per active slot into FP32 so the 40
//     N-blocks of `down_proj` read pre-activated FP32 once instead of reading
//     both `gate_out` and `up_out` 40x and evaluating `expf` 40x per expert.
// ===========================================================================

extern "C" __global__ void __launch_bounds__(T8_THREADS, 4)
moe_expert_gate_up_shared_t_lanes_rows_t8_r3_k160(
    T_LANES_GU_PARAMS, unsigned int rows) {
    gate_up_t_lanes_rows_t8_body<3, 160, T8_OPT, T8_THREADS, 1>(T_LANES_GU_ARGS, rows);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 4)
moe_expert_silu_down_shared_t_lanes_rows_t8_r3_k40(
    T_LANES_DN_PARAMS, unsigned int rows) {
    silu_down_t_lanes_rows_t8_body<3, 40, T8_OPT, T8_THREADS, 1>(T_LANES_DN_ARGS, rows);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 2)
moe_expert_gate_up_shared_t_lanes_rows_t8_r8_k160(
    T_LANES_GU_PARAMS, unsigned int rows) {
    gate_up_t_lanes_rows_t8_body<8, 160, T8_OPT, T8_THREADS, 1>(T_LANES_GU_ARGS, rows);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 2)
moe_expert_silu_down_shared_t_lanes_rows_t8_r8_k40(
    T_LANES_DN_PARAMS, unsigned int rows) {
    silu_down_t_lanes_rows_t8_body<8, 40, T8_OPT, T8_THREADS, 1>(T_LANES_DN_ARGS, rows);
}

// Warp-parallel unique-expert worklist compactor for rows <= 40, top_k <= 10 (max_slots <= 400),
// chunking each distinct expert into groups of at most RMAX = 10 token slots:
//   unique_first_slot[u]    = first flat slot `s` for worklist group `u` (or 0xFFFFFFFFu)
//   unique_cnt[u]           = number of token rows (1..=10) in worklist group `u`
//   unique_flat[u * 10 + r] = flat slot index `s = tok * top_k + k` for member `r`
extern "C" __global__ void __launch_bounds__(32, 1)
moe_build_unique_worklist_r8(
    const unsigned int* __restrict__ expert_indices, // [rows * top_k]
    unsigned int* __restrict__ unique_first_slot,    // [max_slots]
    unsigned int* __restrict__ unique_cnt,           // [max_slots]
    unsigned int* __restrict__ unique_flat,          // [max_slots * 10]
    unsigned int rows,
    unsigned int top_k,
    unsigned int max_slots,
    unsigned int rmax)
{
    const unsigned int lane = threadIdx.x;
    const unsigned int total = rows * top_k;
    __shared__ unsigned int s_idx[400];
    for (unsigned int i = lane; i < total && i < 400u; i += 32u) {
        s_idx[i] = expert_indices[i];
    }
    __syncwarp();

    unsigned int u_base = 0u;
    const unsigned int limit = total < 400u ? total : 400u;
    for (unsigned int base = 0u; base < limit; base += 32u) {
        const unsigned int s = base + lane;
        bool is_leader = false;
        unsigned int eid = 0u;
        if (s < limit) {
            eid = s_idx[s];
            unsigned int prior_hits = 0u;
            for (unsigned int p = 0u; p < s; ++p) {
                if (s_idx[p] == eid) {
                    ++prior_hits;
                }
            }
            is_leader = (prior_hits % rmax) == 0u;
        }
        const unsigned int leader_mask = __ballot_sync(0xFFFFFFFFu, is_leader);
        const unsigned int u_out = u_base + (unsigned int)__popc(leader_mask & ((1u << lane) - 1u));
        if (is_leader && u_out < max_slots) {
            unsigned int cnt = 0u;
            for (unsigned int p = s; p < limit && cnt < rmax; ++p) {
                if (s_idx[p] == eid) {
                    unique_flat[u_out * 10u + cnt] = p;
                    ++cnt;
                }
            }
            unique_first_slot[u_out] = s;
            unique_cnt[u_out] = cnt;
        }
        u_base += (unsigned int)__popc(leader_mask);
    }

    for (unsigned int u = u_base + lane; u < max_slots; u += 32u) {
        unique_first_slot[u] = 0xFFFFFFFFu;
        unique_cnt[u] = 0u;
    }
}

// O(total) counting-sort unique-expert worklist builder for rows <= 128,
// total = rows * top_k <= 1280, and num_experts <= 512.
// Grid: [1, 1, 1], Block: [512, 1, 1]. Produces the exact same
// (unique_first_slot, unique_cnt, unique_flat[u * 10 + m]) worklist ABI
// consumed by `gate_up_worklist_t8_body`, `down_act_t_lanes_rows_t8_body`,
// and `gate_up_worklist_t_lanes_rows_staged_body`.
extern "C" __global__ void __launch_bounds__(512, 1)
moe_build_unique_worklist_e512(
    const unsigned int* __restrict__ expert_indices, // [rows * top_k]
    unsigned int* __restrict__ unique_first_slot,    // [max_slots]
    unsigned int* __restrict__ unique_cnt,           // [max_slots]
    unsigned int* __restrict__ unique_flat,          // [max_slots * 10]
    unsigned int rows,
    unsigned int top_k,
    unsigned int max_slots,
    unsigned int rmax,
    unsigned int num_experts)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int total = rows * top_k;
    const unsigned int limit = total < 1280u ? total : 1280u;
    const unsigned int e_lim = num_experts < 512u ? num_experts : 512u;
    const unsigned int eff_rmax = (rmax == 0u) ? 1u : (rmax > 10u ? 10u : rmax);

    __shared__ unsigned int s_idx[1280];
    __shared__ unsigned int s_count[512];
    __shared__ unsigned int s_group_base[512];
    __shared__ unsigned int s_slot_cursor[512];
    __shared__ unsigned int s_flat_sorted[1280];
    __shared__ unsigned int s_warp_slot_sum[16];
    __shared__ unsigned int s_warp_group_sum[16];
    __shared__ unsigned int s_total_groups;

    s_count[tid] = 0u;
    for (unsigned int i = tid; i < limit; i += 512u) {
        s_idx[i] = expert_indices[i];
    }
    __syncthreads();

    for (unsigned int i = tid; i < limit; i += 512u) {
        const unsigned int eid = s_idx[i];
        if (eid < e_lim) {
            atomicAdd(&s_count[eid], 1u);
        }
    }
    __syncthreads();

    const unsigned int cnt_e = (tid < e_lim) ? s_count[tid] : 0u;
    const unsigned int groups_e = (cnt_e > 0u) ? ((cnt_e + eff_rmax - 1u) / eff_rmax) : 0u;

    unsigned int sc_slot = cnt_e;
    unsigned int sc_grp = groups_e;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const unsigned int up_slot = __shfl_up_sync(0xFFFFFFFFu, sc_slot, offset);
        const unsigned int up_grp = __shfl_up_sync(0xFFFFFFFFu, sc_grp, offset);
        if (lane >= (unsigned int)offset) {
            sc_slot += up_slot;
            sc_grp += up_grp;
        }
    }
    if (lane == 31u) {
        s_warp_slot_sum[warp] = sc_slot;
        s_warp_group_sum[warp] = sc_grp;
    }
    __syncthreads();

    if (warp == 0u) {
        const unsigned int orig_ws = (lane < 16u) ? s_warp_slot_sum[lane] : 0u;
        const unsigned int orig_wg = (lane < 16u) ? s_warp_group_sum[lane] : 0u;
        unsigned int ws = orig_ws;
        unsigned int wg = orig_wg;
        #pragma unroll
        for (int offset = 1; offset < 16; offset <<= 1) {
            const unsigned int up_ws = __shfl_up_sync(0xFFFFFFFFu, ws, offset);
            const unsigned int up_wg = __shfl_up_sync(0xFFFFFFFFu, wg, offset);
            if (lane >= (unsigned int)offset) {
                ws += up_ws;
                wg += up_wg;
            }
        }
        if (lane < 16u) {
            s_warp_slot_sum[lane] = ws - orig_ws;
            s_warp_group_sum[lane] = wg - orig_wg;
            if (lane == 15u) {
                s_total_groups = wg;
            }
        }
    }
    __syncthreads();

    const unsigned int slot_base_e = s_warp_slot_sum[warp] + (sc_slot - cnt_e);
    const unsigned int grp_base_e = s_warp_group_sum[warp] + (sc_grp - groups_e);
    s_slot_cursor[tid] = slot_base_e;
    s_group_base[tid] = grp_base_e;
    __syncthreads();

    for (unsigned int i = tid; i < limit; i += 512u) {
        const unsigned int eid = s_idx[i];
        if (eid < e_lim) {
            const unsigned int pos = atomicAdd(&s_slot_cursor[eid], 1u);
            s_flat_sorted[pos] = i;
        }
    }
    __syncthreads();

    if (tid < e_lim && groups_e > 0u) {
        const unsigned int g_base = s_group_base[tid];
        for (unsigned int g = 0u; g < groups_e; ++g) {
            const unsigned int u_out = g_base + g;
            if (u_out < max_slots) {
                const unsigned int start = slot_base_e + g * eff_rmax;
                const unsigned int rem = cnt_e - g * eff_rmax;
                const unsigned int cnt = rem > eff_rmax ? eff_rmax : rem;
                unique_first_slot[u_out] = s_flat_sorted[start];
                unique_cnt[u_out] = cnt;
                for (unsigned int m = 0u; m < cnt; ++m) {
                    unique_flat[u_out * 10u + m] = s_flat_sorted[start + m];
                }
            }
        }
    }

    const unsigned int u_total = s_total_groups;
    for (unsigned int u = u_total + tid; u < max_slots; u += 512u) {
        unique_first_slot[u] = 0xFFFFFFFFu;
        unique_cnt[u] = 0u;
    }
}

// Worklist-driven `gate_up` body for RMAX = {3, 5, 8} (K16C = 160).
// Consumes `unique_first_slot`, `unique_cnt`, `unique_flat` produced by
// `moe_build_unique_worklist_r8` so duplicate/inactive CTAs exit in O(1)
// without scanning `expert_indices[0..rows*top_k]`. Uses static shared memory
// `s_a32_t8[RMAX * ROW_U32]` and shared `s_flat[RMAX]` (0 stack frame).
template <int RMAX, int K16C, int OPT, int THREADS, int PREFETCH, bool TS4E = false>
__device__ __forceinline__ void gate_up_worklist_t8_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned int* __restrict__ unique_first_slot,
    const unsigned int* __restrict__ unique_cnt,
    const unsigned int* __restrict__ unique_flat,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N,
    unsigned int K,
    unsigned int top_k,
    unsigned int rows)
{
    static_assert(K16C >= T_LANES_K && K16C % T_LANES_K == 0, "every lane owns K16C / 32 k16 steps");
    static_assert(THREADS % 32 == 0 && (OPT == 2 || OPT == 4), "t8 launch shape");
    constexpr int OUTS = (THREADS / 8) * OPT;
    constexpr int KC = 16 * K16C;
    constexpr int ROW_U32 = KC / 2;
    constexpr int STEPS_PER_LANE = K16C / T_LANES_K;
    constexpr int TOTAL_STEPS = 4 * STEPS_PER_LANE;
    (void)K;

    if (rows == 0u || rows > 128u) return;
    const unsigned int proj = blockIdx.z;
    const unsigned int total = rows * top_k;
    const unsigned int shared_ctas = (rows + (unsigned int)RMAX - 1u) / (unsigned int)RMAX;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y < shared_ctas);
    const unsigned int wl_y = y - shared_ctas;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* Cbase;
    unsigned int cnt;
    __shared__ unsigned int s_flat[RMAX];

    if (is_shared) {
        const unsigned int sh_base = y * (unsigned int)RMAX;
        if (sh_base >= rows) return;
        B_packed = proj == 0 ? sh_gate_t_packed : sh_up_t_packed;
        B_scale = proj == 0 ? sh_gate_t_scale : sh_up_t_scale;
        s2 = proj == 0 ? sh_gate_s2 : sh_up_s2;
        Cbase = proj == 0 ? sh_gate_out : sh_up_out;
        const unsigned int rem = rows - sh_base;
        cnt = rem > (unsigned int)RMAX ? (unsigned int)RMAX : rem;
        if (threadIdx.x < (unsigned int)RMAX) s_flat[threadIdx.x] = sh_base + threadIdx.x;
    } else {
        if (wl_y >= total) return;
        const unsigned int first_slot = unique_first_slot[wl_y];
        if (first_slot == 0xFFFFFFFFu) return;
        const unsigned int expert_id = expert_indices[first_slot];
        B_packed = (const unsigned char*)(proj == 0 ? gate_packed_t_ptrs : up_packed_t_ptrs)[expert_id];
        B_scale = (const unsigned char*)(proj == 0 ? gate_scale_t_ptrs : up_scale_t_ptrs)[expert_id];
        s2 = (proj == 0 ? gate_scale2_vals : up_scale2_vals)[expert_id];
        Cbase = proj == 0 ? gate_out : up_out;
        cnt = unique_cnt[wl_y];
        if (threadIdx.x < (unsigned int)RMAX) {
            s_flat[threadIdx.x] = (threadIdx.x < cnt) ? unique_flat[wl_y * 10u + threadIdx.x] : 0u;
        }
    }

    const unsigned int n_block = blockIdx.x * OUTS;
    if (B_packed == 0) {
        __syncthreads();
        if (threadIdx.x < OUTS && n_block + threadIdx.x < N) {
            #pragma unroll
            for (int r = 0; r < RMAX; r++) {
                if ((unsigned int)r < cnt)
                    Cbase[(unsigned long long)s_flat[r] * N + n_block + threadIdx.x] = __float2bfloat16(0.0f);
            }
        }
        return;
    }

    const unsigned int g = threadIdx.x % 32u;
    const unsigned int j = threadIdx.x / 32u;
    const unsigned int n0 = n_block + g * OPT;
    const bool valid = n0 < N;
    const unsigned int base_word = (TS4E && !is_shared && valid)
        ? t_lanes_load<OPT>(B_scale + static_cast<unsigned long long>(KC >> 5) * N, n0, N)
        : 0u;
    auto load_step = [&](int s, unsigned int& sw, unsigned int (&pw)[8]) {
        const unsigned int k16 = 8u * (s / STEPS_PER_LANE) + j + T_LANES_K * (s % STEPS_PER_LANE);
        if constexpr (TS4E) {
            sw = valid
                ? (is_shared
                    ? t_lanes_load<OPT>(B_scale + (unsigned long long)k16 * N, n0, N)
                    : ts4e_load_scale_word<OPT, KC>(B_scale, base_word, k16, n0, N))
                : 0u;
        } else {
            sw = valid ? t_lanes_load<OPT>(B_scale + (unsigned long long)k16 * N, n0, N) : 0u;
        }
        #pragma unroll
        for (int b = 0; b < 8; b++)
            pw[b] = valid ? t_lanes_load<OPT>(B_packed + (unsigned long long)(k16 * 8 + b) * N, n0, N) : 0u;
    };
    unsigned int sw_cur = 0, pw_cur[8];
    if (PREFETCH) load_step(0, sw_cur, pw_cur);

    __shared__ float s_lut[16];
    __shared__ __align__(16) unsigned int s_a32_t8[RMAX * ROW_U32];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    __syncthreads();
    for (unsigned int r = 0; r < cnt; r++) {
        const unsigned int tok = is_shared ? s_flat[r] : (s_flat[r] / top_k);
        const uint4* A128 = reinterpret_cast<const uint4*>(A + (unsigned long long)tok * KC);
        uint4* dst128 = reinterpret_cast<uint4*>(s_a32_t8 + r * ROW_U32);
        for (unsigned int i = threadIdx.x; i < (unsigned int)(KC / 8); i += blockDim.x) {
            dst128[i] = __ldg(&A128[i]);
        }
    }
    __syncthreads();
    if (!PREFETCH) load_step(0, sw_cur, pw_cur);

    if (cnt == 1u) {
        float pacc0[2][OPT];
        float acc0[OPT];
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            #pragma unroll
            for (int o = 0; o < OPT; o++) acc0[o] = 0.0f;
            #pragma unroll
            for (int i = 0; i < STEPS_PER_LANE; i++) {
                const int s = m * STEPS_PER_LANE + i;
                const int ns = s + 1 < TOTAL_STEPS ? s + 1 : -1;
                unsigned int sw_next = 0, pw_next[8];
                if (PREFETCH && ns >= 0) load_step(ns, sw_next, pw_next);
                const unsigned int k16 = 8u * m + j + T_LANES_K * i;
                float sc[OPT];
                #pragma unroll
                for (int o = 0; o < OPT; o++) {
                    __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = (unsigned char)(sw_cur >> (8 * o));
                    sc[o] = (float)fp8 * s2;
                }
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    const unsigned int raw = s_a32_t8[k16 * 8u + b];
                    __nv_bfloat16 lo, hi;
                    *(unsigned short*)&lo = (unsigned short)(raw & 0xFFFF);
                    *(unsigned short*)&hi = (unsigned short)(raw >> 16);
                    const float al = __bfloat162float(lo);
                    const float ah = __bfloat162float(hi);
                    #pragma unroll
                    for (int o = 0; o < OPT; o++) {
                        const unsigned int byte = (pw_cur[b] >> (8 * o)) & 0xFF;
                        const float wl = s_lut[byte & 0xF] * sc[o];
                        const float wh = s_lut[byte >> 4] * sc[o];
                        acc0[o] += al * wl + ah * wh;
                    }
                }
                if (ns >= 0) {
                    if (PREFETCH) {
                        sw_cur = sw_next;
                        #pragma unroll
                        for (int b = 0; b < 8; b++) pw_cur[b] = pw_next[b];
                    } else {
                        load_step(ns, sw_cur, pw_cur);
                    }
                }
            }
            #pragma unroll
            for (int o = 0; o < OPT; o++)
                pacc0[m & 1][o] = m >= 2 ? pacc0[m & 1][o] + acc0[o] : acc0[o];
        }
        float* s_red = reinterpret_cast<float*>(s_a32_t8);
        __syncthreads();
        #pragma unroll
        for (int o = 0; o < OPT; o++) {
            s_red[j * OUTS + g * OPT + o] = pacc0[0][o] + pacc0[1][o];
        }
        __syncthreads();
        if (threadIdx.x < (unsigned int)OUTS && n_block + threadIdx.x < N) {
            const unsigned int c = threadIdx.x;
            const float v04 = s_red[0 * OUTS + c] + s_red[4 * OUTS + c];
            const float v26 = s_red[2 * OUTS + c] + s_red[6 * OUTS + c];
            const float v15 = s_red[1 * OUTS + c] + s_red[5 * OUTS + c];
            const float v37 = s_red[3 * OUTS + c] + s_red[7 * OUTS + c];
            Cbase[(unsigned long long)s_flat[0] * N + n_block + c] =
                __float2bfloat16((v04 + v26) + (v15 + v37));
        }
        return;
    }

    float pacc[2][RMAX][OPT];
    #pragma unroll
    for (int h = 0; h < 2; h++)
        #pragma unroll
        for (int r = 0; r < RMAX; r++)
            #pragma unroll
            for (int o = 0; o < OPT; o++) pacc[h][r][o] = 0.0f;

    auto run_rows = [&](auto active_tag) {
        constexpr int ACTIVE_RMAX = decltype(active_tag)::value;
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            float acc[ACTIVE_RMAX][OPT];
            #pragma unroll
            for (int r = 0; r < ACTIVE_RMAX; r++)
                #pragma unroll
                for (int o = 0; o < OPT; o++) acc[r][o] = 0.0f;
            #pragma unroll 1
            for (int i = 0; i < STEPS_PER_LANE; i++) {
                const int s = m * STEPS_PER_LANE + i;
                const int ns = s + 1 < TOTAL_STEPS ? s + 1 : -1;
                unsigned int sw_next = 0, pw_next[8];
                if (PREFETCH && ns >= 0) load_step(ns, sw_next, pw_next);
                const unsigned int k16 = 8u * m + j + T_LANES_K * i;
                float sc[OPT];
                #pragma unroll
                for (int o = 0; o < OPT; o++) {
                    __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = (unsigned char)(sw_cur >> (8 * o));
                    sc[o] = (float)fp8 * s2;
                }
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    float wl[OPT], wh[OPT];
                    #pragma unroll
                    for (int o = 0; o < OPT; o++) {
                        const unsigned int byte = (pw_cur[b] >> (8 * o)) & 0xFF;
                        wl[o] = s_lut[byte & 0xF] * sc[o];
                        wh[o] = s_lut[byte >> 4] * sc[o];
                    }
                    #pragma unroll
                    for (int r = 0; r < ACTIVE_RMAX; r++) {
                        if (ACTIVE_RMAX <= 3 || (unsigned int)r < cnt) {
                            const unsigned int raw = s_a32_t8[r * ROW_U32 + k16 * 8u + b];
                            __nv_bfloat16 lo, hi;
                            *(unsigned short*)&lo = (unsigned short)(raw & 0xFFFF);
                            *(unsigned short*)&hi = (unsigned short)(raw >> 16);
                            const float al = __bfloat162float(lo);
                            const float ah = __bfloat162float(hi);
                            #pragma unroll
                            for (int o = 0; o < OPT; o++) {
                                if (m < 2) {
                                    pacc[m][r][o] += al * wl[o] + ah * wh[o];
                                } else {
                                    acc[r][o] += al * wl[o] + ah * wh[o];
                                }
                            }
                        }
                    }
                }
                if (ns >= 0) {
                    if (PREFETCH) {
                        sw_cur = sw_next;
                        #pragma unroll
                        for (int b = 0; b < 8; b++) pw_cur[b] = pw_next[b];
                    } else {
                        load_step(ns, sw_cur, pw_cur);
                    }
                }
            }
            if (m >= 2) {
                #pragma unroll
                for (int r = 0; r < ACTIVE_RMAX; r++)
                    #pragma unroll
                    for (int o = 0; o < OPT; o++)
                        pacc[m & 1][r][o] += acc[r][o];
            }
        }
    };
    struct Tag2 { enum { value = 2 }; };
    struct Tag3 { enum { value = (RMAX >= 3 ? 3 : RMAX) }; };
    struct TagMax { enum { value = RMAX }; };
    if (RMAX >= 2 && cnt == 2u) {
        run_rows(Tag2{});
    } else if (RMAX > 3 && cnt == 3u) {
        run_rows(Tag3{});
    } else {
        run_rows(TagMax{});
    }

    float* s_red = reinterpret_cast<float*>(s_a32_t8);
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < RMAX; r++) {
        if ((unsigned int)r < cnt) {
            float* s_part = s_red + r * (8 * OUTS) + j * OUTS + g * OPT;
            #pragma unroll
            for (int o = 0; o < OPT; o++) {
                s_part[o] = pacc[0][r][o] + pacc[1][r][o];
            }
        }
    }
    __syncthreads();
    const unsigned int c = threadIdx.x % (unsigned int)OUTS;
    const unsigned int r_lane = threadIdx.x / (unsigned int)OUTS;
    if (n_block + c < N) {
        for (unsigned int r = r_lane; r < cnt; r += (unsigned int)(THREADS / OUTS)) {
            const float* s_part = s_red + r * (8 * OUTS) + c;
            const float v04 = s_part[0 * OUTS] + s_part[4 * OUTS];
            const float v26 = s_part[2 * OUTS] + s_part[6 * OUTS];
            const float v15 = s_part[1 * OUTS] + s_part[5 * OUTS];
            const float v37 = s_part[3 * OUTS] + s_part[7 * OUTS];
            Cbase[(unsigned long long)s_flat[r] * N + n_block + c] =
                __float2bfloat16((v04 + v26) + (v15 + v37));
        }
    }
}

#define T_LANES_GU_WL_PARAMS \
    const __nv_bfloat16* __restrict__ A, \
    const unsigned long long* __restrict__ gate_packed_t_ptrs, \
    const unsigned long long* __restrict__ gate_scale_t_ptrs, \
    const float* __restrict__ gate_scale2_vals, \
    __nv_bfloat16* __restrict__ gate_out, \
    const unsigned long long* __restrict__ up_packed_t_ptrs, \
    const unsigned long long* __restrict__ up_scale_t_ptrs, \
    const float* __restrict__ up_scale2_vals, \
    __nv_bfloat16* __restrict__ up_out, \
    const unsigned int* __restrict__ expert_indices, \
    const unsigned int* __restrict__ unique_first_slot, \
    const unsigned int* __restrict__ unique_cnt, \
    const unsigned int* __restrict__ unique_flat, \
    const unsigned char* __restrict__ sh_gate_t_packed, \
    const unsigned char* __restrict__ sh_gate_t_scale, \
    float sh_gate_s2, \
    __nv_bfloat16* __restrict__ sh_gate_out, \
    const unsigned char* __restrict__ sh_up_t_packed, \
    const unsigned char* __restrict__ sh_up_t_scale, \
    float sh_up_s2, \
    __nv_bfloat16* __restrict__ sh_up_out, \
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows

#define T_LANES_GU_WL_ARGS \
    A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out, \
    up_packed_t_ptrs, up_scale_t_ptrs, up_scale2_vals, up_out, expert_indices, \
    unique_first_slot, unique_cnt, unique_flat, \
    sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2, sh_gate_out, \
    sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out, N, K, top_k, rows

extern "C" __global__ void __launch_bounds__(T8_THREADS, 3)
moe_expert_gate_up_worklist_t8_r3_k160(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t8_body<3, 160, T8_OPT, T8_THREADS, 1>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 3)
moe_expert_gate_up_worklist_t8_r5_k160(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t8_body<5, 160, T8_OPT, T8_THREADS, 1>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 2)
moe_expert_gate_up_worklist_t8_r8_k160(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t8_body<8, 160, T8_OPT, T8_THREADS, 1>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 3)
moe_expert_gate_up_worklist_t8_r3_k160_ts4e(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t8_body<3, 160, T8_OPT, T8_THREADS, 1, true>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 3)
moe_expert_gate_up_worklist_t8_r5_k160_ts4e(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t8_body<5, 160, T8_OPT, T8_THREADS, 1, true>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 2)
moe_expert_gate_up_worklist_t8_r8_k160_ts4e(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t8_body<8, 160, T8_OPT, T8_THREADS, 1, true>(T_LANES_GU_WL_ARGS);
}

// Materializes `act_out[i] = (gf / (1 + expf(-gf))) * uf` once per slot so the
// 40 N-blocks of `silu_down` do not recompute `expf` or re-read `up_out` 40x.
extern "C" __global__ void __launch_bounds__(256)
moe_silu_mul_rows_f32(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    float* __restrict__ act_out,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    float* __restrict__ sh_act_out,
    unsigned int routed_elems,
    unsigned int shared_elems)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < routed_elems) {
        const float gf = __bfloat162float(__ldg(&gate_out[idx]));
        const float uf = __bfloat162float(__ldg(&up_out[idx]));
        act_out[idx] = (gf / (1.0f + __expf(-gf))) * uf;
    } else {
        const unsigned int s_idx = idx - routed_elems;
        if (s_idx < shared_elems) {
            const float gf = __bfloat162float(__ldg(&sh_gate_in[s_idx]));
            const float uf = __bfloat162float(__ldg(&sh_up_in[s_idx]));
            sh_act_out[s_idx] = (gf / (1.0f + __expf(-gf))) * uf;
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

template <int RMAX, int K16C, int OPT, int THREADS, int PREFETCH, bool TS4E = false>
__device__ __forceinline__ void down_act_t_lanes_rows_t8_body(
    const float* __restrict__ act_in,
    const void* const* __restrict__ packed_t_ptrs,
    const void* const* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    const unsigned int* __restrict__ expert_indices,
    const unsigned int* __restrict__ unique_first_slot,
    const unsigned int* __restrict__ unique_cnt,
    const unsigned int* __restrict__ unique_flat,
    __nv_bfloat16* __restrict__ C,
    const float* __restrict__ sh_act_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N,
    unsigned int K,
    unsigned int top_k,
    unsigned int rows)
{
    static_assert(K16C >= T_LANES_K && K16C <= 2 * T_LANES_K, "every lane owns one or two k16 steps");
    static_assert(K16C % 8 == 0, "per-thread step count must not depend on j");
    static_assert(THREADS % 32 == 0 && (OPT == 2 || OPT == 4), "t8 launch shape");
    constexpr int OUTS = (THREADS / 8) * OPT;
    constexpr int KC = 16 * K16C;
    constexpr int ROW_STRIDE = KC;
    (void)K;

    if (rows == 0u || rows > 128u) return;
    const unsigned int total = rows * top_k;
    const unsigned int shared_ctas = (rows + (unsigned int)RMAX - 1u) / (unsigned int)RMAX;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y < shared_ctas);
    const unsigned int wl_y = y - shared_ctas;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    unsigned int cnt;
    __shared__ unsigned int s_flat[RMAX];

    if (is_shared) {
        const unsigned int sh_base = y * (unsigned int)RMAX;
        if (sh_base >= rows) return;
        B_packed = sh_down_t_packed;
        B_scale = sh_down_t_scale;
        s2 = sh_down_s2;
        const unsigned int rem = rows - sh_base;
        cnt = rem > (unsigned int)RMAX ? (unsigned int)RMAX : rem;
        if (threadIdx.x < (unsigned int)RMAX) s_flat[threadIdx.x] = sh_base + threadIdx.x;
    } else {
        if (wl_y >= total) return;
        const unsigned int first_slot = unique_first_slot[wl_y];
        if (first_slot == 0xFFFFFFFFu) return;
        const unsigned int expert_id = expert_indices[first_slot];
        B_packed = (const unsigned char*)packed_t_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_t_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        cnt = unique_cnt[wl_y];
        if (threadIdx.x < (unsigned int)RMAX) {
            s_flat[threadIdx.x] = (threadIdx.x < cnt) ? unique_flat[wl_y * 10u + threadIdx.x] : 0u;
        }
    }

    const unsigned int n_block = blockIdx.x * OUTS;
    __nv_bfloat16* Cbase = is_shared ? sh_down_out : C;
    if (B_packed == 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
        __syncthreads();
        if (threadIdx.x < OUTS && n_block + threadIdx.x < N) {
            #pragma unroll
            for (int r = 0; r < RMAX; r++) {
                if ((unsigned int)r < cnt)
                    Cbase[(unsigned long long)s_flat[r] * N + n_block + threadIdx.x] = __float2bfloat16(0.0f);
            }
        }
        return;
    }

    const unsigned int g = threadIdx.x % 32u;
    const unsigned int j = threadIdx.x / 32u;
    const unsigned int n0 = n_block + g * OPT;
    const bool valid = n0 < N;
    const unsigned int base_word = (TS4E && !is_shared && valid)
        ? t_lanes_load<OPT>(B_scale + static_cast<unsigned long long>(KC >> 5) * N, n0, N)
        : 0u;
    auto load_step = [&](int s, unsigned int& sw, unsigned int (&pw)[8]) {
        const unsigned int k16 = 8u * (s / 2) + j + T_LANES_K * (s % 2);
        if constexpr (TS4E) {
            sw = valid
                ? (is_shared
                    ? t_lanes_load<OPT>(B_scale + (unsigned long long)k16 * N, n0, N)
                    : ts4e_load_scale_word<OPT, KC>(B_scale, base_word, k16, n0, N))
                : 0u;
        } else {
            sw = valid ? t_lanes_load<OPT>(B_scale + (unsigned long long)k16 * N, n0, N) : 0u;
        }
        #pragma unroll
        for (int b = 0; b < 8; b++)
            pw[b] = valid ? t_lanes_load<OPT>(B_packed + (unsigned long long)(k16 * 8 + b) * N, n0, N) : 0u;
    };
    unsigned int sw_cur = 0, pw_cur[8];
    if (PREFETCH) load_step(0, sw_cur, pw_cur);

    __shared__ float s_lut[16];
    __shared__ __align__(16) float s_act_t8[RMAX * ROW_STRIDE];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    __syncthreads();
    const float* act_base = is_shared ? sh_act_in : act_in;
    for (unsigned int r = 0; r < cnt; r++) {
        const float4* a_ptr4 = reinterpret_cast<const float4*>(act_base + (unsigned long long)s_flat[r] * KC);
        float4* s_row4 = reinterpret_cast<float4*>(s_act_t8 + r * ROW_STRIDE);
        for (unsigned int i = threadIdx.x; i < (unsigned int)(KC / 4); i += blockDim.x) {
            s_row4[i] = a_ptr4[i];
        }
    }
    __syncthreads();
    if (!PREFETCH) load_step(0, sw_cur, pw_cur);

    if (cnt == 1u) {
        float pacc0[2][OPT];
        #pragma unroll
        for (int h = 0; h < 2; h++)
            #pragma unroll
            for (int o = 0; o < OPT; o++) pacc0[h][o] = 0.0f;
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            if (!t8_step_valid<K16C>(s)) continue;
            const int m = s / 2;
            const int ns = t8_next_step<K16C>(s);
            unsigned int sw_next = 0, pw_next[8];
            if (PREFETCH && ns >= 0) load_step(ns, sw_next, pw_next);
            const unsigned int k16 = 8u * m + j + T_LANES_K * (s % 2);
            float sc[OPT];
            #pragma unroll
            for (int o = 0; o < OPT; o++) {
                __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = (unsigned char)(sw_cur >> (8 * o));
                sc[o] = (float)fp8 * s2;
            }
            if (m < 2) {
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    const float2 a = *(const float2*)(s_act_t8 + k16 * 16u + b * 2);
                    #pragma unroll
                    for (int o = 0; o < OPT; o++) {
                        const unsigned int byte = (pw_cur[b] >> (8 * o)) & 0xFF;
                        const float wl = s_lut[byte & 0xF] * sc[o];
                        const float wh = s_lut[byte >> 4] * sc[o];
                        pacc0[m][o] += a.x * wl + a.y * wh;
                    }
                }
            } else {
                float acc_r[OPT];
                #pragma unroll
                for (int o = 0; o < OPT; o++) acc_r[o] = 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    const float2 a = *(const float2*)(s_act_t8 + k16 * 16u + b * 2);
                    #pragma unroll
                    for (int o = 0; o < OPT; o++) {
                        const unsigned int byte = (pw_cur[b] >> (8 * o)) & 0xFF;
                        const float wl = s_lut[byte & 0xF] * sc[o];
                        const float wh = s_lut[byte >> 4] * sc[o];
                        acc_r[o] += a.x * wl + a.y * wh;
                    }
                }
                #pragma unroll
                for (int o = 0; o < OPT; o++) {
                    pacc0[m & 1][o] += acc_r[o];
                }
            }
            if (ns >= 0) {
                if (PREFETCH) {
                    sw_cur = sw_next;
                    #pragma unroll
                    for (int b = 0; b < 8; b++) pw_cur[b] = pw_next[b];
                } else {
                    load_step(ns, sw_cur, pw_cur);
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int o = 0; o < OPT; o++) {
            s_act_t8[j * OUTS + g * OPT + o] = pacc0[0][o] + pacc0[1][o];
        }
        __syncthreads();
        if (threadIdx.x < (unsigned int)OUTS && n_block + threadIdx.x < N) {
            const unsigned int c = threadIdx.x;
            const float v04 = s_act_t8[0 * OUTS + c] + s_act_t8[4 * OUTS + c];
            const float v26 = s_act_t8[2 * OUTS + c] + s_act_t8[6 * OUTS + c];
            const float v15 = s_act_t8[1 * OUTS + c] + s_act_t8[5 * OUTS + c];
            const float v37 = s_act_t8[3 * OUTS + c] + s_act_t8[7 * OUTS + c];
            Cbase[(unsigned long long)s_flat[0] * N + n_block + c] =
                __float2bfloat16((v04 + v26) + (v15 + v37));
        }
        return;
    }

    auto run_rows = [&](auto active_tag) {
        constexpr int ACTIVE_RMAX = decltype(active_tag)::value;
        float pacc[2][ACTIVE_RMAX][OPT];
        #pragma unroll
        for (int h = 0; h < 2; h++)
            #pragma unroll
            for (int r = 0; r < ACTIVE_RMAX; r++)
                #pragma unroll
                for (int o = 0; o < OPT; o++) pacc[h][r][o] = 0.0f;
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            if (!t8_step_valid<K16C>(s)) continue;
            const int m = s / 2;
            const int ns = t8_next_step<K16C>(s);
            unsigned int sw_next = 0, pw_next[8];
            if (PREFETCH && ns >= 0) load_step(ns, sw_next, pw_next);
            const unsigned int k16 = 8u * m + j + T_LANES_K * (s % 2);
            float sc[OPT];
            #pragma unroll
            for (int o = 0; o < OPT; o++) {
                __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = (unsigned char)(sw_cur >> (8 * o));
                sc[o] = (float)fp8 * s2;
            }
            if (m < 2) {
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    float wl[OPT], wh[OPT];
                    #pragma unroll
                    for (int o = 0; o < OPT; o++) {
                        const unsigned int byte = (pw_cur[b] >> (8 * o)) & 0xFF;
                        wl[o] = s_lut[byte & 0xF] * sc[o];
                        wh[o] = s_lut[byte >> 4] * sc[o];
                    }
                    #pragma unroll
                    for (int r = 0; r < ACTIVE_RMAX; r++) {
                        if (ACTIVE_RMAX <= 5 || (unsigned int)r < cnt) {
                            const float2 a = *(const float2*)(s_act_t8 + r * ROW_STRIDE + k16 * 16u + b * 2);
                            #pragma unroll
                            for (int o = 0; o < OPT; o++) {
                                pacc[m][r][o] += a.x * wl[o] + a.y * wh[o];
                            }
                        }
                    }
                }
            } else {
                float acc_m[ACTIVE_RMAX][OPT];
                #pragma unroll
                for (int r = 0; r < ACTIVE_RMAX; r++)
                    #pragma unroll
                    for (int o = 0; o < OPT; o++) acc_m[r][o] = 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    float wl[OPT], wh[OPT];
                    #pragma unroll
                    for (int o = 0; o < OPT; o++) {
                        const unsigned int byte = (pw_cur[b] >> (8 * o)) & 0xFF;
                        wl[o] = s_lut[byte & 0xF] * sc[o];
                        wh[o] = s_lut[byte >> 4] * sc[o];
                    }
                    #pragma unroll
                    for (int r = 0; r < ACTIVE_RMAX; r++) {
                        if (ACTIVE_RMAX <= 5 || (unsigned int)r < cnt) {
                            const float2 a = *(const float2*)(s_act_t8 + r * ROW_STRIDE + k16 * 16u + b * 2);
                            #pragma unroll
                            for (int o = 0; o < OPT; o++) {
                                acc_m[r][o] += a.x * wl[o] + a.y * wh[o];
                            }
                        }
                    }
                }
                #pragma unroll
                for (int r = 0; r < ACTIVE_RMAX; r++) {
                    if (ACTIVE_RMAX <= 5 || (unsigned int)r < cnt) {
                        #pragma unroll
                        for (int o = 0; o < OPT; o++) {
                            pacc[m & 1][r][o] += acc_m[r][o];
                        }
                    }
                }
            }
            if (ns >= 0) {
                if (PREFETCH) {
                    sw_cur = sw_next;
                    #pragma unroll
                    for (int b = 0; b < 8; b++) pw_cur[b] = pw_next[b];
                } else {
                    load_step(ns, sw_cur, pw_cur);
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < ACTIVE_RMAX; r++) {
            if (ACTIVE_RMAX <= 5 || (unsigned int)r < cnt) {
                float* s_part = s_act_t8 + r * (8 * OUTS) + j * OUTS + g * OPT;
                #pragma unroll
                for (int o = 0; o < OPT; o++) {
                    s_part[o] = pacc[0][r][o] + pacc[1][r][o];
                }
            }
        }
    };
    struct Tag2 { enum { value = 2 }; };
    struct Tag3 { enum { value = (RMAX >= 3 ? 3 : RMAX) }; };
    struct Tag4 { enum { value = (RMAX >= 4 ? 4 : RMAX) }; };
    struct Tag5 { enum { value = (RMAX >= 5 ? 5 : RMAX) }; };
    struct TagMax { enum { value = RMAX }; };
    if (RMAX >= 2 && cnt == 2u) {
        run_rows(Tag2{});
    } else if (RMAX > 3 && cnt == 3u) {
        run_rows(Tag3{});
    } else if (RMAX > 4 && cnt == 4u) {
        run_rows(Tag4{});
    } else if (RMAX > 5 && cnt == 5u) {
        run_rows(Tag5{});
    } else {
        run_rows(TagMax{});
    }
    __syncthreads();
    const unsigned int c = threadIdx.x % (unsigned int)OUTS;
    const unsigned int r_lane = threadIdx.x / (unsigned int)OUTS;
    if (n_block + c < N) {
        for (unsigned int r = r_lane; r < cnt; r += (unsigned int)(THREADS / OUTS)) {
            const float* s_part = s_act_t8 + r * (8 * OUTS) + c;
            const float v04 = s_part[0 * OUTS] + s_part[4 * OUTS];
            const float v26 = s_part[2 * OUTS] + s_part[6 * OUTS];
            const float v15 = s_part[1 * OUTS] + s_part[5 * OUTS];
            const float v37 = s_part[3 * OUTS] + s_part[7 * OUTS];
            Cbase[(unsigned long long)s_flat[r] * N + n_block + c] =
                __float2bfloat16((v04 + v26) + (v15 + v37));
        }
    }
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 3)
moe_expert_down_act_shared_t_lanes_rows_t8_r3_k40(
    const float* __restrict__ act_in,
    const void* const* __restrict__ packed_t_ptrs,
    const void* const* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    const unsigned int* __restrict__ expert_indices,
    const unsigned int* __restrict__ unique_first_slot,
    const unsigned int* __restrict__ unique_cnt,
    const unsigned int* __restrict__ unique_flat,
    __nv_bfloat16* __restrict__ C,
    const float* __restrict__ sh_act_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows)
{
    down_act_t_lanes_rows_t8_body<3, 40, T8_OPT, T8_THREADS, 1>(
        act_in, packed_t_ptrs, scale_t_ptrs, scale2_vals, expert_indices,
        unique_first_slot, unique_cnt, unique_flat, C,
        sh_act_in, sh_down_t_packed, sh_down_t_scale, sh_down_s2, sh_down_out,
        N, K, top_k, rows);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 2)
moe_expert_down_act_shared_t_lanes_rows_t8_r5_k40(
    const float* __restrict__ act_in,
    const void* const* __restrict__ packed_t_ptrs,
    const void* const* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    const unsigned int* __restrict__ expert_indices,
    const unsigned int* __restrict__ unique_first_slot,
    const unsigned int* __restrict__ unique_cnt,
    const unsigned int* __restrict__ unique_flat,
    __nv_bfloat16* __restrict__ C,
    const float* __restrict__ sh_act_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows)
{
    down_act_t_lanes_rows_t8_body<5, 40, T8_OPT, T8_THREADS, 1>(
        act_in, packed_t_ptrs, scale_t_ptrs, scale2_vals, expert_indices,
        unique_first_slot, unique_cnt, unique_flat, C,
        sh_act_in, sh_down_t_packed, sh_down_t_scale, sh_down_s2, sh_down_out,
        N, K, top_k, rows);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 2)
moe_expert_down_act_shared_t_lanes_rows_t8_r8_k40(
    const float* __restrict__ act_in,
    const void* const* __restrict__ packed_t_ptrs,
    const void* const* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    const unsigned int* __restrict__ expert_indices,
    const unsigned int* __restrict__ unique_first_slot,
    const unsigned int* __restrict__ unique_cnt,
    const unsigned int* __restrict__ unique_flat,
    __nv_bfloat16* __restrict__ C,
    const float* __restrict__ sh_act_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows)
{
    down_act_t_lanes_rows_t8_body<8, 40, T8_OPT, T8_THREADS, 1>(
        act_in, packed_t_ptrs, scale_t_ptrs, scale2_vals, expert_indices,
        unique_first_slot, unique_cnt, unique_flat, C,
        sh_act_in, sh_down_t_packed, sh_down_t_scale, sh_down_s2, sh_down_out,
        N, K, top_k, rows);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 2)
moe_expert_down_act_shared_t_lanes_rows_t8_r10_k40(
    const float* __restrict__ act_in,
    const void* const* __restrict__ packed_t_ptrs,
    const void* const* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    const unsigned int* __restrict__ expert_indices,
    const unsigned int* __restrict__ unique_first_slot,
    const unsigned int* __restrict__ unique_cnt,
    const unsigned int* __restrict__ unique_flat,
    __nv_bfloat16* __restrict__ C,
    const float* __restrict__ sh_act_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows)
{
    down_act_t_lanes_rows_t8_body<10, 40, T8_OPT, T8_THREADS, 1>(
        act_in, packed_t_ptrs, scale_t_ptrs, scale2_vals, expert_indices,
        unique_first_slot, unique_cnt, unique_flat, C,
        sh_act_in, sh_down_t_packed, sh_down_t_scale, sh_down_s2, sh_down_out,
        N, K, top_k, rows);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 3)
moe_expert_down_act_shared_t_lanes_rows_t8_r3_k40_ts4e(
    const float* __restrict__ act_in,
    const void* const* __restrict__ packed_t_ptrs,
    const void* const* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    const unsigned int* __restrict__ expert_indices,
    const unsigned int* __restrict__ unique_first_slot,
    const unsigned int* __restrict__ unique_cnt,
    const unsigned int* __restrict__ unique_flat,
    __nv_bfloat16* __restrict__ C,
    const float* __restrict__ sh_act_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows)
{
    down_act_t_lanes_rows_t8_body<3, 40, T8_OPT, T8_THREADS, 1, true>(
        act_in, packed_t_ptrs, scale_t_ptrs, scale2_vals, expert_indices,
        unique_first_slot, unique_cnt, unique_flat, C,
        sh_act_in, sh_down_t_packed, sh_down_t_scale, sh_down_s2, sh_down_out,
        N, K, top_k, rows);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 2)
moe_expert_down_act_shared_t_lanes_rows_t8_r5_k40_ts4e(
    const float* __restrict__ act_in,
    const void* const* __restrict__ packed_t_ptrs,
    const void* const* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    const unsigned int* __restrict__ expert_indices,
    const unsigned int* __restrict__ unique_first_slot,
    const unsigned int* __restrict__ unique_cnt,
    const unsigned int* __restrict__ unique_flat,
    __nv_bfloat16* __restrict__ C,
    const float* __restrict__ sh_act_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows)
{
    down_act_t_lanes_rows_t8_body<5, 40, T8_OPT, T8_THREADS, 1, true>(
        act_in, packed_t_ptrs, scale_t_ptrs, scale2_vals, expert_indices,
        unique_first_slot, unique_cnt, unique_flat, C,
        sh_act_in, sh_down_t_packed, sh_down_t_scale, sh_down_s2, sh_down_out,
        N, K, top_k, rows);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 2)
moe_expert_down_act_shared_t_lanes_rows_t8_r8_k40_ts4e(
    const float* __restrict__ act_in,
    const void* const* __restrict__ packed_t_ptrs,
    const void* const* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    const unsigned int* __restrict__ expert_indices,
    const unsigned int* __restrict__ unique_first_slot,
    const unsigned int* __restrict__ unique_cnt,
    const unsigned int* __restrict__ unique_flat,
    __nv_bfloat16* __restrict__ C,
    const float* __restrict__ sh_act_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows)
{
    down_act_t_lanes_rows_t8_body<8, 40, T8_OPT, T8_THREADS, 1, true>(
        act_in, packed_t_ptrs, scale_t_ptrs, scale2_vals, expert_indices,
        unique_first_slot, unique_cnt, unique_flat, C,
        sh_act_in, sh_down_t_packed, sh_down_t_scale, sh_down_s2, sh_down_out,
        N, K, top_k, rows);
}

extern "C" __global__ void __launch_bounds__(T8_THREADS, 2)
moe_expert_down_act_shared_t_lanes_rows_t8_r10_k40_ts4e(
    const float* __restrict__ act_in,
    const void* const* __restrict__ packed_t_ptrs,
    const void* const* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    const unsigned int* __restrict__ expert_indices,
    const unsigned int* __restrict__ unique_first_slot,
    const unsigned int* __restrict__ unique_cnt,
    const unsigned int* __restrict__ unique_flat,
    __nv_bfloat16* __restrict__ C,
    const float* __restrict__ sh_act_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows)
{
    down_act_t_lanes_rows_t8_body<10, 40, T8_OPT, T8_THREADS, 1, true>(
        act_in, packed_t_ptrs, scale_t_ptrs, scale2_vals, expert_indices,
        unique_first_slot, unique_cnt, unique_flat, C,
        sh_act_in, sh_down_t_packed, sh_down_t_scale, sh_down_s2, sh_down_out,
        N, K, top_k, rows);
}

#define T_LANES_GU_WL_PARAMS \
    const __nv_bfloat16* __restrict__ A, \
    const unsigned long long* __restrict__ gate_packed_t_ptrs, \
    const unsigned long long* __restrict__ gate_scale_t_ptrs, \
    const float* __restrict__ gate_scale2_vals, \
    __nv_bfloat16* __restrict__ gate_out, \
    const unsigned long long* __restrict__ up_packed_t_ptrs, \
    const unsigned long long* __restrict__ up_scale_t_ptrs, \
    const float* __restrict__ up_scale2_vals, \
    __nv_bfloat16* __restrict__ up_out, \
    const unsigned int* __restrict__ expert_indices, \
    const unsigned int* __restrict__ unique_first_slot, \
    const unsigned int* __restrict__ unique_cnt, \
    const unsigned int* __restrict__ unique_flat, \
    const unsigned char* __restrict__ sh_gate_t_packed, \
    const unsigned char* __restrict__ sh_gate_t_scale, \
    float sh_gate_s2, \
    __nv_bfloat16* __restrict__ sh_gate_out, \
    const unsigned char* __restrict__ sh_up_t_packed, \
    const unsigned char* __restrict__ sh_up_t_scale, \
    float sh_up_s2, \
    __nv_bfloat16* __restrict__ sh_up_out, \
    unsigned int N, \
    unsigned int K, \
    unsigned int top_k, \
    unsigned int rows

#define T_LANES_GU_WL_ARGS \
    A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out, \
    up_packed_t_ptrs, up_scale_t_ptrs, up_scale2_vals, up_out, expert_indices, \
    unique_first_slot, unique_cnt, unique_flat, \
    sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2, sh_gate_out, \
    sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out, N, K, top_k, rows

template <int OPT, int TPL, int RMAX, bool TS4E = false>
__device__ __forceinline__ void gate_up_worklist_t_lanes_rows_staged_body(T_LANES_GU_WL_PARAMS) {
    constexpr unsigned int OB = TPL * OPT;
    constexpr int SMEM_RMAX = (RMAX > 8) ? 8 : RMAX;
    if (rows == 0u || rows > 128u) return;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    const unsigned int total = rows * top_k;
    const unsigned int shared_ctas = (rows + (unsigned int)SMEM_RMAX - 1u) / (unsigned int)SMEM_RMAX;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y < shared_ctas);
    const unsigned int wl_y = y - shared_ctas;
    const unsigned int sh_base = is_shared ? y * (unsigned int)SMEM_RMAX : 0u;
    if (is_shared && sh_base >= rows) return;

    unsigned int first_slot = 0u;
    unsigned int expert_id = 0u;
    if (!is_shared) {
        if (wl_y >= total) return;
        first_slot = __ldg(&unique_first_slot[wl_y]);
        if (first_slot == 0xFFFFFFFFu) return;
        expert_id = __ldg(&expert_indices[first_slot]);
    }

    const int cnt = is_shared
        ? (int)((rows - sh_base > (unsigned int)SMEM_RMAX) ? (unsigned int)SMEM_RMAX : (rows - sh_base))
        : (int)__ldg(&unique_cnt[wl_y]);
    const unsigned int q = threadIdx.x % TPL;
    const unsigned int l = threadIdx.x / TPL;
    const unsigned int n_block = blockIdx.x * OB;
    const unsigned int n0 = n_block + q * OPT;
    const unsigned char* gate_packed = is_shared ? sh_gate_t_packed : (const unsigned char*)gate_packed_t_ptrs[expert_id];
    if (gate_packed == 0) {
        for (int m = 0; m < cnt; m++) {
            const unsigned int flat_m = is_shared ? (sh_base + (unsigned int)m) : __ldg(&unique_flat[wl_y * 10u + m]);
            if (threadIdx.x < OB && n_block + threadIdx.x < N) {
                (is_shared ? sh_gate_out : gate_out)[(unsigned long long)flat_m * N + n_block + threadIdx.x] = __float2bfloat16(0.0f);
                (is_shared ? sh_up_out : up_out)[(unsigned long long)flat_m * N + n_block + threadIdx.x] = __float2bfloat16(0.0f);
            }
        }
        return;
    }

    __shared__ unsigned int s_flat[RMAX];
    __shared__ float s_lut[16];
    constexpr unsigned int BASE_SLOTS = TS4E ? (2u * TPL) : 1u;
    __shared__ unsigned int s_base[BASE_SLOTS];
    extern __shared__ __align__(16) unsigned int s_a32[]; // [SMEM_RMAX][K/2] raw BF16 pairs followed by part[T_LANES_K / 2][OB+1]
    if (threadIdx.x < 16u) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    if constexpr (TS4E) {
        if (!is_shared && threadIdx.x < 2u * TPL) {
            const unsigned int p = threadIdx.x / TPL;
            const unsigned int qq = threadIdx.x % TPL;
            const unsigned int col0 = n_block + qq * OPT;
            const unsigned char* sc_ptr = reinterpret_cast<const unsigned char*>(
                (p == 0u ? gate_scale_t_ptrs : up_scale_t_ptrs)[expert_id]);
            s_base[threadIdx.x] = (col0 < N)
                ? t_lanes_load<OPT>(sc_ptr + (unsigned long long)(K >> 5) * N, col0, N)
                : 0u;
        }
    }
    if (threadIdx.x < (unsigned int)RMAX) {
        s_flat[threadIdx.x] = is_shared
            ? (sh_base + threadIdx.x)
            : (((int)threadIdx.x < cnt) ? __ldg(&unique_flat[wl_y * 10u + threadIdx.x]) : 0u);
    }
    const int first_smem_cnt = (cnt > SMEM_RMAX) ? SMEM_RMAX : cnt;
    const unsigned int K2 = K / 2;
    if ((K & 7u) == 0u) {
        const unsigned int K8 = K >> 3;
        uint4* s_a128 = reinterpret_cast<uint4*>(s_a32);
        for (int m = 0; m < first_smem_cnt; m++) {
            const unsigned int tok = is_shared ? (sh_base + (unsigned int)m) : (__ldg(&unique_flat[wl_y * 10u + m]) / top_k);
            const uint4* A128 = reinterpret_cast<const uint4*>(A + (unsigned long long)tok * K);
            for (unsigned int i = threadIdx.x; i < K8; i += blockDim.x) s_a128[m * K8 + i] = __ldg(&A128[i]);
        }
    } else {
        for (int m = 0; m < first_smem_cnt; m++) {
            const unsigned int tok = is_shared ? (sh_base + (unsigned int)m) : (__ldg(&unique_flat[wl_y * 10u + m]) / top_k);
            const unsigned int* A32 = (const unsigned int*)(A + (unsigned long long)tok * K);
            for (unsigned int i = threadIdx.x; i < K2; i += blockDim.x) s_a32[m * K2 + i] = __ldg(&A32[i]);
        }
    }
    __syncthreads();

    float (*part)[OB + 1] = reinterpret_cast<float (*)[OB + 1]>(s_a32 + (unsigned int)SMEM_RMAX * K2);
    if (cnt == 1) {
        #pragma unroll 1
        for (unsigned int proj = 0; proj < 2u; ++proj) {
            const unsigned char* B_packed = is_shared
                ? (proj == 0u ? sh_gate_t_packed : sh_up_t_packed)
                : (const unsigned char*)(proj == 0u ? gate_packed_t_ptrs : up_packed_t_ptrs)[expert_id];
            const unsigned char* B_scale = is_shared
                ? (proj == 0u ? sh_gate_t_scale : sh_up_t_scale)
                : (const unsigned char*)(proj == 0u ? gate_scale_t_ptrs : up_scale_t_ptrs)[expert_id];
            const float s2 = is_shared
                ? (proj == 0u ? sh_gate_s2 : sh_up_s2)
                : (proj == 0u ? gate_scale2_vals : up_scale2_vals)[expert_id];
            __nv_bfloat16* Cbase = is_shared
                ? (proj == 0u ? sh_gate_out : sh_up_out)
                : (proj == 0u ? gate_out : up_out);
            float acc1[1][OPT];
            #pragma unroll
            for (int o = 0; o < OPT; o++) acc1[0][o] = 0.0f;
            if (n0 < N) {
                auto act1 = [&](int, unsigned int k16, int b, float& al, float& ah) {
                    const unsigned int raw = s_a32[k16 * 8 + b];
                    __nv_bfloat16 lo, hi;
                    *(unsigned short*)&lo = (unsigned short)(raw & 0xFFFF);
                    *(unsigned short*)&hi = (unsigned short)(raw >> 16);
                    al = __bfloat162float(lo);
                    ah = __bfloat162float(hi);
                };
                t_lanes_accumulate_rows<OPT, 1, 1, TS4E>(
                    B_packed, B_scale, s2, s_lut, l, n0, N, K, 1, act1, acc1,
                    s_base, proj * TPL + q, !is_shared);
            }
            if (proj > 0u) __syncthreads();
            if (l >= 16u) {
                #pragma unroll
                for (int o = 0; o < OPT; o++) part[l - 16u][q * OPT + o] = acc1[0][o];
            }
            __syncthreads();
            if (l < 16u) {
                #pragma unroll
                for (int o = 0; o < OPT; o++) part[l][q * OPT + o] = acc1[0][o] + part[l][q * OPT + o];
            }
            __syncthreads();
            __nv_bfloat16* C = Cbase + (unsigned long long)s_flat[0] * N;
            const unsigned int c = threadIdx.x >> 3;
            const unsigned int j = threadIdx.x & 7u;
            float v = part[j][c] + part[j + 8u][c];
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 4, 8);
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 2, 8);
            v = v + __shfl_down_sync(0xFFFFFFFFu, v, 1, 8);
            if (j == 0u && n_block + c < N) C[n_block + c] = __float2bfloat16(v);
        }
        return;
    }

    for (int m_base = 0; m_base < cnt; m_base += SMEM_RMAX) {
        const int rem_cnt = cnt - m_base;
        const int tile_cnt = (rem_cnt > SMEM_RMAX) ? SMEM_RMAX : rem_cnt;
        if (m_base > 0) {
            __syncthreads();
            if ((K & 7u) == 0u) {
                const unsigned int K8 = K >> 3;
                uint4* s_a128 = reinterpret_cast<uint4*>(s_a32);
                for (int m = 0; m < tile_cnt; m++) {
                    const unsigned int tok = is_shared ? s_flat[m_base + m] : (s_flat[m_base + m] / top_k);
                    const uint4* A128 = reinterpret_cast<const uint4*>(A + (unsigned long long)tok * K);
                    for (unsigned int i = threadIdx.x; i < K8; i += blockDim.x) s_a128[m * K8 + i] = __ldg(&A128[i]);
                }
            } else {
                for (int m = 0; m < tile_cnt; m++) {
                    const unsigned int tok = is_shared ? s_flat[m_base + m] : (s_flat[m_base + m] / top_k);
                    const unsigned int* A32 = (const unsigned int*)(A + (unsigned long long)tok * K);
                    for (unsigned int i = threadIdx.x; i < K2; i += blockDim.x) s_a32[m * K2 + i] = __ldg(&A32[i]);
                }
            }
            __syncthreads();
        }
        #pragma unroll 1
        for (unsigned int proj = 0; proj < 2u; ++proj) {
            const unsigned char* B_packed = is_shared
                ? (proj == 0u ? sh_gate_t_packed : sh_up_t_packed)
                : (const unsigned char*)(proj == 0u ? gate_packed_t_ptrs : up_packed_t_ptrs)[expert_id];
            const unsigned char* B_scale = is_shared
                ? (proj == 0u ? sh_gate_t_scale : sh_up_t_scale)
                : (const unsigned char*)(proj == 0u ? gate_scale_t_ptrs : up_scale_t_ptrs)[expert_id];
            const float s2 = is_shared
                ? (proj == 0u ? sh_gate_s2 : sh_up_s2)
                : (proj == 0u ? gate_scale2_vals : up_scale2_vals)[expert_id];
            __nv_bfloat16* Cbase = is_shared
                ? (proj == 0u ? sh_gate_out : sh_up_out)
                : (proj == 0u ? gate_out : up_out);
            float acc[SMEM_RMAX][OPT];
            #pragma unroll
            for (int m = 0; m < SMEM_RMAX; m++)
                #pragma unroll
                for (int o = 0; o < OPT; o++) acc[m][o] = 0.0f;
            if (n0 < N) {
                const unsigned int row_u32 = (K == 2560u) ? 1280u : K2;
                auto act_smem = [&](int m, unsigned int k16, int b, float& al, float& ah) {
                    const unsigned int raw = s_a32[(unsigned int)m * row_u32 + k16 * 8 + b];
                    __nv_bfloat16 lo, hi;
                    *(unsigned short*)&lo = (unsigned short)(raw & 0xFFFF);
                    *(unsigned short*)&hi = (unsigned short)(raw >> 16);
                    al = __bfloat162float(lo);
                    ah = __bfloat162float(hi);
                };
                const unsigned int base_idx = proj * TPL + q;
                const bool use_ts4e = !is_shared;
                if (tile_cnt == 1) {
                    t_lanes_accumulate_rows<OPT, SMEM_RMAX, 1, TS4E>(B_packed, B_scale, s2, s_lut, l, n0, N, K, 1, act_smem, acc, s_base, base_idx, use_ts4e);
                } else if (SMEM_RMAX >= 2 && tile_cnt == 2) {
                    t_lanes_accumulate_rows<OPT, SMEM_RMAX, 2, TS4E>(B_packed, B_scale, s2, s_lut, l, n0, N, K, 2, act_smem, acc, s_base, base_idx, use_ts4e);
                } else if (SMEM_RMAX >= 3 && tile_cnt == 3) {
                    t_lanes_accumulate_rows<OPT, SMEM_RMAX, 3, TS4E>(B_packed, B_scale, s2, s_lut, l, n0, N, K, 3, act_smem, acc, s_base, base_idx, use_ts4e);
                } else if (SMEM_RMAX >= 4 && tile_cnt == 4) {
                    t_lanes_accumulate_rows<OPT, SMEM_RMAX, 4, TS4E>(B_packed, B_scale, s2, s_lut, l, n0, N, K, 4, act_smem, acc, s_base, base_idx, use_ts4e);
                } else if (SMEM_RMAX >= 5 && tile_cnt <= 5) {
                    t_lanes_accumulate_rows<OPT, SMEM_RMAX, 5, TS4E>(B_packed, B_scale, s2, s_lut, l, n0, N, K, tile_cnt, act_smem, acc, s_base, base_idx, use_ts4e);
                } else {
                    t_lanes_accumulate_rows<OPT, SMEM_RMAX, SMEM_RMAX, TS4E>(B_packed, B_scale, s2, s_lut, l, n0, N, K, tile_cnt, act_smem, acc, s_base, base_idx, use_ts4e);
                }
            }
            #pragma unroll
            for (int m = 0; m < SMEM_RMAX; m++) {
                if (m >= tile_cnt) break;
                if (m > 0 || proj > 0u || m_base > 0) __syncthreads();
                if (l >= 16u) {
                    #pragma unroll
                    for (int o = 0; o < OPT; o++) part[l - 16u][q * OPT + o] = acc[m][o];
                }
                __syncthreads();
                if (l < 16u) {
                    #pragma unroll
                    for (int o = 0; o < OPT; o++) part[l][q * OPT + o] = acc[m][o] + part[l][q * OPT + o];
                }
                __syncthreads();
                __nv_bfloat16* C = Cbase + (unsigned long long)s_flat[m_base + m] * N;
                const unsigned int c = threadIdx.x >> 3;
                const unsigned int j = threadIdx.x & 7u;
                float v = part[j][c] + part[j + 8u][c];
                v = v + __shfl_down_sync(0xFFFFFFFFu, v, 4, 8);
                v = v + __shfl_down_sync(0xFFFFFFFFu, v, 2, 8);
                v = v + __shfl_down_sync(0xFFFFFFFFu, v, 1, 8);
                if (j == 0u && n_block + c < N) C[n_block + c] = __float2bfloat16(v);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(512, 2)
moe_expert_gate_up_worklist_t_lanes_rows_r2(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t_lanes_rows_staged_body<4, 16, 2>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(512, 1)
moe_expert_gate_up_worklist_t_lanes_rows_r3(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t_lanes_rows_staged_body<4, 16, 3>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(512, 1)
moe_expert_gate_up_worklist_t_lanes_rows_r4(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t_lanes_rows_staged_body<4, 16, 4>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(512, 1)
moe_expert_gate_up_worklist_t_lanes_rows_r5(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t_lanes_rows_staged_body<4, 16, 5>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(512, 1)
moe_expert_gate_up_worklist_t_lanes_rows_r8(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t_lanes_rows_staged_body<4, 16, 8>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(512, 1)
moe_expert_gate_up_worklist_t_lanes_rows_r10(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t_lanes_rows_staged_body<4, 16, 10>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(512, 2)
moe_expert_gate_up_worklist_t_lanes_rows_r2_ts4e(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t_lanes_rows_staged_body<4, 16, 2, true>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(512, 1)
moe_expert_gate_up_worklist_t_lanes_rows_r3_ts4e(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t_lanes_rows_staged_body<4, 16, 3, true>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(512, 1)
moe_expert_gate_up_worklist_t_lanes_rows_r4_ts4e(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t_lanes_rows_staged_body<4, 16, 4, true>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(512, 1)
moe_expert_gate_up_worklist_t_lanes_rows_r5_ts4e(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t_lanes_rows_staged_body<4, 16, 5, true>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(512, 1)
moe_expert_gate_up_worklist_t_lanes_rows_r8_ts4e(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t_lanes_rows_staged_body<4, 16, 8, true>(T_LANES_GU_WL_ARGS);
}

extern "C" __global__ void __launch_bounds__(512, 1)
moe_expert_gate_up_worklist_t_lanes_rows_r10_ts4e(T_LANES_GU_WL_PARAMS) {
    gate_up_worklist_t_lanes_rows_staged_body<4, 16, 10, true>(T_LANES_GU_WL_ARGS);
}
