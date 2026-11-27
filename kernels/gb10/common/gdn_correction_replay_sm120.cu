// SPDX-License-Identifier: AGPL-3.0-only
//
// Native Blackwell SM120/SM121 (with SM103a fallback) 1D GDN Correction-Record
// Verify & Device-Driven Cross-Layer Replay Commit Kernels.
//
// Design overview:
//   1. `gdn_verify_record_1d_sm120`:
//      - Reads FP32 `h_state` [B, num_v_heads, 128, 128] ONCE per head using
//        vectorized 128-bit streaming loads (`L2::evict_first` on SM120/SM121)
//        and NEVER mutates `h_state` in DRAM during verification.
//      - Executes exact unroll-4 single-accumulator FP32 GDN recurrence in
//        registers (`H_reg[128]`) for `t = 0 .. live_tokens - 1`.
//      - Supports explicit inline PTX `cvt.rn.bf16.f32` rounding boundaries for
//        `beta = sigmoid(raw_b)` and `o = (S q) * inv_sqrt_d` (before FP32
//        gated RMSNorm) so `fp_contract` / FMA fusion can never silently drift
//        verify vs prefill numerics.
//      - Emits compact 1D FP32 correction records `(delta[t], k[t], decay[t])`:
//          delta: [B * max_tokens, num_v_heads, 128]  (512 B / head / tok)
//          k:     [B * max_tokens, num_k_heads, 128]  (512 B / k_head / tok)
//          decay: [B * max_tokens, num_v_heads]       (4 B / head / tok)
//        giving 63.75x (MHA) to 95.4x (Qwen3.8 GQA 16:48) memory reduction vs
//        2D `h_state_intermediates` (65,536 B / head / tok).
//      - Explicitly zeros inactive tail rows `t in [live_tokens, max_tokens)`
//        in `output`, `delta_out`, `k_out`, and `decay_out`.
//
//   2. `gdn_commit_replay_1d_sm120`:
//      - Single cross-layer GPU `_commit` launch over grid
//        `(num_v_heads, batch_size, num_ssm_layers)`, block `(128, 1, 1)`.
//      - Reads device-resident `const unsigned int* d_num_accepted` (0 host
//        sync!).
//      - Validates fail-closed per-slot `GdnCommitSlotDesc` guards BEFORE
//        touching `h_state`: non-null pointers, `generation == expected_generation`,
//        `sharing_count <= 1`, `accepted <= max_tokens`, and `flags == 0`.
//        Any violation atomically records the fault bit in `status_out` and
//        returns immediately without modifying `h_state`.
//      - Fast-exits with 0 DRAM H traffic when `accepted == 0`.
//      - Replays `H_reg[j] = decay[t] * H_reg[j] + k[t, j] * delta[t, tid]`
//        for `t = 0 .. accepted - 1` in the IDENTICAL FP32 instruction order
//        as verify, guaranteeing 0-ULP bit-for-bit equality with 2D snapshots,
//        and writes updated `h_state` ONCE with `L2::evict_first`.
//
//   3. `gdn_zero_tail_rows_sm120`:
//      - Standalone explicit inactive tail-row zeroing `[live_tokens, max_tokens)`
//        across any non-null subset of `(output, delta_out, k_out, decay_out)`.
//
// Hardware invariants (proved by `repro-kit/formal/verify_ptx_isa_contracts.py`):
//   - 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads across
//     all 3 entrypoints on sm_121a, sm_121f, sm_120a, and sm_103a.
//   - 0 STL / LDL and 0 odd `desc[UR]` SASS instructions.

#include <cuda_bf16.h>
#include <math.h>

#define GDN_1D_DIM 128u

// Verify numeric-boundary control flags (combinable bitmask).
#define GDN_1D_FLAG_BETA_FROM_RAW_LOGIT 1u
#define GDN_1D_FLAG_ROUND_BETA_BF16     2u
#define GDN_1D_FLAG_ROUND_OUT_BF16      4u

// Fail-closed commit descriptor status bitmask written to `status_out`.
#define GDN_1D_STATUS_OK                0u
#define GDN_1D_STATUS_NULL_PTR          1u
#define GDN_1D_STATUS_GEN_MISMATCH      2u
#define GDN_1D_STATUS_SHARED_SLOT       4u
#define GDN_1D_STATUS_ACCEPTED_OOB      8u
#define GDN_1D_STATUS_BAD_GEOMETRY     16u

// Exact 1.0f / sqrtf(128.0f) IEEE-754 single-precision constant (0x3db504f3).
#define GDN_1D_INV_SQRT_128 0.08838834764831845f

// 64-byte ABI-pinned per-(layer, seq) commit descriptor shared with Rust
// `ops::gdn_correction::GdnCommitSlotDesc`.
struct alignas(8) GdnCommitSlotDesc {
    unsigned long long h_state_ptr;
    unsigned long long delta_ptr;
    unsigned long long k_ptr;
    unsigned long long decay_ptr;
    unsigned long long generation;
    unsigned long long expected_generation;
    unsigned int sharing_count;
    unsigned int max_tokens;
    unsigned int flags;
    unsigned int _pad;
};
static_assert(sizeof(GdnCommitSlotDesc) == 64, "GdnCommitSlotDesc must be 64 bytes");
static_assert(alignof(GdnCommitSlotDesc) == 8, "GdnCommitSlotDesc must be 8-byte aligned");

// Explicit inline PTX FP32 -> BF16 (round-to-nearest-even) -> FP32 boundary.
// `asm volatile` prevents `ptxas`/`nvcc` from contracting or folding across
// the BF16 rounding point into FP32 FMA.
__device__ __forceinline__ float gdn_1d_round_bf16_ptx(float x) {
    unsigned short bf16_bits;
    asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(bf16_bits) : "f"(x));
    return __uint_as_float(static_cast<unsigned int>(bf16_bits) << 16);
}

__device__ __forceinline__ unsigned long long gdn_1d_l2_evict_first_policy() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    return pol;
#else
    return 0ull;
#endif
}

__device__ __forceinline__ float4 gdn_1d_ld_global_na_ef_f4(
    const float4* ptr,
    unsigned long long pol_ef
) {
    float4 v;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    asm volatile(
        "ld.global.L1::no_allocate.L2::cache_hint.L2::256B.v4.f32 {%0, %1, %2, %3}, [%4], %5;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(ptr), "l"(pol_ef)
    );
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    // Clean SM103a / pre-SM120 fallback using streaming L2::evict_first cache hint.
    asm volatile(
        "ld.global.L1::no_allocate.L2::cache_hint.v4.f32 {%0, %1, %2, %3}, [%4], %5;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(ptr), "l"(pol_ef)
    );
#else
    (void)pol_ef;
    v = *ptr;
#endif
    return v;
}

__device__ __forceinline__ void gdn_1d_st_global_na_ef_f4(
    float4* ptr,
    float4 v,
    unsigned long long pol_ef
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "st.global.L1::no_allocate.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;"
        :: "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "l"(pol_ef)
        : "memory"
    );
#else
    (void)pol_ef;
    *ptr = v;
#endif
}

__device__ __forceinline__ float4 gdn_1d_lds_f4_volatile(const float* smem_ptr) {
    float4 v;
    const unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ld.volatile.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "r"(addr)
    );
    return v;
}

__device__ __forceinline__ float4 gdn_1d_lds_f4_volatile_f4(const float4* smem_ptr) {
    return gdn_1d_lds_f4_volatile(reinterpret_cast<const float*>(smem_ptr));
}

__device__ __forceinline__ void gdn_1d_sts_f4_volatile(float4* smem_ptr, float4 v) {
    const unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "st.volatile.shared.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "r"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
        : "memory"
    );
}

// Load [128 x 128] FP32 H into H_reg[128] via 32x 128-bit streaming loads
// (4-deep rolling window + warp-local 0-bank-conflict smem transpose,
// 0 __syncthreads(), 0 B spills).
__device__ __forceinline__ void gdn_1d_load_h_f4(
    const float* H,
    float H_reg[GDN_1D_DIM],
    float4 smem_h_tile[4][32]
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int r4 = lane >> 3;
    const unsigned int c4 = lane & 7u;
    const unsigned long long pol_ef = gdn_1d_l2_evict_first_policy();
    const float4* H4_base =
        reinterpret_cast<const float4*>(H) + static_cast<unsigned long long>(r4) * 32ull
        + warp * 8u + c4;

    float4 tmp[4];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        tmp[b] = gdn_1d_ld_global_na_ef_f4(
            H4_base + static_cast<unsigned long long>(b) * 128ull,
            pol_ef
        );
    }
    #pragma unroll
    for (int jb = 0; jb < 32; ++jb) {
        const int slot = jb & 3;
        gdn_1d_sts_f4_volatile(&smem_h_tile[warp][lane], tmp[slot]);
        __syncwarp();
        const float* wtile = reinterpret_cast<const float*>(smem_h_tile[warp]);
        H_reg[jb * 4 + 0] = wtile[0 * 32 + lane];
        H_reg[jb * 4 + 1] = wtile[1 * 32 + lane];
        H_reg[jb * 4 + 2] = wtile[2 * 32 + lane];
        H_reg[jb * 4 + 3] = wtile[3 * 32 + lane];
        __syncwarp();
        if (jb + 4 < 32) {
            tmp[slot] = gdn_1d_ld_global_na_ef_f4(
                H4_base + static_cast<unsigned long long>(jb + 4) * 128ull,
                pol_ef
            );
        }
    }
}

// Store H_reg[128] to [128 x 128] FP32 dst via warp-local smem transpose +
// 32x 128-bit streaming stores (L2::evict_first, 0 __syncthreads(), 0 B spills).
__device__ __forceinline__ void gdn_1d_store_h_f4(
    float* dst,
    const float H_reg[GDN_1D_DIM],
    float4 smem_h_tile[4][32]
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int r4 = lane >> 3;
    const unsigned int c4 = lane & 7u;
    const unsigned long long pol_ef = gdn_1d_l2_evict_first_policy();
    float4* dst4_base =
        reinterpret_cast<float4*>(dst) + static_cast<unsigned long long>(r4) * 32ull
        + warp * 8u + c4;

    #pragma unroll
    for (int jb = 0; jb < 32; ++jb) {
        float* wtile = reinterpret_cast<float*>(smem_h_tile[warp]);
        wtile[0 * 32 + lane] = H_reg[jb * 4 + 0];
        wtile[1 * 32 + lane] = H_reg[jb * 4 + 1];
        wtile[2 * 32 + lane] = H_reg[jb * 4 + 2];
        wtile[3 * 32 + lane] = H_reg[jb * 4 + 3];
        __syncwarp();
        const float4 v = gdn_1d_lds_f4_volatile_f4(&smem_h_tile[warp][lane]);
        gdn_1d_st_global_na_ef_f4(
            dst4_base + static_cast<unsigned long long>(jb) * 128ull,
            v,
            pol_ef
        );
        __syncwarp();
    }
}

// Exact FP32 rank-1 state update shared by both verify and commit replay.
// Explicit `__fmul_rn` / `__fadd_rn` forbids FMA contraction regardless of host
// compiler flags so verify and replay produce identical 32-bit IEEE patterns.
__device__ __forceinline__ void gdn_1d_rank1_update_col(
    float H_reg[GDN_1D_DIM],
    const float* smem_k,
    float decay,
    float delta_i
) {
    #pragma unroll
    for (int j = 0; j < static_cast<int>(GDN_1D_DIM); j += 4) {
        const float4 kv = gdn_1d_lds_f4_volatile(&smem_k[j]);
        H_reg[j + 0] = __fadd_rn(__fmul_rn(decay, H_reg[j + 0]), __fmul_rn(kv.x, delta_i));
        H_reg[j + 1] = __fadd_rn(__fmul_rn(decay, H_reg[j + 1]), __fmul_rn(kv.y, delta_i));
        H_reg[j + 2] = __fadd_rn(__fmul_rn(decay, H_reg[j + 2]), __fmul_rn(kv.z, delta_i));
        H_reg[j + 3] = __fadd_rn(__fmul_rn(decay, H_reg[j + 3]), __fmul_rn(kv.w, delta_i));
    }
}

// Exact unroll-4 single-accumulator dot product `sum_{j=0..127} H_reg[j] * v[j]`.
__device__ __forceinline__ float gdn_1d_col_dot(
    const float H_reg[GDN_1D_DIM],
    const float* smem_vec
) {
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < static_cast<int>(GDN_1D_DIM); j += 4) {
        const float4 vv = gdn_1d_lds_f4_volatile(&smem_vec[j]);
        const float p0 = __fmul_rn(H_reg[j + 0], vv.x);
        const float p1 = __fmul_rn(H_reg[j + 1], vv.y);
        const float p2 = __fmul_rn(H_reg[j + 2], vv.z);
        const float p3 = __fmul_rn(H_reg[j + 3], vv.w);
        const float s01 = __fadd_rn(p0, p1);
        const float s012 = __fadd_rn(s01, p2);
        const float s0123 = __fadd_rn(s012, p3);
        acc = __fadd_rn(acc, s0123);
    }
    return acc;
}

// Entrypoint 1: Read-only verify + 1D correction-record emission + tail zeroing.
// Grid: (num_v_heads, batch_size, 1), Block: (128, 1, 1).
extern "C" __global__ __launch_bounds__(128, 2) void gdn_verify_record_1d_sm120(
    const float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ delta_out,
    float* __restrict__ k_out,
    float* __restrict__ decay_out,
    unsigned int batch_size,
    unsigned int live_tokens,
    unsigned int max_tokens,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int out_stride,
    unsigned int flags
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int seq = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (vh >= num_v_heads || seq >= batch_size || tid >= GDN_1D_DIM) {
        return;
    }
    if (num_k_heads == 0u || num_v_heads == 0u || (num_v_heads % num_k_heads) != 0u) {
        return;
    }
    if (h_state == nullptr || query == nullptr || key == nullptr || value == nullptr
        || gate == nullptr || beta == nullptr || output == nullptr
        || delta_out == nullptr || k_out == nullptr || decay_out == nullptr
        || ((reinterpret_cast<unsigned long long>(h_state) & 15ull) != 0ull)) {
        return;
    }

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const unsigned int sub = vh - kh * head_repeat;
    const unsigned int active_tokens = (live_tokens < max_tokens) ? live_tokens : max_tokens;
    const unsigned long long seq_row_base =
        static_cast<unsigned long long>(seq) * static_cast<unsigned long long>(max_tokens);

    __shared__ float4 smem_h_tile[4][32];
    __shared__ alignas(16) float smem_k[GDN_1D_DIM];
    __shared__ alignas(16) float smem_q[GDN_1D_DIM];

    if (active_tokens > 0u) {
        float H_reg[GDN_1D_DIM];
        const unsigned long long h_head_off =
            (static_cast<unsigned long long>(seq) * num_v_heads + vh)
            * static_cast<unsigned long long>(GDN_1D_DIM * GDN_1D_DIM);
        gdn_1d_load_h_f4(h_state + h_head_off, H_reg, smem_h_tile);

        const bool beta_from_logit = (flags & GDN_1D_FLAG_BETA_FROM_RAW_LOGIT) != 0u;
        const bool round_beta_bf16 = (flags & GDN_1D_FLAG_ROUND_BETA_BF16) != 0u;
        const bool round_out_bf16 = (flags & GDN_1D_FLAG_ROUND_OUT_BF16) != 0u;

        #pragma unroll 1
        for (unsigned int t = 0u; t < active_tokens; ++t) {
            const unsigned long long row = seq_row_base + static_cast<unsigned long long>(t);
            const unsigned long long qk_off =
                row * static_cast<unsigned long long>(qk_stride)
                + static_cast<unsigned long long>(kh) * GDN_1D_DIM + tid;
            const unsigned long long v_off =
                row * static_cast<unsigned long long>(v_stride)
                + static_cast<unsigned long long>(vh) * GDN_1D_DIM + tid;
            const unsigned long long gb_off =
                row * static_cast<unsigned long long>(gb_stride) + vh;

            const float k_val = __ldg(&key[qk_off]);
            const float q_val = __ldg(&query[qk_off]);
            const float v_val = __ldg(&value[v_off]);
            smem_k[tid] = k_val;
            smem_q[tid] = q_val;

            const float g_raw = __ldg(&gate[gb_off]);
            const float decay = fminf(fmaxf(g_raw, 0.0f), 1.0f);
            float bt = __ldg(&beta[gb_off]);
            if (beta_from_logit) {
                bt = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-bt)));
            }
            if (round_beta_bf16) {
                bt = gdn_1d_round_bf16_ptx(bt);
            }
            __syncthreads();

            const float hk_dot = gdn_1d_col_dot(H_reg, smem_k);
            const float delta_i = __fmul_rn(__fsub_rn(v_val, __fmul_rn(decay, hk_dot)), bt);
            gdn_1d_rank1_update_col(H_reg, smem_k, decay, delta_i);
            const float q_dot = gdn_1d_col_dot(H_reg, smem_q);

            float out_val = __fmul_rn(q_dot, GDN_1D_INV_SQRT_128);
            if (round_out_bf16) {
                out_val = gdn_1d_round_bf16_ptx(out_val);
            }

            output[row * static_cast<unsigned long long>(out_stride)
                   + static_cast<unsigned long long>(vh) * GDN_1D_DIM + tid] = out_val;
            delta_out[(row * num_v_heads + vh) * GDN_1D_DIM + tid] = delta_i;
            if (sub == 0u) {
                k_out[(row * num_k_heads + kh) * GDN_1D_DIM + tid] = k_val;
            }
            if (tid == 0u) {
                decay_out[row * num_v_heads + vh] = decay;
            }
            __syncthreads();
        }
    }

    // Zero inactive tail rows [active_tokens, max_tokens).
    #pragma unroll 1
    for (unsigned int t = active_tokens; t < max_tokens; ++t) {
        const unsigned long long row = seq_row_base + static_cast<unsigned long long>(t);
        output[row * static_cast<unsigned long long>(out_stride)
               + static_cast<unsigned long long>(vh) * GDN_1D_DIM + tid] = 0.0f;
        delta_out[(row * num_v_heads + vh) * GDN_1D_DIM + tid] = 0.0f;
        if (sub == 0u) {
            k_out[(row * num_k_heads + kh) * GDN_1D_DIM + tid] = 0.0f;
        }
        if (tid == 0u) {
            decay_out[row * num_v_heads + vh] = 0.0f;
        }
    }
}

// Entrypoint 2: Device-driven cross-layer 1D correction-record replay commit.
// Grid: (num_v_heads, batch_size, num_ssm_layers), Block: (128, 1, 1).
extern "C" __global__ __launch_bounds__(128, 2) void gdn_commit_replay_1d_sm120(
    const GdnCommitSlotDesc* __restrict__ descs,
    const unsigned int* __restrict__ d_num_accepted,
    unsigned int* __restrict__ status_out,
    unsigned int num_ssm_layers,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int seq = blockIdx.y;
    const unsigned int layer = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    if (vh >= num_v_heads || seq >= batch_size || layer >= num_ssm_layers || tid >= GDN_1D_DIM) {
        return;
    }
    if (num_k_heads == 0u || num_v_heads == 0u || (num_v_heads % num_k_heads) != 0u) {
        if (tid == 0u && status_out != nullptr) {
            atomicOr(status_out, GDN_1D_STATUS_BAD_GEOMETRY);
        }
        return;
    }
    if (descs == nullptr || d_num_accepted == nullptr) {
        if (tid == 0u && status_out != nullptr) {
            atomicOr(status_out, GDN_1D_STATUS_NULL_PTR);
        }
        return;
    }

    __shared__ GdnCommitSlotDesc smem_desc;
    __shared__ unsigned int smem_accepted;
    __shared__ unsigned int smem_err_bits;
    __shared__ float4 smem_h_tile[4][32];
    __shared__ alignas(16) float smem_k[GDN_1D_DIM];

    if (tid == 0u) {
        const unsigned long long desc_idx =
            static_cast<unsigned long long>(layer) * static_cast<unsigned long long>(batch_size)
            + static_cast<unsigned long long>(seq);
        const GdnCommitSlotDesc d = descs[desc_idx];
        const unsigned int accepted = __ldg(&d_num_accepted[seq]);
        unsigned int err = GDN_1D_STATUS_OK;
        if (d.h_state_ptr == 0ull || d.delta_ptr == 0ull
            || d.k_ptr == 0ull || d.decay_ptr == 0ull) {
            err |= GDN_1D_STATUS_NULL_PTR;
        }
        if ((d.h_state_ptr & 15ull) != 0ull
            || (d.delta_ptr & 3ull) != 0ull
            || (d.k_ptr & 3ull) != 0ull
            || (d.decay_ptr & 3ull) != 0ull) {
            err |= GDN_1D_STATUS_BAD_GEOMETRY;
        }
        if (d.generation != d.expected_generation) {
            err |= GDN_1D_STATUS_GEN_MISMATCH;
        }
        if (d.sharing_count > 1u) {
            err |= GDN_1D_STATUS_SHARED_SLOT;
        }
        if (accepted > d.max_tokens) {
            err |= GDN_1D_STATUS_ACCEPTED_OOB;
        }
        if (d.flags != 0u || d._pad != 0u) {
            err |= GDN_1D_STATUS_BAD_GEOMETRY;
        }
        if (err != GDN_1D_STATUS_OK && status_out != nullptr) {
            atomicOr(status_out, err);
        }
        smem_desc = d;
        smem_accepted = accepted;
        smem_err_bits = err;
    }
    __syncthreads();

    if (smem_err_bits != GDN_1D_STATUS_OK || smem_accepted == 0u) {
        return;
    }

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const unsigned int accepted = smem_accepted;

    float* h_head = reinterpret_cast<float*>(smem_desc.h_state_ptr)
        + static_cast<unsigned long long>(vh) * static_cast<unsigned long long>(GDN_1D_DIM * GDN_1D_DIM);
    const float* delta_base = reinterpret_cast<const float*>(smem_desc.delta_ptr);
    const float* k_base = reinterpret_cast<const float*>(smem_desc.k_ptr);
    const float* decay_base = reinterpret_cast<const float*>(smem_desc.decay_ptr);

    float H_reg[GDN_1D_DIM];
    gdn_1d_load_h_f4(h_head, H_reg, smem_h_tile);

    #pragma unroll 1
    for (unsigned int t = 0u; t < accepted; ++t) {
        const unsigned long long t_ull = static_cast<unsigned long long>(t);
        smem_k[tid] = __ldg(&k_base[(t_ull * num_k_heads + kh) * GDN_1D_DIM + tid]);
        const float delta_i = __ldg(&delta_base[(t_ull * num_v_heads + vh) * GDN_1D_DIM + tid]);
        const float decay = __ldg(&decay_base[t_ull * num_v_heads + vh]);
        __syncthreads();

        gdn_1d_rank1_update_col(H_reg, smem_k, decay, delta_i);
        __syncthreads();
    }

    gdn_1d_store_h_f4(h_head, H_reg, smem_h_tile);
}

// Entrypoint 3: Explicit inactive tail-row zeroing `[live_tokens, max_tokens)`.
// Grid: (num_v_heads, batch_size, 1), Block: (128, 1, 1).
extern "C" __global__ __launch_bounds__(128, 2) void gdn_zero_tail_rows_sm120(
    float* __restrict__ output,
    float* __restrict__ delta_out,
    float* __restrict__ k_out,
    float* __restrict__ decay_out,
    unsigned int batch_size,
    unsigned int live_tokens,
    unsigned int max_tokens,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int out_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int seq = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (vh >= num_v_heads || seq >= batch_size || tid >= GDN_1D_DIM) {
        return;
    }
    if (live_tokens >= max_tokens) {
        return;
    }
    if (num_k_heads == 0u || num_v_heads == 0u || (num_v_heads % num_k_heads) != 0u) {
        return;
    }

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const unsigned int sub = vh - kh * head_repeat;
    const unsigned long long seq_row_base =
        static_cast<unsigned long long>(seq) * static_cast<unsigned long long>(max_tokens);

    #pragma unroll 1
    for (unsigned int t = live_tokens; t < max_tokens; ++t) {
        const unsigned long long row = seq_row_base + static_cast<unsigned long long>(t);
        if (output != nullptr) {
            output[row * static_cast<unsigned long long>(out_stride)
                   + static_cast<unsigned long long>(vh) * GDN_1D_DIM + tid] = 0.0f;
        }
        if (delta_out != nullptr) {
            delta_out[(row * num_v_heads + vh) * GDN_1D_DIM + tid] = 0.0f;
        }
        if (k_out != nullptr && sub == 0u) {
            k_out[(row * num_k_heads + kh) * GDN_1D_DIM + tid] = 0.0f;
        }
        if (decay_out != nullptr && tid == 0u) {
            decay_out[row * num_v_heads + vh] = 0.0f;
        }
    }
}
