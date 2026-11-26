// SPDX-License-Identifier: AGPL-3.0-only
//
// Fused 1x1x1-CTA Qwen4 GDN recurrent core (conv1d + SiLU + Q/K L2-norm +
// register-resident GDN recurrence + per-head sigmoid gated RMS norm) for
// sm_121a / sm_121f GB10 (48 value heads = 48 CTAs/seq; multi-seq launches
// [48, num_seqs, 1] at 3 CTAs/SM = 144 resident CTAs across GB10's 48 SMs).
//
// Replaces 3 separate kernel launches per GDN layer (108 launches across 36
// layers) and eliminates global round-trips through `ssm_conv_out_f32` and
// `gdn_rows` while preserving 0-ULP IEEE-754 parity with:
//   1. `causal_conv1d_update_l2norm_f32[_sequence]` (`causal_conv1d.cu`)
//   2. `gated_delta_rule_decode_f32[_sequence_{persistent,lazyfinal}]`
//   3. `qwen4_gated_rms_norm_sigmoid_f32[_multi_seq]` (`qwen4_hyper.cu`)
//
// GB10 memory-bandwidth optimizations (all strictly 0-ULP & 0 B spill/stack):
//   - Vectorized 128-bit streaming H load/store (`ld.global.L1::no_allocate
//     .L2::cache_hint.L2::256B.v4.f32` / `st.global.L1::no_allocate
//     .L2::cache_hint.v4.f32` with `L2::evict_first`) + 4-deep rolling MLP
//     (`16` regs, `8 KiB` in flight/CTA = `1.15 MiB` across GB10) +
//     zero-bank-conflict warp-local smem transpose (0 `__syncthreads()`,
//     1 `__syncwarp()`/strip).
//   - CTA-uniform head geometry (`head_repeat`, `kh`, `sub`, channel bases)
//     and multi-seq descriptors staged in shared memory so `ptxas` maps them
//     to uniform registers (`UR`, 0 VGPRs), paired with serialized Q -> K -> V
//     conv1d live ranges (`ld.volatile.shared.v4.f32`) so peak VGPR count stays
//     <= 168 regs/thread (`__launch_bounds__(128, 3)` = 3 CTAs/SM) with
//     0 B stack / 0 B spills on every entrypoint.
//   - 128-bit `ld.volatile.shared.v4.f32` (`LDS.128`) in GDN inner loops
//     (4x fewer LSU ops without hoisting register pressure), double-buffered
//     `smem_k[2][128]`, coalesced `vh == 0` warp gate writes, and barrier-free
//     exact XOR-tree warp reduction in Step 4 RMS norm.
//   - Staggered multi-step compact circular ring `KEEP_LATE`:
//       * Stores only compact raw BF16 `[Q, K, V]` + FP32 `[raw_g, beta]`
//         (`20,864 B/row` on TP1, 3x smaller than FP32 retain) with zero
//         `conv_inter` writes.
//       * Replays `pending_rows` compact rows in registers (advancing
//         `conv_state` + inline `--fmad=false` FP32 conv1d + SiLU + K L2-norm +
//         GDN rank-1 update).
//       * When `flush_now == 0`, skips writing `h_state`, skips writing
//         `conv_state`, and skips `atomicInc` + barriers; when `flush_now == 1`,
//         commits `conv_state` and `h_state` to DRAM. Also supports `num_tokens == 0`
//         cold flush with identical 0-ULP machine instructions.
//       * Optional in-place per-head Scaled-FP16 `h_state` (`LOAD_SCALED_FP16` /
//         `STORE_SCALED_FP16` descriptor flags): stores `128x128` row-major
//         `__half` at bytes `0..32768` + `float` scale at byte offset `32768`
//         inside each 64 KiB head slot (0 B extra VRAM, 2x less H DRAM traffic),
//         while keeping all token updates in full FP32 `H_reg[128]`.
//
// Race-free GQA conv_state update (head_repeat = num_v_heads / num_k_heads):
//   - Every CTA `vh` reads initial Q[kh] and K[kh] conv_state before advancing
//     its local window across `pending_rows` / `num_tokens`.
//   - Intermediate conv snapshots (`conv_inter`) and retain/ring rows are
//     write-only, so only `sub == 0` (`vh % head_repeat == 0`) writes Q[kh]/K[kh]
//     while every `vh` writes V[vh].
//   - Live `conv_state` for Q[kh]/K[kh] is written ONLY by the last CTA in
//     key-head group `kh` to finish, coordinated via a self-resetting GPU
//     release-acquire `atomicInc(&group_done[kh], head_repeat - 1)` counter
//     (wraps back to 0 automatically on the `head_repeat`-th arrival, so zero
//     host memset is needed and CUDA graph capture/replay is 100% safe).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define QWEN4_GDN_DIM 128
#define QWEN4_GDN_LAZY_MAX_K 16u
#define GDN_FUSED_FLAG_COPY_GATES_TO_RETAIN 1u
#define GDN_FUSED_FLAG_KEEP_LATE 2u
#define GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR 4u
#define GDN_FUSED_FLAG_LOAD_SCALED_FP16 16u
#define GDN_FUSED_FLAG_STORE_SCALED_FP16 32u
#define GDN_FUSED_FLAG_LOAD_SCALED_FP8 64u
#define GDN_FUSED_FLAG_STORE_SCALED_FP8 128u
#define QWEN4_GDN_FP8_SCALES_OFFSET 16384u

struct alignas(16) GdnKeepLateCtrl {
    unsigned long long ring_base;
    unsigned int pending_base;
    unsigned int pending_rows;
    unsigned int write_base;
    unsigned int executed_flag;
    unsigned int ring_cap;
    unsigned int flush_now;
};
static_assert(sizeof(GdnKeepLateCtrl) == 32, "GdnKeepLateCtrl must be 32 bytes");

struct alignas(16) GdnFusedMultiSeqDesc {
    unsigned long long conv_state;
    unsigned long long h_state;
    unsigned long long retain_qkv;
    unsigned long long conv_inter;
    // In normal lazy/persistent mode: H snapshot/final write target.
    // When GDN_FUSED_FLAG_KEEP_LATE | GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR is set:
    // constant per-slot pointer to `GdnKeepLateCtrl` holding per-step circular
    // ring coordinates (`ring_base`, `pending_base`, `pending_rows`,
    // `write_base`, `ring_cap`, `flush_now`) — 100% CUDA-graph safe.
    unsigned long long h_state_inter;
    unsigned long long group_done;
    unsigned int row_base;
    unsigned int num_tokens;
    unsigned int flags;
    unsigned int _pad;
};
static_assert(sizeof(GdnFusedMultiSeqDesc) == 64, "GdnFusedMultiSeqDesc must be 64 bytes");

__device__ __forceinline__ unsigned long long qwen4_gdn_l2_evict_first_policy() {
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    return pol;
}

__device__ __forceinline__ float4 qwen4_gdn_ld_global_na_ef_f4(
    const float4* ptr,
    unsigned long long pol_ef
) {
    float4 v;
    asm volatile(
        "ld.global.L1::no_allocate.L2::cache_hint.L2::256B.v4.f32 {%0, %1, %2, %3}, [%4], %5;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(ptr), "l"(pol_ef)
    );
    return v;
}

__device__ __forceinline__ void qwen4_gdn_st_global_na_ef_f4(
    float4* ptr,
    float4 v,
    unsigned long long pol_ef
) {
    asm volatile(
        "st.global.L1::no_allocate.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;"
        :: "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "l"(pol_ef)
        : "memory"
    );
}

__device__ __forceinline__ void qwen4_gdn_st_global_cs_f4(float4* ptr, float4 v) {
    asm volatile(
        "st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
        : "memory"
    );
}

__device__ __forceinline__ void qwen4_gdn_st_global_cs_f32(float* ptr, float v) {
    asm volatile(
        "st.global.cs.f32 [%0], %1;"
        :: "l"(ptr), "f"(v)
        : "memory"
    );
}

__device__ __forceinline__ void qwen4_gdn_st_global_f32(float* ptr, float v) {
    asm volatile(
        "st.global.f32 [%0], %1;"
        :: "l"(ptr), "f"(v)
        : "memory"
    );
}

__device__ __forceinline__ void qwen4_gdn_st_global_u16(__nv_bfloat16* ptr, __nv_bfloat16 v) {
    unsigned short raw = *reinterpret_cast<const unsigned short*>(&v);
    asm volatile(
        "st.global.u16 [%0], %1;"
        :: "l"(ptr), "h"(raw)
        : "memory"
    );
}

// 128-bit volatile shared load (`LDS.128`) prevents ptxas from hoisting
// vectors live simultaneously across the GDN/conv loops (keeping 0 B spills).
__device__ __forceinline__ float4 qwen4_gdn_lds_f4_volatile(const float* smem_ptr) {
    float4 v;
    unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ld.volatile.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "r"(addr)
    );
    return v;
}

__device__ __forceinline__ float4 qwen4_gdn_lds_f4_volatile_f4(const float4* smem_ptr) {
    return qwen4_gdn_lds_f4_volatile(reinterpret_cast<const float*>(smem_ptr));
}

__device__ __forceinline__ void qwen4_gdn_sts_f4_volatile(float4* smem_ptr, float4 v) {
    unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "st.volatile.shared.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "r"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
        : "memory"
    );
}

__device__ __forceinline__ uint4 qwen4_gdn_ld_global_na_ef_u4(
    const uint4* ptr,
    unsigned long long pol_ef
) {
    uint4 v;
    asm volatile(
        "ld.global.L1::no_allocate.L2::cache_hint.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(ptr), "l"(pol_ef)
    );
    return v;
}

__device__ __forceinline__ void qwen4_gdn_st_global_na_ef_u4(
    uint4* ptr,
    uint4 v,
    unsigned long long pol_ef
) {
    asm volatile(
        "st.global.L1::no_allocate.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;"
        :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(pol_ef)
        : "memory"
    );
}

__device__ __forceinline__ void qwen4_gdn_sts_u4_volatile(float4* smem_ptr, uint4 v) {
    unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "st.volatile.shared.v4.u32 [%0], {%1, %2, %3, %4};"
        :: "r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
        : "memory"
    );
}

__device__ __forceinline__ uint4 qwen4_gdn_lds_u4_volatile(const float4* smem_ptr) {
    uint4 v;
    unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "r"(addr)
    );
    return v;
}

__device__ __forceinline__ uint2 qwen4_gdn_ld_global_nc_u32x2_volatile(const uint2* ptr) {
    uint2 v;
    asm volatile(
        "ld.global.nc.v2.u32 {%0, %1}, [%2];"
        : "=r"(v.x), "=r"(v.y)
        : "l"(ptr)
    );
    return v;
}

__device__ __forceinline__ float qwen4_gdn_warp_sum_xor(float v) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, d);
    }
    return v;
}

__device__ __forceinline__ unsigned int qwen4_gdn_group_arrive_last(
    unsigned int* addr,
    unsigned int wrap
) {
    unsigned int old;
    asm volatile(
        "atom.global.acq_rel.gpu.inc.u32 %0, [%1], %2;"
        : "=r"(old)
        : "l"(addr), "r"(wrap)
        : "memory"
    );
    return old;
}

// Load [128 x 128] FP32 H into H_reg[128] via 32x 128-bit streaming loads
// (4-deep rolling window = 16 regs, 8 KiB/CTA = 1.15 MiB in flight across
// GB10's 48 SMs at 3 CTAs/SM). Each 4x32 strip is transposed warp-locally in
// shared memory with 0 bank conflicts and 0 __syncthreads().
__device__ __forceinline__ void qwen4_gdn_load_h_f4(
    const float* H,
    float H_reg[QWEN4_GDN_DIM],
    float4 smem_h_tile[4][32]
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int r4 = lane >> 3;
    const unsigned int c4 = lane & 7u;
    const unsigned long long pol_ef = qwen4_gdn_l2_evict_first_policy();
    const float4* H4_base =
        reinterpret_cast<const float4*>(H) + (unsigned long long)r4 * 32ull + warp * 8u + c4;

    float4 tmp[4];
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        tmp[b] = qwen4_gdn_ld_global_na_ef_f4(
            H4_base + (unsigned long long)b * 128ull, pol_ef
        );
    }
    #pragma unroll
    for (int jb = 0; jb < 32; jb++) {
        const int slot = jb & 3;
        qwen4_gdn_sts_f4_volatile(&smem_h_tile[warp][lane], tmp[slot]);
        __syncwarp();
        const float* wtile = reinterpret_cast<const float*>(smem_h_tile[warp]);
        H_reg[jb * 4 + 0] = wtile[0 * 32 + lane];
        H_reg[jb * 4 + 1] = wtile[1 * 32 + lane];
        H_reg[jb * 4 + 2] = wtile[2 * 32 + lane];
        H_reg[jb * 4 + 3] = wtile[3 * 32 + lane];
        __syncwarp();
        if (jb + 4 < 32) {
            tmp[slot] = qwen4_gdn_ld_global_na_ef_f4(
                H4_base + (unsigned long long)(jb + 4) * 128ull, pol_ef
            );
        }
    }
}

// Store H_reg[128] to [128 x 128] FP32 dst via warp-local smem transpose +
// 32x 128-bit L1-no-allocate / L2-evict-first streaming stores (0 __syncthreads()).
__device__ __forceinline__ void qwen4_gdn_store_h_f4(
    float* dst,
    const float H_reg[QWEN4_GDN_DIM],
    float4 smem_h_tile[4][32]
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int r4 = lane >> 3;
    const unsigned int c4 = lane & 7u;
    const unsigned long long pol_ef = qwen4_gdn_l2_evict_first_policy();
    float4* dst4_base =
        reinterpret_cast<float4*>(dst) + (unsigned long long)r4 * 32ull + warp * 8u + c4;

    #pragma unroll
    for (int jb = 0; jb < 32; jb++) {
        float* wtile = reinterpret_cast<float*>(smem_h_tile[warp]);
        wtile[0 * 32 + lane] = H_reg[jb * 4 + 0];
        wtile[1 * 32 + lane] = H_reg[jb * 4 + 1];
        wtile[2 * 32 + lane] = H_reg[jb * 4 + 2];
        wtile[3 * 32 + lane] = H_reg[jb * 4 + 3];
        __syncwarp();
        float4 v = qwen4_gdn_lds_f4_volatile_f4(&smem_h_tile[warp][lane]);
        qwen4_gdn_st_global_na_ef_f4(dst4_base + (unsigned long long)jb * 128ull, v, pol_ef);
        __syncwarp();
    }
}

// Load in-place per-head Scaled-FP16 [128 x 128] H (`32,768 B` row-major __half
// at bytes 0..32768 + FP32 scale at byte offset 32768) into FP32 H_reg[128] via
// 16x 128-bit streaming uint4 loads (4-deep rolling window, 50% DRAM traffic,
// 0 __syncthreads(), 0 B spills).
__device__ __forceinline__ void qwen4_gdn_load_h_scaled_fp16(
    const float* H,
    float H_reg[QWEN4_GDN_DIM],
    float4 smem_h_tile[4][32]
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int r8 = lane >> 2;
    const unsigned int c8 = lane & 3u;
    const float scale = __ldg(
        reinterpret_cast<const float*>(
            reinterpret_cast<const unsigned char*>(H) + 32768u
        )
    );
    const unsigned long long pol_ef = qwen4_gdn_l2_evict_first_policy();
    const uint4* H8_base =
        reinterpret_cast<const uint4*>(H) + (unsigned long long)r8 * 16ull + warp * 4u + c8;

    uint4 tmp[4];
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        tmp[b] = qwen4_gdn_ld_global_na_ef_u4(
            H8_base + (unsigned long long)b * 128ull, pol_ef
        );
    }
    #pragma unroll
    for (int jb = 0; jb < 16; jb++) {
        const int slot = jb & 3;
        qwen4_gdn_sts_u4_volatile(&smem_h_tile[warp][lane], tmp[slot]);
        __syncwarp();
        const __half* wtile = reinterpret_cast<const __half*>(smem_h_tile[warp]);
        H_reg[jb * 8 + 0] = __half2float(wtile[0 * 32 + lane]) * scale;
        H_reg[jb * 8 + 1] = __half2float(wtile[1 * 32 + lane]) * scale;
        H_reg[jb * 8 + 2] = __half2float(wtile[2 * 32 + lane]) * scale;
        H_reg[jb * 8 + 3] = __half2float(wtile[3 * 32 + lane]) * scale;
        H_reg[jb * 8 + 4] = __half2float(wtile[4 * 32 + lane]) * scale;
        H_reg[jb * 8 + 5] = __half2float(wtile[5 * 32 + lane]) * scale;
        H_reg[jb * 8 + 6] = __half2float(wtile[6 * 32 + lane]) * scale;
        H_reg[jb * 8 + 7] = __half2float(wtile[7 * 32 + lane]) * scale;
        __syncwarp();
        if (jb + 4 < 16) {
            tmp[slot] = qwen4_gdn_ld_global_na_ef_u4(
                H8_base + (unsigned long long)(jb + 4) * 128ull, pol_ef
            );
        }
    }
}

// Store FP32 H_reg[128] in-place as per-head Scaled-FP16 (`32,768 B` row-major
// __half at bytes 0..32768 + FP32 scale at byte offset 32768) via CTA-wide amax
// reduction and 16x 128-bit streaming uint4 stores (50% DRAM traffic, 0 B spills).
__device__ __forceinline__ void qwen4_gdn_store_h_scaled_fp16(
    float* dst,
    const float H_reg[QWEN4_GDN_DIM],
    float4 smem_h_tile[4][32],
    float warp_max_smem[4]
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int r8 = lane >> 2;
    const unsigned int c8 = lane & 3u;

    float local_max = 0.0f;
    #pragma unroll
    for (int j = 0; j < QWEN4_GDN_DIM; j += 4) {
        local_max = fmaxf(local_max, fabsf(H_reg[j]));
        local_max = fmaxf(local_max, fabsf(H_reg[j + 1]));
        local_max = fmaxf(local_max, fabsf(H_reg[j + 2]));
        local_max = fmaxf(local_max, fabsf(H_reg[j + 3]));
    }
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, d));
    }
    if (lane == 0u) {
        warp_max_smem[warp] = local_max;
    }
    __syncthreads();

    const float head_amax = fmaxf(
        fmaxf(warp_max_smem[0], warp_max_smem[1]),
        fmaxf(warp_max_smem[2], warp_max_smem[3])
    );
    const float scale = fmaxf(head_amax / 65000.0f, 1.0e-30f);
    const float inv_scale = 1.0f / scale;
    if (tid == 0u) {
        float* scale_ptr = reinterpret_cast<float*>(
            reinterpret_cast<unsigned char*>(dst) + 32768u
        );
        qwen4_gdn_st_global_f32(scale_ptr, scale);
    }

    const unsigned long long pol_ef = qwen4_gdn_l2_evict_first_policy();
    uint4* dst8_base =
        reinterpret_cast<uint4*>(dst) + (unsigned long long)r8 * 16ull + warp * 4u + c8;
    __half* wtile = reinterpret_cast<__half*>(smem_h_tile[warp]);

    #pragma unroll
    for (int jb = 0; jb < 16; jb++) {
        wtile[0 * 32 + lane] = __float2half_rn(H_reg[jb * 8 + 0] * inv_scale);
        wtile[1 * 32 + lane] = __float2half_rn(H_reg[jb * 8 + 1] * inv_scale);
        wtile[2 * 32 + lane] = __float2half_rn(H_reg[jb * 8 + 2] * inv_scale);
        wtile[3 * 32 + lane] = __float2half_rn(H_reg[jb * 8 + 3] * inv_scale);
        wtile[4 * 32 + lane] = __float2half_rn(H_reg[jb * 8 + 4] * inv_scale);
        wtile[5 * 32 + lane] = __float2half_rn(H_reg[jb * 8 + 5] * inv_scale);
        wtile[6 * 32 + lane] = __float2half_rn(H_reg[jb * 8 + 6] * inv_scale);
        wtile[7 * 32 + lane] = __float2half_rn(H_reg[jb * 8 + 7] * inv_scale);
        __syncwarp();
        uint4 v = qwen4_gdn_lds_u4_volatile(&smem_h_tile[warp][lane]);
        qwen4_gdn_st_global_na_ef_u4(dst8_base + (unsigned long long)jb * 128ull, v, pol_ef);
        __syncwarp();
    }
}

__device__ __forceinline__ void qwen4_gdn_st_global_half(__half* ptr, __half v) {
    unsigned short raw = *reinterpret_cast<const unsigned short*>(&v);
    asm volatile(
        "st.global.u16 [%0], %1;"
        :: "l"(ptr), "h"(raw)
        : "memory"
    );
}

__device__ __forceinline__ unsigned short qwen4_gdn_cvt_f32x2_to_e4m3x2(float lo, float hi) {
    unsigned short out;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(out) : "f"(lo), "f"(hi));
    return out;
}

__device__ __forceinline__ float2 qwen4_gdn_cvt_e4m3x2_to_f32x2(unsigned char lo, unsigned char hi) {
    const unsigned short pair =
        static_cast<unsigned short>(lo) | (static_cast<unsigned short>(hi) << 8);
    unsigned int h2_bits;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2_bits) : "h"(pair));
    return __half22float2(*reinterpret_cast<const __half2*>(&h2_bits));
}

// Load in-place per-value-channel (128-element) Scaled-FP8 E4M3 [128 x 128] H
// (`16,384 B` row-major E4M3 at bytes 0..16384 + `128 x __half = 256 B` scales
// at byte offset 16384 = `16.25 KiB/head`, 0 B extra VRAM, 50% DRAM traffic vs
// Scaled-FP16) into full FP32 `H_reg[128]` via 8x 128-bit streaming uint4 loads.
__device__ __forceinline__ void qwen4_gdn_load_h_scaled_fp8(
    const float* H,
    float H_reg[QWEN4_GDN_DIM],
    float4 smem_h_tile[4][32]
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int r16 = lane >> 1;
    const unsigned int c16 = lane & 1u;
    const __half* scales = reinterpret_cast<const __half*>(
        reinterpret_cast<const unsigned char*>(H) + QWEN4_GDN_FP8_SCALES_OFFSET
    );
    const float scale = __half2float(__ldg(&scales[tid]));
    const unsigned long long pol_ef = qwen4_gdn_l2_evict_first_policy();
    const uint4* H16_base =
        reinterpret_cast<const uint4*>(H) + (unsigned long long)r16 * 8ull + warp * 2u + c16;

    uint4 tmp[4];
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        tmp[b] = qwen4_gdn_ld_global_na_ef_u4(
            H16_base + (unsigned long long)b * 128ull, pol_ef
        );
    }
    #pragma unroll
    for (int jb = 0; jb < 8; jb++) {
        const int slot = jb & 3;
        qwen4_gdn_sts_u4_volatile(&smem_h_tile[warp][lane], tmp[slot]);
        __syncwarp();
        const unsigned char* wtile = reinterpret_cast<const unsigned char*>(smem_h_tile[warp]);
        #pragma unroll
        for (int k = 0; k < 16; k += 2) {
            const float2 pair = qwen4_gdn_cvt_e4m3x2_to_f32x2(
                wtile[k * 32 + lane], wtile[(k + 1) * 32 + lane]
            );
            H_reg[jb * 16 + k] = pair.x * scale;
            H_reg[jb * 16 + k + 1] = pair.y * scale;
        }
        __syncwarp();
        if (jb + 4 < 8) {
            tmp[slot] = qwen4_gdn_ld_global_na_ef_u4(
                H16_base + (unsigned long long)(jb + 4) * 128ull, pol_ef
            );
        }
    }
}

// Store FP32 H_reg[128] in-place as per-value-channel Scaled-FP8 E4M3
// (`16,384 B` E4M3 + `256 B` FP16 scales = `16.25 KiB/head`). Each thread
// `tid` owns all 128 K-rows of value channel `v = tid`, so the per-channel amax
// needs 0 shuffles and 0 `__syncthreads()`.
__device__ __forceinline__ void qwen4_gdn_store_h_scaled_fp8(
    float* dst,
    const float H_reg[QWEN4_GDN_DIM],
    float4 smem_h_tile[4][32]
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int r16 = lane >> 1;
    const unsigned int c16 = lane & 1u;

    float col_amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < QWEN4_GDN_DIM; j += 4) {
        col_amax = fmaxf(col_amax, fabsf(H_reg[j]));
        col_amax = fmaxf(col_amax, fabsf(H_reg[j + 1]));
        col_amax = fmaxf(col_amax, fabsf(H_reg[j + 2]));
        col_amax = fmaxf(col_amax, fabsf(H_reg[j + 3]));
    }
    const __half scale_hp = __float2half_ru(
        fminf(fmaxf(col_amax / 448.0f, 6.103515625e-5f), 65504.0f)
    );
    const float inv_scale = 1.0f / __half2float(scale_hp);
    __half* scales = reinterpret_cast<__half*>(
        reinterpret_cast<unsigned char*>(dst) + QWEN4_GDN_FP8_SCALES_OFFSET
    );
    qwen4_gdn_st_global_half(&scales[tid], scale_hp);

    const unsigned long long pol_ef = qwen4_gdn_l2_evict_first_policy();
    uint4* dst16_base =
        reinterpret_cast<uint4*>(dst) + (unsigned long long)r16 * 8ull + warp * 2u + c16;
    unsigned char* wtile = reinterpret_cast<unsigned char*>(smem_h_tile[warp]);

    #pragma unroll
    for (int jb = 0; jb < 8; jb++) {
        #pragma unroll
        for (int k = 0; k < 16; k += 2) {
            const unsigned short packed = qwen4_gdn_cvt_f32x2_to_e4m3x2(
                H_reg[jb * 16 + k] * inv_scale,
                H_reg[jb * 16 + k + 1] * inv_scale
            );
            wtile[k * 32 + lane] = static_cast<unsigned char>(packed & 0xffu);
            wtile[(k + 1) * 32 + lane] = static_cast<unsigned char>(packed >> 8);
        }
        __syncwarp();
        uint4 v = qwen4_gdn_lds_u4_volatile(&smem_h_tile[warp][lane]);
        qwen4_gdn_st_global_na_ef_u4(dst16_base + (unsigned long long)jb * 128ull, v, pol_ef);
        __syncwarp();
    }
}

// Exact 4-tap conv1d dot + SiLU (`--fmad=false` left-to-right FP32 order).
__device__ __forceinline__ float qwen4_gdn_conv1d_silu_4tap(
    float4 s,
    const __nv_bfloat16* conv_weight,
    const float* conv_bias,
    unsigned int ch
) {
    const uint2 cw_raw = qwen4_gdn_ld_global_nc_u32x2_volatile(
        reinterpret_cast<const uint2*>(conv_weight) + ch
    );
    const __nv_bfloat16* cw = reinterpret_cast<const __nv_bfloat16*>(&cw_raw);
    float acc = (conv_bias != nullptr) ? __ldg(&conv_bias[ch]) : 0.0f;
    acc += s.x * (float)cw[0];
    acc += s.y * (float)cw[1];
    acc += s.z * (float)cw[2];
    acc += s.w * (float)cw[3];
    return acc * (1.0f / (1.0f + __expf(-acc)));
}

// Descriptor storage in shared memory for multi-seq kernels so per-sequence
// CTA-uniform pointers load into uniform registers (UR) instead of VGPRs.
struct alignas(16) GdnMultiSeqSmem {
    float* conv_state;
    float* h_state;
    float* retain_qkv;
    float* conv_inter;
    float* h_state_inter;
    unsigned int* group_done;
    GdnKeepLateCtrl* ctrl_ptr;
    unsigned int row_base;
    unsigned int num_tokens;
    unsigned int flags;
    unsigned int pending_base;
    unsigned int pending_rows;
    unsigned int write_base;
    unsigned int ring_cap;
    unsigned int flush_now;
};

// CTA-uniform head geometry precomputed once by `tid == 0` so integer division
// (`vh / head_repeat`) and channel bases never occupy live VGPRs across H_reg[128].
struct alignas(16) GdnCtaGeom {
    unsigned int head_repeat;
    unsigned int kh;
    unsigned int sub;
    unsigned int q_base;
    unsigned int k_base;
    unsigned int v_base;
    unsigned int z_base;
    unsigned int ring_gates_off;
    unsigned int ring_row_bytes;
    float inv_sqrt_d;
};

template <bool LAZY_FINAL, bool SNAPSHOT_ALL, bool MULTI_RETAIN_GATES = false>
__device__ __forceinline__ void qwen4_gdn_fused_body(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ deinterleaved,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16* __restrict__ normed_out,
    float* __restrict__ retain_qkv,
    float* __restrict__ conv_inter,
    float* __restrict__ h_state_inter,
    unsigned int* __restrict__ group_done,
    unsigned int num_tokens,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    float l2_eps,
    float rms_eps,
    unsigned int input_stride,
    unsigned int gate_beta_stride,
    unsigned int normed_stride,
    unsigned int retain_stride,
    unsigned int conv_inter_stride,
    unsigned int h_inter_stride,
    const GdnMultiSeqSmem* __restrict__ ms_smem = nullptr
) {
    #define GET_CONV_STATE()     (ms_smem ? ms_smem->conv_state     : conv_state)
    #define GET_H_STATE()        (ms_smem ? ms_smem->h_state        : h_state)
    #define GET_RETAIN_QKV()     (ms_smem ? ms_smem->retain_qkv     : retain_qkv)
    #define GET_CONV_INTER()     (ms_smem ? ms_smem->conv_inter     : conv_inter)
    #define GET_H_STATE_INTER()  (ms_smem ? ms_smem->h_state_inter  : h_state_inter)
    #define GET_GROUP_DONE()     (ms_smem ? ms_smem->group_done     : group_done)
    #define GET_NUM_TOKENS()     (ms_smem ? ms_smem->num_tokens     : num_tokens)
    #define GET_COPY_GATES()     (ms_smem ? ((ms_smem->flags & GDN_FUSED_FLAG_COPY_GATES_TO_RETAIN) != 0u) : false)
    #define GET_KEEP_LATE()      (ms_smem ? ((ms_smem->flags & GDN_FUSED_FLAG_KEEP_LATE) != 0u) : false)
    #define GET_LOAD_SCALED_FP16()  (ms_smem ? ((ms_smem->flags & GDN_FUSED_FLAG_LOAD_SCALED_FP16) != 0u) : false)
    #define GET_STORE_SCALED_FP16() (ms_smem ? ((ms_smem->flags & GDN_FUSED_FLAG_STORE_SCALED_FP16) != 0u) : false)
    #define GET_LOAD_SCALED_FP8()   (ms_smem ? ((ms_smem->flags & GDN_FUSED_FLAG_LOAD_SCALED_FP8) != 0u) : false)
    #define GET_STORE_SCALED_FP8()  (ms_smem ? ((ms_smem->flags & GDN_FUSED_FLAG_STORE_SCALED_FP8) != 0u) : false)
    #define GET_PENDING_BASE()   (ms_smem ? ms_smem->pending_base   : 0u)
    #define GET_PENDING_ROWS()   (ms_smem ? ms_smem->pending_rows   : 0u)
    #define GET_WRITE_BASE()     (ms_smem ? ms_smem->write_base     : 0u)
    #define GET_RING_CAP()       (ms_smem ? ms_smem->ring_cap       : 0u)
    #define GET_FLUSH_NOW()      (ms_smem ? (ms_smem->flush_now != 0u) : false)
    #define GET_ROW_BASE()       (ms_smem ? (unsigned long long)ms_smem->row_base : 0ull)

    const unsigned int vh = blockIdx.x;
    if (vh >= num_v_heads) return;
    if (GET_NUM_TOKENS() == 0u) {
        if (!(LAZY_FINAL && MULTI_RETAIN_GATES && GET_KEEP_LATE()
              && GET_FLUSH_NOW()
              && (GET_PENDING_ROWS() > 0u || GET_LOAD_SCALED_FP16() || GET_LOAD_SCALED_FP8()))) {
            return;
        }
    }

    const unsigned int tid = threadIdx.x;

    __shared__ GdnCtaGeom geom;
    __shared__ float warp_sums_q[4];
    __shared__ float warp_sums_k[4];
    __shared__ __align__(16) float smem_k[2][QWEN4_GDN_DIM];
    __shared__ __align__(16) float smem_q[QWEN4_GDN_DIM];
    __shared__ float norm_parts[4];
    __shared__ unsigned int is_last_cta;
    __shared__ __align__(16) float4 smem_h_tile[4][32];
    __shared__ __align__(16) float4 smem_conv_q[QWEN4_GDN_DIM];
    __shared__ __align__(16) float4 smem_conv_k[QWEN4_GDN_DIM];
    __shared__ __align__(16) float4 smem_conv_v[QWEN4_GDN_DIM];

    if (tid == 0u) {
        const unsigned int hr = num_v_heads / num_k_heads;
        const unsigned int kh = vh / hr;
        const unsigned int key_dim = num_k_heads * QWEN4_GDN_DIM;
        const unsigned int val_dim = num_v_heads * QWEN4_GDN_DIM;
        const unsigned int conv_dim = 2u * key_dim + val_dim;
        const unsigned int gates_off = conv_dim * 2u;
        geom.head_repeat = hr;
        geom.kh = kh;
        geom.sub = vh - kh * hr;
        geom.q_base = kh * QWEN4_GDN_DIM;
        geom.k_base = key_dim + kh * QWEN4_GDN_DIM;
        geom.v_base = 2u * key_dim + vh * QWEN4_GDN_DIM;
        geom.z_base = conv_dim + vh * QWEN4_GDN_DIM;
        geom.ring_gates_off = gates_off;
        geom.ring_row_bytes = (gates_off + num_v_heads * 8u + 15u) & ~15u;
        // Compute `rsqrtf((float)blockDim.x)` via hardware `MUFU.RSQ` once per CTA.
        geom.inv_sqrt_d = rsqrtf((float)blockDim.x);
    }
    __syncthreads();

    // Step 1: Load initial 4-tap conv1d state via float4 into shared memory so
    // it consumes 0 registers across `qwen4_gdn_load_h_*`. Skip on pure
    // `num_tokens == 0 && pending_rows == 0` Scaled-FP16 -> FP32 unpacks.
    if (!(LAZY_FINAL && MULTI_RETAIN_GATES && GET_KEEP_LATE())
        || GET_NUM_TOKENS() > 0u || GET_PENDING_ROWS() > 0u) {
        const float4* cs4 = reinterpret_cast<const float4*>(GET_CONV_STATE());
        qwen4_gdn_sts_f4_volatile(&smem_conv_q[tid], cs4[geom.q_base + tid]);
        qwen4_gdn_sts_f4_volatile(&smem_conv_k[tid], cs4[geom.k_base + tid]);
        qwen4_gdn_sts_f4_volatile(&smem_conv_v[tid], cs4[geom.v_base + tid]);
    }

    // Step 2: Load per-head [128 x 128] H into FP32 registers ONCE via 128-bit
    // streaming loads (4-deep rolling window, 0 B spills): in-place per-channel
    // Scaled-FP8 (16.25 KiB), Scaled-FP16 (32 KiB + 4 B scale), or FP32 (64 KiB).
    float H_reg[QWEN4_GDN_DIM];
    if (LAZY_FINAL && MULTI_RETAIN_GATES && GET_KEEP_LATE() && GET_LOAD_SCALED_FP8()) {
        qwen4_gdn_load_h_scaled_fp8(
            GET_H_STATE() + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM),
            H_reg,
            smem_h_tile
        );
    } else if (LAZY_FINAL && MULTI_RETAIN_GATES && GET_KEEP_LATE() && GET_LOAD_SCALED_FP16()) {
        qwen4_gdn_load_h_scaled_fp16(
            GET_H_STATE() + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM),
            H_reg,
            smem_h_tile
        );
    } else {
        qwen4_gdn_load_h_f4(
            GET_H_STATE() + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM),
            H_reg,
            smem_h_tile
        );
    }

    // Step 2b (Staggered Multi-Step Compact Circular Ring KEEP_LATE):
    // Replay `pending_rows` compact rows (`raw BF16 [Q, K, V]` + `FP32 [raw_g, beta]`)
    // in registers, advancing `conv_state` and recomputing exact `--fmad=false`
    // FP32 conv1d + SiLU + K L2-norm + GDN rank-1 updates.
    // Commit `conv_state` (when pending rows ran) and `h_state` (FP32 or in-place
    // Scaled-FP16) to DRAM ONLY when `flush_now != 0`.
    if (LAZY_FINAL && MULTI_RETAIN_GATES && GET_KEEP_LATE()) {
        const bool has_pending =
            GET_PENDING_ROWS() > 0u && GET_RING_CAP() > 0u && GET_RETAIN_QKV() != nullptr;
        if (has_pending) {
            #pragma unroll 1
            for (unsigned int pt = 0u; pt < GET_PENDING_ROWS(); pt++) {
                float v_prev, g_prev, bt_prev;
                {
                    const unsigned int cap = GET_RING_CAP();
                    unsigned int r_idx = GET_PENDING_BASE() + pt;
                    if (r_idx >= cap) r_idx -= cap;

                    const unsigned char* row_ptr =
                        reinterpret_cast<const unsigned char*>(GET_RETAIN_QKV())
                        + (unsigned long long)r_idx * geom.ring_row_bytes;
                    const __nv_bfloat16* prev_raw = reinterpret_cast<const __nv_bfloat16*>(row_ptr);

                    // Q: shift only (GDN state update does not depend on Q).
                    {
                        const unsigned int q_ch = geom.q_base + tid;
                        float4 sq = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_q[tid]);
                        sq.x = sq.y; sq.y = sq.z; sq.z = sq.w; sq.w = (float)__ldg(&prev_raw[q_ch]);
                        qwen4_gdn_sts_f4_volatile(&smem_conv_q[tid], sq);
                    }

                    // K: shift + exact --fmad=false conv1d + SiLU + L2-norm.
                    float silu_k;
                    {
                        const unsigned int k_ch = geom.k_base + tid;
                        float4 sk = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_k[tid]);
                        sk.x = sk.y; sk.y = sk.z; sk.z = sk.w; sk.w = (float)__ldg(&prev_raw[k_ch]);
                        qwen4_gdn_sts_f4_volatile(&smem_conv_k[tid], sk);
                        silu_k = qwen4_gdn_conv1d_silu_4tap(sk, conv_weight, conv_bias, k_ch);
                    }

                    // V: shift + exact --fmad=false conv1d + SiLU.
                    {
                        const unsigned int v_ch = geom.v_base + tid;
                        float4 sv = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_v[tid]);
                        sv.x = sv.y; sv.y = sv.z; sv.z = sv.w; sv.w = (float)__ldg(&prev_raw[v_ch]);
                        qwen4_gdn_sts_f4_volatile(&smem_conv_v[tid], sv);
                        v_prev = qwen4_gdn_conv1d_silu_4tap(sv, conv_weight, conv_bias, v_ch);
                        asm volatile("" : "+f"(v_prev));
                    }

                    // Exact K L2-norm matching causal_conv1d_update_l2norm_f32.
                    asm volatile("" : "+f"(silu_k));
                    float sq_sum_k = silu_k * silu_k;
                    #pragma unroll
                    for (int offset = 16; offset >= 1; offset >>= 1) {
                        sq_sum_k += __shfl_down_sync(0xffffffffu, sq_sum_k, offset);
                    }
                    if ((tid & 31u) == 0u) {
                        warp_sums_k[tid >> 5] = sq_sum_k;
                    }
                    __syncthreads();
                    const float total_k =
                        ((warp_sums_k[0] + warp_sums_k[1]) + warp_sums_k[2]) + warp_sums_k[3];
                    smem_k[pt & 1u][tid] = silu_k * rsqrtf(total_k + l2_eps);

                    const float* prev_gates =
                        reinterpret_cast<const float*>(row_ptr + geom.ring_gates_off);
                    g_prev = fminf(fmaxf(__ldg(&prev_gates[vh]), 0.0f), 1.0f);
                    bt_prev = __ldg(&prev_gates[num_v_heads + vh]);
                    __syncthreads();
                }

                float* sk_buf = smem_k[pt & 1u];
                float hk_dot = 0.0f;
                #pragma unroll
                for (int j = 0; j < QWEN4_GDN_DIM; j += 4) {
                    const float4 kv = qwen4_gdn_lds_f4_volatile(&sk_buf[j]);
                    hk_dot += H_reg[j] * kv.x
                            + H_reg[j + 1] * kv.y
                            + H_reg[j + 2] * kv.z
                            + H_reg[j + 3] * kv.w;
                }
                const float v_new_i = (v_prev - g_prev * hk_dot) * bt_prev;
                #pragma unroll
                for (int j = 0; j < QWEN4_GDN_DIM; j += 4) {
                    const float4 kv = qwen4_gdn_lds_f4_volatile(&sk_buf[j]);
                    H_reg[j]     = g_prev * H_reg[j]     + kv.x * v_new_i;
                    H_reg[j + 1] = g_prev * H_reg[j + 1] + kv.y * v_new_i;
                    H_reg[j + 2] = g_prev * H_reg[j + 2] + kv.z * v_new_i;
                    H_reg[j + 3] = g_prev * H_reg[j + 3] + kv.w * v_new_i;
                }
            }
        }

        // Flush committed conv_state (when pending rows ran) and h_state to
        // DRAM ONLY when scheduled (`flush_now != 0`); also handles
        // `pending_rows == 0` Scaled-FP16/FP8 -> FP32 unpack without touching conv_state.
        if (GET_FLUSH_NOW() && (has_pending || GET_LOAD_SCALED_FP16() || GET_LOAD_SCALED_FP8())) {
            if (has_pending) {
                float4* cs4 = reinterpret_cast<float4*>(GET_CONV_STATE());
                cs4[geom.v_base + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_v[tid]);
                if (geom.head_repeat <= 1u) {
                    cs4[geom.q_base + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_q[tid]);
                    cs4[geom.k_base + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_k[tid]);
                } else {
                    if (tid == 0u) {
                        const unsigned int prev = qwen4_gdn_group_arrive_last(
                            GET_GROUP_DONE() + geom.kh, geom.head_repeat - 1u
                        );
                        is_last_cta = (prev == geom.head_repeat - 1u) ? 1u : 0u;
                    }
                    __syncthreads();
                    if (is_last_cta != 0u) {
                        cs4[geom.q_base + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_q[tid]);
                        cs4[geom.k_base + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_k[tid]);
                    }
                }
            }
            // Ensure all warps completed Step 2 h_state reads (including scales)
            // before any warp overwrites h_state in-place across FP32/FP16/FP8 layouts.
            __syncthreads();
            float* h_head =
                GET_H_STATE() + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM);
            if (GET_STORE_SCALED_FP8()) {
                qwen4_gdn_store_h_scaled_fp8(h_head, H_reg, smem_h_tile);
            } else if (GET_STORE_SCALED_FP16()) {
                qwen4_gdn_store_h_scaled_fp16(h_head, H_reg, smem_h_tile, norm_parts);
            } else {
                qwen4_gdn_store_h_f4(h_head, H_reg, smem_h_tile);
            }
        }
    }

    #pragma unroll 1
    for (unsigned int token = 0u; token < GET_NUM_TOKENS(); token++) {
        float v_val, g, bt;
        {
            const unsigned long long row_idx = GET_ROW_BASE() + (unsigned long long)token;
            const __nv_bfloat16* row_in = deinterleaved + row_idx * input_stride;
            float* rqkv = GET_RETAIN_QKV();
            float* ci = GET_CONV_INTER();
            const bool is_keep_late = LAZY_FINAL && MULTI_RETAIN_GATES && GET_KEEP_LATE();

            const bool write_ring = is_keep_late && rqkv != nullptr && GET_RING_CAP() > 0u;

            #define GET_RING_DST_ROW() ({ \
                const unsigned int _cap = GET_RING_CAP(); \
                unsigned int _w_idx = GET_WRITE_BASE() + token; \
                if (_w_idx >= _cap) _w_idx -= _cap; \
                reinterpret_cast<unsigned char*>(rqkv) \
                    + (unsigned long long)_w_idx * geom.ring_row_bytes; \
            })

            if (write_ring && vh == 0u) {
                unsigned char* dst_row = GET_RING_DST_ROW();
                float* dst_gates = reinterpret_cast<float*>(dst_row + geom.ring_gates_off);
                const unsigned long long gb_row = row_idx * gate_beta_stride;
                const unsigned int two_nv = num_v_heads * 2u;
                for (unsigned int idx = tid; idx < two_nv; idx += 128u) {
                    const float val = (idx < num_v_heads)
                        ? __ldg(&gate[gb_row + idx])
                        : __ldg(&beta[gb_row + (idx - num_v_heads)]);
                    qwen4_gdn_st_global_f32(&dst_gates[idx], val);
                }
            }

            // 1. Causal conv1d shift + dot + SiLU, serialized Q -> K -> V so only
            // one 4-tap window and one weight vector are live at a time (0 B spills).
            float silu_q;
            {
                const unsigned int q_ch = geom.q_base + tid;
                const __nv_bfloat16 raw_q = __ldg(&row_in[q_ch]);
                if (write_ring && geom.sub == 0u) {
                    qwen4_gdn_st_global_u16(
                        &reinterpret_cast<__nv_bfloat16*>(GET_RING_DST_ROW())[q_ch], raw_q
                    );
                }
                float4 sq = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_q[tid]);
                sq.x = sq.y; sq.y = sq.z; sq.z = sq.w; sq.w = (float)raw_q;
                qwen4_gdn_sts_f4_volatile(&smem_conv_q[tid], sq);
                if (ci != nullptr && geom.sub == 0u) {
                    float4* snap_base = reinterpret_cast<float4*>(
                        ci + (unsigned long long)token * conv_inter_stride
                    );
                    qwen4_gdn_st_global_cs_f4(&snap_base[q_ch], sq);
                }
                silu_q = qwen4_gdn_conv1d_silu_4tap(sq, conv_weight, conv_bias, q_ch);
            }

            float silu_k;
            {
                const unsigned int k_ch = geom.k_base + tid;
                const __nv_bfloat16 raw_k = __ldg(&row_in[k_ch]);
                if (write_ring && geom.sub == 0u) {
                    qwen4_gdn_st_global_u16(
                        &reinterpret_cast<__nv_bfloat16*>(GET_RING_DST_ROW())[k_ch], raw_k
                    );
                }
                float4 sk = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_k[tid]);
                sk.x = sk.y; sk.y = sk.z; sk.z = sk.w; sk.w = (float)raw_k;
                qwen4_gdn_sts_f4_volatile(&smem_conv_k[tid], sk);
                if (ci != nullptr && geom.sub == 0u) {
                    float4* snap_base = reinterpret_cast<float4*>(
                        ci + (unsigned long long)token * conv_inter_stride
                    );
                    qwen4_gdn_st_global_cs_f4(&snap_base[k_ch], sk);
                }
                silu_k = qwen4_gdn_conv1d_silu_4tap(sk, conv_weight, conv_bias, k_ch);
            }

            {
                const unsigned int v_ch = geom.v_base + tid;
                const __nv_bfloat16 raw_v = __ldg(&row_in[v_ch]);
                if (write_ring) {
                    qwen4_gdn_st_global_u16(
                        &reinterpret_cast<__nv_bfloat16*>(GET_RING_DST_ROW())[v_ch], raw_v
                    );
                }
                float4 sv = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_v[tid]);
                sv.x = sv.y; sv.y = sv.z; sv.z = sv.w; sv.w = (float)raw_v;
                qwen4_gdn_sts_f4_volatile(&smem_conv_v[tid], sv);
                if (ci != nullptr) {
                    float4* snap_base = reinterpret_cast<float4*>(
                        ci + (unsigned long long)token * conv_inter_stride
                    );
                    qwen4_gdn_st_global_cs_f4(&snap_base[v_ch], sv);
                }
                v_val = qwen4_gdn_conv1d_silu_4tap(sv, conv_weight, conv_bias, v_ch);
                asm volatile("" : "+f"(v_val));
            }
            #undef GET_RING_DST_ROW

            // 2. Per-head Q/K L2 normalization matching causal_conv1d_update_l2norm_f32.
            asm volatile("" : "+f"(silu_q), "+f"(silu_k));
            float sq_sum_q = silu_q * silu_q;
            float sq_sum_k = silu_k * silu_k;
            #pragma unroll
            for (int offset = 16; offset >= 1; offset >>= 1) {
                sq_sum_q += __shfl_down_sync(0xffffffffu, sq_sum_q, offset);
                sq_sum_k += __shfl_down_sync(0xffffffffu, sq_sum_k, offset);
            }
            if ((tid & 31u) == 0u) {
                warp_sums_q[tid >> 5] = sq_sum_q;
                warp_sums_k[tid >> 5] = sq_sum_k;
            }
            __syncthreads();

            const float total_q =
                ((warp_sums_q[0] + warp_sums_q[1]) + warp_sums_q[2]) + warp_sums_q[3];
            const float total_k =
                ((warp_sums_k[0] + warp_sums_k[1]) + warp_sums_k[2]) + warp_sums_k[3];
            const float q_val = silu_q * rsqrtf(total_q + l2_eps);
            const float k_val = silu_k * rsqrtf(total_k + l2_eps);

            smem_q[tid] = q_val;
            smem_k[0][tid] = k_val;

            // Legacy non-KEEP_LATE lazyfinal writes FP32 post-conv QKV + coalesced gates.
            if (LAZY_FINAL && !is_keep_late && rqkv != nullptr) {
                float* retain_row = rqkv + (unsigned long long)token * retain_stride;
                qwen4_gdn_st_global_cs_f32(&retain_row[geom.v_base + tid], v_val);
                if (geom.sub == 0u) {
                    qwen4_gdn_st_global_cs_f32(&retain_row[geom.q_base + tid], q_val);
                    qwen4_gdn_st_global_cs_f32(&retain_row[geom.k_base + tid], k_val);
                }
                if (MULTI_RETAIN_GATES && GET_COPY_GATES() && vh == 0u) {
                    const unsigned long long gb_row = row_idx * gate_beta_stride;
                    float* retain_gates = rqkv
                        + (unsigned long long)QWEN4_GDN_LAZY_MAX_K * retain_stride
                        + (unsigned long long)token * gate_beta_stride;
                    const unsigned int two_nv = num_v_heads * 2u;
                    for (unsigned int idx = tid; idx < two_nv; idx += 128u) {
                        const float val = (idx < num_v_heads)
                            ? __ldg(&gate[gb_row + idx])
                            : __ldg(&beta[gb_row + (idx - num_v_heads)]);
                        qwen4_gdn_st_global_cs_f32(&retain_gates[idx], val);
                    }
                }
            }

            const unsigned long long gb_off = row_idx * gate_beta_stride + vh;
            g = fminf(fmaxf(__ldg(&gate[gb_off]), 0.0f), 1.0f);
            bt = __ldg(&beta[gb_off]);
            __syncthreads();
        }

        // 3. Register-resident GDN recurrence (exact unroll-4 single-accumulator order).
        float hk_dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < QWEN4_GDN_DIM; j += 4) {
            const float4 kv = qwen4_gdn_lds_f4_volatile(&smem_k[0][j]);
            hk_dot += H_reg[j] * kv.x
                    + H_reg[j + 1] * kv.y
                    + H_reg[j + 2] * kv.z
                    + H_reg[j + 3] * kv.w;
        }
        const float v_new_i = (v_val - g * hk_dot) * bt;
        float q_dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < QWEN4_GDN_DIM; j += 4) {
            const float4 kv = qwen4_gdn_lds_f4_volatile(&smem_k[0][j]);
            const float4 qv = qwen4_gdn_lds_f4_volatile(&smem_q[j]);
            float h0 = g * H_reg[j]     + kv.x * v_new_i;
            float h1 = g * H_reg[j + 1] + kv.y * v_new_i;
            float h2 = g * H_reg[j + 2] + kv.z * v_new_i;
            float h3 = g * H_reg[j + 3] + kv.w * v_new_i;
            H_reg[j]     = h0;
            H_reg[j + 1] = h1;
            H_reg[j + 2] = h2;
            H_reg[j + 3] = h3;
            q_dot += h0 * qv.x
                   + h1 * qv.y
                   + h2 * qv.z
                   + h3 * qv.w;
        }

        // 4. Per-head gated RMS norm + sigmoid Z-gate matching qwen4_gated_rms_sigmoid_body.
        {
            const float gdn_val = q_dot * geom.inv_sqrt_d;
            float rms_sq = qwen4_gdn_warp_sum_xor(gdn_val * gdn_val);
            if ((tid & 31u) == 0u) {
                norm_parts[tid >> 5] = rms_sq;
            }
            const unsigned long long row_idx = GET_ROW_BASE() + (unsigned long long)token;
            const float z_val = __bfloat162float(
                __ldg(&deinterleaved[row_idx * input_stride + geom.z_base + tid])
            );
            const float rms_w = __bfloat162float(__ldg(&rms_weight[tid]));
            __syncthreads();

            const float v_part = (norm_parts[0] + norm_parts[2]) + (norm_parts[1] + norm_parts[3]);
            const float inv_rms = rsqrtf(v_part / (float)QWEN4_GDN_DIM + rms_eps);
            normed_out[row_idx * normed_stride + vh * QWEN4_GDN_DIM + tid] =
                __float2bfloat16(gdn_val * inv_rms * rms_w / (1.0f + expf(-z_val)));
        }
        if (SNAPSHOT_ALL) {
            float* snapshot = GET_H_STATE_INTER()
                + (unsigned long long)token * h_inter_stride
                + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM);
            qwen4_gdn_store_h_f4(snapshot, H_reg, smem_h_tile);
        }
    }

    // Commit final H state according to mode:
    // - LAZY_FINAL + KEEP_LATE: h_state already holds committed base/flushed H.
    // - LAZY_FINAL (legacy): h_state stays step-initial; final H written to h_state_inter[ntok - 1].
    // - Otherwise (K=1 decode or persistent verify): final H written directly to h_state.
    if (LAZY_FINAL) {
        if (!(MULTI_RETAIN_GATES && GET_KEEP_LATE())) {
            float* fin = GET_H_STATE_INTER()
                + (unsigned long long)(GET_NUM_TOKENS() - 1u) * h_inter_stride
                + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM);
            qwen4_gdn_store_h_f4(fin, H_reg, smem_h_tile);
        }
    } else {
        qwen4_gdn_store_h_f4(
            GET_H_STATE() + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM),
            H_reg,
            smem_h_tile
        );
    }

    // Commit live conv_state unless LAZY_FINAL + KEEP_LATE is active (where
    // `conv_state` stays pinned at the base/flushed checkpoint).
    if (!(LAZY_FINAL && MULTI_RETAIN_GATES && GET_KEEP_LATE())) {
        float4* cs4 = reinterpret_cast<float4*>(GET_CONV_STATE());
        cs4[geom.v_base + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_v[tid]);
        if (geom.head_repeat <= 1u) {
            cs4[geom.q_base + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_q[tid]);
            cs4[geom.k_base + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_k[tid]);
        } else {
            if (tid == 0u) {
                const unsigned int prev = qwen4_gdn_group_arrive_last(
                    GET_GROUP_DONE() + geom.kh, geom.head_repeat - 1u
                );
                is_last_cta = (prev == geom.head_repeat - 1u) ? 1u : 0u;
            }
            __syncthreads();
            if (is_last_cta != 0u) {
                cs4[geom.q_base + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_q[tid]);
                cs4[geom.k_base + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_k[tid]);
            }
        }
    }

    #undef GET_CONV_STATE
    #undef GET_H_STATE
    #undef GET_RETAIN_QKV
    #undef GET_CONV_INTER
    #undef GET_H_STATE_INTER
    #undef GET_GROUP_DONE
    #undef GET_NUM_TOKENS
    #undef GET_COPY_GATES
    #undef GET_KEEP_LATE
    #undef GET_LOAD_SCALED_FP16
    #undef GET_STORE_SCALED_FP16
    #undef GET_LOAD_SCALED_FP8
    #undef GET_STORE_SCALED_FP8
    #undef GET_PENDING_BASE
    #undef GET_PENDING_ROWS
    #undef GET_WRITE_BASE
    #undef GET_RING_CAP
    #undef GET_FLUSH_NOW
    #undef GET_ROW_BASE
}

extern "C" __global__ void __launch_bounds__(128, 3)
qwen4_gdn_decode_fused_k1(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ deinterleaved,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16* __restrict__ normed_out,
    unsigned int* __restrict__ group_done,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    float l2_eps,
    float rms_eps
) {
    const unsigned int qkvz_stride =
        2u * num_k_heads * QWEN4_GDN_DIM + 2u * num_v_heads * QWEN4_GDN_DIM;
    const unsigned int value_dim = num_v_heads * QWEN4_GDN_DIM;
    qwen4_gdn_fused_body<false, false, false>(
        conv_state, deinterleaved, conv_weight, conv_bias,
        gate, beta, h_state, rms_weight, normed_out,
        nullptr, nullptr, nullptr, group_done,
        1u, num_k_heads, num_v_heads, l2_eps, rms_eps,
        qkvz_stride, num_v_heads * 2u, value_dim, 0u, 0u, 0u
    );
}

extern "C" __global__ void __launch_bounds__(128, 3)
qwen4_gdn_verify_fused_lazyfinal(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ deinterleaved,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16* __restrict__ normed_out,
    float* __restrict__ retain_qkv,
    float* __restrict__ conv_inter,
    float* __restrict__ h_state_inter,
    unsigned int* __restrict__ group_done,
    unsigned int num_tokens,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    float l2_eps,
    float rms_eps,
    unsigned int input_stride,
    unsigned int gate_beta_stride,
    unsigned int normed_stride,
    unsigned int retain_stride,
    unsigned int conv_inter_stride,
    unsigned int h_inter_stride
) {
    qwen4_gdn_fused_body<true, false, false>(
        conv_state, deinterleaved, conv_weight, conv_bias,
        gate, beta, h_state, rms_weight, normed_out,
        retain_qkv, conv_inter, h_state_inter, group_done,
        num_tokens, num_k_heads, num_v_heads, l2_eps, rms_eps,
        input_stride, gate_beta_stride, normed_stride,
        retain_stride, conv_inter_stride, h_inter_stride
    );
}

extern "C" __global__ void __launch_bounds__(128, 3)
qwen4_gdn_verify_fused_persistent(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ deinterleaved,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16* __restrict__ normed_out,
    float* __restrict__ conv_inter,
    float* __restrict__ h_state_inter,
    unsigned int* __restrict__ group_done,
    unsigned int num_tokens,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    float l2_eps,
    float rms_eps,
    unsigned int input_stride,
    unsigned int gate_beta_stride,
    unsigned int normed_stride,
    unsigned int conv_inter_stride,
    unsigned int h_inter_stride
) {
    qwen4_gdn_fused_body<false, true, false>(
        conv_state, deinterleaved, conv_weight, conv_bias,
        gate, beta, h_state, rms_weight, normed_out,
        nullptr, conv_inter, h_state_inter, group_done,
        num_tokens, num_k_heads, num_v_heads, l2_eps, rms_eps,
        input_stride, gate_beta_stride, normed_stride,
        0u, conv_inter_stride, h_inter_stride
    );
}

extern "C" __global__ void __launch_bounds__(128, 3)
qwen4_gdn_decode_fused_k1_multi_seq(
    const GdnFusedMultiSeqDesc* __restrict__ seq_descs,
    const __nv_bfloat16* __restrict__ deinterleaved,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16* __restrict__ normed_out,
    unsigned int num_seqs,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    float l2_eps,
    float rms_eps
) {
    const unsigned int seq_idx = blockIdx.y;
    if (seq_idx >= num_seqs) return;
    __shared__ GdnMultiSeqSmem ms;
    if (threadIdx.x == 0u) {
        const GdnFusedMultiSeqDesc desc = seq_descs[seq_idx];
        ms.conv_state = (float*)desc.conv_state;
        ms.h_state = (float*)desc.h_state;
        ms.retain_qkv = nullptr;
        ms.conv_inter = nullptr;
        ms.h_state_inter = nullptr;
        ms.group_done = (unsigned int*)desc.group_done;
        ms.ctrl_ptr = nullptr;
        ms.row_base = desc.row_base;
        ms.num_tokens = 1u;
        ms.flags = 0u;
        ms.pending_base = 0u;
        ms.pending_rows = 0u;
        ms.write_base = 0u;
        ms.ring_cap = 0u;
        ms.flush_now = 0u;
    }
    __syncthreads();
    const unsigned int qkvz_stride =
        2u * num_k_heads * QWEN4_GDN_DIM + 2u * num_v_heads * QWEN4_GDN_DIM;
    const unsigned int gate_beta_stride = num_v_heads * 2u;
    const unsigned int value_dim = num_v_heads * QWEN4_GDN_DIM;
    qwen4_gdn_fused_body<false, false, false>(
        nullptr, deinterleaved, conv_weight, conv_bias,
        gate, beta, nullptr, rms_weight, normed_out,
        nullptr, nullptr, nullptr, nullptr,
        1u, num_k_heads, num_v_heads, l2_eps, rms_eps,
        qkvz_stride, gate_beta_stride, value_dim,
        0u, 0u, 0u, &ms
    );
}

extern "C" __global__ void __launch_bounds__(128, 3)
qwen4_gdn_verify_fused_lazyfinal_multi_seq(
    const GdnFusedMultiSeqDesc* __restrict__ seq_descs,
    const __nv_bfloat16* __restrict__ deinterleaved,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16* __restrict__ normed_out,
    unsigned int num_seqs,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    float l2_eps,
    float rms_eps,
    unsigned int input_stride,
    unsigned int gate_beta_stride,
    unsigned int normed_stride,
    unsigned int retain_stride,
    unsigned int conv_inter_stride,
    unsigned int h_inter_stride
) {
    const unsigned int seq_idx = blockIdx.y;
    if (seq_idx >= num_seqs) return;
    __shared__ GdnMultiSeqSmem ms;
    if (threadIdx.x == 0u) {
        const GdnFusedMultiSeqDesc desc = seq_descs[seq_idx];
        const bool keep_late = (desc.flags & GDN_FUSED_FLAG_KEEP_LATE) != 0u;
        const bool ctrl_indirect =
            keep_late && ((desc.flags & GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR) != 0u);
        unsigned long long retain_qkv_addr = desc.retain_qkv;
        unsigned long long h_state_inter_addr = desc.h_state_inter;
        GdnKeepLateCtrl* ctrl_ptr = nullptr;
        unsigned int pending_base = 0u;
        unsigned int pending_rows = 0u;
        unsigned int write_base = 0u;
        unsigned int ring_cap = 0u;
        unsigned int flush_now = 0u;
        if (ctrl_indirect && desc.h_state_inter != 0ull) {
            ctrl_ptr = (GdnKeepLateCtrl*)desc.h_state_inter;
            retain_qkv_addr = ctrl_ptr->ring_base;
            h_state_inter_addr = 0ull;
            pending_base = ctrl_ptr->pending_base;
            pending_rows = ctrl_ptr->pending_rows;
            write_base = ctrl_ptr->write_base;
            ring_cap = ctrl_ptr->ring_cap;
            flush_now = ctrl_ptr->flush_now;
        }
        ms.conv_state = (float*)desc.conv_state;
        ms.h_state = (float*)desc.h_state;
        ms.retain_qkv = (float*)retain_qkv_addr;
        ms.conv_inter = (float*)desc.conv_inter;
        ms.h_state_inter = (float*)h_state_inter_addr;
        ms.group_done = (unsigned int*)desc.group_done;
        ms.ctrl_ptr = ctrl_ptr;
        ms.row_base = desc.row_base;
        ms.num_tokens = desc.num_tokens;
        ms.flags = desc.flags;
        ms.pending_base = pending_base;
        ms.pending_rows = pending_rows;
        ms.write_base = write_base;
        ms.ring_cap = ring_cap;
        ms.flush_now = flush_now;
    }
    __syncthreads();
    qwen4_gdn_fused_body<true, false, true>(
        nullptr, deinterleaved, conv_weight, conv_bias,
        gate, beta, nullptr, rms_weight, normed_out,
        nullptr, nullptr, nullptr, nullptr,
        ms.num_tokens, num_k_heads, num_v_heads, l2_eps, rms_eps,
        input_stride, gate_beta_stride, normed_stride,
        retain_stride, conv_inter_stride, h_inter_stride, &ms
    );
    if (blockIdx.x == 0u && threadIdx.x == 0u && ms.ctrl_ptr != nullptr) {
        ms.ctrl_ptr->executed_flag = 1u;
    }
}

extern "C" __global__ void __launch_bounds__(128, 3)
qwen4_gdn_verify_fused_persistent_multi_seq(
    const GdnFusedMultiSeqDesc* __restrict__ seq_descs,
    const __nv_bfloat16* __restrict__ deinterleaved,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16* __restrict__ normed_out,
    unsigned int num_seqs,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    float l2_eps,
    float rms_eps,
    unsigned int input_stride,
    unsigned int gate_beta_stride,
    unsigned int normed_stride,
    unsigned int conv_inter_stride,
    unsigned int h_inter_stride
) {
    const unsigned int seq_idx = blockIdx.y;
    if (seq_idx >= num_seqs) return;
    __shared__ GdnMultiSeqSmem ms;
    if (threadIdx.x == 0u) {
        const GdnFusedMultiSeqDesc desc = seq_descs[seq_idx];
        ms.conv_state = (float*)desc.conv_state;
        ms.h_state = (float*)desc.h_state;
        ms.retain_qkv = nullptr;
        ms.conv_inter = (float*)desc.conv_inter;
        ms.h_state_inter = (float*)desc.h_state_inter;
        ms.group_done = (unsigned int*)desc.group_done;
        ms.ctrl_ptr = nullptr;
        ms.row_base = desc.row_base;
        ms.num_tokens = desc.num_tokens;
        ms.flags = 0u;
        ms.pending_base = 0u;
        ms.pending_rows = 0u;
        ms.write_base = 0u;
        ms.ring_cap = 0u;
        ms.flush_now = 0u;
    }
    __syncthreads();
    qwen4_gdn_fused_body<false, true, false>(
        nullptr, deinterleaved, conv_weight, conv_bias,
        gate, beta, nullptr, rms_weight, normed_out,
        nullptr, nullptr, nullptr, nullptr,
        ms.num_tokens, num_k_heads, num_v_heads, l2_eps, rms_eps,
        input_stride, gate_beta_stride, normed_stride,
        0u, conv_inter_stride, h_inter_stride, &ms
    );
}

// ============================================================================
// Split-V (SPLIT_V = 4) multi-CTA fused Qwen4 GDN decode / verify kernels
// for wide-SM Blackwell GPUs (> 64 SMs, e.g. 188-SM SM120 / SM103):
//   - Grid: [num_v_heads, 4, num_seqs] (192 CTAs at C=1 -> 100% of 188 SMs)
//   - Block: [128, 1, 1] (4 warps x 32 lanes):
//       * CTA (vh, sv) owns 32 V-columns v = sv*32 + lane.
//       * Warp w (0..3) owns 32 K-rows k = w*32 .. w*32+31 -> H_reg[32]
//         (4x fewer VGPRs per thread, high occupancy, 8 KiB rolling loads/CTA
//         = 1.5 MiB DRAM MLP across 188 SMs).
//   - Q/K conv1d + L2-norm runs across all 128 threads identically to baseline;
//     V conv1d runs on warp 0 (tid < 32) for [v_base + sv*32 .. +32).
//   - Race-free conv_state commit:
//       * V slice [v_base + sv*32 .. +32) is disjoint per (vh, sv) CTA.
//       * Q/K conv_state is committed ONLY by the last of `head_repeat * 4`
//         CTAs sharing key head `kh` via self-resetting `group_done[kh]`.
//   - Zero-spin Step 4 RMS norm finalize:
//       * Each CTA writes its 32 unnormalized `gdn_val`s and warp `rms_sq` to
//         L2-coherent scratch (`scratch_gdn`, `scratch_ss`) via `st.global.cg`,
//         then arrives on self-resetting `split_done[seq * num_v_heads + vh]`
//         (wrap = 3). The 4th CTA normalizes all 128 columns in parallel across
//         all tokens using `(s0 + s2) + (s1 + s3)` (matching baseline's tree).
//   - Supports FP32, Scaled-FP16, and Scaled-FP8 H-state staging across the
//     four Split-V CTAs (see `qwen4_gdn_load_h_scaled_fp16_splitv4` and
//     Split-V store helpers below).
// ============================================================================

#define QWEN4_GDN_SPLIT_V 4u
#define QWEN4_GDN_SPLIT_K_PER_WARP 32

__device__ __forceinline__ float qwen4_gdn_ld_global_cg_f32(const float* ptr) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(ptr));
    return v;
}

__device__ __forceinline__ void qwen4_gdn_st_global_cg_f32(float* ptr, float v) {
    asm volatile("st.global.cg.f32 [%0], %1;" :: "l"(ptr), "f"(v) : "memory");
}

__device__ __forceinline__ float4 qwen4_gdn_ld_global_cg_f4(const float4* ptr) {
    float4 v;
    asm volatile(
        "ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(ptr)
    );
    return v;
}

__device__ __forceinline__ void qwen4_gdn_st_global_cg_f4(float4* ptr, float4 v) {
    asm volatile(
        "st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
        : "memory"
    );
}

// Load [128 x 32] V-slice `sv` of in-place per-head Scaled-FP16 H (`32,768 B`
// row-major __half + FP32 scale at byte offset 32768) into warp-sharded
// H_reg[32] (warp w owns K-rows [w*32 .. w*32+31], lane owns V-col sv*32+lane)
// via 4x 128-bit streaming uint4 loads per warp (0 __syncthreads(), 0-ULP match
// with `qwen4_gdn_load_h_scaled_fp16`).
__device__ __forceinline__ void qwen4_gdn_load_h_scaled_fp16_splitv4(
    const float* H,
    float H_reg[QWEN4_GDN_SPLIT_K_PER_WARP],
    float4 smem_h_tile[4][32],
    unsigned int sv
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int r8 = lane >> 2;
    const unsigned int c8 = lane & 3u;
    const float scale = __ldg(
        reinterpret_cast<const float*>(
            reinterpret_cast<const unsigned char*>(H) + 32768u
        )
    );
    const unsigned long long pol_ef = qwen4_gdn_l2_evict_first_policy();
    const uint4* H8_base =
        reinterpret_cast<const uint4*>(H)
        + (unsigned long long)(warp * 32u + r8) * 16ull
        + (unsigned long long)sv * 4ull
        + c8;

    uint4 tmp[4];
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        tmp[b] = qwen4_gdn_ld_global_na_ef_u4(
            H8_base + (unsigned long long)b * 128ull, pol_ef
        );
    }
    #pragma unroll
    for (int jb = 0; jb < 4; jb++) {
        qwen4_gdn_sts_u4_volatile(&smem_h_tile[warp][lane], tmp[jb]);
        __syncwarp();
        const __half* wtile = reinterpret_cast<const __half*>(smem_h_tile[warp]);
        H_reg[jb * 8 + 0] = __half2float(wtile[0 * 32 + lane]) * scale;
        H_reg[jb * 8 + 1] = __half2float(wtile[1 * 32 + lane]) * scale;
        H_reg[jb * 8 + 2] = __half2float(wtile[2 * 32 + lane]) * scale;
        H_reg[jb * 8 + 3] = __half2float(wtile[3 * 32 + lane]) * scale;
        H_reg[jb * 8 + 4] = __half2float(wtile[4 * 32 + lane]) * scale;
        H_reg[jb * 8 + 5] = __half2float(wtile[5 * 32 + lane]) * scale;
        H_reg[jb * 8 + 6] = __half2float(wtile[6 * 32 + lane]) * scale;
        H_reg[jb * 8 + 7] = __half2float(wtile[7 * 32 + lane]) * scale;
        __syncwarp();
    }
}

// Store warp-sharded H_reg[32] into [128 x 32] V-slice `sv` of L2-coherent
// FP32 staging `dst[128 x 128]` via warp-local smem transpose + `st.global.cg.v4`.
__device__ __forceinline__ void qwen4_gdn_store_h_f4_splitv4_cg(
    float* dst,
    const float H_reg[QWEN4_GDN_SPLIT_K_PER_WARP],
    float4 smem_h_tile[4][32],
    unsigned int sv
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int r4 = lane >> 3;
    const unsigned int c4 = lane & 7u;
    float4* dst4_base =
        reinterpret_cast<float4*>(dst)
        + (unsigned long long)(warp * 32u + r4) * 32ull
        + (unsigned long long)sv * 8ull
        + c4;

    #pragma unroll
    for (int jb = 0; jb < 8; jb++) {
        float* wtile = reinterpret_cast<float*>(smem_h_tile[warp]);
        wtile[0 * 32 + lane] = H_reg[jb * 4 + 0];
        wtile[1 * 32 + lane] = H_reg[jb * 4 + 1];
        wtile[2 * 32 + lane] = H_reg[jb * 4 + 2];
        wtile[3 * 32 + lane] = H_reg[jb * 4 + 3];
        __syncwarp();
        float4 v = qwen4_gdn_lds_f4_volatile_f4(&smem_h_tile[warp][lane]);
        qwen4_gdn_st_global_cg_f4(dst4_base + (unsigned long long)jb * 128ull, v);
        __syncwarp();
    }
}

// Last-arriving CTA finalizer: quantize the full [128 x 128] FP32 head staged
// across all 4 `sv` CTAs in L2 into in-place Scaled-FP16 (0-ULP identical scale
// and __float2half_rn rounding to `qwen4_gdn_store_h_scaled_fp16`).
__device__ __forceinline__ void qwen4_gdn_finalize_h_scaled_fp16_from_scratch(
    float* dst_h_head,
    const float* src_scratch_h_head,
    const float* scratch_amax_head
) {
    const unsigned int tid = threadIdx.x;
    const float a0 = qwen4_gdn_ld_global_cg_f32(&scratch_amax_head[0]);
    const float a1 = qwen4_gdn_ld_global_cg_f32(&scratch_amax_head[1]);
    const float a2 = qwen4_gdn_ld_global_cg_f32(&scratch_amax_head[2]);
    const float a3 = qwen4_gdn_ld_global_cg_f32(&scratch_amax_head[3]);
    const float head_amax = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));
    const float scale = fmaxf(head_amax / 65000.0f, 1.0e-30f);
    const float inv_scale = 1.0f / scale;
    if (tid == 0u) {
        float* scale_ptr = reinterpret_cast<float*>(
            reinterpret_cast<unsigned char*>(dst_h_head) + 32768u
        );
        qwen4_gdn_st_global_f32(scale_ptr, scale);
    }

    const unsigned long long pol_ef = qwen4_gdn_l2_evict_first_policy();
    const float4* src4 = reinterpret_cast<const float4*>(src_scratch_h_head);
    uint4* dst8 = reinterpret_cast<uint4*>(dst_h_head);
    #pragma unroll 4
    for (unsigned int it = 0u; it < 16u; it++) {
        const unsigned int idx8 = it * 128u + tid;
        const float4 v0 = qwen4_gdn_ld_global_cg_f4(&src4[idx8 * 2u + 0u]);
        const float4 v1 = qwen4_gdn_ld_global_cg_f4(&src4[idx8 * 2u + 1u]);
        const __half2 h01 = __halves2half2(
            __float2half_rn(v0.x * inv_scale), __float2half_rn(v0.y * inv_scale)
        );
        const __half2 h23 = __halves2half2(
            __float2half_rn(v0.z * inv_scale), __float2half_rn(v0.w * inv_scale)
        );
        const __half2 h45 = __halves2half2(
            __float2half_rn(v1.x * inv_scale), __float2half_rn(v1.y * inv_scale)
        );
        const __half2 h67 = __halves2half2(
            __float2half_rn(v1.z * inv_scale), __float2half_rn(v1.w * inv_scale)
        );
        const uint4 packed = make_uint4(
            *reinterpret_cast<const unsigned int*>(&h01),
            *reinterpret_cast<const unsigned int*>(&h23),
            *reinterpret_cast<const unsigned int*>(&h45),
            *reinterpret_cast<const unsigned int*>(&h67)
        );
        qwen4_gdn_st_global_na_ef_u4(&dst8[idx8], packed, pol_ef);
    }
}

// Load [128 x 32] V-slice `sv` of in-place per-value-channel Scaled-FP8 E4M3 H
// (`16,384 B` row-major E4M3 + `128 x __half = 256 B` scales at byte offset
// 16384 = `16.25 KiB/head`) into warp-sharded H_reg[32] (warp w owns K-rows
// [w*32 .. w*32+31], lane owns V-col sv*32+lane) via 2x 128-bit streaming uint4
// loads per warp (0 __syncthreads(), 0-ULP match with `qwen4_gdn_load_h_scaled_fp8`).
__device__ __forceinline__ void qwen4_gdn_load_h_scaled_fp8_splitv4(
    const float* H,
    float H_reg[QWEN4_GDN_SPLIT_K_PER_WARP],
    float4 smem_h_tile[4][32],
    unsigned int sv
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int r16 = lane >> 1;
    const unsigned int c16 = lane & 1u;
    const __half* scales = reinterpret_cast<const __half*>(
        reinterpret_cast<const unsigned char*>(H) + QWEN4_GDN_FP8_SCALES_OFFSET
    );
    const float scale = __half2float(__ldg(&scales[sv * 32u + lane]));
    const unsigned long long pol_ef = qwen4_gdn_l2_evict_first_policy();
    const uint4* H16_base =
        reinterpret_cast<const uint4*>(H)
        + (unsigned long long)(warp * 32u + r16) * 8ull
        + (unsigned long long)sv * 2ull
        + c16;

    uint4 tmp[2];
    tmp[0] = qwen4_gdn_ld_global_na_ef_u4(H16_base, pol_ef);
    tmp[1] = qwen4_gdn_ld_global_na_ef_u4(H16_base + 128ull, pol_ef);

    #pragma unroll
    for (int jb = 0; jb < 2; jb++) {
        qwen4_gdn_sts_u4_volatile(&smem_h_tile[warp][lane], tmp[jb]);
        __syncwarp();
        const unsigned char* wtile = reinterpret_cast<const unsigned char*>(smem_h_tile[warp]);
        #pragma unroll
        for (int k = 0; k < 16; k += 2) {
            const float2 pair = qwen4_gdn_cvt_e4m3x2_to_f32x2(
                wtile[k * 32 + lane], wtile[(k + 1) * 32 + lane]
            );
            H_reg[jb * 16 + k] = pair.x * scale;
            H_reg[jb * 16 + k + 1] = pair.y * scale;
        }
        __syncwarp();
    }
}

// Last-arriving CTA finalizer: quantize the full [128 x 128] FP32 head staged
// across all 4 `sv` CTAs in L2 into in-place per-value-channel Scaled-FP8 E4M3
// (`16.25 KiB/head`, 0-ULP identical scales and E4M3 payload to
// `qwen4_gdn_store_h_scaled_fp8`).
__device__ __forceinline__ void qwen4_gdn_finalize_h_scaled_fp8_from_scratch(
    float* dst_h_head,
    const float* src_scratch_h_head,
    float4 smem_h_tile[4][32]
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int r16 = lane >> 1;
    const unsigned int c16 = lane & 1u;

    float col_amax = 0.0f;
    #pragma unroll 4
    for (unsigned int k = 0u; k < QWEN4_GDN_DIM; k += 4u) {
        col_amax = fmaxf(
            col_amax, fabsf(qwen4_gdn_ld_global_cg_f32(&src_scratch_h_head[(k + 0u) * QWEN4_GDN_DIM + tid]))
        );
        col_amax = fmaxf(
            col_amax, fabsf(qwen4_gdn_ld_global_cg_f32(&src_scratch_h_head[(k + 1u) * QWEN4_GDN_DIM + tid]))
        );
        col_amax = fmaxf(
            col_amax, fabsf(qwen4_gdn_ld_global_cg_f32(&src_scratch_h_head[(k + 2u) * QWEN4_GDN_DIM + tid]))
        );
        col_amax = fmaxf(
            col_amax, fabsf(qwen4_gdn_ld_global_cg_f32(&src_scratch_h_head[(k + 3u) * QWEN4_GDN_DIM + tid]))
        );
    }
    const __half scale_hp = __float2half_ru(
        fminf(fmaxf(col_amax / 448.0f, 6.103515625e-5f), 65504.0f)
    );
    const float inv_scale = 1.0f / __half2float(scale_hp);
    __half* scales = reinterpret_cast<__half*>(
        reinterpret_cast<unsigned char*>(dst_h_head) + QWEN4_GDN_FP8_SCALES_OFFSET
    );
    qwen4_gdn_st_global_half(&scales[tid], scale_hp);

    const unsigned long long pol_ef = qwen4_gdn_l2_evict_first_policy();
    uint4* dst16_base =
        reinterpret_cast<uint4*>(dst_h_head) + (unsigned long long)r16 * 8ull + warp * 2u + c16;
    unsigned char* wtile = reinterpret_cast<unsigned char*>(smem_h_tile[warp]);

    #pragma unroll 1
    for (unsigned int jb = 0u; jb < 8u; jb++) {
        #pragma unroll 4
        for (unsigned int k = 0u; k < 16u; k += 2u) {
            const unsigned int row0 = jb * 16u + k;
            const float v0 = qwen4_gdn_ld_global_cg_f32(
                &src_scratch_h_head[row0 * QWEN4_GDN_DIM + tid]
            );
            const float v1 = qwen4_gdn_ld_global_cg_f32(
                &src_scratch_h_head[(row0 + 1u) * QWEN4_GDN_DIM + tid]
            );
            const unsigned short packed = qwen4_gdn_cvt_f32x2_to_e4m3x2(
                v0 * inv_scale, v1 * inv_scale
            );
            wtile[k * 32u + lane] = static_cast<unsigned char>(packed & 0xffu);
            wtile[(k + 1u) * 32u + lane] = static_cast<unsigned char>(packed >> 8);
        }
        __syncwarp();
        uint4 v = qwen4_gdn_lds_u4_volatile(&smem_h_tile[warp][lane]);
        qwen4_gdn_st_global_na_ef_u4(dst16_base + (unsigned long long)jb * 128ull, v, pol_ef);
        __syncwarp();
    }
}

// Last-arriving CTA finalizer: unpack the full [128 x 128] FP32 head staged
// across all 4 `sv` CTAs in L2 back into canonical FP32 `dst_h_head`.
__device__ __forceinline__ void qwen4_gdn_finalize_h_f32_from_scratch(
    float* dst_h_head,
    const float* src_scratch_h_head
) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long pol_ef = qwen4_gdn_l2_evict_first_policy();
    const float4* src4 = reinterpret_cast<const float4*>(src_scratch_h_head);
    float4* dst4 = reinterpret_cast<float4*>(dst_h_head);
    #pragma unroll 4
    for (unsigned int it = 0u; it < 32u; it++) {
        const unsigned int idx4 = it * 128u + tid;
        const float4 v = qwen4_gdn_ld_global_cg_f4(&src4[idx4]);
        qwen4_gdn_st_global_na_ef_f4(&dst4[idx4], v, pol_ef);
    }
}

// Load [128 x 32] FP32 V-slice `sv` of H[128 x 128] into warp-sharded
// H_reg[32] (warp w owns K-rows [w*32 .. w*32+31], lane owns V-col sv*32+lane)
// via 8x 128-bit streaming loads per warp (4-deep rolling window = 8 KiB/CTA)
// and zero-bank-conflict warp-local smem transpose (0 __syncthreads()).
__device__ __forceinline__ void qwen4_gdn_load_h_f4_splitv4(
    const float* H,
    float H_reg[QWEN4_GDN_SPLIT_K_PER_WARP],
    float4 smem_h_tile[4][32],
    unsigned int sv
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int r4 = lane >> 3;
    const unsigned int c4 = lane & 7u;
    const unsigned long long pol_ef = qwen4_gdn_l2_evict_first_policy();
    const float4* H4_base =
        reinterpret_cast<const float4*>(H)
        + (unsigned long long)(warp * 32u + r4) * 32ull
        + (unsigned long long)sv * 8ull
        + c4;

    float4 tmp[4];
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        tmp[b] = qwen4_gdn_ld_global_na_ef_f4(
            H4_base + (unsigned long long)b * 128ull, pol_ef
        );
    }
    #pragma unroll
    for (int jb = 0; jb < 8; jb++) {
        const int slot = jb & 3;
        qwen4_gdn_sts_f4_volatile(&smem_h_tile[warp][lane], tmp[slot]);
        __syncwarp();
        const float* wtile = reinterpret_cast<const float*>(smem_h_tile[warp]);
        H_reg[jb * 4 + 0] = wtile[0 * 32 + lane];
        H_reg[jb * 4 + 1] = wtile[1 * 32 + lane];
        H_reg[jb * 4 + 2] = wtile[2 * 32 + lane];
        H_reg[jb * 4 + 3] = wtile[3 * 32 + lane];
        __syncwarp();
        if (jb + 4 < 8) {
            tmp[slot] = qwen4_gdn_ld_global_na_ef_f4(
                H4_base + (unsigned long long)(jb + 4) * 128ull, pol_ef
            );
        }
    }
}

// Store warp-sharded H_reg[32] back to [128 x 32] FP32 V-slice `sv` of
// dst[128 x 128] via warp-local smem transpose + 8x 128-bit streaming stores.
__device__ __forceinline__ void qwen4_gdn_store_h_f4_splitv4(
    float* dst,
    const float H_reg[QWEN4_GDN_SPLIT_K_PER_WARP],
    float4 smem_h_tile[4][32],
    unsigned int sv
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int r4 = lane >> 3;
    const unsigned int c4 = lane & 7u;
    const unsigned long long pol_ef = qwen4_gdn_l2_evict_first_policy();
    float4* dst4_base =
        reinterpret_cast<float4*>(dst)
        + (unsigned long long)(warp * 32u + r4) * 32ull
        + (unsigned long long)sv * 8ull
        + c4;

    #pragma unroll
    for (int jb = 0; jb < 8; jb++) {
        float* wtile = reinterpret_cast<float*>(smem_h_tile[warp]);
        wtile[0 * 32 + lane] = H_reg[jb * 4 + 0];
        wtile[1 * 32 + lane] = H_reg[jb * 4 + 1];
        wtile[2 * 32 + lane] = H_reg[jb * 4 + 2];
        wtile[3 * 32 + lane] = H_reg[jb * 4 + 3];
        __syncwarp();
        float4 v = qwen4_gdn_lds_f4_volatile_f4(&smem_h_tile[warp][lane]);
        qwen4_gdn_st_global_na_ef_f4(dst4_base + (unsigned long long)jb * 128ull, v, pol_ef);
        __syncwarp();
    }
}

template <
    bool LAZY_FINAL,
    bool SNAPSHOT_ALL,
    bool MULTI_RETAIN_GATES = false,
    bool ALLOW_SCALED_FP16 = false
>
__device__ __forceinline__ void qwen4_gdn_fused_splitv4_body(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ deinterleaved,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16* __restrict__ normed_out,
    float* __restrict__ retain_qkv,
    float* __restrict__ conv_inter,
    float* __restrict__ h_state_inter,
    unsigned int* __restrict__ group_done,
    float* __restrict__ scratch_gdn,
    float* __restrict__ scratch_ss,
    unsigned int* __restrict__ split_done_seq,
    unsigned int num_tokens,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    float l2_eps,
    float rms_eps,
    unsigned int input_stride,
    unsigned int gate_beta_stride,
    unsigned int normed_stride,
    unsigned int retain_stride,
    unsigned int conv_inter_stride,
    unsigned int h_inter_stride,
    const GdnMultiSeqSmem* __restrict__ ms_smem = nullptr,
    float* __restrict__ scratch_h_seq = nullptr,
    float* __restrict__ scratch_amax_seq = nullptr
) {
    #define SV_CONV_STATE()     (ms_smem ? ms_smem->conv_state     : conv_state)
    #define SV_H_STATE()        (ms_smem ? ms_smem->h_state        : h_state)
    #define SV_RETAIN_QKV()     (ms_smem ? ms_smem->retain_qkv     : retain_qkv)
    #define SV_CONV_INTER()     (ms_smem ? ms_smem->conv_inter     : conv_inter)
    #define SV_H_STATE_INTER()  (ms_smem ? ms_smem->h_state_inter  : h_state_inter)
    #define SV_GROUP_DONE()     (ms_smem ? ms_smem->group_done     : group_done)
    #define SV_NUM_TOKENS()     (ms_smem ? ms_smem->num_tokens     : num_tokens)
    #define SV_COPY_GATES()     (ms_smem ? ((ms_smem->flags & GDN_FUSED_FLAG_COPY_GATES_TO_RETAIN) != 0u) : false)
    #define SV_KEEP_LATE()      (ms_smem ? ((ms_smem->flags & GDN_FUSED_FLAG_KEEP_LATE) != 0u) : false)
    #define SV_LOAD_SCALED_FP16()  (ms_smem ? ((ms_smem->flags & GDN_FUSED_FLAG_LOAD_SCALED_FP16) != 0u) : false)
    #define SV_STORE_SCALED_FP16() (ms_smem ? ((ms_smem->flags & GDN_FUSED_FLAG_STORE_SCALED_FP16) != 0u) : false)
    #define SV_LOAD_SCALED_FP8()   (ms_smem ? ((ms_smem->flags & GDN_FUSED_FLAG_LOAD_SCALED_FP8) != 0u) : false)
    #define SV_STORE_SCALED_FP8()  (ms_smem ? ((ms_smem->flags & GDN_FUSED_FLAG_STORE_SCALED_FP8) != 0u) : false)
    #define SV_PENDING_BASE()   (ms_smem ? ms_smem->pending_base   : 0u)
    #define SV_PENDING_ROWS()   (ms_smem ? ms_smem->pending_rows   : 0u)
    #define SV_WRITE_BASE()     (ms_smem ? ms_smem->write_base     : 0u)
    #define SV_RING_CAP()       (ms_smem ? ms_smem->ring_cap       : 0u)
    #define SV_FLUSH_NOW()      (ms_smem ? (ms_smem->flush_now != 0u) : false)
    #define SV_ROW_BASE()       (ms_smem ? (unsigned long long)ms_smem->row_base : 0ull)

    const unsigned int vh = blockIdx.x;
    const unsigned int sv = blockIdx.y;
    if (vh >= num_v_heads || sv >= QWEN4_GDN_SPLIT_V) return;
    if (SV_NUM_TOKENS() == 0u) {
        if (!(LAZY_FINAL && MULTI_RETAIN_GATES && SV_KEEP_LATE()
              && SV_FLUSH_NOW()
              && (SV_PENDING_ROWS() > 0u
                  || (ALLOW_SCALED_FP16 && (SV_LOAD_SCALED_FP16() || SV_LOAD_SCALED_FP8()))))) {
            return;
        }
    }

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;

    __shared__ GdnCtaGeom geom;
    __shared__ float warp_sums_q[4];
    __shared__ float warp_sums_k[4];
    __shared__ __align__(16) float smem_k[2][QWEN4_GDN_DIM];
    __shared__ __align__(16) float smem_q[QWEN4_GDN_DIM];
    __shared__ __align__(16) float smem_v[2][32];
    __shared__ __align__(16) float smem_reduce_hk[4][32];
    __shared__ __align__(16) float smem_reduce_q[4][32];
    __shared__ unsigned int is_last_cta;
    __shared__ __align__(16) float4 smem_h_tile[4][32];
    __shared__ __align__(16) float4 smem_conv_q[QWEN4_GDN_DIM];
    __shared__ __align__(16) float4 smem_conv_k[QWEN4_GDN_DIM];
    __shared__ __align__(16) float4 smem_conv_v[32];

    if (tid == 0u) {
        const unsigned int hr = num_v_heads / num_k_heads;
        const unsigned int kh = vh / hr;
        const unsigned int key_dim = num_k_heads * QWEN4_GDN_DIM;
        const unsigned int val_dim = num_v_heads * QWEN4_GDN_DIM;
        const unsigned int conv_dim = 2u * key_dim + val_dim;
        const unsigned int gates_off = conv_dim * 2u;
        geom.head_repeat = hr;
        geom.kh = kh;
        geom.sub = vh - kh * hr;
        geom.q_base = kh * QWEN4_GDN_DIM;
        geom.k_base = key_dim + kh * QWEN4_GDN_DIM;
        geom.v_base = 2u * key_dim + vh * QWEN4_GDN_DIM;
        geom.z_base = conv_dim + vh * QWEN4_GDN_DIM;
        geom.ring_gates_off = gates_off;
        geom.ring_row_bytes = (gates_off + num_v_heads * 8u + 15u) & ~15u;
        geom.inv_sqrt_d = rsqrtf((float)blockDim.x);
    }
    __syncthreads();

    // Step 1: Load initial 4-tap conv1d state. Q/K cover all 128 channels;
    // V covers only this CTA's disjoint 32-channel slice [sv*32 .. sv*32+32).
    // Skip on pure `num_tokens == 0 && pending_rows == 0` Scaled-FP16 -> FP32 unpacks.
    if (!(ALLOW_SCALED_FP16 && LAZY_FINAL && MULTI_RETAIN_GATES && SV_KEEP_LATE())
        || SV_NUM_TOKENS() > 0u || SV_PENDING_ROWS() > 0u) {
        const float4* cs4 = reinterpret_cast<const float4*>(SV_CONV_STATE());
        qwen4_gdn_sts_f4_volatile(&smem_conv_q[tid], cs4[geom.q_base + tid]);
        qwen4_gdn_sts_f4_volatile(&smem_conv_k[tid], cs4[geom.k_base + tid]);
        if (tid < 32u) {
            qwen4_gdn_sts_f4_volatile(&smem_conv_v[tid], cs4[geom.v_base + sv * 32u + tid]);
        }
    }

    // Step 2: Load [128 x 32] V-slice `sv` of H into warp-sharded H_reg[32].
    float H_reg[QWEN4_GDN_SPLIT_K_PER_WARP];
    if (ALLOW_SCALED_FP16 && LAZY_FINAL && MULTI_RETAIN_GATES && SV_KEEP_LATE()
        && SV_LOAD_SCALED_FP8()) {
        qwen4_gdn_load_h_scaled_fp8_splitv4(
            SV_H_STATE() + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM),
            H_reg,
            smem_h_tile,
            sv
        );
    } else if (ALLOW_SCALED_FP16 && LAZY_FINAL && MULTI_RETAIN_GATES && SV_KEEP_LATE()
        && SV_LOAD_SCALED_FP16()) {
        qwen4_gdn_load_h_scaled_fp16_splitv4(
            SV_H_STATE() + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM),
            H_reg,
            smem_h_tile,
            sv
        );
    } else {
        qwen4_gdn_load_h_f4_splitv4(
            SV_H_STATE() + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM),
            H_reg,
            smem_h_tile,
            sv
        );
    }

    bool deferred_h_flush = false;
    // Step 2b (KEEP_LATE compact circular ring replay + optional flush).
    if (LAZY_FINAL && MULTI_RETAIN_GATES && SV_KEEP_LATE()) {
        const bool has_pending =
            SV_PENDING_ROWS() > 0u && SV_RING_CAP() > 0u && SV_RETAIN_QKV() != nullptr;
        if (has_pending) {
            #pragma unroll 1
            for (unsigned int pt = 0u; pt < SV_PENDING_ROWS(); pt++) {
                float g_prev, bt_prev;
                {
                    const unsigned int cap = SV_RING_CAP();
                    unsigned int r_idx = SV_PENDING_BASE() + pt;
                    if (r_idx >= cap) r_idx -= cap;

                    const unsigned char* row_ptr =
                        reinterpret_cast<const unsigned char*>(SV_RETAIN_QKV())
                        + (unsigned long long)r_idx * geom.ring_row_bytes;
                    const __nv_bfloat16* prev_raw = reinterpret_cast<const __nv_bfloat16*>(row_ptr);

                    {
                        const unsigned int q_ch = geom.q_base + tid;
                        float4 sq = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_q[tid]);
                        sq.x = sq.y; sq.y = sq.z; sq.z = sq.w; sq.w = (float)__ldg(&prev_raw[q_ch]);
                        qwen4_gdn_sts_f4_volatile(&smem_conv_q[tid], sq);
                    }

                    float silu_k;
                    {
                        const unsigned int k_ch = geom.k_base + tid;
                        float4 sk = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_k[tid]);
                        sk.x = sk.y; sk.y = sk.z; sk.z = sk.w; sk.w = (float)__ldg(&prev_raw[k_ch]);
                        qwen4_gdn_sts_f4_volatile(&smem_conv_k[tid], sk);
                        silu_k = qwen4_gdn_conv1d_silu_4tap(sk, conv_weight, conv_bias, k_ch);
                    }

                    if (tid < 32u) {
                        const unsigned int v_ch = geom.v_base + sv * 32u + tid;
                        float4 sv_tap = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_v[tid]);
                        sv_tap.x = sv_tap.y; sv_tap.y = sv_tap.z; sv_tap.z = sv_tap.w;
                        sv_tap.w = (float)__ldg(&prev_raw[v_ch]);
                        qwen4_gdn_sts_f4_volatile(&smem_conv_v[tid], sv_tap);
                        smem_v[pt & 1u][tid] = qwen4_gdn_conv1d_silu_4tap(
                            sv_tap, conv_weight, conv_bias, v_ch
                        );
                    }

                    asm volatile("" : "+f"(silu_k));
                    float sq_sum_k = silu_k * silu_k;
                    #pragma unroll
                    for (int offset = 16; offset >= 1; offset >>= 1) {
                        sq_sum_k += __shfl_down_sync(0xffffffffu, sq_sum_k, offset);
                    }
                    if (lane == 0u) {
                        warp_sums_k[warp] = sq_sum_k;
                    }
                    __syncthreads();
                    const float total_k =
                        ((warp_sums_k[0] + warp_sums_k[1]) + warp_sums_k[2]) + warp_sums_k[3];
                    smem_k[pt & 1u][tid] = silu_k * rsqrtf(total_k + l2_eps);

                    const float* prev_gates =
                        reinterpret_cast<const float*>(row_ptr + geom.ring_gates_off);
                    g_prev = fminf(fmaxf(__ldg(&prev_gates[vh]), 0.0f), 1.0f);
                    bt_prev = __ldg(&prev_gates[num_v_heads + vh]);
                    __syncthreads();
                }

                const float* sk_buf = &smem_k[pt & 1u][warp * 32u];
                float hk_part = 0.0f;
                #pragma unroll
                for (int j = 0; j < QWEN4_GDN_SPLIT_K_PER_WARP; j += 4) {
                    const float4 kv = qwen4_gdn_lds_f4_volatile(&sk_buf[j]);
                    hk_part += H_reg[j] * kv.x
                             + H_reg[j + 1] * kv.y
                             + H_reg[j + 2] * kv.z
                             + H_reg[j + 3] * kv.w;
                }
                smem_reduce_hk[warp][lane] = hk_part;
                __syncthreads();
                const float hk_dot =
                    (smem_reduce_hk[0][lane] + smem_reduce_hk[1][lane])
                    + (smem_reduce_hk[2][lane] + smem_reduce_hk[3][lane]);
                const float v_new_i = (smem_v[pt & 1u][lane] - g_prev * hk_dot) * bt_prev;
                #pragma unroll
                for (int j = 0; j < QWEN4_GDN_SPLIT_K_PER_WARP; j += 4) {
                    const float4 kv = qwen4_gdn_lds_f4_volatile(&sk_buf[j]);
                    H_reg[j]     = g_prev * H_reg[j]     + kv.x * v_new_i;
                    H_reg[j + 1] = g_prev * H_reg[j + 1] + kv.y * v_new_i;
                    H_reg[j + 2] = g_prev * H_reg[j + 2] + kv.z * v_new_i;
                    H_reg[j + 3] = g_prev * H_reg[j + 3] + kv.w * v_new_i;
                }
            }
        }

        if (SV_FLUSH_NOW()
            && (has_pending || (ALLOW_SCALED_FP16 && (SV_LOAD_SCALED_FP16() || SV_LOAD_SCALED_FP8())))) {
            if (has_pending) {
                float4* cs4 = reinterpret_cast<float4*>(SV_CONV_STATE());
                if (tid < 32u) {
                    cs4[geom.v_base + sv * 32u + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_v[tid]);
                }
                const unsigned int group_wrap = geom.head_repeat * QWEN4_GDN_SPLIT_V - 1u;
                if (tid == 0u) {
                    const unsigned int prev = qwen4_gdn_group_arrive_last(
                        SV_GROUP_DONE() + geom.kh, group_wrap
                    );
                    is_last_cta = (prev == group_wrap) ? 1u : 0u;
                }
                __syncthreads();
                if (is_last_cta != 0u) {
                    cs4[geom.q_base + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_q[tid]);
                    cs4[geom.k_base + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_k[tid]);
                }
            }
            if (ALLOW_SCALED_FP16
                && (SV_LOAD_SCALED_FP16() || SV_STORE_SCALED_FP16()
                    || SV_LOAD_SCALED_FP8() || SV_STORE_SCALED_FP8())) {
                // Stage this CTA's flushed [128 x 32] FP32 slice into L2-coherent
                // scratch so concurrent `sv` CTAs finish Step 2 reads of `h_state`
                // before the 4th CTA writes the in-place Scaled-FP8 / Scaled-FP16 /
                // FP32 result in Step 4 (zero spin-wait deadlock, 0-ULP scale match).
                float* scratch_h_head =
                    scratch_h_seq + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM);
                qwen4_gdn_store_h_f4_splitv4_cg(scratch_h_head, H_reg, smem_h_tile, sv);
                if (SV_STORE_SCALED_FP16() && !SV_STORE_SCALED_FP8()) {
                    float local_max = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < QWEN4_GDN_SPLIT_K_PER_WARP; j += 4) {
                        local_max = fmaxf(local_max, fabsf(H_reg[j]));
                        local_max = fmaxf(local_max, fabsf(H_reg[j + 1]));
                        local_max = fmaxf(local_max, fabsf(H_reg[j + 2]));
                        local_max = fmaxf(local_max, fabsf(H_reg[j + 3]));
                    }
                    #pragma unroll
                    for (int d = 16; d > 0; d >>= 1) {
                        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, d));
                    }
                    if (lane == 0u) {
                        warp_sums_k[warp] = local_max;
                    }
                    __syncthreads();
                    if (tid == 0u) {
                        const float sv_amax = fmaxf(
                            fmaxf(warp_sums_k[0], warp_sums_k[1]),
                            fmaxf(warp_sums_k[2], warp_sums_k[3])
                        );
                        qwen4_gdn_st_global_cg_f32(
                            &scratch_amax_seq[vh * QWEN4_GDN_SPLIT_V + sv], sv_amax
                        );
                    }
                    __syncthreads();
                }
                deferred_h_flush = true;
            } else {
                float* h_head =
                    SV_H_STATE() + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM);
                qwen4_gdn_store_h_f4_splitv4(h_head, H_reg, smem_h_tile, sv);
            }
        }
    }

    const unsigned int ntok = SV_NUM_TOKENS();
    #pragma unroll 1
    for (unsigned int token = 0u; token < ntok; token++) {
        float g, bt;
        {
            const unsigned long long row_idx = SV_ROW_BASE() + (unsigned long long)token;
            const __nv_bfloat16* row_in = deinterleaved + row_idx * input_stride;
            float* rqkv = SV_RETAIN_QKV();
            float* ci = SV_CONV_INTER();
            const bool is_keep_late = LAZY_FINAL && MULTI_RETAIN_GATES && SV_KEEP_LATE();
            const bool write_ring = is_keep_late && rqkv != nullptr && SV_RING_CAP() > 0u;

            #define SV_RING_DST_ROW() ({ \
                const unsigned int _cap = SV_RING_CAP(); \
                unsigned int _w_idx = SV_WRITE_BASE() + token; \
                if (_w_idx >= _cap) _w_idx -= _cap; \
                reinterpret_cast<unsigned char*>(rqkv) \
                    + (unsigned long long)_w_idx * geom.ring_row_bytes; \
            })

            if (write_ring && vh == 0u && sv == 0u) {
                unsigned char* dst_row = SV_RING_DST_ROW();
                float* dst_gates = reinterpret_cast<float*>(dst_row + geom.ring_gates_off);
                const unsigned long long gb_row = row_idx * gate_beta_stride;
                const unsigned int two_nv = num_v_heads * 2u;
                for (unsigned int idx = tid; idx < two_nv; idx += 128u) {
                    const float val = (idx < num_v_heads)
                        ? __ldg(&gate[gb_row + idx])
                        : __ldg(&beta[gb_row + (idx - num_v_heads)]);
                    qwen4_gdn_st_global_f32(&dst_gates[idx], val);
                }
            }

            float silu_q;
            {
                const unsigned int q_ch = geom.q_base + tid;
                const __nv_bfloat16 raw_q = __ldg(&row_in[q_ch]);
                if (write_ring && geom.sub == 0u && sv == 0u) {
                    qwen4_gdn_st_global_u16(
                        &reinterpret_cast<__nv_bfloat16*>(SV_RING_DST_ROW())[q_ch], raw_q
                    );
                }
                float4 sq = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_q[tid]);
                sq.x = sq.y; sq.y = sq.z; sq.z = sq.w; sq.w = (float)raw_q;
                qwen4_gdn_sts_f4_volatile(&smem_conv_q[tid], sq);
                if (ci != nullptr && geom.sub == 0u && sv == 0u) {
                    float4* snap_base = reinterpret_cast<float4*>(
                        ci + (unsigned long long)token * conv_inter_stride
                    );
                    qwen4_gdn_st_global_cs_f4(&snap_base[q_ch], sq);
                }
                silu_q = qwen4_gdn_conv1d_silu_4tap(sq, conv_weight, conv_bias, q_ch);
            }

            float silu_k;
            {
                const unsigned int k_ch = geom.k_base + tid;
                const __nv_bfloat16 raw_k = __ldg(&row_in[k_ch]);
                if (write_ring && geom.sub == 0u && sv == 0u) {
                    qwen4_gdn_st_global_u16(
                        &reinterpret_cast<__nv_bfloat16*>(SV_RING_DST_ROW())[k_ch], raw_k
                    );
                }
                float4 sk = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_k[tid]);
                sk.x = sk.y; sk.y = sk.z; sk.z = sk.w; sk.w = (float)raw_k;
                qwen4_gdn_sts_f4_volatile(&smem_conv_k[tid], sk);
                if (ci != nullptr && geom.sub == 0u && sv == 0u) {
                    float4* snap_base = reinterpret_cast<float4*>(
                        ci + (unsigned long long)token * conv_inter_stride
                    );
                    qwen4_gdn_st_global_cs_f4(&snap_base[k_ch], sk);
                }
                silu_k = qwen4_gdn_conv1d_silu_4tap(sk, conv_weight, conv_bias, k_ch);
            }

            float v_val_local = 0.0f;
            if (tid < 32u) {
                const unsigned int v_ch = geom.v_base + sv * 32u + tid;
                const __nv_bfloat16 raw_v = __ldg(&row_in[v_ch]);
                if (write_ring) {
                    qwen4_gdn_st_global_u16(
                        &reinterpret_cast<__nv_bfloat16*>(SV_RING_DST_ROW())[v_ch], raw_v
                    );
                }
                float4 sv_tap = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_v[tid]);
                sv_tap.x = sv_tap.y; sv_tap.y = sv_tap.z; sv_tap.z = sv_tap.w;
                sv_tap.w = (float)raw_v;
                qwen4_gdn_sts_f4_volatile(&smem_conv_v[tid], sv_tap);
                if (ci != nullptr) {
                    float4* snap_base = reinterpret_cast<float4*>(
                        ci + (unsigned long long)token * conv_inter_stride
                    );
                    qwen4_gdn_st_global_cs_f4(&snap_base[v_ch], sv_tap);
                }
                v_val_local = qwen4_gdn_conv1d_silu_4tap(
                    sv_tap, conv_weight, conv_bias, v_ch
                );
                smem_v[token & 1u][tid] = v_val_local;
            }
            #undef SV_RING_DST_ROW

            // 2. Per-head Q/K L2 normalization (exact match with baseline).
            asm volatile("" : "+f"(silu_q), "+f"(silu_k));
            float sq_sum_q = silu_q * silu_q;
            float sq_sum_k = silu_k * silu_k;
            #pragma unroll
            for (int offset = 16; offset >= 1; offset >>= 1) {
                sq_sum_q += __shfl_down_sync(0xffffffffu, sq_sum_q, offset);
                sq_sum_k += __shfl_down_sync(0xffffffffu, sq_sum_k, offset);
            }
            if (lane == 0u) {
                warp_sums_q[warp] = sq_sum_q;
                warp_sums_k[warp] = sq_sum_k;
            }
            __syncthreads();

            const float total_q =
                ((warp_sums_q[0] + warp_sums_q[1]) + warp_sums_q[2]) + warp_sums_q[3];
            const float total_k =
                ((warp_sums_k[0] + warp_sums_k[1]) + warp_sums_k[2]) + warp_sums_k[3];
            const float q_val = silu_q * rsqrtf(total_q + l2_eps);
            const float k_val = silu_k * rsqrtf(total_k + l2_eps);

            smem_q[tid] = q_val;
            smem_k[0][tid] = k_val;

            if (LAZY_FINAL && !is_keep_late && rqkv != nullptr) {
                float* retain_row = rqkv + (unsigned long long)token * retain_stride;
                if (tid < 32u) {
                    qwen4_gdn_st_global_cs_f32(&retain_row[geom.v_base + sv * 32u + tid], v_val_local);
                }
                if (geom.sub == 0u && sv == 0u) {
                    qwen4_gdn_st_global_cs_f32(&retain_row[geom.q_base + tid], q_val);
                    qwen4_gdn_st_global_cs_f32(&retain_row[geom.k_base + tid], k_val);
                }
                if (MULTI_RETAIN_GATES && SV_COPY_GATES() && vh == 0u && sv == 0u) {
                    const unsigned long long gb_row = row_idx * gate_beta_stride;
                    float* retain_gates = rqkv
                        + (unsigned long long)QWEN4_GDN_LAZY_MAX_K * retain_stride
                        + (unsigned long long)token * gate_beta_stride;
                    const unsigned int two_nv = num_v_heads * 2u;
                    for (unsigned int idx = tid; idx < two_nv; idx += 128u) {
                        const float val = (idx < num_v_heads)
                            ? __ldg(&gate[gb_row + idx])
                            : __ldg(&beta[gb_row + (idx - num_v_heads)]);
                        qwen4_gdn_st_global_cs_f32(&retain_gates[idx], val);
                    }
                }
            }

            const unsigned long long gb_off = row_idx * gate_beta_stride + vh;
            g = fminf(fmaxf(__ldg(&gate[gb_off]), 0.0f), 1.0f);
            bt = __ldg(&beta[gb_off]);
            __syncthreads();
        }

        // 3. 4-warp K-sharded GDN recurrence over H_reg[32].
        const float* sk_warp = &smem_k[0][warp * 32u];
        const float* sq_warp = &smem_q[warp * 32u];
        float hk_part = 0.0f;
        #pragma unroll
        for (int j = 0; j < QWEN4_GDN_SPLIT_K_PER_WARP; j += 4) {
            const float4 kv = qwen4_gdn_lds_f4_volatile(&sk_warp[j]);
            hk_part += H_reg[j] * kv.x
                     + H_reg[j + 1] * kv.y
                     + H_reg[j + 2] * kv.z
                     + H_reg[j + 3] * kv.w;
        }
        smem_reduce_hk[warp][lane] = hk_part;
        __syncthreads();

        const float hk_dot =
            (smem_reduce_hk[0][lane] + smem_reduce_hk[1][lane])
            + (smem_reduce_hk[2][lane] + smem_reduce_hk[3][lane]);
        const float v_new_i = (smem_v[token & 1u][lane] - g * hk_dot) * bt;

        float q_part = 0.0f;
        #pragma unroll
        for (int j = 0; j < QWEN4_GDN_SPLIT_K_PER_WARP; j += 4) {
            const float4 kv = qwen4_gdn_lds_f4_volatile(&sk_warp[j]);
            const float4 qv = qwen4_gdn_lds_f4_volatile(&sq_warp[j]);
            float h0 = g * H_reg[j]     + kv.x * v_new_i;
            float h1 = g * H_reg[j + 1] + kv.y * v_new_i;
            float h2 = g * H_reg[j + 2] + kv.z * v_new_i;
            float h3 = g * H_reg[j + 3] + kv.w * v_new_i;
            H_reg[j]     = h0;
            H_reg[j + 1] = h1;
            H_reg[j + 2] = h2;
            H_reg[j + 3] = h3;
            q_part += h0 * qv.x
                    + h1 * qv.y
                    + h2 * qv.z
                    + h3 * qv.w;
        }
        smem_reduce_q[warp][lane] = q_part;
        __syncthreads();

        // Warp 0 reduces q_dot across the 4 K-warps, computes this 32-col
        // partial sum-of-squares, and publishes to L2-coherent scratch.
        if (warp == 0u) {
            const float q_dot =
                (smem_reduce_q[0][lane] + smem_reduce_q[1][lane])
                + (smem_reduce_q[2][lane] + smem_reduce_q[3][lane]);
            const float gdn_val = q_dot * geom.inv_sqrt_d;
            const float rms_sq = qwen4_gdn_warp_sum_xor(gdn_val * gdn_val);
            const unsigned long long row_idx = SV_ROW_BASE() + (unsigned long long)token;
            const unsigned long long head_row = row_idx * (unsigned long long)num_v_heads + vh;
            qwen4_gdn_st_global_cg_f32(
                &scratch_gdn[head_row * QWEN4_GDN_DIM + sv * 32u + lane], gdn_val
            );
            if (lane == 0u) {
                qwen4_gdn_st_global_cg_f32(&scratch_ss[head_row * QWEN4_GDN_SPLIT_V + sv], rms_sq);
            }
        }
        if (SNAPSHOT_ALL) {
            float* snapshot = SV_H_STATE_INTER()
                + (unsigned long long)token * h_inter_stride
                + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM);
            qwen4_gdn_store_h_f4_splitv4(snapshot, H_reg, smem_h_tile, sv);
        }
    }

    // Commit final H state for this CTA's 32 V-columns.
    if (LAZY_FINAL) {
        if (!(MULTI_RETAIN_GATES && SV_KEEP_LATE())) {
            float* fin = SV_H_STATE_INTER()
                + (unsigned long long)(ntok - 1u) * h_inter_stride
                + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM);
            qwen4_gdn_store_h_f4_splitv4(fin, H_reg, smem_h_tile, sv);
        }
    } else {
        qwen4_gdn_store_h_f4_splitv4(
            SV_H_STATE() + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM),
            H_reg,
            smem_h_tile,
            sv
        );
    }

    // Commit live conv_state unless LAZY_FINAL + KEEP_LATE is active.
    if (!(LAZY_FINAL && MULTI_RETAIN_GATES && SV_KEEP_LATE())) {
        float4* cs4 = reinterpret_cast<float4*>(SV_CONV_STATE());
        if (tid < 32u) {
            cs4[geom.v_base + sv * 32u + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_v[tid]);
        }
        const unsigned int group_wrap = geom.head_repeat * QWEN4_GDN_SPLIT_V - 1u;
        if (tid == 0u) {
            const unsigned int prev = qwen4_gdn_group_arrive_last(
                SV_GROUP_DONE() + geom.kh, group_wrap
            );
            is_last_cta = (prev == group_wrap) ? 1u : 0u;
        }
        __syncthreads();
        if (is_last_cta != 0u) {
            cs4[geom.q_base + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_q[tid]);
            cs4[geom.k_base + tid] = qwen4_gdn_lds_f4_volatile_f4(&smem_conv_k[tid]);
        }
    }

    // Step 4: Cross-CTA RMS norm + sigmoid Z-gate finalize (and deferred
    // Scaled-FP16 / FP32 in-place `h_state` flush finalize when scheduled).
    // All 4 `sv` CTAs arrive on self-resetting `split_done_seq[vh]` (wrap = 3);
    // only the 4th CTA normalizes all 128 V-columns and commits `h_state`.
    if (ntok > 0u || (ALLOW_SCALED_FP16 && deferred_h_flush)) {
        __syncthreads();
        if (tid == 0u) {
            const unsigned int prev = qwen4_gdn_group_arrive_last(
                split_done_seq + vh, QWEN4_GDN_SPLIT_V - 1u
            );
            is_last_cta = (prev == QWEN4_GDN_SPLIT_V - 1u) ? 1u : 0u;
        }
        __syncthreads();
        if (is_last_cta != 0u) {
            if (ntok > 0u) {
                const float rms_w = __bfloat162float(__ldg(&rms_weight[tid]));
                #pragma unroll 1
                for (unsigned int token = 0u; token < ntok; token++) {
                    const unsigned long long row_idx = SV_ROW_BASE() + (unsigned long long)token;
                    const unsigned long long head_row = row_idx * (unsigned long long)num_v_heads + vh;
                    const float gdn_val =
                        qwen4_gdn_ld_global_cg_f32(&scratch_gdn[head_row * QWEN4_GDN_DIM + tid]);
                    const float s0 =
                        qwen4_gdn_ld_global_cg_f32(&scratch_ss[head_row * QWEN4_GDN_SPLIT_V + 0ull]);
                    const float s1 =
                        qwen4_gdn_ld_global_cg_f32(&scratch_ss[head_row * QWEN4_GDN_SPLIT_V + 1ull]);
                    const float s2 =
                        qwen4_gdn_ld_global_cg_f32(&scratch_ss[head_row * QWEN4_GDN_SPLIT_V + 2ull]);
                    const float s3 =
                        qwen4_gdn_ld_global_cg_f32(&scratch_ss[head_row * QWEN4_GDN_SPLIT_V + 3ull]);
                    const float v_part = (s0 + s2) + (s1 + s3);
                    const float inv_rms = rsqrtf(v_part / (float)QWEN4_GDN_DIM + rms_eps);
                    const float z_val = __bfloat162float(
                        __ldg(&deinterleaved[row_idx * input_stride + geom.z_base + tid])
                    );
                    normed_out[row_idx * normed_stride + vh * QWEN4_GDN_DIM + tid] =
                        __float2bfloat16(gdn_val * inv_rms * rms_w / (1.0f + expf(-z_val)));
                }
            }
            if (ALLOW_SCALED_FP16 && deferred_h_flush) {
                float* h_head =
                    SV_H_STATE() + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM);
                const float* scratch_h_head =
                    scratch_h_seq + (unsigned long long)vh * (QWEN4_GDN_DIM * QWEN4_GDN_DIM);
                if (SV_STORE_SCALED_FP8()) {
                    qwen4_gdn_finalize_h_scaled_fp8_from_scratch(
                        h_head, scratch_h_head, smem_h_tile
                    );
                } else if (SV_STORE_SCALED_FP16()) {
                    qwen4_gdn_finalize_h_scaled_fp16_from_scratch(
                        h_head, scratch_h_head, &scratch_amax_seq[vh * QWEN4_GDN_SPLIT_V]
                    );
                } else {
                    qwen4_gdn_finalize_h_f32_from_scratch(h_head, scratch_h_head);
                }
            }
        }
    }

    #undef SV_CONV_STATE
    #undef SV_H_STATE
    #undef SV_RETAIN_QKV
    #undef SV_CONV_INTER
    #undef SV_H_STATE_INTER
    #undef SV_GROUP_DONE
    #undef SV_NUM_TOKENS
    #undef SV_COPY_GATES
    #undef SV_KEEP_LATE
    #undef SV_LOAD_SCALED_FP16
    #undef SV_STORE_SCALED_FP16
    #undef SV_LOAD_SCALED_FP8
    #undef SV_STORE_SCALED_FP8
    #undef SV_PENDING_BASE
    #undef SV_PENDING_ROWS
    #undef SV_WRITE_BASE
    #undef SV_RING_CAP
    #undef SV_FLUSH_NOW
    #undef SV_ROW_BASE
}

extern "C" __global__ void __launch_bounds__(128, 4)
qwen4_gdn_decode_fused_k1_splitv4(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ deinterleaved,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16* __restrict__ normed_out,
    unsigned int* __restrict__ group_done,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    float l2_eps,
    float rms_eps,
    float* __restrict__ scratch_gdn,
    float* __restrict__ scratch_ss,
    unsigned int* __restrict__ split_done
) {
    const unsigned int qkvz_stride =
        2u * num_k_heads * QWEN4_GDN_DIM + 2u * num_v_heads * QWEN4_GDN_DIM;
    const unsigned int value_dim = num_v_heads * QWEN4_GDN_DIM;
    qwen4_gdn_fused_splitv4_body<false, false, false>(
        conv_state, deinterleaved, conv_weight, conv_bias,
        gate, beta, h_state, rms_weight, normed_out,
        nullptr, nullptr, nullptr, group_done,
        scratch_gdn, scratch_ss, split_done,
        1u, num_k_heads, num_v_heads, l2_eps, rms_eps,
        qkvz_stride, num_v_heads * 2u, value_dim, 0u, 0u, 0u
    );
}

extern "C" __global__ void __launch_bounds__(128, 4)
qwen4_gdn_verify_fused_lazyfinal_splitv4(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ deinterleaved,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16* __restrict__ normed_out,
    float* __restrict__ retain_qkv,
    float* __restrict__ conv_inter,
    float* __restrict__ h_state_inter,
    unsigned int* __restrict__ group_done,
    unsigned int num_tokens,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    float l2_eps,
    float rms_eps,
    unsigned int input_stride,
    unsigned int gate_beta_stride,
    unsigned int normed_stride,
    unsigned int retain_stride,
    unsigned int conv_inter_stride,
    unsigned int h_inter_stride,
    float* __restrict__ scratch_gdn,
    float* __restrict__ scratch_ss,
    unsigned int* __restrict__ split_done
) {
    qwen4_gdn_fused_splitv4_body<true, false, false>(
        conv_state, deinterleaved, conv_weight, conv_bias,
        gate, beta, h_state, rms_weight, normed_out,
        retain_qkv, conv_inter, h_state_inter, group_done,
        scratch_gdn, scratch_ss, split_done,
        num_tokens, num_k_heads, num_v_heads, l2_eps, rms_eps,
        input_stride, gate_beta_stride, normed_stride,
        retain_stride, conv_inter_stride, h_inter_stride
    );
}

extern "C" __global__ void __launch_bounds__(128, 4)
qwen4_gdn_verify_fused_persistent_splitv4(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ deinterleaved,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16* __restrict__ normed_out,
    float* __restrict__ conv_inter,
    float* __restrict__ h_state_inter,
    unsigned int* __restrict__ group_done,
    unsigned int num_tokens,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    float l2_eps,
    float rms_eps,
    unsigned int input_stride,
    unsigned int gate_beta_stride,
    unsigned int normed_stride,
    unsigned int conv_inter_stride,
    unsigned int h_inter_stride,
    float* __restrict__ scratch_gdn,
    float* __restrict__ scratch_ss,
    unsigned int* __restrict__ split_done
) {
    qwen4_gdn_fused_splitv4_body<false, true, false>(
        conv_state, deinterleaved, conv_weight, conv_bias,
        gate, beta, h_state, rms_weight, normed_out,
        nullptr, conv_inter, h_state_inter, group_done,
        scratch_gdn, scratch_ss, split_done,
        num_tokens, num_k_heads, num_v_heads, l2_eps, rms_eps,
        input_stride, gate_beta_stride, normed_stride,
        0u, conv_inter_stride, h_inter_stride
    );
}

extern "C" __global__ void __launch_bounds__(128, 4)
qwen4_gdn_verify_fused_lazyfinal_multi_seq_splitv4(
    const GdnFusedMultiSeqDesc* __restrict__ seq_descs,
    const __nv_bfloat16* __restrict__ deinterleaved,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16* __restrict__ normed_out,
    unsigned int num_seqs,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    float l2_eps,
    float rms_eps,
    unsigned int input_stride,
    unsigned int gate_beta_stride,
    unsigned int normed_stride,
    unsigned int retain_stride,
    unsigned int conv_inter_stride,
    unsigned int h_inter_stride,
    float* __restrict__ scratch_gdn,
    float* __restrict__ scratch_ss,
    unsigned int* __restrict__ split_done
) {
    const unsigned int seq_idx = blockIdx.z;
    if (seq_idx >= num_seqs) return;
    __shared__ GdnMultiSeqSmem ms;
    if (threadIdx.x == 0u) {
        const GdnFusedMultiSeqDesc desc = seq_descs[seq_idx];
        const bool keep_late = (desc.flags & GDN_FUSED_FLAG_KEEP_LATE) != 0u;
        const bool ctrl_indirect =
            keep_late && ((desc.flags & GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR) != 0u);
        unsigned long long retain_qkv_addr = desc.retain_qkv;
        unsigned long long h_state_inter_addr = desc.h_state_inter;
        GdnKeepLateCtrl* ctrl_ptr = nullptr;
        unsigned int pending_base = 0u;
        unsigned int pending_rows = 0u;
        unsigned int write_base = 0u;
        unsigned int ring_cap = 0u;
        unsigned int flush_now = 0u;
        if (ctrl_indirect && desc.h_state_inter != 0ull) {
            ctrl_ptr = (GdnKeepLateCtrl*)desc.h_state_inter;
            retain_qkv_addr = ctrl_ptr->ring_base;
            h_state_inter_addr = 0ull;
            pending_base = ctrl_ptr->pending_base;
            pending_rows = ctrl_ptr->pending_rows;
            write_base = ctrl_ptr->write_base;
            ring_cap = ctrl_ptr->ring_cap;
            flush_now = ctrl_ptr->flush_now;
        }
        ms.conv_state = (float*)desc.conv_state;
        ms.h_state = (float*)desc.h_state;
        ms.retain_qkv = (float*)retain_qkv_addr;
        ms.conv_inter = (float*)desc.conv_inter;
        ms.h_state_inter = (float*)h_state_inter_addr;
        ms.group_done = (unsigned int*)desc.group_done;
        ms.ctrl_ptr = ctrl_ptr;
        ms.row_base = desc.row_base;
        ms.num_tokens = desc.num_tokens;
        ms.flags = desc.flags;
        ms.pending_base = pending_base;
        ms.pending_rows = pending_rows;
        ms.write_base = write_base;
        ms.ring_cap = ring_cap;
        ms.flush_now = flush_now;
    }
    __syncthreads();
    qwen4_gdn_fused_splitv4_body<true, false, true>(
        nullptr, deinterleaved, conv_weight, conv_bias,
        gate, beta, nullptr, rms_weight, normed_out,
        nullptr, nullptr, nullptr, nullptr,
        scratch_gdn, scratch_ss, split_done + (unsigned long long)seq_idx * num_v_heads,
        ms.num_tokens, num_k_heads, num_v_heads, l2_eps, rms_eps,
        input_stride, gate_beta_stride, normed_stride,
        retain_stride, conv_inter_stride, h_inter_stride, &ms
    );
    if (blockIdx.x == 0u && blockIdx.y == 0u && threadIdx.x == 0u && ms.ctrl_ptr != nullptr) {
        ms.ctrl_ptr->executed_flag = 1u;
    }
}

extern "C" __global__ void __launch_bounds__(128, 4)
qwen4_gdn_verify_fused_persistent_multi_seq_splitv4(
    const GdnFusedMultiSeqDesc* __restrict__ seq_descs,
    const __nv_bfloat16* __restrict__ deinterleaved,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16* __restrict__ normed_out,
    unsigned int num_seqs,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    float l2_eps,
    float rms_eps,
    unsigned int input_stride,
    unsigned int gate_beta_stride,
    unsigned int normed_stride,
    unsigned int conv_inter_stride,
    unsigned int h_inter_stride,
    float* __restrict__ scratch_gdn,
    float* __restrict__ scratch_ss,
    unsigned int* __restrict__ split_done
) {
    const unsigned int seq_idx = blockIdx.z;
    if (seq_idx >= num_seqs) return;
    __shared__ GdnMultiSeqSmem ms;
    if (threadIdx.x == 0u) {
        const GdnFusedMultiSeqDesc desc = seq_descs[seq_idx];
        ms.conv_state = (float*)desc.conv_state;
        ms.h_state = (float*)desc.h_state;
        ms.retain_qkv = nullptr;
        ms.conv_inter = (float*)desc.conv_inter;
        ms.h_state_inter = (float*)desc.h_state_inter;
        ms.group_done = (unsigned int*)desc.group_done;
        ms.ctrl_ptr = nullptr;
        ms.row_base = desc.row_base;
        ms.num_tokens = desc.num_tokens;
        ms.flags = 0u;
        ms.pending_base = 0u;
        ms.pending_rows = 0u;
        ms.write_base = 0u;
        ms.ring_cap = 0u;
        ms.flush_now = 0u;
    }
    __syncthreads();
    qwen4_gdn_fused_splitv4_body<false, true, false>(
        nullptr, deinterleaved, conv_weight, conv_bias,
        gate, beta, nullptr, rms_weight, normed_out,
        nullptr, nullptr, nullptr, nullptr,
        scratch_gdn, scratch_ss, split_done + (unsigned long long)seq_idx * num_v_heads,
        ms.num_tokens, num_k_heads, num_v_heads, l2_eps, rms_eps,
        input_stride, gate_beta_stride, normed_stride,
        0u, conv_inter_stride, h_inter_stride, &ms
    );
}

extern "C" __global__ void __launch_bounds__(128, 4)
qwen4_gdn_verify_fused_lazyfinal_multi_seq_scaled_fp16_splitv4(
    const GdnFusedMultiSeqDesc* __restrict__ seq_descs,
    const __nv_bfloat16* __restrict__ deinterleaved,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16* __restrict__ normed_out,
    unsigned int num_seqs,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    float l2_eps,
    float rms_eps,
    unsigned int input_stride,
    unsigned int gate_beta_stride,
    unsigned int normed_stride,
    unsigned int retain_stride,
    unsigned int conv_inter_stride,
    unsigned int h_inter_stride,
    float* __restrict__ scratch_gdn,
    float* __restrict__ scratch_ss,
    unsigned int* __restrict__ split_done,
    float* __restrict__ scratch_h_flush,
    float* __restrict__ scratch_amax
) {
    const unsigned int seq_idx = blockIdx.z;
    if (seq_idx >= num_seqs) return;
    __shared__ GdnMultiSeqSmem ms;
    if (threadIdx.x == 0u) {
        const GdnFusedMultiSeqDesc desc = seq_descs[seq_idx];
        const bool keep_late = (desc.flags & GDN_FUSED_FLAG_KEEP_LATE) != 0u;
        const bool ctrl_indirect =
            keep_late && ((desc.flags & GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR) != 0u);
        unsigned long long retain_qkv_addr = desc.retain_qkv;
        unsigned long long h_state_inter_addr = desc.h_state_inter;
        GdnKeepLateCtrl* ctrl_ptr = nullptr;
        unsigned int pending_base = 0u;
        unsigned int pending_rows = 0u;
        unsigned int write_base = 0u;
        unsigned int ring_cap = 0u;
        unsigned int flush_now = 0u;
        if (ctrl_indirect && desc.h_state_inter != 0ull) {
            ctrl_ptr = (GdnKeepLateCtrl*)desc.h_state_inter;
            retain_qkv_addr = ctrl_ptr->ring_base;
            h_state_inter_addr = 0ull;
            pending_base = ctrl_ptr->pending_base;
            pending_rows = ctrl_ptr->pending_rows;
            write_base = ctrl_ptr->write_base;
            ring_cap = ctrl_ptr->ring_cap;
            flush_now = ctrl_ptr->flush_now;
        }
        ms.conv_state = (float*)desc.conv_state;
        ms.h_state = (float*)desc.h_state;
        ms.retain_qkv = (float*)retain_qkv_addr;
        ms.conv_inter = (float*)desc.conv_inter;
        ms.h_state_inter = (float*)h_state_inter_addr;
        ms.group_done = (unsigned int*)desc.group_done;
        ms.ctrl_ptr = ctrl_ptr;
        ms.row_base = desc.row_base;
        ms.num_tokens = desc.num_tokens;
        ms.flags = desc.flags;
        ms.pending_base = pending_base;
        ms.pending_rows = pending_rows;
        ms.write_base = write_base;
        ms.ring_cap = ring_cap;
        ms.flush_now = flush_now;
    }
    __syncthreads();
    const unsigned long long head_elems =
        (unsigned long long)num_v_heads * (QWEN4_GDN_DIM * QWEN4_GDN_DIM);
    float* scratch_h_seq = scratch_h_flush + (unsigned long long)seq_idx * head_elems;
    float* scratch_amax_seq =
        scratch_amax + (unsigned long long)seq_idx * num_v_heads * QWEN4_GDN_SPLIT_V;
    qwen4_gdn_fused_splitv4_body<true, false, true, true>(
        nullptr, deinterleaved, conv_weight, conv_bias,
        gate, beta, nullptr, rms_weight, normed_out,
        nullptr, nullptr, nullptr, nullptr,
        scratch_gdn, scratch_ss, split_done + (unsigned long long)seq_idx * num_v_heads,
        ms.num_tokens, num_k_heads, num_v_heads, l2_eps, rms_eps,
        input_stride, gate_beta_stride, normed_stride,
        retain_stride, conv_inter_stride, h_inter_stride, &ms,
        scratch_h_seq, scratch_amax_seq
    );
    if (blockIdx.x == 0u && blockIdx.y == 0u && threadIdx.x == 0u && ms.ctrl_ptr != nullptr) {
        ms.ctrl_ptr->executed_flag = 1u;
    }
}
