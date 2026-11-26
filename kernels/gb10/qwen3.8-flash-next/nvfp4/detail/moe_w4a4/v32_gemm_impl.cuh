// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// Core implementation templates for V32 W4A4 (NVFP4 x NVFP4) persistent MoE kernels.
// Included by `qwen4_moe_w4a4_v32.cu` after `oi640` alias, `qwen4_geometry.cuh`,
// and `detail/qwen4_nvfp4_common.cuh`.

namespace {

constexpr unsigned V32_TILE_M = 128u;
constexpr unsigned V32_BC = 128u;
constexpr unsigned V32_THREADS = 256u;
constexpr unsigned V32_K_STEP = 128u;
constexpr unsigned V32_STAGES = 2u;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1000)
constexpr unsigned V32_XQ_PIN_MAX_ROWS = 20480u;
constexpr unsigned V32_OQ_NVFP4_PIN_MAX_ROWS = 20480u;
constexpr unsigned V32_OQ_MXFP8_KEEP_WINDOW = 960u;
constexpr unsigned long long V32_KEEP_SLOT_BYTES = 64ull << 20;
#else
constexpr unsigned V32_XQ_PIN_MAX_ROWS = 8192u;
constexpr unsigned V32_OQ_NVFP4_PIN_MAX_ROWS = 4096u;
constexpr unsigned V32_OQ_MXFP8_KEEP_WINDOW = 256u;
constexpr unsigned long long V32_KEEP_SLOT_BYTES = 12ull << 20;
#endif

template <typename Shape, bool GATEUP, bool TS4E> struct V32W4a4Smem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V32_BC / BMAT; // 64 (GU) or 128 (DN)
    static constexpr unsigned DOWN_SLICES = Shape::INTER / 64u;
    static constexpr unsigned TS4E_KS = Shape::template gemm_k<GATEUP>() / V32_K_STEP;
    static constexpr unsigned TS4E_B_PER_KS = GATEUP ? 8u : 4u; // 4 k32-steps x BMAT
    static constexpr unsigned TS4E_BKTS = TS4E_KS * TS4E_B_PER_KS;
    static constexpr unsigned TS4E_EXC_CAP = 32u;

    static constexpr unsigned A = V32_STAGES * 2u * V32_TILE_M * 32u;  // 16,384 B ([2][2][128][32])
    static constexpr unsigned AS = V32_STAGES * V32_TILE_M * 8u;       //  2,048 B ([2][128][8])
    static constexpr unsigned B = V32_STAGES * 2u * 32u * V32_BC;      // 16,384 B ([2][2][32][128])
    static constexpr unsigned S = V32_STAGES * 2u * 4u * V32_BC;       //  2,048 B ([2][2][4][128])
    static constexpr unsigned SF = V32_STAGES * 2u * V32_BC * 4u;      //  2,048 B ([2][2][128] u32)
    static constexpr unsigned XS2 =
        GATEUP ? (V32_TILE_M * 4u) : ((DOWN_SLICES + 1u) * V32_TILE_M * 4u);
    static constexpr unsigned PTRS = 8u * 8u;                          //     64 B (sPtrs[0..5], sTask[2])

    static constexpr unsigned OFF_AS = A;
    static constexpr unsigned OFF_B = OFF_AS + AS;
    static constexpr unsigned OFF_S = OFF_B + B;
    static constexpr unsigned OFF_SF = OFF_S + S;
    static constexpr unsigned OFF_XS2 = OFF_SF + SF;
    static constexpr unsigned OFF_PTRS = OFF_XS2 + XS2;
    static constexpr unsigned OFF_TASK = OFF_PTRS + 48u;
    static constexpr unsigned OFF_BASE = OFF_PTRS + PTRS;
    static constexpr unsigned OFF_EXC = OFF_BASE + (TS4E ? V32_BC : 0u);
    static constexpr unsigned OFF_BKT_BEG =
        OFF_EXC + (TS4E ? (V32_STAGES * TS4E_B_PER_KS * TS4E_EXC_CAP * 2u) : 0u);
    static constexpr unsigned OFF_BKT_CNT = OFF_BKT_BEG + (TS4E ? (TS4E_BKTS * 4u) : 0u);
    static constexpr unsigned RAW_BYTES = OFF_BKT_CNT + (TS4E ? (TS4E_BKTS * 2u) : 0u);
    static constexpr unsigned BYTES = (RAW_BYTES + 15u) & ~15u;

    static_assert(BYTES <= 49152u, "V32W4a4Smem must stay <= 48 KiB for 2 CTAs/SM");
};

static_assert(V32W4a4Smem<Qwen4FlashNextShape, true, false>::BYTES == 39488u, "V32W4a4Smem flat GU byte contract");
static_assert(V32W4a4Smem<Qwen4FlashNextShape, false, false>::BYTES == 44608u, "V32W4a4Smem flat DN byte contract");
static_assert(V32W4a4Smem<Qwen4FlashNextShape, true, true>::BYTES == 41600u, "V32W4a4Smem ts4e GU byte contract");
static_assert(V32W4a4Smem<Qwen4FlashNextShape, false, true>::BYTES == 45376u, "V32W4a4Smem ts4e DN byte contract");

// 32-byte row swizzle for sA (two 16-byte chunks per row): zero bank conflicts
// for both cp.async.16 writes and ldmatrix.m8n8.x4 reads.
__device__ __forceinline__ unsigned v32_swz32(unsigned row, unsigned chunk) {
    return row * 32u + ((chunk ^ ((row >> 2) & 1u)) << 4);
}

__device__ __forceinline__ void v32_cp16_pol_s32(unsigned s_addr, const void* gmem_ptr,
                                                 unsigned long long pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, 16, %2;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "l"(pol));
}

__device__ __forceinline__ void v32_cp16_pol_pred_s32(unsigned s_addr, const void* gmem_ptr, bool pred,
                                                      unsigned long long pol) {
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "r"(bytes), "l"(pol));
}

__device__ __forceinline__ void v32_cp16_s32(unsigned s_addr, const void* gmem_ptr, bool pred) {
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "r"(bytes));
}

__device__ __forceinline__ void v32_cp8_s32(unsigned s_addr, const void* gmem_ptr, bool pred) {
    const int bytes = pred ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "r"(bytes));
}

__device__ __forceinline__ unsigned v32_ld_su32(unsigned s_addr) {
    unsigned v;
    asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(s_addr));
    return v;
}

__device__ __forceinline__ uint4 v32_ld_sv4(unsigned s_addr) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(s_addr));
    return v;
}

__device__ __forceinline__ void v32_st_su32(unsigned s_addr, unsigned v) {
    asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(s_addr), "r"(v) : "memory");
}

__device__ __forceinline__ const unsigned char* v32_ld_sptr(unsigned ptrs_base, unsigned idx) {
    unsigned long long p;
    asm volatile("ld.shared.u64 %0, [%1];\n" : "=l"(p) : "r"(ptrs_base + idx * 8u));
    return reinterpret_cast<const unsigned char*>(p);
}

__device__ __forceinline__ float v32_ld_sxs2(unsigned xs2_base, unsigned row) {
    float v;
    asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(v) : "r"(xs2_base + row * 4u));
    return v;
}

__device__ __forceinline__ void v32_prefetch_l2(const void* gmem_ptr) {
    asm volatile("prefetch.global.L2 [%0];\n" :: "l"(gmem_ptr));
}

__device__ __forceinline__ void v32_st4_l2_el(void* dst, unsigned v, unsigned long long pol_el) {
    asm volatile("st.global.L2::cache_hint.u32 [%0], %1, %2;\n"
                 :: "l"(dst), "r"(v), "l"(pol_el) : "memory");
}

__device__ __forceinline__ void v32_st4_ef(void* dst, unsigned v) {
    asm volatile("st.global.cs.u32 [%0], %1;\n" :: "l"(dst), "r"(v) : "memory");
}

__device__ __forceinline__ void v32_st8_l2_el(void* dst, uint2 v, unsigned long long pol_el) {
    asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y), "l"(pol_el) : "memory");
}

__device__ __forceinline__ void v32_st8_ef(void* dst, uint2 v) {
    asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y) : "memory");
}

__device__ __forceinline__ void v32_st16_stream(void* dst, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

// Conflict-free sSF u32 index for even/odd paired LDS.128 reads.
// Layout: sSF[sl][kh][wn (0..1)][half (0..1)][gid (0..7)][4 u32: p0_e, p0_o, p1_e, p1_o].
template <bool GATEUP>
__device__ __forceinline__ unsigned v32_sf_eo_u32_idx(unsigned col) {
    if constexpr (GATEUP) {
        const unsigned mtx = col >> 6;
        const unsigned lc = col & 63u;
        const unsigned w = (lc >> 5) & 1u;
        const unsigned p = (lc >> 4) & 1u;
        const unsigned g = (lc >> 1) & 7u;
        const unsigned eo = lc & 1u;
        return ((((w * 2u + mtx) * 8u) + g) << 2) + (p << 1) + eo;
    } else {
        const unsigned w = col >> 6;
        const unsigned h = (col >> 5) & 1u;
        const unsigned p = (col >> 4) & 1u;
        const unsigned g = (col >> 1) & 7u;
        const unsigned eo = col & 1u;
        return ((((w * 2u + h) * 8u) + g) << 2) + (p << 1) + eo;
    }
}

// Register-direct 4-lane rearrange for even/odd paired MMA outputs (0 SMEM).
__device__ __forceinline__ uint4 v32_bf16_reg_quad_eo(
    unsigned u0, unsigned u1, unsigned u2, unsigned u3, unsigned tig
) {
    const bool hi2 = (tig & 2u) != 0u;
    const unsigned s0 = __shfl_xor_sync(0xffffffffu, hi2 ? u0 : u2, 2);
    const unsigned s1 = __shfl_xor_sync(0xffffffffu, hi2 ? u1 : u3, 2);
    const unsigned t0 = hi2 ? s0 : u0;
    const unsigned t1 = hi2 ? s1 : u1;
    const unsigned t2 = hi2 ? u2 : s0;
    const unsigned t3 = hi2 ? u3 : s1;
    const bool odd1 = (tig & 1u) != 0u;
    const unsigned r0 = __shfl_xor_sync(0xffffffffu, odd1 ? t0 : t2, 1);
    const unsigned r1 = __shfl_xor_sync(0xffffffffu, odd1 ? t1 : t3, 1);
    return make_uint4(
        odd1 ? r0 : t0,
        odd1 ? r1 : t1,
        odd1 ? t2 : r0,
        odd1 ? t3 : r1);
}

// Quad-lane MXFP8 quantizer (bit-identical to v28_mxfp8_reg_quad / v16q4).
__device__ __forceinline__ void v32_mxfp8_reg_quad(
    unsigned u0, unsigned u1, unsigned u2, unsigned u3,
    unsigned tig, unsigned (&qq)[2], unsigned& ss
) {
    const uint4 w = v32_bf16_reg_quad_eo(u0, u1, u2, u3, tig);
    const unsigned u[4] = {w.x, w.y, w.z, w.w};
    float v[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[2 * i] = __uint_as_float(u[i] << 16);
        v[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
    }
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    qq[0] = 0u;
    qq[1] = 0u;
    ss = 0u;
    if (amax > 0.0f) {
        int e;
        const float m = frexpf(__fmul_rn(amax, 1.0f / 448.0f), &e);
        int k = (m == 0.5f) ? e - 1 : e;
        k = max(-127, min(127, k));
        const float inv = __int_as_float((127 - k) << 23);
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            qq[i] = qwen4_nvfp4_detail::pack_e4m3x4_nv(
                v[4 * i + 0] * inv,
                v[4 * i + 1] * inv,
                v[4 * i + 2] * inv,
                v[4 * i + 3] * inv);
        }
        ss = static_cast<unsigned>(k + 127);
    }
}

template <typename Shape, bool GATEUP, bool OQ_NVFP4, int EPI, bool PREFETCH, bool TS4E>
__device__ __forceinline__ void v32_w4a4_impl(
    const unsigned char* __restrict__ aq,
    const unsigned char* __restrict__ as,
    const float* __restrict__ x_scale2,
    const unsigned long long* __restrict__ pk0,
    const unsigned long long* __restrict__ sc0,
    const float* __restrict__ s20,
    const unsigned long long* __restrict__ pk1,
    const unsigned long long* __restrict__ sc1,
    const float* __restrict__ s21,
    __nv_bfloat16* __restrict__ C,
    unsigned char* __restrict__ oq,
    unsigned char* __restrict__ os,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned N_in,
    unsigned K_in,
    float oq_scale2,
    float* __restrict__ oq_scale2_out = nullptr,
    const float* __restrict__ oq_scale2_in = nullptr,
    unsigned char* __restrict__ slot_q = nullptr,
    unsigned char* __restrict__ slot_s = nullptr
) {
    using L = V32W4a4Smem<Shape, GATEUP, TS4E>;
    constexpr unsigned N = Shape::template gemm_n<GATEUP>();
    constexpr unsigned K = Shape::template gemm_k<GATEUP>();
    constexpr unsigned TN = GATEUP ? 64u : 128u;
    constexpr unsigned BCOLS = L::BCOLS;
    constexpr unsigned CH = BCOLS / 16u;
    constexpr unsigned DOWN_SLICES = L::DOWN_SLICES;
    using Ts4e = Qwen4Ts4eLayout<N, K>;
    static_assert((N % TN) == 0u, "v32_w4a4 requires N divisible by TN");
    static_assert((K % V32_K_STEP) == 0u, "v32_w4a4 requires K divisible by V32_K_STEP");
    constexpr unsigned TS4E_B_PER_KS = L::TS4E_B_PER_KS;
    constexpr unsigned n_tiles = N / TN;
    constexpr unsigned KS = K / V32_K_STEP;
    constexpr unsigned K_HALF = K / 2u;
    constexpr unsigned K_GRP = K / 16u;
    static_assert(KS >= 2u, "v32_w4a4 requires KS >= 2");

    unsigned dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (gridDim.x == 0u || gridDim.x > oi640::FIXED_GRID || blockDim.x != V32_THREADS ||
        dyn_smem < L::BYTES) {
        if (threadIdx.x == 0u) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }
    const bool shape = (N_in == N && K_in == K);
    const bool s2_gu_ok = (oq_scale2_out != nullptr) || (isfinite(oq_scale2) && oq_scale2 > 0.0f);
    const bool s2_dn_ok = (oq_scale2_in != nullptr) || (isfinite(oq_scale2) && oq_scale2 > 0.0f);
    const bool outs = GATEUP
        ? (oq != nullptr && os != nullptr && (!OQ_NVFP4 || s2_gu_ok))
        : ((EPI == 3 ? (slot_q != nullptr && slot_s != nullptr) : (C != nullptr)) && s2_dn_ok);
    const bool tbl_ok = pk0 != nullptr && sc0 != nullptr && s20 != nullptr &&
                        (!GATEUP || (x_scale2 != nullptr && pk1 != nullptr && sc1 != nullptr && s21 != nullptr));
    if (!shape || !outs || !tbl_ok || aq == nullptr || as == nullptr || expert_offsets == nullptr ||
        sorted_token_ids == nullptr || !oi640::contract_ok(contract) ||
        !oi640::workspace_ok_worksteal(workspace, contract) ||
        status == nullptr || status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0u) oi640::fail(status, oi640::ERR_ABI);
        return;
    }

    const unsigned wc = workspace->work_count;
    const unsigned task_count = wc * n_tiles;
    if (blockIdx.x >= task_count) return;

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    // Keep `sbase` in a general register (not UR0) to avoid the CUDA 13.0 ptxas
    // uniform-register allocator bug that emits odd `desc[UR1]` on `cp.async.L2::cache_hint`.
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n"
                 : "+r"(sbase));
    unsigned long long pol_el, pol_ef;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));

    uint32_t* const next_task_ptr = GATEUP ? &workspace->next_task_gu : &workspace->next_task_dn;
    unsigned task;
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(task));
    unsigned task_slot = 0u;

    #pragma unroll 1
    while (task < task_count) {
        const unsigned raw_wi = task / n_tiles;
        const unsigned wi = (!GATEUP) ? (wc - 1u - raw_wi) : raw_wi;
        const uint2 item_u2 = *reinterpret_cast<const uint2*>(&workspace->items[wi]);
        const unsigned item_expert = item_u2.x;
        const unsigned item_m_tile = item_u2.y;
        const unsigned nt = task % n_tiles;
        const unsigned cta_n = nt * TN;
        if (item_expert >= Shape::EXPERTS) {
            if (threadIdx.x == 0u) oi640::fail(status, oi640::ERR_PLAN);
            return;
        }
        const int m_start = expert_offsets[item_expert];
        const int M_expert = expert_offsets[item_expert + 1] - m_start;
        const int local = static_cast<int>(item_m_tile * V32_TILE_M);
        const bool range_ok = (m_start >= 0 && M_expert > 0 && local >= 0 && local < M_expert);
        const unsigned cta_m = range_ok ? static_cast<unsigned>(m_start + local) : 0u;
        const int valid_rows = range_ok ? min(M_expert - local, static_cast<int>(V32_TILE_M)) : 0;

        if (threadIdx.x == 0u) {
            const unsigned long long pB0 = pk0[item_expert];
            const unsigned long long pS0 = sc0[item_expert];
            const unsigned long long pB1 = GATEUP ? pk1[item_expert] : pB0;
            const unsigned long long pS1 = GATEUP ? sc1[item_expert] : pS0;
            const float f_s2a = s20[item_expert];
            const float f_s2b = GATEUP ? s21[item_expert] : 0.0f;
            const bool ptr_bad = !range_ok || pB0 == 0ull || pS0 == 0ull || pB1 == 0ull || pS1 == 0ull ||
                                 !isfinite(f_s2a) || !isfinite(f_s2b);
            unsigned long long* const sPtrs = reinterpret_cast<unsigned long long*>(smem + L::OFF_PTRS);
            sPtrs[0] = pB0;
            sPtrs[1] = pS0;
            sPtrs[2] = pB1;
            sPtrs[3] = pS1;
            sPtrs[4] = ptr_bad ? 1ull : 0ull;
        }
        if constexpr (GATEUP) {
            const unsigned t0 = threadIdx.x;
            if (t0 < V32_TILE_M) {
                const bool ok_row = static_cast<int>(t0) < valid_rows;
                const unsigned tok_row = ok_row ? static_cast<unsigned>(sorted_token_ids[cta_m + t0]) : 0u;
                reinterpret_cast<float*>(smem + L::OFF_XS2)[t0] = ok_row ? x_scale2[tok_row] : 0.0f;
            }
        } else {
            const unsigned t0 = threadIdx.x;
            float* const s_os2 = reinterpret_cast<float*>(smem + L::OFF_XS2);
            if (t0 < V32_TILE_M) {
                const bool ok_row = static_cast<int>(t0) < valid_rows;
                if (oq_scale2_in != nullptr) {
                    float s_vals[DOWN_SLICES];
                    float max_s2 = 0.0f;
                    if (ok_row) {
                        const float* row_ptr =
                            oq_scale2_in + static_cast<unsigned long long>(cta_m + t0) * DOWN_SLICES;
                        #pragma unroll
                        for (unsigned sl_i = 0u; sl_i < DOWN_SLICES; ++sl_i) {
                            const float sv = __ldg(&row_ptr[sl_i]);
                            const float clean = (isfinite(sv) && sv > 0.0f) ? sv : 0.0f;
                            s_vals[sl_i] = clean;
                            max_s2 = fmaxf(max_s2, clean);
                        }
                    } else {
                        #pragma unroll
                        for (unsigned sl_i = 0u; sl_i < DOWN_SLICES; ++sl_i) s_vals[sl_i] = 0.0f;
                    }
                    const float thresh = __fmul_rn(max_s2, 1e-10f);
                    float prev = 0.0f;
                    #pragma unroll
                    for (unsigned sl_i = 0u; sl_i < DOWN_SLICES; ++sl_i) {
                        const float cur = s_vals[sl_i];
                        if (cur > thresh) {
                            s_os2[sl_i * V32_TILE_M + t0] =
                                (prev > 0.0f) ? __fdiv_rn(prev, cur) : 0.0f;
                            prev = cur;
                        } else {
                            s_os2[sl_i * V32_TILE_M + t0] = -1.0f;
                        }
                    }
                    s_os2[DOWN_SLICES * V32_TILE_M + t0] = prev;
                } else {
                    s_os2[0u * V32_TILE_M + t0] = ok_row ? 0.0f : -1.0f;
                    #pragma unroll
                    for (unsigned sl_i = 1u; sl_i < DOWN_SLICES; ++sl_i) {
                        s_os2[sl_i * V32_TILE_M + t0] = ok_row ? 1.0f : -1.0f;
                    }
                    s_os2[DOWN_SLICES * V32_TILE_M + t0] = ok_row ? oq_scale2 : 0.0f;
                }
            }
        }
        __syncthreads();
        if (reinterpret_cast<const unsigned long long*>(smem + L::OFF_PTRS)[4] != 0ull) {
            if (threadIdx.x == 0u) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }

        const int active_a_rows = (valid_rows + 15) & ~15;
        const bool b_reuse = (!GATEUP)
            ? (item_m_tile > 0u)
            : (local + static_cast<int>(V32_TILE_M) < M_expert);
        const unsigned long long pol_b = b_reuse ? pol_el : pol_ef;
        const bool a_evict_last_nt = (!GATEUP) && (nt + 1u == n_tiles);
        const unsigned col_half = cta_n & 64u;

        // Resolve all TS4E buckets up front.
        if constexpr (TS4E) {
            unsigned t;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
            if (t < L::TS4E_BKTS) {
                const unsigned ks_i = t / TS4E_B_PER_KS;
                const unsigned b = t & (TS4E_B_PER_KS - 1u);
                const unsigned kh = GATEUP ? (b >> 2) : (b >> 1);
                const unsigned sub = GATEUP ? ((b >> 1) & 1u) : 0u;
                const unsigned tt = b & 1u;
                const unsigned char* blob = v32_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                const unsigned bkt = (4u * ks_i + 2u * kh + tt) * Ts4e::N_TILES + (cta_n >> 7);
                const unsigned* page_bases = reinterpret_cast<const unsigned*>(blob + Ts4e::PAGE_BASES_OFF);
                const unsigned short* bkt_off = reinterpret_cast<const unsigned short*>(blob + Ts4e::BKT_OFF_OFF);
                const unsigned pb0 = __ldg(&page_bases[bkt >> 7]);
                const unsigned pb1 = ((bkt & 127u) == 127u) ? __ldg(&page_bases[(bkt + 1u) >> 7]) : pb0;
                const unsigned beg = pb0 + static_cast<unsigned>(__ldg(&bkt_off[bkt]));
                const unsigned end = pb1 + static_cast<unsigned>(__ldg(&bkt_off[bkt + 1u]));
                reinterpret_cast<unsigned*>(smem + L::OFF_BKT_BEG)[t] = beg;
                reinterpret_cast<unsigned short*>(smem + L::OFF_BKT_CNT)[t] =
                    static_cast<unsigned short>(end - beg);
            }
            if (t < 8u) {
                const unsigned bm = GATEUP ? (t >> 2) : 0u;
                const unsigned bc = GATEUP ? (t & 3u) : t;
                const unsigned char* bs = v32_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && bm) ? 3u : 1u);
                v32_cp16_pol_s32(
                    sbase + L::OFF_BASE + t * 16u,
                    bs + static_cast<unsigned long long>(Ts4e::NIBBLES_BYTES) + cta_n + bc * 16u,
                    pol_b);
            }
            __syncthreads();
        }

        // Load K_STEP = 128 slice into stage `ks & 1` (two k64 halves: kh=0, kh=1).
        auto load_a = [&](unsigned ks) {
            unsigned t;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
            const unsigned sl = ks & 1u;
            const unsigned row = t >> 1;
            const unsigned ch = t & 1u;
            if (static_cast<int>(row) < active_a_rows) {
                const bool ok = static_cast<int>(row) < valid_rows;
                const unsigned src_row = ok
                    ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + row]) : (cta_m + row))
                    : 0u;
                const unsigned dst_a0 = sbase + sl * 8192u + v32_swz32(row, ch);
                const unsigned char* g_a0 =
                    aq + static_cast<unsigned long long>(src_row) * K_HALF + ks * 64u + ch * 16u;
                if constexpr (!GATEUP) {
                    if (a_evict_last_nt) {
                        v32_cp16_pol_pred_s32(dst_a0, g_a0, ok, pol_ef);
                        v32_cp16_pol_pred_s32(dst_a0 + 4096u, g_a0 + 32u, ok, pol_ef);
                    } else {
                        v32_cp16_s32(dst_a0, g_a0, ok);
                        v32_cp16_s32(dst_a0 + 4096u, g_a0 + 32u, ok);
                    }
                } else {
                    v32_cp16_s32(dst_a0, g_a0, ok);
                    v32_cp16_s32(dst_a0 + 4096u, g_a0 + 32u, ok);
                }
            }
            if (t < V32_TILE_M && static_cast<int>(t) < active_a_rows) {
                const bool ok = static_cast<int>(t) < valid_rows;
                const unsigned src_row = ok
                    ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + t]) : (cta_m + t))
                    : 0u;
                v32_cp8_s32(
                    sbase + L::OFF_AS + (sl * V32_TILE_M + t) * 8u,
                    as + static_cast<unsigned long long>(src_row) * K_GRP + ks * 8u,
                    ok);
            }
        };

        auto load_b = [&](unsigned ks) {
            unsigned t;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
            const unsigned sl = ks & 1u;
            const unsigned r = t >> 3;
            const unsigned c = t & 7u;
            const unsigned c_swz = c ^ ((r >> 1) & 7u);
            const unsigned mtx = GATEUP ? (c >> 2) : 0u;
            const unsigned lc = GATEUP ? (c & 3u) : c;
            const unsigned char* bp = v32_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && mtx) ? 2u : 0u);
            const unsigned dst_b0 = sbase + L::OFF_B + sl * 8192u + r * 128u + (c_swz << 4);
            const unsigned char* g_b0 =
                bp + static_cast<unsigned long long>(ks * 64u + r) * N + cta_n + lc * 16u;
            v32_cp16_pol_s32(dst_b0, g_b0, pol_b);
            v32_cp16_pol_s32(dst_b0 + 4096u, g_b0 + static_cast<unsigned long long>(32u) * N, pol_b);

            if constexpr (TS4E) {
                const unsigned wid = t >> 5;
                const unsigned ln = t & 31u;
                if (wid < TS4E_B_PER_KS) {
                    const unsigned kh = GATEUP ? (wid >> 2) : (wid >> 1);
                    const unsigned sub = GATEUP ? ((wid >> 1) & 1u) : 0u;
                    const unsigned tt = wid & 1u;
                    const unsigned col_off_s = GATEUP ? (sub << 6) : 0u;
                    if (ln < CH) {
                        const unsigned char* s = v32_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                        const void* g_s =
                            s + static_cast<unsigned long long>(ks * 4u + kh * 2u + tt) * N + cta_n + ln * 16u;
                        const unsigned dst_s =
                            sbase + L::OFF_S + (sl * 2u + kh) * 512u + (2u * tt + 1u) * 128u + col_off_s + ln * 16u;
                        v32_cp16_pol_s32(dst_s, g_s, pol_b);
                    } else if (ln < CH + 4u) {
                        const unsigned ci = ln - CH;
                        const unsigned idx = ks * TS4E_B_PER_KS + wid;
                        const unsigned cnt = static_cast<unsigned>(
                            reinterpret_cast<const unsigned short*>(smem + L::OFF_BKT_CNT)[idx]);
                        if (cnt > 0u) {
                            const unsigned beg = reinterpret_cast<const unsigned*>(smem + L::OFF_BKT_BEG)[idx];
                            const unsigned pad = beg & 7u;
                            if (pad + cnt <= L::TS4E_EXC_CAP && (ci << 3) < (pad + cnt)) {
                                const unsigned char* blob =
                                    v32_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                                const unsigned short* exc =
                                    reinterpret_cast<const unsigned short*>(blob + Ts4e::EXC_OFF);
                                const unsigned dst_e =
                                    sbase + L::OFF_EXC + ((sl * TS4E_B_PER_KS + wid) * L::TS4E_EXC_CAP + (ci << 3)) * 2u;
                                v32_cp16_pol_s32(dst_e, exc + (beg & ~7u) + (ci << 3), pol_b);
                            }
                        }
                    }
                }
            } else {
                if (t < 64u) {
                    const unsigned kh = t >> 5;
                    const unsigned r2 = (t >> 3) & 3u;
                    const unsigned c2 = t & 7u;
                    const unsigned m2 = GATEUP ? (c2 >> 2) : 0u;
                    const unsigned lc2 = GATEUP ? (c2 & 3u) : c2;
                    const unsigned char* sp = v32_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && m2) ? 3u : 1u);
                    v32_cp16_pol_s32(
                        sbase + L::OFF_S + (sl * 2u + kh) * 512u + r2 * 128u + c2 * 16u,
                        sp + static_cast<unsigned long long>(ks * 8u + kh * 4u + r2) * N + cta_n + lc2 * 16u,
                        pol_b);
                }
            }
        };

        auto decode_ts4e_stage = [&](unsigned ks_dec) {
            unsigned t;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
            const unsigned wid = t >> 5;
            if (wid >= TS4E_B_PER_KS) return;
            const unsigned ln = t & 31u;
            __syncwarp();
            unsigned char (*const sS)[2][4][V32_BC] =
                reinterpret_cast<unsigned char (*)[2][4][V32_BC]>(smem + L::OFF_S);
            const unsigned char* const sBase = smem + L::OFF_BASE;
            const unsigned short (*const sExc)[TS4E_B_PER_KS][L::TS4E_EXC_CAP] =
                reinterpret_cast<const unsigned short (*)[TS4E_B_PER_KS][L::TS4E_EXC_CAP]>(smem + L::OFF_EXC);
            const unsigned sl = ks_dec & 1u;
            const unsigned kh = GATEUP ? (wid >> 2) : (wid >> 1);
            const unsigned sub = GATEUP ? ((wid >> 1) & 1u) : 0u;
            const unsigned tt = wid & 1u;
            const unsigned col_off_s = GATEUP ? (sub << 6) : 0u;
            constexpr unsigned LANES_PER_SWAR = BCOLS / 4u;
            const unsigned idx = ks_dec * TS4E_B_PER_KS + wid;
            const unsigned cnt = static_cast<unsigned>(
                reinterpret_cast<const unsigned short*>(smem + L::OFF_BKT_CNT)[idx]);
            const unsigned beg = reinterpret_cast<const unsigned*>(smem + L::OFF_BKT_BEG)[idx];
            const unsigned pad = beg & 7u;
            const bool fast_exc = (cnt != 0u) && (pad + cnt <= L::TS4E_EXC_CAP);
            const unsigned e_fast = (fast_exc && ln < cnt)
                ? static_cast<unsigned>(sExc[sl][wid][pad + ln])
                : 0u;
            if (ln < LANES_PER_SWAR) {
                const unsigned lc4 = col_off_s + (ln << 2);
                const unsigned base_w = *reinterpret_cast<const unsigned*>(&sBase[lc4]);
                const unsigned nib = *reinterpret_cast<const unsigned*>(&sS[sl][kh][2u * tt + 1u][lc4]);

                const unsigned c0 = nib & 0x0F0F0F0Fu;
                const unsigned e0 = (c0 + 0x01010101u) & 0x10101010u;
                *reinterpret_cast<unsigned*>(&sS[sl][kh][2u * tt][lc4]) = base_w + (c0 & ~(e0 - (e0 >> 4)));

                const unsigned c1 = (nib >> 4) & 0x0F0F0F0Fu;
                const unsigned e1 = (c1 + 0x01010101u) & 0x10101010u;
                *reinterpret_cast<unsigned*>(&sS[sl][kh][2u * tt + 1u][lc4]) = base_w + (c1 & ~(e1 - (e1 >> 4)));
            }
            __syncwarp();
            if (cnt != 0u) {
                if (fast_exc) {
                    if (ln < cnt) {
                        const unsigned col128 = e_fast & 127u;
                        if constexpr (GATEUP) {
                            if ((col128 & 64u) == col_half) {
                                sS[sl][kh][2u * tt + ((e_fast >> 7) & 1u)][col_off_s + (col128 & 63u)] =
                                    static_cast<unsigned char>(e_fast >> 8);
                            }
                        } else {
                            sS[sl][kh][2u * tt + ((e_fast >> 7) & 1u)][col128] =
                                static_cast<unsigned char>(e_fast >> 8);
                        }
                    }
                } else {
                    const unsigned char* blob = v32_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                    const unsigned short* exc =
                        reinterpret_cast<const unsigned short*>(blob + Ts4e::EXC_OFF);
                    for (unsigned i = ln; i < cnt; i += 32u) {
                        const unsigned e = static_cast<unsigned>(__ldg(&exc[beg + i]));
                        const unsigned col128 = e & 127u;
                        if constexpr (GATEUP) {
                            if ((col128 & 64u) == col_half) {
                                sS[sl][kh][2u * tt + ((e >> 7) & 1u)][col_off_s + (col128 & 63u)] =
                                    static_cast<unsigned char>(e >> 8);
                            }
                        } else {
                            sS[sl][kh][2u * tt + ((e >> 7) & 1u)][col128] =
                                static_cast<unsigned char>(e >> 8);
                        }
                    }
                }
            }
        };

        auto pack_sf_stage = [&](unsigned sl) {
            unsigned t;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
            const unsigned kh = t >> 7;
            const unsigned col = t & 127u;
            const unsigned q = col & 3u;
            const unsigned sel1 = (q & 1u) ? 0x3715u : 0x6240u;
            const unsigned sel2 = (q & 2u) ? 0x3276u : 0x5410u;
            const unsigned su = v32_ld_su32(
                sbase + L::OFF_S + (sl * 2u + kh) * 512u + q * 128u + (col & ~3u));
            const unsigned sp = qwen4_nvfp4_detail::prmt_b32(su, __shfl_xor_sync(0xffffffffu, su, 1), sel1);
            const unsigned sc = qwen4_nvfp4_detail::prmt_b32(sp, __shfl_xor_sync(0xffffffffu, sp, 2), sel2) & 0x7f7f7f7fu;
            v32_st_su32(
                sbase + L::OFF_SF + (sl * 2u + kh) * 512u + v32_sf_eo_u32_idx<GATEUP>(col) * 4u,
                sc);
        };

        auto prefetch_b = [&](unsigned pf_k128) {
            if (task_count <= gridDim.x) return;
            unsigned raw_peek;
            asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(raw_peek) : "l"(next_task_ptr));
            if (raw_peek >= task_count - gridDim.x) return;
            const unsigned pf_task = gridDim.x + raw_peek;
            const unsigned pf_raw_wi = pf_task / n_tiles;
            const unsigned pf_cta_n = (pf_task % n_tiles) * TN;
            const unsigned pf_wi = (!GATEUP) ? (wc - 1u - pf_raw_wi) : pf_raw_wi;
            const oi640::WorkItem pf_item = workspace->items[pf_wi];
            if (pf_item.expert >= Shape::EXPERTS || pf_item.expert == item_expert) return;
            if constexpr (!GATEUP) {
                if (pf_wi + 1u < wc && workspace->items[pf_wi + 1u].expert == pf_item.expert) return;
            } else {
                if (pf_item.m_tile != 0u) return;
            }
            unsigned t;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
            const unsigned r = t >> 3;
            const unsigned c = t & 7u;
            if ((c & 1u) == 0u) {
                const unsigned mtx = GATEUP ? (c >> 2) : 0u;
                const unsigned lc = GATEUP ? (c & 3u) : c;
                const unsigned char* pB = reinterpret_cast<const unsigned char*>(
                    (GATEUP && mtx) ? pk1[pf_item.expert] : pk0[pf_item.expert]);
                if (pB != nullptr) {
                    const unsigned char* g0 =
                        pB + static_cast<unsigned long long>(pf_k128 * 64u + r) * N + pf_cta_n + lc * 16u;
                    v32_prefetch_l2(g0);
                    v32_prefetch_l2(g0 + static_cast<unsigned long long>(32u) * N);
                }
            }
            constexpr unsigned S_ROWS_LD = TS4E ? 4u : 8u;
            if (t < S_ROWS_LD * 4u) {
                const unsigned r2 = t >> 2;
                const unsigned c32 = (t & 3u) << 5;
                const unsigned m2 = GATEUP ? (c32 >> 6) : 0u;
                const unsigned lc32 = GATEUP ? (c32 & 63u) : c32;
                const unsigned char* pS = reinterpret_cast<const unsigned char*>(
                    (GATEUP && m2) ? sc1[pf_item.expert] : sc0[pf_item.expert]);
                if (pS != nullptr) {
                    v32_prefetch_l2(pS + static_cast<unsigned long long>(pf_k128 * S_ROWS_LD + r2) * N + pf_cta_n + lc32);
                }
            }
        };

        float acc[2][8][4];
        #pragma unroll
        for (int r = 0; r < 2; ++r)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.0f;

        auto mma_step_k128 = [&](unsigned ks, unsigned sl) {
            unsigned t;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
            const unsigned wid = t >> 5;
            const unsigned wm = (wid >> 1) * 32u;
            const int wv = valid_rows - static_cast<int>(wm);
            if (wv <= 0) return;
            const bool two_rt = (wv > 16);

            const unsigned ln = t & 31u;
            const unsigned gid = ln >> 2;
            const unsigned tig = ln & 3u;
            const unsigned wn = wid & 1u;

            const unsigned as_addr = sbase + L::OFF_AS + (sl * V32_TILE_M + wm + gid + (tig & 1u) * 8u) * 8u;
            const unsigned a_base = sbase + sl * 8192u + v32_swz32(wm + (ln & 15u), ln >> 4);
            const unsigned m_ld = ln >> 3;
            const unsigned i_ld = ln & 7u;
            const unsigned r_ld = ((m_ld & 2u) << 3) + (i_ld << 1) + (m_ld & 1u);
            const unsigned b_base = sbase + L::OFF_B + sl * 8192u + r_ld * 128u;
            const unsigned sf_base0 = sbase + L::OFF_SF + (sl * 2u) * 512u + (wn * 16u + gid) * 16u;

            const unsigned c0 = GATEUP ? (wn << 1) : (wn << 2);
            const unsigned o0 = (c0 ^ ((r_ld >> 1) & 7u)) << 4;
            constexpr unsigned XOR_PAIR2 = GATEUP ? 64u : 32u;

            unsigned nr0, nr1, nr2, nr3;
            qwen4_nvfp4_detail::ldsm4_trans(b_base + o0, nr0, nr1, nr2, nr3);

            auto run_half = [&](unsigned kh_off_a, unsigned kh_off_b, unsigned kh_off_sf,
                                unsigned kh_off_as, unsigned kh_slice, bool prefetch_next_kh, bool do_r1) {
                bool skip_sfa0 = false;
                bool skip_sfa1 = false;
                if constexpr (!GATEUP) {
                    const unsigned s2_row0 = kh_slice * V32_TILE_M + wm + gid;
                    const float r00 = v32_ld_sxs2(sbase + L::OFF_XS2, s2_row0);
                    const float r01 = v32_ld_sxs2(sbase + L::OFF_XS2, s2_row0 + 8u);
                    if (r00 > 0.0f && r00 != 1.0f) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            acc[0][j][0] *= r00;
                            acc[0][j][1] *= r00;
                        }
                    }
                    if (r01 > 0.0f && r01 != 1.0f) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            acc[0][j][2] *= r01;
                            acc[0][j][3] *= r01;
                        }
                    }
                    skip_sfa0 = ((tig & 1u) == 0u) ? (r00 < 0.0f) : (r01 < 0.0f);
                    if (do_r1) {
                        const float r10 = v32_ld_sxs2(sbase + L::OFF_XS2, s2_row0 + 16u);
                        const float r11 = v32_ld_sxs2(sbase + L::OFF_XS2, s2_row0 + 24u);
                        if (r10 > 0.0f && r10 != 1.0f) {
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                acc[1][j][0] *= r10;
                                acc[1][j][1] *= r10;
                            }
                        }
                        if (r11 > 0.0f && r11 != 1.0f) {
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                acc[1][j][2] *= r11;
                                acc[1][j][3] *= r11;
                            }
                        }
                        skip_sfa1 = ((tig & 1u) == 0u) ? (r10 < 0.0f) : (r11 < 0.0f);
                    }
                } else {
                    (void)kh_slice;
                }

                const unsigned sfa0 = skip_sfa0 ? 0u : v32_ld_su32(as_addr + kh_off_as);
                const unsigned sfa1 = (do_r1 && !skip_sfa1) ? v32_ld_su32(as_addr + kh_off_as + 16u * 8u) : 0u;
                unsigned afr0[4], afr1[4];
                qwen4_nvfp4_detail::ldsm4(a_base + kh_off_a, afr0[0], afr0[1], afr0[2], afr0[3]);
                if (do_r1) qwen4_nvfp4_detail::ldsm4(a_base + kh_off_a + 512u, afr1[0], afr1[1], afr1[2], afr1[3]);

                const uint4 sfb0 = v32_ld_sv4(sf_base0 + kh_off_sf);

                // Pair 0 (j = 0, 1)
                {
                    const unsigned b0e = qwen4_nvfp4_detail::prmt_b32(nr0, nr1, 0x6240u);
                    const unsigned b0o = qwen4_nvfp4_detail::prmt_b32(nr0, nr1, 0x7351u);
                    const unsigned b1e = qwen4_nvfp4_detail::prmt_b32(nr2, nr3, 0x6240u);
                    const unsigned b1o = qwen4_nvfp4_detail::prmt_b32(nr2, nr3, 0x7351u);
                    qwen4_nvfp4_detail::ldsm4_trans(b_base + kh_off_b + (o0 ^ 16u), nr0, nr1, nr2, nr3);
                    qwen4_nvfp4_detail::mma_nvfp4(acc[0][0], afr0, b0e, b1e, sfa0, sfb0.x);
                    qwen4_nvfp4_detail::mma_nvfp4(acc[0][1], afr0, b0o, b1o, sfa0, sfb0.y);
                    if (do_r1) {
                        qwen4_nvfp4_detail::mma_nvfp4(acc[1][0], afr1, b0e, b1e, sfa1, sfb0.x);
                        qwen4_nvfp4_detail::mma_nvfp4(acc[1][1], afr1, b0o, b1o, sfa1, sfb0.y);
                    }
                }
                // Pair 1 (j = 2, 3)
                {
                    const unsigned b0e = qwen4_nvfp4_detail::prmt_b32(nr0, nr1, 0x6240u);
                    const unsigned b0o = qwen4_nvfp4_detail::prmt_b32(nr0, nr1, 0x7351u);
                    const unsigned b1e = qwen4_nvfp4_detail::prmt_b32(nr2, nr3, 0x6240u);
                    const unsigned b1o = qwen4_nvfp4_detail::prmt_b32(nr2, nr3, 0x7351u);
                    qwen4_nvfp4_detail::ldsm4_trans(b_base + kh_off_b + (o0 ^ XOR_PAIR2), nr0, nr1, nr2, nr3);
                    qwen4_nvfp4_detail::mma_nvfp4(acc[0][2], afr0, b0e, b1e, sfa0, sfb0.z);
                    qwen4_nvfp4_detail::mma_nvfp4(acc[0][3], afr0, b0o, b1o, sfa0, sfb0.w);
                    if (do_r1) {
                        qwen4_nvfp4_detail::mma_nvfp4(acc[1][2], afr1, b0e, b1e, sfa1, sfb0.z);
                        qwen4_nvfp4_detail::mma_nvfp4(acc[1][3], afr1, b0o, b1o, sfa1, sfb0.w);
                    }
                }

                const uint4 sfb1 = v32_ld_sv4(sf_base0 + kh_off_sf + 128u);

                // Pair 2 (j = 4, 5)
                {
                    const unsigned b0e = qwen4_nvfp4_detail::prmt_b32(nr0, nr1, 0x6240u);
                    const unsigned b0o = qwen4_nvfp4_detail::prmt_b32(nr0, nr1, 0x7351u);
                    const unsigned b1e = qwen4_nvfp4_detail::prmt_b32(nr2, nr3, 0x6240u);
                    const unsigned b1o = qwen4_nvfp4_detail::prmt_b32(nr2, nr3, 0x7351u);
                    qwen4_nvfp4_detail::ldsm4_trans(b_base + kh_off_b + ((o0 ^ XOR_PAIR2) ^ 16u), nr0, nr1, nr2, nr3);
                    qwen4_nvfp4_detail::mma_nvfp4(acc[0][4], afr0, b0e, b1e, sfa0, sfb1.x);
                    qwen4_nvfp4_detail::mma_nvfp4(acc[0][5], afr0, b0o, b1o, sfa0, sfb1.y);
                    if (do_r1) {
                        qwen4_nvfp4_detail::mma_nvfp4(acc[1][4], afr1, b0e, b1e, sfa1, sfb1.x);
                        qwen4_nvfp4_detail::mma_nvfp4(acc[1][5], afr1, b0o, b1o, sfa1, sfb1.y);
                    }
                }
                // Pair 3 (j = 6, 7)
                {
                    const unsigned b0e = qwen4_nvfp4_detail::prmt_b32(nr0, nr1, 0x6240u);
                    const unsigned b0o = qwen4_nvfp4_detail::prmt_b32(nr0, nr1, 0x7351u);
                    const unsigned b1e = qwen4_nvfp4_detail::prmt_b32(nr2, nr3, 0x6240u);
                    const unsigned b1o = qwen4_nvfp4_detail::prmt_b32(nr2, nr3, 0x7351u);
                    if (prefetch_next_kh) {
                        qwen4_nvfp4_detail::ldsm4_trans(b_base + 4096u + o0, nr0, nr1, nr2, nr3);
                    }
                    qwen4_nvfp4_detail::mma_nvfp4(acc[0][6], afr0, b0e, b1e, sfa0, sfb1.z);
                    qwen4_nvfp4_detail::mma_nvfp4(acc[0][7], afr0, b0o, b1o, sfa0, sfb1.w);
                    if (do_r1) {
                        qwen4_nvfp4_detail::mma_nvfp4(acc[1][6], afr1, b0e, b1e, sfa1, sfb1.z);
                        qwen4_nvfp4_detail::mma_nvfp4(acc[1][7], afr1, b0o, b1o, sfa1, sfb1.w);
                    }
                }
            };

            const unsigned kh0 = ks * 2u;
            if (two_rt) {
                run_half(0u,    0u,    0u,   0u, kh0,      true,  true);
                run_half(4096u, 4096u, 512u, 4u, kh0 + 1u, false, true);
            } else {
                run_half(0u,    0u,    0u,   0u, kh0,      true,  false);
                run_half(4096u, 4096u, 512u, 4u, kh0 + 1u, false, false);
            }
        };

        // Prologue: issue stage 0 (K = 0..127).
        load_a(0u);
        load_b(0u);
        qwen4_nvfp4_detail::cp_async_commit();

        #pragma unroll 1
        for (unsigned ks = 0u; ks < KS - 1u; ++ks) {
            const unsigned sl = ks & 1u;
            qwen4_nvfp4_detail::cp_async_wait<0>();
            if constexpr (TS4E) {
                if (ks == 0u) __syncthreads();
                decode_ts4e_stage(ks);
            }
            __syncthreads();
            load_a(ks + 1u);
            load_b(ks + 1u);
            qwen4_nvfp4_detail::cp_async_commit();
            pack_sf_stage(sl);
            __syncthreads();
            mma_step_k128(ks, sl);
        }

        // Final stage (ks = KS - 1) with next-task L2 prefetch.
        {
            const unsigned k_last = KS - 1u;
            const unsigned sl = k_last & 1u;
            qwen4_nvfp4_detail::cp_async_wait<0>();
            if constexpr (TS4E) {
                decode_ts4e_stage(k_last);
            }
            __syncthreads();
            if constexpr (PREFETCH) prefetch_b(0u);
            pack_sf_stage(sl);
            __syncthreads();
            mma_step_k128(k_last, sl);
        }

        // 0-SMEM Register-Direct Epilogue (even/odd paired MMA outputs -> contiguous quads).
        unsigned ep_tid;
        asm volatile("mov.u32 %0, %%tid.x;" : "=r"(ep_tid));
        const unsigned ep_wid = ep_tid >> 5;
        const unsigned ep_lane = ep_tid & 31u;
        const unsigned ep_gid = ep_lane >> 2;
        const unsigned ep_tig = ep_lane & 3u;
        const unsigned ep_wm = (ep_wid >> 1) * 32u;
        const unsigned ep_wn = (ep_wid & 1u) * (GATEUP ? 32u : 64u);
        const int ep_wv = valid_rows - static_cast<int>(ep_wm);
        const int ep_rt = ep_wv <= 0 ? 0 : min(2, (ep_wv + 15) >> 4);
        const float ep_s2a = s20[item_expert];
        const float ep_s2b = GATEUP ? s21[item_expert] : 0.0f;

        if constexpr (GATEUP) {
            const unsigned col = cta_n + ep_wn;
            auto pack_silu_quad = [&](int r, int q_lo, int q_hi, float ga, float gb, unsigned (&u)[4]) {
                #pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const int je = 2 * p;
                    const int jo = je + 1;
                    const float ge0 = acc[r][je][q_lo] * ga, ue0 = acc[r][je + 4][q_lo] * gb;
                    const float go0 = acc[r][jo][q_lo] * ga, uo0 = acc[r][jo + 4][q_lo] * gb;
                    const float ve0 = ge0 * (1.0f / (1.0f + __expf(-ge0))) * ue0;
                    const float vo0 = go0 * (1.0f / (1.0f + __expf(-go0))) * uo0;
                    u[2 * p + 0] = qwen4_nvfp4_detail::pack_bf162(ve0, vo0);

                    const float ge1 = acc[r][je][q_hi] * ga, ue1 = acc[r][je + 4][q_hi] * gb;
                    const float go1 = acc[r][jo][q_hi] * ga, uo1 = acc[r][jo + 4][q_hi] * gb;
                    const float ve1 = ge1 * (1.0f / (1.0f + __expf(-ge1))) * ue1;
                    const float vo1 = go1 * (1.0f / (1.0f + __expf(-go1))) * uo1;
                    u[2 * p + 1] = qwen4_nvfp4_detail::pack_bf162(ve1, vo1);
                }
            };

            if constexpr (OQ_NVFP4) {
                const bool keep_oq = (workspace->rows <= V32_OQ_NVFP4_PIN_MAX_ROWS) ||
                                     (raw_wi + V32_OQ_MXFP8_KEEP_WINDOW >= wc);
                const bool dyn_s2_active = (oq_scale2_out != nullptr);
                const float inv_oq_s2_over_6_scalar =
                    dyn_s2_active ? 0.0f : __fdiv_rn(1.0f, __fmul_rn(oq_scale2, 6.0f));
                const unsigned col_byte4 = (col >> 1) + ep_tig * 4u;
                const unsigned grp_pair = col >> 4;
                if (dyn_s2_active) {
                    __syncthreads();
                }
                auto emit_oq4_row = [&](int r, int q_lo, int q_hi, unsigned lr) {
                    const bool active = (r < ep_rt);
                    uint4 w = make_uint4(0u, 0u, 0u, 0u);
                    float gmax = 0.0f;
                    float half_max = 0.0f;
                    if (active) {
                        const float xs2 = v32_ld_sxs2(sbase + L::OFF_XS2, lr);
                        unsigned u[4];
                        pack_silu_quad(r, q_lo, q_hi, __fmul_rn(ep_s2a, xs2), __fmul_rn(ep_s2b, xs2), u);
                        w = v32_bf16_reg_quad_eo(u[0], u[1], u[2], u[3], ep_tig);
                        const unsigned uw[4] = {w.x, w.y, w.z, w.w};
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            gmax = fmaxf(gmax, fmaxf(fabsf(__uint_as_float(uw[i] << 16)),
                                                     fabsf(__uint_as_float(uw[i] & 0xffff0000u))));
                        }
                        gmax = fmaxf(gmax, __shfl_xor_sync(0xffffffffu, gmax, 1));
                        if (dyn_s2_active) {
                            half_max = fmaxf(gmax, __shfl_xor_sync(0xffffffffu, gmax, 2));
                        }
                    }
                    float row_s2 = oq_scale2;
                    float inv_s2_over_6 = inv_oq_s2_over_6_scalar;
                    if (dyn_s2_active) {
                        float* const s_half = reinterpret_cast<float*>(smem);
                        const unsigned wn_bit = ep_wid & 1u;
                        if (ep_tig == 0u) {
                            s_half[wn_bit * V32_TILE_M + lr] = half_max;
                        }
                        __syncthreads();
                        if (active) {
                            const float other_max = s_half[(wn_bit ^ 1u) * V32_TILE_M + lr];
                            const float slice_amax = fmaxf(half_max, other_max);
                            const bool pos = isfinite(slice_amax) && (slice_amax > 1e-28f);
                            row_s2 = pos ? __fdiv_rn(slice_amax, 2688.0f) : 0.0f;
                            inv_s2_over_6 = pos ? __fdiv_rn(448.0f, slice_amax) : 0.0f;
                        }
                    }
                    if (active) {
                        const unsigned uw[4] = {w.x, w.y, w.z, w.w};
                        float v[8];
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            v[2 * i] = __uint_as_float(uw[i] << 16);
                            v[2 * i + 1] = __uint_as_float(uw[i] & 0xffff0000u);
                        }
                        const float sf_f = (gmax > 0.0f && inv_s2_over_6 > 0.0f)
                            ? fminf(__fmul_rn(gmax, inv_s2_over_6), 448.0f)
                            : 0.0f;
                        const unsigned sf_byte = qwen4_nvfp4_detail::rn_ue4m3(sf_f);
                        const float eff = __fmul_rn(qwen4_nvfp4_detail::e4m3_to_float(sf_byte), row_s2);
                        const float inv_eff = (eff > 0.0f) ? __fdiv_rn(1.0f, eff) : 0.0f;
                        const unsigned packed = qwen4_nvfp4_detail::cvt_f32x8_to_e2m1x8(v, inv_eff);
                        const unsigned sf_other = __shfl_xor_sync(0xffffffffu, sf_byte, 2);
                        const unsigned short sf_pair =
                            static_cast<unsigned short>((sf_byte & 0xffu) | ((sf_other & 0xffu) << 8));
                        if (static_cast<int>(lr) < valid_rows) {
                            const unsigned long long orow = cta_m + lr;
                            if (keep_oq) v32_st4_l2_el(oq + orow * (N / 2u) + col_byte4, packed, pol_el);
                            else         v32_st4_ef(oq + orow * (N / 2u) + col_byte4, packed);
                            if (ep_tig == 0u) {
                                *reinterpret_cast<unsigned short*>(os + orow * (N / 16u) + grp_pair) = sf_pair;
                                if (dyn_s2_active && (ep_wid & 1u) == 0u) {
                                    if (keep_oq) {
                                        v32_st4_l2_el(oq_scale2_out + orow * n_tiles + nt,
                                                      __float_as_uint(row_s2), pol_el);
                                    } else {
                                        oq_scale2_out[orow * n_tiles + nt] = row_s2;
                                    }
                                }
                            }
                        }
                    }
                };
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const unsigned lr0 = ep_wm + r * 16u + ep_gid;
                    const unsigned lr1 = lr0 + 8u;
                    emit_oq4_row(r, 0, 1, lr0);
                    asm volatile("" ::: "memory");
                    emit_oq4_row(r, 2, 3, lr1);
                    asm volatile("" ::: "memory");
                }
            } else {
                const bool keep_oq = (workspace->rows <= 4096u) ||
                                     (raw_wi + V32_OQ_MXFP8_KEEP_WINDOW >= wc);
                const unsigned col8 = col + ep_tig * 8u;
                const unsigned blk = col >> 5;
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    if (r >= ep_rt) break;
                    const unsigned lr0 = ep_wm + r * 16u + ep_gid;
                    const unsigned lr1 = lr0 + 8u;
                    {
                        const float xs2_0 = v32_ld_sxs2(sbase + L::OFF_XS2, lr0);
                        unsigned u0[4];
                        pack_silu_quad(r, 0, 1, __fmul_rn(ep_s2a, xs2_0), __fmul_rn(ep_s2b, xs2_0), u0);
                        unsigned qq0[2], ss0;
                        v32_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], ep_tig, qq0, ss0);
                        if (static_cast<int>(lr0) < valid_rows) {
                            const unsigned long long orow = cta_m + lr0;
                            if (keep_oq) v32_st8_l2_el(oq + orow * N + col8, make_uint2(qq0[0], qq0[1]), pol_el);
                            else         v32_st8_ef(oq + orow * N + col8, make_uint2(qq0[0], qq0[1]));
                            if (ep_tig == 0u) os[orow * (N / 32u) + blk] = static_cast<unsigned char>(ss0);
                        }
                    }
                    asm volatile("" ::: "memory");
                    {
                        const float xs2_1 = v32_ld_sxs2(sbase + L::OFF_XS2, lr1);
                        unsigned u1[4];
                        pack_silu_quad(r, 2, 3, __fmul_rn(ep_s2a, xs2_1), __fmul_rn(ep_s2b, xs2_1), u1);
                        unsigned qq1[2], ss1;
                        v32_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], ep_tig, qq1, ss1);
                        if (static_cast<int>(lr1) < valid_rows) {
                            const unsigned long long orow = cta_m + lr1;
                            if (keep_oq) v32_st8_l2_el(oq + orow * N + col8, make_uint2(qq1[0], qq1[1]), pol_el);
                            else         v32_st8_ef(oq + orow * N + col8, make_uint2(qq1[0], qq1[1]));
                            if (ep_tig == 0u) os[orow * (N / 32u) + blk] = static_cast<unsigned char>(ss1);
                        }
                    }
                    asm volatile("" ::: "memory");
                }
            }
        } else {
            const bool keep_slot =
                (static_cast<unsigned long long>(workspace->expanded) * Shape::SLOT_MXFP8_BYTES_ULL) <=
                V32_KEEP_SLOT_BYTES;
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= ep_rt) break;
                const unsigned lr0 = ep_wm + r * 16u + ep_gid;
                const unsigned lr1 = lr0 + 8u;
                const float dn_scale0 =
                    __fmul_rn(ep_s2a, v32_ld_sxs2(sbase + L::OFF_XS2, DOWN_SLICES * V32_TILE_M + lr0));
                const float dn_scale1 =
                    __fmul_rn(ep_s2a, v32_ld_sxs2(sbase + L::OFF_XS2, DOWN_SLICES * V32_TILE_M + lr1));
                #pragma unroll
                for (unsigned b = 0u; b < 2u; ++b) {
                    const unsigned col = cta_n + ep_wn + b * 32u;
                    const unsigned col8 = col + ep_tig * 8u;
                    const int jb = 4 * static_cast<int>(b);
                    {
                        unsigned u0[4];
                        u0[0] = qwen4_nvfp4_detail::pack_bf162(acc[r][jb + 0][0] * dn_scale0, acc[r][jb + 1][0] * dn_scale0);
                        u0[1] = qwen4_nvfp4_detail::pack_bf162(acc[r][jb + 0][1] * dn_scale0, acc[r][jb + 1][1] * dn_scale0);
                        u0[2] = qwen4_nvfp4_detail::pack_bf162(acc[r][jb + 2][0] * dn_scale0, acc[r][jb + 3][0] * dn_scale0);
                        u0[3] = qwen4_nvfp4_detail::pack_bf162(acc[r][jb + 2][1] * dn_scale0, acc[r][jb + 3][1] * dn_scale0);
                        if constexpr (EPI == 3) {
                            unsigned qq0[2], ss0;
                            v32_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], ep_tig, qq0, ss0);
                            if (static_cast<int>(lr0) < valid_rows) {
                                const unsigned long long prow = cta_m + lr0;
                                if (keep_slot) {
                                    v32_st8_l2_el(slot_q + prow * N + col8, make_uint2(qq0[0], qq0[1]), pol_el);
                                } else {
                                    v32_st8_ef(slot_q + prow * N + col8, make_uint2(qq0[0], qq0[1]));
                                }
                                if (ep_tig == 0u) slot_s[prow * (N / 32u) + (col >> 5)] = static_cast<unsigned char>(ss0);
                            }
                        } else {
                            const uint4 w0 = v32_bf16_reg_quad_eo(u0[0], u0[1], u0[2], u0[3], ep_tig);
                            if (static_cast<int>(lr0) < valid_rows) {
                                v32_st16_stream(C + static_cast<unsigned long long>(cta_m + lr0) * N + col8, w0);
                            }
                        }
                    }
                    asm volatile("" ::: "memory");
                    {
                        unsigned u1[4];
                        u1[0] = qwen4_nvfp4_detail::pack_bf162(acc[r][jb + 0][2] * dn_scale1, acc[r][jb + 1][2] * dn_scale1);
                        u1[1] = qwen4_nvfp4_detail::pack_bf162(acc[r][jb + 0][3] * dn_scale1, acc[r][jb + 1][3] * dn_scale1);
                        u1[2] = qwen4_nvfp4_detail::pack_bf162(acc[r][jb + 2][2] * dn_scale1, acc[r][jb + 3][2] * dn_scale1);
                        u1[3] = qwen4_nvfp4_detail::pack_bf162(acc[r][jb + 2][3] * dn_scale1, acc[r][jb + 3][3] * dn_scale1);
                        if constexpr (EPI == 3) {
                            unsigned qq1[2], ss1;
                            v32_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], ep_tig, qq1, ss1);
                            if (static_cast<int>(lr1) < valid_rows) {
                                const unsigned long long prow = cta_m + lr1;
                                if (keep_slot) {
                                    v32_st8_l2_el(slot_q + prow * N + col8, make_uint2(qq1[0], qq1[1]), pol_el);
                                } else {
                                    v32_st8_ef(slot_q + prow * N + col8, make_uint2(qq1[0], qq1[1]));
                                }
                                if (ep_tig == 0u) slot_s[prow * (N / 32u) + (col >> 5)] = static_cast<unsigned char>(ss1);
                            }
                        } else {
                            const uint4 w1 = v32_bf16_reg_quad_eo(u1[0], u1[1], u1[2], u1[3], ep_tig);
                            if (static_cast<int>(lr1) < valid_rows) {
                                v32_st16_stream(C + static_cast<unsigned long long>(cta_m + lr1) * N + col8, w1);
                            }
                        }
                    }
                    asm volatile("" ::: "memory");
                }
            }
        }

        // Race-free self-resetting atomic work-stealing across all resident CTAs.
        task_slot ^= 1u;
        if (ep_tid == 0u) {
            unsigned next_t;
            if (task_count <= gridDim.x) {
                next_t = task_count;
            } else {
                const unsigned raw = atomicAdd(next_task_ptr, 1u);
                next_t = gridDim.x + raw;
                if (raw == task_count - 1u) {
                    atomicExch(next_task_ptr, 0u);
                }
            }
            v32_st_su32(sbase + L::OFF_TASK + task_slot * 4u, next_t);
        }
        __syncthreads();
        task = v32_ld_su32(sbase + L::OFF_TASK + task_slot * 4u);
    }
}

}  // namespace
