// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// V28 twarp / m64 / m64s W4A8 MoE GEMM kernels (flat and Atlas-TS4E-128 scales).

namespace {

// Four QMMAs per 16-row slice for the M64 8-warp N-split (2 Gate + 2 Up in GATEUP, or 4 Down in DOWN).
template <bool GATEUP, int BIDA>
__device__ __forceinline__ void v28_qmma4(float (&acc)[8][4], const unsigned (&a)[4], const unsigned (&b)[4][2],
                                          unsigned sa, unsigned lo, unsigned hi) {
    v28_qmma<BIDA, 0>(acc[0], a, b[0][0], b[0][1], sa, lo);
    v28_qmma<BIDA, 1>(acc[1], a, b[1][0], b[1][1], sa, lo);
    if constexpr (GATEUP) {
        v28_qmma<BIDA, 0>(acc[2], a, b[2][0], b[2][1], sa, hi);
        v28_qmma<BIDA, 1>(acc[3], a, b[3][0], b[3][1], sa, hi);
    } else {
        v28_qmma<BIDA, 2>(acc[2], a, b[2][0], b[2][1], sa, lo);
        v28_qmma<BIDA, 3>(acc[3], a, b[3][0], b[3][1], sa, lo);
    }
}

__device__ __forceinline__ void v28_twarp_cp16_el(void* smem_ptr, const void* gmem_ptr) {
    unsigned s_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{ .reg .u32 t; .reg .u64 pol;\n"
        "  mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t;\n"
        "  createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n"
        "  cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, 16, pol; }\n"
        : "+r"(s_addr) : "l"(gmem_ptr));
}

__device__ __forceinline__ void v28_twarp_cp16_ef(void* smem_ptr, const void* gmem_ptr) {
    unsigned s_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{ .reg .u32 t; .reg .u64 pol;\n"
        "  mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t;\n"
        "  createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n"
        "  cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, 16, pol; }\n"
        : "+r"(s_addr) : "l"(gmem_ptr));
}

__device__ __forceinline__ unsigned v28_ld_su8_vol(unsigned s_addr) {
    unsigned v;
    asm volatile("ld.shared.u8 %0, [%1];\n" : "=r"(v) : "r"(s_addr));
    return v;
}

template <unsigned OFF>
__device__ __forceinline__ unsigned v28_ld_su8_off_vol(unsigned s_addr) {
    unsigned v;
    asm volatile("ld.shared.u8 %0, [%1+%2];\n" : "=r"(v) : "r"(s_addr), "n"(OFF));
    return v;
}

__device__ __forceinline__ unsigned v28_ld_su32_vol(unsigned s_addr) {
    unsigned v;
    asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(s_addr));
    return v;
}

template <unsigned OFF>
__device__ __forceinline__ unsigned v28_ld_su32_off_vol(unsigned s_addr) {
    unsigned v;
    asm volatile("ld.shared.u32 %0, [%1+%2];\n" : "=r"(v) : "r"(s_addr), "n"(OFF));
    return v;
}

__device__ __forceinline__ uint2 v28_ld_lut_vol(unsigned sbase_lut, unsigned r) {
    uint2 tt;
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n" : "=r"(tt.x), "=r"(tt.y) : "r"(sbase_lut + r * 8u));
    return tt;
}

__device__ __forceinline__ void v28_st_su8_vol(unsigned s_addr, unsigned v) {
    asm volatile("st.shared.u8 [%0], %1;\n" :: "r"(s_addr), "r"(v) : "memory");
}

__device__ __forceinline__ void v28_st_sbt_v2_0(unsigned s_addr, unsigned w0, unsigned w1) {
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n" :: "r"(s_addr), "r"(w0), "r"(w1) : "memory");
}

__device__ __forceinline__ void v28_st_sbt_v2_8(unsigned s_addr, unsigned w2, unsigned w3) {
    asm volatile("st.shared.v2.u32 [%0+8], {%1, %2};\n" :: "r"(s_addr), "r"(w2), "r"(w3) : "memory");
}

__device__ __forceinline__ void v28_twarp_cp16_plain(void* smem_ptr, const void* gmem_ptr) {
    unsigned s_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{ .reg .u32 t;\n"
        "  mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t;\n"
        "  cp.async.cg.shared.global [%0], [%1], 16; }\n"
        : "+r"(s_addr) : "l"(gmem_ptr));
}

__device__ __forceinline__ void v28_twarp_cp16_ef_pred(void* smem_ptr, const void* gmem_ptr, bool pred) {
    unsigned s_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    const int bytes = pred ? 16 : 0;
    asm volatile(
        "{ .reg .u32 t; .reg .u64 pol;\n"
        "  mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t;\n"
        "  createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n"
        "  cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, pol; }\n"
        : "+r"(s_addr) : "l"(gmem_ptr), "r"(bytes));
}

// Exact branchless pure-integer E4M3FN scale normalization: computes
// __nv_cvt_float_to_fp8(v10_e4m3_to_float(s_own) * 2^(-e32), __NV_SATFINITE, __NV_E4M3)
// bit-identically across all 65,536 (s_own, s_oth) byte pairs without FP/CVT instructions.
__device__ __forceinline__ unsigned v28_norm_e4m3_scale(unsigned s_own, int e32) {
    const unsigned sgn = s_own & 0x80u;
    const unsigned mag = s_own & 0x7fu;
    const bool norm = (mag >= 8u);
    const int lz = __clz(static_cast<int>(mag));
    const int exp_b = norm ? static_cast<int>(mag >> 3) : (29 - lz);
    const unsigned sig = norm ? (8u | (mag & 7u)) : (mag << ((lz - 28) & 3));
    const int new_exp = exp_b - e32;
    const unsigned normal_res = sgn | (static_cast<unsigned>(new_exp) << 3) | (sig & 7u);
    const unsigned shift = static_cast<unsigned>(min(max(1, 1 - new_exp), 5));
    const unsigned q = sig >> shift;
    const unsigned half = 1u << (shift - 1u);
    const unsigned sub_res = sgn | ((sig + half - 1u + (q & 1u)) >> shift);
    unsigned res = (new_exp >= 1) ? normal_res : sub_res;
    res = (mag == 0u) ? sgn : res;
    res = (mag == 0x7fu) ? 0x7fu : res;
    return res;
}

// V28 dynamic SMEM layout:
// - TS4E=false preserves exact V14Smem (89,088 B).
// - TS4E=true keeps full decoded sS[3][BMAT][4][BCOLS] (3,072 B, aliasing raw nibbles
//   into odd rows 1 and 3 (2u * t + 1u) during cp.async) and appends multi-warp (4-warp) TS4E staging after LUT (89,088 B)
//   so epilogue sE never aliases:
//     sBase[256] (256 B) + sExc[3][4][32] (768 B) + sBktBeg[KS*4] + sBktCnt[KS*4]
//   -> 91,072 B (GATEUP) / 90,352 B (DOWN), well below GB10's 101,376 B dynamic SMEM cap.
template <bool GATEUP, bool TS4E, typename Shape = Qwen4FlashNextShape> struct V28Smem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V14_BC / BMAT;
    static constexpr unsigned S_ROWS = 4u;
    static constexpr unsigned TS4E_KS = Shape::template gemm_k<GATEUP>() / 64u;
    static constexpr unsigned TS4E_BKTS = TS4E_KS * 4u;
    static constexpr unsigned A = V14_DA * V10_TILE_M * 64;
    static constexpr unsigned B = V14_DB * BMAT * 32 * BCOLS;
    static constexpr unsigned S = V14_DB * BMAT * S_ROWS * BCOLS;
    static constexpr unsigned AS = 2 * V10_TILE_M * 4;
    static constexpr unsigned BT = 2 * V14_BC * 64;
    static constexpr unsigned SF = 2 * 2 * V14_BC;
    static constexpr unsigned OFF_B = A;
    static constexpr unsigned OFF_S = OFF_B + B;
    static constexpr unsigned OFF_AS = OFF_S + S;
    static constexpr unsigned OFF_BT = OFF_AS + AS;
    static constexpr unsigned OFF_SF = OFF_BT + BT;
    static constexpr unsigned OFF_LUT = OFF_SF + SF;
    static constexpr unsigned OFF_BASE = OFF_LUT + 256 * 8;
    static constexpr unsigned OFF_EXC = OFF_BASE + (TS4E ? 256u : 0u);
    static constexpr unsigned OFF_BKT_BEG = OFF_EXC + (TS4E ? (V14_DB * 4u * 32u * 2u) : 0u);
    static constexpr unsigned OFF_BKT_CNT = OFF_BKT_BEG + (TS4E ? (TS4E_BKTS * 4u) : 0u);
    static constexpr unsigned BYTES = OFF_BKT_CNT + (TS4E ? (TS4E_BKTS * 2u) : 0u);
    static_assert(
        !TS4E ? (BYTES == 89088u) : (GATEUP ? (BYTES == 91072u) : (BYTES == 90352u)),
        "V28Smem byte contract"
    );
    static_assert(BYTES <= 101376u, "V28Smem fits GB10 dynamic SMEM cap");
    static_assert(!GATEUP || V10_TILE_M * V14_E_STRIDE * 2 <= OFF_AS, "epilogue staging aliases A/B/S only");
};

// MODE 0 (twarp): 4x4 warp layout for valid_rows > 64, 2x8 m64 N-split for valid_rows <= 64,
//                 512-thread SWAR LDS.32+prmt transcode (interleaved for valid_rows > 32, warps 8..15 for <= 32).
// MODE 1 (m64):   same 2x8 m64 N-split for valid_rows <= 64 with standard task order.
// MODE 2 (m64s):  same as MODE 1 with unconditional 512-thread 1-unit transcode for all valid_rows <= 64.
template <bool GATEUP, int MODE, int EPI = 0, bool PREFETCH = (MODE != 2), bool TS4E = false,
          typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void v28_m64_impl(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ pk0, const unsigned long long* __restrict__ sc0,
    const float* __restrict__ s20,
    const unsigned long long* __restrict__ pk1, const unsigned long long* __restrict__ sc1,
    const float* __restrict__ s21,
    __nv_bfloat16* __restrict__ C, unsigned char* __restrict__ oq, unsigned char* __restrict__ os,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned N_in, unsigned K_in,
    unsigned char* __restrict__ slot_q = nullptr, unsigned char* __restrict__ slot_s = nullptr,
    unsigned wi_begin = 0u, unsigned wi_end = 0xffffffffu
) {
    using L = V28Smem<GATEUP, TS4E, Shape>;
    constexpr unsigned N = Shape::template gemm_n<GATEUP>();
    constexpr unsigned K = Shape::template gemm_k<GATEUP>();
    constexpr unsigned TN = GATEUP ? 128u : 256u;
    constexpr unsigned BMAT = L::BMAT, BCOLS = L::BCOLS, S_ROWS = L::S_ROWS;
    using Ts4e = Qwen4Ts4eLayout<N, K>;
    constexpr unsigned TS4E_N_TILES = Ts4e::N_TILES;
    constexpr unsigned TS4E_NIBBLES_BYTES = Ts4e::NIBBLES_BYTES;
    constexpr unsigned TS4E_PAGE_BASES_OFF = Ts4e::PAGE_BASES_OFF;
    constexpr unsigned TS4E_BKT_OFF_OFF = Ts4e::BKT_OFF_OFF;
    constexpr unsigned TS4E_EXC_OFF = Ts4e::EXC_OFF;
    unsigned dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (gridDim.x == 0 || gridDim.x > oi640::FIXED_GRID || blockDim.x != V14_THREADS || dyn_smem < L::BYTES) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }
    const bool shape = (N_in == N && K_in == K);
    const bool outs = GATEUP ? (oq != nullptr && os != nullptr)
                             : (EPI == 3 ? (slot_q != nullptr && slot_s != nullptr) : (C != nullptr));
    const bool tbl_ok = pk0 != nullptr && sc0 != nullptr && s20 != nullptr &&
        (!GATEUP || (pk1 != nullptr && sc1 != nullptr && s21 != nullptr));
    if (!shape || !outs || !tbl_ok || aq == nullptr || as == nullptr || expert_offsets == nullptr ||
        sorted_token_ids == nullptr || !oi640::contract_ok(contract) ||
        !oi640::workspace_ok(workspace, contract) || status == nullptr || status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_ABI);
        return;
    }
    const unsigned wc = workspace->work_count;
    const bool batched = (wi_end != 0u && wi_end != 0xffffffffu);
    unsigned w_lo = 0u, w_hi = wc;
    if (batched) {
        w_lo = min(wi_begin, wc);
        while (w_lo > 0u && w_lo < wc && workspace->items[w_lo].expert == workspace->items[w_lo - 1u].expert) ++w_lo;
        w_hi = min(wi_end, wc);
        while (w_hi > 0u && w_hi < wc && workspace->items[w_hi].expert == workspace->items[w_hi - 1u].expert) ++w_hi;
        if (w_lo >= w_hi) return;
    }
    constexpr unsigned n_tiles = N / TN;
    const unsigned batch_items = w_hi - w_lo;
    const unsigned task_count = batch_items * n_tiles;
    if (blockIdx.x >= task_count) return;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* const sA = smem;                                                     // [DA][128][64] swz
    unsigned char (*const sB)[BMAT][32][BCOLS] = reinterpret_cast<unsigned char (*)[BMAT][32][BCOLS]>(smem + L::OFF_B);
    unsigned char (*const sS)[BMAT][S_ROWS][BCOLS] = reinterpret_cast<unsigned char (*)[BMAT][S_ROWS][BCOLS]>(smem + L::OFF_S);
    unsigned (*const sAS)[V10_TILE_M] = reinterpret_cast<unsigned (*)[V10_TILE_M]>(smem + L::OFF_AS);
    unsigned char* const sBt = smem + L::OFF_BT;                                        // [2][256][64] swz
    unsigned char (*const sSF)[2][V14_BC] = reinterpret_cast<unsigned char (*)[2][V14_BC]>(smem + L::OFF_SF);
    uint2* const lut = reinterpret_cast<uint2*>(smem + L::OFF_LUT);
    unsigned char* const sBase = smem + L::OFF_BASE;
    unsigned short (*const sExc)[4][32] = reinterpret_cast<unsigned short (*)[4][32]>(smem + L::OFF_EXC);
    unsigned* const sBktBeg = reinterpret_cast<unsigned*>(smem + L::OFF_BKT_BEG);
    unsigned short* const sBktCnt = reinterpret_cast<unsigned short*>(smem + L::OFF_BKT_CNT);

    if (threadIdx.x < 256u) {
        lut[threadIdx.x] = V28_E4M3_LUT[threadIdx.x];
    }

    constexpr unsigned KS = K / 64;
    constexpr unsigned kblocks = K / 32;

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const unsigned raw_wi = task / n_tiles;
        const unsigned wi = (MODE == 0 && !GATEUP) ? (w_hi - 1u - raw_wi) : (w_lo + raw_wi);
        const oi640::WorkItem item = workspace->items[wi];
        const unsigned cta_n = (task % n_tiles) * TN;
        if (item.expert >= Shape::EXPERTS) { if (threadIdx.x == 0u) oi640::fail(status, oi640::ERR_PLAN); return; }
        const int m_start = expert_offsets[item.expert];
        const int M_expert = expert_offsets[item.expert + 1] - m_start;
        const int local = static_cast<int>(item.m_tile * V10_TILE_M);
        const unsigned char* B0 = reinterpret_cast<const unsigned char*>(pk0[item.expert]);
        const unsigned char* S0 = reinterpret_cast<const unsigned char*>(sc0[item.expert]);
        const unsigned char* B1 = GATEUP ? reinterpret_cast<const unsigned char*>(pk1[item.expert]) : B0;
        const unsigned char* S1 = GATEUP ? reinterpret_cast<const unsigned char*>(sc1[item.expert]) : S0;
        if (m_start < 0 || M_expert < 0 || local >= M_expert || B0 == nullptr || S0 == nullptr ||
            B1 == nullptr || S1 == nullptr || !isfinite(s20[item.expert]) || (GATEUP && !isfinite(s21[item.expert]))) {
            if (threadIdx.x == 0u) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const unsigned cta_m = static_cast<unsigned>(m_start + local);
        // Always index oq/os by cta_m (strictly within [0..expanded)) so expert-boundary
        // snapping in batched L2 windows can never overflow into os or past expert_gate_out.
        const unsigned aq_base_row = cta_m;
        const int valid_rows = min(M_expert - local, static_cast<int>(V10_TILE_M));
        const bool m64 = (valid_rows <= 64);
        const int active_a_rows = (valid_rows + 15) & ~15;

        // A: only load 16-row slices that active warps actually read (warp-uniform predicate).
        // Keep B0/B1/S0/S1 warp-uniform (Blackwell UR regs = 0 vector regs) and compute thread-varying
        // A/B/S addresses inside load_a/load_b to minimize persistent vector registers across the K-loop.
        const bool b_reuse = (MODE == 0 && !GATEUP)
            ? (item.m_tile > 0u)
            : (local + static_cast<int>(V10_TILE_M) < M_expert);

        auto load_a = [&](unsigned ks) {
            unsigned la_tid;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(la_tid));
            const unsigned a_row = la_tid >> 2, a_ch = la_tid & 3u;
            if (static_cast<int>(a_row) < active_a_rows) {
                const bool a_ok = static_cast<int>(a_row) < valid_rows;
                const unsigned src_row = a_ok
                    ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + a_row]) : (aq_base_row + a_row))
                    : 0u;
                const unsigned off_a = src_row * K + a_ch * 16u;
                const unsigned char* const g_a = aq + (off_a + ks * 64u);
                unsigned char* const dst_a = sA + (ks % V14_DA) * (V10_TILE_M * 64u) + v14_swz(a_row, a_ch);
                if constexpr (MODE == 0 && !GATEUP) {
                    if (!batched && ((task % n_tiles) + 1u == n_tiles)) {
                        v28_twarp_cp16_ef_pred(dst_a, g_a, a_ok);
                    } else {
                        v10_cp16(dst_a, g_a, a_ok);
                    }
                } else {
                    v10_cp16(dst_a, g_a, a_ok);
                }
            }
            if (la_tid >= 384u && (ks & 1u) == 0u) {
                const unsigned sc_row = la_tid - 384u;
                if (static_cast<int>(sc_row) < active_a_rows) {
                    const bool sc_ok = static_cast<int>(sc_row) < valid_rows;
                    const unsigned sc_src_row = sc_ok
                        ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + sc_row]) : (aq_base_row + sc_row))
                        : 0u;
                    v10_cp4(&sAS[(ks >> 1) & 1u][sc_row], as + (sc_src_row * kblocks + ks * 2u), sc_ok);
                }
            }
        };
        auto load_b = [&](unsigned ks) {
            constexpr unsigned CH = BCOLS / 16u;
            constexpr unsigned S_ROWS_LD = TS4E ? 2u : 4u;
            const unsigned sl = ks % V14_DB;
            unsigned lb_tid;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(lb_tid));
            const unsigned mtx = GATEUP ? (lb_tid >> 8) : 0u;
            const unsigned r = (lb_tid / CH) % 32u, cc = lb_tid % CH;
            // Bank-conflict-free 16B chunk XOR-swizzle on sB: chunk bits 2..1 ^= (r & 3)
            // maps every 4-row x 32-col SWAR LDS.32 block onto all 32 SMEM banks (0 conflicts, 0 extra SMEM).
            const unsigned b_dst_col = (cc << 4) ^ ((r & 3u) << 5);
            const unsigned char* src = (GATEUP && (lb_tid >= 256u)) ? B1 : B0;
            const unsigned char* const g_b =
                src + static_cast<unsigned long long>(ks * 32u + r) * N + cta_n + cc * 16u;
            if constexpr (MODE == 0 || MODE == 1) {
                if (b_reuse) {
                    v28_twarp_cp16_el(&sB[sl][mtx][r][b_dst_col], g_b);
                } else {
                    v28_twarp_cp16_ef(&sB[sl][mtx][r][b_dst_col], g_b);
                }
            } else {
                v28_twarp_cp16_plain(&sB[sl][mtx][r][b_dst_col], g_b);
            }
            if constexpr (TS4E) {
                // 4-warp disjoint bucket ownership (Warps 0..3 = buckets b=0..3):
                // each warp loads its own 128-col nibble slice into row (2*t+1) and its own
                // exception staging sExc[sl][b], so v10_wait<1>() + __syncwarp() makes all 4
                // buckets ready for parallel decode_ts4e_stage with 0 extra __syncthreads().
                const unsigned lb_wid = lb_tid >> 5, lb_lane = lb_tid & 31u;
                if (lb_wid < 4u) {
                    const unsigned sub = lb_wid >> 1;
                    const unsigned t = lb_wid & 1u;
                    const unsigned mtx = GATEUP ? sub : 0u;
                    const unsigned col_hi = GATEUP ? 0u : (sub << 7);
                    if (lb_lane < 8u) {
                        const unsigned char* s = (GATEUP && sub) ? S1 : S0;
                        const unsigned char* const g_s =
                            s + static_cast<unsigned long long>(ks * 2u + t) * N + cta_n + col_hi + lb_lane * 16u;
                        void* const dst_s = &sS[sl][mtx][2u * t + 1u][col_hi + lb_lane * 16u];
                        if constexpr (MODE == 0 || MODE == 1) {
                            if (b_reuse) {
                                v28_twarp_cp16_el(dst_s, g_s);
                            } else {
                                v28_twarp_cp16_ef(dst_s, g_s);
                            }
                        } else {
                            v28_twarp_cp16_plain(dst_s, g_s);
                        }
                    } else if (lb_lane < 12u) {
                        const unsigned c = lb_lane - 8u;
                        const unsigned idx = ks * 4u + lb_wid;
                        const unsigned cnt = static_cast<unsigned>(sBktCnt[idx]);
                        if (cnt > 0u) {
                            const unsigned beg = sBktBeg[idx];
                            const unsigned pad = beg & 7u;
                            if (pad + cnt <= 32u && (c << 3) < (pad + cnt)) {
                                const unsigned char* blob = (GATEUP && sub) ? S1 : S0;
                                const unsigned short* exc =
                                    reinterpret_cast<const unsigned short*>(blob + TS4E_EXC_OFF);
                                v10_cp16(&sExc[sl][lb_wid][c << 3], exc + (beg & ~7u) + (c << 3), true);
                            }
                        }
                    }
                }
            } else {
                if (lb_tid < BMAT * S_ROWS_LD * CH) {
                    const unsigned m2 = lb_tid / (S_ROWS_LD * CH), r2 = (lb_tid / CH) % S_ROWS_LD, c2 = lb_tid % CH;
                    const unsigned char* s = (GATEUP && m2) ? S1 : S0;
                    const unsigned char* const g_s =
                        s + static_cast<unsigned long long>(ks * S_ROWS_LD + r2) * N + cta_n + c2 * 16u;
                    if constexpr (MODE == 0 || MODE == 1) {
                        if (b_reuse) {
                            v28_twarp_cp16_el(&sS[sl][m2][r2][c2 * 16u], g_s);
                        } else {
                            v28_twarp_cp16_ef(&sS[sl][m2][r2][c2 * 16u], g_s);
                        }
                    } else {
                        v28_twarp_cp16_plain(&sS[sl][m2][r2][c2 * 16u], g_s);
                    }
                }
            }
        };
        auto decode_ts4e_stage = [&](unsigned ks_dec) {
            unsigned d_tid;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(d_tid));
            const unsigned d_wid = d_tid >> 5;
            if (d_wid >= 4u) return;
            const unsigned d_lane = d_tid & 31u;
            __syncwarp();
            const unsigned sl = ks_dec % V14_DB;
            const unsigned sub = d_wid >> 1;
            const unsigned t = d_wid & 1u;
            const unsigned mtx = GATEUP ? sub : 0u;
            const unsigned col_hi = GATEUP ? 0u : (sub << 7);
            const unsigned lc4 = col_hi + (d_lane << 2);
            // 4-warp parallel carry-free SWAR decode across all 4 SMSPs (1 warp per disjoint
            // 128-col x 2-k16 bucket, 0 serial loops, 0 extra __syncthreads()): issue sBase, nib,
            // sBktCnt, and sBktBeg together to overlap SMEM latency, and overlap sExc load with SWAR ALU.
            const unsigned idx = ks_dec * 4u + d_wid;
            const unsigned base_w = *reinterpret_cast<const unsigned*>(&sBase[(sub << 7) + (d_lane << 2)]);
            const unsigned nib = *reinterpret_cast<const unsigned*>(&sS[sl][mtx][2u * t + 1u][lc4]);
            const unsigned cnt = static_cast<unsigned>(sBktCnt[idx]);
            const unsigned beg = sBktBeg[idx];
            const unsigned pad = beg & 7u;
            const bool fast_exc = (cnt != 0u) && (pad + cnt <= 32u);
            const unsigned e_fast = (fast_exc && d_lane < cnt)
                ? static_cast<unsigned>(sExc[sl][d_wid][pad + d_lane])
                : 0u;

            const unsigned c0 = nib & 0x0F0F0F0Fu;
            const unsigned e0 = (c0 + 0x01010101u) & 0x10101010u;
            *reinterpret_cast<unsigned*>(&sS[sl][mtx][2u * t][lc4]) = base_w + (c0 & ~(e0 - (e0 >> 4)));

            const unsigned c1 = (nib >> 4) & 0x0F0F0F0Fu;
            const unsigned e1 = (c1 + 0x01010101u) & 0x10101010u;
            *reinterpret_cast<unsigned*>(&sS[sl][mtx][2u * t + 1u][lc4]) = base_w + (c1 & ~(e1 - (e1 >> 4)));
            __syncwarp();

            // O(1) direct scatter-patch for this warp's own disjoint bucket b = d_wid.
            if (cnt != 0u) {
                if (fast_exc) {
                    if (d_lane < cnt) {
                        sS[sl][mtx][2u * t + ((e_fast >> 7) & 1u)][col_hi | (e_fast & 127u)] =
                            static_cast<unsigned char>(e_fast >> 8);
                    }
                } else {
                    const unsigned char* blob = (GATEUP && sub) ? S1 : S0;
                    const unsigned short* exc =
                        reinterpret_cast<const unsigned short*>(blob + TS4E_EXC_OFF);
                    for (unsigned i = d_lane; i < cnt; i += 32u) {
                        const unsigned e = static_cast<unsigned>(exc[beg + i]);
                        sS[sl][mtx][2u * t + ((e >> 7) & 1u)][col_hi | (e & 127u)] =
                            static_cast<unsigned char>(e >> 8);
                    }
                }
            }
        };
        auto prefetch_b = [&](unsigned pf_ks) {
            constexpr unsigned PF_DIST = GATEUP ? 10u : 5u;
            const unsigned pf_raw_wi = raw_wi + PF_DIST;
            if (pf_raw_wi >= batch_items) return;
            const unsigned pf_wi = (MODE == 0 && !GATEUP) ? (w_hi - 1u - pf_raw_wi) : (w_lo + pf_raw_wi);
            const oi640::WorkItem pf_item = workspace->items[pf_wi];
            if (pf_item.expert >= Shape::EXPERTS || pf_item.expert == item.expert) return;
            if constexpr (MODE == 0 && !GATEUP) {
                if (pf_wi + 1u < w_hi && workspace->items[pf_wi + 1u].expert == pf_item.expert) return;
            } else {
                if (pf_item.m_tile != 0u) return;
            }
            constexpr unsigned CH = BCOLS / 16u;
            constexpr unsigned S_ROWS_LD = TS4E ? 2u : 4u;
            unsigned pf_tid;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(pf_tid));
            const unsigned mtx = pf_tid / (32u * CH), r = (pf_tid / CH) % 32u, cc = pf_tid % CH;
            if ((cc & 1u) == 0u) {
                const unsigned char* pB = reinterpret_cast<const unsigned char*>(
                    (GATEUP && mtx) ? pk1[pf_item.expert] : pk0[pf_item.expert]);
                if (pB != nullptr) {
                    v28_prefetch_l2(pB + static_cast<unsigned long long>(pf_ks * 32u + r) * N + cta_n + cc * 16u);
                }
            }
            if (pf_tid < BMAT * S_ROWS_LD * CH) {
                const unsigned m2 = pf_tid / (S_ROWS_LD * CH), r2 = (pf_tid / CH) % S_ROWS_LD, c2 = pf_tid % CH;
                if ((c2 & 1u) == 0u) {
                    const unsigned char* pS = reinterpret_cast<const unsigned char*>(
                        (GATEUP && m2) ? sc1[pf_item.expert] : sc0[pf_item.expert]);
                    if (pS != nullptr) {
                        v28_prefetch_l2(pS + static_cast<unsigned long long>(pf_ks * S_ROWS_LD + r2) * N + cta_n + c2 * 16u);
                    }
                }
            }
            if constexpr (TS4E) {
                if (pf_ks == 0u) {
                    if (pf_tid >= 32u && pf_tid < 40u) {
                        const unsigned j = pf_tid - 32u;
                        const unsigned bm = GATEUP ? (j >> 2) : 0u;
                        const unsigned off32 = GATEUP ? ((j & 3u) << 5) : (j << 5);
                        const unsigned char* pS = reinterpret_cast<const unsigned char*>(
                            (GATEUP && bm) ? sc1[pf_item.expert] : sc0[pf_item.expert]);
                        if (pS != nullptr) {
                            v28_prefetch_l2(pS + static_cast<unsigned long long>(TS4E_NIBBLES_BYTES) + cta_n + off32);
                        }
                    } else if (pf_tid >= 40u && pf_tid < (GATEUP ? 56u : 48u)) {
                        const unsigned j = pf_tid - 40u;
                        const unsigned bm = GATEUP ? (j >> 3) : 0u;
                        const unsigned line = j & 7u;
                        const unsigned byte_off = TS4E_PAGE_BASES_OFF + (line << 7);
                        if (byte_off < TS4E_EXC_OFF) {
                            const unsigned char* pS = reinterpret_cast<const unsigned char*>(
                                (GATEUP && bm) ? sc1[pf_item.expert] : sc0[pf_item.expert]);
                            if (pS != nullptr) {
                                v28_prefetch_l2(pS + byte_off);
                            }
                        }
                    }
                }
            }
        };
        // 4-thread-cooperative SWAR transcode: 2x conflict-free LDS.32 per half-k32 (4x LDS.32/k64 vs 16x LDS.U8)
        // + 2-stage butterfly SHFL.BFLY + prmt.b32 4x4 byte transpose, bit-exact to byte-wise sB reads.
        // Uses asm volatile SMEM ops and fresh %tid.x so ptxas never hoists transcode loads above
        // emit_pair(0..2) or coalesces w0..w3 into a 4-reg quad across u1/p1/c1.
        auto transcode_half = [&](unsigned sl, unsigned tb, unsigned t_h_override, unsigned t) {
            unsigned tc_tid;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tc_tid));
            const unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smem));
            const unsigned t_col = tc_tid & 255u;
            const unsigned t_h = (t_h_override == 2u) ? (tc_tid >> 8) : t_h_override;
            const unsigned t_q = tc_tid & 3u;
            const unsigned t_mtx = GATEUP ? (t_col >> 7) : 0u;
            const unsigned t_lc = GATEUP ? (t_col & 127u) : t_col;
            const unsigned ss_base =
                sbase + L::OFF_S + (((sl * BMAT + t_mtx) * S_ROWS + 2u * t) * BCOLS) + t_lc;
            const unsigned s0 = v28_ld_su8_vol(ss_base);
            const unsigned s1 = v28_ld_su8_off_vol<BCOLS>(ss_base);
            const unsigned s_own = (t_h == 0u) ? s0 : s1;
            const unsigned s_max = max(s0 & 0x7fu, s1 & 0x7fu);
            const int e32 = (s_max >= 8u) ? (static_cast<int>(s_max >> 3) - 7)
                          : (s_max != 0u) ? (22 - __clz(static_cast<int>(s_max)))
                                          : 0;
            const unsigned r = v28_norm_e4m3_scale(s_own, e32);
            const uint2 tt = v28_ld_lut_vol(sbase + L::OFF_LUT, r);
            const unsigned c4_swz = (t_lc & ~3u) ^ (t_q << 5);
            const unsigned r0 = 16u * t + 8u * t_h + t_q;
            const unsigned sb_addr =
                sbase + L::OFF_B + (((sl * BMAT + t_mtx) * 32u + r0) * BCOLS) + c4_swz;
            const unsigned u0 = v28_ld_su32_vol(sb_addr);
            const unsigned sel1 = (t_q & 1u) ? 0x3715u : 0x6240u;
            const unsigned sel2 = (t_q & 2u) ? 0x3276u : 0x5410u;
            const unsigned p0 = v10_prmt(u0, __shfl_xor_sync(0xffffffffu, u0, 1), sel1);
            const unsigned c0 = v10_prmt(p0, __shfl_xor_sync(0xffffffffu, p0, 2), sel2);
            const unsigned u1 = v28_ld_su32_off_vol<4u * BCOLS>(sb_addr);
            const unsigned w0 = v10_e2m1x4_to_e4m3x4(c0, tt.x, tt.y);
            const unsigned w1 = v10_e2m1x4_to_e4m3x4(c0 >> 16, tt.x, tt.y);
            const unsigned tc_dst_s = sbase + L::OFF_BT + tb * (V14_BC * 64u) + v14_swz(t_col, 2u * t + t_h);
            v28_st_sbt_v2_0(tc_dst_s, w0, w1);
            const unsigned p1 = v10_prmt(u1, __shfl_xor_sync(0xffffffffu, u1, 1), sel1);
            const unsigned c1 = v10_prmt(p1, __shfl_xor_sync(0xffffffffu, p1, 2), sel2);
            const unsigned w2 = v10_e2m1x4_to_e4m3x4(c1, tt.x, tt.y);
            const unsigned w3 = v10_e2m1x4_to_e4m3x4(c1 >> 16, tt.x, tt.y);
            v28_st_sbt_v2_8(tc_dst_s, w2, w3);
            if (t_h == 0u) {
                const unsigned sf_addr =
                    sbase + L::OFF_SF + (tb * 2u + t) * V14_BC + v28_sf_index<GATEUP>(t_col);
                v28_st_su8_vol(sf_addr, static_cast<unsigned>(e32 + 127) & 0xffu);
            }
        };
        auto transcode_unit = [&](unsigned sl, unsigned tb, unsigned t_h_override) {
            #pragma unroll
            for (unsigned t = 0; t < 2; ++t) {
                transcode_half(sl, tb, t_h_override, t);
            }
        };

        if constexpr (TS4E) {
            static_assert(KS >= 3u, "TS4E prologue resolves ks=0..2 inline on warps 0..3");
            auto resolve_ts4e_bkt = [&](unsigned idx) {
                const unsigned ks_i = idx >> 2;
                const unsigned b = idx & 3u;
                const unsigned sub = b >> 1;
                const unsigned t = b & 1u;
                const unsigned char* blob = (GATEUP && sub) ? S1 : S0;
                const unsigned bkt = (2u * ks_i + t) * TS4E_N_TILES + (cta_n >> 7) + (GATEUP ? 0u : sub);
                const unsigned* page_bases =
                    reinterpret_cast<const unsigned*>(blob + TS4E_PAGE_BASES_OFF);
                const unsigned short* bkt_off =
                    reinterpret_cast<const unsigned short*>(blob + TS4E_BKT_OFF_OFF);
                const unsigned beg = page_bases[bkt >> 7] + static_cast<unsigned>(bkt_off[bkt]);
                const unsigned end =
                    page_bases[(bkt + 1u) >> 7] + static_cast<unsigned>(bkt_off[bkt + 1u]);
                sBktBeg[idx] = beg;
                sBktCnt[idx] = static_cast<unsigned short>(end - beg);
            };
            unsigned tp_tid;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tp_tid));
            const unsigned tp_wid = tp_tid >> 5, tp_lane = tp_tid & 31u;
            if (tp_wid < 4u) {
                if (tp_lane < 3u) {
                    resolve_ts4e_bkt(tp_lane * 4u + tp_wid);
                }
                __syncwarp();
                if (tp_wid == 0u && tp_lane < 16u) {
                    constexpr unsigned BASE_CH = BCOLS / 16u;
                    const unsigned bm = tp_lane / BASE_CH;
                    const unsigned bc = tp_lane % BASE_CH;
                    const unsigned char* bs = (GATEUP && bm) ? S1 : S0;
                    v10_cp16(&sBase[tp_lane * 16u],
                             bs + static_cast<unsigned long long>(TS4E_NIBBLES_BYTES) + cta_n + bc * 16u,
                             true);
                }
            } else {
                const unsigned idx = (tp_tid - 128u) + 12u;
                if (idx < KS * 4u) {
                    resolve_ts4e_bkt(idx);
                }
            }
        }

        // prologue: Gp0 = {B0 (+ sBase/sExc0)}, Gp1 = {A0, B1 (+ sExc1)}, Gp2 = {A1, B2 (+ sExc2)}
        load_b(0);
        v10_commit();
        load_a(0);
        if (KS > 1) load_b(1);
        v10_commit();
        if (KS > 1) load_a(1);
        if (KS > 2) load_b(2);
        v10_commit();
        v10_wait<2>();
        if constexpr (TS4E) {
            __syncthreads();
            decode_ts4e_stage(0u);
        }
        __syncthreads();
        // All 512 threads participate in the prologue transcode (1 unit per thread instead of 2).
        transcode_unit(0, 0, 2u);

        if (m64) {
            // 2x8 warp N-split path for valid_rows <= 64 (acc[2][4][4] = 32 regs only):
            // 0 in-loop m64 branching, bfr[2][4] ping-pong, interleaved SWAR transcode
            // (or warps 8..15 concurrent transcode when valid_rows <= 32), register-direct epilogue.
            // Fresh %tid.x prevents LLVM from hoisting m64 sSF/swizzle addresses into the !m64 path.
            unsigned tid;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
            const unsigned warp_id = tid >> 5, lane = tid & 31u, gid = lane >> 2, tig = lane & 3u;
            const unsigned warp_m = (warp_id >> 3) * 32u;
            const unsigned warp_n = (warp_id & 7u) * (GATEUP ? 16u : 32u);
            const int warp_valid = valid_rows - static_cast<int>(warp_m);
            const int rt_active = warp_valid <= 0 ? 0 : min(2, (warp_valid + 15) >> 4);
            const unsigned sf_off64 = (((warp_id & 6u) * 4u + gid) << 3) +
                                      ((warp_id & 1u) << (GATEUP ? 1u : 2u));
            const unsigned swz_a0 = v14_swz(warp_m + (lane & 15u), lane >> 4);
            const unsigned swz_b0 = v14_swz(warp_n + ((lane >> 4) << 3) + (lane & 7u), (lane >> 3) & 1u);
            constexpr unsigned off_b1 = GATEUP ? (BCOLS * 64u) : 1024u;
            constexpr int BIDB2 = GATEUP ? 0 : 2;
            constexpr int BIDB3 = GATEUP ? 1 : 3;

            float acc[2][4][4];
            #pragma unroll
            for (int r = 0; r < 2; ++r)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.0f;

            unsigned sfa[2] = {0u, 0u};
            auto step_m64 = [&](auto par_tag, unsigned ks) {
                constexpr unsigned PAR = decltype(par_tag)::value;
                constexpr unsigned NEXT_TB = PAR ^ 1u;
                const bool do_tc = (PAR == 0u) ? true : (ks + 1u < KS);
                v10_wait<1>();
                if constexpr (TS4E) {
                    if (do_tc) decode_ts4e_stage(ks + 1u);
                }
                __syncthreads();
                if (ks + 2 < KS) load_a(ks + 2);
                if (ks + 3 < KS) load_b(ks + 3);
                else if constexpr (PREFETCH) prefetch_b(ks + 3u - KS);
                v10_commit();
                const unsigned next_sl = (ks + 1u) % V14_DB;
                if (rt_active == 0) {
                    if (do_tc) {
                        if constexpr (MODE != 2) {
                            transcode_unit(next_sl, NEXT_TB, 0u);
                            transcode_unit(next_sl, NEXT_TB, 1u);
                        } else {
                            transcode_unit(next_sl, NEXT_TB, 2u);
                        }
                    }
                    return;
                }
                if constexpr (PAR == 0) {
                    const unsigned as_buf = (ks >> 1) & 1;
                    sfa[0] = sAS[as_buf][warp_m + gid + (tig & 1) * 8];
                    if (rt_active > 1) sfa[1] = sAS[as_buf][warp_m + 16 + gid + (tig & 1) * 8];
                }
                const unsigned a_slot = ks % V14_DA;
                const bool run_tc = do_tc && ((MODE == 2) ? (valid_rows > 0) : (valid_rows > 32));
                #pragma unroll
                for (unsigned t = 0; t < 2; ++t) {
                    const unsigned sfb_lo = GATEUP
                        ? static_cast<unsigned>(*reinterpret_cast<const unsigned short*>(&sSF[PAR][t][sf_off64]))
                        : *reinterpret_cast<const unsigned*>(&sSF[PAR][t][sf_off64]);
                    const unsigned addr_a0 = static_cast<unsigned>(__cvta_generic_to_shared(
                        sA + a_slot * (V10_TILE_M * 64u) + (swz_a0 ^ (t << 5))));
                    const unsigned addr_b0 = static_cast<unsigned>(__cvta_generic_to_shared(
                        sBt + PAR * (V14_BC * 64u) + (swz_b0 ^ (t << 5))));
                    unsigned afr0[4], afr1[4];
                    unsigned bfr[2][4];
                    v10_ldsm4(addr_b0, bfr[0][0], bfr[0][1], bfr[0][2], bfr[0][3]);
                    v10_ldsm4(addr_a0, afr0[0], afr0[1], afr0[2], afr0[3]);
                    if (rt_active > 1) {
                        v10_ldsm4(addr_a0 + 1024u, afr1[0], afr1[1], afr1[2], afr1[3]);
                    }
                    v10_ldsm4(addr_b0 + off_b1, bfr[1][0], bfr[1][1], bfr[1][2], bfr[1][3]);
                    if (t == 0u) {
                        v28_qmma<2 * PAR, 0>(acc[0][0], afr0, bfr[0][0], bfr[0][1], sfa[0], sfb_lo);
                        v28_qmma<2 * PAR, 1>(acc[0][1], afr0, bfr[0][2], bfr[0][3], sfa[0], sfb_lo);
                        if (rt_active > 1) {
                            v28_qmma<2 * PAR, 0>(acc[1][0], afr1, bfr[0][0], bfr[0][1], sfa[1], sfb_lo);
                            v28_qmma<2 * PAR, 1>(acc[1][1], afr1, bfr[0][2], bfr[0][3], sfa[1], sfb_lo);
                        }
                    } else {
                        v28_qmma<2 * PAR + 1, 0>(acc[0][0], afr0, bfr[0][0], bfr[0][1], sfa[0], sfb_lo);
                        v28_qmma<2 * PAR + 1, 1>(acc[0][1], afr0, bfr[0][2], bfr[0][3], sfa[0], sfb_lo);
                        if (rt_active > 1) {
                            v28_qmma<2 * PAR + 1, 0>(acc[1][0], afr1, bfr[0][0], bfr[0][1], sfa[1], sfb_lo);
                            v28_qmma<2 * PAR + 1, 1>(acc[1][1], afr1, bfr[0][2], bfr[0][3], sfa[1], sfb_lo);
                        }
                    }
                    const unsigned sfb_hi = GATEUP
                        ? static_cast<unsigned>(*reinterpret_cast<const unsigned short*>(&sSF[PAR][t][sf_off64 + 4]))
                        : sfb_lo;
                    if (t == 0u) {
                        v28_qmma<2 * PAR, BIDB2>(acc[0][2], afr0, bfr[1][0], bfr[1][1], sfa[0], sfb_hi);
                        v28_qmma<2 * PAR, BIDB3>(acc[0][3], afr0, bfr[1][2], bfr[1][3], sfa[0], sfb_hi);
                        if (rt_active > 1) {
                            v28_qmma<2 * PAR, BIDB2>(acc[1][2], afr1, bfr[1][0], bfr[1][1], sfa[1], sfb_hi);
                            v28_qmma<2 * PAR, BIDB3>(acc[1][3], afr1, bfr[1][2], bfr[1][3], sfa[1], sfb_hi);
                        }
                    } else {
                        v28_qmma<2 * PAR + 1, BIDB2>(acc[0][2], afr0, bfr[1][0], bfr[1][1], sfa[0], sfb_hi);
                        v28_qmma<2 * PAR + 1, BIDB3>(acc[0][3], afr0, bfr[1][2], bfr[1][3], sfa[0], sfb_hi);
                        if (rt_active > 1) {
                            v28_qmma<2 * PAR + 1, BIDB2>(acc[1][2], afr1, bfr[1][0], bfr[1][1], sfa[1], sfb_hi);
                            v28_qmma<2 * PAR + 1, BIDB3>(acc[1][3], afr1, bfr[1][2], bfr[1][3], sfa[1], sfb_hi);
                        }
                    }
                    if (run_tc) {
                        transcode_half(next_sl, NEXT_TB, 2u, t);
                    }
                }
            };
            #pragma unroll 1
            for (unsigned ks = 0; ks < KS; ks += 2) {
                step_m64(std::integral_constant<unsigned, 0>{}, ks);
                step_m64(std::integral_constant<unsigned, 1>{}, ks + 1);
            }
            v10_wait<0>();

            const float s2a = s20[item.expert];
            const float s2b = GATEUP ? s21[item.expert] : 0.0f;
            if constexpr (GATEUP) {
                // Register-direct m64 2x8 GATEUP MXFP8 epilogue (0 sE staging, 0 __syncthreads):
                // sB (L::OFF_B, 24 KiB) is completely dead & synchronized after the final ks=KS-1 barrier
                // (do_tc=false at ks=KS-1), so sMax[64][8] (2,048 B) safely aliases smem + L::OFF_B.
                // Each partner warp pair {wn, wn^1} exchanges per-16-col absmax via 64-thread bar.sync.
                const bool keep_oq = (MODE != 0) || batched || workspace->rows <= 4096u || raw_wi + 256u >= batch_items;
                float (*const sMax)[8] = reinterpret_cast<float (*)[8]>(smem + L::OFF_B);
                const unsigned wn = warp_id & 7u;
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    if (r >= rt_active) break;
                    const unsigned lr0 = warp_m + r * 16u + gid, lr1 = lr0 + 8u;
                    float f0[4], f1[4];
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        float v[4];
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const float g = acc[r][j][q] * s2a;
                            const float u = acc[r][j + 2][q] * s2b;
                            v[q] = g * (1.0f / (1.0f + __expf(-g))) * u;
                        }
                        const unsigned b0 = v28_pack_bf162(v[0], v[1]);
                        const unsigned b1 = v28_pack_bf162(v[2], v[3]);
                        f0[2 * j] = __uint_as_float(b0 << 16);
                        f0[2 * j + 1] = __uint_as_float(b0 & 0xffff0000u);
                        f1[2 * j] = __uint_as_float(b1 << 16);
                        f1[2 * j + 1] = __uint_as_float(b1 & 0xffff0000u);
                    }
                    float amax0 = 0.0f, amax1 = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        amax0 = fmaxf(amax0, fabsf(f0[i]));
                        amax1 = fmaxf(amax1, fabsf(f1[i]));
                    }
                    amax0 = fmaxf(amax0, __shfl_xor_sync(0xffffffffu, amax0, 1));
                    amax0 = fmaxf(amax0, __shfl_xor_sync(0xffffffffu, amax0, 2));
                    amax1 = fmaxf(amax1, __shfl_xor_sync(0xffffffffu, amax1, 1));
                    amax1 = fmaxf(amax1, __shfl_xor_sync(0xffffffffu, amax1, 2));
                    if (tig == 0u) {
                        sMax[lr0][wn] = amax0;
                        sMax[lr1][wn] = amax1;
                    }
                    const unsigned pair_id = warp_id >> 1;
                    if (pair_id == 0u)      asm volatile("bar.sync 1, 64;\n" ::: "memory");
                    else if (pair_id == 1u) asm volatile("bar.sync 2, 64;\n" ::: "memory");
                    else if (pair_id == 2u) asm volatile("bar.sync 3, 64;\n" ::: "memory");
                    else if (pair_id == 3u) asm volatile("bar.sync 4, 64;\n" ::: "memory");
                    else if (pair_id == 4u) asm volatile("bar.sync 5, 64;\n" ::: "memory");
                    else if (pair_id == 5u) asm volatile("bar.sync 6, 64;\n" ::: "memory");
                    else if (pair_id == 6u) asm volatile("bar.sync 7, 64;\n" ::: "memory");
                    else                    asm volatile("bar.sync 8, 64;\n" ::: "memory");
                    amax0 = fmaxf(amax0, sMax[lr0][wn ^ 1u]);
                    amax1 = fmaxf(amax1, sMax[lr1][wn ^ 1u]);

                    unsigned short r0_j0 = 0u, r0_j1 = 0u, r1_j0 = 0u, r1_j1 = 0u;
                    unsigned ss0 = 0u, ss1 = 0u;
                    if (amax0 > 0.0f) {
                        int e;
                        const float m = frexpf(__fmul_rn(amax0, 1.0f / 448.0f), &e);
                        int k = (m == 0.5f) ? e - 1 : e;
                        k = max(-127, min(127, k));
                        const float inv = __int_as_float((127 - k) << 23);
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r0_j0) : "f"(f0[0] * inv), "f"(f0[1] * inv));
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r0_j1) : "f"(f0[2] * inv), "f"(f0[3] * inv));
                        ss0 = static_cast<unsigned>(k + 127);
                    }
                    if (amax1 > 0.0f) {
                        int e;
                        const float m = frexpf(__fmul_rn(amax1, 1.0f / 448.0f), &e);
                        int k = (m == 0.5f) ? e - 1 : e;
                        k = max(-127, min(127, k));
                        const float inv = __int_as_float((127 - k) << 23);
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r1_j0) : "f"(f1[0] * inv), "f"(f1[1] * inv));
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r1_j1) : "f"(f1[2] * inv), "f"(f1[3] * inv));
                        ss1 = static_cast<unsigned>(k + 127);
                    }
                    const bool odd1 = (tig & 1u) != 0u;
                    const unsigned b0_ex = __shfl_xor_sync(0xffffffffu, static_cast<unsigned>(odd1 ? r0_j0 : r1_j0), 1);
                    const unsigned b1_ex = __shfl_xor_sync(0xffffffffu, static_cast<unsigned>(odd1 ? r0_j1 : r1_j1), 1);
                    const unsigned w_j0 = (odd1 ? b0_ex : static_cast<unsigned>(r0_j0)) |
                                          ((odd1 ? static_cast<unsigned>(r1_j0) : b0_ex) << 16);
                    const unsigned w_j1 = (odd1 ? b1_ex : static_cast<unsigned>(r0_j1)) |
                                          ((odd1 ? static_cast<unsigned>(r1_j1) : b1_ex) << 16);
                    const bool odd2 = (tig & 2u) != 0u;
                    const unsigned recv2 = __shfl_xor_sync(0xffffffffu, odd2 ? w_j0 : w_j1, 2);
                    const unsigned out_x = odd2 ? recv2 : w_j0;
                    const unsigned out_y = odd2 ? w_j1 : recv2;
                    const unsigned out_row = odd1 ? lr1 : lr0;
                    if (static_cast<int>(out_row) < valid_rows) {
                        const unsigned long long orow = aq_base_row + out_row;
                        if (keep_oq) {
                            v28_st8_evict_last(oq + orow * N + cta_n + warp_n + (odd2 ? 8u : 0u), make_uint2(out_x, out_y));
                        } else {
                            v28_st8_evict_first(oq + orow * N + cta_n + warp_n + (odd2 ? 8u : 0u), make_uint2(out_x, out_y));
                        }
                        if ((wn & 1u) == 0u && !odd2) {
                            os[orow * (N / 32u) + cta_n / 32u + (wn >> 1)] =
                                static_cast<unsigned char>(odd1 ? ss1 : ss0);
                        }
                    }
                }
            } else {
                const bool keep_slot =
                    (static_cast<unsigned long long>(workspace->expanded) * Shape::SLOT_MXFP8_BYTES_ULL) <= V28_KEEP_SLOT_BYTES;
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    if (r >= rt_active) break;
                    const unsigned lr0 = warp_m + r * 16u + gid, lr1 = lr0 + 8u;
                    unsigned u0[4], u1[4];
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        u0[jj] = v28_pack_bf162(acc[r][jj][0] * s2a, acc[r][jj][1] * s2a);
                        u1[jj] = v28_pack_bf162(acc[r][jj][2] * s2a, acc[r][jj][3] * s2a);
                    }
                    const unsigned col = cta_n + warp_n;
                    const unsigned col8 = col + tig * 8u;
                    if constexpr (EPI == 3) {
                        unsigned qq0[2], ss0, qq1[2], ss1;
                        v28_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], tig, qq0, ss0);
                        if (static_cast<int>(lr0) < valid_rows) {
                            const unsigned long long prow = cta_m + lr0;
                            if (keep_slot) {
                                v28_st8_l2_evict_last(slot_q + prow * N + col8, make_uint2(qq0[0], qq0[1]));
                            } else {
                                v28_st8_evict_first(slot_q + prow * N + col8, make_uint2(qq0[0], qq0[1]));
                            }
                            if (tig == 0u) slot_s[prow * (N / 32u) + col / 32u] = static_cast<unsigned char>(ss0);
                        }
                        v28_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], tig, qq1, ss1);
                        if (static_cast<int>(lr1) < valid_rows) {
                            const unsigned long long prow = cta_m + lr1;
                            if (keep_slot) {
                                v28_st8_l2_evict_last(slot_q + prow * N + col8, make_uint2(qq1[0], qq1[1]));
                            } else {
                                v28_st8_evict_first(slot_q + prow * N + col8, make_uint2(qq1[0], qq1[1]));
                            }
                            if (tig == 0u) slot_s[prow * (N / 32u) + col / 32u] = static_cast<unsigned char>(ss1);
                        }
                    } else {
                        const uint4 w0 = v28_bf16_reg_quad(u0[0], u0[1], u0[2], u0[3], tig);
                        if (static_cast<int>(lr0) < valid_rows) {
                            v28_st16_stream(C + static_cast<unsigned long long>(cta_m + lr0) * N + col8, w0);
                        }
                        const uint4 w1 = v28_bf16_reg_quad(u1[0], u1[1], u1[2], u1[3], tig);
                        if (static_cast<int>(lr1) < valid_rows) {
                            v28_st16_stream(C + static_cast<unsigned long long>(cta_m + lr1) * N + col8, w1);
                        }
                    }
                }
            }
        } else {
            // 4x4 warp layout for valid_rows > 64: clean register budget (0 m64 state live),
            // bfr[2][4] ping-pong, interleaved SWAR transcode, register-direct epilogue.
            // Fresh %tid.x prevents LLVM from hoisting !m64 sSF/swizzle addresses across task/m64.
            unsigned tid;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
            const unsigned warp_id = tid >> 5, lane = tid & 31u, gid = lane >> 2, tig = lane & 3u;
            const unsigned warp_m = (warp_id >> 2) * 32u;
            const unsigned warp_n = (warp_id & 3u) * (GATEUP ? 32u : 64u);
            const int warp_valid = valid_rows - static_cast<int>(warp_m);
            const int rt_active = warp_valid <= 0 ? 0 : min(2, (warp_valid + 15) >> 4);
            const unsigned swz_a0 = v14_swz(warp_m + (lane & 15u), lane >> 4);
            const unsigned swz_b0 = v14_swz(warp_n + ((lane >> 4) << 3) + (lane & 7u), (lane >> 3) & 1u);

            float acc[2][8][4];
            #pragma unroll
            for (int r = 0; r < 2; ++r)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.0f;

            unsigned sfa[2] = {0u, 0u};
            auto step_128 = [&](auto par_tag, unsigned ks) {
                constexpr unsigned PAR = decltype(par_tag)::value;
                constexpr unsigned NEXT_TB = PAR ^ 1u;
                const bool do_tc = (PAR == 0u) ? true : (ks + 1u < KS);
                v10_wait<1>();
                if constexpr (TS4E) {
                    if (do_tc) decode_ts4e_stage(ks + 1u);
                }
                __syncthreads();
                if (ks + 2 < KS) load_a(ks + 2);
                if (ks + 3 < KS) load_b(ks + 3);
                else if constexpr (PREFETCH) prefetch_b(ks + 3u - KS);
                v10_commit();
                const unsigned next_sl = (ks + 1u) % V14_DB;
                if (rt_active == 0) {
                    if (do_tc) transcode_unit(next_sl, NEXT_TB, 2u);
                    return;
                }
                if constexpr (PAR == 0) {
                    const unsigned as_buf = (ks >> 1) & 1u;
                    sfa[0] = sAS[as_buf][warp_m + gid + (tig & 1) * 8];
                    if (rt_active > 1) sfa[1] = sAS[as_buf][warp_m + 16 + gid + (tig & 1) * 8];
                }
                const unsigned a_slot = ks % V14_DA;
                const bool run_tc = do_tc;
                #pragma unroll
                for (unsigned t = 0; t < 2; ++t) {
                    const uint2 sfb = *reinterpret_cast<const uint2*>(&sSF[PAR][t][((warp_id & 3) * 8 + gid) << 3]);
                    const unsigned addr_a0 = static_cast<unsigned>(__cvta_generic_to_shared(
                        sA + a_slot * (V10_TILE_M * 64u) + (swz_a0 ^ (t << 5))));
                    const unsigned addr_b0 = static_cast<unsigned>(__cvta_generic_to_shared(
                        sBt + PAR * (V14_BC * 64u) + (swz_b0 ^ (t << 5))));
                    constexpr unsigned off0 = 0u;
                    constexpr unsigned off1 = 1024u;
                    constexpr unsigned off2 = GATEUP ? (BCOLS * 64u) : 2048u;
                    constexpr unsigned off3 = GATEUP ? (BCOLS * 64u + 1024u) : 3072u;
                    unsigned afr0[4], afr1[4];
                    auto emit_pair = [&](auto p_tag, const unsigned (&bf)[4], unsigned sfb_w) {
                        constexpr int P = decltype(p_tag)::value;
                        constexpr int BIDB0 = (P & 1) * 2;
                        constexpr int BIDB1 = BIDB0 + 1;
                        if (t == 0) {
                            v28_qmma<2 * PAR, BIDB0>(acc[0][2 * P], afr0, bf[0], bf[1], sfa[0], sfb_w);
                            v28_qmma<2 * PAR, BIDB1>(acc[0][2 * P + 1], afr0, bf[2], bf[3], sfa[0], sfb_w);
                            if (rt_active > 1) {
                                v28_qmma<2 * PAR, BIDB0>(acc[1][2 * P], afr1, bf[0], bf[1], sfa[1], sfb_w);
                                v28_qmma<2 * PAR, BIDB1>(acc[1][2 * P + 1], afr1, bf[2], bf[3], sfa[1], sfb_w);
                            }
                        } else {
                            v28_qmma<2 * PAR + 1, BIDB0>(acc[0][2 * P], afr0, bf[0], bf[1], sfa[0], sfb_w);
                            v28_qmma<2 * PAR + 1, BIDB1>(acc[0][2 * P + 1], afr0, bf[2], bf[3], sfa[0], sfb_w);
                            if (rt_active > 1) {
                                v28_qmma<2 * PAR + 1, BIDB0>(acc[1][2 * P], afr1, bf[0], bf[1], sfa[1], sfb_w);
                                v28_qmma<2 * PAR + 1, BIDB1>(acc[1][2 * P + 1], afr1, bf[2], bf[3], sfa[1], sfb_w);
                            }
                        }
                    };
                    unsigned bfr[2][4];
                    v10_ldsm4(addr_b0 + off0, bfr[0][0], bfr[0][1], bfr[0][2], bfr[0][3]);
                    v10_ldsm4(addr_a0, afr0[0], afr0[1], afr0[2], afr0[3]);
                    if (rt_active > 1) {
                        v10_ldsm4(addr_a0 + 1024u, afr1[0], afr1[1], afr1[2], afr1[3]);
                    }
                    v10_ldsm4(addr_b0 + off1, bfr[1][0], bfr[1][1], bfr[1][2], bfr[1][3]);
                    emit_pair(std::integral_constant<int, 0>{}, bfr[0], sfb.x);
                    v10_ldsm4(addr_b0 + off2, bfr[0][0], bfr[0][1], bfr[0][2], bfr[0][3]);
                    emit_pair(std::integral_constant<int, 1>{}, bfr[1], sfb.x);
                    v10_ldsm4(addr_b0 + off3, bfr[1][0], bfr[1][1], bfr[1][2], bfr[1][3]);
                    emit_pair(std::integral_constant<int, 2>{}, bfr[0], sfb.y);
                    emit_pair(std::integral_constant<int, 3>{}, bfr[1], sfb.y);
                    if (run_tc) {
                        transcode_half(next_sl, NEXT_TB, 2u, t);
                    }
                }
            };
            #pragma unroll 1
            for (unsigned ks = 0; ks < KS; ks += 2) {
                step_128(std::integral_constant<unsigned, 0>{}, ks);
                step_128(std::integral_constant<unsigned, 1>{}, ks + 1);
            }
            v10_wait<0>();

            unsigned ep_tid;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(ep_tid));
            const unsigned ep_wid = ep_tid >> 5, ep_lane = ep_tid & 31u;
            const unsigned ep_gid = ep_lane >> 2, ep_tig = ep_lane & 3u;
            const unsigned ep_wm = (ep_wid >> 2) * 32u;
            const unsigned ep_wn = (ep_wid & 3u) * (GATEUP ? 32u : 64u);
            const int ep_wv = valid_rows - static_cast<int>(ep_wm);
            const int ep_rt = ep_wv <= 0 ? 0 : min(2, (ep_wv + 15) >> 4);
            const float s2a = s20[item.expert];
            const float s2b = GATEUP ? s21[item.expert] : 0.0f;
            if constexpr (GATEUP) {
                const bool keep_oq = (MODE != 0) || batched || workspace->rows <= 4096u || raw_wi + 256u >= batch_items;
                const unsigned blk = ep_wid & 3u;
                const unsigned col8 = ep_wn + ep_tig * 8u;
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    if (r >= ep_rt) break;
                    const unsigned lr0 = ep_wm + r * 16 + ep_gid, lr1 = lr0 + 8;
                    unsigned u0[4], u1[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float v[4];
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const float g = acc[r][j][q] * s2a;
                            const float u = acc[r][j + 4][q] * s2b;
                            v[q] = g * (1.0f / (1.0f + __expf(-g))) * u;
                        }
                        u0[j] = v28_pack_bf162(v[0], v[1]);
                        u1[j] = v28_pack_bf162(v[2], v[3]);
                    }
                    unsigned qq0[2], ss0, qq1[2], ss1;
                    v28_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], ep_tig, qq0, ss0);
                    if (static_cast<int>(lr0) < valid_rows) {
                        const unsigned long long orow = aq_base_row + lr0;
                        if (keep_oq) v28_st8_evict_last(oq + orow * N + cta_n + col8, make_uint2(qq0[0], qq0[1]));
                        else         v28_st8_evict_first(oq + orow * N + cta_n + col8, make_uint2(qq0[0], qq0[1]));
                        if (ep_tig == 0u) os[orow * (N / 32) + cta_n / 32 + blk] = static_cast<unsigned char>(ss0);
                    }
                    v28_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], ep_tig, qq1, ss1);
                    if (static_cast<int>(lr1) < valid_rows) {
                        const unsigned long long orow = aq_base_row + lr1;
                        if (keep_oq) v28_st8_evict_last(oq + orow * N + cta_n + col8, make_uint2(qq1[0], qq1[1]));
                        else         v28_st8_evict_first(oq + orow * N + cta_n + col8, make_uint2(qq1[0], qq1[1]));
                        if (ep_tig == 0u) os[orow * (N / 32) + cta_n / 32 + blk] = static_cast<unsigned char>(ss1);
                    }
                }
            } else {
                const bool keep_slot =
                    (static_cast<unsigned long long>(workspace->expanded) * Shape::SLOT_MXFP8_BYTES_ULL) <= V28_KEEP_SLOT_BYTES;
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    if (r >= ep_rt) break;
                    const unsigned lr0 = ep_wm + r * 16 + ep_gid, lr1 = lr0 + 8;
                    #pragma unroll
                    for (unsigned b = 0; b < 2; ++b) {
                        unsigned u0[4], u1[4];
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int j = 4 * static_cast<int>(b) + jj;
                            u0[jj] = v28_pack_bf162(acc[r][j][0] * s2a, acc[r][j][1] * s2a);
                            u1[jj] = v28_pack_bf162(acc[r][j][2] * s2a, acc[r][j][3] * s2a);
                        }
                        const unsigned col = cta_n + ep_wn + b * 32u;
                        const unsigned col8 = col + ep_tig * 8u;
                        if constexpr (EPI == 3) {
                            unsigned qq0[2], ss0, qq1[2], ss1;
                            v28_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], ep_tig, qq0, ss0);
                            if (static_cast<int>(lr0) < valid_rows) {
                                const unsigned long long prow = cta_m + lr0;
                                if (keep_slot) {
                                    v28_st8_l2_evict_last(slot_q + prow * N + col8, make_uint2(qq0[0], qq0[1]));
                                } else {
                                    v28_st8_evict_first(slot_q + prow * N + col8, make_uint2(qq0[0], qq0[1]));
                                }
                                if (ep_tig == 0u) slot_s[prow * (N / 32) + col / 32] = static_cast<unsigned char>(ss0);
                            }
                            v28_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], ep_tig, qq1, ss1);
                            if (static_cast<int>(lr1) < valid_rows) {
                                const unsigned long long prow = cta_m + lr1;
                                if (keep_slot) {
                                    v28_st8_l2_evict_last(slot_q + prow * N + col8, make_uint2(qq1[0], qq1[1]));
                                } else {
                                    v28_st8_evict_first(slot_q + prow * N + col8, make_uint2(qq1[0], qq1[1]));
                                }
                                if (ep_tig == 0u) slot_s[prow * (N / 32) + col / 32] = static_cast<unsigned char>(ss1);
                            }
                        } else {
                            const uint4 w0 = v28_bf16_reg_quad(u0[0], u0[1], u0[2], u0[3], ep_tig);
                            if (static_cast<int>(lr0) < valid_rows) {
                                v28_st16_stream(C + static_cast<unsigned long long>(cta_m + lr0) * N + col8, w0);
                            }
                            const uint4 w1 = v28_bf16_reg_quad(u1[0], u1[1], u1[2], u1[3], ep_tig);
                            if (static_cast<int>(lr1) < valid_rows) {
                                v28_st16_stream(C + static_cast<unsigned long long>(cta_m + lr1) * N + col8, w1);
                            }
                        }
                    }
                }
            }
        }
        if (task + gridDim.x < task_count) __syncthreads();
    }
}

} // namespace

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v28_m64(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v28_m64_impl<true, 1>(aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
                          up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
                          status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28_m64(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m64_impl<false, 1>(aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
                           expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28q_m64(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m64_impl<false, 1, 3>(aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
                              slot_q, slot_s);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v28_twarp(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v28_m64_impl<true, 0>(aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
                          up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
                          status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28_twarp(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m64_impl<false, 0>(aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
                           expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28q_twarp(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m64_impl<false, 0, 3>(aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
                              slot_q, slot_s);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v28_twarp_win(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K,
    unsigned int wi_begin, unsigned int wi_end
) {
    v28_m64_impl<true, 0>(aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
                          up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
                          status, N, K, nullptr, nullptr, wi_begin, wi_end);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28_twarp_win(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K,
    unsigned int wi_begin, unsigned int wi_end
) {
    v28_m64_impl<false, 0>(aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
                           expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
                           nullptr, nullptr, wi_begin, wi_end);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28q_twarp_win(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K,
    unsigned int wi_begin, unsigned int wi_end
) {
    v28_m64_impl<false, 0, 3>(aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
                              slot_q, slot_s, wi_begin, wi_end);
}

// Atlas-TS4E-128 compressed-scale twins of the v28_twarp and v28_twarp_win kernels
// (91,072 B GATEUP / 90,352 B DOWN dynamic SMEM; 4-warp pre-barrier SWAR + O(1) scatter-patch).
extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v28_twarp_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v28_m64_impl<true, 0, 0, true, true>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28_twarp_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m64_impl<false, 0, 0, true, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28q_twarp_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m64_impl<false, 0, 3, true, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
        slot_q, slot_s);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v28_twarp_win_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K,
    unsigned int wi_begin, unsigned int wi_end
) {
    v28_m64_impl<true, 0, 0, true, true>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K, nullptr, nullptr, wi_begin, wi_end);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28_twarp_win_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K,
    unsigned int wi_begin, unsigned int wi_end
) {
    v28_m64_impl<false, 0, 0, true, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
        nullptr, nullptr, wi_begin, wi_end);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28q_twarp_win_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K,
    unsigned int wi_begin, unsigned int wi_end
) {
    v28_m64_impl<false, 0, 3, true, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
        slot_q, slot_s, wi_begin, wi_end);
}

// Atlas-TS4E-128 compressed-scale twins of the v28_m64 2x8 N-split kernels
// (91,072 B GATEUP / 90,352 B DOWN dynamic SMEM; 2x8 warp N-split for valid_rows <= 64).
extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v28_twarp_m64_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v28_m64_impl<true, 1, 0, true, true>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28_twarp_m64_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m64_impl<false, 1, 0, true, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28q_twarp_m64_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m64_impl<false, 1, 3, true, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
        slot_q, slot_s);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v28_m64s(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v28_m64_impl<true, 2>(aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
                          up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
                          status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28_m64s(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m64_impl<false, 2>(aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
                           expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

