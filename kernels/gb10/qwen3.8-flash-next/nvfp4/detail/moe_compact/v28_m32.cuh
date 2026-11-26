// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// V28 M32 2-CTA/SM W4A8 MoE GEMM kernels (flat and Atlas-TS4E-128 scales).

// =========================================================================
// 2-CTA/SM Small-Row V28 Kernel (TILE_M = 32, BC = 128, 256 threads = 8 warps)
// Dynamic SMEM <= 41,536 B (40.56 KiB) -> 2 resident CTAs/SM on GB10 (99 KiB SMEM),
// doubling active N-tiles (n_tiles=10 for I=640, n_tiles=20 for H=2560) and hiding
// multi-warp (2-warp) TS4E decode stalls across both small batches (rows<=32) and KS=10 down GEMM
// (rows=6..128). Handles V10_TILE_M=128 planner work items with an inner 32-row loop
// (executes once for all experts with <=32 rows; exact for hot experts up to 128 rows).
// =========================================================================
namespace {

constexpr unsigned V28_M32_TILE_M = 32u;
constexpr unsigned V28_M32_BC = 128u;
constexpr unsigned V28_M32_THREADS = 256u;

template <bool GATEUP, bool TS4E, typename Shape = Qwen4FlashNextShape> struct V28M32Smem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V28_M32_BC / BMAT; // 64 (GATEUP) or 128 (DOWN)
    static constexpr unsigned S_ROWS = 4u;
    static constexpr unsigned TS4E_KS = Shape::template gemm_k<GATEUP>() / 64u;
    static constexpr unsigned TS4E_B_PER_KS = GATEUP ? 4u : 2u;
    static constexpr unsigned TS4E_BKTS = TS4E_KS * TS4E_B_PER_KS;
    static constexpr unsigned A = V14_DA * V28_M32_TILE_M * 64u;       // 6,144 B
    static constexpr unsigned B = V14_DB * BMAT * 32u * BCOLS;         // 12,288 B
    static constexpr unsigned S = V14_DB * BMAT * S_ROWS * BCOLS;      // 1,536 B
    static constexpr unsigned AS = 2u * V28_M32_TILE_M * 4u;           // 256 B
    static constexpr unsigned BT = 2u * V28_M32_BC * 64u;              // 16,384 B
    static constexpr unsigned SF = 2u * 2u * V28_M32_BC;               // 512 B
    static constexpr unsigned LUT = 256u * 8u;                         // 2,048 B SMEM lut[256] (avoids LDC.64 serialization)
    static constexpr unsigned MAX = GATEUP ? (V28_M32_TILE_M * 4u * 4u) : 0u; // 512 B sMax[32][4]
    static constexpr unsigned OFF_B = A;
    static constexpr unsigned OFF_S = OFF_B + B;
    static constexpr unsigned OFF_AS = OFF_S + S;
    static constexpr unsigned OFF_BT = OFF_AS + AS;
    static constexpr unsigned OFF_SF = OFF_BT + BT;
    static constexpr unsigned OFF_LUT = OFF_SF + SF;                   // 37,120 B
    static constexpr unsigned OFF_MAX = OFF_LUT + LUT;                 // 39,168 B
    static constexpr unsigned OFF_BASE = OFF_MAX + MAX;                // 39,680 B (GU) / 39,168 B (DN)
    static constexpr unsigned OFF_EXC = OFF_BASE + (TS4E ? 128u : 0u);
    static constexpr unsigned OFF_BKT_BEG = OFF_EXC + (TS4E ? (V14_DB * TS4E_B_PER_KS * 32u * 2u) : 0u);
    static constexpr unsigned OFF_BKT_CNT = OFF_BKT_BEG + (TS4E ? (TS4E_BKTS * 4u) : 0u);
    static constexpr unsigned RAW_BYTES = OFF_BKT_CNT + (TS4E ? (TS4E_BKTS * 2u) : 0u);
    static constexpr unsigned BYTES = (RAW_BYTES + 15u) & ~15u;
    static_assert(
        !TS4E ? (GATEUP ? (BYTES == 39680u) : (BYTES == 39168u))
              : (GATEUP ? (BYTES == 41536u) : (BYTES == 39808u)),
        "V28M32Smem byte contract"
    );
    static_assert(BYTES <= 49152u, "V28M32Smem fits 2 CTAs/SM (<= 48 KiB)");
};

// Packed 4-byte per-(wn, gid) scale index for the 2x4 warp layout (8 warps, BC=128):
// - GATEUP (BCOLS=64, TN=64): wn = (lc >> 4) in 0..3, gid = lc & 7, j = (mtx << 1) | ((lc >> 3) & 1) in 0..3
// - DOWN   (BCOLS=128, TN=128): wn = (t_col >> 5) in 0..3, gid = t_col & 7, j = (t_col >> 3) & 3 in 0..3
// Each (wn, gid) reads one 32-bit word (`LDS.32`) feeding QMMAs j=0..3 via BIDB=0..3.
template <bool GATEUP>
__device__ __forceinline__ unsigned v28_m32_sf_index(unsigned t_col) {
    if constexpr (GATEUP) {
        const unsigned mtx = t_col >> 6, lc = t_col & 63u;
        return ((((lc >> 4) * 8u) + (lc & 7u)) << 2) + ((mtx << 1) | ((lc >> 3) & 1u));
    } else {
        return ((((t_col >> 5) * 8u) + (t_col & 7u)) << 2) + ((t_col >> 3) & 3u);
    }
}

template <bool GATEUP, int EPI = 0, bool TS4E = false, typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void v28_m32_impl(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ pk0, const unsigned long long* __restrict__ sc0,
    const float* __restrict__ s20,
    const unsigned long long* __restrict__ pk1, const unsigned long long* __restrict__ sc1,
    const float* __restrict__ s21,
    __nv_bfloat16* __restrict__ C, unsigned char* __restrict__ oq, unsigned char* __restrict__ os,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned N_in, unsigned K_in,
    unsigned char* __restrict__ slot_q = nullptr, unsigned char* __restrict__ slot_s = nullptr
) {
    using L = V28M32Smem<GATEUP, TS4E, Shape>;
    constexpr unsigned N = Shape::template gemm_n<GATEUP>();
    constexpr unsigned K = Shape::template gemm_k<GATEUP>();
    constexpr unsigned TN = GATEUP ? 64u : 128u;
    constexpr unsigned BMAT = L::BMAT, BCOLS = L::BCOLS, S_ROWS = L::S_ROWS;
    constexpr unsigned TS4E_B_PER_KS = L::TS4E_B_PER_KS;
    using Ts4e = Qwen4Ts4eLayout<N, K>;
    constexpr unsigned TS4E_N_TILES = Ts4e::N_TILES;
    constexpr unsigned TS4E_NIBBLES_BYTES = Ts4e::NIBBLES_BYTES;
    constexpr unsigned TS4E_PAGE_BASES_OFF = Ts4e::PAGE_BASES_OFF;
    constexpr unsigned TS4E_BKT_OFF_OFF = Ts4e::BKT_OFF_OFF;
    constexpr unsigned TS4E_EXC_OFF = Ts4e::EXC_OFF;
    unsigned dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (gridDim.x == 0 || gridDim.x > oi640::FIXED_GRID || blockDim.x != V28_M32_THREADS || dyn_smem < L::BYTES) {
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
    constexpr unsigned n_tiles = N / TN; // 10 for GATEUP (I=640), 20 for DOWN (H=2560)
    const unsigned wc = workspace->work_count;
    const unsigned task_count = wc * n_tiles;
    if (blockIdx.x >= task_count) return;

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* const sA = smem;                                                        // [3][32][64] swz
    unsigned char (*const sB)[BMAT][32][BCOLS] = reinterpret_cast<unsigned char (*)[BMAT][32][BCOLS]>(smem + L::OFF_B);
    unsigned char (*const sS)[BMAT][S_ROWS][BCOLS] = reinterpret_cast<unsigned char (*)[BMAT][S_ROWS][BCOLS]>(smem + L::OFF_S);
    unsigned (*const sAS)[V28_M32_TILE_M] = reinterpret_cast<unsigned (*)[V28_M32_TILE_M]>(smem + L::OFF_AS);
    unsigned char* const sBt = smem + L::OFF_BT;                                           // [2][128][64] swz
    unsigned char (*const sSF)[2][V28_M32_BC] = reinterpret_cast<unsigned char (*)[2][V28_M32_BC]>(smem + L::OFF_SF);
    uint2* const lut = reinterpret_cast<uint2*>(smem + L::OFF_LUT);
    float (*const sMax)[4] = reinterpret_cast<float (*)[4]>(smem + L::OFF_MAX);
    unsigned char* const sBase = smem + L::OFF_BASE;
    unsigned short (*const sExc)[TS4E_B_PER_KS][32] = reinterpret_cast<unsigned short (*)[TS4E_B_PER_KS][32]>(smem + L::OFF_EXC);
    unsigned* const sBktBeg = reinterpret_cast<unsigned*>(smem + L::OFF_BKT_BEG);
    unsigned short* const sBktCnt = reinterpret_cast<unsigned short*>(smem + L::OFF_BKT_CNT);

    const unsigned tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31u;
    lut[tid] = V28_E4M3_LUT[tid];
    const unsigned gid = lane >> 2, tig = lane & 3u;
    const unsigned wn = warp_id & 3u;
    const unsigned warp_m = (warp_id >> 2) * 16u;               // 0 (warps 0..3) or 16 (warps 4..7)
    const unsigned warp_n = wn * (GATEUP ? 16u : 32u);
    const unsigned sf_off = (wn * 8u + gid) << 2;
    constexpr unsigned KS = K / 64u;
    constexpr unsigned kblocks = K / 32u;
    const bool keep_slot =
        (static_cast<unsigned long long>(workspace->expanded) * Shape::SLOT_MXFP8_BYTES_ULL) <= V28_KEEP_SLOT_BYTES;

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const oi640::WorkItem item = workspace->items[task / n_tiles];
        const unsigned cta_n = (task % n_tiles) * TN;
        if (item.expert >= Shape::EXPERTS) { if (tid == 0) oi640::fail(status, oi640::ERR_PLAN); return; }
        const int m_start = expert_offsets[item.expert];
        const int M_expert = expert_offsets[item.expert + 1] - m_start;
        const int local = static_cast<int>(item.m_tile * V10_TILE_M);
        const unsigned char* B0 = reinterpret_cast<const unsigned char*>(pk0[item.expert]);
        const unsigned char* S0 = reinterpret_cast<const unsigned char*>(sc0[item.expert]);
        const unsigned char* B1 = GATEUP ? reinterpret_cast<const unsigned char*>(pk1[item.expert]) : B0;
        const unsigned char* S1 = GATEUP ? reinterpret_cast<const unsigned char*>(sc1[item.expert]) : S0;
        if (m_start < 0 || M_expert < 0 || local >= M_expert || B0 == nullptr || S0 == nullptr ||
            B1 == nullptr || S1 == nullptr || !isfinite(s20[item.expert]) || (GATEUP && !isfinite(s21[item.expert]))) {
            if (tid == 0) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const int item_rows = min(M_expert - local, static_cast<int>(V10_TILE_M));
        const unsigned col_half = cta_n & 64u; // 0 or 64 inside the 128-col TS4E bucket (GATEUP only)
        const unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smem));

        if constexpr (TS4E) {
            static_assert(KS >= 3u, "TS4E prologue resolves ks=0..2 inline on warps 0..TS4E_B_PER_KS-1");
            constexpr unsigned PROLOGUE_BKTS = 3u * TS4E_B_PER_KS;
            constexpr unsigned TS4E_WARP_THREADS = TS4E_B_PER_KS * 32u;
            constexpr unsigned OTHER_THREADS = V28_M32_THREADS - TS4E_WARP_THREADS;
            auto resolve_ts4e_bkt = [&](unsigned idx) {
                const unsigned ks_i = idx / TS4E_B_PER_KS;
                const unsigned b = idx & (TS4E_B_PER_KS - 1u);
                const unsigned sub = b >> 1;
                const unsigned t = b & 1u;
                const unsigned char* blob = (GATEUP && sub) ? S1 : S0;
                const unsigned bkt = (2u * ks_i + t) * TS4E_N_TILES + (cta_n >> 7);
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
            if (warp_id < TS4E_B_PER_KS) {
                if (lane < 3u) {
                    resolve_ts4e_bkt(lane * TS4E_B_PER_KS + warp_id);
                }
                __syncwarp();
                if (warp_id == 0u && lane < 8u) {
                    constexpr unsigned BASE_CH = BCOLS / 16u;
                    const unsigned bm = lane / BASE_CH;
                    const unsigned bc = lane % BASE_CH;
                    const unsigned char* bs = (GATEUP && bm) ? S1 : S0;
                    v10_cp16(&sBase[lane * 16u],
                             bs + static_cast<unsigned long long>(TS4E_NIBBLES_BYTES) + cta_n + bc * 16u,
                             true);
                }
            } else {
                for (unsigned idx = PROLOGUE_BKTS + (tid - TS4E_WARP_THREADS);
                     idx < KS * TS4E_B_PER_KS; idx += OTHER_THREADS) {
                    resolve_ts4e_bkt(idx);
                }
            }
        }

        // Loop in 32-row slices over the planner's V10_TILE_M=128 item (1 iteration for all
        // experts with <=32 rows; exact coverage when a hot expert receives 33..128 tokens).
        for (int m_sub = 0; m_sub < item_rows; m_sub += static_cast<int>(V28_M32_TILE_M)) {
            const unsigned cta_m = static_cast<unsigned>(m_start + local + m_sub);
            const int valid_rows = min(item_rows - m_sub, static_cast<int>(V28_M32_TILE_M));
            const bool warp_active = (static_cast<int>(warp_m) < valid_rows);
            const int active_a_rows = (valid_rows + 15) & ~15;
            const bool b_evict_first = (M_expert <= static_cast<int>(V28_M32_TILE_M));

            const unsigned a_row = tid >> 2, a_ch = tid & 3u;
            const bool a_need = (tid < 128u) && (static_cast<int>(a_row) < active_a_rows);
            const bool a_ok = (tid < 128u) && (static_cast<int>(a_row) < valid_rows);
            const unsigned a_src_row =
                a_ok ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + a_row]) : cta_m + a_row) : 0u;
            const unsigned char* a_src = aq + static_cast<unsigned long long>(a_src_row) * K + a_ch * 16u;
            const unsigned a_dst = v14_swz(a_row, a_ch);

            const unsigned sc_row = tid - 128u;
            const bool sc_need = (tid >= 128u && tid < 160u) && (static_cast<int>(sc_row) < active_a_rows);
            const bool sc_ok = (tid >= 128u && tid < 160u) && (static_cast<int>(sc_row) < valid_rows);
            const unsigned sc_src_row =
                sc_ok ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + sc_row]) : cta_m + sc_row) : 0u;
            const unsigned char* sc_src = as + static_cast<unsigned long long>(sc_src_row) * kblocks;

            auto load_a = [&](unsigned ks) {
                if (a_need) {
                    v10_cp16(sA + (ks % V14_DA) * (V28_M32_TILE_M * 64u) + a_dst, a_src + ks * 64u, a_ok);
                }
                if (sc_need && (ks & 1u) == 0u) {
                    v10_cp4(&sAS[(ks >> 1) & 1u][sc_row], sc_src + ks * 2u, sc_ok);
                }
            };
            auto load_b = [&](unsigned ks) {
                constexpr unsigned CH = BCOLS / 16u;
                constexpr unsigned S_ROWS_LD = TS4E ? 2u : 4u;
                constexpr unsigned SB_SWZ_SHIFT = GATEUP ? 4u : 5u;
                const unsigned sl = ks % V14_DB;
                unsigned lb_tid;
                asm volatile("mov.u32 %0, %%tid.x;" : "=r"(lb_tid));
                const unsigned mtx = lb_tid / (32u * CH), r = (lb_tid / CH) % 32u, cc = lb_tid % CH;
                // Conflict-free 16B chunk XOR-swizzle on sB: all 32 lanes in SWAR LDS.32 hit 32 distinct banks.
                const unsigned b_dst_col = (cc << 4) ^ ((r & 3u) << SB_SWZ_SHIFT);
                const unsigned char* src = (GATEUP && mtx) ? B1 : B0;
                const void* g_b = src + static_cast<unsigned long long>(ks * 32u + r) * N + cta_n + cc * 16u;
                if (b_evict_first) {
                    v28_twarp_cp16_ef(&sB[sl][mtx][r][b_dst_col], g_b);
                } else {
                    v10_cp16(&sB[sl][mtx][r][b_dst_col], g_b, true);
                }
                if constexpr (TS4E) {
                    // 4-warp (GATEUP) / 2-warp (DOWN) disjoint bucket ownership (Warps 0..TS4E_B_PER_KS-1):
                    // each warp loads its own BCOLS nibble slice into row (2*t+1) and its own sExc[sl][b],
                    // so v10_wait<1>() + __syncwarp() makes all buckets ready for parallel decode_ts4e_stage
                    // with 0 extra __syncthreads().
                    const unsigned lb_wid = lb_tid >> 5, lb_lane = lb_tid & 31u;
                    if (lb_wid < TS4E_B_PER_KS) {
                        const unsigned sub = lb_wid >> 1;
                        const unsigned t = lb_wid & 1u;
                        const unsigned s_mtx = GATEUP ? sub : 0u;
                        if (lb_lane < CH) {
                            const unsigned char* s = (GATEUP && sub) ? S1 : S0;
                            const void* g_s =
                                s + static_cast<unsigned long long>(ks * 2u + t) * N + cta_n + lb_lane * 16u;
                            void* const dst_s = &sS[sl][s_mtx][2u * t + 1u][lb_lane * 16u];
                            if (b_evict_first) {
                                v28_twarp_cp16_ef(dst_s, g_s);
                            } else {
                                v10_cp16(dst_s, g_s, true);
                            }
                        } else if (lb_lane < CH + 4u) {
                            const unsigned c = lb_lane - CH;
                            const unsigned idx = ks * TS4E_B_PER_KS + lb_wid;
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
                        const void* g_s =
                            s + static_cast<unsigned long long>(ks * S_ROWS_LD + r2) * N + cta_n + c2 * 16u;
                        if (b_evict_first) {
                            v28_twarp_cp16_ef(&sS[sl][m2][r2][c2 * 16u], g_s);
                        } else {
                            v10_cp16(&sS[sl][m2][r2][c2 * 16u], g_s, true);
                        }
                    }
                }
            };
            auto decode_ts4e_stage = [&](unsigned ks_dec) {
                unsigned d_tid;
                asm volatile("mov.u32 %0, %%tid.x;" : "=r"(d_tid));
                const unsigned d_wid = d_tid >> 5;
                if (d_wid >= TS4E_B_PER_KS) return;
                const unsigned d_lane = d_tid & 31u;
                __syncwarp();
                const unsigned sl = ks_dec % V14_DB;
                const unsigned sub = d_wid >> 1;
                const unsigned t = d_wid & 1u;
                const unsigned mtx = GATEUP ? sub : 0u;
                constexpr unsigned LANES_PER_SWAR = BCOLS / 4u; // 16 (GU) or 32 (DN)
                // Parallel carry-free SWAR decode across Warps 0..TS4E_B_PER_KS-1 (1 warp per disjoint
                // bucket, 0 serial loops, 0 extra __syncthreads()); overlap sExc load with SWAR ALU.
                const unsigned idx = ks_dec * TS4E_B_PER_KS + d_wid;
                const unsigned cnt = static_cast<unsigned>(sBktCnt[idx]);
                const unsigned beg = sBktBeg[idx];
                const unsigned pad = beg & 7u;
                const bool fast_exc = (cnt != 0u) && (pad + cnt <= 32u);
                const unsigned e_fast = (fast_exc && d_lane < cnt)
                    ? static_cast<unsigned>(sExc[sl][d_wid][pad + d_lane])
                    : 0u;
                if (d_lane < LANES_PER_SWAR) {
                    const unsigned base_off = (GATEUP ? (sub << 6) : 0u) + (d_lane << 2);
                    const unsigned lc4 = d_lane << 2;
                    const unsigned base_w = *reinterpret_cast<const unsigned*>(&sBase[base_off]);
                    const unsigned nib = *reinterpret_cast<const unsigned*>(&sS[sl][mtx][2u * t + 1u][lc4]);

                    const unsigned c0 = nib & 0x0F0F0F0Fu;
                    const unsigned e0 = (c0 + 0x01010101u) & 0x10101010u;
                    *reinterpret_cast<unsigned*>(&sS[sl][mtx][2u * t][lc4]) = base_w + (c0 & ~(e0 - (e0 >> 4)));

                    const unsigned c1 = (nib >> 4) & 0x0F0F0F0Fu;
                    const unsigned e1 = (c1 + 0x01010101u) & 0x10101010u;
                    *reinterpret_cast<unsigned*>(&sS[sl][mtx][2u * t + 1u][lc4]) = base_w + (c1 & ~(e1 - (e1 >> 4)));
                }
                __syncwarp();
                if (cnt != 0u) {
                    if (fast_exc) {
                        if (d_lane < cnt) {
                            const unsigned col128 = e_fast & 127u;
                            if constexpr (GATEUP) {
                                if ((col128 & 64u) == col_half) {
                                    sS[sl][mtx][2u * t + ((e_fast >> 7) & 1u)][col128 & 63u] =
                                        static_cast<unsigned char>(e_fast >> 8);
                                }
                            } else {
                                sS[sl][0u][2u * t + ((e_fast >> 7) & 1u)][col128] =
                                    static_cast<unsigned char>(e_fast >> 8);
                            }
                        }
                    } else {
                        const unsigned char* blob = (GATEUP && sub) ? S1 : S0;
                        const unsigned short* exc =
                            reinterpret_cast<const unsigned short*>(blob + TS4E_EXC_OFF);
                        for (unsigned i = d_lane; i < cnt; i += 32u) {
                            const unsigned e = static_cast<unsigned>(exc[beg + i]);
                            const unsigned col128 = e & 127u;
                            if constexpr (GATEUP) {
                                if ((col128 & 64u) == col_half) {
                                    sS[sl][mtx][2u * t + ((e >> 7) & 1u)][col128 & 63u] =
                                        static_cast<unsigned char>(e >> 8);
                                }
                            } else {
                                sS[sl][0u][2u * t + ((e >> 7) & 1u)][col128] =
                                    static_cast<unsigned char>(e >> 8);
                            }
                        }
                    }
                }
            };
            // 4-thread-cooperative conflict-free SWAR transcode: 2x LDS.32 per half-k32 (4x LDS.32/k64
            // vs 32x LDS.U8/k64) + 2-stage butterfly SHFL.BFLY + prmt.b32 4x4 byte transpose.
            auto transcode_16 = [&](unsigned sl, unsigned tb, unsigned t_h_override, unsigned t_blk) {
                constexpr unsigned SB_SWZ_SHIFT = GATEUP ? 4u : 5u;
                unsigned tc_tid;
                asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tc_tid));
                const unsigned t_col = tc_tid & 127u;
                const unsigned t_h = (t_h_override == 2u) ? (tc_tid >> 7) : t_h_override;
                const unsigned t_q = tc_tid & 3u;
                const unsigned t_mtx = GATEUP ? (t_col >> 6) : 0u;
                const unsigned t_lc = GATEUP ? (t_col & 63u) : t_col;
                const unsigned ss_base =
                    sbase + L::OFF_S + ((sl * BMAT + t_mtx) * S_ROWS + 2u * t_blk) * BCOLS + t_lc;
                const unsigned s0 = v28_ld_su8_vol(ss_base);
                const unsigned s1 = v28_ld_su8_off_vol<BCOLS>(ss_base);
                const unsigned s_own = (t_h == 0u) ? s0 : s1;
                const unsigned s_max = max(s0 & 0x7fu, s1 & 0x7fu);
                const int e32 = (s_max >= 8u) ? (static_cast<int>(s_max >> 3) - 7)
                              : (s_max != 0u) ? (22 - __clz(static_cast<int>(s_max)))
                                              : 0;
                const unsigned r = v28_norm_e4m3_scale(s_own, e32);
                const uint2 tt = v28_ld_lut_vol(sbase + L::OFF_LUT, r);
                const unsigned c4_swz = (t_lc & ~3u) ^ (t_q << SB_SWZ_SHIFT);
                const unsigned r0 = 16u * t_blk + 8u * t_h + t_q;
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
                const unsigned sbt_addr =
                    sbase + L::OFF_BT + tb * (V28_M32_BC * 64u) + v14_swz(t_col, 2u * t_blk + t_h);
                v28_st_sbt_v2_0(sbt_addr, w0, w1);
                const unsigned p1 = v10_prmt(u1, __shfl_xor_sync(0xffffffffu, u1, 1), sel1);
                const unsigned c1 = v10_prmt(p1, __shfl_xor_sync(0xffffffffu, p1, 2), sel2);
                const unsigned w2 = v10_e2m1x4_to_e4m3x4(c1, tt.x, tt.y);
                const unsigned w3 = v10_e2m1x4_to_e4m3x4(c1 >> 16, tt.x, tt.y);
                v28_st_sbt_v2_8(sbt_addr, w2, w3);
                if (t_h == 0u) {
                    const unsigned sf_addr =
                        sbase + L::OFF_SF + (tb * 2u + t_blk) * V28_M32_BC + v28_m32_sf_index<GATEUP>(t_col);
                    v28_st_su8_vol(sf_addr, static_cast<unsigned>(e32 + 127) & 0xffu);
                }
            };

            float acc[4][4];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[j][q] = 0.0f;

            // prologue: Gp0 = {B0 (+ sBase/sExc0)}, Gp1 = {A0, B1 (+ sExc1)}, Gp2 = {A1, B2 (+ sExc2)}
            load_b(0u);
            v10_commit();
            load_a(0u);
            if (KS > 1u) load_b(1u);
            v10_commit();
            if (KS > 1u) load_a(1u);
            if (KS > 2u) load_b(2u);
            v10_commit();
            v10_wait<2>();
            if constexpr (TS4E) {
                if (m_sub == 0) __syncthreads();
                decode_ts4e_stage(0u);
            }
            __syncthreads();
            transcode_16(0u, 0u, 2u, 0u);
            transcode_16(0u, 0u, 2u, 1u);

            unsigned sfa = 0u;
            auto step = [&](auto par_tag, unsigned ks) {
                constexpr unsigned PAR = decltype(par_tag)::value;
                const bool do_tc = (PAR == 0u) ? true : (ks + 1u < KS);
                v10_wait<1>();
                if constexpr (TS4E) {
                    if (do_tc) decode_ts4e_stage(ks + 1u);
                }
                __syncthreads();
                if (ks + 2u < KS) load_a(ks + 2u);
                if (ks + 3u < KS) load_b(ks + 3u);
                v10_commit();
                const unsigned next_sl = (ks + 1u) % V14_DB;
                if (!warp_active) {
                    // When valid_rows <= 16, Warps 4..7 (tid >= 128) overlap both transcode halves
                    // while Warps 0..3 execute QMMAs immediately.
                    if (do_tc) {
                        #pragma unroll 1
                        for (unsigned th_i = 0u; th_i < 2u; ++th_i) {
                            transcode_16(next_sl, PAR ^ 1u, th_i, 0u);
                            transcode_16(next_sl, PAR ^ 1u, th_i, 1u);
                        }
                    }
                    return;
                }
                if constexpr (PAR == 0u) {
                    sfa = sAS[(ks >> 1) & 1u][warp_m + gid + (tig & 1u) * 8u];
                }
                const unsigned a_slot = ks % V14_DA;
                const unsigned row_a = warp_m + (lane & 15u);
                const unsigned row_b0 = warp_n + ((lane >> 4) << 3) + (lane & 7u);
                constexpr unsigned b_off1 = GATEUP ? (BCOLS * 64u) : 1024u;
                const bool run_tc = do_tc && (valid_rows > 16);
                #pragma unroll
                for (unsigned t = 0u; t < 2u; ++t) {
                    unsigned afr[4], bfr[4][2];
                    const unsigned addr_a = static_cast<unsigned>(__cvta_generic_to_shared(
                        sA + a_slot * (V28_M32_TILE_M * 64u) + v14_swz(row_a, 2u * t + (lane >> 4))));
                    const unsigned addr_b0 = static_cast<unsigned>(__cvta_generic_to_shared(
                        sBt + PAR * (V28_M32_BC * 64u) + v14_swz(row_b0, 2u * t + ((lane >> 3) & 1u))));
                    const unsigned sfb = *reinterpret_cast<const unsigned*>(&sSF[PAR][t][sf_off]);
                    v10_ldsm4(addr_a, afr[0], afr[1], afr[2], afr[3]);
                    v10_ldsm4(addr_b0, bfr[0][0], bfr[0][1], bfr[1][0], bfr[1][1]);
                    v10_ldsm4(addr_b0 + b_off1, bfr[2][0], bfr[2][1], bfr[3][0], bfr[3][1]);
                    if (t == 0u) {
                        v28_qmma<2 * PAR, 0>(acc[0], afr, bfr[0][0], bfr[0][1], sfa, sfb);
                        v28_qmma<2 * PAR, 1>(acc[1], afr, bfr[1][0], bfr[1][1], sfa, sfb);
                        v28_qmma<2 * PAR, 2>(acc[2], afr, bfr[2][0], bfr[2][1], sfa, sfb);
                        v28_qmma<2 * PAR, 3>(acc[3], afr, bfr[3][0], bfr[3][1], sfa, sfb);
                    } else {
                        v28_qmma<2 * PAR + 1, 0>(acc[0], afr, bfr[0][0], bfr[0][1], sfa, sfb);
                        v28_qmma<2 * PAR + 1, 1>(acc[1], afr, bfr[1][0], bfr[1][1], sfa, sfb);
                        v28_qmma<2 * PAR + 1, 2>(acc[2], afr, bfr[2][0], bfr[2][1], sfa, sfb);
                        v28_qmma<2 * PAR + 1, 3>(acc[3], afr, bfr[3][0], bfr[3][1], sfa, sfb);
                    }
                    if (run_tc) {
                        transcode_16(next_sl, PAR ^ 1u, 2u, t);
                    }
                }
            };
            #pragma unroll 1
            for (unsigned ks = 0u; ks < KS; ks += 2u) {
                step(std::integral_constant<unsigned, 0u>{}, ks);
                step(std::integral_constant<unsigned, 1u>{}, ks + 1u);
            }
            v10_wait<0>();

            const float s2a = s20[item.expert];
            const float s2b = GATEUP ? s21[item.expert] : 0.0f;
            if constexpr (GATEUP) {
                // Register-direct GATEUP MXFP8 epilogue (0 sE staging, 0 __syncthreads):
                // Each warp owns 16 cols (half an MX32 block), exchanges per-16-col absmax with
                // its partner warp (wn ^ 1) via dedicated sMax[32][4] + 64-thread partner barrier,
                // quantizes BF16-rounded SwiGLU outputs in registers (0-ULP), and emits v28_st8_evict_last.
                if (warp_active) {
                    const unsigned lr0 = warp_m + gid, lr1 = lr0 + 8u;
                    float f0[4], f1[4];
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        float v[4];
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const float g = acc[j][q] * s2a;
                            const float u = acc[j + 2][q] * s2b;
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
                    if (pair_id == 0u)      asm volatile("bar.sync 2, 64;\n" ::: "memory");
                    else if (pair_id == 1u) asm volatile("bar.sync 3, 64;\n" ::: "memory");
                    else if (pair_id == 2u) asm volatile("bar.sync 4, 64;\n" ::: "memory");
                    else                    asm volatile("bar.sync 5, 64;\n" ::: "memory");
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
                    // 3-shuffle register transpose across tig=0..3:
                    // tig=0 -> (lr0, cols 0..7), tig=1 -> (lr1, cols 0..7),
                    // tig=2 -> (lr0, cols 8..15), tig=3 -> (lr1, cols 8..15).
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
                        const unsigned long long orow = cta_m + out_row;
                        v28_st8_evict_last(oq + orow * N + cta_n + warp_n + (odd2 ? 8u : 0u), make_uint2(out_x, out_y));
                        if ((wn & 1u) == 0u && !odd2) {
                            os[orow * (N / 32u) + cta_n / 32u + (wn >> 1)] =
                                static_cast<unsigned char>(odd1 ? ss1 : ss0);
                        }
                    }
                }
            } else {
                // Register-direct 4x4 quad transpose (0 SMEM, 0 __syncthreads): each warp owns 32 cols = 1 MX32 block.
                (void)sMax;
                if (warp_active) {
                    const unsigned lr0 = warp_m + gid, lr1 = lr0 + 8u;
                    unsigned u0[4], u1[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        u0[j] = v28_pack_bf162(acc[j][0] * s2a, acc[j][1] * s2a);
                        u1[j] = v28_pack_bf162(acc[j][2] * s2a, acc[j][3] * s2a);
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
                            *reinterpret_cast<uint4*>(C + static_cast<unsigned long long>(cta_m + lr0) * N + col8) = w0;
                        }
                        const uint4 w1 = v28_bf16_reg_quad(u1[0], u1[1], u1[2], u1[3], tig);
                        if (static_cast<int>(lr1) < valid_rows) {
                            *reinterpret_cast<uint4*>(C + static_cast<unsigned long long>(cta_m + lr1) * N + col8) = w1;
                        }
                    }
                }
            }
            if (m_sub + static_cast<int>(V28_M32_TILE_M) < item_rows) __syncthreads();
        }
        if (task + gridDim.x < task_count) __syncthreads();
    }
}

} // namespace

// 2-CTA/SM M32 BC=128 Flat-scale entry points (39,680 B GATEUP / 39,168 B DOWN dynamic SMEM, 256 threads).
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_gateup_k32_v28_m32(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v28_m32_impl<true, 0, false>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_k32_v28_m32(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m32_impl<false, 0, false>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_k32_v28q_m32(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m32_impl<false, 3, false>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
        slot_q, slot_s);
}

// 2-CTA/SM M32 BC=128 Atlas-TS4E-128 entry points (41,536 B GATEUP / 39,808 B DOWN SMEM, 256 threads).
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_gateup_k32_v28_m32_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v28_m32_impl<true, 0, true>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_k32_v28_m32_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m32_impl<false, 0, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_k32_v28q_m32_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m32_impl<false, 3, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
        slot_q, slot_s);
}

