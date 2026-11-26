// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// V28 M16 4-CTA/SM decode/verify W4A8 MoE kernels (TILE_M=16, BC=128, flat and TS4E-128).

// =========================================================================
// Wave-21 Task 1: 4-CTA/SM Decode V28 M16 Kernel (TILE_M = 16, BC = 128, 128 threads = 4 warps)
// Register-direct B transcode (0 B sBt, 0 B sSF) with full DA = DB = 3 (2 K64 stages in flight),
// bank-conflict-free sB XOR-swizzle, SMEM lut[256] (0 LDC), 4-warp/2-warp parallel TS4E decode
// (0 extra __syncthreads() in K-loop), and register-direct MXFP8/BF16 epilogues.
// Dynamic SMEM <= 21,184 B (20.69 KiB <= 24 KiB) -> 4 resident CTAs/SM (192 CTAs across 48 SMs)!
// Strictly 0-ULP bit-identical to v28_m32 / v28_m32_ts4e / v28_twarp.
// =========================================================================
namespace {

constexpr unsigned V28_M16_TILE_M = 16u;
constexpr unsigned V28_M16_BC = 128u;
constexpr unsigned V28_M16_THREADS = 128u;

template <bool GATEUP, bool TS4E, typename Shape = Qwen4FlashNextShape> struct V28M16Smem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V28_M16_BC / BMAT; // 64 (GATEUP) or 128 (DOWN)
    static constexpr unsigned S_ROWS = 4u;
    static constexpr unsigned TS4E_KS = Shape::template gemm_k<GATEUP>() / 64u;
    static constexpr unsigned TS4E_B_PER_KS = GATEUP ? 4u : 2u;
    static constexpr unsigned TS4E_BKTS = TS4E_KS * TS4E_B_PER_KS;
    static constexpr unsigned A = V14_DA * V28_M16_TILE_M * 64u;              //  3,072 B
    static constexpr unsigned B = V14_DB * BMAT * 32u * BCOLS;                // 12,288 B
    static constexpr unsigned S = V14_DB * BMAT * S_ROWS * BCOLS;             //  1,536 B
    static constexpr unsigned AS = 2u * V28_M16_TILE_M * 4u;                  //    128 B
    static constexpr unsigned LUT = 256u * 8u;                                //  2,048 B
    static constexpr unsigned MAX = GATEUP ? (V28_M16_TILE_M * 4u * 4u) : 0u; //    256 B (GU) / 0 B (DN)
    static constexpr unsigned OFF_B = A;
    static constexpr unsigned OFF_S = OFF_B + B;
    static constexpr unsigned OFF_AS = OFF_S + S;
    static constexpr unsigned OFF_LUT = OFF_AS + AS;
    static constexpr unsigned OFF_MAX = OFF_LUT + LUT;
    static constexpr unsigned OFF_BASE = OFF_MAX + MAX;
    static constexpr unsigned OFF_EXC = OFF_BASE + (TS4E ? 128u : 0u);
    static constexpr unsigned OFF_BKT_BEG = OFF_EXC + (TS4E ? (V14_DB * TS4E_B_PER_KS * 32u * 2u) : 0u);
    static constexpr unsigned OFF_BKT_CNT = OFF_BKT_BEG + (TS4E ? (TS4E_BKTS * 4u) : 0u);
    static constexpr unsigned RAW_BYTES = OFF_BKT_CNT + (TS4E ? (TS4E_BKTS * 2u) : 0u);
    static constexpr unsigned BYTES = (RAW_BYTES + 15u) & ~15u;
    static_assert(
        !TS4E ? (GATEUP ? (BYTES == 19328u) : (BYTES == 19072u))
              : (GATEUP ? (BYTES == 21184u) : (BYTES == 19712u)),
        "V28M16Smem byte contract"
    );
    static_assert(BYTES <= 24576u, "V28M16Smem fits 4 CTAs/SM (<= 24 KiB)");
};

template <bool GATEUP, int EPI = 0, bool TS4E = false, typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void v28_m16_impl(
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
    using L = V28M16Smem<GATEUP, TS4E, Shape>;
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
    if (gridDim.x == 0 || gridDim.x > oi640::FIXED_GRID || blockDim.x != V28_M16_THREADS || dyn_smem < L::BYTES) {
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
    unsigned char* const sA = smem;                                                        // [3][16][64] swz
    unsigned char (*const sB)[BMAT][32][BCOLS] = reinterpret_cast<unsigned char (*)[BMAT][32][BCOLS]>(smem + L::OFF_B);
    unsigned char (*const sS)[BMAT][S_ROWS][BCOLS] = reinterpret_cast<unsigned char (*)[BMAT][S_ROWS][BCOLS]>(smem + L::OFF_S);
    unsigned (*const sAS)[V28_M16_TILE_M] = reinterpret_cast<unsigned (*)[V28_M16_TILE_M]>(smem + L::OFF_AS);
    uint2* const lut = reinterpret_cast<uint2*>(smem + L::OFF_LUT);
    float (*const sMax)[4] = reinterpret_cast<float (*)[4]>(smem + L::OFF_MAX);
    unsigned char* const sBase = smem + L::OFF_BASE;
    unsigned short (*const sExc)[TS4E_B_PER_KS][32] = reinterpret_cast<unsigned short (*)[TS4E_B_PER_KS][32]>(smem + L::OFF_EXC);
    unsigned* const sBktBeg = reinterpret_cast<unsigned*>(smem + L::OFF_BKT_BEG);
    unsigned short* const sBktCnt = reinterpret_cast<unsigned short*>(smem + L::OFF_BKT_CNT);

    const unsigned tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31u;
    lut[tid] = V28_E4M3_LUT[tid];
    lut[tid + V28_M16_THREADS] = V28_E4M3_LUT[tid + V28_M16_THREADS];
    const unsigned gid = lane >> 2, tig = lane & 3u;
    const unsigned wn = warp_id; // 0..3 across N
    const unsigned warp_n = wn * (GATEUP ? 16u : 32u);
    constexpr unsigned KS = K / 64u;
    constexpr unsigned kblocks = K / 32u;
    static_assert((KS & 1u) == 0u && KS >= 2u, "v28_m16 requires even KS >= 2");
    const bool keep_slot =
        (static_cast<unsigned long long>(workspace->expanded) * Shape::SLOT_MXFP8_BYTES_ULL) <= V28_KEEP_SLOT_BYTES;

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const oi640::WorkItem item = workspace->items[task / n_tiles];
        const unsigned cta_n = (task % n_tiles) * TN;
        if (item.expert >= Shape::EXPERTS) { if (tid == 0u) oi640::fail(status, oi640::ERR_PLAN); return; }
        const int m_start = expert_offsets[item.expert];
        const int M_expert = expert_offsets[item.expert + 1] - m_start;
        const int local = static_cast<int>(item.m_tile * V10_TILE_M);
        const unsigned char* B0 = reinterpret_cast<const unsigned char*>(pk0[item.expert]);
        const unsigned char* S0 = reinterpret_cast<const unsigned char*>(sc0[item.expert]);
        const unsigned char* B1 = GATEUP ? reinterpret_cast<const unsigned char*>(pk1[item.expert]) : B0;
        const unsigned char* S1 = GATEUP ? reinterpret_cast<const unsigned char*>(sc1[item.expert]) : S0;
        if (m_start < 0 || M_expert < 0 || local >= M_expert || B0 == nullptr || S0 == nullptr ||
            B1 == nullptr || S1 == nullptr || !isfinite(s20[item.expert]) || (GATEUP && !isfinite(s21[item.expert]))) {
            if (tid == 0u) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const int item_rows = min(M_expert - local, static_cast<int>(V10_TILE_M));
        const unsigned col_half = cta_n & 64u;

        if constexpr (TS4E) {
            static_assert(KS >= 2u, "TS4E prologue resolves ks=0..1 inline on warps 0..3");
            constexpr unsigned PROLOGUE_BKTS = 2u * TS4E_B_PER_KS;
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
                if (lane < 2u) {
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
            }
            for (unsigned idx = PROLOGUE_BKTS + tid; idx < KS * TS4E_B_PER_KS; idx += V28_M16_THREADS) {
                resolve_ts4e_bkt(idx);
            }
        }

        // Loop in 16-row slices over the planner's V10_TILE_M=128 item (1 iteration for >98% of
        // experts at C=2..16; exact coverage for hot experts with 17..128 tokens).
        for (int m_sub = 0; m_sub < item_rows; m_sub += static_cast<int>(V28_M16_TILE_M)) {
            const unsigned cta_m = static_cast<unsigned>(m_start + local + m_sub);
            const int valid_rows = min(item_rows - m_sub, static_cast<int>(V28_M16_TILE_M));
            const bool b_evict_first = (M_expert <= static_cast<int>(V28_M16_TILE_M));

            const unsigned a_row = tid >> 2, a_ch = tid & 3u;
            const bool a_need = (tid < 64u);
            const bool a_ok = a_need && (static_cast<int>(a_row) < valid_rows);
            const unsigned a_src_row =
                a_ok ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + a_row]) : cta_m + a_row) : 0u;
            const unsigned char* a_src = aq + static_cast<unsigned long long>(a_src_row) * K + a_ch * 16u;
            const unsigned a_dst = v14_swz(a_row, a_ch);

            const unsigned sc_row = tid - 64u;
            const bool sc_need = (tid >= 64u && tid < 80u);
            const bool sc_ok = sc_need && (static_cast<int>(sc_row) < valid_rows);
            const unsigned sc_src_row =
                sc_ok ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + sc_row]) : cta_m + sc_row) : 0u;
            const unsigned char* sc_src = as + static_cast<unsigned long long>(sc_src_row) * kblocks;

            auto load_a = [&](unsigned ks) {
                if (a_need) {
                    v10_cp16(sA + (ks % V14_DA) * (V28_M16_TILE_M * 64u) + a_dst, a_src + ks * 64u, a_ok);
                }
                if (sc_need && (ks & 1u) == 0u) {
                    v10_cp4(&sAS[(ks >> 1) & 1u][sc_row], sc_src + ks * 2u, sc_ok);
                }
            };
            auto load_b = [&](unsigned ks) {
                constexpr unsigned CH = BCOLS / 16u; // 4 (GU) or 8 (DN)
                const unsigned sl = ks % V14_DB;
                #pragma unroll
                for (unsigned pass = 0u; pass < 2u; ++pass) {
                    const unsigned idx = tid + pass * V28_M16_THREADS;
                    const unsigned mtx = idx / (32u * CH);
                    const unsigned r = (idx / CH) % 32u;
                    const unsigned cc = idx % CH;
                    // Conflict-free 16B chunk XOR-swizzle on sB: chunk ^= ((r >> 1) & 3)
                    // guarantees 0 bank conflicts for direct lane (gid, tig) LDS.U8 reads.
                    const unsigned b_dst_col = (cc << 4) ^ (((r >> 1) & 3u) << 4);
                    const unsigned char* src = (GATEUP && mtx) ? B1 : B0;
                    const void* g_b = src + static_cast<unsigned long long>(ks * 32u + r) * N + cta_n + cc * 16u;
                    if (b_evict_first) {
                        v28_twarp_cp16_ef(&sB[sl][mtx][r][b_dst_col], g_b);
                    } else {
                        v10_cp16(&sB[sl][mtx][r][b_dst_col], g_b, true);
                    }
                }
                if constexpr (TS4E) {
                    if (warp_id < TS4E_B_PER_KS) {
                        const unsigned sub = warp_id >> 1;
                        const unsigned t = warp_id & 1u;
                        const unsigned mtx = GATEUP ? sub : 0u;
                        if (lane < CH) {
                            const unsigned char* s = (GATEUP && sub) ? S1 : S0;
                            const void* g_s = s + static_cast<unsigned long long>(ks * 2u + t) * N + cta_n + lane * 16u;
                            void* const dst_s = &sS[sl][mtx][2u * t + 1u][lane * 16u];
                            if (b_evict_first) {
                                v28_twarp_cp16_ef(dst_s, g_s);
                            } else {
                                v10_cp16(dst_s, g_s, true);
                            }
                        } else if (lane < CH + 4u) {
                            const unsigned c = lane - CH;
                            const unsigned idx = ks * TS4E_B_PER_KS + warp_id;
                            const unsigned cnt = static_cast<unsigned>(sBktCnt[idx]);
                            if (cnt > 0u) {
                                const unsigned beg = sBktBeg[idx];
                                const unsigned pad = beg & 7u;
                                if (pad + cnt <= 32u && (c << 3) < (pad + cnt)) {
                                    const unsigned char* blob = (GATEUP && sub) ? S1 : S0;
                                    const unsigned short* exc =
                                        reinterpret_cast<const unsigned short*>(blob + TS4E_EXC_OFF);
                                    v10_cp16(&sExc[sl][warp_id][c << 3], exc + (beg & ~7u) + (c << 3), true);
                                }
                            }
                        }
                    }
                } else {
                    if (tid < BMAT * 4u * CH) {
                        const unsigned m2 = tid / (4u * CH), r2 = (tid / CH) % 4u, c2 = tid % CH;
                        const unsigned char* s = (GATEUP && m2) ? S1 : S0;
                        const void* g_s = s + static_cast<unsigned long long>(ks * 4u + r2) * N + cta_n + c2 * 16u;
                        if (b_evict_first) {
                            v28_twarp_cp16_ef(&sS[sl][m2][r2][c2 * 16u], g_s);
                        } else {
                            v10_cp16(&sS[sl][m2][r2][c2 * 16u], g_s, true);
                        }
                    }
                }
            };
            auto decode_ts4e_stage = [&](unsigned ks_dec) {
                if (warp_id >= TS4E_B_PER_KS) return;
                __syncwarp();
                const unsigned sl = ks_dec % V14_DB;
                const unsigned sub = warp_id >> 1;
                const unsigned t = warp_id & 1u;
                const unsigned mtx = GATEUP ? sub : 0u;
                constexpr unsigned LANES_PER_SWAR = BCOLS / 4u; // 16 (GU) or 32 (DN)
                const unsigned idx = ks_dec * TS4E_B_PER_KS + warp_id;
                const unsigned cnt = static_cast<unsigned>(sBktCnt[idx]);
                const unsigned beg = sBktBeg[idx];
                const unsigned pad = beg & 7u;
                const bool fast_exc = (cnt != 0u) && (pad + cnt <= 32u);
                const unsigned e_fast = (fast_exc && lane < cnt)
                    ? static_cast<unsigned>(sExc[sl][warp_id][pad + lane])
                    : 0u;
                if (lane < LANES_PER_SWAR) {
                    const unsigned base_off = (GATEUP ? (sub << 6) : 0u) + (lane << 2);
                    const unsigned lc4 = lane << 2;
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
                        if (lane < cnt) {
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
                        for (unsigned i = lane; i < cnt; i += 32u) {
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

            float acc[4][4];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[j][q] = 0.0f;

            // Prologue: 2 full K64 stages in flight on a 3-stage ring (DA = DB = 3).
            // Each step issues ks+2 immediately after the single __syncthreads(), keeping
            // BOTH ks+1 and ks+2 in flight during compute with only 1 barrier per K64 step!
            load_a(0u);
            load_b(0u);
            v10_commit();
            load_a(1u);
            load_b(1u);
            v10_commit();

            unsigned sfa = 0u;
            const unsigned row_a = lane & 15u;
            const unsigned qbase = gid << 2;
            const unsigned my_mtx = GATEUP ? (tig >> 1) : 0u;
            const unsigned my_lc = warp_n + (GATEUP ? ((tig & 1u) << 3) : (tig << 3)) + gid;

            auto step = [&](auto par_tag, unsigned ks) {
                constexpr unsigned PAR = decltype(par_tag)::value;
                v10_wait<1>();
                if constexpr (TS4E) {
                    if constexpr (PAR == 0u) {
                        if (m_sub == 0 && ks == 0u) __syncthreads();
                    }
                    decode_ts4e_stage(ks);
                }
                __syncthreads();
                if (ks + 2u < KS) {
                    load_a(ks + 2u);
                    load_b(ks + 2u);
                }
                v10_commit();

                if constexpr (PAR == 0u) {
                    sfa = sAS[(ks >> 1) & 1u][gid + (tig & 1u) * 8u];
                }
                const unsigned slot = ks % V14_DA;

                auto run_k32 = [&](auto t_tag) {
                    constexpr unsigned T = decltype(t_tag)::value;
                    unsigned afr[4];
                    const unsigned addr_a = static_cast<unsigned>(__cvta_generic_to_shared(
                        sA + slot * (V28_M16_TILE_M * 64u) + v14_swz(row_a, 2u * T + (lane >> 4))));
                    v10_ldsm4(addr_a, afr[0], afr[1], afr[2], afr[3]);

                    // Cooperative quad scale + LUT: lane tig handles j = tig (0 redundant scale math).
                    const unsigned s0 = sS[slot][my_mtx][2u * T][my_lc];
                    const unsigned s1 = sS[slot][my_mtx][2u * T + 1u][my_lc];
                    const unsigned s_max = max(s0 & 0x7fu, s1 & 0x7fu);
                    const int e32 = (s_max >= 8u) ? (static_cast<int>(s_max >> 3) - 7)
                                  : (s_max != 0u) ? (22 - __clz(static_cast<int>(s_max)))
                                                  : 0;
                    const unsigned sf_byte = static_cast<unsigned>(e32 + 127) & 0xffu;
                    const unsigned r0 = v28_norm_e4m3_scale(s0, e32);
                    const unsigned r1 = v28_norm_e4m3_scale(s1, e32);
                    const uint2 tt0 = lut[r0];
                    const uint2 tt1 = lut[r1];

                    const unsigned sf_p1 = __shfl_xor_sync(0xffffffffu, sf_byte, 1);
                    const unsigned sf_pair = (tig & 1u) ? ((sf_byte << 8) | sf_p1) : ((sf_p1 << 8) | sf_byte);
                    const unsigned sf_p2 = __shfl_xor_sync(0xffffffffu, sf_pair, 2);
                    const unsigned sfb = (tig & 2u) ? ((sf_pair << 16) | sf_p2) : ((sf_p2 << 16) | sf_pair);

                    const unsigned rb_base = 16u * T + ((gid & 2u) << 2) + 2u * tig + (gid & 1u);
                    const unsigned sel1 = (gid & 1u) ? 0x3715u : 0x6240u;
                    const unsigned sel2 = (gid & 2u) ? 0x3276u : 0x5410u;
                    auto do_j = [&](auto j_tag) {
                        constexpr int J = decltype(j_tag)::value;
                        const unsigned t0x = __shfl_sync(0xffffffffu, tt0.x, qbase + static_cast<unsigned>(J));
                        const unsigned t0y = __shfl_sync(0xffffffffu, tt0.y, qbase + static_cast<unsigned>(J));
                        const unsigned t1x = __shfl_sync(0xffffffffu, tt1.x, qbase + static_cast<unsigned>(J));
                        const unsigned t1y = __shfl_sync(0xffffffffu, tt1.y, qbase + static_cast<unsigned>(J));
                        const unsigned j_mtx = GATEUP ? (static_cast<unsigned>(J) >> 1) : 0u;
                        const unsigned j_col4 = warp_n + (GATEUP ? ((static_cast<unsigned>(J) & 1u) << 3)
                                                                 : (static_cast<unsigned>(J) << 3)) + (gid & ~3u);
                        const unsigned col4_swz = j_col4 ^ (tig << 4);
                        // 4-thread-cooperative SWAR LDS.32 + 4x4 byte transpose across gid&3 (lanes ^4, ^8):
                        // 1 LDS.32 per (T, J) instead of 4 LDS.U8, yielding x0 in low 16b and x1 in high 16b.
                        const unsigned u = *reinterpret_cast<const unsigned*>(&sB[slot][j_mtx][rb_base][col4_swz]);
                        const unsigned p = v10_prmt(u, __shfl_xor_sync(0xffffffffu, u, 4), sel1);
                        const unsigned w = v10_prmt(p, __shfl_xor_sync(0xffffffffu, p, 8), sel2);
                        const unsigned b0 = v10_e2m1x4_to_e4m3x4(w, t0x, t0y);
                        const unsigned b1 = v10_e2m1x4_to_e4m3x4(w >> 16, t1x, t1y);
                        v28_qmma<2 * PAR + T, J>(acc[J], afr, b0, b1, sfa, sfb);
                    };
                    do_j(std::integral_constant<int, 0>{});
                    do_j(std::integral_constant<int, 1>{});
                    do_j(std::integral_constant<int, 2>{});
                    do_j(std::integral_constant<int, 3>{});
                };
                run_k32(std::integral_constant<unsigned, 0u>{});
                run_k32(std::integral_constant<unsigned, 1u>{});
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
                const unsigned lr0 = gid, lr1 = gid + 8u;
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
                if ((warp_id >> 1) == 0u) asm volatile("bar.sync 2, 64;\n" ::: "memory");
                else                      asm volatile("bar.sync 3, 64;\n" ::: "memory");
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
                    const unsigned long long orow = cta_m + out_row;
                    v28_st8_evict_last(oq + orow * N + cta_n + warp_n + (odd2 ? 8u : 0u), make_uint2(out_x, out_y));
                    if ((wn & 1u) == 0u && !odd2) {
                        os[orow * (N / 32u) + cta_n / 32u + (wn >> 1)] =
                            static_cast<unsigned char>(odd1 ? ss1 : ss0);
                    }
                }
            } else {
                (void)sMax;
                const unsigned lr0 = gid, lr1 = gid + 8u;
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
            if (m_sub + static_cast<int>(V28_M16_TILE_M) < item_rows) __syncthreads();
        }
        if (task + gridDim.x < task_count) __syncthreads();
    }
}

} // namespace

// 4-CTA/SM M16 BC=128 Flat-scale entry points (19,328 B GATEUP / 19,072 B DOWN SMEM, 128 threads).
extern "C" __global__ __launch_bounds__(128, 4) void qwen4_moe_compact_t_gemm_gateup_k32_v28_m16(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v28_m16_impl<true, 0, false>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K);
}

extern "C" __global__ __launch_bounds__(128, 4) void qwen4_moe_compact_t_gemm_k32_v28_m16(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m16_impl<false, 0, false>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(128, 4) void qwen4_moe_compact_t_gemm_k32_v28q_m16(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m16_impl<false, 3, false>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
        slot_q, slot_s);
}

// 4-CTA/SM M16 BC=128 Atlas-TS4E-128 entry points (21,184 B GATEUP / 19,712 B DOWN SMEM, 128 threads).
extern "C" __global__ __launch_bounds__(128, 4) void qwen4_moe_compact_t_gemm_gateup_k32_v28_m16_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v28_m16_impl<true, 0, true>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K);
}

extern "C" __global__ __launch_bounds__(128, 4) void qwen4_moe_compact_t_gemm_k32_v28_m16_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m16_impl<false, 0, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(128, 4) void qwen4_moe_compact_t_gemm_k32_v28q_m16_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_m16_impl<false, 3, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
        slot_q, slot_s);
}

