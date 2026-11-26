// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// V30 2-CTA/SM prefill W4A8 MoE kernels (TILE_M=128, BC=128, flat and TS4E-128).

// =========================================================================
// Wave-21 Task 3: 2-CTA/SM Prefill V30 Kernel (TILE_M = 128, BC = 128, 256 threads = 8 warps, 4x2)
// Keeps TILE_M = 128 (0 weight re-read amplification, reuses existing V7/V8/V28 128-row worklist)
// while halving BC = 128 and using 2-stage rings (DA = DB = 2, BT = 2) so total SMEM <= 47,744 B
// <= 49,152 B (48 KiB) -> 2 resident CTAs/SM (16 warps/SM) hiding LPDDR5X miss latency!
// Features 4-warp/2-warp parallel TS4E decode (0 extra barriers in K-loop), interleaved transcode
// (and Warps 4..7 concurrent transcode for valid_rows <= 64), and register-direct epilogues.
// Strictly 0-ULP bit-identical to v28_twarp / v28_twarp_ts4e.
// =========================================================================
namespace {

constexpr unsigned V30_2CTA_TILE_M = 128u;
constexpr unsigned V30_2CTA_BC = 128u;
constexpr unsigned V30_2CTA_THREADS = 256u;
constexpr unsigned V30_2CTA_DA = 2u;
constexpr unsigned V30_2CTA_DB = 2u;

template <bool GATEUP, bool TS4E, typename Shape = Qwen4FlashNextShape> struct V30Smem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V30_2CTA_BC / BMAT; // 64 (GATEUP) or 128 (DOWN)
    static constexpr unsigned S_ROWS = 4u;
    static constexpr unsigned TS4E_KS = Shape::template gemm_k<GATEUP>() / 64u;
    static constexpr unsigned TS4E_B_PER_KS = GATEUP ? 4u : 2u;
    static constexpr unsigned TS4E_BKTS = TS4E_KS * TS4E_B_PER_KS;
    static constexpr unsigned A = V30_2CTA_DA * V30_2CTA_TILE_M * 64u;   // 16,384 B
    static constexpr unsigned B = V30_2CTA_DB * BMAT * 32u * BCOLS;      //  8,192 B
    static constexpr unsigned S = V30_2CTA_DB * BMAT * S_ROWS * BCOLS;   //  1,024 B
    static constexpr unsigned AS = 2u * V30_2CTA_TILE_M * 4u;            //  1,024 B
    static constexpr unsigned BT = 2u * V30_2CTA_BC * 64u;               // 16,384 B
    static constexpr unsigned SF = 2u * 2u * V30_2CTA_BC;                //    512 B
    static constexpr unsigned LUT = 256u * 8u;                           //  2,048 B
    static constexpr unsigned AOFF = GATEUP ? (V30_2CTA_TILE_M * 4u) : 0u; // 512 B (GU) / 0 B (DN)
    static constexpr unsigned PTRS = 8u * 8u;                            //     64 B
    static constexpr unsigned OFF_B = A;
    static constexpr unsigned OFF_S = OFF_B + B;
    static constexpr unsigned OFF_AS = OFF_S + S;
    static constexpr unsigned OFF_BT = OFF_AS + AS;
    static constexpr unsigned OFF_SF = OFF_BT + BT;
    static constexpr unsigned OFF_LUT = OFF_SF + SF;
    static constexpr unsigned OFF_AOFF = OFF_LUT + LUT;
    static constexpr unsigned OFF_PTRS = OFF_AOFF + AOFF;
    static constexpr unsigned OFF_BASE = OFF_PTRS + PTRS;
    static constexpr unsigned OFF_EXC = OFF_BASE + (TS4E ? 128u : 0u);
    static constexpr unsigned OFF_BKT_BEG = OFF_EXC + (TS4E ? (V30_2CTA_DB * TS4E_B_PER_KS * 32u * 2u) : 0u);
    static constexpr unsigned OFF_BKT_CNT = OFF_BKT_BEG + (TS4E ? (TS4E_BKTS * 4u) : 0u);
    static constexpr unsigned RAW_BYTES = OFF_BKT_CNT + (TS4E ? (TS4E_BKTS * 2u) : 0u);
    static constexpr unsigned BYTES = (RAW_BYTES + 15u) & ~15u;
    static_assert(
        !TS4E ? (GATEUP ? (BYTES == 46144u) : (BYTES == 45632u))
              : (GATEUP ? (BYTES == 47744u) : (BYTES == 46144u)),
        "V30Smem byte contract"
    );
    static_assert(BYTES <= 48128u, "V30Smem fits 2 CTAs/SM (<= 48,128 B <= 48 KiB)");
};

__device__ __forceinline__ void v30_cp16_el_s32(unsigned s_addr, const void* gmem_ptr, bool pred) {
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n" : "+r"(s_addr));
    const int bytes = pred ? 16 : 0;
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "r"(bytes), "l"(pol));
}

template <bool GATEUP, int EPI = 0, bool PREFETCH = true, bool TS4E = false,
          typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void v30_2cta_impl(
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
    using L = V30Smem<GATEUP, TS4E, Shape>;
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
    constexpr unsigned n_tiles = N / TN; // 10 for GATEUP, 20 for DOWN
    constexpr unsigned KS = K / 64u;     // 40 for GATEUP, 10 for DOWN
    constexpr unsigned kblocks = K / 32u;
    static_assert((KS & 1u) == 0u && KS >= 4u, "v30_2cta requires even KS >= 4");

    unsigned dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (gridDim.x == 0 || gridDim.x > oi640::FIXED_GRID || blockDim.x != V30_2CTA_THREADS || dyn_smem < L::BYTES) {
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
    const unsigned task_count = wc * n_tiles;
    if (blockIdx.x >= task_count) return;

    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    unsigned char (*const sS)[BMAT][S_ROWS][BCOLS] = reinterpret_cast<unsigned char (*)[BMAT][S_ROWS][BCOLS]>(smem + L::OFF_S);
    unsigned (*const sAS)[V30_2CTA_TILE_M] = reinterpret_cast<unsigned (*)[V30_2CTA_TILE_M]>(smem + L::OFF_AS);
    unsigned char (*const sSF)[2][V30_2CTA_BC] = reinterpret_cast<unsigned char (*)[2][V30_2CTA_BC]>(smem + L::OFF_SF);
    uint2* const lut = reinterpret_cast<uint2*>(smem + L::OFF_LUT);
    unsigned* const sAOff = reinterpret_cast<unsigned*>(smem + L::OFF_AOFF);
    unsigned long long* const sPtrs = reinterpret_cast<unsigned long long*>(smem + L::OFF_PTRS);
    unsigned char* const sBase = smem + L::OFF_BASE;
    unsigned short (*const sExc)[TS4E_B_PER_KS][32] = reinterpret_cast<unsigned short (*)[TS4E_B_PER_KS][32]>(smem + L::OFF_EXC);
    unsigned* const sBktBeg = reinterpret_cast<unsigned*>(smem + L::OFF_BKT_BEG);
    unsigned short* const sBktCnt = reinterpret_cast<unsigned short*>(smem + L::OFF_BKT_CNT);

    const unsigned tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31u;
    if (tid == 0u) {
        sPtrs[5] = reinterpret_cast<unsigned long long>(status);
    }
    lut[tid] = V28_E4M3_LUT[tid];

    const unsigned gid = lane >> 2, tig = lane & 3u;
    const unsigned warp_m = (warp_id >> 1) * 32u;
    const unsigned warp_n = (warp_id & 1u) * (GATEUP ? 32u : 64u);
    const unsigned sf_off = (((warp_id & 1u) * 8u) + gid) << 3;

    #pragma unroll 1
    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const unsigned raw_wi = task / n_tiles;
        const unsigned wi = (!GATEUP) ? (wc - 1u - raw_wi) : raw_wi;
        const uint2 item_u2 = *reinterpret_cast<const uint2*>(&workspace->items[wi]);
        const unsigned item_expert = item_u2.x;
        const unsigned item_m_tile = item_u2.y;
        const unsigned nt = task % n_tiles;
        const unsigned cta_n = nt * TN;
        if (item_expert >= Shape::EXPERTS) {
            if (tid == 0u) oi640::fail(reinterpret_cast<int*>(sPtrs[5]), oi640::ERR_PLAN);
            return;
        }
        const int m_start = expert_offsets[item_expert];
        const int M_expert = expert_offsets[item_expert + 1] - m_start;
        const int local = static_cast<int>(item_m_tile * V30_2CTA_TILE_M);
        const bool range_ok = (m_start >= 0 && M_expert > 0 && local >= 0 && local < M_expert);
        const unsigned cta_m = range_ok ? static_cast<unsigned>(m_start + local) : 0u;
        const int valid_rows = range_ok ? min(M_expert - local, static_cast<int>(V30_2CTA_TILE_M)) : 0;

        if (tid == 0u) {
            const unsigned long long pB0 = pk0[item_expert];
            const unsigned long long pS0 = sc0[item_expert];
            const unsigned long long pB1 = GATEUP ? pk1[item_expert] : pB0;
            const unsigned long long pS1 = GATEUP ? sc1[item_expert] : pS0;
            const float f_s2a = s20[item_expert];
            const float f_s2b = GATEUP ? s21[item_expert] : 0.0f;
            const bool ptr_bad = !range_ok || pB0 == 0ull || pS0 == 0ull || pB1 == 0ull || pS1 == 0ull ||
                                 !isfinite(f_s2a) || !isfinite(f_s2b);
            sPtrs[0] = pB0;
            sPtrs[1] = pS0;
            sPtrs[2] = pB1;
            sPtrs[3] = pS1;
            sPtrs[4] = ptr_bad ? 1ull : 0ull;
        }
        if constexpr (GATEUP) {
            if (tid < V30_2CTA_TILE_M) {
                sAOff[tid] = (static_cast<int>(tid) < valid_rows)
                    ? static_cast<unsigned>(sorted_token_ids[cta_m + tid])
                    : 0u;
            }
        }
        __syncthreads();
        if (sPtrs[4] != 0ull) {
            if (tid == 0u) oi640::fail(reinterpret_cast<int*>(sPtrs[5]), oi640::ERR_POINTER);
            return;
        }

        const int warp_valid = valid_rows - static_cast<int>(warp_m);
        const int rt_active = warp_valid <= 0 ? 0 : min(2, (warp_valid + 15) >> 4);
        const bool b_evict_first = GATEUP ? (local + static_cast<int>(V30_2CTA_TILE_M) >= M_expert)
                                          : (item_m_tile == 0u);
        const unsigned col_half = cta_n & 64u;

        auto load_a = [&](unsigned ks) {
            const int active_a_rows = (valid_rows + 15) & ~15;
            const unsigned a_r0 = tid >> 2, a_ch = tid & 3u;
            const unsigned a_dst0 = sbase + (ks & 1u) * (V30_2CTA_TILE_M * 64u) + v14_swz(a_r0, a_ch);
            if (static_cast<int>(a_r0) < active_a_rows) {
                const bool a_ok0 = static_cast<int>(a_r0) < valid_rows;
                const unsigned src_row0 = a_ok0 ? (GATEUP ? v29_ld_saoff(sbase + L::OFF_AOFF, a_r0) : (cta_m + a_r0)) : 0u;
                const void* g0 = aq + static_cast<unsigned long long>(src_row0) * K + ks * 64u + a_ch * 16u;
                if constexpr (!GATEUP) {
                    if (nt + 1u == n_tiles) v29_cp16_ef_s32(a_dst0, g0, a_ok0);
                    else                    v29_cp16_s32(a_dst0, g0, a_ok0);
                } else {
                    v29_cp16_s32(a_dst0, g0, a_ok0);
                }
            }
            if (valid_rows > 64) {
                const unsigned a_r1 = a_r0 + 64u;
                if (static_cast<int>(a_r1) < active_a_rows) {
                    const bool a_ok1 = static_cast<int>(a_r1) < valid_rows;
                    const unsigned src_row1 = a_ok1 ? (GATEUP ? v29_ld_saoff(sbase + L::OFF_AOFF, a_r1) : (cta_m + a_r1)) : 0u;
                    const void* g1 = aq + static_cast<unsigned long long>(src_row1) * K + ks * 64u + a_ch * 16u;
                    if constexpr (!GATEUP) {
                        if (nt + 1u == n_tiles) v29_cp16_ef_s32(a_dst0 + 4096u, g1, a_ok1);
                        else                    v29_cp16_s32(a_dst0 + 4096u, g1, a_ok1);
                    } else {
                        v29_cp16_s32(a_dst0 + 4096u, g1, a_ok1);
                    }
                }
            }
            if (tid < V30_2CTA_TILE_M && (ks & 1u) == 0u) {
                if (static_cast<int>(tid) < active_a_rows) {
                    const bool sc_ok = static_cast<int>(tid) < valid_rows;
                    const unsigned sc_src_row = sc_ok ? (GATEUP ? v29_ld_saoff(sbase + L::OFF_AOFF, tid) : (cta_m + tid)) : 0u;
                    v29_cp4_s32(sbase + L::OFF_AS + (((ks >> 1) & 1u) * V30_2CTA_TILE_M + tid) * 4u,
                                as + static_cast<unsigned long long>(sc_src_row) * kblocks + ks * 2u, sc_ok);
                }
            }
        };

        auto load_b = [&](unsigned ks) {
            constexpr unsigned CH = BCOLS / 16u; // 4 (GU) or 8 (DN)
            constexpr unsigned SB_SWZ_SHIFT = GATEUP ? 4u : 5u;
            const unsigned sl = ks & 1u;
            {
                const unsigned mtx = tid / (32u * CH), r = (tid / CH) % 32u, cc = tid % CH;
                // Conflict-free 16B chunk XOR-swizzle on sB: all 32 lanes in SWAR LDS.32 hit 32 distinct banks.
                const unsigned b_dst_col = (cc << 4) ^ ((r & 3u) << SB_SWZ_SHIFT);
                const unsigned char* src = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && mtx) ? 2u : 0u);
                const unsigned off_b = (ks * 32u + r) * N + cta_n + cc * 16u;
                const unsigned dst_b = sbase + L::OFF_B + ((sl * BMAT + mtx) * 32u + r) * BCOLS + b_dst_col;
                if (b_evict_first) v29_cp16_ef_s32(dst_b, src + off_b, true);
                else               v30_cp16_el_s32(dst_b, src + off_b, true);
            }
            if constexpr (TS4E) {
                if (warp_id < TS4E_B_PER_KS) {
                    const unsigned sub = warp_id >> 1;
                    const unsigned t = warp_id & 1u;
                    const unsigned mtx = GATEUP ? sub : 0u;
                    if (lane < CH) {
                        const unsigned char* s = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                        const unsigned off_s = (ks * 2u + t) * N + cta_n + lane * 16u;
                        const unsigned dst_s = sbase + L::OFF_S + ((sl * BMAT + mtx) * S_ROWS + 2u * t + 1u) * BCOLS + lane * 16u;
                        if (b_evict_first) v29_cp16_ef_s32(dst_s, s + off_s, true);
                        else               v30_cp16_el_s32(dst_s, s + off_s, true);
                    } else if (lane < CH + 4u) {
                        const unsigned c = lane - CH;
                        const unsigned idx = ks * TS4E_B_PER_KS + warp_id;
                        const unsigned cnt = static_cast<unsigned>(sBktCnt[idx]);
                        if (cnt > 0u) {
                            const unsigned beg = sBktBeg[idx];
                            const unsigned pad = beg & 7u;
                            if (pad + cnt <= 32u && (c << 3) < (pad + cnt)) {
                                const unsigned char* blob = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                                const unsigned short* exc =
                                    reinterpret_cast<const unsigned short*>(blob + TS4E_EXC_OFF);
                                const unsigned dst_e = sbase + L::OFF_EXC + ((sl * TS4E_B_PER_KS + warp_id) * 32u + (c << 3)) * 2u;
                                v29_cp16_s32(dst_e, exc + (beg & ~7u) + (c << 3), true);
                            }
                        }
                    }
                }
            } else {
                if (tid < BMAT * 4u * CH) {
                    const unsigned m2 = tid / (4u * CH), r2 = (tid / CH) % 4u, c2 = tid % CH;
                    const unsigned char* s = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && m2) ? 3u : 1u);
                    const unsigned off_s = (ks * 4u + r2) * N + cta_n + c2 * 16u;
                    const unsigned dst_s = sbase + L::OFF_S + ((sl * BMAT + m2) * S_ROWS + r2) * BCOLS + c2 * 16u;
                    if (b_evict_first) v29_cp16_ef_s32(dst_s, s + off_s, true);
                    else               v30_cp16_el_s32(dst_s, s + off_s, true);
                }
            }
        };

        auto decode_ts4e_stage = [&](unsigned ks_dec) {
            if (warp_id >= TS4E_B_PER_KS) return;
            __syncwarp();
            const unsigned sl = ks_dec & 1u;
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
                    const unsigned char* blob = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
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

        auto prefetch_b = [&](unsigned pf_ks) {
            constexpr unsigned PF_DIST = GATEUP ? 10u : 5u;
            const unsigned pf_raw_wi = raw_wi + PF_DIST;
            if (pf_raw_wi >= wc) return;
            const unsigned pf_wi = (!GATEUP) ? (wc - 1u - pf_raw_wi) : pf_raw_wi;
            const oi640::WorkItem pf_item = workspace->items[pf_wi];
            if (pf_item.expert >= Shape::EXPERTS || pf_item.expert == item_expert) return;
            if constexpr (!GATEUP) {
                if (pf_wi + 1u < wc && workspace->items[pf_wi + 1u].expert == pf_item.expert) return;
            } else {
                if (pf_item.m_tile != 0u) return;
            }
            constexpr unsigned CH = BCOLS / 16u;
            constexpr unsigned S_ROWS_LD = TS4E ? 2u : 4u;
            {
                const unsigned mtx = tid / (32u * CH), r = (tid / CH) % 32u, cc = tid % CH;
                if ((cc & 1u) == 0u) {
                    const unsigned char* pB = reinterpret_cast<const unsigned char*>(
                        (GATEUP && mtx) ? pk1[pf_item.expert] : pk0[pf_item.expert]);
                    if (pB != nullptr) {
                        v28_prefetch_l2(pB + static_cast<unsigned long long>(pf_ks * 32u + r) * N + cta_n + cc * 16u);
                    }
                }
            }
            if (tid < BMAT * S_ROWS_LD * CH) {
                const unsigned m2 = tid / (S_ROWS_LD * CH), r2 = (tid / CH) % S_ROWS_LD, c2 = tid % CH;
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
                    if (tid >= 32u && tid < 36u) {
                        const unsigned j = tid - 32u;
                        const unsigned bm = GATEUP ? (j >> 1) : 0u;
                        const unsigned off32 = GATEUP ? ((j & 1u) << 5) : (j << 5);
                        const unsigned char* pS = reinterpret_cast<const unsigned char*>(
                            (GATEUP && bm) ? sc1[pf_item.expert] : sc0[pf_item.expert]);
                        if (pS != nullptr) {
                            v28_prefetch_l2(pS + static_cast<unsigned long long>(TS4E_NIBBLES_BYTES) + cta_n + off32);
                        }
                    } else if (tid >= 40u && tid < (GATEUP ? 56u : 48u)) {
                        const unsigned j = tid - 40u;
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
        // Fresh %tid.x prevents ptxas from hoisting transcode addresses across emit_pair(0..3).
        auto transcode_16 = [&](unsigned sl, unsigned tb, unsigned t_h_override, unsigned t_blk) {
            constexpr unsigned SB_SWZ_SHIFT = GATEUP ? 4u : 5u;
            unsigned tc_tid;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tc_tid));
            const unsigned t_col = tc_tid & 127u;
            const unsigned t_h = (t_h_override == 2u) ? (tc_tid >> 7) : t_h_override;
            const unsigned t_q = tc_tid & 3u;
            const unsigned t_mtx = GATEUP ? (t_col >> 6) : 0u;
            const unsigned t_lc = GATEUP ? (t_col & 63u) : t_col;
            const unsigned ss_base = sbase + L::OFF_S + ((sl * BMAT + t_mtx) * S_ROWS + 2u * t_blk) * BCOLS + t_lc;
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
            const unsigned sbt_addr = sbase + L::OFF_BT + tb * (V30_2CTA_BC * 64u) + v14_swz(t_col, 2u * t_blk + t_h);
            v28_st_sbt_v2_0(sbt_addr, w0, w1);
            const unsigned p1 = v10_prmt(u1, __shfl_xor_sync(0xffffffffu, u1, 1), sel1);
            const unsigned c1 = v10_prmt(p1, __shfl_xor_sync(0xffffffffu, p1, 2), sel2);
            const unsigned w2 = v10_e2m1x4_to_e4m3x4(c1, tt.x, tt.y);
            const unsigned w3 = v10_e2m1x4_to_e4m3x4(c1 >> 16, tt.x, tt.y);
            v28_st_sbt_v2_8(sbt_addr, w2, w3);
            if (t_h == 0u) {
                const unsigned sf_addr =
                    sbase + L::OFF_SF + (tb * 2u + t_blk) * V30_2CTA_BC + v29_m256_sf_index<GATEUP>(t_col);
                v28_st_su8_vol(sf_addr, static_cast<unsigned>(e32 + 127) & 0xffu);
            }
        };

        float acc[2][8][4];
        #pragma unroll
        for (int r = 0; r < 2; ++r)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.0f;

        if constexpr (TS4E) {
            constexpr unsigned PROLOGUE_BKTS = 2u * TS4E_B_PER_KS;
            auto resolve_ts4e_bkt = [&](unsigned idx) {
                const unsigned ks_i = idx / TS4E_B_PER_KS;
                const unsigned b = idx & (TS4E_B_PER_KS - 1u);
                const unsigned sub = b >> 1;
                const unsigned t = b & 1u;
                const unsigned char* blob = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
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
                    const unsigned char* bs = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && bm) ? 3u : 1u);
                    v10_cp16(&sBase[lane * 16u],
                             bs + static_cast<unsigned long long>(TS4E_NIBBLES_BYTES) + cta_n + bc * 16u,
                             true);
                }
            } else if (tid >= 128u) {
                for (unsigned idx = PROLOGUE_BKTS + (tid - 128u); idx < KS * TS4E_B_PER_KS; idx += 128u) {
                    resolve_ts4e_bkt(idx);
                }
            }
        }

        // Prologue (DA = DB = 2): Gp0 = {B0 (+ sBase/sExc0)}, Gp1 = {A0, B1 (+ sExc1)}.
        load_b(0u);
        v10_commit();
        load_a(0u);
        load_b(1u);
        v10_commit();
        v10_wait<1>();
        if constexpr (TS4E) {
            __syncthreads();
            decode_ts4e_stage(0u);
        }
        __syncthreads();
        transcode_16(0u, 0u, 2u, 0u);
        transcode_16(0u, 0u, 2u, 1u);

        unsigned sfa[2] = {0u, 0u};
        const unsigned swz_a0 = v14_swz(warp_m + (lane & 15u), lane >> 4);
        const unsigned swz_b0 = v14_swz(warp_n + ((lane >> 4) << 3) + (lane & 7u), (lane >> 3) & 1u);
        constexpr unsigned off0 = 0u;
        constexpr unsigned off1 = 1024u;
        constexpr unsigned off2 = GATEUP ? (BCOLS * 64u) : 2048u;
        constexpr unsigned off3 = GATEUP ? (BCOLS * 64u + 1024u) : 3072u;

        auto step = [&](auto par_tag, unsigned ks) {
            constexpr unsigned PAR = decltype(par_tag)::value;   // ks & 1
            constexpr unsigned NEXT_TB = PAR ^ 1u;
            const bool do_tc = (PAR == 0u) ? true : (ks + 1u < KS);
            v10_wait<0>();                                       // A ks, raw B ks+1 landed
            if constexpr (TS4E) {
                if (do_tc) decode_ts4e_stage(ks + 1u);
            }
            __syncthreads();
            if (ks + 1u < KS) load_a(ks + 1u);
            if (ks + 2u < KS) load_b(ks + 2u);
            else if constexpr (PREFETCH) prefetch_b(ks + 2u - KS);
            v10_commit();

            const unsigned next_sl = NEXT_TB;                    // (ks + 1) & 1 == PAR ^ 1
            if (rt_active == 0) {
                if (do_tc) {
                    if (valid_rows <= 64) {
                        if (tid >= 128u) {
                            #pragma unroll 1
                            for (unsigned th_i = 0u; th_i < 2u; ++th_i) {
                                transcode_16(next_sl, NEXT_TB, th_i, 0u);
                                transcode_16(next_sl, NEXT_TB, th_i, 1u);
                            }
                        }
                    } else {
                        transcode_16(next_sl, NEXT_TB, 2u, 0u);
                        transcode_16(next_sl, NEXT_TB, 2u, 1u);
                    }
                }
                return;
            }

            if constexpr (PAR == 0u) {
                const unsigned as_buf = (ks >> 1) & 1u;
                sfa[0] = sAS[as_buf][warp_m + gid + (tig & 1u) * 8u];
                if (rt_active > 1) sfa[1] = sAS[as_buf][warp_m + 16u + gid + (tig & 1u) * 8u];
            }
            const bool run_tc = (valid_rows > 64) && do_tc;

            auto run_half = [&](auto t_tag) {
                constexpr unsigned T = decltype(t_tag)::value;
                const uint2 sfb = *reinterpret_cast<const uint2*>(&sSF[PAR][T][sf_off]);
                const unsigned addr_a0 = sbase + PAR * (V30_2CTA_TILE_M * 64u) + (swz_a0 ^ (T << 5));
                const unsigned addr_b0 = sbase + L::OFF_BT + PAR * (V30_2CTA_BC * 64u) + (swz_b0 ^ (T << 5));
                unsigned afr0[4], afr1[4], bfr[2][4];
                auto emit_pair = [&](auto p_tag, const unsigned (&bf)[4], unsigned sfb_w) {
                    constexpr int P = decltype(p_tag)::value;
                    constexpr int BIDB0 = (P & 1) * 2;
                    constexpr int BIDB1 = BIDB0 + 1;
                    v28_qmma<2 * PAR + T, BIDB0>(acc[0][2 * P], afr0, bf[0], bf[1], sfa[0], sfb_w);
                    v28_qmma<2 * PAR + T, BIDB1>(acc[0][2 * P + 1], afr0, bf[2], bf[3], sfa[0], sfb_w);
                    if (rt_active > 1) {
                        v28_qmma<2 * PAR + T, BIDB0>(acc[1][2 * P], afr1, bf[0], bf[1], sfa[1], sfb_w);
                        v28_qmma<2 * PAR + T, BIDB1>(acc[1][2 * P + 1], afr1, bf[2], bf[3], sfa[1], sfb_w);
                    }
                };
                // Ping-pong bfr[2][4] overlaps ldsm4(p+1) with QMMAs on p, followed by conflict-free SWAR transcode.
                v10_ldsm4(addr_b0 + off0, bfr[0][0], bfr[0][1], bfr[0][2], bfr[0][3]);
                v10_ldsm4(addr_a0, afr0[0], afr0[1], afr0[2], afr0[3]);
                if (rt_active > 1) v10_ldsm4(addr_a0 + 1024u, afr1[0], afr1[1], afr1[2], afr1[3]);
                v10_ldsm4(addr_b0 + off1, bfr[1][0], bfr[1][1], bfr[1][2], bfr[1][3]);
                emit_pair(std::integral_constant<int, 0>{}, bfr[0], sfb.x);
                v10_ldsm4(addr_b0 + off2, bfr[0][0], bfr[0][1], bfr[0][2], bfr[0][3]);
                emit_pair(std::integral_constant<int, 1>{}, bfr[1], sfb.x);
                v10_ldsm4(addr_b0 + off3, bfr[1][0], bfr[1][1], bfr[1][2], bfr[1][3]);
                emit_pair(std::integral_constant<int, 2>{}, bfr[0], sfb.y);
                emit_pair(std::integral_constant<int, 3>{}, bfr[1], sfb.y);
                if (run_tc) {
                    transcode_16(next_sl, NEXT_TB, 2u, T);
                }
            };
            run_half(std::integral_constant<unsigned, 0u>{});
            run_half(std::integral_constant<unsigned, 1u>{});
        };

        #pragma unroll 1
        for (unsigned ks = 0u; ks < KS; ks += 2u) {
            step(std::integral_constant<unsigned, 0u>{}, ks);
            step(std::integral_constant<unsigned, 1u>{}, ks + 1u);
        }
        v10_wait<0>();

        unsigned ep_tid;
        asm volatile("mov.u32 %0, %%tid.x;" : "=r"(ep_tid));
        const unsigned ep_wid = ep_tid >> 5, ep_lane = ep_tid & 31u;
        const unsigned ep_gid = ep_lane >> 2, ep_tig = ep_lane & 3u;
        const unsigned ep_wm = (ep_wid >> 1) * 32u;
        const unsigned ep_wn = (ep_wid & 1u) * (GATEUP ? 32u : 64u);
        const int ep_wv = valid_rows - static_cast<int>(ep_wm);
        const int ep_rt = ep_wv <= 0 ? 0 : min(2, (ep_wv + 15) >> 4);
        const float s2a = s20[item_expert];
        const float s2b = GATEUP ? s21[item_expert] : 0.0f;
        if constexpr (GATEUP) {
            const unsigned blk = ep_wid & 1u;
            const unsigned col8 = ep_wn + ep_tig * 8u;
            const bool keep_oq = (workspace->rows <= 4096u) || (raw_wi + 256u >= wc);
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= ep_rt) break;
                const unsigned lr0 = ep_wm + r * 16u + ep_gid, lr1 = lr0 + 8u;
                {
                    unsigned u0[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float g0 = acc[r][j][0] * s2a, u0f = acc[r][j + 4][0] * s2b;
                        const float g1 = acc[r][j][1] * s2a, u1f = acc[r][j + 4][1] * s2b;
                        const float v0 = g0 * (1.0f / (1.0f + __expf(-g0))) * u0f;
                        const float v1 = g1 * (1.0f / (1.0f + __expf(-g1))) * u1f;
                        u0[j] = v28_pack_bf162(v0, v1);
                    }
                    unsigned qq0[2], ss0;
                    v28_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], ep_tig, qq0, ss0);
                    if (static_cast<int>(lr0) < valid_rows) {
                        const unsigned long long orow = cta_m + lr0;
                        if (keep_oq) v28_st8_evict_last(oq + orow * N + cta_n + col8, make_uint2(qq0[0], qq0[1]));
                        else         v28_st8_evict_first(oq + orow * N + cta_n + col8, make_uint2(qq0[0], qq0[1]));
                        if (ep_tig == 0u) os[orow * (N / 32u) + cta_n / 32u + blk] = static_cast<unsigned char>(ss0);
                    }
                }
                asm volatile("" ::: "memory");
                {
                    unsigned u1[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float g2 = acc[r][j][2] * s2a, u2f = acc[r][j + 4][2] * s2b;
                        const float g3 = acc[r][j][3] * s2a, u3f = acc[r][j + 4][3] * s2b;
                        const float v2 = g2 * (1.0f / (1.0f + __expf(-g2))) * u2f;
                        const float v3 = g3 * (1.0f / (1.0f + __expf(-g3))) * u3f;
                        u1[j] = v28_pack_bf162(v2, v3);
                    }
                    unsigned qq1[2], ss1;
                    v28_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], ep_tig, qq1, ss1);
                    if (static_cast<int>(lr1) < valid_rows) {
                        const unsigned long long orow = cta_m + lr1;
                        if (keep_oq) v28_st8_evict_last(oq + orow * N + cta_n + col8, make_uint2(qq1[0], qq1[1]));
                        else         v28_st8_evict_first(oq + orow * N + cta_n + col8, make_uint2(qq1[0], qq1[1]));
                        if (ep_tig == 0u) os[orow * (N / 32u) + cta_n / 32u + blk] = static_cast<unsigned char>(ss1);
                    }
                }
                asm volatile("" ::: "memory");
            }
        } else {
            const bool keep_slot =
                (static_cast<unsigned long long>(workspace->expanded) * Shape::SLOT_MXFP8_BYTES_ULL) <= (12ull << 20);
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= ep_rt) break;
                const unsigned lr0 = ep_wm + r * 16u + ep_gid, lr1 = lr0 + 8u;
                #pragma unroll
                for (unsigned b = 0u; b < 2u; ++b) {
                    const unsigned col = cta_n + ep_wn + b * 32u;
                    const unsigned col8 = col + ep_tig * 8u;
                    {
                        unsigned u0[4];
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int j = 4 * static_cast<int>(b) + jj;
                            u0[jj] = v28_pack_bf162(acc[r][j][0] * s2a, acc[r][j][1] * s2a);
                        }
                        if constexpr (EPI == 3) {
                            unsigned qq0[2], ss0;
                            v28_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], ep_tig, qq0, ss0);
                            if (static_cast<int>(lr0) < valid_rows) {
                                const unsigned long long prow = cta_m + lr0;
                                if (keep_slot) {
                                    v28_st8_l2_evict_last(slot_q + prow * N + col8, make_uint2(qq0[0], qq0[1]));
                                } else {
                                    v28_st8_evict_first(slot_q + prow * N + col8, make_uint2(qq0[0], qq0[1]));
                                }
                                if (ep_tig == 0u) slot_s[prow * (N / 32u) + col / 32u] = static_cast<unsigned char>(ss0);
                            }
                        } else {
                            const uint4 w0 = v28_bf16_reg_quad(u0[0], u0[1], u0[2], u0[3], ep_tig);
                            if (static_cast<int>(lr0) < valid_rows) {
                                v28_st16_stream(C + static_cast<unsigned long long>(cta_m + lr0) * N + col8, w0);
                            }
                        }
                    }
                    asm volatile("" ::: "memory");
                    {
                        unsigned u1[4];
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int j = 4 * static_cast<int>(b) + jj;
                            u1[jj] = v28_pack_bf162(acc[r][j][2] * s2a, acc[r][j][3] * s2a);
                        }
                        if constexpr (EPI == 3) {
                            unsigned qq1[2], ss1;
                            v28_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], ep_tig, qq1, ss1);
                            if (static_cast<int>(lr1) < valid_rows) {
                                const unsigned long long prow = cta_m + lr1;
                                if (keep_slot) {
                                    v28_st8_l2_evict_last(slot_q + prow * N + col8, make_uint2(qq1[0], qq1[1]));
                                } else {
                                    v28_st8_evict_first(slot_q + prow * N + col8, make_uint2(qq1[0], qq1[1]));
                                }
                                if (ep_tig == 0u) slot_s[prow * (N / 32u) + col / 32u] = static_cast<unsigned char>(ss1);
                            }
                        } else {
                            const uint4 w1 = v28_bf16_reg_quad(u1[0], u1[1], u1[2], u1[3], ep_tig);
                            if (static_cast<int>(lr1) < valid_rows) {
                                v28_st16_stream(C + static_cast<unsigned long long>(cta_m + lr1) * N + col8, w1);
                            }
                        }
                    }
                    asm volatile("" ::: "memory");
                }
            }
        }
        if (task + gridDim.x < task_count) __syncthreads();
    }
}

} // namespace

// 2-CTA/SM V30 Flat-scale entry points (46,144 B GATEUP / 45,632 B DOWN SMEM, 256 threads).
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_gateup_k32_v30_2cta(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v30_2cta_impl<true, 0, true, false>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_k32_v30_2cta(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v30_2cta_impl<false, 0, true, false>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_k32_v30q_2cta(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v30_2cta_impl<false, 3, true, false>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
        slot_q, slot_s);
}

// 2-CTA/SM V30 Atlas-TS4E-128 entry points (47,744 B GATEUP / 46,144 B DOWN SMEM, 256 threads).
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_gateup_k32_v30_2cta_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v30_2cta_impl<true, 0, true, true>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_k32_v30_2cta_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v30_2cta_impl<false, 0, true, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_k32_v30q_2cta_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v30_2cta_impl<false, 3, true, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
        slot_q, slot_s);
}

