// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// V28Q / V28Q4 W4A8 MoE GEMM kernels with direct MXFP8 slot output and next-task L2 prefetch.

// ---------------------------------------------------------------------------
// K11-D / K11-B2: Routed MoE V28q (W4A8 V28 GATEUP + DOWN MXFP8 Slot Epilogue) with:
//   1. 512-thread cooperative prologue transcode (cuts prologue transcode latency in half)
//   2. In-loop transcode warp-half swap to tid >= 256 (Warps 8..15, which have rt_active == 0
//      whenever valid_rows <= 64, overlapping transcode with Warps 0..7's MMAs)
//   3. Quad-lane (4 lanes per MX32 block) coalesced MXFP8 epilogue for both GATEUP (v28q4)
//      and DOWN MXFP8 slots (v28q, replacing legacy v17q = v14_impl<false, 3>)
//   4. Optional Cross-Task L2 Weight Prefetch (_pf variants) during the final 3 k64 steps
//      when the current task's load_b pipeline is idle, warming the 3 prologue stages of the
//      upcoming cold expert (m_tile == 0) in L2.
//   5. R1+R2+R3 instruction/smem pipeline:
//      - R1: Hoist sfa[0]/sfa[1] (sAS loads) to the (ks, ks+1) pair (PAR == 0) (-75% sAS LDS.U32).
//      - R2: Guard r=1 ldmatrix_a + pipeline LDSM via v28_qmma4_lo / v28_qmma4_hi.
//      - R3: Guard load_a cp.async with active_rows_pad16 = (valid_rows + 15u) & ~15u.
//   6. K11-B2 L2 cache residency policy:
//      - Multi-use activations (aq/as) and multi-tile expert weights (b_reused) use L2::evict_last.
//      - Single-tile / final-tile expert weights (!b_reused) and final N-tile DOWN activations use L2::evict_first.
//      - GATEUP oq stores use L2::evict_last; DOWN MXFP8 slot_q stores use L2::evict_first.
// ---------------------------------------------------------------------------
namespace {

__device__ __forceinline__ void v28_prefetch_l2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];\n" :: "l"(ptr));
}

template <bool GATEUP, bool DOWN_MXFP8, bool PREFETCH, bool W8_ONLY = false, typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void v28q_impl(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ pk0, const unsigned long long* __restrict__ sc0,
    const float* __restrict__ s20,
    const unsigned long long* __restrict__ pk1, const unsigned long long* __restrict__ sc1,
    const float* __restrict__ s21,
    __nv_bfloat16* __restrict__ C, unsigned char* __restrict__ oq, unsigned char* __restrict__ os,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned N, unsigned K
) {
    using L = V14Smem<GATEUP>;
    constexpr unsigned TN = GATEUP ? 128u : 256u;
    constexpr unsigned BMAT = L::BMAT, BCOLS = L::BCOLS;
    unsigned dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (gridDim.x == 0 || gridDim.x > oi640::FIXED_GRID || blockDim.x != V14_THREADS || dyn_smem < L::BYTES) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }
    const bool shape = GATEUP ? (N == Shape::INTER && K == Shape::HIDDEN)
                              : (N == Shape::HIDDEN && K == Shape::INTER);
    const bool outs = (GATEUP || DOWN_MXFP8) ? (oq != nullptr && os != nullptr) : (C != nullptr);
    const bool ptrs = pk0 != nullptr && sc0 != nullptr && s20 != nullptr &&
                      (!GATEUP || (pk1 != nullptr && sc1 != nullptr && s21 != nullptr));
    if (!shape || !outs || !ptrs || aq == nullptr || as == nullptr || expert_offsets == nullptr ||
        sorted_token_ids == nullptr || !oi640::contract_ok(contract) ||
        !oi640::workspace_ok(workspace, contract) || status == nullptr || status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_ABI);
        return;
    }
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* const sA = smem;                                                     // [DA][128][64] swz
    unsigned char (*const sB)[BMAT][32][BCOLS] = reinterpret_cast<unsigned char (*)[BMAT][32][BCOLS]>(smem + L::OFF_B);
    unsigned char (*const sS)[BMAT][4][BCOLS] = reinterpret_cast<unsigned char (*)[BMAT][4][BCOLS]>(smem + L::OFF_S);
    unsigned (*const sAS)[V10_TILE_M] = reinterpret_cast<unsigned (*)[V10_TILE_M]>(smem + L::OFF_AS);
    unsigned char* const sBt = smem + L::OFF_BT;                                        // [2][256][64] swz
    unsigned char (*const sSF)[2][V14_BC] = reinterpret_cast<unsigned char (*)[2][V14_BC]>(smem + L::OFF_SF);
    uint2* const lut = reinterpret_cast<uint2*>(smem + L::OFF_LUT);
    __nv_bfloat16* const sE = reinterpret_cast<__nv_bfloat16*>(smem);

    const unsigned tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    const unsigned n_tiles = N / TN;
    const unsigned task_count = workspace->work_count * n_tiles;
    if (blockIdx.x >= task_count) return;
    if (tid < 256) {
        lut[tid] = V28_E4M3_LUT[tid];
    }

    const unsigned warp_n = (warp_id & 3) * (GATEUP ? 32u : 64u);
    const unsigned gid = lane >> 2, tig = lane & 3;
    const unsigned KS = K / 64;
    const unsigned kblocks = K / 32;

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const oi640::WorkItem item = workspace->items[task / n_tiles];
        const unsigned nt = task % n_tiles;
        const unsigned cta_n = nt * TN;
        if (item.expert >= Shape::EXPERTS) { if (tid == 0) oi640::fail(status, oi640::ERR_PLAN); return; }
        const int m_start = expert_offsets[item.expert];
        const int M_expert = expert_offsets[item.expert + 1] - m_start;
        const int local = static_cast<int>(item.m_tile * V10_TILE_M);
        const unsigned char* B0 = reinterpret_cast<const unsigned char*>(pk0[item.expert]);
        const unsigned char* S0 = reinterpret_cast<const unsigned char*>(sc0[item.expert]);
        const unsigned char* B1 = GATEUP ? reinterpret_cast<const unsigned char*>(pk1[item.expert]) : B0;
        const unsigned char* S1 = GATEUP ? reinterpret_cast<const unsigned char*>(sc1[item.expert]) : S0;
        const float s2a = s20[item.expert];
        const float s2b = GATEUP ? s21[item.expert] : 0.0f;
        if (m_start < 0 || M_expert < 0 || local >= M_expert || B0 == nullptr || S0 == nullptr ||
            B1 == nullptr || S1 == nullptr || !isfinite(s2a) || !isfinite(s2b)) {
            if (tid == 0) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const unsigned cta_m = static_cast<unsigned>(m_start + local);
        const int valid_rows = min(M_expert - local, static_cast<int>(V10_TILE_M));
        const unsigned active_rows_pad16 = (static_cast<unsigned>(valid_rows) + 15u) & ~15u;
        const bool m32 = (valid_rows <= 32);
        const unsigned warp_m = (warp_id >> 2) * (m32 ? 16u : 32u);
        const int warp_valid = valid_rows - static_cast<int>(warp_m);
        const int rt_active = warp_valid <= 0 ? 0 : (m32 ? 1 : min(2, (warp_valid + 15) >> 4));
        const bool a_keep = GATEUP || (nt + 1u < n_tiles);
        const bool b_reused = (local + static_cast<int>(V10_TILE_M) < M_expert);

        // A: 128 rows x 64 bytes per stage = 512 chunks, one per thread.
        // Rows >= active_rows_pad16 belong to inactive 16-row MMA tiles and are never read by ldmatrix_a.
        const unsigned a_row = tid >> 2, a_ch = tid & 3;
        const bool a_ok = static_cast<int>(a_row) < valid_rows;
        const unsigned a_src_row = a_ok ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + a_row]) : cta_m + a_row) : 0u;
        const unsigned char* a_src = aq + static_cast<unsigned long long>(a_src_row) * K + a_ch * 16;
        const unsigned a_dst = v14_swz(a_row, a_ch);
        const unsigned sc_row = tid - 384;
        const bool sc_ok = tid >= 384 && static_cast<int>(sc_row) < valid_rows;
        const unsigned sc_src_row = sc_ok ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + sc_row]) : cta_m + sc_row) : 0u;
        const unsigned char* sc_src = as + static_cast<unsigned long long>(sc_src_row) * kblocks;

        auto load_a = [&](unsigned ks) {
            if (a_row < active_rows_pad16) {
                if (a_keep) {
                    v28_cp16_evict_last_pred(sA + (ks % V14_DA) * (V10_TILE_M * 64) + a_dst, a_src + ks * 64, a_ok);
                } else {
                    v28_cp16_evict_first_pred(sA + (ks % V14_DA) * (V10_TILE_M * 64) + a_dst, a_src + ks * 64, a_ok);
                }
            }
            if (tid >= 384 && (ks & 1u) == 0u && sc_row < active_rows_pad16) {        // four k32 scale bytes cover stages ks, ks+1
                if (a_keep) {
                    v28_cp4_evict_last_pred(&sAS[(ks >> 1) & 1][sc_row], sc_src + ks * 2, sc_ok);
                } else {
                    v28_cp4_evict_first_pred(&sAS[(ks >> 1) & 1][sc_row], sc_src + ks * 2, sc_ok);
                }
            }
        };
        auto load_b = [&](unsigned ks) {
            constexpr unsigned CH = BCOLS / 16;
            const unsigned mtx = tid / (32 * CH), r = (tid / CH) % 32, cc = tid % CH;
            const unsigned char* src = mtx ? B1 : B0;
            const void* g_b = src + static_cast<unsigned long long>(ks * 32 + r) * N + cta_n + cc * 16;
            if (b_reused) {
                v28_cp16_evict_last(&sB[ks % V14_DB][mtx][r][cc * 16], g_b);
            } else {
                v28_cp16_evict_first(&sB[ks % V14_DB][mtx][r][cc * 16], g_b);
            }
            if (tid < BMAT * 4 * CH) {
                const unsigned m2 = tid / (4 * CH), r2 = (tid / CH) % 4, c2 = tid % CH;
                const unsigned char* s = m2 ? S1 : S0;
                const void* g_s = s + static_cast<unsigned long long>(ks * 4 + r2) * N + cta_n + c2 * 16;
                v28_cp16_evict_last(&sS[ks % V14_DB][m2][r2][c2 * 16], g_s);
            }
        };
        auto prefetch_b = [&](unsigned pf_ks) {
            constexpr unsigned PF_DIST = GATEUP ? 10u : 5u;
            const unsigned pf_idx = (task / n_tiles) + PF_DIST;
            if (pf_idx >= workspace->work_count) return;
            const oi640::WorkItem pf_item = workspace->items[pf_idx];
            if (pf_item.m_tile != 0u || pf_item.expert >= Shape::EXPERTS || pf_item.expert == item.expert) return;
            const int p_m = expert_offsets[pf_item.expert];
            const int p_M = expert_offsets[pf_item.expert + 1] - p_m;
            if (p_m < 0 || p_M <= 0) return;
            constexpr unsigned CH = BCOLS / 16u;
            const unsigned mtx = tid / (32u * CH), r = (tid / CH) % 32u, cc = tid % CH;
            if ((cc & 1u) == 0u) {
                const unsigned char* pB = reinterpret_cast<const unsigned char*>(
                    (GATEUP && mtx) ? pk1[pf_item.expert] : pk0[pf_item.expert]);
                if (pB != nullptr) {
                    v28_prefetch_l2(pB + static_cast<unsigned long long>(pf_ks * 32u + r) * N + cta_n + cc * 16u);
                }
            }
            if (tid < BMAT * 4u * CH) {
                const unsigned m2 = tid / (4u * CH), r2 = (tid / CH) % 4u, c2 = tid % CH;
                if ((c2 & 1u) == 0u) {
                    const unsigned char* pS = reinterpret_cast<const unsigned char*>(
                        (GATEUP && m2) ? sc1[pf_item.expert] : sc0[pf_item.expert]);
                    if (pS != nullptr) {
                        v28_prefetch_l2(pS + static_cast<unsigned long long>(pf_ks * 4u + r2) * N + cta_n + c2 * 16u);
                    }
                }
            }
        };
        auto transcode_unit = [&](unsigned sl, unsigned tb, unsigned t_col, unsigned t_h) {
            const unsigned t_mtx = t_col / BCOLS, t_lc = t_col % BCOLS;
            #pragma unroll
            for (unsigned t = 0; t < 2; ++t) {
                const unsigned s_own = sS[sl][t_mtx][2 * t + t_h][t_lc], s_oth = sS[sl][t_mtx][2 * t + (t_h ^ 1)][t_lc];
                const unsigned s_max = max(s_own & 0x7fu, s_oth & 0x7fu);
                const int e32 = (s_max >= 8u) ? (static_cast<int>(s_max >> 3) - 7)
                              : (s_max != 0u) ? (22 - __clz(static_cast<int>(s_max)))
                                              : 0;
                const float inv = __int_as_float((127 - e32) << 23);
                const unsigned r = __nv_cvt_float_to_fp8(v10_e4m3_to_float(s_own) * inv, __NV_SATFINITE, __NV_E4M3);
                const uint2 tt = lut[r];
                unsigned w[4];
                #pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const unsigned x = static_cast<unsigned>(sB[sl][t_mtx][16 * t + 8 * t_h + 2 * g][t_lc]) |
                                       (static_cast<unsigned>(sB[sl][t_mtx][16 * t + 8 * t_h + 2 * g + 1][t_lc]) << 8);
                    w[g] = v10_e2m1x4_to_e4m3x4(x, tt.x, tt.y);
                }
                *reinterpret_cast<uint4*>(sBt + tb * (V14_BC * 64) + v14_swz(t_col, 2 * t + t_h)) =
                    make_uint4(w[0], w[1], w[2], w[3]);
                if (t_h == 0) sSF[tb][t][v28_sf_index<GATEUP>(t_col)] = static_cast<unsigned char>((e32 + 127) & 0xff);
            }
        };
        // Adaptive in-loop transcode (K11-R11):
        // - valid_rows <= 64: warps 8..15 (tid >= 256) have rt_active == 0 and overlap 100% with warps 0..7's MMAs.
        // - valid_rows > 64: all 16 warps have active MMAs, so all 512 threads cooperate to cut transcode latency in half.
        auto transcode = [&](unsigned sl, unsigned tb) {
            if constexpr (W8_ONLY) {
                if (tid >= V14_BC) {
                    const unsigned t_col = tid - V14_BC;
                    #pragma unroll 1
                    for (unsigned th = 0u; th < 2u; ++th) {
                        transcode_unit(sl, tb, t_col, th);
                    }
                }
                return;
            }
            if (valid_rows <= 64) {
                if (tid >= V14_BC) {
                    const unsigned t_col = tid - V14_BC;
                    transcode_unit(sl, tb, t_col, 0);
                    transcode_unit(sl, tb, t_col, 1);
                }
            } else {
                transcode_unit(sl, tb, tid & 255u, tid >> 8);
            }
        };

        float acc[2][8][4];
        #pragma unroll
        for (int r = 0; r < 2; ++r)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.0f;

        // prologue: Gp0 = {B0}, Gp1 = {A0, B1}, Gp2 = {A1, B2}
        load_b(0);
        v10_commit();
        load_a(0);
        if (KS > 1) load_b(1);
        v10_commit();
        if (KS > 1) load_a(1);
        if (KS > 2) load_b(2);
        v10_commit();
        v10_wait<2>();
        __syncthreads();
        // 512-thread cooperative prologue transcode: tid 0..255 -> t_h=0, tid 256..511 -> t_h=1.
        transcode_unit(0, 0, tid & 255u, tid >> 8);

        unsigned sfa[2] = {0u, 0u};
        auto step = [&](auto par_tag, unsigned ks) {
            constexpr unsigned PAR = decltype(par_tag)::value;   // ks & 1
            v10_wait<1>();                                       // A ks, raw B ks+1 landed
            __syncthreads();
            if (ks + 2 < KS) load_a(ks + 2);
            if (ks + 3 < KS) load_b(ks + 3);
            else if (PREFETCH) prefetch_b(ks + 3u - KS);
            v10_commit();
            if (ks + 1 < KS) transcode((ks + 1) % V14_DB, PAR ^ 1);
            if (rt_active == 0) return;
            if constexpr (PAR == 0) {
                const unsigned as_buf = (ks >> 1) & 1;
                sfa[0] = sAS[as_buf][warp_m + gid + (tig & 1) * 8];
                if (rt_active > 1) sfa[1] = sAS[as_buf][warp_m + 16 + gid + (tig & 1) * 8];
            }
            const unsigned a_slot = ks % V14_DA;
            #pragma unroll
            for (unsigned t = 0; t < 2; ++t) {
                unsigned afr0[4], afr1[4], bfr[8][2];
                const unsigned row_a0 = warp_m + (lane & 15);
                const unsigned addr_a0 = static_cast<unsigned>(__cvta_generic_to_shared(
                    sA + a_slot * (V10_TILE_M * 64) + v14_swz(row_a0, 2 * t + (lane >> 4))));
                v10_ldsm4(addr_a0, afr0[0], afr0[1], afr0[2], afr0[3]);
                const uint2 sfb = *reinterpret_cast<const uint2*>(&sSF[PAR][t][((warp_id & 3) * 8 + gid) << 3]);
                #pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const unsigned rbase = GATEUP ? (p >> 1) * BCOLS + warp_n + (p & 1) * 16 : warp_n + p * 16;
                    const unsigned row = rbase + ((lane >> 4) << 3) + (lane & 7);
                    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(
                        sBt + PAR * (V14_BC * 64) + v14_swz(row, 2 * t + ((lane >> 3) & 1))));
                    v10_ldsm4(addr, bfr[2 * p][0], bfr[2 * p][1], bfr[2 * p + 1][0], bfr[2 * p + 1][1]);
                }
                #pragma unroll
                for (int p = 2; p < 4; ++p) {
                    const unsigned rbase = GATEUP ? (p >> 1) * BCOLS + warp_n + (p & 1) * 16 : warp_n + p * 16;
                    const unsigned row = rbase + ((lane >> 4) << 3) + (lane & 7);
                    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(
                        sBt + PAR * (V14_BC * 64) + v14_swz(row, 2 * t + ((lane >> 3) & 1))));
                    v10_ldsm4(addr, bfr[2 * p][0], bfr[2 * p][1], bfr[2 * p + 1][0], bfr[2 * p + 1][1]);
                }
                if (rt_active > 1) {
                    const unsigned row_a1 = warp_m + 16 + (lane & 15);
                    const unsigned addr_a1 = static_cast<unsigned>(__cvta_generic_to_shared(
                        sA + a_slot * (V10_TILE_M * 64) + v14_swz(row_a1, 2 * t + (lane >> 4))));
                    v10_ldsm4(addr_a1, afr1[0], afr1[1], afr1[2], afr1[3]);
                }
                if (t == 0) {
                    v28_qmma4_lo<2 * PAR>(acc[0], afr0, bfr, sfa[0], sfb.x);
                    v28_qmma4_hi<2 * PAR>(acc[0], afr0, bfr, sfa[0], sfb.y);
                    if (rt_active > 1) {
                        v28_qmma4_lo<2 * PAR>(acc[1], afr1, bfr, sfa[1], sfb.x);
                        v28_qmma4_hi<2 * PAR>(acc[1], afr1, bfr, sfa[1], sfb.y);
                    }
                } else {
                    v28_qmma4_lo<2 * PAR + 1>(acc[0], afr0, bfr, sfa[0], sfb.x);
                    v28_qmma4_hi<2 * PAR + 1>(acc[0], afr0, bfr, sfa[0], sfb.y);
                    if (rt_active > 1) {
                        v28_qmma4_lo<2 * PAR + 1>(acc[1], afr1, bfr, sfa[1], sfb.x);
                        v28_qmma4_hi<2 * PAR + 1>(acc[1], afr1, bfr, sfa[1], sfb.y);
                    }
                }
            }
        };
        #pragma unroll 1
        for (unsigned ks = 0; ks < KS; ks += 2) {
            step(std::integral_constant<unsigned, 0>{}, ks);
            step(std::integral_constant<unsigned, 1>{}, ks + 1);
        }
        v10_wait<0>();

        if (GATEUP) {
            (void)sE;
            const unsigned blk = warp_id & 3u;
            const unsigned col8 = warp_n + tig * 8u;
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= rt_active) break;
                const unsigned lr0 = warp_m + r * 16 + gid, lr1 = lr0 + 8;
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
                v28_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], tig, qq0, ss0);
                if (static_cast<int>(lr0) < valid_rows) {
                    const unsigned long long orow = cta_m + lr0;
                    v28_st8_evict_last(oq + orow * N + cta_n + col8, make_uint2(qq0[0], qq0[1]));
                    if (tig == 0u) os[orow * (N / 32) + cta_n / 32 + blk] = static_cast<unsigned char>(ss0);
                }
                v28_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], tig, qq1, ss1);
                if (static_cast<int>(lr1) < valid_rows) {
                    const unsigned long long orow = cta_m + lr1;
                    v28_st8_evict_last(oq + orow * N + cta_n + col8, make_uint2(qq1[0], qq1[1]));
                    if (tig == 0u) os[orow * (N / 32) + cta_n / 32 + blk] = static_cast<unsigned char>(ss1);
                }
            }
        } else if (DOWN_MXFP8) {
            (void)sE;
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= rt_active) break;
                const unsigned lr0 = warp_m + r * 16 + gid, lr1 = lr0 + 8;
                #pragma unroll
                for (unsigned b = 0; b < 2; ++b) {
                    unsigned u0[4], u1[4];
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const int j = 4 * static_cast<int>(b) + jj;
                        u0[jj] = v28_pack_bf162(acc[r][j][0] * s2a, acc[r][j][1] * s2a);
                        u1[jj] = v28_pack_bf162(acc[r][j][2] * s2a, acc[r][j][3] * s2a);
                    }
                    const unsigned blk = (warp_id & 3u) * 2u + b;
                    const unsigned col8 = warp_n + b * 32u + tig * 8u;
                    unsigned qq0[2], ss0, qq1[2], ss1;
                    v28_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], tig, qq0, ss0);
                    if (static_cast<int>(lr0) < valid_rows) {
                        const unsigned long long orow = cta_m + lr0;
                        v28_st8_evict_first(oq + orow * N + cta_n + col8, make_uint2(qq0[0], qq0[1]));
                        if (tig == 0u) os[orow * (N / 32) + cta_n / 32 + blk] = static_cast<unsigned char>(ss0);
                    }
                    v28_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], tig, qq1, ss1);
                    if (static_cast<int>(lr1) < valid_rows) {
                        const unsigned long long orow = cta_m + lr1;
                        v28_st8_evict_first(oq + orow * N + cta_n + col8, make_uint2(qq1[0], qq1[1]));
                        if (tig == 0u) os[orow * (N / 32) + cta_n / 32 + blk] = static_cast<unsigned char>(ss1);
                    }
                }
            }
        } else {
            __syncthreads();
            unsigned char* const stage = sBt + warp_id * (8 * 144);
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= rt_active) break;
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        *reinterpret_cast<__nv_bfloat162*>(stage + gid * 144 + (j * 8 + tig * 2) * 2) =
                            __floats2bfloat162_rn(acc[r][j][2 * hh] * s2a, acc[r][j][2 * hh + 1] * s2a);
                    __syncwarp();
                    #pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const unsigned chunk = lane + 32 * q, row = chunk >> 3, cc = chunk & 7;
                        const unsigned lr = warp_m + r * 16 + hh * 8 + row;
                        if (static_cast<int>(lr) < valid_rows)
                            v28_st16_stream(
                                C + static_cast<unsigned long long>(cta_m + lr) * N + cta_n + warp_n + cc * 8,
                                *reinterpret_cast<const uint4*>(stage + row * 144 + cc * 16));
                    }
                    __syncwarp();
                }
            }
        }
        if (task + gridDim.x < task_count) __syncthreads();
    }
}

} // namespace

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v28q4(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v28q_impl<true, false, false>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v28q4_pf(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v28q_impl<true, false, true>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28q(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28q_impl<false, true, false>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, slot_q, slot_s,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28q_pf(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28q_impl<false, true, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, slot_q, slot_s,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

