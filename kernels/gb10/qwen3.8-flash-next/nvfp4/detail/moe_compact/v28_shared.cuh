// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// V28 shared-expert pretranscode, 4-stage, discard, and 2-stage down GEMM kernels.

namespace {

struct V28SharedSmem {
    static constexpr unsigned A = 4 * V10_TILE_M * 64;
    static constexpr unsigned BT = 4 * V14_BC * 64;
    static constexpr unsigned SF = 4 * 2 * V14_BC;
    static constexpr unsigned AS = 2 * V10_TILE_M * 4;
    static constexpr unsigned OFF_BT = A;
    static constexpr unsigned OFF_SF = OFF_BT + BT;
    static constexpr unsigned OFF_AS = OFF_SF + SF;
    static constexpr unsigned BYTES = OFF_AS + AS;
    static_assert(BYTES == 101376, "DB=4 shared smem fits 99 KiB GB10 cap");
    static_assert(V10_TILE_M * V14_E_STRIDE * 2 <= BYTES, "epilogue staging fits shared memory");
};

__device__ __forceinline__ void v28_cp16_l2_el_s32(unsigned s_addr, const void* gmem_ptr, unsigned long long pol_el) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, 16, %2;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "l"(pol_el));
}

__device__ __forceinline__ void v28_st16_evict_last(void* dst, uint4 v) {
    asm volatile("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__device__ __forceinline__ void v28_st16_stream(void* dst, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

template <bool GATEUP>
__device__ __forceinline__ void v28_pretranscode_slice(
    const unsigned char* __restrict__ B0, const unsigned char* __restrict__ S0,
    const unsigned char* __restrict__ B1, const unsigned char* __restrict__ S1,
    unsigned int N, unsigned int cta_n, unsigned int ks,
    unsigned char* __restrict__ out_bt, unsigned char* __restrict__ out_sf,
    unsigned char (*sB)[32][256], unsigned char (*sS)[4][256],
    unsigned char* sBt, unsigned char (*sSF)[V14_BC], const uint2* lut
) {
    constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    constexpr unsigned BCOLS = V14_BC / BMAT;
    constexpr unsigned CH = BCOLS / 16;
    const unsigned tid = threadIdx.x;
    const unsigned mtx = tid / (32 * CH), r = (tid / CH) % 32, cc = tid % CH;
    const unsigned char* src = (GATEUP && mtx) ? B1 : B0;
    v28_cp16_evict_first(&sB[mtx][r][cc * 16],
                         src + static_cast<unsigned long long>(ks * 32 + r) * N + cta_n + cc * 16);
    if (tid < BMAT * 4 * CH) {
        const unsigned m2 = tid / (4 * CH), r2 = (tid / CH) % 4, c2 = tid % CH;
        const unsigned char* s = (GATEUP && m2) ? S1 : S0;
        v28_cp16_evict_last(&sS[m2][r2][c2 * 16],
                            s + static_cast<unsigned long long>(ks * 4 + r2) * N + cta_n + c2 * 16);
    }
    v10_commit();
    v10_wait<0>();
    __syncthreads();

    // 512-thread cooperative transcode: tid 0..255 -> t_h=0, tid 256..511 -> t_h=1.
    {
        const unsigned t_col = tid & 255u, t_h = tid >> 8;
        const unsigned t_mtx = t_col / BCOLS, t_lc = t_col % BCOLS;
        #pragma unroll
        for (unsigned t = 0; t < 2; ++t) {
            const unsigned s_own = sS[t_mtx][2 * t + t_h][t_lc];
            const unsigned s_oth = sS[t_mtx][2 * t + (t_h ^ 1)][t_lc];
            const unsigned s_max = max(s_own & 0x7fu, s_oth & 0x7fu);
            const int e32 = (s_max >= 8u) ? (static_cast<int>(s_max >> 3) - 7)
                          : (s_max != 0u) ? (22 - __clz(static_cast<int>(s_max)))
                                          : 0;
            const float inv = __int_as_float((127 - e32) << 23);
            const unsigned rc = __nv_cvt_float_to_fp8(v10_e4m3_to_float(s_own) * inv, __NV_SATFINITE, __NV_E4M3);
            const uint2 tt = lut[rc];
            unsigned w[4];
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const unsigned x = static_cast<unsigned>(sB[t_mtx][16 * t + 8 * t_h + 2 * g][t_lc]) |
                                   (static_cast<unsigned>(sB[t_mtx][16 * t + 8 * t_h + 2 * g + 1][t_lc]) << 8);
                w[g] = v10_e2m1x4_to_e4m3x4(x, tt.x, tt.y);
            }
            *reinterpret_cast<uint4*>(sBt + v14_swz(t_col, 2 * t + t_h)) =
                make_uint4(w[0], w[1], w[2], w[3]);
            if (t_h == 0) {
                sSF[t][v28_sf_index<GATEUP>(t_col)] = static_cast<unsigned char>((e32 + 127) & 0xff);
            }
        }
    }
    __syncthreads();

    v28_st16_evict_last(reinterpret_cast<uint4*>(out_bt) + tid,
                        reinterpret_cast<const uint4*>(sBt)[tid]);
    v28_st16_evict_last(reinterpret_cast<uint4*>(out_bt) + tid + 512,
                        reinterpret_cast<const uint4*>(sBt)[tid + 512]);
    if (tid < 32) {
        v28_st16_evict_last(reinterpret_cast<uint4*>(out_sf) + tid,
                            reinterpret_cast<const uint4*>(&sSF[0][0])[tid]);
    }
}

template <bool GATEUP, bool DISCARD = false, typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void v28_shared_impl(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned char* __restrict__ bt_buf, const unsigned char* __restrict__ sf_buf,
    const float* __restrict__ s20, const float* __restrict__ s21,
    __nv_bfloat16* __restrict__ C, unsigned char* __restrict__ oq, unsigned char* __restrict__ os,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned N_in, unsigned K_in,
    const unsigned char* __restrict__ bt_gateup_dead = nullptr,
    unsigned int* __restrict__ done_counters = nullptr
) {
    using L = V28SharedSmem;
    constexpr unsigned N = Shape::template gemm_n<GATEUP>();
    constexpr unsigned K = Shape::template gemm_k<GATEUP>();
    static_assert((N % (GATEUP ? 128u : 256u)) == 0u, "v28_shared requires N divisible by TN");
    static_assert((K % 64u) == 0u, "v28_shared requires K divisible by 64");
    constexpr unsigned TN = GATEUP ? 128u : 256u;
    constexpr unsigned BCOLS = TN;
    unsigned dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (gridDim.x == 0 || blockDim.x != V14_THREADS || dyn_smem < L::BYTES) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }
    const bool shape = (N_in == N && K_in == K);
    const bool outs = GATEUP ? (oq != nullptr && os != nullptr) : (C != nullptr);
    if (!shape || !outs || aq == nullptr || as == nullptr || bt_buf == nullptr || sf_buf == nullptr ||
        expert_offsets == nullptr || sorted_token_ids == nullptr || !oi640::contract_ok(contract) ||
        !oi640::workspace_ok(workspace, contract) || status == nullptr || status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_ABI);
        return;
    }
    constexpr unsigned n_tiles = N / TN;
    const unsigned wc = workspace->work_count;
    const unsigned task_count = wc * n_tiles;
    if (blockIdx.x >= task_count) return;

    const unsigned tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    if constexpr (!GATEUP && DISCARD) {
        // Discard dead bt_gateup + sf_gateup (3.38 MB = 26,400 x 128 B lines) immediately after
        // shared_gateup completed on the stream (0 atomics, 0 races).
        if (bt_gateup_dead != nullptr &&
            ((reinterpret_cast<unsigned long long>(bt_gateup_dead) & 127ull) == 0ull)) {
            constexpr unsigned GU_LINES = (200u * 16384u + 200u * 512u) / 128u;
            const unsigned active_ctas = min(gridDim.x, task_count);
            const unsigned line_stride = active_ctas * V14_THREADS;
            for (unsigned l = blockIdx.x * V14_THREADS + tid; l < GU_LINES; l += line_stride) {
                v28_discard_l2_128(bt_gateup_dead + static_cast<unsigned long long>(l) * 128ull);
            }
        }
    }

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* const sA = smem;
    unsigned (*const sAS)[V10_TILE_M] = reinterpret_cast<unsigned (*)[V10_TILE_M]>(smem + L::OFF_AS);
    unsigned char* const sBt = smem + L::OFF_BT;
    unsigned char (*const sSF)[2][V14_BC] = reinterpret_cast<unsigned char (*)[2][V14_BC]>(smem + L::OFF_SF);
    __nv_bfloat16* const sE = reinterpret_cast<__nv_bfloat16*>(smem);

    unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n" : "+r"(sbase));
    unsigned long long pol_el;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));

    const unsigned warp_n = (warp_id & 3) * (GATEUP ? 32u : 64u);
    const unsigned gid = lane >> 2, tig = lane & 3;
    constexpr unsigned KS = K / 64;
    constexpr unsigned kblocks = K / 32;

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const unsigned wi = task / n_tiles;
        const oi640::WorkItem item = workspace->items[wi];
        const unsigned nt = task % n_tiles;
        const unsigned cta_n = nt * TN;
        if (item.expert >= Shape::EXPERTS) { if (tid == 0) oi640::fail(status, oi640::ERR_PLAN); return; }
        const int m_start = expert_offsets[item.expert];
        const int M_expert = expert_offsets[item.expert + 1] - m_start;
        const int local = static_cast<int>(item.m_tile * V10_TILE_M);
        const float s2a = s20[item.expert];
        const float s2b = GATEUP ? s21[item.expert] : 0.0f;
        if (m_start < 0 || M_expert < 0 || local >= M_expert || !isfinite(s2a) || !isfinite(s2b)) {
            if (tid == 0) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const unsigned cta_m = static_cast<unsigned>(m_start + local);
        const int valid_rows = min(M_expert - local, static_cast<int>(V10_TILE_M));
        const unsigned active_rows_pad16 = (static_cast<unsigned>(valid_rows) + 15u) & ~15u;
        const bool m64 = (valid_rows <= 64);
        const unsigned warp_m = (warp_id >> 2) * (m64 ? 16u : 32u);
        const int warp_valid = valid_rows - static_cast<int>(warp_m);
        const int rt_active = warp_valid <= 0 ? 0 : (m64 ? 1 : min(2, (warp_valid + 15) >> 4));

        const unsigned a_row = tid >> 2, a_ch = tid & 3;
        const bool a_ok = static_cast<int>(a_row) < valid_rows;
        const unsigned a_src_row = a_ok ? (cta_m + a_row) : 0u;
        const unsigned char* a_src = aq + static_cast<unsigned long long>(a_src_row) * K + a_ch * 16;
        const unsigned a_dst = v14_swz(a_row, a_ch);
        const unsigned sc_row = tid - 384;
        const bool sc_ok = tid >= 384 && static_cast<int>(sc_row) < valid_rows;
        const unsigned sc_src_row = sc_ok ? (cta_m + sc_row) : 0u;
        const unsigned char* sc_src = as + static_cast<unsigned long long>(sc_src_row) * kblocks;

        const unsigned char* bt_base = bt_buf + static_cast<unsigned long long>(nt * KS) * 16384ull;
        const unsigned char* sf_base = sf_buf + static_cast<unsigned long long>(nt * KS) * 512ull;
        const bool a_keep = GATEUP || (nt + 1u < n_tiles);
        const bool bt_reused = (local + static_cast<int>(V10_TILE_M) < M_expert);

        auto load_a = [&](unsigned ks) {
            const unsigned slot = ks & 3u;
            if (a_row < active_rows_pad16) {
                if (a_keep) {
                    v28_cp16_evict_last_pred(sA + slot * (V10_TILE_M * 64) + a_dst, a_src + ks * 64, a_ok);
                } else {
                    v28_cp16_evict_first_pred(sA + slot * (V10_TILE_M * 64) + a_dst, a_src + ks * 64, a_ok);
                }
            }
            if (tid >= 384 && (ks & 1u) == 0u && sc_row < active_rows_pad16) {
                if (a_keep) {
                    v28_cp4_evict_last_pred(&sAS[(ks >> 1) & 1][sc_row], sc_src + ks * 2, sc_ok);
                } else {
                    v28_cp4_evict_first_pred(&sAS[(ks >> 1) & 1][sc_row], sc_src + ks * 2, sc_ok);
                }
            }
        };
        auto load_bt = [&](unsigned ks) {
            const unsigned slot = ks & 3u;
            const unsigned char* b_src = bt_base + static_cast<unsigned long long>(ks) * 16384ull;
            const unsigned b_dst_s = sbase + L::OFF_BT + slot * 16384u;
            (void)bt_reused;
            v28_cp16_l2_el_s32(b_dst_s + tid * 16u, b_src + tid * 16u, pol_el);
            v28_cp16_l2_el_s32(b_dst_s + (tid + 512u) * 16u, b_src + (tid + 512u) * 16u, pol_el);
            if (tid < 32u) {
                v28_cp16_l2_el_s32(sbase + L::OFF_SF + slot * 512u + tid * 16u, sf_base + ks * 512u + tid * 16u, pol_el);
            }
        };

        float acc[2][8][4];
        #pragma unroll
        for (int r = 0; r < 2; ++r)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.0f;

        #pragma unroll
        for (unsigned s = 0; s < 3u; ++s) {
            if (s < KS) {
                load_a(s);
                load_bt(s);
            }
            v10_commit();
        }

        unsigned sfa[2] = {0u, 0u};
        auto step = [&](auto par_tag, unsigned ks) {
            constexpr unsigned PAR = decltype(par_tag)::value;   // ks & 1
            v10_wait<2>();
            __syncthreads();
            if (ks + 3u < KS) {
                load_a(ks + 3u);
                load_bt(ks + 3u);
            }
            v10_commit();
            if (rt_active == 0) return;
            if constexpr (PAR == 0) {
                const unsigned as_buf = (ks >> 1) & 1;
                sfa[0] = sAS[as_buf][warp_m + gid + (tig & 1) * 8];
                if (rt_active > 1) sfa[1] = sAS[as_buf][warp_m + 16 + gid + (tig & 1) * 8];
            }
            const unsigned slot = ks & 3u;
            #pragma unroll
            for (unsigned t = 0; t < 2; ++t) {
                const uint2 sfb = *reinterpret_cast<const uint2*>(&sSF[slot][t][((warp_id & 3) * 8 + gid) << 3]);
                const unsigned row_a0 = warp_m + (lane & 15);
                const unsigned addr_a0 = static_cast<unsigned>(__cvta_generic_to_shared(
                    sA + slot * (V10_TILE_M * 64) + v14_swz(row_a0, 2 * t + (lane >> 4))));
                const unsigned row_b0 = warp_n + ((lane >> 4) << 3) + (lane & 7);
                const unsigned addr_b0 = static_cast<unsigned>(__cvta_generic_to_shared(
                    sBt + slot * (V14_BC * 64) + v14_swz(row_b0, 2 * t + ((lane >> 3) & 1))));
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
                // Ping-pong two 4-reg B buffers to overlap ldsm4(p+1) with QMMAs on p (0-ULP).
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
        } else {
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
                    const unsigned col8 = warp_n + b * 32u + tig * 8u;
                    const uint4 w0 = v28_bf16_reg_quad(u0[0], u0[1], u0[2], u0[3], tig);
                    if (static_cast<int>(lr0) < valid_rows) {
                        v28_st16_stream(
                            C + static_cast<unsigned long long>(cta_m + lr0) * N + cta_n + col8, w0);
                    }
                    const uint4 w1 = v28_bf16_reg_quad(u1[0], u1[1], u1[2], u1[3], tig);
                    if (static_cast<int>(lr1) < valid_rows) {
                        v28_st16_stream(
                            C + static_cast<unsigned long long>(cta_m + lr1) * N + cta_n + col8, w1);
                    }
                }
            }
            if constexpr (DISCARD) {
                // Race-free per-item aq/as_ (10.81 MB) and per-nt10 bt_down/sf_down (1.69 MB) L2 discard:
                // acq_rel GPU-scope atomics establish happens-before from all prior readers to the unique last CTA.
                if (done_counters != nullptr) {
                    __syncthreads();
                    unsigned* const s_flags = reinterpret_cast<unsigned*>(smem);
                    if (tid == 0u) {
                        unsigned disc_aq = 0u, disc_bt = 0u;
                        if (wi < 512u) {
                            unsigned prev_wi;
                            asm volatile("atom.global.acq_rel.gpu.add.u32 %0, [%1], 1;\n"
                                         : "=r"(prev_wi) : "l"(done_counters + 32u + wi) : "memory");
                            if (prev_wi + 1u == n_tiles) {
                                asm volatile("st.global.release.gpu.u32 [%0], 0;\n"
                                             :: "l"(done_counters + 32u + wi) : "memory");
                                disc_aq = 1u;
                            }
                        }
                        {
                            unsigned prev_nt;
                            asm volatile("atom.global.acq_rel.gpu.add.u32 %0, [%1], 1;\n"
                                         : "=r"(prev_nt) : "l"(done_counters + nt) : "memory");
                            if (prev_nt + 1u == wc) {
                                asm volatile("st.global.release.gpu.u32 [%0], 0;\n"
                                             :: "l"(done_counters + nt) : "memory");
                                disc_bt = 1u;
                            }
                        }
                        s_flags[0] = disc_aq;
                        s_flags[1] = disc_bt;
                    }
                    __syncthreads();
                    if (s_flags[0] != 0u) {
                        if ((reinterpret_cast<unsigned long long>(aq) & 127ull) == 0ull) {
                            const unsigned aq_lines = static_cast<unsigned>(valid_rows) * 5u; // 640 B/row = 5 x 128 B
                            const unsigned char* const aq_item = aq + static_cast<unsigned long long>(cta_m) * K;
                            for (unsigned l = tid; l < aq_lines; l += V14_THREADS) {
                                v28_discard_l2_128(aq_item + static_cast<unsigned long long>(l) * 128ull);
                            }
                        }
                        const unsigned long long as_beg =
                            reinterpret_cast<unsigned long long>(as) + static_cast<unsigned long long>(cta_m) * kblocks;
                        const unsigned long long as_end =
                            as_beg + static_cast<unsigned long long>(valid_rows) * kblocks;
                        const unsigned long long as_lo = (as_beg + 127ull) & ~127ull;
                        const unsigned long long as_hi = as_end & ~127ull;
                        if (as_lo + static_cast<unsigned long long>(tid) * 128ull < as_hi) {
                            v28_discard_l2_128(reinterpret_cast<const void*>(
                                as_lo + static_cast<unsigned long long>(tid) * 128ull));
                        }
                    }
                    if (s_flags[1] != 0u) {
                        if ((reinterpret_cast<unsigned long long>(bt_buf) & 127ull) == 0ull) {
                            // 10 ks slices x 16,384 B (128 x 128 B lines) = 1,280 contiguous lines per nt10 tile.
                            for (unsigned l = tid; l < 1280u; l += V14_THREADS) {
                                v28_discard_l2_128(bt_base + static_cast<unsigned long long>(l) * 128ull);
                            }
                        }
                        if ((reinterpret_cast<unsigned long long>(sf_buf) & 127ull) == 0ull && tid < 40u) {
                            v28_discard_l2_128(sf_base + static_cast<unsigned long long>(tid) * 128ull);
                        }
                    }
                }
            }
        }
        if (task + gridDim.x < task_count) __syncthreads();
    }
}

template <typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void v28_shared_pretranscode_impl(
    const unsigned long long* __restrict__ tables,
    unsigned char* __restrict__ bt_gateup, unsigned char* __restrict__ sf_gateup,
    unsigned char* __restrict__ bt_down, unsigned char* __restrict__ sf_down
) {
    static_assert((Shape::HIDDEN % 256u) == 0u && (Shape::INTER % 128u) == 0u,
                  "v28_shared_pretranscode requires HIDDEN % 256 == 0 and INTER % 128 == 0");
    constexpr unsigned GU_KS = Shape::HIDDEN / 64u;
    constexpr unsigned GU_NT = Shape::INTER / 128u;
    constexpr unsigned GU_SLICES = GU_NT * GU_KS;
    constexpr unsigned DN_KS = Shape::INTER / 64u;
    constexpr unsigned DN_NT = Shape::HIDDEN / 256u;
    constexpr unsigned DN_SLICES = DN_NT * DN_KS;
    constexpr unsigned TOTAL_SLICES = GU_SLICES + DN_SLICES;

    __shared__ __align__(16) unsigned char sB[2][32][256];
    __shared__ __align__(16) unsigned char sS[2][4][256];
    __shared__ __align__(16) unsigned char sBt[V14_BC * 64];
    __shared__ __align__(16) unsigned char sSF[2][V14_BC];
    __shared__ __align__(16) uint2 lut[256];

    const unsigned tid = threadIdx.x;
    if (tid < 256) {
        lut[tid] = V28_E4M3_LUT[tid];
    }

    const unsigned char* const B0_gu = reinterpret_cast<const unsigned char*>(tables[0]);
    const unsigned char* const S0_gu = reinterpret_cast<const unsigned char*>(tables[1]);
    const unsigned char* const B1_gu = reinterpret_cast<const unsigned char*>(tables[2]);
    const unsigned char* const S1_gu = reinterpret_cast<const unsigned char*>(tables[3]);
    const unsigned char* const B0_dn = reinterpret_cast<const unsigned char*>(tables[4]);
    const unsigned char* const S0_dn = reinterpret_cast<const unsigned char*>(tables[5]);

    for (unsigned blk = blockIdx.x; blk < TOTAL_SLICES; blk += gridDim.x) {
        if (blk < GU_SLICES) {
            const unsigned nt = blk / GU_KS, ks = blk % GU_KS;
            v28_pretranscode_slice<true>(
                B0_gu, S0_gu, B1_gu, S1_gu, Shape::INTER, nt * 128u, ks,
                bt_gateup + static_cast<unsigned long long>(blk) * 16384ull,
                sf_gateup + static_cast<unsigned long long>(blk) * 512ull,
                sB, sS, sBt, sSF, lut);
        } else {
            const unsigned slice = blk - GU_SLICES;
            const unsigned nt = slice / DN_KS, ks = slice % DN_KS;
            v28_pretranscode_slice<false>(
                B0_dn, S0_dn, B0_dn, S0_dn, Shape::HIDDEN, nt * 256u, ks,
                bt_down + static_cast<unsigned long long>(slice) * 16384ull,
                sf_down + static_cast<unsigned long long>(slice) * 512ull,
                sB, sS, sBt, sSF, lut);
        }
        if (blk + gridDim.x < TOTAL_SLICES) __syncthreads();
    }
}

} // namespace

// Slices 0..199: gate+up 5x40 slices; slices 200..299: down 10x10 slices. Block 512, 2 CTAs/SM.
extern "C" __global__ __launch_bounds__(512, 2) void qwen4_shared_pretranscode_v28(
    const unsigned long long* __restrict__ tables,
    unsigned char* __restrict__ bt_gateup, unsigned char* __restrict__ sf_gateup,
    unsigned char* __restrict__ bt_down, unsigned char* __restrict__ sf_down
) {
    v28_shared_pretranscode_impl<Qwen4FlashNextShape>(tables, bt_gateup, sf_gateup, bt_down, sf_down);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_shared_w4a8_gateup_k32_v28(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned char* __restrict__ bt_gateup, const unsigned char* __restrict__ sf_gateup,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v28_shared_impl<true>(aq, as, bt_gateup, sf_gateup, gate_scale2_vals, up_scale2_vals,
                          nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
                          status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_shared_w4a8_gemm_k32_v28(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned char* __restrict__ bt_down, const unsigned char* __restrict__ sf_down,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_shared_impl<false>(aq, as, bt_down, sf_down, scale2_vals, nullptr, C, nullptr, nullptr,
                           expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_shared_w4a8_gemm_k32_v28_discard(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned char* __restrict__ bt_down, const unsigned char* __restrict__ sf_down,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K,
    const unsigned char* __restrict__ bt_gateup_dead, unsigned int* __restrict__ done_counters
) {
    v28_shared_impl<false, true>(aq, as, bt_down, sf_down, scale2_vals, nullptr, C, nullptr, nullptr,
                                 expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
                                 bt_gateup_dead, done_counters);
}

namespace {

// 2-stage <= 49 KiB SMEM (34,304 B = 33.5 KiB) shared_down kernel (256 threads = 4 M x 2 N,
// 2 CTAs/SM on GB10, <= 128 regs/thread) over pre-transcoded bt_down/sf_down slices, with
// ping-pong bfr[2][4] ldsm4/QMMA overlap and race-free L2 discard of dead bt_gateup/sf,
// bt_down/sf, and aq/as_ scratch lines. Strictly 0-ULP bit-identical to v28_shared_impl<false>.
struct V28SharedDown2StageSmem {
    static constexpr unsigned STAGES = 2u;
    static constexpr unsigned BC = 128u;
    static constexpr unsigned A = STAGES * V10_TILE_M * 64u;       // 16,384 B
    static constexpr unsigned BT = STAGES * BC * 64u;              // 16,384 B
    static constexpr unsigned SF = STAGES * 2u * BC;               //    512 B
    static constexpr unsigned AS = 2u * V10_TILE_M * 4u;           //  1,024 B
    static constexpr unsigned OFF_BT = A;
    static constexpr unsigned OFF_SF = OFF_BT + BT;
    static constexpr unsigned OFF_AS = OFF_SF + SF;
    static constexpr unsigned BYTES = OFF_AS + AS;
    static_assert(BYTES == 34304u, "2-stage shared_down SMEM byte contract");
    static_assert(BYTES <= 49152u, "2-stage shared_down fits 2 CTAs/SM (<= 48 KiB)");
};

template <typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void v28_shared_down_2stage_impl(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned char* __restrict__ bt_buf, const unsigned char* __restrict__ sf_buf,
    const float* __restrict__ s20, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned N_in, unsigned K_in,
    const unsigned char* __restrict__ bt_gateup_dead, unsigned int* __restrict__ done_counters
) {
    using L = V28SharedDown2StageSmem;
    constexpr unsigned N = Shape::HIDDEN;
    constexpr unsigned K = Shape::INTER;
    constexpr unsigned DN_KS = K / 64u;
    constexpr unsigned TN = 128u;
    constexpr unsigned n_tiles = N / TN; // 20 (each 256-col pretranscode tile splits into 2x128)
    constexpr unsigned KS = K / 64u;     // 10
    constexpr unsigned kblocks = K / 32u;
    unsigned dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (gridDim.x == 0 || blockDim.x != 256u || dyn_smem < L::BYTES) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }
    if (N_in != N || K_in != K || C == nullptr || aq == nullptr || as == nullptr ||
        bt_buf == nullptr || sf_buf == nullptr || expert_offsets == nullptr ||
        sorted_token_ids == nullptr || !oi640::contract_ok(contract) ||
        !oi640::workspace_ok(workspace, contract) || status == nullptr || status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_ABI);
        return;
    }
    const unsigned wc = workspace->work_count;
    const unsigned task_count = wc * n_tiles;
    if (blockIdx.x >= task_count) return;

    const unsigned tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31u;
    // Discard dead bt_gateup + sf_gateup (3.38 MB = 26,400 x 128 B lines) immediately after
    // shared_gateup completed on the stream (0 atomics, 0 races, 1 instruction/thread at 16K).
    if (bt_gateup_dead != nullptr &&
        ((reinterpret_cast<unsigned long long>(bt_gateup_dead) & 127ull) == 0ull)) {
        constexpr unsigned GU_LINES = (200u * 16384u + 200u * 512u) / 128u;
        const unsigned active_ctas = min(gridDim.x, task_count);
        const unsigned line_stride = active_ctas * 256u;
        for (unsigned l = blockIdx.x * 256u + tid; l < GU_LINES; l += line_stride) {
            v28_discard_l2_128(bt_gateup_dead + static_cast<unsigned long long>(l) * 128ull);
        }
    }

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* const sA = smem;
    unsigned char* const sBt = smem + L::OFF_BT;
    unsigned char (*const sSF)[2][L::BC] = reinterpret_cast<unsigned char (*)[2][L::BC]>(smem + L::OFF_SF);
    unsigned (*const sAS)[V10_TILE_M] = reinterpret_cast<unsigned (*)[V10_TILE_M]>(smem + L::OFF_AS);

    unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n" : "+r"(sbase));

    const unsigned warp_n = (warp_id & 1u) * 64u;
    const unsigned gid = lane >> 2, tig = lane & 3u;

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const unsigned wi = task / n_tiles;
        const oi640::WorkItem item = workspace->items[wi];
        const unsigned nt20 = task % n_tiles;
        const unsigned nt10 = nt20 >> 1;
        const unsigned nhalf = nt20 & 1u;
        if (item.expert >= Shape::EXPERTS) { if (tid == 0) oi640::fail(status, oi640::ERR_PLAN); return; }
        const int m_start = expert_offsets[item.expert];
        const int M_expert = expert_offsets[item.expert + 1] - m_start;
        const int local = static_cast<int>(item.m_tile * V10_TILE_M);
        const float s2a = s20[item.expert];
        if (m_start < 0 || M_expert < 0 || local >= M_expert || !isfinite(s2a)) {
            if (tid == 0) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const unsigned cta_m = static_cast<unsigned>(m_start + local);
        const int valid_rows = min(M_expert - local, static_cast<int>(V10_TILE_M));
        const unsigned wi_nt_vr = (wi << 13) | (static_cast<unsigned>(valid_rows) << 5) | nt20;
        const unsigned active_rows_pad16 = (static_cast<unsigned>(valid_rows) + 15u) & ~15u;
        const bool m64 = (valid_rows <= 64);
        const unsigned warp_m = (warp_id >> 1) * (m64 ? 16u : 32u);
        const int warp_valid = valid_rows - static_cast<int>(warp_m);
        const int rt_active = warp_valid <= 0 ? 0 : (m64 ? 1 : min(2, (warp_valid + 15) >> 4));

        const unsigned a_r0 = tid >> 2, a_ch = tid & 3u;
        const unsigned a_r1 = a_r0 + 64u;
        const bool a_ok0 = static_cast<int>(a_r0) < valid_rows;
        const bool a_ok1 = static_cast<int>(a_r1) < valid_rows;
        const unsigned char* a_src0 = aq + static_cast<unsigned long long>(a_ok0 ? (cta_m + a_r0) : 0u) * K + a_ch * 16u;
        const unsigned sc_row_init = tid - 128u;
        const bool sc_ok = tid >= 128u && static_cast<int>(sc_row_init) < valid_rows;
        const unsigned a_dst_flags =
            v14_swz(a_r0, a_ch) |
            (((a_r0 < active_rows_pad16) ? 1u : 0u) << 16) |
            ((a_ok0 ? 2u : 0u) << 16) |
            (((a_r1 < active_rows_pad16) ? 4u : 0u) << 16) |
            ((a_ok1 ? 8u : 0u) << 16) |
            (((tid >= 128u && sc_row_init < active_rows_pad16) ? 16u : 0u) << 16) |
            ((sc_ok ? 32u : 0u) << 16);

        const unsigned char* bt_base = bt_buf + static_cast<unsigned long long>(nt10 * KS) * 16384ull + nhalf * 8192u;
        const unsigned char* sf_base = sf_buf + static_cast<unsigned long long>(nt10 * KS) * 512ull + nhalf * 128u;

        // Keep load_a on plain cp.async with packed metadata and derived a_src1/sc_src,
        // and materialize pol_el inside load_bt so the 128-reg __launch_bounds__(256, 2)
        // kernel stays at 0 stack / 0 spills while load_bt retains L2::evict_last residency.
        auto load_a = [&](unsigned ks) {
            const unsigned slot = ks & 1u;
            const unsigned a_dst0 = a_dst_flags & 0xffffu;
            const unsigned ld_flags = a_dst_flags >> 16;
            unsigned char* const sA_slot = sA + slot * (V10_TILE_M * 64u);
            if (ld_flags & 1u) {
                v10_cp16(sA_slot + a_dst0, a_src0 + ks * 64u, (ld_flags & 2u) != 0u);
            }
            if (ld_flags & 4u) {
                const bool ok1 = (ld_flags & 8u) != 0u;
                v10_cp16(sA_slot + a_dst0 + 4096u,
                         ok1 ? (a_src0 + 64u * K + ks * 64u) : a_src0, ok1);
            }
            if ((ld_flags & 16u) && (ks & 1u) == 0u) {
                const unsigned sc_r = tid - 128u;
                const bool sc_ok_i = (ld_flags & 32u) != 0u;
                v10_cp4(&sAS[(ks >> 1) & 1u][sc_r],
                        as + static_cast<unsigned long long>(sc_ok_i ? (cta_m + sc_r) : 0u) * kblocks + ks * 2u,
                        sc_ok_i);
            }
        };
        auto load_bt = [&](unsigned ks) {
            const unsigned slot = ks & 1u;
            unsigned long long pol_el;
            asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));
            const unsigned char* b_src = bt_base + static_cast<unsigned long long>(ks) * 16384ull;
            const unsigned b_dst_s = sbase + L::OFF_BT + slot * 8192u;
            v28_cp16_l2_el_s32(b_dst_s + tid * 16u, b_src + tid * 16u, pol_el);
            v28_cp16_l2_el_s32(b_dst_s + (tid + 256u) * 16u, b_src + (tid + 256u) * 16u, pol_el);
            if (tid < 16u) {
                const unsigned t_sf = tid >> 3, c_sf = tid & 7u;
                v28_cp16_l2_el_s32(sbase + L::OFF_SF + (slot * 2u + t_sf) * 128u + c_sf * 16u,
                                   sf_base + ks * 512u + t_sf * 256u + c_sf * 16u, pol_el);
            }
        };

        float acc[2][8][4];
        #pragma unroll
        for (int r = 0; r < 2; ++r)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.0f;

        load_a(0u);
        load_bt(0u);
        v10_commit();

        unsigned sfa[2] = {0u, 0u};
        auto step = [&](auto par_tag, unsigned ks) {
            constexpr unsigned PAR = decltype(par_tag)::value;   // ks & 1 == slot
            v10_wait<0>();
            __syncthreads();
            if (ks + 1u < KS) {
                load_a(ks + 1u);
                load_bt(ks + 1u);
            }
            v10_commit();
            if (rt_active == 0) return;
            if constexpr (PAR == 0u) {
                const unsigned as_buf = (ks >> 1) & 1u;
                sfa[0] = sAS[as_buf][warp_m + gid + (tig & 1u) * 8u];
                if (rt_active > 1) sfa[1] = sAS[as_buf][warp_m + 16u + gid + (tig & 1u) * 8u];
            }
            #pragma unroll
            for (unsigned t = 0u; t < 2u; ++t) {
                const uint2 sfb = *reinterpret_cast<const uint2*>(&sSF[PAR][t][((warp_id & 1u) * 8u + gid) << 3]);
                const unsigned row_a0 = warp_m + (lane & 15u);
                const unsigned addr_a0 = static_cast<unsigned>(__cvta_generic_to_shared(
                    sA + PAR * (V10_TILE_M * 64u) + v14_swz(row_a0, 2u * t + (lane >> 4))));
                const unsigned row_b0 = warp_n + ((lane >> 4) << 3) + (lane & 7u);
                const unsigned addr_b0 = static_cast<unsigned>(__cvta_generic_to_shared(
                    sBt + PAR * (L::BC * 64u) + v14_swz(row_b0, 2u * t + ((lane >> 3) & 1u))));
                unsigned afr0[4], afr1[4];
                auto emit_pair = [&](auto p_tag, const unsigned (&bf)[4], unsigned sfb_w) {
                    constexpr int P = decltype(p_tag)::value;
                    constexpr int BIDB0 = (P & 1) * 2;
                    constexpr int BIDB1 = BIDB0 + 1;
                    if (t == 0u) {
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
                v10_ldsm4(addr_b0, bfr[0][0], bfr[0][1], bfr[0][2], bfr[0][3]);
                v10_ldsm4(addr_a0, afr0[0], afr0[1], afr0[2], afr0[3]);
                if (rt_active > 1) {
                    v10_ldsm4(addr_a0 + 1024u, afr1[0], afr1[1], afr1[2], afr1[3]);
                }
                v10_ldsm4(addr_b0 + 1024u, bfr[1][0], bfr[1][1], bfr[1][2], bfr[1][3]);
                emit_pair(std::integral_constant<int, 0>{}, bfr[0], sfb.x);
                v10_ldsm4(addr_b0 + 2048u, bfr[0][0], bfr[0][1], bfr[0][2], bfr[0][3]);
                emit_pair(std::integral_constant<int, 1>{}, bfr[1], sfb.x);
                v10_ldsm4(addr_b0 + 3072u, bfr[1][0], bfr[1][1], bfr[1][2], bfr[1][3]);
                emit_pair(std::integral_constant<int, 2>{}, bfr[0], sfb.y);
                emit_pair(std::integral_constant<int, 3>{}, bfr[1], sfb.y);
            }
        };
        #pragma unroll 1
        for (unsigned ks = 0u; ks < KS; ks += 2u) {
            step(std::integral_constant<unsigned, 0u>{}, ks);
            step(std::integral_constant<unsigned, 1u>{}, ks + 1u);
        }
        v10_wait<0>();

        const unsigned wi_e = wi_nt_vr >> 13;
        const int valid_rows_e = static_cast<int>((wi_nt_vr >> 5) & 255u);
        const unsigned nt20_e = wi_nt_vr & 31u;
        const unsigned cta_n = nt20_e * TN;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (r >= rt_active) break;
            const unsigned lr0 = warp_m + r * 16u + gid, lr1 = lr0 + 8u;
            #pragma unroll
            for (unsigned b = 0u; b < 2u; ++b) {
                unsigned u0[4], u1[4];
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const int j = 4 * static_cast<int>(b) + jj;
                    u0[jj] = v28_pack_bf162(acc[r][j][0] * s2a, acc[r][j][1] * s2a);
                    u1[jj] = v28_pack_bf162(acc[r][j][2] * s2a, acc[r][j][3] * s2a);
                }
                const unsigned col8 = warp_n + b * 32u + tig * 8u;
                const uint4 w0 = v28_bf16_reg_quad(u0[0], u0[1], u0[2], u0[3], tig);
                if (static_cast<int>(lr0) < valid_rows_e) {
                    v28_st16_stream(
                        C + static_cast<unsigned long long>(cta_m + lr0) * N + cta_n + col8, w0);
                }
                const uint4 w1 = v28_bf16_reg_quad(u1[0], u1[1], u1[2], u1[3], tig);
                if (static_cast<int>(lr1) < valid_rows_e) {
                    v28_st16_stream(
                        C + static_cast<unsigned long long>(cta_m + lr1) * N + cta_n + col8, w1);
                }
            }
        }

        // Race-free per-item aq/as_ (10.81 MB) and per-nt20 bt_down/sf_down (1.69 MB) L2 discard:
        // acq_rel GPU-scope atomics establish happens-before from all prior readers to the unique last CTA.
        if (done_counters != nullptr) {
            __syncthreads();
            unsigned* const s_flags = reinterpret_cast<unsigned*>(smem);
            if (tid == 0u) {
                unsigned disc_aq = 0u, disc_bt = 0u;
                if (wi_e < 512u) {
                    unsigned prev_wi;
                    asm volatile("atom.global.acq_rel.gpu.add.u32 %0, [%1], 1;\n"
                                 : "=r"(prev_wi) : "l"(done_counters + 32u + wi_e) : "memory");
                    if (prev_wi + 1u == n_tiles) {
                        asm volatile("st.global.release.gpu.u32 [%0], 0;\n"
                                     :: "l"(done_counters + 32u + wi_e) : "memory");
                        disc_aq = 1u;
                    }
                }
                {
                    unsigned prev_nt;
                    asm volatile("atom.global.acq_rel.gpu.add.u32 %0, [%1], 1;\n"
                                 : "=r"(prev_nt) : "l"(done_counters + nt20_e) : "memory");
                    if (prev_nt + 1u == wc) {
                        asm volatile("st.global.release.gpu.u32 [%0], 0;\n"
                                     :: "l"(done_counters + nt20_e) : "memory");
                        disc_bt = 1u;
                    }
                }
                s_flags[0] = disc_aq;
                s_flags[1] = disc_bt;
            }
            __syncthreads();
            if (s_flags[0] != 0u) {
                if ((reinterpret_cast<unsigned long long>(aq) & 127ull) == 0ull) {
                    const unsigned aq_lines = static_cast<unsigned>(valid_rows_e) * 5u; // 640 B/row = 5 x 128 B
                    const unsigned char* const aq_item = aq + static_cast<unsigned long long>(cta_m) * K;
                    for (unsigned l = tid; l < aq_lines; l += 256u) {
                        v28_discard_l2_128(aq_item + static_cast<unsigned long long>(l) * 128ull);
                    }
                }
                const unsigned long long as_beg =
                    reinterpret_cast<unsigned long long>(as) + static_cast<unsigned long long>(cta_m) * kblocks;
                const unsigned long long as_end =
                    as_beg + static_cast<unsigned long long>(valid_rows_e) * kblocks;
                const unsigned long long as_lo = (as_beg + 127ull) & ~127ull;
                const unsigned long long as_hi = as_end & ~127ull;
                if (as_lo + static_cast<unsigned long long>(tid) * 128ull < as_hi) {
                    v28_discard_l2_128(reinterpret_cast<const void*>(
                        as_lo + static_cast<unsigned long long>(tid) * 128ull));
                }
            }
            if (s_flags[1] != 0u) {
                if ((reinterpret_cast<unsigned long long>(bt_buf) & 127ull) == 0ull) {
                    // DN_KS ks slices x 8,192 B (64 x 128 B lines) = DN_KS * 64 lines per nt20 half.
                    for (unsigned i = tid; i < DN_KS * 64u; i += 256u) {
                        const unsigned ks_d = i >> 6, ln = i & 63u;
                        v28_discard_l2_128(bt_base + static_cast<unsigned long long>(ks_d) * 16384ull + ln * 128u);
                    }
                }
                if ((reinterpret_cast<unsigned long long>(sf_buf) & 127ull) == 0ull && tid < DN_KS * 2u) {
                    const unsigned ks_d = tid >> 1, t_d = tid & 1u;
                    v28_discard_l2_128(sf_base + ks_d * 512u + t_d * 256u);
                }
            }
        }
        if (task + gridDim.x < task_count) __syncthreads();
    }
}

} // namespace

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_shared_w4a8_gemm_k32_v28_2stage(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned char* __restrict__ bt_down, const unsigned char* __restrict__ sf_down,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K,
    const unsigned char* __restrict__ bt_gateup_dead, unsigned int* __restrict__ done_counters
) {
    v28_shared_down_2stage_impl(aq, as, bt_down, sf_down, scale2_vals, C,
                                expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
                                bt_gateup_dead, done_counters);
}

