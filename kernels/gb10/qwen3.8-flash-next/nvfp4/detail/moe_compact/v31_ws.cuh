// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// V31 warp-specialized W4A8 MoE prefill kernels (TILE_M=128, BC=256, flat and TS4E-128).

// =========================================================================
// Wave-22: 0-ULP V31 Warp-Specialized MoE Prefill (TILE_M = 128, BC = 256)
// Keeps v28_twarp's full TN = 128 (GATEUP, n_tiles=5) / TN = 256 (DOWN, n_tiles=10)
// geometry (0 extra xq/oq gathers) with 512 threads (16 warps = 4 WGs, 1 CTA/SM):
//   - WG0+WG1 (Warps 0..7, 256 threads, setmaxnreg.dec 72 -> 18,432 regs): Producer + Transcode WGs
//     (DA=3, DB=3, wait<1>, 4-warp parallel TS4E decode, 1-col/thread conflict-free SWAR transcode
//      with STS.128, cross-task L2 prefetch — 100% hidden behind Consumer MMA).
//   - WG2+WG3 (Warps 8..15, 256 threads, 4 M x 2 N, setmaxnreg.inc 184 -> 47,104 regs): Consumer MMA WGs
//     (18,432 + 47,104 = 65,536 regs/SM exact; 128 FP32 acc regs + 56 non-acc regs, 0 B spills;
//      64 m16n8k32 QMMAs/warp/step, 1x LDS.128 sSF, 0-SMEM v28_mxfp8_reg_quad / v28_bf16_reg_quad
//      epilogue overlapped with Producer's next task prologue).
// Dynamic SMEM: 87,616 B (Flat GU) / 87,104 B (Flat DN) / 91,904 B (TS4E GU) / 90,672 B (TS4E DN)
// <= 101,376 B (99 KiB GB10 cap; 2,048 B LUT eliminated via 0-ULP hardware FP16->E4M3 PTX).
// =========================================================================
namespace {

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1030)) || defined(V31_WS_FORCE_WORKSTEAL_TEST)
#define V31_WS_USE_WORKSTEAL 1
#else
#define V31_WS_USE_WORKSTEAL 0
#endif

constexpr unsigned V31_WS_TILE_M = V10_TILE_M; // 128
constexpr unsigned V31_WS_BC = V14_BC;         // 256
constexpr unsigned V31_WS_THREADS = 512u;      // 16 warps = 4 WGs (WG0+WG1 prod, WG2+WG3 cons)

template <bool GATEUP, bool TS4E, typename Shape = Qwen4FlashNextShape> struct V31WsSmem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V31_WS_BC / BMAT; // 128 (GU) or 256 (DN)
    static constexpr unsigned S_ROWS = 4u;
    static constexpr unsigned TS4E_KS = Shape::template gemm_k<GATEUP>() / 64u;
    static constexpr unsigned TS4E_BKTS = TS4E_KS * 4u;
    static constexpr unsigned TS4E_EXC_CAP = 128u;
    static constexpr unsigned A = V14_DA * V31_WS_TILE_M * 64u;     // 24,576 B
    static constexpr unsigned B = V14_DB * BMAT * 32u * BCOLS;      // 24,576 B
    static constexpr unsigned S = V14_DB * BMAT * S_ROWS * BCOLS;   // 3,072 B
    static constexpr unsigned AS = 2u * V31_WS_TILE_M * 4u;         // 1,024 B
    static constexpr unsigned BT = 2u * V31_WS_BC * 64u;            // 32,768 B
    static constexpr unsigned SF = 2u * 2u * V31_WS_BC;             // 1,024 B
    static constexpr unsigned AOFF = GATEUP ? (V31_WS_TILE_M * 4u) : 0u; // 512 B (GU) / 0 B (DN)
    static constexpr unsigned PTRS = 8u * 8u;                       // 64 B (0..31 sPtrs, 32..39 sTask[2])
    static constexpr unsigned OFF_B = A;
    static constexpr unsigned OFF_S = OFF_B + B;
    static constexpr unsigned OFF_AS = OFF_S + S;
    static constexpr unsigned OFF_BT = OFF_AS + AS;
    static constexpr unsigned OFF_SF = OFF_BT + BT;
    static constexpr unsigned OFF_AOFF = OFF_SF + SF;
    static constexpr unsigned OFF_PTRS = OFF_AOFF + AOFF;
#if V31_WS_USE_WORKSTEAL
    static constexpr unsigned OFF_TASK = OFF_PTRS + 32u;
#endif
    static constexpr unsigned OFF_BASE = OFF_PTRS + PTRS;
    static constexpr unsigned OFF_EXC = OFF_BASE + (TS4E ? 256u : 0u);
    static constexpr unsigned OFF_BKT_BEG = OFF_EXC + (TS4E ? (V14_DB * 4u * TS4E_EXC_CAP * 2u) : 0u);
    static constexpr unsigned OFF_BKT_CNT = OFF_BKT_BEG + (TS4E ? (TS4E_BKTS * 4u) : 0u);
    static constexpr unsigned RAW_BYTES = OFF_BKT_CNT + (TS4E ? (TS4E_BKTS * 2u) : 0u);
    static constexpr unsigned BYTES = (RAW_BYTES + 15u) & ~15u;
    static_assert(
        !TS4E ? (GATEUP ? (BYTES == 87616u) : (BYTES == 87104u))
              : (GATEUP ? (BYTES == 91904u) : (BYTES == 90672u)),
        "V31WsSmem byte contract"
    );
    static_assert(BYTES <= 101376u, "V31WsSmem fits GB10 99 KiB dynamic SMEM cap");
};

// Zero-SMEM, zero-bank-conflict hardware PTX replacement for V28_E4M3_LUT[r_byte]:
// computes e4m3_rn_satfinite({0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} * e4m3(r_byte))
// bit-for-bit across all 256 uint8 codes (0 diffs across all 4,096 (r, nibble) combos).
__device__ __forceinline__ uint2 v31_e4m3_lut_ptx(unsigned r_byte) {
    const unsigned short r_pair =
        static_cast<unsigned short>((r_byte & 0xffu) | ((r_byte & 0xffu) << 8));
    unsigned h_r, m0, m1, m2, m3;
    unsigned short p0, p1, p2, p3;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(h_r) : "h"(r_pair));
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(m0) : "r"(h_r), "r"(0x38000000u));
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(m1) : "r"(h_r), "r"(0x3e003c00u));
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(m2) : "r"(h_r), "r"(0x42004000u));
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(m3) : "r"(h_r), "r"(0x46004400u));
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(p0) : "r"(m0));
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(p1) : "r"(m1));
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(p2) : "r"(m2));
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(p3) : "r"(m3));
    return make_uint2(
        v10_prmt(static_cast<unsigned>(p0), static_cast<unsigned>(p1), 0x5410u),
        v10_prmt(static_cast<unsigned>(p2), static_cast<unsigned>(p3), 0x5410u));
}

#if V31_WS_USE_WORKSTEAL
__device__ __forceinline__ void v31_st_su32_vol(unsigned s_addr, unsigned v) {
    asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(s_addr), "r"(v) : "memory");
}
#endif

// Bijective packed 16-byte per-(cwn, gid) scale index for 4 M x 2 N consumer warps (BC = 256):
// - GATEUP (BCOLS=128, TN=128): cwn = (lc >> 6) in 0..1, gid = lc & 7, j = (mtx << 3) | ((lc >> 3) & 7) in 0..15
// - DOWN   (BCOLS=256, TN=256): cwn = (t_col >> 7) in 0..1, gid = t_col & 7, j = (t_col >> 3) & 15 in 0..15
// Each (cwn, gid) loads all 16 scale bytes in one 128-bit LDS.128 (0 bank conflicts).
template <bool GATEUP>
__device__ __forceinline__ unsigned v31_ws_sf_index(unsigned t_col) {
    if constexpr (GATEUP) {
        const unsigned mtx = t_col >> 7, lc = t_col & 127u;
        return ((((lc >> 6) * 8u) + (lc & 7u)) << 4) + ((mtx << 3) | ((lc >> 3) & 7u));
    } else {
        return ((((t_col >> 7) * 8u) + (t_col & 7u)) << 4) + ((t_col >> 3) & 15u);
    }
}

__device__ __forceinline__ void v31_st_sbt_v4(unsigned s_addr, unsigned w0, unsigned w1, unsigned w2, unsigned w3) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :: "r"(s_addr), "r"(w0), "r"(w1), "r"(w2), "r"(w3) : "memory");
}

__device__ __forceinline__ uint4 v31_ld_sf_v4(unsigned s_addr) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(s_addr));
    return v;
}

template <int BAR_ID>
__device__ __forceinline__ void v31_bar_sync_512() {
    asm volatile("bar.sync %0, 512;\n" :: "n"(BAR_ID) : "memory");
}

template <int BAR_ID>
__device__ __forceinline__ void v31_bar_arrive_512() {
    asm volatile("bar.arrive %0, 512;\n" :: "n"(BAR_ID) : "memory");
}

template <int BAR_ID>
__device__ __forceinline__ void v31_bar_sync_256() {
    asm volatile("bar.sync %0, 256;\n" :: "n"(BAR_ID) : "memory");
}

template <int BAR_ID>
__device__ __forceinline__ void v31_bar_sync_128() {
    asm volatile("bar.sync %0, 128;\n" :: "n"(BAR_ID) : "memory");
}

__device__ __forceinline__ void v31_cp16_pol_s32(unsigned s_addr, const void* gmem_ptr,
                                                 unsigned long long pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, 16, %2;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "l"(pol));
}

__device__ __forceinline__ void v31_cp16_pol_pred_s32(unsigned s_addr, const void* gmem_ptr, bool pred,
                                                      unsigned long long pol) {
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "r"(bytes), "l"(pol));
}

template <bool GATEUP, int EPI = 0, bool PREFETCH = true, bool TS4E = false,
          typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void v31_ws_impl(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ pk0, const unsigned long long* __restrict__ sc0,
    const float* __restrict__ s20,
    const unsigned long long* __restrict__ pk1, const unsigned long long* __restrict__ sc1,
    const float* __restrict__ s21,
    __nv_bfloat16* __restrict__ C, unsigned char* __restrict__ oq, unsigned char* __restrict__ os,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace, int* status, unsigned N_in, unsigned K_in,
    unsigned char* __restrict__ slot_q = nullptr, unsigned char* __restrict__ slot_s = nullptr
) {
    using L = V31WsSmem<GATEUP, TS4E, Shape>;
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
    constexpr unsigned n_tiles = N / TN; // 5 for GATEUP, 10 for DOWN
    constexpr unsigned KS = K / 64u;     // 40 for GATEUP, 10 for DOWN
    constexpr unsigned kblocks = K / 32u;
    static_assert((KS & 1u) == 0u && KS >= 4u, "v31_ws requires even KS >= 4");

    unsigned dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (gridDim.x == 0 || gridDim.x > oi640::FIXED_GRID || blockDim.x != V31_WS_THREADS || dyn_smem < L::BYTES) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }
    const bool shape = (N_in == N && K_in == K);
    const bool outs = GATEUP ? (oq != nullptr && os != nullptr)
                             : (EPI == 3 ? (slot_q != nullptr && slot_s != nullptr) : (C != nullptr));
    const bool tbl_ok = pk0 != nullptr && sc0 != nullptr && s20 != nullptr &&
                        (!GATEUP || (pk1 != nullptr && sc1 != nullptr && s21 != nullptr));
#if V31_WS_USE_WORKSTEAL
    const bool ws_ok = oi640::workspace_ok_worksteal(workspace, contract);
#else
    const bool ws_ok = oi640::workspace_ok(workspace, contract);
#endif
    if (!shape || !outs || !tbl_ok || aq == nullptr || as == nullptr || expert_offsets == nullptr ||
        sorted_token_ids == nullptr || !oi640::contract_ok(contract) ||
        !ws_ok || status == nullptr || status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_ABI);
        return;
    }
    if (blockIdx.x >= workspace->work_count * n_tiles) return;

    extern __shared__ __align__(16) unsigned char smem[];

    if (threadIdx.x < 256u) {
        // =================================================================
        // WG0+WG1 (Warps 0..7, 256 threads): Producer + TS4E Decode + Transcode
        // 72 + 72 + 184 + 184 = 512 == 4 * 128 exact full 65,536 regs/SM.
        // =================================================================
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;\n");

        unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smem));
        asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n"
                     : "+r"(sbase));
        unsigned long long pol_el, pol_ef;
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));
        asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));
        const unsigned wc = workspace->work_count;
        const unsigned task_count = wc * n_tiles;
        unsigned task;
        asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(task));
#if V31_WS_USE_WORKSTEAL
        uint32_t* const next_task_ptr = GATEUP ? &workspace->next_task_gu : &workspace->next_task_dn;
        unsigned task_slot = 0u;
        #pragma unroll 1
        while (task < task_count) {
#else
        #pragma unroll 1
        for (; task < task_count; task += gridDim.x) {
#endif
            unsigned p_tid;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(p_tid));
            const unsigned raw_wi = task / n_tiles;
            const unsigned wi = (!GATEUP) ? (wc - 1u - raw_wi) : raw_wi;
            const uint2 item_u2 = *reinterpret_cast<const uint2*>(&workspace->items[wi]);
            const unsigned item_expert = item_u2.x;
            const unsigned item_m_tile = item_u2.y;
            const unsigned nt = task % n_tiles;
            const unsigned cta_n = nt * TN;
            if (item_expert >= Shape::EXPERTS) {
                if (p_tid == 0u) oi640::fail(status, oi640::ERR_PLAN);
                return;
            }
            const int m_start = expert_offsets[item_expert];
            const int M_expert = expert_offsets[item_expert + 1] - m_start;
            const int local = static_cast<int>(item_m_tile * V31_WS_TILE_M);
            const unsigned long long pB0 = pk0[item_expert];
            const unsigned long long pS0 = sc0[item_expert];
            const unsigned long long pB1 = GATEUP ? pk1[item_expert] : pB0;
            const unsigned long long pS1 = GATEUP ? sc1[item_expert] : pS0;
            const float f_s2a = s20[item_expert];
            const float f_s2b = GATEUP ? s21[item_expert] : 0.0f;
            if (m_start < 0 || M_expert <= 0 || local < 0 || local >= M_expert ||
                pB0 == 0ull || pS0 == 0ull || pB1 == 0ull || pS1 == 0ull ||
                !isfinite(f_s2a) || !isfinite(f_s2b)) {
                if (p_tid == 0u) oi640::fail(status, oi640::ERR_POINTER);
                return;
            }
            if (p_tid == 0u) {
                unsigned long long* const sPtrs = reinterpret_cast<unsigned long long*>(smem + L::OFF_PTRS);
                sPtrs[0] = pB0;
                sPtrs[1] = pS0;
                sPtrs[2] = pB1;
                sPtrs[3] = pS1;
            }
            const unsigned cta_m = static_cast<unsigned>(m_start + local);
            const int valid_rows = min(M_expert - local, static_cast<int>(V31_WS_TILE_M));
            const int active_a_rows = (valid_rows + 15) & ~15;
            const bool b_reuse = (!GATEUP)
                ? (item_m_tile > 0u)
                : (local + static_cast<int>(V31_WS_TILE_M) < M_expert);
            const unsigned long long pol_b = b_reuse ? pol_el : pol_ef;

            if constexpr (GATEUP) {
                if (p_tid < V31_WS_TILE_M) {
                    reinterpret_cast<unsigned*>(smem + L::OFF_AOFF)[p_tid] =
                        (static_cast<int>(p_tid) < valid_rows)
                            ? static_cast<unsigned>(sorted_token_ids[cta_m + p_tid])
                            : 0u;
                }
            }
            v31_bar_sync_256<5>();

            auto resolve_ts4e_bkt = [&](unsigned idx) {
                const unsigned ks_i = idx >> 2;
                const unsigned b = idx & 3u;
                const unsigned sub = b >> 1;
                const unsigned t = b & 1u;
                const unsigned char* blob = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                const unsigned bkt = (2u * ks_i + t) * TS4E_N_TILES + (cta_n >> 7) + (GATEUP ? 0u : sub);
                const unsigned* page_bases =
                    reinterpret_cast<const unsigned*>(blob + TS4E_PAGE_BASES_OFF);
                const unsigned short* bkt_off =
                    reinterpret_cast<const unsigned short*>(blob + TS4E_BKT_OFF_OFF);
                const unsigned pb0 = __ldg(&page_bases[bkt >> 7]);
                const unsigned pb1 = ((bkt & 127u) == 127u) ? __ldg(&page_bases[(bkt + 1u) >> 7]) : pb0;
                const unsigned beg = pb0 + static_cast<unsigned>(__ldg(&bkt_off[bkt]));
                const unsigned end = pb1 + static_cast<unsigned>(__ldg(&bkt_off[bkt + 1u]));
                reinterpret_cast<unsigned*>(smem + L::OFF_BKT_BEG)[idx] = beg;
                reinterpret_cast<unsigned short*>(smem + L::OFF_BKT_CNT)[idx] =
                    static_cast<unsigned short>(end - beg);
            };

            if constexpr (TS4E) {
                const unsigned p_wid = p_tid >> 5, p_lane = p_tid & 31u;
                if (p_wid < 4u) {
                    if (p_lane < 3u) {
                        resolve_ts4e_bkt(p_lane * 4u + p_wid);
                    }
                    __syncwarp();
                    if (p_wid == 0u && p_lane < 16u) {
                        constexpr unsigned BASE_CH = BCOLS / 16u;
                        const unsigned bm = p_lane / BASE_CH;
                        const unsigned bc = p_lane % BASE_CH;
                        const unsigned char* bs = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && bm) ? 3u : 1u);
                        v31_cp16_pol_s32(
                            sbase + L::OFF_BASE + p_lane * 16u,
                            bs + static_cast<unsigned long long>(TS4E_NIBBLES_BYTES) + cta_n + bc * 16u,
                            pol_b);
                    }
                }
            }

            auto load_a = [&](unsigned ks) {
                unsigned la_tid;
                asm volatile("mov.u32 %0, %%tid.x;" : "=r"(la_tid));
                const unsigned a_slot = ks % V14_DA;
                const unsigned a_r0 = la_tid >> 2, a_ch = la_tid & 3u; // 0..63
                #pragma unroll
                for (unsigned i = 0u; i < 2u; ++i) {
                    const unsigned a_row = a_r0 + i * 64u;
                    if (static_cast<int>(a_row) < active_a_rows) {
                        const bool a_ok = static_cast<int>(a_row) < valid_rows;
                        const unsigned src_row = a_ok
                            ? (GATEUP ? v29_ld_saoff(sbase + L::OFF_AOFF, a_row) : (cta_m + a_row))
                            : 0u;
                        const void* g_a = aq + static_cast<unsigned long long>(src_row) * K + ks * 64u + a_ch * 16u;
                        const unsigned dst_a = sbase + a_slot * (V31_WS_TILE_M * 64u) + v14_swz(a_row, a_ch);
                        if constexpr (!GATEUP) {
                            if (nt + 1u == n_tiles) v31_cp16_pol_pred_s32(dst_a, g_a, a_ok, pol_ef);
                            else                    v29_cp16_s32(dst_a, g_a, a_ok);
                        } else {
                            v29_cp16_s32(dst_a, g_a, a_ok);
                        }
                    }
                }
                if (la_tid < V31_WS_TILE_M && (ks & 1u) == 0u && static_cast<int>(la_tid) < active_a_rows) {
                    const bool sc_ok = static_cast<int>(la_tid) < valid_rows;
                    const unsigned sc_src_row = sc_ok
                        ? (GATEUP ? v29_ld_saoff(sbase + L::OFF_AOFF, la_tid) : (cta_m + la_tid))
                        : 0u;
                    v29_cp4_s32(sbase + L::OFF_AS + (((ks >> 1) & 1u) * V31_WS_TILE_M + la_tid) * 4u,
                                as + static_cast<unsigned long long>(sc_src_row) * kblocks + ks * 2u, sc_ok);
                }
            };

            auto load_b = [&](unsigned ks) {
                unsigned lb_tid;
                asm volatile("mov.u32 %0, %%tid.x;" : "=r"(lb_tid));
                constexpr unsigned CH = BCOLS / 16u; // 8 (GU) or 16 (DN)
                const unsigned sl = ks % V14_DB;
                if constexpr (GATEUP) {
                    const unsigned r = lb_tid >> 3, cc = lb_tid & 7u; // r in 0..31
                    const unsigned b_dst_col = (cc << 4) ^ ((r & 3u) << 5);
                    #pragma unroll
                    for (unsigned mtx = 0u; mtx < 2u; ++mtx) {
                        const unsigned char* src = v29_ld_sptr(sbase + L::OFF_PTRS, mtx ? 2u : 0u);
                        const unsigned dst_b = sbase + L::OFF_B + ((sl * 2u + mtx) * 32u + r) * 128u + b_dst_col;
                        const void* g_b = src + static_cast<unsigned long long>(ks * 32u + r) * N + cta_n + cc * 16u;
                        v31_cp16_pol_s32(dst_b, g_b, pol_b);
                    }
                } else {
                    const unsigned r_base = lb_tid >> 4, cc = lb_tid & 15u; // r_base in 0..15
                    const unsigned char* src = v29_ld_sptr(sbase + L::OFF_PTRS, 0u);
                    #pragma unroll
                    for (unsigned rh = 0u; rh < 2u; ++rh) {
                        const unsigned r = r_base + rh * 16u;
                        const unsigned b_dst_col = (cc << 4) ^ ((r & 3u) << 5);
                        const unsigned dst_b = sbase + L::OFF_B + (sl * 32u + r) * 256u + b_dst_col;
                        const void* g_b = src + static_cast<unsigned long long>(ks * 32u + r) * N + cta_n + cc * 16u;
                        v31_cp16_pol_s32(dst_b, g_b, pol_b);
                    }
                }
                if constexpr (TS4E) {
                    const unsigned lb_wid = lb_tid >> 5, lb_lane = lb_tid & 31u;
                    if (lb_wid < 4u) {
                        const unsigned sub = lb_wid >> 1;
                        const unsigned t = lb_wid & 1u;
                        const unsigned s_mtx = GATEUP ? sub : 0u;
                        const unsigned col_hi = GATEUP ? 0u : (sub << 7);
                        if (lb_lane < 8u) {
                            const unsigned char* s = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                            const void* g_s =
                                s + static_cast<unsigned long long>(ks * 2u + t) * N + cta_n + col_hi + lb_lane * 16u;
                            const unsigned dst_s =
                                sbase + L::OFF_S + ((sl * BMAT + s_mtx) * S_ROWS + 2u * t + 1u) * BCOLS + col_hi + lb_lane * 16u;
                            v31_cp16_pol_s32(dst_s, g_s, pol_b);
                        } else if (lb_lane < 24u) {
                            const unsigned c = lb_lane - 8u;
                            const unsigned idx = ks * 4u + lb_wid;
                            const unsigned cnt = static_cast<unsigned>(
                                reinterpret_cast<const unsigned short*>(smem + L::OFF_BKT_CNT)[idx]);
                            if (cnt > 0u) {
                                const unsigned beg = reinterpret_cast<const unsigned*>(smem + L::OFF_BKT_BEG)[idx];
                                const unsigned pad = beg & 7u;
                                if (pad + cnt <= L::TS4E_EXC_CAP && (c << 3) < (pad + cnt)) {
                                    const unsigned char* blob =
                                        v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                                    const unsigned short* exc =
                                        reinterpret_cast<const unsigned short*>(blob + TS4E_EXC_OFF);
                                    const unsigned dst_e =
                                        sbase + L::OFF_EXC + ((sl * 4u + lb_wid) * L::TS4E_EXC_CAP + (c << 3)) * 2u;
                                    v31_cp16_pol_s32(dst_e, exc + (beg & ~7u) + (c << 3), pol_b);
                                }
                            }
                        }
                    }
                } else {
                    if (lb_tid < 64u) {
                        const unsigned m2 = GATEUP ? (lb_tid >> 5) : 0u;
                        const unsigned r2 = GATEUP ? ((lb_tid >> 3) & 3u) : (lb_tid >> 4);
                        const unsigned c2 = lb_tid & (CH - 1u);
                        const unsigned char* s = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && m2) ? 3u : 1u);
                        const void* g_s =
                            s + static_cast<unsigned long long>(ks * 4u + r2) * N + cta_n + c2 * 16u;
                        const unsigned dst_s =
                            sbase + L::OFF_S + ((sl * BMAT + m2) * S_ROWS + r2) * BCOLS + c2 * 16u;
                        v31_cp16_pol_s32(dst_s, g_s, pol_b);
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
                unsigned char (*const sS)[BMAT][S_ROWS][BCOLS] =
                    reinterpret_cast<unsigned char (*)[BMAT][S_ROWS][BCOLS]>(smem + L::OFF_S);
                unsigned char* const sBase = smem + L::OFF_BASE;
                unsigned short (*const sExc)[4][L::TS4E_EXC_CAP] =
                    reinterpret_cast<unsigned short (*)[4][L::TS4E_EXC_CAP]>(smem + L::OFF_EXC);
                const unsigned sl = ks_dec % V14_DB;
                const unsigned sub = d_wid >> 1;
                const unsigned t = d_wid & 1u;
                const unsigned mtx = GATEUP ? sub : 0u;
                const unsigned col_hi = GATEUP ? 0u : (sub << 7);
                const unsigned lc4 = col_hi + (d_lane << 2);
                const unsigned idx = ks_dec * 4u + d_wid;
                const unsigned base_w = *reinterpret_cast<const unsigned*>(&sBase[(sub << 7) + (d_lane << 2)]);
                const unsigned nib = *reinterpret_cast<const unsigned*>(&sS[sl][mtx][2u * t + 1u][lc4]);
                const unsigned cnt = static_cast<unsigned>(
                    reinterpret_cast<const unsigned short*>(smem + L::OFF_BKT_CNT)[idx]);
                const unsigned beg = reinterpret_cast<const unsigned*>(smem + L::OFF_BKT_BEG)[idx];
                const unsigned pad = beg & 7u;
                const bool fast_exc = (cnt != 0u) && (pad + cnt <= L::TS4E_EXC_CAP);

                const unsigned c0 = nib & 0x0F0F0F0Fu;
                const unsigned e0 = (c0 + 0x01010101u) & 0x10101010u;
                *reinterpret_cast<unsigned*>(&sS[sl][mtx][2u * t][lc4]) = base_w + (c0 & ~(e0 - (e0 >> 4)));

                const unsigned c1 = (nib >> 4) & 0x0F0F0F0Fu;
                const unsigned e1 = (c1 + 0x01010101u) & 0x10101010u;
                *reinterpret_cast<unsigned*>(&sS[sl][mtx][2u * t + 1u][lc4]) = base_w + (c1 & ~(e1 - (e1 >> 4)));
                __syncwarp();

                if (cnt != 0u) {
                    if (fast_exc) {
                        for (unsigned i = d_lane; i < cnt; i += 32u) {
                            const unsigned e = static_cast<unsigned>(sExc[sl][d_wid][pad + i]);
                            sS[sl][mtx][2u * t + ((e >> 7) & 1u)][col_hi | (e & 127u)] =
                                static_cast<unsigned char>(e >> 8);
                        }
                    } else {
                        const unsigned char* blob = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                        const unsigned short* exc =
                            reinterpret_cast<const unsigned short*>(blob + TS4E_EXC_OFF);
                        for (unsigned i = d_lane; i < cnt; i += 32u) {
                            const unsigned e = static_cast<unsigned>(__ldg(&exc[beg + i]));
                            sS[sl][mtx][2u * t + ((e >> 7) & 1u)][col_hi | (e & 127u)] =
                                static_cast<unsigned char>(e >> 8);
                        }
                    }
                }
            };

            auto prefetch_b = [&](unsigned pf_ks) {
#if V31_WS_USE_WORKSTEAL
                if (task_count <= gridDim.x) return;
                unsigned raw_peek;
                asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(raw_peek) : "l"(next_task_ptr));
                if (raw_peek >= task_count - gridDim.x) return;
                const unsigned pf_task = gridDim.x + raw_peek;
#else
                const unsigned pf_task = task + gridDim.x;
                if (pf_task >= task_count) return;
#endif
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
                unsigned pf_tid;
                asm volatile("mov.u32 %0, %%tid.x;" : "=r"(pf_tid));
                constexpr unsigned CH = BCOLS / 16u;
                constexpr unsigned S_ROWS_LD = TS4E ? 2u : 4u;
                if constexpr (GATEUP) {
                    const unsigned mtx = pf_tid >> 7, r = (pf_tid >> 2) & 31u, c32 = (pf_tid & 3u) << 5;
                    const unsigned char* pB = reinterpret_cast<const unsigned char*>(
                        mtx ? pk1[pf_item.expert] : pk0[pf_item.expert]);
                    if (pB != nullptr) {
                        v28_prefetch_l2(pB + static_cast<unsigned long long>(pf_ks * 32u + r) * N + pf_cta_n + c32);
                    }
                } else {
                    const unsigned r = pf_tid >> 3, c32 = (pf_tid & 7u) << 5;
                    const unsigned char* pB = reinterpret_cast<const unsigned char*>(pk0[pf_item.expert]);
                    if (pB != nullptr) {
                        v28_prefetch_l2(pB + static_cast<unsigned long long>(pf_ks * 32u + r) * N + pf_cta_n + c32);
                    }
                }
                if (pf_tid < BMAT * S_ROWS_LD * (CH / 2u)) {
                    const unsigned m2 = pf_tid / (S_ROWS_LD * (CH / 2u));
                    const unsigned r2 = (pf_tid / (CH / 2u)) % S_ROWS_LD;
                    const unsigned c32 = (pf_tid % (CH / 2u)) << 5;
                    const unsigned char* pS = reinterpret_cast<const unsigned char*>(
                        (GATEUP && m2) ? sc1[pf_item.expert] : sc0[pf_item.expert]);
                    if (pS != nullptr) {
                        v28_prefetch_l2(pS + static_cast<unsigned long long>(pf_ks * S_ROWS_LD + r2) * N + pf_cta_n + c32);
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
                                v28_prefetch_l2(pS + static_cast<unsigned long long>(TS4E_NIBBLES_BYTES) + pf_cta_n + off32);
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

            // Conflict-free 4-thread-cooperative SWAR transcode on 256 Producer threads (1 col/thread):
            // overlaps register-only v31_e4m3_lut_ptx + 4x sB LDS.32 across both k16 halves and emits STS.128 into sBt.
            auto transcode_k64 = [&](unsigned sl, unsigned tb) {
                unsigned t_col;
                asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t_col));
                const unsigned t_q = t_col & 3u;
                const unsigned sel1 = (t_q & 1u) ? 0x3715u : 0x6240u;
                const unsigned sel2 = (t_q & 2u) ? 0x3276u : 0x5410u;
                const unsigned t_mtx = GATEUP ? (t_col >> 7) : 0u;
                const unsigned t_lc = GATEUP ? (t_col & 127u) : t_col;
                const unsigned c4_swz = (t_lc & ~3u) ^ (t_q << 5);
                const unsigned sf_idx = v31_ws_sf_index<GATEUP>(t_col);
                #pragma unroll
                for (unsigned t = 0u; t < 2u; ++t) {
                    const unsigned ss_base =
                        sbase + L::OFF_S + (((sl * BMAT + t_mtx) * S_ROWS + 2u * t) * BCOLS) + t_lc;
                    const unsigned s0 = v28_ld_su8_vol(ss_base);
                    const unsigned s1 = v28_ld_su8_off_vol<BCOLS>(ss_base);
                    const unsigned s_max = max(s0 & 0x7fu, s1 & 0x7fu);
                    const int e32 = (s_max >= 8u) ? (static_cast<int>(s_max >> 3) - 7)
                                  : (s_max != 0u) ? (22 - __clz(static_cast<int>(s_max)))
                                                  : 0;
                    v28_st_su8_vol(sbase + L::OFF_SF + (tb * 2u + t) * V31_WS_BC + sf_idx,
                                   static_cast<unsigned>(e32 + 127) & 0xffu);
                    const unsigned r0_sc = v28_norm_e4m3_scale(s0, e32);
                    const unsigned r1_sc = v28_norm_e4m3_scale(s1, e32);
                    const uint2 tt0 = v31_e4m3_lut_ptx(r0_sc);
                    const uint2 tt1 = v31_e4m3_lut_ptx(r1_sc);

                    const unsigned sb_base =
                        sbase + L::OFF_B + (((sl * BMAT + t_mtx) * 32u + 16u * t + t_q) * BCOLS) + c4_swz;
                    const unsigned u0_0 = v28_ld_su32_vol(sb_base);
                    const unsigned u1_0 = v28_ld_su32_off_vol<4u * BCOLS>(sb_base);
                    const unsigned u0_1 = v28_ld_su32_off_vol<8u * BCOLS>(sb_base);
                    const unsigned u1_1 = v28_ld_su32_off_vol<12u * BCOLS>(sb_base);

                    {
                        const unsigned p0 = v10_prmt(u0_0, __shfl_xor_sync(0xffffffffu, u0_0, 1), sel1);
                        const unsigned c0 = v10_prmt(p0, __shfl_xor_sync(0xffffffffu, p0, 2), sel2);
                        const unsigned p1 = v10_prmt(u1_0, __shfl_xor_sync(0xffffffffu, u1_0, 1), sel1);
                        const unsigned c1 = v10_prmt(p1, __shfl_xor_sync(0xffffffffu, p1, 2), sel2);
                        const unsigned w0 = v10_e2m1x4_to_e4m3x4(c0, tt0.x, tt0.y);
                        const unsigned w1 = v10_e2m1x4_to_e4m3x4(c0 >> 16, tt0.x, tt0.y);
                        const unsigned w2 = v10_e2m1x4_to_e4m3x4(c1, tt0.x, tt0.y);
                        const unsigned w3 = v10_e2m1x4_to_e4m3x4(c1 >> 16, tt0.x, tt0.y);
                        v31_st_sbt_v4(sbase + L::OFF_BT + tb * (V31_WS_BC * 64u) + v14_swz(t_col, 2u * t),
                                      w0, w1, w2, w3);
                    }
                    {
                        const unsigned p0 = v10_prmt(u0_1, __shfl_xor_sync(0xffffffffu, u0_1, 1), sel1);
                        const unsigned c0 = v10_prmt(p0, __shfl_xor_sync(0xffffffffu, p0, 2), sel2);
                        const unsigned p1 = v10_prmt(u1_1, __shfl_xor_sync(0xffffffffu, u1_1, 1), sel1);
                        const unsigned c1 = v10_prmt(p1, __shfl_xor_sync(0xffffffffu, p1, 2), sel2);
                        const unsigned w0 = v10_e2m1x4_to_e4m3x4(c0, tt1.x, tt1.y);
                        const unsigned w1 = v10_e2m1x4_to_e4m3x4(c0 >> 16, tt1.x, tt1.y);
                        const unsigned w2 = v10_e2m1x4_to_e4m3x4(c1, tt1.x, tt1.y);
                        const unsigned w3 = v10_e2m1x4_to_e4m3x4(c1 >> 16, tt1.x, tt1.y);
                        v31_st_sbt_v4(sbase + L::OFF_BT + tb * (V31_WS_BC * 64u) + v14_swz(t_col, 2u * t + 1u),
                                      w0, w1, w2, w3);
                    }
                }
            };

            // Prologue: Gp0={sBase,B0,S0,Exc0}, Gp1={A0,AS0,B1,S1,Exc1}, Gp2={A1,B2,S2,Exc2}
            load_b(0u);
            v10_commit();
            load_a(0u);
            load_b(1u);
            v10_commit();
            load_a(1u);
            load_b(2u);
            v10_commit();
            if constexpr (TS4E) {
                unsigned r_tid;
                asm volatile("mov.u32 %0, %%tid.x;" : "=r"(r_tid));
                if (r_tid < 128u) {
                    v10_wait<2>();
                    v31_bar_sync_128<6>();
                    decode_ts4e_stage(0u);
                } else {
                    for (unsigned idx = 12u + (r_tid - 128u); idx < KS * 4u; idx += 128u) {
                        resolve_ts4e_bkt(idx);
                    }
                    v10_wait<2>();
                }
            } else {
                v10_wait<2>();
            }
            v31_bar_sync_256<5>();
            transcode_k64(0u, 0u);
            v10_wait<1>();
            v31_bar_arrive_512<1>(); // B_FULL_0: ks=0 ready

            // Stage ks=1 (buffer 1)
            if constexpr (TS4E) decode_ts4e_stage(1u);
            v31_bar_sync_256<5>();
            load_a(2u);
            load_b(3u);
            v10_commit();
            transcode_k64(1u, 1u);
            v10_wait<1>();
            v31_bar_arrive_512<2>(); // B_FULL_1: ks=1 ready

            // Steady-state producer loop for ks = 2 .. KS-1
            #pragma unroll 1
            for (unsigned ks = 2u; ks < KS; ks += 2u) {
                // Even ks -> buffer 0
                if constexpr (TS4E) decode_ts4e_stage(ks);
                v31_bar_sync_512<3>(); // B_EMPTY_0: ks-2 consumed + WG0+WG1 sync
                load_a(ks + 1u);
                if (ks + 2u < KS) load_b(ks + 2u);
                else if constexpr (PREFETCH) prefetch_b(ks + 2u - KS);
                v10_commit();
                transcode_k64(ks % V14_DB, 0u);
                v10_wait<1>();
                v31_bar_arrive_512<1>(); // B_FULL_0: ks ready

                // Odd ks+1 -> buffer 1
                const unsigned ks_odd = ks + 1u;
                if constexpr (TS4E) decode_ts4e_stage(ks_odd);
                v31_bar_sync_512<4>(); // B_EMPTY_1: ks_odd-2 consumed + WG0+WG1 sync
                if (ks_odd + 1u < KS) {
                    load_a(ks_odd + 1u);
                    if (ks_odd + 2u < KS) load_b(ks_odd + 2u);
                    else if constexpr (PREFETCH) prefetch_b(ks_odd + 2u - KS);
                    v10_commit();
                } else if constexpr (PREFETCH) {
                    prefetch_b(ks_odd + 2u - KS);
                }
                transcode_k64(ks_odd % V14_DB, 1u);
                if (ks_odd + 1u < KS) v10_wait<1>();
                else                  v10_wait<0>();
                v31_bar_arrive_512<2>(); // B_FULL_1: ks_odd ready
            }

            // Wait for Consumer to finish reading KS-2 and KS-1 before starting next task
            // (overlaps next task prologue with Consumer's 0-SMEM epilogue!).
            v31_bar_sync_512<3>();
            v31_bar_sync_512<4>();
#if V31_WS_USE_WORKSTEAL
            task_slot ^= 1u;
            {
                unsigned end_tid;
                asm volatile("mov.u32 %0, %%tid.x;" : "=r"(end_tid));
                if (end_tid == 0u) {
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
                    v31_st_su32_vol(sbase + L::OFF_TASK + task_slot * 4u, next_t);
                }
            }
            v31_bar_sync_256<5>();
            task = v28_ld_su32_vol(sbase + L::OFF_TASK + task_slot * 4u);
            v31_bar_arrive_512<7>();
#endif
        }
    } else {
        // =================================================================
        // WG2+WG3 (Warps 8..15, 256 threads, 4 M x 2 N): Consumer MMA WGs
        // =================================================================
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;\n");

        const unsigned sbase_c = static_cast<unsigned>(__cvta_generic_to_shared(smem));
        const unsigned wc = workspace->work_count;
        const unsigned task_count = wc * n_tiles;
        unsigned task;
        asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(task));
        unsigned c_tid;
        asm volatile("mov.u32 %0, %%tid.x;" : "=r"(c_tid));
        const unsigned cwid = (c_tid - 256u) >> 5; // 0..7
        const unsigned lane = c_tid & 31u;
        const unsigned gid = lane >> 2, tig = lane & 3u;
        const unsigned cwm = cwid >> 1;            // 0..3
        const unsigned cwn = cwid & 1u;            // 0..1
        const unsigned warp_m0 = cwm * 32u;
        const unsigned warp_n0 = cwn * (GATEUP ? 64u : 128u);
        const unsigned sf_off = ((cwn * 8u) + gid) << 4;
        const unsigned as_row0 = warp_m0 + gid + (tig & 1u) * 8u;
        const unsigned swz_a0 = v14_swz(warp_m0 + (lane & 15u), lane >> 4);
        const unsigned swz_b0 = v14_swz(warp_n0 + ((lane >> 4) << 3) + (lane & 7u), (lane >> 3) & 1u);
        constexpr unsigned off0 = 0u;
        constexpr unsigned off1 = 1024u;
        constexpr unsigned off2 = 2048u;
        constexpr unsigned off3 = 3072u;
        constexpr unsigned off4 = GATEUP ? (BCOLS * 64u) : 4096u;
        constexpr unsigned off5 = GATEUP ? (BCOLS * 64u + 1024u) : 5120u;
        constexpr unsigned off6 = GATEUP ? (BCOLS * 64u + 2048u) : 6144u;
        constexpr unsigned off7 = GATEUP ? (BCOLS * 64u + 3072u) : 7168u;

#if V31_WS_USE_WORKSTEAL
        unsigned task_slot = 0u;
        #pragma unroll 1
        while (task < task_count) {
#else
        #pragma unroll 1
        for (; task < task_count; task += gridDim.x) {
#endif
            const unsigned raw_wi = task / n_tiles;
            const unsigned wi = (!GATEUP) ? (wc - 1u - raw_wi) : raw_wi;
            const uint2 item_u2 = *reinterpret_cast<const uint2*>(&workspace->items[wi]);
            const unsigned item_expert = item_u2.x;
            const unsigned item_m_tile = item_u2.y;
            if (item_expert >= Shape::EXPERTS) return;
            const int m_start = expert_offsets[item_expert];
            const int M_expert = expert_offsets[item_expert + 1] - m_start;
            const int local = static_cast<int>(item_m_tile * V31_WS_TILE_M);
            const unsigned long long pB0 = pk0[item_expert];
            const unsigned long long pS0 = sc0[item_expert];
            const unsigned long long pB1 = GATEUP ? pk1[item_expert] : pB0;
            const unsigned long long pS1 = GATEUP ? sc1[item_expert] : pS0;
            const float ck_s2a = s20[item_expert];
            const float ck_s2b = GATEUP ? s21[item_expert] : 0.0f;
            if (m_start < 0 || M_expert <= 0 || local < 0 || local >= M_expert ||
                pB0 == 0ull || pS0 == 0ull || pB1 == 0ull || pS1 == 0ull ||
                !isfinite(ck_s2a) || !isfinite(ck_s2b)) {
                return;
            }
            const int valid_rows0 = min(M_expert - local, static_cast<int>(V31_WS_TILE_M));
            const int warp_valid = valid_rows0 - static_cast<int>(as_row0 & ~31u);
            const int rt_active = warp_valid <= 0 ? 0 : min(2, (warp_valid + 15) >> 4);

            float acc[2][16][4];
            #pragma unroll
            for (int r = 0; r < 2; ++r)
                #pragma unroll
                for (int j = 0; j < 16; ++j)
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.0f;

            unsigned sfa[2] = {0u, 0u};
            auto cons_step = [&](auto par_tag, unsigned ks) {
                constexpr unsigned PAR = decltype(par_tag)::value;
                constexpr int BAR_FULL = PAR == 0u ? 1 : 2;
                constexpr int BAR_EMPTY = PAR == 0u ? 3 : 4;
                v31_bar_sync_512<BAR_FULL>();
                if (rt_active > 0) {
                    if constexpr (PAR == 0u) {
                        const unsigned as_buf = (ks >> 1) & 1u;
                        const unsigned as_addr = sbase_c + L::OFF_AS + (as_buf * V31_WS_TILE_M + as_row0) * 4u;
                        sfa[0] = v28_ld_su32_vol(as_addr);
                        if (rt_active > 1) sfa[1] = v28_ld_su32_off_vol<16u * 4u>(as_addr);
                    }
                    const unsigned a_slot = ks % V14_DA;
                    auto run_half = [&](auto t_tag) {
                        constexpr unsigned T = decltype(t_tag)::value;
                        const uint4 sfb = v31_ld_sf_v4(sbase_c + L::OFF_SF + (PAR * 2u + T) * V31_WS_BC + sf_off);
                        const unsigned addr_a0 = sbase_c + a_slot * (V31_WS_TILE_M * 64u) + (swz_a0 ^ (T << 5));
                        const unsigned addr_b0 = sbase_c + L::OFF_BT + PAR * (V31_WS_BC * 64u) + (swz_b0 ^ (T << 5));
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
                        v10_ldsm4(addr_b0 + off0, bfr[0][0], bfr[0][1], bfr[0][2], bfr[0][3]);
                        v10_ldsm4(addr_a0, afr0[0], afr0[1], afr0[2], afr0[3]);
                        if (rt_active > 1) v10_ldsm4(addr_a0 + 1024u, afr1[0], afr1[1], afr1[2], afr1[3]);
                        v10_ldsm4(addr_b0 + off1, bfr[1][0], bfr[1][1], bfr[1][2], bfr[1][3]);
                        emit_pair(std::integral_constant<int, 0>{}, bfr[0], sfb.x);
                        v10_ldsm4(addr_b0 + off2, bfr[0][0], bfr[0][1], bfr[0][2], bfr[0][3]);
                        emit_pair(std::integral_constant<int, 1>{}, bfr[1], sfb.x);
                        v10_ldsm4(addr_b0 + off3, bfr[1][0], bfr[1][1], bfr[1][2], bfr[1][3]);
                        emit_pair(std::integral_constant<int, 2>{}, bfr[0], sfb.y);
                        v10_ldsm4(addr_b0 + off4, bfr[0][0], bfr[0][1], bfr[0][2], bfr[0][3]);
                        emit_pair(std::integral_constant<int, 3>{}, bfr[1], sfb.y);
                        v10_ldsm4(addr_b0 + off5, bfr[1][0], bfr[1][1], bfr[1][2], bfr[1][3]);
                        emit_pair(std::integral_constant<int, 4>{}, bfr[0], sfb.z);
                        v10_ldsm4(addr_b0 + off6, bfr[0][0], bfr[0][1], bfr[0][2], bfr[0][3]);
                        emit_pair(std::integral_constant<int, 5>{}, bfr[1], sfb.z);
                        v10_ldsm4(addr_b0 + off7, bfr[1][0], bfr[1][1], bfr[1][2], bfr[1][3]);
                        emit_pair(std::integral_constant<int, 6>{}, bfr[0], sfb.w);
                        emit_pair(std::integral_constant<int, 7>{}, bfr[1], sfb.w);
                    };
                    run_half(std::integral_constant<unsigned, 0u>{});
                    run_half(std::integral_constant<unsigned, 1u>{});
                }
                v31_bar_arrive_512<BAR_EMPTY>();
            };

            #pragma unroll 1
            for (unsigned ks = 0u; ks < KS; ks += 2u) {
                cons_step(std::integral_constant<unsigned, 0u>{}, ks);
                cons_step(std::integral_constant<unsigned, 1u>{}, ks + 1u);
            }

            // 0-SMEM Register-Direct Epilogue (overlaps with Producer's next task prologue!).
            // Derive epilogue metadata post-K-loop from L1 so 0 epilogue regs stay live across K.
            unsigned ep_task = task;
            asm volatile("" : "+r"(ep_task));
            unsigned ep_tid;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(ep_tid));
            const unsigned ep_cwid = (ep_tid - 256u) >> 5;
            const unsigned ep_lane = ep_tid & 31u;
            const unsigned ep_gid = ep_lane >> 2, ep_tig = ep_lane & 3u;
            const unsigned ep_warp_m = (ep_cwid >> 1) * 32u;
            const unsigned ep_warp_n = (ep_cwid & 1u) * (GATEUP ? 64u : 128u);
            const unsigned ep_raw_wi = ep_task / n_tiles;
            const unsigned ep_wi = (!GATEUP) ? (wc - 1u - ep_raw_wi) : ep_raw_wi;
            const uint2 ep_item = *reinterpret_cast<const uint2*>(&workspace->items[ep_wi]);
            const unsigned ep_cta_n = (ep_task % n_tiles) * TN;
            const int ep_m_start = expert_offsets[ep_item.x];
            const int ep_M_expert = expert_offsets[ep_item.x + 1] - ep_m_start;
            const int ep_local = static_cast<int>(ep_item.y * V31_WS_TILE_M);
            const unsigned ep_cta_m = static_cast<unsigned>(ep_m_start + ep_local);
            const int ep_valid_rows = min(ep_M_expert - ep_local, static_cast<int>(V31_WS_TILE_M));
            const float ep_s2a = s20[ep_item.x];
            const float ep_s2b = GATEUP ? s21[ep_item.x] : 0.0f;

            if constexpr (GATEUP) {
                const bool keep_oq = (workspace->rows <= 4096u) || (ep_raw_wi + V31_KEEP_OQ_WINDOW >= wc);
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    if (r >= rt_active) break;
                    const unsigned lr0 = ep_warp_m + r * 16u + ep_gid, lr1 = lr0 + 8u;
                    #pragma unroll
                    for (unsigned b = 0u; b < 2u; ++b) {
                        const unsigned col = ep_cta_n + ep_warp_n + b * 32u;
                        const unsigned col8 = col + ep_tig * 8u;
                        const unsigned blk = (col >> 5);
                        {
                            unsigned u0[4];
                            #pragma unroll
                            for (int jj = 0; jj < 4; ++jj) {
                                const int j = 4 * static_cast<int>(b) + jj;
                                const float g0 = acc[r][j][0] * ep_s2a, u0f = acc[r][j + 8][0] * ep_s2b;
                                const float g1 = acc[r][j][1] * ep_s2a, u1f = acc[r][j + 8][1] * ep_s2b;
                                const float v0 = g0 * (1.0f / (1.0f + __expf(-g0))) * u0f;
                                const float v1 = g1 * (1.0f / (1.0f + __expf(-g1))) * u1f;
                                u0[jj] = v28_pack_bf162(v0, v1);
                            }
                            unsigned qq0[2], ss0;
                            v28_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], ep_tig, qq0, ss0);
                            if (static_cast<int>(lr0) < ep_valid_rows) {
                                const unsigned long long orow = ep_cta_m + lr0;
                                if (keep_oq) v28_st8_evict_last(oq + orow * N + col8, make_uint2(qq0[0], qq0[1]));
                                else         v28_st8_evict_first(oq + orow * N + col8, make_uint2(qq0[0], qq0[1]));
                                if (ep_tig == 0u) os[orow * (N / 32u) + blk] = static_cast<unsigned char>(ss0);
                            }
                        }
                        asm volatile("" ::: "memory");
                        {
                            unsigned u1[4];
                            #pragma unroll
                            for (int jj = 0; jj < 4; ++jj) {
                                const int j = 4 * static_cast<int>(b) + jj;
                                const float g2 = acc[r][j][2] * ep_s2a, u2f = acc[r][j + 8][2] * ep_s2b;
                                const float g3 = acc[r][j][3] * ep_s2a, u3f = acc[r][j + 8][3] * ep_s2b;
                                const float v2 = g2 * (1.0f / (1.0f + __expf(-g2))) * u2f;
                                const float v3 = g3 * (1.0f / (1.0f + __expf(-g3))) * u3f;
                                u1[jj] = v28_pack_bf162(v2, v3);
                            }
                            unsigned qq1[2], ss1;
                            v28_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], ep_tig, qq1, ss1);
                            if (static_cast<int>(lr1) < ep_valid_rows) {
                                const unsigned long long orow = ep_cta_m + lr1;
                                if (keep_oq) v28_st8_evict_last(oq + orow * N + col8, make_uint2(qq1[0], qq1[1]));
                                else         v28_st8_evict_first(oq + orow * N + col8, make_uint2(qq1[0], qq1[1]));
                                if (ep_tig == 0u) os[orow * (N / 32u) + blk] = static_cast<unsigned char>(ss1);
                            }
                        }
                        asm volatile("" ::: "memory");
                    }
                }
            } else {
                const bool keep_slot =
                    (static_cast<unsigned long long>(workspace->expanded) * Shape::SLOT_MXFP8_BYTES_ULL) <= V28_KEEP_SLOT_BYTES;
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    if (r >= rt_active) break;
                    const unsigned lr0 = ep_warp_m + r * 16u + ep_gid, lr1 = lr0 + 8u;
                    #pragma unroll
                    for (unsigned b = 0u; b < 4u; ++b) {
                        const unsigned col = ep_cta_n + ep_warp_n + b * 32u;
                        const unsigned col8 = col + ep_tig * 8u;
                        {
                            unsigned u0[4];
                            #pragma unroll
                            for (int jj = 0; jj < 4; ++jj) {
                                const int j = 4 * static_cast<int>(b) + jj;
                                u0[jj] = v28_pack_bf162(acc[r][j][0] * ep_s2a, acc[r][j][1] * ep_s2a);
                            }
                            if constexpr (EPI == 3) {
                                unsigned qq0[2], ss0;
                                v28_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], ep_tig, qq0, ss0);
                                if (static_cast<int>(lr0) < ep_valid_rows) {
                                    const unsigned long long prow = ep_cta_m + lr0;
                                    if (keep_slot) {
                                        v28_st8_l2_evict_last(slot_q + prow * N + col8, make_uint2(qq0[0], qq0[1]));
                                    } else {
                                        v28_st8_evict_first(slot_q + prow * N + col8, make_uint2(qq0[0], qq0[1]));
                                    }
                                    if (ep_tig == 0u) slot_s[prow * (N / 32u) + (col >> 5)] = static_cast<unsigned char>(ss0);
                                }
                            } else {
                                const uint4 w0 = v28_bf16_reg_quad(u0[0], u0[1], u0[2], u0[3], ep_tig);
                                if (static_cast<int>(lr0) < ep_valid_rows) {
                                    v28_st16_stream(C + static_cast<unsigned long long>(ep_cta_m + lr0) * N + col8, w0);
                                }
                            }
                        }
                        asm volatile("" ::: "memory");
                        {
                            unsigned u1[4];
                            #pragma unroll
                            for (int jj = 0; jj < 4; ++jj) {
                                const int j = 4 * static_cast<int>(b) + jj;
                                u1[jj] = v28_pack_bf162(acc[r][j][2] * ep_s2a, acc[r][j][3] * ep_s2a);
                            }
                            if constexpr (EPI == 3) {
                                unsigned qq1[2], ss1;
                                v28_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], ep_tig, qq1, ss1);
                                if (static_cast<int>(lr1) < ep_valid_rows) {
                                    const unsigned long long prow = ep_cta_m + lr1;
                                    if (keep_slot) {
                                        v28_st8_l2_evict_last(slot_q + prow * N + col8, make_uint2(qq1[0], qq1[1]));
                                    } else {
                                        v28_st8_evict_first(slot_q + prow * N + col8, make_uint2(qq1[0], qq1[1]));
                                    }
                                    if (ep_tig == 0u) slot_s[prow * (N / 32u) + (col >> 5)] = static_cast<unsigned char>(ss1);
                                }
                            } else {
                                const uint4 w1 = v28_bf16_reg_quad(u1[0], u1[1], u1[2], u1[3], ep_tig);
                                if (static_cast<int>(lr1) < ep_valid_rows) {
                                    v28_st16_stream(C + static_cast<unsigned long long>(ep_cta_m + lr1) * N + col8, w1);
                                }
                            }
                        }
                        asm volatile("" ::: "memory");
                    }
                }
            }
#if V31_WS_USE_WORKSTEAL
            task_slot ^= 1u;
            v31_bar_sync_512<7>();
            task = v28_ld_su32_vol(sbase_c + L::OFF_TASK + task_slot * 4u);
#endif
        }
    }
}

} // namespace

// V31 Warp-Specialized Flat-scale entry points (87,616 B GU / 87,104 B DN SMEM, 512 threads).
extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v31_ws(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v31_ws_impl<true, 0, true, false>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v31_ws(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v31_ws_impl<false, 0, true, false>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v31q_ws(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v31_ws_impl<false, 3, true, false>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
        slot_q, slot_s);
}

// V31 Warp-Specialized Atlas-TS4E-128 entry points (91,904 B GU / 90,672 B DN SMEM, 512 threads).
extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v31_ws_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v31_ws_impl<true, 0, true, true>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v31_ws_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v31_ws_impl<false, 0, true, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v31q_ws_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v31_ws_impl<false, 3, true, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
        slot_q, slot_s);
}




