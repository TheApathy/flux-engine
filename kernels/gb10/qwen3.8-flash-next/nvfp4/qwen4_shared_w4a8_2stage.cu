// SPDX-License-Identifier: AGPL-3.0-only
//
// 2-stage <= 48 KiB SMEM (34,304 B = 33.5 KiB) pre-transcoded W4A8 shared-expert GateUp
// kernel (`qwen4_shared_w4a8_gateup_k32_v28_2stage`, 256 threads = 4 M x 2 N, 2 CTAs/SM on
// GB10 sm_121a, <= 128 regs/thread, 0 spills).
//
// Consumes the exact `bt_gateup` / `sf_gateup` layout written by `qwen4_shared_pretranscode_v28`
// (5 x 128-col slices split into 10 x 64-col half-tiles: gate cols [0..64) / [64..128) at
// byte offsets +0 / +4096, up cols at +8192 / +12288, packed UE8M0 scales at +0 / +128 per t).
// Strictly 0-ULP bit-identical to `qwen4_shared_w4a8_gateup_k32_v28`, while dropping dynamic
// SMEM from 101,376 B (blocking SM co-residency) to 34,304 B so it co-resides on the same SM
// with concurrent `topk` and 1-CTA `qwen4_moe_sort_plan_t_v8` on the primary stream.

#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define OI640_STEP_K 32
#define OI640_TRANSPOSED 1
#define flash_next_orig_i640_prefill flash_next_f32_transposed_shared_w4a8_2stage
#include "moe_w4a16_orig_i640_compact_prefill.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <string.h>
#include <type_traits>

namespace oi640 = flash_next_f32_transposed_shared_w4a8_2stage;

namespace {

constexpr unsigned SW2_TILE_M = 128u;

__device__ __forceinline__ unsigned sw2_swz(unsigned row, unsigned chunk) {
    return row * 64u + ((chunk ^ ((row >> 1) & 3u)) << 4);
}

__device__ __forceinline__ void sw2_cp16(void* smem, const void* gmem, bool pred) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(gmem), "r"(pred ? 16 : 0));
}

__device__ __forceinline__ void sw2_cp4(void* smem, const void* gmem, bool pred) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(s), "l"(gmem), "r"(pred ? 4 : 0));
}

__device__ __forceinline__ void sw2_cp16_l2_el_s32(
    unsigned s_addr, const void* gmem_ptr, unsigned long long pol_el
) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, 16, %2;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "l"(pol_el));
}

__device__ __forceinline__ void sw2_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void sw2_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void sw2_ldsm4(
    unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3
) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 1200
__device__ __forceinline__ float sw2_scale_ue8m0_pair(float t, unsigned ea, unsigned eb) {
    const unsigned s = ea + eb;
    const unsigned h1 = s >> 1;
    const unsigned h2 = s - h1;
    const unsigned fa_bits = (ea == 255u || eb == 255u) ? 0x7fc00000u : (h1 << 23);
    return (t * __uint_as_float(fa_bits)) * __uint_as_float(h2 << 23);
}
#endif

template <int BIDA, int BIDB>
__device__ __forceinline__ void sw2_qmma(
    float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1, unsigned sa, unsigned sb
) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%14, %13};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb),
          "n"(BIDA), "n"(0), "n"(BIDB));
#else
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(t0), "=f"(t1), "=f"(t2), "=f"(t3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "f"(t0), "f"(t1), "f"(t2), "f"(t3));
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    const unsigned grp = lane & ~3u;
    const unsigned tig = lane & 3u;
    const unsigned sa_b = (sa >> (BIDA * 8)) & 0xffu;
    const unsigned sb_b = (sb >> (BIDB * 8)) & 0xffu;
    const unsigned ea_lo = __shfl_sync(0xffffffffu, sa_b, grp);
    const unsigned ea_hi = __shfl_sync(0xffffffffu, sa_b, grp | 1u);
    const unsigned eb_even = __shfl_sync(0xffffffffu, sb_b, tig << 3);
    const unsigned eb_odd = __shfl_sync(0xffffffffu, sb_b, (tig << 3) | 4u);
    d[0] += sw2_scale_ue8m0_pair(t0, ea_lo, eb_even);
    d[1] += sw2_scale_ue8m0_pair(t1, ea_lo, eb_odd);
    d[2] += sw2_scale_ue8m0_pair(t2, ea_hi, eb_even);
    d[3] += sw2_scale_ue8m0_pair(t3, ea_hi, eb_odd);
#endif
}

__device__ __forceinline__ unsigned sw2_pack_e4m3x4(float a, float b, float c, float d) {
    unsigned short p0, p1;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p0) : "f"(a), "f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p1) : "f"(c), "f"(d));
    unsigned r;
    asm("prmt.b32 %0, %1, %2, 0x5410;"
        : "=r"(r)
        : "r"(static_cast<unsigned>(p0)), "r"(static_cast<unsigned>(p1)));
    return r;
}

__device__ __forceinline__ unsigned sw2_pack_bf162(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    unsigned u;
    memcpy(&u, &v, sizeof(u));
    return u;
}

__device__ __forceinline__ void sw2_st8_evict_last(void* dst, uint2 v) {
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y), "l"(pol)
                 : "memory");
}

// Bit-identical to v28_mxfp8_quad in qwen4_moe_compact_t_k32_v28.cu.
__device__ __forceinline__ void sw2_mxfp8_quad(const uint4 w, unsigned (&qq)[2], unsigned& ss) {
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
            qq[i] = sw2_pack_e4m3x4(
                v[4 * i + 0] * inv,
                v[4 * i + 1] * inv,
                v[4 * i + 2] * inv,
                v[4 * i + 3] * inv);
        }
        ss = static_cast<unsigned>(k + 127);
    }
}

// Bit-identical to v28_mxfp8_reg_quad in qwen4_moe_compact_t_k32_v28.cu.
__device__ __forceinline__ void sw2_mxfp8_reg_quad(
    unsigned u0, unsigned u1, unsigned u2, unsigned u3,
    unsigned tig, unsigned (&qq)[2], unsigned& ss
) {
    const bool odd1 = (tig & 1u) != 0u;
    const unsigned s0 = __shfl_xor_sync(0xffffffffu, odd1 ? u0 : u1, 1);
    const unsigned s2 = __shfl_xor_sync(0xffffffffu, odd1 ? u2 : u3, 1);
    const unsigned t0 = odd1 ? s0 : u0;
    const unsigned t1 = odd1 ? u1 : s0;
    const unsigned t2 = odd1 ? s2 : u2;
    const unsigned t3 = odd1 ? u3 : s2;
    const bool odd2 = (tig & 2u) != 0u;
    const unsigned r0 = __shfl_xor_sync(0xffffffffu, odd2 ? t0 : t2, 2);
    const unsigned r1 = __shfl_xor_sync(0xffffffffu, odd2 ? t1 : t3, 2);
    const uint4 w = make_uint4(
        odd2 ? r0 : t0,
        odd2 ? r1 : t1,
        odd2 ? t2 : r0,
        odd2 ? t3 : r1
    );
    sw2_mxfp8_quad(w, qq, ss);
}

struct SharedGateUp2StageSmem {
    static constexpr unsigned STAGES = 2u;
    static constexpr unsigned BC = 128u;                    // 64 gate + 64 up B columns
    static constexpr unsigned A = STAGES * SW2_TILE_M * 64u; // 16,384 B
    static constexpr unsigned BT = STAGES * BC * 64u;       // 16,384 B
    static constexpr unsigned SF = STAGES * 2u * BC;        //    512 B
    static constexpr unsigned AS = 2u * SW2_TILE_M * 4u;    //  1,024 B
    static constexpr unsigned OFF_BT = A;
    static constexpr unsigned OFF_SF = OFF_BT + BT;
    static constexpr unsigned OFF_AS = OFF_SF + SF;
    static constexpr unsigned BYTES = OFF_AS + AS;
    static_assert(BYTES == 34304u, "2-stage shared_gateup SMEM byte contract");
    static_assert(BYTES <= 49152u, "2-stage shared_gateup fits 2 CTAs/SM (<= 48 KiB)");
};

__device__ __forceinline__ void shared_gateup_2stage_impl(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned char* __restrict__ bt_buf, const unsigned char* __restrict__ sf_buf,
    const float* __restrict__ s20, const float* __restrict__ s21,
    unsigned char* __restrict__ oq, unsigned char* __restrict__ os,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned N_in, unsigned K_in
) {
    using L = SharedGateUp2StageSmem;
    constexpr unsigned N = oi640::INTER;    // 640
    constexpr unsigned K = oi640::HIDDEN;   // 2560
    constexpr unsigned TN = 64u;
    constexpr unsigned n_tiles = N / TN;    // 10 (each 128-col pretranscode tile splits into 2x64)
    constexpr unsigned KS = K / 64u;        // 40
    constexpr unsigned kblocks = K / 32u;   // 80

    unsigned dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (gridDim.x == 0 || blockDim.x != 256u || dyn_smem < L::BYTES) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }
    if (N_in != N || K_in != K || oq == nullptr || os == nullptr || aq == nullptr ||
        as == nullptr || bt_buf == nullptr || sf_buf == nullptr || s20 == nullptr ||
        s21 == nullptr || expert_offsets == nullptr || sorted_token_ids == nullptr ||
        !oi640::contract_ok(contract) || !oi640::workspace_ok(workspace, contract) ||
        status == nullptr || status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_ABI);
        return;
    }

    const unsigned wc = workspace->work_count;
    const unsigned task_count = wc * n_tiles;
    if (blockIdx.x >= task_count) return;

    const unsigned tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31u;

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* const sA = smem;
    unsigned char* const sBt = smem + L::OFF_BT;
    unsigned char (*const sSF)[2][L::BC] =
        reinterpret_cast<unsigned char (*)[2][L::BC]>(smem + L::OFF_SF);
    unsigned (*const sAS)[SW2_TILE_M] =
        reinterpret_cast<unsigned (*)[SW2_TILE_M]>(smem + L::OFF_AS);

    // Keep SMEM base in a thread register (`[R]`) so sm_121a ptxas never folds `[R+UR0]` into
    // `LDGSTS ... desc[UR<odd>]`.
    unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n"
                 : "+r"(sbase));
    unsigned long long pol_el;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));

    const unsigned warp_n = (warp_id & 1u) * 32u;
    const unsigned gid = lane >> 2, tig = lane & 3u;

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const unsigned wi = task / n_tiles;
        const oi640::WorkItem item = workspace->items[wi];
        const unsigned nt10 = task % n_tiles;
        const unsigned nt5 = nt10 >> 1;
        const unsigned nhalf = nt10 & 1u;
        const unsigned cta_n = nt10 * TN;
        if (item.expert >= oi640::EXPERTS) {
            if (tid == 0) oi640::fail(status, oi640::ERR_PLAN);
            return;
        }
        const int m_start = expert_offsets[item.expert];
        const int M_expert = expert_offsets[item.expert + 1] - m_start;
        const int local = static_cast<int>(item.m_tile * SW2_TILE_M);
        const float s2a = s20[item.expert];
        const float s2b = s21[item.expert];
        if (m_start < 0 || M_expert < 0 || local >= M_expert || !isfinite(s2a) || !isfinite(s2b)) {
            if (tid == 0) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const unsigned cta_m = static_cast<unsigned>(m_start + local);
        const int valid_rows = min(M_expert - local, static_cast<int>(SW2_TILE_M));
        const unsigned active_rows_pad16 = (static_cast<unsigned>(valid_rows) + 15u) & ~15u;
        const bool m64 = (valid_rows <= 64);
        const unsigned warp_m = (warp_id >> 1) * (m64 ? 16u : 32u);
        const int warp_valid = valid_rows - static_cast<int>(warp_m);
        const int rt_active = warp_valid <= 0 ? 0 : (m64 ? 1 : min(2, (warp_valid + 15) >> 4));

        const unsigned a_r0 = tid >> 2, a_ch = tid & 3u;
        const unsigned a_r1 = a_r0 + 64u;
        const bool a_ok0 = static_cast<int>(a_r0) < valid_rows;
        const bool a_ok1 = static_cast<int>(a_r1) < valid_rows;
        const unsigned char* a_src0 =
            aq + static_cast<unsigned long long>(a_ok0 ? (cta_m + a_r0) : 0u) * K + a_ch * 16u;
        const unsigned char* a_src1 =
            aq + static_cast<unsigned long long>(a_ok1 ? (cta_m + a_r1) : 0u) * K + a_ch * 16u;
        const unsigned a_dst0 = sw2_swz(a_r0, a_ch);
        const unsigned sc_row = tid - 128u;
        const bool sc_ok = tid >= 128u && static_cast<int>(sc_row) < valid_rows;
        const unsigned char* sc_src =
            as + static_cast<unsigned long long>(sc_ok ? (cta_m + sc_row) : 0u) * kblocks;

        // Each 128-col pretranscode gate+up slice is 16,384 B:
        //   gate [0..64)  at +0,    gate [64..128) at +4096,
        //   up   [0..64)  at +8192, up   [64..128) at +12288.
        // Scale slice is 512 B (2 k32-halves x 256 B): nhalf 0 at +0, nhalf 1 at +128 per half.
        const unsigned char* bt_gate_base =
            bt_buf + static_cast<unsigned long long>(nt5 * KS) * 16384ull + nhalf * 4096u;
        const unsigned char* sf_base =
            sf_buf + static_cast<unsigned long long>(nt5 * KS) * 512ull + nhalf * 128u;

        auto load_a = [&](unsigned ks) {
            const unsigned slot = ks & 1u;
            unsigned char* const sA_slot = sA + slot * (SW2_TILE_M * 64u);
            if (a_r0 < active_rows_pad16) {
                sw2_cp16(sA_slot + a_dst0, a_src0 + ks * 64u, a_ok0);
            }
            if (a_r1 < active_rows_pad16) {
                sw2_cp16(sA_slot + a_dst0 + 4096u, a_src1 + ks * 64u, a_ok1);
            }
            if (tid >= 128u && (ks & 1u) == 0u && sc_row < active_rows_pad16) {
                sw2_cp4(&sAS[(ks >> 1) & 1u][sc_row], sc_src + ks * 2u, sc_ok);
            }
        };
        auto load_bt = [&](unsigned ks) {
            const unsigned slot = ks & 1u;
            const unsigned char* b_gate_src =
                bt_gate_base + static_cast<unsigned long long>(ks) * 16384ull;
            const unsigned char* b_up_src = b_gate_src + 8192u;
            const unsigned b_dst_s = sbase + L::OFF_BT + slot * 8192u;
            sw2_cp16_l2_el_s32(b_dst_s + tid * 16u, b_gate_src + tid * 16u, pol_el);
            sw2_cp16_l2_el_s32(b_dst_s + 4096u + tid * 16u, b_up_src + tid * 16u, pol_el);
            if (tid < 16u) {
                const unsigned t_sf = tid >> 3, c_sf = tid & 7u;
                sw2_cp16_l2_el_s32(sbase + L::OFF_SF + (slot * 2u + t_sf) * 128u + c_sf * 16u,
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
        sw2_commit();

        unsigned sfa[2] = {0u, 0u};
        auto step = [&](auto par_tag, unsigned ks) {
            constexpr unsigned PAR = decltype(par_tag)::value;   // ks & 1 == slot
            sw2_wait<0>();
            __syncthreads();
            if (ks + 1u < KS) {
                load_a(ks + 1u);
                load_bt(ks + 1u);
            }
            sw2_commit();
            if (rt_active == 0) return;
            if constexpr (PAR == 0u) {
                const unsigned as_buf = (ks >> 1) & 1u;
                sfa[0] = sAS[as_buf][warp_m + gid + (tig & 1u) * 8u];
                if (rt_active > 1) sfa[1] = sAS[as_buf][warp_m + 16u + gid + (tig & 1u) * 8u];
            }
            #pragma unroll
            for (unsigned t = 0u; t < 2u; ++t) {
                const uint2 sfb = *reinterpret_cast<const uint2*>(
                    &sSF[PAR][t][((warp_id & 1u) * 8u + gid) << 3]);
                const unsigned row_a0 = warp_m + (lane & 15u);
                const unsigned addr_a0 = static_cast<unsigned>(__cvta_generic_to_shared(
                    sA + PAR * (SW2_TILE_M * 64u) + sw2_swz(row_a0, 2u * t + (lane >> 4))));
                const unsigned row_b0 = warp_n + ((lane >> 4) << 3) + (lane & 7u);
                const unsigned addr_b0 = static_cast<unsigned>(__cvta_generic_to_shared(
                    sBt + PAR * (L::BC * 64u) + sw2_swz(row_b0, 2u * t + ((lane >> 3) & 1u))));
                unsigned afr0[4], afr1[4];
                auto emit_pair = [&](auto p_tag, const unsigned (&bf)[4], unsigned sfb_w) {
                    constexpr int P = decltype(p_tag)::value;
                    constexpr int BIDB0 = (P & 1) * 2;
                    constexpr int BIDB1 = BIDB0 + 1;
                    if (t == 0u) {
                        sw2_qmma<2 * PAR, BIDB0>(acc[0][2 * P], afr0, bf[0], bf[1], sfa[0], sfb_w);
                        sw2_qmma<2 * PAR, BIDB1>(acc[0][2 * P + 1], afr0, bf[2], bf[3], sfa[0], sfb_w);
                        if (rt_active > 1) {
                            sw2_qmma<2 * PAR, BIDB0>(acc[1][2 * P], afr1, bf[0], bf[1], sfa[1], sfb_w);
                            sw2_qmma<2 * PAR, BIDB1>(acc[1][2 * P + 1], afr1, bf[2], bf[3], sfa[1], sfb_w);
                        }
                    } else {
                        sw2_qmma<2 * PAR + 1, BIDB0>(acc[0][2 * P], afr0, bf[0], bf[1], sfa[0], sfb_w);
                        sw2_qmma<2 * PAR + 1, BIDB1>(acc[0][2 * P + 1], afr0, bf[2], bf[3], sfa[0], sfb_w);
                        if (rt_active > 1) {
                            sw2_qmma<2 * PAR + 1, BIDB0>(acc[1][2 * P], afr1, bf[0], bf[1], sfa[1], sfb_w);
                            sw2_qmma<2 * PAR + 1, BIDB1>(acc[1][2 * P + 1], afr1, bf[2], bf[3], sfa[1], sfb_w);
                        }
                    }
                };
                unsigned bfr[2][4];
                sw2_ldsm4(addr_b0, bfr[0][0], bfr[0][1], bfr[0][2], bfr[0][3]);
                sw2_ldsm4(addr_a0, afr0[0], afr0[1], afr0[2], afr0[3]);
                if (rt_active > 1) {
                    sw2_ldsm4(addr_a0 + 1024u, afr1[0], afr1[1], afr1[2], afr1[3]);
                }
                sw2_ldsm4(addr_b0 + 1024u, bfr[1][0], bfr[1][1], bfr[1][2], bfr[1][3]);
                emit_pair(std::integral_constant<int, 0>{}, bfr[0], sfb.x);
                sw2_ldsm4(addr_b0 + 4096u, bfr[0][0], bfr[0][1], bfr[0][2], bfr[0][3]);
                emit_pair(std::integral_constant<int, 1>{}, bfr[1], sfb.x);
                sw2_ldsm4(addr_b0 + 5120u, bfr[1][0], bfr[1][1], bfr[1][2], bfr[1][3]);
                emit_pair(std::integral_constant<int, 2>{}, bfr[0], sfb.y);
                emit_pair(std::integral_constant<int, 3>{}, bfr[1], sfb.y);
            }
        };
        #pragma unroll 1
        for (unsigned ks = 0u; ks < KS; ks += 2u) {
            step(std::integral_constant<unsigned, 0u>{}, ks);
            step(std::integral_constant<unsigned, 1u>{}, ks + 1u);
        }
        sw2_wait<0>();

        const unsigned blk = warp_id & 1u;
        const unsigned col8 = warp_n + tig * 8u;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (r >= rt_active) break;
            const unsigned lr0 = warp_m + r * 16u + gid, lr1 = lr0 + 8u;
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
                u0[j] = sw2_pack_bf162(v[0], v[1]);
                u1[j] = sw2_pack_bf162(v[2], v[3]);
            }
            unsigned qq0[2], ss0, qq1[2], ss1;
            sw2_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], tig, qq0, ss0);
            if (static_cast<int>(lr0) < valid_rows) {
                const unsigned long long orow = cta_m + lr0;
                sw2_st8_evict_last(oq + orow * N + cta_n + col8, make_uint2(qq0[0], qq0[1]));
                if (tig == 0u) os[orow * (N / 32u) + cta_n / 32u + blk] = static_cast<unsigned char>(ss0);
            }
            sw2_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], tig, qq1, ss1);
            if (static_cast<int>(lr1) < valid_rows) {
                const unsigned long long orow = cta_m + lr1;
                sw2_st8_evict_last(oq + orow * N + cta_n + col8, make_uint2(qq1[0], qq1[1]));
                if (tig == 0u) os[orow * (N / 32u) + cta_n / 32u + blk] = static_cast<unsigned char>(ss1);
            }
        }
        if (task + gridDim.x < task_count) __syncthreads();
    }
}

} // namespace

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_shared_w4a8_gateup_k32_v28_2stage(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned char* __restrict__ bt_gateup, const unsigned char* __restrict__ sf_gateup,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q,
    unsigned char* __restrict__ out_s, const float* __restrict__ up_scale2_vals,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    shared_gateup_2stage_impl(aq, as, bt_gateup, sf_gateup, gate_scale2_vals, up_scale2_vals,
                              out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
                              status, N, K);
}
