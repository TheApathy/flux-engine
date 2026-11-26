// SPDX-License-Identifier: AGPL-3.0-only
//
// V12: V10 with two k32 QMMA steps per barrier (k64 stages; A ring 2 stages loaded one iteration
// ahead, raw B ring 2 stages loaded two ahead, transcoded B double-buffered; 47.9 KB smem, 2 CTAs/SM).
// Bit-identical to V10/V9 (standalone, 81920/81920 rows). Selected by ATLAS_QWEN4_PREFILL_MOE_W4A8_K64=1.
//
// Numerics and pipeline (shared with V10, widened to k64 stages):
//  * MXFP8 activations, NVFP4 weights rounded to E4M3 payloads under a UE8M0 per-32 scale,
//    QMMA.SF e4m3 x e4m3, using V7's 128-row worklist planner;
//  * cooperative weight transcode runs once per CTA k64 stage (integer PRMT LUT into swizzled E4M3
//    SMEM for ldmatrix), overlapping stage k+1's transcode with stage k's two k32 QMMA steps
//    (one __syncthreads() barrier per k64 stage);
//  * gate/up writes the down operand directly as MXFP8 (bf16-rounded SwiGLU + qwen4_mxfp8_quant_rows
//    quantisation) from a shared-memory staging tile, avoiding a separate BF16 intermediate.
// There is no exact-weight W4A8 on this MMA: e2m1 x e4m3 needs up to 6 significant bits.
#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define OI640_STEP_K 32
#define OI640_TRANSPOSED 1
#define flash_next_orig_i640_prefill flash_next_f32_transposed_compact_k32_v12
#include "moe_w4a16_orig_i640_compact_prefill.cuh"
#include <type_traits>
namespace oi640 = flash_next_f32_transposed_compact_k32_v12;

namespace {
constexpr unsigned V10_TILE_M = 128;
constexpr unsigned V10_BCOLS = 128;     // transcoded B rows (columns) per CTA, both kernels
constexpr unsigned V10_E_STRIDE = 72;   // bf16 per staging row in the gate/up epilogue (64 + 8 pad)

__device__ __forceinline__ void v10_cp16(void* smem, const void* gmem, bool pred) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void v10_cp4(void* smem, const void* gmem, bool pred) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(s), "l"(gmem), "r"(pred ? 4 : 0));
}
__device__ __forceinline__ void v10_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void v10_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ unsigned v10_prmt(unsigned a, unsigned b, unsigned sel) {
    unsigned d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
}
__device__ __forceinline__ int v10_floor_log2_e4m3(unsigned b) {
    const unsigned e = (b >> 3) & 0xfu, m = b & 7u;
    if (e != 0) return static_cast<int>(e) - 7;
    if (m == 0) return -(1 << 30);
    return (31 - __clz(static_cast<int>(m))) - 9;
}
__device__ __forceinline__ float v10_e4m3_to_float(unsigned b) {
    unsigned h;
    const unsigned short pair = static_cast<unsigned short>(b & 0xffu);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(pair));
    return __low2float(*reinterpret_cast<const __half2*>(&h));
}
// Four E2M1 codes (x: 16 bits, code i in nibble i) -> four E4M3 bytes via the 8-byte magnitude
// table {tlo, thi}; the sign bit of code i becomes bit 7 of byte i.
__device__ __forceinline__ unsigned v10_e2m1x4_to_e4m3x4(unsigned x, unsigned tlo, unsigned thi) {
    const unsigned mag = v10_prmt(tlo, thi, x & 0x7777u);
    const unsigned sgn = v10_prmt(x, x << 4, 0x9D8Cu);
    return mag | (sgn & 0x80808080u);
}

// b12x quantize_block_fp8_mx on 32 values (bit-identical to qwen4_mxfp8_quant_rows).
__device__ __forceinline__ void v10_mxfp8_block(const float (&v)[32], unsigned (&q)[8], unsigned& s) {
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
    if (!(amax > 0.0f)) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) q[i] = 0u;
        s = 0;
        return;
    }
    int e;
    const float m = frexpf(__fmul_rn(amax, 1.0f / 448.0f), &e);
    int k = (m == 0.5f) ? e - 1 : e;
    k = max(-127, min(127, k));
    const float inv = __int_as_float((127 - k) << 23);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned w = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            w |= static_cast<unsigned>(__nv_cvt_float_to_fp8(v[4 * i + j] * inv, __NV_SATFINITE, __NV_E4M3)) << (8 * j);
        q[i] = w;
    }
    s = static_cast<unsigned>(k + 127);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 1200
__device__ __forceinline__ float v12_scale_ue8m0_pair(float t, unsigned ea, unsigned eb) {
    const unsigned s = ea + eb;
    const unsigned h1 = s >> 1;
    const unsigned h2 = s - h1;
    const unsigned fa_bits = (ea == 255u || eb == 255u) ? 0x7fc00000u : (h1 << 23);
    return (t * __uint_as_float(fa_bits)) * __uint_as_float(h2 << 23);
}
#endif

template <int BID>
__device__ __forceinline__ void v10_qmma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1,
                                         unsigned sa, unsigned sb) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%14, %13};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb),
          "n"(BID), "n"(0), "n"(0));
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
    const unsigned sa_b = (sa >> (BID * 8)) & 0xffu;
    const unsigned sb_b = sb & 0xffu;
    const unsigned ea_lo = __shfl_sync(0xffffffffu, sa_b, grp);
    const unsigned ea_hi = __shfl_sync(0xffffffffu, sa_b, grp | 1u);
    const unsigned eb_even = __shfl_sync(0xffffffffu, sb_b, tig << 3);
    const unsigned eb_odd = __shfl_sync(0xffffffffu, sb_b, (tig << 3) | 4u);
    d[0] += v12_scale_ue8m0_pair(t0, ea_lo, eb_even);
    d[1] += v12_scale_ue8m0_pair(t1, ea_lo, eb_odd);
    d[2] += v12_scale_ue8m0_pair(t2, ea_hi, eb_even);
    d[3] += v12_scale_ue8m0_pair(t3, ea_hi, eb_odd);
#endif
}
__device__ __forceinline__ void v10_ldsm4(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
} // namespace

// MXFP8 rows: q[r][k] = e4m3_rn(x / 2^E), s[r][k/32] = E + 127 with 2^E = pow2ceil(amax/448)
// per 32-element block (b12x quantize_block_fp8_mx; all-zero block -> byte 0, payload 0).
// One thread per block, 16-byte loads and stores. `cols` = K (multiple of 32).
extern "C" __global__ void qwen4_mxfp8_quant_rows_v12(
    const __nv_bfloat16* __restrict__ src, unsigned char* __restrict__ q, unsigned char* __restrict__ s,
    unsigned int rows, unsigned int cols
) {
    const unsigned long long blocks = static_cast<unsigned long long>(rows) * (cols / 32);
    for (unsigned long long b = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
         b < blocks; b += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
        const uint4* x = reinterpret_cast<const uint4*>(src + b * 32);
        float v[32];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint4 w = x[c];
            const unsigned u[4] = {w.x, w.y, w.z, w.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[8 * c + 2 * i] = __uint_as_float(u[i] << 16);
                v[8 * c + 2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
            }
        }
        unsigned qq[8], ss;
        v10_mxfp8_block(v, qq, ss);
        uint4* out = reinterpret_cast<uint4*>(q + b * 32);
        out[0] = make_uint4(qq[0], qq[1], qq[2], qq[3]);
        out[1] = make_uint4(qq[4], qq[5], qq[6], qq[7]);
        s[b] = static_cast<unsigned char>(ss);
    }
}


// 64-byte smem rows (four 16-byte chunks): XOR the chunk with bits 1..2 of the row so an ldmatrix
// 8-row phase touches 8 distinct 16-byte bank groups.
__device__ __forceinline__ unsigned v12_swz(unsigned row, unsigned chunk) {
    return row * 64u + ((chunk ^ ((row >> 1) & 3u)) << 4);
}

// V12 mainloop: one barrier per k64 stage (two k32 QMMA steps). A ring: 2 stages, loaded one
// iteration ahead; raw B ring: 2 stages, loaded two iterations ahead of their MMA (one ahead of
// their transcode); transcoded B: 2 buffers. Iteration i: wait (A i, raw B i+1 landed) -> barrier
// -> issue A i+1 + raw B i+2 -> transcode B i+1 -> MMA i.
template <bool GATEUP>
__device__ __forceinline__ void v12_impl(
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
    constexpr unsigned TN = GATEUP ? 64u : 128u;
    constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    constexpr unsigned BCOLS = GATEUP ? 64u : 128u;
    constexpr unsigned B_STRIDE = BCOLS + 16;
    static_assert(BMAT * BCOLS == V10_BCOLS, "transcoded tile is 128 columns");
    if (gridDim.x != oi640::FIXED_GRID || blockDim.x != 256) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }
    const bool shape = GATEUP ? (N == oi640::INTER && K == oi640::HIDDEN)
                              : (N == oi640::HIDDEN && K == oi640::INTER);
    const bool outs = GATEUP ? (oq != nullptr && os != nullptr) : (C != nullptr);
    const bool tbl_ok = pk0 != nullptr && sc0 != nullptr && s20 != nullptr &&
        (!GATEUP || (pk1 != nullptr && sc1 != nullptr && s21 != nullptr));
    if (!shape || !outs || !tbl_ok || aq == nullptr || as == nullptr || expert_offsets == nullptr ||
        sorted_token_ids == nullptr || !oi640::contract_ok(contract) ||
        !oi640::workspace_ok(workspace, contract) || status == nullptr || status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_ABI);
        return;
    }
    constexpr unsigned SZ_A = 2 * V10_TILE_M * 64;
    constexpr unsigned SZ_B = 2 * BMAT * 32 * B_STRIDE;
    constexpr unsigned SZ_S = 2 * BMAT * 4 * B_STRIDE;
    constexpr unsigned SZ_AS = 2 * V10_TILE_M * 4;
    constexpr unsigned SZ_BT = 2 * V10_BCOLS * 64;
    constexpr unsigned SZ_SF = 2 * 2 * V10_BCOLS;
    constexpr unsigned OFF_B = SZ_A, OFF_S = OFF_B + SZ_B, OFF_AS = OFF_S + SZ_S, OFF_BT = OFF_AS + SZ_AS,
                       OFF_SF = OFF_BT + SZ_BT, OFF_LUT = OFF_SF + SZ_SF, SZ_ALL = OFF_LUT + 256 * 8;
    static_assert(V10_TILE_M * V10_E_STRIDE * 2 <= OFF_AS, "epilogue staging tile aliases A/B/S only");
    __shared__ __align__(16) unsigned char smem[SZ_ALL];
    unsigned char* const sA = smem;                                          // [2][128][64] swizzled
    unsigned char (*const sB)[BMAT][32][B_STRIDE] =
        reinterpret_cast<unsigned char (*)[BMAT][32][B_STRIDE]>(smem + OFF_B);
    unsigned char (*const sS)[BMAT][4][B_STRIDE] =
        reinterpret_cast<unsigned char (*)[BMAT][4][B_STRIDE]>(smem + OFF_S);
    unsigned (*const sAS)[V10_TILE_M] = reinterpret_cast<unsigned (*)[V10_TILE_M]>(smem + OFF_AS);
    unsigned char* const sBt = smem + OFF_BT;                                // [2][128][64] swizzled
    unsigned char (*const sSF)[2][V10_BCOLS] = reinterpret_cast<unsigned char (*)[2][V10_BCOLS]>(smem + OFF_SF);
    uint2* const lut = reinterpret_cast<uint2*>(smem + OFF_LUT);
    __nv_bfloat16* const sE = reinterpret_cast<__nv_bfloat16*>(smem);        // epilogue only

    const unsigned tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    {
        const float r = v10_e4m3_to_float(tid);
        const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        unsigned w[2] = {0u, 0u};
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            w[i >> 2] |= static_cast<unsigned>(__nv_cvt_float_to_fp8(mags[i] * r, __NV_SATFINITE, __NV_E4M3)) << (8 * (i & 3));
        lut[tid] = make_uint2(w[0], w[1]);
    }

    const unsigned n_tiles = N / TN;
    const unsigned task_count = workspace->work_count * n_tiles;
    const unsigned warp_m = (warp_id >> 2) * 64;
    const unsigned warp_n = (warp_id & 3) * (GATEUP ? 16u : 32u);
    const unsigned gid = lane >> 2, tig = lane & 3;
    const unsigned KS = K / 64;                                    // k64 stages (even: 40 or 10)
    const unsigned kblocks = K / 32;
    const unsigned t_col = tid % BCOLS, t_h = (tid / BCOLS) & 1u, t_mtx = tid / (2 * BCOLS);
    const unsigned t_row = t_mtx * BCOLS + t_col;

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const oi640::WorkItem item = workspace->items[task / n_tiles];
        const unsigned cta_n = (task % n_tiles) * TN;
        if (item.expert >= oi640::EXPERTS) { if (tid == 0) oi640::fail(status, oi640::ERR_PLAN); return; }
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
        const int warp_valid = valid_rows - static_cast<int>(warp_m);
        const int rt_active = warp_valid <= 0 ? 0 : min(4, (warp_valid + 15) >> 4);

        // A: 128 rows x 64 bytes per stage = 512 chunks, two per thread (rows tid>>2 and +64).
        const unsigned a_row0 = tid >> 2, a_ch = tid & 3;
        const unsigned char* a_src[2];
        bool a_ok[2];
        unsigned a_dst[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned row = a_row0 + 64 * i;
            a_ok[i] = static_cast<int>(row) < valid_rows;
            const unsigned src_row = a_ok[i] ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + row])
                                                       : cta_m + row) : 0u;
            a_src[i] = aq + static_cast<unsigned long long>(src_row) * K + a_ch * 16;
            a_dst[i] = v12_swz(row, a_ch);
        }
        const unsigned sc_row = tid - 128;
        const bool sc_ok = tid >= 128 && static_cast<int>(sc_row) < valid_rows;
        const unsigned sc_src_row = sc_ok ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + sc_row])
                                                    : cta_m + sc_row) : 0u;
        const unsigned char* sc_src = as + static_cast<unsigned long long>(sc_src_row) * kblocks;

        auto load_a = [&](unsigned ks) {
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                v10_cp16(sA + (ks & 1) * (V10_TILE_M * 64) + a_dst[i], a_src[i] + ks * 64, a_ok[i]);
            if (tid >= 128 && (ks & 1) == 0)      // four k32 scale bytes cover stages ks, ks+1
                v10_cp4(&sAS[(ks >> 1) & 1][sc_row], sc_src + ks * 2, sc_ok);
        };
        auto load_b = [&](unsigned ks) {
            constexpr unsigned CH = BCOLS / 16;
            const unsigned mtx = tid / (32 * CH), r = (tid / CH) % 32, cc = tid % CH;
            const unsigned char* src = mtx ? B1 : B0;
            v10_cp16(&sB[ks & 1][mtx][r][cc * 16],
                     src + static_cast<unsigned long long>(ks * 32 + r) * N + cta_n + cc * 16, true);
            if (tid < BMAT * 4 * CH) {
                const unsigned m2 = tid / (4 * CH), r2 = (tid / CH) % 4, c2 = tid % CH;
                const unsigned char* s = m2 ? S1 : S0;
                v10_cp16(&sS[ks & 1][m2][r2][c2 * 16],
                         s + static_cast<unsigned long long>(ks * 4 + r2) * N + cta_n + c2 * 16, true);
            }
        };
        // raw B stage slot `sl` -> transcoded buffer `tb`: this thread's column, block t_h of both k32 steps.
        auto transcode = [&](unsigned sl, unsigned tb) {
            #pragma unroll
            for (unsigned t = 0; t < 2; ++t) {
                const unsigned s_own = sS[sl][t_mtx][2 * t + t_h][t_col], s_oth = sS[sl][t_mtx][2 * t + (t_h ^ 1)][t_col];
                int e32 = max(v10_floor_log2_e4m3(s_own), v10_floor_log2_e4m3(s_oth));
                if (e32 < -(1 << 29)) e32 = 0;
                const float inv = __int_as_float((127 - e32) << 23);
                const unsigned r = __nv_cvt_float_to_fp8(v10_e4m3_to_float(s_own) * inv, __NV_SATFINITE, __NV_E4M3);
                const uint2 tt = lut[r];
                unsigned w[4];
                #pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const unsigned x = static_cast<unsigned>(sB[sl][t_mtx][16 * t + 8 * t_h + 2 * g][t_col]) |
                                       (static_cast<unsigned>(sB[sl][t_mtx][16 * t + 8 * t_h + 2 * g + 1][t_col]) << 8);
                    w[g] = v10_e2m1x4_to_e4m3x4(x, tt.x, tt.y);
                }
                *reinterpret_cast<uint4*>(sBt + tb * (V10_BCOLS * 64) + v12_swz(t_row, 2 * t + t_h)) =
                    make_uint4(w[0], w[1], w[2], w[3]);
                if (t_h == 0) sSF[tb][t][t_row] = static_cast<unsigned char>((e32 + 127) & 0xff);
            }
        };

        float acc[4][4][4];
        #pragma unroll
        for (int r = 0; r < 4; ++r)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.0f;

        // prologue: B0 -> wait -> transcode 0; A0 + B1 in flight.
        load_b(0);
        v10_commit();
        load_a(0);
        if (KS > 1) load_b(1);
        v10_commit();
        v10_wait<1>();
        __syncthreads();
        transcode(0, 0);

        auto step = [&](auto par_tag, unsigned ks) {
            constexpr unsigned PAR = decltype(par_tag)::value;   // ks & 1
            v10_wait<0>();                                       // A ks, raw B ks+1 landed
            __syncthreads();
            if (ks + 1 < KS) load_a(ks + 1);
            if (ks + 2 < KS) load_b(ks + 2);
            v10_commit();
            if (ks + 1 < KS) transcode((ks + 1) & 1, PAR ^ 1);
            if (rt_active == 0) return;
            const unsigned as_buf = (ks >> 1) & 1;
            #pragma unroll
            for (unsigned t = 0; t < 2; ++t) {
                constexpr int dummy = 0; (void)dummy;
                unsigned bfr[4][2], sfb[4];
                #pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const unsigned rbase = GATEUP ? p * BCOLS + warp_n : warp_n + p * 16;
                    const unsigned row = rbase + ((lane >> 4) << 3) + (lane & 7);
                    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(
                        sBt + PAR * (V10_BCOLS * 64) + v12_swz(row, 2 * t + ((lane >> 3) & 1))));
                    v10_ldsm4(addr, bfr[2 * p][0], bfr[2 * p][1], bfr[2 * p + 1][0], bfr[2 * p + 1][1]);
                    sfb[2 * p] = sSF[PAR][t][rbase + gid];
                    sfb[2 * p + 1] = sSF[PAR][t][rbase + 8 + gid];
                }
                unsigned afr[4][4], sfa[4];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const unsigned row = warp_m + r * 16 + (lane & 15);
                    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(
                        sA + PAR * (V10_TILE_M * 64) + v12_swz(row, 2 * t + (lane >> 4))));
                    v10_ldsm4(addr, afr[r][0], afr[r][1], afr[r][2], afr[r][3]);
                    sfa[r] = sAS[as_buf][warp_m + r * 16 + gid + (tig & 1) * 8];
                }
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    if (r < rt_active) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            if (t == 0) v10_qmma<2 * PAR>(acc[r][j], afr[r], bfr[j][0], bfr[j][1], sfa[r], sfb[j]);
                            else        v10_qmma<2 * PAR + 1>(acc[r][j], afr[r], bfr[j][0], bfr[j][1], sfa[r], sfb[j]);
                        }
                    }
                }
            }
        };
        for (unsigned ks = 0; ks < KS; ks += 2) {
            step(std::integral_constant<unsigned, 0>{}, ks);
            step(std::integral_constant<unsigned, 1>{}, ks + 1);
        }
        v10_wait<0>();
        if (GATEUP) {
            __syncthreads();                       // all MMA smem reads done before staging aliases A/B
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (r >= rt_active) break;
                const unsigned lr0 = warp_m + r * 16 + gid, lr1 = lr0 + 8;
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const unsigned c0 = warp_n + j * 8 + tig * 2;
                    float v[4];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const float g = acc[r][j][q] * s2a;
                        const float u = acc[r][j + 2][q] * s2b;
                        v[q] = g * (1.0f / (1.0f + __expf(-g))) * u;
                    }
                    *reinterpret_cast<__nv_bfloat162*>(sE + lr0 * V10_E_STRIDE + c0) = __floats2bfloat162_rn(v[0], v[1]);
                    *reinterpret_cast<__nv_bfloat162*>(sE + lr1 * V10_E_STRIDE + c0) = __floats2bfloat162_rn(v[2], v[3]);
                }
            }
            __syncthreads();
            const unsigned row = tid >> 1, blk = tid & 1;
            if (static_cast<int>(row) < valid_rows) {
                const uint4* e = reinterpret_cast<const uint4*>(sE + row * V10_E_STRIDE + blk * 32);
                float v[32];
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint4 w = e[c];
                    const unsigned u[4] = {w.x, w.y, w.z, w.w};
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        v[8 * c + 2 * i] = __uint_as_float(u[i] << 16);
                        v[8 * c + 2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
                    }
                }
                unsigned qq[8], ss;
                v10_mxfp8_block(v, qq, ss);
                const unsigned long long orow = cta_m + row;
                uint4* o = reinterpret_cast<uint4*>(oq + orow * N + cta_n + blk * 32);
                o[0] = make_uint4(qq[0], qq[1], qq[2], qq[3]);
                o[1] = make_uint4(qq[4], qq[5], qq[6], qq[7]);
                os[orow * (N / 32) + cta_n / 32 + blk] = static_cast<unsigned char>(ss);
            }
        } else {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (r >= rt_active) break;
                const unsigned lr0 = warp_m + r * 16 + gid, lr1 = lr0 + 8;
                const bool r0v = static_cast<int>(lr0) < valid_rows, r1v = static_cast<int>(lr1) < valid_rows;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned c0 = cta_n + warp_n + j * 8 + tig * 2;
                    if (r0v) *reinterpret_cast<__nv_bfloat162*>(C + static_cast<unsigned long long>(cta_m + lr0) * N + c0) = __floats2bfloat162_rn(acc[r][j][0] * s2a, acc[r][j][1] * s2a);
                    if (r1v) *reinterpret_cast<__nv_bfloat162*>(C + static_cast<unsigned long long>(cta_m + lr1) * N + c0) = __floats2bfloat162_rn(acc[r][j][2] * s2a, acc[r][j][3] * s2a);
                }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_gateup_k32_v12(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v12_impl<true>(aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
                   up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
                   status, N, K);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_k32_v12(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v12_impl<false>(aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
                    expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}
