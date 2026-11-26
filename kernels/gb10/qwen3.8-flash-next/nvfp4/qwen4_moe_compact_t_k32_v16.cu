// SPDX-License-Identifier: AGPL-3.0-only
//
// V16: W4A8 compact routed-MoE kernels (ATLAS_QWEN4_PREFILL_MOE_W4A8_WIDE=2; 128 x 256 B-column
// tile per 512-thread CTA, 1 CTA/SM, dynamic smem, 3-deep A and raw-B rings, k64 stage cadence).
// Bit-identical to V14 (same per-output k order and QMMA.SF e4m3 x e4m3 sequence, V7 128-row worklist):
//  * cooperative integer PRMT-LUT weight transcode runs once per CTA k64 stage (warps 8-15 when
//    valid_rows <= 64, all 16 warps when valid_rows > 64) into swizzled E4M3 SMEM (`sBt`) + packed
//    `sSF` scales (`LDS.64`), overlapping stage k+1's transcode with stage k's two k32 QMMA steps
//    (one __syncthreads() barrier per k64 stage);
//  * zero-SMEM register-direct epilogues (`(void)sE;` — warp-shuffle SwiGLU + MXFP8 quantisation
//    for gate/up, and coalesced 16-byte register-direct BF16 stores for down);
//  * also exports the MXFP8 row quantisers (`qwen4_mxfp8_quant_rows_v16`, quad-lane `_v16q4`, and
//    fused shared-gate precomputation `_v16q4_gate`).
// There is no exact-weight W4A8 on this MMA: e2m1 x e4m3 needs up to 6 significant bits.
#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define OI640_STEP_K 32
#define OI640_TRANSPOSED 1
#define flash_next_orig_i640_prefill flash_next_f32_transposed_compact_k32_v16
#include "moe_w4a16_orig_i640_compact_prefill.cuh"
#include <type_traits>
namespace oi640 = flash_next_f32_transposed_compact_k32_v16;

namespace {
constexpr unsigned V10_TILE_M = 128;

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

__device__ __forceinline__ unsigned v10_pack_e4m3x4(float a, float b, float c, float d) {
    unsigned short p0, p1;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p0) : "f"(a), "f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p1) : "f"(c), "f"(d));
    return v10_prmt(static_cast<unsigned>(p0), static_cast<unsigned>(p1), 0x5410u);
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
__device__ __forceinline__ float v16_scale_ue8m0_pair(float t, unsigned ea, unsigned eb) {
    const unsigned s = ea + eb;
    const unsigned h1 = s >> 1;
    const unsigned h2 = s - h1;
    const unsigned fa_bits = (ea == 255u || eb == 255u) ? 0x7fc00000u : (h1 << 23);
    return (t * __uint_as_float(fa_bits)) * __uint_as_float(h2 << 23);
}
#endif

template <int BID, int SFI_B = 0>
__device__ __forceinline__ void v16_qmma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1,
                                         unsigned sa, unsigned sb) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%14, %13};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb),
          "n"(BID), "n"(0), "n"(SFI_B));
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
    const unsigned sb_b = (sb >> (SFI_B * 8)) & 0xffu;
    const unsigned ea_lo = __shfl_sync(0xffffffffu, sa_b, grp);
    const unsigned ea_hi = __shfl_sync(0xffffffffu, sa_b, grp | 1u);
    const unsigned eb_even = __shfl_sync(0xffffffffu, sb_b, tig << 3);
    const unsigned eb_odd = __shfl_sync(0xffffffffu, sb_b, (tig << 3) | 4u);
    d[0] += v16_scale_ue8m0_pair(t0, ea_lo, eb_even);
    d[1] += v16_scale_ue8m0_pair(t1, ea_lo, eb_odd);
    d[2] += v16_scale_ue8m0_pair(t2, ea_hi, eb_even);
    d[3] += v16_scale_ue8m0_pair(t3, ea_hi, eb_odd);
#endif
}
template <int BID>
__device__ __forceinline__ void v16_qmma4_lo(float (&d)[8][4], const unsigned (&a)[4], const unsigned (&bfr)[8][2],
                                             unsigned sa, unsigned lo) {
    v16_qmma<BID, 0>(d[0], a, bfr[0][0], bfr[0][1], sa, lo);
    v16_qmma<BID, 1>(d[1], a, bfr[1][0], bfr[1][1], sa, lo);
    v16_qmma<BID, 2>(d[2], a, bfr[2][0], bfr[2][1], sa, lo);
    v16_qmma<BID, 3>(d[3], a, bfr[3][0], bfr[3][1], sa, lo);
}
template <int BID>
__device__ __forceinline__ void v16_qmma4_hi(float (&d)[8][4], const unsigned (&a)[4], const unsigned (&bfr)[8][2],
                                             unsigned sa, unsigned hi) {
    v16_qmma<BID, 0>(d[4], a, bfr[4][0], bfr[4][1], sa, hi);
    v16_qmma<BID, 1>(d[5], a, bfr[5][0], bfr[5][1], sa, hi);
    v16_qmma<BID, 2>(d[6], a, bfr[6][0], bfr[6][1], sa, hi);
    v16_qmma<BID, 3>(d[7], a, bfr[7][0], bfr[7][1], sa, hi);
}
template <bool GATEUP>
__device__ __forceinline__ unsigned v16_sf_index(unsigned col) {
    const unsigned gid = col & 7u;
    if constexpr (GATEUP) {
        const unsigned mtx = col >> 7;
        const unsigned wn = (col >> 5) & 3u;
        const unsigned p_low = (col >> 4) & 1u;
        const unsigned sub = (col >> 3) & 1u;
        const unsigned j = (mtx << 2) | (p_low << 1) | sub;
        return (wn << 6) | (gid << 3) | j;
    } else {
        const unsigned wn = col >> 6;
        const unsigned j = (col >> 3) & 7u;
        return (wn << 6) | (gid << 3) | j;
    }
}
__device__ __forceinline__ void v10_ldsm4(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
} // namespace

// MXFP8 rows: q[r][k] = e4m3_rn(x / 2^E), s[r][k/32] = E + 127 with 2^E = pow2ceil(amax/448)
// per 32-element block (b12x quantize_block_fp8_mx; all-zero block -> byte 0, payload 0).
// One thread per block, 16-byte loads and stores. `cols` = K (multiple of 32).
extern "C" __global__ void qwen4_mxfp8_quant_rows_v16(
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

// Quad-lane twin of qwen4_mxfp8_quant_rows_v16 (ATLAS_QWEN4_PREFILL_MOE_SLOT_BF16_FUSED=1). Four adjacent
// lanes share one MX32 block: each lane loads its 16 bytes (8 bf16), the block absmax is the quad's fmaxf
// reduction (max over the same 32 magnitudes starting from 0 is exact and order-free, so it equals v16's
// serial chain bit for bit), every lane converts its own 8 values with the block's scale and stores 8 bytes,
// and lane 0 of the quad writes the UE8M0 byte. Per warp and 8 blocks: one contiguous 512-byte load and one
// contiguous 256-byte store (v16: four 16-byte loads per lane at a 64-byte lane stride, half of every sector
// used per instruction). Same scale rule (2^E = pow2ceil(amax/448) via frexpf, E clamped to [-127, 127],
// all-zero block -> byte 0 / payload 0) and the same e4m3 RN-satfinite per value -> payload and scale bytes
// identical to v16. Launch: any grid, blockDim a multiple of 32 (grid-strides over warps).
extern "C" __global__ void qwen4_mxfp8_quant_rows_v16q4(
    const __nv_bfloat16* __restrict__ src, unsigned char* __restrict__ q, unsigned char* __restrict__ s,
    unsigned int rows, unsigned int cols
) {
    const unsigned long long blocks = static_cast<unsigned long long>(rows) * (cols / 32);
    const unsigned lane = threadIdx.x & 31u, quad = lane >> 2, part = lane & 3u;
    const unsigned long long warp =
        (blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x) >> 5;
    const unsigned long long warp_stride = (gridDim.x * static_cast<unsigned long long>(blockDim.x)) >> 5;
    for (unsigned long long base = warp * 8; base < blocks; base += warp_stride * 8) {
        const unsigned long long b = base + quad;
        const bool valid = b < blocks;
        const uint4 w = valid ? *reinterpret_cast<const uint4*>(src + b * 32 + part * 8) : make_uint4(0, 0, 0, 0);
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
        unsigned qq[2] = {0u, 0u}, ss = 0u;
        if (amax > 0.0f) {
            int e;
            const float m = frexpf(__fmul_rn(amax, 1.0f / 448.0f), &e);
            int k = (m == 0.5f) ? e - 1 : e;
            k = max(-127, min(127, k));
            const float inv = __int_as_float((127 - k) << 23);
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                unsigned wq = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    wq |= static_cast<unsigned>(__nv_cvt_float_to_fp8(v[4 * i + j] * inv, __NV_SATFINITE, __NV_E4M3)) << (8 * j);
                qq[i] = wq;
            }
            ss = static_cast<unsigned>(k + 127);
        }
        if (valid) {
            *reinterpret_cast<uint2*>(q + b * 32 + part * 8) = make_uint2(qq[0], qq[1]);
            if (part == 0) s[b] = static_cast<unsigned char>(ss);
        }
    }
}

// 64-byte smem rows (four 16-byte chunks): XOR the chunk with bits 1..2 of the row (ldmatrix 8-row
// phases conflict-free).
__device__ __forceinline__ unsigned v14_swz(unsigned row, unsigned chunk) {
    return row * 64u + ((chunk ^ ((row >> 1) & 3u)) << 4);
}

constexpr unsigned V14_BC = 256;          // transcoded B columns per CTA (gate/up: 128 gate + 128 up)
constexpr unsigned V14_DA = 3;            // A ring depth (k64 stages)
constexpr unsigned V14_DB = 3;            // raw B ring depth
constexpr unsigned V14_E_STRIDE = 136;    // bf16 per staging row in the gate/up epilogue (128 + 8 pad)
constexpr unsigned V14_THREADS = 512;

template <bool GATEUP> struct V14Smem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V14_BC / BMAT;
    static constexpr unsigned A = V14_DA * V10_TILE_M * 64;
    static constexpr unsigned B = V14_DB * BMAT * 32 * BCOLS;
    static constexpr unsigned S = V14_DB * BMAT * 4 * BCOLS;
    static constexpr unsigned AS = 2 * V10_TILE_M * 4;
    static constexpr unsigned BT = 2 * V14_BC * 64;
    static constexpr unsigned SF = 2 * 2 * V14_BC;
    static constexpr unsigned OFF_B = A, OFF_S = OFF_B + B, OFF_AS = OFF_S + S, OFF_BT = OFF_AS + AS,
                              OFF_SF = OFF_BT + BT, OFF_LUT = OFF_SF + SF, BYTES = OFF_LUT + 256 * 8;
    static_assert(!GATEUP || V10_TILE_M * V14_E_STRIDE * 2 <= OFF_AS, "epilogue staging aliases A/B/S only");
};

__device__ __forceinline__ unsigned v16_pack_bf162(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    unsigned u;
    memcpy(&u, &v, sizeof(u));
    return u;
}

__device__ __forceinline__ void v16_mxfp8_quad(const uint4 w, unsigned (&qq)[2], unsigned& ss) {
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
            qq[i] = v10_pack_e4m3x4(
                v[4 * i + 0] * inv,
                v[4 * i + 1] * inv,
                v[4 * i + 2] * inv,
                v[4 * i + 3] * inv);
        }
        ss = static_cast<unsigned>(k + 127);
    }
}

__device__ __forceinline__ void v16_mxfp8_reg_quad(
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
    v16_mxfp8_quad(w, qq, ss);
}

__device__ __forceinline__ uint4 v16_bf16_reg_quad(
    unsigned u0, unsigned u1, unsigned u2, unsigned u3, unsigned tig
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
    return make_uint4(
        odd2 ? r0 : t0,
        odd2 ? r1 : t1,
        odd2 ? t2 : r0,
        odd2 ? t3 : r1
    );
}

// Iteration i: wait (A i, raw B i+1 landed) -> barrier -> issue A i+2 + raw B i+3 -> transcode B i+1 -> MMA i.
// 16 warps = 4 (M, 32 rows; or 16 rows when valid_rows <= 32) x 4 (N, 64 B columns: gate/up 32 gate + 32 up).
template <bool GATEUP>
__device__ __forceinline__ void v14_impl(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ pk0, const unsigned long long* __restrict__ sc0,
    const float* __restrict__ s20,
    const unsigned long long* __restrict__ pk1, const unsigned long long* __restrict__ sc1,
    const float* __restrict__ s21,
    __nv_bfloat16* __restrict__ C, unsigned char* __restrict__ oq, unsigned char* __restrict__ os,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned N_in, unsigned K_in
) {
    using L = V14Smem<GATEUP>;
    constexpr unsigned N = GATEUP ? oi640::INTER : oi640::HIDDEN;
    constexpr unsigned K = GATEUP ? oi640::HIDDEN : oi640::INTER;
    constexpr unsigned TN = GATEUP ? 128u : 256u;
    constexpr unsigned BMAT = L::BMAT, BCOLS = L::BCOLS;
    constexpr unsigned n_tiles = N / TN;
    constexpr unsigned KS = K / 64u;
    constexpr unsigned kblocks = K / 32u;
    unsigned dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (gridDim.x != oi640::FIXED_GRID || blockDim.x != V14_THREADS || dyn_smem < L::BYTES) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }
    const bool shape = (N_in == N && K_in == K);
    const bool outs = GATEUP ? (oq != nullptr && os != nullptr) : (C != nullptr);
    const bool tbl_ok = pk0 != nullptr && sc0 != nullptr && s20 != nullptr &&
        (!GATEUP || (pk1 != nullptr && sc1 != nullptr && s21 != nullptr));
    if (!shape || !outs || !tbl_ok || aq == nullptr || as == nullptr || expert_offsets == nullptr ||
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
    const unsigned task_count = workspace->work_count * n_tiles;
    if (blockIdx.x >= task_count) return;
    if (tid < 256) {
        const float r = v10_e4m3_to_float(tid);
        const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        unsigned w[2] = {0u, 0u};
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            w[i >> 2] |= static_cast<unsigned>(__nv_cvt_float_to_fp8(mags[i] * r, __NV_SATFINITE, __NV_E4M3)) << (8 * (i & 3));
        lut[tid] = make_uint2(w[0], w[1]);
    }

    const unsigned warp_n = (warp_id & 3u) * (GATEUP ? 32u : 64u);
    const unsigned gid = lane >> 2, tig = lane & 3u;
    constexpr unsigned CH = BCOLS / 16u;
    const unsigned b_mtx = tid / (32u * CH), b_r = (tid / CH) % 32u, b_cc = tid % CH;
    const unsigned s_m2 = tid / (4u * CH), s_r2 = (tid / CH) % 4u, s_c2 = tid % CH;
    const unsigned t_col = tid & 255u, t_h = tid >> 8;
    const unsigned t_mtx = t_col / BCOLS, t_lc = t_col % BCOLS;
    const unsigned t_sf_idx = v16_sf_index<GATEUP>(t_col);

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
        const unsigned active_rows_pad16 = (static_cast<unsigned>(valid_rows) + 15u) & ~15u;
        const bool m32 = (valid_rows <= 32);
        const unsigned warp_m = (warp_id >> 2) * (m32 ? 16u : 32u);
        const int warp_valid = valid_rows - static_cast<int>(warp_m);
        const int rt_active = warp_valid <= 0 ? 0 : (m32 ? 1 : min(2, (warp_valid + 15) >> 4));

        // A: 128 rows x 64 bytes per stage = 512 chunks, one per thread.
        const unsigned a_row = tid >> 2, a_ch = tid & 3u;
        const bool a_ok = static_cast<int>(a_row) < valid_rows;
        const unsigned a_src_row = a_ok ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + a_row]) : cta_m + a_row) : 0u;
        const unsigned char* a_src = aq + static_cast<unsigned long long>(a_src_row) * K + a_ch * 16u;
        const unsigned a_dst = v14_swz(a_row, a_ch);
        const unsigned sc_row = tid - 384u;
        const bool sc_ok = tid >= 384u && static_cast<int>(sc_row) < valid_rows;
        const unsigned sc_src_row = sc_ok ? (GATEUP ? static_cast<unsigned>(sorted_token_ids[cta_m + sc_row]) : cta_m + sc_row) : 0u;
        const unsigned char* sc_src = as + static_cast<unsigned long long>(sc_src_row) * kblocks;

        const unsigned char* const b_src_base = (b_mtx ? B1 : B0) + b_r * N + cta_n + b_cc * 16u;
        const unsigned char* const s_src_base = (s_m2 ? S1 : S0) + s_r2 * N + cta_n + s_c2 * 16u;

        auto load_a = [&](unsigned ks) {
            if (a_row < active_rows_pad16)
                v10_cp16(sA + (ks % V14_DA) * (V10_TILE_M * 64) + a_dst, a_src + ks * 64u, a_ok);
            if (tid >= 384u && (ks & 1u) == 0u && sc_row < active_rows_pad16)       // four k32 scale bytes cover stages ks, ks+1
                v10_cp4(&sAS[(ks >> 1) & 1][sc_row], sc_src + ks * 2u, sc_ok);
        };
        auto load_b = [&](unsigned ks) {
            v10_cp16(&sB[ks % V14_DB][b_mtx][b_r][b_cc * 16u], b_src_base + ks * (32u * N), true);
            if (tid < BMAT * 4u * CH) {
                v10_cp16(&sS[ks % V14_DB][s_m2][s_r2][s_c2 * 16u], s_src_base + ks * (4u * N), true);
            }
        };
        auto transcode_512 = [&](unsigned sl, unsigned tb) {
            const unsigned char (*const sb_base)[BCOLS] = &sB[sl][t_mtx][8u * t_h];
            #pragma unroll
            for (unsigned t = 0; t < 2; ++t) {
                const unsigned s0 = sS[sl][t_mtx][2 * t][t_lc];
                const unsigned s1 = sS[sl][t_mtx][2 * t + 1][t_lc];
                const unsigned s_own = t_h ? s1 : s0;
                const unsigned s_max = max(s0 & 0x7fu, s1 & 0x7fu);
                const int e32 = (s_max >= 8u) ? (static_cast<int>(s_max >> 3) - 7)
                              : (s_max != 0u) ? (22 - __clz(static_cast<int>(s_max)))
                                              : 0;
                const float inv = __int_as_float((127 - e32) << 23);
                const unsigned r = __nv_cvt_float_to_fp8(v10_e4m3_to_float(s_own) * inv, __NV_SATFINITE, __NV_E4M3);
                const uint2 tt = lut[r];
                unsigned w[4];
                #pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const unsigned x = static_cast<unsigned>(sb_base[16 * t + 2 * g][t_lc]) |
                                       (static_cast<unsigned>(sb_base[16 * t + 2 * g + 1][t_lc]) << 8);
                    w[g] = v10_e2m1x4_to_e4m3x4(x, tt.x, tt.y);
                }
                *reinterpret_cast<uint4*>(sBt + tb * (V14_BC * 64) + v14_swz(t_col, 2 * t + t_h)) =
                    make_uint4(w[0], w[1], w[2], w[3]);
                if (t_h == 0) sSF[tb][t][t_sf_idx] = static_cast<unsigned char>((e32 + 127) & 0xff);
            }
        };
        auto transcode = [&](unsigned sl, unsigned tb) {
            if (valid_rows <= 64) {
                if (tid < V14_BC) return;
                #pragma unroll
                for (unsigned t = 0; t < 2; ++t) {
                    const unsigned s0 = sS[sl][t_mtx][2 * t][t_lc];
                    const unsigned s1 = sS[sl][t_mtx][2 * t + 1][t_lc];
                    const unsigned s_max = max(s0 & 0x7fu, s1 & 0x7fu);
                    const int e32 = (s_max >= 8u) ? (static_cast<int>(s_max >> 3) - 7)
                                  : (s_max != 0u) ? (22 - __clz(static_cast<int>(s_max)))
                                                  : 0;
                    const float inv = __int_as_float((127 - e32) << 23);
                    sSF[tb][t][t_sf_idx] = static_cast<unsigned char>((e32 + 127) & 0xff);
                    #pragma unroll
                    for (unsigned h = 0; h < 2; ++h) {
                        const unsigned s_own = h ? s1 : s0;
                        const unsigned r = __nv_cvt_float_to_fp8(v10_e4m3_to_float(s_own) * inv, __NV_SATFINITE, __NV_E4M3);
                        const uint2 tt = lut[r];
                        unsigned w[4];
                        #pragma unroll
                        for (int g = 0; g < 4; ++g) {
                            const unsigned x = static_cast<unsigned>(sB[sl][t_mtx][16 * t + 8 * h + 2 * g][t_lc]) |
                                               (static_cast<unsigned>(sB[sl][t_mtx][16 * t + 8 * h + 2 * g + 1][t_lc]) << 8);
                            w[g] = v10_e2m1x4_to_e4m3x4(x, tt.x, tt.y);
                        }
                        *reinterpret_cast<uint4*>(sBt + tb * (V14_BC * 64) + v14_swz(t_col, 2 * t + h)) =
                            make_uint4(w[0], w[1], w[2], w[3]);
                    }
                }
            } else {
                transcode_512(sl, tb);
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
        transcode_512(0, 0);

        unsigned sfa[2] = {0u, 0u};
        auto step = [&](auto par_tag, unsigned ks) {
            constexpr unsigned PAR = decltype(par_tag)::value;   // ks & 1
            v10_wait<1>();                                       // A ks, raw B ks+1 landed
            __syncthreads();
            if (ks + 2 < KS) load_a(ks + 2);
            if (ks + 3 < KS) load_b(ks + 3);
            v10_commit();
            if (ks + 1 < KS) transcode((ks + 1) % V14_DB, PAR ^ 1);
            if (rt_active == 0) return;
            if constexpr (PAR == 0) {
                const unsigned as_buf = (ks >> 1) & 1;
                sfa[0] = sAS[as_buf][warp_m + gid + (tig & 1) * 8];
                if (rt_active > 1) sfa[1] = sAS[as_buf][warp_m + 16 + gid + (tig & 1) * 8];
            }
            const unsigned a_slot = ks % V14_DA;
            const unsigned row_a0 = warp_m + (lane & 15u);
            const unsigned row_b0 = warp_n + ((lane >> 4) << 3) + (lane & 7u);
            #pragma unroll
            for (unsigned t = 0; t < 2; ++t) {
                unsigned afr0[4], afr1[4], bfr[8][2];
                const unsigned addr_a0 = static_cast<unsigned>(__cvta_generic_to_shared(
                    sA + a_slot * (V10_TILE_M * 64) + v14_swz(row_a0, 2 * t + (lane >> 4))));
                v10_ldsm4(addr_a0, afr0[0], afr0[1], afr0[2], afr0[3]);
                const uint2 sb8 = *reinterpret_cast<const uint2*>(
                    &sSF[PAR][t][((warp_id & 3u) << 6) | (gid << 3)]);
                const unsigned addr_b0 = static_cast<unsigned>(__cvta_generic_to_shared(
                    sBt + PAR * (V14_BC * 64) + v14_swz(row_b0, 2 * t + ((lane >> 3) & 1u))));
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const unsigned off_p = GATEUP ? ((p >> 1) * (BCOLS * 64u) + (p & 1) * 1024u)
                                                  : (static_cast<unsigned>(p) * 1024u);
                    v10_ldsm4(addr_b0 + off_p, bfr[2 * p][0], bfr[2 * p][1], bfr[2 * p + 1][0], bfr[2 * p + 1][1]);
                }
                if (rt_active > 1) {
                    v10_ldsm4(addr_a0 + 1024u, afr1[0], afr1[1], afr1[2], afr1[3]);
                }
                if (t == 0) {
                    v16_qmma4_lo<2 * PAR>(acc[0], afr0, bfr, sfa[0], sb8.x);
                    v16_qmma4_hi<2 * PAR>(acc[0], afr0, bfr, sfa[0], sb8.y);
                    if (rt_active > 1) {
                        v16_qmma4_lo<2 * PAR>(acc[1], afr1, bfr, sfa[1], sb8.x);
                        v16_qmma4_hi<2 * PAR>(acc[1], afr1, bfr, sfa[1], sb8.y);
                    }
                } else {
                    v16_qmma4_lo<2 * PAR + 1>(acc[0], afr0, bfr, sfa[0], sb8.x);
                    v16_qmma4_hi<2 * PAR + 1>(acc[0], afr0, bfr, sfa[0], sb8.y);
                    if (rt_active > 1) {
                        v16_qmma4_lo<2 * PAR + 1>(acc[1], afr1, bfr, sfa[1], sb8.x);
                        v16_qmma4_hi<2 * PAR + 1>(acc[1], afr1, bfr, sfa[1], sb8.y);
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
                    u0[j] = v16_pack_bf162(v[0], v[1]);
                    u1[j] = v16_pack_bf162(v[2], v[3]);
                }
                unsigned qq0[2], ss0, qq1[2], ss1;
                v16_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], tig, qq0, ss0);
                if (static_cast<int>(lr0) < valid_rows) {
                    const unsigned long long orow = cta_m + lr0;
                    *reinterpret_cast<uint2*>(oq + orow * N + cta_n + col8) = make_uint2(qq0[0], qq0[1]);
                    if (tig == 0u) os[orow * (N / 32) + cta_n / 32 + blk] = static_cast<unsigned char>(ss0);
                }
                v16_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], tig, qq1, ss1);
                if (static_cast<int>(lr1) < valid_rows) {
                    const unsigned long long orow = cta_m + lr1;
                    *reinterpret_cast<uint2*>(oq + orow * N + cta_n + col8) = make_uint2(qq1[0], qq1[1]);
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
                        u0[jj] = v16_pack_bf162(acc[r][j][0] * s2a, acc[r][j][1] * s2a);
                        u1[jj] = v16_pack_bf162(acc[r][j][2] * s2a, acc[r][j][3] * s2a);
                    }
                    const unsigned col8 = warp_n + b * 32u + tig * 8u;
                    const uint4 w0 = v16_bf16_reg_quad(u0[0], u0[1], u0[2], u0[3], tig);
                    if (static_cast<int>(lr0) < valid_rows) {
                        *reinterpret_cast<uint4*>(C + static_cast<unsigned long long>(cta_m + lr0) * N + cta_n + col8) = w0;
                    }
                    const uint4 w1 = v16_bf16_reg_quad(u1[0], u1[1], u1[2], u1[3], tig);
                    if (static_cast<int>(lr1) < valid_rows) {
                        *reinterpret_cast<uint4*>(C + static_cast<unsigned long long>(cta_m + lr1) * N + cta_n + col8) = w1;
                    }
                }
            }
        }
        if (task + gridDim.x < task_count) __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v16(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v14_impl<true>(aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
                   up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
                   status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v16(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v14_impl<false>(aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
                    expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

// Dynamic shared memory each kernel needs (host side: set MAX_DYNAMIC_SHARED_SIZE_BYTES and launch with it).
extern "C" __global__ void qwen4_moe_compact_v16_smem_bytes(unsigned* out) {
    out[0] = V14Smem<true>::BYTES;
    out[1] = V14Smem<false>::BYTES;
}

// Fused MXFP8 row quantiser + shared-expert gate scalar precomputation (`gatepre`).
// Produces byte-identical (q, s) to `qwen4_mxfp8_quant_rows_v16q4` and byte-identical
// `ss_out[r] = sigmoid(dot(src[r], shared_gate))` to `moe_batched_blend` / `moe_gates` /
// `hc_blend_gate` (256-thread strided serial 10-term dot product in ascending `i = 0..9`,
// `shfl_down_sync` reduction for `offset = 16, 8, 4, 2, 1`, ascending warp sum `w = 0..7`,
// and `1.0f / (1.0f + __expf(-total))` under `--fmad=false`).
// Fuses `x * shared_gate` products into Pass 0/1 while `x` is already in registers (eliminating
// the second global BF16 read of `x_row`), staging products via conflict-free swizzled SMEM
// (`s_prod[10][256]`, 10 KiB) so the final reduction preserves the exact 0-ULP FP32 order.
// Launch: grid (min(rows, cap), 1, 1), block (256, 1, 1).
extern "C" __global__ __launch_bounds__(256) void qwen4_mxfp8_quant_rows_v16q4_gate(
    const __nv_bfloat16* __restrict__ src,
    unsigned char* __restrict__ q,
    unsigned char* __restrict__ s,
    const __nv_bfloat16* __restrict__ shared_gate,
    float* __restrict__ ss_out,
    unsigned int rows,
    const unsigned int* __restrict__ amax_bits,
    unsigned char* __restrict__ a8_out,
    float* __restrict__ a_scale_out
) {
    constexpr unsigned int kHidden = 2560u;
    constexpr unsigned int kBlocksPerRow = kHidden / 32u; // 80
    constexpr unsigned int kBlendThreads = 256u;
    constexpr unsigned int kGateTerms = kHidden / kBlendThreads; // 10
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1030)
    constexpr unsigned int kXqPinMaxRows = 20480u;
    constexpr unsigned int kA8PinMaxRow = 20480u;
#else
    constexpr unsigned int kXqPinMaxRows = 4096u;
    constexpr unsigned int kA8PinMaxRow = 6400u;
#endif

    const unsigned int bt = threadIdx.x;
    const unsigned int warp = bt >> 5;
    const unsigned int lane = bt & 31u;
    const unsigned int quad_in_cta = bt >> 2; // 0..63
    const unsigned int part = bt & 3u;        // 0..3

    const bool want_a8 = (a8_out != nullptr && amax_bits != nullptr);
    const bool want_gate = (shared_gate != nullptr && ss_out != nullptr);
    float fp8_inv = 0.0f;
    uint64_t pol_el = 0;
    uint64_t pol_ef = 0;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));
    if (want_a8) {
        const float fp8_amax = fmaxf(__uint_as_float(amax_bits[0]), 1e-30f);
        fp8_inv = 448.0f / fp8_amax;
        if (a_scale_out != nullptr && blockIdx.x == 0u && bt == 0u) {
            a_scale_out[0] = fp8_amax / 448.0f;
        }
    }

    auto load_bf16x8 = [](const __nv_bfloat16* p, bool aligned16) -> uint4 {
        if (aligned16) {
            return __ldg(reinterpret_cast<const uint4*>(p));
        }
        unsigned short h[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            h[i] = *reinterpret_cast<const unsigned short*>(p + i);
        }
        return make_uint4(
            static_cast<unsigned>(h[0]) | (static_cast<unsigned>(h[1]) << 16),
            static_cast<unsigned>(h[2]) | (static_cast<unsigned>(h[3]) << 16),
            static_cast<unsigned>(h[4]) | (static_cast<unsigned>(h[5]) << 16),
            static_cast<unsigned>(h[6]) | (static_cast<unsigned>(h[7]) << 16));
    };

    // Pass 0 thread `bt` loads `x_row[bt*8 .. bt*8+7]` = term `i = warp` of target lanes `lane*8 + 0..7`.
    // Pass 1 thread `bt < 64` loads `x_row[2048 + bt*8 .. +7]` = term `i = 8 + warp`.
    // `shared_gate` is only 5 KiB (100% L1-resident), so loading on-demand via `__ldg` saves
    // 8 live registers per thread and keeps full 6-CTA/SM occupancy with zero spills.
    const bool gate_aligned16 = want_gate
        && ((reinterpret_cast<uintptr_t>(shared_gate) & 15u) == 0u);

    __shared__ __align__(16) float s_prod[kGateTerms][kBlendThreads];
    __shared__ float s_dot[2][kBlendThreads / 32u];
    const unsigned int prod_b0 = (lane * 8u) ^ (((lane >> 2) & 1u) << 2);
    const unsigned int prod_b1 = prod_b0 ^ 4u;
    const unsigned int swz_bt = bt ^ (((bt >> 5) & 1u) << 2);
    unsigned int row_parity = 0u;
    const bool src_aligned128 = ((reinterpret_cast<uintptr_t>(src) & 127u) == 0u);

    auto stage_gate_prod = [&](int pass, const float (&v)[8]) {
        const unsigned int term_idx = static_cast<unsigned int>(pass) * 8u + warp;
        const uint4 g_u4 = load_bf16x8(
            shared_gate + static_cast<unsigned int>(pass) * 2048u + bt * 8u,
            gate_aligned16);
        *reinterpret_cast<float4*>(&s_prod[term_idx][prod_b0]) = make_float4(
            v[0] * __uint_as_float(g_u4.x << 16),
            v[1] * __uint_as_float(g_u4.x & 0xffff0000u),
            v[2] * __uint_as_float(g_u4.y << 16),
            v[3] * __uint_as_float(g_u4.y & 0xffff0000u));
        *reinterpret_cast<float4*>(&s_prod[term_idx][prod_b1]) = make_float4(
            v[4] * __uint_as_float(g_u4.z << 16),
            v[5] * __uint_as_float(g_u4.z & 0xffff0000u),
            v[6] * __uint_as_float(g_u4.w << 16),
            v[7] * __uint_as_float(g_u4.w & 0xffff0000u));
    };

    for (unsigned int r = blockIdx.x; r < rows; r += gridDim.x) {
        const __nv_bfloat16* x_row = src + static_cast<unsigned long long>(r) * kHidden;
        if (q == nullptr) {
            if (want_a8 || want_gate) {
                const uint4* xr4 = reinterpret_cast<const uint4*>(x_row);
                uint2* out2 = want_a8
                    ? reinterpret_cast<uint2*>(a8_out + static_cast<unsigned long long>(r) * kHidden)
                    : nullptr;
                #pragma unroll
                for (int pass = 0; pass < 2; ++pass) {
                    const unsigned int idx = pass * 256u + bt;
                    if (pass == 0 || idx < (kHidden / 8u)) {
                        uint4 w;
                        asm volatile("ld.global.ca.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                                     : "=r"(w.x), "=r"(w.y), "=r"(w.z), "=r"(w.w)
                                     : "l"(xr4 + idx), "l"(pol_ef));
                        const unsigned u[4] = {w.x, w.y, w.z, w.w};
                        float v[8];
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            v[2 * i] = __uint_as_float(u[i] << 16);
                            v[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
                        }
                        if (want_gate) {
                            stage_gate_prod(pass, v);
                        }
                        if (out2 != nullptr) {
                            const unsigned wa0 = v10_pack_e4m3x4(
                                v[0] * fp8_inv, v[1] * fp8_inv, v[2] * fp8_inv, v[3] * fp8_inv);
                            const unsigned wa1 = v10_pack_e4m3x4(
                                v[4] * fp8_inv, v[5] * fp8_inv, v[6] * fp8_inv, v[7] * fp8_inv);
                            if (r < kA8PinMaxRow) {
                                asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                                             :: "l"(out2 + idx), "r"(wa0), "r"(wa1), "l"(pol_el)
                                             : "memory");
                            } else {
                                out2[idx] = make_uint2(wa0, wa1);
                            }
                        }
                    }
                }
            }
        } else {
            const unsigned long long row_b0 = static_cast<unsigned long long>(r) * kBlocksPerRow;
            // Pass 0: all 64 quads (256 threads) quantize MX32 blocks 0..63 of row `r` and stage terms 0..7.
            // Pass 1: first 16 quads (threads 0..63, Warps 0..1) quantize MX32 blocks 64..79 and stage terms 8..9.
            #pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                if (pass == 0 || bt < 64u) {
                    const unsigned int q_idx = pass * 64u + quad_in_cta;
                    const unsigned long long b = row_b0 + q_idx;
                    uint4 w;
                    const uint4* sptr = reinterpret_cast<const uint4*>(src + b * 32ull + part * 8u);
                    asm volatile("ld.global.ca.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                                 : "=r"(w.x), "=r"(w.y), "=r"(w.z), "=r"(w.w)
                                 : "l"(sptr), "l"(pol_ef));
                    const unsigned u[4] = {w.x, w.y, w.z, w.w};
                    float v[8];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        v[2 * i] = __uint_as_float(u[i] << 16);
                        v[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
                    }
                    if (want_gate) {
                        stage_gate_prod(pass, v);
                    }
                    float amax = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                    unsigned qq[2] = {0u, 0u}, ss = 0u;
                    if (amax > 0.0f) {
                        int e;
                        const float m = frexpf(__fmul_rn(amax, 1.0f / 448.0f), &e);
                        int k = (m == 0.5f) ? e - 1 : e;
                        k = max(-127, min(127, k));
                        const float inv = __int_as_float((127 - k) << 23);
                        #pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            qq[i] = v10_pack_e4m3x4(
                                v[4 * i + 0] * inv,
                                v[4 * i + 1] * inv,
                                v[4 * i + 2] * inv,
                                v[4 * i + 3] * inv);
                        }
                        ss = static_cast<unsigned>(k + 127);
                    }
                    uint2* qptr = reinterpret_cast<uint2*>(q + b * 32ull + part * 8u);
                    if (rows <= kXqPinMaxRows) {
                        asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                                     :: "l"(qptr), "r"(qq[0]), "r"(qq[1]), "l"(pol_el) : "memory");
                    } else {
                        qptr[0] = make_uint2(qq[0], qq[1]);
                    }
                    if (part == 0u) s[b] = static_cast<unsigned char>(ss);
                    if (want_a8) {
                        unsigned wa[2] = {0u, 0u};
                        #pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            wa[i] = v10_pack_e4m3x4(
                                v[4 * i + 0] * fp8_inv,
                                v[4 * i + 1] * fp8_inv,
                                v[4 * i + 2] * fp8_inv,
                                v[4 * i + 3] * fp8_inv);
                        }
                        uint2* optr = reinterpret_cast<uint2*>(a8_out + b * 32ull + part * 8u);
                        if (r < kA8PinMaxRow) {
                            asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                                         :: "l"(optr), "r"(wa[0]), "r"(wa[1]), "l"(pol_el) : "memory");
                        } else {
                            optr[0] = make_uint2(wa[0], wa[1]);
                        }
                    }
                }
            }
        }

        // Exact 256-thread strided dot product + shfl_down + ascending-warp sum (`moe_gates`),
        // consuming the pre-staged products from SMEM with zero second global read of `x_row`.
        if (!want_gate) {
            if (bt == 0u && ss_out != nullptr) ss_out[r] = 1.0f;
            if (want_a8 && src_aligned128) __syncthreads();
        } else {
            __syncthreads();
            float d = 0.0f;
            #pragma unroll
            for (unsigned int i = 0; i < kGateTerms; ++i) {
                d += s_prod[i][swz_bt];
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                d += __shfl_down_sync(0xFFFFFFFFu, d, offset);
            }
            if (lane == 0u) s_dot[row_parity][warp] = d;
            __syncthreads();
            if (bt == 0u) {
                float total = 0.0f;
                #pragma unroll
                for (unsigned int w = 0; w < kBlendThreads / 32u; ++w) total += s_dot[row_parity][w];
                ss_out[r] = 1.0f / (1.0f + __expf(-total));
            }
            row_parity ^= 1u;
        }
        if (want_a8 && src_aligned128 && bt < 40u) {
            asm volatile("discard.global.L2 [%0], 128;\n"
                         :: "l"(reinterpret_cast<const unsigned char*>(x_row) + bt * 128u)
                         : "memory");
        }
    }
}
