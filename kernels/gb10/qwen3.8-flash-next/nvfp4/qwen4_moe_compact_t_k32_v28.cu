// SPDX-License-Identifier: AGPL-3.0-only
//
// V28: W4A8 MoE family built on V16 (ATLAS_QWEN4_PREFILL_MOE_W4A8_WIDE=7).
// Originally introduced packed `sSF` B-column scales (one `LDS.64` per lane quad + QMMA B-scale byte
// selector; subsequently backported into included `qwen4_moe_compact_t_k32_v16.cu`) and extends V16
// with compile-time `V28_E4M3_LUT`, L2 residency hints, shared-expert pretranscode / 2-stage / discard
// kernels, `v28q` / `v28_twarp` / `v28_m64` / `v28_m32` / `v28_m16` (+ Atlas-TS4E-128 compressed scales),
// `v30_2cta`, and `v31_ws` warp-specialized work-stealing variants. Same QMMA sequence and operands
// per accumulator: bit-identical to V16.
//
// Include V16 as the arithmetic/layout SSOT. This module also carries V16's entry points (same-build
// control for the raw gate); runtime dispatch asks it only for the V28 symbols below.
#include "qwen4_moe_compact_t_k32_v16.cu"

// Compile-time E2M1-magnitude x residual-E4M3 lookup table (256 x 8 B = 2 KiB in constant cache).
// Entry r packs e4m3_rn_satfinite({0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} * e4m3_to_float(r))
// bit-identically to the runtime v10_e4m3_to_float + __nv_cvt_float_to_fp8 loop across all 256 codes.
static __constant__ uint2 V28_E4M3_LUT[256] = {
    {0x00000000u, 0x00000000u}, {0x02010000u, 0x06040302u}, {0x03020100u, 0x0c080604u}, {0x04030200u, 0x110c0906u},
    {0x06040200u, 0x14100c08u}, {0x08050200u, 0x17120f0au}, {0x09060300u, 0x1914110cu}, {0x0a070400u, 0x1a16120eu},
    {0x0c080400u, 0x1c181410u}, {0x0e090400u, 0x1e191611u}, {0x0f0a0500u, 0x1f1a1712u}, {0x100b0600u, 0x201b1813u},
    {0x110c0600u, 0x211c1914u}, {0x120d0600u, 0x221d1a15u}, {0x120e0700u, 0x221e1a16u}, {0x130f0800u, 0x231f1b17u},
    {0x14100800u, 0x24201c18u}, {0x16110900u, 0x26211e19u}, {0x17120a00u, 0x27221f1au}, {0x18130b00u, 0x2823201bu},
    {0x19140c00u, 0x2924211cu}, {0x1a150d00u, 0x2a25221du}, {0x1a160e00u, 0x2a26221eu}, {0x1b170f00u, 0x2b27231fu},
    {0x1c181000u, 0x2c282420u}, {0x1e191100u, 0x2e292621u}, {0x1f1a1200u, 0x2f2a2722u}, {0x201b1300u, 0x302b2823u},
    {0x211c1400u, 0x312c2924u}, {0x221d1500u, 0x322d2a25u}, {0x221e1600u, 0x322e2a26u}, {0x231f1700u, 0x332f2b27u},
    {0x24201800u, 0x34302c28u}, {0x26211900u, 0x36312e29u}, {0x27221a00u, 0x37322f2au}, {0x28231b00u, 0x3833302bu},
    {0x29241c00u, 0x3934312cu}, {0x2a251d00u, 0x3a35322du}, {0x2a261e00u, 0x3a36322eu}, {0x2b271f00u, 0x3b37332fu},
    {0x2c282000u, 0x3c383430u}, {0x2e292100u, 0x3e393631u}, {0x2f2a2200u, 0x3f3a3732u}, {0x302b2300u, 0x403b3833u},
    {0x312c2400u, 0x413c3934u}, {0x322d2500u, 0x423d3a35u}, {0x322e2600u, 0x423e3a36u}, {0x332f2700u, 0x433f3b37u},
    {0x34302800u, 0x44403c38u}, {0x36312900u, 0x46413e39u}, {0x37322a00u, 0x47423f3au}, {0x38332b00u, 0x4843403bu},
    {0x39342c00u, 0x4944413cu}, {0x3a352d00u, 0x4a45423du}, {0x3a362e00u, 0x4a46423eu}, {0x3b372f00u, 0x4b47433fu},
    {0x3c383000u, 0x4c484440u}, {0x3e393100u, 0x4e494641u}, {0x3f3a3200u, 0x4f4a4742u}, {0x403b3300u, 0x504b4843u},
    {0x413c3400u, 0x514c4944u}, {0x423d3500u, 0x524d4a45u}, {0x423e3600u, 0x524e4a46u}, {0x433f3700u, 0x534f4b47u},
    {0x44403800u, 0x54504c48u}, {0x46413900u, 0x56514e49u}, {0x47423a00u, 0x57524f4au}, {0x48433b00u, 0x5853504bu},
    {0x49443c00u, 0x5954514cu}, {0x4a453d00u, 0x5a55524du}, {0x4a463e00u, 0x5a56524eu}, {0x4b473f00u, 0x5b57534fu},
    {0x4c484000u, 0x5c585450u}, {0x4e494100u, 0x5e595651u}, {0x4f4a4200u, 0x5f5a5752u}, {0x504b4300u, 0x605b5853u},
    {0x514c4400u, 0x615c5954u}, {0x524d4500u, 0x625d5a55u}, {0x524e4600u, 0x625e5a56u}, {0x534f4700u, 0x635f5b57u},
    {0x54504800u, 0x64605c58u}, {0x56514900u, 0x66615e59u}, {0x57524a00u, 0x67625f5au}, {0x58534b00u, 0x6863605bu},
    {0x59544c00u, 0x6964615cu}, {0x5a554d00u, 0x6a65625du}, {0x5a564e00u, 0x6a66625eu}, {0x5b574f00u, 0x6b67635fu},
    {0x5c585000u, 0x6c686460u}, {0x5e595100u, 0x6e696661u}, {0x5f5a5200u, 0x6f6a6762u}, {0x605b5300u, 0x706b6863u},
    {0x615c5400u, 0x716c6964u}, {0x625d5500u, 0x726d6a65u}, {0x625e5600u, 0x726e6a66u}, {0x635f5700u, 0x736f6b67u},
    {0x64605800u, 0x74706c68u}, {0x66615900u, 0x76716e69u}, {0x67625a00u, 0x77726f6au}, {0x68635b00u, 0x7873706bu},
    {0x69645c00u, 0x7974716cu}, {0x6a655d00u, 0x7a75726du}, {0x6a665e00u, 0x7a76726eu}, {0x6b675f00u, 0x7b77736fu},
    {0x6c686000u, 0x7c787470u}, {0x6e696100u, 0x7e797671u}, {0x6f6a6200u, 0x7e7a7772u}, {0x706b6300u, 0x7e7b7873u},
    {0x716c6400u, 0x7e7c7974u}, {0x726d6500u, 0x7e7d7a75u}, {0x726e6600u, 0x7e7e7a76u}, {0x736f6700u, 0x7e7e7b77u},
    {0x74706800u, 0x7e7e7c78u}, {0x76716900u, 0x7e7e7e79u}, {0x77726a00u, 0x7e7e7e7au}, {0x78736b00u, 0x7e7e7e7bu},
    {0x79746c00u, 0x7e7e7e7cu}, {0x7a756d00u, 0x7e7e7e7du}, {0x7a766e00u, 0x7e7e7e7eu}, {0x7b776f00u, 0x7e7e7e7eu},
    {0x7c787000u, 0x7e7e7e7eu}, {0x7e797100u, 0x7e7e7e7eu}, {0x7e7a7200u, 0x7e7e7e7eu}, {0x7e7b7300u, 0x7e7e7e7eu},
    {0x7e7c7400u, 0x7e7e7e7eu}, {0x7e7d7500u, 0x7e7e7e7eu}, {0x7e7e7600u, 0x7e7e7e7eu}, {0x7f7f7f7fu, 0x7f7f7f7fu},
    {0x80808080u, 0x80808080u}, {0x82818080u, 0x86848382u}, {0x83828180u, 0x8c888684u}, {0x84838280u, 0x918c8986u},
    {0x86848280u, 0x94908c88u}, {0x88858280u, 0x97928f8au}, {0x89868380u, 0x9994918cu}, {0x8a878480u, 0x9a96928eu},
    {0x8c888480u, 0x9c989490u}, {0x8e898480u, 0x9e999691u}, {0x8f8a8580u, 0x9f9a9792u}, {0x908b8680u, 0xa09b9893u},
    {0x918c8680u, 0xa19c9994u}, {0x928d8680u, 0xa29d9a95u}, {0x928e8780u, 0xa29e9a96u}, {0x938f8880u, 0xa39f9b97u},
    {0x94908880u, 0xa4a09c98u}, {0x96918980u, 0xa6a19e99u}, {0x97928a80u, 0xa7a29f9au}, {0x98938b80u, 0xa8a3a09bu},
    {0x99948c80u, 0xa9a4a19cu}, {0x9a958d80u, 0xaaa5a29du}, {0x9a968e80u, 0xaaa6a29eu}, {0x9b978f80u, 0xaba7a39fu},
    {0x9c989080u, 0xaca8a4a0u}, {0x9e999180u, 0xaea9a6a1u}, {0x9f9a9280u, 0xafaaa7a2u}, {0xa09b9380u, 0xb0aba8a3u},
    {0xa19c9480u, 0xb1aca9a4u}, {0xa29d9580u, 0xb2adaaa5u}, {0xa29e9680u, 0xb2aeaaa6u}, {0xa39f9780u, 0xb3afaba7u},
    {0xa4a09880u, 0xb4b0aca8u}, {0xa6a19980u, 0xb6b1aea9u}, {0xa7a29a80u, 0xb7b2afaau}, {0xa8a39b80u, 0xb8b3b0abu},
    {0xa9a49c80u, 0xb9b4b1acu}, {0xaaa59d80u, 0xbab5b2adu}, {0xaaa69e80u, 0xbab6b2aeu}, {0xaba79f80u, 0xbbb7b3afu},
    {0xaca8a080u, 0xbcb8b4b0u}, {0xaea9a180u, 0xbeb9b6b1u}, {0xafaaa280u, 0xbfbab7b2u}, {0xb0aba380u, 0xc0bbb8b3u},
    {0xb1aca480u, 0xc1bcb9b4u}, {0xb2ada580u, 0xc2bdbab5u}, {0xb2aea680u, 0xc2bebab6u}, {0xb3afa780u, 0xc3bfbbb7u},
    {0xb4b0a880u, 0xc4c0bcb8u}, {0xb6b1a980u, 0xc6c1beb9u}, {0xb7b2aa80u, 0xc7c2bfbau}, {0xb8b3ab80u, 0xc8c3c0bbu},
    {0xb9b4ac80u, 0xc9c4c1bcu}, {0xbab5ad80u, 0xcac5c2bdu}, {0xbab6ae80u, 0xcac6c2beu}, {0xbbb7af80u, 0xcbc7c3bfu},
    {0xbcb8b080u, 0xccc8c4c0u}, {0xbeb9b180u, 0xcec9c6c1u}, {0xbfbab280u, 0xcfcac7c2u}, {0xc0bbb380u, 0xd0cbc8c3u},
    {0xc1bcb480u, 0xd1ccc9c4u}, {0xc2bdb580u, 0xd2cdcac5u}, {0xc2beb680u, 0xd2cecac6u}, {0xc3bfb780u, 0xd3cfcbc7u},
    {0xc4c0b880u, 0xd4d0ccc8u}, {0xc6c1b980u, 0xd6d1cec9u}, {0xc7c2ba80u, 0xd7d2cfcau}, {0xc8c3bb80u, 0xd8d3d0cbu},
    {0xc9c4bc80u, 0xd9d4d1ccu}, {0xcac5bd80u, 0xdad5d2cdu}, {0xcac6be80u, 0xdad6d2ceu}, {0xcbc7bf80u, 0xdbd7d3cfu},
    {0xccc8c080u, 0xdcd8d4d0u}, {0xcec9c180u, 0xded9d6d1u}, {0xcfcac280u, 0xdfdad7d2u}, {0xd0cbc380u, 0xe0dbd8d3u},
    {0xd1ccc480u, 0xe1dcd9d4u}, {0xd2cdc580u, 0xe2dddad5u}, {0xd2cec680u, 0xe2dedad6u}, {0xd3cfc780u, 0xe3dfdbd7u},
    {0xd4d0c880u, 0xe4e0dcd8u}, {0xd6d1c980u, 0xe6e1ded9u}, {0xd7d2ca80u, 0xe7e2dfdau}, {0xd8d3cb80u, 0xe8e3e0dbu},
    {0xd9d4cc80u, 0xe9e4e1dcu}, {0xdad5cd80u, 0xeae5e2ddu}, {0xdad6ce80u, 0xeae6e2deu}, {0xdbd7cf80u, 0xebe7e3dfu},
    {0xdcd8d080u, 0xece8e4e0u}, {0xded9d180u, 0xeee9e6e1u}, {0xdfdad280u, 0xefeae7e2u}, {0xe0dbd380u, 0xf0ebe8e3u},
    {0xe1dcd480u, 0xf1ece9e4u}, {0xe2ddd580u, 0xf2edeae5u}, {0xe2ded680u, 0xf2eeeae6u}, {0xe3dfd780u, 0xf3efebe7u},
    {0xe4e0d880u, 0xf4f0ece8u}, {0xe6e1d980u, 0xf6f1eee9u}, {0xe7e2da80u, 0xf7f2efeau}, {0xe8e3db80u, 0xf8f3f0ebu},
    {0xe9e4dc80u, 0xf9f4f1ecu}, {0xeae5dd80u, 0xfaf5f2edu}, {0xeae6de80u, 0xfaf6f2eeu}, {0xebe7df80u, 0xfbf7f3efu},
    {0xece8e080u, 0xfcf8f4f0u}, {0xeee9e180u, 0xfef9f6f1u}, {0xefeae280u, 0xfefaf7f2u}, {0xf0ebe380u, 0xfefbf8f3u},
    {0xf1ece480u, 0xfefcf9f4u}, {0xf2ede580u, 0xfefdfaf5u}, {0xf2eee680u, 0xfefefaf6u}, {0xf3efe780u, 0xfefefbf7u},
    {0xf4f0e880u, 0xfefefcf8u}, {0xf6f1e980u, 0xfefefef9u}, {0xf7f2ea80u, 0xfefefefau}, {0xf8f3eb80u, 0xfefefefbu},
    {0xf9f4ec80u, 0xfefefefcu}, {0xfaf5ed80u, 0xfefefefdu}, {0xfaf6ee80u, 0xfefefefeu}, {0xfbf7ef80u, 0xfefefefeu},
    {0xfcf8f080u, 0xfefefefeu}, {0xfef9f180u, 0xfefefefeu}, {0xfefaf280u, 0xfefefefeu}, {0xfefbf380u, 0xfefefefeu},
    {0xfefcf480u, 0xfefefefeu}, {0xfefdf580u, 0xfefefefeu}, {0xfefef680u, 0xfefefefeu}, {0x7f7f7f7fu, 0x7f7f7f7fu},
};

namespace {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1030)
constexpr unsigned long long V28_KEEP_SLOT_BYTES = 64ull << 20;
constexpr unsigned V31_KEEP_OQ_WINDOW = 960u;
#else
constexpr unsigned long long V28_KEEP_SLOT_BYTES = 12ull << 20;
constexpr unsigned V31_KEEP_OQ_WINDOW = 256u;
#endif

__device__ __forceinline__ void v28_discard_l2_128(const void* gmem_ptr) {
    asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(gmem_ptr) : "memory");
}
// Byte index of transcoded column t_col's scale inside one [t] row of sSF: group (warp_n group, gid)
// owns eight bytes, byte j = the column the warp's B fragment j reads for that gid.
// gate/up: column lc = 32 wn + 8 q + gid of matrix mtx -> j = 4 mtx + q; down: lc = 64 wn + 8 q + gid -> j = q.
template <bool GATEUP>
__device__ __forceinline__ unsigned v28_sf_index(unsigned t_col) {
    if constexpr (GATEUP) {
        const unsigned mtx = t_col >> 7, lc = t_col & 127u;
        return ((((lc >> 5) * 8u) + (lc & 7u)) << 3) + 4u * mtx + ((lc >> 3) & 3u);
    } else {
        return ((((t_col >> 6) * 8u) + (t_col & 7u)) << 3) + ((t_col >> 3) & 7u);
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 1200
__device__ __forceinline__ float v28_scale_ue8m0_pair(float t, unsigned ea, unsigned eb) {
    const unsigned s = ea + eb;
    const unsigned h1 = s >> 1;
    const unsigned h2 = s - h1;
    const unsigned fa_bits = (ea == 255u || eb == 255u) ? 0x7fc00000u : (h1 << 23);
    return (t * __uint_as_float(fa_bits)) * __uint_as_float(h2 << 23);
}
#endif

// V16's QMMA with an explicit B scale byte selector.
template <int BIDA, int BIDB>
__device__ __forceinline__ void v28_qmma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1,
                                         unsigned sa, unsigned sb) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
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
    d[0] += v28_scale_ue8m0_pair(t0, ea_lo, eb_even);
    d[1] += v28_scale_ue8m0_pair(t1, ea_lo, eb_odd);
    d[2] += v28_scale_ue8m0_pair(t2, ea_hi, eb_even);
    d[3] += v28_scale_ue8m0_pair(t3, ea_hi, eb_odd);
#endif
}

// One 16-row slice: the eight QMMAs in V16's order (j = 0..7), scales from the packed pair.
template <int BIDA>
__device__ __forceinline__ void v28_qmma8(float (&acc)[8][4], const unsigned (&a)[4], const unsigned (&b)[8][2],
                                          unsigned sa, unsigned lo, unsigned hi) {
    v28_qmma<BIDA, 0>(acc[0], a, b[0][0], b[0][1], sa, lo);
    v28_qmma<BIDA, 1>(acc[1], a, b[1][0], b[1][1], sa, lo);
    v28_qmma<BIDA, 2>(acc[2], a, b[2][0], b[2][1], sa, lo);
    v28_qmma<BIDA, 3>(acc[3], a, b[3][0], b[3][1], sa, lo);
    v28_qmma<BIDA, 0>(acc[4], a, b[4][0], b[4][1], sa, hi);
    v28_qmma<BIDA, 1>(acc[5], a, b[5][0], b[5][1], sa, hi);
    v28_qmma<BIDA, 2>(acc[6], a, b[6][0], b[6][1], sa, hi);
    v28_qmma<BIDA, 3>(acc[7], a, b[7][0], b[7][1], sa, hi);
}

template <int BIDA>
__device__ __forceinline__ void v28_qmma4_lo(float (&acc)[8][4], const unsigned (&a)[4], const unsigned (&b)[8][2],
                                             unsigned sa, unsigned lo) {
    v28_qmma<BIDA, 0>(acc[0], a, b[0][0], b[0][1], sa, lo);
    v28_qmma<BIDA, 1>(acc[1], a, b[1][0], b[1][1], sa, lo);
    v28_qmma<BIDA, 2>(acc[2], a, b[2][0], b[2][1], sa, lo);
    v28_qmma<BIDA, 3>(acc[3], a, b[3][0], b[3][1], sa, lo);
}

template <int BIDA>
__device__ __forceinline__ void v28_qmma4_hi(float (&acc)[8][4], const unsigned (&a)[4], const unsigned (&b)[8][2],
                                             unsigned sa, unsigned hi) {
    v28_qmma<BIDA, 0>(acc[4], a, b[4][0], b[4][1], sa, hi);
    v28_qmma<BIDA, 1>(acc[5], a, b[5][0], b[5][1], sa, hi);
    v28_qmma<BIDA, 2>(acc[6], a, b[6][0], b[6][1], sa, hi);
    v28_qmma<BIDA, 3>(acc[7], a, b[7][0], b[7][1], sa, hi);
}

__device__ __forceinline__ void v28_st8_evict_last(void* dst, uint2 v) {
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y), "l"(pol)
                 : "memory");
}

__device__ __forceinline__ void v28_st8_evict_first(void* dst, uint2 v) {
    asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y)
                 : "memory");
}

__device__ __forceinline__ void v28_st8_l2_evict_last(void* dst, uint2 v) {
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;\n"
                 :: "l"(dst), "r"(v.x), "r"(v.y), "l"(pol)
                 : "memory");
}

__device__ __forceinline__ void v28_cp16_evict_last(void* smem_ptr, const void* gmem_ptr) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{ .reg .u32 t; .reg .u64 pol;\n"
        "  mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t;\n"
        "  createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n"
        "  cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, 16, pol; }\n"
        : "+r"(s) : "l"(gmem_ptr));
}

__device__ __forceinline__ void v28_cp16_evict_first(void* smem_ptr, const void* gmem_ptr) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{ .reg .u32 t; .reg .u64 pol;\n"
        "  mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t;\n"
        "  createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n"
        "  cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, 16, pol; }\n"
        : "+r"(s) : "l"(gmem_ptr));
}

__device__ __forceinline__ void v28_cp16_evict_last_pred(void* smem_ptr, const void* gmem_ptr, bool pred) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    const int bytes = pred ? 16 : 0;
    asm volatile(
        "{ .reg .u32 t; .reg .u64 pol;\n"
        "  mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t;\n"
        "  createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n"
        "  cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, pol; }\n"
        : "+r"(s) : "l"(gmem_ptr), "r"(bytes));
}

__device__ __forceinline__ void v28_cp16_evict_first_pred(void* smem_ptr, const void* gmem_ptr, bool pred) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    const int bytes = pred ? 16 : 0;
    asm volatile(
        "{ .reg .u32 t; .reg .u64 pol;\n"
        "  mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t;\n"
        "  createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n"
        "  cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, pol; }\n"
        : "+r"(s) : "l"(gmem_ptr), "r"(bytes));
}

__device__ __forceinline__ void v28_cp4_evict_last_pred(void* smem_ptr, const void* gmem_ptr, bool pred) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    const int bytes = pred ? 4 : 0;
    asm volatile(
        "{ .reg .u32 t; .reg .u64 pol;\n"
        "  mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t;\n"
        "  createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n"
        "  cp.async.ca.shared.global.L2::cache_hint [%0], [%1], 4, %2, pol; }\n"
        : "+r"(s) : "l"(gmem_ptr), "r"(bytes));
}

__device__ __forceinline__ void v28_cp4_evict_first_pred(void* smem_ptr, const void* gmem_ptr, bool pred) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    const int bytes = pred ? 4 : 0;
    asm volatile(
        "{ .reg .u32 t; .reg .u64 pol;\n"
        "  mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t;\n"
        "  createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n"
        "  cp.async.ca.shared.global.L2::cache_hint [%0], [%1], 4, %2, pol; }\n"
        : "+r"(s) : "l"(gmem_ptr), "r"(bytes));
}

// Quad-lane (4 adjacent lanes = 1 MX32 block) quantizer from 8 bf16 values in `w` per lane.
// Bit-for-bit identical to v10_mxfp8_block and the v16q4 twin.
__device__ __forceinline__ void v28_mxfp8_quad(const uint4 w, unsigned (&qq)[2], unsigned& ss) {
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

__device__ __forceinline__ unsigned v28_pack_bf162(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    unsigned u;
    memcpy(&u, &v, sizeof(u));
    return u;
}

// Register-direct 4x4 quad transpose + MXFP8 quantizer (0 SMEM, 0 __syncthreads barriers).
__device__ __forceinline__ void v28_mxfp8_reg_quad(
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
    v28_mxfp8_quad(w, qq, ss);
}

// Register-direct 4x4 quad transpose for BF16 epilogue (0 SMEM, 0 __syncthreads, 0 __syncwarp).
__device__ __forceinline__ uint4 v28_bf16_reg_quad(
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
// 16 warps = 4 (M, 32 rows) x 4 (N, 64 B columns: gate/up 32 gate + 32 up).
template <bool GATEUP>
__device__ __forceinline__ void v28_impl(
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
            if (a_row < active_rows_pad16)
                v28_cp16_evict_last_pred(sA + (ks % V14_DA) * (V10_TILE_M * 64) + a_dst, a_src + ks * 64, a_ok);
            if (tid >= 384 && (ks & 1) == 0 && sc_row < active_rows_pad16)          // four k32 scale bytes cover stages ks, ks+1
                v28_cp4_evict_last_pred(&sAS[(ks >> 1) & 1][sc_row], sc_src + ks * 2, sc_ok);
        };
        auto load_b = [&](unsigned ks) {
            constexpr unsigned CH = BCOLS / 16;
            const unsigned mtx = tid / (32 * CH), r = (tid / CH) % 32, cc = tid % CH;
            const unsigned char* src = mtx ? B1 : B0;
            const void* g_b = src + static_cast<unsigned long long>(ks * 32 + r) * N + cta_n + cc * 16;
            v28_cp16_evict_last(&sB[ks % V14_DB][mtx][r][cc * 16], g_b);
            if (tid < BMAT * 4 * CH) {
                const unsigned m2 = tid / (4 * CH), r2 = (tid / CH) % 4, c2 = tid % CH;
                const unsigned char* s = m2 ? S1 : S0;
                const void* g_s = s + static_cast<unsigned long long>(ks * 4 + r2) * N + cta_n + c2 * 16;
                v28_cp16_evict_last(&sS[ks % V14_DB][m2][r2][c2 * 16], g_s);
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
        // Warps 8-15 transcode both 16-blocks while warps 0-7 (always on the critical MMA path) start MMAs at once.
        const unsigned tc_col = tid & 255u;
        const unsigned tc_mtx = tc_col / BCOLS, tc_lc = tc_col % BCOLS;
        const unsigned tc_sf_idx = v28_sf_index<GATEUP>(tc_col);
        auto transcode = [&](unsigned sl, unsigned tb) {
            if (tid >= V14_BC) {
                #pragma unroll
                for (unsigned t = 0; t < 2; ++t) {
                    const unsigned s0 = sS[sl][tc_mtx][2 * t][tc_lc];
                    const unsigned s1 = sS[sl][tc_mtx][2 * t + 1][tc_lc];
                    const unsigned s_max = max(s0 & 0x7fu, s1 & 0x7fu);
                    const int e32 = (s_max >= 8u) ? (static_cast<int>(s_max >> 3) - 7)
                                  : (s_max != 0u) ? (22 - __clz(static_cast<int>(s_max)))
                                                  : 0;
                    const float inv = __int_as_float((127 - e32) << 23);
                    sSF[tb][t][tc_sf_idx] = static_cast<unsigned char>((e32 + 127) & 0xff);
                    #pragma unroll
                    for (unsigned h = 0; h < 2; ++h) {
                        const unsigned s_own = h ? s1 : s0;
                        const unsigned r = __nv_cvt_float_to_fp8(v10_e4m3_to_float(s_own) * inv, __NV_SATFINITE, __NV_E4M3);
                        const uint2 tt = lut[r];
                        unsigned w[4];
                        #pragma unroll
                        for (int g = 0; g < 4; ++g) {
                            const unsigned x = static_cast<unsigned>(sB[sl][tc_mtx][16 * t + 8 * h + 2 * g][tc_lc]) |
                                               (static_cast<unsigned>(sB[sl][tc_mtx][16 * t + 8 * h + 2 * g + 1][tc_lc]) << 8);
                            w[g] = v10_e2m1x4_to_e4m3x4(x, tt.x, tt.y);
                        }
                        *reinterpret_cast<uint4*>(sBt + tb * (V14_BC * 64) + v14_swz(tc_col, 2 * t + h)) =
                            make_uint4(w[0], w[1], w[2], w[3]);
                    }
                }
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
        transcode_unit(0, 0, tid & 255u, tid >> 8);

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
            #pragma unroll
            for (unsigned t = 0; t < 2; ++t) {
                unsigned afr0[4], afr1[4], bfr[8][2];
                const unsigned row_a0 = warp_m + (lane & 15);
                const unsigned addr_a0 = static_cast<unsigned>(__cvta_generic_to_shared(
                    sA + a_slot * (V10_TILE_M * 64) + v14_swz(row_a0, 2 * t + (lane >> 4))));
                v10_ldsm4(addr_a0, afr0[0], afr0[1], afr0[2], afr0[3]);
                // V28: the eight B-column scales this lane's quad feeds (j = 0..7) are contiguous bytes.
                const uint2 sfb = *reinterpret_cast<const uint2*>(&sSF[PAR][t][((warp_id & 3) * 8 + gid) << 3]);
                #pragma unroll
                for (int p = 0; p < 2; ++p) {
                    // gate/up: p 0,1 -> gate cols warp_n + 16p, p 2,3 -> up; down: cols warp_n + 16p
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
                const bool keep_oq = (workspace->rows <= 4096u) || ((task / n_tiles) + 224u >= workspace->work_count);
                v28_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], tig, qq0, ss0);
                if (static_cast<int>(lr0) < valid_rows) {
                    const unsigned long long orow = cta_m + lr0;
                    if (keep_oq) v28_st8_evict_last(oq + orow * N + cta_n + col8, make_uint2(qq0[0], qq0[1]));
                    else *reinterpret_cast<uint2*>(oq + orow * N + cta_n + col8) = make_uint2(qq0[0], qq0[1]);
                    if (tig == 0u) os[orow * (N / 32) + cta_n / 32 + blk] = static_cast<unsigned char>(ss0);
                }
                v28_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], tig, qq1, ss1);
                if (static_cast<int>(lr1) < valid_rows) {
                    const unsigned long long orow = cta_m + lr1;
                    if (keep_oq) v28_st8_evict_last(oq + orow * N + cta_n + col8, make_uint2(qq1[0], qq1[1]));
                    else *reinterpret_cast<uint2*>(oq + orow * N + cta_n + col8) = make_uint2(qq1[0], qq1[1]);
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
                        *reinterpret_cast<uint4*>(C + static_cast<unsigned long long>(cta_m + lr0) * N + cta_n + col8) = w0;
                    }
                    const uint4 w1 = v28_bf16_reg_quad(u1[0], u1[1], u1[2], u1[3], tig);
                    if (static_cast<int>(lr1) < valid_rows) {
                        *reinterpret_cast<uint4*>(C + static_cast<unsigned long long>(cta_m + lr1) * N + cta_n + col8) = w1;
                    }
                }
            }
        }
        if (task + gridDim.x < task_count) __syncthreads();
    }
}

} // namespace

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v28(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v28_impl<true>(aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
                   up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
                   status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v28(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v28_impl<false>(aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
                    expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

// Dynamic shared memory each kernel needs (host side: set MAX_DYNAMIC_SHARED_SIZE_BYTES and launch with it).
extern "C" __global__ void qwen4_moe_compact_v28_smem_bytes(unsigned* out) {
    out[0] = V14Smem<true>::BYTES;
    out[1] = V14Smem<false>::BYTES;
}

// The V16 MXFP8 row quantiser under the V28 tag (same code; the runtime loads quant + GEMMs by tag).
extern "C" __global__ void qwen4_mxfp8_quant_rows_v28(
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

template <bool GATEUP, bool DISCARD = false>
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
    constexpr unsigned N = GATEUP ? oi640::INTER : oi640::HIDDEN;
    constexpr unsigned K = GATEUP ? oi640::HIDDEN : oi640::INTER;
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
        if (item.expert >= oi640::EXPERTS) { if (tid == 0) oi640::fail(status, oi640::ERR_PLAN); return; }
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

} // namespace

// Slices 0..199: gate+up 5x40 slices; slices 200..299: down 10x10 slices. Block 512, 2 CTAs/SM.
extern "C" __global__ __launch_bounds__(512, 2) void qwen4_shared_pretranscode_v28(
    const unsigned long long* __restrict__ tables,
    unsigned char* __restrict__ bt_gateup, unsigned char* __restrict__ sf_gateup,
    unsigned char* __restrict__ bt_down, unsigned char* __restrict__ sf_down
) {
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

    for (unsigned blk = blockIdx.x; blk < 300u; blk += gridDim.x) {
        if (blk < 200u) {
            const unsigned nt = blk / 40u, ks = blk % 40u;
            v28_pretranscode_slice<true>(
                B0_gu, S0_gu, B1_gu, S1_gu, 640u, nt * 128u, ks,
                bt_gateup + static_cast<unsigned long long>(blk) * 16384ull,
                sf_gateup + static_cast<unsigned long long>(blk) * 512ull,
                sB, sS, sBt, sSF, lut);
        } else {
            const unsigned slice = blk - 200u;
            const unsigned nt = slice / 10u, ks = slice % 10u;
            v28_pretranscode_slice<false>(
                B0_dn, S0_dn, B0_dn, S0_dn, 2560u, nt * 256u, ks,
                bt_down + static_cast<unsigned long long>(slice) * 16384ull,
                sf_down + static_cast<unsigned long long>(slice) * 512ull,
                sB, sS, sBt, sSF, lut);
        }
        if (blk + gridDim.x < 300u) __syncthreads();
    }
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
    constexpr unsigned N = oi640::HIDDEN;
    constexpr unsigned K = oi640::INTER;
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
        if (item.expert >= oi640::EXPERTS) { if (tid == 0) oi640::fail(status, oi640::ERR_PLAN); return; }
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
                    // 10 ks slices x 8,192 B (64 x 128 B lines) = 640 lines per nt20 half.
                    for (unsigned i = tid; i < 640u; i += 256u) {
                        const unsigned ks_d = i >> 6, ln = i & 63u;
                        v28_discard_l2_128(bt_base + static_cast<unsigned long long>(ks_d) * 16384ull + ln * 128u);
                    }
                }
                if ((reinterpret_cast<unsigned long long>(sf_buf) & 127ull) == 0ull && tid < 20u) {
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

template <bool GATEUP, bool DOWN_MXFP8, bool PREFETCH, bool W8_ONLY = false>
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
    const bool shape = GATEUP ? (N == oi640::INTER && K == oi640::HIDDEN)
                              : (N == oi640::HIDDEN && K == oi640::INTER);
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
            if (pf_item.m_tile != 0u || pf_item.expert >= oi640::EXPERTS || pf_item.expert == item.expert) return;
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
template <bool GATEUP, bool TS4E> struct V28Smem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V14_BC / BMAT;
    static constexpr unsigned S_ROWS = 4u;
    static constexpr unsigned TS4E_KS = (GATEUP ? oi640::HIDDEN : oi640::INTER) / 64u;
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
template <bool GATEUP, int MODE, int EPI = 0, bool PREFETCH = (MODE != 2), bool TS4E = false>
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
    using L = V28Smem<GATEUP, TS4E>;
    constexpr unsigned N = GATEUP ? oi640::INTER : oi640::HIDDEN;
    constexpr unsigned K = GATEUP ? oi640::HIDDEN : oi640::INTER;
    constexpr unsigned TN = GATEUP ? 128u : 256u;
    constexpr unsigned BMAT = L::BMAT, BCOLS = L::BCOLS, S_ROWS = L::S_ROWS;
    constexpr unsigned TS4E_NUM_K32 = K >> 5;
    constexpr unsigned TS4E_N_TILES = N >> 7;
    constexpr unsigned TS4E_NUM_BKTS = TS4E_NUM_K32 * TS4E_N_TILES;
    constexpr unsigned TS4E_NIBBLES_BYTES = TS4E_NUM_K32 * N;
    constexpr unsigned TS4E_PAGE_BASES_LEN = ((TS4E_NUM_BKTS >> 7) + 4u) & ~3u;
    constexpr unsigned TS4E_PAGE_BASES_OFF = TS4E_NIBBLES_BYTES + N;
    constexpr unsigned TS4E_BKT_OFF_OFF = TS4E_PAGE_BASES_OFF + TS4E_PAGE_BASES_LEN * 4u;
    constexpr unsigned TS4E_BKT_OFF_BYTES_ALIGNED = (((TS4E_NUM_BKTS + 1u) * 2u) + 15u) & ~15u;
    constexpr unsigned TS4E_EXC_OFF = TS4E_BKT_OFF_OFF + TS4E_BKT_OFF_BYTES_ALIGNED;
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
        if (item.expert >= oi640::EXPERTS) { if (threadIdx.x == 0u) oi640::fail(status, oi640::ERR_PLAN); return; }
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
            if (pf_item.expert >= oi640::EXPERTS || pf_item.expert == item.expert) return;
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
                    (static_cast<unsigned long long>(workspace->expanded) * 2640ull) <= V28_KEEP_SLOT_BYTES;
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
                    (static_cast<unsigned long long>(workspace->expanded) * 2640ull) <= V28_KEEP_SLOT_BYTES;
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

template <bool GATEUP, bool TS4E> struct V28M32Smem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V28_M32_BC / BMAT; // 64 (GATEUP) or 128 (DOWN)
    static constexpr unsigned S_ROWS = 4u;
    static constexpr unsigned TS4E_KS = (GATEUP ? oi640::HIDDEN : oi640::INTER) / 64u;
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

template <bool GATEUP, int EPI = 0, bool TS4E = false>
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
    using L = V28M32Smem<GATEUP, TS4E>;
    constexpr unsigned N = GATEUP ? oi640::INTER : oi640::HIDDEN;
    constexpr unsigned K = GATEUP ? oi640::HIDDEN : oi640::INTER;
    constexpr unsigned TN = GATEUP ? 64u : 128u;
    constexpr unsigned BMAT = L::BMAT, BCOLS = L::BCOLS, S_ROWS = L::S_ROWS;
    constexpr unsigned TS4E_B_PER_KS = L::TS4E_B_PER_KS;
    constexpr unsigned TS4E_NUM_K32 = K >> 5;
    constexpr unsigned TS4E_N_TILES = N >> 7;
    constexpr unsigned TS4E_NUM_BKTS = TS4E_NUM_K32 * TS4E_N_TILES;
    constexpr unsigned TS4E_NIBBLES_BYTES = TS4E_NUM_K32 * N;
    constexpr unsigned TS4E_PAGE_BASES_LEN = ((TS4E_NUM_BKTS >> 7) + 4u) & ~3u;
    constexpr unsigned TS4E_PAGE_BASES_OFF = TS4E_NIBBLES_BYTES + N;
    constexpr unsigned TS4E_BKT_OFF_OFF = TS4E_PAGE_BASES_OFF + TS4E_PAGE_BASES_LEN * 4u;
    constexpr unsigned TS4E_BKT_OFF_BYTES_ALIGNED = (((TS4E_NUM_BKTS + 1u) * 2u) + 15u) & ~15u;
    constexpr unsigned TS4E_EXC_OFF = TS4E_BKT_OFF_OFF + TS4E_BKT_OFF_BYTES_ALIGNED;
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
        (static_cast<unsigned long long>(workspace->expanded) * 2640ull) <= V28_KEEP_SLOT_BYTES;

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

// =========================================================================
// Milestone 2: Hierarchical 16-Warp Prefix Scan Worklist Planner (GB10 NVFP4)
// Replaces 512-iteration serial thread-0 prefix loops with 16-warp cooperative
// __shfl_up_sync scan, reducing worklist scan latency by ~67x.
// =========================================================================
namespace {
constexpr unsigned V28_PLAN_MAX_ITEMS =
    (oi640::MAX_EXPANDED + V10_TILE_M - 1) / V10_TILE_M + oi640::EXPERTS;
static_assert(
    V28_PLAN_MAX_ITEMS <= oi640::MAX_WORK_ITEMS,
    "V28 plan items must fit the shared workspace"
);

template <bool INTER, unsigned TILE_M = V10_TILE_M>
__device__ __forceinline__ void v28_plan_impl(
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* __restrict__ status
) {
    __shared__ int s_err;
    __shared__ uint32_t s_rows[oi640::EXPERTS];
    __shared__ uint32_t s_tiles[oi640::EXPERTS];
    __shared__ uint32_t s_prefix[oi640::EXPERTS];
    __shared__ uint32_t s_class[oi640::EXPERTS];
    __shared__ uint32_t s_big, s_single;

    __shared__ uint32_t s_warp_tiles[16];
    __shared__ uint32_t s_warp_big[16];
    __shared__ uint32_t s_warp_single[16];
    __shared__ uint32_t s_warp_active[16];
    __shared__ uint32_t s_warp_max_rows[16];

    __shared__ uint32_t s_warp_base_tiles[16];
    __shared__ uint32_t s_warp_base_big[16];
    __shared__ uint32_t s_warp_base_single[16];

    const unsigned tid = threadIdx.x;
    if (gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != oi640::EXPERTS || blockDim.y != 1 || blockDim.z != 1 ||
        expert_offsets == nullptr || sorted_token_ids == nullptr || contract == nullptr ||
        workspace == nullptr || status == nullptr) return;

    if (tid == 0) {
        s_err = 0;
        if ((reinterpret_cast<uintptr_t>(status) & 3)) { s_err = -1; }
        else if ((reinterpret_cast<uintptr_t>(expert_offsets) & 3) ||
                 (reinterpret_cast<uintptr_t>(sorted_token_ids) & 3) ||
                 (reinterpret_cast<uintptr_t>(contract) & 15) ||
                 (reinterpret_cast<uintptr_t>(workspace) & 15)) { s_err = oi640::ERR_ALIGNMENT; }
        else if (status[0] != oi640::PENDING || !oi640::contract_ok(contract)) { s_err = oi640::ERR_ABI; }
    }
    __syncthreads();
    if (s_err == -1) return;
    if (s_err != 0) { if (tid == 0) status[0] = s_err; return; }

    if (tid == 0) {
        status[0] = oi640::BUILDING;
        workspace->ready_magic = 0;
        workspace->version = oi640::VERSION;
        workspace->rows = contract->rows;
        workspace->expanded = contract->expanded;
        workspace->work_count = 0;
        workspace->active_experts = 0;
        workspace->max_rows = 0;
        workspace->reserved = 0;
        workspace->next_task_dn = 0;
    }
    for (uint32_t i = tid; i < oi640::MAX_WORK_ITEMS; i += blockDim.x)
        workspace->items[i] = {0xffffffffU, 0xffffffffU};

    const uint32_t expanded = contract->expanded;
    for (uint32_t i = tid; i < expanded; i += blockDim.x) {
        const int token = sorted_token_ids[i];
        if (token < 0 || static_cast<uint32_t>(token) >= contract->rows) atomicExch(&s_err, oi640::ERR_TOKEN);
    }
    {
        const int begin = expert_offsets[tid];
        const int end = expert_offsets[tid + 1];
        if (begin < 0 || end < begin || static_cast<uint32_t>(end) > expanded) atomicExch(&s_err, oi640::ERR_OFFSETS);
        const uint32_t rows = (end >= begin) ? static_cast<uint32_t>(end - begin) : 0u;
        s_rows[tid] = rows;
        s_tiles[tid] = (rows + TILE_M - 1u) / TILE_M;
    }
    __syncthreads();
    if (s_err != 0) { if (tid == 0) status[0] = s_err; return; }
    if (tid == 0) {
        if (expert_offsets[0] != 0 || expert_offsets[oi640::EXPERTS] != static_cast<int>(expanded))
            s_err = oi640::ERR_OFFSETS;
    }
    __syncthreads();
    if (s_err != 0) { if (tid == 0) status[0] = s_err; return; }

    const unsigned warp_id = tid >> 5;
    const unsigned lane = tid & 31;
    const uint32_t rows = s_rows[tid];
    const uint32_t tiles = s_tiles[tid];
    const uint32_t big = (tiles > 1) ? tiles : 0u;
    const uint32_t single = (tiles == 1) ? 1u : 0u;
    const uint32_t active = (rows > 0) ? 1u : 0u;
    uint32_t max_r = rows;

    // Step 1: Intra-warp inclusive scan across all 16 warps in parallel
    uint32_t incl_tiles = tiles;
    uint32_t incl_big = big;
    uint32_t incl_single = single;
    uint32_t incl_active = active;

    #pragma unroll
    for (int offset = 1; offset <= 16; offset <<= 1) {
        const uint32_t vt = __shfl_up_sync(0xffffffffu, incl_tiles, offset);
        const uint32_t vb = __shfl_up_sync(0xffffffffu, incl_big, offset);
        const uint32_t vs = __shfl_up_sync(0xffffffffu, incl_single, offset);
        const uint32_t va = __shfl_up_sync(0xffffffffu, incl_active, offset);
        if (lane >= static_cast<unsigned>(offset)) {
            incl_tiles += vt;
            incl_big += vb;
            incl_single += vs;
            incl_active += va;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const uint32_t other = __shfl_down_sync(0xffffffffu, max_r, offset);
        max_r = max(max_r, other);
    }

    if (lane == 31) {
        s_warp_tiles[warp_id] = incl_tiles;
        s_warp_big[warp_id] = incl_big;
        s_warp_single[warp_id] = incl_single;
        s_warp_active[warp_id] = incl_active;
    }
    if (lane == 0) {
        s_warp_max_rows[warp_id] = max_r;
    }
    __syncthreads();

    // Step 2: Inter-warp scan via Warp 0
    if (warp_id == 0) {
        const uint32_t wt = (lane < 16) ? s_warp_tiles[lane] : 0u;
        const uint32_t wb = (lane < 16) ? s_warp_big[lane] : 0u;
        const uint32_t ws = (lane < 16) ? s_warp_single[lane] : 0u;
        const uint32_t wa = (lane < 16) ? s_warp_active[lane] : 0u;
        uint32_t wm = (lane < 16) ? s_warp_max_rows[lane] : 0u;

        uint32_t incl_wt = wt;
        uint32_t incl_wb = wb;
        uint32_t incl_ws = ws;
        uint32_t incl_wa = wa;

        #pragma unroll
        for (int offset = 1; offset <= 8; offset <<= 1) {
            const uint32_t vt = __shfl_up_sync(0xffffffffu, incl_wt, offset);
            const uint32_t vb = __shfl_up_sync(0xffffffffu, incl_wb, offset);
            const uint32_t vs = __shfl_up_sync(0xffffffffu, incl_ws, offset);
            const uint32_t va = __shfl_up_sync(0xffffffffu, incl_wa, offset);
            if (lane >= static_cast<unsigned>(offset)) {
                incl_wt += vt;
                incl_wb += vb;
                incl_ws += vs;
                incl_wa += va;
            }
        }

        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            const uint32_t other = __shfl_down_sync(0xffffffffu, wm, offset);
            wm = max(wm, other);
        }
        wm = __shfl_sync(0xffffffffu, wm, 0);

        if (lane < 16) {
            s_warp_base_tiles[lane] = incl_wt - wt;
            s_warp_base_big[lane] = incl_wb - wb;
            s_warp_base_single[lane] = incl_ws - ws;
        }

        if (lane == 15) {
            if (incl_wt > V28_PLAN_MAX_ITEMS) {
                s_err = oi640::ERR_CAPACITY;
            } else if (incl_wt == 0) {
                s_err = oi640::ERR_PLAN;
            } else {
                workspace->work_count = incl_wt;
                workspace->active_experts = incl_wa;
                workspace->max_rows = wm;
            }
            s_big = incl_wb;
            s_single = incl_ws;
        }
    }
    __syncthreads();
    if (s_err != 0) { if (tid == 0) status[0] = s_err; return; }

    // Step 3: Combine warp base with intra-warp exclusive scan
    const uint32_t base_t = s_warp_base_tiles[warp_id];
    s_prefix[tid] = base_t + (incl_tiles - tiles);
    if constexpr (INTER) {
        if (tiles > 1) {
            s_class[tid] = s_warp_base_big[warp_id] + (incl_big - big);
        } else {
            s_class[tid] = s_warp_base_single[warp_id] + (incl_single - single);
        }
    }

    if (tid == 0) {
        uint64_t hash = 1469598103934665603ULL;
        for (uint32_t e = 0; e < oi640::EXPERTS; ++e) {
            hash = (hash ^ s_rows[e]) * 1099511628211ULL;
        }
        workspace->census_hash = hash;
    }

    if constexpr (INTER) {
        const uint32_t B = s_big, S = s_single;
        if (B == 0 || S == 0) {
            for (uint32_t t = 0; t < tiles; ++t)
                workspace->items[s_prefix[tid] + t] = {tid, t};
        } else if (tiles > 1) {
            for (uint32_t t = 0; t < tiles; ++t) {
                const uint32_t k = s_class[tid] + t;
                workspace->items[k + (k * S) / B] = {tid, t};
            }
        } else if (tiles == 1) {
            const uint32_t j = s_class[tid];
            workspace->items[((j + 1) * B + S - 1) / S + j] = {tid, 0u};
        }
    } else {
        for (uint32_t t = 0; t < tiles; ++t)
            workspace->items[s_prefix[tid] + t] = {tid, t};
    }

    __threadfence();
    __syncthreads();
    if (tid == 0) {
        workspace->ready_magic = oi640::MAGIC;
        __threadfence();
        status[0] = oi640::PLANNED;
    }
}
} // namespace

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_plan_k32_v28(
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* __restrict__ status
) {
    v28_plan_impl<false>(expert_offsets, sorted_token_ids, contract, workspace, status);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_plan_k32_v28_inter(
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* __restrict__ status
) {
    v28_plan_impl<true>(expert_offsets, sorted_token_ids, contract, workspace, status);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_plan_k32_v29_m256(
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* __restrict__ status
) {
    v28_plan_impl<false, 256u>(expert_offsets, sorted_token_ids, contract, workspace, status);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_plan_k32_v29_m256_inter(
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* __restrict__ status
) {
    v28_plan_impl<true, 256u>(expert_offsets, sorted_token_ids, contract, workspace, status);
}

// =========================================================================
// Wave-17 Path A: 0-ULP V29 M256 MoE Prefill (TILE_M = 256, BC = 128, 8x2 warps)
// 512 threads (1 CTA/SM, <= 124 regs/thread, 0 B stack/spills), dynamic SMEM:
//   85,056 B (Flat) / 86,912 B (GATEUP TS4E) / 85,696 B (DOWN TS4E) <= 101,376 B.
// Halves expert-weight DRAM/L2 traffic & transcode ALU (1 k16-unit/thread),
// hides 100% of weight transcode behind QMMA via paired M=256 [TC-0..5]
// instruction-level interleaving (and Warps 8..15 overlap for valid_rows <= 128),
// and reuses single-pass Warp-0 SWAR TS4E decode for BC=128.
// =========================================================================
namespace {

constexpr unsigned V29_M256_TILE_M = 256u;
constexpr unsigned V29_M256_BC = 128u;
constexpr unsigned V29_M256_THREADS = 512u;

template <bool GATEUP, bool TS4E> struct V29M256Smem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V29_M256_BC / BMAT; // 64 (GATEUP) or 128 (DOWN)
    static constexpr unsigned S_ROWS = 4u;
    static constexpr unsigned TS4E_KS = (GATEUP ? oi640::HIDDEN : oi640::INTER) / 64u;
    static constexpr unsigned TS4E_B_PER_KS = GATEUP ? 4u : 2u;
    static constexpr unsigned TS4E_BKTS = TS4E_KS * TS4E_B_PER_KS;
    static constexpr unsigned A = V14_DA * V29_M256_TILE_M * 64u;     // 49,152 B
    static constexpr unsigned B = V14_DB * BMAT * 32u * BCOLS;        // 12,288 B
    static constexpr unsigned S = V14_DB * BMAT * S_ROWS * BCOLS;     // 1,536 B
    static constexpr unsigned AS = 2u * V29_M256_TILE_M * 4u;         // 2,048 B
    static constexpr unsigned BT = 2u * V29_M256_BC * 64u;            // 16,384 B
    static constexpr unsigned SF = 2u * 2u * V29_M256_BC;             // 512 B
    static constexpr unsigned LUT = 256u * 8u;                        // 2,048 B
    static constexpr unsigned AOFF = V29_M256_TILE_M * 4u;            // 1,024 B
    static constexpr unsigned PTRS = 8u * 8u;                         // 64 B
    static constexpr unsigned OFF_B = A;
    static constexpr unsigned OFF_S = OFF_B + B;
    static constexpr unsigned OFF_AS = OFF_S + S;
    static constexpr unsigned OFF_BT = OFF_AS + AS;
    static constexpr unsigned OFF_SF = OFF_BT + BT;
    static constexpr unsigned OFF_LUT = OFF_SF + SF;
    static constexpr unsigned OFF_AOFF = OFF_LUT + LUT;
    static constexpr unsigned OFF_PTRS = OFF_AOFF + AOFF;
    static constexpr unsigned OFF_BASE = OFF_PTRS + PTRS;             // 85,056 B
    static constexpr unsigned OFF_EXC = OFF_BASE + (TS4E ? 128u : 0u);
    static constexpr unsigned OFF_BKT_BEG = OFF_EXC + (TS4E ? (V14_DB * TS4E_B_PER_KS * 32u * 2u) : 0u);
    static constexpr unsigned OFF_BKT_CNT = OFF_BKT_BEG + (TS4E ? (TS4E_BKTS * 4u) : 0u);
    static constexpr unsigned RAW_BYTES = OFF_BKT_CNT + (TS4E ? (TS4E_BKTS * 2u) : 0u);
    static constexpr unsigned BYTES = (RAW_BYTES + 15u) & ~15u;
    unsigned char raw[BYTES];
    static_assert(
        !TS4E ? (BYTES == 85056u) : (GATEUP ? (BYTES == 86912u) : (BYTES == 85696u)),
        "V29M256Smem byte contract"
    );
    static_assert(BYTES <= 101376u, "V29M256Smem fits GB10 dynamic SMEM cap");
};

static_assert(sizeof(V29M256Smem<true, false>) == 85056u && sizeof(V29M256Smem<true, false>) <= 101376u,
              "V29M256Smem<true, false> size");
static_assert(sizeof(V29M256Smem<false, false>) == 85056u && sizeof(V29M256Smem<false, false>) <= 101376u,
              "V29M256Smem<false, false> size");
static_assert(sizeof(V29M256Smem<true, true>) == 86912u && sizeof(V29M256Smem<true, true>) <= 101376u,
              "V29M256Smem<true, true> size");
static_assert(sizeof(V29M256Smem<false, true>) == 85696u && sizeof(V29M256Smem<false, true>) <= 101376u,
              "V29M256Smem<false, true> size");

// Bijective packed 8-byte per-(wn, gid) scale index for the 8x2 warp layout (BC=128):
// - GATEUP (BCOLS=64, TN=64): wn = (lc >> 5) in 0..1, gid = lc & 7, j = (mtx << 2) + ((lc >> 3) & 3) in 0..7
// - DOWN   (BCOLS=128, TN=128): wn = (t_col >> 6) in 0..1, gid = t_col & 7, j = (t_col >> 3) & 7 in 0..7
// Each (wn, gid) reads one 64-bit pair (`LDS.64`, 0 bank conflicts) feeding QMMAs j=0..7.
template <bool GATEUP>
__device__ __forceinline__ unsigned v29_m256_sf_index(unsigned t_col) {
    if constexpr (GATEUP) {
        const unsigned mtx = t_col >> 6, lc = t_col & 63u;
        return ((((lc >> 5) * 8u) + (lc & 7u)) << 3) + (mtx << 2) + ((lc >> 3) & 3u);
    } else {
        return ((((t_col >> 6) * 8u) + (t_col & 7u)) << 3) + ((t_col >> 3) & 7u);
    }
}

__device__ __forceinline__ unsigned long long v29_ld_su64_vol(unsigned s_addr) {
    unsigned long long v;
    asm volatile("ld.shared.u64 %0, [%1];\n" : "=l"(v) : "r"(s_addr));
    return v;
}

__device__ __forceinline__ const unsigned char* v29_ld_sptr(unsigned sbase_ptrs, unsigned idx) {
    return reinterpret_cast<const unsigned char*>(v29_ld_su64_vol(sbase_ptrs + idx * 8u));
}

__device__ __forceinline__ unsigned v29_ld_saoff(unsigned sbase_aoff, unsigned idx) {
    unsigned val;
    asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(val) : "r"(sbase_aoff + idx * 4u));
    return val;
}

__device__ __forceinline__ unsigned v29_ld_su8_vol(unsigned s_addr) {
    unsigned v;
    asm volatile("ld.shared.u8 %0, [%1];\n" : "=r"(v) : "r"(s_addr));
    return v;
}

__device__ __forceinline__ uint2 v29_ld_lut_vol(unsigned sbase_lut, unsigned r) {
    uint2 tt;
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n" : "=r"(tt.x), "=r"(tt.y) : "r"(sbase_lut + r * 8u));
    return tt;
}

template <unsigned OFF, unsigned STRIDE>
__device__ __forceinline__ unsigned v29_ld_sb_pair_off(unsigned s_addr_base) {
    unsigned lo, hi;
    asm volatile(
        "ld.shared.u8 %0, [%2+%3];\n\t"
        "ld.shared.u8 %1, [%2+%4];\n"
        : "=r"(lo), "=r"(hi)
        : "r"(s_addr_base), "n"(OFF), "n"(OFF + STRIDE));
    return lo | (hi << 8);
}

__device__ __forceinline__ void v29_st_sbt_v2_0(unsigned s_addr, unsigned w0, unsigned w1) {
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n" :: "r"(s_addr), "r"(w0), "r"(w1) : "memory");
}

__device__ __forceinline__ void v29_st_sbt_v2_8(unsigned s_addr, unsigned w2, unsigned w3) {
    asm volatile("st.shared.v2.u32 [%0+8], {%1, %2};\n" :: "r"(s_addr), "r"(w2), "r"(w3) : "memory");
}

__device__ __forceinline__ void v29_cp16_s32(unsigned s_addr, const void* gmem_ptr, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "r"(pred ? 16 : 0));
}

__device__ __forceinline__ void v29_cp4_s32(unsigned s_addr, const void* gmem_ptr, bool pred) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "r"(pred ? 4 : 0));
}

__device__ __forceinline__ void v29_cp16_ef_s32(unsigned s_addr, const void* gmem_ptr, bool pred) {
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; shr.u32 t, t, 20; add.u32 %0, %0, t; }\n" : "+r"(s_addr));
    const int bytes = pred ? 16 : 0;
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                 :: "r"(s_addr), "l"(gmem_ptr), "r"(bytes), "l"(pol));
}

template <bool GATEUP, int EPI = 0, bool PREFETCH = true, bool TS4E = false>
__device__ __forceinline__ void v29_m256_impl(
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
    using L = V29M256Smem<GATEUP, TS4E>;
    constexpr unsigned N = GATEUP ? oi640::INTER : oi640::HIDDEN;
    constexpr unsigned K = GATEUP ? oi640::HIDDEN : oi640::INTER;
    constexpr unsigned TN = GATEUP ? 64u : 128u;
    constexpr unsigned BMAT = L::BMAT, BCOLS = L::BCOLS, S_ROWS = L::S_ROWS;
    constexpr unsigned TS4E_B_PER_KS = L::TS4E_B_PER_KS;
    constexpr unsigned TS4E_NUM_K32 = K >> 5;
    constexpr unsigned TS4E_N_TILES = N >> 7;
    constexpr unsigned TS4E_NUM_BKTS = TS4E_NUM_K32 * TS4E_N_TILES;
    constexpr unsigned TS4E_NIBBLES_BYTES = TS4E_NUM_K32 * N;
    constexpr unsigned TS4E_PAGE_BASES_LEN = ((TS4E_NUM_BKTS >> 7) + 4u) & ~3u;
    constexpr unsigned TS4E_PAGE_BASES_OFF = TS4E_NIBBLES_BYTES + N;
    constexpr unsigned TS4E_BKT_OFF_OFF = TS4E_PAGE_BASES_OFF + TS4E_PAGE_BASES_LEN * 4u;
    constexpr unsigned TS4E_BKT_OFF_BYTES_ALIGNED = (((TS4E_NUM_BKTS + 1u) * 2u) + 15u) & ~15u;
    constexpr unsigned TS4E_EXC_OFF = TS4E_BKT_OFF_OFF + TS4E_BKT_OFF_BYTES_ALIGNED;
    constexpr unsigned n_tiles = N / TN; // 10 for GATEUP (I=640), 20 for DOWN (H=2560)
    constexpr unsigned KS = K / 64u;     // 40 for GATEUP, 10 for DOWN
    constexpr unsigned kblocks = K / 32u;
    static_assert((KS & 1u) == 0u && KS >= 4u, "v29_m256 requires even KS >= 4");

    unsigned dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (gridDim.x == 0 || gridDim.x > oi640::FIXED_GRID || blockDim.x != V29_M256_THREADS || dyn_smem < L::BYTES) {
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
    unsigned (*const sAS)[V29_M256_TILE_M] = reinterpret_cast<unsigned (*)[V29_M256_TILE_M]>(smem + L::OFF_AS);
    unsigned char (*const sSF)[2][V29_M256_BC] = reinterpret_cast<unsigned char (*)[2][V29_M256_BC]>(smem + L::OFF_SF);
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
    if (tid < 256u) {
        lut[tid] = V28_E4M3_LUT[tid];
    }

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
        if (item_expert >= oi640::EXPERTS) {
            if (tid == 0u) oi640::fail(reinterpret_cast<int*>(sPtrs[5]), oi640::ERR_PLAN);
            return;
        }
        const int m_start = expert_offsets[item_expert];
        const int M_expert = expert_offsets[item_expert + 1] - m_start;
        const int local = static_cast<int>(item_m_tile * V29_M256_TILE_M);
        const bool range_ok = (m_start >= 0 && M_expert > 0 && local >= 0 && local < M_expert);
        const unsigned cta_m = range_ok ? static_cast<unsigned>(m_start + local) : 0u;
        const int valid_rows = range_ok ? min(M_expert - local, static_cast<int>(V29_M256_TILE_M)) : 0;

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
            if (tid < V29_M256_TILE_M) {
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
        const bool b_evict_first = GATEUP ? (local + static_cast<int>(V29_M256_TILE_M) >= M_expert)
                                          : (item_m_tile == 0u);
        const unsigned col_half = cta_n & 64u;

        auto load_a = [&](unsigned ks) {
            const int active_a_rows = (valid_rows + 15) & ~15;
            const unsigned a_r0 = tid >> 2, a_ch = tid & 3u;
            const unsigned a_dst0 = sbase + (ks % V14_DA) * (V29_M256_TILE_M * 64u) + v14_swz(a_r0, a_ch);
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
            if (valid_rows > 128) {
                const unsigned a_r1 = a_r0 + 128u;
                if (static_cast<int>(a_r1) < active_a_rows) {
                    const bool a_ok1 = static_cast<int>(a_r1) < valid_rows;
                    const unsigned src_row1 = a_ok1 ? (GATEUP ? v29_ld_saoff(sbase + L::OFF_AOFF, a_r1) : (cta_m + a_r1)) : 0u;
                    const void* g1 = aq + static_cast<unsigned long long>(src_row1) * K + ks * 64u + a_ch * 16u;
                    if constexpr (!GATEUP) {
                        if (nt + 1u == n_tiles) v29_cp16_ef_s32(a_dst0 + 8192u, g1, a_ok1);
                        else                    v29_cp16_s32(a_dst0 + 8192u, g1, a_ok1);
                    } else {
                        v29_cp16_s32(a_dst0 + 8192u, g1, a_ok1);
                    }
                }
            }
            if (tid >= 256u && (ks & 1u) == 0u) {
                const unsigned sc_row = tid - 256u;
                if (static_cast<int>(sc_row) < active_a_rows) {
                    const bool sc_ok = static_cast<int>(sc_row) < valid_rows;
                    const unsigned sc_src_row = sc_ok ? (GATEUP ? v29_ld_saoff(sbase + L::OFF_AOFF, sc_row) : (cta_m + sc_row)) : 0u;
                    v29_cp4_s32(sbase + L::OFF_AS + (((ks >> 1) & 1u) * V29_M256_TILE_M + sc_row) * 4u,
                                as + static_cast<unsigned long long>(sc_src_row) * kblocks + ks * 2u, sc_ok);
                }
            }
        };

        auto load_b = [&](unsigned ks) {
            constexpr unsigned CH = BCOLS / 16u;
            constexpr unsigned S_ROWS_LD = TS4E ? 2u : 4u;
            const unsigned sl = ks % V14_DB;
            if (tid < 256u) {
                const unsigned mtx = tid / (32u * CH), r = (tid / CH) % 32u, cc = tid % CH;
                const unsigned char* src = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && mtx) ? 2u : 0u);
                const unsigned off_b = (ks * 32u + r) * N + cta_n + cc * 16u;
                const unsigned dst_b = sbase + L::OFF_B + ((sl * BMAT + mtx) * 32u + r) * BCOLS + cc * 16u;
                if (b_evict_first) v29_cp16_ef_s32(dst_b, src + off_b, true);
                else               v29_cp16_s32(dst_b, src + off_b, true);
            }
            if (tid < BMAT * S_ROWS_LD * CH) {
                const unsigned m2 = tid / (S_ROWS_LD * CH), r2 = (tid / CH) % S_ROWS_LD, c2 = tid % CH;
                const unsigned char* s = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && m2) ? 3u : 1u);
                const unsigned dst_r = TS4E ? (2u + r2) : r2;
                const unsigned off_s = (ks * S_ROWS_LD + r2) * N + cta_n + c2 * 16u;
                const unsigned dst_s = sbase + L::OFF_S + ((sl * BMAT + m2) * S_ROWS + dst_r) * BCOLS + c2 * 16u;
                if (b_evict_first) v29_cp16_ef_s32(dst_s, s + off_s, true);
                else               v29_cp16_s32(dst_s, s + off_s, true);
            }
            if constexpr (TS4E) {
                if (tid < TS4E_B_PER_KS * 4u) {
                    const unsigned b = tid >> 2;
                    const unsigned c = tid & 3u;
                    const unsigned idx = ks * TS4E_B_PER_KS + b;
                    const unsigned cnt = static_cast<unsigned>(sBktCnt[idx]);
                    if (cnt > 0u) {
                        const unsigned beg = sBktBeg[idx];
                        const unsigned pad = beg & 7u;
                        if (pad + cnt <= 32u && (c << 3) < (pad + cnt)) {
                            const unsigned sub = b >> 1;
                            const unsigned char* blob = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && sub) ? 3u : 1u);
                            const unsigned short* exc =
                                reinterpret_cast<const unsigned short*>(blob + TS4E_EXC_OFF);
                            const unsigned dst_e = sbase + L::OFF_EXC + ((sl * TS4E_B_PER_KS + b) * 32u + (c << 3)) * 2u;
                            v29_cp16_s32(dst_e, exc + (beg & ~7u) + (c << 3), true);
                        }
                    }
                }
            }
        };

        auto decode_ts4e_stage = [&](unsigned ks_dec) {
            if (warp_id != 0u) return;
            __syncwarp();
            const unsigned sl = ks_dec % V14_DB;
            // Single-pass carry-free SWAR decode of all 128 columns (32 lanes x 4 contiguous cols).
            const unsigned m2 = GATEUP ? (lane >> 4) : 0u;
            const unsigned lc4 = GATEUP ? ((lane & 15u) << 2) : (lane << 2);
            const unsigned base_w = *reinterpret_cast<const unsigned*>(&sBase[lane << 2]);
            const unsigned nib0 = *reinterpret_cast<const unsigned*>(&sS[sl][m2][2u][lc4]);
            const unsigned nib1 = *reinterpret_cast<const unsigned*>(&sS[sl][m2][3u][lc4]);

            const unsigned c00 = nib0 & 0x0F0F0F0Fu;
            const unsigned e00 = (c00 + 0x01010101u) & 0x10101010u;
            *reinterpret_cast<unsigned*>(&sS[sl][m2][0u][lc4]) = base_w + (c00 & ~(e00 - (e00 >> 4)));

            const unsigned c01 = (nib0 >> 4) & 0x0F0F0F0Fu;
            const unsigned e01 = (c01 + 0x01010101u) & 0x10101010u;
            *reinterpret_cast<unsigned*>(&sS[sl][m2][1u][lc4]) = base_w + (c01 & ~(e01 - (e01 >> 4)));

            const unsigned c10 = nib1 & 0x0F0F0F0Fu;
            const unsigned e10 = (c10 + 0x01010101u) & 0x10101010u;
            *reinterpret_cast<unsigned*>(&sS[sl][m2][2u][lc4]) = base_w + (c10 & ~(e10 - (e10 >> 4)));

            const unsigned c11 = (nib1 >> 4) & 0x0F0F0F0Fu;
            const unsigned e11 = (c11 + 0x01010101u) & 0x10101010u;
            *reinterpret_cast<unsigned*>(&sS[sl][m2][3u][lc4]) = base_w + (c11 & ~(e11 - (e11 >> 4)));
            __syncwarp();

            #pragma unroll 1
            for (unsigned b = 0u; b < TS4E_B_PER_KS; ++b) {
                const unsigned idx = ks_dec * TS4E_B_PER_KS + b;
                const unsigned cnt = static_cast<unsigned>(sBktCnt[idx]);
                if (cnt != 0u) {
                    const unsigned beg = sBktBeg[idx];
                    const unsigned pad = beg & 7u;
                    const unsigned sub = b >> 1;
                    const unsigned t = b & 1u;
                    const unsigned mtx = GATEUP ? sub : 0u;
                    if (pad + cnt <= 32u) {
                        if (lane < cnt) {
                            const unsigned e = static_cast<unsigned>(sExc[sl][b][pad + lane]);
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
            }
        };

        auto prefetch_b = [&](unsigned pf_ks) {
            constexpr unsigned PF_DIST = GATEUP ? 5u : 3u;
            const unsigned pf_raw_wi = raw_wi + PF_DIST;
            if (pf_raw_wi >= wc) return;
            const unsigned pf_wi = (!GATEUP) ? (wc - 1u - pf_raw_wi) : pf_raw_wi;
            const oi640::WorkItem pf_item = workspace->items[pf_wi];
            if (pf_item.expert >= oi640::EXPERTS || pf_item.expert == item_expert) return;
            if constexpr (!GATEUP) {
                if (pf_wi + 1u < wc && workspace->items[pf_wi + 1u].expert == pf_item.expert) return;
            } else {
                if (pf_item.m_tile != 0u) return;
            }
            constexpr unsigned CH = BCOLS / 16u;
            constexpr unsigned S_ROWS_LD = TS4E ? 2u : 4u;
            if (tid < 256u) {
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

        // Single k16-unit transcode (16 weights along K for column t_col, half t_h in {0,1}, k32 block t_blk in {0,1}).
        auto transcode_16 = [&](unsigned sl, unsigned tb, unsigned t_col, unsigned t_h, unsigned t_blk) {
            const unsigned t_mtx = GATEUP ? (t_col >> 6) : 0u;
            const unsigned t_lc = GATEUP ? (t_col & 63u) : t_col;
            const unsigned ss_base = sbase + L::OFF_S + ((sl * BMAT + t_mtx) * S_ROWS + 2u * t_blk) * BCOLS + t_lc;
            const unsigned s_own = v29_ld_su8_vol(ss_base + t_h * BCOLS);
            const unsigned s_oth = v29_ld_su8_vol(ss_base + (t_h ^ 1u) * BCOLS);
            const unsigned s_max = max(s_own & 0x7fu, s_oth & 0x7fu);
            const int e32 = (s_max >= 8u) ? (static_cast<int>(s_max >> 3) - 7)
                          : (s_max != 0u) ? (22 - __clz(static_cast<int>(s_max)))
                                          : 0;
            const unsigned r = v28_norm_e4m3_scale(s_own, e32);
            if (t_h == 0u) {
                sSF[tb][t_blk][v29_m256_sf_index<GATEUP>(t_col)] = static_cast<unsigned char>((e32 + 127) & 0xff);
            }
            const uint2 tt = v29_ld_lut_vol(sbase + L::OFF_LUT, r);
            const unsigned sb_base = sbase + L::OFF_B +
                ((sl * BMAT + t_mtx) * 32u + 16u * t_blk + 8u * t_h) * BCOLS + t_lc;
            const unsigned x0 = v29_ld_sb_pair_off<0u * BCOLS, BCOLS>(sb_base);
            const unsigned x1 = v29_ld_sb_pair_off<2u * BCOLS, BCOLS>(sb_base);
            const unsigned sbt_addr = sbase + L::OFF_BT + tb * (V29_M256_BC * 64u) + v14_swz(t_col, 2u * t_blk + t_h);
            const unsigned w0 = v10_e2m1x4_to_e4m3x4(x0, tt.x, tt.y);
            const unsigned w1 = v10_e2m1x4_to_e4m3x4(x1, tt.x, tt.y);
            v29_st_sbt_v2_0(sbt_addr, w0, w1);
            const unsigned x2 = v29_ld_sb_pair_off<4u * BCOLS, BCOLS>(sb_base);
            const unsigned x3 = v29_ld_sb_pair_off<6u * BCOLS, BCOLS>(sb_base);
            const unsigned w2 = v10_e2m1x4_to_e4m3x4(x2, tt.x, tt.y);
            const unsigned w3 = v10_e2m1x4_to_e4m3x4(x3, tt.x, tt.y);
            v29_st_sbt_v2_8(sbt_addr, w2, w3);
        };

        float acc[2][8][4];
        #pragma unroll
        for (int r = 0; r < 2; ++r)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.0f;

        if constexpr (TS4E) {
            static_assert(KS >= 3u, "TS4E prologue resolves ks=0..2 inline on warp 0");
            constexpr unsigned PROLOGUE_BKTS = 3u * TS4E_B_PER_KS;
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
            if (warp_id == 0u) {
                if (lane < PROLOGUE_BKTS) {
                    resolve_ts4e_bkt(lane);
                }
                __syncwarp();
                if (lane < 8u) {
                    constexpr unsigned BASE_CH = BCOLS / 16u;
                    const unsigned bm = lane / BASE_CH;
                    const unsigned bc = lane % BASE_CH;
                    const unsigned char* bs = v29_ld_sptr(sbase + L::OFF_PTRS, (GATEUP && bm) ? 3u : 1u);
                    v10_cp16(&sBase[lane * 16u],
                             bs + static_cast<unsigned long long>(TS4E_NIBBLES_BYTES) + cta_n + bc * 16u,
                             true);
                }
            } else {
                const unsigned idx = (tid - 32u) + PROLOGUE_BKTS;
                if (idx < KS * TS4E_B_PER_KS) {
                    resolve_ts4e_bkt(idx);
                }
            }
        }

        // prologue: Gp0 = {B0 (+ sBase/sExc0)}, Gp1 = {A0, B1 (+ sExc1)}, Gp2 = {A1, B2 (+ sExc2)}
        load_b(0u);
        v10_commit();
        load_a(0u);
        load_b(1u);
        v10_commit();
        load_a(1u);
        load_b(2u);
        v10_commit();
        v10_wait<2>();
        if constexpr (TS4E) decode_ts4e_stage(0u);
        __syncthreads();
        // 512-thread cooperative prologue transcode (1 k16-unit per thread).
        transcode_16(0u, 0u, tid & 127u, (tid >> 7) & 1u, tid >> 8);

        unsigned sfa[2] = {0u, 0u};
        const unsigned row_a0 = warp_m + (lane & 15u);
        const unsigned row_b0 = warp_n + ((lane >> 4) << 3) + (lane & 7u);
        constexpr unsigned off0 = 0u;
        constexpr unsigned off1 = 1024u;
        constexpr unsigned off2 = GATEUP ? (BCOLS * 64u) : 2048u;
        constexpr unsigned off3 = GATEUP ? (BCOLS * 64u + 1024u) : 3072u;

        // Single unified 64-QMMA step body:
        // - valid_rows <= 128: Warps 8..15 run transcode_16(t_blk=0,1) and return; Warps 0..7 run QMMA with run_tc=false.
        // - valid_rows > 128: inactive warps run transcode_16(tc_blk) and return; active warps interleave [TC-0..3]
        //   across [t=0]'s 4 QMMA groups (peak +2 TC regs during 16-frag window, 0 TC regs live across [t=0]->[t=1]).
        auto step = [&](auto par_tag, unsigned ks) {
            constexpr unsigned PAR = decltype(par_tag)::value;   // ks & 1
            constexpr unsigned NEXT_TB = PAR ^ 1u;
            const bool do_tc = (PAR == 0u) ? true : (ks + 1u < KS);
            v10_wait<1>();                                       // A ks, raw B ks+1 landed
            if constexpr (TS4E) {
                if (do_tc) decode_ts4e_stage(ks + 1u);
            }
            __syncthreads();
            if (ks + 2u < KS) load_a(ks + 2u);
            if (ks + 3u < KS) load_b(ks + 3u);
            else if constexpr (PREFETCH) prefetch_b(ks + 3u - KS);
            v10_commit();

            const unsigned next_sl = (ks + 1u) % V14_DB;
            if (rt_active == 0) {
                if (do_tc) {
                    if (valid_rows <= 128) {
                        if (tid >= 256u) {
                            #pragma unroll 1
                            for (unsigned tb_i = 0u; tb_i < 2u; ++tb_i) {
                                transcode_16(next_sl, NEXT_TB, tid & 127u, (tid >> 7) & 1u, tb_i);
                            }
                        }
                    } else {
                        transcode_16(next_sl, NEXT_TB, tid & 127u, (tid >> 7) & 1u, tid >> 8);
                    }
                }
                return;
            }

            if constexpr (PAR == 0u) {
                const unsigned as_buf = (ks >> 1) & 1u;
                sfa[0] = sAS[as_buf][warp_m + gid + (tig & 1u) * 8u];
                if (rt_active > 1) sfa[1] = sAS[as_buf][warp_m + 16u + gid + (tig & 1u) * 8u];
            }
            const unsigned a_slot = ks % V14_DA;
            const bool run_tc = (valid_rows > 128) && do_tc;

            unsigned afr0[4], afr1[4], bf[4];
            // [t = 0: 4x QMMA Groups = 512 TC cycles/SMSP interleaved with TC-0..TC-3 (0 TC regs live into [t=1])]
            {
                const uint2 sfb = *reinterpret_cast<const uint2*>(&sSF[PAR][0u][sf_off]);
                const unsigned addr_a0 = sbase + a_slot * (V29_M256_TILE_M * 64u) + v14_swz(row_a0, lane >> 4);
                const unsigned addr_b0 = sbase + L::OFF_BT + PAR * (V29_M256_BC * 64u) + v14_swz(row_b0, (lane >> 3) & 1u);
                unsigned tc_r = 0u;
                uint2 tc_tt = make_uint2(0u, 0u);
                auto emit_pair0 = [&](auto p_tag, unsigned sfb_w) {
                    constexpr int P = decltype(p_tag)::value;
                    constexpr int BIDB0 = (P & 1) * 2;
                    constexpr int BIDB1 = BIDB0 + 1;
                    v28_qmma<2 * PAR, BIDB0>(acc[0][2 * P], afr0, bf[0], bf[1], sfa[0], sfb_w);
                    v28_qmma<2 * PAR, BIDB1>(acc[0][2 * P + 1], afr0, bf[2], bf[3], sfa[0], sfb_w);
                    if (rt_active > 1) {
                        v28_qmma<2 * PAR, BIDB0>(acc[1][2 * P], afr1, bf[0], bf[1], sfa[1], sfb_w);
                        v28_qmma<2 * PAR, BIDB1>(acc[1][2 * P + 1], afr1, bf[2], bf[3], sfa[1], sfb_w);
                    }
                };
                v10_ldsm4(addr_a0, afr0[0], afr0[1], afr0[2], afr0[3]);
                if (rt_active > 1) v10_ldsm4(addr_a0 + 1024u, afr1[0], afr1[1], afr1[2], afr1[3]);
                v10_ldsm4(addr_b0 + off0, bf[0], bf[1], bf[2], bf[3]);
                emit_pair0(std::integral_constant<int, 0>{}, sfb.x);

                if (run_tc) {
                    const unsigned tc_col = tid & 127u;
                    const unsigned tc_h = (tid >> 7) & 1u;
                    const unsigned tc_blk = tid >> 8;
                    const unsigned tc_mtx = GATEUP ? (tc_col >> 6) : 0u;
                    const unsigned tc_lc = GATEUP ? (tc_col & 63u) : tc_col;
                    const unsigned ss_base = sbase + L::OFF_S + ((next_sl * BMAT + tc_mtx) * S_ROWS + 2u * tc_blk) * BCOLS + tc_lc;
                    const unsigned s_own = v29_ld_su8_vol(ss_base + tc_h * BCOLS);
                    const unsigned s_oth = v29_ld_su8_vol(ss_base + (tc_h ^ 1u) * BCOLS);
                    const unsigned s_max = max(s_own & 0x7fu, s_oth & 0x7fu);
                    const int e32 = (s_max >= 8u) ? (static_cast<int>(s_max >> 3) - 7)
                                  : (s_max != 0u) ? (22 - __clz(static_cast<int>(s_max)))
                                                  : 0;
                    tc_r = v28_norm_e4m3_scale(s_own, e32);
                    if (tc_h == 0u) {
                        sSF[NEXT_TB][tc_blk][v29_m256_sf_index<GATEUP>(tc_col)] =
                            static_cast<unsigned char>((e32 + 127) & 0xff);
                    }
                }

                asm volatile("" : "+r"(bf[0]), "+r"(bf[1]), "+r"(bf[2]), "+r"(bf[3]));
                v10_ldsm4(addr_b0 + off1, bf[0], bf[1], bf[2], bf[3]);
                emit_pair0(std::integral_constant<int, 1>{}, sfb.x);
                if (run_tc) {
                    tc_tt = v29_ld_lut_vol(sbase + L::OFF_LUT, tc_r);
                }

                asm volatile("" : "+r"(bf[0]), "+r"(bf[1]), "+r"(bf[2]), "+r"(bf[3]));
                v10_ldsm4(addr_b0 + off2, bf[0], bf[1], bf[2], bf[3]);
                emit_pair0(std::integral_constant<int, 2>{}, sfb.y);
                if (run_tc) {
                    const unsigned tc_col = tid & 127u;
                    const unsigned tc_h = (tid >> 7) & 1u;
                    const unsigned tc_blk = tid >> 8;
                    const unsigned tc_mtx = GATEUP ? (tc_col >> 6) : 0u;
                    const unsigned tc_lc = GATEUP ? (tc_col & 63u) : tc_col;
                    const unsigned sb_base = sbase + L::OFF_B +
                        ((next_sl * BMAT + tc_mtx) * 32u + 16u * tc_blk + 8u * tc_h) * BCOLS + tc_lc;
                    const unsigned x0 = v29_ld_sb_pair_off<0u * BCOLS, BCOLS>(sb_base);
                    const unsigned x1 = v29_ld_sb_pair_off<2u * BCOLS, BCOLS>(sb_base);
                    const unsigned w0 = v10_e2m1x4_to_e4m3x4(x0, tc_tt.x, tc_tt.y);
                    const unsigned w1 = v10_e2m1x4_to_e4m3x4(x1, tc_tt.x, tc_tt.y);
                    const unsigned sbt_addr = sbase + L::OFF_BT + NEXT_TB * (V29_M256_BC * 64u) +
                        v14_swz(tc_col, 2u * tc_blk + tc_h);
                    v29_st_sbt_v2_0(sbt_addr, w0, w1);
                }

                asm volatile("" : "+r"(bf[0]), "+r"(bf[1]), "+r"(bf[2]), "+r"(bf[3]));
                v10_ldsm4(addr_b0 + off3, bf[0], bf[1], bf[2], bf[3]);
                emit_pair0(std::integral_constant<int, 3>{}, sfb.y);
                if (run_tc) {
                    const unsigned tc_col = tid & 127u;
                    const unsigned tc_h = (tid >> 7) & 1u;
                    const unsigned tc_blk = tid >> 8;
                    const unsigned tc_mtx = GATEUP ? (tc_col >> 6) : 0u;
                    const unsigned tc_lc = GATEUP ? (tc_col & 63u) : tc_col;
                    const unsigned sb_base = sbase + L::OFF_B +
                        ((next_sl * BMAT + tc_mtx) * 32u + 16u * tc_blk + 8u * tc_h) * BCOLS + tc_lc;
                    const unsigned x2 = v29_ld_sb_pair_off<4u * BCOLS, BCOLS>(sb_base);
                    const unsigned x3 = v29_ld_sb_pair_off<6u * BCOLS, BCOLS>(sb_base);
                    const unsigned w2 = v10_e2m1x4_to_e4m3x4(x2, tc_tt.x, tc_tt.y);
                    const unsigned w3 = v10_e2m1x4_to_e4m3x4(x3, tc_tt.x, tc_tt.y);
                    const unsigned sbt_addr = sbase + L::OFF_BT + NEXT_TB * (V29_M256_BC * 64u) +
                        v14_swz(tc_col, 2u * tc_blk + tc_h);
                    v29_st_sbt_v2_8(sbt_addr, w2, w3);
                }
            }

            // Zero-instruction register-reuse anchor so t=1 reuses t=0's afr0/afr1/bf registers cleanly.
            asm volatile(""
                         : "+r"(afr0[0]), "+r"(afr0[1]), "+r"(afr0[2]), "+r"(afr0[3]),
                           "+r"(bf[0]), "+r"(bf[1]), "+r"(bf[2]), "+r"(bf[3]));
            if (rt_active > 1) {
                asm volatile("" : "+r"(afr1[0]), "+r"(afr1[1]), "+r"(afr1[2]), "+r"(afr1[3]));
            }
            // [t = 1: 4x QMMA Groups = 512 TC cycles/SMSP (0 live transcode regs!)]
            {
                const uint2 sfb = *reinterpret_cast<const uint2*>(&sSF[PAR][1u][sf_off]);
                const unsigned addr_a0 = sbase + a_slot * (V29_M256_TILE_M * 64u) + v14_swz(row_a0, 2u + (lane >> 4));
                const unsigned addr_b0 = sbase + L::OFF_BT + PAR * (V29_M256_BC * 64u) + v14_swz(row_b0, 2u + ((lane >> 3) & 1u));
                auto emit_pair1 = [&](auto p_tag, unsigned sfb_w) {
                    constexpr int P = decltype(p_tag)::value;
                    constexpr int BIDB0 = (P & 1) * 2;
                    constexpr int BIDB1 = BIDB0 + 1;
                    v28_qmma<2 * PAR + 1, BIDB0>(acc[0][2 * P], afr0, bf[0], bf[1], sfa[0], sfb_w);
                    v28_qmma<2 * PAR + 1, BIDB1>(acc[0][2 * P + 1], afr0, bf[2], bf[3], sfa[0], sfb_w);
                    if (rt_active > 1) {
                        v28_qmma<2 * PAR + 1, BIDB0>(acc[1][2 * P], afr1, bf[0], bf[1], sfa[1], sfb_w);
                        v28_qmma<2 * PAR + 1, BIDB1>(acc[1][2 * P + 1], afr1, bf[2], bf[3], sfa[1], sfb_w);
                    }
                };
                v10_ldsm4(addr_a0, afr0[0], afr0[1], afr0[2], afr0[3]);
                if (rt_active > 1) v10_ldsm4(addr_a0 + 1024u, afr1[0], afr1[1], afr1[2], afr1[3]);
                v10_ldsm4(addr_b0 + off0, bf[0], bf[1], bf[2], bf[3]);
                emit_pair1(std::integral_constant<int, 0>{}, sfb.x);
                asm volatile("" : "+r"(bf[0]), "+r"(bf[1]), "+r"(bf[2]), "+r"(bf[3]));
                v10_ldsm4(addr_b0 + off1, bf[0], bf[1], bf[2], bf[3]);
                emit_pair1(std::integral_constant<int, 1>{}, sfb.x);
                asm volatile("" : "+r"(bf[0]), "+r"(bf[1]), "+r"(bf[2]), "+r"(bf[3]));
                v10_ldsm4(addr_b0 + off2, bf[0], bf[1], bf[2], bf[3]);
                emit_pair1(std::integral_constant<int, 2>{}, sfb.y);
                asm volatile("" : "+r"(bf[0]), "+r"(bf[1]), "+r"(bf[2]), "+r"(bf[3]));
                v10_ldsm4(addr_b0 + off3, bf[0], bf[1], bf[2], bf[3]);
                emit_pair1(std::integral_constant<int, 3>{}, sfb.y);
            }
        };

        #pragma unroll 1
        for (unsigned ks = 0u; ks < KS; ks += 2u) {
            step(std::integral_constant<unsigned, 0u>{}, ks);
            step(std::integral_constant<unsigned, 1u>{}, ks + 1u);
        }
        v10_wait<0>();

        const float s2a = s20[item_expert];
        const float s2b = GATEUP ? s21[item_expert] : 0.0f;
        if constexpr (GATEUP) {
            const unsigned blk = warp_id & 1u;
            const unsigned col8 = warp_n + tig * 8u;
            const bool keep_oq = (workspace->rows <= 4096u) || (raw_wi + 128u >= wc);
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= rt_active) break;
                const unsigned lr0 = warp_m + r * 16u + gid, lr1 = lr0 + 8u;
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
                    v28_mxfp8_reg_quad(u0[0], u0[1], u0[2], u0[3], tig, qq0, ss0);
                    if (static_cast<int>(lr0) < valid_rows) {
                        const unsigned long long orow = cta_m + lr0;
                        if (keep_oq) v28_st8_evict_last(oq + orow * N + cta_n + col8, make_uint2(qq0[0], qq0[1]));
                        else         v28_st8_evict_first(oq + orow * N + cta_n + col8, make_uint2(qq0[0], qq0[1]));
                        if (tig == 0u) os[orow * (N / 32u) + cta_n / 32u + blk] = static_cast<unsigned char>(ss0);
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
                    v28_mxfp8_reg_quad(u1[0], u1[1], u1[2], u1[3], tig, qq1, ss1);
                    if (static_cast<int>(lr1) < valid_rows) {
                        const unsigned long long orow = cta_m + lr1;
                        if (keep_oq) v28_st8_evict_last(oq + orow * N + cta_n + col8, make_uint2(qq1[0], qq1[1]));
                        else         v28_st8_evict_first(oq + orow * N + cta_n + col8, make_uint2(qq1[0], qq1[1]));
                        if (tig == 0u) os[orow * (N / 32u) + cta_n / 32u + blk] = static_cast<unsigned char>(ss1);
                    }
                }
                asm volatile("" ::: "memory");
            }
        } else {
            const bool keep_slot =
                (static_cast<unsigned long long>(workspace->expanded) * 2640ull) <= (12ull << 20);
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= rt_active) break;
                const unsigned lr0 = warp_m + r * 16u + gid, lr1 = lr0 + 8u;
                #pragma unroll
                for (unsigned b = 0u; b < 2u; ++b) {
                    const unsigned col = cta_n + warp_n + b * 32u;
                    const unsigned col8 = col + tig * 8u;
                    {
                        unsigned u0[4];
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int j = 4 * static_cast<int>(b) + jj;
                            u0[jj] = v28_pack_bf162(acc[r][j][0] * s2a, acc[r][j][1] * s2a);
                        }
                        if constexpr (EPI == 3) {
                            unsigned qq0[2], ss0;
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
                        } else {
                            const uint4 w0 = v28_bf16_reg_quad(u0[0], u0[1], u0[2], u0[3], tig);
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
                            const uint4 w1 = v28_bf16_reg_quad(u1[0], u1[1], u1[2], u1[3], tig);
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

// Wave-17 Path A: 0-ULP V29 M256 Flat-scale entry points (85,056 B dynamic SMEM, 512 threads).
extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v29_m256(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v29_m256_impl<true, 0, true, false>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v29_m256(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v29_m256_impl<false, 0, true, false>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v29q_m256(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v29_m256_impl<false, 3, true, false>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
        slot_q, slot_s);
}

// Wave-17 Path A: 0-ULP V29 M256 Atlas-TS4E-128 entry points (86,912 B GU / 85,696 B DN SMEM, 512 threads).
extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v29_m256_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ gate_packed_ptrs, const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals, unsigned char* __restrict__ out_q, unsigned char* __restrict__ out_s,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
) {
    v29_m256_impl<true, 0, true, true>(
        aq, as, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs, up_scale_ptrs,
        up_scale2_vals, nullptr, out_q, out_s, expert_offsets, sorted_token_ids, contract, workspace,
        status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v29_m256_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v29_m256_impl<false, 0, true, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, C, nullptr, nullptr,
        expert_offsets, sorted_token_ids, contract, workspace, status, N, K);
}

extern "C" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_k32_v29q_m256_ts4e(
    const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
    const unsigned long long* __restrict__ packed_ptrs, const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals, unsigned char* __restrict__ slot_q, unsigned char* __restrict__ slot_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace,
    int* status, unsigned int N, unsigned int K
) {
    v29_m256_impl<false, 3, true, true>(
        aq, as, packed_ptrs, scale_ptrs, scale2_vals, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, expert_offsets, sorted_token_ids, contract, workspace, status, N, K,
        slot_q, slot_s);
}

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

template <bool GATEUP, bool TS4E> struct V28M16Smem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V28_M16_BC / BMAT; // 64 (GATEUP) or 128 (DOWN)
    static constexpr unsigned S_ROWS = 4u;
    static constexpr unsigned TS4E_KS = (GATEUP ? oi640::HIDDEN : oi640::INTER) / 64u;
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

template <bool GATEUP, int EPI = 0, bool TS4E = false>
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
    using L = V28M16Smem<GATEUP, TS4E>;
    constexpr unsigned N = GATEUP ? oi640::INTER : oi640::HIDDEN;
    constexpr unsigned K = GATEUP ? oi640::HIDDEN : oi640::INTER;
    constexpr unsigned TN = GATEUP ? 64u : 128u;
    constexpr unsigned BMAT = L::BMAT, BCOLS = L::BCOLS, S_ROWS = L::S_ROWS;
    constexpr unsigned TS4E_B_PER_KS = L::TS4E_B_PER_KS;
    constexpr unsigned TS4E_NUM_K32 = K >> 5;
    constexpr unsigned TS4E_N_TILES = N >> 7;
    constexpr unsigned TS4E_NUM_BKTS = TS4E_NUM_K32 * TS4E_N_TILES;
    constexpr unsigned TS4E_NIBBLES_BYTES = TS4E_NUM_K32 * N;
    constexpr unsigned TS4E_PAGE_BASES_LEN = ((TS4E_NUM_BKTS >> 7) + 4u) & ~3u;
    constexpr unsigned TS4E_PAGE_BASES_OFF = TS4E_NIBBLES_BYTES + N;
    constexpr unsigned TS4E_BKT_OFF_OFF = TS4E_PAGE_BASES_OFF + TS4E_PAGE_BASES_LEN * 4u;
    constexpr unsigned TS4E_BKT_OFF_BYTES_ALIGNED = (((TS4E_NUM_BKTS + 1u) * 2u) + 15u) & ~15u;
    constexpr unsigned TS4E_EXC_OFF = TS4E_BKT_OFF_OFF + TS4E_BKT_OFF_BYTES_ALIGNED;

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
        (static_cast<unsigned long long>(workspace->expanded) * 2640ull) <= V28_KEEP_SLOT_BYTES;

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const oi640::WorkItem item = workspace->items[task / n_tiles];
        const unsigned cta_n = (task % n_tiles) * TN;
        if (item.expert >= oi640::EXPERTS) { if (tid == 0u) oi640::fail(status, oi640::ERR_PLAN); return; }
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

template <bool GATEUP, bool TS4E> struct V30Smem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V30_2CTA_BC / BMAT; // 64 (GATEUP) or 128 (DOWN)
    static constexpr unsigned S_ROWS = 4u;
    static constexpr unsigned TS4E_KS = (GATEUP ? oi640::HIDDEN : oi640::INTER) / 64u;
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

template <bool GATEUP, int EPI = 0, bool PREFETCH = true, bool TS4E = false>
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
    using L = V30Smem<GATEUP, TS4E>;
    constexpr unsigned N = GATEUP ? oi640::INTER : oi640::HIDDEN;
    constexpr unsigned K = GATEUP ? oi640::HIDDEN : oi640::INTER;
    constexpr unsigned TN = GATEUP ? 64u : 128u;
    constexpr unsigned BMAT = L::BMAT, BCOLS = L::BCOLS, S_ROWS = L::S_ROWS;
    constexpr unsigned TS4E_B_PER_KS = L::TS4E_B_PER_KS;
    constexpr unsigned TS4E_NUM_K32 = K >> 5;
    constexpr unsigned TS4E_N_TILES = N >> 7;
    constexpr unsigned TS4E_NUM_BKTS = TS4E_NUM_K32 * TS4E_N_TILES;
    constexpr unsigned TS4E_NIBBLES_BYTES = TS4E_NUM_K32 * N;
    constexpr unsigned TS4E_PAGE_BASES_LEN = ((TS4E_NUM_BKTS >> 7) + 4u) & ~3u;
    constexpr unsigned TS4E_PAGE_BASES_OFF = TS4E_NIBBLES_BYTES + N;
    constexpr unsigned TS4E_BKT_OFF_OFF = TS4E_PAGE_BASES_OFF + TS4E_PAGE_BASES_LEN * 4u;
    constexpr unsigned TS4E_BKT_OFF_BYTES_ALIGNED = (((TS4E_NUM_BKTS + 1u) * 2u) + 15u) & ~15u;
    constexpr unsigned TS4E_EXC_OFF = TS4E_BKT_OFF_OFF + TS4E_BKT_OFF_BYTES_ALIGNED;
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
        if (item_expert >= oi640::EXPERTS) {
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
            if (pf_item.expert >= oi640::EXPERTS || pf_item.expert == item_expert) return;
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
                (static_cast<unsigned long long>(workspace->expanded) * 2640ull) <= (12ull << 20);
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

template <bool GATEUP, bool TS4E> struct V31WsSmem {
    static constexpr unsigned BMAT = GATEUP ? 2u : 1u;
    static constexpr unsigned BCOLS = V31_WS_BC / BMAT; // 128 (GU) or 256 (DN)
    static constexpr unsigned S_ROWS = 4u;
    static constexpr unsigned TS4E_KS = (GATEUP ? oi640::HIDDEN : oi640::INTER) / 64u;
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

template <bool GATEUP, int EPI = 0, bool PREFETCH = true, bool TS4E = false>
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
    using L = V31WsSmem<GATEUP, TS4E>;
    constexpr unsigned N = GATEUP ? oi640::INTER : oi640::HIDDEN;
    constexpr unsigned K = GATEUP ? oi640::HIDDEN : oi640::INTER;
    constexpr unsigned TN = GATEUP ? 128u : 256u;
    constexpr unsigned BMAT = L::BMAT, BCOLS = L::BCOLS, S_ROWS = L::S_ROWS;
    constexpr unsigned TS4E_NUM_K32 = K >> 5;
    constexpr unsigned TS4E_N_TILES = N >> 7;
    constexpr unsigned TS4E_NUM_BKTS = TS4E_NUM_K32 * TS4E_N_TILES;
    constexpr unsigned TS4E_NIBBLES_BYTES = TS4E_NUM_K32 * N;
    constexpr unsigned TS4E_PAGE_BASES_LEN = ((TS4E_NUM_BKTS >> 7) + 4u) & ~3u;
    constexpr unsigned TS4E_PAGE_BASES_OFF = TS4E_NIBBLES_BYTES + N;
    constexpr unsigned TS4E_BKT_OFF_OFF = TS4E_PAGE_BASES_OFF + TS4E_PAGE_BASES_LEN * 4u;
    constexpr unsigned TS4E_BKT_OFF_BYTES_ALIGNED = (((TS4E_NUM_BKTS + 1u) * 2u) + 15u) & ~15u;
    constexpr unsigned TS4E_EXC_OFF = TS4E_BKT_OFF_OFF + TS4E_BKT_OFF_BYTES_ALIGNED;
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
            if (item_expert >= oi640::EXPERTS) {
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
                if (pf_item.expert >= oi640::EXPERTS || pf_item.expert == item_expert) return;
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
            if (item_expert >= oi640::EXPERTS) return;
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
                    (static_cast<unsigned long long>(workspace->expanded) * 2640ull) <= V28_KEEP_SLOT_BYTES;
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




