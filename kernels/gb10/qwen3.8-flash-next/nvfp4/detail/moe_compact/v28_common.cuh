// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// Compile-time E4M3 LUT and shared PTX/QMMA/L2-hint/MXFP8 helpers for V28-V31 MoE kernels.

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

}  // namespace
