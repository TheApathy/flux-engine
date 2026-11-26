// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <stdint.h>

// Canonical compile-time model geometry traits for Qwen3.8-Flash-Next MoE and
// manifold HyperConnection (mHC) CUDA kernels.
//
// Kernels accept `template <typename Shape = Qwen4FlashNextShape>` on their
// `__device__ __forceinline__` implementation templates and SMEM layout
// descriptors, while thin `extern "C" __global__` entrypoints instantiate the
// exact exported PTX symbols for `Qwen4FlashNextShape`.

template <
    int Hidden,
    int Inter,
    int NumExperts,
    int TopK,
    int MhcStreams = 4,
    int MhcRank = 320
>
struct Qwen4MoeGeometry {
    static_assert(Hidden > 0 && (Hidden % 128) == 0,
                  "Qwen4MoeGeometry: Hidden must be a positive multiple of 128");
    static_assert(Inter > 0 && (Inter % 64) == 0,
                  "Qwen4MoeGeometry: Inter must be a positive multiple of 64");
    static_assert(NumExperts > 0,
                  "Qwen4MoeGeometry: NumExperts must be positive");
    static_assert(TopK > 0 && TopK <= NumExperts,
                  "Qwen4MoeGeometry: TopK must be in [1, NumExperts]");
    static_assert(MhcStreams > 0 && (MhcStreams & (MhcStreams - 1)) == 0,
                  "Qwen4MoeGeometry: MhcStreams must be a power of 2");
    static_assert(MhcRank > 0 && (MhcRank % 64) == 0,
                  "Qwen4MoeGeometry: MhcRank must be a positive multiple of 64");
    static_assert((MhcRank + MhcStreams) % MhcStreams == 0,
                  "Qwen4MoeGeometry: MhcRank + MhcStreams must be divisible by MhcStreams");

    // Signed integer dimensions (used by mHC cluster/GEMM layouts)
    static constexpr int kHidden = Hidden;
    static constexpr int kInter = Inter;
    static constexpr int kNumExperts = NumExperts;
    static constexpr int kTopK = TopK;
    static constexpr int kMhcStreams = MhcStreams;
    static constexpr int kMhcRank = MhcRank;
    static constexpr int kMhcWidth = Hidden * MhcStreams;
    static constexpr int kMhcProjN = MhcRank + MhcStreams;
    static constexpr int kMxBlock = 32;
    static constexpr int kMxScalesPerRow = Hidden / kMxBlock;
    static constexpr int kMhcRankHalf = MhcRank / 2;
    static constexpr int kMhcRankNvfp4Groups = MhcRank / 16;
    static constexpr int kMhcRankMxfp8Groups = MhcRank / 32;

    // Unsigned 32-bit dimensions (used by MoE compact / W4A4 / thin-SK kernels)
    static constexpr unsigned HIDDEN = static_cast<unsigned>(Hidden);
    static constexpr unsigned INTER = static_cast<unsigned>(Inter);
    static constexpr unsigned EXPERTS = static_cast<unsigned>(NumExperts);
    static constexpr unsigned TOP_K = static_cast<unsigned>(TopK);
    static constexpr unsigned MHC_STREAMS = static_cast<unsigned>(MhcStreams);
    static constexpr unsigned MHC_RANK = static_cast<unsigned>(MhcRank);
    static constexpr unsigned MHC_WIDTH = static_cast<unsigned>(Hidden * MhcStreams);
    static constexpr unsigned MHC_PROJ_N = static_cast<unsigned>(MhcRank + MhcStreams);

    // Derived packed / block-scale strides
    static constexpr unsigned HIDDEN_HALF = HIDDEN / 2u;
    static constexpr unsigned INTER_HALF = INTER / 2u;
    static constexpr unsigned HIDDEN_NVFP4_GROUPS = HIDDEN / 16u;
    static constexpr unsigned INTER_NVFP4_GROUPS = INTER / 16u;
    static constexpr unsigned HIDDEN_MXFP8_GROUPS = HIDDEN / 32u;
    static constexpr unsigned INTER_MXFP8_GROUPS = INTER / 32u;
    static constexpr unsigned GATEUP_PAIR_STRIDE = INTER * 2u;
    static constexpr unsigned SLOT_MXFP8_BYTES = HIDDEN + HIDDEN_MXFP8_GROUPS;

    // 64-bit unsigned strides for pointer arithmetic without integer promotion drift
    static constexpr unsigned long long HIDDEN_ULL = static_cast<unsigned long long>(Hidden);
    static constexpr unsigned long long INTER_ULL = static_cast<unsigned long long>(Inter);
    static constexpr unsigned long long GATEUP_PAIR_STRIDE_ULL =
        static_cast<unsigned long long>(Inter) * 2ull;
    static constexpr unsigned long long HIDDEN_MXFP8_GROUPS_ULL =
        static_cast<unsigned long long>(Hidden / 32);
    static constexpr unsigned long long SLOT_MXFP8_BYTES_ULL =
        static_cast<unsigned long long>(Hidden + Hidden / 32);

    template <bool GateUp>
    __host__ __device__ static constexpr unsigned gemm_n() {
        return GateUp ? INTER : HIDDEN;
    }

    template <bool GateUp>
    __host__ __device__ static constexpr unsigned gemm_k() {
        return GateUp ? HIDDEN : INTER;
    }
};

// Compile-time Atlas-TS4E-128 compressed weight-scale blob layout offsets.
template <unsigned N, unsigned K>
struct Qwen4Ts4eLayout {
    static_assert((K % 32u) == 0u, "TS4E requires K divisible by 32");
    static_assert((N % 128u) == 0u, "TS4E requires N divisible by 128");

    static constexpr unsigned NUM_K32 = K >> 5;
    static constexpr unsigned N_TILES = N >> 7;
    static constexpr unsigned NUM_BKTS = NUM_K32 * N_TILES;
    static constexpr unsigned NIBBLES_BYTES = NUM_K32 * N;
    static constexpr unsigned PAGE_BASES_LEN = ((NUM_BKTS >> 7) + 4u) & ~3u;
    static constexpr unsigned PAGE_BASES_OFF = NIBBLES_BYTES + N;
    static constexpr unsigned BKT_OFF_OFF = PAGE_BASES_OFF + PAGE_BASES_LEN * 4u;
    static constexpr unsigned BKT_OFF_BYTES_ALIGNED = (((NUM_BKTS + 1u) * 2u) + 15u) & ~15u;
    static constexpr unsigned EXC_OFF = BKT_OFF_OFF + BKT_OFF_BYTES_ALIGNED;
};

using Qwen4FlashNextShape = Qwen4MoeGeometry<2560, 640, 512, 10, 4, 320>;
