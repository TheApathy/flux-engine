// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// Precomputed-gate MoE row formation and L2-discard cluster combine GEMM variants.

namespace {

template <bool kFullUnroll = false, bool kDiscardSlots = false>
__device__ __forceinline__ void moe_form_rows_mx16(const MoeRows& m, const float* gate_scalars, int row0,
                                                   unsigned int num_tokens) {
    const int r = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int t = row0 + r;
    if (r >= kMoeRows || t >= (int)num_tokens) return;
    const uint64_t pol = policy_evict_first();
    const uint64_t keep_pol = policy_evict_last();
    const float gate = gate_scalars[t];
    const __nv_bfloat16* sh_row = m.shared_out + (size_t)t * kHidden;
    __nv_bfloat16* out_row = m.out + (size_t)t * kHidden;
    const int my_perm = (lane < kTopk) ? m.token_to_perm[(size_t)t * kTopk + lane] : 0;
    const float my_w = (lane < kTopk) ? m.topk_weights[(size_t)t * kTopk + lane] : 0.0f;
    uint32_t s_word[kTopk];
#pragma unroll
    for (int k = 0; k < kTopk; ++k) {
        const size_t sk = (size_t)(uint32_t)__shfl_sync(0xffffffffu, my_perm, k);
        s_word[k] = (lane < (kMxScalesPerRow / 4))
                        ? reinterpret_cast<const uint32_t*>(m.slot_scales + sk * kMxScalesPerRow)[lane]
                        : 0u;
    }
    const int byte_shift = ((lane >> 1) & 3) * 8;
#pragma unroll 1
    for (int u = 0; u < 5; ++u) {
        const int h0 = (u * 32 + lane) * 16;
        const int word_lane = u * 4 + (lane >> 3);
        Vec8 sh0, sh1, out0, out1;
        sh0.raw = load_row16_stream<false>(sh_row + h0, pol);
        sh1.raw = load_row16_stream<false>(sh_row + h0 + 8, pol);
        float acc0[8], acc1[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] = 0.0f;
            acc1[i] = 0.0f;
        }
        if constexpr (kFullUnroll) {
            uint4 raw[kTopk];
#pragma unroll
            for (int k = 0; k < kTopk; ++k) {
                const size_t sk = (size_t)(uint32_t)__shfl_sync(0xffffffffu, my_perm, k);
                raw[k] = load_row16_stream<false>(
                    reinterpret_cast<const __nv_bfloat16*>(static_cast<const uint8_t*>(m.slots) + sk * kHidden + h0),
                    pol);
            }
#pragma unroll
            for (int k = 0; k < kTopk; ++k) {
                const uint32_t ek = (__shfl_sync(0xffffffffu, s_word[k], word_lane) >> byte_shift) & 0xffu;
                const float wk = __shfl_sync(0xffffffffu, my_w, k);
                const float ws = wk * __uint_as_float(ek << 23);
                decode_slot8_acc_ws<true>(make_uint4(raw[k].x, raw[k].y, 0, 0), ws, acc0);
                decode_slot8_acc_ws<true>(make_uint4(raw[k].z, raw[k].w, 0, 0), ws, acc1);
            }
        } else {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                uint4 raw[5];
#pragma unroll
                for (int j = 0; j < 5; ++j) {
                    const int k = half * 5 + j;
                    const size_t sk = (size_t)(uint32_t)__shfl_sync(0xffffffffu, my_perm, k);
                    raw[j] = load_row16_stream<false>(
                        reinterpret_cast<const __nv_bfloat16*>(static_cast<const uint8_t*>(m.slots) + sk * kHidden + h0),
                        pol);
                }
#pragma unroll
                for (int j = 0; j < 5; ++j) {
                    const int k = half * 5 + j;
                    const uint32_t ek = (__shfl_sync(0xffffffffu, s_word[k], word_lane) >> byte_shift) & 0xffu;
                    const float wk = __shfl_sync(0xffffffffu, my_w, k);
                    const float ws = wk * __uint_as_float(ek << 23);
                    decode_slot8_acc_ws<true>(make_uint4(raw[j].x, raw[j].y, 0, 0), ws, acc0);
                    decode_slot8_acc_ws<true>(make_uint4(raw[j].z, raw[j].w, 0, 0), ws, acc1);
                }
            }
        }
        const __nv_bfloat162* sh0_p = reinterpret_cast<const __nv_bfloat162*>(&sh0);
        const __nv_bfloat162* sh1_p = reinterpret_cast<const __nv_bfloat162*>(&sh1);
        __nv_bfloat162* out0_p = reinterpret_cast<__nv_bfloat162*>(&out0);
        __nv_bfloat162* out1_p = reinterpret_cast<__nv_bfloat162*>(&out1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 s0 = __bfloat1622float2(sh0_p[j]);
            const float2 s1 = __bfloat1622float2(sh1_p[j]);
            const float2 o0 = __bfloat1622float2(__floats2bfloat162_rn(acc0[2 * j], acc0[2 * j + 1]));
            const float2 o1 = __bfloat1622float2(__floats2bfloat162_rn(acc1[2 * j], acc1[2 * j + 1]));
            out0_p[j] = __floats2bfloat162_rn(o0.x + gate * s0.x, o0.y + gate * s0.y);
            out1_p[j] = __floats2bfloat162_rn(o1.x + gate * s1.x, o1.y + gate * s1.y);
        }
        store_row16_stream(out_row + h0, out0.raw, keep_pol);
        store_row16_stream(out_row + h0 + 8, out1.raw, keep_pol);
    }
    if constexpr (kDiscardSlots) {
        if ((reinterpret_cast<uintptr_t>(m.slots) & 127u) == 0u) {
            __syncwarp();
            int dlane;
            asm volatile("mov.u32 %0, %%laneid;\n" : "=r"(dlane));
#pragma unroll
            for (int k = 0; k < kTopk; ++k) {
                const size_t sk = (size_t)(uint32_t)__shfl_sync(0xffffffffu, my_perm, k);
                if (dlane < 20) {
                    const unsigned char* p =
                        static_cast<const unsigned char*>(m.slots) + sk * kHidden + (size_t)dlane * 128;
                    asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
                }
            }
        }
    }
}

template <bool kMx>
__device__ __forceinline__ void moe_prologue_gatepre(const MoeRows& m, const float* gate_scalars,
                                                     const MoeScratch& sc, int row0,
                                                     unsigned int num_tokens) {
    if constexpr (kMx) {
        (void)sc;
        moe_form_rows_mx16<true, true>(m, gate_scalars, row0, num_tokens);
    } else {
        moe_routing(m, sc, row0, num_tokens);
        if (threadIdx.x >= kMoeRows * kTopk && threadIdx.x < kMoeRows * kTopk + kMoeRows) {
            const int r = (int)threadIdx.x - kMoeRows * kTopk;
            const int t = row0 + r;
            sc.gate[r] = (t < (int)num_tokens) ? gate_scalars[t] : 0.0f;
        }
        __syncthreads();  // routing and gates visible
        moe_form_rows<false, true>(m, sc, row0, num_tokens);
    }
}

template <bool kMx, typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void combine_moe_tile_gatepre(unsigned char* smem, const MoeRows& m,
                                                         const float* gate_scalars, __nv_bfloat16* hyper,
                                                         __nv_bfloat16* residual, const __nv_bfloat16* folded,
                                                         const Outputs& out, unsigned int num_tokens, float eps) {
    static_assert(Shape::kHidden == kHidden && Shape::kTopK == kTopk && Shape::kMhcProjN == kNTotal,
                  "combine_moe_tile_gatepre Shape matches HcBaseGeom");
    if (threadIdx.x >= kProducers) {
        preload_b_stage(smem, folded, blockIdx.x, 0, 0);
        preload_b_stage(smem, folded, blockIdx.x, 1, 1);
    }
    const MoeScratch sc{reinterpret_cast<int*>(smem + kOffPerm), reinterpret_cast<float*>(smem + kOffWts),
                        reinterpret_cast<float*>(smem + kOffGate), reinterpret_cast<float*>(smem)};
    moe_prologue_gatepre<kMx>(m, gate_scalars, sc, (int)(blockIdx.y * kBM + blockIdx.x * kMoeRows), num_tokens);
    float pre_scale = 0.0f;
    Vec8 pre_rh[4];
    if constexpr (kMx) {
        if (threadIdx.x < kProducers) {
            const int s = blockIdx.x;
            const int t0 = blockIdx.y * kBM;
            const int pr = threadIdx.x >> 2;
            const int pv = threadIdx.x & 3;
            const int ptoken = t0 + pr;
            const bool prow = ptoken < (int)num_tokens;
            const size_t row_idx = prow ? (size_t)ptoken : 0;
            const __nv_bfloat16* hrow = hyper + row_idx * kWidth + s * kHidden + pv * 8;
            pre_scale = prow ? __bfloat162float(residual[row_idx * kWidth + (kWidth - kStreams) + s]) : 0.0f;
            const uint64_t stream_pol = policy_evict_first();
            const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int stg = 0; stg < 4; ++stg) {
                pre_rh[stg].raw = prow ? load_row16_stream<true>(hrow + stg * kBK, stream_pol) : zero;
            }
        }
    }
    cooperative_groups::this_cluster().sync();
    const Operands op{hyper, m.out, residual, nullptr, nullptr};
    prepare_tile<true, true, true, kMx>(smem, op, folded, out, num_tokens, kHidden, eps, pre_scale, pre_rh);
    const int row0_discard = (int)(blockIdx.y * kBM + blockIdx.x * kMoeRows);
    const int valid_rows = min(kMoeRows, max(0, (int)num_tokens - row0_discard));
    unsigned char* base_discard = reinterpret_cast<unsigned char*>(m.out + (size_t)row0_discard * kHidden);
    const unsigned char* sh_discard = reinterpret_cast<const unsigned char*>(m.shared_out + (size_t)row0_discard * kHidden);
    const bool disc_out = ((reinterpret_cast<uintptr_t>(m.out) & 127u) == 0u);
    const bool disc_sh = (m.shared_out != nullptr) && ((reinterpret_cast<uintptr_t>(m.shared_out) & 127u) == 0u);
    const int total_lines = valid_rows * (kHidden * (int)sizeof(__nv_bfloat16) / 128);
    for (int line = threadIdx.x; line < total_lines; line += kThreads) {
        if (disc_out) {
            unsigned char* p = base_discard + (size_t)line * 128;
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
        }
        if (disc_sh) {
            const unsigned char* ps = sh_discard + (size_t)line * 128;
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(ps) : "memory");
        }
    }
}

template <bool kMx, typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void moe_rows_block_gatepre(__nv_bfloat16* moe_rows, const void* slots,
                                                       const uint8_t* slot_scales, const int* token_to_perm,
                                                       const float* topk_weights, const __nv_bfloat16* shared_out,
                                                       const float* gate_scalars, unsigned int num_tokens) {
    static_assert(Shape::kHidden == kHidden && Shape::kTopK == kTopk,
                  "moe_rows_block_gatepre Shape matches HcBaseGeom");
    const MoeRows m{slots, slot_scales, token_to_perm, topk_weights, shared_out, nullptr, nullptr, moe_rows};
    if constexpr (kMx) {
        moe_form_rows_mx16(m, gate_scalars, (int)(blockIdx.x * kMoeRows), num_tokens);
    } else {
        __shared__ int perm[kMoeRows * kTopk];
        __shared__ float wts[kMoeRows * kTopk];
        __shared__ float gate[kMoeRows];
        moe_prologue_gatepre<false>(m, gate_scalars, MoeScratch{perm, wts, gate, nullptr}, (int)(blockIdx.x * kMoeRows),
                                    num_tokens);
    }
}

}  // namespace

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_gemm_ws_gatepre(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    __nv_bfloat16* residual,
    const __nv_bfloat16* __restrict__ folded,
    float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ down,
    unsigned int num_tokens,
    unsigned int rank,
    unsigned int hidden_size,
    unsigned int hc,
    float scale2_down,
    float scale2_inject,
    float eps) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal) __trap();
    check_dyn_smem<kSmemBytes>();
    extern __shared__ __align__(16) unsigned char smem[];
    const Operands op{hyper, core, residual, nullptr, nullptr};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    prepare_tile<false, true>(smem, op, folded, out, num_tokens, hidden_size, eps);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_inject_gemm_ws_l2h_gatepre(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    __nv_bfloat16* residual,
    const __nv_bfloat16* __restrict__ folded,
    float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ down,
    unsigned int num_tokens,
    unsigned int rank,
    unsigned int hidden_size,
    unsigned int hc,
    float scale2_down,
    float scale2_inject,
    float eps) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal) __trap();
    check_dyn_smem<kSmemBytes>();
    extern __shared__ __align__(16) unsigned char smem[];
    const Operands op{hyper, core, residual, nullptr, nullptr};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    prepare_tile<true, true>(smem, op, folded, out, num_tokens, hidden_size, eps);
    if (core != nullptr && ((reinterpret_cast<uintptr_t>(core) & 127u) == 0u)) {
        const int row0_discard = (int)(blockIdx.y * kBM + blockIdx.x * kMoeRows);
        const int valid_rows = min(kMoeRows, max(0, (int)num_tokens - row0_discard));
        const unsigned char* base_discard = reinterpret_cast<const unsigned char*>(core + (size_t)row0_discard * kHidden);
        const int total_lines = valid_rows * (kHidden * (int)sizeof(__nv_bfloat16) / 128);
        for (int line = threadIdx.x; line < total_lines; line += kThreads) {
            const unsigned char* p = base_discard + (size_t)line * 128;
            asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(p) : "memory");
        }
    }
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_combine_moe_gemm_gatepre(
    __nv_bfloat16* __restrict__ hyper,
    __nv_bfloat16* residual,
    const __nv_bfloat16* __restrict__ folded,
    float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ down,
    __nv_bfloat16* moe_rows,
    const __nv_bfloat16* __restrict__ expert_output,
    const uint8_t* __restrict__ slot_scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int num_tokens,
    unsigned int rank,
    unsigned int hidden_size,
    unsigned int hc,
    unsigned int topk,
    float scale2_down,
    float scale2_inject,
    float eps) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || topk != kTopk) __trap();
    check_dyn_smem<kSmemBytes>();
    extern __shared__ __align__(16) unsigned char smem[];
    const MoeRows m{expert_output, slot_scales, token_to_perm, topk_weights, shared_out, nullptr, nullptr, moe_rows};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    combine_moe_tile_gatepre<false>(smem, m, gate_scalars, hyper, residual, folded, out, num_tokens, eps);
}

extern "C" __global__ void __cluster_dims__(kStreams, 1, 1) __launch_bounds__(kThreads, 1)
qwen4_hc_combine_moe_gemm_mx_gatepre(
    __nv_bfloat16* __restrict__ hyper,
    __nv_bfloat16* residual,
    const __nv_bfloat16* __restrict__ folded,
    float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ down,
    __nv_bfloat16* moe_rows,
    const uint8_t* __restrict__ expert_output,
    const uint8_t* __restrict__ slot_scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int num_tokens,
    unsigned int rank,
    unsigned int hidden_size,
    unsigned int hc,
    unsigned int topk,
    float scale2_down,
    float scale2_inject,
    float eps) {
    if (hidden_size != kHidden || hc != kStreams || rank + hc != kNTotal || topk != kTopk ||
        slot_scales == nullptr || ((reinterpret_cast<uintptr_t>(slot_scales) & 3u) != 0u)) {
        __trap();
    }
    check_dyn_smem<kSmemBytes>();
    extern __shared__ __align__(16) unsigned char smem[];
    const MoeRows m{expert_output, slot_scales, token_to_perm, topk_weights, shared_out, nullptr, nullptr, moe_rows};
    const Outputs out{rstd, down, residual, rank, hc, scale2_down, scale2_inject};
    combine_moe_tile_gatepre<true>(smem, m, gate_scalars, hyper, residual, folded, out, num_tokens, eps);
}

extern "C" __global__ void __launch_bounds__(kThreads, 2)
qwen4_hc_moe_rows_gatepre(
    __nv_bfloat16* __restrict__ moe_rows,
    const __nv_bfloat16* __restrict__ expert_output,
    const uint8_t* __restrict__ slot_scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk) {
    if (hidden_size != kHidden || topk != kTopk) __trap();
    moe_rows_block_gatepre<false>(moe_rows, expert_output, slot_scales, token_to_perm, topk_weights, shared_out,
                                  gate_scalars, num_tokens);
}

extern "C" __global__ void __launch_bounds__(kThreads, 2)
qwen4_hc_moe_rows_mx_gatepre(
    __nv_bfloat16* __restrict__ moe_rows,
    const uint8_t* __restrict__ expert_output,
    const uint8_t* __restrict__ slot_scales,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_scalars,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int topk) {
    if (hidden_size != kHidden || topk != kTopk ||
        slot_scales == nullptr || ((reinterpret_cast<uintptr_t>(slot_scales) & 3u) != 0u)) {
        __trap();
    }
    moe_rows_block_gatepre<true>(moe_rows, expert_output, slot_scales, token_to_perm, topk_weights, shared_out,
                                 gate_scalars, num_tokens);
}

