// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// Hierarchical 16-warp prefix-scan worklist planners (V28 TILE_M=128 and V29 TILE_M=256).

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

template <bool INTER, unsigned TILE_M = V10_TILE_M, typename Shape = Qwen4FlashNextShape>
__device__ __forceinline__ void v28_plan_impl(
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* __restrict__ status
) {
    static_assert((Shape::EXPERTS % 32u) == 0u, "v28_plan requires EXPERTS divisible by warp size");
    __shared__ int s_err;
    __shared__ uint32_t s_rows[Shape::EXPERTS];
    __shared__ uint32_t s_tiles[Shape::EXPERTS];
    __shared__ uint32_t s_prefix[Shape::EXPERTS];
    __shared__ uint32_t s_class[Shape::EXPERTS];
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
        blockDim.x != Shape::EXPERTS || blockDim.y != 1 || blockDim.z != 1 ||
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
        if (expert_offsets[0] != 0 || expert_offsets[Shape::EXPERTS] != static_cast<int>(expanded))
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
        for (uint32_t e = 0; e < Shape::EXPERTS; ++e) {
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

