// SPDX-License-Identifier: AGPL-3.0-only
//
// V8 planner (K10 item D, default-off: ATLAS_QWEN4_PREFILL_MOE_PLAN_ORDER=inter): V7's 128-row work items in an
// order that overlaps the expert-weight DRAM stream with the compute-bound L2 re-read tiles.
//
// V7 (qwen4_moe_compact_t_k32_v7.cu) writes items expert-major: expert id ascending, m_tile ascending. Every
// V16/V28 gate/up and V17q down task is (item = task / n_tiles, N tile = task % n_tiles) over a fixed 4096-CTA
// grid. At 8,192 rows real routing gives ~300 single-tile experts (one 128-row tile each: its 1.84 MB of gate/up
// weights stream from DRAM for that one tile) and ~170 multi-tile experts (tiles after the first re-read the
// weights from L2; the largest expert of a layer reaches 50 tiles = 250 consecutive tasks). In expert-id order
// those arrive in runs, so the GPU alternates between DRAM-starved and DRAM-idle phases: replacing every expert's
// weights by expert 0's (L2-resident, timing only) makes gate/up 17-22% faster [M, micro 173].
//
// V8 keeps every multi-tile expert's tiles contiguous and in m_tile order (expert-id order between experts) and
// spreads the single-tile experts evenly between them: after the k-th multi-tile item come
// floor((k+1) S / B) - floor(k S / B) single-tile experts (B = multi-tile items, S = single-tile experts):
//   multi-tile item k (k = items of lower-id multi-tile experts + m_tile) -> position k + floor(k S / B)
//   single-tile expert j (j = lower-id single-tile experts)              -> position ceil((j+1) B / S) + j
// = bench-prefill/qwen4_k10d/orders.py::inter_items (closed form checked against it on 3,600 real layer
// histograms; the K10 D micro checks this kernel's items byte-for-byte). B == 0 or S == 0 writes V7's order.
//
// Exactness: the same item set and the same header (work_count, active_experts, max_rows, census_hash) as V7; each
// GEMM task computes and writes exactly the rows x columns it does under V7 (no task reads another task's output),
// so every output byte is identical - only which CTA runs a tile, and when, changes. Validation, status protocol
// and failure codes are V7's, line for line.
#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define OI640_STEP_K 32
#define OI640_TRANSPOSED 1
#define flash_next_orig_i640_prefill flash_next_f32_transposed_compact_plan_k32_v8
#include "moe_w4a16_orig_i640_compact_prefill.cuh"
namespace oi640 = flash_next_f32_transposed_compact_plan_k32_v8;

namespace {
constexpr unsigned V8_TILE_M = 128;   // = V7_TILE_M: the V16/V28/V17q kernels' tile
constexpr unsigned V8_MAX_ITEMS = (oi640::MAX_EXPANDED + V8_TILE_M - 1) / V8_TILE_M + oi640::EXPERTS;
static_assert(V8_MAX_ITEMS <= oi640::MAX_WORK_ITEMS, "V8 items must fit the shared workspace");
} // namespace

extern "C" __global__ __launch_bounds__(512, 1)
void qwen4_moe_compact_t_plan_k32_v8(
    const int* expert_offsets,
    const int* sorted_token_ids,
    const oi640::Contract* contract,
    oi640::Workspace* workspace,
    int* status
) {
    __shared__ int s_err;
    __shared__ uint32_t s_rows[oi640::EXPERTS];
    __shared__ uint32_t s_tiles[oi640::EXPERTS];
    __shared__ uint32_t s_prefix[oi640::EXPERTS];   // V7 position of the expert's first tile
    __shared__ uint32_t s_class[oi640::EXPERTS];    // s_class[e] = s_tiles[e] > 1 ? big : single;
    __shared__ uint32_t s_w_tiles[16], s_w_big[16], s_w_single[16], s_w_active[16], s_w_max[16];
    __shared__ uint32_t s_big, s_single;
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    if (gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != oi640::EXPERTS || blockDim.y != 1 || blockDim.z != 1 ||
        expert_offsets == nullptr || sorted_token_ids == nullptr || contract == nullptr ||
        workspace == nullptr || status == nullptr) return;
    if (tid == 0) {
        s_err = 0;
        if ((reinterpret_cast<uintptr_t>(status) & 3) ) { s_err = -1; }
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
    if ((reinterpret_cast<uintptr_t>(workspace->items) & 15u) == 0u && (oi640::MAX_WORK_ITEMS & 1u) == 0u) {
        uint4* items4 = reinterpret_cast<uint4*>(workspace->items);
        const uint4 fill4 = make_uint4(0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU);
        for (uint32_t i = tid; i < oi640::MAX_WORK_ITEMS / 2u; i += blockDim.x)
            items4[i] = fill4;
    } else {
        for (uint32_t i = tid; i < oi640::MAX_WORK_ITEMS; i += blockDim.x)
            workspace->items[i] = {0xffffffffU, 0xffffffffU};
    }
    const uint32_t expanded = contract->expanded;
    const uint32_t c_rows = contract->rows;
    if ((reinterpret_cast<uintptr_t>(sorted_token_ids) & 15u) == 0u) {
        const uint32_t vec_end = expanded & ~3u;
        uint32_t bad_mask = 0u;
        for (uint32_t base = tid * 4u; base < vec_end; base += blockDim.x * 4u) {
            const int4 v = __ldg(reinterpret_cast<const int4*>(sorted_token_ids + base));
            if (static_cast<uint32_t>(v.x) >= c_rows ||
                static_cast<uint32_t>(v.y) >= c_rows ||
                static_cast<uint32_t>(v.z) >= c_rows ||
                static_cast<uint32_t>(v.w) >= c_rows) {
                bad_mask = 1u;
            }
        }
        if (bad_mask != 0u) atomicExch(&s_err, oi640::ERR_TOKEN);
        for (uint32_t i = vec_end + tid; i < expanded; i += blockDim.x) {
            const int token = sorted_token_ids[i];
            if (token < 0 || static_cast<uint32_t>(token) >= contract->rows) atomicExch(&s_err, oi640::ERR_TOKEN);
        }
    } else {
        for (uint32_t i = tid; i < expanded; i += blockDim.x) {
            const int token = sorted_token_ids[i];
            if (token < 0 || static_cast<uint32_t>(token) >= contract->rows) atomicExch(&s_err, oi640::ERR_TOKEN);
        }
    }
    uint32_t my_rows = 0u, my_tiles = 0u;
    {
        const int begin = expert_offsets[tid];
        const int end = expert_offsets[tid + 1];
        if (begin < 0 || end < begin || static_cast<uint32_t>(end) > expanded) atomicExch(&s_err, oi640::ERR_OFFSETS);
        const uint32_t rows = (end >= begin) ? static_cast<uint32_t>(end - begin) : 0u;
        s_rows[tid] = rows;
        s_tiles[tid] = (rows + V8_TILE_M - 1) / V8_TILE_M;
        my_rows = rows;
        my_tiles = s_tiles[tid];
        if (tid == 0 && (begin != 0 || expert_offsets[oi640::EXPERTS] != static_cast<int>(expanded))) {
            atomicExch(&s_err, oi640::ERR_OFFSETS);
        }
    }
    const uint32_t my_big = (my_tiles > 1u) ? my_tiles : 0u;
    const uint32_t my_single = (my_tiles == 1u) ? 1u : 0u;
    uint32_t inc_tiles = my_tiles, inc_big = my_big, inc_single = my_single, w_max = my_rows;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const uint32_t t_up = __shfl_up_sync(0xffffffffu, inc_tiles, d);
        const uint32_t b_up = __shfl_up_sync(0xffffffffu, inc_big, d);
        const uint32_t s_up = __shfl_up_sync(0xffffffffu, inc_single, d);
        if (lane >= static_cast<uint32_t>(d)) {
            inc_tiles += t_up;
            inc_big += b_up;
            inc_single += s_up;
        }
        const uint32_t m_oth = __shfl_xor_sync(0xffffffffu, w_max, d);
        w_max = w_max > m_oth ? w_max : m_oth;
    }
    const uint32_t w_active = static_cast<uint32_t>(__popc(__ballot_sync(0xffffffffu, my_rows > 0u)));
    if (lane == 31u) {
        s_w_tiles[warp] = inc_tiles;
        s_w_big[warp] = inc_big;
        s_w_single[warp] = inc_single;
        s_w_active[warp] = w_active;
        s_w_max[warp] = w_max;
    }
    __syncthreads();
    if (s_err != 0) { if (tid == 0) status[0] = s_err; return; }

    if (warp == 0u) {
        uint32_t wt = (lane < 16u) ? s_w_tiles[lane] : 0u;
        uint32_t wb = (lane < 16u) ? s_w_big[lane] : 0u;
        uint32_t ws = (lane < 16u) ? s_w_single[lane] : 0u;
        uint32_t wa = (lane < 16u) ? s_w_active[lane] : 0u;
        uint32_t wm = (lane < 16u) ? s_w_max[lane] : 0u;
        const uint32_t orig_wt = wt, orig_wb = wb, orig_ws = ws;
        #pragma unroll
        for (int d = 1; d < 16; d <<= 1) {
            const uint32_t t_up = __shfl_up_sync(0xffffffffu, wt, d);
            const uint32_t b_up = __shfl_up_sync(0xffffffffu, wb, d);
            const uint32_t s_up = __shfl_up_sync(0xffffffffu, ws, d);
            if (lane >= static_cast<uint32_t>(d)) {
                wt += t_up;
                wb += b_up;
                ws += s_up;
            }
            wa += __shfl_xor_sync(0xffffffffu, wa, d);
            const uint32_t m_oth = __shfl_xor_sync(0xffffffffu, wm, d);
            wm = wm > m_oth ? wm : m_oth;
        }
        if (lane < 16u) {
            s_w_tiles[lane] = wt - orig_wt;
            s_w_big[lane] = wb - orig_wb;
            s_w_single[lane] = ws - orig_ws;
        }
        if (lane == 15u) {
            const uint32_t count = wt, active = wa, max_rows = wm, big = wb, single = ws;
            if (count > V8_MAX_ITEMS) s_err = oi640::ERR_CAPACITY;
            else if (count == 0) s_err = oi640::ERR_PLAN;
            else {
                workspace->work_count = count;
                workspace->active_experts = active;
                workspace->max_rows = max_rows;
            }
            s_big = big;
            s_single = single;
        }
    }
    __shared__ uint64_t s_hash0;
    if (tid == 511u) {
        uint64_t hash = 1469598103934665603ULL;
        #pragma unroll 8
        for (uint32_t e = 0; e < oi640::EXPERTS; ++e) {
            const uint32_t rows = s_rows[e];
            hash = (hash ^ rows) * 1099511628211ULL;
        }
        s_hash0 = hash;
    }
    __syncthreads();
    if (s_err != 0) { if (tid == 0) status[0] = s_err; return; }
    if (tid == 0) {
        const uint64_t hash = s_hash0;
        workspace->census_hash = hash;
    }
    s_prefix[tid] = s_w_tiles[warp] + inc_tiles - my_tiles;
    s_class[tid] = (my_tiles > 1u)
        ? (s_w_big[warp] + inc_big - my_big)
        : (s_w_single[warp] + inc_single - my_single);
    const uint32_t tiles = s_tiles[tid], B = s_big, S = s_single;
    if (B == 0 || S == 0) {
        for (uint32_t t = 0; t < tiles; ++t)                       // degenerate: V7's order
            workspace->items[s_prefix[tid] + t] = {tid, t};
    } else if (tiles > 1) {
        for (uint32_t t = 0; t < tiles; ++t) {
            const uint32_t k = s_class[tid] + t;                    // k S < 2^32: k < 10752, S <= 512
            workspace->items[k + (k * S) / B] = {tid, t};
        }
    } else if (tiles == 1) {
        const uint32_t j = s_class[tid];                            // (j+1) B < 2^32: j < 512, B <= 10752
        workspace->items[((j + 1) * B + S - 1) / S + j] = {tid, 0u};
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        workspace->ready_magic = oi640::MAGIC;
        __threadfence();
        status[0] = oi640::PLANNED;
    }
}

// Fused 1-launch sort + contract + V8/V7 planner (`ATLAS_QWEN4_PREFILL_MOE_FUSED_SORT_PLAN=1`):
// Replaces `moe_sort_by_expert` + 2x `copy_d2d_async` (contract + status) + `qwen4_moe_compact_t_plan_k32_v8`
// with a single 512-thread CTA that histograms `topk_ids` in shared memory, scans `s_counts` in shared memory,
// writes `expert_offsets`, `contract`, `workspace`, and `status`, and scatters `sorted_token_ids` + `token_to_perm`
// across all 512 threads while eliding the dead `sorted_expert_ids` global stores and redundant `sorted_token_ids`
// global re-read.
// `flags` bitfield (backwards-compatible with legacy `use_v8_inter` 0/1):
//   bit 0 (`flags & 1u`): 0 = V7 expert-major order, 1 = V8 interleaved order
//   bit 1 (`flags & 2u`): 0 = 128-row work items (`v28`), 1 = 256-row super-items (`v29_m256`)
extern "C" __global__ __launch_bounds__(512, 1)
void qwen4_moe_sort_plan_t_v8(
    const uint32_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_offsets,
    int* __restrict__ token_to_perm,
    oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* __restrict__ status,
    uint32_t rows,
    uint32_t flags
) {
    __shared__ int s_err;
    __shared__ uint32_t s_counts[oi640::EXPERTS];
    __shared__ uint32_t s_write_pos[oi640::EXPERTS];
    __shared__ uint32_t s_w_rows[16], s_w_tiles[16], s_w_big[16], s_w_single[16];
    __shared__ uint32_t s_w_active[16], s_w_max[16];
    __shared__ uint32_t s_big, s_single;

    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31u;
    if (gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != oi640::EXPERTS || blockDim.y != 1 || blockDim.z != 1 ||
        topk_ids == nullptr || sorted_token_ids == nullptr || expert_offsets == nullptr ||
        token_to_perm == nullptr || contract == nullptr || workspace == nullptr || status == nullptr) return;

    const uint32_t expanded = rows * oi640::TOP_K;
    if (tid == 0) {
        s_err = 0;
        if ((reinterpret_cast<uintptr_t>(status) & 3)) { s_err = -1; }
        else if ((reinterpret_cast<uintptr_t>(topk_ids) & 3) ||
                 (reinterpret_cast<uintptr_t>(sorted_token_ids) & 3) ||
                 (reinterpret_cast<uintptr_t>(expert_offsets) & 3) ||
                 (reinterpret_cast<uintptr_t>(token_to_perm) & 3) ||
                 (reinterpret_cast<uintptr_t>(contract) & 15) ||
                 (reinterpret_cast<uintptr_t>(workspace) & 15)) { s_err = oi640::ERR_ALIGNMENT; }
        else if (!oi640::supported_rows(rows)) { s_err = oi640::ERR_ABI; }
    }
    s_counts[tid] = 0;
    __syncthreads();
    if (s_err == -1) return;
    if (s_err != 0) { if (tid == 0) status[0] = s_err; return; }

    if (tid == 0) {
        contract->magic = oi640::MAGIC;
        contract->version = oi640::VERSION;
        contract->rows = rows;
        contract->hidden = oi640::HIDDEN;
        contract->intermediate = oi640::INTER;
        contract->experts = oi640::EXPERTS;
        contract->top_k = oi640::TOP_K;
        contract->expanded = expanded;
        contract->m_tile = oi640::TILE_M;
        contract->n_tile = oi640::TILE_N;
        contract->fixed_grid = oi640::FIXED_GRID;
        contract->max_work_items = oi640::MAX_WORK_ITEMS;
        contract->workspace_bytes = static_cast<uint32_t>(sizeof(oi640::Workspace));
        contract->reserved0 = 0;
        contract->reserved1 = 0;
        status[0] = oi640::BUILDING;
        workspace->ready_magic = 0;
        workspace->version = oi640::VERSION;
        workspace->rows = rows;
        workspace->expanded = expanded;
        workspace->work_count = 0;
        workspace->active_experts = 0;
        workspace->max_rows = 0;
        workspace->reserved = 0;
        workspace->next_task_dn = 0;
    }

    if ((reinterpret_cast<uintptr_t>(topk_ids) & 15u) == 0u) {
        const uint32_t vec_end = expanded & ~3u;
        for (uint32_t base = tid * 4u; base < vec_end; base += oi640::EXPERTS * 4u) {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(topk_ids + base));
            if ((v.x | v.y | v.z | v.w) < oi640::EXPERTS) {
                atomicAdd(&s_counts[v.x], 1u);
                atomicAdd(&s_counts[v.y], 1u);
                atomicAdd(&s_counts[v.z], 1u);
                atomicAdd(&s_counts[v.w], 1u);
            } else {
                if (v.x < oi640::EXPERTS) atomicAdd(&s_counts[v.x], 1u); else atomicExch(&s_err, oi640::ERR_OFFSETS);
                if (v.y < oi640::EXPERTS) atomicAdd(&s_counts[v.y], 1u); else atomicExch(&s_err, oi640::ERR_OFFSETS);
                if (v.z < oi640::EXPERTS) atomicAdd(&s_counts[v.z], 1u); else atomicExch(&s_err, oi640::ERR_OFFSETS);
                if (v.w < oi640::EXPERTS) atomicAdd(&s_counts[v.w], 1u); else atomicExch(&s_err, oi640::ERR_OFFSETS);
            }
        }
        for (uint32_t i = vec_end + tid; i < expanded; i += oi640::EXPERTS) {
            const uint32_t e = __ldg(&topk_ids[i]);
            if (e < oi640::EXPERTS) atomicAdd(&s_counts[e], 1u);
            else atomicExch(&s_err, oi640::ERR_OFFSETS);
        }
    } else {
        for (uint32_t i = tid; i < expanded; i += oi640::EXPERTS) {
            const uint32_t e = __ldg(&topk_ids[i]);
            if (e < oi640::EXPERTS) {
                atomicAdd(&s_counts[e], 1u);
            } else {
                atomicExch(&s_err, oi640::ERR_OFFSETS);
            }
        }
    }
    __syncthreads();
    if (s_err != 0) { if (tid == 0) status[0] = s_err; return; }

    // 16-warp parallel prefix scan of (rows, tiles, big_tiles, single_tiles) + active/max_rows reduction.
    const bool inter = (flags & 1u) != 0u;
    const uint32_t tile_m = (flags & 2u) ? 256u : V8_TILE_M;
    const uint32_t my_rows = s_counts[tid];
    const uint32_t my_tiles = (my_rows + tile_m - 1u) / tile_m;
    const uint32_t my_big = (my_tiles > 1u) ? my_tiles : 0u;
    const uint32_t my_single = (my_tiles == 1u) ? 1u : 0u;

    uint32_t inc_rows = my_rows;
    uint32_t inc_tiles = my_tiles;
    uint32_t inc_big = my_big;
    uint32_t inc_single = my_single;
    uint32_t w_max = my_rows;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const uint32_t r_up = __shfl_up_sync(0xffffffffu, inc_rows, d);
        const uint32_t t_up = __shfl_up_sync(0xffffffffu, inc_tiles, d);
        const uint32_t b_up = __shfl_up_sync(0xffffffffu, inc_big, d);
        const uint32_t s_up = __shfl_up_sync(0xffffffffu, inc_single, d);
        if (lane >= static_cast<uint32_t>(d)) {
            inc_rows += r_up;
            inc_tiles += t_up;
            inc_big += b_up;
            inc_single += s_up;
        }
        const uint32_t m_oth = __shfl_xor_sync(0xffffffffu, w_max, d);
        w_max = w_max > m_oth ? w_max : m_oth;
    }
    const uint32_t w_active = static_cast<uint32_t>(__popc(__ballot_sync(0xffffffffu, my_rows > 0u)));
    if (lane == 31u) {
        s_w_rows[warp] = inc_rows;
        s_w_tiles[warp] = inc_tiles;
        s_w_big[warp] = inc_big;
        s_w_single[warp] = inc_single;
        s_w_active[warp] = w_active;
        s_w_max[warp] = w_max;
    }
    __syncthreads();

    if (warp == 0u) {
        uint32_t wr = (lane < 16u) ? s_w_rows[lane] : 0u;
        uint32_t wt = (lane < 16u) ? s_w_tiles[lane] : 0u;
        uint32_t wb = (lane < 16u) ? s_w_big[lane] : 0u;
        uint32_t ws = (lane < 16u) ? s_w_single[lane] : 0u;
        uint32_t wa = (lane < 16u) ? s_w_active[lane] : 0u;
        uint32_t wm = (lane < 16u) ? s_w_max[lane] : 0u;
        const uint32_t orig_wr = wr, orig_wt = wt, orig_wb = wb, orig_ws = ws;
        #pragma unroll
        for (int d = 1; d < 16; d <<= 1) {
            const uint32_t r_up = __shfl_up_sync(0xffffffffu, wr, d);
            const uint32_t t_up = __shfl_up_sync(0xffffffffu, wt, d);
            const uint32_t b_up = __shfl_up_sync(0xffffffffu, wb, d);
            const uint32_t s_up = __shfl_up_sync(0xffffffffu, ws, d);
            if (lane >= static_cast<uint32_t>(d)) {
                wr += r_up;
                wt += t_up;
                wb += b_up;
                ws += s_up;
            }
            wa += __shfl_xor_sync(0xffffffffu, wa, d);
            const uint32_t m_oth = __shfl_xor_sync(0xffffffffu, wm, d);
            wm = wm > m_oth ? wm : m_oth;
        }
        if (lane < 16u) {
            s_w_rows[lane] = wr - orig_wr;
            s_w_tiles[lane] = wt - orig_wt;
            s_w_big[lane] = wb - orig_wb;
            s_w_single[lane] = ws - orig_ws;
        }
        if (lane == 15u) {
            expert_offsets[oi640::EXPERTS] = static_cast<int>(wr);
            if (wr != expanded) s_err = oi640::ERR_OFFSETS;
            else if (wt > V8_MAX_ITEMS) s_err = oi640::ERR_CAPACITY;
            else if (wt == 0u) s_err = oi640::ERR_PLAN;
            else {
                workspace->work_count = wt;
                workspace->active_experts = wa;
                workspace->max_rows = wm;
                workspace->census_hash = 0ULL;
            }
            s_big = wb;
            s_single = ws;
        }
    }
    __syncthreads();
    if (s_err != 0) { if (tid == 0) status[0] = s_err; return; }

    const uint32_t my_offset = s_w_rows[warp] + inc_rows - my_rows;
    const uint32_t my_prefix = s_w_tiles[warp] + inc_tiles - my_tiles;
    const uint32_t my_class = (my_tiles > 1u)
        ? (s_w_big[warp] + inc_big - my_big)
        : (s_w_single[warp] + inc_single - my_single);
    s_write_pos[tid] = my_offset;
    expert_offsets[tid] = static_cast<int>(my_offset);
    __syncthreads();

    const uint32_t tiles = my_tiles, B = s_big, S = s_single;
    if (!inter || B == 0 || S == 0) {
        for (uint32_t t = 0; t < tiles; ++t)
            workspace->items[my_prefix + t] = {tid, t};
    } else if (tiles > 1) {
        for (uint32_t t = 0; t < tiles; ++t) {
            const uint32_t k = my_class + t;
            workspace->items[k + (k * S) / B] = {tid, t};
        }
    } else if (tiles == 1) {
        const uint32_t j = my_class;
        workspace->items[((j + 1) * B + S - 1) / S + j] = {tid, 0u};
    }

    if (((reinterpret_cast<uintptr_t>(topk_ids) | reinterpret_cast<uintptr_t>(token_to_perm)) & 15u) == 0u) {
        const uint32_t vec_end = expanded & ~3u;
        for (uint32_t base = tid * 4u; base < vec_end; base += oi640::EXPERTS * 4u) {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(topk_ids + base));
            const uint32_t p0 = atomicAdd(&s_write_pos[v.x], 1u);
            const uint32_t p1 = atomicAdd(&s_write_pos[v.y], 1u);
            const uint32_t p2 = atomicAdd(&s_write_pos[v.z], 1u);
            const uint32_t p3 = atomicAdd(&s_write_pos[v.w], 1u);
            sorted_token_ids[p0] = static_cast<int>((base + 0u) / oi640::TOP_K);
            sorted_token_ids[p1] = static_cast<int>((base + 1u) / oi640::TOP_K);
            sorted_token_ids[p2] = static_cast<int>((base + 2u) / oi640::TOP_K);
            sorted_token_ids[p3] = static_cast<int>((base + 3u) / oi640::TOP_K);
            *reinterpret_cast<uint4*>(token_to_perm + base) = make_uint4(p0, p1, p2, p3);
        }
        for (uint32_t i = vec_end + tid; i < expanded; i += oi640::EXPERTS) {
            const uint32_t e = __ldg(&topk_ids[i]);
            const uint32_t pos = atomicAdd(&s_write_pos[e], 1u);
            sorted_token_ids[pos] = static_cast<int>(i / oi640::TOP_K);
            token_to_perm[i] = static_cast<int>(pos);
        }
    } else {
        for (uint32_t i = tid; i < expanded; i += oi640::EXPERTS) {
            const uint32_t e = __ldg(&topk_ids[i]);
            const uint32_t pos = atomicAdd(&s_write_pos[e], 1u);
            sorted_token_ids[pos] = static_cast<int>(i / oi640::TOP_K);
            token_to_perm[i] = static_cast<int>(pos);
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        workspace->ready_magic = oi640::MAGIC;
        __threadfence();
        status[0] = oi640::PLANNED;
    }
}

namespace {

__device__ __forceinline__ void multi_cta_slice(
    uint32_t total,
    uint32_t parts,
    uint32_t idx,
    uint32_t& start,
    uint32_t& end
) {
    const uint32_t q = total / parts;
    const uint32_t r = total % parts;
    start = idx * q + (idx < r ? idx : r);
    end = start + q + (idx < r ? 1u : 0u);
}

} // namespace

// Multi-CTA 3-phase deterministic sort + contract + V8/V7 planner for wide-SM GPUs
// (`active_sm_count() > 64`, `rows >= 512`):
//   Phase 1 (`grid=G, block=256`): private 512-bin SMEM histogram per CTA over disjoint
//     contiguous slice `[cta_start, cta_end)` of `expanded = rows * 10`, written coalesced
//     to `hist[cta * 512 + e]`.
//   Phase 2 (`grid=1, block=512`): thread `e` exclusive-scans `hist[0..G-1, e]` in-place,
//     runs the exact same 16-warp parallel scan and V8/V7/M256 WorkItem placement as
//     single-CTA `qwen4_moe_sort_plan_t_v8`, and zeroes `next_task_gu`/`next_task_dn`.
//   Phase 3 (`grid=G, block=256`): deterministic ascending-pair-order scatter into
//     `sorted_token_ids` and `token_to_perm` using warp-private SMEM cursors seeded from
//     `expert_offsets[e] + hist[cta * 512 + e]` (zero SMEM atomics, 100% coalesced loads
//     and `token_to_perm` stores, bit-identical offsets/WorkItem[]/contract/workspace).
extern "C" __global__ __launch_bounds__(256, 2)
void qwen4_moe_sort_plan_t_v8_p1_hist(
    const uint32_t* __restrict__ topk_ids,
    uint32_t* __restrict__ hist,
    uint32_t rows
) {
    if (gridDim.x == 0u || gridDim.y != 1u || gridDim.z != 1u ||
        blockDim.x != 256u || blockDim.y != 1u || blockDim.z != 1u ||
        topk_ids == nullptr || hist == nullptr ||
        (reinterpret_cast<uintptr_t>(topk_ids) & 3u) ||
        (reinterpret_cast<uintptr_t>(hist) & 15u) ||
        !oi640::supported_rows(rows)) {
        return;
    }

    __shared__ uint32_t s_counts[oi640::EXPERTS];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    s_counts[tid] = 0u;
    s_counts[tid + 256u] = 0u;
    __syncthreads();

    const uint32_t expanded = rows * oi640::TOP_K;
    uint32_t cta_start, cta_end;
    multi_cta_slice(expanded, gridDim.x, blockIdx.x, cta_start, cta_end);
    const uint32_t len = cta_end - cta_start;
    const uint32_t rounded_end = cta_start + ((len + 255u) & ~255u);
    for (uint32_t base = cta_start; base < rounded_end; base += 256u) {
        const uint32_t i = base + tid;
        uint32_t e = oi640::EXPERTS;
        if (i < cta_end) {
            e = __ldg(&topk_ids[i]);
        }
        const bool ok = (e < oi640::EXPERTS);
        const uint32_t ok_mask = __ballot_sync(0xffffffffu, ok);
        if (ok) {
            const uint32_t match = __match_any_sync(ok_mask, e);
            const uint32_t leader = static_cast<uint32_t>(__ffs(match) - 1);
            if (lane == leader) {
                atomicAdd(&s_counts[e], static_cast<uint32_t>(__popc(match)));
            }
        }
    }
    __syncthreads();

    uint32_t* const out_row = hist + static_cast<size_t>(blockIdx.x) * oi640::EXPERTS;
    out_row[tid] = s_counts[tid];
    out_row[tid + 256u] = s_counts[tid + 256u];
}

extern "C" __global__ __launch_bounds__(512, 1)
void qwen4_moe_sort_plan_t_v8_p2_plan(
    const uint32_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_offsets,
    int* __restrict__ token_to_perm,
    uint32_t* __restrict__ hist,
    oi640::Contract* __restrict__ contract,
    oi640::Workspace* __restrict__ workspace,
    int* __restrict__ status,
    uint32_t rows,
    uint32_t flags,
    uint32_t num_ctas
) {
    __shared__ int s_err;
    __shared__ uint32_t s_w_rows[16], s_w_tiles[16], s_w_big[16], s_w_single[16];
    __shared__ uint32_t s_w_active[16], s_w_max[16];
    __shared__ uint32_t s_big, s_single;

    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31u;
    if (gridDim.x != 1u || gridDim.y != 1u || gridDim.z != 1u ||
        blockDim.x != oi640::EXPERTS || blockDim.y != 1u || blockDim.z != 1u ||
        topk_ids == nullptr || sorted_token_ids == nullptr || expert_offsets == nullptr ||
        token_to_perm == nullptr || hist == nullptr || contract == nullptr ||
        workspace == nullptr || status == nullptr) {
        return;
    }

    const uint32_t expanded = rows * oi640::TOP_K;
    if (tid == 0u) {
        s_err = 0;
        if (reinterpret_cast<uintptr_t>(status) & 3u) { s_err = -1; }
        else if ((reinterpret_cast<uintptr_t>(topk_ids) & 3u) ||
                 (reinterpret_cast<uintptr_t>(sorted_token_ids) & 3u) ||
                 (reinterpret_cast<uintptr_t>(expert_offsets) & 3u) ||
                 (reinterpret_cast<uintptr_t>(token_to_perm) & 3u) ||
                 (reinterpret_cast<uintptr_t>(hist) & 15u) ||
                 (reinterpret_cast<uintptr_t>(contract) & 15u) ||
                 (reinterpret_cast<uintptr_t>(workspace) & 15u)) { s_err = oi640::ERR_ALIGNMENT; }
        else if (!oi640::supported_rows(rows) || num_ctas == 0u) { s_err = oi640::ERR_ABI; }
    }
    __syncthreads();
    if (s_err == -1) return;
    if (s_err != 0) { if (tid == 0u) status[0] = s_err; return; }

    if (tid == 0u) {
        contract->magic = oi640::MAGIC;
        contract->version = oi640::VERSION;
        contract->rows = rows;
        contract->hidden = oi640::HIDDEN;
        contract->intermediate = oi640::INTER;
        contract->experts = oi640::EXPERTS;
        contract->top_k = oi640::TOP_K;
        contract->expanded = expanded;
        contract->m_tile = oi640::TILE_M;
        contract->n_tile = oi640::TILE_N;
        contract->fixed_grid = oi640::FIXED_GRID;
        contract->max_work_items = oi640::MAX_WORK_ITEMS;
        contract->workspace_bytes = static_cast<uint32_t>(sizeof(oi640::Workspace));
        contract->reserved0 = 0;
        contract->reserved1 = 0;
        status[0] = oi640::BUILDING;
        workspace->ready_magic = 0;
        workspace->version = oi640::VERSION;
        workspace->rows = rows;
        workspace->expanded = expanded;
        workspace->work_count = 0;
        workspace->active_experts = 0;
        workspace->max_rows = 0;
        workspace->reserved = 0;
        workspace->next_task_dn = 0;
    }

    uint32_t my_rows = 0u;
    for (uint32_t cta = 0u; cta < num_ctas; ++cta) {
        uint32_t* const slot = &hist[static_cast<size_t>(cta) * oi640::EXPERTS + tid];
        const uint32_t c = *slot;
        *slot = my_rows;
        my_rows += c;
    }

    const bool inter = (flags & 1u) != 0u;
    const uint32_t tile_m = (flags & 2u) ? 256u : V8_TILE_M;
    const uint32_t my_tiles = (my_rows + tile_m - 1u) / tile_m;
    const uint32_t my_big = (my_tiles > 1u) ? my_tiles : 0u;
    const uint32_t my_single = (my_tiles == 1u) ? 1u : 0u;

    uint32_t inc_rows = my_rows;
    uint32_t inc_tiles = my_tiles;
    uint32_t inc_big = my_big;
    uint32_t inc_single = my_single;
    uint32_t w_max = my_rows;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const uint32_t r_up = __shfl_up_sync(0xffffffffu, inc_rows, d);
        const uint32_t t_up = __shfl_up_sync(0xffffffffu, inc_tiles, d);
        const uint32_t b_up = __shfl_up_sync(0xffffffffu, inc_big, d);
        const uint32_t s_up = __shfl_up_sync(0xffffffffu, inc_single, d);
        if (lane >= static_cast<uint32_t>(d)) {
            inc_rows += r_up;
            inc_tiles += t_up;
            inc_big += b_up;
            inc_single += s_up;
        }
        const uint32_t m_oth = __shfl_xor_sync(0xffffffffu, w_max, d);
        w_max = w_max > m_oth ? w_max : m_oth;
    }
    const uint32_t w_active = static_cast<uint32_t>(__popc(__ballot_sync(0xffffffffu, my_rows > 0u)));
    if (lane == 31u) {
        s_w_rows[warp] = inc_rows;
        s_w_tiles[warp] = inc_tiles;
        s_w_big[warp] = inc_big;
        s_w_single[warp] = inc_single;
        s_w_active[warp] = w_active;
        s_w_max[warp] = w_max;
    }
    __syncthreads();

    if (warp == 0u) {
        uint32_t wr = (lane < 16u) ? s_w_rows[lane] : 0u;
        uint32_t wt = (lane < 16u) ? s_w_tiles[lane] : 0u;
        uint32_t wb = (lane < 16u) ? s_w_big[lane] : 0u;
        uint32_t ws = (lane < 16u) ? s_w_single[lane] : 0u;
        uint32_t wa = (lane < 16u) ? s_w_active[lane] : 0u;
        uint32_t wm = (lane < 16u) ? s_w_max[lane] : 0u;
        const uint32_t orig_wr = wr, orig_wt = wt, orig_wb = wb, orig_ws = ws;
        #pragma unroll
        for (int d = 1; d < 16; d <<= 1) {
            const uint32_t r_up = __shfl_up_sync(0xffffffffu, wr, d);
            const uint32_t t_up = __shfl_up_sync(0xffffffffu, wt, d);
            const uint32_t b_up = __shfl_up_sync(0xffffffffu, wb, d);
            const uint32_t s_up = __shfl_up_sync(0xffffffffu, ws, d);
            if (lane >= static_cast<uint32_t>(d)) {
                wr += r_up;
                wt += t_up;
                wb += b_up;
                ws += s_up;
            }
            wa += __shfl_xor_sync(0xffffffffu, wa, d);
            const uint32_t m_oth = __shfl_xor_sync(0xffffffffu, wm, d);
            wm = wm > m_oth ? wm : m_oth;
        }
        if (lane < 16u) {
            s_w_rows[lane] = wr - orig_wr;
            s_w_tiles[lane] = wt - orig_wt;
            s_w_big[lane] = wb - orig_wb;
            s_w_single[lane] = ws - orig_ws;
        }
        if (lane == 15u) {
            expert_offsets[oi640::EXPERTS] = static_cast<int>(wr);
            if (wr != expanded) s_err = oi640::ERR_OFFSETS;
            else if (wt > V8_MAX_ITEMS) s_err = oi640::ERR_CAPACITY;
            else if (wt == 0u) s_err = oi640::ERR_PLAN;
            else {
                workspace->work_count = wt;
                workspace->active_experts = wa;
                workspace->max_rows = wm;
                workspace->census_hash = 0ULL;
            }
            s_big = wb;
            s_single = ws;
        }
    }
    __syncthreads();
    if (s_err != 0) { if (tid == 0u) status[0] = s_err; return; }

    const uint32_t my_offset = s_w_rows[warp] + inc_rows - my_rows;
    const uint32_t my_prefix = s_w_tiles[warp] + inc_tiles - my_tiles;
    const uint32_t my_class = (my_tiles > 1u)
        ? (s_w_big[warp] + inc_big - my_big)
        : (s_w_single[warp] + inc_single - my_single);
    expert_offsets[tid] = static_cast<int>(my_offset);

    const uint32_t tiles = my_tiles, B = s_big, S = s_single;
    if (!inter || B == 0u || S == 0u) {
        for (uint32_t t = 0u; t < tiles; ++t)
            workspace->items[my_prefix + t] = {tid, t};
    } else if (tiles > 1u) {
        for (uint32_t t = 0u; t < tiles; ++t) {
            const uint32_t k = my_class + t;
            workspace->items[k + (k * S) / B] = {tid, t};
        }
    } else if (tiles == 1u) {
        const uint32_t j = my_class;
        workspace->items[((j + 1u) * B + S - 1u) / S + j] = {tid, 0u};
    }

    __threadfence();
    __syncthreads();
    if (tid == 0u) {
        workspace->ready_magic = oi640::MAGIC;
        __threadfence();
        status[0] = oi640::PLANNED;
    }
}

extern "C" __global__ __launch_bounds__(256, 2)
void qwen4_moe_sort_plan_t_v8_p3_scatter(
    const uint32_t* __restrict__ topk_ids,
    const int* __restrict__ expert_offsets,
    const uint32_t* __restrict__ hist,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ token_to_perm,
    const int* __restrict__ status,
    uint32_t rows
) {
    if (gridDim.x == 0u || gridDim.y != 1u || gridDim.z != 1u ||
        blockDim.x != 256u || blockDim.y != 1u || blockDim.z != 1u ||
        topk_ids == nullptr || expert_offsets == nullptr || hist == nullptr ||
        sorted_token_ids == nullptr || token_to_perm == nullptr || status == nullptr ||
        !oi640::supported_rows(rows)) {
        return;
    }

    __shared__ int s_planned;
    __shared__ uint32_t s_w_pos[8][oi640::EXPERTS];
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31u;
    if (tid == 0u) {
        s_planned = (status[0] == oi640::PLANNED) ? 1 : 0;
    }
    #pragma unroll
    for (uint32_t e = lane; e < oi640::EXPERTS; e += 32u) {
        s_w_pos[warp][e] = 0u;
    }
    __syncthreads();
    if (!s_planned) return;

    const uint32_t expanded = rows * oi640::TOP_K;
    uint32_t cta_start, cta_end;
    multi_cta_slice(expanded, gridDim.x, blockIdx.x, cta_start, cta_end);
    uint32_t w_rel_start, w_rel_end;
    multi_cta_slice(cta_end - cta_start, 8u, warp, w_rel_start, w_rel_end);
    const uint32_t w_start = cta_start + w_rel_start;
    const uint32_t w_end = cta_start + w_rel_end;
    const uint32_t w_len = w_end - w_start;
    const uint32_t w_rounded_end = w_start + ((w_len + 31u) & ~31u);

    for (uint32_t base = w_start; base < w_rounded_end; base += 32u) {
        const uint32_t i = base + lane;
        uint32_t e = oi640::EXPERTS;
        if (i < w_end) {
            e = __ldg(&topk_ids[i]);
        }
        const bool ok = (e < oi640::EXPERTS);
        const uint32_t ok_mask = __ballot_sync(0xffffffffu, ok);
        if (ok) {
            const uint32_t match = __match_any_sync(ok_mask, e);
            const uint32_t leader = static_cast<uint32_t>(__ffs(match) - 1);
            if (lane == leader) {
                s_w_pos[warp][e] += static_cast<uint32_t>(__popc(match));
            }
        }
    }
    __syncthreads();

    const uint32_t* const hist_row = hist + static_cast<size_t>(blockIdx.x) * oi640::EXPERTS;
    #pragma unroll
    for (uint32_t pass = 0u; pass < 2u; ++pass) {
        const uint32_t e = tid + pass * 256u;
        uint32_t cursor = static_cast<uint32_t>(expert_offsets[e]) + __ldg(&hist_row[e]);
        #pragma unroll
        for (uint32_t w = 0u; w < 8u; ++w) {
            const uint32_t c = s_w_pos[w][e];
            s_w_pos[w][e] = cursor;
            cursor += c;
        }
    }
    __syncthreads();

    for (uint32_t base = w_start; base < w_rounded_end; base += 32u) {
        const uint32_t i = base + lane;
        uint32_t e = oi640::EXPERTS;
        if (i < w_end) {
            e = __ldg(&topk_ids[i]);
        }
        const bool ok = (e < oi640::EXPERTS);
        const uint32_t ok_mask = __ballot_sync(0xffffffffu, ok);
        if (ok) {
            const uint32_t match = __match_any_sync(ok_mask, e);
            const uint32_t leader = static_cast<uint32_t>(__ffs(match) - 1);
            const uint32_t rank = static_cast<uint32_t>(__popc(match & ((1u << lane) - 1u)));
            uint32_t base_pos = 0u;
            if (lane == leader) {
                base_pos = s_w_pos[warp][e];
                s_w_pos[warp][e] = base_pos + static_cast<uint32_t>(__popc(match));
            }
            base_pos = __shfl_sync(ok_mask, base_pos, leader);
            const uint32_t pos = base_pos + rank;
            sorted_token_ids[pos] = static_cast<int>(i / oi640::TOP_K);
            token_to_perm[i] = static_cast<int>(pos);
        }
    }
}
