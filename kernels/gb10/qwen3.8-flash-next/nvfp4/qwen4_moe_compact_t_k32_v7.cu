// SPDX-License-Identifier: AGPL-3.0-only
//
// V7 of the transposed compact-worklist MoE GEMM, for wide (>= 8192-row) prefill
// chunks. Same ABI as V2..V6 (contract, workspace, status, pointer tables, packed
// [K/2][N] NVFP4 weights with [K/16][N] E4M3 scales and a per-expert FP32 scale2),
// but its own planner: work items are 128-row tiles (V7_TILE_M) instead of 64.
//
// What changed versus V4, and why (8192 rows give ~160 routed rows per expert, so the
// kernel is MMA/shared-memory bound, not weight-stream bound as at 2048 rows):
//  * B is never materialised in shared memory. Each lane builds its mma.sync B
//    fragments straight from the raw packed bytes: one byte holds the (2*tig, 2*tig+1)
//    k-pair of column gid, which `cvt.rn.f16x2.e2m1x2` (F2FP) expands in one
//    instruction, multiplied by the E4M3 block scale in f16 (exact: <= 6 significant
//    bits) and moved to bf16 (exact). V4's cooperative dequant pass (LUT loads, FP32
//    multiplies, BF16 stores, a second barrier and ldmatrix.trans re-reads) is gone.
//  * 128x128 CTA tile, 8 warps as 2 (M) x 4 (N), warp tile 64x32: every B fragment
//    feeds four m16 MMAs, so the per-lane dequant is amortised over 64 rows.
//  * scale2 is applied to the FP32 accumulator in the epilogue. The weight operand is
//    therefore EXACT (V4 rounds LUT*scale*scale2 to BF16 per weight), so V7 is not
//    bit-identical to V4; it is closer to the FP64 product.
//  * m16 sub-tiles past the expert's last row skip their ldmatrix + MMAs (warp-uniform).
#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define OI640_STEP_K 32
#define OI640_TRANSPOSED 1
#define flash_next_orig_i640_prefill flash_next_f32_transposed_compact_k32_v7
#include "moe_w4a16_orig_i640_compact_prefill.cuh"
namespace oi640 = flash_next_f32_transposed_compact_k32_v7;

namespace {
constexpr unsigned V7_TILE_M = 128;
constexpr unsigned V7_TILE_N = 128;
constexpr unsigned V7_STAGES = 3;
constexpr unsigned V7_A_STRIDE = 40;   // bf16 per smem A row (32 + 8 pad) = 80 B, ldmatrix conflict-free
constexpr unsigned V7_B_STRIDE = 144;  // bytes per raw packed-B row (128 + 16 pad)
constexpr unsigned V7_S_STRIDE = 144;  // bytes per raw scale row
constexpr unsigned V7_MAX_ITEMS = (oi640::MAX_EXPANDED + V7_TILE_M - 1) / V7_TILE_M + oi640::EXPERTS;
static_assert(V7_MAX_ITEMS <= oi640::MAX_WORK_ITEMS, "V7 items must fit the shared workspace");

__device__ __forceinline__ void v7_cp_async_16(void* smem, const void* gmem, bool pred) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void v7_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void v7_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Two E2M1 values (low nibble -> low half) to f16x2, exact.
__device__ __forceinline__ unsigned v7_e2m1x2_to_f16x2(unsigned byte) {
    unsigned out;
    const unsigned short v = static_cast<unsigned short>(byte);
    asm("{\n .reg .b8 b, hi;\n mov.b16 {b, hi}, %1;\n cvt.rn.f16x2.e2m1x2 %0, b;\n}" : "=r"(out) : "h"(v));
    return out;
}
// One E4M3 scale byte broadcast to both halves of an f16x2, exact.
__device__ __forceinline__ unsigned v7_e4m3_to_f16x2_dup(unsigned byte) {
    unsigned out;
    const unsigned short pair = static_cast<unsigned short>(byte | (byte << 8));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(pair));
    return out;
}
// f16x2 (values with <= 6 significant bits) to bf16x2, exact.
__device__ __forceinline__ unsigned v7_f16x2_to_bf16x2(unsigned h) {
    const __half2 v = *reinterpret_cast<const __half2*>(&h);
    const __nv_bfloat162 b = __floats2bfloat162_rn(__low2float(v), __high2float(v));
    return *reinterpret_cast<const unsigned*>(&b);
}
__device__ __forceinline__ unsigned v7_hmul2(unsigned a, unsigned b) {
    const __half2 r = __hmul2(*reinterpret_cast<const __half2*>(&a), *reinterpret_cast<const __half2*>(&b));
    return *reinterpret_cast<const unsigned*>(&r);
}

// floor(log2(v)) of a positive E4M3 byte value; INT_MIN/2 for zero.
__device__ __forceinline__ int v7_e4m3_floor_log2(unsigned b) {
    const unsigned e = (b >> 3) & 0xfu, m = b & 7u;
    if (e != 0) return static_cast<int>(e) - 7;
    if (m == 0) return -(1 << 30);
    return (31 - __clz(static_cast<int>(m))) - 9;
}

// One bf16x2 B register from a packed E2M1 pair.
//  EMU = false: exact NVFP4 weight  bf16(e2m1 * s16)            (scale2 in the epilogue)
//  EMU = true : b12x w4a8 weight     bf16(e4m3_rn(e2m1 * r) * 2^E32), r = e4m3_rn(s16 / 2^E32),
//               E32 = max floor(log2 s16) over the 16-block pair of the k32 group
//               (b12x moe/_shared/kernels/reference.py: _dequant_w4a8_weight_nvfp4_residual).
template <bool EMU>
__device__ __forceinline__ void v7_bfrag(unsigned p0, unsigned p1, unsigned s_this, unsigned s_other,
                                         unsigned& b0, unsigned& b1) {
    if constexpr (!EMU) {
        const unsigned s2 = v7_e4m3_to_f16x2_dup(s_this);
        b0 = v7_f16x2_to_bf16x2(v7_hmul2(v7_e2m1x2_to_f16x2(p0), s2));
        b1 = v7_f16x2_to_bf16x2(v7_hmul2(v7_e2m1x2_to_f16x2(p1), s2));
    } else {
        int e32 = max(v7_e4m3_floor_log2(s_this), v7_e4m3_floor_log2(s_other));
        if (e32 < -(1 << 29)) e32 = 0;
        const unsigned short sraw = static_cast<unsigned short>(s_this | (s_this << 8));
        unsigned sh;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sh) : "h"(sraw));
        const float s = __low2float(*reinterpret_cast<const __half2*>(&sh));
        const float r = s * __int_as_float((127 - e32) << 23);          // exact power-of-two divide
        const __nv_fp8_storage_t rq = __nv_cvt_float_to_fp8(r, __NV_SATFINITE, __NV_E4M3);
        const unsigned rdup = v7_e4m3_to_f16x2_dup(static_cast<unsigned>(rq));
        const float pow2 = __int_as_float((127 + e32) << 23);
        const unsigned p[2] = {p0, p1};
        unsigned out[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned prod = v7_hmul2(v7_e2m1x2_to_f16x2(p[i]), rdup);
            unsigned short q;
            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(q) : "r"(prod));
            unsigned back;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(back) : "h"(q));
            const __half2 bh = *reinterpret_cast<const __half2*>(&back);
            const __nv_bfloat162 bb = __floats2bfloat162_rn(__low2float(bh) * pow2, __high2float(bh) * pow2);
            out[i] = *reinterpret_cast<const unsigned*>(&bb);
        }
        b0 = out[0];
        b1 = out[1];
    }
}
} // namespace

// V7 planner: the V2 planner with 128-row work items (expert-major, tile-minor).
extern "C" __global__ __launch_bounds__(512, 1)
void qwen4_moe_compact_t_plan_k32_v7(
    const int* expert_offsets,
    const int* sorted_token_ids,
    const oi640::Contract* contract,
    oi640::Workspace* workspace,
    int* status
) {
    __shared__ int s_err;
    __shared__ uint32_t s_tiles[oi640::EXPERTS];
    __shared__ uint32_t s_prefix[oi640::EXPERTS];
    const unsigned tid = threadIdx.x;
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
        s_tiles[tid] = (rows + V7_TILE_M - 1) / V7_TILE_M;
    }
    __syncthreads();
    if (tid == 0) {
        if (s_err == 0 && (expert_offsets[0] != 0 || expert_offsets[oi640::EXPERTS] != static_cast<int>(expanded)))
            s_err = oi640::ERR_OFFSETS;
        if (s_err == 0) {
            uint32_t count = 0, active = 0, max_rows = 0;
            uint64_t hash = 1469598103934665603ULL;
            for (uint32_t e = 0; e < oi640::EXPERTS; ++e) {
                const uint32_t rows = static_cast<uint32_t>(expert_offsets[e + 1] - expert_offsets[e]);
                hash = (hash ^ rows) * 1099511628211ULL;
                s_prefix[e] = count;
                if (rows == 0) continue;
                ++active;
                max_rows = max_rows > rows ? max_rows : rows;
                count += s_tiles[e];
            }
            if (count > V7_MAX_ITEMS) s_err = oi640::ERR_CAPACITY;
            else if (count == 0) s_err = oi640::ERR_PLAN;
            else {
                workspace->work_count = count;
                workspace->active_experts = active;
                workspace->max_rows = max_rows;
                workspace->census_hash = hash;
            }
        }
    }
    __syncthreads();
    if (s_err != 0) { if (tid == 0) status[0] = s_err; return; }
    for (uint32_t t = 0; t < s_tiles[tid]; ++t)
        workspace->items[s_prefix[tid] + t] = {tid, t};
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        workspace->ready_magic = oi640::MAGIC;
        __threadfence();
        status[0] = oi640::PLANNED;
    }
}

// EP-filtered V7 planner: identical 128-row worklist ABI as `qwen4_moe_compact_t_plan_k32_v7`,
// emitting work items ONLY for this rank's local `[local_start, local_end)` experts.
// Permits `work_count == 0` (when no routed pair hits a local expert) so downstream W4A8
// kernels execute 0 tasks cleanly and `all_reduce` sums the partial rank outputs.
extern "C" __global__ __launch_bounds__(512, 1)
void qwen4_moe_compact_t_plan_k32_ep(
    const int* expert_offsets,
    const int* sorted_token_ids,
    const oi640::Contract* contract,
    oi640::Workspace* workspace,
    int* status,
    unsigned int local_start,
    unsigned int local_end
) {
    __shared__ int s_err;
    __shared__ uint32_t s_tiles[oi640::EXPERTS];
    __shared__ uint32_t s_prefix[oi640::EXPERTS];
    const unsigned tid = threadIdx.x;
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
        const bool is_local = (tid >= local_start && tid < local_end);
        const uint32_t rows = (is_local && end >= begin) ? static_cast<uint32_t>(end - begin) : 0u;
        s_tiles[tid] = (rows + V7_TILE_M - 1) / V7_TILE_M;
    }
    __syncthreads();
    if (tid == 0) {
        if (s_err == 0 && (expert_offsets[0] != 0 || expert_offsets[oi640::EXPERTS] != static_cast<int>(expanded)))
            s_err = oi640::ERR_OFFSETS;
        if (s_err == 0) {
            uint32_t count = 0, active = 0, max_rows = 0;
            uint64_t hash = 1469598103934665603ULL;
            for (uint32_t e = 0; e < oi640::EXPERTS; ++e) {
                const bool is_local = (e >= local_start && e < local_end);
                const uint32_t rows = is_local ? static_cast<uint32_t>(expert_offsets[e + 1] - expert_offsets[e]) : 0u;
                hash = (hash ^ rows) * 1099511628211ULL;
                s_prefix[e] = count;
                if (rows == 0) continue;
                ++active;
                max_rows = max_rows > rows ? max_rows : rows;
                count += s_tiles[e];
            }
            if (count > V7_MAX_ITEMS) s_err = oi640::ERR_CAPACITY;
            else {
                workspace->work_count = count;
                workspace->active_experts = active;
                workspace->max_rows = max_rows;
                workspace->census_hash = hash;
            }
        }
    }
    __syncthreads();
    if (s_err != 0) { if (tid == 0) status[0] = s_err; return; }
    for (uint32_t t = 0; t < s_tiles[tid]; ++t)
        workspace->items[s_prefix[tid] + t] = {tid, t};
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        workspace->ready_magic = oi640::MAGIC;
        __threadfence();
        status[0] = oi640::PLANNED;
    }
}

// C[pair, n] = bf16(scale2[e] * sum_k A[row(pair), k] * (E2M1[k][n] * E4M3[k/16][n]))
template <bool EMU>
__device__ __forceinline__ void v7_gemm_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K
) {
    if (gridDim.x != oi640::FIXED_GRID || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }
    const bool shape = (N == oi640::INTER && K == oi640::HIDDEN) ||
        (N == oi640::HIDDEN && K == oi640::INTER);
    if (!shape || A == nullptr || packed_ptrs == nullptr || scale_ptrs == nullptr ||
        scale2_vals == nullptr || C == nullptr || expert_offsets == nullptr ||
        sorted_token_ids == nullptr || !oi640::contract_ok(contract) ||
        !oi640::workspace_ok(workspace, contract) || status == nullptr ||
        status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_ABI);
        return;
    }
    __shared__ __align__(16) __nv_bfloat16 sA[V7_STAGES][V7_TILE_M][V7_A_STRIDE];
    __shared__ __align__(16) unsigned char sB[V7_STAGES][16][V7_B_STRIDE];
    __shared__ __align__(16) unsigned char sS[V7_STAGES][2][V7_S_STRIDE];

    const unsigned n_tiles = (N + V7_TILE_N - 1) / V7_TILE_N;
    const unsigned task_count = workspace->work_count * n_tiles;
    const unsigned tid = threadIdx.x;
    const unsigned warp_id = tid >> 5;
    const unsigned lane = tid & 31;
    const unsigned warp_m = (warp_id >> 2) * 64;   // 2 warps down
    const unsigned warp_n = (warp_id & 3) * 32;    // 4 warps across
    const unsigned gid = lane >> 2;
    const unsigned tig = lane & 3;
    const unsigned KT = K / oi640::STEP_K;
    const bool gather = (K == oi640::HIDDEN);

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const oi640::WorkItem item = workspace->items[task / n_tiles];
        const unsigned cta_n = (task % n_tiles) * V7_TILE_N;
        if (item.expert >= oi640::EXPERTS) {
            if (tid == 0) oi640::fail(status, oi640::ERR_PLAN);
            return;
        }
        const int m_start = expert_offsets[item.expert];
        const int m_end = expert_offsets[item.expert + 1];
        const int local = static_cast<int>(item.m_tile * V7_TILE_M);
        const int M_expert = m_end - m_start;
        const unsigned char* B = reinterpret_cast<const unsigned char*>(packed_ptrs[item.expert]);
        const unsigned char* S = reinterpret_cast<const unsigned char*>(scale_ptrs[item.expert]);
        const float scale2 = scale2_vals[item.expert];
        if (m_start < 0 || m_end < m_start || local < 0 || local >= M_expert ||
            B == nullptr || S == nullptr || !isfinite(scale2)) {
            if (tid == 0) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const unsigned cta_m = static_cast<unsigned>(m_start + local);
        const int valid_rows = min(M_expert - local, static_cast<int>(V7_TILE_M));
        // m16 sub-tiles of this warp that hold at least one routed row (warp-uniform).
        const int warp_valid = valid_rows - static_cast<int>(warp_m);
        const int rt_active = warp_valid <= 0 ? 0 : min(4, (warp_valid + 15) >> 4);

        // A: 128 rows x 32 k (64 B) per stage = 512 x 16 B chunks, two per thread.
        const __nv_bfloat16* a_src[2];
        bool a_ok[2];
        unsigned a_row[2], a_cc[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned c = tid + i * 256;
            a_row[i] = c >> 2;
            a_cc[i] = c & 3;
            const bool ok = static_cast<int>(a_row[i]) < valid_rows;
            const unsigned ar = ok
                ? (gather ? static_cast<unsigned>(sorted_token_ids[cta_m + a_row[i]]) : cta_m + a_row[i])
                : 0u;
            a_ok[i] = ok;
            a_src[i] = A + static_cast<unsigned long long>(ar) * K + a_cc[i] * 8;
        }

        auto load_stage = [&](unsigned stage, unsigned kt) {
            const unsigned k_base = kt * oi640::STEP_K;
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                v7_cp_async_16(&sA[stage][a_row[i]][a_cc[i] * 8], a_src[i] + k_base, a_ok[i]);
            if (tid < 128) {
                const unsigned r = tid >> 3, cc = tid & 7;
                v7_cp_async_16(&sB[stage][r][cc * 16],
                               B + static_cast<unsigned long long>(k_base / 2 + r) * N + cta_n + cc * 16, true);
            } else if (tid < 144) {
                const unsigned t = tid - 128;
                const unsigned r = t >> 3, cc = t & 7;
                v7_cp_async_16(&sS[stage][r][cc * 16],
                               S + static_cast<unsigned long long>(k_base / 16 + r) * N + cta_n + cc * 16, true);
            }
        };

        float acc[4][4][4];
        #pragma unroll
        for (int r = 0; r < 4; ++r)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.0f;

        #pragma unroll
        for (unsigned s = 0; s < V7_STAGES - 1; ++s) {
            if (s < KT) load_stage(s, s);
            v7_cp_async_commit();
        }

        for (unsigned kt = 0; kt < KT; ++kt) {
            v7_cp_async_wait<V7_STAGES - 2>();
            __syncthreads();
            {
                const unsigned nk = kt + V7_STAGES - 1;
                if (nk < KT) load_stage(nk % V7_STAGES, nk);
                v7_cp_async_commit();
            }
            if (rt_active == 0) continue;
            const unsigned st = kt % V7_STAGES;
            #pragma unroll
            for (unsigned ks = 0; ks < 2; ++ks) {            // two k16 halves of the 32-deep stage
                // B fragments for this warp's 4 n8 tiles: k-pair rows 8*ks + tig and +4.
                unsigned bfr[4][2];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned n = warp_n + j * 8 + gid;
                    const unsigned p0 = sB[st][ks * 8 + tig][n];
                    const unsigned p1 = sB[st][ks * 8 + 4 + tig][n];
                    v7_bfrag<EMU>(p0, p1, sS[st][ks][n], sS[st][ks ^ 1][n], bfr[j][0], bfr[j][1]);
                }
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    if (r >= rt_active) break;
                    unsigned a0, a1, a2, a3;
                    const unsigned row = warp_m + r * 16 + (lane & 15);
                    const unsigned col = ks * 16 + ((lane >> 4) << 3);
                    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(&sA[st][row][col]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                            : "=f"(acc[r][j][0]), "=f"(acc[r][j][1]), "=f"(acc[r][j][2]), "=f"(acc[r][j][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bfr[j][0]), "r"(bfr[j][1]),
                              "f"(acc[r][j][0]), "f"(acc[r][j][1]), "f"(acc[r][j][2]), "f"(acc[r][j][3]));
                    }
                }
            }
        }
        v7_cp_async_wait<0>();

        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (r >= rt_active) break;
            const unsigned lr0 = warp_m + r * 16 + gid, lr1 = lr0 + 8;
            const bool r0v = static_cast<int>(lr0) < valid_rows;
            const bool r1v = static_cast<int>(lr1) < valid_rows;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned c0 = cta_n + warp_n + j * 8 + tig * 2;
                if (c0 >= N) continue;
                if (r0v) {
                    const __nv_bfloat162 v = __floats2bfloat162_rn(acc[r][j][0] * scale2, acc[r][j][1] * scale2);
                    *reinterpret_cast<__nv_bfloat162*>(C + static_cast<unsigned long long>(cta_m + lr0) * N + c0) = v;
                }
                if (r1v) {
                    const __nv_bfloat162 v = __floats2bfloat162_rn(acc[r][j][2] * scale2, acc[r][j][3] * scale2);
                    *reinterpret_cast<__nv_bfloat162*>(C + static_cast<unsigned long long>(cta_m + lr1) * N + c0) = v;
                }
            }
        }
        __syncthreads();
    }
}

// V7 fused gate/up + SwiGLU (wide chunks). One CTA computes a 128-row x 64-column tile of
// BOTH projections from one shared A tile (the input is read once instead of twice) and
// writes only the activated result act = silu(g) * u, with g = scale2_gate * acc_gate and
// u = scale2_up * acc_up taken from the FP32 accumulators (the separate path rounds g and
// u to BF16, stores both and re-reads them in moe_silu_mul). Same planner/workspace as the
// V7 GEMM; warp tile 64 rows x (16 gate + 16 up) columns, so a thread holds g and u of the
// same (row, column) and the SwiGLU needs no data exchange.
template <bool EMU>
__device__ __forceinline__ void v7_gateup_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K
) {
    constexpr unsigned TN = 64;            // output (activated) columns per CTA
    constexpr unsigned GB_STRIDE = 80;     // 64 + 16 pad bytes
    if (gridDim.x != oi640::FIXED_GRID || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }
    if (N != oi640::INTER || K != oi640::HIDDEN || A == nullptr || C == nullptr ||
        gate_packed_ptrs == nullptr || gate_scale_ptrs == nullptr || gate_scale2_vals == nullptr ||
        up_packed_ptrs == nullptr || up_scale_ptrs == nullptr || up_scale2_vals == nullptr ||
        expert_offsets == nullptr || sorted_token_ids == nullptr || !oi640::contract_ok(contract) ||
        !oi640::workspace_ok(workspace, contract) || status == nullptr || status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_ABI);
        return;
    }
    __shared__ __align__(16) __nv_bfloat16 sA[V7_STAGES][V7_TILE_M][V7_A_STRIDE];
    __shared__ __align__(16) unsigned char sB[V7_STAGES][2][16][GB_STRIDE];   // [gate|up]
    __shared__ __align__(16) unsigned char sS[V7_STAGES][2][2][GB_STRIDE];

    const unsigned n_tiles = N / TN;
    const unsigned task_count = workspace->work_count * n_tiles;
    const unsigned tid = threadIdx.x;
    const unsigned warp_id = tid >> 5;
    const unsigned lane = tid & 31;
    const unsigned warp_m = (warp_id >> 2) * 64;
    const unsigned warp_n = (warp_id & 3) * 16;
    const unsigned gid = lane >> 2;
    const unsigned tig = lane & 3;
    const unsigned KT = K / oi640::STEP_K;

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const oi640::WorkItem item = workspace->items[task / n_tiles];
        const unsigned cta_n = (task % n_tiles) * TN;
        if (item.expert >= oi640::EXPERTS) {
            if (tid == 0) oi640::fail(status, oi640::ERR_PLAN);
            return;
        }
        const int m_start = expert_offsets[item.expert];
        const int m_end = expert_offsets[item.expert + 1];
        const int local = static_cast<int>(item.m_tile * V7_TILE_M);
        const int M_expert = m_end - m_start;
        const unsigned char* Bg = reinterpret_cast<const unsigned char*>(gate_packed_ptrs[item.expert]);
        const unsigned char* Sg = reinterpret_cast<const unsigned char*>(gate_scale_ptrs[item.expert]);
        const unsigned char* Bu = reinterpret_cast<const unsigned char*>(up_packed_ptrs[item.expert]);
        const unsigned char* Su = reinterpret_cast<const unsigned char*>(up_scale_ptrs[item.expert]);
        const float s2g = gate_scale2_vals[item.expert];
        const float s2u = up_scale2_vals[item.expert];
        if (m_start < 0 || m_end < m_start || local < 0 || local >= M_expert || Bg == nullptr ||
            Sg == nullptr || Bu == nullptr || Su == nullptr || !isfinite(s2g) || !isfinite(s2u)) {
            if (tid == 0) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const unsigned cta_m = static_cast<unsigned>(m_start + local);
        const int valid_rows = min(M_expert - local, static_cast<int>(V7_TILE_M));
        const int warp_valid = valid_rows - static_cast<int>(warp_m);
        const int rt_active = warp_valid <= 0 ? 0 : min(4, (warp_valid + 15) >> 4);

        const __nv_bfloat16* a_src[2];
        bool a_ok[2];
        unsigned a_row[2], a_cc[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned c = tid + i * 256;
            a_row[i] = c >> 2;
            a_cc[i] = c & 3;
            const bool ok = static_cast<int>(a_row[i]) < valid_rows;
            const unsigned ar = ok ? static_cast<unsigned>(sorted_token_ids[cta_m + a_row[i]]) : 0u;
            a_ok[i] = ok;
            a_src[i] = A + static_cast<unsigned long long>(ar) * K + a_cc[i] * 8;
        }

        auto load_stage = [&](unsigned stage, unsigned kt) {
            const unsigned k_base = kt * oi640::STEP_K;
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                v7_cp_async_16(&sA[stage][a_row[i]][a_cc[i] * 8], a_src[i] + k_base, a_ok[i]);
            if (tid < 128) {                         // 2 matrices x 16 rows x 4 chunks
                const unsigned mtx = tid >> 6, r = (tid >> 2) & 15, cc = tid & 3;
                const unsigned char* src = mtx ? Bu : Bg;
                v7_cp_async_16(&sB[stage][mtx][r][cc * 16],
                               src + static_cast<unsigned long long>(k_base / 2 + r) * N + cta_n + cc * 16, true);
            } else if (tid < 144) {                  // 2 matrices x 2 rows x 4 chunks
                const unsigned t = tid - 128;
                const unsigned mtx = t >> 3, r = (t >> 2) & 1, cc = t & 3;
                const unsigned char* src = mtx ? Su : Sg;
                v7_cp_async_16(&sS[stage][mtx][r][cc * 16],
                               src + static_cast<unsigned long long>(k_base / 16 + r) * N + cta_n + cc * 16, true);
            }
        };

        float acc[4][4][4];   // [m16][j: gate n8 0,1 | up n8 0,1][4]
        #pragma unroll
        for (int r = 0; r < 4; ++r)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.0f;

        #pragma unroll
        for (unsigned s = 0; s < V7_STAGES - 1; ++s) {
            if (s < KT) load_stage(s, s);
            v7_cp_async_commit();
        }
        for (unsigned kt = 0; kt < KT; ++kt) {
            v7_cp_async_wait<V7_STAGES - 2>();
            __syncthreads();
            {
                const unsigned nk = kt + V7_STAGES - 1;
                if (nk < KT) load_stage(nk % V7_STAGES, nk);
                v7_cp_async_commit();
            }
            if (rt_active == 0) continue;
            const unsigned st = kt % V7_STAGES;
            #pragma unroll
            for (unsigned ks = 0; ks < 2; ++ks) {
                unsigned bfr[4][2];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned mtx = j >> 1;
                    const unsigned n = warp_n + (j & 1) * 8 + gid;
                    const unsigned p0 = sB[st][mtx][ks * 8 + tig][n];
                    const unsigned p1 = sB[st][mtx][ks * 8 + 4 + tig][n];
                    v7_bfrag<EMU>(p0, p1, sS[st][mtx][ks][n], sS[st][mtx][ks ^ 1][n], bfr[j][0], bfr[j][1]);
                }
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    if (r >= rt_active) break;
                    unsigned a0, a1, a2, a3;
                    const unsigned row = warp_m + r * 16 + (lane & 15);
                    const unsigned col = ks * 16 + ((lane >> 4) << 3);
                    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(&sA[st][row][col]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                            : "=f"(acc[r][j][0]), "=f"(acc[r][j][1]), "=f"(acc[r][j][2]), "=f"(acc[r][j][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bfr[j][0]), "r"(bfr[j][1]),
                              "f"(acc[r][j][0]), "f"(acc[r][j][1]), "f"(acc[r][j][2]), "f"(acc[r][j][3]));
                    }
                }
            }
        }
        v7_cp_async_wait<0>();

        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (r >= rt_active) break;
            const unsigned lr0 = warp_m + r * 16 + gid, lr1 = lr0 + 8;
            const bool r0v = static_cast<int>(lr0) < valid_rows;
            const bool r1v = static_cast<int>(lr1) < valid_rows;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const unsigned c0 = cta_n + warp_n + j * 8 + tig * 2;
                float v[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const float g = acc[r][j][q] * s2g;
                    const float u = acc[r][j + 2][q] * s2u;
                    v[q] = g * (1.0f / (1.0f + __expf(-g))) * u;
                }
                if (r0v)
                    *reinterpret_cast<__nv_bfloat162*>(C + static_cast<unsigned long long>(cta_m + lr0) * N + c0) =
                        __floats2bfloat162_rn(v[0], v[1]);
                if (r1v)
                    *reinterpret_cast<__nv_bfloat162*>(C + static_cast<unsigned long long>(cta_m + lr1) * N + c0) =
                        __floats2bfloat162_rn(v[2], v[3]);
            }
        }
        __syncthreads();
    }
}

#define V7_GEMM_ARGS \
    const __nv_bfloat16* __restrict__ A, const unsigned long long* __restrict__ packed_ptrs, \
    const unsigned long long* __restrict__ scale_ptrs, const float* __restrict__ scale2_vals, \
    __nv_bfloat16* __restrict__ C, const int* __restrict__ expert_offsets, \
    const int* __restrict__ sorted_token_ids, const oi640::Contract* __restrict__ contract, \
    const oi640::Workspace* __restrict__ workspace, int* status, unsigned int N, unsigned int K
#define V7_GEMM_PASS A, packed_ptrs, scale_ptrs, scale2_vals, C, expert_offsets, sorted_token_ids, \
    contract, workspace, status, N, K
#define V7_GATEUP_ARGS \
    const __nv_bfloat16* __restrict__ A, const unsigned long long* __restrict__ gate_packed_ptrs, \
    const unsigned long long* __restrict__ gate_scale_ptrs, const float* __restrict__ gate_scale2_vals, \
    __nv_bfloat16* __restrict__ C, const unsigned long long* __restrict__ up_packed_ptrs, \
    const unsigned long long* __restrict__ up_scale_ptrs, const float* __restrict__ up_scale2_vals, \
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids, \
    const oi640::Contract* __restrict__ contract, const oi640::Workspace* __restrict__ workspace, \
    int* status, unsigned int N, unsigned int K
#define V7_GATEUP_PASS A, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, C, up_packed_ptrs, \
    up_scale_ptrs, up_scale2_vals, expert_offsets, sorted_token_ids, contract, workspace, status, N, K

extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_k32_v7(V7_GEMM_ARGS) {
    v7_gemm_impl<false>(V7_GEMM_PASS);
}
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_gateup_k32_v7(V7_GATEUP_ARGS) {
    v7_gateup_impl<false>(V7_GATEUP_PASS);
}
// Quality-gate emulation of b12x's w4a8 weight rounding (ATLAS_QWEN4_PREFILL_FAKEQUANT
// `b12xw`): same ABI, NOT a speed path.
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_k32_v7_b12xw(V7_GEMM_ARGS) {
    v7_gemm_impl<true>(V7_GEMM_PASS);
}
extern "C" __global__ __launch_bounds__(256, 2) void qwen4_moe_compact_t_gemm_gateup_k32_v7_b12xw(V7_GATEUP_ARGS) {
    v7_gateup_impl<true>(V7_GATEUP_PASS);
}
