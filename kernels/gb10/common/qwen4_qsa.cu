// SPDX-License-Identifier: AGPL-3.0-only
//
// Correctness-first Qwen4 QSA kernels.  The cache is keyed by Atlas's main
// paged-KV physical blocks: every 16-token KV page owns four compressed index
// keys and a four-row raw-key ring.  This keeps prefix sharing and block reuse
// aligned without a second host allocator.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "qwen4_qsa_positions.cuh"

namespace {

constexpr int INDEX_HEADS = 4;
constexpr int INDEX_DIM = 128;
constexpr int COMPRESS_RATIO = 4;
constexpr int TOKEN_TOPK = 2048;
constexpr int BLOCK_TOPK = TOKEN_TOPK / COMPRESS_RATIO;
constexpr int OUTPUT_WIDTH = TOKEN_TOPK + COMPRESS_RATIO - 1;
constexpr int NVFP4_GROUP_SIZE = 16;
constexpr int VERIFY_K5_ROWS = 5;
constexpr unsigned int VERIFY_K5_COMPLETE_MASK = (1u << VERIFY_K5_ROWS) - 1u;
// Widest journaled verify frame (QWEN4_HYBRID_MAX_ROWS on the host). The undo
// buffers hold this many rows; K5 keeps its own commit kernel below.
constexpr int VERIFY_MAX_ROWS = 9;
constexpr unsigned long long NO_COMPRESSED_OFFSET = ~0ull;

__device__ __forceinline__ float fp8e4m3_to_f32(__nv_fp8_storage_t value) {
    return __half2float(__nv_cvt_fp8_to_halfraw(value, __NV_E4M3));
}

__device__ __forceinline__ float decode_e2m1(unsigned int nibble) {
    const unsigned int mag = nibble & 7u;
    const float abs_val = (mag <= 4u)
        ? 0.5f * static_cast<float>(mag)
        : ((mag == 5u) ? 3.0f : ((mag == 6u) ? 4.0f : 6.0f));
    return (nibble & 8u) ? -abs_val : abs_val;
}

__device__ __forceinline__ float load_nvfp4(
    const unsigned char* __restrict__ block,
    unsigned long long element,
    unsigned long long data_section_bytes) {
    const unsigned char packed = __ldg(block + (element >> 1));
    const unsigned int nibble = (element & 1ull) ? (packed >> 4) : (packed & 0xfu);
    const auto scale = (__nv_fp8_storage_t)__ldg(
        block + data_section_bytes + element / NVFP4_GROUP_SIZE);
    return decode_e2m1(nibble) * fp8e4m3_to_f32(scale);
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ bool less_pair(float av, int ai, float bv, int bi) {
    return av < bv || (av == bv && ai > bi);
}

__device__ void heap_sift_down(float* values, int* indices, int root, int size) {
    while (true) {
        int child = root * 2 + 1;
        if (child >= size) return;
        if (child + 1 < size &&
            less_pair(values[child + 1], indices[child + 1], values[child], indices[child])) {
            ++child;
        }
        if (!less_pair(values[child], indices[child], values[root], indices[root])) return;
        float tv = values[root]; values[root] = values[child]; values[child] = tv;
        int ti = indices[root]; indices[root] = indices[child]; indices[child] = ti;
        root = child;
    }
}

}  // namespace

// QSA's raw-key cache is a four-row physical-page ring.  A five-row
// speculative frame therefore aliases its fifth write with an earlier row,
// and a rejected suffix cannot be discarded by merely shortening seq_len.
// Journal the exact physical destinations immediately before each K5 write.
// Row zero clears the validity mask inside the captured graph, so graph replay
// cannot inherit a completed journal from the previous frame.
extern "C" __global__ void qwen4_qsa_snapshot_verify_k5_row(
    const __nv_bfloat16* __restrict__ raw_ring,
    const __nv_bfloat16* __restrict__ compressed_cache,
    const long long* __restrict__ slot_ptr,
    __nv_bfloat16* __restrict__ undo_raw,
    __nv_bfloat16* __restrict__ undo_compressed,
    unsigned long long* __restrict__ undo_raw_offsets,
    unsigned long long* __restrict__ undo_compressed_offsets,
    unsigned int* __restrict__ valid_mask,
    unsigned int row,
    unsigned int block_size) {
    const unsigned int d = threadIdx.x;
    if (row >= VERIFY_MAX_ROWS || blockDim.x != INDEX_DIM) return;

    if (row == 0 && d == 0) *valid_mask = 0;
    __syncthreads();

    qwen4_qsa_positions::PhysicalSlot physical;
    if (!qwen4_qsa_positions::physical_slot(*slot_ptr, block_size, physical)) return;
    const unsigned long long raw_offset =
        (physical.block * COMPRESS_RATIO + physical.ring_row) * INDEX_DIM;
    const unsigned long long compressed_offset =
        physical.ring_row == COMPRESS_RATIO - 1
            ? (physical.block * (block_size / COMPRESS_RATIO) + physical.group) * INDEX_DIM
            : NO_COMPRESSED_OFFSET;

    if (d == 0) {
        undo_raw_offsets[row] = raw_offset;
        undo_compressed_offsets[row] = compressed_offset;
    }
    undo_raw[(unsigned long long)row * INDEX_DIM + d] = raw_ring[raw_offset + d];
    if (compressed_offset != NO_COMPRESSED_OFFSET) {
        undo_compressed[(unsigned long long)row * INDEX_DIM + d] =
            compressed_cache[compressed_offset + d];
    }
    __syncthreads();
    if (d == 0) atomicOr(valid_mask, 1u << row);
}

// Restore rejected rows in reverse order.  Reverse traversal is essential:
// row four can alias row zero in the four-row raw ring, so forward restoration
// would overwrite the accepted row-zero image with the pre-frame value.
extern "C" __global__ void qwen4_qsa_commit_verify_k5_prefix(
    __nv_bfloat16* __restrict__ raw_ring,
    __nv_bfloat16* __restrict__ compressed_cache,
    const __nv_bfloat16* __restrict__ undo_raw,
    const __nv_bfloat16* __restrict__ undo_compressed,
    const unsigned long long* __restrict__ undo_raw_offsets,
    const unsigned long long* __restrict__ undo_compressed_offsets,
    unsigned int* __restrict__ valid_mask,
    unsigned int accepted) {
    const unsigned int d = threadIdx.x;
    if (blockDim.x != INDEX_DIM) return;

    const unsigned int valid = *valid_mask;
    // Zero means this layer took a non-transactional K5 route and is a no-op.
    // A zero-accept abort is also the recovery path for a forward that failed
    // between rows: restore every row that made it into the journal.  A real
    // accepted prefix, however, is valid only after all five rows were saved.
    // Host code separately rejects accepted>K.
    if (valid == 0) return;
    if (accepted > VERIFY_K5_ROWS ||
        (accepted != 0 && valid != VERIFY_K5_COMPLETE_MASK)) {
        asm volatile("trap;");
        return;
    }

    for (int row = VERIFY_K5_ROWS - 1; row >= static_cast<int>(accepted); --row) {
        if ((valid & (1u << row)) == 0) continue;
        const unsigned long long raw_offset = undo_raw_offsets[row];
        raw_ring[raw_offset + d] = undo_raw[(unsigned long long)row * INDEX_DIM + d];
        const unsigned long long compressed_offset = undo_compressed_offsets[row];
        if (compressed_offset != NO_COMPRESSED_OFFSET) {
            compressed_cache[compressed_offset + d] =
                undo_compressed[(unsigned long long)row * INDEX_DIM + d];
        }
        __syncthreads();
    }
    if (d == 0) *valid_mask = 0;
}

// Same commit for every other exact frame (2..=4 and 6..=9 rows: the
// confidence-gated widths and wider drafts). Their rejected suffix can also
// cross a compression group in the four-row ring: rows of group g+1 overwrite
// ring rows that g's ring-3 write pools later, so the journal is just as
// necessary. Past four rows a frame writes some ring rows twice (rows r and
// r+4 alias); the reverse restore below still ends at the pre-frame image.
// rows == 5 keeps the K5 kernel above.
extern "C" __global__ void qwen4_qsa_commit_verify_prefix_rows(
    __nv_bfloat16* __restrict__ raw_ring,
    __nv_bfloat16* __restrict__ compressed_cache,
    const __nv_bfloat16* __restrict__ undo_raw,
    const __nv_bfloat16* __restrict__ undo_compressed,
    const unsigned long long* __restrict__ undo_raw_offsets,
    const unsigned long long* __restrict__ undo_compressed_offsets,
    unsigned int* __restrict__ valid_mask,
    unsigned int accepted,
    unsigned int rows) {
    const unsigned int d = threadIdx.x;
    if (blockDim.x != INDEX_DIM) return;

    const unsigned int valid = *valid_mask;
    if (valid == 0) return;
    if (rows < 2 || rows > VERIFY_MAX_ROWS || accepted > rows ||
        (accepted != 0 && valid != (1u << rows) - 1u)) {
        asm volatile("trap;");
        return;
    }

    for (int row = static_cast<int>(rows) - 1; row >= static_cast<int>(accepted); --row) {
        if ((valid & (1u << row)) == 0) continue;
        const unsigned long long raw_offset = undo_raw_offsets[row];
        raw_ring[raw_offset + d] = undo_raw[(unsigned long long)row * INDEX_DIM + d];
        const unsigned long long compressed_offset = undo_compressed_offsets[row];
        if (compressed_offset != NO_COMPRESSED_OFFSET) {
            compressed_cache[compressed_offset + d] =
                undo_compressed[(unsigned long long)row * INDEX_DIM + d];
        }
        __syncthreads();
    }
    if (d == 0) *valid_mask = 0;
}

// projected_qk is [4*128 query | 1*128 raw key].  Store the raw key in the
// physical page's four-row ring and, at each group boundary, average the four
// rows into pooled_key.  first_position is consumed by the ordinary RoPE
// kernel; its current reconstruction is valid only for linear positions.
extern "C" __global__ void qwen4_qsa_stage_pool(
    const __nv_bfloat16* __restrict__ projected_qk,
    __nv_bfloat16* __restrict__ raw_ring,
    __nv_bfloat16* __restrict__ pooled_key,
    unsigned int* __restrict__ first_position,
    const long long* __restrict__ slot_ptr,
    const unsigned int* __restrict__ position_ptr,
    unsigned int block_size) {
    const int d = threadIdx.x;
    if (blockDim.x != INDEX_DIM) return;
    const long long slot = *slot_ptr;
    const unsigned int position = *position_ptr;
    qwen4_qsa_positions::PhysicalSlot physical;
    if (!qwen4_qsa_positions::physical_slot(slot, block_size, physical)) return;
    const unsigned long long physical_block = physical.block;
    const unsigned int ring_row = physical.ring_row;
    const unsigned long long ring_base =
        (physical_block * COMPRESS_RATIO + ring_row) * INDEX_DIM;
    raw_ring[ring_base + d] = projected_qk[INDEX_HEADS * INDEX_DIM + d];
    __syncthreads();
    if (ring_row == COMPRESS_RATIO - 1) {
        float sum = 0.0f;
#pragma unroll
        for (int r = 0; r < COMPRESS_RATIO; ++r) {
            sum += __bfloat162float(raw_ring[(physical_block * COMPRESS_RATIO + r) * INDEX_DIM + d]);
        }
        pooled_key[d] = __float2bfloat16_rn(sum * 0.25f);
    }
    // This legacy 1D reconstruction does not implement first-member T/H/W.
    // Physical ownership alone does not qualify image QSA rotary semantics.
    if (d == 0) *first_position = position - ring_row;
}

extern "C" __global__ void qwen4_qsa_store_compressed(
    const __nv_bfloat16* __restrict__ pooled_key,
    __nv_bfloat16* __restrict__ compressed_cache,
    const long long* __restrict__ slot_ptr,
    const unsigned int* __restrict__ position_ptr,
    unsigned int block_size) {
    const int d = threadIdx.x;
    if (d >= INDEX_DIM) return;  // No CTA barrier below; per-thread guard is safe.
    const long long slot = *slot_ptr;
    (void)position_ptr;  // Retained launch ABI; rotary position does not gate stores.
    qwen4_qsa_positions::PhysicalSlot physical;
    if (!qwen4_qsa_positions::physical_slot(slot, block_size, physical) ||
        physical.ring_row != COMPRESS_RATIO - 1) return;
    const unsigned long long physical_block = physical.block;
    const unsigned int group_in_block = physical.group;
    const unsigned long long dst =
        (physical_block * (block_size / COMPRESS_RATIO) + group_in_block) * INDEX_DIM;
    compressed_cache[dst + d] = pooled_key[d];
}

// Batched prefill index construction. Prefill does not consume query scores;
// only the fifth (key) projection head is needed to populate QSA history.
// Split staging/pooling into separate launches so every raw row is globally
// visible before a four-token group is reduced.
extern "C" __global__ void qwen4_qsa_stage_prefill_raw(
    const __nv_bfloat16* __restrict__ raw_keys,
    __nv_bfloat16* __restrict__ raw_ring,
    const long long* __restrict__ slots,
    const unsigned int* __restrict__ positions,
    unsigned int num_tokens,
    unsigned int block_size) {
    const unsigned int token = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (token >= num_tokens || d >= INDEX_DIM || block_size == 0) return;
    const long long slot = slots[token];
    (void)positions;  // Retained launch ABI; ring ownership uses physical slots.
    qwen4_qsa_positions::PhysicalSlot physical;
    if (!qwen4_qsa_positions::physical_slot(slot, block_size, physical)) return;
    const unsigned long long physical_block = physical.block;
    const unsigned int ring_row = physical.ring_row;
    raw_ring[(physical_block * COMPRESS_RATIO + ring_row) * INDEX_DIM + d] =
        raw_keys[(unsigned long long)token * INDEX_DIM + d];
}

extern "C" __global__ void qwen4_qsa_pool_prefill(
    const __nv_bfloat16* __restrict__ raw_ring,
    __nv_bfloat16* __restrict__ pooled_keys,
    unsigned int* __restrict__ first_positions,
    const long long* __restrict__ slots,
    const unsigned int* __restrict__ positions,
    unsigned int first_endpoint,
    unsigned int num_groups,
    unsigned int block_size) {
    const unsigned int group = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (group >= num_groups || d >= INDEX_DIM || block_size == 0) return;
    const unsigned int token = first_endpoint + group * COMPRESS_RATIO;
    const long long slot = slots[token];
    if (slot < 0) return;
    const unsigned long long physical_block = (unsigned long long)slot / block_size;
    float sum = 0.0f;
#pragma unroll
    for (int row = 0; row < COMPRESS_RATIO; ++row) {
        sum += __bfloat162float(
            raw_ring[(physical_block * COMPRESS_RATIO + row) * INDEX_DIM + d]);
    }
    pooled_keys[(unsigned long long)group * INDEX_DIM + d] =
        __float2bfloat16_rn(sum * 0.25f);
    if (d == 0) first_positions[group] = positions[token] - (COMPRESS_RATIO - 1);
}

extern "C" __global__ void qwen4_qsa_store_prefill_compressed(
    const __nv_bfloat16* __restrict__ pooled_keys,
    __nv_bfloat16* __restrict__ compressed_cache,
    const long long* __restrict__ slots,
    unsigned int first_endpoint,
    unsigned int num_groups,
    unsigned int block_size) {
    const unsigned int group = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (group >= num_groups || d >= INDEX_DIM || block_size < COMPRESS_RATIO) return;
    const unsigned int token = first_endpoint + group * COMPRESS_RATIO;
    const long long slot = slots[token];
    if (slot < 0) return;
    const unsigned long long physical_block = (unsigned long long)slot / block_size;
    const unsigned int group_in_block = ((unsigned int)slot % block_size) / COMPRESS_RATIO;
    const unsigned long long dst =
        (physical_block * (block_size / COMPRESS_RATIO) + group_in_block) * INDEX_DIM;
    compressed_cache[dst + d] = pooled_keys[(unsigned long long)group * INDEX_DIM + d];
}

// Eight warps score eight compressed micro-blocks per CTA.  The trained score
// is sum_h relu(q_h dot k) / sqrt(128), exactly matching Qwen's reference.
extern "C" __global__ void qwen4_qsa_score(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ compressed_cache,
    const int* __restrict__ block_table,
    float* __restrict__ logits,
    unsigned int visible_groups,
    unsigned int block_size) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const unsigned int group = blockIdx.x * 8u + (unsigned int)warp;
    if (group >= visible_groups) return;
    const unsigned int groups_per_page = block_size / COMPRESS_RATIO;
    const unsigned int logical_page = group / groups_per_page;
    const unsigned int group_in_page = group % groups_per_page;
    const int physical_page = block_table[logical_page];
    if (physical_page < 0) {
        if (lane == 0) logits[group] = -CUDART_INF_F;
        return;
    }
    const unsigned long long key_base =
        ((unsigned long long)physical_page * groups_per_page + group_in_page) * INDEX_DIM;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < INDEX_HEADS; ++h) {
        float partial = 0.0f;
#pragma unroll
        for (int d = lane; d < INDEX_DIM; d += 32) {
            partial += __bfloat162float(query[h * INDEX_DIM + d]) *
                       __bfloat162float(compressed_cache[key_base + d]);
        }
        partial = warp_sum(partial);
        if (lane == 0) score += fmaxf(partial, 0.0f);
    }
    if (lane == 0) logits[group] = score * 0.08838834764831845f;  // 1/sqrt(128)
}

// Exact, deterministic correctness path.  A single thread maintains a
// 512-entry min-heap, then sorts the selected logical groups by position so
// sparse attention has a stable reduction order.  This is intentionally easy
// to audit; a parallel radix selector can replace it without changing the ABI.
extern "C" __global__ void qwen4_qsa_select_expand(
    const float* __restrict__ logits,
    int* __restrict__ token_indices,
    unsigned int visible_groups,
    unsigned int position,
    unsigned int sequence_length) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    __shared__ float heap_values[BLOCK_TOPK];
    __shared__ int heap_indices[BLOCK_TOPK];
    for (int i = 0; i < OUTPUT_WIDTH; ++i) token_indices[i] = -1;

    const int count = visible_groups < BLOCK_TOPK ? (int)visible_groups : BLOCK_TOPK;
    if (visible_groups <= BLOCK_TOPK) {
        for (int i = 0; i < count; ++i) heap_indices[i] = i;
    } else {
        for (int i = 0; i < BLOCK_TOPK; ++i) {
            heap_values[i] = logits[i];
            heap_indices[i] = i;
        }
        for (int i = BLOCK_TOPK / 2 - 1; i >= 0; --i) {
            heap_sift_down(heap_values, heap_indices, i, BLOCK_TOPK);
        }
        for (unsigned int i = BLOCK_TOPK; i < visible_groups; ++i) {
            const float value = logits[i];
            if (!less_pair(value, (int)i, heap_values[0], heap_indices[0])) {
                heap_values[0] = value;
                heap_indices[0] = (int)i;
                heap_sift_down(heap_values, heap_indices, 0, BLOCK_TOPK);
            }
        }
        // Stable positional order for the subsequent softmax reduction.
        for (int i = 1; i < BLOCK_TOPK; ++i) {
            int key = heap_indices[i];
            int j = i - 1;
            while (j >= 0 && heap_indices[j] > key) {
                heap_indices[j + 1] = heap_indices[j];
                --j;
            }
            heap_indices[j + 1] = key;
        }
    }

    int out = 0;
    for (int rank = 0; rank < count; ++rank) {
        const int group = heap_indices[rank];
#pragma unroll
        for (int offset = 0; offset < COMPRESS_RATIO; ++offset) {
            const int token = group * COMPRESS_RATIO + offset;
            if ((unsigned int)token < sequence_length) token_indices[out++] = token;
        }
    }
    const unsigned int tail_start = ((position + 1) / COMPRESS_RATIO) * COMPRESS_RATIO;
    for (unsigned int token = tail_start;
         token <= position && token < sequence_length && out < OUTPUT_WIDTH;
         ++token) {
        token_indices[out++] = (int)token;
    }
}

// Graph-capture variant of qwen4_qsa_score.  The launch topology is selected
// by the host before capture, while the advancing sequence length is read from
// Atlas's persistent device metadata on every replay.
extern "C" __global__ void qwen4_qsa_score_device_meta(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ compressed_cache,
    const int* __restrict__ block_table,
    float* __restrict__ logits,
    const unsigned int* __restrict__ sequence_length_ptr,
    unsigned int block_size) {
    if (block_size < COMPRESS_RATIO) return;
    const unsigned int visible_groups = *sequence_length_ptr / COMPRESS_RATIO;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const unsigned int group = blockIdx.x * 8u + (unsigned int)warp;
    if (group >= visible_groups) return;
    const unsigned int groups_per_page = block_size / COMPRESS_RATIO;
    const unsigned int logical_page = group / groups_per_page;
    const unsigned int group_in_page = group % groups_per_page;
    const int physical_page = block_table[logical_page];
    if (physical_page < 0) {
        if (lane == 0) logits[group] = -CUDART_INF_F;
        return;
    }
    const unsigned long long key_base =
        ((unsigned long long)physical_page * groups_per_page + group_in_page) * INDEX_DIM;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < INDEX_HEADS; ++h) {
        float partial = 0.0f;
#pragma unroll
        for (int d = lane; d < INDEX_DIM; d += 32) {
            partial += __bfloat162float(query[h * INDEX_DIM + d]) *
                       __bfloat162float(compressed_cache[key_base + d]);
        }
        partial = warp_sum(partial);
        if (lane == 0) score += fmaxf(partial, 0.0f);
    }
    if (lane == 0) logits[group] = score * 0.08838834764831845f;
}

// Graph-capture variant of qwen4_qsa_select_expand.  Keep the incumbent
// deterministic heap and positional reduction order byte-for-byte equivalent;
// only the three frame-varying scalar arguments move behind stable pointers.
extern "C" __global__ void qwen4_qsa_select_expand_device_meta(
    const float* __restrict__ logits,
    int* __restrict__ token_indices,
    const unsigned int* __restrict__ position_ptr,
    const unsigned int* __restrict__ sequence_length_ptr) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    (void)position_ptr;  // Retained launch ABI; this is not a rotary query index.
    const unsigned int sequence_length = *sequence_length_ptr;
    const unsigned int visible_groups = sequence_length / COMPRESS_RATIO;
    __shared__ float heap_values[BLOCK_TOPK];
    __shared__ int heap_indices[BLOCK_TOPK];
    for (int i = 0; i < OUTPUT_WIDTH; ++i) token_indices[i] = -1;
    unsigned int position;
    if (!qwen4_qsa_positions::physical_query(sequence_length, position)) return;

    const int count = visible_groups < BLOCK_TOPK ? (int)visible_groups : BLOCK_TOPK;
    if (visible_groups <= BLOCK_TOPK) {
        for (int i = 0; i < count; ++i) heap_indices[i] = i;
    } else {
        for (int i = 0; i < BLOCK_TOPK; ++i) {
            heap_values[i] = logits[i];
            heap_indices[i] = i;
        }
        for (int i = BLOCK_TOPK / 2 - 1; i >= 0; --i) {
            heap_sift_down(heap_values, heap_indices, i, BLOCK_TOPK);
        }
        for (unsigned int i = BLOCK_TOPK; i < visible_groups; ++i) {
            const float value = logits[i];
            if (!less_pair(value, (int)i, heap_values[0], heap_indices[0])) {
                heap_values[0] = value;
                heap_indices[0] = (int)i;
                heap_sift_down(heap_values, heap_indices, 0, BLOCK_TOPK);
            }
        }
        for (int i = 1; i < BLOCK_TOPK; ++i) {
            int key = heap_indices[i];
            int j = i - 1;
            while (j >= 0 && heap_indices[j] > key) {
                heap_indices[j + 1] = heap_indices[j];
                --j;
            }
            heap_indices[j + 1] = key;
        }
    }

    int out = 0;
    for (int rank = 0; rank < count; ++rank) {
        const int group = heap_indices[rank];
#pragma unroll
        for (int offset = 0; offset < COMPRESS_RATIO; ++offset) {
            const int token = group * COMPRESS_RATIO + offset;
            if ((unsigned int)token < sequence_length) token_indices[out++] = token;
        }
    }
    const unsigned int tail_start = ((position + 1) / COMPRESS_RATIO) * COMPRESS_RATIO;
    for (unsigned int token = tail_start;
         token <= position && token < sequence_length && out < OUTPUT_WIDTH;
         ++token) {
        token_indices[out++] = (int)token;
    }
}

// Split-token sparse attention. Eight CTAs cooperate per query head so the
// 24-head Qwen4 geometry exposes 192 CTAs instead of underfilling GB10 with
// only 24. Each CTA writes a softmax partial; the reduction kernel combines
// them with the exact log-sum-exp identity.
template <bool SKIP_NEGATIVE>
__device__ __forceinline__ void qwen4_qsa_sparse_attention_bf16_partial_impl(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_elems,
    unsigned long long v_block_stride_elems,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits,
    int index_width) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split = blockIdx.y;
    const unsigned int d = threadIdx.x;
    if (q_head >= num_query_heads || head_dim == 0u || head_dim > 256u || blockDim.x != head_dim) return;
    const unsigned int group_size = num_query_heads / num_kv_heads;
    const unsigned int kv_head = q_head / group_size;
    const float qv = __bfloat162float(query[(unsigned long long)q_head * head_dim + d]);
    float acc = 0.0f;
    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    __shared__ float reduce[256];
    __shared__ float alpha_shared;
    __shared__ float beta_shared;

    for (int rank = (int)split; rank < index_width; rank += (int)num_splits) {
        const int logical_token = token_indices[rank];
        if (logical_token < 0) {
            if (SKIP_NEGATIVE) continue;
            break;
        }
        const unsigned int logical_page = (unsigned int)logical_token / block_size;
        const unsigned int token_in_page = (unsigned int)logical_token % block_size;
        const int physical_page = block_table[logical_page];
        if (physical_page < 0) continue;
        const unsigned long long k_index =
            (unsigned long long)physical_page * k_block_stride_elems +
            ((unsigned long long)token_in_page * num_kv_heads + kv_head) * head_dim + d;
        reduce[d] = qv * __bfloat162float(k_cache[k_index]);
        __syncthreads();
        for (unsigned int stride = head_dim >> 1; stride > 0; stride >>= 1) {
            if (d < stride) reduce[d] += reduce[d + stride];
            __syncthreads();
        }
        if (d == 0) {
            const float score = reduce[0] * softmax_scale;
            const float next_max = fmaxf(row_max, score);
            const float alpha = row_max == -CUDART_INF_F ? 0.0f : __expf(row_max - next_max);
            const float beta = __expf(score - next_max);
            row_sum = row_sum * alpha + beta;
            row_max = next_max;
            alpha_shared = alpha;
            beta_shared = beta;
        }
        __syncthreads();
        const unsigned long long v_index =
            (unsigned long long)physical_page * v_block_stride_elems +
            ((unsigned long long)token_in_page * num_kv_heads + kv_head) * head_dim + d;
        acc = acc * alpha_shared + beta_shared * __bfloat162float(v_cache[v_index]);
        __syncthreads();
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    const unsigned long long part = (unsigned long long)q_head * num_splits + split;
    partial_output[part * head_dim + d] = acc;
    if (d == 0) {
        partial_max[part] = row_max;
        partial_sum[part] = row_sum;
    }
}

extern "C" __global__ void qwen4_qsa_sparse_attention_bf16_partial(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_elems,
    unsigned long long v_block_stride_elems,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits) {
    qwen4_qsa_sparse_attention_bf16_partial_impl<false>(
        query,
        k_cache,
        v_cache,
        token_indices,
        block_table,
        partial_output,
        partial_max,
        partial_sum,
        k_block_stride_elems,
        v_block_stride_elems,
        block_size,
        num_query_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        num_splits,
        OUTPUT_WIDTH);
}

// Reused-selection BF16 split-token sparse attention for MTP draft steps 1..K-1.
// Parameterizes the token-index scan width (`index_width = OUTPUT_WIDTH + step`)
// and skips `-1` holes between the base selection tail and appended draft slots.
extern "C" __global__ void qwen4_qsa_sparse_attention_bf16_partial_reused(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_elems,
    unsigned long long v_block_stride_elems,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits,
    int index_width) {
    qwen4_qsa_sparse_attention_bf16_partial_impl<true>(
        query,
        k_cache,
        v_cache,
        token_indices,
        block_table,
        partial_output,
        partial_max,
        partial_sum,
        k_block_stride_elems,
        v_block_stride_elems,
        block_size,
        num_query_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        num_splits,
        index_width);
}

// Same sparse-token contract as the BF16 path, but consumes Atlas's packed
// NVFP4 paged-KV layout in place. Keeping the 1M-token history compressed is
// load-bearing on a single 128 GiB GB10.
template <bool SKIP_NEGATIVE>
__device__ __forceinline__ void qwen4_qsa_sparse_attention_nvfp4_partial_impl(
    const __nv_bfloat16* __restrict__ query,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_bytes,
    unsigned long long v_block_stride_bytes,
    unsigned long long data_section_bytes,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits,
    int index_width) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split = blockIdx.y;
    const unsigned int d = threadIdx.x;
    if (q_head >= num_query_heads || head_dim == 0u || head_dim > 256u || blockDim.x != head_dim) return;
    const unsigned int group_size = num_query_heads / num_kv_heads;
    const unsigned int kv_head = q_head / group_size;
    const float qv = __bfloat162float(query[(unsigned long long)q_head * head_dim + d]);
    float acc = 0.0f;
    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    __shared__ float reduce[256];
    __shared__ float alpha_shared;
    __shared__ float beta_shared;

    for (int rank = (int)split; rank < index_width; rank += (int)num_splits) {
        const int logical_token = token_indices[rank];
        if (logical_token < 0) {
            if (SKIP_NEGATIVE) continue;
            break;
        }
        const unsigned int logical_page = (unsigned int)logical_token / block_size;
        const unsigned int token_in_page = (unsigned int)logical_token % block_size;
        const int physical_page = block_table[logical_page];
        if (physical_page < 0) continue;
        const unsigned long long element =
            ((unsigned long long)token_in_page * num_kv_heads + kv_head) * head_dim + d;
        const unsigned char* k_block =
            k_cache + (unsigned long long)physical_page * k_block_stride_bytes;
        reduce[d] = qv * load_nvfp4(k_block, element, data_section_bytes);
        __syncthreads();
        for (unsigned int stride = head_dim >> 1; stride > 0; stride >>= 1) {
            if (d < stride) reduce[d] += reduce[d + stride];
            __syncthreads();
        }
        if (d == 0) {
            const float score = reduce[0] * softmax_scale;
            const float next_max = fmaxf(row_max, score);
            const float alpha = row_max == -CUDART_INF_F ? 0.0f : __expf(row_max - next_max);
            const float beta = __expf(score - next_max);
            row_sum = row_sum * alpha + beta;
            row_max = next_max;
            alpha_shared = alpha;
            beta_shared = beta;
        }
        __syncthreads();
        const unsigned char* v_block =
            v_cache + (unsigned long long)physical_page * v_block_stride_bytes;
        acc = acc * alpha_shared + beta_shared * load_nvfp4(v_block, element, data_section_bytes);
        __syncthreads();
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    const unsigned long long part = (unsigned long long)q_head * num_splits + split;
    partial_output[part * head_dim + d] = acc;
    if (d == 0) {
        partial_max[part] = row_max;
        partial_sum[part] = row_sum;
    }
}

extern "C" __global__ void qwen4_qsa_sparse_attention_nvfp4_partial(
    const __nv_bfloat16* __restrict__ query,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_bytes,
    unsigned long long v_block_stride_bytes,
    unsigned long long data_section_bytes,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits) {
    qwen4_qsa_sparse_attention_nvfp4_partial_impl<false>(
        query,
        k_cache,
        v_cache,
        token_indices,
        block_table,
        partial_output,
        partial_max,
        partial_sum,
        k_block_stride_bytes,
        v_block_stride_bytes,
        data_section_bytes,
        block_size,
        num_query_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        num_splits,
        OUTPUT_WIDTH);
}

// Reused-selection NVFP4 split-token sparse attention for MTP draft steps 1..K-1.
extern "C" __global__ void qwen4_qsa_sparse_attention_nvfp4_partial_reused(
    const __nv_bfloat16* __restrict__ query,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_bytes,
    unsigned long long v_block_stride_bytes,
    unsigned long long data_section_bytes,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits,
    int index_width) {
    qwen4_qsa_sparse_attention_nvfp4_partial_impl<true>(
        query,
        k_cache,
        v_cache,
        token_indices,
        block_table,
        partial_output,
        partial_max,
        partial_sum,
        k_block_stride_bytes,
        v_block_stride_bytes,
        data_section_bytes,
        block_size,
        num_query_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        num_splits,
        index_width);
}

extern "C" __global__ void qwen4_qsa_sparse_attention_bf16_reduce(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_query_heads,
    unsigned int head_dim,
    unsigned int num_splits) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (q_head >= num_query_heads || d >= head_dim) return;
    const unsigned long long base = (unsigned long long)q_head * num_splits;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    float global_max = -CUDART_INF_F;
    for (unsigned int split = 0; split < num_splits; ++split)
        global_max = fmaxf(global_max, partial_max[base + split]);
    float denom = 0.0f;
    float acc = 0.0f;
    if (global_max > -CUDART_INF_F) {
        for (unsigned int split = 0; split < num_splits; ++split) {
            const float scale = __expf(partial_max[base + split] - global_max);
            denom += partial_sum[base + split] * scale;
            acc += partial_output[(base + split) * head_dim + d] * scale;
        }
    }
    output[(unsigned long long)q_head * head_dim + d] =
        __float2bfloat16_rn(denom > 0.0f ? acc / denom : 0.0f);
}

extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_sparse_attention_bf16_reduce_rows(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_query_heads,
    unsigned int head_dim,
    unsigned int num_splits) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int d = threadIdx.x;
    if (q_head >= num_query_heads || d >= head_dim) return;
    const unsigned long long base =
        ((unsigned long long)row * num_query_heads + q_head) * num_splits;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    float global_max = -CUDART_INF_F;
    for (unsigned int split = 0; split < num_splits; ++split)
        global_max = fmaxf(global_max, partial_max[base + split]);
    float denom = 0.0f;
    float acc = 0.0f;
    if (global_max > -CUDART_INF_F) {
        for (unsigned int split = 0; split < num_splits; ++split) {
            const float scale = __expf(partial_max[base + split] - global_max);
            denom += partial_sum[base + split] * scale;
            acc += partial_output[(base + split) * head_dim + d] * scale;
        }
    }
    output[((unsigned long long)row * num_query_heads + q_head) * head_dim + d] =
        __float2bfloat16_rn(denom > 0.0f ? acc / denom : 0.0f);
}

extern "C" __global__ void qwen4_qsa_sparse_attention_bf16_reduce_gated(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_query_heads,
    unsigned int head_dim,
    unsigned int num_splits) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (q_head >= num_query_heads || d >= head_dim) return;
    const unsigned long long base = (unsigned long long)q_head * num_splits;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    float global_max = -CUDART_INF_F;
    for (unsigned int split = 0; split < num_splits; ++split)
        global_max = fmaxf(global_max, partial_max[base + split]);
    float denom = 0.0f;
    float acc = 0.0f;
    if (global_max > -CUDART_INF_F) {
        for (unsigned int split = 0; split < num_splits; ++split) {
            const float scale = __expf(partial_max[base + split] - global_max);
            denom += partial_sum[base + split] * scale;
            acc += partial_output[(base + split) * head_dim + d] * scale;
        }
    }
    const unsigned long long elem = (unsigned long long)q_head * head_dim + d;
    const __nv_bfloat16 raw_bf16 =
        __float2bfloat16_rn(denom > 0.0f ? acc / denom : 0.0f);
    const float x = __bfloat162float(raw_bf16);
    const float g = __bfloat162float(gate[elem]);
    const float sigmoid_g = 1.0f / (1.0f + expf(-g));
    output[elem] = __float2bfloat16(x * sigmoid_g);
}

extern "C" __global__ void __launch_bounds__(256)
qwen4_qsa_sparse_attention_bf16_reduce_gated_rows(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_query_heads,
    unsigned int head_dim,
    unsigned int num_splits,
    unsigned int gate_stride) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int d = threadIdx.x;
    if (q_head >= num_query_heads || d >= head_dim) return;
    const unsigned long long base =
        ((unsigned long long)row * num_query_heads + q_head) * num_splits;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    float global_max = -CUDART_INF_F;
    for (unsigned int split = 0; split < num_splits; ++split)
        global_max = fmaxf(global_max, partial_max[base + split]);
    float denom = 0.0f;
    float acc = 0.0f;
    if (global_max > -CUDART_INF_F) {
        for (unsigned int split = 0; split < num_splits; ++split) {
            const float scale = __expf(partial_max[base + split] - global_max);
            denom += partial_sum[base + split] * scale;
            acc += partial_output[(base + split) * head_dim + d] * scale;
        }
    }
    const unsigned long long head_elem = (unsigned long long)q_head * head_dim + d;
    const __nv_bfloat16 raw_bf16 =
        __float2bfloat16_rn(denom > 0.0f ? acc / denom : 0.0f);
    const float x = __bfloat162float(raw_bf16);
    const float g = __bfloat162float(gate[(unsigned long long)row * gate_stride + head_elem]);
    const float sigmoid_g = 1.0f / (1.0f + expf(-g));
    output[((unsigned long long)row * num_query_heads + q_head) * head_dim + d] =
        __float2bfloat16(x * sigmoid_g);
}

// ---------------------------------------------------------------------------
// Parallel top-BLOCK_TOPK select (ATLAS_QWEN4_QSA_SELECT_PAR=1).
//
// Bit-exact replacement for qwen4_qsa_select_expand{,_device_meta}: same
// selected group set under the same total order (higher logit first, lower
// group index first on equal logits, i.e. !less_pair), same ascending-index
// output order, same token expansion, same -1 padding and decode tail.
// The serial kernels ran on ONE thread (heap over all groups + O(k^2)
// insertion sort), ~2 ms per attention layer per decode token at 8k context.
//
// Method: one 1024-thread CTA per row. A 4-pass 8-bit radix select finds the
// threshold key T of the BLOCK_TOPK-th best logit and how many T-valued groups
// to take (lowest indices first); a tiled block scan then compacts the
// selected groups in ascending index order. Keys are the order-preserving
// uint32 image of the float with -0 folded onto +0, so key equality is float
// equality. NaN-free logits are assumed (the score kernels emit relu(.)*c or
// -inf); with NaN the serial heap itself is order-dependent.
// ---------------------------------------------------------------------------
namespace {

constexpr int SELECT_THREADS = 1024;
constexpr int SELECT_WARPS = SELECT_THREADS / 32;

__device__ __forceinline__ unsigned int qsa_order_key(float value) {
    if (value == 0.0f) value = 0.0f;  // fold -0 onto +0 (they compare equal)
    const unsigned int bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Exclusive block scan of one int per thread; returns the block total in *total.
__device__ __forceinline__ int qsa_block_exclusive_scan(int x, int* warp_buf, int* total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int inclusive = x;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += y;
    }
    if (lane == 31) warp_buf[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
        int w = lane < SELECT_WARPS ? warp_buf[lane] : 0;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, offset);
            if (lane >= offset) w += y;
        }
        if (lane < SELECT_WARPS) warp_buf[lane] = w;  // inclusive warp prefix
    }
    __syncthreads();
    const int warp_prefix = warp == 0 ? 0 : warp_buf[warp - 1];
    *total = warp_buf[SELECT_WARPS - 1];
    __syncthreads();  // warp_buf is reused by the next call
    return warp_prefix + inclusive - x;
}

__device__ void qsa_select_expand_parallel(
    const float* __restrict__ logits,
    int* __restrict__ token_indices,
    unsigned int visible_groups,
    unsigned int position,
    unsigned int sequence_length) {
    __shared__ unsigned int hist[256];
    __shared__ int warp_buf[SELECT_WARPS];
    __shared__ unsigned int sel_prefix;
    __shared__ unsigned int sel_remaining;
    const int tid = threadIdx.x;

    for (int i = tid; i < OUTPUT_WIDTH; i += SELECT_THREADS) token_indices[i] = -1;
    __syncthreads();

    const int count = visible_groups < BLOCK_TOPK ? (int)visible_groups : BLOCK_TOPK;
    if (visible_groups <= (unsigned int)BLOCK_TOPK) {
        // Serial path: groups 0..count-1 in order. Every token 4g+o is below
        // 4*visible_groups <= sequence_length, so the expansion never filters.
        for (int r = tid; r < count * COMPRESS_RATIO; r += SELECT_THREADS) {
            if ((unsigned int)r < sequence_length) token_indices[r] = r;
        }
    } else {
        // Radix select of the BLOCK_TOPK-th largest key.
        if (tid == 0) {
            sel_prefix = 0u;
            sel_remaining = (unsigned int)BLOCK_TOPK;
        }
        unsigned int prefix_mask = 0u;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int b = tid; b < 256; b += SELECT_THREADS) hist[b] = 0u;
            __syncthreads();
            const unsigned int prefix = sel_prefix;
            for (unsigned int i = tid; i < visible_groups; i += SELECT_THREADS) {
                const unsigned int key = qsa_order_key(logits[i]);
                if ((key & prefix_mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (tid < 32) {
                // Lane l owns digits [255-8l-7, 255-8l], walked from the top.
                const int lane = tid;
                // Read before any lane can write it below (independent thread scheduling).
                const unsigned int need = sel_remaining;
                unsigned int chunk = 0u;
#pragma unroll
                for (int j = 0; j < 8; ++j) chunk += hist[255 - lane * 8 - j];
                unsigned int inclusive = chunk;
#pragma unroll
                for (int offset = 1; offset < 32; offset <<= 1) {
                    const unsigned int y = __shfl_up_sync(0xffffffffu, inclusive, offset);
                    if (lane >= offset) inclusive += y;
                }
                const unsigned int before = inclusive - chunk;
                const bool mine = before < need && need <= inclusive;
                if (mine) {
                    unsigned int above = before;
                    for (int j = 0; j < 8; ++j) {
                        const unsigned int digit = 255u - (unsigned int)(lane * 8 + j);
                        const unsigned int h = hist[digit];
                        if (above + h >= need) {
                            sel_prefix = prefix | (digit << shift);
                            sel_remaining = need - above;
                            break;
                        }
                        above += h;
                    }
                }
            }
            prefix_mask |= 255u << shift;
            __syncthreads();
        }
        const unsigned int threshold = sel_prefix;
        const int take_equal = (int)sel_remaining;  // T-valued groups to keep, lowest index first

        // Ordered compaction: selected groups in ascending group index.
        int equal_base = 0;
        int out_base = 0;
        for (unsigned int base = 0; base < visible_groups; base += SELECT_THREADS) {
            const unsigned int i = base + (unsigned int)tid;
            const bool valid = i < visible_groups;
            const unsigned int key = valid ? qsa_order_key(logits[i]) : 0u;
            const bool greater = valid && key > threshold;
            const bool equal = valid && key == threshold;
            int equal_total;
            const int equal_rank = equal_base + qsa_block_exclusive_scan(equal ? 1 : 0, warp_buf, &equal_total);
            const bool selected = greater || (equal && equal_rank < take_equal);
            int selected_total;
            const int rank = out_base + qsa_block_exclusive_scan(selected ? 1 : 0, warp_buf, &selected_total);
            if (selected) {
#pragma unroll
                for (int offset = 0; offset < COMPRESS_RATIO; ++offset) {
                    const unsigned int token = i * COMPRESS_RATIO + offset;
                    // i < visible_groups => token < sequence_length; kept for parity.
                    if (token < sequence_length) token_indices[rank * COMPRESS_RATIO + offset] = (int)token;
                }
            }
            equal_base += equal_total;
            out_base += selected_total;
        }
    }
    __syncthreads();
    if (tid == 0) {
        int out = count * COMPRESS_RATIO;
        const unsigned int tail_start = ((position + 1) / COMPRESS_RATIO) * COMPRESS_RATIO;
        for (unsigned int token = tail_start;
             token <= position && token < sequence_length && out < OUTPUT_WIDTH;
             ++token) {
            token_indices[out++] = (int)token;
        }
    }
}

}  // namespace

// Host-argument ABI of qwen4_qsa_select_expand. Launch grid [1,1,1], block [1024,1,1].
extern "C" __global__ void __launch_bounds__(1024) qwen4_qsa_select_expand_par(
    const float* __restrict__ logits,
    int* __restrict__ token_indices,
    unsigned int visible_groups,
    unsigned int position,
    unsigned int sequence_length) {
    if (blockIdx.x != 0) return;
    qsa_select_expand_parallel(logits, token_indices, visible_groups, position, sequence_length);
}

// Device-metadata ABI of qwen4_qsa_select_expand_device_meta (graph-safe).
// Launch grid [1,1,1], block [1024,1,1].
extern "C" __global__ void __launch_bounds__(1024) qwen4_qsa_select_expand_device_meta_par(
    const float* __restrict__ logits,
    int* __restrict__ token_indices,
    const unsigned int* __restrict__ position_ptr,
    const unsigned int* __restrict__ sequence_length_ptr) {
    if (blockIdx.x != 0) return;
    (void)position_ptr;  // same ABI as the serial kernel; position derives from seq_len
    const unsigned int sequence_length = *sequence_length_ptr;
    unsigned int position;
    if (!qwen4_qsa_positions::physical_query(sequence_length, position)) {
        for (int i = threadIdx.x; i < OUTPUT_WIDTH; i += SELECT_THREADS) token_indices[i] = -1;
        return;
    }
    qsa_select_expand_parallel(
        logits, token_indices, sequence_length / COMPRESS_RATIO, position, sequence_length);
}

// Multi-row ABI for batched decode: grid [1, rows, 1], block [1024,1,1].
// Row r reads logits[r*logits_row_stride ..], writes token_indices[r*OUTPUT_WIDTH ..],
// and takes its sequence length from sequence_lengths[r] (position = len-1).
// rows == 1 is byte-identical to qwen4_qsa_select_expand_device_meta_par.
extern "C" __global__ void __launch_bounds__(1024) qwen4_qsa_select_expand_rows(
    const float* __restrict__ logits,
    unsigned long long logits_row_stride,
    int* __restrict__ token_indices,
    const unsigned int* __restrict__ sequence_lengths) {
    if (blockIdx.x != 0) return;
    const unsigned int row = blockIdx.y;
    const float* row_logits = logits + (unsigned long long)row * logits_row_stride;
    int* row_indices = token_indices + (unsigned long long)row * OUTPUT_WIDTH;
    const unsigned int sequence_length = sequence_lengths[row];
    unsigned int position;
    if (!qwen4_qsa_positions::physical_query(sequence_length, position)) {
        for (int i = threadIdx.x; i < OUTPUT_WIDTH; i += SELECT_THREADS) row_indices[i] = -1;
        return;
    }
    qsa_select_expand_parallel(
        row_logits, row_indices, sequence_length / COMPRESS_RATIO, position, sequence_length);
}

// ---------------------------------------------------------------------------
// qsa_select_expand_v3 (fn-decode, ATLAS_QWEN4_QSA_SELECT_V3=1): byte-identical
// to qsa_select_expand_parallel (same threshold digits, same lowest-index tie
// rule, same ascending output, same -1 fill and tail). The incumbent re-reads
// every logit in each of its 4 radix passes and compacts with two block scans
// per 1024 elements (64 at 128k). v3 keeps the incumbent's strided ownership
// (element j of thread t is i = j*1024 + t, so index order = (j, warp, lane)):
//   * keys are loaded once into registers (j < ceil(n/1024));
//   * histogram: integer counts of the same matching keys (order-free); a warp
//     whose matching lanes share one digit adds popc once;
//   * compaction: per-(j, warp) greater/equal ballot counts in shared memory,
//     two block scans over that table in index order, then one write pass.
// visible_groups <= 3072 runs the incumbent; > 1024*SV3_E runs qsa_select_expand_v3_slabbed (same bytes either way).
// ---------------------------------------------------------------------------
namespace {
constexpr int SV3_E = 33;                                   // register-cached keys per thread (<=135K)
constexpr unsigned int SV3_MAX = SELECT_THREADS * SV3_E;    // 33792 groups = 135168 tokens
constexpr int SV3_SLAB_E = 32;                              // slab rows/thread above SV3_MAX (32768 groups/slab)
constexpr int SV3_CELLS = SV3_E * SELECT_WARPS;             // (j, warp) cells (>= SV3_SLAB_E * SELECT_WARPS)
constexpr unsigned int SV3_MIN = 3072;                     // incumbent is faster at or below this

__device__ __forceinline__ unsigned int sv3_lanemask_lt() {
    unsigned int m;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
    return m;
}

// Long-context (>135K..1M+ tokens, visible_groups > SV3_MAX) zero-spill slabbed v3 path:
// keeps v3's warp-aggregated radix histogram (collapsing uniform digits such as ReLU-zero
// groups to 1 atomic per warp) and compacts in 32,768-group (32 * 1024) slabs using the
// shared c_gt/c_eq cell tables (2 block scans per 32K groups instead of 64, and skipping
// empty slabs completely). Bit-for-bit identical (0 ULP) to qsa_select_expand_parallel.
__device__ __forceinline__ void qsa_select_expand_v3_slabbed(
    const float* __restrict__ logits,
    int* __restrict__ token_indices,
    unsigned int visible_groups,
    unsigned int position,
    unsigned int sequence_length,
    unsigned int* hist,
    int* warp_buf,
    int* c_gt,
    int* c_eq,
    unsigned int* sel_prefix_s,
    unsigned int* sel_remaining_s) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned int lt = sv3_lanemask_lt();
    const int jn = (int)((visible_groups + SELECT_THREADS - 1u) / SELECT_THREADS);

    for (int i = tid; i < OUTPUT_WIDTH; i += SELECT_THREADS) token_indices[i] = -1;
    if (tid == 0) {
        *sel_prefix_s = 0u;
        *sel_remaining_s = (unsigned int)BLOCK_TOPK;
    }
    unsigned int prefix_mask = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += SELECT_THREADS) hist[b] = 0u;
        __syncthreads();
        const unsigned int prefix = *sel_prefix_s;
        for (int j = 0; j < jn; ++j) {
            const unsigned int i = (unsigned int)j * SELECT_THREADS + (unsigned int)tid;
            const unsigned int key = (i < visible_groups) ? qsa_order_key(__ldg(logits + i)) : 0u;
            const bool m = (i < visible_groups) && ((key & prefix_mask) == prefix);
            const unsigned int digit = (key >> shift) & 255u;
            const unsigned int mb = __ballot_sync(0xffffffffu, m);
            if (mb != 0u) {
                const int leader = __ffs(mb) - 1;
                const unsigned int ld = __shfl_sync(0xffffffffu, digit, leader);
                if (__all_sync(0xffffffffu, !m || digit == ld)) {
                    if (lane == leader) atomicAdd(&hist[ld], (unsigned int)__popc(mb));
                } else if (m) {
                    atomicAdd(&hist[digit], 1u);
                }
            }
        }
        __syncthreads();
        if (tid < 32) {
            const unsigned int need = *sel_remaining_s;
            unsigned int chunk = 0u;
#pragma unroll
            for (int j = 0; j < 8; ++j) chunk += hist[255 - lane * 8 - j];
            unsigned int inclusive = chunk;
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                const unsigned int y = __shfl_up_sync(0xffffffffu, inclusive, offset);
                if (lane >= offset) inclusive += y;
            }
            const unsigned int before = inclusive - chunk;
            if (before < need && need <= inclusive) {
                unsigned int above = before;
                for (int j = 0; j < 8; ++j) {
                    const unsigned int digit = 255u - (unsigned int)(lane * 8 + j);
                    const unsigned int h = hist[digit];
                    if (above + h >= need) {
                        *sel_prefix_s = prefix | (digit << shift);
                        *sel_remaining_s = need - above;
                        break;
                    }
                    above += h;
                }
            }
        }
        prefix_mask |= 255u << shift;
        __syncthreads();
    }

    const unsigned int threshold = *sel_prefix_s;
    const int take_equal = (int)(*sel_remaining_s);
    int equal_base = 0;
    int out_base = 0;

    for (int slab_j0 = 0; slab_j0 < jn && out_base < BLOCK_TOPK; slab_j0 += SV3_SLAB_E) {
        const int slab_jn = min(SV3_SLAB_E, jn - slab_j0);
        for (int jj = 0; jj < slab_jn; ++jj) {
            const unsigned int i =
                (unsigned int)(slab_j0 + jj) * SELECT_THREADS + (unsigned int)tid;
            const bool valid = i < visible_groups;
            const unsigned int key = valid ? qsa_order_key(__ldg(logits + i)) : 0u;
            const unsigned int bg = __ballot_sync(0xffffffffu, valid && key > threshold);
            const unsigned int be = __ballot_sync(0xffffffffu, valid && key == threshold);
            if (lane == 0) {
                c_gt[jj * SELECT_WARPS + warp] = __popc(bg);
                c_eq[jj * SELECT_WARPS + warp] = __popc(be);
            }
        }
        __syncthreads();

        const int ncells = slab_jn * SELECT_WARPS;
        const int c0 = 2 * tid, c1 = 2 * tid + 1;
        const int e0 = c0 < ncells ? c_eq[c0] : 0;
        const int e1 = c1 < ncells ? c_eq[c1] : 0;
        int slab_eq_total = 0;
        const int eb0 = equal_base + qsa_block_exclusive_scan(e0 + e1, warp_buf, &slab_eq_total);
        const int eb1 = eb0 + e0;
        const int s0 = c0 < ncells ? c_gt[c0] + min(max(take_equal - eb0, 0), e0) : 0;
        const int s1 = c1 < ncells ? c_gt[c1] + min(max(take_equal - eb1, 0), e1) : 0;
        int slab_sel_total = 0;
        const int ob0 = out_base + qsa_block_exclusive_scan(s0 + s1, warp_buf, &slab_sel_total);
        if (c0 < ncells) { c_eq[c0] = eb0; c_gt[c0] = ob0; }
        if (c1 < ncells) { c_eq[c1] = eb1; c_gt[c1] = ob0 + s0; }
        __syncthreads();

        if (slab_sel_total > 0) {
            for (int jj = 0; jj < slab_jn; ++jj) {
                const unsigned int i =
                    (unsigned int)(slab_j0 + jj) * SELECT_THREADS + (unsigned int)tid;
                const bool valid = i < visible_groups;
                const unsigned int key = valid ? qsa_order_key(__ldg(logits + i)) : 0u;
                const bool eq = valid && key == threshold;
                const unsigned int be = __ballot_sync(0xffffffffu, eq);
                const int cell = jj * SELECT_WARPS + warp;
                const bool sel = (valid && key > threshold)
                                 || (eq && c_eq[cell] + __popc(be & lt) < take_equal);
                const unsigned int bs = __ballot_sync(0xffffffffu, sel);
                if (sel) {
                    const int rank = c_gt[cell] + __popc(bs & lt);
#pragma unroll
                    for (int offset = 0; offset < COMPRESS_RATIO; ++offset) {
                        const unsigned int token = i * COMPRESS_RATIO + (unsigned int)offset;
                        if (token < sequence_length) {
                            token_indices[rank * COMPRESS_RATIO + offset] = (int)token;
                        }
                    }
                }
            }
        }
        equal_base += slab_eq_total;
        out_base += slab_sel_total;
        __syncthreads();
    }
    if (tid == 0) {
        int out = BLOCK_TOPK * COMPRESS_RATIO;
        const unsigned int tail_start = ((position + 1) / COMPRESS_RATIO) * COMPRESS_RATIO;
        for (unsigned int token = tail_start;
             token <= position && token < sequence_length && out < OUTPUT_WIDTH;
             ++token) {
            token_indices[out++] = (int)token;
        }
    }
}

__device__ void qsa_select_expand_v3(
    const float* __restrict__ logits,
    int* __restrict__ token_indices,
    unsigned int visible_groups,
    unsigned int position,
    unsigned int sequence_length) {
    // Below ~3k groups (12k tokens) the incumbent's strided loop is shorter than v3's fixed passes.
    if (visible_groups <= SV3_MIN) {
        qsa_select_expand_parallel(logits, token_indices, visible_groups, position, sequence_length);
        return;
    }
    __shared__ unsigned int hist[256];
    __shared__ int warp_buf[SELECT_WARPS];
    __shared__ int c_gt[SV3_CELLS];
    __shared__ int c_eq[SV3_CELLS];
    __shared__ unsigned int sel_prefix;
    __shared__ unsigned int sel_remaining;
    if (visible_groups > SV3_MAX) {
        qsa_select_expand_v3_slabbed(
            logits,
            token_indices,
            visible_groups,
            position,
            sequence_length,
            hist,
            warp_buf,
            c_gt,
            c_eq,
            &sel_prefix,
            &sel_remaining);
        return;
    }
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned int lt = sv3_lanemask_lt();
    const int jn = (int)((visible_groups + SELECT_THREADS - 1) / SELECT_THREADS);  // block-uniform

    for (int i = tid; i < OUTPUT_WIDTH; i += SELECT_THREADS) token_indices[i] = -1;

    unsigned int keys[SV3_E];
#pragma unroll
    for (int j = 0; j < SV3_E; ++j) {
        const unsigned int i = (unsigned int)(j * SELECT_THREADS + tid);
        keys[j] = (j < jn && i < visible_groups) ? qsa_order_key(logits[i]) : 0u;
    }
    if (tid == 0) {
        sel_prefix = 0u;
        sel_remaining = (unsigned int)BLOCK_TOPK;
    }
    unsigned int prefix_mask = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += SELECT_THREADS) hist[b] = 0u;
        __syncthreads();
        const unsigned int prefix = sel_prefix;
#pragma unroll
        for (int j = 0; j < SV3_E; ++j) {
            if (j >= jn) break;
            const unsigned int i = (unsigned int)(j * SELECT_THREADS + tid);
            const bool m = i < visible_groups && (keys[j] & prefix_mask) == prefix;
            const unsigned int digit = (keys[j] >> shift) & 255u;
            const unsigned int mb = __ballot_sync(0xffffffffu, m);
            if (mb == 0u) continue;
            const int leader = __ffs(mb) - 1;
            const unsigned int ld = __shfl_sync(0xffffffffu, digit, leader);
            if (__all_sync(0xffffffffu, !m || digit == ld)) {
                if (lane == leader) atomicAdd(&hist[ld], (unsigned int)__popc(mb));
            } else if (m) {
                atomicAdd(&hist[digit], 1u);
            }
        }
        __syncthreads();
        if (tid < 32) {
            // Same digit walk as qsa_select_expand_parallel.
            const unsigned int need = sel_remaining;
            unsigned int chunk = 0u;
#pragma unroll
            for (int j = 0; j < 8; ++j) chunk += hist[255 - lane * 8 - j];
            unsigned int inclusive = chunk;
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                const unsigned int y = __shfl_up_sync(0xffffffffu, inclusive, offset);
                if (lane >= offset) inclusive += y;
            }
            const unsigned int before = inclusive - chunk;
            const bool mine = before < need && need <= inclusive;
            if (mine) {
                unsigned int above = before;
                for (int j = 0; j < 8; ++j) {
                    const unsigned int digit = 255u - (unsigned int)(lane * 8 + j);
                    const unsigned int h = hist[digit];
                    if (above + h >= need) {
                        sel_prefix = prefix | (digit << shift);
                        sel_remaining = need - above;
                        break;
                    }
                    above += h;
                }
            }
        }
        prefix_mask |= 255u << shift;
        __syncthreads();
    }
    const unsigned int threshold = sel_prefix;
    const int take_equal = (int)sel_remaining;

    // Per-(j, warp) greater / equal counts; cell index c = j*32 + warp is index order.
#pragma unroll
    for (int j = 0; j < SV3_E; ++j) {
        if (j >= jn) break;
        const unsigned int i = (unsigned int)(j * SELECT_THREADS + tid);
        const bool valid = i < visible_groups;
        const unsigned int bg = __ballot_sync(0xffffffffu, valid && keys[j] > threshold);
        const unsigned int be = __ballot_sync(0xffffffffu, valid && keys[j] == threshold);
        if (lane == 0) {
            c_gt[j * SELECT_WARPS + warp] = __popc(bg);
            c_eq[j * SELECT_WARPS + warp] = __popc(be);
        }
    }
    __syncthreads();
    // Exclusive scans over the cells in index order; thread t owns cells 2t, 2t+1.
    const int ncells = jn * SELECT_WARPS;
    const int c0 = 2 * tid, c1 = 2 * tid + 1;
    const int e0 = c0 < ncells ? c_eq[c0] : 0;
    const int e1 = c1 < ncells ? c_eq[c1] : 0;
    int total;
    const int eb0 = qsa_block_exclusive_scan(e0 + e1, warp_buf, &total);
    const int eb1 = eb0 + e0;
    // Equal keys are taken lowest index first: cell c keeps clamp(take_equal - eq_base, 0, eq).
    const int s0 = c0 < ncells ? c_gt[c0] + min(max(take_equal - eb0, 0), e0) : 0;
    const int s1 = c1 < ncells ? c_gt[c1] + min(max(take_equal - eb1, 0), e1) : 0;
    const int ob0 = qsa_block_exclusive_scan(s0 + s1, warp_buf, &total);
    // Reuse the tables as per-cell bases (each thread rewrites only its own cells).
    if (c0 < ncells) { c_eq[c0] = eb0; c_gt[c0] = ob0; }
    if (c1 < ncells) { c_eq[c1] = eb1; c_gt[c1] = ob0 + s0; }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < SV3_E; ++j) {
        if (j >= jn) break;
        const unsigned int i = (unsigned int)(j * SELECT_THREADS + tid);
        const bool valid = i < visible_groups;
        const bool eq = valid && keys[j] == threshold;
        const unsigned int be = __ballot_sync(0xffffffffu, eq);
        const int cell = j * SELECT_WARPS + warp;
        const bool sel = (valid && keys[j] > threshold) || (eq && c_eq[cell] + __popc(be & lt) < take_equal);
        const unsigned int bs = __ballot_sync(0xffffffffu, sel);
        if (sel) {
            const int rank = c_gt[cell] + __popc(bs & lt);
#pragma unroll
            for (int offset = 0; offset < COMPRESS_RATIO; ++offset) {
                const unsigned int token = i * COMPRESS_RATIO + offset;
                if (token < sequence_length) token_indices[rank * COMPRESS_RATIO + offset] = (int)token;
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        int out = BLOCK_TOPK * COMPRESS_RATIO;
        const unsigned int tail_start = ((position + 1) / COMPRESS_RATIO) * COMPRESS_RATIO;
        for (unsigned int token = tail_start;
             token <= position && token < sequence_length && out < OUTPUT_WIDTH;
             ++token) {
            token_indices[out++] = (int)token;
        }
    }
}
}  // namespace

// Host-argument ABI of qwen4_qsa_select_expand_par. Launch grid [1,1,1], block [1024,1,1].
extern "C" __global__ void __launch_bounds__(1024) qwen4_qsa_select_expand_par_v3(
    const float* __restrict__ logits,
    int* __restrict__ token_indices,
    unsigned int visible_groups,
    unsigned int position,
    unsigned int sequence_length) {
    if (blockIdx.x != 0) return;
    qsa_select_expand_v3(logits, token_indices, visible_groups, position, sequence_length);
}

// Device-metadata ABI of qwen4_qsa_select_expand_device_meta_par (graph-safe).
extern "C" __global__ void __launch_bounds__(1024) qwen4_qsa_select_expand_device_meta_par_v3(
    const float* __restrict__ logits,
    int* __restrict__ token_indices,
    const unsigned int* __restrict__ position_ptr,
    const unsigned int* __restrict__ sequence_length_ptr) {
    if (blockIdx.x != 0) return;
    (void)position_ptr;
    const unsigned int sequence_length = *sequence_length_ptr;
    unsigned int position;
    if (!qwen4_qsa_positions::physical_query(sequence_length, position)) {
        for (int i = threadIdx.x; i < OUTPUT_WIDTH; i += SELECT_THREADS) token_indices[i] = -1;
        return;
    }
    qsa_select_expand_v3(
        logits, token_indices, sequence_length / COMPRESS_RATIO, position, sequence_length);
}

// ---------------------------------------------------------------------------
// qwen4_qsa_sparse_attention_bf16_partial_v2 (fn-decode, ATLAS_QWEN4_QSA_ATTN_V2=1):
// byte-identical to qwen4_qsa_sparse_attention_bf16_partial (same ABI, same
// grid [heads, splits], block = head_dim = 256). The incumbent walks the
// split's ~257 tokens one at a time with 10 block barriers and three dependent
// global loads per token. v2 keeps every arithmetic step:
//   * token list: this split's ranks in order, stopping at the first negative
//     token (break) and skipping unmapped pages (continue);
//   * score: one warp per token; lane L holds d = L + 32 j (j < 8), so the
//     incumbent's shared-memory tree (strides 128..1) is the same pair sums:
//     128/64/32 inside the lane, 16..1 via shfl_down;
//   * online softmax scalars: thread 0, sequential, same expressions;
//   * accumulation: thread d, acc = acc * alpha + beta * v in token order,
//     with the V loads independent of acc (in flight together).
// ---------------------------------------------------------------------------
namespace {
constexpr int QSA_V2_MAX_TOKENS_S8 = (OUTPUT_WIDTH + 7) / 8 + 8; // 265 >= ceil(2051 / 8) for splits >= 8
}

template <int MAX_TOKENS = QSA_V2_MAX_TOKENS_S8, bool SKIP_NEGATIVE = false>
__device__ __forceinline__ void qwen4_qsa_sparse_attention_bf16_partial_v2_body(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_elems,
    unsigned long long v_block_stride_elems,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits,
    int index_width = OUTPUT_WIDTH) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split = blockIdx.y;
    const unsigned int d = threadIdx.x;
    const unsigned int lane = d & 31u;
    const unsigned int warp = d >> 5;
    if (q_head >= num_query_heads || head_dim != 256u || blockDim.x != 256u) return;
    const unsigned int group_size = num_query_heads / num_kv_heads;
    const unsigned int kv_head = q_head / group_size;

    __shared__ float s_q[256];
    __shared__ unsigned int s_phys[MAX_TOKENS];
    __shared__ unsigned short s_tip[MAX_TOKENS];
    __shared__ float s_score[MAX_TOKENS];
    __shared__ float s_beta[MAX_TOKENS];
    float* const s_alpha = s_score;
    __shared__ unsigned int s_warp_end[8], s_warp_keep[8], s_warp_off[8];
    __shared__ int s_n, s_done;
    __shared__ float s_wmax[8];
    __shared__ float s_row_max, s_row_sum;

    s_q[d] = __bfloat162float(__ldg(query + (unsigned long long)q_head * head_dim + d));
    if (d == 0) { s_n = 0; s_done = 0; }
    __syncthreads();

    const bool bs_pow2 = block_size != 0u && (block_size & (block_size - 1u)) == 0u;
    const unsigned int bs_shift = bs_pow2 ? static_cast<unsigned int>(__ffs(block_size) - 1) : 0u;
    const unsigned int bs_mask = block_size - 1u;
    const unsigned long long width_u64 =
        index_width > 0 ? static_cast<unsigned long long>(index_width) : 0ull;

    // Token list: rounds of 8 warps x 32 ranks, rank order, same break/continue rules.
    for (unsigned int round = 0;; ++round) {
        const unsigned int idx = round * 256u + warp * 32u + lane;   // walk position
        const unsigned long long rank = (unsigned long long)split + (unsigned long long)idx * num_splits;
        const bool in_range = rank < width_u64;
        const int tok = in_range ? __ldg(token_indices + rank) : -1;
        const unsigned int end_mask =
            __ballot_sync(0xffffffffu, !in_range || (!SKIP_NEGATIVE && tok < 0));
        const unsigned int live = end_mask ? (__ffs(end_mask) - 1) : 32u;
        bool keep = false;
        unsigned int phys_page = 0u;
        unsigned short tip = 0u;
        if (lane < live && (!SKIP_NEGATIVE || tok >= 0)) {
            const unsigned int utok = static_cast<unsigned int>(tok);
            const unsigned int logical_page = bs_pow2 ? (utok >> bs_shift) : (utok / block_size);
            const unsigned int token_in_page = bs_pow2 ? (utok & bs_mask) : (utok % block_size);
            const int physical_page = __ldg(block_table + logical_page);
            if (physical_page >= 0) {
                keep = true;
                phys_page = static_cast<unsigned int>(physical_page);
                tip = static_cast<unsigned short>(token_in_page);
            }
        }
        const unsigned int keep_mask = __ballot_sync(0xffffffffu, keep);
        if (lane == 0) { s_warp_end[warp] = live; s_warp_keep[warp] = __popc(keep_mask); }
        __syncthreads();
        if (d == 0) {
            int off = s_n;
            bool ended = false;
            for (int w = 0; w < 8; ++w) {
                s_warp_off[w] = ended ? 0xffffffffu : (unsigned int)off;   // warps after the end contribute nothing
                if (!ended) {
                    off += (int)s_warp_keep[w];
                    if (s_warp_end[w] < 32u) ended = true;
                }
            }
            s_n = off;
            s_done = ended ? 1 : 0;
        }
        __syncthreads();
        const unsigned int base_off = s_warp_off[warp];
        if (keep && base_off != 0xffffffffu) {
            const unsigned int slot = base_off + __popc(keep_mask & ((1u << lane) - 1u));
            if (slot < (unsigned int)MAX_TOKENS) {
                s_phys[slot] = phys_page;
                s_tip[slot] = tip;
            }
        }
        const int done = s_done;
        __syncthreads();
        if (done) break;
    }
    if (s_n > MAX_TOKENS) {
        asm volatile("trap;");
        return;
    }
    const int n = s_n;

    // Scores: one warp per token (4 tokens' loads in flight), the incumbent's tree order exactly.
    for (int t0 = (int)warp * 4; t0 < n; t0 += 32) {
        float p[4][8];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int t = t0 + u;
            const unsigned long long kb =
                t < n ? ((unsigned long long)s_phys[t] * k_block_stride_elems +
                         ((unsigned long long)s_tip[t] * num_kv_heads + kv_head) * head_dim)
                      : 0ull;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned int dd = lane + 32u * j;
                p[u][j] = t < n ? s_q[dd] * __bfloat162float(__ldg(k_cache + kb + dd)) : 0.0f;
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int j = 0; j < 4; ++j) p[u][j] += p[u][j + 4];   // stride 128
#pragma unroll
            for (int j = 0; j < 2; ++j) p[u][j] += p[u][j + 2];   // stride 64
            p[u][0] += p[u][1];                                    // stride 32
#pragma unroll
            for (int s = 16; s > 0; s >>= 1) {
                const float other = __shfl_down_sync(0xffffffffu, p[u][0], s);
                if (lane < (unsigned int)s) p[u][0] += other;      // strides 16..1
            }
            if (lane == 0 && t0 + u < n) s_score[t0 + u] = p[u][0] * softmax_scale;
        }
    }
    __syncthreads();

    // Online-softmax scalars. next_max_t = fmaxf(next_max_{t-1}, score_t) is a
    // prefix max (fmaxf is associative), so it is computed in parallel; alpha
    // and beta then depend only on (next_max_{t-1}, next_max_t, score_t).
    // Only row_sum stays a sequential chain (thread 0), as in the incumbent.
    float run = -CUDART_INF_F;  // prefix max carried across 256-token chunks
    for (int c0 = 0; c0 < n; c0 += 256) {
        const int t = c0 + (int)d;
        const float sc = t < n ? s_score[t] : -CUDART_INF_F;
        float m = sc;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const float o = __shfl_up_sync(0xffffffffu, m, off);
            if (lane >= (unsigned int)off) m = fmaxf(o, m);
        }
        if (lane == 31) s_wmax[warp] = m;
        __syncthreads();
        float pre = run;  // max over everything before this warp's slice
        for (unsigned int w = 0; w < warp; ++w) pre = fmaxf(pre, s_wmax[w]);
        const float next_max = fmaxf(pre, m);
        const float prev_lane = __shfl_up_sync(0xffffffffu, m, 1);
        const float prev_max = lane == 0 ? pre : fmaxf(pre, prev_lane);
        if (t < n) {
            s_alpha[t] = prev_max == -CUDART_INF_F ? 0.0f : __expf(prev_max - next_max);
            s_beta[t] = __expf(sc - next_max);
        }
        for (unsigned int w = 0; w < 8; ++w) run = fmaxf(run, s_wmax[w]);
        __syncthreads();
    }
    if (d == 0) {
        float row_sum = 0.0f;
        for (int t = 0; t < n; ++t) row_sum = row_sum * s_alpha[t] + s_beta[t];
        s_row_max = run;
        s_row_sum = row_sum;
    }
    __syncthreads();

    float acc = 0.0f;
    int t = 0;
    for (; t + 8 <= n; t += 8) {
        float v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const unsigned long long vb =
                (unsigned long long)s_phys[t + u] * v_block_stride_elems +
                ((unsigned long long)s_tip[t + u] * num_kv_heads + kv_head) * head_dim;
            v[u] = __bfloat162float(__ldg(v_cache + vb + d));
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) acc = acc * s_alpha[t + u] + s_beta[t + u] * v[u];
    }
    for (; t < n; ++t) {
        const unsigned long long vb =
            (unsigned long long)s_phys[t] * v_block_stride_elems +
            ((unsigned long long)s_tip[t] * num_kv_heads + kv_head) * head_dim;
        acc = acc * s_alpha[t] + s_beta[t] * __bfloat162float(__ldg(v_cache + vb + d));
    }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    const unsigned long long part = (unsigned long long)q_head * num_splits + split;
    partial_output[part * head_dim + d] = acc;
    if (d == 0) {
        partial_max[part] = s_row_max;
        partial_sum[part] = s_row_sum;
    }
}

extern "C" __global__ void __launch_bounds__(256, 5) qwen4_qsa_sparse_attention_bf16_partial_v2(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_elems,
    unsigned long long v_block_stride_elems,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits) {
    if (num_splits < 8u) return;
    qwen4_qsa_sparse_attention_bf16_partial_v2_body<QSA_V2_MAX_TOKENS_S8, false>(
        query,
        k_cache,
        v_cache,
        token_indices,
        block_table,
        partial_output,
        partial_max,
        partial_sum,
        k_block_stride_elems,
        v_block_stride_elems,
        block_size,
        num_query_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        num_splits,
        OUTPUT_WIDTH);
}

// Reused-selection v2 BF16 split-token sparse attention for MTP draft steps 1..K-1.
// `QSA_V2_MAX_TOKENS_S8 = 265 >= ceil((OUTPUT_WIDTH + 16) / 8) = 259`.
extern "C" __global__ void __launch_bounds__(256, 5) qwen4_qsa_sparse_attention_bf16_partial_v2_reused(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_elems,
    unsigned long long v_block_stride_elems,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits,
    int index_width) {
    if (num_splits < 8u) return;
    qwen4_qsa_sparse_attention_bf16_partial_v2_body<QSA_V2_MAX_TOKENS_S8, true>(
        query,
        k_cache,
        v_cache,
        token_indices,
        block_table,
        partial_output,
        partial_max,
        partial_sum,
        k_block_stride_elems,
        v_block_stride_elems,
        block_size,
        num_query_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        num_splits,
        index_width);
}

extern "C" __global__ void __launch_bounds__(256, 4) qwen4_qsa_sparse_attention_bf16_partial_rows_v2(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_elems,
    unsigned long long v_block_stride_elems,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits,
    unsigned int query_row_stride,
    unsigned int block_table_row_stride) {
    if (num_splits < 8u) return;
    const unsigned long long row = blockIdx.z;
    const unsigned long long row_parts = (unsigned long long)row * num_query_heads * num_splits;
    qwen4_qsa_sparse_attention_bf16_partial_v2_body<QSA_V2_MAX_TOKENS_S8>(
        query + row * query_row_stride,
        k_cache,
        v_cache,
        token_indices + row * OUTPUT_WIDTH,
        block_table + row * block_table_row_stride,
        partial_output + row_parts * head_dim,
        partial_max + row_parts,
        partial_sum + row_parts,
        k_block_stride_elems,
        v_block_stride_elems,
        block_size,
        num_query_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        num_splits);
}

// ---------------------------------------------------------------------------
// qwen4_qsa_score_v2 / _device_meta_v2 (fn-decode, ATLAS_QWEN4_QSA_SCORE_V2=1):
// byte-identical to qwen4_qsa_score{,_device_meta}. Each warp scores
// QSA_SCORE_G groups (grid = ceil(groups / (8*G))), with every group's
// block-table and key loads issued before any reduction, and the 4x128
// query held in shared memory (s_q[4*128]). Per group the arithmetic is unchanged: lane-local
// sum over d = lane, lane+32, lane+64, lane+96 in order, warp_sum, then
// score += fmaxf(partial, 0) over heads 0..3, times 1/sqrt(128).
// ---------------------------------------------------------------------------
namespace {
constexpr int QSA_SCORE_G = 4;

__device__ __forceinline__ void qsa_score_groups(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ compressed_cache,
    const int* __restrict__ block_table,
    float* __restrict__ logits,
    unsigned int visible_groups,
    unsigned int block_size) {
    __shared__ float s_q[INDEX_HEADS * INDEX_DIM];
    for (int i = threadIdx.x; i < INDEX_HEADS * INDEX_DIM; i += 256) {
        s_q[i] = __bfloat162float(query[i]);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const unsigned int first = (blockIdx.x * 8u + (unsigned int)warp) * QSA_SCORE_G;
    if (first >= visible_groups) return;
    const unsigned int groups_per_page = block_size / COMPRESS_RATIO;
    int page[QSA_SCORE_G];
    if ((groups_per_page & (QSA_SCORE_G - 1u)) == 0u && groups_per_page > 0u) {
        const int p0 = __ldg(block_table + first / groups_per_page);
#pragma unroll
        for (int g = 0; g < QSA_SCORE_G; ++g) {
            page[g] = (first + g < visible_groups) ? p0 : -1;
        }
    } else {
#pragma unroll
        for (int g = 0; g < QSA_SCORE_G; ++g) {
            const unsigned int group = first + g;
            page[g] = group < visible_groups ? __ldg(block_table + group / groups_per_page) : -1;
        }
    }
    float k[QSA_SCORE_G][INDEX_DIM / 32];
#pragma unroll
    for (int g = 0; g < QSA_SCORE_G; ++g) {
        const unsigned int group = first + g;
        const unsigned long long key_base =
            ((unsigned long long)(page[g] < 0 ? 0 : page[g]) * groups_per_page + group % groups_per_page) * INDEX_DIM;
#pragma unroll
        for (int j = 0; j < INDEX_DIM / 32; ++j)
            k[g][j] = (group < visible_groups && page[g] >= 0)
                          ? __bfloat162float(__ldg(compressed_cache + key_base + lane + 32 * j)) : 0.0f;
    }
#pragma unroll
    for (int g = 0; g < QSA_SCORE_G; ++g) {
        const unsigned int group = first + g;
        if (group >= visible_groups) break;
        if (page[g] < 0) {
            if (lane == 0) logits[group] = -CUDART_INF_F;
            continue;
        }
        float score = 0.0f;
#pragma unroll
        for (int h = 0; h < INDEX_HEADS; ++h) {
            float partial = 0.0f;
#pragma unroll
            for (int j = 0; j < INDEX_DIM / 32; ++j)
                partial += s_q[h * INDEX_DIM + lane + 32 * j] * k[g][j];
            partial = warp_sum(partial);
            if (lane == 0) score += fmaxf(partial, 0.0f);
        }
        if (lane == 0) logits[group] = score * 0.08838834764831845f;
    }
}
}  // namespace

// grid [ceil(visible_groups / (8*QSA_SCORE_G))], block 256.
extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_score_v2(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ compressed_cache,
    const int* __restrict__ block_table,
    float* __restrict__ logits,
    unsigned int visible_groups,
    unsigned int block_size) {
    qsa_score_groups(query, compressed_cache, block_table, logits, visible_groups, block_size);
}

// Same, visible groups from device metadata (graph-safe); grid sized by the caller.
extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_score_device_meta_v2(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ compressed_cache,
    const int* __restrict__ block_table,
    float* __restrict__ logits,
    const unsigned int* __restrict__ sequence_length_ptr,
    unsigned int block_size) {
    if (block_size < COMPRESS_RATIO) return;
    qsa_score_groups(query, compressed_cache, block_table, logits, *sequence_length_ptr / COMPRESS_RATIO, block_size);
}

// Multi-row NVFP4 QSA sparse attention split-K partial (exact 0-ULP twin of launching
// qwen4_qsa_sparse_attention_nvfp4_partial once per row):
// grid [num_query_heads, num_splits, num_rows], block [256, 1, 1].
extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_sparse_attention_nvfp4_partial_rows(
    const __nv_bfloat16* __restrict__ query,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_bytes,
    unsigned long long v_block_stride_bytes,
    unsigned long long data_section_bytes,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits,
    unsigned int query_row_stride,
    unsigned int block_table_row_stride) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split = blockIdx.y;
    const unsigned int row = blockIdx.z;
    const unsigned int d = threadIdx.x;
    if (q_head >= num_query_heads || head_dim == 0u || head_dim > 256u || blockDim.x != head_dim || num_kv_heads == 0u || block_size == 0u) return;
    const unsigned int group_size = num_query_heads / num_kv_heads;
    const unsigned int kv_head = q_head / (group_size > 0u ? group_size : 1u);
    const __nv_bfloat16* row_q = query + (unsigned long long)row * query_row_stride;
    const int* row_bt = block_table + (unsigned long long)row * block_table_row_stride;
    const int* row_indices = token_indices + (unsigned long long)row * OUTPUT_WIDTH;
    const float qv = __bfloat162float(row_q[(unsigned long long)q_head * head_dim + d]);
    float acc = 0.0f;
    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    __shared__ float reduce[256];
    __shared__ float alpha_shared;
    __shared__ float beta_shared;

    for (int rank = (int)split; rank < OUTPUT_WIDTH; rank += (int)num_splits) {
        const int logical_token = row_indices[rank];
        if (logical_token < 0) break;
        const unsigned int logical_page = (unsigned int)logical_token / block_size;
        const unsigned int token_in_page = (unsigned int)logical_token % block_size;
        const int physical_page = row_bt[logical_page];
        if (physical_page < 0) continue;
        const unsigned long long element =
            ((unsigned long long)token_in_page * num_kv_heads + kv_head) * head_dim + d;
        const unsigned char* k_block =
            k_cache + (unsigned long long)physical_page * k_block_stride_bytes;
        reduce[d] = qv * load_nvfp4(k_block, element, data_section_bytes);
        __syncthreads();
        for (unsigned int stride = head_dim >> 1; stride > 0; stride >>= 1) {
            if (d < stride) reduce[d] += reduce[d + stride];
            __syncthreads();
        }
        if (d == 0) {
            const float score = reduce[0] * softmax_scale;
            const float next_max = fmaxf(row_max, score);
            const float alpha = row_max == -CUDART_INF_F ? 0.0f : __expf(row_max - next_max);
            const float beta = __expf(score - next_max);
            row_sum = row_sum * alpha + beta;
            row_max = next_max;
            alpha_shared = alpha;
            beta_shared = beta;
        }
        __syncthreads();
        const unsigned char* v_block =
            v_cache + (unsigned long long)physical_page * v_block_stride_bytes;
        acc = acc * alpha_shared + beta_shared * load_nvfp4(v_block, element, data_section_bytes);
        __syncthreads();
    }
    const unsigned long long part =
        ((unsigned long long)row * num_query_heads + q_head) * num_splits + split;
    partial_output[part * head_dim + d] = acc;
    if (d == 0) {
        partial_max[part] = row_max;
        partial_sum[part] = row_sum;
    }
}

// Multi-row NVFP4 QSA sparse attention v2 partial (8 warps/CTA warp-level reduction,
// matching qwen4_qsa_sparse_attention_bf16_partial_rows_v2):
// grid [num_query_heads, num_splits, num_rows], block [256, 1, 1].
namespace {
__device__ __forceinline__ void qwen4_qsa_sparse_attention_nvfp4_partial_rows_v2_body(
    const __nv_bfloat16* __restrict__ query,
    const unsigned char* __restrict__ k_pool,
    const unsigned char* __restrict__ v_pool,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_bytes,
    unsigned long long v_block_stride_bytes,
    unsigned long long data_section_bytes,
    unsigned int page_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits,
    unsigned int query_row_stride,
    unsigned int block_table_row_stride,
    unsigned int row) {
    constexpr int SPA_WARPS = 8;
    const unsigned int q_head = blockIdx.x;
    const unsigned int split = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    if (q_head >= num_query_heads || split >= num_splits || head_dim > 256u ||
        num_query_heads == 0u || num_kv_heads == 0u || page_size == 0u) {
        return;
    }
    const unsigned int group_size = num_query_heads / num_kv_heads;
    const unsigned int kv_head = q_head / (group_size > 0u ? group_size : 1u);
    const unsigned int part = (row * num_query_heads + q_head) * num_splits + split;
    float* part_out = partial_output + (unsigned long long)part * head_dim;
    const __nv_bfloat16* q_ptr =
        query + (unsigned long long)row * query_row_stride + (unsigned long long)q_head * head_dim;
    const int* row_bt = block_table + (unsigned long long)row * block_table_row_stride;
    const int* row_indices = token_indices + (unsigned long long)row * OUTPUT_WIDTH;

    float q_reg[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int d = lane + (unsigned int)(i * 32);
        q_reg[i] = d < head_dim ? __bfloat162float(q_ptr[d]) : 0.0f;
    }

    float running_max = -CUDART_INF_F;
    float running_sum = 0.0f;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int slot = (int)(split + warp * num_splits); slot < OUTPUT_WIDTH;
         slot += (int)(num_splits * (unsigned int)SPA_WARPS)) {
        const int token_idx = row_indices[slot];
        if (token_idx < 0) continue;
        const int page_id = row_bt[(unsigned int)token_idx / page_size];
        if (page_id < 0) continue;
        const unsigned int offset = (unsigned int)token_idx % page_size;
        const unsigned long long elem_base =
            (unsigned long long)(offset * num_kv_heads + kv_head) * head_dim;
        const unsigned char* k_block =
            k_pool + (unsigned long long)page_id * k_block_stride_bytes;
        const unsigned char* v_block =
            v_pool + (unsigned long long)page_id * v_block_stride_bytes;

        float score = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned int d = lane + (unsigned int)(i * 32);
            if (d < head_dim) {
                score += q_reg[i] * load_nvfp4(k_block, elem_base + d, data_section_bytes);
            }
        }
        score = warp_sum(score);
        const float total_score = __shfl_sync(0xffffffffu, score, 0) * softmax_scale;

        const float next_max = fmaxf(running_max, total_score);
        const float scale_old = running_sum == 0.0f ? 0.0f : expf(running_max - next_max);
        const float weight = expf(total_score - next_max);
        running_sum = running_sum * scale_old + weight;
        running_max = next_max;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned int d = lane + (unsigned int)(i * 32);
            if (d < head_dim) {
                acc[i] = acc[i] * scale_old +
                         weight * load_nvfp4(v_block, elem_base + d, data_section_bytes);
            }
        }
    }

    __shared__ float s_acc[SPA_WARPS][256];
    __shared__ float s_max[SPA_WARPS];
    __shared__ float s_sum[SPA_WARPS];

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        s_acc[warp][lane + (unsigned int)(i * 32)] = acc[i];
    }
    if (lane == 0u) {
        s_max[warp] = running_max;
        s_sum[warp] = running_sum;
    }
    __syncthreads();

    const unsigned int c = tid;
    float m = -CUDART_INF_F;
#pragma unroll
    for (int w = 0; w < SPA_WARPS; ++w) {
        if (s_sum[w] > 0.0f) {
            m = fmaxf(m, s_max[w]);
        }
    }
    float total_sum = 0.0f;
    float total_acc = 0.0f;
    if (m > -CUDART_INF_F) {
#pragma unroll
        for (int w = 0; w < SPA_WARPS; ++w) {
            const float w_sum = s_sum[w];
            if (w_sum > 0.0f) {
                const float scale = expf(s_max[w] - m);
                total_sum += w_sum * scale;
                total_acc += s_acc[w][c] * scale;
            }
        }
    }
    if (c < head_dim) {
        part_out[c] = total_acc;
    }
    if (c == 0u) {
        partial_max[part] = m;
        partial_sum[part] = total_sum;
    }
}
}  // namespace

extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_sparse_attention_nvfp4_partial_rows_v2(
    const __nv_bfloat16* __restrict__ query,
    const unsigned char* __restrict__ k_pool,
    const unsigned char* __restrict__ v_pool,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_bytes,
    unsigned long long v_block_stride_bytes,
    unsigned long long data_section_bytes,
    unsigned int page_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits,
    unsigned int query_row_stride,
    unsigned int block_table_row_stride) {
    qwen4_qsa_sparse_attention_nvfp4_partial_rows_v2_body(
        query, k_pool, v_pool, token_indices, block_table,
        partial_output, partial_max, partial_sum,
        k_block_stride_bytes, v_block_stride_bytes, data_section_bytes,
        page_size, num_query_heads, num_kv_heads, head_dim, softmax_scale,
        num_splits, query_row_stride, block_table_row_stride, blockIdx.z);
}

// ===========================================================================
// Cross-Sequence Batched QSA Verify Kernels (C2-C16, 0-ULP, 0 B spill)
// ===========================================================================
namespace {
enum : unsigned int {
    QSA_DESC_FLAG_QSA_REQUIRED = 1u << 0,
    QSA_DESC_FLAG_SPARSE = 1u << 1,
    QSA_DESC_FLAG_JOURNAL = 1u << 2,
};

struct alignas(16) QsaBatchSeqDesc {
    unsigned int row_base;
    unsigned int seq_len;
    unsigned int k;
    unsigned int journal_lane;
    unsigned int flags;
    unsigned int logits_offset;
    unsigned int logits_stride;
    unsigned int score_block_end;
};
static_assert(sizeof(QsaBatchSeqDesc) == 32, "QsaBatchSeqDesc must be 32 bytes");

__device__ __forceinline__ bool qsa_row_mask_active(
    unsigned int row,
    unsigned long long mask_lo,
    unsigned long long mask_hi) {
    return (row < 64u)
        ? (((mask_lo >> row) & 1ull) != 0ull)
        : ((row < 128u) && (((mask_hi >> (row - 64u)) & 1ull) != 0ull));
}
}  // namespace

// Packed 1D multi-sequence QSA scoring:
// Grid:  [total_score_blocks, 1, 1]
// Block: [256, 1, 1]
// Binary-searches `seq_descs[0..num_seqs].score_block_end` to map each CTA to
// its sparse sequence and local group block, then evaluates all `desc.k` rows
// against the shared compressed keys with 0-ULP parity to `qwen4_qsa_score_rows_v2`.
extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_score_multi_seq_rows_v2(
    const __nv_bfloat16* __restrict__ projected_qk,
    const __nv_bfloat16* __restrict__ compressed_cache,
    const int* __restrict__ block_tables,
    float* __restrict__ batch_logits,
    const QsaBatchSeqDesc* __restrict__ seq_descs,
    unsigned int num_seqs,
    unsigned int max_blocks_per_seq,
    unsigned int block_size) {
    if (num_seqs == 0u || block_size < (unsigned int)COMPRESS_RATIO) return;

    unsigned int lo = 0u;
    unsigned int hi = num_seqs - 1u;
    while (lo < hi) {
        const unsigned int mid = (lo + hi) >> 1;
        if (blockIdx.x < seq_descs[mid].score_block_end) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }
    const unsigned int seq_idx = lo;
    const QsaBatchSeqDesc desc = seq_descs[seq_idx];
    const unsigned int prev_end = (seq_idx == 0u) ? 0u : seq_descs[seq_idx - 1u].score_block_end;
    if (blockIdx.x >= desc.score_block_end || (desc.flags & QSA_DESC_FLAG_SPARSE) == 0u) return;

    const unsigned int local_block = blockIdx.x - prev_end;
    const unsigned int num_rows = (desc.k <= (unsigned int)VERIFY_MAX_ROWS)
        ? desc.k : (unsigned int)VERIFY_MAX_ROWS;
    if (num_rows == 0u) return;

    __shared__ __nv_bfloat16 s_q[VERIFY_MAX_ROWS][INDEX_HEADS * INDEX_DIM];
    const unsigned int proj_width = (unsigned int)((INDEX_HEADS + 1) * INDEX_DIM);
    const __nv_bfloat16* seq_qk =
        projected_qk + (unsigned long long)desc.row_base * proj_width;
    for (unsigned int r = 0u; r < num_rows; ++r) {
        const __nv_bfloat16* q_src = seq_qk + (unsigned long long)r * proj_width;
        for (unsigned int i = threadIdx.x; i < (unsigned int)(INDEX_HEADS * INDEX_DIM); i += 256u) {
            s_q[r][i] = q_src[i];
        }
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const unsigned int first = (local_block * 8u + (unsigned int)warp) * QSA_SCORE_G;
    const unsigned int max_vis = (desc.seq_len + num_rows) / (unsigned int)COMPRESS_RATIO;
    if (first >= max_vis) return;

    const int* seq_bt =
        block_tables + (unsigned long long)desc.row_base * max_blocks_per_seq;
    const unsigned int groups_per_page = block_size / (unsigned int)COMPRESS_RATIO;
    int page[QSA_SCORE_G];
    if ((groups_per_page & (QSA_SCORE_G - 1u)) == 0u && groups_per_page > 0u) {
        const int p0 = __ldg(seq_bt + first / groups_per_page);
#pragma unroll
        for (int g = 0; g < QSA_SCORE_G; ++g) {
            page[g] = (first + (unsigned int)g < max_vis) ? p0 : -1;
        }
    } else {
#pragma unroll
        for (int g = 0; g < QSA_SCORE_G; ++g) {
            const unsigned int group = first + (unsigned int)g;
            page[g] = (group < max_vis) ? __ldg(seq_bt + group / groups_per_page) : -1;
        }
    }

    float k_reg[QSA_SCORE_G][INDEX_DIM / 32];
#pragma unroll
    for (int g = 0; g < QSA_SCORE_G; ++g) {
        const unsigned int group = first + (unsigned int)g;
        const unsigned long long key_base =
            ((unsigned long long)(page[g] < 0 ? 0 : page[g]) * groups_per_page
             + group % groups_per_page) * INDEX_DIM;
#pragma unroll
        for (int j = 0; j < INDEX_DIM / 32; ++j) {
            k_reg[g][j] = (group < max_vis && page[g] >= 0)
                ? __bfloat162float(__ldg(compressed_cache + key_base + lane + 32 * j))
                : 0.0f;
        }
    }

    for (unsigned int r = 0u; r < num_rows; ++r) {
        const unsigned int vis_r = (desc.seq_len + r + 1u) / (unsigned int)COMPRESS_RATIO;
        if (first < vis_r) {
            const __nv_bfloat16* q_row = s_q[r];
            float score[QSA_SCORE_G] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int h = 0; h < INDEX_HEADS; ++h) {
                float qh[INDEX_DIM / 32];
#pragma unroll
                for (int j = 0; j < INDEX_DIM / 32; ++j) {
                    qh[j] = __bfloat162float(q_row[h * INDEX_DIM + lane + 32 * j]);
                }
#pragma unroll
                for (int g = 0; g < QSA_SCORE_G; ++g) {
                    float partial = 0.0f;
#pragma unroll
                    for (int j = 0; j < INDEX_DIM / 32; ++j) {
                        partial += qh[j] * k_reg[g][j];
                    }
                    partial = warp_sum(partial);
                    if (lane == 0) score[g] += fmaxf(partial, 0.0f);
                }
            }
            if (lane == 0) {
                float* row_logits = batch_logits
                    + (unsigned long long)desc.logits_offset
                    + (unsigned long long)r * desc.logits_stride;
#pragma unroll
                for (int g = 0; g < QSA_SCORE_G; ++g) {
                    const unsigned int group = first + (unsigned int)g;
                    if (group < vis_r) {
                        row_logits[group] = (page[g] < 0)
                            ? -CUDART_INF_F
                            : score[g] * 0.08838834764831845f;
                    }
                }
            }
        }
    }
}

// Multi-sequence top-512 select & expand (v3):
// Grid:  [max_k, num_seqs, 1]
// Block: [1024, 1, 1]
extern "C" __global__ void __launch_bounds__(1024) qwen4_qsa_select_expand_multi_seq_rows_v3(
    const float* __restrict__ batch_logits,
    int* __restrict__ batch_token_indices,
    const QsaBatchSeqDesc* __restrict__ seq_descs,
    unsigned int num_seqs) {
    const unsigned int local_row = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    if (seq_idx >= num_seqs) return;
    const QsaBatchSeqDesc desc = seq_descs[seq_idx];
    if ((desc.flags & QSA_DESC_FLAG_SPARSE) == 0u || local_row >= desc.k) return;
    const unsigned int global_row = desc.row_base + local_row;
    const float* row_logits = batch_logits
        + (unsigned long long)desc.logits_offset
        + (unsigned long long)local_row * desc.logits_stride;
    int* row_indices = batch_token_indices + (unsigned long long)global_row * OUTPUT_WIDTH;
    const unsigned int sequence_length = desc.seq_len + local_row + 1u;
    const unsigned int position = desc.seq_len + local_row;
    const unsigned int visible_groups = sequence_length / (unsigned int)COMPRESS_RATIO;
    qsa_select_expand_v3(row_logits, row_indices, visible_groups, position, sequence_length);
}

// Multi-sequence top-512 select & expand (parallel radix fallback):
// Grid:  [max_k, num_seqs, 1]
// Block: [1024, 1, 1]
extern "C" __global__ void __launch_bounds__(1024) qwen4_qsa_select_expand_multi_seq_rows_par(
    const float* __restrict__ batch_logits,
    int* __restrict__ batch_token_indices,
    const QsaBatchSeqDesc* __restrict__ seq_descs,
    unsigned int num_seqs) {
    const unsigned int local_row = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    if (seq_idx >= num_seqs) return;
    const QsaBatchSeqDesc desc = seq_descs[seq_idx];
    if ((desc.flags & QSA_DESC_FLAG_SPARSE) == 0u || local_row >= desc.k) return;
    const unsigned int global_row = desc.row_base + local_row;
    const float* row_logits = batch_logits
        + (unsigned long long)desc.logits_offset
        + (unsigned long long)local_row * desc.logits_stride;
    int* row_indices = batch_token_indices + (unsigned long long)global_row * OUTPUT_WIDTH;
    const unsigned int sequence_length = desc.seq_len + local_row + 1u;
    const unsigned int position = desc.seq_len + local_row;
    const unsigned int visible_groups = sequence_length / (unsigned int)COMPRESS_RATIO;
    qsa_select_expand_parallel(row_logits, row_indices, visible_groups, position, sequence_length);
}

// Masked multi-row BF16 QSA sparse attention v2 partial (for mixed dense+sparse frames):
// Grid:  [num_query_heads, num_splits, total_rows]
// Block: [256, 1, 1]
extern "C" __global__ void __launch_bounds__(256, 4)
qwen4_qsa_sparse_attention_bf16_partial_rows_v2_masked(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_elems,
    unsigned long long v_block_stride_elems,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits,
    unsigned int query_row_stride,
    unsigned int block_table_row_stride,
    unsigned long long row_mask_lo,
    unsigned long long row_mask_hi) {
    const unsigned int row = blockIdx.z;
    if (num_splits < 8u || !qsa_row_mask_active(row, row_mask_lo, row_mask_hi)) return;
    const unsigned long long row_u64 = (unsigned long long)row;
    const unsigned long long row_parts = row_u64 * num_query_heads * num_splits;
    qwen4_qsa_sparse_attention_bf16_partial_v2_body<QSA_V2_MAX_TOKENS_S8>(
        query + row_u64 * query_row_stride,
        k_cache,
        v_cache,
        token_indices + row_u64 * OUTPUT_WIDTH,
        block_table + row_u64 * block_table_row_stride,
        partial_output + row_parts * head_dim,
        partial_max + row_parts,
        partial_sum + row_parts,
        k_block_stride_elems,
        v_block_stride_elems,
        block_size,
        num_query_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        num_splits);
}

// Masked multi-row NVFP4 QSA sparse attention v2 partial (for mixed dense+sparse frames):
// Grid:  [num_query_heads, num_splits, total_rows]
// Block: [256, 1, 1]
extern "C" __global__ void __launch_bounds__(256)
qwen4_qsa_sparse_attention_nvfp4_partial_rows_v2_masked(
    const __nv_bfloat16* __restrict__ query,
    const unsigned char* __restrict__ k_pool,
    const unsigned char* __restrict__ v_pool,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_bytes,
    unsigned long long v_block_stride_bytes,
    unsigned long long data_section_bytes,
    unsigned int page_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits,
    unsigned int query_row_stride,
    unsigned int block_table_row_stride,
    unsigned long long row_mask_lo,
    unsigned long long row_mask_hi) {
    const unsigned int row = blockIdx.z;
    if (!qsa_row_mask_active(row, row_mask_lo, row_mask_hi)) return;
    qwen4_qsa_sparse_attention_nvfp4_partial_rows_v2_body(
        query, k_pool, v_pool, token_indices, block_table,
        partial_output, partial_max, partial_sum,
        k_block_stride_bytes, v_block_stride_bytes, data_section_bytes,
        page_size, num_query_heads, num_kv_heads, head_dim, softmax_scale,
        num_splits, query_row_stride, block_table_row_stride, row);
}

// Masked multi-row BF16 QSA sparse attention gated reduction:
// Grid:  [num_query_heads, total_rows, 1]
// Block: [256, 1, 1]
extern "C" __global__ void __launch_bounds__(256)
qwen4_qsa_sparse_attention_bf16_reduce_gated_rows_masked(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_query_heads,
    unsigned int head_dim,
    unsigned int num_splits,
    unsigned int gate_stride,
    unsigned long long row_mask_lo,
    unsigned long long row_mask_hi) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int d = threadIdx.x;
    if (q_head >= num_query_heads || d >= head_dim ||
        !qsa_row_mask_active(row, row_mask_lo, row_mask_hi)) {
        return;
    }
    const unsigned long long base =
        ((unsigned long long)row * num_query_heads + q_head) * num_splits;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    float global_max = -CUDART_INF_F;
    for (unsigned int split = 0; split < num_splits; ++split)
        global_max = fmaxf(global_max, partial_max[base + split]);
    float denom = 0.0f;
    float acc = 0.0f;
    if (global_max > -CUDART_INF_F) {
        for (unsigned int split = 0; split < num_splits; ++split) {
            const float scale = __expf(partial_max[base + split] - global_max);
            denom += partial_sum[base + split] * scale;
            acc += partial_output[(base + split) * head_dim + d] * scale;
        }
    }
    const unsigned long long head_elem = (unsigned long long)q_head * head_dim + d;
    const __nv_bfloat16 raw_bf16 =
        __float2bfloat16_rn(denom > 0.0f ? acc / denom : 0.0f);
    const float x = __bfloat162float(raw_bf16);
    const float g = __bfloat162float(gate[(unsigned long long)row * gate_stride + head_elem]);
    const float sigmoid_g = 1.0f / (1.0f + expf(-g));
    output[((unsigned long long)row * num_query_heads + q_head) * head_dim + d] =
        __float2bfloat16(x * sigmoid_g);
}

// Masked multi-row BF16 QSA sparse attention ungated reduction:
// Grid:  [num_query_heads, total_rows, 1]
// Block: [256, 1, 1]
extern "C" __global__ void __launch_bounds__(256)
qwen4_qsa_sparse_attention_bf16_reduce_rows_masked(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_query_heads,
    unsigned int head_dim,
    unsigned int num_splits,
    unsigned long long row_mask_lo,
    unsigned long long row_mask_hi) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int d = threadIdx.x;
    if (q_head >= num_query_heads || d >= head_dim ||
        !qsa_row_mask_active(row, row_mask_lo, row_mask_hi)) {
        return;
    }
    const unsigned long long base =
        ((unsigned long long)row * num_query_heads + q_head) * num_splits;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    float global_max = -CUDART_INF_F;
    for (unsigned int split = 0; split < num_splits; ++split)
        global_max = fmaxf(global_max, partial_max[base + split]);
    float denom = 0.0f;
    float acc = 0.0f;
    if (global_max > -CUDART_INF_F) {
        for (unsigned int split = 0; split < num_splits; ++split) {
            const float scale = __expf(partial_max[base + split] - global_max);
            denom += partial_sum[base + split] * scale;
            acc += partial_output[(base + split) * head_dim + d] * scale;
        }
    }
    output[((unsigned long long)row * num_query_heads + q_head) * head_dim + d] =
        __float2bfloat16_rn(denom > 0.0f ? acc / denom : 0.0f);
}

// Multi-sequence QSA verify prefix commit (reverse-order undo restoration):
// Grid:  [num_seqs, 1, 1]
// Block: [128, 1, 1]
extern "C" __global__ void __launch_bounds__(128) qwen4_qsa_commit_verify_prefix_multi_seq(
    __nv_bfloat16* __restrict__ raw_ring,
    __nv_bfloat16* __restrict__ compressed_cache,
    const __nv_bfloat16* __restrict__ undo_raw_base,
    const __nv_bfloat16* __restrict__ undo_compressed_base,
    const unsigned long long* __restrict__ undo_raw_offsets_base,
    const unsigned long long* __restrict__ undo_compressed_offsets_base,
    unsigned int* __restrict__ valid_mask_base,
    const unsigned int* __restrict__ lanes,
    const unsigned int* __restrict__ accepted_counts,
    const unsigned int* __restrict__ row_counts,
    unsigned int num_seqs) {
    const unsigned int seq_idx = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (seq_idx >= num_seqs || blockDim.x != (unsigned int)INDEX_DIM) return;

    const unsigned long long lane = (unsigned long long)lanes[seq_idx];
    const unsigned int accepted = accepted_counts[seq_idx];
    const unsigned int rows = row_counts[seq_idx];

    unsigned int* valid_mask = valid_mask_base + lane;
    const unsigned int valid = *valid_mask;
    if (valid == 0u) return;
    if (rows < 2u || rows > (unsigned int)VERIFY_MAX_ROWS || accepted > rows ||
        (accepted != 0u && valid != ((1u << rows) - 1u))) {
        asm volatile("trap;");
        return;
    }

    const __nv_bfloat16* undo_raw =
        undo_raw_base + lane * VERIFY_MAX_ROWS * INDEX_DIM;
    const __nv_bfloat16* undo_compressed =
        undo_compressed_base + lane * VERIFY_MAX_ROWS * INDEX_DIM;
    const unsigned long long* undo_raw_offsets =
        undo_raw_offsets_base + lane * VERIFY_MAX_ROWS;
    const unsigned long long* undo_compressed_offsets =
        undo_compressed_offsets_base + lane * VERIFY_MAX_ROWS;

    for (int row = static_cast<int>(rows) - 1; row >= static_cast<int>(accepted); --row) {
        if ((valid & (1u << row)) == 0u) continue;
        const unsigned long long raw_offset = undo_raw_offsets[row];
        raw_ring[raw_offset + d] = undo_raw[(unsigned long long)row * INDEX_DIM + d];
        const unsigned long long compressed_offset = undo_compressed_offsets[row];
        if (compressed_offset != NO_COMPRESSED_OFFSET) {
            compressed_cache[compressed_offset + d] =
                undo_compressed[(unsigned long long)row * INDEX_DIM + d];
        }
        __syncthreads();
    }
    if (d == 0u) *valid_mask = 0u;
}

// Appends a speculative draft token position at fixed slot
// `base_width + (step_offset - 1)` and sets remaining slots
// `[base_width + step_offset .. base_width + max_drafts)` to `-1`.
// Runs as a single 1-block/1-thread kernel with zero host syncs so it is safe
// in both eager and CUDA-graph execution paths.
extern "C" __global__ void __launch_bounds__(1) qwen4_qsa_append_draft_pos(
    int* __restrict__ token_indices,
    const int* __restrict__ pos_dev,
    int pos_scalar,
    int use_dev_pos,
    int base_width,
    int step_offset,
    int max_drafts) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (base_width < 0 || max_drafts <= 0) return;
    if (step_offset < 0 || step_offset > max_drafts) return;
    if (step_offset >= 1) {
        const int pos = use_dev_pos ? *pos_dev : pos_scalar;
        token_indices[base_width + (step_offset - 1)] = pos;
    }
    for (int i = step_offset; i < max_drafts; ++i) {
        token_indices[base_width + i] = -1;
    }
}
