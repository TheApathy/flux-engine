// SPDX-License-Identifier: AGPL-3.0-only

#include <stdint.h>

// Per-SSM-layer device pointers for one physical sequence slot (64 B, 16 B aligned).
struct alignas(16) SsmLayerCommitPointers {
    uint64_t h_state;
    uint64_t h_ckpt;
    uint64_t conv_state;
    uint64_t conv_ckpt;
    uint64_t h_inter_base;
    uint64_t conv_inter_base;
    uint64_t retain_slot_base;
    uint64_t _pad;
};
static_assert(sizeof(SsmLayerCommitPointers) == 64, "SsmLayerCommitPointers must be 64 B");
static_assert(alignof(SsmLayerCommitPointers) == 16, "SsmLayerCommitPointers must be 16 B aligned");

// Per-sequence batched verify commit descriptor (16 B, 16 B aligned).
// flags bit 0: mirror_ckpt    - mirror H state into h_ckpt when h_ckpt != 0.
// flags bit 1: has_conv_inter - read conv state from conv_inter_base[accepted_idx]
//                               and write conv_state (and conv_ckpt if non-null);
//                               when 0, mirror live conv_state -> conv_ckpt only.
// flags bit 2: h_already_live - when 1, h_state already holds post-K state so
//                               only mirror h_state -> h_ckpt if mirror_ckpt=1;
//                               when 0, copy h_inter_base[accepted_idx] -> h_state
//                               (and h_ckpt if mirror_ckpt=1).
struct alignas(16) BatchCommitEntry {
    uint64_t ssm_table_dev;
    uint32_t accepted_idx;
    uint32_t flags;
};
static_assert(sizeof(BatchCommitEntry) == 16, "BatchCommitEntry must be 16 B");
static_assert(alignof(BatchCommitEntry) == 16, "BatchCommitEntry must be 16 B aligned");

static constexpr uint32_t BATCH_COMMIT_FLAG_MIRROR_CKPT    = 1u << 0;
static constexpr uint32_t BATCH_COMMIT_FLAG_HAS_CONV_INTER = 1u << 1;
static constexpr uint32_t BATCH_COMMIT_FLAG_H_ALREADY_LIVE = 1u << 2;

__device__ __forceinline__ void copy_f32_broadcast_pair(
    uint64_t src_addr,
    uint64_t dst_a_addr,
    uint64_t dst_b_addr,
    uint32_t count_f32s,
    bool write_a,
    bool write_b,
    uint32_t tid,
    uint32_t stride
) {
    if (count_f32s == 0u || src_addr == 0ull || (!write_a && !write_b)) {
        return;
    }
    const uint64_t align_mask =
        src_addr | (write_a ? dst_a_addr : 0ull) | (write_b ? dst_b_addr : 0ull);
    const uint32_t vec4_count = ((align_mask & 15ull) == 0ull) ? (count_f32s >> 2) : 0u;

    if (vec4_count > 0u) {
        const float4* __restrict__ src4 = reinterpret_cast<const float4*>(src_addr);
        float4* __restrict__ dst_a4 = reinterpret_cast<float4*>(dst_a_addr);
        float4* __restrict__ dst_b4 = reinterpret_cast<float4*>(dst_b_addr);
        if (write_a && write_b) {
            for (uint32_t idx = tid; idx < vec4_count; idx += stride) {
                const float4 v = src4[idx];
                dst_a4[idx] = v;
                dst_b4[idx] = v;
            }
        } else if (write_a) {
            for (uint32_t idx = tid; idx < vec4_count; idx += stride) {
                dst_a4[idx] = src4[idx];
            }
        } else {
            for (uint32_t idx = tid; idx < vec4_count; idx += stride) {
                dst_b4[idx] = src4[idx];
            }
        }
    }

    const uint32_t tail_start = vec4_count << 2;
    if (tail_start < count_f32s) {
        const float* __restrict__ src1 = reinterpret_cast<const float*>(src_addr);
        float* __restrict__ dst_a1 = reinterpret_cast<float*>(dst_a_addr);
        float* __restrict__ dst_b1 = reinterpret_cast<float*>(dst_b_addr);
        for (uint32_t idx = tail_start + tid; idx < count_f32s; idx += stride) {
            const float v = src1[idx];
            if (write_a) {
                dst_a1[idx] = v;
            }
            if (write_b) {
                dst_b1[idx] = v;
            }
        }
    }
}

// Batched verify commit across all (num_ssm_layers x batch_size) pairs in 1 launch.
// Grid:  [num_ssm_layers, batch_size, 1]
// Block: [256, 1, 1]
extern "C" __global__ void gdn_batch_commit_copy_f32(
    const BatchCommitEntry* __restrict__ entries,
    uint32_t num_ssm_layers,
    uint32_t h_state_f32s,
    uint32_t conv_state_f32s,
    uint64_t h_inter_stride_bytes,
    uint64_t conv_inter_stride_bytes
) {
    const uint32_t layer_idx = blockIdx.x;
    const uint32_t seq_idx = blockIdx.y;
    if (layer_idx >= num_ssm_layers) {
        return;
    }

    const BatchCommitEntry entry = entries[seq_idx];
    if (entry.ssm_table_dev == 0ull) {
        return;
    }

    const SsmLayerCommitPointers* table =
        reinterpret_cast<const SsmLayerCommitPointers*>(entry.ssm_table_dev);
    const SsmLayerCommitPointers ptrs = table[layer_idx];

    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;
    const bool mirror_ckpt = (entry.flags & BATCH_COMMIT_FLAG_MIRROR_CKPT) != 0u;
    const bool has_conv_inter = (entry.flags & BATCH_COMMIT_FLAG_HAS_CONV_INTER) != 0u;
    const bool h_already_live = (entry.flags & BATCH_COMMIT_FLAG_H_ALREADY_LIVE) != 0u;

    // 1. Commit H state: h_inter[accepted_idx] -> (h_state, h_ckpt) or h_state -> h_ckpt.
    if (ptrs.h_state != 0ull) {
        const uint64_t h_src_addr = h_already_live
            ? ptrs.h_state
            : (ptrs.h_inter_base != 0ull
                ? (ptrs.h_inter_base + static_cast<uint64_t>(entry.accepted_idx) * h_inter_stride_bytes)
                : 0ull);
        const bool write_h_live = !h_already_live && (h_src_addr != 0ull);
        const bool write_h_ckpt = mirror_ckpt && (ptrs.h_ckpt != 0ull) && (h_src_addr != 0ull);
        copy_f32_broadcast_pair(
            h_src_addr,
            ptrs.h_state,
            ptrs.h_ckpt,
            h_state_f32s,
            write_h_live,
            write_h_ckpt,
            tid,
            stride
        );
    }

    // 2. Commit conv state: conv_inter[accepted_idx] -> (conv_state, conv_ckpt) or conv_state -> conv_ckpt.
    if (ptrs.conv_state != 0ull) {
        const bool from_inter = has_conv_inter && (ptrs.conv_inter_base != 0ull);
        const uint64_t c_src_addr = from_inter
            ? (ptrs.conv_inter_base + static_cast<uint64_t>(entry.accepted_idx) * conv_inter_stride_bytes)
            : ptrs.conv_state;
        const bool write_c_live = from_inter;
        const bool write_c_ckpt = (ptrs.conv_ckpt != 0ull);
        copy_f32_broadcast_pair(
            c_src_addr,
            ptrs.conv_state,
            ptrs.conv_ckpt,
            conv_state_f32s,
            write_c_live,
            write_c_ckpt,
            tid,
            stride
        );
    }
}
