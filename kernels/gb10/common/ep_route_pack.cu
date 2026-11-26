// SPDX-License-Identifier: AGPL-3.0-only

// Compact GPU-resident selected-route packet pack/unpack kernels for multi-Spark EP.
//
// Wire layout for one token packet with `slots` in {6, 8, 10} (48B / 64B / 80B):
//   words [0 .. slots)         : expert_ids[0 .. slots] as u32 (unused slots = 0xFFFFFFFFu)
//   words [slots .. 2 * slots) : IEEE-754 weight bit patterns as u32 (unused slots = 0u)
//
// Matches `spark_comm::SelectedRoute` (slots = 6, 48 bytes) and `spark_comm::RoutePacketLayout`
// (`TopK6` = 6, `TopK8` = 8, `TopK10` = 10).
//
// Launch geometry:
//   - Pack  : 1D grid over `num_tokens * slots` items (e.g. block = 128, grid = ceil(N*slots/128))
//   - Unpack: 1D grid over `total_tokens * top_k` items (e.g. block = 128, grid = ceil(N*top_k/128))

extern "C" __global__ void ep_pack_selected_routes_kernel(
    const int* __restrict__ indices,
    const float* __restrict__ weights,
    unsigned int* __restrict__ out_packets_u32,
    unsigned int num_tokens,
    unsigned int top_k,
    unsigned int slots
) {
    if (num_tokens == 0u || top_k == 0u || top_k > slots) {
        return;
    }
    if (slots != 6u && slots != 8u && slots != 10u) {
        return;
    }

    const unsigned long long total_items =
        (unsigned long long)num_tokens * (unsigned long long)slots;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    const unsigned int packet_words = slots * 2u;

    for (unsigned long long idx =
             (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
             (unsigned long long)threadIdx.x;
         idx < total_items;
         idx += stride) {
        const unsigned long long token = idx / (unsigned long long)slots;
        const unsigned int slot = (unsigned int)(idx - token * (unsigned long long)slots);
        const unsigned long long pkt_base = token * (unsigned long long)packet_words;

        unsigned int expert_word = 0xFFFFFFFFu;
        unsigned int weight_word = 0u;
        if (slot < top_k) {
            const unsigned long long src_idx =
                token * (unsigned long long)top_k + (unsigned long long)slot;
            expert_word = (unsigned int)indices[src_idx];
            weight_word = __float_as_uint(weights[src_idx]);
        }

        out_packets_u32[pkt_base + (unsigned long long)slot] = expert_word;
        out_packets_u32[pkt_base + (unsigned long long)slots + (unsigned long long)slot] =
            weight_word;
    }
}

extern "C" __global__ void ep_unpack_selected_routes_kernel(
    const unsigned int* __restrict__ in_packets_u32,
    int* __restrict__ out_indices,
    float* __restrict__ out_weights,
    unsigned int total_tokens,
    unsigned int top_k,
    unsigned int slots
) {
    if (total_tokens == 0u || top_k == 0u || top_k > slots) {
        return;
    }
    if (slots != 6u && slots != 8u && slots != 10u) {
        return;
    }

    const unsigned long long total_items =
        (unsigned long long)total_tokens * (unsigned long long)top_k;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    const unsigned int packet_words = slots * 2u;

    for (unsigned long long idx =
             (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
             (unsigned long long)threadIdx.x;
         idx < total_items;
         idx += stride) {
        const unsigned long long token = idx / (unsigned long long)top_k;
        const unsigned int slot = (unsigned int)(idx - token * (unsigned long long)top_k);
        const unsigned long long pkt_base = token * (unsigned long long)packet_words;

        const unsigned int expert_word = in_packets_u32[pkt_base + (unsigned long long)slot];
        const unsigned int weight_word =
            in_packets_u32[pkt_base + (unsigned long long)slots + (unsigned long long)slot];

        out_indices[idx] = (int)expert_word;
        out_weights[idx] = __uint_as_float(weight_word);
    }
}
