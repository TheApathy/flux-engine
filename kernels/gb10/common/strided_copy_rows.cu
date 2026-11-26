// SPDX-License-Identifier: AGPL-3.0-only
// Byte-exact strided row copy: rows x row_bytes from src (stride src_stride)
// to dst (stride dst_stride). Replaces per-row cuMemcpyDtoDAsync loops
// (e.g. the DFlash prefill capture: 2048 copies per captured layer).
// Requires 16-byte aligned base pointers, strides and row_bytes.
extern "C" __global__ void strided_copy_rows_16(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int rows,
    unsigned int row_bytes,
    unsigned long long src_stride,
    unsigned long long dst_stride
) {
    const unsigned int row = blockIdx.y;
    if (row >= rows) return;
    const unsigned int chunks = row_bytes >> 4;
    const uint4* s = (const uint4*)(src + (unsigned long long)row * src_stride);
    uint4* d = (uint4*)(dst + (unsigned long long)row * dst_stride);
    for (unsigned int c = blockIdx.x * blockDim.x + threadIdx.x; c < chunks; c += gridDim.x * blockDim.x) {
        d[c] = s[c];
    }
}

// Single-launch Q/K/V scratch scatter + optional gated Q/Gate deinterleave into
// the per-sequence strided `qkv_buf` layout:
//   qkv_dst[seq * dst_stride_bf16 + 0 .. q_proj_dim)                <- Q (deinterleaved if gated)
//   qkv_dst[seq * dst_stride_bf16 + q_proj_dim .. +kv_dim)          <- K
//   qkv_dst[seq * dst_stride_bf16 + q_proj_dim + kv_dim .. +kv_dim) <- V
//
// Replaces 3*N `cuMemcpyDtoDAsync` + N `deinterleave_qg` launches with 1 launch.
// Grid: (ceil((q_proj_dim + 2 * kv_dim) / 256), num_seqs, 1)  Block: (256, 1, 1)
extern "C" __global__ void scatter_qkv_multi_seq(
    const unsigned short* __restrict__ q_src,
    const unsigned short* __restrict__ k_src,
    const unsigned short* __restrict__ v_src,
    unsigned short* __restrict__ qkv_dst,
    const unsigned int num_seqs,
    const unsigned int q_proj_dim,
    const unsigned int kv_dim,
    const unsigned int dst_stride_bf16,
    const unsigned int num_q_heads,
    const unsigned int head_dim,
    const unsigned int gated
) {
    const unsigned int seq = blockIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = q_proj_dim + 2u * kv_dim;
    if (seq >= num_seqs || col >= total) return;

    unsigned short* dst_row = qkv_dst + (unsigned long long)seq * dst_stride_bf16;
    if (col < q_proj_dim) {
        unsigned int src_col = col;
        if (gated != 0u) {
            const unsigned int q_dim = num_q_heads * head_dim;
            const unsigned int group_dim = 2u * head_dim;
            if (col < q_dim) {
                const unsigned int h = col / head_dim;
                const unsigned int d = col % head_dim;
                src_col = h * group_dim + d;
            } else {
                const unsigned int gi = col - q_dim;
                const unsigned int h = gi / head_dim;
                const unsigned int d = gi % head_dim;
                src_col = h * group_dim + head_dim + d;
            }
        }
        dst_row[col] = q_src[(unsigned long long)seq * q_proj_dim + src_col];
    } else if (col < q_proj_dim + kv_dim) {
        dst_row[col] = k_src[(unsigned long long)seq * kv_dim + (col - q_proj_dim)];
    } else {
        dst_row[col] = v_src[(unsigned long long)seq * kv_dim + (col - q_proj_dim - kv_dim)];
    }
}
