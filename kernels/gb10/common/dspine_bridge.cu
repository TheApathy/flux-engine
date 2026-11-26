// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>

// Native DSpine (arXiv:2609.36173) Adjacent Causal Injection & Exact-Sender
// Serial-Table Refinement kernels for SM121a / GB10.
//
// Pass-1 (per active causal draft layer, over `rows = blocks * block_rows`
// where row `0` of each block is the verified anchor token and rows
// `1 .. block_rows` are the `gamma` masked draft positions):
//   1. `dspine_bridge_read`:
//        rms = sqrt(max(mean(x^2), FLT_MIN))
//        normalized = bf16(x / (rms + eps))
//   2. [GEMM] projected = normalized @ read_l^T                   [rows, C]
//   3. `dspine_bridge_code`:
//        anchor row (`row % block_rows == 0`): bf16(sqrt(C) * codebook[anchor])
//        draft  row (`row % block_rows != 0`): bf16(sqrt(C) * projected / (||projected||_2 + eps))
//   4. [GEMM] packed = codes @ [gate; predecessor_gate; prev]^T   [rows, 3*R]
//   5. `dspine_bridge_message`:
//        anchor row: 0
//        draft  row: bf16(sigmoid(q_t + s_{t-1} + gate_bias) * v_{t-1})
//   6. [GEMM] delta = message @ write_l^T                         [rows, H]
//   7. `dspine_bridge_write_norm`:
//        optional stash = x (pre-write state on final layer)
//        draft row: x = bf16(x + rms * delta)
//        optional fused RMSNorm: normed = bf16(bf16(x * rsqrt(mean(x^2) + norm_eps)) * norm_w)
//
// Pass-2 (`serial_table` exact-sender refinement over `draft_rows = blocks * (block_rows - 1)`
// using `topk_bf16`'s `[draft_rows, k]` u32 candidate IDs, `k <= 16`):
//   8. `dspine_serial_table`:
//        Sorts each draft row's `k` candidates ascending by token ID so exact
//        ties break to the lowest ID. Scores every `(sender_j, cand_k)` pair:
//          base_k = <raw[row] * norm_w, lm_head[cand_k]>
//          m_{j}  = sigmoid(q[row] + TSV_g[sender_j] + gate_bias) * TSV_v[sender_j]
//          S_{j,k} = base_k + rms[row] * <m_j, TG[cand_k]>
//        where `row = b * block_rows + 1 + d`, `sender_j = anchor` for `d == 0`
//        and `prev_sorted_cand[j]` for `d > 0`.
//   9. `dspine_serial_walk`:
//        Walks `j = 0; for d in 0..gamma: j = idx[d, j]; out[d] = cand[d, j]`.
//
// Compiled with `--fmad=false` (common/ rule) so rounding boundaries match the
// IEEE reference arithmetic at every BF16 cast.

#define DSP_THREADS 256
#define DSP_MAX_K 16
#define DSP_MAX_RANK 512
#define DSP_TAIL_THREADS 512

__device__ __forceinline__ float dsp_block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    if (lane == 0u) {
        red[warp] = v;
    }
    __syncthreads();
    if (warp == 0u) {
        v = (lane < (blockDim.x >> 5)) ? red[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, o);
        }
        if (lane == 0u) {
            red[0] = v;
        }
    }
    __syncthreads();
    const float s = red[0];
    __syncthreads();
    return s;
}

// Grid: (rows), Block: DSP_THREADS (256).
extern "C" __global__ void dspine_bridge_read(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ normalized,
    float* __restrict__ rms_out,
    unsigned int hidden,
    float eps
) {
    __shared__ float red[32];
    const unsigned long long row = blockIdx.x;
    const __nv_bfloat16* xr = x + row * hidden;
    float ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float f = __bfloat162float(xr[i]);
        ss += f * f;
    }
    const float mean = dsp_block_sum(ss, red) / (float)hidden;
    const float rms = sqrtf(fmaxf(mean, 1.1754943508222875e-38f));
    if (threadIdx.x == 0u) {
        rms_out[row] = rms;
    }
    const float denom = rms + eps;
    __nv_bfloat16* out = normalized + row * hidden;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        out[i] = __float2bfloat16(__bfloat162float(xr[i]) / denom);
    }
}

// Grid: (rows), Block: DSP_THREADS (256).
extern "C" __global__ void dspine_bridge_code(
    const __nv_bfloat16* __restrict__ projected,
    const __nv_bfloat16* __restrict__ codebook,
    const unsigned int* __restrict__ anchors,
    unsigned int anchor_stride,
    __nv_bfloat16* __restrict__ codes,
    unsigned int block_rows,
    unsigned int code_size,
    unsigned int vocab,
    float eps
) {
    __shared__ float red[32];
    const unsigned long long row = blockIdx.x;
    const float sqrt_c = sqrtf((float)code_size);
    __nv_bfloat16* out = codes + row * code_size;
    if (row % block_rows == 0ull) {
        unsigned int a = anchors[(row / block_rows) * anchor_stride];
        if (a >= vocab) {
            a = 0u;
        }
        const __nv_bfloat16* cb = codebook + (unsigned long long)a * code_size;
        for (unsigned int i = threadIdx.x; i < code_size; i += blockDim.x) {
            out[i] = __float2bfloat16(sqrt_c * __bfloat162float(cb[i]));
        }
        return;
    }
    const __nv_bfloat16* p = projected + row * code_size;
    float ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < code_size; i += blockDim.x) {
        const float f = __bfloat162float(p[i]);
        ss += f * f;
    }
    const float denom = sqrtf(dsp_block_sum(ss, red)) + eps;
    for (unsigned int i = threadIdx.x; i < code_size; i += blockDim.x) {
        out[i] = __float2bfloat16(sqrt_c * __bfloat162float(p[i]) / denom);
    }
}

// Grid: (rows), Block: DSP_THREADS (256). `packed` layout per row is `[q | s | v]` (`3 * rank`).
extern "C" __global__ void dspine_bridge_message(
    const __nv_bfloat16* __restrict__ packed,
    const __nv_bfloat16* __restrict__ gate_bias,
    __nv_bfloat16* __restrict__ message,
    unsigned int block_rows,
    unsigned int rank
) {
    const unsigned long long row = blockIdx.x;
    __nv_bfloat16* out = message + row * rank;
    if (row % block_rows == 0ull) {
        for (unsigned int i = threadIdx.x; i < rank; i += blockDim.x) {
            out[i] = __float2bfloat16(0.0f);
        }
        return;
    }
    const __nv_bfloat16* own = packed + row * 3ull * rank;
    const __nv_bfloat16* pred = packed + (row - 1ull) * 3ull * rank;
    for (unsigned int i = threadIdx.x; i < rank; i += blockDim.x) {
        const float z = __bfloat162float(own[i])
            + __bfloat162float(pred[rank + i])
            + __bfloat162float(gate_bias[i]);
        const float gate = 1.0f / (1.0f + expf(-z));
        out[i] = __float2bfloat16(gate * __bfloat162float(pred[2u * rank + i]));
    }
}

// Grid: (rows), Block: DSP_THREADS (256).
// Updates `x` in place on draft rows (`row % block_rows != 0`).
// When `stash != nullptr`, saves the pre-write `x` row for Pass-2 `serial_table`.
// When `norm_w != nullptr && normed_out != nullptr`, fuses the next layer's
// (or final head's) RMSNorm into `normed_out`.
extern "C" __global__ void dspine_bridge_write_norm(
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ rms,
    const __nv_bfloat16* __restrict__ delta,
    __nv_bfloat16* __restrict__ stash,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ normed_out,
    unsigned int block_rows,
    unsigned int hidden,
    float norm_eps
) {
    __shared__ float red[32];
    const unsigned long long row = blockIdx.x;
    __nv_bfloat16* xr = x + row * hidden;
    const bool is_draft = (row % block_rows) != 0ull;
    const float r = is_draft ? rms[row] : 0.0f;
    const __nv_bfloat16* d = delta + row * hidden;
    const bool do_norm = (norm_w != nullptr) && (normed_out != nullptr);

    float ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const __nv_bfloat16 raw_bf16 = xr[i];
        if (stash != nullptr) {
            stash[row * hidden + i] = raw_bf16;
        }
        __nv_bfloat16 updated = raw_bf16;
        if (is_draft) {
            const float upd = __bfloat162float(raw_bf16) + r * __bfloat162float(d[i]);
            updated = __float2bfloat16(upd);
            xr[i] = updated;
        }
        if (do_norm) {
            const float xf = __bfloat162float(updated);
            ss += xf * xf;
        }
    }

    if (do_norm) {
        const float variance = dsp_block_sum(ss, red) / (float)hidden;
        const float inv_rms = rsqrtf(variance + norm_eps);
        __nv_bfloat16* yn = normed_out + row * hidden;
        for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
            const __nv_bfloat16 y = __float2bfloat16(__bfloat162float(xr[i]) * inv_rms);
            yn[i] = __float2bfloat16(__bfloat162float(y) * __bfloat162float(norm_w[i]));
        }
    }
}

// Load-time helper: `dst[i] = bf16(src[i] * (col ? col[i % cols] : scalar))`.
extern "C" __global__ void dspine_scale(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    unsigned long long n,
    unsigned int cols,
    float scalar,
    const __nv_bfloat16* __restrict__ col
) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const float s = (col != nullptr) ? __bfloat162float(col[i % cols]) : scalar;
        dst[i] = __float2bfloat16(__bfloat162float(src[i]) * s);
    }
}

// Load-time helper for folding optional `terminal_residual` (`[V, R]`) into
// the value half (`[V, R..2R]`) of `tsv` (`[V, 2*R]`).
extern "C" __global__ void dspine_fold_terminal_tsv(
    __nv_bfloat16* __restrict__ tsv,
    const __nv_bfloat16* __restrict__ residual,
    unsigned int vocab,
    unsigned int rank
) {
    const unsigned long long total = (unsigned long long)vocab * rank;
    for (unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (unsigned long long)gridDim.x * blockDim.x) {
        const unsigned long long v = idx / rank;
        const unsigned int r = (unsigned int)(idx % rank);
        const unsigned long long tsv_off = v * (2ull * rank) + rank + r;
        const float sum = __bfloat162float(tsv[tsv_off]) + __bfloat162float(residual[idx]);
        tsv[tsv_off] = __float2bfloat16(sum);
    }
}

__device__ __constant__ float DSP_E2M1[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float dsp_e4m3(unsigned char b) {
    const unsigned int sign = (b >> 7) & 1u;
    const unsigned int exp = (b >> 3) & 0xFu;
    const unsigned int man = b & 0x7u;
    float v;
    if (exp == 0u) {
        v = (float)man * 0.001953125f;
    } else if (exp == 15u && man == 7u) {
        v = 0.0f;
    } else {
        v = (1.0f + (float)man * 0.125f) * exp2f((float)((int)exp - 7));
    }
    return sign ? -v : v;
}

struct DspHead {
    const __nv_bfloat16* bf16;
    const unsigned char* packed;
    const unsigned char* scales;
    const float* scale2_rows;
    float scale2;
    unsigned int hidden;

    __device__ __forceinline__ float at(unsigned int id, unsigned int i) const {
        if (bf16 != nullptr) {
            return __bfloat162float(bf16[(unsigned long long)id * hidden + i]);
        }
        const unsigned long long r = id;
        const unsigned char byte = packed[r * (hidden >> 1) + (i >> 1)];
        const unsigned int nib = (i & 1u) ? (byte >> 4) : (byte & 0xFu);
        const float g = (scale2_rows != nullptr) ? scale2_rows[id] : scale2;
        return DSP_E2M1[nib] * (dsp_e4m3(scales[r * (hidden >> 4) + (i >> 4)]) * g);
    }
};

// Sort `topk_k` (`<= 16`) candidate IDs of `topk_tokens` row `d_row` ascending
// into `out` so exact score ties always pick the lowest token ID.
__device__ __forceinline__ void dsp_sorted_ids(
    const unsigned int* topk_tokens,
    unsigned int d_row,
    unsigned int topk_k,
    unsigned int* out
) {
    if (threadIdx.x != 0u) {
        return;
    }
    const unsigned int* ids = topk_tokens + (unsigned long long)d_row * topk_k;
    for (unsigned int k = 0u; k < topk_k; k++) {
        const unsigned int v = ids[k];
        unsigned int j = k;
        while (j > 0u && out[j - 1u] > v) {
            out[j] = out[j - 1u];
            j--;
        }
        out[j] = v;
    }
}

// Grid: (draft_rows = blocks * draft_per_block), Block: DSP_TAIL_THREADS (512 = 16 warps).
// `raw`, `rms`, and `packed` have `blocks * (draft_per_block + 1)` rows (anchor + drafts);
// `topk_tokens`, `cand`, and `idx` have `draft_rows` rows of width `topk_k` (`<= 16`).
extern "C" __global__ void dspine_serial_table(
    const unsigned int* __restrict__ topk_tokens,
    const __nv_bfloat16* __restrict__ raw,
    const float* __restrict__ rms,
    const __nv_bfloat16* __restrict__ packed,
    const __nv_bfloat16* __restrict__ norm_w,
    const __nv_bfloat16* __restrict__ gate_bias,
    const __nv_bfloat16* __restrict__ tg,
    const __nv_bfloat16* __restrict__ tsv,
    const unsigned int* __restrict__ anchors,
    unsigned int anchor_stride,
    const __nv_bfloat16* __restrict__ head_bf16,
    const unsigned char* __restrict__ head_packed,
    const unsigned char* __restrict__ head_scales,
    const float* __restrict__ head_scale2_rows,
    float head_scale2,
    unsigned int hidden,
    unsigned int* __restrict__ cand,
    unsigned int* __restrict__ idx,
    unsigned int draft_per_block,
    unsigned int topk_k,
    unsigned int rank,
    unsigned int vocab
) {
    __shared__ unsigned int s_cand[DSP_MAX_K];
    __shared__ unsigned int s_send[DSP_MAX_K];
    __shared__ float s_base[DSP_MAX_K];
    __shared__ float s_score[DSP_MAX_K][DSP_MAX_K];
    __shared__ __nv_bfloat16 s_tg[DSP_MAX_K * DSP_MAX_RANK];

    if (draft_per_block == 0u || topk_k == 0u || topk_k > DSP_MAX_K || rank > DSP_MAX_RANK
        || blockDim.x != DSP_TAIL_THREADS) {
        return;
    }

    const unsigned int d_row = blockIdx.x;
    const unsigned int b = d_row / draft_per_block;
    const unsigned int d = d_row % draft_per_block;
    const unsigned int block_rows = draft_per_block + 1u;
    const unsigned int full_row = b * block_rows + 1u + d;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const DspHead head{head_bf16, head_packed, head_scales, head_scale2_rows, head_scale2, hidden};

    dsp_sorted_ids(topk_tokens, d_row, topk_k, s_cand);
    if (d > 0u) {
        dsp_sorted_ids(topk_tokens, d_row - 1u, topk_k, s_send);
    } else if (threadIdx.x < topk_k) {
        s_send[threadIdx.x] = anchors[b * anchor_stride];
    }
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < topk_k * rank; i += blockDim.x) {
        const unsigned int cid = s_cand[i / rank];
        s_tg[i] = tg[(unsigned long long)(cid < vocab ? cid : 0u) * rank + (i % rank)];
    }

    // base_k = <raw[full_row] * norm_w, head[s_cand[k]]> computed by warp `k`.
    if (warp < topk_k) {
        unsigned int cid = s_cand[warp];
        if (cid >= vocab) {
            cid = 0u;
        }
        const __nv_bfloat16* xr = raw + (unsigned long long)full_row * hidden;
        float acc = 0.0f;
        for (unsigned int i = lane; i < hidden; i += 32u) {
            const float u = __bfloat162float(xr[i]) * __bfloat162float(norm_w[i]);
            acc += u * head.at(cid, i);
        }
        for (int o = 16; o > 0; o >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, o);
        }
        if (lane == 0u) {
            s_base[warp] = acc;
        }
    }
    __syncthreads();

    // Sender slot `j = warp`: compute message `m_j` across 32 lanes and dot with
    // each candidate's `s_tg[k]`.
    if (warp < topk_k) {
        unsigned int sid = s_send[warp];
        if (sid >= vocab) {
            sid = 0u;
        }
        const __nv_bfloat16* q = packed + (unsigned long long)full_row * 3ull * rank;
        const __nv_bfloat16* ts = tsv + (unsigned long long)sid * 2ull * rank;
        float m[DSP_MAX_RANK / 32];
#pragma unroll
        for (int u = 0; u < DSP_MAX_RANK / 32; u++) {
            const unsigned int r = lane + 32u * (unsigned int)u;
            m[u] = 0.0f;
            if (r < rank) {
                const float z = __bfloat162float(q[r])
                    + __bfloat162float(ts[r])
                    + __bfloat162float(gate_bias[r]);
                m[u] = (1.0f / (1.0f + expf(-z))) * __bfloat162float(ts[rank + r]);
            }
        }
        const float rr = rms[full_row];
        for (unsigned int k = 0u; k < topk_k; k++) {
            float dot = 0.0f;
#pragma unroll
            for (int u = 0; u < DSP_MAX_RANK / 32; u++) {
                const unsigned int r = lane + 32u * (unsigned int)u;
                if (r < rank) {
                    dot += m[u] * __bfloat162float(s_tg[k * rank + r]);
                }
            }
            for (int o = 16; o > 0; o >>= 1) {
                dot += __shfl_xor_sync(0xffffffffu, dot, o);
            }
            if (lane == 0u) {
                const unsigned int cid = s_cand[k];
                s_score[warp][k] = (cid < vocab) ? (s_base[k] + rr * dot) : -INFINITY;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < topk_k) {
        const unsigned int j = threadIdx.x;
        unsigned int best = 0u;
        float bv = -INFINITY;
        for (unsigned int k = 0u; k < topk_k; k++) {
            const float v = s_score[j][k];
            if (v > bv) {
                bv = v;
                best = k;
            }
        }
        cand[d_row * topk_k + j] = (s_cand[j] < vocab) ? s_cand[j] : 0u;
        idx[d_row * topk_k + j] = best;
    }
}

// Grid: (blocks), Block: 32.
// Walks `d = 0 .. draft_per_block` starting from sender slot `0` (anchor) and
// writes the selected greedy chain directly to `out_tokens` (`[blocks, draft_per_block]`).
extern "C" __global__ void dspine_serial_walk(
    const unsigned int* __restrict__ cand,
    const unsigned int* __restrict__ idx,
    unsigned int* __restrict__ out_tokens,
    unsigned int draft_per_block,
    unsigned int topk_k
) {
    if (threadIdx.x != 0u || draft_per_block == 0u || topk_k == 0u) {
        return;
    }
    const unsigned int r0 = blockIdx.x * draft_per_block;
    unsigned int j = 0u;
    for (unsigned int d = 0u; d < draft_per_block; d++) {
        j = idx[(r0 + d) * topk_k + j];
        out_tokens[r0 + d] = cand[(r0 + d) * topk_k + j];
    }
}
