// SPDX-License-Identifier: AGPL-3.0-only
//
// Fused QSA & Verify Attention Kernels for Qwen3.8-Flash-Next-NVFP4:
//   1. `qwen4_qsa_verify_rows_stage_pool_rope_store`:
//      Collapses the per-row host loop in `Qwen4QsaIndexer::update_and_select_verify_rows`
//      (`crates/spark-model/src/layers/qwen4_qsa.rs`, previously 28-30 launches/layer)
//      into a SINGLE kernel launch `grid=(1, 1, 1), block=(256, 1, 1)` per layer.
//      For each row `0..num_tokens` sequentially on device:
//        - Snapshots `raw_ring` and (at ring_row == 3) `compressed_cache` into the undo journal
//          (`undo_raw`, `undo_compressed`, `undo_raw_offsets`, `undo_compressed_offsets`, `valid_mask`).
//        - Stages the 5th head (`projected_qk[4*128 .. 5*128]`) into `raw_ring`.
//        - Normalizes the 4 query heads (`projected_qk[0 .. 4*128]`) with `q_norm_weight`
//          using the exact 2-warp (64-thread) per-head FP32 reduction tree as `rms_norm_forward`.
//        - Applies RoPE (YaRN scaled when `inv_freq != nullptr`, or standard RoPE) to the 4 query heads.
//        - When `physical.ring_row == 3`, pools the 4 ring rows with `__float2bfloat16_rn`,
//          applies `k_norm_weight` RMSNorm, applies RoPE at `first_pos = position - 3`, and writes
//          to `compressed_cache` (plus updates `pooled_key` and `first_position`).
//
//   2. `rope_forward_strided_yarn_scaled`:
//      Strided multi-token YaRN-scaled RoPE (`grid=(num_q_heads + num_kv_heads, num_tokens, 1)`,
//      `block=(128, 1, 1)`), collapsing per-token `ops::rope_yarn_scaled` launches
//      (`crates/spark-model/src/layers/qwen3_attention/decode/{qsa_route.rs,attention_forward.rs}`)
//      from `num_tokens` launches/layer (72 launches/step) into 1 launch/layer (12 launches/step).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

constexpr unsigned int INDEX_HEADS = 4u;
constexpr unsigned int INDEX_DIM = 128u;
constexpr unsigned int INDEX_WIDTH = (INDEX_HEADS + 1u) * INDEX_DIM; // 640
constexpr unsigned int COMPRESS_RATIO = 4u;
constexpr unsigned int VERIFY_MAX_ROWS = 9u;
constexpr unsigned long long NO_COMPRESSED_OFFSET = ~0ull;

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

struct PhysicalSlot {
    unsigned long long block;
    unsigned int offset_in_block;
    unsigned int group;
    unsigned int ring_row;
};

__device__ __forceinline__ bool decode_physical_slot(
    long long slot,
    unsigned int block_size,
    PhysicalSlot& out
) {
    if (slot < 0 || block_size == 0u || (block_size % COMPRESS_RATIO) != 0u) {
        return false;
    }
    const unsigned long long linear = static_cast<unsigned long long>(slot);
    out.block = linear / block_size;
    out.offset_in_block = static_cast<unsigned int>(linear % block_size);
    out.group = out.offset_in_block / COMPRESS_RATIO;
    out.ring_row = out.offset_in_block % COMPRESS_RATIO;
    return true;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

__device__ __forceinline__ void unpack_bf16x2(unsigned int packed, float &f0, float &f1) {
    __nv_bfloat16 b0, b1;
    unsigned short lo = (unsigned short)(packed & 0xFFFFu);
    unsigned short hi = (unsigned short)(packed >> 16);
    memcpy(&b0, &lo, 2);
    memcpy(&b1, &hi, 2);
    f0 = __bfloat162float(b0);
    f1 = __bfloat162float(b1);
}

__device__ __forceinline__ unsigned int pack_bf16x2(float f0, float f1) {
    __nv_bfloat16 b0 = __float2bfloat16(f0);
    __nv_bfloat16 b1 = __float2bfloat16(f1);
    unsigned short lo, hi;
    memcpy(&lo, &b0, 2);
    memcpy(&hi, &b1, 2);
    return ((unsigned int)hi << 16) | (unsigned int)lo;
}

// Register-only bit-exact twin of CUDA libdevice's `sinf(x)` + `cosf(x)`.
// Replaces the 28-byte `.local` array in libdevice's Payne-Hanek slowpath with
// scalar registers `p0..p6` so `ptxas` compiles with `0 bytes stack frame` and `0 spills`
// while producing IEEE-754 results bit-for-bit identical to `sinf(x)` and `cosf(x)`.
__device__ __forceinline__ float exact_cuda_trig_poly(float r, int q_poly, int q_sign) {
    const float r2 = __fmaf_rn(r, r, 0.0f) - 0.0f; // mul.rn.f32
    const bool use_sin_poly = (q_poly & 1) != 0;
    const float f23 = use_sin_poly ? r : 1.0f;
    const float f24 = __fmaf_rn(r * r, f23, 0.0f);
    const float f25 = __fmaf_rn(r * r, __int_as_float(0x37CBAC00u), __int_as_float(0xBAB607EDu));
    const float f26 = use_sin_poly ? __int_as_float(0xB94D4153u) : f25;
    const float f27 = use_sin_poly ? __int_as_float(0x3C0885E4u) : __int_as_float(0x3D2AAABBu);
    const float f28 = __fmaf_rn(f26, r * r, f27);
    const float f29 = use_sin_poly ? __int_as_float(0xBE2AAAA8u) : __int_as_float(0xBEFFFFFFu);
    const float f30 = __fmaf_rn(f28, r * r, f29);
    const float f31 = __fmaf_rn(f30, f24, f23);
    return ((q_sign & 2) == 0) ? f31 : (0.0f - f31);
}

__device__ __forceinline__ void exact_cuda_sincosf(float x, float& s_out, float& c_out) {
    int q = 0;
    float r = 0.0f;
    const float ax = fabsf(x);
    if (ax < __int_as_float(0x47CE4780u)) {
        const float prod = __fmul_rn(x, __int_as_float(0x3F22F983u));
        asm("cvt.rni.s32.f32 %0, %1;" : "=r"(q) : "f"(prod));
        const float fn = __int2float_rn(q);
        r = __fmaf_rn(fn, __int_as_float(0xBFC90FDAu), x);
        r = __fmaf_rn(fn, __int_as_float(0xB3A22168u), r);
        r = __fmaf_rn(fn, __int_as_float(0xA7C234C5u), r);
    } else {
        // Register-resident Payne-Hanek 192-bit 2/pi reduction (zero .local stack usage)
        const unsigned int ix = __float_as_uint(x);
        const unsigned int mant = (ix << 8) | 0x80000000u;
        const unsigned int i2opi[6] = {
            0x3c439041u, 0xdb629599u, 0xf534ddc0u,
            0xfc2757d1u, 0x4e441529u, 0xa2f9836eu
        };
        unsigned long long carry = 0ull;
        unsigned int p[7];
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            unsigned long long wide = (unsigned long long)i2opi[i] * (unsigned long long)mant + carry;
            p[i] = (unsigned int)wide;
            carry = wide >> 32;
        }
        p[6] = (unsigned int)carry;
        const unsigned int exp = ix >> 23;
        const unsigned int shift = exp & 31u;
        const unsigned int idx = ((exp & 224u) - 128u) >> 5;
        unsigned int hi = (idx == 0u) ? p[6] : (idx == 1u ? p[5] : (idx == 2u ? p[4] : p[3]));
        unsigned int lo = (idx == 0u) ? p[5] : (idx == 1u ? p[4] : (idx == 2u ? p[3] : p[2]));
        unsigned int ex = (idx == 0u) ? p[4] : (idx == 1u ? p[3] : (idx == 2u ? p[2] : p[1]));
        if (shift != 0u) {
            hi = __funnelshift_l(lo, hi, shift);
            lo = __funnelshift_l(ex, lo, shift);
        }
        const unsigned int q_bits = hi >> 30;
        const unsigned int f_hi = __funnelshift_l(lo, hi, 2u);
        const unsigned int f_lo = lo << 2;
        const int q_abs = (int)((f_hi >> 31) + q_bits);
        q = ((int)ix < 0) ? -q_abs : q_abs;
        const int sign_flip = (int)f_hi ^ (int)ix;
        const int mask = ((int)f_hi) >> 31;
        const unsigned long long mag =
            (((unsigned long long)(f_hi ^ (unsigned int)mask)) << 32) |
            ((unsigned long long)(f_lo ^ (unsigned int)mask));
        const double d_red = __ll2double_rn((long long)mag) * __longlong_as_double(0x3BF921FB54442D19ull);
        const float f_red = __double2float_rn(d_red);
        r = (sign_flip < 0) ? -f_red : f_red;
    }
    // For sinf: q_poly = q + 1 (so (q & 1) == 0 uses sin poly), q_sign = q
    // For cosf: q_poly = q     (so (q & 1) == 1 uses sin poly), q_sign = q + 1
    s_out = exact_cuda_trig_poly(r, q + 1, q);
    c_out = exact_cuda_trig_poly(r, q, q + 1);
}

__device__ __forceinline__ void rotate_pair_bf16(
    __nv_bfloat16* __restrict__ head_ptr,
    unsigned int pair_idx,
    unsigned int rotary_dim,
    unsigned int abs_pos,
    float base_freq,
    float attn_scale
) {
    const unsigned int half_rot = rotary_dim / 2u;
    if (pair_idx >= half_rot) return;
    const float angle = (float)abs_pos * base_freq;
    float sin_val, cos_val;
    exact_cuda_sincosf(angle, sin_val, cos_val);
    cos_val *= attn_scale;
    sin_val *= attn_scale;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + half_rot;
    const float x0 = __bfloat162float(head_ptr[d0]);
    const float x1 = __bfloat162float(head_ptr[d1]);
    head_ptr[d0] = __float2bfloat16((x0 * cos_val) - (x1 * sin_val));
    head_ptr[d1] = __float2bfloat16((x1 * cos_val) + (x0 * sin_val));
}

} // namespace

// ============================================================================
// 1. Fused Multi-Row QSA Stage + Snapshot + RMSNorm + RoPE + Pool + Store
// ============================================================================
// Grid:  (1, 1, 1)
// Block: (256, 1, 1) — 4 heads * 64 threads/head (2 warps/head)
extern "C" __global__ void __launch_bounds__(256) qwen4_qsa_verify_rows_stage_pool_rope_store(
    __nv_bfloat16* __restrict__ projected_qk,           // [num_tokens, 640] BF16 (in/out)
    __nv_bfloat16* __restrict__ raw_ring,               // physical page 4-row ring
    __nv_bfloat16* __restrict__ pooled_key,             // [128] BF16 scratch
    unsigned int* __restrict__ first_position,          // [1] u32 scratch
    __nv_bfloat16* __restrict__ compressed_cache,       // physical compressed keys
    const __nv_bfloat16* __restrict__ q_norm_weight,    // [128] BF16
    const __nv_bfloat16* __restrict__ k_norm_weight,    // [128] BF16
    const long long* __restrict__ slots,                // [num_tokens] i64
    const unsigned int* __restrict__ positions,         // [num_tokens] u32
    __nv_bfloat16* __restrict__ undo_raw,               // [VERIFY_MAX_ROWS, 128]
    __nv_bfloat16* __restrict__ undo_compressed,        // [VERIFY_MAX_ROWS, 128]
    unsigned long long* __restrict__ undo_raw_offsets,  // [VERIFY_MAX_ROWS]
    unsigned long long* __restrict__ undo_compressed_offsets, // [VERIFY_MAX_ROWS]
    unsigned int* __restrict__ valid_mask,              // [1] u32
    const float* __restrict__ inv_freq,                 // [rotary_dim/2]
    float rope_theta,
    float attention_factor,
    float eps,
    unsigned int rotary_dim,
    unsigned int block_size,
    unsigned int num_tokens
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int head = tid / 64u;          // 0..3
    const unsigned int lane64 = tid % 64u;        // 0..63 (pair index inside 128-dim head)
    const unsigned int warp_in_head = lane64 / 32u; // 0..1
    const unsigned int warp_lane = lane64 % 32u;  // 0..31

    __shared__ float s_warp_sums[8];       // 4 heads * 2 warps/head

    const unsigned int rows = (num_tokens <= VERIFY_MAX_ROWS) ? num_tokens : VERIFY_MAX_ROWS;
    const unsigned int half_rot = rotary_dim / 2u;

    float base_freq = 0.0f;
    float attn_scale = 1.0f;
    if (lane64 < half_rot) {
        if (inv_freq != nullptr) {
            base_freq = inv_freq[lane64];
            attn_scale = attention_factor;
        } else {
            const double freq_exp_d = (double)(2u * lane64) / (double)rotary_dim;
            base_freq = (float)(1.0 / pow((double)rope_theta, freq_exp_d));
        }
    }

    const unsigned int* qw32 = (const unsigned int*)q_norm_weight;
    const unsigned int* kw32 = (const unsigned int*)k_norm_weight;
    float qw0 = 0.0f, qw1 = 0.0f, kw0 = 0.0f, kw1 = 0.0f;
    unpack_bf16x2(qw32[lane64], qw0, qw1);
    unpack_bf16x2(kw32[lane64], kw0, kw1);

    if (tid == 0u) {
        *valid_mask = 0u;
    }
    __syncthreads();

    for (unsigned int row = 0u; row < rows; ++row) {
        __nv_bfloat16* row_qk = projected_qk + (unsigned long long)row * INDEX_WIDTH;
        const long long slot = slots[row];
        const unsigned int position = positions[row];

        PhysicalSlot physical;
        const bool slot_ok = decode_physical_slot(slot, block_size, physical);

        unsigned long long raw_offset = 0ull;
        unsigned long long compressed_offset = NO_COMPRESSED_OFFSET;
        if (slot_ok) {
            raw_offset = (physical.block * COMPRESS_RATIO + physical.ring_row) * INDEX_DIM;
            if (physical.ring_row == COMPRESS_RATIO - 1u) {
                compressed_offset =
                    (physical.block * (block_size / COMPRESS_RATIO) + physical.group) * INDEX_DIM;
            }
        }

        // ── Step A: Snapshot verify row into undo journal (threads 0..127) ──
        if (slot_ok) {
            if (tid == 0u) {
                undo_raw_offsets[row] = raw_offset;
                undo_compressed_offsets[row] = compressed_offset;
            }
            if (tid < INDEX_DIM) {
                undo_raw[(unsigned long long)row * INDEX_DIM + tid] = raw_ring[raw_offset + tid];
                if (compressed_offset != NO_COMPRESSED_OFFSET) {
                    undo_compressed[(unsigned long long)row * INDEX_DIM + tid] =
                        compressed_cache[compressed_offset + tid];
                }
            }
        }
        __syncthreads();

        if (slot_ok && tid == 0u) {
            atomicOr(valid_mask, 1u << row);
        }

        // ── Step B: Stage raw key (threads 0..127) + 4-head Query RMSNorm (threads 0..255) ──
        if (slot_ok && tid < INDEX_DIM) {
            raw_ring[raw_offset + tid] = row_qk[INDEX_HEADS * INDEX_DIM + tid];
        }

        __nv_bfloat16* q_head = row_qk + (unsigned long long)head * INDEX_DIM;
        unsigned int* q_head32 = (unsigned int*)q_head;
        float qv0 = 0.0f, qv1 = 0.0f;
        unpack_bf16x2(q_head32[lane64], qv0, qv1);
        float q_sq = warp_reduce_sum(qv0 * qv0 + qv1 * qv1);
        if (warp_lane == 0u) {
            s_warp_sums[head * 2u + warp_in_head] = q_sq;
        }
        __syncthreads();

        const float q_rms = rsqrtf(
            (s_warp_sums[head * 2u] + s_warp_sums[head * 2u + 1u]) / (float)INDEX_DIM + eps);
        q_head32[lane64] = pack_bf16x2(
            qv0 * q_rms * (1.0f + qw0),
            qv1 * q_rms * (1.0f + qw1));
        __syncthreads();

        // ── Step C: 4-head Query RoPE ──
        rotate_pair_bf16(
            q_head,
            lane64,
            rotary_dim,
            position,
            base_freq,
            attn_scale);

        // ── Step D: Pool 4 ring rows + k_norm + k_rope + store_compressed at endpoint ──
        if (slot_ok) {
            const unsigned int first_pos = position - physical.ring_row;
            if (tid == 0u) {
                *first_position = first_pos;
            }
            if (physical.ring_row == COMPRESS_RATIO - 1u) {
                if (tid < INDEX_DIM) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (unsigned int r = 0u; r < COMPRESS_RATIO; ++r) {
                        sum += __bfloat162float(
                            raw_ring[(physical.block * COMPRESS_RATIO + r) * INDEX_DIM + tid]);
                    }
                    pooled_key[tid] = __float2bfloat16_rn(sum * 0.25f);
                }
                __syncthreads();

                if (head == 0u) {
                    unsigned int* pk32 = (unsigned int*)pooled_key;
                    float kv0, kv1;
                    unpack_bf16x2(pk32[lane64], kv0, kv1);
                    float k_sq = warp_reduce_sum(kv0 * kv0 + kv1 * kv1);
                    if (warp_lane == 0u) {
                        s_warp_sums[warp_in_head] = k_sq;
                    }
                }
                __syncthreads();

                if (head == 0u) {
                    unsigned int* pk32 = (unsigned int*)pooled_key;
                    float kv0, kv1;
                    unpack_bf16x2(pk32[lane64], kv0, kv1);
                    const float k_rms = rsqrtf(
                        (s_warp_sums[0] + s_warp_sums[1]) / (float)INDEX_DIM + eps);
                    pk32[lane64] = pack_bf16x2(
                        kv0 * k_rms * (1.0f + kw0),
                        kv1 * k_rms * (1.0f + kw1));
                }
                __syncthreads();

                if (head == 0u) {
                    rotate_pair_bf16(
                        pooled_key,
                        lane64,
                        rotary_dim,
                        first_pos,
                        base_freq,
                        attn_scale);
                }
                __syncthreads();

                if (tid < INDEX_DIM) {
                    compressed_cache[compressed_offset + tid] = pooled_key[tid];
                }
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// 2. Strided Multi-Row YaRN-Scaled RoPE (`rope_forward_strided_yarn_scaled`)
// ============================================================================
// Grid:  (num_q_heads + num_kv_heads, num_tokens, 1)
// Block: (128, 1, 1)
// Bit-identical to `num_tokens` back-to-back launches of `rope_forward_yarn_scaled`
// (or `rope_forward_yarn` when `attention_factor == 1.0f`).
extern "C" __global__ void __launch_bounds__(128) rope_forward_strided_yarn_scaled(
    __nv_bfloat16* __restrict__ qkv_base,
    const unsigned int* __restrict__ positions,
    const unsigned int num_tokens,
    const unsigned int qkv_stride_bf16,
    const unsigned int k_offset_bf16,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const float* __restrict__ inv_freq,
    const float theta,
    const float attention_factor
) {
    (void)theta;

    const unsigned int head_idx = blockIdx.x;
    const unsigned int token = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    if (token >= num_tokens) return;

    const bool is_q = (head_idx < num_q_heads);
    const unsigned int head = is_q ? head_idx : (head_idx - num_q_heads);
    if (!is_q && head >= num_kv_heads) return;

    const unsigned int pairs_per_pos = rotary_dim / 2u;
    if (tid >= pairs_per_pos) return;
    const unsigned int pair_idx = tid;

    const unsigned int abs_pos = positions[token];
    const float angle = (float)abs_pos * inv_freq[pair_idx];
    float sin_val, cos_val;
    exact_cuda_sincosf(angle, sin_val, cos_val);
    cos_val *= attention_factor;
    sin_val *= attention_factor;

    __nv_bfloat16* token_base = qkv_base + (unsigned long long)token * qkv_stride_bf16;
    __nv_bfloat16* ptr = is_q
        ? (token_base + (unsigned long long)head * head_dim)
        : (token_base + k_offset_bf16 + (unsigned long long)head * head_dim);

    const unsigned int half_rot = rotary_dim / 2u;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + half_rot;
    const float x0 = __bfloat162float(ptr[d0]);
    const float x1 = __bfloat162float(ptr[d1]);
    ptr[d0] = __float2bfloat16((x0 * cos_val) - (x1 * sin_val));
    ptr[d1] = __float2bfloat16((x1 * cos_val) + (x0 * sin_val));
}

// ============================================================================
// 3. Multi-Sequence Fused QSA Stage + Snapshot + RMSNorm + RoPE + Pool + Store
// ============================================================================
// Grid:  (num_seqs, 1, 1)
// Block: (256, 1, 1) — 4 heads * 64 threads/head (2 warps/head)
// Uses CTA-local `__shared__ __align__(16) __nv_bfloat16 s_pooled[128]` instead
// of a shared global `pooled_key` buffer so all `num_seqs` CTAs run concurrently
// with 0-ULP parity against `qwen4_qsa_verify_rows_stage_pool_rope_store`.
extern "C" __global__ void __launch_bounds__(256)
qwen4_qsa_verify_rows_stage_pool_rope_store_multi_seq(
    __nv_bfloat16* __restrict__ projected_qk_base,           // [total_rows, 640] BF16 (in/out)
    __nv_bfloat16* __restrict__ raw_ring,                    // physical page 4-row ring
    __nv_bfloat16* __restrict__ compressed_cache,            // physical compressed keys
    const __nv_bfloat16* __restrict__ q_norm_weight,         // [128] BF16
    const __nv_bfloat16* __restrict__ k_norm_weight,         // [128] BF16
    const long long* __restrict__ slots_base,                // [total_rows] i64
    const unsigned int* __restrict__ positions_base,         // [total_rows] u32
    __nv_bfloat16* __restrict__ undo_raw_base,               // [lanes, VERIFY_MAX_ROWS, 128]
    __nv_bfloat16* __restrict__ undo_compressed_base,        // [lanes, VERIFY_MAX_ROWS, 128]
    unsigned long long* __restrict__ undo_raw_offsets_base,  // [lanes, VERIFY_MAX_ROWS]
    unsigned long long* __restrict__ undo_compressed_offsets_base, // [lanes, VERIFY_MAX_ROWS]
    unsigned int* __restrict__ valid_mask_base,              // [lanes] u32
    const QsaBatchSeqDesc* __restrict__ seq_descs,           // [num_seqs]
    const float* __restrict__ inv_freq,                      // [rotary_dim/2]
    float rope_theta,
    float attention_factor,
    float eps,
    unsigned int rotary_dim,
    unsigned int block_size,
    unsigned int num_seqs
) {
    const unsigned int seq_idx = blockIdx.x;
    if (seq_idx >= num_seqs) return;
    const QsaBatchSeqDesc desc = seq_descs[seq_idx];
    if ((desc.flags & QSA_DESC_FLAG_QSA_REQUIRED) == 0u) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head = tid / 64u;
    const unsigned int lane64 = tid % 64u;
    const unsigned int warp_in_head = lane64 / 32u;
    const unsigned int warp_lane = lane64 % 32u;

    __shared__ float s_warp_sums[8];
    __shared__ __align__(16) __nv_bfloat16 s_pooled[INDEX_DIM];

    const unsigned int rows = (desc.k <= VERIFY_MAX_ROWS) ? desc.k : VERIFY_MAX_ROWS;
    const unsigned int half_rot = rotary_dim / 2u;

    float base_freq = 0.0f;
    float attn_scale = 1.0f;
    if (lane64 < half_rot) {
        if (inv_freq != nullptr) {
            base_freq = inv_freq[lane64];
            attn_scale = attention_factor;
        } else {
            const double freq_exp_d = (double)(2u * lane64) / (double)rotary_dim;
            base_freq = (float)(1.0 / pow((double)rope_theta, freq_exp_d));
        }
    }

    const unsigned int* qw32 = (const unsigned int*)q_norm_weight;
    const unsigned int* kw32 = (const unsigned int*)k_norm_weight;
    float qw0 = 0.0f, qw1 = 0.0f, kw0 = 0.0f, kw1 = 0.0f;
    unpack_bf16x2(qw32[lane64], qw0, qw1);
    unpack_bf16x2(kw32[lane64], kw0, kw1);

    __nv_bfloat16* projected_qk =
        projected_qk_base + (unsigned long long)desc.row_base * INDEX_WIDTH;
    const long long* slots = slots_base + desc.row_base;
    const unsigned int* positions = positions_base + desc.row_base;

    const bool do_journal =
        (desc.flags & QSA_DESC_FLAG_JOURNAL) != 0u && undo_raw_base != nullptr;
    const unsigned long long jlane = (unsigned long long)desc.journal_lane;
    __nv_bfloat16* undo_raw =
        do_journal ? (undo_raw_base + jlane * VERIFY_MAX_ROWS * INDEX_DIM) : nullptr;
    __nv_bfloat16* undo_compressed =
        do_journal ? (undo_compressed_base + jlane * VERIFY_MAX_ROWS * INDEX_DIM) : nullptr;
    unsigned long long* undo_raw_offsets =
        do_journal ? (undo_raw_offsets_base + jlane * VERIFY_MAX_ROWS) : nullptr;
    unsigned long long* undo_compressed_offsets =
        do_journal ? (undo_compressed_offsets_base + jlane * VERIFY_MAX_ROWS) : nullptr;
    unsigned int* valid_mask = do_journal ? (valid_mask_base + jlane) : nullptr;

    if (do_journal && tid == 0u) {
        *valid_mask = 0u;
    }
    __syncthreads();

    for (unsigned int row = 0u; row < rows; ++row) {
        __nv_bfloat16* row_qk = projected_qk + (unsigned long long)row * INDEX_WIDTH;
        const long long slot = slots[row];
        const unsigned int position = positions[row];

        PhysicalSlot physical;
        const bool slot_ok = decode_physical_slot(slot, block_size, physical);

        unsigned long long raw_offset = 0ull;
        unsigned long long compressed_offset = NO_COMPRESSED_OFFSET;
        if (slot_ok) {
            raw_offset = (physical.block * COMPRESS_RATIO + physical.ring_row) * INDEX_DIM;
            if (physical.ring_row == COMPRESS_RATIO - 1u) {
                compressed_offset =
                    (physical.block * (block_size / COMPRESS_RATIO) + physical.group) * INDEX_DIM;
            }
        }

        if (do_journal && slot_ok) {
            if (tid == 0u) {
                undo_raw_offsets[row] = raw_offset;
                undo_compressed_offsets[row] = compressed_offset;
            }
            if (tid < INDEX_DIM) {
                undo_raw[(unsigned long long)row * INDEX_DIM + tid] = raw_ring[raw_offset + tid];
                if (compressed_offset != NO_COMPRESSED_OFFSET) {
                    undo_compressed[(unsigned long long)row * INDEX_DIM + tid] =
                        compressed_cache[compressed_offset + tid];
                }
            }
        }
        __syncthreads();

        if (do_journal && slot_ok && tid == 0u) {
            atomicOr(valid_mask, 1u << row);
        }

        if (slot_ok && tid < INDEX_DIM) {
            raw_ring[raw_offset + tid] = row_qk[INDEX_HEADS * INDEX_DIM + tid];
        }

        __nv_bfloat16* q_head = row_qk + (unsigned long long)head * INDEX_DIM;
        unsigned int* q_head32 = (unsigned int*)q_head;
        float qv0 = 0.0f, qv1 = 0.0f;
        unpack_bf16x2(q_head32[lane64], qv0, qv1);
        float q_sq = warp_reduce_sum(qv0 * qv0 + qv1 * qv1);
        if (warp_lane == 0u) {
            s_warp_sums[head * 2u + warp_in_head] = q_sq;
        }
        __syncthreads();

        const float q_rms = rsqrtf(
            (s_warp_sums[head * 2u] + s_warp_sums[head * 2u + 1u]) / (float)INDEX_DIM + eps);
        q_head32[lane64] = pack_bf16x2(
            qv0 * q_rms * (1.0f + qw0),
            qv1 * q_rms * (1.0f + qw1));
        __syncthreads();

        rotate_pair_bf16(
            q_head,
            lane64,
            rotary_dim,
            position,
            base_freq,
            attn_scale);

        if (slot_ok && physical.ring_row == COMPRESS_RATIO - 1u) {
            const unsigned int first_pos = position - physical.ring_row;
            if (tid < INDEX_DIM) {
                float sum = 0.0f;
                #pragma unroll
                for (unsigned int r = 0u; r < COMPRESS_RATIO; ++r) {
                    sum += __bfloat162float(
                        raw_ring[(physical.block * COMPRESS_RATIO + r) * INDEX_DIM + tid]);
                }
                s_pooled[tid] = __float2bfloat16_rn(sum * 0.25f);
            }
            __syncthreads();

            if (head == 0u) {
                unsigned int* pk32 = (unsigned int*)s_pooled;
                float kv0, kv1;
                unpack_bf16x2(pk32[lane64], kv0, kv1);
                float k_sq = warp_reduce_sum(kv0 * kv0 + kv1 * kv1);
                if (warp_lane == 0u) {
                    s_warp_sums[warp_in_head] = k_sq;
                }
            }
            __syncthreads();

            if (head == 0u) {
                unsigned int* pk32 = (unsigned int*)s_pooled;
                float kv0, kv1;
                unpack_bf16x2(pk32[lane64], kv0, kv1);
                const float k_rms = rsqrtf(
                    (s_warp_sums[0] + s_warp_sums[1]) / (float)INDEX_DIM + eps);
                pk32[lane64] = pack_bf16x2(
                    kv0 * k_rms * (1.0f + kw0),
                    kv1 * k_rms * (1.0f + kw1));
            }
            __syncthreads();

            if (head == 0u) {
                rotate_pair_bf16(
                    s_pooled,
                    lane64,
                    rotary_dim,
                    first_pos,
                    base_freq,
                    attn_scale);
            }
            __syncthreads();

            if (tid < INDEX_DIM) {
                compressed_cache[compressed_offset + tid] = s_pooled[tid];
            }
        }
        __syncthreads();
    }
}
