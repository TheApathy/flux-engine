// SPDX-License-Identifier: AGPL-3.0-only

// Atlas BF16 → NVFP4 runtime weight quantization.
//
// Two-phase quantization at model load time:
//   Phase 1: nvfp4_global_absmax — find max |weight| across entire matrix
//   Phase 2: quantize_bf16_to_nvfp4 — per-group E2M1 quantization
//
// Output format matches HuggingFace NVFP4 (compressed-tensors):
//   packed:  [N, K/2] uint8 — two E2M1 nibbles per byte (K-dim packing)
//   scales:  [N, K/16] FP8 E4M3 — one scale per group of 16 elements
//   scale2:  scalar FP32 — per-tensor second-level scale
//
// Dequant: weight = E2M1_LUT[nibble] * fp8_scale * scale2

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#define GROUP_SIZE 16

// ── Software float→FP8 E4M3 conversion ──
//
// SM121 (GB10) may not support the cvt.rn.satfinite.e4m3x2.f32 PTX instruction.
// This software fallback uses IEEE 754 bit manipulation.
__device__ unsigned char float_to_fp8_e4m3(float v) {
    unsigned int bits = __float_as_uint(v);
    unsigned int sign = (bits >> 31) & 1;
    int f32_exp = (int)((bits >> 23) & 0xFF) - 127;  // unbiased exponent
    unsigned int f32_man = bits & 0x7FFFFF;  // 23-bit mantissa

    // Handle zero / denorm
    if ((bits & 0x7FFFFFFF) == 0) return (unsigned char)(sign << 7);

    // Clamp to E4M3 finite range [-448, 448]:
    // max finite encoding is exp=15, man=6 -> (1 + 6/8) * 2^(15 - 7) = 448
    // (exp=15, man=7 is reserved for NaN).

    // Saturate: if |v| > 448, clamp
    float absv = fabsf(v);
    if (absv > 448.0f) absv = 448.0f;

    // Recompute from clamped
    bits = __float_as_uint(absv);
    f32_exp = (int)((bits >> 23) & 0xFF) - 127;
    f32_man = bits & 0x7FFFFF;

    int e4m3_exp;
    unsigned int e4m3_man;

    if (f32_exp < -9) {
        // Too small → zero
        return (unsigned char)(sign << 7);
    } else if (f32_exp < -6) {
        // Subnormal in E4M3: val = man * 2^(-9)
        // man = round(absv / 2^(-9)) = round(absv * 512)
        int man = (int)(absv * 512.0f + 0.5f);
        if (man > 7) man = 7;
        if (man < 0) man = 0;
        return (unsigned char)((sign << 7) | man);
    } else {
        // Normal: val = (1 + man/8) * 2^(exp-7)
        e4m3_exp = f32_exp + 7;  // E4M3 bias = 7
        if (e4m3_exp < 1) e4m3_exp = 1;
        if (e4m3_exp > 15) {
            // Overflow → clamp to max
            e4m3_exp = 15;
            e4m3_man = 6;  // max normal (man=7 is NaN)
        } else {
            // Round mantissa: f32 has 23 bits, E4M3 has 3
            e4m3_man = (f32_man + (1 << 19)) >> 20;  // round to nearest
            if (e4m3_man > 7) {
                e4m3_man = 0;
                e4m3_exp++;
                if (e4m3_exp > 15) {
                    e4m3_exp = 15;
                    e4m3_man = 6;
                }
            }
        }
        return (unsigned char)((sign << 7) | (e4m3_exp << 3) | e4m3_man);
    }
}

// ── E2M1 nearest-value quantizer ──
//
// Maps float v to the nearest E2M1 4-bit value (0..15).
// Positive E2M1 values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
// Negative mirror at bit 3 (sign).
__device__ unsigned int quantize_e2m1(float v) {
    float absv = fabsf(v);
    unsigned int sign = (v < 0.0f) ? 8u : 0u;
    unsigned int idx;

    if      (absv <= 0.25f) idx = 0;  // 0.0
    else if (absv <= 0.75f) idx = 1;  // 0.5
    else if (absv <= 1.25f) idx = 2;  // 1.0
    else if (absv <= 1.75f) idx = 3;  // 1.5
    else if (absv <= 2.5f)  idx = 4;  // 2.0
    else if (absv <= 3.5f)  idx = 5;  // 3.0
    else if (absv <= 5.0f)  idx = 6;  // 4.0
    else                    idx = 7;  // 6.0 (saturate)

    return sign | idx;
}

// ── Phase 1: Global absolute maximum ──
//
// Grid: (min(total/256, 1024), 1, 1)  Block: (256, 1, 1)
// Scans entire [N, K] BF16 matrix, atomicMax to single output float.
// Caller must zero-initialize global_max before launch.
extern "C" __global__ void nvfp4_global_absmax(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ global_max,
    unsigned int total_elements
) {
    float local_max = 0.0f;

    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x) {
        float v = fabsf(__bfloat162float(input[idx]));
        if (v > local_max) local_max = v;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }

    // Cross-warp reduction
    __shared__ float smem[32];
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int num_warps = (blockDim.x + WARP_SIZE - 1u) / WARP_SIZE;
    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_max = 0.0f;
        for (unsigned int w = 0; w < num_warps; w++) {
            if (smem[w] > block_max) block_max = smem[w];
        }
        // AtomicMax for non-negative floats via uint (IEEE 754 ordering)
        atomicMax((unsigned int*)global_max, __float_as_uint(block_max));
    }
}

// ── Phase 2: Quantize BF16 → NVFP4 ──
//
// Grid: (N, 1, 1)  Block: (256, 1, 1)
// One block per row. Each thread processes ceil(K/16/256) groups.
//
// scale2 = global_max / (6.0 * 448.0), passed as kernel arg.
// Per-group fp8_scale = group_max / (6.0 * scale2).
extern "C" __global__ void quantize_bf16_to_nvfp4(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float scale2,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.x;
    if (row >= N) return;

    const __nv_bfloat16* row_in = input + (unsigned long long)row * K;
    unsigned char* row_packed = packed_out + (unsigned long long)row * (K / 2);
    unsigned char* row_scale = scale_out + (unsigned long long)row * (K / GROUP_SIZE);

    float inv_scale2 = (scale2 > 0.0f) ? (1.0f / scale2) : 0.0f;
    unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int base = g * GROUP_SIZE;

        // Find group max absolute value
        float group_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < GROUP_SIZE; i++) {
            float v = fabsf(__bfloat162float(row_in[base + i]));
            if (v > group_max) group_max = v;
        }

        // Compute FP8 scale: fp8_val ≈ group_max / 6.0 / scale2
        // Use software float→E4M3 conversion (SM121 hardware cast may fail)
        float fp8_float = (group_max > 0.0f) ? (group_max * inv_scale2 / 6.0f) : 0.0f;
        unsigned char fp8_byte = float_to_fp8_e4m3(fp8_float);
        row_scale[g] = fp8_byte;

        // Effective scale for quantization — dequant the FP8 scale back to float
        // using the E4M3 LUT (or recompute from bits)
        unsigned int fp8_sign = (fp8_byte >> 7) & 1;
        unsigned int fp8_exp = (fp8_byte >> 3) & 0xF;
        unsigned int fp8_man = fp8_byte & 0x7;
        float fp8_decoded;
        if (fp8_exp == 0) {
            fp8_decoded = (float)fp8_man * 0.001953125f;  // 2^(-9) per mantissa unit
        } else if (fp8_exp == 15 && fp8_man == 7) {
            fp8_decoded = 0.0f;  // NaN → 0
        } else {
            unsigned int f32_bits = ((fp8_exp + 120u) << 23) | (fp8_man << 20);
            fp8_decoded = __uint_as_float(f32_bits);
        }
        if (fp8_sign) fp8_decoded = -fp8_decoded;

        float effective_scale = fp8_decoded * scale2;
        float inv_eff = (effective_scale > 0.0f) ? (1.0f / effective_scale) : 0.0f;

        // Quantize 16 elements → 8 packed bytes
        #pragma unroll
        for (int i = 0; i < GROUP_SIZE; i += 2) {
            float v0 = __bfloat162float(row_in[base + i]) * inv_eff;
            float v1 = __bfloat162float(row_in[base + i + 1]) * inv_eff;

            unsigned int n0 = quantize_e2m1(v0);
            unsigned int n1 = quantize_e2m1(v1);

            // Pack: low nibble = even position, high nibble = odd position
            row_packed[g * 8 + i / 2] = (unsigned char)((n1 << 4) | (n0 & 0xF));
        }
    }
}

// ── Fused SiLU(gate) * up → NVFP4 quantization ──
//
// Grid: (N, 1, 1)  Block: (256, 1, 1)
//
// This is the down-projection input boundary for a SwiGLU FFN. It preserves
// the standalone `moe_silu_mul` numerical contract by materializing each
// FP32 result through BF16 before either the group maximum or E2M1 quantizer
// observes it. The materialization happens in registers instead of through a
// temporary global BF16 matrix.
__device__ __forceinline__ float silu_mul_bf16_round(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    unsigned long long idx
) {
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    return __bfloat162float(__float2bfloat16(g * sigmoid_g * u));
}

extern "C" __global__ void quantize_silu_mul_bf16_to_nvfp4(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float scale2,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.x;
    if (row >= N) return;

    unsigned long long row_base = (unsigned long long)row * K;
    unsigned char* row_packed = packed_out + (unsigned long long)row * (K / 2);
    unsigned char* row_scale = scale_out + (unsigned long long)row * (K / GROUP_SIZE);

    float inv_scale2 = (scale2 > 0.0f) ? (1.0f / scale2) : 0.0f;
    unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int group = threadIdx.x; group < num_groups; group += blockDim.x) {
        unsigned int base = group * GROUP_SIZE;

        float group_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < GROUP_SIZE; i++) {
            float value = fabsf(silu_mul_bf16_round(gate, up, row_base + base + i));
            if (value > group_max) group_max = value;
        }

        float fp8_float = (group_max > 0.0f) ? (group_max * inv_scale2 / 6.0f) : 0.0f;
        unsigned char fp8_byte = float_to_fp8_e4m3(fp8_float);
        row_scale[group] = fp8_byte;

        unsigned int fp8_sign = (fp8_byte >> 7) & 1;
        unsigned int fp8_exp = (fp8_byte >> 3) & 0xF;
        unsigned int fp8_man = fp8_byte & 0x7;
        float fp8_decoded;
        if (fp8_exp == 0) {
            fp8_decoded = (float)fp8_man * 0.001953125f;
        } else if (fp8_exp == 15 && fp8_man == 7) {
            fp8_decoded = 0.0f;
        } else {
            unsigned int f32_bits = ((fp8_exp + 120u) << 23) | (fp8_man << 20);
            fp8_decoded = __uint_as_float(f32_bits);
        }
        if (fp8_sign) fp8_decoded = -fp8_decoded;

        float effective_scale = fp8_decoded * scale2;
        float inv_eff = (effective_scale > 0.0f) ? (1.0f / effective_scale) : 0.0f;

        #pragma unroll
        for (int i = 0; i < GROUP_SIZE; i += 2) {
            float v0 = silu_mul_bf16_round(gate, up, row_base + base + i) * inv_eff;
            float v1 = silu_mul_bf16_round(gate, up, row_base + base + i + 1) * inv_eff;
            unsigned int n0 = quantize_e2m1(v0);
            unsigned int n1 = quantize_e2m1(v1);
            row_packed[group * 8 + i / 2] =
                (unsigned char)((n1 << 4) | (n0 & 0xF));
        }
    }
}

// ABI-separated FlashInfer/CUTLASS physical-scale sibling. Packed activation
// bytes remain logical row-major [N,K/2], while scale bytes occupy the padded
// 128-row x 4-group layout consumed by the SM121 down projection.
__device__ __forceinline__ unsigned long long silu_scale_offset_128x4(
    unsigned int row,
    unsigned int group,
    unsigned int groups
) {
    unsigned int group_blocks = (groups + 3u) / 4u;
    return (((((static_cast<unsigned long long>(row / 128u) * group_blocks + group / 4u) * 32u
                + row % 32u) * 4u + (row % 128u) / 32u) * 4u) + group % 4u);
}

extern "C" __global__ void quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float scale2,
    unsigned int N,
    unsigned int K
) {
    unsigned int padded_rows = ((N + 127u) / 128u) * 128u;
    unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int row = blockIdx.x; row < padded_rows; row += gridDim.x) {
        if (row >= N) {
            for (unsigned int group = threadIdx.x; group < num_groups; group += blockDim.x) {
                scale_out[silu_scale_offset_128x4(row, group, num_groups)] = 0;
            }
            continue;
        }

        unsigned long long row_base = static_cast<unsigned long long>(row) * K;
        unsigned char* row_packed = packed_out + static_cast<unsigned long long>(row) * (K / 2u);
        float inv_scale2 = (scale2 > 0.0f) ? (1.0f / scale2) : 0.0f;

        for (unsigned int group = threadIdx.x; group < num_groups; group += blockDim.x) {
            unsigned int base = group * GROUP_SIZE;
            float group_max = 0.0f;
#pragma unroll
            for (int i = 0; i < GROUP_SIZE; i++) {
                float value = fabsf(silu_mul_bf16_round(gate, up, row_base + base + i));
                if (value > group_max) group_max = value;
            }

            float fp8_float = (group_max > 0.0f) ? (group_max * inv_scale2 / 6.0f) : 0.0f;
            unsigned char fp8_byte = float_to_fp8_e4m3(fp8_float);
            scale_out[silu_scale_offset_128x4(row, group, num_groups)] = fp8_byte;

            unsigned int fp8_sign = (fp8_byte >> 7) & 1;
            unsigned int fp8_exp = (fp8_byte >> 3) & 0xF;
            unsigned int fp8_man = fp8_byte & 0x7;
            float fp8_decoded;
            if (fp8_exp == 0) {
                fp8_decoded = (float)fp8_man * 0.001953125f;
            } else if (fp8_exp == 15 && fp8_man == 7) {
                fp8_decoded = 0.0f;
            } else {
                unsigned int f32_bits = ((fp8_exp + 120u) << 23) | (fp8_man << 20);
                fp8_decoded = __uint_as_float(f32_bits);
            }
            if (fp8_sign) fp8_decoded = -fp8_decoded;

            float effective_scale = fp8_decoded * scale2;
            float inv_eff = (effective_scale > 0.0f) ? (1.0f / effective_scale) : 0.0f;
#pragma unroll
            for (int i = 0; i < GROUP_SIZE; i += 2) {
                float v0 = silu_mul_bf16_round(gate, up, row_base + base + i) * inv_eff;
                float v1 = silu_mul_bf16_round(gate, up, row_base + base + i + 1) * inv_eff;
                unsigned int n0 = quantize_e2m1(v0);
                unsigned int n1 = quantize_e2m1(v1);
                row_packed[group * 8u + i / 2] =
                    (unsigned char)((n1 << 4) | (n0 & 0xF));
            }
        }
    }
}

// Direct sibling for a merged FlashInfer gate/up projection. Each row is
// laid out as [gate K | up K]. Keeping the sixteen BF16-rounded SwiGLU values
// in registers makes the group maximum and E2M1 packing consume the same
// materialization without an intermediate split or a second input pass.
extern "C" __global__ void quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
    const __nv_bfloat16* __restrict__ merged_gate_up,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float scale2,
    unsigned int N,
    unsigned int K
) {
    unsigned int padded_rows = ((N + 127u) / 128u) * 128u;
    unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int row = blockIdx.x; row < padded_rows; row += gridDim.x) {
        if (row >= N) {
            for (unsigned int group = threadIdx.x; group < num_groups; group += blockDim.x) {
                scale_out[silu_scale_offset_128x4(row, group, num_groups)] = 0;
            }
            continue;
        }

        unsigned long long merged_row_base =
            static_cast<unsigned long long>(row) * (2ull * K);
        const __nv_bfloat16* gate = merged_gate_up + merged_row_base;
        const __nv_bfloat16* up = gate + K;
        unsigned char* row_packed = packed_out + static_cast<unsigned long long>(row) * (K / 2u);
        float inv_scale2 = (scale2 > 0.0f) ? (1.0f / scale2) : 0.0f;

        for (unsigned int group = threadIdx.x; group < num_groups; group += blockDim.x) {
            unsigned int base = group * GROUP_SIZE;
            float rounded[GROUP_SIZE];
            float group_max = 0.0f;
#pragma unroll
            for (int i = 0; i < GROUP_SIZE; i++) {
                float value = silu_mul_bf16_round(gate, up, base + i);
                rounded[i] = value;
                float magnitude = fabsf(value);
                if (magnitude > group_max) group_max = magnitude;
            }

            float fp8_float = (group_max > 0.0f) ? (group_max * inv_scale2 / 6.0f) : 0.0f;
            unsigned char fp8_byte = float_to_fp8_e4m3(fp8_float);
            scale_out[silu_scale_offset_128x4(row, group, num_groups)] = fp8_byte;

            unsigned int fp8_sign = (fp8_byte >> 7) & 1;
            unsigned int fp8_exp = (fp8_byte >> 3) & 0xF;
            unsigned int fp8_man = fp8_byte & 0x7;
            float fp8_decoded;
            if (fp8_exp == 0) {
                fp8_decoded = (float)fp8_man * 0.001953125f;
            } else if (fp8_exp == 15 && fp8_man == 7) {
                fp8_decoded = 0.0f;
            } else {
                unsigned int f32_bits = ((fp8_exp + 120u) << 23) | (fp8_man << 20);
                fp8_decoded = __uint_as_float(f32_bits);
            }
            if (fp8_sign) fp8_decoded = -fp8_decoded;

            float effective_scale = fp8_decoded * scale2;
            float inv_eff = (effective_scale > 0.0f) ? (1.0f / effective_scale) : 0.0f;
#pragma unroll
            for (int i = 0; i < GROUP_SIZE; i += 2) {
                unsigned int n0 = quantize_e2m1(rounded[i] * inv_eff);
                unsigned int n1 = quantize_e2m1(rounded[i + 1] * inv_eff);
                row_packed[group * 8u + i / 2] =
                    (unsigned char)((n1 << 4) | (n0 & 0xF));
            }
        }
    }
}

#include <cuda_fp4.h>

// Hardware-RNE BF16 → NVFP4 runtime weight quantizer.
//
// Uses CUDA's native nearest-even E4M3 (`__nv_fp8_e4m3`) and E2M1
// (`__nv_cvt_float2_to_fp4x2(..., __NV_E2M1, cudaRoundNearest)`) conversions,
// fixing midpoint ties, E4M3 subnormal rounding, and signed-zero preservation
// for cold-load runtime weight quantization while keeping the activation
// quantizers above byte-for-byte untouched.
extern "C" __global__ void quantize_bf16_to_nvfp4_weight_rne(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float scale2,
    unsigned int N,
    unsigned int K
) {
    const unsigned int row = blockIdx.x;
    if (row >= N) return;
    const __nv_bfloat16* row_in = input + (unsigned long long)row * K;
    unsigned char* row_packed = packed_out + (unsigned long long)row * (K / 2);
    unsigned char* row_scale = scale_out + (unsigned long long)row * (K / 16);
    const float inv_scale2 = scale2 > 0.0f ? 1.0f / scale2 : 0.0f;
    for (unsigned int g = threadIdx.x; g < K / 16; g += blockDim.x) {
        float group_max = 0.0f;
        #pragma unroll
        for (unsigned int j = 0; j < 16; ++j)
            group_max = fmaxf(group_max, fabsf(__bfloat162float(row_in[g * 16 + j])));
        const float raw_scale = group_max > 0.0f ? group_max * inv_scale2 / 6.0f : 0.0f;
        const __nv_fp8_e4m3 rounded_scale(raw_scale);
        row_scale[g] = *reinterpret_cast<const unsigned char*>(&rounded_scale);
        const float effective_scale = (float)rounded_scale * scale2;
        const float inv_eff = effective_scale > 0.0f ? 1.0f / effective_scale : 0.0f;
        #pragma unroll
        for (unsigned int j = 0; j < 16; j += 2) {
            const float2 pair = make_float2(
                __bfloat162float(row_in[g * 16 + j]) * inv_eff,
                __bfloat162float(row_in[g * 16 + j + 1]) * inv_eff);
            row_packed[g * 8 + j / 2] =
                __nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest);
        }
    }
}

// Async on-device FP32 → BF16 round-to-nearest-even conversion for norm weights
// in `dense_f32_safe`, eliminating synchronous D2H/H2D round-trips during load.
extern "C" __global__ void f32_to_bf16_rne(
    const unsigned int* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2bfloat16(__uint_as_float(in[i]));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Vectorized Blackwell PTX 128x4 NVFP4 prefill quantizers & fused norm
// ═══════════════════════════════════════════════════════════════════

namespace {

__device__ __forceinline__ float qptx_rcp_approx_ftz(float v) {
    float r;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v));
    return r;
}

__device__ __forceinline__ float qptx_mul_rn(float a, float b) {
    float r;
    asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float2 qptx_unpack_bf16x2(unsigned int packed) {
    return make_float2(
        __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(packed & 0xffffu))),
        __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(packed >> 16))));
}

__device__ __forceinline__ unsigned int qptx_pack_bf16x2(float a, float b) {
    unsigned int lo = static_cast<unsigned int>(__bfloat16_as_ushort(__float2bfloat16(a)));
    unsigned int hi = static_cast<unsigned int>(__bfloat16_as_ushort(__float2bfloat16(b)));
    return lo | (hi << 16);
}

__device__ __forceinline__ void qptx_unpack_uint4_bf16(uint4 raw, float* out) {
    const float2 f0 = qptx_unpack_bf16x2(raw.x);
    const float2 f1 = qptx_unpack_bf16x2(raw.y);
    const float2 f2 = qptx_unpack_bf16x2(raw.z);
    const float2 f3 = qptx_unpack_bf16x2(raw.w);
    out[0] = f0.x; out[1] = f0.y;
    out[2] = f1.x; out[3] = f1.y;
    out[4] = f2.x; out[5] = f2.y;
    out[6] = f3.x; out[7] = f3.y;
}

__device__ __forceinline__ unsigned int qptx_e2m1_eight(const float* values, float scale) {
    const float v0 = qptx_mul_rn(values[0], scale);
    const float v1 = qptx_mul_rn(values[1], scale);
    const float v2 = qptx_mul_rn(values[2], scale);
    const float v3 = qptx_mul_rn(values[3], scale);
    const float v4 = qptx_mul_rn(values[4], scale);
    const float v5 = qptx_mul_rn(values[5], scale);
    const float v6 = qptx_mul_rn(values[6], scale);
    const float v7 = qptx_mul_rn(values[7], scale);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    unsigned int result;
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "}"
        : "=r"(result)
        : "f"(v0), "f"(v1), "f"(v2), "f"(v3), "f"(v4), "f"(v5), "f"(v6), "f"(v7));
    return result;
#else
    unsigned int b0 = (quantize_e2m1(v1) << 4) | (quantize_e2m1(v0) & 0xfu);
    unsigned int b1 = (quantize_e2m1(v3) << 4) | (quantize_e2m1(v2) & 0xfu);
    unsigned int b2 = (quantize_e2m1(v5) << 4) | (quantize_e2m1(v4) & 0xfu);
    unsigned int b3 = (quantize_e2m1(v7) << 4) | (quantize_e2m1(v6) & 0xfu);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
#endif
}

__device__ __forceinline__ void qptx_store_group16(
    const float* values,
    float group_max,
    float scale2,
    float inv_scale2_over_6,
    unsigned char* __restrict__ row_packed,
    unsigned char* __restrict__ scale_out,
    unsigned int row,
    unsigned int group,
    unsigned int groups
) {
    const float fp8_float =
        group_max > 0.0f ? fminf(qptx_mul_rn(group_max, inv_scale2_over_6), 448.0f) : 0.0f;
    const __nv_fp8_e4m3 encoded_scale = __nv_fp8_e4m3(fp8_float);
    scale_out[silu_scale_offset_128x4(row, group, groups)] = encoded_scale.__x;

    const float decoded = static_cast<float>(encoded_scale);
    const float effective_scale = qptx_mul_rn(decoded, scale2);
    const float inv_eff = effective_scale > 0.0f ? qptx_rcp_approx_ftz(effective_scale) : 0.0f;

    uint2 packed;
    packed.x = qptx_e2m1_eight(values, inv_eff);
    packed.y = qptx_e2m1_eight(values + 8, inv_eff);
    *reinterpret_cast<uint2*>(row_packed + static_cast<unsigned long long>(group) * 8u) = packed;
}

__device__ __forceinline__ float qptx_silu_mul_pair(
    unsigned int g_packed,
    unsigned int u_packed,
    float& out0,
    float& out1
) {
    const float2 g = qptx_unpack_bf16x2(g_packed);
    const float2 u = qptx_unpack_bf16x2(u_packed);
    const float v0 = qptx_mul_rn(qptx_mul_rn(g.x, qptx_rcp_approx_ftz(1.0f + __expf(-g.x))), u.x);
    const float v1 = qptx_mul_rn(qptx_mul_rn(g.y, qptx_rcp_approx_ftz(1.0f + __expf(-g.y))), u.y);
    out0 = v0;
    out1 = v1;
    return fmaxf(fabsf(v0), fabsf(v1));
}

__device__ __forceinline__ float qptx_silu_mul_8(
    const __nv_bfloat16* __restrict__ gate_ptr,
    const __nv_bfloat16* __restrict__ up_ptr,
    float* out
) {
    const uint4 g_raw = *reinterpret_cast<const uint4*>(gate_ptr);
    const uint4 u_raw = *reinterpret_cast<const uint4*>(up_ptr);
    float m = qptx_silu_mul_pair(g_raw.x, u_raw.x, out[0], out[1]);
    m = fmaxf(m, qptx_silu_mul_pair(g_raw.y, u_raw.y, out[2], out[3]));
    m = fmaxf(m, qptx_silu_mul_pair(g_raw.z, u_raw.z, out[4], out[5]));
    m = fmaxf(m, qptx_silu_mul_pair(g_raw.w, u_raw.w, out[6], out[7]));
    return m;
}

__device__ __forceinline__ float qptx_warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

}  // namespace

// Vectorized Blackwell PTX sibling of `quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4`.
// Reads merged `[gate K | up K]` via 16-byte `uint4` loads, evaluates SwiGLU in FP32,
// and emits packed E2M1 (`uint2` 64-bit stores) + 128x4-swizzled UE4M3 scales.
extern "C" __global__ void __launch_bounds__(256)
quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4_ptx(
    const __nv_bfloat16* __restrict__ merged_gate_up,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float scale2,
    unsigned int N,
    unsigned int K
) {
    const unsigned int padded_rows = ((N + 127u) / 128u) * 128u;
    const unsigned int num_groups = K / GROUP_SIZE;
    const float inv_scale2_over_6 = scale2 > 0.0f ? (1.0f / (scale2 * 6.0f)) : 0.0f;

    for (unsigned int row = blockIdx.x; row < padded_rows; row += gridDim.x) {
        if (row >= N) {
            for (unsigned int group = threadIdx.x; group < num_groups; group += blockDim.x) {
                scale_out[silu_scale_offset_128x4(row, group, num_groups)] = 0;
            }
            continue;
        }

        const unsigned long long merged_row_base =
            static_cast<unsigned long long>(row) * (2ull * K);
        const __nv_bfloat16* gate = merged_gate_up + merged_row_base;
        const __nv_bfloat16* up = gate + K;
        unsigned char* row_packed = packed_out + static_cast<unsigned long long>(row) * (K / 2u);

        for (unsigned int group = threadIdx.x; group < num_groups; group += blockDim.x) {
            const unsigned int base = group * GROUP_SIZE;
            float values[16];
            const float m0 = qptx_silu_mul_8(gate + base, up + base, values);
            const float m1 = qptx_silu_mul_8(gate + base + 8u, up + base + 8u, values + 8);
            const float group_max = fmaxf(m0, m1);

            qptx_store_group16(
                values,
                group_max,
                scale2,
                inv_scale2_over_6,
                row_packed,
                scale_out,
                row,
                group,
                num_groups);
        }
    }
}

// Vectorized Blackwell PTX sibling of `quantize_bf16_to_nvfp4_atlas_128x4_from_absmax`.
// Preserves the exact 8-argument ABI and fail-closed status flags (`BAD_GLOBAL_MAX=1`,
// `NONFINITE_INPUT=2`, `BAD_SCALE2=4`) while replacing scalar loads/stores and software
// E4M3/E2M1 encoding with `uint4` loads, hardware FP8/FP4 PTX conversions, and `uint2` stores.
extern "C" __global__ void __launch_bounds__(256)
quantize_bf16_to_nvfp4_atlas_128x4_from_absmax_ptx(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    const float* __restrict__ global_max,
    float* __restrict__ scale2_out,
    unsigned int* __restrict__ status_out,
    unsigned int rows,
    unsigned int cols
) {
    const float maximum = global_max[0];
    const bool valid_maximum = isfinite(maximum) && maximum >= 0.0f;
    const float scale2 = valid_maximum
        ? (maximum > 0.0f ? __fdiv_rn(maximum, 2688.0f) : 1.0f)
        : 1.0f;
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        scale2_out[0] = scale2;
        if (!valid_maximum) atomicOr(status_out, 1u << 0);
        if (!isfinite(scale2)) atomicOr(status_out, 1u << 2);
    }

    const unsigned int padded_rows = ((rows + 127u) / 128u) * 128u;
    const unsigned int groups = cols / GROUP_SIZE;
    const float inv_scale2_over_6 = scale2 > 0.0f ? (1.0f / (scale2 * 6.0f)) : 0.0f;

    for (unsigned int row = blockIdx.x; row < padded_rows; row += gridDim.x) {
        if (row >= rows) {
            for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
                scale_out[silu_scale_offset_128x4(row, group, groups)] = 0;
            }
            continue;
        }

        const __nv_bfloat16* row_input = input + static_cast<unsigned long long>(row) * cols;
        unsigned char* row_packed = packed_out + static_cast<unsigned long long>(row) * (cols / 2u);

        for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
            const unsigned int base = group * GROUP_SIZE;
            float values[16];
            qptx_unpack_uint4_bf16(*reinterpret_cast<const uint4*>(row_input + base), values);
            qptx_unpack_uint4_bf16(*reinterpret_cast<const uint4*>(row_input + base + 8u), values + 8);

            float group_max = 0.0f;
            bool finite_group = valid_maximum;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (!isfinite(values[i])) {
                    finite_group = false;
                } else {
                    const float a = fabsf(values[i]);
                    if (a > group_max) group_max = a;
                }
            }

            if (!finite_group) {
                atomicOr(status_out, 1u << 1);
                scale_out[silu_scale_offset_128x4(row, group, groups)] = 0;
                *reinterpret_cast<uint2*>(
                    row_packed + static_cast<unsigned long long>(group) * 8u) = make_uint2(0u, 0u);
                continue;
            }

            qptx_store_group16(
                values,
                group_max,
                scale2,
                inv_scale2_over_6,
                row_packed,
                scale_out,
                row,
                group,
                groups);
        }
    }
}

// Fused Residual Add + Qwen3 (1+w) RMSNorm + Static-Scale NVFP4 128x4 Quantizer.
// Eliminates the intermediate BF16 `norm_output` write + read (168 MB/layer at 8K),
// optional dead `residual` write (`residual == nullptr`), and second `hidden` read.
// Grid: (min(padded_rows, cta_cap), 1, 1)
// Block: (groups <= 512 ? ((groups + 31) / 32) * 32 : 256, 1, 1)
extern "C" __global__ void __launch_bounds__(512)
residual_add_rms_norm_quantize_nvfp4_128x4(
    __nv_bfloat16* __restrict__ hidden,       // [rows, cols] in/out: hidden += src
    const __nv_bfloat16* __restrict__ src,    // [rows, cols]
    const __nv_bfloat16* __restrict__ weight, // [cols] Qwen3 (1+w) offset weight
    __nv_bfloat16* __restrict__ residual,     // [rows, cols] optional (nullptr = skip)
    unsigned char* __restrict__ packed_out,   // [rows, cols/2] E2M1
    unsigned char* __restrict__ scale_out,    // [padded_rows, cols/16] 128x4 UE4M3
    float scale2,
    float eps,
    unsigned int rows,
    unsigned int cols
) {
    const unsigned int padded_rows = ((rows + 127u) / 128u) * 128u;
    const unsigned int groups = cols / GROUP_SIZE;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32u;
    const unsigned int lane_id = tid % 32u;
    const unsigned int num_warps = (blockDim.x + 31u) / 32u;
    const float inv_scale2_over_6 = scale2 > 0.0f ? (1.0f / (scale2 * 6.0f)) : 0.0f;
    const bool single_pass = (groups <= blockDim.x);

    __shared__ float warp_sums[32];

    for (unsigned int row = blockIdx.x; row < padded_rows; row += gridDim.x) {
        if (row >= rows) {
            for (unsigned int g = tid; g < groups; g += blockDim.x) {
                scale_out[silu_scale_offset_128x4(row, g, groups)] = 0;
            }
            continue;
        }

        const unsigned long long row_base = static_cast<unsigned long long>(row) * cols;
        __nv_bfloat16* h_row = hidden + row_base;
        const __nv_bfloat16* s_row = src + row_base;
        __nv_bfloat16* r_row = (residual != nullptr) ? (residual + row_base) : nullptr;
        unsigned char* row_packed = packed_out + static_cast<unsigned long long>(row) * (cols / 2u);

        uint4 h_cached_lo = make_uint4(0u, 0u, 0u, 0u);
        uint4 h_cached_hi = make_uint4(0u, 0u, 0u, 0u);
        float sum_sq = 0.0f;

        if (single_pass) {
            if (tid < groups) {
                const unsigned int base = tid * GROUP_SIZE;
                const uint4 h0 = *reinterpret_cast<const uint4*>(h_row + base);
                const uint4 h1 = *reinterpret_cast<const uint4*>(h_row + base + 8u);
                const uint4 s0 = *reinterpret_cast<const uint4*>(s_row + base);
                const uint4 s1 = *reinterpret_cast<const uint4*>(s_row + base + 8u);

                float hv[16], sv[16];
                qptx_unpack_uint4_bf16(h0, hv);
                qptx_unpack_uint4_bf16(h1, hv + 8);
                qptx_unpack_uint4_bf16(s0, sv);
                qptx_unpack_uint4_bf16(s1, sv + 8);

                float nv[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    nv[i] = hv[i] + sv[i];
                    sum_sq += nv[i] * nv[i];
                }
                h_cached_lo.x = qptx_pack_bf16x2(nv[0], nv[1]);
                h_cached_lo.y = qptx_pack_bf16x2(nv[2], nv[3]);
                h_cached_lo.z = qptx_pack_bf16x2(nv[4], nv[5]);
                h_cached_lo.w = qptx_pack_bf16x2(nv[6], nv[7]);
                h_cached_hi.x = qptx_pack_bf16x2(nv[8], nv[9]);
                h_cached_hi.y = qptx_pack_bf16x2(nv[10], nv[11]);
                h_cached_hi.z = qptx_pack_bf16x2(nv[12], nv[13]);
                h_cached_hi.w = qptx_pack_bf16x2(nv[14], nv[15]);

                *reinterpret_cast<uint4*>(h_row + base) = h_cached_lo;
                *reinterpret_cast<uint4*>(h_row + base + 8u) = h_cached_hi;
                if (r_row != nullptr) {
                    *reinterpret_cast<uint4*>(r_row + base) = h_cached_lo;
                    *reinterpret_cast<uint4*>(r_row + base + 8u) = h_cached_hi;
                }
            }
        } else {
            for (unsigned int g = tid; g < groups; g += blockDim.x) {
                const unsigned int base = g * GROUP_SIZE;
                const uint4 h0 = *reinterpret_cast<const uint4*>(h_row + base);
                const uint4 h1 = *reinterpret_cast<const uint4*>(h_row + base + 8u);
                const uint4 s0 = *reinterpret_cast<const uint4*>(s_row + base);
                const uint4 s1 = *reinterpret_cast<const uint4*>(s_row + base + 8u);

                float hv[16], sv[16], nv[16];
                qptx_unpack_uint4_bf16(h0, hv);
                qptx_unpack_uint4_bf16(h1, hv + 8);
                qptx_unpack_uint4_bf16(s0, sv);
                qptx_unpack_uint4_bf16(s1, sv + 8);
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    nv[i] = hv[i] + sv[i];
                    sum_sq += nv[i] * nv[i];
                }
                uint4 o0, o1;
                o0.x = qptx_pack_bf16x2(nv[0], nv[1]);
                o0.y = qptx_pack_bf16x2(nv[2], nv[3]);
                o0.z = qptx_pack_bf16x2(nv[4], nv[5]);
                o0.w = qptx_pack_bf16x2(nv[6], nv[7]);
                o1.x = qptx_pack_bf16x2(nv[8], nv[9]);
                o1.y = qptx_pack_bf16x2(nv[10], nv[11]);
                o1.z = qptx_pack_bf16x2(nv[12], nv[13]);
                o1.w = qptx_pack_bf16x2(nv[14], nv[15]);

                *reinterpret_cast<uint4*>(h_row + base) = o0;
                *reinterpret_cast<uint4*>(h_row + base + 8u) = o1;
                if (r_row != nullptr) {
                    *reinterpret_cast<uint4*>(r_row + base) = o0;
                    *reinterpret_cast<uint4*>(r_row + base + 8u) = o1;
                }
            }
        }

        sum_sq = qptx_warp_reduce_sum(sum_sq);
        if (lane_id == 0u) warp_sums[warp_id] = sum_sq;
        __syncthreads();
        if (warp_id == 0u) {
            float val = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
            val = qptx_warp_reduce_sum(val);
            if (lane_id == 0u) warp_sums[0] = val;
        }
        __syncthreads();

        const float rms = rsqrtf(warp_sums[0] / (float)cols + eps);

        if (single_pass) {
            if (tid < groups) {
                const unsigned int base = tid * GROUP_SIZE;
                float hv[16], wv[16], values[16];
                qptx_unpack_uint4_bf16(h_cached_lo, hv);
                qptx_unpack_uint4_bf16(h_cached_hi, hv + 8);
                qptx_unpack_uint4_bf16(*reinterpret_cast<const uint4*>(weight + base), wv);
                qptx_unpack_uint4_bf16(*reinterpret_cast<const uint4*>(weight + base + 8u), wv + 8);

                float group_max = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    // Match `residual_add_rms_norm` -> BF16 -> `quantize_bf16_to_nvfp4_atlas_128x4_ptx`.
                    const float normed =
                        __bfloat162float(__float2bfloat16(hv[i] * rms * (1.0f + wv[i])));
                    values[i] = normed;
                    group_max = fmaxf(group_max, fabsf(normed));
                }

                qptx_store_group16(
                    values,
                    group_max,
                    scale2,
                    inv_scale2_over_6,
                    row_packed,
                    scale_out,
                    row,
                    tid,
                    groups);
            }
        } else {
            for (unsigned int g = tid; g < groups; g += blockDim.x) {
                const unsigned int base = g * GROUP_SIZE;
                float hv[16], wv[16], values[16];
                qptx_unpack_uint4_bf16(*reinterpret_cast<const uint4*>(h_row + base), hv);
                qptx_unpack_uint4_bf16(*reinterpret_cast<const uint4*>(h_row + base + 8u), hv + 8);
                qptx_unpack_uint4_bf16(*reinterpret_cast<const uint4*>(weight + base), wv);
                qptx_unpack_uint4_bf16(*reinterpret_cast<const uint4*>(weight + base + 8u), wv + 8);

                float group_max = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const float normed =
                        __bfloat162float(__float2bfloat16(hv[i] * rms * (1.0f + wv[i])));
                    values[i] = normed;
                    group_max = fmaxf(group_max, fabsf(normed));
                }

                qptx_store_group16(
                    values,
                    group_max,
                    scale2,
                    inv_scale2_over_6,
                    row_packed,
                    scale_out,
                    row,
                    g,
                    groups);
            }
        }
        __syncthreads();
    }
}

