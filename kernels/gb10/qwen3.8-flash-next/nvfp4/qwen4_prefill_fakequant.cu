// SPDX-License-Identifier: AGPL-3.0-only
//
// Default-off FAKE activation quantization for the Flash-Next prefill quality
// gate (ATLAS_QWEN4_PREFILL_FAKEQUANT, see qwen4_fakequant.rs). A BF16 matrix
// [rows][cols] is rounded to a low-precision GEMM-operand format and back to
// BF16, so the unchanged BF16 GEMMs multiply exactly the operand values a real
// FP8 / NVFP4-activation GEMM would see. The only extra error versus a real
// kernel is the final BF16 rounding of the dequantized value (<= 2^-9 relative,
// against 2^-4 for E4M3 and 2^-2 for E2M1).
//
// Formats (`mode`):
//   0 FP8   E4M3 payload (RN, satfinite), FP32 scale amax/448 per block
//           (block = 0: whole row, i.e. per-token dynamic; 128: 1x128 blockwise)
//   1 MXFP8 E4M3 payload, UE8M0 scale pow2ceil(amax/448) per 32 (b12x w4a8)
//   2 NVFP4 E2M1 payload, E4M3 scale per 16, FP32 global scale2 = amax/2688
//           taken from `tensor_absmax` (per tensor) or from the row (nullptr);
//           block-scale and E2M1 rounding are the 27B activation quantizer's
//           (quantize_bf16_to_nvfp4_cutlass.cu, atlas_* helpers)
//   3 E2M1 payload with NO scales (deliberately bad gate control)
//
// Every division and multiply that sets a rounding boundary is explicit
// (__fdiv_rn / __fmul_rn) so the bundle-wide --use_fast_math cannot move it.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

constexpr unsigned MODE_FP8 = 0;
constexpr unsigned MODE_MXFP8 = 1;
constexpr unsigned MODE_NVFP4 = 2;
constexpr unsigned MODE_E2M1_RAW = 3;
constexpr float E4M3_MAX = 448.0f;

__device__ __constant__ float FQ_E2M1_VALUES[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ __forceinline__ float e4m3_round(float value) {
    const __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(q, __NV_E4M3)));
}

// Byte-for-byte copies of the 27B activation quantizer's legacy conversions
// (quantize_bf16_to_nvfp4_cutlass.cu): half-way and signed-zero behaviour is
// part of what the gate measures.
__device__ __forceinline__ unsigned char atlas_float_to_fp8_e4m3(float value) {
    unsigned int bits = __float_as_uint(value);
    const unsigned int sign = (bits >> 31) & 1u;
    if ((bits & 0x7fffffffu) == 0u) return static_cast<unsigned char>(sign << 7);

    float absolute = fabsf(value);
    if (absolute > 448.0f) absolute = 448.0f;
    bits = __float_as_uint(absolute);
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127;
    const unsigned int mantissa = bits & 0x7fffffu;
    if (exponent < -9) return static_cast<unsigned char>(sign << 7);
    if (exponent < -6) {
        int encoded = static_cast<int>(absolute * 512.0f + 0.5f);
        if (encoded > 7) encoded = 7;
        if (encoded < 0) encoded = 0;
        return static_cast<unsigned char>((sign << 7) | encoded);
    }

    int encoded_exponent = exponent + 7;
    unsigned int encoded_mantissa;
    if (encoded_exponent > 15) {
        encoded_exponent = 15;
        encoded_mantissa = 6;
    } else {
        encoded_mantissa = (mantissa + (1u << 19)) >> 20;
        if (encoded_mantissa > 7) {
            encoded_mantissa = 0;
            ++encoded_exponent;
            if (encoded_exponent > 15) {
                encoded_exponent = 15;
                encoded_mantissa = 6;
            }
        }
    }
    return static_cast<unsigned char>((sign << 7) | (encoded_exponent << 3) |
                                      encoded_mantissa);
}

__device__ __forceinline__ float atlas_decode_fp8_e4m3(unsigned char byte) {
    const unsigned int sign = (byte >> 7) & 1u;
    const unsigned int exponent = (byte >> 3) & 0xfu;
    const unsigned int mantissa = byte & 7u;
    float decoded;
    if (exponent == 0) {
        decoded = static_cast<float>(mantissa) * 0.001953125f;
    } else if (exponent == 15 && mantissa == 7) {
        decoded = 0.0f;
    } else {
        decoded = __uint_as_float(((exponent + 120u) << 23) | (mantissa << 20));
    }
    return sign ? -decoded : decoded;
}

__device__ __forceinline__ float atlas_e2m1_round(float value) {
    const float absolute = fabsf(value);
    unsigned int index;
    if (absolute <= 0.25f) index = 0;
    else if (absolute <= 0.75f) index = 1;
    else if (absolute <= 1.25f) index = 2;
    else if (absolute <= 1.75f) index = 3;
    else if (absolute <= 2.5f) index = 4;
    else if (absolute <= 3.5f) index = 5;
    else if (absolute <= 5.0f) index = 6;
    else index = 7;
    const float magnitude = FQ_E2M1_VALUES[index];
    return value < 0.0f ? -magnitude : magnitude;
}

// pow2ceil(amax / 448) as b12x's quantize_block_fp8_mx: the payload never
// exceeds the E4M3 range and the power-of-two scale is exact.
__device__ __forceinline__ float mx_scale(float amax) {
    const float raw = __fmul_rn(amax, 1.0f / E4M3_MAX);
    int exponent;
    const float mantissa = frexpf(raw, &exponent);  // raw = mantissa * 2^exponent, mantissa in [0.5, 1)
    int k = (mantissa == 0.5f) ? exponent - 1 : exponent;
    k = max(-127, min(127, k));
    return ldexpf(1.0f, k);
}

__device__ __forceinline__ float block_absmax(const __nv_bfloat16* src, unsigned n) {
    float amax = 0.0f;
    for (unsigned i = 0; i < n; ++i) amax = fmaxf(amax, fabsf(__bfloat162float(src[i])));
    return amax;
}

__device__ float cta_max(float value, float* scratch) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    __syncthreads();  // scratch may still be read by the previous row
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    float result = 0.0f;
    const unsigned warps = (blockDim.x + 31u) >> 5;
    for (unsigned w = 0; w < warps; ++w) result = fmaxf(result, scratch[w]);
    return result;
}

// NVFP4 global scale exactly as quantize_bf16_to_nvfp4_atlas_128x4_from_absmax.
__device__ __forceinline__ float nvfp4_scale2(float maximum) {
    return maximum > 0.0f ? __fdiv_rn(maximum, 2688.0f) : 1.0f;
}

__device__ __forceinline__ void nvfp4_block(
    const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned n, float scale2
) {
    const float group_max = block_absmax(src, n);
    const float inverse_scale2 = __fdiv_rn(1.0f, scale2);
    const float sf_float = group_max > 0.0f
        ? __fdiv_rn(__fmul_rn(group_max, inverse_scale2), 6.0f)
        : 0.0f;
    const float decoded = atlas_decode_fp8_e4m3(atlas_float_to_fp8_e4m3(sf_float));
    const float effective = __fmul_rn(decoded, scale2);
    const float inverse_effective = effective > 0.0f ? __fdiv_rn(1.0f, effective) : 0.0f;
    for (unsigned i = 0; i < n; ++i) {
        const float x = __bfloat162float(src[i]);
        const float q = atlas_e2m1_round(__fmul_rn(x, inverse_effective));
        dst[i] = __float2bfloat16_rn(__fmul_rn(q, effective));
    }
}

__device__ __forceinline__ void fp8_block(
    const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned n, float amax
) {
    if (!(amax > 0.0f)) {
        for (unsigned i = 0; i < n; ++i) dst[i] = __float2bfloat16_rn(0.0f);
        return;
    }
    const float scale = __fdiv_rn(amax, E4M3_MAX);
    const float inverse = __fdiv_rn(1.0f, scale);
    for (unsigned i = 0; i < n; ++i) {
        const float q = e4m3_round(__fmul_rn(__bfloat162float(src[i]), inverse));
        dst[i] = __float2bfloat16_rn(__fmul_rn(q, scale));
    }
}

__device__ __forceinline__ void mxfp8_block(
    const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned n
) {
    const float amax = block_absmax(src, n);
    if (!(amax > 0.0f)) {
        for (unsigned i = 0; i < n; ++i) dst[i] = __float2bfloat16_rn(0.0f);
        return;
    }
    const float scale = mx_scale(amax);
    const float inverse = __fdiv_rn(1.0f, scale);  // exact: power of two
    for (unsigned i = 0; i < n; ++i) {
        const float q = e4m3_round(__fmul_rn(__bfloat162float(src[i]), inverse));
        dst[i] = __float2bfloat16_rn(__fmul_rn(q, scale));
    }
}

}  // namespace

// Grid-stride absolute maximum of `count` BF16 values into `out_bits` (the
// float bit pattern; non-negative floats order like unsigned ints). The
// caller zeroes `out_bits` first.
extern "C" __global__ void qwen4_fakequant_absmax(
    const __nv_bfloat16* input,
    unsigned long long count,
    unsigned int* out_bits
) {
    __shared__ float scratch[32];
    float amax = 0.0f;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        amax = fmaxf(amax, fabsf(__bfloat162float(input[i])));
    }
    amax = cta_max(amax, scratch);
    if (threadIdx.x == 0) atomicMax(out_bits, __float_as_uint(amax));
}

// One CTA per row (grid-stride). `input` may equal `output` (in place): each
// element is read before it is written by the same thread, and per-row
// maxima are reduced before any element of that row is written.
extern "C" __global__ void qwen4_fakequant_rows(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int mode,
    unsigned int block,
    const unsigned int* tensor_absmax_bits
) {
    __shared__ float scratch[32];
    for (unsigned row = blockIdx.x; row < rows; row += gridDim.x) {
        const __nv_bfloat16* src = input + static_cast<unsigned long long>(row) * cols;
        __nv_bfloat16* dst = output + static_cast<unsigned long long>(row) * cols;

        if (mode == MODE_FP8 && (block == 0 || block >= cols)) {
            // Per-token dynamic FP8: one scale for the whole row.
            float amax = 0.0f;
            for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) {
                amax = fmaxf(amax, fabsf(__bfloat162float(src[i])));
            }
            amax = cta_max(amax, scratch);
            if (!(amax > 0.0f)) {
                for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) dst[i] = __float2bfloat16_rn(0.0f);
                continue;
            }
            const float scale = __fdiv_rn(amax, E4M3_MAX);
            const float inverse = __fdiv_rn(1.0f, scale);
            for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) {
                const float q = e4m3_round(__fmul_rn(__bfloat162float(src[i]), inverse));
                dst[i] = __float2bfloat16_rn(__fmul_rn(q, scale));
            }
            continue;
        }

        if (mode == MODE_E2M1_RAW) {
            for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) {
                dst[i] = __float2bfloat16_rn(atlas_e2m1_round(__bfloat162float(src[i])));
            }
            continue;
        }

        float scale2 = 1.0f;
        if (mode == MODE_NVFP4) {
            float maximum;
            if (tensor_absmax_bits != nullptr) {
                maximum = __uint_as_float(tensor_absmax_bits[0]);
            } else {
                maximum = 0.0f;
                for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) {
                    maximum = fmaxf(maximum, fabsf(__bfloat162float(src[i])));
                }
                maximum = cta_max(maximum, scratch);
            }
            scale2 = nvfp4_scale2(maximum);
        }

        const unsigned width = mode == MODE_NVFP4 ? 16u : mode == MODE_MXFP8 ? 32u : block;
        const unsigned blocks = (cols + width - 1) / width;
        for (unsigned b = threadIdx.x; b < blocks; b += blockDim.x) {
            const unsigned base = b * width;
            const unsigned n = min(width, cols - base);
            if (mode == MODE_NVFP4) {
                nvfp4_block(src + base, dst + base, n, scale2);
            } else if (mode == MODE_MXFP8) {
                mxfp8_block(src + base, dst + base, n);
            } else {
                fp8_block(src + base, dst + base, n, block_absmax(src + base, n));
            }
        }
    }
}
