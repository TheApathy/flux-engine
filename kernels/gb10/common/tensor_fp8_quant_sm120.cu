// SPDX-License-Identifier: AGPL-3.0-only
//
// Scalar Tensor-FP8 (E4M3FN) activation quantization kernels for Blackwell
// (SM121a/SM121f/SM120a/SM103a).
//
// Provides three entrypoints:
// 1. `tensor_fp8_absmax_bf16_sm120`:
//    Vectorized global tensor absmax over `[rows, cols]` BF16 input into a
//    zero-initialized device FP32 slot (ignores non-finite lanes).
// 2. `bf16_to_tensor_fp8_sm120`:
//    Quantizes `[rows, cols]` BF16 input to `[rows, cols]` E4M3FN with a
//    scalar tensor scale (either derived on-device from `d_absmax_in` with
//    zero host sync or taken from `static_scale`), strict `[-448.0, 448.0]`
//    clamping, and guaranteed zero `0x7F`/`0xFF` NaN bytes.
// 3. `rms_norm_tensor_fp8_sm120`:
//    Fused RMSNorm (`input_bf16[rows, cols]` + `weight_bf16[cols]` with
//    `weight_offset` and `eps`) -> explicit inline PTX `cvt.rn.bf16.f32`
//    BF16 boundary rounding -> scalar Tensor-FP8 E4M3FN quantization, with
//    optional BF16 output mirroring and device scale writeback for
//    `cublaslt::fp8_gemm_device_scaled`.
//
// Safety and ISA invariants:
// - All row/element offsets use signed `int64_t` arithmetic (no 32-bit overflow).
// - All device pointers are `__restrict__` (callers enforce disjoint ranges).
// - Zero `.local` stack frame, zero register spills, zero STL/LDL, and zero
//   odd `desc[UR]` across `sm_121a`, `sm_121f`, `sm_120a`, and `sm_103a`.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWarpsPerBlock = kBlockThreads / 32;
constexpr float kTensorFp8E4m3Max = 448.0f;
constexpr float kTensorFp8MinScale = 1.0e-12f;

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

// Explicit IEEE-754 RNE FP32 -> BF16 conversion via inline PTX so BF16
// rounding boundaries never drift under compiler fp_contract reassociation.
__device__ __forceinline__ uint16_t f32_to_bf16_rn_bits(float v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  uint16_t out;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(out) : "f"(v));
  return out;
#else
  uint32_t u = __float_as_uint(v);
  uint32_t exp = u & 0x7F800000u;
  uint32_t mant = u & 0x007FFFFFu;
  if (exp == 0x7F800000u) {
    if (mant != 0u) {
      return static_cast<uint16_t>((u >> 16) | 0x0040u);
    }
    return static_cast<uint16_t>(u >> 16);
  }
  uint32_t lsb = (u >> 16) & 1u;
  uint32_t rounded = u + 0x00007FFFu + lsb;
  return static_cast<uint16_t>(rounded >> 16);
#endif
}

__device__ __forceinline__ float sanitize_and_clamp_e4m3(float v) {
  if (!isfinite(v)) {
    return 0.0f;
  }
  return fminf(fmaxf(v, -kTensorFp8E4m3Max), kTensorFp8E4m3Max);
}

// Defense-in-depth scrub: E4M3FN NaN encodings (0x7F, 0xFF) and -0.0 (0x80)
// are canonicalized to +0.0 (0x00) so downstream tensor-core FP8 GEMMs never
// observe a NaN byte.
__device__ __forceinline__ uint8_t scrub_e4m3_byte(uint8_t b) {
  if ((b & 0x7Fu) == 0x7Fu || b == 0x80u) {
    return 0x00u;
  }
  return b;
}

#if !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
// Exact IEEE-754 RNE scalar fallback for E4M3FN when compiled below SM89.
__device__ __forceinline__ uint8_t f32_to_e4m3_scalar_rne(float clamped) {
  const uint32_t bits = __float_as_uint(clamped);
  const uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
  const float ax = fabsf(clamped);
  if (ax == 0.0f) {
    return 0x00u;
  }
  if (ax >= kTensorFp8E4m3Max) {
    return static_cast<uint8_t>(sign | 0x7Eu);
  }
  const uint32_t abits = bits & 0x7FFFFFFFu;
  const int exp_unbiased = static_cast<int>((abits >> 23) & 0xFFu) - 127;
  const uint32_t mant23 = abits & 0x007FFFFFu;
  if (exp_unbiased < -6) {
    // Subnormal E4M3FN domain: spacing is 2^-9.
    const float scaled = ax * 512.0f;
    const uint32_t q = static_cast<uint32_t>(rintf(scaled));
    if (q == 0u) {
      return 0x00u;
    }
    if (q >= 8u) {
      return static_cast<uint8_t>(sign | 0x08u);
    }
    return static_cast<uint8_t>(sign | static_cast<uint8_t>(q));
  }
  const int e4 = exp_unbiased + 7;
  // Round 23-bit mantissa to 3 bits (round-to-nearest-even).
  const uint32_t mant_hi = mant23 >> 20;
  const uint32_t rem = mant23 & 0x000FFFFFu;
  uint32_t rounded_m = mant_hi;
  if (rem > 0x00080000u || (rem == 0x00080000u && (mant_hi & 1u) != 0u)) {
    rounded_m += 1u;
  }
  int final_e = e4;
  if (rounded_m == 8u) {
    rounded_m = 0u;
    final_e += 1;
  }
  if (final_e > 15 || (final_e == 15 && rounded_m > 6u)) {
    return static_cast<uint8_t>(sign | 0x7Eu);
  }
  return static_cast<uint8_t>(
      sign | static_cast<uint8_t>((final_e << 3) | static_cast<int>(rounded_m)));
}
#endif

__device__ __forceinline__ uint16_t f32x2_to_e4m3x2_strict(float lo_val, float hi_val) {
  const float clo = sanitize_and_clamp_e4m3(lo_val);
  const float chi = sanitize_and_clamp_e4m3(hi_val);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  uint16_t packed;
  // PTX cvt.rn.satfinite.e4m3x2.f32 d, a, b places b in low byte and a in high byte.
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(chi), "f"(clo));
  const uint8_t lo = scrub_e4m3_byte(static_cast<uint8_t>(packed & 0xFFu));
  const uint8_t hi = scrub_e4m3_byte(static_cast<uint8_t>(packed >> 8));
  return static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
#else
  const uint8_t lo = scrub_e4m3_byte(f32_to_e4m3_scalar_rne(clo));
  const uint8_t hi = scrub_e4m3_byte(f32_to_e4m3_scalar_rne(chi));
  return static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
#endif
}

__device__ __forceinline__ uint8_t f32_to_e4m3_strict(float val) {
  const uint16_t pair = f32x2_to_e4m3x2_strict(val, 0.0f);
  return static_cast<uint8_t>(pair & 0xFFu);
}

__device__ __forceinline__ float resolve_tensor_fp8_scale(
    const float* __restrict__ d_absmax_in,
    float static_scale) {
  float scale = static_scale;
  if (d_absmax_in != nullptr) {
    const float amax = *d_absmax_in;
    if (isfinite(amax) && amax > 0.0f) {
      scale = fmaxf(amax / kTensorFp8E4m3Max, kTensorFp8MinScale);
    } else {
      scale = 1.0f;
    }
  }
  if (!isfinite(scale) || scale <= 0.0f) {
    scale = 1.0f;
  }
  return scale;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, offset));
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
  }
  return val;
}

__device__ __forceinline__ void atomic_max_nonneg_f32(float* __restrict__ addr, float val) {
  if (val > 0.0f && isfinite(val)) {
    atomicMax(reinterpret_cast<unsigned int*>(addr), __float_as_uint(val));
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. tensor_fp8_absmax_bf16_sm120
// ---------------------------------------------------------------------------
// Computes global finite absmax of `input_bf16[rows, cols]` into `*d_absmax_out`.
// Caller must zero `d_absmax_out` before launch.
extern "C" __global__ __launch_bounds__(kBlockThreads) void tensor_fp8_absmax_bf16_sm120(
    const uint16_t* __restrict__ input_bf16,
    float* __restrict__ d_absmax_out,
    uint32_t rows,
    uint32_t cols) {
  if (rows == 0u || cols == 0u || input_bf16 == nullptr || d_absmax_out == nullptr) {
    return;
  }
  const int64_t total = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);

  float local_max = 0.0f;
  const bool vec4_ok =
      ((reinterpret_cast<uintptr_t>(input_bf16) & 7u) == 0u) && ((total & 3LL) == 0LL);

  if (vec4_ok) {
    const uint2* __restrict__ in_v4 = reinterpret_cast<const uint2*>(input_bf16);
    const int64_t total_v4 = total >> 2;
    for (int64_t idx = tid; idx < total_v4; idx += stride) {
      const uint2 packed = in_v4[idx];
      const float v0 = bf16_bits_to_f32(static_cast<uint16_t>(packed.x & 0xFFFFu));
      const float v1 = bf16_bits_to_f32(static_cast<uint16_t>(packed.x >> 16));
      const float v2 = bf16_bits_to_f32(static_cast<uint16_t>(packed.y & 0xFFFFu));
      const float v3 = bf16_bits_to_f32(static_cast<uint16_t>(packed.y >> 16));
      if (isfinite(v0)) local_max = fmaxf(local_max, fabsf(v0));
      if (isfinite(v1)) local_max = fmaxf(local_max, fabsf(v1));
      if (isfinite(v2)) local_max = fmaxf(local_max, fabsf(v2));
      if (isfinite(v3)) local_max = fmaxf(local_max, fabsf(v3));
    }
  } else {
    for (int64_t idx = tid; idx < total; idx += stride) {
      const float v = bf16_bits_to_f32(input_bf16[idx]);
      if (isfinite(v)) {
        local_max = fmaxf(local_max, fabsf(v));
      }
    }
  }

  local_max = warp_reduce_max(local_max);
  __shared__ float warp_max[kWarpsPerBlock];
  const int lane = static_cast<int>(threadIdx.x & 31u);
  const int warp = static_cast<int>(threadIdx.x >> 5);
  if (lane == 0) {
    warp_max[warp] = local_max;
  }
  __syncthreads();

  if (warp == 0) {
    float v = (lane < kWarpsPerBlock) ? warp_max[lane] : 0.0f;
    v = warp_reduce_max(v);
    if (lane == 0) {
      atomic_max_nonneg_f32(d_absmax_out, v);
    }
  }
}

// ---------------------------------------------------------------------------
// 2. bf16_to_tensor_fp8_sm120
// ---------------------------------------------------------------------------
// Quantizes `input_bf16[rows, cols]` to scalar Tensor-FP8 E4M3FN `out_fp8[rows, cols]`.
// Scale resolution:
// - `d_absmax_in != nullptr`: `scale = max((*d_absmax_in) / 448.0f, 1e-12f)` (or 1.0f if 0/non-finite).
// - `d_absmax_in == nullptr`: `scale = static_scale` (or 1.0f if <= 0 / non-finite).
// When `d_scale_out != nullptr`, CTA 0 lane 0 writes `*d_scale_out = scale`.
extern "C" __global__ __launch_bounds__(kBlockThreads) void bf16_to_tensor_fp8_sm120(
    const uint16_t* __restrict__ input_bf16,
    uint8_t* __restrict__ out_fp8,
    const float* __restrict__ d_absmax_in,
    float* __restrict__ d_scale_out,
    float static_scale,
    uint32_t rows,
    uint32_t cols) {
  if (rows == 0u || cols == 0u || input_bf16 == nullptr || out_fp8 == nullptr) {
    return;
  }
  const float scale = resolve_tensor_fp8_scale(d_absmax_in, static_scale);
  if (blockIdx.x == 0u && threadIdx.x == 0u && d_scale_out != nullptr) {
    *d_scale_out = scale;
  }
  const float inv_scale = 1.0f / scale;

  const int64_t total = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);

  const bool vec4_ok = ((reinterpret_cast<uintptr_t>(input_bf16) & 7u) == 0u) &&
                       ((reinterpret_cast<uintptr_t>(out_fp8) & 3u) == 0u) &&
                       ((total & 3LL) == 0LL);

  if (vec4_ok) {
    const uint2* __restrict__ in_v4 = reinterpret_cast<const uint2*>(input_bf16);
    uint32_t* __restrict__ out_v4 = reinterpret_cast<uint32_t*>(out_fp8);
    const int64_t total_v4 = total >> 2;
    for (int64_t idx = tid; idx < total_v4; idx += stride) {
      const uint2 packed = in_v4[idx];
      const float v0 = bf16_bits_to_f32(static_cast<uint16_t>(packed.x & 0xFFFFu)) * inv_scale;
      const float v1 = bf16_bits_to_f32(static_cast<uint16_t>(packed.x >> 16)) * inv_scale;
      const float v2 = bf16_bits_to_f32(static_cast<uint16_t>(packed.y & 0xFFFFu)) * inv_scale;
      const float v3 = bf16_bits_to_f32(static_cast<uint16_t>(packed.y >> 16)) * inv_scale;
      const uint16_t p0 = f32x2_to_e4m3x2_strict(v0, v1);
      const uint16_t p1 = f32x2_to_e4m3x2_strict(v2, v3);
      out_v4[idx] = static_cast<uint32_t>(p0) | (static_cast<uint32_t>(p1) << 16);
    }
  } else {
    for (int64_t idx = tid; idx < total; idx += stride) {
      const float v = bf16_bits_to_f32(input_bf16[idx]) * inv_scale;
      out_fp8[idx] = f32_to_e4m3_strict(v);
    }
  }
}

// ---------------------------------------------------------------------------
// 3. rms_norm_tensor_fp8_sm120
// ---------------------------------------------------------------------------
// Fused RMSNorm + scalar Tensor-FP8 E4M3FN quantization (one CTA per row).
// Each normalized value is rounded across the BF16 boundary via explicit
// inline PTX `cvt.rn.bf16.f32` before scaling and strict `[-448.0, 448.0]`
// E4M3FN quantization so output parity matches standalone `rms_norm` +
// `bf16_to_tensor_fp8_sm120` bit-for-bit.
extern "C" __global__ __launch_bounds__(kBlockThreads) void rms_norm_tensor_fp8_sm120(
    const uint16_t* __restrict__ input_bf16,
    const uint16_t* __restrict__ weight_bf16,
    uint8_t* __restrict__ out_fp8,
    uint16_t* __restrict__ out_bf16_opt,
    const float* __restrict__ d_absmax_in,
    float* __restrict__ d_scale_out,
    float eps,
    float weight_offset,
    float static_scale,
    uint32_t rows,
    uint32_t cols) {
  const uint32_t row = blockIdx.x;
  if (row >= rows || cols == 0u || input_bf16 == nullptr || weight_bf16 == nullptr ||
      out_fp8 == nullptr) {
    return;
  }

  const float scale = resolve_tensor_fp8_scale(d_absmax_in, static_scale);
  if (row == 0u && threadIdx.x == 0u && d_scale_out != nullptr) {
    *d_scale_out = scale;
  }
  const float inv_scale = 1.0f / scale;

  const int64_t row_off = static_cast<int64_t>(row) * static_cast<int64_t>(cols);
  const uint16_t* __restrict__ row_in = input_bf16 + row_off;
  uint8_t* __restrict__ row_out_fp8 = out_fp8 + row_off;
  uint16_t* __restrict__ row_out_bf16 =
      (out_bf16_opt != nullptr) ? (out_bf16_opt + row_off) : nullptr;

  // Pass 1: accumulate FP32 sum of squares across `cols`.
  float sum_sq = 0.0f;
  const bool vec4_in_ok = ((reinterpret_cast<uintptr_t>(row_in) & 7u) == 0u) &&
                          ((reinterpret_cast<uintptr_t>(weight_bf16) & 7u) == 0u) &&
                          ((cols & 3u) == 0u);
  if (vec4_in_ok) {
    const uint2* __restrict__ in_v4 = reinterpret_cast<const uint2*>(row_in);
    const uint32_t cols_v4 = cols >> 2;
    for (uint32_t c = threadIdx.x; c < cols_v4; c += kBlockThreads) {
      const uint2 packed = in_v4[c];
      const float x0 = bf16_bits_to_f32(static_cast<uint16_t>(packed.x & 0xFFFFu));
      const float x1 = bf16_bits_to_f32(static_cast<uint16_t>(packed.x >> 16));
      const float x2 = bf16_bits_to_f32(static_cast<uint16_t>(packed.y & 0xFFFFu));
      const float x3 = bf16_bits_to_f32(static_cast<uint16_t>(packed.y >> 16));
      sum_sq += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    }
  } else {
    for (uint32_t c = threadIdx.x; c < cols; c += kBlockThreads) {
      const float x = bf16_bits_to_f32(row_in[c]);
      sum_sq += x * x;
    }
  }

  sum_sq = warp_reduce_sum(sum_sq);
  __shared__ float warp_sum[kWarpsPerBlock];
  const int lane = static_cast<int>(threadIdx.x & 31u);
  const int warp = static_cast<int>(threadIdx.x >> 5);
  if (lane == 0) {
    warp_sum[warp] = sum_sq;
  }
  __syncthreads();

  __shared__ float shared_inv_rms;
  if (warp == 0) {
    float total_sq = (lane < kWarpsPerBlock) ? warp_sum[lane] : 0.0f;
    total_sq = warp_reduce_sum(total_sq);
    if (lane == 0) {
      shared_inv_rms = rsqrtf(total_sq / static_cast<float>(cols) + eps);
    }
  }
  __syncthreads();
  const float inv_rms = shared_inv_rms;

  // Pass 2: normalize -> explicit cvt.rn.bf16.f32 -> scale -> strict E4M3FN.
  const bool vec4_out_ok = vec4_in_ok && ((reinterpret_cast<uintptr_t>(row_out_fp8) & 3u) == 0u) &&
                           (row_out_bf16 == nullptr ||
                            ((reinterpret_cast<uintptr_t>(row_out_bf16) & 7u) == 0u));
  if (vec4_out_ok) {
    const uint2* __restrict__ in_v4 = reinterpret_cast<const uint2*>(row_in);
    const uint2* __restrict__ w_v4 = reinterpret_cast<const uint2*>(weight_bf16);
    uint32_t* __restrict__ fp8_v4 = reinterpret_cast<uint32_t*>(row_out_fp8);
    uint2* __restrict__ bf16_v4 = reinterpret_cast<uint2*>(row_out_bf16);
    const uint32_t cols_v4 = cols >> 2;
    for (uint32_t c = threadIdx.x; c < cols_v4; c += kBlockThreads) {
      const uint2 xp = in_v4[c];
      const uint2 wp = w_v4[c];
      const float x0 = bf16_bits_to_f32(static_cast<uint16_t>(xp.x & 0xFFFFu));
      const float x1 = bf16_bits_to_f32(static_cast<uint16_t>(xp.x >> 16));
      const float x2 = bf16_bits_to_f32(static_cast<uint16_t>(xp.y & 0xFFFFu));
      const float x3 = bf16_bits_to_f32(static_cast<uint16_t>(xp.y >> 16));
      const float w0 = bf16_bits_to_f32(static_cast<uint16_t>(wp.x & 0xFFFFu)) + weight_offset;
      const float w1 = bf16_bits_to_f32(static_cast<uint16_t>(wp.x >> 16)) + weight_offset;
      const float w2 = bf16_bits_to_f32(static_cast<uint16_t>(wp.y & 0xFFFFu)) + weight_offset;
      const float w3 = bf16_bits_to_f32(static_cast<uint16_t>(wp.y >> 16)) + weight_offset;

      const uint16_t b0 = f32_to_bf16_rn_bits(x0 * inv_rms * w0);
      const uint16_t b1 = f32_to_bf16_rn_bits(x1 * inv_rms * w1);
      const uint16_t b2 = f32_to_bf16_rn_bits(x2 * inv_rms * w2);
      const uint16_t b3 = f32_to_bf16_rn_bits(x3 * inv_rms * w3);

      if (bf16_v4 != nullptr) {
        uint2 out_b;
        out_b.x = static_cast<uint32_t>(b0) | (static_cast<uint32_t>(b1) << 16);
        out_b.y = static_cast<uint32_t>(b2) | (static_cast<uint32_t>(b3) << 16);
        bf16_v4[c] = out_b;
      }

      const uint16_t q0 =
          f32x2_to_e4m3x2_strict(bf16_bits_to_f32(b0) * inv_scale, bf16_bits_to_f32(b1) * inv_scale);
      const uint16_t q1 =
          f32x2_to_e4m3x2_strict(bf16_bits_to_f32(b2) * inv_scale, bf16_bits_to_f32(b3) * inv_scale);
      fp8_v4[c] = static_cast<uint32_t>(q0) | (static_cast<uint32_t>(q1) << 16);
    }
  } else {
    for (uint32_t c = threadIdx.x; c < cols; c += kBlockThreads) {
      const float x = bf16_bits_to_f32(row_in[c]);
      const float w = bf16_bits_to_f32(weight_bf16[c]) + weight_offset;
      const uint16_t b = f32_to_bf16_rn_bits(x * inv_rms * w);
      if (row_out_bf16 != nullptr) {
        row_out_bf16[c] = b;
      }
      row_out_fp8[c] = f32_to_e4m3_strict(bf16_bits_to_f32(b) * inv_scale);
    }
  }
}
