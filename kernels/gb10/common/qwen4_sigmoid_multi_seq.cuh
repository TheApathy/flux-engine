// SPDX-License-Identifier: AGPL-3.0-only

#ifndef ATLAS_QWEN4_SIGMOID_MULTI_SEQ_CUH
#define ATLAS_QWEN4_SIGMOID_MULTI_SEQ_CUH

// Include after qwen4_gated_rms_sigmoid_body. Keep its per-head reduction,
// expf, multiplication/division order and BF16 boundary unchanged. Strides
// are elements of each pointer's type, not bytes; grid.x=head, grid.y=row,
// grid.z=seq.
//
// Overloaded `input` ABI:
//   - gridDim.z == 1: `input` is `const float*`, row offset = blockIdx.y * input_stride.
//   - gridDim.z >  1: `input` is reinterpreted as `const float* const*` (per-seq
//     pointer table for cross-seq verify), indexed by `blockIdx.z` with base
//     offset `(input_stride - output_stride) + blockIdx.y * input_stride`.
//     Precondition when gridDim.z > 1: `input_stride >= output_stride` (unsigned
//     subtraction; caller must guarantee no underflow).
extern "C" __global__ void qwen4_gated_rms_norm_sigmoid_f32_multi_seq(
    const float* input,
    const __nv_bfloat16* gate,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    unsigned int head_dim,
    float eps,
    unsigned int input_stride,
    unsigned int gate_stride,
    unsigned int output_stride) {
    const unsigned long long row = (unsigned long long)blockIdx.z * gridDim.y + blockIdx.y;
    const float* seq_in = (gridDim.z > 1u)
        ? (((const float* const*)input)[blockIdx.z] + (input_stride - output_stride)
           + (unsigned long long)blockIdx.y * input_stride)
        : (input + (unsigned long long)blockIdx.y * input_stride);
    qwen4_gated_rms_sigmoid_body(
        seq_in, gate + row * gate_stride, weight,
        output + row * output_stride, head_dim, eps, head_dim);
}

#endif
