// SPDX-License-Identifier: AGPL-3.0-only

// Flash-Next single-sequence decode runs gated_delta_rule_decode_f32 from the
// qwen3.8-27b family (gated_delta_rule.cu in this directory includes it), and
// that kernel applies no recurrent-state norm clamp. The cross-request C2..C32
// lane runs the common multi-seq twin, which by default rescales every value
// head's FP32 state to Frobenius norm SSM_STATE_MAX_NORM after each step.
// Compile the twin without the clamp so both lanes advance the identical FP32
// recurrence: with --fmad=false the two bodies are then arithmetic-identical
// (same gate clamp, same left-to-right four-term accumulation, same
// g*H + k*v_new update, same q_dot * rsqrtf(k_dim) output).
//
// This file shadows kernels/gb10/common/gated_delta_rule_f32_multi_seq.cu by
// stem; atlas-kernels/build.rs does not notice a new .cu on a warm target dir
// unless build.rs itself is touched.
#define ATLAS_GDN_MULTI_SEQ_NO_STATE_NORM 1
#include "../../common/gated_delta_rule_f32_multi_seq.cu"
