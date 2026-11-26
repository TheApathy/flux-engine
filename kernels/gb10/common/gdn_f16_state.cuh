// SPDX-License-Identifier: AGPL-3.0-only

// FP16 h-state storage helpers shared by the five GDN FP16 WY kernels
// (`ATLAS_SSM_H_FP16`, stage 2):
//   - Wired: `gated_delta_rule_wy_f16.cu`, `gated_delta_rule_wy3_f16.cu`,
//     `gated_delta_rule_wy4_f16.cu` (via `ops::gdn_decode_wy{2,3,4}_f16`).
//   - Unwired prototypes: `gated_delta_rule_wy2_resident_f16.cu`,
//     `gated_delta_rule_wy3_resident_f16.cu` (see `README-f16-ssm-h.md`).
//
// STORAGE-ONLY NARROWING. Arithmetic stays FP32 in registers while the
// h-state and rollback intermediates are stored as `__half` (loaded via
// `__half2float` and stored via `gdn_f16_store`). Each state update is
// rounded through FP16 at the token boundary so rollback checkpoints match
// the carried forward state bit-for-bit.
//
// LAYOUT. The pool stays FP32-sized (prefill still writes FP32), and the FP16
// state occupies the first half of each slot, densely packed — matching
// `ssm_h_state_f32_to_f16`. Within a slot the head offset is `vh * k_dim * v_dim`
// elements. The FP16 WY kernels accept an extra `state_is_table` parameter
// (absent from the FP32 `common/gated_delta_rule_wy{,3,4}.cu` parents):
//   - `state_is_table == 0`: contiguous bases indexed by `(b * num_v_heads + vh)`
//     (valid only at `batch_size == 1` under FP16 pool slot strides).
//   - `state_is_table == 1`: device pointer tables of `batch_size` entries.

#ifndef ATLAS_GDN_F16_STATE_CUH
#define ATLAS_GDN_F16_STATE_CUH

#include <cuda_fp16.h>

// Saturating FP32 -> FP16 h-state store.
//
// Clamps to finite FP16 bounds [-65504, 65504] before narrowing so extreme
// state values degrade smoothly instead of overflowing to +/-inf and
// poisoning subsequent `hk_dot` accumulations with NaN. (Note: the FP16 WY
// kernels themselves do not execute a per-head Frobenius state-norm clamp.)
__device__ __forceinline__ __half gdn_f16_store(float v) {
    return __float2half(fminf(fmaxf(v, -65504.0f), 65504.0f));
}

#endif  // ATLAS_GDN_F16_STATE_CUH
