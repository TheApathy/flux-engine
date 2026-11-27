// SPDX-License-Identifier: AGPL-3.0-only

//! Small-batch MoE decoding and verification routing contracts (Milestone 2).

use super::*;

/// Crossover threshold between scalar GEMV / exact-rows and grouped GEMM.
/// Microbenchmarking confirms scalar GEMV / exact-rows is 2.0x - 15.9x faster for M <= 16,
/// while tensor-core grouped GEMM incurs a flat ~260 µs padding/launch floor.
pub const SMALL_BATCH_MOE_THRESHOLD: usize = 16;

/// Extension trait for small-batch MoE decode and verification routing.
pub trait MoeSmallBatchRoute {
    fn is_small_batch(&self, num_tokens: usize) -> bool;
}

impl MoeSmallBatchRoute for MoeLayer {
    #[inline]
    fn is_small_batch(&self, num_tokens: usize) -> bool {
        is_small_batch_tokens(num_tokens)
    }
}

#[inline]
pub(crate) const fn is_small_batch_tokens(num_tokens: usize) -> bool {
    num_tokens <= SMALL_BATCH_MOE_THRESHOLD
}

/// Pure routing decision helper for `MoeLayer::forward_prefill` small-batch bypass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SmallBatchPrefillAction {
    /// Route M=1 directly to scalar decode `MoeLayer::forward`.
    ScalarDecodeM1,
    /// Route 2 <= M <= 16 on single-GPU unified `_t_lanes_rows` to `MoeLayer::forward_qwen4_exact`.
    ExactRows,
    /// Proceed through the full prefill / grouped-GEMM pipeline.
    GroupedPrefill,
}

#[inline]
pub(crate) const fn decide_small_batch_prefill_with_policy(
    num_tokens: usize,
    frankenstein_decode_via_prefill: bool,
    prefill_native_or_r2_active: bool,
    ep_multi_gpu: bool,
    t_lanes_rows_available: bool,
    has_nvfp4_router: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> SmallBatchPrefillAction {
    if prefill_native_or_r2_active {
        return SmallBatchPrefillAction::GroupedPrefill;
    }
    if num_tokens == 1 {
        if frankenstein_decode_via_prefill {
            return SmallBatchPrefillAction::GroupedPrefill;
        }
        return SmallBatchPrefillAction::ScalarDecodeM1;
    }
    if ep_multi_gpu {
        return SmallBatchPrefillAction::GroupedPrefill;
    }
    let within_exact_rows = if det_batch.is_enabled() {
        num_tokens <= crate::layers::deterministic_batch::DETERMINISTIC_BATCH_MAX_ROWS
    } else {
        is_small_batch_tokens(num_tokens)
    };
    if num_tokens >= 2 && within_exact_rows && t_lanes_rows_available && has_nvfp4_router {
        return SmallBatchPrefillAction::ExactRows;
    }
    SmallBatchPrefillAction::GroupedPrefill
}

#[cfg_attr(not(test), allow(dead_code))]
#[inline]
pub(crate) const fn decide_small_batch_prefill(
    num_tokens: usize,
    frankenstein_decode_via_prefill: bool,
    prefill_native_or_r2_active: bool,
    ep_multi_gpu: bool,
    t_lanes_rows_available: bool,
    has_nvfp4_router: bool,
) -> SmallBatchPrefillAction {
    decide_small_batch_prefill_with_policy(
        num_tokens,
        frankenstein_decode_via_prefill,
        prefill_native_or_r2_active,
        ep_multi_gpu,
        t_lanes_rows_available,
        has_nvfp4_router,
        crate::layers::deterministic_batch::DeterministicBatchPolicy::Disabled,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_batch_threshold_crossover_boundaries() {
        assert_eq!(SMALL_BATCH_MOE_THRESHOLD, 16);
        for m in [0, 1, 2, 4, 5, 9, 15, 16] {
            assert!(is_small_batch_tokens(m), "M={m} should be small batch");
        }
        for m in [17, 32, 64, 256, 4096] {
            assert!(
                !is_small_batch_tokens(m),
                "M={m} must cross over to grouped GEMM"
            );
        }
    }

    #[test]
    fn small_batch_prefill_decision_respects_frankenstein_ep_and_native_guards() {
        // M=0 never enters ScalarDecodeM1 or ExactRows
        assert_eq!(
            decide_small_batch_prefill(0, false, false, false, true, true),
            SmallBatchPrefillAction::GroupedPrefill
        );

        // M=1 normal -> ScalarDecodeM1; Frankenstein -> GroupedPrefill (no ping-pong recursion)
        assert_eq!(
            decide_small_batch_prefill(1, false, false, false, true, true),
            SmallBatchPrefillAction::ScalarDecodeM1
        );
        assert_eq!(
            decide_small_batch_prefill(1, true, false, false, true, true),
            SmallBatchPrefillAction::GroupedPrefill
        );

        // M in 2..=16 single-GPU with t_lanes_rows + nvfp4 router -> ExactRows
        for m in [2, 4, 5, 8, 9, 16] {
            assert_eq!(
                decide_small_batch_prefill(m, false, false, false, true, true),
                SmallBatchPrefillAction::ExactRows,
                "M={m}"
            );
            // Multi-GPU EP must bypass single-GPU ExactRows
            assert_eq!(
                decide_small_batch_prefill(m, false, false, true, true, true),
                SmallBatchPrefillAction::GroupedPrefill,
                "EP M={m}"
            );
            // Missing NVFP4 router (e.g. BF16-only router) falls through to GroupedPrefill
            assert_eq!(
                decide_small_batch_prefill(m, false, false, false, true, false),
                SmallBatchPrefillAction::GroupedPrefill,
                "no nvfp4 router M={m}"
            );
        }

        // Prefill-native or R2 receipt capture keeps full prefill pipeline even for M <= 16
        for m in [1, 5, 16] {
            assert_eq!(
                decide_small_batch_prefill(m, false, true, false, true, true),
                SmallBatchPrefillAction::GroupedPrefill,
                "native/R2 M={m}"
            );
        }

        // M >= 17 always takes GroupedPrefill under Disabled baseline
        for m in [17, 32, 256] {
            assert_eq!(
                decide_small_batch_prefill(m, false, false, false, true, true),
                SmallBatchPrefillAction::GroupedPrefill,
                "M={m}"
            );
        }

        // Under DeterministicBatchPolicy::Enabled, M=2..=128 stays on ExactRows (0-ULP to M=1).
        use crate::layers::deterministic_batch::DeterministicBatchPolicy;
        assert_eq!(
            decide_small_batch_prefill_with_policy(
                1,
                false,
                false,
                false,
                true,
                true,
                DeterministicBatchPolicy::Enabled,
            ),
            SmallBatchPrefillAction::ScalarDecodeM1
        );
        for m in [2usize, 4, 8, 16, 32, 64, 128] {
            assert_eq!(
                decide_small_batch_prefill_with_policy(
                    m,
                    false,
                    false,
                    false,
                    true,
                    true,
                    DeterministicBatchPolicy::Enabled,
                ),
                SmallBatchPrefillAction::ExactRows,
                "deterministic batch must pin ExactRows at M={m}"
            );
        }
    }
}
