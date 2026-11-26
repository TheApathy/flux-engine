// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen4 C2+ decode frame route: native C2-C256 (`QWEN4_NATIVE_MAX_ROWS`)
//! batching by default, or the serialized C1-per-row path with a typed
//! `SerialReason` (`crate::qwen4_batch_policy`). Decided before any mutation,
//! so a serial frame is never a retry of failed native work.

use anyhow::Result;

use super::super::types::TransformerModel;
use super::decode_qwen4_native_batch::{
    qwen4_native_batch_request_eligible, qwen4_native_batch_runtime_eligible,
    qwen4_native_batch_selected,
};
use crate::qwen4_batch_policy::{self, FrameRoute, NativeBatchSelector, ProcessFacts, RowFacts};
use crate::traits::SequenceState;

impl TransformerModel {
    /// The process-wide Qwen4 batch plan, recomputed per frame from fixed
    /// facts so the load log, scheduler counters and this route agree.
    pub(super) fn qwen4_batch_plan_dispatch(&self) -> Result<qwen4_batch_policy::Qwen4BatchPlan> {
        let selector = NativeBatchSelector::from_env().map_err(anyhow::Error::msg)?;
        let facts = ProcessFacts {
            lm_head_nvfp4: self.lm_head_nvfp4.is_some(),
            ..ProcessFacts::from_env(
                self.config.is_qwen4_exp(),
                !qwen4_native_batch_runtime_eligible(self.profile),
                self.config.vocab_size,
                self.config.qwen4_qsa,
            )
        };
        let qsa_rows = qwen4_batch_policy::qsa_rows_requested_from_env()?;
        qwen4_batch_policy::resolve_plan(selector, &facts, qsa_rows).map_err(anyhow::Error::msg)
    }

    /// Decide native versus serial for one Qwen4 C2+ decode frame before any
    /// mutation. A serial frame is never a fallback from failed native work:
    /// the per-width native capability (`decode_qwen4_capability.rs`) is part
    /// of the decision, so the native dispatch never fails a frame that could
    /// have run serially.
    pub(super) fn qwen4_frame_route(&self, seqs: &[&mut SequenceState]) -> Result<FrameRoute> {
        let plan = self.qwen4_batch_plan_dispatch()?;
        // With native batching on, widths above `QWEN4_NATIVE_MAX_ROWS` (256)
        // fail closed here (load refuses them first, `admit_width`). The gate
        // also resolves lane 03's QSA-row mode against the effective plan.
        let native_width = qwen4_native_batch_selected(&plan, seqs.len())?;
        let facts: Vec<RowFacts> = seqs
            .iter()
            .map(|seq| RowFacts {
                qsa_required: seq.qwen4_qsa_required,
                seq_len: seq.seq_len,
            })
            .collect();
        let route = qwen4_batch_policy::route_frame(&plan, &facts)?;
        // The typed policy and the reviewed eligibility predicate must agree.
        debug_assert_eq!(
            matches!(route, FrameRoute::Native),
            native_width && qwen4_native_batch_request_eligible(seqs)
        );
        if !matches!(route, FrameRoute::Native) {
            return Ok(route);
        }
        match self.qwen4_native_capability(seqs.len()) {
            Ok(()) => Ok(route),
            Err(gap) => qwen4_batch_policy::native_gap_route(&plan, seqs.len(), &gap)
                .map_err(anyhow::Error::msg),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::layers::qwen4_qsa::native_batch::{NativeQsaMode, native_attn_route};
    use crate::qwen4_batch_policy::{RowFacts, row_serial_reason};

    // Source-only (no cargo build at 8b4a4d11). `SerialReason::QsaRequired` and
    // `QsaBoundary` exactly where the per-row QSA route refuses a row.
    #[test]
    fn typed_row_reason_is_none_exactly_where_the_qsa_route_admits() {
        let lengths = [
            0,
            2_046,
            2_047,
            2_048,
            9_000,
            u32::MAX as usize - 1,
            u32::MAX as usize,
            usize::MAX,
        ];
        for mode in [NativeQsaMode::Off, NativeQsaMode::PerRow] {
            for qsa_required in [false, true] {
                for seq_len in lengths {
                    let row = RowFacts {
                        qsa_required,
                        seq_len,
                    };
                    assert_eq!(
                        row_serial_reason(row, mode == NativeQsaMode::PerRow).is_none(),
                        native_attn_route(mode, qsa_required, seq_len).is_ok(),
                        "{mode:?} qsa_required={qsa_required} seq_len={seq_len}"
                    );
                }
            }
        }
    }
}
