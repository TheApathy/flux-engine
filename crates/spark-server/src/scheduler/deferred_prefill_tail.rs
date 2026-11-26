// SPDX-License-Identifier: AGPL-3.0-only

//! K8 item A (`ATLAS_MTP_LASTK_DEFER`): run the prefill tail the model deferred past
//! the first token (the MTP last-K prompt replay, ~32 ms per request).
//!
//! Called by every route that emits a first token for a sequence that keeps decoding
//! (`prefill_a_step`, `prefill_b_step`, `promote_completed_prefills`), after the
//! first-token send and the TTFT stamp (`decode_start = now`), before the sequence is
//! handed to decode or anything else touches the model for it. A no-op unless the model
//! deferred something for this sequence. `params`, `suppress_ids` and `top_logprobs` are
//! the arguments the first token was sampled with (history was empty), so the model can
//! redo the device sampler's scratch write when that branch ran.

use super::*;

pub(super) fn finish_deferred_prefill_tail(
    model: &dyn Model,
    seq: &mut SequenceState,
    first: u32,
    params: &SamplingParams,
    suppress_ids: &[u32],
    top_logprobs: Option<u8>,
) -> Result<()> {
    let device_first =
        sample_uses_device_argmax(params, suppress_ids, &[], top_logprobs).then_some(first);
    model.finish_deferred_prefill_tail(seq, device_first, suppress_ids)
}
