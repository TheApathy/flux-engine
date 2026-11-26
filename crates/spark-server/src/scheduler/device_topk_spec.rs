// SPDX-License-Identifier: AGPL-3.0-only

//! Device top-k route for speculative verify rows and spec bootstraps
//! (`ATLAS_SAMPLER_DEVICE_TOPK=1` or `auto`). Same rule as the plain
//! path (`device_topk_route`): an eligible row samples from device
//! candidates through the unchanged `process_seq_logits` interventions; any
//! other row keeps its caller's host path. Each row is routed from the
//! sequence state at the moment it is sampled, after the earlier rows of the
//! walk have been committed, exactly as the host walk sees it.

use super::device_topk_route::{TopkRoute, device_topk_route, note_route};
use super::*;
use spark_runtime::sampler::device_topk;

/// One verify / bootstrap row as the device route handled it.
pub(super) enum DeviceRow {
    /// Not routed (route off or row ineligible): the caller's host path runs.
    Host,
    /// Sampled on the device route: the token and its (absent) logprobs.
    Sampled((u32, Option<crate::api::TokenLogprobs>)),
    /// The device failed; the caller finishes the sequence (no host fallback).
    Failed,
}

/// Fetch top-`m` candidates for one BF16 logits row into a compact O(m)
/// [`device_topk::SparseLogitRow`] without expanding a full-vocab `Vec<f32>`.
///
/// Returns `Ok(None)` when the row has fewer than `m` finite logits (the
/// sampler's empty-row fallback may need the raw non-finite entries, so the
/// caller falls back to the full row). Errors are request-fatal.
fn fetch_sparse_candidate_row(
    model: &dyn Model,
    row: DevicePtr,
    m: usize,
    temperature: f32,
) -> Result<Option<device_topk::SparseLogitRow>> {
    let candidates = model.topk_candidates(row, temperature, m)?;
    anyhow::ensure!(
        candidates.len() <= m,
        "device top-k returned {} candidates for m={m}",
        candidates.len()
    );
    if candidates.len() < m {
        return Ok(None);
    }
    device_topk::SparseLogitRow::from_candidates(model.vocab_size(), &candidates)
        .map(Some)
        .ok_or_else(|| anyhow::anyhow!("device top-k returned a token id outside the vocabulary"))
}

/// Direct O(m) counterpart to `process_seq_sparse_logits` for device-routed
/// rows (`device_topk_route` has already excluded logprobs, logits-FNV,
/// adaptive sampling, active grammar, dsv41, and thinking-efficiency shaping).
///
/// Applies every remaining plain-path intervention directly to `sparse` and
/// samples in O(m), avoiding the 1 MB full-vocab `-inf` expansion and scan.
pub(super) fn process_seq_sparse_row(
    a: &mut ActiveSeq,
    mut sparse: device_topk::SparseLogitRow,
    f2_decision: Option<bool>,
    think_end_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    reflection_suppress_ids: &[u32],
) -> (u32, Option<crate::api::TokenLogprobs>) {
    debug_assert!(a.top_logprobs.is_none());
    debug_assert!(a.inside_thinking || a.grammar_state.is_none());
    debug_assert!(!(a.inside_thinking && crate::scheduler::think_efficiency_config().is_active()));

    let hard_override = super::confidence::compute_think_defer_hard_override(
        a.thinking_tokens,
        a.thinking_budget,
        a.output_tokens.len(),
        a.max_output_tokens,
    );
    if a.inside_thinking
        && !a.force_end_thinking
        && a.thinking_budget.is_none()
        && a.max_output_tokens > 0
        && a.output_tokens.len()
            >= a.max_output_tokens
                .saturating_sub(super::confidence::min_answer_runway(a.max_output_tokens))
    {
        a.force_end_thinking = true;
    }
    let close_allowed_now = super::think_budget_nudge::close_allowed(a)
        || (a.inside_thinking
            && super::confidence::should_inject_think_end(
                a.force_end_thinking,
                a.in_code_fence,
                hard_override,
            ));
    let runway_nudge_forced = a
        .budget_nudge
        .as_ref()
        .filter(|_| {
            close_allowed_now && a.output_tokens.len().saturating_add(1) < a.max_output_tokens
        })
        .and_then(|n| n.pending_token());
    let forced = super::think_budget_nudge::forced_token(a)
        .or(runway_nudge_forced)
        .or_else(|| super::think_nudge::forced_token(a));
    if let Some(forced) = forced {
        return (forced, None);
    }

    for (id, edit) in super::decode_logits_seq::pre_f2_edits(
        a,
        think_end_token,
        reflection_suppress_ids,
        sparse.vocab_size(),
    ) {
        sparse.apply_pre_f2_edit(id, edit);
    }

    if a.inside_thinking
        && !a.force_end_thinking
        && a.thinking_tokens >= super::confidence::CONFIDENCE_EARLY_STOP_MIN_THINKING
        && crate::scheduler::helpers::watchdog_params().confidence_early_stop
    {
        let confident = f2_decision.unwrap_or_else(|| {
            let dense = sparse.to_dense();
            spark_runtime::sampler::f2_certify::host_f2_confident(&dense)
        });
        let (run, force_end) =
            super::confidence::confidence_run_step(confident, a.consecutive_confident);
        a.consecutive_confident = run;
        if force_end {
            a.force_end_thinking = true;
            tracing::info!(
                "Confidence early stop armed: top-1 prob >= 0.95 for {} tokens (after {} thinking tokens){}",
                crate::scheduler::helpers::watchdog_params().confidence_run_length,
                a.thinking_tokens,
                if a.in_code_fence {
                    " — deferred until ``` fence closes"
                } else {
                    ""
                }
            );
        }
    }

    if a.think_ended && !crate::dsv41::serving() {
        if let Some(end_tok) = think_end_token {
            sparse.set_neg_inf(end_tok);
        }
        if let Some(start_tok) = a.think_start_token {
            sparse.set_neg_inf(start_tok);
        }
    }

    if a.inside_thinking {
        if let Some(tc_start) = tool_call_start_token {
            sparse.set_neg_inf(tc_start);
        }
    } else if a.suppress_tool_call
        && let Some(tc_start) = tool_call_start_token
    {
        sparse.sub_logit(tc_start, 12.0);
    }

    let post_f2_hard_override = super::confidence::compute_think_defer_hard_override(
        a.thinking_tokens,
        a.thinking_budget,
        a.output_tokens.len(),
        a.max_output_tokens,
    );
    let post_f2_close_allowed = super::think_budget_nudge::close_allowed(a)
        || (a.inside_thinking
            && super::confidence::should_inject_think_end(
                a.force_end_thinking,
                a.in_code_fence,
                post_f2_hard_override,
            ));
    if post_f2_close_allowed && let Some(end_tok) = think_end_token {
        sparse.pin_token(end_tok, 0.0);
    }

    if a.think_just_ended
        && a.require_tool_call
        && !a.tool_call_opened
        && !a.inside_thinking
        && let Some(start_tok) = tool_call_start_token
        && sparse.pin_token(start_tok, 0.0)
    {
        tracing::debug!("Forced tool_call_start_token after </think> (require_tool_call set)");
    }

    let sampling_temp = a.temperature;
    let _gumbel_guard = (super::stochastic_verify_walk::spec_verify_policy()
        == super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
        && sampling_temp > 0.0)
        .then(|| {
            let seed = super::stochastic_verify_walk::resolve_seq_gumbel_seed(a);
            spark_runtime::sampler::gumbel::GumbelPosGuard::new(
                seed,
                super::stochastic_verify_walk::seq_abs_pos(a),
            )
        });
    let params = a.sampling_params(sampling_temp);
    let sampled = if sampling_temp > 0.0
        && !spark_runtime::sampler::gumbel::gumbel_sampling_enabled()
        && let Some((base_seed, abs_pos)) = spark_runtime::sampler::gumbel::current_gumbel_pos()
    {
        sparse.choose_gumbel_in_place(&params, &a.output_tokens, base_seed, abs_pos)
    } else {
        sparse.sample_with_params_history_in_place(&params, &a.output_tokens)
    };
    (sampled, None)
}

/// Route and, when eligible, sample the BF16 logits row at `row` for `a`.
pub(super) fn sample_spec_row(
    model: &dyn Model,
    a: &mut ActiveSeq,
    row: DevicePtr,
    logits_fp32: bool,
    ctx: &ThinkSpecCtx<'_>,
) -> DeviceRow {
    if !device_topk::device_topk_enabled() {
        return DeviceRow::Host;
    }
    let route = device_topk_route(
        a,
        logits_fp32,
        ctx.adaptive_sampling,
        ctx.reflection_suppress_ids,
    );
    let TopkRoute::Device { m, temperature, f2 } = route else {
        note_route(a, route, std::time::Duration::ZERO);
        return DeviceRow::Host;
    };
    let t0 = std::time::Instant::now();
    let sparse = match fetch_sparse_candidate_row(model, row, m, temperature) {
        Ok(Some(sparse)) => sparse,
        Ok(None) => {
            note_route(a, TopkRoute::Host("few_finite"), t0.elapsed());
            return DeviceRow::Host;
        }
        Err(e) => {
            tracing::error!("device top-k (spec row) failed: {e:#}");
            return DeviceRow::Failed;
        }
    };
    let decision = if f2 {
        let edits = super::decode_logits_seq::pre_f2_edits(
            a,
            a.think_end_token,
            ctx.reflection_suppress_ids,
            model.vocab_size(),
        );
        match super::device_topk_route::f2_decision(model, row, &edits, model.vocab_size() * 2) {
            Ok(d) => Some(d),
            Err(e) => {
                tracing::error!("certified F2 (spec row) failed: {e:#}");
                return DeviceRow::Failed;
            }
        }
    } else {
        None
    };
    let sampled = process_seq_sparse_row(
        a,
        sparse,
        decision,
        a.think_end_token,
        a.tool_call_start_token,
        ctx.reflection_suppress_ids,
    );
    note_route(a, route, t0.elapsed());
    DeviceRow::Sampled(sampled)
}

#[cfg(test)]
#[path = "device_topk_spec_tests.rs"]
mod tests;

#[cfg(test)]
mod sparse_spec_parity_tests {
    use super::*;
    use crate::scheduler::test_fixtures;
    use spark_runtime::sampler::device_topk::{SparseLogitRow, sparse_row_from_candidates};

    fn make_seq(inside_thinking: bool, seed: u64) -> ActiveSeq {
        let mut a = test_fixtures::active_seq(ResponseSink::Blocking(None), 0);
        a.temperature = 0.8;
        a.top_k = 8;
        a.top_p = 0.92;
        a.seed = Some(seed);
        a.max_output_tokens = 256;
        a.remaining = 256;
        a.inside_thinking = inside_thinking;
        a.enable_thinking = inside_thinking;
        a.think_start_token = Some(4);
        a.think_end_token = Some(5);
        a.tool_call_start_token = Some(6);
        a
    }

    #[test]
    fn sparse_row_interventions_match_dense_sparse_logits_exactly() {
        const VOCAB: usize = 128;
        const SUPPRESS: [u32; 2] = [2, 11];
        let candidates = [
            (2u32, 6.2f32),
            (4, 5.9),
            (5, 5.8),
            (6, 6.5),
            (11, 5.4),
            (17, 5.1),
            (23, 4.8),
            (31, 4.7),
            (42, 4.3),
            (55, 3.9),
            (71, 3.6),
            (99, 3.1),
        ];

        // 1. Thinking with reflection suppression + tool_call hard mask + penalties + Gumbel.
        {
            let mut a_dense = make_seq(true, 101);
            a_dense.repetition_penalty = 1.15;
            a_dense.presence_penalty = 0.25;
            a_dense.frequency_penalty = 0.15;
            a_dense.output_tokens = vec![17, 23, 17, 31];
            let mut a_sparse = make_seq(true, 101);
            a_sparse.repetition_penalty = 1.15;
            a_sparse.presence_penalty = 0.25;
            a_sparse.frequency_penalty = 0.15;
            a_sparse.output_tokens = vec![17, 23, 17, 31];

            let _g = spark_runtime::sampler::gumbel::GumbelPosGuard::new(0x1234, 77);
            let dense_row = sparse_row_from_candidates(VOCAB, &candidates).unwrap();
            let sparse_row = SparseLogitRow::from_candidates(VOCAB, &candidates).unwrap();

            let (te, tc) = (a_dense.think_end_token, a_dense.tool_call_start_token);
            let want =
                process_seq_sparse_logits(&mut a_dense, dense_row, None, te, tc, &SUPPRESS, false);
            let (te, tc) = (a_sparse.think_end_token, a_sparse.tool_call_start_token);
            let got = process_seq_sparse_row(&mut a_sparse, sparse_row, None, te, tc, &SUPPRESS);
            assert_eq!(got.0, want.0);
            assert!(got.1.is_none() && want.1.is_none());
        }

        // 2. Forced `</think>` pin on budget/force_end_thinking.
        {
            let mut a_dense = make_seq(true, 202);
            a_dense.force_end_thinking = true;
            let mut a_sparse = make_seq(true, 202);
            a_sparse.force_end_thinking = true;
            let dense_row = sparse_row_from_candidates(VOCAB, &candidates).unwrap();
            let sparse_row = SparseLogitRow::from_candidates(VOCAB, &candidates).unwrap();
            let (te, tc) = (a_dense.think_end_token, a_dense.tool_call_start_token);
            let want =
                process_seq_sparse_logits(&mut a_dense, dense_row, None, te, tc, &SUPPRESS, false);
            let (te, tc) = (a_sparse.think_end_token, a_sparse.tool_call_start_token);
            let got = process_seq_sparse_row(&mut a_sparse, sparse_row, None, te, tc, &SUPPRESS);
            assert_eq!(got.0, want.0);
            assert!(got.1.is_none() && want.1.is_none());
            assert_eq!(got.0, 5);
        }

        // 3. Post-think content with `<think>`/`</think>` masks + `-12.0` tool_call suppression.
        {
            let mut a_dense = make_seq(false, 303);
            a_dense.think_ended = true;
            a_dense.suppress_tool_call = true;
            let mut a_sparse = make_seq(false, 303);
            a_sparse.think_ended = true;
            a_sparse.suppress_tool_call = true;
            let dense_row = sparse_row_from_candidates(VOCAB, &candidates).unwrap();
            let sparse_row = SparseLogitRow::from_candidates(VOCAB, &candidates).unwrap();
            let (te, tc) = (a_dense.think_end_token, a_dense.tool_call_start_token);
            let want =
                process_seq_sparse_logits(&mut a_dense, dense_row, None, te, tc, &SUPPRESS, false);
            let (te, tc) = (a_sparse.think_end_token, a_sparse.tool_call_start_token);
            let got = process_seq_sparse_row(&mut a_sparse, sparse_row, None, te, tc, &SUPPRESS);
            assert_eq!(got.0, want.0);
            assert!(got.1.is_none() && want.1.is_none());
        }

        // 4. One-shot `require_tool_call` pin right after `</think>`.
        {
            let mut a_dense = make_seq(false, 404);
            a_dense.think_ended = true;
            a_dense.think_just_ended = true;
            a_dense.require_tool_call = true;
            let mut a_sparse = make_seq(false, 404);
            a_sparse.think_ended = true;
            a_sparse.think_just_ended = true;
            a_sparse.require_tool_call = true;
            let dense_row = sparse_row_from_candidates(VOCAB, &candidates).unwrap();
            let sparse_row = SparseLogitRow::from_candidates(VOCAB, &candidates).unwrap();
            let (te, tc) = (a_dense.think_end_token, a_dense.tool_call_start_token);
            let want =
                process_seq_sparse_logits(&mut a_dense, dense_row, None, te, tc, &SUPPRESS, false);
            let (te, tc) = (a_sparse.think_end_token, a_sparse.tool_call_start_token);
            let got = process_seq_sparse_row(&mut a_sparse, sparse_row, None, te, tc, &SUPPRESS);
            assert_eq!(got.0, want.0);
            assert!(got.1.is_none() && want.1.is_none());
            assert_eq!(got.0, 6);
        }
    }
}
