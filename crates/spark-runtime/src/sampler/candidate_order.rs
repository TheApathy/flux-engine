// SPDX-License-Identifier: AGPL-3.0-only
//
// Candidate ordering for the sampler's top-k stage.
//
// `Total` pins the order: logit descending, then token id ascending. With a
// total order, "select the k largest, then sort them" equals "sort all, then
// truncate" exactly, so the top-k stage becomes O(V) (`select_nth_unstable_by`)
// instead of O(V log V) over 248,320 vocabulary entries. This is also the
// tie-order contract implemented by device top-k and Gumbel-Max verification.
//
// `Legacy` preserves the historical full `sort_unstable_by` on logit alone when
// explicitly requested via `FLUX_SAMPLER_TIE_ORDER=legacy` (outside Gumbel mode).
//
// Default is `Total` whenever `FLUX_SPEC_VERIFY=gumbel` or
// `FLUX_SAMPLER_TIE_ORDER != Some("legacy")`. A `SAMPLER_TIE_ORDER_RECEIPT`
// line records the choice once at first use.

use std::cmp::Ordering;

/// How the sampler orders its candidates before top-k truncation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CandidateOrder {
    Legacy,
    Total,
}

/// Resolve candidate tie order from `FLUX_SAMPLER_TIE_ORDER` alone.
///
/// Defaults to [`CandidateOrder::Total`] (`select_nth_unstable_by` O(V) partial
/// top-k) unless `"legacy"` is explicitly requested.
pub(crate) fn parse_candidate_order(value: Option<&str>) -> CandidateOrder {
    match value.map(str::trim) {
        Some("legacy") => CandidateOrder::Legacy,
        _ => CandidateOrder::Total,
    }
}

/// Resolve candidate tie order from both `FLUX_SAMPLER_TIE_ORDER` and `FLUX_SPEC_VERIFY`.
///
/// `FLUX_SPEC_VERIFY=gumbel` always requires [`CandidateOrder::Total`] so serial
/// and speculative paths select identical O(V) top-k sets.
pub(crate) fn resolve_candidate_order(
    tie_order: Option<&str>,
    spec_verify: Option<&str>,
) -> CandidateOrder {
    if super::gumbel::is_gumbel_verify_env(spec_verify) {
        CandidateOrder::Total
    } else {
        parse_candidate_order(tie_order)
    }
}

/// The process-wide order, read once.
pub(crate) fn candidate_order() -> CandidateOrder {
    static ORDER: std::sync::OnceLock<CandidateOrder> = std::sync::OnceLock::new();
    *ORDER.get_or_init(|| {
        let raw = std::env::var("FLUX_SAMPLER_TIE_ORDER").ok();
        let spec_verify = std::env::var("FLUX_SPEC_VERIFY")
            .or_else(|_| std::env::var("ATLAS_SPEC_VERIFY"))
            .ok();
        let order = resolve_candidate_order(raw.as_deref(), spec_verify.as_deref());
        if raw
            .as_deref()
            .map(str::trim)
            .is_some_and(|v| !v.is_empty() && v != "total" && v != "legacy")
        {
            tracing::warn!("FLUX_SAMPLER_TIE_ORDER={raw:?} is not legacy|total; using total");
        }
        tracing::info!(
            "SAMPLER_TIE_ORDER_RECEIPT schema=1 order={} topk_select={}",
            match order {
                CandidateOrder::Legacy => "legacy",
                CandidateOrder::Total => "total",
            },
            order == CandidateOrder::Total
        );
        order
    })
}

fn legacy_desc(a: &(u32, f32), b: &(u32, f32)) -> Ordering {
    b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
}

/// Logit descending, then id ascending. Candidates are finite (the sampler
/// filters non-finite values first), so this is a total order; +0.0 and
/// -0.0 compare equal and fall through to the id.
pub(crate) fn total_desc(a: &(u32, f32), b: &(u32, f32)) -> Ordering {
    legacy_desc(a, b).then(a.0.cmp(&b.0))
}

/// Order `logits` descending and keep the first `top_k` (0 = keep all).
pub(crate) fn order_candidates(logits: &mut Vec<(u32, f32)>, top_k: usize, order: CandidateOrder) {
    let truncate = top_k > 0 && top_k < logits.len();
    match order {
        CandidateOrder::Legacy => {
            logits.sort_unstable_by(legacy_desc);
            if truncate {
                logits.truncate(top_k);
            }
        }
        CandidateOrder::Total => {
            if truncate {
                logits.select_nth_unstable_by(top_k - 1, total_desc);
                logits.truncate(top_k);
            }
            logits.sort_unstable_by(total_desc);
        }
    }
}
