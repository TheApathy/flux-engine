// SPDX-License-Identifier: AGPL-3.0-only

//! Read-only summaries of raw logits before sampler policy is applied.

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Candidate {
    pub token: usize,
    pub value: f32,
}

#[derive(Debug, Default, PartialEq)]
pub(super) struct RawSummary {
    pub top1: Option<Candidate>,
    pub top2: Option<Candidate>,
    pub finite_count: usize,
    pub nan_count: usize,
    pub positive_infinity_count: usize,
    pub negative_infinity_count: usize,
}

impl RawSummary {
    pub fn finite_margin(&self) -> Option<f32> {
        let (first, second) = (self.top1?, self.top2?);
        if !first.value.is_finite() || !second.value.is_finite() {
            return None;
        }
        let margin = first.value - second.value;
        margin.is_finite().then_some(margin)
    }
}

pub(super) fn summarize(row: &[f32]) -> RawSummary {
    let mut summary = RawSummary::default();
    for (token, &value) in row.iter().enumerate() {
        if value.is_nan() {
            summary.nan_count += 1;
            continue;
        }
        if value.is_finite() {
            summary.finite_count += 1;
        } else if value.is_sign_positive() {
            summary.positive_infinity_count += 1;
        } else {
            summary.negative_infinity_count += 1;
        }
        let candidate = Candidate { token, value };
        // Ascending token order plus strict comparison retains the first ID
        // for exact ties, including signed zero, as greedy sampling does.
        if summary.top1.is_none_or(|first| value > first.value) {
            summary.top2 = summary.top1;
            summary.top1 = Some(candidate);
        } else if summary.top2.is_none_or(|second| value > second.value) {
            summary.top2 = Some(candidate);
        }
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_two_use_numeric_order_and_lowest_id_for_ties() {
        let summary = summarize(&[-9.0, -2.0, -4.0, -2.0]);
        assert_eq!(
            summary.top1,
            Some(Candidate {
                token: 1,
                value: -2.0
            })
        );
        assert_eq!(
            summary.top2,
            Some(Candidate {
                token: 3,
                value: -2.0
            })
        );
        assert_eq!(summary.finite_count, 4);
        assert_eq!(summary.finite_margin(), Some(0.0));
    }

    #[test]
    fn replacing_first_keeps_the_previous_first_as_runner_up() {
        let summary = summarize(&[2.0, 1.0, 5.0, 4.0]);
        assert_eq!(summary.top1.unwrap().token, 2);
        assert_eq!(summary.top2.unwrap().token, 3);
        assert_eq!(summary.finite_margin(), Some(1.0));
    }

    #[test]
    fn signed_zero_is_a_tie_and_preserves_raw_bits() {
        let summary = summarize(&[-1.0, -0.0, 0.0]);
        let first = summary.top1.unwrap();
        let second = summary.top2.unwrap();
        assert_eq!((first.token, second.token), (1, 2));
        assert_eq!(first.value.to_bits(), (-0.0f32).to_bits());
        assert_eq!(second.value.to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn nonfinite_census_skips_nan_but_ranks_infinities() {
        let summary = summarize(&[
            f32::NAN,
            f32::NEG_INFINITY,
            3.0,
            f32::INFINITY,
            f32::INFINITY,
        ]);
        assert_eq!(summary.top1.unwrap().token, 3);
        assert_eq!(summary.top2.unwrap().token, 4);
        assert_eq!(summary.finite_count, 1);
        assert_eq!(summary.nan_count, 1);
        assert_eq!(summary.positive_infinity_count, 2);
        assert_eq!(summary.negative_infinity_count, 1);
        assert_eq!(summary.finite_margin(), None);
    }

    #[test]
    fn empty_and_nan_only_rows_have_no_candidates() {
        assert_eq!(summarize(&[]), RawSummary::default());
        let summary = summarize(&[f32::NAN, f32::NAN]);
        assert_eq!(summary.top1, None);
        assert_eq!(summary.top2, None);
        assert_eq!(summary.nan_count, 2);
        assert_eq!(summary.finite_margin(), None);
    }

    #[test]
    fn negative_infinity_ties_have_stable_ids() {
        let summary = summarize(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
        assert_eq!(summary.top1.unwrap().token, 0);
        assert_eq!(summary.top2.unwrap().token, 1);
        assert_eq!(summary.finite_margin(), None);
    }

    #[test]
    fn missing_runner_up_and_overflow_have_no_finite_margin() {
        assert_eq!(summarize(&[1.0]).finite_margin(), None);
        assert_eq!(summarize(&[f32::MAX, -f32::MAX]).finite_margin(), None);
    }
}
