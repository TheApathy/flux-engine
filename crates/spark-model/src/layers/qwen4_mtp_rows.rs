// SPDX-License-Identifier: AGPL-3.0-only

//! Draft-row bookkeeping for the native Qwen4 MTP head's own KV cache after a
//! verify frame (opt-in decode levers, default off).
//!
//! A draft chain started at the root token `x0` (position `pre`) writes one
//! head-KV row per computed step: step 0 is `(x0, target hidden of pre-1)`,
//! step `i >= 1` is `(d_i, the head's own hidden from step i-1)`. The verify
//! commits `x0, d_1..d_a` (`a` accepted drafts) and the next chain starts at
//! the bonus position `pre + a + 1`.
//!
//! - Legacy (default `after_verify`): keep `a` rows, so the row of the last
//!   committed token (`pre + a`) is never present in the head cache: one hole
//!   per frame, and for `a = 0` the discarded row is the target-conditioned
//!   root row.
//! - `ATLAS_QWEN4_MTP_KEEP_LAST_ROW=1`: keep `min(a + 1, computed)` rows. Zero
//!   extra work; the hole remains only when every computed draft was accepted.
//! - `ATLAS_QWEN4_MTP_RECONDITION=<last|all|N>`: keep the root row and rewrite
//!   the last `min(a, N)` committed draft rows from the verify's TARGET hidden
//!   rows (`(d_i, target hidden row i-1)`), as vLLM's MTP proposer does; the
//!   remaining middle rows keep their chain values. Costs one MTP layer pass
//!   (no LM head) per rewritten row. Needs the staged verify rows; without
//!   them the plan falls back to KEEP_LAST_ROW.
//!
//! Every policy changes draft tokens only: the target verify decides every
//! committed token, so outputs are identical by construction.
//!
//! Dependency-free (std only) so the plan can be unit-tested with
//! `rustc --edition 2024 --test qwen4_mtp_rows.rs`.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) const KEEP_LAST_ROW_SELECTOR: &str = "ATLAS_QWEN4_MTP_KEEP_LAST_ROW";
pub(crate) const RECONDITION_SELECTOR: &str = "ATLAS_QWEN4_MTP_RECONDITION";

/// Largest number of verify rows any frame can stage (2..=9-row frames).
pub(crate) const MAX_STAGED_ROWS: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowPolicy {
    Legacy,
    KeepLast,
    /// Rewrite at most `max_rows` of the most recent committed draft rows.
    Recondition {
        max_rows: usize,
    },
}

/// Parse the two selectors. `RECONDITION` subsumes `KEEP_LAST_ROW` (it also
/// fills the last row); setting both is allowed and yields `Recondition`.
pub(crate) fn parse_row_policy(
    keep_last: Option<&str>,
    recondition: Option<&str>,
) -> Result<RowPolicy, String> {
    let keep = match keep_last {
        None | Some("") | Some("0") => false,
        Some("1") => true,
        Some(other) => {
            return Err(format!(
                "{KEEP_LAST_ROW_SELECTOR} must be 0 or 1, got {other:?}"
            ));
        }
    };
    let recondition = match recondition {
        None | Some("") | Some("0") => None,
        Some("last") => Some(1),
        Some("all") => Some(MAX_STAGED_ROWS),
        Some(n) => match n.parse::<usize>() {
            Ok(v) if (1..=MAX_STAGED_ROWS).contains(&v) => Some(v),
            _ => {
                return Err(format!(
                    "{RECONDITION_SELECTOR} must be 0, last, all or 1..={MAX_STAGED_ROWS}, got {n:?}"
                ));
            }
        },
    };
    Ok(match (recondition, keep) {
        (Some(max_rows), _) => RowPolicy::Recondition { max_rows },
        (None, true) => RowPolicy::KeepLast,
        (None, false) => RowPolicy::Legacy,
    })
}

/// The process policy, read once. A malformed value is reported by the caller
/// (`policy_error`) and behaves as Legacy (fail closed to the incumbent).
pub(crate) fn policy() -> RowPolicy {
    *policy_cell().0.get_or_init(|| {
        let keep = std::env::var(KEEP_LAST_ROW_SELECTOR).ok();
        let rec = std::env::var(RECONDITION_SELECTOR).ok();
        match parse_row_policy(keep.as_deref(), rec.as_deref()) {
            Ok(p) => p,
            Err(e) => {
                let _ = policy_cell().1.set(e);
                RowPolicy::Legacy
            }
        }
    })
}

/// The parse error, if the selectors were malformed (for the REFUSED receipt).
pub(crate) fn policy_error() -> Option<&'static str> {
    let _ = policy();
    policy_cell().1.get().map(String::as_str)
}

fn policy_cell() -> &'static (OnceLock<RowPolicy>, OnceLock<String>) {
    static CELL: (OnceLock<RowPolicy>, OnceLock<String>) = (OnceLock::new(), OnceLock::new());
    &CELL
}

/// Rows of staged verify hidden state the target model can provide (0 = none).
/// Set once by the model when it allocates the staging buffer.
static STAGED_ROWS_CAPACITY: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn set_staged_rows_capacity(rows: usize) {
    STAGED_ROWS_CAPACITY.store(rows, Ordering::Release);
}

pub(crate) fn staged_rows_capacity() -> usize {
    STAGED_ROWS_CAPACITY.load(Ordering::Acquire)
}

/// Recondition is engaged only with a staging buffer that holds every row a
/// frame can accept.
pub(crate) fn recondition_engaged() -> bool {
    matches!(policy(), RowPolicy::Recondition { .. }) && staged_rows_capacity() >= MAX_STAGED_ROWS
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AfterVerifyPlan {
    /// Chain rows kept, counted from the root row.
    pub keep_rows: usize,
    /// Committed draft rows to rewrite from target hidden rows, immediately
    /// after the kept rows.
    pub replay_rows: usize,
}

/// Plan the head cache after a verify that accepted `accepted` drafts of a
/// chain that wrote `computed` rows. `staged` says whether the target hidden
/// rows of this verify were staged (Recondition falls back to KeepLast
/// without them). Invariant: `keep_rows + replay_rows` rows end up written for
/// positions `pre .. pre + keep_rows + replay_rows`, all of them committed.
pub(crate) fn after_verify_plan(
    policy: RowPolicy,
    computed: usize,
    accepted: usize,
    staged: bool,
) -> AfterVerifyPlan {
    if computed == 0 {
        // No chain this frame (skipped propose or another draft source).
        return AfterVerifyPlan {
            keep_rows: 0,
            replay_rows: 0,
        };
    }
    let accepted = accepted.min(computed);
    match policy {
        RowPolicy::Legacy => AfterVerifyPlan {
            keep_rows: accepted,
            replay_rows: 0,
        },
        RowPolicy::KeepLast => AfterVerifyPlan {
            keep_rows: (accepted + 1).min(computed),
            replay_rows: 0,
        },
        RowPolicy::Recondition { .. } if !staged => {
            after_verify_plan(RowPolicy::KeepLast, computed, accepted, false)
        }
        RowPolicy::Recondition { max_rows } => {
            let replay = accepted.min(max_rows);
            AfterVerifyPlan {
                keep_rows: accepted + 1 - replay,
                replay_rows: replay,
            }
        }
    }
}

/// One rewritten head row: position offset from the root (`pre + index`),
/// the draft that sits there (`drafts[draft]`, `d_index`) and the verify row
/// whose target hidden conditions it (`index - 1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReplayRow {
    pub index: usize,
    pub draft: usize,
    pub hidden_row: usize,
}

pub(crate) fn replay_rows(plan: AfterVerifyPlan) -> impl Iterator<Item = ReplayRow> {
    (plan.keep_rows..plan.keep_rows + plan.replay_rows).map(|index| ReplayRow {
        index,
        draft: index - 1,
        hidden_row: index - 1,
    })
}

/// Rotary position of replay row `index` given the next chain's (bonus)
/// position and the frame's accepted count: the root sits at
/// `bonus - accepted - 1`.
pub(crate) fn replay_position(
    bonus_position: usize,
    accepted: usize,
    index: usize,
) -> Option<usize> {
    bonus_position
        .checked_sub(accepted + 1)
        .and_then(|root| root.checked_add(index))
        .filter(|p| *p < bonus_position)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_selectors() {
        assert_eq!(parse_row_policy(None, None), Ok(RowPolicy::Legacy));
        assert_eq!(
            parse_row_policy(Some("0"), Some("0")),
            Ok(RowPolicy::Legacy)
        );
        assert_eq!(parse_row_policy(Some("1"), None), Ok(RowPolicy::KeepLast));
        assert_eq!(
            parse_row_policy(None, Some("last")),
            Ok(RowPolicy::Recondition { max_rows: 1 })
        );
        assert_eq!(
            parse_row_policy(Some("1"), Some("all")),
            Ok(RowPolicy::Recondition {
                max_rows: MAX_STAGED_ROWS
            })
        );
        assert_eq!(
            parse_row_policy(None, Some("3")),
            Ok(RowPolicy::Recondition { max_rows: 3 })
        );
        for bad in ["2", "yes", "true"] {
            assert!(parse_row_policy(Some(bad), None).is_err(), "{bad}");
        }
        for bad in ["10", "-1", "x", "none"] {
            assert!(parse_row_policy(None, Some(bad)).is_err(), "{bad}");
        }
    }

    /// Legacy keeps `a` rows: the last committed position is always missing.
    #[test]
    fn legacy_leaves_one_hole_per_frame() {
        for computed in 1..=8 {
            for a in 0..=computed {
                let p = after_verify_plan(RowPolicy::Legacy, computed, a, true);
                assert_eq!(
                    p,
                    AfterVerifyPlan {
                        keep_rows: a,
                        replay_rows: 0
                    }
                );
                // Committed positions are pre..=pre+a (a + 1 of them).
                assert_eq!(p.keep_rows + p.replay_rows + 1, a + 1);
            }
        }
    }

    /// KeepLast covers every committed position unless the chain never wrote
    /// the last one (all computed drafts accepted).
    #[test]
    fn keep_last_fills_the_hole_when_the_row_exists() {
        for computed in 1..=8 {
            for a in 0..=computed {
                let p = after_verify_plan(RowPolicy::KeepLast, computed, a, false);
                assert_eq!(p.replay_rows, 0);
                assert!(p.keep_rows <= computed);
                if a < computed {
                    assert_eq!(p.keep_rows, a + 1, "computed={computed} a={a}");
                } else {
                    assert_eq!(p.keep_rows, a, "computed={computed} a={a}");
                }
            }
        }
    }

    /// Recondition always ends with exactly a + 1 committed rows, never keeps
    /// more chain rows than were written, and rewrites the most recent ones.
    #[test]
    fn recondition_covers_every_committed_position() {
        for max_rows in 1..=MAX_STAGED_ROWS {
            for computed in 1..=8 {
                for a in 0..=computed {
                    let p =
                        after_verify_plan(RowPolicy::Recondition { max_rows }, computed, a, true);
                    assert_eq!(
                        p.keep_rows + p.replay_rows,
                        a + 1,
                        "max={max_rows} c={computed} a={a}"
                    );
                    assert!(p.keep_rows >= 1 && p.keep_rows <= computed);
                    assert_eq!(p.replay_rows, a.min(max_rows));
                    let rows: Vec<ReplayRow> = replay_rows(p).collect();
                    assert_eq!(rows.len(), p.replay_rows);
                    for (j, r) in rows.iter().enumerate() {
                        assert_eq!(r.index, p.keep_rows + j);
                        assert!(r.index >= 1 && r.index <= a);
                        assert_eq!(r.draft, r.index - 1);
                        assert_eq!(r.hidden_row, r.index - 1);
                        assert!(r.hidden_row < MAX_STAGED_ROWS);
                    }
                    if let Some(last) = rows.last() {
                        assert_eq!(last.index, a, "the last committed row is rewritten");
                    }
                }
            }
        }
    }

    #[test]
    fn recondition_without_staged_rows_is_keep_last() {
        for computed in 1..=5 {
            for a in 0..=computed {
                assert_eq!(
                    after_verify_plan(RowPolicy::Recondition { max_rows: 9 }, computed, a, false),
                    after_verify_plan(RowPolicy::KeepLast, computed, a, false)
                );
            }
        }
    }

    #[test]
    fn no_chain_means_no_change() {
        for policy in [
            RowPolicy::Legacy,
            RowPolicy::KeepLast,
            RowPolicy::Recondition { max_rows: 4 },
        ] {
            assert_eq!(
                after_verify_plan(policy, 0, 3, true),
                AfterVerifyPlan {
                    keep_rows: 0,
                    replay_rows: 0
                }
            );
        }
    }

    #[test]
    fn accepted_beyond_computed_is_clamped() {
        let p = after_verify_plan(RowPolicy::Recondition { max_rows: 9 }, 2, 5, true);
        assert_eq!(p.keep_rows + p.replay_rows, 3);
    }

    #[test]
    fn replay_positions_run_from_root_plus_one_to_bonus_minus_one() {
        // root at 100, a = 3 accepted, bonus at 104.
        let bonus = 104;
        let a = 3;
        let p = after_verify_plan(RowPolicy::Recondition { max_rows: 9 }, 4, a, true);
        let pos: Vec<usize> = replay_rows(p)
            .map(|r| replay_position(bonus, a, r.index).unwrap())
            .collect();
        assert_eq!(pos, vec![101, 102, 103]);
        assert_eq!(replay_position(bonus, a, 0), Some(100));
        assert_eq!(
            replay_position(bonus, a, 4),
            None,
            "the bonus row belongs to the chain"
        );
        assert_eq!(replay_position(2, 3, 1), None, "underflow is refused");
    }
}
