// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request `LOOKUP_STATS` receipt for opt-in SAM/retrieval drafting.
//!
//! Emitted once at request retire (finish or error), before the proposer
//! state is freed. Silent when no lookup mode is configured, so default
//! serving logs are unchanged.

use spark_model::layers::dflash_head::lookup_stats::{LookupStats, lookup_mode, lookup_stats};

use super::ActiveSeq;

pub(super) fn format_line(
    session_hash: u64,
    mode: &str,
    outcome: &str,
    output_tokens: usize,
    stats: &LookupStats,
) -> String {
    format!(
        "LOOKUP_STATS schema=1 session={session_hash:016x} mode={mode} outcome={outcome} \
         output_tokens={output_tokens} {}",
        stats.receipt_fields()
    )
}

/// Log the receipt for `a` when lookup is configured and the active proposer
/// is DFlash. `outcome` is `finish` or `error`.
pub(super) fn emit(a: &ActiveSeq, outcome: &str) {
    let Some(mode) = lookup_mode() else {
        return;
    };
    let Some(stats) = a.seq.proposer_state.as_deref().and_then(lookup_stats) else {
        return;
    };
    tracing::info!(
        "{}",
        format_line(a.session_hash, mode, outcome, a.output_tokens.len(), &stats)
    );
}

#[cfg(test)]
mod tests {
    use super::format_line;
    use spark_model::layers::dflash_head::lookup_stats::LookupStats;

    #[test]
    fn line_has_stable_schema_and_order() {
        let mut s = LookupStats::default();
        s.record_attempt(true);
        s.record_preempt(16);
        s.take_pending_verify(9);
        let line = format_line(0xabe4ccd1e98b3211, "sam", "finish", 128, &s);
        assert_eq!(
            line,
            "LOOKUP_STATS schema=1 session=abe4ccd1e98b3211 mode=sam outcome=finish \
             output_tokens=128 attempts=1 suppressed=0 hits=1 preempts=1 verified_preempts=1 \
             drafted_from_lookup=16 accepted_from_lookup=9 misfires=0 cooldowns=0 pending_verify=0"
        );
    }

    #[test]
    fn emitted_at_both_retire_paths_before_free() {
        let src = include_str!("lifecycle.rs");
        let finish = src
            .split_once("pub(super) fn finish_sequence_with_cache")
            .and_then(|(_, rest)| rest.split_once("pub fn send_error("))
            .map(|(body, _)| body)
            .expect("finish body");
        let error = src
            .split_once("pub fn send_error(")
            .and_then(|(_, rest)| rest.split_once("pub fn send_error_to_sink"))
            .map(|(body, _)| body)
            .expect("error body");
        for (body, tag) in [(finish, "\"finish\""), (error, "\"error\"")] {
            let call = format!("super::lookup_receipt::emit(a, {tag});");
            assert_eq!(body.matches(&call).count(), 1, "{tag}");
            assert!(body.find(&call) < body.find(".finish_or_release(model,"));
        }
    }
}
