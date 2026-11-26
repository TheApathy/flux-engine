// SPDX-License-Identifier: AGPL-3.0-only

//! Failed speculative state must not become a successful, reusable prefix.
//!
//! Invariant: any speculative bootstrap or step failure disables prefix-cache
//! publication for the sequence, clears pending speculative proposals, notifies
//! the terminal sink without freeing resources early, and leaves exactly one
//! resource retirement pass to the scheduler loop.

use super::super::*;

pub(super) fn fail_bootstrap(
    a: &mut ActiveSeq,
    cache_on_finish: &mut bool,
    stage: &str,
    error: impl std::fmt::Display,
) {
    *cache_on_finish = false;
    fail_speculative_step(a, &format!("bootstrap {stage}"), error);
}

/// Notify without freeing; the MTP verify cohort rejects finished frames from
/// the prefix cache before the scheduler performs its single resource retirement.
pub(in crate::scheduler) fn fail_speculative_step(
    a: &mut ActiveSeq,
    stage: &str,
    error: impl std::fmt::Display,
) {
    a.finished = true;
    proposal_lifecycle::clear(
        &mut a.pending_drafts,
        &mut a.pending_tree_payload,
        &mut a.pending_dflash_sampled_proposal,
    );
    // send_error frees immediately and would double-free at retirement. Take
    // only the terminal sink here so retirement cannot deliver a later Done.
    let mut sink = std::mem::replace(&mut a.sink, ResponseSink::Blocking(None));
    send_error_to_sink(&mut sink, &format!("MTP {stage} failed: {error:#}"));
}

#[cfg(test)]
#[path = "mtp_bootstrap_failure_tests.rs"]
mod tests;
