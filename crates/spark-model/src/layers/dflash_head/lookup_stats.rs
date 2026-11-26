// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request SAM/retrieval lookup counters for the opt-in
//! `FLUX_DFLASH_LOOKUP` drafting paths.
//!
//! The counters are host-only integers on the per-sequence proposer state.
//! They never influence proposal or verification; they exist so a receipt
//! can show whether lookup actually fired on a workload, instead of relying
//! on the process-wide log-once "first hit" line.

use crate::speculative::ProposerState;

/// Counters for the pre-empting (non-portfolio, non-corroborate) lookup path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LookupStats {
    /// Propose steps where the matcher ran.
    pub attempts: u64,
    /// Propose steps skipped by the adaptive cooldown.
    pub suppressed: u64,
    /// Matcher returned any suffix match (before the hybrid_min/width gate).
    pub hits: u64,
    /// Matches that passed the gate and replaced the neural drafter.
    pub preempts: u64,
    /// Pre-empted frames whose verify outcome was observed.
    pub verified_preempts: u64,
    /// Draft tokens accepted by the target from pre-empted frames.
    pub accepted_from_lookup: u64,
    /// Draft tokens proposed by pre-empted frames.
    pub drafted_from_lookup: u64,
    /// Adaptive-gate misfire increments (pre-empt accept below min_accept).
    pub misfires: u64,
    /// Times the adaptive gate entered its cooldown.
    pub cooldowns: u64,
    /// A pre-empted frame is awaiting its verify outcome.
    pub verify_pending: bool,
}

impl LookupStats {
    pub fn record_suppressed(&mut self) {
        self.suppressed += 1;
    }

    pub fn record_attempt(&mut self, hit: bool) {
        self.attempts += 1;
        if hit {
            self.hits += 1;
        }
    }

    pub fn record_preempt(&mut self, drafts: usize) {
        self.preempts += 1;
        self.drafted_from_lookup += drafts as u64;
        self.verify_pending = true;
    }

    /// Called on every verify. Counts `num_accepted` (the verifier's
    /// accepted-draft count) only when the frame came from lookup, then
    /// clears the pending mark so later drafter frames are not attributed.
    pub fn take_pending_verify(&mut self, num_accepted: usize) {
        if std::mem::take(&mut self.verify_pending) {
            self.verified_preempts += 1;
            self.accepted_from_lookup += num_accepted as u64;
        }
    }

    pub fn record_misfire(&mut self) {
        self.misfires += 1;
    }

    pub fn record_cooldown(&mut self) {
        self.cooldowns += 1;
    }

    /// True when the request never reached the lookup path at all.
    pub fn is_idle(&self) -> bool {
        self.attempts == 0 && self.suppressed == 0
    }

    /// Stable `key=value` body for the `LOOKUP_STATS` receipt line.
    pub fn receipt_fields(&self) -> String {
        format!(
            "attempts={} suppressed={} hits={} preempts={} verified_preempts={} \
             drafted_from_lookup={} accepted_from_lookup={} misfires={} cooldowns={} pending_verify={}",
            self.attempts,
            self.suppressed,
            self.hits,
            self.preempts,
            self.verified_preempts,
            self.drafted_from_lookup,
            self.accepted_from_lookup,
            self.misfires,
            self.cooldowns,
            u8::from(self.verify_pending),
        )
    }
}

/// Lookup mode resolved from the same environment the matcher reads.
/// Returns None when neither SAM nor retrieval is enabled.
pub fn lookup_mode() -> Option<&'static str> {
    let cfg = super::retrieval::RetrievalConfig::from_env(1)?;
    Some(if cfg.sam { "sam" } else { "retrieval" })
}

/// Counters from a DFlash proposer state; None for any other proposer.
pub fn lookup_stats(state: &dyn ProposerState) -> Option<LookupStats> {
    state
        .as_any()
        .downcast_ref::<super::DflashProposerState>()
        .map(|d| d.lookup_stats)
}

#[cfg(test)]
mod tests {
    use super::LookupStats;

    #[test]
    fn default_is_idle_and_zero() {
        let s = LookupStats::default();
        assert!(s.is_idle());
        assert_eq!(
            s.receipt_fields(),
            "attempts=0 suppressed=0 hits=0 preempts=0 verified_preempts=0 \
             drafted_from_lookup=0 accepted_from_lookup=0 misfires=0 cooldowns=0 pending_verify=0"
        );
    }

    #[test]
    fn records_a_preempt_cycle() {
        let mut s = LookupStats::default();
        s.record_attempt(false);
        s.record_attempt(true);
        s.record_attempt(true);
        s.record_preempt(16);
        s.take_pending_verify(11);
        // A following drafter frame must not be attributed to lookup.
        s.take_pending_verify(5);
        s.record_misfire();
        s.record_cooldown();
        s.record_suppressed();
        assert!(!s.is_idle());
        assert_eq!(
            s,
            LookupStats {
                attempts: 3,
                suppressed: 1,
                hits: 2,
                preempts: 1,
                verified_preempts: 1,
                accepted_from_lookup: 11,
                drafted_from_lookup: 16,
                misfires: 1,
                cooldowns: 1,
                verify_pending: false,
            }
        );
        assert!(s.receipt_fields().contains(" preempts=1 "));
        assert!(s.receipt_fields().contains("accepted_from_lookup=11"));
    }

    #[test]
    fn unverified_preempt_stays_pending() {
        let mut s = LookupStats::default();
        s.record_attempt(true);
        s.record_preempt(16);
        assert!(s.verify_pending);
        assert_eq!(s.verified_preempts, 0);
        assert_eq!(s.accepted_from_lookup, 0);
        assert!(s.receipt_fields().ends_with("pending_verify=1"));
    }

    #[test]
    fn suppressed_only_is_not_idle() {
        let mut s = LookupStats::default();
        s.record_suppressed();
        assert!(!s.is_idle());
    }
}
