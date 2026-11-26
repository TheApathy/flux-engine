// SPDX-License-Identifier: AGPL-3.0-only

//! Shortest-Remaining-Prefill-First (SRPT) scheduling with anti-starvation aging
//! and guaranteed prefill/decode alternation.
//!
//! When multiple prompts wait in the pending queue or multiple chunked prefills
//! are already in flight, strict FIFO head-of-line blocks shorter prompts behind
//! longer ones. SRPT selects the candidate with the fewest effective remaining
//! tokens:
//!
//!   `effective_remaining(rem, age) = rem.saturating_sub(age * aging_tokens_per_tick)`
//!
//! Anti-starvation aging reduces a skipped request's effective remaining count by
//! `aging_tokens_per_tick` (default 256) each tick it is bypassed, bounding maximum
//! wait time even under continuous short arrivals.
//!
//! Live scheduler wiring currently uses the two split selectors:
//! - [`select_pending_indices`] in `mod_helpers::drain_pending_requests_for_tick`
//!   orders pending admissions shortest-first whenever SRPT is enabled
//!   (`ATLAS_PREFILL_SRPT=1`, or unset while SLO interleave is active).
//! - [`select_in_flight_index`] in `phase_continue_prefills` chooses the shortest
//!   remaining in-flight prefill when multiple prefills are already in flight
//!   during active decode.
//!
//! Because `prefill_slo::TickPlan::admission_batch_cap` currently gates new
//! admissions to `prefilling == 0` while decoding, the unified preemption and
//! hysteresis branch in [`select_prefill_work`] (`preemption_hysteresis_tokens` /
//! `max_concurrent_prefills`) is kept as a tested policy primitive and is not
//! invoked with both `in_flight` and `pending` non-empty in the live tick.
//! Whenever at least one decode sequence is active (`active_decode_count > 0`),
//! [`select_prefill_work`] enforces `prefill_chunks_this_tick <= 1` so every
//! prefill chunk alternates with a decode step.

use std::collections::HashMap;
use std::sync::OnceLock;

use parking_lot::Mutex;

/// Environment selector: `"0"` disables SRPT back to strict FIFO; `"1"` forces
/// SRPT on; unset defaults to enabled whenever SLO interleave is active.
pub const SRPT_ENABLE_ENV: &str = "ATLAS_PREFILL_SRPT";
/// Priority credit (in tokens) accrued per skipped tick to prevent starvation.
pub const DEFAULT_AGING_TOKENS_PER_TICK: usize = 256;
/// Minimum effective-remaining margin a pending prompt must beat an in-flight
/// prefill by in [`select_prefill_work`] (pure unified policy primitive; not
/// wired into live mid-prefill admission while `admission_batch_cap` enforces
/// `prefilling == 0`).
pub const DEFAULT_PREEMPTION_HYSTERESIS_TOKENS: usize = 512;
/// Maximum simultaneous in-flight chunked prefills in [`select_prefill_work`]
/// (pure unified policy primitive; live SLO admission currently admits only
/// when `prefilling == 0`).
pub const DEFAULT_MAX_CONCURRENT_PREFILLS: usize = 4;

/// SRPT scheduler tuning parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SrptConfig {
    pub enabled: bool,
    pub aging_tokens_per_tick: usize,
    pub preemption_hysteresis_tokens: usize,
    pub max_concurrent_prefills: usize,
}

impl Default for SrptConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            aging_tokens_per_tick: DEFAULT_AGING_TOKENS_PER_TICK,
            preemption_hysteresis_tokens: DEFAULT_PREEMPTION_HYSTERESIS_TOKENS,
            max_concurrent_prefills: DEFAULT_MAX_CONCURRENT_PREFILLS,
        }
    }
}

impl SrptConfig {
    /// Cached raw `ATLAS_PREFILL_SRPT` env value (read once per process).
    fn cached_enable_env() -> Option<&'static str> {
        static RAW: OnceLock<Option<String>> = OnceLock::new();
        RAW.get_or_init(|| std::env::var(SRPT_ENABLE_ENV).ok())
            .as_deref()
    }

    /// Parse `ATLAS_PREFILL_SRPT`: `"0"` -> false, `"1"` -> true, unset -> `interleave_active`.
    pub fn parse_enabled(raw: Option<&str>, interleave_active: bool) -> bool {
        match raw.map(str::trim) {
            Some("0") => false,
            Some("1") => true,
            _ => interleave_active,
        }
    }

    /// Resolve SRPT configuration from cached environment, defaulting on when SLO interleave is active.
    pub fn from_env(interleave_active: bool) -> Self {
        Self {
            enabled: Self::parse_enabled(Self::cached_enable_env(), interleave_active),
            ..Self::default()
        }
    }
}

/// Summary of one in-progress chunked prefill candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InFlightPrefill {
    pub prefill_idx: usize,
    pub remaining_tokens: usize,
    pub chunk_offset: usize,
    pub wait_ticks: u64,
}

/// Summary of one queued pending prefill candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PendingPrefill {
    pub pending_idx: usize,
    pub prompt_tokens: usize,
    pub wait_ticks: u64,
}

/// Single-tick prefill work selection returned by [`select_prefill_work`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SrptSelection {
    AdmitPending {
        pending_idx: usize,
        prompt_tokens: usize,
    },
    ContinueInFlight {
        prefill_idx: usize,
        remaining_tokens: usize,
        chunk_offset: usize,
    },
    Idle,
}

/// Effective remaining tokens after subtracting anti-starvation aging credit.
#[inline]
pub fn effective_remaining(
    remaining_tokens: usize,
    age_ticks: u64,
    aging_tokens_per_tick: usize,
) -> usize {
    let rate = u64::try_from(aging_tokens_per_tick).unwrap_or(u64::MAX);
    let credit = usize::try_from(age_ticks.saturating_mul(rate)).unwrap_or(usize::MAX);
    remaining_tokens.saturating_sub(credit)
}

/// Choose at most one prefill work item for the current tick.
///
/// Guarantees `prefill_chunks_this_tick <= 1` whenever `active_decode_count > 0`,
/// ensuring every prefill chunk alternates with a decode step.
pub fn select_prefill_work(
    cfg: SrptConfig,
    in_flight: &[InFlightPrefill],
    pending: &[PendingPrefill],
    active_decode_count: usize,
    prefill_chunks_this_tick: usize,
    can_admit_slot: bool,
) -> SrptSelection {
    if active_decode_count > 0 && prefill_chunks_this_tick >= 1 {
        return SrptSelection::Idle;
    }
    if in_flight.is_empty() && pending.is_empty() {
        return SrptSelection::Idle;
    }
    if !cfg.enabled {
        if let Some(first) = in_flight.first() {
            return SrptSelection::ContinueInFlight {
                prefill_idx: first.prefill_idx,
                remaining_tokens: first.remaining_tokens,
                chunk_offset: first.chunk_offset,
            };
        }
        if can_admit_slot && let Some(first) = pending.first() {
            return SrptSelection::AdmitPending {
                pending_idx: first.pending_idx,
                prompt_tokens: first.prompt_tokens,
            };
        }
        return SrptSelection::Idle;
    }

    let best_in_flight = in_flight
        .iter()
        .map(|f| {
            let eff =
                effective_remaining(f.remaining_tokens, f.wait_ticks, cfg.aging_tokens_per_tick);
            (eff, f)
        })
        .min_by_key(|&(eff, f)| (eff, f.remaining_tokens, f.prefill_idx));

    let admit_allowed = can_admit_slot && in_flight.len() < cfg.max_concurrent_prefills;
    let best_pending = if admit_allowed {
        pending
            .iter()
            .map(|p| {
                let eff =
                    effective_remaining(p.prompt_tokens, p.wait_ticks, cfg.aging_tokens_per_tick);
                (eff, p)
            })
            .min_by_key(|&(eff, p)| (eff, p.prompt_tokens, p.pending_idx))
    } else {
        None
    };

    match (best_in_flight, best_pending) {
        (None, None) => SrptSelection::Idle,
        (Some((_, f)), None) => SrptSelection::ContinueInFlight {
            prefill_idx: f.prefill_idx,
            remaining_tokens: f.remaining_tokens,
            chunk_offset: f.chunk_offset,
        },
        (None, Some((_, p))) => SrptSelection::AdmitPending {
            pending_idx: p.pending_idx,
            prompt_tokens: p.prompt_tokens,
        },
        (Some((eff_f, f)), Some((eff_p, p))) => {
            if eff_p.saturating_add(cfg.preemption_hysteresis_tokens) < eff_f {
                SrptSelection::AdmitPending {
                    pending_idx: p.pending_idx,
                    prompt_tokens: p.prompt_tokens,
                }
            } else {
                SrptSelection::ContinueInFlight {
                    prefill_idx: f.prefill_idx,
                    remaining_tokens: f.remaining_tokens,
                    chunk_offset: f.chunk_offset,
                }
            }
        }
    }
}

/// Stable wait-tracking key for an in-flight `PrefillInProgress` sequence.
#[inline]
pub fn prefill_wait_key(session_hash: u64, slot_idx: usize) -> u64 {
    if slot_idx == usize::MAX {
        session_hash
    } else {
        session_hash ^ (slot_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }
}

/// Stable wait-tracking key for a queued `InferenceRequest` while resident in `PendingQueue`.
#[inline]
pub fn pending_wait_key(tokens_ptr: usize, prompt_len: usize) -> u64 {
    (tokens_ptr as u64) ^ (prompt_len as u64).wrapping_mul(0x517C_C1B7_2722_0A95)
}

/// Tracks per-request skipped ticks externally so `PrefillInProgress` and
/// `PendingQueue` layouts stay unchanged while guaranteeing anti-starvation aging.
#[derive(Clone, Debug, Default)]
pub struct SrptWaitTracker {
    wait_ticks: HashMap<u64, u64>,
}

impl SrptWaitTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn wait_ticks(&self, key: u64) -> u64 {
        self.wait_ticks.get(&key).copied().unwrap_or(0)
    }

    /// Select one in-flight prefill index from `(wait_key, remaining_tokens, chunk_offset)`
    /// tuples, resetting the chosen entry's wait ticks and aging every skipped entry.
    pub fn select_in_flight(
        &mut self,
        cfg: SrptConfig,
        candidates: &[(u64, usize, usize)],
    ) -> Option<usize> {
        self.wait_ticks
            .retain(|key, _| candidates.iter().any(|(k, _, _)| k == key));
        if candidates.is_empty() {
            return None;
        }
        let in_flight: Vec<InFlightPrefill> = candidates
            .iter()
            .enumerate()
            .map(
                |(prefill_idx, &(key, remaining_tokens, chunk_offset))| InFlightPrefill {
                    prefill_idx,
                    remaining_tokens,
                    chunk_offset,
                    wait_ticks: self.wait_ticks(key),
                },
            )
            .collect();
        let chosen = match select_prefill_work(cfg, &in_flight, &[], 1, 0, false) {
            SrptSelection::ContinueInFlight {
                prefill_idx,
                remaining_tokens,
                chunk_offset,
            } => {
                tracing::trace!(
                    prefill_idx,
                    remaining_tokens,
                    chunk_offset,
                    "SRPT selected in-flight prefill continuation"
                );
                prefill_idx
            }
            SrptSelection::AdmitPending { .. } | SrptSelection::Idle => return None,
        };
        for (idx, &(key, _, _)) in candidates.iter().enumerate() {
            if idx == chosen {
                self.wait_ticks.insert(key, 0);
            } else {
                let entry = self.wait_ticks.entry(key).or_insert(0);
                *entry = entry.saturating_add(1);
            }
        }
        Some(chosen)
    }

    /// Select up to `capacity` pending request indices from `(wait_key, prompt_tokens)`
    /// tuples in SRPT order (with aging and FIFO tie-breaking), aging any skipped entries.
    pub fn select_pending_order(
        &mut self,
        cfg: SrptConfig,
        candidates: &[(u64, usize)],
        capacity: usize,
    ) -> Vec<usize> {
        self.wait_ticks
            .retain(|key, _| candidates.iter().any(|(k, _)| k == key));
        let take = capacity.min(candidates.len());
        if take == 0 {
            return Vec::new();
        }
        if !cfg.enabled {
            let selected: Vec<usize> = (0..take).collect();
            for &idx in &selected {
                self.wait_ticks.remove(&candidates[idx].0);
            }
            return selected;
        }

        let mut pending: Vec<PendingPrefill> = candidates
            .iter()
            .enumerate()
            .map(|(pending_idx, &(key, prompt_tokens))| PendingPrefill {
                pending_idx,
                prompt_tokens,
                wait_ticks: self.wait_ticks(key),
            })
            .collect();

        let selected: Vec<usize> = if take == 1 {
            match select_prefill_work(cfg, &[], &pending, 0, 0, true) {
                SrptSelection::AdmitPending {
                    pending_idx,
                    prompt_tokens,
                } => {
                    tracing::trace!(pending_idx, prompt_tokens, "SRPT selected pending prefill");
                    vec![pending_idx]
                }
                SrptSelection::ContinueInFlight { .. } | SrptSelection::Idle => Vec::new(),
            }
        } else {
            pending.sort_by_key(|p| {
                let eff =
                    effective_remaining(p.prompt_tokens, p.wait_ticks, cfg.aging_tokens_per_tick);
                (eff, p.prompt_tokens, p.pending_idx)
            });
            pending
                .into_iter()
                .take(take)
                .map(|p| p.pending_idx)
                .collect()
        };

        for (idx, &(key, _)) in candidates.iter().enumerate() {
            if selected.contains(&idx) {
                self.wait_ticks.remove(&key);
            } else {
                let entry = self.wait_ticks.entry(key).or_insert(0);
                *entry = entry.saturating_add(1);
            }
        }
        selected
    }
}

static IN_FLIGHT_WAIT_TRACKER: OnceLock<Mutex<SrptWaitTracker>> = OnceLock::new();
static PENDING_WAIT_TRACKER: OnceLock<Mutex<SrptWaitTracker>> = OnceLock::new();

/// Process-wide in-flight SRPT selector used by `phase_continue_prefills`.
pub fn select_in_flight_index(
    cfg: SrptConfig,
    candidates: &[(u64, usize, usize)],
) -> Option<usize> {
    IN_FLIGHT_WAIT_TRACKER
        .get_or_init(|| Mutex::new(SrptWaitTracker::new()))
        .lock()
        .select_in_flight(cfg, candidates)
}

/// Process-wide pending-queue SRPT selector used by `mod_helpers::drain_pending_requests`.
pub fn select_pending_indices(
    cfg: SrptConfig,
    candidates: &[(u64, usize)],
    capacity: usize,
) -> Vec<usize> {
    PENDING_WAIT_TRACKER
        .get_or_init(|| Mutex::new(SrptWaitTracker::new()))
        .lock()
        .select_pending_order(cfg, candidates, capacity)
}

#[cfg(test)]
#[path = "prefill_srpt_tests.rs"]
mod tests;
