// SPDX-License-Identifier: AGPL-3.0-only

//! SLO-budgeted prefill interleave (`--decode-gap-target-ms`, default 150).
//!
//! Each tick with at least one decode row, the pending prompt (the one
//! mid-prefill, else the head of the pending queue) gets a [`Decision`]:
//!
//! 1. **MLA**: the whole remainder (single-chunk exemption).
//! 2. **Short-prompt lane**: the remainder fits one admissible chunk whose
//!    predicted time is within [`SHORT_LANE_FACTOR`] x the target: run it
//!    whole (chat-sized prompts are not sliced).
//! 3. **Target met**: the largest prepared width `B` aligned to the offset
//!    whose predicted time is `<= target − decode step` (clipped at the prompt
//!    end).
//! 4. **Starvation guard**: nothing fits, but `N = max(1, floor(target /
//!    decode step))` decode steps have passed without a chunk: run the
//!    smallest aligned width, so a long prompt always progresses.
//! 5. Otherwise **skip**: decode only this tick.
//!
//! Decoders get at least one step between any two chunks: a tick runs at most
//! one chunk and then the decode step (no second admission's chunk 0, no
//! continue chunk after chunk 0). With no decode row the decision is
//! [`Decision::Passthrough`] and scheduling is the historical one.
//!
//! Chunk boundaries are on committed carry positions: a prepared width at an
//! offset that is a multiple of it (the fast family's admission rule, which
//! is also a multiple of the 4-token WY4/QSA group, the 16-row tiles and the
//! KV block), or off a prepared boundary a multiple of the carry alignment
//! (lcm of 16 and the KV block size). The SSM/conv carry is normalized after
//! every chunk as before.

use std::path::PathBuf;
use std::time::Instant;

use super::prefill_chunk_policy::{ChunkPlan, PREFILL_CHUNK_ALIGN, PreparedWidths, SloChunk};
use super::prefill_slo_cost::cache::{SloCacheHandle, SloCostFingerprint, load_entry_with_seed};
use super::prefill_slo_cost::{ChunkCostModel, DecodeStepClock};

#[path = "prefill_srpt.rs"]
pub(super) mod prefill_srpt;

/// The short-prompt lane admits a whole remainder predicted within this
/// multiple of the gap target.
pub const SHORT_LANE_FACTOR: f64 = 2.0;
/// Upper bound on the starvation interval (a decode step measured near 0 ms
/// must not stall prefill for ever).
pub const MAX_STARVATION_STEPS: u32 = 64;
/// Environment name of the parity diagnostic (see [`parse_parity_width`]).
pub const PARITY_WIDTH_ENV: &str = "ATLAS_PREFILL_INTERLEAVE_PARITY_WIDTH";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reason {
    TargetMet,
    ShortLane,
    Starvation,
    SingleChunk,
    Parity,
}

impl Reason {
    pub fn label(self) -> &'static str {
        match self {
            Reason::TargetMet => "target_met",
            Reason::ShortLane => "short_lane",
            Reason::Starvation => "starvation",
            Reason::SingleChunk => "single_chunk",
            Reason::Parity => "parity",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Decision {
    /// No decode row (or interleave off): historical scheduling.
    Passthrough,
    /// Decode only this tick.
    Skip,
    /// One chunk of `rows` tokens for the candidate.
    Chunk { rows: usize, reason: Reason },
}

impl Decision {
    pub fn label(self) -> &'static str {
        match self {
            Decision::Passthrough => "passthrough",
            Decision::Skip => "skip",
            Decision::Chunk { reason, .. } => reason.label(),
        }
    }
}

/// The prompt a decision is for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Candidate {
    pub offset: usize,
    pub remaining: usize,
}

/// Inputs of one decision.
#[derive(Clone, Copy, Debug)]
pub struct DecideInputs {
    pub candidate: Candidate,
    pub widths: PreparedWidths,
    pub max_prefill_tokens: usize,
    pub cap: Option<usize>,
    pub carry_align: usize,
    pub single_chunk_required: bool,
    pub decode_rows_active: bool,
    pub target_ms: f64,
    pub decode_step_ms: f64,
    pub steps_since_chunk: u32,
}

/// `N = max(1, floor(target / decode step))`, at most [`MAX_STARVATION_STEPS`].
pub fn starvation_steps(target_ms: f64, decode_step_ms: f64) -> u32 {
    if decode_step_ms <= 0.0 || !decode_step_ms.is_finite() {
        return MAX_STARVATION_STEPS;
    }
    let n = (target_ms / decode_step_ms).floor();
    if n.is_finite() && n >= 1.0 {
        (n as u32).min(MAX_STARVATION_STEPS)
    } else {
        1
    }
}

/// Smallest admissible chunk at the candidate's offset: the smallest aligned
/// prepared width, or off every prepared boundary the smallest width rounded
/// to the carry alignment; clipped at the prompt end.
fn smallest_chunk(inp: &DecideInputs, limit: usize) -> usize {
    let c = inp.candidate;
    let align = inp.carry_align.max(PREFILL_CHUNK_ALIGN);
    let rows = inp
        .widths
        .aligned_desc(c.offset, limit)
        .last()
        .unwrap_or_else(|| {
            let r = inp.widths.smallest().min(limit);
            (r / align * align).max(align)
        });
    rows.min(c.remaining)
}

/// The SLO decision for one tick. `predict(rows, offset)` is the cost model.
pub fn decide(inp: &DecideInputs, predict: impl Fn(usize, usize) -> f64) -> Decision {
    let c = inp.candidate;
    if !inp.decode_rows_active || c.remaining == 0 || inp.max_prefill_tokens == 0 {
        return Decision::Passthrough;
    }
    if inp.single_chunk_required {
        return Decision::Chunk {
            rows: c.remaining,
            reason: Reason::SingleChunk,
        };
    }
    let limit = inp.max_prefill_tokens.min(inp.cap.unwrap_or(usize::MAX));
    if let Some(width) = inp.widths.largest_aligned(c.offset, limit)
        && c.remaining <= width
        && predict(c.remaining, c.offset) <= SHORT_LANE_FACTOR * inp.target_ms
    {
        return Decision::Chunk {
            rows: c.remaining,
            reason: Reason::ShortLane,
        };
    }
    let budget = inp.target_ms - inp.decode_step_ms;
    let fits = |rows: usize| predict(rows, c.offset) <= budget;
    let aligned = inp
        .widths
        .aligned_desc(c.offset, limit)
        .map(|w| w.min(c.remaining))
        .find(|&rows| fits(rows));
    let off_boundary = inp.widths.largest_aligned(c.offset, limit).is_none();
    let fit = aligned.or_else(|| {
        let rows = smallest_chunk(inp, limit);
        (off_boundary && fits(rows)).then_some(rows)
    });
    if let Some(rows) = fit {
        return Decision::Chunk {
            rows,
            reason: Reason::TargetMet,
        };
    }
    if inp.steps_since_chunk >= starvation_steps(inp.target_ms, inp.decode_step_ms) {
        return Decision::Chunk {
            rows: smallest_chunk(inp, limit),
            reason: Reason::Starvation,
        };
    }
    Decision::Skip
}

/// `ATLAS_PREFILL_INTERLEAVE_PARITY_WIDTH=<B>` (diagnostic, default off):
/// prefill every request in chunks of the prepared width `B` even with no
/// decode row, so the GPU parity test compares a chunked-at-B prefill
/// against the full-W one deterministically. Absent or `0` is off; anything
/// other than a prepared width is an error.
pub fn parse_parity_width(raw: Option<&str>, widths: &[usize]) -> Result<Option<usize>, String> {
    match raw {
        None | Some("0") => Ok(None),
        Some(value) => match value.parse::<usize>() {
            Ok(width) if widths.contains(&width) => Ok(Some(width)),
            _ => Err(format!(
                "{PARITY_WIDTH_ENV} must be absent, 0, or one of the prepared widths {widths:?}; \
                 got {value:?}"
            )),
        },
    }
}

/// What one scheduler tick did, for the cost model.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefillTickLedger {
    chunks: u32,
    rows: usize,
    offset: usize,
    /// A path whose wall time is not "one chunk, then one decode" ran
    /// (mixed forward, batched prefill, two-phase, a start error).
    tainted: bool,
}

impl PrefillTickLedger {
    pub fn record_chunk(&mut self, rows: usize, offset: usize) {
        self.chunks += 1;
        self.rows = rows;
        self.offset = offset;
    }

    pub fn taint(&mut self) {
        self.tainted = true;
    }

    pub fn chunks(&self) -> u32 {
        self.chunks
    }
}

/// One tick's plan: the chunk plan plus the admission and continue gates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TickPlan {
    pub plan: ChunkPlan,
    pub decision: Decision,
    gated: bool,
}

impl TickPlan {
    /// `max_batch_size` to hand `drain_pending_requests_for_tick` (which admits
    /// `max_batch_size − active − prefilling`): while rows decode, a new prompt
    /// starts only on a chunk tick and only when nothing is mid-prefill (so
    /// SRPT orders pending admissions and multiple in-flight prefills without
    /// admitting a second concurrent prefill mid-chunk).
    pub fn admission_batch_cap(
        &self,
        max_batch_size: usize,
        active: usize,
        prefilling: usize,
    ) -> usize {
        if !self.gated {
            return max_batch_size;
        }
        let allowed = match self.decision {
            Decision::Chunk { .. } => usize::from(prefilling == 0),
            Decision::Skip | Decision::Passthrough => 0,
        };
        max_batch_size.min(active + prefilling + allowed)
    }

    /// Whether the continue phase may run a chunk: not on a skip tick, and
    /// not after chunk 0 of a new prompt already ran this tick.
    pub fn continue_allowed(&self, ledger: &PrefillTickLedger) -> bool {
        if !self.gated {
            return true;
        }
        matches!(self.decision, Decision::Chunk { .. }) && ledger.chunks() == 0
    }
}

/// Scheduler-owned SLO state.
#[derive(Clone, Debug)]
pub struct SloRuntime {
    widths: Option<PreparedWidths>,
    target_ms: u32,
    carry_align: usize,
    parity_width: Option<usize>,
    cost: ChunkCostModel,
    decode: DecodeStepClock,
    steps_since_chunk: u32,
    candidate_pending: bool,
    prediction: Option<(usize, usize, f64)>,
    cost_cache: Option<SloCacheHandle>,
}

impl SloRuntime {
    /// Engaged only with chunked prefill, a nonzero target, a multi-chunk
    /// model and valid prepared widths from the model.
    pub fn new(
        widths: &[usize],
        target_ms: u32,
        chunked: bool,
        single_chunk_model: bool,
        block_size: usize,
    ) -> Self {
        let engaged = chunked && target_ms > 0 && !single_chunk_model;
        let block = block_size.max(1);
        let carry_align = PREFILL_CHUNK_ALIGN / gcd(PREFILL_CHUNK_ALIGN, block) * block;
        let rt = Self {
            widths: engaged.then(|| PreparedWidths::new(widths)).flatten(),
            target_ms,
            carry_align,
            parity_width: None,
            cost: ChunkCostModel::default(),
            decode: DecodeStepClock::default(),
            steps_since_chunk: 0,
            candidate_pending: false,
            prediction: None,
            cost_cache: None,
        };
        #[cfg(not(test))]
        if rt.engaged() {
            let user_path = super::prefill_slo_cost::cache::resolve_cache_path();
            let seed_path = super::prefill_slo_cost::cache::resolve_seed_path(user_path.as_deref());
            let fingerprint = SloCostFingerprint::for_runtime(widths, block_size);
            return rt.with_cost_cache(user_path, seed_path, fingerprint);
        }
        rt
    }

    /// Seed cost/decode models from persisted disk cache and arm debounced + `Drop` flushing.
    pub fn with_cost_cache(
        mut self,
        user_path: Option<PathBuf>,
        seed_path: Option<PathBuf>,
        fingerprint: SloCostFingerprint,
    ) -> Self {
        if !self.engaged() {
            return self;
        }
        let key = fingerprint.encoded();
        if let Some(entry) = load_entry_with_seed(user_path.as_deref(), seed_path.as_deref(), &key)
        {
            self.cost = ChunkCostModel::from_persisted(&entry);
            self.decode = DecodeStepClock::from_persisted(entry.decode_step_ms);
            tracing::info!(
                fingerprint = %key,
                cached_cells = entry.cells.len(),
                decode_step_ms = ?entry.decode_step_ms,
                "loaded persistent SLO prefill cost cache"
            );
        }
        self.cost_cache = user_path.map(|path| SloCacheHandle::new(path, key));
        self
    }

    /// Best-effort flush of dirty learned SLO costs to disk.
    pub fn flush_cost_cache(&mut self) {
        if let Some(ref mut handle) = self.cost_cache {
            handle.flush(&self.cost, self.decode.measured_ms());
        }
    }

    fn note_cache_observation(&mut self) {
        let should_flush = self
            .cost_cache
            .as_mut()
            .is_some_and(|handle| handle.record_observation(Instant::now()));
        if should_flush {
            self.flush_cost_cache();
        }
    }

    pub fn engaged(&self) -> bool {
        self.widths.is_some()
    }

    pub fn carry_align(&self) -> usize {
        self.carry_align
    }

    /// Arm the parity diagnostic (engaged runtimes only).
    pub fn with_parity_width(mut self, width: Option<usize>) -> Self {
        let widths = self.widths;
        self.parity_width = width.filter(|w| widths.is_some_and(|ws| ws.as_slice().contains(w)));
        self
    }

    pub fn parity_width(&self) -> Option<usize> {
        self.parity_width
    }

    pub fn cost(&self) -> &ChunkCostModel {
        &self.cost
    }

    /// One startup line.
    pub fn startup_message(&self) -> String {
        match self.widths {
            None => format!(
                "PREFILL_INTERLEAVE off decode_gap_target_ms={} (needs chunked prefill, a nonzero \
                 target, a multi-chunk model and prepared fast-prefill widths)",
                self.target_ms
            ),
            Some(widths) => {
                let table: Vec<String> = widths
                    .as_slice()
                    .iter()
                    .map(|&w| format!("{w}:{:.0}ms", self.cost.predict_ms(w, 0)))
                    .collect();
                format!(
                    "PREFILL_INTERLEAVE on mode=slo decode_gap_target_ms={} widths=[{}] \
                     carry_align={} short_lane<={:.0}ms decode_step_default={:.0}ms: while a row \
                     decodes each tick runs at most one prefill chunk sized to the gap target; \
                     prefill numerics of those requests differ from a full-chunk run",
                    self.target_ms,
                    table.join(" "),
                    self.carry_align,
                    SHORT_LANE_FACTOR * f64::from(self.target_ms),
                    DecodeStepClock::DEFAULT_MS
                )
            }
        }
    }

    /// Plan one tick for `candidate` (the prompt mid-prefill, else the head
    /// of the pending queue).
    pub fn plan_tick(
        &mut self,
        candidate: Option<Candidate>,
        decode_rows_active: bool,
        cap: Option<usize>,
        max_prefill_tokens: usize,
        single_chunk_required: bool,
    ) -> TickPlan {
        self.candidate_pending = candidate.is_some();
        self.prediction = None;
        let Some(widths) = self.widths else {
            return TickPlan {
                plan: ChunkPlan {
                    cap_with_active_decode: cap,
                    ..ChunkPlan::default()
                },
                decision: Decision::Passthrough,
                gated: false,
            };
        };
        let base = ChunkPlan {
            cap_with_active_decode: cap,
            widths: Some(widths),
            slo: None,
            fallback_width: self.parity_width.unwrap_or(widths.smallest()),
            carry_align: self.carry_align,
        };
        let decision = match candidate {
            None => Decision::Passthrough,
            Some(c) if self.parity_width.is_some() && !single_chunk_required => {
                let rows = super::prefill_chunk_policy::bounded_len(
                    widths,
                    base,
                    c.offset,
                    c.remaining,
                    max_prefill_tokens,
                );
                Decision::Chunk {
                    rows,
                    reason: Reason::Parity,
                }
            }
            Some(c) => {
                let inputs = DecideInputs {
                    candidate: c,
                    widths,
                    max_prefill_tokens,
                    cap,
                    carry_align: self.carry_align,
                    single_chunk_required,
                    decode_rows_active,
                    target_ms: f64::from(self.target_ms),
                    decode_step_ms: self.decode.step_ms(),
                    steps_since_chunk: self.steps_since_chunk,
                };
                let cost = &self.cost;
                decide(&inputs, |rows, offset| cost.predict_ms(rows, offset))
            }
        };
        if decode_rows_active {
            crate::metrics::record_interleave_decision(decision.label());
        }
        let slo = match (decision, candidate) {
            (Decision::Chunk { rows, .. }, Some(c)) => {
                self.prediction = Some((rows, c.offset, self.cost.predict_ms(rows, c.offset)));
                Some(SloChunk {
                    rows,
                    remaining: c.remaining,
                })
            }
            _ => None,
        };
        TickPlan {
            plan: ChunkPlan { slo, ..base },
            decision,
            gated: decode_rows_active && candidate.is_some(),
        }
    }

    /// Feed one finished tick: a decode-only tick updates the decode clock
    /// (and the starvation counter while a prompt waits); a tick with exactly
    /// one untainted chunk followed by a decode step records the chunk's
    /// wall time `tick − decode step` in the cost model and `/metrics`.
    pub fn observe_tick(&mut self, ledger: PrefillTickLedger, decoded: bool, tick_ms: f64) {
        if self.widths.is_none() {
            return;
        }
        if ledger.chunks > 0 {
            self.steps_since_chunk = 0;
        } else if decoded && self.candidate_pending {
            self.steps_since_chunk = self.steps_since_chunk.saturating_add(1);
        } else if !self.candidate_pending {
            self.steps_since_chunk = 0;
        }
        if !decoded || ledger.tainted || !tick_ms.is_finite() || tick_ms < 0.0 {
            return;
        }
        if ledger.chunks == 0 {
            self.decode.observe(tick_ms);
            if let Some(ms) = self.decode.measured_ms() {
                crate::metrics::record_decode_step_ms(ms);
                self.note_cache_observation();
            }
            return;
        }
        let Some(decode_ms) = self.decode.measured_ms() else {
            return;
        };
        if ledger.chunks != 1 {
            return;
        }
        let measured = (tick_ms - decode_ms).max(0.0);
        let predicted = match self.prediction {
            Some((rows, offset, ms)) if rows == ledger.rows && offset == ledger.offset => ms,
            _ => self.cost.predict_ms(ledger.rows, ledger.offset),
        };
        if let Some(obs) = self
            .cost
            .observe(ledger.rows, ledger.offset, measured, predicted)
        {
            let band = super::prefill_slo_cost::BAND_LABELS[obs.band];
            crate::metrics::record_prefill_chunk(obs.bucket, band, obs.ewma_ms, obs.error_ms);
            self.note_cache_observation();
        }
    }

    /// Test/diagnostic view of the starvation counter.
    pub fn steps_since_chunk(&self) -> u32 {
        self.steps_since_chunk
    }
}

impl Drop for SloRuntime {
    fn drop(&mut self) {
        self.flush_cost_cache();
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
#[path = "prefill_slo_tests.rs"]
mod tests;
