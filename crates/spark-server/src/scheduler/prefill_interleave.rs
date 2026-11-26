// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in prefill/decode interleave (`ATLAS_PREFILL_INTERLEAVE=1`, default off).
//!
//! The legacy tick launches chunk 0 for EVERY request admitted in it and then
//! one continuation chunk before the decode step, so a burst of K long prompts
//! stalls every running stream for K + 1 chunks. With the interleave on and
//! decode rows present, a tick runs at most ONE prefill chunk (the head
//! mid-prefill request's next chunk, else one new request's chunk 0) and then
//! the decode step, and a chunk runs only after
//! `ATLAS_PREFILL_INTERLEAVE_DECODE_STEPS` decode steps since the previous one
//! (1 = TTFT first, larger = smoother inter-token latency). A tick that starts
//! without decode rows is exactly the legacy tick.
//!
//! On the Qwen4 fast-prefill family (`geometry_locked`), chunks automatically
//! step down to `ROOFLINE_FAST_CHUNK_TOKENS` (2048) by default whenever
//! `--max-prefill-tokens` is a larger multiple of 2048, and explicit
//! `ATLAS_PREFILL_INTERLEAVE_TOKEN_BUDGET` values are honored when they are
//! power-of-two multiples `>= 2048` dividing `--max-prefill-tokens`. Other
//! narrower budgets on the fast-prefill family (or any budget on MLA models)
//! are raised to the full chunk and logged; unlocked, non-MLA models honor any
//! 16-aligned budget directly.

use std::time::Instant;

use lazy_static::lazy_static;
use prometheus::{HistogramVec, register_histogram_vec};

pub(super) const ENABLE_ENV: &str = "ATLAS_PREFILL_INTERLEAVE";
pub(super) const DECODE_STEPS_ENV: &str = "ATLAS_PREFILL_INTERLEAVE_DECODE_STEPS";
pub(super) const TOKEN_BUDGET_ENV: &str = "ATLAS_PREFILL_INTERLEAVE_TOKEN_BUDGET";
/// Root selector of the Qwen4 fast-prefill family (`qwen4_prefill_moe::SELECTOR`).
const FAST_FAMILY_ENV: &str = "ATLAS_QWEN4_PREFILL_MOE_BATCH";
const DEFAULT_DECODE_STEPS: usize = 1;
const MAX_DECODE_STEPS: usize = 64;
/// Budget granularity: the 16-row prefill tile, which also covers the 4-token
/// GDN group that intermediate chunks are aligned to.
const BUDGET_ALIGN: usize = 16;
/// Roofline-saturated minimum chunk width on Qwen4 fast prefill (`MULTI_CHUNK_ROWS`),
/// halving active decode SSE gap vs `4096` while keeping Tensor Cores saturated.
const ROOFLINE_FAST_CHUNK_TOKENS: usize = 2048;

lazy_static! {
    /// Tick start to next tick start while decode rows exist: the inter-token
    /// gap every running request sees for that tick (one decode step or one
    /// verify round per tick). `prefill_chunks` > 0 buckets are decode stalls.
    static ref DECODE_TICK_SECONDS: HistogramVec = register_histogram_vec!(
        "atlas_scheduler_decode_tick_seconds",
        "Wall time of a scheduler tick with decode rows, by prefill chunks run in it",
        &["prefill_chunks"],
        vec![0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]
    )
    .unwrap();
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct InterleaveConfig {
    pub decode_steps_per_chunk: usize,
    /// Prefill chunk size while decode rows exist.
    pub chunk_tokens: usize,
    /// A requested budget the scheduler could not honour, and why.
    pub budget_ignored: Option<(usize, &'static str)>,
}

/// Strict parse of the three selectors. `Ok(None)` is the legacy scheduler.
pub(super) fn parse_config(
    enable: Option<&str>,
    decode_steps: Option<&str>,
    token_budget: Option<&str>,
    max_prefill_tokens: usize,
    geometry_locked: bool,
    mla: bool,
) -> Result<Option<InterleaveConfig>, String> {
    match enable {
        None | Some("0") => return Ok(None),
        Some("1") => {}
        Some(other) => {
            return Err(format!(
                "{ENABLE_ENV} must be absent, 0, or 1; got {other:?}"
            ));
        }
    }
    let decode_steps_per_chunk = match decode_steps {
        None => DEFAULT_DECODE_STEPS,
        Some(raw) => match raw.parse::<usize>() {
            Ok(steps) if (1..=MAX_DECODE_STEPS).contains(&steps) => steps,
            _ => {
                return Err(format!(
                    "{DECODE_STEPS_ENV} must be an integer in 1..={MAX_DECODE_STEPS}; got {raw:?}"
                ));
            }
        },
    };
    let budget = match token_budget {
        None | Some("0") => None,
        Some(raw) => match raw.parse::<usize>() {
            Ok(tokens) if tokens >= BUDGET_ALIGN && tokens.is_multiple_of(BUDGET_ALIGN) => {
                Some(tokens)
            }
            _ => {
                return Err(format!(
                    "{TOKEN_BUDGET_ENV} must be 0 or a positive multiple of {BUDGET_ALIGN}; got {raw:?}"
                ));
            }
        },
    };
    let (chunk_tokens, budget_ignored) = match budget {
        None if geometry_locked
            && !mla
            && max_prefill_tokens > ROOFLINE_FAST_CHUNK_TOKENS
            && max_prefill_tokens.is_multiple_of(ROOFLINE_FAST_CHUNK_TOKENS) =>
        {
            (ROOFLINE_FAST_CHUNK_TOKENS, None)
        }
        None => (max_prefill_tokens, None),
        Some(tokens) if max_prefill_tokens == 0 => (0, Some((tokens, "prefill is not chunked"))),
        // Multi-chunk MLA prefill attends over the current chunk only.
        Some(tokens) if mla => (
            max_prefill_tokens,
            Some((tokens, "MLA prefill is single-chunk")),
        ),
        Some(tokens) if tokens >= max_prefill_tokens => (max_prefill_tokens, None),
        Some(tokens)
            if geometry_locked
                && tokens >= ROOFLINE_FAST_CHUNK_TOKENS
                && tokens.is_power_of_two()
                && max_prefill_tokens.is_multiple_of(tokens) =>
        {
            (tokens, None)
        }
        Some(tokens) if geometry_locked => (
            max_prefill_tokens,
            Some((
                tokens,
                "Qwen4 fast prefill requires a power-of-two budget >= 2048 dividing max_prefill_tokens",
            )),
        ),
        Some(tokens) => (tokens, None),
    };
    Ok(Some(InterleaveConfig {
        decode_steps_per_chunk,
        chunk_tokens,
        budget_ignored,
    }))
}

/// True when the Qwen4 fast-prefill family restricts chunk widths to
/// roofline-saturated power-of-two divisors `>= ROOFLINE_FAST_CHUNK_TOKENS`
/// (2048) of `--max-prefill-tokens` rather than arbitrary 16-aligned budgets.
pub(super) fn fast_geometry_locked(hc_mult: usize, fast_family: Option<&str>) -> bool {
    hc_mult > 0 && fast_family.is_some_and(|value| value != "0")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Decision {
    /// Interleave disabled: legacy tick.
    Off,
    /// No decode rows at tick start: legacy tick.
    Idle,
    /// Fewer than the configured decode steps since the last chunk.
    NotDue,
    /// Advance the head mid-prefill request by one chunk.
    Continue,
    /// Nothing mid-prefill: start at most one queued request (its chunk 0).
    StartOne,
}

impl Decision {
    fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Idle => "idle",
            Self::NotDue => "not_due",
            Self::Continue => "continue",
            Self::StartOne => "start_one",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct TickShape {
    pub decision: Decision,
    /// Requests this tick may start; `None` keeps the legacy admission cap.
    pub start_cap: Option<usize>,
    /// Whether `continue_in_progress_prefills` may launch a chunk.
    pub continue_allowed: bool,
    pub chunk_tokens: usize,
}

pub(super) fn decide(
    config: Option<InterleaveConfig>,
    decode_steps_since_chunk: usize,
    active_len: usize,
    prefilling_len: usize,
    max_prefill_tokens: usize,
) -> TickShape {
    let legacy = |decision| TickShape {
        decision,
        start_cap: None,
        continue_allowed: true,
        chunk_tokens: max_prefill_tokens,
    };
    let Some(config) = config else {
        return legacy(Decision::Off);
    };
    if active_len == 0 {
        return legacy(Decision::Idle);
    }
    let gated = |decision, start_cap, continue_allowed| TickShape {
        decision,
        start_cap: Some(start_cap),
        continue_allowed,
        chunk_tokens: config.chunk_tokens,
    };
    if decode_steps_since_chunk < config.decode_steps_per_chunk {
        gated(Decision::NotDue, 0, false)
    } else if prefilling_len > 0 {
        gated(Decision::Continue, 0, true)
    } else {
        gated(Decision::StartOne, 1, false)
    }
}

/// Decode steps since the last chunk after a tick. The decode step that
/// follows a chunk in the same tick counts as the first one.
pub(super) fn next_decode_steps(previous: usize, chunks_run: usize, decode_ran: bool) -> usize {
    let base = if chunks_run > 0 { 0 } else { previous };
    if decode_ran {
        base.saturating_add(1)
    } else {
        base
    }
}

/// Prefill chunks accepted so far (`record_prefill_progress` receipts).
pub(super) fn prefill_chunks_recorded() -> u64 {
    crate::metrics::PREFILL_PROGRESS_CHUNKS_TOTAL.get()
}

/// Chunks one tick ran. Non-chunked starts publish no progress receipt, so
/// each started request counts as one chunk there.
pub(super) fn chunks_run(before: u64, after: u64, started: usize, chunked: bool) -> usize {
    let recorded = usize::try_from(after.saturating_sub(before)).unwrap_or(usize::MAX);
    if chunked {
        recorded
    } else {
        recorded.max(started)
    }
}

fn chunk_label(chunks: usize) -> &'static str {
    match chunks {
        0 => "0",
        1 => "1",
        _ => "2+",
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TickPlan {
    pub shape: TickShape,
    started_at: Instant,
    decode_rows_at_start: bool,
}

impl TickPlan {
    /// The `max_batch_size` argument for `drain_pending_requests`, whose
    /// capacity is that value minus `active + prefilling`.
    pub(super) fn drain_limit(
        &self,
        max_batch_size: usize,
        active_len: usize,
        prefilling_len: usize,
    ) -> usize {
        match self.shape.start_cap {
            None => max_batch_size,
            Some(cap) => max_batch_size.min(active_len + prefilling_len + cap),
        }
    }

    pub(super) fn continue_allowed(&self) -> bool {
        self.shape.continue_allowed
    }

    pub(super) fn chunk_tokens(&self) -> usize {
        self.shape.chunk_tokens
    }
}

pub(super) struct PrefillInterleave {
    config: Option<InterleaveConfig>,
    decode_steps_since_chunk: usize,
    /// Start and chunk count of the previous tick, when it began with decode rows.
    last_decode_tick: Option<(Instant, usize)>,
}

impl PrefillInterleave {
    pub(super) fn new(config: Option<InterleaveConfig>) -> Self {
        Self {
            config,
            // The first tick with decode rows is due.
            decode_steps_since_chunk: usize::MAX,
            last_decode_tick: None,
        }
    }

    pub(super) fn is_enabled(&self) -> bool {
        self.config.is_some()
    }

    /// Resolve the selectors once at scheduler start; a malformed value keeps
    /// the legacy scheduler and says so.
    pub(super) fn from_env(max_prefill_tokens: usize, hc_mult: usize, mla: bool) -> Self {
        let env = |name: &str| std::env::var(name).ok();
        let locked = fast_geometry_locked(hc_mult, env(FAST_FAMILY_ENV).as_deref());
        match parse_config(
            env(ENABLE_ENV).as_deref(),
            env(DECODE_STEPS_ENV).as_deref(),
            env(TOKEN_BUDGET_ENV).as_deref(),
            max_prefill_tokens,
            locked,
            mla,
        ) {
            Ok(None) => Self::new(None),
            Ok(Some(config)) => {
                tracing::info!(
                    "PREFILL_INTERLEAVE on decode_steps_per_chunk={} chunk_tokens={} full_chunk={max_prefill_tokens} geometry_locked={locked}",
                    config.decode_steps_per_chunk,
                    config.chunk_tokens,
                );
                if let Some((budget, reason)) = config.budget_ignored {
                    tracing::warn!(
                        "PREFILL_INTERLEAVE {TOKEN_BUDGET_ENV}={budget} not applied ({reason}); chunks stay {} tokens",
                        config.chunk_tokens,
                    );
                }
                Self::new(Some(config))
            }
            Err(error) => {
                tracing::error!("PREFILL_INTERLEAVE off: {error}");
                Self::new(None)
            }
        }
    }

    pub(super) fn plan_tick(
        &mut self,
        now: Instant,
        active_len: usize,
        prefilling_len: usize,
        max_prefill_tokens: usize,
    ) -> TickPlan {
        // A previous decode tick closes only if this one also has decode rows;
        // otherwise the gap includes idle waiting for new requests.
        if let Some((start, chunks)) = self.last_decode_tick.take()
            && active_len > 0
        {
            DECODE_TICK_SECONDS
                .with_label_values(&[chunk_label(chunks)])
                .observe(now.duration_since(start).as_secs_f64());
        }
        let shape = decide(
            self.config,
            self.decode_steps_since_chunk,
            active_len,
            prefilling_len,
            max_prefill_tokens,
        );
        crate::metrics::record_interleave_decision(shape.decision.label());
        TickPlan {
            shape,
            started_at: now,
            decode_rows_at_start: active_len > 0,
        }
    }

    pub(super) fn finish_tick(&mut self, plan: &TickPlan, chunks_run: usize, decode_rows: bool) {
        self.decode_steps_since_chunk =
            next_decode_steps(self.decode_steps_since_chunk, chunks_run, decode_rows);
        if decode_rows && plan.decode_rows_at_start {
            self.last_decode_tick = Some((plan.started_at, chunks_run));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on(steps: usize, chunk_tokens: usize) -> Option<InterleaveConfig> {
        Some(InterleaveConfig {
            decode_steps_per_chunk: steps,
            chunk_tokens,
            budget_ignored: None,
        })
    }

    #[test]
    fn absent_or_zero_selector_is_the_legacy_scheduler() {
        for enable in [None, Some("0")] {
            assert_eq!(
                parse_config(enable, Some("9"), Some("512"), 4096, false, false),
                Ok(None)
            );
        }
        for (active, prefilling) in [(0, 0), (3, 0), (3, 2), (0, 2)] {
            assert_eq!(
                decide(None, 0, active, prefilling, 4096),
                TickShape {
                    decision: Decision::Off,
                    start_cap: None,
                    continue_allowed: true,
                    chunk_tokens: 4096,
                }
            );
        }
    }

    #[test]
    fn selectors_parse_strictly() {
        for bad in ["true", "2", ""] {
            assert!(parse_config(Some(bad), None, None, 4096, false, false).is_err());
        }
        for bad in ["0", "65", "-1", "x"] {
            assert!(parse_config(Some("1"), Some(bad), None, 4096, false, false).is_err());
        }
        for bad in ["8", "100", "x"] {
            assert!(parse_config(Some("1"), None, Some(bad), 4096, false, false).is_err());
        }
        assert_eq!(
            parse_config(Some("1"), None, None, 4096, false, false),
            Ok(on(1, 4096))
        );
        assert_eq!(
            parse_config(Some("1"), Some("64"), Some("0"), 4096, false, false),
            Ok(on(64, 4096))
        );
    }

    #[test]
    fn budget_shrinks_only_unlocked_non_mla_chunks() {
        let parse = |locked, mla| {
            parse_config(Some("1"), None, Some("1024"), 4096, locked, mla)
                .unwrap()
                .unwrap()
        };
        assert_eq!(parse(false, false).chunk_tokens, 1024);
        assert_eq!(parse(false, false).budget_ignored, None);
        for (locked, mla) in [(true, false), (false, true), (true, true)] {
            let config = parse(locked, mla);
            assert_eq!(config.chunk_tokens, 4096, "locked={locked} mla={mla}");
            assert_eq!(config.budget_ignored.map(|(tokens, _)| tokens), Some(1024));
        }
        let wide = parse_config(Some("1"), None, Some("8192"), 4096, true, false)
            .unwrap()
            .unwrap();
        assert_eq!((wide.chunk_tokens, wide.budget_ignored), (4096, None));
        let unchunked = parse_config(Some("1"), None, Some("512"), 0, false, false)
            .unwrap()
            .unwrap();
        assert_eq!(unchunked.chunk_tokens, 0);
    }

    #[test]
    fn geometry_lock_needs_a_hyper_connected_model_and_the_fast_family() {
        assert!(fast_geometry_locked(4, Some("1")));
        assert!(!fast_geometry_locked(4, Some("0")));
        assert!(!fast_geometry_locked(4, None));
        assert!(!fast_geometry_locked(0, Some("1")));
    }

    #[test]
    fn a_tick_without_decode_rows_is_the_legacy_tick() {
        let shape = decide(on(4, 1024), 0, 0, 3, 4096);
        assert_eq!(shape.decision, Decision::Idle);
        assert_eq!(shape.start_cap, None);
        assert!(shape.continue_allowed);
        assert_eq!(shape.chunk_tokens, 4096, "idle prefill keeps full chunks");
    }

    #[test]
    fn a_decode_tick_runs_at_most_one_chunk() {
        let mid = decide(on(1, 4096), 1, 5, 2, 4096);
        assert_eq!(
            (mid.decision, mid.start_cap, mid.continue_allowed),
            (Decision::Continue, Some(0), true)
        );
        let fresh = decide(on(1, 4096), 1, 5, 0, 4096);
        assert_eq!(
            (fresh.decision, fresh.start_cap, fresh.continue_allowed),
            (Decision::StartOne, Some(1), false)
        );
        let wait = decide(on(3, 4096), 2, 5, 2, 4096);
        assert_eq!(
            (wait.decision, wait.start_cap, wait.continue_allowed),
            (Decision::NotDue, Some(0), false)
        );
    }

    #[test]
    fn drain_limit_caps_new_starts() {
        let plan = |shape| TickPlan {
            shape,
            started_at: Instant::now(),
            decode_rows_at_start: true,
        };
        assert_eq!(plan(decide(None, 0, 3, 1, 4096)).drain_limit(8, 3, 1), 8);
        assert_eq!(
            plan(decide(on(1, 4096), 1, 3, 0, 4096)).drain_limit(8, 3, 0),
            4
        );
        assert_eq!(
            plan(decide(on(1, 4096), 1, 3, 1, 4096)).drain_limit(8, 3, 1),
            4
        );
        assert_eq!(
            plan(decide(on(1, 4096), 1, 8, 0, 4096)).drain_limit(8, 8, 0),
            8
        );
    }

    #[test]
    fn chunks_are_spaced_by_the_configured_decode_steps() {
        let mut state = PrefillInterleave::new(on(3, 4096));
        let mut pattern = Vec::new();
        for _ in 0..9 {
            let plan = state.plan_tick(Instant::now(), 2, 1, 4096);
            let chunks = usize::from(plan.continue_allowed());
            pattern.push(chunks);
            state.finish_tick(&plan, chunks, true);
        }
        assert_eq!(pattern, [1, 0, 0, 1, 0, 0, 1, 0, 0]);
    }

    #[test]
    fn a_due_tick_that_ran_nothing_stays_due() {
        assert_eq!(next_decode_steps(5, 0, true), 6);
        assert_eq!(next_decode_steps(5, 1, true), 1);
        assert_eq!(next_decode_steps(5, 2, false), 0);
        assert_eq!(next_decode_steps(usize::MAX, 0, true), usize::MAX);
    }

    #[test]
    fn chunk_count_uses_progress_receipts() {
        assert_eq!(chunks_run(10, 12, 1, true), 2);
        assert_eq!(
            chunks_run(10, 10, 1, true),
            0,
            "a deferred or failed start ran nothing"
        );
        assert_eq!(chunks_run(10, 10, 2, false), 2);
        assert_eq!(chunks_run(10, 9, 0, true), 0);
    }

    #[test]
    fn fast_geometry_steps_down_to_roofline_2048_during_interleave() {
        assert_eq!(
            parse_config(Some("1"), Some("3"), None, 4096, true, false),
            Ok(on(3, 2048))
        );
        assert_eq!(
            parse_config(Some("1"), None, Some("2048"), 8192, true, false),
            Ok(on(1, 2048))
        );
        assert_eq!(
            parse_config(Some("1"), None, Some("4096"), 8192, true, false),
            Ok(on(1, 4096))
        );
        let idle = decide(on(3, 2048), 3, 0, 1, 4096);
        assert_eq!(idle.decision, Decision::Idle);
        assert_eq!(idle.chunk_tokens, 4096);
        let active_start = decide(on(3, 2048), 3, 1, 0, 4096);
        assert_eq!(active_start.decision, Decision::StartOne);
        assert_eq!(active_start.chunk_tokens, 2048);
    }
}
