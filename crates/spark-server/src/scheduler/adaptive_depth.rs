// SPDX-License-Identifier: AGPL-3.0-only

//! Adaptive draft depth (`0..=8`) from online empirical per-step survival `p[d]`.
//!
//! Pure controller: [`DepthController`] is an immutable value; `observe`
//! returns a new one. In addition to the scalar geometric acceptance rate `a`,
//! the controller tracks empirical conditional per-step survival probabilities
//! `p_j = P(accept draft j | reached draft j)` for `j in 0..8` and scores each
//! candidate depth `d in 0..=max_depth` by exact expected tokens per unit wall:
//!
//!   `Rate(d) = (1 + sum_{k=1}^d prod_{j=0}^{k-1} p_j) / (1 + draft_cost * d)`
//!
//! picking the `d` maximizing `Rate(d)` (smaller `d` on ties). Depth `0` is a
//! valid answer; while there, every `probe_interval`-th frame drafts once so
//! the estimate can recover when text predictability improves.
//!
//! Per-sequence isolation: requests maintain independent [`DepthController`]
//! states keyed by `session_hash` (via [`ActiveDepthSessionGuard`] at verify
//! settlement and [`seq_next_draft_depth`] at proposal) so one request's
//! survival history never leaks into another. Multi-sequence steps (`live >= 2`)
//! bypass adaptive depth via [`adaptive_depth_allowed_for_live`] so batched
//! cohorts keep uniform draft widths.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;

/// Maximum supported adaptive draft depth (`QWEN4_CROSS_SEQ_MAX_K - 1 == 8`).
pub(super) const MAX_ADAPTIVE_DEPTH: usize = spark_model::layer::QWEN4_CROSS_SEQ_MAX_K - 1;

/// Maximum tracked concurrent/recent sequence controllers before evicting oldest.
const MAX_TRACKED_SEQS: usize = 256;

/// Cap on accumulated effective trial weight per draft slot in
/// [`PopulationAcceptancePrior`], bounding memory so workload shifts adapt smoothly.
pub(super) const POPULATION_WEIGHT_CAP: f32 = 128.0;

/// Effective prior pseudo-count granted to a fresh sequence seeded from the
/// population survival curve (`alpha_eff = 1 / (w + 1)` starts at `1/3`).
pub(super) const NEW_REQUEST_PRIOR_WEIGHT: f32 = 2.0;

/// Lower floor on slot-0 prior survival so a noisy population never starves a
/// fresh sequence at depth `0` on frame 0.
const NEW_REQUEST_P0_FLOOR: f32 = 0.25;

/// Controls whether fresh per-sequence controllers inherit the bounded
/// population survival curve or start isolated at `cfg.initial_rate`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PopulationPriorMode {
    Enabled,
    Disabled,
}

impl PopulationPriorMode {
    pub(super) fn from_env() -> Self {
        let prior_raw = std::env::var("FLUX_SPEC_POPULATION_PRIOR")
            .or_else(|_| std::env::var("ATLAS_SPEC_POPULATION_PRIOR"))
            .ok();
        let det_raw = std::env::var("FLUX_SPEC_DETERMINISTIC")
            .or_else(|_| std::env::var("ATLAS_SPEC_DETERMINISTIC"))
            .ok();
        Self::parse(prior_raw.as_deref(), det_raw.as_deref())
    }

    pub(super) fn parse(prior_env: Option<&str>, deterministic_env: Option<&str>) -> Self {
        if deterministic_env.map(str::trim) == Some("1") {
            return Self::Disabled;
        }
        match prior_env.map(str::trim) {
            Some("0") => Self::Disabled,
            _ => Self::Enabled,
        }
    }
}

/// Bounded-weight process-wide per-slot empirical acceptance prior.
///
/// Each evaluated draft slot accumulates `(accepted_w[j], trials_w[j])`. Once
/// `trials_w[j] > POPULATION_WEIGHT_CAP`, both counters decay by
/// `(CAP - 1) / CAP = 127 / 128`, maintaining `0 <= accepted_w[j] <= trials_w[j] <= 128`
/// while allowing workload shifts to take over with an ~88-trial half-life.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PopulationAcceptancePrior {
    accepted_w: [f32; MAX_ADAPTIVE_DEPTH],
    trials_w: [f32; MAX_ADAPTIVE_DEPTH],
}

impl Default for PopulationAcceptancePrior {
    fn default() -> Self {
        Self::new()
    }
}

impl PopulationAcceptancePrior {
    pub(super) const fn new() -> Self {
        Self {
            accepted_w: [0.0; MAX_ADAPTIVE_DEPTH],
            trials_w: [0.0; MAX_ADAPTIVE_DEPTH],
        }
    }

    /// Fold one settled frame into the per-slot bounded EWMA counts.
    pub(super) fn observe(&mut self, drafted: usize, accepted: usize) {
        if drafted == 0 {
            return;
        }
        let accepted = accepted.min(drafted);
        for slot in 0..accepted.min(MAX_ADAPTIVE_DEPTH) {
            self.observe_slot(slot, 1.0);
        }
        if accepted < drafted && accepted < MAX_ADAPTIVE_DEPTH {
            self.observe_slot(accepted, 0.0);
        }
    }

    fn observe_slot(&mut self, slot: usize, outcome: f32) {
        let decay = (POPULATION_WEIGHT_CAP - 1.0) / POPULATION_WEIGHT_CAP;
        let mut trials = self.trials_w[slot] + 1.0;
        let mut accepted = self.accepted_w[slot] + outcome;
        if trials > POPULATION_WEIGHT_CAP {
            trials *= decay;
            accepted *= decay;
        }
        trials = trials.clamp(0.0, POPULATION_WEIGHT_CAP);
        accepted = accepted.clamp(0.0, trials);
        self.trials_w[slot] = trials;
        self.accepted_w[slot] = accepted;
    }

    /// Effective accumulated trial weight for `slot` (`0.0` if out of range).
    pub(super) fn slot_weight(&self, slot: usize) -> f32 {
        self.trials_w.get(slot).copied().unwrap_or(0.0)
    }

    /// Posterior conditional survival probability for each draft slot,
    /// shrinking toward `fallback_rate` when a slot has few or zero trials.
    pub(super) fn survival_curve(&self, fallback_rate: f32) -> [f32; MAX_ADAPTIVE_DEPTH] {
        let fallback = fallback_rate.clamp(0.0, 1.0);
        let mut curve = [fallback; MAX_ADAPTIVE_DEPTH];
        for (slot, dst) in curve.iter_mut().enumerate() {
            let trials = self.slot_weight(slot);
            if trials > 0.0 {
                let pseudo = NEW_REQUEST_PRIOR_WEIGHT;
                *dst = ((self.accepted_w[slot] + pseudo * fallback) / (trials + pseudo))
                    .clamp(0.0, 1.0);
            }
        }
        curve
    }
}

/// Controller tuning.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DepthConfig {
    pub max_depth: usize,
    /// EWMA weight of one new Bernoulli acceptance observation.
    pub alpha: f32,
    /// Marginal cost of one draft relative to a plain decode step.
    pub draft_cost: f32,
    /// At depth 0, probe with one draft every this many frames.
    pub probe_interval: u32,
    /// Prior acceptance rate before any observation.
    pub initial_rate: f32,
}

impl Default for DepthConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            alpha: 0.05,
            draft_cost: 0.15,
            probe_interval: 16,
            initial_rate: 0.7,
        }
    }
}

impl DepthConfig {
    /// Runtime configuration unlocking full `0..=8` adaptive depth with
    /// GB10 MTP marginal draft cost (`0.12`) and survival prior (`0.78`).
    pub(super) fn runtime() -> Self {
        let max_depth = std::env::var("FLUX_SPEC_MAX_DEPTH")
            .or_else(|_| std::env::var("ATLAS_SPEC_MAX_DEPTH"))
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .map_or(MAX_ADAPTIVE_DEPTH, |d| d.min(MAX_ADAPTIVE_DEPTH));
        let draft_cost = std::env::var("FLUX_SPEC_DRAFT_COST")
            .or_else(|_| std::env::var("ATLAS_SPEC_DRAFT_COST"))
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|c| c.is_finite() && *c > 0.0)
            .unwrap_or(0.12);
        let initial_rate = std::env::var("FLUX_SPEC_INITIAL_RATE")
            .or_else(|_| std::env::var("ATLAS_SPEC_INITIAL_RATE"))
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|r| r.is_finite() && (0.0..=1.0).contains(r))
            .unwrap_or(0.78);
        Self {
            max_depth,
            draft_cost,
            initial_rate,
            ..Self::default()
        }
    }
}

/// Acceptance-rate state; a value, never mutated in place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DepthController {
    cfg: DepthConfig,
    rate: f32,
    step_rates: [f32; MAX_ADAPTIVE_DEPTH],
    step_weights: [f32; MAX_ADAPTIVE_DEPTH],
    frames_at_zero: u32,
}

#[inline]
fn steady_step_weight(alpha: f32) -> f32 {
    if alpha.is_finite() && alpha > 0.0 {
        (1.0 / alpha).max(1.0)
    } else {
        f32::INFINITY
    }
}

#[inline]
fn effective_alpha(weight: f32, base_alpha: f32) -> f32 {
    if !base_alpha.is_finite() || base_alpha <= 0.0 {
        return 0.0;
    }
    if !weight.is_finite() || weight <= 0.0 {
        return base_alpha.clamp(0.0, 1.0);
    }
    (1.0 / (weight + 1.0)).max(base_alpha).clamp(0.0, 1.0)
}

impl DepthController {
    pub(super) fn new(cfg: DepthConfig) -> Self {
        let clamped_max = cfg.max_depth.min(MAX_ADAPTIVE_DEPTH);
        let init = cfg.initial_rate.clamp(0.0, 1.0);
        let steady_w = steady_step_weight(cfg.alpha);
        Self {
            cfg: DepthConfig {
                max_depth: clamped_max,
                ..cfg
            },
            rate: init,
            step_rates: [init; MAX_ADAPTIVE_DEPTH],
            step_weights: [steady_w; MAX_ADAPTIVE_DEPTH],
            frames_at_zero: 0,
        }
    }

    /// Seed a new per-sequence controller from [`PopulationAcceptancePrior`]
    /// with low initial per-slot weight `prior_weight` (typically
    /// [`NEW_REQUEST_PRIOR_WEIGHT`]) and slot-0 floor [`NEW_REQUEST_P0_FLOOR`],
    /// so warm requests start at optimal depth on frame 0 while mismatched
    /// requests converge in 2–5 frames.
    pub(super) fn from_population_prior(
        cfg: DepthConfig,
        prior: &PopulationAcceptancePrior,
        prior_weight: f32,
    ) -> Self {
        let base = Self::new(cfg);
        let mut step_rates = prior.survival_curve(base.rate);
        step_rates[0] = step_rates[0].max(NEW_REQUEST_P0_FLOOR).clamp(0.0, 1.0);
        let steady_w = steady_step_weight(base.cfg.alpha);
        let seeded_w = if prior_weight.is_finite() && prior_weight > 0.0 {
            prior_weight.min(steady_w)
        } else {
            steady_w
        };
        Self {
            rate: step_rates[0],
            step_rates,
            step_weights: [seeded_w; MAX_ADAPTIVE_DEPTH],
            ..base
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn rate(&self) -> f32 {
        self.rate
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn step_rates(&self) -> &[f32; MAX_ADAPTIVE_DEPTH] {
        &self.step_rates
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn step_weights(&self) -> &[f32; MAX_ADAPTIVE_DEPTH] {
        &self.step_weights
    }

    /// Fold one frame: `accepted` of `drafted` drafts were accepted.
    /// Trials are the positions actually evaluated: every accepted draft
    /// (`0..accepted`), plus the rejected position `accepted` if `accepted < drafted`.
    pub(super) fn observe(&self, drafted: usize, accepted: usize) -> Self {
        if drafted == 0 {
            return *self;
        }
        let accepted = accepted.min(drafted);
        let rejected = usize::from(accepted < drafted);
        let after_accepts = (0..accepted).fold(self.rate, |r, _| ewma(r, 1.0, self.cfg.alpha));
        let rate = (0..rejected).fold(after_accepts, |r, _| ewma(r, 0.0, self.cfg.alpha));

        let steady_w = steady_step_weight(self.cfg.alpha);
        let mut step_rates = self.step_rates;
        let mut step_weights = self.step_weights;
        for slot in 0..accepted.min(MAX_ADAPTIVE_DEPTH) {
            let alpha_eff = effective_alpha(step_weights[slot], self.cfg.alpha);
            step_rates[slot] = ewma(step_rates[slot], 1.0, alpha_eff);
            step_weights[slot] = (step_weights[slot] + 1.0).min(steady_w);
        }
        if accepted < drafted && accepted < MAX_ADAPTIVE_DEPTH {
            let alpha_eff = effective_alpha(step_weights[accepted], self.cfg.alpha);
            step_rates[accepted] = ewma(step_rates[accepted], 0.0, alpha_eff);
            step_weights[accepted] = (step_weights[accepted] + 1.0).min(steady_w);
        }
        Self {
            rate,
            step_rates,
            step_weights,
            ..*self
        }
    }

    /// Best depth `0..=max_depth` maximizing empirical `Rate(d)`, without probing.
    pub(super) fn best_depth(&self) -> usize {
        best_depth_empirical(&self.step_rates, self.cfg.draft_cost, self.cfg.max_depth)
    }

    /// Best depth capped at `limit` (`<= max_depth`), without probing.
    pub(super) fn best_depth_capped(&self, limit: usize) -> usize {
        best_depth_empirical(
            &self.step_rates,
            self.cfg.draft_cost,
            self.cfg.max_depth.min(limit),
        )
    }

    /// Geometric-model best depth (retained for comparison & diagnostics).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn best_depth_geometric(&self) -> usize {
        best_depth(self.rate, self.cfg.draft_cost, self.cfg.max_depth)
    }

    /// The depth to use for the next frame, and the controller after it.
    pub(super) fn next_frame(&self) -> (usize, Self) {
        self.next_frame_capped(self.cfg.max_depth)
    }

    /// Like [`Self::next_frame`], bounded above by `limit`.
    pub(super) fn next_frame_capped(&self, limit: usize) -> (usize, Self) {
        if limit == 0 {
            return (0, *self);
        }
        let best = self.best_depth_capped(limit);
        if best > 0 {
            return (
                best,
                Self {
                    frames_at_zero: 0,
                    ..*self
                },
            );
        }
        let frames_at_zero = self.frames_at_zero.saturating_add(1);
        let probe = self.cfg.probe_interval > 0 && frames_at_zero >= self.cfg.probe_interval;
        let next = Self {
            frames_at_zero: if probe { 0 } else { frames_at_zero },
            ..*self
        };
        (usize::from(probe), next)
    }
}

fn ewma(rate: f32, sample: f32, alpha: f32) -> f32 {
    (rate + alpha * (sample - rate)).clamp(0.0, 1.0)
}

/// Expected tokens per frame at depth `k` under constant geometric acceptance `a`.
pub(super) fn expected_tokens(a: f32, k: usize) -> f32 {
    (0..=k).map(|j| a.powi(j as i32)).sum()
}

/// Expected tokens per frame `1 + sum_{k=1}^d prod_{j=0}^{k-1} p_j` from empirical `p_j`.
pub(super) fn expected_tokens_empirical(step_rates: &[f32], depth: usize) -> f32 {
    let depth = depth.min(step_rates.len());
    let mut total = 1.0f32;
    let mut survival = 1.0f32;
    for &p in step_rates.iter().take(depth) {
        survival *= p.clamp(0.0, 1.0);
        total += survival;
    }
    total
}

/// Empirical throughput objective `Rate(d) = E_emp(d) / (1 + draft_cost * d)`.
pub(super) fn rate_score_empirical(step_rates: &[f32], draft_cost: f32, depth: usize) -> f32 {
    expected_tokens_empirical(step_rates, depth) / (1.0 + draft_cost * depth as f32)
}

#[cfg_attr(not(test), allow(dead_code))]
fn best_depth(rate: f32, draft_cost: f32, max_depth: usize) -> usize {
    let score = |k: usize| expected_tokens(rate, k) / (1.0 + draft_cost * k as f32);
    (1..=max_depth).fold(0, |best, k| if score(k) > score(best) { k } else { best })
}

fn best_depth_empirical(step_rates: &[f32], draft_cost: f32, max_depth: usize) -> usize {
    let max_d = max_depth.min(step_rates.len());
    let score = |d: usize| rate_score_empirical(step_rates, draft_cost, d);
    (1..=max_d).fold(0, |best, d| {
        if score(d) > score(best) + 1.0e-6 {
            d
        } else {
            best
        }
    })
}

// ── process-wide + per-sequence wiring ───────────────────────────────────────

struct SeqControllerEntry {
    controller: DepthController,
    last_depth: usize,
    tick: u64,
}

struct PerSeqDepthState {
    global: DepthController,
    population_prior: PopulationAcceptancePrior,
    prior_mode: PopulationPriorMode,
    per_seq: HashMap<u64, SeqControllerEntry>,
    tick: u64,
}

impl PerSeqDepthState {
    #[cfg_attr(not(test), allow(dead_code))]
    fn new(cfg: DepthConfig) -> Self {
        Self::new_with_mode(cfg, PopulationPriorMode::Enabled)
    }

    fn new_with_mode(cfg: DepthConfig, prior_mode: PopulationPriorMode) -> Self {
        Self {
            global: DepthController::new(cfg),
            population_prior: PopulationAcceptancePrior::new(),
            prior_mode,
            per_seq: HashMap::new(),
            tick: 0,
        }
    }

    fn observe_population(&mut self, drafted: usize, accepted: usize) {
        if drafted == 0 {
            return;
        }
        self.global = self.global.observe(drafted, accepted);
        if self.prior_mode == PopulationPriorMode::Enabled {
            self.population_prior.observe(drafted, accepted);
        }
    }

    fn touch_or_insert(&mut self, session_hash: u64) -> &mut SeqControllerEntry {
        self.tick = self.tick.wrapping_add(1);
        let now = self.tick;
        if !self.per_seq.contains_key(&session_hash)
            && self.per_seq.len() >= MAX_TRACKED_SEQS
            && let Some((&oldest_key, _)) = self.per_seq.iter().min_by_key(|(_, v)| v.tick)
        {
            self.per_seq.remove(&oldest_key);
        }
        let cfg = self.global.cfg;
        let prior = self.population_prior;
        let mode = self.prior_mode;
        let entry = self.per_seq.entry(session_hash).or_insert_with(|| {
            let controller = match mode {
                PopulationPriorMode::Enabled => {
                    DepthController::from_population_prior(cfg, &prior, NEW_REQUEST_PRIOR_WEIGHT)
                }
                PopulationPriorMode::Disabled => DepthController::new(cfg),
            };
            SeqControllerEntry {
                controller,
                last_depth: usize::MAX,
                tick: now,
            }
        });
        entry.tick = now;
        entry
    }
}

static STATE: OnceLock<Mutex<PerSeqDepthState>> = OnceLock::new();

fn state() -> &'static Mutex<PerSeqDepthState> {
    STATE.get_or_init(|| {
        Mutex::new(PerSeqDepthState::new_with_mode(
            DepthConfig::runtime(),
            PopulationPriorMode::from_env(),
        ))
    })
}

thread_local! {
    /// Active sequence `session_hash` while settling a verify frame or proposing.
    static ACTIVE_SESSION: Cell<Option<u64>> = const { Cell::new(None) };
}

/// Scoped RAII guard routing [`record_frame`] and [`proposal_suppressed`] to a
/// specific sequence's [`DepthController`].
pub(super) struct ActiveDepthSessionGuard {
    prev: Option<u64>,
}

impl ActiveDepthSessionGuard {
    #[must_use]
    pub(super) fn new(session_hash: u64) -> Self {
        let prev = ACTIVE_SESSION.with(|c| c.replace(Some(session_hash)));
        Self { prev }
    }
}

impl Drop for ActiveDepthSessionGuard {
    fn drop(&mut self) {
        ACTIVE_SESSION.with(|c| c.set(self.prev));
    }
}

pub(super) fn adaptive_depth_enabled_for_policy(
    env_override: Option<&str>,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
) -> bool {
    if env_override.map(str::trim) == Some("0") {
        return false;
    }
    if env_override.map(str::trim) == Some("1") {
        return true;
    }
    matches!(
        policy,
        super::stochastic_verify_walk::SpecVerifyPolicy::Stochastic
            | super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
            | super::stochastic_verify_walk::SpecVerifyPolicy::Rejection
    )
}

pub(super) fn adaptive_depth_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let raw = std::env::var("FLUX_SPEC_ADAPTIVE_DEPTH")
            .or_else(|_| std::env::var("ATLAS_SPEC_ADAPTIVE_DEPTH"))
            .ok();
        adaptive_depth_enabled_for_policy(
            raw.as_deref(),
            super::stochastic_verify_walk::spec_verify_policy(),
        )
    })
}

/// Mix physical `slot_idx` into `session_hash` so concurrent requests sharing
/// identical prompt messages maintain independent adaptive-depth controllers.
#[inline]
pub(super) fn seq_controller_key(session_hash: u64, slot_idx: usize) -> u64 {
    if slot_idx == usize::MAX {
        session_hash
    } else {
        session_hash ^ (slot_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }
}

/// Remove a retired sequence's adaptive-depth controller when its slot is freed.
pub(super) fn retire_seq(seq_key: u64) {
    if let Some(mutex) = STATE.get() {
        mutex.lock().per_seq.remove(&seq_key);
    }
}

/// Fold a settled verify frame into the active sequence's controller (or global fallback).
pub(super) fn record_frame(drafted: usize, accepted: usize) {
    if !adaptive_depth_enabled() || drafted == 0 {
        return;
    }
    let session = ACTIVE_SESSION.with(Cell::get);
    let mut guard = state().lock();
    guard.observe_population(drafted, accepted);
    if let Some(seq_key) = session {
        let entry = guard.touch_or_insert(seq_key);
        entry.controller = entry.controller.observe(drafted, accepted);
    }
}

/// The global controller's last decision; `usize::MAX` until it is first consulted.
static LAST_DEPTH: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Live sequence count for the current scheduler step (`1` by default).
/// Single-sequence adaptive depth is strictly scoped to solo `live <= 1`
/// ticks so `C > 1` servers keep full `C=1` parity without requiring
/// `ATLAS_SPEC_ADAPTIVE_DEPTH=0`, while `live >= 2` cohorts remain cohort-safe.
static CURRENT_STEP_LIVE_SEQS: AtomicUsize = AtomicUsize::new(1);

/// Record the active concurrency for the current scheduler step.
#[inline]
pub(super) fn set_step_live_seqs(live: usize) {
    CURRENT_STEP_LIVE_SEQS.store(live.max(1), Ordering::Relaxed);
}

/// Pure concurrency gate: single-sequence adaptive depth applies only for solo
/// `live <= 1` steps.
#[inline]
#[must_use]
pub(super) const fn adaptive_depth_allowed_for_live(enabled: bool, live: usize) -> bool {
    enabled && live <= 1
}

/// Whether single-sequence adaptive depth is active on the current scheduler step.
#[inline]
#[must_use]
pub(super) fn adaptive_depth_active_for_current_step() -> bool {
    adaptive_depth_allowed_for_live(
        adaptive_depth_enabled(),
        CURRENT_STEP_LIVE_SEQS.load(Ordering::Relaxed),
    )
}

/// Step-level verify width cap (`None` when off or `live >= 2`). When per-sequence
/// controllers are active at solo `live <= 1`, returns the maximum across active
/// per-sequence controllers so a low-acceptance stream never throttles a
/// deterministic stream.
pub(super) fn stochastic_depth(configured: usize) -> Option<usize> {
    if !adaptive_depth_active_for_current_step() || configured == 0 {
        return None;
    }
    let mut guard = state().lock();
    if let Some(max_seq_depth) = guard
        .per_seq
        .values()
        .map(|entry| entry.controller.best_depth_capped(configured))
        .max()
    {
        return Some(max_seq_depth.clamp(1, configured));
    }
    let (depth, next) = guard.global.next_frame_capped(configured);
    guard.global = next;
    LAST_DEPTH.store(depth, Ordering::Relaxed);
    Some(depth.clamp(1, configured))
}

/// Per-sequence draft depth (`0..=configured`) advancing only `seq_key`'s controller.
pub(super) fn seq_next_draft_depth(seq_key: u64, configured: usize) -> Option<usize> {
    if !adaptive_depth_active_for_current_step() || configured == 0 {
        return None;
    }
    let mut guard = state().lock();
    let entry = guard.touch_or_insert(seq_key);
    let (depth, next) = entry.controller.next_frame_capped(configured);
    entry.controller = next;
    entry.last_depth = depth;
    Some(depth.min(configured))
}

/// Whether `seq_key`'s latest adaptive decision suppressed neural drafting (`depth == 0`).
pub(super) fn seq_proposal_suppressed(seq_key: u64) -> bool {
    if !adaptive_depth_active_for_current_step() {
        return false;
    }
    let guard = state().lock();
    guard.per_seq.get(&seq_key).map_or_else(
        || LAST_DEPTH.load(Ordering::Relaxed) == 0,
        |e| e.last_depth == 0,
    )
}

/// Whether the active sequence (or global controller) decided depth 0.
pub(super) fn proposal_suppressed() -> bool {
    if !adaptive_depth_active_for_current_step() {
        return false;
    }
    if let Some(seq_key) = ACTIVE_SESSION.with(Cell::get) {
        return seq_proposal_suppressed(seq_key);
    }
    LAST_DEPTH.load(Ordering::Relaxed) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctl(rate: f32) -> DepthController {
        DepthController::new(DepthConfig {
            initial_rate: rate,
            ..DepthConfig::default()
        })
    }

    #[test]
    fn high_acceptance_drafts_the_maximum() {
        assert_eq!(ctl(0.95).best_depth(), 5);
        let ctl8 = DepthController::new(DepthConfig {
            max_depth: 8,
            initial_rate: 0.95,
            ..DepthConfig::default()
        });
        assert_eq!(ctl8.best_depth(), 8);
    }

    #[test]
    fn low_acceptance_drafts_nothing() {
        assert_eq!(ctl(0.1).best_depth(), 0);
        assert_eq!(ctl(0.0).best_depth(), 0);
    }

    #[test]
    fn depth_is_monotone_in_acceptance() {
        let depths: Vec<usize> = (0..=20)
            .map(|i| ctl(i as f32 / 20.0).best_depth())
            .collect();
        assert!(depths.windows(2).all(|w| w[0] <= w[1]), "{depths:?}");
        assert_eq!(depths.first(), Some(&0));
        assert_eq!(depths.last(), Some(&5));
    }

    #[test]
    fn expected_tokens_matches_geometric_sum() {
        assert_eq!(expected_tokens(0.5, 0), 1.0);
        assert!((expected_tokens(0.5, 3) - 1.875).abs() < 1e-6);
        assert_eq!(expected_tokens(1.0, 5), 6.0);
    }

    #[test]
    fn empirical_survival_captures_non_geometric_cliff_up_to_depth_8() {
        // Steps 0..3 accept at 0.95, then step 3 collapses to 0.02 (e.g. MTP horizon cliff).
        // A scalar geometric mean (~0.75) would over-draft past step 3, whereas empirical
        // Rate(d) stops cleanly at depth 3!
        let mut c = DepthController::new(DepthConfig {
            max_depth: 8,
            alpha: 0.25,
            draft_cost: 0.15,
            initial_rate: 0.7,
            probe_interval: 16,
        });
        for _ in 0..40 {
            c = c.observe(8, 3);
        }
        assert_eq!(c.best_depth(), 3, "step_rates={:?}", c.step_rates());
        assert!(c.step_rates()[2] > 0.95 && c.step_rates()[3] < 0.05);

        // When all 8 drafts consistently accept, depth climbs all the way to 8!
        for _ in 0..60 {
            c = c.observe(8, 8);
        }
        assert_eq!(c.best_depth(), 8);
    }

    #[test]
    fn per_sequence_controllers_isolate_concurrent_streams() {
        let mut table = PerSeqDepthState::new(DepthConfig {
            max_depth: 8,
            alpha: 0.3,
            ..DepthConfig::default()
        });
        // Stream A: deterministic code (8/8 accepted).
        // Stream B: high-entropy creative prose (0/8 accepted).
        for _ in 0..30 {
            let a = table.touch_or_insert(0xAAAA);
            a.controller = a.controller.observe(8, 8);
            let b = table.touch_or_insert(0xBBBB);
            b.controller = b.controller.observe(8, 0);
        }
        let depth_a = table.touch_or_insert(0xAAAA).controller.best_depth();
        let depth_b = table.touch_or_insert(0xBBBB).controller.best_depth();
        assert_eq!(depth_a, 8, "code stream must stay at full depth 8");
        assert_eq!(depth_b, 0, "noisy stream must drop to depth 0");
    }

    #[test]
    fn slot_key_isolates_identical_prompt_sessions() {
        let mut table = PerSeqDepthState::new(DepthConfig {
            max_depth: 5,
            alpha: 0.3,
            ..DepthConfig::default()
        });
        let shared_session = 0x7d07_9931_c8a6_8c09u64;
        let k0 = seq_controller_key(shared_session, 0);
        let k1 = seq_controller_key(shared_session, 1);
        assert_ne!(k0, k1);
        for _ in 0..25 {
            let e0 = table.touch_or_insert(k0);
            e0.controller = e0.controller.observe(4, 4);
            let e1 = table.touch_or_insert(k1);
            e1.controller = e1.controller.observe(4, 0);
        }
        assert_eq!(table.touch_or_insert(k0).controller.best_depth(), 5);
        assert_eq!(table.touch_or_insert(k1).controller.best_depth(), 0);
    }

    #[test]
    fn observe_counts_accepts_and_the_one_reject() {
        let cfg = DepthConfig {
            alpha: 0.5,
            initial_rate: 0.5,
            ..DepthConfig::default()
        };
        let c = DepthController::new(cfg);
        // 2 accepts then a reject: 0.5 -> 0.75 -> 0.875 -> 0.4375.
        assert!((c.observe(5, 2).rate() - 0.4375).abs() < 1e-6);
        // All 3 accepted: no reject trial.
        assert!((c.observe(3, 3).rate() - 0.9375).abs() < 1e-6);
        // Nothing drafted: unchanged.
        assert_eq!(c.observe(0, 0).rate(), 0.5);
        // accepted > drafted is clamped, never panics.
        assert!((c.observe(1, 4).rate() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn observe_does_not_mutate_the_original() {
        let c = ctl(0.5);
        let _ = c.observe(5, 0);
        assert_eq!(c.rate(), 0.5);
    }

    #[test]
    fn sustained_rejection_drives_depth_to_zero_and_recovery_back_up() {
        let down = (0..200).fold(ctl(0.9), |c, _| c.observe(5, 0));
        assert_eq!(down.best_depth(), 0);
        let up = (0..200).fold(down, |c, _| c.observe(5, 5));
        assert_eq!(up.best_depth(), 5);
    }

    #[test]
    fn depth_zero_probes_once_per_interval() {
        let cfg = DepthConfig {
            initial_rate: 0.0,
            probe_interval: 4,
            ..DepthConfig::default()
        };
        let (depths, _) = (0..12).fold((Vec::new(), DepthController::new(cfg)), |(mut d, c), _| {
            let (depth, next) = c.next_frame();
            d.push(depth);
            (d, next)
        });
        assert_eq!(depths, vec![0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
    }

    #[test]
    fn probe_interval_zero_never_probes() {
        let cfg = DepthConfig {
            initial_rate: 0.0,
            probe_interval: 0,
            ..DepthConfig::default()
        };
        let c = DepthController::new(cfg);
        assert!(
            (0..50)
                .try_fold(c, |c, _| {
                    let (depth, next) = c.next_frame();
                    (depth == 0).then_some(next)
                })
                .is_some()
        );
    }

    #[test]
    fn runtime_prior_starts_at_full_k5_without_cold_start_throttle() {
        let cfg = DepthConfig {
            max_depth: MAX_ADAPTIVE_DEPTH,
            draft_cost: 0.12,
            initial_rate: 0.78,
            ..DepthConfig::default()
        };
        let c = DepthController::new(cfg);
        assert_eq!(c.best_depth_capped(4), 4);
    }

    #[test]
    fn population_survival_prior_accelerates_warm_start_and_never_starves_after_noise() {
        let cfg = DepthConfig {
            max_depth: MAX_ADAPTIVE_DEPTH,
            alpha: 0.25,
            draft_cost: 0.12,
            initial_rate: 0.78,
            ..DepthConfig::default()
        };
        let mut warm = PerSeqDepthState::new(cfg);
        for _ in 0..64 {
            warm.observe_population(8, 8);
        }
        assert!(warm.touch_or_insert(0x1111).controller.best_depth() >= 6);

        let mut noisy = PerSeqDepthState::new(cfg);
        for _ in 0..200 {
            noisy.observe_population(8, 0);
        }
        assert!(
            noisy
                .touch_or_insert(0x2222)
                .controller
                .best_depth_capped(4)
                >= 1,
            "bounded population prior must never starve a fresh sequence at depth 0"
        );
    }

    #[test]
    fn population_prior_weight_cap_invariant_and_workload_shift_adaptation() {
        let mut prior = PopulationAcceptancePrior::new();
        for _ in 0..10_000 {
            prior.observe(8, 8);
            for slot in 0..MAX_ADAPTIVE_DEPTH {
                let a = prior.accepted_w[slot];
                let t = prior.trials_w[slot];
                assert!(
                    (0.0..=t).contains(&a) && t <= POPULATION_WEIGHT_CAP,
                    "slot {slot}: invariant violated a={a} t={t}"
                );
            }
        }
        for slot in 0..MAX_ADAPTIVE_DEPTH {
            assert_eq!(prior.slot_weight(slot), prior.trials_w[slot]);
            assert!(
                prior.slot_weight(slot) > 100.0 && prior.slot_weight(slot) <= POPULATION_WEIGHT_CAP,
                "slot {slot}: unexpected settled weight {}",
                prior.slot_weight(slot)
            );
        }
        assert_eq!(prior.slot_weight(MAX_ADAPTIVE_DEPTH), 0.0);
        let high_curve = prior.survival_curve(0.5);
        assert!(
            high_curve.iter().all(|&p| p > 0.95),
            "high_curve={high_curve:?}"
        );

        // Shift workload abruptly to 0% acceptance and verify bounded EWMA adapts.
        for _ in 0..512 {
            prior.observe(8, 0);
            for slot in 0..MAX_ADAPTIVE_DEPTH {
                let a = prior.accepted_w[slot];
                let t = prior.trials_w[slot];
                assert!(
                    (0.0..=t).contains(&a) && t <= POPULATION_WEIGHT_CAP,
                    "post-shift slot {slot}: invariant violated a={a} t={t}"
                );
            }
        }
        assert!(
            prior.survival_curve(0.78)[0] < 0.05,
            "slot 0 must adapt after workload shift: {}",
            prior.survival_curve(0.78)[0]
        );

        // Shift back to 100% acceptance and verify recovery.
        for _ in 0..512 {
            prior.observe(8, 8);
        }
        assert!(
            prior.survival_curve(0.78)[0] > 0.95,
            "slot 0 must recover after shift back: {}",
            prior.survival_curve(0.78)[0]
        );
    }

    #[test]
    fn warm_population_prior_drafts_at_optimal_depth_on_frame_zero() {
        let cfg = DepthConfig {
            max_depth: MAX_ADAPTIVE_DEPTH,
            alpha: 0.05,
            draft_cost: 0.12,
            probe_interval: 16,
            initial_rate: 0.78,
        };
        let mut cliff_state = PerSeqDepthState::new_with_mode(cfg, PopulationPriorMode::Enabled);
        for _ in 0..64 {
            cliff_state.observe_population(8, 3);
        }
        let fresh_cliff = cliff_state.touch_or_insert(0xC1FF).controller;
        assert_eq!(
            fresh_cliff.best_depth(),
            3,
            "warm request must draft at population cliff depth 3 on frame 0; rates={:?}",
            fresh_cliff.step_rates()
        );
        assert_eq!(
            fresh_cliff.step_weights(),
            &[NEW_REQUEST_PRIOR_WEIGHT; MAX_ADAPTIVE_DEPTH]
        );

        let cold_init_cfg = DepthConfig {
            initial_rate: 0.25,
            ..cfg
        };
        let mut full_state =
            PerSeqDepthState::new_with_mode(cold_init_cfg, PopulationPriorMode::Enabled);
        for _ in 0..64 {
            full_state.observe_population(8, 8);
        }
        let fresh_full = full_state.touch_or_insert(0xF00D).controller;
        assert_eq!(
            fresh_full.best_depth(),
            MAX_ADAPTIVE_DEPTH,
            "warm 8/8 population must start new request at max depth on frame 0; rates={:?}",
            fresh_full.step_rates()
        );
    }

    #[test]
    fn mismatched_or_cold_request_converges_in_two_to_five_rounds() {
        let cfg = DepthConfig {
            max_depth: MAX_ADAPTIVE_DEPTH,
            alpha: 0.05,
            draft_cost: 0.12,
            probe_interval: 16,
            initial_rate: 0.78,
        };

        // 1) Warm 8/8 population, mismatched request with true horizon cliff at depth 2.
        let mut high_pop = PopulationAcceptancePrior::new();
        for _ in 0..200 {
            high_pop.observe(8, 8);
        }
        let mut c =
            DepthController::from_population_prior(cfg, &high_pop, NEW_REQUEST_PRIOR_WEIGHT);
        assert_eq!(c.best_depth(), MAX_ADAPTIVE_DEPTH);
        let mut cliff_rounds = 0usize;
        while c.best_depth() != 2 && cliff_rounds < 32 {
            let d = c.best_depth();
            c = c.observe(d, d.min(2));
            cliff_rounds += 1;
        }
        assert!(
            (2..=5).contains(&cliff_rounds),
            "prior-weighted controller must adapt to depth-2 cliff in 2..=5 rounds, took {cliff_rounds}"
        );

        // 2) Noisy 0/8 population, truly 0%-accept request drops to depth 0 in 2..=5 rounds.
        let mut noisy_pop = PopulationAcceptancePrior::new();
        for _ in 0..200 {
            noisy_pop.observe(8, 0);
        }
        let mut zero_req =
            DepthController::from_population_prior(cfg, &noisy_pop, NEW_REQUEST_PRIOR_WEIGHT);
        assert!(zero_req.step_rates()[0] >= NEW_REQUEST_P0_FLOOR);
        assert!(zero_req.best_depth() >= 1);
        let mut drop_rounds = 0usize;
        while zero_req.best_depth() > 0 && drop_rounds < 32 {
            let d = zero_req.best_depth();
            zero_req = zero_req.observe(d, 0);
            drop_rounds += 1;
        }
        assert!(
            (2..=5).contains(&drop_rounds),
            "0%-accept request after noisy prior must reach depth 0 in 2..=5 rounds, took {drop_rounds}"
        );

        // 3) Noisy 0/8 population, 100%-accept request climbs to full depth 8 in 2..=5 rounds.
        let mut hot_req =
            DepthController::from_population_prior(cfg, &noisy_pop, NEW_REQUEST_PRIOR_WEIGHT);
        let mut climb_rounds = 0usize;
        while hot_req.best_depth() < MAX_ADAPTIVE_DEPTH && climb_rounds < 32 {
            let d = hot_req.best_depth();
            hot_req = hot_req.observe(d, d);
            climb_rounds += 1;
        }
        assert!(
            (2..=5).contains(&climb_rounds),
            "100%-accept request after noisy prior must reach depth 8 in 2..=5 rounds, took {climb_rounds}"
        );
    }

    #[test]
    fn disabled_population_prior_mode_produces_bit_identical_isolated_trajectories() {
        assert_eq!(
            PopulationPriorMode::parse(Some("0"), None),
            PopulationPriorMode::Disabled
        );
        assert_eq!(
            PopulationPriorMode::parse(Some("1"), None),
            PopulationPriorMode::Enabled
        );
        assert_eq!(
            PopulationPriorMode::parse(None, None),
            PopulationPriorMode::Enabled
        );
        assert_eq!(
            PopulationPriorMode::parse(Some("1"), Some("1")),
            PopulationPriorMode::Disabled
        );
        assert_eq!(
            PopulationPriorMode::parse(None, Some("1")),
            PopulationPriorMode::Disabled
        );
        assert_eq!(
            PopulationPriorMode::parse(None, Some("0")),
            PopulationPriorMode::Enabled
        );

        let cfg = DepthConfig {
            max_depth: MAX_ADAPTIVE_DEPTH,
            alpha: 0.05,
            draft_cost: 0.12,
            probe_interval: 16,
            initial_rate: 0.78,
        };
        let mut polluted = PerSeqDepthState::new_with_mode(cfg, PopulationPriorMode::Disabled);
        for _ in 0..200 {
            polluted.observe_population(8, 0);
        }
        let mut clean = PerSeqDepthState::new_with_mode(cfg, PopulationPriorMode::Disabled);
        let mut reference = DepthController::new(cfg);

        let frames = [
            (4, 2),
            (4, 4),
            (6, 0),
            (5, 3),
            (8, 8),
            (8, 1),
            (4, 0),
            (4, 0),
            (4, 4),
            (8, 5),
        ];
        for (step, &(drafted, accepted)) in frames.iter().enumerate() {
            let p_entry = polluted.touch_or_insert(0xCAFE);
            let c_entry = clean.touch_or_insert(0xCAFE);
            let (p_d, p_next) = p_entry.controller.next_frame();
            let (c_d, c_next) = c_entry.controller.next_frame();
            let (r_d, r_next) = reference.next_frame();
            assert_eq!(p_d, r_d, "step {step}: polluted depth diverged");
            assert_eq!(c_d, r_d, "step {step}: clean depth diverged");

            p_entry.controller = p_next.observe(drafted, accepted);
            c_entry.controller = c_next.observe(drafted, accepted);
            reference = r_next.observe(drafted, accepted);

            assert_eq!(
                p_entry.controller.rate().to_bits(),
                reference.rate().to_bits()
            );
            assert_eq!(
                c_entry.controller.rate().to_bits(),
                reference.rate().to_bits()
            );
            for slot in 0..MAX_ADAPTIVE_DEPTH {
                assert_eq!(
                    p_entry.controller.step_rates()[slot].to_bits(),
                    reference.step_rates()[slot].to_bits(),
                    "step {step} slot {slot}: step_rate diverged"
                );
                assert_eq!(
                    c_entry.controller.step_rates()[slot].to_bits(),
                    reference.step_rates()[slot].to_bits(),
                    "step {step} slot {slot}: clean step_rate diverged"
                );
                assert_eq!(
                    p_entry.controller.step_weights()[slot].to_bits(),
                    reference.step_weights()[slot].to_bits(),
                    "step {step} slot {slot}: step_weight diverged"
                );
            }
        }
    }

    #[test]
    fn single_seq_adaptive_depth_is_strictly_scoped_to_solo_live_one() {
        assert!(adaptive_depth_allowed_for_live(true, 0));
        assert!(adaptive_depth_allowed_for_live(true, 1));
        for live in [2usize, 3, 4, 8, 16, 32, 64, 128] {
            assert!(
                !adaptive_depth_allowed_for_live(true, live),
                "live={live} must disable single-sequence adaptive depth"
            );
        }
        assert!(!adaptive_depth_allowed_for_live(false, 0));
        assert!(!adaptive_depth_allowed_for_live(false, 1));
        assert!(!adaptive_depth_allowed_for_live(false, 8));
    }

    #[test]
    fn adaptive_depth_enables_for_rejection_policy_and_honors_env_overrides() {
        use super::super::stochastic_verify_walk::SpecVerifyPolicy;

        assert!(!adaptive_depth_enabled_for_policy(
            None,
            SpecVerifyPolicy::Argmax
        ));
        assert!(adaptive_depth_enabled_for_policy(
            None,
            SpecVerifyPolicy::Stochastic
        ));
        assert!(adaptive_depth_enabled_for_policy(
            None,
            SpecVerifyPolicy::Gumbel
        ));
        assert!(adaptive_depth_enabled_for_policy(
            None,
            SpecVerifyPolicy::Rejection
        ));
        assert!(!adaptive_depth_enabled_for_policy(
            Some("0"),
            SpecVerifyPolicy::Rejection
        ));
        assert!(adaptive_depth_enabled_for_policy(
            Some("1"),
            SpecVerifyPolicy::Argmax
        ));
    }
}
