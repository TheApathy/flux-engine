// SPDX-License-Identifier: AGPL-3.0-only

//! Draft-width ladder, cohort acceptance EWMA controller, adaptive width policy, and chain narrowing.

use std::time::Instant;

use spark_model::traits::Model;

use super::planning::batched_ragged_enabled;
use crate::scheduler::ActiveSeq;

/// Opt-in per-concurrency draft-width ladder, e.g. `2:4,4:3,8:2,32:1`: up
/// to 2 speculating sequences propose 4 drafts (K5), up to 4 propose 3, up
/// to 8 propose 2, beyond that 1. Unset keeps the configured width at every
/// concurrency. Draft width changes only acceptance and cost, never output.
pub(in super::super) const LADDER_ENV: &str = "ATLAS_QWEN4_BATCHED_K5_LADDER";

/// Parse `width:drafts` rungs (strictly ascending widths, drafts >= 1).
pub(in super::super) fn parse_ladder(
    raw: &str,
) -> std::result::Result<Vec<(usize, usize)>, String> {
    let mut rungs: Vec<(usize, usize)> = Vec::new();
    for rung in raw.split(',') {
        let (width, drafts) = rung
            .split_once(':')
            .ok_or_else(|| format!("{LADDER_ENV} rung {rung:?} is not width:drafts"))?;
        let width: usize = width
            .parse()
            .map_err(|_| format!("{LADDER_ENV} width {width:?} is not an integer"))?;
        let drafts: usize = drafts
            .parse()
            .map_err(|_| format!("{LADDER_ENV} drafts {drafts:?} is not an integer"))?;
        if width == 0 || drafts == 0 {
            return Err(format!("{LADDER_ENV} rung {rung:?} must be positive"));
        }
        if rungs.last().is_some_and(|&(last, _)| width <= last) {
            return Err(format!("{LADDER_ENV} widths must strictly ascend"));
        }
        rungs.push((width, drafts));
    }
    if rungs.is_empty() {
        return Err(format!("{LADDER_ENV} is empty"));
    }
    Ok(rungs)
}

/// Drafts per sequence at `concurrency`, never above `configured`.
pub(in super::super) fn ladder_drafts(
    ladder: &[(usize, usize)],
    concurrency: usize,
    configured: usize,
) -> usize {
    let drafts = ladder
        .iter()
        .find(|&&(width, _)| concurrency <= width)
        .or(ladder.last())
        .map_or(configured, |&(_, drafts)| drafts);
    drafts.min(configured)
}

fn ladder() -> Option<&'static [(usize, usize)]> {
    static LADDER: std::sync::OnceLock<Option<Vec<(usize, usize)>>> = std::sync::OnceLock::new();
    LADDER
        .get_or_init(|| {
            let raw = std::env::var(LADDER_ENV).ok()?;
            match parse_ladder(&raw) {
                Ok(rungs) => {
                    tracing::info!(ladder = raw, "QWEN4_BATCHED_K5_LADDER_ENGAGED");
                    Some(rungs)
                }
                Err(error) => {
                    tracing::warn!("{error}; keeping the configured draft width");
                    None
                }
            }
        })
        .as_deref()
}

struct CohortAcceptState {
    acc_ewma: f32,
    trials_ewma: f32,
    held_k: usize,
    dwell_steps: u32,
    down_streak: u32,
    up_streak: u32,
    last_live: usize,
    observed_frames: u64,
    evaluated_frames: u64,
    last_activity: Option<Instant>,
}

impl CohortAcceptState {
    const INITIAL_P_STEP: f32 = 0.80;
    const INITIAL_TRIALS_PER_SEQ: f32 = 2.0;
    const INITIAL_ACC_PER_SEQ: f32 = Self::INITIAL_P_STEP * Self::INITIAL_TRIALS_PER_SEQ;
    const EWMA_ALPHA: f32 = 0.15;
    /// Minimum settled batched frames held at the current width `K` before any
    /// upward or downward transition may fire (stabilizes batched CUDA graphs).
    const MIN_DWELL_STEPS: u32 = 6;
    /// Consecutive frames favoring a narrower `K` (by at least
    /// `HYSTERESIS_RATE_MARGIN_DOWN`) required before stepping down.
    const CONFIRM_STREAK_DOWN: u32 = 4;
    /// Consecutive frames favoring a wider `K` (by at least
    /// `HYSTERESIS_RATE_MARGIN_UP`) required before stepping up.
    const CONFIRM_STREAK_UP: u32 = 3;
    /// Asymmetric hysteresis margins (`+4%` down / `+2%` up = `6%` deadband):
    /// stepping down pays CUDA graph warmup/capture and risks under-drafting
    /// on transient dips across `K = 2..=8`, while stepping up recovers
    /// promptly when acceptance rises.
    const HYSTERESIS_RATE_MARGIN_DOWN: f64 = 0.04;
    const HYSTERESIS_RATE_MARGIN_UP: f64 = 0.02;
    /// Idle gap between concurrent cohort steps after which a new burst
    /// warm-starts at `INITIAL_P_STEP` and full cohort `max_k`.
    const IDLE_WARM_START_GAP: std::time::Duration = std::time::Duration::from_secs(2);

    const fn new() -> Self {
        Self {
            acc_ewma: Self::INITIAL_ACC_PER_SEQ,
            trials_ewma: Self::INITIAL_TRIALS_PER_SEQ,
            held_k: spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K,
            dwell_steps: 0,
            down_streak: 0,
            up_streak: 0,
            last_live: 0,
            observed_frames: 0,
            evaluated_frames: 0,
            last_activity: None,
        }
    }

    #[inline]
    fn ewma_p_step(&self) -> f32 {
        if !self.trials_ewma.is_finite() || self.trials_ewma <= 0.0 || !self.acc_ewma.is_finite() {
            return Self::INITIAL_P_STEP;
        }
        (self.acc_ewma / self.trials_ewma).clamp(0.0, 1.0)
    }

    fn warm_start(&mut self, live: usize, max_k: usize) {
        self.acc_ewma = Self::INITIAL_ACC_PER_SEQ;
        self.trials_ewma = Self::INITIAL_TRIALS_PER_SEQ;
        self.held_k = max_k;
        self.dwell_steps = 0;
        self.down_streak = 0;
        self.up_streak = 0;
        self.last_live = live;
        self.evaluated_frames = self.observed_frames;
    }

    fn observe_frame_seqs(
        &mut self,
        drafted_seqs: usize,
        total_accepted: usize,
        total_drafted: usize,
        rejected_seqs: usize,
    ) {
        if drafted_seqs == 0 || total_drafted == 0 {
            return;
        }
        let trials = total_accepted.saturating_add(rejected_seqs);
        if trials == 0 {
            return;
        }
        let seqs_f = drafted_seqs as f32;
        let acc_obs = total_accepted as f32 / seqs_f;
        let trials_obs = trials as f32 / seqs_f;
        let prev_acc = if self.acc_ewma.is_finite() && self.acc_ewma >= 0.0 {
            self.acc_ewma
        } else {
            Self::INITIAL_ACC_PER_SEQ
        };
        let prev_trials = if self.trials_ewma.is_finite() && self.trials_ewma > 0.0 {
            self.trials_ewma
        } else {
            Self::INITIAL_TRIALS_PER_SEQ
        };
        let keep = 1.0 - Self::EWMA_ALPHA;
        self.acc_ewma = (Self::EWMA_ALPHA * acc_obs + keep * prev_acc).max(0.0);
        self.trials_ewma = (Self::EWMA_ALPHA * trials_obs + keep * prev_trials).max(1e-6);
        self.observed_frames = self.observed_frames.wrapping_add(1);
    }

    fn select_k_inner(&mut self, live: usize, max_k: usize, idle_expired: bool) -> usize {
        let min_k = spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MIN_K;
        let upper = max_k.clamp(
            min_k,
            spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K,
        );
        if live <= 1 {
            self.last_live = live;
            return upper;
        }
        let is_new_burst = idle_expired
            || self.last_live <= 1
            || live >= self.last_live.saturating_mul(2)
            || live >= self.last_live.saturating_add(4);
        if is_new_burst {
            self.warm_start(live, upper);
            return self.held_k;
        }
        self.last_live = live;
        self.held_k = self.held_k.clamp(min_k, upper);
        if self.observed_frames != self.evaluated_frames {
            self.evaluated_frames = self.observed_frames;
            self.dwell_steps = self.dwell_steps.saturating_add(1);
            let p = self.ewma_p_step();
            let cand = spark_model::layers::qwen4_verify_lanes::optimal_cohort_k_for_max_rows(
                live,
                p,
                upper,
                spark_model::layers::qwen4_verify_lanes::adaptive_max_frame_rows(),
            );
            let held_rate =
                spark_model::layers::qwen4_verify_lanes::cohort_expected_rate(live, p, self.held_k);
            let cand_rate =
                spark_model::layers::qwen4_verify_lanes::cohort_expected_rate(live, p, cand);
            if cand < self.held_k
                && cand_rate > held_rate * (1.0 + Self::HYSTERESIS_RATE_MARGIN_DOWN)
            {
                self.down_streak = self.down_streak.saturating_add(1);
                self.up_streak = 0;
                if self.dwell_steps >= Self::MIN_DWELL_STEPS
                    && self.down_streak >= Self::CONFIRM_STREAK_DOWN
                {
                    self.held_k = cand;
                    self.dwell_steps = 0;
                    self.down_streak = 0;
                }
            } else if cand > self.held_k
                && cand_rate > held_rate * (1.0 + Self::HYSTERESIS_RATE_MARGIN_UP)
            {
                self.up_streak = self.up_streak.saturating_add(1);
                self.down_streak = 0;
                if self.dwell_steps >= Self::MIN_DWELL_STEPS
                    && self.up_streak >= Self::CONFIRM_STREAK_UP
                {
                    self.held_k = cand;
                    self.dwell_steps = 0;
                    self.up_streak = 0;
                }
            } else {
                self.down_streak = 0;
                self.up_streak = 0;
            }
        }
        self.held_k
    }
}

/// Online cohort acceptance estimator and hysteresis-stabilized verify-width
/// controller for batched native-MTP cohorts (`live >= 2`).
///
/// Tracks exponentially-weighted per-sequence accepted drafts and truncated-
/// geometric trials (`acc_ewma / trials_ewma`, `alpha = 0.15`) rather than an
/// EWMA of per-frame ratios, eliminating Jensen downward bias on small cohorts
/// (`c = 2..4`) while remaining K-invariant across `K = 2..=ADAPTIVE_MAX_K`.
/// Transitions require a minimum dwell (`>= 6` settled frames) plus directional
/// confirmation (`4` frames with `>= +4%` predicted throughput gain to step
/// down, `3` frames with `>= +2%` gain to step up) to prevent under-drafting
/// and keep batched CUDA graph captures minimal. New concurrent bursts
/// (`live <= 1 -> >= 2`, sharp ramp-ups, or `>= 2s` idle gaps) warm-start at
/// `p_step = 0.80` and full single-frame cohort `max_k` (e.g. `K = 7` with
/// `--num-drafts 6` for `c <= 16`, narrowed automatically by `128 / live` at
/// higher concurrency).
pub(in super::super) struct CohortAcceptEwma {
    state: parking_lot::Mutex<CohortAcceptState>,
}

impl CohortAcceptEwma {
    pub(in super::super) const fn new() -> Self {
        Self {
            state: parking_lot::Mutex::new(CohortAcceptState::new()),
        }
    }

    #[cfg(test)]
    #[inline]
    #[must_use]
    pub(in super::super) fn ewma_p_step(&self) -> f32 {
        self.state.lock().ewma_p_step()
    }

    /// Fold one settled batched verify frame into the cohort EWMA using the
    /// unbiased ratio-of-EWMAs truncated-geometric estimator
    /// `ewma(accepted / seqs) / ewma((accepted + rejected_seqs) / seqs)`.
    pub(in super::super) fn observe_frame_seqs(
        &self,
        drafted_seqs: usize,
        total_accepted: usize,
        total_drafted: usize,
        rejected_seqs: usize,
    ) {
        self.state.lock().observe_frame_seqs(
            drafted_seqs,
            total_accepted,
            total_drafted,
            rejected_seqs,
        );
    }

    /// Record a solo/idle step (`live <= 1`) so the next concurrent burst
    /// warm-starts cleanly at `INITIAL_P_STEP` and full cohort `max_k`.
    pub(in super::super) fn mark_solo(&self, live: usize) {
        let mut guard = self.state.lock();
        guard.last_live = live;
        guard.last_activity = None;
    }

    /// Select the hysteresis-stabilized verify width `K` for `live` concurrent
    /// sequences (bounded by `max_k`), warm-starting on new concurrent bursts
    /// or after an idle gap (`>= 2s`).
    pub(in super::super) fn select_k(&self, live: usize, max_k: usize) -> usize {
        let now = Instant::now();
        let mut guard = self.state.lock();
        let idle_expired = live >= 2
            && guard.last_activity.is_some_and(|prev| {
                now.saturating_duration_since(prev) >= CohortAcceptState::IDLE_WARM_START_GAP
            });
        if live >= 2 {
            guard.last_activity = Some(now);
        } else {
            guard.last_activity = None;
        }
        guard.select_k_inner(live, max_k, idle_expired)
    }

    /// Deterministic version of [`Self::select_k`] without wall-clock checks,
    /// used by unit tests to verify dwell, hysteresis, and warm-start transitions.
    #[cfg(test)]
    pub(in super::super) fn select_k_deterministic(
        &self,
        live: usize,
        max_k: usize,
        idle_expired: bool,
    ) -> usize {
        self.state.lock().select_k_inner(live, max_k, idle_expired)
    }
}

pub(super) static COHORT_ACCEPT_EWMA: CohortAcceptEwma = CohortAcceptEwma::new();

pub(super) fn cohort_adaptive_k_enabled() -> bool {
    let read_live = || match spark_model::layers::qwen4_verify_lanes::cohort_adaptive_k_selected() {
        Ok(on) => on,
        Err(error) => {
            tracing::warn!("{error:#}; keeping cohort adaptive K enabled");
            true
        }
    };
    #[cfg(not(test))]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// Fixed batched verify draft count (`K = 5`) when `ATLAS_QWEN4_BATCHED_K5=1`
/// runs without `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1` (`verify_batch_owner` admits
/// `k == 5` only in non-adaptive mode).
const FIXED_BATCHED_K5_DRAFTS: usize = 4;

/// Resolve the cohort width ceiling (`drafts` per sequence) for `live` active
/// speculating sequences before single-sequence stochastic depth is considered,
/// using the stateless GB10 roofline model directly (for unit-test coverage of
/// overrides, ladders, and caps).
#[cfg(test)]
pub(in super::super) fn resolve_cohort_width_capped(
    batched_k5_armed: bool,
    adaptive_k: bool,
    cohort_adaptive_k: bool,
    explicit_ladder: Option<&[(usize, usize)]>,
    live: usize,
    configured: usize,
    ewma_p_step: f32,
) -> usize {
    if !batched_k5_armed || live <= 1 || configured == 0 {
        return configured;
    }
    let adaptive_max_k = spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K;
    let batched_configured = configured.min(adaptive_max_k.saturating_sub(1));
    if let Some(rungs) = explicit_ladder {
        return ladder_drafts(rungs, live, batched_configured);
    }
    if !adaptive_k {
        return batched_configured.min(FIXED_BATCHED_K5_DRAFTS);
    }
    let profile_drafts = adaptive_drafts(live, batched_configured);
    if profile_drafts == 0 || !cohort_adaptive_k {
        return profile_drafts;
    }
    let max_k = (profile_drafts + 1).min(adaptive_max_k);
    let optimal_k = spark_model::layers::qwen4_verify_lanes::optimal_cohort_k_for_max_rows(
        live,
        ewma_p_step,
        max_k,
        spark_model::layers::qwen4_verify_lanes::adaptive_max_frame_rows(),
    );
    optimal_k.saturating_sub(1).clamp(1, profile_drafts)
}

/// Resolve the cohort width ceiling (`drafts` per sequence) for `live` active
/// speculating sequences using `controller`'s hysteresis-stabilized GB10
/// roofline width selection and burst warm-start.
///
/// - Solo `live <= 1` keeps the full `configured` draft width (e.g. `6` -> `K=7`)
///   and marks the cohort idle so the next concurrent burst warm-starts cleanly.
/// - Batched `live >= 2` is bounded by `<= ADAPTIVE_MAX_K - 1` (up to `K=8`
///   under adaptive-K, or `K=5` when adaptive-K is off) and the single-frame
///   `128 / live` row cap (`live = 2..=16` allows up to `K=7` with
///   `--num-drafts 6`, `16 * 7 = 112 <= 128`).
/// - Explicit `ATLAS_QWEN4_BATCHED_K5_LADDER` overrides take precedence.
/// - Otherwise, when `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1` and
///   `ATLAS_QWEN4_COHORT_ADAPTIVE_K` (default ON) are both active, selects `K`
///   via `controller.select_k(live, max_k)`.
pub(in super::super) fn resolve_cohort_width_with_controller(
    batched_k5_armed: bool,
    adaptive_k: bool,
    cohort_adaptive_k: bool,
    explicit_ladder: Option<&[(usize, usize)]>,
    live: usize,
    configured: usize,
    controller: &CohortAcceptEwma,
) -> usize {
    if !batched_k5_armed {
        return configured;
    }
    if live <= 1 {
        controller.mark_solo(live);
        return configured;
    }
    if configured == 0 {
        return 0;
    }
    let adaptive_max_k = spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K;
    let batched_configured = configured.min(adaptive_max_k.saturating_sub(1));
    if let Some(rungs) = explicit_ladder {
        return ladder_drafts(rungs, live, batched_configured);
    }
    if !adaptive_k {
        return batched_configured.min(FIXED_BATCHED_K5_DRAFTS);
    }
    let profile_drafts = adaptive_drafts(live, batched_configured);
    if profile_drafts == 0 || !cohort_adaptive_k {
        return profile_drafts;
    }
    let max_k = (profile_drafts + 1).min(adaptive_max_k);
    let selected_k = controller.select_k(live, max_k);
    selected_k.saturating_sub(1).clamp(1, profile_drafts)
}

pub(super) fn resolve_effective_depth(
    proposer_is_dflash: bool,
    batched_k5_armed: bool,
    _adaptive_k: bool,
    live: usize,
    width_capped: usize,
    stochastic_depth: Option<usize>,
) -> usize {
    let depth = if proposer_is_dflash {
        width_capped
    } else {
        stochastic_depth.unwrap_or(width_capped)
    };
    // Batched cohort mode (`live >= 2`) is governed uniformly by the cohort
    // roofline / ladder / fixed-K5 width (`width_capped`), never by
    // single-sequence `adaptive_depth::stochastic_depth`, while solo `live == 1`
    // retains full single-sequence adaptive-depth parity.
    if batched_k5_armed && live >= 2 {
        width_capped
    } else {
        depth
    }
}

/// True when the current scheduler step's active cohort includes at least one
/// sampled sequence (`temperature > 0.0`). Defaults to `true` so standalone
/// calls and unit tests retain ragged width behavior; greedy-only cohorts
/// (`temperature == 0.0`) route through zero-sync uniform `argmax_bf16_rows`
/// + `CohortAcceptEwma` widths.
static STEP_COHORT_SAMPLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

#[inline]
pub(in super::super) fn set_step_cohort_sampled(any_sampled: bool) {
    STEP_COHORT_SAMPLED.store(any_sampled, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn step_cohort_sampled() -> bool {
    STEP_COHORT_SAMPLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Draft width for this scheduler step: the online-EWMA roofline adaptive width
/// (or explicit ladder rung) for `live >= 2` when batched native MTP is armed,
/// and full single-sequence configured + adaptive depth for solo `live <= 1`.
pub(in super::super) fn effective_num_drafts(
    model: &dyn Model,
    live: usize,
    configured: usize,
) -> usize {
    crate::scheduler::adaptive_depth::set_step_live_seqs(live);
    // FLUX_SPEC_VERIFY=stochastic adaptive depth applies on every route,
    // including C=1 (not armed): C=1 dispatch picks K2/K3/K4/generic from
    // the actual draft count, so a shorter chain is always verifiable.
    let batched_armed = model.qwen4_batched_k5_armed();
    let adaptive_k = adaptive_selected();
    let proposer_is_dflash = model.proposer_is_dflash();
    // When ragged per-sequence `p_cum` compaction is active (TP=1, sampled
    // cohorts), let the cohort draft ceiling stay at `profile_drafts` (up to
    // `max_k - 1`, e.g. 6 drafts -> `K_s` up to 7 across `C=2..16`) so
    // confident streams expand automatically while uncertain streams stop
    // early inside `propose_batch`. Greedy-only cohorts (`T=0`) and explicit
    // `ATLAS_QWEN4_BATCHED_RAGGED_MTP=0` use `CohortAcceptEwma` + zero-sync
    // uniform `argmax_bf16_rows`. DFlash2 emits its full γ block in one
    // forward pass, so bypass `CohortAcceptEwma` narrowing and keep full
    // `min(configured, adaptive_verify_width(live) - 1)` (γ=7, K=8 across
    // C=2..=16 -> 16 × 8 = 128 rows).
    let ragged_active =
        adaptive_k && batched_ragged_enabled() && !model.is_ep() && step_cohort_sampled();
    let width_capped = resolve_cohort_width_with_controller(
        batched_armed,
        adaptive_k,
        cohort_adaptive_k_enabled() && !ragged_active && !proposer_is_dflash,
        ladder(),
        live,
        configured,
        &COHORT_ACCEPT_EWMA,
    );
    // FLUX_SPEC_VERIFY=stochastic: acceptance-driven depth (None when off).
    // DFlash emits an entire block in a single forward pass, so bypass MTP
    // adaptive_depth truncation and keep the full configured block width.
    let stochastic = if proposer_is_dflash {
        None
    } else {
        crate::scheduler::adaptive_depth::stochastic_depth(width_capped)
    };
    resolve_effective_depth(
        proposer_is_dflash,
        batched_armed,
        adaptive_k,
        live,
        width_capped,
        stochastic,
    )
}

/// `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1`, parsed strictly once outside unit tests.
pub(in super::super) fn adaptive_selected() -> bool {
    let read_live = || match spark_model::layers::qwen4_verify_lanes::adaptive_k_selected() {
        Ok(true) => {
            match spark_model::layers::qwen4_verify_lanes::adaptive_verify_profile() {
                    Ok(spark_model::layers::qwen4_verify_lanes::AdaptiveVerifyProfile::K4C7C16) => {
                        tracing::info!(
                            "QWEN4_BATCHED_ADAPTIVE_K_ENGAGED profile=k4_c7_c16 widths=C1-C6:K5,C7-C32:K4,C33-C42:K3,C43-C128:K2"
                        );
                    }
                    Ok(
                        spark_model::layers::qwen4_verify_lanes::AdaptiveVerifyProfile::SingleSweepDefault,
                    ) => {
                        tracing::info!(
                            max_k = spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K,
                            max_frame_rows =
                                spark_model::layers::qwen4_verify_lanes::adaptive_max_frame_rows(),
                            "QWEN4_BATCHED_ADAPTIVE_K_ENGAGED profile=single_frame_adaptive"
                        );
                    }
                    Err(error) => {
                        tracing::warn!(
                            "{error:#}; falling back to default adaptive verify width"
                        );
                    }
                }
            true
        }
        Ok(false) => false,
        Err(error) => {
            tracing::warn!("{error:#}; keeping the fixed verify width");
            false
        }
    };
    #[cfg(not(test))]
    {
        static ADAPTIVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ADAPTIVE.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// Cached [`AdaptiveVerifyProfile`] so `adaptive_drafts` avoids an uncached
/// `std::env::var("ATLAS_QWEN4_ADAPTIVE_K4_C7_C16")` syscall on every step.
pub(super) fn adaptive_profile_cached()
-> Option<spark_model::layers::qwen4_verify_lanes::AdaptiveVerifyProfile> {
    let read_live = || match spark_model::layers::qwen4_verify_lanes::adaptive_verify_profile() {
        Ok(profile) => Some(profile),
        Err(error) => {
            tracing::warn!("{error:#}; falling back to default adaptive verify width");
            None
        }
    };
    #[cfg(not(test))]
    {
        static PROFILE: std::sync::OnceLock<
            Option<spark_model::layers::qwen4_verify_lanes::AdaptiveVerifyProfile>,
        > = std::sync::OnceLock::new();
        *PROFILE.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// Drafts per sequence at `live` under an explicit single-frame row budget
/// `max_frame_rows`, never above `configured`.
pub(in super::super) fn adaptive_drafts_with_max_rows(
    live: usize,
    configured: usize,
    max_frame_rows: usize,
) -> usize {
    let width = match adaptive_profile_cached() {
        Some(spark_model::layers::qwen4_verify_lanes::AdaptiveVerifyProfile::K4C7C16) => {
            spark_model::layers::qwen4_verify_lanes::adaptive_verify_width_for_profile_and_max_rows(
                live,
                spark_model::layers::qwen4_verify_lanes::AdaptiveVerifyProfile::K4C7C16,
                max_frame_rows,
            )
        }
        Some(
            spark_model::layers::qwen4_verify_lanes::AdaptiveVerifyProfile::SingleSweepDefault,
        )
        | None => spark_model::layers::qwen4_verify_lanes::adaptive_verify_width_for_max_rows(
            live,
            max_frame_rows,
        ),
    };
    width
        .map_or(1, |k| k.saturating_sub(1).max(1))
        .min(configured)
}

/// Drafts per sequence at `live` under the active GPU's single-frame adaptive
/// ceiling (`128 / live` on GB10 `<= 64` SMs; `256 / live` on `> 64`-SM GPUs,
/// clamped to `ADAPTIVE_MIN_K..=ADAPTIVE_MAX_K`), or the explicit opt-in
/// `ATLAS_QWEN4_ADAPTIVE_K4_C7_C16=1` (`K4C7C16`) profile, never above
/// `configured`. C129+ has no admitted verify width; startup refuses adaptive
/// servers wider than C128, so the one-draft floor is only defensive.
pub(in super::super) fn adaptive_drafts(live: usize, configured: usize) -> usize {
    adaptive_drafts_with_max_rows(
        live,
        configured,
        spark_model::layers::qwen4_verify_lanes::adaptive_max_frame_rows(),
    )
}

/// Shorten every flat pending chain longer than `num_drafts` to the current
/// width. The proposer already wrote rows for the whole chain; dropping its
/// tail is the same outcome as an MTP confidence-gate truncation, and
/// `after_verify` trims from the rows actually computed. Trees and sampled
/// proposals are left untouched (they never batch).
pub(in super::super) fn narrow_pending_drafts(
    active: &mut [ActiveSeq],
    idxs: &[usize],
    num_drafts: usize,
) {
    for &idx in idxs {
        let a = &mut active[idx];
        if a.pending_tree_payload.is_none()
            && a.pending_dflash_sampled_proposal.is_none()
            && a.pending_drafts.len() > num_drafts
        {
            let target = num_drafts.max(1);
            a.pending_drafts.truncate(target);
            crate::scheduler::stochastic_verify_walk::truncate_seq_rejection_q(a, target);
        }
    }
}
