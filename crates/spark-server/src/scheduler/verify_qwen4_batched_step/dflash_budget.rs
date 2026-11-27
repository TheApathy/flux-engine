// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash2 non-greedy draft width, p_cum truncation, and ragged verify-graph bucket quantization.

use spark_model::traits::Model;

use super::planning::batched_ragged_enabled;
use crate::scheduler::{ActiveSeq, DraftOrigin};

/// Optional override for non-greedy (`T > 0`) DFlash2 draft width under
/// `FLUX_SPEC_VERIFY=gumbel|rejection`: `1..=7` caps `num_drafts` to
/// `configured.min(v)`, `0` keeps full `configured` without narrowing, and
/// unset uses the measured GB10 cohort/policy defaults (`7` for `C <= 8`, `5`
/// for `C > 8` under `Rejection`; `6` for `C <= 2`, `5` for `C <= 4`, `4` for
/// `C > 4` under `Gumbel`).
const DFLASH_GUMBEL_GAMMA_ENV: &str = "ATLAS_QWEN4_DFLASH_GUMBEL_GAMMA";

/// Maximum override value admitted for [`DFLASH_GUMBEL_GAMMA_ENV`] (block-8
/// DFlash2 carries at most 7 draft slots).
const DFLASH_GUMBEL_GAMMA_MAX: usize = 7;

/// Parse [`DFLASH_GUMBEL_GAMMA_ENV`]: `Some(0)` disables narrowing,
/// `Some(1..=7)` sets an explicit cap, and `None` selects the default profile.
pub(in super::super) fn parse_dflash_gumbel_gamma_override(raw: Option<&str>) -> Option<usize> {
    let raw = raw?.trim();
    if raw.is_empty() {
        return None;
    }
    match raw.parse::<usize>() {
        Ok(gamma) if gamma <= DFLASH_GUMBEL_GAMMA_MAX => Some(gamma),
        _ => {
            tracing::warn!(
                "{DFLASH_GUMBEL_GAMMA_ENV}={raw:?} invalid (want 0 or 1..={DFLASH_GUMBEL_GAMMA_MAX}); using default"
            );
            None
        }
    }
}

fn dflash_gumbel_gamma_override() -> Option<usize> {
    static OVERRIDE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        parse_dflash_gumbel_gamma_override(std::env::var(DFLASH_GUMBEL_GAMMA_ENV).ok().as_deref())
    })
}

/// SM count threshold above which DFlash2 draft-width, `p_cum`, and ragged
/// bucket defaults target high-SM discrete Blackwell GPUs (SM120/SM103) while
/// preserving exact GB10 (`sm_count <= 64`) policies.
const DFLASH_HIGH_SM_THRESHOLD: u32 = 64;

/// Pure SM-aware resolver for DFlash2 `T > 0` Gumbel / GPU rejection draft width.
///
/// Narrows ONLY when `proposer_is_dflash`, `policy` is `Gumbel` or `Rejection`,
/// and `all_non_greedy` are all true; any greedy sequence in the cohort (or any
/// non-DFlash / Argmax / Stochastic route) keeps full `configured`.
///
/// Under `SpecVerifyPolicy::Rejection`:
/// - GB10 (`sm_count <= 64`): keeps `gamma = 7` for `C <= 8` and caps `C > 8` at `gamma = 5`.
/// - High-SM Blackwell (`sm_count > 64`): preserves full `gamma = 7` (`K = 8`)
///   through `C <= 32` (`32 * 8 = 256 <= QWEN4_CROSS_SEQ_WIDE_MAX_ROWS`).
#[must_use]
pub(in super::super) const fn resolve_dflash_gumbel_num_drafts_for_sm(
    proposer_is_dflash: bool,
    policy: crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy,
    all_non_greedy: bool,
    live: usize,
    configured: usize,
    env_override: Option<usize>,
    sm_count: u32,
) -> usize {
    if !proposer_is_dflash
        || !matches!(
            policy,
            crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
                | crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Rejection
        )
        || !all_non_greedy
        || configured == 0
    {
        return configured;
    }
    match env_override {
        Some(0) => configured,
        Some(gamma) => {
            if configured < gamma {
                configured
            } else {
                gamma
            }
        }
        None => match policy {
            crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Rejection => {
                let cap = if sm_count > DFLASH_HIGH_SM_THRESHOLD {
                    match live {
                        0..=32 => 7,
                        _ => 5,
                    }
                } else {
                    match live {
                        0..=8 => 7,
                        _ => 5,
                    }
                };
                if configured < cap { configured } else { cap }
            }
            // Coupled Gumbel-Max peaks at wider `gamma = 6..5` on small cohorts
            // (`C <= 4`) and narrows to `gamma = 4` for `C > 4` to avoid
            // bandwidth-bound MoE expert waste.
            crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Gumbel => {
                let cap = match live {
                    0..=2 => 6,
                    3..=4 => 5,
                    _ => 4,
                };
                if configured < cap { configured } else { cap }
            }
            _ => configured,
        },
    }
}

/// Pure resolver for DFlash2 `T > 0` Gumbel / GPU rejection draft width on the
/// active device's SM count.
#[must_use]
pub(in super::super) fn resolve_dflash_gumbel_num_drafts(
    proposer_is_dflash: bool,
    policy: crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy,
    all_non_greedy: bool,
    live: usize,
    configured: usize,
    env_override: Option<usize>,
) -> usize {
    resolve_dflash_gumbel_num_drafts_for_sm(
        proposer_is_dflash,
        policy,
        all_non_greedy,
        live,
        configured,
        env_override,
        atlas_core::device::active_sm_count(),
    )
}

/// Resolve DFlash2 `T > 0` Gumbel draft width for `live` active sequences
/// (used at both `C = 1` in `verify_dflash_step` and `C >= 2` here).
#[must_use]
pub(in super::super) fn dflash_gumbel_num_drafts(
    model: &dyn Model,
    all_non_greedy: bool,
    live: usize,
    configured: usize,
) -> usize {
    resolve_dflash_gumbel_num_drafts(
        model.proposer_is_dflash(),
        crate::scheduler::stochastic_verify_walk::spec_verify_policy(),
        all_non_greedy,
        live,
        configured,
        dflash_gumbel_gamma_override(),
    )
}

/// Count non-finished active sequences and report whether every live sequence
/// is non-greedy (`temperature > 0.0` and finite).
#[must_use]
pub(super) fn cohort_live_and_all_non_greedy(active: &[ActiveSeq]) -> (usize, bool) {
    let mut live = 0usize;
    let mut all_non_greedy = true;
    for a in active {
        if !a.finished {
            live += 1;
            if !a.temperature.is_finite() || a.temperature <= 0.0 {
                all_non_greedy = false;
            }
        }
    }
    (live, live > 0 && all_non_greedy)
}

/// Resolve the cohort draft width for `active`, keeping full `configured`
/// whenever any live sequence is greedy (`temperature <= 0.0`).
#[must_use]
pub(in super::super) fn dflash_gumbel_cohort_num_drafts(
    model: &dyn Model,
    active: &[ActiveSeq],
    configured: usize,
) -> usize {
    let (live, all_non_greedy) = cohort_live_and_all_non_greedy(active);
    dflash_gumbel_num_drafts(model, all_non_greedy, live, configured)
}

/// Optional cumulative selector-probability truncation threshold for DFlash2
/// (`ATLAS_QWEN4_DFLASH_PCUM_THRESH`): `<= 0.0` disables truncation;
/// `(0.0, 1.0]` sets an explicit threshold; unset uses policy- and
/// cohort-calibrated defaults (`0.55/0.58/0.58/0.58` across `C <= 2/4/8/>8`
/// for lossless `max_z q_t(z)` `Rejection` `T > 0`; `0.05/0.06/0.10/0.13`
/// across `C <= 2/4/8/>8` for `Gumbel` `T > 0`; `0.25` for greedy `T == 0`).
const DFLASH_PCUM_THRESH_ENV: &str = "ATLAS_QWEN4_DFLASH_PCUM_THRESH";
/// Calibrated default cumulative selector thresholds for non-greedy (`T > 0`)
/// DFlash2 frames under `SpecVerifyPolicy::Rejection` by cohort size
/// (`C <= 2`, `C <= 4`, `C <= 8`, `C > 8`) on GB10 (`sm_count <= 64`),
/// calibrated for prefix-only `out_max_prob[t] = max_z q_t(z | x_{<t})`
/// (`TV = 0` lossless maximal coupling) with upward bucket alignment.
const DFLASH_PCUM_DEFAULT_REJECTION_C2: f32 = 0.55;
const DFLASH_PCUM_DEFAULT_REJECTION_C4: f32 = 0.58;
const DFLASH_PCUM_DEFAULT_REJECTION_C8: f32 = 0.58;
const DFLASH_PCUM_DEFAULT_REJECTION_HIGH_C: f32 = 0.58;
/// Default `SpecVerifyPolicy::Rejection` (`T > 0`) `p_cum` thresholds on
/// high-SM discrete Blackwell GPUs (`sm_count > 64`): disable truncation
/// (`0.0`, exact fixed `K = 8`, single steady CUDA graph, `TV = 0`) through
/// `C <= 24`, and apply light `0.15` truncation above `C > 24`.
const DFLASH_PCUM_DEFAULT_REJECTION_HIGH_SM_C24: f32 = 0.0;
const DFLASH_PCUM_DEFAULT_REJECTION_HIGH_SM_WIDE: f32 = 0.15;
/// Calibrated default cumulative selector thresholds for non-greedy (`T > 0`)
/// DFlash2 frames under `SpecVerifyPolicy::Gumbel` by cohort size
/// (`C <= 2`, `C <= 4`, `C <= 8`, `C > 8`), avoiding excessive draft truncation
/// while bounding high-concurrency MoE expert waste (`0-ULP` exact output).
const DFLASH_PCUM_DEFAULT_GUMBEL_C2: f32 = 0.05;
const DFLASH_PCUM_DEFAULT_GUMBEL_C4: f32 = 0.06;
const DFLASH_PCUM_DEFAULT_GUMBEL_C8: f32 = 0.10;
const DFLASH_PCUM_DEFAULT_GUMBEL_HIGH_C: f32 = 0.13;
/// Calibrated default cumulative selector threshold for greedy (`T == 0`)
/// DFlash2 frames (keeps `K = 8` on high-acceptance code, narrows uncertain prose).
const DFLASH_PCUM_DEFAULT_GREEDY: f32 = 0.25;

#[must_use]
pub(super) const fn dflash_pcum_default_for_cohort_with_policy_for_sm(
    temperature: f32,
    cohort_size: usize,
    policy: crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy,
    sm_count: u32,
) -> f32 {
    if temperature.is_finite() && temperature > 0.0 {
        match policy {
            crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Rejection => {
                if sm_count > DFLASH_HIGH_SM_THRESHOLD {
                    match cohort_size {
                        0..=24 => DFLASH_PCUM_DEFAULT_REJECTION_HIGH_SM_C24,
                        _ => DFLASH_PCUM_DEFAULT_REJECTION_HIGH_SM_WIDE,
                    }
                } else {
                    match cohort_size {
                        0..=2 => DFLASH_PCUM_DEFAULT_REJECTION_C2,
                        3..=4 => DFLASH_PCUM_DEFAULT_REJECTION_C4,
                        5..=8 => DFLASH_PCUM_DEFAULT_REJECTION_C8,
                        _ => DFLASH_PCUM_DEFAULT_REJECTION_HIGH_C,
                    }
                }
            }
            _ => match cohort_size {
                0..=2 => DFLASH_PCUM_DEFAULT_GUMBEL_C2,
                3..=4 => DFLASH_PCUM_DEFAULT_GUMBEL_C4,
                5..=8 => DFLASH_PCUM_DEFAULT_GUMBEL_C8,
                _ => DFLASH_PCUM_DEFAULT_GUMBEL_HIGH_C,
            },
        }
    } else {
        DFLASH_PCUM_DEFAULT_GREEDY
    }
}

#[must_use]
pub(super) fn dflash_pcum_default_for_cohort_with_policy(
    temperature: f32,
    cohort_size: usize,
    policy: crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy,
) -> f32 {
    dflash_pcum_default_for_cohort_with_policy_for_sm(
        temperature,
        cohort_size,
        policy,
        atlas_core::device::active_sm_count(),
    )
}

/// Parse [`DFLASH_PCUM_THRESH_ENV`]: `Some(0.0)` disables truncation,
/// `Some(tau)` in `(0.0, 1.0]` overrides the default threshold, and `None`
/// selects the temperature-calibrated default.
pub(in super::super) fn parse_dflash_pcum_thresh_override(raw: Option<&str>) -> Option<f32> {
    let raw = raw?.trim();
    if raw.is_empty() {
        return None;
    }
    match raw.parse::<f32>() {
        Ok(tau) if tau.is_finite() && tau <= 1.0 => Some(tau.max(0.0)),
        _ => {
            tracing::warn!(
                "{DFLASH_PCUM_THRESH_ENV}={raw:?} invalid (want finite float <= 1.0, 0 disables); using default"
            );
            None
        }
    }
}

fn dflash_pcum_thresh_override() -> Option<f32> {
    static OVERRIDE: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        parse_dflash_pcum_thresh_override(std::env::var(DFLASH_PCUM_THRESH_ENV).ok().as_deref())
    })
}

/// Pure kept-draft count (`1..=draft_len`) and stopping-frontier cumulative
/// confidence `frontier_pcum` under cumulative selector confidence
/// `p_cum = prod_{j=0..=i} probs[j]` against threshold `tau` resolved for
/// `(temperature, cohort_size, policy)`.
///
/// When `probs[j]` is prefix-only `max_z q_j(z | x_{<j})` (independent of the
/// sampled draft `x_j`), both `keep` and `frontier_pcum` depend solely on
/// `(x_0 .. x_{keep-1})` and NEVER inspect `x_{keep..}`. This makes forward
/// truncation and upward bucket extension strictly `F_t`-measurable (`TV = 0`
/// under maximal-coupling rejection sampling).
///
/// Draft `0` is always retained (`keep >= 1`, so `K_s >= 2`) to preserve
/// batched CUDA-graph and `KEEP_LATE` eligibility.
#[must_use]
pub(in super::super) fn dflash_pcum_keep_len_and_frontier_for_cohort_with_policy(
    draft_len: usize,
    probs: &[f32],
    temperature: f32,
    cohort_size: usize,
    policy: crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy,
    env_override: Option<f32>,
) -> (usize, f32) {
    if draft_len <= 1 || probs.is_empty() {
        return (draft_len, 1.0);
    }
    let tau = match env_override {
        Some(v) => v,
        None => dflash_pcum_default_for_cohort_with_policy(temperature, cohort_size, policy),
    };
    if !tau.is_finite() || tau <= 0.0 {
        return (draft_len, 1.0);
    }
    let n = draft_len.min(probs.len());
    let mut p_cum = 1.0f32;
    let mut keep = 1usize;
    for (i, &raw_p) in probs.iter().take(n).enumerate() {
        let p = if raw_p.is_finite() {
            raw_p.clamp(0.0, 1.0)
        } else {
            0.0
        };
        p_cum *= p;
        if i >= 1 {
            if p_cum >= tau {
                keep = i + 1;
            } else {
                return (keep, p_cum);
            }
        }
    }
    (keep, p_cum)
}

/// Pure kept-draft count (`1..=draft_len`) under cumulative selector confidence
/// `p_cum = prod_{j=0..=i} probs[j]` against threshold `tau` resolved for
/// `(temperature, cohort_size, policy)`.
///
/// Draft `0` is always retained (`keep >= 1`, so `K_s >= 2`) to preserve
/// batched CUDA-graph and `KEEP_LATE` eligibility.
#[must_use]
pub(in super::super) fn dflash_pcum_keep_len_for_cohort_with_policy(
    draft_len: usize,
    probs: &[f32],
    temperature: f32,
    cohort_size: usize,
    policy: crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy,
    env_override: Option<f32>,
) -> usize {
    dflash_pcum_keep_len_and_frontier_for_cohort_with_policy(
        draft_len,
        probs,
        temperature,
        cohort_size,
        policy,
        env_override,
    )
    .0
}

#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(in super::super) fn dflash_pcum_keep_len_for_cohort(
    draft_len: usize,
    probs: &[f32],
    temperature: f32,
    cohort_size: usize,
    env_override: Option<f32>,
) -> usize {
    dflash_pcum_keep_len_for_cohort_with_policy(
        draft_len,
        probs,
        temperature,
        cohort_size,
        crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Gumbel,
        env_override,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(in super::super) fn dflash_pcum_keep_len(
    draft_len: usize,
    probs: &[f32],
    temperature: f32,
    env_override: Option<f32>,
) -> usize {
    dflash_pcum_keep_len_for_cohort(draft_len, probs, temperature, 1, env_override)
}

#[must_use]
fn dflash_pcum_target_len_and_frontier(
    a: &ActiveSeq,
    is_dflash: bool,
    probs: &[f32],
    cohort_size: usize,
    policy: crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy,
    env_override: Option<f32>,
) -> (usize, f32) {
    let before = a.pending_drafts.len();
    if !is_dflash
        || before <= 1
        || probs.is_empty()
        || a.draft_origin != DraftOrigin::Proposer
        || a.pending_tree_payload.is_some()
        || a.pending_dflash_sampled_proposal.is_some()
        || crate::scheduler::think_budget_nudge::close_allowed(a)
        // Under uniform (non-ragged) cohorts, `harmonize_batched_verify_cohort_widths`
        // trims streams down to `min_k`, which would retroactively drop a draft
        // below another stream's stopping point and bias stochastic Rejection.
        || (policy == crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Rejection
            && a.temperature.is_finite()
            && a.temperature > 0.0
            && cohort_size >= 2
            && !batched_ragged_enabled())
    {
        return (before, 1.0);
    }
    dflash_pcum_keep_len_and_frontier_for_cohort_with_policy(
        before,
        probs,
        a.temperature,
        cohort_size,
        policy,
        env_override,
    )
}

#[must_use]
fn dflash_pcum_target_len(
    a: &ActiveSeq,
    is_dflash: bool,
    probs: &[f32],
    cohort_size: usize,
    policy: crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy,
    env_override: Option<f32>,
) -> usize {
    dflash_pcum_target_len_and_frontier(a, is_dflash, probs, cohort_size, policy, env_override).0
}

fn log_dflash_pcum_truncation_once(before: usize, after: usize, temperature: f32) {
    static ENGAGED: std::sync::Once = std::sync::Once::new();
    ENGAGED.call_once(|| {
        tracing::info!(
            before,
            after,
            temperature,
            "DFLASH2_PCUM_TRUNCATION_ENGAGED"
        );
    });
}

/// Truncate freshly-proposed DFlash2 flat drafts using per-draft conditional
/// confidence probabilities from `DflashProposerState::last_draft_probs()`.
///
/// Lossless (`0-ULP`): target verification guarantees bit-identical output
/// tokens for any draft length `1..=7`. Never truncates retrieval chains,
/// forced-nudge chains, tree payloads, or probabilistic sampled-q frames.
pub(in super::super) fn apply_dflash_pcum_truncation(a: &mut ActiveSeq, is_dflash: bool) {
    apply_dflash_pcum_truncation_for_cohort(a, is_dflash, 1);
}

pub(in super::super) fn apply_dflash_pcum_truncation_for_cohort(
    a: &mut ActiveSeq,
    is_dflash: bool,
    cohort_size: usize,
) {
    if !is_dflash
        || a.pending_drafts.len() <= 1
        || !crate::scheduler::verify_dflash_step::dflash_pcum_enabled_for_seq_in_cohort(
            a,
            cohort_size,
        )
    {
        return;
    }
    let keep = {
        let a_ref: &ActiveSeq = a;
        let Some(probs) = a_ref
            .seq
            .proposer_state
            .as_ref()
            .and_then(|s| {
                s.as_any()
                    .downcast_ref::<spark_model::layers::dflash_head::DflashProposerState>()
            })
            .map(spark_model::layers::dflash_head::DflashProposerState::last_draft_probs)
        else {
            return;
        };
        dflash_pcum_target_len(
            a_ref,
            is_dflash,
            probs,
            cohort_size,
            crate::scheduler::stochastic_verify_walk::spec_verify_policy(),
            dflash_pcum_thresh_override(),
        )
    };
    let before = a.pending_drafts.len();
    if keep < before {
        a.pending_drafts.truncate(keep);
        crate::scheduler::stochastic_verify_walk::truncate_seq_rejection_q(a, keep);
        log_dflash_pcum_truncation_once(before, keep, a.temperature);
    }
}

#[cfg(test)]
pub(super) fn apply_dflash_pcum_truncation_with_probs(
    a: &mut ActiveSeq,
    is_dflash: bool,
    probs: &[f32],
    env_override: Option<f32>,
) {
    let before = a.pending_drafts.len();
    let keep = dflash_pcum_target_len(
        a,
        is_dflash,
        probs,
        1,
        crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Gumbel,
        env_override,
    );
    if keep < before {
        a.pending_drafts.truncate(keep);
    }
}

/// Graph-bucket row granularity for ragged DFlash2 batched-verify frames of
/// `b` sequences on a GPU with `sm_count` SMs.
///
/// GB10 (`sm_count <= 64`) preserves exact `2/4/8/16` buckets matching
/// `qwen4_mtp::ragged_row_bucket_step`. High-SM Blackwell (`sm_count > 64`)
/// widens coarse buckets at `b >= 12` (`16` for `12..=23`, `32` for `>= 24`)
/// so ragged `p_cum > 0` runs keep at most 3-4 suffix graphs while staying a
/// multiple of `verify_qwen4_batched::batched_verify_graph_capture_row_admitted`.
#[inline]
#[must_use]
pub(super) const fn dflash_ragged_row_bucket_step_for_sm(b: usize, sm_count: u32) -> usize {
    if sm_count > DFLASH_HIGH_SM_THRESHOLD {
        match b {
            0..=1 => 1,
            2 => 2,
            3..=5 => 4,
            6..=11 => 8,
            12..=23 => 16,
            _ => 32,
        }
    } else {
        match b {
            0..=1 => 1,
            2 => 2,
            3..=5 => 4,
            6..=16 => 8,
            _ => 16,
        }
    }
}

/// Graph-bucket row granularity for ragged DFlash2 batched-verify frames of
/// `b` sequences on the active device.
#[inline]
#[must_use]
pub(super) fn dflash_ragged_row_bucket_step(b: usize) -> usize {
    dflash_ragged_row_bucket_step_for_sm(b, atlas_core::device::active_sm_count())
}

/// Remaining-token lead margin below the cohort median (`rem < median - 8`)
/// that marks a stream as ahead for zero-cost bucket-rounding tie-breaking.
const DFLASH_BUCKET_TRIM_AHEAD_MARGIN: usize = 8;

/// Row-count tile cliff (`r = 56` is not a clean 16/32-row tile boundary for
/// `m32`/`m64` kernels and measured slower than `r = 64` at `C = 16`): when
/// `b >= 12` and bucket quantization lands on `56`, snap down by 8 rows to `48`
/// provided every lane preserves its `len >= 1` (`K >= 2`) floor.
const DFLASH_RAGGED_CLIFF_ROWS: usize = 56;
const DFLASH_RAGGED_CLIFF_MIN_SEQS: usize = 12;
const DFLASH_RAGGED_CLIFF_DROP_ROWS: usize = 8;

/// Compute how many draft rows to trim from a ragged cohort (`b` sequences,
/// `total_rows = sum(len + 1)`, `trimmable = sum(len.saturating_sub(1))`)
/// to hit the verify CUDA-graph bucket grid while skipping the `r = 56` cliff
/// at `b >= 12` (`56 -> 48`).
#[inline]
#[must_use]
fn ragged_bucket_trim_rows(b: usize, total_rows: usize, trimmable: usize) -> usize {
    let step = dflash_ragged_row_bucket_step(b);
    if step <= 1 {
        return 0;
    }
    let mut rem = total_rows % step;
    let quantized = total_rows - rem;
    if b >= DFLASH_RAGGED_CLIFF_MIN_SEQS
        && quantized == DFLASH_RAGGED_CLIFF_ROWS
        && trimmable >= rem + DFLASH_RAGGED_CLIFF_DROP_ROWS
    {
        rem += DFLASH_RAGGED_CLIFF_DROP_ROWS;
    }
    rem
}

/// Snap ragged draft lengths `lens` (`b >= 2`, `min != max`) so
/// `sum(lens[s] + 1)` aligns to [`dflash_ragged_row_bucket_step`] (and skips
/// the `r = 56` cliff down to `48` for `b >= 12`) by trimming the lowest
/// tail-`p_cum` extra draft token(s) first while keeping every `lens[s] >= 1`
/// (`K_s >= 2`). Uniform cohorts (`min == max`) are left untouched because
/// uniform frames are always graph-admitted.
pub(in super::super) fn quantize_ragged_lens_to_graph_bucket(
    lens: &mut [usize],
    pcum_chains: &[Vec<f32>],
) {
    quantize_ragged_lens_to_graph_bucket_with_remaining(lens, pcum_chains, None);
}

/// [`quantize_ragged_lens_to_graph_bucket`] with optional per-sequence
/// remaining token budgets for `D_bucket_trim_ahead_first` straggler-friendly
/// tie-breaking: when rounding down `1..step-1` rows to hit a graph bucket
/// boundary (or dropping 8 additional rows off the `r = 56` cliff at
/// `b >= 12`), streams ahead of the cohort median
/// (`rem < median.saturating_sub(8)`) are trimmed first (lowest `tail_p` among
/// ahead streams), falling back to lowest `tail_p` across all trimmable streams
/// while strictly preserving the `lens[s] >= 1` (`K_s >= 2`) floor.
pub(in super::super) fn quantize_ragged_lens_to_graph_bucket_with_remaining(
    lens: &mut [usize],
    pcum_chains: &[Vec<f32>],
    remaining_tokens: Option<&[usize]>,
) {
    let b = lens.len();
    let step = dflash_ragged_row_bucket_step(b);
    if step <= 1 || b != pcum_chains.len() {
        return;
    }
    let min_len = lens.iter().copied().min().unwrap_or(0);
    let max_len = lens.iter().copied().max().unwrap_or(0);
    if min_len == max_len {
        return;
    }
    let total_rows: usize = lens.iter().map(|&l| l + 1).sum();
    let trimmable: usize = lens.iter().map(|&l| l.saturating_sub(1)).sum();
    let mut rem = ragged_bucket_trim_rows(b, total_rows, trimmable);
    if rem == 0 {
        return;
    }
    let ahead_threshold = remaining_tokens.filter(|rems| rems.len() == b).map(|rems| {
        let mut sorted = rems.to_vec();
        sorted.sort_unstable();
        sorted[b / 2].saturating_sub(DFLASH_BUCKET_TRIM_AHEAD_MARGIN)
    });
    while rem > 0 {
        let mut victim: Option<(usize, bool, f32, usize)> = None;
        for (s, &len) in lens.iter().enumerate() {
            if len <= 1 {
                continue;
            }
            let is_ahead = match (ahead_threshold, remaining_tokens) {
                (Some(threshold), Some(rems)) => rems[s] < threshold,
                _ => false,
            };
            let tail_p = pcum_chains[s].get(len - 1).copied().unwrap_or(1.0);
            let better = match victim {
                None => true,
                Some((best_s, best_ahead, best_p, best_len)) => {
                    (is_ahead && !best_ahead)
                        || (is_ahead == best_ahead
                            && (tail_p < best_p
                                || (tail_p == best_p
                                    && (len > best_len || (len == best_len && s > best_s)))))
                }
            };
            if better {
                victim = Some((s, is_ahead, tail_p, len));
            }
        }
        let Some((s, _, _, _)) = victim else {
            break;
        };
        lens[s] -= 1;
        rem -= 1;
    }
}

/// Compute how many draft rows to extend UPWARD from a ragged cohort (`b`
/// sequences, `total_rows = sum(len + 1)`, `slack = sum(max_len - len)`) to
/// hit the verify CUDA-graph bucket grid without adding a wasteful `+8` jump
/// off `r = 56 -> 64` (unlike downward `56 -> 48` which removes rows, at
/// `gamma = 5` on `C = 16` almost every step lands in `49..=56` rows so
/// jumping `+8` to `64` adds 8 low-confidence tail rows per round). Returns
/// `0` if `total_rows` is already aligned or `slack` cannot cover the upward
/// snap (never trims downward).
#[inline]
#[must_use]
fn ragged_bucket_extend_rows(b: usize, total_rows: usize, slack: usize) -> usize {
    let step = dflash_ragged_row_bucket_step(b);
    if step <= 1 {
        return 0;
    }
    let rem = total_rows % step;
    let add = if rem == 0 { 0 } else { step - rem };
    if slack < add { 0 } else { add }
}

/// Snap ragged `p_cum`-stopped draft lengths `lens` UPWARD (`lens[s] <= max_lens[s]`)
/// so `sum(lens[s] + 1)` aligns to [`dflash_ragged_row_bucket_step`] (without
/// adding a wasteful `+8` jump off `r = 56 -> 64`) using ONLY the
/// `F_{s, L_s}`-measurable stopping lengths `lens`, stopping-frontier
/// `frontier_pcums`, and pre-proposal `remaining_tokens`.
///
/// Because this helper NEVER trims below `lens[s]` and NEVER inspects any draft
/// token or probability at index `>= orig_lens[s]`, the event `{final_len[s] > t}`
/// stays strictly `F_{s, t}`-measurable for every stream `s` and draft slot `t`,
/// preserving exact `TV = 0` maximal-coupling rejection sampling at `C >= 2`.
pub(in super::super) fn quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
    lens: &mut [usize],
    max_lens: &[usize],
    frontier_pcums: &[f32],
    remaining_tokens: Option<&[usize]>,
) {
    let b = lens.len();
    let step = dflash_ragged_row_bucket_step(b);
    if step <= 1 || b != max_lens.len() || b != frontier_pcums.len() {
        return;
    }
    let min_len = lens.iter().copied().min().unwrap_or(0);
    let max_len = lens.iter().copied().max().unwrap_or(0);
    if min_len == max_len {
        return;
    }
    let total_rows: usize = lens.iter().map(|&l| l + 1).sum();
    let slack: usize = lens
        .iter()
        .zip(max_lens.iter())
        .map(|(&l, &cap)| cap.saturating_sub(l))
        .sum();
    let mut need_add = ragged_bucket_extend_rows(b, total_rows, slack);
    if need_add == 0 {
        return;
    }
    let ahead_threshold = remaining_tokens.filter(|rems| rems.len() == b).map(|rems| {
        let mut sorted = rems.to_vec();
        sorted.sort_unstable();
        sorted[b / 2].saturating_sub(DFLASH_BUCKET_TRIM_AHEAD_MARGIN)
    });
    let orig_lens = lens.to_vec();
    while need_add > 0 {
        let mut pick: Option<(usize, bool, usize, f32, usize)> = None;
        for (s, (&len, &cap)) in lens.iter().zip(max_lens.iter()).enumerate() {
            if len >= cap {
                continue;
            }
            // Prefer stragglers / behind streams (`!is_ahead`) so extra draft
            // rows go where remaining output work is highest.
            let is_behind = match (ahead_threshold, remaining_tokens) {
                (Some(threshold), Some(rems)) => rems[s] >= threshold,
                _ => false,
            };
            let added = len - orig_lens[s];
            let fp = frontier_pcums[s];
            let frontier = if fp.is_finite() { fp } else { 0.0 };
            let better = match pick {
                None => true,
                Some((best_s, best_behind, best_added, best_frontier, best_len)) => {
                    (is_behind && !best_behind)
                        || (is_behind == best_behind
                            && (added < best_added
                                || (added == best_added
                                    && (frontier > best_frontier
                                        || (frontier == best_frontier
                                            && (len < best_len
                                                || (len == best_len && s < best_s)))))))
                }
            };
            if better {
                pick = Some((s, is_behind, added, frontier, len));
            }
        }
        let Some((s, _, _, _, _)) = pick else {
            break;
        };
        lens[s] += 1;
        need_add -= 1;
    }
}

/// Apply `p_cum` stopping AND lossless upward graph-bucket quantization in a
/// single pass over freshly proposed `propose_idxs` under `SpecVerifyPolicy::Rejection`,
/// truncating `pending_drafts` and GPU rejection `q` buffers only once final
/// bucket-aligned lengths are known.
pub(super) fn apply_dflash_pcum_and_upward_bucket_quantize_cohort(
    active: &mut [ActiveSeq],
    propose_idxs: &[usize],
    is_dflash: bool,
) {
    if !is_dflash {
        return;
    }
    let policy = crate::scheduler::stochastic_verify_walk::spec_verify_policy();
    let env_override = dflash_pcum_thresh_override();
    let mut lane_idxs: Vec<usize> = Vec::with_capacity(propose_idxs.len());
    let mut max_lens: Vec<usize> = Vec::with_capacity(propose_idxs.len());
    let mut lens: Vec<usize> = Vec::with_capacity(propose_idxs.len());
    let mut frontier_pcums: Vec<f32> = Vec::with_capacity(propose_idxs.len());
    let mut remaining_tokens: Vec<usize> = Vec::with_capacity(propose_idxs.len());
    let cohort_size = propose_idxs
        .iter()
        .filter(|&&idx| {
            active.get(idx).is_some_and(|a| {
                !a.finished
                    && !a.pending_drafts.is_empty()
                    && a.pending_tree_payload.is_none()
                    && a.pending_dflash_sampled_proposal.is_none()
            })
        })
        .count();
    for &idx in propose_idxs {
        let Some(a) = active.get(idx) else {
            continue;
        };
        if a.finished
            || a.pending_drafts.is_empty()
            || a.pending_tree_payload.is_some()
            || a.pending_dflash_sampled_proposal.is_some()
        {
            continue;
        }
        let before = a.pending_drafts.len();
        let (keep, frontier) =
            if !crate::scheduler::verify_dflash_step::dflash_pcum_enabled_for_seq_in_cohort(
                a,
                cohort_size,
            ) {
                (before, 1.0)
            } else {
                a.seq
                    .proposer_state
                    .as_ref()
                    .and_then(|s| {
                        s.as_any()
                            .downcast_ref::<spark_model::layers::dflash_head::DflashProposerState>()
                    })
                    .map(spark_model::layers::dflash_head::DflashProposerState::last_draft_probs)
                    .map_or((before, 1.0), |probs| {
                        dflash_pcum_target_len_and_frontier(
                            a,
                            is_dflash,
                            probs,
                            cohort_size,
                            policy,
                            env_override,
                        )
                    })
            };
        let rem_budget = if a.max_output_tokens > 0 {
            a.remaining
                .min(a.max_output_tokens.saturating_sub(a.output_tokens.len()))
        } else {
            a.remaining
        };
        lane_idxs.push(idx);
        max_lens.push(before);
        lens.push(keep);
        frontier_pcums.push(frontier);
        remaining_tokens.push(rem_budget);
    }
    quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
        &mut lens,
        &max_lens,
        &frontier_pcums,
        Some(&remaining_tokens),
    );
    for (&idx, &target_len) in lane_idxs.iter().zip(lens.iter()) {
        let a = &mut active[idx];
        let before = a.pending_drafts.len();
        if target_len < before {
            a.pending_drafts.truncate(target_len);
            crate::scheduler::stochastic_verify_walk::truncate_seq_rejection_q(a, target_len);
            log_dflash_pcum_truncation_once(before, target_len, a.temperature);
        }
    }
}

/// Quantize ragged DFlash2 `pending_drafts` across `idxs` onto the
/// `BATCHED_VERIFY_GRAPH` row bucket grid so suffix graphs never thrash.
pub(super) fn quantize_dflash_ragged_drafts_to_graph_bucket(
    active: &mut [ActiveSeq],
    idxs: &[usize],
) {
    if crate::scheduler::stochastic_verify_walk::spec_verify_policy()
        == crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Rejection
        && idxs.iter().any(|&idx| {
            active.get(idx).is_some_and(|a| {
                !a.finished
                    && !a.pending_drafts.is_empty()
                    && a.temperature.is_finite()
                    && a.temperature > 0.0
            })
        })
    {
        // Under stochastic Rejection, steady-state cohorts are already aligned
        // UPWARD in `apply_dflash_pcum_and_upward_bucket_quantize_cohort` prior
        // to truncation. Never retroactively trim below a stream's `p_cum`
        // stopping point on join/exit ticks (`TV = 0` invariant).
        return;
    }
    let mut lane_stack = [0usize; spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
    let mut lens_stack = [0usize; spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
    let lane_heap;
    let mut lens_heap;
    let mut min_len = usize::MAX;
    let mut max_len = 0usize;
    let mut total_rows = 0usize;
    let (lane_idxs, lens): (&[usize], &mut [usize]) = if idxs.len() <= lane_stack.len() {
        let mut b = 0usize;
        for &idx in idxs {
            let Some(a) = active.get(idx) else {
                continue;
            };
            if a.finished
                || a.pending_drafts.is_empty()
                || a.pending_tree_payload.is_some()
                || a.pending_dflash_sampled_proposal.is_some()
            {
                continue;
            }
            let len = a.pending_drafts.len();
            min_len = min_len.min(len);
            max_len = max_len.max(len);
            total_rows += len + 1;
            lane_stack[b] = idx;
            lens_stack[b] = len;
            b += 1;
        }
        (&lane_stack[..b], &mut lens_stack[..b])
    } else {
        let mut lh = Vec::with_capacity(idxs.len());
        let mut nh = Vec::with_capacity(idxs.len());
        for &idx in idxs {
            let Some(a) = active.get(idx) else {
                continue;
            };
            if a.finished
                || a.pending_drafts.is_empty()
                || a.pending_tree_payload.is_some()
                || a.pending_dflash_sampled_proposal.is_some()
            {
                continue;
            }
            let len = a.pending_drafts.len();
            min_len = min_len.min(len);
            max_len = max_len.max(len);
            total_rows += len + 1;
            lh.push(idx);
            nh.push(len);
        }
        lane_heap = lh;
        lens_heap = nh;
        (&lane_heap, &mut lens_heap)
    };
    let b = lane_idxs.len();
    let trimmable: usize = lens.iter().map(|&l| l.saturating_sub(1)).sum();
    if min_len == max_len || ragged_bucket_trim_rows(b, total_rows, trimmable) == 0 {
        return;
    }
    let mut pcum_chains: Vec<Vec<f32>> = Vec::with_capacity(b);
    let mut remaining_tokens: Vec<usize> = Vec::with_capacity(b);
    for (&idx, &len) in lane_idxs.iter().zip(lens.iter()) {
        let a = &active[idx];
        let rem_budget = if a.max_output_tokens > 0 {
            a.remaining
                .min(a.max_output_tokens.saturating_sub(a.output_tokens.len()))
        } else {
            a.remaining
        };
        remaining_tokens.push(rem_budget);
        let mut chain = Vec::with_capacity(len);
        if a.draft_origin == DraftOrigin::Proposer
            && !crate::scheduler::think_budget_nudge::close_allowed(a)
            && let Some(probs) = a
                .seq
                .proposer_state
                .as_ref()
                .and_then(|s| {
                    s.as_any()
                        .downcast_ref::<spark_model::layers::dflash_head::DflashProposerState>()
                })
                .map(spark_model::layers::dflash_head::DflashProposerState::last_draft_probs)
            && !probs.is_empty()
        {
            let mut p_cum = 1.0f32;
            for i in 0..len {
                let raw_p = probs.get(i).copied().unwrap_or(1.0);
                let p = if raw_p.is_finite() {
                    raw_p.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                p_cum *= p;
                chain.push(p_cum);
            }
        } else {
            chain.resize(len, 1.0);
        }
        pcum_chains.push(chain);
    }
    quantize_ragged_lens_to_graph_bucket_with_remaining(
        lens,
        &pcum_chains,
        Some(&remaining_tokens),
    );
    for (&idx, &target_len) in lane_idxs.iter().zip(lens.iter()) {
        if active[idx].pending_drafts.len() > target_len {
            active[idx].pending_drafts.truncate(target_len);
            crate::scheduler::stochastic_verify_walk::truncate_seq_rejection_q(
                &active[idx],
                target_len,
            );
        }
    }
}
