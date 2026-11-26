// SPDX-License-Identifier: AGPL-3.0-only

//! Sequence-owned verify transaction lanes for batched native-MTP K5.
//!
//! A K-row speculative verify leaves two pieces of per-layer transaction
//! state behind until its commit: the QSA undo journal (one per attention
//! layer, `qwen4_qsa.rs`) and the GDN lazy-commit retain buffer (one per SSM
//! layer, `qwen3_ssm`). Serial C1 speculation commits each frame before the
//! next verify starts, so one lane per layer suffices.
//!
//! `ATLAS_QWEN4_BATCHED_K5=1` verifies several sequences in one forward and
//! commits them afterwards. Each layer then holds one lane per physical SSM
//! slot (`lanes == max_batch_size`), and the batched verify/commit selects the
//! lane of the sequence it is working on through [`LaneGuard`]. Every other
//! path, including per-sequence verifies inside a batched-mode server, runs
//! with no guard and therefore on lane 0: their verify and commit happen in
//! the same scheduler call, so they never overlap another outstanding frame.
//!
//! With the selector off the lane count is 1 and every pointer is the
//! pre-existing single-lane pointer (lane 0, offset 0).

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, ensure};

/// Selector for cross-sequence batched native-MTP verification.
pub const BATCHED_K5_ENV: &str = "ATLAS_QWEN4_BATCHED_K5";

/// Largest lane count (matches the Qwen4 qualified concurrency ceiling).
pub const MAX_LANES: usize = 128;

/// `generation << 8 | lanes` of the current registration (lanes <= 128 fit
/// in the low byte). Generation 0 with 1 lane is the unregistered state.
static STATE: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static CURRENT: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Strict parser: unset or `0` is off, `1` is on, anything else fails closed.
pub fn parse_batched_k5_selector(raw: Option<&str>) -> std::result::Result<bool, &'static str> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_QWEN4_BATCHED_K5 must be exact value 0 or 1"),
    }
}

fn read_batched_k5_selected_live() -> std::result::Result<bool, String> {
    let raw = match std::env::var(BATCHED_K5_ENV) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(format!("{BATCHED_K5_ENV} must be valid UTF-8"));
        }
    };
    parse_batched_k5_selector(raw.as_deref()).map_err(str::to_owned)
}

/// Read the selector from the environment (strict, `OnceLock`-cached outside
/// unit tests so admission pays zero syscalls while unit tests observe live
/// environment mutations).
pub fn batched_k5_selected() -> Result<bool> {
    #[cfg(not(test))]
    {
        static CACHED: std::sync::OnceLock<std::result::Result<bool, String>> =
            std::sync::OnceLock::new();
        CACHED
            .get_or_init(read_batched_k5_selected_live)
            .as_ref()
            .copied()
            .map_err(|msg| anyhow::Error::msg(msg.clone()))
    }
    #[cfg(test)]
    {
        read_batched_k5_selected_live().map_err(anyhow::Error::msg)
    }
}

pub const FIXED_K5_POLICY_FLAGS: [&str; 6] = [
    "ATLAS_QWEN4_MTP_CONF",
    "FLUX_QWEN4_MTP_CONF",
    "ATLAS_QWEN4_MTP_CONF_EARLY",
    "FLUX_QWEN4_MTP_CONF_EARLY",
    "ATLAS_QWEN4_MTP_VOCAB_PRUNE",
    "FLUX_QWEN4_MTP_VOCAB_PRUNE",
];

fn is_valid_cohort_safe_mtp_conf(raw: Option<&str>) -> bool {
    match raw {
        None | Some("0") => true,
        Some(s) if s.is_empty() || s != s.trim() => false,
        Some(s) => s
            .parse::<f32>()
            .is_ok_and(|v| v.is_finite() && (0.0..=1.0).contains(&v)),
    }
}

fn is_valid_cohort_safe_mtp_conf_early(raw: Option<&str>) -> bool {
    matches!(raw, None | Some("0") | Some("1"))
}

pub fn validate_fixed_k5_policy(
    flags: &[Option<&str>],
    conf_file_present: bool,
    ladder_present: bool,
    gdn_multi: Option<&str>,
) -> std::result::Result<(), &'static str> {
    if flags.len() != FIXED_K5_POLICY_FLAGS.len() {
        return Err(
            "batched fixed K5 requires both ATLAS and FLUX aliases of MTP_CONF, MTP_CONF_EARLY and MTP_VOCAB_PRUNE",
        );
    }
    // `propose_batch` (`b >= 2`) unconditionally forces `ChainConfidenceScope::BatchedCohort`
    // (ignoring per-sequence `MTP_CONF` / `MTP_CONF_EARLY`), so valid confidence selectors are
    // safe to pass through for `b == 1` (`SingleSequence`) parity on `C>1` servers.
    // `MTP_VOCAB_PRUNE` MUST remain exact `"0"` on both aliases.
    if !is_valid_cohort_safe_mtp_conf(flags[0]) || !is_valid_cohort_safe_mtp_conf(flags[1]) {
        return Err(
            "batched fixed K5 requires ATLAS/FLUX_QWEN4_MTP_CONF unset, 0, or a finite threshold in 0.0..=1.0",
        );
    }
    if !is_valid_cohort_safe_mtp_conf_early(flags[2])
        || !is_valid_cohort_safe_mtp_conf_early(flags[3])
    {
        return Err("batched fixed K5 requires ATLAS/FLUX_QWEN4_MTP_CONF_EARLY unset, 0, or 1");
    }
    if flags[4] != Some("0") || flags[5] != Some("0") {
        return Err(
            "batched fixed K5 requires both ATLAS and FLUX aliases of MTP_VOCAB_PRUNE set to exact 0",
        );
    }
    if conf_file_present {
        return Err("batched fixed K5 refuses MTP_CONF_FILE even when empty");
    }
    if ladder_present {
        return Err("batched fixed K5 refuses ATLAS_QWEN4_BATCHED_K5_LADDER");
    }
    if !matches!(gdn_multi, None | Some("0") | Some("1")) {
        return Err("batched fixed K5 requires ATLAS_QWEN4_BATCHED_K5_GDN_MULTI unset, 0 or 1");
    }
    Ok(())
}

fn read_fixed_k5_policy_live() -> std::result::Result<(), String> {
    let mut flags: [Option<String>; FIXED_K5_POLICY_FLAGS.len()] = Default::default();
    for (slot, name) in flags.iter_mut().zip(FIXED_K5_POLICY_FLAGS) {
        *slot = match std::env::var(name) {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(format!("{name} must be valid UTF-8"));
            }
        };
    }
    let borrowed: [Option<&str>; FIXED_K5_POLICY_FLAGS.len()] =
        std::array::from_fn(|i| flags[i].as_deref());
    let file = ["ATLAS_QWEN4_MTP_CONF_FILE", "FLUX_QWEN4_MTP_CONF_FILE"]
        .iter()
        .any(|name| std::env::var_os(name).is_some());
    let ladder = [
        "ATLAS_QWEN4_BATCHED_K5_LADDER",
        "FLUX_QWEN4_BATCHED_K5_LADDER",
    ]
    .iter()
    .any(|name| std::env::var_os(name).is_some());
    let multi = match std::env::var("ATLAS_QWEN4_BATCHED_K5_GDN_MULTI") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err("GDN multi selector must be valid UTF-8".to_owned());
        }
    };
    if let Some(value) = std::env::var_os("FLUX_QWEN4_BATCHED_K5_GDN_MULTI")
        && value != "0"
        && value != "1"
    {
        return Err("FLUX_QWEN4_BATCHED_K5_GDN_MULTI must be unset, 0 or 1".to_owned());
    }
    validate_fixed_k5_policy(&borrowed, file, ladder, multi.as_deref()).map_err(str::to_owned)
}

pub fn require_fixed_k5_policy() -> Result<()> {
    #[cfg(not(test))]
    {
        static CACHED: std::sync::OnceLock<std::result::Result<(), String>> =
            std::sync::OnceLock::new();
        CACHED
            .get_or_init(read_fixed_k5_policy_live)
            .as_ref()
            .copied()
            .map_err(|msg| anyhow::Error::msg(msg.clone()))
    }
    #[cfg(test)]
    {
        read_fixed_k5_policy_live().map_err(anyhow::Error::msg)
    }
}

/// Opt-in adaptive verify width for the batched cohort. With it, the
/// scheduler narrows every sequence's draft chain so one frame of
/// `live * K <= 128` rows verifies the whole cohort. Verify still decides
/// every emitted token. Admitted at C2-C64: past one K2 frame a server runs
/// target-only up to C128.
pub const ADAPTIVE_K_ENV: &str = "ATLAS_QWEN4_BATCHED_ADAPTIVE_K";

/// Rows of the default single-sweep adaptive frame on `<= 64`-SM GPUs (GB10 UMA).
pub const ADAPTIVE_MAX_FRAME_ROWS: usize = 128;
/// Rows of the widest adaptive frame on `> 64`-SM GPUs (e.g. SM120), matching
/// `QWEN4_CROSS_SEQ_MAX_ROWS` (`C=64 x K=4 = 256`, `C=32 x K=8 = 256`).
pub const ADAPTIVE_MAX_FRAME_ROWS_WIDE_SM: usize = crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS;
/// Active SM threshold above which the default adaptive verify frame lifts from
/// `128` to `256` rows. GB10 (`48` SMs) stays strictly at `128` rows.
pub const ADAPTIVE_WIDE_SM_THRESHOLD: u32 = 64;
/// Optional environment override (`2..=256`) for the adaptive verify frame row
/// budget. Unset selects `128` on `<= 64` SMs and `256` on `> 64` SMs.
pub const ADAPTIVE_MAX_ROWS_ENV: &str = "ATLAS_QWEN4_ADAPTIVE_MAX_ROWS";
/// Widest adaptive verify width (`16 x K8 = 128` on GB10, `32 x K8 = 256` on
/// wide-SM GPUs; the QSA journal covers `2..=9`).
pub const ADAPTIVE_MAX_K: usize = 8;
/// Narrowest adaptive verify width: one draft plus the bonus row.
pub const ADAPTIVE_MIN_K: usize = 2;

/// Default adaptive verify frame row budget for `sm_count` streaming
/// multiprocessors: `128` on `<= 64` SMs (GB10 UMA), `256` on `> 64` SMs.
#[must_use]
pub const fn adaptive_max_frame_rows_for_sm(sm_count: u32) -> usize {
    if sm_count > ADAPTIVE_WIDE_SM_THRESHOLD {
        ADAPTIVE_MAX_FRAME_ROWS_WIDE_SM
    } else {
        ADAPTIVE_MAX_FRAME_ROWS
    }
}

/// Strict parser for [`ADAPTIVE_MAX_ROWS_ENV`]: unset selects
/// [`adaptive_max_frame_rows_for_sm`]`(sm_count)`; otherwise accepts an exact
/// decimal integer in `ADAPTIVE_MIN_K..=QWEN4_CROSS_SEQ_MAX_ROWS` (`2..=256`).
pub fn parse_adaptive_max_frame_rows(
    raw: Option<&str>,
    sm_count: u32,
) -> std::result::Result<usize, &'static str> {
    let Some(s) = raw else {
        return Ok(adaptive_max_frame_rows_for_sm(sm_count));
    };
    if s.is_empty() || s != s.trim() {
        return Err("ATLAS_QWEN4_ADAPTIVE_MAX_ROWS must be an integer in 2..=256");
    }
    match s.parse::<usize>() {
        Ok(n) if (ADAPTIVE_MIN_K..=crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS).contains(&n) => Ok(n),
        _ => Err("ATLAS_QWEN4_ADAPTIVE_MAX_ROWS must be an integer in 2..=256"),
    }
}

fn read_adaptive_max_frame_rows_live() -> std::result::Result<usize, String> {
    let sm_count = atlas_core::device::active_sm_count();
    let raw = match std::env::var(ADAPTIVE_MAX_ROWS_ENV) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(format!("{ADAPTIVE_MAX_ROWS_ENV} must be valid UTF-8"));
        }
    };
    parse_adaptive_max_frame_rows(raw.as_deref(), sm_count).map_err(str::to_owned)
}

/// Active adaptive verify frame row budget from [`ADAPTIVE_MAX_ROWS_ENV`] and
/// the active device SM count (`OnceLock`-cached outside unit tests).
///
/// Falls back to [`adaptive_max_frame_rows_for_sm`] with a warning if the env
/// override is malformed.
#[must_use]
pub fn adaptive_max_frame_rows() -> usize {
    let resolve = || match read_adaptive_max_frame_rows_live() {
        Ok(rows) => rows,
        Err(error) => {
            let fallback = adaptive_max_frame_rows_for_sm(atlas_core::device::active_sm_count());
            tracing::warn!("{error}; falling back to {fallback}-row adaptive frame budget");
            fallback
        }
    };
    #[cfg(not(test))]
    {
        static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *CACHED.get_or_init(resolve)
    }
    #[cfg(test)]
    {
        resolve()
    }
}

/// Strict parser: unset or `0` is off, `1` is on, anything else fails closed.
pub fn parse_adaptive_k_selector(raw: Option<&str>) -> std::result::Result<bool, &'static str> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_QWEN4_BATCHED_ADAPTIVE_K must be exact value 0 or 1"),
    }
}

fn read_adaptive_k_selected_live() -> std::result::Result<bool, String> {
    let raw = match std::env::var(ADAPTIVE_K_ENV) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(format!("{ADAPTIVE_K_ENV} must be valid UTF-8"));
        }
    };
    parse_adaptive_k_selector(raw.as_deref()).map_err(str::to_owned)
}

/// Read the adaptive selector from the environment (strict, `OnceLock`-cached
/// outside unit tests like [`batched_k5_selected`]).
pub fn adaptive_k_selected() -> Result<bool> {
    #[cfg(not(test))]
    {
        static CACHED: std::sync::OnceLock<std::result::Result<bool, String>> =
            std::sync::OnceLock::new();
        CACHED
            .get_or_init(read_adaptive_k_selected_live)
            .as_ref()
            .copied()
            .map_err(|msg| anyhow::Error::msg(msg.clone()))
    }
    #[cfg(test)]
    {
        read_adaptive_k_selected_live().map_err(anyhow::Error::msg)
    }
}

/// Verify width for `live` speculating sequences given an explicit single-frame
/// row budget `max_frame_rows` (`128` on GB10 `<= 64` SMs; `256` on `> 64` SMs).
///
/// Clamped to `ADAPTIVE_MIN_K..=ADAPTIVE_MAX_K` (`2..=8`) through `MAX_LANES`
/// (`C128`), and `None` past `MAX_LANES` (`C129+`).
#[must_use]
pub const fn adaptive_verify_width_for_max_rows(
    live: usize,
    max_frame_rows: usize,
) -> Option<usize> {
    if live <= 1 {
        return Some(ADAPTIVE_MAX_K);
    }
    if live > MAX_LANES {
        return None;
    }
    let k = max_frame_rows / live;
    if k < ADAPTIVE_MIN_K {
        Some(ADAPTIVE_MIN_K)
    } else if k > ADAPTIVE_MAX_K {
        Some(ADAPTIVE_MAX_K)
    } else {
        Some(k)
    }
}

/// Verify width for `live` speculating sequences under the default 128-row
/// GB10 ceiling: the widest K in `ADAPTIVE_MIN_K..=ADAPTIVE_MAX_K` whose single
/// frame holds all of them within 128 rows for `C1..=C64`, and `K=2` across two
/// balanced `<=128`-row sub-frames (`130..=256` verify rows/cycle) for
/// `C65..=C128`. `None` past `MAX_LANES` (`C129+`).
///
/// C1-C16 K8, C17-C18 K7, C19-C21 K6, C22-C25 K5, C26-C32 K4, C33-C42 K3,
/// C43-C128 K2, C129+ none.
pub const fn adaptive_verify_width(live: usize) -> Option<usize> {
    adaptive_verify_width_for_max_rows(live, ADAPTIVE_MAX_FRAME_ROWS)
}

/// Optional lower-alpha tuning selector that narrows the single-sweep default
/// ladder relative to [`AdaptiveVerifyProfile::SingleSweepDefault`] (`C1..=C6`:
/// `K8 -> K5`, `C7..=C16`: `K8 -> K4` keeping `M <= 64` for the `m64` tile,
/// `C17..=C25`: `K7..=K5 -> K4`) when set to exact `"1"`.
pub const ADAPTIVE_K4_C7_C16_ENV: &str = "ATLAS_QWEN4_ADAPTIVE_K4_C7_C16";

/// Single-sweep adaptive verify width profile aligned with the `m64`/`m80`/`m128`
/// exact and coalesced NVFP4 tiles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AdaptiveVerifyProfile {
    /// Single-sweep default ladder (`live * K <= 128` through `C64`):
    /// `C1..=C16 => K8` (`M <= 128`), `C17..=C18 => K7` (`M <= 126`),
    /// `C19..=C21 => K6` (`M <= 126`), `C22..=C25 => K5` (`M <= 125`),
    /// `C26..=C32 => K4` (`M <= 128`), `C33..=C42 => K3` (`M <= 126`),
    /// `C43..=C128 => K2` (`M <= 128` per sub-frame).
    #[default]
    SingleSweepDefault,
    /// Lower-alpha workload profile (`ATLAS_QWEN4_ADAPTIVE_K4_C7_C16=1`):
    /// `C1..=C6 => K5`, `C7..=C32 => K4`, `C33..=C42 => K3`, `C43..=C128 => K2`.
    K4C7C16,
}

/// Strict parser for [`ADAPTIVE_K4_C7_C16_ENV`]: unset or `"0"` selects
/// [`AdaptiveVerifyProfile::SingleSweepDefault`], `"1"` selects
/// [`AdaptiveVerifyProfile::K4C7C16`], and any other value fails closed.
pub fn parse_adaptive_k4_c7_c16_selector(
    raw: Option<&str>,
) -> std::result::Result<AdaptiveVerifyProfile, &'static str> {
    match raw {
        None | Some("0") => Ok(AdaptiveVerifyProfile::SingleSweepDefault),
        Some("1") => Ok(AdaptiveVerifyProfile::K4C7C16),
        Some(_) => Err("ATLAS_QWEN4_ADAPTIVE_K4_C7_C16 must be exact value 0 or 1"),
    }
}

fn read_adaptive_verify_profile_live() -> std::result::Result<AdaptiveVerifyProfile, String> {
    let raw = match std::env::var(ADAPTIVE_K4_C7_C16_ENV) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(format!("{ADAPTIVE_K4_C7_C16_ENV} must be valid UTF-8"));
        }
    };
    parse_adaptive_k4_c7_c16_selector(raw.as_deref()).map_err(str::to_owned)
}

/// Read [`ADAPTIVE_K4_C7_C16_ENV`] from the environment (strict,
/// `OnceLock`-cached outside unit tests).
///
/// # Errors
///
/// Returns an error if the environment variable is non-UTF-8 or holds a value
/// other than `"0"` or `"1"`.
pub fn adaptive_verify_profile() -> Result<AdaptiveVerifyProfile> {
    #[cfg(not(test))]
    {
        static CACHED: std::sync::OnceLock<std::result::Result<AdaptiveVerifyProfile, String>> =
            std::sync::OnceLock::new();
        CACHED
            .get_or_init(read_adaptive_verify_profile_live)
            .as_ref()
            .copied()
            .map_err(|msg| anyhow::Error::msg(msg.clone()))
    }
    #[cfg(test)]
    {
        read_adaptive_verify_profile_live().map_err(anyhow::Error::msg)
    }
}

/// Single-sweep adaptive verify width for `live` speculating sequences under
/// `profile` and an explicit `max_frame_rows` row budget.
#[must_use]
pub const fn adaptive_verify_width_for_profile_and_max_rows(
    live: usize,
    profile: AdaptiveVerifyProfile,
    max_frame_rows: usize,
) -> Option<usize> {
    if live > MAX_LANES {
        return None;
    }
    match profile {
        AdaptiveVerifyProfile::SingleSweepDefault => {
            adaptive_verify_width_for_max_rows(live, max_frame_rows)
        }
        AdaptiveVerifyProfile::K4C7C16 => {
            let Some(budget_k) = adaptive_verify_width_for_max_rows(live, max_frame_rows) else {
                return None;
            };
            let profile_k = match live {
                0..=6 => 5,
                7..=32 => 4,
                33..=42 => 3,
                _ => 2,
            };
            Some(if budget_k < profile_k {
                budget_k
            } else {
                profile_k
            })
        }
    }
}

/// Single-sweep adaptive verify width for `live` speculating sequences under
/// `profile` using the default 128-row GB10 budget.
///
/// - [`AdaptiveVerifyProfile::SingleSweepDefault`] — matches
///   [`adaptive_verify_width`] (`C1..=C16 => K8`, `C17..=C18 => K7`,
///   `C19..=C21 => K6`, `C22..=C25 => K5`, `C26..=C32 => K4`,
///   `C33..=C42 => K3`, `C43..=C128 => K2`, `C129+ => None`).
/// - [`AdaptiveVerifyProfile::K4C7C16`] — legacy lower-alpha ceiling
///   (`C1..=C6 => K5`, `C7..=C32 => K4`, `C33..=C42 => K3`, `C43..=C128 => K2`).
#[must_use]
pub const fn adaptive_verify_width_for_profile(
    live: usize,
    profile: AdaptiveVerifyProfile,
) -> Option<usize> {
    adaptive_verify_width_for_profile_and_max_rows(live, profile, ADAPTIVE_MAX_FRAME_ROWS)
}

/// Resolve the single-sweep adaptive verify width using the process environment
/// and active GPU SM count.
///
/// # Errors
///
/// Returns an error if [`ADAPTIVE_K4_C7_C16_ENV`] contains an invalid value.
pub fn adaptive_verify_width_tuned(live: usize) -> Result<Option<usize>> {
    Ok(adaptive_verify_width_for_profile_and_max_rows(
        live,
        adaptive_verify_profile()?,
        adaptive_max_frame_rows(),
    ))
}

/// Enable/disable the online EWMA-acceptance + concurrency-aware GB10 roofline
/// cohort verify-width controller (`optimal_cohort_k`).
///
/// Defaults to enabled (`"1"` or unset); set exact `"0"` for fixed-K5 A/B runs.
pub const COHORT_ADAPTIVE_K_ENV: &str = "ATLAS_QWEN4_COHORT_ADAPTIVE_K";

/// Strict parser for [`COHORT_ADAPTIVE_K_ENV`]: unset or `"1"` enables the
/// roofline cohort controller, `"0"` disables it, and any other value fails
/// closed.
pub fn parse_cohort_adaptive_k_selector(
    raw: Option<&str>,
) -> std::result::Result<bool, &'static str> {
    match raw {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(_) => Err("ATLAS_QWEN4_COHORT_ADAPTIVE_K must be exact value 0 or 1"),
    }
}

fn read_cohort_adaptive_k_selected_live() -> std::result::Result<bool, String> {
    let raw = match std::env::var(COHORT_ADAPTIVE_K_ENV) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(format!("{COHORT_ADAPTIVE_K_ENV} must be valid UTF-8"));
        }
    };
    parse_cohort_adaptive_k_selector(raw.as_deref()).map_err(str::to_owned)
}

/// Read [`COHORT_ADAPTIVE_K_ENV`] from the environment (strict,
/// `OnceLock`-cached outside unit tests).
///
/// # Errors
///
/// Returns an error if the environment variable is non-UTF-8 or holds a value
/// other than `"0"` or `"1"`.
pub fn cohort_adaptive_k_selected() -> Result<bool> {
    #[cfg(not(test))]
    {
        static CACHED: std::sync::OnceLock<std::result::Result<bool, String>> =
            std::sync::OnceLock::new();
        CACHED
            .get_or_init(read_cohort_adaptive_k_selected_live)
            .as_ref()
            .copied()
            .map_err(|msg| anyhow::Error::msg(msg.clone()))
    }
    #[cfg(test)]
    {
        read_cohort_adaptive_k_selected_live().map_err(anyhow::Error::msg)
    }
}

/// Maximum-likelihood estimate of per-step conditional draft acceptance
/// probability `p_step` from a settled batched verify frame.
///
/// For truncated geometric acceptance of width `d = K - 1`, each sequence that
/// accepts fewer than `d` drafts contributes one rejection event while full
/// accepts contribute zero rejections, making
/// `total_accepted / (total_accepted + rejected_seqs)` invariant to `K`.
#[must_use]
pub fn estimate_step_accept_prob(total_accepted: usize, rejected_seqs: usize) -> Option<f32> {
    let denom = total_accepted.saturating_add(rejected_seqs);
    if denom == 0 {
        return None;
    }
    Some((total_accepted as f32 / denom as f32).clamp(0.0, 1.0))
}

/// Recalibrated post-Wave-4.5 GB10 step latency roofline (`verify_ms + draft_ms`)
/// for `live` concurrent sequences at verify width `k`.
///
/// Models Qwen3.8-Flash-Next NVFP4 (48 layers, 512 routed experts, top-10)
/// with empirical coalesced-NVFP4 / cached-FP8 dense and HC routing, 1-launch
/// TC batched routing, `v28_m64_ts4e` Tensor-Core MoE (`rows >= 16`), fused
/// GDN verify, draft-ahead PLE caching, and coalesced batched MTP drafting:
/// - Shared dense/HC/router/LM-head weights + launch floor (`14.5 ms`) plus
///   per-sequence GDN/QSA/KV state traffic (`2.3 ms/seq`, invariant to `k`)
/// - Zipf-concentrated unique MoE experts over an effective hot pool of `224`
///   experts with within-sequence temporal locality (`rho = 0.64`) and a
///   two-slope marginal cost modeling the scalar `_t_lanes` (`0.68 ms/expert`)
///   to Tensor-Core `v28_m64_ts4e` (`0.22 ms/expert`) transition around `64`
///   unique experts (`rows >= 16`)
/// - `k - 1` coalesced batched MTP proposal steps (`0.95 ms + 0.15 ms/seq`,
///   matching the measured `1.21..3.35 ms/step` across `c = 2..16`) plus
///   per-row MoE/GDN/dense compute (`0.28 ms/row`)
#[must_use]
pub fn cohort_step_cost_ms(live: usize, k: usize) -> f64 {
    const EFFECTIVE_EXPERT_POOL: f64 = 224.0;
    const TOP_K: f64 = 10.0;
    const TEMPORAL_LOCALITY_RHO: f64 = 0.64;
    const TC_MOE_EXPERT_KNEE: f64 = 64.0;
    const SCALAR_EXPERT_48L_MS: f64 = 0.68;
    const TC_EXPERT_48L_MS: f64 = 0.22;
    const DENSE_FIXED_MS: f64 = 14.5;
    const STATE_KV_PER_SEQ_MS: f64 = 2.3;
    const MTP_STEP_BASE_MS: f64 = 0.95;
    const MTP_STEP_PER_SEQ_MS: f64 = 0.15;
    const ROW_COMPUTE_MS: f64 = 0.28;

    let c = live.max(1) as f64;
    let kf = k.max(1) as f64;
    let miss_prob = 1.0 - (TOP_K / EFFECTIVE_EXPERT_POOL);
    let m = c * kf;
    let m_eff = c * (1.0 + TEMPORAL_LOCALITY_RHO * (kf - 1.0));
    let unique_experts = EFFECTIVE_EXPERT_POOL * (1.0 - miss_prob.powf(m_eff));
    let moe_ms = SCALAR_EXPERT_48L_MS * unique_experts.min(TC_MOE_EXPERT_KNEE)
        + TC_EXPERT_48L_MS * (unique_experts - TC_MOE_EXPERT_KNEE).max(0.0);
    let dense_ms = DENSE_FIXED_MS + STATE_KV_PER_SEQ_MS * c;
    let mtp_ms = (kf - 1.0) * (MTP_STEP_BASE_MS + MTP_STEP_PER_SEQ_MS * c);
    let row_ms = ROW_COMPUTE_MS * m;
    dense_ms + moe_ms + mtp_ms + row_ms
}

/// Expected emitted tokens per millisecond per sequence at concurrency `live`,
/// per-step acceptance probability `ewma_p_step`, and verify width `k`.
///
/// Returns `0.0` for non-finite `ewma_p_step` or `k == 0`.
#[must_use]
pub fn cohort_expected_rate(live: usize, ewma_p_step: f32, k: usize) -> f64 {
    if !ewma_p_step.is_finite() || k == 0 {
        return 0.0;
    }
    let p = f64::from(ewma_p_step.clamp(0.0, 1.0));
    let kf = k as f64;
    let expected_tokens = if (1.0 - p).abs() < 1e-4 {
        kf
    } else {
        (1.0 - p.powi(i32::try_from(k).unwrap_or(i32::MAX))) / (1.0 - p)
    };
    expected_tokens / cohort_step_cost_ms(live, k)
}

/// Roofline cost model choosing the throughput-optimal batched verify width `K`
/// in `ADAPTIVE_MIN_K..=upper` for `live` concurrent sequences at online
/// per-step acceptance probability `ewma_p_step`, respecting both `max_k` and
/// the single-frame `max_frame_rows` budget (`128` on GB10, `256` on `> 64`-SM
/// GPUs).
#[must_use]
pub fn optimal_cohort_k_for_max_rows(
    live: usize,
    ewma_p_step: f32,
    max_k: usize,
    max_frame_rows: usize,
) -> usize {
    let frame_cap =
        adaptive_verify_width_for_max_rows(live, max_frame_rows).unwrap_or(ADAPTIVE_MIN_K);
    let upper = max_k.min(frame_cap).clamp(ADAPTIVE_MIN_K, ADAPTIVE_MAX_K);
    if live <= 1 || upper <= ADAPTIVE_MIN_K || !ewma_p_step.is_finite() {
        return upper;
    }

    let mut best_k = ADAPTIVE_MIN_K;
    let mut best_rate = f64::NEG_INFINITY;
    for k in ADAPTIVE_MIN_K..=upper {
        let rate = cohort_expected_rate(live, ewma_p_step, k);
        if rate > best_rate {
            best_rate = rate;
            best_k = k;
        }
    }
    best_k
}

/// GB10 roofline cost model choosing the throughput-optimal batched verify
/// width `K` in `ADAPTIVE_MIN_K..=upper` for `live` concurrent sequences at
/// online per-step acceptance probability `ewma_p_step`, where `upper` respects
/// both `max_k` and the single-frame `live * K <= 128` ceiling
/// ([`adaptive_verify_width`]).
///
/// Maximizes [`cohort_expected_rate`] across `ADAPTIVE_MIN_K..=upper` (batched
/// cohorts admit up to `K <= ADAPTIVE_MAX_K = 8`, matching `16 x K8 = 128`
/// rows and the `2..=9` QSA journal): collapses to `K = 2` on very low
/// acceptance, selects `K = 4..6` on typical non-greedy prose (`p ~ 0.55`) as
/// `TC_MOE` and expert saturation flatten the marginal row slope at
/// `c = 4..16`, and unlocks `K = 6..8` on high-acceptance thinking/code
/// (`p = 0.75..0.85`).
#[must_use]
pub fn optimal_cohort_k(live: usize, ewma_p_step: f32, max_k: usize) -> usize {
    optimal_cohort_k_for_max_rows(live, ewma_p_step, max_k, ADAPTIVE_MAX_FRAME_ROWS)
}

/// Facts beyond the fixed-K5 policy that K2..K8 cohort frames depend on.
/// The fixed-K5 policy (cohort-safe MTP_CONF aliases, no conf file, no
/// ladder) still applies: chain width comes only from the scheduler's
/// deterministic rule, never from draft confidence.
pub fn validate_adaptive_k_policy(
    batched: bool,
    hybrid_all_k: bool,
    qsa_journal_all_k_off: bool,
) -> std::result::Result<(), &'static str> {
    if !batched {
        return Err("ATLAS_QWEN4_BATCHED_ADAPTIVE_K requires ATLAS_QWEN4_BATCHED_K5=1");
    }
    if !hybrid_all_k {
        return Err(
            "ATLAS_QWEN4_BATCHED_ADAPTIVE_K requires the exact K2..K8 hybrid rows (ATLAS_QWEN4_HYBRID_ALL_K=1 or ATLAS_QWEN4_VERIFY_FAST_HOST=1)",
        );
    }
    if qsa_journal_all_k_off {
        return Err(
            "ATLAS_QWEN4_BATCHED_ADAPTIVE_K refuses ATLAS_QWEN4_QSA_JOURNAL_ALL_K=0: K2..K8 QSA writes must be journaled",
        );
    }
    Ok(())
}

fn read_adaptive_k_policy_live() -> std::result::Result<(), String> {
    let on = |name: &str| std::env::var(name).ok().as_deref() == Some("1");
    let hybrid_all_k = on("ATLAS_QWEN4_HYBRID_ALL_K") || on("ATLAS_QWEN4_VERIFY_FAST_HOST");
    let journal_off = std::env::var("ATLAS_QWEN4_QSA_JOURNAL_ALL_K")
        .ok()
        .as_deref()
        == Some("0");
    let batched = read_batched_k5_selected_live()?;
    validate_adaptive_k_policy(batched, hybrid_all_k, journal_off).map_err(str::to_owned)
}

pub fn require_adaptive_k_policy() -> Result<()> {
    #[cfg(not(test))]
    {
        static CACHED: std::sync::OnceLock<std::result::Result<(), String>> =
            std::sync::OnceLock::new();
        CACHED
            .get_or_init(read_adaptive_k_policy_live)
            .as_ref()
            .copied()
            .map_err(|msg| anyhow::Error::msg(msg.clone()))
    }
    #[cfg(test)]
    {
        read_adaptive_k_policy_live().map_err(anyhow::Error::msg)
    }
}

/// Lane count a model with `max_batch_size` slots needs.
pub fn lanes_for(batched: bool, max_batch_size: usize) -> Result<usize> {
    if !batched || max_batch_size <= 1 {
        return Ok(1);
    }
    ensure!(
        max_batch_size <= MAX_LANES,
        "{BATCHED_K5_ENV} supports at most {MAX_LANES} verify lanes, got {max_batch_size}"
    );
    Ok(max_batch_size)
}

/// A model's claim on the process lane count. The owning model keeps it for
/// its lifetime; dropping it (model unload or swap) resets the count to 1
/// unless a newer model registered since, so a swap never inherits a stale
/// lane count and an older model's unload never clobbers a newer one.
#[must_use = "the lane count resets when the registration drops"]
#[derive(Debug)]
pub struct LanesRegistration {
    packed: u64,
}

impl Drop for LanesRegistration {
    fn drop(&mut self) {
        let generation = self.packed >> 8;
        let reset = (generation << 8) | 1;
        // Only the current registration resets; a newer one stays in force.
        let _ = STATE.compare_exchange(self.packed, reset, Ordering::AcqRel, Ordering::Acquire);
    }
}

/// Register the lane count for a model being constructed. Must run before
/// that model's first QSA side cache or GDN retain buffer is allocated
/// (both size themselves from [`lanes`] on first use and record the count
/// they were built with). A later registration replaces this one.
pub fn configure_lanes(lanes: usize) -> Result<LanesRegistration> {
    ensure!(
        (1..=MAX_LANES).contains(&lanes),
        "verify lane count {lanes} outside 1..={MAX_LANES}"
    );
    let mut current = STATE.load(Ordering::Acquire);
    loop {
        let packed = (((current >> 8) + 1) << 8) | lanes as u64;
        match STATE.compare_exchange_weak(current, packed, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                if lanes > 1 {
                    tracing::info!("QWEN4_BATCHED_K5_LANES_CONFIGURED lanes={lanes}");
                }
                return Ok(LanesRegistration { packed });
            }
            Err(observed) => current = observed,
        }
    }
}

/// Lane count of the current registration (1 when none).
pub fn lanes() -> usize {
    (STATE.load(Ordering::Acquire) & 0xFF) as usize
}

/// Lane selected for the current thread (0 outside a guard).
pub fn current_lane() -> usize {
    CURRENT.with(|cell| cell.get().unwrap_or(0))
}

/// Whether a [`LaneGuard`] is active on this thread.
pub fn guard_active() -> bool {
    CURRENT.with(|cell| cell.get().is_some())
}

/// Lane owned by a physical SSM slot under the configured lane count.
pub fn lane_for_slot(slot_idx: usize) -> Result<usize> {
    checked_lane_for_slot(slot_idx, lanes()).map_err(anyhow::Error::msg)
}

pub(crate) fn checked_lane_for_slot(
    slot_idx: usize,
    lanes: usize,
) -> std::result::Result<usize, &'static str> {
    if lanes <= 1 {
        return Ok(0);
    }
    if slot_idx >= lanes {
        return Err("sequence slot has no verify lane (slot >= configured lanes)");
    }
    Ok(slot_idx)
}

/// Byte offset of `lane` inside a lane-major allocation of `lane_bytes`.
pub fn lane_offset(lane: usize, lane_bytes: usize) -> Result<usize> {
    lane.checked_mul(lane_bytes)
        .context("verify lane byte offset overflow")
}

/// RAII selection of the verify lane for every transaction-state access made
/// on this thread until drop. Guards nest; drop restores the outer lane.
#[must_use = "the lane is released when the guard is dropped"]
pub struct LaneGuard {
    previous: Option<usize>,
    /// The guard is thread-local state; it must drop on the thread that
    /// entered it (the scheduler thread that launches the frame).
    #[cfg(debug_assertions)]
    thread: std::thread::ThreadId,
}

impl LaneGuard {
    pub fn enter(lane: usize) -> Result<Self> {
        let lanes = lanes();
        ensure!(
            lane < lanes,
            "verify lane {lane} outside the configured {lanes} lanes"
        );
        let previous = CURRENT.with(|cell| cell.replace(Some(lane)));
        Ok(Self {
            previous,
            #[cfg(debug_assertions)]
            thread: std::thread::current().id(),
        })
    }

    /// Enter the lane owned by `slot_idx`.
    pub fn enter_for_slot(slot_idx: usize) -> Result<Self> {
        Self::enter(lane_for_slot(slot_idx)?)
    }
}

impl Drop for LaneGuard {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            self.thread,
            std::thread::current().id(),
            "LaneGuard dropped on a different thread than the one that launched the frame"
        );
        let previous = self.previous;
        CURRENT.with(|cell| cell.set(previous));
    }
}

/// Startup-ledger bytes for verify lanes beyond what the existing single-lane
/// accounting already reserves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LaneReserve {
    /// Extra QSA journal lanes (lanes - 1) across every attention layer. Lane
    /// zero is already inside `qwen4_qsa::checked_side_cache_bytes`.
    pub qsa_journal_bytes: usize,
    /// Every GDN lazy-commit retain lane across every SSM layer. The retain
    /// buffer is lazily allocated and was not ledgered for C1, so under the
    /// selector all lanes are reserved explicitly.
    pub gdn_retain_bytes: usize,
}

impl LaneReserve {
    pub fn total(&self) -> Result<usize> {
        self.qsa_journal_bytes
            .checked_add(self.gdn_retain_bytes)
            .context("verify lane reserve total overflow")
    }
}

pub fn checked_lane_reserve(
    lanes: usize,
    num_attention_layers: usize,
    qsa_journal_lane_bytes: usize,
    num_ssm_layers: usize,
    gdn_retain_lane_bytes: Option<usize>,
) -> Result<LaneReserve> {
    if lanes <= 1 {
        return Ok(LaneReserve::default());
    }
    let extra_lanes = lanes - 1;
    let qsa_journal_bytes = extra_lanes
        .checked_mul(qsa_journal_lane_bytes)
        .and_then(|bytes| bytes.checked_mul(num_attention_layers))
        .context("QSA journal lane reserve overflow")?;
    let gdn_retain_bytes = match gdn_retain_lane_bytes {
        Some(lane_bytes) => lanes
            .checked_mul(lane_bytes)
            .and_then(|bytes| bytes.checked_mul(num_ssm_layers))
            .context("GDN retain lane reserve overflow")?,
        None => 0,
    };
    Ok(LaneReserve {
        qsa_journal_bytes,
        gdn_retain_bytes,
    })
}

#[cfg(test)]
thread_local! {
    static HOLDS_TEST_ENV_LOCK: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
#[must_use = "hold the returned guard until all ScopedEnvVar instances are dropped"]
pub(crate) struct TestEnvGuard {
    _guard: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl Drop for TestEnvGuard {
    fn drop(&mut self) {
        HOLDS_TEST_ENV_LOCK.with(|held| held.set(false));
    }
}

#[cfg(test)]
pub(crate) fn test_env_lock() -> TestEnvGuard {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    HOLDS_TEST_ENV_LOCK.with(|held| held.set(true));
    TestEnvGuard { _guard: guard }
}

#[cfg(test)]
fn assert_holds_test_env_lock(op: &str, key: &str) {
    assert!(
        HOLDS_TEST_ENV_LOCK.with(Cell::get),
        "ScopedEnvVar::{op}({key:?}) requires the current thread to hold test_env_lock()"
    );
}

#[cfg(test)]
#[must_use = "bind ScopedEnvVar so the environment override stays active for the scope"]
pub(crate) struct ScopedEnvVar {
    key: &'static str,
    prev: Option<std::ffi::OsString>,
}

#[cfg(test)]
impl ScopedEnvVar {
    pub(crate) fn set(key: &'static str, val: &str) -> Self {
        assert_holds_test_env_lock("set", key);
        let prev = std::env::var_os(key);
        // SAFETY: POSIX `setenv` requires process-wide exclusion against any
        // concurrent thread reading or writing any environment variable (not
        // only `key`). The assertion above enforces that this thread holds
        // `test_env_lock()`; callers rely on all env-accessing tests in the
        // crate test binary coordinating through this shared lock (or running
        // single-threaded).
        unsafe { std::env::set_var(key, val) };
        Self { key, prev }
    }

    pub(crate) fn remove(key: &'static str) -> Self {
        assert_holds_test_env_lock("remove", key);
        let prev = std::env::var_os(key);
        // SAFETY: POSIX `unsetenv` requires process-wide exclusion against any
        // concurrent thread reading or writing any environment variable (not
        // only `key`). The assertion above enforces that this thread holds
        // `test_env_lock()`; callers rely on all env-accessing tests in the
        // crate test binary coordinating through this shared lock (or running
        // single-threaded).
        unsafe { std::env::remove_var(key) };
        Self { key, prev }
    }
}

#[cfg(test)]
impl Drop for ScopedEnvVar {
    fn drop(&mut self) {
        assert_holds_test_env_lock("drop", self.key);
        match self.prev.as_ref() {
            Some(val) => {
                // SAFETY: POSIX `setenv` requires process-wide exclusion against
                // any concurrent thread reading or writing any environment
                // variable. The assertion above verifies this thread still holds
                // `test_env_lock()` while restoring `self.key`.
                unsafe { std::env::set_var(self.key, val) };
            }
            None => {
                // SAFETY: POSIX `unsetenv` requires process-wide exclusion
                // against any concurrent thread reading or writing any
                // environment variable. The assertion above verifies this
                // thread still holds `test_env_lock()` while restoring
                // `self.key`.
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_k5_policy_requires_vocab_prune_zero_and_validates_cohort_safe_conf() {
        let valid = [Some("0"); 6];
        assert!(validate_fixed_k5_policy(&valid, false, false, None).is_ok());
        // Single-sequence MTP_CONF / MTP_CONF_EARLY passthrough is admitted at C>1
        // because propose_batch (b >= 2) ignores them via BatchedCohort.
        let with_single_seq_conf = [Some("0.6"), None, Some("1"), None, Some("0"), Some("0")];
        assert!(validate_fixed_k5_policy(&with_single_seq_conf, false, false, None).is_ok());
        // Hostile MTP_CONF / MTP_CONF_EARLY values fail closed.
        for index in 0..2 {
            for bad in [Some(""), Some(" 0"), Some("-0.1"), Some("1.5"), Some("nan")] {
                let mut flags = valid;
                flags[index] = bad;
                assert!(validate_fixed_k5_policy(&flags, false, false, None).is_err());
            }
        }
        for index in 2..4 {
            for bad in [Some(""), Some(" 0"), Some("2"), Some("true")] {
                let mut flags = valid;
                flags[index] = bad;
                assert!(validate_fixed_k5_policy(&flags, false, false, None).is_err());
            }
        }
        // MTP_VOCAB_PRUNE aliases MUST remain exact "0".
        for index in 4..6 {
            for bad in [None, Some(""), Some("1"), Some("0.0"), Some(" 0")] {
                let mut flags = valid;
                flags[index] = bad;
                assert!(validate_fixed_k5_policy(&flags, false, false, None).is_err());
            }
        }
    }

    #[test]
    fn fixed_k5_policy_refuses_mutable_files_ladders_and_invalid_multi_kernel() {
        let valid = [Some("0"); 6];
        assert!(validate_fixed_k5_policy(&valid, true, false, None).is_err());
        assert!(validate_fixed_k5_policy(&valid, false, true, None).is_err());
        assert!(validate_fixed_k5_policy(&valid, false, false, Some("2")).is_err());
        assert!(validate_fixed_k5_policy(&valid, false, false, Some("1")).is_ok());
        assert!(validate_fixed_k5_policy(&valid, false, false, Some("0")).is_ok());
    }

    #[test]
    fn adaptive_selector_is_strict_and_default_off() {
        assert_eq!(parse_adaptive_k_selector(None), Ok(false));
        assert_eq!(parse_adaptive_k_selector(Some("0")), Ok(false));
        assert_eq!(parse_adaptive_k_selector(Some("1")), Ok(true));
        for hostile in ["", "true", "01", "2", " 1", "1 "] {
            assert!(
                parse_adaptive_k_selector(Some(hostile)).is_err(),
                "{hostile:?}"
            );
        }
    }

    #[test]
    fn adaptive_width_follows_the_documented_ladder() {
        let expect = |c: usize| match c {
            0..=16 => Some(8),
            17..=18 => Some(7),
            19..=21 => Some(6),
            22..=25 => Some(5),
            26..=32 => Some(4),
            33..=42 => Some(3),
            43..=128 => Some(2),
            _ => None,
        };
        for c in 0..=160 {
            assert_eq!(adaptive_verify_width(c), expect(c), "C{c}");
        }
    }

    #[test]
    fn adaptive_width_fits_one_frame_to_c64_and_two_balanced_frames_to_c128() {
        for c in 1..=160usize {
            match adaptive_verify_width(c) {
                Some(k) if c <= 64 => {
                    assert!((ADAPTIVE_MIN_K..=ADAPTIVE_MAX_K).contains(&k), "C{c} K{k}");
                    assert!(c * k <= ADAPTIVE_MAX_FRAME_ROWS, "C{c} K{k}");
                    assert!(
                        k == ADAPTIVE_MAX_K || c * (k + 1) > ADAPTIVE_MAX_FRAME_ROWS,
                        "C{c} K{k}"
                    );
                }
                Some(k) => {
                    assert!((65..=MAX_LANES).contains(&c), "C{c}");
                    assert_eq!(k, ADAPTIVE_MIN_K, "C{c}");
                    let sub0 = c.div_ceil(2);
                    let sub1 = c / 2;
                    assert!(sub0 >= 2 && sub1 >= 2, "C{c}");
                    assert!(sub0 * k <= ADAPTIVE_MAX_FRAME_ROWS, "C{c} sub0={sub0}");
                    assert!(sub1 * k <= ADAPTIVE_MAX_FRAME_ROWS, "C{c} sub1={sub1}");
                }
                None => assert!(c > MAX_LANES, "C{c}"),
            }
        }
    }

    #[test]
    fn adaptive_policy_needs_batching_all_k_rows_and_the_journal() {
        assert!(validate_adaptive_k_policy(true, true, false).is_ok());
        assert!(validate_adaptive_k_policy(false, true, false).is_err());
        assert!(validate_adaptive_k_policy(true, false, false).is_err());
        assert!(validate_adaptive_k_policy(true, true, true).is_err());
    }

    #[test]
    fn adaptive_k4_c7_c16_selector_is_strict_and_default_single_sweep() {
        assert_eq!(
            parse_adaptive_k4_c7_c16_selector(None),
            Ok(AdaptiveVerifyProfile::SingleSweepDefault)
        );
        assert_eq!(
            parse_adaptive_k4_c7_c16_selector(Some("0")),
            Ok(AdaptiveVerifyProfile::SingleSweepDefault)
        );
        assert_eq!(
            parse_adaptive_k4_c7_c16_selector(Some("1")),
            Ok(AdaptiveVerifyProfile::K4C7C16)
        );
        for hostile in ["", "true", "01", "2", " 1", "1 "] {
            assert!(
                parse_adaptive_k4_c7_c16_selector(Some(hostile)).is_err(),
                "{hostile:?}"
            );
        }
    }

    #[test]
    fn single_sweep_profile_ladders_fit_one_frame_to_c64_and_two_to_c128() {
        let expect_default = |c: usize| match c {
            0..=16 => Some(8),
            17..=18 => Some(7),
            19..=21 => Some(6),
            22..=25 => Some(5),
            26..=32 => Some(4),
            33..=42 => Some(3),
            43..=128 => Some(2),
            _ => None,
        };
        let expect_k4_c7_c16 = |c: usize| match c {
            0..=6 => Some(5),
            7..=32 => Some(4),
            33..=42 => Some(3),
            43..=128 => Some(2),
            _ => None,
        };
        for c in 0..=160 {
            let def =
                adaptive_verify_width_for_profile(c, AdaptiveVerifyProfile::SingleSweepDefault);
            let k4 = adaptive_verify_width_for_profile(c, AdaptiveVerifyProfile::K4C7C16);
            assert_eq!(def, expect_default(c), "default C{c}");
            assert_eq!(k4, expect_k4_c7_c16(c), "k4_c7_c16 C{c}");
            if let Some(k) = def
                && (1..=64).contains(&c)
            {
                assert!(c * k <= ADAPTIVE_MAX_FRAME_ROWS, "default C{c} K{k}");
            }
            if let Some(k) = k4
                && (7..=16).contains(&c)
            {
                assert!(c * k <= 64, "k4_c7_c16 C{c} K{k} fits m64");
            }
        }
    }

    #[test]
    fn selector_is_strict_and_default_off() {
        assert_eq!(parse_batched_k5_selector(None), Ok(false));
        assert_eq!(parse_batched_k5_selector(Some("0")), Ok(false));
        assert_eq!(parse_batched_k5_selector(Some("1")), Ok(true));
        for hostile in ["", "true", "01", "2", " 1", "1 "] {
            assert!(
                parse_batched_k5_selector(Some(hostile)).is_err(),
                "{hostile:?}"
            );
        }
    }

    #[test]
    fn lane_count_is_one_unless_batched_and_concurrent() {
        assert_eq!(lanes_for(false, 8).unwrap(), 1);
        assert_eq!(lanes_for(true, 1).unwrap(), 1);
        assert_eq!(lanes_for(true, 0).unwrap(), 1);
        assert_eq!(lanes_for(true, 8).unwrap(), 8);
        assert_eq!(lanes_for(true, MAX_LANES).unwrap(), MAX_LANES);
        assert!(lanes_for(true, MAX_LANES + 1).is_err());
    }

    #[test]
    fn single_lane_maps_every_slot_to_lane_zero() {
        for slot in [0, 1, 7, 127, usize::MAX] {
            assert_eq!(checked_lane_for_slot(slot, 1), Ok(0));
        }
    }

    #[test]
    fn multi_lane_maps_slot_to_its_own_lane_and_rejects_overflow() {
        for slot in 0..8 {
            assert_eq!(checked_lane_for_slot(slot, 8), Ok(slot));
        }
        assert!(checked_lane_for_slot(8, 8).is_err());
    }

    /// One test owns every use of the process-wide lane count in this
    /// module, so parallel tests cannot observe each other's registrations.
    #[test]
    fn registration_scopes_the_lane_count_and_guards_follow_it() {
        let _lock = test_env_lock();
        guard_defaults_to_lane_zero_and_restores_on_drop();
        assert_eq!(lanes(), 1);
        let first = configure_lanes(4).unwrap();
        assert_eq!(lanes(), 4);
        let guard = LaneGuard::enter_for_slot(3).unwrap();
        assert_eq!(current_lane(), 3);
        drop(guard);
        assert!(
            LaneGuard::enter_for_slot(4).is_err(),
            "slot 4 has no lane at C4"
        );
        // A swap: the new model registers before the old one unloads.
        let second = configure_lanes(2).unwrap();
        assert_eq!(lanes(), 2);
        drop(first);
        assert_eq!(lanes(), 2, "an older unload never resets a newer model");
        drop(second);
        assert_eq!(lanes(), 1, "the current model's unload resets to one lane");
        assert!(configure_lanes(0).is_err());
        assert!(configure_lanes(MAX_LANES + 1).is_err());
        assert_eq!(lanes(), 1);
    }

    fn guard_defaults_to_lane_zero_and_restores_on_drop() {
        // The process lane count is 1 here, so only lane 0 is legal.
        assert_eq!(current_lane(), 0);
        assert!(!guard_active());
        {
            let _outer = LaneGuard::enter(0).unwrap();
            assert!(guard_active());
            assert_eq!(current_lane(), 0);
            {
                let _inner = LaneGuard::enter(0).unwrap();
                assert!(guard_active());
            }
            assert!(guard_active(), "inner drop restores the outer guard");
        }
        assert!(!guard_active());
        assert!(LaneGuard::enter(1).is_err(), "lane 1 does not exist at C1");
    }

    #[test]
    fn lane_offsets_are_lane_major_and_checked() {
        assert_eq!(lane_offset(0, 4_096).unwrap(), 0);
        assert_eq!(lane_offset(3, 4_096).unwrap(), 12_288);
        assert!(lane_offset(usize::MAX, 2).is_err());
    }

    #[test]
    fn reserve_is_zero_for_one_lane() {
        let reserve = checked_lane_reserve(1, 12, 4_756, 36, Some(1_054_720)).unwrap();
        assert_eq!(reserve, LaneReserve::default());
        assert_eq!(reserve.total().unwrap(), 0);
    }

    #[test]
    fn reserve_counts_extra_journal_lanes_and_every_retain_lane() {
        // Flash-Next: 12 attention layers, 36 GDN layers.
        let reserve = checked_lane_reserve(8, 12, 4_756, 36, Some(1_054_720)).unwrap();
        assert_eq!(reserve.qsa_journal_bytes, 7 * 12 * 4_756);
        assert_eq!(reserve.gdn_retain_bytes, 8 * 36 * 1_054_720);
        assert_eq!(
            reserve.total().unwrap(),
            7 * 12 * 4_756 + 8 * 36 * 1_054_720
        );
        let no_lazy = checked_lane_reserve(8, 12, 4_756, 36, None).unwrap();
        assert_eq!(no_lazy.gdn_retain_bytes, 0);
    }

    #[test]
    fn reserve_overflow_fails_closed() {
        assert!(checked_lane_reserve(usize::MAX, 2, usize::MAX, 1, None).is_err());
        assert!(checked_lane_reserve(2, 1, 1, usize::MAX, Some(usize::MAX)).is_err());
    }

    #[test]
    fn cohort_adaptive_k_selector_defaults_on_and_fails_closed() {
        assert!(parse_cohort_adaptive_k_selector(None).unwrap());
        assert!(parse_cohort_adaptive_k_selector(Some("1")).unwrap());
        assert!(!parse_cohort_adaptive_k_selector(Some("0")).unwrap());
        for bad in ["", "true", "false", "2", " 1 "] {
            assert!(
                parse_cohort_adaptive_k_selector(Some(bad)).is_err(),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn estimate_step_accept_prob_is_k_invariant_and_bounded() {
        assert_eq!(estimate_step_accept_prob(0, 0), None);
        assert_eq!(estimate_step_accept_prob(16, 0), Some(1.0));
        assert_eq!(estimate_step_accept_prob(0, 16), Some(0.0));
        // For p = 0.75 across any K, E[accepted] / (E[accepted] + E[rejected]) == 0.75.
        let p = estimate_step_accept_prob(30, 10).unwrap();
        assert!((p - 0.75).abs() < 1e-6, "got {p}");
    }

    #[test]
    fn cohort_step_cost_ms_tracks_gb10_roofline_and_moe_saturation() {
        // Monotonic in both concurrency `c` and verify width `k` across K2..=K8.
        for c in [2usize, 4, 8, 16, 24, 32, 64] {
            for k in ADAPTIVE_MIN_K..ADAPTIVE_MAX_K {
                assert!(
                    cohort_step_cost_ms(c, k) < cohort_step_cost_ms(c, k + 1),
                    "cost must grow with k at c={c} k={k}"
                );
            }
        }
        for k in ADAPTIVE_MIN_K..=ADAPTIVE_MAX_K {
            for pair in [2usize, 4, 8, 16, 24, 32, 64].windows(2) {
                assert!(
                    cohort_step_cost_ms(pair[0], k) < cohort_step_cost_ms(pair[1], k),
                    "cost must grow with c at k={k} ({} -> {})",
                    pair[0],
                    pair[1]
                );
            }
        }

        // Matches the post-Wave-4.5 calibrated GB10 step-latency roofline (verify + draft).
        let targets = [
            (2usize, [42.7_f64, 51.9, 60.7, 69.2, 73.9, 77.6, 81.2]),
            (4usize, [66.9_f64, 76.4, 82.6, 88.5, 94.0, 99.2, 104.1]),
            (8usize, [91.2_f64, 101.2, 110.1, 118.0, 125.2, 131.8, 137.9]),
            (
                16usize,
                [127.5_f64, 140.8, 152.2, 162.2, 171.4, 180.0, 188.4],
            ),
        ];
        for (c, expected_ms) in targets {
            for (idx, &target) in expected_ms.iter().enumerate() {
                let k = ADAPTIVE_MIN_K + idx;
                let predicted = cohort_step_cost_ms(c, k);
                assert!(
                    (predicted - target).abs() <= 1.0,
                    "c={c} k={k}: predicted {predicted:.2}ms vs target {target:.1}ms"
                );
            }
        }

        // Real GB10 K3 -> K5 and K5 -> K8 slopes reflect MoE saturation plus
        // per-row compute (0.28 ms/row) and coalesced MTP drafting (0.95 + 0.15*c ms/step).
        for c in [4usize, 8, 16] {
            let k5_over_k3 = cohort_step_cost_ms(c, 5) / cohort_step_cost_ms(c, 3);
            assert!(
                (1.12..=1.20).contains(&k5_over_k3),
                "expected c={c} K5/K3 in 1.12..=1.20, got {k5_over_k3:.3}"
            );
        }
        let c16_k8_over_k5 = cohort_step_cost_ms(16, 8) / cohort_step_cost_ms(16, 5);
        assert!(
            (1.12..=1.20).contains(&c16_k8_over_k5),
            "expected c=16 K8/K5 in 1.12..=1.20, got {c16_k8_over_k5:.3}"
        );
        assert_eq!(cohort_expected_rate(8, f32::NAN, 8), 0.0);
        assert_eq!(cohort_expected_rate(8, 0.80, 0), 0.0);
        assert!(cohort_expected_rate(8, 0.85, 8) > cohort_expected_rate(8, 0.55, 8));
    }

    #[test]
    fn optimal_cohort_k_narrows_low_acceptance_and_unlocks_k6_to_k8_for_thinking_and_code() {
        // Ultra-low acceptance collapses to K=2 across c=2..16.
        for c in [2usize, 4, 8, 16] {
            assert_eq!(optimal_cohort_k(c, 0.25, 8), 2, "p=0.25 c={c}");
        }

        // Typical non-greedy prose (p = 0.55, emit ~ 2.1-2.4): narrows c=2 to K2,
        // and keeps c=4, c=8, c=16 at K4 so prose avoids K7 over-drafting.
        assert_eq!(optimal_cohort_k(2, 0.55, 8), 2, "p=0.55 c=2");
        assert_eq!(optimal_cohort_k(4, 0.55, 8), 4, "p=0.55 c=4");
        assert_eq!(optimal_cohort_k(8, 0.55, 8), 4, "p=0.55 c=8");
        assert_eq!(optimal_cohort_k(16, 0.55, 8), 4, "p=0.55 c=16");
        // Moderate acceptance prose (p = 0.62, e.g. HashMap prose): c=2 -> K3, c=4..8 -> K4, c=16 -> K5.
        assert_eq!(optimal_cohort_k(2, 0.62, 8), 3, "p=0.62 c=2");
        assert_eq!(optimal_cohort_k(16, 0.62, 8), 5, "p=0.62 c=16");

        // High-acceptance thinking/code (p = 0.75 at c>=4, p = 0.80..1.0 across c=2..16):
        // unlocks K6..8 (K7 under max_k=7, K8 under max_k=8).
        for c in [4usize, 8, 16] {
            let k = optimal_cohort_k(c, 0.75, 8);
            assert!(
                (6..=8).contains(&k),
                "p=0.75 c={c} should unlock K6..8, got K{k}"
            );
        }
        for p in [0.80_f32, 0.85, 0.95, 1.0] {
            for c in [2usize, 4, 8, 16] {
                let k = optimal_cohort_k(c, p, 8);
                assert_eq!(k, 8, "p={p} c={c} should unlock K8, got K{k}");
            }
        }

        // Monotonic in p for every concurrency level across max_k=5 and max_k=8.
        for max_k in [5usize, 8] {
            for c in [2usize, 4, 8, 16] {
                let mut prev = ADAPTIVE_MIN_K;
                for step in 0..=20 {
                    let p = step as f32 * 0.05;
                    let k = optimal_cohort_k(c, p, max_k);
                    assert!(
                        k >= prev,
                        "non-monotonic k at max_k={max_k} c={c} p={p}: {k} < {prev}"
                    );
                    prev = k;
                }
            }
        }

        // Single-frame 128-row ceiling ALWAYS holds even when caller passes max_k=ADAPTIVE_MAX_K.
        for (c, want_k) in [
            (16usize, 8usize),
            (18, 7),
            (20, 6),
            (24, 5),
            (32, 4),
            (40, 3),
            (64, 2),
        ] {
            let k = optimal_cohort_k(c, 0.90, ADAPTIVE_MAX_K);
            assert_eq!(k, want_k, "frame-capped C{c}");
            assert!(c * k <= ADAPTIVE_MAX_FRAME_ROWS, "C{c} K{k} exceeds 128");
        }

        // Respects explicit tighter max_k ceiling and handles non-finite/boundary inputs safely.
        assert_eq!(optimal_cohort_k(24, 0.90, 4), 4);
        assert_eq!(optimal_cohort_k(36, 0.90, 3), 3);
        assert_eq!(optimal_cohort_k(64, 0.90, 2), 2);
        assert_eq!(optimal_cohort_k(8, f32::NAN, 8), 8);
        assert_eq!(optimal_cohort_k(8, f32::NAN, 5), 5);
        assert_eq!(optimal_cohort_k(1, 0.50, 8), 8);
    }

    #[test]
    fn cached_env_helpers_preserve_default_enabled_disabled_and_fail_closed_semantics() {
        let _lock = test_env_lock();

        // `batched_k5_selected`
        {
            let _g = ScopedEnvVar::remove(BATCHED_K5_ENV);
            assert!(!batched_k5_selected().unwrap());
        }
        {
            let _g = ScopedEnvVar::set(BATCHED_K5_ENV, "0");
            assert!(!batched_k5_selected().unwrap());
        }
        {
            let _g = ScopedEnvVar::set(BATCHED_K5_ENV, "1");
            assert!(batched_k5_selected().unwrap());
        }
        {
            let _g = ScopedEnvVar::set(BATCHED_K5_ENV, "invalid");
            assert!(batched_k5_selected().is_err());
        }

        // `adaptive_k_selected`
        {
            let _g = ScopedEnvVar::remove(ADAPTIVE_K_ENV);
            assert!(!adaptive_k_selected().unwrap());
        }
        {
            let _g = ScopedEnvVar::set(ADAPTIVE_K_ENV, "1");
            assert!(adaptive_k_selected().unwrap());
        }
        {
            let _g = ScopedEnvVar::set(ADAPTIVE_K_ENV, "0");
            assert!(!adaptive_k_selected().unwrap());
        }
        {
            let _g = ScopedEnvVar::set(ADAPTIVE_K_ENV, "2");
            assert!(adaptive_k_selected().is_err());
        }

        // `adaptive_verify_profile`
        {
            let _g = ScopedEnvVar::remove(ADAPTIVE_K4_C7_C16_ENV);
            assert_eq!(
                adaptive_verify_profile().unwrap(),
                AdaptiveVerifyProfile::SingleSweepDefault
            );
        }
        {
            let _g = ScopedEnvVar::set(ADAPTIVE_K4_C7_C16_ENV, "1");
            assert_eq!(
                adaptive_verify_profile().unwrap(),
                AdaptiveVerifyProfile::K4C7C16
            );
        }
        {
            let _g = ScopedEnvVar::set(ADAPTIVE_K4_C7_C16_ENV, "0");
            assert_eq!(
                adaptive_verify_profile().unwrap(),
                AdaptiveVerifyProfile::SingleSweepDefault
            );
        }
        {
            let _g = ScopedEnvVar::set(ADAPTIVE_K4_C7_C16_ENV, "yes");
            assert!(adaptive_verify_profile().is_err());
        }

        // `cohort_adaptive_k_selected` (default ON unless "0")
        {
            let _g = ScopedEnvVar::remove(COHORT_ADAPTIVE_K_ENV);
            assert!(cohort_adaptive_k_selected().unwrap());
        }
        {
            let _g = ScopedEnvVar::set(COHORT_ADAPTIVE_K_ENV, "1");
            assert!(cohort_adaptive_k_selected().unwrap());
        }
        {
            let _g = ScopedEnvVar::set(COHORT_ADAPTIVE_K_ENV, "0");
            assert!(!cohort_adaptive_k_selected().unwrap());
        }
        {
            let _g = ScopedEnvVar::set(COHORT_ADAPTIVE_K_ENV, "bad");
            assert!(cohort_adaptive_k_selected().is_err());
        }

        // `require_fixed_k5_policy`
        {
            let _g0 = ScopedEnvVar::set("ATLAS_QWEN4_MTP_CONF", "0");
            let _g1 = ScopedEnvVar::set("FLUX_QWEN4_MTP_CONF", "0");
            let _g2 = ScopedEnvVar::set("ATLAS_QWEN4_MTP_CONF_EARLY", "0");
            let _g3 = ScopedEnvVar::set("FLUX_QWEN4_MTP_CONF_EARLY", "0");
            let _g4 = ScopedEnvVar::set("ATLAS_QWEN4_MTP_VOCAB_PRUNE", "0");
            let _g5 = ScopedEnvVar::set("FLUX_QWEN4_MTP_VOCAB_PRUNE", "0");
            let _g6 = ScopedEnvVar::remove("ATLAS_QWEN4_MTP_CONF_FILE");
            let _g7 = ScopedEnvVar::remove("FLUX_QWEN4_MTP_CONF_FILE");
            let _g8 = ScopedEnvVar::remove("ATLAS_QWEN4_BATCHED_K5_LADDER");
            let _g9 = ScopedEnvVar::remove("FLUX_QWEN4_BATCHED_K5_LADDER");
            let _g10 = ScopedEnvVar::remove("ATLAS_QWEN4_BATCHED_K5_GDN_MULTI");
            let _g11 = ScopedEnvVar::remove("FLUX_QWEN4_BATCHED_K5_GDN_MULTI");
            assert!(require_fixed_k5_policy().is_ok());
            {
                let _bad = ScopedEnvVar::set("ATLAS_QWEN4_MTP_VOCAB_PRUNE", "1");
                assert!(require_fixed_k5_policy().is_err());
            }
            {
                let _bad = ScopedEnvVar::set("FLUX_QWEN4_BATCHED_K5_GDN_MULTI", "2");
                assert!(require_fixed_k5_policy().is_err());
            }
        }

        // `require_adaptive_k_policy`
        {
            let _b = ScopedEnvVar::set(BATCHED_K5_ENV, "1");
            let _h = ScopedEnvVar::set("ATLAS_QWEN4_HYBRID_ALL_K", "1");
            let _f = ScopedEnvVar::remove("ATLAS_QWEN4_VERIFY_FAST_HOST");
            let _j = ScopedEnvVar::remove("ATLAS_QWEN4_QSA_JOURNAL_ALL_K");
            assert!(require_adaptive_k_policy().is_ok());
            {
                let _off = ScopedEnvVar::set("ATLAS_QWEN4_QSA_JOURNAL_ALL_K", "0");
                assert!(require_adaptive_k_policy().is_err());
            }
            {
                let _off = ScopedEnvVar::set(BATCHED_K5_ENV, "0");
                assert!(require_adaptive_k_policy().is_err());
            }
        }

        // `adaptive_max_frame_rows` env override
        {
            let _g = ScopedEnvVar::remove(ADAPTIVE_MAX_ROWS_ENV);
            // Default unit-test device is SM121 (48 SMs <= 64) -> 128 rows.
            assert_eq!(adaptive_max_frame_rows(), ADAPTIVE_MAX_FRAME_ROWS);
        }
        {
            let _g = ScopedEnvVar::set(ADAPTIVE_MAX_ROWS_ENV, "256");
            assert_eq!(adaptive_max_frame_rows(), 256);
        }
        {
            let _g = ScopedEnvVar::set(ADAPTIVE_MAX_ROWS_ENV, "128");
            assert_eq!(adaptive_max_frame_rows(), 128);
        }
        {
            let _g = ScopedEnvVar::set(ADAPTIVE_MAX_ROWS_ENV, "999");
            assert_eq!(adaptive_max_frame_rows(), ADAPTIVE_MAX_FRAME_ROWS);
        }
    }

    #[test]
    fn sm_aware_adaptive_max_frame_rows_preserves_gb10_128_and_unlocks_sm120_256() {
        // <= 64 SMs (GB10 48 SMs, boundary 64) stay strictly at 128 rows.
        for sm in [0u32, 48, 64] {
            assert_eq!(adaptive_max_frame_rows_for_sm(sm), 128, "sm={sm}");
            assert_eq!(parse_adaptive_max_frame_rows(None, sm), Ok(128));
        }
        // > 64 SMs (SM120 170/188 SMs, boundary 65) lift to 256 rows.
        for sm in [65u32, 110, 170, 188] {
            assert_eq!(adaptive_max_frame_rows_for_sm(sm), 256, "sm={sm}");
            assert_eq!(parse_adaptive_max_frame_rows(None, sm), Ok(256));
        }
        // Env override parses 2..=256 and rejects malformed/out-of-range values.
        assert_eq!(parse_adaptive_max_frame_rows(Some("128"), 188), Ok(128));
        assert_eq!(parse_adaptive_max_frame_rows(Some("256"), 48), Ok(256));
        for bad in ["", " 256", "256 ", "0", "1", "257", "abc"] {
            assert!(
                parse_adaptive_max_frame_rows(Some(bad), 188).is_err(),
                "bad={bad:?}"
            );
        }

        // Under 256-row wide-SM budget:
        // - C=24/32 DFlash2 run full K=8 (192/256 rows)
        // - C=48 Native MTP runs up to K=5 (240 rows) / K=4 (192 rows)
        // - C=64 Native MTP runs K=4 (256 rows) instead of K=2 (128 rows)
        assert_eq!(adaptive_verify_width_for_max_rows(24, 256), Some(8));
        assert_eq!(adaptive_verify_width_for_max_rows(32, 256), Some(8));
        assert_eq!(adaptive_verify_width_for_max_rows(48, 256), Some(5));
        assert_eq!(adaptive_verify_width_for_max_rows(64, 256), Some(4));
        assert_eq!(adaptive_verify_width_for_max_rows(128, 256), Some(2));
        assert_eq!(adaptive_verify_width_for_max_rows(129, 256), None);

        // Roofline optimal_cohort_k_for_max_rows unlocks K=4 at C=48/64 and K=8 at C=24/32.
        assert_eq!(optimal_cohort_k_for_max_rows(24, 0.90, 8, 256), 8);
        assert_eq!(optimal_cohort_k_for_max_rows(32, 0.90, 8, 256), 8);
        assert_eq!(optimal_cohort_k_for_max_rows(48, 0.80, 4, 256), 4);
        assert_eq!(optimal_cohort_k_for_max_rows(64, 0.80, 4, 256), 4);
        // GB10 128-row budget still caps C=64 at K=2.
        assert_eq!(optimal_cohort_k_for_max_rows(64, 0.80, 4, 128), 2);
    }

    #[test]
    fn k4_c7_c16_profile_respects_explicit_max_frame_rows_budget() {
        // Under a 64-row budget, K4C7C16 must narrow below its 128-row ladder.
        assert_eq!(
            adaptive_verify_width_for_profile_and_max_rows(32, AdaptiveVerifyProfile::K4C7C16, 64),
            Some(2)
        );
        assert_eq!(
            adaptive_verify_width_for_profile_and_max_rows(17, AdaptiveVerifyProfile::K4C7C16, 64),
            Some(3)
        );
        assert_eq!(
            adaptive_verify_width_for_profile_and_max_rows(16, AdaptiveVerifyProfile::K4C7C16, 64),
            Some(4)
        );
        // Wider budgets remain capped by the K4C7C16 profile ceiling.
        assert_eq!(
            adaptive_verify_width_for_profile_and_max_rows(16, AdaptiveVerifyProfile::K4C7C16, 256),
            Some(4)
        );
        assert_eq!(
            adaptive_verify_width_for_profile_and_max_rows(
                MAX_LANES + 1,
                AdaptiveVerifyProfile::K4C7C16,
                64
            ),
            None
        );
    }

    #[test]
    #[should_panic(expected = "test_env_lock()")]
    fn scoped_env_var_requires_test_env_lock() {
        let _guard = ScopedEnvVar::remove("ATLAS_QWEN4_SCOPED_ENV_LOCK_TEST_UNUSED");
    }
}
