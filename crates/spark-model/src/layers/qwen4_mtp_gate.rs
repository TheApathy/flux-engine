// SPDX-License-Identifier: AGPL-3.0-only

//! Draft-confidence gate policy for the native Qwen4 MTP draft chain.
//!
//! The base policy comes from the environment (read once):
//! - `ATLAS_QWEN4_MTP_CONF=<p>`: keep drafts while their top-1 probability is
//!   at least `p` (the first draft is kept unless zero-draft stop is enabled).
//! - `ATLAS_QWEN4_MTP_CONF_STEPS=t0,t1,...`: up to 8 comma-separated per-step
//!   confidence thresholds (1..=8 values; 4-value backwards-compatible).
//! - `ATLAS_QWEN4_MTP_ALLOW_ZERO=1` (or `p_0 < 0.12` under adaptive depth):
//!   hopeless high-entropy step-0 positions return `0` drafts (`K=1` plain
//!   decode) instead of forcing a wasted `K=2` verify.
//! - `ATLAS_QWEN4_MTP_CONF_CUM=<tau_cum>`: cumulative confidence floor
//!   `prod_{i=0..=step} p_i >= tau_cum` (checked at `step == 0` too when
//!   zero-draft stop is enabled).
//! - `ATLAS_QWEN4_MTP_CONF_EARLY=1`: stop *computing* drafts at the first one
//!   failing the gate, so rejected-by-gate drafts cost one MTP step, not the
//!   rest of the chain. Without it the chain computes every draft and only the
//!   verify width shrinks.
//! - `ATLAS_QWEN4_MTP_CONF_LOG=1`: log every chain's draft probabilities.
//! - `ATLAS_QWEN4_MTP_CONF_FILE=<path>`: measurement knob. The file holds
//!   `<tau|none> <max_drafts|all> <early 0|1>` and overrides the env policy
//!   for the next chain after it changes (checked by mtime), so one server can
//!   serve several gate settings. A malformed file keeps the previous policy.
//! - `ATLAS_QWEN4_MTP_GATE_QSA_MIN_WIDTH=<w>` (default off; 4..=5): for a
//!   request that reaches QSA context, a gated chain that would verify with
//!   width 3..w-1 keeps computing and verifies every draft instead (width 2,
//!   one draft, and width 1, zero drafts, are unchanged).
//!
//! Every setting is lossless: verify still decides every committed token.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::SystemTime;

use anyhow::{Result, bail};
use parking_lot::Mutex;

pub const MTP_VOCAB_PRUNE_SELECTOR: &str = "ATLAS_QWEN4_MTP_VOCAB_PRUNE";
#[allow(dead_code)]
pub const VOCAB_PRUNE_SELECTOR: &str = MTP_VOCAB_PRUNE_SELECTOR;
pub const PRUNED_VOCAB_SIZE: u32 = 96_000;
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.85;
pub const MAX_GATE_STEPS: usize = 8;
pub const DEFAULT_ZERO_DRAFT_THRESHOLD: f32 = 0.12;
pub const DEFAULT_GUMBEL_CUM_THRESHOLD: f32 = 0.38;
pub const DEFAULT_FAST_HOST_DRAFT_VOCAB: usize = 73_728;
pub const DEFAULT_GUMBEL_FAST_HOST_DRAFT_VOCAB: usize = 98_304;

pub fn mtp_vocab_prune_selected() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| {
        let on = std::env::var(MTP_VOCAB_PRUNE_SELECTOR).ok().as_deref() == Some("1");
        if on {
            static VOCAB_PRUNE_LOGGED: std::sync::Once = std::sync::Once::new();
            VOCAB_PRUNE_LOGGED.call_once(|| {
                tracing::info!(
                    "mtp_gate: MTP_VOCAB_PRUNED_96000 hidden_size=2560 default_confidence_threshold=0.85"
                );
            });
        }
        on
    })
}

#[allow(dead_code)]
pub fn vocab_prune_enabled() -> bool {
    mtp_vocab_prune_selected()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DraftVocabEnv {
    Unset,
    Disabled,
    Rows(usize),
}

fn parse_draft_vocab_env(raw: Option<&str>) -> DraftVocabEnv {
    let Some(trimmed) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return DraftVocabEnv::Unset;
    };
    if trimmed == "0" || trimmed.eq_ignore_ascii_case("off") {
        return DraftVocabEnv::Disabled;
    }
    if let Ok(n) = trimmed.parse::<usize>()
        && n >= 1024
    {
        return DraftVocabEnv::Rows(n);
    }
    DraftVocabEnv::Unset
}

fn resolve_draft_vocab_override(
    env: DraftVocabEnv,
    fast_host: bool,
    gumbel_active: bool,
) -> Option<usize> {
    match env {
        DraftVocabEnv::Disabled => None,
        DraftVocabEnv::Rows(n) => Some(n),
        DraftVocabEnv::Unset => {
            if fast_host {
                Some(if gumbel_active {
                    DEFAULT_GUMBEL_FAST_HOST_DRAFT_VOCAB
                } else {
                    DEFAULT_FAST_HOST_DRAFT_VOCAB
                })
            } else {
                None
            }
        }
    }
}

fn draft_vocab_env() -> DraftVocabEnv {
    static ENV: OnceLock<DraftVocabEnv> = OnceLock::new();
    *ENV.get_or_init(|| {
        let raw = std::env::var("ATLAS_QWEN4_MTP_DRAFT_VOCAB")
            .or_else(|_| std::env::var("MTP_DRAFT_VOCAB"))
            .ok();
        let parsed = parse_draft_vocab_env(raw.as_deref());
        if let DraftVocabEnv::Rows(n) = parsed {
            tracing::info!("ATLAS_QWEN4_MTP_DRAFT_VOCAB={n} engaged");
        }
        parsed
    })
}

/// Optional compact draft vocabulary cap (`ATLAS_QWEN4_MTP_DRAFT_VOCAB=<rows>`),
/// parameterized by whether non-greedy CRN Gumbel sampling is active.
///
/// When `ATLAS_QWEN4_VERIFY_FAST_HOST=1` is active and no explicit env cap is set:
/// - Greedy (`gumbel_active == false`) defaults to `73_728` (`288 * 256` rows).
/// - Non-greedy Gumbel (`gumbel_active == true`) raises the default cap to
///   `98_304` (`384 * 256` rows) to reduce tail truncation under `T > 0` while
///   cutting MTP LM-head DRAM traffic by 25% vs 131K.
pub fn mtp_draft_vocab_override_for(gumbel_active: bool) -> Option<usize> {
    let env = draft_vocab_env();
    let fast_host = crate::layers::qwen4_verify_fast_host_selected();
    let resolved = resolve_draft_vocab_override(env, fast_host, gumbel_active);
    if env == DraftVocabEnv::Unset
        && let Some(default_v) = resolved
    {
        if gumbel_active {
            static LOGGED_GUMBEL: std::sync::Once = std::sync::Once::new();
            LOGGED_GUMBEL.call_once(|| {
                tracing::info!(
                    "ATLAS_QWEN4_MTP_DRAFT_VOCAB={default_v} (default fast-host Gumbel compact vocab) engaged"
                );
            });
        } else {
            static LOGGED_GREEDY: std::sync::Once = std::sync::Once::new();
            LOGGED_GREEDY.call_once(|| {
                tracing::info!(
                    "ATLAS_QWEN4_MTP_DRAFT_VOCAB={default_v} (default fast-host compact vocab) engaged"
                );
            });
        }
    }
    resolved
}

/// Optional compact draft vocabulary cap (`ATLAS_QWEN4_MTP_DRAFT_VOCAB=<rows>`).
/// Defaults to `73_728` (`288 * 256` rows, covering 100% of `code0`/`rep0`/`mix128k`
/// and 99%+ of general tokens) when `ATLAS_QWEN4_VERIFY_FAST_HOST=1` is active,
/// unless `ATLAS_QWEN4_MTP_DRAFT_VOCAB=0` is set.
#[allow(dead_code)]
pub fn mtp_draft_vocab_override() -> Option<usize> {
    mtp_draft_vocab_override_for(false)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GatePolicy {
    pub threshold: Option<f32>,
    /// Cap on drafts per chain (clamped to [1, num_drafts] by the caller).
    pub max_drafts: Option<usize>,
    pub early_stop: bool,
    pub log: bool,
}

impl GatePolicy {
    /// Drafts to request this chain.
    pub fn chain_len(&self, num_drafts: usize) -> usize {
        self.max_drafts
            .map_or(num_drafts, |cap| cap.clamp(1, num_drafts.max(1)))
            .min(num_drafts)
    }

    /// Whether probabilities are required for a chain with explicit gate inputs.
    pub(crate) fn wants_probabilities_with(
        &self,
        step_taus: Option<StepThresholds>,
        cum_tau: Option<f32>,
        zero_policy: ZeroDraftPolicy,
    ) -> bool {
        self.log
            || self.threshold.is_some()
            || step_taus.is_some()
            || cum_tau.is_some()
            || zero_policy != ZeroDraftPolicy::KeepAtLeastOne
    }

    pub fn wants_probabilities(&self) -> bool {
        self.wants_probabilities_with(step_thresholds(), cum_threshold(), zero_draft_policy())
    }

    /// Batched (`b >= 2`) cohort-uniform chains enforce
    /// [`ZeroDraftPolicy::KeepAtLeastOne`], so an implicit `gumbel` / adaptive
    /// zero-draft floor never forces probabilities or disables `argmax_bf16_rows`.
    pub fn wants_probabilities_batch(&self) -> bool {
        self.wants_probabilities_with(
            step_thresholds(),
            cum_threshold(),
            batch_zero_draft_policy(),
        )
    }

    /// Base threshold for a chain with explicit gate inputs.
    pub(crate) fn effective_base_threshold_with(
        &self,
        step_taus: Option<StepThresholds>,
        cum_tau: Option<f32>,
        zero_policy: ZeroDraftPolicy,
    ) -> Option<f32> {
        self.threshold.or_else(|| {
            (step_taus.is_some()
                || cum_tau.is_some()
                || zero_policy != ZeroDraftPolicy::KeepAtLeastOne)
                .then_some(0.0)
        })
    }

    pub fn effective_base_threshold(&self) -> Option<f32> {
        self.effective_base_threshold_with(step_thresholds(), cum_threshold(), zero_draft_policy())
    }

    /// Effective base threshold for batched (`b >= 2`) cohort-uniform chains
    /// (`ZeroDraftPolicy::KeepAtLeastOne`).
    pub fn effective_base_threshold_batch(&self) -> Option<f32> {
        self.effective_base_threshold_with(
            step_thresholds(),
            cum_threshold(),
            batch_zero_draft_policy(),
        )
    }

    pub fn early_threshold(&self) -> Option<f32> {
        self.effective_base_threshold().filter(|_| self.early_stop)
    }

    /// Early-stop threshold for batched (`b >= 2`) cohort-uniform chains
    /// (`ZeroDraftPolicy::KeepAtLeastOne`).
    pub fn early_threshold_batch(&self) -> Option<f32> {
        self.effective_base_threshold_batch()
            .filter(|_| self.early_stop)
    }
}

fn parse_threshold(token: &str) -> Result<Option<f32>> {
    if token.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    let p: f32 = token.parse()?;
    if !(p > 0.0 && p < 1.0) {
        bail!("threshold must be in (0, 1) or 'none', got {token}");
    }
    Ok(Some(p))
}

/// Parse control-file text `<tau|none> <max_drafts|all> <early 0|1>`.
pub(crate) fn parse_gate_control(text: &str) -> Result<(Option<f32>, Option<usize>, bool)> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.len() != 3 {
        bail!("expected '<tau|none> <max_drafts|all> <early 0|1>', got {text:?}");
    }
    let threshold = parse_threshold(fields[0])?;
    let max_drafts = if fields[1].eq_ignore_ascii_case("all") {
        None
    } else {
        let m: usize = fields[1].parse()?;
        if m == 0 {
            bail!("max_drafts must be >= 1 or 'all'");
        }
        Some(m)
    };
    let early = match fields[2] {
        "0" => false,
        "1" => true,
        other => bail!("early must be 0 or 1, got {other:?}"),
    };
    Ok((threshold, max_drafts, early))
}

/// `ATLAS_QWEN4_MTP_GATE_QSA_MIN_WIDTH` (read once; 4..=5, else off).
pub(crate) fn qsa_min_width() -> Option<usize> {
    static W: OnceLock<Option<usize>> = OnceLock::new();
    *W.get_or_init(|| {
        let w = std::env::var("ATLAS_QWEN4_MTP_GATE_QSA_MIN_WIDTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|w| (4..=5).contains(w));
        if let Some(w) = w {
            tracing::info!(
                "ATLAS_QWEN4_MTP_GATE_QSA_MIN_WIDTH={w} requested: gated widths 3..{w} round up at QSA context"
            );
        }
        w
    })
}

thread_local! {
    /// Set by the model around one propose call when the request reaches QSA.
    static QSA_CONTEXT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Marks the propose calls made while the guard lives as QSA context.
pub(crate) struct QsaContextGuard(bool);

impl QsaContextGuard {
    pub(crate) fn new(qsa_context: bool) -> Self {
        Self(QSA_CONTEXT.with(|c| c.replace(qsa_context)))
    }
}

impl Drop for QsaContextGuard {
    fn drop(&mut self) {
        QSA_CONTEXT.with(|c| c.set(self.0));
    }
}

/// The width floor for this chain: `Some(w)` when the selector is set and the
/// current request reaches QSA context.
pub(crate) fn chain_min_width() -> Option<usize> {
    qsa_min_width().filter(|_| QSA_CONTEXT.with(|c| c.get()))
}

/// Drafts kept under a width floor: a kept prefix of 2..min_width-2 drafts
/// (verify width 3..min_width-1) becomes every computed draft; zero/one draft
/// (width 1/2) and anything already at the floor stay.
pub(crate) const fn floor_keep(keep: usize, computed: usize, min_width: usize) -> usize {
    if keep >= 2 && keep + 1 < min_width && computed > keep {
        computed
    } else {
        keep
    }
}

/// Per-step confidence thresholds (`1..=8` steps; clamps steps past `len - 1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StepThresholds {
    values: [f32; MAX_GATE_STEPS],
    len: usize,
}

impl StepThresholds {
    #[inline]
    pub(crate) fn for_step(&self, step: usize) -> f32 {
        self.values[step.min(self.len.saturating_sub(1))]
    }

    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn as_slice(&self) -> &[f32] {
        &self.values[..self.len]
    }
}

/// Parse `1..=8` comma-separated probabilities in `[0.0, 1.0]`.
pub(crate) fn parse_step_thresholds(raw: &str) -> Option<StepThresholds> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut values = [0.0f32; MAX_GATE_STEPS];
    let mut len = 0usize;
    for part in trimmed.split(',') {
        if len >= MAX_GATE_STEPS {
            return None;
        }
        let val = part.trim().parse::<f32>().ok()?;
        if !(0.0..=1.0).contains(&val) {
            return None;
        }
        values[len] = val;
        len += 1;
    }
    (len >= 1).then_some(StepThresholds { values, len })
}

/// Optional per-step threshold overrides (`ATLAS_QWEN4_MTP_CONF_STEPS=t0,t1,...`, `1..=8` values).
pub(crate) fn step_thresholds() -> Option<StepThresholds> {
    static STEPS: OnceLock<Option<StepThresholds>> = OnceLock::new();
    *STEPS.get_or_init(|| {
        let raw = std::env::var("ATLAS_QWEN4_MTP_CONF_STEPS").ok()?;
        let parsed = parse_step_thresholds(&raw);
        if let Some(ref st) = parsed {
            tracing::info!(
                "ATLAS_QWEN4_MTP_CONF_STEPS={:?} engaged",
                &st.values[..st.len]
            );
        } else {
            tracing::warn!(
                "ATLAS_QWEN4_MTP_CONF_STEPS={raw:?} ignored (expected 1..={MAX_GATE_STEPS} values in [0, 1])"
            );
        }
        parsed
    })
}

/// Parsed state of `ATLAS_QWEN4_MTP_CONF_CUM`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CumThresholdEnv {
    Unset,
    Disabled,
    Value(f32),
}

pub(crate) fn parse_cum_threshold_env(raw: Option<&str>) -> CumThresholdEnv {
    let Some(trimmed) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return CumThresholdEnv::Unset;
    };
    if trimmed == "0" || trimmed.eq_ignore_ascii_case("off") || trimmed.eq_ignore_ascii_case("none")
    {
        return CumThresholdEnv::Disabled;
    }
    if let Ok(v) = trimmed.parse::<f32>() {
        if v == 0.0 {
            return CumThresholdEnv::Disabled;
        }
        if v > 0.0 && v < 1.0 {
            return CumThresholdEnv::Value(v);
        }
    }
    CumThresholdEnv::Unset
}

pub(crate) fn cum_threshold_for(env: CumThresholdEnv, gumbel_active: bool) -> Option<f32> {
    match env {
        CumThresholdEnv::Value(v) => Some(v),
        CumThresholdEnv::Disabled => None,
        CumThresholdEnv::Unset => gumbel_active.then_some(DEFAULT_GUMBEL_CUM_THRESHOLD),
    }
}

fn cum_threshold_env() -> CumThresholdEnv {
    static CUM: OnceLock<CumThresholdEnv> = OnceLock::new();
    *CUM.get_or_init(|| {
        let raw = std::env::var("ATLAS_QWEN4_MTP_CONF_CUM").ok();
        let parsed = parse_cum_threshold_env(raw.as_deref());
        if let CumThresholdEnv::Value(v) = parsed {
            tracing::info!("ATLAS_QWEN4_MTP_CONF_CUM={v} engaged");
        }
        parsed
    })
}

/// Optional cumulative probability threshold (`ATLAS_QWEN4_MTP_CONF_CUM=<tau_cum>`).
pub(crate) fn cum_threshold() -> Option<f32> {
    cum_threshold_for(cum_threshold_env(), false)
}

/// Cumulative probability threshold for non-greedy Gumbel-CRN single-sequence chains.
///
/// Defaults to [`DEFAULT_GUMBEL_CUM_THRESHOLD`] (`0.38`) when `ATLAS_QWEN4_MTP_CONF_CUM`
/// is unset, honors explicit values in `(0, 1)`, and disables (`None`) when set to `0`/`off`/`none`.
pub(crate) fn gumbel_cum_threshold() -> Option<f32> {
    let env = cum_threshold_env();
    let val = cum_threshold_for(env, true);
    if env == CumThresholdEnv::Unset
        && let Some(v) = val
    {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!("ATLAS_QWEN4_MTP_CONF_CUM={v} (default Gumbel running-product) engaged");
        });
    }
    val
}

/// Whether `ATLAS_QWEN4_MTP_CONF_EARLY=0` explicitly disabled mid-chain early-stop syncs.
pub(crate) fn early_stop_explicitly_disabled() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_MTP_CONF_EARLY")
            .ok()
            .as_deref()
            .map(str::trim)
            == Some("0")
    })
}

/// Step-0 zero-draft early-stop rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ZeroDraftPolicy {
    /// Historical default: always keep at least 1 draft (`step.max(1)`).
    KeepAtLeastOne,
    /// Return `0` drafts when `p_0 < floor` (default `0.12` under adaptive depth)
    /// or `p_0 < cum_threshold`.
    BelowFloor(f32),
    /// `ATLAS_QWEN4_MTP_ALLOW_ZERO=1`: return `0` drafts whenever step 0 fails
    /// its step threshold or `cum_threshold`.
    OnStepZeroFail,
}

impl ZeroDraftPolicy {
    #[inline]
    pub(crate) fn allows_zero(self) -> bool {
        !matches!(self, Self::KeepAtLeastOne)
    }
}

pub(crate) fn resolve_zero_draft_policy(
    allow_zero_env: Option<&str>,
    zero_tau_env: Option<&str>,
    spec_verify_env: Option<&str>,
    adaptive_depth_env: Option<&str>,
) -> ZeroDraftPolicy {
    match allow_zero_env.map(str::trim) {
        Some("0") => return ZeroDraftPolicy::KeepAtLeastOne,
        Some("1") => return ZeroDraftPolicy::OnStepZeroFail,
        _ => {}
    }
    if let Some(tau) = zero_tau_env
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| *v > 0.0 && *v < 1.0)
    {
        return ZeroDraftPolicy::BelowFloor(tau);
    }
    let spec_adaptive = matches!(
        spec_verify_env.map(str::trim),
        Some("stochastic" | "gumbel")
    ) && adaptive_depth_env.map(str::trim) != Some("0");
    if spec_adaptive || adaptive_depth_env.map(str::trim) == Some("1") {
        ZeroDraftPolicy::BelowFloor(DEFAULT_ZERO_DRAFT_THRESHOLD)
    } else {
        ZeroDraftPolicy::KeepAtLeastOne
    }
}

pub(crate) fn zero_draft_policy() -> ZeroDraftPolicy {
    static POLICY: OnceLock<ZeroDraftPolicy> = OnceLock::new();
    *POLICY.get_or_init(|| {
        let allow_zero = std::env::var("ATLAS_QWEN4_MTP_ALLOW_ZERO").ok();
        let zero_tau = std::env::var("ATLAS_QWEN4_MTP_ZERO_TAU").ok();
        let spec_verify = std::env::var("FLUX_SPEC_VERIFY")
            .or_else(|_| std::env::var("ATLAS_SPEC_VERIFY"))
            .ok();
        let adaptive_depth = std::env::var("FLUX_SPEC_ADAPTIVE_DEPTH")
            .or_else(|_| std::env::var("ATLAS_SPEC_ADAPTIVE_DEPTH"))
            .ok();
        let policy = resolve_zero_draft_policy(
            allow_zero.as_deref(),
            zero_tau.as_deref(),
            spec_verify.as_deref(),
            adaptive_depth.as_deref(),
        );
        if policy.allows_zero() {
            tracing::info!("ATLAS_QWEN4_MTP_ZERO_DRAFT policy={policy:?} engaged");
        }
        policy
    })
}

/// Pure parameterized early-stop rule returning `Some(kept_drafts)` when the chain
/// stops after computing `step` (0-based) with probability `p` and cumulative `p_cum`.
pub(crate) fn early_stop_keep_with(
    step: usize,
    p: f32,
    p_cum: f32,
    base_threshold: f32,
    step_taus: Option<StepThresholds>,
    cum_tau: Option<f32>,
    zero_policy: ZeroDraftPolicy,
) -> Option<usize> {
    let thresh = step_taus.map_or(base_threshold, |st| st.for_step(step));
    if step == 0 {
        let cum_fail = zero_policy.allows_zero() && cum_tau.is_some_and(|tc| p_cum < tc);
        match zero_policy {
            ZeroDraftPolicy::OnStepZeroFail if p < thresh || cum_fail => return Some(0),
            ZeroDraftPolicy::BelowFloor(floor) if p < floor || cum_fail => return Some(0),
            _ if p < thresh => return Some(1),
            _ => return None,
        }
    }
    if p < thresh || cum_tau.is_some_and(|tc| p_cum < tc) {
        Some(step)
    } else {
        None
    }
}

/// Post-hoc confident prefix length matching [`early_stop_keep_with`] identically.
pub(crate) fn confident_prefix_with(
    probs: &[f32],
    base_threshold: f32,
    step_taus: Option<StepThresholds>,
    cum_tau: Option<f32>,
    zero_policy: ZeroDraftPolicy,
) -> usize {
    if probs.is_empty() {
        return 0;
    }
    let mut p_cum = 1.0f32;
    for (step, &p) in probs.iter().enumerate() {
        p_cum *= p;
        if let Some(keep) = early_stop_keep_with(
            step,
            p,
            p_cum,
            base_threshold,
            step_taus,
            cum_tau,
            zero_policy,
        ) {
            return keep.min(probs.len());
        }
    }
    probs.len()
}

/// After computing draft `step` (0-based) with probability `p`: `Some(kept)`
/// when the chain stops here under the active environment policy.
#[allow(dead_code)]
pub(crate) fn early_stop_keep(step: usize, p: f32, threshold: f32) -> Option<usize> {
    early_stop_keep_step(step, p, p, threshold)
}

/// Cohort-uniform (`b >= 2`) zero-draft policy: batched drafting never ejects
/// a cohort member to `0` drafts (which would split the batched verify forward
/// into an extra full-weight-read bootstrap pass).
pub(crate) const fn batch_zero_draft_policy() -> ZeroDraftPolicy {
    ZeroDraftPolicy::KeepAtLeastOne
}

/// Environment-configured early-stop check carrying running `p_cum`.
pub(crate) fn early_stop_keep_step(
    step: usize,
    p: f32,
    p_cum: f32,
    base_threshold: f32,
) -> Option<usize> {
    early_stop_keep_with(
        step,
        p,
        p_cum,
        base_threshold,
        step_thresholds(),
        cum_threshold(),
        zero_draft_policy(),
    )
}

/// Gumbel single-sequence early-stop check using `0.0` per-step floor (ignoring
/// greedy per-step `MTP_CONF` / `MTP_CONF_STEPS` floors) and running-product
/// `cum_tau` (`p_cum *= p_chosen < cum_tau`).
pub(crate) fn early_stop_keep_step_gumbel(
    step: usize,
    p: f32,
    p_cum: f32,
    cum_tau: Option<f32>,
) -> Option<usize> {
    early_stop_keep_with(step, p, p_cum, 0.0, None, cum_tau, zero_draft_policy())
}

/// Environment-configured post-hoc confident prefix length (`0..=probs.len()`).
pub(crate) fn confident_prefix_env(probs: &[f32], base_threshold: f32) -> usize {
    confident_prefix_with(
        probs,
        base_threshold,
        step_thresholds(),
        cum_threshold(),
        zero_draft_policy(),
    )
}

/// Gumbel single-sequence post-hoc confident prefix length matching
/// [`early_stop_keep_step_gumbel`] identically (`base_threshold = 0.0`, no
/// greedy per-step floors, running-product `cum_tau`).
pub(crate) fn confident_prefix_gumbel(probs: &[f32], cum_tau: Option<f32>) -> usize {
    confident_prefix_with(probs, 0.0, None, cum_tau, zero_draft_policy())
}

/// Cohort-uniform (`b >= 2`) post-hoc confident prefix length (`1..=probs.len()`
/// for non-empty `probs`), immune to implicit `gumbel` / adaptive zero-draft floors.
#[cfg(test)]
pub(crate) fn confident_prefix_batch_env(probs: &[f32], base_threshold: f32) -> usize {
    confident_prefix_with(
        probs,
        base_threshold,
        step_thresholds(),
        cum_threshold(),
        batch_zero_draft_policy(),
    )
}

/// Effective threshold check for `step` given base `threshold`, `p_cum`, and optional
/// per-step / cumulative overrides. Returns `true` when draft `step` passes.
#[allow(dead_code)]
pub(crate) fn step_passes(step: usize, p: f32, p_cum: f32, base_threshold: f32) -> bool {
    early_stop_keep_step(step, p, p_cum, base_threshold).is_none()
}

fn env_policy() -> GatePolicy {
    static BASE: OnceLock<GatePolicy> = OnceLock::new();
    *BASE.get_or_init(|| {
        let log = std::env::var("ATLAS_QWEN4_MTP_CONF_LOG").is_ok_and(|v| v == "1");
        let prune_active = mtp_vocab_prune_selected();
        let default_thresh = if prune_active {
            Some(DEFAULT_CONFIDENCE_THRESHOLD)
        } else {
            None
        };
        let threshold = std::env::var("ATLAS_QWEN4_MTP_CONF")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|p| *p > 0.0 && *p < 1.0)
            .or(default_thresh);
        let early_stop = std::env::var("ATLAS_QWEN4_MTP_CONF_EARLY").is_ok_and(|v| v == "1");
        let policy = GatePolicy {
            threshold,
            max_drafts: None,
            early_stop,
            log,
        };
        if log || policy.wants_probabilities() || control_path().is_some() {
            tracing::info!(
                "MTP_DRAFT_CONF_ENGAGED log={log} threshold={threshold:?} early={early_stop} \
                 control_file={:?} selector=ATLAS_QWEN4_MTP_CONF",
                control_path()
            );
        }
        policy
    })
}

fn control_path() -> Option<PathBuf> {
    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| std::env::var_os("ATLAS_QWEN4_MTP_CONF_FILE").map(PathBuf::from))
        .clone()
}

struct ControlState {
    mtime: Option<SystemTime>,
    policy: GatePolicy,
}

/// The policy for the next draft chain.
pub(crate) fn current_policy() -> GatePolicy {
    let base = env_policy();
    let Some(path) = control_path() else {
        return base;
    };
    static STATE: OnceLock<Mutex<ControlState>> = OnceLock::new();
    let state = STATE.get_or_init(|| {
        Mutex::new(ControlState {
            mtime: None,
            policy: base,
        })
    });
    let mut guard = state.lock();
    let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
    if mtime.is_some() && mtime != guard.mtime {
        guard.mtime = mtime;
        match std::fs::read_to_string(&path)
            .map_err(anyhow::Error::from)
            .and_then(|text| parse_gate_control(&text))
        {
            Ok((threshold, max_drafts, early_stop)) => {
                guard.policy = GatePolicy {
                    threshold,
                    max_drafts,
                    early_stop,
                    log: base.log,
                };
                tracing::info!(
                    "MTP_GATE_CONTROL threshold={threshold:?} max_drafts={max_drafts:?} \
                     early={early_stop}"
                );
            }
            Err(err) => {
                tracing::warn!("MTP_GATE_CONTROL ignored {path:?}: {err:#}");
            }
        }
    }
    guard.policy
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefix(probs: &[f32], t: f32) -> usize {
        probs
            .iter()
            .position(|p| *p < t)
            .unwrap_or(probs.len())
            .max(1)
            .min(probs.len())
    }

    /// Early stop must keep exactly what the post-hoc prefix rule keeps, and
    /// compute at most one draft past the kept prefix.
    #[test]
    fn early_stop_matches_posthoc_prefix() {
        let grid = [0.05f32, 0.4, 0.59, 0.6, 0.61, 0.9, 0.99];
        for t in [0.5f32, 0.6, 0.75] {
            for a in grid {
                for b in grid {
                    for c in grid {
                        for d in grid {
                            let probs = [a, b, c, d];
                            let mut computed = 0;
                            let mut kept = None;
                            for (step, p) in probs.iter().enumerate() {
                                computed = step + 1;
                                if let Some(k) = early_stop_keep(step, *p, t) {
                                    kept = Some(k);
                                    break;
                                }
                            }
                            let kept = kept.unwrap_or(computed);
                            assert_eq!(kept, prefix(&probs, t), "{probs:?} t={t}");
                            assert_eq!(kept, prefix(&probs[..computed], t), "{probs:?} t={t}");
                            assert!(computed <= (kept + 1).min(probs.len()));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn first_draft_is_always_kept() {
        assert_eq!(early_stop_keep(0, 0.01, 0.6), Some(1));
        assert_eq!(early_stop_keep(0, 0.6, 0.6), None);
        assert_eq!(early_stop_keep(2, 0.3, 0.6), Some(2));
    }

    #[test]
    fn step_thresholds_support_1_through_8_values_with_4_value_compat() {
        let st4 = parse_step_thresholds("0.8, 0.7, 0.6, 0.5").unwrap();
        assert_eq!(st4.as_slice(), &[0.8, 0.7, 0.6, 0.5]);
        assert_eq!(st4.for_step(0), 0.8);
        assert_eq!(st4.for_step(3), 0.5);
        assert_eq!(st4.for_step(7), 0.5, "steps past len-1 clamp to last");

        let st8 = parse_step_thresholds("0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55").unwrap();
        assert_eq!(st8.as_slice().len(), 8);
        assert_eq!(st8.for_step(7), 0.55);

        let st1 = parse_step_thresholds("0.42").unwrap();
        assert_eq!(st1.for_step(5), 0.42);

        for bad in [
            "",
            "0.5,1.2",
            "-0.1,0.5",
            "0.5,bad",
            "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        ] {
            assert!(parse_step_thresholds(bad).is_none(), "accepted {bad:?}");
        }
    }

    #[test]
    fn zero_draft_policy_and_step0_cum_threshold_work_across_0_to_8_depths() {
        assert_eq!(
            resolve_zero_draft_policy(Some("1"), None, None, None),
            ZeroDraftPolicy::OnStepZeroFail
        );
        assert_eq!(
            resolve_zero_draft_policy(None, None, Some("gumbel"), None),
            ZeroDraftPolicy::BelowFloor(DEFAULT_ZERO_DRAFT_THRESHOLD)
        );
        assert_eq!(
            resolve_zero_draft_policy(Some("0"), None, Some("gumbel"), Some("1")),
            ZeroDraftPolicy::KeepAtLeastOne
        );

        // Under OnStepZeroFail: p_0 = 0.25 < 0.5 returns 0 drafts!
        assert_eq!(
            early_stop_keep_with(
                0,
                0.25,
                0.25,
                0.5,
                None,
                None,
                ZeroDraftPolicy::OnStepZeroFail
            ),
            Some(0)
        );
        assert_eq!(
            confident_prefix_with(
                &[0.25, 0.9, 0.9],
                0.5,
                None,
                None,
                ZeroDraftPolicy::OnStepZeroFail
            ),
            0
        );

        // Under BelowFloor(0.12): p_0 = 0.08 < 0.12 returns 0 drafts; p_0 = 0.30 < 0.5 keeps 1!
        let floor = ZeroDraftPolicy::BelowFloor(0.12);
        assert_eq!(
            early_stop_keep_with(0, 0.08, 0.08, 0.5, None, None, floor),
            Some(0)
        );
        assert_eq!(
            early_stop_keep_with(0, 0.30, 0.30, 0.5, None, None, floor),
            Some(1)
        );
        assert_eq!(
            confident_prefix_with(&[0.08, 0.9], 0.5, None, None, floor),
            0
        );
        assert_eq!(
            confident_prefix_with(&[0.30, 0.9], 0.5, None, None, floor),
            1
        );

        // Cumulative threshold at step == 0 also triggers 0 drafts when zero stop is enabled!
        assert_eq!(
            early_stop_keep_with(0, 0.20, 0.20, 0.15, None, Some(0.25), floor),
            Some(0)
        );
        assert_eq!(
            early_stop_keep_with(
                0,
                0.20,
                0.20,
                0.15,
                None,
                Some(0.25),
                ZeroDraftPolicy::KeepAtLeastOne
            ),
            None,
            "legacy KeepAtLeastOne ignores cum_threshold at step 0"
        );

        // 8-step chain with per-step thresholds stops cleanly at step 6.
        let st8 = parse_step_thresholds("0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45").unwrap();
        let probs8 = [0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.49, 0.9];
        assert_eq!(
            confident_prefix_with(&probs8, 0.0, Some(st8), None, floor),
            6
        );
    }

    #[test]
    fn qsa_width_floor_rounds_three_and_four_up_only() {
        // 4 drafts computed, floor 5: widths 3 and 4 become 5; width 1 (0 drafts), 2 and 5 stay.
        assert_eq!(floor_keep(0, 4, 5), 0);
        assert_eq!(floor_keep(1, 4, 5), 1);
        assert_eq!(floor_keep(2, 4, 5), 4);
        assert_eq!(floor_keep(3, 4, 5), 4);
        assert_eq!(floor_keep(4, 4, 5), 4);
        // floor 4: only width 3 rounds up; nothing to round to without more drafts.
        assert_eq!(floor_keep(2, 4, 4), 4);
        assert_eq!(floor_keep(3, 4, 4), 3);
        assert_eq!(floor_keep(2, 2, 5), 2);
    }

    #[test]
    fn parses_control_lines() {
        assert_eq!(
            parse_gate_control("0.6 4 1").unwrap(),
            (Some(0.6), Some(4), true)
        );
        assert_eq!(
            parse_gate_control("none all 0\n").unwrap(),
            (None, None, false)
        );
        assert_eq!(
            parse_gate_control(" 0.75  2 0 ").unwrap(),
            (Some(0.75), Some(2), false)
        );
        for bad in [
            "", "0.6 4", "1.0 4 1", "0 4 1", "0.6 0 1", "0.6 x 1", "0.6 4 2", "a b c d",
        ] {
            assert!(parse_gate_control(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn chain_len_is_clamped_to_the_physical_drafts() {
        let p = |m| GatePolicy {
            threshold: None,
            max_drafts: m,
            early_stop: false,
            log: false,
        };
        assert_eq!(p(None).chain_len(4), 4);
        assert_eq!(p(Some(2)).chain_len(4), 2);
        assert_eq!(p(Some(9)).chain_len(4), 4);
        assert_eq!(p(Some(1)).chain_len(4), 1);
        assert_eq!(p(Some(8)).chain_len(8), 8);
    }

    #[test]
    fn early_threshold_requires_both_settings() {
        let p = |t, e| GatePolicy {
            threshold: t,
            max_drafts: None,
            early_stop: e,
            log: false,
        };
        assert_eq!(p(Some(0.6), true).early_threshold(), Some(0.6));
        assert_eq!(p(Some(0.6), false).early_threshold(), None);
        assert_eq!(p(None, true).early_threshold(), None);
    }

    #[test]
    fn cohort_uniform_batch_policy_never_truncates_to_zero_under_gumbel() {
        // Under ATLAS_SPEC_VERIFY=gumbel with ATLAS_QWEN4_MTP_ALLOW_ZERO and
        // ATLAS_SPEC_ADAPTIVE_DEPTH unset, single-seq resolves BelowFloor(0.12).
        let gumbel_single = resolve_zero_draft_policy(None, None, Some("gumbel"), None);
        assert_eq!(
            gumbel_single,
            ZeroDraftPolicy::BelowFloor(DEFAULT_ZERO_DRAFT_THRESHOLD)
        );
        let batch_zero = batch_zero_draft_policy();
        assert_eq!(batch_zero, ZeroDraftPolicy::KeepAtLeastOne);

        // With ATLAS_QWEN4_MTP_CONF=0 (threshold: None) and no step/cum/log gate:
        let conf_off = GatePolicy {
            threshold: None,
            max_drafts: None,
            early_stop: false,
            log: false,
        };
        // Single-seq wants probabilities to enforce BelowFloor(0.12) and can truncate to 0.
        assert!(conf_off.wants_probabilities_with(None, None, gumbel_single));
        assert_eq!(
            conf_off.effective_base_threshold_with(None, None, gumbel_single),
            Some(0.0)
        );
        assert_eq!(
            confident_prefix_with(&[0.05, 0.9, 0.9, 0.9], 0.0, None, None, gumbel_single),
            0
        );

        // Batched cohort (b >= 2) skips probabilities (unlocking argmax_bf16_rows) and
        // keeps the full uniform draft width without truncating any sequence to 0 drafts.
        assert!(!conf_off.wants_probabilities_with(None, None, batch_zero));
        assert_eq!(
            conf_off.effective_base_threshold_with(None, None, batch_zero),
            None
        );
        assert_eq!(
            confident_prefix_with(&[0.05, 0.9, 0.9, 0.9], 0.0, None, None, batch_zero),
            4
        );

        // Even when an explicit confidence or cumulative threshold IS configured,
        // batched cohort drafting keeps at least 1 draft on low step-0 probability.
        assert_eq!(
            confident_prefix_with(&[0.01, 0.9, 0.9, 0.9], 0.6, None, Some(0.25), batch_zero),
            1
        );
        assert_eq!(
            early_stop_keep_with(0, 0.01, 0.01, 0.6, None, Some(0.25), batch_zero),
            Some(1)
        );
    }

    #[test]
    fn gumbel_cum_threshold_and_vocab_cap_resolve_cleanly() {
        // CumThresholdEnv parsing and resolution.
        assert_eq!(parse_cum_threshold_env(None), CumThresholdEnv::Unset);
        assert_eq!(
            parse_cum_threshold_env(Some("0")),
            CumThresholdEnv::Disabled
        );
        assert_eq!(
            parse_cum_threshold_env(Some("0.0")),
            CumThresholdEnv::Disabled
        );
        assert_eq!(
            parse_cum_threshold_env(Some("off")),
            CumThresholdEnv::Disabled
        );
        assert_eq!(
            parse_cum_threshold_env(Some("none")),
            CumThresholdEnv::Disabled
        );
        assert_eq!(
            parse_cum_threshold_env(Some("0.25")),
            CumThresholdEnv::Value(0.25)
        );
        assert_eq!(cum_threshold_for(CumThresholdEnv::Unset, false), None);
        assert_eq!(
            cum_threshold_for(CumThresholdEnv::Unset, true),
            Some(DEFAULT_GUMBEL_CUM_THRESHOLD)
        );
        assert_eq!(cum_threshold_for(CumThresholdEnv::Disabled, true), None);
        assert_eq!(
            cum_threshold_for(CumThresholdEnv::Value(0.25), false),
            Some(0.25)
        );
        assert_eq!(
            cum_threshold_for(CumThresholdEnv::Value(0.25), true),
            Some(0.25)
        );

        // Gumbel running-product with 0.0 per-step floor: p_0=0.45 survives step 0,
        // stops after step 1 when p_cum = 0.45 * 0.60 = 0.27 < 0.38.
        let floor = ZeroDraftPolicy::BelowFloor(DEFAULT_ZERO_DRAFT_THRESHOLD);
        let tau = Some(DEFAULT_GUMBEL_CUM_THRESHOLD);
        assert_eq!(
            early_stop_keep_with(0, 0.45, 0.45, 0.0, None, tau, floor),
            None
        );
        assert_eq!(
            early_stop_keep_with(1, 0.60, 0.45 * 0.60, 0.0, None, tau, floor),
            Some(1)
        );
        assert_eq!(
            confident_prefix_with(&[0.45, 0.60, 0.95, 0.95], 0.0, None, tau, floor),
            1
        );
        assert_eq!(
            confident_prefix_with(&[0.45, 0.90, 0.95, 0.95], 0.0, None, tau, floor),
            3
        );
        assert_eq!(
            confident_prefix_with(&[0.25, 0.90, 0.90, 0.90], 0.0, None, tau, floor),
            0
        );

        // Draft vocab cap raises fast-host default from 73_728 -> 98_304 under Gumbel.
        assert_eq!(
            resolve_draft_vocab_override(DraftVocabEnv::Unset, true, false),
            Some(DEFAULT_FAST_HOST_DRAFT_VOCAB)
        );
        assert_eq!(
            resolve_draft_vocab_override(DraftVocabEnv::Unset, true, true),
            Some(DEFAULT_GUMBEL_FAST_HOST_DRAFT_VOCAB)
        );
        assert_eq!(
            resolve_draft_vocab_override(DraftVocabEnv::Unset, false, true),
            None
        );
        assert_eq!(
            resolve_draft_vocab_override(DraftVocabEnv::Disabled, true, true),
            None
        );
        assert_eq!(
            resolve_draft_vocab_override(DraftVocabEnv::Rows(96_000), true, true),
            Some(96_000)
        );
    }
}
