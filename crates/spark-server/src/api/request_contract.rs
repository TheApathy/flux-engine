// SPDX-License-Identifier: AGPL-3.0-only

//! Request-lifecycle contract (stability action #4): the total timeout is
//! derived from the request's own size, and an over-long request is either
//! clamped to the context that is left or refused with a structured reason.
//!
//! Both decisions are pure functions over plain numbers so they can be
//! table-tested without a server; the handlers only read the environment and
//! apply the result.

/// Minimum decode rate (output tokens per second) used to derive a request's
/// decode timeout allowance.
pub(crate) const DEFAULT_MIN_DECODE_TOKS_PER_S: f64 = 8.0;
/// Prefill floor used for the prompt part of the allowance.
pub(crate) const DEFAULT_MIN_PREFILL_TOKS_PER_S: f64 = 200.0;
/// Fixed slack for queueing, load jitter and the first token.
pub(crate) const PREFILL_FIXED_ALLOWANCE_S: u64 = 30;
/// Hard ceiling on the derived total (an explicit env base may exceed it).
pub(crate) const DEFAULT_REQUEST_TIMEOUT_MAX_S: u64 = 7200;
/// No new committed token for this long ends the request. Off (0) until the
/// soak measures the inter-token gap distribution; operators opt in with
/// `FLUX_STALL_TIMEOUT=<s>`.
pub(crate) const DEFAULT_STALL_TIMEOUT_S: u64 = 0;
/// The smallest answer room an over-long request may be clamped to.
pub(crate) const DEFAULT_MIN_ANSWER_TOKENS: usize = 256;

/// Inputs of the timeout derivation that come from server configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TimeoutConfig {
    /// `--request-timeout` / `FLUX_REQUEST_TIMEOUT`; 0 disables the total.
    pub base_s: u32,
    pub min_decode_toks_per_s: f64,
    pub min_prefill_toks_per_s: f64,
    pub max_s: u64,
    /// 0 disables the stall timeout.
    pub stall_s: u64,
}

impl TimeoutConfig {
    pub(crate) fn with_base(base_s: u32) -> Self {
        Self {
            base_s,
            min_decode_toks_per_s: DEFAULT_MIN_DECODE_TOKS_PER_S,
            min_prefill_toks_per_s: DEFAULT_MIN_PREFILL_TOKS_PER_S,
            max_s: DEFAULT_REQUEST_TIMEOUT_MAX_S,
            stall_s: DEFAULT_STALL_TIMEOUT_S,
        }
    }

    /// Server base plus the `FLUX_*` overrides; malformed values fall back
    /// to the defaults with a WARN instead of failing the request.
    fn from_env(base_s: u32) -> Self {
        let d = Self::with_base(base_s);
        Self {
            base_s,
            min_decode_toks_per_s: positive_f64_env(
                "FLUX_MIN_DECODE_TOKS_PER_S",
                d.min_decode_toks_per_s,
            ),
            min_prefill_toks_per_s: positive_f64_env(
                "FLUX_MIN_PREFILL_TOKS_PER_S",
                d.min_prefill_toks_per_s,
            ),
            max_s: u64_env("FLUX_REQUEST_TIMEOUT_MAX", d.max_s),
            stall_s: stall_timeout_s(),
        }
    }
}

/// The process's timeout configuration for a server base. The env part is
/// read once (no per-request env reads or WARN storms on a typo).
pub(crate) fn timeout_config(base_s: u32) -> TimeoutConfig {
    static ENV: std::sync::OnceLock<TimeoutConfig> = std::sync::OnceLock::new();
    TimeoutConfig {
        base_s,
        ..*ENV.get_or_init(|| TimeoutConfig::from_env(0))
    }
}

/// Where the applied total came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimeoutSource {
    /// The request's own `timeout` field.
    Request,
    /// The server base (`FLUX_REQUEST_TIMEOUT`) was the larger term or disabled.
    Env,
    /// `max_tokens / floor + prefill allowance` exceeded the base.
    Derived,
}

impl TimeoutSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::Env => "env",
            Self::Derived => "derived",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TimeoutPolicy {
    /// Total wall-clock ceiling in seconds; `None` = no total ceiling.
    pub total_s: Option<f64>,
    /// Inter-token stall limit in seconds; `None` = disabled.
    pub stall_s: Option<u64>,
    pub source: TimeoutSource,
}

/// `ceil(max_tokens / decode_floor) + ceil(prompt / prefill_floor) + slack`,
/// capped at `max_s`.
pub(crate) fn derived_total_s(cfg: &TimeoutConfig, max_tokens: usize, prompt_len: usize) -> u64 {
    let decode = (max_tokens as f64 / cfg.min_decode_toks_per_s).ceil() as u64;
    let prefill = (prompt_len as f64 / cfg.min_prefill_toks_per_s).ceil() as u64;
    decode
        .saturating_add(prefill)
        .saturating_add(PREFILL_FIXED_ALLOWANCE_S)
        .min(cfg.max_s)
}

/// The timeout contract for one request. An explicit request `timeout`
/// wins as before (0 = none); a server base of 0 keeps "no total timeout";
/// otherwise the total is `max(base, derived)`.
pub(crate) fn derive_timeout_policy(
    cfg: &TimeoutConfig,
    explicit_s: Option<f32>,
    max_tokens: usize,
    prompt_len: usize,
) -> TimeoutPolicy {
    let stall_s = (cfg.stall_s > 0).then_some(cfg.stall_s);
    if let Some(explicit) = explicit_s {
        let total_s = (explicit > 0.0).then_some(f64::from(explicit));
        return TimeoutPolicy {
            total_s,
            stall_s,
            source: TimeoutSource::Request,
        };
    }
    if cfg.base_s == 0 {
        return TimeoutPolicy {
            total_s: None,
            stall_s,
            source: TimeoutSource::Env,
        };
    }
    let base = u64::from(cfg.base_s);
    let derived = derived_total_s(cfg, max_tokens, prompt_len);
    let (total, source) = if derived > base {
        (derived, TimeoutSource::Derived)
    } else {
        (base, TimeoutSource::Env)
    };
    TimeoutPolicy {
        total_s: Some(total as f64),
        stall_s,
        source,
    }
}

/// The absolute deadline for a policy's total, `None` when there is none or
/// it is out of the representable range.
pub(crate) fn policy_deadline(
    policy: &TimeoutPolicy,
    now: std::time::Instant,
) -> Option<std::time::Instant> {
    policy
        .total_s
        .and_then(|s| std::time::Duration::try_from_secs_f64(s).ok())
        .and_then(|d| now.checked_add(d))
}

/// Emit the one-per-request DEBUG line naming the applied policy.
pub(crate) fn log_timeout_policy(policy: &TimeoutPolicy) {
    tracing::debug!(
        "REQUEST_TIMEOUT_POLICY total_s={} stall_s={} source={}",
        policy
            .total_s
            .map_or_else(|| "none".to_string(), |s| format!("{s:.0}")),
        policy
            .stall_s
            .map_or_else(|| "none".to_string(), |s| s.to_string()),
        policy.source.as_str(),
    );
}

/// The process-wide stall limit (`FLUX_STALL_TIMEOUT`, default 0 = off),
/// read once. The scheduler consults it on every post-token check.
pub(crate) fn stall_timeout_s() -> u64 {
    static STALL: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *STALL.get_or_init(|| u64_env("FLUX_STALL_TIMEOUT", DEFAULT_STALL_TIMEOUT_S))
}

/// `flux_stop_reason` of a request ended by the stall timer.
pub(crate) const TIMEOUT_STALL_CAUSE: &str = "request_timeout_stall";
/// `flux_stop_reason` of a request ended by its total deadline.
pub(crate) const TIMEOUT_TOTAL_CAUSE: &str = "request_timeout_total";

/// Whether a scheduler stop cause is a request timeout. A timeout keeps
/// `finish_reason = "length"` (the output is truncated, not complete) and is
/// still reported as `flux_stop_reason` plus a `warnings` entry.
pub(crate) fn is_timeout_cause(cause: &str) -> bool {
    cause == TIMEOUT_STALL_CAUSE || cause == TIMEOUT_TOTAL_CAUSE
}

/// The client-visible warning for a timeout cause, `None` for other causes.
pub(crate) fn timeout_warning(cause: Option<&str>) -> Option<String> {
    cause.filter(|c| is_timeout_cause(c)).map(|c| {
        format!("{c}: the server ended generation at its request timeout; output is truncated")
    })
}

// ── Admission ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmissionConfig {
    /// `FLUX_CLAMP_MAX_TOKENS` (default off: byte-identical 400s; opt in with 1).
    pub clamp: bool,
    /// `FLUX_MIN_ANSWER_TOKENS` (default 256).
    pub min_answer_tokens: usize,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            clamp: false,
            min_answer_tokens: DEFAULT_MIN_ANSWER_TOKENS,
        }
    }
}

impl AdmissionConfig {
    /// The process's admission configuration, read from the env once.
    pub(crate) fn current() -> Self {
        static ENV: std::sync::OnceLock<AdmissionConfig> = std::sync::OnceLock::new();
        *ENV.get_or_init(Self::from_env)
    }

    fn from_env() -> Self {
        let d = Self::default();
        Self {
            clamp: bool_env("FLUX_CLAMP_MAX_TOKENS", d.clamp),
            min_answer_tokens: u64_env("FLUX_MIN_ANSWER_TOKENS", d.min_answer_tokens as u64)
                as usize,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum OutputAdmission {
    /// `prompt + max_tokens <= max_seq_len`: unchanged.
    Fits,
    /// Over-long; `max_tokens` lowered to the room left.
    Clamped { from: usize, to: usize },
    /// Clamping is on but the prompt leaves less than the minimum answer room.
    TooLittleRoom { room: usize },
    /// Over-long and clamping is disabled: the pre-contract 400, unchanged.
    Exceeds,
}

/// Decide what to do with `prompt + max_tokens` against `max_seq_len`.
pub(crate) fn admit_output(
    cfg: &AdmissionConfig,
    prompt_len: usize,
    max_tokens: usize,
    max_seq_len: usize,
) -> OutputAdmission {
    if super::context_budget::admitted_total(prompt_len, max_tokens, max_seq_len).is_some() {
        return OutputAdmission::Fits;
    }
    // Same inclusive bound as `admitted_total`: `prompt + room == max_seq_len`
    // is admitted, so `Fits` and `Clamped` agree at the boundary.
    let room = max_seq_len.saturating_sub(prompt_len);
    if !cfg.clamp {
        OutputAdmission::Exceeds
    } else if room < cfg.min_answer_tokens {
        OutputAdmission::TooLittleRoom { room }
    } else {
        OutputAdmission::Clamped {
            from: max_tokens,
            to: room,
        }
    }
}

/// The response-level warning a clamped request carries.
pub(crate) fn clamp_warning(
    from: usize,
    to: usize,
    prompt_len: usize,
    max_seq_len: usize,
) -> String {
    format!(
        "max_tokens_clamped: requested max_tokens={from} reduced to {to} because \
         prompt_tokens={prompt_len} + max_tokens exceeds max_seq_len={max_seq_len}"
    )
}

/// The structured 400 message when the prompt leaves too little room.
pub(crate) fn too_little_room_message(
    prompt_len: usize,
    max_tokens: usize,
    max_seq_len: usize,
    room: usize,
    min_answer_tokens: usize,
) -> String {
    format!(
        "Context too long: prompt_tokens={prompt_len} leaves room for {room} output tokens \
         (max_seq_len={max_seq_len}), below the minimum answer of \
         min_answer_tokens={min_answer_tokens}; requested max_tokens={max_tokens}. \
         Shorten the prompt or raise max_seq_len."
    )
}

// ── env helpers ──────────────────────────────────────────────────────

fn env_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn u64_env(name: &str, default: u64) -> u64 {
    env_value(name).map_or(default, |v| {
        v.parse().unwrap_or_else(|_| {
            tracing::warn!("{name}={v:?} is not a nonnegative integer; using {default}");
            default
        })
    })
}

fn positive_f64_env(name: &str, default: f64) -> f64 {
    env_value(name).map_or(default, |v| match v.parse::<f64>() {
        Ok(x) if x.is_finite() && x > 0.0 => x,
        _ => {
            tracing::warn!("{name}={v:?} is not a positive number; using {default}");
            default
        }
    })
}

fn bool_env(name: &str, default: bool) -> bool {
    env_value(name).map_or(default, |v| match v.to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => true,
        "0" | "false" | "off" | "no" => false,
        _ => {
            tracing::warn!("{name}={v:?} is not a boolean; using {default}");
            default
        }
    })
}

#[cfg(test)]
#[path = "request_contract_tests.rs"]
mod tests;
