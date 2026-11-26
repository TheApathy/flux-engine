// SPDX-License-Identifier: AGPL-3.0-only

//! Per-identity token-bucket rate limiter.
//!
//! Two independent buckets per key (identity): one metered in **requests**,
//! one in **tokens**. Each bucket refills linearly toward its cap based on
//! elapsed wall time.
//!
//! Identity resolution order (first match wins):
//! 1. Presented API key (`Authorization: Bearer <token>`, case-insensitive, or
//!    `x-api-key: <token>`) — aligned with [`crate::auth::extract_presented_token`].
//! 2. First entry of `X-Forwarded-For` — ONLY when proxy trust is explicitly
//!    enabled via `FLUX_TRUST_PROXY=1` (or `ATLAS_TRUST_PROXY=1`). Disabled by
//!    default so callers cannot spoof headers to bypass rate limits.
//! 3. Socket peer address — fallback for unauthenticated direct calls.
//!
//! Env configuration (all default 0 = disabled, pure passthrough):
//!   ATLAS_RATE_LIMIT_RPM       — requests per minute cap
//!   ATLAS_RATE_LIMIT_TPM       — tokens per minute cap
//!   ATLAS_RATE_LIMIT_BURST_RPM — max request burst (default = RPM)
//!   ATLAS_RATE_LIMIT_BURST_TPM — max token burst   (default = TPM)
//!   FLUX_TRUST_PROXY           — set to 1/true to honor X-Forwarded-For (default off)
//!
//! The limiter keeps the static "effectively unlimited" headers
//! byte-for-byte when both RPM and TPM are 0 so existing deployments see no
//! behavior change.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// Configuration for the limiter. Zero means "disabled" for that bucket.
#[derive(Clone, Copy, Debug)]
pub struct RateLimitConfig {
    pub rpm: u64,
    pub tpm: u64,
    pub burst_rpm: u64,
    pub burst_tpm: u64,
}

impl RateLimitConfig {
    pub fn from_env() -> Self {
        let rpm = read_env_u64("ATLAS_RATE_LIMIT_RPM", 0);
        let tpm = read_env_u64("ATLAS_RATE_LIMIT_TPM", 0);
        let burst_rpm = read_env_u64("ATLAS_RATE_LIMIT_BURST_RPM", rpm);
        let burst_tpm = read_env_u64("ATLAS_RATE_LIMIT_BURST_TPM", tpm);
        Self {
            rpm,
            tpm,
            burst_rpm: burst_rpm.max(1),
            burst_tpm: burst_tpm.max(1),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.rpm > 0 || self.tpm > 0
    }
}

fn read_env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Snapshot of a bucket's remaining budget — used to populate the
/// `x-ratelimit-*-remaining` / `-reset` response headers.
#[derive(Clone, Copy, Debug)]
pub struct BucketSnapshot {
    pub limit: u64,
    pub remaining: u64,
    /// Seconds until this bucket fully refills.
    pub reset_secs: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct RateDecision {
    pub allowed: bool,
    pub requests: BucketSnapshot,
    pub tokens: BucketSnapshot,
    /// Seconds client should wait before retrying (Retry-After header).
    /// Zero when `allowed`.
    pub retry_after_secs: u64,
    /// Which bucket triggered the denial — used for the error message.
    pub denied_by: Option<DenialReason>,
}

#[derive(Clone, Copy, Debug)]
pub enum DenialReason {
    Requests,
    Tokens,
}

/// Per-request context carried from the rate-limit middleware into the
/// handler so the streaming true-up can refund over-estimated tokens
/// once the actual usage is known. Injected into
/// `Request::extensions_mut()` by the middleware when the limiter is
/// enabled; extracted by handlers that want to refund.
#[derive(Clone, Debug)]
pub struct RequestContext {
    pub identity: String,
    /// Tokens reserved at admission time (conservative upper bound).
    pub reserved_tokens: u64,
    /// Shared settlement latch so either the handler or the middleware
    /// disconnect/error guard refunds tokens, never both.
    settled: Arc<AtomicBool>,
}

impl RequestContext {
    pub fn new(identity: String, reserved_tokens: u64) -> Self {
        Self {
            identity,
            reserved_tokens,
            settled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Atomically claim settlement of this request's token reservation.
    /// Returns `true` for the first caller and `false` thereafter so
    /// handler true-up and middleware disconnect/error refunds never
    /// double-refund.
    pub fn mark_settled(&self) -> bool {
        !self.settled.swap(true, Ordering::AcqRel)
    }
}

struct Bucket {
    /// Tokens/requests currently available. f64 for fractional refill.
    available: f64,
    /// Last refill tick.
    last_refill: Instant,
}

impl Bucket {
    fn new(burst: u64) -> Self {
        Self {
            available: burst as f64,
            last_refill: Instant::now(),
        }
    }

    /// Refill based on `rate_per_sec`, then try to debit `cost`. Returns
    /// true if the debit succeeded. Caps at `burst`.
    fn try_consume(&mut self, cost: f64, rate_per_sec: f64, burst: f64, now: Instant) -> bool {
        let dt = now
            .saturating_duration_since(self.last_refill)
            .as_secs_f64();
        if dt > 0.0 {
            self.available = (self.available + dt * rate_per_sec).min(burst);
            self.last_refill = now;
        }
        if self.available >= cost {
            self.available -= cost;
            true
        } else {
            false
        }
    }

    fn snapshot(&self, rate_per_sec: f64, burst: f64, now: Instant) -> (f64, u64) {
        let dt = now
            .saturating_duration_since(self.last_refill)
            .as_secs_f64();
        let available = (self.available + dt * rate_per_sec).min(burst);
        // Seconds until fully refilled.
        let deficit = (burst - available).max(0.0);
        let reset = if rate_per_sec > 0.0 {
            (deficit / rate_per_sec).ceil() as u64
        } else {
            0
        };
        (available, reset)
    }

    /// Add tokens back (used when a streaming request admits with a
    /// reservation and actual consumption was lower).
    fn refund(&mut self, amount: f64, burst: f64) {
        self.available = (self.available + amount).min(burst);
    }
}

struct KeyState {
    requests: Bucket,
    tokens: Bucket,
}

/// Shared concurrent rate-limiter state.
pub struct RateLimiter {
    cfg: RateLimitConfig,
    inner: Mutex<HashMap<String, KeyState>>,
    /// Last-scrubbed timestamp for idle-entry cleanup. We sweep once per
    /// `SCRUB_INTERVAL` on the admission hot path.
    last_scrub: Mutex<Instant>,
    /// Upper bound on tracked keys before failing closed to prevent OOM and
    /// key-rotation bypass attacks.
    max_keys: usize,
    /// Whether `X-Forwarded-For` is trusted when resolving client identity.
    trust_proxy: bool,
}

const SCRUB_INTERVAL: Duration = Duration::from_secs(120);
/// Keys with no requests for this long are dropped from the map.
const IDLE_EVICT: Duration = Duration::from_secs(600);
/// Cap the per-key map at this size. Prevents OOM under a DoS where a
/// malicious client rotates headers every request to balloon the map past
/// `SCRUB_INTERVAL`. When the cap is hit, force an out-of-band scrub and
/// fail closed for unknown overflow keys rather than opening the limiter.
const MAX_KEYS: usize = 100_000;

fn read_trust_proxy_env() -> bool {
    std::env::var("FLUX_TRUST_PROXY")
        .or_else(|_| std::env::var("ATLAS_TRUST_PROXY"))
        .ok()
        .is_some_and(|v| {
            let trimmed = v.trim();
            trimmed == "1" || trimmed.eq_ignore_ascii_case("true")
        })
}

impl RateLimiter {
    pub fn from_env() -> Arc<Self> {
        Arc::new(Self {
            cfg: RateLimitConfig::from_env(),
            inner: Mutex::new(HashMap::new()),
            last_scrub: Mutex::new(Instant::now()),
            max_keys: MAX_KEYS,
            trust_proxy: read_trust_proxy_env(),
        })
    }

    pub fn trust_proxy(&self) -> bool {
        self.trust_proxy
    }

    #[cfg(test)]
    pub fn with_config(cfg: RateLimitConfig) -> Arc<Self> {
        Self::with_config_and_max_keys(cfg, MAX_KEYS)
    }

    #[cfg(test)]
    pub fn with_config_and_max_keys(cfg: RateLimitConfig, max_keys: usize) -> Arc<Self> {
        Arc::new(Self {
            cfg,
            inner: Mutex::new(HashMap::new()),
            last_scrub: Mutex::new(Instant::now()),
            max_keys,
            trust_proxy: false,
        })
    }

    #[cfg(test)]
    pub fn key_count(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn config(&self) -> RateLimitConfig {
        self.cfg
    }

    /// Admission check for a new request. `estimated_tokens` is the
    /// caller's best guess at how many tokens this call will burn
    /// (prompt + max completion). Streaming callers pass their worst-case
    /// so the reservation is conservative; on completion they call
    /// `refund_tokens` with the true-up.
    pub fn admit(&self, key: &str, estimated_tokens: u64) -> RateDecision {
        let now = Instant::now();
        self.scrub_if_due(now);

        let rpm = self.cfg.rpm;
        let tpm = self.cfg.tpm;

        // When disabled, return effectively-unlimited snapshots.
        if !self.cfg.is_enabled() {
            return RateDecision {
                allowed: true,
                requests: BucketSnapshot {
                    limit: 1_000_000,
                    remaining: 999_999,
                    reset_secs: 0,
                },
                tokens: BucketSnapshot {
                    limit: 1_000_000_000,
                    remaining: 999_999_999,
                    reset_secs: 0,
                },
                retry_after_secs: 0,
                denied_by: None,
            };
        }

        let req_rate = rpm as f64 / 60.0;
        let tok_rate = tpm as f64 / 60.0;
        let req_burst = self.cfg.burst_rpm as f64;
        let tok_burst = self.cfg.burst_tpm as f64;

        let mut map = self.inner.lock();
        // DoS guard: if the map has grown past MAX_KEYS without natural
        // scrubbing kicking in, force-evict every idle entry now. The
        // periodic scrub still runs every SCRUB_INTERVAL on its own.
        if map.len() >= self.max_keys && !map.contains_key(key) {
            map.retain(|_, state| {
                state.requests.last_refill.elapsed() < IDLE_EVICT
                    || state.tokens.last_refill.elapsed() < IDLE_EVICT
            });
            // If the cap is still hit (every tracked key is actively within
            // IDLE_EVICT), fail closed for untracked overflow keys rather than
            // allowing key-rotation floods to disable rate limiting or evict
            // existing tracked clients' buckets.
            if map.len() >= self.max_keys {
                return RateDecision {
                    allowed: false,
                    requests: BucketSnapshot {
                        limit: self.cfg.burst_rpm,
                        remaining: 0,
                        reset_secs: 60,
                    },
                    tokens: BucketSnapshot {
                        limit: self.cfg.burst_tpm,
                        remaining: 0,
                        reset_secs: 60,
                    },
                    retry_after_secs: 60,
                    denied_by: Some(DenialReason::Requests),
                };
            }
        }
        let state = map.entry(key.to_string()).or_insert_with(|| KeyState {
            requests: Bucket::new(self.cfg.burst_rpm),
            tokens: Bucket::new(self.cfg.burst_tpm),
        });

        // Request bucket.
        let req_allowed = if rpm > 0 {
            state.requests.try_consume(1.0, req_rate, req_burst, now)
        } else {
            true
        };
        if !req_allowed {
            let (req_avail, req_reset) = state.requests.snapshot(req_rate, req_burst, now);
            let (tok_avail, tok_reset) = state.tokens.snapshot(tok_rate, tok_burst, now);
            return RateDecision {
                allowed: false,
                requests: BucketSnapshot {
                    limit: self.cfg.burst_rpm,
                    remaining: req_avail.max(0.0) as u64,
                    reset_secs: req_reset,
                },
                tokens: BucketSnapshot {
                    limit: self.cfg.burst_tpm,
                    remaining: tok_avail.max(0.0) as u64,
                    reset_secs: tok_reset,
                },
                retry_after_secs: req_reset.max(1),
                denied_by: Some(DenialReason::Requests),
            };
        }

        // Token bucket. Zero-cost debit when TPM is disabled.
        let tok_allowed = if tpm > 0 {
            state
                .tokens
                .try_consume(estimated_tokens as f64, tok_rate, tok_burst, now)
        } else {
            true
        };
        if !tok_allowed {
            // Refund the request we just consumed — we're going to deny.
            if rpm > 0 {
                state.requests.refund(1.0, req_burst);
            }
            let (req_avail, req_reset) = state.requests.snapshot(req_rate, req_burst, now);
            let (tok_avail, tok_reset) = state.tokens.snapshot(tok_rate, tok_burst, now);
            return RateDecision {
                allowed: false,
                requests: BucketSnapshot {
                    limit: self.cfg.burst_rpm,
                    remaining: req_avail.max(0.0) as u64,
                    reset_secs: req_reset,
                },
                tokens: BucketSnapshot {
                    limit: self.cfg.burst_tpm,
                    remaining: tok_avail.max(0.0) as u64,
                    reset_secs: tok_reset,
                },
                retry_after_secs: tok_reset.max(1),
                denied_by: Some(DenialReason::Tokens),
            };
        }

        let (req_avail, req_reset) = state.requests.snapshot(req_rate, req_burst, now);
        let (tok_avail, tok_reset) = state.tokens.snapshot(tok_rate, tok_burst, now);
        RateDecision {
            allowed: true,
            requests: BucketSnapshot {
                limit: self.cfg.burst_rpm,
                remaining: req_avail.max(0.0) as u64,
                reset_secs: req_reset,
            },
            tokens: BucketSnapshot {
                limit: self.cfg.burst_tpm,
                remaining: tok_avail.max(0.0) as u64,
                reset_secs: tok_reset,
            },
            retry_after_secs: 0,
            denied_by: None,
        }
    }

    /// Called after a streaming request completes with `(reserved - actual)`
    /// so over-estimated reservations don't burn the token bucket forever.
    pub fn refund_tokens(&self, key: &str, amount: u64) {
        if amount == 0 || !self.cfg.is_enabled() || self.cfg.tpm == 0 {
            return;
        }
        let mut map = self.inner.lock();
        if let Some(state) = map.get_mut(key) {
            state
                .tokens
                .refund(amount as f64, self.cfg.burst_tpm as f64);
        }
    }

    fn scrub_if_due(&self, now: Instant) {
        let mut last = self.last_scrub.lock();
        if now.saturating_duration_since(*last) < SCRUB_INTERVAL {
            return;
        }
        *last = now;
        drop(last);
        let mut map = self.inner.lock();
        map.retain(|_, state| {
            state.requests.last_refill.elapsed() < IDLE_EVICT
                || state.tokens.last_refill.elapsed() < IDLE_EVICT
        });
    }
}

/// Resolve a stable identity from request headers and peer address without
/// trusting spoofable `X-Forwarded-For` headers.
pub fn extract_identity(
    headers: &axum::http::HeaderMap,
    peer: Option<std::net::SocketAddr>,
) -> String {
    extract_identity_with_proxy(headers, peer, false)
}

/// Resolve a stable identity from request headers and peer address.
///
/// Precedence:
/// 1. Presented API key via [`crate::auth::extract_presented_token`] (handles
///    case-insensitive `Authorization: Bearer <token>` and `x-api-key: <token>`).
/// 2. First entry of `X-Forwarded-For` ONLY when `trust_proxy` is `true`.
/// 3. Socket peer IP address.
pub fn extract_identity_with_proxy(
    headers: &axum::http::HeaderMap,
    peer: Option<std::net::SocketAddr>,
    trust_proxy: bool,
) -> String {
    // 1. Presented API key (Bearer or x-api-key), aligned with auth.rs.
    if let Some(tok) = crate::auth::extract_presented_token(headers) {
        // Hash the token so we don't retain sensitive data in map keys or logs.
        return format!("bearer:{}", hash_token(tok));
    }
    // 2. X-Forwarded-For (only when explicitly configured behind a trusted proxy).
    if trust_proxy
        && let Some(xff) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok())
        && let Some(first) = xff.split(',').next()
    {
        let first = first.trim();
        if !first.is_empty() {
            return format!("xff:{first}");
        }
    }
    // 3. Peer socket.
    match peer {
        Some(addr) => format!("peer:{}", addr.ip()),
        None => "peer:unknown".to_string(),
    }
}

/// FNV-1a 64-bit hash. Avoids pulling a crypto dep; unnecessary here since
/// we only need a stable opaque label, not collision resistance.
fn hash_token(tok: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in tok.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

#[cfg(test)]
#[path = "rate_limiter/tests.rs"]
mod tests;
