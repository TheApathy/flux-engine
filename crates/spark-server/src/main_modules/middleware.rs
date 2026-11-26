// SPDX-License-Identifier: AGPL-3.0-only

//! Axum middleware: observability, auth, rate-limiting.

use std::sync::Arc;

use crate::{openai, rate_limiter};

/// OpenAI-compatible observability headers. Injects on every `/v1/*`
/// response:
/// - `x-request-id`: UUID v4 generated per-request (re-used if client
///   supplied one, letting callers correlate logs end-to-end).
/// - `openai-processing-ms`: server wall-clock time in milliseconds.
/// - `x-ratelimit-*`: static "unlimited" stubs so clients that
///   parse them for backoff don't treat missing headers as unlimited
///   (some SDKs assume 0 = exhausted).
/// - `openai-organization`, `openai-version`: static stubs for
///   parity with `api.openai.com` — several wrappers log these.
pub(crate) fn apply_observability_headers(
    headers: &mut axum::http::HeaderMap,
    incoming_req_id: Option<String>,
    elapsed_ms: u128,
) {
    use axum::http::{HeaderName, HeaderValue};

    let rid = incoming_req_id.unwrap_or_else(|| format!("req_{}", openai::uuid_v4()));
    if let Ok(v) = HeaderValue::from_str(&rid) {
        headers.insert(HeaderName::from_static("x-request-id"), v);
    }
    if let Ok(v) = HeaderValue::from_str(&elapsed_ms.to_string()) {
        headers.insert(HeaderName::from_static("openai-processing-ms"), v);
    }
    // Static "effectively unlimited" fallbacks when rate limiting is disabled.
    // Never overwrite real headers already set by `rate_limit_middleware`
    // (including on 429 responses).
    for (k, v) in [
        ("x-ratelimit-limit-requests", "1000000"),
        ("x-ratelimit-remaining-requests", "999999"),
        ("x-ratelimit-reset-requests", "0s"),
        ("x-ratelimit-limit-tokens", "1000000000"),
        ("x-ratelimit-remaining-tokens", "999999999"),
        ("x-ratelimit-reset-tokens", "0s"),
        ("openai-organization", "atlas-local"),
        ("openai-version", "2026-01-01"),
    ] {
        let name = HeaderName::from_static(k);
        if !headers.contains_key(&name) {
            headers.insert(name, HeaderValue::from_static(v));
        }
    }
}

pub(crate) async fn openai_observability_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let start = std::time::Instant::now();
    let is_v1 = req.uri().path().starts_with("/v1/");
    let incoming_req_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let mut resp = next.run(req).await;
    if !is_v1 {
        return resp;
    }
    apply_observability_headers(
        resp.headers_mut(),
        incoming_req_id,
        start.elapsed().as_millis(),
    );
    resp
}

/// Bearer & `x-api-key` scoped authentication gate. Active whenever any auth
/// token source (`--auth-token`, `--api-key`, `--auth-tokens-file`,
/// `FLUX_API_KEY`, or `--require-auth`) is configured on the `ModelHost`.
///
/// Enforces:
/// - `read` scope on GET telemetry, model discovery (`/v1/models`, `/api/tags`, `/api/show`), and stored item reads.
/// - `inference` scope on generation, tokenization, and stateful session writes.
/// - `admin` scope on mutating admin/settings/swap endpoints.
///
/// Health, metrics, and UI landing endpoints remain open as scrape targets.
pub(crate) async fn require_auth_middleware(
    // From the HOST, not from `AppState`. Read off the model, the policy
    // ceases to exist whenever no model is loaded — so every `/v1/*` request
    // during a swap would sail past the gate. Whether a request is authorised
    // cannot depend on whether a model happens to be resident.
    axum::extract::State(host): axum::extract::State<
        Arc<crate::main_modules::model_host::ModelHost>,
    >,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let Some(auth_cfg) = host.auth() else {
        return next.run(req).await;
    };
    match evaluate_auth(&auth_cfg, req.method(), req.uri().path(), req.headers()) {
        Ok(()) => next.run(req).await,
        Err((status, body)) => (status, axum::Json(body)).into_response(),
    }
}

pub(crate) fn evaluate_auth(
    auth_cfg: &crate::auth::AuthConfig,
    method: &axum::http::Method,
    path: &str,
    headers: &axum::http::HeaderMap,
) -> Result<(), (axum::http::StatusCode, serde_json::Value)> {
    let Some(required) = crate::auth::required_scope(method, path) else {
        return Ok(());
    };
    let Some(presented) = crate::auth::extract_presented_token(headers) else {
        return Err((
            axum::http::StatusCode::UNAUTHORIZED,
            serde_json::json!({
                "error": {
                    "message": "Missing Authorization: Bearer or x-api-key header",
                    "type": "invalid_request_error",
                    "param": null,
                    "code": "missing_api_key",
                }
            }),
        ));
    };
    match auth_cfg.authorize(presented.as_bytes(), required) {
        crate::auth::AuthOutcome::Allowed => Ok(()),
        crate::auth::AuthOutcome::InvalidToken => Err((
            axum::http::StatusCode::UNAUTHORIZED,
            serde_json::json!({
                "error": {
                    "message": "Invalid bearer token or API key",
                    "type": "invalid_request_error",
                    "param": null,
                    "code": "invalid_api_key",
                }
            }),
        )),
        crate::auth::AuthOutcome::Forbidden { required } => Err((
            axum::http::StatusCode::FORBIDDEN,
            serde_json::json!({
                "error": {
                    "message": format!(
                        "API key lacks required '{}' scope for {} {}",
                        required.as_str(),
                        method,
                        path
                    ),
                    "type": "permission_error",
                    "param": null,
                    "code": "insufficient_scope",
                }
            }),
        )),
    }
}

/// Conservative TPM token reservation for an incoming `/v1/*` request.
///
/// Only `POST` calls to generation routes reserve `max_seq_len`; read-only
/// methods (`GET`/`HEAD`/`DELETE`/`OPTIONS`) and non-generation `/v1/*` routes
/// reserve `0` tokens while still participating in RPM accounting.
pub(crate) fn estimated_tokens_for_request(
    method: &axum::http::Method,
    path: &str,
    max_seq_len: Option<usize>,
) -> u64 {
    if super::mem_watch::is_generation_request(method, path) {
        max_seq_len.unwrap_or(0) as u64
    } else {
        0
    }
}

/// RAII guard for a per-request TPM token reservation.
///
/// Refunds `reserved_tokens` if the downstream handler future is dropped
/// (client disconnect before completion) or if the response status is
/// 4xx/5xx and the handler never settled actual usage via
/// [`rate_limiter::RequestContext::mark_settled`].
struct ReservationGuard {
    limiter: Arc<rate_limiter::RateLimiter>,
    ctx: rate_limiter::RequestContext,
    armed: bool,
}

impl ReservationGuard {
    fn new(limiter: Arc<rate_limiter::RateLimiter>, ctx: rate_limiter::RequestContext) -> Self {
        let armed = ctx.reserved_tokens > 0;
        Self {
            limiter,
            ctx,
            armed,
        }
    }

    fn finish(mut self, status: axum::http::StatusCode) {
        if !status.is_client_error() && !status.is_server_error() {
            // 2xx/3xx (including streaming SSE handoff): handler owns true-up.
            self.armed = false;
        }
        // On 4xx/5xx, stay armed so Drop refunds `reserved_tokens` iff the
        // handler never marked usage as settled.
    }
}

impl Drop for ReservationGuard {
    fn drop(&mut self) {
        if self.armed && self.ctx.mark_settled() {
            self.limiter
                .refund_tokens(&self.ctx.identity, self.ctx.reserved_tokens);
        }
    }
}

/// Per-identity rate-limit middleware. When the limiter is enabled via
/// `ATLAS_RATE_LIMIT_RPM` / `ATLAS_RATE_LIMIT_TPM`, admission is checked
/// before dispatching to the handler; denied requests short-circuit with
/// 429 + OpenAI error body + `retry-after` header.
///
/// Identity precedence (from `rate_limiter::extract_identity`):
/// Bearer / `x-api-key` token → `X-Forwarded-For` (only when `FLUX_TRUST_PROXY=1`) → socket peer addr.
///
/// Headers emitted in both allow and deny paths:
///   x-ratelimit-limit-{requests,tokens}
///   x-ratelimit-remaining-{requests,tokens}
///   x-ratelimit-reset-{requests,tokens}
///
/// `openai_observability_middleware` preserves these headers when present and
/// only injects static "unlimited" fallbacks when the limiter is disabled.
pub(crate) async fn rate_limit_middleware(
    // Also from the HOST, and for the same reason as `require_auth_middleware`:
    // read off the model, the limiter does not exist during a swap and every
    // request in that window bypasses it entirely. It is also the SAME `Arc`
    // the model's `AppState` holds, so handler-side `refund_tokens` credits the
    // bucket this middleware debited.
    axum::extract::State(host): axum::extract::State<
        Arc<crate::main_modules::model_host::ModelHost>,
    >,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::{HeaderName, HeaderValue, StatusCode};
    use axum::response::IntoResponse;

    // Apply to /v1/* routes plus /tokenize and /detokenize (which accept
    // up to 128 MiB bodies); health/metrics stay open.
    let path = req.uri().path();
    let is_rate_limited = path.starts_with("/v1/") || path == "/tokenize" || path == "/detokenize";
    let Some(limiter) = host.rate_limiter() else {
        // Installed before the listener binds, so unreachable in a running
        // server; passing through rather than 500-ing if that ever changes.
        return next.run(req).await;
    };
    if !is_rate_limited || !limiter.config().is_enabled() {
        return next.run(req).await;
    }

    // Peer addr comes from the ConnectInfo extension injected by
    // `into_make_service_with_connect_info`. Falls through to None when
    // running under a unit-test harness that doesn't set it.
    let peer = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0);
    let identity =
        rate_limiter::extract_identity_with_proxy(req.headers(), peer, limiter.trust_proxy());

    // Estimated token cost = this request's conservative upper bound.
    // We use max_seq_len as the ceiling on POST generation routes; true-up is
    // applied on completion (handlers call `refund_tokens` with the actual
    // usage). Non-generation routes and GET/HEAD/DELETE/OPTIONS reserve 0.
    //
    // `max_seq_len` belongs to the MODEL, and during a swap there is none. A
    // request arriving then cannot consume tokens from a model that is not
    // there — it is about to be rejected with 503 by `CurrentModel` — so the
    // ceiling is 0 and it is charged a request but no tokens. Skipping the
    // limiter entirely in that window would make a swap a free pass.
    let estimated = estimated_tokens_for_request(
        req.method(),
        req.uri().path(),
        host.current().map(|s| s.max_seq_len),
    );

    let decision = limiter.admit(&identity, estimated);
    if !decision.allowed {
        let (param, code) = match decision.denied_by {
            Some(rate_limiter::DenialReason::Requests) => ("requests", "rate_limit_exceeded"),
            Some(rate_limiter::DenialReason::Tokens) => ("tokens", "rate_limit_exceeded"),
            None => ("", "rate_limit_exceeded"),
        };
        let body = serde_json::json!({
            "error": {
                "message": format!("Rate limit exceeded for {param}. Retry after {}s.", decision.retry_after_secs),
                "type": "rate_limit_exceeded",
                "param": param,
                "code": code,
            }
        });
        let mut resp = (StatusCode::TOO_MANY_REQUESTS, axum::Json(body)).into_response();
        apply_rate_headers(resp.headers_mut(), &decision);
        if let Ok(v) = HeaderValue::from_str(&decision.retry_after_secs.to_string()) {
            resp.headers_mut()
                .insert(HeaderName::from_static("retry-after"), v);
        }
        return resp;
    }

    // Stash the identity + reservation on the request so handlers can
    // refund the over-estimated portion once the true usage is known.
    // `reservation_guard` refunds the full reservation if the client
    // disconnects before handler completion or if the handler returns a
    // 4xx/5xx error without settling usage.
    let req_ctx = rate_limiter::RequestContext::new(identity, estimated);
    let reservation_guard = ReservationGuard::new(limiter, req_ctx.clone());
    let mut req = req;
    req.extensions_mut().insert(req_ctx);
    let mut resp = next.run(req).await;
    reservation_guard.finish(resp.status());
    apply_rate_headers(resp.headers_mut(), &decision);
    resp
}

pub(crate) fn apply_rate_headers(
    headers: &mut axum::http::HeaderMap,
    d: &rate_limiter::RateDecision,
) {
    use axum::http::{HeaderName, HeaderValue};
    let set = |h: &mut axum::http::HeaderMap, k: &'static str, v: String| {
        if let Ok(val) = HeaderValue::from_str(&v) {
            h.insert(HeaderName::from_static(k), val);
        }
    };
    set(
        headers,
        "x-ratelimit-limit-requests",
        d.requests.limit.to_string(),
    );
    set(
        headers,
        "x-ratelimit-remaining-requests",
        d.requests.remaining.to_string(),
    );
    set(
        headers,
        "x-ratelimit-reset-requests",
        format!("{}s", d.requests.reset_secs),
    );
    set(
        headers,
        "x-ratelimit-limit-tokens",
        d.tokens.limit.to_string(),
    );
    set(
        headers,
        "x-ratelimit-remaining-tokens",
        d.tokens.remaining.to_string(),
    );
    set(
        headers,
        "x-ratelimit-reset-tokens",
        format!("{}s", d.tokens.reset_secs),
    );
}

/// Build the truthful response for an inference surface whose sole scheduler
/// is permanently unavailable. Health and metrics remain reachable so the
/// failure stays observable.
pub(crate) fn fault_rejection(
    path: &str,
    fault: Option<&str>,
) -> Option<(axum::http::StatusCode, serde_json::Value)> {
    if !path.starts_with("/v1/") {
        return None;
    }
    let reason = fault?;
    Some((
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        serde_json::json!({
            "error": {
                "message": format!("The server cannot process inference requests: {reason}"),
                "type": "server_error",
                "code": "server_fault",
            }
        }),
    ))
}

/// Reject new inference work after a fatal scheduler/process fault instead of
/// admitting it to a channel that no execution thread can ever drain.
pub(crate) async fn fault_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    match fault_rejection(req.uri().path(), atlas_core::fault::global().fault()) {
        None => next.run(req).await,
        Some((status, body)) => (status, axum::Json(body)).into_response(),
    }
}

#[cfg(test)]
#[path = "middleware_fault_tests.rs"]
mod fault_tests;
