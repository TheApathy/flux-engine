// SPDX-License-Identifier: AGPL-3.0-only

const FAULT: &str = "scheduler thread panicked (fixture)";

#[test]
fn faulted_inference_is_refused_with_truthful_503() {
    let (status, body) = super::fault_rejection("/v1/chat/completions", Some(FAULT))
        .expect("faulted inference must be rejected");
    assert_eq!(status, axum::http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "server_fault");
    assert_eq!(body["error"]["type"], "server_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains(FAULT))
    );
}

#[test]
fn healthy_inference_is_admitted() {
    assert!(super::fault_rejection("/v1/chat/completions", None).is_none());
}

#[test]
fn operational_endpoints_remain_visible_during_a_fault() {
    for path in ["/health", "/health/live", "/metrics"] {
        assert!(
            super::fault_rejection(path, Some(FAULT)).is_none(),
            "{path} must remain observable"
        );
    }
}

#[test]
fn scoped_auth_accepts_bearer_and_x_api_key_and_rejects_insufficient_scope() {
    use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("tokens.txt");
    std::fs::write(
        &path,
        "read:read-secret\ninference:inf-secret\nadmin:adm-secret\nlegacy-unprefixed\n",
    )
    .expect("write tokens");
    let cfg = crate::auth::AuthConfig::from_file(&path).expect("parse tokens");

    let empty = HeaderMap::new();
    let (status, body) =
        super::evaluate_auth(&cfg, &Method::GET, "/v1/models", &empty).unwrap_err();
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["code"], "missing_api_key");

    // Health stays open without headers.
    assert!(super::evaluate_auth(&cfg, &Method::GET, "/health", &empty).is_ok());

    // Read token via x-api-key passes GET /v1/memory, fails POST /v1/messages (403).
    let mut read_headers = HeaderMap::new();
    read_headers.insert("x-api-key", HeaderValue::from_static("read-secret"));
    assert!(super::evaluate_auth(&cfg, &Method::GET, "/v1/memory", &read_headers).is_ok());
    let (status, body) =
        super::evaluate_auth(&cfg, &Method::POST, "/v1/messages", &read_headers).unwrap_err();
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["error"]["code"], "insufficient_scope");

    // Inference token passes POST /v1/messages, fails PATCH /v1/flux/settings (403).
    let mut inf_headers = HeaderMap::new();
    inf_headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer inf-secret"),
    );
    assert!(super::evaluate_auth(&cfg, &Method::POST, "/v1/messages", &inf_headers).is_ok());
    let (status, _) =
        super::evaluate_auth(&cfg, &Method::PATCH, "/v1/flux/settings", &inf_headers).unwrap_err();
    assert_eq!(status, StatusCode::FORBIDDEN);

    // Unprefixed legacy token retains full Admin + Inference + Read scope.
    let mut legacy_headers = HeaderMap::new();
    legacy_headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer legacy-unprefixed"),
    );
    assert!(
        super::evaluate_auth(&cfg, &Method::PATCH, "/v1/flux/settings", &legacy_headers).is_ok()
    );
}

#[test]
fn observability_headers_preserve_real_rate_limit_headers_and_fill_fallbacks() {
    use axum::http::HeaderMap;

    let lim = crate::rate_limiter::RateLimiter::with_config(crate::rate_limiter::RateLimitConfig {
        rpm: 1,
        tpm: 1000,
        burst_rpm: 1,
        burst_tpm: 1000,
    });
    assert!(lim.admit("k", 250).allowed);
    let denied = lim.admit("k", 250);
    assert!(!denied.allowed);

    fn hdr<'a>(h: &'a HeaderMap, k: &str) -> Option<&'a str> {
        h.get(k).and_then(|v| v.to_str().ok())
    }

    // Real headers from rate_limit_middleware (including on 429) must win.
    let mut headers = HeaderMap::new();
    super::apply_rate_headers(&mut headers, &denied);
    super::apply_observability_headers(&mut headers, Some("req_client".to_string()), 42);

    assert_eq!(hdr(&headers, "x-ratelimit-limit-requests"), Some("1"));
    assert_eq!(hdr(&headers, "x-ratelimit-remaining-requests"), Some("0"));
    assert_eq!(hdr(&headers, "x-ratelimit-limit-tokens"), Some("1000"));
    assert_eq!(hdr(&headers, "x-ratelimit-remaining-tokens"), Some("750"));
    assert_eq!(hdr(&headers, "x-request-id"), Some("req_client"));
    assert_eq!(hdr(&headers, "openai-processing-ms"), Some("42"));
    assert_eq!(hdr(&headers, "openai-organization"), Some("atlas-local"));

    // When rate_limit_middleware didn't run (disabled), fallback headers populate.
    let mut fallback = HeaderMap::new();
    super::apply_observability_headers(&mut fallback, None, 7);
    assert_eq!(
        hdr(&fallback, "x-ratelimit-limit-requests"),
        Some("1000000")
    );
    assert_eq!(
        hdr(&fallback, "x-ratelimit-remaining-requests"),
        Some("999999")
    );
    assert_eq!(
        hdr(&fallback, "x-ratelimit-limit-tokens"),
        Some("1000000000")
    );
    assert_eq!(
        hdr(&fallback, "x-ratelimit-remaining-tokens"),
        Some("999999999")
    );
}

#[test]
fn rate_limit_token_estimation_charges_only_post_generation_routes() {
    use axum::http::Method;

    for (method, path) in [
        (Method::GET, "/v1/models"),
        (Method::HEAD, "/v1/models"),
        (Method::GET, "/v1/responses/resp_1"),
        (Method::DELETE, "/v1/responses/resp_1"),
        (Method::GET, "/v1/chat/completions/chatcmpl-1"),
        (Method::OPTIONS, "/v1/chat/completions"),
        (Method::POST, "/v1/messages/count_tokens"),
        (Method::POST, "/v1/conversations"),
        (Method::GET, "/v1/disagg/status"),
    ] {
        assert_eq!(
            super::estimated_tokens_for_request(&method, path, Some(32_768)),
            0,
            "{method} {path} must reserve 0 TPM tokens"
        );
    }

    for path in [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/messages",
        "/v1/responses",
        "/v1/decisions",
        "/v1/disagg/prefill",
        "/v1/disagg/decode",
        "/v1/disagg/generate",
    ] {
        assert_eq!(
            super::estimated_tokens_for_request(&Method::POST, path, Some(32_768)),
            32_768,
            "POST {path} must reserve max_seq_len"
        );
    }
    assert_eq!(
        super::estimated_tokens_for_request(&Method::POST, "/v1/chat/completions", None),
        0
    );
}

#[test]
fn reservation_guard_refunds_on_disconnect_and_errors_without_double_refund() {
    use axum::http::StatusCode;

    let make_limiter = || {
        crate::rate_limiter::RateLimiter::with_config(crate::rate_limiter::RateLimitConfig {
            rpm: 0,
            tpm: 1000,
            burst_rpm: 1,
            burst_tpm: 1000,
        })
    };

    // 1. Client disconnect (guard dropped before finish) refunds full reservation.
    let lim = make_limiter();
    assert!(lim.admit("k", 800).allowed);
    let ctx = crate::rate_limiter::RequestContext::new("k".to_string(), 800);
    let guard = super::ReservationGuard::new(lim.clone(), ctx);
    drop(guard);
    assert!(
        lim.admit("k", 800).allowed,
        "dropped reservation guard must refund tokens"
    );

    // 2. 4xx / 5xx response without handler settlement refunds full reservation.
    for status in [
        StatusCode::BAD_REQUEST,
        StatusCode::TOO_MANY_REQUESTS,
        StatusCode::INTERNAL_SERVER_ERROR,
        StatusCode::SERVICE_UNAVAILABLE,
    ] {
        let lim = make_limiter();
        assert!(lim.admit("k", 800).allowed);
        let ctx = crate::rate_limiter::RequestContext::new("k".to_string(), 800);
        let guard = super::ReservationGuard::new(lim.clone(), ctx);
        guard.finish(status);
        assert!(
            lim.admit("k", 800).allowed,
            "unsettled {status} response must refund reserved tokens"
        );
    }

    // 3. 200 OK disarms middleware guard (handler owns true-up).
    let lim = make_limiter();
    assert!(lim.admit("k", 800).allowed);
    let ctx = crate::rate_limiter::RequestContext::new("k".to_string(), 800);
    let guard = super::ReservationGuard::new(lim.clone(), ctx.clone());
    // Simulate handler settling actual usage = 300 (refunding 500 -> 700 remaining).
    assert!(ctx.mark_settled());
    lim.refund_tokens(&ctx.identity, 500);
    guard.finish(StatusCode::OK);
    assert!(
        !lim.admit("k", 800).allowed,
        "200 OK must not double-refund"
    );
    assert!(lim.admit("k", 700).allowed);

    // 4. Late 500/disconnect AFTER handler already settled must NOT double-refund.
    let lim = make_limiter();
    assert!(lim.admit("k", 800).allowed);
    let ctx = crate::rate_limiter::RequestContext::new("k".to_string(), 800);
    let guard = super::ReservationGuard::new(lim.clone(), ctx.clone());
    assert!(ctx.mark_settled());
    lim.refund_tokens(&ctx.identity, 500); // 700 left
    guard.finish(StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        !lim.admit("k", 800).allowed,
        "post-settlement 500 must not double-refund"
    );
    assert!(lim.admit("k", 700).allowed);
}
