// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn cfg(rpm: u64, tpm: u64) -> RateLimitConfig {
    RateLimitConfig {
        rpm,
        tpm,
        burst_rpm: rpm.max(1),
        burst_tpm: tpm.max(1),
    }
}

#[test]
fn disabled_limiter_always_allows() {
    let lim = RateLimiter::with_config(cfg(0, 0));
    for _ in 0..100 {
        let d = lim.admit("k", 10_000);
        assert!(d.allowed);
    }
}

#[test]
fn request_bucket_blocks_after_burst() {
    let lim = RateLimiter::with_config(cfg(2, 0));
    assert!(lim.admit("k", 0).allowed);
    assert!(lim.admit("k", 0).allowed);
    let d = lim.admit("k", 0);
    assert!(!d.allowed);
    assert!(matches!(d.denied_by, Some(DenialReason::Requests)));
    assert!(d.retry_after_secs >= 1);
}

#[test]
fn token_bucket_blocks_on_oversized_request() {
    let lim = RateLimiter::with_config(cfg(0, 1000));
    assert!(lim.admit("k", 600).allowed);
    // 600 consumed, 400 left; next 500 should deny.
    let d = lim.admit("k", 500);
    assert!(!d.allowed);
    assert!(matches!(d.denied_by, Some(DenialReason::Tokens)));
}

#[test]
fn keys_do_not_interfere() {
    let lim = RateLimiter::with_config(cfg(1, 0));
    assert!(lim.admit("a", 0).allowed);
    assert!(lim.admit("b", 0).allowed);
    assert!(!lim.admit("a", 0).allowed);
    assert!(!lim.admit("b", 0).allowed);
}

#[test]
fn identity_prefers_bearer_and_x_api_key_case_insensitively() {
    use axum::http::{HeaderMap, HeaderValue, header};
    let mut h = HeaderMap::new();
    h.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("bearer sk-abc"),
    );
    h.insert("x-forwarded-for", HeaderValue::from_static("1.2.3.4"));
    let id_lower = extract_identity_with_proxy(&h, None, true);
    assert!(id_lower.starts_with("bearer:"), "{id_lower}");

    let mut h_upper = HeaderMap::new();
    h_upper.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer sk-abc"),
    );
    assert_eq!(id_lower, extract_identity(&h_upper, None));

    let mut h_x = HeaderMap::new();
    h_x.insert("x-api-key", HeaderValue::from_static("sk-abc"));
    assert_eq!(extract_identity(&h_x, None), id_lower);
}

#[test]
fn identity_ignores_xff_unless_trust_proxy_enabled() {
    use axum::http::{HeaderMap, HeaderValue};
    let mut h = HeaderMap::new();
    h.insert(
        "x-forwarded-for",
        HeaderValue::from_static("9.9.9.9, 1.1.1.1"),
    );
    let peer: std::net::SocketAddr = "127.0.0.1:9000".parse().unwrap();
    // Default (untrusted proxy): ignores spoofable X-Forwarded-For and uses peer IP.
    assert_eq!(extract_identity(&h, Some(peer)), "peer:127.0.0.1");
    assert_eq!(extract_identity(&h, None), "peer:unknown");
    // Explicit trusted proxy opt-in: honors first X-Forwarded-For entry.
    assert_eq!(
        extract_identity_with_proxy(&h, Some(peer), true),
        "xff:9.9.9.9"
    );
    let empty = HeaderMap::new();
    assert_eq!(
        extract_identity_with_proxy(&empty, Some(peer), true),
        "peer:127.0.0.1"
    );
}

#[test]
fn token_refund_restores_budget() {
    let lim = RateLimiter::with_config(cfg(0, 1000));
    assert!(lim.admit("k", 900).allowed);
    lim.refund_tokens("k", 500);
    assert!(lim.admit("k", 500).allowed);
}

#[test]
fn max_keys_cap_fails_closed_without_unbounded_map_growth() {
    let lim = RateLimiter::with_config_and_max_keys(cfg(1, 1000), 3);
    assert!(lim.admit("k0", 100).allowed);
    assert!(lim.admit("k1", 100).allowed);
    assert!(lim.admit("k2", 100).allowed);
    assert_eq!(lim.key_count(), 3);

    // Beyond MAX_KEYS with all entries active: fail closed without inserting.
    for i in 3..10 {
        let d = lim.admit(&format!("k{i}"), 100);
        assert!(!d.allowed, "overflow key k{i} must fail closed");
        assert!(matches!(d.denied_by, Some(DenialReason::Requests)));
        assert!(d.retry_after_secs > 0);
        assert_eq!(lim.key_count(), 3, "key map must stay strictly bounded");
    }

    // Existing tracked keys still enforce their bucket limits.
    let existing = lim.admit("k0", 100);
    assert!(!existing.allowed);
    assert!(matches!(existing.denied_by, Some(DenialReason::Requests)));
    assert_eq!(lim.key_count(), 3);
}

#[test]
fn request_context_mark_settled_is_once_and_shared_across_clones() {
    let ctx = RequestContext::new("peer:127.0.0.1".to_string(), 4096);
    let cloned = ctx.clone();
    assert!(ctx.mark_settled(), "first settlement must succeed");
    assert!(
        !cloned.mark_settled(),
        "cloned context must observe settlement to prevent double-refund"
    );
    assert!(!ctx.mark_settled());
}
