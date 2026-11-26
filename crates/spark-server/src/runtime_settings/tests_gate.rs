// SPDX-License-Identifier: AGPL-3.0-only

//! The keyless write gate: loopback peer, no proxy headers, local Host and
//! Origin (DNS rebinding), each with the right admin token present.

use std::net::SocketAddr;

use serde_json::Value;

use super::handlers;

pub(super) fn peer(addr: &str) -> SocketAddr {
    addr.parse().unwrap()
}

pub(super) async fn body_json(resp: axum::response::Response) -> (u16, Value) {
    let status = resp.status().as_u16();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

pub(super) fn headers(pairs: &[(&str, &str)]) -> axum::http::HeaderMap {
    let mut h = axum::http::HeaderMap::new();
    for (k, v) in pairs {
        h.insert(
            axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
            v.parse().unwrap(),
        );
    }
    h
}

/// The server's admin token in these tests.
pub(super) const TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// The gate's refusal code for a request that carries the right admin token
/// (unless `hdrs` names the token header itself).
fn gate_code(
    key: bool,
    peer_addr: &str,
    hdrs: &[(&str, &str)],
    bound: Option<&str>,
) -> Option<String> {
    let mut all = hdrs.to_vec();
    if !hdrs.iter().any(|(k, _)| *k == super::local_admin::HEADER) {
        all.push((super::local_admin::HEADER, TOKEN));
    }
    gate_code_with(key, peer_addr, &all, bound, Some(TOKEN))
}

pub(super) fn gate_code_with(
    key: bool,
    peer_addr: &str,
    hdrs: &[(&str, &str)],
    bound: Option<&str>,
    server_token: Option<&str>,
) -> Option<String> {
    let h = headers(hdrs);
    let gate = handlers::WriteGate {
        api_key_configured: key,
        peer: peer(peer_addr),
        headers: &h,
        bound: bound.map(str::to_string),
        admin_token: server_token,
    };
    gate.authorize().err().map(|resp| {
        assert_eq!(resp.status(), axum::http::StatusCode::FORBIDDEN);
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let (_, v) = rt.block_on(body_json(resp));
        v["error"]["code"].as_str().unwrap().to_string()
    })
}

const LOCAL: &[(&str, &str)] = &[("host", "127.0.0.1:8899")];

#[test]
fn without_an_api_key_only_loopback_may_write() {
    for ok in ["127.0.0.1:5000", "[::1]:5000", "[::ffff:127.0.0.1]:5000"] {
        assert_eq!(gate_code(false, ok, LOCAL, None), None, "{ok}");
    }
    for refused in ["172.17.0.1:5000", "10.0.0.7:1", "[fe80::1]:1", "0.0.0.0:1"] {
        assert_eq!(
            gate_code(false, refused, LOCAL, None).as_deref(),
            Some("loopback_only"),
            "{refused}"
        );
    }
    // With API keys the middleware has authenticated the caller already.
    assert_eq!(gate_code(true, "172.17.0.1:5000", &[], None), None);
    assert!(handlers::LOOPBACK_ONLY_MESSAGE.contains("set an API key"));
    assert!(handlers::LOOPBACK_ONLY_MESSAGE.contains("docker exec"));
}

#[test]
fn without_an_api_key_a_proxied_request_is_refused() {
    for header in [
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-real-ip",
        "via",
        "cf-connecting-ip",
        "true-client-ip",
    ] {
        let hdrs = [("host", "127.0.0.1:8899"), (header, "203.0.113.9")];
        assert_eq!(
            gate_code(false, "127.0.0.1:40000", &hdrs, None).as_deref(),
            Some("proxied_request"),
            "{header}"
        );
        // An API key is the boundary behind a proxy.
        assert_eq!(gate_code(true, "127.0.0.1:40000", &hdrs, None), None);
    }
    assert!(handlers::PROXIED_MESSAGE.contains("API key"));
}

#[test]
fn without_an_api_key_dns_rebinding_hosts_and_origins_are_refused() {
    let p = "127.0.0.1:40000";
    // Local hosts, with or without a port, and the bound address.
    for host in ["localhost:8899", "LOCALHOST", "127.0.0.1", "[::1]:8899"] {
        assert_eq!(gate_code(false, p, &[("host", host)], None), None, "{host}");
    }
    assert_eq!(
        gate_code(
            false,
            p,
            &[("host", "192.168.1.5:8899")],
            Some("192.168.1.5")
        ),
        None
    );
    // A rebinding page sends its own name; a wildcard bind names no host.
    for (host, bound) in [
        ("attacker.example:8899", None),
        ("attacker.example", Some("0.0.0.0")),
        ("0.0.0.0:8899", Some("0.0.0.0")),
        ("127.0.0.1.attacker.example", None),
    ] {
        assert_eq!(
            gate_code(false, p, &[("host", host)], bound).as_deref(),
            Some("host_not_allowed"),
            "{host}"
        );
    }
    assert_eq!(
        gate_code(false, p, &[], None).as_deref(),
        Some("host_not_allowed"),
        "no Host header"
    );
    // Origin: absent or local passes; foreign or opaque is refused.
    for origin in [
        "http://localhost:3000",
        "https://127.0.0.1",
        "http://[::1]:8899",
    ] {
        let hdrs = [("host", "127.0.0.1:8899"), ("origin", origin)];
        assert_eq!(gate_code(false, p, &hdrs, None), None, "{origin}");
    }
    for origin in [
        "http://attacker.example",
        "null",
        "file://",
        "http://localhost.attacker.example",
    ] {
        let hdrs = [("host", "127.0.0.1:8899"), ("origin", origin)];
        assert_eq!(
            gate_code(false, p, &hdrs, None).as_deref(),
            Some("cross_origin"),
            "{origin}"
        );
    }
}

#[test]
fn a_non_utf8_origin_is_foreign() {
    let mut h = headers(&[
        ("host", "127.0.0.1:8899"),
        (super::local_admin::HEADER, TOKEN),
    ]);
    h.insert(
        "origin",
        axum::http::HeaderValue::from_bytes(b"http://loc\xffalhost").unwrap(),
    );
    let gate = handlers::WriteGate {
        api_key_configured: false,
        peer: peer("127.0.0.1:1"),
        headers: &h,
        bound: None,
        admin_token: Some(TOKEN),
    };
    let resp = gate.authorize().unwrap_err();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let (_, v) = rt.block_on(body_json(resp));
    assert_eq!(v["error"]["code"], "cross_origin");
}
