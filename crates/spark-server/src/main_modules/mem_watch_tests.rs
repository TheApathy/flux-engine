// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::collections::HashMap;

const MIB_U: usize = 1024 * 1024;

fn cfg() -> WatchConfig {
    WatchConfig {
        floor_bytes: 12_288 * MIB_U,
        recover_margin_bytes: 1_024 * MIB_U,
        interval: Duration::from_millis(500),
        load_action: FloorAction::Abort,
        serve_action: FloorAction::Refuse,
    }
}

fn mib(n: usize) -> Option<usize> {
    Some(n * MIB_U)
}

fn lookup(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let map: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    move |key: &str| map.get(key).cloned()
}

#[test]
fn defaults_match_the_contract() {
    let parsed = WatchConfig::from_lookup(lookup(&[])).expect("defaults parse");
    assert_eq!(parsed, Some(cfg()));
}

#[test]
fn flux_names_win_over_atlas_aliases() {
    let parsed = WatchConfig::from_lookup(lookup(&[
        ("FLUX_MEM_FLOOR_MIB", "8192"),
        ("ATLAS_MEM_FLOOR_MIB", "4096"),
        ("ATLAS_MEM_WATCH_INTERVAL_MS", "250"),
        ("FLUX_MEM_FLOOR_ACTION_SERVE", "log"),
        ("FLUX_MEM_FLOOR_ACTION_LOAD", "refuse"),
    ]))
    .expect("parse")
    .expect("enabled");
    assert_eq!(parsed.floor_bytes, 8_192 * MIB_U);
    assert_eq!(parsed.interval, Duration::from_millis(250));
    assert_eq!(parsed.serve_action, FloorAction::Log);
    assert_eq!(parsed.load_action, FloorAction::Refuse);
}

#[test]
fn watch_off_disables_and_bad_values_fail_closed() {
    assert_eq!(
        WatchConfig::from_lookup(lookup(&[("FLUX_MEM_WATCH", "off")])).expect("parse"),
        None
    );
    for (key, value) in [
        ("FLUX_MEM_FLOOR_MIB", "12G"),
        ("FLUX_MEM_WATCH_INTERVAL_MS", "0"),
        ("FLUX_MEM_FLOOR_ACTION_SERVE", "kill"),
        ("FLUX_MEM_WATCH", "maybe"),
    ] {
        assert!(
            WatchConfig::from_lookup(lookup(&[(key, value)])).is_err(),
            "{key}={value} must be a launch error"
        );
    }
}

#[test]
fn unreadable_sample_changes_nothing() {
    let (state, effects) = step(&cfg(), WatchState::Healthy, Phase::Serve, None);
    assert_eq!(state, WatchState::Healthy);
    assert!(effects.is_empty());
}

#[test]
fn serve_breach_refuses_until_recovery_above_floor_plus_margin() {
    let c = cfg();
    let (s, e) = step(&c, WatchState::Healthy, Phase::Serve, mib(12_000));
    assert_eq!(s, WatchState::Breached);
    assert_eq!(
        e,
        vec![Effect::Breach {
            mem_available: 12_000 * MIB_U,
            phase: Phase::Serve,
            action: FloorAction::Refuse,
        }]
    );
    assert!(refusing(&c, s, Phase::Serve));
    // Above the floor but inside the hysteresis band: still refusing, no log spam.
    let (s, e) = step(&c, s, Phase::Serve, mib(12_800));
    assert_eq!((s, e.len()), (WatchState::Breached, 0));
    assert!(refusing(&c, s, Phase::Serve));
    // Still below: no repeated breach line.
    let (s, e) = step(&c, s, Phase::Serve, mib(11_000));
    assert_eq!((s, e.len()), (WatchState::Breached, 0));
    let (s, e) = step(&c, s, Phase::Serve, mib(13_312));
    assert_eq!(s, WatchState::Healthy);
    assert_eq!(
        e,
        vec![Effect::Recovered {
            mem_available: 13_312 * MIB_U,
            phase: Phase::Serve,
        }]
    );
    assert!(!refusing(&c, s, Phase::Serve));
}

#[test]
fn load_breach_aborts_by_default() {
    let (s, e) = step(&cfg(), WatchState::Healthy, Phase::Load, mib(9_000));
    assert_eq!(s, WatchState::Breached);
    assert_eq!(
        e,
        vec![
            Effect::Breach {
                mem_available: 9_000 * MIB_U,
                phase: Phase::Load,
                action: FloorAction::Abort,
            },
            Effect::Abort,
        ]
    );
}

#[test]
fn log_action_never_refuses() {
    let c = WatchConfig {
        serve_action: FloorAction::Log,
        ..cfg()
    };
    let (s, e) = step(&c, WatchState::Healthy, Phase::Serve, mib(100));
    assert_eq!(s, WatchState::Breached);
    assert_eq!(e.len(), 1);
    assert!(!refusing(&c, s, Phase::Serve));
}

#[test]
fn healthy_sample_at_exact_floor_is_not_a_breach() {
    let (s, e) = step(&cfg(), WatchState::Healthy, Phase::Serve, mib(12_288));
    assert_eq!((s, e.len()), (WatchState::Healthy, 0));
}

#[test]
fn breach_line_carries_the_operator_fields() {
    let line = breach_line(12_000 * MIB_U, &cfg(), Phase::Load);
    assert_eq!(
        line,
        "MEM_FLOOR_BREACH memavail_mib=12000 floor_mib=12288 phase=load action=abort"
    );
}

#[test]
fn shared_state_tracks_load_minimum_and_refusal() {
    let shared = WatchShared::new(100_000 * MIB_U);
    shared.observe(Phase::Load, 90_000 * MIB_U);
    shared.observe(Phase::Load, 95_000 * MIB_U);
    shared.observe(Phase::Serve, 10_000 * MIB_U);
    assert_eq!(shared.peak_load_bytes(), 10_000 * MIB_U);
    assert!(!shared.is_refusing());
    shared.set_refusing(true);
    assert!(shared.is_refusing());
}

#[test]
fn tick_applies_effects_and_exit_code_is_75() {
    let shared = WatchShared::new(100_000 * MIB_U);
    let exits = std::cell::RefCell::new(Vec::new());
    let state = tick(&cfg(), &shared, WatchState::Healthy, mib(1_000), &|code| {
        exits.borrow_mut().push(code)
    });
    assert_eq!(state, WatchState::Breached);
    assert_eq!(exits.into_inner(), vec![ABORT_EXIT_CODE]);
    assert_eq!(ABORT_EXIT_CODE, 75);

    shared.set_phase(Phase::Serve);
    let state = tick(&cfg(), &shared, WatchState::Healthy, mib(1_000), &|_| {
        panic!("serve refuse must not exit")
    });
    assert_eq!(state, WatchState::Breached);
    assert!(shared.is_refusing());
    let state = tick(&cfg(), &shared, state, mib(50_000), &|_| unreachable!());
    assert_eq!(state, WatchState::Healthy);
    assert!(!shared.is_refusing());
}

#[test]
fn floor_rejection_only_blocks_new_inference() {
    use axum::http::{Method, StatusCode, header};

    let (status, body) = floor_rejection(&Method::POST, "/v1/chat/completions", true)
        .expect("refusing must reject POST /v1/chat/completions");
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "memory_floor");
    assert_eq!(body["error"]["param"], serde_json::Value::Null);

    let resp = floor_rejection_response(&Method::POST, "/v1/chat/completions", true)
        .expect("503 response with Retry-After");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        resp.headers()
            .get(header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok()),
        Some("5")
    );

    assert!(floor_rejection(&Method::POST, "/v1/chat/completions", false).is_none());
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
        assert!(
            floor_rejection(&Method::POST, path, true).is_some(),
            "POST {path} must be shed"
        );
    }

    // Method-aware & exact-path allowlist: GET/DELETE stored-item reads,
    // count_tokens, status, models, health, and metrics must never be shed.
    for (method, path) in [
        (Method::GET, "/v1/responses/resp_123"),
        (Method::DELETE, "/v1/responses/resp_123"),
        (Method::GET, "/v1/responses/resp_123/input_items"),
        (Method::POST, "/v1/responses/resp_123/cancel"),
        (Method::GET, "/v1/chat/completions/chatcmpl-123"),
        (Method::GET, "/v1/chat/completions"),
        (Method::OPTIONS, "/v1/chat/completions"),
        (Method::POST, "/v1/messages/count_tokens"),
        (Method::GET, "/v1/disagg/status"),
        (Method::GET, "/health"),
        (Method::GET, "/health/live"),
        (Method::GET, "/metrics"),
        (Method::GET, "/v1/models"),
        (Method::GET, "/v1/models/flux"),
        (Method::POST, "/v1/completionsx"),
    ] {
        assert!(
            floor_rejection(&method, path, true).is_none(),
            "{method} {path} must not be shed"
        );
    }
}

#[test]
fn ready_line_reports_margin() {
    let line = ready_line(
        30_000 * MIB_U,
        90_000 * MIB_U,
        Some(92_000 * MIB_U),
        12_288 * MIB_U,
    );
    assert_eq!(
        line,
        "READY memavail_mib=30000 peak_load_mib=90000 fit_need_mib=92000 headroom_mib=17712"
    );
    let line = ready_line(10_000 * MIB_U, 1, None, 12_288 * MIB_U);
    assert!(
        line.ends_with("fit_need_mib=none headroom_mib=-2288"),
        "{line}"
    );
}
