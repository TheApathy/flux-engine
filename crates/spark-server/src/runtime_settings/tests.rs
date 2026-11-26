// SPDX-License-Identifier: AGPL-3.0-only

//! Runtime settings: validation, restart-only refusal, atomic publication,
//! sources, request-wins precedence, persistence and the write gate.

use std::net::SocketAddr;

use serde_json::json;

use super::*;

fn settings() -> RuntimeSettings {
    RuntimeSettings::for_tests(65_536)
}

fn startup_kwargs_settings(kwargs: &str, disable_thinking: bool) -> RuntimeSettings {
    let kw = crate::openai::ChatTemplateKwargs::from_json(kwargs);
    RuntimeSettings::new(
        StartupDefaults::new(
            &atlas_kernels::ModelBehavior::default(),
            true,
            atlas_kernels::SamplingPresets::default(),
            kw.as_ref(),
            disable_thinking,
            Some(32_768),
            65_536,
            crate::openai::qwen38_effort_thinking_budget,
        ),
        NudgeControl::unavailable(),
        None,
        "fixture".into(),
    )
}

fn source_of(s: &RuntimeSettings, key: &str) -> String {
    s.describe()["settings"][key]["source"]
        .as_str()
        .unwrap()
        .to_string()
}

fn value_of(s: &RuntimeSettings, key: &str) -> Value {
    s.describe()["settings"][key]["value"].clone()
}

// ── Default identity ────────────────────────────────────────────────────────

#[test]
fn untouched_settings_publish_exactly_the_startup_values() {
    let behavior = atlas_kernels::ModelBehavior::default();
    let presets = atlas_kernels::SamplingPresets::default();
    let s = settings();
    let d = s.snapshot();
    assert_eq!(d.thinking_default, behavior.thinking_default);
    assert!(!d.thinking_default_runtime);
    assert_eq!(d.max_thinking_budget, behavior.max_thinking_budget);
    assert_eq!(d.thinking_answer_reserve, behavior.thinking_answer_reserve);
    assert!(d.effort_budgets.is_empty());
    assert_eq!(d.default_max_tokens, None);
    assert_eq!(
        format!("{:?}", d.sampling_presets),
        format!("{presets:?}"),
        "presets are the startup presets, field for field"
    );
    assert!(d.runtime_keys.is_empty());
    assert_eq!(d.log_suffix(), "", "the request log line is unchanged");
    assert!(Arc::ptr_eq(&d, &s.snapshot()), "snapshots share one Arc");
}

#[test]
fn describe_lists_every_key_with_a_source_and_no_secrets() {
    let s = startup_kwargs_settings(r#"{"enable_thinking": true}"#, false);
    let out = s.describe();
    let keys = out["settings"].as_object().unwrap();
    assert_eq!(keys.len(), catalog::all_keys().len());
    assert_eq!(source_of(&s, "thinking.default"), "startup");
    assert_eq!(value_of(&s, "thinking.default"), json!(true));
    assert_eq!(source_of(&s, "thinking.max_budget"), "startup");
    assert_eq!(source_of(&s, "max_tokens.default"), "startup");
    assert_eq!(value_of(&s, "max_tokens.default"), json!(32_768));
    assert_eq!(source_of(&s, "sampling.tools.temperature"), "default");
    assert_eq!(value_of(&s, "sampling.tools.temperature"), json!(0.6));
    // The Qwen3.8 effort table defers `high` to the model budget.
    assert_eq!(value_of(&s, "thinking.effort_budget.low"), json!(1024));
    assert_eq!(value_of(&s, "thinking.effort_budget.high"), Value::Null);
    assert_eq!(out["restart_only"], json!(RESTART_ONLY));
    let text = out.to_string();
    for secret in ["auth", "token\"", "api_key", "Bearer"] {
        assert!(!text.contains(secret), "{secret} in {text}");
    }
}

// ── Validation ──────────────────────────────────────────────────────────────

#[test]
fn every_value_is_type_and_range_checked() {
    let bad = [
        json!({"thinking.default": "yes"}),
        json!({"thinking.max_budget": 0}),
        json!({"thinking.max_budget": -5}),
        json!({"thinking.max_budget": 1.5}),
        json!({"thinking.answer_reserve": 2_000_000}),
        json!({"thinking.effort_budget.low": "big"}),
        json!({"sampling.thinking_text.temperature": 2.5}),
        json!({"sampling.thinking_text.temperature": -0.1}),
        json!({"sampling.non_thinking.top_p": 0}),
        json!({"sampling.non_thinking.top_p": 1.01}),
        json!({"sampling.tools.top_k": -1}),
        json!({"sampling.tools.min_p": 1.5}),
        json!({"sampling.tools.presence_penalty": 3}),
        json!({"sampling.tools.frequency_penalty": -2.5}),
        json!({"sampling.tools.repetition_penalty": 0}),
        json!({"max_tokens.default": 0}),
        json!({"max_tokens.default": 65_537}),
    ];
    let s = settings();
    for body in bad {
        match s.apply_patch(&body) {
            Err(SettingError::Invalid { .. }) => {}
            other => panic!("{body}: {other:?}"),
        }
    }
    assert!(s.snapshot().runtime_keys.is_empty(), "nothing applied");
}

#[test]
fn unknown_keys_and_malformed_bodies_are_refused() {
    let s = settings();
    for body in [
        json!({"thinking.defualt": true}),
        json!({"sampling.thinking_coding.temperature": 0.5}),
        json!({"sampling.tools.typical_p": 0.5}),
        json!({"thinking.effort_budget.ultra": 9}),
        json!({}),
        json!([1, 2]),
        json!("thinking.default=true"),
    ] {
        assert!(
            matches!(s.apply_patch(&body), Err(SettingError::Invalid { .. })),
            "{body}"
        );
    }
}

#[test]
fn restart_only_settings_are_refused_by_name() {
    let s = settings();
    let mut names: Vec<&str> = RESTART_ONLY.to_vec();
    names.extend(["speculation.mode", "watchdog.think_loop_min_repeats"]);
    for name in names {
        let mut body = serde_json::Map::new();
        body.insert(name.into(), json!(1));
        match s.apply_patch(&Value::Object(body)) {
            Err(SettingError::RestartRequired { key, .. }) => assert_eq!(key, name),
            other => panic!("{name}: {other:?}"),
        }
    }
}

#[test]
fn thinking_default_is_restart_only_under_disable_thinking() {
    let s = startup_kwargs_settings("", true);
    assert!(!s.snapshot().thinking_default);
    assert!(matches!(
        s.apply_patch(&json!({"thinking.default": true})),
        Err(SettingError::RestartRequired { .. })
    ));
}

#[test]
fn a_patch_with_one_bad_key_changes_nothing() {
    let s = settings();
    let before = s.snapshot();
    let err = s.apply_patch(&json!({
        "sampling.thinking_text.temperature": 0.3,
        "max_tokens.default": 8192,
        "kv_cache_dtype": "fp8",
    }));
    assert!(matches!(err, Err(SettingError::RestartRequired { .. })));
    assert!(Arc::ptr_eq(&before, &s.snapshot()), "no new publication");
    assert_eq!(source_of(&s, "max_tokens.default"), "default");
}

// ── Apply, reset, sources ───────────────────────────────────────────────────

#[test]
fn a_patch_applies_reports_changes_and_null_resets() {
    let s = settings();
    let changes = s
        .apply_patch(&json!({
            "sampling.thinking_text.temperature": 0.3,
            "thinking.max_budget": 12_000,
            "thinking.effort_budget.medium": 4096,
            "max_tokens.default": 16_384,
        }))
        .unwrap();
    assert_eq!(changes.len(), 4);
    let d = s.snapshot();
    assert_eq!(d.sampling_presets.thinking_text.temperature, 0.3);
    assert_eq!(d.max_thinking_budget, 12_000);
    assert_eq!(d.effort_budget("medium", |_| Some(1)), Some(4096));
    assert_eq!(d.effort_budget("low", |_| Some(1)), Some(1));
    assert_eq!(d.default_max_tokens, Some(16_384));
    assert_eq!(
        d.log_suffix(),
        ", runtime=[max_tokens.default,sampling.thinking_text.temperature,\
         thinking.effort_budget.medium,thinking.max_budget]"
    );
    assert_eq!(source_of(&s, "thinking.max_budget"), "runtime");
    let temp = changes
        .iter()
        .find(|c| c.key == "sampling.thinking_text.temperature")
        .unwrap();
    assert_eq!(
        (temp.old.clone(), temp.new.clone()),
        (json!(0.7), json!(0.3))
    );

    s.apply_patch(&json!({"thinking.max_budget": null, "max_tokens.default": null}))
        .unwrap();
    let d = s.snapshot();
    assert_eq!(
        d.max_thinking_budget,
        atlas_kernels::ModelBehavior::default().max_thinking_budget
    );
    assert_eq!(d.default_max_tokens, None);
    assert_eq!(source_of(&s, "thinking.max_budget"), "default");
    assert_eq!(
        source_of(&s, "sampling.thinking_text.temperature"),
        "runtime"
    );
}

#[test]
fn a_runtime_thinking_default_replaces_the_launch_kwargs_default() {
    let s = startup_kwargs_settings(r#"{"enable_thinking": true}"#, false);
    s.apply_patch(&json!({"thinking.default": false})).unwrap();
    let d = s.snapshot();
    assert!(!d.thinking_default && d.thinking_default_runtime);
}

#[test]
fn nudge_keys_are_refused_on_a_model_without_the_nudge() {
    let s = settings();
    assert!(matches!(
        s.apply_patch(&json!({"think_nudge.enabled": true})),
        Err(SettingError::Invalid { .. })
    ));
    assert_eq!(value_of(&s, "think_nudge.enabled"), json!(false));
}

// ── Atomicity ───────────────────────────────────────────────────────────────

#[test]
fn an_admitted_request_keeps_its_snapshot_across_a_change() {
    let s = settings();
    let in_flight = s.snapshot();
    s.apply_patch(&json!({"sampling.non_thinking.temperature": 1.2}))
        .unwrap();
    assert_eq!(in_flight.sampling_presets.non_thinking.temperature, 0.7);
    assert_eq!(s.snapshot().sampling_presets.non_thinking.temperature, 1.2);
}

#[test]
fn concurrent_readers_never_see_a_half_applied_patch() {
    // Each PATCH sets two keys to the same value; a reader must always see
    // them equal (old pair or new pair), never one of each.
    let s = Arc::new(settings());
    s.apply_patch(&json!({
        "sampling.tools.temperature": 0.1,
        "sampling.tools.top_p": 0.1,
    }))
    .unwrap();
    // Writers keep going until the readers have really run alongside them
    // (a loaded machine can otherwise finish every PATCH before a reader is
    // scheduled), and each reader checks at least once before it looks at
    // `stop`.
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reads = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let (s, stop, reads) = (Arc::clone(&s), Arc::clone(&stop), Arc::clone(&reads));
            std::thread::spawn(move || {
                let mut seen = 0u64;
                loop {
                    let d = s.snapshot();
                    let t = &d.sampling_presets.tools;
                    assert_eq!(t.temperature, t.top_p, "torn snapshot");
                    seen += 1;
                    reads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if stop.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                }
                seen
            })
        })
        .collect();
    let writers: Vec<_> = (0..2)
        .map(|w| {
            let (s, reads) = (Arc::clone(&s), Arc::clone(&reads));
            std::thread::spawn(move || {
                for i in 0..1_000_000u64 {
                    let x = 0.1 + 0.01 * ((i * 2 + w) % 80) as f64;
                    s.apply_patch(&json!({
                        "sampling.tools.temperature": x,
                        "sampling.tools.top_p": x,
                    }))
                    .unwrap();
                    if i >= 300 && reads.load(std::sync::atomic::Ordering::Relaxed) >= 2000 {
                        break;
                    }
                }
            })
        })
        .collect();
    for w in writers {
        w.join().unwrap();
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for r in readers {
        assert!(r.join().unwrap() > 0);
    }
    let d = s.snapshot();
    assert_eq!(
        d.sampling_presets.tools.temperature,
        d.sampling_presets.tools.top_p
    );
}

// ── Request-wins precedence ─────────────────────────────────────────────────

fn chat_request(body: Value) -> crate::openai::ChatCompletionRequest {
    let mut body = body;
    body["messages"] = json!([{"role": "user", "content": "hi"}]);
    serde_json::from_value(body).unwrap()
}

fn resolve(d: &RequestDefaults, body: Value) -> (bool, Option<u32>) {
    let req = chat_request(body);
    let model_default = d.thinking_default;
    req.resolve_thinking_with_effort(model_default, &|e| {
        d.effort_budget(e, crate::openai::qwen38_effort_thinking_budget)
    })
}

#[test]
fn request_fields_win_over_runtime_thinking_defaults() {
    let s = settings();
    s.apply_patch(&json!({
        "thinking.default": true,
        "thinking.effort_budget.low": 3000,
    }))
    .unwrap();
    let d = s.snapshot();
    // Silent request: the runtime default applies.
    assert_eq!(resolve(&d, json!({})), (true, None));
    // The runtime effort table applies to a request that names the effort.
    assert_eq!(
        resolve(&d, json!({"reasoning_effort": "low"})),
        (true, Some(3000))
    );
    // Explicit request fields win.
    assert_eq!(
        resolve(
            &d,
            json!({"chat_template_kwargs": {"enable_thinking": false}})
        ),
        (false, Some(0))
    );
    assert_eq!(
        resolve(&d, json!({"thinking_token_budget": 100})),
        (true, Some(100))
    );
    assert_eq!(
        resolve(
            &d,
            json!({"thinking": {"type": "enabled", "budget_tokens": 50}})
        ),
        (true, Some(50))
    );
}

#[test]
fn request_max_tokens_wins_over_the_runtime_default() {
    let s = settings();
    s.apply_patch(&json!({"max_tokens.default": 9000})).unwrap();
    let d = s.snapshot();
    let mut silent = chat_request(json!({}));
    silent.apply_default_max_tokens(false, d.default_max_tokens);
    assert_eq!(silent.max_tokens, 9000);
    let mut explicit = chat_request(json!({"max_tokens": 64}));
    explicit.apply_default_max_tokens(true, d.default_max_tokens);
    assert_eq!(explicit.max_tokens, 64);
}

#[test]
fn pin_takes_one_snapshot_per_request() {
    let s = settings();
    let mut req = chat_request(json!({}));
    let first = pin(&mut req, &s);
    s.apply_patch(&json!({"thinking.max_budget": 77})).unwrap();
    let again = pin(&mut req, &s);
    assert!(Arc::ptr_eq(&first, &again), "the admission snapshot stays");
    assert_ne!(again.max_thinking_budget, 77);
    let mut next = chat_request(json!({}));
    assert_eq!(pin(&mut next, &s).max_thinking_budget, 77);
}

// ── Persistence ─────────────────────────────────────────────────────────────

fn persisted(path: &std::path::Path, model: &str) -> RuntimeSettings {
    RuntimeSettings::new(
        settings().startup.clone(),
        NudgeControl::unavailable(),
        Some(SettingsFile::new(path.to_path_buf())),
        model.into(),
    )
}

#[test]
fn changes_persist_and_reapply_for_the_same_model_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    let a = persisted(&path, "model-a");
    a.apply_patch(&json!({"sampling.tools.temperature": 0.4}))
        .unwrap();
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(doc["schema"], "flux.runtime_settings.v1");
    assert_eq!(doc["settings"], json!({"sampling.tools.temperature": 0.4}));
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    let restarted = persisted(&path, "model-a");
    restarted.load_persisted().unwrap();
    assert_eq!(restarted.snapshot().sampling_presets.tools.temperature, 0.4);
    assert_eq!(
        source_of(&restarted, "sampling.tools.temperature"),
        "runtime"
    );

    let other = persisted(&path, "model-b");
    other.load_persisted().unwrap();
    assert_eq!(other.snapshot().sampling_presets.tools.temperature, 0.6);
}

#[test]
fn a_change_that_cannot_be_saved_is_not_applied() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing-dir").join("settings.json");
    let s = persisted(&path, "m");
    let before = s.snapshot();
    assert!(matches!(
        s.apply_patch(&json!({"max_tokens.default": 100})),
        Err(SettingError::Internal { .. })
    ));
    assert!(Arc::ptr_eq(&before, &s.snapshot()));
}

// ── Write gate and HTTP answers ─────────────────────────────────────────────

fn peer(addr: &str) -> SocketAddr {
    addr.parse().unwrap()
}

async fn body_json(resp: axum::response::Response) -> (u16, Value) {
    let status = resp.status().as_u16();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn patch_answers_carry_codes_and_the_new_settings() {
    let s = settings();
    let p = peer("127.0.0.1:1");
    let (status, v) = body_json(handlers::patch(&s, br#"{"max_seq_len": 4}"#, p)).await;
    assert_eq!(
        (status, v["error"]["code"].clone()),
        (409, json!("restart_required"))
    );
    assert_eq!(v["error"]["param"], "max_seq_len");
    let (status, v) = body_json(handlers::patch(&s, br#"{"nope": 1}"#, p)).await;
    assert_eq!(
        (status, v["error"]["code"].clone()),
        (400, json!("unknown_setting"))
    );
    let (status, v) = body_json(handlers::patch(&s, b"{", p)).await;
    assert_eq!(
        (status, v["error"]["code"].clone()),
        (400, json!("invalid_value"))
    );
    let (status, v) = body_json(handlers::patch(
        &s,
        br#"{"sampling.non_thinking.temperature": 0.25}"#,
        p,
    ))
    .await;
    assert_eq!(status, 200);
    assert_eq!(v["changed"][0]["key"], "sampling.non_thinking.temperature");
    assert_eq!(v["changed"][0]["new"], json!(0.25));
    assert_eq!(
        v["settings"]["sampling.non_thinking.temperature"],
        json!({"value": 0.25, "source": "runtime"})
    );
}
