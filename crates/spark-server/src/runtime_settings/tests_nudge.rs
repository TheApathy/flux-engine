// SPDX-License-Identifier: AGPL-3.0-only

//! The thinking nudge is published inside the request-defaults snapshot, so
//! one request never mixes a new nudge with old defaults.

use serde_json::json;

use super::*;
use crate::scheduler::think_nudge::{ThinkNudgeConfig, ThinkNudgePolicy};

fn tables() -> ThinkNudgeConfig {
    ThinkNudgeConfig::build(
        ThinkNudgePolicy::default(),
        |text| Ok(text.bytes().map(|b| 1000 + u32::from(b)).collect()),
        vec![1; 64],
        &[],
    )
    .unwrap()
}

fn with_nudge(startup_policy: Option<ThinkNudgePolicy>) -> RuntimeSettings {
    RuntimeSettings::new(
        RuntimeSettings::for_tests(65_536).startup.clone(),
        NudgeControl::new(Some(tables()), startup_policy, false),
        None,
        "m".into(),
    )
}

#[test]
fn the_startup_nudge_is_in_the_first_snapshot() {
    assert!(with_nudge(None).snapshot().think_nudge.is_none());
    let on = with_nudge(Some(ThinkNudgePolicy::default()));
    let cfg = on.snapshot().think_nudge.clone().expect("on at launch");
    assert_eq!(cfg.policy, ThinkNudgePolicy::default());
}

#[test]
fn enabling_and_tuning_the_nudge_publishes_a_new_snapshot() {
    let s = with_nudge(None);
    let before = s.snapshot();
    s.apply_patch(&json!({"think_nudge.enabled": true, "think_nudge.rate": 0.4}))
        .unwrap();
    let after = s.snapshot();
    assert!(
        before.think_nudge.is_none(),
        "admitted requests keep theirs"
    );
    let cfg = after.think_nudge.clone().unwrap();
    assert_eq!(cfg.policy.rate as f32, 0.4);
    s.apply_patch(&json!({"think_nudge.enabled": false}))
        .unwrap();
    assert!(s.snapshot().think_nudge.is_none());
    assert!(after.think_nudge.is_some());
}

#[test]
fn readers_never_see_a_new_nudge_with_old_defaults() {
    // Each PATCH moves the nudge rate and a sampling default together.
    let s = Arc::new(with_nudge(Some(ThinkNudgePolicy::default())));
    let set = |x: f64| json!({"think_nudge.rate": x, "sampling.tools.temperature": x});
    s.apply_patch(&set(0.5)).unwrap();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reads = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let reader = {
        let (s, stop, reads) = (Arc::clone(&s), Arc::clone(&stop), Arc::clone(&reads));
        std::thread::spawn(move || {
            loop {
                let d = s.snapshot();
                let rate = d.think_nudge.as_ref().unwrap().policy.rate as f32;
                assert_eq!(
                    rate, d.sampling_presets.tools.temperature,
                    "torn nudge/defaults"
                );
                reads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if stop.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
            }
        })
    };
    // Keep writing until the reader has really run alongside (see the
    // defaults test in tests.rs).
    for i in 0..1_000_000u32 {
        s.apply_patch(&set(0.3 + 0.001 * f64::from(i % 200)))
            .unwrap();
        if i >= 300 && reads.load(std::sync::atomic::Ordering::Relaxed) >= 1000 {
            break;
        }
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    reader.join().unwrap();
}
