// SPDX-License-Identifier: AGPL-3.0-only

use std::ffi::OsStr;
use std::fs;
use std::time::{Duration, Instant};

use tempfile::TempDir;

use super::super::{ChunkCostModel, DecodeStepClock, EWMA_ALPHA, MIN_SAMPLES, static_chunk_ms};
use super::*;
use crate::scheduler::prefill_slo::{PrefillTickLedger, SloRuntime};

fn sample_fingerprint(model: &str, wide: &str) -> SloCostFingerprint {
    SloCostFingerprint {
        gpu: "NVIDIA_GB10".to_string(),
        model: model.to_string(),
        widths: vec![512, 1024, 2048, 4096],
        block_size: 16,
        prefill_wide: wide.to_string(),
        qsa_mode: "attn2=1,score2=1".to_string(),
    }
}

#[test]
fn valid_save_load_round_trip_and_slo_runtime_flush_on_drop() {
    let dir = TempDir::new().expect("temp dir");
    let path = dir.path().join("slo_cost_v1.json");
    let fp = sample_fingerprint("Qwen3.8-Flash-Next", "16384");
    let key = fp.encoded();

    let mut model = ChunkCostModel::default();
    assert_eq!(model.observe(512, 0, 9_999.0, 350.0), None); // warmup dropped
    model.observe(512, 0, 180.0, 350.0).expect("sample 1");
    model.observe(512, 0, 200.0, 180.0).expect("sample 2");
    let expected_512 = 180.0 + EWMA_ALPHA * (200.0 - 180.0);

    let entry = model.to_persisted(Some(42.5));
    assert_eq!(entry.decode_step_ms, Some(42.5));
    assert!(entry.cells.contains_key("512:0"));
    assert!(entry.global_ratio_count >= 2);
    assert!(save_entry_atomic(&path, &key, &entry));

    let loaded = load_entry_with_seed(Some(&path), None, &key).expect("round-trip entry");
    assert_eq!(loaded.decode_step_ms, Some(42.5));
    assert_eq!(loaded.cells["512:0"].samples, 2);
    assert!((loaded.cells["512:0"].ewma_ms - expected_512).abs() < 1e-9);
    assert!((loaded.global_ratio_sum - entry.global_ratio_sum).abs() < 1e-9);
    assert_eq!(loaded.global_ratio_count, entry.global_ratio_count);

    // End-to-end SloRuntime: auto-load on init, learn another bucket, flush on Drop.
    let runtime_path = dir.path().join("runtime_slo_cost_v1.json");
    let widths = [512, 1024, 2048, 4096];
    {
        let mut rt = SloRuntime::new(&widths, 150, true, false, 16).with_cost_cache(
            Some(runtime_path.clone()),
            None,
            fp.clone(),
        );
        rt.plan_tick(None, true, None, 4096, false);
        rt.observe_tick(PrefillTickLedger::default(), true, 45.0);
        let mut chunk = PrefillTickLedger::default();
        chunk.record_chunk(1024, 0);
        // 1 warmup + 2 measured samples -> calibrated cell 1024:0 = 260 ms.
        for _ in 0..3 {
            rt.observe_tick(chunk, true, 305.0);
        }
        // Drop flushes dirty state cleanly without hitting the 16-observation threshold.
    }
    let reloaded = SloRuntime::new(&widths, 150, true, false, 16).with_cost_cache(
        Some(runtime_path),
        None,
        fp,
    );
    assert!(reloaded.cost().is_measured(1024, 0));
    assert!((reloaded.cost().predict_ms(1024, 0) - 260.0).abs() < 1e-9);
}

#[test]
fn schema_mismatch_is_ignored() {
    let dir = TempDir::new().expect("temp dir");
    let path = dir.path().join("slo_cost_v1.json");
    let bad_schema = format!(
        r#"{{"schema": {}, "entries": {{"k": {{"cells": {{"512:0": {{"ewma_ms": 111.0, "samples": 4}}}}, "decode_step_ms": 30.0, "global_ratio_sum": 1.0, "global_ratio_count": 1, "updated_unix_secs": 1}}}}}}"#,
        SCHEMA_VERSION + 1
    );
    fs::write(&path, bad_schema).expect("write schema mismatch file");
    assert!(load_entries_from_path(&path).is_empty());
    assert!(load_entry_with_seed(Some(&path), None, "k").is_none());
}

#[test]
fn corrupt_json_is_ignored() {
    let dir = TempDir::new().expect("temp dir");
    let path = dir.path().join("slo_cost_v1.json");
    fs::write(&path, b"{truncated_not_valid_json").expect("write corrupt file");
    assert!(load_entries_from_path(&path).is_empty());
    assert!(load_entry_with_seed(Some(&path), None, "any").is_none());
}

#[test]
fn read_only_and_unwritable_dir_are_non_fatal() {
    let dir = TempDir::new().expect("temp dir");
    let mut cells = BTreeMap::new();
    cells.insert(
        encode_cell_key(512, 0),
        PersistedCell {
            ewma_ms: 190.0,
            samples: 3,
        },
    );
    let entry = PersistedSloCostEntry {
        cells,
        decode_step_ms: Some(40.0),
        global_ratio_sum: 0.6,
        global_ratio_count: 1,
        updated_unix_secs: 100,
    };

    // A path beneath a regular file always fails with ENOTDIR (even under root).
    let blocker = dir.path().join("regular_file_not_a_dir");
    fs::write(&blocker, b"x").expect("create blocker file");
    let unwritable = blocker.join("slo_cost_v1.json");
    assert!(
        !save_entry_atomic(&unwritable, "k", &entry),
        "saving under a non-directory path must fail non-fatally"
    );

    // Also verify a read-only directory path never panics.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let ro_dir = dir.path().join("readonly_dir");
        fs::create_dir(&ro_dir).expect("create ro_dir");
        fs::set_permissions(&ro_dir, fs::Permissions::from_mode(0o555)).expect("chmod 0555");
        let ro_path = ro_dir.join("slo_cost_v1.json");
        let _ = save_entry_atomic(&ro_path, "k", &entry);
        fs::set_permissions(&ro_dir, fs::Permissions::from_mode(0o755)).expect("restore chmod");
    }
}

#[test]
fn concurrent_merge_preserves_sibling_fingerprints() {
    let dir = TempDir::new().expect("temp dir");
    let path = dir.path().join("slo_cost_v1.json");

    std::thread::scope(|scope| {
        for rank in 0..8usize {
            let p = path.clone();
            scope.spawn(move || {
                let fp = sample_fingerprint(&format!("Model-{rank}"), "16384");
                let mut cells = BTreeMap::new();
                cells.insert(
                    encode_cell_key(512, 0),
                    PersistedCell {
                        ewma_ms: 150.0 + rank as f64 * 10.0,
                        samples: 4,
                    },
                );
                let entry = PersistedSloCostEntry {
                    cells,
                    decode_step_ms: Some(35.0 + rank as f64),
                    global_ratio_sum: 0.8,
                    global_ratio_count: 1,
                    updated_unix_secs: 1_700_000_000 + rank as u64,
                };
                assert!(save_entry_atomic(&p, &fp.encoded(), &entry));
            });
        }
    });

    let all = load_entries_from_path(&path);
    assert_eq!(
        all.len(),
        8,
        "all 8 concurrent sibling fingerprints must be preserved under advisory flock"
    );
    for rank in 0..8usize {
        let key = sample_fingerprint(&format!("Model-{rank}"), "16384").encoded();
        let entry = all.get(&key).expect("rank fingerprint present");
        assert!((entry.cells["512:0"].ewma_ms - (150.0 + rank as f64 * 10.0)).abs() < 1e-9);
        assert_eq!(entry.decode_step_ms, Some(35.0 + rank as f64));
    }
}

#[test]
fn tick_zero_predicts_persisted_ms_while_dropping_first_warmup_sample() {
    let mut cells = BTreeMap::new();
    cells.insert(
        encode_cell_key(512, 0),
        PersistedCell {
            ewma_ms: 185.0,
            samples: 12,
        },
    );
    cells.insert(
        encode_cell_key(4096, 0),
        PersistedCell {
            ewma_ms: 540.0,
            samples: 9,
        },
    );
    // Invalid cells (NaN, non-positive, bad key, zero samples) must be skipped safely.
    cells.insert(
        encode_cell_key(256, 0),
        PersistedCell {
            ewma_ms: f64::NAN,
            samples: 5,
        },
    );
    cells.insert(
        encode_cell_key(1024, 0),
        PersistedCell {
            ewma_ms: -10.0,
            samples: 5,
        },
    );
    cells.insert(
        "999:0".to_string(),
        PersistedCell {
            ewma_ms: 100.0,
            samples: 5,
        },
    );
    cells.insert(
        encode_cell_key(2048, 0),
        PersistedCell {
            ewma_ms: 400.0,
            samples: 0,
        },
    );

    let entry = PersistedSloCostEntry {
        cells,
        decode_step_ms: Some(42.0),
        global_ratio_sum: 1.5,
        global_ratio_count: 2,
        updated_unix_secs: 1_700_000_000,
    };

    let mut model = ChunkCostModel::from_persisted(&entry);
    let clock = DecodeStepClock::from_persisted(entry.decode_step_ms);
    assert_eq!(clock.measured_ms(), Some(42.0));
    assert_eq!(clock.step_ms(), 42.0);
    assert_eq!(
        DecodeStepClock::from_persisted(Some(f64::NAN)).measured_ms(),
        None
    );
    assert_eq!(
        DecodeStepClock::from_persisted(Some(-1.0)).measured_ms(),
        None
    );

    // Tick-0 immediately predicts the persisted latency (<600 ms), ignoring invalid cells.
    assert!(model.is_measured(512, 0));
    assert!(model.is_measured(4096, 0));
    assert!(!model.is_measured(256, 0));
    assert!(!model.is_measured(1024, 0));
    assert!(!model.is_measured(2048, 0));
    assert!((model.predict_ms(512, 0) - 185.0).abs() < 1e-9);
    assert!((model.predict_ms(4096, 0) - 540.0).abs() < 1e-9);
    assert!(model.predict_ms(4096, 0) < 600.0);
    // Unmeasured cell uses the persisted global ratio (1.5 / 2 = 0.75).
    assert!((model.predict_ms(2048, 0) - static_chunk_ms(2048, 0) * 0.75).abs() < 1e-6);

    // First post-restart sample is dropped as kernel warmup jitter!
    assert_eq!(model.observe(512, 0, 5_000.0, 185.0), None);
    assert!(
        (model.predict_ms(512, 0) - 185.0).abs() < 1e-9,
        "first post-restart warmup spike must not corrupt persisted prediction"
    );

    // Second post-restart sample blends smoothly into the persisted EWMA.
    let obs = model
        .observe(512, 0, 205.0, 185.0)
        .expect("second post-restart sample accepted");
    let expected = 185.0 + EWMA_ALPHA * (205.0 - 185.0);
    assert!((obs.ewma_ms - expected).abs() < 1e-9);
    assert!((model.predict_ms(512, 0) - expected).abs() < 1e-9);

    // Samples started at MIN_SAMPLES (2) and incremented to 3.
    let reserialized = model.to_persisted(clock.measured_ms());
    assert_eq!(reserialized.cells["512:0"].samples, MIN_SAMPLES + 1);
}

#[test]
fn seed_cache_path_resolution_and_debounce_policy() {
    let dir = TempDir::new().expect("temp dir");
    let seed_path = dir.path().join("seed.json");
    let user_path = dir.path().join("user.json");
    let fp_seed = sample_fingerprint("SeedModel", "16384").encoded();
    let fp_both = sample_fingerprint("BothModel", "16384").encoded();

    let seed_entry = PersistedSloCostEntry {
        decode_step_ms: Some(60.0),
        ..PersistedSloCostEntry::default()
    };
    let user_entry = PersistedSloCostEntry {
        decode_step_ms: Some(44.0),
        ..PersistedSloCostEntry::default()
    };
    assert!(save_entry_atomic(&seed_path, &fp_seed, &seed_entry));
    assert!(save_entry_atomic(&seed_path, &fp_both, &seed_entry));
    assert!(save_entry_atomic(&user_path, &fp_both, &user_entry));

    assert_eq!(
        load_entry_with_seed(Some(&user_path), Some(&seed_path), &fp_seed)
            .and_then(|e| e.decode_step_ms),
        Some(60.0),
        "seed must populate fingerprints absent from user cache"
    );
    assert_eq!(
        load_entry_with_seed(Some(&user_path), Some(&seed_path), &fp_both)
            .and_then(|e| e.decode_step_ms),
        Some(44.0),
        "user cache must win over bundled seed on conflict"
    );

    assert_eq!(
        resolve_cache_path_with(
            Some("0"),
            Some("/tmp/override.json"),
            Some(OsStr::new("/h"))
        ),
        None
    );
    assert_eq!(
        resolve_cache_path_with(None, Some("/tmp/override.json"), Some(OsStr::new("/h"))),
        Some(PathBuf::from("/tmp/override.json"))
    );
    assert_eq!(
        resolve_cache_path_with(None, None, Some(OsStr::new("/tmp/user_root"))),
        Some(PathBuf::from("/tmp/user_root/.cache/flux/slo_cost_v1.json"))
    );
    assert_eq!(resolve_cache_path_with(None, None, None), None);
    assert_eq!(
        resolve_seed_path_with(true, false),
        Some(PathBuf::from(BUNDLED_SEED_CACHE_PATH))
    );
    assert_eq!(resolve_seed_path_with(true, true), None);
    assert_eq!(resolve_seed_path_with(false, false), None);

    // Exercise runtime fingerprint + env helpers to keep non-test paths verified.
    let runtime_fp = SloCostFingerprint::for_runtime(&[512, 1024], 16);
    assert!(runtime_fp.encoded().starts_with("v1|gpu="));
    let _ = resolve_cache_path();
    let _ = resolve_seed_path(None);

    // Debounce triggers on 16th observation or >= 30s elapsed.
    let mut handle = SloCacheHandle::new(user_path, fp_both);
    let t0 = Instant::now();
    for _ in 0..(SAVE_OBSERVATION_INTERVAL - 1) {
        assert!(!handle.record_observation(t0));
    }
    assert!(handle.record_observation(t0));

    let mut time_handle = SloCacheHandle::new(dir.path().join("t.json"), fp_seed);
    let t1 = Instant::now();
    assert!(!time_handle.record_observation(t1));
    assert!(time_handle.record_observation(t1 + Duration::from_secs(SAVE_TIME_INTERVAL_SECS)));
}
