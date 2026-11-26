// SPDX-License-Identifier: AGPL-3.0-only

//! Settings-file hardening: stale keys, symlinks, temp files.

use serde_json::{Value, json};

use super::*;

fn persisted(path: &std::path::Path, model: &str) -> RuntimeSettings {
    RuntimeSettings::new(
        RuntimeSettings::for_tests(65_536).startup.clone(),
        NudgeControl::unavailable(),
        Some(SettingsFile::new(path.to_path_buf())),
        model.into(),
    )
}

#[test]
fn a_stale_settings_file_skips_unknown_keys_and_fails_on_bad_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    // Renamed / removed / restart-only keys: skipped, the rest applies, and
    // the rewritten file drops them.
    std::fs::write(
        &path,
        r#"{"schema":"flux.runtime_settings.v1","model":"m","settings":
           {"sampling.tools.old_name":1,"max_seq_len":4,"max_tokens.default":512}}"#,
    )
    .unwrap();
    let s = persisted(&path, "m");
    s.load_persisted().unwrap();
    assert_eq!(s.snapshot().default_max_tokens, Some(512));
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(doc["settings"], json!({"max_tokens.default": 512}));
    // A known key with a bad value fails the load.
    std::fs::write(
        &path,
        r#"{"schema":"flux.runtime_settings.v1","model":"m","settings":{"max_tokens.default":-1}}"#,
    )
    .unwrap();
    assert!(persisted(&path, "m").load_persisted().is_err());
    std::fs::write(&path, "not json").unwrap();
    assert!(persisted(&path, "m").load_persisted().is_err());
}

#[test]
fn the_settings_write_never_follows_a_planted_symlink() {
    let dir = tempfile::tempdir().unwrap();
    let victim = dir.path().join("victim");
    std::fs::write(&victim, "keep").unwrap();
    let path = dir.path().join("settings.json");
    // A symlink where the settings file goes is replaced, not written through.
    std::os::unix::fs::symlink(&victim, &path).unwrap();
    persisted(&path, "m")
        .apply_patch(&json!({"max_tokens.default": 100}))
        .unwrap();
    assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep");
    assert!(
        !std::fs::symlink_metadata(&path)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    // No temp files are left behind.
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty());
}
