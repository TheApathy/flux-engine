// SPDX-License-Identifier: AGPL-3.0-only

//! The local admin token: required for every keyless write, never shown.

use super::tests_gate::{TOKEN, gate_code_with, headers, peer};
use super::*;

const LOCAL_HOST: (&str, &str) = ("host", "127.0.0.1:8899");

#[test]
fn a_keyless_write_without_the_token_is_refused() {
    assert_eq!(
        gate_code_with(false, "127.0.0.1:1", &[LOCAL_HOST], None, Some(TOKEN)).as_deref(),
        Some("admin_token_required")
    );
}

#[test]
fn a_keyless_write_with_a_wrong_token_is_refused() {
    let short = TOKEN[..10].to_string();
    let flipped = format!("{}0", &TOKEN[..TOKEN.len() - 1]);
    for wrong in ["", "nope", short.as_str(), flipped.as_str()] {
        let hdrs = [LOCAL_HOST, (local_admin::HEADER, wrong)];
        assert_eq!(
            gate_code_with(false, "127.0.0.1:1", &hdrs, None, Some(TOKEN)).as_deref(),
            Some("admin_token_invalid"),
            "{wrong:?}"
        );
    }
}

#[test]
fn the_right_token_from_loopback_is_accepted() {
    let hdrs = [LOCAL_HOST, (local_admin::HEADER, TOKEN)];
    assert_eq!(
        gate_code_with(false, "127.0.0.1:1", &hdrs, None, Some(TOKEN)),
        None
    );
    // The token alone is not enough: the network-facing checks still apply.
    assert_eq!(
        gate_code_with(false, "172.17.0.1:1", &hdrs, None, Some(TOKEN)).as_deref(),
        Some("loopback_only")
    );
    let proxied = [
        LOCAL_HOST,
        (local_admin::HEADER, TOKEN),
        ("via", "1.1 nginx"),
    ];
    assert_eq!(
        gate_code_with(false, "127.0.0.1:1", &proxied, None, Some(TOKEN)).as_deref(),
        Some("proxied_request")
    );
}

#[test]
fn without_a_token_file_keyless_writes_are_refused_and_the_api_key_still_works() {
    let hdrs = [LOCAL_HOST, (local_admin::HEADER, TOKEN)];
    assert_eq!(
        gate_code_with(false, "127.0.0.1:1", &hdrs, None, None).as_deref(),
        Some("admin_token_unavailable")
    );
    assert_eq!(gate_code_with(true, "203.0.113.5:1", &[], None, None), None);
    assert!(handlers::ADMIN_TOKEN_MESSAGE.contains("docker exec"));
}

#[test]
fn token_comparison_is_exact() {
    assert!(local_admin::matches(TOKEN, TOKEN.as_bytes()));
    assert!(!local_admin::matches(TOKEN, b""));
    assert!(!local_admin::matches(TOKEN, &TOKEN.as_bytes()[1..]));
    assert!(!local_admin::matches(TOKEN, format!("{TOKEN}0").as_bytes()));
}

#[test]
fn the_token_file_is_private_and_replaces_whatever_was_there() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("run")
        .join("flux")
        .join(local_admin::file_name(8899));
    local_admin::write_at(&path, "first").unwrap();
    let mode = |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&path), 0o600);
    assert_eq!(mode(path.parent().unwrap()), 0o700);
    // Regenerated each start: the old token is gone.
    local_admin::write_at(&path, "second").unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
    // A planted symlink is replaced, never written through.
    let victim = dir.path().join("victim");
    std::fs::write(&victim, "keep").unwrap();
    std::fs::remove_file(&path).unwrap();
    std::os::unix::fs::symlink(&victim, &path).unwrap();
    local_admin::write_at(&path, "third").unwrap();
    assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "third");
}

/// A `tracing` writer that keeps everything logged, for leak checks.
#[derive(Clone, Default)]
struct Capture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn the_token_never_appears_in_get_output_or_logs() {
    let settings = RuntimeSettings::for_tests(4096);
    // `tracing` caches per-callsite interest process-wide; a callsite another
    // test thread hit first, with no subscriber in scope, can stay "never"
    // for a moment even after `rebuild_interest_cache`. Retry until the
    // capture demonstrably works, and check every attempt for leaks.
    let mut captured = false;
    for _ in 0..20 {
        let capture = Capture::default();
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_writer(move || writer.clone())
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            tracing::callsite::rebuild_interest_cache();
            for presented in [TOKEN, "wrong-token-value"] {
                let h = headers(&[LOCAL_HOST, (local_admin::HEADER, presented)]);
                let _ = handlers::WriteGate {
                    api_key_configured: false,
                    peer: peer("127.0.0.1:1"),
                    headers: &h,
                    bound: None,
                    admin_token: Some(TOKEN),
                }
                .authorize();
            }
            let _ = handlers::patch(
                &settings,
                br#"{"max_tokens.default": 100}"#,
                peer("127.0.0.1:1"),
            );
            let _ = handlers::patch(&settings, br#"{"max_seq_len": 1}"#, peer("127.0.0.1:1"));
            let _ = handlers::patch(
                &settings,
                br#"{"max_tokens.default": null}"#,
                peer("127.0.0.1:1"),
            );
        });
        let logs = String::from_utf8(capture.0.lock().unwrap().clone()).unwrap();
        assert!(!logs.contains(TOKEN), "{logs}");
        assert!(!logs.contains("wrong-token-value"), "{logs}");
        if logs.contains("Runtime setting changed") {
            captured = true;
            break;
        }
    }
    assert!(captured, "the capture subscriber never saw the change log");
    let get = settings.describe().to_string();
    assert!(!get.contains(TOKEN));
    assert!(!get.to_lowercase().contains("admin"));
}

#[test]
fn the_token_file_is_named_per_port() {
    assert_eq!(local_admin::file_name(8899), "admin.8899.token");
    assert_ne!(local_admin::file_name(8899), local_admin::file_name(8900));
}

#[test]
fn a_shared_or_symlinked_token_directory_is_refused() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    // Group- or world-writable: another local user could swap the file.
    for mode in [0o770, 0o777, 0o722] {
        let dir = root.path().join(format!("shared{mode:o}"));
        std::fs::create_dir(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(mode)).unwrap();
        assert!(local_admin::dir_is_private(&dir).is_err(), "{mode:o}");
        assert!(local_admin::write_at(&dir.join("admin.1.token"), "t").is_err());
        assert!(!dir.join("admin.1.token").exists());
    }
    // A symlink to a directory is not accepted as the directory.
    let real = root.path().join("real");
    std::fs::create_dir(&real).unwrap();
    std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o700)).unwrap();
    let link = root.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    assert!(local_admin::dir_is_private(&link).is_err());
    assert!(local_admin::write_at(&link.join("admin.1.token"), "t").is_err());
    // Ours and private: accepted.
    assert!(local_admin::dir_is_private(&real).is_ok());
    local_admin::write_at(&real.join("admin.1.token"), "t").unwrap();
}
