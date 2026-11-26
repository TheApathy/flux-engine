// SPDX-License-Identifier: AGPL-3.0-only

use std::io::Write;
use std::sync::Arc;

use parking_lot::Mutex;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;

use super::*;
use crate::load_debug::{self, LoadCapture, Mode};

#[derive(Clone, Default)]
struct Buf(Arc<Mutex<Vec<u8>>>);

impl Buf {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().clone()).unwrap()
    }
}

impl Write for Buf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Buf {
    type Writer = Buf;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("debug-log-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A start-and-serve sequence: banner-style INFO, WARN, ERROR, debug noise,
/// span context and structured fields, as the server emits them.
fn emit_startup_and_serving() {
    tracing::info!("Atlas Spark starting...");
    tracing::debug!(shard = 3, "safetensors header parsed");
    tracing::warn!("Atlas is listening on 0.0.0.0:8899 — reachable from any host");
    let span = tracing::info_span!("request", id = 7);
    let _g = span.enter();
    tracing::info!(prompt_tokens = 12, "Prefilling");
    tracing::debug!(tokens = ?[1, 2, 3], "DECODE");
    tracing::error!("synthetic error line");
    tracing::trace!("trace never shows");
}

/// Replace each RFC 3339 timestamp (`2026-10-29T13:33:09.565124Z`) with
/// `TS`; it is the only thing that differs run to run.
fn without_timestamps(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        let looks_like_ts = bytes.len() - i > 20
            && bytes[i..i + 4].iter().all(u8::is_ascii_digit)
            && bytes[i + 4] == b'-'
            && bytes[i + 10] == b'T';
        if looks_like_ts && let Some(z) = text[i..].find('Z') {
            out.push_str("TS");
            i += z + 1;
            continue;
        }
        let ch = text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

#[test]
fn console_is_byte_identical_to_the_old_builder_with_both_layers_on() {
    // Before: exactly what main.rs installs when both features are off.
    let before = Buf::default();
    let old = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .with_writer(before.clone())
        .finish();
    tracing::subscriber::with_default(old, emit_startup_and_serving);

    // After: the console layer beside the load capture and the debug file.
    let dir = temp_dir("identity");
    let path = dir.join("flux-debug.log");
    let file = Arc::new(Mutex::new(RotatingFile::open(&path, 1 << 20).unwrap()));
    let capture = LoadCapture::new(Mode::Auto, None);
    let after = Buf::default();
    let new = tracing_subscriber::registry()
        .with(console_layer(after.clone(), EnvFilter::new("info")))
        .with(load_debug::layer_for(capture.clone()))
        .with(layer_for(file, EnvFilter::new("debug")));
    tracing::subscriber::with_default(new, emit_startup_and_serving);

    let (b, a) = (
        without_timestamps(&before.text()),
        without_timestamps(&after.text()),
    );
    assert_eq!(
        b.lines().count(),
        4,
        "info, warn, info, error expected: {b}"
    );
    assert_eq!(a, b, "console output changed");
    assert!(!a.contains("safetensors header parsed"));

    // The debug file gets everything at debug and above, nothing at trace.
    let logged = std::fs::read_to_string(&path).unwrap();
    for needle in [
        "Atlas Spark starting",
        "safetensors header parsed",
        "DECODE",
        "synthetic error line",
    ] {
        assert!(
            logged.contains(needle),
            "{needle} missing from debug file: {logged}"
        );
    }
    assert!(!logged.contains("trace never shows"));
    assert!(
        !logged.contains('\u{1b}'),
        "debug file must not carry ANSI codes"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn rust_log_still_sets_the_console_with_the_debug_file_on() {
    let dir = temp_dir("rustlog");
    let path = dir.join("flux-debug.log");
    let file = Arc::new(Mutex::new(RotatingFile::open(&path, 1 << 20).unwrap()));
    let console = Buf::default();
    let s = tracing_subscriber::registry()
        .with(console_layer(console.clone(), EnvFilter::new("warn")))
        .with(layer_for(file, EnvFilter::new("debug")));
    tracing::subscriber::with_default(s, emit_startup_and_serving);
    let shown = console.text();
    assert!(shown.contains("reachable from any host"));
    assert!(!shown.contains("Atlas Spark starting"));
    assert!(
        std::fs::read_to_string(&path)
            .unwrap()
            .contains("Atlas Spark starting")
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn file_rotates_once_at_the_cap() {
    let dir = temp_dir("rotate");
    let path = dir.join("flux-debug.log");
    let mut f = RotatingFile::open(&path, 100).unwrap();
    for i in 0..30 {
        f.write_line(format!("line {i:02}\n").as_bytes()); // 8 bytes each
    }
    let current = std::fs::read_to_string(&path).unwrap();
    let rotated = std::fs::read_to_string(dir.join("flux-debug.log.1")).unwrap();
    assert!(current.len() <= 100 && rotated.len() <= 100);
    assert!(current.ends_with("line 29\n"));
    // Only one generation is kept, so total disk stays under twice the cap.
    assert!(!dir.join("flux-debug.log.2").exists());
    assert!(
        !rotated.contains("line 00"),
        "oldest lines were dropped by the second rotation"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn reopening_counts_existing_bytes_toward_the_cap() {
    let dir = temp_dir("reopen");
    let path = dir.join("flux-debug.log");
    std::fs::write(&path, vec![b'x'; 95]).unwrap();
    let mut f = RotatingFile::open(&path, 100).unwrap();
    f.write_line(b"0123456789\n");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "0123456789\n");
    assert_eq!(
        std::fs::read(dir.join("flux-debug.log.1")).unwrap().len(),
        95
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn switches_parse() {
    for on in ["1", "true", "ON", " yes "] {
        assert_eq!(parse_enabled(Some(on)), Ok(true), "{on}");
    }
    for off in ["", "0", "false", "off", "no"] {
        assert_eq!(parse_enabled(Some(off)), Ok(false), "{off}");
    }
    assert_eq!(parse_enabled(None), Ok(false));
    assert!(parse_enabled(Some("debug")).is_err());
    assert_eq!(parse_max_bytes(None), Ok(256 << 20));
    assert_eq!(parse_max_bytes(Some("8")), Ok(8 << 20));
    assert!(parse_max_bytes(Some("0")).is_err());
    assert!(parse_max_bytes(Some("lots")).is_err());
}

#[test]
fn default_filter_keeps_model_and_kernel_crates_at_info() {
    let spec = default_filter_spec();
    assert!(!filter_enables_model_debug(&spec), "{spec}");
    assert!(filter_enables_model_debug("debug"));
    assert!(filter_enables_model_debug("info,spark_model=debug"));
    assert!(filter_enables_model_debug("trace"));
    assert!(!filter_enables_model_debug("info"));
    assert!(!filter_enables_model_debug(
        "debug,spark_model=info,spark_runtime=info,atlas=info"
    ));

    let dir = temp_dir("model-info");
    let path = dir.join("flux-debug.log");
    let file = Arc::new(Mutex::new(RotatingFile::open(&path, 1 << 20).unwrap()));
    let s = tracing_subscriber::registry()
        .with(console_layer(Buf::default(), EnvFilter::new("info")))
        .with(layer_for(file, EnvFilter::new(spec)));
    tracing::subscriber::with_default(s, || {
        for target in [
            "spark_model::layers::qwen3_ssm::ssm_forward",
            "spark_model::layers::moe::forward",
            "spark_runtime::cuda_backend",
            "atlas_kernels::launch",
        ] {
            assert!(!debug_enabled_for(target), "{target}");
        }
        assert!(debug_enabled_for("spark::scheduler"));
        assert!(debug_enabled_for("atlas::tui::progress"));
    });
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[test]
fn debug_file_and_its_rotation_are_owner_only() {
    use std::os::unix::fs::PermissionsExt;
    let dir = temp_dir("mode");
    let path = dir.join("flux-debug.log");
    let mut f = RotatingFile::open(&path, 16).unwrap();
    f.write_line(b"0123456789\n");
    f.write_line(b"0123456789\n"); // rotates
    for p in [path.clone(), dir.join("flux-debug.log.1")] {
        let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "{}", p.display());
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[test]
fn failed_rotation_switches_the_log_off_instead_of_growing() {
    use std::os::unix::fs::PermissionsExt;
    let dir = temp_dir("failclosed");
    let path = dir.join("flux-debug.log");
    let mut f = RotatingFile::open(&path, 16).unwrap();
    f.write_line(b"0123456789\n");
    // A read-only directory makes the rename fail.
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o500)).unwrap();
    f.write_line(b"0123456789\n");
    f.write_line(b"0123456789\n");
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(!f.is_open());
    assert_eq!(
        std::fs::read(&path).unwrap().len(),
        11,
        "file grew past the cap"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn ranks_above_zero_get_their_own_file() {
    assert_eq!(
        per_rank_path("/results/flux-debug.log", 0),
        "/results/flux-debug.log"
    );
    assert_eq!(
        per_rank_path("/results/flux-debug.log", 3),
        "/results/flux-debug.log.rank3"
    );
}

#[cfg(unix)]
#[test]
fn private_files_take_the_directory_owner() {
    use std::os::unix::fs::MetadataExt;
    let dir = temp_dir("owner");
    let path = dir.join("flux-debug.log");
    let _f = open_private(&path, true).unwrap();
    let (meta, dmeta) = (
        std::fs::metadata(&path).unwrap(),
        std::fs::metadata(&dir).unwrap(),
    );
    assert_eq!((meta.uid(), meta.gid()), (dmeta.uid(), dmeta.gid()));
    assert_eq!(meta.mode() & 0o777, 0o600);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[test]
fn refused_chown_falls_back_to_0640() {
    use std::os::unix::fs::MetadataExt;
    let dir = temp_dir("owner-refused");
    let path = dir.join("flux-debug.log");
    let f = open_private(&path, true).unwrap();
    // Unknown owner stands in for a refused chown (as a non-root test user,
    // a chown to another uid would be refused the same way).
    adopt_owner(&f, &path, None);
    assert_eq!(std::fs::metadata(&path).unwrap().mode() & 0o777, 0o640);
    std::fs::remove_dir_all(&dir).unwrap();
}
