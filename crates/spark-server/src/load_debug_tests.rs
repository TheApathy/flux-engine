// SPDX-License-Identifier: AGPL-3.0-only

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use tracing::Subscriber;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer as _;
use tracing_subscriber::layer::SubscriberExt;

use super::*;

/// A console stand-in: what the fmt layer would have printed to stdout.
#[derive(Clone, Default)]
struct Console(Arc<Mutex<Vec<u8>>>);

impl Console {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().clone()).unwrap()
    }
}

impl Write for Console {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Console {
    type Writer = Console;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

fn stack(console: &Console, capture: &Arc<LoadCapture>) -> impl Subscriber + Send + Sync {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(console.clone())
                .with_filter(EnvFilter::new("info")),
        )
        .with(layer_for(capture.clone()))
}

fn report(capture: &LoadCapture) -> Option<String> {
    let mut out = Vec::new();
    capture
        .report_failure(&mut out, "model load failed")
        .then(|| String::from_utf8(out).unwrap())
}

#[test]
fn mode_parse_accepts_the_three_values_and_defaults_to_auto() {
    assert_eq!(Mode::parse(None), Ok(Mode::Auto));
    assert_eq!(Mode::parse(Some("")), Ok(Mode::Auto));
    assert_eq!(Mode::parse(Some("auto")), Ok(Mode::Auto));
    assert_eq!(Mode::parse(Some(" Always ")), Ok(Mode::Always));
    assert_eq!(Mode::parse(Some("OFF")), Ok(Mode::Off));
    assert!(
        Mode::parse(Some("debug"))
            .unwrap_err()
            .contains("auto|always|off")
    );
}

#[test]
fn console_keeps_its_filter_while_capture_sees_debug() {
    let console = Console::default();
    let capture = LoadCapture::new(Mode::Auto, None);
    tracing::subscriber::with_default(stack(&console, &capture), || {
        let span = tracing::info_span!("load", phase = 2);
        let _g = span.enter();
        tracing::debug!("shard index parsed");
        tracing::info!("Loading weights");
        tracing::trace!("trace is never captured");
    });
    let shown = console.text();
    assert!(shown.contains("Loading weights"));
    assert!(
        !shown.contains("shard index parsed"),
        "debug leaked to console: {shown}"
    );

    let dump = report(&capture).expect("active capture reports");
    assert!(dump.contains("shard index parsed"));
    assert!(dump.contains("Loading weights"));
    assert!(!dump.contains("trace is never captured"));
    assert!(dump.contains("last 2 of 2 lines"));
    assert!(!dump.contains('\u{1b}'), "ring must not carry ANSI codes");
    // A second report (e.g. the abort hook after main's error path) is silent.
    assert_eq!(report(&capture), None);
}

#[test]
fn finished_capture_drops_its_hint_and_records_nothing() {
    let console = Console::default();
    let capture = LoadCapture::new(Mode::Auto, None);
    let subscriber = stack(&console, &capture);
    assert_eq!(subscriber.max_level_hint(), Some(LevelFilter::DEBUG));

    assert!(capture.deactivate());
    assert!(!capture.deactivate(), "deactivate is one-shot");
    // The global max level is the max over layer hints; with the capture
    // OFF it is the console filter's again.
    assert_eq!(subscriber.max_level_hint(), Some(LevelFilter::INFO));

    tracing::subscriber::with_default(subscriber, || {
        tracing::callsite::rebuild_interest_cache();
        tracing::debug!("serving-time debug");
        tracing::info!("serving-time info");
    });
    assert!(console.text().contains("serving-time info"));
    assert!(!console.text().contains("serving-time debug"));
    assert_eq!(report(&capture), None, "a successful load never dumps");
}

#[test]
fn off_mode_captures_nothing() {
    let capture = LoadCapture::new(Mode::Off, None);
    assert!(!capture.is_active());
    tracing::subscriber::with_default(stack(&Console::default(), &capture), || {
        tracing::debug!("ignored");
    });
    assert_eq!(report(&capture), None);
}

#[test]
fn ring_is_bounded_and_counts_dropped_lines() {
    let mut ring = Ring::default();
    for i in 0..(RING_MAX_LINES + 5) {
        ring.push(format!("line {i}\n"));
    }
    assert_eq!(ring.lines.len(), RING_MAX_LINES);
    assert_eq!(ring.dropped, 5);
    assert_eq!(ring.lines.front().unwrap(), "line 5\n");

    let mut ring = Ring::default();
    ring.push("x".repeat(RING_MAX_BYTES + 1));
    ring.push("tail\n".into());
    assert_eq!(ring.lines.len(), 1);
    assert!(ring.bytes <= RING_MAX_BYTES);
}

#[test]
fn file_sink_keeps_every_line_and_marks_the_dump() {
    let dir = std::env::temp_dir().join(format!("load-debug-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("flux-load-debug.log");
    let capture = LoadCapture::new(Mode::Auto, Some(path.display().to_string()));
    tracing::subscriber::with_default(stack(&Console::default(), &capture), || {
        tracing::debug!("before failure");
    });
    let dump = report(&capture).unwrap();
    assert!(dump.contains(&format!("full log: {}", path.display())));
    let file = std::fs::read_to_string(&path).unwrap();
    assert!(file.contains("before failure"));
    assert_eq!(file.lines().last(), Some(DUMPED_MARKER));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn successful_load_closes_the_file_with_the_loaded_marker() {
    let dir = std::env::temp_dir().join(format!("load-debug-ok-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("flux-load-debug.log");
    let capture = LoadCapture::new(Mode::Auto, Some(path.display().to_string()));
    tracing::subscriber::with_default(stack(&Console::default(), &capture), || {
        tracing::debug!("loaded fine");
    });
    assert!(capture.deactivate());
    let file = std::fs::read_to_string(&path).unwrap();
    assert!(file.contains("loaded fine"));
    assert!(!file.contains(DUMPED_MARKER));
    assert_eq!(file.lines().last(), Some(LOADED_MARKER));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn always_mode_reports_a_footer_instead_of_repeating_the_lines() {
    let capture = LoadCapture::new(Mode::Always, None);
    capture.push(b"streamed already\n");
    let dump = report(&capture).unwrap();
    assert!(dump.contains("streamed above"));
    assert!(!dump.contains("streamed already"));
}

#[test]
fn capture_never_enables_debug_for_model_and_kernel_crates() {
    let console = Console::default();
    let capture = LoadCapture::new(Mode::Auto, None);
    tracing::subscriber::with_default(stack(&console, &capture), || {
        // What spark_model's GPU-syncing `enabled!(DEBUG)` paths would see.
        use crate::debug_log::debug_enabled_for;
        for target in [
            "spark_model::layers::moe::forward",
            "spark_model::layers::qwen3_ssm::trait_decode",
            "spark_runtime::cuda_backend",
            "atlas_kernels::launch",
            "atlas::qwen4_ple",
            "atlas_bounded_transpose_allocation",
        ] {
            assert!(!debug_enabled_for(target), "{target}");
        }
        // The rest of the load still gets debug.
        assert!(debug_enabled_for("spark::main_modules::serve"));
        assert!(debug_enabled_for("atlas::tui::progress"));
        assert!(debug_enabled_for("hyper::client"));
    });
    assert!(report(&capture).is_some());
}

#[test]
fn bind_failure_is_reported_as_one() {
    let capture = LoadCapture::new(Mode::Auto, None);
    assert_eq!(capture.failure_reason(), "model load failed");
    capture.binding.store(true, Ordering::SeqCst);
    assert_eq!(
        capture.failure_reason(),
        "server failed to bind its listener"
    );
}

#[cfg(unix)]
#[test]
fn load_debug_file_is_owner_only_even_if_it_existed() {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!("load-debug-mode-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("flux-load-debug.log");
    std::fs::write(&path, "old").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    let _capture = LoadCapture::new(Mode::Auto, Some(path.display().to_string()));
    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
    std::fs::remove_dir_all(&dir).unwrap();
}
