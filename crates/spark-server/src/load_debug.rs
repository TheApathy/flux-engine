// SPDX-License-Identifier: AGPL-3.0-only

//! Load-phase debug capture: debug output only when the first load fails.
//!
//! While the first model loads, debug-level events are formatted into a
//! bounded in-memory ring (and, when `FLUX_LOAD_DEBUG_FILE` names one, a file
//! that survives a SIGKILL). The console keeps its normal `RUST_LOG` filter,
//! so a clean start prints exactly what it printed before. If startup fails —
//! `serve()` returns an error, or a panic aborts the process — the ring is
//! written to stderr ahead of the error, together with a backtrace for a
//! panic. Once the listener is bound the capture is switched off for good:
//! its filter reports `OFF`, the callsite interest cache is rebuilt, and the
//! global max level drops back to the console filter's.
//!
//! `FLUX_LOAD_DEBUG`: `auto` (default) captures and
//! dumps only on failure; `always` also streams the captured lines to stderr
//! live; `off` installs nothing and leaves the pre-existing subscriber stack
//! untouched.
//!
//! `FLUX_LOAD_STALL_SECS` (default 600, `0` = off): if no load progress (no
//! captured line) arrives for that long, the captured lines and the current
//! load stage are printed once to stderr and the load keeps waiting. Nothing
//! is killed; a hung load is just no longer silent. Progress re-arms it.

use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tracing::level_filters::LevelFilter;
use tracing::subscriber::Interest;
use tracing::{Metadata, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::{Context, Filter};
use tracing_subscriber::registry::LookupSpan;

/// Lines kept for the console dump. A failed load's cause is in its last
/// few hundred lines; the file sink, when configured, keeps everything.
const RING_MAX_LINES: usize = 1000;
/// Byte ceiling on the ring, so a few huge debug lines cannot balloon it.
const RING_MAX_BYTES: usize = 512 * 1024;
/// Written as the last line of the file sink once the ring has gone to
/// stderr, so the launcher knows not to print the file's tail a second time.
pub const DUMPED_MARKER: &str = "flux-load-debug: dumped to stderr";
/// Written as the last line once the listener is up, so a later crash is not
/// mistaken for a failed load.
pub const LOADED_MARKER: &str = "flux-load-debug: load complete";
/// Default seconds without load progress before the stall report.
pub const DEFAULT_STALL_SECS: u64 = 600;

/// `FLUX_LOAD_STALL_SECS`: `None` = off (`0`); unset or empty = the default.
pub fn parse_stall_secs(raw: Option<&str>) -> Result<Option<Duration>, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(Some(Duration::from_secs(DEFAULT_STALL_SECS))),
        Some(value) => match value.parse::<u64>() {
            Ok(0) => Ok(None),
            Ok(secs) => Ok(Some(Duration::from_secs(secs))),
            Err(_) => Err(format!(
                "FLUX_LOAD_STALL_SECS={value:?} is not a whole number of seconds; using {DEFAULT_STALL_SECS}"
            )),
        },
    }
}

/// The current load stage, from a captured `ev="phase" ... name="<stage>"` line.
fn phase_name(line: &str) -> Option<&str> {
    if !line.contains("ev=\"phase\"") {
        return None;
    }
    let rest = &line[line.find("name=\"")? + "name=\"".len()..];
    Some(&rest[..rest.find('"')?])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Auto,
    Always,
    Off,
}

impl Mode {
    /// Parse the env value; unset or empty means `auto`.
    pub fn parse(raw: Option<&str>) -> Result<Mode, String> {
        match raw.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("auto") => Ok(Mode::Auto),
            Some("always") => Ok(Mode::Always),
            Some("off") => Ok(Mode::Off),
            Some(other) => Err(format!(
                "FLUX_LOAD_DEBUG={other:?} is not one of auto|always|off; using auto"
            )),
        }
    }

    /// The process setting, `FLUX_LOAD_DEBUG` (not aliased to `ATLAS_`).
    pub fn from_env() -> Mode {
        let raw = std::env::var("FLUX_LOAD_DEBUG").ok();
        Mode::parse(raw.as_deref()).unwrap_or_else(|msg| {
            eprintln!("[flux] WARN: {msg}");
            Mode::Auto
        })
    }
}

#[derive(Default)]
struct Ring {
    lines: VecDeque<String>,
    bytes: usize,
    dropped: usize,
}

impl Ring {
    fn push(&mut self, line: String) {
        self.bytes += line.len();
        self.lines.push_back(line);
        while self.lines.len() > RING_MAX_LINES || self.bytes > RING_MAX_BYTES {
            let Some(old) = self.lines.pop_front() else {
                break;
            };
            self.bytes -= old.len();
            self.dropped += 1;
        }
    }
}

/// One load's capture state. The process uses a single instance (see
/// [`install`]); tests build their own.
pub struct LoadCapture {
    active: AtomicBool,
    /// Set once the model is up and only the listener bind remains, so a
    /// bind failure is reported as one rather than as a failed load.
    binding: AtomicBool,
    echo: bool,
    ring: Mutex<Ring>,
    file: Mutex<Option<File>>,
    file_path: Option<String>,
    /// When the last line arrived, and the last load stage named.
    progress: Mutex<(Instant, Option<String>)>,
    /// The current stall has been reported; progress clears it.
    stall_reported: AtomicBool,
}

impl LoadCapture {
    pub fn new(mode: Mode, file_path: Option<String>) -> Arc<Self> {
        let file = file_path.as_deref().and_then(|p| {
            match crate::debug_log::open_private(std::path::Path::new(p), false) {
                Ok(f) => Some(f),
                Err(e) => {
                    eprintln!("[flux] WARN: cannot open load-debug file {p}: {e}");
                    None
                }
            }
        });
        let file_path = file.as_ref().and(file_path);
        Arc::new(Self {
            active: AtomicBool::new(mode != Mode::Off),
            binding: AtomicBool::new(false),
            echo: mode == Mode::Always,
            ring: Mutex::new(Ring::default()),
            file: Mutex::new(file),
            file_path,
            progress: Mutex::new((Instant::now(), None)),
            stall_reported: AtomicBool::new(false),
        })
    }

    fn failure_reason(&self) -> &'static str {
        if self.binding.load(Ordering::SeqCst) {
            "server failed to bind its listener"
        } else {
            "model load failed"
        }
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    fn push(&self, buf: &[u8]) {
        if !self.is_active() {
            return;
        }
        if let Some(f) = self.file.lock().as_mut() {
            let _ = f.write_all(buf);
        }
        if self.echo {
            let _ = std::io::stderr().write_all(buf);
        }
        let line = String::from_utf8_lossy(buf).into_owned();
        {
            let mut progress = self.progress.lock();
            progress.0 = Instant::now();
            if let Some(stage) = phase_name(&line) {
                progress.1 = Some(stage.to_string());
            }
        }
        self.stall_reported.store(false, Ordering::SeqCst);
        self.ring.lock().push(line);
    }

    /// If the load has made no progress for `stall` as of `now`, write the
    /// captured lines and the current stage to `out` once per stall and keep
    /// the capture running. Returns whether a report was written.
    pub fn report_stall(&self, now: Instant, stall: Duration, out: &mut dyn Write) -> bool {
        if !self.is_active() {
            return false;
        }
        let (last, stage) = {
            let progress = self.progress.lock();
            (progress.0, progress.1.clone())
        };
        let idle = now.saturating_duration_since(last);
        if idle < stall || self.stall_reported.swap(true, Ordering::SeqCst) {
            return false;
        }
        let stage = stage.as_deref().unwrap_or("before the first stage");
        let full = match &self.file_path {
            Some(p) => format!(" (full log: {p})"),
            None => String::new(),
        };
        let _ = writeln!(
            out,
            "\n[flux] WARN: no load progress for {}s (stage: {stage}); still waiting, nothing was stopped",
            idle.as_secs()
        );
        if self.echo {
            let _ = writeln!(out, "[flux] load-phase debug log streamed above{full}");
        } else {
            let lines: Vec<String> = self.ring.lock().lines.iter().cloned().collect();
            let _ = writeln!(
                out,
                "[flux] ===== load-phase debug log so far, last {} lines =====",
                lines.len()
            );
            for line in &lines {
                let _ = out.write_all(line.as_bytes());
            }
            let _ = writeln!(
                out,
                "[flux] ===== end of load-phase debug log so far{full} ====="
            );
        }
        let _ = out.flush();
        true
    }

    /// The load succeeded: stop capturing and free the ring. Returns false if
    /// the capture had already ended. Callers must then rebuild the callsite
    /// interest cache (see [`finish_load`]); kept separate so tests can drive
    /// the state without touching process-global tracing state.
    fn deactivate(&self) -> bool {
        if !self.active.swap(false, Ordering::SeqCst) {
            return false;
        }
        *self.ring.lock() = Ring::default();
        if let Some(mut f) = self.file.lock().take() {
            let _ = writeln!(f, "{LOADED_MARKER}");
        }
        true
    }

    /// Write the captured lines to `out` and end the capture. Returns false
    /// (writing nothing) if the capture had already ended.
    pub fn report_failure(&self, out: &mut dyn Write, reason: &str) -> bool {
        if !self.active.swap(false, Ordering::SeqCst) {
            return false;
        }
        let ring = std::mem::take(&mut *self.ring.lock());
        let full = match &self.file_path {
            Some(p) => format!(" (full log: {p})"),
            None => String::new(),
        };
        if self.echo {
            let _ = writeln!(
                out,
                "[flux] ===== {reason}: load-phase debug log streamed above{full} ====="
            );
        } else {
            let total = ring.lines.len() + ring.dropped;
            let _ = writeln!(
                out,
                "\n[flux] ===== {reason}: load-phase debug log, last {} of {total} lines =====",
                ring.lines.len()
            );
            for line in &ring.lines {
                let _ = out.write_all(line.as_bytes());
            }
            let _ = writeln!(out, "[flux] ===== end of load-phase debug log{full} =====");
        }
        let _ = out.flush();
        if let Some(mut f) = self.file.lock().take() {
            let _ = writeln!(f, "{DUMPED_MARKER}");
        }
        true
    }
}

/// Per-layer filter: debug and above while the capture is active, nothing
/// after. `max_level_hint` is what lets the global max level fall back once
/// the interest cache is rebuilt.
pub struct LoadFilter(Arc<LoadCapture>);

impl<S> Filter<S> for LoadFilter {
    fn enabled(&self, meta: &Metadata<'_>, _cx: &Context<'_, S>) -> bool {
        self.0.is_active() && captures(meta)
    }

    fn callsite_enabled(&self, meta: &'static Metadata<'static>) -> Interest {
        if self.0.is_active() && captures(meta) {
            Interest::sometimes()
        } else {
            Interest::never()
        }
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Some(if self.0.is_active() {
            LevelFilter::DEBUG
        } else {
            LevelFilter::OFF
        })
    }
}

/// DEBUG and above, except DEBUG for the model and kernel crates (see
/// `debug_log::MODEL_KERNEL_TARGETS`): this filter excludes them from what
/// it captures, but `max_level_hint` below is necessarily a single,
/// non-per-target level — while this capture is active it still raises the
/// GLOBAL ambient tracing level to DEBUG for every target, including these.
/// That used to be enough, together with a per-layer `Filter`'s
/// `register_callsite`/`enabled` semantics (a per-layer filter saying "no"
/// never statically disables a callsite — it defers to the Registry asking
/// dynamically every time), to flip `tracing::enabled!(DEBUG)` inside
/// spark_model's decode dump-and-sync helpers even once this capture had
/// deactivated — a confirmed GPU A/B regression (`FLUX_LOAD_DEBUG=auto` vs
/// `off`, PLE suffix-graph capture bug). Those call sites now use an
/// explicit opt-in instead of the ambient level
/// (`spark_model::layer::decode_debug_dumps_enabled`), so this exclusion is
/// defense for *log output* only, not a safety requirement for them.
fn captures(meta: &Metadata<'_>) -> bool {
    let level = *meta.level();
    level <= LevelFilter::INFO
        || (level <= LevelFilter::DEBUG && !crate::debug_log::is_model_kernel_target(meta.target()))
}

/// `MakeWriter` into the capture. The fmt layer formats a whole event and
/// hands it over in one `write`, so one call is one line.
#[derive(Clone)]
pub struct CaptureWriter(Arc<LoadCapture>);

impl Write for CaptureWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.push(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CaptureWriter {
    type Writer = CaptureWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// The capture layer for a given capture instance.
pub fn layer_for<S>(capture: Arc<LoadCapture>) -> impl Layer<S> + Send + Sync
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    tracing_subscriber::fmt::layer()
        .fmt_fields(crate::debug_log::PlainFields::default())
        .with_ansi(false)
        .with_writer(CaptureWriter(capture.clone()))
        .with_filter(LoadFilter(capture))
}

// STATIC, DELIBERATELY — process lifecycle: one first load per process, read
// by the panic hook, `main`'s error path and the listener's ready point.
static CAPTURE: OnceLock<Arc<LoadCapture>> = OnceLock::new();

/// Create the process capture from the environment. `None` when
/// `FLUX_LOAD_DEBUG=off`; then callers keep their original subscriber.
pub fn install(rank: u32) -> Option<Arc<LoadCapture>> {
    let mode = Mode::from_env();
    if mode == Mode::Off {
        return None;
    }
    let file = std::env::var("FLUX_LOAD_DEBUG_FILE")
        .ok()
        .filter(|p| !p.is_empty() && !p.eq_ignore_ascii_case("none"))
        .map(|p| crate::debug_log::per_rank_path(&p, rank));
    let capture = CAPTURE.get_or_init(|| LoadCapture::new(mode, file)).clone();
    install_panic_hook();
    spawn_stall_watch(&capture);
    Some(capture)
}

/// Poll the capture for a stall until the load ends (`FLUX_LOAD_STALL_SECS`).
fn spawn_stall_watch(capture: &Arc<LoadCapture>) {
    let raw = std::env::var("FLUX_LOAD_STALL_SECS").ok();
    let stall = parse_stall_secs(raw.as_deref()).unwrap_or_else(|msg| {
        eprintln!("[flux] WARN: {msg}");
        Some(Duration::from_secs(DEFAULT_STALL_SECS))
    });
    let Some(stall) = stall else {
        return;
    };
    let capture = Arc::clone(capture);
    let spawned = std::thread::Builder::new()
        .name("load-stall-watch".into())
        .spawn(move || {
            while capture.is_active() {
                std::thread::sleep(Duration::from_secs(1));
                capture.report_stall(Instant::now(), stall, &mut std::io::stderr().lock());
            }
        });
    if let Err(e) = spawned {
        eprintln!("[flux] WARN: load stall reporter not started: {e}");
    }
}

/// The process capture's layer, or `None` (a no-op layer) when it is off.
pub fn layer<S>() -> Option<impl Layer<S> + Send + Sync>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    CAPTURE.get().map(|c| layer_for(c.clone()))
}

fn active_capture() -> Option<&'static Arc<LoadCapture>> {
    CAPTURE.get().filter(|c| c.is_active())
}

/// The listener is up: switch the capture off for the life of the process.
pub fn finish_load() {
    if CAPTURE.get().is_some_and(|c| c.deactivate()) {
        // Recompute every callsite's interest and the global max level now
        // that the capture's hint is OFF, then let the `log` facade's fast
        // path follow (LogTracer took its max level while capture was on).
        tracing::callsite::rebuild_interest_cache();
        log::set_max_level(as_log_filter(LevelFilter::current()));
    }
}

fn as_log_filter(level: LevelFilter) -> log::LevelFilter {
    match level {
        LevelFilter::OFF => log::LevelFilter::Off,
        LevelFilter::ERROR => log::LevelFilter::Error,
        LevelFilter::WARN => log::LevelFilter::Warn,
        LevelFilter::INFO => log::LevelFilter::Info,
        LevelFilter::DEBUG => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    }
}

/// The model is up; only binding the listener remains.
pub fn note_bind() {
    if let Some(c) = CAPTURE.get() {
        c.binding.store(true, Ordering::SeqCst);
    }
}

/// Startup returned an error before the listener came up: print the capture.
pub fn report_load_failure() {
    if let Some(c) = active_capture() {
        c.report_failure(&mut std::io::stderr().lock(), c.failure_reason());
    }
}

fn backtrace_env_enabled() -> bool {
    std::env::var_os("RUST_BACKTRACE").is_some_and(|v| v != "0")
}

/// Chain a hook that, during the load only, dumps the capture once (whether
/// panics unwind or abort) and adds a backtrace when `RUST_BACKTRACE` is unset.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let Some(capture) = active_capture() else {
            previous(info);
            return;
        };
        // `report_failure` atomically clears `active`, so a subsequent
        // `report_load_failure()` on the unwind path never double-dumps.
        capture.report_failure(&mut std::io::stderr().lock(), "panic during model load");
        previous(info);
        if !backtrace_env_enabled() {
            eprintln!(
                "load-phase panic backtrace:\n{}",
                std::backtrace::Backtrace::force_capture()
            );
        }
    }));
}

#[cfg(test)]
#[path = "load_debug_tests.rs"]
mod tests;

#[cfg(test)]
mod stall_tests {
    use super::*;

    const PHASE: &str =
        "2026-10-29T16:39:57Z DEBUG atlas::tui::progress: ev=\"phase\" phase=2 name=\"config\"\n";

    fn report(capture: &LoadCapture, at: Instant, stall: Duration) -> Option<String> {
        let mut out = Vec::new();
        capture
            .report_stall(at, stall, &mut out)
            .then(|| String::from_utf8(out).unwrap())
    }

    #[test]
    fn stall_seconds_default_off_and_bad_values() {
        let default = Some(Duration::from_secs(DEFAULT_STALL_SECS));
        assert_eq!(parse_stall_secs(None), Ok(default));
        assert_eq!(parse_stall_secs(Some(" ")), Ok(default));
        assert_eq!(parse_stall_secs(Some("0")), Ok(None));
        assert_eq!(
            parse_stall_secs(Some("45")),
            Ok(Some(Duration::from_secs(45)))
        );
        assert!(parse_stall_secs(Some("ten")).is_err());
        assert!(parse_stall_secs(Some("-1")).is_err());
    }

    #[test]
    fn phase_lines_name_the_stage() {
        assert_eq!(phase_name(PHASE), Some("config"));
        assert_eq!(phase_name("INFO Model path: /models/x\n"), None);
    }

    #[test]
    fn a_stall_is_reported_once_with_the_stage_and_lines_and_the_load_continues() {
        let capture = LoadCapture::new(Mode::Auto, None);
        capture.push(PHASE.as_bytes());
        capture.push(b"DEBUG loading shard 3 of 48\n");
        let stall = Duration::from_secs(600);
        let last = capture.progress.lock().0;
        assert_eq!(
            report(&capture, last + Duration::from_secs(599), stall),
            None
        );
        let text = report(&capture, last + stall, stall).expect("stall report");
        assert!(
            text.contains("no load progress for 600s (stage: config)"),
            "{text}"
        );
        assert!(text.contains("loading shard 3 of 48"), "{text}");
        assert!(text.contains("nothing was stopped"));
        // Once per stall, and the capture is still live for a later failure.
        assert_eq!(report(&capture, last + stall * 2, stall), None);
        assert!(capture.is_active());
        let mut failure = Vec::new();
        assert!(capture.report_failure(&mut failure, "model load failed"));
        assert!(
            String::from_utf8(failure)
                .unwrap()
                .contains("loading shard 3 of 48")
        );
    }

    #[test]
    fn progress_re_arms_the_report() {
        let capture = LoadCapture::new(Mode::Auto, None);
        let stall = Duration::from_secs(10);
        let first = capture.progress.lock().0;
        assert!(report(&capture, first + stall, stall).is_some());
        capture.push(b"DEBUG next step\n");
        let second = capture.progress.lock().0;
        assert_eq!(
            report(&capture, second + Duration::from_secs(9), stall),
            None
        );
        let text = report(&capture, second + stall, stall).expect("second stall");
        assert!(text.contains("before the first stage"), "{text}");
    }

    #[test]
    fn a_finished_or_off_load_never_reports() {
        let capture = LoadCapture::new(Mode::Auto, None);
        let last = capture.progress.lock().0;
        assert!(capture.deactivate());
        assert_eq!(
            report(
                &capture,
                last + Duration::from_secs(3600),
                Duration::from_secs(1)
            ),
            None
        );
        let off = LoadCapture::new(Mode::Off, None);
        let last = off.progress.lock().0;
        assert_eq!(
            report(
                &off,
                last + Duration::from_secs(3600),
                Duration::from_secs(1)
            ),
            None
        );
    }

    #[test]
    fn echo_mode_points_at_the_streamed_lines() {
        let capture = LoadCapture::new(Mode::Always, None);
        let last = capture.progress.lock().0;
        let text = report(
            &capture,
            last + Duration::from_secs(5),
            Duration::from_secs(5),
        )
        .unwrap();
        assert!(text.contains("streamed above"), "{text}");
        assert!(!text.contains("last 0 lines"));
    }
}
