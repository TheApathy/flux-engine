// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in full debug log: `FLUX_DEBUG=1` writes debug-level server logs, for
//! the load and for serving, to a size-capped file. The console is untouched:
//! this is one more layer beside the console layer, with its own filter, so
//! every line the console prints today still prints, unchanged and undelayed.
//!
//! - `FLUX_DEBUG`: `1|true|on|yes` enables; unset, empty, `0|false|off|no` disables.
//! - `FLUX_DEBUG_FILE`: the file (default `/results/flux-debug.log`).
//! - `FLUX_DEBUG_MAX_MB`: size cap per file (default 256). At the cap the file
//!   is renamed to `<file>.1` (replacing any older one) and a new file starts,
//!   so the log never takes more than twice the cap.
//! - `FLUX_DEBUG_FILTER`: an `EnvFilter` spec for the file (default: debug,
//!   with the model and kernel crates at info — see [`MODEL_KERNEL_TARGETS`]).
//!   A spec that turns DEBUG on for them gets a warning.
//!
//! Files are created owner-only (0600), and EP ranks above 0 write
//! `<file>.rank<N>`.
//!
//! Lines are written unbuffered, one `write` per event, so a crash loses at
//! most the event in flight.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;
use tracing::Subscriber;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::registry::LookupSpan;

pub const DEFAULT_FILE: &str = "/results/flux-debug.log";
const DEFAULT_MAX_MB: u64 = 256;

/// Targets that stay at INFO in every capture this crate adds: the model and
/// kernel crates, and the explicit targets they log under. Their DEBUG paths
/// are not just log lines — spark_model's decode dump-and-sync helpers
/// (`layers/moe/forward.rs`, `layers/qwen3_ssm/{ssm_forward,trait_decode}.rs`)
/// synchronise the GPU and copy device data to the host, which changes
/// timing and can disturb CUDA-graph capture. Prefix match, as `EnvFilter`.
///
/// Those call sites no longer key off the ambient tracing level at all —
/// see `spark_model::layer::decode_debug_dumps_enabled` — precisely because
/// this crate installing a per-layer log filter (e.g. `FLUX_LOAD_DEBUG`,
/// `load_debug::LoadFilter`) could flip `tracing::enabled!(DEBUG)` for those
/// targets independently of this list's intent: a `Filtered` layer's
/// `register_callsite` always answers `Interest::always()` to the registry
/// regardless of what its wrapped filter decided, and `max_level_hint` has
/// no way to express "DEBUG, except these targets" in the first place. This
/// list still documents and enforces the *log output* exclusion (what
/// prints), which is a separate, still-valid concern from the dump-and-sync
/// gate (what runs).
pub const MODEL_KERNEL_TARGETS: &[&str] = &[
    "spark_model",
    "spark_runtime",
    "spark_comm",
    "atlas",
    "cufile_sys",
    "xgrammar",
    "glm53_vision_timing",
    "sm_121",
];
/// Exceptions inside [`MODEL_KERNEL_TARGETS`]: pure progress events.
pub const MODEL_KERNEL_ALLOWED: &[&str] = &["atlas::tui::progress"];

/// Whether DEBUG for `target` must stay off in the added captures.
pub fn is_model_kernel_target(target: &str) -> bool {
    MODEL_KERNEL_TARGETS.iter().any(|t| target.starts_with(t))
        && !MODEL_KERNEL_ALLOWED.iter().any(|t| target.starts_with(t))
}

/// The default `FLUX_DEBUG_FILTER`: debug, with the model/kernel targets at info.
pub fn default_filter_spec() -> String {
    let mut spec = String::from("debug");
    for t in MODEL_KERNEL_TARGETS {
        spec.push_str(&format!(",{t}=info"));
    }
    for t in MODEL_KERNEL_ALLOWED {
        spec.push_str(&format!(",{t}=debug"));
    }
    spec
}

/// One target inside each crate whose DEBUG paths do GPU work.
const MODEL_DEBUG_PROBES: &[&str] = &[
    "spark_model::layers::moe::forward",
    "spark_model::layers::qwen3_ssm::ssm_forward",
    "spark_runtime::cuda_backend",
    "atlas_kernels::launch",
];

/// Whether an `EnvFilter` spec turns DEBUG on for a model/kernel target.
pub fn filter_enables_model_debug(spec: &str) -> bool {
    let Ok(filter) = EnvFilter::try_new(spec) else {
        return false;
    };
    // Static directives decide by target and level alone, which is what
    // `callsite_enabled` answers; `sometimes` (span-field directives) counts
    // as enabled.
    MODEL_DEBUG_PROBES.iter().any(|target| {
        !tracing_subscriber::layer::Filter::<tracing_subscriber::Registry>::callsite_enabled(
            &filter,
            probe_metadata(target),
        )
        .is_never()
    })
}

/// Whether `tracing::enabled!(target: .., DEBUG)` could be true under the
/// current dispatcher.
///
/// With per-layer filters, `Subscriber::enabled` answers true whenever any
/// filter bit is clear, so the real gate is the callsite interest: a callsite
/// every layer calls `never` is skipped by `enabled!` before the dispatcher is
/// asked. This asks the dispatcher for that interest directly (the macro's
/// own cached interest is shared by every dispatcher in the process, so it
/// would answer for other test threads too).
pub fn debug_enabled_for(target: &str) -> bool {
    let meta = probe_metadata(target);
    tracing::dispatcher::get_default(|d| !d.register_callsite(meta).is_never())
}

fn probe_metadata(target: &str) -> &'static tracing::Metadata<'static> {
    struct ProbeCallsite;
    static PROBE_CALLSITE: ProbeCallsite = ProbeCallsite;
    impl tracing::callsite::Callsite for ProbeCallsite {
        fn set_interest(&self, _: tracing::subscriber::Interest) {}
        fn metadata(&self) -> &tracing::Metadata<'_> {
            unreachable!("probe metadata is never registered")
        }
    }
    // Leaked once per probe target at startup: a handful of bytes.
    let target: &'static str = Box::leak(target.to_string().into_boxed_str());
    Box::leak(Box::new(tracing::Metadata::new(
        "load-debug probe",
        target,
        tracing::Level::DEBUG,
        None,
        None,
        None,
        tracing::field::FieldSet::new(&[], tracing::callsite::Identifier(&PROBE_CALLSITE)),
        tracing::metadata::Kind::EVENT,
    )))
}

/// Create (or truncate, or append to) a private log file, owned like its
/// directory.
///
/// The image runs as root, so a root-owned 0600 file under a host `/results`
/// mount would be unreadable to the host user. The file therefore takes the
/// directory's uid/gid and stays 0600; if that chown fails it falls back to
/// 0640 with one warning.
pub fn open_private(path: &Path, append: bool) -> std::io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.create(true);
    if append {
        opts.append(true);
    } else {
        opts.write(true).truncate(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let file = opts.open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let dir = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let owner = std::fs::metadata(dir).ok().map(|m| (m.uid(), m.gid()));
        adopt_owner(&file, path, owner);
    }
    Ok(file)
}

#[cfg(unix)]
static OWNERSHIP_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Give `file` the `owner` (uid, gid) and mode 0600, or 0640 and one warning
/// if the chown is refused. Also tightens a file that already existed.
#[cfg(unix)]
pub fn adopt_owner(file: &File, path: &Path, owner: Option<(u32, u32)>) {
    use std::os::unix::fs::PermissionsExt;
    let chowned = match owner {
        Some((uid, gid)) => std::os::unix::fs::fchown(file, Some(uid), Some(gid)),
        None => Err(std::io::Error::other("directory owner unknown")),
    };
    let mode = match &chowned {
        Ok(()) => 0o600,
        Err(e) => {
            if !OWNERSHIP_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[flux] WARN: cannot give {} the owner of its directory ({e}); it is mode 0640",
                    path.display()
                );
            }
            0o640
        }
    };
    let _ = file.set_permissions(std::fs::Permissions::from_mode(mode));
}

/// `<path>.rank<N>` for ranks above 0, so EP workers sharing one env do not
/// clobber the head's file.
pub fn per_rank_path(path: &str, rank: u32) -> String {
    if rank == 0 {
        path.to_string()
    } else {
        format!("{path}.rank{rank}")
    }
}

/// Parse `FLUX_DEBUG`. `Err` carries a warning; the caller treats it as off.
pub fn parse_enabled(raw: Option<&str>) -> Result<bool, String> {
    match raw.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("0") | Some("false") | Some("off") | Some("no") => Ok(false),
        Some("1") | Some("true") | Some("on") | Some("yes") => Ok(true),
        Some(other) => Err(format!(
            "FLUX_DEBUG={other:?} is not one of 1|0 (true/false, on/off); debug log stays off"
        )),
    }
}

/// Parse `FLUX_DEBUG_MAX_MB`; unset or invalid gives the default.
pub fn parse_max_bytes(raw: Option<&str>) -> Result<u64, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(DEFAULT_MAX_MB << 20),
        Some(v) => match v.parse::<u64>() {
            Ok(mb) if mb > 0 => mb.checked_mul(1 << 20).ok_or_else(|| {
                format!("FLUX_DEBUG_MAX_MB={v:?} overflows byte limit; using {DEFAULT_MAX_MB}")
            }),
            _ => Err(format!(
                "FLUX_DEBUG_MAX_MB={v:?} is not a positive integer; using {DEFAULT_MAX_MB}"
            )),
        },
    }
}

/// A file that rotates to `<path>.1` when the next write would pass `cap`.
///
/// Fails closed: if a rotation cannot rename or recreate the file, the log is
/// switched off with one warning rather than allowed to grow past the cap.
pub struct RotatingFile {
    path: PathBuf,
    file: Option<File>,
    written: u64,
    cap: u64,
}

impl RotatingFile {
    /// Open for append (owner-only); existing bytes count toward the cap.
    pub fn open(path: &Path, cap: u64) -> std::io::Result<Self> {
        let file = open_private(path, true)?;
        let written = file.metadata().map(|m| m.len()).unwrap_or(0);
        Ok(Self {
            path: path.to_path_buf(),
            file: Some(file),
            written,
            cap,
        })
    }

    /// False once a failed rotation has switched the log off.
    pub fn is_open(&self) -> bool {
        self.file.is_some()
    }

    fn rotate(&mut self) -> std::io::Result<()> {
        std::fs::rename(&self.path, self.rotated_path())?;
        self.file = Some(open_private(&self.path, false)?);
        self.written = 0;
        Ok(())
    }

    fn rotated_path(&self) -> PathBuf {
        let mut name = self.path.clone().into_os_string();
        name.push(".1");
        PathBuf::from(name)
    }

    pub fn write_line(&mut self, buf: &[u8]) {
        if self.file.is_none() {
            return;
        }
        if self.written > 0
            && self.written + buf.len() as u64 > self.cap
            && let Err(e) = self.rotate()
        {
            self.file = None;
            eprintln!(
                "[flux] WARN: FLUX_DEBUG: cannot rotate {} ({e}); debug log switched off",
                self.path.display()
            );
            return;
        }
        if let Some(f) = self.file.as_mut()
            && f.write_all(buf).is_ok()
        {
            self.written += buf.len() as u64;
        }
    }
}

/// `MakeWriter` into a shared [`RotatingFile`].
#[derive(Clone)]
pub struct DebugFileWriter(Arc<Mutex<RotatingFile>>);

impl Write for DebugFileWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().write_line(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for DebugFileWriter {
    type Writer = DebugFileWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// `DefaultFields` under its own type. The fmt layer caches a span's
/// formatted fields in the span's extensions keyed by the formatter TYPE, so a
/// second layer with plain `DefaultFields` would reuse the console layer's
/// ANSI-coloured copy. A distinct type gives file layers their own plain copy
/// and leaves the console's untouched.
#[derive(Default)]
pub struct PlainFields(tracing_subscriber::fmt::format::DefaultFields);

impl<'w> tracing_subscriber::fmt::FormatFields<'w> for PlainFields {
    fn format_fields<R: tracing_subscriber::field::RecordFields>(
        &self,
        writer: tracing_subscriber::fmt::format::Writer<'w>,
        fields: R,
    ) -> std::fmt::Result {
        self.0.format_fields(writer, fields)
    }
}

/// The debug-file layer for a given file and filter.
pub fn layer_for<S>(
    file: Arc<Mutex<RotatingFile>>,
    filter: EnvFilter,
) -> impl Layer<S> + Send + Sync
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    tracing_subscriber::fmt::layer()
        .fmt_fields(PlainFields::default())
        .with_ansi(false)
        .with_writer(DebugFileWriter(file))
        .with_filter(filter)
}

/// The plain-mode console layer: what `fmt().with_env_filter(..).init()`
/// installs, as a layer with a per-layer filter so the load capture and the
/// debug file can sit beside it. `log_internal_errors(true)` is the builder's
/// setting. Output is byte-identical to the builder's (see the tests).
pub fn console_layer<S, W>(writer: W, filter: EnvFilter) -> impl Layer<S> + Send + Sync
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    W: for<'w> tracing_subscriber::fmt::MakeWriter<'w> + Send + Sync + 'static,
{
    tracing_subscriber::fmt::layer()
        .log_internal_errors(true)
        .with_writer(writer)
        .with_filter(filter)
}

struct DebugLog {
    file: Arc<Mutex<RotatingFile>>,
    filter_spec: String,
}

// STATIC, DELIBERATELY — process lifecycle: the file outlives every model.
static DEBUG_LOG: OnceLock<DebugLog> = OnceLock::new();

/// Open the debug file from the environment. `false` when `FLUX_DEBUG` is
/// off or the file cannot be opened (with a warning on stderr). `rank` > 0
/// writes to `<file>.rank<N>`.
pub fn install(rank: u32) -> bool {
    let enabled =
        parse_enabled(std::env::var("FLUX_DEBUG").ok().as_deref()).unwrap_or_else(|msg| {
            eprintln!("[flux] WARN: {msg}");
            false
        });
    if !enabled {
        return false;
    }
    let cap =
        parse_max_bytes(std::env::var("FLUX_DEBUG_MAX_MB").ok().as_deref()).unwrap_or_else(|msg| {
            eprintln!("[flux] WARN: {msg}");
            DEFAULT_MAX_MB << 20
        });
    let path = std::env::var("FLUX_DEBUG_FILE")
        .ok()
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_FILE.to_string());
    let path = per_rank_path(&path, rank);
    let filter_spec = std::env::var("FLUX_DEBUG_FILTER")
        .ok()
        .filter(|f| !f.trim().is_empty())
        .unwrap_or_else(default_filter_spec);
    if let Err(e) = EnvFilter::try_new(&filter_spec) {
        eprintln!(
            "[flux] WARN: FLUX_DEBUG_FILTER={filter_spec:?} is invalid ({e}); debug log stays off"
        );
        return false;
    }
    if filter_enables_model_debug(&filter_spec) {
        eprintln!(
            "[flux] WARN: FLUX_DEBUG_FILTER={filter_spec:?} enables DEBUG for model/kernel crates \
             (spark_model, spark_runtime, atlas_*). Their debug paths synchronise the GPU and copy \
             device data, which changes serving speed and can affect CUDA-graph capture."
        );
    }
    match RotatingFile::open(Path::new(&path), cap) {
        Ok(file) => {
            let _ = DEBUG_LOG.set(DebugLog {
                file: Arc::new(Mutex::new(file)),
                filter_spec,
            });
            true
        }
        Err(e) => {
            eprintln!("[flux] WARN: FLUX_DEBUG: cannot open {path}: {e}; debug log stays off");
            false
        }
    }
}

/// The process debug-file layer, or `None` (a no-op layer) when it is off.
pub fn layer<S>() -> Option<impl Layer<S> + Send + Sync>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    DEBUG_LOG.get().map(|d| {
        // Validated in `install`.
        let filter = EnvFilter::try_new(&d.filter_spec)
            .unwrap_or_else(|_| EnvFilter::new(default_filter_spec()));
        layer_for(d.file.clone(), filter)
    })
}

#[cfg(test)]
#[path = "debug_log_tests.rs"]
mod tests;
