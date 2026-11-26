// SPDX-License-Identifier: AGPL-3.0-only

//! Persistent on-disk cache for learned SLO prefill chunk costs and decode step timing.
//!
//! Stores per-fingerprint [`PersistedSloCostEntry`] snapshots as schema-v1 JSON under
//! `$HOME/.cache/flux/slo_cost_v1.json` (overridable via `ATLAS_SLO_COST_CACHE_PATH`,
//! disableable via `ATLAS_SLO_COST_CACHE=0`, with an optional read-only bundled seed at
//! `/opt/flux/cache/slo_cost_v1.json`).
//!
//! Writes use an advisory `flock` + read-merge + temp-file + `fsync` + atomic `rename`
//! sequence so concurrent server ranks never clobber sibling fingerprints and crashes
//! never truncate the file. All filesystem errors are non-fatal and degrade cleanly to
//! the static cost table.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::{Condvar, Mutex};
use serde::{Deserialize, Serialize};

use super::{BAND_LABELS, ChunkCostModel, MIN_BUCKET};

/// On-disk JSON schema version. Mismatched versions are ignored cleanly.
pub const SCHEMA_VERSION: u32 = 1;
/// Environment variable (`"0"` disables persistence).
pub const CACHE_ENABLE_ENV: &str = "ATLAS_SLO_COST_CACHE";
/// Environment variable overriding the cache JSON path.
pub const CACHE_PATH_ENV: &str = "ATLAS_SLO_COST_CACHE_PATH";
/// Optional read-only pre-seeded cache bundled with images/appliances.
pub const BUNDLED_SEED_CACHE_PATH: &str = "/opt/flux/cache/slo_cost_v1.json";
/// Flush cache after this many new chunk/decode observations.
pub const SAVE_OBSERVATION_INTERVAL: u32 = 16;
/// Or flush cache once this many seconds have elapsed since the previous flush.
pub const SAVE_TIME_INTERVAL_SECS: u64 = 30;

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Hardware, model, chunk-width, wide-prefill, and QSA configuration key.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SloCostFingerprint {
    pub gpu: String,
    pub model: String,
    pub widths: Vec<usize>,
    pub block_size: usize,
    pub prefill_wide: String,
    pub qsa_mode: String,
}

impl SloCostFingerprint {
    /// Build a fingerprint for the running scheduler instance from prepared
    /// chunk widths, KV block size, host GPU identity, and active route selectors.
    pub fn for_runtime(widths: &[usize], block_size: usize) -> Self {
        Self {
            gpu: detect_gpu_name(),
            model: detect_model_name(),
            widths: widths.to_vec(),
            block_size,
            prefill_wide: detect_prefill_wide(),
            qsa_mode: detect_qsa_mode(),
        }
    }

    /// Canonical map key inside the schema-v1 JSON file.
    pub fn encoded(&self) -> String {
        let widths_csv: String = self
            .widths
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "v{SCHEMA_VERSION}|gpu={}|model={}|widths={widths_csv}|bs={}|wide={}|qsa={}",
            sanitize_key_component(&self.gpu),
            sanitize_key_component(&self.model),
            self.block_size,
            sanitize_key_component(&self.prefill_wide),
            sanitize_key_component(&self.qsa_mode),
        )
    }
}

/// Persisted EWMA state for one `(row_bucket, offset_band)` cell.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersistedCell {
    pub ewma_ms: f64,
    pub samples: u32,
}

/// Persisted learned chunk-cost and decode-step snapshot for one [`SloCostFingerprint`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PersistedSloCostEntry {
    pub cells: BTreeMap<String, PersistedCell>,
    pub decode_step_ms: Option<f64>,
    pub global_ratio_sum: f64,
    pub global_ratio_count: u32,
    pub updated_unix_secs: u64,
}

impl PersistedSloCostEntry {
    /// True when the entry carries no calibrated cells, decode timing, or ratio data.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty() && self.decode_step_ms.is_none() && self.global_ratio_count == 0
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct DiskCacheFile {
    schema: u32,
    entries: BTreeMap<String, PersistedSloCostEntry>,
}

/// Encode a `(bucket_rows, band_index)` key for JSON map storage.
pub fn encode_cell_key(bucket_rows: usize, band_index: usize) -> String {
    format!("{bucket_rows}:{band_index}")
}

/// Decode and validate a `"<bucket>:<band>"` cell key from disk.
pub fn decode_cell_key(raw: &str) -> Option<(usize, usize)> {
    let (b_str, band_str) = raw.split_once(':')?;
    let bucket_rows = b_str.parse::<usize>().ok()?;
    let band_index = band_str.parse::<usize>().ok()?;
    if bucket_rows < MIN_BUCKET || !bucket_rows.is_power_of_two() || band_index >= BAND_LABELS.len()
    {
        return None;
    }
    Some((bucket_rows, band_index))
}

/// Current UNIX timestamp in seconds (returns `0` if system clock precedes epoch).
pub fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Resolve the user-writable cache path from process environment.
pub fn resolve_cache_path() -> Option<PathBuf> {
    resolve_cache_path_with(
        std::env::var(CACHE_ENABLE_ENV).ok().as_deref(),
        std::env::var(CACHE_PATH_ENV).ok().as_deref(),
        std::env::var_os("HOME").as_deref(),
    )
}

/// Pure, testable cache-path resolver honoring disable, explicit path, and `$HOME`.
pub fn resolve_cache_path_with(
    enable_env: Option<&str>,
    override_path: Option<&str>,
    home: Option<&OsStr>,
) -> Option<PathBuf> {
    if enable_env.map(str::trim) == Some("0") {
        return None;
    }
    if let Some(explicit) = override_path.map(str::trim)
        && !explicit.is_empty()
    {
        return Some(PathBuf::from(explicit));
    }
    let home = home?;
    if home.is_empty() {
        return None;
    }
    Some(
        PathBuf::from(home)
            .join(".cache")
            .join("flux")
            .join("slo_cost_v1.json"),
    )
}

/// Resolve the optional read-only bundled seed path when no explicit override is set.
pub fn resolve_seed_path(user_path: Option<&Path>) -> Option<PathBuf> {
    let explicit_override = std::env::var(CACHE_PATH_ENV)
        .ok()
        .is_some_and(|v| !v.trim().is_empty());
    resolve_seed_path_with(user_path.is_some(), explicit_override)
}

/// Pure, testable bundled-seed path resolver.
pub fn resolve_seed_path_with(has_user_path: bool, explicit_override: bool) -> Option<PathBuf> {
    if has_user_path && !explicit_override {
        Some(PathBuf::from(BUNDLED_SEED_CACHE_PATH))
    } else {
        None
    }
}

/// Load all schema-v1 entries from `path`; missing, mismatched, or corrupt files return empty.
pub fn load_entries_from_path(path: &Path) -> BTreeMap<String, PersistedSloCostEntry> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return BTreeMap::new(),
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "SLO cost cache read failed");
            return BTreeMap::new();
        }
    };
    match serde_json::from_slice::<DiskCacheFile>(&bytes) {
        Ok(parsed) if parsed.schema == SCHEMA_VERSION => parsed.entries,
        Ok(parsed) => {
            tracing::info!(
                path = %path.display(),
                found_schema = parsed.schema,
                expected_schema = SCHEMA_VERSION,
                "SLO cost cache schema mismatch; ignoring cached entries"
            );
            BTreeMap::new()
        }
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "SLO cost cache corrupt; ignoring");
            BTreeMap::new()
        }
    }
}

/// Load a single fingerprint entry, falling back to `seed_path` when absent from `user_path`.
pub fn load_entry_with_seed(
    user_path: Option<&Path>,
    seed_path: Option<&Path>,
    fingerprint_key: &str,
) -> Option<PersistedSloCostEntry> {
    if let Some(p) = user_path
        && let Some(entry) = load_entries_from_path(p).remove(fingerprint_key)
    {
        return Some(entry);
    }
    seed_path.and_then(|p| load_entries_from_path(p).remove(fingerprint_key))
}

#[cfg(unix)]
struct AdvisoryFileLock {
    file: fs::File,
}

#[cfg(unix)]
impl AdvisoryFileLock {
    fn acquire(lock_path: &Path) -> Option<Self> {
        use std::os::unix::io::AsRawFd;
        let file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(lock_path)
            .ok()?;
        // SAFETY: `file.as_raw_fd()` is a valid open file descriptor owned by `file`.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return None;
        }
        Some(Self { file })
    }
}

#[cfg(unix)]
impl Drop for AdvisoryFileLock {
    fn drop(&mut self) {
        use std::os::unix::io::AsRawFd;
        // SAFETY: `self.file` remains open until after `drop` returns.
        unsafe {
            let _ = libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

/// Merge and atomically persist `entry` under `fingerprint_key` into `path`.
///
/// Returns `true` on success and `false` on any non-fatal I/O failure.
pub fn save_entry_atomic(
    path: &Path,
    fingerprint_key: &str,
    entry: &PersistedSloCostEntry,
) -> bool {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
        && let Err(e) = fs::create_dir_all(parent)
    {
        tracing::debug!(path = %parent.display(), error = %e, "SLO cost cache dir unavailable");
        return false;
    }

    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("slo_cost_v1.json");
    #[cfg(unix)]
    let _file_lock = AdvisoryFileLock::acquire(&path.with_file_name(format!(".{file_name}.lock")));

    // Merge under advisory flock so concurrent processes preserve sibling fingerprints.
    let mut merged = load_entries_from_path(path);
    merged.insert(fingerprint_key.to_string(), entry.clone());

    let payload = DiskCacheFile {
        schema: SCHEMA_VERSION,
        entries: merged,
    };
    let Ok(serialized) = serde_json::to_vec_pretty(&payload) else {
        return false;
    };

    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp_path = path.with_file_name(format!(".{file_name}.tmp.{}.{seq}", std::process::id()));
    let write_res = fs::File::create(&tmp_path).and_then(|mut f| {
        f.write_all(&serialized)?;
        f.flush()?;
        f.sync_all()
    });
    if let Err(e) = write_res {
        let _ = fs::remove_file(&tmp_path);
        tracing::debug!(path = %tmp_path.display(), error = %e, "SLO cost cache tmp write failed");
        return false;
    }
    if let Err(e) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        tracing::debug!(path = %path.display(), error = %e, "SLO cost cache rename failed");
        return false;
    }
    true
}

struct PendingWrite {
    path: PathBuf,
    fingerprint_key: String,
    entry: PersistedSloCostEntry,
}

struct WriterState {
    pending: Option<(u64, PendingWrite)>,
    submitted_seq: u64,
    completed_seq: u64,
    closed: bool,
    exited: bool,
}

struct WriterShared {
    state: Mutex<WriterState>,
    work_cvar: Condvar,
    done_cvar: Condvar,
}

struct WriterExitGuard {
    shared: Arc<WriterShared>,
}

impl Drop for WriterExitGuard {
    fn drop(&mut self) {
        let mut state = self.shared.state.lock();
        state.exited = true;
        drop(state);
        self.shared.done_cvar.notify_all();
    }
}

/// Lightweight single-thread coalescing background writer for atomic SLO cost
/// cache snapshots. Keeps at most one unstarted [`PendingWrite`] so rapid
/// flushes coalesce with zero unbounded queue growth and zero scheduler-thread
/// `flock`/`fsync` stalls.
struct BackgroundSloWriter {
    shared: Arc<WriterShared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl BackgroundSloWriter {
    fn spawn() -> std::io::Result<Self> {
        Self::spawn_with("slo-cost-writer", |write| {
            let _ = save_entry_atomic(&write.path, &write.fingerprint_key, &write.entry);
        })
    }

    fn spawn_with<F>(name: &str, mut write_fn: F) -> std::io::Result<Self>
    where
        F: FnMut(PendingWrite) + Send + 'static,
    {
        let shared = Arc::new(WriterShared {
            state: Mutex::new(WriterState {
                pending: None,
                submitted_seq: 0,
                completed_seq: 0,
                closed: false,
                exited: false,
            }),
            work_cvar: Condvar::new(),
            done_cvar: Condvar::new(),
        });
        let worker_shared = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let _exit_guard = WriterExitGuard {
                    shared: Arc::clone(&worker_shared),
                };
                loop {
                    let next = {
                        let mut state = worker_shared.state.lock();
                        loop {
                            if let Some(item) = state.pending.take() {
                                break Some(item);
                            }
                            if state.closed {
                                break None;
                            }
                            worker_shared.work_cvar.wait(&mut state);
                        }
                    };
                    let Some((seq, write)) = next else {
                        break;
                    };
                    write_fn(write);
                    let mut state = worker_shared.state.lock();
                    if seq > state.completed_seq {
                        state.completed_seq = seq;
                    }
                    drop(state);
                    worker_shared.done_cvar.notify_all();
                }
            })?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }

    fn is_exited(&self) -> bool {
        self.shared.state.lock().exited
    }

    fn enqueue(&self, write: PendingWrite) -> Result<u64, PendingWrite> {
        let mut state = self.shared.state.lock();
        if state.closed || state.exited {
            return Err(write);
        }
        state.submitted_seq = state.submitted_seq.saturating_add(1);
        let seq = state.submitted_seq;
        state.pending = Some((seq, write));
        drop(state);
        self.shared.work_cvar.notify_one();
        Ok(seq)
    }

    fn wait_until(&self, target_seq: u64) {
        let mut state = self.shared.state.lock();
        while state.completed_seq < target_seq && !state.exited {
            self.shared.done_cvar.wait(&mut state);
        }
    }

    fn wait_idle(&self) {
        let mut state = self.shared.state.lock();
        let target_seq = state.submitted_seq;
        while state.completed_seq < target_seq && !state.exited {
            self.shared.done_cvar.wait(&mut state);
        }
    }
}

impl std::fmt::Debug for BackgroundSloWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundSloWriter")
            .field("active", &self.thread.is_some())
            .finish()
    }
}

impl Drop for BackgroundSloWriter {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock();
            state.closed = true;
        }
        self.shared.work_cvar.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Debounced persistence controller owned by [`super::super::prefill_slo::SloRuntime`].
#[derive(Debug)]
pub struct SloCacheHandle {
    path: PathBuf,
    fingerprint_key: String,
    pending_observations: u32,
    dirty: bool,
    last_saved: Instant,
    writer: Option<BackgroundSloWriter>,
}

impl Clone for SloCacheHandle {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            fingerprint_key: self.fingerprint_key.clone(),
            pending_observations: 0,
            dirty: false,
            last_saved: Instant::now(),
            writer: None,
        }
    }
}

impl SloCacheHandle {
    pub fn new(path: PathBuf, fingerprint_key: String) -> Self {
        Self {
            path,
            fingerprint_key,
            pending_observations: 0,
            dirty: false,
            last_saved: Instant::now(),
            writer: None,
        }
    }

    fn ensure_writer(&mut self) -> Option<&BackgroundSloWriter> {
        if self
            .writer
            .as_ref()
            .is_some_and(BackgroundSloWriter::is_exited)
        {
            self.writer = None;
        }
        if self.writer.is_none() {
            match BackgroundSloWriter::spawn() {
                Ok(writer) => self.writer = Some(writer),
                Err(error) => {
                    tracing::debug!(
                        path = %self.path.display(),
                        error = %error,
                        "SLO cost cache background writer unavailable"
                    );
                    return None;
                }
            }
        }
        self.writer.as_ref()
    }

    /// Record one new observation and return `true` when the debounce threshold
    /// (`SAVE_OBSERVATION_INTERVAL` observations or `SAVE_TIME_INTERVAL_SECS` elapsed)
    /// warrants a disk flush.
    pub fn record_observation(&mut self, now: Instant) -> bool {
        self.dirty = true;
        self.pending_observations = self.pending_observations.saturating_add(1);
        self.pending_observations >= SAVE_OBSERVATION_INTERVAL
            || now.saturating_duration_since(self.last_saved)
                >= Duration::from_secs(SAVE_TIME_INTERVAL_SECS)
    }

    /// Enqueue dirty cost-model state onto the coalescing background writer
    /// thread without blocking the scheduler thread (best-effort, non-fatal).
    ///
    /// Rapid updates coalesce into the latest snapshot (bounded to 1 pending
    /// entry). Dropping [`SloCacheHandle`] drains the latest pending snapshot
    /// and joins the writer thread cleanly; use [`Self::flush_sync`] when an
    /// immediate synchronous persistence barrier is needed.
    pub fn flush(&mut self, cost: &ChunkCostModel, decode_step_ms: Option<f64>) {
        if !self.dirty {
            return;
        }
        let entry = cost.to_persisted(decode_step_ms);
        self.dirty = false;
        self.pending_observations = 0;
        self.last_saved = Instant::now();
        if entry.is_empty() {
            return;
        }
        let pending = PendingWrite {
            path: self.path.clone(),
            fingerprint_key: self.fingerprint_key.clone(),
            entry,
        };
        if let Some(writer) = self.ensure_writer() {
            let _ = writer.enqueue(pending);
        }
    }

    /// Flush dirty cost-model state and block until all pending background
    /// writes have completed on disk (best-effort, non-fatal).
    pub fn flush_sync(&mut self, cost: &ChunkCostModel, decode_step_ms: Option<f64>) {
        if self.dirty {
            let entry = cost.to_persisted(decode_step_ms);
            self.dirty = false;
            self.pending_observations = 0;
            self.last_saved = Instant::now();
            if !entry.is_empty() {
                let pending = PendingWrite {
                    path: self.path.clone(),
                    fingerprint_key: self.fingerprint_key.clone(),
                    entry,
                };
                if let Some(writer) = self.ensure_writer() {
                    match writer.enqueue(pending) {
                        Ok(target_seq) => {
                            writer.wait_until(target_seq);
                            return;
                        }
                        Err(returned) => {
                            let _ = save_entry_atomic(
                                &returned.path,
                                &returned.fingerprint_key,
                                &returned.entry,
                            );
                            return;
                        }
                    }
                }
                let _ = save_entry_atomic(&self.path, &self.fingerprint_key, &pending.entry);
                return;
            }
        }
        if let Some(writer) = self.writer.as_ref() {
            writer.wait_idle();
        }
    }
}

fn sanitize_key_component(raw: &str) -> String {
    raw.trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn basename_or_value(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let leaf = Path::new(trimmed)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(trimmed);
    let clean = sanitize_key_component(leaf);
    (!clean.is_empty()).then_some(clean)
}

#[cfg(target_os = "linux")]
fn read_nvidia_proc_model() -> Option<String> {
    let mut dirs: Vec<PathBuf> = fs::read_dir("/proc/driver/nvidia/gpus")
        .ok()?
        .filter_map(|e| e.ok().map(|entry| entry.path()))
        .collect();
    dirs.sort();
    for gpu_dir in dirs {
        let Ok(info) = fs::read_to_string(gpu_dir.join("information")) else {
            continue;
        };
        if let Some(clean) = info
            .lines()
            .filter_map(|l| l.strip_prefix("Model:").map(sanitize_key_component))
            .find(|s| !s.is_empty())
        {
            return Some(clean);
        }
    }
    None
}

fn detect_gpu_name() -> String {
    if let Some(clean) = std::env::var("ATLAS_SLO_COST_GPU")
        .ok()
        .map(|v| sanitize_key_component(&v))
        .filter(|v| !v.is_empty())
    {
        return clean;
    }
    #[cfg(target_os = "linux")]
    if let Some(model) = read_nvidia_proc_model() {
        return model;
    }
    #[cfg(feature = "cuda")]
    let backend = "cuda";
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    let backend = "metal";
    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
    let backend = "cpu";
    format!("{backend}_{}", std::env::consts::ARCH)
}

fn detect_model_name() -> String {
    if let Some(name) = std::env::var("FLUX_MODEL_ID")
        .or_else(|_| std::env::var("ATLAS_MODEL_PATH"))
        .ok()
        .and_then(|v| basename_or_value(&v))
    {
        return name;
    }
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        let val = if arg == "--model" || arg == "-m" {
            args.next()
        } else {
            arg.strip_prefix("--model=").map(str::to_owned)
        };
        if let Some(name) = val.as_deref().and_then(basename_or_value) {
            return name;
        }
    }
    "default".to_string()
}

fn detect_prefill_wide() -> String {
    std::env::var("ATLAS_QWEN4_PREFILL_WIDE")
        .ok()
        .map(|v| sanitize_key_component(&v))
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "default".to_string())
}

fn detect_qsa_mode() -> String {
    const VARS: [(&str, &str); 7] = [
        ("moe", "ATLAS_QWEN4_PREFILL_MOE_BATCH"),
        ("mc", "ATLAS_QWEN4_PREFILL_MULTI_CHUNK"),
        ("attn2", "ATLAS_QWEN4_QSA_ATTN_V2"),
        ("score2", "ATLAS_QWEN4_QSA_SCORE_V2"),
        ("selpar", "ATLAS_QWEN4_QSA_SELECT_PAR"),
        ("sel3", "ATLAS_QWEN4_QSA_SELECT_V3"),
        ("pgemm", "ATLAS_QWEN4_QSA_PREFILL_GEMM"),
    ];
    let parts: Vec<String> = VARS
        .iter()
        .filter_map(|&(tag, name)| {
            let val = std::env::var(name).ok()?;
            let clean = sanitize_key_component(&val);
            (!clean.is_empty()).then(|| format!("{tag}={clean}"))
        })
        .collect();
    if parts.is_empty() {
        "default".to_string()
    } else {
        parts.join(",")
    }
}

#[cfg(test)]
mod async_writer_tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{Arc, Barrier};
    use std::time::Instant;

    use parking_lot::Mutex;
    use tempfile::TempDir;

    use super::{
        BackgroundSloWriter, ChunkCostModel, PendingWrite, PersistedSloCostEntry, SloCacheHandle,
        load_entry_with_seed,
    };

    #[test]
    fn background_writer_coalesces_rapid_updates_to_latest_snapshot_and_joins_on_drop() {
        let entered_first = Arc::new(Barrier::new(2));
        let release_first = Arc::new(Barrier::new(2));
        let written = Arc::new(Mutex::new(Vec::<Option<f64>>::new()));

        let worker_entered = Arc::clone(&entered_first);
        let worker_release = Arc::clone(&release_first);
        let worker_written = Arc::clone(&written);
        let writer = BackgroundSloWriter::spawn_with("slo-writer-test", move |write| {
            let step = write.entry.decode_step_ms;
            let mut log = worker_written.lock();
            let first = log.is_empty();
            log.push(step);
            drop(log);
            if first {
                worker_entered.wait();
                worker_release.wait();
            }
        })
        .expect("spawn test writer");

        let make_write = |ms: f64| PendingWrite {
            path: PathBuf::from("unused.json"),
            fingerprint_key: "k".to_string(),
            entry: PersistedSloCostEntry {
                decode_step_ms: Some(ms),
                ..PersistedSloCostEntry::default()
            },
        };

        assert_eq!(writer.enqueue(make_write(1.0)).ok(), Some(1));
        // Wait until write #1 is actively blocked inside `write_fn`.
        entered_first.wait();

        // Rapidly enqueue writes #2..#5 without blocking: #2..#4 must coalesce into #5.
        for ms in [2.0, 3.0, 4.0, 5.0] {
            assert!(writer.enqueue(make_write(ms)).is_ok());
        }

        // Release write #1 and drop the writer: Drop must drain #5 and join cleanly.
        release_first.wait();
        drop(writer);

        assert_eq!(*written.lock(), vec![Some(1.0), Some(5.0)]);
    }

    #[test]
    fn handle_flush_is_async_and_flush_sync_or_drop_persists_deterministically() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("slo_cost_v1.json");
        let key = "v1|gpu=GB10|model=test|widths=512|bs=16|wide=default|qsa=default".to_string();

        let mut model = ChunkCostModel::default();
        assert_eq!(model.observe(512, 0, 500.0, 350.0), None); // warmup dropped
        model.observe(512, 0, 180.0, 350.0).expect("sample 1");
        model.observe(512, 0, 200.0, 180.0).expect("sample 2");

        let mut handle = SloCacheHandle::new(path.clone(), key.clone());
        let now = Instant::now();
        let _ = handle.record_observation(now);
        handle.flush(&model, Some(33.5));
        // Synchronous barrier guarantees the async write has hit disk.
        handle.flush_sync(&model, Some(33.5));

        let loaded = load_entry_with_seed(Some(&path), None, &key).expect("flushed entry");
        assert_eq!(loaded.decode_step_ms, Some(33.5));
        assert_eq!(loaded.cells["512:0"].samples, 2);

        // Clone starts clean without sharing or spawning a background thread.
        let cloned = handle.clone();
        assert!(cloned.writer.is_none());

        // Async flush immediately followed by Drop must deterministically persist latest state.
        let _ = handle.record_observation(now);
        handle.flush(&model, Some(28.25));
        drop(handle);

        let after_drop = load_entry_with_seed(Some(&path), None, &key).expect("entry after drop");
        assert_eq!(after_drop.decode_step_ms, Some(28.25));
    }

    #[test]
    fn handle_unwritable_path_degrades_nonfatally_without_deadlock() {
        let dir = TempDir::new().expect("temp dir");
        let blocker = dir.path().join("not_a_dir");
        fs::write(&blocker, b"x").expect("create blocker");
        let unwritable = blocker.join("slo_cost_v1.json");

        let mut model = ChunkCostModel::default();
        let _ = model.observe(512, 0, 500.0, 350.0);
        let _ = model.observe(512, 0, 180.0, 350.0);

        let mut handle = SloCacheHandle::new(unwritable, "k".to_string());
        let _ = handle.record_observation(Instant::now());
        handle.flush_sync(&model, Some(40.0));
        drop(handle);
    }
}

#[cfg(test)]
#[path = "prefill_slo_cost_cache_tests.rs"]
mod tests;
