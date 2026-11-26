// SPDX-License-Identifier: AGPL-3.0-only

//! The Benchmarks pane: box readiness and harness timing summaries.
//!
//! This pane intentionally reads the artifacts produced by the repository's
//! standalone `bench/` harnesses rather than running a separate in-TUI
//! benchmark runner, ensuring the dashboard always reflects the exact
//! prewarm-controlled measurements recorded on disk:
//!
//! 1. **Whether the box is fit to measure on.** A run taken while another job
//!    holds the GPU lock, or during a build, invalidates a measurement.
//! 2. **What the harnesses recorded**, read from their own `timing.txt`
//!    files, so the pane cannot disagree with the files under review.

use std::path::{Path, PathBuf};

/// Is the box fit to measure on right now?
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoxState {
    /// PID holding the GPU lock, if any. `None` means free.
    pub lock_holder: Option<u32>,
    /// PIDs queued behind it, in the kernel's grant order.
    pub waiters: Vec<u32>,
    /// Host memory available, MiB. A load can need tens of GB.
    pub mem_available_mib: u64,
    /// `cargo`/`rustc` processes running. A build during a measurement window
    /// slows GPU work on this SoC by roughly 1.15x.
    pub builders: usize,
}

impl BoxState {
    /// One line naming every reason NOT to measure, or that it is clear.
    ///
    /// Reports EVERY blocker rather than the first: an operator who clears one
    /// and hits the next learned nothing the first message could have told
    /// them, which is the same rule `validate_serve_args` follows.
    pub fn verdict(&self) -> String {
        let mut blockers = Vec::new();
        if let Some(pid) = self.lock_holder {
            blockers.push(format!("GPU lock held by pid {pid}"));
        }
        if !self.waiters.is_empty() {
            blockers.push(format!("{} job(s) queued behind it", self.waiters.len()));
        }
        if self.builders > 0 {
            blockers.push(format!("{} build process(es) running", self.builders));
        }
        if self.mem_available_mib < 40 * 1024 {
            blockers.push(format!(
                "only {:.1} GB host memory free",
                self.mem_available_mib as f64 / 1024.0
            ));
        }
        if blockers.is_empty() {
            "box is clear — a measurement taken now is not contended".into()
        } else {
            format!("DO NOT MEASURE: {}", blockers.join(" · "))
        }
    }
}

/// Read the lock table for `lock_path`'s inode.
///
/// `/proc/locks`, not `fuser`: `fuser` returns an undifferentiated pid list
/// and cannot say which one HOLDS and which are WAITING, so it cannot answer
/// "who runs next". `/proc/locks` marks waiters with a leading `->`.
///
/// A holder pid that is dead is NOT a stale lock — the lock survives through a
/// descriptor inherited by a live child, which is the normal shape for a
/// harness that re-execs. Reported as held, because it is.
pub fn read_lock(lock_path: &Path, proc_locks: &str) -> (Option<u32>, Vec<u32>) {
    let Ok(meta) = std::fs::metadata(lock_path) else {
        return (None, Vec::new());
    };
    let ino = {
        use std::os::unix::fs::MetadataExt as _;
        meta.ino()
    };
    let (mut holder, mut waiters) = (None, Vec::new());
    for line in proc_locks.lines() {
        // `116: FLOCK  ADVISORY  WRITE 346214 103:02:16529524 0 EOF`
        // `116: ->    FLOCK  ADVISORY  WRITE 409729 ...`
        // `117: POSIX  ADVISORY  WRITE 346214 103:02:16529524 0 4096`
        let Some(rest) = line.split_once(':').map(|(_, r)| r) else {
            continue;
        };
        let is_waiter = rest.trim_start().starts_with("->");
        let fields: Vec<&str> = rest.split_whitespace().collect();
        let Some(inode_idx) = fields
            .iter()
            .position(|f| f.contains(':') && !f.contains('.'))
        else {
            continue;
        };
        let Some(found) = fields[inode_idx]
            .rsplit(':')
            .next()
            .and_then(|i| i.parse::<u64>().ok())
        else {
            continue;
        };
        if found != ino {
            continue;
        }
        // In `/proc/locks`, the owner PID is always the field immediately
        // before `dev:dev:ino` — never after it (byte-range locks put numeric
        // `<start> <end>` offsets after the inode).
        let pid = inode_idx
            .checked_sub(1)
            .and_then(|i| fields.get(i))
            .and_then(|f| f.parse::<u32>().ok())
            .filter(|p| *p > 0);
        let Some(pid) = pid else { continue };
        if is_waiter {
            waiters.push(pid);
        } else {
            holder = Some(pid);
        }
    }
    (holder, waiters)
}

/// Sample the box. Cheap enough for the pane's refresh tick.
pub fn box_state(lock_path: &Path) -> BoxState {
    let proc_locks = std::fs::read_to_string("/proc/locks").unwrap_or_default();
    let (lock_holder, waiters) = read_lock(lock_path, &proc_locks);
    let mem_available_mib = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map(|kib| kib / 1024)
        .unwrap_or(0);
    let builders = std::fs::read_dir("/proc")
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    let comm = e.path().join("comm");
                    std::fs::read_to_string(comm)
                        .map(|c| {
                            let c = c.trim();
                            c == "cargo" || c == "rustc" || c == "nvcc" || c == "ptxas"
                        })
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0);
    BoxState {
        lock_holder,
        waiters,
        mem_available_mib,
        builders,
    }
}

/// One harness arm, as its own `timing.txt` recorded it.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmSummary {
    pub name: String,
    pub trials: usize,
    /// Median tok/s. The harnesses report medians, so the pane does too —
    /// quoting a mean beside a file that records a median is two numbers for
    /// one quantity.
    pub median_tok_s: f64,
    pub min_tok_s: f64,
    pub max_tok_s: f64,
}

/// Parse a harness `timing.txt`.
///
/// Returns `None` for a file with no complete trial lines, rather than an arm
/// with zero trials: an arm that produced nothing is not an arm that measured
/// zero, and the pane must not render it as a result.
pub fn parse_timing(text: &str, name: &str) -> Option<ArmSummary> {
    let mut v: Vec<f64> = text
        .lines()
        .filter(|l| l.starts_with("trial-"))
        .filter_map(|l| l.rsplit_once("tok_s=")?.1.trim().parse::<f64>().ok())
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let median = if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    };
    Some(ArmSummary {
        name: name.to_string(),
        trials: n,
        median_tok_s: median,
        min_tok_s: v[0],
        max_tok_s: v[n - 1],
    })
}

/// Every arm under `runs_dir`, newest first.
pub fn scan_runs(runs_dir: &Path) -> Vec<ArmSummary> {
    let Ok(rd) = std::fs::read_dir(runs_dir) else {
        return Vec::new();
    };
    let mut out: Vec<(std::time::SystemTime, ArmSummary)> = Vec::new();
    for e in rd.flatten() {
        let t = e.path().join("timing.txt");
        let Ok(text) = std::fs::read_to_string(&t) else {
            continue;
        };
        let when = t
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        if let Some(a) = parse_timing(&text, &e.file_name().to_string_lossy()) {
            out.push((when, a));
        }
    }
    out.sort_by(|a, b| b.0.cmp(&a.0));
    out.into_iter().map(|(_, a)| a).collect()
}

/// Where the harnesses write, from `ATLAS_BENCH_RUNS` (colon-separated).
pub fn runs_dirs() -> Vec<PathBuf> {
    std::env::var("ATLAS_BENCH_RUNS")
        .map(|v| {
            v.split(':')
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .collect()
        })
        .unwrap_or_default()
}

/// THE SPREAD OF ARMS THAT SHOULD BE THE SAME NUMBER.
///
/// The single most useful figure this campaign produced: identical control
/// arms on GLM spread 2.16%, which is the floor under every delta measured on
/// that harness. An effect smaller than this is not a small effect, it is
/// invisible — and three kernel sweeps were spent before anyone computed it.
///
/// `None` for fewer than two arms: a spread needs two, and one arm reported as
/// a 0% spread would be the same two-sample underestimate that hid this for a
/// night.
pub fn control_spread_pct(controls: &[ArmSummary]) -> Option<f64> {
    if controls.len() < 2 {
        return None;
    }
    let (mut lo, mut hi, mut sum) = (f64::MAX, f64::MIN, 0.0);
    for c in controls {
        lo = lo.min(c.median_tok_s);
        hi = hi.max(c.median_tok_s);
        sum += c.median_tok_s;
    }
    let mean = sum / controls.len() as f64;
    (mean > 0.0).then(|| (hi - lo) / mean * 100.0)
}

/// Resolve `.gb10.lock` from `ATLAS_GPU_LOCK`, `FLUX_ATLAS_ROOT`, or nearest
/// ancestor directory containing `.gb10.lock` (falling back to `./.gb10.lock`).
pub(crate) fn resolve_gpu_lock_path(
    atlas_gpu_lock: Option<&str>,
    flux_atlas_root: Option<&str>,
    cwd: Option<&Path>,
) -> PathBuf {
    if let Some(explicit) = atlas_gpu_lock.map(str::trim).filter(|s| !s.is_empty()) {
        return PathBuf::from(explicit);
    }
    if let Some(root) = flux_atlas_root.map(str::trim).filter(|s| !s.is_empty()) {
        return Path::new(root).join(".gb10.lock");
    }
    if let Some(mut dir) = cwd {
        loop {
            let candidate = dir.join(".gb10.lock");
            if candidate.exists() {
                return candidate;
            }
            let Some(parent) = dir.parent() else {
                break;
            };
            dir = parent;
        }
    }
    PathBuf::from(".gb10.lock")
}

/// Cached snapshot of the Benchmarks pane's `/proc` and disk reads, sampled on
/// a background worker thread so the 10 Hz render loop never blocks on I/O.
#[derive(Debug, Clone)]
pub struct BenchSnapshot {
    pub lock_path: PathBuf,
    pub box_state: BoxState,
    pub runs_dirs: Vec<PathBuf>,
    pub arms: Vec<ArmSummary>,
    pub sampled: bool,
}

impl Default for BenchSnapshot {
    fn default() -> Self {
        Self {
            lock_path: PathBuf::from(".gb10.lock"),
            box_state: BoxState {
                lock_holder: None,
                waiters: Vec::new(),
                mem_available_mib: 0,
                builders: 0,
            },
            runs_dirs: Vec::new(),
            arms: Vec::new(),
            sampled: false,
        }
    }
}

/// Sample Box readiness (`/proc/locks`, `/proc/meminfo`, `/proc/*/comm`) and
/// recorded harness runs (`timing.txt` under `ATLAS_BENCH_RUNS`).
pub fn sample_snapshot() -> BenchSnapshot {
    let lock_path = resolve_gpu_lock_path(
        std::env::var("ATLAS_GPU_LOCK").ok().as_deref(),
        std::env::var("FLUX_ATLAS_ROOT").ok().as_deref(),
        std::env::current_dir().ok().as_deref(),
    );
    let state = box_state(&lock_path);
    let dirs = runs_dirs();
    let mut arms = Vec::new();
    for d in &dirs {
        arms.extend(scan_runs(d));
    }
    BenchSnapshot {
        lock_path,
        box_state: state,
        runs_dirs: dirs,
        arms,
        sampled: true,
    }
}

/// Run [`sample_snapshot`] on a one-shot background thread (`atlas-bench`) so
/// the render loop only ever `try_recv`s.
pub fn sample_in_background() -> std::sync::mpsc::Receiver<BenchSnapshot> {
    crate::tui::worker::spawn("atlas-bench", sample_snapshot, |_| BenchSnapshot::default())
}

#[cfg(test)]
#[path = "bench_tests.rs"]
mod tests;
