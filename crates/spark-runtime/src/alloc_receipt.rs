// SPDX-License-Identifier: AGPL-3.0-only

//! Construction allocation receipt.
//!
//! The startup ledger measures the host `MemAvailable` delta around each
//! phase. On its own that cannot say whether a gap is an allocation nobody
//! logged, allocator rounding, or memory the GPU backend never saw. This
//! module counts what the backend was asked for (device allocations and frees,
//! new page-locked host buffers) in process-wide relaxed atomics, and turns
//! the counters into one `STARTUP_ALLOC_RECEIPT` line per ledger phase and per
//! loader pass: requested and freed bytes next to the measured delta, with the
//! difference printed as `unattributed_mib`.
//!
//! Logging only: nothing here changes an allocation, a size or a kernel.
//! Loader passes are summed over their instances (48 SSM layers make one
//! line), so a load prints a few dozen lines, not one per layer.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Rounding model, not a measured allocator quantum: every allocation and free
/// rounded up to 2 MiB bounds what per-allocation rounding can cost.
pub const ROUNDING_MODEL_BYTES: u64 = 2 * 1024 * 1024;

const MIB: f64 = 1024.0 * 1024.0;

/// Allocation counters. [`GLOBAL`] is the one the CUDA backend feeds; tests
/// give a mock backend its own instance so parallel tests do not mix.
#[derive(Debug, Default)]
pub struct AllocCounters {
    alloc_bytes: AtomicU64,
    allocs: AtomicU64,
    free_bytes: AtomicU64,
    frees: AtomicU64,
    small_allocs: AtomicU64,
    rounded_alloc_bytes: AtomicU64,
    rounded_free_bytes: AtomicU64,
    pinned_bytes: AtomicU64,
    pinned_free_bytes: AtomicU64,
}

/// Counters for every allocation the CUDA backend makes in this process.
pub static GLOBAL: AllocCounters = AllocCounters::new();

fn round_up(bytes: u64) -> u64 {
    bytes
        .div_ceil(ROUNDING_MODEL_BYTES)
        .saturating_mul(ROUNDING_MODEL_BYTES)
}

impl AllocCounters {
    pub const fn new() -> Self {
        Self {
            alloc_bytes: AtomicU64::new(0),
            allocs: AtomicU64::new(0),
            free_bytes: AtomicU64::new(0),
            frees: AtomicU64::new(0),
            small_allocs: AtomicU64::new(0),
            rounded_alloc_bytes: AtomicU64::new(0),
            rounded_free_bytes: AtomicU64::new(0),
            pinned_bytes: AtomicU64::new(0),
            pinned_free_bytes: AtomicU64::new(0),
        }
    }

    /// A successful device (or managed) allocation of `bytes`.
    pub fn note_alloc(&self, bytes: usize) {
        let bytes = bytes as u64;
        self.alloc_bytes.fetch_add(bytes, Relaxed);
        self.allocs.fetch_add(1, Relaxed);
        if bytes < ROUNDING_MODEL_BYTES {
            self.small_allocs.fetch_add(1, Relaxed);
        }
        self.rounded_alloc_bytes.fetch_add(round_up(bytes), Relaxed);
    }

    /// A successful free of an allocation that was `bytes` long.
    pub fn note_free(&self, bytes: usize) {
        let bytes = bytes as u64;
        self.free_bytes.fetch_add(bytes, Relaxed);
        self.frees.fetch_add(1, Relaxed);
        self.rounded_free_bytes.fetch_add(round_up(bytes), Relaxed);
    }

    /// A new page-locked host buffer (pool hits are not new memory).
    pub fn note_pinned_alloc(&self, bytes: usize) {
        self.pinned_bytes.fetch_add(bytes as u64, Relaxed);
    }

    /// A page-locked host buffer returned to the OS.
    pub fn note_pinned_free(&self, bytes: usize) {
        self.pinned_free_bytes.fetch_add(bytes as u64, Relaxed);
    }

    pub fn snapshot(&self) -> AllocSnapshot {
        AllocSnapshot {
            alloc_bytes: self.alloc_bytes.load(Relaxed),
            allocs: self.allocs.load(Relaxed),
            free_bytes: self.free_bytes.load(Relaxed),
            frees: self.frees.load(Relaxed),
            small_allocs: self.small_allocs.load(Relaxed),
            rounded_alloc_bytes: self.rounded_alloc_bytes.load(Relaxed),
            rounded_free_bytes: self.rounded_free_bytes.load(Relaxed),
            pinned_bytes: self.pinned_bytes.load(Relaxed),
            pinned_free_bytes: self.pinned_free_bytes.load(Relaxed),
        }
    }
}

/// Monotonic counter values at one instant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocSnapshot {
    pub alloc_bytes: u64,
    pub allocs: u64,
    pub free_bytes: u64,
    pub frees: u64,
    pub small_allocs: u64,
    pub rounded_alloc_bytes: u64,
    pub rounded_free_bytes: u64,
    pub pinned_bytes: u64,
    pub pinned_free_bytes: u64,
}

/// Counters plus host `MemAvailable` at one instant.
#[derive(Clone, Copy, Debug)]
pub struct Mark {
    counters: AllocSnapshot,
    mem_available: Option<usize>,
}

/// Where a receipt reads its counters and `MemAvailable`. Production reads
/// [`GLOBAL`] and `/proc/meminfo`; tests pass their own pair.
#[derive(Clone, Copy)]
pub struct Source {
    counters: &'static AllocCounters,
    mem_available: fn() -> Option<usize>,
}

impl Source {
    pub fn live() -> Self {
        Self::new(&GLOBAL, crate::startup_ledger::mem_available_bytes)
    }

    pub const fn new(
        counters: &'static AllocCounters,
        mem_available: fn() -> Option<usize>,
    ) -> Self {
        Self {
            counters,
            mem_available,
        }
    }

    pub fn mark(&self) -> Mark {
        Mark {
            counters: self.counters.snapshot(),
            mem_available: (self.mem_available)(),
        }
    }
}

/// What one phase or pass asked the allocator for, against what the host lost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Receipt {
    pub requested_bytes: u64,
    pub allocs: u64,
    pub freed_bytes: u64,
    pub frees: u64,
    /// Allocations below [`ROUNDING_MODEL_BYTES`].
    pub small_allocs: u64,
    /// Net device bytes with every allocation and free rounded up to 2 MiB.
    pub rounded_net_bytes: i64,
    /// New page-locked host bytes minus page-locked bytes returned to the OS.
    pub pinned_net_bytes: i64,
    /// `MemAvailable` before minus after (positive = consumed); `None` when
    /// either read failed.
    pub mem_available_delta_bytes: Option<i64>,
}

fn diff(after: u64, before: u64) -> u64 {
    after.saturating_sub(before)
}

fn signed(bytes: u64) -> i64 {
    i64::try_from(bytes).unwrap_or(i64::MAX)
}

impl Receipt {
    pub fn between(start: Mark, end: Mark) -> Self {
        let (a, b) = (start.counters, end.counters);
        let rounded_alloc = diff(b.rounded_alloc_bytes, a.rounded_alloc_bytes);
        let rounded_free = diff(b.rounded_free_bytes, a.rounded_free_bytes);
        let pinned = diff(b.pinned_bytes, a.pinned_bytes);
        let pinned_free = diff(b.pinned_free_bytes, a.pinned_free_bytes);
        Self {
            requested_bytes: diff(b.alloc_bytes, a.alloc_bytes),
            allocs: diff(b.allocs, a.allocs),
            freed_bytes: diff(b.free_bytes, a.free_bytes),
            frees: diff(b.frees, a.frees),
            small_allocs: diff(b.small_allocs, a.small_allocs),
            rounded_net_bytes: signed(rounded_alloc) - signed(rounded_free),
            pinned_net_bytes: signed(pinned) - signed(pinned_free),
            mem_available_delta_bytes: crate::startup_ledger::signed_delta(
                start.mem_available,
                end.mem_available,
            ),
        }
    }

    /// Requested minus freed device bytes plus net new page-locked host bytes:
    /// what the backend can account for.
    pub fn net_bytes(&self) -> i64 {
        signed(self.requested_bytes) - signed(self.freed_bytes) + self.pinned_net_bytes
    }

    /// The part of the `MemAvailable` delta the backend counters do not
    /// explain: allocator rounding, driver-internal memory, host heap, and
    /// allocations that bypass the backend.
    pub fn unattributed_bytes(&self) -> Option<i64> {
        self.mem_available_delta_bytes
            .map(|delta| delta - self.net_bytes())
    }

    fn accumulate(&mut self, other: &Receipt) {
        self.requested_bytes += other.requested_bytes;
        self.allocs += other.allocs;
        self.freed_bytes += other.freed_bytes;
        self.frees += other.frees;
        self.small_allocs += other.small_allocs;
        self.rounded_net_bytes += other.rounded_net_bytes;
        self.pinned_net_bytes += other.pinned_net_bytes;
        self.mem_available_delta_bytes = self
            .mem_available_delta_bytes
            .zip(other.mem_available_delta_bytes)
            .map(|(a, b)| a + b);
    }

    fn minus(&self, other: &Receipt) -> Receipt {
        Receipt {
            requested_bytes: self.requested_bytes.saturating_sub(other.requested_bytes),
            allocs: self.allocs.saturating_sub(other.allocs),
            freed_bytes: self.freed_bytes.saturating_sub(other.freed_bytes),
            frees: self.frees.saturating_sub(other.frees),
            small_allocs: self.small_allocs.saturating_sub(other.small_allocs),
            rounded_net_bytes: self.rounded_net_bytes - other.rounded_net_bytes,
            pinned_net_bytes: self.pinned_net_bytes - other.pinned_net_bytes,
            mem_available_delta_bytes: self
                .mem_available_delta_bytes
                .zip(other.mem_available_delta_bytes)
                .map(|(a, b)| a - b),
        }
    }

    fn fields(&self) -> String {
        let mib = |bytes: i64| format!("{:.1}", bytes as f64 / MIB);
        let opt = |bytes: Option<i64>| bytes.map_or_else(|| "-".to_string(), mib);
        format!(
            "requested_mib={} freed_mib={} net_mib={} allocs={} frees={} small_allocs={} \
             rounded_net_mib={} pinned_net_mib={} mem_available_delta_mib={} unattributed_mib={}",
            mib(signed(self.requested_bytes)),
            mib(signed(self.freed_bytes)),
            mib(self.net_bytes()),
            self.allocs,
            self.frees,
            self.small_allocs,
            mib(self.rounded_net_bytes),
            mib(self.pinned_net_bytes),
            opt(self.mem_available_delta_bytes),
            opt(self.unattributed_bytes()),
        )
    }
}

/// One loader pass, summed over its instances.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassReceipt {
    pub name: String,
    pub instances: u64,
    pub receipt: Receipt,
}

/// One phase: its total and the loader passes (or nested phases) inside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseReceipt {
    pub phase: String,
    pub total: Receipt,
    pub passes: Vec<PassReceipt>,
}

impl PhaseReceipt {
    /// What ran inside the phase but outside every pass.
    pub fn outside_passes(&self) -> Receipt {
        let mut passes = Receipt {
            mem_available_delta_bytes: Some(0),
            ..Receipt::default()
        };
        for pass in &self.passes {
            passes.accumulate(&pass.receipt);
        }
        self.total.minus(&passes)
    }

    /// The phase line, one line per pass, and the outside-passes remainder
    /// when the phase had passes.
    pub fn lines(&self) -> Vec<String> {
        let mut lines = vec![format!(
            "STARTUP_ALLOC_RECEIPT phase={} {}",
            self.phase,
            self.total.fields()
        )];
        for pass in &self.passes {
            lines.push(format!(
                "STARTUP_ALLOC_RECEIPT phase={} pass={} instances={} {}",
                self.phase,
                pass.name,
                pass.instances,
                pass.receipt.fields()
            ));
        }
        if !self.passes.is_empty() {
            lines.push(format!(
                "STARTUP_ALLOC_RECEIPT phase={} pass=outside-passes {}",
                self.phase,
                self.outside_passes().fields()
            ));
        }
        lines
    }

    pub fn log(&self) {
        for line in self.lines() {
            tracing::info!(target: "atlas_startup_budget", "{line}");
        }
    }
}

struct Frame {
    phase: String,
    source: Source,
    start: Mark,
    passes: Vec<PassReceipt>,
    pass_open: bool,
}

impl Frame {
    fn add_pass(&mut self, name: &str, receipt: Receipt) {
        match self.passes.iter_mut().find(|pass| pass.name == name) {
            Some(pass) => {
                pass.instances += 1;
                pass.receipt.accumulate(&receipt);
            }
            None => self.passes.push(PassReceipt {
                name: name.to_string(),
                instances: 1,
                receipt,
            }),
        }
    }
}

#[derive(Clone, Copy)]
struct Session {
    source: Source,
    start: Mark,
    in_phases: Receipt,
}

thread_local! {
    // Phases and passes run on the loading thread; per-thread state keeps
    // parallel tests (and a concurrent load on another thread) apart.
    static FRAMES: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
}

/// An open receipt phase on this thread. [`PhaseGuard::finish`] closes it and
/// returns its receipt; dropping it unfinished (an early return, a panic)
/// closes it too, so a later load never attaches to a dead frame.
#[must_use = "a phase is measured until it is finished or dropped"]
pub struct PhaseGuard {
    depth: usize,
    source: Source,
    log_on_drop: bool,
    closed: bool,
}

/// Open a receipt phase. Passes opened on this thread until it closes are
/// summed into it; a nested phase counts as one pass of its parent; a
/// top-level phase is added to the load session.
pub fn begin_phase(source: Source, phase: &str) -> PhaseGuard {
    let depth = FRAMES.with(|frames| {
        let mut frames = frames.borrow_mut();
        frames.push(Frame {
            phase: phase.to_string(),
            source,
            start: source.mark(),
            passes: Vec::new(),
            pass_open: false,
        });
        frames.len() - 1
    });
    PhaseGuard {
        depth,
        source,
        log_on_drop: false,
        closed: false,
    }
}

/// [`begin_phase`] whose guard logs its receipt lines when dropped: for load
/// steps that are a run of statements rather than one closure.
pub fn begin_logged_phase(source: Source, phase: &str) -> PhaseGuard {
    let mut guard = begin_phase(source, phase);
    guard.log_on_drop = true;
    guard
}

impl PhaseGuard {
    pub fn finish(mut self) -> PhaseReceipt {
        self.close()
    }

    fn close(&mut self) -> PhaseReceipt {
        self.closed = true;
        let end = self.source.mark();
        let frame = FRAMES.with(|frames| {
            let mut frames = frames.borrow_mut();
            frames.truncate(self.depth + 1);
            frames.pop()
        });
        let Some(frame) = frame else {
            return PhaseReceipt {
                phase: String::new(),
                total: Receipt::default(),
                passes: Vec::new(),
            };
        };
        let receipt = PhaseReceipt {
            phase: frame.phase,
            total: Receipt::between(frame.start, end),
            passes: frame.passes,
        };
        let attached = FRAMES.with(|frames| match frames.borrow_mut().last_mut() {
            // An open pass already covers this phase's bytes.
            Some(parent) if parent.pass_open => true,
            Some(parent) => {
                parent.add_pass(&receipt.phase, receipt.total);
                true
            }
            None => false,
        });
        if !attached {
            SESSION.with(|session| {
                if let Some(session) = session.borrow_mut().as_mut() {
                    session.in_phases.accumulate(&receipt.total);
                }
            });
        }
        receipt
    }
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        if !self.closed {
            let receipt = self.close();
            if self.log_on_drop {
                receipt.log();
            }
        }
    }
}

/// Run `body` as one receipt phase (see [`begin_phase`]).
pub fn run_phase<T>(source: Source, phase: &str, body: impl FnOnce() -> T) -> (T, PhaseReceipt) {
    let guard = begin_phase(source, phase);
    let value = body();
    (value, guard.finish())
}

/// Measures one loader pass until dropped.
#[must_use = "a pass is measured until the guard drops"]
pub struct PassGuard {
    name: &'static str,
    start: Option<(Source, Mark, usize)>,
}

/// Count one loader pass into the innermost open phase on this thread.
/// Passes do not nest (an inner pass is folded into the outer one), and a pass
/// outside every phase measures nothing.
pub fn pass(name: &'static str) -> PassGuard {
    FRAMES.with(|frames| {
        let mut frames = frames.borrow_mut();
        let depth = frames.len();
        match frames.last_mut() {
            Some(frame) if !frame.pass_open => {
                frame.pass_open = true;
                let source = frame.source;
                PassGuard {
                    name,
                    start: Some((source, source.mark(), depth)),
                }
            }
            _ => PassGuard { name, start: None },
        }
    })
}

impl Drop for PassGuard {
    fn drop(&mut self) {
        let Some((source, start, depth)) = self.start.take() else {
            return;
        };
        let receipt = Receipt::between(start, source.mark());
        FRAMES.with(|frames| {
            let mut frames = frames.borrow_mut();
            if let Some(frame) = depth.checked_sub(1).and_then(|index| frames.get_mut(index)) {
                frame.pass_open = false;
                frame.add_pass(self.name, receipt);
            }
        });
    }
}

/// Everything since [`begin_session`], split into top-level phases and the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionReceipt {
    pub total: Receipt,
    pub in_phases: Receipt,
}

impl SessionReceipt {
    /// What happened outside every top-level phase.
    pub fn outside_phases(&self) -> Receipt {
        self.total.minus(&self.in_phases)
    }

    pub fn line(&self) -> String {
        let outside = self.outside_phases();
        let mib = |bytes: Option<i64>| {
            bytes.map_or_else(|| "-".to_string(), |b| format!("{:.1}", b as f64 / MIB))
        };
        format!(
            "STARTUP_ALLOC_RECEIPT_TOTAL since=ledger-reset {} in_phases_mem_available_delta_mib={} \
             outside_phases_net_mib={} outside_phases_mem_available_delta_mib={}",
            self.total.fields(),
            mib(self.in_phases.mem_available_delta_bytes),
            mib(Some(outside.net_bytes())),
            mib(outside.mem_available_delta_bytes),
        )
    }
}

/// Start a load session on this thread (the startup ledger's `reset`).
pub fn begin_session(source: Source) {
    let start = source.mark();
    SESSION.with(|session| {
        *session.borrow_mut() = Some(Session {
            source,
            start,
            in_phases: Receipt {
                mem_available_delta_bytes: Some(0),
                ..Receipt::default()
            },
        });
    });
}

/// The receipt since [`begin_session`] on this thread, if one was started.
pub fn session_receipt() -> Option<SessionReceipt> {
    SESSION.with(|session| {
        let session = (*session.borrow())?;
        Some(SessionReceipt {
            total: Receipt::between(session.start, session.source.mark()),
            in_phases: session.in_phases,
        })
    })
}

#[cfg(test)]
#[path = "alloc_receipt_tests.rs"]
mod tests;
