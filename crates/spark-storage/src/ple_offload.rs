// SPDX-License-Identifier: AGPL-3.0-only

//! Sparse direct-I/O or page-cache reader for Qwen3.8-Flash-Next PLE n-gram rows.
//!
//! The table has 320M rows but inference selects only 16 rows per token.
//! Sidecars page-pack 91 NVFP4 records into 8 KiB, so every selected row is
//! fulfilled by exactly one aligned read with only two padding bytes/page.

use anyhow::{Context, Result, bail, ensure};
use io_uring::{IoUring, opcode, types};
use serde::Deserialize;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::{HashMap, VecDeque, hash_map::Entry};
use std::fs::File;
use std::ops::Range;
use std::os::fd::{FromRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::{Duration, Instant};

#[path = "ple_verify_read.rs"]
mod verify_read;
pub use verify_read::VERIFY_PARALLEL_PREAD;

const EXPECTED_FORMAT: &str = "qwen38-flash-next-ple-nvfp4-direct-v1";
const EXPECTED_PAGE_BYTES: usize = 8192;
const EXPECTED_RECORD_BYTES: usize = 90;
const EXPECTED_RECORDS_PER_PAGE: usize = 91;
// ple_plan.rs is dependency-free and declares the same geometry; keep them tied.
const _: () = assert!(
    crate::ple_plan::PAGE_BYTES == EXPECTED_PAGE_BYTES
        && crate::ple_plan::RECORD_BYTES == EXPECTED_RECORD_BYTES
        && crate::ple_plan::RECORDS_PER_PAGE == EXPECTED_RECORDS_PER_PAGE
        && EXPECTED_PAGE_BYTES.is_multiple_of(crate::ple_plan::SECTOR_BYTES)
);

#[derive(Debug, Deserialize)]
struct Manifest {
    format: String,
    page_bytes: usize,
    record_bytes: usize,
    records_per_page: usize,
    packed_bytes: usize,
    scale_bytes: usize,
    group_size: usize,
    entries: Vec<ManifestEntry>,
}

#[derive(Debug, Deserialize)]
struct ManifestEntry {
    shard: usize,
    file: PathBuf,
    rows: usize,
    width: usize,
    scale2: f32,
    bytes: usize,
}

struct AlignedPage {
    ptr: *mut u8,
    layout: Layout,
}

// SAFETY: the allocation is uniquely owned and this type deliberately does
// not implement Sync. Moving ownership to an I/O worker is safe.
unsafe impl Send for AlignedPage {}

impl AlignedPage {
    fn new() -> Self {
        let layout = Layout::from_size_align(EXPECTED_PAGE_BYTES, EXPECTED_PAGE_BYTES)
            .expect("valid PLE direct-I/O page layout");
        // SAFETY: `layout` has non-zero size (`EXPECTED_PAGE_BYTES = 8192`) and
        // power-of-two alignment; null return is handled immediately below.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { ptr, layout }
    }

    /// Borrow a writable sub-slice `[offset..offset + len)` of this aligned page buffer.
    fn as_mut_slice(&mut self, offset: usize, len: usize) -> &mut [u8] {
        assert!(
            offset
                .checked_add(len)
                .is_some_and(|end| end <= EXPECTED_PAGE_BYTES),
            "PLE page slice [{offset}..+{len}) outside {EXPECTED_PAGE_BYTES}-byte page buffer"
        );
        // SAFETY: `self.ptr` points to a uniquely owned, initialized
        // `EXPECTED_PAGE_BYTES`-byte allocation (exclusive `&mut self` ensures
        // no I/O or other alias is active), and `offset + len <= EXPECTED_PAGE_BYTES`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset), len) }
    }

    fn record(&self, slot: usize) -> [u8; EXPECTED_RECORD_BYTES] {
        let byte_end = slot
            .checked_add(1)
            .and_then(|next| next.checked_mul(EXPECTED_RECORD_BYTES))
            .expect("PLE record slot overflow");
        assert!(
            byte_end <= EXPECTED_PAGE_BYTES,
            "PLE record slot {slot} outside the {EXPECTED_PAGE_BYTES}-byte page buffer"
        );
        let mut record = [0u8; EXPECTED_RECORD_BYTES];
        // SAFETY: `(slot + 1) * EXPECTED_RECORD_BYTES <= EXPECTED_PAGE_BYTES` was
        // asserted above, `self.ptr` is an initialized page buffer with no read in
        // flight, and `record` is a disjoint stack array.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.add(slot * EXPECTED_RECORD_BYTES),
                record.as_mut_ptr(),
                EXPECTED_RECORD_BYTES,
            );
        }
        record
    }

    /// The record at byte `offset` of the page buffer (planned reads).
    fn record_at(&self, offset: usize) -> [u8; EXPECTED_RECORD_BYTES] {
        assert!(
            offset + EXPECTED_RECORD_BYTES <= EXPECTED_PAGE_BYTES,
            "PLE record offset {offset} outside the page buffer"
        );
        let mut record = [0u8; EXPECTED_RECORD_BYTES];
        // SAFETY: the buffer is EXPECTED_PAGE_BYTES long (asserted bound above)
        // and no read targeting it is in flight while the caller copies.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.add(offset),
                record.as_mut_ptr(),
                EXPECTED_RECORD_BYTES,
            );
        }
        record
    }
}

impl Drop for AlignedPage {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated with `alloc_zeroed(self.layout)` in
        // `AlignedPage::new` and is deallocated once here after any owning ring drops.
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

struct Shard {
    file: File,
    rows: usize,
    scale2: f32,
}

/// Environment variable controlling read-only `mmap` PLE shard backing:
/// - `"1"` forces `mmap` on;
/// - `"0"` forces `mmap` off (always uses `io_uring` / `pread64`);
/// - unset / empty auto-enables `mmap` when the PLE directory resides on
///   Linux `tmpfs` or `ramfs`.
pub const PLE_MMAP_SELECTOR: &str = "ATLAS_PLE_MMAP";

/// Minimum selection count before `mmap` reads parallelize `memcpy` across
/// scoped worker threads.
const MMAP_PARALLEL_MIN_SELECTIONS: usize = 1_024;
/// Upper bound on worker threads for parallel `mmap` record copies.
const MMAP_MAX_COPY_THREADS: usize = 8;

/// Linux `RAMFS_MAGIC` from `<linux/magic.h>` (`libc` 0.2 exports `TMPFS_MAGIC`
/// but omits `RAMFS_MAGIC`).
#[cfg(target_os = "linux")]
const LINUX_RAMFS_MAGIC: libc::c_long = 0x8584_58f6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PleMmapPolicy {
    ForceOn,
    ForceOff,
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InMemoryFs {
    Tmpfs,
    Ramfs,
}

impl InMemoryFs {
    const fn label(self) -> &'static str {
        match self {
            Self::Tmpfs => "tmpfs",
            Self::Ramfs => "ramfs",
        }
    }
}

fn parse_ple_mmap_policy(value: Option<&str>) -> Result<PleMmapPolicy> {
    match value.map(str::trim) {
        None | Some("") => Ok(PleMmapPolicy::Auto),
        Some("1") => Ok(PleMmapPolicy::ForceOn),
        Some("0") => Ok(PleMmapPolicy::ForceOff),
        Some(other) => bail!("{PLE_MMAP_SELECTOR} must be unset, 0, or 1; got {other:?}"),
    }
}

const fn resolve_ple_mmap_trigger(
    policy: PleMmapPolicy,
    in_memory_fs: Option<InMemoryFs>,
) -> Option<&'static str> {
    match policy {
        PleMmapPolicy::ForceOn => Some("env"),
        PleMmapPolicy::ForceOff => None,
        PleMmapPolicy::Auto => match in_memory_fs {
            Some(fs) => Some(fs.label()),
            None => None,
        },
    }
}

#[cfg(target_os = "linux")]
const fn classify_fs_magic(f_type: libc::c_long) -> Option<InMemoryFs> {
    if f_type == libc::TMPFS_MAGIC {
        Some(InMemoryFs::Tmpfs)
    } else if f_type == LINUX_RAMFS_MAGIC {
        Some(InMemoryFs::Ramfs)
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn detect_in_memory_fs(dir: &Path) -> Option<InMemoryFs> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let path = CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut stat = std::mem::MaybeUninit::<libc::statfs>::zeroed();
    // SAFETY: `path` is a NUL-terminated C string and `stat.as_mut_ptr()` points
    // to valid writable memory for `libc::statfs`.
    let rc = unsafe { libc::statfs(path.as_ptr(), stat.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: `libc::statfs` returned 0, so `stat` is fully initialized.
    let stat = unsafe { stat.assume_init() };
    classify_fs_magic(stat.f_type)
}

#[cfg(not(target_os = "linux"))]
fn detect_in_memory_fs(_dir: &Path) -> Option<InMemoryFs> {
    None
}

/// Safe RAII wrapper for a read-only `mmap(PROT_READ, MAP_SHARED)` shard view.
///
/// Unmaps cleanly in [`Drop`] and bounds-checks every record copy against `len`.
struct MappedShard {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
}

// SAFETY: `MappedShard` owns a read-only (`PROT_READ`) shared mapping whose
// address range stays valid until `Drop`. Concurrent read-only copies across
// threads and moving ownership across threads are both sound.
unsafe impl Send for MappedShard {}
unsafe impl Sync for MappedShard {}

impl MappedShard {
    fn map(file: &File, len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::NonNull::dangling(),
                len: 0,
            });
        }
        let fd = raw_fd(file);
        // SAFETY: `ptr::null_mut()` lets the kernel pick the address, `len > 0`
        // matches the validated shard file size, `fd` is an open readable file
        // descriptor, and `offset = 0`. `MAP_FAILED` is checked immediately.
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("mmap PLE shard ({len} B) failed"));
        }
        let ptr = std::ptr::NonNull::new(mapped.cast::<u8>())
            .context("mmap PLE shard returned null pointer")?;
        #[cfg(target_os = "linux")]
        // SAFETY: `[mapped, mapped + len)` is a live mapping from `mmap` above.
        // `MADV_RANDOM` is an advisory hint; ignore errors if unsupported.
        unsafe {
            let _ = libc::madvise(mapped, len, libc::MADV_RANDOM);
        }
        Ok(Self { ptr, len })
    }

    #[inline]
    const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `self.ptr` points to `self.len` readable bytes from a live
        // `PROT_READ` mapping that remains valid for the lifetime of `&self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    fn read_record(&self, row: usize) -> Result<[u8; EXPECTED_RECORD_BYTES]> {
        let page = row / EXPECTED_RECORDS_PER_PAGE;
        let slot = row % EXPECTED_RECORDS_PER_PAGE;
        let offset = page
            .checked_mul(EXPECTED_PAGE_BYTES)
            .and_then(|p| p.checked_add(slot * EXPECTED_RECORD_BYTES))
            .context("PLE mmap record offset overflow")?;
        let end = offset
            .checked_add(EXPECTED_RECORD_BYTES)
            .context("PLE mmap record end overflow")?;
        let slice = self.as_slice();
        ensure!(
            end <= slice.len(),
            "PLE mmap record range {offset}..{end} exceeds mapped shard length {}",
            slice.len()
        );
        let mut record = [0u8; EXPECTED_RECORD_BYTES];
        record.copy_from_slice(&slice[offset..end]);
        Ok(record)
    }
}

impl Drop for MappedShard {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        // SAFETY: `self.ptr` and `self.len` come from a successful `libc::mmap`
        // call in `MappedShard::map` and are unmapped exactly once on drop.
        // Destructors never panic.
        unsafe {
            let _ = libc::munmap(self.ptr.as_ptr().cast(), self.len);
        }
    }
}

#[inline]
fn mmap_copy_threads(selections: usize) -> usize {
    if selections < MMAP_PARALLEL_MIN_SELECTIONS {
        return 1;
    }
    let avail = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let max_useful = selections.div_ceil(256);
    avail.min(MMAP_MAX_COPY_THREADS).min(max_useful).max(1)
}

fn validate_mmap_selections(
    shards: &[Shard],
    mmaps: &[MappedShard],
    selections: &[(usize, usize)],
) -> Result<()> {
    for &(shard_idx, row) in selections {
        let shard = shards
            .get(shard_idx)
            .with_context(|| format!("PLE shard {shard_idx} out of range"))?;
        ensure!(
            row < shard.rows,
            "PLE row {row} out of range for shard {shard_idx}"
        );
        ensure!(
            shard_idx < mmaps.len(),
            "PLE mmap shard {shard_idx} missing"
        );
    }
    Ok(())
}

fn copy_mmap_chunk(
    shards: &[Shard],
    mmaps: &[MappedShard],
    selections: &[(usize, usize)],
    out: &mut [std::mem::MaybeUninit<PleNvfp4Row>],
) -> Result<()> {
    ensure!(
        selections.len() == out.len(),
        "PLE mmap copy chunk length mismatch: {} selections for {} slots",
        selections.len(),
        out.len()
    );
    for (&(shard_idx, row), dst) in selections.iter().zip(out.iter_mut()) {
        let shard = shards
            .get(shard_idx)
            .with_context(|| format!("PLE shard {shard_idx} out of range"))?;
        ensure!(
            row < shard.rows,
            "PLE row {row} out of range for shard {shard_idx}"
        );
        let mapped = mmaps
            .get(shard_idx)
            .with_context(|| format!("PLE mmap shard {shard_idx} missing"))?;
        let record = mapped.read_record(row)?;
        dst.write(PleNvfp4Row {
            record,
            scale2: shard.scale2,
        });
    }
    Ok(())
}

fn read_mmap_rows(
    shards: &[Shard],
    mmaps: &[MappedShard],
    selections: &[(usize, usize)],
) -> Result<Vec<PleNvfp4Row>> {
    if selections.is_empty() {
        return Ok(Vec::new());
    }
    let threads = mmap_copy_threads(selections.len());
    if threads <= 1 {
        let mut out = Vec::with_capacity(selections.len());
        for &(shard_idx, row) in selections {
            let shard = shards
                .get(shard_idx)
                .with_context(|| format!("PLE shard {shard_idx} out of range"))?;
            ensure!(
                row < shard.rows,
                "PLE row {row} out of range for shard {shard_idx}"
            );
            let mapped = mmaps
                .get(shard_idx)
                .with_context(|| format!("PLE mmap shard {shard_idx} missing"))?;
            let record = mapped.read_record(row)?;
            out.push(PleNvfp4Row {
                record,
                scale2: shard.scale2,
            });
        }
        return Ok(out);
    }

    let mut out = Vec::<PleNvfp4Row>::with_capacity(selections.len());
    let spare = &mut out.spare_capacity_mut()[..selections.len()];
    let chunk_len = selections.len().div_ceil(threads);
    let results: Vec<Result<()>> = std::thread::scope(|scope| {
        let mut sel_chunks = selections.chunks(chunk_len);
        let mut out_chunks = spare.chunks_mut(chunk_len);
        let first = sel_chunks.next().zip(out_chunks.next());
        let handles: Vec<_> = sel_chunks
            .zip(out_chunks)
            .map(|(sel_chunk, out_chunk)| {
                scope.spawn(move || copy_mmap_chunk(shards, mmaps, sel_chunk, out_chunk))
            })
            .collect();
        let mut res = Vec::with_capacity(handles.len() + usize::from(first.is_some()));
        if let Some((first_sel, first_out)) = first {
            res.push(copy_mmap_chunk(shards, mmaps, first_sel, first_out));
        }
        for handle in handles {
            res.push(
                handle
                    .join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("PLE mmap copy worker panicked"))),
            );
        }
        res
    });
    for result in results {
        result?;
    }
    // SAFETY: `chunks` / `chunks_mut` partition `0..selections.len()` into
    // disjoint equal-length pairs, and every chunk returned `Ok(())`, proving
    // every slot in `spare[..selections.len()]` was initialized via
    // `MaybeUninit::write`. `PleNvfp4Row` has no drop glue (asserted below).
    unsafe {
        out.set_len(selections.len());
    }
    Ok(out)
}

/// Storage policy for sparse PLE pages.
///
/// `Direct` avoids displacing unified-memory model state. `PageCache` uses
/// ordinary buffered I/O so Linux can retain useful PLE pages in evictable
/// system RAM without a second permanently-owned Atlas allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PleIoMode {
    Direct,
    PageCache,
}

/// One packed PLE row: 80 E2M1 bytes, 10 E4M3 group scales, and its global scale.
#[derive(Clone, Debug)]
pub struct PleNvfp4Row {
    pub record: [u8; EXPECTED_RECORD_BYTES],
    pub scale2: f32,
}

const _: () = assert!(!std::mem::needs_drop::<PleNvfp4Row>());

/// Batched sparse PLE reader. A decode request normally contains 16 rows.
pub struct PleOffloadReader {
    /// Persistent `ple-ring-0..N` worker pool (`ATLAS_PLE_READ_RINGS` = 2 or 4):
    /// workers `1..N` own their extra rings and registered buffers, while worker
    /// `0` borrows `ring` + `pages` during planned multi-ring reads so verify
    /// steps avoid per-call OS thread creation/join overhead. Declared before
    /// `ring`, `pages`, and `shards` so all workers join and release their
    /// `io_uring` rings before primary buffers or shard fds drop.
    ring_pool: Option<RingWorkerPool>,
    /// Persistent `ple-pread-0..N` worker pool used when `io_uring` is
    /// unavailable (e.g. Docker/cloud default seccomp profiles). Declared
    /// before `shards` so all workers join before shard fds drop.
    pread_pool: Option<PreadWorkerPool>,
    ring: Option<IoUring>,
    fixed_buffers: bool,
    pages: Vec<AlignedPage>,
    /// Optional read-only `mmap(PROT_READ, MAP_SHARED)` backing views per shard
    /// (`ATLAS_PLE_MMAP=1` or auto-engaged on `tmpfs`/`ramfs`). Declared before
    /// `shards` so mappings unmap cleanly before shard fds close.
    shard_mmaps: Vec<MappedShard>,
    shards: Arc<[Shard]>,
    queue_depth: usize,
    io_timeout: Duration,
    cache: HashMap<(usize, usize), CachedPage>,
    cache_order: VecDeque<((usize, usize), u64)>,
    cache_capacity_pages: usize,
    cache_clock: u64,
    /// `ATLAS_PLE_HALF_PAGE=1` (Direct mode, no private cache): read only the
    /// 4 KiB half of an 8 KiB page that holds the record (1 of 91 slots
    /// straddles the halves and still reads the whole page).
    half_page: bool,
    /// K9 item W selectors for streamed prefill reads (all default off).
    read_plan: ReadPlanConfig,
    /// One page buffer per synchronous `pread64` lane, allocated only when
    /// io_uring is unavailable (`ATLAS_PLE_PREAD_LANES`, default 256). Empty
    /// when the reader has a ring.
    pread_pages: Vec<AlignedPage>,
    /// Bounded in-memory FIFO cache of 90-byte NVFP4 records keyed by `(shard, row)`.
    record_cache: RecordCache,
    /// Fixed-footprint lock-free direct-mapped atomic cache of 90-byte NVFP4
    /// records (`ATLAS_PLE_PARKED_CACHE_MB`, default 128 MiB -> 104.0 MiB exact).
    parked_cache: Arc<ParkedPleCache>,
    /// `ATLAS_PLE_VERIFY_PARALLEL_PREAD` (default unset = auto-parallel for
    /// small verify misses without io_uring, no per-call receipts).
    verify_read: verify_read::VerifyReadConfig,
}

const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<PleOffloadReader>();
};

const MAX_RECORD_CACHE_ENTRIES: usize = 2_097_152;
const INITIAL_RECORD_CACHE_CAPACITY: usize = 16_384;
/// Small default entry cap (~110 KiB) used when `cache_bytes == 0` and neither
/// `ATLAS_PLE_CACHE_MB` nor `ATLAS_PLE_RECORD_CACHE_ENTRIES` is set. Keeps
/// short repeated decode/verify reads warm without allocating the 440 MiB full
/// record cache.
const DEFAULT_UNCACHED_RECORD_CACHE_ENTRIES: usize = 1_024;
/// Approximate byte footprint per [`RecordCache`] entry (90 B record + HashMap
/// table overhead + FIFO ring slot) used to size the cache from `cache_bytes`.
const ESTIMATED_RECORD_CACHE_ENTRY_BYTES: usize = 256;
const RECORD_CACHE_ENTRIES_ENV: &str = "ATLAS_PLE_RECORD_CACHE_ENTRIES";
const PLE_CACHE_MB_ENV: &str = "ATLAS_PLE_CACHE_MB";

/// Resolve the maximum entry count for [`RecordCache`].
///
/// Precedence:
/// 1. Explicit `ATLAS_PLE_RECORD_CACHE_ENTRIES` override (`0` disables caching).
/// 2. Explicit `ATLAS_PLE_CACHE_MB=0` disables the record cache completely (`0`
///    entries, zero heap allocation).
/// 3. `cache_bytes == 0` with no env override uses a tiny working-set cap
///    ([`DEFAULT_UNCACHED_RECORD_CACHE_ENTRIES`]).
/// 4. `cache_bytes > 0` scales proportionally up to [`MAX_RECORD_CACHE_ENTRIES`].
fn resolve_record_cache_max_entries(
    cache_bytes: usize,
    entries_override: Option<&str>,
    ple_cache_mb: Option<&str>,
) -> usize {
    if let Some(raw) = entries_override
        && let Ok(parsed) = raw.trim().parse::<usize>()
    {
        return parsed;
    }
    if ple_cache_mb.map(str::trim) == Some("0") {
        return 0;
    }
    if cache_bytes == 0 {
        return DEFAULT_UNCACHED_RECORD_CACHE_ENTRIES;
    }
    (cache_bytes / ESTIMATED_RECORD_CACHE_ENTRY_BYTES).clamp(
        DEFAULT_UNCACHED_RECORD_CACHE_ENTRIES,
        MAX_RECORD_CACHE_ENTRIES,
    )
}

type RecordCacheKey = (u16, u32);

/// Bounded O(1) FIFO cache for packed 90-byte NVFP4 records.
///
/// Once full (`max_entries`), inserting a fresh key evicts the oldest key via
/// an in-place ring buffer so memory stays strictly capped while the cache
/// continuously tracks the active working set. Re-inserting an existing key
/// updates its payload in place without duplicating it in the eviction ring.
struct RecordCache {
    map: HashMap<RecordCacheKey, [u8; EXPECTED_RECORD_BYTES]>,
    ring: Vec<RecordCacheKey>,
    cursor: usize,
    max_entries: usize,
}

impl RecordCache {
    fn new(max_entries: usize, initial_capacity: usize) -> Self {
        let initial = initial_capacity.min(max_entries);
        Self {
            map: HashMap::with_capacity(initial),
            ring: Vec::with_capacity(initial),
            cursor: 0,
            max_entries,
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.map.len()
    }

    #[cfg(test)]
    fn ring_len(&self) -> usize {
        self.ring.len()
    }

    #[inline]
    fn get(&self, key: &RecordCacheKey) -> Option<&[u8; EXPECTED_RECORD_BYTES]> {
        self.map.get(key)
    }

    #[inline]
    fn insert(&mut self, key: RecordCacheKey, record: [u8; EXPECTED_RECORD_BYTES]) {
        if self.max_entries == 0 {
            return;
        }
        if self.ring.len() < self.max_entries {
            match self.map.entry(key) {
                Entry::Occupied(entry) => {
                    *entry.into_mut() = record;
                }
                Entry::Vacant(entry) => {
                    entry.insert(record);
                    self.ring.push(key);
                }
            }
            return;
        }
        if let Some(existing) = self.map.get_mut(&key) {
            *existing = record;
            return;
        }
        let oldest = std::mem::replace(&mut self.ring[self.cursor], key);
        self.cursor += 1;
        if self.cursor == self.max_entries {
            self.cursor = 0;
        }
        self.map.remove(&oldest);
        self.map.insert(key, record);
    }
}

/// Environment variable sizing the fixed-footprint lock-free direct-mapped PLE
/// parked record cache (`0` disables allocation completely; default `128` MiB).
pub const PLE_PARKED_CACHE_MB_ENV: &str = "ATLAS_PLE_PARKED_CACHE_MB";
/// Default budget in MiB for [`ParkedPleCache`]. With 104-byte slots rounded
/// down to a power of two, 128 MiB yields `2^20 = 1,048,576` slots (`104.0 MiB` exact).
pub const DEFAULT_PARKED_CACHE_MB: usize = 128;
/// Hard ceiling on [`ParkedPleCache`] budget in MiB.
const MAX_PARKED_CACHE_MB: usize = 8192;
const EXPECTED_RECORD_U64_WORDS: usize = EXPECTED_RECORD_BYTES / 8;
const PARKED_KEY_HI_MASK: u64 = !0xFFFFu64;

const _: () = assert!(
    EXPECTED_RECORD_BYTES == EXPECTED_RECORD_U64_WORDS * 8 + 2 && EXPECTED_RECORD_U64_WORDS == 11
);

#[repr(C)]
struct ParkedSlot {
    /// 64-bit seqlock counter: `0` = empty, odd = write in progress, even >= 2 = published.
    seq: AtomicU64,
    /// Packed `(shard: u16) << 48 | (row: u32) << 16 | tail_bytes_88_90: u16`.
    tag_and_tail: AtomicU64,
    /// Bytes `0..88` of the 90-byte packed NVFP4 PLE record as native-endian words.
    words: [AtomicU64; EXPECTED_RECORD_U64_WORDS],
}

const _: () = assert!(std::mem::size_of::<ParkedSlot>() == 104);

impl ParkedSlot {
    #[inline]
    fn empty() -> Self {
        Self {
            seq: AtomicU64::new(0),
            tag_and_tail: AtomicU64::new(0),
            words: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

#[inline]
fn encode_parked_key_hi(shard_idx: usize, row: usize) -> Option<u64> {
    let shard = u16::try_from(shard_idx).ok()?;
    let row = u32::try_from(row).ok()?;
    Some((u64::from(shard) << 48) | (u64::from(row) << 16))
}

#[inline]
fn parked_slot_index(key_hi: u64, mask: usize) -> usize {
    // SplitMix64 finalizer over the 48-bit packed `(shard << 32) | row` key.
    let mut x = key_hi >> 16;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    (x as usize) & mask
}

const fn parked_cache_slots_for_mb(mb: usize) -> usize {
    if mb == 0 {
        return 0;
    }
    let budget_bytes = mb.saturating_mul(1024 * 1024);
    let max_slots = budget_bytes / std::mem::size_of::<ParkedSlot>();
    if max_slots == 0 {
        return 0;
    }
    1usize << (usize::BITS - 1 - max_slots.leading_zeros())
}

/// Resolve the power-of-two slot count for [`ParkedPleCache`] from
/// `ATLAS_PLE_PARKED_CACHE_MB` (`"0"` disables allocation; unset/unparseable
/// defaults to [`DEFAULT_PARKED_CACHE_MB`]).
fn resolve_parked_cache_slots(raw: Option<&str>) -> usize {
    let mb = raw
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_PARKED_CACHE_MB)
        .min(MAX_PARKED_CACHE_MB);
    parked_cache_slots_for_mb(mb)
}

/// Fixed-footprint, lock-free direct-mapped atomic cache for 90-byte PLE records.
///
/// Each 104-byte slot uses a 64-bit seqlock (`seq`) and fully atomic `u64`
/// payload words so concurrent verify/decode readers and background warm-up
/// writers never block on a mutex and never observe torn reads. All storage is
/// allocated once at initialization; eviction overwrites the hashed slot in
/// place with zero heap growth.
pub struct ParkedPleCache {
    slots: Box<[ParkedSlot]>,
    mask: usize,
    shard_rows: Box<[usize]>,
    shard_scale2: Box<[f32]>,
    hits: AtomicU64,
    misses: AtomicU64,
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ParkedPleCache>();
};

impl std::fmt::Debug for ParkedPleCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (hits, misses) = self.stats();
        f.debug_struct("ParkedPleCache")
            .field("slots", &self.slots.len())
            .field("bytes", &self.byte_footprint())
            .field("shards", &self.shard_rows.len())
            .field("hits", &hits)
            .field("misses", &misses)
            .finish()
    }
}

impl ParkedPleCache {
    /// Allocate a fixed-footprint parked cache with `num_slots` rounded down
    /// to a power of two (`0` allocates zero slot memory).
    pub fn with_slots(num_slots: usize, shard_rows: Vec<usize>, shard_scale2: Vec<f32>) -> Self {
        assert_eq!(
            shard_rows.len(),
            shard_scale2.len(),
            "PLE parked cache shard metadata length mismatch"
        );
        let slot_count = if num_slots == 0 {
            0
        } else {
            1usize << (usize::BITS - 1 - num_slots.leading_zeros())
        };
        let slots = if slot_count == 0 {
            Box::new([])
        } else {
            let mut vec = Vec::with_capacity(slot_count);
            vec.resize_with(slot_count, ParkedSlot::empty);
            vec.into_boxed_slice()
        };
        Self {
            slots,
            mask: slot_count.saturating_sub(1),
            shard_rows: shard_rows.into_boxed_slice(),
            shard_scale2: shard_scale2.into_boxed_slice(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Construct a parked cache sized from `ATLAS_PLE_PARKED_CACHE_MB`.
    pub fn from_env(shard_rows: Vec<usize>, shard_scale2: Vec<f32>) -> Self {
        let slots =
            resolve_parked_cache_slots(std::env::var(PLE_PARKED_CACHE_MB_ENV).ok().as_deref());
        Self::with_slots(slots, shard_rows, shard_scale2)
    }

    /// Returns `true` when at least one cache slot is allocated.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        !self.slots.is_empty()
    }

    /// Fixed number of direct-mapped slots allocated by this cache.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Exact slot-table heap footprint in bytes (`slot_count() * 104`).
    #[inline]
    pub fn byte_footprint(&self) -> usize {
        self.slots.len() * std::mem::size_of::<ParkedSlot>()
    }

    /// Cumulative `(hits, misses)` recorded by batch lookups against this cache.
    #[inline]
    pub fn stats(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }

    /// Fast lock-free probe checking whether `(shard_idx, row)` is currently
    /// resident without copying its 90-byte payload.
    #[inline]
    pub fn contains(&self, shard_idx: usize, row: usize) -> bool {
        if self.slots.is_empty() {
            return false;
        }
        let Some(key_hi) = encode_parked_key_hi(shard_idx, row) else {
            return false;
        };
        let slot = &self.slots[parked_slot_index(key_hi, self.mask)];
        let seq = slot.seq.load(Ordering::Acquire);
        if seq == 0 || (seq & 1) != 0 {
            return false;
        }
        (slot.tag_and_tail.load(Ordering::Relaxed) & PARKED_KEY_HI_MASK) == key_hi
    }

    /// Lock-free single-record lookup. Returns `None` on miss or if a
    /// concurrent writer is actively updating the slot.
    #[inline]
    pub fn get(&self, shard_idx: usize, row: usize) -> Option<[u8; EXPECTED_RECORD_BYTES]> {
        if self.slots.is_empty() {
            return None;
        }
        let key_hi = encode_parked_key_hi(shard_idx, row)?;
        let slot = &self.slots[parked_slot_index(key_hi, self.mask)];
        for attempt in 0..2 {
            let seq1 = slot.seq.load(Ordering::Acquire);
            if seq1 == 0 {
                return None;
            }
            if (seq1 & 1) != 0 {
                if attempt == 0 {
                    std::hint::spin_loop();
                    continue;
                }
                return None;
            }
            let tag_and_tail = slot.tag_and_tail.load(Ordering::Relaxed);
            if (tag_and_tail & PARKED_KEY_HI_MASK) != key_hi {
                return None;
            }
            let mut record = [0u8; EXPECTED_RECORD_BYTES];
            for (i, word) in slot.words.iter().enumerate() {
                let base = i * 8;
                let bytes = word.load(Ordering::Relaxed).to_ne_bytes();
                record[base..base + 8].copy_from_slice(&bytes);
            }
            let tail = (tag_and_tail as u16).to_ne_bytes();
            record[88..90].copy_from_slice(&tail);
            // Pairs with the writer's Release fence between publishing odd
            // `busy` and writing payload words so a torn read forces `seq2 != seq1`.
            std::sync::atomic::fence(Ordering::Acquire);
            let seq2 = slot.seq.load(Ordering::Relaxed);
            if seq1 == seq2 {
                return Some(record);
            }
            if attempt == 0 {
                std::hint::spin_loop();
            }
        }
        None
    }

    /// Lock-free direct-mapped insert. If another writer is concurrently
    /// updating the same slot, skips without blocking.
    #[inline]
    pub fn insert(&self, shard_idx: usize, row: usize, record: &[u8; EXPECTED_RECORD_BYTES]) {
        if self.slots.is_empty() {
            return;
        }
        let Some(key_hi) = encode_parked_key_hi(shard_idx, row) else {
            return;
        };
        let slot = &self.slots[parked_slot_index(key_hi, self.mask)];
        let s = slot.seq.load(Ordering::Relaxed);
        if (s & 1) != 0 {
            return;
        }
        let busy = s.wrapping_add(1);
        // Acquire synchronizes with the previous writer's Release store so our
        // payload stores happen-after theirs in modification order.
        if slot
            .seq
            .compare_exchange(s, busy, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        // Release fence between publishing odd `busy` and mutating payload words
        // pairs with the reader's Acquire fence before loading `seq2`.
        std::sync::atomic::fence(Ordering::Release);

        let tail = u16::from_ne_bytes([record[88], record[89]]);
        slot.tag_and_tail
            .store(key_hi | u64::from(tail), Ordering::Relaxed);
        for (i, word) in slot.words.iter().enumerate() {
            let base = i * 8;
            let val = u64::from_ne_bytes([
                record[base],
                record[base + 1],
                record[base + 2],
                record[base + 3],
                record[base + 4],
                record[base + 5],
                record[base + 6],
                record[base + 7],
            ]);
            word.store(val, Ordering::Relaxed);
        }

        let next_even = s.wrapping_add(2);
        let published = if next_even == 0 { 2 } else { next_even };
        slot.seq.store(published, Ordering::Release);
    }

    /// Lock-free single-row lookup returning [`PleNvfp4Row`] (record + `scale2`)
    /// on hit, `Ok(None)` on miss/disabled, or `Err` if `(shard_idx, row)` is
    /// out of range.
    pub fn get_row(&self, shard_idx: usize, row: usize) -> Result<Option<PleNvfp4Row>> {
        let shard_rows = *self
            .shard_rows
            .get(shard_idx)
            .with_context(|| format!("PLE shard {shard_idx} out of range"))?;
        ensure!(
            row < shard_rows,
            "PLE row {row} out of range for shard {shard_idx}"
        );
        Ok(self.get(shard_idx, row).map(|record| PleNvfp4Row {
            record,
            scale2: self.shard_scale2[shard_idx],
        }))
    }

    /// Lock-free batch lookup returning `Ok(Some(rows))` when every selection
    /// hits in cache, or `Ok(None)` if disabled, empty, or any selection misses.
    pub fn try_read_all(&self, selections: &[(usize, usize)]) -> Result<Option<Vec<PleNvfp4Row>>> {
        if !self.is_enabled() || selections.is_empty() {
            return Ok(None);
        }
        // Probe first so partial-hit batches do not allocate an output Vec.
        for &(shard_idx, row) in selections {
            let shard_rows = *self
                .shard_rows
                .get(shard_idx)
                .with_context(|| format!("PLE shard {shard_idx} out of range"))?;
            ensure!(
                row < shard_rows,
                "PLE row {row} out of range for shard {shard_idx}"
            );
            if !self.contains(shard_idx, row) {
                return Ok(None);
            }
        }
        let mut out = Vec::with_capacity(selections.len());
        for &(shard_idx, row) in selections {
            let Some(record) = self.get(shard_idx, row) else {
                return Ok(None);
            };
            out.push(PleNvfp4Row {
                record,
                scale2: self.shard_scale2[shard_idx],
            });
        }
        self.hits
            .fetch_add(selections.len() as u64, Ordering::Relaxed);
        Ok(Some(out))
    }

    /// Lock-free direct-to-staging copy: writes packed 90-byte records into
    /// `records_out` and little-endian `f32` scales into `scales_out`. Returns
    /// `Ok(true)` if every selection hit (zero mutex waits, zero syscalls, zero
    /// heap allocations), or `Ok(false)` on first miss.
    pub fn try_fill_staging(
        &self,
        selections: &[(usize, usize)],
        records_out: &mut [u8],
        scales_out: &mut [u8],
    ) -> Result<bool> {
        if !self.is_enabled() || selections.is_empty() {
            return Ok(false);
        }
        ensure!(
            records_out.len() == selections.len() * EXPECTED_RECORD_BYTES,
            "PLE parked cache staging record buffer extent mismatch"
        );
        ensure!(
            scales_out.len() == selections.len() * std::mem::size_of::<f32>(),
            "PLE parked cache staging scale buffer extent mismatch"
        );
        for (index, &(shard_idx, row)) in selections.iter().enumerate() {
            let shard_rows = *self
                .shard_rows
                .get(shard_idx)
                .with_context(|| format!("PLE shard {shard_idx} out of range"))?;
            ensure!(
                row < shard_rows,
                "PLE row {row} out of range for shard {shard_idx}"
            );
            let Some(record) = self.get(shard_idx, row) else {
                return Ok(false);
            };
            let rec_start = index * EXPECTED_RECORD_BYTES;
            records_out[rec_start..rec_start + EXPECTED_RECORD_BYTES].copy_from_slice(&record);
            let scale_start = index * std::mem::size_of::<f32>();
            scales_out[scale_start..scale_start + std::mem::size_of::<f32>()]
                .copy_from_slice(&self.shard_scale2[shard_idx].to_le_bytes());
        }
        self.hits
            .fetch_add(selections.len() as u64, Ordering::Relaxed);
        Ok(true)
    }
}

#[inline]
fn build_read_sqe(
    fixed_buffers: bool,
    fd: RawFd,
    ptr: *mut u8,
    len: u32,
    buf_index: u16,
    offset: u64,
    user_data: u64,
) -> io_uring::squeue::Entry {
    if fixed_buffers {
        opcode::ReadFixed::new(types::Fd(fd), ptr, len, buf_index)
            .offset(offset)
            .build()
            .user_data(user_data)
    } else {
        opcode::Read::new(types::Fd(fd), ptr, len)
            .offset(offset)
            .build()
            .user_data(user_data)
    }
}

fn pread_exact_aligned(fd: RawFd, buf: &mut [u8], offset: u64) -> Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let cur_offset = offset
            .checked_add(done as u64)
            .context("PLE pread64 offset overflow")?;
        let cur_off_t = libc::off_t::try_from(cur_offset)
            .with_context(|| format!("PLE pread64 offset {cur_offset} exceeds off_t::MAX"))?;
        let remaining = &mut buf[done..];
        // SAFETY: `fd` is an open file descriptor owned by `PleOffloadReader`, and
        // `remaining` is a valid writable slice of `remaining.len()` bytes.
        let n = unsafe {
            libc::pread(
                fd,
                remaining.as_mut_ptr().cast(),
                remaining.len(),
                cur_off_t,
            )
        };
        if n < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            bail!(
                "PLE pread64 at offset {cur_offset} ({} B) failed: {err}",
                remaining.len()
            );
        }
        if n == 0 {
            bail!(
                "PLE pread64 at offset {offset} reached unexpected EOF after {done}/{} B",
                buf.len()
            );
        }
        done += n as usize;
    }
    Ok(())
}

/// RAII drop guard ensuring an `io_uring` with in-flight SQEs targeting fixed
/// or pinned page buffers is synchronously cancelled and drained before the
/// caller returns or unwinds on panic.
struct InFlightRingGuard<'a> {
    ring: &'a mut IoUring,
    in_flight: usize,
    operation: &'static str,
}

impl<'a> InFlightRingGuard<'a> {
    fn new(ring: &'a mut IoUring, operation: &'static str) -> Self {
        Self {
            ring,
            in_flight: 0,
            operation,
        }
    }

    fn ring(&mut self) -> &mut IoUring {
        self.ring
    }

    fn in_flight(&self) -> usize {
        self.in_flight
    }

    fn add_in_flight(&mut self, n: usize) {
        self.in_flight += n;
    }

    fn dec_in_flight(&mut self) {
        self.in_flight = self.in_flight.saturating_sub(1);
    }

    fn cancel_and_drain(&mut self, operation: &str) {
        crate::io_wait::cancel_and_drain_or_exit(self.ring, operation);
        self.in_flight = 0;
    }
}

impl Drop for InFlightRingGuard<'_> {
    fn drop(&mut self) {
        if self.in_flight > 0 {
            crate::io_wait::cancel_and_drain_or_exit(self.ring, self.operation);
            self.in_flight = 0;
        }
    }
}

/// One extra io_uring (or pread64 lane) and its page buffers. Field order matters:
/// the ring (and its buffer registration) drops before the pages.
struct ReadRing {
    ring: Option<IoUring>,
    pages: Vec<AlignedPage>,
    fixed_buffers: bool,
}

impl ReadRing {
    fn open(queue_depth: usize) -> Result<Self> {
        let mut pages: Vec<AlignedPage> = (0..queue_depth).map(|_| AlignedPage::new()).collect();
        let iovecs: Vec<libc::iovec> = pages
            .iter_mut()
            .map(|page| libc::iovec {
                iov_base: page.ptr.cast(),
                iov_len: EXPECTED_PAGE_BYTES,
            })
            .collect();
        let (ring, fixed_buffers) = match IoUring::new(queue_depth as u32) {
            Ok(ring) if ring.params().is_feature_ext_arg() => {
                // SAFETY: the buffers live as long as the ring (dropped after it).
                let fixed_buffers = match unsafe { ring.submitter().register_buffers(&iovecs) } {
                    Ok(()) => true,
                    Err(err) => {
                        tracing::warn!(
                            "PLE extra io_uring register_buffers failed ({err}); falling back to unregistered O_DIRECT reads"
                        );
                        false
                    }
                };
                (Some(ring), fixed_buffers)
            }
            _ => (None, false),
        };
        Ok(Self {
            ring,
            pages,
            fixed_buffers,
        })
    }
}

/// What a ring's thread reports to the reading thread.
enum LaneMsg {
    /// Rows of completed reads: (request index, row).
    Rows(Vec<(u32, PleNvfp4Row)>),
    /// The ring finished (its result; on error every read it had in flight was
    /// completed or cancelled first).
    Done(Result<()>),
}

/// Apply `ATLAS_PLE_READ_AFFINITY` once when a persistent `ple-ring-*` worker
/// thread starts, logging the process receipt on first resolution.
fn apply_ple_worker_affinity() {
    let (plan, first) = crate::host_affinity::process_plan();
    if first && let Some(receipt) = plan.receipt() {
        if matches!(plan, crate::host_affinity::Plan::Refused { .. }) {
            tracing::warn!(target: "atlas::qwen4_ple", "{receipt}");
        } else {
            tracing::info!(target: "atlas::qwen4_ple", "{receipt}");
        }
    }
    if let Err(error) = crate::host_affinity::apply_to_current_thread(plan) {
        tracing::warn!(
            target: "atlas::qwen4_ple",
            "PLE_READ_AFFINITY_REFUSED selector={} reason=pin_failed error={error} (ring worker runs unpinned)",
            crate::host_affinity::SELECTOR
        );
    }
}

struct RingJob {
    plan: Arc<crate::ple_plan::ReadPlan>,
    rings: usize,
    deadline: Instant,
    io_timeout: Duration,
    abort: Arc<AtomicBool>,
    tx: mpsc::Sender<LaneMsg>,
    /// Present only for worker `0`, which borrows `PleOffloadReader`'s primary
    /// `io_uring` + page buffers for the duration of one multi-ring call and
    /// returns them via [`RingWorkerShared::returned_lane`] before signaling `Done`.
    borrowed_lane: Option<ReadRing>,
}

struct RingWorkerState {
    job: Option<RingJob>,
    closed: bool,
}

struct RingWorkerShared {
    state: Mutex<RingWorkerState>,
    cvar: Condvar,
    returned_lane: Mutex<Option<ReadRing>>,
}

struct RingWorkerExitGuard {
    shared: Arc<RingWorkerShared>,
}

impl Drop for RingWorkerExitGuard {
    fn drop(&mut self) {
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        state.closed = true;
        drop(state);
        self.shared.cvar.notify_all();
    }
}

struct RingWorker {
    shared: Arc<RingWorkerShared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for RingWorker {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            state.closed = true;
        }
        self.shared.cvar.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn spawn_ring_worker(
    ring_index: usize,
    mut owned_lane: Option<ReadRing>,
    shards: Arc<[Shard]>,
) -> Result<RingWorker> {
    let shared = Arc::new(RingWorkerShared {
        state: Mutex::new(RingWorkerState {
            job: None,
            closed: false,
        }),
        cvar: Condvar::new(),
        returned_lane: Mutex::new(None),
    });
    let worker_shared = Arc::clone(&shared);
    let thread = std::thread::Builder::new()
        .name(format!("ple-ring-{ring_index}"))
        .spawn(move || {
            apply_ple_worker_affinity();
            let _exit_guard = RingWorkerExitGuard {
                shared: Arc::clone(&worker_shared),
            };
            loop {
                let job = {
                    let mut state = worker_shared
                        .state
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    loop {
                        if let Some(job) = state.job.take() {
                            break Some(job);
                        }
                        if state.closed {
                            break None;
                        }
                        state = worker_shared
                            .cvar
                            .wait(state)
                            .unwrap_or_else(|e| e.into_inner());
                    }
                };
                let Some(mut job) = job else {
                    break;
                };
                if let Some(borrowed) = job.borrowed_lane.take() {
                    // Ensures the borrowed primary ring + buffers are returned
                    // before `Done` (and even on unwind) so the reader never loses
                    // its primary ring.
                    struct ReturnLaneGuard<'a> {
                        lane: Option<ReadRing>,
                        slot: &'a Mutex<Option<ReadRing>>,
                    }
                    impl Drop for ReturnLaneGuard<'_> {
                        fn drop(&mut self) {
                            if let Some(lane) = self.lane.take() {
                                *self.slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(lane);
                            }
                        }
                    }
                    let mut return_guard = ReturnLaneGuard {
                        lane: Some(borrowed),
                        slot: &worker_shared.returned_lane,
                    };
                    let result = {
                        let lane = return_guard.lane.as_mut().expect("borrowed lane present");
                        drain_ring(
                            lane.ring.as_mut(),
                            lane.fixed_buffers,
                            &mut lane.pages,
                            &shards,
                            &job.plan,
                            ring_index,
                            job.rings,
                            job.deadline,
                            job.io_timeout,
                            &job.abort,
                            &job.tx,
                        )
                    };
                    drop(return_guard);
                    if result.is_err() {
                        job.abort.store(true, Ordering::Release);
                    }
                    let _ = job.tx.send(LaneMsg::Done(result));
                } else {
                    let result = match owned_lane.as_mut() {
                        Some(lane) => drain_ring(
                            lane.ring.as_mut(),
                            lane.fixed_buffers,
                            &mut lane.pages,
                            &shards,
                            &job.plan,
                            ring_index,
                            job.rings,
                            job.deadline,
                            job.io_timeout,
                            &job.abort,
                            &job.tx,
                        ),
                        None => Err(anyhow::anyhow!(
                            "PLE ring worker {ring_index} missing its ring lane"
                        )),
                    };
                    if result.is_err() {
                        job.abort.store(true, Ordering::Release);
                    }
                    let _ = job.tx.send(LaneMsg::Done(result));
                }
            }
        })
        .with_context(|| format!("spawn persistent PLE ring worker {ring_index}"))?;
    Ok(RingWorker {
        shared,
        thread: Some(thread),
    })
}

/// Persistent `ple-ring-0..N` worker pool stored on [`PleOffloadReader`].
///
/// Workers `1..N` permanently own their extra [`ReadRing`]s (avoiding per-call
/// OS thread spawns/joins on every multi-ring verify or prefill read), while
/// worker `0` borrows the reader's primary ring + buffers for the duration of
/// each multi-ring call and restores them before returning `Done`.
struct RingWorkerPool {
    workers: Vec<RingWorker>,
}

impl RingWorkerPool {
    fn new(extra_rings: Vec<ReadRing>, shards: Arc<[Shard]>) -> Result<Self> {
        let total_rings = 1 + extra_rings.len();
        let mut workers = Vec::with_capacity(total_rings);
        workers.push(spawn_ring_worker(0, None, Arc::clone(&shards))?);
        for (offset, extra) in extra_rings.into_iter().enumerate() {
            workers.push(spawn_ring_worker(
                offset + 1,
                Some(extra),
                Arc::clone(&shards),
            )?);
        }
        Ok(Self { workers })
    }

    #[inline]
    fn ring_count(&self) -> usize {
        self.workers.len()
    }

    fn dispatch(
        &self,
        primary_lane: ReadRing,
        plan: Arc<crate::ple_plan::ReadPlan>,
        active_rings: usize,
        deadline: Instant,
        io_timeout: Duration,
        abort: Arc<AtomicBool>,
    ) -> (mpsc::Receiver<LaneMsg>, usize, Option<anyhow::Error>) {
        let (tx, rx) = mpsc::channel::<LaneMsg>();
        let mut primary = Some(primary_lane);
        let mut dispatched = 0usize;
        let mut dispatch_err = None;
        for (ring_index, worker) in self.workers.iter().take(active_rings).enumerate() {
            let job = RingJob {
                plan: Arc::clone(&plan),
                rings: active_rings,
                deadline,
                io_timeout,
                abort: Arc::clone(&abort),
                tx: tx.clone(),
                borrowed_lane: if ring_index == 0 {
                    primary.take()
                } else {
                    None
                },
            };
            let mut state = worker
                .shared
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if state.closed {
                if let Some(returned) = job.borrowed_lane {
                    *worker
                        .shared
                        .returned_lane
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = Some(returned);
                }
                drop(state);
                abort.store(true, Ordering::Release);
                dispatch_err = Some(anyhow::anyhow!(
                    "PLE ring worker {ring_index} terminated unexpectedly"
                ));
                break;
            }
            debug_assert!(
                state.job.is_none(),
                "PLE ring worker {ring_index} still held an unconsumed job"
            );
            state.job = Some(job);
            drop(state);
            worker.shared.cvar.notify_one();
            dispatched += 1;
        }
        drop(tx);
        (rx, dispatched, dispatch_err)
    }

    fn take_returned_primary(&self) -> Option<ReadRing> {
        self.workers
            .first()?
            .shared
            .returned_lane
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    }

    #[cfg(test)]
    fn worker_thread_ids(&self) -> Vec<std::thread::ThreadId> {
        self.workers
            .iter()
            .filter_map(|w| w.thread.as_ref().map(|t| t.thread().id()))
            .collect()
    }
}

impl Drop for RingWorkerPool {
    fn drop(&mut self) {
        for worker in &self.workers {
            let mut state = worker
                .shared
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            state.closed = true;
            drop(state);
            worker.shared.cvar.notify_all();
        }
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

/// Cap on persistent `ple-pread-*` worker threads (`64 × 8 KiB = 512 KiB`
/// buffers). Covers 100% of verify/decode batches without per-step OS thread
/// spawn/join or ARM64 `mmap`/`munmap` TLB shootdowns; huge prefill chunks that
/// request `> 64` lanes still fan out across `pread_pages`.
const MAX_PERSISTENT_PREAD_WORKERS: usize = 64;

struct PreadJob {
    plan: Arc<crate::ple_plan::ReadPlan>,
    tile_size: usize,
    next_tile: Arc<AtomicUsize>,
    deadline: Instant,
    io_timeout: Duration,
    abort: Arc<AtomicBool>,
    tx: mpsc::Sender<LaneMsg>,
}

struct PreadWorkerState {
    job: Option<PreadJob>,
    closed: bool,
}

struct PreadWorkerShared {
    state: Mutex<PreadWorkerState>,
    cvar: Condvar,
}

struct PreadWorkerExitGuard {
    shared: Arc<PreadWorkerShared>,
}

impl Drop for PreadWorkerExitGuard {
    fn drop(&mut self) {
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        state.closed = true;
        drop(state);
        self.shared.cvar.notify_all();
    }
}

struct PreadWorker {
    shared: Arc<PreadWorkerShared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for PreadWorker {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            state.closed = true;
        }
        self.shared.cvar.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn spawn_pread_worker(lane_index: usize, shards: Arc<[Shard]>) -> Result<PreadWorker> {
    let shared = Arc::new(PreadWorkerShared {
        state: Mutex::new(PreadWorkerState {
            job: None,
            closed: false,
        }),
        cvar: Condvar::new(),
    });
    let worker_shared = Arc::clone(&shared);
    let thread = std::thread::Builder::new()
        .name(format!("ple-pread-{lane_index}"))
        .spawn(move || {
            let _exit_guard = PreadWorkerExitGuard {
                shared: Arc::clone(&worker_shared),
            };
            apply_ple_worker_affinity();
            let mut page = AlignedPage::new();
            loop {
                let job = {
                    let mut state = worker_shared
                        .state
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    loop {
                        if let Some(job) = state.job.take() {
                            break Some(job);
                        }
                        if state.closed {
                            break None;
                        }
                        state = worker_shared
                            .cvar
                            .wait(state)
                            .unwrap_or_else(|e| e.into_inner());
                    }
                };
                let Some(job) = job else {
                    return;
                };
                let result = drain_pread_lane(
                    &mut page,
                    &shards,
                    &job.plan,
                    job.tile_size,
                    &job.next_tile,
                    job.deadline,
                    job.io_timeout,
                    &job.abort,
                    &job.tx,
                );
                if result.is_err() {
                    job.abort.store(true, Ordering::Release);
                }
                let _ = job.tx.send(LaneMsg::Done(result));
            }
        })
        .with_context(|| format!("spawn persistent PLE pread worker {lane_index}"))?;
    Ok(PreadWorker {
        shared,
        thread: Some(thread),
    })
}

/// Persistent `ple-pread-0..N` worker pool stored on [`PleOffloadReader`] when
/// `io_uring` is unavailable. Eliminates per-step thread creation/teardown and
/// stack `mmap`/`munmap` TLB shootdowns across decode/verify batches.
struct PreadWorkerPool {
    workers: Vec<PreadWorker>,
}

impl PreadWorkerPool {
    fn new(worker_count: usize, shards: Arc<[Shard]>) -> Result<Self> {
        let mut workers = Vec::with_capacity(worker_count);
        for lane in 0..worker_count {
            workers.push(spawn_pread_worker(lane, Arc::clone(&shards))?);
        }
        Ok(Self { workers })
    }

    #[inline]
    fn worker_count(&self) -> usize {
        self.workers.len()
    }

    fn dispatch(
        &self,
        plan: Arc<crate::ple_plan::ReadPlan>,
        active_lanes: usize,
        tile_size: usize,
        deadline: Instant,
        io_timeout: Duration,
        abort: Arc<AtomicBool>,
    ) -> (mpsc::Receiver<LaneMsg>, usize, Option<anyhow::Error>) {
        let (tx, rx) = mpsc::channel::<LaneMsg>();
        let next_tile = Arc::new(AtomicUsize::new(0));
        let mut dispatched = 0usize;
        let mut dispatch_err = None;
        for (lane, worker) in self.workers.iter().take(active_lanes).enumerate() {
            let job = PreadJob {
                plan: Arc::clone(&plan),
                tile_size,
                next_tile: Arc::clone(&next_tile),
                deadline,
                io_timeout,
                abort: Arc::clone(&abort),
                tx: tx.clone(),
            };
            let mut state = worker
                .shared
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if state.closed {
                drop(state);
                abort.store(true, Ordering::Release);
                dispatch_err = Some(anyhow::anyhow!(
                    "PLE pread worker {lane} terminated unexpectedly"
                ));
                break;
            }
            debug_assert!(
                state.job.is_none(),
                "PLE pread worker {lane} still held an unconsumed job"
            );
            state.job = Some(job);
            drop(state);
            worker.shared.cvar.notify_one();
            dispatched += 1;
        }
        drop(tx);
        (rx, dispatched, dispatch_err)
    }

    #[cfg(test)]
    fn worker_thread_ids(&self) -> Vec<std::thread::ThreadId> {
        self.workers
            .iter()
            .filter_map(|w| w.thread.as_ref().map(|t| t.thread().id()))
            .collect()
    }
}

impl Drop for PreadWorkerPool {
    fn drop(&mut self) {
        for worker in &self.workers {
            let mut state = worker
                .shared
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            state.closed = true;
            drop(state);
            worker.shared.cvar.notify_all();
        }
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

const RING_TILE: usize = 64;

/// Minimum deduplicated verify misses before `io_uring` fans out across
/// configured rings (`4` tokens × `16` heads = `64` selections). Keeps
/// [`MIN_PREAD_LANE_SELECTIONS`] (`128`) intact for the no-`io_uring` pread path.
const VERIFY_MIN_RING_SELECTIONS: usize = 64;

/// Tile size for [`drain_ring`]. Uses [`RING_TILE`] (`64`) once `reads` fills
/// every ring with a full tile (`reads >= rings * 64`, e.g. `>= 256` on 4 rings
/// and all prefill chunks), and divides smaller verify batches (`64..256`)
/// evenly across all configured rings so every ring shares the NVMe I/O.
#[inline]
fn io_uring_tile_size(reads: usize, rings: usize) -> usize {
    let ring_count = rings.max(1);
    if reads >= ring_count * RING_TILE {
        RING_TILE
    } else {
        reads.div_ceil(ring_count).clamp(1, RING_TILE)
    }
}

#[inline]
fn ring_tile_of(read: usize, rings: usize, tile: usize) -> usize {
    (read / tile.max(1)) % rings.max(1)
}

#[inline]
fn resolve_io_uring_lanes(
    rings: usize,
    available_rings: usize,
    selections: usize,
    min_ring_selections: usize,
) -> Option<usize> {
    (rings > 1 && rings <= available_rings && selections >= min_ring_selections).then_some(rings)
}

/// Drain the plan's reads in `tile_size` tiles (`(i / tile_size) % rings == ring_index`)
/// through one ring with the original loop's discipline: deadline, cancel-and-drain on any
/// error, tag and length checks, a buffer reused only after its read completed
/// and its records were copied. Rows go to `tx` once per completion batch.
#[allow(clippy::too_many_arguments)]
fn drain_ring(
    ring: Option<&mut IoUring>,
    fixed_buffers: bool,
    pages: &mut [AlignedPage],
    shards: &[Shard],
    plan: &crate::ple_plan::ReadPlan,
    ring_index: usize,
    rings: usize,
    deadline: Instant,
    io_timeout: Duration,
    abort: &std::sync::atomic::AtomicBool,
    tx: &std::sync::mpsc::Sender<LaneMsg>,
) -> Result<()> {
    let reads = plan.reads();
    let queue_depth = pages.len();
    let ring_count = rings.max(1);
    let tile_size = io_uring_tile_size(reads, ring_count);
    let mut my_reads = Vec::with_capacity(reads / ring_count + tile_size);
    let mut tile_start = ring_index * tile_size;
    while tile_start < reads {
        let tile_end = (tile_start + tile_size).min(reads);
        let base = my_reads.len();
        my_reads.extend(tile_start..tile_end);
        my_reads[base..].sort_unstable_by_key(|&i| (plan.keys[i].0, plan.span(i).file_offset));
        tile_start += ring_count * tile_size;
    }
    let mine = my_reads.len();
    if ring.is_none() {
        for tile in my_reads.chunks(tile_size) {
            if abort.load(Ordering::Acquire) {
                bail!("PLE ring {ring_index} stopped: another PLE read ring failed");
            }
            if Instant::now() >= deadline {
                bail!("PLE ring read exceeded its {io_timeout:?} operation deadline");
            }
            let mut rows = Vec::new();
            for &index in tile {
                let waiters = plan.waiters_of(index);
                if waiters.is_empty() {
                    continue;
                }
                let shard_idx = plan.keys[index].0;
                let span = plan.span(index);
                let fd = raw_fd(&shards[shard_idx].file);
                pread_exact_aligned(
                    fd,
                    pages[0].as_mut_slice(span.buffer_offset, span.len),
                    span.file_offset,
                )?;
                for waiter in waiters {
                    rows.push((
                        waiter.request_index,
                        PleNvfp4Row {
                            record: pages[0].record_at(waiter.record_offset as usize),
                            scale2: waiter.scale2,
                        },
                    ));
                }
            }
            if !rows.is_empty() && tx.send(LaneMsg::Rows(rows)).is_err() {
                bail!("PLE ring {ring_index}: the reading thread went away");
            }
        }
        return Ok(());
    }
    let mut guard = InFlightRingGuard::new(ring.unwrap(), "PLE ring read unwind/abort");
    let mut free: Vec<usize> = (0..queue_depth).rev().collect();
    let mut cursor = 0usize;
    let mut done = 0usize;
    let mut completed = vec![false; reads];
    let mut completions: Vec<(u64, i32)> = Vec::with_capacity(queue_depth);
    while done < mine {
        if abort.load(Ordering::Acquire) {
            guard.cancel_and_drain("PLE ring read: a peer ring failed");
            bail!("PLE ring {ring_index} stopped: another PLE read ring failed");
        }
        if Instant::now() >= deadline {
            guard.cancel_and_drain("PLE ring read operation deadline");
            bail!("PLE ring read exceeded its {io_timeout:?} operation deadline");
        }
        let mut submitted = 0usize;
        while cursor < mine {
            let next = my_reads[cursor];
            if plan.waiters_of(next).is_empty() {
                completed[next] = true;
                done += 1;
                cursor += 1;
                continue;
            }
            let Some(buffer) = free.pop() else { break };
            let shard_idx = plan.keys[next].0;
            let span = plan.span(next);
            let fd = raw_fd(&shards[shard_idx].file);
            let entry = build_read_sqe(
                fixed_buffers,
                fd,
                // SAFETY: `read_span` keeps buffer_offset + len within the
                // EXPECTED_PAGE_BYTES buffer at this index.
                unsafe { pages[buffer].ptr.add(span.buffer_offset) },
                span.len as u32,
                buffer as u16,
                span.file_offset,
                ((next as u64) << 16) | buffer as u64,
            );
            // SAFETY: `fd` is owned by `shards` for the duration of this scope, `buffer`
            // is uniquely claimed from `free` until its CQE arrives, and `guard` drains
            // all in-flight SQEs before `pages` or `shards` can be reused or dropped.
            if unsafe { guard.ring().submission().push(&entry) }.is_err() {
                eprintln!(
                    "fatal: PLE ring submission queue filled unexpectedly; exiting before queued fixed-buffer reads can escape"
                );
                std::process::exit(70);
            }
            cursor += 1;
            submitted += 1;
        }
        guard.add_in_flight(submitted);
        if done >= mine && guard.in_flight() == 0 {
            break;
        }
        ensure!(
            guard.in_flight() > 0,
            "PLE ring read stalled with nothing in flight"
        );
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .max(Duration::from_nanos(1));
        let want = guard.in_flight().min(16);
        crate::io_wait::submit_and_wait_bounded(guard.ring(), want, remaining, "PLE ring read")?;
        completions.clear();
        completions.extend(
            guard
                .ring()
                .completion()
                .map(|entry| (entry.user_data(), entry.result())),
        );
        let mut rows = Vec::new();
        for &(tag, result) in &completions {
            let (index, buffer) = ((tag >> 16) as usize, (tag & 0xffff) as usize);
            if index >= reads
                || ring_tile_of(index, rings, tile_size) != ring_index
                || buffer >= queue_depth
            {
                guard.cancel_and_drain("PLE ring read invalid completion tag");
                bail!("PLE io_uring returned invalid tag {tag}");
            }
            if completed[index] {
                guard.cancel_and_drain("PLE ring read duplicate completion");
                bail!("PLE io_uring returned duplicate completion for read {index}");
            }
            let expected = plan.span(index).len;
            if result != expected as i32 {
                guard.cancel_and_drain("PLE ring read completion error");
                bail!("PLE read {index} returned {result}, expected {expected}");
            }
            completed[index] = true;
            for waiter in plan.waiters_of(index) {
                rows.push((
                    waiter.request_index,
                    PleNvfp4Row {
                        record: pages[buffer].record_at(waiter.record_offset as usize),
                        scale2: waiter.scale2,
                    },
                ));
            }
            free.push(buffer);
            guard.dec_in_flight();
            done += 1;
        }
        if !rows.is_empty() && tx.send(LaneMsg::Rows(rows)).is_err() {
            guard.cancel_and_drain("PLE ring read: reader went away");
            bail!("PLE ring {ring_index}: the reading thread went away");
        }
    }
    if guard.in_flight() != 0 {
        let remaining_in_flight = guard.in_flight();
        guard.cancel_and_drain("PLE ring read incomplete retirement");
        bail!("PLE ring read ended with {remaining_in_flight} requests still in flight");
    }
    Ok(())
}

/// Tile size for [`drain_pread_lane`]. Preserves [`RING_TILE`] for both tiny
/// tests (`< MIN_PREAD_LANE_SELECTIONS`) and full prefill chunks (`>= 2048`),
/// while shrinking the tile for medium verify batches (`128..2048` misses) so
/// 16–32 pread lanes stay active instead of starving at 2–10 lanes.
fn pread_ring_tile_size(reads: usize) -> usize {
    if reads < MIN_PREAD_LANE_SELECTIONS {
        return RING_TILE;
    }
    (reads / 32).clamp(8, RING_TILE)
}

/// One no-io_uring lane: claims `tile_size` tiles of the plan from
/// `next_tile` (so however many lanes start, they complete the plan, in
/// first-use order) and reads each with `pread64` into its one page buffer.
/// Same deadline, abort and row hand-off as [`drain_ring`].
#[allow(clippy::too_many_arguments)]
fn drain_pread_lane(
    page: &mut AlignedPage,
    shards: &[Shard],
    plan: &crate::ple_plan::ReadPlan,
    tile_size: usize,
    next_tile: &std::sync::atomic::AtomicUsize,
    deadline: Instant,
    io_timeout: Duration,
    abort: &std::sync::atomic::AtomicBool,
    tx: &std::sync::mpsc::Sender<LaneMsg>,
) -> Result<()> {
    let reads = plan.reads();
    let tile_size = tile_size.max(1);
    let mut order = Vec::with_capacity(tile_size);
    loop {
        let start = next_tile
            .fetch_add(1, Ordering::Relaxed)
            .saturating_mul(tile_size);
        if start >= reads {
            return Ok(());
        }
        if abort.load(Ordering::Acquire) {
            bail!("PLE pread lane stopped: another PLE read lane failed");
        }
        if Instant::now() >= deadline {
            bail!("PLE pread lane exceeded its {io_timeout:?} operation deadline");
        }
        order.clear();
        order.extend(start..(start + tile_size).min(reads));
        order.sort_unstable_by_key(|&i| (plan.keys[i].0, plan.span(i).file_offset));
        let mut rows = Vec::new();
        for &index in &order {
            let waiters = plan.waiters_of(index);
            if waiters.is_empty() {
                continue;
            }
            let span = plan.span(index);
            pread_exact_aligned(
                raw_fd(&shards[plan.keys[index].0].file),
                page.as_mut_slice(span.buffer_offset, span.len),
                span.file_offset,
            )?;
            for waiter in waiters {
                rows.push((
                    waiter.request_index,
                    PleNvfp4Row {
                        record: page.record_at(waiter.record_offset as usize),
                        scale2: waiter.scale2,
                    },
                ));
            }
        }
        if !rows.is_empty() && tx.send(LaneMsg::Rows(rows)).is_err() {
            bail!("PLE pread lane: the reading thread went away");
        }
    }
}

const FAST_PLAN_SELECTOR: &str = "ATLAS_PLE_FAST_PLAN";
const SECTOR_READ_SELECTOR: &str = "ATLAS_PLE_SECTOR_READ";
const READ_TRIGGER: &str = "ATLAS_PLE_READ_TRIGGER";
const READ_STATS: &str = "ATLAS_PLE_READ_STATS";
const READ_RINGS: &str = "ATLAS_PLE_READ_RINGS";
/// Synchronous `pread64` lanes used when io_uring is unavailable (for example
/// Docker's default seccomp profile rejects `io_uring_setup` with `EPERM`).
const PREAD_LANES: &str = "ATLAS_PLE_PREAD_LANES";
/// Diagnostic: skip io_uring at open so the `pread64` fallback can be measured
/// on a host where io_uring works.
const FORCE_PREAD: &str = "ATLAS_PLE_FORCE_PREAD";
/// One synchronous lane keeps only one read in flight, so the fallback needs
/// many lanes to approach the queue depth the io_uring rings sustain: on GB10
/// a 16,384-token chunk (262,144 sector reads) took 1,547 / 667 / 315 / 192 ms
/// at 32 / 64 / 128 / 256 lanes vs 187 ms on 4 io_uring rings.
const DEFAULT_PREAD_LANES: usize = 256;
const MAX_PREAD_LANES: usize = 256;
/// Planned reads smaller than this stay on one lane (thread spawn cost).
const MIN_PREAD_LANE_SELECTIONS: usize = 128;
/// Decode batches with at least this many unique pages read on parallel lanes.
const MIN_PREAD_DECODE_PAGES: usize = 4;
/// Upper bound on lanes for one decode batch (16 rows per token).
const MAX_PREAD_DECODE_LANES: usize = 16;

fn parse_pread_lanes(value: Option<&str>) -> Result<usize> {
    match value {
        None | Some("") => Ok(DEFAULT_PREAD_LANES),
        Some(text) => {
            let lanes: usize = text
                .parse()
                .with_context(|| format!("{PREAD_LANES} must be an integer; got {text:?}"))?;
            ensure!(
                (1..=MAX_PREAD_LANES).contains(&lanes),
                "{PREAD_LANES} must be between 1 and {MAX_PREAD_LANES}; got {lanes}"
            );
            Ok(lanes)
        }
    }
}

/// Streamed prefill read selectors, resolved once at open.
///
/// - `ATLAS_PLE_FAST_PLAN=1`: plan with [`crate::ple_plan::plan_fast`] (the
///   same reads in the same order as the original loop) at the reader's own
///   granularity.
/// - `ATLAS_PLE_SECTOR_READ=1`: planned reads of the 512-byte sectors that
///   hold each record, admitted after a load-time O_DIRECT probe.
/// - `ATLAS_PLE_READ_TRIGGER=<path>`: both apply only while the file exists
///   (content `plan` or `legacy` narrows them; see `resolve_mode`), checked
///   per streamed read, so one load can serve an A/B/A.
/// - `ATLAS_PLE_READ_STATS=1` (diagnostic): one `QWEN4_PLE_READ_STATS` line
///   per streamed read, in every mode.
///
/// Both read selectors require the private page cache to be disabled
/// (`cache_pages == 0`); otherwise they are refused at open. Both `Direct` and
/// `PageCache` `io_mode`s are admitted when `cache_pages == 0` (with
/// `ATLAS_PLE_SECTOR_READ=1` additionally gated on the sector read probe).
#[derive(Clone, Debug)]
struct ReadPlanConfig {
    fast_plan: bool,
    sector_ready: bool,
    trigger: Option<PathBuf>,
    stats: bool,
    /// `ATLAS_PLE_READ_RINGS` (1 = off), admitted only with a planned mode.
    rings: usize,
}

fn read_switch(name: &str) -> Result<bool> {
    let value = std::env::var(name).ok();
    crate::ple_plan::parse_switch(name, value.as_deref()).map_err(anyhow::Error::msg)
}

/// Log `line` once per process.
fn log_once(line: String) {
    static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&line) {
        tracing::info!("{line}");
        seen.push(line);
    }
}

/// One 512-byte O_DIRECT read at offset 512 must return the bytes of the
/// 4 KiB read at offset 0 (sector-aligned direct I/O is admitted by the
/// filesystem and the device's logical block size).
fn probe_sector_reads(file: &File) -> std::result::Result<(), String> {
    let fd = raw_fd(file);
    let sector = crate::ple_plan::SECTOR_BYTES;
    let whole = AlignedPage::new();
    let part = AlignedPage::new();
    let half = EXPECTED_PAGE_BYTES / 2;
    // SAFETY: both buffers are EXPECTED_PAGE_BYTES long and 8 KiB aligned;
    // the reads are synchronous and the lengths are within the buffers.
    let got_whole = unsafe { libc::pread(fd, whole.ptr.cast(), half, 0) };
    if got_whole != half as isize {
        return Err(format!(
            "probe_4k_read_returned_{got_whole}_{}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: as above.
    let got_part = unsafe { libc::pread(fd, part.ptr.cast(), sector, sector as libc::off_t) };
    if got_part != sector as isize {
        return Err(format!(
            "probe_sector_read_returned_{got_part}_{}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: both ranges were just filled by the reads above.
    let same = unsafe {
        std::slice::from_raw_parts(whole.ptr.add(sector), sector)
            == std::slice::from_raw_parts(part.ptr, sector)
    };
    if same {
        Ok(())
    } else {
        Err("probe_sector_bytes_differ".to_string())
    }
}

impl ReadPlanConfig {
    fn from_env(io_mode: PleIoMode, cache_pages: usize, shards: &[Shard]) -> Result<Self> {
        let fast_plan = read_switch(FAST_PLAN_SELECTOR)?;
        let sector = read_switch(SECTOR_READ_SELECTOR)?;
        let stats = read_switch(READ_STATS)?;
        let rings_value = std::env::var(READ_RINGS).ok();
        let rings = crate::ple_plan::parse_rings(READ_RINGS, rings_value.as_deref())
            .map_err(anyhow::Error::msg)?;
        let trigger = std::env::var(READ_TRIGGER)
            .ok()
            .filter(|path| !path.is_empty())
            .map(PathBuf::from);
        let trigger_label = trigger
            .as_ref()
            .map_or_else(|| "none".to_string(), |path| path.display().to_string());
        let mut config = Self {
            fast_plan,
            sector_ready: false,
            trigger,
            stats,
            rings: 1,
        };
        if !(fast_plan || sector) {
            if rings > 1 {
                tracing::warn!(
                    "QWEN4_PLE_READ_RINGS_REFUSED selector={READ_RINGS} reason=needs_{FAST_PLAN_SELECTOR}_or_{SECTOR_READ_SELECTOR}"
                );
            }
            return Ok(config);
        }
        let refusal = if cache_pages != 0 {
            Some("private_page_cache")
        } else {
            let _ = io_mode;
            None
        };
        if let Some(reason) = refusal {
            for (on, name) in [
                (fast_plan, FAST_PLAN_SELECTOR),
                (sector, SECTOR_READ_SELECTOR),
                (rings > 1, READ_RINGS),
            ] {
                if on {
                    tracing::warn!("QWEN4_PLE_READ_PLAN_REFUSED selector={name} reason={reason}");
                }
            }
            config.fast_plan = false;
            return Ok(config);
        }
        if fast_plan {
            tracing::info!(
                "QWEN4_PLE_READ_PLAN_ENGAGED selector={FAST_PLAN_SELECTOR} planner=open_addressing trigger={trigger_label}"
            );
        }
        if sector {
            let probe = shards.first().map_or_else(
                || Err("no_shards".to_string()),
                |shard| probe_sector_reads(&shard.file),
            );
            match probe {
                Ok(()) => {
                    config.sector_ready = true;
                    tracing::info!(
                        "QWEN4_PLE_SECTOR_READ_ENGAGED selector={SECTOR_READ_SELECTOR} sector_bytes={} planner=open_addressing probe=ok trigger={trigger_label}",
                        crate::ple_plan::SECTOR_BYTES
                    );
                }
                Err(reason) => tracing::warn!(
                    "QWEN4_PLE_SECTOR_READ_REFUSED selector={SECTOR_READ_SELECTOR} reason={reason}"
                ),
            }
        }
        if rings > 1 && (config.fast_plan || config.sector_ready) {
            config.rings = rings;
            tracing::info!(
                "QWEN4_PLE_READ_RINGS_ENGAGED selector={READ_RINGS} rings={rings} threads=one_per_ring trigger={trigger_label}"
            );
        } else if rings > 1 {
            tracing::warn!(
                "QWEN4_PLE_READ_RINGS_REFUSED selector={READ_RINGS} reason=no_planned_mode_admitted"
            );
        }
        Ok(config)
    }

    /// The mode and ring count of one streamed read (one trigger-file read).
    fn mode_and_rings(&self) -> (crate::ple_plan::ReadMode, usize) {
        let trigger = self
            .trigger
            .as_ref()
            .map(|path| std::fs::read_to_string(path).ok());
        let trigger = trigger.as_ref().map(|content| content.as_deref());
        let mode = crate::ple_plan::resolve_mode(self.fast_plan, self.sector_ready, trigger);
        (
            mode,
            crate::ple_plan::resolve_rings(self.rings, mode, trigger),
        )
    }

    /// The mode of one streamed read (reads the trigger file when set).
    fn mode(&self) -> crate::ple_plan::ReadMode {
        let trigger = self
            .trigger
            .as_ref()
            .map(|path| std::fs::read_to_string(path).ok());
        crate::ple_plan::resolve_mode(
            self.fast_plan,
            self.sector_ready,
            trigger.as_ref().map(|content| content.as_deref()),
        )
    }
}

struct CachedPage {
    bytes: Box<[u8; EXPECTED_PAGE_BYTES]>,
    generation: u64,
}

fn selection_batch_ranges(total: usize, queue_depth: usize) -> Result<Vec<Range<usize>>> {
    ensure!(queue_depth > 0, "PLE selection queue depth is zero");
    let mut ranges = Vec::with_capacity(total.div_ceil(queue_depth));
    let mut start = 0;
    while start < total {
        let end = start.saturating_add(queue_depth).min(total);
        ranges.push(start..end);
        start = end;
    }
    Ok(ranges)
}

/// In-order completion tracking for a streamed read published in blocks of
/// `rows_per_block` selections (the last block may be shorter). A block is
/// ready once all of its rows are filled and every earlier block was ready.
#[derive(Debug)]
pub struct PleBlockTracker {
    rows_per_block: usize,
    total: usize,
    remaining: Vec<usize>,
    next: usize,
}

impl PleBlockTracker {
    pub fn new(total: usize, rows_per_block: usize) -> Result<Self> {
        ensure!(rows_per_block > 0, "PLE block size is zero");
        let blocks = total.div_ceil(rows_per_block);
        let remaining = (0..blocks)
            .map(|block| rows_per_block.min(total - block * rows_per_block))
            .collect();
        Ok(Self {
            rows_per_block,
            total,
            remaining,
            next: 0,
        })
    }

    pub fn blocks(&self) -> usize {
        self.remaining.len()
    }

    /// Selection range of `block`.
    pub fn range(&self, block: usize) -> Range<usize> {
        let start = block * self.rows_per_block;
        start..(start + self.rows_per_block).min(self.total)
    }

    /// Count one filled selection (each must be filled exactly once).
    pub fn fill(&mut self, request_index: usize) -> Result<()> {
        ensure!(
            request_index < self.total,
            "PLE selection {request_index} outside {} selections",
            self.total
        );
        let block = request_index / self.rows_per_block;
        let left = self
            .remaining
            .get_mut(block)
            .filter(|left| **left > 0)
            .with_context(|| {
                format!("PLE selection {request_index} filled twice or out of range")
            })?;
        *left -= 1;
        Ok(())
    }

    /// The next block to publish, in order, if it is complete.
    pub fn pop_ready(&mut self) -> Option<usize> {
        let block = self.next;
        (self.remaining.get(block) == Some(&0)).then(|| {
            self.next += 1;
            block
        })
    }

    pub fn all_published(&self) -> bool {
        self.next == self.remaining.len()
    }
}

/// Receives each complete block of a [`PleOffloadReader::read_rows_streamed_blocks`]
/// read, in order: `(block index, rows of that block)`.
pub type PleBlockSink<'a> = &'a mut dyn FnMut(usize, Vec<PleNvfp4Row>) -> Result<()>;

impl PleOffloadReader {
    pub fn open(
        manifest_path: &Path,
        queue_depth: usize,
        cache_bytes: usize,
        io_timeout: Duration,
    ) -> Result<Self> {
        Self::open_with_mode(
            manifest_path,
            queue_depth,
            cache_bytes,
            PleIoMode::Direct,
            io_timeout,
        )
    }

    pub fn open_with_mode(
        manifest_path: &Path,
        queue_depth: usize,
        cache_bytes: usize,
        io_mode: PleIoMode,
        io_timeout: Duration,
    ) -> Result<Self> {
        let force_pread = read_switch(FORCE_PREAD)?;
        Self::open_inner(
            manifest_path,
            queue_depth,
            cache_bytes,
            io_mode,
            io_timeout,
            force_pread,
        )
    }

    fn open_inner(
        manifest_path: &Path,
        queue_depth: usize,
        cache_bytes: usize,
        io_mode: PleIoMode,
        io_timeout: Duration,
        force_pread: bool,
    ) -> Result<Self> {
        let mmap_policy = parse_ple_mmap_policy(std::env::var(PLE_MMAP_SELECTOR).ok().as_deref())?;
        Self::open_inner_with_mmap(
            manifest_path,
            queue_depth,
            cache_bytes,
            io_mode,
            io_timeout,
            force_pread,
            mmap_policy,
        )
    }

    fn open_inner_with_mmap(
        manifest_path: &Path,
        queue_depth: usize,
        cache_bytes: usize,
        io_mode: PleIoMode,
        io_timeout: Duration,
        force_pread: bool,
        mmap_policy: PleMmapPolicy,
    ) -> Result<Self> {
        ensure!(
            queue_depth >= 16,
            "PLE offload queue depth must be at least 16"
        );
        // Honor the operator's configured queue depth (e.g., 256 per ring)
        // instead of forcing 512/ring (2048 concurrent O_DIRECT IOs across 4
        // rings), which oversubscribes single-NVMe command queues and mlock.
        ensure!(
            queue_depth <= u16::MAX as usize,
            "PLE offload queue depth ({queue_depth}) exceeds io_uring fixed-buffer index range ({})",
            u16::MAX
        );
        ensure!(!io_timeout.is_zero(), "PLE offload I/O timeout is zero");
        let manifest: Manifest = serde_json::from_slice(
            &std::fs::read(manifest_path)
                .with_context(|| format!("read PLE manifest {}", manifest_path.display()))?,
        )?;
        ensure!(
            manifest.format == EXPECTED_FORMAT,
            "unsupported PLE offload format {}",
            manifest.format
        );
        ensure!(
            manifest.page_bytes == EXPECTED_PAGE_BYTES,
            "PLE page size mismatch"
        );
        ensure!(
            manifest.record_bytes == EXPECTED_RECORD_BYTES,
            "PLE record size mismatch"
        );
        ensure!(
            manifest.records_per_page == EXPECTED_RECORDS_PER_PAGE,
            "PLE records/page mismatch"
        );
        ensure!(
            manifest.packed_bytes == 80 && manifest.scale_bytes == 10,
            "PLE field geometry mismatch"
        );
        ensure!(manifest.group_size == 16, "PLE NVFP4 group size must be 16");
        ensure!(!manifest.entries.is_empty(), "PLE manifest has no shards");

        let base = manifest_path
            .parent()
            .context("PLE manifest has no parent directory")?;
        let in_memory_fs = detect_in_memory_fs(base);
        let mmap_trigger = resolve_ple_mmap_trigger(mmap_policy, in_memory_fs);
        // `tmpfs`/`ramfs` reject `O_DIRECT` with `EINVAL`; when `mmap` is
        // requested or auto-engaged, open shards without `O_DIRECT`.
        let effective_io_mode = if mmap_trigger.is_some() {
            PleIoMode::PageCache
        } else {
            io_mode
        };

        let mut ordered = manifest.entries;
        ordered.sort_by_key(|entry| entry.shard);
        let mut shards = Vec::with_capacity(ordered.len());
        let mut shard_byte_lens = Vec::with_capacity(ordered.len());
        for (expected, entry) in ordered.into_iter().enumerate() {
            ensure!(
                entry.shard == expected,
                "PLE shards must be contiguous: expected {expected}, got {}",
                entry.shard
            );
            ensure!(
                entry.width == 160,
                "PLE shard {expected} width {} != 160",
                entry.width
            );
            ensure!(
                entry.scale2.is_finite() && entry.scale2 > 0.0,
                "PLE shard {expected} has invalid scale2"
            );
            let pages = entry.rows.div_ceil(EXPECTED_RECORDS_PER_PAGE);
            ensure!(
                entry.bytes == pages * EXPECTED_PAGE_BYTES,
                "PLE shard {expected} byte count mismatch"
            );
            let path = base.join(entry.file);
            let file = open_shard(&path, effective_io_mode).with_context(|| {
                format!("open PLE shard {} ({effective_io_mode:?})", path.display())
            })?;
            ensure!(
                file.metadata()?.len() as usize == entry.bytes,
                "PLE shard {expected} file size mismatch"
            );
            shard_byte_lens.push(entry.bytes);
            shards.push(Shard {
                file,
                rows: entry.rows,
                scale2: entry.scale2,
            });
        }

        let shard_mmaps: Vec<MappedShard> = if let Some(trigger) = mmap_trigger {
            let mapped = shards
                .iter()
                .zip(&shard_byte_lens)
                .enumerate()
                .map(|(idx, (shard, &len))| {
                    MappedShard::map(&shard.file, len)
                        .with_context(|| format!("mmap PLE shard {idx} ({len} B)"))
                })
                .collect::<Result<Vec<_>>>();
            match mapped {
                Ok(mmaps) => {
                    let total_bytes: usize = mmaps.iter().map(MappedShard::len).sum();
                    tracing::info!(
                        "QWEN4_PLE_MMAP_ENGAGED shards={} total_bytes={} trigger={trigger}",
                        mmaps.len(),
                        total_bytes
                    );
                    mmaps
                }
                Err(err) if mmap_policy == PleMmapPolicy::ForceOn => return Err(err),
                Err(err) => {
                    tracing::warn!(
                        "QWEN4_PLE_MMAP_FALLBACK trigger={trigger} error={err:#}; falling back to file I/O"
                    );
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };
        let use_mmap = !shard_mmaps.is_empty();

        let (pages, ring, fixed_buffers, pread_lanes) = if use_mmap {
            (Vec::new(), None, false, 1usize)
        } else {
            let mut pages: Vec<AlignedPage> =
                (0..queue_depth).map(|_| AlignedPage::new()).collect();
            let iovecs: Vec<libc::iovec> = pages
                .iter_mut()
                .map(|page| libc::iovec {
                    iov_base: page.ptr.cast(),
                    iov_len: EXPECTED_PAGE_BYTES,
                })
                .collect();
            let pread_lanes = parse_pread_lanes(std::env::var(PREAD_LANES).ok().as_deref())?;
            let ring_result = if force_pread {
                tracing::warn!(
                    "QWEN4_PLE_FORCE_PREAD selector={FORCE_PREAD}: skipping io_uring (diagnostic)"
                );
                None
            } else {
                Some(IoUring::new(queue_depth as u32))
            };
            let (ring, fixed_buffers) = match ring_result {
                None => (None, false),
                Some(Ok(ring)) if ring.params().is_feature_ext_arg() => {
                    // SAFETY: Every `iovec` points to an 8 KiB-aligned `AlignedPage` in
                    // `pages` that is declared after `ring` on `PleOffloadReader`, so
                    // `ring` closes and drops before `pages` are deallocated.
                    let fixed_buffers = match unsafe { ring.submitter().register_buffers(&iovecs) }
                    {
                        Ok(()) => true,
                        Err(err) => {
                            tracing::warn!(
                                "PLE io_uring register_buffers failed ({err}); falling back to unregistered O_DIRECT reads (pass --ulimit memlock=-1 to register fixed buffers)"
                            );
                            false
                        }
                    };
                    (Some(ring), fixed_buffers)
                }
                Some(Ok(_)) => {
                    tracing::warn!(
                        "PLE io_uring lacks IORING_FEAT_EXT_ARG; falling back to synchronous pread64 O_DIRECT reads"
                    );
                    (None, false)
                }
                Some(Err(err)) => {
                    tracing::warn!(
                        "PLE io_uring creation failed ({err}); falling back to {pread_lanes} synchronous pread64 O_DIRECT lanes (pass --security-opt seccomp=unconfined to enable io_uring)"
                    );
                    (None, false)
                }
            };
            (pages, ring, fixed_buffers, pread_lanes)
        };
        let shards: Arc<[Shard]> = shards.into();
        let mut read_plan = ReadPlanConfig::from_env(
            effective_io_mode,
            cache_bytes / EXPECTED_PAGE_BYTES,
            &shards,
        )?;
        // Without io_uring (or when mmap bypasses file I/O) extra rings are unused.
        let extra_rings = if use_mmap || ring.is_none() {
            Vec::new()
        } else {
            match (1..read_plan.rings)
                .map(|_| ReadRing::open(queue_depth))
                .collect::<Result<Vec<_>>>()
            {
                Ok(rings) => rings,
                Err(err) => {
                    tracing::warn!(
                        "PLE extra ring lane creation failed ({err}); using single lane"
                    );
                    read_plan.rings = 1;
                    Vec::new()
                }
            }
        };
        let ring_pool = if !use_mmap && ring.is_some() && !extra_rings.is_empty() {
            match RingWorkerPool::new(extra_rings, Arc::clone(&shards)) {
                Ok(pool) => Some(pool),
                Err(err) => {
                    tracing::warn!(
                        "PLE persistent ring worker pool creation failed ({err}); using single lane"
                    );
                    read_plan.rings = 1;
                    None
                }
            }
        } else {
            None
        };
        let pread_pages: Vec<AlignedPage> = if !use_mmap && ring.is_none() && pread_lanes > 1 {
            tracing::info!(
                "QWEN4_PLE_PREAD_LANES_ENGAGED selector={PREAD_LANES} lanes={pread_lanes} reason={} planned={}",
                if force_pread {
                    "forced"
                } else {
                    "io_uring_unavailable"
                },
                read_plan.fast_plan || read_plan.sector_ready
            );
            (0..pread_lanes).map(|_| AlignedPage::new()).collect()
        } else {
            Vec::new()
        };
        let pread_pool = if !use_mmap && ring.is_none() && pread_lanes > 1 {
            let workers = pread_lanes.min(MAX_PERSISTENT_PREAD_WORKERS);
            match PreadWorkerPool::new(workers, Arc::clone(&shards)) {
                Ok(pool) => {
                    tracing::info!(
                        "QWEN4_PLE_PREAD_POOL_ENGAGED workers={workers} max_lanes={pread_lanes}"
                    );
                    Some(pool)
                }
                Err(err) => {
                    tracing::warn!(
                        "PLE persistent pread worker pool creation failed ({err}); falling back to scoped threads"
                    );
                    None
                }
            }
        } else {
            None
        };
        let verify_read = verify_read::VerifyReadConfig::from_env()?;
        let record_cache_max_entries = resolve_record_cache_max_entries(
            cache_bytes,
            std::env::var(RECORD_CACHE_ENTRIES_ENV).ok().as_deref(),
            std::env::var(PLE_CACHE_MB_ENV).ok().as_deref(),
        );
        let shard_rows: Vec<usize> = shards.iter().map(|s| s.rows).collect();
        let shard_scale2: Vec<f32> = shards.iter().map(|s| s.scale2).collect();
        let parked_cache = Arc::new(ParkedPleCache::from_env(shard_rows, shard_scale2));
        if parked_cache.is_enabled() {
            tracing::info!(
                slots = parked_cache.slot_count(),
                bytes = parked_cache.byte_footprint(),
                slot_bytes = std::mem::size_of::<ParkedSlot>(),
                selector = PLE_PARKED_CACHE_MB_ENV,
                "PLE_PARKED_CACHE_ENGAGED"
            );
        } else {
            tracing::info!(
                selector = PLE_PARKED_CACHE_MB_ENV,
                "PLE_PARKED_CACHE_DISABLED"
            );
        }
        let reader = Self {
            ring_pool,
            pread_pool,
            ring,
            fixed_buffers,
            pages,
            shard_mmaps,
            shards,
            queue_depth,
            io_timeout,
            cache: HashMap::new(),
            cache_order: VecDeque::new(),
            cache_capacity_pages: cache_bytes / EXPECTED_PAGE_BYTES,
            cache_clock: 0,
            half_page: effective_io_mode == PleIoMode::Direct
                && cache_bytes / EXPECTED_PAGE_BYTES == 0
                && std::env::var("ATLAS_PLE_HALF_PAGE").ok().as_deref() == Some("1"),
            read_plan,
            pread_pages,
            record_cache: RecordCache::new(record_cache_max_entries, INITIAL_RECORD_CACHE_CAPACITY),
            parked_cache,
            verify_read,
        };
        reader.log_verify_read_receipt();
        Ok(reader)
    }

    #[inline]
    fn available_io_uring_rings(&self) -> usize {
        self.ring_pool
            .as_ref()
            .map_or(usize::from(self.ring.is_some()), RingWorkerPool::ring_count)
    }

    /// Lanes for one planned read of `selections` rows, when it should fan
    /// out: the pread lanes without io_uring, otherwise the configured rings
    /// once the read reaches `min_ring_selections`.
    fn planned_lanes(
        &self,
        rings: usize,
        selections: usize,
        min_ring_selections: usize,
    ) -> Option<usize> {
        if self.ring.is_none() {
            let lanes = self.pread_pages.len();
            return (lanes > 1 && selections >= MIN_PREAD_LANE_SELECTIONS).then_some(lanes);
        }
        resolve_io_uring_lanes(
            rings,
            self.available_io_uring_rings(),
            selections,
            min_ring_selections,
        )
    }

    fn cached_record(
        &mut self,
        key: (usize, usize),
        slot: usize,
    ) -> Option<[u8; EXPECTED_RECORD_BYTES]> {
        let cached = self.cache.get_mut(&key)?;
        self.cache_clock = self.cache_clock.wrapping_add(1);
        cached.generation = self.cache_clock;
        self.cache_order.push_back((key, self.cache_clock));
        let mut record = [0u8; EXPECTED_RECORD_BYTES];
        let start = slot * EXPECTED_RECORD_BYTES;
        record.copy_from_slice(&cached.bytes[start..start + EXPECTED_RECORD_BYTES]);
        Some(record)
    }

    /// Row count of every shard, in shard order.
    pub fn shard_rows(&self) -> Vec<usize> {
        self.shards.iter().map(|s| s.rows).collect()
    }

    /// Unique pages one `read_rows` call may submit.
    pub fn queue_depth(&self) -> usize {
        self.queue_depth
    }

    /// True when the reader is backed by a live Linux `io_uring` ring rather
    /// than the synchronous `pread64` fallback.
    #[must_use]
    pub fn has_io_uring(&self) -> bool {
        self.ring.is_some()
    }

    /// True when PLE shards are served via read-only `mmap` views rather than
    /// `pread64` or `io_uring` syscalls.
    #[must_use]
    pub fn has_mmap(&self) -> bool {
        !self.shard_mmaps.is_empty()
    }

    /// Shared handle to the fixed-footprint lock-free direct-mapped parked
    /// record cache. Callers may query it without acquiring the reader mutex;
    /// storage misses read through this reader populate the same cache.
    pub fn parked_cache(&self) -> Arc<ParkedPleCache> {
        Arc::clone(&self.parked_cache)
    }

    /// The mode the next streamed prefill read will use (K9 item W selectors
    /// and `ATLAS_PLE_READ_TRIGGER`, resolved now).
    pub fn streamed_read_mode(&self) -> crate::ple_plan::ReadMode {
        self.read_plan.mode()
    }

    /// The ring count the next streamed prefill read will use (1 = one ring).
    pub fn streamed_read_rings(&self) -> usize {
        self.read_plan.mode_and_rings().1
    }

    /// Read selected `(shard, row)` pairs in caller order. Identical pages in
    /// a batch are coalesced before submission.
    pub fn read_rows(&mut self, selections: &[(usize, usize)]) -> Result<Vec<PleNvfp4Row>> {
        self.read_rows_bounded(selections, Instant::now() + self.io_timeout)
    }

    fn read_rows_bounded(
        &mut self,
        selections: &[(usize, usize)],
        deadline: Instant,
    ) -> Result<Vec<PleNvfp4Row>> {
        ensure!(!selections.is_empty(), "PLE offload selection is empty");
        if !self.shard_mmaps.is_empty() {
            ensure!(
                Instant::now() < deadline,
                "PLE sparse batch read exceeded its operation deadline"
            );
            let rows = read_mmap_rows(&self.shards, &self.shard_mmaps, selections)?;
            for (&(shard_idx, row), row_val) in selections.iter().zip(&rows) {
                self.parked_cache.insert(shard_idx, row, &row_val.record);
                self.record_cache
                    .insert((shard_idx as u16, row as u32), row_val.record);
            }
            return Ok(rows);
        }
        let mut unique = Vec::<(usize, usize)>::new();
        let mut lookup = HashMap::<(usize, usize), usize>::new();
        let mut requested = Vec::with_capacity(selections.len());
        let mut output = vec![None; selections.len()];
        for (request_index, &(shard_idx, row)) in selections.iter().enumerate() {
            let (shard_rows, scale2) = self
                .shards
                .get(shard_idx)
                .map(|shard| (shard.rows, shard.scale2))
                .with_context(|| format!("PLE shard {shard_idx} out of range"))?;
            ensure!(
                row < shard_rows,
                "PLE row {row} out of range for shard {shard_idx}"
            );
            let page = row / EXPECTED_RECORDS_PER_PAGE;
            let key = (shard_idx, page);
            let slot = row % EXPECTED_RECORDS_PER_PAGE;
            if let Some(record) = self.parked_cache.get(shard_idx, row) {
                output[request_index] = Some(PleNvfp4Row { record, scale2 });
                continue;
            }
            if let Some(&record) = self.record_cache.get(&(shard_idx as u16, row as u32)) {
                self.parked_cache.insert(shard_idx, row, &record);
                output[request_index] = Some(PleNvfp4Row { record, scale2 });
                continue;
            }
            if let Some(record) = self.cached_record(key, slot) {
                self.parked_cache.insert(shard_idx, row, &record);
                self.record_cache
                    .insert((shard_idx as u16, row as u32), record);
                output[request_index] = Some(PleNvfp4Row { record, scale2 });
                continue;
            }
            let index = match lookup.get(&key) {
                Some(&index) => index,
                None => {
                    let index = unique.len();
                    lookup.insert(key, index);
                    unique.push(key);
                    index
                }
            };
            requested.push((request_index, index, slot, scale2));
        }
        if unique.is_empty() {
            return output
                .into_iter()
                .map(|row| row.context("PLE cache lookup left an empty row"))
                .collect();
        }
        ensure!(
            unique.len() <= self.queue_depth,
            "PLE request needs {} unique pages but queue depth is {}",
            unique.len(),
            self.queue_depth
        );
        ensure!(
            Instant::now() < deadline,
            "PLE sparse batch read exceeded its operation deadline"
        );

        if let Some(ring) = &mut self.ring {
            for (index, &(shard_idx, page)) in unique.iter().enumerate() {
                let fd = raw_fd(&self.shards[shard_idx].file);
                let entry = build_read_sqe(
                    self.fixed_buffers,
                    fd,
                    self.pages[index].ptr,
                    EXPECTED_PAGE_BYTES as u32,
                    index as u16,
                    (page * EXPECTED_PAGE_BYTES) as u64,
                    index as u64,
                );
                // SAFETY: `fd` belongs to `self.shards`, `self.pages[index]` is uniquely
                // owned for this batch slot (`unique.len() <= self.queue_depth`), and all
                // queued entries are completed or synchronously cancelled below before return.
                if unsafe { ring.submission().push(&entry) }.is_err() {
                    eprintln!(
                        "fatal: PLE sparse batch submission queue filled unexpectedly; exiting before queued fixed-buffer reads can escape"
                    );
                    std::process::exit(70);
                }
            }
            // Entries already placed in the SQ must be completed or cancelled;
            // never return and let another call submit them against reused pages.
            let remaining = deadline
                .saturating_duration_since(Instant::now())
                .max(Duration::from_nanos(1));
            crate::io_wait::submit_and_wait_bounded(
                ring,
                unique.len(),
                remaining,
                "PLE sparse batch read",
            )?;
            let mut completed = vec![false; unique.len()];
            let mut completed_count = 0usize;
            let mut first_err: Option<anyhow::Error> = None;
            for entry in ring.completion() {
                let index = entry.user_data() as usize;
                let res = entry.result();
                if index >= completed.len() {
                    first_err.get_or_insert_with(|| {
                        anyhow::anyhow!("PLE io_uring returned invalid tag {index}")
                    });
                    continue;
                }
                if completed[index] {
                    first_err.get_or_insert_with(|| {
                        anyhow::anyhow!(
                            "PLE io_uring returned duplicate completion for read {index}"
                        )
                    });
                    continue;
                }
                if res != EXPECTED_PAGE_BYTES as i32 {
                    first_err.get_or_insert_with(|| {
                        anyhow::anyhow!(
                            "PLE read {index} returned {res}, expected {EXPECTED_PAGE_BYTES}"
                        )
                    });
                    continue;
                }
                completed[index] = true;
                completed_count += 1;
            }
            if first_err.is_some() || completed_count < unique.len() {
                crate::io_wait::cancel_and_drain_or_exit(
                    ring,
                    "PLE sparse batch read error/incomplete",
                );
                if let Some(err) = first_err {
                    return Err(err);
                }
                bail!(
                    "PLE io_uring batch completed partially ({completed_count}/{})",
                    unique.len()
                );
            }
        } else if self.pread_pages.len() > 1 && unique.len() >= MIN_PREAD_DECODE_PAGES {
            // No io_uring: spread the batch's pages over synchronous lanes
            // (each page index is read by exactly one lane into its own buffer).
            let lanes = self
                .pread_pages
                .len()
                .min(MAX_PREAD_DECODE_LANES)
                .min(unique.len());
            let per_lane = unique.len().div_ceil(lanes);
            let shards: &[Shard] = &self.shards;
            let results: Vec<Result<()>> = std::thread::scope(|scope| {
                let handles: Vec<_> = self.pages[..unique.len()]
                    .chunks_mut(per_lane)
                    .zip(unique.chunks(per_lane))
                    .map(|(lane_pages, lane_keys)| {
                        scope.spawn(move || -> Result<()> {
                            for (page, &(shard_idx, page_no)) in
                                lane_pages.iter_mut().zip(lane_keys)
                            {
                                ensure!(
                                    Instant::now() < deadline,
                                    "PLE sparse batch read exceeded its operation deadline"
                                );
                                pread_exact_aligned(
                                    raw_fd(&shards[shard_idx].file),
                                    page.as_mut_slice(0, EXPECTED_PAGE_BYTES),
                                    (page_no * EXPECTED_PAGE_BYTES) as u64,
                                )?;
                            }
                            Ok(())
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| {
                        handle
                            .join()
                            .unwrap_or_else(|_| Err(anyhow::anyhow!("PLE pread lane panicked")))
                    })
                    .collect()
            });
            for result in results {
                result?;
            }
        } else {
            for (index, &(shard_idx, page)) in unique.iter().enumerate() {
                ensure!(
                    Instant::now() < deadline,
                    "PLE sparse batch read exceeded its operation deadline"
                );
                let fd = raw_fd(&self.shards[shard_idx].file);
                pread_exact_aligned(
                    fd,
                    self.pages[index].as_mut_slice(0, EXPECTED_PAGE_BYTES),
                    (page * EXPECTED_PAGE_BYTES) as u64,
                )?;
            }
        }
        for (page_index, &key) in unique.iter().enumerate() {
            // Copying a cold 8 KiB page is bounded by the configured cache;
            // cache_bytes=0 keeps the pure O_DIRECT path allocation-free.
            let source_ptr = self.pages[page_index].ptr;
            let source = AlignedPageRef { ptr: source_ptr };
            self.cache_page_ref(key, source);
        }
        for (request_index, page, slot, scale2) in requested {
            let record = self.pages[page].record(slot);
            if let Some(&(s_idx, r_idx)) = selections.get(request_index) {
                self.parked_cache.insert(s_idx, r_idx, &record);
                self.record_cache
                    .insert((s_idx as u16, r_idx as u32), record);
            }
            output[request_index] = Some(PleNvfp4Row { record, scale2 });
        }
        output
            .into_iter()
            .map(|row| row.context("PLE read left an empty row"))
            .collect()
    }

    /// Read an arbitrarily long ordered selection in queue-depth-bounded
    /// windows. This keeps callers from maintaining a second batch-size
    /// constant that can underfill the registered io_uring queue.
    pub fn read_rows_windowed(
        &mut self,
        selections: &[(usize, usize)],
    ) -> Result<Vec<PleNvfp4Row>> {
        Ok(self.read_rows_windowed_inner(selections, false)?.0)
    }

    /// [`Self::read_rows_windowed`] with the record-cache tally, and, for the
    /// speculative-verify caller (`verify = true`), the optional parallel
    /// scheduling of small misses (`ATLAS_PLE_VERIFY_PARALLEL_PREAD=1`).
    /// Same rows, same bytes, same caller order on every route.
    fn read_rows_windowed_inner(
        &mut self,
        selections: &[(usize, usize)],
        verify: bool,
    ) -> Result<(Vec<PleNvfp4Row>, verify_read::WindowedTally)> {
        ensure!(!selections.is_empty(), "PLE offload selection is empty");
        let deadline = Instant::now() + self.io_timeout;
        let mut output: Vec<Option<PleNvfp4Row>> = vec![None; selections.len()];
        let mut miss_selections: Vec<(usize, usize)> = Vec::new();
        let mut miss_waiters: Vec<Vec<usize>> = Vec::new();
        let mut miss_lookup: HashMap<(u16, u32), usize> = HashMap::new();
        let mut hits = 0usize;
        let mut parked_hits = 0u64;
        let mut parked_misses = 0u64;
        let mut route = verify_read::VerifyRoute::CacheOnly;
        for (req_idx, &(shard_idx, row)) in selections.iter().enumerate() {
            let (shard_rows, scale2) = self
                .shards
                .get(shard_idx)
                .map(|shard| (shard.rows, shard.scale2))
                .with_context(|| format!("PLE shard {shard_idx} out of range"))?;
            ensure!(
                row < shard_rows,
                "PLE row {row} out of range for shard {shard_idx}"
            );
            if let Some(record) = self.parked_cache.get(shard_idx, row) {
                output[req_idx] = Some(PleNvfp4Row { record, scale2 });
                hits += 1;
                parked_hits += 1;
                continue;
            }
            let rkey = (shard_idx as u16, row as u32);
            if let Some(&record) = self.record_cache.get(&rkey) {
                self.parked_cache.insert(shard_idx, row, &record);
                output[req_idx] = Some(PleNvfp4Row { record, scale2 });
                hits += 1;
                continue;
            }
            parked_misses += 1;
            let miss_idx = match miss_lookup.get(&rkey) {
                Some(&idx) => idx,
                None => {
                    let idx = miss_selections.len();
                    miss_lookup.insert(rkey, idx);
                    miss_selections.push((shard_idx, row));
                    miss_waiters.push(Vec::new());
                    idx
                }
            };
            miss_waiters[miss_idx].push(req_idx);
        }
        if parked_hits > 0 {
            self.parked_cache
                .hits
                .fetch_add(parked_hits, Ordering::Relaxed);
        }
        if parked_misses > 0 {
            self.parked_cache
                .misses
                .fetch_add(parked_misses, Ordering::Relaxed);
        }
        if !miss_selections.is_empty() {
            let miss_rows = if !self.shard_mmaps.is_empty() {
                ensure!(
                    Instant::now() < deadline,
                    "PLE windowed read exceeded its {:?} operation deadline",
                    self.io_timeout
                );
                route = verify_read::VerifyRoute::Bounded;
                read_mmap_rows(&self.shards, &self.shard_mmaps, &miss_selections)?
            } else if self.cache_capacity_pages == 0
                && (self.read_plan.sector_ready || self.read_plan.fast_plan)
            {
                let (mode, rings) = self.read_plan.mode_and_rings();
                if let Some(lanes) =
                    self.planned_lanes(rings, miss_selections.len(), VERIFY_MIN_RING_SELECTIONS)
                {
                    route = verify_read::VerifyRoute::PlannedLanes;
                    self.read_rows_streamed_rings(&miss_selections, None, None, true, mode, lanes)?
                } else {
                    // ATLAS_PLE_VERIFY_PARALLEL_PREAD=1 (verify only, no io_uring,
                    // < MIN_PREAD_LANE_SELECTIONS misses): the same planned sector
                    // reads, spread over pread lanes. 0 = the serial loop.
                    let lanes = if verify && self.verify_small_parallel(miss_selections.len()) {
                        route = verify_read::VerifyRoute::Parallel;
                        self.verify_pread_lanes()
                    } else {
                        route = verify_read::VerifyRoute::Serial;
                        0
                    };
                    self.read_rows_streamed_planned(
                        &miss_selections,
                        None,
                        None,
                        true,
                        mode,
                        lanes,
                    )?
                }
            } else if miss_selections.len() > self.queue_depth || self.half_page {
                route = verify_read::VerifyRoute::Streamed;
                self.read_rows_streamed_inner(&miss_selections, None, None, true)?
            } else {
                route = verify_read::VerifyRoute::Bounded;
                self.read_rows_bounded_windows(&miss_selections, deadline)?
            };
            ensure!(
                miss_rows.len() == miss_selections.len(),
                "PLE windowed read extent mismatch"
            );
            for (miss_idx, row_val) in miss_rows.into_iter().enumerate() {
                if let Some(&(s_idx, r_idx)) = miss_selections.get(miss_idx) {
                    self.parked_cache.insert(s_idx, r_idx, &row_val.record);
                    self.record_cache
                        .insert((s_idx as u16, r_idx as u32), row_val.record);
                }
                for &req_idx in &miss_waiters[miss_idx] {
                    output[req_idx] = Some(row_val.clone());
                }
            }
        }
        let misses = miss_selections.len();
        let rows = output
            .into_iter()
            .map(|row| row.context("PLE windowed read left an empty row"))
            .collect::<Result<Vec<_>>>()?;
        Ok((
            rows,
            verify_read::WindowedTally {
                hits,
                misses,
                route,
            },
        ))
    }

    /// Queue-depth windows of [`Self::read_rows_bounded`] (its pread fallback
    /// spreads each window's pages over up to `MAX_PREAD_DECODE_LANES` threads).
    fn read_rows_bounded_windows(
        &mut self,
        selections: &[(usize, usize)],
        deadline: Instant,
    ) -> Result<Vec<PleNvfp4Row>> {
        let mut rows = Vec::with_capacity(selections.len());
        for range in selection_batch_ranges(selections.len(), self.queue_depth)? {
            let remaining = deadline.saturating_duration_since(Instant::now());
            ensure!(
                !remaining.is_zero(),
                "PLE windowed read exceeded its {:?} operation deadline",
                self.io_timeout
            );
            rows.extend(self.read_rows_bounded(&selections[range], deadline)?);
        }
        Ok(rows)
    }

    /// Same rows, same bytes, same caller order as [`Self::read_rows_windowed`],
    /// but pages are deduplicated across the WHOLE selection and the ring is
    /// kept full: each completion's buffer is refilled with the next page at
    /// once, instead of draining a 256-read window before submitting the next.
    /// A 2048-token prefill is 32,768 rows in 128 windows, and waiting for
    /// each window's slowest read left the ring mostly empty.
    pub fn read_rows_streamed(
        &mut self,
        selections: &[(usize, usize)],
    ) -> Result<Vec<PleNvfp4Row>> {
        self.read_rows_streamed_inner(selections, None, None, true)
    }

    /// Stream rows directly to `visit(request_index, row)` without allocating
    /// or populating the intermediate `Vec<Option<PleNvfp4Row>>` (`collect_output = false`).
    pub fn visit_rows_streamed(
        &mut self,
        selections: &[(usize, usize)],
        visit: &mut dyn FnMut(usize, PleNvfp4Row) -> Result<()>,
    ) -> Result<()> {
        self.read_rows_streamed_inner(selections, None, Some(visit), false)?;
        Ok(())
    }

    /// [`Self::read_rows_streamed`] (same reads, same submission order) that
    /// hands every block of `rows_per_block` selections to `sink`, in order,
    /// as soon as its last row has arrived, so the caller can consume the
    /// head of a prefill chunk while its tail is still in flight.
    pub fn read_rows_streamed_blocks(
        &mut self,
        selections: &[(usize, usize)],
        rows_per_block: usize,
        sink: PleBlockSink<'_>,
    ) -> Result<()> {
        let mut tracker = PleBlockTracker::new(selections.len(), rows_per_block)?;
        self.read_rows_streamed_inner(selections, Some((&mut tracker, sink)), None, true)?;
        ensure!(
            tracker.all_published(),
            "PLE block read left blocks unpublished"
        );
        Ok(())
    }

    fn read_rows_streamed_mmap(
        &mut self,
        selections: &[(usize, usize)],
        blocks: Option<(&mut PleBlockTracker, PleBlockSink<'_>)>,
        mut visit: Option<&mut dyn FnMut(usize, PleNvfp4Row) -> Result<()>>,
        collect_output: bool,
    ) -> Result<Vec<PleNvfp4Row>> {
        ensure!(!selections.is_empty(), "PLE offload selection is empty");
        let stats_start = self.read_plan.stats.then(Instant::now);
        let cache_records = self.record_cache.max_entries > 0;
        let record_bytes_total = (selections.len() * EXPECTED_RECORD_BYTES) as u64;

        if let Some((tracker, sink)) = blocks {
            // Validate every selection upfront so invalid indices fail before
            // publishing any partial blocks to the sink.
            validate_mmap_selections(&self.shards, &self.shard_mmaps, selections)?;
            if tracker.rows_per_block >= MMAP_PARALLEL_MIN_SELECTIONS {
                for block_idx in 0..tracker.blocks() {
                    let range = tracker.range(block_idx);
                    let block_selections = &selections[range.clone()];
                    let rows = read_mmap_rows(&self.shards, &self.shard_mmaps, block_selections)?;
                    if cache_records {
                        for (&(s_idx, r_idx), row_val) in block_selections.iter().zip(&rows) {
                            self.record_cache
                                .insert((s_idx as u16, r_idx as u32), row_val.record);
                        }
                    }
                    if let Some(visit_fn) = visit.as_mut() {
                        for (offset, row_val) in rows.iter().enumerate() {
                            visit_fn(range.start + offset, row_val.clone())?;
                        }
                    }
                    for request_index in range {
                        tracker.fill(request_index)?;
                    }
                    let ready = tracker
                        .pop_ready()
                        .context("PLE mmap block tracker did not advance in order")?;
                    sink(ready, rows)?;
                }
            } else {
                let rows = read_mmap_rows(&self.shards, &self.shard_mmaps, selections)?;
                if cache_records {
                    for (&(s_idx, r_idx), row_val) in selections.iter().zip(&rows) {
                        self.record_cache
                            .insert((s_idx as u16, r_idx as u32), row_val.record);
                    }
                }
                if let Some(visit_fn) = visit.as_mut() {
                    for (request_index, row_val) in rows.iter().enumerate() {
                        visit_fn(request_index, row_val.clone())?;
                    }
                }
                let mut iter = rows.into_iter();
                for block_idx in 0..tracker.blocks() {
                    let range = tracker.range(block_idx);
                    let len = range.len();
                    let mut block_rows = Vec::with_capacity(len);
                    block_rows.extend(iter.by_ref().take(len));
                    for request_index in range {
                        tracker.fill(request_index)?;
                    }
                    let ready = tracker
                        .pop_ready()
                        .context("PLE mmap block tracker did not advance in order")?;
                    sink(ready, block_rows)?;
                }
            }
            if let Some(start) = stats_start {
                log_read_stats(
                    "mmap",
                    "record",
                    selections.len(),
                    selections.len(),
                    record_bytes_total,
                    Duration::ZERO,
                    start.elapsed(),
                );
            }
            return Ok(Vec::new());
        }

        let rows = read_mmap_rows(&self.shards, &self.shard_mmaps, selections)?;
        if cache_records {
            for (&(s_idx, r_idx), row_val) in selections.iter().zip(&rows) {
                self.record_cache
                    .insert((s_idx as u16, r_idx as u32), row_val.record);
            }
        }
        if let Some(visit_fn) = visit.as_mut() {
            if collect_output {
                for (request_index, row_val) in rows.iter().enumerate() {
                    visit_fn(request_index, row_val.clone())?;
                }
            } else {
                for (request_index, row_val) in rows.into_iter().enumerate() {
                    visit_fn(request_index, row_val)?;
                }
                if let Some(start) = stats_start {
                    log_read_stats(
                        "mmap",
                        "record",
                        selections.len(),
                        selections.len(),
                        record_bytes_total,
                        Duration::ZERO,
                        start.elapsed(),
                    );
                }
                return Ok(Vec::new());
            }
        }
        if let Some(start) = stats_start {
            log_read_stats(
                "mmap",
                "record",
                selections.len(),
                selections.len(),
                record_bytes_total,
                Duration::ZERO,
                start.elapsed(),
            );
        }
        Ok(if collect_output { rows } else { Vec::new() })
    }

    fn read_rows_streamed_inner(
        &mut self,
        selections: &[(usize, usize)],
        mut blocks: Option<(&mut PleBlockTracker, PleBlockSink<'_>)>,
        mut visit: Option<&mut dyn FnMut(usize, PleNvfp4Row) -> Result<()>>,
        collect_output: bool,
    ) -> Result<Vec<PleNvfp4Row>> {
        ensure!(!selections.is_empty(), "PLE offload selection is empty");
        if !self.shard_mmaps.is_empty() {
            return self.read_rows_streamed_mmap(selections, blocks, visit, collect_output);
        }
        // K9 item W (default off): planned reads; the loop below is unchanged.
        let (mode, rings) = self.read_plan.mode_and_rings();
        if mode != crate::ple_plan::ReadMode::Legacy {
            if let Some(lanes) = self.planned_lanes(rings, selections.len(), 512) {
                return self.read_rows_streamed_rings(
                    selections,
                    blocks,
                    visit,
                    collect_output,
                    mode,
                    lanes,
                );
            }
            return self.read_rows_streamed_planned(
                selections,
                blocks,
                visit,
                collect_output,
                mode,
                0,
            );
        }
        let stats_start = self.read_plan.stats.then(Instant::now);
        let deadline = Instant::now() + self.io_timeout;
        let mut unique = Vec::<(usize, usize)>::new();
        let mut lookup = HashMap::<(usize, usize), usize>::new();
        // Per unique page: the (request_index, slot, scale2) rows it serves.
        let mut waiters = Vec::<Vec<(usize, usize, f32)>>::new();
        let mut output = if collect_output {
            vec![None; selections.len()]
        } else {
            Vec::new()
        };
        for (request_index, &(shard_idx, row)) in selections.iter().enumerate() {
            let (shard_rows, scale2) = self
                .shards
                .get(shard_idx)
                .map(|shard| (shard.rows, shard.scale2))
                .with_context(|| format!("PLE shard {shard_idx} out of range"))?;
            ensure!(
                row < shard_rows,
                "PLE row {row} out of range for shard {shard_idx}"
            );
            let page = row / EXPECTED_RECORDS_PER_PAGE;
            let slot = row % EXPECTED_RECORDS_PER_PAGE;
            let key = if self.half_page {
                // (shard, page * 4 + part): part 0/1 = first/second 4 KiB half,
                // part 3 = the whole page (record straddles the halves).
                let first = slot * EXPECTED_RECORD_BYTES;
                let last = first + EXPECTED_RECORD_BYTES - 1;
                let half = EXPECTED_PAGE_BYTES / 2;
                let part = if last < half {
                    0
                } else if first >= half {
                    1
                } else {
                    3
                };
                (shard_idx, page * 4 + part)
            } else {
                (shard_idx, page)
            };
            if let Some(&record) = self.record_cache.get(&(shard_idx as u16, row as u32)) {
                let row_val = PleNvfp4Row { record, scale2 };
                if let Some(visit_fn) = visit.as_mut() {
                    visit_fn(request_index, row_val.clone())?;
                }
                if collect_output {
                    output[request_index] = Some(row_val);
                }
                if let Some((tracker, _)) = blocks.as_mut() {
                    tracker.fill(request_index)?;
                }
                continue;
            }
            if !self.half_page
                && let Some(record) = self.cached_record(key, slot)
            {
                self.record_cache
                    .insert((shard_idx as u16, row as u32), record);
                let row_val = PleNvfp4Row { record, scale2 };
                if let Some(visit_fn) = visit.as_mut() {
                    visit_fn(request_index, row_val.clone())?;
                }
                if collect_output {
                    output[request_index] = Some(row_val);
                }
                if let Some((tracker, _)) = blocks.as_mut() {
                    tracker.fill(request_index)?;
                }
                continue;
            }
            let index = *lookup.entry(key).or_insert_with(|| {
                unique.push(key);
                waiters.push(Vec::new());
                unique.len() - 1
            });
            waiters[index].push((request_index, slot, scale2));
        }
        let stats_planned = stats_start.map(|start| start.elapsed());

        let mut free: Vec<usize> = (0..self.queue_depth).rev().collect();
        let mut next = 0usize;
        let mut done = 0usize;
        let mut completed = vec![false; unique.len()];
        if self.ring.is_none() {
            for (index, &(shard_idx, key_page)) in unique.iter().enumerate() {
                ensure!(
                    Instant::now() < deadline,
                    "PLE streamed read exceeded its {:?} operation deadline",
                    self.io_timeout
                );
                let (page, skip, len) = if self.half_page {
                    let half = EXPECTED_PAGE_BYTES / 2;
                    match key_page % 4 {
                        0 => (key_page / 4, 0, half),
                        1 => (key_page / 4, half, half),
                        _ => (key_page / 4, 0, EXPECTED_PAGE_BYTES),
                    }
                } else {
                    (key_page, 0, EXPECTED_PAGE_BYTES)
                };
                let fd = raw_fd(&self.shards[shard_idx].file);
                pread_exact_aligned(
                    fd,
                    self.pages[0].as_mut_slice(skip, len),
                    (page * EXPECTED_PAGE_BYTES + skip) as u64,
                )?;
                for &(request_index, slot, scale2) in &waiters[index] {
                    let record = self.pages[0].record(slot);
                    if let Some(&(s_idx, r_idx)) = selections.get(request_index) {
                        self.record_cache
                            .insert((s_idx as u16, r_idx as u32), record);
                    }
                    let row_val = PleNvfp4Row { record, scale2 };
                    if let Some(visit_fn) = visit.as_mut() {
                        visit_fn(request_index, row_val.clone())?;
                    }
                    if collect_output {
                        output[request_index] = Some(row_val);
                    }
                    if let Some((tracker, _)) = blocks.as_mut() {
                        tracker.fill(request_index)?;
                    }
                }
                if !self.half_page {
                    let source = AlignedPageRef {
                        ptr: self.pages[0].ptr,
                    };
                    self.cache_page_ref(unique[index], source);
                }
                if let Some((tracker, sink)) = blocks.as_mut() {
                    publish_ready(tracker, &mut **sink, &mut output)?;
                }
                completed[index] = true;
                done += 1;
            }
        }
        if done < unique.len() {
            let ring = self
                .ring
                .as_mut()
                .expect("streamed io_uring loop requires Some(ring)");
            let mut guard = InFlightRingGuard::new(ring, "PLE streamed read unwind/abort");
            let mut loop_err: Option<anyhow::Error> = None;
            'io_loop: while done < unique.len() {
                if Instant::now() >= deadline {
                    guard.cancel_and_drain("PLE streamed read operation deadline");
                    bail!(
                        "PLE streamed read exceeded its {:?} operation deadline",
                        self.io_timeout
                    );
                }
                let mut submitted = 0usize;
                while next < unique.len() {
                    let Some(buffer) = free.pop() else { break };
                    let (shard_idx, key_page) = unique[next];
                    let (page, skip, len) = if self.half_page {
                        let half = EXPECTED_PAGE_BYTES / 2;
                        match key_page % 4 {
                            0 => (key_page / 4, 0, half),
                            1 => (key_page / 4, half, half),
                            _ => (key_page / 4, 0, EXPECTED_PAGE_BYTES),
                        }
                    } else {
                        (key_page, 0, EXPECTED_PAGE_BYTES)
                    };
                    assert!(
                        skip + len <= EXPECTED_PAGE_BYTES,
                        "PLE streamed read slice [{skip}..+{len}) exceeds page buffer"
                    );
                    let fd = raw_fd(&self.shards[shard_idx].file);
                    let entry = build_read_sqe(
                        self.fixed_buffers,
                        fd,
                        // SAFETY: `skip + len <= EXPECTED_PAGE_BYTES` was asserted above,
                        // so `ptr.add(skip)` stays inside `self.pages[buffer]`.
                        unsafe { self.pages[buffer].ptr.add(skip) },
                        len as u32,
                        buffer as u16,
                        (page * EXPECTED_PAGE_BYTES + skip) as u64,
                        ((next as u64) << 16) | buffer as u64,
                    );
                    // SAFETY: `fd` belongs to `self.shards`, `buffer` is uniquely claimed
                    // from `free` until its CQE arrives, and `guard` drains all in-flight
                    // SQEs on error or panic before `self.pages` can be reused or dropped.
                    if unsafe { guard.ring().submission().push(&entry) }.is_err() {
                        eprintln!(
                            "fatal: PLE streamed submission queue filled unexpectedly; exiting before queued fixed-buffer reads can escape"
                        );
                        std::process::exit(70);
                    }
                    next += 1;
                    submitted += 1;
                }
                guard.add_in_flight(submitted);
                ensure!(
                    guard.in_flight() > 0,
                    "PLE streamed read stalled with nothing in flight"
                );
                // Once entries are in the SQ they must be submitted and either
                // completed or synchronously cancelled; returning with them queued
                // would let a later call submit reads for buffers it has reused.
                let remaining = deadline
                    .saturating_duration_since(Instant::now())
                    .max(Duration::from_nanos(1));
                let want = if next == unique.len() && blocks.is_none() {
                    guard.in_flight()
                } else {
                    1
                };
                crate::io_wait::submit_and_wait_bounded(
                    guard.ring(),
                    want,
                    remaining,
                    "PLE streamed read",
                )?;
                let completions: Vec<(u64, i32)> = guard
                    .ring()
                    .completion()
                    .map(|entry| (entry.user_data(), entry.result()))
                    .collect();
                for (tag, result) in completions {
                    let (index, buffer) = ((tag >> 16) as usize, (tag & 0xffff) as usize);
                    if index >= unique.len() || buffer >= self.queue_depth {
                        guard.cancel_and_drain("PLE streamed read invalid completion tag");
                        bail!("PLE io_uring returned invalid tag {tag}");
                    }
                    if completed[index] {
                        guard.cancel_and_drain("PLE streamed read duplicate completion");
                        bail!("PLE io_uring returned duplicate completion for read {index}");
                    }
                    let expected = if self.half_page && unique[index].1 % 4 != 3 {
                        EXPECTED_PAGE_BYTES / 2
                    } else {
                        EXPECTED_PAGE_BYTES
                    };
                    if result != expected as i32 {
                        guard.cancel_and_drain("PLE streamed read completion error");
                        bail!("PLE read {index} returned {result}, expected {expected}");
                    }
                    completed[index] = true;
                    guard.dec_in_flight();
                    done += 1;
                    for &(request_index, slot, scale2) in &waiters[index] {
                        let record = self.pages[buffer].record(slot);
                        if let Some(&(s_idx, r_idx)) = selections.get(request_index) {
                            self.record_cache
                                .insert((s_idx as u16, r_idx as u32), record);
                        }
                        let row_val = PleNvfp4Row { record, scale2 };
                        if let Some(visit_fn) = visit.as_mut()
                            && let Err(err) = visit_fn(request_index, row_val.clone())
                        {
                            loop_err = Some(err);
                            break 'io_loop;
                        }
                        if collect_output {
                            output[request_index] = Some(row_val);
                        }
                        if let Some((tracker, _)) = blocks.as_mut()
                            && let Err(err) = tracker.fill(request_index)
                        {
                            loop_err = Some(err);
                            break 'io_loop;
                        }
                    }
                    if !self.half_page {
                        let source = AlignedPageRef {
                            ptr: self.pages[buffer].ptr,
                        };
                        cache_page_ref_fields(
                            self.cache_capacity_pages,
                            &mut self.cache_clock,
                            &mut self.cache,
                            &mut self.cache_order,
                            unique[index],
                            source,
                        );
                    }
                    free.push(buffer);
                }
                if let Some((tracker, sink)) = blocks.as_mut()
                    && let Err(err) = publish_ready(tracker, &mut **sink, &mut output)
                {
                    loop_err = Some(err);
                    break 'io_loop;
                }
            }
            if let Some(err) = loop_err {
                if guard.in_flight() != 0 {
                    guard.cancel_and_drain("PLE streamed read visitor/sink error");
                }
                return Err(err);
            }
            if guard.in_flight() != 0 {
                let remaining_in_flight = guard.in_flight();
                guard.cancel_and_drain("PLE streamed read incomplete retirement");
                bail!(
                    "PLE streamed read ended with {remaining_in_flight} requests still in flight"
                );
            }
        }
        if completed.iter().any(|done| !done) {
            bail!("PLE streamed read ended with incomplete requests");
        }
        if let (Some(start), Some(planned)) = (stats_start, stats_planned) {
            let bytes: u64 = unique
                .iter()
                .map(|&(_, key)| {
                    if self.half_page && key % 4 != 3 {
                        (EXPECTED_PAGE_BYTES / 2) as u64
                    } else {
                        EXPECTED_PAGE_BYTES as u64
                    }
                })
                .sum();
            log_read_stats(
                "legacy",
                if self.half_page { "half_page" } else { "page" },
                selections.len(),
                unique.len(),
                bytes,
                planned,
                start.elapsed(),
            );
        }
        if let Some((tracker, sink)) = blocks.as_mut() {
            // Blocks served entirely by the private page cache, or a read with no I/O.
            publish_ready(tracker, &mut **sink, &mut output)?;
            return Ok(Vec::new());
        }
        if !collect_output {
            return Ok(Vec::new());
        }
        output
            .into_iter()
            .map(|row| row.context("PLE streamed read left an empty row"))
            .collect()
    }

    /// Serve `record_cache` hits directly and plan storage reads only for
    /// uncached misses, remapping each waiter's `request_index` back to its
    /// index in `selections`. When the cache is empty, delegates straight to
    /// [`crate::ple_plan::plan_fast`].
    #[allow(clippy::too_many_arguments)]
    fn plan_uncached_reads(
        &self,
        selections: &[(usize, usize)],
        granularity: crate::ple_plan::ReadGranularity,
        collect_output: bool,
        output: &mut [Option<PleNvfp4Row>],
        mut filled_bits: Option<&mut [u64]>,
        blocks: &mut Option<(&mut PleBlockTracker, PleBlockSink<'_>)>,
        visit: &mut Option<&mut dyn FnMut(usize, PleNvfp4Row) -> Result<()>>,
    ) -> Result<crate::ple_plan::ReadPlan> {
        use crate::ple_plan::{ReadPlan, plan_fast};
        if self.record_cache.is_empty() {
            let shard_rows: Vec<usize> = self.shards.iter().map(|shard| shard.rows).collect();
            let shard_scale2: Vec<f32> = self.shards.iter().map(|shard| shard.scale2).collect();
            return plan_fast(selections, &shard_rows, &shard_scale2, granularity)
                .map_err(anyhow::Error::msg);
        }
        ensure!(
            selections.len() <= u32::MAX as usize,
            "PLE selection count exceeds u32"
        );
        let mut miss_selections = Vec::new();
        let mut miss_indices = Vec::new();
        for (request_index, &(shard_idx, row)) in selections.iter().enumerate() {
            let (shard_row_count, scale2) = self
                .shards
                .get(shard_idx)
                .map(|shard| (shard.rows, shard.scale2))
                .with_context(|| format!("PLE shard {shard_idx} out of range"))?;
            ensure!(
                row < shard_row_count,
                "PLE row {row} out of range for shard {shard_idx}"
            );
            if let Some(&record) = self.record_cache.get(&(shard_idx as u16, row as u32)) {
                let row_val = PleNvfp4Row { record, scale2 };
                if collect_output {
                    if let Some(visit_fn) = visit.as_mut() {
                        visit_fn(request_index, row_val.clone())?;
                    }
                    output[request_index] = Some(row_val);
                } else {
                    if let Some(bits) = filled_bits.as_deref_mut() {
                        let word = request_index / 64;
                        let bit = 1u64 << (request_index % 64);
                        bits[word] |= bit;
                    }
                    if let Some(visit_fn) = visit.as_mut() {
                        visit_fn(request_index, row_val)?;
                    }
                }
                if let Some((tracker, _)) = blocks.as_mut() {
                    tracker.fill(request_index)?;
                }
            } else {
                miss_selections.push((shard_idx, row));
                miss_indices.push(request_index as u32);
            }
        }
        if let Some((tracker, sink)) = blocks.as_mut() {
            publish_ready(tracker, &mut **sink, output)?;
        }
        if miss_selections.is_empty() {
            return Ok(ReadPlan {
                granularity,
                keys: Vec::new(),
                offsets: vec![0],
                waiters: Vec::new(),
            });
        }
        let shard_rows: Vec<usize> = self.shards.iter().map(|shard| shard.rows).collect();
        let shard_scale2: Vec<f32> = self.shards.iter().map(|shard| shard.scale2).collect();
        let mut plan = plan_fast(&miss_selections, &shard_rows, &shard_scale2, granularity)
            .map_err(anyhow::Error::msg)?;
        for waiter in &mut plan.waiters {
            waiter.request_index = miss_indices[waiter.request_index as usize];
        }
        Ok(plan)
    }

    /// [`Self::read_rows_streamed_inner`] for `ATLAS_PLE_FAST_PLAN` /
    /// `ATLAS_PLE_SECTOR_READ`: the plan comes from
    /// [`crate::ple_plan::plan_fast`] and every read lands where
    /// [`crate::ple_plan::read_span`] says. Same submission discipline, deadline,
    /// cancellation, completion checks and block publication as the original
    /// loop; requires the private page cache to be off (`cache_pages == 0`,
    /// checked at open).
    fn read_rows_streamed_planned(
        &mut self,
        selections: &[(usize, usize)],
        mut blocks: Option<(&mut PleBlockTracker, PleBlockSink<'_>)>,
        mut visit: Option<&mut dyn FnMut(usize, PleNvfp4Row) -> Result<()>>,
        collect_output: bool,
        mode: crate::ple_plan::ReadMode,
        pread_lanes: usize,
    ) -> Result<Vec<PleNvfp4Row>> {
        if !self.shard_mmaps.is_empty() {
            return self.read_rows_streamed_mmap(selections, blocks, visit, collect_output);
        }
        use crate::ple_plan::{ReadGranularity, ReadMode};
        let start = Instant::now();
        let deadline = start + self.io_timeout;
        let granularity = match mode {
            ReadMode::Sector => ReadGranularity::Sector,
            ReadMode::Planned | ReadMode::Legacy if self.half_page => ReadGranularity::HalfPage,
            ReadMode::Planned | ReadMode::Legacy => ReadGranularity::Page,
        };
        ensure!(
            self.cache_capacity_pages == 0,
            "PLE planned reads require the private page cache to be off"
        );
        let mut output = if collect_output {
            vec![None; selections.len()]
        } else {
            Vec::new()
        };
        let plan = self.plan_uncached_reads(
            selections,
            granularity,
            collect_output,
            &mut output,
            None,
            &mut blocks,
            &mut visit,
        )?;
        let planned = start.elapsed();
        log_once(format!(
            "QWEN4_PLE_READ_PLAN_RAN mode={} granularity={}",
            mode.label(),
            granularity.label()
        ));
        let reads = plan.reads();
        let plan_bytes = plan.bytes();
        if reads > 0 {
            let mut free: Vec<usize> = (0..self.queue_depth).rev().collect();
            let mut next = 0usize;
            let mut completed = vec![false; reads];
            if self.ring.is_none() {
                if pread_lanes > 1 && self.pread_pool.is_some() {
                    let io_timeout = self.io_timeout;
                    let total_selections = selections.len();
                    let mut filled_bits: Vec<u64> = if collect_output {
                        Vec::new()
                    } else {
                        vec![0u64; total_selections.div_ceil(64)]
                    };
                    let Self {
                        pread_pool,
                        record_cache,
                        ..
                    } = self;
                    let pool = pread_pool.as_ref().expect("checked is_some above");
                    let active_lanes = pread_lanes.min(pool.worker_count()).min(reads);
                    let tile_size = (reads / active_lanes.max(1)).clamp(1, 8);
                    let abort = Arc::new(AtomicBool::new(false));
                    let (rx, dispatched, dispatch_err) = pool.dispatch(
                        Arc::new(plan),
                        active_lanes,
                        tile_size,
                        deadline,
                        io_timeout,
                        Arc::clone(&abort),
                    );
                    struct PreadDrainGuard<'a> {
                        abort: &'a AtomicBool,
                        rx: Option<mpsc::Receiver<LaneMsg>>,
                    }
                    impl Drop for PreadDrainGuard<'_> {
                        fn drop(&mut self) {
                            if let Some(rx) = self.rx.take() {
                                self.abort.store(true, Ordering::Release);
                                while rx.recv().is_ok() {}
                            }
                        }
                    }
                    let mut drain_guard = PreadDrainGuard {
                        abort: &abort,
                        rx: Some(rx),
                    };
                    let mut first_error = dispatch_err;
                    let mut done_count = 0usize;
                    if let Some(rx_ref) = drain_guard.rx.as_ref() {
                        for message in rx_ref.iter() {
                            match message {
                                LaneMsg::Rows(rows) => {
                                    if first_error.is_some() {
                                        continue;
                                    }
                                    let filled = apply_ring_rows(
                                        rows,
                                        selections,
                                        total_selections,
                                        collect_output,
                                        record_cache,
                                        &mut output,
                                        &mut filled_bits,
                                        &mut blocks,
                                        &mut visit,
                                    );
                                    if let Err(error) = filled {
                                        abort.store(true, Ordering::Release);
                                        first_error = Some(error);
                                    }
                                }
                                LaneMsg::Done(result) => {
                                    done_count += 1;
                                    if let Err(error) = result {
                                        first_error.get_or_insert(error);
                                    }
                                }
                            }
                        }
                    }
                    drain_guard.rx = None;
                    drop(drain_guard);
                    if done_count < dispatched {
                        first_error.get_or_insert_with(|| {
                            anyhow::anyhow!(
                                "PLE pread worker exited unexpectedly ({done_count}/{dispatched} completed)"
                            )
                        });
                    }
                    if let Some(error) = first_error {
                        return Err(error);
                    }
                    completed.fill(true);
                } else {
                    // Parallel lanes fallback when no persistent pread pool exists:
                    // read every span into its own page buffer first, then consume
                    // them in index order below, exactly as the serial loop does.
                    let parallel = pread_lanes > 1 && reads <= self.pages.len();
                    if parallel {
                        let jobs: Vec<Option<verify_read::SpanRead>> = (0..reads)
                            .map(|index| {
                                (!plan.waiters_of(index).is_empty()).then(|| {
                                    let span = plan.span(index);
                                    verify_read::SpanRead {
                                        fd: raw_fd(&self.shards[plan.keys[index].0].file),
                                        buffer_offset: span.buffer_offset,
                                        len: span.len,
                                        file_offset: span.file_offset,
                                    }
                                })
                            })
                            .collect();
                        verify_read::pread_spans_parallel(
                            &mut self.pages[..reads],
                            &jobs,
                            pread_lanes,
                            deadline,
                            self.io_timeout,
                        )?;
                    }
                    for (index, completed_slot) in completed.iter_mut().enumerate() {
                        let waiters = plan.waiters_of(index);
                        if waiters.is_empty() {
                            *completed_slot = true;
                            continue;
                        }
                        let buffer = if parallel { index } else { 0 };
                        if !parallel {
                            ensure!(
                                Instant::now() < deadline,
                                "PLE planned read exceeded its {:?} operation deadline",
                                self.io_timeout
                            );
                            let shard_idx = plan.keys[index].0;
                            let span = plan.span(index);
                            let fd = raw_fd(&self.shards[shard_idx].file);
                            pread_exact_aligned(
                                fd,
                                self.pages[0].as_mut_slice(span.buffer_offset, span.len),
                                span.file_offset,
                            )?;
                        }
                        for waiter in waiters {
                            let request_index = waiter.request_index as usize;
                            let record =
                                self.pages[buffer].record_at(waiter.record_offset as usize);
                            if let Some(&(s_idx, r_idx)) = selections.get(request_index) {
                                self.record_cache
                                    .insert((s_idx as u16, r_idx as u32), record);
                            }
                            let row_val = PleNvfp4Row {
                                record,
                                scale2: waiter.scale2,
                            };
                            if let Some(visit_fn) = visit.as_mut() {
                                visit_fn(request_index, row_val.clone())?;
                            }
                            if collect_output {
                                output[request_index] = Some(row_val);
                            }
                            if let Some((tracker, _)) = blocks.as_mut() {
                                tracker.fill(request_index)?;
                            }
                        }
                        if let Some((tracker, sink)) = blocks.as_mut() {
                            publish_ready(tracker, &mut **sink, &mut output)?;
                        }
                        *completed_slot = true;
                    }
                }
            } else if let Some(ring) = self.ring.as_mut() {
                let mut guard = InFlightRingGuard::new(ring, "PLE planned read unwind/abort");
                let mut done = 0usize;
                let mut loop_err: Option<anyhow::Error> = None;
                'io_loop: while done < reads {
                    if Instant::now() >= deadline {
                        guard.cancel_and_drain("PLE planned read operation deadline");
                        bail!(
                            "PLE planned read exceeded its {:?} operation deadline",
                            self.io_timeout
                        );
                    }
                    let mut submitted = 0usize;
                    while next < reads {
                        if plan.waiters_of(next).is_empty() {
                            completed[next] = true;
                            done += 1;
                            next += 1;
                            continue;
                        }
                        let Some(buffer) = free.pop() else { break };
                        let shard_idx = plan.keys[next].0;
                        let span = plan.span(next);
                        let fd = raw_fd(&self.shards[shard_idx].file);
                        let entry = build_read_sqe(
                            self.fixed_buffers,
                            fd,
                            // SAFETY: `read_span` keeps buffer_offset + len within the
                            // EXPECTED_PAGE_BYTES buffer (ple_plan tests).
                            unsafe { self.pages[buffer].ptr.add(span.buffer_offset) },
                            span.len as u32,
                            buffer as u16,
                            span.file_offset,
                            ((next as u64) << 16) | buffer as u64,
                        );
                        // SAFETY: `fd` belongs to `self.shards`, `buffer` is uniquely claimed
                        // from `free` until its CQE arrives, and `guard` drains all in-flight
                        // SQEs on error or panic before `self.pages` can be reused or dropped.
                        if unsafe { guard.ring().submission().push(&entry) }.is_err() {
                            eprintln!(
                                "fatal: PLE planned submission queue filled unexpectedly; exiting before queued fixed-buffer reads can escape"
                            );
                            std::process::exit(70);
                        }
                        next += 1;
                        submitted += 1;
                    }
                    guard.add_in_flight(submitted);
                    if done >= reads && guard.in_flight() == 0 {
                        break;
                    }
                    ensure!(
                        guard.in_flight() > 0,
                        "PLE planned read stalled with nothing in flight"
                    );
                    // Same rule as the original loop: entries in the SQ are submitted
                    // and either completed or synchronously cancelled before return.
                    let remaining = deadline
                        .saturating_duration_since(Instant::now())
                        .max(Duration::from_nanos(1));
                    let want = if next == reads && blocks.is_none() {
                        guard.in_flight()
                    } else {
                        1
                    };
                    crate::io_wait::submit_and_wait_bounded(
                        guard.ring(),
                        want,
                        remaining,
                        "PLE planned read",
                    )?;
                    let completions: Vec<(u64, i32)> = guard
                        .ring()
                        .completion()
                        .map(|entry| (entry.user_data(), entry.result()))
                        .collect();
                    for (tag, result) in completions {
                        let (index, buffer) = ((tag >> 16) as usize, (tag & 0xffff) as usize);
                        if index >= reads || buffer >= self.queue_depth {
                            guard.cancel_and_drain("PLE planned read invalid completion tag");
                            bail!("PLE io_uring returned invalid tag {tag}");
                        }
                        if completed[index] {
                            guard.cancel_and_drain("PLE planned read duplicate completion");
                            bail!("PLE io_uring returned duplicate completion for read {index}");
                        }
                        let expected = plan.span(index).len;
                        if result != expected as i32 {
                            guard.cancel_and_drain("PLE planned read completion error");
                            bail!("PLE read {index} returned {result}, expected {expected}");
                        }
                        completed[index] = true;
                        guard.dec_in_flight();
                        done += 1;
                        for waiter in plan.waiters_of(index) {
                            let request_index = waiter.request_index as usize;
                            let record =
                                self.pages[buffer].record_at(waiter.record_offset as usize);
                            if let Some(&(s_idx, r_idx)) = selections.get(request_index) {
                                self.record_cache
                                    .insert((s_idx as u16, r_idx as u32), record);
                            }
                            let row_val = PleNvfp4Row {
                                record,
                                scale2: waiter.scale2,
                            };
                            if let Some(visit_fn) = visit.as_mut()
                                && let Err(err) = visit_fn(request_index, row_val.clone())
                            {
                                loop_err = Some(err);
                                break 'io_loop;
                            }
                            if collect_output {
                                output[request_index] = Some(row_val);
                            }
                            if let Some((tracker, _)) = blocks.as_mut()
                                && let Err(err) = tracker.fill(request_index)
                            {
                                loop_err = Some(err);
                                break 'io_loop;
                            }
                        }
                        free.push(buffer);
                    }
                    if let Some((tracker, sink)) = blocks.as_mut()
                        && let Err(err) = publish_ready(tracker, &mut **sink, &mut output)
                    {
                        loop_err = Some(err);
                        break 'io_loop;
                    }
                }
                if let Some(err) = loop_err {
                    if guard.in_flight() != 0 {
                        guard.cancel_and_drain("PLE planned read visitor/sink error");
                    }
                    return Err(err);
                }
                if guard.in_flight() != 0 {
                    let remaining_in_flight = guard.in_flight();
                    guard.cancel_and_drain("PLE planned read incomplete retirement");
                    bail!(
                        "PLE planned read ended with {remaining_in_flight} requests still in flight"
                    );
                }
            }
            if completed.iter().any(|done| !done) {
                bail!("PLE planned read ended with incomplete requests");
            }
        }
        if self.read_plan.stats {
            log_read_stats(
                mode.label(),
                granularity.label(),
                selections.len(),
                reads,
                plan_bytes,
                planned,
                start.elapsed(),
            );
        }
        if let Some((tracker, sink)) = blocks.as_mut() {
            publish_ready(tracker, &mut **sink, &mut output)?;
            return Ok(Vec::new());
        }
        if !collect_output {
            return Ok(Vec::new());
        }
        output
            .into_iter()
            .map(|row| row.context("PLE planned read left an empty row"))
            .collect()
    }

    /// [`Self::read_rows_streamed_planned`] spread over `rings` io_uring rings
    /// (`ATLAS_PLE_READ_RINGS`), one persistent worker thread each: the same
    /// plan, reads dealt in `RING_TILE = 64` tiles, each ring drained by
    /// [`drain_ring`]. This thread receives the rows, fills the block tracker
    /// and publishes blocks in order exactly as the one-ring loop does; the
    /// bytes of every row are the same, only the issuing thread differs.
    #[allow(clippy::too_many_arguments)]
    fn read_rows_streamed_rings(
        &mut self,
        selections: &[(usize, usize)],
        mut blocks: Option<(&mut PleBlockTracker, PleBlockSink<'_>)>,
        mut visit: Option<&mut dyn FnMut(usize, PleNvfp4Row) -> Result<()>>,
        collect_output: bool,
        mode: crate::ple_plan::ReadMode,
        rings: usize,
    ) -> Result<Vec<PleNvfp4Row>> {
        if !self.shard_mmaps.is_empty() {
            return self.read_rows_streamed_mmap(selections, blocks, visit, collect_output);
        }
        use crate::ple_plan::{ReadGranularity, ReadMode};
        use std::sync::atomic::AtomicUsize;
        let start = Instant::now();
        let deadline = start + self.io_timeout;
        let granularity = match mode {
            ReadMode::Sector => ReadGranularity::Sector,
            ReadMode::Planned | ReadMode::Legacy if self.half_page => ReadGranularity::HalfPage,
            ReadMode::Planned | ReadMode::Legacy => ReadGranularity::Page,
        };
        ensure!(
            self.cache_capacity_pages == 0,
            "PLE planned reads require the private page cache to be off"
        );
        let pread = self.ring.is_none();
        let available = if pread {
            self.pread_pages.len()
        } else {
            self.available_io_uring_rings()
        };
        ensure!(
            (2..=available).contains(&rings),
            "PLE read asked for {rings} {}, {available} available",
            if pread { "pread lanes" } else { "rings" }
        );
        let total_selections = selections.len();
        let mut output: Vec<Option<PleNvfp4Row>> = if collect_output {
            vec![None; total_selections]
        } else {
            Vec::new()
        };
        let mut filled_bits: Vec<u64> = if collect_output {
            Vec::new()
        } else {
            vec![0u64; total_selections.div_ceil(64)]
        };
        let plan = self.plan_uncached_reads(
            selections,
            granularity,
            collect_output,
            &mut output,
            (!collect_output).then_some(filled_bits.as_mut_slice()),
            &mut blocks,
            &mut visit,
        )?;
        let planned = start.elapsed();
        log_once(format!(
            "QWEN4_PLE_READ_PLAN_RAN mode={} granularity={} {}={rings}",
            mode.label(),
            granularity.label(),
            if pread { "pread_lanes" } else { "rings" }
        ));
        let plan_reads = plan.reads();
        let plan_bytes = plan.bytes();
        if plan_reads > 0 {
            let io_timeout = self.io_timeout;
            let mut first_error: Option<anyhow::Error> = None;
            if pread {
                let tile_size = pread_ring_tile_size(plan_reads);
                let active_lanes = rings.min(plan_reads.div_ceil(tile_size));
                let use_persistent_pool = self
                    .pread_pool
                    .as_ref()
                    .is_some_and(|pool| active_lanes <= pool.worker_count());
                if use_persistent_pool {
                    let abort = Arc::new(AtomicBool::new(false));
                    let Self {
                        pread_pool,
                        record_cache,
                        ..
                    } = self;
                    let pool = pread_pool.as_ref().expect("checked is_some_and above");
                    let (rx, dispatched, dispatch_err) = pool.dispatch(
                        Arc::new(plan),
                        active_lanes,
                        tile_size,
                        deadline,
                        io_timeout,
                        Arc::clone(&abort),
                    );
                    if let Some(err) = dispatch_err {
                        first_error = Some(err);
                    }
                    struct PreadPoolDrainGuard<'a> {
                        abort: &'a AtomicBool,
                        rx: Option<mpsc::Receiver<LaneMsg>>,
                    }
                    impl Drop for PreadPoolDrainGuard<'_> {
                        fn drop(&mut self) {
                            if let Some(rx) = self.rx.take() {
                                self.abort.store(true, Ordering::Release);
                                while rx.recv().is_ok() {}
                            }
                        }
                    }
                    let mut drain_guard = PreadPoolDrainGuard {
                        abort: &abort,
                        rx: Some(rx),
                    };
                    let mut done_count = 0usize;
                    if let Some(rx_ref) = drain_guard.rx.as_ref() {
                        for message in rx_ref.iter() {
                            match message {
                                LaneMsg::Rows(rows) => {
                                    if first_error.is_some() {
                                        continue;
                                    }
                                    let filled = apply_ring_rows(
                                        rows,
                                        selections,
                                        total_selections,
                                        collect_output,
                                        record_cache,
                                        &mut output,
                                        &mut filled_bits,
                                        &mut blocks,
                                        &mut visit,
                                    );
                                    if let Err(error) = filled {
                                        abort.store(true, Ordering::Release);
                                        first_error = Some(error);
                                    }
                                }
                                LaneMsg::Done(result) => {
                                    done_count += 1;
                                    if let Err(error) = result {
                                        first_error.get_or_insert(error);
                                    }
                                }
                            }
                        }
                    }
                    drain_guard.rx = None;
                    drop(drain_guard);
                    if done_count < dispatched {
                        first_error.get_or_insert_with(|| {
                            anyhow::anyhow!(
                                "PLE pread worker exited unexpectedly ({done_count}/{dispatched} completed)"
                            )
                        });
                    }
                } else {
                    let abort = AtomicBool::new(false);
                    // pread lanes claim RING_TILE tiles of the plan from this counter.
                    let next_tile = AtomicUsize::new(0);
                    let (tx, rx) = mpsc::channel::<LaneMsg>();
                    let Self {
                        shards,
                        pread_pages,
                        record_cache,
                        ..
                    } = self;
                    let shards: &[Shard] = shards.as_ref();
                    let plan_ref = &plan;
                    std::thread::scope(|scope| {
                        let next_tile = &next_tile;
                        let mut spawned_lanes = 0usize;
                        for (lane, page) in pread_pages.iter_mut().take(active_lanes).enumerate() {
                            let tx = tx.clone();
                            let abort = &abort;
                            let spawned = std::thread::Builder::new()
                                .name(format!("ple-pread-{lane}"))
                                .spawn_scoped(scope, move || {
                                    let result = drain_pread_lane(
                                        page, shards, plan_ref, tile_size, next_tile, deadline,
                                        io_timeout, abort, &tx,
                                    );
                                    if result.is_err() {
                                        abort.store(true, Ordering::Release);
                                    }
                                    let _ = tx.send(LaneMsg::Done(result));
                                });
                            match spawned {
                                Ok(_) => spawned_lanes += 1,
                                Err(error) if spawned_lanes == 0 => {
                                    first_error.get_or_insert(
                                        anyhow::Error::from(error).context("spawn PLE pread lane"),
                                    );
                                    break;
                                }
                                Err(error) => {
                                    // Lanes claim tiles dynamically, so the running
                                    // lanes still complete the whole plan.
                                    tracing::warn!(
                                        "QWEN4_PLE_PREAD_LANES_SHORT spawned={spawned_lanes} requested={active_lanes} error={error}"
                                    );
                                    break;
                                }
                            }
                        }
                        drop(tx);
                        // Every spawned lane sends Done exactly once (or the channel closes).
                        for message in rx.iter() {
                            match message {
                                LaneMsg::Rows(rows) => {
                                    if first_error.is_some() {
                                        continue;
                                    }
                                    let filled = apply_ring_rows(
                                        rows,
                                        selections,
                                        total_selections,
                                        collect_output,
                                        record_cache,
                                        &mut output,
                                        &mut filled_bits,
                                        &mut blocks,
                                        &mut visit,
                                    );
                                    if let Err(error) = filled {
                                        abort.store(true, Ordering::Release);
                                        first_error = Some(error);
                                    }
                                }
                                LaneMsg::Done(result) => {
                                    if let Err(error) = result {
                                        first_error.get_or_insert(error);
                                    }
                                }
                            }
                        }
                    });
                }
            } else {
                let active_rings = rings.min(plan_reads);
                let plan_arc = Arc::new(plan);
                let abort = Arc::new(AtomicBool::new(false));
                let Self {
                    ring_pool,
                    ring,
                    fixed_buffers,
                    pages,
                    record_cache,
                    ..
                } = self;
                let pool = ring_pool
                    .as_ref()
                    .expect("io_uring multi-ring read requires persistent ring pool");
                let primary_lane = ReadRing {
                    ring: ring.take(),
                    pages: std::mem::take(pages),
                    fixed_buffers: *fixed_buffers,
                };
                let (rx, dispatched, dispatch_err) = pool.dispatch(
                    primary_lane,
                    plan_arc,
                    active_rings,
                    deadline,
                    io_timeout,
                    Arc::clone(&abort),
                );
                if let Some(err) = dispatch_err {
                    first_error = Some(err);
                }
                // RAII guard ensures peer rings abort, `rx` drains, and the
                // primary ring + pages are restored onto `self` even if a
                // caller `visit` or `sink` callback unwinds.
                struct PrimaryRestoreGuard<'a> {
                    ring_slot: &'a mut Option<IoUring>,
                    pages_slot: &'a mut Vec<AlignedPage>,
                    pool: &'a RingWorkerPool,
                    abort: &'a AtomicBool,
                    rx: Option<mpsc::Receiver<LaneMsg>>,
                }
                impl Drop for PrimaryRestoreGuard<'_> {
                    fn drop(&mut self) {
                        if let Some(rx) = self.rx.take() {
                            self.abort.store(true, Ordering::Release);
                            while rx.recv().is_ok() {}
                        }
                        if let Some(returned) = self.pool.take_returned_primary() {
                            *self.ring_slot = returned.ring;
                            *self.pages_slot = returned.pages;
                        }
                    }
                }
                let mut restore = PrimaryRestoreGuard {
                    ring_slot: ring,
                    pages_slot: pages,
                    pool,
                    abort: &abort,
                    rx: Some(rx),
                };
                let mut done_count = 0usize;
                if let Some(rx_ref) = restore.rx.as_ref() {
                    for message in rx_ref.iter() {
                        match message {
                            LaneMsg::Rows(rows) => {
                                if first_error.is_some() {
                                    continue;
                                }
                                let filled = apply_ring_rows(
                                    rows,
                                    selections,
                                    total_selections,
                                    collect_output,
                                    record_cache,
                                    &mut output,
                                    &mut filled_bits,
                                    &mut blocks,
                                    &mut visit,
                                );
                                if let Err(error) = filled {
                                    abort.store(true, Ordering::Release);
                                    first_error = Some(error);
                                }
                            }
                            LaneMsg::Done(result) => {
                                done_count += 1;
                                if let Err(error) = result {
                                    first_error.get_or_insert(error);
                                }
                            }
                        }
                    }
                }
                restore.rx = None;
                drop(restore);
                if done_count < dispatched {
                    first_error.get_or_insert_with(|| {
                        anyhow::anyhow!(
                            "PLE ring worker exited unexpectedly ({done_count}/{dispatched} completed)"
                        )
                    });
                }
            }
            if let Some(error) = first_error {
                return Err(error);
            }
        }
        if self.read_plan.stats {
            log_read_stats(
                mode.label(),
                granularity.label(),
                selections.len(),
                plan_reads,
                plan_bytes,
                planned,
                start.elapsed(),
            );
        }
        if let Some((tracker, sink)) = blocks.as_mut() {
            publish_ready(tracker, &mut **sink, &mut output)?;
            ensure!(
                tracker.all_published(),
                "PLE ring read left blocks unpublished"
            );
            return Ok(Vec::new());
        }
        if !collect_output {
            for (request_index, _) in selections.iter().enumerate() {
                let word = request_index / 64;
                let bit = 1u64 << (request_index % 64);
                ensure!(
                    (filled_bits[word] & bit) != 0,
                    "PLE ring read left an empty row"
                );
            }
            return Ok(Vec::new());
        }
        output
            .into_iter()
            .map(|row| row.context("PLE ring read left an empty row"))
            .collect()
    }

    fn cache_page_ref(&mut self, key: (usize, usize), source: AlignedPageRef) {
        cache_page_ref_fields(
            self.cache_capacity_pages,
            &mut self.cache_clock,
            &mut self.cache,
            &mut self.cache_order,
            key,
            source,
        );
    }
}

fn cache_page_ref_fields(
    cache_capacity_pages: usize,
    cache_clock: &mut u64,
    cache: &mut HashMap<(usize, usize), CachedPage>,
    cache_order: &mut VecDeque<((usize, usize), u64)>,
    key: (usize, usize),
    source: AlignedPageRef,
) {
    if cache_capacity_pages == 0 {
        return;
    }
    *cache_clock = cache_clock.wrapping_add(1);
    let generation = *cache_clock;
    let mut bytes = Box::new([0u8; EXPECTED_PAGE_BYTES]);
    // SAFETY: `source.ptr` points to a completed `AlignedPage` of `EXPECTED_PAGE_BYTES`
    // bytes with no in-flight DMA targeting it, disjoint from the fresh heap `bytes` box.
    unsafe {
        std::ptr::copy_nonoverlapping(source.ptr, bytes.as_mut_ptr(), EXPECTED_PAGE_BYTES);
    }
    cache.insert(key, CachedPage { bytes, generation });
    cache_order.push_back((key, generation));
    while cache.len() > cache_capacity_pages {
        let Some((old_key, old_generation)) = cache_order.pop_front() else {
            break;
        };
        if cache
            .get(&old_key)
            .is_some_and(|page| page.generation == old_generation)
        {
            cache.remove(&old_key);
        }
    }
}

/// `ATLAS_PLE_READ_STATS=1`: one line per streamed prefill read. `plan_us` is
/// the CPU time before the first read could be submitted; `total_us` ends
/// when the last read has completed.
fn log_read_stats(
    mode: &str,
    granularity: &str,
    selections: usize,
    reads: usize,
    bytes: u64,
    planned: Duration,
    total: Duration,
) {
    tracing::info!(
        "QWEN4_PLE_READ_STATS mode={mode} granularity={granularity} selections={selections} reads={reads} bytes={bytes} plan_us={} total_us={} io_us={}",
        planned.as_micros(),
        total.as_micros(),
        total.saturating_sub(planned).as_micros()
    );
}

/// Hand every newly complete block, in order, to `sink` (moving its rows out).
fn publish_ready(
    tracker: &mut PleBlockTracker,
    sink: &mut dyn FnMut(usize, Vec<PleNvfp4Row>) -> Result<()>,
    output: &mut [Option<PleNvfp4Row>],
) -> Result<()> {
    while let Some(block) = tracker.pop_ready() {
        let rows = output[tracker.range(block)]
            .iter_mut()
            .map(|row| row.take().context("PLE block published an empty row"))
            .collect::<Result<Vec<_>>>()?;
        sink(block, rows)?;
    }
    Ok(())
}

/// Apply one batch of completed ring rows into `record_cache`, `output` /
/// `filled_bits`, `visit`, and `blocks`, publishing any newly ready blocks.
#[allow(clippy::too_many_arguments)]
fn apply_ring_rows(
    rows: Vec<(u32, PleNvfp4Row)>,
    selections: &[(usize, usize)],
    total_selections: usize,
    collect_output: bool,
    record_cache: &mut RecordCache,
    output: &mut [Option<PleNvfp4Row>],
    filled_bits: &mut [u64],
    blocks: &mut Option<(&mut PleBlockTracker, PleBlockSink<'_>)>,
    visit: &mut Option<&mut dyn FnMut(usize, PleNvfp4Row) -> Result<()>>,
) -> Result<()> {
    for (request_index, row) in rows {
        let request_index = request_index as usize;
        if let Some(&(s_idx, r_idx)) = selections.get(request_index) {
            record_cache.insert((s_idx as u16, r_idx as u32), row.record);
        }
        if collect_output {
            ensure!(
                request_index < output.len() && output[request_index].is_none(),
                "PLE ring read filled selection {request_index} twice or out of range"
            );
            if let Some(visit_fn) = visit.as_mut() {
                visit_fn(request_index, row.clone())?;
            }
            output[request_index] = Some(row);
        } else {
            let word = request_index / 64;
            let bit = 1u64 << (request_index % 64);
            ensure!(
                request_index < total_selections && (filled_bits[word] & bit) == 0,
                "PLE ring read filled selection {request_index} twice or out of range"
            );
            filled_bits[word] |= bit;
            if let Some(visit_fn) = visit.as_mut() {
                visit_fn(request_index, row)?;
            }
        }
        if let Some((tracker, _)) = blocks.as_mut() {
            tracker.fill(request_index)?;
        }
    }
    if let Some((tracker, sink)) = blocks.as_mut() {
        publish_ready(tracker, &mut **sink, output)?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct AlignedPageRef {
    ptr: *const u8,
}

#[cfg(target_os = "linux")]
fn open_direct(path: &Path) -> std::io::Result<File> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
    // SAFETY: `path.as_ptr()` is a valid NUL-terminated C string; `O_RDONLY | O_DIRECT | O_CLOEXEC`
    // does not create a file and therefore requires no mode argument.
    let fd = unsafe {
        libc::open(
            path.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECT | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `fd >= 0` is a freshly opened unique file descriptor returned by `libc::open`.
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[cfg(target_os = "linux")]
fn open_page_cache(path: &Path) -> std::io::Result<File> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
    // SAFETY: `path.as_ptr()` is a valid NUL-terminated C string; `O_RDONLY | O_CLOEXEC`
    // does not create a file and therefore requires no mode argument.
    let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // Sparse 8 KiB lookups are random. Avoid pulling adjacent PLE pages that
    // were not selected by the n-gram hash; pages actually read remain
    // reclaimable in the normal Linux file cache.
    // SAFETY: `fd >= 0` is an open file descriptor returned by `libc::open` above.
    let advise = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_RANDOM) };
    if advise != 0 {
        // SAFETY: `fd` is exclusively owned locally and closed once on this error path.
        unsafe { libc::close(fd) };
        return Err(std::io::Error::from_raw_os_error(advise));
    }
    // SAFETY: `fd >= 0` is a live, exclusively owned file descriptor not yet wrapped in `File`.
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[cfg(target_os = "linux")]
fn open_shard(path: &Path, mode: PleIoMode) -> std::io::Result<File> {
    match mode {
        PleIoMode::Direct => open_direct(path),
        PleIoMode::PageCache => open_page_cache(path),
    }
}

#[cfg(not(target_os = "linux"))]
fn open_direct(_path: &Path) -> std::io::Result<File> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "PLE O_DIRECT offload requires Linux",
    ))
}

#[cfg(not(target_os = "linux"))]
fn open_shard(_path: &Path, _mode: PleIoMode) -> std::io::Result<File> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "PLE offload requires Linux io_uring",
    ))
}

#[cfg(unix)]
fn raw_fd(file: &File) -> RawFd {
    use std::os::fd::AsRawFd;
    file.as_raw_fd()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_tracker_publishes_complete_blocks_in_order() {
        let mut tracker = PleBlockTracker::new(10, 4).expect("tracker");
        assert_eq!(tracker.blocks(), 3);
        assert_eq!(tracker.range(2), 8..10);
        // Block 1 completes first but waits for block 0.
        for index in 4..8 {
            tracker.fill(index).expect("fill");
        }
        assert_eq!(tracker.pop_ready(), None);
        for index in 0..4 {
            tracker.fill(index).expect("fill");
        }
        assert_eq!(tracker.pop_ready(), Some(0));
        assert_eq!(tracker.pop_ready(), Some(1));
        assert_eq!(tracker.pop_ready(), None);
        tracker.fill(9).expect("fill");
        assert_eq!(tracker.pop_ready(), None);
        tracker.fill(8).expect("fill");
        assert_eq!(tracker.pop_ready(), Some(2));
        assert!(tracker.all_published());
    }

    #[test]
    fn block_tracker_rejects_double_and_out_of_range_fills() {
        let mut tracker = PleBlockTracker::new(3, 2).expect("tracker");
        tracker.fill(2).expect("fill");
        assert!(tracker.fill(2).is_err());
        assert!(tracker.fill(3).is_err());
        let mut tail = PleBlockTracker::new(5, 4).expect("tracker");
        assert!(
            tail.fill(6).is_err(),
            "an index past the end must not count toward the short tail block"
        );
        assert!(PleBlockTracker::new(3, 0).is_err());
    }

    #[test]
    fn publish_moves_each_block_once() {
        let row = |byte: u8| PleNvfp4Row {
            record: [byte; EXPECTED_RECORD_BYTES],
            scale2: f32::from(byte),
        };
        let mut output: Vec<Option<PleNvfp4Row>> = (0..5).map(|i| Some(row(i))).collect();
        let mut tracker = PleBlockTracker::new(5, 2).expect("tracker");
        for index in 0..5 {
            tracker.fill(index).expect("fill");
        }
        let mut seen = Vec::new();
        publish_ready(
            &mut tracker,
            &mut |block, rows: Vec<PleNvfp4Row>| {
                seen.push((block, rows.iter().map(|r| r.record[0]).collect::<Vec<_>>()));
                Ok(())
            },
            &mut output,
        )
        .expect("publish");
        assert_eq!(seen, vec![(0, vec![0, 1]), (1, vec![2, 3]), (2, vec![4])]);
        assert!(output.iter().all(Option::is_none));
        assert!(tracker.all_published());
    }

    #[test]
    fn selection_windows_cover_prefill_once_in_order() {
        let windows = selection_batch_ranges(32_768, 256).expect("valid PLE window plan");
        assert_eq!(windows.len(), 128);
        assert_eq!(windows.first(), Some(&(0..256)));
        assert_eq!(windows.last(), Some(&(32_512..32_768)));
        assert!(windows.iter().all(|window| window.len() <= 256));
        assert!(windows.windows(2).all(|pair| pair[0].end == pair[1].start));
    }

    #[test]
    fn selection_windows_keep_partial_tail_and_reject_zero_depth() {
        let windows = selection_batch_ranges(513, 256).expect("valid PLE tail plan");
        assert_eq!(windows, [0..256, 256..512, 512..513]);
        assert!(selection_batch_ranges(1, 0).is_err());
        assert!(
            selection_batch_ranges(0, 256)
                .expect("empty PLE plan")
                .is_empty()
        );
    }

    #[test]
    fn pread_lanes_default_and_bounds() {
        assert_eq!(parse_pread_lanes(None).unwrap(), DEFAULT_PREAD_LANES);
        assert_eq!(parse_pread_lanes(Some("")).unwrap(), DEFAULT_PREAD_LANES);
        assert_eq!(parse_pread_lanes(Some("1")).unwrap(), 1);
        assert_eq!(parse_pread_lanes(Some("64")).unwrap(), 64);
        assert_eq!(parse_pread_lanes(Some("256")).unwrap(), MAX_PREAD_LANES);
        for bad in ["0", "257", "-1", "32x", " 32"] {
            assert!(parse_pread_lanes(Some(bad)).is_err(), "{bad:?}");
        }
    }

    /// Byte parity and timing of the no-io_uring `pread64` lanes against the
    /// io_uring rings on the real PLE sidecar (O_DIRECT, so no page cache).
    ///
    /// ATLAS_PLE_TEST_MANIFEST=<model>/ple-offload/manifest.json \
    /// ATLAS_PLE_FAST_PLAN=1 ATLAS_PLE_SECTOR_READ=1 ATLAS_PLE_READ_RINGS=4 \
    /// cargo test -p spark-storage --release --lib -- --ignored pread_lanes_match --nocapture
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "needs the real PLE sidecar via ATLAS_PLE_TEST_MANIFEST"]
    fn pread_lanes_match_io_uring_bytes() {
        let manifest = std::env::var("ATLAS_PLE_TEST_MANIFEST")
            .expect("set ATLAS_PLE_TEST_MANIFEST to a ple-offload/manifest.json");
        let path = Path::new(&manifest);
        let timeout = Duration::from_secs(120);
        let open = |force_pread: bool| {
            PleOffloadReader::open_inner(path, 256, 0, PleIoMode::Direct, timeout, force_pread)
                .expect("open PLE reader")
        };
        // Deterministic uniform (shard, row) picks, like the n-gram hash.
        let selections = |seed: u64, shard_rows: &[usize], count: usize| {
            let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            (0..count)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let shard = (state as usize) % shard_rows.len();
                    let row = ((state >> 20) as usize) % shard_rows[shard];
                    (shard, row)
                })
                .collect::<Vec<_>>()
        };
        // One 16,384-token prefill chunk: 16 rows per token, 1,024-token blocks.
        const CHUNK_ROWS: usize = 16_384 * 16;
        const BLOCK_ROWS: usize = 1_024 * 16;
        let prefill = |force_pread: bool, seed: u64| {
            // A fresh reader per read keeps the record cache empty.
            let mut reader = open(force_pread);
            assert_eq!(reader.ring.is_none(), force_pread);
            let picks = selections(seed, &reader.shard_rows(), CHUNK_ROWS);
            let mut rows = Vec::with_capacity(CHUNK_ROWS);
            let started = Instant::now();
            reader
                .read_rows_streamed_blocks(&picks, BLOCK_ROWS, &mut |_, block| {
                    rows.extend(block);
                    Ok(())
                })
                .expect("streamed PLE read");
            (rows, started.elapsed())
        };
        let same = |a: &[PleNvfp4Row], b: &[PleNvfp4Row]| {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(x, y)| x.record == y.record && x.scale2.to_bits() == y.scale2.to_bits())
        };

        if open(false).ring.is_none() {
            // Inside a seccomp-restricted container io_uring is refused, so
            // there is nothing to compare against: time the fallback alone.
            let mut times = Vec::new();
            for seed in 11..14 {
                times.push(prefill(true, seed).1.as_secs_f64() * 1e3);
            }
            let (_, decode_time) = {
                let mut reader = open(true);
                let shard_rows = reader.shard_rows();
                let started = Instant::now();
                for step in 0..256u64 {
                    let picks = selections(9 * 1_000_003 + step, &shard_rows, 16);
                    reader.read_rows(&picks).expect("decode PLE read");
                }
                ((), started.elapsed())
            };
            eprintln!(
                "PLE io_uring unavailable: pread lanes 16K chunk {:?} ms; decode {:.0} us/token",
                times.iter().map(|t| t.round()).collect::<Vec<_>>(),
                decode_time.as_secs_f64() * 1e6 / 256.0
            );
            return;
        }
        let (uring_rows, _) = prefill(false, 1);
        let (pread_rows, _) = prefill(true, 1);
        assert!(same(&uring_rows, &pread_rows), "prefill rows differ");
        let (_, uring_time) = prefill(false, 2);
        let (_, pread_time) = prefill(true, 3);
        eprintln!(
            "PLE prefill 16K chunk ({CHUNK_ROWS} rows): io_uring {:.1} ms, pread lanes {:.1} ms",
            uring_time.as_secs_f64() * 1e3,
            pread_time.as_secs_f64() * 1e3
        );

        // Decode batches: 16 rows per token, fresh picks each step.
        let decode = |force_pread: bool, seed: u64, steps: u64| {
            let mut reader = open(force_pread);
            let shard_rows = reader.shard_rows();
            let mut rows = Vec::new();
            let started = Instant::now();
            for step in 0..steps {
                let picks = selections(seed * 1_000_003 + step, &shard_rows, 16);
                rows.extend(reader.read_rows(&picks).expect("decode PLE read"));
            }
            (rows, started.elapsed())
        };
        let (uring_rows, _) = decode(false, 4, 64);
        let (pread_rows, _) = decode(true, 4, 64);
        assert!(same(&uring_rows, &pread_rows), "decode rows differ");
        let serial = {
            let mut reader = open(true);
            reader.pread_pages.clear(); // the pre-fix one-lane fallback
            let shard_rows = reader.shard_rows();
            let started = Instant::now();
            for step in 0..256u64 {
                let picks = selections(7 * 1_000_003 + step, &shard_rows, 16);
                reader.read_rows(&picks).expect("serial decode PLE read");
            }
            started.elapsed()
        };
        let (_, uring_time) = decode(false, 5, 256);
        let (_, pread_time) = decode(true, 6, 256);
        eprintln!(
            "PLE decode per token (16 rows): io_uring {:.0} us, pread lanes {:.0} us, serial pread {:.0} us",
            uring_time.as_secs_f64() * 1e6 / 256.0,
            pread_time.as_secs_f64() * 1e6 / 256.0,
            serial.as_secs_f64() * 1e6 / 256.0
        );
    }

    #[test]
    fn page_packing_has_no_cross_page_rows() {
        assert_eq!(EXPECTED_RECORDS_PER_PAGE * EXPECTED_RECORD_BYTES, 8190);
        for row in 0..10_000 {
            let slot = row % EXPECTED_RECORDS_PER_PAGE;
            assert!((slot + 1) * EXPECTED_RECORD_BYTES <= EXPECTED_PAGE_BYTES);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn page_cache_mode_opens_without_direct_io() {
        use std::os::fd::AsRawFd;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "atlas-ple-page-cache-{}-{nonce}.bin",
            std::process::id()
        ));
        std::fs::write(&path, [0xA5; EXPECTED_PAGE_BYTES]).expect("write temporary PLE page");
        let file = open_shard(&path, PleIoMode::PageCache).expect("open buffered PLE page");
        // SAFETY: `file.as_raw_fd()` is a live open descriptor and `F_GETFL` takes no extra args.
        let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
        assert!(flags >= 0, "F_GETFL failed");
        assert_eq!(flags & libc::O_DIRECT, 0, "page-cache fd retained O_DIRECT");
        drop(file);
        std::fs::remove_file(path).expect("remove temporary PLE page");
    }

    fn test_record(tag: u8) -> [u8; EXPECTED_RECORD_BYTES] {
        std::array::from_fn(|i| tag.wrapping_add(i as u8))
    }

    #[test]
    fn record_cache_stays_bounded_and_evicts_oldest_after_saturation() {
        let mut cache = RecordCache::new(4, 2);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.ring_len(), 0);

        for row in 0..4u32 {
            cache.insert((0, row), test_record(row as u8));
        }
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.ring_len(), 4);
        for row in 0..4u32 {
            assert_eq!(cache.get(&(0, row)), Some(&test_record(row as u8)));
        }

        // Saturate across multiple full ring wraparounds: oldest keys evict in
        // strict FIFO order while fresh keys are immediately served as hits.
        for row in 4..12u32 {
            let evicted = row - 4;
            cache.insert((0, row), test_record(row as u8));
            assert_eq!(cache.len(), 4, "cache exceeded capacity at row {row}");
            assert_eq!(cache.ring_len(), 4, "ring exceeded capacity at row {row}");
            assert_eq!(
                cache.get(&(0, evicted)),
                None,
                "oldest key (0, {evicted}) should be evicted at row {row}"
            );
            assert_eq!(
                cache.get(&(0, row)),
                Some(&test_record(row as u8)),
                "fresh key (0, {row}) should be served as a hit"
            );
        }
        for row in 8..12u32 {
            assert_eq!(cache.get(&(0, row)), Some(&test_record(row as u8)));
        }

        // Re-inserting a previously evicted key evicts the current oldest (0, 8).
        cache.insert((0, 0), test_record(99));
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.ring_len(), 4);
        assert_eq!(cache.get(&(0, 8)), None);
        assert_eq!(cache.get(&(0, 0)), Some(&test_record(99)));
        for row in 9..12u32 {
            assert_eq!(cache.get(&(0, row)), Some(&test_record(row as u8)));
        }
    }

    #[test]
    fn record_cache_updating_existing_key_does_not_grow_ring_or_shrink_capacity() {
        let mut cache = RecordCache::new(4, 4);
        cache.insert((1, 10), test_record(1));
        cache.insert((1, 20), test_record(2));

        // Re-inserting before saturation updates in place without duplicating ring slots.
        for rev in 10..15u8 {
            cache.insert((1, 10), test_record(rev));
            assert_eq!(cache.len(), 2);
            assert_eq!(cache.ring_len(), 2);
            assert_eq!(cache.get(&(1, 10)), Some(&test_record(rev)));
        }

        // All 4 distinct slots must remain usable (effective capacity did not shrink).
        cache.insert((1, 30), test_record(3));
        cache.insert((1, 40), test_record(4));
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.ring_len(), 4);
        assert_eq!(cache.get(&(1, 10)), Some(&test_record(14)));
        assert_eq!(cache.get(&(1, 20)), Some(&test_record(2)));
        assert_eq!(cache.get(&(1, 30)), Some(&test_record(3)));
        assert_eq!(cache.get(&(1, 40)), Some(&test_record(4)));

        // Re-inserting existing keys at saturation must neither grow the ring nor evict.
        for rev in 20..25u8 {
            cache.insert((1, 10), test_record(rev));
            cache.insert((1, 30), test_record(rev.wrapping_add(1)));
            assert_eq!(cache.len(), 4);
            assert_eq!(cache.ring_len(), 4);
        }
        assert_eq!(cache.get(&(1, 10)), Some(&test_record(24)));
        assert_eq!(cache.get(&(1, 20)), Some(&test_record(2)));
        assert_eq!(cache.get(&(1, 30)), Some(&test_record(25)));
        assert_eq!(cache.get(&(1, 40)), Some(&test_record(4)));

        // Next fresh key still evicts the original oldest key (1, 10) in FIFO order.
        cache.insert((1, 50), test_record(5));
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.ring_len(), 4);
        assert_eq!(cache.get(&(1, 10)), None);
        assert_eq!(cache.get(&(1, 20)), Some(&test_record(2)));
        assert_eq!(cache.get(&(1, 30)), Some(&test_record(25)));
        assert_eq!(cache.get(&(1, 40)), Some(&test_record(4)));
        assert_eq!(cache.get(&(1, 50)), Some(&test_record(5)));
    }

    #[test]
    fn record_cache_handles_zero_and_single_entry_capacities() {
        let mut zero = RecordCache::new(0, 16);
        zero.insert((0, 1), test_record(1));
        assert!(zero.is_empty());
        assert_eq!(zero.len(), 0);
        assert_eq!(zero.ring_len(), 0);
        assert_eq!(zero.get(&(0, 1)), None);

        let mut one = RecordCache::new(1, 16);
        one.insert((0, 1), test_record(1));
        assert_eq!(one.len(), 1);
        assert_eq!(one.ring_len(), 1);
        one.insert((0, 1), test_record(2));
        assert_eq!(one.len(), 1);
        assert_eq!(one.ring_len(), 1);
        assert_eq!(one.get(&(0, 1)), Some(&test_record(2)));
        one.insert((0, 2), test_record(3));
        assert_eq!(one.len(), 1);
        assert_eq!(one.ring_len(), 1);
        assert_eq!(one.get(&(0, 1)), None);
        assert_eq!(one.get(&(0, 2)), Some(&test_record(3)));
    }

    #[test]
    fn record_cache_sizing_honors_ple_cache_mb_zero_and_override() {
        // Explicit ATLAS_PLE_CACHE_MB=0 disables the record cache completely.
        assert_eq!(resolve_record_cache_max_entries(0, None, Some("0")), 0);
        assert_eq!(resolve_record_cache_max_entries(0, None, Some(" 0 ")), 0);

        // Explicit ATLAS_PLE_RECORD_CACHE_ENTRIES always overrides ATLAS_PLE_CACHE_MB.
        assert_eq!(
            resolve_record_cache_max_entries(0, Some("4096"), Some("0")),
            4096
        );
        assert_eq!(
            resolve_record_cache_max_entries(512 * 1024 * 1024, Some("0"), Some("512")),
            0
        );

        // cache_bytes == 0 with unset env stays on the small default cap (~110 KiB).
        assert_eq!(
            resolve_record_cache_max_entries(0, None, None),
            DEFAULT_UNCACHED_RECORD_CACHE_ENTRIES
        );

        // Positive cache_bytes scales proportionally and caps at MAX_RECORD_CACHE_ENTRIES.
        assert_eq!(
            resolve_record_cache_max_entries(64 * 1024 * 1024, None, Some("64")),
            262_144
        );
        assert_eq!(
            resolve_record_cache_max_entries(1024 * 1024 * 1024, None, Some("1024")),
            MAX_RECORD_CACHE_ENTRIES
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn planned_and_ring_reads_prefilter_record_cache_and_evict_bounded() {
        use std::time::{SystemTime, UNIX_EPOCH};

        const SHARDS: usize = 2;
        const PAGES_PER_SHARD: usize = 4;
        const SCALE2: [f32; SHARDS] = [0.5, 1.25];
        let rows_per_shard = PAGES_PER_SHARD * EXPECTED_RECORDS_PER_PAGE;
        let bytes_per_shard = PAGES_PER_SHARD * EXPECTED_PAGE_BYTES;

        let byte_at = |shard: usize, offset: usize| -> u8 {
            (shard.wrapping_mul(131) ^ (offset.wrapping_mul(2_654_435_761) >> 7)) as u8
        };
        let expected_row = |shard: usize, row: usize| -> PleNvfp4Row {
            let page = row / EXPECTED_RECORDS_PER_PAGE;
            let slot = row % EXPECTED_RECORDS_PER_PAGE;
            let base = page * EXPECTED_PAGE_BYTES + slot * EXPECTED_RECORD_BYTES;
            let record = std::array::from_fn(|i| byte_at(shard, base + i));
            PleNvfp4Row {
                record,
                scale2: SCALE2[shard],
            }
        };
        let assert_exact = |got: &[PleNvfp4Row], picks: &[(usize, usize)]| {
            assert_eq!(got.len(), picks.len());
            for (idx, (row, &(shard, id))) in got.iter().zip(picks).enumerate() {
                let want = expected_row(shard, id);
                assert_eq!(row.record, want.record, "row {idx} ({shard}, {id}) record");
                assert_eq!(
                    row.scale2.to_bits(),
                    want.scale2.to_bits(),
                    "row {idx} ({shard}, {id}) scale2"
                );
            }
        };

        struct TempDirGuard(PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let dir = TempDirGuard(std::env::temp_dir().join(format!(
            "atlas-ple-record-cache-{}-{nonce}",
            std::process::id()
        )));
        std::fs::create_dir_all(&dir.0).expect("create fixture dir");
        let mut entries = Vec::with_capacity(SHARDS);
        for (shard, &scale2) in SCALE2.iter().enumerate() {
            let data: Vec<u8> = (0..bytes_per_shard)
                .map(|offset| byte_at(shard, offset))
                .collect();
            let file = format!("shard{shard}.bin");
            std::fs::write(dir.0.join(&file), data).expect("write shard");
            entries.push(serde_json::json!({
                "shard": shard,
                "file": file,
                "rows": rows_per_shard,
                "width": 160,
                "scale2": scale2,
                "bytes": bytes_per_shard,
            }));
        }
        let manifest = serde_json::json!({
            "format": EXPECTED_FORMAT,
            "page_bytes": EXPECTED_PAGE_BYTES,
            "record_bytes": EXPECTED_RECORD_BYTES,
            "records_per_page": EXPECTED_RECORDS_PER_PAGE,
            "packed_bytes": 80,
            "scale_bytes": 10,
            "group_size": 16,
            "entries": entries,
        });
        let manifest_path = dir.0.join("manifest.json");
        std::fs::write(&manifest_path, manifest.to_string()).expect("write manifest");

        for use_rings in [false, true] {
            let mut reader = PleOffloadReader::open_inner(
                &manifest_path,
                32,
                0,
                PleIoMode::Direct,
                Duration::from_secs(30),
                true,
            )
            .expect("open fixture reader");
            reader.read_plan.fast_plan = true;
            reader.read_plan.sector_ready = true;
            while reader.pread_pages.len() < 4 {
                reader.pread_pages.push(AlignedPage::new());
            }
            reader.record_cache = RecordCache::new(8, 4);

            let mode = crate::ple_plan::ReadMode::Sector;
            let read_vec =
                |r: &mut PleOffloadReader, picks: &[(usize, usize)]| -> Result<Vec<PleNvfp4Row>> {
                    if use_rings {
                        r.read_rows_streamed_rings(picks, None, None, true, mode, 2)
                    } else {
                        r.read_rows_streamed_planned(picks, None, None, true, mode, 0)
                    }
                };

            // 1) Cold read in ascending (shard, file_offset) order fills 6 entries.
            let batch1 = [(0, 0), (0, 100), (0, 200), (1, 1), (1, 101), (1, 201)];
            let got1 = read_vec(&mut reader, &batch1).expect("cold read");
            assert_exact(&got1, &batch1);
            assert_eq!(reader.record_cache.len(), 6);
            assert_eq!(reader.record_cache.ring_len(), 6);

            // 2) Partial-hit read with duplicates + 5 fresh rows saturates and evicts oldest 3.
            let batch2 = [
                (0, 200),
                (0, 10),
                (1, 201),
                (0, 110),
                (0, 10),
                (0, 210),
                (1, 11),
                (1, 111),
            ];
            let got2 = read_vec(&mut reader, &batch2).expect("partial hit + eviction read");
            assert_exact(&got2, &batch2);
            assert_eq!(reader.record_cache.len(), 8);
            assert_eq!(reader.record_cache.ring_len(), 8);
            for evicted in [(0u16, 0u32), (0, 100), (0, 200)] {
                assert!(
                    reader.record_cache.get(&evicted).is_none(),
                    "key {evicted:?} should have been evicted"
                );
            }
            for fresh in [(0u16, 10u32), (0, 110), (0, 210), (1, 11), (1, 111)] {
                assert!(
                    reader.record_cache.get(&fresh).is_some(),
                    "fresh key {fresh:?} should be cached"
                );
            }

            // 3) 100% cache-hit read via collect and visit (!collect_output).
            let all_hits = [
                (1, 1),
                (1, 101),
                (1, 201),
                (0, 10),
                (0, 110),
                (0, 210),
                (1, 11),
                (1, 111),
            ];
            let got3 = read_vec(&mut reader, &all_hits).expect("100% hit collect read");
            assert_exact(&got3, &all_hits);

            let mut visited: [Option<PleNvfp4Row>; 8] = std::array::from_fn(|_| None);
            {
                let mut visit_fn = |idx: usize, row: PleNvfp4Row| -> Result<()> {
                    assert!(visited[idx].is_none(), "selection {idx} visited twice");
                    visited[idx] = Some(row);
                    Ok(())
                };
                let visit_ref: &mut dyn FnMut(usize, PleNvfp4Row) -> Result<()> = &mut visit_fn;
                if use_rings {
                    reader
                        .read_rows_streamed_rings(&all_hits, None, Some(visit_ref), false, mode, 2)
                        .expect("100% hit ring visit");
                } else {
                    reader
                        .read_rows_streamed_planned(
                            &all_hits,
                            None,
                            Some(visit_ref),
                            false,
                            mode,
                            0,
                        )
                        .expect("100% hit planned visit");
                }
            }
            let visited_rows: Vec<PleNvfp4Row> = visited
                .into_iter()
                .map(|r| r.expect("visited row"))
                .collect();
            assert_exact(&visited_rows, &all_hits);

            // 4) Block streaming: block 0 is 100% cached, blocks 1-2 mix hits and misses.
            let mixed = [(1, 1), (0, 10), (0, 0), (1, 111), (0, 100), (1, 250)];
            let mut tracker = PleBlockTracker::new(mixed.len(), 2).expect("tracker");
            let mut streamed_blocks = Vec::new();
            let mut next_block = 0usize;
            {
                let mut sink_fn = |block: usize, rows: Vec<PleNvfp4Row>| -> Result<()> {
                    assert_eq!(block, next_block, "blocks must publish in order");
                    next_block += 1;
                    streamed_blocks.extend(rows);
                    Ok(())
                };
                let sink_ref: PleBlockSink<'_> = &mut sink_fn;
                if use_rings {
                    reader
                        .read_rows_streamed_rings(
                            &mixed,
                            Some((&mut tracker, sink_ref)),
                            None,
                            true,
                            mode,
                            2,
                        )
                        .expect("mixed hit/miss ring blocks");
                } else {
                    reader
                        .read_rows_streamed_planned(
                            &mixed,
                            Some((&mut tracker, sink_ref)),
                            None,
                            true,
                            mode,
                            0,
                        )
                        .expect("mixed hit/miss planned blocks");
                }
            }
            assert_eq!(next_block, 3);
            assert!(tracker.all_published());
            assert_exact(&streamed_blocks, &mixed);
            assert_eq!(reader.record_cache.len(), 8);
            assert_eq!(reader.record_cache.ring_len(), 8);

            // 5) Out-of-range selection with non-empty record_cache still fails cleanly.
            let err = read_vec(&mut reader, &[(0, 10), (0, rows_per_shard)])
                .expect_err("out-of-range row must fail")
                .to_string();
            assert!(err.contains("out of range"), "{err}");
        }
    }

    fn parked_payload(shard: usize, row: usize) -> [u8; EXPECTED_RECORD_BYTES] {
        let mut state = ((shard as u64).wrapping_add(1)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (row as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
            | 1;
        std::array::from_fn(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as u8).wrapping_add(i as u8)
        })
    }

    #[test]
    fn parked_cache_sizing_and_zero_disable() {
        assert_eq!(std::mem::size_of::<ParkedSlot>(), 104);

        // Explicit 0 disables allocation completely.
        assert_eq!(resolve_parked_cache_slots(Some("0")), 0);
        assert_eq!(resolve_parked_cache_slots(Some(" 0 ")), 0);
        let disabled = ParkedPleCache::with_slots(0, vec![100], vec![1.0]);
        assert!(!disabled.is_enabled());
        assert_eq!(disabled.slot_count(), 0);
        assert_eq!(disabled.byte_footprint(), 0);
        disabled.insert(0, 7, &parked_payload(0, 7));
        assert_eq!(disabled.get(0, 7), None);
        assert!(disabled.get_row(0, 7).unwrap().is_none());
        assert!(!disabled.contains(0, 7));
        assert!(disabled.try_read_all(&[(0, 7)]).unwrap().is_none());
        let mut rec = [0u8; EXPECTED_RECORD_BYTES];
        let mut sc = [0u8; 4];
        assert!(
            !disabled
                .try_fill_staging(&[(0, 7)], &mut rec, &mut sc)
                .unwrap()
        );

        // Default (128 MiB) rounds down to 2^20 = 1,048,576 slots = 104.0 MiB exact.
        assert_eq!(DEFAULT_PARKED_CACHE_MB, 128);
        assert_eq!(resolve_parked_cache_slots(None), 1_048_576);
        assert_eq!(resolve_parked_cache_slots(Some("128")), 1_048_576);
        assert_eq!(resolve_parked_cache_slots(Some("invalid")), 1_048_576);
        assert_eq!(
            1_048_576 * std::mem::size_of::<ParkedSlot>(),
            104 * 1024 * 1024
        );
        assert_eq!(resolve_parked_cache_slots(Some("256")), 2_097_152);

        // Power-of-two budget scaling and hard cap.
        assert_eq!(resolve_parked_cache_slots(Some("1")), 8_192);
        assert_eq!(resolve_parked_cache_slots(Some("16")), 131_072);
        assert_eq!(resolve_parked_cache_slots(Some("32")), 262_144);
        assert_eq!(resolve_parked_cache_slots(Some("64")), 524_288);
        assert_eq!(resolve_parked_cache_slots(Some("2048")), 16_777_216);
        assert_eq!(
            resolve_parked_cache_slots(Some("16384")),
            parked_cache_slots_for_mb(MAX_PARKED_CACHE_MB)
        );
    }

    #[test]
    fn parked_cache_hit_after_miss_returns_exact_payload_and_stays_bounded() {
        let cache = ParkedPleCache::with_slots(64, vec![10_000, 10_000], vec![0.25, 1.75]);
        assert!(cache.is_enabled());
        assert_eq!(cache.slot_count(), 64);
        assert_eq!(cache.byte_footprint(), 64 * 104);

        // Key (0, 0) must be representable (empty is encoded by seq == 0, not key == 0).
        let p0 = parked_payload(0, 0);
        let p1 = parked_payload(1, 42);
        assert_eq!(cache.get(0, 0), None);
        assert!(!cache.contains(0, 0));
        cache.insert(0, 0, &p0);
        cache.insert(1, 42, &p1);
        assert!(cache.contains(0, 0));
        assert!(cache.contains(1, 42));
        assert_eq!(cache.get(0, 0), Some(p0));
        assert_eq!(cache.get(1, 42), Some(p1));
        let row0 = cache.get_row(0, 0).expect("valid").expect("hit");
        assert_eq!(row0.record, p0);
        assert_eq!(row0.scale2.to_bits(), 0.25f32.to_bits());
        assert!(cache.get_row(0, 9999).expect("valid").is_none());
        assert!(cache.get_row(2, 0).is_err());
        assert!(cache.get_row(0, 10_000).is_err());

        // Batch read and direct-to-staging copy return byte-identical payloads & scales.
        let picks = [(0, 0), (1, 42), (0, 0)];
        let rows = cache
            .try_read_all(&picks)
            .expect("valid picks")
            .expect("all hits");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].record, p0);
        assert_eq!(rows[0].scale2.to_bits(), 0.25f32.to_bits());
        assert_eq!(rows[1].record, p1);
        assert_eq!(rows[1].scale2.to_bits(), 1.75f32.to_bits());
        assert_eq!(rows[2].record, p0);

        let mut rec_stage = vec![0u8; picks.len() * EXPECTED_RECORD_BYTES];
        let mut scale_stage = vec![0u8; picks.len() * 4];
        assert!(
            cache
                .try_fill_staging(&picks, &mut rec_stage, &mut scale_stage)
                .expect("staging fill")
        );
        assert_eq!(&rec_stage[0..90], &p0);
        assert_eq!(&rec_stage[90..180], &p1);
        assert_eq!(&rec_stage[180..270], &p0);
        assert_eq!(&scale_stage[0..4], &0.25f32.to_le_bytes());
        assert_eq!(&scale_stage[4..8], &1.75f32.to_le_bytes());
        assert_eq!(&scale_stage[8..12], &0.25f32.to_le_bytes());

        // Partial miss fast-fails cleanly.
        assert!(cache.try_read_all(&[(0, 0), (0, 9999)]).unwrap().is_none());
        let mut rec2 = vec![0u8; 2 * EXPECTED_RECORD_BYTES];
        let mut sc2 = vec![0u8; 8];
        assert!(
            !cache
                .try_fill_staging(&[(0, 0), (0, 9999)], &mut rec2, &mut sc2)
                .unwrap()
        );

        // Out-of-range shard/row fails cleanly.
        assert!(
            cache
                .try_read_all(&[(2, 0)])
                .expect_err("OOB shard")
                .to_string()
                .contains("out of range")
        );
        assert!(
            cache
                .try_read_all(&[(0, 10_000)])
                .expect_err("OOB row")
                .to_string()
                .contains("out of range")
        );

        // Heavy collision/eviction stress: 64x capacity inserts never grow slot table.
        for row in 0..4_096usize {
            let shard = row & 1;
            cache.insert(shard, row, &parked_payload(shard, row));
        }
        assert_eq!(cache.slot_count(), 64);
        assert_eq!(cache.byte_footprint(), 64 * 104);
        let mut surviving = 0usize;
        for row in 0..4_096usize {
            let shard = row & 1;
            if let Some(got) = cache.get(shard, row) {
                assert_eq!(got, parked_payload(shard, row), "corrupt surviving slot");
                surviving += 1;
            }
        }
        assert!(surviving > 0 && surviving <= 64, "surviving={surviving}");
    }

    #[test]
    fn parked_cache_concurrent_reader_writer_never_observes_torn_payload() {
        use std::sync::atomic::AtomicUsize;

        // Tiny 16-slot cache forces maximal concurrent slot collisions and evictions.
        let cache = Arc::new(ParkedPleCache::with_slots(
            16,
            vec![1_024, 1_024, 1_024, 1_024],
            vec![0.5, 1.0, 1.5, 2.0],
        ));
        let total_hits = Arc::new(AtomicUsize::new(0));

        // Pre-seed a few entries so readers immediately observe hits too.
        for key in 0..32usize {
            let shard = key & 3;
            cache.insert(shard, key, &parked_payload(shard, key));
        }

        std::thread::scope(|scope| {
            for writer_id in 0..4usize {
                let cache = Arc::clone(&cache);
                scope.spawn(move || {
                    for iter in 0..15_000usize {
                        let key = (iter.wrapping_mul(31) + writer_id * 7) & 63;
                        let shard = key & 3;
                        let payload = parked_payload(shard, key);
                        cache.insert(shard, key, &payload);
                    }
                });
            }
            for reader_id in 0..4usize {
                let cache = Arc::clone(&cache);
                let total_hits = Arc::clone(&total_hits);
                scope.spawn(move || {
                    let mut local_hits = 0usize;
                    for iter in 0..25_000usize {
                        let key = (iter.wrapping_mul(17) + reader_id * 13) & 63;
                        let shard = key & 3;
                        if let Some(got) = cache.get(shard, key) {
                            assert_eq!(
                                got,
                                parked_payload(shard, key),
                                "torn read observed for ({shard}, {key})"
                            );
                            local_hits += 1;
                        }
                        if iter.is_multiple_of(64) {
                            let picks = [(shard, key)];
                            if let Some(rows) = cache.try_read_all(&picks).expect("valid") {
                                assert_eq!(rows[0].record, parked_payload(shard, key));
                                local_hits += 1;
                            }
                        }
                    }
                    total_hits.fetch_add(local_hits, Ordering::Relaxed);
                });
            }
        });

        assert_eq!(cache.slot_count(), 16);
        assert_eq!(cache.byte_footprint(), 16 * 104);
        assert!(
            total_hits.load(Ordering::Relaxed) > 0,
            "concurrent readers should observe valid cache hits"
        );
    }

    #[test]
    fn pread_ring_tile_size_adapts_for_medium_verify_batches() {
        assert_eq!(pread_ring_tile_size(6), RING_TILE);
        assert_eq!(pread_ring_tile_size(127), RING_TILE);
        assert_eq!(pread_ring_tile_size(128), 8);
        assert_eq!(pread_ring_tile_size(640), 20);
        assert_eq!(pread_ring_tile_size(2_048), RING_TILE);
        assert_eq!(pread_ring_tile_size(262_144), RING_TILE);
    }

    #[test]
    fn io_uring_verify_threshold_and_tiles_balance_four_rings() {
        assert_eq!(VERIFY_MIN_RING_SELECTIONS, 64);
        assert_eq!(MIN_PREAD_LANE_SELECTIONS, 128);
        assert_eq!(
            resolve_io_uring_lanes(4, 4, 63, VERIFY_MIN_RING_SELECTIONS),
            None
        );
        assert_eq!(
            resolve_io_uring_lanes(4, 4, 64, VERIFY_MIN_RING_SELECTIONS),
            Some(4)
        );
        assert_eq!(
            resolve_io_uring_lanes(4, 4, 96, VERIFY_MIN_RING_SELECTIONS),
            Some(4)
        );
        assert_eq!(
            resolve_io_uring_lanes(4, 4, 256, VERIFY_MIN_RING_SELECTIONS),
            Some(4)
        );
        assert_eq!(
            resolve_io_uring_lanes(1, 4, 256, VERIFY_MIN_RING_SELECTIONS),
            None
        );

        assert_eq!(io_uring_tile_size(64, 4), 16);
        assert_eq!(io_uring_tile_size(96, 4), 24);
        assert_eq!(io_uring_tile_size(160, 4), 40);
        assert_eq!(io_uring_tile_size(256, 4), RING_TILE);
        assert_eq!(io_uring_tile_size(262_144, 4), RING_TILE);

        // Verify exact 1-to-1 partition across 4 rings for every batch size.
        for reads in [1usize, 2, 17, 64, 96, 97, 128, 160, 255, 256, 300] {
            let rings = 4usize;
            let tile = io_uring_tile_size(reads, rings);
            let mut seen = vec![0usize; reads];
            for ring_index in 0..rings {
                let mut tile_start = ring_index * tile;
                while tile_start < reads {
                    let tile_end = (tile_start + tile).min(reads);
                    for (offset, count) in seen[tile_start..tile_end].iter_mut().enumerate() {
                        let index = tile_start + offset;
                        assert_eq!(
                            ring_tile_of(index, rings, tile),
                            ring_index,
                            "reads={reads} tile={tile} index={index}"
                        );
                        *count += 1;
                    }
                    tile_start += rings * tile;
                }
            }
            assert!(
                seen.iter().all(|&c| c == 1),
                "reads={reads} tile={tile} partition mismatch: {seen:?}"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn persistent_ring_worker_pool_reuses_threads_and_restores_primary_lane() {
        use std::collections::HashSet;
        use std::time::{SystemTime, UNIX_EPOCH};

        const SHARDS: usize = 2;
        const PAGES_PER_SHARD: usize = 4;
        const SCALE2: [f32; SHARDS] = [0.75, 1.5];
        let rows_per_shard = PAGES_PER_SHARD * EXPECTED_RECORDS_PER_PAGE;
        let bytes_per_shard = PAGES_PER_SHARD * EXPECTED_PAGE_BYTES;

        let byte_at = |shard: usize, offset: usize| -> u8 {
            (shard.wrapping_mul(197) ^ (offset.wrapping_mul(2_654_435_761) >> 5)) as u8
        };
        let expected_row = |shard: usize, row: usize| -> PleNvfp4Row {
            let page = row / EXPECTED_RECORDS_PER_PAGE;
            let slot = row % EXPECTED_RECORDS_PER_PAGE;
            let base = page * EXPECTED_PAGE_BYTES + slot * EXPECTED_RECORD_BYTES;
            PleNvfp4Row {
                record: std::array::from_fn(|i| byte_at(shard, base + i)),
                scale2: SCALE2[shard],
            }
        };

        struct TempDirGuard(PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let dir = TempDirGuard(std::env::temp_dir().join(format!(
            "atlas-ple-ring-pool-{}-{nonce}",
            std::process::id()
        )));
        std::fs::create_dir_all(&dir.0).expect("create fixture dir");

        let mut shard_vec = Vec::with_capacity(SHARDS);
        for (shard, &scale2) in SCALE2.iter().enumerate() {
            let data: Vec<u8> = (0..bytes_per_shard)
                .map(|offset| byte_at(shard, offset))
                .collect();
            let path = dir.0.join(format!("shard{shard}.bin"));
            std::fs::write(&path, data).expect("write shard");
            let file = open_shard(&path, PleIoMode::Direct).expect("open O_DIRECT shard");
            shard_vec.push(Shard {
                file,
                rows: rows_per_shard,
                scale2,
            });
        }
        let shards: Arc<[Shard]> = shard_vec.into();
        let shard_rows = vec![rows_per_shard; SHARDS];
        let extra_rings: Vec<ReadRing> = (0..3)
            .map(|_| ReadRing {
                ring: None,
                pages: vec![AlignedPage::new()],
                fixed_buffers: false,
            })
            .collect();
        let pool = RingWorkerPool::new(extra_rings, Arc::clone(&shards)).expect("spawn ring pool");
        assert_eq!(pool.ring_count(), 4);
        let initial_ids = pool.worker_thread_ids();
        assert_eq!(initial_ids.len(), 4);
        assert_eq!(
            initial_ids.iter().copied().collect::<HashSet<_>>().len(),
            4,
            "all 4 persistent ring workers must run on distinct OS threads"
        );

        let mut primary = ReadRing {
            ring: None,
            pages: vec![AlignedPage::new()],
            fixed_buffers: false,
        };
        let timeout = Duration::from_secs(30);

        for (round, (active_rings, count)) in
            [(2usize, 2usize), (4, 24), (4, 96)].into_iter().enumerate()
        {
            let picks: Vec<(usize, usize)> = (0..count)
                .map(|i| ((i + round) % SHARDS, (i * 7 + round) % rows_per_shard))
                .collect();
            let plan = crate::ple_plan::plan_fast(
                &picks,
                &shard_rows,
                &SCALE2,
                crate::ple_plan::ReadGranularity::Sector,
            )
            .expect("plan");
            let use_rings = active_rings.min(plan.reads());
            let abort = Arc::new(AtomicBool::new(false));
            let (rx, dispatched, dispatch_err) = pool.dispatch(
                primary,
                Arc::new(plan),
                use_rings,
                Instant::now() + timeout,
                timeout,
                abort,
            );
            assert!(dispatch_err.is_none(), "round {round} dispatch failed");
            assert_eq!(dispatched, use_rings);

            let mut out: Vec<Option<PleNvfp4Row>> = vec![None; picks.len()];
            let mut done = 0usize;
            for msg in rx.iter() {
                match msg {
                    LaneMsg::Rows(rows) => {
                        for (idx, row) in rows {
                            let idx = idx as usize;
                            assert!(out[idx].is_none(), "duplicate fill at {idx}");
                            out[idx] = Some(row);
                        }
                    }
                    LaneMsg::Done(res) => {
                        res.expect("ring worker completed cleanly");
                        done += 1;
                    }
                }
            }
            assert_eq!(done, dispatched);
            primary = pool
                .take_returned_primary()
                .expect("primary lane must be restored after every dispatch");
            assert_eq!(primary.pages.len(), 1);
            assert_eq!(
                pool.worker_thread_ids(),
                initial_ids,
                "persistent pool must reuse the exact same worker threads across calls"
            );
            for (idx, (got, &(shard, row))) in out.into_iter().zip(&picks).enumerate() {
                let got = got.expect("row filled");
                let want = expected_row(shard, row);
                assert_eq!(got.record, want.record, "round {round} row {idx}");
                assert_eq!(got.scale2.to_bits(), want.scale2.to_bits());
            }
        }

        // Pre-aborted dispatch returns Err cleanly AND still restores primary lane.
        let picks = vec![(0usize, 1usize), (1, 2), (0, 100), (1, 200)];
        let plan = crate::ple_plan::plan_fast(
            &picks,
            &shard_rows,
            &SCALE2,
            crate::ple_plan::ReadGranularity::Sector,
        )
        .expect("abort plan");
        let abort = Arc::new(AtomicBool::new(true));
        let (rx, dispatched, dispatch_err) = pool.dispatch(
            primary,
            Arc::new(plan),
            4,
            Instant::now() + timeout,
            timeout,
            abort,
        );
        assert!(dispatch_err.is_none());
        assert_eq!(dispatched, 4);
        let mut saw_err = false;
        let mut done = 0usize;
        for msg in rx.iter() {
            if let LaneMsg::Done(res) = msg {
                done += 1;
                if res.is_err() {
                    saw_err = true;
                }
            }
        }
        assert_eq!(done, 4);
        assert!(saw_err, "aborted dispatch must propagate worker error");
        let restored = pool
            .take_returned_primary()
            .expect("primary lane must be restored even after abort");
        assert_eq!(restored.pages.len(), 1);
        assert_eq!(pool.worker_thread_ids(), initial_ids);

        // Clean Drop joins all persistent ring workers.
        drop(pool);
    }

    #[test]
    fn pread_worker_pool_reuses_threads_and_page_cache_planned_reads_match_exact() {
        use std::time::{SystemTime, UNIX_EPOCH};

        struct TempDirGuard(PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        const SHARDS: usize = 2;
        const SCALE2: [f32; SHARDS] = [0.5, 1.25];
        let rows_per_shard = 400usize;
        let pages_per_shard = rows_per_shard.div_ceil(EXPECTED_RECORDS_PER_PAGE);
        let bytes_per_shard = pages_per_shard * EXPECTED_PAGE_BYTES;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let dir = TempDirGuard(std::env::temp_dir().join(format!(
            "atlas-ple-pread-pool-{}-{nonce}",
            std::process::id()
        )));
        std::fs::create_dir_all(&dir.0).expect("create temp dir");

        let mut entries_json = Vec::new();
        for (shard, &scale2) in SCALE2.iter().enumerate() {
            let mut data = vec![0u8; bytes_per_shard];
            for row in 0..rows_per_shard {
                let page = row / EXPECTED_RECORDS_PER_PAGE;
                let slot = row % EXPECTED_RECORDS_PER_PAGE;
                let offset = page * EXPECTED_PAGE_BYTES + slot * EXPECTED_RECORD_BYTES;
                let rec = parked_payload(shard, row);
                data[offset..offset + EXPECTED_RECORD_BYTES].copy_from_slice(&rec);
            }
            let fname = format!("ple-ngram-nvfp4-{shard:03}.bin");
            std::fs::write(dir.0.join(&fname), &data).expect("write shard");
            entries_json.push(format!(
                r#"{{"shard":{shard},"file":"{fname}","rows":{rows_per_shard},"width":160,"scale2":{scale2},"bytes":{bytes_per_shard}}}"#
            ));
        }
        let manifest_path = dir.0.join("manifest.json");
        let manifest = format!(
            r#"{{"format":"{EXPECTED_FORMAT}","page_bytes":{EXPECTED_PAGE_BYTES},"record_bytes":{EXPECTED_RECORD_BYTES},"records_per_page":{EXPECTED_RECORDS_PER_PAGE},"packed_bytes":80,"scale_bytes":10,"group_size":16,"entries":[{}]}}"#,
            entries_json.join(",")
        );
        std::fs::write(&manifest_path, manifest).expect("write manifest");

        let timeout = Duration::from_secs(10);
        let mut reader = PleOffloadReader::open_inner(
            &manifest_path,
            32,
            0,
            PleIoMode::PageCache,
            timeout,
            true,
        )
        .expect("open PageCache pread-pool reader");
        assert!(!reader.has_io_uring());
        // Enable fast_plan + sector mode directly to avoid mutating process env across parallel tests.
        reader.read_plan.fast_plan = true;
        reader.read_plan.sector_ready = true;
        reader.read_plan.rings = 4;

        let pool = reader
            .pread_pool
            .as_ref()
            .expect("persistent pread pool must initialize when io_uring is absent");
        let initial_ids = pool.worker_thread_ids();
        assert!(initial_ids.len() >= 16);

        // Small verify batch (80 rows > queue_depth=32, < MIN_PREAD_LANE_SELECTIONS=128)
        // runs through persistent pread_pool without queue_depth=32 serial fallback.
        let small_picks: Vec<(usize, usize)> = (0..80).map(|i| (i % 2, (i * 4) % 400)).collect();
        let small_rows = reader
            .read_rows_verify(&small_picks)
            .expect("small verify read via persistent pread pool");
        assert_eq!(small_rows.len(), small_picks.len());
        for (&(shard, row), got) in small_picks.iter().zip(&small_rows) {
            assert_eq!(got.record, parked_payload(shard, row));
            assert_eq!(got.scale2, SCALE2[shard]);
        }

        // Large verify batch (200 rows >= MIN_PREAD_LANE_SELECTIONS) reuses the exact
        // same persistent worker threads with exact byte parity.
        let large_picks: Vec<(usize, usize)> = (0..200).map(|i| (i % 2, (i * 2) % 400)).collect();
        let large_rows = reader
            .read_rows_verify(&large_picks)
            .expect("large verify read via persistent pread pool");
        assert_eq!(large_rows.len(), large_picks.len());
        for (&(shard, row), got) in large_picks.iter().zip(&large_rows) {
            assert_eq!(got.record, parked_payload(shard, row));
            assert_eq!(got.scale2, SCALE2[shard]);
        }

        let after_ids = reader
            .pread_pool
            .as_ref()
            .expect("pool intact")
            .worker_thread_ids();
        assert_eq!(
            after_ids, initial_ids,
            "persistent pread workers must not be respawned across verify batches"
        );

        // Verify ReadPlanConfig::from_env admits PageCache mode with cache_pages=0.
        let plan_cfg = ReadPlanConfig::from_env(PleIoMode::PageCache, 0, &reader.shards)
            .expect("PageCache ReadPlanConfig::from_env must succeed");
        let _ = plan_cfg;
    }

    #[test]
    fn ple_mmap_policy_and_fs_classification() {
        assert_eq!(parse_ple_mmap_policy(None).unwrap(), PleMmapPolicy::Auto);
        assert_eq!(
            parse_ple_mmap_policy(Some("")).unwrap(),
            PleMmapPolicy::Auto
        );
        assert_eq!(
            parse_ple_mmap_policy(Some("1")).unwrap(),
            PleMmapPolicy::ForceOn
        );
        assert_eq!(
            parse_ple_mmap_policy(Some(" 1 ")).unwrap(),
            PleMmapPolicy::ForceOn
        );
        assert_eq!(
            parse_ple_mmap_policy(Some("0")).unwrap(),
            PleMmapPolicy::ForceOff
        );
        assert_eq!(
            parse_ple_mmap_policy(Some(" 0 ")).unwrap(),
            PleMmapPolicy::ForceOff
        );
        for bad in ["2", "true", "false", "auto", "-1"] {
            assert!(
                parse_ple_mmap_policy(Some(bad)).is_err(),
                "accepted invalid {PLE_MMAP_SELECTOR}={bad:?}"
            );
        }

        assert_eq!(
            resolve_ple_mmap_trigger(PleMmapPolicy::ForceOn, None),
            Some("env")
        );
        assert_eq!(
            resolve_ple_mmap_trigger(PleMmapPolicy::ForceOn, Some(InMemoryFs::Tmpfs)),
            Some("env")
        );
        assert_eq!(
            resolve_ple_mmap_trigger(PleMmapPolicy::ForceOff, Some(InMemoryFs::Tmpfs)),
            None
        );
        assert_eq!(
            resolve_ple_mmap_trigger(PleMmapPolicy::ForceOff, Some(InMemoryFs::Ramfs)),
            None
        );
        assert_eq!(
            resolve_ple_mmap_trigger(PleMmapPolicy::Auto, Some(InMemoryFs::Tmpfs)),
            Some("tmpfs")
        );
        assert_eq!(
            resolve_ple_mmap_trigger(PleMmapPolicy::Auto, Some(InMemoryFs::Ramfs)),
            Some("ramfs")
        );
        assert_eq!(resolve_ple_mmap_trigger(PleMmapPolicy::Auto, None), None);

        #[cfg(target_os = "linux")]
        {
            assert_eq!(
                classify_fs_magic(libc::TMPFS_MAGIC),
                Some(InMemoryFs::Tmpfs)
            );
            assert_eq!(
                classify_fs_magic(LINUX_RAMFS_MAGIC),
                Some(InMemoryFs::Ramfs)
            );
            assert_eq!(classify_fs_magic(0xEF53), None);
        }

        assert_eq!(mmap_copy_threads(1), 1);
        assert_eq!(mmap_copy_threads(MMAP_PARALLEL_MIN_SELECTIONS - 1), 1);
        assert!(mmap_copy_threads(MMAP_PARALLEL_MIN_SELECTIONS) >= 1);
        assert!(mmap_copy_threads(262_144) <= MMAP_MAX_COPY_THREADS);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mmap_mode_matches_pread_bit_for_bit_across_decode_and_streamed_paths() {
        use std::time::{SystemTime, UNIX_EPOCH};

        struct TempDirGuard(PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        const SHARDS: usize = 3;
        const PAGES_PER_SHARD: usize = 6;
        const SCALE2: [f32; SHARDS] = [0.25, 1.125, 2.75];
        let rows_per_shard = PAGES_PER_SHARD * EXPECTED_RECORDS_PER_PAGE;
        let bytes_per_shard = PAGES_PER_SHARD * EXPECTED_PAGE_BYTES;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let dir = TempDirGuard(std::env::temp_dir().join(format!(
            "atlas-ple-mmap-parity-{}-{nonce}",
            std::process::id()
        )));
        std::fs::create_dir_all(&dir.0).expect("create fixture dir");

        let mut entries = Vec::with_capacity(SHARDS);
        for (shard, &scale2) in SCALE2.iter().enumerate() {
            let mut data = vec![0xEEu8; bytes_per_shard];
            for row in 0..rows_per_shard {
                let page = row / EXPECTED_RECORDS_PER_PAGE;
                let slot = row % EXPECTED_RECORDS_PER_PAGE;
                let offset = page * EXPECTED_PAGE_BYTES + slot * EXPECTED_RECORD_BYTES;
                let rec = parked_payload(shard, row);
                data[offset..offset + EXPECTED_RECORD_BYTES].copy_from_slice(&rec);
            }
            let file = format!("shard-{shard}.bin");
            std::fs::write(dir.0.join(&file), &data).expect("write synthetic shard");
            entries.push(serde_json::json!({
                "shard": shard,
                "file": file,
                "rows": rows_per_shard,
                "width": 160,
                "scale2": scale2,
                "bytes": bytes_per_shard,
            }));
        }
        let manifest_path = dir.0.join("manifest.json");
        let manifest = serde_json::json!({
            "format": EXPECTED_FORMAT,
            "page_bytes": EXPECTED_PAGE_BYTES,
            "record_bytes": EXPECTED_RECORD_BYTES,
            "records_per_page": EXPECTED_RECORDS_PER_PAGE,
            "packed_bytes": 80,
            "scale_bytes": 10,
            "group_size": 16,
            "entries": entries,
        });
        std::fs::write(&manifest_path, manifest.to_string()).expect("write manifest");

        let timeout = Duration::from_secs(30);
        let mut pread_reader = PleOffloadReader::open_inner_with_mmap(
            &manifest_path,
            32,
            0,
            PleIoMode::PageCache,
            timeout,
            true,
            PleMmapPolicy::ForceOff,
        )
        .expect("open pread reader");
        let mut mmap_reader = PleOffloadReader::open_inner_with_mmap(
            &manifest_path,
            32,
            0,
            PleIoMode::Direct,
            timeout,
            true,
            PleMmapPolicy::ForceOn,
        )
        .expect("open mmap reader");

        assert!(!pread_reader.has_mmap());
        assert!(mmap_reader.has_mmap());
        assert!(!mmap_reader.has_io_uring());
        assert!(mmap_reader.pread_pool.is_none());
        assert!(mmap_reader.pages.is_empty());

        let assert_identical = |label: &str, a: &[PleNvfp4Row], b: &[PleNvfp4Row]| {
            assert_eq!(a.len(), b.len(), "{label}: length mismatch");
            for (idx, (lhs, rhs)) in a.iter().zip(b).enumerate() {
                assert_eq!(lhs.record, rhs.record, "{label}: row {idx} bytes differ");
                assert_eq!(
                    lhs.scale2.to_bits(),
                    rhs.scale2.to_bits(),
                    "{label}: row {idx} scale2 differs"
                );
            }
        };

        // 1) Decode batch covering slot 0, slot 45 (4 KiB straddle), slot 90 (page end),
        //    multi-shard picks, and duplicates.
        let decode_picks = vec![
            (0, 0),
            (0, 45),
            (0, 90),
            (0, 91),
            (1, 45),
            (1, rows_per_shard - 1),
            (2, 0),
            (2, 181),
            (0, 45),
            (2, rows_per_shard - 1),
            (1, 0),
            (1, 90),
            (0, 272),
            (2, 364),
            (1, 455),
            (0, rows_per_shard - 1),
        ];
        let pread_decode = pread_reader.read_rows(&decode_picks).expect("pread decode");
        let mmap_decode = mmap_reader.read_rows(&decode_picks).expect("mmap decode");
        assert_identical("read_rows", &pread_decode, &mmap_decode);

        // 2) Windowed and verify paths (cold + warm cache).
        let pread_win = pread_reader
            .read_rows_windowed(&decode_picks)
            .expect("pread windowed");
        let mmap_win = mmap_reader
            .read_rows_windowed(&decode_picks)
            .expect("mmap windowed");
        assert_identical("read_rows_windowed", &pread_win, &mmap_win);

        let verify_picks: Vec<(usize, usize)> = (0..80)
            .map(|i| (i % SHARDS, (i * 19 + 3) % rows_per_shard))
            .collect();
        let pread_ver = pread_reader
            .read_rows_verify(&verify_picks)
            .expect("pread verify");
        let mmap_ver = mmap_reader
            .read_rows_verify(&verify_picks)
            .expect("mmap verify");
        assert_identical("read_rows_verify", &pread_ver, &mmap_ver);

        // 3) Large streamed prefill batch (>= MMAP_PARALLEL_MIN_SELECTIONS = 1024)
        //    exercising multi-threaded parallel memcpy in read_rows_streamed,
        //    visit_rows_streamed, read_rows_streamed_blocks (both small and large
        //    block sizes), read_rows_streamed_planned, and read_rows_streamed_rings.
        let large_picks: Vec<(usize, usize)> = (0..3_072)
            .map(|i| ((i * 7) % SHARDS, (i * 31 + 11) % rows_per_shard))
            .collect();
        let pread_streamed = pread_reader
            .read_rows_streamed(&large_picks)
            .expect("pread streamed");
        let mmap_streamed = mmap_reader
            .read_rows_streamed(&large_picks)
            .expect("mmap streamed");
        assert_identical("read_rows_streamed(3072)", &pread_streamed, &mmap_streamed);

        let mut visited: Vec<Option<PleNvfp4Row>> = vec![None; large_picks.len()];
        mmap_reader
            .visit_rows_streamed(&large_picks, &mut |idx, row| {
                assert!(visited[idx].is_none(), "duplicate visit {idx}");
                visited[idx] = Some(row);
                Ok(())
            })
            .expect("mmap visit_rows_streamed");
        let visited_rows: Vec<PleNvfp4Row> = visited
            .into_iter()
            .map(|r| r.expect("visited row"))
            .collect();
        assert_identical("visit_rows_streamed(3072)", &pread_streamed, &visited_rows);

        for block_size in [128usize, 1_024usize] {
            let mut got_blocks = Vec::with_capacity(large_picks.len());
            let mut expected_block = 0usize;
            mmap_reader
                .read_rows_streamed_blocks(&large_picks, block_size, &mut |block, rows| {
                    assert_eq!(block, expected_block, "block_size={block_size} order");
                    expected_block += 1;
                    got_blocks.extend(rows);
                    Ok(())
                })
                .expect("mmap read_rows_streamed_blocks");
            assert_eq!(expected_block, large_picks.len().div_ceil(block_size));
            assert_identical(
                &format!("read_rows_streamed_blocks({block_size})"),
                &pread_streamed,
                &got_blocks,
            );
        }

        let mmap_planned = mmap_reader
            .read_rows_streamed_planned(
                &large_picks,
                None,
                None,
                true,
                crate::ple_plan::ReadMode::Sector,
                0,
            )
            .expect("mmap planned");
        assert_identical("read_rows_streamed_planned", &pread_streamed, &mmap_planned);

        let mmap_rings = mmap_reader
            .read_rows_streamed_rings(
                &large_picks,
                None,
                None,
                true,
                crate::ple_plan::ReadMode::Sector,
                4,
            )
            .expect("mmap rings");
        assert_identical("read_rows_streamed_rings", &pread_streamed, &mmap_rings);

        // 4) Out-of-range shard/row fails cleanly before publishing any blocks.
        let mut published_on_error = 0usize;
        let bad = vec![(0, 1), (0, rows_per_shard)];
        let err = mmap_reader
            .read_rows_streamed_blocks(&bad, 1, &mut |_, _| {
                published_on_error += 1;
                Ok(())
            })
            .expect_err("OOB selection must fail");
        assert!(err.to_string().contains("out of range"), "{err}");
        assert_eq!(published_on_error, 0, "must not publish blocks on OOB");

        // 5) Drop unmaps cleanly without panicking.
        drop(mmap_reader);
        drop(pread_reader);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mmap_auto_engages_on_tmpfs_and_respects_force_off() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let shm = Path::new("/dev/shm");
        if detect_in_memory_fs(shm) != Some(InMemoryFs::Tmpfs) {
            return;
        }

        struct TempDirGuard(PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let dir = TempDirGuard(shm.join(format!(
            "atlas-ple-tmpfs-auto-{}-{nonce}",
            std::process::id()
        )));
        if std::fs::create_dir_all(&dir.0).is_err() {
            return;
        }

        let rows = EXPECTED_RECORDS_PER_PAGE * 2;
        let bytes = EXPECTED_PAGE_BYTES * 2;
        let mut data = vec![0u8; bytes];
        for row in 0..rows {
            let page = row / EXPECTED_RECORDS_PER_PAGE;
            let slot = row % EXPECTED_RECORDS_PER_PAGE;
            let offset = page * EXPECTED_PAGE_BYTES + slot * EXPECTED_RECORD_BYTES;
            data[offset..offset + EXPECTED_RECORD_BYTES].copy_from_slice(&parked_payload(0, row));
        }
        std::fs::write(dir.0.join("shard0.bin"), &data).expect("write tmpfs shard");
        let manifest_path = dir.0.join("manifest.json");
        let manifest = serde_json::json!({
            "format": EXPECTED_FORMAT,
            "page_bytes": EXPECTED_PAGE_BYTES,
            "record_bytes": EXPECTED_RECORD_BYTES,
            "records_per_page": EXPECTED_RECORDS_PER_PAGE,
            "packed_bytes": 80,
            "scale_bytes": 10,
            "group_size": 16,
            "entries": [{
                "shard": 0,
                "file": "shard0.bin",
                "rows": rows,
                "width": 160,
                "scale2": 0.75,
                "bytes": bytes,
            }],
        });
        std::fs::write(&manifest_path, manifest.to_string()).expect("write tmpfs manifest");

        let timeout = Duration::from_secs(10);
        // Auto mode on tmpfs engages mmap automatically even if Direct was requested.
        let mut auto_reader = PleOffloadReader::open_inner_with_mmap(
            &manifest_path,
            32,
            0,
            PleIoMode::Direct,
            timeout,
            true,
            PleMmapPolicy::Auto,
        )
        .expect("auto mmap reader on tmpfs");
        assert!(auto_reader.has_mmap());

        // ForceOff disables mmap even on tmpfs (uses PageCache pread).
        let mut off_reader = PleOffloadReader::open_inner_with_mmap(
            &manifest_path,
            32,
            0,
            PleIoMode::PageCache,
            timeout,
            true,
            PleMmapPolicy::ForceOff,
        )
        .expect("force-off reader on tmpfs");
        assert!(!off_reader.has_mmap());

        let picks = [(0, 0), (0, 45), (0, 90), (0, 91), (0, rows - 1)];
        let a = auto_reader.read_rows(&picks).expect("auto tmpfs read");
        let b = off_reader.read_rows(&picks).expect("pread tmpfs read");
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.record, y.record);
            assert_eq!(x.scale2.to_bits(), y.scale2.to_bits());
        }
    }
}
