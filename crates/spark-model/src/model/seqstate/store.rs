// SPDX-License-Identifier: AGPL-3.0-only

//! Conversation-state cache: `SeqStateBlob`s by exact token prefix, in a RAM
//! tier bounded by a MemAvailable margin and an optional disk tier bounded by
//! size, free space, and a write budget.
//!
//! A lookup returns the longest saved prefix of a prompt, compared token by
//! token. On GB10 the GPU and the host share one pool, so the RAM tier never
//! allocates below the margin and gives memory back (to disk, else dropped)
//! when other tenants grow: a slower resume, never a host OOM.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use parking_lot::Mutex;
use ring::digest::{Context as DigestContext, SHA256};

use super::format::{BlobReader, DIGEST_BYTES};

/// MemAvailable source, injectable for tests.
pub trait MemProbe: Send + Sync {
    fn mem_available(&self) -> Result<u64>;
}

/// `/proc/meminfo` MemAvailable.
pub struct ProcMeminfo;

impl MemProbe for ProcMeminfo {
    fn mem_available(&self) -> Result<u64> {
        let text = fs::read_to_string("/proc/meminfo").context("reading /proc/meminfo")?;
        parse_mem_available(&text)
    }
}

pub(crate) fn parse_mem_available(meminfo: &str) -> Result<u64> {
    let line = meminfo
        .lines()
        .find(|l| l.starts_with("MemAvailable:"))
        .context("no MemAvailable line in /proc/meminfo")?;
    let kib: u64 = line
        .split_whitespace()
        .nth(1)
        .context("malformed MemAvailable line")?
        .parse()
        .context("malformed MemAvailable value")?;
    kib.checked_mul(1024).context("MemAvailable overflows")
}

#[derive(Clone, Debug)]
pub struct StoreConfig {
    /// The RAM tier keeps MemAvailable at or above this.
    pub ram_margin_bytes: u64,
    /// Below the margin, free down to margin + this (hysteresis).
    pub ram_release_slack_bytes: u64,
    pub disk_dir: Option<PathBuf>,
    pub disk_cap_bytes: u64,
    /// A disk write never leaves less free space than this.
    pub disk_free_min_bytes: u64,
    /// Token bucket refill; 0 = unlimited.
    pub write_bytes_per_hour: u64,
}

impl Default for StoreConfig {
    fn default() -> Self {
        const GIB: u64 = 1 << 30;
        Self {
            ram_margin_bytes: 4 * GIB,
            ram_release_slack_bytes: GIB,
            disk_dir: None,
            disk_cap_bytes: 128 * GIB,
            disk_free_min_bytes: 32 * GIB,
            write_bytes_per_hour: 32 * GIB,
        }
    }
}

/// Who may resume an entry: lookups never cross tenants or model configs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Scope {
    pub tenant: u64,
    pub fingerprint: [u8; DIGEST_BYTES],
}

#[derive(Clone, Debug)]
pub enum Payload {
    Ram(Arc<Vec<u8>>),
    Disk(PathBuf),
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub tokens: usize,
    pub payload: Payload,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// An entry left RAM for the disk tier.
    Demoted { tokens: usize, bytes: u64 },
    /// An entry left RAM and could not be written (budget, space or no disk).
    Dropped {
        tokens: usize,
        bytes: u64,
        reason: &'static str,
    },
    /// A put was refused: keeping it would breach the margin.
    Refused { tokens: usize, bytes: u64 },
}

struct Entry {
    scope: Scope,
    tokens: Vec<u32>,
    bytes: u64,
    ram: Option<Arc<Vec<u8>>>,
    disk: Option<PathBuf>,
    last_used: u64,
    generation: u64,
    demoting: bool,
}

struct Bucket {
    tokens: f64,
    capacity: f64,
    per_sec: f64,
    at: Instant,
}

impl Bucket {
    fn new(per_hour: u64, now: Instant) -> Option<Self> {
        (per_hour > 0).then(|| Self {
            tokens: per_hour as f64,
            capacity: per_hour as f64,
            per_sec: per_hour as f64 / 3600.0,
            at: now,
        })
    }
    fn take(&mut self, bytes: u64, now: Instant) -> bool {
        let dt = now.saturating_duration_since(self.at).as_secs_f64();
        self.tokens = (self.tokens + dt * self.per_sec).min(self.capacity);
        self.at = now;
        if self.tokens >= bytes as f64 {
            self.tokens -= bytes as f64;
            true
        } else {
            false
        }
    }
}

struct Inner {
    entries: HashMap<[u8; DIGEST_BYTES], Entry>,
    ram_bytes: u64,
    disk_bytes: u64,
    clock: u64,
    next_gen: u64,
    bucket: Option<Bucket>,
}

static TMP_FILE_SEQ: AtomicU64 = AtomicU64::new(0);

pub struct StateStore {
    cfg: StoreConfig,
    probe: Box<dyn MemProbe>,
    inner: Mutex<Inner>,
}

fn entry_id(scope: &Scope, tokens: &[u32]) -> [u8; DIGEST_BYTES] {
    let mut ctx = DigestContext::new(&SHA256);
    ctx.update(&scope.tenant.to_le_bytes());
    ctx.update(&scope.fingerprint);
    ctx.update(&(tokens.len() as u64).to_le_bytes());
    for t in tokens {
        ctx.update(&t.to_le_bytes());
    }
    let mut out = [0u8; DIGEST_BYTES];
    out.copy_from_slice(ctx.finish().as_ref());
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

const FILE_SUFFIX: &str = "fss";

impl StateStore {
    pub fn new(cfg: StoreConfig, probe: Box<dyn MemProbe>) -> Result<Self> {
        if let Some(dir) = &cfg.disk_dir {
            fs::create_dir_all(dir)
                .with_context(|| format!("creating prompt-cache dir {}", dir.display()))?;
        }
        let bucket = Bucket::new(cfg.write_bytes_per_hour, Instant::now());
        Ok(Self {
            cfg,
            probe,
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                ram_bytes: 0,
                disk_bytes: 0,
                clock: 0,
                next_gen: 0,
                bucket,
            }),
        })
    }

    /// Re-indexes the disk tier: keeps files written for `fingerprint`
    /// (any tenant, recorded in the file name), deletes the rest and any
    /// half-written temporary.
    pub fn load_disk(&self, fingerprint: [u8; DIGEST_BYTES]) -> Result<usize> {
        let Some(dir) = self.cfg.disk_dir.clone() else {
            return Ok(0);
        };
        let mut kept = 0;
        for item in fs::read_dir(&dir)? {
            let path = item?.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()).map(str::to_owned) else {
                continue;
            };
            if name.ends_with(".tmp") {
                let _ = fs::remove_file(&path);
                continue;
            }
            let Some(stem) = name.strip_suffix(&format!(".{FILE_SUFFIX}")) else {
                continue;
            };
            let tenant = stem
                .split('-')
                .next()
                .and_then(|t| u64::from_str_radix(t, 16).ok());
            let header = fs::File::open(&path)
                .map_err(anyhow::Error::from)
                .and_then(|mut f| BlobReader::open(&mut f).map(|b| b.header().clone()));
            match (tenant, header) {
                (Some(tenant), Ok(h)) if h.fingerprint == fingerprint => {
                    let bytes = fs::metadata(&path)?.len();
                    let scope = Scope {
                        tenant,
                        fingerprint,
                    };
                    let id = entry_id(&scope, &h.tokens);
                    let mut inner = self.inner.lock();
                    inner.clock += 1;
                    let last_used = inner.clock;
                    inner.next_gen += 1;
                    let generation = inner.next_gen;
                    if let Some(prev) = inner.entries.remove(&id) {
                        if prev.ram.is_some() {
                            inner.ram_bytes = inner.ram_bytes.saturating_sub(prev.bytes);
                        }
                        if prev.disk.is_some() {
                            inner.disk_bytes = inner.disk_bytes.saturating_sub(prev.bytes);
                        }
                    }
                    inner.disk_bytes += bytes;
                    inner.entries.insert(
                        id,
                        Entry {
                            scope,
                            tokens: h.tokens,
                            bytes,
                            ram: None,
                            disk: Some(path),
                            last_used,
                            generation,
                            demoting: false,
                        },
                    );
                    kept += 1;
                }
                _ => {
                    let _ = fs::remove_file(&path);
                }
            }
        }
        Ok(kept)
    }

    /// Longest saved prefix of `prompt` within `scope`. Marks it used.
    pub fn lookup(&self, scope: &Scope, prompt: &[u32]) -> Option<Hit> {
        let mut inner = self.inner.lock();
        inner.clock += 1;
        let now = inner.clock;
        let best = inner
            .entries
            .iter_mut()
            .filter(|(_, e)| {
                e.scope == *scope && e.tokens.len() <= prompt.len() && prompt.starts_with(&e.tokens)
            })
            .max_by_key(|(_, e)| e.tokens.len())?;
        let entry = best.1;
        entry.last_used = now;
        let payload = match (&entry.ram, &entry.disk) {
            (Some(blob), _) => Payload::Ram(blob.clone()),
            (None, Some(path)) => Payload::Disk(path.clone()),
            (None, None) => return None,
        };
        Some(Hit {
            tokens: entry.tokens.len(),
            payload,
        })
    }

    /// Keeps `blob` (the state after `tokens`) in RAM if that leaves the
    /// margin intact after evicting older entries; otherwise writes it
    /// straight to disk when allowed, else refuses it.
    pub fn put(&self, scope: Scope, tokens: Vec<u32>, blob: Vec<u8>) -> Result<Vec<Event>> {
        let bytes = blob.len() as u64;
        let id = entry_id(&scope, &tokens);
        let mut events = Vec::new();
        self.remove_id(id);
        // Room is made before the blob is kept: the bytes already exist, but
        // the RAM tier must not hold them unless the margin survives.
        let available = self.probe.mem_available()?;
        let need = self.cfg.ram_margin_bytes;
        if available < need {
            events.extend(self.release_until(need + self.cfg.ram_release_slack_bytes, available)?);
        }
        let available = self.probe.mem_available()?;
        let mut inner = self.inner.lock();
        inner.clock += 1;
        let last_used = inner.clock;
        if available >= need {
            inner.next_gen += 1;
            let generation = inner.next_gen;
            if let Some(prev) = inner.entries.remove(&id) {
                if prev.ram.is_some() {
                    inner.ram_bytes = inner.ram_bytes.saturating_sub(prev.bytes);
                }
                if let Some(old_path) = prev.disk {
                    let _ = fs::remove_file(old_path);
                    inner.disk_bytes = inner.disk_bytes.saturating_sub(prev.bytes);
                }
            }
            inner.ram_bytes += bytes;
            inner.entries.insert(
                id,
                Entry {
                    scope,
                    tokens,
                    bytes,
                    ram: Some(Arc::new(blob)),
                    disk: None,
                    last_used,
                    generation,
                    demoting: false,
                },
            );
            return Ok(events);
        }
        drop(inner);
        match self.write_disk(&scope, id, &blob)? {
            Ok(path) => {
                let mut inner = self.inner.lock();
                inner.next_gen += 1;
                let generation = inner.next_gen;
                if let Some(prev) = inner.entries.remove(&id) {
                    if prev.ram.is_some() {
                        inner.ram_bytes = inner.ram_bytes.saturating_sub(prev.bytes);
                    }
                    if let Some(old_path) = prev.disk {
                        if old_path != path {
                            let _ = fs::remove_file(old_path);
                        }
                        inner.disk_bytes = inner.disk_bytes.saturating_sub(prev.bytes);
                    }
                }
                inner.disk_bytes += bytes;
                inner.entries.insert(
                    id,
                    Entry {
                        scope,
                        tokens: tokens.clone(),
                        bytes,
                        ram: None,
                        disk: Some(path),
                        last_used,
                        generation,
                        demoting: false,
                    },
                );
                events.push(Event::Demoted {
                    tokens: tokens.len(),
                    bytes,
                });
            }
            Err(_) => events.push(Event::Refused {
                tokens: tokens.len(),
                bytes,
            }),
        }
        Ok(events)
    }

    /// The background check (call about once a second): below the margin,
    /// LRU RAM entries leave for disk, else are dropped, until MemAvailable
    /// is back to margin + slack.
    pub fn enforce_margin(&self) -> Result<Vec<Event>> {
        let available = self.probe.mem_available()?;
        if available >= self.cfg.ram_margin_bytes {
            return Ok(Vec::new());
        }
        self.release_until(
            self.cfg.ram_margin_bytes + self.cfg.ram_release_slack_bytes,
            available,
        )
    }

    fn release_until(&self, target: u64, mut available: u64) -> Result<Vec<Event>> {
        let mut events = Vec::new();
        while available < target {
            let victim = {
                let mut inner = self.inner.lock();
                let victim_id = inner
                    .entries
                    .iter()
                    .filter(|(_, e)| e.ram.is_some() && !e.demoting)
                    .min_by_key(|(_, e)| e.last_used)
                    .map(|(id, _)| *id);
                victim_id.and_then(|id| {
                    let e = inner.entries.get_mut(&id)?;
                    e.demoting = true;
                    Some((
                        id,
                        e.scope,
                        e.tokens.len(),
                        e.bytes,
                        e.ram.clone()?,
                        e.disk.is_some(),
                        e.generation,
                    ))
                })
            };
            let Some((id, scope, n_tokens, bytes, blob, on_disk, generation)) = victim else {
                break;
            };
            let written = if on_disk {
                Ok(None)
            } else {
                match self.write_disk(&scope, id, &blob) {
                    Ok(res) => res.map(Some),
                    Err(err) => {
                        let mut inner = self.inner.lock();
                        if let Some(e) = inner.entries.get_mut(&id)
                            && e.generation == generation
                        {
                            e.demoting = false;
                        }
                        return Err(err);
                    }
                }
            };
            let mut inner = self.inner.lock();
            let still_live = inner
                .entries
                .get(&id)
                .is_some_and(|e| e.generation == generation && e.ram.is_some());
            let mut released_bytes = 0u64;
            if still_live {
                inner.ram_bytes = inner.ram_bytes.saturating_sub(bytes);
                released_bytes = bytes;
                match written {
                    Ok(Some(path)) => {
                        let mut added_disk = false;
                        if let Some(e) = inner.entries.get_mut(&id) {
                            e.demoting = false;
                            e.ram = None;
                            if e.disk.is_none() {
                                added_disk = true;
                                e.disk = Some(path);
                            } else if e.disk.as_deref() != Some(path.as_path()) {
                                let _ = fs::remove_file(&path);
                            }
                        }
                        if added_disk {
                            inner.disk_bytes += bytes;
                        }
                        events.push(Event::Demoted {
                            tokens: n_tokens,
                            bytes,
                        });
                    }
                    Ok(None) => {
                        let still_on_disk =
                            inner.entries.get(&id).is_some_and(|e| e.disk.is_some());
                        if still_on_disk {
                            if let Some(e) = inner.entries.get_mut(&id) {
                                e.demoting = false;
                                e.ram = None;
                            }
                            events.push(Event::Demoted {
                                tokens: n_tokens,
                                bytes,
                            });
                        } else {
                            inner.entries.remove(&id);
                            events.push(Event::Dropped {
                                tokens: n_tokens,
                                bytes,
                                reason: "disk entry evicted",
                            });
                        }
                    }
                    Err(reason) => {
                        let has_disk = inner.entries.get(&id).is_some_and(|e| e.disk.is_some());
                        if has_disk {
                            if let Some(e) = inner.entries.get_mut(&id) {
                                e.demoting = false;
                                e.ram = None;
                            }
                        } else {
                            inner.entries.remove(&id);
                        }
                        events.push(Event::Dropped {
                            tokens: n_tokens,
                            bytes,
                            reason,
                        });
                    }
                }
            } else if let Ok(Some(path)) = written {
                let owned_by_replacement = inner
                    .entries
                    .get(&id)
                    .is_some_and(|e| e.disk.as_deref() == Some(path.as_path()));
                if !owned_by_replacement {
                    let _ = fs::remove_file(&path);
                }
            }
            drop(inner);
            drop(blob);
            available = self
                .probe
                .mem_available()?
                .max(available.saturating_add(released_bytes));
        }
        Ok(events)
    }

    /// Writes one entry atomically (temp file, fsync, rename). The inner
    /// result is the refusal reason when the tier's rules forbid the write.
    fn write_disk(
        &self,
        scope: &Scope,
        id: [u8; DIGEST_BYTES],
        blob: &[u8],
    ) -> Result<std::result::Result<PathBuf, &'static str>> {
        let Some(dir) = &self.cfg.disk_dir else {
            return Ok(Err("no disk tier"));
        };
        let bytes = blob.len() as u64;
        {
            let mut inner = self.inner.lock();
            if inner.disk_bytes + bytes > self.cfg.disk_cap_bytes {
                self.evict_disk_lru(&mut inner, bytes);
                if inner.disk_bytes + bytes > self.cfg.disk_cap_bytes {
                    return Ok(Err("disk tier full"));
                }
            }
            if free_bytes(dir).is_some_and(|free| free < bytes + self.cfg.disk_free_min_bytes) {
                return Ok(Err("filesystem free-space floor"));
            }
            let now = Instant::now();
            if let Some(bucket) = inner.bucket.as_mut()
                && !bucket.take(bytes, now)
            {
                return Ok(Err("write budget spent"));
            }
        }
        let name = format!("{:016x}-{}.{FILE_SUFFIX}", scope.tenant, hex(&id));
        let path = dir.join(&name);
        let seq = TMP_FILE_SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = dir.join(format!("{name}.{}.{seq}.tmp", std::process::id()));
        let write = (|| -> Result<()> {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(blob)?;
            f.sync_all()?;
            fs::rename(&tmp, &path)?;
            Ok(())
        })();
        if let Err(e) = write {
            let _ = fs::remove_file(&tmp);
            return Err(e.context(format!("writing prompt-cache entry {}", path.display())));
        }
        Ok(Ok(path))
    }

    fn evict_disk_lru(&self, inner: &mut Inner, need: u64) {
        while inner.disk_bytes + need > self.cfg.disk_cap_bytes {
            let Some((id, path, bytes, in_ram)) = inner
                .entries
                .iter()
                .filter(|(_, e)| e.disk.is_some())
                .min_by_key(|(_, e)| e.last_used)
                .map(|(id, e)| (*id, e.disk.clone(), e.bytes, e.ram.is_some()))
            else {
                return;
            };
            if let Some(path) = path {
                let _ = fs::remove_file(path);
            }
            inner.disk_bytes = inner.disk_bytes.saturating_sub(bytes);
            if in_ram {
                if let Some(e) = inner.entries.get_mut(&id) {
                    e.disk = None;
                }
            } else {
                inner.entries.remove(&id);
            }
        }
    }

    fn remove_id(&self, id: [u8; DIGEST_BYTES]) {
        let mut inner = self.inner.lock();
        if let Some(e) = inner.entries.remove(&id) {
            if e.ram.is_some() {
                inner.ram_bytes = inner.ram_bytes.saturating_sub(e.bytes);
            }
            if let Some(path) = e.disk {
                let _ = fs::remove_file(path);
                inner.disk_bytes = inner.disk_bytes.saturating_sub(e.bytes);
            }
        }
    }

    /// Drops every entry of `scope` whose tokens are exactly `tokens`.
    pub fn remove(&self, scope: &Scope, tokens: &[u32]) {
        self.remove_id(entry_id(scope, tokens));
    }

    /// (RAM entries, RAM bytes, disk entries, disk bytes).
    pub fn stats(&self) -> (usize, u64, usize, u64) {
        let inner = self.inner.lock();
        let ram = inner.entries.values().filter(|e| e.ram.is_some()).count();
        let disk = inner.entries.values().filter(|e| e.disk.is_some()).count();
        (ram, inner.ram_bytes, disk, inner.disk_bytes)
    }
}

/// Free bytes on the filesystem holding `dir` (None when unknown).
fn free_bytes(dir: &Path) -> Option<u64> {
    #[cfg(any(
        all(
            target_os = "linux",
            target_pointer_width = "64",
            any(target_env = "gnu", target_env = "musl")
        ),
        all(target_os = "macos", target_pointer_width = "64")
    ))]
    {
        use std::os::unix::ffi::OsStrExt;
        let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
        let mut st = std::mem::MaybeUninit::<libc_statvfs::Statvfs>::uninit();
        // SAFETY: `c` is a valid NUL-terminated path and `st` is a writable
        // buffer matching the target platform's exact 64-bit C `struct statvfs`
        // layout (glibc/musl 112-byte layout on Linux, Darwin 64-byte layout on macOS).
        let rc = unsafe { libc_statvfs::statvfs(c.as_ptr(), st.as_mut_ptr()) };
        if rc != 0 {
            return None;
        }
        // SAFETY: `statvfs` returned 0, so it initialized every field of `st`.
        let st = unsafe { st.assume_init() };
        Some(st.avail_bytes())
    }
    #[cfg(not(any(
        all(
            target_os = "linux",
            target_pointer_width = "64",
            any(target_env = "gnu", target_env = "musl")
        ),
        all(target_os = "macos", target_pointer_width = "64")
    )))]
    {
        let _ = dir;
        None
    }
}

#[cfg(any(
    all(
        target_os = "linux",
        target_pointer_width = "64",
        any(target_env = "gnu", target_env = "musl")
    ),
    all(target_os = "macos", target_pointer_width = "64")
))]
mod libc_statvfs {
    /// 64-bit Linux `struct statvfs` (glibc and musl share this 112-byte layout).
    #[cfg(all(
        target_os = "linux",
        target_pointer_width = "64",
        any(target_env = "gnu", target_env = "musl")
    ))]
    #[allow(dead_code)]
    #[repr(C)]
    pub struct Statvfs {
        pub f_bsize: u64,
        pub f_frsize: u64,
        pub f_blocks: u64,
        pub f_bfree: u64,
        pub f_bavail: u64,
        pub f_files: u64,
        pub f_ffree: u64,
        pub f_favail: u64,
        pub f_fsid: u64,
        pub f_flag: u64,
        pub f_namemax: u64,
        pub __spare: [i32; 6],
    }

    /// 64-bit Darwin/macOS `struct statvfs` (64-byte layout with 32-bit
    /// `fsblkcnt_t` / `fsfilcnt_t`).
    #[cfg(all(target_os = "macos", target_pointer_width = "64"))]
    #[allow(dead_code)]
    #[repr(C)]
    pub struct Statvfs {
        pub f_bsize: u64,
        pub f_frsize: u64,
        pub f_blocks: u32,
        pub f_bfree: u32,
        pub f_bavail: u32,
        pub f_files: u32,
        pub f_ffree: u32,
        pub f_favail: u32,
        pub f_fsid: u64,
        pub f_flag: u64,
        pub f_namemax: u64,
    }

    impl Statvfs {
        #[cfg(all(
            target_os = "linux",
            target_pointer_width = "64",
            any(target_env = "gnu", target_env = "musl")
        ))]
        pub fn avail_bytes(&self) -> u64 {
            self.f_bavail.saturating_mul(self.f_frsize)
        }

        #[cfg(all(target_os = "macos", target_pointer_width = "64"))]
        pub fn avail_bytes(&self) -> u64 {
            u64::from(self.f_bavail).saturating_mul(self.f_frsize)
        }
    }

    // SAFETY: POSIX `statvfs(const char *, struct statvfs *)` is exported by libc
    // on 64-bit Linux (glibc/musl) and 64-bit macOS, and `Statvfs` matches each
    // gated target's exact C struct layout above.
    unsafe extern "C" {
        pub fn statvfs(path: *const std::ffi::c_char, buf: *mut Statvfs) -> std::ffi::c_int;
    }
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
