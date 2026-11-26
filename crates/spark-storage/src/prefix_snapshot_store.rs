// SPDX-License-Identifier: AGPL-3.0-only

//! UMA-aware `O_DIRECT` persistent hybrid prefix snapshot store.
//!
//! Persists block-aligned KV page payloads (BF16, FP8, NVFP4, and TurboQuant+
//! `turbo2`/`turbo3`/`turbo4`/`turbo8` layouts), optional QSA compressed
//! selector keys, GDN recurrent `h_state`/`conv_state`, and Qwen4 PLE
//! `conv_state` across server restarts and multi-session evictions without
//! duplicating unified memory in the Linux page cache.

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Hardware/filesystem page alignment required for Linux `O_DIRECT` on NVMe.
pub const SNAPSHOT_PAGE_ALIGN: usize = 4096;

/// Default persistent prefix-cache capacity: 32 GiB.
pub const DEFAULT_MAX_STORE_BYTES: u64 = 32 * 1024 * 1024 * 1024;

/// Default daily NVMe write cap protecting SSD endurance: 128 GiB/day.
pub const DEFAULT_DAILY_WRITE_CAP_BYTES: u64 = 128 * 1024 * 1024 * 1024;

/// Default checkpoint grid granularity in tokens (must be a multiple of `block_size`).
pub const DEFAULT_SNAPSHOT_GRID_TOKENS: usize = 2048;

/// Environment variable selecting the persistent prefix snapshot directory.
pub const PREFIX_CACHE_DIR_ENV: &str = "FLUX_PREFIX_CACHE_DIR";
/// Optional environment override for maximum persistent store bytes.
pub const PREFIX_CACHE_MAX_BYTES_ENV: &str = "FLUX_PREFIX_CACHE_MAX_BYTES";
/// Optional environment override for daily NVMe write cap in bytes.
pub const PREFIX_CACHE_DAILY_WRITE_CAP_ENV: &str = "FLUX_PREFIX_CACHE_DAILY_WRITE_CAP_BYTES";

const FILE_MAGIC: [u8; 8] = *b"ATLPFX01";
const FILE_FORMAT_VERSION: u32 = 1;
const MANIFEST_FILENAME: &str = "manifest.json";
const SECONDS_PER_DAY: u64 = 86_400;
const SECTION_COUNT: usize = 5;

static STAGING_COUNTER: AtomicU64 = AtomicU64::new(1);

/// I/O mode used to read and write 4,096-byte aligned `.snap` files.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapshotIoMode {
    /// Attempt Linux `O_DIRECT` first to bypass host page cache on UMA, falling
    /// back automatically to buffered `pread`/`pwrite` on `EINVAL` (e.g. `tmpfs`).
    #[default]
    AutoDirect,
    /// Always use buffered POSIX `pread`/`pwrite` (with `POSIX_FADV_DONTNEED` on Linux).
    Buffered,
}

/// Supported KV page storage formats, including Atlas TurboQuant+ compressed pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotKvDtype {
    Bf16,
    Fp8,
    Nvfp4,
    Turbo2,
    Turbo3,
    Turbo4,
    Turbo8,
    Bf16KTurbo4V,
    Bf16KTurbo3V,
    Bf16KTurbo2V,
    Fp8KTurbo4V,
    Fp8KTurbo3V,
    Fp8KTurbo2V,
    Turbo3KTurbo8V,
    Turbo4KTurbo3V,
    Turbo4KTurbo8V,
}

impl SnapshotKvDtype {
    /// Stable wire tag written into the 4,096-byte binary header.
    #[must_use]
    pub const fn wire_tag(self) -> u32 {
        match self {
            Self::Bf16 => 1,
            Self::Fp8 => 2,
            Self::Nvfp4 => 3,
            Self::Turbo2 => 4,
            Self::Turbo3 => 5,
            Self::Turbo4 => 6,
            Self::Turbo8 => 7,
            Self::Bf16KTurbo4V => 8,
            Self::Bf16KTurbo3V => 9,
            Self::Bf16KTurbo2V => 10,
            Self::Fp8KTurbo4V => 11,
            Self::Fp8KTurbo3V => 12,
            Self::Fp8KTurbo2V => 13,
            Self::Turbo3KTurbo8V => 14,
            Self::Turbo4KTurbo3V => 15,
            Self::Turbo4KTurbo8V => 16,
        }
    }

    /// Canonical lowercase identifier used in content addressing and CLI flags.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Nvfp4 => "nvfp4",
            Self::Turbo2 => "turbo2",
            Self::Turbo3 => "turbo3",
            Self::Turbo4 => "turbo4",
            Self::Turbo8 => "turbo8",
            Self::Bf16KTurbo4V => "bf16_k_turbo4_v",
            Self::Bf16KTurbo3V => "bf16_k_turbo3_v",
            Self::Bf16KTurbo2V => "bf16_k_turbo2_v",
            Self::Fp8KTurbo4V => "fp8_k_turbo4_v",
            Self::Fp8KTurbo3V => "fp8_k_turbo3_v",
            Self::Fp8KTurbo2V => "fp8_k_turbo2_v",
            Self::Turbo3KTurbo8V => "turbo3_k_turbo8_v",
            Self::Turbo4KTurbo3V => "turbo4_k_turbo3_v",
            Self::Turbo4KTurbo8V => "turbo4_k_turbo8_v",
        }
    }

    /// True when this KV layout uses Atlas TurboQuant+ Walsh-Hadamard compression.
    #[must_use]
    pub const fn is_turboquant(self) -> bool {
        !matches!(self, Self::Bf16 | Self::Fp8 | Self::Nvfp4)
    }

    fn from_wire_tag(tag: u32) -> Result<Self> {
        match tag {
            1 => Ok(Self::Bf16),
            2 => Ok(Self::Fp8),
            3 => Ok(Self::Nvfp4),
            4 => Ok(Self::Turbo2),
            5 => Ok(Self::Turbo3),
            6 => Ok(Self::Turbo4),
            7 => Ok(Self::Turbo8),
            8 => Ok(Self::Bf16KTurbo4V),
            9 => Ok(Self::Bf16KTurbo3V),
            10 => Ok(Self::Bf16KTurbo2V),
            11 => Ok(Self::Fp8KTurbo4V),
            12 => Ok(Self::Fp8KTurbo3V),
            13 => Ok(Self::Fp8KTurbo2V),
            14 => Ok(Self::Turbo3KTurbo8V),
            15 => Ok(Self::Turbo4KTurbo3V),
            16 => Ok(Self::Turbo4KTurbo8V),
            other => bail!("unknown snapshot KV dtype wire tag {other}"),
        }
    }
}

impl fmt::Display for SnapshotKvDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SnapshotKvDtype {
    type Err = anyhow::Error;

    fn from_str(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            "fp8" | "fp8_e4m3" => Ok(Self::Fp8),
            "nvfp4" | "fp4" => Ok(Self::Nvfp4),
            "turbo2" => Ok(Self::Turbo2),
            "turbo3" => Ok(Self::Turbo3),
            "turbo4" => Ok(Self::Turbo4),
            "turbo8" => Ok(Self::Turbo8),
            "bf16_k_turbo4_v" | "bf16-turbo4" => Ok(Self::Bf16KTurbo4V),
            "bf16_k_turbo3_v" | "bf16-turbo3" => Ok(Self::Bf16KTurbo3V),
            "bf16_k_turbo2_v" | "bf16-turbo2" => Ok(Self::Bf16KTurbo2V),
            "fp8_k_turbo4_v" | "fp8-turbo4" => Ok(Self::Fp8KTurbo4V),
            "fp8_k_turbo3_v" | "fp8-turbo3" => Ok(Self::Fp8KTurbo3V),
            "fp8_k_turbo2_v" | "fp8-turbo2" => Ok(Self::Fp8KTurbo2V),
            "turbo3_k_turbo8_v" | "turbo3-turbo8" => Ok(Self::Turbo3KTurbo8V),
            "turbo4_k_turbo3_v" | "turbo4-turbo3" => Ok(Self::Turbo4KTurbo3V),
            "turbo4_k_turbo8_v" | "turbo4-turbo8" => Ok(Self::Turbo4KTurbo8V),
            other => bail!("unsupported snapshot KV dtype {other:?}"),
        }
    }
}

/// Content-addressed lookup key for a persistent hybrid prefix checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrefixSnapshotKey {
    pub model_digest: String,
    pub kv_dtype: SnapshotKvDtype,
    pub block_size: u32,
    pub grid_token_len: u64,
    pub token_prefix_hash: [u8; 32],
}

impl PrefixSnapshotKey {
    /// Construct a validated key from an exact token prefix slice.
    ///
    /// # Errors
    /// Returns an error if `model_digest` is empty, `block_size` is zero,
    /// `tokens` is empty, or `tokens.len()` is not a multiple of `block_size`.
    pub fn from_tokens(
        model_digest: impl Into<String>,
        kv_dtype: SnapshotKvDtype,
        block_size: u32,
        tokens: &[u32],
    ) -> Result<Self> {
        let model_digest = model_digest.into();
        let key = Self {
            model_digest,
            kv_dtype,
            block_size,
            grid_token_len: tokens.len() as u64,
            token_prefix_hash: hash_tokens_sha256(tokens),
        };
        key.validate()?;
        Ok(key)
    }

    /// Validate structural invariants of the content-addressed key.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.model_digest.trim().is_empty(),
            "prefix snapshot model_digest must not be empty"
        );
        ensure!(
            self.block_size > 0,
            "prefix snapshot block_size must be > 0"
        );
        ensure!(
            self.grid_token_len > 0,
            "prefix snapshot grid_token_len must be > 0"
        );
        ensure!(
            self.grid_token_len
                .is_multiple_of(u64::from(self.block_size)),
            "prefix snapshot grid_token_len ({}) must be a multiple of block_size ({})",
            self.grid_token_len,
            self.block_size
        );
        Ok(())
    }

    /// Raw 32-byte SHA-256 content digest uniquely binding all key fields.
    #[must_use]
    pub fn content_digest(&self) -> [u8; 32] {
        let dtype_str = self.kv_dtype.as_str().as_bytes();
        let model_bytes = self.model_digest.as_bytes();
        let mut buf = Vec::with_capacity(64 + model_bytes.len() + dtype_str.len());
        buf.extend_from_slice(b"atlas.prefix_snapshot.v1\0");
        buf.extend_from_slice(&(model_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(model_bytes);
        buf.extend_from_slice(&(dtype_str.len() as u64).to_le_bytes());
        buf.extend_from_slice(dtype_str);
        buf.extend_from_slice(&self.block_size.to_le_bytes());
        buf.extend_from_slice(&self.grid_token_len.to_le_bytes());
        buf.extend_from_slice(&self.token_prefix_hash);
        sha256_bytes(&buf)
    }

    /// Lowercase 64-character hex content ID used as the `.snap` filename stem.
    #[must_use]
    pub fn content_id(&self) -> String {
        hex_encode(&self.content_digest())
    }
}

/// Complete hybrid state payload persisted in a single `.snap` checkpoint file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefixSnapshotPayload {
    /// Raw paged KV blocks (data + scales for NVFP4 / TurboQuant+ `turbo3/4/8` / FP8 / BF16).
    pub kv_pages: Vec<u8>,
    /// Optional Qwen4 QSA compressed selector keys (`raw_ring` + `compressed_keys`).
    pub qsa_selectors: Vec<u8>,
    /// Recurrent GDN `h_state` (stored in canonical FP32 across `num_ssm_layers`).
    pub gdn_h_state: Vec<u8>,
    /// Recurrent GDN `conv_state` across `num_ssm_layers`.
    pub gdn_conv_state: Vec<u8>,
    /// Qwen4 PLE `conv_state` (`[ple_dim, conv_hist]` BF16 across `num_ple_layers`).
    pub ple_conv_state: Vec<u8>,
    pub num_kv_layers: u32,
    pub num_ssm_layers: u32,
    pub num_ple_layers: u32,
}

impl PrefixSnapshotPayload {
    /// Logical (unpadded) byte count across all sections.
    #[must_use]
    pub fn logical_bytes(&self) -> u64 {
        (self.kv_pages.len()
            + self.qsa_selectors.len()
            + self.gdn_h_state.len()
            + self.gdn_conv_state.len()
            + self.ple_conv_state.len()) as u64
    }

    /// Total on-disk 4,096-byte aligned file size including the 4 KiB header page.
    pub fn aligned_file_bytes(&self) -> Result<u64> {
        let mut total = SNAPSHOT_PAGE_ALIGN as u64;
        for section_len in [
            self.kv_pages.len(),
            self.qsa_selectors.len(),
            self.gdn_h_state.len(),
            self.gdn_conv_state.len(),
            self.ple_conv_state.len(),
        ] {
            let aligned = align_up_usize(section_len, SNAPSHOT_PAGE_ALIGN)? as u64;
            total = total
                .checked_add(aligned)
                .context("snapshot aligned file size overflow")?;
        }
        Ok(total)
    }
}

/// Configuration for [`PrefixSnapshotStore`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixSnapshotStoreConfig {
    pub dir: PathBuf,
    pub max_bytes: u64,
    pub daily_write_cap_bytes: u64,
    pub grid_tokens: usize,
    pub io_mode: SnapshotIoMode,
}

impl PrefixSnapshotStoreConfig {
    /// Create a store configuration for `dir` with 32 GiB budget, 128 GiB/day write cap,
    /// 2,048-token grid, and automatic `O_DIRECT` mode.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            max_bytes: DEFAULT_MAX_STORE_BYTES,
            daily_write_cap_bytes: DEFAULT_DAILY_WRITE_CAP_BYTES,
            grid_tokens: DEFAULT_SNAPSHOT_GRID_TOKENS,
            io_mode: SnapshotIoMode::AutoDirect,
        }
    }

    /// Resolve configuration from an explicit CLI `--prefix-cache-dir` path or
    /// `FLUX_PREFIX_CACHE_DIR`. Returns `Ok(None)` when neither is set.
    pub fn from_env_or_path(explicit_dir: Option<&Path>) -> Result<Option<Self>> {
        let dir = match explicit_dir {
            Some(p) => p.to_path_buf(),
            None => match std::env::var_os(PREFIX_CACHE_DIR_ENV) {
                Some(raw) if !raw.is_empty() => PathBuf::from(raw),
                _ => return Ok(None),
            },
        };
        let mut cfg = Self::new(dir);
        if let Ok(raw) = std::env::var(PREFIX_CACHE_MAX_BYTES_ENV) {
            cfg.max_bytes = raw
                .trim()
                .parse::<u64>()
                .with_context(|| format!("invalid {PREFIX_CACHE_MAX_BYTES_ENV}={raw:?}"))?;
        }
        if let Ok(raw) = std::env::var(PREFIX_CACHE_DAILY_WRITE_CAP_ENV) {
            cfg.daily_write_cap_bytes = raw
                .trim()
                .parse::<u64>()
                .with_context(|| format!("invalid {PREFIX_CACHE_DAILY_WRITE_CAP_ENV}={raw:?}"))?;
        }
        Ok(Some(cfg))
    }
}

/// Outcome of persisting a hybrid checkpoint into [`PrefixSnapshotStore`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StorePutOutcome {
    /// Snapshot written (or refreshed if identical content ID already existed).
    Stored {
        content_id: String,
        aligned_bytes: u64,
        evicted_entries: usize,
        used_direct_io: bool,
    },
    /// Snapshot alone exceeds `max_bytes`; skipped cleanly without mutating store.
    ExceedsMaxBudget { required_bytes: u64, max_bytes: u64 },
    /// Writing this snapshot would exceed `daily_write_cap_bytes` for the UTC day.
    DailyWriteCapReached {
        attempted_bytes: u64,
        remaining_daily_bytes: u64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ManifestEntry {
    key: PrefixSnapshotKey,
    file_name: String,
    aligned_bytes: u64,
    logical_bytes: u64,
    last_access_seq: u64,
    hit_count: u64,
    created_unix_secs: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct StoreManifest {
    version: u32,
    access_seq: u64,
    daily_day_index: u64,
    daily_bytes_written: u64,
    entries: BTreeMap<String, ManifestEntry>,
}

/// Persistent content-addressed disk store for hybrid KV + QSA + GDN + PLE snapshots.
pub struct PrefixSnapshotStore {
    cfg: PrefixSnapshotStoreConfig,
    manifest: StoreManifest,
    used_bytes: u64,
    now_unix_secs_fn: Box<dyn Fn() -> u64 + Send + Sync>,
}

impl fmt::Debug for PrefixSnapshotStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefixSnapshotStore")
            .field("dir", &self.cfg.dir)
            .field("used_bytes", &self.used_bytes)
            .field("max_bytes", &self.cfg.max_bytes)
            .field("entries", &self.manifest.entries.len())
            .finish()
    }
}

/// TensorFold live cost-pricing admission rule: fault a prefix checkpoint from
/// NVMe only when reading `total_bytes` at `nvme_read_bps` is strictly faster
/// than recomputing `cached_tokens` at `prefill_tps`.
///
/// Formula: `(total_bytes / nvme_read_bps) < (cached_tokens / prefill_tps)`.
#[must_use]
pub fn should_restore_from_disk(
    cached_tokens: usize,
    total_bytes: u64,
    nvme_read_bps: f64,
    prefill_tps: f64,
) -> bool {
    if cached_tokens == 0 {
        return false;
    }
    if !prefill_tps.is_finite() || prefill_tps <= 0.0 {
        return false;
    }
    if total_bytes == 0 {
        return true;
    }
    if !nvme_read_bps.is_finite() || nvme_read_bps <= 0.0 {
        return false;
    }
    let read_seconds = (total_bytes as f64) / nvme_read_bps;
    let prefill_seconds = (cached_tokens as f64) / prefill_tps;
    read_seconds < prefill_seconds
}

/// Compute SHA-256 over little-endian `u32` token IDs.
#[must_use]
pub fn hash_tokens_sha256(tokens: &[u32]) -> [u8; 32] {
    let mut raw = Vec::with_capacity(std::mem::size_of_val(tokens));
    for &tok in tokens {
        raw.extend_from_slice(&tok.to_le_bytes());
    }
    sha256_bytes(&raw)
}

impl PrefixSnapshotStore {
    /// Open or initialize a persistent prefix snapshot store at `cfg.dir`.
    ///
    /// Reconciles `manifest.json` against on-disk `.snap` files, cleans up any
    /// interrupted `.tmp-*` staging artifacts, and enforces `cfg.max_bytes`.
    pub fn open(cfg: PrefixSnapshotStoreConfig) -> Result<Self> {
        Self::open_with_clock(cfg, Box::new(default_unix_secs))
    }

    /// Open a store with an injected wall-clock source (used for deterministic
    /// daily-write-cap rollover tests).
    pub fn open_with_clock(
        cfg: PrefixSnapshotStoreConfig,
        now_unix_secs_fn: Box<dyn Fn() -> u64 + Send + Sync>,
    ) -> Result<Self> {
        ensure!(
            cfg.grid_tokens > 0,
            "prefix snapshot grid_tokens must be positive"
        );
        fs::create_dir_all(&cfg.dir)
            .with_context(|| format!("create prefix snapshot dir {}", cfg.dir.display()))?;
        cleanup_orphan_staging_files(&cfg.dir);

        let manifest_path = cfg.dir.join(MANIFEST_FILENAME);
        let mut manifest = if manifest_path.is_file() {
            let raw = fs::read_to_string(&manifest_path)
                .with_context(|| format!("read {}", manifest_path.display()))?;
            serde_json::from_str::<StoreManifest>(&raw)
                .with_context(|| format!("parse {}", manifest_path.display()))?
        } else {
            StoreManifest {
                version: FILE_FORMAT_VERSION,
                ..StoreManifest::default()
            }
        };
        if manifest.version == 0 {
            manifest.version = FILE_FORMAT_VERSION;
        }

        let mut used_bytes = 0u64;
        let mut pruned = false;
        manifest.entries.retain(|content_id, entry| {
            let snap_path = cfg.dir.join(&entry.file_name);
            let ok = entry.key.validate().is_ok()
                && &entry.key.content_id() == content_id
                && entry.aligned_bytes >= SNAPSHOT_PAGE_ALIGN as u64
                && entry
                    .aligned_bytes
                    .is_multiple_of(SNAPSHOT_PAGE_ALIGN as u64)
                && fs::metadata(&snap_path)
                    .map(|m| m.is_file() && m.len() == entry.aligned_bytes)
                    .unwrap_or(false);
            if ok {
                used_bytes = used_bytes.saturating_add(entry.aligned_bytes);
            } else {
                let _ = fs::remove_file(&snap_path);
                pruned = true;
            }
            ok
        });

        let mut store = Self {
            cfg,
            manifest,
            used_bytes,
            now_unix_secs_fn,
        };
        let evicted = store.evict_until_within(store.cfg.max_bytes);
        if pruned || evicted > 0 {
            store.persist_manifest()?;
        }
        Ok(store)
    }

    #[must_use]
    pub fn config(&self) -> &PrefixSnapshotStoreConfig {
        &self.cfg
    }

    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.manifest.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.manifest.entries.is_empty()
    }

    #[must_use]
    pub fn daily_bytes_written(&self) -> u64 {
        let today = (self.now_unix_secs_fn)() / SECONDS_PER_DAY;
        if self.manifest.daily_day_index == today {
            self.manifest.daily_bytes_written
        } else {
            0
        }
    }

    /// Check whether `key` is resident in the manifest and return its aligned disk size.
    #[must_use]
    pub fn contains_bytes(&self, key: &PrefixSnapshotKey) -> Option<u64> {
        let id = key.content_id();
        self.manifest.entries.get(&id).map(|e| e.aligned_bytes)
    }

    /// Persist a hybrid snapshot atomically (`.tmp` -> `fsync` -> `rename`),
    /// enforcing LRU byte-budget eviction and the daily NVMe write cap.
    pub fn put(
        &mut self,
        key: PrefixSnapshotKey,
        payload: &PrefixSnapshotPayload,
    ) -> Result<StorePutOutcome> {
        key.validate()?;
        ensure!(
            !payload.kv_pages.is_empty()
                || !payload.gdn_h_state.is_empty()
                || !payload.ple_conv_state.is_empty(),
            "refusing to persist an empty prefix snapshot payload"
        );
        let aligned_bytes = payload.aligned_file_bytes()?;
        if aligned_bytes > self.cfg.max_bytes {
            return Ok(StorePutOutcome::ExceedsMaxBudget {
                required_bytes: aligned_bytes,
                max_bytes: self.cfg.max_bytes,
            });
        }

        let content_id = key.content_id();
        if let Some(existing) = self.manifest.entries.get_mut(&content_id) {
            let snap_path = self.cfg.dir.join(&existing.file_name);
            if snap_path.is_file() {
                self.manifest.access_seq = self.manifest.access_seq.saturating_add(1);
                existing.last_access_seq = self.manifest.access_seq;
                existing.hit_count = existing.hit_count.saturating_add(1);
                let aligned = existing.aligned_bytes;
                self.persist_manifest()?;
                return Ok(StorePutOutcome::Stored {
                    content_id,
                    aligned_bytes: aligned,
                    evicted_entries: 0,
                    used_direct_io: false,
                });
            }
        }

        let now_secs = (self.now_unix_secs_fn)();
        self.roll_daily_window(now_secs);
        let next_daily = self
            .manifest
            .daily_bytes_written
            .saturating_add(aligned_bytes);
        if next_daily > self.cfg.daily_write_cap_bytes {
            let remaining = self
                .cfg
                .daily_write_cap_bytes
                .saturating_sub(self.manifest.daily_bytes_written);
            return Ok(StorePutOutcome::DailyWriteCapReached {
                attempted_bytes: aligned_bytes,
                remaining_daily_bytes: remaining,
            });
        }

        let target_budget = self.cfg.max_bytes.saturating_sub(aligned_bytes);
        let evicted_entries = self.evict_until_within(target_budget);

        let image = serialize_aligned_snapshot(&key, payload)?;
        debug_assert_eq!(image.len() as u64, aligned_bytes);

        let file_name = format!("{content_id}.snap");
        let final_path = self.cfg.dir.join(&file_name);
        let tmp_path = staging_path_for(&self.cfg.dir, &file_name);

        let used_direct_io = match write_aligned_file(&tmp_path, &image, self.cfg.io_mode) {
            Ok(direct) => direct,
            Err(err) => {
                let _ = fs::remove_file(&tmp_path);
                return Err(err);
            }
        };

        if let Err(err) = fs::rename(&tmp_path, &final_path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(err).with_context(|| {
                format!(
                    "atomic commit {} -> {}",
                    tmp_path.display(),
                    final_path.display()
                )
            });
        }

        self.manifest.access_seq = self.manifest.access_seq.saturating_add(1);
        self.manifest.daily_bytes_written = next_daily;
        if let Some(prev) = self.manifest.entries.insert(
            content_id.clone(),
            ManifestEntry {
                key,
                file_name,
                aligned_bytes,
                logical_bytes: payload.logical_bytes(),
                last_access_seq: self.manifest.access_seq,
                hit_count: 0,
                created_unix_secs: now_secs,
            },
        ) {
            self.used_bytes = self.used_bytes.saturating_sub(prev.aligned_bytes);
        }
        self.used_bytes = self.used_bytes.saturating_add(aligned_bytes);
        self.persist_manifest()?;

        Ok(StorePutOutcome::Stored {
            content_id,
            aligned_bytes,
            evicted_entries,
            used_direct_io,
        })
    }

    /// Load and verify a snapshot by `key`, updating its LRU/hit metadata.
    /// Returns `Ok(None)` when `key` is not in the manifest.
    pub fn get(&mut self, key: &PrefixSnapshotKey) -> Result<Option<PrefixSnapshotPayload>> {
        key.validate()?;
        let content_id = key.content_id();
        let (file_name, expected_len) = match self.manifest.entries.get(&content_id) {
            Some(entry) => (entry.file_name.clone(), entry.aligned_bytes),
            None => return Ok(None),
        };
        let path = self.cfg.dir.join(&file_name);
        if !path.is_file() {
            if let Some(removed) = self.manifest.entries.remove(&content_id) {
                self.used_bytes = self.used_bytes.saturating_sub(removed.aligned_bytes);
                let _ = self.persist_manifest();
            }
            return Ok(None);
        }

        let len = usize::try_from(expected_len).context("snapshot file exceeds usize")?;
        let image = read_aligned_file(&path, len, self.cfg.io_mode)?;
        let payload = deserialize_and_verify_snapshot(key, image.as_slice())?;

        self.manifest.access_seq = self.manifest.access_seq.saturating_add(1);
        if let Some(entry) = self.manifest.entries.get_mut(&content_id) {
            entry.last_access_seq = self.manifest.access_seq;
            entry.hit_count = entry.hit_count.saturating_add(1);
        }
        Ok(Some(payload))
    }

    /// Fault a snapshot from disk ONLY when it exists AND passes TensorFold's
    /// live cost-pricing check (`should_restore_from_disk`).
    pub fn load_if_cost_effective(
        &mut self,
        key: &PrefixSnapshotKey,
        nvme_read_bps: f64,
        prefill_tps: f64,
    ) -> Result<Option<PrefixSnapshotPayload>> {
        let Some(total_bytes) = self.contains_bytes(key) else {
            return Ok(None);
        };
        let cached_tokens =
            usize::try_from(key.grid_token_len).context("grid_token_len exceeds usize")?;
        if !should_restore_from_disk(cached_tokens, total_bytes, nvme_read_bps, prefill_tps) {
            return Ok(None);
        }
        self.get(key)
    }

    fn roll_daily_window(&mut self, now_unix_secs: u64) {
        let today = now_unix_secs / SECONDS_PER_DAY;
        if self.manifest.daily_day_index != today {
            self.manifest.daily_day_index = today;
            self.manifest.daily_bytes_written = 0;
        }
    }

    fn evict_until_within(&mut self, byte_limit: u64) -> usize {
        let mut evicted = 0usize;
        while self.used_bytes > byte_limit && !self.manifest.entries.is_empty() {
            // Forecast-weighted LRU score: older (`last_access_seq`) and colder (`hit_count`)
            // entries have the lowest retention score and evict first.
            let victim_id = self
                .manifest
                .entries
                .iter()
                .min_by_key(|(_, entry)| {
                    let weight = u128::from(entry.hit_count.saturating_add(1));
                    (
                        u128::from(entry.last_access_seq).saturating_mul(weight),
                        entry.last_access_seq,
                    )
                })
                .map(|(id, _)| id.clone());
            let Some(victim_id) = victim_id else {
                break;
            };
            if let Some(removed) = self.manifest.entries.remove(&victim_id) {
                let _ = fs::remove_file(self.cfg.dir.join(&removed.file_name));
                self.used_bytes = self.used_bytes.saturating_sub(removed.aligned_bytes);
                evicted += 1;
            }
        }
        evicted
    }

    fn persist_manifest(&self) -> Result<()> {
        let json = serde_json::to_vec_pretty(&self.manifest).context("serialize store manifest")?;
        let final_path = self.cfg.dir.join(MANIFEST_FILENAME);
        let tmp_path = staging_path_for(&self.cfg.dir, MANIFEST_FILENAME);
        let write_res = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp_path)
                .with_context(|| format!("create staging manifest {}", tmp_path.display()))?;
            file.write_all(&json)?;
            file.sync_all()?;
            Ok(())
        })();
        if let Err(err) = write_res {
            let _ = fs::remove_file(&tmp_path);
            return Err(err);
        }
        if let Err(err) = fs::rename(&tmp_path, &final_path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(err).with_context(|| {
                format!(
                    "atomic manifest rename {} -> {}",
                    tmp_path.display(),
                    final_path.display()
                )
            });
        }
        Ok(())
    }
}

/// 4,096-byte aligned heap buffer suitable for Linux `O_DIRECT` DMA transfers.
struct AlignedPageBuf {
    ptr: *mut u8,
    len: usize,
}

impl AlignedPageBuf {
    fn new_zeroed(len: usize) -> Result<Self> {
        ensure!(
            len > 0 && len.is_multiple_of(SNAPSHOT_PAGE_ALIGN),
            "AlignedPageBuf length {len} must be a non-zero multiple of {SNAPSHOT_PAGE_ALIGN}"
        );
        #[cfg(unix)]
        {
            let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: `SNAPSHOT_PAGE_ALIGN` (4096) is a power of two and a multiple
            // of `sizeof(void*)`; `len` is validated non-zero and page-aligned above.
            let rc = unsafe { libc::posix_memalign(&mut raw, SNAPSHOT_PAGE_ALIGN, len) };
            if rc != 0 || raw.is_null() {
                bail!("posix_memalign({SNAPSHOT_PAGE_ALIGN}, {len}) failed with errno {rc}");
            }
            let ptr = raw.cast::<u8>();
            // SAFETY: `ptr` points to `len` valid writable bytes freshly allocated
            // by `posix_memalign`; zero-initializing prevents leaking heap padding.
            unsafe {
                std::ptr::write_bytes(ptr, 0, len);
            }
            Ok(Self { ptr, len })
        }
        #[cfg(not(unix))]
        {
            let layout = std::alloc::Layout::from_size_align(len, SNAPSHOT_PAGE_ALIGN)
                .context("invalid AlignedPageBuf layout")?;
            // SAFETY: `layout` has non-zero size and power-of-two alignment.
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            if ptr.is_null() {
                bail!("alloc_zeroed({len}) failed");
            }
            Ok(Self { ptr, len })
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` is non-null, aligned, initialized across `0..self.len`,
        // and uniquely owned by `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `self.ptr` is non-null, aligned, initialized across `0..self.len`,
        // and `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedPageBuf {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        #[cfg(unix)]
        {
            // SAFETY: `self.ptr` was allocated by `libc::posix_memalign` in `new_zeroed`
            // and is freed exactly once here on drop.
            unsafe {
                libc::free(self.ptr.cast::<std::ffi::c_void>());
            }
        }
        #[cfg(not(unix))]
        {
            if let Ok(layout) = std::alloc::Layout::from_size_align(self.len, SNAPSHOT_PAGE_ALIGN) {
                // SAFETY: `self.ptr` was allocated with the identical `layout`.
                unsafe {
                    std::alloc::dealloc(self.ptr, layout);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SectionDesc {
    offset: u64,
    logical_len: u64,
    aligned_len: u64,
    crc32: u32,
}

fn serialize_aligned_snapshot(
    key: &PrefixSnapshotKey,
    payload: &PrefixSnapshotPayload,
) -> Result<AlignedPageBuf> {
    let total_bytes =
        usize::try_from(payload.aligned_file_bytes()?).context("snapshot image exceeds usize")?;
    let mut buf = AlignedPageBuf::new_zeroed(total_bytes)?;
    let sections = [
        payload.kv_pages.as_slice(),
        payload.qsa_selectors.as_slice(),
        payload.gdn_h_state.as_slice(),
        payload.gdn_conv_state.as_slice(),
        payload.ple_conv_state.as_slice(),
    ];

    let mut descs = [SectionDesc {
        offset: 0,
        logical_len: 0,
        aligned_len: 0,
        crc32: 0,
    }; SECTION_COUNT];

    let mut cursor = SNAPSHOT_PAGE_ALIGN;
    for (idx, data) in sections.iter().enumerate() {
        let aligned_len = align_up_usize(data.len(), SNAPSHOT_PAGE_ALIGN)?;
        let section_offset = if data.is_empty() { 0 } else { cursor as u64 };
        if !data.is_empty() {
            buf.as_mut_slice()[cursor..cursor + data.len()].copy_from_slice(data);
        }
        descs[idx] = SectionDesc {
            offset: section_offset,
            logical_len: data.len() as u64,
            aligned_len: aligned_len as u64,
            crc32: crc32_ieee(data),
        };
        cursor += aligned_len;
    }
    debug_assert_eq!(cursor, total_bytes);

    let payload_sha256 = sha256_bytes(&buf.as_slice()[SNAPSHOT_PAGE_ALIGN..total_bytes]);
    let content_digest = key.content_digest();

    let header = &mut buf.as_mut_slice()[..SNAPSHOT_PAGE_ALIGN];
    header[0..8].copy_from_slice(&FILE_MAGIC);
    header[8..12].copy_from_slice(&FILE_FORMAT_VERSION.to_le_bytes());
    // bytes [12..16] reserved for header_crc32 computed over [16..4096].
    header[16..48].copy_from_slice(&payload_sha256);
    header[48..80].copy_from_slice(&key.token_prefix_hash);
    header[80..112].copy_from_slice(&content_digest);
    header[112..116].copy_from_slice(&key.block_size.to_le_bytes());
    header[116..120].copy_from_slice(&key.kv_dtype.wire_tag().to_le_bytes());
    header[120..128].copy_from_slice(&key.grid_token_len.to_le_bytes());
    header[128..132].copy_from_slice(&payload.num_kv_layers.to_le_bytes());
    header[132..136].copy_from_slice(&payload.num_ssm_layers.to_le_bytes());
    header[136..140].copy_from_slice(&payload.num_ple_layers.to_le_bytes());
    let has_qsa = u32::from(!payload.qsa_selectors.is_empty());
    header[140..144].copy_from_slice(&has_qsa.to_le_bytes());

    let mut table_off = 144usize;
    for desc in &descs {
        header[table_off..table_off + 8].copy_from_slice(&desc.offset.to_le_bytes());
        header[table_off + 8..table_off + 16].copy_from_slice(&desc.logical_len.to_le_bytes());
        header[table_off + 16..table_off + 24].copy_from_slice(&desc.aligned_len.to_le_bytes());
        header[table_off + 24..table_off + 28].copy_from_slice(&desc.crc32.to_le_bytes());
        table_off += 32;
    }

    let hdr_crc = crc32_ieee(&header[16..SNAPSHOT_PAGE_ALIGN]);
    header[12..16].copy_from_slice(&hdr_crc.to_le_bytes());
    Ok(buf)
}

fn deserialize_and_verify_snapshot(
    expected_key: &PrefixSnapshotKey,
    image: &[u8],
) -> Result<PrefixSnapshotPayload> {
    ensure!(
        image.len() >= SNAPSHOT_PAGE_ALIGN && image.len().is_multiple_of(SNAPSHOT_PAGE_ALIGN),
        "corrupt snapshot image length {} (not 4096-byte aligned)",
        image.len()
    );
    let header = &image[..SNAPSHOT_PAGE_ALIGN];
    ensure!(
        header[0..8] == FILE_MAGIC,
        "prefix snapshot header magic mismatch"
    );
    let version = read_u32_le(&header[8..12]);
    ensure!(
        version == FILE_FORMAT_VERSION,
        "unsupported prefix snapshot format version {version}"
    );
    let expected_hdr_crc = read_u32_le(&header[12..16]);
    let actual_hdr_crc = crc32_ieee(&header[16..SNAPSHOT_PAGE_ALIGN]);
    ensure!(
        expected_hdr_crc == actual_hdr_crc,
        "prefix snapshot header CRC32 mismatch (expected {expected_hdr_crc:#010x}, got {actual_hdr_crc:#010x})"
    );

    let expected_payload_sha = &header[16..48];
    let actual_payload_sha = sha256_bytes(&image[SNAPSHOT_PAGE_ALIGN..]);
    ensure!(
        expected_payload_sha == actual_payload_sha.as_slice(),
        "prefix snapshot payload SHA-256 integrity check failed"
    );

    ensure!(
        header[48..80] == expected_key.token_prefix_hash,
        "prefix snapshot token_prefix_hash mismatch"
    );
    ensure!(
        header[80..112] == expected_key.content_digest(),
        "prefix snapshot content_id digest mismatch"
    );
    let block_size = read_u32_le(&header[112..116]);
    let kv_dtype = SnapshotKvDtype::from_wire_tag(read_u32_le(&header[116..120]))?;
    let grid_token_len = read_u64_le(&header[120..128]);
    ensure!(
        block_size == expected_key.block_size
            && kv_dtype == expected_key.kv_dtype
            && grid_token_len == expected_key.grid_token_len,
        "prefix snapshot header metadata disagreed with lookup key"
    );

    let num_kv_layers = read_u32_le(&header[128..132]);
    let num_ssm_layers = read_u32_le(&header[132..136]);
    let num_ple_layers = read_u32_le(&header[136..140]);

    let mut sections: [Vec<u8>; SECTION_COUNT] = Default::default();
    let mut table_off = 144usize;
    for out_vec in &mut sections {
        let offset = usize::try_from(read_u64_le(&header[table_off..table_off + 8]))
            .context("section offset overflow")?;
        let logical_len = usize::try_from(read_u64_le(&header[table_off + 8..table_off + 16]))
            .context("section logical_len overflow")?;
        let aligned_len = usize::try_from(read_u64_le(&header[table_off + 16..table_off + 24]))
            .context("section aligned_len overflow")?;
        let expected_crc = read_u32_le(&header[table_off + 24..table_off + 28]);
        table_off += 32;

        if logical_len == 0 {
            ensure!(aligned_len == 0, "empty section had non-zero aligned_len");
            ensure!(
                expected_crc == crc32_ieee(&[]),
                "empty section CRC32 mismatch"
            );
            continue;
        }
        ensure!(
            offset >= SNAPSHOT_PAGE_ALIGN && offset.is_multiple_of(SNAPSHOT_PAGE_ALIGN),
            "section offset {offset} violates 4096-byte alignment"
        );
        ensure!(
            aligned_len >= logical_len && aligned_len.is_multiple_of(SNAPSHOT_PAGE_ALIGN),
            "section aligned_len {aligned_len} invalid for logical_len {logical_len}"
        );
        let end = offset
            .checked_add(aligned_len)
            .context("section end overflow")?;
        ensure!(end <= image.len(), "section range exceeds snapshot image");
        let slice = &image[offset..offset + logical_len];
        let actual_crc = crc32_ieee(slice);
        ensure!(
            actual_crc == expected_crc,
            "section CRC32 mismatch (expected {expected_crc:#010x}, got {actual_crc:#010x})"
        );
        *out_vec = slice.to_vec();
    }

    let [
        kv_pages,
        qsa_selectors,
        gdn_h_state,
        gdn_conv_state,
        ple_conv_state,
    ] = sections;
    Ok(PrefixSnapshotPayload {
        kv_pages,
        qsa_selectors,
        gdn_h_state,
        gdn_conv_state,
        ple_conv_state,
        num_kv_layers,
        num_ssm_layers,
        num_ple_layers,
    })
}

fn write_aligned_file(path: &Path, buf: &AlignedPageBuf, mode: SnapshotIoMode) -> Result<bool> {
    #[cfg(target_os = "linux")]
    if mode == SnapshotIoMode::AutoDirect
        && let Ok(file) = open_linux_direct_write(path)
    {
        match pwrite_all(&file, buf.as_slice(), 0) {
            Ok(()) => {
                file.sync_all()
                    .with_context(|| format!("fsync O_DIRECT {}", path.display()))?;
                return Ok(true);
            }
            Err(err) if is_direct_io_unsupported(&err) => {
                drop(file);
                let _ = fs::remove_file(path);
            }
            Err(err) => {
                return Err(err).with_context(|| format!("pwrite O_DIRECT {}", path.display()));
            }
        }
    }

    let _ = mode;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create buffered snapshot {}", path.display()))?;
    pwrite_all(&file, buf.as_slice(), 0)
        .with_context(|| format!("pwrite buffered {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("fsync buffered {}", path.display()))?;
    drop_linux_page_cache(&file, buf.len());
    Ok(false)
}

fn read_aligned_file(
    path: &Path,
    expected_len: usize,
    mode: SnapshotIoMode,
) -> Result<AlignedPageBuf> {
    let mut buf = AlignedPageBuf::new_zeroed(expected_len)?;

    #[cfg(target_os = "linux")]
    if mode == SnapshotIoMode::AutoDirect
        && let Ok(file) = open_linux_direct_read(path)
    {
        match pread_exact(&file, buf.as_mut_slice(), 0) {
            Ok(()) => return Ok(buf),
            Err(err) if is_direct_io_unsupported(&err) => {}
            Err(err) => {
                return Err(err).with_context(|| format!("pread O_DIRECT {}", path.display()));
            }
        }
    }

    let _ = mode;
    let file = File::open(path).with_context(|| format!("open buffered {}", path.display()))?;
    pread_exact(&file, buf.as_mut_slice(), 0)
        .with_context(|| format!("pread buffered {}", path.display()))?;
    drop_linux_page_cache(&file, expected_len);
    Ok(buf)
}

#[cfg(target_os = "linux")]
fn open_linux_direct_write(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

#[cfg(target_os = "linux")]
fn open_linux_direct_read(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

#[cfg(target_os = "linux")]
fn is_direct_io_unsupported(err: &std::io::Error) -> bool {
    matches!(
        err.raw_os_error(),
        Some(libc::EINVAL | libc::EOPNOTSUPP | libc::ENOSYS)
    )
}

fn drop_linux_page_cache(file: &File, len: usize) {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        if let Ok(off_len) = libc::off_t::try_from(len) {
            // SAFETY: `file.as_raw_fd()` is a valid open descriptor owned by `file`,
            // and `POSIX_FADV_DONTNEED` is an advisory kernel hint with no memory hazard.
            let _ = unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, off_len, libc::POSIX_FADV_DONTNEED)
            };
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, len);
    }
}

#[cfg(unix)]
fn pwrite_all(file: &File, data: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(data, offset)
}

#[cfg(not(unix))]
fn pwrite_all(mut file: &File, data: &[u8], _offset: u64) -> std::io::Result<()> {
    file.write_all(data)
}

#[cfg(unix)]
fn pread_exact(file: &File, dst: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(dst, offset)
}

#[cfg(not(unix))]
fn pread_exact(mut file: &File, dst: &mut [u8], _offset: u64) -> std::io::Result<()> {
    use std::io::Read;
    file.read_exact(dst)
}

fn align_up_usize(value: usize, align: usize) -> Result<usize> {
    if value == 0 {
        return Ok(0);
    }
    let rem = value % align;
    if rem == 0 {
        Ok(value)
    } else {
        value
            .checked_add(align - rem)
            .context("aligned size overflow")
    }
}

fn staging_path_for(dir: &Path, final_name: &str) -> PathBuf {
    let seq = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    dir.join(format!(".{final_name}.tmp-{pid}-{seq}"))
}

fn cleanup_orphan_staging_files(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if let Some(s) = name.to_str()
            && s.starts_with('.')
            && s.contains(".tmp-")
        {
            let _ = fs::remove_file(entry.path());
        }
    }
}

fn default_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn read_u32_le(bytes: &[u8]) -> u32 {
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&bytes[..4]);
    u32::from_le_bytes(arr)
}

fn read_u64_le(bytes: &[u8]) -> u64 {
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(arr)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// IEEE 802.3 CRC32 checksum.
#[must_use]
pub fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

/// FIPS 180-4 SHA-256 digest.
#[must_use]
pub fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    fn compress(state: &mut [u32; 8], block: &[u8]) {
        let mut words = [0u32; 64];
        for (i, word) in words[..16].iter_mut().enumerate() {
            let mut b = [0u8; 4];
            b.copy_from_slice(&block[i * 4..i * 4 + 4]);
            *word = u32::from_be_bytes(b);
        }
        for i in 16..64 {
            let s0 = words[i - 15].rotate_right(7)
                ^ words[i - 15].rotate_right(18)
                ^ (words[i - 15] >> 3);
            let s1 = words[i - 2].rotate_right(17)
                ^ words[i - 2].rotate_right(19)
                ^ (words[i - 2] >> 10);
            words[i] = words[i - 16]
                .wrapping_add(s0)
                .wrapping_add(words[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(words[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, val) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(val);
        }
    }

    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut state = INITIAL;
    let mut chunks = bytes.chunks_exact(64);
    for block in &mut chunks {
        compress(&mut state, block);
    }
    let rem = chunks.remainder();
    let mut tail = [0u8; 128];
    tail[..rem.len()].copy_from_slice(rem);
    tail[rem.len()] = 0x80;
    let padded = if rem.len() < 56 { 64 } else { 128 };
    tail[padded - 8..padded].copy_from_slice(&bit_len.to_be_bytes());
    for block in tail[..padded].chunks_exact(64) {
        compress(&mut state, block);
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct TempDirGuard(PathBuf);

    impl TempDirGuard {
        fn new(tag: &str) -> Self {
            let seq = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "atlas-prefix-snap-{tag}-{}-{seq}",
                std::process::id()
            ));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn sample_payload(seed: u8) -> PrefixSnapshotPayload {
        PrefixSnapshotPayload {
            kv_pages: vec![seed; 9_000],
            qsa_selectors: vec![seed.wrapping_add(1); 1_536],
            gdn_h_state: vec![seed.wrapping_add(2); 8_192],
            gdn_conv_state: vec![seed.wrapping_add(3); 2_048],
            ple_conv_state: vec![seed.wrapping_add(4); 6_144],
            num_kv_layers: 12,
            num_ssm_layers: 36,
            num_ple_layers: 1,
        }
    }

    #[test]
    fn checksum_primitives_match_standard_vectors() {
        assert_eq!(crc32_ieee(b"123456789"), 0xcbf4_3926);
        assert_eq!(
            hex_encode(&sha256_bytes(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn round_trips_all_kv_dtypes_including_turboquant_across_reopen() {
        let tmp = TempDirGuard::new("dtypes");
        let tokens: Vec<u32> = (0..2048).collect();

        for (idx, dtype) in [
            SnapshotKvDtype::Bf16,
            SnapshotKvDtype::Fp8,
            SnapshotKvDtype::Nvfp4,
            SnapshotKvDtype::Turbo3,
            SnapshotKvDtype::Turbo4,
            SnapshotKvDtype::Turbo8,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                dtype.is_turboquant(),
                matches!(
                    dtype,
                    SnapshotKvDtype::Turbo3 | SnapshotKvDtype::Turbo4 | SnapshotKvDtype::Turbo8
                )
            );
            let parsed: SnapshotKvDtype = dtype.to_string().parse().unwrap();
            assert_eq!(parsed, dtype);

            let key =
                PrefixSnapshotKey::from_tokens("qwen4-digest-v1", dtype, 16, &tokens).unwrap();
            let payload = sample_payload(idx as u8 + 10);
            let expected_aligned = payload.aligned_file_bytes().unwrap();
            assert_eq!(expected_aligned % SNAPSHOT_PAGE_ALIGN as u64, 0);

            {
                let mut store =
                    PrefixSnapshotStore::open(PrefixSnapshotStoreConfig::new(tmp.path())).unwrap();
                let outcome = store.put(key.clone(), &payload).unwrap();
                match outcome {
                    StorePutOutcome::Stored { aligned_bytes, .. } => {
                        assert_eq!(aligned_bytes, expected_aligned);
                    }
                    other => panic!("unexpected outcome: {other:?}"),
                }
            }

            // Reopen from disk to prove restart persistence + 4096-byte file alignment.
            let mut reopened =
                PrefixSnapshotStore::open(PrefixSnapshotStoreConfig::new(tmp.path())).unwrap();
            assert_eq!(reopened.contains_bytes(&key), Some(expected_aligned));
            let snap_file = tmp.path().join(format!("{}.snap", key.content_id()));
            assert_eq!(
                fs::metadata(&snap_file).unwrap().len() % SNAPSHOT_PAGE_ALIGN as u64,
                0
            );
            let loaded = reopened.get(&key).unwrap().expect("snapshot must exist");
            assert_eq!(loaded, payload);
        }
    }

    #[test]
    fn detects_header_crc_and_payload_sha256_corruption() {
        let tmp = TempDirGuard::new("corruption");
        let tokens: Vec<u32> = (0..2048).collect();
        let key_hdr =
            PrefixSnapshotKey::from_tokens("m1", SnapshotKvDtype::Nvfp4, 16, &tokens).unwrap();
        let key_body =
            PrefixSnapshotKey::from_tokens("m2", SnapshotKvDtype::Turbo4, 16, &tokens).unwrap();
        let payload = sample_payload(0x55);

        let mut store =
            PrefixSnapshotStore::open(PrefixSnapshotStoreConfig::new(tmp.path())).unwrap();
        store.put(key_hdr.clone(), &payload).unwrap();
        store.put(key_body.clone(), &payload).unwrap();

        // Corrupt 1 byte inside header metadata.
        let hdr_path = tmp.path().join(format!("{}.snap", key_hdr.content_id()));
        let mut bytes = fs::read(&hdr_path).unwrap();
        bytes[120] ^= 0x01;
        fs::write(&hdr_path, &bytes).unwrap();
        let err = store.get(&key_hdr).unwrap_err();
        assert!(
            err.to_string().contains("CRC32"),
            "unexpected error: {err:#}"
        );

        // Corrupt 1 byte inside KV page payload.
        let body_path = tmp.path().join(format!("{}.snap", key_body.content_id()));
        let mut body_bytes = fs::read(&body_path).unwrap();
        body_bytes[SNAPSHOT_PAGE_ALIGN + 17] ^= 0x80;
        fs::write(&body_path, &body_bytes).unwrap();
        let err = store.get(&key_body).unwrap_err();
        assert!(
            err.to_string().contains("SHA-256"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn enforces_lru_byte_budget_and_daily_write_cap_with_rollover() {
        let tmp = TempDirGuard::new("evict-cap");
        let payload = sample_payload(1);
        let entry_bytes = payload.aligned_file_bytes().unwrap();

        let clock = Arc::new(AtomicU64::new(10 * SECONDS_PER_DAY + 100));
        let clock_clone = Arc::clone(&clock);
        let mut cfg = PrefixSnapshotStoreConfig::new(tmp.path());
        cfg.max_bytes = entry_bytes * 2;
        cfg.daily_write_cap_bytes = entry_bytes * 3;
        cfg.io_mode = SnapshotIoMode::Buffered;

        let mut store = PrefixSnapshotStore::open_with_clock(
            cfg,
            Box::new(move || clock_clone.load(Ordering::SeqCst)),
        )
        .unwrap();

        let t1: Vec<u32> = (0..64).collect();
        let t2: Vec<u32> = (100..164).collect();
        let t3: Vec<u32> = (200..264).collect();
        let t4: Vec<u32> = (300..364).collect();
        let k1 = PrefixSnapshotKey::from_tokens("m", SnapshotKvDtype::Nvfp4, 16, &t1).unwrap();
        let k2 = PrefixSnapshotKey::from_tokens("m", SnapshotKvDtype::Nvfp4, 16, &t2).unwrap();
        let k3 = PrefixSnapshotKey::from_tokens("m", SnapshotKvDtype::Nvfp4, 16, &t3).unwrap();
        let k4 = PrefixSnapshotKey::from_tokens("m", SnapshotKvDtype::Nvfp4, 16, &t4).unwrap();

        assert!(matches!(
            store.put(k1.clone(), &payload).unwrap(),
            StorePutOutcome::Stored { .. }
        ));
        assert!(matches!(
            store.put(k2.clone(), &payload).unwrap(),
            StorePutOutcome::Stored { .. }
        ));
        // Touch k1 so k2 becomes the LRU victim when k3 arrives.
        assert!(store.get(&k1).unwrap().is_some());

        match store.put(k3.clone(), &payload).unwrap() {
            StorePutOutcome::Stored {
                evicted_entries, ..
            } => assert_eq!(evicted_entries, 1),
            other => panic!("expected Stored, got {other:?}"),
        }
        assert!(store.contains_bytes(&k1).is_some(), "hot k1 must survive");
        assert!(
            store.contains_bytes(&k2).is_none(),
            "cold k2 must be evicted"
        );
        assert!(store.contains_bytes(&k3).is_some());
        assert!(store.used_bytes() <= entry_bytes * 2);

        // Daily write cap (3 entries) is now exhausted for day 10.
        assert_eq!(
            store.put(k4.clone(), &payload).unwrap(),
            StorePutOutcome::DailyWriteCapReached {
                attempted_bytes: entry_bytes,
                remaining_daily_bytes: 0,
            }
        );

        // Advance clock to next UTC day -> daily write budget resets cleanly.
        clock.store(11 * SECONDS_PER_DAY + 5, Ordering::SeqCst);
        assert_eq!(store.daily_bytes_written(), 0);
        assert!(matches!(
            store.put(k4.clone(), &payload).unwrap(),
            StorePutOutcome::Stored { .. }
        ));
        assert!(store.contains_bytes(&k4).is_some());
    }

    #[test]
    fn tensorfold_live_cost_pricing_rule_admits_only_faster_than_prefill() {
        // 16,384 tokens @ 6,200 tok/s = 2.642s prefill vs 300 MiB @ 6.5 GB/s = 0.048s disk -> fault!
        assert!(should_restore_from_disk(
            16_384,
            300 * 1024 * 1024,
            6.5e9,
            6_200.0
        ));
        // 32 tokens @ 6,200 tok/s = 5.16ms prefill vs 75 MiB GDN state @ 6.5 GB/s = 12.1ms disk -> recompute!
        assert!(!should_restore_from_disk(
            32,
            75 * 1024 * 1024,
            6.5e9,
            6_200.0
        ));
        // Degenerate / boundary inputs fail closed.
        assert!(!should_restore_from_disk(0, 4096, 6.5e9, 6_200.0));
        assert!(!should_restore_from_disk(4096, 4096, 0.0, 6_200.0));
        assert!(!should_restore_from_disk(4096, 4096, f64::NAN, 6_200.0));
        assert!(!should_restore_from_disk(4096, 4096, 6.5e9, 0.0));
        assert!(!should_restore_from_disk(4096, 4096, 6.5e9, f64::INFINITY));

        let tmp = TempDirGuard::new("cost-gate");
        let tokens: Vec<u32> = (0..64).collect();
        let key = PrefixSnapshotKey::from_tokens("m", SnapshotKvDtype::Bf16, 16, &tokens).unwrap();
        let payload = sample_payload(9);
        let mut store =
            PrefixSnapshotStore::open(PrefixSnapshotStoreConfig::new(tmp.path())).unwrap();
        store.put(key.clone(), &payload).unwrap();

        // Slow disk (1 KB/s) vs fast prefill (100k tok/s) -> cost gate skips disk read.
        assert!(
            store
                .load_if_cost_effective(&key, 1_000.0, 100_000.0)
                .unwrap()
                .is_none()
        );
        // Fast NVMe (6 GB/s) vs realistic prefill -> cost gate faults snapshot in.
        assert_eq!(
            store
                .load_if_cost_effective(&key, 6.0e9, 5_000.0)
                .unwrap()
                .as_ref(),
            Some(&payload)
        );
    }

    #[test]
    fn key_rejects_unaligned_or_empty_token_prefixes() {
        assert!(PrefixSnapshotKey::from_tokens("", SnapshotKvDtype::Bf16, 16, &[1; 16]).is_err());
        assert!(PrefixSnapshotKey::from_tokens("m", SnapshotKvDtype::Bf16, 0, &[1; 16]).is_err());
        assert!(PrefixSnapshotKey::from_tokens("m", SnapshotKvDtype::Bf16, 16, &[]).is_err());
        assert!(PrefixSnapshotKey::from_tokens("m", SnapshotKvDtype::Bf16, 16, &[1; 17]).is_err());
    }
}
