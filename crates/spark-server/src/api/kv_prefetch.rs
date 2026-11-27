// SPDX-License-Identifier: AGPL-3.0-only

//! TripleKV (`arXiv:2611.25112`) agentic KV cache prefetch endpoint (`POST /v1/kv_cache/prefetch`),
//! Eq. (1) tool-latency issue-time scheduler, per-tool latency estimator, and async L3 -> L2
//! staging bridge backed by [`spark_storage::PrefixSnapshotStore`] and [`spark_runtime::radix_tree::RadixTree`].

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use spark_runtime::prefix_cache::{DEFAULT_PREFETCH_TTL, ProtectionClass};
use spark_runtime::radix_tree::RadixTree;
use spark_storage::{
    PrefixSnapshotKey, PrefixSnapshotPayload, PrefixSnapshotStore, PrefixSnapshotStoreConfig,
    SnapshotKvDtype,
};
use tokio::sync::Semaphore;

use super::compact::openai_error_response;
use crate::main_modules::model_host::ModelHost;

/// Default NVMe sequential read bandwidth assumed when caller omits `nvme_bandwidth_bps` (3.5 GB/s).
pub const DEFAULT_NVME_READ_BPS: f64 = 3.5e9;

/// Default safety margin `epsilon` (seconds) in TripleKV Eq. (1) so prefetch lands before arrival.
pub const DEFAULT_PREFETCH_SAFETY_MARGIN_SECS: f64 = 0.05;

/// Maximum client-requested TTL to prevent unbounded eternal pinning (1 hour).
pub const MAX_PREFETCH_TTL_SECS: u64 = 3600;

/// Maximum number of concurrent async L3 -> L2 snapshot reads.
const MAX_CONCURRENT_PREFETCH_IO: usize = 4;

/// Maximum staged warm L2 payloads retained concurrently in memory.
const DEFAULT_MAX_STAGED_ENTRIES: usize = 64;

/// Maximum total logical bytes staged in the warm L2 prefetch buffer (512 MiB).
const DEFAULT_MAX_STAGED_BYTES: u64 = 512 * 1024 * 1024;

/// Maximum historical tool-duration samples stored per tool name.
const MAX_TOOL_SAMPLES_PER_NAME: usize = 32;

/// Maximum distinct tool names tracked to bound memory against unbounded caller strings.
const MAX_TRACKED_TOOL_NAMES: usize = 256;

/// Inputs to TripleKV Eq. (1): `t_issue = t_end + q - S_L3 / B - eps`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefetchScheduleInput {
    /// Predicted remaining tool execution time `t_end - t_now` in seconds.
    pub remaining_tool_secs: f64,
    /// Estimated scheduler queue wait `q` in seconds.
    pub queue_wait_secs: f64,
    /// Snapshot byte size `S_L3` on NVMe/L3 storage.
    pub snapshot_bytes: u64,
    /// Effective L3 -> L2 read bandwidth `B` in bytes/sec.
    pub nvme_bandwidth_bps: f64,
    /// Safety margin `epsilon` in seconds so transfer completes ahead of turn arrival.
    pub safety_margin_secs: f64,
}

/// Computed TripleKV Eq. (1) prefetch launch timing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefetchSchedule {
    /// Estimated L3 -> L2 transfer duration `S_L3 / B` in seconds.
    pub io_load_secs: f64,
    /// Delay from `now` before launching L3 -> L2 transfer (`max(0, t_end + q - S_L3/B - eps)`).
    pub issue_delay: Duration,
    /// True when the transfer should launch immediately without waiting.
    pub immediate: bool,
}

/// Compute TripleKV Eq. (1) relative issue delay:
/// `delta_t_issue = max(0, remaining_tool_secs + queue_wait_secs - snapshot_bytes / B - eps)`.
///
/// # Errors
/// Returns an error message when bandwidth is non-positive/non-finite or timing inputs are invalid.
pub fn compute_prefetch_schedule(
    input: PrefetchScheduleInput,
) -> std::result::Result<PrefetchSchedule, String> {
    if !input.remaining_tool_secs.is_finite() || input.remaining_tool_secs < 0.0 {
        return Err(format!(
            "remaining_tool_secs={} must be finite and >= 0",
            input.remaining_tool_secs
        ));
    }
    if !input.queue_wait_secs.is_finite() || input.queue_wait_secs < 0.0 {
        return Err(format!(
            "queue_wait_secs={} must be finite and >= 0",
            input.queue_wait_secs
        ));
    }
    if !input.nvme_bandwidth_bps.is_finite() || input.nvme_bandwidth_bps <= 0.0 {
        return Err(format!(
            "nvme_bandwidth_bps={} must be finite and > 0",
            input.nvme_bandwidth_bps
        ));
    }
    if !input.safety_margin_secs.is_finite() || input.safety_margin_secs < 0.0 {
        return Err(format!(
            "safety_margin_secs={} must be finite and >= 0",
            input.safety_margin_secs
        ));
    }

    let io_load_secs = (input.snapshot_bytes as f64) / input.nvme_bandwidth_bps;
    let raw_delay =
        input.remaining_tool_secs + input.queue_wait_secs - io_load_secs - input.safety_margin_secs;
    if raw_delay <= 0.0 {
        Ok(PrefetchSchedule {
            io_load_secs,
            issue_delay: Duration::ZERO,
            immediate: true,
        })
    } else {
        Ok(PrefetchSchedule {
            io_load_secs,
            issue_delay: Duration::from_secs_f64(raw_delay),
            immediate: false,
        })
    }
}

/// Per-tool streaming quantile latency predictor (TripleKV §3.1, Table 2).
#[derive(Debug, Default)]
pub struct ToolLatencyPredictor {
    windows: HashMap<String, VecDeque<f64>>,
}

impl ToolLatencyPredictor {
    /// Create an empty predictor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one observed tool execution duration in seconds (ignored if non-finite or negative).
    pub fn record_observation(&mut self, tool_name: &str, duration_secs: f64) {
        let name = tool_name.trim();
        if name.is_empty() || !duration_secs.is_finite() || duration_secs < 0.0 {
            return;
        }
        if !self.windows.contains_key(name) && self.windows.len() >= MAX_TRACKED_TOOL_NAMES {
            return;
        }
        let ring = self
            .windows
            .entry(name.to_string())
            .or_insert_with(|| VecDeque::with_capacity(MAX_TOOL_SAMPLES_PER_NAME));
        if ring.len() >= MAX_TOOL_SAMPLES_PER_NAME {
            ring.pop_front();
        }
        ring.push_back(duration_secs);
    }

    /// Estimate tool duration at `quantile in [0.0, 1.0]` (e.g. `0.35` conservative lower quantile
    /// so prefetches land before the agent turn arrives).
    #[must_use]
    pub fn predict_secs(&self, tool_name: &str, quantile: f64) -> Option<f64> {
        let ring = self.windows.get(tool_name.trim())?;
        if ring.is_empty() {
            return None;
        }
        let mut sorted: Vec<f64> = ring.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let q = quantile.clamp(0.0, 1.0);
        let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
        Some(sorted[idx])
    }

    /// Number of distinct tools currently tracked.
    #[must_use]
    pub fn tracked_tools(&self) -> usize {
        self.windows.len()
    }
}

struct StagedEntry {
    payload: PrefixSnapshotPayload,
    logical_bytes: u64,
    expires_at: Instant,
    access_seq: u64,
}

struct StagedPrefetchCache {
    entries: HashMap<String, StagedEntry>,
    used_bytes: u64,
    max_entries: usize,
    max_bytes: u64,
    access_seq: u64,
}

impl StagedPrefetchCache {
    fn new(max_entries: usize, max_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            used_bytes: 0,
            max_entries,
            max_bytes,
            access_seq: 0,
        }
    }

    fn purge_expired(&mut self, now: Instant) {
        let mut reclaimed = 0u64;
        self.entries.retain(|_, entry| {
            let keep = now < entry.expires_at;
            if !keep {
                reclaimed = reclaimed.saturating_add(entry.logical_bytes);
            }
            keep
        });
        self.used_bytes = self.used_bytes.saturating_sub(reclaimed);
    }

    fn is_fresh(&mut self, content_id: &str, now: Instant) -> bool {
        self.purge_expired(now);
        if let Some(entry) = self.entries.get_mut(content_id) {
            self.access_seq = self.access_seq.saturating_add(1);
            entry.access_seq = self.access_seq;
            true
        } else {
            false
        }
    }

    fn insert(&mut self, content_id: String, payload: PrefixSnapshotPayload, expires_at: Instant) {
        let now = Instant::now();
        self.purge_expired(now);
        let logical_bytes = payload.logical_bytes();
        if logical_bytes > self.max_bytes {
            return;
        }
        if let Some(prev) = self.entries.remove(&content_id) {
            self.used_bytes = self.used_bytes.saturating_sub(prev.logical_bytes);
        }
        while !self.entries.is_empty()
            && (self.entries.len() >= self.max_entries
                || self.used_bytes.saturating_add(logical_bytes) > self.max_bytes)
        {
            let victim = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.access_seq)
                .map(|(k, _)| k.clone());
            let Some(victim_key) = victim else {
                break;
            };
            if let Some(removed) = self.entries.remove(&victim_key) {
                self.used_bytes = self.used_bytes.saturating_sub(removed.logical_bytes);
            }
        }
        self.access_seq = self.access_seq.saturating_add(1);
        self.used_bytes = self.used_bytes.saturating_add(logical_bytes);
        self.entries.insert(
            content_id,
            StagedEntry {
                payload,
                logical_bytes,
                expires_at,
                access_seq: self.access_seq,
            },
        );
    }

    fn take_if_fresh(&mut self, content_id: &str, now: Instant) -> Option<PrefixSnapshotPayload> {
        self.purge_expired(now);
        let entry = self.entries.remove(content_id)?;
        self.used_bytes = self.used_bytes.saturating_sub(entry.logical_bytes);
        Some(entry.payload)
    }
}

/// Snapshot of [`KvPrefetcher`] live telemetry exposed by `GET /v1/kv_cache/prefetch`.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct KvPrefetchStatus {
    pub store_enabled: bool,
    pub radix_tree_attached: bool,
    pub staged_entries: usize,
    pub staged_bytes: u64,
    pub tracked_tools: usize,
    pub scheduled_total: u64,
    pub completed_total: u64,
    pub hit_hot_total: u64,
    pub miss_total: u64,
    pub errors_total: u64,
}

/// Outcome returned by [`KvPrefetcher::schedule_prefetch`].
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct KvPrefetchResponse {
    pub status: &'static str,
    pub content_id: String,
    pub aligned_tokens: usize,
    pub block_size: u32,
    pub snapshot_bytes: Option<u64>,
    pub issue_delay_secs: f64,
    pub io_load_secs: f64,
    pub ttl_secs: u64,
    pub radix_nodes_protected: usize,
}

/// Request body for `POST /v1/kv_cache/prefetch`.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct KvPrefetchRequest {
    /// Pre-tokenized prefix IDs (preferred; zero tokenization overhead).
    #[serde(default)]
    pub token_ids: Option<Vec<u32>>,
    /// Raw prompt string tokenized with the currently loaded model when `token_ids` is absent.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Model identifier / checkpoint digest (defaults to currently loaded model name).
    #[serde(default)]
    pub model_digest: Option<String>,
    /// KV cache dtype (`bf16`, `fp8`, `nvfp4`, etc.; defaults to server's configured dtype).
    #[serde(default)]
    pub kv_dtype: Option<String>,
    /// KV block size in tokens (defaults to server's `--block-size` or `16`).
    #[serde(default)]
    pub block_size: Option<u32>,
    /// Protection TTL in seconds (defaults to 120s, capped at 3600s).
    #[serde(default)]
    pub ttl_secs: Option<u64>,
    /// Optional tool identifier used by [`ToolLatencyPredictor`].
    #[serde(default)]
    pub tool_name: Option<String>,
    /// Optional remaining tool duration estimate `t_end - t_now` in seconds.
    #[serde(default)]
    pub predicted_tool_remaining_secs: Option<f64>,
    /// Optional scheduler queue wait `q` in seconds (default `0.0`).
    #[serde(default)]
    pub estimated_queue_wait_secs: Option<f64>,
    /// Optional NVMe read bandwidth `B` in bytes/sec (default 3.5 GB/s).
    #[serde(default)]
    pub nvme_bandwidth_bps: Option<f64>,
    /// Optional Eq. (1) safety margin `epsilon` in seconds (default `0.05`).
    #[serde(default)]
    pub safety_margin_secs: Option<f64>,
    /// Optional completed tool duration feedback to refine future predictions for `tool_name`.
    #[serde(default)]
    pub tool_duration_observation_secs: Option<f64>,
}

/// Normalized, validated prefetch parameters handed to [`KvPrefetcher::schedule_prefetch`].
#[derive(Clone, Debug)]
pub struct PrefetchParams {
    pub tokens: Vec<u32>,
    pub model_digest: String,
    pub kv_dtype: SnapshotKvDtype,
    pub block_size: u32,
    pub ttl: Duration,
    pub tool_name: Option<String>,
    pub predicted_tool_remaining_secs: Option<f64>,
    pub queue_wait_secs: f64,
    pub nvme_bandwidth_bps: f64,
    pub safety_margin_secs: f64,
    pub tool_duration_observation_secs: Option<f64>,
}

/// Async TripleKV L3 -> L2 prefetch coordinator wired into [`ModelHost`].
pub struct KvPrefetcher {
    store: Option<Arc<Mutex<PrefixSnapshotStore>>>,
    radix_tree: RwLock<Option<Arc<RadixTree>>>,
    staged: Arc<Mutex<StagedPrefetchCache>>,
    predictor: Mutex<ToolLatencyPredictor>,
    io_permits: Arc<Semaphore>,
    scheduled_total: Arc<AtomicU64>,
    completed_total: Arc<AtomicU64>,
    hit_hot_total: Arc<AtomicU64>,
    miss_total: Arc<AtomicU64>,
    errors_total: Arc<AtomicU64>,
}

impl KvPrefetcher {
    /// Construct a disabled prefetcher (zero files opened, zero background work).
    #[must_use]
    pub fn disabled() -> Self {
        Self::with_store_and_tree(None, None)
    }

    /// Construct from `FLUX_PREFIX_CACHE_DIR` when configured, falling back cleanly to disabled
    /// when unset (bit-identical default baseline).
    #[must_use]
    pub fn from_env_best_effort() -> Self {
        match PrefixSnapshotStoreConfig::from_env_or_path(None) {
            Ok(Some(cfg)) => match PrefixSnapshotStore::open(cfg) {
                Ok(store) => {
                    tracing::info!(
                        dir = %store.config().dir.display(),
                        entries = store.len(),
                        "TripleKV /v1/kv_cache/prefetch backed by PrefixSnapshotStore"
                    );
                    Self::with_store_and_tree(Some(store), None)
                }
                Err(err) => {
                    tracing::warn!(
                        "TripleKV failed to open PrefixSnapshotStore (prefetch L3 disabled): {err:#}"
                    );
                    Self::disabled()
                }
            },
            Ok(None) => Self::disabled(),
            Err(err) => {
                tracing::warn!(
                    "TripleKV invalid FLUX_PREFIX_CACHE_* configuration (prefetch L3 disabled): {err:#}"
                );
                Self::disabled()
            }
        }
    }

    /// Construct with explicit optional snapshot store and live [`RadixTree`].
    #[must_use]
    pub fn with_store_and_tree(
        store: Option<PrefixSnapshotStore>,
        radix_tree: Option<Arc<RadixTree>>,
    ) -> Self {
        Self {
            store: store.map(|s| Arc::new(Mutex::new(s))),
            radix_tree: RwLock::new(radix_tree),
            staged: Arc::new(Mutex::new(StagedPrefetchCache::new(
                DEFAULT_MAX_STAGED_ENTRIES,
                DEFAULT_MAX_STAGED_BYTES,
            ))),
            predictor: Mutex::new(ToolLatencyPredictor::new()),
            io_permits: Arc::new(Semaphore::new(MAX_CONCURRENT_PREFETCH_IO)),
            scheduled_total: Arc::new(AtomicU64::new(0)),
            completed_total: Arc::new(AtomicU64::new(0)),
            hit_hot_total: Arc::new(AtomicU64::new(0)),
            miss_total: Arc::new(AtomicU64::new(0)),
            errors_total: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Attach or replace the live [`RadixTree`] so prefetches tag `ProtectionClass::Prefetched`.
    pub fn attach_radix_tree(&self, tree: Option<Arc<RadixTree>>) {
        *self.radix_tree.write() = tree;
    }

    /// Consume a staged warm L2 snapshot payload if present and unexpired.
    #[must_use]
    pub fn take_staged_payload(&self, content_id: &str) -> Option<PrefixSnapshotPayload> {
        self.staged.lock().take_if_fresh(content_id, Instant::now())
    }

    /// Return live status metrics for observability.
    #[must_use]
    pub fn status(&self) -> KvPrefetchStatus {
        let mut staged = self.staged.lock();
        staged.purge_expired(Instant::now());
        KvPrefetchStatus {
            store_enabled: self.store.is_some(),
            radix_tree_attached: self.radix_tree.read().is_some(),
            staged_entries: staged.entries.len(),
            staged_bytes: staged.used_bytes,
            tracked_tools: self.predictor.lock().tracked_tools(),
            scheduled_total: self.scheduled_total.load(Ordering::Relaxed),
            completed_total: self.completed_total.load(Ordering::Relaxed),
            hit_hot_total: self.hit_hot_total.load(Ordering::Relaxed),
            miss_total: self.miss_total.load(Ordering::Relaxed),
            errors_total: self.errors_total.load(Ordering::Relaxed),
        }
    }

    /// Schedule an async TripleKV L3 -> L2 prefetch and tag matching live `RadixTree` nodes.
    ///
    /// # Errors
    /// Returns `(StatusCode, String)` on invalid parameters (empty tokens, zero block_size,
    /// invalid Eq. (1) schedule inputs).
    pub fn schedule_prefetch(
        &self,
        params: PrefetchParams,
    ) -> std::result::Result<(StatusCode, KvPrefetchResponse), (StatusCode, String)> {
        if params.block_size == 0 {
            return Err((
                StatusCode::BAD_REQUEST,
                "block_size must be > 0".to_string(),
            ));
        }
        let block_size_usize = params.block_size as usize;
        let aligned_tokens = (params.tokens.len() / block_size_usize) * block_size_usize;
        if aligned_tokens == 0 {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "prefix length ({}) must contain at least one full block (block_size={})",
                    params.tokens.len(),
                    params.block_size
                ),
            ));
        }

        // Update and query the per-tool latency predictor.
        let predicted_remaining = {
            let mut pred = self.predictor.lock();
            if let (Some(tool), Some(obs)) = (
                params.tool_name.as_deref(),
                params.tool_duration_observation_secs,
            ) {
                pred.record_observation(tool, obs);
            }
            match params.predicted_tool_remaining_secs {
                Some(explicit) => explicit,
                None => params
                    .tool_name
                    .as_deref()
                    .and_then(|name| pred.predict_secs(name, 0.35))
                    .unwrap_or(0.0),
            }
        };

        let aligned_slice = &params.tokens[..aligned_tokens];
        let key = PrefixSnapshotKey::from_tokens(
            params.model_digest,
            params.kv_dtype,
            params.block_size,
            aligned_slice,
        )
        .map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;
        let content_id = key.content_id();

        let now = Instant::now();
        let ttl = params
            .ttl
            .min(Duration::from_secs(MAX_PREFETCH_TTL_SECS))
            .max(Duration::from_secs(1));
        let protection = ProtectionClass::prefetched_until(now, ttl);

        // Immediately protect any already-resident matching blocks in the live RadixTree.
        let radix_nodes_protected = self.radix_tree.read().as_ref().map_or(0, |tree| {
            tree.set_prefix_protection(aligned_slice, block_size_usize, protection)
        });

        // Fast path: payload is already staged hot in warm L2 memory.
        if self.staged.lock().is_fresh(&content_id, now) {
            self.hit_hot_total.fetch_add(1, Ordering::Relaxed);
            return Ok((
                StatusCode::OK,
                KvPrefetchResponse {
                    status: "already_hot",
                    content_id,
                    aligned_tokens,
                    block_size: params.block_size,
                    snapshot_bytes: None,
                    issue_delay_secs: 0.0,
                    io_load_secs: 0.0,
                    ttl_secs: ttl.as_secs(),
                    radix_nodes_protected,
                },
            ));
        }

        let Some(store) = self.store.as_ref().map(Arc::clone) else {
            let sched = compute_prefetch_schedule(PrefetchScheduleInput {
                remaining_tool_secs: predicted_remaining,
                queue_wait_secs: params.queue_wait_secs,
                snapshot_bytes: 0,
                nvme_bandwidth_bps: params.nvme_bandwidth_bps,
                safety_margin_secs: params.safety_margin_secs,
            })
            .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
            return Ok((
                StatusCode::OK,
                KvPrefetchResponse {
                    status: "store_disabled",
                    content_id,
                    aligned_tokens,
                    block_size: params.block_size,
                    snapshot_bytes: None,
                    issue_delay_secs: sched.issue_delay.as_secs_f64(),
                    io_load_secs: sched.io_load_secs,
                    ttl_secs: ttl.as_secs(),
                    radix_nodes_protected,
                },
            ));
        };

        let snapshot_bytes = store.lock().contains_bytes(&key);
        let Some(bytes) = snapshot_bytes else {
            self.miss_total.fetch_add(1, Ordering::Relaxed);
            return Ok((
                StatusCode::OK,
                KvPrefetchResponse {
                    status: "miss_not_in_store",
                    content_id,
                    aligned_tokens,
                    block_size: params.block_size,
                    snapshot_bytes: None,
                    issue_delay_secs: 0.0,
                    io_load_secs: 0.0,
                    ttl_secs: ttl.as_secs(),
                    radix_nodes_protected,
                },
            ));
        };

        let sched = compute_prefetch_schedule(PrefetchScheduleInput {
            remaining_tool_secs: predicted_remaining,
            queue_wait_secs: params.queue_wait_secs,
            snapshot_bytes: bytes,
            nvme_bandwidth_bps: params.nvme_bandwidth_bps,
            safety_margin_secs: params.safety_margin_secs,
        })
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;

        self.scheduled_total.fetch_add(1, Ordering::Relaxed);

        let staged = Arc::clone(&self.staged);
        let io_permits = Arc::clone(&self.io_permits);
        let completed_total = Arc::clone(&self.completed_total);
        let errors_total = Arc::clone(&self.errors_total);
        let radix_tree = self.radix_tree.read().clone();
        let aligned_owned = aligned_slice.to_vec();
        let task_content_id = content_id.clone();
        let delay = sched.issue_delay;

        tokio::spawn(async move {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let Ok(_permit) = io_permits.acquire_owned().await else {
                return;
            };
            let load_result = tokio::task::spawn_blocking(move || store.lock().get(&key)).await;
            match load_result {
                Ok(Ok(Some(payload))) => {
                    let finish_now = Instant::now();
                    staged
                        .lock()
                        .insert(task_content_id, payload, finish_now + ttl);
                    if let Some(tree) = radix_tree {
                        let _ = tree.set_prefix_protection(
                            &aligned_owned,
                            block_size_usize,
                            ProtectionClass::prefetched_until(finish_now, ttl),
                        );
                    }
                    completed_total.fetch_add(1, Ordering::Relaxed);
                }
                Ok(Ok(None)) => {}
                Ok(Err(err)) => {
                    tracing::warn!("TripleKV async prefetch load failed: {err:#}");
                    errors_total.fetch_add(1, Ordering::Relaxed);
                }
                Err(err) => {
                    tracing::warn!("TripleKV prefetch blocking task failed: {err}");
                    errors_total.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        Ok((
            StatusCode::ACCEPTED,
            KvPrefetchResponse {
                status: "scheduled",
                content_id,
                aligned_tokens,
                block_size: params.block_size,
                snapshot_bytes: Some(bytes),
                issue_delay_secs: sched.issue_delay.as_secs_f64(),
                io_load_secs: sched.io_load_secs,
                ttl_secs: ttl.as_secs(),
                radix_nodes_protected,
            },
        ))
    }
}

/// `POST /v1/kv_cache/prefetch` — TripleKV async L3 -> L2 prefix prefetch & protection endpoint.
pub async fn kv_cache_prefetch(
    State(host): State<Arc<ModelHost>>,
    payload: std::result::Result<Json<KvPrefetchRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(json) => json,
        Err(err) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid JSON body: {err}"),
            );
        }
    };

    let current_model = host.current();
    let args = host.args();

    let tokens = match (req.token_ids, req.prompt) {
        (Some(ids), _) if !ids.is_empty() => ids,
        (Some(_), _) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                "token_ids must not be empty".to_string(),
            );
        }
        (None, Some(prompt)) if !prompt.is_empty() => {
            let Some(state) = current_model.as_ref() else {
                return openai_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "prompt tokenization requires a loaded model; supply token_ids directly"
                        .to_string(),
                );
            };
            match state.tokenizer.encode(&prompt) {
                Ok(ids) if !ids.is_empty() => ids,
                Ok(_) => {
                    return openai_error_response(
                        StatusCode::BAD_REQUEST,
                        "prompt tokenized to zero tokens".to_string(),
                    );
                }
                Err(err) => {
                    return openai_error_response(
                        StatusCode::BAD_REQUEST,
                        format!("prompt tokenization failed: {err:#}"),
                    );
                }
            }
        }
        _ => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                "either non-empty token_ids or prompt must be provided".to_string(),
            );
        }
    };

    let model_digest = req
        .model_digest
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| current_model.as_ref().map(|s| s.model_name.clone()))
        .unwrap_or_else(|| "default".to_string());

    let dtype_raw = req
        .kv_dtype
        .as_deref()
        .or_else(|| args.as_ref().map(|a| a.kv_cache_dtype.as_str()))
        .unwrap_or("nvfp4");
    let kv_dtype = match dtype_raw.parse::<SnapshotKvDtype>() {
        Ok(dt) => dt,
        Err(err) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid kv_dtype {dtype_raw:?}: {err}"),
            );
        }
    };

    let block_size = req
        .block_size
        .unwrap_or_else(|| args.as_ref().map_or(16u32, |a| a.block_size.max(1) as u32));
    let ttl = Duration::from_secs(
        req.ttl_secs
            .unwrap_or(DEFAULT_PREFETCH_TTL.as_secs())
            .clamp(1, MAX_PREFETCH_TTL_SECS),
    );

    let params = PrefetchParams {
        tokens,
        model_digest,
        kv_dtype,
        block_size,
        ttl,
        tool_name: req.tool_name,
        predicted_tool_remaining_secs: req.predicted_tool_remaining_secs,
        queue_wait_secs: req.estimated_queue_wait_secs.unwrap_or(0.0),
        nvme_bandwidth_bps: req.nvme_bandwidth_bps.unwrap_or(DEFAULT_NVME_READ_BPS),
        safety_margin_secs: req
            .safety_margin_secs
            .unwrap_or(DEFAULT_PREFETCH_SAFETY_MARGIN_SECS),
        tool_duration_observation_secs: req.tool_duration_observation_secs,
    };

    match host.kv_prefetcher().schedule_prefetch(params) {
        Ok((status, body)) => (status, Json(body)).into_response(),
        Err((status, msg)) => openai_error_response(status, msg),
    }
}

/// `GET /v1/kv_cache/prefetch` — live TripleKV prefetch buffer and telemetry status.
pub async fn kv_cache_prefetch_status(State(host): State<Arc<ModelHost>>) -> Response {
    Json(host.kv_prefetcher().status()).into_response()
}

#[cfg(test)]
#[path = "kv_prefetch_tests.rs"]
mod tests;
