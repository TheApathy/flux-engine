// SPDX-License-Identifier: AGPL-3.0-only

//! Batched speculative-verify PLE (`ATLAS_QWEN4_PLE_VERIFY_BATCH=1`, fn-mtp).
//!
//! The per-row verify loop calls [`Qwen4PleLayer::forward_token`] once per
//! verify row: one io_uring round trip, one producer-stream drain and one
//! synchronous pageable upload per row, so the GPU idles while the host reads
//! the next row's records. This path selects every row's 16 records on the
//! host first (row r hashes against `prior ++ tokens[..r]`, as the loop does),
//! reads all of them in one reader call, packs them into one page-locked
//! staging buffer, uploads it with a single stream-ordered copy, dequantizes
//! every row, runs both projections once for all rows (`dense_gemv_bf16_batchn_tf`,
//! bit-identical per row to `dense_gemv_bf16`, token-fastest grid so the 52 MB
//! key projection streams once), then fuses rows in order and saves each row's
//! conv history exactly like the loop.
//! Same kernels, same inputs, same order: every row's PLE contribution and
//! conv history are byte-identical to the per-row path.

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, PinnedHostBuffer};
use spark_storage::ple_offload::{ParkedPleCache, PleNvfp4Row, PleOffloadReader};

use super::{PLE_EMBED, PLE_MAX_SPEC_ROWS, PLE_RECORD_BYTES, Qwen4PleLayer};
use crate::layers::ops;
use crate::layers::qwen4_ple::{QWEN4_PLE_HEADS, Qwen4PleHasher};

const ROW_RECORD_BYTES: usize = QWEN4_PLE_HEADS * PLE_RECORD_BYTES;
const ROW_SCALE_BYTES: usize = QWEN4_PLE_HEADS * 4;
const RAGGED_DESC_BYTES: usize = std::mem::size_of::<u32>();
const RAGGED_DESC_MAX_BYTES: usize = PLE_MAX_BATCH_ROWS * RAGGED_DESC_BYTES;
/// Maximum PLE `(shard, row)` misses read per `reader.try_lock()` hold during
/// speculative background `DraftAhead` warming (`PLE_MAX_BATCH_ROWS` = `256`
/// tokens × `16` heads = `4096` selections). Coalesces a full cross-sequence
/// draft-ahead batch into one 4-ring NVMe pass so an arriving foreground
/// `Verify` never splits the batch into two serial NVMe reads, while checking
/// `verify_pending` per token and immediately before/after `reader.try_lock()`
/// so `DraftAhead` yields before issuing I/O whenever `Verify` arrives first.
const DRAFT_AHEAD_CHUNK_SELECTIONS: usize = PLE_MAX_BATCH_ROWS * QWEN4_PLE_HEADS;
/// Upper bound on waiting for the previous verify's staging upload. The
/// scheduler drains the stream for every verify's logits, so the event is
/// already complete in practice; a stuck copy must fail, not hang.
const STAGING_WAIT: Duration = Duration::from_secs(5);

const RAGGED_PLE_DESC_MAX_ROW_OFFSET: usize = 0xFFF;
const RAGGED_PLE_DESC_MAX_ROWS: usize = 0xF;

const _: () = {
    assert!(PLE_MAX_SPEC_ROWS <= PLE_MAX_BATCH_ROWS);
    assert!(PLE_MAX_BATCH_ROWS <= RAGGED_PLE_DESC_MAX_ROW_OFFSET + 1);
};

/// One real request row in the native Qwen4 concurrent-decode PLE prefix.
///
/// The batch contains active rows only.  Padding is deliberately absent from
/// this type: padding rows have no token history and must never advance a PLE
/// convolution state.  `hyper` remains a contiguous active-row tile, while
/// `slot_idx` selects the request-owned persistent state for each row.
#[derive(Clone, Copy, Debug)]
pub struct PleActiveDecodeRow<'a> {
    pub current_token: u32,
    pub prior_tokens: &'a [u32],
    pub slot_idx: usize,
    pub reset_state: bool,
}

/// Compact stack-only PLE warm-up descriptor for one sequence in a
/// cross-sequence verify frame (`48 B`).
///
/// The trigram PLE hasher only inspects at most the last two prior tokens, so
/// storing `prior_tail` avoids cloning full 32K–128K sequence histories on
/// every speculative verify step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PleWarmSeq {
    pub draft_tokens: [u32; 9],
    pub draft_len: u8,
    pub prior_tail: [u32; 2],
    pub prior_len: u8,
}

impl PleWarmSeq {
    /// Maximum draft tokens stored inline (`QWEN4_CROSS_SEQ_MAX_K`).
    pub const MAX_DRAFT_LEN: usize = 9;
    /// Maximum prior-history tokens inspected by the trigram PLE hasher.
    pub const MAX_PRIOR_TAIL: usize = 2;

    /// Pack a sequence's draft window and at most the last two prior tokens.
    pub fn from_slices(draft: &[u32], prior: &[u32]) -> Result<Self> {
        ensure!(
            draft.len() <= Self::MAX_DRAFT_LEN,
            "PleWarmSeq draft length {} exceeds {}",
            draft.len(),
            Self::MAX_DRAFT_LEN
        );
        let mut draft_tokens = [0u32; Self::MAX_DRAFT_LEN];
        draft_tokens[..draft.len()].copy_from_slice(draft);
        let prior_slice = &prior[prior.len().saturating_sub(Self::MAX_PRIOR_TAIL)..];
        let mut prior_tail = [0u32; Self::MAX_PRIOR_TAIL];
        prior_tail[..prior_slice.len()].copy_from_slice(prior_slice);
        Ok(Self {
            draft_tokens,
            draft_len: draft.len() as u8,
            prior_tail,
            prior_len: prior_slice.len() as u8,
        })
    }

    /// Active draft token window for this sequence.
    pub fn draft(&self) -> &[u32] {
        let len = usize::from(self.draft_len).min(Self::MAX_DRAFT_LEN);
        &self.draft_tokens[..len]
    }

    /// Prior context tail (`0..=2` tokens) preceding the draft window.
    pub fn prior(&self) -> &[u32] {
        let len = usize::from(self.prior_len).min(Self::MAX_PRIOR_TAIL);
        &self.prior_tail[..len]
    }

    /// Pack per-sequence `(draft, prior)` slices into a validated cross-sequence
    /// verify prefetch batch (`1..=PLE_MAX_BATCH_ROWS` total rows, each sequence
    /// `1..=Self::MAX_DRAFT_LEN` rows). Returns `None` if empty or out of bounds.
    #[must_use]
    pub fn pack_batch(seqs: &[(&[u32], &[u32])]) -> Option<Vec<Self>> {
        if seqs.is_empty() {
            return None;
        }
        let mut total_rows = 0usize;
        let mut out = Vec::with_capacity(seqs.len());
        for &(draft, prior) in seqs {
            if draft.is_empty() || draft.len() > Self::MAX_DRAFT_LEN {
                return None;
            }
            total_rows = total_rows.checked_add(draft.len())?;
            if total_rows > PLE_MAX_BATCH_ROWS {
                return None;
            }
            out.push(Self::from_slices(draft, prior).ok()?);
        }
        Some(out)
    }
}

/// Stack-only 2-token rolling context matching `Qwen4PleHasher::select_decode`'s
/// exact trigram lookback without per-sequence heap allocations.
struct RollingWindow {
    buf: [u32; 2],
    len: usize,
}

impl RollingWindow {
    #[inline]
    fn from_prior(prior: &[u32]) -> Self {
        let tail = &prior[prior.len().saturating_sub(2)..];
        let mut buf = [0u32; 2];
        buf[..tail.len()].copy_from_slice(tail);
        Self {
            buf,
            len: tail.len(),
        }
    }

    #[inline]
    fn as_slice(&self) -> &[u32] {
        &self.buf[..self.len]
    }

    #[inline]
    fn push(&mut self, token: u32) {
        if self.len < 2 {
            self.buf[self.len] = token;
            self.len += 1;
        } else {
            self.buf[0] = self.buf[1];
            self.buf[1] = token;
        }
    }
}

/// Fixed-capacity stack buffer of `(shard, row)` PLE selections that derefs to
/// `&[(usize, usize)]` so `c=1` verify paths avoid per-step heap allocations.
struct StackSelections<const N: usize> {
    buf: [(usize, usize); N],
    len: usize,
}

impl<const N: usize> StackSelections<N> {
    #[inline]
    fn new() -> Self {
        Self {
            buf: [(0, 0); N],
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, selection: (usize, usize)) {
        debug_assert!(self.len < N, "StackSelections capacity {N} exceeded");
        self.buf[self.len] = selection;
        self.len += 1;
    }
}

impl<const N: usize> Deref for StackSelections<N> {
    type Target = [(usize, usize)];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.buf[..self.len]
    }
}

#[inline]
fn pack_records_to_staging(staging: &mut [u8], records: &[PleNvfp4Row], scale_base: usize) {
    for (index, row) in records.iter().enumerate() {
        let start = index * PLE_RECORD_BYTES;
        staging[start..start + PLE_RECORD_BYTES].copy_from_slice(&row.record);
        let scale = scale_base + index * 4;
        staging[scale..scale + 4].copy_from_slice(&row.scale2.to_le_bytes());
    }
}

fn build_ple_warm_request(hasher: &Qwen4PleHasher, seqs: &[PleWarmSeq]) -> Vec<(usize, usize)> {
    let total_rows: usize = seqs.iter().map(|s| s.draft().len()).sum();
    if total_rows == 0 || total_rows > PLE_MAX_BATCH_ROWS {
        return Vec::new();
    }
    let mut request = Vec::with_capacity(total_rows * QWEN4_PLE_HEADS);
    for seq in seqs {
        let mut context = RollingWindow::from_prior(seq.prior());
        for &token in seq.draft() {
            let selections = hasher.select_decode(token, context.as_slice());
            request.extend(selections.iter().map(|s| (s.shard, s.row)));
            context.push(token);
        }
    }
    request
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PleDraftAheadSpan {
    start: u16,
    len: u16,
    prior_tail: [u32; 2],
    prior_len: u8,
}

/// Compact token-only draft-ahead payload handed to [`PleWarmWorker`] so
/// `submit_draft_ahead_seqs` performs zero hashing and zero `ParkedPleCache`
/// probes on the scheduler critical path before layer 0.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PleDraftAheadBatch {
    tokens: Vec<u32>,
    spans: Vec<PleDraftAheadSpan>,
}

impl PleDraftAheadBatch {
    fn from_seqs(seqs: &[(&[u32], &[u32])], max_rows: usize) -> Self {
        let cap = max_rows.min(u16::MAX as usize);
        let mut remaining = cap;
        let mut total_tokens = 0usize;
        let mut span_count = 0usize;
        for &(tokens, _) in seqs {
            if remaining == 0 {
                break;
            }
            let take = tokens.len().min(remaining);
            if take > 0 {
                total_tokens += take;
                span_count += 1;
                remaining -= take;
            }
        }
        if total_tokens == 0 {
            return Self {
                tokens: Vec::new(),
                spans: Vec::new(),
            };
        }
        let mut flat_tokens = Vec::with_capacity(total_tokens);
        let mut spans = Vec::with_capacity(span_count);
        let mut rem = cap;
        for &(tokens, prior_tokens) in seqs {
            if rem == 0 {
                break;
            }
            let take = tokens.len().min(rem);
            if take == 0 {
                continue;
            }
            let start = flat_tokens.len() as u16;
            flat_tokens.extend_from_slice(&tokens[..take]);
            let tail = &prior_tokens[prior_tokens.len().saturating_sub(2)..];
            let mut prior_tail = [0u32; 2];
            prior_tail[..tail.len()].copy_from_slice(tail);
            spans.push(PleDraftAheadSpan {
                start,
                len: take as u16,
                prior_tail,
                prior_len: tail.len() as u8,
            });
            rem -= take;
        }
        Self {
            tokens: flat_tokens,
            spans,
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// High-priority foreground verify job executed on [`PleWarmWorker`].
/// Every variant always produces exactly one response on `resp_tx`.
enum PleVerifyJob {
    Seqs(Vec<PleWarmSeq>),
    #[cfg(test)]
    Request(Vec<(usize, usize)>),
}

/// Low-priority speculative draft-ahead job executed on [`PleWarmWorker`]
/// when idle. Never sends on `resp_tx` and yields immediately when a
/// foreground [`PleVerifyJob`] arrives.
enum PleDraftAheadJob {
    Batch(PleDraftAheadBatch),
    #[cfg(test)]
    Selections(Vec<(usize, usize)>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DraftSendOutcome {
    Sent,
    Busy,
    Disconnected,
}

fn read_verify_with_parked_cache(
    parked_cache: &ParkedPleCache,
    reader: &parking_lot::Mutex<PleOffloadReader>,
    request: &[(usize, usize)],
) -> Result<Vec<PleNvfp4Row>> {
    if let Some(cached) = parked_cache.try_read_all(request)? {
        return Ok(cached);
    }
    reader.lock().read_rows_verify(request)
}

/// Fast path for [`PleWarmWorker`] foreground verify: first tries lock-free
/// [`ParkedPleCache::try_read_all`]; if any key collided in the direct-mapped
/// parked table after a coalesced `DraftAhead` read, satisfies the batch from
/// `recent_draft ∪ parked_cache` with zero additional NVMe I/O.
fn try_read_from_parked_or_recent(
    parked_cache: &ParkedPleCache,
    recent_draft: &mut HashMap<(usize, usize), PleNvfp4Row>,
    request: &[(usize, usize)],
) -> Result<Option<Vec<PleNvfp4Row>>> {
    if let Some(cached) = parked_cache.try_read_all(request)? {
        recent_draft.clear();
        return Ok(Some(cached));
    }
    if recent_draft.is_empty() {
        return Ok(None);
    }
    // Probe first so genuine partial-hit batches avoid allocating or copying.
    for &(shard, row) in request {
        if !recent_draft.contains_key(&(shard, row)) && !parked_cache.contains(shard, row) {
            recent_draft.clear();
            return Ok(None);
        }
    }
    let mut out = Vec::with_capacity(request.len());
    for &(shard, row) in request {
        if let Some(hit) = recent_draft.get(&(shard, row)) {
            out.push(hit.clone());
        } else if let Some(hit) = parked_cache.get_row(shard, row)? {
            out.push(hit);
        } else {
            recent_draft.clear();
            return Ok(None);
        }
    }
    recent_draft.clear();
    Ok(Some(out))
}

fn read_verify_with_parked_or_recent(
    parked_cache: &ParkedPleCache,
    recent_draft: &parking_lot::Mutex<HashMap<(usize, usize), PleNvfp4Row>>,
    reader: &parking_lot::Mutex<PleOffloadReader>,
    request: &[(usize, usize)],
) -> Result<Vec<PleNvfp4Row>> {
    let hit = {
        let mut recent = recent_draft.lock();
        try_read_from_parked_or_recent(parked_cache, &mut recent, request)?
    };
    match hit {
        Some(rows) => Ok(rows),
        None => reader.lock().read_rows_verify(request),
    }
}

fn flush_draft_ahead_chunk(
    reader: &parking_lot::Mutex<PleOffloadReader>,
    io_poisoned: &AtomicBool,
    verify_pending: &AtomicBool,
    chunk: &mut Vec<(usize, usize)>,
    recent_draft: &mut HashMap<(usize, usize), PleNvfp4Row>,
) -> bool {
    if chunk.is_empty() {
        return true;
    }
    if verify_pending.load(Ordering::Acquire) || io_poisoned.load(Ordering::Acquire) {
        chunk.clear();
        return false;
    }
    // Non-blocking try_lock: speculative background warm-up yields immediately
    // if foreground prefill or verify owns the storage reader.
    let Some(mut guard) = reader.try_lock() else {
        chunk.clear();
        return false;
    };
    // Re-check after acquiring `reader` so a Verify arriving concurrently
    // never stalls behind a fresh DraftAhead NVMe batch.
    if verify_pending.load(Ordering::Acquire) || io_poisoned.load(Ordering::Acquire) {
        drop(guard);
        chunk.clear();
        return false;
    }
    let read_res = guard.read_rows_verify(chunk);
    drop(guard);
    if let Ok(rows) = read_res {
        if recent_draft.len().saturating_add(rows.len()) > 2 * DRAFT_AHEAD_CHUNK_SELECTIONS {
            recent_draft.clear();
        }
        recent_draft.reserve(rows.len());
        for (&key, row) in chunk.iter().zip(rows) {
            recent_draft.insert(key, row);
        }
    }
    chunk.clear();
    true
}

fn warm_draft_ahead_batch_into_cache(
    hasher: &Qwen4PleHasher,
    parked_cache: &ParkedPleCache,
    reader: &parking_lot::Mutex<PleOffloadReader>,
    io_poisoned: &AtomicBool,
    verify_pending: &AtomicBool,
    batch: &PleDraftAheadBatch,
    recent_draft: &mut HashMap<(usize, usize), PleNvfp4Row>,
) {
    if !parked_cache.is_enabled()
        || batch.is_empty()
        || io_poisoned.load(Ordering::Acquire)
        || verify_pending.load(Ordering::Acquire)
    {
        return;
    }
    let mut chunk = Vec::with_capacity(
        batch
            .tokens
            .len()
            .saturating_mul(QWEN4_PLE_HEADS)
            .min(DRAFT_AHEAD_CHUNK_SELECTIONS),
    );
    for span in &batch.spans {
        if verify_pending.load(Ordering::Acquire) || io_poisoned.load(Ordering::Acquire) {
            return;
        }
        let start = usize::from(span.start);
        let end = start + usize::from(span.len);
        let prior_len = usize::from(span.prior_len).min(2);
        let mut context = RollingWindow::from_prior(&span.prior_tail[..prior_len]);
        for &token in &batch.tokens[start..end] {
            if verify_pending.load(Ordering::Acquire) || io_poisoned.load(Ordering::Acquire) {
                return;
            }
            let selections = hasher.select_decode(token, context.as_slice());
            for sel in selections {
                let key = (sel.shard, sel.row);
                if !parked_cache.contains(sel.shard, sel.row) && !recent_draft.contains_key(&key) {
                    chunk.push(key);
                    if chunk.len() == DRAFT_AHEAD_CHUNK_SELECTIONS
                        && !flush_draft_ahead_chunk(
                            reader,
                            io_poisoned,
                            verify_pending,
                            &mut chunk,
                            recent_draft,
                        )
                    {
                        return;
                    }
                }
            }
            context.push(token);
        }
    }
    let _ = flush_draft_ahead_chunk(
        reader,
        io_poisoned,
        verify_pending,
        &mut chunk,
        recent_draft,
    );
}

#[cfg(test)]
fn warm_draft_ahead_into_cache(
    parked_cache: &ParkedPleCache,
    reader: &parking_lot::Mutex<PleOffloadReader>,
    io_poisoned: &AtomicBool,
    verify_pending: &AtomicBool,
    selections: &[(usize, usize)],
    recent_draft: &mut HashMap<(usize, usize), PleNvfp4Row>,
) {
    if !parked_cache.is_enabled()
        || selections.is_empty()
        || io_poisoned.load(Ordering::Acquire)
        || verify_pending.load(Ordering::Acquire)
    {
        return;
    }
    let mut chunk = Vec::with_capacity(selections.len().min(DRAFT_AHEAD_CHUNK_SELECTIONS));
    for &(shard, row) in selections {
        if verify_pending.load(Ordering::Acquire) || io_poisoned.load(Ordering::Acquire) {
            return;
        }
        let key = (shard, row);
        if !parked_cache.contains(shard, row) && !recent_draft.contains_key(&key) {
            chunk.push(key);
            if chunk.len() == DRAFT_AHEAD_CHUNK_SELECTIONS
                && !flush_draft_ahead_chunk(
                    reader,
                    io_poisoned,
                    verify_pending,
                    &mut chunk,
                    recent_draft,
                )
            {
                return;
            }
        }
    }
    let _ = flush_draft_ahead_chunk(
        reader,
        io_poisoned,
        verify_pending,
        &mut chunk,
        recent_draft,
    );
}

#[cfg(test)]
fn collect_draft_ahead_misses(
    hasher: &Qwen4PleHasher,
    parked_cache: &ParkedPleCache,
    seqs: &[(&[u32], &[u32])],
    max_rows: usize,
) -> Vec<(usize, usize)> {
    let batch = PleDraftAheadBatch::from_seqs(seqs, max_rows);
    let mut misses = Vec::new();
    for span in &batch.spans {
        let start = usize::from(span.start);
        let end = start + usize::from(span.len);
        let prior_len = usize::from(span.prior_len).min(2);
        let mut context = RollingWindow::from_prior(&span.prior_tail[..prior_len]);
        for &token in &batch.tokens[start..end] {
            let selections = hasher.select_decode(token, context.as_slice());
            for sel in selections {
                if !parked_cache.contains(sel.shard, sel.row) {
                    misses.push((sel.shard, sel.row));
                }
            }
            context.push(token);
        }
    }
    misses
}

struct WarmQueueState {
    verify: Option<PleVerifyJob>,
    draft_ahead: Option<PleDraftAheadJob>,
    closed: bool,
}

struct WarmQueueShared {
    state: parking_lot::Mutex<WarmQueueState>,
    cvar: parking_lot::Condvar,
    /// Set `true` as soon as a foreground verify is submitted (or worker
    /// shutdown starts) so any in-flight `DraftAhead` yields immediately.
    verify_pending: AtomicBool,
}

struct WorkerExitGuard {
    shared: Arc<WarmQueueShared>,
}

impl Drop for WorkerExitGuard {
    fn drop(&mut self) {
        let mut state = self.shared.state.lock();
        state.closed = true;
        self.shared.verify_pending.store(false, Ordering::Release);
        self.shared.cvar.notify_all();
    }
}

/// Persistent background worker that overlaps PLE hashing, [`ParkedPleCache`]
/// lookups, and `read_rows_verify` I/O with embedding + layer 0 (and services
/// preemptible `DRAFT_AHEAD` warm-ups into [`ParkedPleCache`] when idle)
/// without spawning an OS thread per step or blocking before layer 0.
pub(super) struct PleWarmWorker {
    shared: Arc<WarmQueueShared>,
    resp_rx: mpsc::Receiver<Result<Vec<PleNvfp4Row>>>,
    /// Key of a pre-submitted reusable verify prefetch whose single unread
    /// response is in flight or buffered in `resp_rx`.
    pending_prefetch: Option<Vec<PleWarmSeq>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl PleWarmWorker {
    fn spawn(
        hasher: Arc<Qwen4PleHasher>,
        reader: Arc<parking_lot::Mutex<PleOffloadReader>>,
        parked_cache: Arc<ParkedPleCache>,
        io_poisoned: Arc<AtomicBool>,
    ) -> Result<Self> {
        let verify_hasher = Arc::clone(&hasher);
        let verify_cache = Arc::clone(&parked_cache);
        let verify_reader = Arc::clone(&reader);
        // Uncontended worker-private retention of the most recent DraftAhead
        // read so direct-mapped ParkedPleCache collisions inside a 4096-selection
        // batch never trigger a second NVMe read on the immediately following Verify.
        let recent_draft = Arc::new(parking_lot::Mutex::new(HashMap::<
            (usize, usize),
            PleNvfp4Row,
        >::new()));
        let verify_recent = Arc::clone(&recent_draft);
        Self::spawn_jobs(
            "ple-warm-worker",
            move |job| match job {
                PleVerifyJob::Seqs(seqs) => {
                    let request = build_ple_warm_request(&verify_hasher, &seqs);
                    if request.is_empty() {
                        verify_recent.lock().clear();
                        return Ok(Vec::new());
                    }
                    read_verify_with_parked_or_recent(
                        &verify_cache,
                        &verify_recent,
                        &verify_reader,
                        &request,
                    )
                }
                #[cfg(test)]
                PleVerifyJob::Request(request) => {
                    if request.is_empty() {
                        verify_recent.lock().clear();
                        return Ok(Vec::new());
                    }
                    read_verify_with_parked_or_recent(
                        &verify_cache,
                        &verify_recent,
                        &verify_reader,
                        &request,
                    )
                }
            },
            move |job, verify_pending| {
                let mut recent = recent_draft.lock();
                match job {
                    PleDraftAheadJob::Batch(batch) => warm_draft_ahead_batch_into_cache(
                        &hasher,
                        &parked_cache,
                        &reader,
                        &io_poisoned,
                        verify_pending,
                        &batch,
                        &mut recent,
                    ),
                    #[cfg(test)]
                    PleDraftAheadJob::Selections(selections) => warm_draft_ahead_into_cache(
                        &parked_cache,
                        &reader,
                        &io_poisoned,
                        verify_pending,
                        &selections,
                        &mut recent,
                    ),
                }
            },
        )
    }

    #[cfg(test)]
    fn spawn_with<F>(name: &str, mut run: F) -> Result<Self>
    where
        F: FnMut(Vec<PleWarmSeq>) -> Result<Vec<PleNvfp4Row>> + Send + 'static,
    {
        Self::spawn_jobs(
            name,
            move |job| match job {
                PleVerifyJob::Seqs(seqs) => run(seqs),
                PleVerifyJob::Request(_) => Err(anyhow::anyhow!(
                    "unexpected VerifyRequest in PleWarmWorker::spawn_with"
                )),
            },
            |_draft, _verify_pending| {},
        )
    }

    fn spawn_jobs<V, D>(name: &str, mut run_verify: V, mut run_draft: D) -> Result<Self>
    where
        V: FnMut(PleVerifyJob) -> Result<Vec<PleNvfp4Row>> + Send + 'static,
        D: FnMut(PleDraftAheadJob, &AtomicBool) + Send + 'static,
    {
        let shared = Arc::new(WarmQueueShared {
            state: parking_lot::Mutex::new(WarmQueueState {
                verify: None,
                draft_ahead: None,
                closed: false,
            }),
            cvar: parking_lot::Condvar::new(),
            verify_pending: AtomicBool::new(false),
        });
        let worker_shared = Arc::clone(&shared);
        let (resp_tx, resp_rx) = mpsc::sync_channel::<Result<Vec<PleNvfp4Row>>>(1);
        let thread = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                // Pin persistent PLE warm worker once on startup according to
                // `ATLAS_PLE_READ_AFFINITY` so verify hashing and cache/NVMe
                // orchestration stay on high-capacity cores.
                let _ = super::read_affinity::begin();
                let _exit_guard = WorkerExitGuard {
                    shared: Arc::clone(&worker_shared),
                };
                loop {
                    enum NextWork {
                        Verify(PleVerifyJob),
                        Draft(PleDraftAheadJob),
                        Exit,
                    }
                    let next = {
                        let mut state = worker_shared.state.lock();
                        loop {
                            if let Some(verify_job) = state.verify.take() {
                                break NextWork::Verify(verify_job);
                            }
                            if state.closed {
                                break NextWork::Exit;
                            }
                            if !worker_shared.verify_pending.load(Ordering::Acquire)
                                && let Some(draft_job) = state.draft_ahead.take()
                            {
                                break NextWork::Draft(draft_job);
                            }
                            worker_shared.cvar.wait(&mut state);
                        }
                    };
                    match next {
                        NextWork::Verify(job) => {
                            let result = run_verify(job);
                            // Clear `verify_pending` before delivering the response:
                            // `PleWarmTicket` holds `self.warm_worker.lock()` until
                            // `resp_rx.recv()` returns, so no next `Verify` can race.
                            worker_shared.verify_pending.store(false, Ordering::Release);
                            if resp_tx.send(result).is_err() {
                                break;
                            }
                        }
                        NextWork::Draft(job) => {
                            run_draft(job, &worker_shared.verify_pending);
                        }
                        NextWork::Exit => break,
                    }
                }
            })
            .context("spawn persistent Qwen4 PLE warm-up worker")?;
        Ok(Self {
            shared,
            resp_rx,
            pending_prefetch: None,
            thread: Some(thread),
        })
    }

    fn send_verify(
        &self,
        job: PleVerifyJob,
    ) -> std::result::Result<(), (PleVerifyJob, anyhow::Error)> {
        // Signal preemption before locking `state` so any in-flight `DraftAhead`
        // yields at its next check without waiting for channel/queue contention.
        self.shared.verify_pending.store(true, Ordering::Release);
        let mut state = self.shared.state.lock();
        if state.closed {
            self.shared.verify_pending.store(false, Ordering::Release);
            return Err((
                job,
                anyhow::anyhow!("PLE warm-up worker terminated unexpectedly"),
            ));
        }
        debug_assert!(
            state.verify.is_none(),
            "concurrent foreground PLE verify submitted while prior ticket is active"
        );
        // Supersede any queued speculative draft-ahead so foreground verify
        // executes immediately without redundant pre-warm work ahead of it.
        state.draft_ahead = None;
        state.verify = Some(job);
        drop(state);
        self.shared.cvar.notify_one();
        Ok(())
    }

    fn try_send_draft_ahead_job(&self, job: PleDraftAheadJob) -> DraftSendOutcome {
        if self.pending_prefetch.is_some() || self.shared.verify_pending.load(Ordering::Acquire) {
            return DraftSendOutcome::Busy;
        }
        let mut state = self.shared.state.lock();
        if state.closed {
            return DraftSendOutcome::Disconnected;
        }
        if state.verify.is_some() {
            return DraftSendOutcome::Busy;
        }
        state.draft_ahead = Some(job);
        drop(state);
        self.shared.cvar.notify_one();
        DraftSendOutcome::Sent
    }

    /// Non-blocking submission of a keyed reusable `Verify` prefetch that the
    /// next [`submit_warm_verify_with`] call adopts directly on exact key match
    /// (`prefetched == req_seqs`) without preemption or re-reading storage.
    fn try_submit_verify_prefetch(&mut self, seqs: Vec<PleWarmSeq>) -> DraftSendOutcome {
        if let Some(existing) = self.pending_prefetch.as_ref() {
            if *existing == seqs {
                return DraftSendOutcome::Sent;
            }
            match self.resp_rx.try_recv() {
                Ok(_) => {
                    self.pending_prefetch = None;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.pending_prefetch = None;
                    return DraftSendOutcome::Disconnected;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    let mut state = self.shared.state.lock();
                    if state.closed {
                        return DraftSendOutcome::Disconnected;
                    }
                    if state.verify.is_some() {
                        state.draft_ahead = None;
                        state.verify = Some(PleVerifyJob::Seqs(seqs.clone()));
                        self.shared.verify_pending.store(true, Ordering::Release);
                        drop(state);
                        self.pending_prefetch = Some(seqs);
                        self.shared.cvar.notify_one();
                        return DraftSendOutcome::Sent;
                    }
                    // Previous prefetch is actively reading storage; do not block
                    // the scheduler thread. Foreground verify drains it cleanly.
                    return DraftSendOutcome::Busy;
                }
            }
        }
        match self.send_verify(PleVerifyJob::Seqs(seqs.clone())) {
            Ok(()) => {
                self.pending_prefetch = Some(seqs);
                DraftSendOutcome::Sent
            }
            Err(_) => DraftSendOutcome::Disconnected,
        }
    }

    fn try_send_draft_ahead_batch(&self, batch: PleDraftAheadBatch) -> DraftSendOutcome {
        self.try_send_draft_ahead_job(PleDraftAheadJob::Batch(batch))
    }

    #[cfg(test)]
    fn try_send_draft_ahead(&self, selections: Vec<(usize, usize)>) -> DraftSendOutcome {
        self.try_send_draft_ahead_job(PleDraftAheadJob::Selections(selections))
    }

    fn recv_channel(&self) -> std::result::Result<Result<Vec<PleNvfp4Row>>, mpsc::RecvError> {
        self.resp_rx.recv()
    }
}

impl Drop for PleWarmWorker {
    fn drop(&mut self) {
        // Preempt any running DraftAhead, drop any unstarted queued Verify/prefetch,
        // and signal shutdown before joining.
        self.shared.verify_pending.store(true, Ordering::Release);
        {
            let mut state = self.shared.state.lock();
            state.closed = true;
            state.draft_ahead = None;
            state.verify = None;
        }
        self.pending_prefetch = None;
        self.shared.cvar.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

enum PleWarmTicketInner<'a> {
    /// Immediately satisfied (empty/out-of-bounds batch) with zero worker wakeups.
    Immediate(Option<Vec<PleNvfp4Row>>),
    /// Dispatched to the persistent background [`PleWarmWorker`].
    Worker {
        guard: parking_lot::MutexGuard<'a, Option<PleWarmWorker>>,
        consumed: bool,
    },
}

/// RAII handle for an in-flight persistent [`PleWarmWorker`] request (or an
/// immediately satisfied empty batch).
///
/// For worker-backed tickets, holds the worker lock for the duration of the
/// outstanding request so at most one verify warm-up batch is in flight. If
/// dropped before [`Self::wait`] is called (for example on an early `?` error
/// before layer 1), [`Drop`] drains the pending response so the worker never
/// desynchronizes on the next step.
#[must_use = "call wait() to consume the pre-read PLE rows"]
pub struct PleWarmTicket<'a> {
    inner: PleWarmTicketInner<'a>,
}

impl PleWarmTicket<'_> {
    fn immediate(rows: Vec<PleNvfp4Row>) -> Self {
        Self {
            inner: PleWarmTicketInner::Immediate(Some(rows)),
        }
    }

    /// Wait for the persistent worker to finish reading the submitted PLE
    /// rows and return them.
    pub fn wait(mut self) -> Result<Vec<PleNvfp4Row>> {
        match &mut self.inner {
            PleWarmTicketInner::Immediate(rows) => Ok(rows.take().unwrap_or_default()),
            PleWarmTicketInner::Worker { guard, consumed } => {
                *consumed = true;
                let recv = match guard.as_ref() {
                    Some(worker) => worker.recv_channel(),
                    None => bail!("PLE warm-up worker was cleared while ticket was active"),
                };
                match recv {
                    Ok(result) => result,
                    Err(_) => {
                        guard.take();
                        bail!("PLE warm-up worker exited before sending response");
                    }
                }
            }
        }
    }
}

impl Drop for PleWarmTicket<'_> {
    fn drop(&mut self) {
        if let PleWarmTicketInner::Worker { guard, consumed } = &mut self.inner
            && !*consumed
        {
            *consumed = true;
            let disconnected = guard
                .as_ref()
                .is_some_and(|worker| worker.recv_channel().is_err());
            if disconnected {
                guard.take();
            }
        }
    }
}

fn submit_warm_verify_with<'a, F>(
    slot: &'a parking_lot::Mutex<Option<PleWarmWorker>>,
    job: PleVerifyJob,
    mut spawn_fn: F,
) -> Result<PleWarmTicket<'a>>
where
    F: FnMut() -> Result<PleWarmWorker>,
{
    let mut guard = slot.lock();
    let mut pending = Some(job);
    let prefetched = guard
        .as_mut()
        .and_then(|worker| worker.pending_prefetch.take());
    if let Some(prefetched) = prefetched {
        let exact_match = matches!(
            pending.as_ref(),
            Some(PleVerifyJob::Seqs(req_seqs)) if *req_seqs == prefetched
        );
        if exact_match {
            static REUSED_RECEIPT: std::sync::Once = std::sync::Once::new();
            REUSED_RECEIPT.call_once(|| {
                let rows: usize = prefetched.iter().map(|s| s.draft().len()).sum();
                tracing::info!(
                    seqs = prefetched.len(),
                    rows,
                    "QWEN4_PLE_VERIFY_PREFETCH_REUSED"
                );
            });
            return Ok(PleWarmTicket {
                inner: PleWarmTicketInner::Worker {
                    guard,
                    consumed: false,
                },
            });
        }
        // Key mismatch: replace in-place if the stale prefetch has not started
        // yet; otherwise drain its single response so `resp_rx` stays in lockstep.
        let replaced_in_place = if let Some(worker) = guard.as_ref() {
            let mut state = worker.shared.state.lock();
            if !state.closed && state.verify.is_some() {
                state.draft_ahead = None;
                state.verify = pending.take();
                worker.shared.verify_pending.store(true, Ordering::Release);
                drop(state);
                worker.shared.cvar.notify_one();
                true
            } else {
                false
            }
        } else {
            false
        };
        if replaced_in_place {
            return Ok(PleWarmTicket {
                inner: PleWarmTicketInner::Worker {
                    guard,
                    consumed: false,
                },
            });
        }
        let disconnected = guard
            .as_ref()
            .is_some_and(|worker| worker.recv_channel().is_err());
        if disconnected {
            guard.take();
        }
    }
    for attempt in 0..2 {
        if guard.is_none() {
            *guard = Some(spawn_fn()?);
        }
        let payload = pending.take().expect("payload present on send attempt");
        let send_res = guard
            .as_ref()
            .expect("worker initialized")
            .send_verify(payload);
        match send_res {
            Ok(()) => {
                return Ok(PleWarmTicket {
                    inner: PleWarmTicketInner::Worker {
                        guard,
                        consumed: false,
                    },
                });
            }
            Err((returned, _err)) if attempt == 0 => {
                guard.take();
                pending = Some(returned);
            }
            Err((_, err)) => {
                guard.take();
                return Err(err);
            }
        }
    }
    unreachable!("submit loop returns on attempt 0 or 1")
}

#[cfg(test)]
fn submit_warm_with<'a, F>(
    slot: &'a parking_lot::Mutex<Option<PleWarmWorker>>,
    seqs: Vec<PleWarmSeq>,
    spawn_fn: F,
) -> Result<PleWarmTicket<'a>>
where
    F: FnMut() -> Result<PleWarmWorker>,
{
    submit_warm_verify_with(slot, PleVerifyJob::Seqs(seqs), spawn_fn)
}

#[cfg(test)]
fn submit_verify_prefetch_with<F>(
    slot: &parking_lot::Mutex<Option<PleWarmWorker>>,
    seqs: Vec<PleWarmSeq>,
    mut spawn_fn: F,
) -> DraftSendOutcome
where
    F: FnMut() -> Result<PleWarmWorker>,
{
    let total_rows: usize = seqs.iter().map(|s| s.draft().len()).sum();
    if total_rows == 0 || total_rows > PLE_MAX_BATCH_ROWS {
        return DraftSendOutcome::Busy;
    }
    let Some(mut guard) = slot.try_lock() else {
        return DraftSendOutcome::Busy;
    };
    if guard.is_none() {
        match spawn_fn() {
            Ok(worker) => *guard = Some(worker),
            Err(_) => return DraftSendOutcome::Disconnected,
        }
    }
    let outcome = guard
        .as_mut()
        .expect("worker initialized")
        .try_submit_verify_prefetch(seqs);
    if outcome == DraftSendOutcome::Disconnected {
        guard.take();
    }
    outcome
}

const PLE_MAX_BATCH_ROWS: usize = crate::layers::QWEN4_NATIVE_MAX_ROWS;

fn validate_active_decode_layout(
    rows: usize,
    row_stride: usize,
    residual: usize,
    slots: &[usize],
    max_slots: usize,
) -> Result<()> {
    ensure!(
        (2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&rows),
        "Qwen4 native concurrent PLE currently admits C2-C{}, got C{rows}",
        crate::layers::QWEN4_NATIVE_MAX_ROWS
    );
    ensure!(
        rows <= PLE_MAX_BATCH_ROWS,
        "Qwen4 native concurrent PLE tile exceeds {PLE_MAX_BATCH_ROWS} rows"
    );
    ensure!(
        row_stride == residual.checked_mul(2).context("PLE row stride overflow")?,
        "Qwen4 native concurrent PLE requires contiguous BF16 hyper rows"
    );
    ensure!(slots.len() == rows, "PLE active slot extent mismatch");
    for (row, &slot) in slots.iter().enumerate() {
        ensure!(
            slot < max_slots,
            "PLE active row {row} slot {slot} out of range"
        );
        ensure!(
            !slots[..row].contains(&slot),
            "PLE active rows {row} and {} alias persistent slot {slot}",
            slots[..row]
                .iter()
                .position(|&prior| prior == slot)
                .expect("duplicate was found above")
        );
    }
    Ok(())
}

/// `ATLAS_QWEN4_PLE_VERIFY_BATCH` (default on unless `ATLAS_QWEN4_PLE_VERIFY_BATCH=0`,
/// read once, logs its receipt).
pub fn ple_verify_batch_selected() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_PLE_VERIFY_BATCH")
            .ok()
            .as_deref()
            != Some("0");
        if on {
            tracing::info!(
                "ENGAGED ATLAS_QWEN4_PLE_VERIFY_BATCH: one PLE read + one pinned upload per verify"
            );
        }
        on
    })
}

/// Batched verify projections are on under the batch flag unless
/// `ATLAS_QWEN4_PLE_VERIFY_BATCH_PROJ=0` (A/B control: per-row projections).
fn batched_projections_selected() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_PLE_VERIFY_BATCH_PROJ")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// Fused multi-row verify PLE join + intermediate capture is on unless
/// `ATLAS_QWEN4_PLE_VERIFY_FUSE_ROWS=0`. Cached once so hot verify steps avoid
/// per-call libc environment locks and `String` allocations.
fn fuse_verify_rows_selected() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_PLE_VERIFY_FUSE_ROWS")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// Environment selector controlling PLE side-stream preparation overlap
/// (`H2D + dequant_k + key_proj + value_proj` on `secondary_stream` concurrently
/// with Layer-0 on the main stream):
/// - unset / `""` / `"auto"`: auto-ON on `active_sm_count() > 64` (SM120/SM103);
///   OFF on GB10 (`<= 64` SMs, preserving exact single-stream behavior).
/// - `"0"` / `"off"` / `"false"`: explicitly disabled.
/// - `"1"` / `"on"` / `"true"`: explicitly enabled.
const PLE_VERIFY_OVERLAP_ENV: &str = "ATLAS_QWEN4_PLE_OVERLAP";
const PLE_VERIFY_OVERLAP_HIGH_SM_THRESHOLD: u32 = 64;

#[must_use]
fn parse_ple_verify_overlap_override(raw: Option<&str>) -> Option<bool> {
    let raw = raw?.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return None;
    }
    if raw == "0" || raw.eq_ignore_ascii_case("off") || raw.eq_ignore_ascii_case("false") {
        Some(false)
    } else if raw == "1" || raw.eq_ignore_ascii_case("on") || raw.eq_ignore_ascii_case("true") {
        Some(true)
    } else {
        None
    }
}

#[cfg(test)]
#[must_use]
fn parse_ple_verify_overlap_for_sm(raw: Option<&str>, sm_count: u32) -> bool {
    parse_ple_verify_overlap_override(raw)
        .unwrap_or(sm_count > PLE_VERIFY_OVERLAP_HIGH_SM_THRESHOLD)
}

/// Environment selector controlling unified 48-layer C=1 verify CUDA graph
/// capture when PLE is `mmap`-backed (`ATLAS_PLE_MMAP=1` / `--ple-ram`).
/// Default ON unless `"0"` / `"off"` / `"false"`. Disk-backed PLE always stays
/// on the existing `PleWarmWorker` + 2-graph path.
const PLE_UNIFIED_GRAPH_ENV: &str = "ATLAS_QWEN4_PLE_UNIFIED_GRAPH";

#[must_use]
fn parse_ple_unified_graph_enabled(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(str::trim),
        Some(v) if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")
    )
}

/// Number of rotating pinned host staging slots for inline mmap C=1 PLE uploads
/// (`4 * 13,536 B = 54 KiB`), avoiding single-slot `wait_staging_free` spin-polls.
const PLE_UNIFIED_INLINE_SLOTS: usize = 4;
/// Byte offset of scales inside each unified-graph inline PLE slot/device buffer
/// (`9 * 16 * 90 = 12,960 B`, 16-byte aligned).
const PLE_INLINE_SCALE_BASE: usize = PLE_MAX_SPEC_ROWS * ROW_RECORD_BYTES;
/// Total bytes per inline unified-graph PLE slot (`12,960 + 576 = 13,536 B`).
const PLE_INLINE_SLOT_BYTES: usize = PLE_INLINE_SCALE_BASE + PLE_MAX_SPEC_ROWS * ROW_SCALE_BYTES;

/// Device + pinned staging for up to `PLE_MAX_BATCH_ROWS` verify/decode rows.
pub(super) struct Qwen4PleVerifyScratch {
    records_dev: DevicePtr,
    scales_dev: DevicePtr,
    /// Fixed contiguous device buffer for inline mmap unified-graph uploads:
    /// `[0..PLE_INLINE_SCALE_BASE)` records + `[PLE_INLINE_SCALE_BASE..PLE_INLINE_SLOT_BYTES)`
    /// scales, allowing one single ~13.5 KiB pinned H2D per C=1 verify round.
    inline_dev: DevicePtr,
    /// `[rows][PLE_EMBED]` BF16 dequantized embeddings.
    embedding_rows: DevicePtr,
    /// `[rows][residual]` BF16 key projections.
    key_rows: DevicePtr,
    /// `[rows][PLE_EMBED]` BF16 value projections.
    value_rows: DevicePtr,
    /// `[rows][16][90]` records, `[rows][16]` f32 scales, and up to
    /// `PLE_MAX_BATCH_ROWS` packed `u32` ragged fuse descriptors.
    staging: PinnedHostBuffer,
    /// Rotating `PLE_UNIFIED_INLINE_SLOTS`-slot pinned host buffer for inline
    /// mmap C=1 unified-graph uploads (`54 KiB`).
    inline_staging: PinnedHostBuffer,
    /// Next rotating slot index in `0..PLE_UNIFIED_INLINE_SLOTS`.
    inline_slot: usize,
    /// Per-slot upload completion events + armed flags for `inline_staging`.
    inline_upload_done: [u64; PLE_UNIFIED_INLINE_SLOTS],
    inline_upload_armed: [bool; PLE_UNIFIED_INLINE_SLOTS],
    /// Recorded after the staging upload; the next verify waits on it before
    /// rewriting `staging`.
    upload_done: u64,
    /// Recorded on the main stream inside a unified verify CUDA graph to fork
    /// `dequant_k + key_proj + value_proj` onto `prep_stream` concurrently with Layer 0.
    prep_fork: u64,
    /// Recorded on `prep_stream` after `H2D + dequant + key_proj + value_proj`
    /// finish so the main stream waits on it right before `fuse`.
    prep_done: u64,
    /// Most recently registered cuBLASLt side-stream workspace handle (0 if unset).
    registered_prep_stream: u64,
}

impl Qwen4PleVerifyScratch {
    pub(super) fn allocate(gpu: &dyn GpuBackend, residual: usize) -> Result<Self> {
        // The pinned staging buffers free themselves on drop if a later step fails.
        let staging = gpu.alloc_host_pinned(
            PLE_MAX_BATCH_ROWS * (ROW_RECORD_BYTES + ROW_SCALE_BYTES) + RAGGED_DESC_MAX_BYTES,
        )?;
        let inline_staging =
            gpu.alloc_host_pinned(PLE_UNIFIED_INLINE_SLOTS * PLE_INLINE_SLOT_BYTES)?;
        let mut created_events = Vec::with_capacity(3 + PLE_UNIFIED_INLINE_SLOTS);
        let create_ev = |events: &mut Vec<u64>| -> Result<u64> {
            match gpu.create_event() {
                Ok(ev) => {
                    events.push(ev);
                    Ok(ev)
                }
                Err(error) => {
                    for ev in events.drain(..) {
                        let _ = gpu.destroy_event(ev);
                    }
                    Err(error)
                }
            }
        };
        let upload_done = create_ev(&mut created_events)?;
        let prep_fork = create_ev(&mut created_events)?;
        let prep_done = create_ev(&mut created_events)?;
        let mut inline_upload_done = [0u64; PLE_UNIFIED_INLINE_SLOTS];
        for slot_ev in &mut inline_upload_done {
            *slot_ev = create_ev(&mut created_events)?;
        }
        let sizes = [
            PLE_MAX_BATCH_ROWS * ROW_RECORD_BYTES,
            PLE_MAX_BATCH_ROWS * ROW_SCALE_BYTES + RAGGED_DESC_MAX_BYTES,
            PLE_INLINE_SLOT_BYTES,
            PLE_MAX_BATCH_ROWS * PLE_EMBED * 2,
            PLE_MAX_BATCH_ROWS * residual * 2,
            PLE_MAX_BATCH_ROWS * PLE_EMBED * 2,
        ];
        let mut ptrs = Vec::with_capacity(sizes.len());
        for bytes in sizes {
            match gpu.alloc(bytes) {
                Ok(ptr) => ptrs.push(ptr),
                Err(error) => {
                    for ptr in ptrs {
                        let _ = gpu.free(ptr);
                    }
                    for ev in created_events {
                        let _ = gpu.destroy_event(ev);
                    }
                    return Err(error);
                }
            }
        }
        Ok(Self {
            records_dev: ptrs[0],
            scales_dev: ptrs[1],
            inline_dev: ptrs[2],
            embedding_rows: ptrs[3],
            key_rows: ptrs[4],
            value_rows: ptrs[5],
            staging,
            inline_staging,
            inline_slot: 0,
            inline_upload_done,
            inline_upload_armed: [false; PLE_UNIFIED_INLINE_SLOTS],
            upload_done,
            prep_fork,
            prep_done,
            registered_prep_stream: 0,
        })
    }

    pub(super) fn release(self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut errors = Vec::new();
        let mut staging_drained = true;
        if let Err(error) = self.wait_staging_free(gpu) {
            errors.push(format!("staging wait: {error:#}"));
            if let Err(sync_error) = gpu.synchronize_device() {
                staging_drained = false;
                errors.push(format!("staging fallback device sync: {sync_error:#}"));
            }
        }
        let mut inline_staging_drained = true;
        for (idx, (&ev, &armed)) in self
            .inline_upload_done
            .iter()
            .zip(self.inline_upload_armed.iter())
            .enumerate()
        {
            if !armed {
                continue;
            }
            if let Err(error) = self.wait_event_bounded(gpu, ev, "inline mmap staging upload") {
                errors.push(format!("inline staging wait[{idx}]: {error:#}"));
                if let Err(sync_error) = gpu.synchronize_device() {
                    inline_staging_drained = false;
                    errors.push(format!(
                        "inline staging fallback device sync[{idx}]: {sync_error:#}"
                    ));
                }
            }
        }
        let Self {
            records_dev,
            scales_dev,
            inline_dev,
            embedding_rows,
            key_rows,
            value_rows,
            staging,
            inline_staging,
            inline_slot: _,
            inline_upload_done,
            inline_upload_armed: _,
            upload_done,
            prep_fork,
            prep_done,
            registered_prep_stream: _,
        } = self;
        if staging_drained {
            drop(staging);
        } else {
            // Refuse to unmap page-locked host memory while an async H2D copy
            // may still be reading from it.
            std::mem::forget(staging);
        }
        if inline_staging_drained {
            drop(inline_staging);
        } else {
            // Refuse to unmap page-locked host memory while an async H2D copy
            // may still be reading from it.
            std::mem::forget(inline_staging);
        }
        for (name, ptr) in [
            ("records", records_dev),
            ("scales", scales_dev),
            ("inline", inline_dev),
            ("embeddings", embedding_rows),
            ("keys", key_rows),
            ("values", value_rows),
        ] {
            if let Err(error) = gpu.free(ptr) {
                errors.push(format!("{name}: {error:#}"));
            }
        }
        for (name, ev) in [
            ("prep_done event", prep_done),
            ("prep_fork event", prep_fork),
            ("upload event", upload_done),
        ] {
            if let Err(error) = gpu.destroy_event(ev) {
                errors.push(format!("{name}: {error:#}"));
            }
        }
        for (idx, ev) in inline_upload_done.into_iter().enumerate() {
            if let Err(error) = gpu.destroy_event(ev) {
                errors.push(format!("inline_upload_done[{idx}]: {error:#}"));
            }
        }
        ensure!(errors.is_empty(), "{}", errors.join("; "));
        Ok(())
    }

    fn wait_event_bounded(&self, gpu: &dyn GpuBackend, ev: u64, label: &str) -> Result<()> {
        let deadline = Instant::now() + STAGING_WAIT;
        while !gpu.poll_event(ev)? {
            if Instant::now() >= deadline {
                bail!("Qwen4 PLE {label} did not complete within {STAGING_WAIT:?}");
            }
            std::hint::spin_loop();
        }
        Ok(())
    }

    fn wait_staging_free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        self.wait_event_bounded(gpu, self.upload_done, "verify staging upload")
    }
}

/// The two-token tail of `prior` is all `select_decode` reads (it walks back
/// at most two tokens and stops at EOS), so hashing against the tail gives the
/// same selections as hashing against the full history.
#[cfg(test)]
fn selection_context(prior: &[u32]) -> Vec<u32> {
    prior[prior.len().saturating_sub(2)..].to_vec()
}

impl Qwen4PleLayer {
    /// Reserve native/verify PLE scratch while the model is being constructed,
    /// never after a request has already advanced layer 0.
    pub(crate) fn initialize_verify_scratch(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut verify = self.verify.lock();
        if verify.is_none() {
            *verify = Some(Qwen4PleVerifyScratch::allocate(gpu, self.residual_width())?);
        }
        drop(verify);
        let mut worker = self.warm_worker.lock();
        if worker.is_none()
            && let Ok(spawned) = PleWarmWorker::spawn(
                Arc::clone(&self.hasher),
                Arc::clone(&self.reader),
                Arc::clone(&self.parked_cache),
                Arc::clone(&self.io_poisoned),
            )
        {
            *worker = Some(spawned);
        }
        Ok(())
    }

    /// Read-only native-decode admission. The model calls this for the whole
    /// active tile before allocating KV, embedding, uploading metadata, or
    /// advancing an earlier layer. Execution repeats the poison checks under
    /// `scratch_submit` so a concurrent failure still cannot race admission.
    pub(crate) fn preflight_active_decode_rows(
        &self,
        rows: usize,
        row_stride: usize,
        slots: &[usize],
    ) -> Result<()> {
        validate_active_decode_layout(
            rows,
            row_stride,
            self.residual_width(),
            slots,
            self.max_batch_size,
        )?;
        ensure!(
            !self
                .scratch_poisoned
                .load(std::sync::atomic::Ordering::Acquire),
            "Qwen4 PLE shared scratch is poisoned after an unfenceable GPU failure"
        );
        ensure!(
            !self.io_poisoned.load(std::sync::atomic::Ordering::Acquire),
            "Qwen4 PLE I/O is poisoned; process restart is required"
        );
        ensure!(
            self.dequant_k.0 != 0 && self.fuse_k.0 != 0 && self.dense_gemv_k.0 != 0,
            "Qwen4 native concurrent PLE kernel set is incomplete"
        );
        ensure!(
            self.verify.lock().is_some(),
            "Qwen4 native concurrent PLE scratch was not reserved at model initialization"
        );
        Ok(())
    }

    /// The state-independent half of [`Self::preflight_active_decode_rows`]
    /// for a C`rows` tile: layout and slot-pool capacity (with the distinct
    /// slots `0..rows`), kernels and reserved scratch. The poison checks stay
    /// in the per-frame preflight: a poisoned PLE fails serial decode too.
    pub(crate) fn native_decode_capability(&self, rows: usize, row_stride: usize) -> Result<()> {
        let slots: Vec<usize> = (0..rows).collect();
        validate_active_decode_layout(
            rows,
            row_stride,
            self.residual_width(),
            &slots,
            self.max_batch_size,
        )?;
        ensure!(
            self.dequant_k.0 != 0 && self.fuse_k.0 != 0 && self.dense_gemv_k.0 != 0,
            "Qwen4 native concurrent PLE kernel set is incomplete"
        );
        ensure!(
            self.verify.lock().is_some(),
            "Qwen4 native concurrent PLE scratch was not reserved at model initialization"
        );
        Ok(())
    }

    /// Apply PLE once to each real sequence in a native concurrent-decode
    /// tile. C2-C256 share the scratch geometry already reserved at model load.
    ///
    /// Unlike [`Self::forward_tokens_verify`], rows are independent requests:
    /// every row hashes against its own history and fuses into its own
    /// `slot_idx`.  All row selection and storage reads are coalesced, and the
    /// dequant/key/value projections run as one row batch.  Only the tiny
    /// stateful fuse remains row ordered.  The input slice represents active
    /// rows only; padding rows are never constructed and therefore cannot
    /// mutate PLE state.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_active_decode_rows(
        &self,
        rows: &[PleActiveDecodeRow<'_>],
        hyper: DevicePtr,
        row_stride: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let row_count = rows.len();
        let mut slots_buf = [0usize; PLE_MAX_BATCH_ROWS];
        let slots: &[usize] = if row_count <= PLE_MAX_BATCH_ROWS {
            for (dst, row) in slots_buf.iter_mut().zip(rows) {
                *dst = row.slot_idx;
            }
            &slots_buf[..row_count]
        } else {
            &[]
        };
        self.preflight_active_decode_rows(row_count, row_stride, slots)?;
        ensure!(
            !hyper.is_null(),
            "Qwen4 native concurrent PLE hyper tile is null"
        );

        let _submit = self.scratch_submit.lock();
        ensure!(
            !self
                .scratch_poisoned
                .load(std::sync::atomic::Ordering::Acquire),
            "Qwen4 PLE shared scratch is poisoned after an unfenceable GPU failure"
        );
        ensure!(
            !self.io_poisoned.load(std::sync::atomic::Ordering::Acquire),
            "Qwen4 PLE I/O is poisoned; process restart is required"
        );

        // Token-major request packing preserves the exact 16-head order of
        // `forward_token`, but every token uses its own request history.
        let mut request = Vec::with_capacity(row_count * QWEN4_PLE_HEADS);
        for row in rows {
            let selections = self
                .hasher
                .select_decode(row.current_token, row.prior_tokens);
            request.extend(
                selections
                    .iter()
                    .map(|selection| (selection.shard, selection.row)),
            );
        }

        let record_bytes = row_count * ROW_RECORD_BYTES;
        let scale_base = PLE_MAX_BATCH_ROWS * ROW_RECORD_BYTES;
        let scale_bytes = row_count * ROW_SCALE_BYTES;

        let mut verify = self.verify.lock();
        verify
            .as_mut()
            .context("Qwen4 PLE verify scratch was not initialized")?
            .wait_staging_free(gpu)?;

        let staged_from_cache = {
            let staging = verify
                .as_mut()
                .context("Qwen4 PLE verify scratch was not initialized")?
                .staging
                .as_mut_slice();
            let (records_part, scales_part) = staging.split_at_mut(scale_base);
            self.parked_cache.try_fill_staging(
                &request,
                &mut records_part[..record_bytes],
                &mut scales_part[..scale_bytes],
            )?
        };
        if !staged_from_cache {
            drop(verify);
            let Some(mut reader) = self.reader.try_lock_for(self.io_timeout) else {
                self.io_poisoned
                    .store(true, std::sync::atomic::Ordering::Release);
                bail!(
                    "Qwen4 PLE reader lock exceeded its deadline; PLE is poisoned and process restart is required"
                );
            };
            let records = reader.read_rows_windowed(&request)?;
            ensure!(
                records.len() == row_count * QWEN4_PLE_HEADS,
                "Qwen4 native concurrent PLE read {} rows, expected {}",
                records.len(),
                row_count * QWEN4_PLE_HEADS
            );
            drop(reader);

            verify = self.verify.lock();
            let staging = verify
                .as_mut()
                .context("Qwen4 PLE verify scratch was not initialized")?
                .staging
                .as_mut_slice();
            pack_records_to_staging(staging, &records, scale_base);
        }
        let scratch = verify
            .as_mut()
            .context("Qwen4 PLE verify scratch was not initialized")?;

        gpu.stream_wait_event(stream, self.scratch_event)?;
        self.upload_staging_slices(
            scratch,
            StagingUpload {
                record_bytes,
                scale_base,
                scale_upload_bytes: scale_bytes,
            },
            gpu,
            stream,
        )?;

        let residual = self.residual_width();
        let embed_row = PLE_EMBED * 2;
        let key_row = residual * 2;
        let exec_res: Result<()> = (|| {
            spark_runtime::kernel_args::KernelLaunch::new(gpu, self.dequant_k)
                .grid([(PLE_EMBED as u32).div_ceil(256), row_count as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(scratch.records_dev)
                .arg_ptr(scratch.scales_dev)
                .arg_ptr(scratch.embedding_rows)
                .launch(stream)?;
            if self.dense_gemv_batchn_tf_k.0 != 0 && batched_projections_selected() {
                ops::dense_gemv_batchn_tf(
                    gpu,
                    self.dense_gemv_batchn_tf_k,
                    scratch.embedding_rows,
                    &self.key_proj,
                    scratch.key_rows,
                    row_count as u32,
                    residual as u32,
                    PLE_EMBED as u32,
                    stream,
                )?;
                ops::dense_gemv_batchn_tf(
                    gpu,
                    self.dense_gemv_batchn_tf_k,
                    scratch.embedding_rows,
                    &self.value_proj,
                    scratch.value_rows,
                    row_count as u32,
                    PLE_EMBED as u32,
                    PLE_EMBED as u32,
                    stream,
                )?;
            } else {
                for row in 0..row_count {
                    self.launch_projections(
                        scratch.embedding_rows.offset(row * embed_row),
                        scratch.key_rows.offset(row * key_row),
                        scratch.value_rows.offset(row * embed_row),
                        gpu,
                        stream,
                    )?;
                }
            }

            // Stateful work is active-row-only and every row names its own slot.
            // There is intentionally no loop to a padded/capture width here.
            for (row_idx, row) in rows.iter().enumerate() {
                self.launch_fuse(
                    hyper.offset(row_idx * row_stride),
                    scratch.key_rows.offset(row_idx * key_row),
                    scratch.value_rows.offset(row_idx * embed_row),
                    row.slot_idx,
                    row.reset_state,
                    gpu,
                    stream,
                )?;
            }
            Ok(())
        })();
        if let Err(err) = exec_res {
            return self.fence_or_poison_scratch(gpu, stream, None, err, "active decode");
        }
        self.record_scratch_fence(gpu, stream)?;
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::info!(
                "QWEN4_NATIVE_CONCURRENT_PLE_ENGAGED rows={row_count} state=slot_indexed padding=excluded"
            );
        });
        Ok(())
    }

    /// Apply PLE to `tokens.len()` consecutive verify rows (row r at
    /// `hyper + r * row_stride`) and save each row's conv history as
    /// intermediate r, equivalent to calling `forward_token` then
    /// `save_intermediate` per row with `prior ++ tokens[..r]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tokens_verify(
        &self,
        tokens: &[u32],
        prior_tokens: &[u32],
        hyper: DevicePtr,
        row_stride: usize,
        slot_idx: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let rows = tokens.len();
        ensure!(
            (1..=PLE_MAX_SPEC_ROWS).contains(&rows),
            "Qwen4 PLE verify batch needs 1..={PLE_MAX_SPEC_ROWS} rows, got {rows}"
        );
        ensure!(
            slot_idx < self.max_batch_size,
            "PLE slot {slot_idx} out of range"
        );
        let _submit = self.scratch_submit.lock();
        ensure!(
            !self
                .scratch_poisoned
                .load(std::sync::atomic::Ordering::Acquire),
            "Qwen4 PLE shared scratch is poisoned after an unfenceable GPU failure"
        );
        ensure!(
            !self.io_poisoned.load(std::sync::atomic::Ordering::Acquire),
            "Qwen4 PLE I/O is poisoned; process restart is required"
        );

        let mut context = RollingWindow::from_prior(prior_tokens);
        let mut request = StackSelections::<{ PLE_MAX_SPEC_ROWS * QWEN4_PLE_HEADS }>::new();
        for &token in tokens {
            let selections = self.hasher.select_decode(token, context.as_slice());
            for s in selections {
                request.push((s.shard, s.row));
            }
            context.push(token);
        }

        let record_bytes = rows * ROW_RECORD_BYTES;
        let scale_base = PLE_MAX_BATCH_ROWS * ROW_RECORD_BYTES;
        let scale_bytes = rows * ROW_SCALE_BYTES;

        let mut slot = self.verify.lock();
        slot.as_mut()
            .context("Qwen4 PLE verify scratch was not initialized")?
            .wait_staging_free(gpu)?;

        let staged_from_cache = {
            let staging = slot
                .as_mut()
                .context("Qwen4 PLE verify scratch was not initialized")?
                .staging
                .as_mut_slice();
            let (records_part, scales_part) = staging.split_at_mut(scale_base);
            self.parked_cache.try_fill_staging(
                &request,
                &mut records_part[..record_bytes],
                &mut scales_part[..scale_bytes],
            )?
        };
        if !staged_from_cache {
            drop(slot);
            let Some(mut reader) = self.reader.try_lock_for(self.io_timeout) else {
                self.io_poisoned
                    .store(true, std::sync::atomic::Ordering::Release);
                bail!(
                    "Qwen4 PLE reader lock exceeded its deadline; PLE is poisoned and process restart is required"
                );
            };
            let records = reader.read_rows_verify(&request)?;
            drop(reader);
            ensure!(
                records.len() == rows * QWEN4_PLE_HEADS,
                "Qwen4 PLE verify batch read {} rows, expected {}",
                records.len(),
                rows * QWEN4_PLE_HEADS
            );
            slot = self.verify.lock();
            let staging = slot
                .as_mut()
                .context("Qwen4 PLE verify scratch was not initialized")?
                .staging
                .as_mut_slice();
            pack_records_to_staging(staging, &records, scale_base);
        }
        let scratch = slot
            .as_mut()
            .context("Qwen4 PLE verify scratch was not initialized")?;

        gpu.stream_wait_event(stream, self.scratch_event)?;
        self.upload_staging_slices(
            scratch,
            StagingUpload {
                record_bytes,
                scale_base,
                scale_upload_bytes: scale_bytes,
            },
            gpu,
            stream,
        )?;
        spark_runtime::cublaslt::stage_timer::mark("ple_upload", stream);

        // Dequantize and project every row first (independent of the conv
        // history), then fuse rows in order: the same per-row kernels and inputs
        // as forward_token's interleaved dequant -> project -> fuse sequence.
        let residual = self.residual_width();
        let embed_row = PLE_EMBED * 2;
        let key_row = residual * 2;
        let exec_res: Result<()> = (|| {
            spark_runtime::kernel_args::KernelLaunch::new(gpu, self.dequant_k)
                .grid([(PLE_EMBED as u32).div_ceil(256), rows as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(scratch.records_dev)
                .arg_ptr(scratch.scales_dev)
                .arg_ptr(scratch.embedding_rows)
                .launch(stream)?;
            spark_runtime::cublaslt::stage_timer::mark("ple_dequant", stream);
            if self.dense_gemv_batchn_tf_k.0 != 0 && batched_projections_selected() {
                ops::dense_gemv_batchn_tf(
                    gpu,
                    self.dense_gemv_batchn_tf_k,
                    scratch.embedding_rows,
                    &self.key_proj,
                    scratch.key_rows,
                    rows as u32,
                    residual as u32,
                    PLE_EMBED as u32,
                    stream,
                )?;
                ops::dense_gemv_batchn_tf(
                    gpu,
                    self.dense_gemv_batchn_tf_k,
                    scratch.embedding_rows,
                    &self.value_proj,
                    scratch.value_rows,
                    rows as u32,
                    PLE_EMBED as u32,
                    PLE_EMBED as u32,
                    stream,
                )?;
            } else {
                for row in 0..rows {
                    self.launch_projections(
                        scratch.embedding_rows.offset(row * embed_row),
                        scratch.key_rows.offset(row * key_row),
                        scratch.value_rows.offset(row * embed_row),
                        gpu,
                        stream,
                    )?;
                }
            }
            spark_runtime::cublaslt::stage_timer::mark("ple_proj", stream);
            if self.fuse_verify_rows_k.0 != 0
                && row_stride.is_multiple_of(2)
                && fuse_verify_rows_selected()
            {
                self.launch_fuse_verify_rows(
                    hyper,
                    (row_stride / 2) as u32,
                    scratch.key_rows,
                    scratch.value_rows,
                    slot_idx,
                    rows as u32,
                    gpu,
                    stream,
                )?;
            } else {
                for row in 0..rows {
                    self.launch_fuse(
                        hyper.offset(row * row_stride),
                        scratch.key_rows.offset(row * key_row),
                        scratch.value_rows.offset(row * embed_row),
                        slot_idx,
                        false,
                        gpu,
                        stream,
                    )?;
                    self.save_intermediate(slot_idx, row, gpu, stream)?;
                }
            }
            Ok(())
        })();
        if let Err(err) = exec_res {
            return self.fence_or_poison_scratch(gpu, stream, None, err, "verify batch");
        }
        spark_runtime::cublaslt::stage_timer::mark("ple_fuse", stream);
        self.record_scratch_fence(gpu, stream)
    }

    /// Pre-read the PLE records for `seqs = &[(tokens, prior_tokens)]`
    /// on a background thread so [`Self::forward_cross_seq_verify`] does not
    /// repeat token hashing or `read_rows_verify`.
    #[allow(dead_code)]
    pub fn warm_cross_seq_verify(&self, seqs: &[(&[u32], &[u32])]) -> Result<Vec<PleNvfp4Row>> {
        let total_rows: usize = seqs.iter().map(|(t, _)| t.len()).sum();
        if total_rows == 0 || total_rows > PLE_MAX_BATCH_ROWS {
            return Ok(Vec::new());
        }
        let mut request = Vec::with_capacity(total_rows * QWEN4_PLE_HEADS);
        for &(tokens, prior_tokens) in seqs {
            let mut context = RollingWindow::from_prior(prior_tokens);
            for &token in tokens {
                let selections = self.hasher.select_decode(token, context.as_slice());
                request.extend(selections.iter().map(|s| (s.shard, s.row)));
                context.push(token);
            }
        }
        read_verify_with_parked_cache(&self.parked_cache, &self.reader, &request)
    }

    /// Submit a cross-sequence PLE pre-read to the persistent background
    /// [`PleWarmWorker`], spawning the worker lazily if not pre-spawned.
    ///
    /// Performs zero hashing, zero [`ParkedPleCache`] probing/copying, and zero
    /// storage I/O on the caller thread so layer 0 launches immediately while
    /// PLE lookup and any miss reads overlap layer 0's GPU execution. Signals
    /// immediate preemption of any in-flight or queued `DraftAhead` task;
    /// dropping the ticket before [`PleWarmTicket::wait`] drains the response
    /// so the worker never desynchronizes.
    pub fn submit_warm_cross_seq_verify(&self, seqs: Vec<PleWarmSeq>) -> Result<PleWarmTicket<'_>> {
        ensure!(
            !self.io_poisoned.load(Ordering::Acquire),
            "Qwen4 PLE I/O is poisoned; process restart is required"
        );
        let total_rows: usize = seqs.iter().map(|s| s.draft().len()).sum();
        if total_rows == 0 || total_rows > PLE_MAX_BATCH_ROWS {
            return Ok(PleWarmTicket::immediate(Vec::new()));
        }
        submit_warm_verify_with(&self.warm_worker, PleVerifyJob::Seqs(seqs), || {
            PleWarmWorker::spawn(
                Arc::clone(&self.hasher),
                Arc::clone(&self.reader),
                Arc::clone(&self.parked_cache),
                Arc::clone(&self.io_poisoned),
            )
        })
    }

    /// Submit a non-blocking, keyed reusable `Verify` pre-read to the
    /// persistent background [`PleWarmWorker`] right after draft proposal
    /// finishes so storage I/O overlaps with tick-end host work, KV
    /// allocation, metadata upload, and Layer 0.
    ///
    /// If the next [`Self::submit_warm_cross_seq_verify`] presents the exact
    /// same `Vec<PleWarmSeq>` key, it adopts this in-flight (or completed)
    /// worker response directly with zero preemption and zero extra I/O; on
    /// any key mismatch the stale prefetch is replaced in-place or drained
    /// cleanly while still warming `ParkedPleCache` / `RecordCache`.
    pub fn submit_verify_prefetch(&self, seqs: Vec<PleWarmSeq>) {
        let total_rows: usize = seqs.iter().map(|s| s.draft().len()).sum();
        if total_rows == 0
            || total_rows > PLE_MAX_BATCH_ROWS
            || self.io_poisoned.load(Ordering::Acquire)
        {
            return;
        }
        let Some(mut guard) = self.warm_worker.try_lock() else {
            return;
        };
        if guard.is_none() {
            match PleWarmWorker::spawn(
                Arc::clone(&self.hasher),
                Arc::clone(&self.reader),
                Arc::clone(&self.parked_cache),
                Arc::clone(&self.io_poisoned),
            ) {
                Ok(worker) => *guard = Some(worker),
                Err(_) => return,
            }
        }
        if let Some(worker) = guard.as_mut()
            && worker.try_submit_verify_prefetch(seqs) == DraftSendOutcome::Disconnected
        {
            guard.take();
        }
    }

    /// Asynchronously warm [`ParkedPleCache`] with the PLE rows for per-sequence
    /// `(tokens, prior_tokens)` windows on the persistent background
    /// [`PleWarmWorker`] without touching any GPU staging buffers, hashing on
    /// the scheduler thread, or blocking the caller.
    ///
    /// Copies only the compact token windows (`<= PLE_MAX_BATCH_ROWS` tokens)
    /// and hands them to the background worker's preemptible low-priority slot,
    /// where each sequence hashes with its own rolling 2-token `prior_tokens`
    /// context matching [`Self::forward_cross_seq_verify`] identically across
    /// all 8 bigram and 8 trigram heads. If the cache is disabled, a foreground
    /// verify ticket is active, or the worker is busy with verify, returns
    /// immediately with zero blocking waits.
    pub fn submit_draft_ahead_seqs(&self, seqs: &[(&[u32], &[u32])]) {
        if seqs.is_empty()
            || !self.draft_ahead_enabled
            || !self.parked_cache.is_enabled()
            || self.io_poisoned.load(Ordering::Acquire)
        {
            return;
        }
        let Some(mut guard) = self.warm_worker.try_lock() else {
            return;
        };
        let batch = PleDraftAheadBatch::from_seqs(seqs, PLE_MAX_BATCH_ROWS);
        if batch.is_empty() {
            return;
        }
        if guard.is_none() {
            match PleWarmWorker::spawn(
                Arc::clone(&self.hasher),
                Arc::clone(&self.reader),
                Arc::clone(&self.parked_cache),
                Arc::clone(&self.io_poisoned),
            ) {
                Ok(worker) => *guard = Some(worker),
                Err(_) => return,
            }
        }
        if let Some(worker) = guard.as_ref()
            && worker.try_send_draft_ahead_batch(batch) == DraftSendOutcome::Disconnected
        {
            guard.take();
        }
    }

    /// Asynchronously warm [`ParkedPleCache`] with the PLE rows for a single
    /// contiguous `tokens` slice (empty prior context) on the persistent
    /// background [`PleWarmWorker`]. Prefer [`Self::submit_draft_ahead_seqs`]
    /// when per-sequence prior context is available so row-0/row-1 bigram and
    /// trigram hashes match cross-sequence verification 100%.
    pub fn submit_draft_ahead_tokens(&self, tokens: &[u32]) {
        if tokens.is_empty() {
            return;
        }
        self.submit_draft_ahead_seqs(&[(tokens, &[])]);
    }

    /// Whether cross-sequence PLE preparation (`H2D + dequant_k + key_proj +
    /// value_proj`) should overlap on a side stream concurrently with Layer-0.
    ///
    /// Caches only the parsed env override so unset/`"auto"` resolves against live
    /// `active_sm_count()` (auto-ON on `> 64` SMs, OFF on GB10 `<= 64` SMs).
    #[must_use]
    pub fn verify_overlap_selected() -> bool {
        static OVERRIDE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
        let parsed = *OVERRIDE.get_or_init(|| {
            parse_ple_verify_overlap_override(std::env::var(PLE_VERIFY_OVERLAP_ENV).ok().as_deref())
        });
        parsed.unwrap_or_else(|| {
            atlas_core::device::active_sm_count() > PLE_VERIFY_OVERLAP_HIGH_SM_THRESHOLD
        })
    }

    /// Like [`Self::forward_cross_seq_verify`], optionally running the stateless
    /// preparation stages (`H2D + dequant_k + key_proj + value_proj`) on
    /// `prep_stream` concurrently with Layer-0 on `stream`, synchronized via
    /// `record_event`/`stream_wait_event` right before the stateful fuse kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cross_seq_verify_with_prep_stream(
        &self,
        seqs: &[(&[u32], &[u32], usize)],
        pre_read: Option<Vec<spark_storage::ple_offload::PleNvfp4Row>>,
        hyper: DevicePtr,
        row_stride: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
        prep_stream: Option<u64>,
    ) -> Result<()> {
        self.forward_cross_seq_verify_inner(
            seqs,
            pre_read,
            hyper,
            row_stride,
            gpu,
            stream,
            prep_stream,
        )
    }

    /// Batched PLE step across multiple sequences' speculative verify windows
    /// (`seqs = &[(tokens, prior_tokens, slot_idx)]`).
    ///
    /// Reads all `total_rows = sum_s tokens_s.len()` records in one windowed
    /// io_uring call (or consumes `pre_read` from [`Self::warm_cross_seq_verify`]),
    /// uploads in one pinned H2D transfer, dequantizes all rows at once, runs
    /// `key_proj` and `value_proj` once across all `total_rows`
    /// (`dense_gemv_batchn_tf`, streaming 65.5 MB PLE weights once instead of
    /// `seqs` times), and then fuses all sequences in one launch (including
    /// ragged mixed-`K_s` frames via `qwen4_ple_fuse_verify_rows_ragged`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cross_seq_verify(
        &self,
        seqs: &[(&[u32], &[u32], usize)],
        pre_read: Option<Vec<spark_storage::ple_offload::PleNvfp4Row>>,
        hyper: DevicePtr,
        row_stride: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.forward_cross_seq_verify_inner(seqs, pre_read, hyper, row_stride, gpu, stream, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_cross_seq_verify_inner(
        &self,
        seqs: &[(&[u32], &[u32], usize)],
        pre_read: Option<Vec<spark_storage::ple_offload::PleNvfp4Row>>,
        hyper: DevicePtr,
        row_stride: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
        prep_stream: Option<u64>,
    ) -> Result<()> {
        let total_rows: usize = seqs.iter().map(|(t, _, _)| t.len()).sum();
        ensure!(
            (1..=PLE_MAX_BATCH_ROWS).contains(&total_rows),
            "Qwen4 PLE cross-seq verify batch needs 1..={PLE_MAX_BATCH_ROWS} rows, got {total_rows}"
        );
        for &(tokens, _, slot_idx) in seqs {
            ensure!(
                tokens.len() <= PLE_MAX_SPEC_ROWS,
                "Qwen4 PLE per-sequence verify window {} exceeds {PLE_MAX_SPEC_ROWS}",
                tokens.len()
            );
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
        }
        let _submit = self.scratch_submit.lock();
        ensure!(
            !self
                .scratch_poisoned
                .load(std::sync::atomic::Ordering::Acquire),
            "Qwen4 PLE shared scratch is poisoned after an unfenceable GPU failure"
        );
        ensure!(
            !self.io_poisoned.load(std::sync::atomic::Ordering::Acquire),
            "Qwen4 PLE I/O is poisoned; process restart is required"
        );

        let record_bytes = total_rows * ROW_RECORD_BYTES;
        let scale_base = PLE_MAX_BATCH_ROWS * ROW_RECORD_BYTES;
        let scale_bytes = total_rows * ROW_SCALE_BYTES;

        let fuse_flag = row_stride.is_multiple_of(2) && fuse_verify_rows_selected();
        let has_uniform_k = self.fuse_verify_rows_k.0 != 0;
        let has_ragged_k = self.fuse_verify_rows_ragged_k.0 != 0;
        let seq0_rows = seqs[0].0.len();
        let all_uniform_small_slot = is_uniform_small_slot(seqs);
        let ragged_descs = if fuse_flag
            && has_ragged_k
            && !(has_uniform_k && all_uniform_small_slot && seqs.len() <= 8)
        {
            pack_ragged_ple_fuse_descs(seqs)
        } else {
            None
        };
        let fuse_route = select_ple_verify_fuse_route(
            fuse_flag,
            has_uniform_k,
            has_ragged_k,
            seqs.len(),
            all_uniform_small_slot,
            ragged_descs.is_some(),
        );

        let mut slot = self.verify.lock();
        slot.as_mut()
            .context("Qwen4 PLE verify scratch was not initialized")?
            .wait_staging_free(gpu)?;

        match pre_read {
            Some(r) if r.len() == total_rows * QWEN4_PLE_HEADS => {
                let staging = slot
                    .as_mut()
                    .context("Qwen4 PLE verify scratch was not initialized")?
                    .staging
                    .as_mut_slice();
                pack_records_to_staging(staging, &r, scale_base);
            }
            _ => {
                let mut request = Vec::with_capacity(total_rows * QWEN4_PLE_HEADS);
                for &(tokens, prior_tokens, _) in seqs {
                    let mut context = RollingWindow::from_prior(prior_tokens);
                    for &token in tokens {
                        let selections = self.hasher.select_decode(token, context.as_slice());
                        request.extend(selections.iter().map(|s| (s.shard, s.row)));
                        context.push(token);
                    }
                }
                let staged_from_cache = {
                    let staging = slot
                        .as_mut()
                        .context("Qwen4 PLE verify scratch was not initialized")?
                        .staging
                        .as_mut_slice();
                    let (records_part, scales_part) = staging.split_at_mut(scale_base);
                    self.parked_cache.try_fill_staging(
                        &request,
                        &mut records_part[..record_bytes],
                        &mut scales_part[..scale_bytes],
                    )?
                };
                if !staged_from_cache {
                    drop(slot);
                    let Some(mut reader) = self.reader.try_lock_for(self.io_timeout) else {
                        self.io_poisoned
                            .store(true, std::sync::atomic::Ordering::Release);
                        bail!(
                            "Qwen4 PLE reader lock exceeded its deadline; PLE is poisoned and process restart is required"
                        );
                    };
                    let records = reader.read_rows_verify(&request)?;
                    drop(reader);
                    ensure!(
                        records.len() == total_rows * QWEN4_PLE_HEADS,
                        "Qwen4 PLE cross-seq verify batch read {} rows, expected {}",
                        records.len(),
                        total_rows * QWEN4_PLE_HEADS
                    );
                    slot = self.verify.lock();
                    let staging = slot
                        .as_mut()
                        .context("Qwen4 PLE verify scratch was not initialized")?
                        .staging
                        .as_mut_slice();
                    pack_records_to_staging(staging, &records, scale_base);
                }
            }
        }
        let scratch = slot
            .as_mut()
            .context("Qwen4 PLE verify scratch was not initialized")?;

        let desc_bytes = if fuse_route == PleVerifyFuseRoute::RaggedSingleLaunch
            && let Some(descs) = ragged_descs.as_ref()
        {
            let bytes = descs.byte_len();
            let desc_start = scale_base + scale_bytes;
            descs.write_le_bytes(
                &mut scratch.staging.as_mut_slice()[desc_start..desc_start + bytes],
            );
            bytes
        } else {
            0
        };

        let tc_dense_active = crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active()
            && crate::layers::qwen4_hyper::batched_verify_tc_component_active(
                crate::layers::qwen4_hyper::BatchedVerifyTcComponent::PleProj,
                total_rows,
                false,
            );
        let proj_route = select_ple_cross_seq_proj_route_with_policy(
            batched_projections_selected(),
            tc_dense_active,
            self.dense_gemv_batchn_tf_k.0 != 0,
            crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
        );
        let mut effective_prep_stream = match prep_stream {
            Some(ps) if ps != 0 && ps != stream && scratch.prep_done != 0 => ps,
            _ => stream,
        };
        if effective_prep_stream != stream
            && proj_route == PleCrossSeqProjRoute::CublasBf16Gemm
            && scratch.registered_prep_stream != effective_prep_stream
        {
            // Give `effective_prep_stream` a dedicated cuBLASLt workspace so
            // concurrent `key_proj`/`value_proj` GEMMs never race on the global
            // workspace with Layer-0 cuBLASLt launches on `stream`.
            match spark_runtime::cublaslt::register_stream_workspace(effective_prep_stream) {
                Ok(()) => scratch.registered_prep_stream = effective_prep_stream,
                Err(err) => {
                    tracing::warn!(
                        stream = effective_prep_stream,
                        "QWEN4_PLE_VERIFY_STREAM_OVERLAP workspace registration failed ({err:#}); \
                         falling back to main stream"
                    );
                    effective_prep_stream = stream;
                }
            }
        }
        let overlapping = effective_prep_stream != stream;

        gpu.stream_wait_event(effective_prep_stream, self.scratch_event)?;
        self.upload_staging_slices(
            scratch,
            StagingUpload {
                record_bytes,
                scale_base,
                scale_upload_bytes: scale_bytes + desc_bytes,
            },
            gpu,
            effective_prep_stream,
        )?;
        if !overlapping {
            spark_runtime::cublaslt::stage_timer::mark("ple_upload", stream);
        }

        let residual = self.residual_width();
        let embed_row = PLE_EMBED * 2;
        let key_row = residual * 2;
        let exec_res: Result<()> = (|| {
            spark_runtime::kernel_args::KernelLaunch::new(gpu, self.dequant_k)
                .grid([(PLE_EMBED as u32).div_ceil(256), total_rows as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(scratch.records_dev)
                .arg_ptr(scratch.scales_dev)
                .arg_ptr(scratch.embedding_rows)
                .launch(effective_prep_stream)?;
            if !overlapping {
                spark_runtime::cublaslt::stage_timer::mark("ple_dequant", stream);
            }
            match proj_route {
                PleCrossSeqProjRoute::CublasBf16Gemm => {
                    if overlapping {
                        // Use the side-stream-workspace-isolated typed cuBLASLt path
                        // (`Bf16 -> Bf16`, FP32 accumulation) so concurrent Layer-0
                        // GEMMs on `stream` cannot collide on `ctx.workspace`.
                        let rows_u32 = total_rows as u32;
                        let embed_u32 = PLE_EMBED as u32;
                        let residual_u32 = residual as u32;
                        spark_runtime::cublaslt::gemm_act_weight_t_typed_pinned(
                            scratch.embedding_rows.0,
                            embed_u32,
                            self.key_proj.weight.0,
                            scratch.key_rows.0,
                            residual_u32,
                            rows_u32,
                            residual_u32,
                            embed_u32,
                            spark_runtime::cublaslt::GemmDtype::Bf16,
                            spark_runtime::cublaslt::GemmDtype::Bf16,
                            false,
                            rows_u32,
                            effective_prep_stream,
                        )?;
                        spark_runtime::cublaslt::gemm_act_weight_t_typed_pinned(
                            scratch.embedding_rows.0,
                            embed_u32,
                            self.value_proj.weight.0,
                            scratch.value_rows.0,
                            embed_u32,
                            rows_u32,
                            embed_u32,
                            embed_u32,
                            spark_runtime::cublaslt::GemmDtype::Bf16,
                            spark_runtime::cublaslt::GemmDtype::Bf16,
                            false,
                            rows_u32,
                            effective_prep_stream,
                        )?;
                    } else {
                        // Exact transpose & output layout parity with `dense_gemv_batchn_tf`:
                        // both compute `out[M, N] = act[M, K] @ weight[N, K]^T` in BF16 row-major
                        // with FP32 accumulation (same call as prefill PLE in `qwen4_ple.rs`).
                        ops::cublas_bf16_proj_dense(
                            scratch.embedding_rows,
                            &self.key_proj,
                            scratch.key_rows,
                            total_rows as u32,
                            residual as u32,
                            PLE_EMBED as u32,
                            stream,
                        )?;
                        ops::cublas_bf16_proj_dense(
                            scratch.embedding_rows,
                            &self.value_proj,
                            scratch.value_rows,
                            total_rows as u32,
                            PLE_EMBED as u32,
                            PLE_EMBED as u32,
                            stream,
                        )?;
                    }
                    static PLE_TC_LOGGED: std::sync::Once = std::sync::Once::new();
                    PLE_TC_LOGGED.call_once(|| {
                        tracing::info!(
                            selector = crate::layers::qwen4_hyper::BATCHED_VERIFY_TC_DENSE_ENV,
                            component = "ple_proj",
                            route = if overlapping {
                                "cublas_bf16_typed_side_stream"
                            } else {
                                "cublas_bf16_proj_dense"
                            },
                            rows = total_rows,
                            residual,
                            embed = PLE_EMBED,
                            "QWEN4_BATCHED_VERIFY_TC_DENSE_ENGAGED"
                        );
                    });
                }
                PleCrossSeqProjRoute::BatchedGemvTf => {
                    ops::dense_gemv_batchn_tf(
                        gpu,
                        self.dense_gemv_batchn_tf_k,
                        scratch.embedding_rows,
                        &self.key_proj,
                        scratch.key_rows,
                        total_rows as u32,
                        residual as u32,
                        PLE_EMBED as u32,
                        effective_prep_stream,
                    )?;
                    ops::dense_gemv_batchn_tf(
                        gpu,
                        self.dense_gemv_batchn_tf_k,
                        scratch.embedding_rows,
                        &self.value_proj,
                        scratch.value_rows,
                        total_rows as u32,
                        PLE_EMBED as u32,
                        PLE_EMBED as u32,
                        effective_prep_stream,
                    )?;
                }
                PleCrossSeqProjRoute::PerRowGemv => {
                    for row in 0..total_rows {
                        self.launch_projections(
                            scratch.embedding_rows.offset(row * embed_row),
                            scratch.key_rows.offset(row * key_row),
                            scratch.value_rows.offset(row * embed_row),
                            gpu,
                            effective_prep_stream,
                        )?;
                    }
                }
            }
            if overlapping {
                gpu.record_event(scratch.prep_done, effective_prep_stream)?;
                gpu.stream_wait_event(stream, scratch.prep_done)?;
                static OVERLAP_LOGGED: std::sync::Once = std::sync::Once::new();
                OVERLAP_LOGGED.call_once(|| {
                    tracing::info!(
                        selector = PLE_VERIFY_OVERLAP_ENV,
                        rows = total_rows,
                        prep_stream = effective_prep_stream,
                        main_stream = stream,
                        "QWEN4_PLE_VERIFY_STREAM_OVERLAP_ENGAGED"
                    );
                });
            } else {
                spark_runtime::cublaslt::stage_timer::mark("ple_proj", stream);
            }
            match fuse_route {
                PleVerifyFuseRoute::UniformSingleLaunch => {
                    let mut packed_slots = 0u64;
                    for (s, &(_, _, slot_idx)) in seqs.iter().enumerate() {
                        packed_slots |= (slot_idx as u64) << (s * 8);
                    }
                    self.launch_fuse_verify_rows_multi(
                        hyper,
                        (row_stride / 2) as u32,
                        scratch.key_rows,
                        scratch.value_rows,
                        seqs.len() as u32,
                        seq0_rows as u32,
                        packed_slots,
                        gpu,
                        stream,
                    )?;
                }
                PleVerifyFuseRoute::RaggedSingleLaunch => {
                    let descs = ragged_descs
                        .as_ref()
                        .expect("RaggedSingleLaunch requires packed ragged descriptors");
                    self.launch_fuse_verify_rows_ragged(
                        hyper,
                        (row_stride / 2) as u32,
                        scratch.key_rows,
                        scratch.value_rows,
                        descs.as_slice().len() as u32,
                        scratch.scales_dev.offset(scale_bytes),
                        gpu,
                        stream,
                    )?;
                }
                PleVerifyFuseRoute::UniformChunkedFallback => {
                    let mut row_base = 0usize;
                    for chunk in seqs.chunks(8) {
                        let mut packed_slots = 0u64;
                        for (s, &(_, _, slot_idx)) in chunk.iter().enumerate() {
                            packed_slots |= (slot_idx as u64) << (s * 8);
                        }
                        self.launch_fuse_verify_rows_multi(
                            hyper.offset(row_base * row_stride),
                            (row_stride / 2) as u32,
                            scratch.key_rows.offset(row_base * key_row),
                            scratch.value_rows.offset(row_base * embed_row),
                            chunk.len() as u32,
                            seq0_rows as u32,
                            packed_slots,
                            gpu,
                            stream,
                        )?;
                        row_base += chunk.len() * seq0_rows;
                    }
                }
                PleVerifyFuseRoute::PerSeqFallback => {
                    let mut row_base = 0usize;
                    for &(tokens, _, slot_idx) in seqs {
                        let seq_rows = tokens.len();
                        self.launch_fuse_verify_rows(
                            hyper.offset(row_base * row_stride),
                            (row_stride / 2) as u32,
                            scratch.key_rows.offset(row_base * key_row),
                            scratch.value_rows.offset(row_base * embed_row),
                            slot_idx,
                            seq_rows as u32,
                            gpu,
                            stream,
                        )?;
                        row_base += seq_rows;
                    }
                }
                PleVerifyFuseRoute::PerRowFallback => {
                    let mut row_base = 0usize;
                    for &(tokens, _, slot_idx) in seqs {
                        let seq_rows = tokens.len();
                        for r in 0..seq_rows {
                            let row = row_base + r;
                            self.launch_fuse(
                                hyper.offset(row * row_stride),
                                scratch.key_rows.offset(row * key_row),
                                scratch.value_rows.offset(row * embed_row),
                                slot_idx,
                                false,
                                gpu,
                                stream,
                            )?;
                            self.save_intermediate(slot_idx, r, gpu, stream)?;
                        }
                        row_base += seq_rows;
                    }
                }
            }
            Ok(())
        })();
        if let Err(err) = exec_res {
            return self.fence_or_poison_scratch(
                gpu,
                stream,
                overlapping.then_some(effective_prep_stream),
                err,
                "cross-seq verify",
            );
        }
        spark_runtime::cublaslt::stage_timer::mark("ple_fuse", stream);
        self.record_scratch_fence(gpu, stream)
    }

    /// Upload `(records, scales[+descs])` from `scratch.staging` on
    /// `upload_stream` and fence `scratch.upload_done` even on partial errors.
    fn upload_staging_slices(
        &self,
        scratch: &Qwen4PleVerifyScratch,
        upload: StagingUpload,
        gpu: &dyn GpuBackend,
        upload_stream: u64,
    ) -> Result<()> {
        let records_slice = scratch.staging.pinned_slice_range(0, upload.record_bytes)?;
        let scales_slice = scratch
            .staging
            .pinned_slice_range(upload.scale_base, upload.scale_upload_bytes)?;
        // SAFETY: `scratch.staging` is backend-owned page-locked memory owned
        // by this layer, `wait_staging_free` proved the previous upload
        // completed before `staging` was rewritten, and these eager copies are
        // never recorded into a CUDA graph. Once the first copy is enqueued,
        // the error path below always records `scratch.upload_done` or
        // synchronizes `upload_stream` (poisoning `scratch_poisoned` if both
        // fail) so `staging` is never mutated or dropped while DMA is in flight.
        unsafe {
            gpu.copy_h2d_pinned_async(records_slice, scratch.records_dev, upload_stream)?;
        }
        // SAFETY: same lifetime and fence invariant as the records copy above.
        let second_res =
            unsafe { gpu.copy_h2d_pinned_async(scales_slice, scratch.scales_dev, upload_stream) };
        let record_res = gpu.record_event(scratch.upload_done, upload_stream);
        match (second_res, record_res) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(copy_err), Ok(())) => {
                self.fence_or_poison_scratch(gpu, upload_stream, None, copy_err, "staging upload")
            }
            (copy_outcome, Err(rec_err)) => {
                if let Err(sync_err) = gpu.synchronize(upload_stream) {
                    self.scratch_poisoned.store(true, Ordering::Release);
                    tracing::error!(
                        stream = upload_stream,
                        record_err = %format!("{rec_err:#}"),
                        sync_err = %format!("{sync_err:#}"),
                        "Qwen4 PLE failed to fence or synchronize in-flight staging DMA; shared scratch poisoned"
                    );
                    let base = copy_outcome.err().unwrap_or(rec_err);
                    return Err(base.context(format!(
                        "Qwen4 PLE staging upload fence and stream sync both failed ({sync_err:#}); shared scratch poisoned"
                    )));
                }
                let base = copy_outcome.err().unwrap_or(rec_err);
                self.fence_or_poison_scratch(gpu, upload_stream, None, base, "staging upload fence")
            }
        }
    }

    /// Synchronize active stream(s) and re-record `self.scratch_event` after a
    /// mid-execution GPU error, poisoning `scratch_poisoned` if fencing fails.
    fn fence_or_poison_scratch(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        extra_stream: Option<u64>,
        error: anyhow::Error,
        stage: &'static str,
    ) -> Result<()> {
        let prep_fence = match extra_stream {
            Some(prep) if prep != stream => gpu.synchronize(prep),
            _ => Ok(()),
        };
        let fence = prep_fence
            .and_then(|()| gpu.synchronize(stream))
            .and_then(|()| gpu.record_event(self.scratch_event, stream));
        match fence {
            Ok(()) => Err(error.context(format!(
                "Qwen4 PLE {stage} failed after shared scratch was fenced"
            ))),
            Err(fence_error) => {
                self.scratch_poisoned.store(true, Ordering::Release);
                tracing::error!(
                    stage,
                    stream,
                    ?extra_stream,
                    error = %format!("{error:#}"),
                    fence_error = %format!("{fence_error:#}"),
                    "Qwen4 PLE failed to fence shared scratch after GPU error; shared scratch poisoned"
                );
                Err(error.context(format!(
                    "Qwen4 PLE {stage} failed and fencing failed ({fence_error:#}); shared scratch poisoned"
                )))
            }
        }
    }

    /// Record `self.scratch_event` on `stream` after successful scratch use,
    /// poisoning `scratch_poisoned` if recording fails.
    fn record_scratch_fence(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        if let Err(error) = gpu.record_event(self.scratch_event, stream) {
            self.scratch_poisoned.store(true, Ordering::Release);
            tracing::error!(
                stream,
                error = %format!("{error:#}"),
                "Qwen4 PLE post-success scratch fence failed; shared scratch poisoned"
            );
            return Err(error
                .context("Qwen4 PLE post-success scratch fence failed; shared scratch poisoned"));
        }
        Ok(())
    }

    /// True when PLE shards are served from read-only `mmap` views (`--ple-ram` /
    /// `ATLAS_PLE_MMAP=1`) rather than disk `pread64` / `io_uring`.
    #[must_use]
    pub fn is_mmap_backed(&self) -> bool {
        self.reader.lock().has_mmap()
    }

    /// Whether `ATLAS_QWEN4_PLE_UNIFIED_GRAPH` permits capturing PLE inside one
    /// unified 48-layer C=1 verify CUDA graph when PLE is `mmap`-backed (default ON).
    #[must_use]
    pub fn unified_graph_selected() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            parse_ple_unified_graph_enabled(std::env::var(PLE_UNIFIED_GRAPH_ENV).ok().as_deref())
        })
    }

    /// Check whether this C=1 verify frame can run PLE inline (`mmap` pack + 1 pinned
    /// H2D before Layer 0, then in-graph `dequant + key_proj + value_proj + fuse`
    /// inside a single unified 48-layer CUDA graph).
    #[must_use]
    pub fn unified_graph_eligible(&self, rows: usize, row_stride: usize, slot_idx: usize) -> bool {
        Self::unified_graph_selected()
            && ple_verify_batch_selected()
            && self.is_mmap_backed()
            && (1..=PLE_MAX_SPEC_ROWS).contains(&rows)
            && slot_idx < self.max_batch_size
            && row_stride.is_multiple_of(2)
            && !self.scratch_poisoned.load(Ordering::Acquire)
            && !self.io_poisoned.load(Ordering::Acquire)
            && self.dequant_k.0 != 0
            && self.verify.lock().is_some()
    }

    fn unified_verify_proj_route(&self, rows: usize) -> PleCrossSeqProjRoute {
        let tc_dense_active = crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active()
            && crate::layers::qwen4_hyper::batched_verify_tc_component_active(
                crate::layers::qwen4_hyper::BatchedVerifyTcComponent::PleProj,
                rows,
                false,
            );
        select_ple_cross_seq_proj_route_with_policy(
            batched_projections_selected(),
            tc_dense_active,
            self.dense_gemv_batchn_tf_k.0 != 0,
            crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
        )
    }

    /// Pre-warm cuBLASLt side-stream workspace and pinned algorithms strictly OUTSIDE
    /// CUDA graph capture for a unified 48-layer C=1 verify graph. Returns `Ok(true)`
    /// when in-graph PLE preparation overlap on `prep_stream` is ready to engage.
    pub fn prepare_unified_verify_graph(
        &self,
        rows: usize,
        prep_stream: Option<u64>,
        stream: u64,
    ) -> Result<bool> {
        let proj_route = self.unified_verify_proj_route(rows);
        let mut slot = self.verify.lock();
        let scratch = slot
            .as_mut()
            .context("Qwen4 PLE verify scratch was not initialized")?;
        let mut want_overlap = Self::verify_overlap_selected()
            && prep_stream.is_some_and(|ps| ps != 0 && ps != stream)
            && scratch.prep_fork != 0
            && scratch.prep_done != 0;
        if proj_route == PleCrossSeqProjRoute::CublasBf16Gemm {
            let residual = self.residual_width() as u32;
            let embed = PLE_EMBED as u32;
            let rows_u32 = rows as u32;
            if want_overlap && let Some(ps) = prep_stream {
                if scratch.registered_prep_stream != ps {
                    match spark_runtime::cublaslt::register_stream_workspace(ps) {
                        Ok(()) => scratch.registered_prep_stream = ps,
                        Err(err) => {
                            tracing::warn!(
                                stream = ps,
                                "QWEN4_PLE_UNIFIED_GRAPH side-stream cuBLASLt workspace registration failed ({err:#}); falling back to in-graph serial PLE"
                            );
                            want_overlap = false;
                        }
                    }
                }
                if want_overlap {
                    // Pin both cuBLASLt algorithms strictly outside graph capture so
                    // `gemm_act_weight_t_typed_pinned` does zero lazy initialization inside capture.
                    let key_pin = spark_runtime::cublaslt::pinned_algo_config(
                        embed,
                        residual,
                        residual,
                        embed,
                        spark_runtime::cublaslt::GemmDtype::Bf16,
                        spark_runtime::cublaslt::GemmDtype::Bf16,
                        false,
                        rows_u32,
                    );
                    let val_pin = spark_runtime::cublaslt::pinned_algo_config(
                        embed,
                        embed,
                        embed,
                        embed,
                        spark_runtime::cublaslt::GemmDtype::Bf16,
                        spark_runtime::cublaslt::GemmDtype::Bf16,
                        false,
                        rows_u32,
                    );
                    if key_pin.is_err() || val_pin.is_err() {
                        want_overlap = false;
                    }
                }
            }
        }
        Ok(want_overlap)
    }

    /// Hash and pack `tokens` inline on the main thread from `mmap`-backed PLE shards
    /// (`<3 µs`, zero `PleWarmWorker` wakeup, zero `wait_staging_free` single-slot stall)
    /// and issue 1 single pinned H2D into `inline_dev` before launching/capturing the
    /// unified 48-layer C=1 verify CUDA graph.
    pub fn upload_verify_records_inline_mmap(
        &self,
        tokens: &[u32],
        prior_tokens: &[u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let rows = tokens.len();
        ensure!(
            (1..=PLE_MAX_SPEC_ROWS).contains(&rows),
            "Qwen4 PLE inline mmap upload needs 1..={PLE_MAX_SPEC_ROWS} rows, got {rows}"
        );
        let _submit = self.scratch_submit.lock();
        ensure!(
            !self.scratch_poisoned.load(Ordering::Acquire),
            "Qwen4 PLE shared scratch is poisoned after an unfenceable GPU failure"
        );
        ensure!(
            !self.io_poisoned.load(Ordering::Acquire),
            "Qwen4 PLE I/O is poisoned; process restart is required"
        );

        let mut context = RollingWindow::from_prior(prior_tokens);
        let mut request = StackSelections::<{ PLE_MAX_SPEC_ROWS * QWEN4_PLE_HEADS }>::new();
        for &token in tokens {
            let selections = self.hasher.select_decode(token, context.as_slice());
            for s in selections {
                request.push((s.shard, s.row));
            }
            context.push(token);
        }

        let record_bytes = rows * ROW_RECORD_BYTES;
        let scale_bytes = rows * ROW_SCALE_BYTES;
        let upload_bytes = PLE_INLINE_SCALE_BASE + scale_bytes;

        let mut slot_guard = self.verify.lock();
        let scratch = slot_guard
            .as_mut()
            .context("Qwen4 PLE verify scratch was not initialized")?;
        let slot_idx = scratch.inline_slot;
        scratch.inline_slot = (slot_idx + 1) % PLE_UNIFIED_INLINE_SLOTS;
        if scratch.inline_upload_armed[slot_idx] {
            let ev = scratch.inline_upload_done[slot_idx];
            if !gpu.poll_event(ev)? {
                scratch.wait_event_bounded(gpu, ev, "inline mmap staging upload")?;
            }
        }
        let slot_base = slot_idx * PLE_INLINE_SLOT_BYTES;
        let staged_from_cache = {
            let slot_slice = &mut scratch.inline_staging.as_mut_slice()
                [slot_base..slot_base + PLE_INLINE_SLOT_BYTES];
            let (records_part, scales_part) = slot_slice.split_at_mut(PLE_INLINE_SCALE_BASE);
            self.parked_cache.try_fill_staging(
                &request,
                &mut records_part[..record_bytes],
                &mut scales_part[..scale_bytes],
            )?
        };
        if !staged_from_cache {
            drop(slot_guard);
            let records = self.reader.lock().read_rows_verify(&request)?;
            ensure!(
                records.len() == rows * QWEN4_PLE_HEADS,
                "Qwen4 PLE inline mmap read {} rows, expected {}",
                records.len(),
                rows * QWEN4_PLE_HEADS
            );
            slot_guard = self.verify.lock();
            let scratch = slot_guard
                .as_mut()
                .context("Qwen4 PLE verify scratch was not initialized")?;
            let slot_slice = &mut scratch.inline_staging.as_mut_slice()
                [slot_base..slot_base + PLE_INLINE_SLOT_BYTES];
            pack_records_to_staging(slot_slice, &records, PLE_INLINE_SCALE_BASE);
        }
        let scratch = slot_guard
            .as_mut()
            .context("Qwen4 PLE verify scratch was not initialized")?;

        gpu.stream_wait_event(stream, self.scratch_event)?;
        // SAFETY: `inline_staging` is backend-owned page-locked memory owned by this
        // layer; `slot_idx` rotates across 4 independent slots and any still-in-flight
        // prior upload on `slot_idx` completed before this slot was overwritten. This
        // upload always executes outside CUDA graph capture before the unified graph,
        // and the error branch below drains `stream` (or poisons `scratch_poisoned`)
        // if recording `inline_upload_done[slot_idx]` fails after DMA enqueue.
        unsafe {
            gpu.copy_h2d_pinned_async(
                scratch
                    .inline_staging
                    .pinned_slice_range(slot_base, upload_bytes)?,
                scratch.inline_dev,
                stream,
            )?;
        }
        if let Err(rec_err) = gpu.record_event(scratch.inline_upload_done[slot_idx], stream) {
            if let Err(sync_err) = gpu.synchronize(stream) {
                self.scratch_poisoned.store(true, Ordering::Release);
                tracing::error!(
                    stream,
                    slot_idx,
                    record_err = %format!("{rec_err:#}"),
                    sync_err = %format!("{sync_err:#}"),
                    "Qwen4 PLE failed to fence or synchronize inline mmap staging DMA; shared scratch poisoned"
                );
                return Err(rec_err.context(format!(
                    "Qwen4 PLE inline mmap staging upload fence and stream sync both failed ({sync_err:#}); shared scratch poisoned"
                )));
            }
            return Err(rec_err
                .context("Qwen4 PLE inline mmap staging upload fence failed after stream sync"));
        }
        scratch.inline_upload_armed[slot_idx] = true;
        Ok(())
    }

    fn launch_inline_dequant_and_projections(
        &self,
        scratch: &Qwen4PleVerifyScratch,
        rows: usize,
        proj_route: PleCrossSeqProjRoute,
        overlapping: bool,
        gpu: &dyn GpuBackend,
        exec_stream: u64,
    ) -> Result<()> {
        let residual = self.residual_width();
        let embed_row = PLE_EMBED * 2;
        let key_row = residual * 2;
        spark_runtime::kernel_args::KernelLaunch::new(gpu, self.dequant_k)
            .grid([(PLE_EMBED as u32).div_ceil(256), rows as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(scratch.inline_dev)
            .arg_ptr(scratch.inline_dev.offset(PLE_INLINE_SCALE_BASE))
            .arg_ptr(scratch.embedding_rows)
            .launch(exec_stream)?;
        match proj_route {
            PleCrossSeqProjRoute::CublasBf16Gemm => {
                if overlapping {
                    let rows_u32 = rows as u32;
                    let embed_u32 = PLE_EMBED as u32;
                    let residual_u32 = residual as u32;
                    spark_runtime::cublaslt::gemm_act_weight_t_typed_pinned(
                        scratch.embedding_rows.0,
                        embed_u32,
                        self.key_proj.weight.0,
                        scratch.key_rows.0,
                        residual_u32,
                        rows_u32,
                        residual_u32,
                        embed_u32,
                        spark_runtime::cublaslt::GemmDtype::Bf16,
                        spark_runtime::cublaslt::GemmDtype::Bf16,
                        false,
                        rows_u32,
                        exec_stream,
                    )?;
                    spark_runtime::cublaslt::gemm_act_weight_t_typed_pinned(
                        scratch.embedding_rows.0,
                        embed_u32,
                        self.value_proj.weight.0,
                        scratch.value_rows.0,
                        embed_u32,
                        rows_u32,
                        embed_u32,
                        embed_u32,
                        spark_runtime::cublaslt::GemmDtype::Bf16,
                        spark_runtime::cublaslt::GemmDtype::Bf16,
                        false,
                        rows_u32,
                        exec_stream,
                    )?;
                } else {
                    ops::cublas_bf16_proj_dense(
                        scratch.embedding_rows,
                        &self.key_proj,
                        scratch.key_rows,
                        rows as u32,
                        residual as u32,
                        PLE_EMBED as u32,
                        exec_stream,
                    )?;
                    ops::cublas_bf16_proj_dense(
                        scratch.embedding_rows,
                        &self.value_proj,
                        scratch.value_rows,
                        rows as u32,
                        PLE_EMBED as u32,
                        PLE_EMBED as u32,
                        exec_stream,
                    )?;
                }
            }
            PleCrossSeqProjRoute::BatchedGemvTf => {
                ops::dense_gemv_batchn_tf(
                    gpu,
                    self.dense_gemv_batchn_tf_k,
                    scratch.embedding_rows,
                    &self.key_proj,
                    scratch.key_rows,
                    rows as u32,
                    residual as u32,
                    PLE_EMBED as u32,
                    exec_stream,
                )?;
                ops::dense_gemv_batchn_tf(
                    gpu,
                    self.dense_gemv_batchn_tf_k,
                    scratch.embedding_rows,
                    &self.value_proj,
                    scratch.value_rows,
                    rows as u32,
                    PLE_EMBED as u32,
                    PLE_EMBED as u32,
                    exec_stream,
                )?;
            }
            PleCrossSeqProjRoute::PerRowGemv => {
                for row in 0..rows {
                    self.launch_projections(
                        scratch.embedding_rows.offset(row * embed_row),
                        scratch.key_rows.offset(row * key_row),
                        scratch.value_rows.offset(row * embed_row),
                        gpu,
                        exec_stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Fork `dequant_k + key_proj + value_proj` onto `prep_stream` inside the
    /// unified 48-layer verify CUDA graph so PLE preparation overlaps concurrently
    /// with `batched_embed + Layer 0` on `main_stream`.
    ///
    /// When this returns `Ok(true)`, the caller MUST call either
    /// [`Self::launch_verify_fuse_in_graph`] (with `prep_forked = true`) or
    /// [`Self::join_verify_prep_in_graph`] (on error unwind) before ending graph capture.
    pub fn launch_verify_prep_fork_in_graph(
        &self,
        rows: usize,
        gpu: &dyn GpuBackend,
        main_stream: u64,
        prep_stream: u64,
    ) -> Result<bool> {
        ensure!(
            prep_stream != 0 && prep_stream != main_stream,
            "Qwen4 PLE in-graph overlap requires a distinct side stream"
        );
        let proj_route = self.unified_verify_proj_route(rows);
        let slot = self.verify.lock();
        let scratch = slot
            .as_ref()
            .context("Qwen4 PLE verify scratch was not initialized")?;
        gpu.record_event(scratch.prep_fork, main_stream)?;
        gpu.stream_wait_event(prep_stream, scratch.prep_fork)?;
        let launch_res = self.launch_inline_dequant_and_projections(
            scratch,
            rows,
            proj_route,
            true,
            gpu,
            prep_stream,
        );
        let rec_res = gpu.record_event(scratch.prep_done, prep_stream);
        if let Err(err) = launch_res {
            if rec_res.is_ok() {
                let _ = gpu.stream_wait_event(main_stream, scratch.prep_done);
            }
            return Err(err);
        }
        rec_res?;
        Ok(true)
    }

    /// Join a previously forked in-graph `prep_stream` back onto `main_stream`
    /// (used on Layer-0 error unwind so `CaptureGuard` never aborts with an unjoined branch).
    pub fn join_verify_prep_in_graph(&self, gpu: &dyn GpuBackend, main_stream: u64) -> Result<()> {
        let slot = self.verify.lock();
        let scratch = slot
            .as_ref()
            .context("Qwen4 PLE verify scratch was not initialized")?;
        gpu.stream_wait_event(main_stream, scratch.prep_done)
    }

    /// Execute the in-graph PLE fuse stage between Layer 0 and Layer 1 inside a
    /// unified 48-layer C=1 verify CUDA graph.
    ///
    /// If `prep_forked` is true, joins `prep_done` onto `stream` before fusing;
    /// otherwise runs `dequant_k + key_proj + value_proj` inline on `stream` first.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_verify_fuse_in_graph(
        &self,
        rows: usize,
        hyper: DevicePtr,
        row_stride: usize,
        slot_idx: usize,
        prep_forked: bool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let proj_route = self.unified_verify_proj_route(rows);
        let slot = self.verify.lock();
        let scratch = slot
            .as_ref()
            .context("Qwen4 PLE verify scratch was not initialized")?;
        if prep_forked {
            gpu.stream_wait_event(stream, scratch.prep_done)?;
        } else {
            self.launch_inline_dequant_and_projections(
                scratch, rows, proj_route, false, gpu, stream,
            )?;
        }
        let residual = self.residual_width();
        let embed_row = PLE_EMBED * 2;
        let key_row = residual * 2;
        if self.fuse_verify_rows_k.0 != 0
            && row_stride.is_multiple_of(2)
            && fuse_verify_rows_selected()
        {
            self.launch_fuse_verify_rows(
                hyper,
                (row_stride / 2) as u32,
                scratch.key_rows,
                scratch.value_rows,
                slot_idx,
                rows as u32,
                gpu,
                stream,
            )?;
        } else {
            for row in 0..rows {
                self.launch_fuse(
                    hyper.offset(row * row_stride),
                    scratch.key_rows.offset(row * key_row),
                    scratch.value_rows.offset(row * embed_row),
                    slot_idx,
                    false,
                    gpu,
                    stream,
                )?;
                self.save_intermediate(slot_idx, row, gpu, stream)?;
            }
        }
        static UNIFIED_LOGGED: std::sync::Once = std::sync::Once::new();
        UNIFIED_LOGGED.call_once(|| {
            tracing::info!(
                selector = PLE_UNIFIED_GRAPH_ENV,
                rows,
                overlap = prep_forked,
                proj_route = ?proj_route,
                "QWEN4_PLE_UNIFIED_48L_VERIFY_GRAPH_ENGAGED"
            );
        });
        Ok(())
    }

    /// Record `self.scratch_event` strictly OUTSIDE CUDA graph capture after a
    /// unified verify graph has been submitted on `stream`, keeping any subsequent
    /// non-graph PLE calls cleanly ordered after the graph's scratch usage.
    pub fn note_verify_graph_submitted(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        self.record_scratch_fence(gpu, stream)
    }
}

#[derive(Clone, Copy)]
struct StagingUpload {
    record_bytes: usize,
    scale_base: usize,
    scale_upload_bytes: usize,
}

/// True when every sequence has the same non-zero verify width, fits in the
/// 8-bit inline `packed_slots` encoding (`slot_idx < 256`), and no two
/// sequences alias the same `slot_idx` (preventing concurrent GPU state races).
fn is_uniform_small_slot(seqs: &[(&[u32], &[u32], usize)]) -> bool {
    let Some(&(first_tokens, _, _)) = seqs.first() else {
        return false;
    };
    let seq0_rows = first_tokens.len();
    if seq0_rows == 0 {
        return false;
    }
    let mut seen = [false; 256];
    for &(tokens, _, slot_idx) in seqs {
        if tokens.len() != seq0_rows || slot_idx >= 256 || seen[slot_idx] {
            return false;
        }
        seen[slot_idx] = true;
    }
    true
}

/// Pack one sequence's ragged PLE verify fuse metadata into a single `u32`
/// matching `qwen4_ple_fuse_verify_rows_ragged` in `kernels/gb10/common/qwen4_hyper.cu`:
/// - bits `0..=11`:  `row_offset` (`0..=4095`, `0xFFF`)
/// - bits `12..=15`: `rows` (`1..=15`, `0xF`)
/// - bits `16..=31`: `slot_idx` (`0..=65535`, `0xFFFF`)
#[inline]
#[must_use]
const fn encode_ragged_ple_fuse_desc(
    row_offset: usize,
    rows: usize,
    slot_idx: usize,
) -> Option<u32> {
    if row_offset > RAGGED_PLE_DESC_MAX_ROW_OFFSET
        || rows == 0
        || rows > RAGGED_PLE_DESC_MAX_ROWS
        || rows > PLE_MAX_SPEC_ROWS
        || slot_idx > u16::MAX as usize
    {
        return None;
    }
    Some((row_offset as u32) | ((rows as u32) << 12) | ((slot_idx as u32) << 16))
}

#[cfg(test)]
#[inline]
#[must_use]
const fn decode_ragged_ple_fuse_desc(desc: u32) -> (u32, u32, u32) {
    (desc & 0xFFF, (desc >> 12) & 0xF, desc >> 16)
}

/// Fixed-capacity stack table of packed per-sequence descriptors for
/// `qwen4_ple_fuse_verify_rows_ragged` (zero heap allocations on the verify hot path).
#[derive(Clone, Debug, PartialEq, Eq)]
struct RaggedPleFuseDescs {
    buf: [u32; PLE_MAX_BATCH_ROWS],
    len: usize,
}

impl RaggedPleFuseDescs {
    #[inline]
    fn as_slice(&self) -> &[u32] {
        &self.buf[..self.len]
    }

    #[inline]
    fn byte_len(&self) -> usize {
        self.len * RAGGED_DESC_BYTES
    }

    #[inline]
    fn write_le_bytes(&self, dst: &mut [u8]) {
        debug_assert_eq!(dst.len(), self.byte_len());
        for (index, &desc) in self.as_slice().iter().enumerate() {
            let start = index * RAGGED_DESC_BYTES;
            dst[start..start + RAGGED_DESC_BYTES].copy_from_slice(&desc.to_le_bytes());
        }
    }
}

/// Pack per-sequence `(row_offset, rows, slot_idx)` descriptors for a 1-launch
/// ragged PLE verify fuse. Returns `None` if no non-empty sequences exist, any
/// extent exceeds the kernel's bit-packed or scratch limits, or two non-empty
/// sequences alias the same `slot_idx` (preventing concurrent state races).
fn pack_ragged_ple_fuse_descs(seqs: &[(&[u32], &[u32], usize)]) -> Option<RaggedPleFuseDescs> {
    let mut buf = [0u32; PLE_MAX_BATCH_ROWS];
    let mut len = 0usize;
    let mut row_base = 0usize;
    for &(tokens, _, slot_idx) in seqs {
        let seq_rows = tokens.len();
        if seq_rows == 0 {
            continue;
        }
        if len >= PLE_MAX_BATCH_ROWS || row_base.checked_add(seq_rows)? > PLE_MAX_BATCH_ROWS {
            return None;
        }
        let desc = encode_ragged_ple_fuse_desc(row_base, seq_rows, slot_idx)?;
        let slot_u32 = slot_idx as u32;
        if buf[..len].iter().any(|&prior| (prior >> 16) == slot_u32) {
            return None;
        }
        buf[len] = desc;
        len += 1;
        row_base += seq_rows;
    }
    if len == 0 {
        return None;
    }
    Some(RaggedPleFuseDescs { buf, len })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PleVerifyFuseRoute {
    /// Uniform `C <= 8` (`slot_idx < 256`): 1 launch of `qwen4_ple_fuse_verify_rows`
    /// with inline `packed_slots: u64` (zero descriptor upload).
    UniformSingleLaunch,
    /// Ragged `C = 2..=256` or uniform `C = 9..=256` (or `slot_idx >= 256`):
    /// 1 launch of `qwen4_ple_fuse_verify_rows_ragged` with per-sequence
    /// descriptors piggybacked on the `scales_dev` pinned H2D upload.
    RaggedSingleLaunch,
    /// Fallback when `qwen4_ple_fuse_verify_rows_ragged` is absent on uniform
    /// `C > 8`: `ceil(C / 8)` launches of `qwen4_ple_fuse_verify_rows`.
    UniformChunkedFallback,
    /// Fallback when `qwen4_ple_fuse_verify_rows_ragged` is absent (or packing
    /// declined) on ragged sequences: 1 launch of `qwen4_ple_fuse_verify_rows`
    /// per sequence.
    PerSeqFallback,
    /// Legacy per-row `launch_fuse` + `save_intermediate` fallback.
    PerRowFallback,
}

#[inline]
#[must_use]
const fn select_ple_verify_fuse_route(
    fuse_rows_flag: bool,
    has_uniform_k: bool,
    has_ragged_k: bool,
    num_seqs: usize,
    all_uniform_small_slot: bool,
    has_ragged_descs: bool,
) -> PleVerifyFuseRoute {
    if !fuse_rows_flag {
        return PleVerifyFuseRoute::PerRowFallback;
    }
    if has_uniform_k && all_uniform_small_slot && num_seqs <= 8 {
        return PleVerifyFuseRoute::UniformSingleLaunch;
    }
    if has_ragged_k && has_ragged_descs {
        return PleVerifyFuseRoute::RaggedSingleLaunch;
    }
    if has_uniform_k && all_uniform_small_slot {
        return PleVerifyFuseRoute::UniformChunkedFallback;
    }
    if has_uniform_k {
        return PleVerifyFuseRoute::PerSeqFallback;
    }
    PleVerifyFuseRoute::PerRowFallback
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PleCrossSeqProjRoute {
    CublasBf16Gemm,
    BatchedGemvTf,
    PerRowGemv,
}

#[inline]
#[must_use]
const fn select_ple_cross_seq_proj_route_with_policy(
    batched_proj: bool,
    tc_dense_active: bool,
    has_batchn_tf: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> PleCrossSeqProjRoute {
    let effective_tc = !det_batch.is_enabled() && tc_dense_active;
    select_ple_cross_seq_proj_route(batched_proj, effective_tc, has_batchn_tf)
}

#[inline]
#[must_use]
const fn select_ple_cross_seq_proj_route(
    batched_proj: bool,
    tc_dense_active: bool,
    has_batchn_tf: bool,
) -> PleCrossSeqProjRoute {
    if batched_proj && tc_dense_active {
        PleCrossSeqProjRoute::CublasBf16Gemm
    } else if batched_proj && has_batchn_tf {
        PleCrossSeqProjRoute::BatchedGemvTf
    } else {
        PleCrossSeqProjRoute::PerRowGemv
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DraftSendOutcome, PLE_MAX_BATCH_ROWS, PLE_RECORD_BYTES, PleCrossSeqProjRoute,
        PleDraftAheadJob, PleVerifyFuseRoute, PleVerifyJob, PleWarmSeq, PleWarmTicket,
        PleWarmWorker, RAGGED_PLE_DESC_MAX_ROW_OFFSET, RAGGED_PLE_DESC_MAX_ROWS, RollingWindow,
        StackSelections, build_ple_warm_request, collect_draft_ahead_misses,
        decode_ragged_ple_fuse_desc, encode_ragged_ple_fuse_desc, is_uniform_small_slot,
        pack_ragged_ple_fuse_descs, select_ple_cross_seq_proj_route, select_ple_verify_fuse_route,
        selection_context, submit_verify_prefetch_with, submit_warm_verify_with, submit_warm_with,
        validate_active_decode_layout,
    };
    use crate::layers::qwen4_ple::Qwen4PleHasher;
    use spark_storage::ple_offload::{ParkedPleCache, PleNvfp4Row};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    fn test_hasher() -> Qwen4PleHasher {
        Qwen4PleHasher::new(
            [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071],
            [
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
                20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
                20_000_161, 20_000_171,
            ],
            [
                0,
                20_000_003,
                40_000_026,
                60_000_059,
                80_000_106,
                100_000_165,
                120_000_228,
                140_000_297,
                160_000_374,
                180_000_455,
                200_000_548,
                220_000_655,
                240_000_802,
                260_000_955,
                280_001_114,
                300_001_275,
            ],
            248_044,
            2_500_012,
        )
        .unwrap()
    }

    #[test]
    fn selection_context_keeps_at_most_the_last_two_tokens() {
        assert!(selection_context(&[]).is_empty());
        assert_eq!(selection_context(&[7]), vec![7]);
        assert_eq!(selection_context(&[1, 2, 3, 4]), vec![3, 4]);
    }

    #[test]
    fn ple_warm_seq_is_48_bytes_and_matches_full_slice_selection_context() {
        assert_eq!(std::mem::size_of::<PleWarmSeq>(), 48);
        assert!(PleWarmSeq::from_slices(&[1u32; 10], &[]).is_err());

        let hasher = test_hasher();
        let long_prior: Vec<u32> = (0u32..32_768).collect();
        let eos_prior = [11u32, 22, 33, 248_044];
        let cases: &[(&[u32], &[u32])] = &[
            (&[101, 102, 103, 104, 105], &[]),
            (&[55, 66], &[7]),
            (&[1, 2, 3, 4, 5, 6, 7, 8, 9], &[99, 100]),
            (&[500, 501, 502, 503, 504], &long_prior),
            (&[700, 701, 702], &eos_prior),
        ];

        let mut expected = Vec::new();
        let mut compact = Vec::with_capacity(cases.len());
        for &(draft, prior) in cases {
            let seq = PleWarmSeq::from_slices(draft, prior).unwrap();
            assert_eq!(seq.draft(), draft);
            assert_eq!(seq.prior(), selection_context(prior).as_slice());
            compact.push(seq);

            let mut context = selection_context(prior);
            for &token in draft {
                let selections = hasher.select_decode(token, &context);
                expected.extend(selections.iter().map(|s| (s.shard, s.row)));
                context.push(token);
            }
        }
        assert_eq!(build_ple_warm_request(&hasher, &compact), expected);
    }

    #[test]
    fn ple_warm_ticket_drop_drains_response_without_desync_and_worker_joins_cleanly() {
        let slot: parking_lot::Mutex<Option<PleWarmWorker>> = parking_lot::Mutex::new(None);
        let calls = Arc::new(AtomicUsize::new(0));
        let exited = Arc::new(AtomicBool::new(false));
        let spawn = || {
            let calls = Arc::clone(&calls);
            let exited = Arc::clone(&exited);
            struct ExitGuard(Arc<AtomicBool>);
            impl Drop for ExitGuard {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::Release);
                }
            }
            let guard = ExitGuard(exited);
            PleWarmWorker::spawn_with("ple-warm-test", move |seqs| {
                let _keep = &guard;
                let step = calls.fetch_add(1, Ordering::AcqRel) + 1;
                let tag = seqs
                    .first()
                    .and_then(|s| s.draft().first().copied())
                    .unwrap_or(0);
                if tag == 999 {
                    anyhow::bail!("injected io error at step {step}");
                }
                Ok(vec![PleNvfp4Row {
                    record: [0u8; PLE_RECORD_BYTES],
                    scale2: tag as f32,
                }])
            })
        };

        // 1. Submit request 1 and drop the ticket without calling wait():
        //    Drop must drain response 1 so request 2 does not read stale data.
        let seq1 = vec![PleWarmSeq::from_slices(&[11], &[1, 2]).unwrap()];
        let dropped_ticket = submit_warm_with(&slot, seq1, spawn).unwrap();
        drop(dropped_ticket);

        // 2. Submit request 2 and wait(): must observe tag 22, not 11.
        let seq2 = vec![PleWarmSeq::from_slices(&[22], &[3, 4]).unwrap()];
        let rows2 = submit_warm_with(&slot, seq2, spawn)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(rows2.len(), 1);
        assert_eq!(rows2[0].scale2, 22.0);

        // 3. Worker I/O error propagates cleanly without killing the worker.
        let seq_err = vec![PleWarmSeq::from_slices(&[999], &[]).unwrap()];
        assert!(
            submit_warm_with(&slot, seq_err, spawn)
                .unwrap()
                .wait()
                .is_err()
        );

        // 4. Subsequent request still uses the same healthy worker and succeeds.
        let seq3 = vec![PleWarmSeq::from_slices(&[33], &[5]).unwrap()];
        let rows3 = submit_warm_with(&slot, seq3, spawn)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(rows3.len(), 1);
        assert_eq!(rows3[0].scale2, 33.0);
        assert_eq!(calls.load(Ordering::Acquire), 4);

        // 5. Dropping the worker closes the channel and joins the thread.
        assert!(!exited.load(Ordering::Acquire));
        drop(slot.lock().take());
        assert!(exited.load(Ordering::Acquire));
    }

    #[test]
    fn active_decode_layout_admits_every_c2_c256_width_with_disjoint_slots() {
        for rows in 2..=crate::layers::QWEN4_NATIVE_MAX_ROWS {
            let slots: Vec<_> = (0..rows).collect();
            assert!(
                validate_active_decode_layout(
                    rows,
                    20_480,
                    10_240,
                    &slots,
                    crate::layers::QWEN4_NATIVE_MAX_ROWS
                )
                .is_ok(),
                "C{rows}"
            );
        }
    }

    #[test]
    fn active_decode_layout_rejects_padding_widths_and_state_aliases() {
        assert!(validate_active_decode_layout(1, 20_480, 10_240, &[0], 256).is_err());
        assert!(
            validate_active_decode_layout(257, 20_480, 10_240, &(0..257).collect::<Vec<_>>(), 512)
                .is_err()
        );
        assert!(validate_active_decode_layout(4, 20_480, 10_240, &[0, 1, 1, 2], 256).is_err());
        assert!(validate_active_decode_layout(2, 20_480, 10_240, &[0, 256], 256).is_err());
        assert!(validate_active_decode_layout(2, 10_240, 10_240, &[0, 1], 256).is_err());
    }

    #[test]
    fn ple_cross_seq_proj_preserves_gemv_below_4_and_uses_cublas_at_4_and_above() {
        use crate::layers::qwen4_hyper::{
            BatchedVerifyTcComponent, batched_verify_tc_component_admits,
        };
        for rows in [1, 2, 3] {
            let active = batched_verify_tc_component_admits(
                BatchedVerifyTcComponent::PleProj,
                true,
                None,
                rows,
                false,
            );
            assert_eq!(
                select_ple_cross_seq_proj_route(true, active, true),
                PleCrossSeqProjRoute::BatchedGemvTf,
                "rows={rows}"
            );
        }
        for rows in [4, 5, 6, 7, 8, 10, 16, 20, 40, 80, 128] {
            let active = batched_verify_tc_component_admits(
                BatchedVerifyTcComponent::PleProj,
                true,
                None,
                rows,
                false,
            );
            assert_eq!(
                select_ple_cross_seq_proj_route(true, active, true),
                PleCrossSeqProjRoute::CublasBf16Gemm,
                "rows={rows}"
            );
            // Disabled TC dense falls back cleanly to BatchedGemvTf.
            assert_eq!(
                select_ple_cross_seq_proj_route(true, false, true),
                PleCrossSeqProjRoute::BatchedGemvTf,
                "rows={rows} disabled"
            );
            // Disabled batched projections stay on PerRowGemv.
            assert_eq!(
                select_ple_cross_seq_proj_route(false, active, true),
                PleCrossSeqProjRoute::PerRowGemv,
                "rows={rows} unbatched"
            );
        }
        assert_eq!(
            select_ple_cross_seq_proj_route(true, false, false),
            PleCrossSeqProjRoute::PerRowGemv
        );
    }

    #[test]
    fn rolling_window_matches_selection_context_and_stack_selections_derefs() {
        let hasher = test_hasher();
        let cases: &[(&[u32], &[u32])] = &[
            (&[], &[10, 20, 30, 40]),
            (&[7], &[11, 248_044, 12, 13]),
            (&[1, 2, 3, 4], &[5, 6, 7]),
            (&[9, 248_044], &[88, 89]),
        ];
        for &(prior, tokens) in cases {
            let mut vec_ctx = selection_context(prior);
            let mut stack_ctx = RollingWindow::from_prior(prior);
            let mut expected = Vec::new();
            let mut stacked = StackSelections::<64>::new();
            for &token in tokens {
                let s1 = hasher.select_decode(token, &vec_ctx);
                let s2 = hasher.select_decode(token, stack_ctx.as_slice());
                assert_eq!(s1, s2, "prior={prior:?} token={token}");
                for sel in s1 {
                    expected.push((sel.shard, sel.row));
                    stacked.push((sel.shard, sel.row));
                }
                vec_ctx.push(token);
                stack_ctx.push(token);
            }
            let slice: &[(usize, usize)] = &stacked;
            assert_eq!(slice, expected.as_slice());
        }
    }

    #[test]
    fn draft_ahead_populates_parked_cache_without_desyncing_verify_and_immediate_ticket_works() {
        use std::sync::Barrier;

        let cache = Arc::new(ParkedPleCache::with_slots(
            1_024,
            vec![2_500_012; 128],
            vec![0.5; 128],
        ));
        let worker_cache = Arc::clone(&cache);
        let draft_done = Arc::new(AtomicBool::new(false));
        let worker_draft_done = Arc::clone(&draft_done);
        let draft_finished = Arc::new(Barrier::new(2));
        let slot: parking_lot::Mutex<Option<PleWarmWorker>> = parking_lot::Mutex::new(None);

        let spawn = || {
            let verify_cache = Arc::clone(&worker_cache);
            let draft_cache = Arc::clone(&worker_cache);
            let worker_draft_done = Arc::clone(&worker_draft_done);
            let draft_finished = Arc::clone(&draft_finished);
            PleWarmWorker::spawn_jobs(
                "ple-draft-ahead-test",
                move |job| match job {
                    PleVerifyJob::Request(request) => {
                        verify_cache.try_read_all(&request).and_then(|opt| {
                            opt.ok_or_else(|| anyhow::anyhow!("expected DraftAhead cache hit"))
                        })
                    }
                    PleVerifyJob::Seqs(_) => Ok(Vec::new()),
                },
                move |job, _verify_pending| {
                    if let PleDraftAheadJob::Selections(selections) = job {
                        for (shard, row) in selections {
                            let rec = [(shard as u8).wrapping_add(row as u8); PLE_RECORD_BYTES];
                            draft_cache.insert(shard, row, &rec);
                        }
                        worker_draft_done.store(true, Ordering::Release);
                        draft_finished.wait();
                    }
                },
            )
        };

        // 1. Immediate ticket serves empty/pre-populated results without touching the worker slot.
        let seeded = vec![PleNvfp4Row {
            record: [0x5Au8; PLE_RECORD_BYTES],
            scale2: 1.25,
        }];
        let immediate = PleWarmTicket::immediate(seeded).wait().expect("immediate");
        assert_eq!(immediate.len(), 1);
        assert_eq!(immediate[0].record, [0x5Au8; PLE_RECORD_BYTES]);
        assert!(
            slot.lock().is_none(),
            "immediate ticket must not spawn worker"
        );

        // 2. Enqueue a DraftAhead job while idle, let it warm ParkedPleCache,
        //    then verify on the same worker: DraftAhead sends no resp_tx item,
        //    so Verify receives its exact matching rows without desync.
        let targets = vec![(0usize, 10usize), (3, 77), (12, 999)];
        {
            let mut guard = slot.lock();
            *guard = Some(spawn().expect("spawn test worker"));
            assert_eq!(
                guard
                    .as_ref()
                    .expect("worker")
                    .try_send_draft_ahead(targets.clone()),
                DraftSendOutcome::Sent
            );
        }
        draft_finished.wait();
        let verified =
            submit_warm_verify_with(&slot, PleVerifyJob::Request(targets.clone()), spawn)
                .expect("submit verify")
                .wait()
                .expect("verify rows");
        assert!(draft_done.load(Ordering::Acquire));
        assert_eq!(verified.len(), targets.len());
        for (&(shard, row), got) in targets.iter().zip(&verified) {
            let expected = [(shard as u8).wrapping_add(row as u8); PLE_RECORD_BYTES];
            assert_eq!(got.record, expected);
            assert_eq!(cache.get(shard, row), Some(expected));
        }

        // Clean worker teardown.
        drop(slot.lock().take());
    }

    #[test]
    fn draft_ahead_seqs_matches_verify_bigram_trigram_hashes_100pct_and_skips_cached_rows() {
        let hasher = test_hasher();
        let cache = ParkedPleCache::with_slots(1_024, vec![2_500_012; 128], vec![0.5; 128]);
        let seqs: &[(&[u32], &[u32])] = &[
            (&[101, 102, 103, 104, 105], &[11, 12, 13]),
            (&[201, 202, 203, 204, 205], &[21, 22]),
            (&[301, 302, 303], &[31]),
        ];
        let warm_seqs: Vec<PleWarmSeq> = seqs
            .iter()
            .map(|&(window, prior)| PleWarmSeq::from_slices(window, prior).unwrap())
            .collect();
        let expected = build_ple_warm_request(&hasher, &warm_seqs);
        let misses = collect_draft_ahead_misses(&hasher, &cache, seqs, 128);
        assert_eq!(
            misses, expected,
            "per-sequence draft-ahead must match cross-seq verify selections 100%"
        );

        // Flat context-free hashing disagrees at sequence boundaries (row 0 = 16/16 heads,
        // row 1 = 8/16 trigram heads) because every PLE head is bigram or trigram.
        let flat: Vec<u32> = seqs
            .iter()
            .flat_map(|&(window, _)| window.iter().copied())
            .collect();
        let flat_misses = collect_draft_ahead_misses(&hasher, &cache, &[(&flat, &[])], 128);
        assert_ne!(flat_misses, expected);

        // Pre-populating row 0 of seq 0 skips cached entries lock-free.
        for &(shard, row) in &expected[..16] {
            cache.insert(shard, row, &[0x3Cu8; PLE_RECORD_BYTES]);
        }
        let expected_remaining: Vec<(usize, usize)> = expected
            .iter()
            .copied()
            .filter(|&(s, r)| !cache.contains(s, r))
            .collect();
        let remaining = collect_draft_ahead_misses(&hasher, &cache, seqs, 128);
        assert!(remaining.len() < expected.len());
        assert_eq!(remaining, expected_remaining);

        // Row cap stops collecting cleanly at `max_rows` (2 rows = 32 selections).
        let empty_cache = ParkedPleCache::with_slots(1_024, vec![2_500_012; 128], vec![0.5; 128]);
        let capped = collect_draft_ahead_misses(&hasher, &empty_cache, seqs, 2);
        assert_eq!(capped, expected[..32]);
    }

    #[test]
    fn verify_submit_is_nonblocking_and_preempts_in_flight_and_queued_draft_ahead() {
        use std::sync::Barrier;

        let cache = Arc::new(ParkedPleCache::with_slots(
            1_024,
            vec![2_500_012; 128],
            vec![0.5; 128],
        ));
        let started = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let observed_preemption = Arc::new(AtomicBool::new(false));
        let ran_superseded_draft = Arc::new(AtomicBool::new(false));
        let slot: parking_lot::Mutex<Option<PleWarmWorker>> = parking_lot::Mutex::new(None);
        // Reuse the collision-free keys validated in the test above.
        let key_a = (0usize, 10usize);
        let key_b = (3usize, 77usize);
        let key_c = (12usize, 999usize);

        let spawn = || {
            let verify_cache = Arc::clone(&cache);
            let draft_cache = Arc::clone(&cache);
            let started = Arc::clone(&started);
            let release = Arc::clone(&release);
            let observed_preemption = Arc::clone(&observed_preemption);
            let ran_superseded_draft = Arc::clone(&ran_superseded_draft);
            PleWarmWorker::spawn_jobs(
                "ple-draft-preempt-test",
                move |job| match job {
                    PleVerifyJob::Request(request) => {
                        let mut out = Vec::with_capacity(request.len());
                        for (shard, row) in request {
                            let record = verify_cache
                                .get(shard, row)
                                .unwrap_or([row as u8; PLE_RECORD_BYTES]);
                            out.push(PleNvfp4Row {
                                record,
                                scale2: 0.5,
                            });
                        }
                        Ok(out)
                    }
                    PleVerifyJob::Seqs(_) => Ok(Vec::new()),
                },
                move |job, verify_pending| {
                    if let PleDraftAheadJob::Selections(selections) = job {
                        if selections.as_slice() == [key_a, key_b] {
                            // First chunk populates key_a before simulating an in-flight read.
                            draft_cache.insert(
                                key_a.0,
                                key_a.1,
                                &[key_a.1 as u8; PLE_RECORD_BYTES],
                            );
                            started.wait();
                            release.wait();
                            // Must observe preemption before warming the remaining chunk (key_b).
                            if verify_pending.load(Ordering::Acquire) {
                                observed_preemption.store(true, Ordering::Release);
                                return;
                            }
                            draft_cache.insert(
                                key_b.0,
                                key_b.1,
                                &[key_b.1 as u8; PLE_RECORD_BYTES],
                            );
                        } else if selections.as_slice() == [key_c] {
                            ran_superseded_draft.store(true, Ordering::Release);
                        }
                    }
                },
            )
        };

        {
            let mut guard = slot.lock();
            *guard = Some(spawn().expect("spawn worker"));
            assert_eq!(
                guard
                    .as_ref()
                    .expect("worker")
                    .try_send_draft_ahead(vec![key_a, key_b]),
                DraftSendOutcome::Sent
            );
        }
        // Wait until worker is actively mid-flight inside DraftAhead #1.
        started.wait();

        // While DraftAhead #1 is mid-flight, a newer DraftAhead queues in the
        // latest-wins slot, and submitting foreground Verify returns immediately
        // BEFORE releasing DraftAhead #1 (proving zero main-thread blocking).
        {
            let guard = slot.lock();
            let worker = guard.as_ref().expect("worker");
            assert_eq!(
                worker.try_send_draft_ahead(vec![key_b]),
                DraftSendOutcome::Sent
            );
            assert_eq!(
                worker.try_send_draft_ahead(vec![key_c]),
                DraftSendOutcome::Sent
            );
        }
        let ticket =
            submit_warm_verify_with(&slot, PleVerifyJob::Request(vec![key_a, key_b]), spawn)
                .expect("foreground verify must enqueue without blocking behind DraftAhead");

        // Release the paused DraftAhead #1 chunk; it must observe verify_pending,
        // yield before key_b, skip superseded queued DraftAhead (key_c), and run Verify.
        release.wait();

        let rows = ticket.wait().expect("verify rows");
        assert!(
            observed_preemption.load(Ordering::Acquire),
            "in-flight DraftAhead must observe verify_pending preemption"
        );
        assert!(
            !ran_superseded_draft.load(Ordering::Acquire),
            "queued stale DraftAhead must be superseded when foreground Verify arrives"
        );
        assert!(
            cache.get(key_a.0, key_a.1).is_some(),
            "first DraftAhead chunk before preemption must remain cached"
        );
        assert!(
            cache.get(key_b.0, key_b.1).is_none(),
            "preempted DraftAhead tail must yield before warming remaining chunks"
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].record, [key_a.1 as u8; PLE_RECORD_BYTES]);
        assert_eq!(rows[1].record, [key_b.1 as u8; PLE_RECORD_BYTES]);

        // After ticket.wait() completes, verify_pending is cleared and a new DraftAhead is accepted.
        {
            let guard = slot.lock();
            assert_eq!(
                guard
                    .as_ref()
                    .expect("worker")
                    .try_send_draft_ahead(vec![key_b]),
                DraftSendOutcome::Sent
            );
        }
        drop(slot.lock().take());
    }

    #[test]
    fn ragged_ple_fuse_desc_packing_handles_mixed_widths_bounds_and_slot_aliases() {
        // 1. Mixed ragged widths [2, 5, 8, 1, 6] with non-contiguous slots (including > 255).
        let t0 = [1u32; 2];
        let t1 = [2u32; 5];
        let t2 = [3u32; 8];
        let t3 = [4u32; 1];
        let t4 = [5u32; 6];
        let seqs: &[(&[u32], &[u32], usize)] = &[
            (&t0, &[], 14),
            (&t1, &[], 3),
            (&t2, &[], 27),
            (&t3, &[], 0),
            (&t4, &[], 511),
        ];
        let descs = pack_ragged_ple_fuse_descs(seqs).expect("ragged widths must pack");
        assert_eq!(descs.as_slice().len(), 5);
        assert_eq!(descs.byte_len(), 20);
        let decoded: Vec<(u32, u32, u32)> = descs
            .as_slice()
            .iter()
            .copied()
            .map(decode_ragged_ple_fuse_desc)
            .collect();
        assert_eq!(
            decoded,
            vec![(0, 2, 14), (2, 5, 3), (7, 8, 27), (15, 1, 0), (16, 6, 511),]
        );
        let mut bytes = vec![0u8; descs.byte_len()];
        descs.write_le_bytes(&mut bytes);
        for (idx, chunk) in bytes.chunks_exact(4).enumerate() {
            let word = u32::from_le_bytes(chunk.try_into().expect("4-byte u32"));
            assert_eq!(word, descs.as_slice()[idx]);
        }

        // 2. Full C=16 ragged DFlash2 frame (80 total rows -> 16 descriptors = 64 bytes).
        let widths = [8usize, 5, 2, 8, 3, 6, 1, 8, 4, 7, 5, 2, 8, 6, 3, 4];
        let backing: Vec<Vec<u32>> = widths.iter().map(|&k| vec![42u32; k]).collect();
        let empty_prior: &[u32] = &[];
        let c16: Vec<(&[u32], &[u32], usize)> = backing
            .iter()
            .enumerate()
            .map(|(i, v)| (v.as_slice(), empty_prior, i * 2 + 1))
            .collect();
        let c16_descs = pack_ragged_ple_fuse_descs(&c16).expect("C=16 ragged must pack");
        assert_eq!(c16_descs.as_slice().len(), 16);
        assert_eq!(c16_descs.byte_len(), 64);
        let mut expected_off = 0u32;
        for (i, (&desc, &k)) in c16_descs.as_slice().iter().zip(&widths).enumerate() {
            assert_eq!(
                decode_ragged_ple_fuse_desc(desc),
                (expected_off, k as u32, (i * 2 + 1) as u32)
            );
            expected_off += k as u32;
        }
        assert_eq!(expected_off, 80);

        // 3. Empty sequences are skipped cleanly without shifting non-empty row offsets.
        let with_empty: &[(&[u32], &[u32], usize)] = &[
            (&[], &[], 9),
            (&t0, &[], 10),
            (&[], &[], 11),
            (&t3, &[], 12),
        ];
        let compact = pack_ragged_ple_fuse_descs(with_empty).expect("non-empty subset packs");
        assert_eq!(compact.as_slice().len(), 2);
        assert_eq!(
            decode_ragged_ple_fuse_desc(compact.as_slice()[0]),
            (0, 2, 10)
        );
        assert_eq!(
            decode_ragged_ple_fuse_desc(compact.as_slice()[1]),
            (2, 1, 12)
        );

        // 4. Reject all-empty, aliased slots, out-of-range widths, and total_rows > max.
        assert!(pack_ragged_ple_fuse_descs(&[(&[], &[], 0)]).is_none());
        assert!(
            pack_ragged_ple_fuse_descs(&[(&t0, &[], 5), (&t1, &[], 7), (&t3, &[], 5)]).is_none()
        );
        let too_wide = vec![1u32; RAGGED_PLE_DESC_MAX_ROWS + 1];
        assert!(pack_ragged_ple_fuse_descs(&[(&too_wide, &[], 0)]).is_none());
        let wide_max = vec![1u32; RAGGED_PLE_DESC_MAX_ROWS];
        let over_batch: Vec<(&[u32], &[u32], usize)> = (0..18)
            .map(|slot| (wide_max.as_slice(), empty_prior, slot))
            .collect();
        assert!(over_batch.iter().map(|(t, _, _)| t.len()).sum::<usize>() > PLE_MAX_BATCH_ROWS);
        assert!(pack_ragged_ple_fuse_descs(&over_batch).is_none());
        assert!(encode_ragged_ple_fuse_desc(0, 0, 0).is_none());
        assert!(encode_ragged_ple_fuse_desc(0, RAGGED_PLE_DESC_MAX_ROWS + 1, 0).is_none());
        assert!(encode_ragged_ple_fuse_desc(RAGGED_PLE_DESC_MAX_ROW_OFFSET + 1, 1, 0).is_none());
        assert!(encode_ragged_ple_fuse_desc(0, 1, usize::from(u16::MAX) + 1).is_none());
        assert_eq!(
            decode_ragged_ple_fuse_desc(
                encode_ragged_ple_fuse_desc(
                    RAGGED_PLE_DESC_MAX_ROW_OFFSET,
                    RAGGED_PLE_DESC_MAX_ROWS,
                    usize::from(u16::MAX),
                )
                .expect("max legal bounds")
            ),
            (
                RAGGED_PLE_DESC_MAX_ROW_OFFSET as u32,
                RAGGED_PLE_DESC_MAX_ROWS as u32,
                u32::from(u16::MAX)
            )
        );

        // 5. Full 256-row frames: C=32 x K=8 and C=64 x K=4 pack cleanly with zero bit collisions.
        let k8_tok = [7u32; 8];
        let c32_k8: Vec<(&[u32], &[u32], usize)> = (0..32)
            .map(|slot| (k8_tok.as_slice(), empty_prior, slot))
            .collect();
        let descs_c32 = pack_ragged_ple_fuse_descs(&c32_k8).expect("C=32 x K=8 = 256 rows");
        assert_eq!(descs_c32.as_slice().len(), 32);
        assert_eq!(
            decode_ragged_ple_fuse_desc(descs_c32.as_slice()[31]),
            (248, 8, 31)
        );

        let k4_tok = [9u32; 4];
        let c64_k4: Vec<(&[u32], &[u32], usize)> = (0..64)
            .map(|slot| (k4_tok.as_slice(), empty_prior, slot))
            .collect();
        let descs_c64 = pack_ragged_ple_fuse_descs(&c64_k4).expect("C=64 x K=4 = 256 rows");
        assert_eq!(descs_c64.as_slice().len(), 64);
        assert_eq!(
            decode_ragged_ple_fuse_desc(descs_c64.as_slice()[63]),
            (252, 4, 63)
        );
    }

    #[test]
    fn uniform_small_slot_rejects_duplicate_slot_indices() {
        let t = [1u32, 2, 3, 4];
        assert!(is_uniform_small_slot(&[(&t, &[], 0), (&t, &[], 7)]));
        assert!(
            !is_uniform_small_slot(&[(&t, &[], 5), (&t, &[], 7), (&t, &[], 5)]),
            "duplicate slot_idx must not enter multi-seq parallel fuse"
        );
        assert!(!is_uniform_small_slot(&[(&t, &[], 256)]));
        assert!(!is_uniform_small_slot(&[]));
        assert!(!is_uniform_small_slot(&[(&[], &[], 0)]));
    }

    #[test]
    fn ple_verify_fuse_route_selects_single_launch_for_ragged_and_c16_with_fallbacks() {
        // Uniform C <= 8 with small slots stays on inline u64 UniformSingleLaunch.
        for c in [1usize, 2, 4, 8] {
            assert_eq!(
                select_ple_verify_fuse_route(true, true, true, c, true, false),
                PleVerifyFuseRoute::UniformSingleLaunch,
                "uniform C={c}"
            );
        }
        // Ragged C=2..16 and uniform C=9..16 take 1-launch RaggedSingleLaunch.
        for c in [2usize, 4, 8, 9, 16] {
            assert_eq!(
                select_ple_verify_fuse_route(true, true, true, c, false, true),
                PleVerifyFuseRoute::RaggedSingleLaunch,
                "ragged C={c}"
            );
        }
        for c in [9usize, 12, 16] {
            assert_eq!(
                select_ple_verify_fuse_route(true, true, true, c, true, true),
                PleVerifyFuseRoute::RaggedSingleLaunch,
                "uniform C={c}"
            );
        }
        // Missing ragged kernel falls back cleanly to UniformChunkedFallback / PerSeqFallback.
        assert_eq!(
            select_ple_verify_fuse_route(true, true, false, 16, true, false),
            PleVerifyFuseRoute::UniformChunkedFallback
        );
        assert_eq!(
            select_ple_verify_fuse_route(true, true, false, 16, false, false),
            PleVerifyFuseRoute::PerSeqFallback
        );
        assert_eq!(
            select_ple_verify_fuse_route(true, true, true, 16, false, false),
            PleVerifyFuseRoute::PerSeqFallback
        );
        // Disabled flag or missing kernels fall back to PerRowFallback.
        assert_eq!(
            select_ple_verify_fuse_route(false, true, true, 16, false, true),
            PleVerifyFuseRoute::PerRowFallback
        );
        assert_eq!(
            select_ple_verify_fuse_route(true, false, false, 16, false, false),
            PleVerifyFuseRoute::PerRowFallback
        );
    }

    #[test]
    fn draft_ahead_chunk_fills_four_ring_ple_tiles() {
        assert_eq!(super::DRAFT_AHEAD_CHUNK_SELECTIONS, 4096);
        assert_eq!(
            super::DRAFT_AHEAD_CHUNK_SELECTIONS,
            PLE_MAX_BATCH_ROWS * super::QWEN4_PLE_HEADS
        );
        assert!(super::DRAFT_AHEAD_CHUNK_SELECTIONS.is_multiple_of(super::QWEN4_PLE_HEADS));
    }

    #[test]
    fn try_read_from_parked_or_recent_avoids_second_read_on_parked_cache_collision() {
        use std::collections::HashMap;

        // 1-slot direct-mapped ParkedPleCache guarantees (0, 10) and (1, 77) collide.
        let cache = ParkedPleCache::with_slots(1, vec![1_000, 1_000], vec![0.25, 1.5]);
        let rec0 = [0x11u8; PLE_RECORD_BYTES];
        let rec1 = [0x22u8; PLE_RECORD_BYTES];
        let rec2 = [0x33u8; PLE_RECORD_BYTES];

        // Simulate DraftAhead populating ParkedPleCache (second insert evicts first)
        // while retaining the flushed rows in `recent_draft`.
        cache.insert(0, 10, &rec0);
        cache.insert(1, 77, &rec1);
        assert!(
            cache.try_read_all(&[(0, 10), (1, 77)]).unwrap().is_none(),
            "1-slot direct-mapped cache must evict (0, 10) on collision"
        );

        let mut recent = HashMap::new();
        recent.insert(
            (0, 10),
            PleNvfp4Row {
                record: rec0,
                scale2: 0.25,
            },
        );
        // Only (0, 10) is in recent_draft; (1, 77) is served from ParkedPleCache.
        let rows = super::try_read_from_parked_or_recent(&cache, &mut recent, &[(0, 10), (1, 77)])
            .expect("valid request")
            .expect("combined parked + recent_draft must hit 100% without a second NVMe read");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].record, rec0);
        assert_eq!(rows[0].scale2.to_bits(), 0.25f32.to_bits());
        assert_eq!(rows[1].record, rec1);
        assert_eq!(rows[1].scale2.to_bits(), 1.5f32.to_bits());
        assert!(
            recent.is_empty(),
            "recent_draft must clear after Verify consumes it"
        );

        // Partial miss falls back cleanly to storage and clears stale recent_draft.
        recent.insert(
            (0, 10),
            PleNvfp4Row {
                record: rec2,
                scale2: 0.25,
            },
        );
        assert!(
            super::try_read_from_parked_or_recent(&cache, &mut recent, &[(0, 10), (0, 999)])
                .expect("valid request")
                .is_none()
        );
        assert!(recent.is_empty());
    }

    #[test]
    fn verify_prefetch_reuses_exact_key_and_drains_mismatches_without_desync() {
        use std::sync::Barrier;

        // 1. Validate PleWarmSeq::pack_batch bounds and exact packing.
        assert!(PleWarmSeq::pack_batch(&[]).is_none());
        assert!(PleWarmSeq::pack_batch(&[(&[], &[1])]).is_none());
        assert!(PleWarmSeq::pack_batch(&[(&[1u32; 10], &[])]).is_none());
        let at_cap: Vec<(&[u32], &[u32])> = vec![(&[1u32; 8], &[]); 32];
        assert!(PleWarmSeq::pack_batch(&at_cap).is_some());
        let over_cap: Vec<(&[u32], &[u32])> = vec![(&[1u32; 8], &[]); 33];
        assert!(PleWarmSeq::pack_batch(&over_cap).is_none());
        let packed = PleWarmSeq::pack_batch(&[(&[11, 12], &[1, 2, 3]), (&[21], &[4])])
            .expect("valid cross-seq prefetch batch");
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0].draft(), &[11, 12]);
        assert_eq!(packed[0].prior(), &[2, 3]);
        assert_eq!(packed[1].draft(), &[21]);
        assert_eq!(packed[1].prior(), &[4]);

        let slot: parking_lot::Mutex<Option<PleWarmWorker>> = parking_lot::Mutex::new(None);
        let calls = Arc::new(AtomicUsize::new(0));
        let in_flight_started = Arc::new(Barrier::new(2));
        let in_flight_release = Arc::new(Barrier::new(2));
        let spawn = || {
            let calls = Arc::clone(&calls);
            let started = Arc::clone(&in_flight_started);
            let release = Arc::clone(&in_flight_release);
            PleWarmWorker::spawn_with("ple-verify-prefetch-test", move |seqs| {
                calls.fetch_add(1, Ordering::AcqRel);
                let tag = seqs
                    .first()
                    .and_then(|s| s.draft().first().copied())
                    .unwrap_or(0);
                if tag == 500 {
                    started.wait();
                    release.wait();
                }
                Ok(vec![PleNvfp4Row {
                    record: [tag as u8; PLE_RECORD_BYTES],
                    scale2: tag as f32,
                }])
            })
        };

        // 2. Exact key match adopts the pre-submitted prefetch with zero duplicate work.
        let key_a = vec![
            PleWarmSeq::from_slices(&[11, 12], &[1, 2]).unwrap(),
            PleWarmSeq::from_slices(&[21, 22], &[3, 4]).unwrap(),
        ];
        assert_eq!(
            submit_verify_prefetch_with(&slot, key_a.clone(), spawn),
            DraftSendOutcome::Sent
        );
        // Duplicate prefetch for the identical key is a zero-cost no-op.
        assert_eq!(
            submit_verify_prefetch_with(&slot, key_a.clone(), spawn),
            DraftSendOutcome::Sent
        );
        let rows_a = submit_warm_with(&slot, key_a, spawn)
            .expect("adopt prefetch")
            .wait()
            .expect("prefetched rows");
        assert_eq!(rows_a.len(), 1);
        assert_eq!(rows_a[0].scale2, 11.0);
        assert_eq!(
            calls.load(Ordering::Acquire),
            1,
            "exact-key prefetch match must invoke worker once"
        );

        // 3. Dropping an adopted prefetch ticket drains cleanly without desyncing the next verify.
        let key_b = vec![PleWarmSeq::from_slices(&[22], &[5]).unwrap()];
        assert_eq!(
            submit_verify_prefetch_with(&slot, key_b.clone(), spawn),
            DraftSendOutcome::Sent
        );
        let dropped = submit_warm_with(&slot, key_b, spawn).expect("adopt key_b");
        drop(dropped);
        let key_c = vec![PleWarmSeq::from_slices(&[33], &[6]).unwrap()];
        let rows_c = submit_warm_with(&slot, key_c, spawn)
            .expect("verify key_c")
            .wait()
            .expect("rows_c");
        assert_eq!(rows_c[0].scale2, 33.0);

        // 4. Key mismatch (different prior or tokens) never returns stale rows.
        let key_stale = vec![PleWarmSeq::from_slices(&[44, 45], &[7, 8]).unwrap()];
        let key_diff_prior = vec![PleWarmSeq::from_slices(&[44, 45], &[7, 9]).unwrap()];
        assert_ne!(key_stale, key_diff_prior);
        assert_eq!(
            submit_verify_prefetch_with(&slot, key_stale, spawn),
            DraftSendOutcome::Sent
        );
        let key_d = vec![PleWarmSeq::from_slices(&[55], &[9]).unwrap()];
        let rows_d = submit_warm_with(&slot, key_d, spawn)
            .expect("mismatched verify key_d")
            .wait()
            .expect("rows_d");
        assert_eq!(rows_d[0].scale2, 55.0);

        // 5. While a prefetch is actively in flight, another mismatched prefetch returns Busy
        //    without blocking, and a subsequent mismatched foreground verify drains and executes.
        let key_inflight = vec![PleWarmSeq::from_slices(&[500], &[10]).unwrap()];
        assert_eq!(
            submit_verify_prefetch_with(&slot, key_inflight, spawn),
            DraftSendOutcome::Sent
        );
        in_flight_started.wait();
        let key_other = vec![PleWarmSeq::from_slices(&[60], &[11]).unwrap()];
        assert_eq!(
            submit_verify_prefetch_with(&slot, key_other, spawn),
            DraftSendOutcome::Busy
        );
        in_flight_release.wait();
        let key_after = vec![PleWarmSeq::from_slices(&[66], &[12]).unwrap()];
        let rows_after = submit_warm_with(&slot, key_after, spawn)
            .expect("verify after in-flight mismatch")
            .wait()
            .expect("rows_after");
        assert_eq!(rows_after[0].scale2, 66.0);

        // 6. Teardown with an unread prefetch joins cleanly without deadlock.
        let key_unread = vec![PleWarmSeq::from_slices(&[77], &[13]).unwrap()];
        assert_eq!(
            submit_verify_prefetch_with(&slot, key_unread, spawn),
            DraftSendOutcome::Sent
        );
        drop(slot.lock().take());
    }

    #[test]
    fn ple_verify_overlap_auto_enables_only_above_64_sms_and_honors_overrides() {
        use super::parse_ple_verify_overlap_for_sm;

        // GB10 (<= 64 SMs): default OFF (single-stream behavior unchanged).
        for gb10_sms in [0u32, 48, 64] {
            assert!(!parse_ple_verify_overlap_for_sm(None, gb10_sms));
            assert!(!parse_ple_verify_overlap_for_sm(Some(""), gb10_sms));
            assert!(!parse_ple_verify_overlap_for_sm(Some("auto"), gb10_sms));
            assert!(parse_ple_verify_overlap_for_sm(Some("1"), gb10_sms));
            assert!(parse_ple_verify_overlap_for_sm(Some("true"), gb10_sms));
            assert!(!parse_ple_verify_overlap_for_sm(Some("0"), gb10_sms));
        }
        // SM120 / SM103 (> 64 SMs): auto-ON, overridable via =0/off/false.
        for sm120_sms in [65u32, 148, 188] {
            assert!(parse_ple_verify_overlap_for_sm(None, sm120_sms));
            assert!(parse_ple_verify_overlap_for_sm(Some(""), sm120_sms));
            assert!(parse_ple_verify_overlap_for_sm(Some("auto"), sm120_sms));
            assert!(!parse_ple_verify_overlap_for_sm(Some("0"), sm120_sms));
            assert!(!parse_ple_verify_overlap_for_sm(Some("off"), sm120_sms));
            assert!(!parse_ple_verify_overlap_for_sm(Some("false"), sm120_sms));
            assert!(parse_ple_verify_overlap_for_sm(Some("1"), sm120_sms));
        }
    }

    #[test]
    fn ple_unified_graph_env_defaults_on_and_inline_layout_is_16b_aligned() {
        use super::{
            PLE_INLINE_SCALE_BASE, PLE_INLINE_SLOT_BYTES, PLE_UNIFIED_INLINE_SLOTS,
            parse_ple_unified_graph_enabled,
        };

        assert!(parse_ple_unified_graph_enabled(None));
        assert!(parse_ple_unified_graph_enabled(Some("")));
        assert!(parse_ple_unified_graph_enabled(Some("1")));
        assert!(parse_ple_unified_graph_enabled(Some("auto")));
        assert!(!parse_ple_unified_graph_enabled(Some("0")));
        assert!(!parse_ple_unified_graph_enabled(Some("off")));
        assert!(!parse_ple_unified_graph_enabled(Some("FALSE")));
        const {
            assert!(PLE_INLINE_SCALE_BASE.is_multiple_of(16));
            assert!(PLE_INLINE_SLOT_BYTES.is_multiple_of(16));
            assert!(PLE_UNIFIED_INLINE_SLOTS >= 2);
        }
    }

    #[test]
    fn deterministic_batch_disables_ple_cublas_flip_across_1_to_128() {
        use super::select_ple_cross_seq_proj_route_with_policy;
        use crate::layers::deterministic_batch::DeterministicBatchPolicy;

        for _m in [1usize, 2, 4, 8, 16, 32, 64, 128] {
            assert_eq!(
                select_ple_cross_seq_proj_route_with_policy(
                    true,
                    true,
                    true,
                    DeterministicBatchPolicy::Enabled,
                ),
                PleCrossSeqProjRoute::BatchedGemvTf
            );
            assert_eq!(
                select_ple_cross_seq_proj_route_with_policy(
                    true,
                    true,
                    false,
                    DeterministicBatchPolicy::Enabled,
                ),
                PleCrossSeqProjRoute::PerRowGemv
            );
        }
        assert_eq!(
            select_ple_cross_seq_proj_route_with_policy(
                true,
                true,
                true,
                DeterministicBatchPolicy::Disabled,
            ),
            PleCrossSeqProjRoute::CublasBf16Gemm
        );
    }
}
