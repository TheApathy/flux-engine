// SPDX-License-Identifier: AGPL-3.0-only

//! Depth-aware KV admission: reserve decode room, not just the prompt.
//!
//! Batch-slot admission alone can overcommit the paged-KV pool when many
//! requests have a substantial decode budget. This gate reserves
//! `prompt + min(max_tokens, watermark)` tokens for each selected request,
//! together with the same remaining commitment for every in-flight request.
//! Requests that do not fit are returned to the front of the pending queue.
//!
//! The default watermark is the served context ceiling, which makes each
//! request's own `max_tokens` the conservative bound. Operators may set
//! `ATLAS_KV_ADMIT_WATERMARK` lower to permit deliberate overcommit; `0`
//! restores legacy prompt-only reservation and emits a warning.

use super::*;

/// Resolve the admission watermark once at scheduler start.
pub(super) fn resolve_admit_watermark(max_seq_len: usize) -> usize {
    let default = if max_seq_len > 0 {
        max_seq_len
    } else {
        usize::MAX
    };
    match std::env::var("ATLAS_KV_ADMIT_WATERMARK") {
        Err(_) => default,
        Ok(v) => match v.parse::<usize>() {
            Ok(w) => {
                if w < default {
                    tracing::warn!(
                        "ATLAS_KV_ADMIT_WATERMARK={w} < the honest bound ({default}): \
                         admission may overcommit the KV pool; 0 restores legacy \
                         prompt-only reservation"
                    );
                }
                w
            }
            Err(_) => {
                tracing::warn!(
                    "ATLAS_KV_ADMIT_WATERMARK={v:?} is not an integer; using default {default}"
                );
                default
            }
        },
    }
}

/// Draft rows one verify frame may write past a request's emitted depth.
///
/// Every active speculative proposer (native MTP, DFlash2, self-spec, n-gram)
/// writes up to `num_drafts` temporary KV rows beyond `seq_len + 1` before any
/// rejected suffix is rolled back, so admission reserves `num_drafts` whenever
/// speculation is enabled (`num_drafts > 0`; plain decode passes `0`).
pub(super) fn resolve_verify_lookahead_tokens(
    _serial_k5_gate: bool,
    _native_mtp: bool,
    num_drafts: usize,
) -> usize {
    num_drafts
}

/// Match the allocator's legacy block-count shape exactly.
pub(super) fn blocks_for_tokens(tokens: usize, block_size: usize) -> usize {
    (tokens / block_size.max(1)).saturating_add(1)
}

/// One in-flight sequence's current depth and remaining generation budget.
pub(super) struct SeqDemand {
    pub current_tokens: usize,
    pub budget_tokens: usize,
}

/// Blocks committed by one sequence after watermark and context clamping.
pub(super) fn seq_commitment_blocks(
    demand: &SeqDemand,
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> usize {
    seq_commitment_blocks_with_lookahead(demand, watermark, max_seq_len, block_size, 0)
}

/// [`seq_commitment_blocks`] plus `lookahead` verify rows past the clamped
/// depth. A frame near the ceiling still writes its draft rows before
/// rollback, so the lookahead is added after the ceiling clamp.
pub(super) fn seq_commitment_blocks_with_lookahead(
    demand: &SeqDemand,
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
    lookahead: usize,
) -> usize {
    let mut depth = demand
        .current_tokens
        .saturating_add(demand.budget_tokens.min(watermark));
    if max_seq_len > 0 {
        // Never reserve beyond the served ceiling, but do not pretend an
        // already-materialized over-ceiling depth is smaller than it is.
        depth = depth.min(max_seq_len.max(demand.current_tokens));
    }
    blocks_for_tokens(depth.saturating_add(lookahead), block_size)
}

/// Total reserved blocks for everything already in flight or parked.
pub(super) fn committed_blocks(
    demands: &[SeqDemand],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> usize {
    committed_blocks_with_lookahead(demands, watermark, max_seq_len, block_size, 0)
}

pub(super) fn committed_blocks_with_lookahead(
    demands: &[SeqDemand],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
    lookahead: usize,
) -> usize {
    demands.iter().fold(0usize, |total, demand| {
        total.saturating_add(seq_commitment_blocks_with_lookahead(
            demand,
            watermark,
            max_seq_len,
            block_size,
            lookahead,
        ))
    })
}

/// Number of requests in the selected prefix that fit the remaining pool.
///
/// Returns `(admit_count, forced_oversize)`. Selection stops at the first
/// request that does not fit, so later requests cannot jump that request.
/// If the pool is otherwise idle, a lone oversize request is admitted so it
/// can fail or back-pressure in the existing allocation path instead of
/// remaining queued forever.
pub(super) fn admit_count(
    total_blocks: usize,
    committed: usize,
    reqs: &[(usize, usize)],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> (usize, bool) {
    admit_count_with_lookahead(
        total_blocks,
        committed,
        reqs,
        watermark,
        max_seq_len,
        block_size,
        0,
    )
}

pub(super) fn admit_count_with_lookahead(
    total_blocks: usize,
    committed: usize,
    reqs: &[(usize, usize)],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
    lookahead: usize,
) -> (usize, bool) {
    let mut used = committed;
    let mut count = 0usize;
    for &(prompt_tokens, max_tokens) in reqs {
        let need = seq_commitment_blocks_with_lookahead(
            &SeqDemand {
                current_tokens: prompt_tokens,
                budget_tokens: max_tokens,
            },
            watermark,
            max_seq_len,
            block_size,
            lookahead,
        );
        match used.checked_add(need) {
            Some(next) if next <= total_blocks => {
                used = next;
                count += 1;
            }
            _ if count == 0 && committed == 0 => return (1, true),
            _ => break,
        }
    }
    (count, false)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PendingEnqueueKey {
    prompt_ptr: usize,
    prompt_len: usize,
    session_hash: u64,
    cancel_ptr: usize,
}

fn pending_enqueue_key(req: &InferenceRequest) -> PendingEnqueueKey {
    match req {
        InferenceRequest::Blocking {
            prompt_tokens,
            session_hash,
            ..
        } => PendingEnqueueKey {
            prompt_ptr: prompt_tokens.as_ptr() as usize,
            prompt_len: prompt_tokens.len(),
            session_hash: *session_hash,
            cancel_ptr: 0,
        },
        InferenceRequest::Streaming {
            prompt_tokens,
            session_hash,
            cancel_flag,
            ..
        } => PendingEnqueueKey {
            prompt_ptr: prompt_tokens.as_ptr() as usize,
            prompt_len: prompt_tokens.len(),
            session_hash: *session_hash,
            cancel_ptr: std::sync::Arc::as_ptr(cancel_flag) as usize,
        },
    }
}

/// Cap on the enqueue-timestamp sidecar map so dropped or unwound queues can
/// never accumulate unbounded entries.
const PENDING_ENQUEUE_MAP_CAP: usize = 8_192;

fn pending_enqueue_times() -> &'static Mutex<std::collections::HashMap<PendingEnqueueKey, Instant>>
{
    static TIMES: std::sync::OnceLock<
        Mutex<std::collections::HashMap<PendingEnqueueKey, Instant>>,
    > = std::sync::OnceLock::new();
    TIMES.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Record when a queued request entered `PendingQueue` without changing
/// `PendingQueue`'s struct layout.
pub(super) fn record_pending_enqueued(req: &InferenceRequest, enqueued_at: Instant) {
    let key = pending_enqueue_key(req);
    let mut map = pending_enqueue_times().lock();
    if map.len() >= PENDING_ENQUEUE_MAP_CAP && !map.contains_key(&key) {
        let mut entries: Vec<(PendingEnqueueKey, Instant)> =
            map.iter().map(|(&k, &t)| (k, t)).collect();
        entries.sort_unstable_by_key(|&(_, t)| t);
        for (stale_key, _) in entries.into_iter().take(map.len() / 2) {
            map.remove(&stale_key);
        }
    }
    map.insert(key, enqueued_at);
}

fn lookup_pending_enqueued(req: &InferenceRequest) -> Option<Instant> {
    let key = pending_enqueue_key(req);
    pending_enqueue_times().lock().get(&key).copied()
}

fn take_pending_enqueued(req: &InferenceRequest) -> Option<Instant> {
    let key = pending_enqueue_key(req);
    pending_enqueue_times().lock().remove(&key)
}

fn parse_queue_timeout(raw: Option<&str>) -> Option<std::time::Duration> {
    let raw = raw?.trim();
    if raw.is_empty() {
        return None;
    }
    match raw.parse::<u64>() {
        Ok(0) => None,
        Ok(secs) => Some(std::time::Duration::from_secs(secs)),
        Err(_) => {
            tracing::warn!(
                "FLUX_QUEUE_TIMEOUT={raw:?} is not a nonnegative integer; using 0 (disabled)"
            );
            None
        }
    }
}

/// Optional queue-wait cap for server-default long-horizon requests
/// (`FLUX_QUEUE_TIMEOUT`, seconds; 0 or unset disables artificial queue expiry
/// while the client remains connected).
fn queue_timeout_from_env() -> Option<std::time::Duration> {
    static TIMEOUT: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *TIMEOUT
        .get_or_init(|| parse_queue_timeout(std::env::var("FLUX_QUEUE_TIMEOUT").ok().as_deref()))
}

fn shift_request_deadlines_for_wait(
    req: &mut InferenceRequest,
    enqueued_at: Option<Instant>,
    now: Instant,
) {
    let Some(enqueued_at) = enqueued_at else {
        return;
    };
    let wait = now.saturating_duration_since(enqueued_at);
    if wait.is_zero() {
        return;
    }
    match req {
        InferenceRequest::Blocking {
            timeout_at,
            decode_timeout_at,
            ..
        }
        | InferenceRequest::Streaming {
            timeout_at,
            decode_timeout_at,
            ..
        } => {
            if let Some(decode_deadline) = *decode_timeout_at {
                if let Some(base_deadline) = *timeout_at
                    && let Some(shifted) = base_deadline.checked_add(wait)
                {
                    *timeout_at = Some(shifted);
                }
                if let Some(shifted) = decode_deadline.checked_add(wait) {
                    *decode_timeout_at = Some(shifted);
                }
            }
        }
    }
}

fn shift_admitted_deadlines(admitted: &mut [InferenceRequest], now: Instant) {
    for req in admitted {
        let enqueued_at = take_pending_enqueued(req);
        shift_request_deadlines_for_wait(req, enqueued_at, now);
    }
}

/// Extra KV block reserve held back when live active sequences have already
/// grown past `watermark` tokens, signaling that the static `watermark`
/// underestimates active growth and admitting additional queued requests risks
/// mid-decode `KV_POOL_EXHAUSTED`.
fn active_growth_reserve_blocks(
    active: &[ActiveSeq],
    watermark: usize,
    total_blocks: usize,
    block_size: usize,
) -> usize {
    if watermark == 0 || active.is_empty() || total_blocks == 0 {
        return 0;
    }
    let any_past_watermark = active
        .iter()
        .any(|a| a.seq.seq_len.saturating_add(1) > watermark);
    if !any_past_watermark {
        return 0;
    }
    let watermark_blocks = blocks_for_tokens(watermark, block_size);
    watermark_blocks.min(total_blocks / 4)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PendingPruneReason {
    Disconnected,
    Cancelled,
    Retiring,
    TimedOut,
}

fn pending_prune_reason_with(
    req: &InferenceRequest,
    enqueued_at: Option<Instant>,
    now: Instant,
    retiring: bool,
    queue_timeout: Option<std::time::Duration>,
) -> Option<PendingPruneReason> {
    let (sink_closed, cancelled, timeout_at, decode_timeout_at) = match req {
        InferenceRequest::Blocking {
            response_tx,
            timeout_at,
            decode_timeout_at,
            ..
        } => (
            response_tx.is_closed(),
            false,
            *timeout_at,
            *decode_timeout_at,
        ),
        InferenceRequest::Streaming {
            token_tx,
            cancel_flag,
            timeout_at,
            decode_timeout_at,
            ..
        } => (
            token_tx.is_closed(),
            cancel_flag.load(std::sync::atomic::Ordering::Acquire),
            *timeout_at,
            *decode_timeout_at,
        ),
    };
    if sink_closed {
        Some(PendingPruneReason::Disconnected)
    } else if cancelled {
        Some(PendingPruneReason::Cancelled)
    } else if retiring {
        Some(PendingPruneReason::Retiring)
    } else if decode_timeout_at.is_some() {
        // Server-default long-horizon request (client omitted `req.timeout`):
        // decouple queue wait from the execution deadline while connected.
        if let (Some(limit), Some(since)) = (queue_timeout, enqueued_at)
            && now.saturating_duration_since(since) >= limit
        {
            Some(PendingPruneReason::TimedOut)
        } else {
            None
        }
    } else if timeout_at.is_some_and(|deadline| now >= deadline) {
        Some(PendingPruneReason::TimedOut)
    } else {
        None
    }
}

fn notify_pruned_pending(req: InferenceRequest, reason: PendingPruneReason) {
    let msg = match reason {
        PendingPruneReason::Disconnected => {
            tracing::debug!("Pruned disconnected request from pending queue");
            return;
        }
        PendingPruneReason::Cancelled => {
            tracing::debug!("Pruned cancelled request from pending queue");
            "Request cancelled while queued"
        }
        PendingPruneReason::Retiring => {
            tracing::debug!("Pruned queued request during server shutdown");
            "Server shutting down; queued request was not started"
        }
        PendingPruneReason::TimedOut => {
            tracing::warn!("Request timeout: reason=total_deadline phase=pending_queue");
            "Request timed out while queued"
        }
    };
    match req {
        InferenceRequest::Blocking { response_tx, .. } => {
            let _ = response_tx.send(Err(anyhow::anyhow!("{msg}")));
        }
        InferenceRequest::Streaming { token_tx, .. } => {
            // Nonblocking: a full or closed stream must never park the scheduler thread.
            let _ = token_tx.try_send(StreamEvent::Error(msg.to_string()));
        }
    }
}

/// Remove queued requests whose sink is closed, whose cancel flag is set,
/// whose explicit deadline has expired (`timeout_at <= now` with
/// `decode_timeout_at == None`), whose optional `FLUX_QUEUE_TIMEOUT` elapsed,
/// or once shutdown retirement has begun. Live-but-expired/cancelled/retiring
/// sinks receive a terminal error before being dropped.
///
/// Preserves the FIFO order of surviving requests and returns the exact number
/// removed so callers can release `QUEUE_GATE` places and wake parked waiters.
pub(super) fn prune_dead_pending_requests(
    requests: &mut Vec<InferenceRequest>,
    now: Instant,
) -> usize {
    let retiring = super::retirement::requested();
    let queue_timeout = queue_timeout_from_env();
    if !requests.iter().any(|req| {
        pending_prune_reason_with(
            req,
            lookup_pending_enqueued(req),
            now,
            retiring,
            queue_timeout,
        )
        .is_some()
    }) {
        return 0;
    }
    let mut kept = Vec::with_capacity(requests.len());
    let mut pruned = 0usize;
    for req in requests.drain(..) {
        let enqueued_at = lookup_pending_enqueued(&req);
        if let Some(reason) =
            pending_prune_reason_with(&req, enqueued_at, now, retiring, queue_timeout)
        {
            let _ = take_pending_enqueued(&req);
            notify_pruned_pending(req, reason);
            pruned += 1;
        } else {
            kept.push(req);
        }
    }
    *requests = kept;
    pruned
}

/// Split this tick's selected requests into an admissible prefix and an
/// overflow tail, returning the tail to the front of the pending queue.
#[allow(clippy::too_many_arguments)]
pub(super) fn gate_admissions(
    model: &dyn Model,
    pending: &std::sync::Arc<(Mutex<PendingQueue>, Condvar)>,
    new_reqs: Vec<InferenceRequest>,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    swapped: &[SwappedSeq],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
    verify_lookahead: usize,
) -> Vec<InferenceRequest> {
    gate_admissions_with_reserved(
        model,
        pending,
        new_reqs,
        active,
        prefilling,
        swapped,
        &[],
        watermark,
        max_seq_len,
        block_size,
        verify_lookahead,
    )
}

/// Like [`gate_admissions`], additionally reserving `extra_reserved` demands
/// (such as already-admitted burst requests waiting in `start_new_requests`'s
/// local `req_queue` before their prefill starts).
#[allow(clippy::too_many_arguments)]
pub(super) fn gate_admissions_with_reserved(
    model: &dyn Model,
    pending: &std::sync::Arc<(Mutex<PendingQueue>, Condvar)>,
    new_reqs: Vec<InferenceRequest>,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    swapped: &[SwappedSeq],
    extra_reserved: &[SeqDemand],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
    verify_lookahead: usize,
) -> Vec<InferenceRequest> {
    let total_blocks = model.num_total_blocks();
    if total_blocks > 0 && block_size > 0 {
        crate::metrics::compat_metrics::set_cache_config(
            block_size,
            total_blocks,
            total_blocks.saturating_mul(block_size),
        );
        crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());
    }
    let mem_gate = if !crate::main_modules::mem_watch::is_refusing() {
        MemFloorGate::Allow
    } else {
        MemFloorGate::Refusing
    };
    let admitted = gate_admissions_with_blocks(
        model.num_admission_blocks(),
        pending,
        new_reqs,
        active,
        prefilling,
        swapped,
        extra_reserved,
        watermark,
        max_seq_len,
        block_size,
        verify_lookahead,
        mem_gate,
        &super::queue_gate::QUEUE_GATE,
    );
    if mem_gate == MemFloorGate::Refusing
        && admitted.is_empty()
        && active.is_empty()
        && prefilling.is_empty()
    {
        // Avoid hot-spinning an idle scheduler loop while queued requests wait
        // for host MemAvailable to recover (or for shutdown).
        let mut queue = pending.0.lock();
        if !queue.closed && !queue.requests.is_empty() {
            let _ = pending
                .1
                .wait_for(&mut queue, std::time::Duration::from_millis(25));
        }
    }
    admitted
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemFloorGate {
    Allow,
    Refusing,
}

#[allow(clippy::too_many_arguments)]
fn gate_admissions_with_blocks(
    total_blocks: usize,
    pending: &std::sync::Arc<(Mutex<PendingQueue>, Condvar)>,
    mut new_reqs: Vec<InferenceRequest>,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    swapped: &[SwappedSeq],
    extra_reserved: &[SeqDemand],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
    verify_lookahead: usize,
    mem_gate: MemFloorGate,
    gate: &super::queue_gate::QueueGate,
) -> Vec<InferenceRequest> {
    if new_reqs.is_empty() {
        return new_reqs;
    }
    let now = Instant::now();
    // Prune disconnected, cancelled, retiring, or expired requests BEFORE KV
    // demand accounting so a dead head-of-line request cannot block live
    // requests behind it or reserve KV capacity it will never use.
    let pruned = prune_dead_pending_requests(&mut new_reqs, now);
    if pruned > 0 {
        gate.leave(pruned);
        pending.1.notify_all();
        if new_reqs.is_empty() {
            crate::metrics::compat_metrics::set_scheduler_counts(
                active.len() + prefilling.len(),
                pending.0.lock().requests.len(),
            );
            return new_reqs;
        }
    }
    static MEM_FLOOR_PAUSED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    if mem_gate == MemFloorGate::Refusing {
        if !MEM_FLOOR_PAUSED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::warn!(
                queued = new_reqs.len(),
                active = active.len(),
                prefilling = prefilling.len(),
                "Memory floor breached: pausing PendingQueue prefill promotion until \
                 MemAvailable recovers while active sequences drain"
            );
        }
        let waiting = {
            let mut queue = pending.0.lock();
            for (index, req) in new_reqs.into_iter().enumerate() {
                queue.requests.insert(index, req);
            }
            queue.requests.len()
        };
        crate::metrics::compat_metrics::set_scheduler_counts(
            active.len() + prefilling.len(),
            waiting,
        );
        return Vec::new();
    }
    if MEM_FLOOR_PAUSED.swap(false, std::sync::atomic::Ordering::Relaxed) {
        tracing::info!("Memory floor recovered: resuming PendingQueue prefill promotion");
    }
    if total_blocks == 0 {
        // Backends without paged-KV occupancy information retain the prior
        // admission behavior.
        shift_admitted_deadlines(&mut new_reqs, now);
        gate.leave(new_reqs.len());
        crate::metrics::compat_metrics::set_scheduler_counts(
            active.len() + prefilling.len() + new_reqs.len(),
            pending.0.lock().requests.len(),
        );
        return new_reqs;
    }

    let mut demands =
        Vec::with_capacity(active.len() + prefilling.len() + swapped.len() + extra_reserved.len());
    demands.extend(active.iter().map(|a| SeqDemand {
        // `last_token` has been emitted but is still the next decode input.
        current_tokens: a.seq.seq_len + 1,
        budget_tokens: a.remaining,
    }));
    demands.extend(prefilling.iter().map(|p| SeqDemand {
        current_tokens: p.prompt_tokens.len(),
        budget_tokens: p.max_tokens,
    }));
    demands.extend(swapped.iter().map(|s| SeqDemand {
        current_tokens: s.seq_len + 1,
        budget_tokens: s.remaining,
    }));
    demands.extend(extra_reserved.iter().map(|r| SeqDemand {
        current_tokens: r.current_tokens,
        budget_tokens: r.budget_tokens,
    }));

    let committed = committed_blocks_with_lookahead(
        &demands,
        watermark,
        max_seq_len,
        block_size,
        verify_lookahead,
    );
    let growth_reserve = active_growth_reserve_blocks(active, watermark, total_blocks, block_size);
    let effective_committed = committed.saturating_add(growth_reserve);
    let infos: Vec<(usize, usize)> = new_reqs
        .iter()
        .map(|r| (r.prompt_len(), r.max_tokens()))
        .collect();
    let (admit, forced) = admit_count_with_lookahead(
        total_blocks,
        effective_committed,
        &infos,
        watermark,
        max_seq_len,
        block_size,
        verify_lookahead,
    );
    if forced {
        tracing::warn!(
            "admitting a request whose reservation exceeds the whole KV pool \
             ({} prompt + min({}, watermark {}) tokens vs {} blocks); the existing \
             allocation path will provide final back-pressure",
            infos[0].0,
            infos[0].1,
            watermark,
            total_blocks,
        );
    }

    static LAST_QUEUED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    // Everything admitted stops waiting (`--max-queue-depth`); overflow goes
    // back to the pending queue and keeps its place.
    if admit >= new_reqs.len() {
        LAST_QUEUED.store(0, std::sync::atomic::Ordering::Relaxed);
        shift_admitted_deadlines(&mut new_reqs, now);
        gate.leave(new_reqs.len());
        crate::metrics::compat_metrics::set_scheduler_counts(
            active.len() + prefilling.len() + new_reqs.len(),
            pending.0.lock().requests.len(),
        );
        return new_reqs;
    }

    let mut admitted = new_reqs;
    let overflow = admitted.split_off(admit);
    if LAST_QUEUED.swap(overflow.len(), std::sync::atomic::Ordering::Relaxed) != overflow.len() {
        tracing::info!(
            "KV admission: {} of {} selected request(s) fit ({} blocks committed + {} growth reserve of {}); \
             {} queued until capacity frees",
            admitted.len(),
            admitted.len() + overflow.len(),
            committed,
            growth_reserve,
            total_blocks,
            overflow.len(),
        );
    }
    let waiting = {
        let mut queue = pending.0.lock();
        for (index, req) in overflow.into_iter().enumerate() {
            queue.requests.insert(index, req);
        }
        queue.requests.len()
    };
    shift_admitted_deadlines(&mut admitted, now);
    gate.leave(admitted.len());
    crate::metrics::compat_metrics::set_scheduler_counts(
        active.len() + prefilling.len() + admitted.len(),
        waiting,
    );
    admitted
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK_SIZE: usize = 16;
    const POOL_BLOCKS: usize = 102_000 / BLOCK_SIZE;
    const PROMPT: usize = 202;
    const MAX_TOKENS: usize = 1024;
    const MAX_SEQ_LEN: usize = 8192;

    fn request_blocks() -> usize {
        blocks_for_tokens(PROMPT + MAX_TOKENS, BLOCK_SIZE)
    }

    #[test]
    fn block_math_matches_legacy_formula() {
        assert_eq!(blocks_for_tokens(0, 16), 1);
        assert_eq!(blocks_for_tokens(15, 16), 1);
        assert_eq!(blocks_for_tokens(16, 16), 2);
        assert_eq!(blocks_for_tokens(1226, 16), 77);
        assert_eq!(blocks_for_tokens(usize::MAX, 1), usize::MAX);
    }

    #[test]
    fn commitment_math_saturates_fail_closed() {
        let demands = vec![
            SeqDemand {
                current_tokens: usize::MAX,
                budget_tokens: usize::MAX,
            },
            SeqDemand {
                current_tokens: usize::MAX,
                budget_tokens: usize::MAX,
            },
        ];
        assert_eq!(committed_blocks(&demands, usize::MAX, 0, 1), usize::MAX);
        assert_eq!(
            admit_count(usize::MAX, usize::MAX, &[(1, 1)], usize::MAX, 0, 1),
            (0, false)
        );
    }

    #[test]
    fn watermark_and_context_ceiling_bound_commitment() {
        let demand = SeqDemand {
            current_tokens: 200,
            budget_tokens: 4096,
        };
        assert_eq!(
            seq_commitment_blocks(&demand, 512, MAX_SEQ_LEN, BLOCK_SIZE),
            blocks_for_tokens(712, BLOCK_SIZE)
        );
        assert_eq!(
            seq_commitment_blocks(&demand, usize::MAX, MAX_SEQ_LEN, BLOCK_SIZE),
            blocks_for_tokens(4296, BLOCK_SIZE)
        );
        assert_eq!(
            seq_commitment_blocks(&demand, 0, MAX_SEQ_LEN, BLOCK_SIZE),
            blocks_for_tokens(200, BLOCK_SIZE)
        );

        let long = SeqDemand {
            current_tokens: 8000,
            budget_tokens: 4096,
        };
        assert_eq!(
            seq_commitment_blocks(&long, usize::MAX, MAX_SEQ_LEN, BLOCK_SIZE),
            blocks_for_tokens(MAX_SEQ_LEN, BLOCK_SIZE)
        );
    }

    #[test]
    fn c64_fits_but_c128_queues_the_overflow() {
        let c64 = vec![(PROMPT, MAX_TOKENS); 64];
        assert_eq!(
            admit_count(POOL_BLOCKS, 0, &c64, MAX_SEQ_LEN, MAX_SEQ_LEN, BLOCK_SIZE,),
            (64, false)
        );

        let c128 = vec![(PROMPT, MAX_TOKENS); 128];
        let (count, forced) =
            admit_count(POOL_BLOCKS, 0, &c128, MAX_SEQ_LEN, MAX_SEQ_LEN, BLOCK_SIZE);
        assert_eq!(count, POOL_BLOCKS / request_blocks());
        assert!(count < 128);
        assert!(!forced);
        assert!(count * request_blocks() <= POOL_BLOCKS);
    }

    #[test]
    fn tight_pool_excludes_the_permanent_dummy_block() {
        let raw_total_including_dummy = 2usize;
        let usable = raw_total_including_dummy.saturating_sub(1);
        let requests = vec![(15, 0), (15, 0)];
        assert_eq!(admit_count(usable, 0, &requests, 0, 16, 16), (1, false));
    }

    #[test]
    fn in_flight_commitments_reduce_capacity() {
        let demands: Vec<SeqDemand> = (0..40)
            .map(|_| SeqDemand {
                current_tokens: 600,
                budget_tokens: 700,
            })
            .collect();
        let committed = committed_blocks(&demands, MAX_SEQ_LEN, MAX_SEQ_LEN, BLOCK_SIZE);
        assert_eq!(committed, 40 * blocks_for_tokens(1300, BLOCK_SIZE));
        let reqs = vec![(PROMPT, MAX_TOKENS); 128];
        let (count, _) = admit_count(
            POOL_BLOCKS,
            committed,
            &reqs,
            MAX_SEQ_LEN,
            MAX_SEQ_LEN,
            BLOCK_SIZE,
        );
        assert_eq!(count, (POOL_BLOCKS - committed) / request_blocks());
    }

    #[test]
    fn admission_preserves_head_of_line_and_idle_liveness() {
        let reqs = vec![
            (PROMPT, MAX_TOKENS),
            (100_000, MAX_TOKENS),
            (PROMPT, MAX_TOKENS),
        ];
        assert_eq!(
            admit_count(POOL_BLOCKS, 0, &reqs, usize::MAX, 0, BLOCK_SIZE,),
            (1, false)
        );

        let oversize = vec![(200_000, MAX_TOKENS)];
        assert_eq!(
            admit_count(POOL_BLOCKS, 0, &oversize, usize::MAX, 0, BLOCK_SIZE,),
            (1, true)
        );
        assert_eq!(
            admit_count(POOL_BLOCKS, 10, &oversize, usize::MAX, 0, BLOCK_SIZE,),
            (0, false)
        );
    }

    #[test]
    fn watermark_zero_restores_prompt_only_admission() {
        let reqs = vec![(PROMPT, MAX_TOKENS); 128];
        assert_eq!(
            admit_count(POOL_BLOCKS, 0, &reqs, 0, MAX_SEQ_LEN, BLOCK_SIZE,),
            (128, false)
        );
    }

    #[test]
    fn verify_lookahead_is_reserved_for_all_active_speculative_proposers() {
        assert_eq!(resolve_verify_lookahead_tokens(true, true, 4), 4);
        assert_eq!(resolve_verify_lookahead_tokens(false, true, 4), 4);
        assert_eq!(resolve_verify_lookahead_tokens(true, false, 15), 15);
        assert_eq!(resolve_verify_lookahead_tokens(false, false, 15), 15);
        assert_eq!(resolve_verify_lookahead_tokens(true, true, 0), 0);
        assert_eq!(resolve_verify_lookahead_tokens(false, false, 0), 0);
    }

    #[test]
    fn zero_lookahead_is_byte_identical_to_the_legacy_reservation() {
        for (current, budget, watermark, ceiling) in [
            (PROMPT, MAX_TOKENS, usize::MAX, MAX_SEQ_LEN),
            (8_000, MAX_TOKENS, usize::MAX, MAX_SEQ_LEN),
            (9_000, 5, usize::MAX, MAX_SEQ_LEN),
            (PROMPT, MAX_TOKENS, 0, 0),
        ] {
            let demand = SeqDemand {
                current_tokens: current,
                budget_tokens: budget,
            };
            assert_eq!(
                seq_commitment_blocks_with_lookahead(&demand, watermark, ceiling, BLOCK_SIZE, 0),
                seq_commitment_blocks(&demand, watermark, ceiling, BLOCK_SIZE)
            );
        }
        let reqs = vec![(PROMPT, MAX_TOKENS); 128];
        assert_eq!(
            admit_count_with_lookahead(
                POOL_BLOCKS,
                0,
                &reqs,
                usize::MAX,
                MAX_SEQ_LEN,
                BLOCK_SIZE,
                0
            ),
            admit_count(POOL_BLOCKS, 0, &reqs, usize::MAX, MAX_SEQ_LEN, BLOCK_SIZE)
        );
    }

    #[test]
    fn k5_lookahead_adds_draft_rows_after_the_ceiling_clamp() {
        // The draft rows are counted after the depth: 1215 + 4 = 1219 tokens.
        let fits = SeqDemand {
            current_tokens: 1_215,
            budget_tokens: 0,
        };
        assert_eq!(
            seq_commitment_blocks_with_lookahead(&fits, usize::MAX, 0, BLOCK_SIZE, 4),
            blocks_for_tokens(1_219, BLOCK_SIZE)
        );
        let at_ceiling = SeqDemand {
            current_tokens: PROMPT,
            budget_tokens: MAX_SEQ_LEN,
        };
        assert_eq!(
            seq_commitment_blocks_with_lookahead(
                &at_ceiling,
                usize::MAX,
                MAX_SEQ_LEN,
                BLOCK_SIZE,
                4
            ),
            blocks_for_tokens(MAX_SEQ_LEN + 4, BLOCK_SIZE)
        );
        let crossing = SeqDemand {
            current_tokens: 1_212,
            budget_tokens: 0,
        };
        assert_eq!(
            seq_commitment_blocks_with_lookahead(&crossing, usize::MAX, 0, BLOCK_SIZE, 4),
            seq_commitment_blocks(&crossing, usize::MAX, 0, BLOCK_SIZE) + 1
        );
    }

    /// `--kv-pool-tokens 1048576` at `--max-seq-len 524288 --max-batch-size 4`:
    /// the admission view of the shared pool (dummy block excluded).
    fn shared_512k_pool() -> usize {
        const SEQ: usize = 524_288;
        (2 * SEQ / BLOCK_SIZE + 4 + 1) - 1
    }

    #[test]
    fn shared_pool_runs_two_deep_sessions_and_queues_the_rest() {
        const SEQ: usize = 524_288;
        let pool = shared_512k_pool();
        // Two 480K sessions with 32K of decode each, then two 32K requests
        // whose default budget is the rest of the context.
        let reqs = vec![
            (491_520, 32_768),
            (491_520, 32_768),
            (32_768, SEQ - 32_768),
            (32_768, SEQ - 32_768),
        ];
        let (count, forced) = admit_count(pool, 0, &reqs, SEQ, SEQ, BLOCK_SIZE);
        assert_eq!((count, forced), (2, false));
        // A third deep session queues behind the two running ones.
        let running: Vec<SeqDemand> = (0..2)
            .map(|_| SeqDemand {
                current_tokens: 491_520,
                budget_tokens: 32_768,
            })
            .collect();
        let committed = committed_blocks(&running, SEQ, SEQ, BLOCK_SIZE);
        assert!(committed <= pool);
        assert_eq!(
            admit_count(pool, committed, &reqs[2..], SEQ, SEQ, BLOCK_SIZE),
            (0, false)
        );
    }

    #[test]
    fn shared_pool_capacity_is_reused_after_a_session_finishes() {
        const SEQ: usize = 524_288;
        let pool = shared_512k_pool();
        let waiting = vec![(32_768, SEQ - 32_768), (32_768, SEQ - 32_768)];
        for _cycle in 0..1_000 {
            // One deep session running: exactly one waiting request fits.
            let one = [SeqDemand {
                current_tokens: 491_520,
                budget_tokens: 32_768,
            }];
            let committed = committed_blocks(&one, SEQ, SEQ, BLOCK_SIZE);
            assert_eq!(
                admit_count(pool, committed, &waiting, SEQ, SEQ, BLOCK_SIZE),
                (1, false)
            );
            // It finishes: both fit, and nothing leaks between cycles.
            assert_eq!(
                admit_count(pool, 0, &waiting, SEQ, SEQ, BLOCK_SIZE),
                (2, false)
            );
        }
    }

    #[test]
    fn shared_pool_short_decode_budgets_run_four_at_once() {
        const SEQ: usize = 524_288;
        let pool = shared_512k_pool();
        // Explicit small max_tokens: two deep and two shallow all fit.
        let reqs = vec![
            (400_000, 4_096),
            (400_000, 4_096),
            (32_768, 4_096),
            (32_768, 4_096),
        ];
        assert_eq!(
            admit_count(pool, 0, &reqs, SEQ, SEQ, BLOCK_SIZE),
            (4, false)
        );
    }

    #[test]
    fn k5_lookahead_can_queue_the_last_request_at_c8() {
        // Eight requests whose legacy reservation exactly fills the pool: the
        // four-row lookahead moves each into the next block, so only seven fit.
        let per_request = blocks_for_tokens(1_212, BLOCK_SIZE);
        let pool = per_request * 8;
        let reqs = vec![(1_212, 0); 8];
        assert_eq!(
            admit_count(pool, 0, &reqs, usize::MAX, 0, BLOCK_SIZE),
            (8, false)
        );
        assert_eq!(
            admit_count_with_lookahead(pool, 0, &reqs, usize::MAX, 0, BLOCK_SIZE, 4),
            (7, false)
        );
        assert_eq!(
            committed_blocks_with_lookahead(
                &[SeqDemand {
                    current_tokens: 1_212,
                    budget_tokens: 0,
                }],
                usize::MAX,
                0,
                BLOCK_SIZE,
                4,
            ),
            per_request + 1
        );
    }

    #[test]
    fn burst_reserved_demands_prevent_mid_loop_kv_over_admission() {
        // Reproduces lavd-test (10 x 26,053 prompt tokens, watermark 8,192,
        // 262,144-token shared pool -> 16,400 usable blocks). Initial admission
        // accepts 7 and leaves 3 in pending; mid-loop burst drain inside
        // start_new_requests must account for the 7 already-admitted requests
        // so it admits 0 of the 3 overflow requests instead of all 3.
        const POOL_262K: usize = 16_400;
        const LAVD_PROMPT: usize = 26_053;
        const LAVD_BUDGET: usize = 236_091;
        const WATERMARK: usize = 8_192;
        const CEILING: usize = 262_144;

        let initial = vec![(LAVD_PROMPT, LAVD_BUDGET); 10];
        let (admitted_initial, forced) =
            admit_count(POOL_262K, 0, &initial, WATERMARK, CEILING, BLOCK_SIZE);
        assert_eq!((admitted_initial, forced), (7, false));

        let burst_reserved: Vec<SeqDemand> = (0..admitted_initial)
            .map(|_| SeqDemand {
                current_tokens: LAVD_PROMPT,
                budget_tokens: LAVD_BUDGET,
            })
            .collect();
        let committed = committed_blocks(&burst_reserved, WATERMARK, CEILING, BLOCK_SIZE);
        assert!(committed <= POOL_262K);
        assert_eq!(
            admit_count(
                POOL_262K,
                committed,
                &initial[7..],
                WATERMARK,
                CEILING,
                BLOCK_SIZE
            ),
            (0, false),
            "mid-loop drain must not admit overflow requests past the 7 already in req_queue"
        );
    }

    fn blocking_req_with(
        prompt_tokens: Vec<u32>,
        req_max_tokens: usize,
        req_timeout_at: Option<Instant>,
    ) -> (
        InferenceRequest,
        tokio::sync::oneshot::Receiver<Result<InferenceResponse>>,
    ) {
        let (mut req, rx) = crate::scheduler::test_fixtures::blocking_request(prompt_tokens);
        if let InferenceRequest::Blocking {
            max_tokens,
            timeout_at,
            ..
        } = &mut req
        {
            *max_tokens = req_max_tokens;
            *timeout_at = req_timeout_at;
        }
        (req, rx)
    }

    fn streaming_req_with(
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        timeout_at: Option<Instant>,
    ) -> (
        InferenceRequest,
        tokio::sync::mpsc::Receiver<StreamEvent>,
        std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) {
        let (token_tx, rx) = tokio::sync::mpsc::channel(4);
        let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let req = InferenceRequest::Streaming {
            prompt_tokens,
            session_hash: 0,
            image_pixels: Vec::new(),
            max_tokens,
            min_tokens: 0,
            ignore_eos: false,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            top_n_sigma: 0.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 3,
            lz_penalty: 0.0,
            logit_bias: Vec::new(),
            stop_tokens: Vec::new(),
            enable_thinking: false,
            thinking_budget: None,
            require_tool_call: false,
            suppress_tool_call: false,
            disable_mtp: false,
            grammar_spec: None,
            seed: None,
            top_logprobs: None,
            timeout_at,
            decode_timeout_at: None,
            intervention_ledger: None,
            think_nudge: None,
            token_tx,
            cancel_flag: std::sync::Arc::clone(&cancel_flag),
        };
        (req, rx, cancel_flag)
    }

    #[test]
    fn prune_dead_pending_requests_drops_closed_cancelled_and_expired_in_order() {
        let now = Instant::now();
        let past = now
            .checked_sub(std::time::Duration::from_secs(1))
            .unwrap_or(now);
        let future = now + std::time::Duration::from_secs(60);

        let (dead_blocking, dead_rx) = blocking_req_with(vec![10], 8, None);
        drop(dead_rx);
        let (live_a, _live_a_rx) = blocking_req_with(vec![1], 8, Some(future));
        let (cancelled_stream, mut cancelled_rx, cancel_flag) =
            streaming_req_with(vec![20], 8, None);
        cancel_flag.store(true, std::sync::atomic::Ordering::Release);
        let (expired_blocking, mut expired_blocking_rx) =
            blocking_req_with(vec![30], 8, Some(past));
        let (expired_stream, mut expired_stream_rx, _flag) =
            streaming_req_with(vec![40], 8, Some(past));
        let (live_b, _live_b_rx, _live_b_flag) = streaming_req_with(vec![2], 8, Some(future));

        let mut queue = vec![
            dead_blocking,
            live_a,
            cancelled_stream,
            expired_blocking,
            expired_stream,
            live_b,
        ];
        let pruned = prune_dead_pending_requests(&mut queue, now);

        assert_eq!(pruned, 4);
        let surviving: Vec<&[u32]> = queue.iter().map(InferenceRequest::prompt_tokens).collect();
        assert_eq!(surviving, vec![&[1u32][..], &[2u32][..]]);
        assert!(
            matches!(cancelled_rx.try_recv(), Ok(StreamEvent::Error(msg)) if msg.contains("cancelled")),
            "live cancelled stream must receive cancellation error"
        );
        let err = match expired_blocking_rx.try_recv() {
            Ok(Err(err)) => err,
            Ok(Ok(_)) => panic!("expired blocking request unexpectedly succeeded"),
            Err(err) => panic!("expected blocking timeout error, got {err:?}"),
        };
        assert!(err.to_string().contains("timed out"), "{err:#}");
        assert!(
            matches!(expired_stream_rx.try_recv(), Ok(StreamEvent::Error(msg)) if msg.contains("timed out")),
            "live expired stream must receive timeout error"
        );
    }

    #[test]
    fn gate_admissions_prunes_dead_head_of_line_before_kv_accounting() {
        let now = Instant::now();
        let past = now
            .checked_sub(std::time::Duration::from_secs(1))
            .unwrap_or(now);
        let gate = crate::scheduler::queue_gate::QueueGate::new();
        gate.configure(Some(4));
        for _ in 0..3 {
            gate.try_enter().expect("enter queue gate");
        }
        assert_eq!(gate.waiting(), 3);

        // Oversize dead and expired requests sit ahead of a small live request.
        // Without pruning before KV demand accounting, head-of-line admission
        // would stop at index 0 and refuse to admit the live request.
        let (dead_oversize, dead_rx) = blocking_req_with(vec![1; 100_000], MAX_TOKENS, None);
        drop(dead_rx);
        let (expired_oversize, mut expired_rx) =
            blocking_req_with(vec![2; 100_000], MAX_TOKENS, Some(past));
        let (live_small, _live_rx) = blocking_req_with(vec![3; PROMPT], MAX_TOKENS, None);

        let pending = std::sync::Arc::new((
            Mutex::new(PendingQueue {
                requests: Vec::new(),
                closed: false,
            }),
            Condvar::new(),
        ));

        let admitted = gate_admissions_with_blocks(
            request_blocks(),
            &pending,
            vec![dead_oversize, expired_oversize, live_small],
            &[],
            &[],
            &[],
            &[],
            MAX_SEQ_LEN,
            MAX_SEQ_LEN,
            BLOCK_SIZE,
            0,
            MemFloorGate::Allow,
            &gate,
        );

        assert_eq!(
            admitted.len(),
            1,
            "live request behind dead HOL is admitted"
        );
        assert_eq!(admitted[0].prompt_len(), PROMPT);
        assert!(pending.0.lock().requests.is_empty());
        assert_eq!(
            gate.waiting(),
            0,
            "pruned + admitted requests must release every queue-gate slot"
        );
        let err = match expired_rx.try_recv() {
            Ok(Err(err)) => err,
            Ok(Ok(_)) => panic!("expired request unexpectedly succeeded"),
            Err(err) => panic!("expired request must receive timeout error, got {err:?}"),
        };
        assert!(err.to_string().contains("timed out"), "{err:#}");
    }

    #[test]
    fn drain_pending_requests_prunes_dead_entries_when_batch_cap_is_zero() {
        let now = Instant::now();
        let past = now
            .checked_sub(std::time::Duration::from_secs(1))
            .unwrap_or(now);
        let gate = crate::scheduler::queue_gate::QueueGate::new();
        gate.configure(Some(4));
        for _ in 0..3 {
            gate.try_enter().expect("enter queue gate");
        }

        let (dead_req, dead_rx) = blocking_req_with(vec![1], 8, None);
        drop(dead_rx);
        let (expired_req, mut expired_rx) = blocking_req_with(vec![2], 8, Some(past));
        let (live_req, _live_rx) = blocking_req_with(vec![3], 8, None);

        let pending = std::sync::Arc::new((
            Mutex::new(PendingQueue {
                requests: vec![dead_req, expired_req, live_req],
                closed: false,
            }),
            Condvar::new(),
        ));
        let active = vec![crate::scheduler::test_fixtures::active_seq(
            ResponseSink::Blocking(None),
            0,
        )];

        let drained = crate::scheduler::mod_helpers::drain_pending_requests_with_gate(
            &pending,
            &active,
            &[],
            &crate::scheduling_policy::FifoPolicy,
            1, // max_batch_size == active.len() => cap == 0
            &gate,
        );

        assert!(drained.is_empty());
        {
            let g = pending.0.lock();
            assert_eq!(g.requests.len(), 1, "only live request remains queued");
            assert_eq!(g.requests[0].prompt_tokens(), &[3u32]);
        }
        assert_eq!(
            gate.waiting(),
            1,
            "pruned queued requests release their queue-gate slots even at cap == 0"
        );
        let err = match expired_rx.try_recv() {
            Ok(Err(err)) => err,
            Ok(Ok(_)) => panic!("expired queued request unexpectedly succeeded"),
            Err(err) => panic!("expired queued request must receive timeout error, got {err:?}"),
        };
        assert!(err.to_string().contains("timed out"), "{err:#}");
    }

    fn blocking_req_with_deadlines(
        prompt_tokens: Vec<u32>,
        req_max_tokens: usize,
        req_timeout_at: Option<Instant>,
        req_decode_timeout_at: Option<Instant>,
    ) -> (
        InferenceRequest,
        tokio::sync::oneshot::Receiver<Result<InferenceResponse>>,
    ) {
        let (mut req, rx) = crate::scheduler::test_fixtures::blocking_request(prompt_tokens);
        if let InferenceRequest::Blocking {
            max_tokens,
            timeout_at,
            decode_timeout_at,
            ..
        } = &mut req
        {
            *max_tokens = req_max_tokens;
            *timeout_at = req_timeout_at;
            *decode_timeout_at = req_decode_timeout_at;
        }
        (req, rx)
    }

    #[test]
    fn server_default_request_survives_past_timeout_at_in_pending_queue_while_connected() {
        use std::time::Duration;

        // Forward Instant arithmetic avoids any dependency on container uptime.
        let enqueued_at = Instant::now();
        let past_base = enqueued_at + Duration::from_secs(300);
        let now = enqueued_at + Duration::from_secs(600);
        let derived_decode = now + Duration::from_secs(3_600);

        let (server_default_req, _live_rx) = blocking_req_with_deadlines(
            vec![7; PROMPT],
            MAX_TOKENS,
            Some(past_base),
            Some(derived_decode),
        );
        record_pending_enqueued(&server_default_req, enqueued_at);

        let mut queue = vec![server_default_req];
        let pruned = prune_dead_pending_requests(&mut queue, now);
        assert_eq!(
            pruned, 0,
            "connected server-default request must not expire on timeout_at while queued"
        );
        assert_eq!(queue.len(), 1);
        assert_eq!(
            pending_prune_reason_with(&queue[0], Some(enqueued_at), now, false, None),
            None,
            "default FLUX_QUEUE_TIMEOUT=0 never prunes connected server-default requests"
        );
        assert_eq!(
            pending_prune_reason_with(
                &queue[0],
                Some(enqueued_at),
                now,
                false,
                Some(Duration::from_secs(600)),
            ),
            Some(PendingPruneReason::TimedOut),
            "explicit FLUX_QUEUE_TIMEOUT must prune once queue wait reaches the limit"
        );
        assert_eq!(
            pending_prune_reason_with(
                &queue[0],
                Some(enqueued_at),
                now,
                false,
                Some(Duration::from_secs(601)),
            ),
            None,
        );
        let _ = take_pending_enqueued(&queue[0]);
    }

    #[test]
    fn server_default_request_shifts_deadlines_forward_by_queue_wait_on_admission() {
        use std::time::Duration;

        // Pure shift check: uses forward Instant arithmetic so it never depends
        // on container uptime being larger than the simulated wait.
        let enqueued_at = Instant::now();
        let wait = Duration::from_secs(120);
        let admit_now = enqueued_at + wait;
        let base_deadline = enqueued_at + Duration::from_secs(60);
        let decode_deadline = enqueued_at + Duration::from_secs(7_200);

        let (mut pure_req, _pure_rx) = blocking_req_with_deadlines(
            vec![10; PROMPT],
            MAX_TOKENS,
            Some(base_deadline),
            Some(decode_deadline),
        );
        shift_request_deadlines_for_wait(&mut pure_req, Some(enqueued_at), admit_now);
        assert_eq!(pure_req.timeout_at(), Some(base_deadline + wait));
        assert_eq!(pure_req.decode_timeout_at(), Some(decode_deadline + wait));

        // End-to-end gate_admissions_with_blocks shifts both deadlines and clears
        // the enqueue sidecar entry.
        let gate_enqueued = Instant::now();
        let gate_base = gate_enqueued + Duration::from_secs(60);
        let gate_decode = gate_enqueued + Duration::from_secs(7_200);
        let (req, _rx) = blocking_req_with_deadlines(
            vec![11; PROMPT],
            MAX_TOKENS,
            Some(gate_base),
            Some(gate_decode),
        );
        record_pending_enqueued(&req, gate_enqueued);
        std::thread::sleep(Duration::from_millis(5));

        let gate = crate::scheduler::queue_gate::QueueGate::new();
        let pending = std::sync::Arc::new((
            Mutex::new(PendingQueue {
                requests: Vec::new(),
                closed: false,
            }),
            Condvar::new(),
        ));

        let admitted = gate_admissions_with_blocks(
            request_blocks(),
            &pending,
            vec![req],
            &[],
            &[],
            &[],
            &[],
            MAX_SEQ_LEN,
            MAX_SEQ_LEN,
            BLOCK_SIZE,
            0,
            MemFloorGate::Allow,
            &gate,
        );
        assert_eq!(admitted.len(), 1);
        let shifted_base = admitted[0].timeout_at().expect("shifted timeout_at");
        let shifted_decode = admitted[0]
            .decode_timeout_at()
            .expect("shifted decode_timeout_at");
        assert!(
            shifted_base > gate_base,
            "prefill timeout_at must shift forward by queue wait on admission"
        );
        assert!(
            shifted_decode > gate_decode,
            "decode_timeout_at must shift forward by queue wait on admission"
        );
        assert!(
            lookup_pending_enqueued(&admitted[0]).is_none(),
            "admitted request cleans up its enqueue sidecar entry"
        );
    }

    #[test]
    fn server_default_request_prunes_on_disconnect_and_explicit_timeout_still_expires() {
        use std::time::Duration;

        let past = Instant::now();
        let now = past + Duration::from_secs(10);
        let future_decode = now + Duration::from_secs(3_600);

        // Disconnected server-default request prunes immediately.
        let (disconnected_default, dead_rx) =
            blocking_req_with_deadlines(vec![13; 8], MAX_TOKENS, Some(past), Some(future_decode));
        drop(dead_rx);

        // Explicit-timeout request (decode_timeout_at == None) still prunes at timeout_at.
        let (explicit_expired, mut explicit_rx) =
            blocking_req_with_deadlines(vec![14; 8], MAX_TOKENS, Some(past), None);

        let mut queue = vec![disconnected_default, explicit_expired];
        let pruned = prune_dead_pending_requests(&mut queue, now);
        assert_eq!(pruned, 2);
        assert!(queue.is_empty());
        let err = match explicit_rx.try_recv() {
            Ok(Err(err)) => err,
            Ok(Ok(_)) => panic!("explicitly expired request unexpectedly succeeded"),
            Err(err) => panic!("expected timeout error, got {err:?}"),
        };
        assert!(err.to_string().contains("timed out"), "{err:#}");
    }

    #[test]
    fn active_growth_reserve_holds_back_admission_when_active_exceeds_watermark() {
        const WATERMARK: usize = 512;
        const CEILING: usize = 8_192;

        let mut active_seq =
            crate::scheduler::test_fixtures::active_seq(ResponseSink::Blocking(None), 0);
        // current_tokens = seq_len + 1 = 1025 > WATERMARK (512).
        active_seq.seq.seq_len = 1_024;
        active_seq.remaining = 2_048;

        let active_demand = SeqDemand {
            current_tokens: active_seq.seq.seq_len + 1,
            budget_tokens: active_seq.remaining,
        };
        let active_committed = committed_blocks(&[active_demand], WATERMARK, CEILING, BLOCK_SIZE);
        let incoming_need = seq_commitment_blocks(
            &SeqDemand {
                current_tokens: PROMPT,
                budget_tokens: MAX_TOKENS,
            },
            WATERMARK,
            CEILING,
            BLOCK_SIZE,
        );
        // Pool holds active + 1 incoming request without growth reserve, but NOT
        // with the extra growth reserve (watermark_blocks = 33 blocks).
        let tight_pool = active_committed + incoming_need;
        let expected_reserve = blocks_for_tokens(WATERMARK, BLOCK_SIZE).min(tight_pool / 4);
        assert!(expected_reserve > 0);
        assert_eq!(
            active_growth_reserve_blocks(&[active_seq], WATERMARK, tight_pool, BLOCK_SIZE),
            expected_reserve
        );

        let mut active_seq_for_gate =
            crate::scheduler::test_fixtures::active_seq(ResponseSink::Blocking(None), 0);
        active_seq_for_gate.seq.seq_len = 1_024;
        active_seq_for_gate.remaining = 2_048;

        let (queued_req, _rx) = blocking_req_with(vec![19; PROMPT], MAX_TOKENS, None);
        let gate = crate::scheduler::queue_gate::QueueGate::new();
        let pending = std::sync::Arc::new((
            Mutex::new(PendingQueue {
                requests: Vec::new(),
                closed: false,
            }),
            Condvar::new(),
        ));
        let admitted = gate_admissions_with_blocks(
            tight_pool,
            &pending,
            vec![queued_req],
            &[active_seq_for_gate],
            &[],
            &[],
            &[],
            WATERMARK,
            CEILING,
            BLOCK_SIZE,
            0,
            MemFloorGate::Allow,
            &gate,
        );
        assert!(
            admitted.is_empty(),
            "growth reserve must hold back additional admission when active seq exceeds watermark"
        );
        assert_eq!(pending.0.lock().requests.len(), 1);
    }

    #[test]
    fn memory_floor_refusal_pauses_pending_promotion_while_pruning_dead_requests() {
        let gate = crate::scheduler::queue_gate::QueueGate::new();
        gate.configure(Some(4));
        for _ in 0..4 {
            gate.try_enter().expect("enter queue gate");
        }

        let (dead_req, dead_rx) = blocking_req_with(vec![1; PROMPT], MAX_TOKENS, None);
        drop(dead_rx);
        let (live_a, _rx_a) = blocking_req_with(vec![2; PROMPT], MAX_TOKENS, None);
        let (live_b, _rx_b) = blocking_req_with(vec![3; PROMPT], MAX_TOKENS, None);
        let (tail_c, _rx_c) = blocking_req_with(vec![4; PROMPT], MAX_TOKENS, None);

        let pending = std::sync::Arc::new((
            Mutex::new(PendingQueue {
                requests: vec![tail_c],
                closed: false,
            }),
            Condvar::new(),
        ));

        // While MemAvailable is breached, dead entries still prune (freeing
        // QueueGate slots), while live requests return to the front of pending
        // in FIFO order even when KV blocks are plentiful (or total_blocks == 0).
        let paused = gate_admissions_with_blocks(
            POOL_BLOCKS,
            &pending,
            vec![dead_req, live_a, live_b],
            &[],
            &[],
            &[],
            &[],
            MAX_SEQ_LEN,
            MAX_SEQ_LEN,
            BLOCK_SIZE,
            0,
            MemFloorGate::Refusing,
            &gate,
        );
        assert!(
            paused.is_empty(),
            "breached memory floor must pause promotion"
        );
        assert_eq!(
            gate.waiting(),
            3,
            "pruned dead request releases its slot; 3 live requests remain queued"
        );
        {
            let q = pending.0.lock();
            let first_tokens: Vec<u32> = q.requests.iter().map(|r| r.prompt_tokens()[0]).collect();
            assert_eq!(
                first_tokens,
                vec![2, 3, 4],
                "FIFO order preserved ahead of tail"
            );
        }

        // Non-paged backends (`total_blocks == 0`) must pause identically.
        let drained_for_zero_pool: Vec<InferenceRequest> = {
            let mut q = pending.0.lock();
            q.requests.drain(..2).collect()
        };
        let paused_zero_pool = gate_admissions_with_blocks(
            0,
            &pending,
            drained_for_zero_pool,
            &[],
            &[],
            &[],
            &[],
            MAX_SEQ_LEN,
            MAX_SEQ_LEN,
            BLOCK_SIZE,
            0,
            MemFloorGate::Refusing,
            &gate,
        );
        assert!(paused_zero_pool.is_empty());
        assert_eq!(gate.waiting(), 3);

        // Once MemAvailable recovers, queued requests promote normally.
        let ready: Vec<InferenceRequest> = {
            let mut q = pending.0.lock();
            q.requests.drain(..).collect()
        };
        let admitted = gate_admissions_with_blocks(
            POOL_BLOCKS,
            &pending,
            ready,
            &[],
            &[],
            &[],
            &[],
            MAX_SEQ_LEN,
            MAX_SEQ_LEN,
            BLOCK_SIZE,
            0,
            MemFloorGate::Allow,
            &gate,
        );
        assert_eq!(admitted.len(), 3);
        let admitted_first_tokens: Vec<u32> =
            admitted.iter().map(|r| r.prompt_tokens()[0]).collect();
        assert_eq!(admitted_first_tokens, vec![2, 3, 4]);
        assert!(pending.0.lock().requests.is_empty());
        assert_eq!(gate.waiting(), 0);
    }
}
