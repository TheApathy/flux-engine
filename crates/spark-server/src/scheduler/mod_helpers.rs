// SPDX-License-Identifier: AGPL-3.0-only

//! Per-iteration helpers used by `scheduler::run`:
//!   • install_high_speed_swap — orchestrator install after CUDA bind
//!   • drain_pending_requests — pop policy-selected reqs off the queue
//!   • retire_finished_sequences — swap_remove + slot compaction

use parking_lot::{Condvar, Mutex};
use spark_model::traits::Model;
use std::sync::Arc;

use super::*;
use crate::api::InferenceRequest;
use crate::scheduling_policy::{ActiveSeqTiming, PendingRequestInfo, SchedulingPolicy};

/// Install --high-speed-swap orchestrator after bind_gpu_to_thread.
pub(super) fn install_high_speed_swap(
    model: &dyn Model,
    cfg: Option<spark_storage::HighSpeedSwapConfig>,
) {
    let Some(cfg) = cfg else { return };
    match model.high_speed_swap_dims() {
        Some(dims) => {
            tracing::info!(
                "--high-speed-swap installing: dir={}, scratch={} blocks, qd={}, rank={}, \
                 model: {} layers × {}/{} (q/kv) heads × hd={}, bs={}, max_blocks={}",
                cfg.dir.display(),
                cfg.resident_blocks,
                cfg.qd,
                cfg.rank,
                dims.num_layers,
                dims.num_q_heads,
                dims.num_kv_heads,
                dims.head_dim,
                dims.block_size,
                dims.max_blocks_per_layer,
            );
            // Use the model's default stream (cuMemcpyHtoDAsync(stream=0))
            // for orchestrator setup. The hot-path API takes its own stream.
            if let Err(e) = spark_storage::install_local(0, cfg, dims) {
                tracing::error!("--high-speed-swap install failed: {e:#}");
            } else {
                tracing::info!("--high-speed-swap orchestrator installed on scheduler thread");
                if let Ok(raw) = std::env::var("ATLAS_HIGH_SPEED_SWAP_REPLACE") {
                    let trimmed = raw.trim();
                    if !trimmed.is_empty()
                        && trimmed != "0"
                        && !trimmed.eq_ignore_ascii_case("false")
                    {
                        tracing::warn!(
                            value = trimmed,
                            "ATLAS_HIGH_SPEED_SWAP_REPLACE is not implemented and is ignored."
                        );
                    }
                }
            }
        }
        None => {
            tracing::warn!(
                "--high-speed-swap requested but model does not expose high_speed_swap_dims; \
                 orchestrator NOT installed"
            );
        }
    }
}

/// Depth of the scheduler-side pending queue before the receiver thread
/// stops draining the tokio request channel. `ATLAS_PENDING_QUEUE_CAP`
/// overrides; the default keeps a few batches' worth of work queued so a
/// burst still pipelines, while an abandoned-client flood cannot grow host
/// memory without bound (each queued request owns its prompt tokens and
/// image pixels).
pub(super) fn pending_queue_cap(max_batch_size: usize) -> usize {
    pending_queue_cap_with(
        std::env::var("ATLAS_PENDING_QUEUE_CAP").ok().as_deref(),
        max_batch_size,
    )
}

fn pending_queue_cap_with(override_var: Option<&str>, max_batch_size: usize) -> usize {
    let default = max_batch_size.saturating_mul(4).max(32);
    override_var
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// Receiver thread: blocks on the tokio request channel and hands each
/// request to the scheduler through the pending queue + condvar.
///
/// Bounded: the tokio channel is sized `max_num_seqs`, but this thread used
/// to drain it into an unbounded `Vec` immediately, so the bound never
/// applied — every HTTP request was accepted and parked here forever (prompt
/// tokens, image pixels, grammar specs and all) with no backpressure on
/// `request_tx.send().await`. Parking here while the queue holds `cap`
/// requests lets the tokio channel fill, which is what finally makes the
/// handlers return overload from their nonblocking admission attempt.
/// `drain_pending_requests` wakes this thread when it takes requests; both
/// waiter kinds share the condvar, hence `notify_all`.
pub(super) fn spawn_request_receiver(
    request_rx: tokio::sync::mpsc::Receiver<InferenceRequest>,
    pending: Arc<(Mutex<PendingQueue>, Condvar)>,
    cap: usize,
) -> request_receiver::RequestReceiver {
    request_receiver::RequestReceiver::spawn(request_rx, pending, cap)
}

fn push_pending_bounded(
    pending: &(Mutex<PendingQueue>, Condvar),
    req: InferenceRequest,
    cap: usize,
) -> Option<InferenceRequest> {
    let (mtx, cv) = pending;
    let mut g = mtx.lock();
    while g.requests.len() >= cap && !g.closed {
        cv.wait(&mut g);
    }
    if g.closed {
        return Some(req);
    }
    super::admission::record_pending_enqueued(&req, Instant::now());
    g.requests.push(req);
    crate::metrics::compat_metrics::set_waiting_count(g.requests.len());
    drop(g);
    cv.notify_all();
    None
}

#[path = "request_receiver.rs"]
mod request_receiver;

fn burst_gather_quiet_ms() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        if let Ok(raw) = std::env::var("ATLAS_SCHED_BURST_GATHER_MS")
            && let Ok(parsed) = raw.parse::<u64>()
        {
            return parsed;
        }
        if std::env::var("ATLAS_QWEN4_VERIFY_FAST_HOST")
            .ok()
            .as_deref()
            == Some("1")
        {
            8
        } else {
            0
        }
    })
}

/// Drain pending request queue and policy-select prefills to start.
#[cfg(test)]
pub(super) fn drain_pending_requests(
    pending: &Arc<(Mutex<PendingQueue>, Condvar)>,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    policy: &dyn SchedulingPolicy,
    max_batch_size: usize,
) -> Vec<InferenceRequest> {
    drain_pending_requests_for_tick(pending, active, prefilling, policy, max_batch_size, false)
}

/// Drain pending request queue using the same `ATLAS_PREFILL_SRPT` default
/// resolution as in-flight continuation (`interleave_active` defaults SRPT on).
pub(super) fn drain_pending_requests_for_tick(
    pending: &Arc<(Mutex<PendingQueue>, Condvar)>,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    policy: &dyn SchedulingPolicy,
    max_batch_size: usize,
    interleave_active: bool,
) -> Vec<InferenceRequest> {
    drain_pending_requests_with_gate_and_srpt(
        pending,
        active,
        prefilling,
        policy,
        max_batch_size,
        &super::queue_gate::QUEUE_GATE,
        interleave_active,
    )
}

#[cfg(test)]
pub(super) fn drain_pending_requests_with_gate(
    pending: &Arc<(Mutex<PendingQueue>, Condvar)>,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    policy: &dyn SchedulingPolicy,
    max_batch_size: usize,
    gate: &super::queue_gate::QueueGate,
) -> Vec<InferenceRequest> {
    drain_pending_requests_with_gate_and_srpt(
        pending,
        active,
        prefilling,
        policy,
        max_batch_size,
        gate,
        false,
    )
}

fn drain_pending_requests_with_gate_and_srpt(
    pending: &Arc<(Mutex<PendingQueue>, Condvar)>,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    policy: &dyn SchedulingPolicy,
    max_batch_size: usize,
    gate: &super::queue_gate::QueueGate,
    interleave_active: bool,
) -> Vec<InferenceRequest> {
    let (ref mtx, ref cv) = **pending;
    let mut g = mtx.lock();
    if active.is_empty() && prefilling.is_empty() {
        // Block until signalled (no busy-wait, no polling).
        if g.requests.is_empty() {
            crate::metrics::compat_metrics::set_scheduler_counts(0, 0);
        }
        while g.requests.is_empty() && !g.closed {
            cv.wait(&mut g);
        }
        if g.closed && g.requests.is_empty() {
            crate::metrics::compat_metrics::set_scheduler_counts(0, 0);
            return Vec::new();
        }
        let quiet_ms = burst_gather_quiet_ms();
        if quiet_ms > 0 && max_batch_size > 1 && !g.closed && g.requests.len() < max_batch_size {
            let quiet = std::time::Duration::from_millis(quiet_ms);
            let mut deadline = Instant::now() + std::time::Duration::from_millis(quiet_ms * 5);
            let ingress_cap = Instant::now() + std::time::Duration::from_millis(quiet_ms * 20);
            while g.requests.len() < max_batch_size && !g.closed {
                let now = Instant::now();
                let in_flight = crate::metrics::in_flight_ingress();
                if in_flight > 0 && deadline < ingress_cap {
                    deadline =
                        (now + std::time::Duration::from_millis(quiet_ms * 5)).min(ingress_cap);
                }
                if now >= deadline {
                    break;
                }
                let prev_len = g.requests.len();
                let wait = quiet.min(deadline - now);
                let _ = cv.wait_for(&mut g, wait);
                if g.requests.len() == prev_len && crate::metrics::in_flight_ingress() == 0 {
                    break;
                }
            }
        }
    }

    // Prune disconnected, cancelled, retiring, or expired requests while
    // waiting — BEFORE `should_prefill` or `cap == 0` early returns — so
    // dead requests release host memory, `QUEUE_GATE` places, and
    // pending-queue headroom instead of starving live arrivals.
    let pruned = super::admission::prune_dead_pending_requests(&mut g.requests, Instant::now());
    if pruned > 0 {
        gate.leave(pruned);
        cv.notify_all();
    }

    // Ask policy whether to accept prefills this iteration.
    let timings: Vec<ActiveSeqTiming> = active
        .iter()
        .map(|a| ActiveSeqTiming {
            last_token_time: a.last_token_time,
        })
        .collect();

    if g.requests.is_empty() || !policy.should_prefill(&timings) {
        crate::metrics::compat_metrics::set_scheduler_counts(
            active.len() + prefilling.len(),
            g.requests.len(),
        );
        return Vec::new();
    }

    // Account for both active and in-progress prefilling sequences.
    let cap = max_batch_size.saturating_sub(active.len() + prefilling.len());
    if cap == 0 {
        crate::metrics::compat_metrics::set_scheduler_counts(
            active.len() + prefilling.len(),
            g.requests.len(),
        );
        return Vec::new();
    }

    let infos: Vec<PendingRequestInfo> = g
        .requests
        .iter()
        .enumerate()
        .map(|(i, req)| PendingRequestInfo {
            prompt_len: req.prompt_len(),
            index: i,
        })
        .collect();
    let srpt_enabled =
        super::prefill_slo::prefill_srpt::SrptConfig::from_env(interleave_active).enabled;
    let selected = select_pending_with_srpt(
        policy,
        &g.requests,
        &infos,
        cap,
        active.is_empty() && prefilling.is_empty(),
        srpt_enabled,
    );

    // Remove selected indices from pending (reverse order to preserve indices).
    let mut remove_indices = selected.clone();
    remove_indices.sort_unstable_by(|a, b| b.cmp(a));
    let mut taken: Vec<(usize, InferenceRequest)> = Vec::with_capacity(selected.len());
    for idx in remove_indices {
        taken.push((idx, g.requests.remove(idx)));
    }

    // Re-sort into policy-selected order.
    let mut result = Vec::with_capacity(selected.len());
    for &sel_idx in &selected {
        if let Some(pos) = taken.iter().position(|(i, _)| *i == sel_idx) {
            let (_, req) = taken.swap_remove(pos);
            result.push(req);
        }
    }
    crate::metrics::compat_metrics::set_scheduler_counts(
        active.len() + prefilling.len(),
        g.requests.len(),
    );
    if !result.is_empty() {
        // The receiver thread parks on this condvar while the queue is at
        // capacity (see `scheduler::run`); room was just made.
        cv.notify_all();
    }
    result
}

/// Select pending request indices: shortest-prompt-first with anti-starvation
/// aging and FIFO tie-breaking when SRPT is enabled (`ATLAS_PREFILL_SRPT=1`,
/// or unset while SLO interleave is active), otherwise deferring to
/// `policy.select_prefills`.
fn select_pending_with_srpt(
    policy: &dyn SchedulingPolicy,
    requests: &[InferenceRequest],
    infos: &[PendingRequestInfo],
    cap: usize,
    idle_drain: bool,
    srpt_enabled: bool,
) -> Vec<usize> {
    if srpt_enabled && (idle_drain || requests.len() > 1) {
        let cfg = super::prefill_slo::prefill_srpt::SrptConfig {
            enabled: true,
            ..super::prefill_slo::prefill_srpt::SrptConfig::default()
        };
        let candidates: Vec<(u64, usize)> = requests
            .iter()
            .map(|req| {
                let key = super::prefill_slo::prefill_srpt::pending_wait_key(
                    req.prompt_tokens().as_ptr() as usize,
                    req.prompt_len(),
                );
                (key, req.prompt_len())
            })
            .collect();
        return super::prefill_slo::prefill_srpt::select_pending_indices(cfg, &candidates, cap);
    }
    policy.select_prefills(infos, cap)
}

/// Retire finished sequences. After swap_remove, the last element moves to
/// position i. Compact its SSM states into the slot the RETIRED sequence
/// owned, so the pool stays dense without ever touching a slot held by a
/// sequence that is not in `active` (chunked prefills and swap-ins claim
/// pool slots too, and the LIFO free list hands them out in arbitrary order,
/// so vector index `i` is NOT a safe compaction destination).
///
/// Every batched decode path addresses recurrent state through
/// `seq.layer_states` / `seq.slot_idx` (pointer tables, slot-keyed graphs),
/// so no consumer requires `slot_idx == index`.
///
/// ORDER (upstream 11e83f5b, claim-first): the retired sequence is finished
/// — and its slot RELEASED to the pool — BEFORE the survivor is compacted
/// onto it. `Model::compact_sequence` only accepts a target it can claim
/// from the free list, so a slot anyone still owns (the retired sequence
/// before its release, a prefilling request the active list cannot see) is
/// refused with `Ok(false)` and the survivor stays where it is. Contiguity
/// is best-effort; no path needs it for correctness. The scheduler runs on
/// one thread, so nothing can claim the vacated slot between the release
/// and the survivor's claim.
pub(super) fn retire_finished_sequences(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    cache_on_finish: &mut Vec<bool>,
) {
    assert_eq!(active.len(), cache_on_finish.len());
    let mut i = 0;
    while i < active.len() {
        if active[i].finished {
            let (mut a, cache_sequence) = if model.hc_mult() > 0 {
                (active.remove(i), cache_on_finish.remove(i))
            } else {
                (active.swap_remove(i), cache_on_finish.swap_remove(i))
            };
            // Destination = the retired sequence's own slot (`a.seq.slot_idx`),
            // never the vector index `i`: a chunked prefill or a swap-in may
            // own slot `i` while not being represented in `active`, and the
            // old `compact(.., i)` overwrote that request's SSM/PLE state and
            // leaked the retired sequence's real slot (sentinel set, slot
            // never released) whenever `a.seq.slot_idx != i`.
            let retired_slot = a.seq.slot_idx;
            // Finish FIRST: streams Done, admits the prefix-cache snapshot
            // (read from the retired slot, still intact) and frees the KV
            // blocks and the slot. Only a free slot is a legal target below.
            finish_sequence_with_cache(model, &mut a, cache_sequence);
            // The v1 Qwen4 concurrency route is deliberately slot-stable.
            // Per-row C1 decode is slot-agnostic, so leave the swapped
            // survivor in its original slot there.
            if model.hc_mult() == 0
                && i < active.len()
                && retired_slot != usize::MAX
                && active[i].seq.slot_idx > retired_slot
            {
                compact_survivor_into_vacated_slot(model, &mut active[i], retired_slot, "retire");
            }
        } else {
            i += 1;
        }
    }
}

/// Compact `survivor` DOWN into `vacated`, a slot its previous owner has
/// already released. `compact_sequence` claims the target from the pool
/// before copying; a refusal (`Ok(false)`) means the slot is still owned —
/// a bookkeeping fault upstream of this call — and the survivor is left in
/// place rather than overwriting the owner's recurrent state. A model-side
/// error likewise leaves the survivor bound to its source slot (the trait
/// contract), so nothing is freed or sentinelled here.
pub(super) fn compact_survivor_into_vacated_slot(
    model: &dyn Model,
    survivor: &mut ActiveSeq,
    vacated: usize,
    site: &str,
) {
    let from = survivor.seq.slot_idx;
    match model.compact_sequence(&mut survivor.seq, vacated) {
        Ok(true) => {}
        Ok(false) => tracing::error!(
            "{site}: slot {vacated} was released by its owner yet the pool \
             refused the claim (still owned); survivor stays on slot {from}"
        ),
        Err(e) => tracing::error!("{site} compact_sequence: {e:#}"),
    }
}

/// Truncate a CSK-verified sequence down to `target_len`, failing closed (and
/// notifying the client sink) if the verifier failed to advance `seq_len` past
/// `target_len`.
pub(super) fn truncate_csk_verify_extent(
    a: &mut ActiveSeq,
    orig_len: usize,
    target_len: usize,
    k: usize,
) -> bool {
    if target_len > a.seq.seq_len {
        tracing::error!(
            orig_len,
            target_len,
            seq_len = a.seq.seq_len,
            k,
            "CSK verify did not advance seq_len past target_len"
        );
        fail_speculative_step(
            a,
            "CSK verify seq_len invariant",
            format!("seq_len {} < target_len {target_len}", a.seq.seq_len),
        );
        return false;
    }
    a.seq.tokens.truncate(target_len);
    a.seq.seq_len = target_len;
    true
}

/// Reconcile host token buffers and GPU recurrent/verify state when a CSK
/// sequence finishes mid-emission (`consumed_rows` in `0..=k`).
pub(super) fn reconcile_terminal_csk_verify(
    model: &dyn Model,
    a: &mut ActiveSeq,
    orig_len: usize,
    consumed_rows: usize,
    k: usize,
) {
    let final_len = orig_len + consumed_rows;
    a.seq.tokens.truncate(final_len);
    a.seq.seq_len = final_len;
    let res = if consumed_rows == 0 {
        super::verify_dflash_step::rollback_failed_verify_frame(model, &mut a.seq, k)
    } else {
        model.commit_verify_state_async(&mut a.seq, consumed_rows, k)
    };
    if let Err(e) = res {
        tracing::error!("CSK K{k} terminal reconcile (consumed={consumed_rows}): {e:#}");
        fail_speculative_step(a, "CSK terminal reconcile", e);
    }
}

#[cfg(test)]
#[path = "mod_helpers_tests.rs"]
mod tests;
