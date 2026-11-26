// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-free tests for the per-iteration helpers: retire-path slot
//! compaction and the pending-queue hand-off.

use super::*;
use crate::scheduler::test_fixtures::{active_seq, model};

/// `(from, to)` slot moves the model saw.
fn compacts(m: &model::TestModel) -> Vec<(usize, usize)> {
    m.compacts.lock().unwrap().clone()
}

/// `slot_idx` of every sequence the scheduler asked the model to free.
fn frees(m: &model::TestModel) -> Vec<usize> {
    m.frees.lock().unwrap().clone()
}

fn slots(active: &[ActiveSeq]) -> Vec<usize> {
    active.iter().map(|a| a.seq.slot_idx).collect()
}

/// C1: a finished sequence at vector index 1 owns slot 3 while a chunked
/// prefill (not in `active`) owns slot 1. The survivor that `swap_remove`
/// moves into index 1 must be compacted into slot 3 — the retired slot —
/// never into slot 1, which would overwrite the prefill's recurrent state
/// and leak slot 3 behind the `usize::MAX` sentinel.
#[test]
fn retire_compacts_the_survivor_into_the_retired_slot_not_the_vector_index() {
    let m = model::recording();
    let mut active = vec![
        active_seq(ResponseSink::Blocking(None), 0),
        active_seq(ResponseSink::Blocking(None), 3),
        active_seq(ResponseSink::Blocking(None), 4),
    ];
    active[1].finished = true;
    let mut cache = vec![false; 3];

    retire_finished_sequences(&m, &mut active, &mut cache);

    assert_eq!(
        compacts(&m),
        vec![(4, 3)],
        "survivor moves into the retired slot"
    );
    assert!(
        compacts(&m).iter().all(|&(_, to)| to != 1),
        "slot 1 belongs to a prefill outside `active` and must never be a destination"
    );
    assert_eq!(slots(&active), vec![0, 3]);
    assert_eq!(
        frees(&m),
        vec![3],
        "the retired sequence releases its own slot BEFORE the survivor claims it"
    );
    assert_eq!(cache, vec![false; 2]);
}

/// Claim-first (upstream 11e83f5b): a destination the pool will not hand
/// over — owned by a sequence the active list cannot see — is refused with
/// `Ok(false)`, and the survivor stays on its source slot. Nothing is
/// overwritten and the retired sequence still released its own slot.
#[test]
fn retire_leaves_the_survivor_in_place_when_the_pool_refuses_the_vacated_slot() {
    let m = model::recording();
    m.held_slots.lock().unwrap().push(3);
    let mut active = vec![
        active_seq(ResponseSink::Blocking(None), 0),
        active_seq(ResponseSink::Blocking(None), 3),
        active_seq(ResponseSink::Blocking(None), 4),
    ];
    active[1].finished = true;
    let mut cache = vec![false; 3];

    retire_finished_sequences(&m, &mut active, &mut cache);

    assert_eq!(compacts(&m), vec![(4, 3)], "the claim was attempted");
    assert_eq!(
        slots(&active),
        vec![0, 4],
        "refused: survivor keeps its source"
    );
    assert_eq!(
        frees(&m),
        vec![3],
        "the retired sequence's own free still runs"
    );
}

/// A model-level post-copy barrier failure is reported as `Err` with the
/// survivor still bound to its source slot. The scheduler must free only the
/// retired destination and must never sentinel it as a completed migration.
#[test]
fn retire_after_compaction_failure_keeps_survivor_and_frees_destination_owner() {
    use std::sync::atomic::Ordering;

    let m = model::recording();
    m.fail_compact.store(true, Ordering::SeqCst);
    let mut active = vec![
        active_seq(ResponseSink::Blocking(None), 0),
        active_seq(ResponseSink::Blocking(None), 3),
        active_seq(ResponseSink::Blocking(None), 4),
    ];
    active[1].finished = true;
    let mut cache = vec![false; 3];

    retire_finished_sequences(&m, &mut active, &mut cache);

    assert_eq!(compacts(&m), vec![(4, 3)]);
    assert_eq!(slots(&active), vec![0, 4], "survivor keeps its source");
    assert_eq!(
        frees(&m),
        vec![3],
        "only the retired destination owner is freed, and before the attempt"
    );
}

/// Production copy/synchronization errors fail-stop before returning to the
/// scheduler's destination cleanup. Host ownership moves only after success.
#[test]
fn model_compaction_commits_host_ownership_only_after_copy_sync() {
    let source = include_str!("../../../spark-model/src/model/trait_impl/sequence.rs");
    let body = source
        .split("pub(super) fn compact_sequence_dispatch")
        .nth(1)
        .expect("compact_sequence_dispatch source");
    let body: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    let copy = body
        .find(".copy_slot(old_slot,new_slot")
        .expect("device copy");
    let barrier = body
        .find("before_reuse(||self.gpu.synchronize(stream)")
        .expect("copy barrier");
    let host_commit = body.find("seq.slot_idx=new_slot").expect("host commit");
    let source_release = body
        .find("self.ssm_pool.release_slot(old_slot)")
        .expect("source release");

    assert!(copy < barrier);
    assert!(barrier < host_commit);
    assert!(host_commit < source_release);
}

/// The production allocation path must arm exact-slot rollback in the same
/// expression that claims the slot, before any fallible zeroing or state
/// construction, and disarm it only after the complete SequenceState exists.
#[test]
fn model_allocation_arms_rollback_for_every_post_claim_failure() {
    let source = include_str!("../../../spark-model/src/model/trait_impl/meta.rs");
    let helper = source
        .split("fn claim_slot_transaction")
        .nth(1)
        .expect("claim_slot_transaction source");
    let helper = helper
        .split("impl TransformerModel")
        .next()
        .expect("claim helper boundary");
    assert!(helper.contains("let slot = claim()?;"));
    assert!(helper.contains("Ok((slot, ClaimedSlotRollback::new(slot, rollback)))"));

    let body = source
        .split("fn alloc_sequence_inner")
        .nth(1)
        .expect("alloc_sequence_inner source");
    let claim = body
        .find("let (slot, mut slot_rollback) = claim_slot_transaction(")
        .expect("transactional slot claim");
    let first_setup = body
        .find("ple.zero_slot(slot")
        .expect("fallible slot setup");
    let state = body
        .find("let mut state = SequenceState {")
        .expect("completed sequence state");
    let commit = body
        .find("slot_rollback.commit()")
        .expect("rollback disarm");

    assert!(claim < first_setup);
    assert!(first_setup < state);
    assert!(state < commit);
}

/// C1: when the survivor already sits in a lower slot than the retired one,
/// there is nothing to compact; the retired sequence must then release its
/// own slot instead of being sentinel'd (which leaked it).
#[test]
fn retire_frees_the_retired_slot_when_the_survivor_already_sits_lower() {
    let m = model::recording();
    let mut active = vec![
        active_seq(ResponseSink::Blocking(None), 0),
        active_seq(ResponseSink::Blocking(None), 4),
        active_seq(ResponseSink::Blocking(None), 2),
    ];
    active[1].finished = true;
    let mut cache = vec![false; 3];

    retire_finished_sequences(&m, &mut active, &mut cache);

    assert!(compacts(&m).is_empty(), "no upward move, no copy");
    assert_eq!(slots(&active), vec![0, 2], "the survivor keeps its slot");
    assert_eq!(
        frees(&m),
        vec![4],
        "the retired sequence releases its real slot"
    );
}

/// Retiring the last element never compacts; the retired slot is released.
#[test]
fn retire_of_the_last_element_releases_its_slot_without_compaction() {
    let m = model::recording();
    let mut active = vec![
        active_seq(ResponseSink::Blocking(None), 0),
        active_seq(ResponseSink::Blocking(None), 5),
    ];
    active[1].finished = true;
    let mut cache = vec![false; 2];

    retire_finished_sequences(&m, &mut active, &mut cache);

    assert!(compacts(&m).is_empty());
    assert_eq!(slots(&active), vec![0]);
    assert_eq!(frees(&m), vec![5]);
}

/// M1: the receiver thread must park at the pending-queue cap so the bounded
/// tokio request channel eventually reports `Full` to nonblocking HTTP
/// admission.
mod pending_queue {
    use super::*;
    use crate::scheduler::test_fixtures::blocking_request;
    use crate::scheduling_policy::FifoPolicy;
    use std::time::{Duration, Instant};
    use tokio::sync::mpsc::error::TrySendError;

    fn pending() -> Arc<(Mutex<PendingQueue>, Condvar)> {
        Arc::new((
            Mutex::new(PendingQueue {
                requests: Vec::new(),
                closed: false,
            }),
            Condvar::new(),
        ))
    }

    fn queued(p: &Arc<(Mutex<PendingQueue>, Condvar)>) -> usize {
        p.0.lock().requests.len()
    }

    fn wait_until(mut pred: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !pred() {
            if Instant::now() > deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    }

    /// `try_send` with retries: the receiver may be between `recv` and
    /// `push`, in which case a capacity-1 channel is momentarily full.
    fn send(tx: &tokio::sync::mpsc::Sender<InferenceRequest>, mut req: InferenceRequest) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            req = match tx.try_send(req) {
                Ok(()) => return,
                Err(TrySendError::Full(req)) => req,
                Err(TrySendError::Closed(_)) => panic!("request channel closed"),
            };
            assert!(Instant::now() < deadline, "request channel never accepted");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn cap_defaults_to_four_batches_with_a_floor_and_honours_the_override() {
        assert_eq!(pending_queue_cap_with(None, 8), 32);
        assert_eq!(pending_queue_cap_with(None, 16), 64);
        assert_eq!(pending_queue_cap_with(None, 1), 32);
        assert_eq!(pending_queue_cap_with(Some(" 7 "), 16), 7);
        assert_eq!(
            pending_queue_cap_with(Some("0"), 16),
            64,
            "zero is not a cap"
        );
        assert_eq!(pending_queue_cap_with(Some("many"), 16), 64);
    }

    #[test]
    fn receiver_parks_at_capacity_so_the_request_channel_bound_applies() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let p = pending();
        let _receiver = spawn_request_receiver(rx, Arc::clone(&p), 2);
        let mut clients = Vec::new();

        // Two requests fill the pending queue. The third is either parked in
        // the channel or already held by the receiver while it waits to push.
        for i in 0..3 {
            let (req, client) = blocking_request(vec![i]);
            send(&tx, req);
            clients.push(client);
        }
        assert!(wait_until(|| queued(&p) == 2));
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(queued(&p), 2, "the receiver parks at the cap");

        // There can be one extra receiver-owned request between blocking_recv
        // and the bounded pending push. Therefore at most one further request
        // fits in the capacity-1 channel before truthful overload is reported.
        let mut accepted_after_cap = 0usize;
        for i in 3..5 {
            let (req, client) = blocking_request(vec![i]);
            clients.push(client);
            match tx.try_send(req) {
                Ok(()) => accepted_after_cap += 1,
                Err(TrySendError::Full(_)) => break,
                Err(TrySendError::Closed(_)) => panic!("request channel closed"),
            }
        }
        assert!(
            accepted_after_cap <= 1,
            "pending cap + one receiver-held request + channel capacity is the hard bound"
        );

        // Closing plus repeated drains eventually lets the receiver publish
        // every already-admitted request and then mark the queue closed.
        drop(tx);
        while !p.0.lock().closed {
            let _ = drain_pending_requests(&p, &[], &[], &FifoPolicy, 8);
        }
        let _ = drain_pending_requests(&p, &[], &[], &FifoPolicy, 8);
        assert!(p.0.lock().closed);
    }

    #[test]
    fn scheduler_unwind_joins_receiver_and_errors_every_admitted_request() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let p = pending();
        let mut clients = Vec::new();
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _receiver = spawn_request_receiver(rx, Arc::clone(&p), 1);
            // One pending, one receiver-held, and one channel-buffered request.
            for i in 0..3 {
                let (request, client) = blocking_request(vec![i]);
                send(&tx, request);
                clients.push(client);
            }
            assert!(wait_until(|| queued(&p) == 1));
            panic!("injected scheduler failure");
        }));

        assert!(unwind.is_err());
        assert!(
            tx.is_closed(),
            "receiver is dropped and joined before unwind returns"
        );
        assert!(p.0.lock().closed);
        assert!(p.0.lock().requests.is_empty());
        for mut client in clients {
            let error = match client.try_recv().expect("explicit queued error") {
                Err(error) => error,
                Ok(_) => panic!("queued request unexpectedly succeeded"),
            };
            assert!(error.to_string().contains("Scheduler stopped"));
        }
    }

    #[test]
    fn closing_a_full_pending_queue_rejects_the_receiver_held_request() {
        let p = pending();
        let (first, _client) = blocking_request(vec![1]);
        assert!(push_pending_bounded(&p, first, 1).is_none());
        let (second, client) = blocking_request(vec![2]);
        let worker_pending = Arc::clone(&p);
        let worker =
            std::thread::spawn(move || push_pending_bounded(&worker_pending, second, 1).is_some());
        p.0.lock().closed = true;
        p.1.notify_all();
        assert!(worker.join().unwrap());
        assert!(client.blocking_recv().is_err());
        assert_eq!(queued(&p), 1, "nothing is pushed after closure");
    }

    #[test]
    fn sends_racing_receiver_shutdown_never_leave_an_admitted_sink_alive() {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let p = pending();
        let receiver = spawn_request_receiver(rx, p, 1);
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let sender_barrier = Arc::clone(&barrier);
        let sender = std::thread::spawn(move || {
            let mut accepted = Vec::new();
            sender_barrier.wait();
            for i in 0..64 {
                let (request, client) = blocking_request(vec![i]);
                match tx.try_send(request) {
                    Ok(()) => accepted.push(client),
                    Err(TrySendError::Full(_)) => {}
                    Err(TrySendError::Closed(_)) => break,
                }
            }
            accepted
        });
        barrier.wait();
        drop(receiver);
        for mut client in sender.join().unwrap() {
            assert!(matches!(client.try_recv(), Ok(Err(_))));
        }
    }

    #[test]
    fn srpt_pending_drain_orders_shortest_first_with_fifo_tie_break_and_fifo_fallback() {
        let (long_req, _c0) = blocking_request(vec![1; 4_096]);
        let (short_a, _c1) = blocking_request(vec![2; 128]);
        let (mid_req, _c2) = blocking_request(vec![3; 512]);
        let (short_b, _c3) = blocking_request(vec![4; 128]);
        let requests = [long_req, short_a, mid_req, short_b];
        let infos: Vec<PendingRequestInfo> = requests
            .iter()
            .enumerate()
            .map(|(i, r)| PendingRequestInfo {
                prompt_len: r.prompt_len(),
                index: i,
            })
            .collect();

        // Disabled -> preserves strict FifoPolicy order.
        assert_eq!(
            select_pending_with_srpt(&FifoPolicy, &requests, &infos, 3, true, false),
            vec![0, 1, 2]
        );
        // Enabled -> shortest prompts first, equal lengths keep FIFO tie-break (1 before 3).
        assert_eq!(
            select_pending_with_srpt(&FifoPolicy, &requests, &infos, 3, true, true),
            vec![1, 3, 2]
        );
        // Multiple pending while busy (idle_drain = false) also selects shortest first.
        assert_eq!(
            select_pending_with_srpt(&FifoPolicy, &requests, &infos, 2, false, true),
            vec![1, 3]
        );
    }
}
