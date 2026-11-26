// SPDX-License-Identifier: AGPL-3.0-only

//! `--max-queue-depth`: an exact bound on requests waiting to start.
//!
//! Running requests are bounded by `--max-batch-size` (drain) and by the KV
//! pool (depth-aware admission returns overflow to the pending queue). This
//! gate bounds the rest: a request enters when HTTP admission hands it to the
//! scheduler channel and leaves when `gate_admissions` admits it (it starts,
//! or fails on its own) or when the scheduler rejects it on shutdown. While
//! `waiting == limit`, HTTP admission answers 429 instead of queueing.
//!
//! Disabled (the default) it never counts, so the only bound is the existing
//! `--max-num-seqs` channel plus the pending-queue cap.

use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) struct QueueGate {
    /// 0 = disabled.
    limit: AtomicUsize,
    waiting: AtomicUsize,
}

impl QueueGate {
    pub(crate) const fn new() -> Self {
        Self {
            limit: AtomicUsize::new(0),
            waiting: AtomicUsize::new(0),
        }
    }

    /// Set at model load, before the scheduler for that model accepts work.
    /// The previous scheduler (if any) has already rejected its queue.
    pub(crate) fn configure(&self, limit: Option<usize>) {
        self.waiting.store(0, Ordering::SeqCst);
        self.limit.store(limit.unwrap_or(0), Ordering::SeqCst);
    }

    pub(crate) fn limit(&self) -> Option<usize> {
        match self.limit.load(Ordering::SeqCst) {
            0 => None,
            n => Some(n),
        }
    }

    pub(crate) fn waiting(&self) -> usize {
        self.waiting.load(Ordering::SeqCst)
    }

    /// Reserve one waiting place. `Err(limit)` when the queue is full.
    /// Always `Ok` (and uncounted) when disabled.
    pub(crate) fn try_enter(&self) -> Result<(), usize> {
        let Some(limit) = self.limit() else {
            return Ok(());
        };
        self.waiting
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |w| {
                (w < limit).then_some(w + 1)
            })
            .map(|_| ())
            .map_err(|_| limit)
    }

    /// `n` requests stopped waiting. Saturates, so leaving while disabled (or
    /// after a reconfigure) can never underflow.
    pub(crate) fn leave(&self, n: usize) {
        if n == 0 || self.limit().is_none() {
            return;
        }
        let _ = self
            .waiting
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |w| {
                Some(w.saturating_sub(n))
            });
    }
}

/// The process's gate; configured by `serve_load` from `--max-queue-depth`.
pub(crate) static QUEUE_GATE: QueueGate = QueueGate::new();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_gate_never_counts_or_refuses() {
        let gate = QueueGate::new();
        for _ in 0..1000 {
            assert_eq!(gate.try_enter(), Ok(()));
        }
        assert_eq!(gate.waiting(), 0);
        gate.leave(5);
        assert_eq!(gate.waiting(), 0);
        assert_eq!(gate.limit(), None);
    }

    #[test]
    fn full_queue_refuses_until_a_request_leaves() {
        let gate = QueueGate::new();
        gate.configure(Some(2));
        assert_eq!(gate.try_enter(), Ok(()));
        assert_eq!(gate.try_enter(), Ok(()));
        assert_eq!(gate.try_enter(), Err(2));
        assert_eq!(gate.waiting(), 2);
        gate.leave(1);
        assert_eq!(gate.try_enter(), Ok(()));
        assert_eq!(gate.try_enter(), Err(2));
    }

    #[test]
    fn many_enter_leave_cycles_do_not_leak() {
        let gate = QueueGate::new();
        gate.configure(Some(4));
        for cycle in 0..10_000 {
            let entered = (0..6).filter(|_| gate.try_enter().is_ok()).count();
            assert_eq!(entered, 4, "cycle {cycle}");
            gate.leave(entered);
            assert_eq!(gate.waiting(), 0, "cycle {cycle}");
        }
    }

    #[test]
    fn leave_saturates_and_reconfigure_resets() {
        let gate = QueueGate::new();
        gate.configure(Some(3));
        gate.try_enter().unwrap();
        gate.leave(10);
        assert_eq!(gate.waiting(), 0);
        gate.try_enter().unwrap();
        gate.configure(Some(1));
        assert_eq!(gate.waiting(), 0);
        gate.try_enter().unwrap();
        assert_eq!(gate.try_enter(), Err(1));
        gate.configure(None);
        assert_eq!(gate.try_enter(), Ok(()));
    }

    #[test]
    fn concurrent_entries_never_exceed_the_limit() {
        let gate = std::sync::Arc::new(QueueGate::new());
        gate.configure(Some(7));
        let admitted = std::sync::Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let gate = std::sync::Arc::clone(&gate);
                let admitted = std::sync::Arc::clone(&admitted);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        if gate.try_enter().is_ok() {
                            admitted.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(admitted.load(Ordering::SeqCst), 7);
        assert_eq!(gate.waiting(), 7);
    }
}
