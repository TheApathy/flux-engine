// SPDX-License-Identifier: AGPL-3.0-only

//! Concurrency gate for large tokenizer inputs.
//!
//! `tokenizers` allocates transient workspace proportional to input size for
//! the duration of an `encode` / `encode_fast` call. Bounding concurrent
//! `>= 1 MiB` encodes keeps multi-client long-context traffic from multiplying
//! peak host RSS while leaving ordinary `< 1 MiB` prompts completely lock-free.

use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

/// Inputs below 1 MiB bypass the gate without touching the mutex.
pub(super) const LARGE_INPUT_BYTES: usize = 1024 * 1024;

/// Default cap on simultaneous `>= 1 MiB` tokenizer encodes.
pub(super) const DEFAULT_LARGE_ENCODE_CONCURRENCY: usize = 2;

pub(super) struct Gate {
    limit: usize,
    active: Mutex<usize>,
    available: Condvar,
}

pub(super) struct Permit<'a> {
    gate: &'a Gate,
}

impl Gate {
    pub(super) fn new(limit: usize) -> Self {
        Self {
            limit: limit.max(1),
            active: Mutex::new(0),
            available: Condvar::new(),
        }
    }

    pub(super) fn default_gate() -> Arc<Self> {
        Arc::new(Self::new(DEFAULT_LARGE_ENCODE_CONCURRENCY))
    }

    pub(super) fn acquire(&self, bytes: usize) -> Option<Permit<'_>> {
        if bytes < LARGE_INPUT_BYTES {
            return None;
        }
        let mut active = self.active.lock();
        while *active >= self.limit {
            self.available.wait(&mut active);
        }
        *active += 1;
        Some(Permit { gate: self })
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut active = self.gate.active.lock();
        *active = active.saturating_sub(1);
        drop(active);
        self.gate.available.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn small_inputs_bypass_saturated_gate() {
        let gate = Gate::new(1);
        let held = gate.acquire(LARGE_INPUT_BYTES);
        assert!(held.is_some());

        // Inputs strictly below 1 MiB do not contend on the gate even when all
        // large-encode slots are occupied.
        assert!(gate.acquire(0).is_none());
        assert!(gate.acquire(LARGE_INPUT_BYTES - 1).is_none());
    }

    #[test]
    fn large_encodes_block_past_limit_and_wake_on_drop() {
        let gate = Gate::new(2);
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let third_entered = AtomicUsize::new(0);
        let two_inside = Barrier::new(3);
        let release_first_two = Barrier::new(3);

        std::thread::scope(|s| {
            for _ in 0..2 {
                s.spawn(|| {
                    let _permit = gate.acquire(LARGE_INPUT_BYTES).unwrap();
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    two_inside.wait();
                    release_first_two.wait();
                    active.fetch_sub(1, Ordering::SeqCst);
                });
            }

            two_inside.wait();
            assert_eq!(active.load(Ordering::SeqCst), 2);

            let third = s.spawn(|| {
                let _permit = gate.acquire(LARGE_INPUT_BYTES).unwrap();
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                third_entered.store(1, Ordering::SeqCst);
                active.fetch_sub(1, Ordering::SeqCst);
            });

            std::thread::sleep(Duration::from_millis(25));
            assert_eq!(
                third_entered.load(Ordering::SeqCst),
                0,
                "third >=1 MiB encode must wait while two permits are held"
            );

            // Small inputs still bypass while the third large encode waits.
            assert!(gate.acquire(1024).is_none());

            release_first_two.wait();
            third.join().unwrap();
        });

        assert_eq!(peak.load(Ordering::SeqCst), 2);
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn error_and_panic_unwind_release_permits() {
        let gate = Gate::new(1);

        let fallible: Result<(), &'static str> = {
            let _permit = gate.acquire(LARGE_INPUT_BYTES).unwrap();
            Err("simulated encode failure")
        };
        assert!(fallible.is_err());

        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _permit = gate.acquire(LARGE_INPUT_BYTES).unwrap();
            panic!("simulated encode panic");
        }));
        assert!(unwound.is_err());

        // Permit must be available again after both early error return and unwind.
        let permit = gate.acquire(LARGE_INPUT_BYTES);
        assert!(permit.is_some());
    }
}
