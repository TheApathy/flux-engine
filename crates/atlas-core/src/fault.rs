// SPDX-License-Identifier: AGPL-3.0-only

//! Process-fatal execution fault latch.
//!
//! Some failures make the serving process permanently unable to answer new
//! inference requests. A scheduler-thread panic is one: the sole owner of
//! model execution is gone while the HTTP listener may still be alive. The
//! first failure is the useful diagnostic, so this latch is one-shot and
//! first-writer-wins.

use std::sync::OnceLock;

/// A one-shot fault flag whose reason is published atomically.
///
/// `OnceLock<String>` makes "faulted" and "has a reason" the same state and
/// guarantees that later fault echoes cannot overwrite the original cause.
#[derive(Debug, Default)]
pub struct FaultLatch {
    reason: OnceLock<String>,
}

impl FaultLatch {
    pub const fn new() -> Self {
        Self {
            reason: OnceLock::new(),
        }
    }

    /// Record a fatal fault. Returns `true` only for the first writer.
    pub fn latch(&self, reason: impl Into<String>) -> bool {
        self.reason.set(reason.into()).is_ok()
    }

    /// The first fatal reason, or `None` while the process is healthy.
    pub fn fault(&self) -> Option<&str> {
        self.reason.get().map(String::as_str)
    }

    pub fn is_faulted(&self) -> bool {
        self.reason.get().is_some()
    }
}

static GLOBAL: FaultLatch = FaultLatch::new();

/// The process-wide serving fault latch.
pub fn global() -> &'static FaultLatch {
    &GLOBAL
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::FaultLatch;

    #[test]
    fn first_writer_publishes_reason_atomically() {
        let latch = FaultLatch::new();
        assert!(!latch.is_faulted());
        assert!(latch.latch("original fault"));
        assert!(latch.is_faulted());
        assert_eq!(latch.fault(), Some("original fault"));
        assert!(!latch.latch("later echo"));
        assert_eq!(latch.fault(), Some("original fault"));
    }

    #[test]
    fn concurrent_faults_have_exactly_one_first_writer() {
        let latch = Arc::new(FaultLatch::new());
        let barrier = Arc::new(Barrier::new(8));
        let writers: Vec<_> = (0..8)
            .map(|index| {
                let latch = Arc::clone(&latch);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    latch.latch(format!("fault-{index}"))
                })
            })
            .collect();
        let winners = writers
            .into_iter()
            .map(|writer| writer.join().expect("fault writer"))
            .filter(|won| *won)
            .count();

        assert_eq!(winners, 1);
        assert!(
            latch
                .fault()
                .is_some_and(|reason| reason.starts_with("fault-"))
        );
    }
}
