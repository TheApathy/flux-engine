// SPDX-License-Identifier: AGPL-3.0-only

//! Unresolved verify rollback makes the existing serving fault latch fatal.

use anyhow::Result;
use atlas_core::fault::FaultLatch;

const ABORT_FAULT: &str = "speculative verify rollback failed; engine restart required";
const OWNER_FAULT: &str = "speculative verify ownership failed; engine restart required";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AbortOwner {
    pub slot: usize,
    pub session: u64,
    pub cursor: usize,
    pub rows: usize,
}

fn check_latch(latch: &FaultLatch) -> Result<()> {
    anyhow::ensure!(
        !latch.is_faulted(),
        "engine is unavailable after fatal execution fault; restart required"
    );
    Ok(())
}

pub(super) fn check() -> Result<()> {
    check_latch(atlas_core::fault::global())
}

pub(super) fn poison_abort(owner: AbortOwner) {
    // Health, liveness, admission, and active workloads share this one latch.
    let first = atlas_core::fault::global().latch(ABORT_FAULT);
    tracing::error!(
        first,
        slot = owner.slot,
        session_hash = owner.session,
        target_position = owner.cursor,
        actual_rows = owner.rows,
        "VERIFY_ABORT_FATAL"
    );
}

pub(super) fn poison_owner() {
    let first = atlas_core::fault::global().latch(OWNER_FAULT);
    tracing::error!(first, "VERIFY_OWNER_FATAL");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_process_admits_work_before_any_fatal_fault() {
        let latch = FaultLatch::new();
        for _ in 0..8 {
            assert!(check_latch(&latch).is_ok());
        }
    }

    #[test]
    fn fatal_rollback_blocks_all_later_work_and_preserves_first_reason() {
        let latch = FaultLatch::new();
        assert!(latch.latch(ABORT_FAULT));
        for slot in 0..8 {
            assert!(check_latch(&latch).is_err());
            assert!(!latch.latch(format!("later slot {slot}")));
            assert_eq!(latch.fault(), Some(ABORT_FAULT));
        }
    }

    #[test]
    fn preexisting_process_fault_also_blocks_model_work() {
        let latch = FaultLatch::new();
        assert!(latch.latch("scheduler owner exited"));
        assert!(check_latch(&latch).is_err());
        assert!(!latch.latch(ABORT_FAULT));
        assert_eq!(latch.fault(), Some("scheduler owner exited"));
    }

    #[test]
    fn failure_response_omits_private_root_diagnostics() {
        let latch = FaultLatch::new();
        latch.latch("private diagnostic detail");
        assert_eq!(
            check_latch(&latch).unwrap_err().to_string(),
            "engine is unavailable after fatal execution fault; restart required"
        );
    }

    #[test]
    fn owner_fault_reaches_global_admission_and_preserves_first_reason() {
        const CHILD_MARKER: &str = "FLUX_VERIFY_OWNER_FAULT_TEST_CHILD";
        let (_, module) = module_path!()
            .split_once("::")
            .expect("test module must have a crate prefix");
        let name =
            format!("{module}::owner_fault_reaches_global_admission_and_preserves_first_reason");
        if std::env::var(CHILD_MARKER).as_deref() != Ok(name.as_str()) {
            // A process-global latch must never poison the parent test suite.
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name.as_str(), "--test-threads=1", "--nocapture"])
                .env(CHILD_MARKER, &name)
                .output()
                .expect("isolated owner fault regression must start");
            assert!(
                output.status.success(),
                "isolated owner fault regression failed: {output:?}"
            );
            assert!(
                String::from_utf8_lossy(&output.stdout)
                    .contains("test result: ok. 1 passed; 0 failed;"),
                "isolated owner fault regression did not run exactly one passing test: {output:?}"
            );
            return;
        }
        assert!(!atlas_core::fault::global().is_faulted());
        poison_owner();
        assert!(check().is_err());
        assert_eq!(atlas_core::fault::global().fault(), Some(OWNER_FAULT));
        poison_abort(AbortOwner {
            slot: 0,
            session: 0,
            cursor: 8192,
            rows: 5,
        });
        assert!(check().is_err());
        assert_eq!(atlas_core::fault::global().fault(), Some(OWNER_FAULT));
    }

    #[test]
    fn local_owner_fault_rejects_work_and_wins_over_later_abort() {
        let latch = FaultLatch::new();
        assert!(latch.latch(OWNER_FAULT));
        assert!(check_latch(&latch).is_err());
        assert!(!latch.latch(ABORT_FAULT));
        assert_eq!(latch.fault(), Some(OWNER_FAULT));
    }

    #[test]
    fn local_prior_fault_survives_later_owner_failure() {
        let latch = FaultLatch::new();
        assert!(latch.latch("prior execution fault"));
        assert!(!latch.latch(OWNER_FAULT));
        assert!(check_latch(&latch).is_err());
        assert_eq!(latch.fault(), Some("prior execution fault"));
    }

    #[test]
    fn every_thread_observes_the_same_fatal_process_state() {
        let latch = std::sync::Arc::new(FaultLatch::new());
        latch.latch(ABORT_FAULT);
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let latch = latch.clone();
                std::thread::spawn(move || assert!(check_latch(&latch).is_err()))
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
    }
}
