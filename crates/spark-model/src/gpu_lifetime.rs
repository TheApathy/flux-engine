// SPDX-License-Identifier: AGPL-3.0-only

//! A failed completion barrier is not permission to recycle device storage.

use anyhow::Result;

/// Fail without unwinding: destructors could otherwise publish or free buffers
/// still referenced by queued work. This policy also applies in unwind builds.
pub(crate) fn require_completion<T>(result: Result<T>, stage: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            eprintln!("fatal GPU lifetime failure at {stage}: {error:#}");
            std::process::abort();
        }
    }
}

pub(crate) fn before_reuse<T>(
    complete: impl FnOnce() -> Result<()>,
    reuse: impl FnOnce() -> T,
    stage: &str,
) -> T {
    require_completion(complete(), stage);
    reuse()
}

#[cfg(test)]
mod tests {
    use super::{before_reuse, require_completion};

    #[test]
    fn successful_completion_preserves_the_result() {
        assert_eq!(require_completion(Ok(7), "fixture"), 7);
        assert_eq!(before_reuse(|| Ok(()), || 9, "fixture"), 9);
    }

    #[test]
    fn production_slot_release_and_compaction_use_the_tested_completion_boundary() {
        let source = include_str!("model/trait_impl/sequence.rs");
        let normalized: String = source.chars().filter(|c| !c.is_whitespace()).collect();
        // free: the slot is released only inside the completion barrier's reuse
        // closure (after the native MTP hidden-frame retirement it also guards).
        let free = normalized.split("fnfree_sequence_dispatch").nth(1).unwrap();
        let free_barrier = free
            .find("before_reuse(||self.gpu.synchronize(stream),||{")
            .unwrap();
        let free_publish = free.find("self.ssm_pool.release_slot(slot)").unwrap();
        assert!(free_barrier < free_publish);
        assert_eq!(free.matches("self.ssm_pool.release_slot(slot)").count(), 1);
        let compact = normalized
            .split("fncompact_sequence_dispatch")
            .nth(1)
            .unwrap();
        let barrier = compact
            .find("before_reuse(||self.gpu.synchronize(stream),||{")
            .unwrap();
        let publish = compact
            .find("self.ssm_pool.release_slot(old_slot)")
            .unwrap();
        assert!(barrier < publish);
    }

    #[cfg(unix)]
    #[test]
    fn failed_barrier_never_returns_to_slot_release() {
        const CHILD: &str = "ATLAS_TEST_FAILED_GPU_COMPLETION_CHILD";
        if std::env::var_os(CHILD).is_some() {
            before_reuse(
                || Err(anyhow::anyhow!("injected copy barrier")),
                || eprintln!("UNSAFE_SLOT_RELEASE"),
                "fixture",
            );
            return;
        }
        use std::os::unix::process::ExitStatusExt;
        let output = std::process::Command::new("sh")
            .args(["-c", "ulimit -c 0; exec \"$@\"", "atlas-gpu-lifetime-test"])
            .arg(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "gpu_lifetime::tests::failed_barrier_never_returns_to_slot_release",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert_eq!(output.status.signal(), Some(6));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("injected copy barrier"));
        assert!(!stderr.contains("UNSAFE_SLOT_RELEASE"));
    }
}
