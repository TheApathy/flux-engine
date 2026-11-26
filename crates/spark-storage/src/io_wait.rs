// SPDX-License-Identifier: AGPL-3.0-only

//! Deadline-bounded `io_uring` completion waits.
//!
//! Returning while fixed-buffer I/O is still live would let the reader reuse
//! or free memory the kernel can still write.  A timed-out wait therefore uses
//! bounded synchronous cancellation.  If the kernel cannot prove cancellation
//! complete, the only memory-safe outcome is fail-stop so the OS reclaims the
//! whole address space.

use std::time::Duration;

use anyhow::{Context, Result};
use io_uring::IoUring;
use io_uring::types::{CancelBuilder, SubmitArgs, Timespec};

const CANCEL_GRACE: Duration = Duration::from_secs(2);

fn fail_stop(operation: &str, detail: &str) -> ! {
    eprintln!(
        "fatal: {operation}: {detail}; exiting without unwinding before fixed I/O buffers can be reused"
    );
    std::process::exit(70)
}

fn require_empty_submission_queue(ring: &mut IoUring, operation: &str) {
    let pending = ring.submission().len();
    if pending != 0 {
        // `IORING_REGISTER_SYNC_CANCEL` does not cover SQEs that userspace
        // published but the kernel has not consumed. There is no safe return
        // while those entries still name fixed buffers the caller may reuse.
        fail_stop(
            operation,
            &format!("{pending} submission queue entries were not consumed"),
        );
    }
}

/// Cancel every submitted request and drain its CQE before fixed buffers are
/// reused. Any inability to prove that state terminates without unwinding.
pub(crate) fn cancel_and_drain_or_exit(ring: &mut IoUring, operation: &str) {
    require_empty_submission_queue(ring, operation);
    let cancel_timeout = Timespec::from(CANCEL_GRACE);
    let cancellation = ring
        .submitter()
        .register_sync_cancel(Some(cancel_timeout), CancelBuilder::any());
    if let Err(error) = cancellation
        && error.raw_os_error() != Some(libc::ENOENT)
    {
        fail_stop(
            operation,
            &format!("io_uring cancellation was not proven within {CANCEL_GRACE:?}: {error}"),
        );
    }
    require_empty_submission_queue(ring, operation);

    // Cancellation completed, or ENOENT proved there was nothing submitted.
    // Either way all CQEs are now safe to consume before returning ownership.
    for _ in ring.completion() {}
}

pub(crate) fn submit_and_wait_bounded(
    ring: &mut IoUring,
    want: usize,
    timeout: Duration,
    operation: &str,
) -> Result<()> {
    if want == 0 {
        require_empty_submission_queue(ring, operation);
        anyhow::bail!("{operation}: completion count must be positive");
    }
    if timeout.is_zero() {
        require_empty_submission_queue(ring, operation);
        anyhow::bail!("{operation}: I/O timeout must be positive");
    }
    if !ring.params().is_feature_ext_arg() {
        require_empty_submission_queue(ring, operation);
        anyhow::bail!("{operation}: kernel lacks IORING_FEAT_EXT_ARG required for bounded waits");
    }

    let timeout_spec = Timespec::from(timeout);
    let args = SubmitArgs::new().timespec(&timeout_spec);
    let wait = ring.submitter().submit_with_args(want, &args);
    require_empty_submission_queue(ring, operation);
    let completed = ring.completion().len();
    if wait.is_ok() && completed >= want {
        return Ok(());
    }

    let wait_detail = wait.err().map_or_else(
        || format!("only {completed}/{want} completions arrived"),
        |e| e.to_string(),
    );
    cancel_and_drain_or_exit(ring, operation);
    Err(anyhow::anyhow!(
        "{operation} exceeded its {timeout:?} deadline ({wait_detail}); all in-flight reads were cancelled"
    ))
    .context("bounded io_uring wait")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_deadline_before_submission() {
        let mut ring = match IoUring::new(2) {
            Ok(ring) => ring,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(error) => panic!("create test ring: {error}"),
        };
        let error = submit_and_wait_bounded(&mut ring, 1, Duration::ZERO, "test")
            .unwrap_err()
            .to_string();
        assert!(error.contains("I/O timeout must be positive"));
    }

    #[test]
    fn completed_operation_remains_available_to_caller() {
        let mut ring = match IoUring::new(2) {
            Ok(ring) => ring,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(error) => panic!("create test ring: {error}"),
        };
        let entry = io_uring::opcode::Nop::new().build().user_data(77);
        unsafe { ring.submission().push(&entry) }.unwrap();
        submit_and_wait_bounded(&mut ring, 1, Duration::from_secs(1), "test nop").unwrap();
        let completion = ring.completion().next().expect("NOP completion retained");
        assert_eq!(completion.user_data(), 77);
        assert_eq!(completion.result(), 0);
    }
}
