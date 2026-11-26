// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded scheduler retirement.
//!
//! A scheduler owns the model until its thread exits.  Dropping its
//! [`std::thread::JoinHandle`] only detaches it; it does not make GPU teardown
//! safe.  Production callers therefore poll the handle to a deadline and join
//! only after `is_finished()` proves the join cannot become an unbounded wait.

use std::any::Any;
use std::time::{Duration, Instant};

use anyhow::Result;

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&'static str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

/// Overrides the shutdown retirement grace, in whole seconds. A long-context
/// prefill chunk (W16384 near 300K tokens) can run longer than the default.
pub(super) const RETIRE_GRACE_ENV: &str = "ATLAS_SCHEDULER_RETIRE_GRACE_SECS";

fn parse_retire_grace(raw: Option<&str>, default: Duration) -> Duration {
    match raw.map(str::trim) {
        None | Some("") => default,
        Some(value) => match value.parse::<u64>() {
            Ok(secs) if secs > 0 => Duration::from_secs(secs),
            _ => {
                tracing::warn!(
                    "{RETIRE_GRACE_ENV}={value:?} is not a positive integer; using {}s",
                    default.as_secs()
                );
                default
            }
        },
    }
}

/// The shutdown retirement grace: `default` unless [`RETIRE_GRACE_ENV`] sets it.
pub(super) fn retire_grace(default: Duration) -> Duration {
    parse_retire_grace(std::env::var(RETIRE_GRACE_ENV).ok().as_deref(), default)
}

/// Wait no longer than `grace` for the scheduler to finish.
pub(super) fn wait_until_finished(handle: &std::thread::JoinHandle<()>, grace: Duration) -> bool {
    let deadline = Instant::now() + grace;
    while !handle.is_finished() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    handle.is_finished()
}

/// Join a scheduler already proven finished, returning ownership unchanged if
/// the proof is absent.
pub(super) fn join_finished(
    handle: std::thread::JoinHandle<()>,
) -> std::result::Result<Result<()>, std::thread::JoinHandle<()>> {
    if !handle.is_finished() {
        return Err(handle);
    }
    if let Err(payload) = handle.join() {
        return Ok(Err(anyhow::anyhow!(
            "scheduler thread panicked: {}",
            panic_message(&*payload)
        )));
    }
    Ok(Ok(()))
}

/// Terminate without unwinding when a live scheduler prevents safe teardown.
///
/// `_exit` is intentional: unwinding would drop the host/GPU while the
/// scheduler may still launch kernels, and `process::exit` runs the CUDA
/// runtime's atexit teardown underneath that live thread, whose next stream
/// sync then fails and aborts (SIGABRT instead of this exit code). The
/// OS/supervisor tears down the CUDA context and starts a clean process.
pub(super) fn fail_stop(message: &str) -> ! {
    // Avoid tracing/tee locks here: the thread that failed to retire may hold
    // one. Terminal restoration is lock-free at its lifecycle boundary and is
    // the only best-effort cleanup before non-unwinding exit.
    crate::tui::terminal_guard::restore();
    eprintln!("fatal scheduler retirement failure: {message}");
    // SAFETY: `_exit` takes no pointers and never returns; skipping atexit
    // handlers and destructors is the point (see above).
    unsafe { libc::_exit(70) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn non_exiting_scheduler_is_bounded_and_not_joined() {
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            let _ = release_rx.recv();
        });
        let began = Instant::now();
        assert!(!wait_until_finished(&handle, Duration::from_millis(30)));
        assert!(began.elapsed() < Duration::from_secs(1));
        let handle = join_finished(handle).expect_err("running handle is returned intact");
        drop(release_tx);
        assert!(wait_until_finished(&handle, Duration::from_secs(1)));
        join_finished(handle).unwrap().unwrap();
    }

    #[test]
    fn retire_grace_accepts_only_positive_whole_seconds() {
        let default = Duration::from_secs(30);
        assert_eq!(parse_retire_grace(None, default), default);
        assert_eq!(parse_retire_grace(Some(""), default), default);
        assert_eq!(
            parse_retire_grace(Some("300"), default),
            Duration::from_secs(300)
        );
        for bad in ["0", "-5", "1.5", "ten"] {
            assert_eq!(parse_retire_grace(Some(bad), default), default, "{bad}");
        }
    }

    #[test]
    fn finished_scheduler_can_be_joined_and_panics_are_reported() {
        let ok = std::thread::spawn(|| {});
        assert!(wait_until_finished(&ok, Duration::from_secs(1)));
        join_finished(ok).unwrap().unwrap();

        let panic = std::thread::spawn(|| panic!("retire-test"));
        assert!(wait_until_finished(&panic, Duration::from_secs(1)));
        let error = join_finished(panic).unwrap().unwrap_err().to_string();
        assert!(error.contains("scheduler thread panicked"));
        assert!(error.contains("retire-test"));
    }
}
