// SPDX-License-Identifier: AGPL-3.0-only

//! Host-visible lifetime barriers for asynchronous prefill work.

use anyhow::{Context, Result};
use spark_model::traits::Model;

trait PrefillStreamIo {
    fn default_stream(&self) -> u64;
    fn synchronize_stream(&self, stream: u64) -> Result<()>;
    fn record_event(&self, event: u64, stream: u64) -> Result<()>;
    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()>;
}

impl<T: Model + ?Sized> PrefillStreamIo for T {
    fn default_stream(&self) -> u64 {
        Model::default_stream(self)
    }

    fn synchronize_stream(&self, stream: u64) -> Result<()> {
        Model::synchronize_stream(self, stream)
    }

    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        Model::record_event(self, event, stream)
    }

    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        Model::stream_wait_event(self, stream, event)
    }
}

fn try_blocking_drain(model: &(impl PrefillStreamIo + ?Sized), prefill_stream: u64) -> Result<()> {
    model
        .synchronize_stream(prefill_stream)
        .context("prefill stream drain failed")?;
    let default_stream = model.default_stream();
    if default_stream != prefill_stream {
        model
            .synchronize_stream(default_stream)
            .context("default stream drain failed")?;
    }
    Ok(())
}

fn try_success_handoff(
    model: &(impl PrefillStreamIo + ?Sized),
    prefill_stream: u64,
    prefill_event: u64,
) -> Result<bool> {
    let default_stream = model.default_stream();
    let event_handoff = model
        .record_event(prefill_event, prefill_stream)
        .and_then(|()| model.stream_wait_event(default_stream, prefill_event));
    match event_handoff {
        Ok(()) => Ok(false),
        Err(event_error) => {
            tracing::warn!(
                error = %event_error,
                "Prefill event handoff failed; using a blocking stream drain"
            );
            try_blocking_drain(model, prefill_stream)
                .context("blocking fallback after prefill event failure")?;
            Ok(true)
        }
    }
}

#[cold]
fn abort_after_barrier_failure(stage: &str, error: &anyhow::Error) -> ! {
    tracing::error!(
        stage,
        error = format!("{error:#}"),
        "Prefill stream barrier failed; aborting process"
    );
    std::process::abort()
}

pub(super) fn drain_before_prefill_cleanup(model: &dyn Model, prefill_stream: u64, stage: &str) {
    if let Err(error) = try_blocking_drain(model, prefill_stream) {
        abort_after_barrier_failure(stage, &error);
    }
}

pub(super) fn complete_prefill_handoff(
    model: &dyn Model,
    prefill_stream: u64,
    prefill_event: u64,
    stage: &str,
) {
    if let Err(error) = try_success_handoff(model, prefill_stream, prefill_event) {
        abort_after_barrier_failure(stage, &error);
    }
}

pub(super) fn abort_after_mixed_forward_error(
    model: &dyn Model,
    prefill_stream: u64,
    stage: &str,
    forward_error: &anyhow::Error,
) -> ! {
    if let Err(drain_error) = try_blocking_drain(model, prefill_stream) {
        tracing::error!(
            stage,
            error = %forward_error,
            drain_error = %drain_error,
            "Mixed forward failed and its stream drain also failed; aborting process"
        );
    } else {
        tracing::error!(
            stage,
            error = %forward_error,
            "Mixed forward may have mutated decode state; streams drained, aborting process"
        );
    }
    std::process::abort()
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use anyhow::{Result, bail};

    use super::{PrefillStreamIo, try_blocking_drain, try_success_handoff};

    struct FakeIo {
        default: u64,
        record_fails: bool,
        wait_fails: bool,
        calls: Mutex<Vec<String>>,
    }

    impl FakeIo {
        fn new(default: u64) -> Self {
            Self {
                default,
                record_fails: false,
                wait_fails: false,
                calls: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl PrefillStreamIo for FakeIo {
        fn default_stream(&self) -> u64 {
            self.default
        }

        fn synchronize_stream(&self, stream: u64) -> Result<()> {
            self.calls.lock().unwrap().push(format!("sync:{stream}"));
            Ok(())
        }

        fn record_event(&self, event: u64, stream: u64) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("record:{event}:{stream}"));
            if self.record_fails {
                bail!("record failed");
            }
            Ok(())
        }

        fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("wait:{stream}:{event}"));
            if self.wait_fails {
                bail!("wait failed");
            }
            Ok(())
        }
    }

    #[test]
    fn blocking_drain_orders_prefill_before_distinct_default() {
        let io = FakeIo::new(9);
        try_blocking_drain(&io, 7).unwrap();
        assert_eq!(io.calls(), ["sync:7", "sync:9"]);
    }

    #[test]
    fn blocking_drain_deduplicates_one_stream() {
        let io = FakeIo::new(7);
        try_blocking_drain(&io, 7).unwrap();
        assert_eq!(io.calls(), ["sync:7"]);
    }

    #[test]
    fn event_record_failure_falls_back_to_blocking_drain() {
        let mut io = FakeIo::new(9);
        io.record_fails = true;
        assert!(try_success_handoff(&io, 7, 11).unwrap());
        assert_eq!(io.calls(), ["record:11:7", "sync:7", "sync:9"]);
    }

    #[test]
    fn event_wait_failure_falls_back_to_blocking_drain() {
        let mut io = FakeIo::new(9);
        io.wait_fails = true;
        assert!(try_success_handoff(&io, 7, 11).unwrap());
        assert_eq!(io.calls(), ["record:11:7", "wait:9:11", "sync:7", "sync:9"]);
    }
}
