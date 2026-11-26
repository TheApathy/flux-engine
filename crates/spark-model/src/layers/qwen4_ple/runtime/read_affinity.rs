// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_PLE_READ_AFFINITY` (K10 item A, default off) at the two PLE prefill
//! read threads (`ple-prefill-blocks`, `ple-prefill-read`): the plan lives in
//! `spark_storage::host_affinity`; this logs its receipt once, applies it to
//! the calling read thread, and (when on) logs where each read ran:
//! `PLE_READ_PLACEMENT mode= start_cpu= end_cpu= read_ms= selections=`. Under
//! `ATLAS_K10A_TRIGGER` a pinning plan pins only while the trigger file exists.
//! Scheduling only: the reads, their order and every copied byte are unchanged.

use spark_storage::host_affinity::{self, Plan};

/// Placement of one read, captured at its start when the plan observes.
pub(super) struct ReadPlacement {
    mode: &'static str,
    start_cpu: usize,
    t0: std::time::Instant,
}

/// Apply the process plan to this (read) thread. Call first thing in the thread.
pub(super) fn begin() -> Option<ReadPlacement> {
    let (plan, first) = host_affinity::process_plan();
    if first && let Some(receipt) = plan.receipt() {
        if matches!(plan, Plan::Refused { .. }) {
            tracing::warn!(target: "atlas::qwen4_ple", "{receipt}");
        } else {
            tracing::info!(target: "atlas::qwen4_ple", "{receipt}");
        }
    }
    // ATLAS_K10A_TRIGGER (in-process A/B/A): with the trigger configured and its
    // file absent, the read is observed (placement logged) but not pinned.
    static OBSERVE: Plan = Plan::Observe;
    let plan = if matches!(plan, Plan::Pin { .. }) && !crate::model::k10a_host::trigger_present() {
        &OBSERVE
    } else {
        plan
    };
    let mode = match plan {
        Plan::Off => return None,
        Plan::Observe => "observe",
        Plan::Pin { mode, .. } => mode.name(),
        Plan::Refused { .. } => "refused",
    };
    match host_affinity::apply_to_current_thread(plan) {
        Ok(cpu) => cpu.map(|start_cpu| ReadPlacement {
            mode,
            start_cpu,
            t0: std::time::Instant::now(),
        }),
        Err(error) => {
            tracing::warn!(
                target: "atlas::qwen4_ple",
                "PLE_READ_AFFINITY_REFUSED selector={} reason=pin_failed error={error} (this read runs unpinned)",
                host_affinity::SELECTOR
            );
            None
        }
    }
}

/// Log where the read ended (call last thing in the read thread).
pub(super) fn end(placement: Option<ReadPlacement>, selections: usize) {
    if let Some(p) = placement {
        tracing::info!(
            target: "atlas::qwen4_ple",
            "PLE_READ_PLACEMENT mode={} start_cpu={} end_cpu={} read_ms={:.1} selections={selections}",
            p.mode,
            p.start_cpu,
            host_affinity::current_cpu().map_or_else(|| "?".to_string(), |c| c.to_string()),
            p.t0.elapsed().as_secs_f64() * 1e3,
        );
    }
}
