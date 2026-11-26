// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_PLE_READ_AFFINITY` (K10 item A, default off): where the PLE prefill
//! read thread runs.
//!
//! The chunk-0 PLE read is CPU-bound in its submitting thread (K9 W, micro 155:
//! thread CPU = wall). Each read runs on a freshly spawned thread
//! (`ple-prefill-blocks`; W's ring threads are spawned inside it), and GB10's
//! CPU is asymmetric: ten Cortex-X925 (3.9 GHz, `cpu_capacity` 997-1024) and ten
//! Cortex-A725 (2.8 GHz, 718-731). With the `performance` governor there is no
//! energy-aware placement, so a fresh thread can start on an A725 and only
//! migrates once its utilisation marks it misfit. The read is the 8k prefill's
//! critical path (layer 1 waits on it), so its placement is TTFT.
//!
//! Modes:
//! - unset / `off`: nothing (today).
//! - `observe`: no pinning; the caller logs the CPU each read started and ended on.
//! - `big`: the read thread pins itself to the highest-capacity CPUs (those
//!   within 90 % of the maximum `cpu_capacity`) that its current affinity mask
//!   allows; threads it spawns (the read rings) inherit the mask.
//! - `little`: the complement (measurement only: the slow bound).
//!
//! Exactness: scheduling only. The same reads are submitted and every row is
//! copied from its own completed read, wherever the thread runs.
//!
//! The pure part (parsing, CPU classes) is std-only so `rustc --test` runs its
//! tests; the two syscalls go through `libc`.

use std::sync::OnceLock;

#[path = "host_affinity/plan.rs"]
mod plan;
pub use plan::{Mode, Plan, SELECTOR, classes, cpu_list, parse_mode, resolve};

/// `cpu_capacity` of every CPU 0..n (stops at the first CPU directory that is absent).
pub fn read_capacities() -> Vec<Option<u64>> {
    let mut caps = Vec::new();
    for cpu in 0usize.. {
        let dir = format!("/sys/devices/system/cpu/cpu{cpu}");
        if !std::path::Path::new(&dir).exists() {
            break;
        }
        caps.push(
            std::fs::read_to_string(format!("{dir}/cpu_capacity"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok()),
        );
        if cpu >= 1023 {
            break;
        }
    }
    caps
}

/// CPUs in the calling thread's affinity mask.
pub fn allowed_cpus() -> Result<Vec<usize>, String> {
    // SAFETY: cpu_set_t is plain data; sched_getaffinity writes at most its size.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Err(format!(
                "sched_getaffinity: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok((0..libc::CPU_SETSIZE as usize)
            .filter(|&c| libc::CPU_ISSET(c, &set))
            .collect())
    }
}

/// Restrict the calling thread to `cpus` (threads it spawns afterwards inherit it).
pub fn pin_current_thread(cpus: &[usize]) -> Result<(), String> {
    // SAFETY: cpu_set_t is plain data; CPU_SET bounds-checks against CPU_SETSIZE.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cpus {
            if c < libc::CPU_SETSIZE as usize {
                libc::CPU_SET(c, &mut set);
            }
        }
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(format!(
                "sched_setaffinity: {}",
                std::io::Error::last_os_error()
            ));
        }
    }
    Ok(())
}

/// The CPU the calling thread is running on now.
pub fn current_cpu() -> Option<usize> {
    // SAFETY: no arguments; returns -1 on error.
    let cpu = unsafe { libc::sched_getcpu() };
    usize::try_from(cpu).ok()
}

/// The process-wide plan from [`SELECTOR`], resolved on first use. The second
/// value is true exactly once (the caller logs the receipt then).
pub fn process_plan() -> (&'static Plan, bool) {
    static PLAN: OnceLock<Plan> = OnceLock::new();
    let mut first = false;
    let plan = PLAN.get_or_init(|| {
        first = true;
        match parse_mode(std::env::var(SELECTOR).ok().as_deref()) {
            Err(reason) => Plan::Refused {
                mode: Mode::Off,
                reason: reason.replace(' ', "_"),
            },
            Ok(mode) => {
                let allowed = match allowed_cpus() {
                    Ok(a) => a,
                    Err(e) => {
                        return Plan::Refused {
                            mode,
                            reason: e.replace(' ', "_"),
                        };
                    }
                };
                resolve(mode, &read_capacities(), &allowed)
            }
        }
    });
    (plan, first)
}

/// Apply the process plan to the calling (read) thread. Returns the CPU the
/// thread is on afterwards when the plan observes placement. A failed pin
/// leaves the thread where it is (the read itself is unaffected).
pub fn apply_to_current_thread(plan: &Plan) -> Result<Option<usize>, String> {
    if let Plan::Pin { cpus, .. } = plan {
        pin_current_thread(cpus)?;
    }
    Ok(plan.observing().then(current_cpu).flatten())
}
