// SPDX-License-Identifier: AGPL-3.0-only

//! Run-scoped scheduler levers exposed to the TUI dashboard.
//!
//! Holds run-scoped atomic flags (currently the loop watchdog) that are
//! published when a scheduler starts and replaced on model swap, keeping state
//! scoped to the active run rather than a process global.

use std::sync::atomic::{AtomicBool, Ordering};

/// Levers for one run. Published to the dashboard when the scheduler starts and
/// replaced on a hot-swap, so `/watchdog` always addresses the live run.
pub struct SchedLevers {
    loop_watchdog: AtomicBool,
}

impl SchedLevers {
    /// Constructs levers with the loop watchdog disarmed by default.
    ///
    /// Operators can opt in at startup via `ATLAS_LOOP_WATCHDOG=1` or toggle it
    /// from the Ops pane.
    pub fn defaults() -> Self {
        Self {
            loop_watchdog: AtomicBool::new(false),
        }
    }

    /// Build run-scoped levers, honouring an operator environment override.
    ///
    /// `FLUX_LOOP_WATCHDOG=1` (or legacy `ATLAS_LOOP_WATCHDOG=1`) boots with
    /// the watchdog armed; otherwise it starts disarmed.
    pub fn from_env() -> Self {
        let armed = matches!(
            std::env::var("FLUX_LOOP_WATCHDOG")
                .or_else(|_| std::env::var("ATLAS_LOOP_WATCHDOG"))
                .as_deref(),
            Ok("1") | Ok("true")
        );
        Self {
            loop_watchdog: AtomicBool::new(armed),
        }
    }

    /// Is the loop watchdog armed?
    pub fn loop_watchdog(&self) -> bool {
        self.loop_watchdog.load(Ordering::Relaxed)
    }

    /// Arm or disarm the watchdog lever (`SeqCst` write, `Relaxed` read).
    pub fn set_loop_watchdog(&self, on: bool) {
        self.loop_watchdog.store(on, Ordering::SeqCst);
    }
}

impl Default for SchedLevers {
    fn default() -> Self {
        Self::defaults()
    }
}
