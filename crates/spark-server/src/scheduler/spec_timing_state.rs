// SPDX-License-Identifier: AGPL-3.0-only

//! Cached diagnostic gate and allocation-free C=1 request identity tracker.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Gate {
    pub(super) enabled: bool,
}

impl Gate {
    pub(super) fn resolve(
        flag: Option<&str>,
        async_flag: Option<&str>,
        max_batch: usize,
    ) -> Result<Self, &'static str> {
        match flag {
            None | Some("0") => Ok(Self { enabled: false }),
            Some("1") if max_batch != 1 => Err("spec-cycle v2 requires max_batch_size=1"),
            Some("1") if async_flag != Some("0") => {
                Err("spec-cycle v2 requires explicit ATLAS_DFLASH_ASYNC=0")
            }
            Some("1") => Ok(Self { enabled: true }),
            Some(_) => Err("ATLAS_DFLASH_SPEC_CYCLE_V2 must be 0 or 1"),
        }
    }

    pub(super) fn enabled() -> bool {
        GATE.get().is_some_and(|gate| gate.enabled)
    }
}

static GATE: OnceLock<Gate> = OnceLock::new();

fn resolve_for_configure(
    flag: Option<&str>,
    async_flag: Option<&str>,
    max_batch: usize,
) -> Result<(Gate, bool), &'static str> {
    if flag == Some("1") && max_batch > 1 {
        return Ok((Gate { enabled: false }, true));
    }
    Gate::resolve(flag, async_flag, max_batch).map(|gate| (gate, false))
}

pub(super) fn configure(max_batch: usize) -> Result<bool, &'static str> {
    let flag = match std::env::var("ATLAS_DFLASH_SPEC_CYCLE_V2") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err("ATLAS_DFLASH_SPEC_CYCLE_V2 is not valid Unicode");
        }
    };
    let async_flag = if flag.as_deref() == Some("1") && max_batch == 1 {
        match std::env::var("ATLAS_DFLASH_ASYNC") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err("ATLAS_DFLASH_ASYNC is not valid Unicode");
            }
        }
    } else {
        None
    };
    let (gate, skipped_multi_batch) =
        resolve_for_configure(flag.as_deref(), async_flag.as_deref(), max_batch)?;
    if skipped_multi_batch {
        tracing::info!(
            max_batch_size = max_batch,
            "DFlash spec-cycle schema 2 timing skipped: requires max_batch_size=1"
        );
    }
    match GATE.set(gate) {
        Ok(()) => Ok(gate.enabled),
        Err(_) if GATE.get() == Some(&gate) => Ok(gate.enabled),
        Err(_) => Err("spec-cycle v2 scheduler configuration changed after initialization"),
    }
}

#[derive(Clone, Copy)]
struct RequestState {
    id: u64,
    expected_pre: u64,
}

thread_local! {
    static REQUEST: Cell<RequestState> = const { Cell::new(RequestState { id: 0, expected_pre: 0 }) };
}
static NEXT_REQUEST: AtomicU64 = AtomicU64::new(1);

pub(super) fn begin_request(pre: u64) -> Option<u64> {
    REQUEST.with(|slot| {
        let mut state = slot.get();
        if state.id == 0 {
            state = RequestState {
                id: NEXT_REQUEST.fetch_add(1, Ordering::Relaxed),
                expected_pre: pre,
            };
            slot.set(state);
        }
        (state.expected_pre == pre).then_some(state.id)
    })
}

pub(super) fn finish_request(id: u64, pre: u64, next_pre: u64, terminal: bool) -> bool {
    REQUEST.with(|slot| {
        let mut state = slot.get();
        if state.id != id || state.expected_pre != pre || next_pre < pre {
            return false;
        }
        if terminal {
            state = RequestState {
                id: 0,
                expected_pre: 0,
            };
        } else {
            state.expected_pre = next_pre;
        }
        slot.set(state);
        true
    })
}

pub(super) fn abandon_request(id: u64) {
    REQUEST.with(|slot| {
        if slot.get().id == id {
            clear_request(slot);
        }
    });
}

/// Retire the C=1 timing identity at the scheduler request boundary.
///
/// A completed speculative cycle advances `expected_pre`, but a client can
/// cancel before another cycle gets a terminal record.  Without an explicit
/// lifecycle reset, a later request that happens to reach that sequence length
/// can resume the old identity and splice two HTTP requests into one trace.
pub(super) fn retire_request() {
    REQUEST.with(clear_request);
}

fn clear_request(slot: &Cell<RequestState>) {
    slot.set(RequestState {
        id: 0,
        expected_pre: 0,
    });
}

#[cfg(test)]
mod state_tests {
    use super::{Gate, resolve_for_configure};

    #[test]
    fn configure_skips_cleanly_for_multi_batch_while_gate_resolve_keeps_strict_c1_contract() {
        // Multi-batch serving with ATLAS_DFLASH_SPEC_CYCLE_V2=1 skips cleanly without Err.
        for max_batch in [2usize, 8, 16, 64] {
            assert_eq!(
                resolve_for_configure(Some("1"), None, max_batch),
                Ok((Gate { enabled: false }, true)),
                "max_batch={max_batch}"
            );
            assert_eq!(
                resolve_for_configure(Some("1"), Some("0"), max_batch),
                Ok((Gate { enabled: false }, true)),
                "max_batch={max_batch}"
            );
            assert!(
                Gate::resolve(Some("1"), Some("0"), max_batch).is_err(),
                "Gate::resolve preserves strict C=1 contract for max_batch={max_batch}"
            );
        }
        // C=1 and invalid flags preserve exact validation semantics.
        assert_eq!(
            resolve_for_configure(Some("1"), Some("0"), 1),
            Ok((Gate { enabled: true }, false))
        );
        assert!(resolve_for_configure(Some("1"), None, 1).is_err());
        assert!(resolve_for_configure(Some("1"), Some("1"), 1).is_err());
        assert!(resolve_for_configure(Some("1"), Some("0"), 0).is_err());
        assert!(resolve_for_configure(Some("yes"), Some("0"), 16).is_err());
        assert_eq!(
            resolve_for_configure(None, None, 16),
            Ok((Gate { enabled: false }, false))
        );
        assert_eq!(
            resolve_for_configure(Some("0"), Some("1"), 16),
            Ok((Gate { enabled: false }, false))
        );
    }
}
