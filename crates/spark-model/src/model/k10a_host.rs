// SPDX-License-Identifier: AGPL-3.0-only

//! K10 item A: the 8k per-request host path (selectors, trigger, receipts).
//!
//! - `ATLAS_ALLOC_SEQ_ASYNC=1`: `alloc_sequence` zeroes the new SSM slot (every
//!   byte `zero_slot` and `reset_slot` zero today, each once: `slot_zero_plan`)
//!   with stream-ordered memsets on the model's default stream, records an event
//!   and makes the caller's prefill stream wait on it, with no host sync. Today
//!   it issues 72 async memsets plus a sync, then 576 synchronous memsets plus a
//!   sync: 9.4-11.0 ms of host time per request before the chunk-0 PLE read is
//!   issued (v6 8k census, `ALLOC_SEQUENCE ... reset_slot=8.8-9.9`). Only the
//!   chunked first-chunk route asks for it (`Model::alloc_sequence_on`, called by
//!   the scheduler's `prefill_a_step` with its prefill stream); every other
//!   caller keeps the synchronous path.
//! - `ATLAS_PLE_READ_AFFINITY=big` (plan in `spark_storage::host_affinity`): the
//!   PLE prefill read thread pins itself to the big cores.
//! - `ATLAS_K10A_TRIGGER=<path>` (optional): while set, both levers act only
//!   while `<path>` exists (one load serves an in-process A/B/A); with the file
//!   absent the read threads log their placement without pinning.
//!
//! Std-only so `rustc --edition 2024 --test k10a_host.rs` runs its tests.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub(crate) const ALLOC_SEQ_ASYNC_SELECTOR: &str = "ATLAS_ALLOC_SEQ_ASYNC";
pub(crate) const TRIGGER: &str = "ATLAS_K10A_TRIGGER";

fn parse_on_off(selector: &str, value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{selector} must be 0 or 1; got {other:?}")),
    }
}

/// `ATLAS_ALLOC_SEQ_ASYNC` as configured (parsed once).
pub(crate) fn alloc_seq_async_selected() -> Result<bool, String> {
    static SELECTED: OnceLock<Result<bool, String>> = OnceLock::new();
    SELECTED
        .get_or_init(|| {
            parse_on_off(
                ALLOC_SEQ_ASYNC_SELECTOR,
                std::env::var(ALLOC_SEQ_ASYNC_SELECTOR).ok().as_deref(),
            )
        })
        .clone()
}

/// The async slot zeroing is selected and the trigger (if any) allows it now.
pub(crate) fn alloc_seq_async_armed() -> Result<bool, String> {
    Ok(alloc_seq_async_selected()? && trigger_present())
}

/// No [`TRIGGER`] configured, or its file exists now.
pub(crate) fn trigger_present() -> bool {
    static TRIGGER_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    let trigger = TRIGGER_PATH.get_or_init(|| {
        std::env::var(TRIGGER)
            .ok()
            .filter(|t| !t.is_empty())
            .map(PathBuf::from)
    });
    trigger_allows(trigger.as_deref(), |p| p.exists())
}

fn trigger_allows(trigger: Option<&Path>, exists: impl Fn(&Path) -> bool) -> bool {
    trigger.is_none_or(exists)
}

/// True the first time `key` is seen in this process (one receipt line per key).
pub(crate) fn first_time(key: &'static str) -> bool {
    static SEEN: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.contains(&key) {
        false
    } else {
        seen.push(key);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_is_strict() {
        let s = ALLOC_SEQ_ASYNC_SELECTOR;
        assert_eq!(parse_on_off(s, None), Ok(false));
        assert_eq!(parse_on_off(s, Some("")), Ok(false));
        assert_eq!(parse_on_off(s, Some("0")), Ok(false));
        assert_eq!(parse_on_off(s, Some("1")), Ok(true));
        assert!(parse_on_off(s, Some("true")).is_err());
        assert!(parse_on_off(s, Some("2")).is_err());
    }

    #[test]
    fn trigger_gates_only_when_configured() {
        assert!(trigger_allows(None, |_| false));
        assert!(trigger_allows(Some(Path::new("/x")), |_| true));
        assert!(!trigger_allows(Some(Path::new("/x")), |_| false));
    }

    #[test]
    fn receipts_once() {
        assert!(first_time("K10A_TEST_KEY"));
        assert!(!first_time("K10A_TEST_KEY"));
    }
}
