// SPDX-License-Identifier: AGPL-3.0-only

//! Receipts for the K=γ verify CUDA-graph lifecycle (logging only).
//!
//! A Qwen4 QSA request can admit CUDA graphs below seq_len 2048 via the
//! dense-prefix fused path (`qsa_dense_prefix_fused`) and at/above 2048 via
//! `qwen4_qsa_graph_route_safe` (or device-side attention topology). Graphs
//! are slot-keyed and retained across free/compaction by default in
//! `sequence.rs` (invalidated on QSA/rotary regime change or explicit
//! release). These receipts make graph engagement and release visible per
//! sequence; they never change what executes.

use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

static ENGAGED_SLOTS: LazyLock<Mutex<HashSet<usize>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Records that `slot` is running graphed verify. Returns true exactly once
/// per sequence lifetime (until `reset_slot`).
pub(super) fn note_engaged(slot: usize) -> bool {
    ENGAGED_SLOTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(slot)
}

/// Forgets the engaged mark for `slot` (sequence freed or compacted away).
pub(super) fn reset_slot(slot: usize) {
    ENGAGED_SLOTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(&slot);
}

pub(super) fn engaged_reason(
    qsa_required: bool,
    device_topology: bool,
    qsa_dense_prefix: bool,
) -> &'static str {
    if !qsa_required {
        "qsa_not_required"
    } else if device_topology {
        "device_attention_topology"
    } else if qsa_dense_prefix {
        "qsa_dense_prefix_le_2048"
    } else {
        "qsa_graph_safe_ge_2048"
    }
}

pub(super) fn engaged_line(session: u64, slot: usize, seq_len: usize, reason: &str) -> String {
    format!(
        "VERIFY_GRAPH_ENGAGED session={session:016x} slot={slot} seq_len={seq_len} reason={reason}"
    )
}

pub(super) fn released_line(site: &str, session: u64, slot: usize, destroyed: usize) -> String {
    format!(
        "VERIFY_GRAPH_RELEASED site={site} session={session:016x} slot={slot} destroyed={destroyed}"
    )
}

pub(super) fn captured_line(
    session: u64,
    slot: usize,
    k: usize,
    seq_len: usize,
    pack_key: u32,
) -> String {
    format!(
        "Captured CUDA graph for K=γ verify (slot={slot} K={k}) session={session:016x} \
         seq_len={seq_len} pack_key={pack_key}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const VERIFY_D: &str = include_str!("verify_d.rs");
    const SEQUENCE: &str = include_str!("sequence.rs");

    #[test]
    fn switch_emits_once_per_sequence_and_rearms_after_release() {
        let slot = 900_001;
        assert!(note_engaged(slot));
        assert!(!note_engaged(slot));
        assert!(!note_engaged(slot));
        reset_slot(slot);
        assert!(note_engaged(slot));
        reset_slot(slot);
    }

    #[test]
    fn slots_are_independent() {
        let (a, b) = (900_011, 900_012);
        assert!(note_engaged(a));
        assert!(note_engaged(b));
        reset_slot(a);
        assert!(!note_engaged(b));
        assert!(note_engaged(a));
        reset_slot(a);
        reset_slot(b);
    }

    #[test]
    fn reason_names_the_gate_that_opened() {
        assert_eq!(engaged_reason(true, false, false), "qsa_graph_safe_ge_2048");
        assert_eq!(
            engaged_reason(true, false, true),
            "qsa_dense_prefix_le_2048"
        );
        assert_eq!(
            engaged_reason(true, true, false),
            "device_attention_topology"
        );
        assert_eq!(
            engaged_reason(true, true, true),
            "device_attention_topology"
        );
        assert_eq!(engaged_reason(false, false, false), "qsa_not_required");
    }

    #[test]
    fn lines_carry_their_fields() {
        let c = captured_line(0x02ce97c4280d19fe, 0, 2, 3353, 5);
        assert!(c.starts_with("Captured CUDA graph for K=γ verify (slot=0 K=2)"));
        assert!(c.contains("session=02ce97c4280d19fe"));
        assert!(c.contains("seq_len=3353"));
        assert!(c.contains("pack_key=5"));
        assert_eq!(
            engaged_line(5, 1, 2049, "qsa_graph_safe_ge_2048"),
            "VERIFY_GRAPH_ENGAGED session=0000000000000005 slot=1 seq_len=2049 reason=qsa_graph_safe_ge_2048"
        );
        assert_eq!(
            released_line("free_sequence", 5, 1, 4),
            "VERIFY_GRAPH_RELEASED site=free_sequence session=0000000000000005 slot=1 destroyed=4"
        );
    }

    #[test]
    fn verify_d_wires_the_receipts() {
        assert!(VERIFY_D.contains("verify_graph_receipts::captured_line("));
        assert!(VERIFY_D.contains("verify_graph_receipts::note_engaged(seq.slot_idx)"));
        assert!(VERIFY_D.contains("verify_graph_receipts::engaged_line("));
        assert!(!VERIFY_D.contains("\"Captured CUDA graph for K=γ verify (slot={} K={})\""));
    }

    #[test]
    fn free_and_compaction_release_with_the_destroyed_count() {
        assert_eq!(
            SEQUENCE
                .matches("verify_graph_receipts::released_line(")
                .count(),
            2
        );
        assert_eq!(
            SEQUENCE
                .matches("verify_graph_receipts::reset_slot(")
                .count(),
            2
        );
        assert!(SEQUENCE.contains("\"free_sequence\""));
        assert!(SEQUENCE.contains("\"compact_sequence\""));
        assert!(SEQUENCE.contains("let destroyed = stale_keys.len();"));
    }
}
