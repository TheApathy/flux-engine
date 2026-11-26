// SPDX-License-Identifier: AGPL-3.0-only

use crate::speculative::DraftProposerKind;

/// Plain decode has no device-selected dense/sparse topology. Even the
/// device-metadata indexer cannot make a pre-2048 dense capture safe to reuse
/// after the host switches to sparse attention.
pub(super) const fn plain_qsa_graph_capture_allowed(
    target_is_qwen4: bool,
    qsa_required: bool,
    host_seq_len: usize,
    graph_safe_qsa: bool,
) -> bool {
    !target_is_qwen4 || !qsa_required || (graph_safe_qsa && host_seq_len >= 2_048)
}

/// Whether the Qwen4 PLE segmented suffix graph may execute plain/bootstrap
/// single-token decode steps given the installed proposer arms.
///
/// - Target-only (`None, None`) and `Qwen4NativeMtp`-only builds are safe:
///   Native-MTP has no per-layer capture hook during `decode` and reads
///   `buffers.hidden_states()` after `decode` completes on the same stream.
/// - `DFlash` (or any active `dflash_hidden_save` buffer) requires the per-layer
///   `try_dflash_capture` hook in the eager decode loop and must stay excluded.
pub(super) const fn ple_suffix_graph_proposer_allowed(
    arm0: Option<DraftProposerKind>,
    arm1: Option<DraftProposerKind>,
    dflash_capture_active: bool,
) -> bool {
    if dflash_capture_active {
        return false;
    }
    let arm0_ok = matches!(arm0, None | Some(DraftProposerKind::Qwen4NativeMtp));
    let arm1_ok = matches!(arm1, None | Some(DraftProposerKind::Qwen4NativeMtp));
    arm0_ok && arm1_ok
}

/// Whether a slot's cached PLE suffix graph must be invalidated because the
/// host QSA route crossed between dense (`< 2048`) and sparse (`>= 2048`).
/// Handles both single-token steps (`2047 -> 2048`) and multi-token speculative
/// `verify_draft_tokens` jumps (`2045 -> 2050`) that skip `seq_len == 2048`.
pub(super) const fn ple_suffix_graph_needs_recapture(
    qsa_graph_safe_active: bool,
    cached_sparse_regime: Option<bool>,
    host_seq_len: usize,
) -> bool {
    if !qsa_graph_safe_active {
        return false;
    }
    let now_sparse = host_seq_len >= 2_048;
    match cached_sparse_regime {
        Some(was_sparse) => was_sparse != now_sparse,
        None => host_seq_len == 2_048,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        plain_qsa_graph_capture_allowed, ple_suffix_graph_needs_recapture,
        ple_suffix_graph_proposer_allowed,
    };
    use crate::speculative::DraftProposerKind;

    #[test]
    fn non_qwen4_scheduler_qsa_hint_retains_graphs() {
        for seq_len in [0, 1, 2_047, 2_048, 2_049, 262_144, usize::MAX] {
            for graph_safe in [false, true] {
                assert!(plain_qsa_graph_capture_allowed(
                    false, true, seq_len, graph_safe
                ));
            }
        }
    }

    #[test]
    fn non_qsa_decode_retains_graphs() {
        for seq_len in [0, 1, 2_047, 2_048, 2_049, 262_144, usize::MAX] {
            for graph_safe in [false, true] {
                assert!(plain_qsa_graph_capture_allowed(
                    true, false, seq_len, graph_safe
                ));
            }
        }
    }

    #[test]
    fn host_qsa_metadata_is_never_captured() {
        for seq_len in [0, 1, 2_047, 2_048, 2_049, 8_192, 262_144, usize::MAX] {
            assert!(!plain_qsa_graph_capture_allowed(true, true, seq_len, false));
        }
    }

    #[test]
    fn graph_safe_indexer_does_not_make_dense_topology_dynamic() {
        for seq_len in [0, 1, 2_046, 2_047] {
            assert!(!plain_qsa_graph_capture_allowed(true, true, seq_len, true));
        }
    }

    #[test]
    fn graph_safe_sparse_topology_can_be_captured() {
        for seq_len in [2_048, 2_049, 8_192, 135_168, 262_144, usize::MAX] {
            assert!(plain_qsa_graph_capture_allowed(true, true, seq_len, true));
        }
    }

    #[test]
    fn first_sparse_step_is_the_admission_boundary() {
        assert!(!plain_qsa_graph_capture_allowed(true, true, 2_047, true));
        assert!(plain_qsa_graph_capture_allowed(true, true, 2_048, true));
    }

    #[test]
    fn ple_suffix_graph_admits_target_only_and_native_mtp_but_excludes_dflash() {
        assert!(ple_suffix_graph_proposer_allowed(None, None, false));
        assert!(ple_suffix_graph_proposer_allowed(
            Some(DraftProposerKind::Qwen4NativeMtp),
            None,
            false
        ));
        assert!(ple_suffix_graph_proposer_allowed(
            Some(DraftProposerKind::Qwen4NativeMtp),
            Some(DraftProposerKind::Qwen4NativeMtp),
            false
        ));
        assert!(!ple_suffix_graph_proposer_allowed(
            Some(DraftProposerKind::DFlash),
            None,
            false
        ));
        assert!(!ple_suffix_graph_proposer_allowed(
            Some(DraftProposerKind::Qwen4NativeMtp),
            Some(DraftProposerKind::DFlash),
            false
        ));
        assert!(!ple_suffix_graph_proposer_allowed(
            Some(DraftProposerKind::Other),
            None,
            false
        ));
        assert!(!ple_suffix_graph_proposer_allowed(
            Some(DraftProposerKind::Qwen4NativeMtp),
            None,
            true
        ));
    }

    #[test]
    fn ple_suffix_recapture_catches_speculative_jumps_across_2048() {
        // Dense graph (Some(false)) jumping past 2048 (e.g., 2045 -> 2050 via K=5 verify) must recapture.
        assert!(ple_suffix_graph_needs_recapture(true, Some(false), 2_048));
        assert!(ple_suffix_graph_needs_recapture(true, Some(false), 2_050));
        // Already sparse graph (Some(true)) stays valid for all >= 2048 steps.
        assert!(!ple_suffix_graph_needs_recapture(true, Some(true), 2_048));
        assert!(!ple_suffix_graph_needs_recapture(true, Some(true), 2_053));
        // Non-QSA or graph-safe disabled never triggers QSA regime recapture.
        assert!(!ple_suffix_graph_needs_recapture(false, Some(false), 2_050));
    }
}
