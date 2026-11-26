// SPDX-License-Identifier: AGPL-3.0-only

use super::{speculation_routes, target_only_receipt};

#[test]
fn global_control_disables_dspark_like_internal_spec_with_trait_default() {
    // Dsv41Model advertises internal speculation but does not carry the
    // TransformerModel-only last-K control field.  The server-owned bool must
    // still disable its scheduler route.
    assert_eq!(speculation_routes(true, true, false, false), (false, false));
}

#[test]
fn global_control_disables_loaded_qwen_proposer_route() {
    assert_eq!(speculation_routes(true, false, true, true), (false, false));
}

#[test]
fn admitted_routes_are_unchanged_when_control_is_clear() {
    assert_eq!(speculation_routes(false, true, false, false), (true, false));
    assert_eq!(speculation_routes(false, false, true, true), (false, true));
}

#[test]
fn target_only_scheduler_logs_the_ordinary_path_receipt() {
    // FLUX_DISABLE_SPECULATION=1 / FLUX_SPEC_MODE=none: no proposer is built.
    assert_eq!(
        target_only_receipt(false, 0, false, false).as_deref(),
        Some(
            "SPECULATION_CONTROL_RECEIPT schema=1 control=unloaded proposer_loaded=false scheduler_path=ordinary-target-only use_mtp=false num_drafts=0"
        )
    );
    // ATLAS_DISABLE_SPECULATION=1 with the proposer still loaded.
    let (internal, mtp) = speculation_routes(true, false, true, true);
    let receipt = target_only_receipt(internal, usize::from(mtp), true, true).unwrap();
    assert!(
        receipt
            .contains("control=disabled proposer_loaded=true scheduler_path=ordinary-target-only")
    );
    assert!(
        target_only_receipt(false, 0, true, false)
            .unwrap()
            .contains("control=not-requested")
    );
}

#[test]
fn speculating_schedulers_log_no_target_only_receipt() {
    assert_eq!(target_only_receipt(false, 4, true, false), None);
    assert_eq!(target_only_receipt(true, 0, false, false), None);
}
