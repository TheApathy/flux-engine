// SPDX-License-Identifier: AGPL-3.0-only

//! Every route that commits a native-MTP verify frame records exactly one
//! settled-frame receipt (committed acceptance + adaptive depth), and only
//! after its settlement succeeded. At C=1 the frames take the generic
//! `step_verify_dflash` route or the fixed K2/K3/K4 steps, never the
//! batched-K5 `finalize_commit`; ACC2 logged no receipts because only the
//! batched route recorded.

use super::short_verify_frame::ShortVerifyFrame;
use super::*;

fn body_of<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let s = source.find(start).expect("function start");
    let e = source[s..].find(end).map_or(source.len(), |off| s + off);
    &source[s..e]
}

#[test]
fn generic_route_records_after_trim_and_after_each_terminal_reconcile() {
    let source = include_str!("verify_dflash_step.rs");
    let body = body_of(
        source,
        "pub fn step_verify_dflash(",
        "\n/// Whether any DFlash TREE",
    );
    assert_eq!(
        body.matches("record_settled(a,").count(),
        4,
        "1 settled + 3 terminal sites"
    );
    // Each terminal reconcile records only in its success branch.
    let reconciles = body
        .matches("if let Err(e) = reconcile_terminal_verify_state(")
        .count();
    assert_eq!(reconciles, 3);
    assert!(body.matches(", true);").count() >= 3);
    // The non-terminal record follows the proposer trim and precedes re-propose.
    let trim = body
        .find("model.trim_proposer_state(&mut a.seq, num_accepted, 0)")
        .expect("trim");
    let settled = body
        .find("record_settled(a, k_verify, num_accepted, consumed, false)")
        .expect("settled");
    let propose = body
        .find("proposal_lifecycle::propose_and_install_with(")
        .expect("propose");
    assert!(trim < settled && settled < propose);
    // The old pre-settlement receipt next to `record_accept` is gone.
    assert!(!body.contains("record_committed_frame("));
}

#[test]
fn fixed_k_steps_record_once_per_trim_and_on_terminal_success() {
    for (name, source) in [
        ("k2", include_str!("verify_k2_step.rs")),
        ("k3", include_str!("verify_k3_step.rs")),
        ("k4", include_str!("verify_k4_step.rs")),
    ] {
        let trims = source
            .matches("model.trim_proposer_state(&mut a.seq,")
            .count();
        assert!(trims > 0, "{name}");
        assert_eq!(source.matches("frame.settled(a,").count(), trims, "{name}");
    }
    let frame = include_str!("short_verify_frame.rs");
    let ok = frame
        .find("if result.is_ok() {")
        .expect("terminal success gate");
    let rec = frame
        .find("record_settled_frame(&self.receipt(a, accepted, true))")
        .expect("terminal record");
    assert!(ok < rec);
}

#[test]
fn batched_route_records_only_after_settlement() {
    let source = include_str!("verify_qwen4_batched_step.rs");
    assert_eq!(source.matches("record_settled_frame(").count(), 2);
    let trim = source
        .find(".context(\"batched verify trim proposer\")?;")
        .expect("trim");
    let rec = source
        .find("record_settled_frame(&receipt(a, num_accepted, frame_consumed_rows, false))")
        .expect("rec");
    assert!(trim < rec);
}

#[test]
fn adaptive_depth_is_fed_only_through_the_settled_helper() {
    for source in [
        include_str!("verify_dflash_step.rs"),
        include_str!("verify_qwen4_batched_step.rs"),
        include_str!("verify_k2_step.rs"),
        include_str!("verify_k3_step.rs"),
        include_str!("verify_k4_step.rs"),
        include_str!("short_verify_frame.rs"),
        include_str!("stochastic_verify_walk.rs"),
    ] {
        assert!(!source.contains("adaptive_depth::record_frame("));
    }
    assert!(
        include_str!("spec_accept_log.rs")
            .contains("super::adaptive_depth::record_frame(r.drafted, r.accepted)")
    );
}

#[test]
fn short_frame_receipt_describes_a_c1_k4_frame() {
    let mut a = super::test_fixtures::active_seq(ResponseSink::Blocking(None), 0);
    a.inside_thinking = true;
    let before = a.output_tokens.len();
    let mut frame = ShortVerifyFrame::capture(&a, 4);
    frame.emit(&mut a, 9, None);
    frame.emit(&mut a, 10, None);
    let r = frame.receipt_for_test(&a, 1, false);
    assert_eq!(r.verified_rows, 4);
    assert_eq!(r.drafted, 3);
    assert!(r.thinking_entry);
    assert_eq!(r.accepted, 1);
    assert_eq!(r.consumed_rows, a.output_tokens.len() - before);
    assert_eq!(r.emitted, a.output_tokens.len() - before);
    assert!(!r.terminal);
}

#[test]
fn adaptive_depth_cap_applies_before_the_batched_armed_check() {
    let source = include_str!("verify_qwen4_batched_step.rs");
    let body = body_of(source, "pub(super) fn effective_num_drafts(", "\n}\n");
    assert!(
        !body.contains("return configured;"),
        "C=1 must not bypass the depth cap"
    );
    let armed = body.find("qwen4_batched_k5_armed()").expect("armed check");
    let cap = body
        .find("adaptive_depth::stochastic_depth(width_capped)")
        .expect("cap");
    assert!(armed < cap);
}

#[test]
fn receipts_are_labelled_settled_post() {
    let log = include_str!("spec_accept_log.rs");
    assert_eq!(log.matches("settled = \"post\"").count(), 2);
}
