// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn moe_only_receipt_requires_all_48_layers_and_labels_serial_core() {
    let selectors = Selectors::from_values(None, None).with_moe_only(true);
    let mut capture = Capture::new(selectors, 256, tests::flash_geometry()).unwrap();
    for _ in 0..12 {
        capture.engage(PrefillPath::Attention, 256).unwrap();
    }
    for _ in 0..36 {
        capture.engage(PrefillPath::Ssm, 256).unwrap();
    }
    let lines = capture.finish().unwrap();
    assert_eq!(lines.len(), 2);
    for line in lines {
        assert!(line.contains("selector=ATLAS_QWEN4_PREFILL_MOE_BATCH value=1"));
        assert!(line.contains("ffn=grouped core=serial_token_ordered"));
        assert!(line.contains("attention_selector=1 ssm_selector=1"));
        assert!(!line.contains("parity=pass"));
    }
}

#[test]
fn moe_only_receipt_cannot_credit_just_attention_or_wrong_rows() {
    let selectors = Selectors::from_values(None, None).with_moe_only(true);
    let mut capture = Capture::new(selectors, 256, tests::flash_geometry()).unwrap();
    assert!(capture.engage(PrefillPath::Ssm, 255).is_err());
    for _ in 0..12 {
        capture.engage(PrefillPath::Attention, 256).unwrap();
    }
    assert!(capture.finish().is_err());
}

#[test]
fn absent_moe_only_mode_preserves_existing_selectors() {
    for attention in [None, Some("0"), Some("1")] {
        for ssm in [None, Some("0"), Some("1")] {
            let selectors = Selectors::from_values(attention, ssm);
            assert_eq!(selectors.with_moe_only(false), selectors);
        }
    }
}

#[test]
fn f12_receipt_distinguishes_recurrence_and_hyper_from_attention() {
    let mut selectors = Selectors::from_values(None, None).with_moe_only(true);
    selectors.hyper_exact = true;
    selectors.ssm_exact = true;
    selectors.moe_compact = true;
    let mut capture = Capture::new(selectors, 33, tests::flash_geometry()).unwrap();
    for _ in 0..12 {
        capture.engage(PrefillPath::Attention, 33).unwrap();
    }
    for _ in 0..36 {
        capture.engage(PrefillPath::Ssm, 33).unwrap();
    }
    let lines = capture.finish().unwrap();
    assert!(lines[0].contains("core=serial_token_ordered"));
    assert!(lines[1].contains("core=exact_projection_sequence_nosnap"));
    for line in lines {
        assert!(line.contains("hc=exact_m32"));
        assert!(line.contains("routed_schedule=original_compact"));
        assert!(!line.contains("parity=pass"));
    }
}

#[test]
fn f13_receipts_do_not_claim_exact_or_checked_tensorcore_projections() {
    for (mode, core) in [
        (Mode::Off, "exact_projection_sequence_nosnap"),
        (Mode::Out, "exact_qkvz_bf16_mma_output_sequence_nosnap"),
        (Mode::All, "bf16_mma_projections_sequence_nosnap"),
    ] {
        let mut selectors = Selectors::from_values(None, None).with_moe_only(true);
        selectors.hyper_exact = true;
        selectors.ssm_exact = true;
        selectors.moe_compact = true;
        selectors.ssm_gemm = mode;
        let mut capture = Capture::new(selectors, 65, tests::flash_geometry()).unwrap();
        for _ in 0..12 {
            capture.engage(PrefillPath::Attention, 65).unwrap();
        }
        for _ in 0..36 {
            capture.engage(PrefillPath::Ssm, 65).unwrap();
        }
        let lines = capture.finish().unwrap();
        assert!(lines[0].contains("core=serial_token_ordered"));
        assert!(lines[1].contains(&format!("core={core} ")));
        if mode != Mode::Off {
            assert!(!lines[1].contains("core=exact_projection_sequence_nosnap"));
        }
        for line in lines {
            assert!(line.contains(&format!("ssm_projection_gemm={}", mode.as_str())));
            assert!(line.contains("hc=exact_m32"));
            assert!(line.contains("routed_schedule=original_compact"));
            assert!(!line.contains("parity=pass"));
        }
    }
}

#[test]
fn r2_receipt_requires_one_consumer_per_model_layer_and_counts_routes() {
    let mut selectors = Selectors::from_values(None, None).with_moe_only(true);
    selectors.slot_mxfp8_r2 = true;
    selectors.moe_compact = true;
    let mut capture = Capture::new(selectors, 256, tests::flash_geometry()).unwrap();
    for _ in 0..12 {
        capture.engage(PrefillPath::Attention, 256).unwrap();
    }
    for _ in 0..36 {
        capture.engage(PrefillPath::Ssm, 256).unwrap();
    }
    for _ in 0..16 {
        for consumer in [
            MoeSlotR2Consumer::Direct,
            MoeSlotR2Consumer::Materialized,
            MoeSlotR2Consumer::FusedRstd,
        ] {
            capture.preflight_slot_r2_producer(256).unwrap();
            capture.commit_slot_r2_producer();
            capture.preflight_slot_r2(consumer, 256).unwrap();
            capture.commit_slot_r2(consumer);
        }
    }
    let lines = capture.finish().unwrap();
    let receipt = lines
        .iter()
        .find(|line| line.contains("family=moe_slot_r2"))
        .unwrap();
    assert!(receipt.contains("expected_layers=48 engaged_layers=48"));
    assert!(receipt.contains("producer_enqueued=48"));
    assert!(receipt.contains("direct=16 materialized=16 fused_rstd=16"));
    assert!(receipt.contains("producer_kernel=v17qr2"));
}

#[test]
fn r2_receipt_fails_closed_on_row_census_and_selector_drift() {
    let mut selectors = Selectors::from_values(None, None).with_moe_only(true);
    selectors.slot_mxfp8_r2 = true;
    selectors.moe_compact = true;
    let mut capture = Capture::new(selectors, 256, tests::flash_geometry()).unwrap();
    assert!(
        capture
            .preflight_slot_r2(MoeSlotR2Consumer::Direct, 255)
            .is_err()
    );
    assert!(
        capture
            .preflight_slot_r2(MoeSlotR2Consumer::Direct, 256)
            .is_err()
    );
    capture.preflight_slot_r2_producer(256).unwrap();
    capture.commit_slot_r2_producer();
    assert!(capture.preflight_slot_r2_producer(256).is_err());
    capture
        .preflight_slot_r2(MoeSlotR2Consumer::Direct, 256)
        .unwrap();
    capture.commit_slot_r2(MoeSlotR2Consumer::Direct);
    for _ in 0..12 {
        capture.engage(PrefillPath::Attention, 256).unwrap();
    }
    for _ in 0..36 {
        capture.engage(PrefillPath::Ssm, 256).unwrap();
    }
    for _ in 0..46 {
        capture.preflight_slot_r2_producer(256).unwrap();
        capture.commit_slot_r2_producer();
        capture
            .preflight_slot_r2(MoeSlotR2Consumer::Direct, 256)
            .unwrap();
        capture.commit_slot_r2(MoeSlotR2Consumer::Direct);
    }
    assert!(capture.finish().is_err());

    let selectors = Selectors::from_values(None, None).with_moe_only(true);
    let capture = Capture::new(selectors, 256, tests::flash_geometry()).unwrap();
    assert!(
        capture
            .preflight_slot_r2(MoeSlotR2Consumer::Direct, 256)
            .is_err()
    );
}

#[test]
fn r2_hooks_bracket_successful_launches_and_receipts_follow_flushes() {
    fn preflight_launch_commit(source: &str, preflight: &str, commit: &str) {
        let preflight = source.find(preflight).expect("request-scoped R2 preflight");
        let commit = source.find(commit).expect("request-scoped R2 commit");
        let launch = source[preflight..commit]
            .find(".launch(stream)?;")
            .map(|offset| preflight + offset)
            .expect("kernel launch between R2 preflight and commit");
        assert!(
            preflight < launch && launch < commit,
            "R2 preflight and commit must bracket the successful launch"
        );
    }

    let producer = include_str!("../../layers/moe/qwen4_compact_ops.rs");
    let direct = include_str!("../../layers/moe/forward_prefill.rs");
    let materialized = include_str!("../../layers/qwen4_hyper/pending.rs");
    let fused = include_str!("../../layers/qwen4_hyper/prefill_inject_launch.rs");
    preflight_launch_commit(
        producer,
        "preflight_slot_r2_producer(rows)?;",
        "commit_slot_r2_producer();",
    );
    preflight_launch_commit(direct, "preflight_slot_r2(", "commit_slot_r2(");
    preflight_launch_commit(materialized, "preflight_slot_r2(", "commit_slot_r2(");
    preflight_launch_commit(fused, "preflight_slot_r2(", "commit_slot_r2(");

    for source in [
        include_str!("../trait_impl/prefill_a.rs"),
        include_str!("../trait_impl/prefill_b/forward_layers.rs"),
    ] {
        let pending = source
            .rfind("qwen4_hyper::pending::flush")
            .expect("pending flush");
        let status = source
            .rfind("qwen4_moe_status::flush")
            .expect("status flush");
        let receipt = source.rfind("receipt.finish()?").expect("receipt finish");
        assert!(pending < status && status < receipt);
    }
}

#[test]
fn qwen4_direct_two_phase_rejects_before_any_model_or_gpu_effect() {
    let trait_impl = include_str!("../trait_impl/mod.rs");
    let capability_start = trait_impl
        .find("fn supports_prefill_twophase(&self) -> bool")
        .unwrap();
    let capability_body = &trait_impl[capability_start..];
    let capability_end = capability_body.find("fn prefill_twophase(").unwrap();
    assert!(
        capability_body[..capability_end].contains("!self.config.is_qwen4_exp()"),
        "Qwen4 must be excluded before scheduler two-phase dispatch"
    );

    let source = include_str!("../trait_impl/prefill_c.rs");
    let start = source.find("fn prefill_twophase_dispatch(").unwrap();
    let body = &source[start..];
    let reject = body.find("!self.config.is_qwen4_exp()").unwrap();
    let first_effect = body.find("self.validate_vision_prompt").unwrap();
    assert!(reject < first_effect);
}
