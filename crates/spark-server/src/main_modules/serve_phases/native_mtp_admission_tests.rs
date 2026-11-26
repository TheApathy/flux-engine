// SPDX-License-Identifier: AGPL-3.0-only

use spark_model::speculative::DraftProposerKind;

use super::{
    NativeMtpAdmission, ProposerSnapshot, assess_native_mtp_admission, validate_mtp_gate_request,
};

#[test]
fn qwen4_native_startup_receipts_are_exact_and_role_distinct() {
    let source = include_str!("native_mtp_admission.rs");
    assert_eq!(
        source
            .matches("QWEN4_SPECULATION_ADMISSION_ENGAGED mode=native-only")
            .count(),
        1
    );
    assert_eq!(
        source
            .matches("QWEN4_SPECULATION_ADMISSION_LOADED_SERIAL schema=1 mode=native-only")
            .count(),
        1
    );
    assert!(!source.contains("\"SPECULATION_ADMISSION_ENGAGED mode=native-only\""));
}

fn assess(
    native: bool,
    dflash: bool,
    disabled: bool,
    count: usize,
    arm0: Option<DraftProposerKind>,
    arm1: Option<DraftProposerKind>,
) -> anyhow::Result<NativeMtpAdmission> {
    assess_native_mtp_admission(
        DraftProposerKind::Qwen4NativeMtp,
        native,
        dflash,
        disabled,
        ProposerSnapshot {
            has_proposer: count > 0,
            arm_count: count,
            active_arm: 0,
            active_is_dflash: arm0 == Some(DraftProposerKind::DFlash),
            arm0_kind: arm0,
            arm1_kind: arm1,
        },
    )
}

#[test]
fn plain_serving_is_inert() {
    assert_eq!(
        assess_native_mtp_admission(
            DraftProposerKind::Other,
            false,
            false,
            false,
            ProposerSnapshot {
                has_proposer: false,
                arm_count: 0,
                active_arm: 0,
                active_is_dflash: false,
                arm0_kind: None,
                arm1_kind: None,
            },
        )
        .unwrap(),
        NativeMtpAdmission::NotRequested
    );
    assert_eq!(
        assess(false, false, true, 0, None, None).unwrap(),
        NativeMtpAdmission::NotRequested
    );
}

#[test]
fn native_only_requires_exactly_one_native_arm() {
    assert_eq!(
        assess(
            true,
            false,
            false,
            1,
            Some(DraftProposerKind::Qwen4NativeMtp),
            None,
        )
        .unwrap(),
        NativeMtpAdmission::NativeOnly
    );
    let disabled = assess(
        true,
        false,
        true,
        1,
        Some(DraftProposerKind::Qwen4NativeMtp),
        None,
    )
    .unwrap();
    assert_eq!(disabled, NativeMtpAdmission::NativeLoadedSerial);
    let absent = assess(true, false, false, 0, None, None)
        .unwrap_err()
        .to_string();
    assert!(absent.contains("scheduler sees no proposer"));
    let wrong_kind = assess(true, false, false, 1, Some(DraftProposerKind::Other), None)
        .unwrap_err()
        .to_string();
    assert!(wrong_kind.contains("identity mismatch"));
}

#[test]
fn dflash_only_requires_exactly_one_dflash_arm() {
    assert_eq!(
        assess(false, true, false, 1, Some(DraftProposerKind::DFlash), None,).unwrap(),
        NativeMtpAdmission::DflashOnly
    );
    let disabled = assess(false, true, true, 1, Some(DraftProposerKind::DFlash), None)
        .unwrap_err()
        .to_string();
    assert!(disabled.contains("native-MTP-only Qwen4 loaded-serial control"));
    let leaked_native = assess(
        false,
        true,
        false,
        2,
        Some(DraftProposerKind::DFlash),
        Some(DraftProposerKind::Qwen4NativeMtp),
    )
    .unwrap_err()
    .to_string();
    assert!(leaked_native.contains("exactly 1 proposer arm"));
}

#[test]
fn dual_request_requires_exact_arm_identities() {
    assert_eq!(
        assess(
            true,
            true,
            false,
            2,
            Some(DraftProposerKind::DFlash),
            Some(DraftProposerKind::Qwen4NativeMtp),
        )
        .unwrap(),
        NativeMtpAdmission::DflashPrimaryWithNativeAlternate
    );
    let disabled = assess(
        true,
        true,
        true,
        2,
        Some(DraftProposerKind::DFlash),
        Some(DraftProposerKind::Qwen4NativeMtp),
    )
    .unwrap_err()
    .to_string();
    assert!(disabled.contains("native-MTP-only Qwen4 loaded-serial control"));
    let dflash_only = assess(true, true, false, 1, Some(DraftProposerKind::DFlash), None)
        .unwrap_err()
        .to_string();
    assert!(dflash_only.contains("exactly 2 proposer arm"));
    for (arm0, arm1) in [
        (DraftProposerKind::Other, DraftProposerKind::Qwen4NativeMtp),
        (DraftProposerKind::DFlash, DraftProposerKind::Other),
        (DraftProposerKind::Qwen4NativeMtp, DraftProposerKind::DFlash),
    ] {
        let mismatch = assess(true, true, false, 2, Some(arm0), Some(arm1))
            .unwrap_err()
            .to_string();
        assert!(mismatch.contains("identity mismatch"));
    }
}

#[test]
fn scheduler_views_must_agree_with_physical_arms() {
    let valid = ProposerSnapshot {
        has_proposer: true,
        arm_count: 1,
        active_arm: 0,
        active_is_dflash: false,
        arm0_kind: Some(DraftProposerKind::Qwen4NativeMtp),
        arm1_kind: None,
    };
    for (snapshot, expected) in [
        (
            ProposerSnapshot {
                has_proposer: false,
                ..valid
            },
            "scheduler sees no proposer",
        ),
        (
            ProposerSnapshot {
                active_arm: 1,
                ..valid
            },
            "must start on physical arm 0",
        ),
        (
            ProposerSnapshot {
                active_is_dflash: true,
                ..valid
            },
            "active proposer kind disagrees",
        ),
    ] {
        let error = assess_native_mtp_admission(
            DraftProposerKind::Qwen4NativeMtp,
            true,
            false,
            false,
            snapshot,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn pinned_gate_must_select_the_named_physical_kind() {
    let native_only = ProposerSnapshot {
        has_proposer: true,
        arm_count: 1,
        active_arm: 0,
        active_is_dflash: false,
        arm0_kind: Some(DraftProposerKind::Qwen4NativeMtp),
        arm1_kind: None,
    };
    let dflash_only = ProposerSnapshot {
        active_is_dflash: true,
        arm0_kind: Some(DraftProposerKind::DFlash),
        ..native_only
    };
    let dual = ProposerSnapshot {
        arm_count: 2,
        active_is_dflash: true,
        arm0_kind: Some(DraftProposerKind::DFlash),
        arm1_kind: Some(DraftProposerKind::Qwen4NativeMtp),
        ..native_only
    };

    for gate in [None, Some("auto"), Some("force"), Some("mtp")] {
        validate_mtp_gate_request(gate, DraftProposerKind::Qwen4NativeMtp, native_only).unwrap();
    }
    validate_mtp_gate_request(
        Some("dflash"),
        DraftProposerKind::Qwen4NativeMtp,
        dflash_only,
    )
    .unwrap();
    validate_mtp_gate_request(Some("dflash"), DraftProposerKind::Qwen4NativeMtp, dual).unwrap();
    validate_mtp_gate_request(Some("mtp"), DraftProposerKind::Qwen4NativeMtp, dual).unwrap();

    let wrong_dflash = validate_mtp_gate_request(
        Some("dflash"),
        DraftProposerKind::Qwen4NativeMtp,
        native_only,
    )
    .unwrap_err()
    .to_string();
    assert!(wrong_dflash.contains("arm 0"));
    let wrong_mtp =
        validate_mtp_gate_request(Some("mtp"), DraftProposerKind::Qwen4NativeMtp, dflash_only)
            .unwrap_err()
            .to_string();
    assert!(wrong_mtp.contains("arm 0"));

    let swapped_dual = ProposerSnapshot {
        arm0_kind: Some(DraftProposerKind::Qwen4NativeMtp),
        arm1_kind: Some(DraftProposerKind::DFlash),
        ..dual
    };
    for gate in [Some("dflash"), Some("mtp")] {
        assert!(
            validate_mtp_gate_request(gate, DraftProposerKind::Qwen4NativeMtp, swapped_dual)
                .is_err()
        );
    }
}

#[test]
fn dense_native_and_dual_modes_require_other_kind() {
    let native = ProposerSnapshot {
        has_proposer: true,
        arm_count: 1,
        active_arm: 0,
        active_is_dflash: false,
        arm0_kind: Some(DraftProposerKind::Other),
        arm1_kind: None,
    };
    assert_eq!(
        assess_native_mtp_admission(DraftProposerKind::Other, true, false, false, native,).unwrap(),
        NativeMtpAdmission::NativeOnly
    );
    let dual = ProposerSnapshot {
        arm_count: 2,
        active_is_dflash: true,
        arm0_kind: Some(DraftProposerKind::DFlash),
        arm1_kind: Some(DraftProposerKind::Other),
        ..native
    };
    assert_eq!(
        assess_native_mtp_admission(DraftProposerKind::Other, true, true, false, dual,).unwrap(),
        NativeMtpAdmission::DflashPrimaryWithNativeAlternate
    );
    validate_mtp_gate_request(Some("mtp"), DraftProposerKind::Other, dual).unwrap();

    let disabled = assess_native_mtp_admission(DraftProposerKind::Other, true, false, true, native)
        .unwrap_err()
        .to_string();
    assert!(disabled.contains("native-MTP-only Qwen4 loaded-serial control"));
}
