// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed admission for explicitly requested draft proposers.
//!
//! A missing/rejected checkpoint or contradictory pinned gate must not degrade
//! into plain decode or select a differently typed physical arm while leaving
//! the process apparently healthy.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_model::speculative::DraftProposerKind;

use crate::cli;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeMtpAdmission {
    NotRequested,
    DflashOnly,
    NativeOnly,
    NativeLoadedSerial,
    DflashPrimaryWithNativeAlternate,
}

#[derive(Debug, Clone, Copy)]
struct ProposerSnapshot {
    has_proposer: bool,
    arm_count: usize,
    active_arm: u8,
    active_is_dflash: bool,
    arm0_kind: Option<DraftProposerKind>,
    arm1_kind: Option<DraftProposerKind>,
}

fn assess_native_mtp_admission(
    expected_native_kind: DraftProposerKind,
    native_requested: bool,
    dflash_requested: bool,
    speculation_forced_disabled: bool,
    proposer: ProposerSnapshot,
) -> Result<NativeMtpAdmission> {
    if !(native_requested || dflash_requested) {
        return Ok(NativeMtpAdmission::NotRequested);
    }

    anyhow::ensure!(
        proposer.has_proposer,
        "speculative serving was requested, but the scheduler sees no proposer"
    );
    anyhow::ensure!(
        proposer.active_arm == 0,
        "proposer must start on physical arm 0; active arm is {}",
        proposer.active_arm
    );

    let (mode, expected_count, expected_arm0, expected_arm1) =
        match (dflash_requested, native_requested) {
            (false, true) => (
                NativeMtpAdmission::NativeOnly,
                1,
                Some(expected_native_kind),
                None,
            ),
            (true, false) => (
                NativeMtpAdmission::DflashOnly,
                1,
                Some(DraftProposerKind::DFlash),
                None,
            ),
            (true, true) => (
                NativeMtpAdmission::DflashPrimaryWithNativeAlternate,
                2,
                Some(DraftProposerKind::DFlash),
                Some(expected_native_kind),
            ),
            (false, false) => unreachable!("plain Qwen4 returned above"),
        };
    anyhow::ensure!(
        proposer.arm_count == expected_count,
        "{mode:?} requires exactly {expected_count} proposer arm(s); built {}",
        proposer.arm_count
    );
    anyhow::ensure!(
        proposer.arm0_kind == expected_arm0 && proposer.arm1_kind == expected_arm1,
        "{mode:?} proposer identity mismatch: expected [{expected_arm0:?}, {expected_arm1:?}], built [{:?}, {:?}]",
        proposer.arm0_kind,
        proposer.arm1_kind,
    );
    anyhow::ensure!(
        proposer.active_is_dflash == dflash_requested,
        "{mode:?} active proposer kind disagrees with physical arm 0"
    );
    if speculation_forced_disabled {
        anyhow::ensure!(
            expected_native_kind == DraftProposerKind::Qwen4NativeMtp
                && mode == NativeMtpAdmission::NativeOnly,
            "ATLAS_DISABLE_SPECULATION=1 is admitted only for a native-MTP-only Qwen4 loaded-serial control"
        );
        Ok(NativeMtpAdmission::NativeLoadedSerial)
    } else {
        Ok(mode)
    }
}

fn validate_mtp_gate_request(
    gate: Option<&str>,
    expected_native_kind: DraftProposerKind,
    proposer: ProposerSnapshot,
) -> Result<()> {
    let Some(gate) = gate else {
        return Ok(());
    };
    let (selected_arm, expected, actual) = match gate {
        "auto" | "force" => return Ok(()),
        "dflash" => (0, DraftProposerKind::DFlash, proposer.arm0_kind),
        "mtp" if proposer.arm_count >= 2 => (1, expected_native_kind, proposer.arm1_kind),
        "mtp" => (0, expected_native_kind, proposer.arm0_kind),
        _ => anyhow::bail!("--mtp-gate {gate}: unknown gate mode"),
    };
    anyhow::ensure!(
        actual == Some(expected),
        "--mtp-gate {gate} selects physical arm {selected_arm}, but that arm is {actual:?}, not {expected:?}"
    );
    Ok(())
}

pub(in crate::main_modules) fn speculation_forced_disabled() -> Result<bool> {
    match std::env::var("ATLAS_DISABLE_SPECULATION") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("ATLAS_DISABLE_SPECULATION is not valid Unicode")
        }
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Ok(_) => anyhow::bail!("ATLAS_DISABLE_SPECULATION must be exactly 0 or 1"),
    }
}

pub(crate) fn validate_speculation_admission(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    model: &dyn spark_model::traits::Model,
    forced_disabled: bool,
) -> Result<()> {
    if !(args.speculative || args.dflash)
        || config.pp_rank > 0
        || config.tp_rank > 0
        || config.ep_rank > 0
    {
        return Ok(());
    }
    let expected_native_kind = if config.is_qwen4_exp() {
        DraftProposerKind::Qwen4NativeMtp
    } else {
        DraftProposerKind::Other
    };
    let proposer = ProposerSnapshot {
        has_proposer: model.has_proposer(),
        arm_count: model.proposer_arm_count(),
        active_arm: model.proposer_arm(),
        active_is_dflash: model.proposer_is_dflash(),
        arm0_kind: model.proposer_arm_kind(0),
        arm1_kind: model.proposer_arm_kind(1),
    };
    let admission = assess_native_mtp_admission(
        expected_native_kind,
        args.speculative,
        args.dflash,
        forced_disabled,
        proposer,
    )?;
    if matches!(
        admission,
        NativeMtpAdmission::NativeOnly
            | NativeMtpAdmission::NativeLoadedSerial
            | NativeMtpAdmission::DflashPrimaryWithNativeAlternate
    ) {
        anyhow::ensure!(
            model.speculation_forced_disabled() == forced_disabled,
            "Qwen4 native-MTP model control disagrees with immutable server admission"
        );
    }
    anyhow::ensure!(
        admission != NativeMtpAdmission::NativeLoadedSerial || args.mtp_gate.is_none(),
        "Qwen4 native-MTP loaded-serial control cannot also select --mtp-gate"
    );
    validate_mtp_gate_request(args.mtp_gate.as_deref(), expected_native_kind, proposer)?;
    match admission {
        NativeMtpAdmission::NotRequested => {}
        NativeMtpAdmission::DflashOnly => tracing::info!(
            proposer_arms = model.proposer_arm_count(),
            "SPECULATION_ADMISSION_ENGAGED mode=dflash-only"
        ),
        NativeMtpAdmission::NativeOnly => tracing::info!(
            proposer_arms = model.proposer_arm_count(),
            "QWEN4_SPECULATION_ADMISSION_ENGAGED mode=native-only"
        ),
        NativeMtpAdmission::NativeLoadedSerial => tracing::info!(
            proposer_arms = model.proposer_arm_count(),
            "QWEN4_SPECULATION_ADMISSION_LOADED_SERIAL schema=1 mode=native-only control=ATLAS_DISABLE_SPECULATION scheduler_path=ordinary-target-only"
        ),
        NativeMtpAdmission::DflashPrimaryWithNativeAlternate => tracing::info!(
            proposer_arms = model.proposer_arm_count(),
            "SPECULATION_ADMISSION_ENGAGED mode=dflash-primary-native-alternate"
        ),
    }
    Ok(())
}

#[cfg(test)]
#[path = "native_mtp_admission_tests.rs"]
mod tests;
