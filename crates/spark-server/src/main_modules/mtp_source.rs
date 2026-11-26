// SPDX-License-Identifier: AGPL-3.0-only

//! Which MTP routed-expert bank a Qwen4 native-MTP server runs: the target
//! checkpoint's own `mtp.layers.0.mlp.experts.*` or an `--mtp-from-path`
//! sidecar such as the bundled Inferact drafter.
//!
//! The release argv passes `--mtp-from-path` unconditionally because the
//! Offload checkpoint ships no MTP experts. Checkpoints that do ship them
//! (QAD5500 Hybrid, qad-step-4000) used to hit `merge_disjoint`'s duplicate
//! refusal. `ATLAS_MTP_SOURCE` (alias `FLUX_MTP_SOURCE`) now decides:
//!
//! * `auto` (default): the checkpoint bank wins when present; the sidecar
//!   fills in only when the checkpoint has none.
//! * `checkpoint`: never load a sidecar; a checkpoint without a bank fails.
//! * `sidecar`: the sidecar replaces the checkpoint bank when both exist.
//!
//! Only the routed experts come from a sidecar (the loader filters to
//! `mtp.layers.0.mlp.experts.{e}.`), so the router, attention and fc tensors
//! always stay the checkpoint's.

use anyhow::{Result, bail};

pub(crate) const MTP_SOURCE_ENV: &str = "ATLAS_MTP_SOURCE";
pub(crate) const MTP_EXPERT_PREFIX: &str = "mtp.layers.0.mlp.experts.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MtpSource {
    Auto,
    Checkpoint,
    Sidecar,
}

impl MtpSource {
    /// Absent or empty means `auto`; any other unknown value fails closed so a
    /// typo cannot silently pick a bank.
    pub(crate) fn parse(raw: Option<&str>) -> Result<Self> {
        match raw.map(str::trim).unwrap_or("") {
            "" => Ok(Self::Auto),
            value if value.eq_ignore_ascii_case("auto") => Ok(Self::Auto),
            value if value.eq_ignore_ascii_case("checkpoint") => Ok(Self::Checkpoint),
            value if value.eq_ignore_ascii_case("sidecar") => Ok(Self::Sidecar),
            other => bail!("{MTP_SOURCE_ENV}={other:?} is not one of auto, checkpoint, sidecar"),
        }
    }

    pub(crate) fn from_env() -> Result<Self> {
        match std::env::var(MTP_SOURCE_ENV) {
            Ok(value) => Self::parse(Some(&value)),
            Err(std::env::VarError::NotPresent) => Ok(Self::Auto),
            Err(std::env::VarError::NotUnicode(_)) => {
                bail!("{MTP_SOURCE_ENV} is not valid Unicode")
            }
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Checkpoint => "checkpoint",
            Self::Sidecar => "sidecar",
        }
    }

    /// Whether to look for a sidecar at all, before any directory resolves.
    pub(crate) fn wants_sidecar(self, target_has_bank: bool) -> bool {
        match self {
            Self::Auto => !target_has_bank,
            Self::Checkpoint => false,
            Self::Sidecar => true,
        }
    }
}

/// What to do with a resolved sidecar directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MtpSidecarPlan {
    /// Do not load it; the checkpoint bank serves.
    Skip,
    /// Load it into a target that has no bank.
    Merge,
    /// Load it, take the target bank out, then merge.
    Replace,
}

pub(crate) fn plan_sidecar(source: MtpSource, target_has_bank: bool) -> Result<MtpSidecarPlan> {
    Ok(match (source, target_has_bank) {
        (MtpSource::Auto | MtpSource::Checkpoint, true) => MtpSidecarPlan::Skip,
        (MtpSource::Sidecar, true) => MtpSidecarPlan::Replace,
        (MtpSource::Auto | MtpSource::Sidecar, false) => MtpSidecarPlan::Merge,
        (MtpSource::Checkpoint, false) => bail!(
            "{MTP_SOURCE_ENV}=checkpoint but the target checkpoint has no MTP routed experts \
             ({MTP_EXPERT_PREFIX}*); use auto or sidecar with --mtp-from-path"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_the_closed_set_and_defaults_to_auto() {
        assert_eq!(MtpSource::parse(None).unwrap(), MtpSource::Auto);
        assert_eq!(MtpSource::parse(Some("")).unwrap(), MtpSource::Auto);
        assert_eq!(MtpSource::parse(Some(" Auto ")).unwrap(), MtpSource::Auto);
        assert_eq!(
            MtpSource::parse(Some("CHECKPOINT")).unwrap(),
            MtpSource::Checkpoint
        );
        assert_eq!(
            MtpSource::parse(Some("sidecar")).unwrap(),
            MtpSource::Sidecar
        );
        for bad in ["side-car", "1", "inferact", "override"] {
            assert!(MtpSource::parse(Some(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn checkpoint_bank_wins_unless_sidecar_is_explicit() {
        assert_eq!(
            plan_sidecar(MtpSource::Auto, true).unwrap(),
            MtpSidecarPlan::Skip
        );
        assert_eq!(
            plan_sidecar(MtpSource::Checkpoint, true).unwrap(),
            MtpSidecarPlan::Skip
        );
        assert_eq!(
            plan_sidecar(MtpSource::Sidecar, true).unwrap(),
            MtpSidecarPlan::Replace
        );
    }

    #[test]
    fn bankless_target_merges_or_fails_closed() {
        assert_eq!(
            plan_sidecar(MtpSource::Auto, false).unwrap(),
            MtpSidecarPlan::Merge
        );
        assert_eq!(
            plan_sidecar(MtpSource::Sidecar, false).unwrap(),
            MtpSidecarPlan::Merge
        );
        let error = plan_sidecar(MtpSource::Checkpoint, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("no MTP routed experts"), "{error}");
    }

    #[test]
    fn sidecar_lookup_follows_the_policy() {
        assert!(MtpSource::Auto.wants_sidecar(false));
        assert!(!MtpSource::Auto.wants_sidecar(true));
        assert!(!MtpSource::Checkpoint.wants_sidecar(false));
        assert!(MtpSource::Sidecar.wants_sidecar(true));
    }
}
