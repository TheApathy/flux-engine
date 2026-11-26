// SPDX-License-Identifier: AGPL-3.0-only

//! Reviewed opt-in selector for the DFlash context-lookup drafters.
//!
//! `ATLAS_DFLASH_LOOKUP` (spelled `FLUX_DFLASH_LOOKUP` by operators; the
//! startup alias pass maps it) is the only way the release-flat DFlash
//! contract admits a lookup drafter. The engine still reads its historical
//! controls (`ATLAS_DFLASH_SAM`, `ATLAS_DFLASH_RETRIEVAL`); this selector owns
//! exactly those two names and demands that they agree with it, so the
//! decision is one explicit value instead of a relaxed allowlist. Every SAM
//! tuning control stays unset: the engine defaults are the measured
//! configuration (adaptive misfire gate on, min_accept 3, misfire_limit 3,
//! cooldown 24, hybrid_min = l_max = 16).
//!
//! Lookup drafts are proposals only. The flat verify commits the target's
//! greedy token and accepts a draft solely when it equals that token, so the
//! selector can change throughput but never committed output.

pub(crate) const LOOKUP_ENV: &str = "ATLAS_DFLASH_LOOKUP";
pub(crate) const SAM_ENV: &str = "ATLAS_DFLASH_SAM";
pub(crate) const RETRIEVAL_ENV: &str = "ATLAS_DFLASH_RETRIEVAL";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DflashLookup {
    /// Default: neural drafter only. Both engine controls absent or `0`.
    Off,
    /// Longest-suffix (SAM-style) matcher over prompt + generated tokens.
    Sam,
    /// Legacy fixed-window (l_min..=l_max) matcher.
    Retrieval,
}

/// One engine control as the selected mode constrains it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OwnedControl {
    pub name: &'static str,
    /// Value the mode requires; `None` means absent or `0`.
    pub required: Option<&'static str>,
}

impl DflashLookup {
    pub(crate) fn parse(value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("off") => Ok(Self::Off),
            Some("sam") => Ok(Self::Sam),
            Some("retrieval") => Ok(Self::Retrieval),
            Some(other) => Err(format!(
                "{LOOKUP_ENV}={other:?} is not a DFlash lookup mode (must be one of: off, sam, retrieval)"
            )),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Sam => "sam",
            Self::Retrieval => "retrieval",
        }
    }

    pub(crate) fn owned_controls(self) -> [OwnedControl; 2] {
        let (sam, retrieval) = match self {
            Self::Off => (None, None),
            Self::Sam => (Some("1"), None),
            Self::Retrieval => (None, Some("1")),
        };
        [
            OwnedControl {
                name: SAM_ENV,
                required: sam,
            },
            OwnedControl {
                name: RETRIEVAL_ENV,
                required: retrieval,
            },
        ]
    }

    pub(crate) fn owns(name: &str) -> bool {
        name == SAM_ENV || name == RETRIEVAL_ENV
    }

    /// The exact value `name` must carry under this mode, given the value a
    /// sealed profile declares for the default (`off`) run. `None` when the
    /// selector does not own `name`.
    pub(crate) fn sealed_value(self, name: &str, declared: &str) -> Option<String> {
        let control = self
            .owned_controls()
            .into_iter()
            .find(|control| control.name == name)?;
        Some(control.required.unwrap_or(declared).to_string())
    }
}

/// Selector mode from the environment, rejecting malformed values.
pub(crate) fn lookup_mode(
    mut get: impl FnMut(&str) -> Option<String>,
) -> Result<DflashLookup, String> {
    DflashLookup::parse(get(LOOKUP_ENV).as_deref())
}

/// Engine controls the selector implies and the process does not set yet.
/// An explicit operator value is never replaced: if it disagrees with the
/// selector, the release contract refuses it instead. A malformed selector
/// implies nothing (the contract reports it).
pub(crate) fn implied_controls(
    mut get: impl FnMut(&str) -> Option<String>,
) -> Vec<(&'static str, &'static str)> {
    let Ok(mode) = lookup_mode(&mut get) else {
        return Vec::new();
    };
    mode.owned_controls()
        .into_iter()
        .filter_map(|control| Some((control.name, control.required?)))
        .filter(|(name, _)| get(name).is_none())
        .collect()
}

/// Problems with the selector-owned controls under the release-flat contract.
pub(crate) fn validate_owned_controls(
    mode: DflashLookup,
    mut get: impl FnMut(&str) -> Option<String>,
) -> Vec<String> {
    let mut problems = Vec::new();
    for control in mode.owned_controls() {
        let value = get(control.name);
        let admitted = match control.required {
            Some(required) => value.as_deref() == Some(required),
            None => matches!(value.as_deref(), None | Some("0")),
        };
        if admitted {
            continue;
        }
        let wanted = match control.required {
            Some(required) => format!("must be {required}"),
            None => "must be unset or 0".to_string(),
        };
        let shown = value.as_deref().unwrap_or("<unset>");
        problems.push(format!(
            "{}={shown:?} violates the release-flat DFlash contract under {LOOKUP_ENV}={} ({wanted}; \
             lookup drafting is enabled only through FLUX_DFLASH_LOOKUP=sam|retrieval)",
            control.name,
            mode.as_str(),
        ));
    }
    problems
}

/// One-line receipt proving which lookup drafter the load engaged.
pub(crate) fn engaged_receipt(mode: DflashLookup) -> Option<String> {
    let controls = match mode {
        DflashLookup::Off => return None,
        DflashLookup::Sam => "ATLAS_DFLASH_SAM=1 matcher=longest-suffix",
        DflashLookup::Retrieval => "ATLAS_DFLASH_RETRIEVAL=1 matcher=fixed-window",
    };
    Some(format!(
        "DFLASH_LOOKUP_ENGAGED mode={} {controls} adaptive=default(min_accept=3,misfire_limit=3,\
         cooldown=24) hybrid_min=l_max=16 verify=flat-greedy-oracle",
        mode.as_str()
    ))
}

#[cfg(test)]
#[path = "cli_dflash_lookup_tests.rs"]
mod tests;
