// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed environment contract for the release-flat DFlash path.

pub(crate) struct FlatDflashContract<'a> {
    pub enabled: bool,
    pub method: &'a str,
    pub gamma: Option<usize>,
    pub window_size: usize,
    pub quantization: &'a str,
}

#[derive(Clone, Copy)]
pub(crate) struct EnvironmentRule {
    pub name: &'static str,
    allowed_when_present: &'static [&'static str],
}

#[path = "cli_dflash_environment_rules.rs"]
mod rules;

use super::dflash_lookup as lookup;

pub(crate) const MAX_NODES_ENV: &str = "ATLAS_DDTREE_MAX_NODES";
pub(crate) const DRAFT_CAP_ENV: &str = "ATLAS_DFLASH_DRAFT_CAP";
pub(crate) const CTX_WINDOW_ENV: &str = "ATLAS_DFLASH_CTX_WINDOW";
pub(crate) const QUANT_ENV: &str = "ATLAS_DFLASH_QUANT";

fn normalize_environment_value(value: Result<String, std::env::VarError>) -> Option<String> {
    match value {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => Some("<non-Unicode>".to_string()),
    }
}

pub(crate) fn live_environment_value(name: &str) -> Option<String> {
    normalize_environment_value(std::env::var(name))
}

fn describe_allowed(allowed: &[&str]) -> String {
    if allowed.is_empty() {
        "must be unset".to_string()
    } else {
        format!("must be one of: {}", allowed.join(", "))
    }
}

pub(crate) fn validate_flat_dflash_environment(
    contract: FlatDflashContract<'_>,
    mut get: impl FnMut(&str) -> Option<String>,
) -> Result<(), String> {
    if !contract.enabled {
        return Ok(());
    }

    let mut problems = Vec::new();
    if contract.method != "flat" {
        problems.push(format!(
            "--dflash-method {:?} reached the runtime gate, but only `flat` is supported; \
             refusing to load an engine that could silently execute another method",
            contract.method
        ));
    }

    // The lookup selector owns ATLAS_DFLASH_SAM/RETRIEVAL; a malformed
    // selector is reported and those controls are then held to `off`.
    let lookup = lookup::lookup_mode(&mut get).unwrap_or_else(|problem| {
        problems.push(problem);
        lookup::DflashLookup::Off
    });
    problems.extend(lookup::validate_owned_controls(lookup, &mut get));

    for rule in rules::RELEASE_ENVIRONMENT_RULES {
        if lookup::DflashLookup::owns(rule.name) {
            continue;
        }
        if let Some(value) = get(rule.name)
            && !rule.allowed_when_present.contains(&value.as_str())
        {
            let authority = if rule.name == "ATLAS_DFLASH_METHOD" {
                "; the authoritative --dflash-method flat runtime cannot be overridden"
            } else {
                ""
            };
            problems.push(format!(
                "{}={value:?} violates the release-flat DFlash contract ({}{})",
                rule.name,
                describe_allowed(rule.allowed_when_present),
                authority,
            ));
        }
    }

    validate_usize_authority(
        &mut problems,
        DRAFT_CAP_ENV,
        get(DRAFT_CAP_ENV),
        contract.gamma,
        "authoritative --dflash-gamma",
    );
    validate_usize_authority(
        &mut problems,
        CTX_WINDOW_ENV,
        get(CTX_WINDOW_ENV),
        Some(contract.window_size),
        "authoritative --dflash-window-size",
    );
    if let Some(value) = get(QUANT_ENV)
        && value != contract.quantization
    {
        problems.push(format!(
            "{QUANT_ENV}={value:?} disagrees with authoritative --dflash-quantization {:?}",
            contract.quantization
        ));
    }

    if let Some(value) = get(MAX_NODES_ENV) {
        let parsed = value.trim().parse::<usize>().ok();
        let flat_width = contract.gamma.and_then(|gamma| gamma.checked_add(1));
        if parsed != flat_width {
            problems.push(format!(
                "{MAX_NODES_ENV}={value:?} is not provably the flat verify width; \
                 explicit --dflash-gamma gives {flat_width:?}"
            ));
        }
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; "))
    }
}

fn validate_usize_authority(
    problems: &mut Vec<String>,
    name: &str,
    value: Option<String>,
    expected: Option<usize>,
    authority: &str,
) {
    let Some(value) = value else { return };
    let parsed = value.trim().parse::<usize>().ok();
    if parsed != expected {
        problems.push(format!(
            "{name}={value:?} disagrees with {authority} value {expected:?}"
        ));
    }
}

#[cfg(test)]
#[path = "cli_dflash_environment_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "cli_dflash_environment_census_tests.rs"]
mod census_tests;
