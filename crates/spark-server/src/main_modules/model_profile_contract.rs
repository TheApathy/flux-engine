// SPDX-License-Identifier: AGPL-3.0-only

//! Exact launch-environment contracts for measured built-in profiles.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail};
use ring::digest::{SHA256, digest};

use crate::cli::dflash_lookup::{DflashLookup, LOOKUP_ENV};
use crate::recipe::yaml::Yaml;

const POLICY: &str = "sealed-v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProfileContract {
    kind: String,
    environment_sha256: String,
    forbidden_env: BTreeSet<String>,
    build_track: Option<String>,
}

impl super::ModelProfile {
    /// Validate the complete sealed environment and compiled build track, returning the
    /// receipt which may be logged only after every check succeeds.
    pub fn validate_environment_with(
        &self,
        actual: impl FnMut(&str) -> Result<Option<String>>,
        compiled_build_track: Option<&str>,
    ) -> Result<Option<super::ProfileReceipt>> {
        self.contract
            .as_ref()
            .map(|contract| {
                contract.validate(&self.recipe_id, &self.env, actual, compiled_build_track)
            })
            .transpose()
    }

    pub fn profile_kind(&self) -> Option<&str> {
        self.contract.as_ref().map(ProfileContract::kind)
    }
}

/// Convert one process value without treating non-Unicode bytes as "unset".
pub(crate) fn unicode_environment_value(
    name: &str,
    value: Option<std::ffi::OsString>,
) -> Result<Option<String>> {
    value
        .map(|value| {
            value.into_string().map_err(|_| {
                anyhow::anyhow!("{name} is not valid Unicode in the selected launch environment")
            })
        })
        .transpose()
}

pub(crate) fn log_receipt(receipt: &super::ProfileReceipt) {
    tracing::info!(
        schema = 1,
        recipe_id = %receipt.recipe_id,
        profile_kind = %receipt.profile_kind,
        environment_sha256 = %receipt.environment_sha256,
        build_track = receipt.build_track.as_deref().unwrap_or("none"),
        "ATLAS_PROFILE_RECEIPT"
    );
    if let Some(lookup) = receipt.lookup {
        tracing::info!(
            recipe_id = %receipt.recipe_id,
            lookup,
            "ATLAS_PROFILE_LOOKUP_OVERLAY"
        );
    }
}

impl ProfileContract {
    pub(super) fn parse(
        recipe_id: &str,
        block: &BTreeMap<String, Yaml>,
        env: &BTreeMap<String, String>,
    ) -> Result<Option<Self>> {
        let field = |name: &str| block.get(name).and_then(Yaml::as_str);
        let any = [
            "environment_policy",
            "profile_kind",
            "environment_sha256",
            "forbidden_env",
            "build_track",
        ]
        .iter()
        .any(|name| block.contains_key(*name));
        if !any {
            return Ok(None);
        }
        let policy = field("environment_policy")
            .with_context(|| format!("{recipe_id}: atlas.environment_policy is required"))?;
        if policy != POLICY {
            bail!("{recipe_id}: atlas.environment_policy must be {POLICY:?}, got {policy:?}");
        }
        let kind = required_token(recipe_id, "profile_kind", field("profile_kind"))?;
        let expected = field("environment_sha256")
            .with_context(|| format!("{recipe_id}: atlas.environment_sha256 is required"))?;
        if !is_sha256(expected) {
            bail!("{recipe_id}: atlas.environment_sha256 must be 64 lowercase hex digits");
        }
        if let Some(name) = env.keys().find(|name| !valid_environment_name(name)) {
            bail!("{recipe_id}: invalid sealed environment name {name:?}");
        }
        let found = environment_sha256(env);
        if found != expected {
            bail!(
                "{recipe_id}: sealed environment receipt mismatch: declared {expected}, computed {found}"
            );
        }
        let forbidden_env = match block.get("forbidden_env") {
            None => BTreeSet::new(),
            Some(Yaml::List(values)) => values
                .iter()
                .map(|value| {
                    let name = value.as_str().with_context(|| {
                        format!("{recipe_id}: atlas.forbidden_env entries must be scalars")
                    })?;
                    if !valid_environment_name(name) {
                        bail!("{recipe_id}: invalid forbidden environment name {name:?}");
                    }
                    Ok(name.to_string())
                })
                .collect::<Result<BTreeSet<_>>>()?,
            Some(_) => bail!("{recipe_id}: atlas.forbidden_env must be a list"),
        };
        if let Some(name) = forbidden_env.iter().find(|name| env.contains_key(*name)) {
            bail!("{recipe_id}: {name} is both required and forbidden");
        }
        let build_track = field("build_track")
            .map(|value| required_token(recipe_id, "build_track", Some(value)))
            .transpose()?;
        Ok(Some(Self {
            kind,
            environment_sha256: expected.to_string(),
            forbidden_env,
            build_track,
        }))
    }

    pub(super) fn kind(&self) -> &str {
        &self.kind
    }

    pub(super) fn validate(
        &self,
        recipe_id: &str,
        expected: &BTreeMap<String, String>,
        mut actual: impl FnMut(&str) -> Result<Option<String>>,
        compiled_build_track: Option<&str>,
    ) -> Result<super::ProfileReceipt> {
        if self.build_track.as_deref() != compiled_build_track && self.build_track.is_some() {
            bail!(
                "{recipe_id}: profile build track {:?} does not match this binary's {:?}; use the separately sealed binary",
                self.build_track,
                compiled_build_track
            );
        }
        let lookup = sealed_lookup_overlay(recipe_id, expected, &mut actual)?;
        let mut realized = BTreeMap::new();
        for (name, wanted) in expected {
            let sealed = lookup
                .sealed_value(name, wanted)
                .unwrap_or_else(|| wanted.clone());
            let found = actual(name)?;
            if found.as_deref() != Some(sealed.as_str()) {
                bail!(
                    "{recipe_id}: sealed environment differs at {name}: expected {sealed:?}, got {:?}",
                    found.as_deref().unwrap_or("<unset>")
                );
            }
            // The receipt hashes the declared base. A lookup overlay changes
            // only the selector-owned keys, each checked exactly above, and
            // is reported beside the receipt rather than folded into it.
            realized.insert(name.clone(), wanted.clone());
        }
        for name in &self.forbidden_env {
            if let Some(value) = actual(name)? {
                bail!("{recipe_id}: forbidden environment {name} is set to {value:?}");
            }
        }
        let found = environment_sha256(&realized);
        if found != self.environment_sha256 {
            bail!("{recipe_id}: realized environment receipt changed after validation");
        }
        Ok(super::ProfileReceipt {
            recipe_id: recipe_id.to_string(),
            profile_kind: self.kind.clone(),
            environment_sha256: found,
            build_track: self.build_track.clone(),
            lookup: (lookup != DflashLookup::Off).then_some(lookup.as_str()),
        })
    }
}

/// The reviewed DFlash lookup selector may overlay a sealed profile only when
/// that profile seals a selector-owned control (the dense DFlash profiles pin
/// `ATLAS_DFLASH_SAM=0`). Every other profile ignores the selector entirely.
fn sealed_lookup_overlay(
    recipe_id: &str,
    expected: &BTreeMap<String, String>,
    actual: &mut impl FnMut(&str) -> Result<Option<String>>,
) -> Result<DflashLookup> {
    if !expected.keys().any(|name| DflashLookup::owns(name)) {
        return Ok(DflashLookup::Off);
    }
    DflashLookup::parse(actual(LOOKUP_ENV)?.as_deref())
        .map_err(|problem| anyhow::anyhow!("{recipe_id}: {problem}"))
}

fn required_token(recipe_id: &str, field: &str, value: Option<&str>) -> Result<String> {
    let value = value.with_context(|| format!("{recipe_id}: atlas.{field} is required"))?;
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        bail!("{recipe_id}: atlas.{field} must be a lowercase slug, got {value:?}");
    }
    Ok(value.to_string())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_environment_name(value: &str) -> bool {
    value.len() > "ATLAS_".len()
        && value.starts_with("ATLAS_")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
}

pub(super) fn environment_sha256(env: &BTreeMap<String, String>) -> String {
    let mut canonical = Vec::new();
    for (name, value) in env {
        canonical.extend_from_slice(name.as_bytes());
        canonical.push(b'=');
        canonical.extend_from_slice(value.as_bytes());
        canonical.push(b'\n');
    }
    digest(&SHA256, &canonical)
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
