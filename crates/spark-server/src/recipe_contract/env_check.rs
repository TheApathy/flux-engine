// SPDX-License-Identifier: AGPL-3.0-only

//! Check the process environment against the recipe's declarations.

use super::catalog::{Catalog, EnvClass};
use super::precision;
use super::refusal::Refusal;
use super::schema::{OperationalRule, RecipeFile, RoutePolicy};
use std::collections::BTreeMap;

/// Where an effective value came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Pinned by the recipe and present in the environment with that value.
    Recipe,
    /// Not set and not pinned: the engine's code default applies.
    Default,
    /// An operational var the recipe allows (declared range, or undeclared
    /// RESOURCE/OBSERVABILITY/DEBUG).
    EnvAllowed,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Recipe => "recipe",
            Self::Default => "default",
            Self::EnvAllowed => "env-allowed",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    /// `None` when unset (code default).
    pub value: Option<String>,
    pub source: Source,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EnvReport {
    pub numerics: Vec<Entry>,
    pub route: Vec<Entry>,
    /// ROUTE vars set but not declared (allowed under `route_policy = "flag"`).
    pub route_undeclared: Vec<Entry>,
    pub operational: Vec<Entry>,
    /// Set `ATLAS_*`/`FLUX_*` names with no reader in this build.
    pub ignored: Vec<String>,
    /// `FLUX_` names whose differing value lost to the `ATLAS_` spelling.
    pub alias_shadowed: Vec<String>,
}

/// Fold launcher-level `FLUX_X` mirrors onto the `ATLAS_X` the engine reads.
///
/// Precedence matches `main`'s alias sync: an `ATLAS_` value always wins and
/// the engine never reads the `FLUX_` spelling. A `FLUX_` value that differs
/// is not a refusal; it is returned in `shadowed` and printed in the receipt.
pub fn canonicalize(raw: &BTreeMap<String, String>, catalog: &Catalog) -> Canonical {
    let mut env: BTreeMap<String, String> = raw
        .iter()
        .filter(|(name, _)| catalog.canonical(name) == name.as_str())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut shadowed = Vec::new();
    for (name, value) in raw {
        let canon = catalog.canonical(name).into_owned();
        if canon == *name {
            continue;
        }
        match env.get(&canon) {
            Some(winner) if winner != value => shadowed.push(name.clone()),
            Some(_) => {}
            None => {
                env.insert(canon, value.clone());
            }
        }
    }
    Canonical { env, shadowed }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Canonical {
    pub env: BTreeMap<String, String>,
    /// `FLUX_` names whose value lost to an existing `ATLAS_` value.
    pub shadowed: Vec<String>,
}

/// Canonicalize and class-check the recipe's own env tables.
pub fn validate_declarations(recipe: &RecipeFile, catalog: &Catalog) -> Result<Declared, Refusal> {
    let numerics = declared(&recipe.numerics_env, catalog, "numerics_env", |c| {
        c == EnvClass::Numerics
    })?;
    let route = declared(&recipe.route_env, catalog, "route_env", |c| {
        c == EnvClass::Route
    })?;
    if let Some(name) = numerics.keys().find(|n| precision::is_owned(n)) {
        return Err(Refusal::Invalid(format!(
            "{name} belongs to [precision], not numerics_env"
        )));
    }
    if numerics.get("ATLAS_SSM_H_FP16").is_some_and(|v| v != "0") {
        return Err(Refusal::Invalid(
            "gdn_state = fp32 forbids ATLAS_SSM_H_FP16".into(),
        ));
    }
    let mut operational = BTreeMap::new();
    for (name, rule) in &recipe.operational_env {
        let canon = catalog.canonical(name).into_owned();
        match catalog.class(&canon) {
            Some(c) if c.is_operational() => {}
            other => {
                return Err(Refusal::Invalid(format!(
                    "operational_env {name} has class {other:?}"
                )));
            }
        }
        if rule.values.is_some() == (rule.min.is_some() || rule.max.is_some()) {
            return Err(Refusal::Invalid(format!(
                "operational_env {name}: give values or min/max"
            )));
        }
        operational.insert(canon, rule.clone());
    }
    Ok(Declared {
        numerics,
        route,
        operational,
    })
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Declared {
    pub numerics: BTreeMap<String, String>,
    pub route: BTreeMap<String, String>,
    pub operational: BTreeMap<String, OperationalRule>,
}

fn declared(
    table: &BTreeMap<String, String>,
    catalog: &Catalog,
    label: &str,
    class_ok: impl Fn(EnvClass) -> bool,
) -> Result<BTreeMap<String, String>, Refusal> {
    let mut out = BTreeMap::new();
    for (name, value) in table {
        let canon = catalog.canonical(name).into_owned();
        match catalog.class(&canon) {
            Some(c) if class_ok(c) => {}
            Some(c) => return Err(Refusal::Invalid(format!("{label} {name} has class {c:?}"))),
            None => {
                return Err(Refusal::Invalid(format!(
                    "{label} {name} has no reader in this build"
                )));
            }
        }
        if out
            .insert(canon.clone(), value.clone())
            .is_some_and(|prev| &prev != value)
        {
            return Err(Refusal::Invalid(format!(
                "{label} {canon} declared twice with different values"
            )));
        }
    }
    Ok(out)
}

/// Check a canonical environment against the declarations.
pub fn check(
    env: &BTreeMap<String, String>,
    declared: &Declared,
    policy: RoutePolicy,
    catalog: &Catalog,
) -> Result<EnvReport, Vec<Refusal>> {
    let mut report = EnvReport::default();
    let mut refusals = Vec::new();
    for (name, value) in env {
        let entry = |source| Entry {
            name: name.clone(),
            value: Some(value.clone()),
            source,
        };
        match catalog.class(name) {
            // Absent from the generated build catalog: no production reader
            // exists in this build, so record in `ignored` for receipt visibility.
            None => report.ignored.push(name.clone()),
            Some(EnvClass::Numerics) if precision::is_owned(name) => {}
            Some(EnvClass::Numerics) => match declared.numerics.get(name) {
                Some(want) if want == value => report.numerics.push(entry(Source::Recipe)),
                Some(want) => refusals.push(mismatch(name, want, value)),
                None => refusals.push(Refusal::UnknownNumericsEnv(name.clone())),
            },
            Some(EnvClass::Route) => match declared.route.get(name) {
                Some(want) if want == value => report.route.push(entry(Source::Recipe)),
                Some(want) => refusals.push(mismatch(name, want, value)),
                None if policy == RoutePolicy::Refuse => {
                    refusals.push(Refusal::UnknownRouteEnv(name.clone()))
                }
                None => report.route_undeclared.push(entry(Source::EnvAllowed)),
            },
            Some(_) => match declared.operational.get(name) {
                Some(rule) if !rule.allows(value) => {
                    refusals.push(Refusal::OperationalOutOfRange {
                        name: name.clone(),
                        got: value.clone(),
                    })
                }
                _ => report.operational.push(entry(Source::EnvAllowed)),
            },
        }
    }
    for name in declared.numerics.keys().chain(declared.route.keys()) {
        if !env.contains_key(name) {
            refusals.push(Refusal::EnvMissing(name.clone()));
        }
    }
    for name in catalog.names_of(EnvClass::Numerics) {
        if !env.contains_key(name) && !precision::is_owned(name) {
            report.numerics.push(Entry {
                name: name.to_string(),
                value: None,
                source: Source::Default,
            });
        }
    }
    report.numerics.sort_by(|a, b| a.name.cmp(&b.name));
    if refusals.is_empty() {
        Ok(report)
    } else {
        Err(refusals)
    }
}

fn mismatch(name: &str, want: &str, got: &str) -> Refusal {
    Refusal::EnvMismatch {
        name: name.into(),
        want: want.into(),
        got: got.into(),
    }
}
