// SPDX-License-Identifier: AGPL-3.0-only

//! The env-gate catalog: every `ATLAS_*`/`FLUX_*` name this build reads, with
//! its class. Generated from source by `repro-kit/tools/env_inventory.py`
//! (with trailing-underscore prefix fragments and synthetic test dummies
//! excluded); a name absent here has no reader in this build.

use std::collections::BTreeMap;

const BUILTIN: &str = include_str!("env_catalog.tsv");

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EnvClass {
    Numerics,
    Route,
    Resource,
    Observability,
    Debug,
}

impl EnvClass {
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "NUMERICS" => Some(Self::Numerics),
            "ROUTE" => Some(Self::Route),
            "RESOURCE" => Some(Self::Resource),
            "OBSERVABILITY" => Some(Self::Observability),
            "DEBUG" => Some(Self::Debug),
            _ => None,
        }
    }

    pub fn is_operational(self) -> bool {
        matches!(self, Self::Resource | Self::Observability | Self::Debug)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Catalog {
    classes: BTreeMap<String, EnvClass>,
}

impl Catalog {
    /// The catalog generated for this build.
    pub fn builtin() -> Result<Self, String> {
        Self::parse(BUILTIN)
    }

    /// `name<TAB>CLASS` lines; `#` comments and blank lines are skipped.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut classes = BTreeMap::new();
        for (no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (name, class) = line
                .split_once('\t')
                .ok_or_else(|| format!("env catalog line {}: expected name<TAB>class", no + 1))?;
            let class = EnvClass::parse(class.trim())
                .ok_or_else(|| format!("env catalog line {}: unknown class {class:?}", no + 1))?;
            classes.insert(name.trim().to_string(), class);
        }
        Ok(Self { classes })
    }

    pub fn class(&self, name: &str) -> Option<EnvClass> {
        self.classes.get(name).copied()
    }

    pub fn has_reader(&self, name: &str) -> bool {
        self.classes.contains_key(name)
    }

    pub fn names_of(&self, class: EnvClass) -> impl Iterator<Item = &str> {
        self.classes
            .iter()
            .filter(move |(_, c)| **c == class)
            .map(|(n, _)| n.as_str())
    }

    /// The name the engine reads for `name`: `main` copies every non-meta
    /// `FLUX_X` onto its `ATLAS_` alias before startup (an existing `ATLAS_`
    /// value wins), so the `ATLAS_` spelling is the one that takes effect.
    /// Uses the same mapping function as `main`.
    pub fn canonical<'a>(&self, name: &'a str) -> std::borrow::Cow<'a, str> {
        // A `FLUX_` name with its own reader keeps its spelling when the
        // alias has none (e.g. FLUX_TARGET_QUANT), so it is still classified.
        match crate::flux_env_alias_key(std::ffi::OsStr::new(name)) {
            Some(alias) if self.has_reader(&alias) || !self.has_reader(name) => {
                std::borrow::Cow::Owned(alias)
            }
            _ => std::borrow::Cow::Borrowed(name),
        }
    }
}
