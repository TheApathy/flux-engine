// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeSet;

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum ModelSelector {
    All,
    Exact(Vec<String>),
}

/// Parse the strict selector syntax accepted by `ATLAS_TARGET_MODEL`.
///
/// Exact names are sorted so target indexes do not depend on the order in
/// which a caller lists them. Whitespace is not trimmed: model names must
/// match their kernel directory exactly.
pub(crate) fn parse_model_selector(spec: &str) -> Result<ModelSelector, String> {
    if spec == "*" {
        return Ok(ModelSelector::All);
    }

    let mut names = BTreeSet::new();
    for name in spec.split(',') {
        if name.is_empty() {
            return Err("model list contains an empty entry".to_owned());
        }
        if name.trim() != name {
            return Err(format!(
                "model name {name:?} contains surrounding whitespace"
            ));
        }
        if name == "*" {
            return Err("wildcard '*' cannot be mixed with exact model names".to_owned());
        }
        if !names.insert(name.to_owned()) {
            return Err(format!("duplicate model name {name:?}"));
        }
    }

    Ok(ModelSelector::Exact(names.into_iter().collect()))
}

/// Resolve a selector against registered model directory names.
pub(crate) fn resolve_model_names(
    spec: &str,
    mut available: Vec<String>,
) -> Result<Vec<String>, String> {
    available.sort();
    available.dedup();

    match parse_model_selector(spec)? {
        ModelSelector::All => Ok(available),
        ModelSelector::Exact(names) => {
            if let Some(missing) = names
                .iter()
                .find(|name| available.binary_search(name).is_err())
            {
                return Err(format!(
                    "model kernel directory {missing:?} is not registered; available models: {}",
                    available.join(", ")
                ));
            }
            Ok(names)
        }
    }
}
