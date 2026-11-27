// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use super::{
    CTX_WINDOW_ENV, DRAFT_CAP_ENV, QUANT_ENV, lookup::LOOKUP_ENV, rules::RELEASE_ENVIRONMENT_RULES,
};

const PREFIX: &str = "ATLAS_DFLASH_";
const CENSUS_COUNT: usize = 139;

fn collect_tokens(text: &str, found: &mut BTreeSet<String>) {
    let mut remaining = text;
    while let Some(start) = remaining.find(PREFIX) {
        remaining = &remaining[start..];
        let end = remaining
            .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
            .unwrap_or(remaining.len());
        let token = &remaining[..end];
        if !token.ends_with('_') {
            found.insert(token.to_string());
        }
        remaining = &remaining[end..];
    }
}

fn collect_rust_sources(root: &Path, found: &mut BTreeSet<String>) {
    for entry in fs::read_dir(root).unwrap_or_else(|error| panic!("{}: {error}", root.display())) {
        let path = entry.expect("read source directory entry").path();
        if path.is_dir() {
            collect_rust_sources(&path, found);
            continue;
        }
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if name.starts_with("cli_dflash_environment") {
            continue;
        }
        let text =
            fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        collect_tokens(&text, found);
    }
}

#[test]
fn every_dflash_source_control_has_one_fail_closed_classification() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut source = BTreeSet::new();
    collect_rust_sources(&manifest.join("src"), &mut source);
    collect_rust_sources(&manifest.join("../spark-model/src"), &mut source);
    let recipe = manifest.join("src/recipe/builtin/qwen3.8-27b-dflash2-safe-local.yaml");
    collect_tokens(
        &fs::read_to_string(&recipe)
            .unwrap_or_else(|error| panic!("{}: {error}", recipe.display())),
        &mut source,
    );

    assert_eq!(
        source.len(),
        CENSUS_COUNT,
        "the reviewed DFlash source census changed; classify each added or removed control"
    );

    let registered: BTreeSet<String> = RELEASE_ENVIRONMENT_RULES
        .iter()
        .map(|rule| rule.name)
        .filter(|name| name.starts_with(PREFIX))
        .chain([DRAFT_CAP_ENV, CTX_WINDOW_ENV, QUANT_ENV, LOOKUP_ENV])
        .map(str::to_string)
        .collect();
    assert_eq!(
        source, registered,
        "every ATLAS_DFLASH_* source/recipe control must be statically allowlisted, CLI-authoritative, or the lookup selector"
    );
}
