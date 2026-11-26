// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint and work-tree paths for tests that read local artifacts.
//!
//! Paths using `${FLUX_MODELS_ROOT}` or `${FLUX_ATLAS_ROOT}` expand from the
//! corresponding environment variables so tests do not hardcode machine-local
//! paths. An unset variable fails loudly rather than silently skipping.

const PLACEHOLDERS: [(&str, &str); 2] = [
    ("${FLUX_MODELS_ROOT}", "FLUX_MODELS_ROOT"),
    ("${FLUX_ATLAS_ROOT}", "FLUX_ATLAS_ROOT"),
];

/// `path` with each placeholder replaced by its environment variable.
pub(crate) fn resolve(path: &str) -> String {
    resolve_with(path, |name| std::env::var(name).ok())
}

fn resolve_with(path: &str, var: impl Fn(&str) -> Option<String>) -> String {
    let mut out = path.to_string();
    for (placeholder, name) in PLACEHOLDERS {
        if out.contains(placeholder) {
            let root = var(name).filter(|v| !v.is_empty()).unwrap_or_else(|| {
                panic!(
                    "{path} needs {name} (the directory {placeholder} stands for); \
                     a missing checkpoint is a failure, not a skip"
                )
            });
            out = out.replace(placeholder, root.trim_end_matches('/'));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::resolve_with;

    #[test]
    fn placeholders_resolve_from_their_variables_and_plain_paths_pass_through() {
        let var = |name: &str| match name {
            "FLUX_MODELS_ROOT" => Some("/m/".to_string()),
            "FLUX_ATLAS_ROOT" => Some("/a".to_string()),
            _ => None,
        };
        assert_eq!(
            resolve_with("${FLUX_MODELS_ROOT}/X/tokenizer.json", var),
            "/m/X/tokenizer.json"
        );
        assert_eq!(resolve_with("${FLUX_ATLAS_ROOT}/parity", var), "/a/parity");
        assert_eq!(resolve_with("/abs/path", |_| None), "/abs/path");
    }

    #[test]
    #[should_panic(expected = "needs FLUX_MODELS_ROOT")]
    fn an_unset_root_fails_and_names_the_variable() {
        resolve_with("${FLUX_MODELS_ROOT}/X", |_| None);
    }
}
