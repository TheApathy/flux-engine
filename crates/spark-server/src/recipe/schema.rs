// SPDX-License-Identifier: AGPL-3.0-only

//! Recipe `defaults:` keys → `spark serve` flags.
//!
//! **clap stays the single source of truth for the flag surface.** `ServeArgs`
//! is not given a `Serialize` derive: `atlas-recipes` is a separate repo that
//! cannot be renamed atomically with this one, so making the CLI a public
//! serialization format would turn every future flag rename into a compat
//! break. Instead a recipe is converted to argv and handed back through
//! `ServeArgs::try_parse_from` — clap validates it exactly as if a person had
//! typed it, and a key this table gets wrong fails loudly at parse.
//!
//! Most keys are the field name with underscores swapped for dashes. The
//! exceptions below are real and were verified against `cli.rs`; each
//! one silently serves the wrong thing if it is dropped.

/// Keys whose recipe spelling differs from the flag.
///
/// Verified against `cli.rs` (`ServeArgs`): neither `max_model_len`
/// nor `tensor_parallel` exists as a field, and the listen address is `--bind`.
pub(crate) const RENAMES: &[(&str, &str)] = &[
    // vLLM's spelling, kept in the recipes for cross-runtime familiarity.
    ("max_model_len", "max-seq-len"),
    ("tensor_parallel", "tp-size"),
    ("host", "bind"),
];

/// `defaults:` keys that are not `spark serve` flags at all.
///
/// `port` IS a flag and is not listed here.
const NOT_FLAGS: &[&str] = &[];

/// The flag for a recipe key, or `None` if the key is not a flag.
pub fn flag_for(key: &str) -> Option<String> {
    if NOT_FLAGS.contains(&key) {
        return None;
    }
    if let Some((_, flag)) = RENAMES.iter().find(|(k, _)| *k == key) {
        return Some((*flag).to_string());
    }
    Some(key.replace('_', "-"))
}

/// Render one `key: value` pair as argv.
///
/// For a **presence-only** flag (clap `SetTrue`, e.g. `--speculative`),
/// `true` becomes the bare `--flag` and `false` is **omitted rather than
/// negated**: `--speculative false` is not accepted by a `SetTrue` flag, and
/// emitting it would fail the parse for a recipe that is merely restating a
/// default.
///
/// Every other boolean takes its value on the command line and KEEPS it —
/// for `Option<bool>` levers on `ServeArgs` (such as `--high-speed-swap-graph`),
/// explicit `false` is a different config from absent (absent falls back to the
/// derived/default behavior; explicit `false` disables it). The presence-only
/// set is read directly from clap, so every new `Option<bool>` lever keeps its
/// explicit value automatically.
pub fn argv_for(key: &str, value: &str) -> Option<Vec<String>> {
    let flag = flag_for(key)?;
    let dashed = format!("--{flag}");
    match value {
        "true" if presence_only(&flag) => Some(vec![dashed]),
        "false" if presence_only(&flag) => None,
        other => Some(vec![dashed, other.to_string()]),
    }
}

/// Whether `flag` is a presence-only boolean (clap action `SetTrue`).
///
/// Asked of clap once per process rather than written down again: the set is
/// derived from `ServeArgs` itself, so it cannot drift from it. A flag not in
/// `ServeArgs` at all is not presence-only; its value passes through and clap
/// rejects the unknown flag by name at parse, which is the SSOT typo shield.
fn presence_only(flag: &str) -> bool {
    use std::sync::OnceLock;
    static SET_TRUE: OnceLock<std::collections::HashSet<String>> = OnceLock::new();
    SET_TRUE
        .get_or_init(|| {
            use clap::CommandFactory as _;
            crate::cli::ServeArgs::command()
                .get_arguments()
                .filter(|a| matches!(a.get_action(), clap::ArgAction::SetTrue))
                .filter_map(|a| a.get_long().map(str::to_string))
                .collect()
        })
        .contains(flag)
}

#[cfg(test)]
#[path = "schema_tests.rs"]
mod tests;
