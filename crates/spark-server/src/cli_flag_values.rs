// SPDX-License-Identifier: AGPL-3.0-only

//! The closed value sets of the enumerated `spark serve` string flags.
//!
//! Centralized here so the dashboard's option picker and CLI value validation
//! share a single source of truth instead of drifting out of sync.
//!
//! `--kv-cache-dtype` is deliberately NOT a literal list here. Its authority is
//! `spark_runtime::kv_cache::KvCacheDtype::ALL`, derived from the enum itself,
//! so a variant added there appears in the picker automatically instead of
//! silently missing from a hand-maintained copy.
//!
//! Deliberately NOT wired into clap as `PossibleValuesParser`: that would be a
//! second enforcement point over the same data, and two enforcement points is
//! how the two come to disagree. clap stays the authority on the flag SURFACE
//! (names, help, defaults, arity); this module is the authority on the
//! enumerated VALUES.

/// The closed value set for `long`, or `None` when the flag is free text.
///
/// `long` is the flag's long name WITHOUT the leading dashes, as clap reports
/// it — `kv-cache-dtype`, not `--kv-cache-dtype`.
pub(crate) fn options_for_flag(long: &str) -> Option<Vec<String>> {
    match long {
        // Dispatched in `main_modules::serve` with a bail on anything else —
        // AFTER the weight load. Listing it here moves the typo diagnosis to
        // the Library form, where it costs milliseconds instead of minutes.
        "scheduling-policy" => Some(vec!["fifo".to_string(), "slai".to_string()]),
        "kv-cache-dtype" => Some(
            spark_runtime::kv_cache::KvCacheDtype::ALL
                .iter()
                .map(|d| d.name().to_string())
                .collect(),
        ),
        // DDTree still has runtime error paths that fall back to a flat
        // payload.  A release profile must never advertise that as an active
        // method, so the only selectable value is the fully wired flat path.
        "dflash-method" => Some(vec!["flat".to_string()]),
        "dflash-draft-sample-method" => {
            Some(vec!["greedy".to_string(), "probabilistic".to_string()])
        }
        "role" => Some(
            atlas_core::config::DisaggRole::ALL
                .iter()
                .map(|r| r.as_str().to_string())
                .collect(),
        ),
        // Everything else is free text or a number. Returning `None` rather
        // than an empty vec keeps "no closed set" distinguishable from "a
        // closed set that happens to be empty" — the picker renders the first
        // as a text field and would render the second as an empty list.
        _ => None,
    }
}

#[cfg(test)]
#[path = "cli_flag_values_tests.rs"]
mod tests;
