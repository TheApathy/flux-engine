// SPDX-License-Identifier: AGPL-3.0-only

//! Startup gate for F1 reflection suppression (`ATLAS_REFLECTION_SUPPRESS`).
//!
//! F1 subtracts [`REFLECTION_PENALTY`] from single-token reflection words
//! ("wait", "however", ...) on every thinking token. It is Flux-invented,
//! not part of any checkpoint's sampling recipe, and penalising RL-trained
//! self-verification pivots degrades reasoning/math accuracy and MTP draft
//! acceptance. `off` (default, unset or empty) resolves an empty list, which
//! makes the apply site in `decode_logits_seq` a no-op and matches public
//! inference engines (`vLLM`, `SGLang`, `transformers`); `on` explicitly
//! enables the legacy id list. Any other value refuses startup.
//!
//! Known inconsistency: the batched qwen4 verify path always passes an
//! empty list, so speculative verify rows are never suppressed. The
//! receipt names that path `verify_path=unsuppressed`.

use anyhow::{Result, bail};

pub(crate) const REFLECTION_ENV: &str = "ATLAS_REFLECTION_SUPPRESS";

/// Logit penalty applied in `scheduler::decode_logits_seq` (receipt only).
pub(crate) const REFLECTION_PENALTY: f32 = -10.0;

pub(crate) const REFLECTION_WORDS: [&str; 8] = [
    "wait", "Wait", "however", "However", "actually", "Actually", "hmm", "Hmm",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectionMode {
    On,
    Off,
}

impl ReflectionMode {
    fn as_str(self) -> &'static str {
        match self {
            ReflectionMode::On => "on",
            ReflectionMode::Off => "off",
        }
    }
}

pub(crate) fn parse_reflection_mode(raw: Option<&str>) -> Result<ReflectionMode> {
    match raw.map(str::trim) {
        None | Some("") | Some("off") => Ok(ReflectionMode::Off),
        Some("on") => Ok(ReflectionMode::On),
        Some(other) => bail!("{REFLECTION_ENV}={other:?}: expected on|off"),
    }
}

/// Resolve the suppressed ids: single-token encodings of
/// [`REFLECTION_WORDS`] when on, empty when off.
pub(crate) fn resolve_reflection_ids<F>(mode: ReflectionMode, encode: F) -> Vec<u32>
where
    F: Fn(&str) -> Option<Vec<u32>>,
{
    match mode {
        ReflectionMode::Off => Vec::new(),
        ReflectionMode::On => REFLECTION_WORDS
            .iter()
            .filter_map(|word| encode(word))
            .filter(|ids| ids.len() == 1)
            .map(|ids| ids[0])
            .collect(),
    }
}

pub(crate) fn reflection_receipt(mode: ReflectionMode, ids: usize) -> String {
    format!(
        "REFLECTION_SUPPRESS mode={} ids={ids} penalty={REFLECTION_PENALTY} verify_path=unsuppressed",
        mode.as_str()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_encode(word: &str) -> Option<Vec<u32>> {
        match word {
            // Multi-token words are dropped, as with the real tokenizer.
            "hmm" => Some(vec![1, 2]),
            "Hmm" => None,
            w => Some(vec![w.len() as u32 * 100 + w.as_bytes()[0] as u32]),
        }
    }

    #[test]
    fn default_is_off_and_explicit_on_preserves_list() {
        assert_eq!(parse_reflection_mode(None).unwrap(), ReflectionMode::Off);
        assert_eq!(
            parse_reflection_mode(Some("")).unwrap(),
            ReflectionMode::Off
        );
        assert_eq!(
            parse_reflection_mode(Some(" on ")).unwrap(),
            ReflectionMode::On
        );
        let legacy: Vec<u32> = REFLECTION_WORDS
            .iter()
            .filter_map(|w| fake_encode(w))
            .filter(|ids| ids.len() == 1)
            .map(|ids| ids[0])
            .collect();
        let ids = resolve_reflection_ids(ReflectionMode::On, fake_encode);
        assert_eq!(ids, legacy);
        assert_eq!(ids.len(), 6);
    }

    #[test]
    fn off_resolves_empty() {
        assert_eq!(
            parse_reflection_mode(Some("off")).unwrap(),
            ReflectionMode::Off
        );
        assert!(resolve_reflection_ids(ReflectionMode::Off, fake_encode).is_empty());
    }

    #[test]
    fn receipt_line() {
        assert_eq!(
            reflection_receipt(ReflectionMode::On, 7),
            "REFLECTION_SUPPRESS mode=on ids=7 penalty=-10 verify_path=unsuppressed"
        );
        assert_eq!(
            reflection_receipt(ReflectionMode::Off, 0),
            "REFLECTION_SUPPRESS mode=off ids=0 penalty=-10 verify_path=unsuppressed"
        );
    }

    #[test]
    fn unknown_value_refuses() {
        for bad in ["0", "1", "false", "true", "OFF", "maybe"] {
            let err = parse_reflection_mode(Some(bad)).unwrap_err().to_string();
            assert!(err.contains(REFLECTION_ENV), "{err}");
        }
    }
}
