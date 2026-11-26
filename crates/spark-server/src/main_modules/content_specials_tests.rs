// SPDX-License-Identifier: AGPL-3.0-only

use super::content_specials::{ContentSpecialTokens, resolve};

fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |name| {
        pairs
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.to_string())
    }
}

#[test]
fn default_is_literal() {
    assert_eq!(
        resolve(false, env(&[])).unwrap(),
        ContentSpecialTokens::Literal
    );
}

#[test]
fn explicit_modes_parse() {
    let structural = [("FLUX_SPECIAL_TOKENS_IN_CONTENT", "structural")];
    assert_eq!(
        resolve(false, env(&structural)).unwrap(),
        ContentSpecialTokens::Structural
    );
    let literal = [("FLUX_SPECIAL_TOKENS_IN_CONTENT", "literal")];
    assert_eq!(
        resolve(false, env(&literal)).unwrap(),
        ContentSpecialTokens::Literal
    );
    let atlas = [("ATLAS_SPECIAL_TOKENS_IN_CONTENT", "structural")];
    assert_eq!(
        resolve(false, env(&atlas)).unwrap(),
        ContentSpecialTokens::Structural
    );
}

#[test]
fn legacy_flag_and_env_still_opt_out() {
    assert_eq!(
        resolve(true, env(&[])).unwrap(),
        ContentSpecialTokens::Structural
    );
    let legacy = [("FLUX_ALLOW_SPECIAL_TOKENS_IN_CONTENT", "1")];
    assert_eq!(
        resolve(false, env(&legacy)).unwrap(),
        ContentSpecialTokens::Structural
    );
}

#[test]
fn unknown_value_fails_fast() {
    let bad = [("FLUX_SPECIAL_TOKENS_IN_CONTENT", "raw")];
    let err = resolve(false, env(&bad)).unwrap_err().to_string();
    assert!(err.contains("literal|structural"), "{err}");
}

#[test]
fn contradictory_settings_fail_fast() {
    let lit = [("FLUX_SPECIAL_TOKENS_IN_CONTENT", "literal")];
    assert!(resolve(true, env(&lit)).is_err());
    let split = [
        ("FLUX_SPECIAL_TOKENS_IN_CONTENT", "literal"),
        ("ATLAS_SPECIAL_TOKENS_IN_CONTENT", "structural"),
    ];
    assert!(resolve(false, env(&split)).is_err());
}

#[test]
fn receipt_names_the_mode() {
    assert_eq!(
        ContentSpecialTokens::Literal.receipt(),
        "CONTENT_SPECIAL_TOKENS mode=literal"
    );
    assert_eq!(
        ContentSpecialTokens::Structural.receipt(),
        "CONTENT_SPECIAL_TOKENS mode=structural"
    );
    assert!(ContentSpecialTokens::Structural.allows_structural());
    assert!(!ContentSpecialTokens::Literal.allows_structural());
}
