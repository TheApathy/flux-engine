// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../build_target_selector.rs"]
mod build_target_selector;

use build_target_selector::{ModelSelector, parse_model_selector, resolve_model_names};

fn available() -> Vec<String> {
    [
        "glm5.3-flash",
        "qwen3.8-27b",
        "qwen3.8-flash-next",
        "unselected-model",
    ]
    .map(str::to_owned)
    .to_vec()
}

#[test]
fn singular_exact_name_is_preserved() {
    assert_eq!(
        parse_model_selector("qwen3.8-27b"),
        Ok(ModelSelector::Exact(vec!["qwen3.8-27b".to_owned()]))
    );
}

#[test]
fn wildcard_selects_every_registered_model_in_deterministic_order() {
    let mut unordered = available();
    unordered.reverse();
    assert_eq!(resolve_model_names("*", unordered), Ok(available()));
}

#[test]
fn exact_list_selects_only_named_models_in_deterministic_order() {
    assert_eq!(
        resolve_model_names("qwen3.8-flash-next,qwen3.8-27b", available(),),
        Ok(vec![
            "qwen3.8-27b".to_owned(),
            "qwen3.8-flash-next".to_owned(),
        ])
    );
}

#[test]
fn empty_entries_are_rejected() {
    for spec in ["", ",qwen3.8-27b", "qwen3.8-27b,", "a,,b"] {
        assert!(
            parse_model_selector(spec)
                .unwrap_err()
                .contains("empty entry"),
            "unexpected result for {spec:?}"
        );
    }
}

#[test]
fn duplicates_are_rejected() {
    assert_eq!(
        parse_model_selector("qwen3.8-27b,qwen3.8-27b"),
        Err("duplicate model name \"qwen3.8-27b\"".to_owned())
    );
}

#[test]
fn wildcard_cannot_be_mixed_with_exact_names() {
    for spec in ["*,qwen3.8-27b", "qwen3.8-27b,*"] {
        assert!(
            parse_model_selector(spec)
                .unwrap_err()
                .contains("cannot be mixed"),
            "unexpected result for {spec:?}"
        );
    }
}

#[test]
fn unknown_and_inexact_names_are_rejected() {
    assert!(resolve_model_names("not-registered", available()).is_err());
    assert!(resolve_model_names(" qwen3.8-27b", available()).is_err());
}
