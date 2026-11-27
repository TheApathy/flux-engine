// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeSet;

use super::{
    CTX_WINDOW_ENV, DRAFT_CAP_ENV, FlatDflashContract, QUANT_ENV, live_environment_value,
    normalize_environment_value, rules::RELEASE_ENVIRONMENT_RULES,
    validate_flat_dflash_environment,
};

fn dflash_contract() -> FlatDflashContract<'static> {
    FlatDflashContract {
        enabled: true,
        method: "flat",
        gamma: Some(7),
        window_size: 4096,
        quantization: "nvfp4",
    }
}

fn validate_one(name: &str, value: &str) -> Result<(), String> {
    validate_flat_dflash_environment(dflash_contract(), |candidate| {
        (candidate == name).then(|| value.to_string())
    })
}

#[test]
fn every_registered_environment_control_fails_closed_on_an_unlisted_value() {
    let registered: BTreeSet<&str> = RELEASE_ENVIRONMENT_RULES
        .iter()
        .map(|rule| rule.name)
        .collect();
    assert_eq!(
        registered.len(),
        RELEASE_ENVIRONMENT_RULES.len(),
        "the registry must not contain duplicate names"
    );

    for name in registered {
        let value = "__stale_operator_override__";
        let error = validate_one(name, value).expect_err("unlisted control value was admitted");
        assert!(error.contains(name), "{name}={value}: {error}");
    }
}

#[test]
fn official_release_and_preexisting_neutral_values_remain_admissible() {
    for (name, value) in [
        ("ATLAS_DFLASH_METHOD", "flat"),
        ("ATLAS_DFLASH_ASYNC", "0"),
        ("ATLAS_DFLASH_NOISE_ONLY", "1"),
        ("ATLAS_DFLASH_SPEC_CYCLE_V2", "1"),
        ("ATLAS_DFLASH_SWA", "1"),
        ("ATLAS_DFLASH_HONOR_IS_CAUSAL", "1"),
        ("ATLAS_DFLASH_DRAFT_SPLITK", "8"),
        ("ATLAS_DFLASH_PREFILL_PIPE", "1"),
        ("ATLAS_DFLASH_CAPTURE_STRIDED", "1"),
        ("ATLAS_DFLASH_LM_HEAD_NVFP4", "1"),
        ("ATLAS_DFLASH_LM_HEAD_FP8", "0"),
        ("ATLAS_DFLASH_FFN_KGAMMA", "1"),
        ("ATLAS_DFLASH_ATTN_KGAMMA", "1"),
        ("ATLAS_DFLASH_FUSED", "0"),
        ("ATLAS_DFLASH_BRANCH", "0"),
        ("ATLAS_DFLASH_CATERPILLAR", "0"),
        ("ATLAS_DFLASH_FREE_SLOTS", "0"),
        ("ATLAS_DFLASH_FREE_SLOTS_TAIL", "4"),
        ("ATLAS_DFLASH_PORTFOLIO", "0"),
        ("ATLAS_DFLASH_RETRIEVAL", "0"),
        ("ATLAS_DFLASH_SAM", "0"),
        ("ATLAS_DFLASH_KEY_RETRIEVAL", "0"),
        ("ATLAS_DFLASH_RETR_WIDE", "31"),
        ("ATLAS_DFLASH_KERNEL_PROFILE", "0"),
        ("ATLAS_DFLASH_TREE_COMMIT", "0"),
        ("ATLAS_DFLASH_ECHO", "0"),
        ("ATLAS_DISABLE_TREE_WY", "1"),
        ("ATLAS_DDTREE_MAX_NODES", "8"),
        ("ATLAS_DDTREE_UNCAP", "0"),
        ("ATLAS_DDTREE_TREE_AWARE_VERIFY", "0"),
        ("ATLAS_DDTREE_TREE_TOKENS_VERIFY", "0"),
        ("ATLAS_DDTREE_TREE_CONV_EXACT", "0"),
        ("ATLAS_TREE_AWARE_ATTN", "0"),
        (DRAFT_CAP_ENV, "7"),
        (CTX_WINDOW_ENV, "4096"),
        (QUANT_ENV, "nvfp4"),
    ] {
        validate_one(name, value).unwrap_or_else(|error| panic!("{name}={value}: {error}"));
    }
}

#[test]
fn release_nvfp4_head_rejects_the_unused_fp8_mirror() {
    let error = validate_one("ATLAS_DFLASH_LM_HEAD_FP8", "1")
        .expect_err("release-flat must not construct both LM-head formats");
    assert!(error.contains("ATLAS_DFLASH_LM_HEAD_FP8"), "{error}");
    assert!(error.contains("must be one of: 0"), "{error}");
}

#[test]
fn cli_authoritative_environment_values_must_match_exactly() {
    for (name, value) in [
        (DRAFT_CAP_ENV, "6"),
        (CTX_WINDOW_ENV, "8192"),
        (QUANT_ENV, "bf16"),
    ] {
        let error = validate_one(name, value).expect_err("mismatched CLI duplicate was admitted");
        assert!(error.contains(name), "{name}={value}: {error}");
        assert!(error.contains("authoritative") || error.contains("explicit"));
    }
}

#[test]
fn tree_capacity_must_equal_the_explicit_flat_verify_width() {
    let no_explicit_gamma = FlatDflashContract {
        enabled: true,
        method: "flat",
        gamma: None,
        window_size: 4096,
        quantization: "nvfp4",
    };
    let error = validate_flat_dflash_environment(no_explicit_gamma, |name| {
        (name == "ATLAS_DDTREE_MAX_NODES").then(|| "8".to_string())
    })
    .expect_err("checkpoint-derived gamma cannot prove an environment capacity is flat-only");
    assert!(error.contains("ATLAS_DDTREE_MAX_NODES"), "{error}");

    let malformed = validate_one("ATLAS_DDTREE_MAX_NODES", "eight")
        .expect_err("malformed capacity must not silently use the default");
    assert!(malformed.contains("ATLAS_DDTREE_MAX_NODES"), "{malformed}");
}

#[test]
fn tree_environment_is_irrelevant_when_dflash_is_off() {
    let plain = FlatDflashContract {
        enabled: false,
        method: "ddtree",
        gamma: None,
        window_size: 0,
        quantization: "unsafe",
    };
    validate_flat_dflash_environment(plain, |_| Some("unsafe".to_string()))
        .expect("DFlash-only environment must not block a plain target");
}

#[test]
fn a_non_unicode_operator_value_is_present_and_therefore_rejected() {
    let _live_reader: fn(&str) -> Option<String> = live_environment_value;
    let value = normalize_environment_value(Err(std::env::VarError::NotUnicode("bad".into())));
    let error = validate_flat_dflash_environment(dflash_contract(), |name| {
        (name == "ATLAS_DFLASH_BRANCH")
            .then(|| value.clone())
            .flatten()
    })
    .expect_err("a non-Unicode operator value must not masquerade as absent");
    assert!(error.contains("ATLAS_DFLASH_BRANCH"), "{error}");
}

fn validate_pairs(pairs: &[(&str, &str)]) -> Result<(), String> {
    validate_flat_dflash_environment(dflash_contract(), |candidate| {
        pairs
            .iter()
            .find(|(name, _)| *name == candidate)
            .map(|(_, value)| value.to_string())
    })
}

#[test]
fn lookup_drafting_is_refused_without_the_reviewed_selector() {
    for name in ["ATLAS_DFLASH_SAM", "ATLAS_DFLASH_RETRIEVAL"] {
        let error = validate_one(name, "1").expect_err("bare lookup control was admitted");
        assert!(error.contains(name), "{error}");
        assert!(error.contains("FLUX_DFLASH_LOOKUP"), "{error}");
    }
}

#[test]
fn the_selector_admits_exactly_its_engine_control() {
    validate_pairs(&[("ATLAS_DFLASH_LOOKUP", "sam"), ("ATLAS_DFLASH_SAM", "1")])
        .expect("sam selector with its control");
    validate_pairs(&[
        ("ATLAS_DFLASH_LOOKUP", "retrieval"),
        ("ATLAS_DFLASH_RETRIEVAL", "1"),
        ("ATLAS_DFLASH_SAM", "0"),
    ])
    .expect("retrieval selector with its control");
    validate_pairs(&[("ATLAS_DFLASH_LOOKUP", "off"), ("ATLAS_DFLASH_SAM", "0")])
        .expect("explicit off");

    let missing = validate_pairs(&[("ATLAS_DFLASH_LOOKUP", "sam")])
        .expect_err("a selector whose control is absent would not engage");
    assert!(missing.contains("ATLAS_DFLASH_SAM"), "{missing}");
    let both = validate_pairs(&[
        ("ATLAS_DFLASH_LOOKUP", "sam"),
        ("ATLAS_DFLASH_SAM", "1"),
        ("ATLAS_DFLASH_RETRIEVAL", "1"),
    ])
    .expect_err("one selector engages one matcher");
    assert!(both.contains("ATLAS_DFLASH_RETRIEVAL"), "{both}");
    let malformed = validate_pairs(&[("ATLAS_DFLASH_LOOKUP", "yes"), ("ATLAS_DFLASH_SAM", "1")])
        .expect_err("a malformed selector must not admit SAM");
    assert!(malformed.contains("ATLAS_DFLASH_LOOKUP"), "{malformed}");
    assert!(malformed.contains("ATLAS_DFLASH_SAM"), "{malformed}");
}

#[test]
fn the_selector_keeps_every_other_rule_strict() {
    for (name, value) in [
        ("ATLAS_DFLASH_SAM_ADAPTIVE", "0"),
        ("ATLAS_DFLASH_SAM_MIN_ACCEPT", "1"),
        ("ATLAS_DFLASH_SAM_MISFIRE_LIMIT", "9"),
        ("ATLAS_DFLASH_SAM_COOLDOWN", "0"),
        ("ATLAS_DFLASH_SAM_ASYNC", "1"),
        ("ATLAS_DFLASH_SAM_CORROBORATE", "1"),
        ("ATLAS_DFLASH_PORTFOLIO", "1"),
        ("ATLAS_DFLASH_KEY_RETRIEVAL", "1"),
        ("ATLAS_DFLASH_PLD", "1"),
        ("ATLAS_DFLASH_BRANCH", "1"),
        ("ATLAS_DFLASH_RETR_WIDE", "15"),
        ("ATLAS_RETRIEVAL_HYBRID_MIN", "4"),
        ("ATLAS_RETRIEVAL_LMAX", "32"),
        ("ATLAS_RETRIEVAL_LMIN", "2"),
    ] {
        let error = validate_pairs(&[
            ("ATLAS_DFLASH_LOOKUP", "sam"),
            ("ATLAS_DFLASH_SAM", "1"),
            (name, value),
        ])
        .expect_err("the selector must not relax an unrelated control");
        assert!(error.contains(name), "{name}={value}: {error}");
    }
}

#[test]
fn the_selector_is_irrelevant_when_dflash_is_off() {
    let plain = FlatDflashContract {
        enabled: false,
        method: "flat",
        gamma: None,
        window_size: 0,
        quantization: "nvfp4",
    };
    validate_flat_dflash_environment(plain, |name| {
        (name == "ATLAS_DFLASH_LOOKUP").then(|| "bogus".to_string())
    })
    .expect("a non-DFlash target (Flash-Next) never reads the selector");
}

#[test]
fn v12_opt_in_dflash_controls_admit_zero_or_one_and_fail_closed_otherwise() {
    for name in [
        "ATLAS_DFLASH_ASYNC",
        "ATLAS_DFLASH_DENSE_PCUM",
        "ATLAS_DFLASH_EARLY_STAGE1",
        "ATLAS_DFLASH_ECHO",
        "ATLAS_DFLASH_ECHO_GAMMA7",
        "ATLAS_DFLASH_NGRAM_BONUS",
        "ATLAS_DFLASH_SPEC_CYCLE_V2",
    ] {
        validate_one(name, "0").unwrap_or_else(|e| panic!("{name}=0: {e}"));
        validate_one(name, "1").unwrap_or_else(|e| panic!("{name}=1: {e}"));
        let err = validate_one(name, "2").expect_err("non-binary value must fail closed");
        assert!(err.contains(name), "{name}=2: {err}");
    }
}
