// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    DflashLookup, LOOKUP_ENV, RETRIEVAL_ENV, SAM_ENV, engaged_receipt, implied_controls,
    lookup_mode, validate_owned_controls,
};

fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl FnMut(&str) -> Option<String> + 'a {
    move |name| {
        pairs
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value.to_string())
    }
}

#[test]
fn selector_parses_exactly_three_modes_and_defaults_off() {
    assert_eq!(lookup_mode(env(&[])), Ok(DflashLookup::Off));
    for (value, mode) in [
        ("off", DflashLookup::Off),
        ("sam", DflashLookup::Sam),
        ("retrieval", DflashLookup::Retrieval),
    ] {
        assert_eq!(lookup_mode(env(&[(LOOKUP_ENV, value)])), Ok(mode));
        assert_eq!(mode.as_str(), value);
    }
    for bad in ["", "SAM", "1", "on", "sam ", "pld"] {
        let error = lookup_mode(env(&[(LOOKUP_ENV, bad)])).expect_err(bad);
        assert!(error.contains(LOOKUP_ENV), "{error}");
    }
}

#[test]
fn each_mode_maps_to_exactly_one_engine_control() {
    assert!(implied_controls(env(&[])).is_empty());
    assert!(implied_controls(env(&[(LOOKUP_ENV, "off")])).is_empty());
    assert_eq!(
        implied_controls(env(&[(LOOKUP_ENV, "sam")])),
        vec![(SAM_ENV, "1")]
    );
    assert_eq!(
        implied_controls(env(&[(LOOKUP_ENV, "retrieval")])),
        vec![(RETRIEVAL_ENV, "1")]
    );
    assert!(implied_controls(env(&[(LOOKUP_ENV, "bogus")])).is_empty());
}

#[test]
fn mapping_never_overrides_an_explicit_operator_value() {
    // The recipe exports ATLAS_DFLASH_SAM=0; a conflicting explicit value is
    // left in place so the contract refuses it rather than silently winning.
    let pairs = [(LOOKUP_ENV, "sam"), (SAM_ENV, "0")];
    assert!(implied_controls(env(&pairs)).is_empty());
    let problems = validate_owned_controls(DflashLookup::Sam, env(&pairs));
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains(SAM_ENV) && problems[0].contains("must be 1"));
}

#[test]
fn owned_controls_must_agree_with_the_selected_mode() {
    let admitted: &[(DflashLookup, &[(&str, &str)])] = &[
        (DflashLookup::Off, &[]),
        (DflashLookup::Off, &[(SAM_ENV, "0"), (RETRIEVAL_ENV, "0")]),
        (DflashLookup::Sam, &[(SAM_ENV, "1")]),
        (DflashLookup::Sam, &[(SAM_ENV, "1"), (RETRIEVAL_ENV, "0")]),
        (DflashLookup::Retrieval, &[(RETRIEVAL_ENV, "1")]),
        (
            DflashLookup::Retrieval,
            &[(RETRIEVAL_ENV, "1"), (SAM_ENV, "0")],
        ),
    ];
    for (mode, pairs) in admitted {
        let problems = validate_owned_controls(*mode, env(pairs));
        assert!(problems.is_empty(), "{mode:?} {pairs:?}: {problems:?}");
    }
    let refused: &[(DflashLookup, &[(&str, &str)], &str)] = &[
        (DflashLookup::Off, &[(SAM_ENV, "1")], SAM_ENV),
        (DflashLookup::Off, &[(RETRIEVAL_ENV, "1")], RETRIEVAL_ENV),
        (DflashLookup::Sam, &[], SAM_ENV),
        (DflashLookup::Sam, &[(SAM_ENV, "true")], SAM_ENV),
        (
            DflashLookup::Sam,
            &[(SAM_ENV, "1"), (RETRIEVAL_ENV, "1")],
            RETRIEVAL_ENV,
        ),
        (DflashLookup::Retrieval, &[], RETRIEVAL_ENV),
        (
            DflashLookup::Retrieval,
            &[(RETRIEVAL_ENV, "1"), (SAM_ENV, "1")],
            SAM_ENV,
        ),
    ];
    for (mode, pairs, culprit) in refused {
        let problems = validate_owned_controls(*mode, env(pairs));
        assert_eq!(problems.len(), 1, "{mode:?} {pairs:?}: {problems:?}");
        assert!(problems[0].contains(culprit), "{}", problems[0]);
        assert!(problems[0].contains(mode.as_str()), "{}", problems[0]);
    }
}

#[test]
fn sealed_overlay_rewrites_only_owned_keys() {
    assert_eq!(
        DflashLookup::Off.sealed_value(SAM_ENV, "0").as_deref(),
        Some("0")
    );
    assert_eq!(
        DflashLookup::Sam.sealed_value(SAM_ENV, "0").as_deref(),
        Some("1")
    );
    assert_eq!(
        DflashLookup::Retrieval
            .sealed_value(SAM_ENV, "0")
            .as_deref(),
        Some("0")
    );
    assert_eq!(
        DflashLookup::Retrieval
            .sealed_value(RETRIEVAL_ENV, "0")
            .as_deref(),
        Some("1")
    );
    assert_eq!(
        DflashLookup::Sam.sealed_value("ATLAS_DFLASH_RETR_WIDE", "31"),
        None
    );
}

#[test]
fn engaged_receipt_names_the_mode_and_is_silent_when_off() {
    assert_eq!(engaged_receipt(DflashLookup::Off), None);
    let sam = engaged_receipt(DflashLookup::Sam).expect("sam receipt");
    assert!(sam.starts_with("DFLASH_LOOKUP_ENGAGED mode=sam "), "{sam}");
    assert!(sam.contains("ATLAS_DFLASH_SAM=1"), "{sam}");
    let retrieval = engaged_receipt(DflashLookup::Retrieval).expect("retrieval receipt");
    assert!(retrieval.contains("mode=retrieval"), "{retrieval}");
}
