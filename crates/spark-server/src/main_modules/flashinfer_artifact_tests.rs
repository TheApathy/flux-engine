// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use super::*;

fn profile(values: &[(&str, &str)]) -> ModelProfile {
    ModelProfile {
        recipe_id: "test/profile".into(),
        model_type: "qwen3_5".into(),
        resident_gb: 0.0,
        dspark_extra_gb: 0.0,
        headroom_gb: 0.0,
        run_alone: false,
        env: values
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect::<BTreeMap<_, _>>(),
        contract: None,
    }
}

fn lookup(values: &[(&str, OsString)], name: &str) -> Option<OsString> {
    values
        .iter()
        .find(|(candidate, _)| *candidate == name)
        .map(|(_, value)| value.clone())
}

#[test]
fn profile_only_gate_and_library_are_verified() {
    let expected = Path::new("/qualified/libatlas-fi.so");
    let profile = profile(&[
        (FLASHINFER_GATES[0], "1"),
        (FLASHINFER_LIBRARY, expected.to_str().unwrap()),
    ]);
    let mut seen = None;
    preflight_with(
        Some(&profile),
        |_| None,
        |path| {
            seen = Some(path.to_path_buf());
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(seen.as_deref(), Some(expected));
}

#[test]
fn operator_values_override_profile_defaults() {
    let profile = profile(&[
        (FLASHINFER_GATES[0], "1"),
        (FLASHINFER_LIBRARY, "/profile/library.so"),
    ]);
    let operator = [
        (FLASHINFER_GATES[0], OsString::from("0")),
        (FLASHINFER_GATES[1], OsString::from("1")),
        (FLASHINFER_LIBRARY, OsString::from("/operator/library.so")),
    ];
    let mut seen = None;
    preflight_with(
        Some(&profile),
        |name| lookup(&operator, name),
        |path| {
            seen = Some(path.to_path_buf());
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(seen, Some(PathBuf::from("/operator/library.so")));
}

#[test]
fn explicit_zero_suppresses_profile_one_and_library_verification() {
    let profile = profile(&[
        (FLASHINFER_GATES[0], "1"),
        (FLASHINFER_LIBRARY, "/profile/library.so"),
    ]);
    let mut called = false;
    preflight_with(
        Some(&profile),
        |name| (name == FLASHINFER_GATES[0]).then(|| OsString::from("0")),
        |_| {
            called = true;
            Ok(())
        },
    )
    .unwrap();
    assert!(!called);
}

#[test]
fn each_gate_independently_requires_the_library() {
    for enabled_gate in FLASHINFER_GATES {
        let operator = [(enabled_gate, OsString::from("1"))];
        let error = preflight_with(
            None,
            |name| lookup(&operator, name),
            |_| panic!("missing library must refuse before verification"),
        )
        .unwrap_err();
        assert!(
            error.contains(FLASHINFER_LIBRARY),
            "{enabled_gate}: {error}"
        );
    }
}

#[test]
fn malformed_gate_values_are_rejected_exactly() {
    for malformed in ["", "true", "01", " 1 ", "2"] {
        let values = [(FLASHINFER_GATES[0], OsString::from(malformed))];
        let error = preflight_with(
            None,
            |name| lookup(&values, name),
            |_| panic!("malformed gate must refuse before verification"),
        )
        .unwrap_err();
        assert!(error.contains("unset, 0, or 1"), "{malformed:?}: {error}");
    }
}

#[cfg(unix)]
#[test]
fn non_unicode_operator_gate_is_rejected() {
    use std::os::unix::ffi::OsStringExt;

    let values = [(FLASHINFER_GATES[2], OsString::from_vec(vec![0xff]))];
    let error = preflight_with(
        None,
        |name| lookup(&values, name),
        |_| panic!("non-Unicode gate must refuse before verification"),
    )
    .unwrap_err();
    assert!(error.contains("not valid Unicode"), "{error}");
}

#[test]
fn absent_and_zero_gates_are_a_no_op() {
    let profile = profile(&[
        (FLASHINFER_GATES[0], "0"),
        (FLASHINFER_LIBRARY, "/unused/library.so"),
    ]);
    let mut called = false;
    preflight_with(
        Some(&profile),
        |_| None,
        |_| {
            called = true;
            Ok(())
        },
    )
    .unwrap();
    assert!(!called);

    preflight_with(None, |_| None, |_| Err("must not run".into())).unwrap();
}

#[test]
fn verifier_failure_is_preserved() {
    let operator = [
        (FLASHINFER_GATES[0], OsString::from("1")),
        (FLASHINFER_LIBRARY, OsString::from("/wrong/library.so")),
    ];
    let error = preflight_with(
        None,
        |name| lookup(&operator, name),
        |_| Err("qualified SHA-256 mismatch".into()),
    )
    .unwrap_err();
    assert_eq!(error, "qualified SHA-256 mismatch");
}

#[test]
fn native_mtp_profile_does_not_activate_flashinfer_preflight() {
    let recipes = crate::recipe::builtin::recipes();
    let recipe = recipes
        .iter()
        .find(|candidate| candidate.id == "qwen3.8/qwen3.8-flash-next-native-mtp-safe-local")
        .expect("native-MTP built-in");
    let profile = super::super::model_profile::profile_for_recipe(recipe)
        .expect("valid built-in profile")
        .expect("native-MTP profile");
    let operator = [(FLASHINFER_GATES[0], OsString::from("1"))];
    let mut called = false;
    preflight_model_with(
        &profile.model_type,
        Some(&profile),
        |name| lookup(&operator, name),
        |_| {
            called = true;
            Ok(())
        },
    )
    .unwrap();
    assert!(
        !called,
        "native MTP must remain isolated from dense FlashInfer"
    );
}
