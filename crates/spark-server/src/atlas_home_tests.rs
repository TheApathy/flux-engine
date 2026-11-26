// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::ffi::OsString;

/// POSITIVE: an explicit `FLUX_HOME` or `ATLAS_HOME` wins outright, and reports which var matched.
#[test]
fn an_explicit_atlas_home_wins() {
    let h = resolve_from(
        Some(("ATLAS_HOME", OsString::from("/tmp/explicit-root"))),
        None,
    )
    .unwrap();
    assert_eq!(h.root, PathBuf::from("/tmp/explicit-root"));
    assert_eq!(h.source, HomeSource::Env("ATLAS_HOME"));
    assert!(h.describe().contains("(ATLAS_HOME)"), "{}", h.describe());

    let f = resolve_from(Some(("FLUX_HOME", OsString::from("/tmp/flux-root"))), None).unwrap();
    assert_eq!(f.root, PathBuf::from("/tmp/flux-root"));
    assert_eq!(f.source, HomeSource::Env("FLUX_HOME"));
    assert!(f.describe().contains("(FLUX_HOME)"), "{}", f.describe());
}

/// An empty `FLUX_HOME` / `ATLAS_HOME` is an error, not a silent fall back to `$HOME`.
/// Falling back would place artifacts somewhere the operator did not ask for
/// while the variable they set said otherwise.
#[test]
fn an_empty_atlas_home_is_an_error_not_a_fallback() {
    let e = resolve_from(
        Some(("ATLAS_HOME", OsString::new())),
        Some(OsString::from("/tmp/u")),
    )
    .unwrap_err();
    assert!(e.to_string().contains("ATLAS_HOME is set but empty"), "{e}");

    let ef = resolve_from(
        Some(("FLUX_HOME", OsString::new())),
        Some(OsString::from("/tmp/u")),
    )
    .unwrap_err();
    assert!(
        ef.to_string().contains("FLUX_HOME is set but empty"),
        "{ef}"
    );
}

/// The default is `$HOME/.atlas`.
#[test]
fn the_default_is_dot_atlas() {
    let tmp = tempfile::tempdir().unwrap();
    let h = resolve_from(None, Some(tmp.path().into())).unwrap();
    assert_eq!(h.root, tmp.path().join(".atlas"));
    assert_eq!(h.source, HomeSource::HomeDefault);
}

/// None of the variables set is an error naming all three.
#[test]
fn no_home_at_all_names_both_variables() {
    let e = resolve_from(None, None).unwrap_err();
    let s = e.to_string();
    assert!(
        s.contains("FLUX_HOME") && s.contains("ATLAS_HOME") && s.contains("HOME"),
        "{s}"
    );
}
