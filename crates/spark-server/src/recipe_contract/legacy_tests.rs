// SPDX-License-Identifier: AGPL-3.0-only

//! Legacy mode: no `--recipe` and no `FLUX_RECIPE` must change nothing.

use super::{Mode, RECIPE_ENV, enforce_at_startup, mode};
use crate::cli::ServeArgs;
use clap::Parser;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

#[test]
fn no_flag_and_no_env_is_legacy() {
    assert_eq!(mode(None, None), Mode::Legacy);
    assert_eq!(mode(None, Some(OsStr::new(""))), Mode::Legacy);
}

#[test]
fn the_flag_wins_over_the_env() {
    let flag = Path::new("/a.toml");
    assert_eq!(
        mode(Some(flag), Some(OsStr::new("/b.toml"))),
        Mode::Recipe(PathBuf::from("/a.toml"))
    );
    assert_eq!(
        mode(None, Some(OsStr::new("/b.toml"))),
        Mode::Recipe(PathBuf::from("/b.toml"))
    );
}

#[test]
fn the_flag_parses_and_defaults_to_none() {
    let without = ServeArgs::parse_from(["spark", "org/model", "--max-batch-size", "1"]);
    assert_eq!(without.recipe, None);
    let mut with = ServeArgs::parse_from([
        "spark",
        "org/model",
        "--max-batch-size",
        "1",
        "--recipe",
        "/r.toml",
    ]);
    assert_eq!(with.recipe.as_deref(), Some(Path::new("/r.toml")));
    // The flag adds one field and touches no other parsed value.
    with.recipe = None;
    assert_eq!(with, without);
}

#[test]
fn legacy_startup_checks_nothing() {
    // Only meaningful when the test process itself has no FLUX_RECIPE; a
    // recipe-mode harness would otherwise be exercised by its own tests.
    if std::env::var_os(RECIPE_ENV).is_some() {
        return;
    }
    // A hub id with no local directory would refuse in recipe mode; legacy
    // mode must not look at it.
    let args = ServeArgs::parse_from(["spark", "org/not-a-local-dir"]);
    assert!(enforce_at_startup(&args).is_ok());
    assert!(enforce_at_startup(&args).is_ok(), "idempotent");
}
