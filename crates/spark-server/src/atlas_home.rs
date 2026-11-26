// SPDX-License-Identifier: AGPL-3.0-only

//! Where Flux keeps recipes, artifacts and run history: `~/.atlas` (overridable
//! via `FLUX_HOME` or `ATLAS_HOME`).

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// Where the root came from — worth keeping, because two commands disagreeing
/// about the root is otherwise invisible to an operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HomeSource {
    /// `FLUX_HOME` or `ATLAS_HOME` was set (carries the matched env var name).
    Env(&'static str),
    /// `$HOME/.atlas`.
    HomeDefault,
}

#[derive(Clone, Debug)]
pub struct AtlasHome {
    pub root: PathBuf,
    pub source: HomeSource,
}

impl AtlasHome {
    pub fn resolve() -> Result<Self> {
        let env_override = std::env::var_os("FLUX_HOME")
            .map(|v| ("FLUX_HOME", v))
            .or_else(|| std::env::var_os("ATLAS_HOME").map(|v| ("ATLAS_HOME", v)));
        resolve_from(env_override, std::env::var_os("HOME"))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// One line naming the root and where it came from.
    pub fn describe(&self) -> String {
        let src = match self.source {
            HomeSource::Env(var) => var,
            HomeSource::HomeDefault => "$HOME/.atlas",
        };
        format!("{} ({src})", self.root.display())
    }
}

/// [`AtlasHome::resolve`] over explicit inputs, so the rules can be tested.
///
/// Pure over the ENVIRONMENT on purpose: `set_var` is unsafe and
/// process-global, so a test that mutated `HOME` could race another test's read
/// and produce intermittent failures.
pub(crate) fn resolve_from(
    env_override: Option<(&'static str, std::ffi::OsString)>,
    home: Option<std::ffi::OsString>,
) -> Result<AtlasHome> {
    if let Some((var, explicit)) = env_override {
        let root = PathBuf::from(explicit);
        if root.as_os_str().is_empty() {
            bail!("{var} is set but empty");
        }
        return Ok(AtlasHome {
            root,
            source: HomeSource::Env(var),
        });
    }
    let home = home
        .map(PathBuf::from)
        .filter(|h| !h.as_os_str().is_empty())
        .context("none of FLUX_HOME, ATLAS_HOME, or HOME is set — cannot place ~/.atlas")?;
    let root = home.join(".atlas");
    Ok(AtlasHome {
        root,
        source: HomeSource::HomeDefault,
    })
}

/// Can this process actually use the home? `None` means yes.
///
/// Probes by writing rather than reading mode bits: ownership, ACLs, a
/// read-only mount, and a full disk all present differently in metadata but
/// identically in whether a write succeeds.
pub fn check_usable(root: &Path) -> Option<String> {
    if root.exists() && !root.is_dir() {
        return Some(format!("{} exists and is not a directory", root.display()));
    }
    if !root.exists()
        && let Err(e) = std::fs::create_dir_all(root)
    {
        return Some(format!("could not create {}: {e}", root.display()));
    }
    let probe = root.join(".atlas-write-probe");
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            None
        }
        Err(e) => Some(format!("{} is not writable: {e}", root.display())),
    }
}

#[cfg(test)]
#[path = "atlas_home_tests.rs"]
mod tests;
