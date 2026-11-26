// SPDX-License-Identifier: AGPL-3.0-only
// Shared build-script policy. These flags do not qualify CUDA device or AOT bytes.

use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;

const FROM: &str = "ATLAS_RELEASE_PATH_REMAP_FROM";
const TO: &str = "ATLAS_RELEASE_PATH_REMAP_TO";
const POLICY: &str = "ATLAS_RELEASE_PATH_REMAP_POLICY";
const SCHEMA: &str = "flux.release-path-remap-policy/1";
const PUBLIC_ROOT: &str = "/flux-build";

fn canonical_root(value: &str) -> bool {
    value.len() > 1
        && value.len() <= 4096
        && value.starts_with('/')
        && !value.ends_with('/')
        && !value.contains("//")
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"/_-.".contains(&c))
        && value
            .split('/')
            .skip(1)
            .all(|c| !c.is_empty() && c != "." && c != "..")
}

fn environment(key: &str) -> Result<Option<String>, &'static str> {
    std::env::var_os(key)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| "non-UTF8 remap environment refused")
        })
        .transpose()
}

fn roots() -> Result<Option<(String, String)>, &'static str> {
    let from = environment(FROM)?;
    let to = environment(TO)?;
    let policy = environment(POLICY)?;
    match (from, to) {
        (None, None) if policy.is_none() => Ok(None),
        (Some(from), Some(to)) => {
            if !canonical_root(&from)
                || !canonical_root(&to)
                || (to != PUBLIC_ROOT && to != "/opt/fluxsrc")
            {
                return Err("canonical remap roots and fixed public destination required");
            }
            let source = Path::new(&from);
            let destination = Path::new(&to);
            if source.starts_with(destination) || destination.starts_with(source) {
                return Err("disjoint remap roots required");
            }
            if !source.is_dir() || std::fs::canonicalize(source).ok().as_deref() != Some(source) {
                return Err("existing canonical remap source directory required");
            }
            if policy.as_deref().is_some_and(|value| value != SCHEMA) {
                return Err("unknown path-remap policy refused");
            }
            Ok(Some((from, to)))
        }
        _ => Err("path-remap roots must be configured together"),
    }
}

pub fn configure() -> bool {
    println!("cargo:rerun-if-changed=../../build-support/release_path_remap.rs");
    for key in [FROM, TO, POLICY] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    roots().expect("release path-remap HOLD").is_some()
}

pub fn compiler(tool: impl AsRef<OsStr>, nvcc: bool) -> Command {
    let mut command = Command::new(tool);
    if let Some((from, to)) = roots().expect("release path-remap HOLD") {
        for option in [
            "-ffile-prefix-map",
            "-fmacro-prefix-map",
            "-fdebug-prefix-map",
        ] {
            let flag = format!("{option}={from}={to}");
            if nvcc {
                command.arg("-Xcompiler");
            }
            command.arg(flag);
        }
    }
    command
}

#[cfg(test)]
mod tests {
    use super::canonical_root;

    #[test]
    fn lexical_roots_are_unambiguous() {
        assert!(canonical_root("/private/build-root"));
        assert!(canonical_root("/flux-build"));
        for value in [
            "",
            "/",
            "relative",
            "/a/",
            "/a//b",
            "/a/./b",
            "/a/../b",
            "/a=b",
            "/a,b",
            "/a b",
            "/a\\b",
            "/a\n",
            "/a\u{1f}b",
        ] {
            assert!(!canonical_root(value));
        }
    }
}
