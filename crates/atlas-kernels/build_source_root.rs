// SPDX-License-Identifier: AGPL-3.0-only

//! Which source tree a build-script run belongs to, for a shared
//! `CARGO_TARGET_DIR`.
//!
//! Cargo hashes a workspace member's package id relative to the workspace
//! root, so every worktree of this repository shares one atlas-kernels
//! `OUT_DIR`. The script's `rerun-if-changed` paths outside the crate
//! (`kernels/...`) are absolute into whichever worktree ran it last, so a
//! second worktree saw them unchanged and reused that tree's `target_ptx.rs`
//! and PTX. Two fixes, both keyed on the source root:
//!
//! - `.cargo/config.toml` sets `ATLAS_BUILD_SOURCE_ROOT` (relative, forced)
//!   and the script declares `rerun-if-env-changed` on it, so switching trees
//!   reruns the script while a same-tree no-op build stays fresh;
//! - PTX lives in `OUT_DIR/src-<hash of the root>/`, so a rerun never trusts
//!   another tree's mtimes and each tree keeps its own incremental cache.

use std::path::{Path, PathBuf};

pub const SOURCE_ROOT_ENV: &str = "ATLAS_BUILD_SOURCE_ROOT";

/// 64-bit FNV-1a: stable across toolchains, unlike `DefaultHasher`.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Canonical workspace root, or the path as given when it cannot be resolved.
pub fn canonical_root(workspace_root: &Path) -> PathBuf {
    workspace_root
        .canonicalize()
        .unwrap_or_else(|_| workspace_root.to_path_buf())
}

/// The per-source-tree PTX directory inside `out_dir`.
pub fn ptx_dir(out_dir: &Path, workspace_root: &Path) -> PathBuf {
    let root = canonical_root(workspace_root);
    let hash = fnv1a64(root.as_os_str().as_encoded_bytes());
    out_dir.join(format!("src-{hash:016x}"))
}

/// A `cargo:warning` when `ATLAS_BUILD_SOURCE_ROOT` cannot key this run:
/// unset (cargo ran without this repository's `.cargo/config.toml`, e.g.
/// `--manifest-path` from another directory), or naming a different tree.
pub fn source_root_warning(declared: Option<&str>, workspace_root: &Path) -> Option<String> {
    let root = canonical_root(workspace_root);
    match declared {
        None | Some("") => Some(format!(
            "{SOURCE_ROOT_ENV} is unset (build from the repository so .cargo/config.toml applies); \
             a shared CARGO_TARGET_DIR cannot tell this tree ({}) from another worktree",
            root.display()
        )),
        Some(value) if canonical_root(Path::new(value)) != root => Some(format!(
            "{SOURCE_ROOT_ENV}={value} names a different tree than {}; a shared CARGO_TARGET_DIR \
             may rerun this script on every switch",
            root.display()
        )),
        Some(_) => None,
    }
}
