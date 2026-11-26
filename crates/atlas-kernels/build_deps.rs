// SPDX-License-Identifier: AGPL-3.0-only

//! Quoted `#include` closure of a kernel source, for the PTX cache.
//!
//! Several target `.cu` files are one-line wrappers around another
//! target's source (`#include "../../qwen3.8-27b/nvfp4/gated_delta_rule.cu"`).
//! The per-file freshness check and cargo's `rerun-if-changed` list only saw
//! the wrapper, so an edit to the included file shipped stale PTX. Quoted
//! includes are resolved relative to the including file, and across explicit
//! `-I` search directories via [`include_closure_with_dirs`]; system `<...>`
//! includes (CUDA, ExLlamaV3) are outside our dependency walk.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Targets of `#include "..."` lines, in source order.
pub fn quoted_includes(source: &str) -> Vec<String> {
    source
        .lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix('#')?.trim_start();
            let rest = rest.strip_prefix("include")?.trim_start();
            let rest = rest.strip_prefix('"')?;
            rest.find('"').map(|end| rest[..end].to_string())
        })
        .collect()
}

/// Every file `source` reaches through quoted includes (not `source` itself),
/// sorted and deduplicated. Cycles and unreadable files stop the walk.
pub fn include_closure(source: &Path) -> Vec<PathBuf> {
    include_closure_with_dirs(source, &[])
}

/// Like [`include_closure`], additionally falling back to `-I` search directories
/// (`include_dirs`, searched in order) when a quoted include is not present next
/// to the including file.
pub fn include_closure_with_dirs(source: &Path, include_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![source.to_path_buf()];
    while let Some(file) = stack.pop() {
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        let dir = file.parent().unwrap_or_else(|| Path::new("."));
        for include in quoted_includes(&text) {
            let resolved = dir
                .join(&include)
                .canonicalize()
                .ok()
                .filter(|p| p.is_file())
                .or_else(|| {
                    include_dirs.iter().find_map(|inc| {
                        inc.join(&include)
                            .canonicalize()
                            .ok()
                            .filter(|p| p.is_file())
                    })
                });
            if let Some(path) = resolved
                && seen.insert(path.clone())
            {
                stack.push(path);
            }
        }
    }
    let own = source.canonicalize().ok();
    seen.into_iter()
        .filter(|p| Some(p) != own.as_ref())
        .collect()
}

/// Newest modification time among `paths` (missing files are skipped).
pub fn latest_mtime(paths: &[PathBuf]) -> Option<SystemTime> {
    paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
        .max()
}

/// Describe the resolved include set for a successful compiler invocation.
///
/// Missing textual includes are deliberately absent: they may belong to inactive
/// preprocessor branches or compiler search paths. Comparing this manifest with
/// the previous successful invocation detects a formerly resolved file vanishing.
/// Unreadable resolved inputs make the cache ineligible, leaving diagnostics to
/// the compiler rather than treating an incomplete traversal as a cache hit.
///
/// Each path also records its content: SHA-256 and size of the bytes. A
/// rename (`mv`) or an in-place copy that keeps the old mtime (`cp -p`,
/// `rsync -t`) puts different content at the same path, which the mtime
/// test alone accepted; different bytes now miss. Identical bytes (an
/// editor's rename-on-save) keep the cache. Cargo reruns build.rs on mtimes
/// only, so an in-place `cp -p` alone is caught at the next rerun for any
/// reason; release builds are fresh clones and always compile every kernel.
pub fn dependency_manifest(source: &Path, deps: &[PathBuf]) -> std::io::Result<String> {
    let mut paths = BTreeSet::new();
    paths.insert(source.canonicalize()?);
    for dep in deps {
        paths.insert(dep.canonicalize()?);
    }
    let mut manifest = String::from("atlas-kernel-deps-v3\n");
    for path in paths {
        let bytes = std::fs::read(&path)?;
        std::str::from_utf8(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let digest = ring::digest::digest(&ring::digest::SHA256, &bytes);
        let hex: String = digest.as_ref().iter().map(|b| format!("{b:02x}")).collect();
        // Debug formatting escapes path delimiters, including embedded newlines.
        manifest.push_str(&format!("{path:?} {hex} {}\n", bytes.len()));
    }
    Ok(manifest)
}
