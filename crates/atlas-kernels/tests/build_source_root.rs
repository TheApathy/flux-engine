// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../build_source_root.rs"]
mod build_source_root;

use build_source_root::{ptx_dir, source_root_warning};
use std::path::Path;

#[test]
fn ptx_dir_is_stable_per_tree_and_distinct_across_trees() {
    let out = Path::new("/target/release/build/atlas-kernels-0/out");
    let a = ptx_dir(out, Path::new("/nonexistent/worktree-a"));
    assert_eq!(a, ptx_dir(out, Path::new("/nonexistent/worktree-a")));
    assert_ne!(a, ptx_dir(out, Path::new("/nonexistent/worktree-b")));
    assert_eq!(a.parent(), Some(out));
    let name = a.file_name().unwrap().to_str().unwrap();
    assert!(
        name.starts_with("src-") && name.len() == "src-".len() + 16,
        "{name}"
    );
}

#[test]
fn ptx_dir_keys_the_canonical_root() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let canonical = root.canonicalize().unwrap();
    let out = Path::new("/out");
    assert_eq!(ptx_dir(out, &root), ptx_dir(out, &canonical));
}

#[test]
fn warning_only_when_the_root_is_unset_or_another_tree() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    // cargo's `relative = true` renders "." as "<repo>/.".
    let declared = format!("{}/.", root.canonicalize().unwrap().display());
    assert_eq!(source_root_warning(Some(&declared), &root), None);
    for unset in [None, Some("")] {
        let warning = source_root_warning(unset, &root).unwrap();
        assert!(warning.contains("is unset"), "{warning}");
    }
    let other = source_root_warning(Some("/nonexistent/other-tree"), &root).unwrap();
    assert!(other.contains("different tree"), "{other}");
}
