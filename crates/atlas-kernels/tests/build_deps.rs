// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../build_deps.rs"]
mod build_deps;

use build_deps::{dependency_manifest, include_closure, latest_mtime, quoted_includes};
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// A fresh scratch directory (no tempfile dev-dependency in this crate).
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("atlas-build-deps-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn only_quoted_includes_are_followed() {
    let source = "#include <cuda_fp16.h>\n  #  include \"../../qwen3.8-27b/nvfp4/gated_delta_rule.cu\"\n// #include \"commented.cu\" is still a line\n#define X 1\n";
    assert_eq!(
        quoted_includes(source),
        vec!["../../qwen3.8-27b/nvfp4/gated_delta_rule.cu".to_string()]
    );
}

#[test]
fn wrapper_reaches_the_included_source_and_its_headers() {
    let root = Scratch::new("wrapper");
    let fn_dir = root.path().join("qwen3.8-flash-next/nvfp4");
    let src_dir = root.path().join("qwen3.8-27b/nvfp4");
    std::fs::create_dir_all(&fn_dir).unwrap();
    std::fs::create_dir_all(&src_dir).unwrap();
    let wrapper = fn_dir.join("gated_delta_rule.cu");
    std::fs::write(
        &wrapper,
        "#include \"../../qwen3.8-27b/nvfp4/gated_delta_rule.cu\"\n",
    )
    .unwrap();
    std::fs::write(
        src_dir.join("gated_delta_rule.cu"),
        "#include \"gdn_common.cuh\"\n#include <cuda.h>\n",
    )
    .unwrap();
    std::fs::write(
        src_dir.join("gdn_common.cuh"),
        "#include \"gated_delta_rule.cu\"\n",
    )
    .unwrap();

    let deps = include_closure(&wrapper);
    let names: Vec<_> = deps
        .iter()
        .map(|p| {
            p.strip_prefix(root.path().canonicalize().unwrap())
                .unwrap()
                .to_path_buf()
        })
        .collect();
    assert_eq!(
        names,
        vec![
            std::path::PathBuf::from("qwen3.8-27b/nvfp4/gated_delta_rule.cu"),
            std::path::PathBuf::from("qwen3.8-27b/nvfp4/gdn_common.cuh"),
        ]
    );

    // The included file being newer than the cached PTX must invalidate it.
    let later = SystemTime::now() + Duration::from_secs(3600);
    std::fs::File::options()
        .write(true)
        .open(src_dir.join("gated_delta_rule.cu"))
        .unwrap()
        .set_modified(later)
        .unwrap();
    assert_eq!(latest_mtime(&deps), Some(later));
}

#[test]
fn missing_includes_and_a_file_without_includes_yield_nothing() {
    let root = Scratch::new("lone");
    let lone = root.path().join("lone.cu");
    std::fs::write(
        &lone,
        "#include \"not_there.cuh\"\n__global__ void k() {}\n",
    )
    .unwrap();
    assert!(include_closure(&lone).is_empty());
    assert_eq!(latest_mtime(&[]), None);
}

#[test]
fn deleting_a_resolved_include_changes_manifest_despite_unchanged_wrapper() {
    let root = Scratch::new("deleted-manifest");
    let wrapper = root.path().join("wrapper.cu");
    let header = root.path().join("dependency.cuh");
    std::fs::write(&wrapper, "#include \"dependency.cuh\"\n").unwrap();
    std::fs::write(&header, "// original\n").unwrap();
    let before = dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap();
    std::fs::remove_file(&header).unwrap();
    let after = dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap();
    assert_ne!(
        before, after,
        "deleted resolved input must invalidate the cache"
    );
    // Different restored bytes recompile.
    std::fs::write(&header, "// restored\n").unwrap();
    let restored = dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap();
    assert_ne!(after, restored);
}

#[test]
fn inactive_missing_include_does_not_prevent_a_stable_manifest() {
    let root = Scratch::new("inactive-manifest");
    let wrapper = root.path().join("wrapper.cu");
    std::fs::write(&wrapper, "#if 0\n#include \"optional.cuh\"\n#endif\n").unwrap();
    let first = dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap();
    assert_eq!(
        first,
        dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap()
    );
}

#[test]
fn unreadable_text_cannot_validate_an_incomplete_closure() {
    let root = Scratch::new("unreadable-manifest");
    let wrapper = root.path().join("wrapper.cu");
    let header = root.path().join("dependency.cuh");
    std::fs::write(&wrapper, "#include \"dependency.cuh\"\n").unwrap();
    std::fs::write(&header, [0xff]).unwrap();
    let deps = include_closure(&wrapper);
    assert_eq!(deps.len(), 1);
    assert!(dependency_manifest(&wrapper, &deps).is_err());
}

#[test]
fn input_removed_after_traversal_cannot_validate_a_manifest() {
    let root = Scratch::new("removed-after-walk");
    let wrapper = root.path().join("wrapper.cu");
    let header = root.path().join("dependency.cuh");
    std::fs::write(&wrapper, "#include \"dependency.cuh\"\n").unwrap();
    std::fs::write(&header, "// original\n").unwrap();
    let deps = include_closure(&wrapper);
    std::fs::remove_file(&header).unwrap();
    assert!(dependency_manifest(&wrapper, &deps).is_err());
}

#[test]
fn renaming_another_file_over_an_include_changes_manifest() {
    // `mv other.cuh dependency.cuh` keeps other.cuh's (older) mtime, so the
    // PTX cache's mtime test alone accepted the old PTX; the path set is the
    // same, so only the file identity can tell.
    let root = Scratch::new("renamed-over-manifest");
    let wrapper = root.path().join("wrapper.cu");
    let header = root.path().join("dependency.cuh");
    let other = root.path().join("other.cuh");
    std::fs::write(&wrapper, "#include \"dependency.cuh\"\n").unwrap();
    std::fs::write(&header, "#define V 1\n").unwrap();
    std::fs::write(&other, "#define V 2 // different\n").unwrap();
    let before = dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap();
    std::fs::rename(&other, &header).unwrap();
    let after = dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap();
    assert_ne!(
        before, after,
        "a file renamed over an include must invalidate the cache"
    );
}

#[test]
fn renaming_an_include_away_changes_manifest() {
    let root = Scratch::new("renamed-away-manifest");
    let wrapper = root.path().join("wrapper.cu");
    let header = root.path().join("dependency.cuh");
    std::fs::write(&wrapper, "#include \"dependency.cuh\"\n").unwrap();
    std::fs::write(&header, "// original\n").unwrap();
    let before = dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap();
    std::fs::rename(&header, root.path().join("renamed.cuh")).unwrap();
    let after = dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap();
    assert_ne!(
        before, after,
        "an include renamed away must invalidate the cache"
    );
}

#[test]
fn an_unchanged_tree_keeps_the_same_manifest() {
    let root = Scratch::new("stable-manifest");
    let wrapper = root.path().join("wrapper.cu");
    std::fs::write(&wrapper, "#include \"dependency.cuh\"\n").unwrap();
    std::fs::write(root.path().join("dependency.cuh"), "// x\n").unwrap();
    let first = dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap();
    assert_eq!(
        first,
        dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap()
    );
}

#[test]
fn same_size_overwrite_keeping_mtime_changes_manifest() {
    // `cp -p other dep` / `rsync -t`: same path and inode, old mtime, same size.
    let root = Scratch::new("cp-p-manifest");
    let wrapper = root.path().join("wrapper.cu");
    let header = root.path().join("dependency.cuh");
    std::fs::write(&wrapper, "#include \"dependency.cuh\"\n").unwrap();
    std::fs::write(&header, "#define V 1\n").unwrap();
    let before = dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap();
    let mtime = std::fs::metadata(&header).unwrap().modified().unwrap();
    std::fs::write(&header, "#define V 2\n").unwrap();
    std::fs::File::options()
        .write(true)
        .open(&header)
        .unwrap()
        .set_modified(mtime)
        .unwrap();
    let after = dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap();
    assert_ne!(
        before, after,
        "same-size in-place overwrite must invalidate the cache"
    );
}

#[test]
fn identical_bytes_under_a_new_inode_keep_the_manifest() {
    // An editor's save-by-rename writes the same bytes to a new inode.
    let root = Scratch::new("rename-on-save-manifest");
    let wrapper = root.path().join("wrapper.cu");
    let header = root.path().join("dependency.cuh");
    std::fs::write(&wrapper, "#include \"dependency.cuh\"\n").unwrap();
    std::fs::write(&header, "#define V 1\n").unwrap();
    let before = dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap();
    let staged = root.path().join("dependency.cuh~");
    std::fs::write(&staged, "#define V 1\n").unwrap();
    std::fs::rename(&staged, &header).unwrap();
    assert_eq!(
        before,
        dependency_manifest(&wrapper, &include_closure(&wrapper)).unwrap()
    );
}
