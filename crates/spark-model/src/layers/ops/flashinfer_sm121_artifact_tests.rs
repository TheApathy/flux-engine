// SPDX-License-Identifier: AGPL-3.0-only

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

fn scratch(name: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "atlas-fi-artifact-{}-{name}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn sha256_vectors_and_qualified_identity_are_frozen() {
    assert_eq!(
        hex_sha256(sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex_sha256(sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        hex_sha256(QUALIFIED_FLASHINFER_SM121_SHA256),
        "a007a82566ca3d3115c8cc0e73e2bbfc0bd1c76b6342313fba7ee5483e49c020"
    );
}

#[test]
fn stable_reader_accepts_only_the_expected_regular_file() {
    let path = scratch("bytes");
    std::fs::write(&path, b"qualified test bytes").unwrap();
    let expected = sha256(b"qualified test bytes");
    assert!(verify_source(&path, expected).is_ok());
    assert!(
        verify_source(&path, [0; 32])
            .unwrap_err()
            .contains("SHA-256 mismatch")
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn hostile_paths_fail_closed() {
    assert!(preflight_qualified_source(Path::new("relative.so")).is_err());
    assert!(preflight_qualified_source(Path::new("/definitely/missing/atlas-fi.so")).is_err());
    let directory = scratch("directory");
    std::fs::create_dir(&directory).unwrap();
    assert!(verify_source(&directory, [0; 32]).is_err());
    std::fs::remove_dir(directory).unwrap();

    let empty = scratch("empty");
    std::fs::write(&empty, b"").unwrap();
    assert!(verify_source(&empty, sha256(b"")).is_err());
    std::fs::remove_file(empty).unwrap();

    let oversized = scratch("oversized");
    std::fs::File::create(&oversized)
        .unwrap()
        .set_len(MAX_LIBRARY_BYTES + 1)
        .unwrap();
    assert!(verify_source(&oversized, [0; 32]).is_err());
    std::fs::remove_file(oversized).unwrap();
}
