// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only identity admission for the qualified FlashInfer SM121 library.
//!
//! Kept separate from the CUDA ABI wrapper so launch planning and hot-swap can
//! reject a missing or replaced artifact before touching a live model.

use std::fs::File;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

pub const QUALIFIED_FLASHINFER_SM121_SHA256: [u8; 32] = [
    0xa0, 0x07, 0xa8, 0x25, 0x66, 0xca, 0x3d, 0x31, 0x15, 0xc8, 0xcc, 0x0e, 0x73, 0xe2, 0xbb, 0xfc,
    0x0b, 0xd1, 0xc7, 0x6b, 0x63, 0x42, 0x31, 0x3f, 0xba, 0x7e, 0xe5, 0x48, 0x3e, 0x49, 0xc0, 0x20,
];
pub(crate) const MAX_LIBRARY_BYTES: u64 = 64 * 1024 * 1024;

pub(crate) type VerifiedSource = (PathBuf, Vec<u8>, [u8; 32]);

/// Validate the exact production artifact without loading it or touching CUDA.
pub fn preflight_qualified_source(path: &Path) -> Result<(), String> {
    verify_source(path, QUALIFIED_FLASHINFER_SM121_SHA256).map(|_| ())
}

/// Read one immutable candidate and prove its expected content identity.
pub(crate) fn verify_source(
    path: &Path,
    expected_sha256: [u8; 32],
) -> Result<VerifiedSource, String> {
    if !path.is_absolute() {
        return Err("FlashInfer SM121 library path must be absolute".into());
    }
    let canonical = path.canonicalize().map_err(|error| {
        format!(
            "canonicalize FlashInfer library {}: {error}",
            path.display()
        )
    })?;
    let mut file = File::open(&canonical)
        .map_err(|error| format!("open FlashInfer library {}: {error}", canonical.display()))?;
    let before = file
        .metadata()
        .map_err(|error| format!("stat FlashInfer library {}: {error}", canonical.display()))?;
    if !before.is_file() {
        return Err(format!(
            "FlashInfer SM121 library must be a regular file: {}",
            canonical.display()
        ));
    }
    if before.len() == 0 || before.len() > MAX_LIBRARY_BYTES {
        return Err(format!(
            "FlashInfer SM121 library size {} is outside 1..={MAX_LIBRARY_BYTES} bytes",
            before.len()
        ));
    }

    let mut bytes = Vec::new();
    (&mut file)
        .take(MAX_LIBRARY_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read FlashInfer library {}: {error}", canonical.display()))?;
    if bytes.len() as u64 > MAX_LIBRARY_BYTES {
        return Err(format!(
            "FlashInfer SM121 library exceeds {MAX_LIBRARY_BYTES} bytes while hashing"
        ));
    }
    let actual_sha256 = sha256(&bytes);
    if actual_sha256 != expected_sha256 {
        return Err(format!(
            "FlashInfer SM121 library SHA-256 mismatch: expected {}, got {}",
            hex_sha256(expected_sha256),
            hex_sha256(actual_sha256)
        ));
    }
    let after = file.metadata().map_err(|error| {
        format!(
            "restat FlashInfer library {} after hashing: {error}",
            canonical.display()
        )
    })?;
    if !same_file_identity(&before, &after) {
        return Err(format!(
            "FlashInfer SM121 source library changed while it was being read: {}",
            canonical.display()
        ));
    }
    Ok((canonical, bytes, actual_sha256))
}

fn same_file_identity(before: &std::fs::Metadata, after: &std::fs::Metadata) -> bool {
    before.dev() == after.dev()
        && before.ino() == after.ino()
        && before.len() == after.len()
        && before.mtime() == after.mtime()
        && before.mtime_nsec() == after.mtime_nsec()
        && before.ctime() == after.ctime()
        && before.ctime_nsec() == after.ctime_nsec()
}

pub(crate) fn hex_sha256(digest: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

// Small dependency-free SHA-256 retained from the runtime loader.
pub(crate) fn sha256(input: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let padded_len = (input.len() + 9).div_ceil(64) * 64;
    let mut padded = Vec::with_capacity(padded_len);
    padded.extend_from_slice(input);
    padded.push(0x80);
    padded.resize(padded_len - 8, 0);
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (index, word) in chunk.chunks_exact(4).enumerate() {
            w[index] = u32::from_be_bytes(word.try_into().expect("four-byte SHA-256 word"));
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut digest = [0u8; 32];
    for (word, bytes) in state.iter().zip(digest.chunks_exact_mut(4)) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

#[cfg(test)]
#[path = "flashinfer_sm121_artifact_tests.rs"]
mod tests;
