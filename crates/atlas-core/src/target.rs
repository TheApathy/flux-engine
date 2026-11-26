// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel target descriptor for (Hardware, Model_quantization) tuples
//! and feature-independent AOT CUBIN registration store.
//!
//! Every set of Atlas kernels is hyperoptimized for a specific target.
//! This module provides the `KernelTarget` type that serves as the
//! indexing key for PTX module sets, benchmarks, and kernel dispatch,
//! plus the content-addressed AOT CUBIN lookup table populated by
//! `atlas-kernels` and consumed by `crate::registry` at CUDA init time.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use ring::digest::{SHA256, digest as sha256_digest};

/// Describes a (Hardware, Model, Quantization) target for kernel selection.
///
/// Each unique `KernelTarget` maps to a distinct set of PTX modules
/// that have been hyperoptimized for that specific combination.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelTarget {
    /// SM architecture identifier (e.g., "sm_121", "sm_100a").
    pub arch: &'static str,
    /// Model identifier (e.g., "qwen3-next-80b-a3b").
    pub model: &'static str,
    /// Quantization scheme (e.g., "nvfp4", "fp8", "bf16").
    pub quant: &'static str,
}

impl KernelTarget {
    /// GB10 + Qwen3-Next-80B-A3B + NVFP4.
    pub const GB10_QWEN3_NVFP4: Self = Self {
        arch: "sm_121",
        model: "qwen3-next-80b-a3b",
        quant: "nvfp4",
    };

    /// GB10 + Qwen3.5-35B-A3B + NVFP4.
    pub const GB10_QWEN35_NVFP4: Self = Self {
        arch: "sm_121",
        model: "qwen3.5-35b-a3b",
        quant: "nvfp4",
    };

    /// GB10 + Qwen3.5-122B-A10B + NVFP4.
    pub const GB10_QWEN35_122B_NVFP4: Self = Self {
        arch: "sm_121",
        model: "qwen3.5-122b-a10b",
        quant: "nvfp4",
    };

    /// Check if this target's model name contains the given substring.
    pub fn model_contains(&self, substring: &str) -> bool {
        self.model.contains(substring)
    }
}

impl std::fmt::Display for KernelTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.arch, self.model, self.quant)
    }
}

/// Wire encoding of an embedded AOT CUBIN payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AotCubinEncoding {
    /// Raw ELF CUBIN bytes ready for `cuModuleLoadData`.
    Raw,
    /// XOR-stream obfuscated ELF CUBIN bytes (decoded on lookup).
    Obfuscated,
}

/// Content-addressed key tying an AOT CUBIN to the exact PTX bytes it was
/// compiled from, preventing stale or synthetic-PTX mismatches.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AotCubinKey {
    pub module_name: String,
    pub ptx_len: usize,
    pub ptx_sha256: [u8; 32],
}

impl AotCubinKey {
    /// Construct a key from a precomputed PTX byte length and SHA-256 digest.
    #[must_use]
    pub fn from_digest(
        module_name: impl Into<String>,
        ptx_len: usize,
        ptx_sha256: [u8; 32],
    ) -> Self {
        Self {
            module_name: module_name.into(),
            ptx_len,
            ptx_sha256,
        }
    }

    /// Compute a content key from the raw PTX source text.
    #[must_use]
    pub fn from_ptx(module_name: &str, ptx_src: &str) -> Self {
        Self {
            module_name: module_name.to_owned(),
            ptx_len: ptx_src.len(),
            ptx_sha256: sha256_bytes(ptx_src.as_bytes()),
        }
    }
}

/// Compute the SHA-256 digest of a byte slice.
#[must_use]
pub fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    let digest = sha256_digest(&SHA256, bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(digest.as_ref());
    out
}

/// Decode XOR-obfuscated binary bytes (e.g. AOT CUBIN ELF containing NUL bytes).
///
/// Mirrors `atlas-kernels/build_codegen.rs::obfuscate_ptx_bytes` (SplitMix64
/// keystream keyed on buffer length) and is its own inverse.
#[must_use]
pub fn decode_obf_bytes(src: &[u8]) -> Vec<u8> {
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15 ^ (src.len() as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    src.iter()
        .map(|&b| {
            s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            b ^ (z as u8)
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct AotCubinEntry {
    bytes: &'static [u8],
    encoding: AotCubinEncoding,
}

fn aot_cubin_store() -> &'static Mutex<HashMap<AotCubinKey, AotCubinEntry>> {
    static STORE: OnceLock<Mutex<HashMap<AotCubinKey, AotCubinEntry>>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register an embedded AOT CUBIN keyed by `(module_name, ptx_len, ptx_sha256)`.
///
/// Stores only the static slice reference and encoding tag — no copy or
/// deobfuscation occurs until [`lookup_aot_cubin`] finds an exact match.
pub fn register_aot_cubin(
    module_name: &'static str,
    ptx_len: usize,
    ptx_sha256: [u8; 32],
    cubin_bytes: &'static [u8],
    encoding: AotCubinEncoding,
) {
    let key = AotCubinKey::from_digest(module_name, ptx_len, ptx_sha256);
    let entry = AotCubinEntry {
        bytes: cubin_bytes,
        encoding,
    };
    let mut map = aot_cubin_store()
        .lock()
        .expect("AOT CUBIN registry lock poisoned");
    map.insert(key, entry);
}

/// Look up a pre-registered AOT CUBIN matching the exact `(module_name, ptx_src)`
/// content digest.
///
/// Returns [`Cow::Borrowed`] for raw CUBINs (zero-copy) and [`Cow::Owned`] for
/// obfuscated CUBINs (decoded on demand into binary bytes). Returns `None` if no
/// CUBIN matches that exact PTX content.
#[must_use]
pub fn lookup_aot_cubin(module_name: &str, ptx_src: &str) -> Option<Cow<'static, [u8]>> {
    let key = AotCubinKey::from_ptx(module_name, ptx_src);
    let entry = {
        let map = aot_cubin_store()
            .lock()
            .expect("AOT CUBIN registry lock poisoned");
        map.get(&key).copied()
    }?;
    match entry.encoding {
        AotCubinEncoding::Raw => Some(Cow::Borrowed(entry.bytes)),
        AotCubinEncoding::Obfuscated => Some(Cow::Owned(decode_obf_bytes(entry.bytes))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_target_display() {
        let t = KernelTarget::GB10_QWEN3_NVFP4;
        assert_eq!(t.to_string(), "(sm_121, qwen3-next-80b-a3b, nvfp4)");
    }

    #[test]
    fn targets_are_comparable() {
        let a = KernelTarget::GB10_QWEN3_NVFP4;
        let b = KernelTarget {
            arch: "sm_100a",
            model: "llama-70b",
            quant: "fp8",
        };
        assert_ne!(a, b);
        assert_eq!(a, KernelTarget::GB10_QWEN3_NVFP4);
    }

    #[test]
    fn decode_obf_bytes_is_binary_safe_and_self_inverse() {
        let elf_with_nuls: &[u8] = &[
            0x7f, b'E', b'L', b'F', 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x00,
            0xfe, 0x00,
        ];
        let encoded = decode_obf_bytes(elf_with_nuls);
        assert_ne!(encoded.as_slice(), elf_with_nuls);
        assert_eq!(decode_obf_bytes(&encoded).as_slice(), elf_with_nuls);
        assert!(decode_obf_bytes(&[]).is_empty());
    }

    #[test]
    fn aot_cubin_store_raw_and_obfuscated_lookup() {
        const PTX: &str = ".version 8.8\n.target sm_121f\n.address_size 64\n";
        const RAW_CUBIN: &[u8] = &[0x7f, b'E', b'L', b'F', 0x00, 0x01, 0x00, 0x42];

        let sha = sha256_bytes(PTX.as_bytes());
        register_aot_cubin(
            "__test_target_raw",
            PTX.len(),
            sha,
            RAW_CUBIN,
            AotCubinEncoding::Raw,
        );

        let hit = lookup_aot_cubin("__test_target_raw", PTX).expect("expected raw CUBIN hit");
        assert!(matches!(hit, Cow::Borrowed(_)));
        assert_eq!(hit.as_ref(), RAW_CUBIN);

        // Mutating even a single byte of PTX must miss.
        let mutated_ptx = ".version 8.8\n.target sm_121a\n.address_size 64\n";
        assert!(lookup_aot_cubin("__test_target_raw", mutated_ptx).is_none());
        // Different module name with identical PTX must miss.
        assert!(lookup_aot_cubin("__test_target_other", PTX).is_none());

        // Obfuscated binary payload with embedded 0x00 bytes must round-trip cleanly.
        let obf_bytes: &'static [u8] = Box::leak(decode_obf_bytes(RAW_CUBIN).into_boxed_slice());
        register_aot_cubin(
            "__test_target_obf",
            PTX.len(),
            sha,
            obf_bytes,
            AotCubinEncoding::Obfuscated,
        );
        let obf_hit =
            lookup_aot_cubin("__test_target_obf", PTX).expect("expected obfuscated CUBIN hit");
        assert!(matches!(obf_hit, Cow::Owned(_)));
        assert_eq!(obf_hit.as_ref(), RAW_CUBIN);
    }
}
