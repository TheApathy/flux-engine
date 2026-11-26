// SPDX-License-Identifier: AGPL-3.0-only

//! On-disk store format.
//!
//! One file, four sections, all little-endian and 16-byte aligned so the
//! whole thing can be `mmap`ed and the token/SA arrays reinterpreted in
//! place with no copy and no per-element decode:
//!
//! ```text
//! 0                       HEADER_BYTES (128)
//! ├── header ─────────────┤
//!                         ├── tokens: [u32; n_tokens] ──┤
//!                                                       ├── sa: [u32; n_tokens] ──┤
//!                                                                                 ├── doc_starts: [u64; n_docs + 1] ──┤
//! ```
//!
//! `doc_starts` is a prefix-offset array (`n_docs + 1` entries, last entry
//! == `n_tokens`) so a document's token range is `doc_starts[i]..doc_starts[i+1]`
//! and the owning document of a match position is one binary search away.

use anyhow::{Result, bail};

/// File magic. Changing the meaning of any section must bump
/// [`CACHE_FORMAT_VERSION`], not this.
pub const MAGIC: [u8; 8] = *b"ATLRESTS";

/// Bumped whenever the on-disk layout or the semantics of a section
/// change. A store whose version differs is rejected, never reinterpreted.
pub const CACHE_FORMAT_VERSION: u32 = 1;

/// Fixed header size in bytes (sections align to [`ALIGN`] = 16 bytes).
pub const HEADER_BYTES: usize = 128;

/// Section alignment, in bytes.
pub const ALIGN: usize = 16;

/// Round `n` up to the next [`ALIGN`] boundary, returning `None` on overflow.
pub const fn align_up_checked(n: usize) -> Option<usize> {
    n.checked_next_multiple_of(ALIGN)
}

/// Round `n` up to the next [`ALIGN`] boundary.
///
/// # Panics
/// Panics if rounding up overflows `usize`.
pub const fn align_up(n: usize) -> usize {
    match align_up_checked(n) {
        Some(v) => v,
        None => panic!("REST store align_up overflow"),
    }
}

/// Parsed store header.
///
/// Field order matches the on-disk encoding; see [`StoreHeader::encode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreHeader {
    /// Format version the file was written with.
    pub format_version: u32,
    /// Fingerprint of the `tokenizer.json` used to build the store.
    /// A store built with a different tokenizer produces token ids that
    /// mean nothing to the target model, so this is a hard gate.
    pub tokenizer_fp: u64,
    /// Number of tokens in the concatenated corpus stream.
    pub n_tokens: u64,
    /// Number of source documents.
    pub n_docs: u64,
    /// Separator token id inserted between documents. Continuations never
    /// cross it.
    pub sep_token: u32,
    /// Unix seconds at build time. Informational only.
    pub built_unix: u64,
}

impl StoreHeader {
    /// Byte offset of the token array.
    pub const fn tokens_offset(&self) -> usize {
        HEADER_BYTES
    }

    /// Byte offset of the suffix array, or `None` on arithmetic overflow.
    pub fn sa_offset_checked(&self) -> Option<usize> {
        let tokens_bytes = usize::try_from(self.n_tokens)
            .ok()?
            .checked_mul(std::mem::size_of::<u32>())?;
        align_up_checked(HEADER_BYTES.checked_add(tokens_bytes)?)
    }

    /// Byte offset of the suffix array.
    pub fn sa_offset(&self) -> usize {
        self.sa_offset_checked()
            .expect("REST store suffix-array offset overflow")
    }

    /// Byte offset of the document prefix-offset array, or `None` on arithmetic overflow.
    pub fn docs_offset_checked(&self) -> Option<usize> {
        let sa_bytes = usize::try_from(self.n_tokens)
            .ok()?
            .checked_mul(std::mem::size_of::<u32>())?;
        align_up_checked(self.sa_offset_checked()?.checked_add(sa_bytes)?)
    }

    /// Byte offset of the document prefix-offset array.
    pub fn docs_offset(&self) -> usize {
        self.docs_offset_checked()
            .expect("REST store docs offset overflow")
    }

    /// Total file size implied by this header, or `None` if layout arithmetic
    /// overflows `usize` or exceeds `isize::MAX` (the maximum valid Rust slice byte length).
    pub fn total_bytes_checked(&self) -> Option<usize> {
        let docs_len = usize::try_from(self.n_docs).ok()?.checked_add(1)?;
        let docs_bytes = docs_len.checked_mul(std::mem::size_of::<u64>())?;
        let total = align_up_checked(self.docs_offset_checked()?.checked_add(docs_bytes)?)?;
        (total <= isize::MAX as usize).then_some(total)
    }

    /// Total file size implied by this header.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes_checked()
            .expect("REST store total byte size overflow")
    }

    /// Serialize to the fixed-size on-disk header block.
    pub fn encode(&self) -> [u8; HEADER_BYTES] {
        let mut buf = [0u8; HEADER_BYTES];
        buf[0..8].copy_from_slice(&MAGIC);
        buf[8..12].copy_from_slice(&self.format_version.to_le_bytes());
        // [12..16) reserved flags, must stay zero for version 1.
        buf[16..24].copy_from_slice(&self.tokenizer_fp.to_le_bytes());
        buf[24..32].copy_from_slice(&self.n_tokens.to_le_bytes());
        buf[32..40].copy_from_slice(&self.n_docs.to_le_bytes());
        buf[40..44].copy_from_slice(&self.sep_token.to_le_bytes());
        buf[48..56].copy_from_slice(&(self.tokens_offset() as u64).to_le_bytes());
        buf[56..64].copy_from_slice(&(self.sa_offset() as u64).to_le_bytes());
        buf[64..72].copy_from_slice(&(self.docs_offset() as u64).to_le_bytes());
        buf[72..80].copy_from_slice(&self.built_unix.to_le_bytes());
        buf
    }

    /// Parse and validate a header from the first [`HEADER_BYTES`] of a file.
    ///
    /// Rejects a wrong magic, a wrong format version, and any header whose
    /// self-declared section offsets disagree with the ones derived from
    /// `n_tokens`/`n_docs` — the latter guards against a truncated or
    /// hand-edited file being mmap'ed and read out of bounds.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_BYTES {
            bail!(
                "REST store too small for a header: {} bytes < {HEADER_BYTES}",
                buf.len()
            );
        }
        if buf[0..8] != MAGIC {
            bail!("REST store magic mismatch — not a rest-store file");
        }
        let format_version = u32::from_le_bytes(buf[8..12].try_into()?);
        if format_version != CACHE_FORMAT_VERSION {
            bail!(
                "REST store format version {format_version} != expected \
                 {CACHE_FORMAT_VERSION}; rebuild the store"
            );
        }
        let hdr = Self {
            format_version,
            tokenizer_fp: u64::from_le_bytes(buf[16..24].try_into()?),
            n_tokens: u64::from_le_bytes(buf[24..32].try_into()?),
            n_docs: u64::from_le_bytes(buf[32..40].try_into()?),
            sep_token: u32::from_le_bytes(buf[40..44].try_into()?),
            built_unix: u64::from_le_bytes(buf[72..80].try_into()?),
        };
        // The suffix array indexes tokens with u32.
        if hdr.n_tokens > u32::MAX as u64 {
            bail!(
                "REST store has {} tokens, exceeding the u32 suffix-array index limit",
                hdr.n_tokens
            );
        }
        let sa_off = hdr.sa_offset_checked().ok_or_else(|| {
            anyhow::anyhow!(
                "REST store suffix-array offset overflow for n_tokens={}",
                hdr.n_tokens
            )
        })?;
        let docs_off = hdr.docs_offset_checked().ok_or_else(|| {
            anyhow::anyhow!(
                "REST store docs offset overflow for n_tokens={}",
                hdr.n_tokens
            )
        })?;
        if hdr.total_bytes_checked().is_none() {
            bail!(
                "REST store size overflow for n_tokens={} n_docs={}",
                hdr.n_tokens,
                hdr.n_docs
            );
        }
        let declared = (
            u64::from_le_bytes(buf[48..56].try_into()?),
            u64::from_le_bytes(buf[56..64].try_into()?),
            u64::from_le_bytes(buf[64..72].try_into()?),
        );
        let derived = (hdr.tokens_offset() as u64, sa_off as u64, docs_off as u64);
        if declared != derived {
            bail!(
                "REST store section offsets {declared:?} disagree with offsets derived \
                 from n_tokens={} n_docs={} ({derived:?}); file is corrupt",
                hdr.n_tokens,
                hdr.n_docs
            );
        }
        Ok(hdr)
    }
}

/// Fingerprint of a tokenizer file's raw bytes (FNV-1a, 64-bit).
///
/// This detects "you rebuilt the store with a different tokenizer", which
/// is an accident, not an attack — so a non-cryptographic hash is the
/// right tool. It is deliberately computed over the file bytes rather than
/// over a parsed vocab so that merges, added tokens, and normalizer
/// changes all move the fingerprint.
pub fn tokenizer_fingerprint(tokenizer_json: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &b in tokenizer_json {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> StoreHeader {
        StoreHeader {
            format_version: CACHE_FORMAT_VERSION,
            tokenizer_fp: 0xdead_beef_1234_5678,
            n_tokens: 1000,
            n_docs: 7,
            sep_token: 0,
            built_unix: 1_700_000_000,
        }
    }

    #[test]
    fn header_roundtrips() {
        let h = sample();
        assert_eq!(StoreHeader::decode(&h.encode()).unwrap(), h);
    }

    #[test]
    fn sections_are_aligned_and_ordered() {
        let h = sample();
        for off in [h.tokens_offset(), h.sa_offset(), h.docs_offset()] {
            assert_eq!(off % ALIGN, 0, "section offset {off} not {ALIGN}-aligned");
        }
        assert!(h.tokens_offset() < h.sa_offset());
        assert!(h.sa_offset() < h.docs_offset());
        assert!(h.docs_offset() < h.total_bytes());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut buf = sample().encode();
        buf[0] = b'X';
        assert!(StoreHeader::decode(&buf).is_err());
    }

    #[test]
    fn rejects_version_mismatch() {
        let mut buf = sample().encode();
        buf[8..12].copy_from_slice(&(CACHE_FORMAT_VERSION + 1).to_le_bytes());
        assert!(StoreHeader::decode(&buf).is_err());
    }

    #[test]
    fn rejects_inconsistent_offsets() {
        // Claim more tokens than the offsets were computed for.
        let mut buf = sample().encode();
        buf[24..32].copy_from_slice(&2000u64.to_le_bytes());
        assert!(StoreHeader::decode(&buf).is_err());
    }

    #[test]
    fn rejects_n_docs_overflow() {
        for bad_n_docs in [u64::MAX, (usize::MAX / 8) as u64, (isize::MAX as u64) / 8] {
            let mut buf = StoreHeader {
                n_tokens: 0,
                n_docs: 0,
                ..sample()
            }
            .encode();
            buf[32..40].copy_from_slice(&bad_n_docs.to_le_bytes());
            assert!(
                StoreHeader::decode(&buf).is_err(),
                "must reject crafted n_docs={bad_n_docs:#x}"
            );
        }
    }

    #[test]
    fn fingerprint_is_sensitive_to_content() {
        assert_ne!(tokenizer_fingerprint(b"{}"), tokenizer_fingerprint(b"{ }"));
        assert_eq!(tokenizer_fingerprint(b"abc"), tokenizer_fingerprint(b"abc"));
    }
}
