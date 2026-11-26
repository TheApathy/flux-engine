// SPDX-License-Identifier: AGPL-3.0-only

//! Byte layout of one sequence's exact resumable state (`SeqStateBlob`).
//!
//! ```text
//! magic "FLXSEQS1" | u32 version | u32 header_len | header | u32 section_count
//! repeat section_count: u16 kind | u16 index | u64 bytes | payload | sha256(payload)
//! end magic "FLXSEQND"
//! ```
//!
//! Every integer is little-endian. The header carries the digest of the
//! ordered `(kind, index, bytes)` section list, so a reader can refuse a blob
//! whose geometry differs from its destination before any device byte is
//! written. Payloads are raw device bytes exactly as stored (KV in its cache
//! dtype, recurrent state in its slot representation), never re-encoded, so a
//! round trip is byte-identical by construction.

use std::io::{Read, Write};

use anyhow::{Context, Result, bail, ensure};
use ring::digest::{Context as DigestContext, SHA256};

pub(crate) const MAGIC: [u8; 8] = *b"FLXSEQS1";
pub(crate) const END_MAGIC: [u8; 8] = *b"FLXSEQND";
pub(crate) const VERSION: u32 = 1;
pub(crate) const DIGEST_BYTES: usize = 32;
/// Header bytes beyond the token list.
const HEADER_FIXED_BYTES: usize = 32 + 8 + 4 + 4 + 4 + 8 + 4 + 4 + 8 + 8 + 4 + 8 + 32;
/// Upper bound on tokens one blob may describe (16M positions).
pub(crate) const MAX_TOKENS: usize = 1 << 24;
pub(crate) const MAX_SECTIONS: usize = 4096;

/// What a section holds. The numeric values are part of the format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum SectionKind {
    /// Target K cache, one section per attention layer: every block of the
    /// sequence's block table in logical order, each block's full stride.
    KvK = 1,
    KvV = 2,
    /// QSA raw index-key ring rows of the same blocks, per attention layer.
    QsaRaw = 3,
    /// QSA compressed group keys of the same blocks, per attention layer.
    QsaCompressed = 4,
    /// GDN recurrent state of one linear-attention layer, slot bytes as stored.
    GdnH = 5,
    GdnConv = 6,
    /// PLE convolution history (live and speculative checkpoint).
    PleLive = 7,
    PleCheckpoint = 8,
    /// Native-MTP head K/V cache blocks, in its own block-table order.
    MtpKvK = 9,
    MtpKvV = 10,
    /// Retained native-MTP target-hidden prefix of the sequence's slot.
    MtpHidden = 11,
}

impl SectionKind {
    pub(crate) fn from_u16(v: u16) -> Result<Self> {
        Ok(match v {
            1 => Self::KvK,
            2 => Self::KvV,
            3 => Self::QsaRaw,
            4 => Self::QsaCompressed,
            5 => Self::GdnH,
            6 => Self::GdnConv,
            7 => Self::PleLive,
            8 => Self::PleCheckpoint,
            9 => Self::MtpKvK,
            10 => Self::MtpKvV,
            11 => Self::MtpHidden,
            _ => bail!("seqstate: unknown section kind {v}"),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SectionSpec {
    pub kind: SectionKind,
    pub index: u16,
    pub bytes: u64,
}

/// Digest of the ordered section list: the blob's geometry identity.
pub(crate) fn layout_digest(layout: &[SectionSpec]) -> [u8; DIGEST_BYTES] {
    let mut ctx = DigestContext::new(&SHA256);
    ctx.update(&(layout.len() as u64).to_le_bytes());
    for spec in layout {
        ctx.update(&(spec.kind as u16).to_le_bytes());
        ctx.update(&spec.index.to_le_bytes());
        ctx.update(&spec.bytes.to_le_bytes());
    }
    to_array(ctx.finish().as_ref())
}

fn to_array(bytes: &[u8]) -> [u8; DIGEST_BYTES] {
    let mut out = [0u8; DIGEST_BYTES];
    out.copy_from_slice(bytes);
    out
}

pub(crate) mod flags {
    /// The sequence maintained the QSA side cache (`qwen4_qsa_required`).
    pub const QSA_REQUIRED: u32 = 1 << 0;
    /// GDN h-state slots hold the FP16 representation (`h_is_f16`).
    pub const H_FP16: u32 = 1 << 1;
    /// Native-MTP proposer state is present.
    pub const MTP: u32 = 1 << 2;
    /// A retained native-MTP target-hidden frame is present.
    pub const MTP_HIDDEN: u32 = 1 << 3;
    pub const KNOWN: u32 = QSA_REQUIRED | H_FP16 | MTP | MTP_HIDDEN;
}

/// Everything about the sequence that is not raw device state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SeqStateHeader {
    /// Model / cache / numerics identity; a blob restores only into a model
    /// with the same fingerprint.
    pub fingerprint: [u8; DIGEST_BYTES],
    pub seq_len: u64,
    pub block_size: u32,
    pub num_blocks: u32,
    pub flags: u32,
    pub prompt_len: u64,
    pub mtp_num_blocks: u32,
    pub mtp_hidden_selected_row: u32,
    pub mtp_seq_len: u64,
    pub session_hash: u64,
    pub mtp_last_num_drafted: u32,
    /// Decode position the retained MTP target-hidden frame is bound to.
    pub mtp_hidden_position: u64,
    pub layout_digest: [u8; DIGEST_BYTES],
    pub tokens: Vec<u32>,
    pub mtp_last_drafts: Vec<u32>,
}

impl SeqStateHeader {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.flags & !flags::KNOWN == 0,
            "seqstate: unknown header flags {:#x}",
            self.flags & !flags::KNOWN
        );
        ensure!(
            self.tokens.len() <= MAX_TOKENS,
            "seqstate: {} tokens exceed the {MAX_TOKENS} limit",
            self.tokens.len()
        );
        ensure!(
            self.seq_len as usize == self.tokens.len(),
            "seqstate: seq_len {} does not match {} tokens",
            self.seq_len,
            self.tokens.len()
        );
        ensure!(
            self.prompt_len <= self.seq_len,
            "seqstate: prompt_len {} exceeds seq_len {}",
            self.prompt_len,
            self.seq_len
        );
        ensure!(self.block_size > 0, "seqstate: zero block size");
        let needed = self.seq_len.div_ceil(u64::from(self.block_size));
        ensure!(
            u64::from(self.num_blocks) >= needed,
            "seqstate: {} blocks cannot hold {} positions at block size {}",
            self.num_blocks,
            self.seq_len,
            self.block_size
        );
        let mtp = self.flags & flags::MTP != 0;
        ensure!(
            mtp || (self.mtp_num_blocks == 0
                && self.mtp_seq_len == 0
                && self.mtp_last_num_drafted == 0
                && self.mtp_last_drafts.is_empty()
                && self.flags & flags::MTP_HIDDEN == 0),
            "seqstate: MTP fields set without the MTP flag"
        );
        ensure!(
            self.flags & flags::MTP_HIDDEN != 0
                || (self.mtp_hidden_position == 0 && self.mtp_hidden_selected_row == 0),
            "seqstate: MTP hidden fields set without the MTP_HIDDEN flag"
        );
        ensure!(
            self.mtp_last_drafts.len() <= MAX_TOKENS,
            "seqstate: oversized MTP draft list"
        );
        Ok(())
    }

    fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut out = Vec::with_capacity(
            HEADER_FIXED_BYTES + 8 + 4 * (self.tokens.len() + self.mtp_last_drafts.len()),
        );
        out.extend_from_slice(&self.fingerprint);
        out.extend_from_slice(&self.seq_len.to_le_bytes());
        out.extend_from_slice(&self.block_size.to_le_bytes());
        out.extend_from_slice(&self.num_blocks.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&self.prompt_len.to_le_bytes());
        out.extend_from_slice(&self.mtp_num_blocks.to_le_bytes());
        out.extend_from_slice(&self.mtp_hidden_selected_row.to_le_bytes());
        out.extend_from_slice(&self.mtp_seq_len.to_le_bytes());
        out.extend_from_slice(&self.session_hash.to_le_bytes());
        out.extend_from_slice(&self.mtp_last_num_drafted.to_le_bytes());
        out.extend_from_slice(&self.mtp_hidden_position.to_le_bytes());
        out.extend_from_slice(&self.layout_digest);
        out.extend_from_slice(&(self.tokens.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.mtp_last_drafts.len() as u32).to_le_bytes());
        for t in self.tokens.iter().chain(&self.mtp_last_drafts) {
            out.extend_from_slice(&t.to_le_bytes());
        }
        Ok(out)
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let mut cur = Cursor { bytes, pos: 0 };
        let fingerprint = cur.array()?;
        let seq_len = cur.u64()?;
        let block_size = cur.u32()?;
        let num_blocks = cur.u32()?;
        let flags = cur.u32()?;
        let prompt_len = cur.u64()?;
        let mtp_num_blocks = cur.u32()?;
        let mtp_hidden_selected_row = cur.u32()?;
        let mtp_seq_len = cur.u64()?;
        let session_hash = cur.u64()?;
        let mtp_last_num_drafted = cur.u32()?;
        let mtp_hidden_position = cur.u64()?;
        let layout_digest = cur.array()?;
        let n_tokens = cur.u32()? as usize;
        let n_drafts = cur.u32()? as usize;
        ensure!(
            n_tokens <= MAX_TOKENS && n_drafts <= MAX_TOKENS,
            "seqstate: header token counts out of range"
        );
        ensure!(
            cur.remaining() == 4 * (n_tokens + n_drafts),
            "seqstate: header length does not match its token counts"
        );
        let tokens = (0..n_tokens)
            .map(|_| cur.u32())
            .collect::<Result<Vec<_>>>()?;
        let mtp_last_drafts = (0..n_drafts)
            .map(|_| cur.u32())
            .collect::<Result<Vec<_>>>()?;
        let header = Self {
            fingerprint,
            seq_len,
            block_size,
            num_blocks,
            flags,
            prompt_len,
            mtp_num_blocks,
            mtp_hidden_selected_row,
            mtp_seq_len,
            session_hash,
            mtp_last_num_drafted,
            mtp_hidden_position,
            layout_digest,
            tokens,
            mtp_last_drafts,
        };
        header.validate()?;
        Ok(header)
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|end| *end <= self.bytes.len())
            .context("seqstate: header truncated")?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn array(&mut self) -> Result<[u8; DIGEST_BYTES]> {
        Ok(to_array(self.take(DIGEST_BYTES)?))
    }
    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }
}

fn read_u16(r: &mut dyn Read) -> Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).context("seqstate: truncated")?;
    Ok(u16::from_le_bytes(b))
}
fn read_u32(r: &mut dyn Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).context("seqstate: truncated")?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(r: &mut dyn Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).context("seqstate: truncated")?;
    Ok(u64::from_le_bytes(b))
}

/// Streams one blob. The section order and sizes are fixed by the layout
/// passed to [`BlobWriter::begin`]; any deviation is an error.
pub(crate) struct BlobWriter<'w> {
    w: &'w mut dyn Write,
    layout: Vec<SectionSpec>,
    next: usize,
    open: Option<OpenSection>,
}

struct OpenSection {
    spec: SectionSpec,
    written: u64,
    digest: DigestContext,
}

impl<'w> BlobWriter<'w> {
    /// `header.layout_digest` is overwritten with the digest of `layout`.
    pub(crate) fn begin(
        w: &'w mut dyn Write,
        header: &SeqStateHeader,
        layout: Vec<SectionSpec>,
    ) -> Result<Self> {
        ensure!(
            layout.len() <= MAX_SECTIONS,
            "seqstate: {} sections exceed the {MAX_SECTIONS} limit",
            layout.len()
        );
        let mut header = header.clone();
        header.layout_digest = layout_digest(&layout);
        let encoded = header.encode()?;
        w.write_all(&MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&u32::try_from(encoded.len())?.to_le_bytes())?;
        w.write_all(&encoded)?;
        w.write_all(&u32::try_from(layout.len())?.to_le_bytes())?;
        Ok(Self {
            w,
            layout,
            next: 0,
            open: None,
        })
    }

    pub(crate) fn begin_section(&mut self, kind: SectionKind, index: u16) -> Result<()> {
        ensure!(self.open.is_none(), "seqstate: previous section still open");
        let spec = *self
            .layout
            .get(self.next)
            .context("seqstate: more sections written than the layout declares")?;
        ensure!(
            spec.kind == kind && spec.index == index,
            "seqstate: section {:?}/{} written where the layout declares {:?}/{}",
            kind,
            index,
            spec.kind,
            spec.index
        );
        self.w.write_all(&(kind as u16).to_le_bytes())?;
        self.w.write_all(&index.to_le_bytes())?;
        self.w.write_all(&spec.bytes.to_le_bytes())?;
        self.open = Some(OpenSection {
            spec,
            written: 0,
            digest: DigestContext::new(&SHA256),
        });
        Ok(())
    }

    pub(crate) fn write(&mut self, bytes: &[u8]) -> Result<()> {
        let open = self
            .open
            .as_mut()
            .context("seqstate: write outside a section")?;
        let after = open.written + bytes.len() as u64;
        ensure!(
            after <= open.spec.bytes,
            "seqstate: section {:?}/{} overflows its {} bytes",
            open.spec.kind,
            open.spec.index,
            open.spec.bytes
        );
        open.digest.update(bytes);
        open.written = after;
        self.w.write_all(bytes)?;
        Ok(())
    }

    pub(crate) fn end_section(&mut self) -> Result<()> {
        let open = self.open.take().context("seqstate: no open section")?;
        ensure!(
            open.written == open.spec.bytes,
            "seqstate: section {:?}/{} wrote {} of {} bytes",
            open.spec.kind,
            open.spec.index,
            open.written,
            open.spec.bytes
        );
        self.w.write_all(open.digest.finish().as_ref())?;
        self.next += 1;
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<()> {
        ensure!(self.open.is_none(), "seqstate: finish with an open section");
        ensure!(
            self.next == self.layout.len(),
            "seqstate: {} of {} sections written",
            self.next,
            self.layout.len()
        );
        self.w.write_all(&END_MAGIC)?;
        self.w.flush()?;
        Ok(())
    }
}

/// Reads one blob, checking every section against the destination layout.
pub(crate) struct BlobReader<'r> {
    r: &'r mut dyn Read,
    header: SeqStateHeader,
    section_count: usize,
    layout: Option<Vec<SectionSpec>>,
    next: usize,
    open: Option<OpenSection>,
}

impl<'r> BlobReader<'r> {
    pub(crate) fn open(r: &'r mut dyn Read) -> Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)
            .context("seqstate: truncated magic")?;
        ensure!(magic == MAGIC, "seqstate: not a sequence-state blob");
        let version = read_u32(r)?;
        ensure!(
            version == VERSION,
            "seqstate: unsupported version {version}"
        );
        let header_len = read_u32(r)? as usize;
        ensure!(
            (HEADER_FIXED_BYTES + 8..=HEADER_FIXED_BYTES + 8 + 8 * MAX_TOKENS)
                .contains(&header_len),
            "seqstate: header length {header_len} out of range"
        );
        let mut raw = vec![0u8; header_len];
        r.read_exact(&mut raw)
            .context("seqstate: truncated header")?;
        let header = SeqStateHeader::decode(&raw)?;
        let section_count = read_u32(r)? as usize;
        ensure!(
            section_count <= MAX_SECTIONS,
            "seqstate: {section_count} sections exceed the {MAX_SECTIONS} limit"
        );
        Ok(Self {
            r,
            header,
            section_count,
            layout: None,
            next: 0,
            open: None,
        })
    }

    pub(crate) fn header(&self) -> &SeqStateHeader {
        &self.header
    }

    /// Binds the destination geometry. Fails, before any payload is read,
    /// unless the blob's declared layout is exactly `layout`.
    pub(crate) fn expect_layout(&mut self, layout: Vec<SectionSpec>) -> Result<()> {
        ensure!(self.layout.is_none(), "seqstate: layout already bound");
        ensure!(
            layout.len() == self.section_count
                && layout_digest(&layout) == self.header.layout_digest,
            "seqstate: blob geometry ({} sections) does not match the destination ({} sections)",
            self.section_count,
            layout.len()
        );
        self.layout = Some(layout);
        Ok(())
    }

    pub(crate) fn begin_section(&mut self, kind: SectionKind, index: u16) -> Result<u64> {
        ensure!(self.open.is_none(), "seqstate: previous section still open");
        let layout = self.layout.as_ref().context("seqstate: layout not bound")?;
        let spec = *layout
            .get(self.next)
            .context("seqstate: more sections read than the layout declares")?;
        ensure!(
            spec.kind == kind && spec.index == index,
            "seqstate: section {:?}/{} requested where the layout declares {:?}/{}",
            kind,
            index,
            spec.kind,
            spec.index
        );
        let got_kind = SectionKind::from_u16(read_u16(self.r)?)?;
        let got_index = read_u16(self.r)?;
        let got_bytes = read_u64(self.r)?;
        ensure!(
            got_kind == spec.kind && got_index == spec.index && got_bytes == spec.bytes,
            "seqstate: section {:?}/{} ({} bytes) found where {:?}/{} ({} bytes) belongs",
            got_kind,
            got_index,
            got_bytes,
            spec.kind,
            spec.index,
            spec.bytes
        );
        self.open = Some(OpenSection {
            spec,
            written: 0,
            digest: DigestContext::new(&SHA256),
        });
        Ok(spec.bytes)
    }

    pub(crate) fn read(&mut self, buf: &mut [u8]) -> Result<()> {
        let open = self
            .open
            .as_mut()
            .context("seqstate: read outside a section")?;
        let after = open.written + buf.len() as u64;
        ensure!(
            after <= open.spec.bytes,
            "seqstate: read past the end of section {:?}/{}",
            open.spec.kind,
            open.spec.index
        );
        self.r
            .read_exact(buf)
            .context("seqstate: truncated section")?;
        open.digest.update(buf);
        open.written = after;
        Ok(())
    }

    pub(crate) fn end_section(&mut self) -> Result<()> {
        let open = self.open.take().context("seqstate: no open section")?;
        ensure!(
            open.written == open.spec.bytes,
            "seqstate: section {:?}/{} read {} of {} bytes",
            open.spec.kind,
            open.spec.index,
            open.written,
            open.spec.bytes
        );
        let mut stored = [0u8; DIGEST_BYTES];
        self.r
            .read_exact(&mut stored)
            .context("seqstate: truncated section digest")?;
        ensure!(
            stored.as_slice() == open.digest.finish().as_ref(),
            "seqstate: section {:?}/{} digest mismatch (corrupt blob)",
            open.spec.kind,
            open.spec.index
        );
        self.next += 1;
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<()> {
        ensure!(self.open.is_none(), "seqstate: finish with an open section");
        ensure!(
            self.next == self.section_count,
            "seqstate: {} of {} sections read",
            self.next,
            self.section_count
        );
        let mut magic = [0u8; 8];
        self.r
            .read_exact(&mut magic)
            .context("seqstate: truncated end marker")?;
        ensure!(magic == END_MAGIC, "seqstate: bad end marker");
        Ok(())
    }
}

#[cfg(test)]
#[path = "format_tests.rs"]
mod tests;
