// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn header() -> SeqStateHeader {
    SeqStateHeader {
        fingerprint: [7u8; DIGEST_BYTES],
        seq_len: 5,
        block_size: 4,
        num_blocks: 2,
        flags: flags::QSA_REQUIRED | flags::MTP,
        prompt_len: 3,
        mtp_num_blocks: 2,
        mtp_hidden_selected_row: 0,
        mtp_seq_len: 5,
        session_hash: 0xfeed,
        mtp_last_num_drafted: 2,
        mtp_hidden_position: 0,
        layout_digest: [0u8; DIGEST_BYTES],
        tokens: vec![10, 11, 12, 13, 14],
        mtp_last_drafts: vec![20, 21],
    }
}

fn layout() -> Vec<SectionSpec> {
    vec![
        SectionSpec {
            kind: SectionKind::KvK,
            index: 0,
            bytes: 6,
        },
        SectionSpec {
            kind: SectionKind::GdnH,
            index: 3,
            bytes: 0,
        },
        SectionSpec {
            kind: SectionKind::PleLive,
            index: 0,
            bytes: 3,
        },
    ]
}

fn payloads() -> Vec<Vec<u8>> {
    vec![vec![1, 2, 3, 4, 5, 6], vec![], vec![9, 8, 7]]
}

fn encode(h: &SeqStateHeader, layout: Vec<SectionSpec>, data: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut w = BlobWriter::begin(&mut out, h, layout.clone()).unwrap();
    for (spec, bytes) in layout.iter().zip(data) {
        w.begin_section(spec.kind, spec.index).unwrap();
        // Split writes exercise the streaming digest.
        let (a, b) = bytes.split_at(bytes.len() / 2);
        w.write(a).unwrap();
        w.write(b).unwrap();
        w.end_section().unwrap();
    }
    w.finish().unwrap();
    out
}

fn decode(blob: &[u8], layout: Vec<SectionSpec>) -> Result<(SeqStateHeader, Vec<Vec<u8>>)> {
    let mut r: &[u8] = blob;
    let mut reader = BlobReader::open(&mut r)?;
    let header = reader.header().clone();
    reader.expect_layout(layout.clone())?;
    let mut data = Vec::new();
    for spec in &layout {
        let n = reader.begin_section(spec.kind, spec.index)?;
        let mut buf = vec![0u8; n as usize];
        reader.read(&mut buf)?;
        reader.end_section()?;
        data.push(buf);
    }
    reader.finish()?;
    Ok((header, data))
}

#[test]
fn round_trip_is_exact() {
    let blob = encode(&header(), layout(), &payloads());
    let (h, data) = decode(&blob, layout()).unwrap();
    let mut expected = header();
    expected.layout_digest = layout_digest(&layout());
    assert_eq!(h, expected);
    assert_eq!(data, payloads());
}

#[test]
fn every_flipped_payload_byte_is_detected() {
    let blob = encode(&header(), layout(), &payloads());
    let header_end = blob.len() - 8 - (6 + 3) - 3 * (12 + DIGEST_BYTES);
    // Flip each byte after the header: section headers, payloads, digests and
    // the end marker must all be covered by some check.
    for i in header_end..blob.len() {
        let mut bad = blob.clone();
        bad[i] ^= 0x01;
        assert!(
            decode(&bad, layout()).is_err(),
            "flip at byte {i} was accepted"
        );
    }
}

#[test]
fn truncation_at_any_point_is_an_error() {
    let blob = encode(&header(), layout(), &payloads());
    for len in 0..blob.len() {
        assert!(
            decode(&blob[..len], layout()).is_err(),
            "prefix of {len} bytes accepted"
        );
    }
}

#[test]
fn geometry_mismatch_is_refused_before_any_payload() {
    let blob = encode(&header(), layout(), &payloads());
    let mut other = layout();
    other[0].bytes = 7;
    let mut r: &[u8] = &blob;
    let mut reader = BlobReader::open(&mut r).unwrap();
    assert!(reader.expect_layout(other).is_err());
    // A refused bind leaves the reader unbound: no section can be read.
    assert!(reader.begin_section(SectionKind::KvK, 0).is_err());
    let mut fewer = layout();
    fewer.pop();
    let mut r2: &[u8] = &blob;
    let mut reader2 = BlobReader::open(&mut r2).unwrap();
    assert!(reader2.expect_layout(fewer).is_err());
}

#[test]
fn writer_enforces_declared_order_and_sizes() {
    let mut out = Vec::new();
    let mut w = BlobWriter::begin(&mut out, &header(), layout()).unwrap();
    assert!(
        w.begin_section(SectionKind::GdnH, 3).is_err(),
        "out-of-order section"
    );
    w.begin_section(SectionKind::KvK, 0).unwrap();
    assert!(w.write(&[0u8; 7]).is_err(), "overflowing write");
    w.write(&[0u8; 5]).unwrap();
    assert!(w.end_section().is_err(), "short section");

    let mut out2 = Vec::new();
    let mut w2 = BlobWriter::begin(&mut out2, &header(), layout()).unwrap();
    w2.begin_section(SectionKind::KvK, 0).unwrap();
    w2.write(&[0u8; 6]).unwrap();
    w2.end_section().unwrap();
    assert!(w2.finish().is_err(), "missing sections");
}

#[test]
fn header_validation_rejects_inconsistent_fields() {
    let mut h = header();
    h.seq_len = 4;
    assert!(h.encode().is_err(), "seq_len != tokens");
    let mut h = header();
    h.prompt_len = 6;
    assert!(h.encode().is_err(), "prompt past seq_len");
    let mut h = header();
    h.num_blocks = 1;
    assert!(h.encode().is_err(), "blocks cannot hold positions");
    let mut h = header();
    h.flags = 0;
    assert!(h.encode().is_err(), "MTP fields without MTP flag");
    let mut h = header();
    h.flags |= 1 << 31;
    assert!(h.encode().is_err(), "unknown flag");
}

#[test]
fn wrong_magic_and_version_are_refused() {
    let blob = encode(&header(), layout(), &payloads());
    let mut bad = blob.clone();
    bad[0] = b'X';
    assert!(decode(&bad, layout()).is_err());
    let mut bad = blob;
    bad[8] = 2;
    assert!(decode(&bad, layout()).is_err());
}

#[test]
fn layout_digest_covers_kind_index_and_size() {
    let base = layout_digest(&layout());
    let mut a = layout();
    a[1].index = 4;
    let mut b = layout();
    b[2].kind = SectionKind::PleCheckpoint;
    let mut c = layout();
    c.swap(0, 2);
    for other in [a, b, c] {
        assert_ne!(layout_digest(&other), base);
    }
}
