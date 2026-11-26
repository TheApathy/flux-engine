// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::model::seqstate::format::{BlobReader, DIGEST_BYTES, SeqStateHeader};
use spark_runtime::gpu::mock::MockGpuBackend;

fn header() -> SeqStateHeader {
    SeqStateHeader {
        fingerprint: [1u8; DIGEST_BYTES],
        seq_len: 3,
        block_size: 16,
        num_blocks: 1,
        flags: 0,
        prompt_len: 3,
        mtp_num_blocks: 0,
        mtp_hidden_selected_row: 0,
        mtp_seq_len: 0,
        session_hash: 0,
        mtp_last_num_drafted: 0,
        mtp_hidden_position: 0,
        layout_digest: [0u8; DIGEST_BYTES],
        tokens: vec![1, 2, 3],
        mtp_last_drafts: vec![],
    }
}

fn filled(gpu: &MockGpuBackend, bytes: usize, seed: u8) -> DevicePtr {
    let ptr = gpu.alloc(bytes).unwrap();
    let data: Vec<u8> = (0..bytes)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect();
    gpu.copy_h2d(&data, ptr).unwrap();
    ptr
}

fn read(gpu: &MockGpuBackend, ptr: DevicePtr, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    gpu.copy_d2h(ptr, &mut out).unwrap();
    out
}

#[test]
fn push_merges_contiguous_ranges_only() {
    let mut r = StateRegion::new(SectionKind::KvK, 0);
    r.push(DevicePtr(0x1000), 16);
    r.push(DevicePtr(0x1010), 16);
    r.push(DevicePtr(0x2000), 8);
    r.push(DevicePtr(0x3000), 0);
    assert_eq!(
        r.parts,
        vec![(DevicePtr(0x1000), 32), (DevicePtr(0x2000), 8)]
    );
    assert_eq!(r.bytes(), 40);
}

#[test]
fn overlapping_or_null_ranges_are_refused() {
    let mut a = StateRegion::new(SectionKind::KvK, 0);
    a.push(DevicePtr(0x1000), 32);
    let mut b = StateRegion::new(SectionKind::KvV, 0);
    b.push(DevicePtr(0x1010), 8);
    assert!(check_disjoint(&[a.clone(), b]).is_err());
    let mut c = StateRegion::new(SectionKind::KvV, 0);
    c.push(DevicePtr(0), 8);
    assert!(check_disjoint(&[a, c]).is_err());
}

/// Scattered source blocks restored into differently placed destination
/// blocks come back byte-identical, section by section.
#[test]
fn capture_then_apply_restores_scattered_parts_exactly() {
    let gpu = MockGpuBackend::new();
    let block = 48usize;
    let pool_src = filled(&gpu, block * 8, 3);
    let pool_dst = gpu.alloc(block * 8).unwrap();
    let h_src = filled(&gpu, 100, 9);
    let h_dst = gpu.alloc(100).unwrap();
    let src_table = [5u32, 1, 2, 7];
    let dst_table = [0u32, 3, 4, 6];

    let regions_for = |pool: DevicePtr, table: &[u32], h: DevicePtr| {
        let mut kv = StateRegion::new(SectionKind::KvK, 0);
        for &b in table {
            kv.push(pool.offset(b as usize * block), block);
        }
        let mut gdn = StateRegion::new(SectionKind::GdnH, 1);
        gdn.push(h, 100);
        vec![kv, gdn]
    };

    let src = regions_for(pool_src, &src_table, h_src);
    let dst = regions_for(pool_dst, &dst_table, h_dst);
    let mut blob = Vec::new();
    capture(&gpu, &header(), &src, &mut blob).unwrap();

    let mut r: &[u8] = &blob;
    apply(&gpu, BlobReader::open(&mut r).unwrap(), &dst).unwrap();

    for (s, d) in src_table.iter().zip(dst_table) {
        assert_eq!(
            read(&gpu, pool_src.offset(*s as usize * block), block),
            read(&gpu, pool_dst.offset(d as usize * block), block)
        );
    }
    assert_eq!(read(&gpu, h_src, 100), read(&gpu, h_dst, 100));
    // Blocks outside the destination table are untouched.
    assert_eq!(read(&gpu, pool_dst.offset(block), block), vec![0u8; block]);
}

#[test]
fn geometry_mismatch_writes_nothing() {
    let gpu = MockGpuBackend::new();
    let src_ptr = filled(&gpu, 64, 5);
    let dst_ptr = gpu.alloc(64).unwrap();
    let mut src = StateRegion::new(SectionKind::PleLive, 0);
    src.push(src_ptr, 64);
    let mut blob = Vec::new();
    capture(&gpu, &header(), &[src], &mut blob).unwrap();

    let mut dst = StateRegion::new(SectionKind::PleLive, 0);
    dst.push(dst_ptr, 32);
    let mut r: &[u8] = &blob;
    assert!(apply(&gpu, BlobReader::open(&mut r).unwrap(), &[dst]).is_err());
    assert_eq!(read(&gpu, dst_ptr, 64), vec![0u8; 64]);
}

#[test]
fn large_parts_stream_through_bounded_staging() {
    let gpu = MockGpuBackend::new();
    let bytes = STAGING_BYTES * 2 + 123;
    let src_ptr = filled(&gpu, bytes, 1);
    let dst_ptr = gpu.alloc(bytes).unwrap();
    let mut src = StateRegion::new(SectionKind::GdnH, 0);
    src.push(src_ptr, bytes);
    let mut dst = StateRegion::new(SectionKind::GdnH, 0);
    dst.push(dst_ptr, bytes);
    let mut blob = Vec::new();
    capture(&gpu, &header(), &[src], &mut blob).unwrap();
    let mut r: &[u8] = &blob;
    apply(&gpu, BlobReader::open(&mut r).unwrap(), &[dst]).unwrap();
    assert!(read(&gpu, src_ptr, bytes) == read(&gpu, dst_ptr, bytes));
}
