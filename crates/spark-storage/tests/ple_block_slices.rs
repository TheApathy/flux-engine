// SPDX-License-Identifier: AGPL-3.0-only

//! The lock-yielding PLE prefill read (`ATLAS_QWEN4_PLE_QUEUE_PREFETCH`) reads
//! a chunk as one `read_rows_streamed_blocks` call per 1,024-token block
//! instead of one call for the whole chunk. Both must return byte-identical
//! rows, in selection order, for the same token list, in every reader mode,
//! including duplicated rows (the planner's page dedup) and the completion
//! reordering of several io_uring rings / pread lanes. Every row is also
//! checked against the fixture's known bytes.

#![cfg(target_os = "linux")]

use spark_storage::ple_offload::{PleNvfp4Row, PleOffloadReader};
use std::path::{Path, PathBuf};
use std::time::Duration;

const PAGE: usize = 8192;
const RECORD: usize = 90;
const PER_PAGE: usize = 91;
const SHARDS: usize = 4;
const ROWS: usize = 91 * 64 + 17; // a partial last page
const HEADS: usize = 16;
const BLOCK_ROWS: usize = 64 * HEADS; // small blocks: many slices per chunk

fn record_bytes(shard: usize, row: usize) -> [u8; RECORD] {
    let mut out = [0u8; RECORD];
    let mut state = (shard as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ row as u64;
    for byte in &mut out {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = state as u8;
    }
    out
}

fn scale2(shard: usize) -> f32 {
    1.0e-5 * (shard as f32 + 1.0)
}

fn write_fixture(dir: &Path) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let pages = ROWS.div_ceil(PER_PAGE);
    let mut entries = Vec::new();
    for shard in 0..SHARDS {
        let mut bytes = vec![0u8; pages * PAGE];
        for row in 0..ROWS {
            let at = (row / PER_PAGE) * PAGE + (row % PER_PAGE) * RECORD;
            bytes[at..at + RECORD].copy_from_slice(&record_bytes(shard, row));
        }
        let file = format!("ple-ngram-nvfp4-{shard:03}.bin");
        std::fs::write(dir.join(&file), &bytes).unwrap();
        entries.push(serde_json::json!({
            "shard": shard, "file": file, "rows": ROWS, "width": 160,
            "scale2": scale2(shard), "bytes": bytes.len(),
        }));
    }
    let manifest = serde_json::json!({
        "format": "qwen38-flash-next-ple-nvfp4-direct-v1",
        "page_bytes": PAGE, "record_bytes": RECORD, "records_per_page": PER_PAGE,
        "packed_bytes": 80, "scale_bytes": 10, "group_size": 16, "entries": entries,
    });
    let path = dir.join("manifest.json");
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    path
}

/// A prefill-shaped request: 16 rows per token, uniform picks with heavy
/// repetition (repeated n-grams), and runs of the very same row.
fn selections(tokens: usize) -> Vec<(usize, usize)> {
    let mut state = 0x5eed_u64;
    let mut out = Vec::with_capacity(tokens * HEADS);
    for token in 0..tokens {
        for head in 0..HEADS {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let pick = if token % 7 == 3 {
                // A repeated n-gram: the same rows as an earlier token.
                out[(token / 2) * HEADS + head]
            } else if token % 11 == 0 {
                (head % SHARDS, 5) // one hot row, many times
            } else {
                ((state as usize) % SHARDS, (state >> 24) as usize % ROWS)
            };
            out.push(pick);
        }
    }
    out
}

fn read_whole(reader: &mut PleOffloadReader, picks: &[(usize, usize)]) -> Vec<PleNvfp4Row> {
    let mut rows = Vec::with_capacity(picks.len());
    let mut next_block = 0;
    reader
        .read_rows_streamed_blocks(picks, BLOCK_ROWS, &mut |block, block_rows| {
            assert_eq!(block, next_block, "blocks publish in order");
            next_block += 1;
            rows.extend(block_rows);
            Ok(())
        })
        .unwrap();
    rows
}

/// What the yielding read does: one call per block, in block order.
fn read_yielding(reader: &mut PleOffloadReader, picks: &[(usize, usize)]) -> Vec<PleNvfp4Row> {
    let mut rows = Vec::with_capacity(picks.len());
    for slice in picks.chunks(BLOCK_ROWS) {
        let mut calls = 0;
        reader
            .read_rows_streamed_blocks(slice, BLOCK_ROWS, &mut |block, block_rows| {
                assert_eq!(block, 0, "one block per slice");
                calls += 1;
                rows.extend(block_rows);
                Ok(())
            })
            .unwrap();
        assert_eq!(calls, 1);
    }
    rows
}

fn assert_rows(label: &str, picks: &[(usize, usize)], rows: &[PleNvfp4Row]) {
    assert_eq!(rows.len(), picks.len(), "{label}: row count");
    for (i, (&(shard, row), got)) in picks.iter().zip(rows).enumerate() {
        assert_eq!(
            got.record,
            record_bytes(shard, row),
            "{label}: row {i} ({shard}, {row})"
        );
        assert_eq!(
            got.scale2.to_bits(),
            scale2(shard).to_bits(),
            "{label}: scale2 {i}"
        );
    }
}

#[test]
fn yielding_block_slices_match_the_whole_chunk_read_in_every_reader_mode() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("ple-slices-{}", std::process::id()));
    let manifest = write_fixture(&dir);
    let picks = selections(1_000); // 16,000 rows, 16 blocks (the last one short)

    // (FAST_PLAN, SECTOR_READ, READ_RINGS, FORCE_PREAD): legacy, the production
    // planned sector reads on 4 rings, and the no-io_uring pread lanes.
    let modes: [(&str, &[(&str, &str)]); 3] = [
        ("legacy", &[]),
        (
            "planned-4-rings",
            &[
                ("ATLAS_PLE_FAST_PLAN", "1"),
                ("ATLAS_PLE_SECTOR_READ", "1"),
                ("ATLAS_PLE_READ_RINGS", "4"),
            ],
        ),
        (
            "pread-lanes",
            &[
                ("ATLAS_PLE_FAST_PLAN", "1"),
                ("ATLAS_PLE_SECTOR_READ", "1"),
                ("ATLAS_PLE_FORCE_PREAD", "1"),
            ],
        ),
    ];
    let all = [
        "ATLAS_PLE_FAST_PLAN",
        "ATLAS_PLE_SECTOR_READ",
        "ATLAS_PLE_READ_RINGS",
        "ATLAS_PLE_FORCE_PREAD",
    ];
    for (label, env) in modes {
        // SAFETY: this is the only test in this binary, and the reader reads
        // its selectors only inside `open`, below, on this thread.
        unsafe {
            for name in all {
                std::env::remove_var(name);
            }
            for (name, value) in env {
                std::env::set_var(name, value);
            }
        }
        // A fresh reader per read keeps the record cache empty.
        let open = || PleOffloadReader::open(&manifest, 32, 0, Duration::from_secs(30)).unwrap();
        let whole = read_whole(&mut open(), &picks);
        let yielding = read_yielding(&mut open(), &picks);
        assert_rows(&format!("{label} whole"), &picks, &whole);
        assert_rows(&format!("{label} yielding"), &picks, &yielding);
        // Byte-identical to each other, row for row.
        assert!(
            whole
                .iter()
                .zip(&yielding)
                .all(|(a, b)| a.record == b.record && a.scale2.to_bits() == b.scale2.to_bits()),
            "{label}: yielding read differs from the whole-chunk read"
        );
        // The same reader, warm (record cache / reused rings), still agrees.
        let mut reader = open();
        let first = read_whole(&mut reader, &picks);
        let second = read_yielding(&mut reader, &picks);
        assert_rows(&format!("{label} warm whole"), &picks, &first);
        assert_rows(&format!("{label} warm yielding"), &picks, &second);
    }
    std::fs::remove_dir_all(&dir).ok();
}
