// SPDX-License-Identifier: AGPL-3.0-only

//! Prepared-width admission (bounded interleave chunks) for the wide
//! geometry, and the unchanged legacy geometry.

use super::super::attn16::{
    admit_surface_for_geometry, begin_fast_lineage_for_geometry, commit_fast_lineage_for_geometry,
};
use super::super::{
    ChunkGeometry, LEGACY_GEOMETRY, WIDE_TOTAL_ROWS, chunk_window_for_geometry,
    exact_chunk_window_for_geometry,
};
use super::MIN_PREPARED_ROWS;
use crate::traits::Qwen4PrefillLineage;

fn wide(rows: usize) -> ChunkGeometry {
    ChunkGeometry {
        rows,
        total: WIDE_TOTAL_ROWS,
        tail: true,
    }
}

#[test]
fn prepared_widths_are_powers_of_two_from_512_up_to_w() {
    assert_eq!(wide(4096).prepared_widths(), [512, 1024, 2048, 4096]);
    assert_eq!(wide(8192).prepared_widths(), [512, 1024, 2048, 4096, 8192]);
    assert_eq!(
        wide(32768).prepared_widths(),
        [512, 1024, 2048, 4096, 8192, 16384, 32768]
    );
    // Legacy prepares only its own width: its admission is unchanged.
    assert_eq!(LEGACY_GEOMETRY.prepared_widths(), [2048]);
    for w in [4096, 8192, 16384, 32768] {
        for b in wide(w).prepared_widths() {
            // Every prepared width meets the 16-row attention/HC tile and the
            // 128-row alignment the interleave scheduler assumes.
            assert!(
                b.is_multiple_of(128) && b.is_multiple_of(16) && b <= w,
                "{b}"
            );
        }
    }
}

#[test]
fn chunk_rows_at_accepts_prepared_widths_at_aligned_offsets() {
    let g = wide(4096);
    let total = 16_384;
    for b in [512, 1024, 2048, 4096] {
        for k in 0..total / b {
            assert_eq!(g.chunk_rows_at(k * b, total, b), Some(b), "B={b} k={k}");
        }
    }
    // A prompt ending inside a prepared chunk: the remainder.
    assert_eq!(g.chunk_rows_at(5632, 6000, 512), Some(368));
    assert_eq!(g.chunk_rows_at(4096, 6000, 4096), Some(1904));
}

#[test]
fn chunk_rows_at_rejects_unprepared_widths_and_misaligned_offsets() {
    let g = wide(4096);
    // Not prepared: below 512, not a power of two, above W.
    for b in [0, 16, 128, 256, 768, 1000, 1536, 3072, 8192] {
        assert_eq!(g.chunk_rows_at(0, 16_384, b), None, "B={b}");
    }
    // Prepared width at an offset that is not a multiple of it.
    for (start, b) in [
        (512, 1024),
        (1024, 2048),
        (2048, 4096),
        (256, 512),
        (5370, 512),
    ] {
        assert_eq!(
            g.chunk_rows_at(start, 16_384, b),
            None,
            "start={start} B={b}"
        );
    }
    // At or past the prompt end.
    assert_eq!(g.chunk_rows_at(6144, 6000, 512), None);
    assert_eq!(g.chunk_rows_at(6000, 6000, 16), None);
}

#[test]
fn legacy_geometry_admission_is_unchanged() {
    let g = LEGACY_GEOMETRY;
    for start in [0, 2048, 4096, 6144] {
        assert_eq!(g.chunk_rows_at(start, 8192, 2048), Some(2048));
        assert!(g.chunk_admissible(start, 2048, 8192));
    }
    for (start, rows) in [(0, 1024), (0, 512), (1024, 1024), (2048, 1024)] {
        assert!(!g.chunk_admissible(start, rows, 8192), "{start}+{rows}");
    }
    assert!(g.chunked_prompt(8192));
    assert!(!g.chunked_prompt(8191) && !g.chunked_prompt(4096));
    assert!(g.window_rows_admitted(2048, 4096));
    assert!(!g.window_rows_admitted(1024, 4096) && !g.window_rows_admitted(2048, 1024));
}

#[test]
fn w_only_chunking_is_still_admitted_exactly_as_before() {
    // No decode row ever active: W chunks, then the remainder.
    let g = wide(4096);
    assert!(g.chunk_admissible(0, 4096, 6000));
    assert!(g.chunk_admissible(4096, 1904, 6000));
    let g = wide(8192);
    for start in (0..65_536).step_by(8192) {
        assert!(g.chunk_admissible(start, 8192, 65_536));
        assert!(g.window_rows_admitted(8192, start));
    }
}

#[test]
fn chunk_admissible_covers_bounded_chunks_and_rejects_the_rest() {
    let g = wide(4096);
    for (start, rows, total) in [
        (0, 512, 6000),
        (512, 512, 6000),
        (1024, 1024, 6000),
        (2048, 2048, 6000),
        (4096, 512, 6000),
        (5632, 368, 6000),
        (2048, 952, 3000),
    ] {
        assert!(
            g.chunk_admissible(start, rows, total),
            "{start}+{rows}/{total}"
        );
    }
    for (start, rows, total) in [
        (512, 1024, 6000),  // 1024 at a 512-only offset
        (0, 1000, 6000),    // not a prepared width and not the remainder
        (1024, 3072, 6000), // not prepared
        (512, 5488, 6000),  // remainder is larger than any width aligned at 512
        (0, 0, 6000),
        (5632, 369, 6000), // past the prompt end
    ] {
        assert!(
            !g.chunk_admissible(start, rows, total),
            "{start}+{rows}/{total}"
        );
    }
}

#[test]
fn chunked_prompt_includes_short_prompts_that_bounded_chunks_split() {
    let g = wide(4096);
    assert!(!g.chunked_prompt(MIN_PREPARED_ROWS));
    assert!(g.chunked_prompt(MIN_PREPARED_ROWS + 1));
    assert!(g.chunked_prompt(3000) && g.chunked_prompt(4096));
    assert!(g.chunked_prompt(6000) && g.lineage_prompt(6000));
    // The historical lineage set is untouched (it drives the alternate-route
    // marking in prefill_a, which must not suppress single-chunk prompts).
    assert!(!g.lineage_prompt(3000) && !g.lineage_prompt(4096));
    assert!(!g.chunked_prompt(WIDE_TOTAL_ROWS + 1));
}

#[test]
fn window_admission_is_a_superset_of_the_w_rule() {
    let g = wide(4096);
    assert!(g.window_rows_admitted(512, 512));
    assert!(g.window_rows_admitted(1024, 3072));
    assert!(g.window_rows_admitted(1904, 4096)); // historical tail
    assert!(!g.window_rows_admitted(1024, 512));
    assert!(!g.window_rows_admitted(1, 0));
    assert!(!g.window_rows_admitted(4097, 0));
    for start in [0, 4096, 8192] {
        assert!(chunk_window_for_geometry(4096, start, true, g));
        assert!(exact_chunk_window_for_geometry(4096, start, true, g));
    }
    assert!(exact_chunk_window_for_geometry(512, 4608, true, g));
    assert!(!exact_chunk_window_for_geometry(512, 4608, false, g));
    assert!(!exact_chunk_window_for_geometry(1024, 4608, true, g));
}

fn walk(g: ChunkGeometry, prompt: &[u32], chunks: &[usize]) -> Qwen4PrefillLineage {
    let mut state = Qwen4PrefillLineage::Unset;
    let mut start = 0;
    for &rows in chunks {
        let ticket = begin_fast_lineage_for_geometry(g, &mut state, prompt, start, rows, true)
            .unwrap_or_else(|| panic!("chunk {start}+{rows} must admit"));
        assert_eq!(state, Qwen4PrefillLineage::Ordinary);
        commit_fast_lineage_for_geometry(g, &mut state, ticket, prompt, start, start + rows)
            .unwrap();
        start += rows;
    }
    assert_eq!(start, prompt.len());
    state
}

#[test]
fn lineage_carries_a_short_prompt_across_bounded_chunks() {
    let prompt: Vec<u32> = (0..3000).collect();
    let state = walk(wide(4096), &prompt, &[512, 512, 1024, 952]);
    assert!(matches!(
        state,
        Qwen4PrefillLineage::Fast {
            next_offset: 3000,
            ..
        }
    ));
}

#[test]
fn lineage_regrows_from_bounded_chunks_back_to_w() {
    // Decode active for two chunks, then the box drained: 512, 512, 1024,
    // 2048, then a W chunk at the W-aligned offset, then the remainder.
    let prompt: Vec<u32> = (0..10_000).collect();
    let state = walk(wide(4096), &prompt, &[512, 512, 1024, 2048, 4096, 1808]);
    assert!(matches!(
        state,
        Qwen4PrefillLineage::Fast {
            next_offset: 10_000,
            ..
        }
    ));
}

#[test]
fn single_chunk_prompt_keeps_its_unset_state() {
    let prompt: Vec<u32> = (0..3000).collect();
    let mut state = Qwen4PrefillLineage::Unset;
    assert!(
        begin_fast_lineage_for_geometry(wide(4096), &mut state, &prompt, 0, 3000, true).is_none()
    );
    assert_eq!(state, Qwen4PrefillLineage::Unset);
}

#[test]
fn misaligned_bounded_chunk_fails_closed_for_the_request() {
    let prompt: Vec<u32> = (0..6000).collect();
    let g = wide(4096);
    // 512 at 0 admits; 1024 at 512 does not.
    let mut state = Qwen4PrefillLineage::Unset;
    let ticket = begin_fast_lineage_for_geometry(g, &mut state, &prompt, 0, 512, true).unwrap();
    commit_fast_lineage_for_geometry(g, &mut state, ticket, &prompt, 0, 512).unwrap();
    assert!(begin_fast_lineage_for_geometry(g, &mut state, &prompt, 512, 1024, true).is_none());
    assert_eq!(state, Qwen4PrefillLineage::Ordinary);
    // Ordinary is terminal: a later aligned chunk cannot resurrect it.
    assert!(begin_fast_lineage_for_geometry(g, &mut state, &prompt, 1024, 1024, true).is_none());
    assert_eq!(state, Qwen4PrefillLineage::Ordinary);
}

fn surface(g: ChunkGeometry, start: usize, rows: usize, total: usize, last: bool) -> bool {
    admit_surface_for_geometry(
        false, start, start, rows, total, last, true, true, true, false, g,
    )
    .is_ok()
}

#[test]
fn surface_admits_bounded_chunks_and_still_refuses_non_16_multiples() {
    let g = wide(4096);
    assert!(surface(g, 0, 512, 6000, false));
    assert!(surface(g, 512, 512, 6000, false));
    assert!(surface(g, 1024, 1024, 6000, false));
    assert!(surface(g, 5632, 368, 6000, true));
    // Odd tail only as the prompt's final chunk (historical rule).
    assert!(surface(g, 2048, 3, 2051, true));
    assert!(!surface(g, 1024, 1024, 6000, true)); // is_last must match the extent
    assert!(!surface(g, 512, 1024, 6000, false)); // misaligned
    // The rows=5370 Xid 13 shape (non-16-multiple, not the prompt end) stays refused.
    let g8 = wide(8192);
    assert!(!surface(g8, 0, 5370, 16_000, false));
    assert!(!surface(g, 0, 1000, 6000, false));
}
