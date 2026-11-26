// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    CORE32_SELECTOR, DEVICE_SELECTOR, HC_SELECTOR, TILE_ROWS, admit_surface_for_selection,
    begin_fast_lineage, commit_fast_lineage, continuation_state_ready,
    mark_ordinary_for_selected_route, parse, parse_core32, parse_device, parse_hc, plan,
    plan_for_tile, request_chain_matches,
};
use crate::traits::Qwen4PrefillLineage;

#[test]
fn selector_is_exact_and_default_off() {
    assert!(!parse(None).unwrap());
    assert!(!parse(Some("0")).unwrap());
    assert!(parse(Some("1")).unwrap());
    for bad in ["", "true", "01", "2", " 1"] {
        assert!(parse(Some(bad)).is_err(), "accepted {bad:?}");
    }
}

#[test]
fn device_selector_is_exact_default_off_and_nested() {
    assert!(!parse_device(None, false).unwrap());
    assert!(!parse_device(Some("0"), false).unwrap());
    assert!(parse_device(Some("1"), true).unwrap());
    assert!(parse_device(Some("1"), false).is_err());
    for bad in ["", "true", "01", "2", " 1"] {
        assert!(parse_device(Some(bad), true).is_err(), "accepted {bad:?}");
    }
    assert_eq!(DEVICE_SELECTOR, "ATLAS_QWEN4_PREFILL_ATTN_DEVICE16");
}

#[test]
fn hc_selector_is_exact_default_off_and_nested_under_f37() {
    assert!(!parse_hc(None, false, false).unwrap());
    assert!(!parse_hc(Some("0"), true, true).unwrap());
    assert!(parse_hc(Some("1"), true, true).unwrap());
    assert!(parse_hc(Some("1"), false, true).is_err());
    assert!(parse_hc(Some("1"), true, false).is_err());
    for bad in ["", "true", "01", "2", " 1"] {
        assert!(parse_hc(Some(bad), true, true).is_err(), "accepted {bad:?}");
    }
    assert_eq!(HC_SELECTOR, "ATLAS_QWEN4_PREFILL_ATTN_HC16");
}

#[test]
fn core32_selector_is_exact_default_off_and_fully_nested() {
    assert!(!parse_core32(None, true, true, true).unwrap());
    assert!(!parse_core32(Some("0"), true, true, true).unwrap());
    assert!(parse_core32(Some("1"), true, true, true).unwrap());
    for deps in [
        (false, true, true),
        (true, false, true),
        (true, true, false),
    ] {
        assert!(parse_core32(Some("1"), deps.0, deps.1, deps.2).is_err());
    }
    for bad in ["", "true", "01", "2", " 1"] {
        assert!(
            parse_core32(Some(bad), true, true, true).is_err(),
            "accepted {bad:?}"
        );
    }
    assert_eq!(CORE32_SELECTOR, "ATLAS_QWEN4_PREFILL_ATTN_CORE32");
}

#[test]
fn planner_covers_full_tiles_and_scalar_tails() {
    for (rows, tiles, tail) in [
        (16, 1, 0),
        (17, 1, 1),
        (31, 1, 15),
        (32, 2, 0),
        (2048, 128, 0),
    ] {
        let p = plan(rows, 0, 9, 4096).unwrap();
        assert_eq!((p.full_tiles, p.tail_rows), (tiles, tail));
        assert_eq!(p.block_table_words, TILE_ROWS * 9);
        assert_eq!(p.seq_lens_offset % 8, 0);
        assert!(p.scratch_bytes <= 4096);
    }
}

#[test]
fn planner_fails_before_unsafe_geometry() {
    for rows in [0, 1, 15, 2049, usize::MAX] {
        assert!(plan(rows, 0, 9, usize::MAX).is_err());
    }
    assert!(plan(16, 1, 9, 4096).is_err());
    assert!(plan(16, 0, 0, 4096).is_err());
    let needed = plan(16, 0, 9, usize::MAX).unwrap().scratch_bytes;
    assert!(plan(16, 0, 9, needed - 1).is_err());
}

#[test]
fn planner32_halves_complete_tiles_and_sizes_causal_metadata() {
    let p = plan_for_tile(2048, 0, 9, 8192, 32).unwrap();
    assert_eq!((p.full_tiles, p.tail_rows), (64, 0));
    assert_eq!(p.block_table_words, 32 * 9);
    assert_eq!(p.seq_lens_offset % 8, 0);
    assert!(p.scratch_bytes <= 8192);
    for tile in [0, 1, 15, 17, 31, 33, 64, usize::MAX] {
        if tile != 16 && tile != 32 {
            assert!(plan_for_tile(2048, 0, 9, usize::MAX, tile).is_err());
        }
    }
}

#[test]
fn surface_is_cold_text_single_chunk_without_swap() {
    assert!(
        admit_surface_for_selection(false, 0, 0, 2048, 2048, true, false, true, true, false)
            .is_ok()
    );
    for args in [
        (true, 0, 0, 2048, 2048, true, false, true, true, false),
        (false, 1, 0, 2048, 2048, true, false, true, true, false),
        (false, 0, 1, 2047, 2048, true, false, true, true, false),
        (false, 0, 0, 1024, 2048, true, false, true, true, false),
        (false, 0, 0, 2048, 2048, true, false, true, true, true),
    ] {
        assert!(
            admit_surface_for_selection(
                args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9,
            )
            .is_err()
        );
    }
}

#[test]
fn multi_surface_requires_the_complete_cold_four_chunk_chain() {
    for (start, is_last) in [(0, false), (2048, false), (4096, false), (6144, true)] {
        assert!(
            admit_surface_for_selection(
                false, start, start, 2048, 8192, is_last, true, true, true, false,
            )
            .is_ok(),
            "rejected canonical chunk at {start}"
        );
    }
    for args in [
        // Aligned but not the selected 8K prompt.
        (2048, 2048, 2048, 4096, true, true, true, false),
        // Warm/drifted sequence cursor.
        (4096, 2048, 2048, 8192, false, true, true, false),
        // Token/prompt identity mismatch.
        (2048, 2048, 2048, 8192, false, false, true, false),
        // Missing or stale persistent page table.
        (2048, 2048, 2048, 8192, false, true, false, false),
        // Premature final chunk marker.
        (2048, 2048, 2048, 8192, true, true, true, false),
        // HSS cannot provide absolute page-table ownership.
        (2048, 2048, 2048, 8192, false, true, true, true),
    ] {
        assert!(
            admit_surface_for_selection(
                false, args.0, args.1, args.2, args.3, args.4, true, args.5, args.6, args.7,
            )
            .is_err()
        );
    }
}

#[test]
fn request_chain_rejects_warm_or_mismatched_sequence_state() {
    let prompt: Vec<u32> = (0..8192).collect();
    assert!(request_chain_matches(&prompt, &[], 0, 0, 0));
    assert!(!request_chain_matches(&prompt, &[], 0, 8192, 0));
    assert!(request_chain_matches(
        &prompt,
        &prompt[..2048],
        2048,
        8192,
        2048,
    ));
    let mut wrong = prompt[..2048].to_vec();
    wrong[2047] ^= 1;
    assert!(!request_chain_matches(&prompt, &wrong, 2048, 8192, 2048));
    assert!(!request_chain_matches(
        &prompt,
        &prompt[..2048],
        4096,
        8192,
        2048,
    ));
    assert!(!request_chain_matches(
        &prompt,
        &prompt[..2048],
        2048,
        4096,
        2048,
    ));
}

#[test]
fn continuation_requires_live_uploaded_prior_page_metadata() {
    assert!(continuation_state_ready(0, 8192, 16, 0, None));
    assert!(continuation_state_ready(
        2048,
        8192,
        16,
        128,
        Some((512, 128, true)),
    ));
    for meta in [
        None,
        Some((127, 128, true)),
        Some((512, 127, true)),
        Some((512, 129, true)),
        Some((512, 128, false)),
    ] {
        assert!(!continuation_state_ready(2048, 8192, 16, 128, meta));
    }
    assert!(!continuation_state_ready(
        2048,
        8192,
        16,
        127,
        Some((512, 128, true)),
    ));
}

fn fast_prompt() -> Vec<u32> {
    (0..super::super::MULTI_TOTAL_ROWS as u32).collect()
}

fn finish_fast_chunk(state: &mut Qwen4PrefillLineage, prompt: &[u32], chunk_start: usize) {
    let ticket = begin_fast_lineage(
        state,
        prompt,
        chunk_start,
        super::super::MULTI_CHUNK_ROWS,
        true,
    )
    .expect("canonical transition must admit");
    assert_eq!(*state, Qwen4PrefillLineage::Ordinary);
    commit_fast_lineage(
        state,
        ticket,
        prompt,
        chunk_start,
        chunk_start + super::super::MULTI_CHUNK_ROWS,
    )
    .unwrap();
}

#[test]
fn fast_lineage_advances_only_the_exact_four_chunk_chain() {
    let prompt = fast_prompt();
    let mut state = Qwen4PrefillLineage::Unset;
    for expected in [2048, 4096, 6144, 8192] {
        finish_fast_chunk(&mut state, &prompt, expected - 2048);
        assert!(matches!(
            &state,
            Qwen4PrefillLineage::Fast {
                next_offset,
                prompt_tokens,
            } if *next_offset == expected && prompt_tokens.as_ref() == prompt.as_slice()
        ));
    }
}

#[test]
fn suppressed_first_chunk_permanently_blocks_later_singular_reentry() {
    let prompt = fast_prompt();
    let mut state = Qwen4PrefillLineage::Unset;
    assert!(
        begin_fast_lineage(&mut state, &prompt, 0, 2048, false).is_none(),
        "a suppressed surface must not mint a ticket"
    );
    assert_eq!(state, Qwen4PrefillLineage::Ordinary);
    assert!(begin_fast_lineage(&mut state, &prompt, 2048, 2048, true).is_none());
    assert_eq!(state, Qwen4PrefillLineage::Ordinary);
}

#[test]
fn suppressed_middle_chunk_permanently_blocks_later_singular_reentry() {
    let prompt = fast_prompt();
    let mut state = Qwen4PrefillLineage::Unset;
    finish_fast_chunk(&mut state, &prompt, 0);
    assert!(begin_fast_lineage(&mut state, &prompt, 2048, 2048, false).is_none());
    assert_eq!(state, Qwen4PrefillLineage::Ordinary);
    assert!(begin_fast_lineage(&mut state, &prompt, 4096, 2048, true).is_none());
}

#[test]
fn rollback_wrong_prompt_and_new_sequence_continuation_fail_closed() {
    let prompt = fast_prompt();

    let mut rollback = Qwen4PrefillLineage::Unset;
    finish_fast_chunk(&mut rollback, &prompt, 0);
    assert!(begin_fast_lineage(&mut rollback, &prompt, 0, 2048, true).is_none());
    assert_eq!(rollback, Qwen4PrefillLineage::Ordinary);

    let mut identity = Qwen4PrefillLineage::Unset;
    finish_fast_chunk(&mut identity, &prompt, 0);
    let mut wrong_prompt = prompt.clone();
    wrong_prompt[4096] ^= 1;
    assert!(begin_fast_lineage(&mut identity, &wrong_prompt, 2048, 2048, true).is_none());
    assert_eq!(identity, Qwen4PrefillLineage::Ordinary);

    let mut new_sequence = Qwen4PrefillLineage::Unset;
    assert!(begin_fast_lineage(&mut new_sequence, &prompt, 2048, 2048, true).is_none());
    assert_eq!(new_sequence, Qwen4PrefillLineage::Ordinary);
}

#[test]
fn dropped_or_mismatched_ticket_leaves_the_sequence_ordinary() {
    let prompt = fast_prompt();
    let mut dropped = Qwen4PrefillLineage::Unset;
    let ticket = begin_fast_lineage(&mut dropped, &prompt, 0, 2048, true).unwrap();
    drop(ticket);
    assert_eq!(dropped, Qwen4PrefillLineage::Ordinary);

    let mut mismatch = Qwen4PrefillLineage::Unset;
    let ticket = begin_fast_lineage(&mut mismatch, &prompt, 0, 2048, true).unwrap();
    assert!(commit_fast_lineage(&mut mismatch, ticket, &prompt, 0, 4096).is_err());
    assert_eq!(mismatch, Qwen4PrefillLineage::Ordinary);

    let mut changed = Qwen4PrefillLineage::Unset;
    let ticket = begin_fast_lineage(&mut changed, &prompt, 0, 2048, true).unwrap();
    changed = Qwen4PrefillLineage::Unset;
    assert!(commit_fast_lineage(&mut changed, ticket, &prompt, 0, 2048).is_err());
    assert_eq!(changed, Qwen4PrefillLineage::Ordinary);
}

#[test]
fn alternate_route_and_prompt_switch_poison_only_selected_live_lineage() {
    let prompt = fast_prompt();
    let mut selected = Qwen4PrefillLineage::Unset;
    assert!(mark_ordinary_for_selected_route(
        &mut selected,
        prompt.len()
    ));
    assert_eq!(selected, Qwen4PrefillLineage::Ordinary);

    let mut unrelated = Qwen4PrefillLineage::Unset;
    assert!(!mark_ordinary_for_selected_route(&mut unrelated, 2048));
    assert_eq!(unrelated, Qwen4PrefillLineage::Unset);

    let mut switched = Qwen4PrefillLineage::Unset;
    finish_fast_chunk(&mut switched, &prompt, 0);
    assert!(mark_ordinary_for_selected_route(&mut switched, 4096));
    assert_eq!(switched, Qwen4PrefillLineage::Ordinary);
}
