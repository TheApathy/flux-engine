// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const W4096: [usize; 4] = [512, 1024, 2048, 4096];

fn widths() -> PreparedWidths {
    PreparedWidths::new(&W4096).unwrap()
}

fn cap_plan(cap: Option<usize>) -> ChunkPlan {
    ChunkPlan {
        cap_with_active_decode: cap,
        ..ChunkPlan::default()
    }
}

/// Interleave engaged, no SLO decision (an idle tick).
fn engaged() -> ChunkPlan {
    ChunkPlan {
        widths: Some(widths()),
        fallback_width: 512,
        carry_align: 16,
        ..ChunkPlan::default()
    }
}

fn decided(rows: usize, remaining: usize) -> ChunkPlan {
    ChunkPlan {
        slo: Some(SloChunk { rows, remaining }),
        ..engaged()
    }
}

fn req(remaining: usize, max: usize, cap: Option<usize>, active: bool) -> ChunkRequest {
    ChunkRequest {
        offset: 0,
        remaining,
        max_prefill_tokens: max,
        plan: cap_plan(cap),
        decode_rows_active: active,
        single_chunk_required: false,
    }
}

fn at(offset: usize, total: usize, plan: ChunkPlan, active: bool) -> ChunkRequest {
    ChunkRequest {
        offset,
        remaining: total - offset,
        max_prefill_tokens: 4096,
        plan,
        decode_rows_active: active,
        single_chunk_required: false,
    }
}

/// Walk a prompt with one plan per chunk (the last repeats), decode state per
/// chunk likewise; returns the chunk lengths.
fn walk(total: usize, plan: impl Fn(usize, usize) -> ChunkPlan, active: &[bool]) -> Vec<usize> {
    let mut offset = 0;
    let mut chunks = Vec::new();
    while offset < total {
        let decode = active[chunks.len().min(active.len() - 1)];
        let len = prefill_chunk_len(at(offset, total, plan(offset, total - offset), decode));
        assert!(len > 0);
        chunks.push(len);
        offset += len;
    }
    chunks
}

#[test]
fn cap_resolution_is_strict() {
    assert_eq!(resolve_chunk_cap(4096, 0).unwrap(), None);
    assert_eq!(resolve_chunk_cap(4096, 1024).unwrap(), Some(1024));
    assert_eq!(resolve_chunk_cap(4096, 16).unwrap(), Some(16));
    assert_eq!(resolve_chunk_cap(4096, 4096).unwrap(), None);
    assert_eq!(resolve_chunk_cap(4096, 8192).unwrap(), None);
    assert_eq!(resolve_chunk_cap(0, 1024).unwrap(), None);
    for bad in [1, 8, 24, 1000, 1025] {
        assert!(resolve_chunk_cap(4096, bad).is_err(), "{bad}");
    }
}

#[test]
fn unset_cap_reproduces_the_historical_chunking_exactly() {
    assert_eq!(prefill_chunk_len(req(6000, 4096, None, false)), 4096);
    assert_eq!(prefill_chunk_len(req(6000, 4096, None, true)), 4096);
    assert_eq!(prefill_chunk_len(req(1904, 4096, None, true)), 1904);
    assert_eq!(prefill_chunk_len(req(3, 4096, None, true)), 3);
    assert_eq!(prefill_chunk_len(req(6000, 4094, None, false)), 4092);
    assert_eq!(prefill_chunk_len(req(6000, 3, None, false)), 3);
}

#[test]
fn cap_applies_only_while_decode_rows_exist() {
    let cap = Some(1024);
    assert_eq!(prefill_chunk_len(req(6000, 4096, cap, false)), 4096);
    assert_eq!(prefill_chunk_len(req(6000, 4096, cap, true)), 1024);
    assert_eq!(
        prefill_chunk_len(req(6000, 4096, Some(1040), true)) % PREFILL_CHUNK_ALIGN,
        0
    );
    assert_eq!(prefill_chunk_len(req(900, 4096, cap, true)), 900);
    assert_eq!(prefill_chunk_len(req(1024, 4096, cap, true)), 1024);
}

#[test]
fn single_chunk_models_ignore_the_cap() {
    let mut r = req(6000, 4096, Some(1024), true);
    r.single_chunk_required = true;
    assert_eq!(prefill_chunk_len(r), 6000);
}

/// Review A HIGH: chunk 0 of a single-chunk (MLA) model must not be capped
/// or bounded while another row decodes; GLM-5.3 bails on any partial prompt.
#[test]
fn single_chunk_models_ignore_the_cap_and_slo_on_chunk_zero() {
    let cap = cap_plan(Some(1024));
    assert_eq!(
        initial_chunk_budget(8192, 8193, cap, true, false, 9000),
        1024
    );
    assert_eq!(
        initial_chunk_budget(8192, 8193, cap, true, true, 9000),
        8192
    );
    assert_eq!(
        initial_chunk_budget(2048, 8193, cap, true, true, 9000),
        2048
    );
    assert_eq!(
        initial_chunk_budget(8192, 8193, cap, false, false, 9000),
        8192
    );
    assert_eq!(
        initial_chunk_budget(8192, 8193, cap_plan(None), true, false, 9000),
        8192
    );
    let mut r = at(0, 6000, decided(512, 6000), true);
    r.single_chunk_required = true;
    assert_eq!(prefill_chunk_len(r), 6000);
    assert_eq!(
        initial_chunk_budget(4096, 4097, decided(512, 6000), true, true, 6000),
        4096
    );
}

/// The chunk-0 call site passes the model's single-chunk predicate and the
/// prompt length; run_standard passes the offset.
#[test]
fn call_sites_pass_the_single_chunk_predicate_offset_and_prompt() {
    let start: String = include_str!("phase_start_prefills.rs")
        .split_whitespace()
        .collect();
    assert!(start.contains(
        "initial_chunk_budget(max_prefill_tokens,max_batch_tokens,chunk_plan,!complete_all_chunks&&!active.is_empty(),model.is_mla(),prompt_len,)"
    ));
    let later: String = include_str!("phase_continue_prefills/run_standard.rs")
        .split_whitespace()
        .collect();
    assert!(later.contains("single_chunk_required:model.is_mla(),"));
    assert!(later.contains("offset:p.chunk_offset,"));
}

#[test]
fn six_thousand_token_prompt_under_cap_walks_aligned_boundaries() {
    let chunks = walk(6000, |_, _| cap_plan(Some(1024)), &[true]);
    assert_eq!(chunks, [1024, 1024, 1024, 1024, 1024, 880]);
}

#[test]
fn decode_draining_mid_prefill_returns_to_full_chunks() {
    let chunks = walk(6000, |_, _| cap_plan(Some(1024)), &[true, false]);
    assert_eq!(chunks, [1024, 4096, 880]);
}

#[test]
fn initial_budget_never_consumes_decode_slot_capacity_and_honours_cap() {
    let none = cap_plan(None);
    let cap = cap_plan(Some(1_024));
    assert_eq!(
        initial_chunk_budget(8_192, 8_193, none, false, false, 9_000),
        8_192
    );
    assert_eq!(
        initial_chunk_budget(2_048, 8_193, none, true, false, 9_000),
        2_048
    );
    assert_eq!(
        initial_chunk_budget(8_192, 8_193, cap, false, false, 9_000),
        8_192
    );
    assert_eq!(
        initial_chunk_budget(8_192, 8_193, cap, true, false, 9_000),
        1_024
    );
    assert_eq!(
        initial_chunk_budget(8_192, 512, cap, true, false, 9_000),
        512
    );
}

#[test]
fn fast_path_warning_fires_only_below_the_geometry_without_interleave() {
    assert!(fast_path_warning(None, Some(4096), false).is_none());
    assert!(fast_path_warning(Some(1024), None, false).is_none());
    assert!(fast_path_warning(Some(4096), Some(4096), false).is_none());
    assert!(fast_path_warning(Some(8192), Some(4096), false).is_none());
    assert!(fast_path_warning(Some(1024), Some(4096), true).is_none());
    let msg = fast_path_warning(Some(1024), Some(4096), false).unwrap();
    assert!(msg.contains("1024") && msg.contains("4096"));
}

// ── SLO interleave sizing ───────────────────────────────────────────────

#[test]
fn prepared_widths_validate_and_align() {
    assert!(PreparedWidths::new(&[]).is_none());
    assert!(PreparedWidths::new(&[1024, 512]).is_none());
    assert!(PreparedWidths::new(&[512, 1000]).is_none());
    assert!(PreparedWidths::new(&[0, 512]).is_none());
    assert!(PreparedWidths::new(&[16; 13]).is_none());
    let w = widths();
    assert_eq!(w.as_slice(), W4096);
    assert_eq!(w.largest_aligned(0, 4096), Some(4096));
    assert_eq!(w.largest_aligned(1024, 4096), Some(1024));
    assert_eq!(w.largest_aligned(1000, 4096), None);
    assert_eq!(
        w.aligned_desc(2048, 4096).collect::<Vec<_>>(),
        [2048, 1024, 512]
    );
}

#[test]
fn no_decode_rows_is_byte_identical_to_the_historical_sizing() {
    for total in [100, 4096, 6000, 16_384, 20_000] {
        let with = walk(total, |_, _| engaged(), &[false]);
        let without = walk(total, |_, _| cap_plan(None), &[false]);
        assert_eq!(with, without, "total={total}");
        assert_eq!(
            initial_chunk_budget(4096, 4097, engaged(), false, false, total),
            initial_chunk_budget(4096, 4097, cap_plan(None), false, false, total)
        );
    }
}

#[test]
fn slo_decision_sets_the_chunk_for_its_prompt_only() {
    // The decided prompt gets the decision (clipped at its end)...
    assert_eq!(
        prefill_chunk_len(at(0, 6000, decided(1024, 6000), true)),
        1024
    );
    assert_eq!(
        prefill_chunk_len(at(5632, 6000, decided(512, 368), true)),
        368
    );
    assert_eq!(prefill_chunk_len(at(0, 300, decided(300, 300), true)), 300);
    // ...any other prompt gets the bounded fallback width, aligned.
    assert_eq!(prefill_chunk_len(at(0, 9000, decided(300, 300), true)), 512);
    assert_eq!(
        prefill_chunk_len(at(1024, 9000, decided(300, 300), false)),
        512
    );
    assert_eq!(
        initial_chunk_budget(4096, 4097, decided(1024, 6000), true, false, 6000),
        1024
    );
    assert_eq!(
        initial_chunk_budget(4096, 4097, decided(300, 300), true, false, 9000),
        512
    );
}

#[test]
fn idle_ticks_regrow_to_w_alignment_after_bounded_chunks() {
    // Two decided 512 chunks while decoding, then the box drains.
    let chunks = walk(
        12_000,
        |offset, remaining| {
            if offset < 1024 {
                decided(512, remaining)
            } else {
                engaged()
            }
        },
        &[true, true, false],
    );
    assert_eq!(chunks, [512, 512, 1024, 2048, 4096, 3808]);
}

#[test]
fn off_boundary_offsets_take_carry_aligned_bounded_chunks() {
    // A prefix-cache start at 1000 is on no prepared boundary: the fallback
    // is the smallest width rounded to the carry alignment.
    let mut plan = engaged();
    plan.carry_align = 32;
    assert_eq!(bounded_len(widths(), plan, 1000, 9000, 4096), 512);
    plan.fallback_width = 1000;
    assert_eq!(bounded_len(widths(), plan, 1000, 9000, 4096), 992);
    // The cap bounds the fallback too.
    plan.cap_with_active_decode = Some(256);
    assert_eq!(bounded_len(widths(), plan, 0, 9000, 4096), 256);
    // Idle and off-boundary: historical sizing.
    assert_eq!(prefill_chunk_len(at(1000, 9000, engaged(), false)), 4096);
}

#[test]
fn parity_fallback_width_bounds_every_prompt_on_an_idle_box() {
    let parity = ChunkPlan {
        slo: Some(SloChunk {
            rows: 1024,
            remaining: 6000,
        }),
        fallback_width: 1024,
        ..engaged()
    };
    assert_eq!(prefill_chunk_len(at(0, 6000, parity, false)), 1024);
    // Another prompt in the same tick: the parity width, aligned.
    assert_eq!(prefill_chunk_len(at(1024, 3000, parity, false)), 1024);
    assert_eq!(
        initial_chunk_budget(4096, 4097, parity, false, false, 3000),
        1024
    );
}
