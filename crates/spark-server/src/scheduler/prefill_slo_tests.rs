// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const W4096: [usize; 4] = [512, 1024, 2048, 4096];

fn inputs(offset: usize, remaining: usize) -> DecideInputs {
    DecideInputs {
        candidate: Candidate { offset, remaining },
        widths: PreparedWidths::new(&W4096).unwrap(),
        max_prefill_tokens: 4096,
        cap: None,
        carry_align: 16,
        single_chunk_required: false,
        decode_rows_active: true,
        target_ms: 150.0,
        decode_step_ms: 30.0,
        steps_since_chunk: 0,
    }
}

/// A linear model: `ms_per_token` per row (no offset dependence).
fn linear(ms_per_token: f64) -> impl Fn(usize, usize) -> f64 {
    move |rows, _| rows as f64 * ms_per_token
}

fn chunk(rows: usize, reason: Reason) -> Decision {
    Decision::Chunk { rows, reason }
}

#[test]
fn no_decode_rows_is_passthrough() {
    let mut i = inputs(0, 6000);
    i.decode_rows_active = false;
    assert_eq!(decide(&i, linear(0.01)), Decision::Passthrough);
}

#[test]
fn mla_keeps_the_single_chunk_exemption() {
    let mut i = inputs(0, 6000);
    i.single_chunk_required = true;
    assert_eq!(decide(&i, linear(10.0)), chunk(6000, Reason::SingleChunk));
}

#[test]
fn target_met_picks_the_largest_width_under_target_minus_decode() {
    // budget = 150 − 30 = 120 ms; 0.1 ms/token → 1024 rows = 102.4 fits,
    // 2048 = 204.8 does not.
    assert_eq!(
        decide(&inputs(0, 6000), linear(0.1)),
        chunk(1024, Reason::TargetMet)
    );
    // A slower decode step shrinks the budget: 150 − 80 = 70 → 512.
    let mut i = inputs(0, 6000);
    i.decode_step_ms = 80.0;
    assert_eq!(decide(&i, linear(0.1)), chunk(512, Reason::TargetMet));
    // Alignment: at offset 512 only 512 divides the offset.
    assert_eq!(
        decide(&inputs(512, 6000), linear(0.01)),
        chunk(512, Reason::TargetMet)
    );
    // A cheap model on a W-aligned offset takes the full W, and a remainder
    // inside one W chunk goes whole.
    assert_eq!(
        decide(&inputs(4096, 6000), linear(0.01)),
        chunk(4096, Reason::TargetMet)
    );
    assert_eq!(
        decide(&inputs(4096, 1904), linear(0.01)),
        chunk(1904, Reason::ShortLane)
    );
    assert_eq!(
        decide(&inputs(0, 9000), linear(0.01)),
        chunk(4096, Reason::TargetMet)
    );
}

#[test]
fn chunk_is_clipped_at_the_prompt_end() {
    // Offset 5632 (11 x 512): the aligned width is 512, the remainder 368.
    let i = inputs(5632, 368);
    assert_eq!(decide(&i, linear(0.1)), chunk(368, Reason::ShortLane));
    let mut i = inputs(5632, 368);
    i.target_ms = 60.0; // short lane needs <= 120 ms: 36.8 fits it
    assert_eq!(decide(&i, linear(0.1)), chunk(368, Reason::ShortLane));
    let mut i = inputs(2048, 952);
    i.target_ms = 40.0; // short lane: 95.2 > 80; budget 10: nothing fits
    i.steps_since_chunk = 100;
    assert_eq!(decide(&i, linear(0.1)), chunk(512, Reason::Starvation));
}

#[test]
fn short_prompt_lane_runs_a_chat_sized_prompt_whole() {
    // 900-token prompt, 0.25 ms/token = 225 ms: over the 120 ms budget but
    // within 2 x 150 = 300 ms, and one 1024-row chunk holds it.
    assert_eq!(
        decide(&inputs(0, 900), linear(0.25)),
        chunk(900, Reason::ShortLane)
    );
    // Over 2 x target: sliced (here starvation-bounded) instead.
    let mut i = inputs(0, 900);
    i.steps_since_chunk = 10;
    assert_eq!(decide(&i, linear(0.5)), chunk(512, Reason::Starvation));
    // The remainder must fit ONE admissible chunk at its offset: at 512 only
    // a 512 width is aligned, so a 700-token remainder is not a short lane.
    let i = inputs(512, 700);
    assert_ne!(decide(&i, linear(0.01)), chunk(700, Reason::ShortLane));
    assert_eq!(decide(&i, linear(0.01)), chunk(512, Reason::TargetMet));
}

#[test]
fn starvation_guard_runs_the_smallest_width_every_n_steps() {
    // Nothing fits (0.5 ms/token: 512 rows = 256 ms > 120 ms).
    assert_eq!(starvation_steps(150.0, 30.0), 5);
    for steps in 0..5 {
        let mut i = inputs(0, 6000);
        i.steps_since_chunk = steps;
        assert_eq!(decide(&i, linear(0.5)), Decision::Skip, "steps={steps}");
    }
    let mut i = inputs(0, 6000);
    i.steps_since_chunk = 5;
    assert_eq!(decide(&i, linear(0.5)), chunk(512, Reason::Starvation));
}

#[test]
fn starvation_interval_formula_is_bounded() {
    assert_eq!(starvation_steps(150.0, 50.0), 3);
    assert_eq!(starvation_steps(150.0, 151.0), 1); // decode slower than the target
    assert_eq!(starvation_steps(150.0, 0.0), MAX_STARVATION_STEPS);
    assert_eq!(starvation_steps(150.0, 0.5), MAX_STARVATION_STEPS);
    assert_eq!(starvation_steps(150.0, f64::NAN), MAX_STARVATION_STEPS);
}

#[test]
fn cap_bounds_the_widths_the_slo_may_pick() {
    let mut i = inputs(0, 9000);
    i.cap = Some(1024);
    assert_eq!(decide(&i, linear(0.01)), chunk(1024, Reason::TargetMet));
}

#[test]
fn off_boundary_offsets_use_carry_aligned_chunks() {
    let mut i = inputs(1000, 6000);
    i.carry_align = 32;
    assert_eq!(decide(&i, linear(0.1)), chunk(512, Reason::TargetMet));
    i.steps_since_chunk = 10;
    assert_eq!(decide(&i, linear(1.0)), chunk(512, Reason::Starvation));
}

#[test]
fn every_chunk_boundary_is_a_committed_carry_position() {
    // Walk prompts through the decision with several cost models: every
    // intermediate boundary is a multiple of the chunk's prepared width (the
    // fast family's rule) and therefore of 16 and of the 4-token groups.
    for per_token in [0.01, 0.1, 0.25, 0.5] {
        for total in [700, 3000, 6000, 10_000] {
            let mut offset = 0;
            let mut steps = 0;
            while offset < total {
                let mut i = inputs(offset, total - offset);
                i.steps_since_chunk = steps;
                match decide(&i, linear(per_token)) {
                    Decision::Chunk { rows, .. } => {
                        let end = offset + rows;
                        assert!(
                            end == total || (W4096.contains(&rows) && offset.is_multiple_of(rows)),
                            "per_token={per_token} total={total} offset={offset} rows={rows}"
                        );
                        assert!(end == total || end.is_multiple_of(16));
                        offset = end;
                        steps = 0;
                    }
                    Decision::Skip => steps += 1,
                    Decision::Passthrough => panic!("decode rows are active"),
                }
            }
        }
    }
}

#[test]
fn tick_plan_gates_admission_and_continue() {
    let mut rt = SloRuntime::new(&W4096, 150, true, false, 16);
    let c = Some(Candidate {
        offset: 0,
        remaining: 6000,
    });
    // No decode row: ungated passthrough.
    let idle = rt.plan_tick(c, false, None, 4096, false);
    assert_eq!(idle.decision, Decision::Passthrough);
    assert_eq!(idle.plan.slo, None);
    assert_eq!(idle.admission_batch_cap(8, 0, 1), 8);
    assert!(idle.continue_allowed(&PrefillTickLedger::default()));
    // Decoding, static table (512 ≈ 350 ms > 100 ms budget), no starvation
    // yet: skip, so no admission and no continue chunk.
    let skip = rt.plan_tick(c, true, None, 4096, false);
    assert_eq!(skip.decision, Decision::Skip);
    assert_eq!(skip.admission_batch_cap(8, 4, 0), 4);
    assert!(!skip.continue_allowed(&PrefillTickLedger::default()));
    // A short prompt takes the lane: admit one when nothing is mid-prefill,
    // and the continue phase may run it unless chunk 0 already ran.
    let short = rt.plan_tick(
        Some(Candidate {
            offset: 0,
            remaining: 100,
        }),
        true,
        None,
        4096,
        false,
    );
    assert_eq!(short.decision, chunk(100, Reason::ShortLane));
    assert_eq!(short.admission_batch_cap(8, 4, 0), 5);
    assert_eq!(short.admission_batch_cap(8, 4, 1), 5); // nothing new beyond the one mid-prefill
    let mut ran = PrefillTickLedger::default();
    ran.record_chunk(100, 0);
    assert!(short.continue_allowed(&PrefillTickLedger::default()));
    assert!(!short.continue_allowed(&ran));
    assert_eq!(
        short.plan.slo,
        Some(SloChunk {
            rows: 100,
            remaining: 100
        })
    );
}

#[test]
fn runtime_starvation_counter_counts_decode_steps_while_a_prompt_waits() {
    let mut rt = SloRuntime::new(&W4096, 150, true, false, 16);
    let c = Some(Candidate {
        offset: 0,
        remaining: 6000,
    });
    // Default decode step 50 ms → N = 3.
    for _ in 0..3 {
        assert_eq!(
            rt.plan_tick(c, true, None, 4096, false).decision,
            Decision::Skip
        );
        rt.observe_tick(PrefillTickLedger::default(), true, 50.0);
    }
    assert_eq!(rt.steps_since_chunk(), 3);
    let forced = rt.plan_tick(c, true, None, 4096, false);
    assert_eq!(forced.decision, chunk(512, Reason::Starvation));
    let mut ran = PrefillTickLedger::default();
    ran.record_chunk(512, 0);
    rt.observe_tick(ran, true, 400.0);
    assert_eq!(rt.steps_since_chunk(), 0);
    // No prompt waiting: the counter stays at zero.
    rt.plan_tick(None, true, None, 4096, false);
    rt.observe_tick(PrefillTickLedger::default(), true, 50.0);
    assert_eq!(rt.steps_since_chunk(), 0);
}

#[test]
fn runtime_learns_chunk_cost_from_one_chunk_then_decode_ticks() {
    let mut rt = SloRuntime::new(&W4096, 150, true, false, 16);
    rt.plan_tick(None, true, None, 4096, false);
    rt.observe_tick(PrefillTickLedger::default(), true, 40.0); // decode 40 ms
    let mut tick = PrefillTickLedger::default();
    tick.record_chunk(512, 0);
    for _ in 0..3 {
        rt.observe_tick(tick, true, 240.0); // chunk = 200 ms (first is warm-up)
    }
    assert!(rt.cost().is_measured(512, 0));
    assert!((rt.cost().predict_ms(512, 0) - 200.0).abs() < 1e-9);
    // Tainted ticks and ticks without a decode step teach nothing.
    let mut tainted = tick;
    tainted.taint();
    rt.observe_tick(tainted, true, 10_000.0);
    rt.observe_tick(tick, false, 10_000.0);
    assert!((rt.cost().predict_ms(512, 0) - 200.0).abs() < 1e-9);
}

#[test]
fn runtime_engages_only_with_widths_target_chunking_and_multi_chunk_models() {
    let on = SloRuntime::new(&W4096, 150, true, false, 16);
    assert!(on.engaged());
    assert!(
        on.startup_message()
            .contains("PREFILL_INTERLEAVE on mode=slo")
    );
    for off in [
        SloRuntime::new(&W4096, 0, true, false, 16),
        SloRuntime::new(&W4096, 150, false, false, 16),
        SloRuntime::new(&W4096, 150, true, true, 16),
        SloRuntime::new(&[], 150, true, false, 16),
        SloRuntime::new(&[1024, 512], 150, true, false, 16),
    ] {
        assert!(!off.engaged());
        assert!(off.startup_message().contains("PREFILL_INTERLEAVE off"));
    }
    // Disengaged plans are the historical sizing.
    let mut off = SloRuntime::new(&W4096, 0, true, false, 16);
    let plan = off.plan_tick(None, true, Some(1024), 4096, false);
    assert_eq!(plan.plan.widths, None);
    assert_eq!(plan.plan.cap_with_active_decode, Some(1024));
    assert_eq!(plan.admission_batch_cap(8, 4, 0), 8);
}

#[test]
fn carry_alignment_is_the_lcm_of_tiles_and_kv_block() {
    assert_eq!(
        SloRuntime::new(&W4096, 150, true, false, 16).carry_align(),
        16
    );
    assert_eq!(
        SloRuntime::new(&W4096, 150, true, false, 32).carry_align(),
        32
    );
    assert_eq!(
        SloRuntime::new(&W4096, 150, true, false, 24).carry_align(),
        48
    );
    assert_eq!(
        SloRuntime::new(&W4096, 150, true, false, 0).carry_align(),
        16
    );
}

#[test]
fn parity_width_is_strict_and_forces_the_width_on_an_idle_box() {
    assert_eq!(parse_parity_width(None, &W4096), Ok(None));
    assert_eq!(parse_parity_width(Some("0"), &W4096), Ok(None));
    assert_eq!(parse_parity_width(Some("1024"), &W4096), Ok(Some(1024)));
    for bad in ["768", "8192", "x", "", " 512"] {
        assert!(parse_parity_width(Some(bad), &W4096).is_err(), "{bad:?}");
    }
    let mut rt = SloRuntime::new(&W4096, 150, true, false, 16).with_parity_width(Some(1024));
    let tick = rt.plan_tick(
        Some(Candidate {
            offset: 0,
            remaining: 3000,
        }),
        false,
        None,
        4096,
        false,
    );
    assert_eq!(tick.decision, chunk(1024, Reason::Parity));
    assert_eq!(tick.plan.fallback_width, 1024);
    assert_eq!(tick.admission_batch_cap(8, 0, 0), 8); // idle gating unchanged
    let off = SloRuntime::new(&W4096, 0, true, false, 16).with_parity_width(Some(1024));
    assert_eq!(off.parity_width(), None);
}
