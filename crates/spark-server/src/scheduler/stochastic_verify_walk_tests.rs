// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-free tests for the `FLUX_SPEC_VERIFY=stochastic` accept walk.

use super::spec_policy_accept::VerifyLogitsFormat;
use super::stochastic_verify::DraftDistribution;
use super::stochastic_verify_walk::*;
use super::*;

const VOCAB: usize = 32;
const THINK_END: u32 = 5;
const THINK_START: u32 = 4;

fn seq(temperature: f32) -> ActiveSeq {
    let mut a = super::test_fixtures::active_seq(ResponseSink::Blocking(None), 0);
    a.temperature = temperature;
    a.top_k = 20;
    a.top_p = 0.95;
    a.seed = Some(1234);
    a
}

fn ctx() -> ThinkSpecCtx<'static> {
    ThinkSpecCtx {
        enabled: true,
        code_fence_token: None,
        reflection_suppress_ids: &[],
        adaptive_sampling: false,
        honor_eos_inside_thinking: Some(false),
    }
}

fn bf16_row(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// A row that puts essentially all target mass on `top`.
fn peaked(top: u32) -> Vec<u8> {
    let mut v = vec![-30.0f32; VOCAB];
    v[top as usize] = 30.0;
    bf16_row(&v)
}

fn serve(rows: &[Vec<u8>]) -> impl FnMut(usize, &mut Vec<u8>) -> bool + '_ {
    move |i, buf| {
        let Some(row) = rows.get(i) else { return false };
        buf.clear();
        buf.extend_from_slice(row);
        true
    }
}

fn walk(a: &mut ActiveSeq, drafts: &[u32], rows: &[Vec<u8>]) -> SpecAcceptOutcome {
    stochastic_content_accept(
        a,
        drafts,
        rows.len(),
        VerifyLogitsFormat::Bf16,
        VOCAB,
        |_| DraftDistribution::PointMass,
        serve(rows),
    )
}

#[test]
fn policy_parse_defaults_to_argmax() {
    assert_eq!(parse_spec_verify_policy(None), SpecVerifyPolicy::Argmax);
    assert_eq!(parse_spec_verify_policy(Some("")), SpecVerifyPolicy::Argmax);
    assert_eq!(
        parse_spec_verify_policy(Some("argmax")),
        SpecVerifyPolicy::Argmax
    );
    assert_eq!(
        parse_spec_verify_policy(Some("Stochastic")),
        SpecVerifyPolicy::Argmax
    );
    assert_eq!(
        parse_spec_verify_policy(Some("stochastic")),
        SpecVerifyPolicy::Stochastic
    );
    assert_eq!(
        parse_spec_verify_policy(Some(" stochastic ")),
        SpecVerifyPolicy::Stochastic
    );
    assert_eq!(
        parse_spec_verify_policy(Some("gumbel")),
        SpecVerifyPolicy::Gumbel
    );
    assert_eq!(
        parse_spec_verify_policy(Some(" gumbel ")),
        SpecVerifyPolicy::Gumbel
    );
}

#[test]
fn masked_row_widens_bf16_and_fp32_and_masks() {
    let vals = [1.0f32, 2.0, 3.0];
    let fp32: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = masked_f32_row(&bf16_row(&vals), VerifyLogitsFormat::Bf16, 3, &[1]).unwrap();
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(got, vec![1.0, f32::NEG_INFINITY, 3.0]);
    assert_eq!(
        masked_f32_row(&fp32, VerifyLogitsFormat::Fp32, 3, &[]).unwrap(),
        fp32
    );
    assert!(masked_f32_row(&fp32, VerifyLogitsFormat::Fp32, 4, &[]).is_none());
}

#[test]
fn eligibility_requires_sampling_content_without_interventions() {
    assert!(stochastic_seq_eligible(&seq(1.0), &ctx()));
    assert!(
        !stochastic_seq_eligible(&seq(0.0), &ctx()),
        "T=0 keeps the argmax path"
    );
    let closes: [fn(&mut ActiveSeq); 7] = [
        |a| a.inside_thinking = true,
        |a| a.think_just_ended = true,
        |a| a.suppress_tool_call = true,
        |a| a.require_tool_call = true,
        |a| a.inside_tool_body = true,
        |a| a.top_logprobs = Some(5),
        |a| a.min_tokens = 10,
    ];
    for (i, close) in closes.iter().enumerate() {
        let mut a = seq(1.0);
        close(&mut a);
        assert!(
            !stochastic_seq_eligible(&a, &ctx()),
            "intervention {i} must close the gate"
        );
    }
    let adaptive = ThinkSpecCtx {
        adaptive_sampling: true,
        ..ctx()
    };
    assert!(!stochastic_seq_eligible(&seq(1.0), &adaptive));
}

#[test]
fn all_accepted_frame_commits_drafts_and_bonus() {
    let mut a = seq(1.0);
    let out = walk(&mut a, &[9, 10], &[peaked(9), peaked(10), peaked(11)]);
    assert_eq!(
        (out.num_accepted, out.bonus, out.consumed_rows),
        (2, Some(11), 3)
    );
    assert_eq!(a.output_tokens, vec![31, 9, 10, 11]);
    assert_eq!(a.last_token, 11);
}

#[test]
fn rejection_stops_the_walk_with_the_target_token() {
    let mut a = seq(1.0);
    let out = walk(
        &mut a,
        &[9, 10, 12],
        &[peaked(9), peaked(14), peaked(12), peaked(13)],
    );
    assert_eq!(
        (out.num_accepted, out.bonus, out.consumed_rows),
        (1, Some(14), 2)
    );
    assert_eq!(a.output_tokens, vec![31, 9, 14]);
}

#[test]
fn post_thinking_mask_is_applied_to_target() {
    let mut a = seq(1.0);
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    let mut v = vec![-30.0f32; VOCAB];
    v[THINK_END as usize] = 30.0;
    v[THINK_START as usize] = 29.0;
    v[9] = 0.0;
    let out = walk(&mut a, &[THINK_END], &[bf16_row(&v), peaked(10)]);
    assert_eq!((out.num_accepted, out.bonus), (0, Some(9)));
}

#[test]
fn seeded_walk_is_reproducible() {
    let spread = bf16_row(&(0..VOCAB).map(|i| (i % 7) as f32 * 0.3).collect::<Vec<_>>());
    let rows = vec![spread.clone(), spread.clone(), spread];
    let run = || {
        let mut a = seq(1.0);
        let out = walk(&mut a, &[3, 4], &rows);
        (out.num_accepted, out.bonus, a.output_tokens)
    };
    assert_eq!(run(), run());
}

#[test]
fn missing_row_finishes_the_sequence() {
    let mut a = seq(1.0);
    let out = stochastic_content_accept(
        &mut a,
        &[9],
        2,
        VerifyLogitsFormat::Bf16,
        VOCAB,
        |_| DraftDistribution::PointMass,
        |_, _| false,
    );
    assert!(a.finished);
    assert_eq!(
        (out.num_accepted, out.bonus, out.consumed_rows),
        (0, None, 0)
    );
}

#[test]
fn malformed_row_finishes_the_sequence() {
    let mut a = seq(1.0);
    let short = vec![0u8; 4];
    let out = walk(&mut a, &[], &[short]);
    assert!(a.finished);
    assert_eq!(out.bonus, None);
}

/// The identity gate: for a seeded request, the walk's committed tokens are
/// exactly what plain decode samples at the same positions, whatever the
/// drafts were. Plain decode = `sample_with_params_history` on the same
/// masked row with `sampling_params` (seed + output index).
#[test]
fn seeded_walk_is_token_identical_to_plain_sampling() {
    // Ids 17 (fixture EOS) and 23 (tool-call close) would end the walk.
    let spread: Vec<f32> = (0..VOCAB)
        .map(|i| {
            if i == 17 || i == 23 {
                -30.0
            } else {
                ((i * 7) % 11) as f32 * 0.25
            }
        })
        .collect();
    let rows = vec![bf16_row(&spread); 4];
    for seed in 0..200u64 {
        let mut plain = seq(1.0);
        plain.seed = Some(seed);
        let f32_row = masked_f32_row(&rows[0], VerifyLogitsFormat::Bf16, VOCAB, &[]).unwrap();
        let mut expected = plain.output_tokens.clone();
        for _ in 0..4 {
            let params = plain.sampling_params(plain.temperature);
            let token = sample_with_params_history(&f32_row, &params, &plain.output_tokens);
            plain.output_tokens.push(token);
            expected.push(token);
        }
        // Drafts that are mostly right: the plain tokens themselves.
        let mut a = seq(1.0);
        a.seed = Some(seed);
        let drafts = [expected[1], expected[2], expected[3]];
        let out = walk(&mut a, &drafts, &rows);
        assert_eq!(out.num_accepted, 3, "seed {seed}");
        assert_eq!(a.output_tokens, expected, "seed {seed}");
        // Drafts that are wrong at position 1: the walk still emits the
        // plain token there, then stops.
        let mut b = seq(1.0);
        b.seed = Some(seed);
        let wrong = (expected[2] + 1) % VOCAB as u32;
        let out = walk(&mut b, &[expected[1], wrong, expected[3]], &rows);
        assert_eq!(out.num_accepted, 1, "seed {seed}");
        assert_eq!(b.output_tokens, expected[..3].to_vec(), "seed {seed}");
    }
}

/// Generate at least `n` tokens with frames of `k` drafts. Drafts are a fixed
/// guess pattern (often wrong), so frame boundaries land at different
/// absolute positions for different `k`.
fn generate_with_depth(seed: u64, k: usize, n: usize, row: &[u8]) -> Vec<u32> {
    let mut a = seq(1.0);
    a.seed = Some(seed);
    while a.output_tokens.len() < n + 1 && !a.finished {
        let len = a.output_tokens.len();
        let drafts: Vec<u32> = (0..k).map(|j| ((len * 3 + j * 5) % 7) as u32 + 8).collect();
        let rows = vec![row.to_vec(); k + 1];
        walk(&mut a, &drafts, &rows);
    }
    a.output_tokens.truncate(n + 1);
    a.output_tokens
}

/// The position-keying invariant: every draw is keyed by the committed
/// absolute output position, never by frame start or column, so the same
/// seed commits the same tokens whatever the draft depth.
#[test]
fn same_seed_commits_identical_tokens_at_draft_depth_1_and_5() {
    let spread: Vec<f32> = (0..VOCAB)
        .map(|i| {
            if i == 17 || i == 23 {
                -30.0
            } else {
                ((i * 5) % 9) as f32 * 0.3
            }
        })
        .collect();
    let row = bf16_row(&spread);
    for seed in 0..100u64 {
        let k1 = generate_with_depth(seed, 1, 24, &row);
        let k5 = generate_with_depth(seed, 5, 24, &row);
        assert_eq!(k1, k5, "seed {seed}");
        assert_eq!(k1.len(), 25);
    }
}

#[test]
fn gumbel_content_walk_is_bit_identical_to_serial_k1_across_any_draft_depth_and_prompt_len() {
    let spread: Vec<f32> = (0..VOCAB)
        .map(|i| {
            if i == 17 || i == 23 {
                -30.0
            } else {
                ((i * 7 + 3) % 13) as f32 * 0.25 - 1.0
            }
        })
        .collect();
    let row = bf16_row(&spread);
    let f32_row = masked_f32_row(&row, VerifyLogitsFormat::Bf16, VOCAB, &[]).unwrap();

    for seed in 0..80u64 {
        let prompt_len = ((seed % 5) as usize) * 11;
        // 1. Reference K=1 serial decode using `choose_gumbel(logits, seed, prompt_len + out_len)`.
        let mut serial = seq(0.85);
        serial.seed = Some(seed);
        serial.seq.prompt_len = prompt_len;
        serial.repetition_penalty = 1.1;
        serial.presence_penalty = 0.4;
        for _ in 0..20 {
            let params = serial.sampling_params(serial.temperature);
            let abs_pos = (serial.seq.prompt_len + serial.output_tokens.len()) as u64;
            let tok = spark_runtime::sampler::choose_gumbel(
                &f32_row,
                &params,
                &serial.output_tokens,
                seed,
                abs_pos,
            );
            serial.output_tokens.push(tok);
        }

        // 2. Speculative Gumbel walk at K=2 (1 draft) and K=6 (5 drafts), mixing hits and misses.
        for k in [1usize, 4, 5] {
            let mut spec = seq(0.85);
            spec.seed = Some(seed);
            spec.seq.prompt_len = prompt_len;
            spec.repetition_penalty = 1.1;
            spec.presence_penalty = 0.4;
            while spec.output_tokens.len() < serial.output_tokens.len() && !spec.finished {
                let cur = spec.output_tokens.len();
                let drafts: Vec<u32> = (0..k)
                    .map(|j| {
                        // Inject deliberate hits at even positions and misses at odd ones!
                        if (cur + j).is_multiple_of(2) && cur + j < serial.output_tokens.len() {
                            serial.output_tokens[cur + j]
                        } else {
                            ((cur + j * 3) % 10) as u32 + 2
                        }
                    })
                    .collect();
                let rows = vec![row.clone(); k + 1];
                stochastic_content_accept_with_policy(
                    &mut spec,
                    &drafts,
                    rows.len(),
                    VerifyLogitsFormat::Bf16,
                    VOCAB,
                    SpecVerifyPolicy::Gumbel,
                    |_| DraftDistribution::PointMass,
                    serve(&rows),
                );
            }
            spec.output_tokens.truncate(serial.output_tokens.len());
            assert_eq!(
                spec.output_tokens,
                serial.output_tokens,
                "Gumbel K={} diverged from K=1 serial at seed={seed} prompt_len={prompt_len}",
                k + 1
            );
        }
    }
}

#[test]
fn gumbel_thinking_walk_is_eligible_and_reproduces_masks_and_phase_boundary() {
    let mut a = seq(1.0);
    a.inside_thinking = true;
    a.think_ended = false;
    a.think_end_token = Some(THINK_END);
    a.tool_call_start_token = Some(6);
    let reflect_ids = [8u32];
    let think_ctx = ThinkSpecCtx {
        enabled: true,
        code_fence_token: None,
        reflection_suppress_ids: &reflect_ids,
        adaptive_sampling: false,
        honor_eos_inside_thinking: Some(false),
    };
    assert!(gumbel_thinking_seq_eligible(&a, &think_ctx));
    assert!(gumbel_seq_eligible(&a, &think_ctx));

    // Row 0 puts huge mass on masked `<tool_call>` (6) and suppressed reflection id (8),
    // leaving token 9 as the post-intervention winner; Row 1 emits `</think>` (phase boundary).
    let mut r0 = vec![-30.0f32; VOCAB];
    r0[6] = 50.0; // hard-masked inside thinking
    r0[8] = 9.0; // F1 -10.0 drops to -1.0
    r0[9] = 8.0; // survives at +8.0 -> wins!
    let mut r1 = vec![-30.0f32; VOCAB];
    r1[THINK_END as usize] = 40.0;
    let rows = vec![bf16_row(&r0), bf16_row(&r1), peaked(12)];

    let mut boundary_snaps = 0usize;
    let out = gumbel_thinking_accept(
        &mut a,
        &[9, THINK_END],
        rows.len(),
        VerifyLogitsFormat::Bf16,
        VOCAB,
        &think_ctx,
        serve(&rows),
        |_| boundary_snaps += 1,
    );
    // Token 9 accepted as draft 0; `</think>` is a phase boundary so it ends the walk as bonus!
    assert_eq!(
        (out.num_accepted, out.bonus, out.consumed_rows),
        (1, Some(THINK_END), 2)
    );
    assert!(!a.inside_thinking);
    assert!(a.think_ended);
    assert_eq!(boundary_snaps, 1);
}
