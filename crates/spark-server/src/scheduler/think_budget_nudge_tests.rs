// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::Arc;

use super::*;

fn toks() -> Arc<[u32]> {
    Arc::from(vec![10u32, 11, 12])
}

#[test]
fn enabled_by_default_and_disabled_by_zero() {
    assert!(parse_enabled(None).unwrap());
    assert!(parse_enabled(Some("")).unwrap());
    assert!(parse_enabled(Some("1")).unwrap());
    assert!(!parse_enabled(Some("0")).unwrap());
    assert!(!parse_enabled(Some(" off ")).unwrap());
    assert!(parse_enabled(Some("maybe")).is_err());
}

#[test]
fn text_is_the_qwen_budget_wording() {
    assert!(BUDGET_NUDGE_TEXT.contains(
        "Considering the limited time by the user, I have to give the solution based on \
         the thinking directly now."
    ));
    assert!(!BUDGET_NUDGE_TEXT.contains("</think>"));
}

#[test]
fn validate_rejects_empty_and_structural() {
    assert!(validate_tokens(vec![], &[]).is_err());
    assert!(validate_tokens(vec![1, 2, 3], &[2]).is_err());
    assert_eq!(&*validate_tokens(vec![1, 3], &[2]).unwrap(), &[1, 3]);
}

#[test]
fn plan_needs_room_for_nudge_plus_think_end() {
    // 3 nudge tokens + 1 `</think>` = 4 positions.
    assert!(BudgetNudge::plan(Some(toks()), 96, 100).is_some());
    // Room 3 < 4: bare `</think>` only.
    assert!(BudgetNudge::plan(Some(toks()), 97, 100).is_none());
    // Exhausted or over the cap: no nudge, no underflow.
    assert!(BudgetNudge::plan(Some(toks()), 100, 100).is_none());
    assert!(BudgetNudge::plan(Some(toks()), 150, 100).is_none());
    // Disabled.
    assert!(BudgetNudge::plan(None, 0, 100).is_none());
}

#[test]
fn advances_through_every_token_then_completes() {
    let n = BudgetNudge::plan(Some(toks()), 0, 100).unwrap();
    assert_eq!(n.len(), 3);
    assert_eq!(n.pending_token(), Some(10));
    let n = n.advanced(10).unwrap();
    assert_eq!(n.pending_token(), Some(11));
    let n = n.advanced(11).unwrap();
    assert_eq!(n.pending_token(), Some(12));
    assert!(n.advanced(12).is_none());
}

#[test]
fn a_foreign_token_mid_nudge_abandons_it() {
    let n = BudgetNudge::plan(Some(toks()), 0, 100).unwrap();
    // Before the first nudge token (the arming token, a fence deferral) the
    // nudge waits.
    assert_eq!(n.advanced(99), Some(n.clone()));
    let n = n.advanced(10).unwrap();
    assert!(n.advanced(99).is_none());
}

#[test]
fn injected_positions_never_pass_the_response_cap() {
    // Every boundary: if a nudge starts, output_len + nudge + `</think>`
    // fits max_output_tokens.
    for max in 0..12usize {
        for used in 0..=max + 2 {
            if let Some(n) = BudgetNudge::plan(Some(toks()), used, max) {
                assert!(used + n.len() < max, "used={used} max={max}");
            }
        }
    }
}

// ── Scheduler paths (mock model) ────────────────────────────────────────────

const TE: u32 = 60;
const THINK_TOK: u32 = 40;
const NUDGE: [u32; 3] = [49, 50, 51];

fn thinking_seq(
    budget: u32,
    max_output: usize,
) -> (
    crate::scheduler::ActiveSeq,
    tokio::sync::oneshot::Receiver<anyhow::Result<crate::api::InferenceResponse>>,
) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let mut a = crate::scheduler::test_fixtures::active_seq(
        crate::scheduler::ResponseSink::Blocking(Some(tx)),
        0,
    );
    a.output_tokens.clear();
    a.max_output_tokens = max_output;
    a.remaining = max_output;
    a.inside_thinking = true;
    a.enable_thinking = true;
    a.think_ended = false;
    a.content_started = false;
    a.content_tokens = 0;
    a.think_end_token = Some(TE);
    a.thinking_budget = Some(budget);
    (a, rx)
}

fn model() -> crate::scheduler::test_fixtures::model::TestModel {
    let m = crate::scheduler::test_fixtures::model::recording();
    let mut row = vec![0.0f32; 64];
    row[THINK_TOK as usize] = 8.0;
    *m.host_logits.lock().unwrap() = row;
    m
}

fn steps(
    model: &dyn spark_model::traits::Model,
    a: crate::scheduler::ActiveSeq,
    n: usize,
) -> crate::scheduler::ActiveSeq {
    let mut active = vec![a];
    for _ in 0..n {
        if active[0].finished {
            break;
        }
        crate::scheduler::process_decode_logits(
            model,
            &mut active,
            spark_runtime::gpu::DevicePtr::NULL,
            std::time::Instant::now(),
            Some(TE),
            None,
            None,
            None,
            None,
            &[],
            false,
            false,
        );
    }
    active.pop().expect("sequence stays active")
}

fn run(budget: u32, max_output: usize, tokens: Option<Arc<[u32]>>) -> Vec<u32> {
    set_test_tokens(tokens);
    let m = model();
    let (a, _rx) = thinking_seq(budget, max_output);
    let a = steps(&m, a, budget as usize + 8);
    set_test_tokens(None);
    assert!(a.output_tokens.len() <= max_output, "never past max_tokens");
    a.output_tokens
}

#[test]
fn budget_close_injects_the_nudge_before_think_end() {
    let out = run(4, 100, Some(Arc::from(NUDGE.to_vec())));
    let te = out.iter().position(|&t| t == TE).expect("thinking closed");
    assert_eq!(
        &out[..te],
        &[THINK_TOK, THINK_TOK, THINK_TOK, THINK_TOK, 49, 50, 51]
    );
}

#[test]
fn nudge_off_keeps_the_bare_think_end() {
    let out = run(4, 100, None);
    let te = out.iter().position(|&t| t == TE).expect("thinking closed");
    assert_eq!(&out[..te], &[THINK_TOK; 4]);
}

#[test]
fn no_room_for_the_nudge_closes_bare() {
    // 4 thinking tokens used, 3 left: nudge (3) + `</think>` (1) does not fit.
    let out = run(4, 7, Some(Arc::from(NUDGE.to_vec())));
    assert_eq!(&out[..5], &[THINK_TOK, THINK_TOK, THINK_TOK, THINK_TOK, TE]);
    // Exactly enough room: nudge then `</think>` fills the cap.
    let out = run(4, 8, Some(Arc::from(NUDGE.to_vec())));
    assert_eq!(
        out,
        vec![THINK_TOK, THINK_TOK, THINK_TOK, THINK_TOK, 49, 50, 51, TE]
    );
}

#[test]
fn nudge_waits_for_a_closed_code_fence_like_think_end() {
    let (mut a, _rx) = thinking_seq(4, 100);
    a.thinking_tokens = 4;
    a.force_end_thinking = true;
    a.budget_nudge = BudgetNudge::plan(Some(toks()), 4, 100);
    a.in_code_fence = true;
    assert_eq!(forced_token(&a), None, "deferred inside a fence");
    on_thinking_token(&mut a, THINK_TOK);
    assert!(a.budget_nudge.is_some(), "a deferred nudge keeps waiting");
    a.in_code_fence = false;
    assert_eq!(forced_token(&a), Some(10));
    // Past 3x the budget the hard override fires mid-fence, nudge first.
    a.in_code_fence = true;
    a.thinking_tokens = 12;
    assert_eq!(forced_token(&a), Some(10));
    on_think_closed(&mut a);
    assert_eq!(forced_token(&a), None);
}

#[test]
fn speculative_budget_nudge_splices_and_accepts_full_frame_without_d2h() {
    use crate::scheduler::think_spec_accept::{ThinkSpecCtx, dflash_thinking_accept};

    let (mut a, _rx) = thinking_seq(4, 100);
    a.thinking_tokens = 4;
    a.temperature = 1.0;
    a.top_k = 20;
    a.top_p = 0.95;
    a.force_end_thinking = true;
    a.budget_nudge = BudgetNudge::plan(Some(Arc::from(vec![49u32, 50, 51])), 4, 100);

    // Budget nudge stays in speculative/batched mode (does not eject cohort).
    assert!(!crate::scheduler::think_nudge::injection_pending(&a));
    // Full K=5 draft window (4 drafts) gets `[49, 50, 51, TE]` directly.
    let chain = full_forced_draft_chain(&a, 4).expect("4 forced tokens fill K=5");
    assert_eq!(chain, vec![49, 50, 51, TE]);

    // Non-greedy verify walk commits all 3 nudge tokens + `</think>` bonus
    // in ONE frame with ZERO D2H fetches (`fetch_row` panics if called).
    let ctx = ThinkSpecCtx {
        enabled: true,
        code_fence_token: None,
        reflection_suppress_ids: &[],
        adaptive_sampling: false,
        honor_eos_inside_thinking: Some(false),
    };
    let verified = [THINK_TOK; 5];
    let mut think_ended_fired = false;
    let out = dflash_thinking_accept(
        &mut a,
        &chain,
        &verified,
        &ctx,
        |_, _| panic!("forced budget nudge must not issue D2H even at T=1.0"),
        |_| think_ended_fired = true,
    );
    assert_eq!(out.num_accepted, 3);
    assert_eq!(out.bonus, Some(TE));
    assert_eq!(a.output_tokens, vec![49, 50, 51, TE]);
    assert!(!a.inside_thinking);
    assert!(a.think_ended);
    assert!(think_ended_fired);
}
