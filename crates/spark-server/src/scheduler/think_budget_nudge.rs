// SPDX-License-Identifier: AGPL-3.0-only

//! Thinking-budget nudge (`FLUX_THINK_BUDGET_NUDGE`, default ON).
//!
//! When a turn's thinking reaches its budget, the scheduler used to force a
//! bare `</think>`. The model then often kept reasoning in the answer channel
//! (9.15.6 LCB arm G: 9 of 11 length-cut answers had no code). Now one
//! sentence telling the model to answer is forced into the think block first,
//! and the existing forced `</think>` follows it.
//!
//! Every injected token is a forced sample through the same seam as the
//! forced `</think>` and the repetition nudge ([`super::think_nudge`]): it is
//! committed through the ordinary per-token path, counted as a thinking token
//! and as an output token, so the KV/SSM state describes exactly the
//! committed tokens and `max_output_tokens` bounds the nudge. A nudge that
//! would not leave room for itself plus `</think>` is not started: the close
//! is the bare `</think>` as before.
//!
//! The in-fence deferral of the forced `</think>` ([`should_inject_think_end`])
//! gates the nudge too, so it never lands mid code statement.

use std::sync::{Arc, OnceLock};

use anyhow::{Result, bail};

use super::ActiveSeq;
use super::confidence::{compute_think_defer_hard_override, should_inject_think_end};

/// Qwen3's published thinking-budget wording. The Qwen3.8 chat template
/// carries no budget string, so this is the fallback text, verbatim.
/// The forced `</think>` follows this text; the model writes the answer.
pub const BUDGET_NUDGE_TEXT: &str = "\n\nConsidering the limited time by the user, I have to \
give the solution based on the thinking directly now.\n";

static NUDGE_TOKENS: OnceLock<Option<Arc<[u32]>>> = OnceLock::new();

/// `FLUX_THINK_BUDGET_NUDGE`: unset/empty/on = on, `0`/`off`/`false` = off.
pub fn parse_enabled(raw: Option<&str>) -> Result<bool> {
    match raw.map(str::trim) {
        None | Some("") | Some("on") | Some("1") | Some("true") => Ok(true),
        Some("off") | Some("0") | Some("false") => Ok(false),
        Some(other) => bail!("FLUX_THINK_BUDGET_NUDGE={other:?}: expected on|off"),
    }
}

/// Validate the tokenized nudge: non-empty, no structural ids.
pub fn validate_tokens(ids: Vec<u32>, forbidden: &[u32]) -> Result<Arc<[u32]>> {
    if ids.is_empty() {
        bail!("thinking-budget nudge tokenizes to nothing");
    }
    if let Some(bad) = ids.iter().find(|id| forbidden.contains(id)) {
        bail!("thinking-budget nudge contains structural token {bad}");
    }
    Ok(Arc::from(ids))
}

/// Install the process-wide nudge tokens once per load (`None` = off). A
/// second install (model swap) keeps the first value and is reported.
pub fn install(tokens: Option<Arc<[u32]>>) {
    if NUDGE_TOKENS.set(tokens).is_err() {
        tracing::warn!("thinking-budget nudge tokens already installed; keeping the first set");
    }
}

#[cfg(not(test))]
fn installed() -> Option<Arc<[u32]>> {
    NUDGE_TOKENS.get().and_then(Clone::clone)
}

// Tests never install the process-wide tokens (that would turn the nudge on
// for every other test); a test sets its own thread's tokens instead.
#[cfg(test)]
thread_local! {
    static TEST_TOKENS: std::cell::RefCell<Option<Arc<[u32]>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn installed() -> Option<Arc<[u32]>> {
    TEST_TOKENS.with(|t| t.borrow().clone())
}

#[cfg(test)]
pub(super) fn set_test_tokens(tokens: Option<Arc<[u32]>>) {
    TEST_TOKENS.with(|t| *t.borrow_mut() = tokens);
}

/// One sequence's in-flight budget nudge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetNudge {
    tokens: Arc<[u32]>,
    next: usize,
}

impl BudgetNudge {
    /// Start a nudge if the response cap leaves room for every nudge token
    /// plus the forced `</think>`; `None` means a bare `</think>` close.
    pub fn plan(
        tokens: Option<Arc<[u32]>>,
        output_len: usize,
        max_output_tokens: usize,
    ) -> Option<Self> {
        let tokens = tokens?;
        let room = max_output_tokens.saturating_sub(output_len);
        (room > tokens.len()).then_some(Self { tokens, next: 0 })
    }

    pub fn pending_token(&self) -> Option<u32> {
        self.tokens.get(self.next).copied()
    }

    /// The state after `tok` was committed: advanced on the expected token,
    /// `None` once complete or when another token was committed mid-nudge
    /// (the close then falls back to the bare `</think>`). Before the first
    /// nudge token any other token leaves it waiting: the token that armed
    /// the close, and tokens while the close is deferred inside a fence.
    pub fn advanced(&self, tok: u32) -> Option<Self> {
        if self.pending_token() != Some(tok) {
            return (self.next == 0).then(|| self.clone());
        }
        let next = self.next + 1;
        (next < self.tokens.len()).then(|| Self {
            tokens: Arc::clone(&self.tokens),
            next,
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }
}

// ── Scheduler hooks ──────────────────────────────────────────────────────────

/// Every "thinking budget exhausted" site calls this: arms the forced close,
/// plans the nudge and writes the receipt. `unpushed` is 1 when the token
/// that exhausted the budget is not in `output_tokens` yet but will be (the
/// serial decode step arms before it pushes), else 0.
pub(super) fn arm_budget_close(a: &mut ActiveSeq, budget: u32, unpushed: usize) {
    a.force_end_thinking = true;
    let output_len = a.output_tokens.len() + unpushed;
    a.budget_nudge = BudgetNudge::plan(installed(), output_len, a.max_output_tokens);
    let nudge = if a.budget_nudge.is_some() {
        "on"
    } else {
        "off"
    };
    crate::metrics::THINK_BUDGET_FORCED_CLOSE
        .with_label_values(&[nudge])
        .inc();
    tracing::info!(
        "THINK_BUDGET_FORCED_CLOSE session={:016x} thinking_tokens={} cap={} nudge={}",
        a.session_hash,
        a.thinking_tokens,
        budget,
        nudge,
    );
}

/// Whether the forced close may fire at this position (the boundary gate
/// shared by the nudge and the forced `</think>`).
pub(super) fn close_allowed(a: &ActiveSeq) -> bool {
    let defer_hard_override = compute_think_defer_hard_override(
        a.thinking_tokens,
        a.thinking_budget,
        a.output_tokens.len(),
        a.max_output_tokens,
    );
    a.inside_thinking
        && should_inject_think_end(a.force_end_thinking, a.in_code_fence, defer_hard_override)
}

/// The nudge token this position must commit, if one is being injected.
pub(super) fn forced_token(a: &ActiveSeq) -> Option<u32> {
    a.budget_nudge
        .as_ref()
        .filter(|_| close_allowed(a))
        .and_then(BudgetNudge::pending_token)
}

/// Up to `max_len` known forced tokens queued for this sequence (`budget_nudge`
/// remainder followed by `</think>`) when `close_allowed(a)` is active.
pub(super) fn remaining_forced_drafts(a: &ActiveSeq, max_len: usize) -> Vec<u32> {
    if max_len == 0
        || !close_allowed(a)
        || a.pending_tree_payload.is_some()
        || a.pending_dflash_sampled_proposal.is_some()
    {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(max_len);
    if let Some(ref nudge) = a.budget_nudge {
        for &tok in nudge.tokens.get(nudge.next..).unwrap_or(&[]) {
            if out.len() >= max_len {
                break;
            }
            out.push(tok);
        }
    }
    if out.len() < max_len
        && let Some(end) = a.think_end_token
    {
        out.push(end);
    }
    out
}

/// Return a full `num_drafts` forced chain when the queued budget nudge (+ `</think>`)
/// covers the entire draft window, letting callers skip the drafter forward pass while
/// keeping cohort width uniform.
pub(super) fn full_forced_draft_chain(a: &ActiveSeq, num_drafts: usize) -> Option<Vec<u32>> {
    if num_drafts == 0 {
        return None;
    }
    let chain = remaining_forced_drafts(a, num_drafts);
    (chain.len() == num_drafts).then_some(chain)
}

/// Overwrite the leading positions of a freshly proposed flat draft chain with
/// any queued budget-nudge tokens (+ `</think>`) so speculative verification
/// accepts them in a single batched verify step without ejecting the cohort.
pub(super) fn splice_forced_drafts(a: &ActiveSeq, drafts: &mut [u32]) {
    if drafts.is_empty() {
        return;
    }
    let forced = remaining_forced_drafts(a, drafts.len());
    if !forced.is_empty() {
        drafts[..forced.len()].copy_from_slice(&forced);
    }
}

/// Call once per committed thinking-content token.
pub(super) fn on_thinking_token(a: &mut ActiveSeq, tok: u32) {
    a.budget_nudge = a.budget_nudge.as_ref().and_then(|n| n.advanced(tok));
}

/// Call when the think block closes (or is abandoned).
pub(super) fn on_think_closed(a: &mut ActiveSeq) {
    a.budget_nudge = None;
}

#[cfg(test)]
#[path = "think_budget_nudge_tests.rs"]
mod think_budget_nudge_tests;

#[cfg(test)]
mod runway_override_tests {
    use super::*;

    #[test]
    fn close_allowed_and_forced_token_fire_mid_fence_when_answer_runway_exhausted() {
        const MAX_OUTPUT: usize = 4_096;
        const BUDGET: u32 = 3_072; // 3x = 9,216 >> MAX_OUTPUT
        let runway = crate::scheduler::confidence::min_answer_runway(MAX_OUTPUT);
        assert_eq!(runway, 409);
        let cutoff = MAX_OUTPUT - runway;

        let mut a = crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::ResponseSink::Blocking(None),
            0,
        );
        a.inside_thinking = true;
        a.enable_thinking = true;
        a.think_ended = false;
        a.think_end_token = Some(60);
        a.max_output_tokens = MAX_OUTPUT;
        a.remaining = MAX_OUTPUT;
        a.thinking_budget = Some(BUDGET);
        a.force_end_thinking = true;
        a.in_code_fence = true;
        a.thinking_tokens = (cutoff - 1) as u32;
        a.output_tokens = vec![40; cutoff - 1];
        a.budget_nudge = BudgetNudge::plan(Some(Arc::from(vec![49u32, 50, 51])), 3_072, MAX_OUTPUT);

        assert!(
            !close_allowed(&a),
            "still deferred inside code fence 1 token before answer runway"
        );
        assert_eq!(forced_token(&a), None);

        a.output_tokens.push(40);
        a.thinking_tokens += 1;
        assert!(
            close_allowed(&a),
            "answer-runway exhaustion must override in-fence deferral via SSOT"
        );
        assert_eq!(forced_token(&a), Some(49));
    }
}
