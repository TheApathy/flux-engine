// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in deferred SSE emission & O(1) accept-loop bookkeeping for speculative
//! verification (`ATLAS_DFLASH_DEFER_EMIT=1`, default OFF).
//!
//! During a multi-token speculative verify step (e.g. DFlash2 `K=8..16`),
//! immediate per-token channel sends and per-token `Instant::now()` syscalls
//! run on the critical path between `verify_complete` and `commit + propose`
//! enqueue, and wake the Tokio SSE consumer once per token (triggering `2 * k`
//! tokenizer `decode()` calls instead of `2`).
//!
//! When `ATLAS_DFLASH_DEFER_EMIT=1` is enabled:
//! - Accepted plain tokens are coalesced into a single `StreamEvent::TokenBatch`
//!   (while preserving strict FIFO order with `TokenWithLogprobs`,
//!   `CompletionProgress`, and `ThinkingDone`), and dispatched ONLY AFTER
//!   `commit_verify_state_async` and async `propose` are already executing on
//!   the GPU (or immediately before terminal/error exits and `finish_sequence`).
//! - `Instant::now()` and `stall_timeout_s()` are sampled at most once per
//!   verify step while keeping exact per-token timeout stop semantics.
//! - Tool-call start/end positions for `fuzzy_repetition_outside_tool` are
//!   tracked incrementally in O(1) amortized time per token instead of O(N)
//!   reverse scans over `output_tokens`.
//!
//! When `ATLAS_DFLASH_DEFER_EMIT` is unset or `0` (default), `DeferredEmitState`
//! remains inactive and every SSE event is dispatched immediately on the exact
//! baseline path.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::api::StreamEvent;

/// Environment selector for deferred speculative SSE emission & batch
/// detokenization (strict `"0"` / `"1"`, default OFF).
pub const DFLASH_DEFER_EMIT_ENV: &str = "ATLAS_DFLASH_DEFER_EMIT";

/// Parse `ATLAS_DFLASH_DEFER_EMIT`: exact `"1"` (trimmed) enables; default OFF.
#[inline]
#[must_use]
pub fn parse_dflash_defer_emit_enabled(raw: Option<&str>) -> bool {
    raw.map(str::trim) == Some("1")
}

/// Cached process-lifetime gate for `ATLAS_DFLASH_DEFER_EMIT=1` (default OFF).
#[must_use]
pub fn dflash_defer_emit_enabled() -> bool {
    static GATE: OnceLock<bool> = OnceLock::new();
    *GATE.get_or_init(|| match std::env::var(DFLASH_DEFER_EMIT_ENV) {
        Ok(val) => parse_dflash_defer_emit_enabled(Some(&val)),
        Err(std::env::VarError::NotPresent) => false,
        Err(std::env::VarError::NotUnicode(_)) => {
            tracing::warn!(
                "{DFLASH_DEFER_EMIT_ENV} contains non-UTF-8 bytes; failing closed to OFF"
            );
            false
        }
    })
}

enum DeferredStreamItem {
    TokenRun(Vec<u32>),
    Event(StreamEvent),
}

/// Per-sequence deferred SSE buffer and O(1) incremental tool-tag tracker.
#[derive(Default)]
pub(super) struct DeferredEmitState {
    active: bool,
    items: Vec<DeferredStreamItem>,
    step_now: Option<Instant>,
    step_stall: Option<Option<Duration>>,
    last_tc_start_pos: Option<usize>,
    last_tc_end_pos: Option<usize>,
    tc_pos_tracked_len: usize,
    tc_start_tok_cached: Option<u32>,
    tc_end_tok_cached: Option<u32>,
}

impl DeferredEmitState {
    /// Arm deferred emission for the current speculative verify step.
    #[inline]
    pub(super) fn begin(&mut self) {
        self.active = true;
        self.step_now = None;
        self.step_stall = None;
    }

    /// True while streaming events for this step should be queued instead of
    /// sent immediately onto the SSE channel.
    #[inline]
    pub(super) fn is_active(&self) -> bool {
        self.active
    }

    /// True when deferral is armed or any queued events await flushing.
    #[inline]
    pub(super) fn has_pending(&self) -> bool {
        self.active || !self.items.is_empty()
    }

    /// Return the step-cached `(now, stall_duration)` pair, sampling the OS
    /// clock at most once per verify step while preserving exact per-token
    /// timeout enforcement.
    #[inline]
    pub(super) fn step_clock(&mut self) -> (Instant, Option<Duration>) {
        let now = *self.step_now.get_or_insert_with(Instant::now);
        let stall = *self.step_stall.get_or_insert_with(|| {
            let stall_s = crate::api::request_contract::stall_timeout_s();
            (stall_s > 0).then(|| Duration::from_secs(stall_s))
        });
        (now, stall)
    }

    /// Queue one streaming event while coalescing contiguous plain
    /// `StreamEvent::Token` runs into a single `StreamEvent::TokenBatch`
    /// without reordering across structural or logprob events.
    pub(super) fn push_event(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Token(tok) => {
                if let Some(DeferredStreamItem::TokenRun(run)) = self.items.last_mut() {
                    run.push(tok);
                } else {
                    let mut run = Vec::with_capacity(8);
                    run.push(tok);
                    self.items.push(DeferredStreamItem::TokenRun(run));
                }
            }
            other => self.items.push(DeferredStreamItem::Event(other)),
        }
    }

    /// Disarm deferral and drain all queued items in exact FIFO order.
    /// Single-token runs remain `StreamEvent::Token(t)`; runs of 2+ tokens
    /// become `StreamEvent::TokenBatch(vec)`.
    pub(super) fn drain_events(&mut self) -> Vec<StreamEvent> {
        self.active = false;
        self.step_now = None;
        self.step_stall = None;
        if self.items.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(self.items.len());
        for item in self.items.drain(..) {
            match item {
                DeferredStreamItem::TokenRun(run) => match run.len() {
                    0 => {}
                    1 => out.push(StreamEvent::Token(run[0])),
                    _ => out.push(StreamEvent::TokenBatch(run)),
                },
                DeferredStreamItem::Event(ev) => out.push(ev),
            }
        }
        out
    }

    /// Invalidate cached tool-call tag positions after any non-append rollback
    /// or truncation of `output_tokens`.
    #[inline]
    pub(super) fn invalidate_tool_call_positions(&mut self) {
        self.last_tc_start_pos = None;
        self.last_tc_end_pos = None;
        self.tc_pos_tracked_len = 0;
    }

    /// Return `(last_tool_call_start_pos, last_tool_call_end_pos)` matching
    /// `output_tokens.iter().rposition(...)` in O(1) amortized time per token.
    /// Self-heals automatically if `output_tokens` shrunk or configured tag IDs
    /// changed.
    pub(super) fn sync_tool_call_positions(
        &mut self,
        output_tokens: &[u32],
        start_tok: Option<u32>,
        end_tok: Option<u32>,
    ) -> (Option<usize>, Option<usize>) {
        if output_tokens.len() < self.tc_pos_tracked_len
            || self.tc_start_tok_cached != start_tok
            || self.tc_end_tok_cached != end_tok
        {
            self.invalidate_tool_call_positions();
            self.tc_start_tok_cached = start_tok;
            self.tc_end_tok_cached = end_tok;
        }
        if start_tok.is_none() && end_tok.is_none() {
            self.tc_pos_tracked_len = output_tokens.len();
            return (None, None);
        }
        let base = self.tc_pos_tracked_len;
        for (offset, &tok) in output_tokens[base..].iter().enumerate() {
            let pos = base + offset;
            if start_tok == Some(tok) {
                self.last_tc_start_pos = Some(pos);
            }
            if end_tok == Some(tok) {
                self.last_tc_end_pos = Some(pos);
            }
        }
        self.tc_pos_tracked_len = output_tokens.len();
        (self.last_tc_start_pos, self.last_tc_end_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defer_emit_env_parser_defaults_off_and_accepts_only_exact_one() {
        assert!(!parse_dflash_defer_emit_enabled(None));
        assert!(!parse_dflash_defer_emit_enabled(Some("")));
        assert!(!parse_dflash_defer_emit_enabled(Some("0")));
        assert!(!parse_dflash_defer_emit_enabled(Some("true")));
        assert!(!parse_dflash_defer_emit_enabled(Some("2")));
        assert!(parse_dflash_defer_emit_enabled(Some("1")));
        assert!(parse_dflash_defer_emit_enabled(Some(" 1\n")));
    }

    #[test]
    fn defer_emit_coalesces_plain_tokens_and_preserves_interleaved_fifo_order() {
        let mut state = DeferredEmitState::default();
        assert!(!state.is_active());
        assert!(!state.has_pending());

        state.begin();
        assert!(state.is_active());
        state.push_event(StreamEvent::Token(10));
        state.push_event(StreamEvent::Token(11));
        state.push_event(StreamEvent::Token(12));
        state.push_event(StreamEvent::ThinkingDone);
        state.push_event(StreamEvent::Token(20));
        state.push_event(StreamEvent::CompletionProgress);
        state.push_event(StreamEvent::Token(30));
        state.push_event(StreamEvent::Token(31));

        let drained = state.drain_events();
        assert!(!state.is_active());
        assert!(!state.has_pending());
        assert_eq!(drained.len(), 5);
        assert!(matches!(&drained[0], StreamEvent::TokenBatch(v) if v == &[10, 11, 12]));
        assert!(matches!(&drained[1], StreamEvent::ThinkingDone));
        assert!(matches!(&drained[2], StreamEvent::Token(20)));
        assert!(matches!(&drained[3], StreamEvent::CompletionProgress));
        assert!(matches!(&drained[4], StreamEvent::TokenBatch(v) if v == &[30, 31]));
    }

    #[test]
    fn incremental_tool_call_position_tracking_matches_rposition_including_truncation() {
        let mut state = DeferredEmitState::default();
        let start = Some(100u32);
        let end = Some(200u32);
        let mut toks = Vec::new();

        for &t in &[1, 2, 100, 3, 4, 200, 5, 100, 6] {
            toks.push(t);
            let expected = (
                toks.iter().rposition(|&x| x == 100),
                toks.iter().rposition(|&x| x == 200),
            );
            assert_eq!(state.sync_tool_call_positions(&toks, start, end), expected);
        }

        // Simulate rollback truncation below tracked length -> self-heals.
        toks.truncate(4);
        assert_eq!(
            state.sync_tool_call_positions(&toks, start, end),
            (Some(2), None)
        );

        // Explicit invalidation after in-place rewrite of same length -> exact rescan.
        toks[2] = 999;
        state.invalidate_tool_call_positions();
        assert_eq!(
            state.sync_tool_call_positions(&toks, start, end),
            (None, None)
        );
    }
}
