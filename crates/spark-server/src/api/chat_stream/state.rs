// SPDX-License-Identifier: AGPL-3.0-only
//
// Mutable per-stream state captured by the `flat_map` closure in
// `chat_stream.rs`. Lifted out of that closure so each `StreamEvent`
// arm can be extracted to a free function (`handle_token`,
// `handle_done`, `handle_error`) that takes `&mut StreamState` plus
// any additional non-state arguments.
//
// Read-only context (`Arc<AppState>`, model name, tool defs, ...) is
// passed via `StreamCtx` (see `ctx.rs`) so the helpers don't need to
// duplicate two dozen function-parameter slots.

use std::collections::HashMap;

use crate::tool_parser;

pub(super) struct StreamState {
    /// Token IDs accumulated since the last reset (cleared at the
    /// `</think>` boundary so post-thinking content decodes cleanly).
    pub(super) all_toks: Vec<u32>,
    /// Byte offset into the thinking-phase decoded text already
    /// emitted as `reasoning_chunk` deltas.
    pub(super) emitted: usize,
    /// Cumulative stable decode for the current thinking/content phase.  The
    /// stream emits byte slices from this full-decode-compatible buffer rather
    /// than trusting `DecodeStream` to preserve standalone whitespace tokens.
    pub(super) decoded_text: String,
    /// Incremental full-decode offsets into `all_toks`.  Together they retain
    /// enough left context for a decoder while keeping the hot path O(n).
    pub(super) detok_prefix_offset: usize,
    pub(super) detok_read_offset: usize,
    /// Decoded output suffix that is also a prefix of a configured
    /// stop string. It is withheld until the next delta proves it is
    /// ordinary content, completes the stop, or Done flushes it.
    pub(super) stop_holdback: String,
    /// Mirror of the post-sanitizer content stream; used by the
    /// post-stream refusal classifier and the `--dump` synthesiser.
    pub(super) refusal_scan_buf: String,
    /// Flips true on first stop-string match or on watchdog/dedup
    /// trip; suppresses further content emissions.
    pub(super) stop_string_triggered: bool,
    /// Why generation was stopped.  A caller-provided stop string still owns
    /// the safe prose bytes before the match, so Done must drain detector and
    /// sanitizer tails for that cause.  Hostile/capped/tool-complete stops must
    /// suppress every later detector output.
    pub(super) stop_reason: Option<StreamStopReason>,
    /// Sanitiser state: suppressing content while waiting for a
    /// matching `</parameter>` close after an orphan `<parameter=`.
    pub(super) suppressing_param_leak: bool,
    /// Sanitiser state: currently inside a tool-call envelope opener
    /// (e.g. `<minimax:tool_call>`); inner `<invoke ...>` etc. are
    /// legitimate content while this is true.
    pub(super) inside_envelope: bool,
    /// Mirror of `inside_envelope` for the reasoning sanitiser.
    pub(super) reasoning_inside_envelope: bool,
    /// Tag-scan buffer for the content sanitiser.
    pub(super) tag_scan_buf: String,
    /// Sanitiser state for reasoning content (parallel to
    /// `suppressing_param_leak` above).
    pub(super) reasoning_suppressing_leak: bool,
    /// Tag-scan buffer for the reasoning sanitiser.
    pub(super) reasoning_tag_scan_buf: String,
    /// Repetition-loop watchdog: tail buffer for line-level
    /// duplicate detection.
    pub(super) loop_scan_buf: String,
    /// Set true when the watchdog or SimHash guard fires.
    pub(super) loop_watchdog_triggered: bool,
    /// Set true when the watchdog salvages a fenced/XML tool intent
    /// into a synthetic `tool_call` so the Done arm picks the right
    /// `finish_reason`.
    pub(super) salvaged_tool_call: bool,
    /// F4: SimHash semantic-loop guard for paraphrased restarts.
    pub(super) simhash_guard: crate::loop_simhash::SimHashLoopGuard,
    /// F4: pending bytes accumulated until a sentence-boundary or
    /// 1KB force-flush triggers a `simhash_guard.check()`.
    pub(super) simhash_pending: String,
    /// F5: cross-flush tool-arg dedup (default thresholds).
    pub(super) tool_arg_dedup: crate::tool_arg_dedup::ToolArgDedup,
    /// F11: tighter within-response tool-arg dedup for the
    /// streaming `ToolCallEnd` path.
    pub(super) tool_arg_dedup_within: crate::tool_arg_dedup::ToolArgDedup,
    /// Per-streaming-toolcall accumulator keyed by `oa_idx`. Holds
    /// `(id, name, args_so_far)`. The start header stays server-side until the
    /// complete arguments pass validation and loop guards, preventing rejected
    /// calls from leaving a dangling client-visible tool object.
    pub(super) streaming_tool_args: HashMap<usize, (String, String, String)>,
    /// F12: per-response total tool-call count.
    pub(super) tool_calls_emitted_count: usize,
    /// Bug-2 (OpenClaw 2026-05-08): per-tool-name consecutive-call
    /// guard. F11 keys on `(name, canonical_args)` and is defeated by
    /// runaway loops where the model varies args slightly each
    /// iteration (e.g. timestamps, sequence numbers, IDs). This
    /// counter trips whenever the same tool name fires in N
    /// successive `ToolCallEnd` events regardless of args drift,
    /// catching the `cron`+`exec` alternation pattern observed when
    /// the streaming detector did successfully classify the calls.
    /// `(last_name, run_length)`. `last_name = None` means the run
    /// was just broken by a different tool name.
    pub(super) name_run: Option<(String, u32)>,
    /// Set true when ANY hostile tool path forcibly ends the response: invalid
    /// publication, the F12 total cap, Bug-2 name-run cap, F11 within-response
    /// dedup, F5 cross-flush dedup, or F44 permanent-failure circuit-breaker.
    /// `handle_done` reads this and overrides `finish_reason` to
    /// `"length"` — without the override the response otherwise looks
    /// like a normal `"tool_calls"` completion (because tool calls
    /// were emitted), and agent clients (opencode, etc.) cheerfully
    /// run the tools and send the next request, perpetuating the loop
    /// from the outside. `"length"` is the OpenAI-spec slot for
    /// "response was forcibly truncated" and gives every agent a
    /// clean hook to break its outer retry loop.
    pub(super) tool_loop_capped: bool,
    /// Cooperative cancellation flag shared with the scheduler. Flipped
    /// true on any forced-stop condition (`tool_loop_capped`, loop-
    /// watchdog fire, …); the scheduler reads it in
    /// `emit_step::emit_token` and finalises the sequence. Without
    /// this, `stop_string_triggered` only suppresses output and the
    /// scheduler keeps generating until natural EOS / max_tokens.
    pub(super) cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Opt-in report-only serving intervention audit.
    pub(super) intervention_ledger: Option<crate::openai::SharedInterventionLedger>,
    /// Streaming tool-call detector (`Some` iff `tools_active`).
    pub(super) detector: Option<tool_parser::StreamingToolDetector>,
    /// True iff the reasoning/`<think>` phase has finished. Starts
    /// `true` when the request did not enable thinking.
    pub(super) thinking_done: bool,
    /// Optional streaming markdown-fence unwrapper enabled when
    /// `response_format` requires JSON (`json_object` / `json_schema`).
    pub(super) json_fence_stripper: Option<super::super::chat::repair_json::JsonFenceStripper>,
}

impl StreamState {
    pub(super) fn new(
        tools_active: bool,
        enable_thinking: bool,
        cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
        intervention_ledger: Option<crate::openai::SharedInterventionLedger>,
    ) -> Self {
        Self {
            all_toks: Vec::new(),
            emitted: 0,
            decoded_text: String::new(),
            detok_prefix_offset: 0,
            detok_read_offset: 0,
            stop_holdback: String::new(),
            refusal_scan_buf: String::new(),
            stop_string_triggered: false,
            stop_reason: None,
            suppressing_param_leak: false,
            inside_envelope: false,
            reasoning_inside_envelope: false,
            tag_scan_buf: String::new(),
            reasoning_suppressing_leak: false,
            reasoning_tag_scan_buf: String::new(),
            loop_scan_buf: String::new(),
            loop_watchdog_triggered: false,
            salvaged_tool_call: false,
            simhash_guard: crate::loop_simhash::SimHashLoopGuard::new(),
            simhash_pending: String::new(),
            tool_arg_dedup: crate::tool_arg_dedup::ToolArgDedup::new(),
            tool_arg_dedup_within: crate::tool_arg_dedup::ToolArgDedup::with_params(4, 2, 3),
            streaming_tool_args: HashMap::new(),
            tool_calls_emitted_count: 0,
            name_run: None,
            tool_loop_capped: false,
            cancel_flag,
            intervention_ledger,
            detector: if tools_active {
                Some(tool_parser::StreamingToolDetector::new())
            } else {
                None
            },
            thinking_done: !enable_thinking,
            json_fence_stripper: None,
        }
    }

    /// End generation at the next scheduler boundary.  This is the single
    /// publication point for stream-side cancellation so no guard can suppress
    /// output while leaving the model decoding in the background.
    pub(super) fn stop_generation(&mut self, reason: StreamStopReason) {
        self.stop_string_triggered = true;
        match reason {
            StreamStopReason::ToolLoopOrInvalid => self.tool_loop_capped = true,
            StreamStopReason::ContentLoop => self.loop_watchdog_triggered = true,
            StreamStopReason::ClientStop | StreamStopReason::SingleToolComplete => {}
        }
        // Preserve the strongest cause if two guards meet at the same token.
        // In particular, a valid salvage after a content-loop stop must not
        // downgrade the stop into SingleToolComplete.
        if self
            .stop_reason
            .is_none_or(|current| reason.priority() > current.priority())
        {
            self.stop_reason = Some(reason);
        }
        self.cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub(super) fn record_intervention(&self, code: &str, action: &str) {
        crate::openai::record_shared_intervention(self.intervention_ledger.as_ref(), code, action);
    }

    pub(super) fn record_parser_recoveries(&self, count: usize) {
        for _ in 0..count {
            self.record_intervention("tool_parser_recovery", "recovered_noncanonical_call");
        }
    }

    /// Reset detokenizer/parser state at a reasoning-to-content boundary.
    /// The caller drains any held reasoning bytes before invoking this.
    pub(super) fn complete_thinking_phase(&mut self) {
        self.thinking_done = true;
        if let Some(ref mut detector) = self.detector {
            detector.reset();
        }
        self.emitted = 0;
        self.all_toks.clear();
        self.decoded_text.clear();
        self.detok_prefix_offset = 0;
        self.detok_read_offset = 0;
        self.reasoning_inside_envelope = false;
        self.reasoning_suppressing_leak = false;
        self.reasoning_tag_scan_buf.clear();
    }

    pub(super) fn should_flush_terminal_pipeline(&self) -> bool {
        matches!(self.stop_reason, None | Some(StreamStopReason::ClientStop))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum StreamStopReason {
    /// Invalid, duplicate, permanently failed, or capped tool output.  The
    /// terminal wire reason must be `length`, not a successful `tool_calls`.
    ToolLoopOrInvalid,
    /// Semantic/text loop watchdog.  Existing wire behavior is `stop`.
    ContentLoop,
    /// A caller-provided stop sequence matched.
    ClientStop,
    /// `parallel_tool_calls=false`: one valid call completed normally, so the
    /// wire reason remains `tool_calls` even though scheduler work is cancelled.
    SingleToolComplete,
}

impl StreamStopReason {
    fn priority(self) -> u8 {
        match self {
            Self::ClientStop => 0,
            Self::SingleToolComplete => 1,
            Self::ContentLoop => 2,
            Self::ToolLoopOrInvalid => 3,
        }
    }
}

impl Drop for StreamState {
    fn drop(&mut self) {
        // Axum drops the response-body stream when the HTTP peer disconnects
        // or its timeout expires.  The scheduler owns another sender clone,
        // so dropping ReceiverStream alone does not reliably close the token
        // channel.  Explicitly publish cancellation while the shared flag is
        // still alive; emit_token observes it at the next token boundary and
        // releases the sequence instead of generating to max_tokens.
        self.cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

#[cfg(test)]
mod disconnect_tests {
    use super::{StreamState, StreamStopReason};
    use crate::openai::{
        InterventionLedger, SharedInterventionLedger, snapshot_shared_interventions,
    };
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn parser_recovery_count_is_published_to_the_opt_in_ledger() {
        let cancel = Arc::new(AtomicBool::new(false));
        let ledger: SharedInterventionLedger = Arc::new(Mutex::new(InterventionLedger::default()));
        let state = StreamState::new(false, false, cancel, Some(Arc::clone(&ledger)));

        state.record_parser_recoveries(2);

        let snapshot = snapshot_shared_interventions(Some(&ledger)).unwrap();
        assert_eq!(snapshot.events.len(), 1);
        assert_eq!(snapshot.events[0].code, "tool_parser_recovery");
        assert_eq!(snapshot.events[0].action, "recovered_noncanonical_call");
        assert_eq!(snapshot.events[0].count, 2);
    }

    #[test]
    fn dropping_http_stream_cancels_scheduler_work() {
        let cancel = Arc::new(AtomicBool::new(false));
        let state = StreamState::new(false, false, Arc::clone(&cancel), None);

        drop(state);

        assert!(cancel.load(Ordering::Acquire));
    }

    #[test]
    fn forced_tool_stop_is_capped_and_release_published() {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut state = StreamState::new(false, false, Arc::clone(&cancel), None);

        state.stop_generation(StreamStopReason::ToolLoopOrInvalid);

        assert!(state.stop_string_triggered);
        assert!(state.tool_loop_capped);
        assert_eq!(state.stop_reason, Some(StreamStopReason::ToolLoopOrInvalid));
        assert!(cancel.load(Ordering::Acquire));
    }

    #[test]
    fn single_valid_tool_stop_cancels_without_claiming_a_cap() {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut state = StreamState::new(false, false, Arc::clone(&cancel), None);

        state.stop_generation(StreamStopReason::SingleToolComplete);

        assert!(state.stop_string_triggered);
        assert!(!state.tool_loop_capped);
        assert_eq!(
            state.stop_reason,
            Some(StreamStopReason::SingleToolComplete)
        );
        assert!(cancel.load(Ordering::Acquire));
    }

    #[test]
    fn only_natural_and_client_stop_drain_safe_terminal_tails() {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut state = StreamState::new(false, false, Arc::clone(&cancel), None);
        assert!(state.should_flush_terminal_pipeline());
        state.stop_generation(StreamStopReason::ClientStop);
        assert!(state.should_flush_terminal_pipeline());

        state.stop_generation(StreamStopReason::ToolLoopOrInvalid);
        assert!(!state.should_flush_terminal_pipeline());
        assert_eq!(state.stop_reason, Some(StreamStopReason::ToolLoopOrInvalid));
    }

    #[test]
    fn hidden_reasoning_boundary_resets_decoder_before_next_content_token() {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut state = StreamState::new(false, true, Arc::clone(&cancel), None);
        state.all_toks.push(11); // visible reasoning token
        state.emitted = 3;
        state.decoded_text.push_str("why");
        state.detok_read_offset = 1;

        state.complete_thinking_phase();

        assert!(state.thinking_done);
        assert!(state.all_toks.is_empty());
        assert_eq!(state.emitted, 0);
        assert!(state.decoded_text.is_empty());
        assert_eq!(state.detok_read_offset, 0);

        // The next scheduler Token event starts a fresh content accumulator;
        // handle_token therefore takes its content branch, not reasoning.
        state.all_toks.push(21);
        assert_eq!(state.all_toks, [21]);
        assert!(state.thinking_done);
    }

    #[test]
    fn tool_handlers_cannot_bypass_central_stop_publication() {
        let handlers = include_str!("tool_handlers.rs");
        assert!(!handlers.contains("stop_string_triggered = true"));
        assert!(!handlers.contains("tool_loop_capped = true"));
        assert!(!handlers.contains("cancel_flag.store"));
        assert!(handlers.contains("stop_generation(StreamStopReason::ToolLoopOrInvalid)"));
    }
}
