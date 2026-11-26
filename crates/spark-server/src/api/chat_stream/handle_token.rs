// SPDX-License-Identifier: AGPL-3.0-only

//! Token-event decoding and client-visible SSE emission.
use crate::openai::ChatCompletionChunk;
use crate::tool_parser;

use super::super::failures::{check_loop_watchdog, flush_content_sanitizer_with_report};
use super::super::sanitizer::sanitize_content_chunk_with_report;
use super::ChunkVec;
use super::ctx::StreamCtx;
use super::state::{StreamState, StreamStopReason};
use super::tool_handlers::{
    admit_salvaged_tool_call, handle_complete_tool_call, handle_tool_call_delta,
    handle_tool_call_end, handle_tool_call_start,
};

#[inline]
fn has_wire_bytes(delta: &str) -> bool {
    !delta.is_empty()
}

fn incremental_text(state: &mut StreamState, ctx: &StreamCtx) -> String {
    match ctx.state.tokenizer.incremental_decode(
        &state.all_toks,
        &mut state.detok_prefix_offset,
        &mut state.detok_read_offset,
    ) {
        Ok(text) => text,
        Err(error) => {
            tracing::warn!(
                ?error,
                "incremental tokenizer decode failed; deferring to Done"
            );
            String::new()
        }
    }
}

fn sanitize_reasoning_delta(
    state: &mut StreamState,
    leak_markers: &tool_parser::LeakMarkers,
    mut cleaned: String,
) -> String {
    if !crate::api::chat_phases::agent_guards_enabled() {
        return cleaned;
    }
    let original = cleaned.clone();
    // Strip format tokens that shouldn't appear in thinking (opt-in agent guards).
    cleaned = cleaned.replace("<think>", "");
    if let Some(rest) = cleaned.strip_prefix("assistant\n") {
        cleaned = rest.to_string();
    } else if let Some(rest) = cleaned.strip_prefix("assistant") {
        cleaned = rest.to_string();
    }
    while let Some(start) = cleaned.find("<tool_call>") {
        if let Some(end) = cleaned[start..].find("</tool_call>") {
            cleaned = format!(
                "{}{}",
                &cleaned[..start],
                &cleaned[start + end + "</tool_call>".len()..]
            );
        } else {
            cleaned = cleaned[..start].to_string();
            break;
        }
    }
    if let Some(start) = cleaned.find("<function=") {
        cleaned = cleaned[..start].to_string();
    }
    for tag in &["</parameter>", "</function>", "</tool_call>"] {
        cleaned = cleaned.replace(tag, "");
    }
    for word in &["user", "assistant", "tool"] {
        let pair = format!("{word}{word}");
        while cleaned.contains(&pair) {
            cleaned = cleaned.replace(&pair, "");
        }
        let nl_form = format!("\n{word}\n");
        while cleaned.contains(&nl_form) {
            cleaned = cleaned.replace(&nl_form, "\n");
        }
    }
    let (sanitized, sanitizer_modified) = sanitize_content_chunk_with_report(
        &cleaned,
        &mut state.reasoning_tag_scan_buf,
        &mut state.reasoning_suppressing_leak,
        &mut state.reasoning_inside_envelope,
        leak_markers,
    );
    if cleaned != original || sanitizer_modified {
        state.record_intervention("content_output_repair", "sanitize_reasoning_delta");
    }
    sanitized
}

fn flush_reasoning_tail(
    state: &mut StreamState,
    leak_markers: &tool_parser::LeakMarkers,
) -> String {
    let (tail, modified) = flush_content_sanitizer_with_report(
        &mut state.reasoning_tag_scan_buf,
        &mut state.reasoning_suppressing_leak,
        leak_markers,
    );
    if modified {
        state.record_intervention("content_output_repair", "drop_partial_reasoning_markup");
    }
    tail
}

pub(super) fn dispatch_detector_batch(
    outputs: Vec<tool_parser::DetectorOutput>,
    mut dispatch: impl FnMut(tool_parser::DetectorOutput) -> bool,
) {
    for output in outputs {
        if dispatch(output) {
            break;
        }
    }
}

fn normalize_content_delta(state: &mut StreamState, mut delta: String) -> String {
    // Strip residual think tags from content after thinking is done.
    if state.thinking_done {
        for tag in &[
            "</think>",
            "</thinking>",
            "<thinking>",
            "</analysis>",
            "<analysis>",
        ] {
            while let Some(pos) = delta.find(tag) {
                delta = format!("{}{}", &delta[..pos], delta[pos + tag.len()..].trim_start());
            }
        }
        // If model re-opens <think>, suppress content from <think> onward.
        if let Some(pos) = delta.find("<think>") {
            delta = delta[..pos].to_string();
            state.thinking_done = false;
            state.all_toks.clear();
            state.emitted = 0;
            state.decoded_text.clear();
            state.detok_prefix_offset = 0;
            state.detok_read_offset = 0;
        }
    }

    // Role boundaries are scheduler control tokens, not ordinary role words.
    // Under ignore_eos, keep the same decoded text as the blocking response.
    delta
}

fn route_content_delta(state: &mut StreamState, ctx: &StreamCtx, delta: String) -> ChunkVec {
    let mut sse_events = Vec::new();
    let original_delta = delta.clone();
    // Only run heuristic post-think tag stripping / `<think>` re-entry when
    // FLUX_AGENT_GUARDS=1; by default literal tags in prose/code pass verbatim.
    let mut delta = if ctx.agent_guards {
        normalize_content_delta(state, delta)
    } else {
        delta
    };
    if delta != original_delta {
        state.record_intervention("content_output_repair", "sanitize_content_delta");
    }
    if delta.is_empty() {
        return sse_events;
    }

    // Hold back any trailing proper prefix of a configured stop before
    // detector/sanitizer processing.
    delta = filter_stream_content(state, ctx, &delta);
    if delta.is_empty() {
        return sse_events;
    }

    if state.detector.is_some() {
        let (outputs, parser_recoveries) = {
            let det = state.detector.as_mut().expect("detector is Some");
            let outputs = det.process(&delta);
            let parser_recoveries = det.take_recovery_count();
            (outputs, parser_recoveries)
        };
        state.record_parser_recoveries(parser_recoveries);
        dispatch_detector_batch(outputs, |output| {
            match output {
                tool_parser::DetectorOutput::Content(text) => {
                    if let Some(events_out) = detector_content_arm(state, ctx, &text) {
                        sse_events.extend(events_out);
                    }
                }
                tool_parser::DetectorOutput::ToolCall(mut tc, tc_idx) => {
                    handle_complete_tool_call(state, ctx, &mut tc, tc_idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallStart {
                    id: tc_id,
                    name,
                    idx,
                } => {
                    handle_tool_call_start(state, ctx, tc_id, name, idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallDelta { args, idx } => {
                    handle_tool_call_delta(state, ctx, args, idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                    handle_tool_call_end(state, ctx, idx);
                }
            }
            state.stop_string_triggered
                && !matches!(state.stop_reason, Some(StreamStopReason::ClientStop))
        });
    } else {
        let (sanitized, modified) = sanitize_content_chunk_with_report(
            &delta,
            &mut state.tag_scan_buf,
            &mut state.suppressing_param_leak,
            &mut state.inside_envelope,
            &ctx.leak_markers,
        );
        if modified {
            state.record_intervention("content_output_repair", "sanitize_tool_markup");
        }
        if let Some(events_out) = process_detector_content(state, ctx, &sanitized) {
            sse_events.extend(events_out);
        }
    }

    sse_events
}

/// Process one token. Returns the SSE events to forward to the
/// client (empty `Vec` is valid).
pub(super) fn handle_token(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> ChunkVec {
    // Counted HERE, as it is produced, so the dashboard's tok/s is a live
    // rate rather than one spike at completion.
    crate::metrics::DECODED_TOKENS_TOTAL.inc();
    let mut sse_events: ChunkVec = Vec::new();
    if state.stop_string_triggered {
        return sse_events;
    }
    state.all_toks.push(tok);

    // ── Thinking-phase: token-ID based </think> detection ────────────
    if !state.thinking_done {
        if let Some(end_id) = ctx.state.think_end_token_id
            && tok == end_id
        {
            // The delimiter itself is structural. Both a literal close and a
            // scheduler `ThinkingDone` event enter the shared phase transition
            // with only visible reasoning tokens in the accumulator.
            let removed = state.all_toks.pop();
            debug_assert_eq!(removed, Some(tok));
            return handle_thinking_done(state, ctx);
        }
        // Still in thinking — accumulate but don't emit as content
        if ctx.enable_thinking {
            // Derive streaming bytes from the same full-decode contract used by
            // blocking responses.  This preserves standalone spaces/newlines
            // at token boundaries while retaining O(n) decode work.
            let stable = incremental_text(state, ctx);
            state.decoded_text.push_str(&stable);
            let stable_end = state.decoded_text.len();
            if stable_end <= state.emitted {
                return sse_events;
            }
            let delta = state.decoded_text[state.emitted..stable_end].to_string();
            state.emitted = stable_end;
            if !delta.is_empty() {
                let cleaned = sanitize_reasoning_delta(state, &ctx.leak_markers, delta);
                // Whitespace-only deltas are semantically significant stream
                // boundaries.  `emitted` already advanced, so dropping one here
                // would permanently glue neighboring words or lines together.
                if has_wire_bytes(&cleaned) {
                    let chunk = ChatCompletionChunk::reasoning_chunk(&ctx.model, &ctx.id, cleaned);
                    sse_events.push(chunk);
                }
            }
        }
        return sse_events;
    }

    // ── Content phase: incremental full-decode + stable slice ─────────
    let stable = incremental_text(state, ctx);
    state.decoded_text.push_str(&stable);
    let stable_end = state.decoded_text.len();
    if stable_end <= state.emitted {
        return sse_events;
    }
    let delta = state.decoded_text[state.emitted..stable_end].to_string();
    state.emitted = stable_end;
    sse_events.extend(route_content_delta(state, ctx, delta));
    sse_events
}

/// Complete a reasoning phase which has no client-visible delimiter. This is
/// also the shared implementation for a literal `</think>` after that control
/// token has been removed from the detokenizer accumulator.
pub(super) fn handle_thinking_done(state: &mut StreamState, ctx: &StreamCtx) -> ChunkVec {
    if state.thinking_done {
        return Vec::new();
    }

    let residual = if state.all_toks.is_empty() {
        Ok(String::new())
    } else {
        ctx.state
            .tokenizer
            .terminal_decode_residual(&state.all_toks, state.emitted)
    };
    finish_thinking_phase(
        state,
        ctx.enable_thinking,
        &ctx.leak_markers,
        &ctx.model,
        &ctx.id,
        residual,
    )
}

fn finish_thinking_phase(
    state: &mut StreamState,
    enable_thinking: bool,
    leak_markers: &tool_parser::LeakMarkers,
    model: &str,
    id: &str,
    residual: anyhow::Result<String>,
) -> ChunkVec {
    // Emit only reasoning bytes not already sent incrementally (including an
    // incomplete UTF-8 tail), then drain the leak sanitizer's ordinary tail.
    let mut final_reasoning = String::new();
    if enable_thinking {
        match residual {
            Ok(residual) => {
                final_reasoning.push_str(&sanitize_reasoning_delta(state, leak_markers, residual));
            }
            Err(error) => {
                tracing::warn!(
                    ?error,
                    "reasoning phase-boundary tokenizer residual decode failed"
                );
            }
        }
        final_reasoning.push_str(&flush_reasoning_tail(state, leak_markers));
    }

    state.complete_thinking_phase();
    if has_wire_bytes(&final_reasoning) {
        vec![ChatCompletionChunk::reasoning_chunk(
            model,
            id,
            final_reasoning,
        )]
    } else {
        Vec::new()
    }
}

/// Drain bytes that incremental decoding intentionally withheld until the
/// terminal boundary.  The residual goes through the same reasoning or
/// content pipeline as an ordinary token delta; this is what makes a
/// max-tokens cut in the middle of a UTF-8 sequence match blocking decode.
pub(super) fn flush_terminal_decode(state: &mut StreamState, ctx: &StreamCtx) -> ChunkVec {
    if !state.should_flush_terminal_pipeline() || state.all_toks.is_empty() {
        return Vec::new();
    }
    let residual = match ctx
        .state
        .tokenizer
        .terminal_decode_residual(&state.all_toks, state.emitted)
    {
        Ok(text) => text,
        Err(error) => {
            tracing::warn!(?error, "terminal tokenizer residual decode failed");
            return Vec::new();
        }
    };
    state.decoded_text.push_str(&residual);
    state.emitted = state.decoded_text.len();
    state.detok_prefix_offset = state.detok_read_offset;
    state.detok_read_offset = state.all_toks.len();

    if !state.thinking_done {
        if !ctx.enable_thinking {
            return Vec::new();
        }
        let mut reasoning = sanitize_reasoning_delta(state, &ctx.leak_markers, residual);
        reasoning.push_str(&flush_reasoning_tail(state, &ctx.leak_markers));
        if has_wire_bytes(&reasoning) {
            return vec![ChatCompletionChunk::reasoning_chunk(
                &ctx.model, &ctx.id, reasoning,
            )];
        }
        return Vec::new();
    }

    route_content_delta(state, ctx, residual)
}

/// Common processing for a sanitized content chunk: SimHash semantic
/// guard, token-level loop watchdog, salvage on trip, otherwise
/// emit a `content_chunk`. Returns `Some(events)` whenever it emits wire
/// output. Callers must inspect `state.stop_string_triggered` rather than
/// treating ordinary content as a reason to drop later detector outputs from
/// the same batch.
///
/// Note: when called from the detector-active branch, `sanitized`
/// has already been routed through `sanitize_content_chunk`. When
/// called from the no-detector branch, the caller must pre-sanitize
/// (the no-detector path uses the same sanitizer state).
pub(super) fn process_detector_content(
    state: &mut StreamState,
    ctx: &StreamCtx,
    sanitized_or_raw: &str,
) -> Option<ChunkVec> {
    // From the detector-active branch the input is the Content(text)
    // payload that still needs sanitization. From the no-detector
    // branch the input is already sanitized. Distinguish via a thin
    // wrapper: detector branch ALSO sanitizes; non-detector branch
    // skips by passing the already-sanitized text. To keep the call
    // site simple, we sanitize here only when the input contains the
    // hallmark of an unfiltered Content payload — which we can't
    // reliably detect. Solution: split into two paths.
    //
    // Inlining: this helper is only called once per branch with the
    // correct input type; it never re-sanitizes. The parameter is the
    // post-sanitizer text in both call sites.
    let sanitized = sanitized_or_raw;

    let allow_loop_watchdog = !ctx.ignore_eos
        && state.all_toks.len() >= ctx.min_tokens
        && !ctx.state.disable_simhash_watchdog
        && crate::scheduler::enable_loop_watchdog();

    // F4 SimHash guard.
    let semantic_trip = if allow_loop_watchdog && !state.loop_watchdog_triggered {
        state.simhash_pending.push_str(sanitized);
        let mut dup = false;
        if crate::loop_simhash::ends_at_sentence_boundary(&state.simhash_pending).is_some()
            || state.simhash_pending.len() >= 1024
        {
            dup = state.simhash_guard.check(&state.simhash_pending);
            state.simhash_pending.clear();
        }
        dup
    } else {
        false
    };

    let token_trip = allow_loop_watchdog
        && check_loop_watchdog(
            sanitized,
            &mut state.loop_scan_buf,
            state.loop_watchdog_triggered,
        );

    if semantic_trip || token_trip {
        if semantic_trip {
            tracing::warn!(
                ring_len = state.simhash_guard.len(),
                "SimHash semantic-loop watchdog fired (paraphrased sentence repeat)"
            );
            state.record_intervention("F4", "cancel_generation");
        }
        if token_trip {
            state.record_intervention("api_repetition_watchdog", "cancel_generation");
        }
        state.stop_generation(StreamStopReason::ContentLoop);

        let salvaged =
            crate::tool_salvage::salvage(&state.loop_scan_buf, &ctx.tool_defs_for_backfill);
        let mut events: ChunkVec = Vec::new();
        let mut emitted_salvage = false;
        for (idx, mut tc) in salvaged.into_iter().enumerate() {
            if !admit_salvaged_tool_call(state, ctx, &mut tc) {
                break;
            }
            tracing::warn!(
                tool = %tc.function.name,
                block_index = idx,
                "watchdog salvage: emitting synthetic tool_call",
            );
            let start = ChatCompletionChunk::tool_call_start_chunk(&ctx.model, &ctx.id, &tc, idx);
            events.push(start);
            let frag = ChatCompletionChunk::tool_call_args_fragment(
                &ctx.model,
                &ctx.id,
                idx,
                &tc.function.arguments,
            );
            events.push(frag);
            emitted_salvage = true;
            if !ctx.allow_parallel_tool_calls {
                state.stop_generation(StreamStopReason::SingleToolComplete);
                break;
            }
        }
        if emitted_salvage {
            state.salvaged_tool_call = true;
            state.record_intervention("watchdog_tool_salvage", "emit_synthetic_tool_call");
        }
        return Some(events);
    }

    if has_wire_bytes(sanitized) {
        let out = if let Some(stripper) = state.json_fence_stripper.as_mut() {
            stripper.push(sanitized)
        } else {
            sanitized.to_string()
        };
        if has_wire_bytes(&out) {
            if state.refusal_scan_buf.len() < 16_384 {
                state.refusal_scan_buf.push_str(&out);
            }
            let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, out);
            let events: ChunkVec = vec![chunk];
            return Some(events);
        }
    }
    None
}

/// Detector-active branch's `Content(text)` arm: sanitize first,
/// then run the shared semantic/token watchdog + emit pipeline.
fn detector_content_arm(state: &mut StreamState, ctx: &StreamCtx, text: &str) -> Option<ChunkVec> {
    let (sanitized, modified) = sanitize_content_chunk_with_report(
        text,
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &mut state.inside_envelope,
        &ctx.leak_markers,
    );
    if modified {
        state.record_intervention("content_output_repair", "sanitize_tool_markup");
    }
    process_detector_content(state, ctx, &sanitized)
}

/// Return content that cannot still become part of a stop string. A
/// trailing prefix is retained across deltas, preventing irreversible
/// partial-marker emission on the SSE wire.
pub(super) fn filter_stream_content(
    state: &mut StreamState,
    ctx: &StreamCtx,
    content: &str,
) -> String {
    if state.stop_string_triggered {
        return String::new();
    }
    let (safe, matched) = filter_stop_delta(&mut state.stop_holdback, content, &ctx.stop_strings);
    if matched {
        state.stop_generation(StreamStopReason::ClientStop);
    }
    safe
}

pub(in crate::api) fn filter_stop_delta(
    holdback: &mut String,
    delta: &str,
    stops: &[String],
) -> (String, bool) {
    let mut candidate = std::mem::take(holdback);
    candidate.push_str(delta);

    if let Some(stop_at) = stops
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| candidate.find(stop))
        .min()
    {
        candidate.truncate(stop_at);
        return (candidate, true);
    }

    let mut held_bytes = 0;
    for stop in stops.iter().filter(|stop| !stop.is_empty()) {
        for (start, _) in candidate.char_indices() {
            let suffix = &candidate[start..];
            if suffix.len() < stop.len() && stop.starts_with(suffix) {
                held_bytes = held_bytes.max(suffix.len());
            }
        }
    }
    let safe_len = candidate.len() - held_bytes;
    *holdback = candidate[safe_len..].to_string();
    (candidate[..safe_len].to_string(), false)
}

#[cfg(test)]
mod content_delta_tests {
    use super::normalize_content_delta;
    use crate::api::chat_stream::state::StreamState;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    fn content_phase_state() -> StreamState {
        StreamState::new(false, false, Arc::new(AtomicBool::new(false)), None)
    }

    #[test]
    fn ordinary_role_words_survive_individual_and_coalesced_deltas() {
        let mut state = content_phase_state();
        let deltas = [
            "ask the",
            " user",
            " and the",
            " tool",
            " then",
            " assistant",
            ".",
        ];
        let streamed: String = deltas
            .into_iter()
            .map(|delta| normalize_content_delta(&mut state, delta.to_string()))
            .collect();
        assert_eq!(streamed, "ask the user and the tool then assistant.");
        for text in [
            "user",
            "tool",
            "assistant",
            "user\n",
            " tool ",
            "\u{FFFD}user",
        ] {
            assert_eq!(normalize_content_delta(&mut state, text.to_string()), text);
        }
    }

    #[test]
    fn actual_thinking_tags_keep_the_existing_phase_transition() {
        let mut state = content_phase_state();
        assert_eq!(
            normalize_content_delta(&mut state, "</think>\n\nHello user".to_string()),
            "Hello user"
        );
        assert!(state.thinking_done);
        state.all_toks.push(7);
        assert_eq!(
            normalize_content_delta(&mut state, "done<think>secret".to_string()),
            "done"
        );
        assert!(!state.thinking_done);
        assert!(state.all_toks.is_empty());
        assert_eq!(state.emitted, 0);
        assert_eq!(
            normalize_content_delta(&mut state, "</think>user".to_string()),
            "</think>user"
        );
    }
}

#[cfg(test)]
mod stop_prefix_tests {
    use super::{filter_stop_delta, finish_thinking_phase, has_wire_bytes};
    use crate::api::chat_stream::state::StreamState;
    use crate::api::failures::flush_content_sanitizer;
    use crate::api::sanitizer::sanitize_content_chunk;
    use crate::tool_parser::{Qwen3CoderParser, ToolCallParser};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn reasoning_and_content_publish_whitespace_only_deltas() {
        assert!(has_wire_bytes(" "));
        assert!(has_wire_bytes("\n\n"));
        assert!(!has_wire_bytes(""));
    }

    #[test]
    fn hidden_boundary_flushes_reasoning_then_arms_content_phase() {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut state = StreamState::new(false, true, cancel, None);
        state.all_toks.push(11);

        let chunks = finish_thinking_phase(
            &mut state,
            true,
            &crate::tool_parser::LeakMarkers::EMPTY,
            "model",
            "id",
            Ok("reason ".to_string()),
        );

        assert!(state.thinking_done);
        assert!(state.all_toks.is_empty());
        let wire = serde_json::to_value(&chunks[0]).unwrap();
        assert_eq!(wire["choices"][0]["delta"]["reasoning_content"], "reason ");
        assert!(wire["choices"][0]["delta"].get("content").is_none());
    }

    #[test]
    fn qwen_reasoning_phase_boundary_flushes_ordinary_tail_exactly() {
        let markers = Qwen3CoderParser.leak_markers();
        let mut scan = String::new();
        let mut suppressing = false;
        let mut inside = false;
        let input = "reasoning ends with ordinary text and space ";
        let mut output =
            sanitize_content_chunk(input, &mut scan, &mut suppressing, &mut inside, &markers);
        output.push_str(&flush_content_sanitizer(
            &mut scan,
            &mut suppressing,
            &markers,
        ));
        assert_eq!(output, input);
        let source = include_str!("handle_token.rs");
        assert!(source.contains("final_reasoning.push_str(&flush_reasoning_tail("));
    }

    #[test]
    fn qwen_safe_prose_before_split_client_stop_is_not_lost() {
        let markers = Qwen3CoderParser.leak_markers();
        let mut held = String::new();
        let mut scan = String::new();
        let mut suppressing = false;
        let mut inside = false;
        let stops = ["<STOP>".to_string()];

        let (safe, matched) = filter_stop_delta(&mut held, "hello world<ST", &stops);
        assert!(!matched);
        let mut output =
            sanitize_content_chunk(&safe, &mut scan, &mut suppressing, &mut inside, &markers);
        let (safe, matched) = filter_stop_delta(&mut held, "OP>ignored", &stops);
        assert!(matched);
        output.push_str(&sanitize_content_chunk(
            &safe,
            &mut scan,
            &mut suppressing,
            &mut inside,
            &markers,
        ));
        output.push_str(&flush_content_sanitizer(
            &mut scan,
            &mut suppressing,
            &markers,
        ));
        assert_eq!(output, "hello world");
    }

    #[test]
    fn split_stop_prefix_is_never_emitted() {
        let stops = vec!["<STOP>".to_string()];
        let mut held = String::new();
        assert_eq!(
            filter_stop_delta(&mut held, "answer<ST", &stops),
            ("answer".into(), false)
        );
        assert_eq!(held, "<ST");
        assert_eq!(
            filter_stop_delta(&mut held, "OP>leak", &stops),
            (String::new(), true)
        );
        assert!(held.is_empty());
    }

    #[test]
    fn overlap_unicode_and_incomplete_final_prefix_are_preserved() {
        let stops = vec!["abab".to_string(), "🛑終".to_string()];
        let mut held = String::new();
        assert_eq!(
            filter_stop_delta(&mut held, "xaba", &stops),
            ("x".into(), false)
        );
        assert_eq!(held, "aba");
        assert_eq!(
            filter_stop_delta(&mut held, "x🛑", &stops),
            ("abax".into(), false)
        );
        assert_eq!(held, "🛑");
        assert_eq!(
            filter_stop_delta(&mut held, "終tail", &stops),
            (String::new(), true)
        );
        assert!(held.is_empty());
    }

    #[test]
    fn first_repeated_occurrence_wins_and_both_streams_share_the_filter() {
        let mut held = String::new();
        assert_eq!(
            filter_stop_delta(&mut held, "a<X>b<X>c", &["<X>".to_string()]),
            ("a".into(), true)
        );

        let completion = include_str!("../completions.rs");
        assert!(completion.contains("super::chat_stream::filter_stop_delta("));
        assert!(completion.contains("std::mem::take(&mut stop_holdback)"));
        assert!(completion.contains("cancel_guard.cancel();"));

        let done = include_str!("handle_done.rs");
        assert!(done.contains("std::mem::take(&mut state.stop_holdback)"));
        assert!(done.contains("det.process(&incomplete_prefix)"));
    }
}
