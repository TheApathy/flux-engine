// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Done { ... }` arm of the streaming `flat_map`
// closure (originally ~396 LoC).

use crate::openai::{ChatCompletionChunk, Usage};
use crate::tool_parser;

use super::super::failures::flush_content_sanitizer_with_report;
use super::super::sanitizer::sanitize_content_chunk_with_report;
use super::ChunkVec;
use super::completion_token_ids::{terminal_chunks, validate_or_cancel};
use super::ctx::StreamCtx;
use super::handle_token::{flush_terminal_decode, process_detector_content};
use super::state::{StreamState, StreamStopReason};
use super::tool_handlers::{
    admit_salvaged_tool_call, handle_complete_tool_call, handle_tool_call_delta,
    handle_tool_call_end, handle_tool_call_start,
};

fn resolved_finish_reason(
    finish_reason: &str,
    tool_loop_capped: bool,
    has_tool_calls: bool,
    loop_watchdog_triggered: bool,
) -> &str {
    if tool_loop_capped {
        // Tool loops must remain visibly truncated so agent clients do not
        // execute another tool round and perpetuate the outer loop.
        "length"
    } else if has_tool_calls {
        "tool_calls"
    } else if loop_watchdog_triggered {
        // A content-loop guard is a deliberate server stop, not exhaustion of
        // the request's token budget. Reporting `length` makes benchmark
        // clients retry the same degenerate response with a larger allowance.
        "stop"
    } else {
        finish_reason
    }
}

/// The guard behind a choice finish, reported as `flux_stop_reason`.
/// Timeout causes (`"request_timeout_stall"` / `"request_timeout_total"`) are
/// returned even on `"length"` finishes; non-timeout guards (the scheduler's
/// cause, else the stream's content-loop `"watchdog"`) are returned only when
/// `finish_reason == "stop"`.
fn guard_stop_reason(
    finish_reason: &str,
    scheduler_cause: Option<&'static str>,
    loop_watchdog_triggered: bool,
) -> Option<&'static str> {
    if let Some(cause) =
        scheduler_cause.filter(|c| crate::api::request_contract::is_timeout_cause(c))
    {
        return Some(cause);
    }
    if finish_reason != "stop" {
        return None;
    }
    scheduler_cause.or(loop_watchdog_triggered.then_some("watchdog"))
}

/// Attach the request's warnings (clamp, timeout) to the finish-bearing
/// terminal chunk(s); other chunks and warning-free requests are unchanged.
fn with_terminal_warnings(
    terminal: Vec<ChatCompletionChunk>,
    contract_warnings: &[String],
    flux_stop_reason: Option<&'static str>,
) -> Vec<ChatCompletionChunk> {
    let warnings: Vec<String> = contract_warnings
        .iter()
        .cloned()
        .chain(crate::api::request_contract::timeout_warning(
            flux_stop_reason,
        ))
        .collect();
    if warnings.is_empty() {
        return terminal;
    }
    terminal
        .into_iter()
        .map(|chunk| {
            if chunk.choices.iter().any(|c| c.finish_reason.is_some()) {
                ChatCompletionChunk {
                    warnings: Some(warnings.clone()),
                    ..chunk
                }
            } else {
                chunk
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_done(
    state: &mut StreamState,
    ctx: &StreamCtx,
    finish_reason: String,
    completion_token_ids: Vec<u32>,
    completion_tokens: usize,
    time_to_first_token_ms: f64,
    decode_time_ms: f64,
    reasoning_tokens: u32,
    cached_prompt_tokens: u32,
    think_nudges: u32,
    stop_cause: Option<&'static str>,
) -> Result<ChunkVec, String> {
    let mut sse_events: ChunkVec = Vec::new();

    // Validate the scheduler snapshot before emitting any final buffered text,
    // usage, or finish frame. Usage remains scheduler-owned and is never
    // rewritten to conceal a transport mismatch.
    validate_or_cancel(
        &state.cancel_flag,
        ctx.req_return_token_ids,
        completion_tokens,
        &completion_token_ids,
    )?;

    // Incremental detokenization withholds incomplete UTF-8 until it becomes
    // stable.  At a natural/max-token terminal boundary, drain the ordinary
    // blocking-decode residual through the same reasoning/content pipeline.
    sse_events.extend(flush_terminal_decode(state, ctx));

    // A natural completion can end on a proper prefix of a configured
    // stop string. Since no later delta can complete it, feed that raw
    // decoded suffix through the ordinary detector/sanitizer pipeline.
    let incomplete_prefix = if state.should_flush_terminal_pipeline() {
        std::mem::take(&mut state.stop_holdback)
    } else {
        String::new()
    };

    // ── Detector flush ──────────────────────────────────────────────
    if state.should_flush_terminal_pipeline() && state.detector.is_some() {
        let (outputs, parser_recoveries) = {
            let det = state.detector.as_mut().expect("detector is Some");
            let mut outputs = if incomplete_prefix.is_empty() {
                Vec::new()
            } else {
                det.process(&incomplete_prefix)
            };
            outputs.extend(det.flush());
            let parser_recoveries = det.take_recovery_count();
            (outputs, parser_recoveries)
        };
        state.record_parser_recoveries(parser_recoveries);
        for output in outputs {
            match output {
                tool_parser::DetectorOutput::Content(text) => {
                    let (sanitized, modified) = sanitize_content_chunk_with_report(
                        &text,
                        &mut state.tag_scan_buf,
                        &mut state.suppressing_param_leak,
                        &mut state.inside_envelope,
                        &ctx.leak_markers,
                    );
                    if modified {
                        state.record_intervention("content_output_repair", "sanitize_tool_markup");
                    }
                    if let Some(events) = process_detector_content(state, ctx, &sanitized) {
                        sse_events.extend(events);
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
            if state.stop_string_triggered
                && !matches!(state.stop_reason, Some(StreamStopReason::ClientStop))
            {
                break;
            }
        }
    } else if !incomplete_prefix.is_empty() {
        let (sanitized, modified) = sanitize_content_chunk_with_report(
            &incomplete_prefix,
            &mut state.tag_scan_buf,
            &mut state.suppressing_param_leak,
            &mut state.inside_envelope,
            &ctx.leak_markers,
        );
        if modified {
            state.record_intervention("content_output_repair", "sanitize_tool_markup");
        }
        if let Some(events) = process_detector_content(state, ctx, &sanitized) {
            sse_events.extend(events);
        }
    }

    // ── Sanitizer tail flush ────────────────────────────────────────
    if state.should_flush_terminal_pipeline() {
        let (tail, modified) = flush_content_sanitizer_with_report(
            &mut state.tag_scan_buf,
            &mut state.suppressing_param_leak,
            &ctx.leak_markers,
        );
        if modified {
            state.record_intervention("content_output_repair", "drop_partial_tool_markup");
        }
        if let Some(events) = process_detector_content(state, ctx, &tail) {
            sse_events.extend(events);
        }
        if let Some(stripper) = state.json_fence_stripper.as_mut() {
            let remainder = stripper.flush();
            if !remainder.is_empty() {
                if state.refusal_scan_buf.len() < 16_384 {
                    state.refusal_scan_buf.push_str(&remainder);
                }
                sse_events.push(ChatCompletionChunk::content_chunk(
                    &ctx.model, &ctx.id, remainder,
                ));
            }
        }
    }

    // ── Usage block ─────────────────────────────────────────────────
    let tps = if decode_time_ms > 0.0 {
        completion_tokens.saturating_sub(1) as f64 / (decode_time_ms / 1000.0)
    } else {
        0.0
    };
    let usage = Usage {
        prompt_tokens: ctx.prompt_len,
        completion_tokens,
        total_tokens: ctx.prompt_len + completion_tokens,
        prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
            cached_tokens: cached_prompt_tokens as usize,
            audio_tokens: 0,
        }),
        completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
            reasoning_tokens: reasoning_tokens as usize,
            audio_tokens: 0,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
            think_nudges,
        }),
        time_to_first_token_ms,
        response_tokens_per_second: tps,
    };

    // ── Last-resort tool salvage (opt-in via FLUX_AGENT_GUARDS=1) ───
    if crate::api::chat_phases::agent_guards_enabled()
        && !state.stop_string_triggered
        && !state.salvaged_tool_call
        && !state.detector.as_ref().is_some_and(|d| d.has_tool_calls())
    {
        let salvaged =
            crate::tool_salvage::salvage(&state.refusal_scan_buf, &ctx.tool_defs_for_backfill);
        let mut emitted_salvage = false;
        for (idx, mut tc) in salvaged.into_iter().enumerate() {
            if !admit_salvaged_tool_call(state, ctx, &mut tc) {
                break;
            }
            tracing::warn!(
                tool = %tc.function.name,
                block_index = idx,
                "tool_salvage: emitting synthetic tool_call from prose",
            );
            let start = ChatCompletionChunk::tool_call_start_chunk(&ctx.model, &ctx.id, &tc, idx);
            sse_events.push(start);
            let frag = ChatCompletionChunk::tool_call_args_fragment(
                &ctx.model,
                &ctx.id,
                idx,
                &tc.function.arguments,
            );
            sse_events.push(frag);
            emitted_salvage = true;
            if !ctx.allow_parallel_tool_calls {
                state.stop_generation(StreamStopReason::SingleToolComplete);
                break;
            }
        }
        if emitted_salvage {
            state.salvaged_tool_call = true;
            state.record_intervention("tool_salvage", "emit_synthetic_tool_call");
        }
    }

    let effective_finish = if matches!(state.stop_reason, Some(StreamStopReason::ClientStop)) {
        "stop"
    } else {
        finish_reason.as_str()
    };
    let fr = resolved_finish_reason(
        effective_finish,
        state.tool_loop_capped,
        state.detector.as_ref().is_some_and(|d| d.has_tool_calls()) || state.salvaged_tool_call,
        state.loop_watchdog_triggered,
    );
    let flux_stop_reason = guard_stop_reason(fr, stop_cause, state.loop_watchdog_triggered);

    // Refusal classification (opt-in via FLUX_AGENT_GUARDS=1).
    let refusal_signal = if crate::api::chat_phases::agent_guards_enabled()
        && !state.salvaged_tool_call
        && state.detector.as_ref().is_none_or(|d| !d.has_tool_calls())
    {
        crate::refusal::detect(&state.refusal_scan_buf)
    } else {
        None
    };
    if let Some(ref r) = refusal_signal {
        state.record_intervention("response_classification", "publish_refusal_field");
        let chunk = ChatCompletionChunk::refusal_chunk(&ctx.model, &ctx.id, r.clone());
        sse_events.push(chunk);
    }

    let usage_for_dump = usage.clone();
    // The terminal choice is the sole token-ID owner. Text chunks remain
    // unchanged, while opted-in clients receive the exact scheduler sequence.
    let terminal = terminal_chunks(
        &ctx.model,
        &ctx.id,
        fr,
        usage,
        ctx.req_stream_include_usage,
        ctx.req_return_token_ids,
        completion_token_ids,
        crate::openai::snapshot_shared_interventions(ctx.intervention_ledger.as_ref()),
        ctx.rendered_prompt_evidence.clone(),
        flux_stop_reason,
    )?;
    sse_events.extend(with_terminal_warnings(
        terminal,
        &ctx.contract_warnings,
        flux_stop_reason,
    ));

    // Metrics.
    if !ctx
        .active_guard
        .swap(true, std::sync::atomic::Ordering::SeqCst)
    {
        crate::metrics::REQUESTS_ACTIVE.dec();
    }
    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(ctx.prompt_len as u64);
    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(completion_tokens as u64);
    crate::metrics::TTFT_SECONDS.observe(time_to_first_token_ms / 1000.0);

    // Rate-limit true-up.
    if let Some(ref rctx) = ctx.req_ctx {
        let actual = (ctx.prompt_len + completion_tokens) as u64;
        let refund = rctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            ctx.state.rate_limiter.refund_tokens(&rctx.identity, refund);
        }
    }

    // --dump synthesized response entry.
    if let (Some(seq), Some(dump)) = (ctx.dump_seq, ctx.state.dump_writer.as_ref()) {
        let has_tool_calls =
            state.detector.as_ref().is_some_and(|d| d.has_tool_calls()) || state.salvaged_tool_call;
        let body = serde_json::json!({
            "id": ctx.id,
            "model": ctx.model,
            "object": "chat.completion.synthesized",
            "finish_reason": fr,
            "content": state.refusal_scan_buf,
            "has_tool_calls": has_tool_calls,
            "usage": usage_for_dump,
            "stop_string_triggered": state.stop_string_triggered,
            "loop_watchdog_triggered": state.loop_watchdog_triggered,
            "tool_loop_capped": state.tool_loop_capped,
            "_note": "Synthesized from post-sanitizer accumulators; \
                      per-chunk capture is a follow-up.",
        });
        dump.dump_response("/v1/chat/completions", seq, &body, true);
    }

    Ok(sse_events)
}

#[cfg(test)]
mod finish_reason_tests {
    use super::resolved_finish_reason;

    #[test]
    fn content_loop_is_a_stop_not_a_retryable_token_limit() {
        assert_eq!(resolved_finish_reason("length", false, false, true), "stop");
    }

    #[test]
    fn tool_protocol_reasons_keep_precedence() {
        assert_eq!(resolved_finish_reason("length", true, true, true), "length");
        assert_eq!(
            resolved_finish_reason("stop", true, false, false),
            "length",
            "an invalid or capped tool turn stays visibly truncated even when no call was published"
        );
        assert_eq!(
            resolved_finish_reason("length", false, true, true),
            "tool_calls",
            "parallel_tool_calls=false still completes its one valid call normally"
        );
    }

    #[test]
    fn ordinary_completion_preserves_scheduler_reason() {
        assert_eq!(
            resolved_finish_reason("length", false, false, false),
            "length"
        );
        assert_eq!(resolved_finish_reason("stop", false, false, false), "stop");
    }

    #[test]
    fn only_stop_finishes_name_a_guard() {
        use super::guard_stop_reason;
        let catastrophic = Some("catastrophic_guard");
        assert_eq!(guard_stop_reason("stop", catastrophic, false), catastrophic);
        assert_eq!(guard_stop_reason("stop", None, true), Some("watchdog"));
        assert_eq!(guard_stop_reason("stop", catastrophic, true), catastrophic);
        assert_eq!(guard_stop_reason("stop", None, false), None);
        assert_eq!(guard_stop_reason("length", Some("watchdog"), true), None);
        assert_eq!(
            guard_stop_reason("tool_calls", Some("watchdog"), false),
            None
        );
    }
}

#[cfg(test)]
mod contract_warning_tests {
    use super::{ChatCompletionChunk, guard_stop_reason, with_terminal_warnings};
    use crate::api::request_contract::TIMEOUT_STALL_CAUSE;

    fn usage() -> crate::openai::Usage {
        crate::openai::Usage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            prompt_tokens_details: None,
            completion_tokens_details: None,
            time_to_first_token_ms: 0.0,
            response_tokens_per_second: 0.0,
        }
    }

    #[test]
    fn a_timeout_cause_is_reported_even_on_a_length_finish() {
        assert_eq!(
            guard_stop_reason("length", Some(TIMEOUT_STALL_CAUSE), false),
            Some(TIMEOUT_STALL_CAUSE)
        );
        assert_eq!(guard_stop_reason("length", Some("watchdog"), false), None);
    }

    #[test]
    fn warnings_land_only_on_the_finish_bearing_chunk() {
        let terminal = vec![
            ChatCompletionChunk::final_chunk_no_usage("m", "id", "length"),
            ChatCompletionChunk::usage_only_chunk("m", "id", usage()),
        ];
        let out = with_terminal_warnings(
            terminal,
            &["max_tokens_clamped: x".to_string()],
            Some(TIMEOUT_STALL_CAUSE),
        );
        let w = out[0].warnings.as_ref().expect("final chunk warnings");
        assert_eq!(w.len(), 2);
        assert_eq!(w[0], "max_tokens_clamped: x");
        assert!(w[1].starts_with(TIMEOUT_STALL_CAUSE));
        assert!(out[1].warnings.is_none(), "usage-only chunk unchanged");
    }

    #[test]
    fn no_warnings_leaves_terminal_chunks_untouched() {
        let out = with_terminal_warnings(
            vec![ChatCompletionChunk::final_chunk_no_usage("m", "id", "stop")],
            &[],
            None,
        );
        assert!(out[0].warnings.is_none());
    }
}
