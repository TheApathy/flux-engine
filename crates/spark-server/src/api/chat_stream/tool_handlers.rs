// SPDX-License-Identifier: AGPL-3.0-only
//
// Helpers for the four `DetectorOutput` variants emitted by the
// streaming tool-call detector. Shared by both `handle_token` (mid-
// stream `process()` outputs) and `handle_done` (end-of-stream
// `flush()` outputs).

use crate::openai::ChatCompletionChunk;
use crate::tool_parser;

use super::super::failures::{
    F39FailureCache, bump_f12_tool_call_count, f44_check_permanent_failure,
    flush_content_sanitizer_with_report,
};
use super::ChunkVec;
use super::ctx::StreamCtx;
use super::state::{StreamState, StreamStopReason};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ForcedToolStop {
    Invalid,
    CrossResponseDedup,
    WithinResponseDedup,
    PermanentFailure,
    SameNameCap,
    TotalCap,
}

impl ForcedToolStop {
    fn code(self) -> &'static str {
        match self {
            Self::Invalid => "invalid_tool_call_guard",
            Self::CrossResponseDedup => "F5",
            Self::WithinResponseDedup => "F11",
            Self::PermanentFailure => "F44",
            Self::SameNameCap => "same_name_tool_cap",
            Self::TotalCap => "F12",
        }
    }
}

#[derive(Clone, Copy)]
enum DedupScope {
    CrossFlush,
    WithinResponse,
}

/// One exit for every hostile/invalid tool path. Besides suppressing further
/// wire output, this publishes a Release-ordered scheduler cancellation and
/// marks the terminal response as forcibly truncated.
fn force_tool_stop(state: &mut StreamState, cause: ForcedToolStop) {
    tracing::debug!(?cause, "forcing tool stream stop");
    state.record_intervention(cause.code(), "cancel_generation");
    state.stop_generation(StreamStopReason::ToolLoopOrInvalid);
}

fn record_tool_repairs(state: &StreamState, backfilled: usize, normalized: usize) {
    for _ in 0..backfilled {
        state.record_intervention("tool_output_repair", "backfill_required_params");
    }
    for _ in 0..normalized {
        state.record_intervention("tool_output_repair", "normalize_paths");
    }
}

pub(super) fn reserve_tool_call_slot(state: &mut StreamState, max: usize) -> bool {
    if bump_f12_tool_call_count(&mut state.tool_calls_emitted_count, max) {
        true
    } else {
        force_tool_stop(state, ForcedToolStop::TotalCap);
        false
    }
}

fn advance_name_run(name_run: &mut Option<(String, u32)>, name: &str) -> u32 {
    let run_len = match name_run {
        Some((previous, run_len)) if previous == name => *run_len + 1,
        _ => 1,
    };
    *name_run = Some((name.to_string(), run_len));
    run_len
}

fn admit_name_run(state: &mut StreamState, name: &str) -> bool {
    let run_len = advance_name_run(&mut state.name_run, name);
    if run_len < MAX_CONSEC_SAME_NAME_CALLS {
        return true;
    }
    tracing::warn!(
        tool = %name,
        run = run_len,
        "same-name tool-call cap tripped: refusing successive same-name tool call"
    );
    force_tool_stop(state, ForcedToolStop::SameNameCap);
    false
}

fn stop_after_complete_if_serial(state: &mut StreamState, allow_parallel: bool) {
    if !allow_parallel {
        state.stop_generation(StreamStopReason::SingleToolComplete);
    }
}

pub(super) fn stage_tool_call_start(state: &mut StreamState, idx: usize, id: String, name: String) {
    state
        .streaming_tool_args
        .insert(idx, (id, name, String::new()));
}

fn admit_guarded_payload(
    state: &mut StreamState,
    f44_cache: Option<&F39FailureCache>,
    name: &str,
    arguments: &str,
    dedup_scope: DedupScope,
) -> bool {
    let duplicate = match dedup_scope {
        DedupScope::CrossFlush => state.tool_arg_dedup.check(name, arguments),
        DedupScope::WithinResponse => state.tool_arg_dedup_within.check(name, arguments),
    };
    if duplicate {
        let cause = match dedup_scope {
            DedupScope::CrossFlush => ForcedToolStop::CrossResponseDedup,
            DedupScope::WithinResponse => ForcedToolStop::WithinResponseDedup,
        };
        tracing::warn!(tool = %name, ?cause, "tool-argument dedup tripped; refusing call");
        force_tool_stop(state, cause);
        return false;
    }
    if f44_cache.is_some_and(|cache| f44_check_permanent_failure(cache, name, arguments)) {
        tracing::warn!(
            tool = %name,
            "permanent-failure circuit breaker tripped: refusing permanently-failed tool call"
        );
        force_tool_stop(state, ForcedToolStop::PermanentFailure);
        return false;
    }
    admit_name_run(state, name)
}

/// Streaming must use the same attachment policy as blocking responses.
/// An ordinary/read-only call with a missing required argument is emitted as
/// the model produced it so the client's schema validator can return useful
/// feedback. Structurally invalid calls and unexecutable write/shell calls are
/// still withheld.
fn validate_for_stream(
    call: &tool_parser::ToolCall,
    tools: &[tool_parser::ToolDefinition],
) -> Result<(), String> {
    match tool_parser::assess_tool_call(call, tools) {
        Ok(()) => Ok(()),
        Err(tool_parser::ToolCallIssue::MissingParam(message)) => {
            tracing::warn!(
                tool = %call.function.name,
                "tool call has a missing ordinary parameter; emitting for client schema feedback: {message}"
            );
            Ok(())
        }
        Err(issue) => Err(issue.into_message()),
    }
}

/// Apply the same validation and hostile-loop guards to a synthetic salvage
/// call before it reaches the wire.  Salvage is a publication path too; it
/// must not bypass invalid-call, F11/F44, same-name, or F12 enforcement.
pub(super) fn admit_salvaged_tool_call(
    state: &mut StreamState,
    ctx: &StreamCtx,
    call: &mut tool_parser::ToolCall,
) -> bool {
    let backfilled = tool_parser::backfill_required_params(
        std::slice::from_mut(call),
        &ctx.tool_defs_for_backfill,
    );
    let normalized = ctx.cwd_for_normalize.as_ref().map_or(0, |cwd| {
        tool_parser::normalize_paths(std::slice::from_mut(call), cwd)
    });
    record_tool_repairs(state, backfilled, normalized);
    if let Err(error) = validate_for_stream(call, &ctx.tool_defs_for_backfill) {
        tracing::warn!(
            tool = %call.function.name,
            "salvaged tool call validation error: {error}; refusing call"
        );
        force_tool_stop(state, ForcedToolStop::Invalid);
        return false;
    }
    (!ctx.agent_guards
        || admit_guarded_payload(
            state,
            ctx.f44_cache_active.then_some(&ctx.f44_cache),
            &call.function.name,
            &call.function.arguments,
            DedupScope::WithinResponse,
        ))
        && reserve_tool_call_slot(state, ctx.max_tool_calls_per_response)
}

/// Native GLM publication. The request's tool grammar carries the exact
/// allowed set (a specific tool_choice narrows it); never recover that
/// restriction from the all-tools list, and never repair raw native values.
pub(super) fn validate_native_glm(
    call: &mut tool_parser::ToolCall,
    allowed: Option<&[tool_parser::ToolDefinition]>,
) -> Result<(), String> {
    let allowed =
        allowed.ok_or("native GLM publication is missing its exact tool grammar binding")?;
    *call = tool_parser::native_publication::validate_call(call, allowed)?;
    Ok(())
}

/// `DetectorOutput::ToolCall(tc, idx)`: complete tool call.
pub(super) fn handle_complete_tool_call(
    state: &mut StreamState,
    ctx: &StreamCtx,
    tc: &mut tool_parser::ToolCall,
    tc_idx: usize,
    sse_events: &mut ChunkVec,
) {
    // Discard any stale unpublished staging entry for this index before
    // publishing a complete tool call.
    state.streaming_tool_args.remove(&tc_idx);
    // Content → Tool boundary: flush sanitiser tail.
    let (pre_tool_tail, modified) = flush_content_sanitizer_with_report(
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &ctx.leak_markers,
    );
    if modified {
        state.record_intervention("content_output_repair", "drop_partial_tool_markup");
    }
    if !pre_tool_tail.is_empty() {
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, pre_tool_tail);
        sse_events.push(chunk);
    }
    let native_glm = ctx
        .state
        .tool_call_parser
        .as_ref()
        .is_some_and(|p| p.name() == "glm_xml");
    let validation = if native_glm {
        validate_native_glm(tc, ctx.native_tool_defs.as_deref())
    } else {
        let backfilled = tool_parser::backfill_required_params(
            std::slice::from_mut(tc),
            &ctx.tool_defs_for_backfill,
        );
        let normalized = ctx.cwd_for_normalize.as_ref().map_or(0, |cwd| {
            tool_parser::normalize_paths(std::slice::from_mut(tc), cwd)
        });
        record_tool_repairs(state, backfilled, normalized);
        validate_for_stream(tc, &ctx.tool_defs_for_backfill)
    };
    if let Err(e) = validation {
        tracing::warn!(
            tool = %tc.function.name,
            "tool call validation error: {e}; replacing with content and ending"
        );
        let msg = format!("[flux] Tool call rejected: {e}");
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, msg);
        sse_events.push(chunk);
        force_tool_stop(state, ForcedToolStop::Invalid);
    } else {
        if ctx.agent_guards
            && !admit_guarded_payload(
                state,
                ctx.f44_cache_active.then_some(&ctx.f44_cache),
                &tc.function.name,
                &tc.function.arguments,
                DedupScope::CrossFlush,
            )
        {
            return;
        }
        if !reserve_tool_call_slot(state, ctx.max_tool_calls_per_response) {
            return;
        }
        // Successful complete-call path — log + metric to match the
        // blocking and incremental-streaming paths.
        let preview: String = tc.function.arguments.chars().take(120).collect();
        let s = if tc.function.arguments.len() > preview.len() {
            "…"
        } else {
            ""
        };
        tracing::info!("Tool call: {}({preview}{s})", tc.function.name);
        crate::metrics::TOOL_CALLS_TOTAL.inc();
        let start = ChatCompletionChunk::tool_call_start_chunk(&ctx.model, &ctx.id, tc, tc_idx);
        sse_events.push(start);
        let frag = ChatCompletionChunk::tool_call_args_fragment(
            &ctx.model,
            &ctx.id,
            tc_idx,
            &tc.function.arguments,
        );
        sse_events.push(frag);
        stop_after_complete_if_serial(state, ctx.allow_parallel_tool_calls);
    }
}

/// `DetectorOutput::ToolCallStart` — stage the header server-side. Publication
/// waits for the complete argument payload to pass validation and every guard.
///
/// If `idx` is already staged (unpublished), this acts as an authoritative
/// name override from envelope-close parsing and updates the staged header in
/// place without re-flushing the content sanitizer.
pub(super) fn handle_tool_call_start(
    state: &mut StreamState,
    ctx: &StreamCtx,
    tc_id: String,
    name: String,
    idx: usize,
    sse_events: &mut ChunkVec,
) {
    if let Some((existing_id, existing_name, existing_args)) =
        state.streaming_tool_args.get_mut(&idx)
        && existing_args.is_empty()
    {
        if !tc_id.is_empty() {
            *existing_id = tc_id;
        }
        *existing_name = name;
        return;
    }
    let (pre_tool_tail, modified) = flush_content_sanitizer_with_report(
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &ctx.leak_markers,
    );
    if modified {
        state.record_intervention("content_output_repair", "drop_partial_tool_markup");
    }
    if !pre_tool_tail.is_empty() {
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, pre_tool_tail);
        sse_events.push(chunk);
    }
    stage_tool_call_start(state, idx, tc_id, name);
}

/// `DetectorOutput::ToolCallDelta` — validate the complete args, reserve a
/// publication slot, then atomically append the start + args wire chunks.
///
/// For qwen3_coder XML the streaming detector emits a single Delta with
/// the full parsed-and-canonicalised JSON arguments at the `</tool_call>`
/// boundary (see `streaming_impl.rs::process` line ~67 — args can't be
/// streamed character-by-character because XML parameter blocks must
/// finish before they convert to JSON). This is the natural spot to run
/// the same `backfill_required_params` + `validate_single_tool_call`
/// chain that the complete-tool-call path runs at `handle_complete_tool_call`,
/// so that streaming and non-streaming responses behave identically.
///
/// Without this, a model that emits `<function=NAME></function>` with no
/// `<parameter=>` blocks streams literal `"{}"` to the client even when
/// required parameters are declared in the schema, whereas the non-streaming
/// path runs the same no-fabrication backfill and family-specific disposition.
pub(super) fn handle_tool_call_delta(
    state: &mut StreamState,
    ctx: &StreamCtx,
    args: String,
    idx: usize,
    sse_events: &mut ChunkVec,
) {
    publish_staged_tool_call_with_guards(
        state,
        idx,
        args,
        &ctx.tool_defs_for_backfill,
        ctx.cwd_for_normalize.as_deref(),
        ctx.f44_cache_active.then_some(&ctx.f44_cache),
        ctx.max_tool_calls_per_response,
        ctx.agent_guards,
        &ctx.model,
        &ctx.id,
        sse_events,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn publish_staged_tool_call(
    state: &mut StreamState,
    idx: usize,
    args: String,
    tool_defs: &[tool_parser::ToolDefinition],
    cwd_for_normalize: Option<&str>,
    f44_cache: Option<&F39FailureCache>,
    max_tool_calls: usize,
    model: &str,
    request_id: &str,
    sse_events: &mut ChunkVec,
) -> bool {
    publish_staged_tool_call_with_guards(
        state,
        idx,
        args,
        tool_defs,
        cwd_for_normalize,
        f44_cache,
        max_tool_calls,
        true,
        model,
        request_id,
        sse_events,
    )
}

#[allow(clippy::too_many_arguments)]
fn publish_staged_tool_call_with_guards(
    state: &mut StreamState,
    idx: usize,
    args: String,
    tool_defs: &[tool_parser::ToolDefinition],
    cwd_for_normalize: Option<&str>,
    f44_cache: Option<&F39FailureCache>,
    max_tool_calls: usize,
    agent_guards: bool,
    model: &str,
    request_id: &str,
    sse_events: &mut ChunkVec,
) -> bool {
    let Some((tc_id, name, _)) = state.streaming_tool_args.remove(&idx) else {
        // Never publish argument bytes without a corresponding accepted start.
        return false;
    };
    let mut tc = tool_parser::ToolCall {
        id: tc_id,
        call_type: "function".into(),
        function: tool_parser::FunctionCall {
            name: name.clone(),
            arguments: args,
        },
    };
    let backfilled =
        tool_parser::backfill_required_params(std::slice::from_mut(&mut tc), tool_defs);
    let normalized = cwd_for_normalize.map_or(0, |cwd| {
        tool_parser::normalize_paths(std::slice::from_mut(&mut tc), cwd)
    });
    record_tool_repairs(state, backfilled, normalized);
    if let Err(e) = validate_for_stream(&tc, tool_defs) {
        tracing::warn!(
            tool = %name,
            "tool call validation error (stream Δ): {e}; replacing with content and ending"
        );
        let msg = format!("[flux] Tool call rejected: {e}");
        let chunk = ChatCompletionChunk::content_chunk(model, request_id, msg);
        sse_events.push(chunk);
        force_tool_stop(state, ForcedToolStop::Invalid);
        return false;
    }

    let emit_args = tc.function.arguments;
    if agent_guards
        && !admit_guarded_payload(
            state,
            f44_cache,
            &name,
            &emit_args,
            DedupScope::WithinResponse,
        )
    {
        return false;
    }
    if !reserve_tool_call_slot(state, max_tool_calls) {
        return false;
    }
    let tc_id = tc.id;
    state
        .streaming_tool_args
        .insert(idx, (tc_id.clone(), name.clone(), emit_args.clone()));
    let tc = tool_parser::ToolCall {
        id: tc_id,
        call_type: "function".to_string(),
        function: tool_parser::FunctionCall {
            name,
            arguments: String::new(),
        },
    };
    let start = ChatCompletionChunk::tool_call_start_chunk(model, request_id, &tc, idx);
    sse_events.push(start);
    if !emit_args.is_empty() {
        let frag = ChatCompletionChunk::tool_call_args_fragment(model, request_id, idx, &emit_args);
        sse_events.push(frag);
    }
    true
}

/// `DetectorOutput::ToolCallEnd` — publish successful completion metadata.
/// Validation and every loop guard run before argument publication in
/// `handle_tool_call_delta`; reaching End therefore means this call is valid.
///
/// Bug-2 cap (`MAX_CONSEC_SAME_NAME_CALLS`): trips when the same tool
/// name fires N times in a row regardless of args. F11 keys on
/// `(name, canonical_args)` and is defeated by runaway loops where
/// the model rolls a fresh timestamp / sequence number / id into the
/// payload each iteration; the F12 total cap (default 12) is the
/// only other server-side circuit, but a runaway can already have
/// flooded the SSE channel before F12 fires. The name-run cap is
/// strictly tighter than F11 and F12 for the runaway pattern.
const MAX_CONSEC_SAME_NAME_CALLS: u32 = 6;

pub(super) fn handle_tool_call_end(state: &mut StreamState, ctx: &StreamCtx, idx: usize) {
    finish_staged_tool_call(state, ctx.allow_parallel_tool_calls, idx);
}

pub(super) fn finish_staged_tool_call(
    state: &mut StreamState,
    allow_parallel: bool,
    idx: usize,
) -> bool {
    if let Some((_id, name, args_json)) = state.streaming_tool_args.remove(&idx) {
        // Staged by ToolCallStart but never published via ToolCallDelta
        // (e.g. envelope parsing returned no call on close/flush): drop the
        // stale entry without logging, counting metrics, or stopping the stream.
        if args_json.is_empty() {
            return false;
        }
        // Successful streaming tool call — log + metric to match the
        // blocking and complete-call paths.
        let preview: String = args_json.chars().take(120).collect();
        let s = if args_json.len() > preview.len() {
            "…"
        } else {
            ""
        };
        tracing::info!("Tool call: {name}({preview}{s})");
        crate::metrics::TOOL_CALLS_TOTAL.inc();
        stop_after_complete_if_serial(state, allow_parallel);
        return true;
    }
    false
}

#[cfg(test)]
mod disposition_tests {
    use super::{
        DedupScope, ForcedToolStop, MAX_CONSEC_SAME_NAME_CALLS, admit_guarded_payload,
        admit_name_run, finish_staged_tool_call, force_tool_stop, publish_staged_tool_call,
        reserve_tool_call_slot, stage_tool_call_start, stop_after_complete_if_serial,
        validate_for_stream,
    };
    use crate::api::chat_stream::handle_token::dispatch_detector_batch;
    use crate::api::chat_stream::state::StreamState;
    use crate::api::failures::F39FailureCache;
    use crate::openai::{
        ChatCompletionChunk, InterventionLedger, SharedInterventionLedger,
        snapshot_shared_interventions,
    };
    use crate::tool_parser::{
        DetectorOutput, FunctionCall, FunctionDefinition, StreamingToolDetector, ToolCall,
        ToolDefinition,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    fn state() -> (StreamState, Arc<AtomicBool>) {
        let cancel = Arc::new(AtomicBool::new(false));
        (
            StreamState::new(false, false, Arc::clone(&cancel), None),
            cancel,
        )
    }

    fn state_with_ledger() -> (StreamState, SharedInterventionLedger) {
        let cancel = Arc::new(AtomicBool::new(false));
        let ledger = Arc::new(Mutex::new(InterventionLedger::default()));
        (
            StreamState::new(false, false, cancel, Some(Arc::clone(&ledger))),
            ledger,
        )
    }

    fn tool(name: &str, required: &str) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: name.into(),
                description: None,
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {required: {"type": "string"}},
                    "required": [required]
                })),
            },
        }
    }

    fn empty_call(name: &str) -> ToolCall {
        ToolCall {
            id: "call_test".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: "{}".into(),
            },
        }
    }

    fn tool_chunk_count(events: &[ChatCompletionChunk]) -> usize {
        events
            .iter()
            .filter(|event| {
                event
                    .choices
                    .first()
                    .is_some_and(|choice| choice.delta.tool_calls.is_some())
            })
            .count()
    }

    #[test]
    fn streaming_matches_blocking_attachment_disposition() {
        assert!(
            validate_for_stream(&empty_call("get_weather"), &[tool("get_weather", "city")]).is_ok(),
            "ordinary missing param stays attached"
        );
        assert!(
            validate_for_stream(&empty_call("write"), &[tool("write", "filePath")]).is_err(),
            "write without a path is withheld"
        );
        assert!(
            validate_for_stream(&empty_call("exec"), &[tool("exec", "command")]).is_err(),
            "shell without a command is withheld"
        );
    }

    #[test]
    fn invalid_dedup_and_f44_stops_all_publish_release_cancellation() {
        for cause in [
            ForcedToolStop::Invalid,
            ForcedToolStop::CrossResponseDedup,
            ForcedToolStop::WithinResponseDedup,
            ForcedToolStop::PermanentFailure,
        ] {
            let (mut state, cancel) = state();
            force_tool_stop(&mut state, cause);
            assert!(state.stop_string_triggered, "cause={cause:?}");
            assert!(state.tool_loop_capped, "cause={cause:?}");
            assert!(cancel.load(Ordering::Acquire), "cause={cause:?}");
        }
    }

    #[test]
    fn f11_duplicate_payload_is_rejected_before_a_second_completion() {
        let (mut state, cancel) = state();
        let args = r#"{"filePath":"/tmp/a"}"#;
        assert!(admit_guarded_payload(
            &mut state,
            None,
            "read",
            args,
            DedupScope::WithinResponse,
        ));
        assert!(!admit_guarded_payload(
            &mut state,
            None,
            "read",
            args,
            DedupScope::WithinResponse,
        ));
        assert!(state.tool_loop_capped);
        assert!(cancel.load(Ordering::Acquire));
    }

    #[test]
    fn f44_permanent_failure_payload_is_rejected_and_cancelled() {
        let (mut state, cancel) = state();
        let mut cache = F39FailureCache::default();
        cache
            .missing_bins_by_tool
            .entry("bash".into())
            .or_default()
            .insert("missing-bin".into(), 1);

        assert!(!admit_guarded_payload(
            &mut state,
            Some(&cache),
            "bash",
            r#"{"command":"missing-bin --version"}"#,
            DedupScope::WithinResponse,
        ));
        assert!(state.tool_loop_capped);
        assert!(cancel.load(Ordering::Acquire));
    }

    #[test]
    fn same_name_cap_rejects_capped_call_and_cancels() {
        let (mut state, cancel) = state();
        for _ in 1..MAX_CONSEC_SAME_NAME_CALLS {
            assert!(admit_name_run(&mut state, "exec"));
        }
        assert!(!admit_name_run(&mut state, "exec"));
        assert!(state.tool_loop_capped);
        assert!(cancel.load(Ordering::Acquire));
    }

    #[test]
    fn f12_allows_twelve_and_never_publishes_a_thirteenth_slot() {
        let (mut state, cancel) = state();
        for expected in 1..=12 {
            assert!(reserve_tool_call_slot(&mut state, 12));
            assert_eq!(state.tool_calls_emitted_count, expected);
        }
        assert!(!reserve_tool_call_slot(&mut state, 12));
        assert_eq!(state.tool_calls_emitted_count, 12);
        assert!(state.tool_loop_capped);
        assert!(cancel.load(Ordering::Acquire));
    }

    #[test]
    fn incremental_start_is_staged_without_reserving_or_publishing() {
        let (mut state, cancel) = state();
        stage_tool_call_start(&mut state, 7, "call_7".into(), "write".into());

        assert_eq!(state.tool_calls_emitted_count, 0);
        let expected = ("call_7".to_string(), "write".to_string(), String::new());
        assert_eq!(state.streaming_tool_args.get(&7), Some(&expected));
        assert!(!state.stop_string_triggered);
        assert!(!cancel.load(Ordering::Acquire));

        let source = include_str!("tool_handlers.rs");
        let start_body = source
            .split("pub(super) fn handle_tool_call_start")
            .nth(1)
            .unwrap()
            .split("pub(super) fn handle_tool_call_delta")
            .next()
            .unwrap();
        assert!(!start_body.contains("tool_call_start_chunk"));
        assert!(!start_body.contains("reserve_tool_call_slot"));
    }

    #[test]
    fn parallel_false_stops_after_one_valid_call_without_cap() {
        let (mut serial, serial_cancel) = state();
        stop_after_complete_if_serial(&mut serial, false);
        assert!(serial.stop_string_triggered);
        assert!(!serial.tool_loop_capped);
        assert!(serial_cancel.load(Ordering::Acquire));

        let (mut parallel, parallel_cancel) = state();
        stop_after_complete_if_serial(&mut parallel, true);
        assert!(!parallel.stop_string_triggered);
        assert!(!parallel_cancel.load(Ordering::Acquire));
    }

    #[test]
    fn detector_content_start_delta_end_publishes_one_serial_call() {
        let (mut state, cancel) = state();
        let mut detector = StreamingToolDetector::new();
        let defs = [tool("get_weather", "city")];
        let mut events = Vec::new();
        let first = detector.process("before<tool_call><function=get_weather>");
        assert!(matches!(first.first(), Some(DetectorOutput::Content(text)) if text == "before"));
        dispatch_detector_batch(first, |output| {
            match output {
                DetectorOutput::Content(text) => {
                    events.push(ChatCompletionChunk::content_chunk("model", "request", text))
                }
                DetectorOutput::ToolCallStart { id, name, idx } => {
                    stage_tool_call_start(&mut state, idx, id, name);
                }
                _ => panic!("unexpected detector output in content/start batch"),
            }
            state.stop_string_triggered
        });
        assert_eq!(events.len(), 1, "leading content must be published");
        assert!(state.streaming_tool_args.contains_key(&0));

        let second = detector.process("<parameter=city>Paris</parameter></function></tool_call>");
        dispatch_detector_batch(second, |output| {
            match output {
                DetectorOutput::ToolCallDelta { args, idx } => {
                    assert!(publish_staged_tool_call(
                        &mut state,
                        idx,
                        args,
                        &defs,
                        None,
                        None,
                        12,
                        "model",
                        "request",
                        &mut events,
                    ));
                }
                DetectorOutput::ToolCallEnd { idx } => {
                    assert!(finish_staged_tool_call(&mut state, false, idx));
                }
                _ => panic!("unexpected detector output in delta/end batch"),
            }
            state.stop_string_triggered
        });

        assert_eq!(events.len(), 3, "content plus start plus args fragment");
        assert_eq!(tool_chunk_count(&events), 2, "start plus args fragment");
        assert_eq!(state.tool_calls_emitted_count, 1);
        assert!(state.stop_string_triggered);
        assert!(!state.tool_loop_capped);
        assert!(cancel.load(Ordering::Acquire));
    }

    #[test]
    fn invalid_staged_call_publishes_no_partial_tool_object() {
        let (mut state, cancel) = state();
        stage_tool_call_start(&mut state, 0, "call_invalid".into(), "write".into());
        let mut events = Vec::new();
        assert!(!publish_staged_tool_call(
            &mut state,
            0,
            "{}".into(),
            &[tool("write", "filePath")],
            None,
            None,
            12,
            "model",
            "request",
            &mut events,
        ));
        assert_eq!(tool_chunk_count(&events), 0);
        assert!(!state.streaming_tool_args.contains_key(&0));
        assert!(state.tool_loop_capped);
        assert!(cancel.load(Ordering::Acquire));
    }

    #[test]
    fn published_streaming_repairs_are_recorded_in_the_intervention_ledger() {
        let (mut state, ledger) = state_with_ledger();
        let defs = [ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "probe".into(),
                description: None,
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "file_path": {"type": "string"}
                    },
                    "required": ["count", "file_path"]
                })),
            },
        }];
        stage_tool_call_start(&mut state, 0, "call_repair".into(), "probe".into());
        let mut events = Vec::new();
        assert!(publish_staged_tool_call(
            &mut state,
            0,
            r#"{"count":"2","file_path":"/work/tree/file.txt"}"#.into(),
            &defs,
            Some("/work/tree"),
            None,
            12,
            "model",
            "request",
            &mut events,
        ));

        let snapshot = snapshot_shared_interventions(Some(&ledger)).unwrap();
        assert_eq!(snapshot.events.len(), 2);
        assert_eq!(snapshot.events[0].code, "tool_output_repair");
        assert_eq!(snapshot.events[0].action, "backfill_required_params");
        assert_eq!(snapshot.events[0].count, 1);
        assert_eq!(snapshot.events[1].code, "tool_output_repair");
        assert_eq!(snapshot.events[1].action, "normalize_paths");
        assert_eq!(snapshot.events[1].count, 1);
    }

    #[test]
    fn thirteenth_staged_call_has_no_wire_header_or_arguments() {
        let (mut state, cancel) = state();
        state.tool_calls_emitted_count = 12;
        stage_tool_call_start(&mut state, 12, "call_13".into(), "probe".into());
        let mut events = Vec::new();
        assert!(!publish_staged_tool_call(
            &mut state,
            12,
            r#"{"value":"x"}"#.into(),
            &[tool("probe", "value")],
            None,
            None,
            12,
            "model",
            "request",
            &mut events,
        ));
        assert_eq!(tool_chunk_count(&events), 0);
        assert_eq!(state.tool_calls_emitted_count, 12);
        assert!(state.tool_loop_capped);
        assert!(cancel.load(Ordering::Acquire));
    }

    #[test]
    fn authoritative_name_override_replaces_staged_name_before_publication() {
        let (mut state, cancel) = state();
        let defs = [tool("get_weather", "city")];
        // Speculative streaming start stages the wrong name.
        stage_tool_call_start(&mut state, 0, "call_auth".into(), "wrong_name".into());
        // Authoritative re-stage at close overrides the name before publication.
        stage_tool_call_start(&mut state, 0, "call_auth".into(), "get_weather".into());

        let mut events = Vec::new();
        assert!(publish_staged_tool_call(
            &mut state,
            0,
            r#"{"city":"Paris"}"#.into(),
            &defs,
            None,
            None,
            12,
            "model",
            "request",
            &mut events,
        ));
        assert!(finish_staged_tool_call(&mut state, false, 0));

        assert_eq!(
            tool_chunk_count(&events),
            2,
            "one start plus one args chunk"
        );
        let start_wire = serde_json::to_value(&events[0]).unwrap();
        assert_eq!(
            start_wire["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_auth"
        );
        assert_eq!(
            start_wire["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(state.tool_calls_emitted_count, 1);
        assert!(!state.tool_loop_capped);
        assert!(cancel.load(Ordering::Acquire));
    }

    #[test]
    fn unparsed_staged_call_is_discarded_on_end_without_publishing_or_stopping() {
        let (mut state, cancel) = state();
        let mut detector = StreamingToolDetector::new();
        let mut events = Vec::new();

        for output in detector
            .process(r#"<tool_call>{"name": "ghost""#)
            .into_iter()
            .chain(detector.process(" not json</tool_call>"))
        {
            match output {
                DetectorOutput::ToolCallStart { id, name, idx } => {
                    stage_tool_call_start(&mut state, idx, id, name);
                }
                DetectorOutput::ToolCallDelta { args, idx } => {
                    let _ = publish_staged_tool_call(
                        &mut state,
                        idx,
                        args,
                        &[tool("ghost", "x")],
                        None,
                        None,
                        12,
                        "model",
                        "request",
                        &mut events,
                    );
                }
                DetectorOutput::ToolCallEnd { idx } => {
                    assert!(
                        !finish_staged_tool_call(&mut state, false, idx),
                        "unpublished staged header must be dropped without completing"
                    );
                }
                _ => {}
            }
        }

        assert!(
            state.streaming_tool_args.is_empty(),
            "stale staged entry must be removed"
        );
        assert_eq!(tool_chunk_count(&events), 0);
        assert_eq!(state.tool_calls_emitted_count, 0);
        assert!(!state.stop_string_triggered);
        assert!(!cancel.load(Ordering::Acquire));
    }
}
