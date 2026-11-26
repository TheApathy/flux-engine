// SPDX-License-Identifier: AGPL-3.0-only

//! Blocking (non-streaming) `/v1/chat/completions` path. Supports `n >= 1`
//! (multiple choices per request) by running the scheduler send + decode +
//! tool-parse pipeline once per choice index.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::AppState;
use crate::openai::{ChatCompletionRequest, ChatCompletionResponse, Usage};
use crate::tool_parser;

use super::compact::openai_error_response;
use super::failures::f60_disable_mtp_for_request;
use super::inference_impl::{extract_thinking, strip_stop_sequences, try_admit_scheduler};
use super::inference_types::{GrammarSpec, InferenceRequest};

pub(super) struct BlockingPathArgs {
    pub state: Arc<AppState>,
    pub req: ChatCompletionRequest,
    pub req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    pub dump_seq: Option<u64>,
    pub prompt_tokens: Vec<u32>,
    pub session_hash: u64,
    pub image_pixels: Vec<(Vec<f32>, usize, usize)>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    pub lz_penalty: f32,
    pub logit_bias: Vec<(u32, f32)>,
    pub stop_tokens: Vec<u32>,
    pub enable_thinking: bool,
    pub thinking_budget: Option<u32>,
    pub tools_active: bool,
    pub tool_choice_required: bool,
    pub suppress_tool_call: bool,
    pub grammar_spec: Option<GrammarSpec>,
    pub top_logprobs: Option<u8>,
    pub timeout_at: Option<std::time::Instant>,
    /// Deadline that replaces `timeout_at` once the first token is emitted
    /// (request-lifecycle contract). None = keep `timeout_at`.
    pub decode_timeout_at: Option<std::time::Instant>,
    pub cwd_hint: Option<String>,
    pub prompt_len: usize,
    pub intervention_ledger: Option<crate::openai::SharedInterventionLedger>,
    pub rendered_prompt_evidence_source: Option<(String, String)>,
    /// Request-lifecycle warnings (e.g. `max_tokens` clamped at admission).
    pub warnings: Vec<String>,
}

/// RAII guard for the `REQUESTS_ACTIVE` count handed off by
/// `chat_completions_inner` via `ActiveRequestGuard::release()`.
///
/// Guarantees every exit from `run_blocking_path` — normal completion, early
/// admission/scheduler error, client disconnect (dropped future), or unwind —
/// decrements `REQUESTS_ACTIVE` exactly once.
struct BlockingActiveGuard<'a>(&'a prometheus::IntGauge);

impl BlockingActiveGuard<'static> {
    fn new() -> Self {
        Self(&crate::metrics::REQUESTS_ACTIVE)
    }
}

impl Drop for BlockingActiveGuard<'_> {
    fn drop(&mut self) {
        self.0.dec();
    }
}

pub(super) async fn run_blocking_path(args: BlockingPathArgs) -> Response {
    let _active_guard = BlockingActiveGuard::new();
    let BlockingPathArgs {
        state,
        req,
        req_ctx,
        dump_seq,
        prompt_tokens,
        session_hash,
        image_pixels,
        max_tokens,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        stop_tokens,
        enable_thinking,
        thinking_budget,
        tools_active,
        tool_choice_required,
        suppress_tool_call,
        grammar_spec,
        top_logprobs,
        timeout_at,
        decode_timeout_at,
        cwd_hint,
        prompt_len,
        intervention_ledger,
        rendered_prompt_evidence_source,
        warnings,
    } = args;

    let n = req.n.max(1);
    let mut all_choices: Vec<crate::openai::ChatChoice> = Vec::with_capacity(n);
    let mut total_completion_tokens = 0usize;
    let mut total_decode_tokens = 0usize;
    let mut first_ttft = 0.0f64;
    let mut total_decode_time_ms = 0.0f64;
    let mut total_reasoning_tokens = 0u32;
    let mut total_cached_prompt_tokens = 0u32;
    let mut total_think_nudges = 0u32;

    for choice_idx in 0..n {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let request = InferenceRequest::Blocking {
            prompt_tokens: prompt_tokens.clone(),
            session_hash,
            image_pixels: image_pixels.clone(),
            max_tokens,
            min_tokens: req.min_tokens,
            ignore_eos: req.ignore_eos,
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            lz_penalty,
            logit_bias: logit_bias.clone(),
            stop_tokens: stop_tokens.clone(),
            enable_thinking,
            thinking_budget,
            require_tool_call: tool_choice_required,
            suppress_tool_call,
            disable_mtp: f60_disable_mtp_for_request(tools_active),
            grammar_spec: grammar_spec.clone(),
            seed: req.seed.map(|s| s.wrapping_add(choice_idx as u64)),
            top_logprobs,
            timeout_at,
            decode_timeout_at,
            intervention_ledger: intervention_ledger.clone(),
            think_nudge: req
                .runtime_defaults
                .as_ref()
                .and_then(|d| d.think_nudge.clone()),
            response_tx: tx,
        };

        if let Err((status, message)) = try_admit_scheduler(&state.request_tx, request) {
            return openai_error_response(status, message);
        }

        let response = match rx.await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                tracing::error!("chat inference error: {e:#}");
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Internal inference error".to_string(),
                );
            }
            Err(_) => {
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Inference cancelled".to_string(),
                );
            }
        };

        if choice_idx == 0 {
            first_ttft = response.time_to_first_token_ms;
        }
        total_decode_time_ms += response.decode_time_ms;

        let num_completion = response.output_tokens.len();
        total_completion_tokens += num_completion;
        total_decode_tokens += num_completion.saturating_sub(1);
        total_reasoning_tokens += response.reasoning_tokens;
        total_think_nudges += response.think_nudges;
        // cached_prompt_tokens is a per-request prefix-cache hit count; for
        // n>1 we only charge once (same prompt reused).
        total_cached_prompt_tokens = total_cached_prompt_tokens.max(response.cached_prompt_tokens);

        if state.dsv41 {
            // deepseek_v41: app.py's output routing and DSML parsing (api/dsv41.rs).
            let (mut message, finish_reason) = super::dsv41::blocking_choice(
                &state.tokenizer,
                &response.output_tokens,
                response.finish_reason.as_str(),
                enable_thinking,
                &req.stop,
            );
            if message.tool_calls.is_none()
                && let Some(content) = message.content.take()
            {
                message.content = Some(super::chat::repair_json::strip_json_markdown_fence(
                    &req, content,
                ));
            }
            if let Some(calls) = &mut message.tool_calls
                && enforce_parallel_tool_call_policy(calls, req.parallel_tool_calls.unwrap_or(true))
            {
                crate::openai::record_shared_intervention(
                    intervention_ledger.as_ref(),
                    "tool_output_policy",
                    "truncate_parallel_calls",
                );
            }
            let flux_stop_reason = guard_stop_reason(&finish_reason, &response);
            all_choices.push(crate::openai::ChatChoice {
                index: choice_idx,
                message,
                finish_reason,
                logprobs: build_logprobs(&state, &response),
                token_ids: if req.return_token_ids {
                    response.output_tokens.clone()
                } else {
                    Vec::new()
                },
                flux_stop_reason,
            });
            continue;
        }

        let (reasoning_content_i, output_text_i) = decode_response_text(
            &state,
            &response,
            enable_thinking,
            intervention_ledger.as_ref(),
        );
        let output_text_i = strip_stop_sequences(output_text_i, &req.stop);
        let output_text_i =
            super::chat::repair_json::strip_json_markdown_fence(&req, output_text_i);
        let output_text_i =
            repair_json_object_prefix_with_audit(&req, output_text_i, intervention_ledger.as_ref());

        let (message, finish_reason_i) = build_choice_message(
            &state,
            &req,
            &response,
            reasoning_content_i,
            output_text_i,
            tools_active,
            cwd_hint.as_deref(),
            choice_idx,
            intervention_ledger.as_ref(),
        );

        let flux_stop_reason = guard_stop_reason(&finish_reason_i, &response);
        all_choices.push(crate::openai::ChatChoice {
            index: choice_idx,
            message,
            finish_reason: finish_reason_i,
            logprobs: build_logprobs(&state, &response),
            token_ids: if req.return_token_ids {
                response.output_tokens.clone()
            } else {
                Vec::new()
            },
            flux_stop_reason,
        });
    }

    finalize_response(
        state,
        req,
        req_ctx,
        dump_seq,
        all_choices,
        total_completion_tokens,
        total_decode_tokens,
        first_ttft,
        total_decode_time_ms,
        total_reasoning_tokens,
        total_cached_prompt_tokens,
        total_think_nudges,
        prompt_len,
        intervention_ledger,
        rendered_prompt_evidence_source,
        warnings,
    )
}

fn repair_json_object_prefix_with_audit(
    req: &ChatCompletionRequest,
    content: String,
    intervention_ledger: Option<&crate::openai::SharedInterventionLedger>,
) -> String {
    let original = content.clone();
    let repaired = super::chat::repair_json::repair_json_object_prefix(req, content);
    if repaired != original {
        crate::openai::record_shared_intervention(
            intervention_ledger,
            "json_output_repair",
            "repair_json_object_prefix",
        );
    }
    repaired
}

/// Decode `(reasoning_content, output_text)` from the scheduler's
/// response. When `enable_thinking=true`, split at the first `</think>`
/// token. When `enable_thinking=false`, decode all output_tokens as
/// content — mirrors streaming's `thinking_done = !enable_thinking`
/// init in chat_stream/state.rs and recovers the answer Qwen3.x emits
/// inside `<think>...</think>` when it ignores a closed-thinking
/// prefill (issue #40).
fn decode_response_text(
    state: &AppState,
    response: &super::inference_types::InferenceResponse,
    enable_thinking: bool,
    intervention_ledger: Option<&crate::openai::SharedInterventionLedger>,
) -> (Option<String>, String) {
    decode_response_token_text(
        &response.output_tokens,
        enable_thinking,
        state.think_end_token_id,
        response.thinking_output_boundary,
        |tokens| state.tokenizer.decode(tokens).ok(),
        |text, thinking| extract_thinking(text, thinking, state.reasoning_parser.as_deref()),
        intervention_ledger,
    )
}

fn decode_response_token_text(
    output_tokens: &[u32],
    enable_thinking: bool,
    think_end_token: Option<u32>,
    hidden_boundary: Option<usize>,
    mut decode: impl FnMut(&[u32]) -> Option<String>,
    mut extract: impl FnMut(&str, bool) -> (Option<String>, String),
    intervention_ledger: Option<&crate::openai::SharedInterventionLedger>,
) -> (Option<String>, String) {
    let (thinking_tokens, content_tokens) = split_response_token_phases(
        output_tokens,
        enable_thinking,
        think_end_token,
        hidden_boundary,
    );
    if let Some(thinking_tokens) = thinking_tokens {
        let reasoning = if !thinking_tokens.is_empty() {
            let decoded = decode(thinking_tokens);
            if decoded
                .as_ref()
                .is_some_and(|text| !text.is_empty() && text.trim().is_empty())
            {
                crate::openai::record_shared_intervention(
                    intervention_ledger,
                    "reasoning_output_repair",
                    "drop_blank_reasoning",
                );
            }
            decoded.filter(|s| !s.trim().is_empty())
        } else {
            None
        };
        // Preserve decoded answer whitespace as part of the model output.
        let content = decode(content_tokens).unwrap_or_default();
        return (reasoning, content);
    }

    let text = decode(output_tokens).unwrap_or_default();
    if think_end_token.is_some() {
        (None, text)
    } else {
        extract(&text, enable_thinking)
    }
}

/// Partition the scheduler-owned visible token sequence at either the first
/// literal reasoning delimiter or the explicit hidden boundary supplied by
/// the scheduler. Splitting on the first delimiter matches streaming and
/// vLLM/SGLang so quoting `</think>` in an answer never swallows the answer
/// into `reasoning_content`.
fn split_response_token_phases(
    output_tokens: &[u32],
    enable_thinking: bool,
    think_end_token: Option<u32>,
    hidden_boundary: Option<usize>,
) -> (Option<&[u32]>, &[u32]) {
    if !enable_thinking {
        return (None, output_tokens);
    }
    if let Some(pos) = think_end_token.and_then(|end| output_tokens.iter().position(|&t| t == end))
    {
        return (Some(&output_tokens[..pos]), &output_tokens[pos + 1..]);
    }
    if let Some(pos) = hidden_boundary.filter(|&pos| pos <= output_tokens.len()) {
        return (Some(&output_tokens[..pos]), &output_tokens[pos..]);
    }
    (None, output_tokens)
}

#[cfg(test)]
mod guard_stop_reason_tests {
    use super::guard_stop_reason;
    use crate::api::inference_types::InferenceResponse;

    fn response(stop_cause: Option<&'static str>) -> InferenceResponse {
        InferenceResponse {
            output_tokens: vec![1, 2],
            thinking_output_boundary: None,
            finish_reason: "stop".into(),
            time_to_first_token_ms: 0.0,
            decode_time_ms: 0.0,
            logprobs: vec![],
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            think_nudges: 0,
            stop_cause,
        }
    }

    #[test]
    fn blocking_choice_names_the_guard_only_on_a_stop_finish() {
        let guarded = response(Some("watchdog"));
        assert_eq!(guard_stop_reason("stop", &guarded), Some("watchdog"));
        // The tool-call parser can turn the finish into tool_calls.
        assert_eq!(guard_stop_reason("tool_calls", &guarded), None);
        assert_eq!(guard_stop_reason("stop", &response(None)), None);
    }

    #[test]
    fn chat_choice_serializes_the_cause_only_when_set() {
        let choice = |flux_stop_reason| crate::openai::ChatChoice {
            index: 0,
            message: serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "content": "x",
            }))
            .unwrap(),
            finish_reason: "stop".into(),
            logprobs: None,
            token_ids: Vec::new(),
            flux_stop_reason,
        };
        let set = serde_json::to_value(choice(Some("catastrophic_guard"))).unwrap();
        assert_eq!(set["flux_stop_reason"], "catastrophic_guard");
        let unset = serde_json::to_value(choice(None)).unwrap();
        assert!(unset.get("flux_stop_reason").is_none());
    }
}

#[cfg(test)]
mod hidden_thinking_boundary_tests {
    use super::split_response_token_phases;

    #[test]
    fn hidden_boundary_splits_reasoning_from_following_content() {
        let output = [11, 12, 21, 22];
        let (reasoning, content) = split_response_token_phases(&output, true, Some(99), Some(2));
        assert_eq!(reasoning, Some(&output[..2]));
        assert_eq!(content, &output[2..]);
    }

    #[test]
    fn hidden_first_token_close_keeps_all_following_tokens_as_content() {
        let output = [21, 22];
        let (reasoning, content) = split_response_token_phases(&output, true, Some(99), Some(0));
        assert_eq!(reasoning, Some(&output[..0]));
        assert_eq!(content, &output[..]);
    }

    #[test]
    fn disabled_thinking_ignores_boundary_metadata() {
        let output = [11, 21];
        let (reasoning, content) = split_response_token_phases(&output, false, Some(99), Some(1));
        assert!(reasoning.is_none());
        assert_eq!(content, &output[..]);
    }
}

#[cfg(test)]
mod answer_prefix_tests {
    use super::{decode_response_token_text, repair_json_object_prefix_with_audit};
    use crate::openai::{
        ChatCompletionRequest, InterventionLedger, SharedInterventionLedger,
        snapshot_shared_interventions,
    };
    use std::sync::{Arc, Mutex};

    fn ledger() -> SharedInterventionLedger {
        Arc::new(Mutex::new(InterventionLedger::default()))
    }

    fn split_decode(
        output: &[u32],
        hidden_boundary: Option<usize>,
        reasoning: &str,
        content: &str,
        audit: Option<&SharedInterventionLedger>,
    ) -> (Option<String>, String) {
        decode_response_token_text(
            output,
            true,
            Some(99),
            hidden_boundary,
            |tokens| match tokens {
                [10] => Some(reasoning.to_string()),
                [20] => Some(content.to_string()),
                [] => Some(String::new()),
                _ => panic!("unexpected decoded token slice: {tokens:?}"),
            },
            |_, _| panic!("split responses must not use the fallback parser"),
            audit,
        )
    }

    #[test]
    fn literal_boundary_preserves_ascii_answer_prefixes() {
        for prefix in ["", " ", "\n\n", "\t\r\n"] {
            let output = [10, 99, 20];
            let original = output;
            let audit = ledger();
            let content = format!("{prefix}18");
            let decoded = split_decode(&output, None, "calculation", &content, Some(&audit));
            assert_eq!(decoded, (Some("calculation".into()), content));
            assert_eq!(output, original);
            assert!(
                snapshot_shared_interventions(Some(&audit))
                    .unwrap()
                    .events
                    .is_empty()
            );
        }
    }

    #[test]
    fn literal_boundary_preserves_unicode_answer_prefixes() {
        for prefix in ["\u{a0}", "\u{2003}", "\n\u{a0}\t\u{2028}"] {
            let content = format!("{prefix}18");
            assert_eq!(
                split_decode(&[10, 99, 20], None, "reasoning", &content, None).1,
                content
            );
        }
    }

    #[test]
    fn hidden_boundary_preserves_answer_prefix() {
        let decoded = split_decode(&[10, 20], Some(1), "calculation", "\n\u{2003}18", None);
        assert_eq!(decoded, (Some("calculation".into()), "\n\u{2003}18".into()));
    }

    #[test]
    fn hidden_zero_boundary_keeps_all_content_without_reasoning() {
        assert_eq!(
            split_decode(&[20], Some(0), "unused", "\n18", None),
            (None, "\n18".into())
        );
    }

    #[test]
    fn reporting_toggle_does_not_change_published_prefix() {
        let audit = ledger();
        let unreported = split_decode(&[10, 99, 20], None, "reasoning", "\n\n18", None);
        let reported = split_decode(&[10, 99, 20], None, "reasoning", "\n\n18", Some(&audit));
        assert_eq!(unreported, reported);
        assert!(
            snapshot_shared_interventions(Some(&audit))
                .unwrap()
                .events
                .is_empty()
        );
    }

    #[test]
    fn blank_reasoning_still_reports_its_actual_repair() {
        let audit = ledger();
        assert_eq!(
            split_decode(&[10, 99, 20], None, " \t\n", "\n18", Some(&audit)),
            (None, "\n18".into())
        );
        let snapshot = snapshot_shared_interventions(Some(&audit)).unwrap();
        assert_eq!(snapshot.events.len(), 1);
        assert_eq!(snapshot.events[0].code, "reasoning_output_repair");
        assert_eq!(snapshot.events[0].action, "drop_blank_reasoning");
    }

    #[test]
    fn empty_reasoning_and_content_remain_empty() {
        let audit = ledger();
        assert_eq!(
            split_decode(&[99], None, "unused", "unused", Some(&audit)),
            (None, String::new())
        );
        assert!(
            snapshot_shared_interventions(Some(&audit))
                .unwrap()
                .events
                .is_empty()
        );
    }

    #[test]
    fn disabled_thinking_keeps_the_entire_decoded_output() {
        let output = [10, 99, 20];
        let content = "\n<think>calculation</think>\n18";
        let decoded = decode_response_token_text(
            &output,
            false,
            Some(99),
            Some(1),
            |tokens| {
                assert_eq!(tokens, &output);
                Some(content.into())
            },
            |_, _| panic!("literal-delimiter support keeps the existing unsplit path"),
            None,
        );
        assert_eq!(decoded, (None, content.into()));
    }

    #[test]
    fn unsplit_response_with_delimiter_support_keeps_decoded_text() {
        let output = [10, 20];
        let decoded = decode_response_token_text(
            &output,
            true,
            Some(99),
            None,
            |tokens| {
                assert_eq!(tokens, &output);
                Some("\n18".into())
            },
            |_, _| panic!("the existing unsplit path must not invoke the parser"),
            None,
        );
        assert_eq!(decoded, (None, "\n18".into()));
    }

    #[test]
    fn unsplit_response_without_delimiter_support_uses_existing_parser() {
        let decoded = decode_response_token_text(
            &[10, 20],
            true,
            None,
            None,
            |_| Some("\nraw text".into()),
            |text, thinking| {
                assert_eq!(text, "\nraw text");
                assert!(thinking);
                (Some("parser reasoning".into()), "parser content".into())
            },
            None,
        );
        assert_eq!(
            decoded,
            (Some("parser reasoning".into()), "parser content".into())
        );
    }

    #[test]
    fn decode_errors_keep_the_existing_empty_fallbacks() {
        let decoded = decode_response_token_text(
            &[10, 99, 20],
            true,
            Some(99),
            None,
            |_| None,
            |_, _| panic!("split responses must not use the fallback parser"),
            None,
        );
        assert_eq!(decoded, (None, String::new()));
    }

    #[test]
    fn literal_boundary_still_wins_over_hidden_boundary() {
        let decoded = decode_response_token_text(
            &[10, 11, 99, 20],
            true,
            Some(99),
            Some(1),
            |tokens| match tokens {
                [10, 11] => Some("complete reasoning".into()),
                [20] => Some("\n18".into()),
                _ => panic!("wrong boundary selected: {tokens:?}"),
            },
            |_, _| panic!("unexpected parser"),
            None,
        );
        assert_eq!(decoded, (Some("complete reasoning".into()), "\n18".into()));
    }

    #[test]
    fn prefixed_json_repair_remains_parseable_and_reported() {
        let audit = ledger();
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "probe", "messages": [{"role": "user", "content": "json"}],
            "response_format": {"type": "json_object"}, "report_interventions": true
        }))
        .unwrap();
        let (_, content) = split_decode(
            &[10, 99, 20],
            None,
            "reasoning",
            "\n\u{a0}{\"{\"score\":50}",
            Some(&audit),
        );
        let repaired = repair_json_object_prefix_with_audit(&request, content, Some(&audit));
        assert_eq!(repaired, "\n{\"score\":50}");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&repaired).unwrap(),
            serde_json::json!({"score": 50})
        );
        let snapshot = snapshot_shared_interventions(Some(&audit)).unwrap();
        assert_eq!(snapshot.events.len(), 1);
        assert_eq!(snapshot.events[0].code, "json_output_repair");
        assert_eq!(snapshot.events[0].action, "repair_json_object_prefix");
    }

    #[test]
    fn legal_prefixed_json_is_preserved_without_interventions() {
        let audit = ledger();
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "probe", "messages": [{"role": "user", "content": "json"}],
            "response_format": {"type": "json_object"}, "report_interventions": true
        }))
        .unwrap();
        let (_, content) = split_decode(
            &[10, 20],
            Some(1),
            "reasoning",
            "\n\t{\"score\":50}",
            Some(&audit),
        );
        assert_eq!(
            repair_json_object_prefix_with_audit(&request, content, Some(&audit)),
            "\n\t{\"score\":50}"
        );
        assert!(
            snapshot_shared_interventions(Some(&audit))
                .unwrap()
                .events
                .is_empty()
        );
    }

    #[test]
    fn explicit_stop_processing_keeps_the_preserved_answer_prefix() {
        let (_, content) = split_decode(&[10, 99, 20], None, "reasoning", "\n18<STOP>tail", None);
        assert_eq!(
            super::strip_stop_sequences(content, &["<STOP>".into()]),
            "\n18"
        );
    }

    #[test]
    fn preserved_prefix_still_participates_in_earliest_stop_matching() {
        let (_, content) = split_decode(&[10, 99, 20], None, "reasoning", "\n\n18<LATER>", None);
        assert_eq!(
            super::strip_stop_sequences(content, &["<LATER>".into(), "\n".into()]),
            ""
        );
    }
}

#[cfg(test)]
#[path = "glm_tool_publication_blocking_tests.rs"]
mod glm_tool_publication_blocking_tests;

/// Publication validation for one choice's parsed tool calls. The native GLM
/// contract validates exactly (no backfill, fuzzy-name repair or path
/// rewriting, which would change raw native values); every other parser keeps
/// the legacy repair + validation pipeline unchanged.
fn validate_parsed_tool_calls(
    native_glm: bool,
    mut calls: Vec<tool_parser::ToolCall>,
    tools: &[tool_parser::ToolDefinition],
    tool_choice: Option<&tool_parser::ToolChoice>,
    cwd_hint: Option<&str>,
    intervention_ledger: Option<&crate::openai::SharedInterventionLedger>,
) -> tool_parser::ValidatedToolCalls {
    if native_glm {
        return tool_parser::native_publication::validate_calls(calls, tools, tool_choice);
    }
    let backfilled = tool_parser::backfill_required_params(&mut calls, tools);
    let normalized = cwd_hint.map_or(0, |cwd| tool_parser::normalize_paths(&mut calls, cwd));
    for _ in 0..backfilled {
        crate::openai::record_shared_intervention(
            intervention_ledger,
            "tool_output_repair",
            "backfill_required_params",
        );
    }
    for _ in 0..normalized {
        crate::openai::record_shared_intervention(
            intervention_ledger,
            "tool_output_repair",
            "normalize_paths",
        );
    }
    let original_names: Vec<(String, String)> = calls
        .iter()
        .map(|call| (call.id.clone(), call.function.name.clone()))
        .collect();
    let validated = tool_parser::validate_tool_calls(calls, tools);
    for call in &validated.valid {
        if original_names
            .iter()
            .any(|(id, name)| id == &call.id && name != &call.function.name)
        {
            crate::openai::record_shared_intervention(
                intervention_ledger,
                "tool_output_repair",
                "fuzzy_tool_name",
            );
        }
    }
    validated
}

/// Native GLM: match streaming's explicit rejection feedback. Never turn
/// invalid native calls into an unexplained empty response.
fn native_rejection_feedback(content: Option<String>, errors: &[String]) -> Option<String> {
    if errors.is_empty() {
        return content;
    }
    let feedback = format!("[flux] Tool call rejected: {}", errors.join("; "));
    Some(match content {
        Some(text) if !text.is_empty() => format!("{text}\n{feedback}"),
        _ => feedback,
    })
}

fn enforce_parallel_tool_call_policy<T>(calls: &mut Vec<T>, allow_parallel: bool) -> bool {
    if !allow_parallel && calls.len() > 1 {
        calls.truncate(1);
        return true;
    }
    false
}

#[cfg(test)]
mod tool_repair_intervention_tests {
    use super::{repair_json_object_prefix_with_audit, validate_parsed_tool_calls};
    use crate::openai::{
        InterventionLedger, SharedInterventionLedger, snapshot_shared_interventions,
    };
    use crate::tool_parser::{FunctionCall, FunctionDefinition, ToolCall, ToolDefinition};
    use std::sync::{Arc, Mutex};

    #[test]
    fn published_blocking_repairs_are_recorded_in_the_intervention_ledger() {
        let ledger: SharedInterventionLedger = Arc::new(Mutex::new(InterventionLedger::default()));
        let tools = [ToolDefinition {
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
        let calls = vec![ToolCall {
            id: "call_repair".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "probe".into(),
                arguments: r#"{"count":"2","file_path":"/work/tree/file.txt"}"#.into(),
            },
        }];

        let validated = validate_parsed_tool_calls(
            false,
            calls,
            &tools,
            None,
            Some("/work/tree"),
            Some(&ledger),
        );
        assert_eq!(validated.valid.len(), 1);
        assert!(validated.errors.is_empty());

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
    fn fuzzy_name_repair_is_recorded_before_publication() {
        let ledger: SharedInterventionLedger = Arc::new(Mutex::new(InterventionLedger::default()));
        let tools = [ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "get_weather".into(),
                description: None,
                parameters: Some(serde_json::json!({"type": "object"})),
            },
        }];
        let calls = vec![ToolCall {
            id: "call_fuzzy".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "Get_Weather".into(),
                arguments: "{}".into(),
            },
        }];

        let validated = validate_parsed_tool_calls(false, calls, &tools, None, None, Some(&ledger));
        assert_eq!(validated.valid[0].function.name, "get_weather");
        let snapshot = snapshot_shared_interventions(Some(&ledger)).unwrap();
        assert!(snapshot.events.iter().any(|event| {
            event.code == "tool_output_repair" && event.action == "fuzzy_tool_name"
        }));
    }

    #[test]
    fn json_prefix_repair_is_recorded_before_publication() {
        let ledger: SharedInterventionLedger = Arc::new(Mutex::new(InterventionLedger::default()));
        let request: crate::openai::ChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": "probe",
                "messages": [{"role": "user", "content": "json"}],
                "response_format": {"type": "json_object"},
                "report_interventions": true
            }))
            .unwrap();
        let repaired = repair_json_object_prefix_with_audit(
            &request,
            "{\"{\"score\":50}".into(),
            Some(&ledger),
        );
        assert_eq!(repaired, "{\"score\":50}");
        let snapshot = snapshot_shared_interventions(Some(&ledger)).unwrap();
        assert!(snapshot.events.iter().any(|event| {
            event.code == "json_output_repair" && event.action == "repair_json_object_prefix"
        }));
    }
}

/// Request warnings plus one timeout warning per timed-out choice; `None`
/// when there is nothing to report (the field is then omitted).
fn blocking_warnings(
    contract: Vec<String>,
    choices: &[crate::openai::ChatChoice],
) -> Option<Vec<String>> {
    let all: Vec<String> = contract
        .into_iter()
        .chain(
            choices
                .iter()
                .filter_map(|c| super::request_contract::timeout_warning(c.flux_stop_reason)),
        )
        .collect();
    (!all.is_empty()).then_some(all)
}

/// The guard behind a choice finish, reported as `flux_stop_reason`.
/// Timeout causes (`"request_timeout_stall"` / `"request_timeout_total"`) are
/// reported even on `"length"` finishes; non-timeout guards (`"watchdog"` /
/// `"catastrophic_guard"`) are reported only when `finish_reason == "stop"`.
fn guard_stop_reason(
    finish_reason: &str,
    response: &super::inference_types::InferenceResponse,
) -> Option<&'static str> {
    let timeout = response
        .stop_cause
        .filter(|c| super::request_contract::is_timeout_cause(c));
    timeout.or((finish_reason == "stop")
        .then_some(response.stop_cause)
        .flatten())
}

/// Build the assistant message + finish_reason for one choice. Tool
/// parsing, validation, content-strip + refusal-classifier all live
/// here.
fn build_choice_message(
    state: &AppState,
    req: &ChatCompletionRequest,
    response: &super::inference_types::InferenceResponse,
    reasoning_content_i: Option<String>,
    output_text_i: String,
    tools_active: bool,
    cwd_hint: Option<&str>,
    choice_idx: usize,
    intervention_ledger: Option<&crate::openai::SharedInterventionLedger>,
) -> (crate::openai::ChatMessage, String) {
    let _ = response; // currently only used for finish_reason.clone() below
    let mut message = crate::openai::ChatMessage {
        role: "assistant".to_string(),
        reasoning_content: reasoning_content_i.clone(),
        reasoning: reasoning_content_i,
        annotations: crate::citation::merged_annotations(&output_text_i),
        refusal: None,
        content: Some(output_text_i.clone()),
        tool_calls: None,
    };
    let mut finish_reason_i = response.finish_reason.clone();

    if tools_active {
        let log_tool_raw = std::env::var("FLUX_LOG_TOOL_RAW")
            .ok()
            .or_else(|| std::env::var("ATLAS_LOG_TOOL_RAW").ok());
        if log_tool_raw.as_deref() == Some("1") {
            tracing::info!(
                target: "flux::tool_debug",
                "raw pre-parse output (tools_active, choice {choice_idx}): {output_text_i:?}"
            );
        }
        let (content, tool_calls_i) = if intervention_ledger.is_some() {
            let parser_name = state
                .tool_call_parser
                .as_ref()
                .map_or("", |parser| parser.name());
            let (content, calls, audit) =
                tool_parser::parse_tool_calls_with_audit(&output_text_i, parser_name);
            for _ in 0..audit.recovery_count {
                crate::openai::record_shared_intervention(
                    intervention_ledger,
                    "tool_parser_recovery",
                    "recovered_noncanonical_call",
                );
            }
            (content, calls)
        } else {
            tool_parser::parse_tool_calls(&output_text_i)
        };
        if !tool_calls_i.is_empty() {
            let tools_ref = req.tools.as_ref().cloned().unwrap_or_default();
            let native_glm = state
                .tool_call_parser
                .as_ref()
                .is_some_and(|p| p.name() == "glm_xml");
            let mut validated = validate_parsed_tool_calls(
                native_glm,
                tool_calls_i,
                &tools_ref,
                req.tool_choice.as_ref(),
                cwd_hint,
                intervention_ledger,
            );
            if !validated.errors.is_empty() {
                for err in &validated.errors {
                    tracing::warn!("Tool call validation error: {err}");
                    crate::openai::record_shared_intervention(
                        intervention_ledger,
                        "tool_output_policy",
                        "reject_invalid_tool_call",
                    );
                }
            }
            if enforce_parallel_tool_call_policy(
                &mut validated.valid,
                req.parallel_tool_calls.unwrap_or(true),
            ) {
                crate::openai::record_shared_intervention(
                    intervention_ledger,
                    "tool_output_policy",
                    "truncate_parallel_calls",
                );
            }
            let content =
                if crate::api::chat_phases::agent_guards_enabled() && !validated.valid.is_empty() {
                    content.map(|mut c| {
                        let original = c.clone();
                        for tag in &["</parameter>", "</function>", "</tool_call>", "<tool_call>"] {
                            c = c.replace(tag, "");
                        }
                        while let Some(start) = c.find("<function=") {
                            let end = c[start..]
                                .find('>')
                                .map(|p| start + p + 1)
                                .unwrap_or(c.len());
                            c = format!("{}{}", &c[..start], &c[end..]);
                        }
                        while let Some(start) = c.find("```") {
                            let after_open = start + 3;
                            let Some(rel_close) = c[after_open..].find("```") else {
                                break;
                            };
                            let close_end = after_open + rel_close + 3;
                            c = format!("{}{}", &c[..start], &c[close_end..]);
                        }
                        let cleaned = c.trim().to_string();
                        if cleaned != original {
                            crate::openai::record_shared_intervention(
                                intervention_ledger,
                                "tool_output_repair",
                                "strip_tool_markup",
                            );
                        }
                        cleaned
                    })
                } else {
                    content
                };
            message.content = if native_glm {
                if !validated.errors.is_empty() {
                    crate::openai::record_shared_intervention(
                        intervention_ledger,
                        "tool_output_repair",
                        "append_native_rejection_feedback",
                    );
                }
                native_rejection_feedback(content, &validated.errors)
            } else {
                content
            };
            if !validated.valid.is_empty() {
                for tc in &validated.valid {
                    let p: String = tc.function.arguments.chars().take(120).collect();
                    let s = ["", "…"][usize::from(tc.function.arguments.len() > p.len())];
                    tracing::info!("Tool call: {}({p}{s})", tc.function.name);
                    crate::metrics::TOOL_CALLS_TOTAL.inc();
                }
                message.tool_calls = Some(validated.valid);
                finish_reason_i = "tool_calls".to_string();
            }
        }
    }

    // Heuristic refusal classifier (opt-in via FLUX_AGENT_GUARDS=1 so benign
    // answers opening with "I cannot ..." never have message.content wiped).
    if crate::api::chat_phases::agent_guards_enabled()
        && message.tool_calls.is_none()
        && let Some(content_text) = message.content.as_deref()
        && let Some(refusal_sentence) = crate::refusal::detect(content_text)
    {
        crate::openai::record_shared_intervention(
            intervention_ledger,
            "response_classification",
            "publish_refusal_field",
        );
        message.refusal = Some(refusal_sentence);
        message.content = None;
        message.annotations = None;
    }

    (message, finish_reason_i)
}

#[cfg(test)]
mod parallel_tool_call_policy_tests {
    use super::enforce_parallel_tool_call_policy;

    #[test]
    fn explicit_false_publishes_only_first_valid_call() {
        let mut calls = vec!["first", "second", "third"];
        assert!(enforce_parallel_tool_call_policy(&mut calls, false));
        assert_eq!(calls, ["first"]);
    }

    #[test]
    fn default_or_true_preserves_multi_call_output() {
        let mut calls = vec!["first", "second", "third"];
        assert!(!enforce_parallel_tool_call_policy(&mut calls, true));
        assert_eq!(calls, ["first", "second", "third"]);
    }
}

/// Convert internal logprobs to OpenAI `ChoiceLogprobs` format.
fn build_logprobs(
    state: &AppState,
    response: &super::inference_types::InferenceResponse,
) -> Option<crate::openai::ChoiceLogprobs> {
    if response.logprobs.is_empty() {
        return None;
    }
    Some(crate::openai::ChoiceLogprobs {
        content: response
            .logprobs
            .iter()
            .map(|lp| {
                let token_str = state.tokenizer.decode(&[lp.token_id]).unwrap_or_default();
                crate::openai::TokenLogprobInfo {
                    token: token_str,
                    logprob: lp.logprob,
                    bytes: None,
                    top_logprobs: lp
                        .top
                        .iter()
                        .map(|&(tid, lp_val)| crate::openai::TopLogprob {
                            token: state.tokenizer.decode(&[tid]).unwrap_or_default(),
                            logprob: lp_val,
                            bytes: None,
                        })
                        .collect(),
                }
            })
            .collect(),
    })
}

/// Build the final `ChatCompletionResponse` plus metrics, store, and
/// rate-limit refund. Returns the JSON-encoded HTTP response.
#[allow(clippy::too_many_arguments)]
fn finalize_response(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    dump_seq: Option<u64>,
    all_choices: Vec<crate::openai::ChatChoice>,
    total_completion_tokens: usize,
    total_decode_tokens: usize,
    first_ttft: f64,
    total_decode_time_ms: f64,
    total_reasoning_tokens: u32,
    total_cached_prompt_tokens: u32,
    total_think_nudges: u32,
    prompt_len: usize,
    intervention_ledger: Option<crate::openai::SharedInterventionLedger>,
    rendered_prompt_evidence_source: Option<(String, String)>,
    warnings: Vec<String>,
) -> Response {
    let tokens_per_second = if total_decode_time_ms > 0.0 && total_decode_tokens > 0 {
        total_decode_tokens as f64 / (total_decode_time_ms / 1000.0)
    } else {
        0.0
    };
    let usage = Usage {
        prompt_tokens: prompt_len,
        completion_tokens: total_completion_tokens,
        total_tokens: prompt_len + total_completion_tokens,
        prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
            cached_tokens: total_cached_prompt_tokens as usize,
            audio_tokens: 0,
        }),
        completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
            reasoning_tokens: total_reasoning_tokens as usize,
            audio_tokens: 0,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
            think_nudges: total_think_nudges,
        }),
        time_to_first_token_ms: first_ttft,
        response_tokens_per_second: tokens_per_second,
    };

    let completion_id = format!("chatcmpl-{}", crate::openai::uuid_v4());
    let created_at = crate::openai::unix_timestamp();
    let completion = ChatCompletionResponse {
        id: completion_id.clone(),
        object: "chat.completion".to_string(),
        created: created_at,
        model: state.model_name.clone(),
        system_fingerprint: Some("fp_atlas".to_string()),
        choices: all_choices,
        usage: usage.clone(),
        service_tier: req.service_tier.clone(),
        metadata: req.metadata.clone(),
        interventions: crate::openai::snapshot_shared_interventions(intervention_ledger.as_ref()),
        rendered_prompt_evidence: rendered_prompt_evidence_source.map(|(nonce, prompt_sha256)| {
            crate::openai::RenderedPromptEvidence::from_digest(
                nonce,
                completion_id.clone(),
                prompt_sha256,
            )
        }),
        warnings: None,
    };
    let completion = ChatCompletionResponse {
        warnings: blocking_warnings(warnings, &completion.choices),
        ..completion
    };

    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(prompt_len as u64);
    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(total_completion_tokens as u64);
    crate::metrics::TTFT_SECONDS.observe(first_ttft / 1000.0);

    // Completion-storage backend: when `store: true`, persist the
    // serialized body so a subsequent GET /v1/chat/completions/{id}
    // can return it. Bounded LRU + TTL in response_store.
    if req.store.unwrap_or(false)
        && let Ok(body) = serde_json::to_value(&completion)
    {
        state
            .response_store
            .insert(crate::response_store::StoredEntry {
                id: completion_id,
                kind: crate::response_store::StoredKind::ChatCompletion,
                model: state.model_name.clone(),
                created_at,
                messages: Vec::new(),
                body,
                last_access: std::time::Instant::now(),
            });
    }

    // Rate-limit true-up. Middleware admitted with a conservative
    // reservation of `max_seq_len` tokens; refund the difference.
    if let Some(axum::extract::Extension(ref ctx)) = req_ctx
        && ctx.mark_settled()
    {
        let actual = (prompt_len + total_completion_tokens) as u64;
        let refund = ctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            state.rate_limiter.refund_tokens(&ctx.identity, refund);
        }
    }

    // --dump: record the non-streaming response body, correlated with
    // the request via the shared seq number.
    if let (Some(seq), Some(dump)) = (dump_seq, state.dump_writer.as_ref()) {
        dump.dump_response("/v1/chat/completions", seq, &completion, false);
    }

    Json(completion).into_response()
}

#[cfg(test)]
mod active_guard_tests {
    use super::BlockingActiveGuard;

    #[tokio::test]
    async fn blocking_active_guard_decrements_when_future_is_dropped_on_disconnect() {
        let gauge = prometheus::IntGauge::new("test_blocking_active_disconnect", "test")
            .expect("create isolated gauge");
        gauge.inc();
        assert_eq!(gauge.get(), 1);

        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
        let mut fut = Box::pin(async {
            let _guard = BlockingActiveGuard(&gauge);
            let _ = rx.await;
        });

        // Poll once so the guard is live inside the suspended future.
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(fut.as_mut().poll(&mut cx).is_pending());
        assert_eq!(gauge.get(), 1);

        // Drop the future to simulate a blocking HTTP client disconnecting mid-request.
        drop(fut);
        assert_eq!(
            gauge.get(),
            0,
            "dropping a suspended blocking future must decrement REQUESTS_ACTIVE exactly once"
        );
    }

    #[test]
    fn blocking_path_decrements_requests_active_only_via_raii_guard() {
        let source = include_str!("chat_blocking.rs");
        let prod = source
            .split("mod active_guard_tests")
            .next()
            .expect("production source precedes active_guard_tests");
        assert!(
            !prod.contains("REQUESTS_ACTIVE.dec()"),
            "manual REQUESTS_ACTIVE.dec() calls in chat_blocking.rs would double-decrement against BlockingActiveGuard"
        );
    }
}
