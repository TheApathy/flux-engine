// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use crate::main_modules::model_host::CurrentModel;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

// Sibling-cluster items hoisted from the original `api.rs`. These uses
// give every sub-file access to helpers that the un-split file took for
// granted via single-module visibility.
use super::chat::chat_completions_inner;
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::failures::{
    F23ProgressMetrics, F29EnvironmentFact, F37FailureClass, F39FailureCache,
    F39PermanentFailureMatch, F49DuplicateWrite, append_f7_reminder_to_last_user,
    build_f7_stall_reminder, bump_f12_tool_call_count, check_loop_watchdog,
    collect_f7_stall_buckets, f23_build_reminder, f23_normalize_and_hash, f23_refuse_threshold,
    f23_score_progress, f23_warn_threshold, f28_text_looks_like_error,
    f29_extract_binary_from_error_line, f29_extract_environment_facts,
    f29_inject_environment_facts, f31_inject_hard_refusal, f32_reposition_failed_tool_result,
    f37_classify_failure, f39_build_circuit_breaker_banner, f39_build_failure_cache,
    f39_class_label, f39_detect_recent_retries, f39_extract_binary_name,
    f44_check_permanent_failure, f49_build_banner, f49_detect_duplicate_writes,
    f49_extract_write_path_and_content, f50_append_original_error, f60_disable_mtp_for_request,
    flush_content_sanitizer, prepend_reminder_to_system, recent_message_is_tool_error,
    strip_xml_leaks_from_assistant_content,
};
use super::inference_impl::{
    extract_thinking, strip_stop_sequences, tokenize_stop_sequences, try_admit_scheduler,
};
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};
use super::strip::strip_thinking_tags;

/// Owns the scheduler cancellation edge for one legacy-completions SSE
/// response. If the HTTP body is dropped before `Done`, dropping the stream
/// closure flips the shared flag with Release ordering so generation cannot
/// continue after its consumer has gone away.
struct CompletionCancelGuard {
    scheduler_cancel: Arc<AtomicBool>,
    armed: AtomicBool,
}

impl CompletionCancelGuard {
    fn new(scheduler_cancel: Arc<AtomicBool>) -> Self {
        Self {
            scheduler_cancel,
            armed: AtomicBool::new(true),
        }
    }

    fn cancel(&self) {
        self.scheduler_cancel.store(true, Ordering::Release);
        self.armed.store(false, Ordering::Release);
    }

    fn disarm(&self) {
        self.armed.store(false, Ordering::Release);
    }
}

impl Drop for CompletionCancelGuard {
    fn drop(&mut self) {
        if self.armed.load(Ordering::Acquire) {
            self.scheduler_cancel.store(true, Ordering::Release);
        }
    }
}

fn completion_stream_error(guard: &CompletionCancelGuard, msg: String) -> Event {
    guard.cancel();
    let payload = serde_json::json!({
        "error": {
            "message": msg,
            "type": "server_error",
            "code": 500,
        }
    })
    .to_string();
    Event::default().data(payload)
}

// Re-export sibling helpers via crate::api::* for short paths.
use super::failures::*;
use super::inference_types::*;
use super::sanitizer::*;

#[cfg(test)]
#[path = "completions_request_tests.rs"]
mod completion_request_tests;

fn checked_completion_deadline(
    timeout_secs: f32,
    now: std::time::Instant,
) -> Result<Option<std::time::Instant>, &'static str> {
    if !timeout_secs.is_finite() || timeout_secs < 0.0 {
        return Err("timeout must be a finite, nonnegative number of seconds");
    }
    if timeout_secs == 0.0 {
        return Ok(None);
    }
    let duration = std::time::Duration::try_from_secs_f32(timeout_secs)
        .map_err(|_| "timeout exceeds the supported duration range")?;
    now.checked_add(duration)
        .map(Some)
        .ok_or("timeout exceeds the supported deadline range")
}

/// Post-first-token deadline under the request-lifecycle contract, matching
/// `/v1/chat/completions` (`chat::sampling_setup::contract_decode_deadline`).
fn completion_decode_deadline(
    base_timeout_s: u32,
    dsv41: bool,
    explicit_s: Option<f32>,
    max_tokens: usize,
    prompt_len: usize,
    now: std::time::Instant,
) -> Option<std::time::Instant> {
    use crate::api::request_contract::{
        TimeoutSource, derive_timeout_policy, log_timeout_policy, policy_deadline, timeout_config,
    };
    let base = if dsv41 { 0 } else { base_timeout_s };
    let policy = derive_timeout_policy(&timeout_config(base), explicit_s, max_tokens, prompt_len);
    log_timeout_policy(&policy);
    (policy.source == TimeoutSource::Derived)
        .then(|| policy_deadline(&policy, now))
        .flatten()
}

#[allow(clippy::result_large_err)]
fn validate_completion_input(req: &CompletionRequest) -> Result<(), Response> {
    if req.requested_max_tokens == Some(0) {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "max_tokens must be at least 1".into(),
            Some("max_tokens"),
            None,
        ));
    }
    if req.stop.len() > 4 {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "stop must contain at most 4 sequences".into(),
            Some("stop"),
            None,
        ));
    }
    if req.stop.iter().any(String::is_empty) {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "stop sequences must not be empty".into(),
            Some("stop"),
            None,
        ));
    }
    if req.stream && req.return_token_ids {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "return_token_ids is currently supported only for blocking completions".into(),
            Some("return_token_ids"),
            None,
        ));
    }
    if let Some(temperature) = req.temperature
        && !(0.0..=2.0).contains(&temperature)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "temperature must be between 0 and 2".into(),
            Some("temperature"),
            None,
        ));
    }
    if let Some(top_p) = req.top_p
        && !(top_p > 0.0 && top_p <= 1.0)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "top_p must be between 0 (exclusive) and 1".into(),
            Some("top_p"),
            None,
        ));
    }
    if let Some(top_n_sigma) = req.top_n_sigma
        && !(top_n_sigma.is_finite() && top_n_sigma >= 0.0)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "top_n_sigma must be finite and non-negative".into(),
            Some("top_n_sigma"),
            None,
        ));
    }
    if let Some(min_p) = req.min_p
        && !(0.0..=1.0).contains(&min_p)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "min_p must be between 0 and 1".into(),
            Some("min_p"),
            None,
        ));
    }
    if let Some(repetition_penalty) = req.repetition_penalty
        && !(repetition_penalty.is_finite() && repetition_penalty > 0.0)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "repetition_penalty must be finite and greater than 0".into(),
            Some("repetition_penalty"),
            None,
        ));
    }
    for (name, penalty) in [
        ("presence_penalty", req.presence_penalty),
        ("frequency_penalty", req.frequency_penalty),
    ] {
        if let Some(value) = penalty
            && !(-2.0..=2.0).contains(&value)
        {
            return Err(openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                format!("{name} must be between -2.0 and 2.0"),
                Some(name),
                None,
            ));
        }
    }
    if let Some(logit_bias) = &req.logit_bias {
        for (token_id, bias) in logit_bias {
            if token_id.parse::<u32>().is_err() {
                return Err(openai_error_response_with_param(
                    StatusCode::BAD_REQUEST,
                    format!("logit_bias key '{token_id}' must be a token ID"),
                    Some("logit_bias"),
                    None,
                ));
            }
            if !(-100.0..=100.0).contains(bias) {
                return Err(openai_error_response_with_param(
                    StatusCode::BAD_REQUEST,
                    "logit_bias values must be between -100 and 100".into(),
                    Some("logit_bias"),
                    None,
                ));
            }
        }
    }
    if let Some(timeout) = req.timeout
        && let Err(message) = checked_completion_deadline(timeout, std::time::Instant::now())
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            message.to_owned(),
            Some("timeout"),
            Some("invalid_timeout"),
        ));
    }
    Ok(())
}

/// The prompt ids for a completion request.
///
/// Exact-token prompts -- `prompt_token_ids`, or `prompt` given as token ids /
/// token-id batches -- are prefilled verbatim: no tokenization, no think-prefix.
/// (`prompt_token_ids` is what the DFlash hidden-capture harness uses so its
/// hiddens align with the trainer's own input_ids.) They must be non-empty and
/// inside the vocabulary. Text is tokenized; for thinking models without a
/// `</think>` it gets `<think></think>\n\n` prepended to suppress think-tag
/// leakage (the model expects it in raw completions mode), and for
/// deepseek_v41 it is tokenized as given with BOS prepended, as the Python
/// server does.
fn resolve_prompt_tokens(
    req: &CompletionRequest,
    tokenizer: &crate::tokenizer::ChatTokenizer,
    dsv41: bool,
) -> Result<Vec<u32>, String> {
    resolve_prompt_tokens_with_audit(req, tokenizer, dsv41, None)
}

fn resolve_prompt_tokens_with_audit(
    req: &CompletionRequest,
    tokenizer: &crate::tokenizer::ChatTokenizer,
    dsv41: bool,
    ledger: Option<&crate::openai::SharedInterventionLedger>,
) -> Result<Vec<u32>, String> {
    use crate::openai::CompletionPrompt;
    let exact = match (&req.prompt_token_ids, &req.prompt) {
        (Some(ids), _) => Some(("prompt_token_ids", ids)),
        (None, CompletionPrompt::Tokens(ids)) => Some(("prompt", ids)),
        (None, CompletionPrompt::Text(_)) => None,
    };
    if let Some((field, ids)) = exact {
        if ids.is_empty() {
            return Err(format!("{field} provided but empty"));
        }
        let vocab = tokenizer.vocab_size();
        if let Some(bad) = ids.iter().find(|&&t| t as usize >= vocab) {
            return Err(format!(
                "{field} contains token id {bad}, outside the vocabulary (size {vocab})"
            ));
        }
        return Ok(ids.clone());
    }
    let CompletionPrompt::Text(text) = &req.prompt else {
        unreachable!("token prompts returned above")
    };
    if dsv41 {
        return tokenizer
            .encode(text)
            .map(|t| {
                let bos = tokenizer.inner().token_to_id(crate::dsv41::encoding::BOS);
                if bos.is_some_and(|bos| t.first() != Some(&bos)) {
                    crate::openai::record_shared_intervention(
                        ledger,
                        "completion_bos_prefix",
                        "prepend_prompt",
                    );
                }
                with_bos(t, bos)
            })
            .map_err(|e| format!("Tokenization error: {e}"));
    }
    let inject_thinking = tokenizer.supports_thinking() && !text.contains("</think>");
    let raw_prompt = if inject_thinking {
        format!("<think></think>\n\n{text}")
    } else {
        text.clone()
    };
    let tokens = tokenizer
        .encode(&raw_prompt)
        .map_err(|e| format!("Tokenization error: {e}"))?;
    if inject_thinking {
        crate::openai::record_shared_intervention(
            ledger,
            "completion_thinking_prefix",
            "prepend_prompt",
        );
    }
    Ok(tokens)
}

fn completion_output_with_audit(
    text: String,
    stops: &[String],
    ledger: Option<&crate::openai::SharedInterventionLedger>,
) -> String {
    let original = ledger.map(|_| text.clone());
    let stopped = strip_stop_sequences(text, stops);
    if original.as_ref().is_some_and(|text| *text != stopped) {
        crate::openai::record_shared_intervention(
            ledger,
            "completion_stop_sequence",
            "strip_output",
        );
    }
    let cleaned = strip_thinking_tags(&stopped);
    if cleaned != stopped {
        crate::openai::record_shared_intervention(
            ledger,
            "completion_output_cleanup",
            "strip_output",
        );
    }
    cleaned
}

/// Prepend `bos` unless the ids already start with it (app.py `add_bos`).
fn with_bos(ids: Vec<u32>, bos: Option<u32>) -> Vec<u32> {
    match bos {
        Some(bos) if ids.first() != Some(&bos) => std::iter::once(bos).chain(ids).collect(),
        _ => ids,
    }
}

#[cfg(test)]
mod prompt_token_tests {
    use super::resolve_prompt_tokens;
    use crate::openai::CompletionRequest;

    fn tokenizer() -> crate::tokenizer::ChatTokenizer {
        crate::tokenizer::ChatTokenizer::from_model_dir(
            std::path::Path::new(&crate::test_paths::resolve(
                "${FLUX_MODELS_ROOT}/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K",
            )),
            1,
            true,
            "deepseek_v41",
            None,
        )
        .expect("checkpoint tokenizer (a missing tokenizer is a failure, not a skip)")
    }

    fn req(v: serde_json::Value) -> CompletionRequest {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    #[ignore = "needs FLUX_MODELS_ROOT/FLUX_ATLAS_ROOT + DeepSeek-V4.1 checkpoint"]
    fn token_prompts_prefill_exactly_those_ids() {
        let tok = tokenizer();
        for dsv41 in [false, true] {
            let ids = resolve_prompt_tokens(
                &req(serde_json::json!({"prompt": [100, 200, 300]})),
                &tok,
                dsv41,
            );
            assert_eq!(ids, Ok(vec![100, 200, 300]), "dsv41={dsv41}");
            let ids = resolve_prompt_tokens(
                &req(serde_json::json!({"prompt": [[100, 200], [300]]})),
                &tok,
                dsv41,
            );
            assert_eq!(ids, Ok(vec![100, 200, 300]), "batches, dsv41={dsv41}");
        }
        // prompt_token_ids still wins over prompt
        let ids = resolve_prompt_tokens(
            &req(serde_json::json!({"prompt": [1, 2], "prompt_token_ids": [7, 8]})),
            &tok,
            false,
        );
        assert_eq!(ids, Ok(vec![7, 8]));
        // CONTROL: the same ids as text go through the tokenizer (and, for a
        // thinking model, the think prefix) -- they are not those ids.
        let text = resolve_prompt_tokens(
            &req(serde_json::json!({"prompt": "100 200 300"})),
            &tok,
            false,
        )
        .unwrap();
        assert_ne!(text, vec![100, 200, 300]);
        assert_eq!(text, tok.encode("<think></think>\n\n100 200 300").unwrap());
    }

    #[test]
    #[ignore = "needs FLUX_MODELS_ROOT/FLUX_ATLAS_ROOT + DeepSeek-V4.1 checkpoint"]
    fn out_of_vocab_and_empty_token_prompts_are_refused() {
        let tok = tokenizer();
        let vocab = tok.inner().get_vocab_size(true) as u32;
        let last = resolve_prompt_tokens(
            &req(serde_json::json!({"prompt": [vocab - 1]})),
            &tok,
            false,
        );
        assert_eq!(last, Ok(vec![vocab - 1]));
        let err =
            resolve_prompt_tokens(&req(serde_json::json!({"prompt": [5, vocab]})), &tok, false)
                .unwrap_err();
        assert!(
            err.contains(&format!("token id {vocab}")) && err.contains("outside the vocabulary"),
            "{err}"
        );
        let err = resolve_prompt_tokens(
            &req(serde_json::json!({"prompt_token_ids": [vocab + 7]})),
            &tok,
            false,
        )
        .unwrap_err();
        assert!(
            err.starts_with("prompt_token_ids contains token id"),
            "{err}"
        );
        let err = resolve_prompt_tokens(
            &req(serde_json::json!({"prompt_token_ids": []})),
            &tok,
            false,
        )
        .unwrap_err();
        assert!(err.contains("empty"), "{err}");
    }
}

#[cfg(test)]
mod dsv41_bos_tests {
    use super::with_bos;

    #[test]
    fn bos_is_prepended_once() {
        assert_eq!(with_bos(vec![5, 6], Some(0)), vec![0, 5, 6]);
        assert_eq!(with_bos(vec![0, 5], Some(0)), vec![0, 5]);
        assert_eq!(with_bos(vec![5], None), vec![5]);
        assert_eq!(with_bos(Vec::new(), Some(0)), vec![0]);
    }
}

pub async fn completions(
    CurrentModel(state): CurrentModel,
    req: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let mut ingress_guard = crate::metrics::IngressGuard::new();
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    if let Err(response) = validate_completion_input(&req) {
        return response;
    }
    let intervention_ledger = req.report_interventions.then(|| {
        Arc::new(std::sync::Mutex::new(
            crate::openai::InterventionLedger::default(),
        ))
    });
    let prompt_tokens = match resolve_prompt_tokens_with_audit(
        &req,
        &state.tokenizer,
        state.dsv41,
        intervention_ledger.as_ref(),
    ) {
        Ok(ids) => ids,
        Err(message) => {
            return openai_error_response(StatusCode::BAD_REQUEST, message);
        }
    };
    ingress_guard.finish();

    let prompt_len = prompt_tokens.len();
    // The one settings snapshot this request resolves from (`/v1/flux/settings`).
    let defaults = state.runtime.snapshot();
    let (max_tokens, max_tokens_defaulted) = req.max_tokens_or(defaults.default_max_tokens);
    let max_tokens = if max_tokens_defaulted {
        super::context_budget::defaulted_output_allowance(max_tokens, prompt_len, state.max_seq_len)
    } else {
        max_tokens
    };
    if super::context_budget::admitted_total(prompt_len, max_tokens, state.max_seq_len).is_none() {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "Context too long: {prompt_len} prompt + {} requested output tokens exceeds max_seq_len {}",
                max_tokens, state.max_seq_len
            ),
        );
    }

    let temperature = req.temperature.unwrap_or(state.default_temperature);
    let top_k = req.top_k.unwrap_or(state.default_top_k);
    let top_p = req.top_p.unwrap_or(state.default_top_p);
    let top_n_sigma = req.top_n_sigma.unwrap_or(state.default_top_n_sigma);
    let min_p = req.min_p.unwrap_or(state.default_min_p);
    let repetition_penalty = req
        .repetition_penalty
        .unwrap_or(defaults.sampling_presets.non_thinking.repetition_penalty);
    let presence_penalty = req.presence_penalty.unwrap_or(0.0);
    let frequency_penalty = req.frequency_penalty.unwrap_or(0.0);
    // Convert logit_bias from OpenAI format (string keys) to Vec<(u32, f32)>
    let logit_bias: Vec<(u32, f32)> = req.logit_bias.as_ref().map_or(Vec::new(), |map| {
        map.iter()
            .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
            .collect()
    });
    let stop_tokens = tokenize_stop_sequences(&state.tokenizer, &req.stop);
    let now = std::time::Instant::now();
    let timeout_secs = req.timeout.unwrap_or(if state.dsv41 {
        0.0
    } else {
        state.request_timeout as f32
    });
    let timeout_at = match checked_completion_deadline(timeout_secs, now) {
        Ok(deadline) => deadline,
        Err(message) => {
            return openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                message.to_owned(),
                Some("timeout"),
                Some("invalid_timeout"),
            );
        }
    };
    let decode_timeout_at = completion_decode_deadline(
        state.request_timeout,
        state.dsv41,
        req.timeout,
        max_tokens,
        prompt_len,
        now,
    );

    if req.stream {
        return match completions_stream(
            state,
            prompt_tokens,
            max_tokens,
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            logit_bias.clone(),
            stop_tokens,
            req.stop.clone(),
            req.seed,
            req.ignore_eos,
            timeout_at,
            decode_timeout_at,
            intervention_ledger,
            defaults.think_nudge.clone(),
        )
        .await
        {
            Ok(r) => r,
            Err((status, msg)) => openai_error_response(status, msg),
        };
    }

    // ── Blocking path ──
    let (tx, rx) = tokio::sync::oneshot::channel();
    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let request = InferenceRequest::Blocking {
        prompt_tokens,
        session_hash,
        image_pixels: Vec::new(),
        max_tokens,
        min_tokens: 0,
        ignore_eos: req.ignore_eos,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        // Legacy /v1/completions path doesn't have tool semantics, so
        // no DRY. (DRY on raw completion would dampen legitimate
        // long-repeated prose.)
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        lz_penalty: 0.0,
        logit_bias,
        stop_tokens,
        enable_thinking: false,
        thinking_budget: None,
        require_tool_call: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_spec: None,
        seed: req.seed,
        top_logprobs: None,
        timeout_at,
        decode_timeout_at,
        intervention_ledger: intervention_ledger.clone(),
        think_nudge: defaults.think_nudge.clone(),
        response_tx: tx,
    };

    if let Err((status, message)) = try_admit_scheduler(&state.request_tx, request) {
        return openai_error_response(status, message);
    }

    let response = match rx.await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            tracing::error!("completion inference error: {e:#}");
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

    let output_text = match state.tokenizer.decode(&response.output_tokens) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("completion detokenization error: {e:#}");
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal detokenization error".to_string(),
            );
        }
    };
    let output_text =
        completion_output_with_audit(output_text, &req.stop, intervention_ledger.as_ref());
    let completion_token_ids = if req.return_token_ids {
        response.output_tokens.clone()
    } else {
        Vec::new()
    };

    let num_completion = response.output_tokens.len();
    let tokens_per_second = if response.decode_time_ms > 0.0 {
        (num_completion.saturating_sub(1)) as f64 / (response.decode_time_ms / 1000.0)
    } else {
        0.0
    };
    let usage = Usage {
        prompt_tokens: prompt_len,
        completion_tokens: num_completion,
        total_tokens: prompt_len + num_completion,
        prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
            cached_tokens: response.cached_prompt_tokens as usize,
            audio_tokens: 0,
        }),
        completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
            reasoning_tokens: response.reasoning_tokens as usize,
            audio_tokens: 0,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
            think_nudges: response.think_nudges,
        }),
        time_to_first_token_ms: response.time_to_first_token_ms,
        response_tokens_per_second: tokens_per_second,
    };

    Json(
        CompletionResponse::new(
            &state.model_name,
            output_text,
            usage,
            &response.finish_reason,
            completion_token_ids,
        )
        .with_interventions(crate::openai::snapshot_shared_interventions(
            intervention_ledger.as_ref(),
        )),
    )
    .into_response()
}

/// SSE streaming path for legacy completions.
#[allow(clippy::too_many_arguments)]
pub(super) async fn completions_stream(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    top_n_sigma: f32,
    min_p: f32,
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    logit_bias: Vec<(u32, f32)>,
    stop_tokens: Vec<u32>,
    stop_strings: Vec<String>,
    seed: Option<u64>,
    ignore_eos: bool,
    timeout_at: Option<std::time::Instant>,
    decode_timeout_at: Option<std::time::Instant>,
    intervention_ledger: Option<crate::openai::SharedInterventionLedger>,
    think_nudge: Option<std::sync::Arc<crate::scheduler::think_nudge::ThinkNudgeConfig>>,
) -> Result<Response, (StatusCode, String)> {
    // Match chat_stream/mod.rs sizing; see comment there.
    let (token_tx, token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(1024);
    let prompt_len = prompt_tokens.len();

    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let request = InferenceRequest::Streaming {
        prompt_tokens,
        session_hash,
        image_pixels: Vec::new(),
        max_tokens,
        min_tokens: 0,
        ignore_eos,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        // Legacy /v1/completions path doesn't have tool semantics.
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        lz_penalty: 0.0,
        logit_bias,
        stop_tokens,
        enable_thinking: false,
        thinking_budget: None,
        require_tool_call: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_spec: None,
        seed,
        top_logprobs: None,
        timeout_at,
        decode_timeout_at,
        intervention_ledger: intervention_ledger.clone(),
        think_nudge,
        token_tx,
        // Legacy completions has no agentic guard pipeline; this flag
        // is reserved for the streaming string-stop matcher so a full
        // match terminates scheduler work as well as suppressing SSE.
        cancel_flag: cancel_flag.clone(),
    };

    try_admit_scheduler(&state.request_tx, request)?;

    let chunk_id = crate::openai::new_completion_id();
    let model_name = state.model_name.clone();

    let model = model_name.clone();
    let id = chunk_id.clone();
    let mut all_toks: Vec<u32> = Vec::new();
    let mut emitted: usize = 0;
    let mut detok_prefix_offset: usize = 0;
    let mut detok_read_offset: usize = 0;
    let mut stop_holdback = String::new();
    let mut stop_triggered = false;
    let cancel_guard = CompletionCancelGuard::new(cancel_flag);
    let token_stream = ReceiverStream::new(token_rx).flat_map(move |event| {
        let mut events = Vec::new();
        match event {
            StreamEvent::Token(tok) | StreamEvent::TokenWithLogprobs(tok, _) => {
                if stop_triggered {
                    return futures::stream::iter(events);
                }
                all_toks.push(tok);
                let delta = match state.tokenizer.incremental_decode(
                    &all_toks,
                    &mut detok_prefix_offset,
                    &mut detok_read_offset,
                ) {
                    Ok(text) => text,
                    Err(error) => {
                        tracing::warn!(
                            ?error,
                            "completion incremental tokenizer decode failed; deferring to Done"
                        );
                        String::new()
                    }
                };
                if delta.is_empty() {
                    return futures::stream::iter(events);
                }
                emitted += delta.len();
                let (safe, matched) = super::chat_stream::filter_stop_delta(
                    &mut stop_holdback,
                    &delta,
                    &stop_strings,
                );
                if matched {
                    crate::openai::record_shared_intervention(
                        intervention_ledger.as_ref(),
                        "completion_stop_sequence",
                        "stop_stream",
                    );
                    stop_triggered = true;
                    cancel_guard.cancel();
                }
                if !safe.is_empty() {
                    let chunk = CompletionChunk::text_chunk(&model, &id, safe);
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    events.push(Ok::<_, std::convert::Infallible>(
                        Event::default().data(json),
                    ));
                }
            }
            StreamEvent::CompletionProgress
            | StreamEvent::CompletionProgressWithLogprobs(_)
            | StreamEvent::ThinkingDone => {}
            StreamEvent::Done {
                finish_reason,
                completion_tokens,
                time_to_first_token_ms,
                decode_time_ms,
                reasoning_tokens,
                cached_prompt_tokens,
                think_nudges,
                ..
            } => {
                cancel_guard.disarm();
                if !stop_triggered {
                    if !all_toks.is_empty() {
                        match state.tokenizer.terminal_decode_residual(&all_toks, emitted) {
                            Ok(residual) if !residual.is_empty() => {
                                emitted += residual.len();
                                let (safe, matched) = super::chat_stream::filter_stop_delta(
                                    &mut stop_holdback,
                                    &residual,
                                    &stop_strings,
                                );
                                if matched {
                                    crate::openai::record_shared_intervention(
                                        intervention_ledger.as_ref(),
                                        "completion_stop_sequence",
                                        "stop_stream",
                                    );
                                    stop_triggered = true;
                                }
                                if !safe.is_empty() {
                                    let chunk = CompletionChunk::text_chunk(&model, &id, safe);
                                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                                    events.push(Ok(Event::default().data(json)));
                                }
                            }
                            Ok(_) => {}
                            Err(error) => {
                                tracing::warn!(
                                    ?error,
                                    "completion terminal tokenizer residual decode failed"
                                );
                            }
                        }
                    }
                    if !stop_triggered && !stop_holdback.is_empty() {
                        let chunk = CompletionChunk::text_chunk(
                            &model,
                            &id,
                            std::mem::take(&mut stop_holdback),
                        );
                        let json = serde_json::to_string(&chunk).unwrap_or_default();
                        events.push(Ok(Event::default().data(json)));
                    }
                }
                let tps = if decode_time_ms > 0.0 {
                    completion_tokens.saturating_sub(1) as f64 / (decode_time_ms / 1000.0)
                } else {
                    0.0
                };
                let usage = Usage {
                    prompt_tokens: prompt_len,
                    completion_tokens,
                    total_tokens: prompt_len + completion_tokens,
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
                let effective_finish_reason = if stop_triggered {
                    "stop"
                } else {
                    finish_reason.as_str()
                };
                let chunk =
                    CompletionChunk::done_chunk(&model, &id, effective_finish_reason, usage)
                        .with_interventions(crate::openai::snapshot_shared_interventions(
                            intervention_ledger.as_ref(),
                        ));
                let json = serde_json::to_string(&chunk).unwrap_or_default();
                events.push(Ok(Event::default().data(json)));
            }
            StreamEvent::Error(msg) => {
                events.push(Ok(completion_stream_error(&cancel_guard, msg)));
            }
        }
        futures::stream::iter(events)
    });

    let done_event = futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
    });
    let full_stream = token_stream.chain(done_event);

    Ok(Sse::new(full_stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

#[cfg(test)]
mod completion_cancel_tests {
    use super::{CompletionCancelGuard, completion_stream_error};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn dropped_completion_stream_release_cancels_scheduler() {
        let flag = Arc::new(AtomicBool::new(false));
        drop(CompletionCancelGuard::new(flag.clone()));
        assert!(flag.load(Ordering::Acquire));
    }

    #[test]
    fn completion_error_release_cancels_scheduler() {
        let flag = Arc::new(AtomicBool::new(false));
        let guard = CompletionCancelGuard::new(flag.clone());
        let _ = completion_stream_error(&guard, "boom".into());
        assert!(flag.load(Ordering::Acquire));
    }

    #[test]
    fn normal_completion_disarms_disconnect_guard() {
        let flag = Arc::new(AtomicBool::new(false));
        let guard = CompletionCancelGuard::new(flag.clone());
        guard.disarm();
        drop(guard);
        assert!(!flag.load(Ordering::Acquire));
    }
}

#[cfg(test)]
mod completion_prompt_intervention_tests {
    use super::resolve_prompt_tokens_with_audit;
    use crate::openai::{CompletionRequest, InterventionLedger, snapshot_shared_interventions};
    use crate::tokenizer::ChatTokenizer;
    use std::sync::{Arc, Mutex};

    fn tokenizer(thinking: bool) -> ChatTokenizer {
        let directory = tempfile::tempdir().unwrap();
        let vocabulary = [("[UNK]".to_string(), 0), ("hello".to_string(), 1)]
            .into_iter()
            .collect();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocabulary)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer
            .add_special_tokens([tokenizers::AddedToken::from(
                crate::dsv41::encoding::BOS,
                true,
            )])
            .unwrap();
        tokenizer
            .save(directory.path().join("tokenizer.json"), false)
            .unwrap();
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{"chat_template":"{{ messages }}"}"#,
        )
        .unwrap();
        ChatTokenizer::from_model_dir(directory.path(), 0, thinking, "ledger-fixture", None)
            .unwrap()
    }

    fn request(value: serde_json::Value) -> CompletionRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn exact_completion_prompts_never_record_or_receive_prompt_injection() {
        let tokenizer = tokenizer(true);
        for deepseek in [false, true] {
            for value in [
                serde_json::json!({"prompt_token_ids": [1, 0]}),
                serde_json::json!({"prompt": [1, 0]}),
            ] {
                let ledger = Arc::new(Mutex::new(InterventionLedger::default()));
                assert_eq!(
                    resolve_prompt_tokens_with_audit(
                        &request(value),
                        &tokenizer,
                        deepseek,
                        Some(&ledger),
                    )
                    .unwrap(),
                    vec![1, 0]
                );
                assert!(
                    snapshot_shared_interventions(Some(&ledger))
                        .unwrap()
                        .events
                        .is_empty()
                );
            }
        }
    }

    #[test]
    fn completion_thinking_prefix_is_recorded_only_when_inserted() {
        let tokenizer = tokenizer(true);
        for (text, inserted) in [("hello", true), ("</think>hello", false)] {
            let ledger = Arc::new(Mutex::new(InterventionLedger::default()));
            let ids = resolve_prompt_tokens_with_audit(
                &request(serde_json::json!({"prompt": text})),
                &tokenizer,
                false,
                Some(&ledger),
            )
            .unwrap();
            let expected = if inserted {
                format!("<think></think>\n\n{text}")
            } else {
                text.to_string()
            };
            assert_eq!(ids, tokenizer.encode(&expected).unwrap());
            let snapshot = snapshot_shared_interventions(Some(&ledger)).unwrap();
            assert_eq!(snapshot.events.len(), usize::from(inserted));
            if inserted {
                assert_eq!(snapshot.events[0].code, "completion_thinking_prefix");
            }
        }
    }

    #[test]
    fn completion_bos_is_recorded_only_when_inserted() {
        let tokenizer = tokenizer(false);
        let bos = tokenizer
            .inner()
            .token_to_id(crate::dsv41::encoding::BOS)
            .unwrap();
        for (text, inserted) in [
            ("hello".to_string(), true),
            (format!("{}hello", crate::dsv41::encoding::BOS), false),
        ] {
            let ledger = Arc::new(Mutex::new(InterventionLedger::default()));
            let ids = resolve_prompt_tokens_with_audit(
                &request(serde_json::json!({"prompt": text})),
                &tokenizer,
                true,
                Some(&ledger),
            )
            .unwrap();
            assert_eq!(ids.first(), Some(&bos));
            let snapshot = snapshot_shared_interventions(Some(&ledger)).unwrap();
            assert_eq!(snapshot.events.len(), usize::from(inserted));
            if inserted {
                assert_eq!(snapshot.events[0].code, "completion_bos_prefix");
            }
        }
    }
}

#[cfg(test)]
mod completion_output_intervention_tests {
    use super::completion_output_with_audit;
    use crate::openai::{InterventionLedger, snapshot_shared_interventions};
    use std::sync::{Arc, Mutex};

    #[test]
    fn unchanged_completion_keeps_an_explicit_empty_ledger() {
        let ledger = Arc::new(Mutex::new(InterventionLedger::default()));
        assert_eq!(
            completion_output_with_audit("answer".into(), &[], Some(&ledger)),
            "answer"
        );
        assert!(
            snapshot_shared_interventions(Some(&ledger))
                .unwrap()
                .events
                .is_empty()
        );
    }

    #[test]
    fn blocking_stop_strip_is_recorded_without_changing_its_behavior() {
        let ledger = Arc::new(Mutex::new(InterventionLedger::default()));
        let stops = vec!["<stop>".into()];
        let raw = "answer<stop>unpublished";
        assert_eq!(
            completion_output_with_audit(raw.into(), &stops, None),
            "answer"
        );
        assert_eq!(
            completion_output_with_audit(raw.into(), &stops, Some(&ledger)),
            "answer"
        );
        let snapshot = snapshot_shared_interventions(Some(&ledger)).unwrap();
        assert_eq!(snapshot.events.len(), 1);
        assert_eq!(snapshot.events[0].code, "completion_stop_sequence");
        assert_eq!(snapshot.events[0].action, "strip_output");
    }

    #[test]
    fn hidden_thinking_cleanup_is_recorded_and_scheduler_events_survive() {
        let ledger = Arc::new(Mutex::new(InterventionLedger::default()));
        ledger
            .lock()
            .unwrap()
            .record("content_loop_watchdog", "stop");
        let raw = "<think>private reasoning</think>answer";
        let expected = super::strip_thinking_tags(raw);
        assert_ne!(expected, raw);
        assert_eq!(
            completion_output_with_audit(raw.into(), &[], Some(&ledger)),
            expected
        );
        let snapshot = snapshot_shared_interventions(Some(&ledger)).unwrap();
        assert_eq!(snapshot.events.len(), 2);
        assert_eq!(snapshot.events[0].code, "content_loop_watchdog");
        assert_eq!(snapshot.events[1].code, "completion_output_cleanup");
    }
}

fn model_info(state: &AppState) -> ModelInfo {
    ModelInfo {
        id: state.model_name.clone(),
        object: "model".to_string(),
        created: crate::openai::unix_timestamp(),
        owned_by: "atlas-spark".to_string(),
        max_model_len: state.max_seq_len,
    }
}

/// GET /v1/models
pub async fn list_models(CurrentModel(state): CurrentModel) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![model_info(&state)],
    })
}

/// GET /v1/models/{model_id} — retrieve a single model (OpenAI SDK `client.models.retrieve()`).
pub async fn get_model(
    CurrentModel(state): CurrentModel,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Response {
    if model_id == state.model_name {
        Json(model_info(&state)).into_response()
    } else {
        openai_error_response(
            StatusCode::NOT_FOUND,
            format!("The model '{model_id}' does not exist"),
        )
    }
}

/// POST /v1/embeddings — stub for clients that probe this endpoint during auto-detection.
pub async fn embeddings_stub() -> Response {
    openai_error_response(
        StatusCode::NOT_IMPLEMENTED,
        "Embeddings are not supported by this model. Flux Engine serves generative (chat/completion) models only.".into(),
    )
}

/// Generic 501 "not supported" response used by the auto-probe stubs
/// below. OpenAI-SDK auto-detection and observability wrappers expect a
/// 501 + `error.type = server_error`; returning 404 would be interpreted
/// as "wrong URL".
pub(super) fn not_supported(message: &'static str) -> Response {
    openai_error_response(StatusCode::NOT_IMPLEMENTED, message.into())
}

#[cfg(test)]
mod completion_input_validation_tests {
    use super::validate_completion_input;
    use axum::http::StatusCode;

    fn request() -> crate::openai::CompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test",
            "prompt": "hello"
        }))
        .expect("valid completion request")
    }

    #[test]
    fn completion_token_ids_are_blocking_only_and_opt_in() {
        let mut value = request();
        assert!(!value.return_token_ids);
        assert!(validate_completion_input(&value).is_ok());

        value.return_token_ids = true;
        assert!(validate_completion_input(&value).is_ok());

        value.stream = true;
        assert_eq!(
            validate_completion_input(&value).unwrap_err().status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn completion_numeric_contract_matches_chat() {
        let mut value = request();
        assert!(validate_completion_input(&value).is_ok());

        value.requested_max_tokens = Some(0);
        assert_eq!(
            validate_completion_input(&value).unwrap_err().status(),
            StatusCode::BAD_REQUEST
        );

        value = request();
        for invalid in [-1.0, 2.1, f32::NAN, f32::INFINITY] {
            value.temperature = Some(invalid);
            assert_eq!(
                validate_completion_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        value = request();
        for invalid in [0.0, -0.1, 1.1, f32::NAN, f32::INFINITY] {
            value.top_p = Some(invalid);
            assert_eq!(
                validate_completion_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        value.top_p = None;
        for invalid in [-2.1, 2.1, f32::NAN, f32::INFINITY] {
            value.presence_penalty = Some(invalid);
            assert_eq!(
                validate_completion_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
            value.presence_penalty = None;
            value.frequency_penalty = Some(invalid);
            assert_eq!(
                validate_completion_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
            value.frequency_penalty = None;
        }
    }

    #[test]
    fn completion_logit_bias_rejects_invalid_keys_and_values() {
        let mut value = request();
        value.logit_bias = Some(std::collections::HashMap::from([
            ("0".to_string(), -100.0),
            (u32::MAX.to_string(), 100.0),
        ]));
        assert!(validate_completion_input(&value).is_ok());

        for key in ["not-token", "-1", "4294967296"] {
            value.logit_bias = Some(std::collections::HashMap::from([(key.to_string(), 0.0)]));
            assert_eq!(
                validate_completion_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        for bias in [-100.1, 100.1, f32::NAN, f32::INFINITY] {
            value.logit_bias = Some(std::collections::HashMap::from([("0".to_string(), bias)]));
            assert_eq!(
                validate_completion_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }
    }

    #[test]
    fn completion_extended_sampling_contract_is_fail_closed() {
        let mut value = request();
        for invalid in [-0.1, f32::NAN, f32::INFINITY] {
            value.top_n_sigma = Some(invalid);
            assert_eq!(
                validate_completion_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        value.top_n_sigma = None;
        for invalid in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
            value.min_p = Some(invalid);
            assert_eq!(
                validate_completion_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        value.min_p = None;
        for invalid in [0.0, -0.1, f32::NAN, f32::INFINITY] {
            value.repetition_penalty = Some(invalid);
            assert_eq!(
                validate_completion_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        value.repetition_penalty = Some(f32::MIN_POSITIVE);
        assert!(validate_completion_input(&value).is_ok());
    }

    #[test]
    fn completion_stop_contract_rejects_empty_or_excess_entries() {
        let mut value = request();
        value.stop = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        assert!(validate_completion_input(&value).is_ok());

        value.stop.push("e".into());
        assert_eq!(
            validate_completion_input(&value).unwrap_err().status(),
            StatusCode::BAD_REQUEST
        );

        value.stop = vec![String::new()];
        assert_eq!(
            validate_completion_input(&value).unwrap_err().status(),
            StatusCode::BAD_REQUEST
        );
    }
}
